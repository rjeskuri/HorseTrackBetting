"""The C4 meta-model at the final price (#87, epic #73; 08 §3.2 C4, §4 E9, E40, R13).

The frame holds race_id, horse_id, won, ln_q_final, one out-of-fold log-probability column per
member and the segment columns. The utility of runner i is

    stage2:  u_i = sum_k alpha_k(z_i) * m_ki + beta * ln q_i      (alpha and beta unbounded)
    offset:  u_i = ln q_i + sum_k alpha_k(z_i) * m_ki             (q's coefficient fixed at 1)

with alpha_k(z) = alpha_k0 + sum_j alpha_kj * z_j over `segment_cols` (an interaction column
m_k * z_j per pair: feature-weighted stacking, so a selection study can read alpha_s), and
P(i wins) = softmax of u within the race, fitted by `condlogit.fit_conditional_logit`. Terms are
named by member (alpha_k0), "<member>:<segment col>" (alpha_kj) and "ln_q_final" (beta, stage2
only); `quadratic` adds "ln_q_final^2", a free coefficient on (ln q)^2 in either form (the
favourite-longshot bend of 08's C4 gamma). q is normalised within each race before its log is
taken, at fit and at predict, so only ratios of q within a race matter.

`ridge` shrinks the member terms (alpha_k0, alpha_kj) toward 0. The owner's shrinkage target
(2026-11-24, #87) is alpha = 0, the market alone, so the market terms are never shrunk: under a
huge ridge the stage-2 model is a power of q (with its bend, if quadratic) and the offset model q.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from horsetrack.core.eval.folds import Fold
from horsetrack.countries.japan.models.condlogit import fit_conditional_logit

Q = "ln_q_final"
Q2 = "ln_q_final^2"
FORMS = ("stage2", "offset")


@dataclass(frozen=True)
class MetaSpec:
    members: list[str]
    quadratic: bool
    segment_cols: list[str]
    form: str
    ridge: float

    def __post_init__(self) -> None:
        if self.form not in FORMS:
            raise ValueError(f"form must be one of {FORMS}, not {self.form!r}")
        names = [*self.members, *self.segment_cols]
        if not all(isinstance(name, str) and name for name in names):
            raise ValueError(f"members and segment_cols must be column names, got {names!r}")
        if len(set(names)) != len(names) or {Q, Q2} & set(names):
            raise ValueError(f"members and segment_cols must be distinct names other than {Q}")
        if not self.members:
            raise ValueError("a meta-model needs at least one member")
        if not (np.isfinite(self.ridge) and self.ridge >= 0):
            raise ValueError(f"ridge must be finite and non-negative, got {self.ridge!r}")

    def terms(self) -> list[tuple[str, str | None, str | None]]:
        """(name, member, segment) per term in design order; a market term has no member."""
        terms: list[tuple[str, str | None, str | None]] = [(m, m, None) for m in self.members]
        terms += [(f"{m}:{z}", m, z) for m in self.members for z in self.segment_cols]
        terms += [(Q, None, None)] if self.form == "stage2" else []
        return terms + ([(Q2, None, None)] if self.quadratic else [])


def _race_codes(frame: pd.DataFrame) -> np.ndarray:
    codes = pd.factorize(frame["race_id"].to_numpy(dtype=object))[0]
    if (codes < 0).any():
        raise ValueError("race_id has missing values; every row needs its race")
    return codes


def _log_normalised(ln_q: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """ln q minus the race's log-sum-exp of ln q: the log of q normalised within each race."""
    if not np.all(np.isfinite(ln_q)):
        raise ValueError("q must be positive and finite for every row")
    top = pd.Series(ln_q).groupby(codes).transform("max").to_numpy()
    return ln_q - top - np.log(np.bincount(codes, weights=np.exp(ln_q - top))[codes])


def _ln_q(q: Any, codes: np.ndarray) -> np.ndarray:
    """ln of q normalised within each race; q must be positive and finite, one per row."""
    q = np.asarray(q, dtype=float)
    if q.shape != codes.shape:
        raise ValueError(f"q has shape {q.shape}, the frame {codes.shape[0]} rows")
    if not np.all(q > 0):
        raise ValueError("q must be positive and finite for every row")
    with np.errstate(divide="ignore", invalid="ignore"):
        return _log_normalised(np.log(q), codes)


def _design(frame: pd.DataFrame, spec: MetaSpec, ln_q: np.ndarray) -> tuple[np.ndarray, list[str]]:
    missing = [c for c in (*spec.members, *spec.segment_cols) if c not in frame.columns]
    if missing:
        raise KeyError(f"the frame has no column(s) {missing}")
    columns, names = [], []
    for name, member, segment in spec.terms():
        if member is None:
            columns.append(ln_q if name == Q else ln_q * ln_q)
        else:
            column = frame[member].to_numpy(dtype=float)
            columns.append(column if segment is None else column * frame[segment].to_numpy(float))
        names.append(name)
    return np.column_stack(columns), names


def _softmax(u: np.ndarray, codes: np.ndarray) -> np.ndarray:
    top = pd.Series(u).groupby(codes).transform("max").to_numpy()
    e = np.exp(u - top)
    return e / np.bincount(codes, weights=e)[codes]


@dataclass(frozen=True, eq=False)
class MetaFit:
    """A fitted meta-model: coef and se are Series on the term names; predict reads coef."""

    spec: MetaSpec
    coef: pd.Series
    se: pd.Series
    loglik: float
    n_races: int

    def predict(self, frame: pd.DataFrame, q: np.ndarray) -> np.ndarray:
        """P(win | x, q) per frame row, in row order, for any price vector q (one positive value
        per row; only ratios within a race matter). Reads race_id, the members and the segment
        columns; a ln_q_final column in the frame is ignored. Each race sums to 1."""
        codes = _race_codes(frame)
        ln_q = _ln_q(q, codes)
        X, names = _design(frame, self.spec, ln_q)
        u = X @ self.coef.reindex(names).to_numpy(dtype=float)
        if self.spec.form == "offset":
            u = u + ln_q
        if not np.all(np.isfinite(u)):
            raise ValueError("a member or segment value is missing or not finite")
        return _softmax(u, codes)


def fit_meta(frame: pd.DataFrame, spec: MetaSpec) -> MetaFit:
    """The meta-model's conditional-logit maximum likelihood on `frame` (see the module doc)."""
    if not isinstance(spec, MetaSpec):
        raise TypeError(f"spec must be a MetaSpec, got {type(spec).__name__}")
    codes = _race_codes(frame)
    ln_q = _log_normalised(frame[Q].to_numpy(dtype=float), codes)
    X, names = _design(frame, spec, ln_q)
    penalty = np.array([0.0 if member is None else spec.ridge for _, member, _ in spec.terms()])
    fit = fit_conditional_logit(
        X,
        codes,
        frame["won"].to_numpy(),
        offset=ln_q if spec.form == "offset" else None,
        ridge=penalty,
    )
    return MetaFit(
        spec=spec,
        coef=pd.Series(fit.coef, index=names, dtype=float),
        se=pd.Series(fit.se, index=names, dtype=float),
        loglik=fit.loglik,
        n_races=fit.n_races,
    )


def oof_predictions(
    frame: pd.DataFrame,
    folds: Sequence[Fold],
    fit: Callable[[pd.DataFrame], Any],
    predict: Callable[[Any, pd.DataFrame], Any],
) -> pd.Series:
    """Out-of-fold values: for each fold, `fit(frame.iloc[train_idx])` then `predict(model,
    frame.iloc[test_idx])`, one value per test row. The Series is on frame's index (unique labels)
    and NaN on rows in no test window. A fold that trains on a row it tests, two folds that test
    one row, or a position outside the frame is a ValueError."""
    if not frame.index.is_unique:
        raise ValueError("the frame's index has duplicate labels")
    n = len(frame)
    tested = np.zeros(n, dtype=bool)
    for k, fold in enumerate(folds):
        train, test = np.asarray(fold.train_idx), np.asarray(fold.test_idx)
        for name, idx in (("train", train), ("test", test)):
            if idx.size and (idx.min() < 0 or idx.max() >= n):
                raise ValueError(f"fold {k}'s {name} positions fall outside the frame's {n} rows")
        if np.isin(test, train).any():
            raise ValueError(f"fold {k} trains on a row it tests")
        if len(np.unique(test)) != len(test) or tested[test].any():
            raise ValueError(f"fold {k} tests a row that is already tested")
        tested[test] = True

    out = np.full(n, np.nan)
    for k, fold in enumerate(folds):
        test = np.asarray(fold.test_idx)
        model = fit(frame.iloc[np.asarray(fold.train_idx)])
        values = np.asarray(predict(model, frame.iloc[test]), dtype=float)
        if values.shape != test.shape:
            raise ValueError(f"fold {k}: predict gave {values.shape}, the test rows {test.shape}")
        out[test] = values
    return pd.Series(out, index=frame.index, dtype=float)


def encompassing(frame: pd.DataFrame, member: str) -> tuple[float, float]:
    """The member's coefficient and standard error in an unpenalised stage-2 fit on that member
    and ln_q_final alone (Fair-Shiller encompassing)."""
    fit = fit_meta(
        frame, MetaSpec([member], quadratic=False, segment_cols=[], form="stage2", ridge=0.0)
    )
    return float(fit.coef[member]), float(fit.se[member])


def ceiling_table(
    frame: pd.DataFrame, *, pi_col: str, q_col: str, thresholds: Sequence[float]
) -> pd.DataFrame:
    """The information-ceiling table (E9) of per-race probabilities pi against q: per threshold,
    in the order given, `q_mass`, the share of the total q mass on runners with ln(pi/q) above it
    (overlays, one-sided), and `rms`, the q-weighted root mean square of ln(pi/q) over every
    runner (the same on every row)."""
    pi = frame[pi_col].to_numpy(dtype=float)
    q = frame[q_col].to_numpy(dtype=float)
    if not (np.all(np.isfinite(pi) & np.isfinite(q)) and np.all(pi > 0) and np.all(q > 0)):
        raise ValueError(f"{pi_col} and {q_col} must be positive and finite on every row")
    x = np.log(pi / q)
    total = q.sum()
    rms = float(np.sqrt((q * x * x).sum() / total))
    rows = [
        {"threshold": float(t), "q_mass": float(q[x > t].sum() / total), "rms": rms}
        for t in thresholds
    ]
    return pd.DataFrame(rows, columns=["threshold", "q_mass", "rms"])
