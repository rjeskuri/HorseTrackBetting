"""The meld of the out-of-fold members in the C4 meta-model (#98, epic #73; 08 §3.2 C4 and
cross-cutting, §4 E41, E42).

08's C4 is one concave conditional logit at the final price,

    u_i = sum_k alpha_k(z_r) * w_ik * m_ik + beta * ln q_i (+ gamma * (ln q_i)^2),

with hard gates on observable z (a separate alpha_k per gate cell), feature-weighted segment slopes
and an optional precision weight w_ik = 1 / (1 + kappa_k * sigma_ik^2), so a runner's member term
carries alpha_k0 / (1 + kappa_k sigma_ik^2) (E41). The shrinkage target is alpha = 0, the market
alone (the owner, 2026-11-24): the ridge falls on every member term and never on a market term.

- `member_frame(members, base)` aligns each member's out-of-fold `logp` onto the meld's rows by
  (race_id, horse_id). `base` carries `fold`, the meld's own fold of the row (the test quarter's
  start, as `core.eval.folds` labels it); a member row is usable only when its `oof_fold` is that
  same label (#418's guard: a null, blank or other label raises `stacking.InvalidProvenance`
  naming the member). A base race in which some runner lacks some member is dropped whole.
- `precision_weight(sigma, *, alpha0, kappa)`: alpha0 / (1 + kappa sigma^2), elementwise.
- `fit_meld(frame, *, members, gates, segment_cols, precision, ...)` is `models.meta.fit_meta` on
  the expanded design (one column per member and gate cell, precision-weighted), returned as a
  `MeldFit`, a `MetaFit` whose `predict(frame, q)` reads the raw columns and takes any price
  vector (#87's DoD 4). A gate cell is the tuple of a row's gate values, written "gate=value"
  joined by "|"; a cell the fit never saw predicts with its member terms at 0 (alpha = 0, the
  market alone).
- `marginal_ablation(frame, members, *, clusters, n_boot, level, seed, ...)`: the marginal table,
  walk-forward over the frame's sorted `fold` labels (each fold after the first scored by fits on
  every earlier fold), every variant's ΔLL against the market-only meta fitted the same way and
  each member's marginal gain, all bounded by one meeting-cluster bootstrap whose draws depend on
  the seed and the clusters alone (`core.eval.bootstrap.paired_bootstrap` on the same races).
- `ambiguity(frame, members)`: Heskes' ambiguity term, the mean over races of the mean over
  members of KL(p_bar || p_k), p_bar the normalised geometric mean (equal weights);
  `ambiguity_terms` gives the per-race terms behind it.
- `member_correlations(frame, members)`: Pearson correlations of the logp demeaned within race.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.scoring import per_race_log_loss
from horsetrack.countries.japan.models.condlogit import fit_conditional_logit
from horsetrack.countries.japan.models.meta import (
    FORMS,
    MetaFit,
    MetaSpec,
    Q,
    _log_normalised,
    _race_codes,
    _softmax,
    fit_meta,
)
from horsetrack.countries.japan.models.stacking import InvalidProvenance

KEY = ["race_id", "horse_id"]
MEMBER_COLUMNS = (*KEY, "logp", "oof_fold")
FOLD = "fold"
ALL = "all"
MARKET = "market"  # the market-only meta's column of the walk-forward predictions
TABLE_COLUMNS = [
    "variant",
    "races",
    "delta_ll",
    "delta_ll_lo",
    "delta_ll_hi",
    "gain",
    "gain_lo",
    "gain_hi",
]


# --------------------------------------------------------------------------- member_frame


def _labels(values: pd.Series) -> np.ndarray:
    """Fold labels as stripped strings, None where missing."""
    raw = values.to_numpy(dtype=object)
    missing = pd.isna(raw)
    out = np.empty(len(raw), dtype=object)
    out[missing] = None
    out[~missing] = [str(v).strip() for v in raw[~missing]]
    return out


def _key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([frame[c].to_numpy(dtype=object) for c in KEY])


def member_frame(members: Mapping[str, pd.DataFrame], base: pd.DataFrame) -> pd.DataFrame:
    """The base rows with one logp column per member (see the module doc)."""
    if not members:
        raise ValueError("a meld needs at least one member")
    missing = [c for c in (*KEY, FOLD) if c not in base.columns]
    if missing:
        raise KeyError(f"the base has no column(s) {missing}")
    clash = sorted(set(members) & set(base.columns))
    if clash:
        raise ValueError(f"the base already has column(s) named for members: {clash}")
    if base.duplicated(KEY).any():
        raise ValueError("the base repeats a (race_id, horse_id)")
    fold = _labels(base[FOLD])
    if any(f is None or f == "" for f in fold):
        raise ValueError("every base row needs its fold")

    keys = _key_index(base)
    have = np.ones(len(base), dtype=bool)
    columns: dict[str, np.ndarray] = {}
    for name, member in members.items():
        absent = [c for c in MEMBER_COLUMNS if c not in member.columns]
        if absent:
            raise KeyError(f"member {name!r} has no column(s) {absent}")
        at = keys.get_indexer(_key_index(member))
        aligned = at >= 0
        rows = at[aligned]
        if len(np.unique(rows)) != len(rows):
            raise ValueError(f"member {name!r} repeats a (race_id, horse_id) of the base")
        own = _labels(member["oof_fold"].loc[aligned])
        bad = np.flatnonzero(own != fold[rows])
        if bad.size:
            first = base.iloc[rows[bad[0]]]
            raise InvalidProvenance(
                f"member {name!r}: {bad.size} aligned rows are not out of fold for the meld's "
                f"fold, e.g. {first['race_id']} {first['horse_id']} (oof_fold "
                f"{own[bad[0]]!r}, the meld's fold {fold[rows[bad[0]]]!r})"
            )
        column = np.full(len(base), np.nan)
        column[rows] = member["logp"].to_numpy(dtype=float)[aligned]
        have &= np.isfinite(column)
        columns[name] = column

    codes = _race_codes(base)
    keep = np.bincount(codes, weights=(~have).astype(float))[codes] == 0
    out = base.iloc[np.flatnonzero(keep)].copy()
    for name, column in columns.items():
        out[name] = column[keep]
    return out


# --------------------------------------------------------------------------- fit_meld


def precision_weight(sigma, *, alpha0: float, kappa: float) -> np.ndarray:
    """alpha0 / (1 + kappa * sigma^2), elementwise in the shape of sigma (E41)."""
    if not (np.isfinite(kappa) and kappa >= 0):
        raise ValueError(f"kappa must be finite and non-negative, got {kappa!r}")
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(over="ignore"):
        return alpha0 / (1.0 + kappa * sigma * sigma)


def _cells(frame: pd.DataFrame, gates: Sequence[str]) -> np.ndarray:
    """Each row's gate cell, "gate=value" per gate joined by "|"; ALL without gates."""
    if not gates:
        return np.full(len(frame), ALL, dtype=object)
    parts = [[f"{g}={v}" for v in frame[g].to_numpy(dtype=object)] for g in gates]
    return np.asarray(["|".join(p) for p in zip(*parts, strict=True)], dtype=object)


@dataclass(frozen=True, eq=False)
class MeldFit(MetaFit):
    """A fitted meld: the `MetaFit` of the expanded design and how to expand a raw frame.
    `cells` are the gate cells the fit saw, in term order; `terms` maps each expanded member term
    to its (member, cell)."""

    members: tuple[str, ...] = ()
    gates: tuple[str, ...] = ()
    cells: tuple[str, ...] = ()
    precision: Mapping[str, str] = field(default_factory=dict)
    kappa: Mapping[str, float] = field(default_factory=dict)

    @property
    def terms(self) -> dict[str, tuple[str, str]]:
        return {_term(k, c): (k, c) for k in self.members for c in self.cells}

    def expand(self, frame: pd.DataFrame) -> pd.DataFrame:
        """race_id, the segment columns and one weighted column per (member, cell)."""
        return _expand(
            frame,
            self.members,
            self.gates,
            self.cells,
            self.spec.segment_cols,
            self.precision,
            self.kappa,
        )

    def predict(self, frame: pd.DataFrame, q: np.ndarray) -> np.ndarray:
        """P(win | members, q) per row from the raw columns, for any positive q (see MetaFit)."""
        return super().predict(self.expand(frame), q)


def _term(member: str, cell: str) -> str:
    return f"{member}@{cell}"


def _expand(frame, members, gates, cells, segment_cols, precision, kappa) -> pd.DataFrame:
    needed = [*members, *gates, *segment_cols, *(precision[k] for k in members if k in precision)]
    missing = [c for c in dict.fromkeys(needed) if c not in frame.columns]
    if missing:
        raise KeyError(f"the frame has no column(s) {missing}")
    label = _cells(frame, gates)
    out = {"race_id": frame["race_id"].to_numpy(dtype=object)}
    for z in segment_cols:
        out[z] = frame[z].to_numpy(dtype=float)
    for k in members:
        m = frame[k].to_numpy(dtype=float)
        if k in precision:
            m = m * precision_weight(
                frame[precision[k]].to_numpy(dtype=float), alpha0=1.0, kappa=kappa[k]
            )
        if not np.all(np.isfinite(m)):
            raise ValueError(f"member {k!r} (or its sigma) is missing or not finite on some row")
        for c in cells:
            out[_term(k, c)] = np.where(label == c, m, 0.0)
    return pd.DataFrame(out, index=frame.index)


def fit_meld(
    frame: pd.DataFrame,
    *,
    members: Sequence[str],
    gates: Sequence[str],
    segment_cols: Sequence[str],
    precision: Mapping[str, str],
    kappa: Mapping[str, float] | None = None,
    form: str = "stage2",
    ridge: float = 0.0,
    quadratic: bool = False,
    cluster_col: str | None = None,
) -> MeldFit:
    """`fit_meta` on the gated, precision-weighted members (see the module doc). `kappa` has one
    entry per member in `precision`; `cluster_col` fills `se_cluster` as in `fit_meta`."""
    members, gates = tuple(members), tuple(gates)
    kappa = dict(kappa or {})
    if not members or len(set(members)) != len(members):
        raise ValueError(f"members must be distinct and at least one, got {members!r}")
    if set(precision) - set(members) or set(kappa) != set(precision):
        raise ValueError(
            f"precision {sorted(precision)} must name members and kappa {sorted(kappa)} the same"
        )
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}, not {form!r}")
    cells = tuple(sorted(set(_cells(frame, gates))))
    design = _expand(frame, members, gates, cells, segment_cols, precision, kappa)
    design["won"] = frame["won"].to_numpy(dtype=bool)
    design[Q] = frame[Q].to_numpy(dtype=float)
    if cluster_col is not None:
        design[cluster_col] = frame[cluster_col].to_numpy(dtype=object)
    spec = MetaSpec(
        members=[_term(k, c) for k in members for c in cells],
        quadratic=bool(quadratic),
        segment_cols=list(segment_cols),
        form=form,
        ridge=float(ridge),
    )
    fit = fit_meta(design, spec, cluster_col=cluster_col)
    return MeldFit(
        spec=fit.spec,
        coef=fit.coef,
        se=fit.se,
        loglik=fit.loglik,
        n_races=fit.n_races,
        se_cluster=fit.se_cluster,
        members=members,
        gates=gates,
        cells=cells,
        precision=dict(precision),
        kappa=kappa,
    )


# --------------------------------------------------------------------------- marginal_ablation


@dataclass(frozen=True)
class _MarketFit:
    """The market-only meta: stage2 softmax(beta ln q (+ gamma ln q^2)), offset q (bent)."""

    form: str
    quadratic: bool
    coef: np.ndarray

    def predict(self, frame: pd.DataFrame, q: np.ndarray) -> np.ndarray:
        codes = _race_codes(frame)
        ln_q = _log_normalised(np.log(np.asarray(q, dtype=float)), codes)
        X = _market_design(ln_q, self.form, self.quadratic)
        u = X @ self.coef + (ln_q if self.form == "offset" else 0.0)
        return _softmax(u, codes)


def _market_design(ln_q: np.ndarray, form: str, quadratic: bool) -> np.ndarray:
    columns = ([ln_q] if form == "stage2" else []) + ([ln_q * ln_q] if quadratic else [])
    return np.column_stack(columns) if columns else np.zeros((len(ln_q), 0))


def fit_market(frame: pd.DataFrame, *, form: str = "stage2", quadratic: bool = False) -> _MarketFit:
    """The market-only meta in `form` (alpha = 0: no member), its market terms unpenalised."""
    codes = _race_codes(frame)
    ln_q = _log_normalised(frame[Q].to_numpy(dtype=float), codes)
    fit = fit_conditional_logit(
        _market_design(ln_q, form, quadratic),
        frame["race_id"].to_numpy(dtype=object),
        frame["won"].to_numpy(),
        offset=ln_q if form == "offset" else None,
    )
    return _MarketFit(form=form, quadratic=quadratic, coef=fit.coef)


def winner_log_loss(frame: pd.DataFrame, p: np.ndarray) -> pd.Series:
    """D6's winner log loss per race (a dead heat at 1/k per co-winner, a void race dropped)."""
    p = pd.Series(np.asarray(p, dtype=float), index=frame.index)
    no = pd.Series(False, index=frame.index)
    race = pd.Series(frame["race_id"].to_numpy(dtype=object), index=frame.index)
    return per_race_log_loss(p, frame["won"].astype(bool), race, no).per_race


def meld_options(members: Sequence[str], **meld) -> dict:
    """`fit_meld`'s keywords for `members` from shared ones: the precision and kappa entries of
    the members kept, the other keywords as given (no gates and no segments by default)."""
    precision, kappa = dict(meld.pop("precision", None) or {}), dict(meld.pop("kappa", None) or {})
    return (
        {"gates": (), "segment_cols": ()}
        | meld
        | {
            "members": list(members),
            "precision": {k: s for k, s in precision.items() if k in members},
            "kappa": {k: v for k, v in kappa.items() if k in members},
        }
    )


def walk_forward_predictions(
    frame: pd.DataFrame,
    variants: Mapping[str, Mapping],
    *,
    market: Mapping | None = None,
    embargo_days: int = 0,
) -> pd.DataFrame:
    """Out-of-fold P(win) on the scored rows, walk-forward over the frame's sorted `fold` labels:
    each fold after the first predicted by fits on every earlier fold's rows, the first never
    scored. Column MARKET is the market-only meta (`fit_market` with `market`'s form and
    quadratic), then one column per variant, `variants` mapping its name to `fit_meld`'s keywords
    (one with no members is the market-only meta). With `embargo_days`, a fold trains only on
    rows whose `race_date` is that many days or more before its first race day (D9's embargo).
    Indexed by the scored rows' labels, in frame order."""
    if FOLD not in frame.columns:
        raise KeyError(f"the frame has no {FOLD!r} column")
    if MARKET in variants:
        raise ValueError(f"{MARKET!r} is the market-only meta's column, not a variant name")
    fold = _labels(frame[FOLD])
    if any(f is None for f in fold):
        raise ValueError("every row needs its fold")
    labels = sorted(set(fold))
    if len(labels) < 2:
        raise ValueError(f"walk-forward needs two folds or more, got {labels}")
    day = pd.to_datetime(frame["race_date"]).to_numpy() if embargo_days else None
    names = [MARKET, *variants]
    out = np.full((len(frame), len(names)), np.nan)
    for label in labels[1:]:
        test_at = np.flatnonzero(fold == label)
        train = fold < label
        if day is not None:
            train &= day <= day[test_at].min() - np.timedelta64(embargo_days, "D")
        train_rows, test = frame.iloc[np.flatnonzero(train)], frame.iloc[test_at]
        q = np.exp(test[Q].to_numpy(dtype=float))
        out[test_at, 0] = fit_market(train_rows, **dict(market or {})).predict(test, q)
        for j, name in enumerate(variants, start=1):
            options = dict(variants[name])
            if options.get("members"):
                out[test_at, j] = fit_meld(train_rows, **options).predict(test, q)
            else:
                out[test_at, j] = out[test_at, 0]
    scored = np.flatnonzero(fold != labels[0])
    return pd.DataFrame(out[scored], index=frame.index[scored], columns=names)


def walk_forward_loss(frame: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Per scored race (indexed by race_id), the winner log loss of each column of `p`, the
    out-of-fold probabilities on `frame`'s rows `p.index` (`walk_forward_predictions`)."""
    rows = frame.loc[p.index]
    return pd.DataFrame({name: winner_log_loss(rows, p[name]) for name in p.columns})


def ablation_table(
    ll: pd.DataFrame,
    members: Sequence[str],
    *,
    clusters: pd.Series,
    n_boot: int,
    level: float,
    seed: int,
) -> pd.DataFrame:
    """The marginal table from per-race losses with columns MARKET, ALL and f"all minus {k}":
    every variant's ΔLL against the market and each member's gain, all on one set of meeting
    draws (`paired_bootstrap` on the same races with the same seed)."""

    def bounds(diff: pd.Series) -> tuple[float, float, float]:
        iv = paired_bootstrap(diff, clusters, n_boot=n_boot, level=level, seed=seed)
        return iv.mean, iv.lo, iv.hi

    rows = []
    for name in [ALL, *(f"all minus {k}" for k in members)]:
        delta = bounds(ll[MARKET] - ll[name])
        gain = (np.nan,) * 3 if name == ALL else bounds(ll[name] - ll[ALL])
        rows.append([name, len(ll), *delta, *gain])
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def marginal_ablation(
    frame: pd.DataFrame,
    members: Sequence[str],
    *,
    clusters: pd.Series,
    n_boot: int,
    level: float,
    seed: int,
    embargo_days: int = 0,
    **meld,
) -> pd.DataFrame:
    """The marginal table (see the module doc): "all", then "all minus k" per member in order.
    `meld` passes gates, segment_cols, precision, kappa, form, ridge and quadratic to every fit
    (the market-only meta takes its form and quadratic); `embargo_days` is
    `walk_forward_predictions`'."""
    members = list(members)
    if not members or len(set(members)) != len(members):
        raise ValueError(f"members must be distinct and at least one, got {members!r}")
    variants = {ALL: meld_options(members, **meld)} | {
        f"all minus {k}": meld_options([m for m in members if m != k], **meld) for k in members
    }
    market = {key: meld[key] for key in ("form", "quadratic") if key in meld}
    p = walk_forward_predictions(frame, variants, market=market, embargo_days=embargo_days)
    ll = walk_forward_loss(frame, p)
    return ablation_table(ll, members, clusters=clusters, n_boot=n_boot, level=level, seed=seed)


# --------------------------------------------------------------------------- diversity


def _log_softmax(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    top = pd.Series(values).groupby(codes).transform("max").to_numpy()
    shifted = values - top
    return shifted - np.log(np.bincount(codes, weights=np.exp(shifted))[codes])


def ambiguity_terms(frame: pd.DataFrame, members: Sequence[str]) -> pd.Series:
    """Per race (indexed by race_id, first-appearance order), the mean over members of
    KL(p_bar || p_k), p_bar the normalised geometric mean of the members within the race."""
    members = list(members)
    if len(members) < 2:
        raise ValueError(f"ambiguity needs two members or more, got {members!r}")
    codes = _race_codes(frame)
    logs = np.column_stack([_log_softmax(frame[k].to_numpy(dtype=float), codes) for k in members])
    if not np.all(np.isfinite(logs)):
        raise ValueError("a member is missing or not finite on some row")
    log_bar = _log_softmax(logs.mean(axis=1), codes)
    p_bar = np.exp(log_bar)
    kl = np.column_stack(
        [np.bincount(codes, weights=p_bar * (log_bar - logs[:, j])) for j in range(len(members))]
    )
    races = pd.unique(frame["race_id"].to_numpy(dtype=object))
    return pd.Series(kl.mean(axis=1), index=pd.Index(races, name="race_id"), name="ambiguity")


def ambiguity(frame: pd.DataFrame, members: Sequence[str]) -> float:
    """Heskes' ambiguity term: the mean over races of `ambiguity_terms`."""
    return float(ambiguity_terms(frame, members).mean())


def demeaned(frame: pd.DataFrame, members: Sequence[str]) -> np.ndarray:
    """The members' columns less each race's mean, one column per member."""
    codes = _race_codes(frame)
    sizes = np.bincount(codes)
    out = []
    for k in members:
        v = frame[k].to_numpy(dtype=float)
        out.append(v - (np.bincount(codes, weights=v) / sizes)[codes])
    return np.column_stack(out)


def member_correlations(frame: pd.DataFrame, members: Sequence[str]) -> pd.DataFrame:
    """The Pearson correlations of the members' logp demeaned within race."""
    members = list(members)
    corr = np.corrcoef(demeaned(frame, members), rowvar=False)
    return pd.DataFrame(np.atleast_2d(corr), index=members, columns=members)
