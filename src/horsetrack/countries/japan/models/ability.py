"""C2 ability v0 (#92, epic #73; 08 §3.2 C2, §4 E33, E34; 05 §2.2, §2.4, §2.5): a discounted
top-k Plackett-Luce model of the finishing order with pooled jockey, trainer and sire effects.

Likelihood. A race's top-k order is k successive choices: stage s picks the runner placed s from
the runners not placed 1..s-1, with probability softmax(u) over them. `pl_loglik` sums over races,
and over the stages s <= k that have a runner placed s, discounts[s-1] * (u of that runner less
the logsumexp of u over the stage's choice set). A runner placed beyond k, or without a place (a
non-finisher), stays in every stage's choice set and is never chosen: at k = 3 positions 4+ add no
stage of their own (eased runners, 05 §2.5).

Dead heats (05 §2.4) use JRA places: co-placed runners share the place and the next runner skips
it (1, 1, 3). The m runners placed s each count at weight 1/m in stage s, against the same choice
set, so the stage carries one unit as #69 scores co-winners; a place the dead heat skips has no
runner and adds no stage. At k = 1 with discounts (1,) this is `models.condlogit`'s likelihood,
dead heats included.

Model. u = X @ beta + the sum over entity columns e of gamma_e[level]. beta is unpenalised; each
entity column's effects have a N(0, tau_e^2) prior, so a level seen in few races is shrunk toward
0, the average level. The fit is the penalised MAP: Newton's method on the concave objective
-pl_loglik + 1/2 * sum_e |gamma_e|^2 / tau_e^2, each step halved until it does not worsen it.
tau_e^2 is set on the training rows by empirical Bayes: MacKay's evidence fixed point
tau_e^2 = |gamma_e|^2 / (n_e - tr(Sigma_e) / tau_e^2), n_e the levels seen and Sigma the Laplace
posterior covariance, the inverse of the penalised Hessian at the MAP (floored at TAU2_FLOOR).
The fit reads only the rows it is given: they are its training window.

`predict` reads race_id, horse_id, the features and the entity columns only. se_u = sqrt(d' Sigma
d), d the runner's design row (the standard error of u itself, not of u less its race mean); an
entity level the fit never saw, or a missing id, has effect 0 and adds its prior variance tau_e^2.
With discounts below 1 Sigma is the inverse Hessian of the discounted objective, which is
conservative: the sandwich H^-1 J H^-1 is no larger, since a stage's information enters J at
weight w^2 <= w. p_ability = softmax(u) within race.

`fit_discounts` picks the top-3 discounts (1, w2, w3) from DISCOUNT_GRID by the summed winner log
loss of AbilityModels fitted on each fold's training rows and predicting its test rows. It reads
only the rows the folds name.

Memory: flat arrays sorted by choice set (np.add.reduceat over segments) and a sparse design; the
only dense matrices are P x P, P the features plus the entity levels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

MAX_STEPS = 200
STEP_TOL = 1e-10  # the largest Newton step on a coefficient when a fit stops
HALVINGS = 60
EB_MAX = 100
EB_TOL = 1e-6  # the largest change in ln tau2 the fixed point still asks for when it stops
EB_JUMP = 3.0  # the largest step in ln tau2
TAU2_START = 0.1
TAU2_FLOOR = 1e-6
STEPS = (0.0, 0.25, 0.5, 0.75, 1.0)
# w1 = 1 (the Win stage at full weight); w3 <= w2 (a lower place is no more informative)
DISCOUNT_GRID = tuple((1.0, w2, w3) for w2 in STEPS for w3 in STEPS if w3 <= w2)
OUT_COLUMNS = ["race_id", "horse_id", "u", "se_u", "p_ability"]

__all__ = ["DISCOUNT_GRID", "AbilityModel", "fit_discounts", "pl_loglik"]


@dataclass(frozen=True)
class _Stages:
    """The stacked choice sets: rows (positions into the input) set by set, stage by stage."""

    rows: np.ndarray
    starts: np.ndarray
    sizes: np.ndarray
    weight: np.ndarray  # per stacked row: the stage's discount / m on a chosen row, else 0
    discount: np.ndarray  # per choice set


def _discounts(k: int, discounts: Sequence[float] | None) -> tuple[float, ...]:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    if discounts is None:
        return (1.0,) * int(k)
    out = tuple(float(w) for w in discounts)
    if len(out) != k:
        raise ValueError(f"discounts has {len(out)} entries, k is {k}")
    if not all(math.isfinite(w) and w >= 0 for w in out):
        raise ValueError(f"discounts must be finite and non-negative, got {out}")
    return out


def _race_codes(race_id) -> np.ndarray:
    race_id = np.asarray(race_id, dtype=object)
    if pd.isna(race_id).any():
        raise ValueError("race_id has missing values; every row needs its race")
    return pd.factorize(race_id)[0]


def _stages(codes: np.ndarray, pos: np.ndarray, discounts: tuple[float, ...]) -> _Stages:
    n_races = int(codes.max()) + 1 if len(codes) else 0
    rows, sets, weight, discount = [], [], [], []
    for s, w in enumerate(discounts, start=1):
        if w == 0:
            continue
        chosen = pos == s  # a missing place is never chosen
        m = np.bincount(codes, weights=chosen, minlength=n_races)
        keep = np.flatnonzero((m[codes] > 0) & ~(pos < s))  # a missing place stays in
        keep = keep[np.argsort(codes[keep], kind="stable")]
        rows.append(keep)
        sets.append(codes[keep] + (s - 1) * n_races)
        weight.append(np.where(chosen[keep], w / m[codes[keep]], 0.0))
        discount.append(np.full(int((m > 0).sum()), w))
    if not rows or not sum(len(r) for r in rows):
        empty = np.array([], dtype=np.int64)
        return _Stages(empty, empty, empty, np.array([]), np.array([]))
    sets = np.concatenate(sets)
    starts = np.flatnonzero(np.r_[True, sets[1:] != sets[:-1]])
    return _Stages(
        rows=np.concatenate(rows),
        starts=starts,
        sizes=np.diff(np.r_[starts, len(sets)]),
        weight=np.concatenate(weight),
        discount=np.concatenate(discount),
    )


def _softmax(v: np.ndarray, st: _Stages) -> tuple[np.ndarray, np.ndarray]:
    """p within each choice set and each set's logsumexp."""
    top = np.maximum.reduceat(v, st.starts)
    e = np.exp(v - np.repeat(top, st.sizes))
    total = np.add.reduceat(e, st.starts)
    return e / np.repeat(total, st.sizes), np.log(total) + top


def _loglik(v: np.ndarray, st: _Stages) -> float:
    if not len(st.rows):
        return 0.0
    _, lse = _softmax(v, st)
    return float(st.weight @ v - st.discount @ lse)


def pl_loglik(u, race_id, finish_pos, *, k: int, discounts: Sequence[float]) -> float:
    """The discounted top-k Plackett-Luce log-likelihood of u (see the module docstring). One
    race_id per row, rows in any order; finish_pos 1 = winner, missing = no place."""
    weights = _discounts(k, discounts)
    u = np.asarray(u, dtype=float)
    codes = _race_codes(race_id)
    pos = np.asarray(pd.to_numeric(pd.Series(np.asarray(finish_pos))), dtype=float)
    if not (len(u) == len(codes) == len(pos)):
        raise ValueError(f"u, race_id and finish_pos have {len(u)}, {len(codes)}, {len(pos)} rows")
    if not np.isfinite(u).all():
        raise ValueError("u has missing or non-finite values")
    st = _stages(codes, pos, weights)
    return _loglik(u[st.rows], st)


def _features(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise ValueError(f"frame lacks feature column(s) {missing}")
    X = frame[features].to_numpy(dtype=float).reshape(len(frame), len(features))
    if not np.isfinite(X).all():
        raise ValueError("a feature has missing or non-finite values")
    return X


class AbilityModel:
    """u = features @ coef + one shrunk effect per entity column; see the module docstring."""

    def __init__(
        self,
        *,
        k: int,
        discounts: Sequence[float] | None = None,
        entity_cols: Sequence[str] = (),
        features: Sequence[str] = (),
    ) -> None:
        self.k = int(k)
        self.discounts = _discounts(k, discounts)
        self.entity_cols = tuple(entity_cols)
        self.features = list(features)
        if len(set(self.entity_cols) | set(self.features)) != len(self.entity_cols) + len(
            self.features
        ):
            raise ValueError("a column is named twice among the features and entity columns")
        self.coef_: pd.Series | None = None
        self.effects_: dict[str, pd.Series] = {}
        self.prior_var_: dict[str, float] = {}
        self.loglik_: float | None = None
        self._levels: dict[str, pd.Index] = {}
        self._cov: np.ndarray | None = None

    # -- fit ------------------------------------------------------------------------------------

    def _design(self, frame: pd.DataFrame) -> tuple[sp.csr_matrix, list[np.ndarray]]:
        """The sparse design over the fitted columns and each entity column's level positions
        (-1 for a level the fit never saw or a missing id)."""
        X = _features(frame, self.features)
        n, f = X.shape
        blocks, codes, offset = [sp.csr_matrix(X)], [], f
        for col in self.entity_cols:
            if col not in frame.columns:
                raise ValueError(f"frame lacks entity column {col!r}")
            code = self._levels[col].get_indexer(frame[col].to_numpy(dtype=object))
            seen = np.flatnonzero(code >= 0)
            size = len(self._levels[col])
            blocks.append(sp.csr_matrix((np.ones(len(seen)), (seen, code[seen])), shape=(n, size)))
            codes.append(np.where(code >= 0, code + offset, -1))
            offset += size
        return sp.hstack(blocks, format="csr"), codes

    def fit(self, frame: pd.DataFrame) -> AbilityModel:
        """The penalised MAP on frame's rows (race_id, horse_id, finish_pos, the features and the
        entity columns), with empirical-Bayes prior variances. Returns self."""
        for col in ("race_id", "finish_pos"):
            if col not in frame.columns:
                raise ValueError(f"frame lacks {col!r}")
        self._levels = {}
        for col in self.entity_cols:
            if col not in frame.columns:
                raise ValueError(f"frame lacks entity column {col!r}")
            values = frame[col].to_numpy(dtype=object)
            self._levels[col] = pd.Index(pd.unique(values[~pd.isna(values)]))
        D, _ = self._design(frame)
        codes = _race_codes(frame["race_id"])
        pos = pd.to_numeric(frame["finish_pos"]).to_numpy(dtype=float)
        st = _stages(codes, pos, self.discounts)
        if not len(st.rows):
            raise ValueError("no race has a placed runner to fit")
        Ds = D[st.rows]
        agg = sp.csr_matrix(
            (
                np.ones(len(st.rows)),
                (np.repeat(np.arange(len(st.starts)), st.sizes), np.arange(len(st.rows))),
            ),
            shape=(len(st.starts), len(st.rows)),
        )
        f = len(self.features)
        blocks, at = [], f
        for col in self.entity_cols:
            blocks.append(np.arange(at, at + len(self._levels[col])))
            at += len(self._levels[col])
        theta, loglik, cov, tau2 = _empirical_bayes(Ds, agg, st, blocks)

        self.coef_ = pd.Series(theta[:f], index=self.features, dtype=float)
        self.effects_ = {
            col: pd.Series(theta[cols], index=self._levels[col], dtype=float)
            for col, cols in zip(self.entity_cols, blocks, strict=True)
        }
        self.prior_var_ = dict(zip(self.entity_cols, tau2, strict=True))
        self.loglik_ = loglik
        self._theta, self._cov = theta, cov
        return self

    # -- predict --------------------------------------------------------------------------------

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """One row per frame row (frame's index): race_id, horse_id, u, se_u and p_ability."""
        if self._cov is None:
            raise RuntimeError("fit the model before predicting")
        D, codes = self._design(frame)
        u = D @ self._theta
        f = len(self.features)
        X = D[:, :f].toarray()
        cov = self._cov
        var = np.einsum("ij,jk,ik->i", X, cov[:f, :f], X)
        for col, c in zip(self.entity_cols, codes, strict=True):
            seen = c >= 0
            var[seen] += 2.0 * (X[seen] * cov[:f, c[seen]].T).sum(axis=1)
            var[~seen] += self.prior_var_[col]
        for a in codes:
            for b in codes:
                both = (a >= 0) & (b >= 0)
                var[both] += cov[a[both], b[both]]
        race = _race_codes(frame["race_id"])
        top = pd.Series(u).groupby(race).transform("max").to_numpy()
        e = np.exp(u - top)
        p = e / pd.Series(e).groupby(race).transform("sum").to_numpy()
        return pd.DataFrame(
            {
                "race_id": frame["race_id"].to_numpy(),
                "horse_id": frame["horse_id"].to_numpy(),
                "u": u,
                "se_u": np.sqrt(np.maximum(var, 0.0)),
                "p_ability": p,
            },
            index=frame.index,
        )[OUT_COLUMNS]


def _evaluate(theta, Ds, agg, st: _Stages):
    """The log-likelihood, the score and the information at theta."""
    v = Ds @ theta
    p, lse = _softmax(v, st)
    loglik = float(st.weight @ v - st.discount @ lse)
    dp = np.repeat(st.discount, st.sizes) * p
    score = Ds.T @ (st.weight - dp)
    first = (Ds.T @ Ds.multiply(dp[:, None])).toarray()
    mean = agg @ Ds.multiply(p[:, None])  # each choice set's p-weighted design row
    second = (mean.T @ mean.multiply(st.discount[:, None])).toarray()
    return loglik, np.asarray(score).ravel(), first - second


def _newton(theta, Ds, agg, st, precision):
    def objective(c: np.ndarray, loglik: float) -> float:
        return -loglik + 0.5 * float(precision @ (c * c))

    loglik, score, info = _evaluate(theta, Ds, agg, st)
    for _ in range(MAX_STEPS):
        try:
            step = np.linalg.solve(info + np.diag(precision), score - precision * theta)
        except np.linalg.LinAlgError as exc:
            raise ValueError("the design is not identified: singular information") from exc
        current = objective(theta, loglik)
        for _ in range(HALVINGS):
            trial = theta + step
            trial_eval = _evaluate(trial, Ds, agg, st)
            if objective(trial, trial_eval[0]) <= current or np.abs(step).max() < STEP_TOL:
                break
            step = step / 2.0
        theta = trial
        loglik, score, info = trial_eval
        if np.abs(step).max() < STEP_TOL:
            break
    else:
        raise RuntimeError(f"the ability fit did not converge in {MAX_STEPS} Newton steps")
    if not np.all(np.isfinite(theta)):
        raise RuntimeError("the ability fit stopped at non-finite coefficients")
    return theta, loglik, info


def _empirical_bayes(Ds, agg, st: _Stages, blocks: list[np.ndarray]):
    """The MAP at the prior variances that solve MacKay's fixed point, one per entity block:
    (theta, loglik, Sigma, tau2). The fixed point runs on z = ln tau2 with a per-block secant
    step on the residual ln(update) - z (the plain update when the secant has no valid slope), so
    it converges where the plain update crawls (a variance the data barely support); a block
    whose update goes below TAU2_FLOOR stays at the floor, and a step is capped at EB_JUMP."""
    theta = np.zeros(Ds.shape[1])
    z = np.full(len(blocks), math.log(TAU2_START))
    floor = math.log(TAU2_FLOOR)
    last = None
    for _ in range(EB_MAX):
        precision = np.zeros(Ds.shape[1])
        for cols, zz in zip(blocks, z, strict=True):
            precision[cols] = math.exp(-zz)
        theta, loglik, info = _newton(theta, Ds, agg, st, precision)
        cov = _inverse(info + np.diag(precision))
        update = np.empty(len(blocks))
        for b, cols in enumerate(blocks):
            effective = len(cols) - np.trace(cov[np.ix_(cols, cols)]) * math.exp(-z[b])
            square = float(theta[cols] @ theta[cols])
            ratio = square / effective if effective > 0 and square > 0 else 0.0
            update[b] = max(math.log(ratio), floor) if ratio > 0 else floor
        r = update - z
        if not len(blocks) or np.abs(r).max() < EB_TOL:
            return theta, loglik, cov, [math.exp(zz) for zz in z]
        step = r.copy()
        if last is not None:
            dz, dr = z - last[0], r - last[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                secant = -r * dz / dr
            valid = (dz != 0) & (dr * dz < 0) & np.isfinite(secant)
            step = np.where(valid, secant, r)
        last = (z, r)
        z = np.maximum(z + np.clip(step, -EB_JUMP, EB_JUMP), floor)
    raise RuntimeError(f"empirical Bayes did not converge in {EB_MAX} updates")


def _inverse(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the design is not identified: singular information") from exc


def fit_discounts(
    frame: pd.DataFrame,
    folds: Sequence,
    *,
    features: Sequence[str],
    entity_cols: Sequence[str] = (),
    grid: Sequence[tuple[float, float, float]] = DISCOUNT_GRID,
) -> tuple[float, float, float]:
    """The top-3 discounts in `grid` with the lowest winner log loss summed over the folds' test
    rows, each predicted by an AbilityModel fitted on its fold's training rows. `folds` are
    `core.eval.folds.Fold` positions into frame (the inner folds over an outer fold's training
    rows); no other row is read. A tie goes to the earlier grid entry."""
    if not len(folds):
        raise ValueError("fit_discounts needs at least one fold")
    parts = [(frame.iloc[f.train_idx], frame.iloc[f.test_idx]) for f in folds]
    best, best_loss = None, math.inf
    for discounts in grid:
        loss = 0.0
        for train, test in parts:
            model = AbilityModel(
                k=3, discounts=discounts, entity_cols=entity_cols, features=features
            ).fit(train)
            u = model.predict(test)["u"].to_numpy()
            loss -= pl_loglik(u, test["race_id"], test["finish_pos"], k=1, discounts=(1.0,))
        if loss < best_loss:
            best, best_loss = discounts, loss
    return tuple(float(w) for w in best)
