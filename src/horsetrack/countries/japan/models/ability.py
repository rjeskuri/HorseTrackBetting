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
The fit reads only the rows it is given: they are its training window. Each update of the fixed
point but the last takes one Newton step; once the residual is within EB_TOL the MAP is solved
to STEP_TOL at the same prior variances and the residual re-checked there, so a fit stops only at
a MAP within STEP_TOL whose prior variances meet the fixed point within EB_TOL.

`predict` reads race_id, horse_id, the features and the entity columns only. se_u = sqrt(d' Sigma
d), d the runner's design row (the standard error of u itself, not of u less its race mean); an
entity level the fit never saw, or a missing id, has effect 0 and adds its prior variance tau_e^2.
With discounts below 1 Sigma is the inverse Hessian of the discounted objective, which is
conservative: the sandwich H^-1 J H^-1 is no larger, since a stage's information enters J at
weight w^2 <= w. p_ability = softmax(u) within race.

`fit_discounts` picks the top-3 discounts (1, w2, w3) from DISCOUNT_GRID by the summed winner log
loss of AbilityModels fitted on each fold's training rows and predicting its test rows. It reads
only the rows the folds name.

Memory and time: flat arrays sorted by choice set (np.add.reduceat over segments) and a sparse
design over the input rows; the information's choice-set term is formed race by race (`_Problem`),
and the penalised Hessian is solved and inverted by Cholesky. The only dense matrices are P x P,
P the features plus the entity levels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse as sp

MAX_STEPS = 200
STEP_TOL = 1e-10  # the largest Newton step on a coefficient when a fit stops
LOOSE_TOL = 1e-6  # the same, for the MAP inside an empirical-Bayes update before the last
INNER_STEPS = 1  # at most this many Newton steps per empirical-Bayes update before the last
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
    race: np.ndarray  # per choice set: its race code
    stage: np.ndarray  # per choice set: its stage s


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
    rows, sets, weight, discount, race, stage = [], [], [], [], [], []
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
        has = np.flatnonzero(m > 0)  # the stage's races, in the order of its sets
        discount.append(np.full(len(has), w))
        race.append(has)
        stage.append(np.full(len(has), s))
    if not rows or not sum(len(r) for r in rows):
        empty = np.array([], dtype=np.int64)
        return _Stages(empty, empty, empty, np.array([]), np.array([]), empty, empty)
    sets = np.concatenate(sets)
    starts = np.flatnonzero(np.r_[True, sets[1:] != sets[:-1]])
    return _Stages(
        rows=np.concatenate(rows),
        starts=starts,
        sizes=np.diff(np.r_[starts, len(sets)]),
        weight=np.concatenate(weight),
        discount=np.concatenate(discount),
        race=np.concatenate(race),
        stage=np.concatenate(stage),
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
        problem = _problem(D, codes, pos, st)
        f = len(self.features)
        blocks, at = [], f
        for col in self.entity_cols:
            blocks.append(np.arange(at, at + len(self._levels[col])))
            at += len(self._levels[col])
        theta, loglik, cov, tau2 = _empirical_bayes(problem, blocks)

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


@dataclass(frozen=True)
class _Problem:
    """A fit's design over its input rows and the structure of its choice sets.

    A race's sets are nested (stage s keeps the runners not placed before s), so set k's
    p-weighted design row is (a_r + e_k) / Z_k: a_r sums q * d over `base`, the rows still in at
    the race's last stage, e_k over `extra`, the rows set k holds beyond them (placed from s_k up
    to the last stage), with q = exp(u - the largest u in base) and Z_k the sum of q over set k.
    The information's second term, sum_k w_k m_k m_k', is then one outer product per race and
    small ones per set, all sums of products of positive weights."""

    D: sp.csr_matrix
    st: _Stages
    row_race: np.ndarray
    base_rows: np.ndarray  # input rows, race by race
    base_indptr: np.ndarray  # per race code
    nonempty: np.ndarray  # race codes with a base row
    extra_rows: np.ndarray  # input rows, set by set
    extra_indptr: np.ndarray  # per choice set
    to_race: sp.csr_matrix  # races x sets: 1 where the set is the race's


def _problem(D: sp.csr_matrix, codes: np.ndarray, pos: np.ndarray, st: _Stages) -> _Problem:
    n_races, n_sets = int(codes.max()) + 1, len(st.starts)
    last = np.zeros(n_races)
    np.maximum.at(last, st.race, st.stage)
    row_last = last[codes]
    base = np.flatnonzero((row_last > 0) & ~(pos < row_last))  # a missing place stays in
    base = base[np.argsort(codes[base], kind="stable")]
    per_race = np.bincount(codes[base], minlength=n_races)
    lookup = np.full((int(st.stage.max()) + 1, n_races), -1)
    lookup[st.stage, st.race] = np.arange(n_sets)
    sets, rows = [], []
    for s in np.unique(st.stage):
        held = np.flatnonzero((pos >= s) & (pos < row_last))  # placed s .. the last stage - 1
        k = lookup[s, codes[held]]
        sets.append(k[k >= 0])
        rows.append(held[k >= 0])
    sets, rows = np.concatenate(sets), np.concatenate(rows)
    order = np.lexsort((rows, sets))
    return _Problem(
        D=D,
        st=st,
        row_race=codes,
        base_rows=base,
        base_indptr=np.r_[0, np.cumsum(per_race)],
        nonempty=np.flatnonzero(per_race > 0),
        extra_rows=rows[order],
        extra_indptr=np.r_[0, np.cumsum(np.bincount(sets, minlength=n_sets))],
        to_race=sp.csr_matrix(
            (np.ones(n_sets), (st.race, np.arange(n_sets))), shape=(n_races, n_sets)
        ),
    )


def _scale_rows(matrix: sp.csr_matrix, by: np.ndarray) -> sp.csr_matrix:
    out = matrix.copy()
    out.data *= np.repeat(by, np.diff(out.indptr))
    return out


def _evaluate(theta, pb: _Problem):
    """The log-likelihood, the score and the information at theta. Sums over stacked rows that
    are linear in the design are taken on the input rows (a row's stacked weights added first);
    the second term of the information is formed race by race (see `_Problem`)."""
    st, D = pb.st, pb.D
    n, n_races, n_sets = D.shape[0], pb.base_indptr.size - 1, len(st.starts)
    v_all = D @ theta
    v = v_all[st.rows]
    p, lse = _softmax(v, st)
    loglik = float(st.weight @ v - st.discount @ lse)
    dp = np.repeat(st.discount, st.sizes) * p
    score = D.T @ np.bincount(st.rows, weights=st.weight - dp, minlength=n)
    first = (D.T @ _scale_rows(D, np.bincount(st.rows, weights=dp, minlength=n))).toarray()

    top = np.zeros(n_races)
    top[pb.nonempty] = np.maximum.reduceat(v_all[pb.base_rows], pb.base_indptr[pb.nonempty])
    with np.errstate(over="ignore"):  # rows of races without a set are never read
        q = np.exp(v_all - top[pb.row_race])
    z = np.add.reduceat(q[st.rows], st.starts)
    coef = st.discount / (z * z)
    a = sp.csr_matrix((q[pb.base_rows], pb.base_rows, pb.base_indptr), shape=(n_races, n)) @ D
    e = sp.csr_matrix((q[pb.extra_rows], pb.extra_rows, pb.extra_indptr), shape=(n_sets, n)) @ D
    e_coef = _scale_rows(e, coef)
    cross = (a.T @ (pb.to_race @ e_coef)).toarray()
    second = (a.T @ _scale_rows(a, pb.to_race @ coef)).toarray() + cross + cross.T
    second += (e.T @ e_coef).toarray()
    return loglik, np.asarray(score).ravel(), first - second


def _solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """matrix^-1 rhs for the penalised Hessian (symmetric positive definite when identified):
    Cholesky, else LU; singular raises ValueError."""
    try:
        factor = scipy.linalg.cho_factor(matrix, check_finite=False)
        return scipy.linalg.cho_solve(factor, rhs, check_finite=False)
    except np.linalg.LinAlgError:
        pass
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the design is not identified: singular information") from exc


def _newton(theta, pb: _Problem, precision, *, tol=STEP_TOL, at=None, steps=None):
    """Newton's method from theta until a step's largest entry is below tol, or after `steps`
    steps when given (no error then). `at` is `_evaluate` at theta when already known (it does
    not depend on the precision). Returns (theta, loglik, info, the evaluation at theta)."""

    def objective(c: np.ndarray, loglik: float) -> float:
        return -loglik + 0.5 * float(precision @ (c * c))

    loglik, score, info = _evaluate(theta, pb) if at is None else at
    for _ in range(MAX_STEPS if steps is None else steps):
        step = _solve(info + np.diag(precision), score - precision * theta)
        current = objective(theta, loglik)
        for _ in range(HALVINGS):
            trial = theta + step
            trial_eval = _evaluate(trial, pb)
            if objective(trial, trial_eval[0]) <= current or np.abs(step).max() < tol:
                break
            step = step / 2.0
        theta = trial
        loglik, score, info = trial_eval
        if np.abs(step).max() < tol:
            break
    else:
        if steps is None:
            raise RuntimeError(f"the ability fit did not converge in {MAX_STEPS} Newton steps")
    if not np.all(np.isfinite(theta)):
        raise RuntimeError("the ability fit stopped at non-finite coefficients")
    return theta, loglik, info, (loglik, score, info)


def _empirical_bayes(pb: _Problem, blocks: list[np.ndarray], base_precision=None):
    """The MAP at the prior variances that solve MacKay's fixed point, one per entity block:
    (theta, loglik, Sigma, tau2). `base_precision` (per coefficient, default 0: unpenalised) is
    the fixed prior precision of the coefficients outside the blocks (#95's cell deviations at
    tau_cell). The fixed point runs on z = ln tau2 with a per-block secant
    step on the residual ln(update) - z (the plain update when the secant has no valid slope), so
    it converges where the plain update crawls (a variance the data barely support); a block
    whose update goes below TAU2_FLOOR stays at the floor, and a step is capped at EB_JUMP.
    Until the residual is within EB_TOL each update follows INNER_STEPS Newton steps; that MAP
    is then solved to STEP_TOL at the same z and the residual re-checked there, so the returned
    fit is a STEP_TOL MAP whose z meets the fixed point within EB_TOL."""
    size = pb.D.shape[1]
    theta = np.zeros(size)
    z = np.full(len(blocks), math.log(TAU2_START))
    floor = math.log(TAU2_FLOOR)
    last, at, tight = None, None, not len(blocks)
    for _ in range(EB_MAX):
        precision = np.zeros(size) if base_precision is None else np.array(base_precision)
        for cols, zz in zip(blocks, z, strict=True):
            precision[cols] = math.exp(-zz)
        tol, steps = (STEP_TOL, None) if tight else (LOOSE_TOL, INNER_STEPS)
        theta, loglik, info, at = _newton(theta, pb, precision, tol=tol, at=at, steps=steps)
        cov = _inverse(info + np.diag(precision))
        update = np.empty(len(blocks))
        for b, cols in enumerate(blocks):
            effective = len(cols) - np.trace(cov[np.ix_(cols, cols)]) * math.exp(-z[b])
            square = float(theta[cols] @ theta[cols])
            ratio = square / effective if effective > 0 and square > 0 else 0.0
            update[b] = max(math.log(ratio), floor) if ratio > 0 else floor
        r = update - z
        if not len(blocks) or np.abs(r).max() < EB_TOL:
            if tight:
                return theta, loglik, cov, [math.exp(zz) for zz in z]
            tight = True  # solve this MAP to STEP_TOL at the same z, then re-check
            continue
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
    """The inverse of the penalised Hessian: Cholesky (LAPACK potrf, potri), else LU."""
    upper, info = scipy.linalg.lapack.dpotrf(matrix, lower=False, clean=False)
    if info == 0:
        inv, info = scipy.linalg.lapack.dpotri(upper, lower=False)
        if info == 0:
            return np.triu(inv) + np.triu(inv, 1).T
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
