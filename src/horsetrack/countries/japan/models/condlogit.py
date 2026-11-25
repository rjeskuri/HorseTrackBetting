"""The conditional logit within race (#87; 08 §3.2 C4): the fit under the C4 meta-model.

P(i wins) = exp(u_i) / sum_j exp(u_j) over the rows of i's race, u = offset + X @ coef. The offset
enters at coefficient 1 and has no entry in `coef`; nothing is bounded. Races are treated as in
`core.eval.scoring` (the owner's answer on #69, 2026-11-24): a race with no winner is dropped, and a
dead heat with k co-winners counts each at weight 1/k, so every race carries one unit:

    loglik = sum over races of (1/k) * sum over its winners of (u_w - logsumexp of the race's u).

The objective minimised is -loglik + 1/2 * sum_j ridge_j * coef_j^2: `ridge` is the precision of a
N(0, 1/ridge) prior on each coefficient, as in `models.v0.residual`. A scalar ridge applies to
every column; an array gives one precision per column (0 leaves that coefficient unpenalised).
The fit is Newton's method on that concave objective, each step halved until it does not worsen
it. `se` is sqrt(diag((I + diag(ridge))^-1)), I the information at `coef` (the sum over races of
the p-weighted covariance of the race's X rows): at ridge 0 the inverse information. `loglik` is
the unpenalised log-likelihood at `coef`.

`se_cluster` (with `cluster`, one label per row, constant within a race; D7's unit is the meeting)
is the cluster sandwich sqrt(diag(H^-1 B H^-1)): H = I + diag(ridge), the bread of `se`, and
B = sum over clusters of g_c g_c^T, g_c the sum over the cluster's races of the unpenalised race
score sum_i (w_i - p_i) x_i. A race with no winner adds nothing; there is no small-sample factor.

Memory: flat arrays sorted by race (np.add.reduceat over race segments); nothing is quadratic in
the rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_STEPS = 200
STEP_TOL = 1e-10  # the largest Newton step on a coefficient when the fit stops
HALVINGS = 60


@dataclass(frozen=True)
class CLFit:
    """The fitted coefficients, their standard errors, the unpenalised log-likelihood there, the
    number of races in the likelihood and, with `cluster`, the cluster-sandwich standard errors."""

    coef: np.ndarray
    se: np.ndarray
    loglik: float
    n_races: int
    se_cluster: np.ndarray | None = None


@dataclass(frozen=True)
class _Races:
    """The rows of the races with a winner, sorted by race."""

    X: np.ndarray
    offset: np.ndarray
    w: np.ndarray  # 1/k on each of a race's k winners, else 0
    starts: np.ndarray
    sizes: np.ndarray
    cluster: np.ndarray | None  # each race's cluster code, in race order; None without `cluster`


def _as_float(values, name: str, n: int) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.shape[0] != n:
        raise ValueError(f"{name} has {out.shape[0]} rows, the race ids {n}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} has missing or non-finite values")
    return out


def _cluster_codes(cluster, codes: np.ndarray) -> np.ndarray:
    """One cluster code per race code from one label per row; a label missing, a race with two
    labels or an array of another length is a ValueError."""
    cluster = np.asarray(cluster, dtype=object)
    if cluster.shape != codes.shape:
        raise ValueError(f"cluster has shape {cluster.shape}, the race ids ({len(codes)},)")
    unit = pd.factorize(cluster)[0]  # a missing label (None, NaN) is -1
    if (unit < 0).any():
        raise ValueError("cluster has missing labels; every row needs its cluster")
    by_race = pd.Series(unit).groupby(codes)
    low, high = by_race.min().to_numpy(), by_race.max().to_numpy()
    if (low != high).any():
        raise ValueError("a race's rows carry more than one cluster label")
    return low


def _races(X, race_id, won, offset, cluster=None) -> _Races:
    race_id = np.asarray(race_id, dtype=object)
    n = len(race_id)
    if pd.isna(race_id).any():
        raise ValueError("race_id has missing values; every row needs its race")
    X = np.asarray(X, dtype=float)
    X = _as_float(X.reshape(len(X), -1) if X.ndim == 1 else X, "X", n)
    if X.ndim != 2:
        raise ValueError(f"X must be a (rows, k) array, got shape {X.shape}")
    won = np.asarray(won)
    if won.shape != (n,):
        raise ValueError(f"won has shape {won.shape}, the race ids ({n},)")
    if pd.isna(won).any():
        raise ValueError("won has missing values")
    won = won.astype(bool)
    offset = np.zeros(n) if offset is None else _as_float(offset, "offset", n)
    if offset.ndim != 1:
        raise ValueError(f"offset must be one value per row, got shape {offset.shape}")

    codes = pd.factorize(race_id)[0]
    race_cluster = None if cluster is None else _cluster_codes(cluster, codes)
    winners = np.bincount(codes, weights=won)
    rows = np.flatnonzero(winners[codes] > 0)
    rows = rows[np.argsort(codes[rows], kind="stable")]
    if len(rows) == 0:
        raise ValueError("no race has a winner to fit")
    race = codes[rows]
    starts = np.flatnonzero(np.r_[True, race[1:] != race[:-1]])
    return _Races(
        X=X[rows],
        offset=offset[rows],
        w=won[rows] / winners[race],
        starts=starts,
        sizes=np.diff(np.r_[starts, len(rows)]),
        cluster=None if race_cluster is None else race_cluster[race[starts]],
    )


def _evaluate(coef: np.ndarray, races: _Races) -> tuple[float, np.ndarray, np.ndarray]:
    """The log-likelihood, the score and the information at coef."""
    u = races.offset + races.X @ coef
    top = np.maximum.reduceat(u, races.starts)
    e = np.exp(u - np.repeat(top, races.sizes))
    total = np.add.reduceat(e, races.starts)
    loglik = float(races.w @ u - (np.log(total) + top).sum())
    p = e / np.repeat(total, races.sizes)
    xp = races.X * p[:, None]
    xbar = np.add.reduceat(xp, races.starts, axis=0)
    score = races.X.T @ races.w - xbar.sum(axis=0)
    info = xp.T @ races.X - xbar.T @ xbar
    return loglik, score, info


def _cluster_meat(coef: np.ndarray, races: _Races) -> np.ndarray:
    """B: the sum over clusters of g_c g_c^T, g_c the summed unpenalised race scores."""
    u = races.offset + races.X @ coef
    top = np.maximum.reduceat(u, races.starts)
    e = np.exp(u - np.repeat(top, races.sizes))
    p = e / np.repeat(np.add.reduceat(e, races.starts), races.sizes)
    race_score = np.add.reduceat((races.w - p)[:, None] * races.X, races.starts, axis=0)
    unit = pd.factorize(races.cluster)[0]
    g = np.zeros((unit.max() + 1, races.X.shape[1]))
    np.add.at(g, unit, race_score)
    return g.T @ g


def _penalty(ridge, k: int) -> np.ndarray:
    penalty = np.broadcast_to(np.asarray(ridge, dtype=float), (k,)).copy()
    if not np.all(np.isfinite(penalty)) or np.any(penalty < 0):
        raise ValueError(f"ridge must be finite and non-negative, got {ridge!r}")
    return penalty


def fit_conditional_logit(
    X: np.ndarray,
    race_id: np.ndarray,
    won: np.ndarray,
    *,
    offset: np.ndarray | None = None,
    ridge: float | np.ndarray = 0.0,
    cluster: np.ndarray | None = None,
) -> CLFit:
    """The (ridge-penalised) conditional-logit maximum likelihood of u = offset + X @ coef.

    `X` is (rows, k), `race_id` one label per row (rows need not be grouped by race), `won` a bool
    per row, `offset` None or one float per row and `cluster` None or one label per row, constant
    within a race (it fills `se_cluster` and changes nothing else). Raises ValueError on missing
    or non-finite X or offset, inputs of different lengths, a negative ridge, a missing cluster
    label or a race with two, no race with a winner, or a design
    the races cannot identify (singular information at ridge 0); RuntimeError when Newton does not
    converge (no finite maximum, as when a column separates the winners perfectly).
    """
    races = _races(X, race_id, won, offset, cluster)
    k = races.X.shape[1]
    penalty = _penalty(ridge, k)

    def objective(c: np.ndarray, loglik: float) -> float:
        return -loglik + 0.5 * float(penalty @ (c * c))

    coef = np.zeros(k)
    loglik, score, info = _evaluate(coef, races)
    for _ in range(MAX_STEPS):
        if k == 0:
            break
        try:
            step = np.linalg.solve(info + np.diag(penalty), score - penalty * coef)
        except np.linalg.LinAlgError as exc:
            raise ValueError("the design is not identified: singular information") from exc
        current = objective(coef, loglik)
        for _ in range(HALVINGS):
            trial = coef + step
            trial_eval = _evaluate(trial, races)
            if objective(trial, trial_eval[0]) <= current or np.abs(step).max() < STEP_TOL:
                break
            step = step / 2.0
        coef = trial
        loglik, score, info = trial_eval
        if np.abs(step).max() < STEP_TOL:
            break
    else:
        raise RuntimeError(f"the conditional logit did not converge in {MAX_STEPS} Newton steps")
    if not np.all(np.isfinite(coef)):
        raise RuntimeError(f"the conditional logit stopped at non-finite coefficients {coef}")
    try:
        cov = np.linalg.inv(info + np.diag(penalty))
    except np.linalg.LinAlgError as exc:
        raise ValueError("the design is not identified: singular information") from exc
    se_cluster = None
    if races.cluster is not None:
        se_cluster = np.sqrt(np.diag(cov @ _cluster_meat(coef, races) @ cov))
    return CLFit(
        coef=coef,
        se=np.sqrt(np.diag(cov)),
        loglik=loglik,
        n_races=len(races.starts),
        se_cluster=se_cluster,
    )
