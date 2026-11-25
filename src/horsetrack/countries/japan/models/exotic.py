"""The C5 exotic layer: Win probabilities to exotic-pool probabilities, priced against the pools'
own dividends (ticket #99, epic #73; research 08 §3.2 C5, §4 E43, E44).

- `dividend_share` / `dividend_shares`: the dividend-implied share of a pool's money on a winning
  combination, R * 100 / (D + 5): R the race's own payout rate (`takeout.race_takeout`, boosts
  included, #76), D the dividend per 100 yen, 5 yen the mean breakage (dividends are floored to
  10 yen) added back (04 §1.2-1.3). This is the exotic market baseline of 08 R15 for a race with
  one winning combination.
- `winning_combinations`: the combinations a result pays, dead heats per 04 §1.4 (every
  combination some tie-break of the dead heats would make the result).
- `Ordering`: the ordered first three from one race's Win probabilities, a softmax over the
  runners left at each stage. Families: "harville"; "henery" (independent unit-variance normal
  running times, means solved so the Win probabilities hold, the trifecta by quadrature);
  "quadratic_context" (HKTrack research 16/19: slope 1 + a + h (mean ln p placed - h0) +
  e (H - e0), curvature b, per stage); "discounted_harville" (#115's v0 family, weights p^lambda,
  fitted by `models.v0.ordering.fit_ordering`). `.fit(frame, folds)` refits on each fold's
  training rows only.
- `pool_blend`: the log-linear pool of the model and the pool's own board, u_c = a ln p_model(c) +
  b ln q_pool(c), for races whose full final board is logged (08 R21).

Pool codes are jra.yaml's. A combination is the store's `dividends.combination` string: horse
numbers joined by ",", in finishing order for FORECAST and TIERCE, ascending for QUINELLA,
QUINELLA_PLACE and TRIO; WIN and PLACE name one horse.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import cache
from itertools import combinations, groupby, permutations, product
from numbers import Real

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp, ndtr

from horsetrack.core import config
from horsetrack.countries.japan import takeout
from horsetrack.countries.japan.models.v0 import ordering as v0_ordering

WIN, PLACE, QUINELLA, WIDE = "WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE"
EXACTA, TRIO, TRIFECTA = "FORECAST", "TRIO", "TIERCE"
POOLS = (WIN, PLACE, QUINELLA, WIDE, EXACTA, TRIO, TRIFECTA)
JOINT_POOLS = (QUINELLA, WIDE, EXACTA, TRIO, TRIFECTA)

UNIT_STAKE = 100.0  # yen: dividends are quoted per 100 yen
MEAN_BREAKAGE = 5.0  # yen per winning 100-yen ticket: dividends are floored to 10 yen (04 §1.3)

FAMILIES = ("harville", "henery", "quadratic_context", "discounted_harville")
QC_KEYS = ("a2", "b2", "h2", "e2", "a3", "b3", "h3", "e3")
DH_KEYS = ("lambda2", "lambda3")
# (h0, e0): where the context terms vanish, about the mean ln p of a placed runner and the entropy
# of a full field. HKTrack research 19's centring; another one gives the same model with another a.
CONTEXT_REF = (-1.6, 2.2)
# Loose bounds on (a, b, h, e) per stage (HKTrack's, with a measured from Harville's slope of 1):
# the likelihood is convex, so they only stop a separable training set running away.
QC_BOUNDS = ((-3.0, 2.0), (-1.0, 1.0), (-2.0, 2.0), (-2.0, 2.0))

# Henery quadrature: a uniform grid in running time (unit-variance normals), PAD standard deviations
# beyond the fastest and slowest means. The trapezoid rule is near exact for the smooth,
# fast-decaying win integrals; the nested third-place integral is O(STEP^2) and is renormalised.
HENERY_STEP = 0.02
HENERY_PAD = 7.0
HENERY_TOL = 1e-11
HENERY_MAX_ITER = 100


# ---------------------------------------------------------------------------------------------
# Dividend-implied shares
# ---------------------------------------------------------------------------------------------


def dividend_share(dividend: float, payout_rate: float) -> float:
    """payout_rate * 100 / (dividend + 5): the pool's share on a winning combination paying
    `dividend` per 100 yen, mean breakage added back. ValueError for a dividend that is not a
    positive number or a payout rate outside (0, 1]."""
    d = _real(dividend, "dividend")
    r = _real(payout_rate, "payout_rate")
    if not d > 0:
        raise ValueError(f"a dividend must be positive, got {dividend!r}")
    if not 0 < r <= 1:
        raise ValueError(f"a payout rate must be in (0, 1], got {payout_rate!r}")
    return r * UNIT_STAKE / (d + MEAN_BREAKAGE)


def dividend_shares(
    dividends: pd.DataFrame, *, calendar: takeout.BoostCalendar | None = None
) -> pd.Series:
    """`dividend_share` for every row of a frame with the store's dividends columns (race_id, pool,
    combination, dividend), each at its own race's payout rate, `takeout.race_takeout(race_id,
    pool)` with `calendar` (None: the shipped boost calendar). An optional `race_class_raw`
    column is passed on for the age-restricted boosts. The Series is on the frame's index."""
    missing = [c for c in ("race_id", "pool", "combination", "dividend") if c not in dividends]
    if missing:
        raise ValueError(f"the dividends frame is missing columns: {missing}")
    classes = (
        dividends["race_class_raw"]
        if "race_class_raw" in dividends
        else pd.Series(None, index=dividends.index, dtype=object)
    )
    rates: dict[tuple, float] = {}
    shares = []
    for race_id, pool, label, dividend in zip(
        dividends["race_id"], dividends["pool"], classes, dividends["dividend"], strict=True
    ):
        label = None if label is None or (isinstance(label, float) and math.isnan(label)) else label
        key = (race_id, pool, label)
        if key not in rates:
            rates[key] = takeout.race_takeout(
                race_id, pool, calendar=calendar, race_class_raw=label
            ).payout_rate
        shares.append(dividend_share(dividend, rates[key]))
    return pd.Series(shares, index=dividends.index, dtype=float, name="share")


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real | np.number):
        raise ValueError(f"{name} must be a number, got {value!r}")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return x


# ---------------------------------------------------------------------------------------------
# Winning combinations
# ---------------------------------------------------------------------------------------------


@cache
def _jra_pools() -> Mapping[str, config.Pool]:
    return config.load_source("japan", "jra").pools


def _check_pool(pool: str) -> None:
    if pool not in POOLS:
        raise ValueError(f"unknown pool {pool!r}: expected one of {POOLS}")


def _places_paid(pool: str, starters: int) -> int:
    """The first `places` finishers a pool's combinations are drawn from."""
    if pool == PLACE:
        return _jra_pools()[PLACE].places_for(starters)
    return {WIN: 1, QUINELLA: 2, EXACTA: 2, WIDE: 3, TRIO: 3, TRIFECTA: 3}[pool]


def _key(horses: Sequence[int]) -> str:
    return ",".join(str(int(h)) for h in horses)


def _combinations_of(top: Sequence[int], pool: str) -> list[str]:
    """The combinations a strict finishing order `top` (first places only) pays in `pool`."""
    if pool == WIN:
        return [_key(top[:1])]
    if pool == PLACE:
        return [_key([h]) for h in top]
    if pool == QUINELLA:
        return [_key(sorted(top[:2]))]
    if pool == EXACTA:
        return [_key(top[:2])]
    if pool == WIDE:
        return [_key(pair) for pair in combinations(sorted(top[:3]), 2)]
    if pool == TRIO:
        return [_key(sorted(top[:3]))]
    return [_key(top[:3])]


def winning_combinations(finish: Mapping[int, int], pool: str) -> list[str]:
    """The combinations `pool` pays for a result, each once.

    `finish` maps every starter's horse number to its finishing position (ties share a position,
    the next runner skips: 1, 2, 3, 3, 5; None or NaN for a runner that did not finish). A dead
    heat pays every combination some tie-break of it would make the result (04 §1.4): a dead heat
    for third pays both Trios and both Trifectas, and Wide pays every pair of a possible first
    three, never the two horses that dead-heated for third. PLACE pays the first two or three by
    the number of starters (jra.yaml); a pool not offered to the field, or a result with too few
    finishers, pays nothing. ValueError for an unknown pool.
    """
    _check_pool(pool)
    places = _places_paid(pool, len(finish))
    if places == 0:
        return []
    placed = sorted(
        (int(pos), int(h))
        for h, pos in finish.items()
        if pos is not None and not (isinstance(pos, float) and math.isnan(pos)) and pos <= places
    )
    groups = [[h for _, h in g] for _, g in groupby(placed, key=lambda item: item[0])]
    if sum(len(g) for g in groups) < places:
        return []
    out: dict[str, None] = {}
    for tie_break in product(*(permutations(g) for g in groups)):
        top = [h for group in tie_break for h in group][:places]
        for combination in _combinations_of(top, pool):
            out.setdefault(combination, None)
    return list(out)


# ---------------------------------------------------------------------------------------------
# Orderings
# ---------------------------------------------------------------------------------------------


def _check_p(p_win: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_win, dtype=float)
    if p.ndim != 1 or len(p) < 3:
        raise ValueError(f"p_win must be one race's probabilities for 3 or more runners: {p.shape}")
    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("every Win probability must be finite and positive")
    return p / p.sum()


def _check_params(family: str, params: Mapping[str, float] | None) -> dict[str, float] | None:
    if family in ("harville", "henery"):
        if params:
            raise ValueError(f"the {family} ordering takes no parameters, got {dict(params)}")
        return None
    if params is None:
        return None
    keys = QC_KEYS if family == "quadratic_context" else DH_KEYS
    if set(params) != set(keys):
        raise ValueError(f"{family} params must be exactly {keys}, got {sorted(params)}")
    out = {k: _real(params[k], k) for k in keys}
    if family == "discounted_harville" and any(v <= 0 for v in out.values()):
        raise ValueError(f"discounted Harville lambdas must be positive, got {out}")
    return out


def _masked_softmax(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """softmax of z over the last axis, zero where `mask` is True."""
    z = np.where(mask, -np.inf, z)
    top = z.max(axis=-1, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    w = np.exp(z - top)
    total = w.sum(axis=-1, keepdims=True)
    return np.divide(w, total, out=np.zeros_like(w), where=total > 0)


class Ordering:
    """The ordered first three of a race from its Win probabilities, for one ordering family.

    `params`: none for "harville" and "henery"; QC_KEYS for "quadratic_context"; DH_KEYS for
    "discounted_harville". `fold_params` is filled by `fit`: one params dict per fold.
    """

    def __init__(self, family: str, params: Mapping[str, float] | None = None) -> None:
        if family not in FAMILIES:
            raise ValueError(f"unknown ordering family {family!r}: expected one of {FAMILIES}")
        self.family = family
        self.params = _check_params(family, params)
        self.fold_params: list[dict[str, float]] = []

    def __repr__(self) -> str:
        return f"Ordering({self.family!r}, {self.params!r})"

    def at_fold(self, k: int) -> Ordering:
        """The ordering with fold k's fitted params."""
        return Ordering(self.family, self.fold_params[k] or None)

    # -- probabilities ------------------------------------------------------------------------

    def trifecta(self, p_win: Sequence[float]) -> np.ndarray:
        """P(i first, j second, k third) as an n x n x n array (zero where two indices meet)."""
        p = _check_p(p_win)
        if self.family == "henery":
            return _henery_trifecta(p)
        if self.family in ("quadratic_context", "discounted_harville") and self.params is None:
            raise ValueError(f"the {self.family} ordering has no params: pass them or fit it")
        lp = np.log(p)
        n = len(p)
        eye = np.eye(n, dtype=bool)
        if self.family == "harville":
            z2 = np.broadcast_to(lp, (n, n))
            z3 = np.broadcast_to(lp, (n, n, n))
        elif self.family == "discounted_harville":
            z2 = np.broadcast_to(self.params["lambda2"] * lp, (n, n))
            z3 = np.broadcast_to(self.params["lambda3"] * lp, (n, n, n))
        else:
            q = self.params
            h0, e0 = CONTEXT_REF
            entropy = float(-(p * lp).sum())
            slope2 = 1 + q["a2"] + q["h2"] * (lp - h0) + q["e2"] * (entropy - e0)
            z2 = slope2[:, None] * lp[None, :] + q["b2"] * (lp * lp)[None, :]
            placed = 0.5 * (lp[:, None] + lp[None, :])
            slope3 = 1 + q["a3"] + q["h3"] * (placed - h0) + q["e3"] * (entropy - e0)
            z3 = slope3[:, :, None] * lp[None, None, :] + q["b3"] * (lp * lp)[None, None, :]
        second = _masked_softmax(z2, eye)
        third = _masked_softmax(z3, eye[:, None, :] | eye[None, :, :])
        out = p[:, None, None] * second[:, :, None] * third
        idx = np.arange(n)
        out[idx, idx, :] = 0.0
        return out

    def joint_probs(self, p_win: Sequence[float], pool: str) -> dict[str, float]:
        """Every combination of `pool` and its probability; p_win[i] is horse number i + 1."""
        _check_pool(pool)
        t = self.trifecta(p_win)
        return pool_probs(t, pool)

    # -- fitting ------------------------------------------------------------------------------

    def fit(self, frame: pd.DataFrame, folds: Sequence) -> Ordering:
        """A copy whose `fold_params` holds one fit per fold, each on that fold's `train_idx`
        rows of `frame` alone (positions, as `core.eval.folds.walk_forward` gives them).

        `frame` has one row per runner: race_id, horse_no, p_win, finish_pos (1 = winner) and
        dead_heat. Races with a dead heat in the first three or without one horse each in
        first, second and third are left out of each fit (v0 ordering's rule). "harville" and
        "henery" have nothing to fit and get an empty dict per fold; "discounted_harville" is
        `models.v0.ordering.fit_ordering`; "quadratic_context" is the maximum-likelihood
        conditional logit of each stage, p renormalised over each race's runners.
        """
        fitted = Ordering(self.family, self.params)
        for fold in folds:
            rows = frame.iloc[np.asarray(fold.train_idx, dtype=int)]
            fitted.fold_params.append(_fit_family(self.family, rows))
        return fitted


def pool_probs(t: np.ndarray, pool: str) -> dict[str, float]:
    """A pool's combinations and probabilities from a trifecta array (index i = horse i + 1)."""
    _check_pool(pool)
    arr = pool_array(t, pool)
    n = t.shape[0]
    horses = range(1, n + 1)
    if pool == WIN or pool == PLACE:
        return {_key([i]): float(arr[i - 1]) for i in horses}
    if pool == EXACTA:
        return {_key(c): float(arr[c[0] - 1, c[1] - 1]) for c in permutations(horses, 2)}
    if pool in (QUINELLA, WIDE):
        return {_key(c): float(arr[c[0] - 1, c[1] - 1]) for c in combinations(horses, 2)}
    if pool == TRIO:
        return {_key(c): float(arr[c[0] - 1, c[1] - 1, c[2] - 1]) for c in combinations(horses, 3)}
    return {_key(c): float(t[c[0] - 1, c[1] - 1, c[2] - 1]) for c in permutations(horses, 3)}


def pool_array(t: np.ndarray, pool: str) -> np.ndarray:
    """A pool's probabilities as an array indexed by runner position: WIN and PLACE n, FORECAST
    [first, second], QUINELLA and QUINELLA_PLACE symmetric n x n, TRIO symmetric n x n x n,
    TIERCE the trifecta itself. Every one is a marginal of `t`."""
    _check_pool(pool)
    if pool == WIN:
        return t.sum(axis=(1, 2))
    if pool == PLACE:
        places = _places_paid(PLACE, t.shape[0])
        if places == 0:
            raise ValueError(f"PLACE is not sold to a field of {t.shape[0]}")
        positions = (t.sum(axis=(1, 2)), t.sum(axis=(0, 2)), t.sum(axis=(0, 1)))
        return sum(positions[:places])
    if pool == TRIFECTA:
        return t
    exacta = t.sum(axis=2)
    if pool == EXACTA:
        return exacta
    if pool == QUINELLA:
        return exacta + exacta.T
    trio = sum(np.transpose(t, axes) for axes in permutations(range(3)))
    if pool == TRIO:
        return trio
    return trio.sum(axis=2)  # WIDE: each pair with every third runner


# ---------------------------------------------------------------------------------------------
# Henery (1981): independent normal running times
# ---------------------------------------------------------------------------------------------


def _henery_grid(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = theta.min() - HENERY_PAD, theta.max() + HENERY_PAD
    t = lo + HENERY_STEP * np.arange(int(np.ceil((hi - lo) / HENERY_STEP)) + 1)
    z = t[None, :] - theta[:, None]
    return z, np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _henery_win(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(P(win), its Jacobian in theta) for running times N(theta_i, 1), the fastest winning."""
    z, f = _henery_grid(theta)
    log_s = log_ndtr(-z)
    log_all = log_s.sum(axis=0)
    win = HENERY_STEP * (f * np.exp(log_all[None, :] - log_s)).sum(axis=1)
    # d P_i / d theta_m = int f_i f_m prod_{l != i, m} S_l dt for m != i; each row sums to zero.
    jac = HENERY_STEP * np.einsum(
        "it,mt,imt->im",
        f,
        f,
        np.exp(log_all[None, None, :] - log_s[:, None, :] - log_s[None, :, :]),
    )
    np.fill_diagonal(jac, 0.0)
    np.fill_diagonal(jac, -jac.sum(axis=1))
    return win, jac


def _henery_means(p: np.ndarray) -> np.ndarray:
    """Running-time means (unit variance, mean zero) whose win probabilities are p."""
    theta = -0.6 * np.log(p)
    theta -= theta.mean()
    for _ in range(HENERY_MAX_ITER):
        win, jac = _henery_win(theta)
        resid = p - win
        if np.abs(resid).max() < HENERY_TOL:
            return theta
        step = np.linalg.lstsq(jac, resid, rcond=None)[0]
        theta = theta + step
        theta -= theta.mean()
    win, _ = _henery_win(theta)
    if np.abs(p - win).max() > 1e-8:
        raise ValueError("Henery means did not converge for these Win probabilities")
    return theta


def _henery_trifecta(p: np.ndarray) -> np.ndarray:
    """P(i, j, k) = int f_k(t) P(T_i < T_j < t) prod_{l not in ijk} S_l(t) dt, renormalised."""
    theta = _henery_means(p)
    n = len(p)
    z, f = _henery_grid(theta)
    log_s = log_ndtr(-z)
    # G[i, j, t] = P(T_i < T_j < t) = int_{-inf}^t f_j(s) F_i(s) ds
    g = cumulative_trapezoid(
        ndtr(z)[:, None, :] * f[None, :, :], dx=HENERY_STEP, axis=-1, initial=0.0
    )
    a = g * np.exp(-log_s[:, None, :] - log_s[None, :, :])
    b = f * np.exp(log_s.sum(axis=0)[None, :] - log_s)
    out = (a.reshape(n * n, -1) @ b.T * HENERY_STEP).reshape(n, n, n)
    idx = np.arange(n)
    out[idx, idx, :] = 0.0
    out[:, idx, idx] = 0.0
    out[idx, :, idx] = 0.0
    out = np.clip(out, 0.0, None)
    return out / out.sum()


# ---------------------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------------------


def _fit_family(family: str, rows: pd.DataFrame) -> dict[str, float]:
    if family in ("harville", "henery"):
        return {}
    if family == "discounted_harville":
        fit = v0_ordering.fit_ordering(rows, family="discounted_harville")
        return dict(zip(DH_KEYS, fit.lambdas, strict=True))
    return _fit_quadratic_context(rows)


def _fit_quadratic_context(rows: pd.DataFrame) -> dict[str, float]:
    races = v0_ordering._prepare(rows)
    present = races.present
    logp = np.where(present, races.logp, -np.inf)
    logp = logp - logsumexp(logp, axis=1, keepdims=True)  # p renormalised over the race
    lp = np.where(present, logp, 0.0)
    entropy = -(np.where(present, np.exp(lp) * lp, 0.0)).sum(axis=1)
    rows_idx = np.arange(len(lp))
    h0, e0 = CONTEXT_REF
    left = present.copy()
    left[rows_idx, races.chosen[:, 0]] = False
    out: dict[str, float] = {}
    for stage in (1, 2):
        chosen = races.chosen[:, stage]
        placed_mean = lp[rows_idx[:, None], races.chosen[:, :stage]].mean(axis=1)
        theta = _fit_stage(lp, left, chosen, placed_mean - h0, entropy - e0)
        out.update({f"{name}{stage + 1}": v for name, v in zip("abhe", theta, strict=True)})
        left[rows_idx, chosen] = False
    return {k: out[k] for k in QC_KEYS}


def _fit_stage(
    lp: np.ndarray, left: np.ndarray, chosen: np.ndarray, dh: np.ndarray, de: np.ndarray
) -> tuple[float, ...]:
    """Maximum likelihood (a, b, h, e) of one stage: a conditional logit over the runners left on
    (ln p, (ln p)^2, dh ln p, de ln p) with ln p itself as an offset (Harville at zero)."""
    rows = np.arange(len(lp))
    sq = lp * lp
    lp_c, sq_c = lp[rows, chosen], sq[rows, chosen]
    feats = (lp, sq, dh[:, None] * lp, de[:, None] * lp)
    feats_c = np.array([lp_c, sq_c, dh * lp_c, de * lp_c])

    def nll(theta: np.ndarray) -> tuple[float, np.ndarray]:
        a, b, h, e = theta
        slope = 1 + a + h * dh + e * de
        z = np.where(left, slope[:, None] * lp + b * sq, -np.inf)
        top = z.max(axis=1, keepdims=True)
        w = np.exp(z - top)
        total = w.sum(axis=1)
        z_c = slope * lp_c + b * sq_c
        value = np.mean(np.log(total) + top[:, 0] - z_c)
        prob = w / total[:, None]
        grad = np.array([np.mean((prob * x).sum(axis=1)) for x in feats]) - feats_c.mean(axis=1)
        return float(value), grad

    result = minimize(
        nll,
        x0=np.zeros(4),
        jac=True,
        bounds=QC_BOUNDS,
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000},
    )
    return tuple(float(np.clip(v, lo, hi)) for v, (lo, hi) in zip(result.x, QC_BOUNDS, strict=True))


# ---------------------------------------------------------------------------------------------
# The blend with the pool
# ---------------------------------------------------------------------------------------------


def pool_blend(p_model: pd.Series, q_pool: pd.Series, *, a: float, b: float) -> pd.Series:
    """exp(a ln p_model(c) + b ln q_pool(c)), normalised over one race's combinations.

    Both Series are indexed by combination string; q_pool is aligned to p_model by label and the
    result is on p_model's index, in its order. A combination of p_model missing from q_pool (no
    full board) is a ValueError, as is a probability that is not finite and positive.
    """
    a, b = _real(a, "a"), _real(b, "b")
    for name, s in (("p_model", p_model), ("q_pool", q_pool)):
        if not s.index.is_unique:
            raise ValueError(f"{name} has a combination twice")
    missing = p_model.index.difference(q_pool.index)
    if len(missing):
        raise ValueError(
            f"{len(missing)} combination(s) have no pool price (no full board), e.g. {missing[0]!r}"
        )
    p = p_model.to_numpy(dtype=float)
    q = q_pool.reindex(p_model.index).to_numpy(dtype=float)
    for name, arr in (("p_model", p), ("q_pool", q)):
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
            raise ValueError(f"every {name} probability must be finite and positive")
    u = a * np.log(p) + b * np.log(q)
    w = np.exp(u - u.max())
    return pd.Series(w / w.sum(), index=p_model.index, name="p_blend")
