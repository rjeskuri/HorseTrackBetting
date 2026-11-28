"""The C5 exotic layer: Win probabilities to exotic-pool probabilities, priced against the pools'
own dividends (ticket #99, epic #73; research 08 §3.2 C5, §4 E43, E44).

- `dividend_share` / `statute_shares` / `dividend_shares`: the dividend-implied share of a pool's
  money on each winning combination, the exotic market baseline of 08 R15. JRA pays each of a
  race-pool's P winning combinations (W + L / P) x R (04 §1.2, 競馬法 第8条: W the money on it,
  L the money on everything that lost, R the race's own payout rate, `takeout.race_takeout`,
  boosts included, #76); D, the dividend per 100 yen, is floored to 10 yen, so the 5-yen mean
  breakage is added back (04 §1.3). One winner (P = 1) holds R * 100 / (D + 5) of the pool,
  `dividend_share`; `statute_shares` solves P winners together; `dividend_shares` applies it to
  every (race_id, pool) of a dividends frame; `dividend_floor_bound` marks the race-pools where a
  dividend is the 110-yen floor rather than the formula.
- `winning_combinations`: the combinations a result pays, dead heats per 04 §1.4 (every
  combination some tie-break of the dead heats would make the result), Place's and Wide's places
  by the field when sales opened.
- `Ordering`: the ordered first three from one race's Win probabilities, a softmax over the
  runners left at each stage. Families: "harville"; "henery" (independent unit-variance normal
  running times, means solved so the Win probabilities hold, the trifecta by quadrature);
  "quadratic_context" (HKTrack research 16/19: slope 1 + a + h (mean ln p placed - h0) +
  e (H - e0), curvature b, per stage); "discounted_harville" (#115's v0 family, weights p^lambda,
  fitted by `models.v0.ordering.fit_ordering`). `.fit(frame, folds)` refits on each fold's
  training rows only.
- `pool_blend`: the log-linear pool of the model and the pool's own board, u_c = a ln p_model(c) +
  b ln q_pool(c), for races whose full final board is logged (08 R21), in a pool that pays one
  combination per result.

Pool codes are jra.yaml's. A combination is the store's `dividends.combination` string: horse
numbers joined by ",", in finishing order for FORECAST and TIERCE, ascending for QUINELLA,
QUINELLA_PLACE and TRIO; WIN and PLACE name one horse. BRACKET_QUINELLA's is the brackets of the
first two finishers, "x,y" with x <= y ("x,x" for two horses of one bracket, a combination only for
a bracket with two or more runners); it needs the card's horse -> bracket map, `brackets`, which
the other pools ignore (#513).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from functools import cache
from itertools import combinations, groupby, permutations, product
from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp, ndtr, ndtri

from horsetrack.core import config
from horsetrack.countries.japan import takeout
from horsetrack.countries.japan.models.v0 import ordering as v0_ordering

WIN, PLACE, QUINELLA, WIDE = "WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE"
EXACTA, TRIO, TRIFECTA = "FORECAST", "TRIO", "TIERCE"
BRACKET_QUINELLA = "BRACKET_QUINELLA"
POOLS = (WIN, PLACE, BRACKET_QUINELLA, QUINELLA, WIDE, EXACTA, TRIO, TRIFECTA)

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
HENERY_TOL = 1e-11  # |p - P(win)|, every runner
HENERY_RTOL = 1e-9  # |ln p - ln P(win)|, every runner: a long shot's own Win probability too
HENERY_MAX_ITER = 100
# The means solve: each step moves no mean more than HENERY_MAX_MOVE standard deviations, and the
# means never spread over HENERY_MAX_SPAN: from about 19.5 the trifecta's exp(-ln S_i - ln S_j)
# overflows at the grid's edge, and a 1e-60 long shot against 17 equals needs 15.8 (1e-80 is
# refused). So no grid is over HENERY_MAX_POINTS long; `_henery_grid` refuses one that would be.
HENERY_MAX_MOVE = 1.0
HENERY_MAX_SPAN = 18.0
HENERY_MAX_POINTS = int((HENERY_MAX_SPAN + 2 * HENERY_PAD) / HENERY_STEP) + 2

BLEND_SUM_TOL = 1e-6  # |sum p_model - 1| that `pool_blend` allows


# ---------------------------------------------------------------------------------------------
# Dividend-implied shares
# ---------------------------------------------------------------------------------------------


def dividend_share(dividend: float, payout_rate: float) -> float:
    """payout_rate * 100 / (dividend + 5): the pool's share on the one winning combination of a
    race-pool, paying `dividend` per 100 yen, mean breakage added back. ValueError for a dividend
    that is not a number of at least 100 (see `statute_shares`) or a payout rate outside (0, 1]."""
    d = _winning_dividend(dividend)
    r = _payout_rate(payout_rate)
    return r * UNIT_STAKE / (d + MEAN_BREAKAGE)


def statute_shares(dividends: Sequence[float], payout_rate: float) -> np.ndarray:
    """The money shares W_c / S of one race-pool's P winning combinations, in input order, from
    their dividends per 100 yen.

    A winning ticket on c gets (W_c + L / P) x R / W_c per yen (04 §1.2, 競馬法 第8条; S the
    pool, L = S - sum W the money that lost). With r_c = (D_c + 5) / (100 R), that is
    W_c / L = 1 / (P (r_c - 1)), and S / L = 1 + K, K the sum of those: x_c = 1 / ((1 + K) P
    (r_c - 1)). P = 1 is `dividend_share`. ValueError for an empty sequence, a dividend that is
    not a number of at least 100, or a payout rate outside (0, 1]. A dividend under 100 yen (the
    70-yen special payout when no ticket won) is not a share of winning money; from 100, the
    stake back, r_c >= 1.05 / R > 1, so somebody lost.
    """
    r = _payout_rate(payout_rate)
    d = np.array([_winning_dividend(x) for x in dividends], dtype=float)
    if not len(d):
        raise ValueError("a race-pool pays at least one winning combination, got none")
    excess = (d + MEAN_BREAKAGE) / (UNIT_STAKE * r) - 1.0
    if len(d) == 1:
        return np.array([r * UNIT_STAKE / (d[0] + MEAN_BREAKAGE)])  # `dividend_share`, exactly
    on_vs_lost = 1.0 / (len(d) * excess)  # W_c / L
    return on_vs_lost / (1.0 + on_vs_lost.sum())


def dividend_shares(
    dividends: pd.DataFrame, *, calendar: takeout.BoostCalendar | None = None
) -> pd.Series:
    """The money share of every row of a frame with the store's dividends columns (race_id, pool,
    combination, dividend): `statute_shares` over each (race_id, pool) group, P its row count,
    at that race's own payout rate, `takeout.race_takeout(race_id, pool)` with `calendar` (None:
    the shipped boost calendar). The frame must hold every winning combination of a race-pool,
    as the store's dividends table does: a missing one leaves P short and the others' shares
    too large. An optional `race_class_raw` column is passed on for the age-restricted boosts.
    A combination twice in one race-pool, or a race-pool with two conditions labels, is a
    ValueError. The Series is on the frame's index, in its order.

    A race-pool with a dividend under 100 yen (the 70-yen special payout when no ticket won) holds
    no share of winning money, so `statute_shares` refuses it: here every row of it is NaN, the
    rest of the frame is computed as without it, and `dividend_floor_bound` flags it.

    A dividend at or under its pool's minimum (110 yen, Plus 10; 100 when the pool cannot fund it)
    is the floor, not the formula: its share here misstates the money on it and, with P > 1,
    every co-winner's share too, through K. `dividend_floor_bound` marks those race-pools for the
    caller; how to treat them is ticket #459's."""
    groups = _race_pool_groups(dividends)
    labels = (
        [None if pd.isna(x) else str(x) for x in dividends["race_class_raw"]]
        if "race_class_raw" in dividends
        else [None] * len(dividends)
    )
    combos = dividends["combination"].tolist()
    values = dividends["dividend"].tolist()
    shares = np.empty(len(dividends), dtype=float)
    for (race_id, pool), rows in groups.items():
        if len({combos[i] for i in rows}) < len(rows):
            raise ValueError(f"{race_id} {pool}: a combination is given twice")
        label = {labels[i] for i in rows}
        if len(label) > 1:
            raise ValueError(f"{race_id} {pool}: more than one race_class_raw, {label!r}")
        rate = takeout.race_takeout(
            race_id, pool, calendar=calendar, race_class_raw=label.pop()
        ).payout_rate
        paid = [_dividend(values[i]) for i in rows]
        if min(paid) < UNIT_STAKE:
            shares[rows] = np.nan
            continue
        shares[rows] = statute_shares(paid, rate)
    return pd.Series(shares, index=dividends.index, dtype=float, name="share")


def dividend_floor_bound(dividends: pd.DataFrame) -> pd.Series:
    """True on every row of a (race_id, pool) group in which some dividend is at or under the
    pool's `min_dividend` (jra.yaml, 110 yen), False elsewhere: there JRA paid the floor, not the
    statute formula, so `dividend_shares` misstates the group's shares (see there). The frame
    needs `dividend_shares`' columns; a missing one, a pool jra.yaml does not list, or a dividend
    that is not a positive number is a ValueError. The Series is on the frame's index, in its
    order."""
    groups = _race_pool_groups(dividends)
    values = dividends["dividend"].tolist()
    pools = _jra_pools()
    flagged = np.zeros(len(dividends), dtype=bool)
    for (race_id, pool), rows in groups.items():
        if pool not in pools:
            raise ValueError(f"{race_id}: pool {pool!r} is not in jra.yaml")
        floor = pools[pool].min_dividend
        lowest = min(_dividend(values[i]) for i in rows)
        flagged[rows] = floor is not None and lowest <= floor
    return pd.Series(flagged, index=dividends.index, dtype=bool, name="floor_bound")


def _race_pool_groups(dividends: pd.DataFrame) -> dict[tuple, list[int]]:
    """The row positions of each (race_id, pool) of a dividends frame, in first-seen order."""
    missing = [c for c in ("race_id", "pool", "combination", "dividend") if c not in dividends]
    if missing:
        raise ValueError(f"the dividends frame is missing columns: {missing}")
    groups: dict[tuple, list[int]] = {}
    for i, key in enumerate(zip(dividends["race_id"], dividends["pool"], strict=True)):
        groups.setdefault(key, []).append(i)
    return groups


def _dividend(value: object) -> float:
    d = _real(value, "dividend")
    if not d > 0:
        raise ValueError(f"a dividend must be positive, got {value!r}")
    return d


def _winning_dividend(value: object) -> float:
    """A dividend that is a share of winning money: at least 100 yen per 100, the stake back."""
    d = _dividend(value)
    if d < UNIT_STAKE:
        raise ValueError(
            f"a dividend under {UNIT_STAKE:g} yen (the special payout when no ticket won) is not "
            f"a share of winning money, got {value!r}"
        )
    return d


def _payout_rate(value: object) -> float:
    r = _real(value, "payout_rate")
    if not 0 < r <= 1:
        raise ValueError(f"a payout rate must be in (0, 1], got {value!r}")
    return r


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


def _places_paid(pool: str, field_at_sale: int) -> int:
    """The first `places` finishers a pool's combinations are drawn from, 0 where it was not sold.
    Place's and Wide's by the runners when sales opened (jra.yaml `places_paid`: Place two from 5
    and three from 8, Wide three from 4), not by those that started; the others' are fixed."""
    if pool in (PLACE, WIDE):
        return _jra_pools()[pool].places_for(field_at_sale)
    return {WIN: 1, BRACKET_QUINELLA: 2, QUINELLA: 2, EXACTA: 2, TRIO: 3, TRIFECTA: 3}[pool]


def _field_size(field_at_sale: object, runners: int) -> int:
    """The runners when sales opened: `field_at_sale`, or `runners` (those given) when None.
    ValueError for one that is not a whole number or is smaller than `runners`."""
    if field_at_sale is None:
        return runners
    if isinstance(field_at_sale, bool) or not isinstance(field_at_sale, Integral):
        raise ValueError(f"field_at_sale must be a whole number, got {field_at_sale!r}")
    if field_at_sale < runners:
        raise ValueError(f"field_at_sale {field_at_sale} is under the {runners} runners given")
    return int(field_at_sale)


def _horse_numbers(horses: Sequence[int] | None, n: int) -> list[int]:
    """The horse number of each of n runners: `horses`, or 1..n when None. ValueError for a
    length that is not n, a number twice, or one that is not a positive whole number."""
    if horses is None:
        return list(range(1, n + 1))
    numbers = list(horses)
    if len(numbers) != n:
        raise ValueError(f"{len(numbers)} horse numbers for {n} runners")
    if any(isinstance(h, bool) or not isinstance(h, Integral) or h < 1 for h in numbers):
        raise ValueError(f"horse numbers must be positive whole numbers, got {numbers!r}")
    if len(set(numbers)) < n:
        raise ValueError(f"a horse number is given twice: {numbers!r}")
    return [int(h) for h in numbers]


def _position(value: object) -> int | None:
    """A finishing position: None for a runner that did not finish (None or NaN). ValueError for
    one that is not a whole number from 1."""
    if pd.isna(value):
        return None
    x = _real(value, "a finishing position")
    if x < 1 or not x.is_integer():
        raise ValueError(f"a finishing position must be a whole number from 1, got {value!r}")
    return int(x)


def _key(horses: Sequence[int]) -> str:
    return ",".join(str(int(h)) for h in horses)


def _bracket_number(value: object) -> int:
    """A bracket number. ValueError for one that is not a positive whole number (a numpy integer
    is one; a bool, a float, a string or None is not)."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"a bracket must be a positive whole number, got {value!r}")
    return int(value)


def _bracket_map(brackets: Mapping[int, int] | None, numbers: Collection[int]) -> dict[int, int]:
    """The bracket of each horse number in `numbers`, from the card's horse -> bracket map
    (entries for other horses are ignored). ValueError for no map, or a horse without a bracket
    or with one that is not a positive whole number."""
    if brackets is None:
        raise ValueError(f"{BRACKET_QUINELLA} needs the horse -> bracket map (`brackets`)")
    lookup = {int(h): b for h, b in brackets.items() if isinstance(h, Integral)}
    missing = sorted(set(numbers) - lookup.keys())
    if missing:
        raise ValueError(f"no bracket for horse numbers {missing}")
    return {h: _bracket_number(lookup[h]) for h in numbers}


def _combinations_of(
    top: Sequence[int], pool: str, bracket_of: Mapping[int, int] | None = None
) -> list[str]:
    """The combinations a strict finishing order `top` (first places only) pays in `pool`;
    `bracket_of` is each horse's bracket, for BRACKET_QUINELLA."""
    if pool == BRACKET_QUINELLA:
        assert bracket_of is not None
        return [_key(sorted(bracket_of[h] for h in top[:2]))]
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


def winning_combinations(
    finish: Mapping[int, int],
    pool: str,
    *,
    field_at_sale: int | None = None,
    brackets: Mapping[int, int] | None = None,
) -> list[str]:
    """The combinations `pool` pays for a result, each once.

    `finish` maps every starter's horse number to its finishing position (ties share a position,
    the next runner skips: 1, 2, 3, 3, 5; None or NaN for a runner that did not finish). A dead
    heat pays every combination some tie-break of it would make the result (04 §1.4): a dead heat
    for third pays both Trios and both Trifectas, and Wide pays every pair of a possible first
    three, never the two horses that dead-heated for third. PLACE and QUINELLA_PLACE are settled
    by `field_at_sale`, the runners when sales opened (jra.yaml; None: the runners in `finish`):
    PLACE pays the first two from 5 and three from 8, so a late scratch from eight to seven still
    pays three places, and QUINELLA_PLACE is sold from 4, so a field of 3 at sale pays no Wide
    however many finish; no other pool depends on it. A pool not offered to the field pays
    nothing. With fewer finishers than its places, a sold PLACE pays every finisher and a sold
    QUINELLA_PLACE every pair of finishers (none with one); the other pools pay nothing (TRIO and
    TIERCE need three finishers, QUINELLA, BRACKET_QUINELLA and FORECAST two). BRACKET_QUINELLA is
    QUINELLA with each horse replaced by its bracket, "x,y" with x <= y, so two horses of one
    bracket first and second pay "x,x"; `brackets` (horse number -> bracket) must hold every
    starter in `finish`, and the other pools ignore it. ValueError for an
    unknown pool, a position that is not a whole number from 1 (a whole float, 2.0, is one), a
    `field_at_sale` smaller than the runners in `finish`, or for BRACKET_QUINELLA, a missing map
    or a starter without a bracket.
    """
    _check_pool(pool)
    positions = {int(h): _position(pos) for h, pos in finish.items()}
    bracket_of = _bracket_map(brackets, positions) if pool == BRACKET_QUINELLA else None
    places = _places_paid(pool, _field_size(field_at_sale, len(finish)))
    if places == 0:
        return []
    placed = sorted((pos, h) for h, pos in positions.items() if pos is not None and pos <= places)
    if len(placed) < places:
        if pool not in (PLACE, WIDE):
            return []
        places = len(placed)  # every finisher places; Wide's pairs of them
    groups = [[h for _, h in g] for _, g in groupby(placed, key=lambda item: item[0])]
    out: dict[str, None] = {}
    for tie_break in product(*(permutations(g) for g in groups)):
        top = [h for group in tie_break for h in group][:places]
        for combination in _combinations_of(top, pool, bracket_of):
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

    def joint_probs(
        self,
        p_win: Sequence[float],
        pool: str,
        *,
        horses: Sequence[int] | None = None,
        field_at_sale: int | None = None,
        brackets: Mapping[int, int] | None = None,
    ) -> dict[str, float]:
        """Every combination of `pool` and its probability, keyed by horse number: p_win[i] is
        horse horses[i] (None: i + 1). `field_at_sale` fixes Place's places, as in `pool_array`;
        `brackets` is BRACKET_QUINELLA's horse -> bracket map, as in `pool_probs`.
        """
        _check_pool(pool)
        t = self.trifecta(p_win)
        return pool_probs(t, pool, horses=horses, field_at_sale=field_at_sale, brackets=brackets)

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


def pool_probs(
    t: np.ndarray,
    pool: str,
    *,
    horses: Sequence[int] | None = None,
    field_at_sale: int | None = None,
    brackets: Mapping[int, int] | None = None,
) -> dict[str, float]:
    """A pool's combinations and probabilities from a trifecta array whose index i is horse
    horses[i] (None: i + 1), keyed by horse number: ascending for QUINELLA, QUINELLA_PLACE and
    TRIO, in finishing order for FORECAST and TIERCE. `field_at_sale`: as in `pool_array`.
    BRACKET_QUINELLA is keyed by bracket, "x,y" with x <= y: every pair of brackets with a runner
    and "x,x" for each bracket with two or more; `brackets` maps horse number -> bracket and must
    hold every runner (entries for other horses are ignored), and the other pools ignore it.
    ValueError for horse numbers that are not one positive, distinct number per runner, and for
    BRACKET_QUINELLA, a missing map or a runner without a valid bracket."""
    _check_pool(pool)
    numbers = _horse_numbers(horses, t.shape[0])
    if pool == BRACKET_QUINELLA:
        bracket_of = _bracket_map(brackets, numbers)
        arr = pool_array(
            t, pool, field_at_sale=field_at_sale, brackets=[bracket_of[h] for h in numbers]
        )
        runners = sorted(Counter(bracket_of.values()).items())
        out = {}
        for a, (x, _) in enumerate(runners):
            for y, count in runners[a:]:
                if x != y or count >= 2:
                    out[_key([x, y])] = float(arr[x - 1, y - 1])
        return out
    arr = pool_array(t, pool, field_at_sale=field_at_sale)
    # Positions in horse-number order, so an unordered pool's combinations list ascending numbers.
    by_number = sorted(range(len(numbers)), key=numbers.__getitem__)
    if pool in (WIN, PLACE):
        return {_key([numbers[i]]): float(arr[i]) for i in by_number}
    size = 3 if pool in (TRIO, TRIFECTA) else 2
    choose = permutations if pool in (EXACTA, TRIFECTA) else combinations
    return {_key([numbers[i] for i in c]): float(arr[c]) for c in choose(by_number, size)}


def pool_array(
    t: np.ndarray,
    pool: str,
    *,
    field_at_sale: int | None = None,
    brackets: Sequence[int] | None = None,
) -> np.ndarray:
    """A pool's probabilities as an array indexed by runner position: WIN and PLACE n, FORECAST
    [first, second], QUINELLA and QUINELLA_PLACE symmetric n x n, TRIO symmetric n x n x n,
    TIERCE the trifecta itself. Every one is a marginal of `t`. PLACE pays the places of
    `field_at_sale`, the runners when sales opened (None: the n in `t`); ValueError for one
    smaller than n, in every pool, and for PLACE when it was not sold to that field.
    BRACKET_QUINELLA is the symmetric K x K array, K the highest bracket, index bracket - 1, from
    `brackets`, the bracket of each runner position: [x - 1, y - 1] and [y - 1, x - 1] each hold
    P(the first two are in brackets x and y) for x != y, and [x - 1, x - 1] holds P(both are in
    x), counted once, so the upper triangle sums to 1. ValueError for no `brackets` there, a
    length that is not n, or one that is not a positive whole number; other pools ignore it."""
    _check_pool(pool)
    field = _field_size(field_at_sale, t.shape[0])
    if pool == BRACKET_QUINELLA:
        if brackets is None:
            raise ValueError(f"{BRACKET_QUINELLA} needs the bracket of each runner (`brackets`)")
        by_position = [_bracket_number(b) for b in brackets]
        n = t.shape[0]
        if len(by_position) != n:
            raise ValueError(f"{len(by_position)} brackets for {n} runners")
        member = np.zeros((n, max(by_position)))
        member[np.arange(n), np.asarray(by_position) - 1] = 1.0
        ordered = member.T @ t.sum(axis=2) @ member  # [x, y]: first in x, second in y
        out = ordered + ordered.T
        idx = np.arange(len(out))
        out[idx, idx] = ordered[idx, idx]  # a same-bracket pair is one combination, not two
        return out
    if pool == WIN:
        return t.sum(axis=(1, 2))
    if pool == PLACE:
        places = _places_paid(PLACE, field)
        if places == 0:
            raise ValueError(f"PLACE is not sold to a field of {field}")
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
    """(z, f): t - theta_i and the normal density on the quadrature grid, HENERY_PAD beyond the
    means. ValueError for a theta that is not finite or a grid over HENERY_MAX_POINTS, before
    anything is allocated."""
    if not np.all(np.isfinite(theta)):
        raise ValueError("Henery means must be finite")
    lo, hi = theta.min() - HENERY_PAD, theta.max() + HENERY_PAD
    points = int(np.ceil((hi - lo) / HENERY_STEP)) + 1
    if points > HENERY_MAX_POINTS:
        raise ValueError(
            f"a Henery grid of {points} points is over the bound of {HENERY_MAX_POINTS}: the "
            f"means spread {hi - lo - 2 * HENERY_PAD:.3g} standard deviations"
        )
    t = lo + HENERY_STEP * np.arange(points)
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
    """Running-time means (unit variance, mean zero) whose win probabilities are p.

    Gauss-Newton on r = ln p - ln P(win), from theta = -Phi^-1(p): in logs a long shot's
    equation is as well scaled as the favourite's. Each step moves no mean more than
    HENERY_MAX_MOVE, keeps the means within HENERY_MAX_SPAN and is halved until |r| falls (a
    backtracking line search), so the solve cannot run off the grid (an undamped step on
    p - P(win) did, on ordinary 18-runner fields: the review of PR #461). Converged when every
    |p - P(win)| < HENERY_TOL and every |r| < HENERY_RTOL. When no step lowers |r| first (the
    quadrature's own floor), the means are kept if every |p - P(win)| <= 1e-8 and every
    |r| <= 1e-4; otherwise, and for a field that needs a wider spread, a ValueError.
    """
    log_p = np.log(p)
    theta = np.clip(-ndtri(p), -HENERY_MAX_SPAN, HENERY_MAX_SPAN)
    theta = np.minimum(theta, theta.min() + HENERY_MAX_SPAN)
    theta -= theta.mean()

    def residual(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        win, jac = _henery_win(theta)
        with np.errstate(divide="ignore"):
            return log_p - np.log(win), win, jac

    r, win, jac = residual(theta)
    for _ in range(HENERY_MAX_ITER):
        if np.abs(p - win).max() < HENERY_TOL and np.abs(r).max() < HENERY_RTOL:
            return theta
        if not np.all(np.isfinite(r)):
            break
        step = np.linalg.lstsq(jac / win[:, None], r, rcond=None)[0]
        step *= min(1.0, HENERY_MAX_MOVE / max(np.abs(step).max(), 1e-300))
        merit, alpha = float(r @ r), 1.0
        while alpha > 1e-6:
            trial = theta + alpha * step
            trial -= trial.mean()
            if np.ptp(trial) <= HENERY_MAX_SPAN:
                r_trial, win_trial, jac_trial = residual(trial)
                if float(r_trial @ r_trial) <= (1.0 - 1e-4 * alpha) * merit:  # False for NaN
                    break
            alpha *= 0.5
        else:
            break  # no step lowers the residual: at the quadrature's floor, or stuck
        theta, r, win, jac = trial, r_trial, win_trial, jac_trial
    if np.abs(p - win).max() <= 1e-8 and np.abs(r).max() <= 1e-4:
        return theta
    raise ValueError("Henery means did not converge for these Win probabilities")


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
    total = out.sum()
    if not (np.isfinite(total) and total > 0):
        raise ValueError("the Henery trifecta quadrature failed for these Win probabilities")
    return out / total


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


def _scratch_numbers(scratched: Collection[int]) -> set[int]:
    """The horse numbers of the confirmed scratches. ValueError for one that is not a positive
    whole number (a numpy integer is one; a bool, a float or a string is not)."""
    numbers = list(scratched)
    if any(isinstance(h, bool) or not isinstance(h, Integral) or h < 1 for h in numbers):
        raise ValueError(f"scratched horse numbers must be positive whole numbers, got {numbers!r}")
    return {int(h) for h in numbers}


def _holds_any(combination: str, numbers: set[int]) -> bool:
    """Whether a combination string holds one of the horse numbers, compared as numbers."""
    return not numbers.isdisjoint(int(h) for h in str(combination).split(","))


def pool_blend(
    p_model: pd.Series,
    q_pool: pd.Series,
    *,
    a: float,
    b: float,
    pool: str | None = None,
    scratched: Collection[int] = (),
) -> pd.Series:
    """exp(a ln p_model(c) + b ln q_pool(c)), normalised over one race's combinations.

    The log-linear pool of a pool that pays one combination per result (WIN, QUINELLA, FORECAST,
    TRIO, TIERCE), and it sums to 1. p_model is that pool's probabilities and must sum to 1
    within 1e-6 (BLEND_SUM_TOL); only q_pool is scale-invariant. `pool` names that pool; a
    single-winner code or None gives the same values. PLACE and QUINELLA_PLACE pay several
    combinations on every result, so their ticket chances sum to the places paid and this blend
    would rescale them to 1: either code is a ValueError, as is a p_model that does not sum to 1,
    which also refuses them under None or a single-winner code with the same keys (how to blend
    them is #459's registered variant). A code not in POOLS is a ValueError. The keys are not
    checked against `pool`.

    Both Series are indexed by combination string. `scratched` holds the horse numbers confirmed
    scratched or refunded, each a positive whole number (a ValueError otherwise): every q_pool
    row whose combination holds one of them, compared as numbers, is dropped first, null or
    priced, and a number with no row on the board drops nothing; p_model is never trimmed. Any
    other null q_pool price is a ValueError ("no pool price"), whether or not p_model has its
    combination: a live runner the board does not price never shrinks the field. Then both cover
    the same combinations: q_pool is aligned to p_model by label and the result is on p_model's
    index, in its order. A combination of p_model with no q_pool row (no full board, or a
    scratched runner the model still prices), or a priced one of q_pool missing from p_model (a
    runner the model lacks), is a ValueError, as is a probability that is not finite and positive.
    """
    a, b = _real(a, "a"), _real(b, "b")
    gone = _scratch_numbers(scratched)
    if pool is not None:
        _check_pool(pool)
        if pool in (PLACE, WIDE):
            raise ValueError(
                f"{pool} pays several combinations per result, so the log-linear pool would "
                "rescale its ticket chances to 1: how to blend Wide and Place is #459's "
                "registered variant"
            )
    for name, s in (("p_model", p_model), ("q_pool", q_pool)):
        if not s.index.is_unique:
            raise ValueError(f"{name} has a combination twice")
    if gone:  # a confirmed scratch's rows are refund slots, whatever price they show
        q_pool = q_pool[np.array([not _holds_any(c, gone) for c in q_pool.index], dtype=bool)]
    null = q_pool.index[q_pool.isna().to_numpy()]
    if len(null):
        raise ValueError(
            f"{len(null)} combination(s) on the board have no pool price (a null price on a "
            f"runner not confirmed scratched), e.g. {null[0]!r}"
        )
    missing = p_model.index.difference(q_pool.index)
    if len(missing):
        raise ValueError(
            f"{len(missing)} combination(s) have no pool price (no full board, or a scratched "
            f"runner p_model still prices), e.g. {missing[0]!r}"
        )
    unpriced = q_pool.index.difference(p_model.index)
    if len(unpriced):
        raise ValueError(
            f"{len(unpriced)} combination(s) on the board are not in p_model (a runner the model "
            f"lacks), e.g. {unpriced[0]!r}"
        )
    p = p_model.to_numpy(dtype=float)
    q = q_pool.reindex(p_model.index).to_numpy(dtype=float)
    for name, arr in (("p_model", p), ("q_pool", q)):
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
            raise ValueError(f"every {name} probability must be finite and positive")
    if abs(p.sum() - 1.0) > BLEND_SUM_TOL:
        raise ValueError(
            f"p_model sums to {p.sum():.9g}, not 1: the blend takes one pool's probabilities over "
            "its combinations (Place and Wide sum to the places paid; how to blend them is #459's "
            "registered variant)"
        )
    u = a * np.log(p) + b * np.log(q)
    w = np.exp(u - u.max())
    return pd.Series(w / w.sum(), index=p_model.index, name="p_blend")
