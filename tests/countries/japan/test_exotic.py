"""The C5 exotic layer against the pools' own dividends (ticket #99, epic #73; 08 §8 B7, E43, E44).

Contract pinned here (`horsetrack.countries.japan.models.exotic`):
- Pools are jra.yaml's codes: WIN, PLACE, QUINELLA, QUINELLA_PLACE (Wide), FORECAST (Exacta), TRIO,
  TIERCE (Trifecta). A combination is the store's `dividends.combination` string: horse numbers
  joined by ",", in finishing order for FORECAST and TIERCE, ascending by number for QUINELLA,
  QUINELLA_PLACE and TRIO ("2,11", not "11,2"); WIN and PLACE name one horse ("7").
- `dividend_share(dividend, payout_rate) -> float` = payout_rate * 100 / (dividend + 5): the share
  of the pool's money on the winning combination, implied by its dividend per 100 yen, with the
  5-yen mean breakage (dividends are floored to 10 yen) added back. A dividend that is not a
  positive number, or a payout rate outside (0, 1], is a ValueError.
- `statute_shares(dividends, payout_rate) -> np.ndarray` (owner answer 3(a), 2026-11-26): the
  money shares W_c / S of one race-pool's P winning combinations, in input order, from their
  dividends per 100 yen. JRA pays every winning combination (W + L / P) x R (04 §1.2, 競馬法 第8条;
  W the money on it, L the money on everything that lost, R the payout rate), so with
  r_c = (D_c + 5) / (100 R), K = sum_c 1 / (P (r_c - 1)): x_c = 1 / ((1 + K) P (r_c - 1)). With
  P = 1 it is `dividend_share`. An empty sequence, a dividend that is not a positive number, a
  payout rate outside (0, 1], or any r_c <= 1 is a ValueError.
- `dividend_shares(dividends) -> pd.Series`: the share of every row of a frame with the store's
  dividends columns race_id ("jra:<YYYY-MM-DD>:<VENUE>:<NN>"), pool, combination and dividend:
  `statute_shares` over each (race_id, pool) group, P = its row count (the frame holds every
  winning combination of a race-pool, as the store's dividends table does), at that race's own
  payout rate, `takeout.race_takeout(race_id, pool)` with the shipped boost calendar (#76) and
  the optional `race_class_raw` column passed on. A one-row group is `dividend_share`. The Series
  is on the frame's index, in its order.
- `winning_combinations(finish, pool) -> list[str]`: `finish` maps every starter's horse number to
  its finishing position, ties sharing a position and the next runner skipping (1, 2, 3, 3, 5).
  The winners are every combination a tie-break of the dead heats could make the result (04 §1.4):
  a dead heat for third pays both Trios and Trifectas, and Wide pays every pair of a possible first
  three, so the two horses that dead-heated for third are not a winning Wide pair. PLACE pays the
  first three here (every field below has 8 or more starters). Each combination appears once;
  an unknown pool is a ValueError.
- `Ordering(family, params=None)`, family "harville", "henery" or "quadratic_context"; any other
  family is a ValueError or NotImplementedError (at construction or at use).
  `.joint_probs(p_win, pool) -> dict[str, float]`: p_win is one race's Win probabilities, horse
  number = position + 1 (p_win[0] is horse 1); every combination of the pool is a key. TIERCE,
  FORECAST, QUINELLA and TRIO sum to 1 and QUINELLA_PLACE to 3, all derived from one ordered
  first three, whose first-place marginal is p_win. An unknown pool is a ValueError.
  - harville: P(i, j, k) = p_i * p_j / (1 - p_i) * p_k / (1 - p_i - p_j).
  - henery: Henery's (1981) independent-normal model fitted to p_win, or an approximation of it
    within the tolerance below.
  - quadratic_context (HKTrack research 16/19), params keys a2, b2, h2, e2 (the contest for
    second) and a3, b3, h3, e3 (third). The runner taking position s has weight
        exp((1 + a_s + h_s * (mean ln p of the runners already placed - h0) + e_s * (H - e0))
            * ln p + b_s * (ln p)^2)
    among the runners left, H = -sum p ln p, (h0, e0) a fixed centring of the implementation's
    choice. All eight at zero is exactly Harville. Only the a and b terms are pinned exactly here;
    h and e are pinned by what they move (stage 3 never moves the first two).
  `.fit(frame, folds) -> Ordering`: frame has one row per runner with race_id, horse_no, p_win,
  finish_pos (1 = winner) and dead_heat (other columns, e.g. a date, are ignored); folds are
  `core.eval.folds.Fold`s, positions into the frame's rows (from `walk_forward` over a per-row
  date). The result's `fold_params` is a list with one params dict per fold, in fold order, each
  the maximum-likelihood quadratic_context fit on that fold's `train_idx` rows alone.
- `pool_blend(p_model, q_pool, *, a, b) -> pd.Series`: one race's combinations (index labels =
  combination strings), u_c = a ln p_model(c) + b ln q_pool(c), returned as exp(u) normalised to
  sum to 1, on p_model's index in p_model's order (q_pool is aligned by label). A combination of
  p_model missing from q_pool (no full board) is a ValueError.

Expected values are derived independently: closed-form Harville, the stage weights written out
here, JRA's published payout rates with the boost calendar's rules, a quadrature value of Henery's
model, and hand-listed JRA dead-heat payouts. All data is synthetic and every RNG is seeded.
"""

from __future__ import annotations

import importlib
import math
from datetime import date, timedelta
from itertools import combinations, permutations

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.folds import FoldScheme, walk_forward

MODULE = "horsetrack.countries.japan.models.exotic"

WIN, PLACE, QUINELLA, WIDE = "WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE"
EXACTA, TRIO, TRIFECTA = "FORECAST", "TRIO", "TIERCE"
JOINT_POOLS = (EXACTA, QUINELLA, WIDE, TRIO, TRIFECTA)

FAMILIES = ("harville", "henery", "quadratic_context")
QC_KEYS = ("a2", "b2", "h2", "e2", "a3", "b3", "h3", "e3")
STAGE2_KEYS = ("a2", "b2", "h2", "e2")
STAGE3_KEYS = ("a3", "b3", "h3", "e3")
ZERO = dict.fromkeys(QC_KEYS, 0.0)
# Non-zero coefficients of the size HKTrack research 19 fitted (slope offsets below Harville's 1).
QC_PARAMS = {
    "a2": -0.3,
    "b2": -0.03,
    "h2": 0.05,
    "e2": -0.1,
    "a3": -0.6,
    "b3": -0.05,
    "h3": 0.03,
    "e3": -0.08,
}

P3 = np.array([0.5, 0.3, 0.2])
P5 = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
P8 = np.array([0.3, 0.2, 0.15, 0.12, 0.09, 0.07, 0.05, 0.02])
P6_FAV4 = np.array([0.05, 0.1, 0.15, 0.4, 0.2, 0.1])  # the favourite is horse 4
P10 = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.03, 0.02, 0.01])
FIELDS = {"P3": P3, "P5": P5, "P8": P8, "P6_FAV4": P6_FAV4}


def exotic():
    """The module under test, imported at call time: each test fails on its own until it exists."""
    return importlib.import_module(MODULE)


def make_ordering(family: str):
    return exotic().Ordering(family, dict(QC_PARAMS) if family == "quadratic_context" else None)


def key(*horses: int) -> str:
    return ",".join(str(h) for h in horses)


# ---------------------------------------------------------------------------------------------
# Independent references
# ---------------------------------------------------------------------------------------------


def stage_weight(p: np.ndarray, a: float = 0.0, b: float = 0.0) -> np.ndarray:
    """exp((1 + a) ln p + b (ln p)^2): the quadratic stage weight with no context term."""
    lp = np.log(p)
    return np.exp((1.0 + a) * lp + b * lp * lp)


def trifecta_reference(p, w2=None, w3=None) -> dict[str, float]:
    """P(i, j, k) = p_i * w2_j / sum_{l != i} w2_l * w3_k / sum_{l not in {i, j}} w3_l."""
    p = np.asarray(p, dtype=float)
    w2 = p if w2 is None else w2
    w3 = p if w3 is None else w3
    out = {}
    for i, j, k in permutations(range(len(p)), 3):
        second = w2[j] / (w2.sum() - w2[i])
        third = w3[k] / (w3.sum() - w3[i] - w3[j])
        out[key(i + 1, j + 1, k + 1)] = p[i] * second * third
    return out


def exacta_reference(p, w2=None) -> dict[str, float]:
    p = np.asarray(p, dtype=float)
    w2 = p if w2 is None else w2
    return {
        key(i + 1, j + 1): p[i] * w2[j] / (w2.sum() - w2[i])
        for i, j in permutations(range(len(p)), 2)
    }


def harville_closed_form(p) -> dict[str, float]:
    """Classic Harville written out: p_i * p_j / (1 - p_i) * p_k / (1 - p_i - p_j)."""
    return {
        key(i + 1, j + 1, k + 1): p[i] * p[j] / (1 - p[i]) * p[k] / (1 - p[i] - p[j])
        for i, j, k in permutations(range(len(p)), 3)
    }


def max_abs_diff(got: dict[str, float], ref: dict[str, float]) -> float:
    assert set(got) == set(ref)
    return max(abs(got[c] - ref[c]) for c in ref)


def kl(truth: dict[str, float], model: dict[str, float]) -> float:
    return sum(t * math.log(t / model[c]) for c, t in truth.items())


def horses(combination: str) -> list[int]:
    return [int(h) for h in combination.split(",")]


# ---------------------------------------------------------------------------------------------
# Dividend-implied shares: breakage and each race's own payout rate
# ---------------------------------------------------------------------------------------------


def test_share_from_dividend_adds_back_breakage():
    # The ticket's case: 75% paid out, a 1,000-yen dividend: 75 / 1,005, not 75 / 1,000.
    assert exotic().dividend_share(1000, 0.75) == pytest.approx(75 / 1005, rel=1e-12)
    assert round(exotic().dividend_share(1000, 0.75), 4) == 0.0746
    # Across rates and dividends, from the floor (110) to a long trifecta: R * 100 / (D + 5).
    table = [
        (0.80, 1000, 80 / 1005),
        (0.725, 110, 72.5 / 115),
        (0.775, 25670, 77.5 / 25675),
        (0.85, 4530, 85 / 4535),
        (0.75, 99990, 75 / 99995),
        (0.80, 160, 80 / 165),
    ]
    for rate, dividend, expected in table:
        assert exotic().dividend_share(dividend, rate) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("dividend", "rate"),
    [
        (0, 0.75),
        (-100, 0.75),
        (math.nan, 0.75),
        (1000, 0.0),
        (1000, -0.2),
        (1000, 1.2),
        (1000, math.nan),
    ],
)
def test_dividend_share_rejects_impossible_inputs(dividend, rate):
    with pytest.raises(ValueError):
        exotic().dividend_share(dividend, rate)


# (race id, pool, dividend, payout rate). The rates are JRA's published ones (WIN 80%, QUINELLA and
# QUINELLA_PLACE 77.5%, FORECAST and TRIO 75%, TIERCE 72.5%) with the shipped calendar's entries:
# super_premium sets 80%, premium adds 5%, both stack.
SHARE_ROWS = [
    ("jra:2025-10-26:TOK:11", TRIFECTA, 1000, 0.80),  # Super Premium, every venue
    ("jra:2025-10-19:TOK:11", TRIFECTA, 1000, 0.725),  # the Sunday before: no boost
    ("jra:2026-09-21:HSN:5", TRIO, 1000, 0.80),  # Super Premium at Hanshin only
    ("jra:2026-09-21:NAK:5", TRIO, 1000, 0.75),  # the same day at Nakayama: no boost
    ("jra:2023-12-28:NAK:11", WIDE, 2450, 0.85),  # Super Premium plus the Wide premium
    ("jra:2023-12-28:NAK:11", TRIO, 2450, 0.80),  # the same race's Trio: Super Premium only
    ("jra:2016-04-10:HSN:11", QUINELLA, 880, 0.825),  # premium Quinella (桜花賞)
    ("jra:2016-04-10:HSN:10", QUINELLA, 880, 0.775),  # the race before: no boost
    ("jra:2026-10-10:TOK:3", EXACTA, 13570, 0.80),  # Super Premium
    ("jra:2026-10-11:TOK:3", EXACTA, 13570, 0.75),  # the next day: no boost
]


def test_boosted_days_use_their_own_takeout():
    combination = {TRIFECTA: "3,1,7", TRIO: "1,3,7", WIDE: "3,7", QUINELLA: "1,3", EXACTA: "3,1"}
    frame = pd.DataFrame(
        {
            "race_id": [r[0] for r in SHARE_ROWS],
            "pool": [r[1] for r in SHARE_ROWS],
            "combination": [combination[r[1]] for r in SHARE_ROWS],
            "dividend": [r[2] for r in SHARE_ROWS],
        },
        index=pd.Index([f"d{i}" for i in range(len(SHARE_ROWS))][::-1]),
    )
    shares = exotic().dividend_shares(frame)
    assert isinstance(shares, pd.Series)
    assert list(shares.index) == list(frame.index)
    expected = [rate * 100 / (dividend + 5) for _, _, dividend, rate in SHARE_ROWS]
    np.testing.assert_allclose(shares.to_numpy(dtype=float), expected, rtol=1e-12)
    # The ticket's case: the same 1,000-yen Trifecta pays 80 / 1,005 on the Super Premium day and
    # 72.5 / 1,005 a week earlier.
    assert shares.iloc[0] == pytest.approx(80 / 1005, rel=1e-12)
    assert shares.iloc[1] == pytest.approx(72.5 / 1005, rel=1e-12)


# ---------------------------------------------------------------------------------------------
# Several winning combinations: the statute's split (04 §1.2, owner answer 3(a))
# ---------------------------------------------------------------------------------------------

UNIT = 100.0  # yen: dividends are quoted per 100 yen
BREAKAGE = 5.0  # yen: the mean breakage a dividend floored to 10 yen loses


def statute_dividends(money: dict[str, float], pool_total: float, rate: float) -> dict[str, float]:
    """The unrounded dividend per 100 yen of each winning combination, built from the pool's money
    under 競馬法 第8条 (04 §1.2): a winning ticket on combination c gets (W_c + L / P) x R / W_c per
    yen, W_c the money on c, L the money on everything that lost, P the number of winning
    combinations; less the 5-yen mean breakage that the shares add back."""
    lost = pool_total - sum(money.values())
    assert lost > 0
    n_winners = len(money)
    paid = {c: UNIT * rate * (w + lost / n_winners) / w for c, w in money.items()}
    # The fixture's own check: the winning tickets take exactly R of the whole pool.
    total_paid = sum(money[c] * paid[c] / UNIT for c in money)
    assert total_paid == pytest.approx(rate * pool_total, rel=1e-12)
    return {c: p - BREAKAGE for c, p in paid.items()}


WIDE_MONEY = {
    "1,2": 9_000_000,
    "2,4": 1_200_000,
    "1,3": 6_000_000,
    "2,3": 2_000_000,
    "1,4": 3_500_000,
}
TRIO_MONEY = {"1,2,4": 800_000, "1,2,3": 3_000_000}

# (pool, payout rate, yen on each winning combination in input order, the whole pool in yen). The
# rates are JRA's published ones; 0.80 on TRIO is a Super Premium day, 0.85 on Wide a Super
# Premium day with the Wide premium on top.
DEAD_HEATS = {
    "trio_dead_heat_for_third": (TRIO, 0.75, TRIO_MONEY, 100_000_000),
    "trio_dead_heat_for_third_super_premium": (TRIO, 0.80, TRIO_MONEY, 100_000_000),
    "wide_dead_heat_for_third": (WIDE, 0.775, WIDE_MONEY, 60_000_000),
    "wide_dead_heat_for_third_ultra_premium": (WIDE, 0.85, WIDE_MONEY, 60_000_000),
    "place_dead_heat_for_third": (
        PLACE,
        0.80,
        {"7": 5_000_000, "3": 9_000_000, "11": 1_500_000, "5": 3_000_000},
        40_000_000,
    ),
    "place_no_dead_heat": (
        PLACE,
        0.80,
        {"3": 9_000_000, "7": 5_000_000, "5": 3_000_000},
        40_000_000,
    ),
    "win_dead_heat": (WIN, 0.80, {"12": 25_000_000, "4": 10_000_000}, 100_000_000),
    "exacta_dead_heat_for_first": (EXACTA, 0.75, {"4,12": 1_100_000, "12,4": 400_000}, 50_000_000),
    "trifecta_three_way_dead_heat_for_first": (
        TRIFECTA,
        0.725,
        {
            "2,1,3": 300_000,
            "1,2,3": 120_000,
            "1,3,2": 90_000,
            "3,1,2": 60_000,
            "2,3,1": 45_000,
            "3,2,1": 30_000,
        },
        200_000_000,
    ),
}


@pytest.mark.parametrize("case", list(DEAD_HEATS))
def test_statute_shares_recover_the_money_on_every_winning_combination(case):
    pool, rate, money, total = DEAD_HEATS[case]
    dividends = statute_dividends(money, total, rate)
    got = exotic().statute_shares([dividends[c] for c in money], rate)
    assert isinstance(got, np.ndarray)
    assert got.shape == (len(money),)
    expected = np.array([w / total for w in money.values()])
    np.testing.assert_allclose(got, expected, rtol=1e-10)
    # The one-winner share R * 100 / (D + 5) overstates every one of them by 40% or more: a dead
    # heat's (or Place's, or Wide's) winners split the losers' money P ways.
    single = np.array([exotic().dividend_share(dividends[c], rate) for c in money])
    assert np.all(single > 1.4 * expected), (pool, single / expected)


def test_statute_shares_of_one_winner_are_the_single_winner_share():
    # P = 1: the pool's money on the winner is R * 100 / (D + 5) of it, `dividend_share` exactly.
    table = [
        (1000, 0.75, 75 / 1005),
        (1000, 0.80, 80 / 1005),
        (110, 0.725, 72.5 / 115),
        (25670, 0.775, 77.5 / 25675),
        (4530, 0.85, 85 / 4535),
        (99990, 0.75, 75 / 99995),
        (160, 0.80, 80 / 165),
        (1000, 1.0, 100 / 1005),  # a payout rate of 1 is allowed
    ]
    for dividend, rate, expected in table:
        got = exotic().statute_shares((dividend,), rate)
        assert isinstance(got, np.ndarray)
        assert got.shape == (1,)
        assert got[0] == pytest.approx(expected, rel=1e-12)
        assert got[0] == pytest.approx(exotic().dividend_share(dividend, rate), rel=1e-12)


@pytest.mark.parametrize(
    ("dividends", "rate"),
    [
        ([], 0.75),  # no winning combination
        ([0], 0.75),
        ([1270, -100], 0.75),
        ([1270, math.nan], 0.75),
        ([None, 4580], 0.75),
        ([1270, 4580], 0.0),
        ([1270, 4580], -0.2),
        ([1270, 4580], 1.2),
        ([1270, 4580], math.nan),
        ([70], 0.75),  # (70 + 5) / 75 = 1: R per yen back means nobody lost
        ([60], 0.75),  # under R per yen back
        ([1270, 70], 0.75),  # one of two winners at r = 1
        ([4580, 1270, 50], 0.80),  # one of three under
        ([95], 1.0),  # r = 1 at a payout rate of 1
    ],
    ids=lambda v: repr(v).replace(" ", ""),
)
def test_statute_shares_rejects_impossible_inputs(dividends, rate):
    with pytest.raises(ValueError):
        exotic().statute_shares(dividends, rate)


# (race id, pool, payout rate, race_class_raw, yen on each winning combination, the whole pool).
# The rates follow JRA's published ones with the shipped calendar: 80% on a Super Premium day, 85%
# with a premium on top (Wide on 2023-12-28; Win in a 2-year-old race in the 2016 summer, which
# the race's conditions label alone tells apart from the 3-year-old-and-up race that day).
DEAD_HEAT_GROUPS = [
    ("jra:2026-09-21:HSN:5", TRIO, 0.80, None, TRIO_MONEY, 100_000_000),
    ("jra:2026-09-21:NAK:5", TRIO, 0.75, None, TRIO_MONEY, 100_000_000),
    ("jra:2023-12-28:NAK:11", WIDE, 0.85, None, WIDE_MONEY, 60_000_000),
    (
        "jra:2023-12-28:NAK:11",
        TRIO,
        0.80,
        None,
        {"1,2,3": 2_400_000, "1,2,4": 1_600_000},
        90_000_000,
    ),
    ("jra:2026-10-10:TOK:3", EXACTA, 0.80, None, {"4,12": 1_100_000, "12,4": 400_000}, 50_000_000),
    (
        "jra:2016-07-02:FKS:5",
        WIN,
        0.85,
        "2歳新馬",
        {"12": 25_000_000, "4": 10_000_000},
        100_000_000,
    ),
    (
        "jra:2016-07-02:FKS:7",
        WIN,
        0.80,
        "3歳以上500万下",
        {"12": 25_000_000, "4": 10_000_000},
        100_000_000,
    ),
    (
        "jra:2026-10-11:TOK:3",
        PLACE,
        0.80,
        "3歳以上1勝クラス",
        {"3": 9_000_000, "7": 5_000_000, "5": 3_000_000},
        40_000_000,
    ),
]
# (race id, pool, payout rate, race_class_raw, combination, dividend): one winning combination.
SINGLE_WINNER_GROUPS = [
    ("jra:2025-10-26:TOK:11", TRIFECTA, 0.80, None, "3,1,7", 25670),
    ("jra:2025-10-19:TOK:11", TRIFECTA, 0.725, None, "3,1,7", 25670),
    ("jra:2016-04-10:HSN:11", QUINELLA, 0.825, None, "1,3", 880),
]
DIVIDEND_COLUMNS = ["race_id", "pool", "combination", "dividend", "race_class_raw"]


def test_dividend_shares_split_each_race_pool_among_its_winning_combinations():
    rows = []
    for race_id, pool, rate, label, money, total in DEAD_HEAT_GROUPS:
        dividends = statute_dividends(money, total, rate)
        for c, w in money.items():
            rows.append((race_id, pool, c, dividends[c], label, w / total, rate, False))
    for race_id, pool, rate, label, c, dividend in SINGLE_WINNER_GROUPS:
        rows.append((race_id, pool, c, dividend, label, rate * 100 / (dividend + 5), rate, True))
    frame = pd.DataFrame(rows, columns=[*DIVIDEND_COLUMNS, "expected", "rate", "single"])
    frame.index = pd.Index([f"div{n:02d}" for n in range(len(frame))])
    frame = frame.iloc[np.random.default_rng(99).permutation(len(frame))]
    # The fixture's own check: the shuffle splits race-pools, so no group is a run of rows.
    group_key = frame["race_id"] + "|" + frame["pool"]
    n_groups = len(DEAD_HEAT_GROUPS) + len(SINGLE_WINNER_GROUPS)
    assert (group_key != group_key.shift()).sum() > n_groups + 5

    shares = exotic().dividend_shares(frame[DIVIDEND_COLUMNS])
    assert isinstance(shares, pd.Series)
    assert list(shares.index) == list(frame.index)
    # Each race-pool's rows recover the money on them, at that race's own rate: the same money
    # pays other dividends on a boosted day and gives the same shares back; one race's Wide and
    # Trio, or one day's two Trios, are separate splits.
    np.testing.assert_allclose(shares.to_numpy(dtype=float), frame["expected"], rtol=1e-10)
    # One-row groups keep today's value exactly.
    single = frame[frame["single"]]
    np.testing.assert_allclose(
        shares[single.index].to_numpy(dtype=float),
        [
            exotic().dividend_share(d, r)
            for d, r in zip(single["dividend"], single["rate"], strict=True)
        ],
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------------------------
# Winning combinations and dead heats
# ---------------------------------------------------------------------------------------------


def settle(finish: dict[int, int], pool: str) -> list[str]:
    got = exotic().winning_combinations(finish, pool)
    assert len(got) == len(set(got)), f"{pool}: a combination is listed twice: {got}"
    return sorted(got)


def test_dead_heat_combinations_settle_correctly():
    # Horse 1 first, horse 2 second, 3 and 4 dead-heat for third, ten starters.
    finish = {1: 1, 2: 2, 3: 3, 4: 3, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}
    assert settle(finish, WIDE) == sorted(["1,2", "1,3", "1,4", "2,3", "2,4"])  # 3-4 does not win
    assert settle(finish, TRIO) == sorted(["1,2,3", "1,2,4"])
    assert settle(finish, TRIFECTA) == sorted(["1,2,3", "1,2,4"])
    assert settle(finish, PLACE) == sorted(["1", "2", "3", "4"])
    assert settle(finish, WIN) == ["1"]
    assert settle(finish, QUINELLA) == ["1,2"]
    assert settle(finish, EXACTA) == ["1,2"]


def _field(top: dict[int, int], starters: int) -> dict[int, int]:
    """`top` plus the other starters in number order behind it, positions continuing after it."""
    finish = dict(top)
    rest = [h for h in range(1, starters + 1) if h not in finish]
    for pos, h in enumerate(rest, start=len(top) + 1):
        finish[h] = pos
    return finish


SETTLEMENTS = {
    # No tie, two-digit numbers in the first three: unordered pools sort by number.
    "no_tie": (
        _field({11: 1, 2: 2, 9: 3}, 12),
        {
            WIN: ["11"],
            PLACE: ["11", "2", "9"],
            QUINELLA: ["2,11"],
            WIDE: ["2,9", "2,11", "9,11"],
            EXACTA: ["11,2"],
            TRIO: ["2,9,11"],
            TRIFECTA: ["11,2,9"],
        },
    ),
    "dead_heat_first": (
        _field({12: 1, 4: 1, 7: 3}, 12),
        {
            WIN: ["12", "4"],
            PLACE: ["12", "4", "7"],
            QUINELLA: ["4,12"],
            WIDE: ["4,7", "4,12", "7,12"],
            EXACTA: ["4,12", "12,4"],
            TRIO: ["4,7,12"],
            TRIFECTA: ["4,12,7", "12,4,7"],
        },
    ),
    "three_way_dead_heat_first": (
        _field({1: 1, 2: 1, 3: 1}, 9),
        {
            WIN: ["1", "2", "3"],
            PLACE: ["1", "2", "3"],
            QUINELLA: ["1,2", "1,3", "2,3"],
            WIDE: ["1,2", "1,3", "2,3"],
            EXACTA: [key(*o) for o in permutations((1, 2, 3), 2)],
            TRIO: ["1,2,3"],
            TRIFECTA: [key(*o) for o in permutations((1, 2, 3))],
        },
    ),
    "dead_heat_second": (
        _field({1: 1, 2: 2, 3: 2}, 9),
        {
            WIN: ["1"],
            PLACE: ["1", "2", "3"],
            QUINELLA: ["1,2", "1,3"],
            WIDE: ["1,2", "1,3", "2,3"],
            EXACTA: ["1,2", "1,3"],
            TRIO: ["1,2,3"],
            TRIFECTA: ["1,2,3", "1,3,2"],
        },
    ),
    "three_way_dead_heat_third": (
        _field({1: 1, 2: 2, 3: 3, 4: 3, 5: 3}, 10),
        {
            WIN: ["1"],
            PLACE: ["1", "2", "3", "4", "5"],
            QUINELLA: ["1,2"],
            WIDE: ["1,2", "1,3", "1,4", "1,5", "2,3", "2,4", "2,5"],
            EXACTA: ["1,2"],
            TRIO: ["1,2,3", "1,2,4", "1,2,5"],
            TRIFECTA: ["1,2,3", "1,2,4", "1,2,5"],
        },
    ),
    "dead_heats_for_first_and_third": (
        _field({1: 1, 2: 1, 3: 3, 4: 3}, 10),
        {
            WIN: ["1", "2"],
            PLACE: ["1", "2", "3", "4"],
            QUINELLA: ["1,2"],
            WIDE: ["1,2", "1,3", "1,4", "2,3", "2,4"],
            EXACTA: ["1,2", "2,1"],
            TRIO: ["1,2,3", "1,2,4"],
            TRIFECTA: ["1,2,3", "1,2,4", "2,1,3", "2,1,4"],
        },
    ),
}


@pytest.mark.parametrize("case", list(SETTLEMENTS))
def test_winning_combinations_per_pool(case):
    finish, expected = SETTLEMENTS[case]
    for pool, winners in expected.items():
        assert settle(finish, pool) == sorted(winners), pool


def test_winning_combinations_rejects_an_unknown_pool():
    with pytest.raises(ValueError):
        exotic().winning_combinations(_field({1: 1, 2: 2, 3: 3}, 8), "SUPERFECTA")


# ---------------------------------------------------------------------------------------------
# Joint probabilities
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("field", list(FIELDS))
def test_ordered_probabilities_sum_to_one_over_all_trifectas(family, field):
    p = FIELDS[field]
    n = len(p)
    probs = make_ordering(family).joint_probs(p, TRIFECTA)
    assert set(probs) == {key(i + 1, j + 1, k + 1) for i, j, k in permutations(range(n), 3)}
    values = np.array(list(probs.values()), dtype=float)
    assert np.all(np.isfinite(values)) and np.all(values >= 0)
    assert values.sum() == pytest.approx(1.0, abs=1e-6 if family == "henery" else 1e-9)
    first = np.zeros(n)
    for combination, value in probs.items():
        first[horses(combination)[0] - 1] += value
    np.testing.assert_allclose(first, p, atol=1e-3 if family == "henery" else 1e-9)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("field", list(FIELDS))
def test_every_pool_is_a_marginal_of_the_trifectas(family, field):
    p = FIELDS[field]
    n = len(p)
    ordering = make_ordering(family)
    got = {pool: ordering.joint_probs(p, pool) for pool in JOINT_POOLS}
    t = got[TRIFECTA]
    exacta, quinella, trio, wide = {}, {}, {}, {}
    for combination, value in t.items():
        i, j, k = horses(combination)
        exacta[key(i, j)] = exacta.get(key(i, j), 0.0) + value
        quinella[key(*sorted((i, j)))] = quinella.get(key(*sorted((i, j))), 0.0) + value
        trio[key(*sorted((i, j, k)))] = trio.get(key(*sorted((i, j, k))), 0.0) + value
        for pair in combinations(sorted((i, j, k)), 2):
            wide[key(*pair)] = wide.get(key(*pair), 0.0) + value
    assert set(exacta) == {key(i + 1, j + 1) for i, j in permutations(range(n), 2)}
    assert set(trio) == {key(*c) for c in combinations(range(1, n + 1), 3)}
    assert set(wide) == set(quinella) == {key(*c) for c in combinations(range(1, n + 1), 2)}
    for pool, ref in ((EXACTA, exacta), (QUINELLA, quinella), (TRIO, trio), (WIDE, wide)):
        assert max_abs_diff(got[pool], ref) < 1e-9, pool
    for pool, total in ((EXACTA, 1.0), (QUINELLA, 1.0), (TRIO, 1.0), (WIDE, 3.0)):
        assert sum(got[pool].values()) == pytest.approx(total, abs=1e-6), pool


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("n", [3, 6, 9])
def test_equal_chances_give_equal_combinations(family, n):
    p = np.full(n, 1.0 / n)
    probs = make_ordering(family).joint_probs(p, TRIFECTA)
    atol = 1e-6 if family == "henery" else 1e-9
    np.testing.assert_allclose(list(probs.values()), 1.0 / (n * (n - 1) * (n - 2)), atol=atol)


def test_quadratic_context_reduces_to_harville_at_zero_coefficients():
    p = P8
    harville_t = harville_closed_form(p)
    harville_e = exacta_reference(p)
    zero = exotic().Ordering("quadratic_context", dict(ZERO))
    assert max_abs_diff(zero.joint_probs(p, TRIFECTA), harville_t) < 1e-12
    assert max_abs_diff(zero.joint_probs(p, EXACTA), harville_e) < 1e-12
    assert max_abs_diff(exotic().Ordering("harville").joint_probs(p, TRIFECTA), harville_t) < 1e-12
    # Every coefficient moves the result once it is not zero; the third-place terms never move
    # the first two.
    for name in QC_KEYS:
        moved = exotic().Ordering("quadratic_context", {**ZERO, name: 0.25})
        assert max_abs_diff(moved.joint_probs(p, TRIFECTA), harville_t) > 1e-4, name
        exacta_shift = max_abs_diff(moved.joint_probs(p, EXACTA), harville_e)
        if name in STAGE3_KEYS:
            assert exacta_shift < 1e-12, name
        else:
            assert exacta_shift > 1e-4, name


@pytest.mark.parametrize(
    "coefs",
    [
        {"a2": -0.4},
        {"a2": 0.5},
        {"b2": -0.1},
        {"a3": -0.5},
        {"b3": 0.08},
        {"a2": -0.3, "b2": -0.05, "a3": -0.6, "b3": -0.04},
    ],
    ids=lambda c: "_".join(f"{k}={v}" for k, v in c.items()),
)
def test_quadratic_slope_and_curvature_follow_the_stage_weights(coefs):
    p = P8
    params = {**ZERO, **coefs}
    w2 = stage_weight(p, params["a2"], params["b2"])
    w3 = stage_weight(p, params["a3"], params["b3"])
    got = exotic().Ordering("quadratic_context", params).joint_probs(p, TRIFECTA)
    assert max_abs_diff(got, trifecta_reference(p, w2, w3)) < 1e-12


def test_henery_discounts_the_favourite_for_second():
    # Henery's normal model for this field puts the favourite second with probability 0.2407
    # (quadrature, offline); Harville says 0.2906, a discounted Harville with lambda 0.86 0.266.
    p = np.array([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
    exacta = exotic().Ordering("henery").joint_probs(p, EXACTA)
    favourite_second = sum(v for c, v in exacta.items() if horses(c)[1] == 1)
    harville = sum(p[i] * p[0] / (1 - p[i]) for i in range(1, len(p)))
    assert harville == pytest.approx(0.2906, abs=1e-4)
    assert favourite_second == pytest.approx(0.2407, abs=0.01)


def test_ordering_rejects_an_unknown_family_or_pool():
    with pytest.raises((ValueError, NotImplementedError)):
        exotic().Ordering("plackett_luce").joint_probs(P5, TRIFECTA)
    with pytest.raises(ValueError):
        exotic().Ordering("harville").joint_probs(P5, "SUPERFECTA")


# ---------------------------------------------------------------------------------------------
# Fitting on folds
# ---------------------------------------------------------------------------------------------


def draw_finish(
    rng: np.random.Generator, p: np.ndarray, lambdas: tuple[float, float]
) -> np.ndarray:
    """Finishing positions: the first three drawn from a discounted Harville model (Gumbel-max per
    stage, weight p^lambda), the rest in random order behind them."""
    n = len(p)
    logp = np.log(p)
    taken = np.zeros(n, dtype=bool)
    sort_key = rng.random(n) + 3.0
    for stage, lam in enumerate((1.0, *lambdas)):
        score = lam * logp + rng.gumbel(size=n)
        score[taken] = -np.inf
        pick = int(score.argmax())
        taken[pick] = True
        sort_key[pick] = stage
    finish = np.empty(n, dtype=int)
    finish[np.argsort(sort_key)] = np.arange(1, n + 1)
    return finish


def simulate_frame(rng, days, races_per_day, n_runners, lambdas_on) -> pd.DataFrame:
    """One row per runner (race_id, date, horse_no, p_win, finish_pos, dead_heat), rows shuffled."""
    parts = []
    race = 0
    for day in days:
        for _ in range(races_per_day):
            logits = rng.normal(0.0, 1.0, n_runners)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            parts.append(
                pd.DataFrame(
                    {
                        "race_id": f"S{race:05d}",
                        "date": day,
                        "horse_no": np.arange(1, n_runners + 1),
                        "p_win": p,
                        "finish_pos": draw_finish(rng, p, lambdas_on(day)),
                        "dead_heat": False,
                    }
                )
            )
            race += 1
    frame = pd.concat(parts, ignore_index=True)
    return frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)


def scramble(frame: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """The races of `rows` rewritten: p_win flattened (square root, renormalised) and the finish
    reversed, longest shot first."""
    out = frame.copy()
    selected = out.iloc[rows]
    for _, race in selected.groupby("race_id"):
        p = np.sqrt(race["p_win"].to_numpy(dtype=float))
        p /= p.sum()
        out.loc[race.index, "p_win"] = p
        out.loc[race.index, "finish_pos"] = np.argsort(np.argsort(p)) + 1
    return out


def vec(params: dict) -> np.ndarray:
    return np.array([params[k] for k in QC_KEYS], dtype=float)


def fold_vectors(frame, folds) -> list[np.ndarray]:
    fitted = exotic().Ordering("quadratic_context").fit(frame, folds)
    params = list(fitted.fold_params)
    assert len(params) == len(folds)
    vectors = [vec(p) for p in params]
    assert all(np.all(np.isfinite(v)) for v in vectors)
    return vectors


def saturdays(first: date, last: date) -> list[date]:
    return [first + timedelta(weeks=w) for w in range((last - first).days // 7 + 1)]


def test_ordering_refits_on_inner_folds():
    rng = np.random.default_rng(99)
    # Strong favourites until the autumn of 2020, a deep discount after: folds that train on
    # more of the later races should fit other coefficients.
    days = saturdays(date(2020, 1, 4), date(2021, 6, 26))
    frame = simulate_frame(
        rng, days, 4, 10, lambda d: (1.3, 1.3) if d < date(2020, 10, 1) else (0.35, 0.3)
    )
    scheme = FoldScheme(
        unit="quarter",
        test_length=1,
        step=1,
        min_train=2,
        window="expanding",
        embargo_days=7,
        anchor_month=1,
    )
    folds = walk_forward(frame["date"], scheme=scheme)
    assert [f.test_start for f in folds] == [date(2020, 10, 1), date(2021, 1, 1), date(2021, 4, 1)]

    base = fold_vectors(frame, folds)
    # Each fold is its own fit: the regimes differ, so the first and last folds' coefficients do.
    assert np.abs(base[0] - base[2]).max() > 0.05
    everything = np.arange(len(frame))
    for k, fold in enumerate(folds):
        outside = np.setdiff1d(everything, fold.train_idx)
        refit = fold_vectors(scramble(frame, outside), folds)
        # Nothing outside fold k's training rows (its embargo, its test window, later races)
        # reaches fold k's coefficients ...
        np.testing.assert_allclose(refit[k], base[k], atol=1e-6, err_msg=f"fold {k}")
        # ... while the next fold, which trains on fold k's test window, sees the change.
        if k + 1 < len(folds):
            assert np.abs(refit[k + 1] - base[k + 1]).max() > 0.05, f"fold {k + 1}"
    # Rewriting half of the first fold's training races moves its coefficients.
    first_train = frame.iloc[folds[0].train_idx]["race_id"].unique()
    half = frame.index[frame["race_id"].isin(sorted(first_train)[::2])].to_numpy()
    moved = fold_vectors(scramble(frame, half), folds)
    assert np.abs(moved[0] - base[0]).max() > 0.05


def test_quadratic_context_fit_recovers_the_generating_ordering():
    rng = np.random.default_rng(7)
    truth = (0.7, 0.55)  # discounted Harville: a2 = -0.3, a3 = -0.45, the rest 0
    days = [date(2020, 1, 1) + timedelta(days=d) for d in range(366)]
    frame = simulate_frame(rng, days, 4, 10, lambda d: truth)
    scheme = FoldScheme(
        unit="month",
        test_length=1,
        step=1,
        min_train=11,
        window="expanding",
        embargo_days=7,
        anchor_month=1,
    )
    folds = walk_forward(frame["date"], scheme=scheme)
    assert len(folds) == 1 and folds[0].test_start == date(2020, 12, 1)
    (params,) = exotic().Ordering("quadratic_context").fit(frame, folds).fold_params
    fitted = exotic().Ordering("quadratic_context", dict(params)).joint_probs(P10, TRIFECTA)
    true_t = trifecta_reference(P10, P10 ** truth[0], P10 ** truth[1])
    # About 1,300 training races: the fit lands far closer to the generating model than Harville.
    assert kl(true_t, fitted) < 0.2 * kl(true_t, harville_closed_form(P10))


# ---------------------------------------------------------------------------------------------
# The blend with the pool
# ---------------------------------------------------------------------------------------------


def _blend_inputs() -> tuple[pd.Series, pd.Series]:
    trio = {}
    for combination, value in harville_closed_form(P5).items():
        c = key(*sorted(horses(combination)))
        trio[c] = trio.get(c, 0.0) + value
    p_model = pd.Series(trio, dtype=float) * 2.5  # any positive scale
    weights = np.linspace(1.0, 3.0, len(p_model)) ** 2
    q_pool = pd.Series(weights / weights.sum(), index=p_model.index[::-1], dtype=float)
    return p_model, q_pool


@pytest.mark.parametrize(("a", "b"), [(0.0, 1.0), (1.0, 0.0), (0.5, 0.5), (0.8, 0.3), (1.4, -0.2)])
def test_pool_blend_is_a_log_linear_pool(a, b):
    p_model, q_pool = _blend_inputs()
    got = exotic().pool_blend(p_model, q_pool, a=a, b=b)
    assert list(got.index) == list(p_model.index)
    q = q_pool.reindex(p_model.index).to_numpy()
    u = a * np.log(p_model.to_numpy()) + b * np.log(q)
    expected = np.exp(u) / np.exp(u).sum()
    np.testing.assert_allclose(got.to_numpy(dtype=float), expected, rtol=1e-10, atol=1e-15)
    assert got.sum() == pytest.approx(1.0, abs=1e-12)
    if (a, b) == (0.0, 1.0):
        np.testing.assert_allclose(got.to_numpy(dtype=float), q, rtol=1e-10)
    if (a, b) == (1.0, 0.0):
        p = p_model.to_numpy()
        np.testing.assert_allclose(got.to_numpy(dtype=float), p / p.sum(), rtol=1e-10)


def test_pool_blend_needs_the_full_board():
    p_model, q_pool = _blend_inputs()
    with pytest.raises(ValueError):
        exotic().pool_blend(p_model, q_pool.drop(q_pool.index[3]), a=0.5, b=0.5)
