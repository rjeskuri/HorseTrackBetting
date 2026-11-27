"""Ticket #432 (lifted out of #100): the own-stake dividend of a single-winner pool.

Contract pinned here for `horsetrack.countries.japan.models.decision` (new; #100 extends it):

- `own_stake_dividend(pool_total, combo_money, stake, payout_rate) -> int`: the dividend in yen
  per 100 yen once our `stake` is in the pool and on the combination,
  floor(100 x rate x (V + s) / (W + s)) to a multiple of 10 (breakage), and never below 110
  (the Plus-10 floor), V the pool, W the money already on the combination, s the stake, all in
  yen. The arithmetic is decimal: a dividend that is exactly a multiple of 10 (7,250 below) stays
  that multiple, whatever binary floating point makes of 0.725. The result is a Python int.
- ValueError for a pool that is not positive, money on the combination below 0 or above the
  pool, a negative stake, no money on the combination at all (W + s = 0), a payout rate outside
  (0, 1], or any value that is not a finite number.
- The withheld-Plus-10 case (100 yen when paying 110 would exceed the pool) and #100's
  `test_the_floor_and_plus10_apply_to_the_dividend` stay in #100: every case here that pays the
  floor keeps (W + s) / (V + s) at 0.85 or less, where 110 is always paid.

Every figure is invented; nothing reads data.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party.
from __future__ import annotations  # noqa: I001

import math
from fractions import Fraction

import numpy as np
import pytest

from horsetrack.countries.japan.models.decision import own_stake_dividend

RATES = ("0.8", "0.775", "0.75", "0.725", "0.7")  # decimal payout rates, as strings


def expected_dividend(pool: int, combo: int, stake: int, rate: str) -> int:
    """floor(100 x rate x (V + s) / (W + s)) to 10 yen, at least 110, in exact arithmetic."""
    raw = 100 * Fraction(rate) * Fraction(pool + stake, combo + stake)
    return max(110, math.floor(raw / 10) * 10)


def test_own_stake_lowers_the_dividend():
    """#100's example: pool 1,000,000 yen, 10,000 on the combination, payout rate 0.725: 7,250
    per 100 yen with no stake and 6,590 with a 1,000-yen stake."""
    assert own_stake_dividend(1_000_000, 10_000, 0, 0.725) == 7250
    assert own_stake_dividend(1_000_000, 10_000, 1_000, 0.725) == 6590

    # The same by keyword, and as a Python int.
    got = own_stake_dividend(
        pool_total=1_000_000, combo_money=10_000, stake=1_000, payout_rate=0.725
    )
    assert got == 6590 and isinstance(got, int) and not isinstance(got, bool)

    # Exact multiples of 10 survive binary floating point: 100 x 0.8 x 500,000 / 40,000 = 1,000.
    assert own_stake_dividend(500_000, 40_000, 0, 0.8) == 1000
    assert own_stake_dividend(2_000_000, 50_000, 0, 0.775) == 3100
    assert own_stake_dividend(1_200_000, 30_000, 0, 0.75) == 3000

    # A spread of pools, money and stakes: the dividend is the identity, broken to 10 yen, and it
    # falls as the stake rises.
    rng = np.random.default_rng(4321)
    for _ in range(400):
        pool = int(np.exp(rng.uniform(np.log(2e5), np.log(5e9))))
        combo = max(100, int(pool * np.exp(rng.uniform(np.log(1e-5), np.log(0.6)))) // 100 * 100)
        rate = RATES[int(rng.integers(len(RATES)))]
        stakes = sorted({0, 100, 1_000, 10_000, int(rng.integers(100, 200_000)) // 100 * 100})
        previous = None
        for stake in stakes:
            want = expected_dividend(pool, combo, stake, rate)
            got = own_stake_dividend(float(pool), float(combo), float(stake), float(rate))
            assert got == want, (pool, combo, stake, rate, got, want)
            assert isinstance(got, int) and got % 10 == 0
            if previous is not None:
                assert got <= previous, (pool, combo, stake, rate)
            previous = got


def test_own_stake_dividend_never_pays_below_110():
    """The Plus-10 floor: a dividend that breaks below 110 pays 110; one from 110 up breaks to
    its multiple of 10 as usual. (W + s) / (V + s) stays at 0.85 or less here, where 110 is paid
    whatever #100 later adds for the withheld case."""
    pool = 1_000_000
    for combo, rate, want in (
        (800_000, "0.8", 110),  # 100 x 0.8 x 1.25 = 100
        (750_000, "0.8", 110),  # 106.67
        (700_000, "0.8", 110),  # 114.29: breaks to 110
        (690_000, "0.8", 110),  # 115.94
        (640_000, "0.8", 120),  # 125.00: breaks to 120
        (600_000, "0.75", 120),  # 125.00
        (680_000, "0.75", 110),  # 110.29
        (720_000, "0.725", 110),  # 100.69
        (850_000, "0.7", 110),  # 82.35
    ):
        assert expected_dividend(pool, combo, 0, rate) == want
        assert own_stake_dividend(pool, combo, 0, float(rate)) == want, (combo, rate)

    # A stake large enough to sink a long price to the floor: never below 110, always a multiple
    # of 10, never rising with the stake.
    for pool, combo, rate in (
        (1_000_000, 10_000, "0.8"),
        (3_000_000, 250_000, "0.75"),
        (50_000_000, 400_000, "0.725"),
    ):
        previous = None
        floored = False
        for stake in np.unique(np.geomspace(100, 5 * pool, 120).astype(int) // 100 * 100):
            stake = int(stake)
            if (combo + stake) / (pool + stake) > 0.85:
                continue
            got = own_stake_dividend(pool, combo, stake, float(rate))
            assert got == expected_dividend(pool, combo, stake, rate), (pool, combo, stake)
            assert got >= 110 and got % 10 == 0
            if previous is not None:
                assert got <= previous
            previous = got
            floored = floored or got == 110
        assert floored, (pool, combo, rate)  # the sweep reached the floor


@pytest.mark.parametrize(
    "pool, combo, stake, rate",
    [
        (0, 0, 100, 0.8),  # no pool
        (-1_000_000, 10_000, 100, 0.8),
        (1_000_000, -10_000, 100, 0.8),  # negative money on the combination
        (1_000_000, 1_000_001, 100, 0.8),  # more on the combination than in the pool
        (1_000_000, 10_000, -100, 0.8),  # a negative stake
        (1_000_000, 0, 0, 0.8),  # nobody on the combination: no dividend
        (1_000_000, 10_000, 100, 0.0),
        (1_000_000, 10_000, 100, -0.2),
        (1_000_000, 10_000, 100, 1.2),
        (float("nan"), 10_000, 100, 0.8),
        (1_000_000, float("nan"), 100, 0.8),
        (1_000_000, 10_000, float("nan"), 0.8),
        (1_000_000, 10_000, 100, float("nan")),
        (float("inf"), 10_000, 100, 0.8),
        (1_000_000, 10_000, float("inf"), 0.8),
    ],
)
def test_own_stake_dividend_rejects_impossible_inputs(pool, combo, stake, rate):
    with pytest.raises(ValueError):
        own_stake_dividend(pool, combo, stake, rate)
