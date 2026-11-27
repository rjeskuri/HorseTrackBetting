"""The decision layer's pricing of our own stake (#432; #100 extends this file).

#432 (N3, pool depth) lifted `own_stake_dividend` out of #100 because the capacity study
(`research/pool_depth.py`) needs the identity now. #100 (the shadow decider) adds the rest here:
the EV over draws, the co-placer enumeration, the refits, `ShadowDecider` and the withheld-Plus-10
case (100 yen when paying 110 would exceed the pool), with its
`test_the_floor_and_plus10_apply_to_the_dividend`. Extend this module; do not start another.

`own_stake_dividend` is the single-winner identity (Win, Quinella, Exacta, Trio, Trifecta): the
dividend per 100 yen is 100 x rate x V / W, V the pool and W the money on the winning combination,
both in yen. Our stake s goes into both, so it pays floor(100 x rate x (V + s) / (W + s)) broken
down to 10 yen, and never below the 110-yen floor (Plus 10). The arithmetic is exact: each value is
read as the decimal it prints as, so a dividend that is a multiple of 10 (7,250 at a rate of
0.725) is not lost to binary floating point. Place and Wide pay by the stake-back-share rule with
co-placers, which this identity does not cover.
"""

from __future__ import annotations

import math
from fractions import Fraction
from numbers import Integral, Real

UNIT = 100  # yen per ticket: dividends are quoted per 100 yen
STEP = 10  # breakage: dividends are multiples of 10 yen
FLOOR = 110  # the Plus-10 minimum dividend per 100 yen


def _exact(name: str, value: object) -> Fraction:
    """`value` as an exact fraction of the decimal it prints as; ValueError when not finite."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number, not {value!r}")
    if isinstance(value, Integral):
        return Fraction(int(value))
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, not {value!r}")
    return Fraction(repr(number))


def own_stake_dividend(
    pool_total: float, combo_money: float, stake: float, payout_rate: float
) -> int:
    """The dividend in yen per 100 yen once `stake` is in the pool and on the combination.

    floor(100 x payout_rate x (V + s) / (W + s)) to a multiple of 10, at least 110; V
    `pool_total`, W `combo_money` (the money already on the combination) and s `stake`, all in
    yen. ValueError for a pool that is not positive, W below 0 or above V, a negative stake, no
    money on the combination at all (W + s = 0), a payout rate outside (0, 1], or a value that is
    not a finite number.
    """
    pool = _exact("pool_total", pool_total)
    combo = _exact("combo_money", combo_money)
    own = _exact("stake", stake)
    rate = _exact("payout_rate", payout_rate)
    if pool <= 0:
        raise ValueError(f"pool_total must be positive, not {pool_total!r}")
    if not 0 <= combo <= pool:
        raise ValueError(f"combo_money must be in [0, pool_total], not {combo_money!r}")
    if own < 0:
        raise ValueError(f"stake must not be negative, not {stake!r}")
    if combo + own == 0:
        raise ValueError("no money on the combination: it has no dividend")
    if not 0 < rate <= 1:
        raise ValueError(f"payout_rate must be in (0, 1], not {payout_rate!r}")
    raw = UNIT * rate * (pool + own) / (combo + own)
    return max(FLOOR, math.floor(raw / STEP) * STEP)
