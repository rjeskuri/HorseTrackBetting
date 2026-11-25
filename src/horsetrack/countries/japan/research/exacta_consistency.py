"""Exacta cross-pool consistency (#131): does the exacta pool overload the favourite?

For each race the exacta pool's implied share of a combination (D5's primary baseline) is set
against the Win-derived discounted Harville probability (D5's secondary baseline), and the ratio
is summed by category: the favourite first, the favourite second, and without the favourite.

- The implied share is ``rate / ((D + step/2) / unit)``: the payout identity at the final pool,
  with half a 10-yen breakage step added back (a dividend D per 100 yen stands for a true payout in
  [D, D + 10)). At or below the 110-yen minimum the floor may bind, so the share is not identified
  (NaN).
- ``rate`` is the payout rate the race paid, ``countries/japan/takeout.race_takeout(race_id,
  "FORECAST").payout_rate``, looked up per race: Premium days raise it for designated races, so it
  is never the constant 0.75.
- History (the store's ``dividends`` table) holds only the winning combination's dividend, so a
  historical race gives one combination's ratio and NaN for the other two categories. The full
  per-race category ratio needs the whole final exacta board, logged only from #130 onward.

Pure functions only: no data reads, no registry, no CLI. The study's design is
docs/specs/exacta-overload-prereg.md (approved by the owner 2026-11-25, frozen).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from horsetrack.countries.japan.models.v0 import ordering

CATEGORIES = ("fav_first", "fav_second", "no_fav")


def _check_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


def implied_share(
    dividend: float, *, rate: float, unit: float = 100, step: float = 10, min_dividend: float = 110
) -> float:
    """The pool share a combination paying `dividend` per `unit` held, breakage added back.

    NaN at or below `min_dividend` (the minimum payout may have bound); ValueError for a
    non-finite or non-positive dividend or rate."""
    dividend = _check_positive("dividend", dividend)
    rate = _check_positive("rate", rate)
    if dividend <= min_dividend:
        return math.nan
    return rate / ((dividend + step / 2) / unit)


def harville_exacta(p: np.ndarray, lambda2: float = 1.0) -> np.ndarray:
    """P(first = i, second = j), n x n with a zero diagonal, from Win probabilities `p`.

    The v0 ordering model's first two stages (discounted Harville, lambda2 = 1 is Harville); `p` is
    renormalised."""
    return ordering._order_probs(ordering._check_p(p), (float(lambda2), 1.0), 2)


def favourite(win_odds: Mapping[int, float]) -> int:
    """The horse with the lowest Win odds; a tie goes to the lowest horse number."""
    if not win_odds:
        raise ValueError("win_odds is empty")
    for horse, odds in win_odds.items():
        _check_positive(f"the Win odds of horse {horse}", odds)
    return min(win_odds, key=lambda h: (win_odds[h], h))


def combo_category(combo: tuple[int, int], favourite: int) -> str:
    """The category of an exacta (first, second): "fav_first", "fav_second" or "no_fav"."""
    first, second = combo
    if first == second:
        raise ValueError(f"an exacta names two different horses, got {combo}")
    if first == favourite:
        return "fav_first"
    if second == favourite:
        return "fav_second"
    return "no_fav"


@dataclass(frozen=True)
class RaceRatios:
    """One race's pool-to-model ratios.

    categories: sum of q_c / sum of p_c over the category's identified combinations (NaN if none).
    combinations: q_c / p_c for every supplied combination (NaN when q_c is not identified)."""

    categories: dict[str, float]
    combinations: dict[tuple[int, int], float]


def race_ratios(
    p: Mapping[int, float],
    dividends: Mapping[tuple[int, int], float],
    *,
    favourite: int,
    rate: float,
    lambda2: float = 1.0,
) -> RaceRatios:
    """The exacta pool's implied probability over the ordering model's, by combination and category.

    `p` maps horse number to Win probability over the field (renormalised); `dividends` maps
    (first, second) to the final dividend per 100 yen; `rate` is the race's exacta payout rate.
    A combination or favourite naming a horse not in `p` raises ValueError."""
    _check_positive("rate", rate)
    horses = sorted(p)
    index = {h: k for k, h in enumerate(horses)}
    if favourite not in index:
        raise ValueError(f"the favourite {favourite} is not in the field {horses}")
    model = harville_exacta(np.array([p[h] for h in horses], dtype=float), lambda2=lambda2)

    q_sum = dict.fromkeys(CATEGORIES, 0.0)
    p_sum = dict.fromkeys(CATEGORIES, 0.0)
    combinations: dict[tuple[int, int], float] = {}
    for combo, dividend in dividends.items():
        first, second = combo
        absent = [h for h in combo if h not in index]
        if absent:
            raise ValueError(f"combination {combo} names horses not in the field: {absent}")
        category = combo_category(combo, favourite)
        q = implied_share(dividend, rate=rate)
        p_c = float(model[index[first], index[second]])
        combinations[combo] = q / p_c if not math.isnan(q) else math.nan
        if not math.isnan(q):
            q_sum[category] += q
            p_sum[category] += p_c
    categories = {c: q_sum[c] / p_sum[c] if p_sum[c] > 0 else math.nan for c in CATEGORIES}
    return RaceRatios(categories=categories, combinations=combinations)
