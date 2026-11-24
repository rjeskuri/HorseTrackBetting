"""Reference maths for test_formations.py (#139), derived from first principles.

Nothing here imports the code under test. Every race is synthetic: Plackett-Luce finish orders
from fixed strengths drawn from a seeded RNG, and dividends priced off a second, "public"
Plackett-Luce with the JRA breakage (floored to 10 yen per 100, never under 110).
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from functools import cache

Combo = tuple[int, ...]

ORDERED = {"TIERCE": True, "FORECAST": True, "TRIO": False, "QUINELLA": False}
LEGS = {"TIERCE": 3, "FORECAST": 2, "TRIO": 3, "QUINELLA": 2}
RATE = {"TIERCE": 0.725, "FORECAST": 0.75, "TRIO": 0.75, "QUINELLA": 0.775}


def canon(combo: Iterable[int], ordered: bool) -> Combo:
    t = tuple(combo)
    return t if ordered else tuple(sorted(t))


def ref_formation(legsets: Sequence[Iterable[int]], ordered: bool) -> set[Combo]:
    """Every distinct-horse pick, one horse from each position's set; unordered picks as sets."""
    legs = len(legsets)
    return {
        canon(t, ordered)
        for t in itertools.product(*[tuple(s) for s in legsets])
        if len(set(t)) == legs
    }


def pl_order_probs(strengths: Mapping[int, float], legs: int) -> dict[Combo, float]:
    """P(first `legs` finishers in this order) under Plackett-Luce."""
    total = sum(strengths.values())
    out: dict[Combo, float] = {}
    for order in itertools.permutations(sorted(strengths), legs):
        p, rest = 1.0, total
        for horse in order:
            p *= strengths[horse] / rest
            rest -= strengths[horse]
        out[order] = p
    return out


def pool_probs(strengths: Mapping[int, float], pool: str) -> dict[Combo, float]:
    ordered = pl_order_probs(strengths, LEGS[pool])
    if ORDERED[pool]:
        return ordered
    out: dict[Combo, float] = {}
    for order, p in ordered.items():
        key = tuple(sorted(order))
        out[key] = out.get(key, 0.0) + p
    return out


def paid(share: float, rate: float) -> int:
    """JRA dividend per 100 yen: 100 * rate / share floored to 10, never under 110."""
    return max(110, math.floor(100 * rate / share / 10 + 1e-9) * 10)


def synthetic_race(seed: int, pool: str, n: int) -> tuple[dict[Combo, float], dict[Combo, int]]:
    """(model probabilities, dividends per 100 yen) for one synthetic race."""
    rng = random.Random(seed)
    model = {h: math.exp(rng.gauss(0.0, 0.9)) for h in range(1, n + 1)}
    public = {h: model[h] * math.exp(rng.gauss(0.0, 0.7)) for h in model}
    probs = pool_probs(model, pool)
    shares = pool_probs(public, pool)
    return probs, {c: paid(shares[c], RATE[pool]) for c in probs}


def _subsets(field: Sequence[int]) -> list[tuple[int, ...]]:
    return [s for r in range(1, len(field) + 1) for s in itertools.combinations(field, r)]


@cache
def all_formations(pool: str, n: int) -> tuple[frozenset[Combo], ...]:
    """Every distinct non-empty combination set any per-position formation on horses 1..n gives."""
    field = tuple(range(1, n + 1))
    found = {
        frozenset(ref_formation(legsets, ORDERED[pool]))
        for legsets in itertools.product(_subsets(field), repeat=LEGS[pool])
    }
    found.discard(frozenset())
    return tuple(found)


def all_boxes(pool: str, n: int) -> tuple[frozenset[Combo], ...]:
    field = tuple(range(1, n + 1))
    return tuple(
        frozenset(ref_formation([s] * LEGS[pool], ORDERED[pool]))
        for r in range(LEGS[pool], n + 1)
        for s in itertools.combinations(field, r)
    )


def values(probs: Mapping[Combo, float], dividends: Mapping[Combo, float]) -> dict[Combo, float]:
    """Gross return per yen of each combination: p * dividend / 100."""
    return {c: probs[c] * dividends[c] / 100 for c in probs}


def ranked(
    candidates: Iterable[frozenset[Combo]],
    value: Mapping[Combo, float],
    *,
    budget: int,
    unit: int,
    min_ev: float,
) -> list[tuple[float, frozenset[Combo]]]:
    """Feasible candidates (cost within budget, every value >= min_ev, profit > 0), best first.

    Profit is sum(value - 1) * unit, in yen.
    """
    most = budget // unit
    out = []
    for s in candidates:
        if len(s) > most or any(value[c] < min_ev for c in s):
            continue
        profit = sum(value[c] - 1.0 for c in s) * unit
        if profit > 0:
            out.append((profit, s))
    out.sort(key=lambda x: (-x[0], len(x[1]), sorted(x[1])))
    return out
