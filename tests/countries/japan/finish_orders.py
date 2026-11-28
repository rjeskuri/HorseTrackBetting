"""Independent references for the model probabilities of every pool (ticket #506).

- `triples(p, lambdas)`: P(i, j, k first three in that order) under the discounted Harville model,
  P = p_i * p_j^l2 / sum_{l != i} p_l^l2 * p_k^l3 / sum_{l not in {i, j}} p_l^l3, enumerated.
- `pool_probabilities(p, lambdas, brackets=None, starters=None)`: every pool's probabilities
  summed from those triples, keyed by the odds log's combination texts (WIN/PLACE "7"; QUINELLA,
  QUINELLA_PLACE "a,b" with a < b; FORECAST "a,b" in finishing order; TRIO "a,b,c" ascending;
  TIERCE "a,b,c" in finishing order; BRACKET_QUINELLA "x,y" with x <= y over the horses'
  brackets). `starters` (default len(p)) is the runners at the start of sales: it sets the places
  Place pays and whether Place (5+) and Wide (4+) are sold; the triples are over `p` alone.
- `jra_brackets(drawn)`: JRA's draw of `drawn` horses into the 8 brackets, in number order: up to
  8 horses one bracket each; 9-16: brackets 1..16-N one horse, the rest two; 17: brackets 1-7
  two, bracket 8 three; 18: brackets 1-6 two, 7 and 8 three. Written out as a table.
- `places_paid(starters)`: JRA Place, 3 places at 8+ starters, 2 at 5-7, none at 4 or fewer;
  `wide_sold(starters)`: JRA Wide, three places, sold from 4 starters.

Nothing here calls the code under test.
"""

from __future__ import annotations

from collections import defaultdict

BRACKET_SIZES = {  # drawn horses -> horses in brackets 1..8
    9: (1, 1, 1, 1, 1, 1, 1, 2),
    10: (1, 1, 1, 1, 1, 1, 2, 2),
    11: (1, 1, 1, 1, 1, 2, 2, 2),
    12: (1, 1, 1, 1, 2, 2, 2, 2),
    13: (1, 1, 1, 2, 2, 2, 2, 2),
    14: (1, 1, 2, 2, 2, 2, 2, 2),
    15: (1, 2, 2, 2, 2, 2, 2, 2),
    16: (2, 2, 2, 2, 2, 2, 2, 2),
    17: (2, 2, 2, 2, 2, 2, 2, 3),
    18: (2, 2, 2, 2, 2, 2, 3, 3),
}


def jra_brackets(drawn: int) -> dict[int, int]:
    if drawn <= 8:
        return {h: h for h in range(1, drawn + 1)}
    out, horse = {}, 1
    for bracket, size in enumerate(BRACKET_SIZES[drawn], start=1):
        for _ in range(size):
            out[horse] = bracket
            horse += 1
    return out


def places_paid(starters: int) -> int:
    return 3 if starters >= 8 else 2 if starters >= 5 else 0


def wide_sold(starters: int) -> bool:
    return starters >= 4


def triples(p: dict[int, float], lambdas: tuple[float, float]) -> dict[tuple, float]:
    total = sum(p.values())
    q = {h: v / total for h, v in p.items()}
    l2, l3 = lambdas
    out = {}
    for i in q:
        second = sum(q[h] ** l2 for h in q if h != i)
        for j in q:
            if j == i:
                continue
            first_two = q[i] * q[j] ** l2 / second
            third = sum(q[h] ** l3 for h in q if h not in (i, j))
            for k in q:
                if k not in (i, j):
                    out[(i, j, k)] = first_two * q[k] ** l3 / third
    return out


def _text(horses, ordered: bool) -> str:
    return ",".join(str(h) for h in (horses if ordered else sorted(horses)))


def pool_probabilities(
    p: dict[int, float],
    lambdas: tuple[float, float],
    brackets: dict[int, int] | None = None,
    starters: int | None = None,
) -> dict[str, dict[str, float]]:
    total = sum(p.values())
    starters = len(p) if starters is None else starters
    places = places_paid(starters)
    out: dict[str, dict[str, float]] = {
        pool: defaultdict(float)
        for pool in ("PLACE", "QUINELLA", "QUINELLA_PLACE", "FORECAST", "TRIO", "TIERCE")
    }
    if brackets is not None:
        out["BRACKET_QUINELLA"] = defaultdict(float)
    for (i, j, k), prob in triples(p, lambdas).items():
        out["TIERCE"][_text((i, j, k), True)] += prob
        out["TRIO"][_text((i, j, k), False)] += prob
        out["FORECAST"][_text((i, j), True)] += prob
        out["QUINELLA"][_text((i, j), False)] += prob
        for pair in ((i, j), (i, k), (j, k)):
            out["QUINELLA_PLACE"][_text(pair, False)] += prob
        for h in (i, j, k)[:places]:
            out["PLACE"][str(h)] += prob
        if brackets is not None:
            out["BRACKET_QUINELLA"][_text((brackets[i], brackets[j]), False)] += prob
    if not places:
        del out["PLACE"]
    if not wide_sold(starters):
        del out["QUINELLA_PLACE"]
    result = {pool: dict(entries) for pool, entries in out.items()}
    result["WIN"] = {str(h): v / total for h, v in p.items()}
    return result


def pool_totals(starters: int, with_brackets: bool) -> dict[str, float]:
    """What each pool's probabilities sum to, for `starters` at the start of sales."""
    out = {"WIN": 1.0, "QUINELLA": 1.0, "FORECAST": 1.0, "TRIO": 1.0, "TIERCE": 1.0}
    if wide_sold(starters):
        out["QUINELLA_PLACE"] = 3.0
    if places_paid(starters):
        out["PLACE"] = float(places_paid(starters))
    if with_brackets:
        out["BRACKET_QUINELLA"] = 1.0
    return out
