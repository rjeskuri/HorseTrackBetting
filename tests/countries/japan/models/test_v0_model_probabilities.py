"""Model probabilities of every pool from the v0 models (ticket #506): `p_model` per combination.

Contract pinned here (`horsetrack.countries.japan.models.v0.shadow_values.model_probabilities`):

`model_probabilities(p, ordering, *, brackets=None) -> dict[str, dict[str, float]]`
- `p`: horse number -> Win probability of the runners in the race (scratched runners left out).
  It is renormalised to sum to 1 first, as the shadow values renormalise q over the board's
  runners.
- `ordering`: the v0 ordering fit (`OrderingFit`, family discounted_harville, lambdas (l2, l3)),
  or None. P(i, j, k first three in that order) = p_i * p_j^l2 / sum_{l != i} p_l^l2
  * p_k^l3 / sum_{l not in {i, j}} p_l^l3; every pool is an exact sum over these ordered triples.
- The result maps pool code -> combination text (the odds-log format) -> probability:
  - WIN "7": p_7;
  - PLACE "7": P(7 finishes in the first `places`), places from the jra config for len(p)
    runners (3 at 8+, 2 at 5-7; none at 4 or fewer);
  - QUINELLA "a,b" (a < b): P({first two} = {a, b});
  - QUINELLA_PLACE (Wide) "a,b" (a < b): P(a and b both in the first three);
  - FORECAST (exacta) "a,b": P(a first, b second);
  - TRIO "a,b,c" (a < b < c): P({first three} = {a, b, c});
  - TIERCE (trifecta) "a,b,c": P(a first, b second, c third);
  - BRACKET_QUINELLA "x,y" (x <= y), only when `brackets` (horse -> bracket) is given; horses
    not in `p` are ignored: P({bracket of the first, bracket of the second} = {x, y}). "x,x"
    exists when bracket x has two runners or more.
- Each pool has a key for every combination of the runners in `p` and no other (no scratched
  runner, no bracket pair that cannot happen). Unordered keys sort as numbers: "2,10", not "10,2".
- Totals: WIN, QUINELLA, FORECAST, TRIO, TIERCE and BRACKET_QUINELLA sum to 1, PLACE to the
  places paid, Wide to 3.
- Without an ordering fit only WIN has probabilities.
- Display only: `shadow_values` and `ShadowValue` are unchanged (the shadow log's records are
  built from them), so p_model sits beside the values, never in them.

Every expected figure is an independent brute-force enumeration of the ordered triples written
in this file, or a hand-worked Harville number. All data is synthetic.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import dataclasses
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import combinations

import numpy as np
import pytest

from horsetrack.countries.japan.board.snapshot import RaceBoard
from horsetrack.countries.japan.models.v0 import shadow_values as sv
from horsetrack.countries.japan.models.v0.ordering import OrderingFit
from horsetrack.countries.japan.models.v0.shadow_values import (
    PoolDraws,
    ShadowModels,
    ShadowValue,
    shadow_values,
)

FAMILY = "discounted_harville"
WIN, PLACE, QUINELLA, WIDE = "WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE"
FORECAST, TRIO, TIERCE, BRACKET = "FORECAST", "TRIO", "TIERCE", "BRACKET_QUINELLA"
ORDERED_POOLS = (PLACE, QUINELLA, WIDE, FORECAST, TRIO, TIERCE)
REL, ABS = 1e-9, 1e-12


def model_probabilities(p, ordering, **options) -> dict[str, dict[str, float]]:
    """The function under test, looked up on its module when called (the module exists; the
    function is #506's)."""
    return sv.model_probabilities(p, ordering, **options)


def fit(lambdas: tuple[float, float]) -> OrderingFit:
    return OrderingFit(FAMILY, lambdas, "ordering-test", 100)


# ---------------------------------------------------------------------------------------------
# The independent reference: every ordered first three, enumerated
# ---------------------------------------------------------------------------------------------


def triples(p: dict[int, float], lambdas: tuple[float, float]) -> dict[tuple, float]:
    """P(i, j, k first three in that order) for every ordered triple, written out per stage."""
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
                if k in (i, j):
                    continue
                out[(i, j, k)] = first_two * q[k] ** l3 / third
    return out


def text(*horses: int, ordered: bool) -> str:
    return ",".join(str(h) for h in (horses if ordered else sorted(horses)))


def places_paid(runners: int) -> int:
    """JRA Place: 3 places at 8+ runners, 2 at 5-7, not sold at 4 or fewer."""
    return 3 if runners >= 8 else 2 if runners >= 5 else 0


def brute_force(p, lambdas, brackets=None) -> dict[str, dict[str, float]]:
    """Every pool's probabilities by summing the enumerated ordered triples."""
    total = sum(p.values())
    places = places_paid(len(p))
    out: dict[str, dict[str, float]] = {pool: defaultdict(float) for pool in ORDERED_POOLS}
    if brackets is not None:
        out[BRACKET] = defaultdict(float)
    for (i, j, k), prob in triples(p, lambdas).items():
        out[TIERCE][text(i, j, k, ordered=True)] += prob
        out[TRIO][text(i, j, k, ordered=False)] += prob
        out[FORECAST][text(i, j, ordered=True)] += prob
        out[QUINELLA][text(i, j, ordered=False)] += prob
        for a, b in ((i, j), (i, k), (j, k)):
            out[WIDE][text(a, b, ordered=False)] += prob
        for h in (i, j, k)[:places]:
            out[PLACE][str(h)] += prob
        if brackets is not None:
            out[BRACKET][text(brackets[i], brackets[j], ordered=False)] += prob
    if not places:
        del out[PLACE]
    out = {pool: dict(entries) for pool, entries in out.items()}
    out[WIN] = {str(h): v / total for h, v in p.items()}
    return out


def totals(runners: int, with_brackets: bool) -> dict[str, float]:
    out = {WIN: 1.0, QUINELLA: 1.0, WIDE: 3.0, FORECAST: 1.0, TRIO: 1.0, TIERCE: 1.0}
    if places_paid(runners):
        out[PLACE] = float(places_paid(runners))
    if with_brackets:
        out[BRACKET] = 1.0
    return out


def offered(got: dict) -> set[str]:
    """The pools of a result that carry probabilities (a pool not offered may be absent or {})."""
    return {pool for pool, entries in got.items() if entries}


def assert_matches(got: dict, want: dict, where) -> None:
    assert offered(got) == set(want), (where, sorted(offered(got)), sorted(want))
    for pool, entries in want.items():
        assert set(got[pool]) == set(entries), (where, pool, sorted(set(got[pool]) ^ set(entries)))
        for combination, prob in entries.items():
            value = got[pool][combination]
            assert isinstance(value, float), (where, pool, combination, type(value))
            assert math.isclose(value, prob, rel_tol=REL, abs_tol=ABS), (
                where,
                pool,
                combination,
                value,
                prob,
            )


def assert_totals(got: dict, runners: int, with_brackets: bool, where) -> None:
    for pool, total in totals(runners, with_brackets).items():
        assert math.isclose(math.fsum(got[pool].values()), total, rel_tol=1e-9), (where, pool)


def dirichlet(horses, seed: int) -> dict[int, float]:
    rng = np.random.default_rng(seed)
    horses = list(horses)
    return dict(
        zip(horses, (float(x) for x in rng.dirichlet(np.full(len(horses), 0.9))), strict=True)
    )


def jra_draw(sizes: tuple[int, ...]) -> dict[int, int]:
    """Horse -> bracket from the number of horses each of the 8 brackets takes, in number order."""
    out, horse = {}, 1
    for bracket, size in enumerate(sizes, start=1):
        for _ in range(size):
            out[horse] = bracket
            horse += 1
    return out


# ---------------------------------------------------------------------------------------------
# Every pool against the enumeration, over a spread of fields and discounts
# ---------------------------------------------------------------------------------------------

FIELDS = {  # id: (horse numbers, (l2, l3), seed)
    "5 runners, plain Harville": (range(1, 6), (1.0, 1.0), 5),
    "6 runners": (range(1, 7), (0.8, 0.7), 6),
    "7 runners, l2 above 1": (range(1, 8), (1.2, 0.6), 7),
    "8 runners": (range(1, 9), (0.85, 0.65), 8),
    "8 runners, gaps in the numbers": ((1, 2, 4, 5, 7, 9, 11, 12), (0.9, 0.75), 81),
    "12 runners": (range(1, 13), (0.95, 0.8), 12),
    "16 runners, strong discounts": (range(1, 17), (0.5, 0.3), 16),
}


@pytest.mark.parametrize(("horses", "lambdas", "seed"), list(FIELDS.values()), ids=list(FIELDS))
def test_every_pool_matches_the_enumeration_of_ordered_finishes(horses, lambdas, seed):
    p = dirichlet(horses, seed)

    got = model_probabilities(p, fit(lambdas))

    assert_matches(got, brute_force(p, lambdas), (len(p), lambdas))
    assert_totals(got, len(p), False, (len(p), lambdas))
    assert not got.get(BRACKET), "bracket quinella without a bracket map"
    # the discounts are read: another (l2, l3) moves the second and third places
    other = model_probabilities(p, fit((lambdas[0] + 0.3, lambdas[1] + 0.4)))
    moved = [c for c, v in got[FORECAST].items() if not math.isclose(v, other[FORECAST][c])]
    assert moved, "FORECAST ignores l2"
    moved = [c for c, v in got[TIERCE].items() if not math.isclose(v, other[TIERCE][c])]
    assert moved, "TIERCE ignores the discounts"


def test_plain_harville_by_hand():
    """Four runners, (l2, l3) = (1, 1): P(i, j, k) = p_i p_j / (1 - p_i) * p_k / (1 - p_i - p_j)."""
    p = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}

    got = model_probabilities(p, fit((1.0, 1.0)))

    hand = {
        (TIERCE, "1,2,3"): 0.4 * (0.3 / 0.6) * (0.2 / 0.3),
        (TIERCE, "4,3,2"): 0.1 * (0.2 / 0.9) * (0.3 / 0.7),
        (TIERCE, "2,1,4"): 0.3 * (0.4 / 0.7) * (0.1 / 0.3),
        (FORECAST, "2,1"): 0.3 * (0.4 / 0.7),
        (FORECAST, "1,2"): 0.4 * (0.3 / 0.6),
        (FORECAST, "4,1"): 0.1 * (0.4 / 0.9),
        (QUINELLA, "1,2"): 0.4 * (0.3 / 0.6) + 0.3 * (0.4 / 0.7),
        (QUINELLA, "3,4"): 0.2 * (0.1 / 0.8) + 0.1 * (0.2 / 0.9),
        # the first three without runner 1: the six orders of 2, 3, 4
        (TRIO, "2,3,4"): 0.3 * (0.2 / 0.7) * (0.1 / 0.5)
        + 0.3 * (0.1 / 0.7) * (0.2 / 0.6)
        + 0.2 * (0.3 / 0.8) * (0.1 / 0.5)
        + 0.2 * (0.1 / 0.8) * (0.3 / 0.7)
        + 0.1 * (0.3 / 0.9) * (0.2 / 0.6)
        + 0.1 * (0.2 / 0.9) * (0.3 / 0.7),
        (WIN, "3"): 0.2,
    }
    for (pool, combination), want in hand.items():
        assert math.isclose(got[pool][combination], want, rel_tol=REL), (pool, combination)
    # four runners: one of them misses the first three, so Wide 1-2 is 1 - P(1 out) - P(2 out)
    one_out = got[TRIO]["2,3,4"]
    two_out = got[TRIO]["1,3,4"]
    assert math.isclose(got[WIDE]["1,2"], 1.0 - one_out - two_out, rel_tol=REL)
    assert not got.get(PLACE), "Place is not sold with four runners"


# ---------------------------------------------------------------------------------------------
# Scratches: no key for the runner, the others renormalised
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("horses", "out", "lambdas"),
    [
        (range(1, 9), {3}, (0.85, 0.7)),  # 8 drawn, 7 run: Place pays two places
        (range(1, 11), {1, 10}, (0.9, 0.6)),  # the favourite's number and the last one out
        (range(1, 13), {4, 5, 11}, (1.1, 0.8)),
    ],
)
def test_scratched_runners_have_no_key_and_the_rest_are_renormalised(horses, out, lambdas):
    full = dirichlet(horses, 33)
    running = {h: v for h, v in full.items() if h not in out}  # sums to less than 1

    got = model_probabilities(running, fit(lambdas))

    want = brute_force(running, lambdas)
    assert_matches(got, want, out)
    assert_totals(got, len(running), False, out)
    for pool, entries in got.items():
        for combination in entries:
            assert not {int(h) for h in combination.split(",")} & out, (pool, combination)
    # only the ratios count: the same runners on another scale give the same probabilities
    scaled = model_probabilities({h: 7.5 * v for h, v in running.items()}, fit(lambdas))
    assert_matches(scaled, want, ("scaled", out))


# ---------------------------------------------------------------------------------------------
# Bracket quinella: summed over the brackets' runners
# ---------------------------------------------------------------------------------------------

DRAWS = {  # drawn horses: horses per bracket 1-8 (JRA: the extra horses go to the last brackets)
    9: (1, 1, 1, 1, 1, 1, 1, 2),
    10: (1, 1, 1, 1, 1, 1, 2, 2),
    12: (1, 1, 1, 1, 2, 2, 2, 2),
    16: (2, 2, 2, 2, 2, 2, 2, 2),
    18: (2, 2, 2, 2, 2, 2, 3, 3),
}


@pytest.mark.parametrize(
    ("drawn", "out", "lambdas"),
    [
        (9, set(), (0.85, 0.7)),
        (10, set(), (1.0, 1.0)),
        (12, {3, 6}, (0.9, 0.65)),  # bracket 3 loses its only runner, bracket 5 one of two
        (16, {16}, (0.8, 0.75)),
        (18, {13, 17, 18}, (0.95, 0.7)),
    ],
)
def test_bracket_quinella_sums_the_brackets_runners(drawn, out, lambdas):
    brackets = jra_draw(DRAWS[drawn])
    running = {h: v for h, v in dirichlet(range(1, drawn + 1), drawn).items() if h not in out}

    # the full draw is passed, scratched horses included: they are ignored
    got = model_probabilities(running, fit(lambdas), brackets=brackets)

    want = brute_force(running, lambdas, brackets)
    assert_matches(got, want, (drawn, out))
    assert_totals(got, len(running), True, (drawn, out))
    left = defaultdict(int)
    for h in running:
        left[brackets[h]] += 1
    same = {f"{b},{b}" for b, n in left.items() if n >= 2}
    assert {c for c in got[BRACKET] if c.split(",")[0] == c.split(",")[1]} == same
    assert (
        set(got[BRACKET])
        == {text(a, b, ordered=False) for a, b in combinations(sorted(left), 2)} | same
    )
    only_running = {h: b for h, b in brackets.items() if h in running}
    again = model_probabilities(running, fit(lambdas), brackets=only_running)
    assert_matches(again, want, ("running map", drawn))


def test_an_18_runner_race_has_every_combination():
    p = dirichlet(range(1, 19), 1818)
    lambdas = (0.85, 0.65)

    got = model_probabilities(p, fit(lambdas), brackets=jra_draw(DRAWS[18]))

    counts = {pool: len(entries) for pool, entries in got.items()}
    assert counts == {
        WIN: 18,
        PLACE: 18,
        QUINELLA: 153,
        WIDE: 153,
        FORECAST: 18 * 17,
        TRIO: 816,
        TIERCE: 4_896,
        BRACKET: 36,  # every bracket has two runners or more: 28 pairs and 8 same-bracket
    }, counts
    assert_matches(got, brute_force(p, lambdas, jra_draw(DRAWS[18])), 18)
    assert_totals(got, 18, True, 18)
    assert "2,10" in got[QUINELLA] and "10,2" not in got[QUINELLA]
    assert "10,2,5" in got[TIERCE] and "2,5,10" in got[TRIO] and "10,2,5" not in got[TRIO]


# ---------------------------------------------------------------------------------------------
# Without an ordering fit; beside the shadow values, never in them
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("brackets", [None, jra_draw(DRAWS[10])], ids=["no-brackets", "brackets"])
def test_without_an_ordering_fit_only_win_has_probabilities(brackets):
    p = {h: float(h) for h in range(1, 11)}  # unnormalised on purpose

    got = model_probabilities(p, None, brackets=brackets)

    assert offered(got) == {WIN}, sorted(offered(got))
    assert set(got[WIN]) == {str(h) for h in range(1, 11)}
    for h in range(1, 11):
        assert math.isclose(got[WIN][str(h)], h / 55.0, rel_tol=REL), h


SHADOW_VALUE_FIELDS = [  # the shadow log's source as it is before #506
    "race_id",
    "pool",
    "combination",
    "tau_min",
    "board_odds",
    "board_odds_hi",
    "value_mean",
    "value_q05",
    "value_q95",
    "value_at_low",
    "value_at_high",
    "reason",
]


class _Identity:
    def predict(self, q, race=None):
        return np.asarray(q, dtype=float)


class _Unmoved:
    """The board's own figures on every draw."""

    def draws(self, board, *, pool, tau_min, shocks):
        n = len(next(iter(shocks.values())))
        if pool == WIN:
            keys = sorted(board.win)
            return PoolDraws(
                WIN, tuple(map(str, keys)), np.tile([board.win[h] for h in keys], (n, 1)), None
            )
        figures = board.place if pool == PLACE else board.wide
        keys = sorted(figures)
        combos = tuple(str(k) if pool == PLACE else f"{k[0]},{k[1]}" for k in keys)
        low = np.tile([figures[k][0] for k in keys], (n, 1)).astype(float)
        high = np.tile([figures[k][1] for k in keys], (n, 1)).astype(float)
        return PoolDraws(pool, combos, low, high)


def test_model_probabilities_sit_beside_the_shadow_values_which_are_unchanged():
    """The sealed shadow log is built from shadow_values: its rows stay WIN, PLACE and Wide, with
    the same fields, while the exotic pools' probabilities come from model_probabilities."""
    at = datetime(2026, 11, 28, 5, 51, tzinfo=timezone.utc)
    horses = range(1, 9)
    win = {h: round(1.9 + 2.1 * h, 1) for h in horses}
    board = RaceBoard(
        race_id="jra:2026-11-28:TOK:11",
        post_utc=at + timedelta(minutes=10),
        at_utc=at,
        capture_utc=at,
        final=False,
        win=win,
        place={h: (round(1.1 + 0.3 * h, 1), round(1.6 + 0.5 * h, 1)) for h in horses},
        wide={
            (a, b): (round(1.2 + 0.4 * (a + b), 1), round(2.0 + 0.6 * (a + b), 1))
            for a, b in combinations(horses, 2)
        },  # fmt: skip
        totals={},
        scratched=frozenset(),
    )
    lambdas = (0.85, 0.7)
    models = ShadowModels(win=_Identity(), ordering=fit(lambdas), nowcast=_Unmoved(), ids={})

    rows = shadow_values(board, tau_min=5, models=models, n_draws=4, seed=1)

    assert [f.name for f in dataclasses.fields(ShadowValue)] == SHADOW_VALUE_FIELDS
    assert {r.pool for r in rows} == {WIN, PLACE, WIDE}
    assert len(rows) == 8 + 8 + 28
    inv = {h: 1.0 / o for h, o in win.items()}
    got = model_probabilities(inv, models.ordering)
    assert_matches(got, brute_force(inv, lambdas), "board")
    # the Place value is still the mean of P * low, P from the same ordering fit
    place = {r.combination: r for r in rows if r.pool == PLACE}
    for h in horses:
        low = board.place[h][0]
        assert math.isclose(place[str(h)].value_at_low, got[PLACE][str(h)] * low, rel_tol=1e-9), h
