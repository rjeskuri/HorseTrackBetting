"""Ticket #139: exotic formations (box, wheel, multi, formation) and an EV-per-yen optimizer.

Contract pinned here, module ``horsetrack.countries.japan.betting.formations``:

Pools are the jra.yaml codes: ``TIERCE`` (三連単, ordered, 3 legs), ``FORECAST`` (馬単, ordered,
2), ``TRIO`` (三連複, unordered, 3), ``QUINELLA`` (馬連, unordered, 2), ``QUINELLA_PLACE`` (ワイド,
unordered, 2). Any other pool (``WIN``, ``PLACE``, ``WIN5``) raises ValueError. A combination is a
tuple of horse numbers: finishing order for an ordered pool, ascending for an unordered one.

- ``box(pool, horses) -> TicketSet``; ``wheel(pool, axis, opponents, *, positions=None,
  multi=False) -> TicketSet``; ``formation(pool, legs) -> TicketSet``.
  ``TicketSet`` has ``pool``, ``combinations`` (tuple of combos, no repeats), ``n_tickets`` and
  ``cost(*, unit) -> int`` (= n_tickets x unit; unit a positive multiple of 100, else ValueError).
  ``wheel`` on an ordered pool without multi puts ``axis[i]`` at 1-based ``positions[i]``;
  ``multi=True`` lets the axis horses take any positions; multi on an unordered pool is ValueError.
- ``paid_dividend(pool, share, *, rate) -> int``: 100 x rate / share floored to the pool's
  dividend step (10) and never under its min dividend (110), both from jra.yaml.
- ``evaluate(tickets, probs, dividends, *, unit) -> Evaluation`` with ``pool, combinations,
  n_tickets, cost, expected_return, expected_profit, ev_per_yen``: expected_return = sum p x
  dividend / 100 x unit (yen), expected_profit = expected_return - cost, ev_per_yen =
  expected_return / cost (None when empty). ``dividends`` are per 100 yen.
- ``optimise_formation(pool, field, probs, dividends, *, budget, unit, min_ev) ->
  FormationChoice``; budget, unit and min_ev are owner decisions: keyword-only, no default.
  Search space: every per-position formation over ``field`` (A/B/C sets included). Feasible:
  cost <= budget and every combination's value p x dividend / 100 >= min_ev. Objective: the
  largest expected_profit, which must be > 0, else the empty choice. Ties: fewer tickets, then the
  lexicographically smallest sorted combination list.
  ``FormationChoice`` (a dataclass) has ``pool``, ``legs`` (per-position horse tuples that expand
  through ``formation`` to the chosen set; ``()`` when empty), ``chosen: Evaluation`` (empty
  evaluation when nothing qualifies), ``best_single`` and ``best_box`` (``Evaluation | None``:
  the best single combination and the best box under the same rules, for comparison) and
  ``shadow`` (True). No field anywhere names a stake, bet, kelly, amount or wager.
- ``log_choice(choice, *, day, race_id, tau_min, decision_utc, board_capture_utc, written_utc,
  mode, models, git_sha) -> int`` appends one record to the day's sealed shadow log (#119) and
  returns 1. The record carries ``pool``, ``race_id``, ``tau_min``, ``n_tickets`` and
  ``combinations`` (texts like ``"4,7,9"``, the store's form); an empty choice is logged too.

Every race is synthetic (formations_support.py). HORSETRACK_DATA_DIR points at tmp_path.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import itertools
import math
import random
from datetime import date

import formations_support as S
import pytest

from horsetrack.core import paths
from horsetrack.countries.japan.shadow.log import STAKE_LIKE, ShadowLog, shadow_log_path

F = importlib.import_module(
    "horsetrack.countries.japan.betting.formations"
)  # missing = the red state

ORDERED_POOLS = {"TIERCE": 3, "FORECAST": 2}
UNORDERED_POOLS = {"TRIO": 3, "QUINELLA": 2, "QUINELLA_PLACE": 2}
ALL_POOLS = {**ORDERED_POOLS, **UNORDERED_POOLS}
DEV_DAY = date(2026, 11, 21)  # before the Japan holdout start: a development day
LABELS = {1: 9, 2: 3, 3: 14, 4: 1, 5: 6, 6: 11}  # non-monotonic relabelling of horses 1..6


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    data = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    return data


def is_ordered(pool: str) -> bool:
    return pool in ORDERED_POOLS


def as_set(tickets) -> set[tuple[int, ...]]:
    combos = [tuple(c) for c in tickets.combinations]
    assert len(combos) == len(set(combos)), "a combination is counted twice"
    return set(combos)


def stake_like_key(value):
    if isinstance(value, dict):
        for name, inner in value.items():
            if any(part in str(name).lower() for part in STAKE_LIKE):
                return str(name)
            found = stake_like_key(inner)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for inner in value:
            found = stake_like_key(inner)
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------- expansion: box


BOX_COUNT = {
    "TIERCE": lambda n: n * (n - 1) * (n - 2),
    "FORECAST": lambda n: n * (n - 1),
    "TRIO": lambda n: math.comb(n, 3),
    "QUINELLA": lambda n: math.comb(n, 2),
    "QUINELLA_PLACE": lambda n: math.comb(n, 2),
}


@pytest.mark.parametrize("pool", sorted(ALL_POOLS))
def test_box_counts_and_combinations_match_jra_mark_cards(pool):
    """Box (ボックス) of n horses: 三連単 n(n-1)(n-2) (5 -> 60, 6 -> 120), 馬単 n(n-1) (4 -> 12),
    三連複 C(n,3) (5 -> 10), 馬連 and ワイド C(n,2) (5 -> 10)."""
    legs = ALL_POOLS[pool]
    pick = [17, 2, 9, 5, 12, 1, 8, 14]
    for n in range(legs, len(pick) + 1):
        horses = pick[:n]
        tickets = F.box(pool, horses)
        if is_ordered(pool):
            expected = set(itertools.permutations(horses, legs))
        else:
            expected = {tuple(sorted(c)) for c in itertools.combinations(horses, legs)}
        assert tickets.pool == pool
        assert as_set(tickets) == expected
        assert tickets.n_tickets == BOX_COUNT[pool](n) == len(expected)


def test_box_worked_examples():
    """The ticket's own example and a few more, counted by hand."""
    assert F.box("TIERCE", [1, 2, 3, 4, 5]).n_tickets == 60
    assert F.box("TIERCE", [3, 7, 11]).n_tickets == 6
    assert F.box("TRIO", [1, 2, 3, 4, 5]).n_tickets == 10
    assert F.box("FORECAST", [4, 8, 12, 16]).n_tickets == 12
    assert F.box("QUINELLA", [1, 2, 3, 4]).n_tickets == 6
    assert as_set(F.box("TRIO", [9, 2, 5])) == {(2, 5, 9)}
    assert as_set(F.box("FORECAST", [6, 2])) == {(6, 2), (2, 6)}


# ---------------------------------------------------------------- expansion: wheel and multi


def ref_wheel(pool, axis, opponents, positions, multi):
    """From the definition: every distinct-horse pick from axis + opponents holding each axis horse
    (at its fixed position unless multi)."""
    legs = ALL_POOLS[pool]
    out = set()
    for t in itertools.permutations(list(axis) + list(opponents), legs):
        if not all(a in t for a in axis):
            continue
        if is_ordered(pool) and not multi:
            if any(t[p - 1] != a for a, p in zip(axis, positions, strict=True)):
                continue
        out.add(t if is_ordered(pool) else tuple(sorted(t)))
    return out


# (pool, positions or None, multi, count formula in k opponents)
WHEELS = [
    ("TIERCE", (1,), False, lambda k: k * (k - 1)),  # 1着ながし
    ("TIERCE", (2,), False, lambda k: k * (k - 1)),  # 2着ながし
    ("TIERCE", (3,), False, lambda k: k * (k - 1)),  # 3着ながし
    ("TIERCE", (1, 2), False, lambda k: k),  # 1・2着ながし
    ("TIERCE", (1, 3), False, lambda k: k),  # 1・3着ながし
    ("TIERCE", (2, 3), False, lambda k: k),  # 2・3着ながし
    ("TIERCE", 1, True, lambda k: 3 * k * (k - 1)),  # 軸1頭マルチ: 5 opponents -> 60
    ("TIERCE", 2, True, lambda k: 6 * k),  # 軸2頭マルチ: 5 opponents -> 30
    ("FORECAST", (1,), False, lambda k: k),  # 1着ながし
    ("FORECAST", (2,), False, lambda k: k),  # 2着ながし
    ("FORECAST", 1, True, lambda k: 2 * k),  # マルチ: 5 opponents -> 10
    ("TRIO", 1, False, lambda k: math.comb(k, 2)),  # 軸1頭: 5 opponents -> 10
    ("TRIO", 2, False, lambda k: k),  # 軸2頭
    ("QUINELLA", 1, False, lambda k: k),
    ("QUINELLA_PLACE", 1, False, lambda k: k),
]


@pytest.mark.parametrize(("pool", "positions", "multi", "count"), WHEELS)
def test_wheel_and_multi_counts_and_combinations(pool, positions, multi, count):
    n_axis = positions if isinstance(positions, int) else len(positions)
    axis_pool = [7, 13, 2]
    opponents_pool = [1, 16, 4, 10, 5, 18, 3, 11]
    axis = axis_pool[:n_axis]
    for k in range(max(1, ALL_POOLS[pool] - n_axis), len(opponents_pool) + 1):
        opponents = opponents_pool[:k]
        if multi:
            tickets = F.wheel(pool, axis, opponents, multi=True)
        elif is_ordered(pool):
            tickets = F.wheel(pool, axis, opponents, positions=positions)
        else:
            tickets = F.wheel(pool, axis, opponents)
        expected = ref_wheel(pool, axis, opponents, positions, multi)
        assert as_set(tickets) == expected, (pool, positions, multi, k)
        assert tickets.n_tickets == count(k) == len(expected), (pool, positions, multi, k)


def test_wheel_worked_examples():
    """Counted by hand: axis 3 first over {1,5,7} is 3·2 = 6 tickets; axis 3 multi over {1,5} is
    3 positions x 2·1 = 6; 馬単 axis 4 second over {1,2} is 2; 三連複 axis 2 over {1,5,9} is 3."""
    assert as_set(F.wheel("TIERCE", [3], [1, 5, 7], positions=[1])) == {
        (3, 1, 5),
        (3, 1, 7),
        (3, 5, 1),
        (3, 5, 7),
        (3, 7, 1),
        (3, 7, 5),
    }
    assert as_set(F.wheel("TIERCE", [3], [1, 5], multi=True)) == {
        (3, 1, 5),
        (3, 5, 1),
        (1, 3, 5),
        (5, 3, 1),
        (1, 5, 3),
        (5, 1, 3),
    }
    assert as_set(F.wheel("FORECAST", [4], [1, 2], positions=[2])) == {(1, 4), (2, 4)}
    assert as_set(F.wheel("TRIO", [2], [1, 5, 9])) == {(1, 2, 5), (1, 2, 9), (2, 5, 9)}
    assert F.wheel("TIERCE", [3], [1, 2, 4, 5, 6], multi=True).n_tickets == 60
    assert F.wheel("TIERCE", [3, 8], [1, 2, 4, 5, 6], multi=True).n_tickets == 30


def test_multi_on_an_unordered_pool_is_refused():
    with pytest.raises(ValueError):
        F.wheel("TRIO", [1], [2, 3, 4], multi=True)


# ---------------------------------------------------------------- expansion: formation


def test_formation_worked_examples_do_not_double_count_overlaps():
    """Counted by hand.

    三連単 1st {1,2} 2nd {1,2,3} 3rd {1,2,3,4,5}: first 1 -> second 2 or 3 -> 3 thirds each = 6;
    first 2 likewise 6; 12 tickets. 三連複 on the same sets: 123 124 125 134 135 234 235 = 7.
    三連単 A/B/C 1st {1} 2nd {2,3} 3rd {2,3,4,5,6}: 2 x 4 = 8. 馬単 {1,2} -> {1,2,3}: 12 13 21 23
    = 4. 馬連 {1,2} x {1,2,3}: 12 13 23 = 3.
    """
    t = F.formation("TIERCE", [[1, 2], [1, 2, 3], [1, 2, 3, 4, 5]])
    assert t.n_tickets == 12 and len(as_set(t)) == 12
    trio = F.formation("TRIO", [[1, 2], [1, 2, 3], [1, 2, 3, 4, 5]])
    assert as_set(trio) == {
        (1, 2, 3),
        (1, 2, 4),
        (1, 2, 5),
        (1, 3, 4),
        (1, 3, 5),
        (2, 3, 4),
        (2, 3, 5),
    }
    assert trio.n_tickets == 7
    assert F.formation("TIERCE", [[1], [2, 3], [2, 3, 4, 5, 6]]).n_tickets == 8
    assert as_set(F.formation("FORECAST", [[1, 2], [1, 2, 3]])) == {(1, 2), (1, 3), (2, 1), (2, 3)}
    assert as_set(F.formation("QUINELLA", [[1, 2], [1, 2, 3]])) == {(1, 2), (1, 3), (2, 3)}


@pytest.mark.parametrize("pool", sorted(ALL_POOLS))
def test_formation_matches_the_definition_on_varied_specs(pool):
    rng = random.Random(139 + len(pool))
    legs = ALL_POOLS[pool]
    for _ in range(25):
        field = rng.sample(range(1, 19), rng.randint(legs, 8))
        spec = [rng.sample(field, rng.randint(1, len(field))) for _ in range(legs)]
        expected = S.ref_formation(spec, is_ordered(pool))
        tickets = F.formation(pool, spec)
        assert as_set(tickets) == expected, spec
        assert tickets.n_tickets == len(expected), spec


@pytest.mark.parametrize("pool", ["WIN", "PLACE", "WIN5"])
def test_pools_without_combinations_are_refused(pool):
    with pytest.raises(ValueError):
        F.box(pool, [1, 2, 3])
    with pytest.raises(ValueError):
        F.formation(pool, [[1, 2], [3, 4]])


def test_cost_is_tickets_times_a_unit_of_100_yen_multiples():
    sixty = F.box("TIERCE", [1, 2, 3, 4, 5])
    eight = F.formation("TIERCE", [[1], [2, 3], [2, 3, 4, 5, 6]])
    assert sixty.cost(unit=100) == 6000
    assert sixty.cost(unit=300) == 18000
    assert eight.cost(unit=100) == 800
    assert eight.cost(unit=1000) == 8000
    for bad in (0, 50, 150, -100):
        with pytest.raises(ValueError):
            sixty.cost(unit=bad)
    assert inspect.signature(sixty.cost).parameters["unit"].default is inspect.Parameter.empty


# ---------------------------------------------------------------- EV: breakage and evaluation


@pytest.mark.parametrize(
    ("pool", "share", "rate", "expected"),
    [
        ("TIERCE", 0.01, 0.725, 7250),  # exactly 7250: no step lost to float error
        ("TIERCE", 0.0123, 0.725, 5890),  # 5894.3 floored to 10
        ("TIERCE", 0.05, 0.85, 1700),  # a Premium-boosted rate is the caller's
        ("TRIO", 0.2, 0.75, 370),  # 375 -> 370
        ("FORECAST", 0.003, 0.75, 25000),
        ("FORECAST", 0.0031, 0.75, 24190),  # 24193.5 -> 24190
        ("QUINELLA", 0.7, 0.775, 110),  # 110.7 -> 110
        ("QUINELLA", 0.9, 0.775, 110),  # 86.1 -> the 110 floor
    ],
)
def test_paid_dividend_applies_takeout_and_breakage(pool, share, rate, expected):
    assert F.paid_dividend(pool, share, rate=rate) == expected


@pytest.mark.parametrize(("share", "rate"), [(0.05, 0.775), (0.3, 0.775), (0.12, 0.85)])
def test_paid_dividend_refuses_wide_which_pays_three_pairs(share, rate):
    # Every Wide result pays three pairs, each its stake back plus a third of the losers' money:
    # one pair's share alone does not give its dividend (core.pricing.stake_back_share_dividend).
    with pytest.raises(ValueError):
        F.paid_dividend("QUINELLA_PLACE", share, rate=rate)


def test_six_trifectas_against_one_trio():
    """The research's L1 comparison, by hand. The six orders of {1,2,3} have p .05 .04 .03 .02
    .01 .01 (sum .16 = the trio's p) and pay 2000 3000 4000 5000 9000 12000 per 100: returns 100
    120 120 100 90 120 = 650 on 600 staked (profit 50, 650/600 per yen). The trio pays 700: 112 on
    100 (profit 12, 1.12 per yen). At 200 a point every yen figure doubles."""
    orders = [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
    probs = dict(zip(orders, [0.05, 0.04, 0.03, 0.02, 0.01, 0.01], strict=True))
    divs = dict(zip(orders, [2000, 3000, 4000, 5000, 9000, 12000], strict=True))
    for unit, scale in ((100, 1), (200, 2)):
        six = F.evaluate(F.box("TIERCE", [1, 2, 3]), probs, divs, unit=unit)
        assert six.pool == "TIERCE"
        assert six.n_tickets == 6
        assert six.cost == 600 * scale
        assert six.expected_return == pytest.approx(650 * scale)
        assert six.expected_profit == pytest.approx(50 * scale)
        assert six.ev_per_yen == pytest.approx(650 / 600)
        one = F.evaluate(F.box("TRIO", [3, 1, 2]), {(1, 2, 3): 0.16}, {(1, 2, 3): 700}, unit=unit)
        assert one.n_tickets == 1
        assert one.cost == 100 * scale
        assert one.expected_return == pytest.approx(112 * scale)
        assert one.expected_profit == pytest.approx(12 * scale)
        assert one.ev_per_yen == pytest.approx(1.12)


def test_evaluate_matches_the_definition_on_synthetic_races():
    for seed, pool, legsets in [
        (11, "TIERCE", [[1, 2], [1, 2, 3], [2, 3, 4, 5]]),
        (12, "TRIO", [[1], [2, 3], [2, 3, 4, 5]]),
        (13, "FORECAST", [[2, 5], [1, 3, 4, 6]]),
        (14, "QUINELLA", [[1, 2, 3], [4, 5]]),
    ]:
        n = 5 if S.LEGS[pool] == 3 else 6
        probs, divs = S.synthetic_race(seed, pool, n)
        tickets = F.formation(pool, legsets)
        combos = S.ref_formation(legsets, S.ORDERED[pool])
        ret = sum(probs[c] * divs[c] for c in combos) / 100 * 200
        got = F.evaluate(tickets, probs, divs, unit=200)
        assert got.n_tickets == len(combos)
        assert got.cost == 200 * len(combos)
        assert got.expected_return == pytest.approx(ret)
        assert got.expected_profit == pytest.approx(ret - 200 * len(combos))
        assert got.ev_per_yen == pytest.approx(ret / (200 * len(combos)))


# ---------------------------------------------------------------- the optimizer: brute force


def relabel(combo, ordered):
    return S.canon((LABELS[h] for h in combo), ordered)


SCENARIOS = [
    *[
        ("TIERCE", 5, seed, b)
        for seed in (1, 3, 4, 6)
        for b in ((1000, 100, 1.0), (2400, 200, 1.1), (100000, 100, 1.0))
    ],
    *[("TRIO", 5, seed, b) for seed in (1, 4, 5, 6) for b in ((1000, 100, 1.0), (2400, 200, 1.1))],
    *[("FORECAST", 6, seed, b) for seed in (1, 3, 5) for b in ((1000, 100, 1.0), (2400, 200, 1.1))],
    *[("QUINELLA", 6, seed, b) for seed in (3, 4, 6) for b in ((1000, 100, 1.0), (2400, 200, 1.1))],
]


@pytest.mark.parametrize(("pool", "n", "seed", "rules"), SCENARIOS)
@pytest.mark.parametrize("relabelled", [False, True])
def test_optimizer_finds_the_brute_force_best_formation(pool, n, seed, rules, relabelled):
    """Every per-position formation over the field is enumerated here; the optimizer must pick the
    feasible one with the largest expected profit, and report the best box and single the same
    way."""
    budget, unit, min_ev = rules
    ordered = S.ORDERED[pool]
    probs, divs = S.synthetic_race(seed, pool, n)
    field = list(range(1, n + 1))
    candidates = {
        "formation": S.all_formations(pool, n),
        "box": S.all_boxes(pool, n),
        "single": tuple(frozenset([c]) for c in probs),
    }
    if relabelled:
        probs = {relabel(c, ordered): p for c, p in probs.items()}
        divs = {relabel(c, ordered): d for c, d in divs.items()}
        field = [LABELS[h] for h in field]
        candidates = {
            kind: tuple(frozenset(relabel(c, ordered) for c in s) for s in sets)
            for kind, sets in candidates.items()
        }
    value = S.values(probs, divs)
    best = {
        kind: S.ranked(sets, value, budget=budget, unit=unit, min_ev=min_ev)
        for kind, sets in candidates.items()
    }
    top = best["formation"]
    assert top, "fixture: some formation must qualify"
    assert len(top) == 1 or top[0][0] - top[1][0] > 1e-6, "fixture: the best must be unique"

    choice = F.optimise_formation(pool, field, probs, divs, budget=budget, unit=unit, min_ev=min_ev)

    assert choice.pool == pool
    assert choice.shadow is True
    assert as_set(choice.chosen) == set(top[0][1])
    assert choice.chosen.n_tickets == len(top[0][1])
    assert choice.chosen.cost == unit * len(top[0][1]) <= budget
    assert choice.chosen.expected_profit == pytest.approx(top[0][0])
    assert as_set(F.formation(pool, choice.legs)) == set(top[0][1])
    for kind, attr in (("box", "best_box"), ("single", "best_single")):
        got = getattr(choice, attr)
        if not best[kind]:
            assert got is None, attr
            continue
        assert as_set(got) == set(best[kind][0][1]), attr
        assert got.expected_profit == pytest.approx(best[kind][0][0]), attr
        assert choice.chosen.expected_profit >= got.expected_profit - 1e-9


# (pool, budget, unit, min_ev, scenario seed), each brute-forced with a floor below 1. The fixture
# check in the test makes sure the floor matters: the best formation holds a combination worth
# less than 1 and beats the best formation at a floor of 1.
LOW_FLOOR_SCENARIOS = [
    ("TIERCE", 5, 30, (3000, 100, 0.7)),
    ("TIERCE", 5, 10, (2000, 100, 0.8)),
    ("TRIO", 5, 28, (2000, 100, 0.8)),
    ("TRIO", 5, 8, (3000, 100, 0.7)),
    ("FORECAST", 6, 18, (3000, 100, 0.7)),
    ("QUINELLA", 6, 9, (2000, 100, 0.8)),
]


@pytest.mark.parametrize(("pool", "n", "seed", "rules"), LOW_FLOOR_SCENARIOS)
def test_optimizer_brute_force_with_a_floor_below_one(pool, n, seed, rules):
    """min_ev < 1 admits losing combinations; the best formation may carry one to reach others.
    Every per-position formation is enumerated; the optimizer must return the brute-force best."""
    budget, unit, min_ev = rules
    probs, divs = S.synthetic_race(seed, pool, n)
    value = S.values(probs, divs)
    forms = S.all_formations(pool, n)
    top = S.ranked(forms, value, budget=budget, unit=unit, min_ev=min_ev)
    at_one = S.ranked(forms, value, budget=budget, unit=unit, min_ev=1.0)
    assert len(top) == 1 or top[0][0] - top[1][0] > 1e-6, "fixture: the best must be unique"
    assert any(value[c] < 1.0 for c in top[0][1]), "fixture: the best holds a value under 1"
    assert top[0][0] > at_one[0][0] + 1e-6, "fixture: the low floor must pay"

    choice = F.optimise_formation(
        pool, range(1, n + 1), probs, divs, budget=budget, unit=unit, min_ev=min_ev
    )

    assert as_set(choice.chosen) == set(top[0][1])
    assert choice.chosen.expected_profit == pytest.approx(top[0][0])
    assert min(value[c] for c in as_set(choice.chosen)) >= min_ev
    assert as_set(F.formation(pool, choice.legs)) == set(top[0][1])


# ---------------------------------------------------------------- the optimizer: planted races
#
# One big combination on horses B plus a cluster box on a disjoint set of horses C, every other
# combination below the floor. No feasible formation mixes the two: a formation holding the big
# combination and a cluster combination puts some cluster horse x in the leg of a big horse, and
# then (x with the other big horses in their legs) is a combination of the formation that is
# neither, so it is below the floor. The best formation is therefore the big combination alone or
# the best formation inside C; with every cluster combination +EV and the budget covering the
# whole cluster box, the latter is that box. So the optimum is whichever of the two gains more.


def cluster_box(pool, horses):
    return S.ref_formation([horses] * S.LEGS[pool], S.ORDERED[pool])


# (pool, big combination and its value in cents, cluster horses, cluster value in cents,
#  expected profit at 100 yen a point)
REVIEW_COUNTEREXAMPLES = [
    # 三連複, 9 runners: (1,2,3) at 11 (gain 10) against the 20 triples of 4-9 at 1.6 (gain 12).
    # Each cluster horse is in 10 triples, potential 6 < 10, so the big horses are searched first.
    ("TRIO", (1, 2, 3), 1100, range(4, 10), 160, 1200.0),
    # 三連単: (1,2,3) at 13 (gain 12) against the 24 orders of 4-7 at 1.6 (gain 14.4).
    ("TIERCE", (1, 2, 3), 1300, range(4, 8), 160, 1440.0),
]


@pytest.mark.parametrize(
    ("pool", "big", "big_cents", "cluster", "cluster_cents", "profit"), REVIEW_COUNTEREXAMPLES
)
def test_optimizer_prefers_a_moderate_cluster_over_one_big_combination(
    pool, big, big_cents, cluster, cluster_cents, profit
):
    """The reviewer's counterexamples (PR #207): every other combination is worth 0.5; budget
    5000, unit 100, min_ev 1.0. The optimum is the cluster box (see the section note), and the
    search must not cut it in favour of the big single."""
    cluster = list(cluster)
    box_combos = cluster_box(pool, cluster)
    cents = {c: cluster_cents for c in box_combos}
    cents[big] = big_cents
    probs, divs = S.planted_race(pool, range(1, 10), cents, other=50)

    choice = F.optimise_formation(
        pool, range(1, 10), probs, divs, budget=5000, unit=100, min_ev=1.0
    )

    assert as_set(choice.chosen) == box_combos
    assert choice.chosen.expected_profit == pytest.approx(profit)
    assert as_set(F.formation(pool, choice.legs)) == box_combos
    assert choice.best_single is not None and as_set(choice.best_single) == {big}
    assert choice.best_single.expected_profit == pytest.approx(big_cents - 100)
    assert choice.best_box is not None and as_set(choice.best_box) == box_combos
    assert choice.best_box.expected_profit == pytest.approx(profit)
    assert choice.chosen.expected_profit >= choice.best_box.expected_profit - 1e-9
    assert choice.chosen.expected_profit >= choice.best_single.expected_profit - 1e-9


CLUSTER_SIZES = {"TRIO": (5, 6), "TIERCE": (4,), "QUINELLA": (4, 5), "FORECAST": (3, 4)}
PLANTED = [
    *[(pool, seed) for pool in ("TRIO", "TIERCE") for seed in range(8)],
    *[(pool, seed) for pool in ("QUINELLA", "FORECAST") for seed in range(4)],
]


def planted_case(pool, seed):
    """A seeded planted race: (field, cents, big, cluster combos, unit, budget, min_ev,
    expected combos, expected profit in yen). Three seeds in four make the cluster win with the
    big combination still the strongest per horse (the adversarial shape); the fourth makes the
    big combination win."""
    rng = random.Random(1390 + 31 * seed + len(pool))
    k = S.LEGS[pool]
    m = rng.choice(CLUSTER_SIZES[pool])
    n = min(10, k + m + rng.randint(0, 2))
    field = rng.sample(range(1, 19), n)
    big = S.canon(rng.sample(field[:k], k), S.ORDERED[pool])
    combos = cluster_box(pool, field[k : k + m])
    cents = {c: rng.randint(110, 190) for c in combos}
    total = sum(v - 100 for v in cents.values())
    potential = max(sum(cents[c] - 100 for c in combos if h in c) for h in field[k : k + m])
    if seed % 4 != 3:
        assert potential + 2 < total - 2, "fixture: room for the adversarial big value"
        big_gain = rng.randint(potential + 2, total - 2)
    else:
        big_gain = rng.randint(total + 2, total + 200)
    cents[big] = 100 + big_gain
    unit = rng.choice((100, 200))
    budget = unit * (len(combos) + rng.randint(0, 10))
    min_ev = rng.choice((1.0, 1.05))
    other = rng.choice((10, 50, 90))
    if big_gain > total:
        expected, gain = {big}, big_gain
    else:
        expected, gain = combos, total
    return field, cents, other, big, combos, unit, budget, min_ev, expected, gain * unit / 100


@pytest.mark.parametrize(("pool", "seed"), PLANTED)
def test_optimizer_is_never_worse_than_the_best_box_or_single_on_planted_races(pool, seed):
    field, cents, other, big, combos, unit, budget, min_ev, expected, profit = planted_case(
        pool, seed
    )
    probs, divs = S.planted_race(pool, field, cents, other)

    choice = F.optimise_formation(pool, field, probs, divs, budget=budget, unit=unit, min_ev=min_ev)

    assert choice.best_single is not None and choice.best_box is not None
    assert choice.chosen.expected_profit >= choice.best_box.expected_profit - 1e-6
    assert choice.chosen.expected_profit >= choice.best_single.expected_profit - 1e-6
    assert as_set(choice.chosen) == expected
    assert choice.chosen.expected_profit == pytest.approx(profit)
    assert choice.chosen.cost <= budget
    assert as_set(choice.best_single) == {big}
    total = sum(cents[c] - 100 for c in combos) * unit / 100
    big_profit = (cents[big] - 100) * unit / 100
    # a 3-horse 三連複 box or a 2-horse 馬連 box is the big combination alone
    box_profit = total if S.ORDERED[pool] else max(total, big_profit)
    assert choice.best_box.expected_profit == pytest.approx(box_profit)


# A strong favourite 1 and six +EV orders, set by hand (dividend 1000 for all, so value = 10 p).
ABC_VALUES = {
    (1, 2, 3): 2.0,
    (1, 2, 4): 1.8,
    (1, 3, 2): 1.6,
    (1, 3, 4): 1.4,
    (1, 2, 5): 1.3,
    (1, 3, 5): 1.2,
}


def abc_race():
    rest = [t for t in itertools.permutations(range(1, 7), 3) if t not in ABC_VALUES]
    probs = {t: v / 10 for t, v in ABC_VALUES.items()}
    probs.update({t: 0.07 / len(rest) for t in rest})
    return probs, dict.fromkeys(probs, 1000)


@pytest.mark.parametrize(
    ("budget", "min_ev", "combos", "profit"),
    [
        # all six: the A/B/C formation 1 / {2,3} / {2,3,4,5}; (2+1.8+1.6+1.4+1.3+1.2-6) x 100
        (1000, 1.0, set(ABC_VALUES), 330.0),
        (600, 1.0, set(ABC_VALUES), 330.0),
        # four tickets: 1 / {2,3} / {2,3,4} = 2.8 beats 1 / {2,3} / {4,5} = 1.7 and any three
        (400, 1.0, {(1, 2, 3), (1, 2, 4), (1, 3, 2), (1, 3, 4)}, 280.0),
        # a 1.35 floor drops (1,2,5) and (1,3,5)
        (1000, 1.35, {(1, 2, 3), (1, 2, 4), (1, 3, 2), (1, 3, 4)}, 280.0),
        # three tickets: 1 / {2,3} / {3,4} = (2+1.8+1.4-3) = 2.2 beats 1 / 2 / {3,4,5} = 2.1
        (300, 1.0, {(1, 2, 3), (1, 2, 4), (1, 3, 4)}, 220.0),
        (100, 1.0, {(1, 2, 3)}, 100.0),
    ],
)
def test_optimizer_known_answer_abc_formation(budget, min_ev, combos, profit):
    probs, divs = abc_race()
    choice = F.optimise_formation(
        "TIERCE", range(1, 7), probs, divs, budget=budget, unit=100, min_ev=min_ev
    )
    assert as_set(choice.chosen) == combos
    assert choice.chosen.expected_profit == pytest.approx(profit)
    assert as_set(F.formation("TIERCE", choice.legs)) == combos
    assert choice.best_single is not None and as_set(choice.best_single) == {(1, 2, 3)}


def test_optimizer_returns_an_empty_choice_when_nothing_qualifies():
    probs, divs = abc_race()
    for budget, min_ev in ((99, 1.0), (0, 1.0), (1000, 2.5)):
        choice = F.optimise_formation(
            "TIERCE", range(1, 7), probs, divs, budget=budget, unit=100, min_ev=min_ev
        )
        assert choice.chosen.combinations == ()
        assert choice.chosen.n_tickets == 0
        assert choice.chosen.cost == 0
        assert choice.chosen.expected_profit == 0
        assert choice.legs == ()
        assert choice.best_single is None and choice.best_box is None
    # every value at most 0.8: a floor below 1 admits them, but no formation has a positive profit
    low = {c: 0.4 * p for c, p in probs.items()}
    choice = F.optimise_formation(
        "TIERCE", range(1, 7), low, divs, budget=10000, unit=100, min_ev=0.5
    )
    assert choice.chosen.n_tickets == 0
    assert choice.best_single is None and choice.best_box is None


def test_optimizer_ties_go_to_fewer_tickets_then_the_smaller_combinations():
    """馬連 on 1..5. (1,2) is worth 1.5, (3,4) and (3,5) 1.25 each, all exact in binary; the
    rest 0.011. {(1,2)} and the formation 3 / {4,5} both make 50 yen: the one ticket wins. With
    (1,3) and (2,4) both worth 1.5 and no formation holding both, the smaller (1,3) wins."""
    pairs = list(itertools.combinations(range(1, 6), 2))
    probs = dict.fromkeys(pairs, 0.01)
    divs = dict.fromkeys(pairs, 110)
    probs[(1, 2)], divs[(1, 2)] = 0.25, 600
    probs[(3, 4)], divs[(3, 4)] = 0.125, 1000
    probs[(3, 5)], divs[(3, 5)] = 0.125, 1000
    choice = F.optimise_formation(
        "QUINELLA", range(1, 6), probs, divs, budget=300, unit=100, min_ev=1.0
    )
    assert as_set(choice.chosen) == {(1, 2)}
    assert choice.chosen.expected_profit == pytest.approx(50.0)

    probs = dict.fromkeys(pairs, 0.01)
    divs = dict.fromkeys(pairs, 110)
    probs[(2, 4)], divs[(2, 4)] = 0.25, 600
    probs[(1, 3)], divs[(1, 3)] = 0.25, 600
    for _ in range(2):  # the same answer on a repeat call and whatever the mapping's order
        choice = F.optimise_formation(
            "QUINELLA", [5, 4, 3, 2, 1], probs, divs, budget=200, unit=100, min_ev=1.0
        )
        assert as_set(choice.chosen) == {(1, 3)}
        probs = dict(reversed(list(probs.items())))
        divs = dict(reversed(list(divs.items())))


def test_budget_unit_and_min_ev_are_required_owner_inputs():
    for fn, names in (
        (F.optimise_formation, ("budget", "unit", "min_ev")),
        (F.evaluate, ("unit",)),
    ):
        params = inspect.signature(fn).parameters
        for name in names:
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, (fn.__name__, name)
            assert params[name].default is inspect.Parameter.empty, (fn.__name__, name)
    probs, divs = abc_race()
    with pytest.raises(TypeError):
        F.optimise_formation("TIERCE", range(1, 7), probs, divs, unit=100, min_ev=1.0)
    with pytest.raises(TypeError):
        F.optimise_formation("TIERCE", range(1, 7), probs, divs, budget=1000, unit=100)
    with pytest.raises(ValueError):
        F.optimise_formation("WIN", range(1, 7), probs, divs, budget=1000, unit=100, min_ev=1.0)


# ---------------------------------------------------------------- shadow-only and logged


def log_args(race_no=11, tau=2):
    d = DEV_DAY.isoformat()
    return {
        "day": DEV_DAY,
        "race_id": f"jra:{d}:TOK:{race_no:02d}",
        "tau_min": tau,
        "decision_utc": f"{d}T06:38:00Z",
        "board_capture_utc": f"{d}T06:37:44Z",
        "written_utc": f"{d}T06:38:16Z",
        "mode": "replay",
        "models": {"ordering": "b2"},
        "git_sha": "0123abc",
    }


def test_choice_carries_no_stake_suggestion():
    probs, divs = abc_race()
    choice = F.optimise_formation(
        "TIERCE", range(1, 7), probs, divs, budget=400, unit=100, min_ev=1.0
    )
    assert dataclasses.is_dataclass(choice)
    assert choice.shadow is True
    assert stake_like_key(dataclasses.asdict(choice)) is None
    assert stake_like_key(dataclasses.asdict(choice.chosen)) is None


def test_choice_is_logged_to_the_sealed_shadow_log():
    probs, divs = abc_race()
    choice = F.optimise_formation(
        "TIERCE", range(1, 7), probs, divs, budget=400, unit=100, min_ev=1.0
    )
    assert F.log_choice(choice, **log_args()) == 1
    empty = F.optimise_formation(
        "TIERCE", range(1, 7), probs, divs, budget=1000, unit=100, min_ev=2.5
    )
    assert F.log_choice(empty, **log_args(race_no=12)) == 1

    path = shadow_log_path(DEV_DAY)
    assert path.is_file()
    records = ShadowLog(path, day=DEV_DAY).read()
    assert len(records) == 2
    first, second = records
    assert first["race_id"] == log_args()["race_id"]
    assert first["pool"] == "TIERCE"
    assert first["tau_min"] == 2
    assert first["n_tickets"] == 4
    assert sorted(first["combinations"]) == ["1,2,3", "1,2,4", "1,3,2", "1,3,4"]
    assert second["race_id"] == log_args(race_no=12)["race_id"]
    assert second["n_tickets"] == 0
    assert second["combinations"] == []
    for record in records:
        assert record["day_status"] == "development"
        assert stake_like_key(record) is None
    assert ShadowLog(path, day=DEV_DAY).verify().n_records == 2
