"""Exotic pool-choice rules L1, L4 and L5 on the same races (ticket #81; 08 §5.1, §8 A4).

Contract pinned here (`horsetrack.countries.japan.studies.avoidance`):
- Pool codes are the store's (jra.yaml, docs/specs/store.md): TIERCE is the trifecta, FORECAST the
  exacta; TRIO, QUINELLA and BRACKET_QUINELLA as named. Combinations are the dividends table's
  strings: horse numbers joined by ",", ascending by number in an unordered pool ("2,10,13"), in
  finishing order in an ordered pool ("13,2,10"), bracket numbers in BRACKET_QUINELLA ("3,3" is a
  same-bracket pair).
- `Ticket(pool, combination, stake)`, stake in yen. Dividends are per 100 yen (the source's unit
  stake): a ticket on a paying row of its race, pool and combination pays dividend * stake / 100;
  a ticket on an `is_refund` row pays its stake back, whatever that row's dividend.
- `trio_as_trifectas(trio)` and `quinella_as_exactas(pair)`: the ordered combinations covering the
  same event. `same_bracket_to_quinellas(bracket_pair, brackets)`: `brackets` maps horse number to
  bracket number; a same-bracket pair maps to every quinella on two horses of that bracket.
- `settle(tickets, dividends, race_id) -> (stake, payout)` over that race's rows only.
- `RaceView(race_id, board, selection)`: `board` has pool, combination and odds (decimal, stake
  included). `view_at(con, race_id, at_utc, *, rules)` is the board at t: odds_timeseries through
  `core.store.asof.read_asof` under the country's availability rules, that race only, the latest
  available row per (pool, combination); its selection never says look-ahead. A view of the final
  price on history carries the selection 'final (look-ahead)'.
- `express_set(view, pool, combination, *, threshold, stake)` (L1, E11): the direct pool (TRIO or
  QUINELLA) when the set's odds on the view's board are under `threshold`, the ordered synthetic
  (six TIERCE or two FORECAST tickets, `stake` split evenly) when over; no tickets when the board
  has no price for the set (a decision-time board not logged yet: 'data pending').
- `compare(race_ids, rule, comparator, *, view, dividends, clusters, n_boot, level, seed)`:
  `view(race_id) -> RaceView`; `rule(view)` and `comparator(view)` return their tickets, or None
  or [] where they do not cover the race. The paired set is the races both cover whose dividends
  can settle every ticket's pool. `diff` is, per paired race, rule payout / stake minus comparator
  payout / stake; `dropped` lists every other race of `race_ids`; `interval` is #69's
  `paired_bootstrap` on `diff`, or None when no race is paired; `selection` is the views' label,
  and views with different labels are a ValueError.
- `neutral_ratio(race_id, ordered_pool, unordered_pool)`: the ordered pool's payout rate over the
  unordered pool's, from `takeout.race_takeout` on the shipped boost calendar.

Expected values come from first principles: the event equivalences from permutations, payouts
from a reference settlement over plain dicts, intervals from `paired_bootstrap` on the reference
differences, rates from JRA's published takeouts. Every frame is synthetic; no date is a Japan
holdout day (on or after 2026-11-28).
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import itertools
from datetime import datetime, timedelta, timezone

import duckdb
import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.store.asof import load_availability
from horsetrack.countries.japan.studies.avoidance import (
    RaceView,
    Ticket,
    compare,
    express_set,
    neutral_ratio,
    quinella_as_exactas,
    same_bracket_to_quinellas,
    settle,
    trio_as_trifectas,
    view_at,
)
from horsetrack.countries.japan.takeout import TakeoutUnknown

NAN = float("nan")
UTC = timezone.utc
UNIT = 100  # yen: the dividends' unit stake
LOOK_AHEAD = "final (look-ahead)"
BOOT = {"n_boot": 400, "level": 0.95, "seed": 81}
DIVIDEND_COLUMNS = ["jurisdiction", "race_id", "pool", "combination", "dividend", "is_refund"]
BOARD_COLUMNS = ["pool", "combination", "odds"]
ORDERED = {"TIERCE": True, "FORECAST": True, "TRIO": False, "QUINELLA": False}
LEGS = {"TIERCE": 3, "TRIO": 3, "FORECAST": 2, "QUINELLA": 2}
SYNTHETIC = {"TRIO": "TIERCE", "QUINELLA": "FORECAST"}


# ---------------------------------------------------------------------------------------------
# Synthetic races, dividends and a reference settlement
# ---------------------------------------------------------------------------------------------


def rid(day: str, venue: str, race_no: int) -> str:
    return f"jra:{day}:{venue}:{race_no:02d}"


def meeting_of(race_id: str) -> str:
    _, day, venue, _ = race_id.split(":")
    return f"{day}:{venue}"


def clusters_for(race_ids) -> pd.Series:
    return pd.Series({r: meeting_of(r) for r in race_ids})


def joined(horses) -> str:
    return ",".join(str(int(h)) for h in horses)


def unordered(horses) -> str:
    return joined(sorted(int(h) for h in horses))


def combo(pool: str, horses) -> str:
    return joined(horses) if ORDERED[pool] else unordered(horses)


def horses_of(combination: str) -> list[int]:
    return [int(h) for h in combination.split(",")]


def result_rows(race_id: str, finish, prices: dict) -> list[tuple]:
    """The paying rows of `prices`' pools for a finishing order, as dividends_frame takes them."""
    return [
        (race_id, pool, combo(pool, finish[: LEGS[pool]]), d, False) for pool, d in prices.items()
    ]


def dividends_frame(rows) -> pd.DataFrame:
    """The store's dividends table (docs/specs/store.md) holding `rows`."""
    return pd.DataFrame(
        [
            ("jra", race, pool, combination, NAN if d is None else float(d), bool(refund))
            for race, pool, combination, d, refund in rows
        ],
        columns=DIVIDEND_COLUMNS,
    )


def paying(rows) -> dict:
    """race -> {(pool, combination): (dividend, is_refund)}, from the rows themselves."""
    out: dict = {}
    for race, pool, combination, d, refund in rows:
        out.setdefault(race, {})[(pool, combination)] = (d, refund)
    return out


def ref_settle(tickets, pays: dict) -> tuple[int, float]:
    """Reference settlement of (pool, combination, stake) tickets against one race's rows."""
    stake, payout = 0, 0.0
    for pool, combination, yen in tickets:
        stake += yen
        hit = pays.get((pool, combination))
        if hit is not None:
            dividend, refund = hit
            payout += yen if refund else dividend * yen / UNIT
    return stake, payout


def ref_return(tickets, pays: dict) -> float:
    stake, payout = ref_settle(tickets, pays)
    return payout / stake


def as_tickets(specs) -> list | None:
    if specs is None:
        return None
    return [Ticket(pool, combination, stake) for pool, combination, stake in specs]


def spec_of(tickets) -> list[tuple]:
    return sorted((t.pool, t.combination, t.stake) for t in tickets)


def board(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=BOARD_COLUMNS)


EMPTY_BOARD = board([])


def history_view(race_id: str) -> RaceView:
    return RaceView(race_id=race_id, board=EMPTY_BOARD, selection=LOOK_AHEAD)


def strategy(by_race: dict):
    """A strategy betting fixed tickets per race: None (or absent) where it does not cover it."""
    return lambda view: as_tickets(by_race.get(view.race_id))


def assert_interval(got, diff: pd.Series, clusters: pd.Series) -> None:
    want = paired_bootstrap(diff, clusters, **BOOT)
    assert got.n_clusters == want.n_clusters
    for name in ("mean", "lo", "hi"):
        assert getattr(got, name) == pytest.approx(getattr(want, name), rel=1e-9, abs=1e-12)


def assert_diff(got: pd.Series, want: dict) -> None:
    assert isinstance(got, pd.Series)
    assert sorted(got.index) == sorted(want)
    for race, value in want.items():
        assert float(got[race]) == pytest.approx(value, rel=1e-12, abs=1e-12), race


# ---------------------------------------------------------------------------------------------
# Settlement against the dividends table
# ---------------------------------------------------------------------------------------------


def test_settle_pays_each_ticket_its_own_race_pool_and_combination():
    race, heat, other = (rid("2026-06-07", "TOK", n) for n in (11, 12, 10))
    rows = [
        (race, "WIN", "7", 350, False),
        (race, "BRACKET_QUINELLA", "2,4", 980, False),
        (race, "QUINELLA", "3,7", 1240, False),
        (race, "FORECAST", "7,3", 2890, False),
        (race, "TRIO", "3,7,12", 5630, False),
        (race, "TIERCE", "7,3,12", 31250, False),
        # a dead heat for third: two paying combinations in each three-horse pool
        (heat, "TRIO", "1,4,9", 2100, False),
        (heat, "TRIO", "1,4,11", 1880, False),
        (heat, "TIERCE", "4,1,9", 9800, False),
        (heat, "TIERCE", "4,1,11", 8700, False),
        # another race paying the same combinations at other prices
        (other, "TRIO", "3,7,12", 990, False),
        (other, "QUINELLA", "3,7", 410, False),
    ]
    dividends = dividends_frame(rows)
    cases = [  # (race, tickets, stake, payout), payouts worked by hand
        (race, [("TRIO", "3,7,12", 100)], 100, 5630.0),
        (race, [("TRIO", "3,7,12", 300)], 300, 16890.0),
        (race, [("TIERCE", "7,3,12", 100)], 100, 31250.0),
        (race, [("TIERCE", "3,7,12", 100)], 100, 0.0),  # the trio's string, not the order
        (race, [("TIERCE", "12,3,7", 500)], 500, 0.0),
        (race, [("QUINELLA", "3,7", 200), ("FORECAST", "7,3", 100)], 300, 2480.0 + 2890.0),
        (race, [("FORECAST", "3,7", 100), ("QUINELLA", "3,12", 100)], 200, 0.0),
        (race, [("BRACKET_QUINELLA", "2,4", 100), ("BRACKET_QUINELLA", "4,4", 100)], 200, 980.0),
        (race, [("WIN", "7", 1000), ("WIN", "3", 1000)], 2000, 3500.0),
        (race, [], 0, 0.0),
        (
            heat,
            [("TRIO", "1,4,9", 100), ("TRIO", "1,4,11", 100), ("TRIO", "1,9,11", 100)],
            300,
            2100.0 + 1880.0,
        ),
        (heat, [("TIERCE", "4,1,11", 200), ("TIERCE", "1,4,11", 100)], 300, 17400.0),
        (other, [("TRIO", "3,7,12", 100)], 100, 990.0),
        (other, [("QUINELLA", "3,7", 300), ("TIERCE", "7,3,12", 100)], 400, 1230.0),
    ]
    for race_id, specs, want_stake, want_payout in cases:
        stake, payout = settle(as_tickets(specs), dividends, race_id)
        assert stake == want_stake, (race_id, specs)
        assert payout == pytest.approx(want_payout, abs=1e-9), (race_id, specs)
        # the reference settlement agrees with the hand-worked figures
        assert ref_settle(specs, paying(rows)[race_id]) == (want_stake, pytest.approx(want_payout))


def test_refunded_tickets_return_their_stake():
    race, other = rid("2026-06-07", "HSN", 5), rid("2026-06-07", "HSN", 6)
    rows = [
        (race, "TRIO", "2,5,8", 4410, False),
        (race, "TIERCE", "8,2,5", 26800, False),
        # horse 13 was scratched after sales opened: its combinations are refunded
        (race, "TRIO", "2,5,13", None, True),
        (race, "TIERCE", "13,2,5", 100, True),
        (race, "QUINELLA", "5,13", 0, True),
        (race, "QUINELLA", "2,5", 1320, False),
        (other, "TRIO", "1,2,3", None, True),  # another race's refund refunds nothing here
    ]
    dividends = dividends_frame(rows)
    cases = [  # (tickets, stake, payout)
        ([("TRIO", "2,5,13", 300)], 300, 300.0),
        ([("TIERCE", "13,2,5", 200)], 200, 200.0),
        ([("QUINELLA", "5,13", 700)], 700, 700.0),
        ([("TRIO", "2,5,8", 100), ("TRIO", "2,5,13", 500), ("TRIO", "1,2,3", 100)], 700, 4910.0),
        ([("TIERCE", "13,2,5", 100), ("TIERCE", "2,13,5", 100)], 200, 100.0),  # only its order
        ([("QUINELLA", "2,5", 200), ("QUINELLA", "5,13", 100)], 300, 2640.0 + 100.0),
    ]
    for specs, want_stake, want_payout in cases:
        stake, payout = settle(as_tickets(specs), dividends, race)
        assert stake == want_stake, specs
        assert payout == pytest.approx(want_payout, abs=1e-9), specs
    stake, payout = settle(as_tickets([("TRIO", "1,2,3", 400)]), dividends, other)
    assert (stake, payout) == (400, pytest.approx(400.0))


# ---------------------------------------------------------------------------------------------
# Ticket equivalences: the same event in two pools
# ---------------------------------------------------------------------------------------------

RUNNERS = (2, 5, 9, 10, 13, 16)  # two-digit numbers: a string sort would misplace them


def test_six_trifectas_and_one_trio_cover_the_same_event():
    trios = [unordered(c) for c in itertools.combinations(RUNNERS, 3)]
    synthetic = {trio: trio_as_trifectas(trio) for trio in trios}
    for trio, trifectas in synthetic.items():
        assert len(trifectas) == 6 and len(set(trifectas)) == 6, trio
        assert set(trifectas) == {joined(p) for p in itertools.permutations(horses_of(trio))}

    # Every finish order of the six runners: the trio wins iff exactly one of its trifectas wins.
    for finish in itertools.permutations(RUNNERS):
        winner = joined(finish[:3])
        for trio, trifectas in synthetic.items():
            wins = set(finish[:3]) == set(horses_of(trio))
            assert sum(t == winner for t in trifectas) == (1 if wins else 0), (finish, trio)

    # Settled against each result's dividends, the trio pays iff its six trifectas pay.
    race = rid("2026-06-07", "TOK", 9)
    for k, top3 in enumerate(itertools.permutations(RUNNERS, 3)):
        dividends = dividends_frame(result_rows(race, top3, {"TRIO": 500 + 10 * k, "TIERCE": 3000}))
        for trio, trifectas in synthetic.items():
            wins = set(top3) == set(horses_of(trio))
            _, direct = settle([Ticket("TRIO", trio, 100)], dividends, race)
            _, six = settle([Ticket("TIERCE", t, 100) for t in trifectas], dividends, race)
            assert direct == pytest.approx(500 + 10 * k if wins else 0.0), (top3, trio)
            assert six == pytest.approx(3000.0 if wins else 0.0), (top3, trio)


def test_two_exactas_and_one_quinella_cover_the_same_event():
    pairs = [unordered(c) for c in itertools.combinations(RUNNERS, 2)]
    synthetic = {pair: quinella_as_exactas(pair) for pair in pairs}
    for pair, exactas in synthetic.items():
        a, b = horses_of(pair)
        assert sorted(exactas) == sorted([joined((a, b)), joined((b, a))]), pair

    for finish in itertools.permutations(RUNNERS):
        winner = joined(finish[:2])
        for pair, exactas in synthetic.items():
            wins = set(finish[:2]) == set(horses_of(pair))
            assert sum(e == winner for e in exactas) == (1 if wins else 0), (finish, pair)

    race = rid("2026-06-07", "TOK", 8)
    for k, top2 in enumerate(itertools.permutations(RUNNERS, 2)):
        prices = {"QUINELLA": 300 + 10 * k, "FORECAST": 1170}
        dividends = dividends_frame(result_rows(race, top2, prices))
        for pair, exactas in synthetic.items():
            wins = set(top2) == set(horses_of(pair))
            _, direct = settle([Ticket("QUINELLA", pair, 100)], dividends, race)
            _, two = settle([Ticket("FORECAST", e, 100) for e in exactas], dividends, race)
            assert direct == pytest.approx(300 + 10 * k if wins else 0.0), (top2, pair)
            assert two == pytest.approx(1170.0 if wins else 0.0), (top2, pair)


def test_the_synthetic_trio_stakes_six_units():
    trio, race = "3,8,14", rid("2026-06-07", "KYO", 11)
    for k, order in enumerate(itertools.permutations((3, 8, 14))):
        tierce, trio_dividend = 4000 + 1370 * k, 900 + 40 * k
        finish = (*order, 1, 2)
        dividends = dividends_frame(
            result_rows(race, finish, {"TRIO": trio_dividend, "TIERCE": tierce})
        )
        trifectas = trio_as_trifectas(trio)

        stake, payout = settle([Ticket("TIERCE", t, 100) for t in trifectas], dividends, race)
        assert stake == 600
        assert payout == pytest.approx(tierce)  # the one winning trifecta's dividend
        stake, payout = settle([Ticket("TIERCE", t, 200) for t in trifectas], dividends, race)
        assert (stake, payout) == (1200, pytest.approx(2 * tierce))
        # the direct comparator at the same stake
        stake, payout = settle([Ticket("TRIO", trio, 600)], dividends, race)
        assert (stake, payout) == (600, pytest.approx(6 * trio_dividend))

        # L1's synthetic expression of the set is those six 100-yen tickets
        long = RaceView(race, board([("TRIO", trio, 500.0)]), LOOK_AHEAD)
        tickets = express_set(long, "TRIO", trio, threshold=100.0, stake=600)
        assert spec_of(tickets) == sorted(("TIERCE", t, 100) for t in trifectas)
        assert settle(tickets, dividends, race) == (600, pytest.approx(tierce))

    # A set that misses: six tickets staked, nothing paid.
    dividends = dividends_frame(result_rows(race, (3, 8, 5, 14), {"TRIO": 700, "TIERCE": 2900}))
    stake, payout = settle(
        [Ticket("TIERCE", t, 100) for t in trio_as_trifectas(trio)], dividends, race
    )
    assert (stake, payout) == (600, 0.0)


def jra_brackets(n: int) -> dict[int, int]:
    """Horse number -> bracket as JRA draws them: one horse per bracket up to 8 runners; from 9,
    n // 8 horses in each bracket and one more in each of the last n % 8, in horse order."""
    if n <= 8:
        return {h: h for h in range(1, n + 1)}
    base, extra = divmod(n, 8)
    out, horse = {}, 1
    for bracket in range(1, 9):
        for _ in range(base + (bracket > 8 - extra)):
            out[horse] = bracket
            horse += 1
    return out


def test_same_bracket_quinella_maps_to_the_quinellas_within_the_bracket():
    assert jra_brackets(16)[5] == jra_brackets(16)[6] == 3  # bracket 3 holds horses 5 and 6
    assert same_bracket_to_quinellas("3,3", jra_brackets(16)) == ["5,6"]

    for n in (9, 10, 12, 14, 15, 16, 17, 18):
        brackets = jra_brackets(n)
        for bracket in range(1, 9):
            members = sorted(h for h, b in brackets.items() if b == bracket)
            if len(members) < 2:
                continue
            want = sorted(unordered(p) for p in itertools.combinations(members, 2))
            got = same_bracket_to_quinellas(f"{bracket},{bracket}", brackets)
            assert sorted(got) == want and len(got) == len(set(got)), (n, bracket)

    # The brackets come from the mapping given, not from the horse numbers.
    drawn = {1: 1, 4: 3, 9: 3, 12: 3, 2: 5, 7: 6, 11: 6}
    assert sorted(same_bracket_to_quinellas("3,3", drawn)) == ["4,12", "4,9", "9,12"]
    assert same_bracket_to_quinellas("6,6", drawn) == ["7,11"]

    # The same event: on every top two of 18 runners, a same-bracket pair wins iff exactly one
    # of its quinellas wins.
    brackets = jra_brackets(18)
    mapped = {b: same_bracket_to_quinellas(f"{b},{b}", brackets) for b in range(1, 9)}
    for a, b in itertools.permutations(range(1, 19), 2):
        winner = unordered((a, b))
        for bracket, quinellas in mapped.items():
            wins = brackets[a] == brackets[b] == bracket
            assert sum(q == winner for q in quinellas) == (1 if wins else 0), (a, b, bracket)


# ---------------------------------------------------------------------------------------------
# compare: the paired difference over the races both strategies cover
# ---------------------------------------------------------------------------------------------

POOLS = ("TRIO", "TIERCE", "QUINELLA", "FORECAST")


def random_race(rng, race_id: str) -> tuple[int, list[tuple]]:
    """A field size and the race's paying rows: the four exotic pools, sometimes a refund."""
    n = int(rng.integers(8, 17))
    finish = [int(h) for h in rng.permutation(np.arange(1, n + 1))]
    rows = result_rows(race_id, finish, {p: int(rng.integers(11, 5000)) * 10 for p in POOLS})
    if rng.random() < 0.3:  # a scratch refunds a combination
        pool = POOLS[int(rng.integers(len(POOLS)))]
        taken = {c for _, p, c, _, _ in rows if p == pool}
        while (
            refunded := combo(pool, rng.choice(np.arange(1, n + 1), LEGS[pool], False))
        ) in taken:
            pass
        rows.append((race_id, pool, refunded, None if rng.random() < 0.5 else 100, True))
    return n, rows


def random_tickets(rng, n: int, rows: list[tuple]) -> list[tuple]:
    specs = []
    for _ in range(int(rng.integers(1, 4))):
        if rng.random() < 0.45:  # a paying or refunded row's combination
            _, pool, combination, _, _ = rows[int(rng.integers(len(rows)))]
        else:
            pool = POOLS[int(rng.integers(len(POOLS)))]
            combination = combo(pool, rng.choice(np.arange(1, n + 1), LEGS[pool], replace=False))
        specs.append((pool, combination, int(rng.choice([100, 200, 300, 500, 1000]))))
    return specs


@pytest.mark.parametrize("seed", [811, 812, 813])
def test_compare_scores_every_race_both_strategies_cover(seed):
    rng = np.random.default_rng(seed)
    days, venues = ("2026-05-30", "2026-05-31", "2026-06-06", "2026-06-07"), ("KYO", "TOK")
    race_ids = sorted(rid(d, v, k) for d in days for v in venues for k in range(1, 7))
    rows, rule_bets, comparator_bets = [], {}, {}
    for race in race_ids:
        n, race_rows = random_race(rng, race)
        rows += race_rows
        for bets in (rule_bets, comparator_bets):
            if rng.random() < 0.75:
                bets[race] = random_tickets(rng, n, race_rows)
            elif rng.random() < 0.5:
                bets[race] = []  # covers nothing, like None
    pays = paying(rows)
    paired = [r for r in race_ids if rule_bets.get(r) and comparator_bets.get(r)]
    want = {
        r: ref_return(rule_bets[r], pays[r]) - ref_return(comparator_bets[r], pays[r])
        for r in paired
    }
    assert 10 < len(paired) < len(race_ids)  # the draw pairs some races and drops others
    clusters = clusters_for(race_ids)

    got = compare(
        race_ids,
        strategy(rule_bets),
        strategy(comparator_bets),
        view=history_view,
        dividends=dividends_frame(rows),
        clusters=clusters,
        **BOOT,
    )

    assert got.n_races == len(paired)
    assert got.dropped == [r for r in race_ids if r not in want]
    assert_diff(got.diff, want)
    assert_interval(got.interval, pd.Series(want), clusters)
    assert got.selection == LOOK_AHEAD


def test_paired_difference_uses_identical_races():
    r1, r2 = rid("2026-06-06", "TOK", 1), rid("2026-06-06", "TOK", 2)
    r3, r4 = rid("2026-06-07", "TOK", 3), rid("2026-06-07", "TOK", 4)
    rows = [
        (r1, "TRIO", "1,4,7", 1200, False),
        (r1, "QUINELLA", "1,4", 300, False),
        *result_rows(
            r2, (5, 2, 8), {"TRIO": 750, "QUINELLA": 300, "FORECAST": 610, "TIERCE": 3900}
        ),
        *result_rows(r3, (9, 3, 6), {"TRIO": 2480, "TIERCE": 15000, "QUINELLA": 880}),
        (r4, "QUINELLA", "1,2", 450, False),
    ]
    rule = {  # covers races 1, 2 and 3
        r1: [("TRIO", "1,4,7", 100)],
        r2: [("TRIO", "2,5,8", 200)],
        r3: [("TIERCE", "3,6,9", 100), ("TIERCE", "6,3,9", 100)],
    }
    comparator = {  # covers races 2, 3 and 4
        r2: [("QUINELLA", "2,5", 100), ("FORECAST", "2,5", 100)],
        r3: [("TRIO", "3,6,9", 300)],
        r4: [("QUINELLA", "1,2", 100)],
    }
    pays = paying(rows)
    want = {r: ref_return(rule[r], pays[r]) - ref_return(comparator[r], pays[r]) for r in (r2, r3)}
    assert want[r2] == pytest.approx(1500 / 200 - 300 / 200)  # a ratio of returns, not yen
    assert want[r3] == pytest.approx(0.0 - 7440 / 300)
    clusters = clusters_for([r1, r2, r3, r4])

    got = compare(
        [r1, r2, r3, r4],
        strategy(rule),
        strategy(comparator),
        view=history_view,
        dividends=dividends_frame(rows),
        clusters=clusters,
        **BOOT,
    )

    assert got.n_races == 2
    assert got.dropped == [r1, r4]
    assert_diff(got.diff, want)
    assert_interval(got.interval, pd.Series(want), clusters)
    assert got.interval.n_clusters == 2


def test_races_that_cannot_be_settled_are_dropped():
    paid, lost = rid("2026-06-07", "NAK", 11), rid("2026-06-07", "NAK", 12)
    no_result, unsold = rid("2026-06-07", "NAK", 9), rid("2026-06-07", "NAK", 10)
    rows = [
        (paid, "BRACKET_QUINELLA", "3,3", 1890, False),
        (paid, "QUINELLA", "5,6", 2210, False),
        # bracket 3 lost: the pool settled, so the race is kept at a zero return
        (lost, "BRACKET_QUINELLA", "1,4", 760, False),
        (lost, "QUINELLA", "1,7", 690, False),
        # an 8-runner race: no bracket quinella was sold, so the rule's ticket cannot be settled
        (unsold, "QUINELLA", "5,6", 1430, False),
        (unsold, "TRIO", "2,5,6", 3100, False),
        # `no_result` has no dividend rows at all
    ]
    races = [no_result, unsold, paid, lost]
    zorome = {r: [("BRACKET_QUINELLA", "3,3", 100)] for r in races}  # rule
    pair = {r: [("QUINELLA", "5,6", 100)] for r in races}  # comparator: the Quinella on the pair

    got = compare(
        races,
        strategy(zorome),
        strategy(pair),
        view=history_view,
        dividends=dividends_frame(rows),
        clusters=clusters_for(races),
        **BOOT,
    )

    assert got.n_races == 2
    assert got.dropped == [no_result, unsold]
    assert_diff(got.diff, {paid: 18.90 - 22.10, lost: 0.0 - 0.0})


def test_a_comparison_without_a_paired_race_drops_every_race():
    """A decision-time run before exotic boards are logged: no set has a price at t, so L1 covers
    no race ('data pending'), and the comparison says so instead of failing."""
    races = [rid("2026-06-07", "TOK", k) for k in (1, 2, 3)]
    sets = dict(zip(races, ("1,2,3", "4,8,11", "2,6,9"), strict=True))
    win_only = board([("WIN", "1", 2.4), ("WIN", "2", 5.1), ("WIN", "3", 9.8)])
    rows = [r for race in races for r in result_rows(race, (1, 2, 3), {"TRIO": 300, "TIERCE": 900})]

    got = compare(
        races,
        lambda v: express_set(v, "TRIO", sets[v.race_id], threshold=40.0, stake=600),
        lambda v: [Ticket("TRIO", sets[v.race_id], 600)],
        view=lambda r: RaceView(race_id=r, board=win_only, selection="board at t (T-2)"),
        dividends=dividends_frame(rows),
        clusters=clusters_for(races),
        **BOOT,
    )

    assert got.n_races == 0
    assert len(got.diff) == 0
    assert got.dropped == races
    assert got.interval is None


def test_the_comparison_carries_its_views_selection():
    races = [rid("2026-06-06", "HSN", 1), rid("2026-06-06", "HSN", 2)]
    rows = [
        r for race in races for r in result_rows(race, (4, 1, 6), {"TRIO": 520, "TIERCE": 2600})
    ]
    rule = {r: [("TIERCE", t, 100) for t in trio_as_trifectas("1,4,6")] for r in races}
    comparator = {r: [("TRIO", "1,4,6", 600)] for r in races}
    kwargs = {
        "dividends": dividends_frame(rows),
        "clusters": clusters_for(races),
        **BOOT,
    }

    for label in (LOOK_AHEAD, "board at t (T-2)"):
        got = compare(
            races,
            strategy(rule),
            strategy(comparator),
            view=lambda r, label=label: RaceView(race_id=r, board=EMPTY_BOARD, selection=label),
            **kwargs,
        )
        assert got.selection == label
        assert got.n_races == 2

    mixed = dict(zip(races, (LOOK_AHEAD, "board at t (T-2)"), strict=True))
    with pytest.raises(ValueError):
        compare(
            races,
            strategy(rule),
            strategy(comparator),
            view=lambda r: RaceView(race_id=r, board=EMPTY_BOARD, selection=mixed[r]),
            **kwargs,
        )


# ---------------------------------------------------------------------------------------------
# L1: express a set in the direct pool when short, in the ordered synthetic when long
# ---------------------------------------------------------------------------------------------

EXPRESS_CASES = [  # (pool, set, odds on the board, threshold, stake, expressed in)
    ("TRIO", "1,4,7", 9.9, 10.0, 600, "direct"),
    ("TRIO", "1,4,7", 10.1, 10.0, 600, "synthetic"),
    ("TRIO", "1,4,7", 10.1, 12.5, 600, "direct"),
    ("TRIO", "3,11,15", 250.0, 300.0, 1200, "direct"),
    ("TRIO", "3,11,15", 350.0, 300.0, 1200, "synthetic"),
    ("QUINELLA", "6,12", 4.0, 25.0, 200, "direct"),
    ("QUINELLA", "6,12", 30.0, 25.0, 200, "synthetic"),
    ("QUINELLA", "6,12", 30.0, 50.0, 600, "direct"),
    ("QUINELLA", "2,9", 75.5, 60.0, 600, "synthetic"),
]


@pytest.mark.parametrize(
    ("pool", "combination", "odds", "threshold", "stake", "kind"), EXPRESS_CASES
)
def test_express_set_follows_the_price_rule(pool, combination, odds, threshold, stake, kind):
    horses = horses_of(combination)
    decoy = combo(pool, [*horses[:-1], horses[-1] + 1])  # another set, priced the other way
    decoy_odds = threshold / 3 if kind == "synthetic" else threshold * 3
    ordered = [joined(p) for p in itertools.permutations(horses)]
    rows = [
        ("WIN", str(horses[0]), 3.1),
        (pool, decoy, decoy_odds),
        *((SYNTHETIC[pool], c, odds * 4.0) for c in ordered),
        (pool, combination, odds),
    ]
    view = RaceView(rid("2026-06-07", "TOK", 7), board(rows), "board at t (T-2)")

    got = express_set(view, pool, combination, threshold=threshold, stake=stake)

    if kind == "direct":
        assert spec_of(got) == [(pool, combination, stake)]
    else:
        k = len(ordered)  # 6 trifectas or 2 exactas, the stake split evenly
        assert spec_of(got) == sorted((SYNTHETIC[pool], c, stake // k) for c in ordered)

    # No price for the set on the board (not logged yet, or unpriced): no tickets.
    missing = RaceView(view.race_id, board([r for r in rows if r[1] != combination]), "t")
    assert not express_set(missing, pool, combination, threshold=threshold, stake=stake)
    unpriced = RaceView(view.race_id, board([*rows[:-1], (pool, combination, NAN)]), "t")
    assert not express_set(unpriced, pool, combination, threshold=threshold, stake=stake)


ODDS_LABELS = ("dividend", "is_refund", "is_final_capture")


def odds_frame(captures, post: datetime) -> pd.DataFrame:
    """odds_timeseries rows (logger/compact.py SCHEMA) from (race, pool, combination, snapshot,
    fetched, odds, final, dividend) captures."""
    frame = pd.DataFrame(
        [
            {
                "race_id": race,
                "venue": race.split(":")[2],
                "race_no": int(race.split(":")[3]),
                "pool": pool,
                "combination": combination,
                "snapshot_utc": snap,
                "fetched_utc": fetched,
                "post_utc": post,
                "minutes_to_post": (post - snap).total_seconds() / 60,
                "odds": odds,
                "odds_hi": NAN,
                "investment": 1000.0,
                "status_raw": "",
                "is_final_capture": final,
                "dividend": dividend,
                "is_refund": False,
            }
            for race, pool, combination, snap, fetched, odds, final, dividend in captures
        ]
    )
    for column in ("snapshot_utc", "fetched_utc", "post_utc"):
        frame[column] = pd.Series(pd.to_datetime(list(frame[column]), utc=True))
    return frame


def test_selection_reads_the_board_at_t_not_the_final():
    """A set short at t and long at the final is expressed in the direct pool; one long at t and
    short at the final in the synthetic. The board at t is odds_timeseries through read_asof."""
    day = datetime(2026, 6, 7, tzinfo=UTC)

    def at(hh, mm, ss=0):
        return day + timedelta(hours=hh, minutes=mm, seconds=ss)

    t, post = at(5, 55), at(6, 2)
    short, long_, pending = (rid("2026-06-07", "TOK", n) for n in (11, 12, 10))
    elsewhere = rid("2026-06-07", "KYO", 11)
    sets = {short: "2,9,14", long_: "1,6,10", pending: "1,3,4"}
    captures = [
        # short at t (21.0), long once fetched after t (95.0) and at the final (88.4)
        (short, "TRIO", "2,9,14", at(5, 30), at(5, 30, 4), 18.5, False, 8900.0),
        (short, "TRIO", "2,9,14", at(5, 50), at(5, 50, 3), 21.0, False, 8900.0),
        (short, "TRIO", "2,9,14", at(5, 54), at(5, 56, 10), 95.0, False, 8900.0),
        (short, "TRIO", "2,9,14", at(6, 1), at(6, 1, 5), 88.4, True, 8900.0),
        (short, "WIN", "9", at(5, 50), at(5, 50, 3), 3.2, False, 450.0),
        # long at t (60.0), short once fetched after t (12.5) and at the final (12.0)
        (long_, "TRIO", "1,6,10", at(5, 40), at(5, 40, 2), 64.0, False, 1450.0),
        (long_, "TRIO", "1,6,10", at(5, 52), at(5, 52, 1), 60.0, False, 1450.0),
        (long_, "TRIO", "1,6,10", at(5, 53), at(5, 58), 12.5, False, 1450.0),
        (long_, "TRIO", "1,6,10", at(6, 1), at(6, 1, 6), 12.0, True, 1450.0),
        # no exotic board logged for this race: WIN only
        (pending, "WIN", "3", at(5, 45), at(5, 45, 2), 2.2, False, 220.0),
        (pending, "WIN", "3", at(6, 1), at(6, 1, 4), 2.0, True, 220.0),
        # another race, the same combinations, later captures before t: never on these boards
        (elsewhere, "TRIO", "2,9,14", at(5, 53), at(5, 53, 30), 150.0, False, 0.0),
        (elsewhere, "TRIO", "1,6,10", at(5, 54), at(5, 54, 1), 9.0, False, 0.0),
    ]
    frame = odds_frame(captures, post)
    con = duckdb.connect(":memory:")
    con.register("_frame", frame)
    con.execute('CREATE TABLE "odds_timeseries" AS SELECT * FROM _frame')
    con.unregister("_frame")
    rules = load_availability("japan")

    view = view_at(con, short, t, rules=rules)

    assert view.race_id == short
    assert "look-ahead" not in view.selection
    for column in ODDS_LABELS:
        assert column not in view.board.columns, f"{column} is on the board at t"
    assert not view.board.duplicated(["pool", "combination"]).any()
    if "race_id" in view.board.columns:
        assert set(view.board["race_id"]) == {short}
    on_board = view.board.set_index(["pool", "combination"])["odds"]
    assert float(on_board[("TRIO", "2,9,14")]) == 21.0
    assert float(on_board[("WIN", "9")]) == 3.2

    threshold = 40.0
    tickets = express_set(view, "TRIO", "2,9,14", threshold=threshold, stake=600)
    assert spec_of(tickets) == [("TRIO", "2,9,14", 600)]
    at_t = view_at(con, long_, t, rules=rules)
    assert float(at_t.board.set_index(["pool", "combination"])["odds"][("TRIO", "1,6,10")]) == 60.0
    tickets = express_set(at_t, "TRIO", "1,6,10", threshold=threshold, stake=600)
    assert spec_of(tickets) == sorted(("TIERCE", c, 100) for c in trio_as_trifectas("1,6,10"))

    # The final price reverses both choices: that is the look-ahead version.
    final = {
        race: RaceView(
            race,
            board(
                [(p, c, o) for r, p, c, _, _, o, is_final, _ in captures if r == race and is_final]
            ),
            LOOK_AHEAD,
        )
        for race in sets
    }
    tickets = express_set(final[short], "TRIO", "2,9,14", threshold=threshold, stake=600)
    assert spec_of(tickets) == sorted(("TIERCE", c, 100) for c in trio_as_trifectas("2,9,14"))
    tickets = express_set(final[long_], "TRIO", "1,6,10", threshold=threshold, stake=600)
    assert spec_of(tickets) == [("TRIO", "1,6,10", 600)]

    # Paired against always using the direct pool, at t and at the final.
    rows = [
        *result_rows(short, (9, 2, 14), {"TRIO": 8900, "TIERCE": 52000}),
        *result_rows(long_, (6, 10, 1), {"TRIO": 1450, "TIERCE": 9100}),
        *result_rows(pending, (3, 1, 4), {"TRIO": 620, "TIERCE": 2100}),
    ]
    pays = paying(rows)

    def l1(v):
        return express_set(v, "TRIO", sets[v.race_id], threshold=threshold, stake=600)

    def direct(v):
        return [Ticket("TRIO", sets[v.race_id], 600)]

    def synthetic_minus_direct(race):
        six = [("TIERCE", c, 100) for c in trio_as_trifectas(sets[race])]
        return ref_return(six, pays[race]) - ref_return([("TRIO", sets[race], 600)], pays[race])

    races = sorted(sets)  # pending (10), short (11), long (12)
    kwargs = {"dividends": dividends_frame(rows), "clusters": clusters_for(races), **BOOT}
    got_t = compare(races, l1, direct, view=lambda r: view_at(con, r, t, rules=rules), **kwargs)
    got_final = compare(races, l1, direct, view=lambda r: final[r], **kwargs)
    con.close()

    assert got_t.dropped == [pending] and got_t.n_races == 2
    assert_diff(got_t.diff, {short: 0.0, long_: synthetic_minus_direct(long_)})
    assert "look-ahead" not in got_t.selection
    assert got_final.dropped == [pending] and got_final.n_races == 2
    assert_diff(got_final.diff, {short: synthetic_minus_direct(short), long_: 0.0})
    assert got_final.selection == LOOK_AHEAD
    assert synthetic_minus_direct(short) != 0.0 and synthetic_minus_direct(long_) != 0.0


# ---------------------------------------------------------------------------------------------
# L5: the takeout calendar (E16's arithmetic)
# ---------------------------------------------------------------------------------------------

# JRA's published payout rates from 2014-06-07 (1 - takeout), independent of jra.yaml.
RATE = {"TIERCE": 1 - 0.275, "TRIO": 1 - 0.25, "FORECAST": 1 - 0.25, "QUINELLA": 1 - 0.225}
SUPER_PREMIUM_RATE = 0.80
PREMIUM_TOP_UP = 0.05

# Races of the shipped calendar (config/takeout_boosts.yaml).
ORDINARY_RACES = [
    "jra:2026-09-21:CKY:11",  # a Super Premium day at HSN only: another venue pays the base rate
    "jra:2024-02-18:TOK:11",  # the race after an Ultra Premium GI-day race
    "jra:2019-06-02:TOK:05",
    "jra:2021-03-14:CKY:07",
]
SUPER_PREMIUM_RACES = [
    "jra:2017-12-28:NAK:03",
    "jra:2022-12-28:NAK:11",  # a Premium on WIN and Wide that day does not touch these pools
    "jra:2025-10-26:TOK:11",
    "jra:2025-10-26:KYO:01",
    "jra:2026-09-21:HSN:11",
]
ULTRA_PREMIUM_RACE = "jra:2024-02-18:TOK:10"
QUINELLA_PREMIUM_RACE = "jra:2015-05-31:TOK:11"  # ダービーウィーク馬連: QUINELLA alone +5%


def test_super_premium_day_uses_the_80_percent_rate():
    expected = {}
    for race in ORDINARY_RACES:
        expected[race] = (RATE["TIERCE"] / RATE["TRIO"], RATE["FORECAST"] / RATE["QUINELLA"])
    for race in SUPER_PREMIUM_RACES:
        expected[race] = (SUPER_PREMIUM_RATE / SUPER_PREMIUM_RATE,) * 2
    ultra = SUPER_PREMIUM_RATE + PREMIUM_TOP_UP
    expected[ULTRA_PREMIUM_RACE] = (ultra / ultra,) * 2
    expected[QUINELLA_PREMIUM_RACE] = (
        RATE["TIERCE"] / RATE["TRIO"],
        RATE["FORECAST"] / (RATE["QUINELLA"] + PREMIUM_TOP_UP),
    )
    assert expected[ORDINARY_RACES[0]][0] == pytest.approx(0.725 / 0.75)
    assert expected[SUPER_PREMIUM_RACES[0]] == (1.0, 1.0)

    for race, (trio_ratio, quinella_ratio) in expected.items():
        assert neutral_ratio(race, "TIERCE", "TRIO") == pytest.approx(trio_ratio, abs=1e-12), race
        got = neutral_ratio(race, "FORECAST", "QUINELLA")
        assert got == pytest.approx(quinella_ratio, abs=1e-12), race

    # Before 2014-06-07 no single rate applies (D10): the ratio is not made up.
    with pytest.raises(TakeoutUnknown):
        neutral_ratio("jra:2013-12-28:NAK:11", "TIERCE", "TRIO")
    # The store's pool codes only: "TRIFECTA" is not one.
    with pytest.raises(ValueError):
        neutral_ratio(ORDINARY_RACES[0], "TRIFECTA", "TRIO")
