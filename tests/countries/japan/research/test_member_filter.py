"""Ticket #716: each best-bet rule combined with a 'horse beats the crowd' member filter
(pre-registered): the filter, the crowd's probability, the paired gain, the frozen spec and the
freeze command, on synthetic values only.

Contract pinned here for `horsetrack.countries.japan.research.member_filter` (the coder implements
what these tests call; the run itself is pinned in `test_member_filter_run.py`):

- `CHECKPOINTS == (10, 5, 2)` (#713's checkpoints) and `K_GRID == (1.0, 1.25, 1.5, 2.0)`, the
  frozen grid of the filter's threshold k.
- `MEMBERS`: member id -> member, each with `.ticket` and `.unavailable` (None, or why it cannot be
  measured yet). The members are #438's debut/maiden combined member, #87's C4 meta (form), #92's
  C2 ability and #93's interactions (all available) and the JRDB comment member of #715,
  registered but unavailable (its rows are unmeasured, never a crash).
- `combination_id(rule_id, member_id, k) -> str`, distinct per triple and naming the rule and the
  member; `COMBINATIONS`: combination id -> combination with `.rule_id`, `.member_id`, `.k`, one
  per rule of #713's `predicted_final.RULES` x `MEMBERS` x `K_GRID`. `FAMILIES`: family id ->
  combination ids, a partition of `COMBINATIONS` (the registry families); `family_spec(family_id)`
  the plain-JSON spec each is frozen with; `spec_hash()`. Each spec carries `member_predictions`,
  a dict with `out_of_fold: True`, `final_board_inputs: False` and `provenance`, a list naming at
  least `fit_id`, `out_of_fold` and `final_board_inputs` (the declaration `run_study` takes).
- `crowd_q(board) -> {horse: q}`: the crowd's win probability on a (projected) Win board,
  normalised: q_h = (1 / O_h) / sum over the board of (1 / O_j).
- `qualifying(tickets, *, p_member, q, k, brackets=None) -> TicketSet | None`: the tickets of
  `tickets` that pass the filter, same pool (None or an empty set when none passes).

  Definitional choices (pinned below):
  * a horse passes when it has a finite p_member and a finite q > 0 and p_member / q >= k (the
    boundary qualifies). A horse without a p_member (the member gave none) or without a q, or with
    q = 0 or a NaN, does not pass; it never raises.
  * a ticket qualifies when every one of its legs passes, in every pool: WIN and PLACE, the
    runner; QUINELLA, QUINELLA_PLACE (Wide) and TRIO, every horse of the combination; FORECAST
    (exacta) and TIERCE (trifecta), every horse of the order, whatever its position. #713's rules
    hand over a `TicketSet` of combinations and nothing marks an axis, so no leg is singled out.
  * BRACKET_QUINELLA: a combination names brackets; a bracket leg passes when at least one runner
    of that bracket (`brackets`: horse -> bracket) passes, since any of them carries the bracket.
- `paired_gain(bets, *, seed, n_boot, level) -> .gain, .lo, .hi`: `bets` holds one tuple per race
  the rule bought (on the races the member covers), (meeting_id, rule_staked, rule_returned,
  combo_staked, combo_returned), combo_staked 0 where the filter passed no ticket.
  * gain = 100 x sum(combo_returned) / sum(combo_staked) - 100 x sum(rule_returned) /
    sum(rule_staked): the combination's return per 100 yen less the rule's alone on the same
    races (a difference of ratios of sums, as the leaderboard ranks; not a mean of per-race
    ratios); None when the combination staked nothing.
  * lo, hi: the one-sided (1 - level) and level quantiles of the same difference over a meeting
    bootstrap (D7) that resamples the meetings once per replicate for both sides (paired); None
    when the combination's bets fall in fewer than two meetings, or there is no gain.
    `core.eval.bootstrap.paired_bootstrap` bounds a per-race mean, which this ratio difference is
    not, so the two sides share each replicate's meeting weights instead.
- `member_filter_app`, registered as `horsetrack japan member-filter`, with `freeze`: freezes every
  family's spec in the japan registry (again is a no-op).

Every expected value below is written out by hand or read from a reference stated here.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party.
from __future__ import annotations  # noqa: I001

import itertools
import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from typer.testing import CliRunner

from horsetrack.core.eval.registry import _spec_hash, default_registry
from horsetrack.countries.japan.betting.formations import TicketSet
from horsetrack.countries.japan.board.snapshot import RaceBoard
from horsetrack.countries.japan.research import predicted_final as pf
from horsetrack.countries.japan.research import member_filter as mf

DATA_VAR = "HORSETRACK_DATA_DIR"
LEVEL = 0.95  # D7, scoring.yaml


def combos(tickets) -> set[tuple[int, ...]]:
    """A TicketSet's combinations as tuples of ints; None or an empty set is no ticket."""
    if tickets is None:
        return set()
    out = set()
    for c in tickets.combinations:
        if isinstance(c, (int, np.integer)):
            out.add((int(c),))
        elif isinstance(c, str):
            out.add(tuple(int(x) for x in c.split(",")))
        else:
            out.add(tuple(int(x) for x in c))
    return out


# --------------------------------------------------------------------------- the pre-registration


def test_checkpoints_and_k_grid():
    assert tuple(mf.CHECKPOINTS) == (10, 5, 2)
    assert tuple(mf.K_GRID) == (1.0, 1.25, 1.5, 2.0)


def test_the_members_are_registered_with_the_jrdb_comment_member_unavailable():
    tickets = {m.ticket for m in mf.MEMBERS.values()}
    assert {438, 87, 92, 93, 715} <= tickets
    available = {m.ticket for m in mf.MEMBERS.values() if m.unavailable is None}
    assert {438, 87, 92, 93} <= available
    jrdb = [m for m in mf.MEMBERS.values() if m.ticket == 715]
    assert jrdb and all(isinstance(m.unavailable, str) and m.unavailable.strip() for m in jrdb)


def test_every_rule_member_and_k_is_one_combination():
    expected = {
        mf.combination_id(r, m, k): (r, m, k)
        for r in pf.RULES
        for m in mf.MEMBERS
        for k in mf.K_GRID
    }
    assert len(expected) == len(pf.RULES) * len(mf.MEMBERS) * len(mf.K_GRID), "ids are distinct"
    assert set(mf.COMBINATIONS) == set(expected)
    for cid, (rule_id, member_id, k) in expected.items():
        combination = mf.COMBINATIONS[cid]
        assert (combination.rule_id, combination.member_id, combination.k) == (
            rule_id,
            member_id,
            k,
        )
        assert rule_id in cid and member_id in cid, cid
    members = [c for family in mf.FAMILIES.values() for c in family]
    assert sorted(members) == sorted(mf.COMBINATIONS), "every combination is in exactly one family"


def _values(spec) -> list:
    out = [spec]
    if isinstance(spec, dict):
        for key, value in spec.items():
            out += [key, *_values(value)]
    elif isinstance(spec, list):
        for value in spec:
            out += _values(value)
    return out


def test_each_family_freezes_a_plain_json_spec_with_its_rules_members_k_grid_and_checkpoints():
    texts = []
    for family_id, members in mf.FAMILIES.items():
        spec = mf.family_spec(family_id)
        assert isinstance(spec, dict) and spec
        assert json.loads(json.dumps(spec)) == spec, "a family spec is plain JSON"
        text = json.dumps(spec)
        named = {mf.COMBINATIONS[c].rule_id for c in members} | {
            mf.COMBINATIONS[c].member_id for c in members
        }
        missing = [x for x in named if json.dumps(x)[1:-1] not in text]
        assert not missing, f"{family_id}'s spec does not name {missing[:3]}"
        values = _values(spec)
        assert list(mf.K_GRID) in values, f"{family_id}'s spec does not carry the k grid"
        assert [10, 5, 2] in values, f"{family_id}'s spec does not carry the checkpoints"
        assert pf.spec_hash() in values, "the rules are #713's frozen rules, by their spec hash"
        assert not family_id.startswith(pf.FAMILY_PREFIX), "read as #713's rows otherwise"
        texts.append(json.dumps(spec, sort_keys=True))
    assert len(set(texts)) == len(texts)
    assert isinstance(mf.spec_hash(), str) and mf.spec_hash()


PROVENANCE_FIELDS = {"fit_id", "out_of_fold", "final_board_inputs"}


def test_each_family_spec_requires_out_of_fold_decision_time_member_predictions():
    """A member's p is compared with the projected q at post - k, so it must be out-of-fold and
    built only from what is known at post - k (#87's C4 meta reads the final market otherwise):
    each frozen spec carries `member_predictions` with `out_of_fold` True, `final_board_inputs`
    False and `provenance`, the fields every scored member declares and the run records."""
    for family_id in mf.FAMILIES:
        spec = mf.family_spec(family_id)
        req = spec.get("member_predictions")
        assert isinstance(req, dict), f"{family_id}'s spec states no member-prediction requirement"
        assert req.get("out_of_fold") is True, family_id
        assert req.get("final_board_inputs") is False, family_id
        assert PROVENANCE_FIELDS <= set(req.get("provenance", ())), family_id


# --------------------------------------------------------------------------- the crowd's q


def board_of(win: dict[int, float]) -> RaceBoard:
    post = datetime(2025, 6, 7, 3, 0, tzinfo=timezone.utc)
    at = post - timedelta(minutes=2)
    return RaceBoard("jra:2025-06-07:TOK:01", post, at, at, False, win, {}, {}, {}, frozenset())


CROWD_CASES = [
    ({1: 2.0, 2: 4.0, 3: 4.0}, {1: 0.5, 2: 0.25, 3: 0.25}),
    ({1: 1.6, 2: 3.2, 3: 8.0}, {1: 0.625 / 1.0625, 2: 0.3125 / 1.0625, 3: 0.125 / 1.0625}),
    ({1: 3.0, 2: 3.0, 3: 3.0, 4: 3.0}, {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}),
    ({1: 2.5, 2: 5.0, 3: 5.0, 4: 10.0, 5: 10.0}, {1: 0.4, 2: 0.2, 3: 0.2, 4: 0.1, 5: 0.1}),
    ({3: 4.0, 7: 1.25}, {3: 0.25 / 1.05, 7: 0.8 / 1.05}),
]


@pytest.mark.parametrize(("win", "expected"), CROWD_CASES)
def test_crowd_q_is_the_normalised_implied_win_probability(win, expected):
    got = mf.crowd_q(board_of(win))
    assert set(got) == set(expected)
    assert got == pytest.approx(expected, abs=1e-12)
    assert sum(got.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- the filter

# Binary fractions, so p / q is exact: horse -> (q, p) and its ratio p / q.
Q = {1: 0.25, 2: 0.5, 3: 0.125, 4: 0.25, 5: 0.125, 6: 0.0625}
P = {1: 0.5, 2: 0.625, 3: 0.125, 4: 0.1875, 5: 0.1875, 6: 0.15625}
RATIO = {1: 2.0, 2: 1.25, 3: 1.0, 4: 0.75, 5: 1.5, 6: 2.5}
# The horses that pass at each k, by hand (the boundary passes).
PASS = {
    0.5: {1, 2, 3, 4, 5, 6},
    1.0: {1, 2, 3, 5, 6},
    1.25: {1, 2, 5, 6},
    1.5: {1, 5, 6},
    2.0: {1, 6},
    2.5: {6},
    3.0: set(),
}
assert all(P[h] / Q[h] == RATIO[h] for h in Q)
assert all(PASS[k] == {h for h in RATIO if RATIO[h] >= k} for k in PASS)

SINGLES = tuple((h,) for h in range(1, 7))
PAIRS = tuple(itertools.combinations(range(1, 7), 2))
ORDERED_PAIRS = tuple(itertools.permutations((1, 2, 4, 5), 2))
TRIPLES = tuple(itertools.combinations((1, 2, 3, 5, 6), 3)) + ((1, 4, 6), (2, 4, 5))
ORDERS = ((1, 4, 6), (1, 5, 6), (6, 5, 1), (2, 1, 6), (5, 6, 2), (3, 1, 5))
POOL_COMBOS = {
    "WIN": SINGLES,
    "PLACE": SINGLES,
    "QUINELLA_PLACE": PAIRS,
    "QUINELLA": PAIRS,
    "FORECAST": ORDERED_PAIRS,
    "TRIO": TRIPLES,
    "TIERCE": ORDERS,
}

# A few written out in full: (pool, k, combinations in, combinations that qualify).
HAND_CASES = [
    ("WIN", 1.5, SINGLES, {(1,), (5,), (6,)}),
    ("WIN", 2.0, SINGLES, {(1,), (6,)}),  # horse 1 sits on the boundary and qualifies
    ("PLACE", 1.25, SINGLES, {(1,), (2,), (5,), (6,)}),
    ("QUINELLA_PLACE", 1.5, PAIRS, {(1, 5), (1, 6), (5, 6)}),
    ("QUINELLA_PLACE", 2.0, PAIRS, {(1, 6)}),
    ("QUINELLA", 1.0, ((1, 4), (2, 3), (4, 5)), {(2, 3)}),
    ("FORECAST", 1.25, ORDERED_PAIRS, {(1, 2), (2, 1), (1, 5), (5, 1), (2, 5), (5, 2)}),
    ("FORECAST", 1.5, ORDERED_PAIRS, {(1, 5), (5, 1)}),  # both legs, either position
    ("TRIO", 1.25, TRIPLES, {(1, 2, 5), (1, 2, 6), (1, 5, 6), (2, 5, 6)}),
    ("TIERCE", 1.5, ORDERS, {(1, 5, 6), (6, 5, 1)}),
    ("TIERCE", 2.5, ORDERS, set()),
]


@pytest.mark.parametrize(("pool", "k", "given", "expected"), HAND_CASES)
def test_a_ticket_qualifies_when_every_horse_on_it_beats_the_crowd_by_k(pool, k, given, expected):
    tickets = TicketSet(pool, tuple(given))
    got = mf.qualifying(tickets, p_member=P, q=Q, k=k)
    assert combos(got) == expected
    if got is not None and got.combinations:
        assert got.pool == pool


@pytest.mark.parametrize("pool", sorted(POOL_COMBOS))
@pytest.mark.parametrize("k", sorted(PASS))
def test_the_filter_over_every_pool_and_k(pool, k):
    given = POOL_COMBOS[pool]
    expected = {c for c in given if all(h in PASS[k] for h in c)}
    got = mf.qualifying(TicketSet(pool, given), p_member=P, q=Q, k=k)
    assert combos(got) == expected
    if expected:
        assert got.pool == pool


def test_the_filter_reads_the_ratio_not_p_or_q_alone():
    """Ranking the horses by p alone, or by 1 / q alone, gives other passing sets."""
    by_p = sorted(P, key=lambda h: -P[h])
    by_q = sorted(Q, key=lambda h: Q[h])
    for k in (1.25, 1.5, 2.0):
        got = {c[0] for c in combos(mf.qualifying(TicketSet("WIN", SINGLES), p_member=P, q=Q, k=k))}
        assert got == PASS[k]
        n = len(got)
        assert set(by_p[:n]) != got and set(by_q[:n]) != got, k


BRACKETS = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4}
BRACKET_COMBOS = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (2, 2))
BRACKET_CASES = [
    # bracket 1 = {1}, 2 = {2, 3}, 3 = {4, 5}, 4 = {6}; a bracket passes on any one of its runners
    (1.0, set(BRACKET_COMBOS)),
    (1.25, set(BRACKET_COMBOS)),  # bracket 2 on horse 2, bracket 3 on horse 5
    (1.5, {(1, 3), (1, 4), (3, 4)}),  # bracket 2's runners fall short; 3 holds on horse 5
    (2.0, {(1, 4)}),
    (2.5, set()),  # bracket 1's horse 1 falls short
]


@pytest.mark.parametrize(("k", "expected"), BRACKET_CASES)
def test_a_bracket_qualifies_on_any_runner_of_each_bracket(k, expected):
    tickets = TicketSet("BRACKET_QUINELLA", BRACKET_COMBOS)
    got = mf.qualifying(tickets, p_member=P, q=Q, k=k, brackets=BRACKETS)
    assert combos(got) == expected


EDGE_CASES = [
    # (what, p_member, q, k, combinations of Wide in, combinations that qualify)
    ("no p for horse 2", {h: p for h, p in P.items() if h != 2}, Q, 1.0,
     ((1, 2), (1, 3), (2, 5)), {(1, 3)}),
    ("no q for horse 3", P, {h: q for h, q in Q.items() if h != 3}, 1.0,
     ((1, 2), (1, 3), (3, 5)), {(1, 2)}),
    ("q zero for horse 5", P, {**Q, 5: 0.0}, 1.0, ((1, 5), (1, 6), (5, 6)), {(1, 6)}),
    ("p NaN for horse 1", {**P, 1: float("nan")}, Q, 1.0, ((1, 2), (2, 6)), {(2, 6)}),
    ("p None for horse 6", {**P, 6: None}, Q, 1.0, ((1, 6), (1, 2)), {(1, 2)}),
    ("q NaN for horse 2", P, {**Q, 2: float("nan")}, 1.0, ((1, 2), (5, 6)), {(5, 6)}),
    ("a horse off the board", P, Q, 1.0, ((1, 9), (1, 5)), {(1, 5)}),
    ("nothing qualifies", {h: 0.0 for h in P}, Q, 1.0, ((1, 2), (5, 6)), set()),
]  # fmt: skip


@pytest.mark.parametrize(
    ("what", "p_member", "q", "k", "given", "expected"), EDGE_CASES, ids=[c[0] for c in EDGE_CASES]
)
def test_a_horse_without_a_usable_ratio_never_qualifies(what, p_member, q, k, given, expected):
    got = mf.qualifying(TicketSet("QUINELLA_PLACE", given), p_member=p_member, q=q, k=k)
    assert combos(got) == expected, what


def test_the_filter_matches_a_plain_reading_over_varied_races():
    rng = np.random.default_rng(716)
    seen = set()
    for i in range(40):
        n = int(rng.integers(4, 12))
        q = dict(enumerate(rng.dirichlet(np.ones(n)).tolist(), start=1))
        ratio = rng.choice([0.4, 0.8, 0.9, 1.1, 1.3, 1.6, 1.8, 2.3, 3.1], size=n)
        p = {h: float(ratio[h - 1] * q[h]) for h in q}
        k = float(rng.choice(mf.K_GRID))
        pool = ("WIN", "QUINELLA_PLACE", "FORECAST", "TRIO")[i % 4]
        legs = {"WIN": 1, "QUINELLA_PLACE": 2, "FORECAST": 2, "TRIO": 3}[pool]
        maker = itertools.permutations if pool == "FORECAST" else itertools.combinations
        given = tuple(maker(sorted(q), legs))
        passes = {h for h in q if ratio[h - 1] >= k}
        expected = {c for c in given if set(c) <= passes}
        got = mf.qualifying(TicketSet(pool, given), p_member=p, q=q, k=k)
        assert combos(got) == expected, i
        seen.add(len(expected))
    assert len(seen) > 4, "the matches come from the ratios"


# --------------------------------------------------------------------------- the paired gain

# Two meetings: the bootstrap's only replicates are M1 twice (gain 150), M1 and M2 (the sample's
# own 58.33) and M2 twice (-33.33), with probabilities 1/4, 1/2 and 1/4, so at 1000 replicates the
# 5% and 95% quantiles are -33.33 and 150 whatever the seed.
TWO_MEETINGS = [
    ("M1", 300.0, 0.0, 100.0, 0.0),
    ("M1", 300.0, 450.0, 100.0, 450.0),
    ("M2", 300.0, 200.0, 0.0, 0.0),
    ("M2", 300.0, 0.0, 200.0, 0.0),
]
# rule: 650 / 1200; combination: 450 / 400 (a mean of per-race ratios would read 150 for it)
TWO_GAIN = 100.0 * 450 / 400 - 100.0 * 650 / 1200
TWO_LO = 100.0 * 0 / 400 - 100.0 * 400 / 1200  # M2 twice
TWO_HI = 100.0 * 900 / 400 - 100.0 * 900 / 1200  # M1 twice


@pytest.mark.parametrize("seed", [1, 7, 20261129])
def test_the_paired_gain_and_its_meeting_bounds_by_hand(seed):
    got = mf.paired_gain(TWO_MEETINGS, seed=seed, n_boot=1000, level=LEVEL)
    assert got.gain == pytest.approx(TWO_GAIN, abs=1e-9)
    assert got.lo == pytest.approx(TWO_LO, abs=1e-9)
    assert got.hi == pytest.approx(TWO_HI, abs=1e-9)


GAIN_CASES = [
    # (bets, gain by hand)
    ([("M1", 200.0, 300.0, 100.0, 300.0), ("M2", 100.0, 0.0, 0.0, 0.0)],
     100.0 * 300 / 100 - 100.0 * 300 / 300),
    ([("M1", 100.0, 0.0, 100.0, 0.0), ("M2", 100.0, 180.0, 0.0, 0.0),
      ("M3", 100.0, 0.0, 100.0, 0.0)],
     0.0 - 100.0 * 180 / 300),  # the filter dropped the only winner
    ([("M1", 500.0, 1850.0, 300.0, 1850.0), ("M2", 400.0, 0.0, 100.0, 0.0)],
     100.0 * 1850 / 400 - 100.0 * 1850 / 900),
    ([("A", 300.0, 710.0, 300.0, 710.0), ("B", 100.0, 220.0, 0.0, 0.0)],
     100.0 * 710 / 300 - 100.0 * 930 / 400),
]  # fmt: skip


@pytest.mark.parametrize(("bets", "gain"), GAIN_CASES)
def test_the_paired_gain_is_the_difference_of_the_returns_per_100_on_the_same_races(bets, gain):
    got = mf.paired_gain(bets, seed=3, n_boot=200, level=LEVEL)
    assert got.gain == pytest.approx(gain, abs=1e-9)


def test_no_match_has_no_gain_and_one_meeting_no_bounds():
    none = mf.paired_gain(
        [("M1", 100.0, 0.0, 0.0, 0.0), ("M2", 100.0, 250.0, 0.0, 0.0)],
        seed=1,
        n_boot=200,
        level=LEVEL,
    )
    assert none.gain is None and none.lo is None and none.hi is None
    one = mf.paired_gain(
        [
            ("M1", 200.0, 300.0, 100.0, 300.0),
            ("M1", 100.0, 0.0, 0.0, 0.0),
            ("M2", 100.0, 0.0, 0.0, 0.0),
        ],
        seed=1,
        n_boot=200,
        level=LEVEL,
    )
    assert one.gain == pytest.approx(100.0 * 300 / 100 - 100.0 * 300 / 400)
    assert one.lo is None and one.hi is None, "the combination bet in one meeting only"


def test_a_filter_that_passes_every_ticket_gains_nothing_in_every_replicate():
    bets = [
        ("M1", 200.0, 150.0, 200.0, 150.0),
        ("M2", 100.0, 0.0, 100.0, 0.0),
        ("M3", 300.0, 600.0, 300.0, 600.0),
        ("M3", 100.0, 40.0, 100.0, 40.0),
    ]
    got = mf.paired_gain(bets, seed=11, n_boot=300, level=LEVEL)
    assert got.gain == pytest.approx(0.0, abs=1e-9)
    assert got.lo == pytest.approx(0.0, abs=1e-9) and got.hi == pytest.approx(0.0, abs=1e-9)


def test_the_bounds_are_reproducible_under_the_seed():
    bets = [
        ("M1", 300.0, 0.0, 100.0, 0.0),
        ("M1", 200.0, 500.0, 100.0, 500.0),
        ("M2", 300.0, 300.0, 200.0, 300.0),
        ("M3", 100.0, 0.0, 100.0, 0.0),
        ("M4", 400.0, 900.0, 100.0, 0.0),
    ]
    a = mf.paired_gain(bets, seed=42, n_boot=500, level=LEVEL)
    b = mf.paired_gain(bets, seed=42, n_boot=500, level=LEVEL)
    assert (a.gain, a.lo, a.hi) == (b.gain, b.lo, b.hi)
    assert a.lo is not None and a.hi is not None
    assert math.isfinite(a.lo) and math.isfinite(a.hi) and a.lo <= a.hi


# --------------------------------------------------------------------------- the freeze command


def test_the_freeze_command_freezes_every_family(tmp_path, monkeypatch):
    from horsetrack.cli import app

    monkeypatch.setenv(DATA_VAR, str(tmp_path / "data"))
    runner = CliRunner()
    first = runner.invoke(app, ["japan", "member-filter", "freeze"])
    assert first.exit_code == 0, (first.output, first.exception)
    registry = default_registry("japan")
    for family_id in mf.FAMILIES:
        assert registry.frozen_spec_hash(family_id) == _spec_hash(mf.family_spec(family_id))
    again = runner.invoke(app, ["japan", "member-filter", "freeze"])
    assert again.exit_code == 0, "freezing again with the same spec is a no-op"
    assert registry.records() == [], "freezing appends no run"
