"""The avoid flags on the board at t (ticket #117).

Contract pinned here (`horsetrack.countries.japan.board.flags`):
- `avoid_flags(board, *, rules, now_utc)` returns {(pool, combination): (flag, ...)}. Keys are
  ('WIN', '7'), ('PLACE', '7') and ('QUINELLA_PLACE', '3,7') (Wide: a,b with a < b). A key exists
  for every priced runner or pair on the board and for every scratched runner (on its WIN and PLACE
  keys); keys without a flag are absent; each key's flags are sorted by name.
- floor: WIN odds, or the PLACE / Wide low figure, at or below `floor_max_odds`.
- tail: a WIN or PLACE key of a runner whose WIN odds are above `tail_min_odds` (strictly).
- tail_member: a Wide pair with a runner whose WIN-odds rank is `tail_rank` or worse. Rank 1 is the
  lowest odds; tied odds all take the better rank (1 + number of runners with strictly lower odds).
- stale: every key, when `board.capture_utc` is more than `stale_s` seconds before `now_utc`.
- scratched: the WIN and PLACE keys of a runner in `board.scratched`.
- A pure function of (board, rules, now_utc): the board is not mutated and `now_utc` must be aware.

Every board is built here in code from synthetic numbers.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from itertools import combinations

import pytest

from horsetrack.core import config
from horsetrack.countries.japan.board.flags import FlagRules, avoid_flags, load_flag_rules
from horsetrack.countries.japan.board.snapshot import RaceBoard

UTC = timezone.utc
NOW = datetime(2026, 11, 28, 5, 58, 0, tzinfo=UTC)
RACE = "jra:2026-11-28:TOK:11"
WIDE = "QUINELLA_PLACE"

RULES = FlagRules(floor_max_odds=1.1, tail_min_odds=100.0, tail_rank=9, stale_s=120.0)


def rules(**over) -> FlagRules:
    return FlagRules(
        **{
            "floor_max_odds": 1.1,
            "tail_min_odds": 100.0,
            "tail_rank": 9,
            "stale_s": 120.0,
            **over,
        }
    )


def board(
    win,
    place=None,
    wide=None,
    *,
    scratched=(),
    age_s: float = 10.0,
    at_age_s: float | None = None,
) -> RaceBoard:
    """A board whose capture is `age_s` seconds older than NOW (its at_utc `at_age_s`, default the
    same)."""
    capture = NOW - timedelta(seconds=age_s)
    at = NOW - timedelta(seconds=age_s if at_age_s is None else at_age_s)
    return RaceBoard(
        race_id=RACE,
        post_utc=NOW + timedelta(minutes=2),
        at_utc=at,
        capture_utc=capture,
        final=False,
        win=dict(win),
        place=dict(place or {}),
        wide=dict(wide or {}),
        totals={},
        scratched=frozenset(scratched),
    )


def pair_key(a: int, b: int) -> tuple[str, str]:
    lo, hi = sorted((a, b))
    return (WIDE, f"{lo},{hi}")


# Ten runners; the runner number does not follow the favourite order. Favourite order (rank 1..10):
# 3 (2.0), 7 (3.0), 6 (4.5), 10 (6.0), 5 (8.0), 9 (11.0), 1 (15.0), 8 (22.0), 4 (35.0), 2 (60.0)
TEN_WIN = {1: 15.0, 2: 60.0, 3: 2.0, 4: 35.0, 5: 8.0, 6: 4.5, 7: 3.0, 8: 22.0, 9: 11.0, 10: 6.0}
ALL_PAIRS = list(combinations(range(1, 11), 2))
SAFE_WIDE = {pair: (5.0, 9.0) for pair in ALL_PAIRS}


def ten_runner_board(win=None) -> RaceBoard:
    return board(win or TEN_WIN, wide=SAFE_WIDE)


# ------------------------------------------------------------------ floor


@pytest.mark.parametrize(
    ("low", "floor_max", "flagged"),
    [
        (1.0, 1.1, True),
        (1.1, 1.1, True),
        (1.11, 1.1, False),
        (1.2, 1.1, False),
        (9.9, 1.1, False),
        (1.5, 2.0, True),  # the threshold, not a constant
        (2.0, 2.0, True),
        (2.1, 2.0, False),
    ],
)
def test_floor_flags_prices_at_or_below_1_1_in_every_pool(low, floor_max, flagged):
    r = rules(floor_max_odds=floor_max)
    # WIN: the price under test on runner 1; the others are well clear of any floor.
    got = avoid_flags(board({1: low, 2: 6.0, 3: 9.0}), rules=r, now_utc=NOW)
    assert got == ({("WIN", "1"): ("floor",)} if flagged else {})
    # PLACE: the low figure decides, the high figure is irrelevant.
    got = avoid_flags(
        board(
            {1: 5.0, 2: 6.0, 3: 9.0},
            place={1: (low, None), 2: (3.1, 3.6), 3: (4.0, 5.0)},
        ),
        rules=r,
        now_utc=NOW,
    )
    assert got == ({("PLACE", "1"): ("floor",)} if flagged else {})
    got = avoid_flags(
        board(
            {1: 5.0, 2: 6.0, 3: 9.0},
            place={1: (low, 7.5), 2: (3.1, 3.6), 3: (4.0, 5.0)},
        ),
        rules=r,
        now_utc=NOW,
    )
    assert got == ({("PLACE", "1"): ("floor",)} if flagged else {})
    # Wide: the low figure of the pair.
    got = avoid_flags(
        board(
            {1: 5.0, 2: 6.0, 3: 9.0},
            wide={(1, 2): (low, 3.3), (1, 3): (6.0, 9.0), (2, 3): (7.0, None)},
        ),
        rules=r,
        now_utc=NOW,
    )
    assert got == ({pair_key(1, 2): ("floor",)} if flagged else {})


def test_floor_is_judged_per_pool_not_across_pools():
    """A floor price in one pool puts the flag on that pool's key only."""
    b = board(
        {1: 1.1, 2: 6.0, 3: 9.0},
        place={1: (1.0, 1.2), 2: (1.8, 2.4), 3: (2.5, 3.5)},
        wide={(1, 2): (1.1, 1.4), (1, 3): (1.3, 1.6), (2, 3): (3.0, 4.0)},
    )
    assert avoid_flags(b, rules=RULES, now_utc=NOW) == {
        ("WIN", "1"): ("floor",),
        ("PLACE", "1"): ("floor",),
        pair_key(1, 2): ("floor",),
    }


# ------------------------------------------------------------------ tail


@pytest.mark.parametrize(
    ("tail_min", "flagged"),
    [
        (100.0, {1}),  # 100.1 is above, 100.0 is not above
        (100.1, set()),  # the cutoff itself is not flagged
        (50.0, {1, 2}),
        (200.0, set()),
    ],
)
def test_tail_flags_win_and_place_above_the_cutoff(tail_min, flagged):
    win = {1: 100.1, 2: 100.0, 3: 3.0, 4: 5.0}
    place = {1: (20.0, 30.0), 2: (20.0, 30.0), 3: (1.5, 2.0), 4: (2.0, 3.0)}
    got = avoid_flags(board(win, place), rules=rules(tail_min_odds=tail_min), now_utc=NOW)
    expected = {}
    for runner in flagged:
        expected[("WIN", str(runner))] = ("tail",)
        expected[("PLACE", str(runner))] = ("tail",)
    assert got == expected


def test_tail_is_not_set_on_wide_pairs_and_needs_a_win_price_above_the_cutoff():
    """A runner with a long PLACE price but a short WIN price is not a tail runner, and a tail
    runner's Wide pairs carry tail_member at most, never tail."""
    win = {1: 150.0, 2: 4.0, 3: 6.0}
    place = {1: (40.0, 60.0), 2: (200.0, 300.0), 3: (2.0, 3.0)}
    wide = {(1, 2): (50.0, 80.0), (1, 3): (60.0, 90.0), (2, 3): (5.0, 8.0)}
    got = avoid_flags(board(win, place, wide), rules=RULES, now_utc=NOW)
    assert got == {("WIN", "1"): ("tail",), ("PLACE", "1"): ("tail",)}


# ------------------------------------------------------------------ tail_member


def test_tail_member_flags_wide_pairs_with_a_low_ranked_runner():
    """Ten runners, tail_rank 9: the ninth and tenth favourites (runners 4 and 2) are rank 9 or
    worse. Every pair with either is flagged; the 28 pairs of ranks 1-8 only are not."""
    got = avoid_flags(ten_runner_board(), rules=RULES, now_utc=NOW)
    assert set(got) == {pair_key(a, b) for a, b in ALL_PAIRS if {a, b} & {2, 4}}
    assert len(got) == 45 - 28
    assert all(flags_ == ("tail_member",) for flags_ in got.values())


def test_tail_member_depends_on_the_rank_threshold_and_the_runners():
    # tail_rank 5 on the same board: ranks 5.. are runners 5, 9, 1, 8, 4, 2.
    got = avoid_flags(ten_runner_board(), rules=rules(tail_rank=5), now_utc=NOW)
    worse = {5, 9, 1, 8, 4, 2}
    assert set(got) == {pair_key(a, b) for a, b in ALL_PAIRS if {a, b} & worse}
    # tail_rank 10: only the 10th favourite (runner 2).
    got = avoid_flags(ten_runner_board(), rules=rules(tail_rank=10), now_utc=NOW)
    assert set(got) == {pair_key(a, b) for a, b in ALL_PAIRS if 2 in (a, b)}
    # Swap the long shot: runner 3 becomes the 10th favourite and the flag follows.
    swapped = dict(TEN_WIN)
    swapped[3], swapped[2] = 60.0, 2.0
    got = avoid_flags(ten_runner_board(swapped), rules=rules(tail_rank=10), now_utc=NOW)
    assert set(got) == {pair_key(a, b) for a, b in ALL_PAIRS if 3 in (a, b)}


def test_no_tail_member_when_the_race_has_fewer_runners_than_tail_rank():
    win = {n: 2.0 + n for n in range(1, 9)}  # eight runners
    wide = {pair: (4.0, 6.0) for pair in combinations(range(1, 9), 2)}
    assert avoid_flags(board(win, wide=wide), rules=RULES, now_utc=NOW) == {}


@pytest.mark.parametrize(
    ("tail_rank", "win_8_9_10", "flagged_runners"),
    [
        # Runners 8 and 9 tie for ranks 8 and 9: both take rank 8; runner 10 is rank 10.
        (9, {8: 20.0, 9: 20.0, 10: 60.0}, {10}),
        # Runners 9 and 10 tie for ranks 9 and 10: both take rank 9, so both are flagged.
        (9, {8: 15.0, 9: 20.0, 10: 20.0}, {9, 10}),
        # The same tie with tail_rank 10: both are rank 9, so neither is flagged.
        (10, {8: 15.0, 9: 20.0, 10: 20.0}, set()),
        # A three-way tie for ranks 8-10: all rank 8, none reaches 9.
        (9, {8: 20.0, 9: 20.0, 10: 20.0}, set()),
        # Tied for ranks 8 and 9 with tail_rank 8: both rank 8, flagged, and so is rank 10.
        (8, {8: 20.0, 9: 20.0, 10: 60.0}, {8, 9, 10}),
    ],
)
def test_tied_odds_take_the_better_rank(tail_rank, win_8_9_10, flagged_runners):
    win = {n: 1.0 + n for n in range(1, 8)}  # runners 1-7: odds 2..8, ranks 1-7
    win.update(win_8_9_10)
    b = board(win, wide={pair: (4.0, 6.0) for pair in ALL_PAIRS})
    got = avoid_flags(b, rules=rules(tail_rank=tail_rank), now_utc=NOW)
    assert set(got) == {pair_key(a, c) for a, c in ALL_PAIRS if {a, c} & flagged_runners}


# ------------------------------------------------------------------ stale


@pytest.mark.parametrize(
    ("age_s", "stale_s", "stale"),
    [
        (121.0, 120.0, True),
        (120.0, 120.0, False),
        (120.5, 120.0, True),
        (119.0, 120.0, False),
        (0.0, 120.0, False),
        (31.0, 30.0, True),
        (30.0, 30.0, False),
        (400.0, 600.0, False),
    ],
)
def test_a_board_older_than_stale_s_is_flagged_stale(age_s, stale_s, stale):
    b = board(
        {1: 5.0, 2: 6.0, 3: 9.0},
        place={1: (1.5, 2.0), 2: (1.8, 2.4), 3: (2.5, 3.5)},
        wide={(1, 2): (4.0, 5.0), (1, 3): (6.0, 9.0), (2, 3): (7.0, None)},
        scratched={4},
        age_s=age_s,
    )
    got = avoid_flags(b, rules=rules(stale_s=stale_s), now_utc=NOW)
    every_key = {
        ("WIN", "1"),
        ("WIN", "2"),
        ("WIN", "3"),
        ("PLACE", "1"),
        ("PLACE", "2"),
        ("PLACE", "3"),
        pair_key(1, 2),
        pair_key(1, 3),
        pair_key(2, 3),
        ("WIN", "4"),
        ("PLACE", "4"),
    }
    if stale:
        assert set(got) == every_key
        assert all("stale" in value for value in got.values())
        assert got[("WIN", "4")] == ("scratched", "stale")
        assert got[("WIN", "1")] == ("stale",)
    else:
        assert all("stale" not in value for value in got.values())
        assert got == {("WIN", "4"): ("scratched",), ("PLACE", "4"): ("scratched",)}


def test_staleness_reads_capture_utc_against_the_clock_not_at_utc():
    # Old at_utc but a fresh capture: not stale. Fresh at_utc but an old capture: stale.
    fresh_capture = board({1: 5.0, 2: 6.0}, age_s=5.0, at_age_s=900.0)
    assert avoid_flags(fresh_capture, rules=RULES, now_utc=NOW) == {}
    old_capture = board({1: 5.0, 2: 6.0}, age_s=300.0, at_age_s=5.0)
    assert avoid_flags(old_capture, rules=RULES, now_utc=NOW) == {
        ("WIN", "1"): ("stale",),
        ("WIN", "2"): ("stale",),
    }
    # The same board is stale later and fresh earlier: the clock is the input.
    b = board({1: 5.0, 2: 6.0}, age_s=10.0)
    assert avoid_flags(b, rules=RULES, now_utc=NOW + timedelta(seconds=110)) == {}
    assert set(avoid_flags(b, rules=RULES, now_utc=NOW + timedelta(seconds=111))) == {
        ("WIN", "1"),
        ("WIN", "2"),
    }


def test_now_must_be_timezone_aware_and_other_offsets_give_the_same_answer():
    b = board({1: 5.0, 2: 6.0}, age_s=300.0)
    with pytest.raises((ValueError, TypeError)):
        avoid_flags(b, rules=RULES, now_utc=NOW.replace(tzinfo=None))
    jst = timezone(timedelta(hours=9))
    assert avoid_flags(b, rules=RULES, now_utc=NOW.astimezone(jst)) == avoid_flags(
        b, rules=RULES, now_utc=NOW
    )


# ------------------------------------------------------------------ scratched


def test_scratched_runners_are_flagged():
    """Scratched runners carry no price; they are flagged on their WIN and PLACE keys only."""
    b = board(
        {1: 5.0, 2: 6.0, 3: 9.0},
        place={1: (1.5, 2.0), 2: (1.8, 2.4), 3: (2.5, 3.5)},
        wide={(1, 2): (4.0, 5.0), (1, 3): (6.0, 9.0), (2, 3): (7.0, None)},
        scratched={4, 7},
    )
    assert avoid_flags(b, rules=RULES, now_utc=NOW) == {
        ("WIN", "4"): ("scratched",),
        ("PLACE", "4"): ("scratched",),
        ("WIN", "7"): ("scratched",),
        ("PLACE", "7"): ("scratched",),
    }
    assert avoid_flags(board({1: 5.0, 2: 6.0}), rules=RULES, now_utc=NOW) == {}


def test_scratched_runner_does_not_count_in_the_win_rank():
    """Ten runners priced and one scratched: the scratched runner takes no rank, so the pairs
    flagged as tail_member are those of the 9th and 10th priced runners."""
    b = board(
        TEN_WIN,
        wide={pair: (5.0, 9.0) for pair in ALL_PAIRS if 11 not in pair},
        scratched={11},
    )
    got = avoid_flags(b, rules=RULES, now_utc=NOW)
    assert got.pop(("WIN", "11")) == ("scratched",)
    assert got.pop(("PLACE", "11")) == ("scratched",)
    assert set(got) == {pair_key(a, c) for a, c in ALL_PAIRS if {a, c} & {2, 4}}


# ------------------------------------------------------------------ shape


def test_flags_are_sorted_and_unflagged_keys_absent():
    """One key can carry floor, stale, tail and tail_member together; names are sorted."""
    win = {1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0, 6: 7.0, 7: 8.0, 8: 9.0, 9: 150.0}
    # Runner 9 is the ninth favourite and a tail runner (WIN 150.0); its PLACE price is a floor.
    place = {9: (1.0, 1.5), 1: (1.4, 1.8)}
    wide = {(1, 9): (1.1, 2.0), (1, 2): (3.0, 4.0)}
    b = board(win, place, wide, scratched={10}, age_s=500.0)
    got = avoid_flags(b, rules=RULES, now_utc=NOW)
    assert got[("PLACE", "9")] == ("floor", "stale", "tail")
    assert got[pair_key(1, 9)] == ("floor", "stale", "tail_member")
    assert got[("WIN", "9")] == ("stale", "tail")
    assert got[("WIN", "10")] == ("scratched", "stale")
    assert got[pair_key(1, 2)] == ("stale",)
    for value in got.values():
        assert value == tuple(sorted(value)) and len(set(value)) == len(value)
    # Fresh board, no rule firing: nothing at all, not empty tuples.
    quiet = board(
        {1: 5.0, 2: 6.0, 3: 9.0},
        place={1: (1.5, 2.0), 2: (1.8, 2.4), 3: (2.5, 3.5)},
        wide={(1, 2): (4.0, 5.0), (1, 3): (6.0, 9.0), (2, 3): (7.0, None)},
    )
    assert avoid_flags(quiet, rules=RULES, now_utc=NOW) == {}
    assert all(value for value in got.values())  # no key with an empty tuple


def test_keys_name_wide_pairs_as_ordered_comma_pairs_and_runners_as_numbers():
    b = board({3: 2.0, 7: 150.0, 12: 9.0}, wide={(3, 12): (1.1, 1.3), (7, 12): (2.0, 3.0)})
    got = avoid_flags(b, rules=RULES, now_utc=NOW)
    assert got == {
        ("WIN", "7"): ("tail",),
        (WIDE, "3,12"): ("floor",),
    }


def test_the_result_is_a_pure_function_of_board_rules_and_now():
    b = board(
        TEN_WIN,
        place={4: (1.0, 1.4), 2: (30.0, 40.0)},
        wide=SAFE_WIDE,
        scratched={11},
        age_s=200.0,
    )
    before = copy.deepcopy(b)
    first = avoid_flags(b, rules=RULES, now_utc=NOW)
    second = avoid_flags(b, rules=RULES, now_utc=NOW)
    assert first == second
    assert b == before  # the board is not mutated
    # A different clock changes the answer only through staleness.
    earlier = avoid_flags(b, rules=RULES, now_utc=NOW - timedelta(seconds=100))
    without_stale = {k: tuple(f for f in v if f != "stale") for k, v in first.items()}
    assert earlier == {k: v for k, v in without_stale.items() if v}
    assert any("stale" in v for v in first.values())


# ------------------------------------------------------------------ config


def test_the_shipped_rules_parse():
    loaded = load_flag_rules()
    assert isinstance(loaded, FlagRules)
    assert isinstance(loaded.floor_max_odds, (int, float)) and loaded.floor_max_odds == 1.1
    assert isinstance(loaded.tail_min_odds, (int, float)) and loaded.tail_min_odds == 100.0
    assert isinstance(loaded.tail_rank, int) and loaded.tail_rank == 9
    assert isinstance(loaded.stale_s, (int, float)) and loaded.stale_s == 120
    raw = config.load_country_yaml("japan", "avoid_flags")
    assert {"floor_max_odds", "tail_min_odds", "tail_rank", "stale_s"} <= set(raw)


def test_load_flag_rules_reads_the_given_file(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text(
        "floor_max_odds: 1.3\ntail_min_odds: 80.5\ntail_rank: 7\nstale_s: 45\n", encoding="utf-8"
    )
    loaded = load_flag_rules(path)
    assert loaded == FlagRules(floor_max_odds=1.3, tail_min_odds=80.5, tail_rank=7, stale_s=45.0)
    assert loaded != load_flag_rules()
