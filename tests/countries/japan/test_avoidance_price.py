"""Floor, tail and favourite avoidance rules L2, L3 and L7 (ticket #82; 08 §8 A4, §5.1, §4 E3, E7,
E12, E14, R3; 04 §1.3).

Contract pinned here (`horsetrack.countries.japan.studies.avoidance_price`, built on #81's
`studies.avoidance`: `RaceView`, `Ticket`, `view_at`, `compare`, LOOK_AHEAD, BOARD_AT_T):

- `at_floor(board_odds) -> pd.Series`: True where a board price is at or below 1.1 (the floor,
  owner 2026-11-24); a bool Series on the input's index. It reads any pool's `odds` column (WIN, the
  PLACE and Wide low figure). An unpriced figure (missing, non-finite or below 1.0, as
  `core.pricing.clean_odds` reads it) is never at the floor.
- `plus10_withheld(total_payout, top_ups, sales) -> bool`: True when total payout plus top-ups
  exceed sales; 'exceed' is strict, so equal to sales is not withheld (04 §1.3).
  `predicted_floor_dividend(...) -> int`: 100 when Plus-10 is withheld, else 110. A non-finite
  input is a ValueError for both.
- `win_ranks(view) -> dict[int, int]`: each runner priced in the WIN pool of the view's board -> its
  rank by WIN odds, 1 the shortest. Other pools, `__pool__` rows and unpriced runners are not
  ranked. The fixtures hold no tied prices, so the tie rule is not pinned here.
- `worst_member_rank(combination, rank_at_t) -> int`: the largest rank among the combination's
  horses (2 or 3 legs, any pool's order). A horse without a rank raises KeyError or ValueError.
- `drop_tail(view, tickets, *, max_odds) -> list[Ticket]`: L3's Win/Place rule. A WIN or PLACE
  ticket whose runner's WIN odds on the view's board are above `max_odds` is dropped; at `max_odds`
  it is kept; the rest keep their order. A PLACE ticket is judged by its runner's WIN odds ('a
  runner above 100/1'), as #117's avoid flags read it.
- `entropy_quintile(entropy, cut_points) -> int`: 1 to 5, 1 the lowest normalised entropy (the most
  concentrated market) and 5 the highest (the flattest: L7's 'flattest-entropy quintile' is 5). A
  value equal to a cut point goes to the upper quintile (`numpy.searchsorted(..., side="right") +
  1`, the binning of #77's E3 and #277). Cut points other than four strictly increasing finite
  numbers, or an entropy that is not a number in [0, 1], are a ValueError.
- `OWNER_CHOICES: dict[str, tuple[str, ...]]`: every owner decision the study implements as a named
  option (for example the base strategy's tickets, favourite ties, favourite exotic boxes, the
  Plus-10 inputs), key -> its options, the owner's answer (2026-11-26) first: the synthetic config
  below takes `options[0]` of each. `plus10_inputs` starts with `jravan_pool_votes` (answer 7: the
  per-runner Place stakes are the JRA-VAN H1 votes, the jravan `pool_votes` table of #368).
- `run_study(out_dir)`: reads config/avoidance_price.yaml through
  `load_country_yaml("japan", "avoidance_price")` with no defaults. Its keys are `l3_tail_odds`
  (a list of decimal odds, each its own variant: a runner priced above it is in the tail),
  `l7_entropy_cut_points` (four increasing numbers in (0, 1)) and one key per `OWNER_CHOICES` entry
  holding one of its options. There is no `owner_signoff` (owner reframe 2026-11-27: the freeze line
  in the registry is the record): the study never asks for it, a config that still carries it is
  refused as an unknown key (ValueError) before anything is read or written, and neither the frozen
  spec, the records nor the report cite a sign-off. A missing file or key raises
  `calibration.MissingOwnerDecision` before anything is read or written. Otherwise the study
  freezes one new family in `default_registry("japan")` before any store row is read (its spec
  carries the configured cut points, `variants` and `family_size`), reads development rows only
  (a holdout day's rows and odds file are never opened), appends one record per variant (ticket
  82; hypothesis ids L2, L3 and L7 for the rules; status 'scored' or 'failed', never adopted) and
  writes its report under `out_dir`. Each record's `components["selection"]` says what chose the
  tickets: #81's LOOK_AHEAD for the final price, a label starting with #81's BOARD_AT_T for the
  board at t (odds_timeseries through `read_asof` at post_utc - 2 min, D2). Every L2, L3 and L7
  record carries one of the two.
- Every option is a ranked row (owner reframe 2026-11-27: no D18 pass/fail, no adoption; an avoid
  rule drops nothing from the data, the option with the flagged tickets and the one without both
  stay rows). Every scored record carries the family size, the variants-searched count, as a
  metric or a param whose name holds `variants_searched` or `family_size`. Each scored paired
  comparison (every L2, L3 and L7 record but the 100-vs-110 check) holds the paired difference (a
  metric named with `difference`, with its clustered `lo` and `hi`) and the rule's and the
  comparator's own mean return per unit (named with `rule` or `comparator` and `return`) over the
  same n, whose difference it is. No written line says pass or adopted unless it negates it.
- The 100-vs-110 check under `jravan_pool_votes`: for each floor-bound placed favourite at the
  final price (the favourite on the final Win odds whose Place paid at most 110), the Plus-10
  prediction from its race's PLACE pool in
  `$HORSETRACK_DATA_DIR/processed/jravan/pool_votes.parquet` (`jravan.votes.POOL_VOTES_SCHEMA`, a
  label-only table read for the settled race, never at t; development races only). Sales are the
  `__pool__` row's votes less its `refund_votes`, x 100 yen. JRA's Place rule: the net pool (80% of
  sales, the Place takeout since 2014-06-07) first returns the placed horses' stakes (votes x 100),
  the profit left is split equally over the placed horses and paid per ticket rounded down to 10
  yen, and a horse whose share is under 10 yen a ticket (or with nothing left) is paid 100. The
  top-ups are 10 yen a ticket on each placed horse paid 100; then `predicted_floor_dividend`'s
  rule. The record's metrics hold one count per cell of predicted (100, 110) x realised (100, 110,
  above 110), named `predicted_<p>_realised_<r>` (r `100`, `110` or `above_110`, either order),
  with n the favourites predicted; a cell left out counts 0. The fixture's pools sit far from every
  boundary (payout plus top-ups at 85% or 107% of sales), so any faithful reading of the rule gives
  the same predictions, and each race's WIN pool tells the opposite story, so a prediction read
  from the WIN rows flips.
- D21: the votes are JRA-VAN's (`core.eval.reports.WITHHELD_SOURCES`). The 100-vs-110 record, and
  every record whose figures move with the votes, declares a withheld source in `sources`; the L3
  and L7 records read the jra store alone and do not. Every file the study writes goes through
  `core.eval.reports.write_report`, and every written text that moves with the votes is written
  with a withheld source, so no public (jra-only) output carries a JRA-VAN figure. The holdout
  day's votes rows move nothing and no written text names a holdout day.

How the run-level tests prove it without fixing the owner's betting rules: the same synthetic
store is run under one change at a time and the registry records are compared. Rows not available
at t, and the final odds, must not move any board-at-t record; the board at t must not move any
record that is not labelled board at t; the final odds move only look-ahead records; the holdout
day's rows move nothing; the cut points move only L7; the Place votes move only JRA-VAN records
and files. The store appears only when the family is frozen, so a read before the freeze fails.

Expected values come from first principles: the ranks, floors and tails from the fixture prices
written below, the quintiles from the convention above, entropies from a reference formula, the
Plus-10 predictions from a reference Place payout. Every frame is synthetic; no date is a Japan
holdout day (on or after 2026-11-28). The holdout day of the run-level tests is a synthetic one,
declared in memory by patching the holdout definition.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import copy
import itertools
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core import config as core_config
from horsetrack.core.eval import holdout as holdout_module
from horsetrack.core.eval import reports as reports_module
from horsetrack.core.eval.registry import Registry
from horsetrack.core.eval.reports import is_withheld
from horsetrack.core.store import schemas
from horsetrack.core.store.asof import load_availability
from horsetrack.core.store.tables import ARROW_TYPES
from horsetrack.countries.japan.jravan import votes as jravan_votes
from horsetrack.countries.japan.logger.compact import SCHEMA as ODDS_SCHEMA
from horsetrack.countries.japan.odds_log import POOL_TOTAL
from horsetrack.countries.japan.studies import avoidance, avoidance_price
from horsetrack.countries.japan.studies.avoidance import (
    BOARD_AT_T,
    LOOK_AHEAD,
    RaceView,
    Ticket,
    view_at,
)
from horsetrack.countries.japan.studies.avoidance_price import (
    at_floor,
    drop_tail,
    entropy_quintile,
    plus10_withheld,
    predicted_floor_dividend,
    win_ranks,
    worst_member_rank,
)
from horsetrack.countries.japan.studies.calibration import MissingOwnerDecision

UTC = timezone.utc
JST = timezone(timedelta(hours=9))
NAN = float("nan")
DECISION_MIN = 2  # D2: config/decision.yaml primary_min
SHIPPED_LOAD = core_config.load_country_yaml
CONFIG_NAME = "avoidance_price"
RULES = ("L2", "L3", "L7")


def keys_where(board: pd.DataFrame, flags: pd.Series) -> set[tuple[str, str]]:
    """The (pool, combination) of the board rows a flag Series marks True."""
    assert flags.index.equals(board.index)
    picked = board.loc[flags.to_numpy(dtype=bool)]
    return set(zip(picked["pool"], picked["combination"], strict=True))


def reference_entropy(odds: dict[int, float]) -> float:
    """Shannon entropy of the normalised inverse odds over ln n (E3's normalised entropy)."""
    inv = [1.0 / o for o in odds.values()]
    total = sum(inv)
    return -sum(x / total * math.log(x / total) for x in inv) / math.log(len(inv))


# ---------------------------------------------------------------------------------------------
# One race's board at t and its final board (14 runners), read through #81's view_at
# ---------------------------------------------------------------------------------------------

AT_RACE = "jra:2026-06-13:TOK:11"
AT_POST = datetime(2026, 6, 13, 6, 25, tzinfo=UTC)  # 15:25 JST
AT_T = AT_POST - timedelta(minutes=DECISION_MIN)

# WIN odds at t (capture at T-3, fetched before t) and at the final. No two runners share a price.
WIN_T = {
    1: 8.8, 2: 1.1, 3: 48.0, 4: 3.4, 5: 101.0, 6: 22.0, 7: 6.2,
    8: 180.0, 9: 15.0, 10: 51.0, 11: 12.5, 12: 100.0, 13: 30.5, 14: 75.0,
}  # fmt: skip
WIN_FINAL = {
    1: 2.9, 2: 1.4, 3: 9.5, 4: 1.1, 5: 60.0, 6: 14.0, 7: 25.0,
    8: 70.0, 9: 120.0, 10: 33.0, 11: 49.0, 12: 52.0, 13: 102.0, 14: 160.0,
}  # fmt: skip
WIN_EARLY = {**WIN_T, 2: 1.3, 5: 99.0, 8: 150.0, 12: 110.0}  # T-20: superseded by the T-3 capture
# rank by WIN odds, 1 the shortest, worked by hand from the prices above
RANKS_T = {
    2: 1,
    4: 2,
    7: 3,
    1: 4,
    11: 5,
    9: 6,
    6: 7,
    13: 8,
    3: 9,
    10: 10,
    14: 11,
    12: 12,
    5: 13,
    8: 14,
}
RANKS_FINAL = {
    4: 1,
    2: 2,
    1: 3,
    3: 4,
    6: 5,
    7: 6,
    10: 7,
    11: 8,
    12: 9,
    5: 10,
    8: 11,
    13: 12,
    9: 13,
    14: 14,
}
WIDE_T = {"2,4": (1.1, 1.3), "2,7": (1.6, 2.0), "1,4": (2.4, 3.1), "3,8": (88.0, 120.0)}
WIDE_FINAL = {"2,4": (1.2, 1.4), "2,7": (2.2, 2.6), "1,4": (1.1, 1.3), "3,8": (60.0, 80.0)}
WIDE_EARLY = {**WIDE_T, "2,4": (1.3, 1.5)}


def place_of(win: float) -> tuple[float, float]:
    """A PLACE (low, high) figure for a runner at `win`: well under its WIN odds for a long shot."""
    low = max(1.0, round(1.0 + (win - 1.0) / 4.5, 1))
    return low, round(low * 1.4, 1)


def wide_of(a: float, b: float) -> tuple[float, float]:
    low = max(1.0, round(math.sqrt(a * b) / 2.2, 1))
    return low, round(low * 1.5, 1)


def odds_row(race_id, post, snap, fetched, pool, combination, odds, odds_hi=None, **extra) -> dict:
    """One odds_timeseries row (logger/compact.py SCHEMA)."""
    _, _, venue, race_no = race_id.split(":")
    row = {
        "race_id": race_id,
        "venue": venue,
        "race_no": int(race_no),
        "pool": pool,
        "combination": combination,
        "snapshot_utc": snap,
        "fetched_utc": fetched,
        "post_utc": post,
        "minutes_to_post": (post - snap).total_seconds() / 60,
        "odds": odds,
        "odds_hi": odds_hi,
        "investment": None,
        "status_raw": "",
        "is_final_capture": False,
        "dividend": None,
        "is_refund": False,
    }
    return row | extra


def at_race_capture(snap, fetched, win, wide, *, final=False, totals=1.0) -> list[dict]:
    rows = []
    for h, o in win.items():
        rows.append(odds_row(AT_RACE, AT_POST, snap, fetched, "WIN", str(h), o))
        low, high = place_of(o)
        rows.append(odds_row(AT_RACE, AT_POST, snap, fetched, "PLACE", str(h), low, high))
    for pair, (low, high) in wide.items():
        rows.append(odds_row(AT_RACE, AT_POST, snap, fetched, "QUINELLA_PLACE", pair, low, high))
    for pool, total in (("WIN", 9.0e7), ("PLACE", 6.0e7), ("QUINELLA_PLACE", 4.0e7)):
        rows.append(
            odds_row(
                AT_RACE, AT_POST, snap, fetched, pool, POOL_TOTAL, None, investment=total * totals
            )
        )
    return [r | {"is_final_capture": final} for r in rows]


def at_race_rows() -> list[dict]:
    post, s = AT_POST, timedelta(seconds=1)
    return [
        *at_race_capture(post - timedelta(minutes=20), post - timedelta(minutes=20) + 3 * s,
                         WIN_EARLY, WIDE_EARLY, totals=0.3),
        *at_race_capture(post - timedelta(minutes=3), post - timedelta(minutes=3) + 2 * s,
                         WIN_T, WIDE_T, totals=0.8),
        # taken before t, fetched after it: not on the board at t
        *at_race_capture(post - timedelta(minutes=2, seconds=20), post - timedelta(seconds=100),
                         WIN_FINAL, WIDE_FINAL, totals=0.9),
        *at_race_capture(post + timedelta(minutes=1), post + timedelta(minutes=1) + 5 * s,
                         WIN_FINAL, WIDE_FINAL, final=True),
    ]  # fmt: skip


@pytest.fixture(scope="module")
def boards() -> tuple[RaceView, RaceView]:
    """The race's board at t (#81's view_at over odds_timeseries) and its final board."""
    con = duckdb.connect(":memory:")
    try:
        con.register("_rows", pa.Table.from_pylist(at_race_rows(), schema=ODDS_SCHEMA))
        con.execute('CREATE TABLE "odds_timeseries" AS SELECT * FROM _rows')
        con.unregister("_rows")
        at_t = view_at(con, AT_RACE, AT_T, rules=load_availability("japan"))
    finally:
        con.close()
    final_rows = [r for r in at_race_rows() if r["is_final_capture"]]
    final = RaceView(
        AT_RACE,
        pd.DataFrame(final_rows)[["pool", "combination", "odds", "odds_hi", "investment"]],
        LOOK_AHEAD,
    )
    # the fixture: the board at t holds the T-3 capture's prices
    win = at_t.board.loc[at_t.board["pool"].eq("WIN") & at_t.board["combination"].ne(POOL_TOTAL)]
    assert dict(zip(win["combination"].astype(int), win["odds"], strict=True)) == WIN_T
    return at_t, final


# ---------------------------------------------------------------------------------------------
# L2: the floor
# ---------------------------------------------------------------------------------------------

FLOOR_TABLE = [  # (board price, at the floor)
    (1.0, True),
    (1.1, True),
    (1.2, False),
    (1.3, False),
    (1.5, False),
    (2.0, False),
    (9.9, False),
    (51.0, False),
    (100.0, False),
    (101.0, False),
]


def test_floor_rule_flags_board_prices_at_the_floor(boards):
    assert at_floor(pd.Series([1.0, 1.1, 1.2])).tolist() == [True, True, False]

    index = pd.Index([11, 3, 7, 1, 15, 2, 9, 4, 12, 6], name="horse_no")
    odds = pd.Series([o for o, _ in FLOOR_TABLE], index=index, name="odds")
    got = at_floor(odds)
    assert isinstance(got, pd.Series)
    assert got.index.equals(index)
    assert got.dtype == bool
    assert got.tolist() == [flag for _, flag in FLOOR_TABLE]
    # each price keeps its own label, in any order
    backwards = at_floor(odds.iloc[::-1])
    assert backwards.index.equals(index[::-1])
    assert backwards.tolist() == [flag for _, flag in reversed(FLOOR_TABLE)]

    # an unpriced figure is never at the floor
    unpriced = pd.Series([NAN, None, 0.0, 0.5, -1.1, math.inf], index=list("abcdef"), dtype=float)
    got = at_floor(unpriced)
    assert got.index.equals(unpriced.index)
    assert not got.any()

    # On the board at t: WIN, the PLACE low figure and the Wide low figure, as priced at t, not at
    # the final; the pool-total rows carry no price.
    at_t, final = boards
    assert keys_where(at_t.board, at_floor(at_t.board["odds"])) == {
        ("WIN", "2"),
        ("PLACE", "2"),
        ("QUINELLA_PLACE", "2,4"),
    }
    assert keys_where(final.board, at_floor(final.board["odds"])) == {
        ("WIN", "4"),
        ("PLACE", "4"),
        ("PLACE", "2"),  # 1.4 to win, 1.1 to place at the final
        ("QUINELLA_PLACE", "1,4"),
    }


# ---------------------------------------------------------------------------------------------
# Plus-10 (04 §1.3): the floor pays 110 unless payouts plus top-ups would exceed sales
# ---------------------------------------------------------------------------------------------

PLUS10_TABLE = [  # (total payout, top-ups, sales, withheld)
    (990, 20, 1000, True),  # the ticket's case: 1,010 exceeds 1,000
    (970, 20, 1000, False),  # the ticket's case: 990
    (980, 20, 1000, False),  # exactly the sales: not exceeded
    (980.01, 20, 1000, True),
    (990, 20, 1010, False),  # exactly the sales again, at other sales
    (990, 20, 1009, True),
    (1000, 0, 1000, False),
    (1000.5, 0, 1000, True),  # the payout alone exceeds the sales
    (500, 600, 1000, True),  # the top-ups push it over
    (0, 0, 1000, False),
    (79_990_000, 10_020_000, 90_000_000, True),
    (79_990_000, 10_010_000, 90_000_000, False),
    (52_300_000, 1_200_000, 61_000_000, False),
    (60_000_000, 1_000_001, 61_000_000, True),
]


def test_plus10_is_withheld_when_top_ups_would_exceed_sales():
    assert plus10_withheld(990, 20, 1000) is True
    assert plus10_withheld(970, 20, 1000) is False
    for payout, top_ups, sales, withheld in PLUS10_TABLE:
        got = plus10_withheld(payout, top_ups, sales)
        assert isinstance(got, bool), (payout, top_ups, sales)
        assert got is withheld, (payout, top_ups, sales)
        got = plus10_withheld(float(payout), float(top_ups), float(sales))
        assert got is withheld, (payout, top_ups, sales)


def test_predicted_floor_dividend_is_100_only_when_plus10_is_withheld():
    assert predicted_floor_dividend(990, 20, 1000) == 100
    assert predicted_floor_dividend(970, 20, 1000) == 110
    for payout, top_ups, sales, withheld in PLUS10_TABLE:
        got = predicted_floor_dividend(payout, top_ups, sales)
        assert isinstance(got, int) and not isinstance(got, bool), (payout, top_ups, sales)
        assert got == (100 if withheld else 110), (payout, top_ups, sales)
        assert got == (100 if plus10_withheld(payout, top_ups, sales) else 110)


def test_plus10_refuses_an_input_that_is_not_a_number():
    """Ruling: a missing pool figure is a ValueError, never a silent 110 (NaN > x is False)."""
    for bad in (NAN, math.inf, -math.inf):
        for args in ((bad, 20, 1000), (990, bad, 1000), (990, 20, bad)):
            with pytest.raises(ValueError):
                plus10_withheld(*args)
            with pytest.raises(ValueError):
                predicted_floor_dividend(*args)


# ---------------------------------------------------------------------------------------------
# L3: the tail (worst member rank, and runners above the cut-off) on the board at t
# ---------------------------------------------------------------------------------------------

RANKS = {
    1: 6,
    2: 1,
    3: 14,
    4: 2,
    5: 9,
    6: 11,
    7: 3,
    8: 12,
    9: 4,
    10: 13,
    11: 5,
    12: 8,
    13: 7,
    14: 10,
}
WORST_CASES = [  # (combination, its worst member's rank under RANKS), worked by hand
    ("2,4", 2),
    ("1,2", 6),
    ("4,9", 4),
    ("5,7", 9),  # the first leg
    ("9,10", 13),
    ("10,9", 13),  # an ordered pool's order
    ("3,14", 14),  # the lower horse number
    ("12,13", 8),
    ("2,4,7", 3),
    ("7,2,4", 3),
    ("11,12,13", 8),  # the middle leg
    ("1,5,6", 11),
    ("6,1,5", 11),
    ("2,3,4", 14),
    ("8,13,14", 12),
]


def test_worst_member_rank_is_the_worst_ranked_member():
    for combination, worst in WORST_CASES:
        got = worst_member_rank(combination, RANKS)
        assert isinstance(got, int) and not isinstance(got, bool), combination
        assert got == worst, combination
    # other ranks for the same horses give other answers
    mirrored = {h: 15 - r for h, r in RANKS.items()}
    for combination, _ in WORST_CASES:
        want = max(mirrored[int(h)] for h in combination.split(","))
        assert worst_member_rank(combination, mirrored) == want, combination


def test_worst_member_rank_refuses_a_horse_without_a_rank():
    ranks = {h: r for h, r in RANKS.items() if h != 6}
    for combination in ("2,6", "6,2", "6,2,4", "1,4,6"):
        with pytest.raises((KeyError, ValueError)):
            worst_member_rank(combination, ranks)


WORST_AT_T = [  # (combination, worst rank on the board at t, worst rank on the final board)
    ("2,4", 2, 2),
    ("3,6", 9, 5),  # a tail member (rank 9 or worse) at t only
    ("2,3,6", 9, 5),
    ("10,11", 10, 8),
    ("4,7,11", 5, 8),
    ("1,9,13", 8, 13),  # a tail member at the final only
    ("12,5", 13, 10),  # rank 13 or worse at t only
    ("5,10", 13, 10),
    ("12,13,14", 12, 14),  # rank 13 or worse at the final only
    ("8,14", 14, 14),
]


def test_worst_member_rank_uses_the_board_at_t(boards):
    at_t, final = boards
    ranks_t, ranks_final = win_ranks(at_t), win_ranks(final)

    # every runner priced in the WIN pool, ranked by its own board's WIN odds; the pool-total
    # row and the PLACE and Wide rows are not runners
    assert ranks_t == RANKS_T
    assert ranks_final == RANKS_FINAL
    assert ranks_t != ranks_final

    for combination, worst_t, worst_final in WORST_AT_T:
        assert worst_member_rank(combination, ranks_t) == worst_t, combination
        assert worst_member_rank(combination, ranks_final) == worst_final, combination
    for cut in (9, 13):  # the request's rank cut-offs select other combinations at t
        at_t_tail = {c for c, w, _ in WORST_AT_T if w >= cut}
        final_tail = {c for c, _, w in WORST_AT_T if w >= cut}
        assert at_t_tail != final_tail


ABOVE_T = {  # runners whose WIN odds at t are above the cut-off, by hand from WIN_T
    51.0: {5, 8, 12, 14},  # 10 is at 51.0, kept
    100.0: {5, 8},  # 12 is at 100.0, kept
    101.0: {8},  # 5 is at 101.0, kept
}
ABOVE_FINAL = {51.0: {5, 8, 9, 12, 13, 14}, 100.0: {9, 13, 14}, 101.0: {9, 13, 14}}
TICKET_ORDER = (3, 14, 1, 8, 5, 12, 10, 2, 13, 4, 9, 6, 11, 7)


def test_tail_rule_drops_runners_above_the_cutoff_at_t(boards):
    at_t, final = boards
    tickets = [
        *(Ticket("WIN", str(h), 100) for h in TICKET_ORDER[:7]),
        *(Ticket("PLACE", str(h), 300) for h in TICKET_ORDER),
        *(Ticket("WIN", str(h), 100) for h in TICKET_ORDER[7:]),
    ]
    # A Place ticket is judged by its runner's WIN odds: no tail runner is above any cut-off on
    # the PLACE board itself.
    place = at_t.board.loc[at_t.board["pool"].eq("PLACE"), "odds"]
    assert float(place.max()) < 51.0

    for cut in (51.0, 100.0, 101.0):
        got = drop_tail(at_t, tickets, max_odds=cut)
        assert got == [t for t in tickets if int(t.combination) not in ABOVE_T[cut]], cut
        got_final = drop_tail(final, tickets, max_odds=cut)
        assert got_final == [t for t in tickets if int(t.combination) not in ABOVE_FINAL[cut]], cut
        assert got != got_final, cut
    assert drop_tail(at_t, [], max_odds=100.0) == []


# ---------------------------------------------------------------------------------------------
# L7: entropy quintiles against frozen cut points
# ---------------------------------------------------------------------------------------------

QUINTILE_TABLE = [  # (cut points, entropy, quintile): 1 most concentrated, 5 flattest
    ((0.70, 0.75, 0.79, 0.82), 0.0, 1),
    ((0.70, 0.75, 0.79, 0.82), 0.35, 1),
    ((0.70, 0.75, 0.79, 0.82), 0.6999, 1),
    ((0.70, 0.75, 0.79, 0.82), 0.70, 2),
    ((0.70, 0.75, 0.79, 0.82), 0.7001, 2),
    ((0.70, 0.75, 0.79, 0.82), 0.7499, 2),
    ((0.70, 0.75, 0.79, 0.82), 0.75, 3),
    ((0.70, 0.75, 0.79, 0.82), 0.77, 3),
    ((0.70, 0.75, 0.79, 0.82), 0.7899, 3),
    ((0.70, 0.75, 0.79, 0.82), 0.79, 4),
    ((0.70, 0.75, 0.79, 0.82), 0.80, 4),
    ((0.70, 0.75, 0.79, 0.82), 0.8199, 4),
    ((0.70, 0.75, 0.79, 0.82), 0.82, 5),
    ((0.70, 0.75, 0.79, 0.82), 0.8201, 5),
    ((0.70, 0.75, 0.79, 0.82), 0.95, 5),
    ((0.70, 0.75, 0.79, 0.82), 1.0, 5),
    ((0.2, 0.4, 0.6, 0.8), 0.0, 1),
    ((0.2, 0.4, 0.6, 0.8), 0.1999, 1),
    ((0.2, 0.4, 0.6, 0.8), 0.2, 2),
    ((0.2, 0.4, 0.6, 0.8), 0.3, 2),
    ((0.2, 0.4, 0.6, 0.8), 0.4, 3),
    ((0.2, 0.4, 0.6, 0.8), 0.5999, 3),
    ((0.2, 0.4, 0.6, 0.8), 0.6, 4),
    ((0.2, 0.4, 0.6, 0.8), 0.7999, 4),
    ((0.2, 0.4, 0.6, 0.8), 0.8, 5),
    ((0.2, 0.4, 0.6, 0.8), 1.0, 5),
    ((0.5, 0.9, 0.95, 0.99), 0.49, 1),
    ((0.5, 0.9, 0.95, 0.99), 0.5, 2),
    ((0.5, 0.9, 0.95, 0.99), 0.8, 2),
    ((0.5, 0.9, 0.95, 0.99), 0.9, 3),
    ((0.5, 0.9, 0.95, 0.99), 0.949, 3),
    ((0.5, 0.9, 0.95, 0.99), 0.95, 4),
    ((0.5, 0.9, 0.95, 0.99), 0.989, 4),
    ((0.5, 0.9, 0.95, 0.99), 0.99, 5),
]


def test_entropy_quintile_bins_against_the_cut_points():
    for cuts, entropy, quintile in QUINTILE_TABLE:
        got = entropy_quintile(entropy, cuts)
        assert isinstance(got, int) and not isinstance(got, bool), (cuts, entropy)
        assert got == quintile, (cuts, entropy)
    # one entropy, three sets of cut points: the cut points decide, not the value alone
    assert entropy_quintile(0.8, (0.70, 0.75, 0.79, 0.82)) == 4
    assert entropy_quintile(0.8, (0.2, 0.4, 0.6, 0.8)) == 5
    assert entropy_quintile(0.8, (0.5, 0.9, 0.95, 0.99)) == 2


def test_entropy_quintile_refuses_cut_points_that_are_not_four_increasing_numbers():
    assert entropy_quintile(0.5, (0.2, 0.4, 0.6, 0.8)) == 3
    for cuts in (
        (),
        (0.2, 0.4, 0.6),
        (0.2, 0.4, 0.6, 0.8, 0.9),
        (0.4, 0.2, 0.6, 0.8),
        (0.2, 0.4, 0.4, 0.8),
        (0.8, 0.6, 0.4, 0.2),
        (0.2, NAN, 0.6, 0.8),
        (0.2, 0.4, 0.6, math.inf),
    ):
        with pytest.raises(ValueError):
            entropy_quintile(0.5, cuts)


def test_entropy_quintile_refuses_an_entropy_it_cannot_bin():
    """Ruling: a race without a normalised entropy (NaN, under two priced runners) or a value
    outside [0, 1] (an unnormalised entropy) has no quintile."""
    for entropy in (NAN, -0.01, 1.01, 2.3, math.inf):
        with pytest.raises(ValueError):
            entropy_quintile(entropy, (0.2, 0.4, 0.6, 0.8))


# ---------------------------------------------------------------------------------------------
# The study: a synthetic store, run under one change at a time
# ---------------------------------------------------------------------------------------------

NUMERIC_KEYS = ("l3_tail_odds", "l7_entropy_cut_points")
SIGNOFF = "owner_signoff"  # removed by the owner's reframe of 2026-11-27
TAIL_ODDS = [101.0, 51.0]
CUTS = [0.5, 0.65, 0.8, 0.9]
F_CUTS = [0.5, 0.65, 0.8, 0.95]  # only the last cut point moves


def owner_config(**changes) -> dict:
    """A synthetic config/avoidance_price.yaml: every key the study needs, the first option of each
    choice the study offers, and no `owner_signoff`."""
    raw = {key: options[0] for key, options in avoidance_price.OWNER_CHOICES.items()}
    raw |= {"l3_tail_odds": list(TAIL_ODDS), "l7_entropy_cut_points": list(CUTS)}
    return raw | changes


def use_owner_config(mp: pytest.MonkeyPatch, raw: dict | None) -> None:
    """config/avoidance_price.yaml reads as `raw` (None: the file does not exist); every other
    config as shipped."""

    def load(country, name, *args, **kwargs):
        if name != CONFIG_NAME:
            return SHIPPED_LOAD(country, name, *args, **kwargs)
        if raw is None:
            raise FileNotFoundError(f"countries/{country}/config/{name}.yaml")
        return copy.deepcopy(raw)

    mp.setattr(core_config, "load_country_yaml", load)
    mp.setattr(avoidance_price, "load_country_yaml", load, raising=False)


HORSES = tuple(range(1, 11))
BRACKETS = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 7, 9: 8, 10: 8}  # JRA's draw, 10 runners
POST_LOCAL = {1: "10:05", 2: "10:35"}  # JST
PROFILES = {  # ten WIN prices, shortest first, and their normalised entropy
    "conc": (1.1, 8.0, 12.0, 20.0, 35.0, 48.0, 66.0, 90.0, 140.0, 210.0),  # 0.465
    "conc2": (1.4, 4.5, 9.0, 14.0, 22.0, 30.0, 45.0, 70.0, 110.0, 160.0),  # 0.611
    "mod": (2.4, 4.8, 6.5, 9.0, 13.0, 18.0, 28.0, 45.0, 80.0, 130.0),  # 0.782
    "semi": (3.8, 5.0, 6.1, 7.4, 9.2, 11.5, 14.0, 19.0, 26.0, 40.0),  # 0.914
    "flat": (5.2, 6.0, 7.1, 8.3, 9.4, 10.8, 12.5, 14.9, 17.6, 21.0),  # 0.961
}


def priced(profile: str, *first: int) -> dict[int, float]:
    """Ten runners priced by `profile`: the horses `first` take its shortest prices in that order,
    the rest follow by horse number."""
    order = [*first, *(h for h in HORSES if h not in first)]
    assert sorted(order) == list(HORSES)
    return dict(zip(order, PROFILES[profile], strict=True))


def rid(day: str, venue: str, race_no: int) -> str:
    return f"jra:{day}:{venue}:{race_no:02d}"


@dataclass(frozen=True)
class Race:
    race_id: str
    weight_type: str
    finish: tuple[int, int, int]  # the first three; the others finish in horse-number order
    final: dict[int, float]  # final WIN odds (runners.win_odds)
    at_t: dict[int, float] | None = None  # the WIN board at t: logged races only
    place: dict[int, int] = field(default_factory=dict)  # PLACE dividends other than the default
    scale: int = 1  # dividends multiplied (the holdout day's poison)


D1, D2, D3, D4 = "2026-06-06", "2026-06-07", "2026-06-13", "2026-06-14"
HOLDOUT_DAY = "2026-06-20"  # a synthetic holdout day, declared in memory only

HISTORY = [  # no board logged: only the final price exists
    # the floor favourite (3, at 1.1) runs second
    Race(rid(D1, "TOK", 1), "age", (7, 3, 1), priced("conc", 3, 7, 1)),
    # a handicap: the favourite (5) runs second, the 130/1 shot (10) third
    Race(rid(D1, "TOK", 2), "handicap", (2, 5, 10), priced("mod", 5, 2)),
    # a flat market (0.914): the favourite (6) runs second
    Race(rid(D1, "KYO", 1), "set", (1, 6, 8), priced("semi", 6, 1)),
    Race(rid(D1, "KYO", 2), "allowance", (9, 2, 7), priced("flat", 4)),
    Race(rid(D2, "TOK", 1), "handicap", (1, 6, 10), priced("conc2", 1)),
    # a floor favourite wins, and its Place pays 100 though its votes predict 110
    Race(rid(D2, "TOK", 2), "age", (8, 2, 5), priced("conc", 8), place={8: 100}),
    Race(rid(D2, "KYO", 1), "handicap", (10, 4, 2), priced("semi", 2, 4)),
    Race(rid(D2, "KYO", 2), "set", (7, 1, 3), priced("mod", 7, 1, 3)),
]
LOGGED = [  # a WIN, PLACE and Wide board logged: the board at t and the final price
    # floor at t (2 at 1.1, placed second), not at the final
    Race(rid(D3, "TOK", 1), "age", (5, 2, 8), priced("conc2", 2, 5, 8), priced("conc", 2, 5, 8)),
    # a handicap: favourite at t (4) third; the 130/1 shot at t (9) second, 45/1 at the final
    Race(
        rid(D3, "TOK", 2),
        "handicap",
        (1, 9, 4),
        priced("mod", 4, 1, 3, 5, 6, 7, 2, 9, 8, 10),
        priced("mod", 4, 1, 3, 5, 6, 7, 2, 8, 10, 9),
    ),
    # a flat board at t (0.914): the favourite (5) second
    Race(rid(D3, "KYO", 1), "set", (3, 5, 10), priced("semi", 5, 3, 10), priced("semi", 5, 3, 10)),
    # a floor favourite wins, and its Place pays 100: Plus-10 withheld
    Race(
        rid(D3, "KYO", 2),
        "allowance",
        (1, 6, 2),
        priced("conc", 1, 6, 2),
        priced("conc", 1, 6, 2),
        place={1: 100},
    ),
    Race(
        rid(D4, "TOK", 1), "handicap", (3, 7, 1), priced("flat", 3, 7, 1), priced("flat", 3, 7, 1)
    ),
    # the 160/1 shot at t (4) wins
    Race(
        rid(D4, "TOK", 2),
        "age",
        (4, 2, 10),
        priced("conc2", 10, 2, 3, 5, 6, 7, 8, 9, 4, 1),
        priced("conc2", 10, 2, 3, 5, 6, 7, 8, 9, 1, 4),
    ),
    Race(rid(D4, "KYO", 1), "set", (8, 1, 9), priced("mod", 6, 8, 1, 9), priced("mod", 6, 8, 1, 9)),
    Race(
        rid(D4, "KYO", 2), "handicap", (2, 9, 5), priced("semi", 9, 2, 5), priced("semi", 9, 2, 5)
    ),
]
HOLDOUT = [  # poison: if a holdout row is read, every result moves
    Race(rid(HOLDOUT_DAY, "TOK", 1), "handicap", (1, 2, 3), priced("conc", 1), scale=50),
    Race(rid(HOLDOUT_DAY, "TOK", 2), "age", (4, 2, 9), priced("semi", 9, 2), scale=50),
]
H1, H2, H3, H4, H5, H6, H7, H8 = (race.race_id for race in HISTORY)
G1, G2, G3, G4, G5, G6, G7, G8 = (race.race_id for race in LOGGED)
RACES = {race.race_id: race for race in (*HISTORY, *LOGGED, *HOLDOUT)}

# E: other final odds on history, the winners' prices and every dividend unchanged
E_FINAL = {
    H1: {3: 1.6},  # no floor runner any more
    H2: {5: 13.5, 4: 2.2, 10: 44.0},  # the favourite is now 4 (unplaced); 10 is out of the tail
    H3: {6: 1.3, 2: 18.3, 3: 22.2, 4: 27.6, 5: 34.5, 7: 42.0, 8: 57.0, 9: 78.0, 10: 120.0},
}
# C: another board at t (the T-3 capture only), the final price unchanged
C_AT_T = {
    G1: priced("conc", 6, 5, 8, 2),  # the floor runner at t is 6 (unplaced), not 2 (placed)
    G2: priced("mod", 7, 1, 3, 5, 6, 4, 2, 9, 8, 10),  # favourite 7 (unplaced); 9 at 45/1
    G3: priced("conc", 5, 3, 10),  # concentrated, and 5 at the floor
    G4: priced("flat", 1, 6, 2),  # flat, no floor
    G5: priced("conc", 8, 3, 7, 1),  # favourite 8 (unplaced)
    G6: priced("conc2", 10, 2, 4, 5, 6, 7, 8, 9, 1, 3),  # 4 at 9.0, the tail runner is 3
    G7: priced("mod", 1, 8, 6, 9),  # favourite 1 (placed) instead of 6
}


def r10(x: float) -> int:
    return int(round(x / 10.0)) * 10


def dividend_rows(race: Race) -> list[dict]:
    """The race's paying rows in every pool, from its own (base) final odds."""
    o, (first, second, third) = race.final, race.finish
    top3 = race.finish
    rows = [("WIN", str(first), max(100, r10(100 * o[first])))]
    rows += [
        ("PLACE", str(h), race.place.get(h, max(110, r10(100 + 25 * (o[h] - 1))))) for h in top3
    ]
    for a, b in itertools.combinations(sorted(top3), 2):
        rows.append(("QUINELLA_PLACE", f"{a},{b}", max(110, r10(50 * math.sqrt(o[a] * o[b])))))
    quinella = max(110, r10(100 * o[first] * o[second] / 3))
    rows.append(("QUINELLA", ",".join(map(str, sorted((first, second)))), quinella))
    rows.append(("FORECAST", f"{first},{second}", 2 * quinella))
    trio = max(110, r10(100 * o[first] * o[second] * o[third] / 15))
    rows.append(("TRIO", ",".join(map(str, sorted(top3))), trio))
    rows.append(("TIERCE", f"{first},{second},{third}", 6 * trio))
    b1, b2 = sorted((BRACKETS[first], BRACKETS[second]))
    rows.append(("BRACKET_QUINELLA", f"{b1},{b2}", max(110, r10(0.8 * quinella))))
    return [
        {
            "jurisdiction": "jra",
            "race_id": race.race_id,
            "pool": pool,
            "combination": combination,
            "dividend": float(dividend * race.scale),
            "is_refund": False,
        }
        for pool, combination, dividend in rows
    ]


def race_keys(race_id: str) -> dict:
    _, day, venue, race_no = race_id.split(":")
    return {
        "jurisdiction": "jra",
        "race_id": race_id,
        "meeting_id": f"jra:{day}:{venue}",
        "race_date": date.fromisoformat(day),
        "venue": venue,
        "race_no": int(race_no),
    }


def runner_rows(race: Race, final: dict[int, float]) -> list[dict]:
    order = [*race.finish, *(h for h in HORSES if h not in race.finish)]
    position = {h: i + 1 for i, h in enumerate(order)}
    favourite = {h: i + 1 for i, h in enumerate(sorted(final, key=final.get))}
    return [
        race_keys(race.race_id)
        | {
            "horse_id": f"{race.race_id.replace(':', '-')}-{h:02d}",
            "horse_no": h,
            "placing_raw": str(position[h]),
            "finish_pos": position[h],
            "dead_heat": False,
            "win_odds": final[h],
            "is_scratched": False,
            "bracket": BRACKETS[h],
            "win_fav_rank": favourite[h],
        }
        for h in HORSES
    ]


def post_of(race_id: str) -> datetime:
    _, day, _, race_no = race_id.split(":")
    hh, mm = (int(x) for x in POST_LOCAL[int(race_no)].split(":"))
    return datetime.fromisoformat(day).replace(hour=hh, minute=mm, tzinfo=JST).astimezone(UTC)


def board_capture(race_id, snap, fetched, win, *, final=False, totals=1.0, paying=None) -> list:
    post, paying = post_of(race_id), paying or {}
    rows = []
    for h in HORSES:
        rows.append(odds_row(race_id, post, snap, fetched, "WIN", str(h), win[h]))
        rows.append(odds_row(race_id, post, snap, fetched, "PLACE", str(h), *place_of(win[h])))
    for a, b in itertools.combinations(HORSES, 2):
        low, high = wide_of(win[a], win[b])
        rows.append(odds_row(race_id, post, snap, fetched, "QUINELLA_PLACE", f"{a},{b}", low, high))
    for pool, total in (("WIN", 1.2e8), ("PLACE", 8.0e7), ("QUINELLA_PLACE", 6.0e7)):
        rows.append(
            odds_row(
                race_id, post, snap, fetched, pool, POOL_TOTAL, None, investment=total * totals
            )
        )
    return [
        r | {"is_final_capture": final, "dividend": paying.get((r["pool"], r["combination"]))}
        for r in rows
    ]


def mirrored(board: dict[int, float]) -> dict[int, float]:
    """The same prices, the shortest runner given the longest price and so on."""
    by_price = sorted(board, key=board.get)
    return dict(zip(by_price, sorted(board.values(), reverse=True), strict=True))


# The 100-vs-110 check's inputs: each race's final Place and Win votes, JRA-VAN H1 as the jravan
# pool_votes table holds them (#368, jravan.votes.POOL_VOTES_SCHEMA), synthetic.

PLACE_TAKEOUT = 0.20  # JRA Win and Place since 2014-06-07 (docs/problem-statement.md, the ledger)
# The floor-bound placed favourites at the final price: race -> (favourite, realised Place
# dividend), by hand from the fixture (the favourite on the final Win odds, placed, paid <= 110).
FLOOR_FAVOURITES = {
    H1: (3, 110),
    H5: (1, 110),
    H6: (8, 100),
    G1: (2, 110),
    G4: (1, 100),
    G6: (10, 110),
}
# race -> its votes profile; every other race (the holdout day's too) is "mid"
VOTES = {H1: "plus10", H5: "withheld", H6: "plus10", G1: "plus10", G4: "withheld", G6: "plus10"}
V_VOTES = {H1: "withheld", G4: "plus10"}  # V: other Place votes, every price and dividend unchanged
PREDICTED = {"plus10": 110, "withheld": 100}
REFUNDED = {H6: 12_000}  # tickets refunded in the Place pool (in the pool row, on no runner)
MID_VOTES = (260_000, 180_000, 140_000, 110_000, 90_000, 70_000, 60_000, 40_000, 30_000, 20_000)


def _shares(race: Race, favourite: int, placed: tuple[int, ...], unplaced) -> dict[int, int]:
    """The favourite's, the other two placed runners' and the rest's votes, by final Win price."""
    order = sorted(race.final, key=race.final.get)
    assert order[0] == favourite in race.finish
    others = [h for h in race.finish if h != favourite]
    rest = [h for h in order if h not in race.finish]
    votes = dict(zip([favourite, *others, *rest], [*placed, *unplaced], strict=True))
    assert sum(votes.values()) == 1_000_000
    return votes


def place_votes(race: Race, profile: str) -> dict[int, int]:
    """Each runner's final Place votes (100-yen tickets), refunds excluded.

    plus10: the favourite holds 70% and the other two placed runners 3% and 2%. The profit left
    after the placed stakes (5% of sales) leaves the favourite 2.4 yen a ticket, so it is paid 100,
    and payout plus its top-ups are 85.1% of sales: 110.
    withheld: 88%, 5% and 4%. The net pool (80%) does not cover the placed stakes (97%), every
    placed runner is paid 100, and the top-ups take the total to 106.7% of sales: 100.
    mid: by final Win price, 26% down to 2%: every placed runner is paid at least 120."""
    if profile == "mid":
        return dict(zip(sorted(race.final, key=race.final.get), MID_VOTES, strict=True))
    favourite = min(race.final, key=race.final.get)
    if profile == "plus10":
        unplaced = (50_000, 40_000, 40_000, 35_000, 30_000, 30_000, 25_000)
        return _shares(race, favourite, (700_000, 30_000, 20_000), unplaced)
    assert profile == "withheld"
    unplaced = (6_000, 5_000, 5_000, 4_000, 4_000, 3_000, 3_000)
    return _shares(race, favourite, (880_000, 50_000, 40_000), unplaced)


def win_votes(race: Race, profile: str) -> dict[int, int]:
    """The WIN pool's votes, the opposite story: read as Place stakes they flip the prediction."""
    order = sorted(race.final, key=race.final.get)
    shares = {
        "plus10": (950_000, 10_000, 8_000, 7_000, 6_000, 5_000, 5_000, 4_000, 3_000, 2_000),
        "withheld": (300_000, 160_000, 120_000, 100_000, 90_000, 70_000, 60_000, 50_000, 30_000,
                     20_000),
    }.get(profile, MID_VOTES)  # fmt: skip
    return dict(zip(order, shares, strict=True))


def reference_floor_dividend(votes: dict[int, int], placed, horse: int) -> int:
    """The Place dividend per 100 yen JRA pays `horse`, from first principles (04 §1.3): sales are
    the pool's tickets (refunds excluded) x 100; the net pool, 80% of sales, first returns the
    placed stakes; the profit left is split equally over the placed horses and paid per ticket,
    rounded down to 10 yen (100 when nothing is left). A horse paid 100 is topped up to 110, 10
    yen a ticket on every placed horse paid 100, unless payout plus top-ups exceed sales."""
    sales = 100 * sum(votes.values())
    profit = sales * (1 - PLACE_TAKEOUT) - sum(100 * votes[h] for h in placed)
    paid = {}
    for h in placed:
        per_ticket = profit / len(placed) / votes[h] if profit > 0 else 0.0
        paid[h] = 100 + int(per_ticket // 10) * 10
    if paid[horse] > 100:
        return paid[horse]
    payout = sum(paid[h] * votes[h] for h in placed)
    top_ups = sum(10 * votes[h] for h in placed if paid[h] == 100)
    return 100 if payout + top_ups > sales else 110


def realised_cell(dividend: float) -> str:
    return "100" if dividend <= 100 else "110" if dividend <= 110 else "above_110"


def expected_cross_tab(profiles: dict[str, str]) -> Counter:
    """(predicted, realised) -> favourites, from the profiles and the realised dividends."""
    return Counter(
        (PREDICTED[profiles[race]], realised_cell(paid))
        for race, (_, paid) in FLOOR_FAVOURITES.items()
    )


def pool_votes_rows(race: Race, profile: str) -> list[dict]:
    """The race's PLACE and WIN rows of pool_votes: one per runner and a `__pool__` row whose
    votes hold the refunded tickets too (votes.py's layout)."""
    keys = race_keys(race.race_id)
    base = {k: keys[k] for k in ("jurisdiction", "race_id", "race_date", "venue", "race_no")}
    rows = []
    for pool, votes in (("PLACE", place_votes(race, profile)), ("WIN", win_votes(race, profile))):
        refund = REFUNDED.get(race.race_id, 0) if pool == "PLACE" else 0
        ranked = {h: i + 1 for i, h in enumerate(sorted(votes, key=lambda h: (-votes[h], h)))}
        rows += [
            base
            | {"pool": pool, "combination": str(h), "votes": n, "refund_votes": None}
            | {"fav_rank": ranked[h], "data_kubun": "4"}
            for h, n in votes.items()
        ]
        rows.append(
            base
            | {"pool": pool, "combination": jravan_votes.POOL_ROW}
            | {"votes": sum(votes.values()) + refund, "refund_votes": refund, "fav_rank": None}
            | {"data_kubun": "4"}
        )
    return rows


@dataclass(frozen=True)
class Scenario:
    final: dict = field(default_factory=dict)  # race -> final WIN odds changed (E)
    at_t: dict = field(default_factory=dict)  # race -> another board at t (C)
    late: bool = False  # B: every row not available at t (and the final capture) changes
    holdout: bool = True  # the synthetic holdout day's rows are in the store (all but D)
    votes: dict = field(default_factory=dict)  # race -> another Place votes profile (V)


def store_schema(table: str) -> pa.Schema:
    return pa.schema([(c, ARROW_TYPES[k]) for c, k in schemas.TABLES[table].columns.items()])


def write_store(data_dir: Path, scenario: Scenario) -> None:
    """The store's races, runners, dividends and meetings, the logged days' odds_timeseries, and
    the jravan pool_votes table (every race's PLACE and WIN votes, the holdout day's included)."""
    folder = data_dir / "processed" / "jra"
    (folder / "odds_timeseries").mkdir(parents=True)
    races = [*HISTORY, *LOGGED, *(HOLDOUT if scenario.holdout else [])]
    races.sort(key=lambda race: race.race_id)  # the store's sort key

    profiles = VOTES | scenario.votes
    votes = sorted(
        (row for r in races for row in pool_votes_rows(r, profiles.get(r.race_id, "mid"))),
        key=lambda row: (row["race_id"], row["pool"], row["combination"]),  # votes.py's order
    )
    jravan = data_dir / "processed" / "jravan"
    jravan.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(votes, schema=jravan_votes.POOL_VOTES_SCHEMA),
        jravan / jravan_votes.FILENAME,
    )

    finals = {r.race_id: {**r.final, **scenario.final.get(r.race_id, {})} for r in races}
    meetings = {race_keys(r.race_id)["meeting_id"]: race_keys(r.race_id) for r in races}
    tables = {
        "meetings": [
            {k: keys[k] for k in ("jurisdiction", "meeting_id", "race_date", "venue")}
            | {"status": "run"}
            for keys in meetings.values()
        ],
        "races": [
            race_keys(r.race_id)
            | {
                "race_class": "C1",
                "distance_m": 1600,
                "surface": "turf",
                "post_time_local": POST_LOCAL[int(r.race_id.split(":")[3])],
                "field_size": len(HORSES),
                "status": "run",
                "weight_type": r.weight_type,
            }
            for r in races
        ],
        "runners": [row for r in races for row in runner_rows(r, finals[r.race_id])],
        "dividends": [row for r in races for row in dividend_rows(r)],
    }
    for name, rows in tables.items():
        pq.write_table(
            pa.Table.from_pylist(rows, schema=store_schema(name)), folder / f"{name}.parquet"
        )

    by_day: dict[str, list[dict]] = {}
    for race in LOGGED:
        post, s = post_of(race.race_id), timedelta(seconds=1)
        at_t = scenario.at_t.get(race.race_id, race.at_t)
        late = mirrored(race.final) if scenario.late else race.final
        paying = {
            (row["pool"], row["combination"]): row["dividend"]
            for row in dividend_rows(race)
            if row["pool"] in ("WIN", "PLACE", "QUINELLA_PLACE")
        }
        shift = 1.7 if scenario.late else 1.0
        rows = [
            # T-20 (the A board at t in every scenario), T-3 (the board at t)
            *board_capture(race.race_id, post - timedelta(minutes=20),
                           post - timedelta(minutes=20) + 3 * s, race.at_t, totals=0.35),
            *board_capture(race.race_id, post - timedelta(minutes=3),
                           post - timedelta(minutes=3) + 2 * s, at_t, totals=0.75),
            # taken before t but fetched after it, then the final capture: not on the board at t
            *board_capture(race.race_id, post - timedelta(minutes=2, seconds=20),
                           post - timedelta(seconds=100), late, totals=0.9 * shift),
            *board_capture(race.race_id, post + timedelta(minutes=1),
                           post + timedelta(minutes=1) + 5 * s, late, final=True,
                           totals=shift, paying=paying),
        ]  # fmt: skip
        by_day.setdefault(race.race_id.split(":")[1], []).extend(rows)
    for day, rows in by_day.items():
        table = pa.Table.from_pylist(rows, schema=ODDS_SCHEMA)
        pq.write_table(table, folder / "odds_timeseries" / f"{day}.parquet")
    if scenario.holdout:  # never opened: reading it would fail
        (folder / "odds_timeseries" / f"{HOLDOUT_DAY}.parquet").write_bytes(b"holdout: unread")


def use_synthetic_holdout(mp: pytest.MonkeyPatch) -> None:
    """HOLDOUT_DAY is the only holdout day (in memory; holdout.yaml is untouched)."""
    start = date.fromisoformat(HOLDOUT_DAY)
    mp.setattr(
        holdout_module, "_load_definition", lambda countries_dir, country: (start, "s", None)
    )
    mp.setattr(holdout_module, "_is_holdout_day", lambda seed, day: day == HOLDOUT_DAY)


HEXES = re.compile(r"\b[0-9a-f]{12,64}\b")  # git and spec hashes, file digests
STAMPS = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})?")


@dataclass
class Outcome:
    entries: list[dict]
    texts: dict[str, str]
    registry_path: Path
    base: Path = Path()
    out_dir: Path = Path()
    writes: list[tuple[Path, str, tuple]] = field(default_factory=list)  # each write_report call

    def masked(self, text: str) -> str:
        """`text` without what differs between two runs of one family: this run's directory,
        hashes and digests, clock times."""
        return STAMPS.sub("<time>", HEXES.sub("<hash>", text.replace(str(self.base), "<base>")))

    def written(self) -> dict[str, tuple[str, tuple]]:
        """Each written file (its path from the run's directory, masked) -> (masked text,
        sources)."""
        return {
            self.masked(str(path)): (self.masked(text), sources)
            for path, text, sources in self.writes
        }

    @property
    def freezes(self) -> list[dict]:
        return [e for e in self.entries if e["kind"] == "freeze"]

    @property
    def records(self) -> list[dict]:
        return [e["record"] for e in self.entries if e["kind"] == "run"]

    @property
    def spec(self) -> dict:
        assert len(self.freezes) == 1
        return self.freezes[0]["spec"]

    @property
    def spec_hash(self) -> str:
        assert len(self.freezes) == 1
        return self.freezes[0]["spec_hash"]

    def by_variant(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for record in self.records:
            assert record["variant"] not in out, record["variant"]
            out[record["variant"]] = record
        return out


def run_scenario(base: Path, scenario: Scenario, owner: dict) -> Outcome:
    """run_study on a fresh data directory whose store appears only once a family is frozen."""
    data_dir, out_dir = base / "data", base / "out"
    data_dir.mkdir(parents=True)
    out_dir.mkdir()
    original, written = Registry.freeze_family, []
    real_write, writes = reports_module.write_report, []

    def freeze(self, family_id, spec):
        spec_hash = original(self, family_id, spec)
        if not written:
            write_store(data_dir, scenario)
            written.append(family_id)
        return spec_hash

    def write_report(path, text, *, sources, **kwargs):
        writes.append((Path(path), text, tuple(sources)))
        return real_write(path, text, sources=sources, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HORSETRACK_DATA_DIR", str(data_dir))
        use_owner_config(mp, owner)
        use_synthetic_holdout(mp)
        mp.setattr(Registry, "freeze_family", freeze)
        for module in (reports_module, avoidance_price, avoidance):
            mp.setattr(module, "write_report", write_report, raising=False)
        avoidance_price.run_study(out_dir)
    assert written, "run_study never froze its family"
    registry_path = data_dir / "registry" / "japan.jsonl"
    entries = [json.loads(line) for line in registry_path.read_text().splitlines()]
    texts = {
        str(p.relative_to(out_dir)): p.read_text(encoding="utf-8")
        for p in sorted(out_dir.rglob("*"))
        if p.is_file() and p.suffix in (".md", ".csv", ".txt", ".json")
    }
    return Outcome(entries, texts, registry_path, base, out_dir, writes)


SCENARIOS = {
    "A": lambda: (Scenario(), owner_config()),
    "B": lambda: (Scenario(late=True), owner_config()),
    "C": lambda: (Scenario(at_t=C_AT_T), owner_config()),
    "D": lambda: (Scenario(holdout=False), owner_config()),
    "E": lambda: (Scenario(final=E_FINAL), owner_config()),
    "F": lambda: (Scenario(), owner_config(l7_entropy_cut_points=list(F_CUTS))),
    "V": lambda: (Scenario(votes=V_VOTES), owner_config()),
}


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    """Each scenario's outcome, run once for the module on first use."""
    done: dict[str, Outcome] = {}

    def get(name: str) -> Outcome:
        if name not in done:
            scenario, owner = SCENARIOS[name]()
            done[name] = run_scenario(tmp_path_factory.mktemp(f"run-{name}"), scenario, owner)
        return done[name]

    return get


def selection(record: dict) -> str:
    return str(record["components"].get("selection", ""))


def is_at_t(record: dict) -> bool:
    return selection(record).startswith(BOARD_AT_T) and "look-ahead" not in selection(record)


def is_look_ahead(record: dict) -> bool:
    return selection(record) == LOOK_AHEAD


def _close(x, y) -> bool:
    if x is None or y is None:
        return x is None and y is None
    return math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12)


def same(a: dict, b: dict) -> bool:
    """The same status and metrics (names, n, values and bounds)."""
    if a["status"] != b["status"]:
        return False

    def ordered(record):
        return sorted(record["metrics"], key=lambda m: (m["fold"], m["metric"], m["n"]))

    ma, mb = ordered(a), ordered(b)
    if [(m["fold"], m["metric"], m["n"]) for m in ma] != [
        (m["fold"], m["metric"], m["n"]) for m in mb
    ]:
        return False
    return all(
        _close(x[k], y[k]) for x, y in zip(ma, mb, strict=True) for k in ("value", "lo", "hi")
    )


def moved(first: Outcome, second: Outcome) -> list[str]:
    """The variants whose results differ between two runs of the same family."""
    a, b = first.by_variant(), second.by_variant()
    assert a.keys() == b.keys()
    return [name for name in a if not same(a[name], b[name])]


def holds_sequence(value, wanted) -> bool:
    """Whether a list equal to `wanted` (as floats) is anywhere in the JSON value."""
    if isinstance(value, list):
        if len(value) == len(wanted) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and _close(float(v), w)
            for v, w in zip(value, wanted, strict=True)
        ):
            return True
        return any(holds_sequence(v, wanted) for v in value)
    if isinstance(value, dict):
        return any(holds_sequence(v, wanted) for v in value.values())
    return False


def holds_number(value, wanted: float) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _close(float(value), wanted)
    if isinstance(value, list):
        return any(holds_number(v, wanted) for v in value)
    if isinstance(value, dict):
        return any(holds_number(v, wanted) for v in value.values())
    return False


def scored(records: dict[str, dict], names, rule: str) -> list[str]:
    return [
        n for n in names if records[n]["hypothesis_id"] == rule and records[n]["status"] == "scored"
    ]


def has_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(has_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(has_key(v, key) for v in value)
    return False


def cross_tab(record: dict) -> dict[tuple[int, str], tuple[float, int]]:
    """The record's 100-vs-110 cells: (predicted, realised) -> (favourites, n)."""
    cells: dict[tuple[int, str], tuple[float, int]] = {}
    for m in record["metrics"]:
        name = m["metric"].lower()
        predicted = re.search(r"predicted_(100|110)(?!\d)", name)
        realised = re.search(r"realised_(above_110|100|110)(?!\d)", name)
        if predicted and realised:
            cell = (int(predicted.group(1)), realised.group(1))
            assert cell not in cells, (record["variant"], name)
            cells[cell] = (m["value"], m["n"])
    return cells


def side_returns(metrics: list[dict], side: str, n: int) -> list[dict]:
    """The metrics naming one side's ('rule', 'comparator') return over n races."""
    return [
        m
        for m in metrics
        if side in m["metric"]
        and "return" in m["metric"]
        and "difference" not in m["metric"]
        and m["n"] == n
    ]


def variants_searched(record: dict) -> list:
    """The family size the record carries, as a metric or a param."""
    names = ("variants_searched", "family_size")
    found = [m["value"] for m in record["metrics"] if any(k in m["metric"].lower() for k in names)]
    found += [v for k, v in record["params"].items() if any(n in k.lower() for n in names)]
    return found


NEGATIONS = ("no ", "not ", "none", "never", "nothing", "owner")
VERDICT = re.compile(r"\badopt|\bpass(es|ed)?\b")  # D18 pass/fail and adoption are gone
VERDICT_NEGATIONS = ("no ", "not ", "none", "never", "nothing")


def test_run_study_refuses_a_missing_owner_decision_before_reading_anything(monkeypatch, tmp_path):
    choices = avoidance_price.OWNER_CHOICES
    assert isinstance(choices, dict)
    for key, options in choices.items():
        assert isinstance(key, str) and key not in NUMERIC_KEYS, key
        assert len(options) >= 1 and len(set(options)) == len(options), key
        assert all(isinstance(option, str) for option in options), key

    full = owner_config()
    data_dir, out_dir = tmp_path / "data", tmp_path / "out"
    data_dir.mkdir()
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(data_dir))
    # No store exists: a study that read a row before checking its inputs would fail otherwise.
    for raw in (None, {}, *({k: v for k, v in full.items() if k != gone} for gone in full)):
        use_owner_config(monkeypatch, raw)
        with pytest.raises(MissingOwnerDecision):
            avoidance_price.run_study(out_dir)
    assert not [p for p in data_dir.rglob("*") if p.is_file()]  # no registry entry, nothing else
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_the_study_needs_no_owner_signoff(study, monkeypatch, tmp_path):
    """Owner reframe 2026-11-27: `owner_signoff` is removed; the freeze line is the record."""
    a = study("A")  # a config without it runs
    assert SIGNOFF not in owner_config()
    assert a.records
    assert not has_key(a.spec, SIGNOFF)
    assert all(entry.get(SIGNOFF) is None for entry in a.records)
    for name, text in a.texts.items():  # no sign-off issue is cited
        assert not re.search(r"(?i)sign-?off[^\n]*#\d", text), name

    data_dir, out_dir = tmp_path / "data", tmp_path / "out"
    data_dir.mkdir()
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(data_dir))
    # the study does not ask for it
    use_owner_config(monkeypatch, {})
    with pytest.raises(MissingOwnerDecision) as missing:
        avoidance_price.run_study(out_dir)
    assert "signoff" not in re.sub(r"[^a-z]", "", str(missing.value).lower())
    # a config that still carries it is refused as an unknown key, before anything is read
    use_owner_config(monkeypatch, owner_config() | {SIGNOFF: 82})
    with pytest.raises(ValueError, match=SIGNOFF):
        avoidance_price.run_study(out_dir)
    assert not [p for p in data_dir.rglob("*") if p.is_file()]
    assert not out_dir.exists() or not any(out_dir.iterdir())
    # and the shipped config, once it exists, does not carry it
    try:
        shipped = SHIPPED_LOAD("japan", CONFIG_NAME)
    except FileNotFoundError:
        shipped = None
    assert shipped is None or SIGNOFF not in shipped


def test_run_study_freezes_one_family_before_reading_and_records_every_variant(study):
    a = study("A")  # the store appeared only when the family was frozen

    assert a.entries[0]["kind"] == "freeze"
    assert len(a.freezes) == 1
    family, spec = a.freezes[0]["family_id"], a.spec
    assert family != avoidance.FAMILY_ID  # a new family, not #81's
    records = a.records
    assert records
    assert all(r["family_id"] == family for r in records)
    assert all(r["ticket"] == 82 for r in records)
    names = [r["variant"] for r in records]
    assert len(set(names)) == len(names)
    assert sorted(names) == sorted(spec["variants"])
    assert spec["family_size"] == len(spec["variants"]) == len(records)
    registry = Registry(a.registry_path, holdout_log=a.registry_path.with_name("no-holdout.jsonl"))
    registry.verify()
    assert registry.family_size(family) == spec["family_size"]

    # each rule's paired comparison is scored, with its clustered interval and n
    assert {"L2", "L3", "L7"} <= {r["hypothesis_id"] for r in records}
    by_variant = a.by_variant()
    for rule in RULES:
        with_ci = [
            n
            for n in scored(by_variant, by_variant, rule)
            if any(m["lo"] is not None and m["hi"] is not None and m["n"] >= 1
                   for m in by_variant[n]["metrics"])
        ]  # fmt: skip
        assert with_ci, rule

    # nothing is adopted: no record says so, and no line of the report claims it
    assert {r["status"] for r in records} <= {"scored", "failed"}
    assert any(name.endswith(".md") for name in a.texts)
    for name, text in a.texts.items():
        for line in text.splitlines():
            if "adopt" in line.lower():
                assert any(word in line.lower() for word in NEGATIONS), (name, line)


def test_final_odds_selection_is_labelled_look_ahead(study):
    a, e = study("A"), study("E")
    records = a.by_variant()
    # the fixture: E moves the floor, the tail, a handicap favourite and a flat market on history
    assert HISTORY[0].final[3] <= 1.1 < E_FINAL[H1][3]
    assert HISTORY[1].final[10] > max(TAIL_ODDS) and E_FINAL[H2][10] < min(TAIL_ODDS)
    assert reference_entropy(HISTORY[2].final) >= CUTS[3]
    assert reference_entropy({**HISTORY[2].final, **E_FINAL[H3]}) < CUTS[3]

    for name, record in records.items():
        if record["hypothesis_id"] in RULES:
            assert is_look_ahead(record) or is_at_t(record), (name, selection(record))
    changed = moved(a, e)
    for name in changed:  # whatever the final odds choose is labelled look-ahead
        assert is_look_ahead(records[name]), (name, selection(records[name]))
    for rule in RULES:  # and each rule has a history run at the final price
        assert scored(records, changed, rule), rule


def test_selection_at_t_reads_only_the_board_at_t(study):
    a = study("A")
    records = a.by_variant()
    # the fixture: C moves the floor, the tail, the favourite and the entropy quintile at t
    boards_t = {race.race_id: race.at_t for race in LOGGED}
    assert boards_t[G1][2] <= 1.1 < C_AT_T[G1][2]
    assert boards_t[G2][9] > max(TAIL_ODDS) and C_AT_T[G2][9] < min(TAIL_ODDS)
    assert reference_entropy(boards_t[G3]) >= CUTS[3] > reference_entropy(C_AT_T[G3])
    assert min(boards_t[G5], key=boards_t[G5].get) != min(C_AT_T[G5], key=C_AT_T[G5].get)

    at_t_names = [name for name, record in records.items() if is_at_t(record)]
    # rows not available at t (fetched after it, the final capture) and the final odds in the
    # runners table move no board-at-t record
    for other in ("B", "E"):
        assert not set(moved(a, study(other))) & set(at_t_names), other
    # the board at t moves only records labelled board at t, and each rule has some
    changed = moved(a, study("C"))
    for name in changed:
        assert is_at_t(records[name]), (name, selection(records[name]))
    for rule in RULES:
        assert scored(records, changed, rule), rule


def test_holdout_days_are_never_read(study):
    # A's store holds the holdout day's rows, its odds file and its pool_votes rows
    a, d = study("A"), study("D")
    assert a.spec_hash == d.spec_hash
    assert moved(a, d) == []
    # nothing written moves with the holdout day's rows, or names the day
    assert a.written() == d.written()
    assert a.writes
    for _, text, _ in a.writes:
        assert HOLDOUT_DAY not in text


def test_entropy_quintiles_use_the_frozen_cut_points(study):
    a, e, f = study("A"), study("E"), study("F")
    # the fixture: a race with a placed favourite is flattest under CUTS and not under F_CUTS
    flat = [
        race
        for race in HISTORY
        if race.weight_type != "handicap"
        and min(race.final, key=race.final.get) in race.finish
        and CUTS[3] <= reference_entropy(race.final) < F_CUTS[3]
    ]
    assert flat

    # the frozen spec carries the owner's cut points and tail odds
    assert holds_sequence(a.spec, CUTS)
    for odds in TAIL_ODDS:
        assert holds_number(a.spec, odds)
    # other scored data (E's final odds), the same frozen family: the data do not move them
    assert e.spec_hash == a.spec_hash
    # other cut points are another spec, and they move L7 alone
    assert holds_sequence(f.spec, F_CUTS) and not holds_sequence(f.spec, CUTS)
    assert f.spec_hash != a.spec_hash
    ra, rf = a.by_variant(), f.by_variant()
    for name, record in ra.items():
        if record["hypothesis_id"] in ("L2", "L3"):
            assert name in rf and same(record, rf[name]), name

    def l7(records):
        return sorted(json.dumps(r["metrics"], sort_keys=True) + r["status"]
                      for r in records.values() if r["hypothesis_id"] == "L7")  # fmt: skip

    assert l7(ra) != l7(rf)


def test_the_100_vs_110_table_reports_predicted_against_realised_dividends(study):
    # the fixture: the floor-bound placed favourites at the final price, from the prices and the
    # Place dividends written
    floor_bound = {}
    for race in (*HISTORY, *LOGGED):
        favourite = min(race.final, key=race.final.get)
        paid = {
            r["combination"]: r["dividend"] for r in dividend_rows(race) if r["pool"] == "PLACE"
        }
        if str(favourite) in paid and paid[str(favourite)] <= 110:
            floor_bound[race.race_id] = (favourite, paid[str(favourite)])
    assert floor_bound == FLOOR_FAVOURITES
    # the reference Place rule gives each profile's prediction; the WIN rows tell another story
    for profiles in (VOTES, VOTES | V_VOTES):
        for race_id, (favourite, _) in FLOOR_FAVOURITES.items():
            race, profile = RACES[race_id], profiles[race_id]
            place = reference_floor_dividend(place_votes(race, profile), race.finish, favourite)
            assert place == PREDICTED[profile], race_id
            win = reference_floor_dividend(win_votes(race, profile), race.finish, favourite)
            assert win != PREDICTED[profile], race_id
        for race in (*HISTORY, *LOGGED):  # no other favourite is near the floor on its votes
            favourite = min(race.final, key=race.final.get)
            if race.race_id not in FLOOR_FAVOURITES and favourite in race.finish:
                votes = place_votes(race, profiles.get(race.race_id, "mid"))
                assert reference_floor_dividend(votes, race.finish, favourite) >= 120
    assert expected_cross_tab(VOTES) == {(110, "110"): 3, (110, "100"): 1, (100, "110"): 1,
                                         (100, "100"): 1}  # fmt: skip
    assert expected_cross_tab(VOTES | V_VOTES) == {(110, "110"): 2, (110, "100"): 2,
                                                   (100, "110"): 2}  # fmt: skip

    # the owner's answer (7) is the first option, so the synthetic config runs it
    assert avoidance_price.OWNER_CHOICES["plus10_inputs"][0] == "jravan_pool_votes"
    for name, profiles in (("A", VOTES), ("V", VOTES | V_VOTES)):
        checks = [
            record
            for record in study(name).records
            if record["hypothesis_id"] == "L2" and is_look_ahead(record) and cross_tab(record)
        ]
        assert len(checks) == 1, name  # the check at the final price
        record = checks[0]
        assert record["status"] == "scored", name
        assert is_withheld(record["sources"]), (name, record["sources"])  # D21: JRA-VAN figures
        cells = cross_tab(record)
        counts = {cell: value for cell, (value, _) in cells.items() if value}
        assert counts == dict(expected_cross_tab(profiles)), name
        assert {n for _, n in cells.values()} == {len(FLOOR_FAVOURITES)}, name


def test_only_jravan_outputs_move_with_the_place_votes(study):
    a, v = study("A"), study("V")  # V: other Place votes, the jra store unchanged
    records, other = a.by_variant(), v.by_variant()
    changed = moved(a, v)
    assert changed
    for name in changed:  # only the 100-vs-110 check reads the votes, and it says so
        assert records[name]["hypothesis_id"] == "L2", name
        assert is_withheld(records[name]["sources"]), (name, records[name]["sources"])
        assert is_withheld(other[name]["sources"]), (name, other[name]["sources"])
    for name, record in records.items():  # the jra store's rules stay public
        if record["hypothesis_id"] in ("L3", "L7"):
            assert not is_withheld(record["sources"]), (name, record["sources"])

    for outcome in (a, v):  # every file the study writes goes through write_report
        through = {path.resolve() for path, _, _ in outcome.writes}
        on_disk = {p.resolve() for p in outcome.out_dir.rglob("*") if p.is_file()}
        assert on_disk and on_disk <= through
    wa, wv = a.written(), v.written()
    assert wa.keys() == wv.keys()
    differ = [key for key in wa if wa[key][0] != wv[key][0]]
    assert differ  # the 100-vs-110 output reads the votes
    for key in differ:  # and it is written as JRA-VAN's, so never inside the repository
        assert is_withheld(wa[key][1]) and is_withheld(wv[key][1]), (key, wa[key][1])


def test_each_scored_variant_is_a_ranked_row_against_its_comparator(study):
    """Owner reframe 2026-11-27: every option is a ranked row with n, CI and the variants searched;
    the rule's tickets and its comparator's both stay rows; no pass/fail and no adoption."""
    a = study("A")
    size = a.spec["family_size"]
    rows = dict.fromkeys(RULES, 0)
    for name, record in a.by_variant().items():
        if record["hypothesis_id"] not in RULES or record["status"] != "scored":
            continue
        found = variants_searched(record)
        assert found and all(value == size for value in found), (name, found, size)
        metrics = record["metrics"]
        if any("realised" in m["metric"] for m in metrics):
            continue  # the 100-vs-110 check, not a paired comparison
        differences = [
            m
            for m in metrics
            if "difference" in m["metric"] and m["lo"] is not None and m["hi"] is not None
        ]
        assert len(differences) == 1, name
        diff = differences[0]
        assert diff["n"] >= 1 and diff["lo"] <= diff["hi"], name
        # each side's own mean return over the same races, whose difference it is
        assert any(
            math.isclose(r["value"] - c["value"], diff["value"], rel_tol=1e-9, abs_tol=1e-9)
            for r in side_returns(metrics, "rule", diff["n"])
            for c in side_returns(metrics, "comparator", diff["n"])
        ), name
        rows[record["hypothesis_id"]] += 1
    assert all(rows.values()), rows

    for name, text in a.texts.items():
        for line in text.splitlines():
            if VERDICT.search(line.lower()):
                assert any(word in line.lower() for word in VERDICT_NEGATIONS), (name, line)
