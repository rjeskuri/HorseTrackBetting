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
  label-only table read for the settled race, never at t; development races only). S, the
  pool's tickets, is the `__pool__` row's votes less its `refund_votes`; sales are S x 100 yen.
  JRA's Place rule (払戻金の計算方法, the stake-back share; `core.pricing.
  stake_back_share_dividend`): with P placed horses, W_h the tickets on placed horse h and D = S -
  the placed horses' W the tickets on starters that did not place, horse h's payout pool is (W_h +
  D/P) x 0.80 (R0, the Place rate without any premium), shared over its own tickets, so its base
  dividend per 100 yen is 0.80 x (W_h + D/P) / W_h x 100 rounded down to 10 yen, never under 100.
  `place_payout(votes, placed, sales, payout_rate)` (votes: tickets by horse, unplaced horses
  allowed; sales in yen) returns (each placed horse's base dividend, 100 at the floor; the base
  total payout, the sum of base x W; the top-ups, 10 yen a ticket on each placed horse at base
  100), D read from `sales`. Plus-10 then pays a horse at base 100
  `predicted_floor_dividend(total, top_ups, sales)`, on the base total (premium excluded). On a
  premium race (`takeout.race_takeout`'s boost `premium` or `ultra_premium` on the PLACE pool),
  each placed horse then gains 5% of sales split equally over the P horses and paid per ticket,
  rounded down to 10 yen by itself: 10 x floor(S / (2 P W_h)) yen. The record's metrics hold one
  count per cell of predicted (100, 110) x realised (100, 110, above 110), named
  `predicted_<p>_realised_<r>` (r `100`, `110` or `above_110`, either order), with n the
  favourites predicted; once one cell is present, a cell left out counts 0. The fixture's pools
  sit far from every boundary (payout plus top-ups at 94.2%, 91.6% or 105.4% of sales), so any
  faithful reading gives the same predictions, and each race's WIN pool tells the opposite story,
  so a prediction read from the WIN rows flips.
- D21: the votes are JRA-VAN's (`core.eval.reports.WITHHELD_SOURCES`). The 100-vs-110 record, and
  every record whose figures move with the votes, declares a withheld source in `sources`; the L3
  and L7 records read the jra store alone and do not. Every file the study writes goes through
  `core.eval.reports.write_report`, and every written text that moves with the votes is written
  with a withheld source, so no public (jra-only) output carries a JRA-VAN figure. The holdout
  day's votes rows move nothing and no written text names a holdout day.

Review round 1 of b9a6b6d (each pinned by its own test below):

- The board at t's post time: a race's capture's own `post_utc`; when its capture carries none,
  the scheduled post time of `board.snapshot.post_times_for(day, venue)` (the `<VENUE>.races.json`
  sidecar beside the odds log, else HKTrack's stored card), so t is that time - 2 min (D2). The
  capture's post_utc wins over a schedule that disagrees. With neither, the race is data pending.
  The frozen spec or the board-at-t records name that scheduled source (post_times_for, the
  sidecar or the card).
- Selection at t reads the board at t alone: a runner priced on it is a runner, whatever the final
  results say later. A runner scratched after t is still selected (the favourite, the top-3 box,
  every set), and every ticket holding it is settled as JRA settles it: refunded, stake back. The
  store carries no refund row for a scratch (its `is_refund` is always false), so the refund comes
  from `runners.is_scratched`. Look-ahead variants may read the final starters. Each record's
  `params` carry its `pool` and, for a cut variant, its `cut`.
- L2's no-bet race: when every ticket the comparator bets in a race is at the floor, the race stays
  paired and the rule's side returns its stake (`stake_back`, named in the spec's L2 rules); only
  a race without board data (the comparator bets nothing) drops.
- The 100-vs-110 check at t counts a race only once its favourite at t has a PLACE price at t.
  With no such race the at-t check is data pending (a 'failed' record, reported as data pending),
  never scored with n = 0; a covered race whose favourite is off the floor is a scored zero.
- The reconstruction check: before its cross-tab, the 100-vs-110 check reconstructs the Place
  dividend of every placed horse of its races (the floor favourite and the non-floor ones) from
  the votes and compares it with the realised one. Its figures are metrics of the 100-vs-110
  record (named with `validation`, n the placed horses checked), withheld with it. The tolerance
  is in the frozen spec (a key holding `tolerance`): 10 yen per 100 staked, one rounding step (JRA
  rounds Place dividends down to 10 yen, and a floor horse's 100 or 110 is the check's own
  question). A miss above it is a diagnostic on its race's row (fix round 4, B below), never a
  rejection of the record. A check with no race to check has no miss.
- Missing or unreadable votes: the realised counts from the jra store (metrics `realised_<r>`)
  survive in a 'scored' 100-vs-110 record and only the prediction is unavailable (no predicted
  cell counts a favourite). The public report still gives the realised counts and never says
  that no floor-bound favourite placed.

Review round 2 of 360216f (each pinned by its own test below):

- The Place rule above, JRA's stake-back share, replaces the equal split of the profit: hand-worked
  cases of `place_payout` (P = 2, 3 and 4, Plus-10 paid and withheld) and the fixture's realised
  dividends from the same rule. (A premium race is no longer rebuilt: fix round 4, A below.)
- Eligible favourites whose prediction cannot be made (the votes missing or unreadable): the
  prediction is unavailable, shown by absent cells: the record holds no `predicted_*` cell and no
  `validation` metric, and no written row gives the check a verdict (`within`). A covered
  selection with no eligible favourite keeps its scored zero: its cells present, all 0.
- A scratch at t: a runner whose latest WIN or PLACE row at t is SCR has no price in any pool at
  t, Wide pairs included, even when an older row of the other pool still prices it (the
  board-at-t rule of docs/specs/board-at-t.md and `board.snapshot`).
- The post-time input of the at-t records is one data build `<identity>@<digest>` (a hex digest
  of the race -> post-time mapping): the identity the same whatever the coverage, the digest
  moving when one race's post time changes or is backfilled, so the registry counts one family
  size over such runs. The coverage counts stay in the report.

Fix round 4 of 1774fb3 (the owner-requested restart plan, GPT-6 Astra on #82, 2026-11-27: keep
every observation and every attempted prediction, each with its availability and diagnostic
status; each pinned by its own test below):

- A. A race whose Place premium settlement is unresolved (the boost calendar gives its PLACE pool
  a premium, which JRA may withhold: `takeout`'s `boost_may_be_withheld`) has its prediction
  unavailable, its CSV row's `why` naming the premium. The premium is never inferred from the
  realised dividends, so a race where it was paid (Q) and one where it was withheld (Q0) give the
  same prediction rows. Its realised dividend and every other race stay.
- B. No whole-table rejection: the reconstruction check is a diagnostic. The 100-vs-110 row of a
  race any of whose placed horses misses its rebuilt dividend by more than the tolerance is
  labelled discrepant (a cell matching 'discrepan') and keeps its prediction; the cross-tab is
  scored over every available prediction and the record keeps its metrics (status 'scored', no
  error). Counts come from the final row statuses: each 100-vs-110 record carries one metric each
  named with `attempted` (the floor-bound placed favourites), `available` (those with a
  prediction), `unavailable` (those without) and `discrepant` (the available ones labelled
  discrepant); the private report has a table with those four columns and one row per check; the
  private CSV's rows give the same counts (available: a `predicted` value; discrepant: a cell
  matching 'discrepan'); the public report gives the attempted count (its realised table's
  `placed`) and, the others being built on the votes (D21), none of the other three.
- C. The read boundary (D14): `read_store_table` materialises no non-date column of a holdout day
  (it may read the date column first), and `read_place_votes` decodes no votes column of a
  holdout row, even in a row group mixing development and holdout races: no Python frame of the
  package or of pyarrow's Python layer holds such a value while the reader runs (a trace of the
  call), and a row group holding only a holdout day, whose other columns cannot be decoded, is
  never decoded. The store reader's own test is in tests/countries/japan/models/test_v0_store.py.
- D. A votes file that exists but cannot be opened (PermissionError) does not abort the run: every
  record is written, the comparisons are scored as in A, the realised counts stand, the
  prediction is unavailable and the votes' data build of the check's records names the failure.
- E. Walk-forward rows. Each whole-development record keeps #81's scheme
  'paired_same_races:development' and its name, as the descriptive output. Beside it, each paired
  comparison has one walk-forward record (`fold_scheme` starting 'walk_forward', another variant
  name, the same hypothesis and at least the same components and params) scoring only the races
  in the test windows of `core.eval.folds.walk_forward` over the development race days under
  Japan's scheme (`load_country_scheme`, config/folds.yaml; here a day scheme the tests control,
  as the synthetic days span a week and Japan's quarters need 22 of them to train). Its pooled
  metrics (fold labels naming no single window day) hold the paired difference with its bounds
  and each side's mean return over those races; each fold's races are labelled with their fold:
  one metric named with 'difference' whose fold label names the window's first day, n the fold's
  paired races. The scheme is frozen in the spec; it moves no descriptive record.
- F. run_study's returned summary and main's stdout tell no withheld check's status or result:
  they are the same, masked of paths, hashes and times, whatever the votes give (A, V, M, Q0,
  and R, a votes reader that raises).
- G. The family is `japan-family-l-floor-tail-favourite-v5`, superseding v4, frozen before any race
  is read.

Fix round 5 of e487d61 (each pinned by its own test below):

- 1. A processed jravan directory without search permission (so even asking whether the votes
  file exists is refused: `stat` raises PermissionError) does not abort the run, as D above: every
  record is written, nothing the votes do not build moves, the realised counts stand, the
  prediction is unavailable and the votes' data build names the failure.
- 2. The votes' data build keeps one identity whatever the file's availability: its value is
  `<identity>@<refresh>`, the identity (the part before the last '@', as the registry's
  `family_size` reads it) the same when the file is readable, unreadable or missing, and only the
  refresh differing (a hex digest of the file when it is read; otherwise naming the failure). The
  same frozen family run again against one registry once the file is readable appends the
  refreshed checks as further runs of the same variants: the registry's family size stays the
  frozen spec's.

How the run-level tests prove it without fixing the owner's betting rules: the same synthetic
store is run under one change at a time and the registry records are compared. Rows not available
at t, and the final odds, must not move any board-at-t record; the board at t must not move any
record that is not labelled board at t; the final odds move only look-ahead records; the holdout
day's rows move nothing; the cut points move only L7; the Place votes move only JRA-VAN records
and files. The store appears only when the family is frozen, so a read before the freeze fails.

Expected values come from first principles: the ranks, floors and tails from the fixture prices
written below, the quintiles from the convention above, entropies from a reference formula, the
Plus-10 predictions from a reference Place payout. Every race's realised Place dividends follow
that reference payout of its own votes, save the floor favourites' 100 or 110 set by hand, so a
faithful reconstruction passes its check. Every frame is synthetic; no date is a Japan
holdout day (on or after 2026-11-28). The holdout day of the run-level tests is a synthetic one,
declared in memory by patching the holdout definition.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import contextlib
import copy
import dataclasses
import io
import itertools
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core import config as core_config
from horsetrack.core.eval import folds as folds_module
from horsetrack.core.eval import holdout as holdout_module
from horsetrack.core.eval import reports as reports_module
from horsetrack.core.eval.folds import FoldScheme, walk_forward
from horsetrack.core.eval.registry import Registry
from horsetrack.core.eval.reports import is_withheld
from horsetrack.core.store import schemas
from horsetrack.core.store.asof import load_availability
from horsetrack.core.store.tables import ARROW_TYPES
from horsetrack.countries.japan import odds_log, takeout
from horsetrack.countries.japan.board.snapshot import SCRATCHED
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
    place_payout,
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
# JRA's Place rule (払戻金の計算方法): the stake-back share, then Plus-10, then any premium
# ---------------------------------------------------------------------------------------------

R0 = Fraction(4, 5)  # the Place payout rate without any premium (takeout 20% since 2014-06-07)


def reference_place_payout(
    votes: dict[int, int], placed, tickets: int
) -> tuple[dict[int, int], int, int]:
    """JRA's base Place payout, from first principles in exact arithmetic: with P placed horses
    and D = tickets - their tickets, horse h's pool (W_h + D/P) x R0 is shared over its own W_h
    tickets, rounded down to 10 yen per 100, never under 100. (Base dividends, base total payout
    in yen, top-ups: 10 yen a ticket on each placed horse at base 100.)"""
    w = {h: votes[h] for h in placed}
    p, lost = len(w), tickets - sum(w.values())
    base = {}
    for h, n in w.items():
        raw = R0 * (n + Fraction(lost, p)) / n * 100
        base[h] = max(100, 10 * math.floor(raw / 10))
    total = sum(base[h] * n for h, n in w.items())
    top_ups = sum(10 * n for h, n in w.items() if base[h] == 100)
    return base, total, top_ups


# (every horse's tickets, the placed horses, base dividends, base total, top-ups, paid after
# Plus-10), each worked by hand: S = the tickets, D = S - the placed tickets, the base
# 0.8 x (W + D/P) / W x 100 rounded down to 10 and at least 100, Plus-10 on the base total
PLACE_CASES = [
    # the review's case: S 1,000, D 350; 98.7 -> 100, 173.3 -> 170, 266.7 -> 260; 80,000 plus
    # 5,000 top-ups is within 100,000 sales: 110. (The equal split gives 110 / 150 / 200.)
    ({1: 500, 2: 100, 3: 50, 4: 200, 5: 150}, (1, 2, 3),
     {1: 100, 2: 170, 3: 260}, 80_000, 5_000, {1: 110, 2: 170, 3: 260}),
    # no floor horse: D 550; 153.3 -> 150, 177.8 -> 170, 226.7 -> 220
    ({1: 200, 2: 150, 3: 100, 4: 300, 5: 250}, (1, 2, 3),
     {1: 150, 2: 170, 3: 220}, 77_500, 0, {1: 150, 2: 170, 3: 220}),
    # Plus-10 withheld: D 50; 81.6 -> 100, 102.2 -> 100, 113.3 -> 110 (a base 110, not Plus-10);
    # 95,400 plus 9,100 exceeds 100,000: 100
    ({4: 850, 7: 60, 9: 40, 1: 30, 2: 20}, (4, 7, 9),
     {4: 100, 7: 100, 9: 110}, 95_400, 9_100, {4: 100, 7: 100, 9: 110}),
    # P = 2, Plus-10 paid: D 210; 92.0 -> 100, 173.3 -> 170; 85,300 plus 7,000: 110
    ({3: 700, 5: 90, 1: 120, 2: 90}, (3, 5),
     {3: 100, 5: 170}, 85_300, 7_000, {3: 110, 5: 170}),
    # P = 2, withheld: D 55; 82.4 -> 100, 128.9 -> 120; 95,400 plus 9,000 exceeds 100,000: 100
    ({6: 900, 2: 45, 1: 30, 3: 25}, (6, 2),
     {6: 100, 2: 120}, 95_400, 9_000, {6: 100, 2: 120}),
    # P = 4 (a dead heat): D 245; 92.3 -> 100, 104.5 -> 100, 124.5 -> 120, 188.9 -> 180;
    # 81,300 plus 6,000: 110 for both horses at the floor
    ({1: 400, 2: 200, 3: 110, 4: 45, 5: 145, 6: 100}, (1, 2, 3, 4),
     {1: 100, 2: 100, 3: 120, 4: 180}, 81_300, 6_000, {1: 110, 2: 110, 3: 120, 4: 180}),
    # the fixture's plus10 pool: S 1,000,000, D 250,000; 89.5 -> 100, 302.2 -> 300, 413.3 -> 410;
    # 87.2m plus 7m top-ups within 100m: 110
    ({1: 700_000, 2: 30_000, 3: 20_000, 4: 250_000}, (1, 2, 3),
     {1: 100, 2: 300, 3: 410}, 87_200_000, 7_000_000, {1: 110, 2: 300, 3: 410}),
]  # fmt: skip


def test_place_payout_shares_the_losing_stakes_by_stake_back():
    """Review 2 (high): JRA's Place rule is the stake-back share, (W_h + D/P) x 0.80 over horse
    h's own tickets, not the placed stakes back and the profit split equally; Plus-10 on the base
    total. D comes from the sales, so the votes may hold the placed horses only."""
    for votes, placed, base, total, top_ups, paid in PLACE_CASES:
        tickets = sum(votes.values())
        sales = 100.0 * tickets
        # the reference rule agrees with the hand-worked figures
        assert reference_place_payout(votes, placed, tickets) == (base, total, top_ups), placed
        floor = 100 if total + top_ups > sales else 110
        assert {h: d if d > 100 else floor for h, d in base.items()} == paid, placed

        for given in (votes, {h: votes[h] for h in placed}):
            got, got_total, got_top_ups = place_payout(given, list(placed), sales, 0.80)
            assert {int(h): d for h, d in got.items()} == base, (placed, got)
            assert math.isclose(got_total, total, abs_tol=1e-6), (placed, got_total)
            assert math.isclose(got_top_ups, top_ups, abs_tol=1e-6), (placed, got_top_ups)
            dividend = predicted_floor_dividend(got_total, got_top_ups, sales)
            final = {int(h): d if d > 100 else dividend for h, d in got.items()}
            assert final == paid, (placed, final)


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


# Fix round 4, E: the walk-forward scheme the tests control (config/folds.yaml's `folds:` section,
# in memory only). Japan's own (quarters, min_train 22) has no fold on a week of synthetic days.
# Over the development days D1..D4 below, windows of one day from D1 + 1: D1 trains only, and
# D2, D3 and D4 are each a fold (worked by hand, checked against core.eval.folds.walk_forward).
TEST_FOLDS = {
    "unit": "day",
    "test_length": 1,
    "step": 1,
    "min_train": 1,
    "window": "expanding",
    "embargo_days": 0,
    "anchor_month": 1,
}
LATE_FOLDS = TEST_FOLDS | {"min_train": 8}  # K: the first window is D1 + 8 days, D4 alone


def use_owner_config(mp: pytest.MonkeyPatch, raw: dict | None, folds: dict | None = None) -> None:
    """config/avoidance_price.yaml reads as `raw` (None: the file does not exist) and
    config/folds.yaml's scheme as `folds` (TEST_FOLDS by default); every other config as
    shipped."""
    scheme = dict(TEST_FOLDS if folds is None else folds)

    def load(country, name, *args, **kwargs):
        if name == "folds":
            return {"folds": dict(scheme)}
        if name != CONFIG_NAME:
            return SHIPPED_LOAD(country, name, *args, **kwargs)
        if raw is None:
            raise FileNotFoundError(f"countries/{country}/config/{name}.yaml")
        return copy.deepcopy(raw)

    mp.setattr(core_config, "load_country_yaml", load)
    for module in (avoidance_price, avoidance, folds_module):
        mp.setattr(module, "load_country_yaml", load, raising=False)


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
    # a floor favourite wins, and its Place pays 110 though its votes predict 100
    Race(rid(D2, "TOK", 1), "handicap", (1, 6, 10), priced("conc2", 1), place={1: 110}),
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
# E: each test window's first day -> its races, by hand from the schemes (D1 trains only)
FOLD_RACES = {D2: (H5, H6, H7, H8), D3: (G1, G2, G3, G4), D4: (G5, G6, G7, G8)}  # TEST_FOLDS
LATE_FOLD_RACES = {D4: (G5, G6, G7, G8)}  # LATE_FOLDS (K)
FOLD_DAYS = (D2, D3, D4)

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


def dividend_rows(
    race: Race, place: dict[int, int] | None = None, *, premium: bool = False
) -> list[dict]:
    """The race's paying rows in every pool, from its own (base) final odds; the PLACE rows from
    the reference Place payout of its (base) votes (with the 5% premium on a premium race, Q),
    then `race.place`, then `place` (M)."""
    o, (first, second, third) = race.final, race.finish
    top3 = race.finish
    votes = place_votes(race, VOTES.get(race.race_id, "mid"))
    paid = reference_place_dividends(votes, top3, premium=premium) | race.place | (place or {})
    rows = [("WIN", str(first), max(100, r10(100 * o[first])))]
    rows += [("PLACE", str(h), paid[h]) for h in top3]
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


def runner_rows(race: Race, final: dict[int, float], scratched: int | None = None) -> list[dict]:
    """The race's runners; `scratched` (S) is out of the race: no placing, no final price."""
    assert scratched not in race.finish
    order = [*race.finish, *(h for h in HORSES if h not in race.finish and h != scratched)]
    position = {h: i + 1 for i, h in enumerate(order)}
    running = sorted((h for h in final if h != scratched), key=final.get)
    favourite = {h: i + 1 for i, h in enumerate(running)}
    return [
        race_keys(race.race_id)
        | {
            "horse_id": f"{race.race_id.replace(':', '-')}-{h:02d}",
            "horse_no": h,
            "dead_heat": False,
            "bracket": BRACKETS[h],
        }
        | (
            {"placing_raw": "除外", "finish_pos": None, "win_odds": None, "is_scratched": True}
            | {"win_fav_rank": None}
            if h == scratched
            else {"placing_raw": str(position[h]), "finish_pos": position[h]}
            | {"win_odds": final[h], "is_scratched": False, "win_fav_rank": favourite[h]}
        )
        for h in HORSES
    ]


def post_of(race_id: str) -> datetime:
    _, day, _, race_no = race_id.split(":")
    hh, mm = (int(x) for x in POST_LOCAL[int(race_no)].split(":"))
    return datetime.fromisoformat(day).replace(hour=hh, minute=mm, tzinfo=JST).astimezone(UTC)


def board_capture(
    race_id,
    snap,
    fetched,
    win,
    *,
    final=False,
    totals=1.0,
    paying=None,
    timed=True,
    place=True,
    scratched=frozenset(),
    stale=frozenset(),
) -> list:
    """One capture of the race's WIN, PLACE and Wide boards. `timed` False: the rows carry no
    post_utc (N, N0). `place` False: no PLACE row (W, P); a set of horses: only those priced on
    the PLACE board, the others' rows unpriced (P). A `scratched` horse's rows are unpriced. A
    `stale` horse's WIN row is SCR while its PLACE and Wide rows are left out of the capture, so
    an older capture's priced rows of those pools stay its latest (X)."""
    post, paying = post_of(race_id), paying or {}
    priced_place = set(HORSES) if place is True else set(place or ())
    rows = []
    for h in HORSES:
        rows.append(odds_row(race_id, post, snap, fetched, "WIN", str(h), win[h]))
        low, high = place_of(win[h]) if h in priced_place else (None, None)
        rows.append(odds_row(race_id, post, snap, fetched, "PLACE", str(h), low, high))
    for a, b in itertools.combinations(HORSES, 2):
        low, high = wide_of(win[a], win[b])
        rows.append(odds_row(race_id, post, snap, fetched, "QUINELLA_PLACE", f"{a},{b}", low, high))
    for pool, total in (("WIN", 1.2e8), ("PLACE", 8.0e7), ("QUINELLA_PLACE", 6.0e7)):
        rows.append(
            odds_row(
                race_id, post, snap, fetched, pool, POOL_TOTAL, None, investment=total * totals
            )
        )
    if place is False:
        rows = [r for r in rows if r["pool"] != "PLACE"]

    def horses_of(row) -> set[int]:
        combination = row["combination"]
        return set() if combination == POOL_TOTAL else set(map(int, combination.split(",")))

    rows = [r for r in rows if r["pool"] == "WIN" or not horses_of(r) & set(stale)]
    for r in rows:
        horses = horses_of(r)
        if horses & set(scratched) or (r["pool"] == "WIN" and horses & set(stale)):
            r |= {"odds": None, "odds_hi": None, "status_raw": SCRATCHED}
        if not timed:
            r |= {"post_utc": None, "minutes_to_post": None}
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
VOTES = {H1: "floor110", H5: "withheld", H6: "plus10", G1: "plus10", G4: "withheld", G6: "plus10"}
# V: other Place votes, every price and dividend unchanged. Each flip is between floor110 and
# withheld, whose other two placed horses hold the same share of the losing stakes and so are paid
# the same: every placed horse of the race stays within one 10-yen step of its realised dividend
# (the favourite's 110 against 100) and the reconstruction check still passes.
V_VOTES = {H1: "withheld", G4: "floor110"}
PREDICTED = {"plus10": 110, "floor110": 110, "withheld": 100}
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
    """Each runner's final Place votes (100-yen tickets), refunds excluded: S = 1,000,000. By
    JRA's rule (`reference_place_dividends`), with D the tickets on the unplaced runners, a placed
    horse's base is 0.8 x (W + D/3) / W x 100 rounded down to 10 yen, at least 100.

    plus10: the favourite 700,000, the other two placed runners 30,000 and 20,000 (D 250,000):
    89.5 -> 100, 302.2 -> 300, 413.3 -> 410; payout 87.2m plus 7m top-ups is 94.2% of sales: 110.
    floor110: 600,000, 120,000 and 80,000 (D 200,000): 88.9 -> 100, 124.4 -> 120, 146.7 -> 140;
    payout 85.6m plus 6m top-ups is 91.6% of sales: 110.
    withheld: 900,000, 30,000 and 20,000 (D 50,000): 81.5 -> 100, 124.4 -> 120, 146.7 -> 140;
    payout 96.4m plus 9m top-ups is 105.4% of sales: 100. The other two hold the same W / D as
    under floor110, so they are paid the same.
    mid: by final Win price, 26% down to 2%: every placed runner is paid at least 120 (the
    favourite's 260,000 with the next two, 180,000 and 140,000, leaves D 420,000: 123.1)."""
    if profile == "mid":
        return dict(zip(sorted(race.final, key=race.final.get), MID_VOTES, strict=True))
    favourite = min(race.final, key=race.final.get)
    if profile == "plus10":
        unplaced = (50_000, 40_000, 40_000, 35_000, 30_000, 30_000, 25_000)
        return _shares(race, favourite, (700_000, 30_000, 20_000), unplaced)
    if profile == "floor110":
        unplaced = (40_000, 35_000, 30_000, 30_000, 25_000, 20_000, 20_000)
        return _shares(race, favourite, (600_000, 120_000, 80_000), unplaced)
    assert profile == "withheld"
    unplaced = (10_000, 9_000, 8_000, 7_000, 6_000, 5_000, 5_000)
    return _shares(race, favourite, (900_000, 30_000, 20_000), unplaced)


def win_votes(race: Race, profile: str) -> dict[int, int]:
    """The WIN pool's votes, the opposite story: read as Place stakes they flip the prediction."""
    order = sorted(race.final, key=race.final.get)
    one_sided = (950_000, 10_000, 8_000, 7_000, 6_000, 5_000, 5_000, 4_000, 3_000, 2_000)
    shares = {
        "plus10": one_sided,
        "floor110": one_sided,
        "withheld": (300_000, 160_000, 120_000, 100_000, 90_000, 70_000, 60_000, 50_000, 30_000,
                     20_000),
    }.get(profile, MID_VOTES)  # fmt: skip
    return dict(zip(order, shares, strict=True))


def reference_place_dividends(votes: dict[int, int], placed, *, premium=False) -> dict[int, int]:
    """The Place dividend per 100 yen JRA pays each placed horse, from first principles (JRA's
    払戻金の計算方法): S, the pool's tickets (refunds excluded), x 100 yen are the sales; the base
    is `reference_place_payout`'s stake-back share at 0.80; a horse at base 100 is topped up to
    110 unless the base payout plus the top-ups (10 yen a ticket on each such horse) exceed sales;
    on a premium race each placed horse then gains 5% of sales over the P horses, per ticket,
    rounded down to 10 yen by itself (10 x floor(S / (2 P W)))."""
    assert R0 == 1 - Fraction(str(PLACE_TAKEOUT))
    tickets = sum(votes.values())
    base, payout, top_ups = reference_place_payout(votes, placed, tickets)
    floor = 100 if payout + top_ups > 100 * tickets else 110
    paid = {h: d if d > 100 else floor for h, d in base.items()}
    if premium:
        p = len(paid)
        paid = {h: d + 10 * (tickets // (2 * p * votes[h])) for h, d in paid.items()}
    return paid


def reference_floor_dividend(votes: dict[int, int], placed, horse: int) -> int:
    """The Place dividend per 100 yen JRA pays `horse` (`reference_place_dividends`)."""
    return reference_place_dividends(votes, placed)[horse]


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
    # review round 1 of b9a6b6d
    untimed: frozenset = frozenset()  # races whose captures carry no post_utc (N, N0)
    # (day, venue) -> ("sidecar" or "card", minutes added to the true post): post_times_for (N)
    schedule: dict = field(default_factory=dict)
    scratched: dict = field(default_factory=dict)  # race -> horse scratched after t (S)
    place: dict = field(default_factory=dict)  # race -> {horse: realised Place dividend} (M)
    no_place_at_t: frozenset = frozenset()  # races with no PLACE row at t (W, W1, P)
    place_at_t: dict = field(default_factory=dict)  # race -> the only horses priced at t (P)
    # "garbage" (U), "missing" (U0), or "unopenable": written, then mode 000 (PU, fix round 4 D)
    votes_file: str = "parquet"
    # review round 2 of 360216f
    premium: frozenset = frozenset()  # races whose PLACE pool pays JRA's 5% premium (Q)
    # fix round 4: on the boost calendar with a premium, which the race did not pay (Q0, A)
    premium_withheld: frozenset = frozenset()
    reader_fault: bool = False  # the pool_votes reader raises a RuntimeError (R, F)
    folds: dict | None = None  # config/folds.yaml's scheme (None: TEST_FOLDS; K: LATE_FOLDS)
    # race -> a horse scratched by t: SCR on its WIN row at T-3, whose PLACE and Wide rows are not
    # refreshed after T-20; the results mark it scratched (X)
    stale_scratch: dict = field(default_factory=dict)
    # race -> minutes added to its scheduled post time, None: left out of the schedule (N1, N2)
    post_shift: dict = field(default_factory=dict)
    # fix round 5: "untraversable": the votes file written readable, then the processed jravan
    # directory left without search permission, so its stat is refused (PD)
    votes_dir: str = "open"


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
    if scenario.votes_file in ("parquet", "unopenable"):
        pq.write_table(
            pa.Table.from_pylist(votes, schema=jravan_votes.POOL_VOTES_SCHEMA),
            jravan / jravan_votes.FILENAME,
        )
        if scenario.votes_file == "unopenable":  # it exists, and opening it is refused
            (jravan / jravan_votes.FILENAME).chmod(0)
    elif scenario.votes_file == "garbage":
        (jravan / jravan_votes.FILENAME).write_bytes(b"PAR1 a truncated download, not parquet")
    else:
        assert scenario.votes_file == "missing"
    if scenario.votes_dir == "untraversable":  # restored by run_scenario however the run ends
        jravan.chmod(0o600)

    finals = {r.race_id: {**r.final, **scenario.final.get(r.race_id, {})} for r in races}
    meetings = {race_keys(r.race_id)["meeting_id"]: race_keys(r.race_id) for r in races}
    out = scenario.scratched | scenario.stale_scratch  # race -> the horse the results scratch

    def paying_rows(r: Race) -> list[dict]:
        return dividend_rows(
            r, scenario.place.get(r.race_id), premium=r.race_id in scenario.premium
        )

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
        "runners": [
            row for r in races for row in runner_rows(r, finals[r.race_id], out.get(r.race_id))
        ],
        "dividends": [row for r in races for row in paying_rows(r)],
    }
    for name, rows in tables.items():
        pq.write_table(
            pa.Table.from_pylist(rows, schema=store_schema(name)), folder / f"{name}.parquet"
        )

    by_day: dict[str, list[dict]] = {}
    for race in LOGGED:
        post, s, race_id = post_of(race.race_id), timedelta(seconds=1), race.race_id
        at_t = scenario.at_t.get(race_id, race.at_t)
        late = mirrored(race.final) if scenario.late else race.final
        paying = {
            (row["pool"], row["combination"]): row["dividend"]
            for row in paying_rows(race)
            if row["pool"] in ("WIN", "PLACE", "QUINELLA_PLACE")
        }
        shift = 1.7 if scenario.late else 1.0
        timed = race_id not in scenario.untimed
        place = race_id not in scenario.no_place_at_t and scenario.place_at_t.get(race_id, True)
        before = {"timed": timed, "place": place}  # the captures on the board at t
        after = {"timed": timed, "scratched": {out.get(race_id)} - {None}}
        stale = {scenario.stale_scratch.get(race_id)} - {None}
        rows = [
            # T-20 (the A board at t in every scenario), T-3 (the board at t)
            *board_capture(race_id, post - timedelta(minutes=20),
                           post - timedelta(minutes=20) + 3 * s, race.at_t, totals=0.35, **before),
            *board_capture(race_id, post - timedelta(minutes=3),
                           post - timedelta(minutes=3) + 2 * s, at_t, totals=0.75, stale=stale,
                           **before),
            # taken before t but fetched after it, then the final capture: not on the board at t
            *board_capture(race_id, post - timedelta(minutes=2, seconds=20),
                           post - timedelta(seconds=100), late, totals=0.9 * shift, **after),
            *board_capture(race_id, post + timedelta(minutes=1),
                           post + timedelta(minutes=1) + 5 * s, late, final=True,
                           totals=shift, paying=paying, **after),
        ]  # fmt: skip
        by_day.setdefault(race_id.split(":")[1], []).extend(rows)
    for day, rows in by_day.items():
        table = pa.Table.from_pylist(rows, schema=ODDS_SCHEMA)
        pq.write_table(table, folder / "odds_timeseries" / f"{day}.parquet")
    write_schedules(data_dir, scenario.schedule, scenario.post_shift)
    if scenario.holdout:  # never opened: reading it would fail
        (folder / "odds_timeseries" / f"{HOLDOUT_DAY}.parquet").write_bytes(b"holdout: unread")


def write_schedules(data_dir: Path, schedule: dict, post_shift: dict | None = None) -> None:
    """The scheduled post times `board.snapshot.post_times_for` reads (N): a `<VENUE>.races.json`
    sidecar beside the day's odds logs (the logger's own writer), or HKTrack's stored card
    `processed/jra/upcoming/jra_<day>_<VENUE>/races.parquet` (post_time_local in JST). A race in
    `post_shift` is scheduled that many minutes later, or left out with None (N1, N2)."""
    post_shift = post_shift or {}
    for (day, venue), (kind, minutes) in schedule.items():
        posts = {
            no: post_of(rid(day, venue, no))
            + timedelta(minutes=minutes + post_shift.get(rid(day, venue, no), 0))
            for no in POST_LOCAL
            if post_shift.get(rid(day, venue, no), 0) is not None
        }
        if kind == "sidecar":
            sidecar = data_dir / "odds_snapshots" / "jra" / day / f"{venue}.races.json"
            odds_log.write_post_times(
                sidecar, posts, datetime.fromisoformat(day).replace(tzinfo=UTC)
            )
            continue
        assert kind == "card"
        card = data_dir / "processed" / "jra" / "upcoming" / f"jra_{day}_{venue}" / "races.parquet"
        card.parent.mkdir(parents=True)
        local = [posts[no].astimezone(JST).isoformat() for no in sorted(posts)]
        pq.write_table(pa.table({"race_no": sorted(posts), "post_time_local": local}), card)


def use_synthetic_holdout(mp: pytest.MonkeyPatch) -> None:
    """HOLDOUT_DAY is the only holdout day (in memory; holdout.yaml is untouched)."""
    start = date.fromisoformat(HOLDOUT_DAY)
    mp.setattr(
        holdout_module, "_load_definition", lambda countries_dir, country: (start, "s", None)
    )
    mp.setattr(holdout_module, "_is_holdout_day", lambda seed, day: day == HOLDOUT_DAY)


def use_premium(mp: pytest.MonkeyPatch, races) -> None:
    """The boost calendar (config/takeout_boosts.yaml, in memory only) holds one `premium` entry on
    the PLACE pool of each race in `races` and nothing else (Q)."""
    entries = []
    for race_id in sorted(races):
        _, day, venue, race_no = race_id.split(":")
        entries.append({"kind": "premium", "date": day, "venue": venue, "race_no": int(race_no)})
        entries[-1]["pools"] = ["PLACE"]
    loader = core_config.load_country_yaml

    def load(country, name, *args, **kwargs):
        if name == "takeout_boosts":
            return {"boosts": copy.deepcopy(entries)}
        return loader(country, name, *args, **kwargs)

    mp.setattr(core_config, "load_country_yaml", load)
    mp.setattr(avoidance_price, "load_country_yaml", load, raising=False)
    calendar = takeout.load_boosts()
    mp.setattr(takeout, "shipped_boosts", lambda: calendar)


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
    summary: dict = field(default_factory=dict)  # what run_study returned
    stdout: str = ""  # what main printed

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


def run_scenario(base: Path, scenario: Scenario, owner: dict, *, again: bool = False) -> Outcome:
    """The study's command line (main --run --out) on a fresh data directory whose store appears
    only once a family is frozen; run_study's returned summary and main's stdout are kept. With
    `again`, a rerun on the data directory already under `base` (its store and its registry as
    they were left): nothing is written to the store. An untraversable jravan directory (PD) gets
    its search permission back however the run ends, so the temporary directory can be removed."""
    data_dir, out_dir = base / "data", base / "out"
    if not again:
        data_dir.mkdir(parents=True)
    out_dir.mkdir()
    original, written = Registry.freeze_family, []
    real_write, writes = reports_module.write_report, []
    real_run, summaries = avoidance_price.run_study, []

    def freeze(self, family_id, spec):
        spec_hash = original(self, family_id, spec)
        if not written:
            if not again:
                write_store(data_dir, scenario)
            written.append(family_id)
        return spec_hash

    def write_report(path, text, *, sources, **kwargs):
        writes.append((Path(path), text, tuple(sources)))
        return real_write(path, text, sources=sources, **kwargs)

    def run_study(*args, **kwargs):
        summaries.append(real_run(*args, **kwargs))
        return summaries[-1]

    def broken_reader(*args, **kwargs):
        raise RuntimeError("synthetic fault: the pool_votes reader broke")

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HORSETRACK_DATA_DIR", str(data_dir))
        use_owner_config(mp, owner, scenario.folds)
        use_synthetic_holdout(mp)
        if scenario.premium or scenario.premium_withheld:
            use_premium(mp, scenario.premium | scenario.premium_withheld)
        if scenario.reader_fault:
            mp.setattr(avoidance_price, "read_place_votes", broken_reader)
        mp.setattr(Registry, "freeze_family", freeze)
        for module in (reports_module, avoidance_price, avoidance):
            mp.setattr(module, "write_report", write_report, raising=False)
        mp.setattr(avoidance_price, "run_study", run_study)
        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed):
                code = avoidance_price.main(["--run", "--out", str(out_dir)])
        finally:
            if scenario.votes_dir == "untraversable":
                (data_dir / "processed" / "jravan").chmod(0o755)
    assert code == 0
    assert written, "run_study never froze its family"
    assert len(summaries) == 1, "main runs the study once"
    registry_path = data_dir / "registry" / "japan.jsonl"
    entries = [json.loads(line) for line in registry_path.read_text().splitlines()]
    texts = {
        str(p.relative_to(out_dir)): p.read_text(encoding="utf-8")
        for p in sorted(out_dir.rglob("*"))
        if p.is_file() and p.suffix in (".md", ".csv", ".txt", ".json")
    }
    return Outcome(
        entries, texts, registry_path, base, out_dir, writes, summaries[0], printed.getvalue()
    )


SCENARIOS = {
    "A": lambda: (Scenario(), owner_config()),
    "B": lambda: (Scenario(late=True), owner_config()),
    "C": lambda: (Scenario(at_t=C_AT_T), owner_config()),
    "D": lambda: (Scenario(holdout=False), owner_config()),
    "E": lambda: (Scenario(final=E_FINAL), owner_config()),
    "F": lambda: (Scenario(), owner_config(l7_entropy_cut_points=list(F_CUTS))),
    "V": lambda: (Scenario(votes=V_VOTES), owner_config()),
    # review round 1 of b9a6b6d
    "N": lambda: (Scenario(untimed=UNTIMED, schedule=SCHEDULE), owner_config()),
    "N0": lambda: (Scenario(untimed=frozenset(LOGGED_IDS)), owner_config()),
    "S": lambda: (Scenario(scratched={SCRATCH_RACE: SCRATCH_HORSE}), owner_config()),
    "M": lambda: (Scenario(place=mispriced()), owner_config()),
    "W": lambda: (Scenario(no_place_at_t=frozenset(LOGGED_IDS)), owner_config()),
    "W1": lambda: (Scenario(no_place_at_t=frozenset(LOGGED_IDS) - {G2}), owner_config()),
    "P": lambda: (Scenario(no_place_at_t=frozenset({G5}), place_at_t={G1: {2}}), owner_config()),
    "U": lambda: (Scenario(votes_file="garbage"), owner_config()),
    "U0": lambda: (Scenario(votes_file="missing"), owner_config()),
    # review round 2 of 360216f
    "Q": lambda: (Scenario(premium=frozenset({PREMIUM_RACE})), owner_config()),
    "X": lambda: (Scenario(stale_scratch={STALE_RACE: STALE_HORSE}), owner_config()),
    "N1": lambda: (
        Scenario(untimed=UNTIMED, schedule=SCHEDULE, post_shift={SHIFTED_RACE: 1}),
        owner_config(),
    ),
    "N2": lambda: (
        Scenario(untimed=UNTIMED, schedule=SCHEDULE, post_shift={SHIFTED_RACE: None}),
        owner_config(),
    ),
    # fix round 4 of 1774fb3
    "Q0": lambda: (Scenario(premium_withheld=frozenset({PREMIUM_RACE})), owner_config()),
    "PU": lambda: (Scenario(votes_file="unopenable"), owner_config()),
    "R": lambda: (Scenario(reader_fault=True), owner_config()),
    "K": lambda: (Scenario(folds=dict(LATE_FOLDS)), owner_config()),
    # fix round 5 of e487d61
    "PD": lambda: (Scenario(votes_dir="untraversable"), owner_config()),
}
LOGGED_IDS = tuple(race.race_id for race in LOGGED)
# N: D3 TOK's captures keep their post_utc beside a stale sidecar (one minute late) that must not
# override it; D3 KYO's carry none and HKTrack's card gives the post; D4's carry none and the
# sidecar gives it.
UNTIMED = frozenset({G3, G4, G5, G6, G7, G8})
SCHEDULE = {
    (D3, "TOK"): ("sidecar", 1),
    (D3, "KYO"): ("card", 0),
    (D4, "TOK"): ("sidecar", 0),
    (D4, "KYO"): ("sidecar", 0),
}
# S: G7's favourite at t is excluded at the gate after t (the final results mark it scratched)
SCRATCH_RACE, SCRATCH_HORSE = G7, 6
# M: a non-floor placed horse's realised Place dividend, 30 yen above its reconstruction
MISPRICED_RACE, MISPRICED_HORSE, MISPRICE = H6, 2, 30
VALIDATION_TOLERANCE = 10  # yen per 100 staked, in the frozen spec
# Q: a floor-favourite race (in both 100-vs-110 checks) whose PLACE pool pays the 5% premium
PREMIUM_RACE = G6
# X: G8's 19/1 shot (unplaced) is SCR on the WIN board at T-3, its PLACE and Wide rows still the
# T-20 prices; the results mark it scratched
STALE_RACE, STALE_HORSE = G8, 7
# N1, N2: G8's scheduled post (its captures carry none in N) a minute later, or not scheduled
SHIFTED_RACE = G8


def mispriced() -> dict:
    race = RACES[MISPRICED_RACE]
    paid = reference_place_dividends(place_votes(race, VOTES[MISPRICED_RACE]), race.finish)
    assert paid[MISPRICED_HORSE] > 110  # not at the floor
    return {MISPRICED_RACE: {MISPRICED_HORSE: paid[MISPRICED_HORSE] + MISPRICE}}


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


DESCRIPTIVE = "paired_same_races:development"  # #81's scheme: every development race once


def is_descriptive(record: dict) -> bool:
    """A whole-development record (the historical descriptive output)."""
    return record["fold_scheme"] == DESCRIPTIVE


def is_walk_forward(record: dict) -> bool:
    return str(record["fold_scheme"]).startswith("walk_forward")


def fold_day(label) -> str | None:
    """The one test window's first day a metric's fold label names, or None (a pooled label)."""
    days = {day for day in FOLD_DAYS if day in str(label)}
    return days.pop() if len(days) == 1 else None


def pooled(record: dict) -> list[dict]:
    """The record's metrics over all its races (a walk-forward record's per-fold ones aside)."""
    return [m for m in record["metrics"] if fold_day(m["fold"]) is None]


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
    # By hand under JRA's stake-back rule (place_votes' docstring): plus10 and floor110 predict
    # 110, withheld 100. A: H1 floor110 (110, paid 110), H5 withheld (100, paid 110 by hand), H6
    # plus10 (110, paid 100 by hand), G1 and G6 plus10 (110, paid 110), G4 withheld (100, paid
    # 100). V: H1 withheld (100, paid 110), G4 floor110 (110, paid 100), the rest as in A.
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
            if record["hypothesis_id"] == "L2"
            and is_descriptive(record)
            and is_look_ahead(record)
            and cross_tab(record)
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
        metrics = pooled(record)  # a walk-forward record's per-fold metrics aside
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


# ---------------------------------------------------------------------------------------------
# Review round 1 of b9a6b6d: the post-time fallback, scratches after t, the reconstruction check,
# Plus-10 coverage at t, L2's no-bet race, the realised counts without the votes
# ---------------------------------------------------------------------------------------------

# The floor-bound placed favourites at t: race -> (favourite on the Win board at t, realised Place
# dividend), by hand from the fixture (the favourite's PLACE low figure at t at most 1.1, placed).
AT_T_FLOOR_FAVOURITES = {G1: (2, 110), G4: (1, 100), G6: (10, 110)}
EXOTIC_LEGS = {"QUINELLA_PLACE": 2, "QUINELLA": 2, "FORECAST": 2, "TRIO": 3, "TIERCE": 3}
ORDERED_POOLS = ("FORECAST", "TIERCE")
SCORED_CONTENT = re.compile(r"return|difference|realised|predicted|favourites|placed")
STATUS_WORDS = re.compile(r"(?i)\b(scored|failed|pending|withheld)\b")
SCHEDULED_SOURCE = re.compile(r"post_times_for|races\.json|sidecar|\bcard\b")


def place_paid(race: Race) -> dict[int, float]:
    return {
        int(r["combination"]): r["dividend"] for r in dividend_rows(race) if r["pool"] == "PLACE"
    }


def same_content(a: dict, b: dict) -> bool:
    """The same status and scored figures (returns, differences, the 100-vs-110 counts), other
    metrics (provenance counts among them) aside."""

    def scored_part(record):
        metrics = [m for m in record["metrics"] if SCORED_CONTENT.search(m["metric"])]
        return {"status": record["status"], "metrics": metrics}

    return same(scored_part(a), scored_part(b))


def table_rows(text: str) -> list[list[str]]:
    """The cells of each markdown table row in `text` (separator rows left out)."""
    rows = []
    for line in text.splitlines():
        if line.lstrip().startswith("|") and not set(line.strip()) <= set("|-: "):
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def rows_naming(outcome: Outcome, variant: str) -> list[list[str]]:
    """Every table row of every written file with a cell that is `variant`."""
    return [
        cells for _, text, _ in outcome.writes for cells in table_rows(text) if variant in cells
    ]


def reported_pending(outcome: Outcome, variant: str) -> bool:
    return any("data pending" in cell for cells in rows_naming(outcome, variant) for cell in cells)


def public_report(outcome: Outcome) -> str:
    """The main report: the one markdown file written on the jra store alone."""
    found = [
        text
        for path, text, sources in outcome.writes
        if path.suffix == ".md" and not is_withheld(sources)
    ]
    assert len(found) == 1, [str(path) for path, _, _ in outcome.writes]
    return found[0]


def plus10_names(outcome: Outcome) -> dict[str, str]:
    """The 100-vs-110 checks' variants: 'final' (the look-ahead one) and 'at t': the
    whole-development L2 records holding its cells, or built on the JRA-VAN votes (D21: a failed
    one holds no cell)."""
    found: dict[str, str] = {}
    for record in outcome.records:
        if not is_descriptive(record):
            continue
        if record["hypothesis_id"] == "L2" and (
            cross_tab(record) or realised_counts(record) or is_withheld(record["sources"])
        ):
            tag = "final" if is_look_ahead(record) else "at t" if is_at_t(record) else None
            assert tag and tag not in found, record["variant"]
            found[tag] = record["variant"]
    assert set(found) == {"final", "at t"}, found
    return found


def realised_counts(record: dict) -> dict[str, tuple[float, int]]:
    """The record's realised-only cells (`realised_<r>`, no prediction in the name)."""
    cells: dict[str, tuple[float, int]] = {}
    for m in record["metrics"]:
        name = m["metric"].lower()
        found = re.search(r"realised_(above_110|100|110)(?!\d)", name)
        if found and "predicted" not in name:
            assert found.group(1) not in cells, (record["variant"], name)
            cells[found.group(1)] = (m["value"], m["n"])
    return cells


def paired_means(record: dict) -> tuple[float, float, int]:
    """(the rule's mean return, the comparator's, n) of a scored paired comparison, over all its
    races (a walk-forward record's per-fold metrics aside)."""
    metrics = pooled(record)
    (diff,) = [m for m in metrics if "difference" in m["metric"] and m["lo"] is not None]
    n = diff["n"]
    (rule,) = side_returns(metrics, "rule", n)
    (comparator,) = side_returns(metrics, "comparator", n)
    return rule["value"], comparator["value"], n


def is_comparison(record: dict) -> bool:
    """A scored L2, L3 or L7 paired comparison (the 100-vs-110 check is not one)."""
    if record["hypothesis_id"] not in RULES or record["status"] != "scored":
        return False
    return not any(
        "realised" in m["metric"] or "predicted" in m["metric"] for m in record["metrics"]
    )


def numbers_in(value) -> list[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return [x for v in value for x in numbers_in(v)]
    return []


def numbers_under(value, part: str) -> list[float]:
    """The numbers held under every key whose name holds `part`."""
    if isinstance(value, dict):
        return [
            x
            for k, v in value.items()
            for x in (numbers_in(v) if part in str(k).lower() else numbers_under(v, part))
        ]
    if isinstance(value, list):
        return [x for v in value for x in numbers_under(v, part)]
    return []


def holds_text(value, text: str) -> bool:
    if isinstance(value, str):
        return value == text
    if isinstance(value, dict):
        value = list(value.values())
    return isinstance(value, list) and any(holds_text(v, text) for v in value)


def test_the_board_at_t_falls_back_to_the_scheduled_post_time(study):
    """Review 1 (high): a capture without post_utc takes post_times_for's scheduled post, t is that
    post - 2 min; the capture's own post_utc wins; with neither the race is data pending."""
    a, n, n0 = study("A"), study("N"), study("N0")
    # the fixture: N's stale sidecar (D3 TOK, one minute late) would put the capture fetched at
    # T-1:40 on the board at t, and G1's floor runner (2) is not at the floor there
    assert RACES[G1].at_t[2] <= 1.1 < RACES[G1].final[2]
    assert {(r.split(":")[1], r.split(":")[2]) for r in UNTIMED} | {(D3, "TOK")} == set(SCHEDULE)

    ra, rn, rn0 = a.by_variant(), n.by_variant(), n0.by_variant()
    at_t = [name for name, record in ra.items() if is_at_t(record)]
    assert [name for name in at_t if ra[name]["status"] == "scored"]
    assert n.spec_hash == a.spec_hash  # the schedule is data, not the spec
    for name in at_t:  # the scheduled post gives the same t as the capture's own
        assert same_content(ra[name], rn[name]), (name, rn[name]["status"], rn[name]["metrics"])
    # the run says where the post time came from
    assert SCHEDULED_SOURCE.search(json.dumps([n.spec, *(rn[name] for name in at_t)]))
    for name in at_t:  # neither the capture nor a schedule: data pending, not a board
        assert rn0[name]["status"] == "failed", name
        assert reported_pending(n0, name), name


def scratch_shares(record: dict) -> tuple[float, float]:
    """(rule, comparator): the share of the record's tickets in SCRATCH_RACE holding the scratched
    horse, from the frozen rules on the race's board at t."""
    horse, win = SCRATCH_HORSE, RACES[SCRATCH_RACE].at_t
    pool, cut, rule = record["params"]["pool"], record["params"].get("cut"), record["hypothesis_id"]
    legs = EXOTIC_LEGS.get(pool, 1)
    ranks = {h: 1 + sum(o < win[h] for o in win.values()) for h in win}

    def sets(horses) -> list[tuple[int, ...]]:
        if pool in ORDERED_POOLS:
            return list(itertools.permutations(horses, legs))
        return list(itertools.combinations(sorted(horses), legs))

    comparator = sets(HORSES)
    if rule == "L2":  # every set priced on the pool's board at t, less those at the floor
        low = {
            "WIN": lambda s: win[s[0]],
            "PLACE": lambda s: place_of(win[s[0]])[0],
            "QUINELLA_PLACE": lambda s: wide_of(win[s[0]], win[s[1]])[0],
        }[pool]
        kept = [s for s in comparator if low(s) > 1.1]
    elif rule == "L3" and legs == 1:  # the runners priced at most the cut at t
        kept = [s for s in comparator if win[s[0]] <= cut]
    elif rule == "L3":  # the combinations whose worst rank at t is better than the cut
        kept = [s for s in comparator if max(ranks[h] for h in s) < cut]
    else:  # L7: the favourite's ticket or the top-3 box at t, in a race the rule does not skip
        kept = comparator = sets(sorted(win, key=win.get)[: 3 if legs > 1 else 1])

    def share(tickets) -> float:
        return sum(horse in s for s in tickets) / len(tickets)

    return share(kept), share(comparator)


def test_a_runner_scratched_after_t_is_selected_and_refunded(study):
    """Review 1 (high): selection at t reads the board at t, not the final results. The favourite
    at t, excluded after t, is still the favourite, still in the box and every set, and each
    ticket holding it is refunded (stake back) though the store has no refund row for it."""
    a, s = study("A"), study("S")
    race, horse = RACES[SCRATCH_RACE], SCRATCH_HORSE
    # the fixture: the horse is the favourite at t and unplaced (a ticket holding it loses in A),
    # in a race L7 does not skip at t (no handicap, not the flattest quintile)
    assert min(race.at_t, key=race.at_t.get) == horse and horse not in race.finish
    assert race.weight_type != "handicap"
    assert entropy_quintile(reference_entropy(race.at_t), CUTS) < 5
    # the jra store says scratched, and holds no refund row
    assert not any(r["is_refund"] for r in dividend_rows(race))

    ra, rs = a.by_variant(), s.by_variant()
    checked = Counter()
    for name, record in ra.items():
        if not (is_at_t(record) and is_comparison(record)):
            continue
        rule_a, comparator_a, n = paired_means(record)
        assert rs[name]["status"] == "scored", name
        rule_s, comparator_s, n_s = paired_means(rs[name])
        assert n_s == n, name  # the race stays paired, on the same tickets
        rule_share, comparator_share = scratch_shares(record)
        assert rule_share > 0 and comparator_share > 0, name
        # each ticket holding the horse returns its stake instead of losing: the race's return
        # rises by the share of such tickets, the mean by that over n
        got = (rule_s - rule_a, comparator_s - comparator_a)
        want = (rule_share / n, comparator_share / n)
        assert all(
            math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12) for x, y in zip(got, want, strict=True)
        ), (name, got, want)
        checked[record["hypothesis_id"]] += 1
    assert all(checked[rule] for rule in RULES), checked


def within_tolerance(record: dict) -> tuple[float, int]:
    """(placed horses within the tolerance, placed horses checked): the validation metric."""
    (metric,) = [
        m
        for m in pooled(record)
        if "validat" in m["metric"].lower() and "within" in m["metric"].lower()
    ]
    return metric["value"], metric["n"]


def test_the_place_payout_reconstruction_is_checked_before_the_cross_tab(study):
    """Review 1 (high): the 100-vs-110 check validates its reconstructed Place dividends against
    the realised ones of every placed horse of its races; the check is in the (withheld) record,
    its tolerance in the frozen spec. Fix round 4 (B), changed from 'a miss fails the record': a
    miss above the tolerance is a diagnostic, never a rejection: the record stays scored with its
    metrics, its cross-tab and realised counts those of A, the miss shown in its validation
    figures and labelled discrepant on its race's row."""
    a, m, v = study("A"), study("M"), study("V")
    # the fixture: every placed horse of the check's races is within one 10-yen step of the
    # reference payout of its votes (A's and V's), the non-floor ones exactly; M misprices one
    # non-floor horse by more than the tolerance
    non_floor = 0
    for profiles in (VOTES, VOTES | V_VOTES):
        for race_id in (*FLOOR_FAVOURITES, *AT_T_FLOOR_FAVOURITES):
            race = RACES[race_id]
            want = reference_place_dividends(place_votes(race, profiles[race_id]), race.finish)
            paid = place_paid(race)
            for h in race.finish:
                assert abs(paid[h] - want[h]) <= VALIDATION_TOLERANCE, (race_id, h)
                if want[h] > 110:
                    assert paid[h] == want[h], (race_id, h)
                    non_floor += 1
    assert non_floor
    assert MISPRICED_RACE in FLOOR_FAVOURITES and MISPRICE > VALIDATION_TOLERANCE
    assert {
        race: {h: d + MISPRICE for h, d in place_paid(RACES[race]).items() if h == MISPRICED_HORSE}
        for race in (MISPRICED_RACE,)
    } == mispriced()

    for outcome in (a, m):  # the tolerance is frozen with the family
        tolerances = numbers_under(outcome.spec, "tolerance")
        assert tolerances and set(tolerances) == {VALIDATION_TOLERANCE}, tolerances

    names = plus10_names(a)
    races = {"final": len(FLOOR_FAVOURITES), "at t": len(AT_T_FLOOR_FAVOURITES)}
    for outcome in (a, v):  # within the tolerance: scored, with the check in the record
        for tag, name in names.items():
            record = outcome.by_variant()[name]
            assert record["status"] == "scored", (tag, name)
            checks = [x for x in record["metrics"] if re.search("validat", x["metric"].lower())]
            assert checks, (tag, [x["metric"] for x in record["metrics"]])
            # every placed horse of its races, the non-floor ones included
            assert max(x["n"] for x in checks) >= 3 * races[tag], (tag, checks)
            assert is_withheld(record["sources"]), (tag, record["sources"])
            within, checked = within_tolerance(record)
            assert within == checked >= 3 * races[tag], (tag, within, checked)

    # a miss above the tolerance (one placed horse of H6): a diagnostic, the evidence kept
    record, kept = m.by_variant()[names["final"]], a.by_variant()[names["final"]]
    assert record["status"] == "scored", record["metrics"]
    assert cross_tab(record) == cross_tab(kept), cross_tab(record)
    assert realised_counts(record) == realised_counts(kept)
    within, checked = within_tolerance(record)
    assert (within, checked) == (within_tolerance(kept)[0] - 1, within_tolerance(kept)[1])
    assert not reported_pending(m, names["final"]), rows_naming(m, names["final"])
    # the check at t holds no race of H6's day: nothing moves there
    record, kept = m.by_variant()[names["at t"]], a.by_variant()[names["at t"]]
    assert same(record, kept), record["metrics"]


def test_plus10_at_t_counts_a_race_only_with_a_place_price_at_t(study):
    """Review 1 (medium): a race with Win prices at t but no PLACE price for its favourite at t is
    not covered; with no covered race the at-t check is data pending, never scored with n = 0."""
    a, w, w1 = study("A"), study("W"), study("W1")
    name = plus10_names(a)["at t"]
    # the fixture: G2's favourite at t has a PLACE price at t, off the floor
    favourite = min(RACES[G2].at_t, key=RACES[G2].at_t.get)
    assert place_of(RACES[G2].at_t[favourite])[0] > 1.1

    record = w.by_variant()[name]  # the Place board arrives after t in every race
    assert record["status"] == "failed", record["metrics"]
    assert reported_pending(w, name)
    record = w1.by_variant()[name]  # one covered race, its favourite off the floor: a zero
    assert record["status"] == "scored", name
    assert not any(value for value, _ in cross_tab(record).values())
    assert not any(value for value, _ in realised_counts(record).values())
    assert not reported_pending(w1, name)


def place_floor_returns(race: Race, lows: dict[int, float]) -> tuple[float, float]:
    """L2 on one race's PLACE board at t (horse -> low figure): the (rule, comparator) return per
    unit staked at the realised dividends; the rule's side returns its stake when it bets none."""
    paid = place_paid(race)

    def back(horses) -> float:
        return sum(paid.get(h, 0.0) for h in horses) / (100 * len(horses))

    kept = [h for h, low in lows.items() if low > 1.1]
    return (back(kept) if kept else 1.0), back(list(lows))


def test_l2_keeps_a_no_bet_race_in_the_pairing(study):
    """Review 1 (medium): a race whose every priced ticket is at the floor stays paired, the
    rule's side returning its stake (stake_back, in the spec); only missing board data drops."""
    a, p = study("A"), study("P")
    # the fixture: on G1's PLACE board at t only its floor runner (2) is priced in P; G5 has no
    # PLACE board at t in P
    lows_g1 = {h: place_of(RACES[G1].at_t[h])[0] for h in HORSES}
    lows_g5 = {h: place_of(RACES[G5].at_t[h])[0] for h in HORSES}
    assert [h for h, low in lows_g1.items() if low <= 1.1] == [2]

    ra, rp = a.by_variant(), p.by_variant()
    (name,) = [
        n
        for n, record in ra.items()
        if record["hypothesis_id"] == "L2"
        and is_descriptive(record)
        and is_at_t(record)
        and is_comparison(record)
        and record["params"]["pool"] == "PLACE"
    ]
    rule_a, comparator_a, n_a = paired_means(ra[name])
    assert n_a == len(LOGGED)
    assert rp[name]["status"] == "scored", name
    rule_p, comparator_p, n_p = paired_means(rp[name])
    assert n_p == n_a - 1  # G1 stays (no bet is a decision), G5 drops (no board)

    rule_g1, comparator_g1 = place_floor_returns(RACES[G1], lows_g1)
    rule_g5, comparator_g5 = place_floor_returns(RACES[G5], lows_g5)
    no_bet, floor_ticket = place_floor_returns(RACES[G1], {2: lows_g1[2]})
    assert 2 in RACES[G1].finish and no_bet == 1.0
    assert floor_ticket == place_paid(RACES[G1])[2] / 100
    assert math.isclose(rule_p * n_p, rule_a * n_a - rule_g1 - rule_g5 + no_bet, rel_tol=1e-9)
    assert math.isclose(
        comparator_p * n_p,
        comparator_a * n_a - comparator_g1 - comparator_g5 + floor_ticket,
        rel_tol=1e-9,
    )
    assert holds_text(p.spec["rules"]["L2"], "stake_back")


def test_realised_floor_dividends_survive_without_the_votes(study):
    """Review 1 (medium): a missing or unreadable pool_votes.parquet leaves the realised counts of
    the jra store in a scored record and in the public report; only the prediction is gone."""
    a = study("A")
    names = plus10_names(a)
    # the fixture: the realised cells by hand, at the final price and at t
    derived = {}
    for race in LOGGED:
        favourite = min(race.at_t, key=race.at_t.get)
        if place_of(race.at_t[favourite])[0] <= 1.1 and favourite in race.finish:
            derived[race.race_id] = (favourite, place_paid(race)[favourite])
    assert derived == AT_T_FLOOR_FAVOURITES
    want = {
        "final": Counter(realised_cell(paid) for _, paid in FLOOR_FAVOURITES.values()),
        "at t": Counter(realised_cell(paid) for _, paid in AT_T_FLOOR_FAVOURITES.values()),
    }
    assert want["final"] == {"110": 4, "100": 2} and want["at t"] == {"110": 2, "100": 1}

    shown = [  # the public report's realised rows in A (no status row)
        cells
        for cells in table_rows(public_report(a))
        if set(names.values()) & set(cells) and not STATUS_WORDS.search(" ".join(cells))
    ]
    assert shown
    for outcome in (a, study("U"), study("U0")):
        for tag, name in names.items():
            record = outcome.by_variant()[name]
            assert record["status"] == "scored", (tag, name)
            got = {cell: value for cell, (value, _) in realised_counts(record).items() if value}
            assert got == dict(want[tag]), (tag, got)
            if outcome is not a:  # no prediction without the votes
                assert not any(value for value, _ in cross_tab(record).values()), tag
        text = public_report(outcome)
        assert not re.search(r"(?i)no floor-bound favourites? placed", text)
        rows = table_rows(text)
        for cells in shown:
            assert any(set(cells) <= set(other) for other in rows), cells


# ---------------------------------------------------------------------------------------------
# Review round 2 of 360216f: JRA's Place rule and its premium, a prediction that cannot be made,
# a scratch at t with a stale price opposite it, the rejected predictions' CSV rows, the
# post-time build's identity
# ---------------------------------------------------------------------------------------------

VERDICT_CELL = re.compile(r"(?i)^within( tolerance)?$")
HEX_DIGEST = re.compile(r"[0-9a-f]{8,64}")


def expected_cells(floor: dict[str, tuple[int, float]], races) -> Counter:
    """(predicted, realised) -> favourites over `races` of a check's floor favourites, from the
    votes profiles (PREDICTED) and the realised dividends written."""
    return Counter(
        (PREDICTED[VOTES[race]], realised_cell(floor[race][1])) for race in floor if race in races
    )


def test_an_unresolved_premium_leaves_the_prediction_unavailable(study):
    """Fix round 4 (A), changed from review 2's 'a premium race is rebuilt with its premium': JRA
    may withhold a premium (takeout's boost_may_be_withheld) and the store holds no settlement
    signal, so a race the boost calendar gives a Place premium has its prediction unavailable,
    with a reason naming the premium, and the premium is never inferred from the realised
    dividends: the race where it was withheld (Q0) and the race where it was paid (Q) give the
    same prediction rows. The realised dividend and every other race stay."""
    a, q, q0 = study("A"), study("Q"), study("Q0")
    race = RACES[PREMIUM_RACE]
    favourite, realised = FLOOR_FAVOURITES[PREMIUM_RACE]
    votes = place_votes(race, VOTES[PREMIUM_RACE])
    base = reference_place_dividends(votes, race.finish)
    paid = reference_place_dividends(votes, race.finish, premium=True)
    # the fixture, by hand: the favourite (700,000 tickets) gains nothing and stays at 110; the
    # others (30,000 and 20,000) gain 10 x floor(1,000,000 / 180,000) = 50 and 10 x floor(
    # 1,000,000 / 120,000) = 80 when the premium is paid (Q). Q0's store is A's: the premium
    # withheld, the dividends the base ones, 50 and 80 below what a premium rebuild would give.
    others = [h for h in race.finish if h != favourite]
    assert paid[favourite] == base[favourite] == realised == 110
    assert [paid[h] - base[h] for h in others] == [50, 80]
    assert [base[h] for h in others] == [300, 410]
    assert {h: d for h, d in place_paid(race).items()} == base  # A's and Q0's store
    assert PREMIUM_RACE in FLOOR_FAVOURITES and PREMIUM_RACE in AT_T_FLOOR_FAVOURITES
    # the boost calendar of Q and Q0 marks the race's PLACE pool, and only that pool
    with pytest.MonkeyPatch.context() as mp:
        use_premium(mp, {PREMIUM_RACE})
        place = takeout.race_takeout(PREMIUM_RACE, "PLACE")
        assert place.boost == "premium" and place.boost_may_be_withheld
        assert takeout.race_takeout(PREMIUM_RACE, "WIN").boost is None
        assert takeout.race_takeout(G1, "PLACE").boost is None

    names = plus10_names(a)
    floors = {"final": FLOOR_FAVOURITES, "at t": AT_T_FLOOR_FAVOURITES}
    for tag, name in names.items():
        floor = floors[tag]
        others_races = [r for r in floor if r != PREMIUM_RACE]
        want = expected_cells(floor, others_races)
        assert sum(want.values()) == len(floor) - 1
        rows = {}
        for outcome in (q0, q):
            record = outcome.by_variant()[name]
            assert record["status"] == "scored", (tag, record["metrics"])
            assert realised_counts(record) == realised_counts(a.by_variant()[name]), tag
            # the other races' predictions, the premium race's left out
            cells = cross_tab(record)
            assert {c: v for c, (v, _) in cells.items() if v} == dict(want), (tag, cells)
            assert {n for _, n in cells.values()} == {len(floor) - 1}, (tag, cells)
            within, checked = within_tolerance(record)  # nothing rebuilt misses
            assert within == checked, (tag, within, checked)

            frame = plus10_csv(outcome)
            frame = frame.loc[frame["variant"].eq(name)]
            got = {
                r["race_id"]: r
                for r in frame.to_dict("records")
                if r["race_id"] in floor  # every floor favourite keeps its row
            }
            assert set(got) == set(floor), (tag, sorted(got))
            for race_id, row in got.items():
                assert float(row["realised"]) == floor[race_id][1], (tag, row)
                if race_id == PREMIUM_RACE:  # unavailable, with its reason
                    assert pd.isna(row["predicted"]), (tag, row)
                    assert re.search(r"(?i)premium", str(row["why"])), (tag, row)
                else:
                    assert row["predicted"] == PREDICTED[VOTES[race_id]], (tag, row)
            rows[outcome is q] = sorted(
                (r["race_id"], str(r["predicted"]), str(r["why"])) for r in got.values()
            )
        assert rows[True] == rows[False], tag  # paid or withheld: the same prediction rows

    # nothing else moves with an unresolved premium (Q0's store is A's): only the checks built on
    # the votes
    votes_built = {n for n, r in a.by_variant().items() if is_withheld(r["sources"])}
    assert set(names.values()) <= votes_built
    assert set(moved(a, q0)) <= votes_built, sorted(set(moved(a, q0)) - votes_built)


def test_unreadable_votes_leave_the_prediction_unavailable(study):
    """Review 2 (medium): with floor-bound placed favourites and no votes to rebuild any of them,
    the check is not run and the prediction is unavailable (absent cells): no predicted cell, no
    validation figure, no 'within' verdict anywhere. A covered selection with no eligible
    favourite keeps its scored zero, every cell present at 0."""
    a, w1 = study("A"), study("W1")
    names = plus10_names(a)
    for outcome in (study("U"), study("U0")):
        for tag, name in names.items():
            record = outcome.by_variant()[name]
            assert record["status"] == "scored", (tag, name)  # the realised counts stand
            assert any(value for value, _ in realised_counts(record).values()), tag
            assert cross_tab(record) == {}, (tag, cross_tab(record))
            validation = [m["metric"] for m in record["metrics"] if "validat" in m["metric"]]
            assert validation == [], (tag, validation)
            verdicts = [c for cells in rows_naming(outcome, name) for c in cells]
            assert not [c for c in verdicts if VERDICT_CELL.match(c)], (tag, verdicts)

    # W1: one covered race, its favourite off the floor: a scored zero, not an unavailable one
    record = w1.by_variant()[names["at t"]]
    assert record["status"] == "scored"
    cells = cross_tab(record)
    assert cells and not any(value for value, _ in cells.values()), cells


def raw_board_at_t(outcome: Outcome, race_id: str) -> pd.DataFrame:
    """The race's odds_timeseries rows at t in the outcome's store, through #81's view_at."""
    day = race_id.split(":")[1]
    path = outcome.base / "data" / "processed" / "jra" / "odds_timeseries" / f"{day}.parquet"
    con = duckdb.connect(":memory:")
    try:
        con.register("_rows", pq.read_table(path))
        con.execute('CREATE TABLE "odds_timeseries" AS SELECT * FROM _rows')
        con.unregister("_rows")
        t = post_of(race_id) - timedelta(minutes=DECISION_MIN)
        return view_at(con, race_id, t, rules=load_availability("japan")).board
    finally:
        con.close()


def sets_of(pool: str, horses) -> list[tuple[int, ...]]:
    legs = EXOTIC_LEGS.get(pool, 1)
    if pool in ORDERED_POOLS:
        return list(itertools.permutations(sorted(horses), legs))
    return list(itertools.combinations(sorted(horses), legs))


def flat_return(race: Race, pool: str, tickets) -> float:
    """One unit on each ticket (a tuple of horses, in the pool's order), at the race's dividends:
    the payout per unit staked."""
    paid = {(r["pool"], r["combination"]): r["dividend"] for r in dividend_rows(race)}
    return sum(paid.get((pool, ",".join(map(str, t))), 0.0) for t in tickets) / (100 * len(tickets))


def stale_race_returns(record: dict, runners) -> tuple[float, float]:
    """(rule, comparator) return in STALE_RACE of an at-t L2 PLACE or Wide record, or an at-t L3
    rank record, when the runners at t are `runners`, from the frozen rules on its board at t."""
    race = RACES[STALE_RACE]
    win = {h: race.at_t[h] for h in runners}
    pool, cut = record["params"]["pool"], record["params"].get("cut")
    comparator = sets_of(pool, runners)
    if record["hypothesis_id"] == "L2":  # the sets priced at t less those at the floor
        low = {
            "PLACE": lambda s: place_of(win[s[0]])[0],
            "QUINELLA_PLACE": lambda s: wide_of(win[s[0]], win[s[1]])[0],
        }[pool]
        rule = [s for s in comparator if low(s) > 1.1]
    else:  # the combinations whose worst rank by WIN odds at t is better than the cut
        ranks = {h: 1 + sum(o < win[h] for o in win.values()) for h in win}
        rule = [s for s in comparator if max(ranks[h] for h in s) < cut]
    return flat_return(race, pool, rule), flat_return(race, pool, comparator)


def test_a_scratch_at_t_clears_the_runner_from_every_pool(study):
    """Review 2 (medium): the board-at-t rule (docs/specs/board-at-t.md): a runner whose latest
    WIN or PLACE row at t is SCR is scratched and has no price in any pool, Wide pairs included.
    A stale PLACE or Wide price opposite the SCR row does not make it a runner at t: L2 does not
    bet it, and L3 keeps the race, ranking the runners left."""
    a, x = study("A"), study("X")
    race, horse = RACES[STALE_RACE], STALE_HORSE
    # the fixture: the horse is unplaced, not the favourite at t, and nothing of the race is at
    # the floor at t; on X's raw board at t its WIN row is SCR while its PLACE row and its Wide
    # pairs still carry the T-20 prices
    assert horse not in race.finish and min(race.at_t, key=race.at_t.get) != horse
    assert all(place_of(o)[0] > 1.1 for o in race.at_t.values())
    board = raw_board_at_t(x, STALE_RACE)
    rows = {(p, c): r for p, c, r in zip(board["pool"], board["combination"],
                                         board.to_dict("records"), strict=True)}  # fmt: skip
    win_row, place_row = rows[("WIN", str(horse))], rows[("PLACE", str(horse))]
    assert win_row["status_raw"] == SCRATCHED and pd.isna(win_row["odds"])
    assert math.isclose(place_row["odds"], place_of(race.at_t[horse])[0])
    pairs = [
        r for (p, c), r in rows.items() if p == "QUINELLA_PLACE" and str(horse) in c.split(",")
    ]
    assert len(pairs) == len(HORSES) - 1 and all(pd.notna(r["odds"]) for r in pairs)

    ra, rx = a.by_variant(), x.by_variant()
    left = [h for h in HORSES if h != horse]
    checked = Counter()
    for name, record in ra.items():
        if not (is_at_t(record) and is_comparison(record)):
            continue
        rule, pool = record["hypothesis_id"], record["params"]["pool"]
        if not (
            (rule == "L2" and pool in ("PLACE", "QUINELLA_PLACE"))
            or (rule == "L3" and pool in EXOTIC_LEGS)
        ):
            continue
        rule_a, comparator_a, n = paired_means(record)
        assert rx[name]["status"] == "scored", name
        rule_x, comparator_x, n_x = paired_means(rx[name])
        assert n_x == n, (name, n_x, n)  # the race stays paired
        before, after = stale_race_returns(record, HORSES), stale_race_returns(record, left)
        want = ((after[0] - before[0]) / n, (after[1] - before[1]) / n)
        got = (rule_x - rule_a, comparator_x - comparator_a)
        assert all(
            math.isclose(g, w, rel_tol=1e-9, abs_tol=1e-12) for g, w in zip(got, want, strict=True)
        ), (name, got, want)
        if rule == "L2":  # the horse's own tickets are gone, not refunded
            assert want[1] != 0, name
        checked[(rule, pool)] += 1
    assert set(checked) == {("L2", "PLACE"), ("L2", "QUINELLA_PLACE")} | {
        ("L3", pool) for pool in EXOTIC_LEGS
    }, checked


def plus10_csv(outcome: Outcome) -> pd.DataFrame:
    """The private companion CSV of the 100-vs-110 check (written with the JRA-VAN source)."""
    found = [
        text
        for path, text, sources in outcome.writes
        if path.suffix == ".csv" and "plus10" in path.name and is_withheld(sources)
    ]
    assert len(found) == 1, [str(path) for path, _, _ in outcome.writes]
    return pd.read_csv(io.StringIO(found[0]))


DISCREPANT = re.compile(r"(?i)discrepan")


def test_a_discrepancy_is_labelled_on_its_row_and_keeps_the_prediction(study):
    """Fix round 4 (B), changed from review 2's 'a failed check leaves no prediction in its CSV':
    a reconstruction miss above the tolerance (M: a placed horse of H6 that is not the favourite)
    labels the row of its race discrepant in the private CSV, and that row keeps its realised
    dividend and its prediction; no other row is labelled, nor any row when nothing misses (A)."""
    a, m = study("A"), study("M")
    names = plus10_names(a)
    assert MISPRICED_RACE in FLOOR_FAVOURITES and MISPRICED_RACE not in AT_T_FLOOR_FAVOURITES

    for outcome in (a, m):
        for tag, floor in (("final", FLOOR_FAVOURITES), ("at t", AT_T_FLOOR_FAVOURITES)):
            frame = plus10_csv(outcome)
            rows = frame.loc[frame["variant"].eq(names[tag])]
            triples = zip(rows["race_id"], rows["horse_no"], rows["realised"], strict=True)
            got = {(r, int(h)): float(d) for r, h, d in triples}
            assert got == {(r, h): float(d) for r, (h, d) in floor.items()}, (tag, got)
            for row in rows.to_dict("records"):
                # every attempted prediction stays, whatever its diagnostic
                assert row["predicted"] == PREDICTED[VOTES[row["race_id"]]], (tag, row)
                labelled = any(DISCREPANT.search(str(v)) for v in row.values() if pd.notna(v))
                want = outcome is m and row["race_id"] == MISPRICED_RACE
                assert labelled == want, (tag, row)


def post_builds(outcome: Outcome) -> dict[str, tuple[str, str, str]]:
    """Each at-t record's post-time data build: variant -> (build name, identity, refresh)."""
    found = {}
    for record in outcome.records:
        if not is_at_t(record):
            continue
        keys = [k for k in record["data_builds"] if "post" in k.lower()]
        assert len(keys) == 1, (record["variant"], record["data_builds"])
        identity, _, refresh = record["data_builds"][keys[0]].rpartition("@")
        found[record["variant"]] = (keys[0], identity, refresh)
    assert found
    return found


def test_the_post_time_build_keeps_one_identity(study, tmp_path):
    """Review 2 (medium): the at-t records' post-time input is one build, '<identity>@<digest>',
    the digest a hex digest of the race -> post-time mapping. One race's post time moved (N1) or
    backfilled (N against N2) moves the digest, never the identity, so the registry counts one
    family size over those runs; the coverage counts stay in the report."""
    outcomes = {name: study(name) for name in ("A", "N", "N1", "N2")}
    assert len({o.spec_hash for o in outcomes.values()}) == 1
    builds = {name: post_builds(o) for name, o in outcomes.items()}
    for name, found in builds.items():
        for variant, (_, identity, refresh) in found.items():
            assert identity and HEX_DIGEST.fullmatch(refresh), (name, variant, identity, refresh)
    identities = {
        (key, identity) for found in builds.values() for key, identity, _ in found.values()
    }
    assert len(identities) == 1, identities  # whatever the coverage, one identity

    def refreshes(name: str) -> set[str]:
        return {refresh for _, _, refresh in builds[name].values()}

    for other in ("N1", "N2"):  # one race's post time changed, or backfilled: another digest
        assert not refreshes("N") & refreshes(other), (other, refreshes("N"), refreshes(other))

    combined = Registry(tmp_path / "combined.jsonl", holdout_log=tmp_path / "no-holdout.jsonl")
    family = outcomes["A"].freezes[0]["family_id"]
    for outcome in outcomes.values():
        reader = Registry(outcome.registry_path, holdout_log=tmp_path / "no-holdout.jsonl")
        present = combined.records()
        for record in reader.records(family):
            if record not in present:
                combined.append(record)
    assert combined.family_size(family) == outcomes["A"].spec["family_size"]

    def coverage(outcome: Outcome) -> list[list[str]]:
        return [cells for cells in table_rows(public_report(outcome)) if "post" in cells[0]]

    assert coverage(outcomes["N"]) and coverage(outcomes["N"]) != coverage(outcomes["N2"])


# ---------------------------------------------------------------------------------------------
# Fix round 4 of 1774fb3 (the owner-requested restart plan): counts from the row statuses, the
# read boundary, an unopenable votes file, walk-forward rows, the CLI summary, the v5 family
# ---------------------------------------------------------------------------------------------

COUNT_WORDS = {  # B: the counts, in this order, from the final row statuses
    "attempted": re.compile(r"attempted"),
    "available": re.compile(r"(?<!un)available"),
    "unavailable": re.compile(r"unavailable"),
    "discrepant": re.compile(r"discrepan"),
}
# (attempted, available, unavailable, discrepant) of each check, by hand from the fixture: A every
# floor-bound placed favourite predicted, nothing off; M one race's reconstruction off (H6, in the
# check at the final price only); U the votes unreadable; Q0 the premium race (G6, in both checks)
# unresolved
EXPECTED_COUNTS = {
    "A": {"final": (6, 6, 0, 0), "at t": (3, 3, 0, 0)},
    "M": {"final": (6, 6, 0, 1), "at t": (3, 3, 0, 0)},
    "U": {"final": (6, 0, 6, 0), "at t": (3, 0, 3, 0)},
    "Q0": {"final": (6, 5, 1, 0), "at t": (3, 2, 1, 0)},
}


def markdown_tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Each markdown table in `text`: (its header cells, its rows' cells)."""
    tables: list[tuple[list[str], list[list[str]]]] = []
    header: list[str] | None = None
    rows: list[list[str]] = []
    for line in [*text.splitlines(), ""]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if header is not None:
                tables.append((header, rows))
            header, rows = None, []
            continue
        if set(stripped) <= set("|-: "):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if header is None:
            header = cells
        else:
            rows.append(cells)
    return tables


def private_report(outcome: Outcome) -> str:
    """The private report: the one markdown file written with the JRA-VAN source."""
    found = [
        text
        for path, text, sources in outcome.writes
        if path.suffix == ".md" and is_withheld(sources)
    ]
    assert len(found) == 1, [str(path) for path, _, _ in outcome.writes]
    return found[0]


def registry_counts(record: dict) -> tuple[int, ...]:
    got = []
    for word, pattern in COUNT_WORDS.items():
        found = [m for m in pooled(record) if pattern.search(m["metric"].lower())]
        names = [m["metric"] for m in record["metrics"]]
        assert len(found) == 1, (record["variant"], word, names)
        got.append(int(found[0]["value"]))
    return tuple(got)


def private_counts(outcome: Outcome, variant: str) -> tuple[int, ...]:
    """The counts row of `variant` in the private report's table holding the four counts."""
    hits = []
    for header, rows in markdown_tables(private_report(outcome)):
        where = {
            word: [i for i, cell in enumerate(header) if pattern.search(cell.lower())]
            for word, pattern in COUNT_WORDS.items()
        }
        if not all(len(found) == 1 for found in where.values()):
            continue
        for cells in rows:
            if variant in cells:
                hits.append(tuple(int(float(cells[where[w][0]])) for w in COUNT_WORDS))
    assert len(hits) == 1, (variant, hits)
    return hits[0]


def flagged(row: dict) -> bool:
    return any(DISCREPANT.search(str(value)) for value in row.values() if pd.notna(value))


def csv_counts(outcome: Outcome, variant: str) -> tuple[int, ...]:
    """The counts the private CSV's rows of `variant` give: a row with a prediction is available,
    one with a cell matching 'discrepan' discrepant."""
    frame = plus10_csv(outcome)
    rows = frame.loc[frame["variant"].eq(variant)]
    available = int(rows["predicted"].notna().sum())
    discrepant = sum(flagged(row) for row in rows.to_dict("records"))
    return len(rows), available, len(rows) - available, discrepant


def test_prediction_counts_agree_across_every_output(study):
    """Fix round 4 (B): the counts come from the final row statuses, attempted, available,
    unavailable and discrepant (separately), and agree across the registry metrics, the private
    report, the private CSV and, for the attempted count alone, the public report; the public
    report gives none of the counts built on the votes (D21)."""
    names = plus10_names(study("A"))
    for scenario, expected in EXPECTED_COUNTS.items():
        outcome = study(scenario)
        public = markdown_tables(public_report(outcome))
        for tag, name in names.items():
            want = expected[tag]
            attempted, available, unavailable, discrepant = want
            assert attempted == available + unavailable and discrepant <= available
            record = outcome.by_variant()[name]
            assert record["status"] == "scored", (scenario, tag, record["metrics"])
            assert sum(v for v, _ in realised_counts(record).values()) == attempted
            assert registry_counts(record) == want, (scenario, tag)
            assert private_counts(outcome, name) == want, (scenario, tag)
            assert csv_counts(outcome, name) == want, (scenario, tag)

            placed = set()
            for header, rows in public:
                naming = [cells for cells in rows if name in cells]
                if not naming:
                    continue
                for word in ("available", "unavailable", "discrepant"):  # built on the votes
                    assert not any(COUNT_WORDS[word].search(c.lower()) for c in header), header
                assert not any(DISCREPANT.search(c) for cells in naming for c in cells), naming
                where = [i for i, c in enumerate(header) if c.lower() in ("placed", "attempted")]
                placed |= {int(float(cells[where[0]])) for cells in naming if where}
            assert placed == {attempted}, (scenario, tag, placed)


# C: the read boundary, traced. A frame of the package or of pyarrow's Python layer that holds a
# holdout row's value while a reader runs has materialised it.
PACKAGE_DIR = os.path.realpath(Path(avoidance_price.__file__).resolve().parents[3])
PYARROW_DIR = os.path.realpath(Path(pa.__file__).resolve().parent)


def holds(value, hit, depth: int = 0) -> bool:
    """Whether `value` (Arrow, pandas or numpy data, a container or dataclass of them, or a
    scalar) holds a scalar that `hit` flags."""
    if depth > 4:
        return False
    if isinstance(value, (pa.Table, pa.RecordBatch)):
        return any(holds(column, hit, depth + 1) for column in value.columns)
    if isinstance(value, (pa.Array, pa.ChunkedArray)):
        return any(hit(x) for x in value.to_pylist())
    if isinstance(value, pa.Scalar):
        return hit(value.as_py())
    if isinstance(value, pd.DataFrame):
        return any(holds(value.iloc[:, i], hit, depth + 1) for i in range(value.shape[1]))
    if isinstance(value, (pd.Series, pd.Index, np.ndarray)):
        return any(hit(x) for x in np.asarray(value, dtype=object).ravel().tolist())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(holds(v, hit, depth + 1) for v in value)
    if isinstance(value, dict):
        return any(holds(v, hit, depth + 1) for v in value.values())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        return any(holds(getattr(value, f.name), hit, depth + 1) for f in fields)
    return bool(hit(value))


class ReadTrace:
    """While the block runs, every local of every Python frame of the package or of pyarrow's
    Python layer, and every value such a frame returns or yields, searched for a scalar `hit`
    flags; `found` names the frames and lines where one was held."""

    def __init__(self, hit) -> None:
        self.hit, self.found = hit, []
        self._watched: dict[str, bool] = {}

    def __enter__(self) -> ReadTrace:
        self._previous = sys.gettrace()
        sys.settrace(self._call)
        return self

    def __exit__(self, *exc) -> bool:
        sys.settrace(self._previous)
        return False

    def _call(self, frame, event, arg):
        name = frame.f_code.co_filename
        if name not in self._watched:
            path = os.path.realpath(name)
            self._watched[name] = path.startswith((PACKAGE_DIR, PYARROW_DIR))
        return self._line if self._watched[name] else None

    def _line(self, frame, event, arg):
        if event in ("line", "return") and len(self.found) < 20:
            values = [*frame.f_locals.values(), *([arg] if event == "return" else [])]
            try:
                held = any(holds(value, self.hit) for value in values)
            except Exception:  # a value the search cannot read holds nothing it can flag
                held = False
            if held:
                where = f"{Path(frame.f_code.co_filename).name}:{frame.f_lineno}"
                self.found.append(f"{where} {frame.f_code.co_name}")
        return self._line


def unreadable_row_group(path: Path, group: int, keep) -> None:
    """Overwrite the data of row group `group`'s column chunks, those named in `keep` aside, so
    that decoding one fails; the footer (schema, row-group statistics) stays."""
    meta = pq.read_metadata(path).row_group(group)
    data = bytearray(path.read_bytes())
    for j in range(meta.num_columns):
        column = meta.column(j)
        if column.path_in_schema in keep:
            continue
        start = (
            column.dictionary_page_offset if column.has_dictionary_page else column.data_page_offset
        )
        data[start : start + column.total_compressed_size] = b"\xff" * column.total_compressed_size
    path.write_bytes(bytes(data))


VOTES_POISON = 9_090_000  # a holdout row's votes figures start here; no development figure does


def votes_rows_of(race_id: str, k: int, holdout: bool) -> list[dict]:
    """One race's PLACE and WIN rows of pool_votes (six runners and the `__pool__` row, its votes
    holding the refunds too), in votes.py's order. A holdout race's figures are all poison."""
    keys = race_keys(race_id)
    base = {c: keys[c] for c in ("jurisdiction", "race_id", "race_date", "venue", "race_no")}
    rows = []
    for pool in ("PLACE", "WIN"):
        bump = 5 if pool == "WIN" else 0
        votes = {
            h: VOTES_POISON + 100 * k + 10 * h + bump if holdout else 1000 * h + 10 * k + bump
            for h in range(1, 7)
        }
        refund = VOTES_POISON + 99 if holdout else (30 * (k + 1) if pool == "PLACE" else 0)
        rows += [
            base
            | {"pool": pool, "combination": str(h), "votes": n, "refund_votes": None}
            | {"fav_rank": None, "data_kubun": "4"}
            for h, n in votes.items()
        ]
        rows.append(
            base
            | {"pool": pool, "combination": jravan_votes.POOL_ROW}
            | {"votes": sum(votes.values()) + refund, "refund_votes": refund}
            | {"fav_rank": None, "data_kubun": "4"}
        )
    return sorted(rows, key=lambda r: (r["race_id"], r["pool"], r["combination"]))


def test_read_place_votes_decodes_no_holdout_votes(tmp_path, monkeypatch):
    """Fix round 4 (C): read_place_votes decodes no votes column of a holdout row, even when a row
    group mixes development and holdout races, and never decodes a row group holding a holdout
    race alone; what it returns for the development races is their settled PLACE pools."""
    dev = [rid("2026-06-13", "TOK", 1), rid("2026-06-13", "TOK", 2), rid("2026-06-27", "TOK", 1)]
    mixed = [rid("2026-06-20", "TOK", 1), rid("2026-06-20", "TOK", 2)]
    alone = rid("2026-06-21", "TOK", 1)
    holdout_days = {"2026-06-20", "2026-06-21"}
    monkeypatch.setattr(
        holdout_module,
        "_load_definition",
        lambda countries_dir, country: (date(2026, 6, 20), "s", None),
    )
    monkeypatch.setattr(holdout_module, "_is_holdout_day", lambda seed, day: day in holdout_days)

    rows = {race: votes_rows_of(race, k, race in (*mixed, alone)) for k, race in enumerate(
        [*dev, *mixed, alone])}  # fmt: skip
    groups = [  # one mixed row group, one holding the holdout race alone, one development
        [row for race in (dev[0], dev[1], *mixed) for row in rows[race]],
        rows[alone],
        rows[dev[2]],
    ]
    path = tmp_path / jravan_votes.FILENAME
    schema = jravan_votes.POOL_VOTES_SCHEMA
    with pq.ParquetWriter(path, schema) as writer:
        for group in groups:
            writer.write_table(pa.Table.from_pylist(group, schema=schema))
    keys = ("race_id", "race_date", "pool")
    unreadable_row_group(path, 1, keep=keys)
    # the fixture: three row groups, the first mixing the races; the second's keys read, the rest
    # of it cannot be decoded
    assert pq.read_metadata(path).num_row_groups == 3
    assert pq.ParquetFile(path).read_row_group(1, columns=list(keys)).num_rows == len(rows[alone])
    with pytest.raises((OSError, pa.ArrowException)):
        pq.ParquetFile(path).read_row_group(1, columns=["votes"])
    poison = {
        n
        for race in (*mixed, alone)
        for row in rows[race]
        for n in (row["votes"], row["refund_votes"])
        if n is not None
    }
    assert min(poison) >= VOTES_POISON > max(row["votes"] for race in dev for row in rows[race])

    def expected(races) -> dict:
        want = {}
        for race in races:
            place = [r for r in rows[race] if r["pool"] == "PLACE"]
            runners = {
                int(r["combination"]): r["votes"] for r in place if r["combination"].isdigit()
            }
            want[race] = avoidance_price.PlacePool(sum(runners.values()), runners)
        return want

    def hit(x) -> bool:
        return isinstance(x, (int, np.integer)) and not isinstance(x, bool) and int(x) in poison

    got = {}
    with ReadTrace(hit) as trace:
        try:
            for races in (dev, dev[:2], [dev[2]]):
                got[tuple(races)] = avoidance_price.read_place_votes(races, path=path)
        except (OSError, pa.ArrowException) as exc:
            pytest.fail(f"a row group holding a holdout race alone was decoded: {exc!r}")
    assert not trace.found, f"a holdout row's votes were decoded: {trace.found}"
    for races, pools in got.items():
        assert pools == expected(races), races
    with pytest.raises(ValueError):  # a holdout race is never asked of the table
        avoidance_price.read_place_votes([dev[0], mixed[0]], path=path)


def test_a_votes_file_that_cannot_be_opened_leaves_the_prediction_unavailable(study):
    """Fix round 4 (D): the votes file exists and opening it is refused (PermissionError). The run
    is not aborted: every record is written, the comparisons are scored as in A, the realised
    counts stand, the prediction is unavailable and the votes' data build names the failure."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root opens a file whatever its mode")
    a = study("A")
    try:
        pu = study("PU")
    except OSError as exc:
        pytest.fail(f"run_study aborted on a votes file it cannot open: {exc!r}")
    names = plus10_names(a)
    ra, rp = a.by_variant(), pu.by_variant()
    assert ra.keys() == rp.keys()
    assert len(pu.records) == pu.spec["family_size"]
    votes_built = {n for n, r in ra.items() if is_withheld(r["sources"])}
    assert set(names.values()) <= votes_built
    for name, record in ra.items():  # scoring goes on: nothing the votes do not build moves
        if name not in votes_built:
            assert same(record, rp[name]), name

    def votes_builds(record: dict) -> list[str]:
        return [v for k, v in record["data_builds"].items() if "votes" in k.lower()]

    for tag, name in names.items():
        record = rp[name]
        assert record["status"] == "scored", (tag, record["metrics"])
        assert realised_counts(record) == realised_counts(ra[name]), tag
        assert any(value for value, _ in realised_counts(record).values()), tag
        assert cross_tab(record) == {}, (tag, cross_tab(record))  # unavailable: no cell
        (build,) = votes_builds(record)
        assert build not in votes_builds(ra[name]), (tag, build)
        assert re.search(r"(?i)unreadable|unopen|permission|cannot", build), (tag, build)
    assert private_report(pu)


def dropped_races(outcome: Outcome) -> set[tuple[str, str]]:
    """(variant, race) of every dropped race in the public dropped-races CSV."""
    found = [text for path, text, _ in outcome.writes if path.name.endswith("-dropped.csv")]
    assert len(found) == 1, [str(path) for path, _, _ in outcome.writes]
    frame = pd.read_csv(io.StringIO(found[0]))
    return set(zip(frame["variant"], frame["race_id"], strict=True))


def counterpart(forward: dict, record: dict) -> bool:
    """Whether a walk-forward record is `record`'s: the same hypothesis and at least its
    components and params."""
    return (
        forward["hypothesis_id"] == record["hypothesis_id"]
        and all(forward["components"].get(k) == v for k, v in record["components"].items())
        and all(forward["params"].get(k) == v for k, v in record["params"].items())
    )


def fold_differences(record: dict) -> dict[str, list[dict]]:
    """Each window's first day -> the record's metrics labelled with that fold and named with
    'difference', those of no race aside."""
    found: dict[str, list[dict]] = {}
    for m in record["metrics"]:
        day = fold_day(m["fold"])
        if day is not None and "difference" in m["metric"] and m["n"]:
            found.setdefault(day, []).append(m)
    return found


ALL_IDS = (*(race.race_id for race in HISTORY), *LOGGED_IDS)


def test_walk_forward_rows_score_only_the_fold_test_races(study):
    """Fix round 4 (E): beside each whole-development record (#81's scheme, its name), a
    walk-forward record scores only the paired races in the test windows of
    `core.eval.folds.walk_forward` under Japan's scheme (config/folds.yaml), each fold's labelled
    with its fold; the first training span (D1 here) is scored by none. The scheme moves no
    descriptive record."""
    # the fixture: walk_forward over the development race days gives the folds worked by hand
    days = sorted({date.fromisoformat(race.split(":")[1]) for race in ALL_IDS})
    for scheme, want in ((TEST_FOLDS, FOLD_RACES), (LATE_FOLDS, LATE_FOLD_RACES)):
        folds = walk_forward(days, scheme=FoldScheme.from_mapping(scheme))
        assert [fold.test_start.isoformat() for fold in folds] == sorted(want)
        assert all(race.split(":")[1] == day for day, races in want.items() for race in races)
        assert {race for races in want.values() for race in races} == {
            race for race in ALL_IDS if race.split(":")[1] in want
        }
    assert avoidance.SCHEME == DESCRIPTIVE  # #81's records keep their scheme

    a, k = study("A"), study("K")
    for outcome, fold_races in ((a, FOLD_RACES), (k, LATE_FOLD_RACES)):
        records = outcome.records
        schemes = {r["fold_scheme"] for r in records}
        assert all(is_descriptive(r) or is_walk_forward(r) for r in records), schemes
        descriptive = [r for r in records if is_descriptive(r)]
        forward = [r for r in records if is_walk_forward(r)]
        assert forward, schemes
        assert not {r["variant"] for r in descriptive} & {r["variant"] for r in forward}
        plus10 = set(plus10_names(outcome).values())
        dropped = dropped_races(outcome)
        checked = Counter()
        for record in descriptive:
            if record["hypothesis_id"] not in RULES or record["variant"] in plus10:
                continue
            twins = [w for w in forward if counterpart(w, record)]
            assert len(twins) == 1, (record["variant"], [w["variant"] for w in twins])
            if not is_comparison(record):
                continue
            (twin,) = twins
            considered = LOGGED_IDS if is_at_t(record) else ALL_IDS
            paired = {race for race in considered if (record["variant"], race) not in dropped}
            assert paired_means(record)[2] == len(paired), record["variant"]  # every race once
            per_fold = {day: len(paired & set(races)) for day, races in fold_races.items()}
            if not sum(per_fold.values()):
                assert twin["status"] == "failed", twin["variant"]
                continue
            assert twin["status"] == "scored", twin["variant"]
            assert paired_means(twin)[2] == sum(per_fold.values()), twin["variant"]
            differences = fold_differences(twin)
            assert set(differences) <= set(fold_races), (twin["variant"], sorted(differences))
            for day, count in per_fold.items():
                found = [m["n"] for m in differences.get(day, [])]
                assert found == ([count] if count else []), (twin["variant"], day, found)
            checked[(record["hypothesis_id"], "at t" if is_at_t(record) else "final")] += 1
        assert {rule for rule, _ in checked} == set(RULES), checked
        assert {tag for _, tag in checked} == {"final", "at t"}, checked

    rk = k.by_variant()  # another fold scheme: the descriptive records do not move
    for name, record in a.by_variant().items():
        if is_descriptive(record):
            assert same(record, rk[name]), name


def favourite_win_returns(race: Race, board: dict[int, float]) -> tuple[float, float]:
    """L7's handicap rule on one race, from first principles: one unit on the favourite of
    `board` to win, the race skipped (its stake back) when a handicap. (rule, comparator) return
    per unit staked at the race's WIN dividend."""
    favourite = min(board, key=board.get)
    paid = {(r["pool"], r["combination"]): r["dividend"] for r in dividend_rows(race)}
    back = paid.get(("WIN", str(favourite)), 0.0) / 100
    return (1.0 if race.weight_type == "handicap" else back), back


def test_walk_forward_rows_return_their_fold_races_own_returns(study):
    """Fix round 4 (E): a walk-forward record's figures are those of its fold races alone: L7's
    handicap Win rule, worked from the fixture, over the races of the test windows (pooled: each
    side's mean and the paired difference; per fold: the fold's difference), at the final price
    and at t, under two schemes."""
    for scenario, fold_races in (("A", FOLD_RACES), ("K", LATE_FOLD_RACES)):
        seen = Counter()
        for record in study(scenario).records:
            if not (
                is_walk_forward(record)
                and record["hypothesis_id"] == "L7"
                and record["params"].get("pool") == "WIN"
            ):
                continue
            at_t = is_at_t(record)
            folds = {
                day: [
                    favourite_win_returns(RACES[r], RACES[r].at_t if at_t else RACES[r].final)
                    for r in races
                    if not at_t or RACES[r].at_t is not None
                ]
                for day, races in fold_races.items()
            }
            every = [x for returns in folds.values() for x in returns]
            rule, comparator, n = paired_means(record)
            assert n == len(every), (scenario, record["variant"], n)
            want_rule = sum(r for r, _ in every) / n
            want_comparator = sum(c for _, c in every) / n
            assert math.isclose(rule, want_rule, rel_tol=1e-9), (scenario, rule, want_rule)
            assert math.isclose(comparator, want_comparator, rel_tol=1e-9), (scenario, comparator)
            (diff,) = [
                m for m in pooled(record) if "difference" in m["metric"] and m["lo"] is not None
            ]
            assert math.isclose(diff["value"], want_rule - want_comparator, abs_tol=1e-12)
            differences = fold_differences(record)
            for day, returns in folds.items():
                if not returns:
                    continue
                (m,) = differences[day]
                want = sum(r - c for r, c in returns) / len(returns)
                assert m["n"] == len(returns), (scenario, day, m)
                assert math.isclose(m["value"], want, abs_tol=1e-12), (scenario, day, m, want)
            seen["at t" if at_t else "final"] += 1
        assert seen == {"final": 1, "at t": 1}, (scenario, seen)


def summary_text(outcome: Outcome) -> str:
    return outcome.masked(json.dumps(outcome.summary, indent=2, sort_keys=True))


def test_the_cli_summary_tells_no_withheld_result(study):
    """Fix round 4 (F, D21): run_study's returned summary and main's stdout (which prints it) are
    the same, masked of paths, hashes and times, whatever the JRA-VAN votes give: other votes (V),
    a reconstruction miss (M), an unresolved premium (Q0), a votes reader that raises (R)."""
    a = study("A")
    votes_built = {n for n, r in a.by_variant().items() if is_withheld(r["sources"])}
    assert votes_built
    want = summary_text(a)
    for name in ("A", "V", "M", "Q0", "R"):
        outcome = study(name)
        assert json.loads(outcome.stdout) == json.loads(json.dumps(outcome.summary)), name
        if name == "A":
            continue
        # the fixture: the checks built on the votes give other results in each of these runs
        assert set(moved(a, outcome)) & votes_built, name
        got = summary_text(outcome)
        assert got == want, (name, sorted(set(got.splitlines()) ^ set(want.splitlines())))


FAMILY_V5 = "japan-family-l-floor-tail-favourite-v5"


def holds_mapping(value, wanted: dict) -> bool:
    """Whether a mapping holding every item of `wanted` is anywhere in the JSON value."""
    if isinstance(value, dict):
        if all(key in value and value[key] == item for key, item in wanted.items()):
            return True
        value = list(value.values())
    return isinstance(value, list) and any(holds_mapping(v, wanted) for v in value)


def test_the_changed_family_is_frozen_as_v5_before_any_race_is_read(study):
    """Fix round 4 (G): the changed semantics are a new family, v5, superseding v4 (whose records
    stay), frozen before any race is read, its walk-forward scheme frozen with it."""
    a, k = study("A"), study("K")
    assert avoidance_price.FAMILY_ID == FAMILY_V5
    assert a.entries[0]["kind"] == "freeze"  # the store appeared only once it was frozen
    assert a.freezes[0]["family_id"] == FAMILY_V5
    assert {r["family_id"] for r in a.records} == {FAMILY_V5}
    assert "japan-family-l-floor-tail-favourite-v4" in a.spec["supersedes"]
    assert holds_mapping(a.spec, TEST_FOLDS) and not holds_mapping(a.spec, LATE_FOLDS)
    assert holds_mapping(k.spec, LATE_FOLDS)
    assert k.spec_hash != a.spec_hash  # another scheme, another spec


# ---------------------------------------------------------------------------------------------
# Fix round 5 of e487d61: an untraversable votes directory, the votes build's identity across the
# file's availability
# ---------------------------------------------------------------------------------------------

# what each unavailable state's refresh names (D: unreadable; U0: missing; PD: its stat refused)
FAILURE_WORDS = {
    "PU": re.compile(r"(?i)unreadable|unopen|permission|cannot|inaccessible"),
    "PD": re.compile(r"(?i)unreadable|unopen|permission|cannot|inaccessible"),
    "U0": re.compile(r"(?i)missing|absent|not found|no file"),
}


def votes_build(record: dict) -> tuple[str, str, str]:
    """The record's votes data build: (build name, identity, refresh), its value split at the last
    '@' as the registry's `family_size` splits it ('' identity when the value holds no '@')."""
    keys = [k for k in record["data_builds"] if "votes" in k.lower()]
    assert len(keys) == 1, (record["variant"], record["data_builds"])
    identity, _, refresh = record["data_builds"][keys[0]].rpartition("@")
    return keys[0], identity, refresh


def not_root() -> None:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root reads a file and traverses a directory whatever its mode")


def test_an_untraversable_votes_directory_does_not_abort_the_run(study):
    """Fix round 5 (1): the processed jravan directory has no search permission, so even the
    probe of the votes file is refused (PermissionError from stat). The run is not aborted: every
    record is written, every record the votes do not build equals A's, the realised counts stand,
    the prediction is unavailable and the votes' data build names the failure, under the identity
    it has when the file is read."""
    not_root()
    a = study("A")
    try:
        pd_ = study("PD")
    except OSError as exc:
        pytest.fail(f"run_study aborted on a votes directory it cannot traverse: {exc!r}")
    assert pd_.spec_hash == a.spec_hash
    names = plus10_names(a)
    ra, rp = a.by_variant(), pd_.by_variant()
    assert ra.keys() == rp.keys()
    assert len(pd_.records) == pd_.spec["family_size"]
    votes_built = {n for n, r in ra.items() if is_withheld(r["sources"])}
    assert set(names.values()) <= votes_built
    for name, record in ra.items():  # nothing the votes do not build moves, its builds included
        if name not in votes_built:
            assert rp[name] == record, name

    for tag, name in names.items():
        record = rp[name]
        assert record["status"] == "scored", (tag, record["metrics"])
        assert realised_counts(record) == realised_counts(ra[name]), tag
        assert any(value for value, _ in realised_counts(record).values()), tag
        assert cross_tab(record) == {}, (tag, cross_tab(record))  # unavailable: no cell
        key, identity, refresh = votes_build(record)
        assert (key, identity) == votes_build(ra[name])[:2], (tag, record["data_builds"])
        assert refresh != votes_build(ra[name])[2], (tag, refresh)
        assert FAILURE_WORDS["PD"].search(refresh), (tag, refresh)
    assert private_report(pd_)


def rerun_readable(first: Outcome, votes_file: Path, base: Path) -> Outcome:
    """The same frozen family run again against `first`'s registry: on a copy of `first`'s data
    directory (its store and its registry as the first run left them), with `votes_file` copied
    in, readable, as the jravan pool_votes.parquet."""
    data = base / "data"
    shutil.copytree(first.base / "data", data, ignore=shutil.ignore_patterns(jravan_votes.FILENAME))
    shutil.copyfile(votes_file, data / "processed" / "jravan" / jravan_votes.FILENAME)
    return run_scenario(base, Scenario(), owner_config(), again=True)


@pytest.mark.parametrize("first_state", ["PU", "U0"])
def test_the_votes_build_keeps_one_identity_when_the_file_comes_back(study, tmp_path, first_state):
    """Fix round 5 (2): the same frozen family run first with the votes file unreadable (PU) or
    missing (U0), then, against the same registry, with the file readable. The second run appends
    the refreshed checks (A's records) as further runs of the same variants and nothing else: the
    registry's family size stays the frozen spec's. The votes build's identity is one in the three
    states (readable in A, unavailable in the first run, readable in the rerun); only the refresh
    differs: the file's hex digest when it is read, a word naming the failure otherwise."""
    if first_state == "PU":
        not_root()
    a = study("A")
    ra = a.by_variant()
    family = a.freezes[0]["family_id"]
    votes_built = sorted(n for n, r in ra.items() if is_withheld(r["sources"]))
    assert votes_built
    readable = {n: votes_build(ra[n]) for n in votes_built}
    for name, (_, identity, refresh) in readable.items():
        assert identity and HEX_DIGEST.fullmatch(refresh), (name, identity, refresh)

    first = study(first_state)
    assert first.spec_hash == a.spec_hash
    votes_file = a.base / "data" / "processed" / "jravan" / jravan_votes.FILENAME
    second = rerun_readable(first, votes_file, tmp_path)
    assert second.spec_hash == a.spec_hash  # one freeze: the same frozen family
    new = [e["record"] for e in second.entries[len(first.entries) :] if e["kind"] == "run"]
    assert sorted(r["variant"] for r in new) == votes_built  # the rest already present
    assert second.summary["records_appended"] == len(votes_built)
    for record in new:  # the refreshed checks: A's records, prediction and all
        assert record == ra[record["variant"]], record["variant"]

    registry = Registry(second.registry_path, holdout_log=tmp_path / "no-holdout.jsonl")
    assert {r.variant for r in registry.records(family)} == set(ra)
    assert registry.family_size(family) == a.spec["family_size"], first_state

    rf = first.by_variant()
    for name in votes_built:
        key, identity, refresh = votes_build(rf[name])
        assert (key, identity) == readable[name][:2], (name, rf[name]["data_builds"])
        assert refresh and refresh != readable[name][2], (name, refresh)
        assert FAILURE_WORDS[first_state].search(refresh), (name, refresh)
