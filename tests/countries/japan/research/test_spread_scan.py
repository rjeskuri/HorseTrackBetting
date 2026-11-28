"""Ticket #503: scan popularity-rank spreads (box, wheel, formation) by race class on paid
dividends (N8), on synthetic stores only.

Contract pinned here for `horsetrack.countries.japan.research.spread_scan` (the coder implements
what these tests call):

- `SPREADS`, a mapping keyed by exactly the pre-registered spread ids of `SPREADS` below; each id
  names its pool and its legs in Win popularity ranks (`exacta:2-4/1` is ranks 2-4 first, the
  favourite second). `FAMILIES`, a mapping family id -> spread ids, partitions `SPREADS`: the
  registry families. `family_spec(family_id) -> dict`, the plain-JSON spec a family is frozen with.
- `expand(spread_id, ranked, *, brackets=None) -> formations.TicketSet` (#139's type): `ranked` is
  the horse numbers in Win popularity order (rank 1 first); the result is the pool code and the
  distinct combinations a JRA mark card buys, one ticket each (finishing order for FORECAST and
  TIERCE, ascending for the others), so its cost is 100 yen x tickets. `bracket:1/2-3`
  (BRACKET_QUINELLA) takes `brackets` (horse number -> bracket) and buys the favourite's bracket
  with the bracket of rank 2 and with that of rank 3, a same-bracket pair `(b, b)` included,
  each distinct pair once.
- `win_ranks(board, *, at_utc, scratched=()) -> list[int]`: `board` is one race's rows in the odds
  log's schema (`logger.compact.SCHEMA`, `processed/jravan/odds_timeseries`), any pools. Each WIN
  runner's latest row with `snapshot_utc <= at_utc` (at or before: a row at `at_utc` counts, a
  later one never does) gives its price; a runner whose latest such row has no price, or that is
  in `scratched` (the runners table's `is_scratched`), is dropped before ranking. The horse numbers
  in ascending price, ties by the lower horse number (numerically). The `__pool__` row and other
  pools are never read as runners. The scan ranks at `post_utc - 2 min` (D2's primary T-2).
- `settle(tickets, paid) -> Settlement` with `cost` (100 yen per ticket) and `ret` (yen): `paid` is
  one race's `dividends` rows (race_id, pool, combination, dividend per 100 yen, is_refund), any
  pools. A ticket returns the dividend of every paying row of its pool with its combination text
  (`"7,5"` finishing order for FORECAST and TIERCE, `"2,10"` numerically ascending otherwise,
  bracket numbers for BRACKET_QUINELLA), so Wide and dead heats can pay several tickets; a ticket
  with an `is_refund` row returns its 100-yen stake.
- `class_level(race_class) -> str`: DEBUT newcomer, MAIDEN maiden, C1/C2/C3 conditions, OP and L
  open, G3, G2 and G1 as themselves; anything else (None, '', an unknown code) raises ValueError.
- `race_cells(*, race_class, field_size, fav_odds, pool_total, pool_edges, surface)
  -> frozenset[str]`: exactly 'all', `class:<level>`, `field:<band>`, `fav:<band>`,
  `surface:<surface>`, `pool_size:<low|mid|high>`, `class:<level>+fav:<band>` and
  `class:<level>+field:<band>`. The field, favourite and pool-size bands are
  `pool_scan.race_cells`'s (field le4, 5-10, 11-14, 15-18; favourite at the T-2 Win price le1.9,
  2.0-2.9, 3.0-4.9, ge5.0, lower edge inclusive; pool size low below the first edge, mid below
  the second, high from it). An unknown class or a field size outside 1-18 raises ValueError.
- `pool_tertiles(totals) -> (low_edge, high_edge)`: edges under which `race_cells` puts the
  smallest third of the totals in `pool_size:low`, the middle third in `mid` and the largest third
  in `high`.
- `run_scan(*, date_from, date_to, out_dir, seed, n_boot, registry) -> ScanResult`:
  - reads the jravan store (`processed/jravan/`: races, runners, dividends, final_odds and
    `odds_timeseries/<YYYY-MM-DD>.parquet`) on development days only: a holdout day
    (`core.eval.holdout.is_holdout_day`) is neither read nor scored, its day file never opened, and
    nothing before 2014-06-07 is scored (D10: an earlier `date_from` is clamped, not refused);
    `LockedHoldout.open` is never called;
  - prices every spread at the paid dividends, flat 100 yen a ticket, in every race of the span,
    in two variants: 't2' ranks on the T-2 Win board, 'lookahead' on the final Win board (a
    ceiling, not a strategy). Segments are the race's `race_cells` (favourite odds at T-2,
    starters, the race's pool size in tertiles);
  - refuses (a ValueError, nothing scored, no candidate file) when a family is frozen in
    `registry` with a spec other than `family_spec(family_id)`; appends one 'scored' record per
    (family, variant), with `family_id`, `variant` 't2' or 'lookahead', `ticket == 503` and
    `sources == ('jravan',)`;
  - writes through `core.eval.reports.write_report` (sources naming jravan, a withheld source) the
    private report (`ScanResult.report_path`, under `out_dir`) and the candidate cells,
    `ScanResult.cells_path == out_dir / 'candidate-cells' / 'spread-scan.json'`, a strict JSON
    array of `ScanResult.rows`. An `out_dir` inside the repository raises `LicensedOutputInRepo`
    and leaves no file there;
  - skips, in both variants, a race without a T-2 Win board: a day file may hold no WIN row at all
    (the JRA-VAN builder writes a day from either feed alone, e.g. only the O2 QUINELLA rows), and
    the scan completes whatever the year's scratches;
  - bets both variants on the same races: a spread is bought in a race in both variants or in
    neither (both boards rank its highest rank, the race pays its pool), so the t2 and lookahead
    rows of one (spread, cell) hold the same races;
  - cuts the pool-size tertiles per calendar year over the year's races holding at least one bet
    (a race skipped for its segments or without any bet moves no edge);
  - records no 'scored' run when the range holds no bet (a refusal or 'failed' records are both
    allowed) and then leaves an existing candidate-cell file as it was;
  - names the private report per run, so another seed, range or store never overwrites an earlier
    run's report, and every registry record of the run names that report (its file name);
  - keeps that report reference out of what makes a variant: a rerun of the same spec, range, seed
    and replicates at another commit, or on a refreshed store (the same rows in new files), is new
    runs (new run ids, its own report) of the same variants, so `Registry.family_size` of each
    family stays at the two variants; another seed is new variants.
- `horsetrack japan spread-scan run` writes the registry summary into `summary_dir('japan')` and
  keeps in its index every family the index lists, whether or not the local registry holds it.
  It never drops a row an existing family file publishes: when the local registry does not hold
  the runs a family's file lists, that file stays as it was (or the command refuses); when it
  does, the file is rewritten with every run.
- `ScanResult.rows`: one row per (variant, spread, cell with at least one bet), the 't2' rows in
  rank order then the 'lookahead' rows in rank order; `rank` is 1, 2, ... within each variant,
  rows with a `shrunk_edge` first by `shrunk_edge` descending, then the rows without one. A row
  holds the candidate-cell keys of #430 (`study` 'spread-scan', `family_id` (the family holding
  the spread), `cell_id` (unique), `pool` (the spread's), `class_group` (the cell's class level or
  null), `odds_band` (the cell's favourite band or null), `payout_rate`, `n_bets` (races the spread
  was bought in), `n_race_days`, `n_meetings`, `mean_return` (total return / total stake),
  `se_mean_return` (the meeting bootstrap), `shrunk_edge`, `mean_odds`) and `spread_id`,
  `variant`, `segment` (the `race_cells` label), `rank`, `n_tickets`, `ci_lo` and `ci_hi` (the
  meeting-bootstrap interval of `mean_return`) and `p_return_above_100` (P(return > 100 yen)).

The stores are built here. A synthetic meeting is 20 races in which the first two finishers, in T-2
Win rank, run through every ordered pair of ranks 1-5 once, so each pool's dividend can be set in
closed form to 100 x 20 / (the combination's hits in the meeting): every combination then returns
exactly its stake over a meeting, each spread whose combinations all hit returns exactly break-even
(rho 1), and no spread returns more. The planted meetings (maiden, favourite at 1.5) pay the
exacta "rank 4 beats the favourite" at 4,400 instead of 2,000, which puts `exacta:2-4/1` at rho 1.4
on them and nothing else above 1.3. Holdout days are picked with `is_holdout_day`, never assumed.
No real data is read.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party.
from __future__ import annotations  # noqa: I001

import builtins
import itertools
import json
import math
import os
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from horsetrack.core.eval import holdout, reports
from horsetrack.core.eval import registry as registry_module
from horsetrack.core.eval.registry import Registry
from horsetrack.core.eval.reports import LicensedOutputInRepo
from horsetrack.core.store.build import arrow_schema
from horsetrack.countries.japan import takeout
from horsetrack.countries.japan.betting.formations import TicketSet
from horsetrack.countries.japan.jravan.odds import FINAL_ODDS_SCHEMA
from horsetrack.countries.japan.logger.compact import SCHEMA as BOARD_SCHEMA
from horsetrack.countries.japan.research import spread_scan as ss

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_VAR = "HORSETRACK_DATA_DIR"
JST = timezone(timedelta(hours=9))
SEED = 20261127
N_BOOT = 200
STAKE = 100
ORDERED_POOLS = frozenset({"FORECAST", "TIERCE"})
PAID_POOLS = ("FORECAST", "QUINELLA", "QUINELLA_PLACE", "TRIO", "TIERCE", "BRACKET_QUINELLA")
ALL_POOLS = ("WIN", "PLACE", *PAID_POOLS)
FAIR_POOLS = ("QUINELLA", "QUINELLA_PLACE", "BRACKET_QUINELLA")  # break-even in every cell
VARIANTS = ("t2", "lookahead")


def span(a: int, b: int) -> list[int]:
    return list(range(a, b + 1))


# --- the pre-registered spreads: pool and legs in Win popularity ranks ---------------------------

BRACKET = "bracket:1/2-3"
SPREADS: dict[str, tuple[str, list[list[int]] | None]] = {
    # Exacta: the favourite first over ranks 2..k; ranks 2..k over the favourite (2-4 is the
    # "contender over the favourite"); boxes.
    "exacta:1/2-3": ("FORECAST", [[1], span(2, 3)]),
    "exacta:1/2-4": ("FORECAST", [[1], span(2, 4)]),
    "exacta:1/2-5": ("FORECAST", [[1], span(2, 5)]),
    "exacta:2-3/1": ("FORECAST", [span(2, 3), [1]]),
    "exacta:2-4/1": ("FORECAST", [span(2, 4), [1]]),
    "exacta:2-5/1": ("FORECAST", [span(2, 5), [1]]),
    "exacta:box1-3": ("FORECAST", [span(1, 3)] * 2),
    "exacta:box1-4": ("FORECAST", [span(1, 4)] * 2),
    # Quinella and Wide: the favourite with ranks 2..k; box 1-4; box 2-5 (against the favourite).
    "quinella:1/2-3": ("QUINELLA", [[1], span(2, 3)]),
    "quinella:1/2-4": ("QUINELLA", [[1], span(2, 4)]),
    "quinella:1/2-5": ("QUINELLA", [[1], span(2, 5)]),
    "quinella:box1-4": ("QUINELLA", [span(1, 4)] * 2),
    "quinella:box2-5": ("QUINELLA", [span(2, 5)] * 2),
    "wide:1/2-3": ("QUINELLA_PLACE", [[1], span(2, 3)]),
    "wide:1/2-4": ("QUINELLA_PLACE", [[1], span(2, 4)]),
    "wide:1/2-5": ("QUINELLA_PLACE", [[1], span(2, 5)]),
    "wide:box1-4": ("QUINELLA_PLACE", [span(1, 4)] * 2),
    "wide:box2-5": ("QUINELLA_PLACE", [span(2, 5)] * 2),
    # Trio: the favourite as axis with ranks 2-5 and 2-6; boxes; formation 1-2 / 1-4 / 2-6.
    "trio:1/2-5": ("TRIO", [[1], span(2, 5), span(2, 5)]),
    "trio:1/2-6": ("TRIO", [[1], span(2, 6), span(2, 6)]),
    "trio:box1-4": ("TRIO", [span(1, 4)] * 3),
    "trio:box1-5": ("TRIO", [span(1, 5)] * 3),
    "trio:1-2/1-4/2-6": ("TRIO", [span(1, 2), span(1, 4), span(2, 6)]),
    # Trifecta: the favourite first with 2-4 second and third; 1 / 2-4 / 2-7; 2-4 / 1 / 2-7.
    "trifecta:1/2-4/2-4": ("TIERCE", [[1], span(2, 4), span(2, 4)]),
    "trifecta:1/2-4/2-7": ("TIERCE", [[1], span(2, 4), span(2, 7)]),
    "trifecta:2-4/1/2-7": ("TIERCE", [span(2, 4), [1], span(2, 7)]),
    # Bracket quinella: the favourite's bracket with the brackets of ranks 2 and 3.
    BRACKET: ("BRACKET_QUINELLA", None),
}
# Tickets per mark card, counted by hand (trio 1-2/1-4/2-6: 9 sets holding rank 1, 5 more holding
# rank 2 and not rank 1).
N_TICKETS = {
    "exacta:1/2-3": 2,
    "exacta:1/2-4": 3,
    "exacta:1/2-5": 4,
    "exacta:2-3/1": 2,
    "exacta:2-4/1": 3,
    "exacta:2-5/1": 4,
    "exacta:box1-3": 6,
    "exacta:box1-4": 12,
    "quinella:1/2-3": 2,
    "quinella:1/2-4": 3,
    "quinella:1/2-5": 4,
    "quinella:box1-4": 6,
    "quinella:box2-5": 6,
    "wide:1/2-3": 2,
    "wide:1/2-4": 3,
    "wide:1/2-5": 4,
    "wide:box1-4": 6,
    "wide:box2-5": 6,
    "trio:1/2-5": 6,
    "trio:1/2-6": 10,
    "trio:box1-4": 4,
    "trio:box1-5": 10,
    "trio:1-2/1-4/2-6": 14,
    "trifecta:1/2-4/2-4": 6,
    "trifecta:1/2-4/2-7": 15,
    "trifecta:2-4/1/2-7": 15,
}
PLANTED_SPREAD = "exacta:2-4/1"
PLANTED_CELL = "class:maiden+fav:le1.9"


def expected_combos(
    spread_id: str, ranked: Sequence[int], brackets: dict[int, int] | None = None
) -> set[tuple[int, ...]]:
    """The mark card's distinct combinations, from the rank legs (independent of the module)."""
    pool, legs = SPREADS[spread_id]
    if legs is None:
        fav = brackets[ranked[0]]
        return {tuple(sorted((fav, brackets[ranked[r - 1]]))) for r in (2, 3)}
    horses = [[ranked[r - 1] for r in leg] for leg in legs]
    out = set()
    for pick in itertools.product(*horses):
        if len(set(pick)) == len(pick):
            out.add(pick if pool in ORDERED_POOLS else tuple(sorted(pick)))
    return out


def combo_text(pool: str, combo: Sequence[int]) -> str:
    legs = combo if pool in ORDERED_POOLS else sorted(combo)
    return ",".join(str(int(h)) for h in legs)


def jra_brackets(field_size: int) -> dict[int, int]:
    """JRA's 枠番 allocation: one horse per bracket up to 8, then the last brackets take more."""
    if field_size <= 8:
        return {h: h for h in range(1, field_size + 1)}
    base, extra = divmod(field_size, 8)
    out, horse = {}, 1
    for b in range(1, 9):
        for _ in range(base + (1 if b > 8 - extra else 0)):
            out[horse] = b
            horse += 1
    return out


def _check_tables() -> None:
    assert set(N_TICKETS) | {BRACKET} == set(SPREADS)
    ranked = list(range(1, 19))
    for spread_id, n in N_TICKETS.items():
        assert len(expected_combos(spread_id, ranked)) == n, spread_id
    assert jra_brackets(12) == {
        **{h: h for h in range(1, 5)},
        5: 5,
        6: 5,
        7: 6,
        8: 6,
        9: 7,
        10: 7,
        11: 8,
        12: 8,
    }
    assert Counter(jra_brackets(18).values()) == {**{b: 2 for b in range(1, 7)}, 7: 3, 8: 3}


_check_tables()


# --- segments (independent of the module) --------------------------------------------------------

CLASS_LEVEL = {
    "DEBUT": "newcomer",
    "MAIDEN": "maiden",
    "C1": "conditions",
    "C2": "conditions",
    "C3": "conditions",
    "OP": "open",
    "L": "open",
    "G3": "G3",
    "G2": "G2",
    "G1": "G1",
}
CLASS_RAW = {
    "DEBUT": "701",
    "MAIDEN": "703",
    "C1": "005",
    "C2": "010",
    "C3": "016",
    "OP": "999",
    "L": "999",
    "G3": "999",
    "G2": "999",
    "G1": "999",
}


def field_band(n: int) -> str:
    return "le4" if n <= 4 else "5-10" if n <= 10 else "11-14" if n <= 14 else "15-18"


def fav_band(odds: float) -> str:
    if odds < 2.0:
        return "le1.9"
    if odds < 3.0:
        return "2.0-2.9"
    return "3.0-4.9" if odds < 5.0 else "ge5.0"


def expected_cells(race_class, field_size, fav_odds, pool_level, surface) -> frozenset[str]:
    c, f, v = CLASS_LEVEL[race_class], field_band(field_size), fav_band(fav_odds)
    return frozenset(
        {
            "all",
            f"class:{c}",
            f"field:{f}",
            f"fav:{v}",
            f"surface:{surface}",
            f"pool_size:{pool_level}",
            f"class:{c}+fav:{v}",
            f"class:{c}+field:{f}",
        }
    )


def segment_parts(segment: str) -> dict[str, str]:
    if segment == "all":
        return {}
    return dict(part.split(":", 1) for part in segment.split("+"))


# --- spreads: the pre-registered set and their mark cards ----------------------------------------


def test_the_spreads_are_the_preregistered_set_and_the_families_partition_them():
    assert set(ss.SPREADS) == set(SPREADS)
    members = [s for spreads in ss.FAMILIES.values() for s in spreads]
    assert sorted(members) == sorted(SPREADS), "every spread is in exactly one family"
    for family_id in ss.FAMILIES:
        spec = ss.family_spec(family_id)
        assert isinstance(spec, dict) and spec
        assert json.loads(json.dumps(spec)) == spec, "a family spec is plain JSON"
    specs = [json.dumps(ss.family_spec(f), sort_keys=True) for f in ss.FAMILIES]
    assert len(set(specs)) == len(specs), "each family freezes its own spec"


def _rankings() -> list[tuple[int, list[int]]]:
    """Seeded Win orders over fields of 8, 12, 16 and 18: (field size, horse numbers by rank)."""
    rng = np.random.default_rng(SEED)
    out = [(8, list(range(1, 9))), (18, list(range(18, 0, -1)))]
    for n in (8, 12, 16, 18, 12):
        out.append((n, [int(h) for h in rng.permutation(n) + 1]))
    return out


RANKINGS = _rankings()


@pytest.mark.parametrize("spread_id", sorted(N_TICKETS))
def test_each_spread_expands_to_exactly_its_mark_card(spread_id):
    pool = SPREADS[spread_id][0]
    seen = set()
    for _, ranked in RANKINGS:
        tickets = ss.expand(spread_id, ranked)
        assert isinstance(tickets, TicketSet)
        assert tickets.pool == pool
        combos = [tuple(int(h) for h in c) for c in tickets.combinations]
        assert len(combos) == len(set(combos)), "one ticket per distinct combination"
        assert set(combos) == expected_combos(spread_id, ranked)
        assert tickets.n_tickets == N_TICKETS[spread_id]
        assert tickets.cost(unit=STAKE) == STAKE * N_TICKETS[spread_id]
        seen.add(frozenset(combos))
    assert len(seen) > 1, "the horses come from the ranking, not from the rank numbers"


BRACKET_CASES = [
    # (field size, horses by rank, the bracket pairs bought)
    (8, [3, 5, 1, 2, 4, 6, 7, 8], {(3, 5), (1, 3)}),
    (16, [5, 6, 11, 1, 2, 3, 4, 7], {(3, 3), (3, 6)}),  # rank 2 shares the favourite's bracket
    (16, [1, 9, 10, 2, 3, 4, 5, 6], {(1, 5)}),  # ranks 2 and 3 share a bracket: one ticket
    (12, [12, 2, 11, 1, 3, 4, 5, 6], {(2, 8), (8, 8)}),
    (18, [16, 17, 18, 1, 2, 3, 4, 5], {(8, 8)}),  # all three in bracket 8
]


@pytest.mark.parametrize(("field_size", "ranked", "pairs"), BRACKET_CASES)
def test_the_bracket_spread_buys_the_favourites_bracket_with_ranks_two_and_three(
    field_size, ranked, pairs
):
    brackets = jra_brackets(field_size)
    assert expected_combos(BRACKET, ranked, brackets) == pairs
    tickets = ss.expand(BRACKET, ranked, brackets=brackets)
    assert isinstance(tickets, TicketSet)
    assert tickets.pool == "BRACKET_QUINELLA"
    combos = [tuple(int(b) for b in c) for c in tickets.combinations]
    assert len(combos) == len(set(combos))
    assert set(combos) == pairs
    assert tickets.cost(unit=STAKE) == STAKE * len(pairs)


# --- the T-2 Win ranking -------------------------------------------------------------------------

POST = datetime(2030, 1, 5, 6, 30, tzinfo=timezone.utc)
AT = POST - timedelta(minutes=2)


def board_row(race, pool, combination, snap, *, odds=None, investment=None, status=None):
    return {
        "race_id": race,
        "venue": race.split(":")[2],
        "race_no": int(race.split(":")[3]),
        "pool": pool,
        "combination": combination,
        "snapshot_utc": snap,
        "fetched_utc": None,
        "post_utc": POST,
        "minutes_to_post": (POST - snap).total_seconds() / 60,
        "odds": odds,
        "odds_hi": None if pool != "PLACE" or odds is None else odds * 1.5,
        "investment": investment,
        "status_raw": status,
        "is_final_capture": False,
        "dividend": None,
        "is_refund": None,
    }


def win_board(captures: dict[float, dict[int, float | None]]) -> pd.DataFrame:
    """One race's odds-log rows: per capture (minutes before post -> horse -> Win price or None
    for a scratched runner) a WIN row per runner, the WIN `__pool__` row and PLACE rows priced in
    the reverse order (a decoy pool)."""
    race = "jra:2030-01-05:TOK:05"
    rows = []
    last = min(captures)
    for minutes, prices in captures.items():
        snap = POST - timedelta(minutes=minutes)
        for horse, odds in prices.items():
            rows.append(
                board_row(race, "WIN", str(horse), snap, odds=odds, status=None if odds else "取消")
            )
            if odds is not None:
                rows.append(board_row(race, "PLACE", str(horse), snap, odds=round(100 / odds, 1)))
        rows.append(board_row(race, "WIN", "__pool__", snap, investment=3.0e7))
    for row in rows:
        row["is_final_capture"] = row["snapshot_utc"] == POST - timedelta(minutes=last)
    return pa.Table.from_pylist(rows, schema=BOARD_SCHEMA).to_pandas()


RANK_CASES = [
    (
        "the-t2-board-not-the-earlier-or-later-one",
        {
            15: {1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0},
            2: {1: 5.0, 2: 2.2, 3: 9.0, 4: 3.1, 5: 7.5},
            1: {1: 1.8, 2: 9.9, 3: 2.5, 4: 30.0, 5: 4.4},
        },
        (),
        [2, 4, 1, 5, 3],
    ),
    (
        "a-row-after-the-cutoff-never-counts",
        {2.5: {1: 4.0, 2: 2.0, 3: 3.0}, 1.9: {1: 1.5, 2: 5.0, 3: 9.0}},
        (),
        [2, 3, 1],
    ),
    (
        "ties-go-to-the-lower-horse-number",
        {2: {7: 3.0, 3: 3.0, 5: 2.0, 9: 3.0, 1: 10.0, 10: 2.0}},
        (),
        [5, 10, 3, 7, 9, 1],
    ),
    (
        "ties-compare-horse-numbers-not-text",
        {6: {9: 2.0, 10: 9.0, 11: 9.0, 2: 1.5}, 2: {9: 4.0, 10: 4.0, 11: 4.0, 2: 8.0}},
        (),
        [9, 10, 11, 2],
    ),
    (
        "a-runner-scratched-before-the-cutoff-is-dropped",
        {
            15: {1: 2.0, 2: 3.0, 3: 4.0, 4: 1.5},
            5: {4: None},
            2: {1: 2.4, 2: 3.3, 3: 4.1},
        },
        (),
        [1, 2, 3],
    ),
    (
        "an-unpriced-row-at-t2-is-dropped",
        {10: {1: 3.0, 2: 1.4, 3: 5.0, 4: 2.2}, 2: {1: 3.0, 2: None, 3: 5.0, 4: 2.2}},
        (),
        [4, 1, 3],
    ),
    (
        "a-scratched-favourite-makes-rank-two-the-favourite",
        {2: {1: 1.5, 2: 3.0, 3: 4.0, 4: 6.0}},
        (1,),
        [2, 3, 4],
    ),
    (
        "a-scratched-runner-mid-order-is-skipped",
        {2: {1: 1.5, 2: 3.0, 3: 4.0, 4: 6.0}},
        (3,),
        [1, 2, 4],
    ),
    (
        "each-runner-keeps-its-own-latest-row",
        {8: {1: 6.0, 2: 3.0, 3: 2.5, 6: 1.9}, 2: {1: 2.1, 2: 3.4, 3: 2.6}},
        (),
        [6, 1, 3, 2],
    ),
]


@pytest.mark.parametrize(
    ("captures", "scratched", "expected"),
    [case[1:] for case in RANK_CASES],
    ids=[case[0] for case in RANK_CASES],
)
def test_win_ranks_come_from_the_t2_win_board(captures, scratched, expected):
    ranked = ss.win_ranks(win_board(captures), at_utc=AT, scratched=scratched)
    assert [int(h) for h in ranked] == expected


def test_a_row_exactly_at_the_cutoff_counts():
    board = win_board({5: {1: 2.0, 2: 3.0, 3: 4.0}, 2: {1: 6.0, 2: 3.0, 3: 1.7}})
    assert [int(h) for h in ss.win_ranks(board, at_utc=AT)] == [3, 2, 1]
    later = AT - timedelta(seconds=1)  # one second earlier: only the T-5 board is in hand
    assert [int(h) for h in ss.win_ranks(board, at_utc=later)] == [1, 2, 3]


# --- pricing at the paid dividends ---------------------------------------------------------------


def paid(rows) -> pd.DataFrame:
    """One race's `dividends` rows from (pool, combination, dividend[, is_refund])."""
    full = [(*r, False) if len(r) == 3 else tuple(r) for r in rows]
    table = pa.Table.from_pylist(
        [
            {
                "jurisdiction": "jra",
                "race_id": "jra:2030-01-05:TOK:05",
                "pool": p,
                "combination": c,
                "dividend": d,
                "is_refund": refund,
            }
            for p, c, d, refund in full
        ],
        schema=arrow_schema("dividends"),
    )
    return table.to_pandas()


SETTLE_CASES = [
    (
        "exacta-hit-other-pools-ignored",
        TicketSet("FORECAST", ((7, 3), (7, 5), (7, 12))),
        [("WIN", "7", 310.0), ("QUINELLA", "5,7", 990.0), ("FORECAST", "7,5", 2350.0)],
        2350.0,
    ),
    (
        "exacta-is-ordered",
        TicketSet("FORECAST", ((5, 7),)),
        [("FORECAST", "7,5", 2350.0), ("QUINELLA", "5,7", 990.0)],
        0.0,
    ),
    (
        "wide-pays-every-pair-held",
        TicketSet("QUINELLA_PLACE", ((1, 2), (1, 4), (1, 9))),
        [
            ("QUINELLA_PLACE", "1,2", 150.0),
            ("QUINELLA_PLACE", "1,4", 320.0),
            ("QUINELLA_PLACE", "2,4", 410.0),
        ],
        470.0,
    ),
    (
        "dead-heat-pays-both-exactas",
        TicketSet("FORECAST", ((5, 1), (5, 3), (5, 8))),
        [("FORECAST", "5,3", 1200.0), ("FORECAST", "5,8", 900.0)],
        2100.0,
    ),
    (
        "trifecta-is-ordered",
        TicketSet("TIERCE", ((1, 2, 3), (1, 3, 2))),
        [("TIERCE", "1,3,2", 5600.0), ("TRIO", "1,2,3", 800.0)],
        5600.0,
    ),
    (
        "trifecta-miss",
        TicketSet("TIERCE", ((1, 2, 3),)),
        [("TIERCE", "1,3,2", 5600.0), ("TRIO", "1,2,3", 800.0)],
        0.0,
    ),
    (
        "trio-ascending-with-two-digit-numbers",
        TicketSet("TRIO", ((2, 9, 10), (2, 9, 11))),
        [("TRIO", "2,9,10", 4410.0)],
        4410.0,
    ),
    (
        "quinella-ascending-with-two-digit-numbers",
        TicketSet("QUINELLA", ((2, 10), (10, 12))),
        [("QUINELLA", "2,10", 1830.0), ("FORECAST", "10,2", 3900.0)],
        1830.0,
    ),
    (
        "a-refunded-ticket-returns-its-stake",
        TicketSet("FORECAST", ((4, 7), (7, 3), (7, 4))),
        [("FORECAST", "4,7", None, True), ("FORECAST", "7,3", 1500.0)],
        1600.0,
    ),
    (
        "refunds-and-no-hit",
        TicketSet("TRIO", ((1, 4, 6), (1, 4, 8), (1, 6, 8))),
        [("TRIO", "1,4,8", None, True), ("TRIO", "1,6,8", None, True), ("TRIO", "2,3,5", 9990.0)],
        200.0,
    ),
    (
        "bracket-same-bracket-pair",
        TicketSet("BRACKET_QUINELLA", ((3, 3), (3, 6))),
        [("BRACKET_QUINELLA", "3,3", 870.0), ("QUINELLA", "5,6", 2210.0)],
        870.0,
    ),
    (
        "no-row-of-the-pool",
        TicketSet("TRIO", ((1, 2, 3),)),
        [("WIN", "1", 250.0)],
        0.0,
    ),
]


@pytest.mark.parametrize(
    ("tickets", "rows", "ret"),
    [case[1:] for case in SETTLE_CASES],
    ids=[case[0] for case in SETTLE_CASES],
)
def test_settle_returns_the_paid_dividends_of_the_hitting_combinations(tickets, rows, ret):
    settled = ss.settle(tickets, paid(rows))
    assert settled.cost == STAKE * len(tickets.combinations)
    assert float(settled.ret) == pytest.approx(ret, abs=1e-9)


# --- segments ------------------------------------------------------------------------------------


@pytest.mark.parametrize("race_class", sorted(CLASS_LEVEL))
def test_class_levels(race_class):
    assert ss.class_level(race_class) == CLASS_LEVEL[race_class]


@pytest.mark.parametrize("bad", [None, "", "graded", "JG1", "maiden", float("nan")])
def test_an_unknown_class_raises(bad):
    with pytest.raises(ValueError):
        ss.class_level(bad)


def _cell_inputs() -> list[dict]:
    classes = sorted(CLASS_LEVEL)
    fields = [1, 4, 5, 8, 10, 11, 14, 15, 18, 12]
    favs = [1.0, 1.9, 2.0, 2.4, 2.9, 3.0, 4.9, 5.0, 12.5, 1.4, 3.7]
    totals = [1.0e6, 9.9e6, 1.01e7, 2.5e7, 4.99e7, 5.01e7, 2.0e8]
    surfaces = ["turf", "dirt"]
    out = []
    for i in range(40):
        out.append(
            {
                "race_class": classes[i % len(classes)],
                "field_size": fields[(3 * i) % len(fields)],
                "fav_odds": favs[(7 * i) % len(favs)],
                "pool_total": totals[(5 * i) % len(totals)],
                "pool_edges": (1.0e7, 5.0e7),
                "surface": surfaces[(i // 3) % 2],
            }
        )
    return out


def test_race_cells_put_a_race_in_one_level_of_each_segment_and_the_two_crosses():
    seen: dict[str, set] = {}
    for kw in _cell_inputs():
        low, high = kw["pool_edges"]
        total = kw["pool_total"]
        level = "low" if total < low else "mid" if total < high else "high"
        expected = expected_cells(
            kw["race_class"], kw["field_size"], kw["fav_odds"], level, kw["surface"]
        )
        assert ss.race_cells(**kw) == expected, kw
        for cell in expected:
            seen.setdefault(cell.split(":")[0], set()).add(cell)
    # the inputs reach every level of every segment
    assert {c.split(":")[1] for c in seen["class"] if "+" not in c} == set(CLASS_LEVEL.values())
    assert {f"field:{b}" for b in ("le4", "5-10", "11-14", "15-18")} <= seen["field"]
    assert {f"fav:{b}" for b in ("le1.9", "2.0-2.9", "3.0-4.9", "ge5.0")} <= seen["fav"]
    assert {f"pool_size:{b}" for b in ("low", "mid", "high")} <= seen["pool_size"]


@pytest.mark.parametrize(("race_class", "field_size"), [("CX", 8), ("MAIDEN", 0), ("G1", 19)])
def test_race_cells_refuse_an_unknown_class_or_field_size(race_class, field_size):
    with pytest.raises(ValueError):
        ss.race_cells(
            race_class=race_class,
            field_size=field_size,
            fav_odds=2.5,
            pool_total=3.0e7,
            pool_edges=(1.0e7, 5.0e7),
            surface="turf",
        )


@pytest.mark.parametrize("n", [9, 12, 30])
def test_pool_tertiles_split_the_races_into_thirds(n):
    rng = np.random.default_rng([SEED, n])
    totals = [float(x) for x in rng.permutation(np.arange(1, n + 1)) * 1.0e6 + 3.3e5]
    edges = ss.pool_tertiles(totals)
    levels = {}
    for total in totals:
        cells = ss.race_cells(
            race_class="C1",
            field_size=12,
            fav_odds=2.5,
            pool_total=total,
            pool_edges=edges,
            surface="dirt",
        )
        (level,) = [c.split(":")[1] for c in cells if c.startswith("pool_size:")]
        levels[total] = level
    ordered = [levels[t] for t in sorted(totals)]
    third = n // 3
    assert ordered == ["low"] * third + ["mid"] * third + ["high"] * third


# --- synthetic jravan stores ---------------------------------------------------------------------

TOP = (1, 2, 3, 4, 5)  # the ranks that fill the first three places in every synthetic race
UNIT = 20  # races in a meeting: every ordered pair of TOP first and second once
PAIRS = list(itertools.permutations(TOP, 2))
ODDS = {
    "A": (
        1.5,
        4.0,
        6.0,
        9.0,
        12.0,
        16.0,
        22.0,
        30.0,
        45.0,
        60.0,
        80.0,
        99.0,
        120.0,
        150.0,
        200.0,
        250.0,
        300.0,
        400.0,
    ),
    "B": (
        3.5,
        4.2,
        5.0,
        7.0,
        9.5,
        13.0,
        18.0,
        25.0,
        33.0,
        48.0,
        66.0,
        90.0,
        110.0,
        140.0,
        180.0,
        220.0,
        280.0,
        350.0,
    ),
}
POOL_LEVELS = {"low": 2.0e7, "mid": 5.0e7, "high": 1.2e8}  # the Win pool at T-2
POOL_SHARE = {
    "WIN": 1.0,
    "PLACE": 0.6,
    "BRACKET_QUINELLA": 0.2,
    "QUINELLA": 1.5,
    "QUINELLA_PLACE": 0.7,
    "FORECAST": 1.2,
    "TRIO": 2.0,
    "TIERCE": 3.5,
}
PLANTED_DIVIDEND = 4400.0  # the exacta rank 4 over the favourite on planted meetings (fair 2,000)
HUGE = 500.0  # every dividend of a holdout or pre-era meeting is multiplied by this


def _thirds() -> dict[tuple[int, int], int]:
    """A third-placed rank for each (first, second), chosen so every three-set of TOP finishes in
    the first three at least once a meeting (trio boxes inside TOP are then break-even)."""
    covered: Counter = Counter()
    out = {}
    for a, b in PAIRS:
        rest = [c for c in TOP if c not in (a, b)]
        c = min(rest, key=lambda c: (covered[frozenset((a, b, c))], c))
        covered[frozenset((a, b, c))] += 1
        out[(a, b)] = c
    assert len(covered) == math.comb(len(TOP), 3)
    return out


THIRD = _thirds()


def _rank_dividends() -> dict[str, dict[tuple[int, ...], float]]:
    """Per pool, the dividend (per 100 yen) of each rank combination that pays in a meeting:
    100 x UNIT / its hits, so it returns exactly its stake over the meeting."""
    hits: dict[str, Counter] = {pool: Counter() for pool in PAID_POOLS}
    for a, b in PAIRS:
        c = THIRD[(a, b)]
        hits["FORECAST"][(a, b)] += 1
        hits["QUINELLA"][tuple(sorted((a, b)))] += 1
        hits["BRACKET_QUINELLA"][tuple(sorted((a, b)))] += 1
        for pair in itertools.combinations(sorted((a, b, c)), 2):
            hits["QUINELLA_PLACE"][pair] += 1
        hits["TRIO"][tuple(sorted((a, b, c)))] += 1
        hits["TIERCE"][(a, b, c)] += 1
    return {
        pool: {combo: STAKE * UNIT / n for combo, n in counts.items()}
        for pool, counts in hits.items()
    }


RANK_DIVIDENDS = _rank_dividends()


@dataclass(frozen=True)
class Meeting:
    day: date
    venue: str
    race_class: str
    fav: str  # 'A' (favourite at 1.5: le1.9) or 'B' (at 3.5: 3.0-4.9)
    field: int
    surface: str
    pool_level: str
    planted: bool = False
    huge: bool = False

    @property
    def meeting_id(self) -> str:
        return f"jra:{self.day.isoformat()}:{self.venue}"


@dataclass
class RaceFacts:
    race_id: str
    meeting: Meeting
    ranked: dict[str, list[int]]  # variant -> horse numbers by rank
    brackets: dict[int, int]
    paid: list[tuple[str, str, float]] = field(default_factory=list)

    @property
    def cells(self) -> frozenset[str]:
        m = self.meeting
        return expected_cells(m.race_class, m.field, ODDS[m.fav][0], m.pool_level, m.surface)


def _swap(ranked: list[int], r: int, s: int) -> list[int]:
    out = list(ranked)
    out[r - 1], out[s - 1] = out[s - 1], out[r - 1]
    return out


def _utc(stamp: datetime) -> datetime:
    return stamp.astimezone(timezone.utc)


def write_store(data: Path, meetings: Sequence[Meeting]) -> list[RaceFacts]:
    """`processed/jravan/`: races, meetings, runners, dividends, final_odds and one odds_timeseries
    day file per race day, for `meetings`; returns each race's facts for the reference.

    Per race the T-2 Win order is a seeded permutation (ranks 1-5 in distinct brackets); the board
    at T-10 has ranks 3 and 6 swapped, the T-1 board (the final one: final_odds, runners.win_odds
    and win_fav_rank agree with it) ranks 4 and 5 swapped; the favourite and its price never move.
    """
    folder = data / "processed" / "jravan"
    (folder / "odds_timeseries").mkdir(parents=True, exist_ok=True)
    facts: list[RaceFacts] = []
    races, runners, dividends, finals = [], [], [], []
    boards: dict[date, list[dict]] = {}
    meeting_rows = {}
    for mi, m in enumerate(meetings):
        brackets = jra_brackets(m.field)
        odds = ODDS[m.fav][: m.field]
        meeting_rows[m.meeting_id] = {
            "jurisdiction": "jra",
            "meeting_id": m.meeting_id,
            "race_date": m.day,
            "venue": m.venue,
            "season": str(m.day.year),
            "status": "run",
        }
        for k, (a, b) in enumerate(PAIRS):
            race_no = k + 1
            rid = f"jra:{m.day.isoformat()}:{m.venue}:{race_no:02d}"
            rng = np.random.default_rng([SEED, mi, k])
            while True:
                t2 = [int(h) for h in rng.permutation(m.field) + 1]
                if len({brackets[h] for h in t2[: len(TOP)]}) == len(TOP):
                    break
            final = _swap(t2, 4, 5)
            early = _swap(t2, 3, 6)
            post_local = datetime(m.day.year, m.day.month, m.day.day, 10, 0, tzinfo=JST)
            post_local += timedelta(minutes=15 * k)
            post = _utc(post_local)
            c = THIRD[(a, b)]
            order_ranks = [a, b, c] + [r for r in range(1, m.field + 1) if r not in (a, b, c)]
            finish = {t2[r - 1]: pos for pos, r in enumerate(order_ranks, 1)}
            h1, h2, h3 = (t2[r - 1] for r in (a, b, c))
            fact = RaceFacts(rid, m, {"t2": t2, "lookahead": final}, brackets)
            scale = HUGE if m.huge else 1.0

            def pay(pool, combination, dividend, fact=fact, scale=scale):
                fact.paid.append((pool, combination, dividend * scale))

            fd = RANK_DIVIDENDS["FORECAST"][(a, b)]
            if m.planted and (a, b) == (4, 1):
                fd = PLANTED_DIVIDEND
            pay("WIN", str(h1), 100.0 * odds[a - 1])
            for h in (h1, h2, h3):
                pay("PLACE", str(h), 150.0)
            pay("FORECAST", f"{h1},{h2}", fd)
            pay(
                "QUINELLA",
                combo_text("QUINELLA", (h1, h2)),
                RANK_DIVIDENDS["QUINELLA"][tuple(sorted((a, b)))],
            )
            for (x, hx), (y, hy) in itertools.combinations(sorted([(a, h1), (b, h2), (c, h3)]), 2):
                pay(
                    "QUINELLA_PLACE",
                    combo_text("QUINELLA_PLACE", (hx, hy)),
                    RANK_DIVIDENDS["QUINELLA_PLACE"][(x, y)],
                )
            pay(
                "TRIO",
                combo_text("TRIO", (h1, h2, h3)),
                RANK_DIVIDENDS["TRIO"][tuple(sorted((a, b, c)))],
            )
            pay("TIERCE", f"{h1},{h2},{h3}", RANK_DIVIDENDS["TIERCE"][(a, b, c)])
            pay(
                "BRACKET_QUINELLA",
                combo_text("BRACKET_QUINELLA", (brackets[h1], brackets[h2])),
                RANK_DIVIDENDS["BRACKET_QUINELLA"][tuple(sorted((a, b)))],
            )
            facts.append(fact)
            for pool, combination, dividend in fact.paid:
                dividends.append(
                    {
                        "jurisdiction": "jra",
                        "race_id": rid,
                        "pool": pool,
                        "combination": combination,
                        "dividend": dividend,
                        "is_refund": False,
                    }
                )

            races.append(
                {
                    "jurisdiction": "jra",
                    "race_id": rid,
                    "meeting_id": m.meeting_id,
                    "race_date": m.day,
                    "venue": m.venue,
                    "race_no": race_no,
                    "season": str(m.day.year),
                    "race_name": f"synthetic {m.race_class}",
                    "race_class": m.race_class,
                    "distance_m": 1600,
                    "surface": m.surface,
                    "post_time_local": post_local.strftime("%H:%M"),
                    "field_size": m.field,
                    "status": "run",
                    "weight_type": "age",
                    "race_class_raw": CLASS_RAW[m.race_class],
                }
            )
            final_rank = {h: r for r, h in enumerate(final, 1)}
            for h in range(1, m.field + 1):
                runners.append(
                    {
                        "jurisdiction": "jra",
                        "race_id": rid,
                        "meeting_id": m.meeting_id,
                        "race_date": m.day,
                        "venue": m.venue,
                        "race_no": race_no,
                        "season": str(m.day.year),
                        "horse_id": f"h{mi:02d}{k:02d}{h:02d}",
                        "horse_no": h,
                        "horse_name": f"Synthetic {h}",
                        "draw": h,
                        "finish_pos": finish[h],
                        "placing_raw": str(finish[h]),
                        "dead_heat": False,
                        "win_odds": odds[final_rank[h] - 1],
                        "is_scratched": False,
                        "bracket": brackets[h],
                        "win_fav_rank": final_rank[h],
                    }
                )
            published = post + timedelta(minutes=3)
            base = {
                "jurisdiction": "jra",
                "race_id": rid,
                "race_date": m.day,
                "venue": m.venue,
                "race_no": race_no,
                "odds_hi": None,
                "status_raw": None,
                "published_utc": published,
                "data_kubun": "4",
            }
            for h in range(1, m.field + 1):
                finals.append(
                    {
                        **base,
                        "pool": "WIN",
                        "combination": str(h),
                        "odds": odds[final_rank[h] - 1],
                        "fav_rank": final_rank[h],
                        "investment": None,
                    }
                )
            for pool in ALL_POOLS:
                finals.append(
                    {
                        **base,
                        "pool": pool,
                        "combination": "__pool__",
                        "odds": None,
                        "fav_rank": None,
                        "investment": POOL_LEVELS[m.pool_level] * 1.05 * POOL_SHARE[pool],
                    }
                )

            day_rows = boards.setdefault(m.day, [])
            for minutes, ranked, grow in ((10, early, 0.7), (2, t2, 1.0), (1, final, 1.05)):
                snap = post - timedelta(minutes=minutes)
                for r, h in enumerate(ranked, 1):
                    for pool, price in (
                        ("WIN", odds[r - 1]),
                        ("PLACE", round(100 / odds[r - 1], 1)),
                    ):
                        day_rows.append(
                            {
                                "race_id": rid,
                                "venue": m.venue,
                                "race_no": race_no,
                                "pool": pool,
                                "combination": str(h),
                                "snapshot_utc": snap,
                                "fetched_utc": None,
                                "post_utc": post,
                                "minutes_to_post": float(minutes),
                                "odds": price,
                                "odds_hi": price * 1.4 if pool == "PLACE" else None,
                                "investment": None,
                                "status_raw": None,
                                "is_final_capture": minutes == 1,
                                "dividend": None,
                                "is_refund": None,
                            }
                        )
                day_rows.append(
                    {
                        "race_id": rid,
                        "venue": m.venue,
                        "race_no": race_no,
                        "pool": "WIN",
                        "combination": "__pool__",
                        "snapshot_utc": snap,
                        "fetched_utc": None,
                        "post_utc": post,
                        "minutes_to_post": float(minutes),
                        "odds": None,
                        "odds_hi": None,
                        "investment": POOL_LEVELS[m.pool_level] * grow,
                        "status_raw": None,
                        "is_final_capture": minutes == 1,
                        "dividend": None,
                        "is_refund": None,
                    }
                )

    def write(name, rows, schema, key):
        rows = sorted(rows, key=key)
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), folder / f"{name}.parquet")

    write("races", races, arrow_schema("races"), lambda r: r["race_id"])
    write("runners", runners, arrow_schema("runners"), lambda r: (r["race_id"], r["horse_no"]))
    write("dividends", dividends, arrow_schema("dividends"), lambda r: r["race_id"])
    write(
        "meetings",
        list(meeting_rows.values()),
        arrow_schema("meetings"),
        lambda r: (r["race_date"], r["venue"]),
    )
    write(
        "final_odds",
        finals,
        FINAL_ODDS_SCHEMA,
        lambda r: (r["race_id"], r["pool"], r["combination"]),
    )
    for day, rows in boards.items():
        rows.sort(key=lambda r: (r["race_id"], r["pool"], r["combination"], r["snapshot_utc"]))
        pq.write_table(
            pa.Table.from_pylist(rows, schema=BOARD_SCHEMA),
            folder / "odds_timeseries" / f"{day.isoformat()}.parquet",
        )
    return facts


def reference(facts: Sequence[RaceFacts]) -> dict[tuple[str, str, str], dict]:
    """(variant, spread, cell) -> stake, return, bets, tickets, days and meetings, computed from
    the facts the store was written from (independent of the module)."""
    acc: dict[tuple[str, str, str], dict] = {}
    for race in facts:
        for variant, ranked in race.ranked.items():
            for spread_id, (pool, _) in SPREADS.items():
                tickets = {
                    combo_text(pool, c) for c in expected_combos(spread_id, ranked, race.brackets)
                }
                ret = sum(d for p, text, d in race.paid if p == pool and text in tickets)
                for cell in race.cells:
                    a = acc.setdefault(
                        (variant, spread_id, cell),
                        {
                            "cost": 0,
                            "ret": 0.0,
                            "n_bets": 0,
                            "n_tickets": 0,
                            "days": set(),
                            "meetings": set(),
                        },
                    )
                    a["cost"] += STAKE * len(tickets)
                    a["ret"] += ret
                    a["n_bets"] += 1
                    a["n_tickets"] += len(tickets)
                    a["days"].add(race.meeting.day)
                    a["meetings"].add(race.meeting.meeting_id)
    for a in acc.values():
        a["rho"] = a["ret"] / a["cost"]
    return acc


def boost_free(day: date, venues: Sequence[str]) -> bool:
    calendar = takeout.shipped_boosts()
    return not any(
        calendar.matching(day, venue, n, pool)
        for venue in venues
        for n in range(1, UNIT + 1)
        for pool in ALL_POOLS
    )


def boost_free_days(start: date, count: int, venues: Sequence[str]) -> list[date]:
    days, day = [], start
    while len(days) < count:
        if boost_free(day, venues):
            days.append(day)
        day += timedelta(days=7)
    return days


# The planted store: three days of 2019, a TOK and a KYO meeting each. P1 and P2 are the planted
# maiden meetings with a strong favourite; every other cell holding them also holds at least as
# many fair races (D1-D4), so the planted cross cell is the one cell at rho 1.4.
def planted_meetings() -> list[Meeting]:
    d0, d1, d2 = boost_free_days(date(2019, 2, 2), 3, ("TOK", "KYO"))
    return [
        Meeting(d0, "TOK", "MAIDEN", "A", 8, "turf", "low", planted=True),  # P1
        Meeting(d1, "KYO", "MAIDEN", "A", 12, "dirt", "mid", planted=True),  # P2
        Meeting(d0, "KYO", "MAIDEN", "B", 8, "turf", "low"),  # D1
        Meeting(d1, "TOK", "MAIDEN", "B", 12, "dirt", "mid"),  # D2
        Meeting(d2, "TOK", "C1", "A", 8, "dirt", "high"),  # D3
        Meeting(d2, "KYO", "C1", "A", 12, "turf", "high"),  # D4
    ]


# --- guards and spies ----------------------------------------------------------------------------


class Forbidden(BaseException):
    """A guarded call was made. A BaseException, so that no `except Exception` hides it."""


def temp_registry(root: Path) -> Registry:
    return Registry(root / "registry" / "japan.jsonl", holdout_log=root / "holdout_log.jsonl")


def frozen_registry(root: Path) -> Registry:
    registry = temp_registry(root)
    for family_id in ss.FAMILIES:
        registry.freeze_family(family_id, ss.family_spec(family_id))
    return registry


def registry_lines(registry: Registry) -> list[dict]:
    if not registry.path.exists():
        return []
    return [json.loads(line) for line in registry.path.read_text().splitlines() if line]


def guard(mp: pytest.MonkeyPatch) -> dict[str, list]:
    """Never open the holdout; keep `write_summary` out of the repository; record every
    `write_report` call (path, sources) and call through."""
    calls: dict[str, list] = {"write_report": [], "holdout_open": []}

    def no_open(*args, **kwargs):
        calls["holdout_open"].append(args)
        raise Forbidden("LockedHoldout.open was called: the scan reads development days only")

    mp.setattr(holdout.LockedHoldout, "open", no_open)
    real_report = reports.write_report

    def spy_report(path, text, *, sources, repo_root=None):
        calls["write_report"].append((Path(path), tuple(sources)))
        return real_report(path, text, sources=sources, repo_root=repo_root)

    mp.setattr(reports, "write_report", spy_report)
    mp.setattr(ss, "write_report", spy_report, raising=False)
    real_summary = registry_module.write_summary

    def kept_out(registry, directory):
        if not reports.is_inside(Path(directory), REPO_ROOT):
            real_summary(registry, directory)

    mp.setattr(registry_module, "write_summary", kept_out)
    mp.setattr(ss, "write_summary", kept_out, raising=False)
    return calls


def run(mp, data: Path, out_dir: Path, registry: Registry, *, date_from=None, date_to=None):
    mp.setenv(DATA_VAR, str(data))
    return ss.run_scan(
        date_from=date_from or date(2014, 6, 7),
        date_to=date_to or date(2019, 12, 31),
        out_dir=out_dir,
        seed=SEED,
        n_boot=N_BOOT,
        registry=registry,
    )


@dataclass
class PlantedRun:
    result: object
    facts: list[RaceFacts]
    ref: dict
    registry: Registry
    out_dir: Path
    calls: dict


@pytest.fixture(scope="module")
def planted(tmp_path_factory) -> PlantedRun:
    root = tmp_path_factory.mktemp("planted").resolve()
    data = root / "data"
    facts = write_store(data, planted_meetings())
    ref = reference(facts)
    # the store is what the module docstring says: the planted cell is the one top t2 cell
    t2 = sorted(((a["rho"], s, c) for (v, s, c), a in ref.items() if v == "t2"), reverse=True)
    assert (t2[0][1], t2[0][2]) == (PLANTED_SPREAD, PLANTED_CELL)
    assert t2[0][0] == pytest.approx(1.4) and t2[1][0] <= 1.3 + 1e-9
    assert all(
        a["rho"] == pytest.approx(1.0)
        for (v, s, c), a in ref.items()
        if v == "t2" and SPREADS[s][0] in FAIR_POOLS
    )
    registry = frozen_registry(root)
    out_dir = data / "reports"
    with pytest.MonkeyPatch.context() as mp:
        calls = guard(mp)
        result = run(mp, data, out_dir, registry)
    return PlantedRun(result, facts, ref, registry, out_dir, calls)


def rows_by_key(rows) -> dict[tuple[str, str, str], dict]:
    out = {}
    for row in rows:
        key = (row["variant"], row["spread_id"], row["segment"])
        assert key not in out, f"{key} is two rows"
        out[key] = row
    return out


# --- the scan on the planted store ---------------------------------------------------------------


def test_every_cell_matches_the_independent_reference(planted):
    rows = rows_by_key(planted.result.rows)
    assert set(rows) == set(planted.ref), "one row per (variant, spread, cell with bets)"
    for key, row in rows.items():
        a = planted.ref[key]
        assert row["pool"] == SPREADS[key[1]][0], key
        assert row["n_bets"] == a["n_bets"], key
        assert row["n_tickets"] == a["n_tickets"], key
        assert row["n_race_days"] == len(a["days"]), key
        assert row["n_meetings"] == len(a["meetings"]), key
        assert row["mean_return"] == pytest.approx(a["rho"], rel=1e-9), key


def test_the_planted_cell_ranks_first(planted):
    t2 = [r for r in planted.result.rows if r["variant"] == "t2"]
    top = t2[0]
    assert (top["spread_id"], top["segment"]) == (PLANTED_SPREAD, PLANTED_CELL)
    assert top["rank"] == 1
    assert top["class_group"] == "maiden" and top["odds_band"] == "le1.9"
    assert top["mean_return"] == pytest.approx(1.4, rel=1e-9)
    assert top["ci_lo"] > 1.0, "every planted meeting returns 1.4"
    assert top["p_return_above_100"] > 0.5
    assert (
        top["shrunk_edge"] is not None and 0 < top["shrunk_edge"] <= top["mean_return"] - 1 + 1e-9
    )
    assert all(r["shrunk_edge"] is None or r["shrunk_edge"] <= top["shrunk_edge"] for r in t2)


def test_a_fair_pools_cells_cover_break_even(planted):
    fair = [r for r in planted.result.rows if r["variant"] == "t2" and r["pool"] in FAIR_POOLS]
    assert {r["pool"] for r in fair} == set(FAIR_POOLS)
    for row in fair:
        assert row["mean_return"] == pytest.approx(1.0, rel=1e-9), row["cell_id"]
        assert row["ci_lo"] <= 1.0 + 1e-9 and row["ci_hi"] >= 1.0 - 1e-9, row["cell_id"]
        assert row["rank"] > 1
    # A cell that mixes planted (1.4) and fair (1.0) meetings of equal stake: every meeting
    # resample lies between the two, and the bootstrap is not degenerate.
    mixed = rows_by_key(planted.result.rows)[("t2", PLANTED_SPREAD, "class:maiden")]
    assert mixed["mean_return"] == pytest.approx(1.2, rel=1e-9)
    assert 1.0 - 1e-9 <= mixed["ci_lo"] < mixed["ci_hi"] <= 1.4 + 1e-9
    assert mixed["se_mean_return"] > 0


def test_the_lookahead_variant_is_labelled_and_ranks_on_the_final_board(planted):
    rows = planted.result.rows
    assert {r["variant"] for r in rows} == set(VARIANTS)
    by = rows_by_key(rows)
    pairs = {(s, c) for v, s, c in by}
    assert all((v, s, c) in by for v in VARIANTS for s, c in pairs)
    # rank 4 and 5 swap on the final board: the look-ahead exacta 2-4 over the favourite no longer
    # holds the planted combination, and 2-5 over the favourite still does
    assert by[("t2", PLANTED_SPREAD, PLANTED_CELL)]["mean_return"] == pytest.approx(1.4)
    assert by[("lookahead", PLANTED_SPREAD, PLANTED_CELL)]["mean_return"] == pytest.approx(1.0)
    assert by[("lookahead", "exacta:2-5/1", PLANTED_CELL)]["mean_return"] == pytest.approx(1.3)
    # t2 rows first, then the look-ahead rows, each block ranked 1, 2, ... by shrunk_edge
    variants = [r["variant"] for r in rows]
    assert variants == sorted(variants, key=VARIANTS.index)
    for variant in VARIANTS:
        block = [r for r in rows if r["variant"] == variant]
        assert [r["rank"] for r in block] == list(range(1, len(block) + 1))
        edges = [r["shrunk_edge"] for r in block]
        known = [e for e in edges if e is not None]
        assert edges[: len(known)] == known, "rows with a shrunk_edge come first"
        assert known == sorted(known, reverse=True)


SCHEMA_KEYS = {
    "study": (str,),
    "family_id": (str,),
    "cell_id": (str,),
    "pool": (str,),
    "class_group": (str, type(None)),
    "odds_band": (str, type(None)),
    "payout_rate": (float,),
    "n_bets": (int,),
    "n_race_days": (int,),
    "n_meetings": (int,),
    "mean_return": (float,),
    "se_mean_return": (float,),
    "shrunk_edge": (float, type(None)),
    "mean_odds": (float,),
    "spread_id": (str,),
    "variant": (str,),
    "segment": (str,),
    "rank": (int,),
    "n_tickets": (int,),
    "ci_lo": (float,),
    "ci_hi": (float,),
    "p_return_above_100": (float,),
}


def _strict(text: str):
    def refuse(constant):
        raise ValueError(f"{constant} is not JSON")

    return json.loads(text, parse_constant=refuse)


def test_candidate_cells_follow_the_schema(planted):
    result = planted.result
    assert Path(result.cells_path) == planted.out_dir / "candidate-cells" / "spread-scan.json"
    rows = _strict(Path(result.cells_path).read_text(encoding="utf-8"))
    assert isinstance(rows, list) and rows
    assert rows == result.rows
    assert len({r["cell_id"] for r in rows}) == len(rows)
    for row in rows:
        for key, types in SCHEMA_KEYS.items():
            assert key in row, key
            value = row[key]
            assert isinstance(value, types) and not isinstance(value, bool), (key, value)
            if isinstance(value, float):
                assert math.isfinite(value), (key, value)
        assert row["study"] == "spread-scan"
        assert row["spread_id"] in ss.FAMILIES[row["family_id"]]
        assert row["pool"] == SPREADS[row["spread_id"]][0]
        parts = segment_parts(row["segment"])
        assert row["class_group"] == parts.get("class")
        assert row["odds_band"] == parts.get("fav")
        assert 0 < row["payout_rate"] <= 1
        assert min(row["n_bets"], row["n_race_days"], row["n_meetings"], row["n_tickets"]) >= 1
        assert row["se_mean_return"] >= 0
        assert row["ci_lo"] <= row["ci_hi"]
        assert 0.0 <= row["p_return_above_100"] <= 1.0


def test_one_registry_record_per_family_and_variant(planted):
    registry = planted.registry
    registry.verify()
    records = registry.records()
    seen = Counter((r.family_id, r.variant) for r in records)
    assert set(seen) == {(f, v) for f in ss.FAMILIES for v in VARIANTS}
    assert all(n == 1 for n in seen.values())
    for record in records:
        assert record.status == "scored"
        assert record.ticket == 503
        assert tuple(record.sources) == ("jravan",)
        assert record.holdout_open is None
    for family_id in ss.FAMILIES:
        assert registry.frozen_spec_hash(family_id) is not None


def test_outputs_are_written_through_write_report_outside_the_repository(planted):
    calls = planted.calls["write_report"]
    assert planted.calls["holdout_open"] == []
    written = {path.resolve() for path, _ in calls}
    assert Path(planted.result.cells_path).resolve() in written
    report = Path(planted.result.report_path).resolve()
    assert report in written and report.is_file()
    for path, sources in calls:
        assert reports.is_withheld(sources) and "jravan" in sources
        assert reports.is_inside(path, planted.out_dir)
        assert not reports.is_inside(path, REPO_ROOT)


# --- holdout days, the price era and the guards ---------------------------------------------------


def _late_days() -> tuple[date, date]:
    """The first development day and the first holdout day from 2026-11-28 (asked, not assumed)."""
    days = [date(2026, 11, 28) + timedelta(days=k) for k in range(30)]
    late_dev = next(d for d in days if not holdout.is_holdout_day("japan", d))
    held = next(d for d in days if holdout.is_holdout_day("japan", d))
    return late_dev, held


def holdout_meetings(with_excluded: bool) -> tuple[list[Meeting], date]:
    late_dev, held = _late_days()
    (mid,) = boost_free_days(date(2019, 3, 2), 1, ("TOK",))
    scored = [
        Meeting(date(2014, 6, 7), "TOK", "OP", "B", 10, "turf", "low"),
        Meeting(mid, "TOK", "C1", "A", 12, "dirt", "mid"),
        Meeting(late_dev, "KYO", "G3", "B", 16, "turf", "high"),
    ]
    excluded = [
        Meeting(held, "TOK", "G1", "A", 8, "dirt", "high", huge=True),  # a holdout day (D14)
        Meeting(date(2014, 6, 1), "NAK", "G2", "A", 8, "turf", "low", huge=True),  # before D10
    ]
    return scored + (excluded if with_excluded else []), held


def test_holdout_days_are_never_read_and_pre_era_days_never_scored(tmp_path, monkeypatch):
    with_meetings, held = holdout_meetings(True)
    without_meetings, _ = holdout_meetings(False)
    data_a, data_b = tmp_path / "a", tmp_path / "b"
    write_store(data_a, with_meetings)
    write_store(data_b, without_meetings)
    held_file = data_a / "processed" / "jravan" / "odds_timeseries" / f"{held.isoformat()}.parquet"
    assert held_file.is_file()

    opened: list[str] = []
    for module, name in (
        (pq, "read_table"),
        (pq, "read_metadata"),
        (pq, "read_schema"),
        (ds, "dataset"),
        (pa, "memory_map"),
        (pd, "read_parquet"),
        (builtins, "open"),
        (ss, "read_table"),
        (ss, "dataset"),
        (ss, "memory_map"),
        (ss, "read_parquet"),
    ):
        real = getattr(module, name, None)
        if real is None:
            continue

        def spy(*args, _real=real, _name=name, **kwargs):
            values = [*args, *kwargs.values()]
            values += [v for x in values if isinstance(x, (list, tuple)) for v in x]
            for value in values:
                if isinstance(value, (str, Path)) and held.isoformat() in str(value):
                    opened.append(f"{_name}({value})")
                    raise Forbidden(f"{_name} opened {value}: a holdout day file")
            return _real(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)

    guard(monkeypatch)
    late_dev, _ = _late_days()
    end = held + timedelta(days=3)
    result_a = run(
        monkeypatch,
        data_a,
        tmp_path / "out-a",
        frozen_registry(tmp_path / "ra"),
        date_from=date(2014, 1, 1),
        date_to=end,
    )
    assert opened == []
    result_b = run(
        monkeypatch,
        data_b,
        tmp_path / "out-b",
        frozen_registry(tmp_path / "rb"),
        date_from=date(2014, 1, 1),
        date_to=end,
    )

    segments = {r["segment"] for r in result_a.rows}
    # 2014-06-07 (OP), 2019 (C1) and the development day from 2026-11-28 (G3) are scored
    assert {"class:open", "class:conditions", "class:G3"} <= segments
    assert not {"class:G1", "class:G2"} & segments
    assert {r["n_race_days"] for r in result_a.rows if r["segment"] == "all"} == {3}
    g3 = [r for r in result_a.rows if r["segment"] == "class:G3"]
    assert g3 and all(r["n_race_days"] == 1 and r["n_bets"] == UNIT for r in g3), late_dev
    assert result_a.rows == result_b.rows, "the holdout and pre-era meetings change nothing"


def tiny_store(root: Path) -> Path:
    data = root / "data"
    (day,) = boost_free_days(date(2019, 4, 6), 1, ("TOK",))
    write_store(data, [Meeting(day, "TOK", "MAIDEN", "A", 8, "turf", "low")])
    return data


def test_an_out_dir_inside_the_repository_is_refused(tmp_path, monkeypatch):
    data = tiny_store(tmp_path)
    guard(monkeypatch)
    inside = REPO_ROOT / f".spread-scan-test-{uuid.uuid4().hex[:8]}"
    try:
        with pytest.raises(LicensedOutputInRepo):
            run(monkeypatch, data, inside, frozen_registry(tmp_path))
        assert not inside.exists() or not any(p.is_file() for p in inside.rglob("*"))
    finally:
        shutil.rmtree(inside, ignore_errors=True)


def test_a_family_frozen_with_another_spec_is_refused(tmp_path, monkeypatch):
    data = tiny_store(tmp_path)
    guard(monkeypatch)
    registry = temp_registry(tmp_path)
    first = next(iter(ss.FAMILIES))
    registry.freeze_family(first, {"not": "the pre-registered spec"})
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError):
        run(monkeypatch, data, out_dir, registry)
    assert [e for e in registry_lines(registry) if e["kind"] == "run"] == []
    assert not (out_dir / "candidate-cells" / "spread-scan.json").exists()


# --- review fixes (PR #524): coverage, tertiles, empty ranges, run-specific outputs --------------

TINY_POOL = 1.0e6  # the T-2 Win pool written over an unscored meeting's races


def rewrite(path: Path, schema: pa.Schema, change) -> None:
    """Rewrite one synthetic store file through `change` (its rows -> the rows to write)."""
    rows = pq.read_table(path).to_pylist()
    pq.write_table(pa.Table.from_pylist(change(rows), schema=schema), path)


def race_no(race_id: str) -> int:
    return int(race_id.split(":")[3])


def top_rank(spread_id: str) -> int:
    legs = SPREADS[spread_id][1]
    return 3 if legs is None else max(max(leg) for leg in legs)


def scored_reference(facts: Sequence[RaceFacts]) -> dict[tuple[str, str, str], dict]:
    """(variant, spread, cell) -> stake, return, bets, tickets, days and meetings under the spec's
    rules, from the facts alone: a spread is bought in a race in both variants or in neither (both
    rankings reach its highest rank and the race pays its pool); the pool-size tertiles are numpy's
    linear 1/3 and 2/3 quantiles per calendar year over the races holding at least one bet."""
    bets: dict[str, dict[str, dict[str, tuple[int, float, int]]]] = {}
    for race in facts:
        pools = {p for p, _, _ in race.paid}
        shortest = min(len(r) for r in race.ranked.values())
        for spread_id, (pool, _) in SPREADS.items():
            if pool not in pools or shortest < top_rank(spread_id):
                continue
            for variant, ranked in race.ranked.items():
                tickets = {
                    combo_text(pool, c) for c in expected_combos(spread_id, ranked, race.brackets)
                }
                ret = sum(d for p, text, d in race.paid if p == pool and text in tickets)
                per = bets.setdefault(race.race_id, {}).setdefault(spread_id, {})
                per[variant] = (STAKE * len(tickets), ret, len(tickets))
    scored = [race for race in facts if race.race_id in bets]
    levels: dict[str, str] = {}
    for year in {race.meeting.day.year for race in scored}:
        group = [race for race in scored if race.meeting.day.year == year]
        totals = [POOL_LEVELS[race.meeting.pool_level] for race in group]
        low, high = np.quantile(totals, [1 / 3, 2 / 3])
        for race, total in zip(group, totals, strict=True):
            levels[race.race_id] = "low" if total < low else "mid" if total < high else "high"
    acc: dict[tuple[str, str, str], dict] = {}
    for race in scored:
        m = race.meeting
        cells = expected_cells(
            m.race_class, m.field, ODDS[m.fav][0], levels[race.race_id], m.surface
        )
        for spread_id, by_variant in bets[race.race_id].items():
            for variant, (cost, ret, n_tickets) in by_variant.items():
                for cell in cells:
                    a = acc.setdefault(
                        (variant, spread_id, cell),
                        {
                            "cost": 0,
                            "ret": 0.0,
                            "n_bets": 0,
                            "n_tickets": 0,
                            "days": set(),
                            "meetings": set(),
                        },
                    )
                    a["cost"] += cost
                    a["ret"] += ret
                    a["n_bets"] += 1
                    a["n_tickets"] += n_tickets
                    a["days"].add(m.day)
                    a["meetings"].add(m.meeting_id)
    for a in acc.values():
        a["rho"] = a["ret"] / a["cost"]
    return acc


def assert_rows_match(result, ref) -> None:
    by = rows_by_key(result.rows)
    assert set(by) == set(ref), (
        f"rows the reference has not: {sorted(set(by) - set(ref))[:5]}; "
        f"reference cells without a row: {sorted(set(ref) - set(by))[:5]}; skipped {result.skipped}"
    )
    for key, row in by.items():
        a = ref[key]
        assert row["n_bets"] == a["n_bets"], key
        assert row["n_tickets"] == a["n_tickets"], key
        assert row["n_race_days"] == len(a["days"]), key
        assert row["n_meetings"] == len(a["meetings"]), key
        assert row["mean_return"] == pytest.approx(a["rho"], rel=1e-9), key


def test_a_day_file_without_win_rows_skips_its_races_and_the_scan_completes(tmp_path, monkeypatch):
    d0, d1 = boost_free_days(date(2019, 3, 2), 2, ("TOK",))
    kept = Meeting(d0, "TOK", "MAIDEN", "A", 8, "turf", "low")
    o2_only = Meeting(d1, "TOK", "C1", "B", 12, "dirt", "mid")
    data = tmp_path / "data"
    facts = write_store(data, [kept, o2_only])
    folder = data / "processed" / "jravan"

    def quinella_only(rows):
        """The day as the builder writes it from the O2 feed alone: QUINELLA rows, no WIN row."""
        out = []
        for row in rows:
            if row["pool"] != "WIN":
                continue
            if row["combination"] == "__pool__":
                out.append({**row, "pool": "QUINELLA", "investment": row["investment"] * 1.5})
            elif int(row["combination"]) > 1:
                out.append({**row, "pool": "QUINELLA", "combination": f"1,{row['combination']}"})
        assert out and not any(r["pool"] == "WIN" for r in out)
        return out

    rewrite(folder / "odds_timeseries" / f"{d1.isoformat()}.parquet", BOARD_SCHEMA, quinella_only)
    # one scratched runner in the year (in a race of the O2-only day)
    scratch = (f"{o2_only.meeting_id}:03", 7)

    def scratched(rows):
        for row in rows:
            if (row["race_id"], row["horse_no"]) == scratch:
                row.update(is_scratched=True, finish_pos=None, win_odds=None, win_fav_rank=None)
        return rows

    rewrite(folder / "runners.parquet", arrow_schema("runners"), scratched)
    assert pq.read_table(folder / "runners.parquet").column("is_scratched").to_pylist().count(True)

    guard(monkeypatch)
    result = run(
        monkeypatch,
        data,
        tmp_path / "out",
        frozen_registry(tmp_path),
        date_from=date(2019, 1, 1),
        date_to=date(2019, 12, 31),
    )
    assert_rows_match(result, scored_reference([f for f in facts if f.meeting == kept]))
    no_board = {k: n for k, n in result.skipped.items() if "no T-2 Win board" in k}
    assert sum(no_board.values()) == UNIT, result.skipped


def unequal_coverage_store(data: Path) -> list[RaceFacts]:
    """Two meetings of eight runners. In X's races 1-10 ranks 7 and 8 have no price at T-2 (six
    ranked there, eight on the final board); in Y's races 1-5 the final board lacks ranks 6-8
    (eight ranked at T-2, five on the final board)."""
    d0, d1 = boost_free_days(date(2019, 3, 9), 2, ("TOK",))
    x = Meeting(d0, "TOK", "MAIDEN", "A", 8, "turf", "low")
    y = Meeting(d1, "TOK", "C1", "B", 8, "dirt", "high")
    facts = write_store(data, [x, y])
    folder = data / "processed" / "jravan"
    short_t2 = {f.race_id: f for f in facts if f.meeting == x and race_no(f.race_id) <= 10}
    short_final = {f.race_id: f for f in facts if f.meeting == y and race_no(f.race_id) <= 5}
    unpriced = {(rid, str(h)) for rid, f in short_t2.items() for h in f.ranked["t2"][6:]}
    missing = {(rid, str(h)) for rid, f in short_final.items() for h in f.ranked["lookahead"][5:]}

    def unprice_at_t2(rows):
        for row in rows:
            at_t2 = row["minutes_to_post"] == 2.0
            if row["pool"] == "WIN" and at_t2 and (row["race_id"], row["combination"]) in unpriced:
                row["odds"] = None
        return rows

    def drop_final(rows):
        return [
            r
            for r in rows
            if not (r["pool"] == "WIN" and (r["race_id"], r["combination"]) in missing)
        ]

    rewrite(folder / "odds_timeseries" / f"{d0.isoformat()}.parquet", BOARD_SCHEMA, unprice_at_t2)
    rewrite(folder / "final_odds.parquet", FINAL_ODDS_SCHEMA, drop_final)
    for f in short_t2.values():
        f.ranked["t2"] = f.ranked["t2"][:6]
    for f in short_final.values():
        f.ranked["lookahead"] = f.ranked["lookahead"][:5]
    return facts


def test_both_variants_bet_the_same_races_when_the_boards_rank_different_fields(
    tmp_path, monkeypatch
):
    data = tmp_path / "data"
    facts = unequal_coverage_store(data)
    ref = scored_reference(facts)
    # the store reaches the case: a seven-rank and a six-rank spread lose races, unequally
    seven, six = "trifecta:1/2-4/2-7", "trio:1/2-6"
    assert ref[("t2", seven, "all")]["n_bets"] == 2 * UNIT - 10 - 5
    assert ref[("t2", six, "all")]["n_bets"] == 2 * UNIT - 5
    guard(monkeypatch)
    result = run(
        monkeypatch,
        data,
        tmp_path / "out",
        frozen_registry(tmp_path),
        date_from=date(2019, 1, 1),
        date_to=date(2019, 12, 31),
    )
    by = rows_by_key(result.rows)
    pairs = {(s, c) for _, s, c in by}
    for spread_id, cell in sorted(pairs):
        t2, ahead = by.get(("t2", spread_id, cell)), by.get(("lookahead", spread_id, cell))
        assert t2 is not None and ahead is not None, f"{spread_id} {cell} has one variant only"
        assert (t2["n_bets"], t2["n_meetings"], t2["n_race_days"]) == (
            ahead["n_bets"],
            ahead["n_meetings"],
            ahead["n_race_days"],
        ), f"{spread_id} {cell}: the variants bet different races"
    assert_rows_match(result, ref)


def test_pool_size_tertiles_come_from_the_races_holding_a_bet(tmp_path, monkeypatch):
    d0, d1, d2 = boost_free_days(date(2019, 2, 2), 3, ("TOK", "KYO"))
    assert {d0.year, d1.year, d2.year} == {2019}
    valid = [
        Meeting(d0, "TOK", "MAIDEN", "A", 8, "turf", "low"),
        Meeting(d1, "TOK", "C1", "B", 12, "dirt", "mid"),
        Meeting(d2, "TOK", "OP", "A", 10, "turf", "high"),
    ]
    unknown_class = Meeting(d0, "KYO", "C2", "A", 8, "dirt", "mid")  # race_cells refuses it
    no_dividends = Meeting(d1, "KYO", "C3", "B", 8, "turf", "mid")  # no spread can be settled
    data = tmp_path / "data"
    facts = write_store(data, [*valid, unknown_class, no_dividends])
    folder = data / "processed" / "jravan"
    unscored = {unknown_class.meeting_id, no_dividends.meeting_id}

    def unknown(rows):
        for row in rows:
            if row["meeting_id"] == unknown_class.meeting_id:
                row["race_class"] = "JG1"
        return rows

    def unsettled(rows):
        return [r for r in rows if not r["race_id"].startswith(no_dividends.meeting_id + ":")]

    def tiny_pool(rows):
        for row in rows:
            meeting = row["race_id"].rsplit(":", 1)[0]
            if meeting in unscored and row["pool"] == "WIN" and row["combination"] == "__pool__":
                row["investment"] = TINY_POOL
        return rows

    rewrite(folder / "races.parquet", arrow_schema("races"), unknown)
    rewrite(folder / "dividends.parquet", arrow_schema("dividends"), unsettled)
    for day in (d0, d1):
        rewrite(folder / "odds_timeseries" / f"{day.isoformat()}.parquet", BOARD_SCHEMA, tiny_pool)

    scored = [f for f in facts if f.meeting in valid]
    ref = scored_reference(scored)
    # the store discriminates: edges over every race with a T-2 board would move valid races
    every = [POOL_LEVELS[f.meeting.pool_level] for f in scored] + [TINY_POOL] * 2 * UNIT
    low, high = np.quantile(every, [1 / 3, 2 / 3])
    moved = {
        f.meeting.pool_level
        for f in scored
        if ("low" if POOL_LEVELS[f.meeting.pool_level] < low else "mid"
            if POOL_LEVELS[f.meeting.pool_level] < high else "high") != f.meeting.pool_level
    }  # fmt: skip
    assert moved, "the unscored races must shift the edges for this test to discriminate"
    assert {c: ref[("t2", "exacta:1/2-3", f"pool_size:{c}")]["n_bets"] for c in POOL_LEVELS} == {
        c: UNIT for c in POOL_LEVELS
    }

    guard(monkeypatch)
    result = run(
        monkeypatch,
        data,
        tmp_path / "out",
        frozen_registry(tmp_path),
        date_from=date(2019, 1, 1),
        date_to=date(2019, 12, 31),
    )
    assert_rows_match(result, ref)


PROGRAMMING_ERRORS = (TypeError, KeyError, AttributeError, IndexError, NameError)


def test_a_range_without_a_bet_records_no_scored_run_and_keeps_the_candidate_cells(
    tmp_path, monkeypatch
):
    _, held = _late_days()
    (day,) = boost_free_days(date(2019, 4, 6), 1, ("TOK",))
    data = tmp_path / "data"
    write_store(
        data,
        [
            Meeting(day, "TOK", "MAIDEN", "A", 8, "turf", "low"),
            Meeting(held, "TOK", "G1", "A", 8, "dirt", "high", huge=True),  # a holdout day
        ],
    )
    guard(monkeypatch)
    registry = frozen_registry(tmp_path)
    out_dir = tmp_path / "out"
    first = run(
        monkeypatch,
        data,
        out_dir,
        registry,
        date_from=date(2019, 1, 1),
        date_to=date(2019, 12, 31),
    )
    cells = out_dir / "candidate-cells" / "spread-scan.json"
    exported = cells.read_bytes()
    assert first.rows and _strict(exported.decode("utf-8")) == first.rows
    seen = len(registry_lines(registry))
    for lo, hi in ((held, held), (date(2018, 1, 1), date(2018, 12, 31))):
        try:
            run(monkeypatch, data, out_dir, registry, date_from=lo, date_to=hi)
        except Exception as exc:  # a refusal is an allowed outcome; a crash is not
            assert not isinstance(exc, PROGRAMMING_ERRORS), repr(exc)
        assert cells.read_bytes() == exported, f"{lo}..{hi}: the candidate cells were replaced"
        lines = registry_lines(registry)
        new = [e["record"] for e in lines[seen:] if e["kind"] == "run"]
        assert not [r for r in new if r["status"] == "scored"], (
            f"{lo}..{hi} holds no bet, yet a run was recorded as scored"
        )
        seen = len(lines)
    registry.verify()
    assert Counter(r.status for r in registry.records())["scored"] == len(ss.FAMILIES) * 2


def test_each_run_keeps_its_own_report_and_its_records_name_it(tmp_path, monkeypatch):
    data_a = tiny_store(tmp_path / "a")
    data_b = tmp_path / "b" / "data"
    (day,) = boost_free_days(date(2019, 4, 13), 1, ("KYO",))
    write_store(data_b, [Meeting(day, "KYO", "C1", "B", 12, "dirt", "mid")])
    guard(monkeypatch)
    registry = frozen_registry(tmp_path)
    out_dir = tmp_path / "out"
    runs = [  # (store, first day, seed): each differs from the first in one input
        (data_a, date(2014, 6, 7), SEED),
        (data_a, date(2014, 6, 7), SEED + 1),
        (data_a, date(2019, 1, 1), SEED),
        (data_b, date(2014, 6, 7), SEED),
    ]
    written: list[tuple[Path, bytes, list[dict]]] = []  # per run: report, its bytes, its records
    seen = len(registry_lines(registry))
    for data, start, seed in runs:
        monkeypatch.setenv(DATA_VAR, str(data))
        result = ss.run_scan(
            date_from=start,
            date_to=date(2019, 12, 31),
            out_dir=out_dir,
            seed=seed,
            n_boot=N_BOOT,
            registry=registry,
        )
        assert result.rows, "every run scores the store's meeting"
        report = Path(result.report_path)
        assert reports.is_inside(report, out_dir) and report.is_file()
        lines = registry_lines(registry)
        written.append(
            (report, report.read_bytes(), [e for e in lines[seen:] if e["kind"] == "run"])
        )
        seen = len(lines)
    paths_seen = [path.resolve() for path, _, _ in written]
    assert len(set(paths_seen)) == len(runs), "a rerun overwrote an earlier run's report"
    for path, text, _ in written:
        assert path.read_bytes() == text, f"{path.name} was rewritten by a later run"
    for path, _, new in written:
        assert len(new) == len(ss.FAMILIES) * 2
        for entry in new:
            assert path.name in json.dumps(entry["record"]), (
                f"run {entry['run_id']} does not name its report {path.name}"
            )


class WriteFailed(OSError):
    """A write the test makes fail: an `Exception`, so the scan's failure path handles it."""


@pytest.mark.parametrize("failing", ["candidate-cells", "report"])
def test_a_failed_run_names_its_report_only_once_the_report_is_written(
    tmp_path, monkeypatch, failing
):
    data = tiny_store(tmp_path)
    guard(monkeypatch)
    passthrough = ss.write_report

    def failing_write(path, text, *, sources, repo_root=None):
        target = "candidate-cells" if Path(path).parent.name == "candidate-cells" else "report"
        if target == failing:
            raise WriteFailed(f"the {target} write failed: {path}")
        return passthrough(path, text, sources=sources, repo_root=repo_root)

    monkeypatch.setattr(reports, "write_report", failing_write)
    monkeypatch.setattr(ss, "write_report", failing_write)
    registry = frozen_registry(tmp_path)
    out_dir = tmp_path / "out"
    seen = len(registry_lines(registry))
    with pytest.raises(WriteFailed):
        run(monkeypatch, data, out_dir, registry, date_from=date(2019, 1, 1))
    new = [e["record"] for e in registry_lines(registry)[seen:] if e["kind"] == "run"]
    assert Counter((r["family_id"], r["variant"]) for r in new) == Counter(
        itertools.product(ss.FAMILIES, VARIANTS)
    ), "one record per family and variant"
    assert {r["status"] for r in new} == {"failed"}, "a run whose write failed is never scored"
    written = sorted(p for p in out_dir.rglob("*") if p.is_file()) if out_dir.exists() else []
    if failing == "candidate-cells":  # the report was written, then the export failed
        assert len(written) == 1 and written[0].parent == out_dir, written
        expected = f"reports/spread-scan@{written[0].name}"
    else:  # the report write itself failed: there is no report to name
        assert written == [], written
        expected = "reports/spread-scan@not-written"
    for record in new:
        assert record["data_builds"]["report"] == expected, (
            f"{record['family_id']}/{record['variant']}: {record['data_builds']['report']!r}"
        )
    registry.verify()


def _tree(folder: Path) -> dict[str, bytes]:
    return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def test_the_run_command_keeps_every_family_the_summary_index_lists(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from horsetrack.cli import app

    committed_dir = REPO_ROOT / "docs" / "experiments" / "registry" / "japan"
    committed_before = _tree(committed_dir) if committed_dir.is_dir() else {}
    data = tiny_store(tmp_path)
    monkeypatch.setenv(DATA_VAR, str(data))
    summary = tmp_path / "summary" / "registry" / "japan"
    summary.mkdir(parents=True)
    others = ["japan-alpha-final-win", "japan-beta-place-e1", "win5-gamma-v1"]
    for family in others:
        summary.joinpath(f"{family}.md").write_text(f"## {family}\n\nsynthetic\n", "utf-8")
    index = ["# Experiment registry index", "", *(f"- [{f}]({f}.md)" for f in others)]
    summary.joinpath("README.md").write_text("\n".join(index) + "\n", "utf-8")
    before = {family: summary.joinpath(f"{family}.md").read_bytes() for family in others}

    monkeypatch.setattr(ss, "summary_dir", lambda country: summary, raising=False)
    monkeypatch.setattr(registry_module, "summary_dir", lambda country: summary)
    guard(monkeypatch)
    for module in (reports, ss, registry_module):  # nothing is written into the repository
        real = getattr(module, "write_report", None)
        if real is None:
            continue

        def outside(path, text, *, sources, repo_root=None, _real=real):
            if reports.is_inside(Path(path), REPO_ROOT):
                raise Forbidden(f"write_report({path}) is inside the repository")
            return _real(path, text, sources=sources, repo_root=repo_root)

        monkeypatch.setattr(module, "write_report", outside)

    runner = CliRunner()
    frozen = runner.invoke(app, ["japan", "spread-scan", "freeze"])
    assert frozen.exit_code == 0, frozen.output
    done = runner.invoke(
        app,
        ["japan", "spread-scan", "run", "--from", "2019-01-01", "--to", "2019-12-31"]
        + ["--n-boot", "50"],
    )
    assert done.exit_code == 0, (done.output, done.exception)

    listed = summary.joinpath("README.md").read_text("utf-8")
    for family in [*others, *ss.FAMILIES]:
        assert f"({family}.md)" in listed, f"the index dropped {family}"
    for family in ss.FAMILIES:
        assert summary.joinpath(f"{family}.md").is_file(), family
    for family, data_bytes in before.items():
        assert summary.joinpath(f"{family}.md").read_bytes() == data_bytes, f"{family}.md changed"
    after = _tree(committed_dir) if committed_dir.is_dir() else {}
    assert after == committed_before, "the committed summary was touched"


# --- review fixes, round 2 (PR #524): variant identity and the published summary ----------------

FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"  # another commit, for a code-only rerun


def refresh_store(data: Path) -> None:
    """A store refresh: every store file rewritten with the same rows in new bytes (uncompressed)
    and a later modification time, so any fingerprint of the files changes and no row does."""
    for path in sorted((data / "processed" / "jravan").rglob("*.parquet")):
        pq.write_table(pq.read_table(path), path, compression="none")
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))


def test_a_refresh_rerun_is_new_runs_of_the_same_variants(tmp_path, monkeypatch):
    data = tiny_store(tmp_path)
    guard(monkeypatch)
    registry = frozen_registry(tmp_path)
    out_dir = tmp_path / "out"
    seen = len(registry_lines(registry))
    runs: list[tuple[str, object, bytes, list[dict]]] = []  # label, result, report bytes, records

    def scan(label: str, seed: int = SEED) -> None:
        nonlocal seen
        monkeypatch.setenv(DATA_VAR, str(data))
        result = ss.run_scan(
            date_from=date(2019, 1, 1),
            date_to=date(2019, 12, 31),
            out_dir=out_dir,
            seed=seed,
            n_boot=N_BOOT,
            registry=registry,
        )
        lines = registry_lines(registry)
        new = [e for e in lines[seen:] if e["kind"] == "run"]
        seen = len(lines)
        runs.append((label, result, Path(result.report_path).read_bytes(), new))

    scan("first")
    refresh_store(data)
    scan("store refresh")
    monkeypatch.setattr(ss, "_git_sha", lambda: FAKE_SHA, raising=False)
    monkeypatch.setattr(holdout, "_git_sha", lambda: FAKE_SHA)
    scan("code-only rerun")

    first_rows = runs[0][1].rows
    assert first_rows, "the tiny store's meeting is scored"
    ids_before: set[str] = set()
    for label, result, _, new in runs:
        assert result.rows == first_rows, f"{label}: the same rows give the same cells"
        assert len(new) == len(ss.FAMILIES) * len(VARIANTS), f"{label}: one record per variant"
        assert {e["record"]["status"] for e in new} == {"scored"}, label
        ids = {e["run_id"] for e in new}
        assert set(result.run_ids.values()) == ids, label
        assert not ids & ids_before, f"{label}: a rerun must be new runs, not the earlier ones"
        ids_before |= ids
    reports = [Path(result.report_path).resolve() for _, result, _, _ in runs]
    assert len(set(reports)) == len(runs), "a refresh rerun overwrote an earlier run's report"
    for (label, _, text, new), path in zip(runs, reports, strict=True):
        assert path.read_bytes() == text, f"{label}: its report was rewritten by a later run"
        for entry in new:
            assert path.name in json.dumps(entry["record"]), f"{label}: a record lost its report"

    registry.verify()
    for family_id in ss.FAMILIES:
        assert registry.family_size(family_id) == len(VARIANTS), (
            f"{family_id}: a store refresh or a code-only rerun of the same spec, range and seed "
            f"counted as new variants ({registry.family_size(family_id)}, not {len(VARIANTS)})"
        )
    scan("another seed", seed=SEED + 1)
    for family_id in ss.FAMILIES:
        assert registry.family_size(family_id) == 2 * len(VARIANTS), (
            f"{family_id}: another seed is new variants"
        )


_SUMMARY_ROW = re.compile(r"\| P503 \| (t2|lookahead) \| (\w+) \| .*\|")


def summary_rows(path: Path) -> Counter:
    """A family file's run rows, each line as written, as a multiset."""
    return Counter(
        line for line in path.read_text("utf-8").splitlines() if _SUMMARY_ROW.fullmatch(line)
    )


def by_variant(rows: Counter) -> Counter:
    return Counter({_SUMMARY_ROW.fullmatch(line).groups(): n for line, n in rows.items()})


def test_the_run_command_never_drops_a_row_a_family_summary_publishes(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from horsetrack.cli import app

    committed_dir = REPO_ROOT / "docs" / "experiments" / "registry" / "japan"
    committed_before = _tree(committed_dir) if committed_dir.is_dir() else {}
    summary = tmp_path / "summary" / "registry" / "japan"
    summary.mkdir(parents=True)
    monkeypatch.setattr(ss, "summary_dir", lambda country: summary, raising=False)
    monkeypatch.setattr(registry_module, "summary_dir", lambda country: summary)
    guard(monkeypatch)
    for module in (reports, ss, registry_module):  # nothing is written into the repository
        real = getattr(module, "write_report", None)
        if real is None:
            continue

        def outside(path, text, *, sources, repo_root=None, _real=real):
            if reports.is_inside(Path(path), REPO_ROOT):
                raise Forbidden(f"write_report({path}) is inside the repository")
            return _real(path, text, sources=sources, repo_root=repo_root)

        monkeypatch.setattr(module, "write_report", outside)

    runner = CliRunner()
    scan = ["japan", "spread-scan", "run", "--from", "2019-01-01", "--to", "2019-12-31"]
    scan += ["--n-boot", "50"]

    def family_files() -> dict[str, Path]:
        return {family: summary / f"{family}.md" for family in ss.FAMILIES}

    # 1-2. one data directory, two runs: its registry holds every run the files list
    monkeypatch.setenv(DATA_VAR, str(tiny_store(tmp_path / "a")))
    frozen = runner.invoke(app, ["japan", "spread-scan", "freeze"])
    assert frozen.exit_code == 0, frozen.output
    for n_runs, seed in ((1, SEED), (2, SEED + 1)):
        done = runner.invoke(app, [*scan, "--seed", str(seed)])
        assert done.exit_code == 0, (done.output, done.exception)
        for family, path in family_files().items():
            assert path.is_file(), family
            assert by_variant(summary_rows(path)) == Counter(
                {(v, "scored"): n_runs for v in VARIANTS}
            ), f"{family}: after {n_runs} run(s) the file lists every run of the registry"
    published = {family: summary_rows(path) for family, path in family_files().items()}

    # 3. a fresh data directory: its registry lacks the two published runs of each family
    monkeypatch.setenv(DATA_VAR, str(tiny_store(tmp_path / "b")))
    frozen = runner.invoke(app, ["japan", "spread-scan", "freeze"])
    assert frozen.exit_code == 0, frozen.output
    done = runner.invoke(app, [*scan, "--seed", str(SEED)])
    assert not isinstance(done.exception, PROGRAMMING_ERRORS), (done.output, done.exception)
    for family, path in family_files().items():
        assert path.is_file(), f"{family}.md was removed"
        dropped = published[family] - summary_rows(path)
        assert not dropped, (
            f"{family}.md dropped {sum(dropped.values())} published row(s) its local registry "
            "does not hold"
        )
    listed = summary.joinpath("README.md").read_text("utf-8")
    for family in ss.FAMILIES:
        assert f"({family}.md)" in listed, f"the index dropped {family}"
    after = _tree(committed_dir) if committed_dir.is_dir() else {}
    assert after == committed_before, "the committed summary was touched"


# --- #88: RunRecord.notes keeps the ids of records without notes ---------------------------------


def test_an_identical_rerun_returns_the_run_ids_the_registry_wrote(tmp_path, monkeypatch):
    """#88 adds ``RunRecord.notes``, a line without notes is written without the key, so a record
    without notes keeps the id it had before the field. An identical rerun (same store, commit,
    range and seed) appends nothing and returns the ids the first run's registry lines carry."""
    data = tiny_store(tmp_path)
    guard(monkeypatch)
    registry = frozen_registry(tmp_path)
    out_dir = tmp_path / "out"
    seen = len(registry_lines(registry))
    first = run(monkeypatch, data, out_dir, registry, date_from=date(2019, 1, 1))
    lines = registry_lines(registry)
    written = {
        f"{e['record']['family_id']}/{e['record']['variant']}": e["run_id"]
        for e in lines[seen:]
        if e["kind"] == "run"
    }
    assert len(written) == len(ss.FAMILIES) * len(VARIANTS)
    assert first.run_ids == written
    again = run(monkeypatch, data, out_dir, registry, date_from=date(2019, 1, 1))
    assert registry_lines(registry) == lines, "an identical rerun appended a record"
    assert again.run_ids == written, "a rerun's run ids are not the registry's"
