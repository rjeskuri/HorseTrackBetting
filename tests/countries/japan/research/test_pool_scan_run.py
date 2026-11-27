"""Ticket #430: pre-register and run the final-board segment scan (N1), on synthetic stores only.

The owner's reframe of 2026-11-27 applies: every cell is ranked (nothing is closed), the rows are
the ranked cells in the candidate-cell schema, and `test_a_planted_cell_ranks_first` replaces the
open/closed test.

Contract pinned here for `horsetrack.countries.japan.research.pool_scan_run` (the coder implements
what these tests call):

- `FAMILY_ID: str` and `prereg_spec() -> dict`, the pre-registered spec (plain JSON). The family is
  frozen with `registry.freeze_family(FAMILY_ID, prereg_spec())` before the scan runs.
- `run_scan(*, date_from, date_to, out_dir, seed, registry=None) -> ScanResult`:
  - raises a ValueError (any subclass) when `registry.frozen_spec_hash(FAMILY_ID)` is not the hash
    of `prereg_spec()` (never frozen, or frozen with another spec), before any store file is opened
    and without a scored record or a report;
  - reads the jravan store (`processed/jravan/{final_odds,races,dividends}.parquet`) on development
    days only (`core.eval.holdout.is_holdout_day`), from max(date_from, 2014-06-07) to date_to (D10:
    an earlier `date_from` is clamped, not refused), flat races only (a `jump` race is not scored);
    `LockedHoldout` is never opened;
  - scores the six pools `pool_scan.POOLS` (store codes: QUINELLA_PLACE is Wide, FORECAST the
    exacta) and appends registry records for every pool, failures included, each with
    `family_id == FAMILY_ID`, `sources == ('jravan',)`, `hypothesis_id == 'E8'`, `ticket == 430` and
    `params['pool']` the pool's code. A pool whose `core.eval.segment_roi.fit_segment_roi` raises
    gets a `failed` record and the other pools are still scored; run_scan does not raise for it;
  - writes through `core.eval.reports.write_report`, with a withheld source (jravan) among the
    sources, the private report `out_dir / f'pool-scan-japan-{spec_hash}.md'` (the frozen hash)
    and the candidate-cell file `out_dir / 'candidate-cells' / 'pool-scan.json'`. An `out_dir`
    inside the repository raises `LicensedOutputInRepo` and writes no file there. Nothing else is
    written inside the repository (these tests also keep `write_summary` out of it).
- `ScanResult.rows: list[dict]`: every cell of every pool, one row per (pool, cell_id), in rank
  order: rows with a `shrunk_edge` first, by `shrunk_edge` descending (the shrunk expected return
  per unit at the final price, takeout included, minus 1), then the rows without one. The
  candidate-cell file is a strict JSON array of the same rows in the same order. Each row holds the
  candidate-cell keys of the ticket's Context and four more: `omega` (the shrunk omega), `omega_lo`
  and `omega_hi` (its one-sided 95% lower and upper bounds) and `variants_searched` (int: the
  registry family size after the run). Where set, `omega_lo <= omega <= omega_hi` and
  `shrunk_edge == omega * payout_rate - 1`. `class_group` is the cell's `pool_scan.class_group`
  level, null for a cell without a class level; `mean_return` is the cell's flat return per unit
  staked at the final dividend, `mean_odds` the mean final odds of its bets (both over its bets),
  `n_bets`, `n_race_days` and `n_meetings` count its staked combinations, race days and meetings
  (`races.meeting_id`). A cell with bets has float figures; a cell without may hold nulls. Every
  pool has a newcomer and a maiden cell.
- `render_public_report(result) -> str`: names the family id, the spec hash, every run id the scan
  appended and the private report's file name, carries 'withheld (D21)', and shows no figure from
  the rows and no absolute data-directory path.

Spec v2 (review of PR #496), pinned on small synthetic stores of their own:

- A cell mixing payout rates (a Super Premium day's 80% beside a pool's ordinary rate) has
  `1 + shrunk_edge` equal to its mean over bets of rate x omega: for 'all' = newcomer (boosted) +
  conditions (ordinary), the bet-weighted mean of the two class cells' `1 + shrunk_edge`
  ('readings', omega_full weighted by payout rate), not mean omega x mean rate.
- A PLACE or Wide odds band with bets and no winner ('zero_return_band'): every cell with bets has
  finite `omega`, bounds and `shrunk_edge`; the band's cells have omega 0 with bounds 0 (baseline
  0, sd 0); each class x band cell's two registered readings on `MAIN_SPAN` (metric
  `omega[<cell_id>]`) are 0 (omega_full) and 1 (segment_effect, the family's prior mean).
- A race and pool sold on the board with no dividends row ('exclusions') is left out of that pool
  (its counts and mean return are the settled races' alone), counted in
  `result.pool_excluded[pool]['unsettled: no paying dividends row']`, and scored in its other pools.
- A narrower rerun on the same registry, where a cell scored by the earlier run has no bets
  ('spa_stepm'): `result.snooping[pool]` holds `spa` (pvalue, `n_rules` = this run's cells with
  bets, `family_size` = the registry size of `spa_family(pool)`) and `stepm` (a subset of those
  cells), and no `error`.

The stores are synthetic and built here (`write_store`): five runners per race, every pool's board
priced so that each combination's realised wins over a unit of races match its implied
probability exactly (counts per unit below), so every cell's flat return is known in closed form:
the pool's payout rate when fair, 1.6 x 0.8 = 1.28 for the planted WIN newcomer races, whose
boards are the open class's and whose winners are 1.6x as many (dead heats). The private report
names every row's `cell_id`. Holdout days are picked with `is_holdout_day`, never assumed. No real
data is read except by the `local` test, which skips without the real store.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party.
from __future__ import annotations  # noqa: I001

import copy
import itertools
import json
import math
import shutil
import subprocess
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from horsetrack.core import paths
from horsetrack.core.eval import holdout, reports, segment_roi
from horsetrack.core.eval import registry as registry_module
from horsetrack.core.eval.registry import Registry
from horsetrack.core.eval.reports import LicensedOutputInRepo
from horsetrack.core.store.build import arrow_schema
from horsetrack.countries.japan import takeout
from horsetrack.countries.japan.jravan.odds import FINAL_ODDS_SCHEMA
from horsetrack.countries.japan.models.v0 import store as v0store
from horsetrack.countries.japan.research import pool_scan_run as psr

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_VAR = "HORSETRACK_DATA_DIR"
PRICE_ERA_START = date(2014, 6, 7)
SEED = 20261127
WITHHELD = "withheld (D21)"

POOLS = ("WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE", "FORECAST", "TRIO")
# Payout rates (1 - takeout, config/jra.yaml, since 2014-06-07); no synthetic race is boosted.
RATES = {
    "WIN": 0.80,
    "PLACE": 0.80,
    "QUINELLA": 0.775,
    "QUINELLA_PLACE": 0.775,
    "FORECAST": 0.75,
    "TRIO": 0.75,
}
RANGE_POOLS = ("PLACE", "QUINELLA_PLACE")
GROUPS = ("newcomer", "maiden", "conditions", "open", "graded")
GROUP = {"DEBUT": "newcomer", "MAIDEN": "maiden", "C1": "conditions", "OP": "open", "G3": "graded"}
FLAT_CLASSES = tuple(GROUP)
RACE_NO = {"DEBUT": 1, "MAIDEN": 2, "C1": 3, "OP": 4, "G3": 5}
CLASS_RAW = {"DEBUT": "701", "MAIDEN": "703", "C1": "005", "OP": "999", "G3": "999"}
JUMP_RACE_NO = 6
HORSES = (1, 2, 3, 4, 5)


def _combos(pool: str) -> list[str]:
    """Every combination of `pool` over five runners, written as the jravan store writes it."""
    if pool in ("WIN", "PLACE"):
        return [str(h) for h in HORSES]
    if pool == "FORECAST":
        return [f"{a},{b}" for a, b in itertools.permutations(HORSES, 2)]
    legs = 3 if pool == "TRIO" else 2
    return [",".join(str(h) for h in c) for c in itertools.combinations(HORSES, legs)]


COMBOS = {pool: _combos(pool) for pool in POOLS}
PAYING = {"WIN": 1, "PLACE": 2, "QUINELLA": 1, "QUINELLA_PLACE": 3, "FORECAST": 1, "TRIO": 1}

# Wins per combination (in COMBOS order) over one unit of races; a combination's implied
# probability is count / unit and its final odds rate * unit / count, so its realised return per
# unit staked over the unit is exactly the payout rate. WIN's units are 40 races; the newcomer
# (DEBUT) board is priced as the open class's, so no odds level, band or favourite level belongs to
# newcomers alone, and the planted newcomer races win 1.6x as often (dead heats pay each co-winner).
WIN_UNIT = 40
WIN_COUNTS = {"E": (15, 10, 5, 5, 5), "B": (10, 10, 8, 6, 6), "C": (14, 10, 8, 6, 2)}
WIN_PROFILE = {"DEBUT": "E", "MAIDEN": "B", "C1": "B", "OP": "E", "G3": "C"}
OTHER_COUNTS = {
    "PLACE": (20, (14, 9, 7, 6, 4)),
    "QUINELLA": (20, (4, 3, 3, 2, 2, 2, 1, 1, 1, 1)),
    "QUINELLA_PLACE": (20, (10, 9, 8, 7, 6, 6, 5, 4, 3, 2)),
    "FORECAST": (40, (5,) * 2 + (3,) * 6 + (1,) * 12),  # odds 6, 10, 30: none of TRIO's
    "TRIO": (20, (4, 3, 3, 2, 2, 2, 1, 1, 1, 1)),
}
POOL_SHARE = {
    "WIN": 1.0,
    "PLACE": 0.5,
    "QUINELLA": 2.0,
    "QUINELLA_PLACE": 0.8,
    "FORECAST": 1.2,
    "TRIO": 2.5,
}
PLANTED_BOOST = 1.6  # the WIN newcomer races' omega: realised wins 1.6x the implied probability
UNIT = 20  # the dev store's scored days: one unit of each class


def counts_for(pool: str, race_class: str) -> tuple[int, tuple[int, ...]]:
    """(unit, implied wins per combination over the unit): the board's prices."""
    if pool == "WIN":
        return WIN_UNIT, WIN_COUNTS[WIN_PROFILE[race_class]]
    return OTHER_COUNTS[pool]


def wins_for(pool: str, race_class: str, boost: float) -> tuple[int, tuple[int, ...]]:
    """(unit, realised wins per combination over the unit): `boost` x the implied wins."""
    unit, counts = counts_for(pool, race_class)
    wins = tuple(round(c * boost) for c in counts)
    assert all(math.isclose(w, c * boost) for w, c in zip(wins, counts, strict=True))
    return unit, wins


def prices(pool: str, race_class: str, rate: float | None = None) -> list[float]:
    """The board's final odds, priced at `rate` (None: the pool's ordinary payout rate)."""
    unit, counts = counts_for(pool, race_class)
    return [(RATES[pool] if rate is None else rate) * unit / c for c in counts]


def paying_slots(unit: int, wins: Sequence[int]) -> list[list[int]]:
    """Per slot of a unit, the distinct combinations that pay; combination i pays wins[i] times
    over the unit (a slot holding two WIN combinations is a dead heat)."""
    assert max(wins) <= unit
    out: list[list[int]] = [[] for _ in range(unit)]
    for s, i in enumerate(i for i, n in enumerate(wins) for _ in range(n)):
        out[s % unit].append(i)
    assert all(len(set(slot)) == len(slot) for slot in out)
    return out


def _check_profiles() -> None:
    for pool in POOLS:
        for race_class in FLAT_CLASSES:
            unit, counts = counts_for(pool, race_class)
            assert len(counts) == len(COMBOS[pool])
            assert sum(counts) == unit * PAYING[pool]
            assert all(len(s) == PAYING[pool] for s in paying_slots(unit, counts))
    _, planted = wins_for("WIN", "DEBUT", PLANTED_BOOST)
    assert planted == (24, 16, 8, 8, 8)
    assert all(1 <= len(s) <= 2 for s in paying_slots(WIN_UNIT, planted))
    for pool, counts in LONGSHOT_COUNTS.items():
        fair = [c for c in COMBOS[pool] if LONGSHOT_HORSE not in c.split(",")]
        assert len(fair) == len(counts) and sum(counts) == LONGSHOT_UNIT * PAYING[pool]
        assert all(len(s) == PAYING[pool] for s in paying_slots(LONGSHOT_UNIT, counts))


# A PLACE or Wide board whose combinations with horse 5 never pay on the span: horse 5 at odds 8
# on PLACE (band 5.0-10.0), its pairs at odds 30 on Wide (band 20.0-50.0), bands no other price
# falls in; the other combinations are fair over a unit of 20 races.
LONGSHOT_HORSE = "5"
LONGSHOT_UNIT = 20
LONGSHOT_ODDS = {"PLACE": 8.0, "QUINELLA_PLACE": 30.0}
LONGSHOT_COUNTS = {"PLACE": (14, 10, 9, 7), "QUINELLA_PLACE": (14, 12, 10, 10, 8, 6)}


_check_profiles()


# --- synthetic jravan stores ----------------------------------------------------------------------


@dataclass(frozen=True)
class Race:
    day: date
    venue: str
    race_no: int
    race_class: str  # races.race_class
    surface: str
    position: int  # the race's place in its class's unit (which combinations pay)
    pool_rank: int  # the race's pool size, in steps of 1e6 yen
    win_boost: float = 1.0  # realised wins over implied, in the pools `boost_pools`
    boost_pools: tuple[str, ...] = ("WIN",)
    priced_at: float | None = None  # the payout rate every board is priced at (None: RATES)
    longshot: bool = False  # PLACE and Wide boards carry the never-paying long shot
    unpaid: tuple[str, ...] = ()  # pools sold on the board with no dividends row

    @property
    def race_id(self) -> str:
        return f"jra:{self.day.isoformat()}:{self.venue}:{self.race_no:02d}"

    @property
    def meeting_id(self) -> str:
        return f"jravan:{self.day.isoformat()}:{self.venue}"


def board_for(pool: str, race: Race) -> tuple[int, list[float], tuple[int, ...]]:
    """(unit, final odds per combination in COMBOS order, realised wins per combination over the
    unit) of `pool`'s board in `race`."""
    if race.longshot and pool in LONGSHOT_ODDS:
        counts, price, wins = iter(LONGSHOT_COUNTS[pool]), [], []
        for combination in COMBOS[pool]:
            if LONGSHOT_HORSE in combination.split(","):
                price.append(LONGSHOT_ODDS[pool])
                wins.append(0)
            else:
                c = next(counts)
                price.append(RATES[pool] * LONGSHOT_UNIT / c)
                wins.append(c)
        return LONGSHOT_UNIT, price, tuple(wins)
    boost = race.win_boost if pool in race.boost_pools else 1.0
    unit, wins = wins_for(pool, race.race_class, boost)
    return unit, prices(pool, race.race_class, race.priced_at), wins


def paying(pool: str, race: Race) -> list[int]:
    """The combinations (indices into COMBOS[pool]) that pay in `race`."""
    unit, _, wins = board_for(pool, race)
    return paying_slots(unit, wins)[(7 * race.position) % unit]


def race_return(pool: str, race: Race) -> float:
    """The flat box's return in `race` at the final dividend (a dead heat pays each winner)."""
    _, price, _ = board_for(pool, race)
    return sum(price[i] for i in paying(pool, race))


def write_store(data: Path, races: Sequence[Race]) -> None:
    """`processed/jravan/` final_odds (FINAL_ODDS_SCHEMA, one row group per year), races,
    meetings and dividends (the store schemas) for `races`."""
    folder = data / "processed" / "jravan"
    folder.mkdir(parents=True, exist_ok=True)
    board, paid, race_rows, meetings = [], [], [], {}
    for race in races:
        rid = race.race_id
        when = datetime(race.day.year, race.day.month, race.day.day, 6, tzinfo=timezone.utc)
        base = {"jurisdiction": "jra", "race_id": rid, "race_date": race.day, "venue": race.venue}
        base["race_no"] = race.race_no
        for pool in POOLS:
            _, price, _ = board_for(pool, race)
            order = sorted(range(len(price)), key=lambda i, price=price: (price[i], i))
            rank = {i: r + 1 for r, i in enumerate(order)}
            for i, (combination, odds) in enumerate(zip(COMBOS[pool], price, strict=True)):
                board.append(
                    {
                        **base,
                        "pool": pool,
                        "combination": combination,
                        "odds": odds,
                        "odds_hi": odds if pool in RANGE_POOLS else None,
                        "fav_rank": rank[i],
                        "investment": None,
                        "status_raw": None,
                        "published_utc": when,
                        "data_kubun": "4",
                    }
                )
            board.append(
                {
                    **base,
                    "pool": pool,
                    "combination": "__pool__",
                    "odds": None,
                    "odds_hi": None,
                    "fav_rank": None,
                    "investment": race.pool_rank * 1_000_000 * POOL_SHARE[pool],
                    "status_raw": None,
                    "published_utc": when,
                    "data_kubun": "4",
                }
            )
            if pool in race.unpaid:
                continue
            for i in paying(pool, race):
                paid.append(
                    {
                        "jurisdiction": "jra",
                        "race_id": rid,
                        "pool": pool,
                        "combination": COMBOS[pool][i],
                        "dividend": price[i] * 100,
                        "is_refund": False,
                    }
                )
        race_rows.append(
            {
                "jurisdiction": "jra",
                "race_id": rid,
                "meeting_id": race.meeting_id,
                "race_date": race.day,
                "venue": race.venue,
                "race_no": race.race_no,
                "season": str(race.day.year),
                "race_name": f"synthetic {race.race_class}",
                "race_class": race.race_class,
                "distance_m": 3000 if race.surface == "jump" else 1600,
                "surface": race.surface,
                "field_size": len(HORSES),
                "status": "run",
                "weight_type": "age",
                "race_class_raw": CLASS_RAW[race.race_class],
            }
        )
        meetings[race.meeting_id] = {
            "jurisdiction": "jra",
            "meeting_id": race.meeting_id,
            "race_date": race.day,
            "venue": race.venue,
            "season": str(race.day.year),
            "status": "run",
        }

    board.sort(key=lambda r: (r["race_id"], r["pool"], r["combination"]))
    with pq.ParquetWriter(folder / "final_odds.parquet", FINAL_ODDS_SCHEMA) as writer:
        for _, rows in itertools.groupby(board, key=lambda r: r["race_date"].year):
            writer.write_table(pa.Table.from_pylist(list(rows), schema=FINAL_ODDS_SCHEMA))
    race_rows.sort(key=lambda r: r["race_id"])
    pq.write_table(
        pa.Table.from_pylist(race_rows, schema=arrow_schema("races")), folder / "races.parquet"
    )
    pq.write_table(
        pa.Table.from_pylist(
            sorted(meetings.values(), key=lambda r: (r["race_date"], r["venue"])),
            schema=arrow_schema("meetings"),
        ),
        folder / "meetings.parquet",
    )
    paid.sort(key=lambda r: r["race_id"])
    pq.write_table(
        pa.Table.from_pylist(paid, schema=arrow_schema("dividends")), folder / "dividends.parquet"
    )


def boost_free(day: date, venues: Iterable[str], race_nos: Iterable[int]) -> bool:
    calendar = takeout.shipped_boosts()
    return not any(
        calendar.matching(day, venue, n, pool)
        for venue in venues
        for n in race_nos
        for pool in POOLS
    )


def boost_free_days(start: date, count: int, venues, race_nos) -> list[date]:
    days, day = [], start
    while len(days) < count:
        if boost_free(day, venues, race_nos):
            days.append(day)
        day += timedelta(days=1)
    return days


# The planted store: 40 days of 2019, a TOK (turf) and a KYO (dirt) meeting each day, one race of
# each flat class per meeting. A (class, venue) unit has its own pool size; the ranks interleave
# the classes so that every pool-size tertile mixes classes. WIN newcomer (DEBUT) races win 1.6x
# as often as their boards imply (omega 1.6, flat return 1.28); every other board is fair.
PLANTED_VENUES = {"TOK": "turf", "KYO": "dirt"}
PLANTED_RANKS = {
    ("DEBUT", "TOK"): 1,
    ("MAIDEN", "TOK"): 2,
    ("OP", "KYO"): 3,
    ("C1", "TOK"): 4,
    ("G3", "KYO"): 5,
    ("DEBUT", "KYO"): 6,
    ("MAIDEN", "KYO"): 7,
    ("G3", "TOK"): 8,
    ("OP", "TOK"): 9,
    ("C1", "KYO"): 10,
}
PLANTED_DAYS = 40


def planted_races(days: Sequence[date]) -> list[Race]:
    return [
        Race(
            day=day,
            venue=venue,
            race_no=RACE_NO[c],
            race_class=c,
            surface=surface,
            position=j,
            pool_rank=PLANTED_RANKS[(c, venue)],
            win_boost=PLANTED_BOOST if c == "DEBUT" else 1.0,
        )
        for j, day in enumerate(days)
        for venue, surface in PLANTED_VENUES.items()
        for c in FLAT_CLASSES
    ]


@dataclass(frozen=True)
class DevStore:
    """One TOK meeting a day: the five flat classes and a jump maiden (race 6)."""

    scored: tuple[date, ...]  # development days from 2014-06-07 to date_to
    pre_era: tuple[date, ...]  # before 2014-06-07 (D10)
    held: tuple[date, ...]  # holdout days inside the requested range (D14)
    after: tuple[date, ...]  # a development day after date_to
    date_to: date
    races: tuple[Race, ...]


DEV_RANKS = {"DEBUT": 2, "MAIDEN": 4, "C1": 1, "OP": 5, "G3": 3}


def dev_store() -> DevStore:
    early = [date(2014, 6, 7), date(2014, 6, 8), date(2014, 6, 14), date(2014, 6, 15)]
    pre_era = [date(2014, 6, 1), date(2014, 6, 6)]
    late = [date(2026, 11, 28) + timedelta(days=k) for k in range(10)]
    held = [d for d in late if holdout.is_holdout_day("japan", d)]
    late_dev = [d for d in late if not holdout.is_holdout_day("japan", d)]
    assert len(held) >= 2 and len(late_dev) >= 2, "the committed holdout rule should split the days"
    date_to = late[-1]
    after = date_to + timedelta(days=1)
    while holdout.is_holdout_day("japan", after):
        after += timedelta(days=1)
    race_nos = range(1, JUMP_RACE_NO + 1)
    mid = boost_free_days(date(2019, 1, 5), UNIT - len(early) - len(late_dev), ("TOK",), race_nos)
    scored = sorted(early + mid + late_dev)
    assert len(scored) == UNIT
    assert all(boost_free(d, ("TOK",), race_nos) for d in scored)
    assert all(not holdout.is_holdout_day("japan", d) for d in scored)

    position = {d: i for i, d in enumerate(scored)}
    everything = sorted([*scored, *pre_era, *held, after])
    races = []
    for k, day in enumerate(everything):
        j = position.get(day, k % UNIT)
        for c in FLAT_CLASSES:
            races.append(Race(day, "TOK", RACE_NO[c], c, "turf", j, DEV_RANKS[c]))
        races.append(Race(day, "TOK", JUMP_RACE_NO, "MAIDEN", "jump", j, 3))
    return DevStore(
        scored=tuple(scored),
        pre_era=tuple(pre_era),
        held=tuple(held),
        after=(after,),
        date_to=date_to,
        races=tuple(races),
    )


# --- registry, guards and spies -------------------------------------------------------------------


def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = (tmp_path / "data").resolve()
    data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(DATA_VAR, str(data))
    return data


def temp_registry(root: Path) -> Registry:
    return Registry(root / "registry" / "japan.jsonl", holdout_log=root / "holdout_log.jsonl")


def frozen_registry(root: Path) -> tuple[Registry, str]:
    registry = temp_registry(root)
    return registry, registry.freeze_family(psr.FAMILY_ID, psr.prereg_spec())


def registry_lines(registry: Registry) -> list[dict]:
    if not registry.path.exists():
        return []
    return [json.loads(line) for line in registry.path.read_text().splitlines() if line]


def run_ids(registry: Registry) -> list[str]:
    return [
        e["run_id"]
        for e in registry_lines(registry)
        if e["kind"] == "run" and e["record"]["family_id"] == psr.FAMILY_ID
    ]


class Forbidden(BaseException):
    """A guarded call was made. A BaseException, so that no `except Exception` hides it."""


def guard(mp: pytest.MonkeyPatch) -> dict[str, list]:
    """Never open the holdout; keep `write_summary` out of the repository; record every
    `write_report` call (path, sources) and call through."""
    calls: dict[str, list] = {"holdout_open": [], "write_report": [], "summary": []}

    def no_open(*args, **kwargs):
        calls["holdout_open"].append(args)
        raise Forbidden("LockedHoldout.open was called: the scan reads development days only")

    mp.setattr(holdout.LockedHoldout, "open", no_open)

    real_report = reports.write_report

    def spy_report(path, text, *, sources, repo_root=None):
        calls["write_report"].append((Path(path), tuple(sources)))
        return real_report(path, text, sources=sources, repo_root=repo_root)

    mp.setattr(reports, "write_report", spy_report)
    mp.setattr(psr, "write_report", spy_report, raising=False)

    real_summary = registry_module.write_summary

    def kept_out(registry, directory):
        calls["summary"].append(Path(directory))
        if not reports.is_inside(Path(directory), REPO_ROOT):
            real_summary(registry, directory)

    mp.setattr(registry_module, "write_summary", kept_out)
    mp.setattr(psr, "write_summary", kept_out, raising=False)
    return calls


def block_store(mp: pytest.MonkeyPatch) -> list[str]:
    """Make every route into a Parquet file raise `Forbidden`, and record it."""
    opened: list[str] = []

    def blocked(name):
        def refuse(*args, **kwargs):
            opened.append(name)
            raise Forbidden(f"{name} was called before the family check")

        return refuse

    mp.setattr(v0store, "read_store_table", blocked("read_store_table"))
    mp.setattr(psr, "read_store_table", blocked("read_store_table"), raising=False)
    mp.setattr(pa, "memory_map", blocked("pyarrow.memory_map"))
    mp.setattr(pq, "read_table", blocked("pyarrow.parquet.read_table"))
    mp.setattr(pq, "ParquetFile", blocked("pyarrow.parquet.ParquetFile"))
    mp.setattr(pq, "ParquetDataset", blocked("pyarrow.parquet.ParquetDataset"))
    mp.setattr(ds, "dataset", blocked("pyarrow.dataset.dataset"))
    mp.setattr(pd, "read_parquet", blocked("pandas.read_parquet"))
    try:
        import polars
    except ImportError:
        pass
    else:
        mp.setattr(polars, "scan_parquet", blocked("polars.scan_parquet"))
        mp.setattr(polars, "read_parquet", blocked("polars.read_parquet"))
    try:
        import duckdb
    except ImportError:
        pass
    else:
        mp.setattr(duckdb, "connect", blocked("duckdb.connect"))
    return opened


def repo_status() -> str | None:
    """`git status` of the repository, or None where git cannot say."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def scan(store: Path, registry: Registry, *, date_from: date, date_to: date, out_dir=None):
    return psr.run_scan(
        date_from=date_from,
        date_to=date_to,
        out_dir=store / "reports" if out_dir is None else out_dir,
        seed=SEED,
        registry=registry,
    )


def pool_rows(rows: Sequence[dict], pool: str) -> list[dict]:
    return [r for r in rows if r["pool"] == pool]


def with_bets(rows: Iterable[dict]) -> list[dict]:
    return [r for r in rows if r["n_bets"] > 0]


# --- the planted run (shared by the ranking, schema, report and public-report tests) --------------


@dataclass(frozen=True)
class PlantedRun:
    result: object
    data: Path
    out_dir: Path
    registry: Registry
    spec_hash: str
    calls: dict
    repo_before: str | None
    repo_after: str | None


@pytest.fixture(scope="module")
def planted(tmp_path_factory: pytest.TempPathFactory) -> PlantedRun:
    root = tmp_path_factory.mktemp("planted")
    data = (root / "data").resolve()
    data.mkdir()
    days = boost_free_days(date(2019, 1, 5), PLANTED_DAYS, PLANTED_VENUES, RACE_NO.values())
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(DATA_VAR, str(data))
        write_store(data, planted_races(days))
        registry, spec_hash = frozen_registry(root)
        calls = guard(mp)
        before = repo_status()
        result = scan(data, registry, date_from=date(2019, 1, 1), date_to=date(2019, 12, 31))
        after = repo_status()
    return PlantedRun(result, data, data / "reports", registry, spec_hash, calls, before, after)


# ---------------------------------------------------------------------------------------------
# DoD: the family's freeze line comes before every scored record
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("frozen", ["never", "with another spec"])
def test_run_scan_refuses_an_unfrozen_family(tmp_path, monkeypatch, frozen):
    data = data_dir(tmp_path, monkeypatch)
    store = dev_store()
    write_store(data, store.races)  # a complete store: only the family check can refuse
    registry = temp_registry(tmp_path)
    if frozen == "with another spec":
        other = {**psr.prereg_spec(), "test_only_member": "not the pre-registered spec"}
        registry.freeze_family(psr.FAMILY_ID, other)
    calls = guard(monkeypatch)
    opened = block_store(monkeypatch)

    with pytest.raises(ValueError):
        scan(data, registry, date_from=PRICE_ERA_START, date_to=store.date_to)

    assert opened == []
    assert calls["holdout_open"] == []
    assert [r for r in registry.records(psr.FAMILY_ID) if r.status == "scored"] == []
    out = data / "reports"
    written = [p for p in out.rglob("*") if p.is_file()] if out.exists() else []
    assert written == [], f"a refused scan wrote {written}"


# ---------------------------------------------------------------------------------------------
# DoD: no holdout day is read (D14), no price row before 2014-06-07 enters (D10), LockedHoldout is
# never opened; flat races only
# ---------------------------------------------------------------------------------------------


def test_run_scan_reads_development_days_from_2014_06_07_only(tmp_path, monkeypatch):
    data = data_dir(tmp_path, monkeypatch)
    store = dev_store()
    write_store(data, store.races)
    registry, _ = frozen_registry(tmp_path)
    calls = guard(monkeypatch)
    # the excluded days sit inside the requested range, so only the rules can exclude them
    date_from = date(2014, 6, 1)
    assert all(date_from <= d <= store.date_to for d in (*store.pre_era, *store.held))
    assert all(d > store.date_to for d in store.after)

    result = scan(data, registry, date_from=date_from, date_to=store.date_to)

    assert isinstance(result, psr.ScanResult)
    assert isinstance(result.rows, list)
    n_days = len(store.scored)  # one meeting a day, one race of each flat class
    for pool in POOLS:
        mine = with_bets(pool_rows(result.rows, pool))
        assert mine, f"{pool} has no scored cell"
        # a cell covers at most the scored days; each class cell covers every one of them
        assert max(r["n_race_days"] for r in mine) == n_days, pool
        assert max(r["n_meetings"] for r in mine) == n_days, pool
        for group in GROUPS:
            cells = [r for r in mine if r["class_group"] == group]
            assert cells, f"{pool} has no {group} cell"
            # every combination of the class's flat races on the scored days, and no more: the jump
            # maiden, the pre-era days, the holdout days and the day after date_to add none
            assert max(r["n_bets"] for r in cells) == n_days * len(COMBOS[pool]), (pool, group)
    assert calls["holdout_open"] == []


# ---------------------------------------------------------------------------------------------
# DoD: the private report's ranking; the planted cell ranks first, a fair pool's cells cover
# break-even and rank below it
# ---------------------------------------------------------------------------------------------


def test_a_planted_cell_ranks_first(planted):
    rows = planted.result.rows
    win = with_bets(pool_rows(rows, "WIN"))
    assert win
    top = win[0]
    assert top["class_group"] == "newcomer", f"WIN's first cell is {top['cell_id']}"
    # the planted cell's figures: every newcomer race returns 1.6 x the WIN payout rate
    assert top["mean_return"] == pytest.approx(PLANTED_BOOST * RATES["WIN"], abs=1e-9)
    if top["odds_band"] is None:
        assert top["mean_odds"] == pytest.approx(np.mean(prices("WIN", "DEBUT")))
    for r in win:
        if r["class_group"] == "newcomer":
            assert r["mean_return"] == pytest.approx(1.28, abs=1e-9), r["cell_id"]
        elif r["class_group"] is not None:
            assert r["mean_return"] == pytest.approx(RATES["WIN"], abs=1e-9), r["cell_id"]
        else:  # a mixture of newcomer and fair races
            assert RATES["WIN"] - 1e-9 <= r["mean_return"] <= 1.28 + 1e-9, r["cell_id"]
    # within WIN the newcomer cell's shrunk omega is the highest
    assert all(top["omega"] >= r["omega"] for r in win if r["omega"] is not None)
    assert top["omega"] > 1.0

    first = next(i for i, r in enumerate(rows) if r is top)
    quinella = [(i, r) for i, r in enumerate(rows) if r["pool"] == "QUINELLA" and r["n_bets"] > 0]
    assert quinella
    break_even = 1 / RATES["QUINELLA"]
    quinella_mean_odds = np.mean(prices("QUINELLA", "C1"))
    for i, r in quinella:
        assert i > first, f"QUINELLA {r['cell_id']} ranks above the planted cell"
        assert r["mean_return"] == pytest.approx(RATES["QUINELLA"], abs=1e-9), r["cell_id"]
        if r["odds_band"] is None:
            assert r["mean_odds"] == pytest.approx(quinella_mean_odds), r["cell_id"]
        lo, omega, hi = r["omega_lo"], r["omega"], r["omega_hi"]
        assert None not in (lo, omega, hi), r["cell_id"]
        assert lo <= omega <= hi, r["cell_id"]
        assert lo <= 1.0 <= hi, f"{r['cell_id']}: [{lo}, {hi}] misses the fair omega 1"
        # the one-sided lower bound covers break-even: no fair cell is shown to beat the takeout
        assert lo <= break_even, r["cell_id"]
        assert abs(omega - 1.0) <= 0.1, r["cell_id"]
    # every other fair pool returns exactly its payout rate in every cell
    for pool in ("PLACE", "QUINELLA_PLACE", "FORECAST", "TRIO"):
        for r in with_bets(pool_rows(rows, pool)):
            assert r["mean_return"] == pytest.approx(RATES[pool], abs=1e-9), (pool, r["cell_id"])


# ---------------------------------------------------------------------------------------------
# DoD: every pool's run, scored or failed, is a registry record with the family id and
# sources=('jravan',); the family size is reported with every result
# ---------------------------------------------------------------------------------------------

TRIO_ODDS = frozenset(round(o, 6) for o in prices("TRIO", "C1"))


def _is_trio(bets: pd.DataFrame, takeout_col: str) -> bool:
    """TRIO's bets in the dev store: takeout 0.25 and only TRIO's prices (FORECAST's differ)."""
    odds = {round(float(o), 6) for o in bets["odds"]}
    return bool(np.allclose(bets[takeout_col].astype(float), 0.25)) and odds <= TRIO_ODDS


def test_a_failed_pool_still_writes_a_registry_record(tmp_path, monkeypatch):
    assert not TRIO_ODDS & {round(o, 6) for o in prices("FORECAST", "C1")}
    data = data_dir(tmp_path, monkeypatch)
    store = dev_store()
    write_store(data, store.races)
    registry, _ = frozen_registry(tmp_path)
    guard(monkeypatch)
    real_fit = segment_roi.fit_segment_roi

    def fit_failing_on_trio(bets, **kwargs):
        if _is_trio(bets, kwargs["takeout_col"]):
            raise RuntimeError("injected: the TRIO shrinkage fit fails")
        return real_fit(bets, **kwargs)

    monkeypatch.setattr(segment_roi, "fit_segment_roi", fit_failing_on_trio)
    monkeypatch.setattr(psr, "fit_segment_roi", fit_failing_on_trio, raising=False)

    result = scan(data, registry, date_from=PRICE_ERA_START, date_to=store.date_to)

    records = registry.records(psr.FAMILY_ID)
    assert records
    for r in records:
        assert r.family_id == psr.FAMILY_ID
        assert r.sources == ("jravan",)
        assert r.hypothesis_id == "E8"
        assert r.ticket == 430
        assert r.params.get("pool") in POOLS, r.variant
    by_pool = {pool: [r for r in records if r.params["pool"] == pool] for pool in POOLS}
    assert any(r.status == "failed" for r in by_pool["TRIO"])
    for pool in POOLS:
        if pool == "TRIO":
            continue
        assert any(r.status == "scored" for r in by_pool[pool]), f"{pool} was not scored"
        scored = [
            r for r in with_bets(pool_rows(result.rows, pool)) if r["shrunk_edge"] is not None
        ]
        assert scored, f"{pool} has no ranked cell"

    # the freeze line comes before every record of the family
    lines = registry_lines(registry)
    freeze = [i for i, e in enumerate(lines) if e["kind"] == "freeze"]
    runs = [i for i, e in enumerate(lines) if e["kind"] == "run"]
    assert len(freeze) == 1 and runs and min(runs) > freeze[0]
    # the family size, failed variants included, is on every row
    size = registry.family_size(psr.FAMILY_ID)
    assert size >= len(POOLS)
    assert {r["variants_searched"] for r in result.rows} == {size}


# ---------------------------------------------------------------------------------------------
# DoD: results read from the jravan store go through write_report, under the data dir (D21)
# ---------------------------------------------------------------------------------------------


def test_reports_go_through_write_report_outside_the_repo(planted, tmp_path, monkeypatch):
    report = planted.out_dir / f"pool-scan-japan-{planted.spec_hash}.md"
    candidates = planted.out_dir / "candidate-cells" / "pool-scan.json"
    for path in (report, candidates):
        assert path.is_file(), f"missing {path}"
        assert path.resolve().is_relative_to(planted.data)
        sources = [s for p, s in planted.calls["write_report"] if p.resolve() == path.resolve()]
        assert sources, f"{path.name} was not written through write_report"
        assert all(reports.is_withheld(s) for s in sources), sources
    text = report.read_text(encoding="utf-8")
    for row in planted.result.rows:
        assert row["cell_id"] in text, f"the private report lacks {row['cell_id']}"
    if planted.repo_before is not None and planted.repo_after is not None:
        assert planted.repo_after == planted.repo_before, "run_scan wrote inside the repository"

    # a target inside the repository is refused, and nothing lands there
    data = data_dir(tmp_path, monkeypatch)
    store = dev_store()
    write_store(data, store.races)
    registry, _ = frozen_registry(tmp_path)
    guard(monkeypatch)
    target = REPO_ROOT / f".pool-scan-test-{uuid.uuid4().hex[:12]}"
    try:
        with pytest.raises(LicensedOutputInRepo):
            scan(data, registry, date_from=PRICE_ERA_START, date_to=store.date_to, out_dir=target)
        landed = [p for p in target.rglob("*") if p.is_file()] if target.exists() else []
        assert landed == [], f"files landed inside the repository: {landed}"
    finally:
        shutil.rmtree(target, ignore_errors=True)


# ---------------------------------------------------------------------------------------------
# DoD: the public report carries no JRA-VAN count, rate, span, sign or bound
# ---------------------------------------------------------------------------------------------

FIGURE_RHO, FIGURE_N = 1.2345, 98765
COUNT_KEYS = ("n_bets", "n_race_days", "n_meetings")


def test_public_report_carries_no_figures(planted):
    result = copy.deepcopy(planted.result)
    assert result.rows
    for row in result.rows:
        for key, value in list(row.items()):
            if isinstance(value, float) and not isinstance(value, bool):
                row[key] = FIGURE_RHO
        for key in COUNT_KEYS:
            row[key] = FIGURE_N

    text = psr.render_public_report(result)

    assert isinstance(text, str)
    assert "1.2345" not in text
    assert str(FIGURE_N) not in text
    assert WITHHELD in text
    assert psr.FAMILY_ID in text
    assert planted.spec_hash in text
    ids = run_ids(planted.registry)
    assert ids
    for run_id in ids:
        assert run_id in text, f"run {run_id} is not named"
    assert f"pool-scan-japan-{planted.spec_hash}.md" in text
    assert str(planted.data) not in text, "the public report names the private data directory"


# ---------------------------------------------------------------------------------------------
# DoD: the candidate-cell file lists every cell, ranked, in the shared schema
# ---------------------------------------------------------------------------------------------

STR_KEYS = ("study", "family_id", "cell_id", "pool")
OPTIONAL_STR_KEYS = ("class_group", "odds_band")
INT_KEYS = ("n_bets", "n_race_days", "n_meetings", "variants_searched")
FLOAT_KEYS = ("payout_rate", "mean_return", "se_mean_return", "mean_odds")
OPTIONAL_FLOAT_KEYS = ("shrunk_edge", "omega", "omega_lo", "omega_hi")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_float(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _strict_json(text: str):
    def refuse(constant):
        raise ValueError(f"not strict JSON: {constant}")

    return json.loads(text, parse_constant=refuse)


def test_candidate_cells_follow_the_schema(planted):
    path = planted.out_dir / "candidate-cells" / "pool-scan.json"
    cells = _strict_json(path.read_text(encoding="utf-8"))
    rows = planted.result.rows
    assert isinstance(cells, list) and len(cells) == len(rows) > 0
    size = planted.registry.family_size(psr.FAMILY_ID)

    for cell, row in zip(cells, rows, strict=True):
        assert isinstance(cell, dict)
        for key in STR_KEYS:
            assert isinstance(cell[key], str) and cell[key].strip(), key
        for key in OPTIONAL_STR_KEYS:
            assert cell[key] is None or isinstance(cell[key], str), key
        for key in INT_KEYS:
            assert _is_int(cell[key]) and cell[key] >= 0, key
        figures = FLOAT_KEYS if cell["n_bets"] > 0 else ("payout_rate",)
        for key in figures:
            assert _is_float(cell[key]), (key, cell[key])
        for key in (*FLOAT_KEYS, *OPTIONAL_FLOAT_KEYS):
            assert cell[key] is None or _is_float(cell[key]), (key, cell[key])
        # the file holds the rows, in rank order
        for key in (*STR_KEYS, *OPTIONAL_STR_KEYS, *INT_KEYS):
            assert cell[key] == row[key], key
        for key in (*FLOAT_KEYS, *OPTIONAL_FLOAT_KEYS):
            assert (cell[key] is None) == (row[key] is None), key
            if cell[key] is not None:
                assert cell[key] == pytest.approx(row[key], rel=1e-12, abs=1e-12), key

        assert cell["family_id"] == psr.FAMILY_ID
        assert cell["pool"] in POOLS
        assert cell["class_group"] in (*GROUPS, None)
        assert cell["payout_rate"] == pytest.approx(RATES[cell["pool"]], abs=1e-9)
        assert cell["variants_searched"] == size
        if cell["n_bets"] > 0:
            assert cell["n_bets"] >= cell["n_meetings"] >= cell["n_race_days"] >= 1
            assert cell["mean_odds"] >= 1.0
            assert cell["se_mean_return"] >= 0.0
            if cell["n_meetings"] > 1:  # every synthetic cell's returns vary across meetings
                assert cell["se_mean_return"] > 0.0, cell["cell_id"]
        if cell["omega"] is not None:
            assert cell["omega_lo"] <= cell["omega"] <= cell["omega_hi"], cell["cell_id"]
            assert cell["shrunk_edge"] == pytest.approx(
                cell["omega"] * cell["payout_rate"] - 1, abs=1e-9
            ), cell["cell_id"]

    assert len({c["study"] for c in cells}) == 1
    keys = [(c["pool"], c["cell_id"]) for c in cells]
    assert len(set(keys)) == len(keys), "a cell is listed twice"
    edges = [c["shrunk_edge"] for c in cells]
    ranked = [e for e in edges if e is not None]
    assert ranked, "no cell is ranked"
    assert edges[: len(ranked)] == ranked, "unranked cells come after the ranked ones"
    assert all(a >= b for a, b in itertools.pairwise(ranked)), "cells are not in rank order"

    # every pool has its newcomer and maiden cells, with every bet of the class counted
    meetings = PLANTED_DAYS * len(PLANTED_VENUES)
    races_per_class = meetings  # one race of each class per meeting
    for pool in POOLS:
        mine = with_bets(c for c in cells if c["pool"] == pool)
        assert max(c["n_race_days"] for c in mine) == PLANTED_DAYS, pool
        assert max(c["n_meetings"] for c in mine) == meetings, pool
        for group in GROUPS:
            of_group = [c for c in mine if c["class_group"] == group]
            if group in ("newcomer", "maiden"):
                assert of_group, f"{pool} has no {group} cell"
            if of_group:
                want = races_per_class * len(COMBOS[pool])
                assert max(c["n_bets"] for c in of_group) == want, (pool, group)


# ---------------------------------------------------------------------------------------------
# Review of PR #496 (spec v2): a cell mixing payout rates ranks by its mean expected return
# ---------------------------------------------------------------------------------------------

SUPER_PREMIUM = takeout.SUPER_PREMIUM_RATE  # a Super Premium day pays 80% in every pool
# The #78 pools whose ordinary rate is below 80%, so that Super Premium raises it (WIN pays 80%
# either way, and a premium, which may be withheld, leaves the race out).
MIXED_POOLS = ("QUINELLA", "FORECAST", "TRIO")
MIXED_BOOST = 2.0
MIXED_PER_DAY = 10


def super_premium_days(year: int, venue: str, race_nos: Sequence[int]) -> list[date]:
    """Days of `year` on which every MIXED_POOLS pool of races `race_nos` at `venue` matches
    Super Premium entries only (no premium and no age entry), read off the shipped calendar."""
    calendar = takeout.shipped_boosts()
    days = {e.date for e in calendar.entries if e.kind == "super_premium" and e.date.year == year}
    return [
        day
        for day in sorted(days)
        if {
            (e.kind, e.age)
            for n in race_nos
            for pool in MIXED_POOLS
            for e in calendar.matching(day, venue, n, pool)
        }
        == {("super_premium", None)}
    ]


def mixed_rate_races() -> list[Race]:
    """20 newcomer races on two Super Premium days, their boards priced at 80% and their
    MIXED_POOLS combinations winning twice as often as they imply; 20 fair conditions races on
    two ordinary days."""
    race_nos = range(1, MIXED_PER_DAY + 1)
    boosted = super_premium_days(2019, "TOK", race_nos)[:2]
    assert len(boosted) == 2, "the shipped calendar should hold two such days in 2019"
    ordinary = boost_free_days(date(2019, 1, 5), 2, ("TOK",), race_nos)
    races = [
        Race(day, "TOK", n, "DEBUT", "turf", j, 1 + j % 5, MIXED_BOOST, MIXED_POOLS, SUPER_PREMIUM)
        for j, (day, n) in enumerate(itertools.product(boosted, race_nos))
    ]
    races += [
        Race(day, "TOK", n, "C1", "turf", j, 1 + (j + 2) % 5)
        for j, (day, n) in enumerate(itertools.product(ordinary, race_nos))
    ]
    return races


def test_a_mixed_rate_cell_ranks_by_its_rate_weighted_expected_return(tmp_path, monkeypatch):
    data = data_dir(tmp_path, monkeypatch)
    races = mixed_rate_races()
    write_store(data, races)
    registry, _ = frozen_registry(tmp_path)
    guard(monkeypatch)

    result = scan(data, registry, date_from=date(2019, 1, 1), date_to=date(2019, 12, 31))

    per_class = sum(r.race_class == "DEBUT" for r in races)
    assert per_class == sum(r.race_class == "C1" for r in races)
    for pool in MIXED_POOLS:
        cells = {r["cell_id"]: r for r in pool_rows(result.rows, pool)}
        boosted, ordinary, mixed = cells["class:newcomer"], cells["class:conditions"], cells["all"]
        n = per_class * len(COMBOS[pool])
        assert (boosted["n_bets"], ordinary["n_bets"], mixed["n_bets"]) == (n, n, 2 * n), pool
        assert boosted["payout_rate"] == pytest.approx(SUPER_PREMIUM, abs=1e-12), pool
        assert ordinary["payout_rate"] == pytest.approx(RATES[pool], abs=1e-12), pool
        assert mixed["payout_rate"] == pytest.approx((SUPER_PREMIUM + RATES[pool]) / 2), pool
        for r in (boosted, ordinary, mixed):
            assert r["omega"] is not None and r["shrunk_edge"] is not None, (pool, r["cell_id"])
            want = r["omega"] * r["payout_rate"] - 1
            assert r["shrunk_edge"] == pytest.approx(want, abs=1e-12), (pool, r["cell_id"])
        # Expected return is the mean over bets of rate x omega, and a bet's omega is the same in
        # every cell that holds it, so the mixed cell's expected return is the bet-weighted mean
        # of its two single-rate halves'.
        halves = (2 + boosted["shrunk_edge"] + ordinary["shrunk_edge"]) / 2
        assert 1 + mixed["shrunk_edge"] == pytest.approx(halves, rel=1e-9), pool
        # The fixture tells it apart from mean omega x mean rate.
        assert boosted["omega"] > ordinary["omega"] + 0.3, pool
        naive = (boosted["omega"] + ordinary["omega"]) / 2 * mixed["payout_rate"]
        assert abs(halves - naive) > 1e-3, pool


# ---------------------------------------------------------------------------------------------
# Review of PR #496 (spec v2, 'zero_return_band'): a Place or Wide band with bets and no winner
# ---------------------------------------------------------------------------------------------


def longshot_races() -> list[Race]:
    """A maiden and a conditions race on each of 20 days, every PLACE and Wide board carrying the
    long shot (LONGSHOT_ODDS), which never places."""
    classes = ("MAIDEN", "C1")
    days = boost_free_days(date(2019, 1, 5), LONGSHOT_UNIT, ("TOK",), [RACE_NO[c] for c in classes])
    return [
        Race(day, "TOK", RACE_NO[c], c, "turf", j, DEV_RANKS[c], longshot=True)
        for j, day in enumerate(days)
        for c in classes
    ]


# A winless band's loss is constant per bet, so arch warns that its studentized statistic is
# degenerate; SPA is not what this test is about.
@pytest.mark.filterwarnings("ignore:The loss differential has a variance of 0")
def test_a_winless_place_or_wide_band_keeps_finite_readings(tmp_path, monkeypatch):
    data = data_dir(tmp_path, monkeypatch)
    races = longshot_races()
    write_store(data, races)
    registry, _ = frozen_registry(tmp_path)
    guard(monkeypatch)

    result = scan(data, registry, date_from=date(2019, 1, 1), date_to=date(2019, 12, 31))

    records = registry.records(psr.FAMILY_ID)
    run = {GROUP[c]: sum(r.race_class == c for r in races) for c in {r.race_class for r in races}}
    for pool in RANGE_POOLS:
        mine = with_bets(pool_rows(result.rows, pool))
        for r in mine:
            figures = [r[k] for k in ("omega", "omega_lo", "omega_hi", "shrunk_edge")]
            assert all(v is not None and math.isfinite(v) for v in figures), (pool, r["cell_id"])
            assert r["omega_lo"] <= r["omega"] <= r["omega_hi"], (pool, r["cell_id"])
        longshots = sum(LONGSHOT_HORSE in c.split(",") for c in COMBOS[pool])
        winless = [
            r
            for r in mine
            if r["odds_band"] is not None and r["mean_odds"] == pytest.approx(LONGSHOT_ODDS[pool])
        ]
        # the band itself, and its cross with each class run
        assert sorted(r["class_group"] or "" for r in winless) == sorted(["", *run]), pool
        for r in winless:
            group = r["class_group"]
            want = longshots * (run[group] if group else len(races))
            assert r["n_bets"] == want, (pool, r["cell_id"])
            assert r["mean_return"] == 0.0, (pool, r["cell_id"])
            # its band baseline is 0, so omega_full is 0 with sd 0 and the expected return 0
            got = (r["omega"], r["omega_lo"], r["omega_hi"], r["shrunk_edge"])
            assert got == pytest.approx((0.0, 0.0, 0.0, -1.0), abs=1e-12), (pool, r["cell_id"])
            if group is None:
                continue
            # both readings are registered: omega_full 0, segment_effect the family's prior mean 1
            name = f"omega[{r['cell_id']}]"
            shown = [
                m
                for rec in records
                if rec.params.get("pool") == pool
                for m in rec.metrics
                if m.metric == name and m.fold == psr.MAIN_SPAN
            ]
            values = sorted(m.value for m in shown)
            assert values == pytest.approx([0.0, 1.0], abs=1e-12), (pool, name, values)
            for m in shown:
                assert None not in (m.lo, m.hi), (pool, name)
                assert math.isfinite(m.lo) and math.isfinite(m.hi), (pool, name)
                assert m.lo <= m.value <= m.hi, (pool, name)


# ---------------------------------------------------------------------------------------------
# Review of PR #496 (spec v2, 'exclusions'): a sold race and pool with no paying dividends row
# ---------------------------------------------------------------------------------------------

UNSETTLED_REASON = "unsettled: no paying dividends row"
UNSETTLED = (("QUINELLA",), ("TRIO",), ("TRIO", "PLACE"))  # one extra race each, unpaid pools


def unsettled_races() -> list[Race]:
    """40 settled conditions races on 20 days, then one race a day per UNSETTLED entry whose
    pools named there have no dividends row (sold: priced, with a pool total)."""
    race_nos = (3, 4)
    days = boost_free_days(date(2019, 1, 5), 20 + len(UNSETTLED), ("TOK",), race_nos)
    races = [
        Race(day, "TOK", n, "C1", "turf", 2 * j + k, 1 + (2 * j + k) % 3)
        for j, day in enumerate(days[:20])
        for k, n in enumerate(race_nos)
    ]
    races += [
        Race(day, "TOK", 3, "C1", "turf", 5 + k, 2, unpaid=pools)
        for k, (day, pools) in enumerate(zip(days[20:], UNSETTLED, strict=True))
    ]
    return races


def test_a_sold_pool_without_a_paying_dividend_is_excluded_and_counted(tmp_path, monkeypatch):
    data = data_dir(tmp_path, monkeypatch)
    races = unsettled_races()
    write_store(data, races)
    registry, _ = frozen_registry(tmp_path)
    guard(monkeypatch)

    result = scan(data, registry, date_from=date(2019, 1, 1), date_to=date(2019, 12, 31))

    for pool in POOLS:
        settled = [r for r in races if pool not in r.unpaid]
        n = len(settled) * len(COMBOS[pool])
        whole = next(r for r in pool_rows(result.rows, pool) if r["cell_id"] == "all")
        assert whole["n_bets"] == n, pool
        assert whole["n_race_days"] == len({r.day for r in settled}), pool
        # the settled races alone: an unsettled one is not scored as if every ticket lost
        want = sum(race_return(pool, r) for r in settled) / n
        assert whole["mean_return"] == pytest.approx(want, rel=1e-9), pool
        counted = result.pool_excluded.get(pool, {}).get(UNSETTLED_REASON, 0)
        assert counted == len(races) - len(settled), pool


# ---------------------------------------------------------------------------------------------
# Review of PR #496 (spec v2, 'spa_stepm'): a narrower rerun on the same registry
# ---------------------------------------------------------------------------------------------


NARROW_FROM = 10  # the narrower rerun starts on this day of narrowing_races


def narrowing_races() -> tuple[list[date], list[Race]]:
    """30 days, one meeting each: a maiden and a conditions race on the first 10, a conditions
    race alone on the last 20 (more meetings than SPA_BLOCK, so the narrower run can be tested)."""
    days = boost_free_days(date(2019, 1, 5), 30, ("TOK",), (RACE_NO["MAIDEN"], RACE_NO["C1"]))
    assert len(days) - NARROW_FROM > psr.SPA_BLOCK
    races = [
        Race(day, "TOK", RACE_NO[c], c, "turf", j, DEV_RANKS[c] + j % 2)
        for j, day in enumerate(days)
        for c in (("MAIDEN", "C1") if j < NARROW_FROM else ("C1",))
    ]
    return days, races


def test_a_narrower_rerun_on_the_same_registry_still_runs_spa_and_stepm(tmp_path, monkeypatch):
    data = data_dir(tmp_path, monkeypatch)
    days, races = narrowing_races()
    write_store(data, races)
    registry, _ = frozen_registry(tmp_path)
    guard(monkeypatch)

    full = scan(data, registry, date_from=date(2019, 1, 1), date_to=date(2019, 12, 31))
    narrow = scan(data, registry, date_from=days[NARROW_FROM], date_to=date(2019, 12, 31))

    for pool in POOLS:
        before = {r["cell_id"]: r for r in pool_rows(full.rows, pool)}
        after = {r["cell_id"]: r for r in pool_rows(narrow.rows, pool)}
        assert before["class:maiden"]["n_bets"] > 0, pool
        assert after["class:maiden"]["n_bets"] == 0, pool
        for result, rows in ((full, before), (narrow, after)):
            snoop = result.snooping[pool]
            assert "error" not in snoop, (pool, snoop.get("error"))
            tested = {cell for cell, r in rows.items() if r["n_bets"] > 0}
            assert 0.0 <= snoop["spa"]["pvalue"] <= 1.0, pool
            assert snoop["spa"]["n_rules"] == len(tested), pool
            assert set(snoop["stepm"]) <= tested, pool
        # the narrower run's SPA reports every variant tried, both runs' cell rules included
        size = registry.family_size(psr.spa_family(pool))
        assert narrow.snooping[pool]["spa"]["family_size"] == size, pool


# ---------------------------------------------------------------------------------------------
# The real store (local: needs $HORSETRACK_DATA_DIR/processed/jravan/final_odds.parquet)
# ---------------------------------------------------------------------------------------------


def _last_board_day(path: Path) -> date:
    """The last race_date of the file, from its row-group statistics (no row is read)."""
    meta = pq.ParquetFile(path).metadata
    column = meta.schema.to_arrow_schema().get_field_index("race_date")
    last = []
    for g in range(meta.num_row_groups):
        stats = meta.row_group(g).column(column).statistics
        if stats is None or not stats.has_min_max:
            pytest.skip(f"{path} has a row group without race_date statistics")
        last.append(pd.Timestamp(stats.max).date())
    if not last:
        pytest.skip(f"{path} has no row group")
    return max(last)


@pytest.mark.local
def test_real_store_one_month_reads_development_days_only(tmp_path, monkeypatch):
    try:
        real = paths.data_dir()
    except paths.DataDirNotSet:
        pytest.skip(f"{DATA_VAR} is not set: the real jravan store is unavailable")
    board = real / "processed" / "jravan" / "final_odds.parquet"
    if not board.is_file():
        pytest.skip(f"no real jravan final_odds at {board}")
    date_to = _last_board_day(board)
    date_from = max(PRICE_ERA_START, date_to - timedelta(days=30))
    races = v0store.read_store_table(
        "races",
        columns=["race_date", "surface"],
        start=date_from,
        end=date_to,
        touches_price=False,
        source="jravan",
    ).to_pydict()
    flat_days = {
        d for d, s in zip(races["race_date"], races["surface"], strict=True) if s != "jump"
    }
    assert flat_days and not any(holdout.is_holdout_day("japan", d) for d in flat_days)

    registry, _ = frozen_registry(tmp_path)
    calls = guard(monkeypatch)
    seen: list[tuple[str, str, set]] = []
    real_read = v0store.read_store_table

    def read_and_record(name, *, columns, start, end, touches_price, where=None, source="jra"):
        asked = list(columns)
        extra = [] if "race_date" in asked else ["race_date"]
        table = real_read(
            name,
            columns=asked + extra,
            start=start,
            end=end,
            touches_price=touches_price,
            where=where,
            source=source,
        )
        seen.append((name, source, set(table.column("race_date").to_pylist())))
        return table.select(asked)

    monkeypatch.setattr(v0store, "read_store_table", read_and_record)
    monkeypatch.setattr(psr, "read_store_table", read_and_record, raising=False)

    result = psr.run_scan(
        date_from=date_from,
        date_to=date_to,
        out_dir=tmp_path / "reports",
        seed=SEED,
        registry=registry,
    )

    assert any(name == "final_odds" and source == "jravan" for name, source, _ in seen)
    for name, _, days in seen:
        assert all(date_from <= d <= date_to for d in days), name
        assert not any(holdout.is_holdout_day("japan", d) for d in days), name
        if name == "final_odds":
            assert min(days, default=PRICE_ERA_START) >= PRICE_ERA_START
    scored = with_bets(result.rows)
    assert scored
    assert max(r["n_race_days"] for r in scored) <= len(flat_days)
    records = registry.records(psr.FAMILY_ID)
    assert records and all(r.sources == ("jravan",) for r in records)
    assert calls["holdout_open"] == []
