"""Miho and Ritto workouts as of decision time, and the workout features (ticket #137).

Source (owner decision on #137, 2026-11-24): JRDB CHA (one 本追切 per runner per race) and CYB (the
per-course workout counts, merged into `jrdb_pre_race` by #57). JRDB may feed private models only
(D21): the features are an opt-in JRDB feature set behind a flag, and their log-loss report is
written under the data dir only. Nothing in this file is a JRDB figure, layout, offset or sample:
every record, code and spec below is invented.

Contract pinned here (`horsetrack.countries.japan.handicap.workouts`):

- Vocabularies: `CENTRES == ("miho", "ritto", "other")`,
  `COURSES == ("wood", "hill", "poly", "turf", "dirt", "other")` (wood chip, 坂路, polytrack, turf,
  dirt). `EFFORTS`, a tuple of distinct effort descriptors containing "other", and
  `HARD_EFFORTS`, a non-empty subset of it that leaves at least one effort that is neither hard nor
  "other". `N_RECENT == 3`. `FEATURE_COLUMNS == ("ff_best_vs_par_s", "ff_trend_s",
  "days_since_hard", "course_kinds_since_last_race")`.
- Hooks, all looked up as module attributes at call time (the tests monkeypatch them):
  `PAR_MIN_WORKOUTS` (int), `CYB_COUNT_COLUMNS` (the `jrdb_pre_race` columns holding CYB's
  per-course workout counts), and `cha_workouts(parsed: pandas.DataFrame) -> pandas.DataFrame`,
  the layout-specific step: parsed CHA records (`parse.parse_bytes("CHA", ...)`: `race_key`,
  `horse_no` and the spec's columns) -> one row per record with `race_key`, `horse_no` and the
  stored data columns below, codes mapped into the vocabularies, times in seconds.
- `build_workout_table() -> Path`: every CHA delivery under `raw_dir("jrdb")/CHA/` parsed with the
  spec `<data dir>/jrdb/specs/cha.csv`, passed through `cha_workouts`, joined to the store
  (`jrdb.join.join_store`: race key -> race_id and race_date, horse number -> horse_id), rows whose
  race or horse did not match dropped, written to `processed_dir("jra")/jrdb_workouts.parquet`.
  Stored columns (others allowed): `race_id, race_date, horse_no, horse_id` (the record's race and
  the horse in it), `workout_date, centre, course, furlongs, split_first_s, split_mid_s,
  final_furlong_s, effort, partner`.
- `published_at(race_date) -> datetime` (Ruling, the coder's): when the CHA/CYB records delivered
  for race day D count as published. Timezone-aware, never before (D - 1 day) 00:00 Asia/Tokyo
  and never after D 09:00 Asia/Tokyo (before the day's first post, so the race's own 本追切 is
  usable on race morning), increasing with D.
- `workout_table(race_id, *, as_of) -> polars.DataFrame`: one row per workout of each declared
  (not scratched) runner of the race, ordered by horse_no, then workout_date newest first. Columns
  `race_id` (the race asked for), `horse_no` (the runner's number in it), `horse_id`,
  `workout_race_id` (the record's race), `workout_date, centre, course, furlongs, split_first_s,
  split_mid_s, final_furlong_s, effort, partner` (as stored), `published_at`
  (= published_at(record's race_date), a Datetime in UTC). A runner without workouts has no row;
  every call has the same columns. A record counts when it is the horse's, its race_date is on or
  before the race day (a later race's record never counts, whatever `as_of`), its workout_date is
  not after the race day, and published_at(its race_date) is strictly before `as_of`. `as_of`
  must be timezone-aware (ValueError otherwise).
- Holdout: every store read goes through #114's `read_store_table` (`runners`, `jrdb_workouts`,
  `jrdb_pre_race`), so a holdout day's records are never read. The tests patch `is_holdout_day`
  on the holdout module, on v0's store and on `workouts`, as test_v0_store does.
- `workout_features(race_id, *, as_of, enabled=None) -> polars.DataFrame`: one row per declared
  runner, ordered by horse_no: `race_id, horse_no, horse_id`, then `FEATURE_COLUMNS`:
  - par(centre, course): the median final_furlong_s of every horse's records at that centre and
    course whose race_date is before the race day and that were published before `as_of`
    (holdout days unread); null with fewer than `PAR_MIN_WORKOUTS` such times. The race's own
    records, same-day records and later ones never enter a par.
  - recent: the runner's newest `N_RECENT` counted workouts (table rule above) that have a
    par-relative time, rel = final_furlong_s - par.
  - `ff_best_vs_par_s`: min rel over recent (null when recent is empty).
  - `ff_trend_s`: the least-squares slope of rel against 0, 1, 2 (oldest to newest) over recent,
    seconds per workout ((newest - oldest) / 2); null unless recent holds exactly 3.
  - `days_since_hard`: race day minus the newest counted workout_date whose effort is in
    `HARD_EFFORTS`, in days (Int64); null when there is none.
  - `course_kinds_since_last_race`: the number of course kinds used since the last race, the sum
    of the non-null `CYB_COUNT_COLUMNS` of the runner's own `jrdb_pre_race` row (CYB's per-course
    fields are presence flags, one per course kind, so this is not a workout count; it was named
    `workouts_since_last_race` before #241). Row (race_id, horse_no), counted only when
    published_at(race day) is before
    `as_of`; null when there is no row, every count is null, or it is not yet published (Int64).
- The flag: `jrdb_features_enabled()` reads `HORSETRACK_JRDB_FEATURES` through
  `core.env.get_setting` (process environment, else the repo-root `.env`): unset, "1", "true",
  "on", "yes" -> True; "0", "false", "off", "no" -> False, any case. `enabled=None` follows it;
  when off, `workout_features` returns `race_id, horse_no, horse_id` only, one row per declared
  runner, and reads no JRDB table.
- `write_logloss_report(*, scheme=None, out_dir=None) -> Path`: on development days only (store
  reads through the scoped reader), the v0 calibrated-market Win model (#114's `oof_predict`,
  default knots) against the same model with the workout features, both out of fold over
  `walk_forward(race dates, scheme=scheme)` (None: Japan's scheme), scored on the same races by
  the mean per-race Win log loss, -ln p(winner). JSON written under `paths.reports_dir()` (or
  `out_dir`; an `out_dir` inside the repository raises `paths.DataDirInsideRepo` and writes
  nothing). Keys: `baseline_logloss`, `workout_logloss`, `logloss_gain` (baseline minus
  workout), `n_races`, `features` (FEATURE_COLUMNS) and `folds`, one entry per fold with
  `test_start`, `test_end` (ISO dates), `n_races`, `baseline_logloss`, `workout_logloss`,
  `logloss_gain`. Overall figures are means over all scored races. Aggregates only: no race id,
  horse id or per-runner row anywhere in it.

Data: synthetic only, under tmp_path; HORSETRACK_DATA_DIR points there (monkeypatched). One test
is marked `local`: it runs the real `cha_workouts` over one real CHA delivery on the owner's data
dir and checks only vocabulary and plausibility, never a figure.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import json
import math
import random
import re
import statistics
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core import env, paths
from horsetrack.core.eval import holdout
from horsetrack.core.eval.folds import FoldScheme, walk_forward
from horsetrack.core.store import schemas
from horsetrack.core.store import tables as store_tables
from horsetrack.countries.japan.handicap import workouts
from horsetrack.countries.japan.jrdb import parse
from horsetrack.countries.japan.models.v0 import store as v0_store
from horsetrack.countries.japan.models.v0.calibration import oof_predict

REAL_CYB_COUNT_COLUMNS = workouts.CYB_COUNT_COLUMNS  # the coder's real mapping, before patching

DATA_VAR = "HORSETRACK_DATA_DIR"
FLAG = "HORSETRACK_JRDB_FEATURES"
JST = ZoneInfo("Asia/Tokyo")
UTC = timezone.utc
REPO = Path(__file__).resolve().parents[4]
FEATURES = ("ff_best_vs_par_s", "ff_trend_s", "days_since_hard", "course_kinds_since_last_race")
KEYS = ["race_id", "horse_no", "horse_id"]
TABLE_COLUMNS = [
    "race_id", "horse_no", "horse_id", "workout_race_id", "workout_date", "centre", "course",
    "furlongs", "split_first_s", "split_mid_s", "final_furlong_s", "effort", "partner",
    "published_at",
]  # fmt: skip
STORED_DATA = [
    "workout_date", "centre", "course", "furlongs", "split_first_s", "split_mid_s",
    "final_furlong_s", "effort", "partner",
]  # fmt: skip
WORKOUT_SCHEMA = pa.schema(
    [
        ("jurisdiction", pa.string()), ("race_id", pa.string()), ("race_date", pa.date32()),
        ("horse_no", pa.int64()), ("horse_id", pa.string()), ("workout_date", pa.date32()),
        ("centre", pa.string()), ("course", pa.string()), ("furlongs", pa.int64()),
        ("split_first_s", pa.float64()), ("split_mid_s", pa.float64()),
        ("final_furlong_s", pa.float64()), ("effort", pa.string()), ("partner", pa.string()),
    ]
)  # fmt: skip
CYB_COLUMNS = ("cyb_count_x", "cyb_count_y")  # synthetic stand-ins for CYB's per-course counts
PAR_MIN = 3
VENUES = ["TOK", "NAK", "KYO", "HSN", "SAP"]


def rid(day: date, venue: str, race_no: int) -> str:
    return f"jra:{day.isoformat()}:{venue}:{race_no:02d}"


def day_of(race_id: str) -> date:
    return date.fromisoformat(race_id.split(":")[1])


def at_jst(day: date, hh: int, mm: int = 0, ss: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm, ss), tzinfo=JST)


def hard() -> str:
    return sorted(workouts.HARD_EFFORTS)[0]


def easy() -> str:
    return next(e for e in workouts.EFFORTS if e not in workouts.HARD_EFFORTS and e != "other")


# ---------------------------------------------------------------------------------------------
# Writing a synthetic data dir
# ---------------------------------------------------------------------------------------------


def point_data_dir(tmp_path: Path, monkeypatch) -> Path:
    data = (tmp_path / "data").resolve()
    data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(DATA_VAR, str(data))
    return data


def write_store_table(name: str, rows: list[dict]) -> None:
    """A store table with every declared column (None where the test has no value), typed."""
    spec = schemas.TABLES[name].columns
    schema = pa.schema([(c, store_tables.ARROW_TYPES[k]) for c, k in spec.items()])
    shuffled = list(rows)
    random.Random(11).shuffle(shuffled)  # nothing may rely on the file's row order
    columns = {c: [r.get(c) for r in shuffled] for c in spec}
    path = paths.processed_dir("jra") / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns, schema=schema), path)


def write_table(name: str, rows: list[dict], schema: pa.Schema) -> Path:
    shuffled = list(rows)
    random.Random(13).shuffle(shuffled)
    path = paths.processed_dir("jra") / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(shuffled, schema=schema), path)
    return path


def pre_race_schema() -> pa.Schema:
    keys = schemas.TABLES["jrdb_pre_race"].columns
    fields = [(c, store_tables.ARROW_TYPES[k]) for c, k in keys.items()]
    return pa.schema(fields + [(c, pa.float64()) for c in CYB_COLUMNS])


@dataclass(frozen=True)
class W:
    """One stored CHA record: a workout of `horse_id` before the race `race_id`."""

    horse_id: str
    race_id: str
    horse_no: int  # the horse's number in race_id
    workout_date: date
    centre: str
    course: str
    ff: float | None
    effort: str  # "H" hard, "E" easy: resolved against the module's vocabulary when written
    partner: str | None = None
    furlongs: int | None = 4
    first: float | None = None
    mid: float | None = None

    @property
    def race_date(self) -> date:
        return day_of(self.race_id)

    def effort_value(self) -> str:
        return hard() if self.effort == "H" else easy()

    def stored(self) -> dict:
        return {
            "jurisdiction": "jra", "race_id": self.race_id, "race_date": self.race_date,
            "horse_no": self.horse_no, "horse_id": self.horse_id,
            "workout_date": self.workout_date, "centre": self.centre, "course": self.course,
            "furlongs": self.furlongs, "split_first_s": self.first, "split_mid_s": self.mid,
            "final_furlong_s": self.ff, "effort": self.effort_value(), "partner": self.partner,
        }  # fmt: skip


@dataclass
class World:
    runners: list[dict] = field(default_factory=list)  # store runners rows
    races: list[dict] = field(default_factory=list)
    records: list[W] = field(default_factory=list)
    cyb: list[dict] = field(default_factory=list)  # race_id, horse_no, horse_id, counts

    def race(self, race_id: str, entries: dict[str, int], scratched=frozenset(), fields=None):
        day = day_of(race_id)
        _, _, venue, no = race_id.split(":")
        keys = {
            "jurisdiction": "jra", "race_id": race_id, "meeting_id": f"jra:{day}:{venue}",
            "race_date": day, "venue": venue, "race_no": int(no),
        }  # fmt: skip
        self.races.append({**keys, "status": "run"})
        for horse_id, horse_no in entries.items():
            row = {**keys, "horse_id": horse_id, "horse_no": horse_no}
            row["is_scratched"] = horse_id in scratched
            row["dead_heat"] = False
            row.update((fields or {}).get(horse_id, {}))
            self.runners.append(row)

    def add(self, horse_id, race_id, horse_no, workout_date, centre, course, ff, effort, **kw):
        if any(w.horse_id == horse_id and w.race_id == race_id for w in self.records):
            raise AssertionError(f"fixture bug: two records of {horse_id} in {race_id}")
        self.records.append(
            W(horse_id, race_id, horse_no, workout_date, centre, course, ff, effort, **kw)
        )

    def count(self, race_id, horse_no, horse_id, x, y):
        self.cyb.append(
            {"race_id": race_id, "horse_no": horse_no, "horse_id": horse_id, "counts": (x, y)}
        )

    def write(self) -> None:
        write_store_table("races", self.races)
        write_store_table("runners", self.runners)
        write_table("jrdb_workouts", [w.stored() for w in self.records], WORKOUT_SCHEMA)
        rows = []
        for c in self.cyb:
            row = {
                "jurisdiction": "jra", "race_id": c["race_id"], "race_date": day_of(c["race_id"]),
                "horse_no": c["horse_no"], "horse_id": c["horse_id"], "matched_by": "horse_no",
            }  # fmt: skip
            row.update(dict(zip(CYB_COLUMNS, c["counts"], strict=True)))
            rows.append(row)
        write_table("jrdb_pre_race", rows, pre_race_schema())


def patch_holdout(monkeypatch, days):
    """Mark exactly `days` as holdout wherever the reader (or workouts) looks is_holdout_day up."""
    marked = set(days)

    def fake(country, day):
        return day in marked

    monkeypatch.setattr(holdout, "is_holdout_day", fake)
    monkeypatch.setattr(v0_store, "is_holdout_day", fake, raising=False)
    monkeypatch.setattr(workouts, "is_holdout_day", fake, raising=False)


def no_dotenv(tmp_path: Path, monkeypatch, text: str | None = None) -> None:
    """The repo-root `.env` is a fresh one under tmp_path (absent unless `text` is given)."""
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    if text is not None:
        (root / ".env").write_text(text)
    monkeypatch.setattr(env, "repo_root", lambda: root)


@pytest.fixture(autouse=True)
def hooks(monkeypatch):
    """The flag is unset and the coder's tunable hooks take the tests' synthetic values."""
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.setattr(workouts, "PAR_MIN_WORKOUTS", PAR_MIN, raising=False)
    monkeypatch.setattr(workouts, "CYB_COUNT_COLUMNS", CYB_COLUMNS, raising=False)


# ---------------------------------------------------------------------------------------------
# The main synthetic world: race day T (a Sunday), target race TOK 11.
#   A (no 3): records for races on 03-02 (hard), 04-13, 05-18 and its own race; one for a race
#             two weeks after T, its workout dated after T (bait).
#   B (no 7): no record at all.
#   C (no 1): two records, one at a course whose par is too thin; no hard workout; no own record.
#   D (no 5): its own race's record (hard) and one for a race the day after T whose workout is
#             dated before T: published by that race's morning, still never a workout of T's race.
#   E (no 9): scratched.
# Par pools: filler horses' records before T; same-day and later records that would move every
# median if they leaked in.
# ---------------------------------------------------------------------------------------------

T = date(2025, 6, 15)
TARGET = rid(T, "TOK", 11)
SAME_DAY = rid(T, "TOK", 2)
ENTRIES = {"h:charlie": 1, "h:alpha": 3, "h:delta": 5, "h:bravo": 7, "h:echo": 9}
DECLARED = {h: n for h, n in ENTRIES.items() if h != "h:echo"}
AS_OF = at_jst(T, 10)  # race morning, after any allowed publication instant of T's records
EVE = at_jst(T - timedelta(days=2), 23, 59, 59)  # before any allowed publication of T's records
MID = at_jst(date(2025, 5, 20), 12)
LATE = at_jst(T + timedelta(days=30), 12)
A5_RACE = rid(date(2025, 6, 29), "TOK", 11)
D2_RACE = rid(date(2025, 6, 16), "HSN", 12)

POOL = [  # (centre, course, final furlong, race day) of filler horses' records before T
    ("miho", "hill", 12.0, date(2025, 5, 11)), ("miho", "hill", 12.2, date(2025, 6, 1)),
    ("miho", "hill", 12.6, date(2025, 4, 27)), ("miho", "hill", 13.0, date(2025, 6, 8)),
    ("miho", "wood", 11.9, date(2025, 5, 4)), ("miho", "wood", 12.3, date(2025, 6, 1)),
    ("miho", "wood", 12.5, date(2025, 5, 18)),
    ("ritto", "hill", 12.1, date(2025, 4, 20)), ("ritto", "hill", 12.5, date(2025, 5, 25)),
    ("ritto", "hill", 12.7, date(2025, 6, 8)),
    ("ritto", "wood", 11.8, date(2025, 5, 11)), ("ritto", "wood", 12.2, date(2025, 6, 1)),
    ("ritto", "wood", 12.4, date(2025, 4, 27)),
    ("ritto", "poly", 12.6, date(2025, 5, 18)),
]  # fmt: skip
BAIT_COURSES = [("miho", "hill"), ("miho", "wood"), ("ritto", "hill"), ("ritto", "wood")]


def build_main_world() -> World:
    w = World()
    w.race(TARGET, ENTRIES, scratched={"h:echo"})
    same = {f"h:same{i}": i + 1 for i in range(len(BAIT_COURSES))}
    w.race(SAME_DAY, same)

    w.add("h:alpha", rid(date(2025, 3, 2), "NAK", 9), 4, date(2025, 2, 26), "miho", "hill",
          12.8, "H", partner="ahead", furlongs=4, first=52.1, mid=38.5)  # fmt: skip
    w.add("h:alpha", rid(date(2025, 4, 13), "NAK", 10), 6, date(2025, 4, 9), "miho", "wood",
          12.1, "E", partner=None, furlongs=6, first=83.0, mid=67.2)  # fmt: skip
    w.add("h:alpha", rid(date(2025, 5, 18), "TOK", 10), 2, date(2025, 5, 14), "miho", "hill",
          12.4, "E", partner="level", furlongs=4, first=53.0, mid=None)  # fmt: skip
    w.add("h:alpha", TARGET, 3, date(2025, 6, 11), "miho", "wood", 11.7, "E",
          partner="behind", furlongs=5, first=68.4, mid=53.9)  # fmt: skip
    w.add("h:alpha", A5_RACE, 8, date(2025, 6, 25), "miho", "hill", 10.9, "H")  # later race
    w.add("h:charlie", rid(date(2025, 5, 4), "KYO", 7), 5, date(2025, 4, 30), "ritto", "poly",
          12.9, "E", furlongs=5, first=66.1, mid=51.0)  # fmt: skip
    w.add("h:charlie", rid(date(2025, 5, 25), "KYO", 8), 2, date(2025, 5, 21), "ritto", "wood",
          12.0, "E", partner="ahead", furlongs=6, first=82.2, mid=66.0)  # fmt: skip
    w.add("h:delta", TARGET, 5, date(2025, 6, 12), "ritto", "hill", 12.3, "H", partner="level")
    w.add("h:delta", D2_RACE, 1, date(2025, 6, 13), "ritto", "hill", 11.0, "H")  # the day after
    w.add("h:echo", TARGET, 9, date(2025, 6, 11), "ritto", "wood", 11.5, "H")  # scratched

    for i, (centre, course, ff, day) in enumerate(POOL):
        race_id = rid(day, VENUES[i % 5], 1 + i % 12)
        w.add(f"h:pool{i:02d}", race_id, 1 + i % 10, day - timedelta(days=4), centre, course, ff,
              "H" if i % 2 else "E", partner="level" if i % 3 else None)  # fmt: skip
    for i, (centre, course) in enumerate(BAIT_COURSES):
        w.add(f"h:same{i}", SAME_DAY, i + 1, T - timedelta(days=4), centre, course, 9.5, "H")
        w.add(f"h:late{i}", rid(date(2025, 6, 22), "TOK", 3), i + 1, date(2025, 6, 18), centre,
              course, 9.0, "H")  # fmt: skip

    w.count(TARGET, 3, "h:alpha", 2, 1)
    w.count(TARGET, 1, "h:charlie", None, 4)
    w.count(TARGET, 5, "h:delta", None, None)
    w.count(TARGET, 9, "h:echo", 5, 5)
    w.count(rid(date(2025, 5, 18), "TOK", 10), 2, "h:alpha", 7, 7)  # an earlier race: not T's
    w.count(A5_RACE, 8, "h:alpha", 9, 9)  # a later race
    return w


@pytest.fixture
def main_world(tmp_path, monkeypatch) -> World:
    point_data_dir(tmp_path, monkeypatch)
    w = build_main_world()
    w.write()
    return w


# ---------------------------------------------------------------------------------------------
# The reference: the contract's rules applied to the fixture's own records
# ---------------------------------------------------------------------------------------------


def visible(race_date: date, as_of: datetime, hidden, exact=None) -> bool:
    """Whether a record delivered for `race_date` counts as published before `as_of`, for any
    rule inside the allowed window [(D-1) 00:00, D 09:00] JST; `exact` pins the instant."""
    if race_date in hidden:
        return False
    if exact and race_date in exact:
        return exact[race_date] < as_of
    lo = datetime.combine(race_date - timedelta(days=1), time(0), tzinfo=JST)
    hi = datetime.combine(race_date, time(9), tzinfo=JST)
    if as_of > hi:
        return True
    if as_of <= lo:
        return False
    raise AssertionError(f"fixture bug: {as_of} is inside {race_date}'s publication window")


class Expect:
    def __init__(self, world, race_id, *, as_of, hidden=frozenset(), exact=None, par_min=PAR_MIN):
        self.w, self.race_id, self.day = world, race_id, day_of(race_id)
        self.as_of, self.hidden, self.exact = as_of, set(hidden), exact
        self.par_min = par_min

    def _visible(self, race_date):
        return visible(race_date, self.as_of, self.hidden, self.exact)

    def history(self, horse_id) -> list[W]:
        out = []
        for r in self.w.records:
            if r.horse_id != horse_id or r.race_date > self.day or r.workout_date > self.day:
                continue
            if r.race_date == self.day:
                assert r.race_id == self.race_id, "fixture bug: a same-day record of another race"
            if self._visible(r.race_date):
                out.append(r)
        return sorted(out, key=lambda r: r.workout_date, reverse=True)

    def par(self, centre, course) -> float | None:
        times = [
            r.ff
            for r in self.w.records
            if r.centre == centre
            and r.course == course
            and r.ff is not None
            and r.race_date < self.day
            and self._visible(r.race_date)
        ]
        return statistics.median(times) if len(times) >= self.par_min else None

    def table(self) -> list[dict]:
        rows = []
        for horse_id, horse_no in sorted(DECLARED.items(), key=lambda kv: kv[1]):
            for r in self.history(horse_id):
                stored = r.stored()
                rows.append(
                    {
                        "race_id": self.race_id, "horse_no": horse_no, "horse_id": horse_id,
                        "workout_race_id": r.race_id, **{c: stored[c] for c in STORED_DATA},
                    }
                )  # fmt: skip
        return rows

    def features(self, horse_id, horse_no) -> dict:
        hist = self.history(horse_id)
        rel = []
        for r in hist:
            par = self.par(r.centre, r.course)
            if r.ff is not None and par is not None:
                rel.append(r.ff - par)
        recent = rel[:3]  # newest first
        hard_dates = [r.workout_date for r in hist if r.effort == "H"]
        count = None
        for c in self.w.cyb:
            if (c["race_id"], c["horse_no"]) == (self.race_id, horse_no):
                if self._visible(self.day):
                    known = [v for v in c["counts"] if v is not None]
                    count = int(sum(known)) if known else None
        return {
            "ff_best_vs_par_s": min(recent) if recent else None,
            "ff_trend_s": (recent[0] - recent[2]) / 2 if len(recent) == 3 else None,
            "days_since_hard": (self.day - max(hard_dates)).days if hard_dates else None,
            "course_kinds_since_last_race": count,
        }


def table_rows(frame: pl.DataFrame) -> list[dict]:
    missing = [c for c in TABLE_COLUMNS if c not in frame.columns]
    assert not missing, f"workout_table lacks columns {missing}"
    return frame.select([c for c in TABLE_COLUMNS if c != "published_at"]).to_dicts()


def feature_rows(frame: pl.DataFrame) -> dict[str, dict]:
    missing = [c for c in (*KEYS, *FEATURES) if c not in frame.columns]
    assert not missing, f"workout_features lacks columns {missing}"
    assert frame["horse_no"].to_list() == sorted(DECLARED.values())
    return {r["horse_id"]: {c: r[c] for c in FEATURES} for r in frame.to_dicts()}


def assert_features(got: dict, want: dict, label) -> None:
    for c in FEATURES:
        g, x = got[c], want[c]
        if x is None:
            assert g is None, (label, c, g)
        else:
            assert g is not None and g == pytest.approx(x, abs=1e-9), (label, c, g, x)


# ---------------------------------------------------------------------------------------------
# Vocabularies and constants
# ---------------------------------------------------------------------------------------------


def test_vocabularies_and_feature_names():
    assert tuple(workouts.CENTRES) == ("miho", "ritto", "other")
    assert tuple(workouts.COURSES) == ("wood", "hill", "poly", "turf", "dirt", "other")
    efforts = tuple(workouts.EFFORTS)
    assert "other" in efforts and len(set(efforts)) == len(efforts)
    assert all(isinstance(e, str) and e for e in efforts)
    hard_set = set(workouts.HARD_EFFORTS)
    assert hard_set and hard_set < set(efforts) - {"other"}
    assert workouts.N_RECENT == 3
    assert tuple(workouts.FEATURE_COLUMNS) == FEATURES


OLD_NAME = "workouts_since_last_race"
NEW_NAME = "course_kinds_since_last_race"


def test_old_feature_name_is_gone(main_world):
    """#241: the feature counts course kinds, so the misleading old name is retired."""
    assert NEW_NAME in workouts.FEATURE_COLUMNS
    assert OLD_NAME not in workouts.FEATURE_COLUMNS
    frame = workouts.workout_features(TARGET, as_of=AS_OF, enabled=True)
    assert NEW_NAME in frame.columns
    assert OLD_NAME not in frame.columns


REPO = Path(__file__).resolve().parents[4]


def test_ledger_and_model_card_use_the_new_name():
    """#241: name and definition agree in the ledger and the model card; none of the docs that
    describe the feature keeps the old name."""
    ledger = (REPO / "docs" / "ledger.md").read_text()
    card = (REPO / "src/horsetrack/countries/japan/MODEL_CARD.md").read_text()
    modeling = (REPO / "docs" / "modeling.md").read_text()
    assert NEW_NAME in ledger
    assert NEW_NAME in card
    for name, text in (("ledger", ledger), ("model card", card), ("modeling", modeling)):
        assert OLD_NAME not in text, f"{name} still uses {OLD_NAME}"


# ---------------------------------------------------------------------------------------------
# DoD: the as-of table excludes workouts published after the decision time (leakage)
# ---------------------------------------------------------------------------------------------

AS_OFS = {
    "race-morning": AS_OF,
    "race-morning-utc": AS_OF.astimezone(UTC),
    "eve": EVE,
    "mid-may": MID,
    "a-month-on": LATE,
}


@pytest.mark.parametrize("as_of", list(AS_OFS.values()), ids=list(AS_OFS))
def test_workout_table_holds_each_runners_published_workouts(main_world, as_of):
    frame = workouts.workout_table(TARGET, as_of=as_of)
    assert isinstance(frame, pl.DataFrame)
    assert table_rows(frame) == Expect(main_world, TARGET, as_of=as_of).table()
    zone = frame.schema["published_at"]
    assert isinstance(zone, pl.Datetime) and zone.time_zone == "UTC"
    for row in frame.select("workout_race_id", "published_at").to_dicts():
        want = workouts.published_at(day_of(row["workout_race_id"]))
        assert row["published_at"] == want
        assert row["published_at"] < as_of


def test_the_cut_is_live(main_world):
    """Each as_of gives a different table: the cut is not a constant."""
    counts = {
        k: dict(workouts.workout_table(TARGET, as_of=v).group_by("horse_id").len().iter_rows())
        for k, v in (("eve", EVE), ("mid", MID), ("morning", AS_OF))
    }
    assert counts["mid"] == {"h:alpha": 3, "h:charlie": 1}
    assert counts["eve"] == {"h:alpha": 3, "h:charlie": 2}
    assert counts["morning"] == {"h:alpha": 4, "h:charlie": 2, "h:delta": 1}


@pytest.mark.parametrize(
    "day",
    [T, T - timedelta(days=1), date(2024, 12, 31), date(2025, 1, 1), date(2016, 2, 29),
     date(2026, 11, 28), date(2009, 7, 4)],
)  # fmt: skip
def test_published_at_is_after_the_eve_and_before_the_races(day):
    p = workouts.published_at(day)
    assert p.tzinfo is not None and p.utcoffset() is not None
    assert datetime.combine(day - timedelta(days=1), time(0), tzinfo=JST) <= p
    assert p <= datetime.combine(day, time(9), tzinfo=JST)
    assert workouts.published_at(day + timedelta(days=1)) > p


def test_publication_boundary_is_strictly_before(main_world):
    """At exactly published_at(T) the race's own records (and its CYB counts) are not yet in;
    one microsecond later they are."""
    p = workouts.published_at(T)
    for as_of, own in ((p, False), (p + timedelta(microseconds=1), True)):
        frame = workouts.workout_table(TARGET, as_of=as_of)
        expect = Expect(main_world, TARGET, as_of=as_of, exact={T: p})
        assert table_rows(frame) == expect.table()
        assert (TARGET in frame["workout_race_id"].to_list()) is own
        got = feature_rows(workouts.workout_features(TARGET, as_of=as_of, enabled=True))
        for horse_id, horse_no in DECLARED.items():
            assert_features(got[horse_id], expect.features(horse_id, horse_no), (as_of, horse_id))
        assert (got["h:alpha"]["course_kinds_since_last_race"] is not None) is own


@pytest.mark.parametrize("as_of", [AS_OF, LATE, at_jst(T + timedelta(days=400), 12)])
def test_later_races_and_workouts_after_the_race_never_appear(main_world, as_of):
    """A's record for a race two weeks on (workout after T), D's record for the race the day
    after T (workout before T, published by that race's morning) and every filler's later record
    are in the store and, with a late as_of, published: none is a workout of T's race, and none
    moves a par or a feature."""
    frame = workouts.workout_table(TARGET, as_of=as_of)
    assert {A5_RACE, D2_RACE}.isdisjoint(frame["workout_race_id"].to_list())
    assert all(day_of(r) <= T for r in frame["workout_race_id"].to_list())
    assert all(d <= T for d in frame["workout_date"].to_list())
    assert table_rows(frame) == Expect(main_world, TARGET, as_of=AS_OF).table()
    got = feature_rows(workouts.workout_features(TARGET, as_of=as_of, enabled=True))
    assert got == feature_rows(workouts.workout_features(TARGET, as_of=AS_OF, enabled=True))


def test_a_race_without_workouts_has_the_same_columns(main_world):
    frame = workouts.workout_table(SAME_DAY, as_of=EVE)
    assert frame.height == 0
    assert set(TABLE_COLUMNS) <= set(frame.columns)
    full = workouts.workout_table(TARGET, as_of=AS_OF)
    assert frame.columns == full.columns


@pytest.mark.parametrize("call", ["workout_table", "workout_features"])
def test_a_naive_as_of_is_refused(main_world, call):
    with pytest.raises(ValueError):
        getattr(workouts, call)(TARGET, as_of=datetime(2025, 6, 15, 10, 0))


# ---------------------------------------------------------------------------------------------
# The features: hand-computed values, then the reference over every as_of
# ---------------------------------------------------------------------------------------------

HAND = {  # at AS_OF: pars miho hill 12.5, miho wood 12.2, ritto hill 12.5, ritto wood 12.1,
    # ritto poly none (2 times < PAR_MIN); 02-26 to 06-15 is 109 days
    "h:alpha": {"ff_best_vs_par_s": -0.5, "ff_trend_s": -0.2, "days_since_hard": 109,
                "course_kinds_since_last_race": 3},
    "h:bravo": dict.fromkeys(FEATURES),
    "h:charlie": {"ff_best_vs_par_s": -0.1, "ff_trend_s": None, "days_since_hard": None,
                  "course_kinds_since_last_race": 4},
    "h:delta": {"ff_best_vs_par_s": -0.2, "ff_trend_s": None, "days_since_hard": 3,
                "course_kinds_since_last_race": None},
}  # fmt: skip


def test_features_by_hand(main_world):
    for horse_id, horse_no in DECLARED.items():  # the reference agrees with the hand values
        assert_features(Expect(main_world, TARGET, as_of=AS_OF).features(horse_id, horse_no),
                        HAND[horse_id], ("reference", horse_id))  # fmt: skip
    frame = workouts.workout_features(TARGET, as_of=AS_OF, enabled=True)
    assert isinstance(frame, pl.DataFrame)
    assert frame["race_id"].to_list() == [TARGET] * len(DECLARED)
    assert dict(zip(frame["horse_id"], frame["horse_no"], strict=True)) == DECLARED
    for c in ("ff_best_vs_par_s", "ff_trend_s"):
        assert frame.schema[c].is_float(), c
    for c in ("days_since_hard", "course_kinds_since_last_race"):
        assert frame.schema[c].is_integer(), c
    got = feature_rows(frame)
    for horse_id in DECLARED:
        assert_features(got[horse_id], HAND[horse_id], horse_id)


@pytest.mark.parametrize("as_of", list(AS_OFS.values()), ids=list(AS_OFS))
def test_features_follow_the_as_of_rules(main_world, as_of):
    got = feature_rows(workouts.workout_features(TARGET, as_of=as_of, enabled=True))
    expect = Expect(main_world, TARGET, as_of=as_of)
    for horse_id, horse_no in DECLARED.items():
        assert_features(got[horse_id], expect.features(horse_id, horse_no), horse_id)


PAR_MIN_HAND = {  # (best, trend); pool sizes: miho hill 6, miho wood 4, ritto hill 3,
    # ritto wood 4, ritto poly 2 (C's poly workout gets a par of 12.75 at 2: rel 0.15)
    2: {"h:alpha": (-0.5, -0.2), "h:charlie": (-0.1, None), "h:delta": (-0.2, None)},
    4: {"h:alpha": (-0.5, -0.2), "h:charlie": (-0.1, None), "h:delta": (None, None)},
    5: {"h:alpha": (-0.1, None), "h:charlie": (None, None), "h:delta": (None, None)},
    7: {"h:alpha": (None, None), "h:charlie": (None, None), "h:delta": (None, None)},
}
PAR_FEATURES = ("ff_best_vs_par_s", "ff_trend_s")


@pytest.mark.parametrize("par_min", sorted(PAR_MIN_HAND))
def test_a_par_needs_par_min_workouts(main_world, monkeypatch, par_min):
    """A course with fewer than PAR_MIN_WORKOUTS times before the race has no par, so its
    workouts drop out of best and trend."""
    monkeypatch.setattr(workouts, "PAR_MIN_WORKOUTS", par_min)
    got = feature_rows(workouts.workout_features(TARGET, as_of=AS_OF, enabled=True))
    expect = Expect(main_world, TARGET, as_of=AS_OF, par_min=par_min)
    for horse_id, hand in PAR_MIN_HAND[par_min].items():
        want = expect.features(horse_id, DECLARED[horse_id])
        hand_row = dict(zip(PAR_FEATURES, hand, strict=True))
        assert_features({**want, **hand_row}, want, ("reference", horse_id))
        assert_features(got[horse_id], want, horse_id)


# ---------------------------------------------------------------------------------------------
# Holdout: no holdout day's rows are read
# ---------------------------------------------------------------------------------------------


def test_a_holdout_day_is_never_read(main_world, monkeypatch):
    """05-18 marked holdout: A's record of that day and two pool records vanish. By hand: miho
    hill par 12.6, miho wood par 12.1; A's rel -0.4 (own), 0.0 (04-13), 0.2 (03-02)."""
    hidden = date(2025, 5, 18)
    patch_holdout(monkeypatch, [hidden])
    frame = workouts.workout_table(TARGET, as_of=AS_OF)
    assert hidden not in [day_of(r) for r in frame["workout_race_id"].to_list()]
    assert table_rows(frame) == Expect(main_world, TARGET, as_of=AS_OF, hidden={hidden}).table()
    got = feature_rows(workouts.workout_features(TARGET, as_of=AS_OF, enabled=True))
    want = {"ff_best_vs_par_s": -0.4, "ff_trend_s": -0.3, "days_since_hard": 109,
            "course_kinds_since_last_race": 3}  # fmt: skip
    assert_features(got["h:alpha"], want, "h:alpha")
    expect = Expect(main_world, TARGET, as_of=AS_OF, hidden={hidden})
    for horse_id, horse_no in DECLARED.items():
        assert_features(got[horse_id], expect.features(horse_id, horse_no), horse_id)


def test_the_races_own_holdout_day_reads_nothing(main_world, monkeypatch):
    patch_holdout(monkeypatch, [T])
    assert workouts.workout_table(TARGET, as_of=AS_OF).height == 0
    assert workouts.workout_features(TARGET, as_of=AS_OF, enabled=True).height == 0


def test_committed_holdout_days_are_never_read(tmp_path, monkeypatch):
    """Unpatched: a record of a day #71's committed rule marks holdout (2026-11-28 on) is not a
    workout of a later development race."""
    start = date(2026, 11, 28)
    hold = next(
        start + timedelta(days=k)
        for k in range(60)
        if holdout.is_holdout_day("japan", start + timedelta(days=k))
    )
    race_day = next(
        hold + timedelta(days=k)
        for k in range(1, 60)
        if not holdout.is_holdout_day("japan", hold + timedelta(days=k))
    )
    dev = date(2026, 11, 21)
    assert not holdout.is_holdout_day("japan", dev)
    point_data_dir(tmp_path, monkeypatch)
    target = rid(race_day, "TOK", 11)
    w = World()
    w.race(target, {"h:golf": 2, "h:hotel": 4})
    w.add("h:golf", rid(dev, "TOK", 5), 3, dev - timedelta(days=4), "miho", "wood", 12.0, "E")
    w.add("h:golf", rid(hold, "TOK", 6), 1, hold - timedelta(days=4), "miho", "wood", 11.0, "H")
    w.count(target, 2, "h:golf", 1, 1)
    w.write()
    frame = workouts.workout_table(target, as_of=at_jst(race_day, 10))
    assert frame["workout_race_id"].to_list() == [rid(dev, "TOK", 5)]
    got = workouts.workout_features(target, as_of=at_jst(race_day, 10), enabled=True)
    golf = got.filter(pl.col("horse_id") == "h:golf").to_dicts()[0]
    assert golf["days_since_hard"] is None


# ---------------------------------------------------------------------------------------------
# The opt-in flag: off means no JRDB read; on by default
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, on",
    [(None, True), ("1", True), ("true", True), ("on", True), ("yes", True), ("TRUE", True),
     ("0", False), ("false", False), ("off", False), ("no", False), ("FALSE", False),
     ("Off", False)],
)  # fmt: skip
def test_flag_values(tmp_path, monkeypatch, value, on):
    no_dotenv(tmp_path, monkeypatch)
    if value is not None:
        monkeypatch.setenv(FLAG, value)
    assert workouts.jrdb_features_enabled() is on


@pytest.mark.parametrize("text, on", [(f"{FLAG}=0\n", False), (f"{FLAG}=1\n", True)])
def test_flag_from_dotenv(tmp_path, monkeypatch, text, on):
    no_dotenv(tmp_path, monkeypatch, text)
    assert workouts.jrdb_features_enabled() is on


def test_features_are_on_by_default(main_world, tmp_path, monkeypatch):
    no_dotenv(tmp_path, monkeypatch)
    got = feature_rows(workouts.workout_features(TARGET, as_of=AS_OF))
    for horse_id in DECLARED:
        assert_features(got[horse_id], HAND[horse_id], horse_id)


def corrupt_jrdb_tables() -> None:
    for name in ("jrdb_workouts", "jrdb_pre_race"):
        (paths.processed_dir("jra") / f"{name}.parquet").write_bytes(b"not a parquet file")


@pytest.mark.parametrize("how", ["argument", "env", "dotenv"])
def test_flag_off_reads_no_jrdb_table(main_world, tmp_path, monkeypatch, how):
    """With the JRDB tables unreadable, the off switch still returns the runners' keys."""
    corrupt_jrdb_tables()
    no_dotenv(tmp_path, monkeypatch, f"{FLAG}=off\n" if how == "dotenv" else None)
    kwargs = {}
    if how == "argument":
        kwargs["enabled"] = False
    elif how == "env":
        monkeypatch.setenv(FLAG, "0")
    frame = workouts.workout_features(TARGET, as_of=AS_OF, **kwargs)
    assert frame.columns == KEYS
    assert frame["horse_no"].to_list() == sorted(DECLARED.values())
    assert dict(zip(frame["horse_id"], frame["horse_no"], strict=True)) == DECLARED
    assert set(frame["race_id"].to_list()) == {TARGET}


def test_flag_on_does_read_the_jrdb_tables(main_world):
    """The guard for the test above: the corrupt files are read when the flag is on."""
    corrupt_jrdb_tables()
    with pytest.raises(Exception):  # noqa: B017 - any failure to read the corrupt file
        workouts.workout_features(TARGET, as_of=AS_OF, enabled=True)


# ---------------------------------------------------------------------------------------------
# The build: CHA deliveries -> jrdb_workouts, joined to the store
# ---------------------------------------------------------------------------------------------

VENUE_NO = {"KYO": "08", "TOK": "05"}
BUILD_LAYOUT = [  # invented: the race key parts (parse's names), then synthetic fields
    ("venue_code", 2, "skip"), ("year", 2, "skip"), ("kai", 1, "skip"), ("nichi", 1, "skip"),
    ("race_no", 2, "skip"), ("horse_no", 2, "num"), ("syn_day", 8, "date"),
    ("syn_place", 2, "code"), ("syn_last", 3, "num"), ("eol", 2, "skip"),
]  # fmt: skip
SYN_PLACES = {"01": ("miho", "hill"), "02": ("ritto", "wood")}


@dataclass(frozen=True)
class BuildRace:
    venue: str
    day: date
    kai: int
    nichi: int
    no: int
    horses: dict[int, str]  # horse_no -> store horse_id
    printed: tuple[int, ...]  # the horse numbers the CHA file prints
    in_store: bool

    @property
    def race_id(self) -> str:
        return rid(self.day, self.venue, self.no)

    @property
    def jrdb_key(self) -> str:
        return (
            f"{VENUE_NO[self.venue]}{self.day.year % 100:02d}{self.kai}{self.nichi:x}{self.no:02d}"
        )

    @property
    def source_key(self) -> str:
        return f"{self.day.year}{VENUE_NO[self.venue]}{self.kai:02d}{self.nichi:02d}{self.no:02d}"


B1 = BuildRace("KYO", date(2025, 5, 18), 3, 6, 1,
               {1: "2021100001", 2: "2021100002", 3: "2021100003"}, (1, 2, 3, 4), True)  # fmt: skip
B2 = BuildRace("TOK", date(2025, 6, 15), 3, 10, 11,
               {1: "2021100010", 5: "2021100002", 6: "2021100011"}, (1, 5, 6), True)  # fmt: skip
B3 = BuildRace("TOK", date(2025, 6, 15), 3, 10, 12, {}, (1, 2), False)
BUILD_RACES = (B1, B2, B3)


def syn_values(race: BuildRace, h: int) -> tuple[date, str, int]:
    return (
        race.day - timedelta(days=2 + h),
        "01" if (h + race.no) % 2 else "02",
        118 + 3 * h + race.no,
    )


def cha_record(race: BuildRace, h: int) -> bytes:
    day, place, last = syn_values(race, h)
    raw = race.jrdb_key.encode() + f"{h:2d}".encode() + f"{day:%Y%m%d}".encode()
    return raw + place.encode() + f"{last:3d}".encode()


def write_build_world(data: Path) -> None:
    specs = data / "jrdb" / "specs"
    specs.mkdir(parents=True, exist_ok=True)
    rows, pos = ["# synthetic: an invented layout for tests, not a JRDB spec"], 1
    rows.append("name,start,length,kind,added,label")
    for name, length, kind in BUILD_LAYOUT:
        rows.append(f"{name},{pos},{length},{kind},2020-01-01,synthetic {name}")
        pos += length
    (specs / "cha.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    races, runners = [], []
    for race in BUILD_RACES:
        if not race.in_store:
            continue
        keys = {
            "jurisdiction": "jra", "race_id": race.race_id, "race_date": race.day,
            "meeting_id": f"jra:{race.day}:{race.venue}", "venue": race.venue, "race_no": race.no,
        }  # fmt: skip
        races.append({**keys, "status": "run", "source_race_key": race.source_key})
        runners += [
            {**keys, "horse_id": hid, "horse_no": no, "is_scratched": False}
            for no, hid in race.horses.items()
        ]
    write_store_table("races", races)
    write_store_table("runners", runners)

    folder = paths.raw_dir("jrdb") / "CHA"
    folder.mkdir(parents=True, exist_ok=True)
    for day in sorted({r.day for r in BUILD_RACES}):
        name = f"CHA{day:%y%m%d}"
        body = b"".join(
            cha_record(r, h) + b"\r\n" for r in BUILD_RACES if r.day == day for h in r.printed
        )
        with zipfile.ZipFile(folder / f"{name}.zip", "w") as archive:
            archive.writestr(f"{name}.txt", body)


def fake_cha_workouts(parsed: pd.DataFrame) -> pd.DataFrame:
    """The tests' stand-in for the layout-specific step, over the invented layout."""
    places = parsed["syn_place"].map(SYN_PLACES)
    return pd.DataFrame(
        {
            "race_key": parsed["race_key"],
            "horse_no": parsed["horse_no"],
            "workout_date": parsed["syn_day"],
            "centre": [p[0] for p in places],
            "course": [p[1] for p in places],
            "furlongs": 4,
            "split_first_s": None,
            "split_mid_s": None,
            "final_furlong_s": parsed["syn_last"] / 10.0,
            "effort": easy(),
            "partner": None,
        }
    )


def test_build_joins_cha_records_to_the_store(tmp_path, monkeypatch):
    data = point_data_dir(tmp_path, monkeypatch)
    write_build_world(data)
    monkeypatch.setattr(workouts, "cha_workouts", fake_cha_workouts)

    path = Path(workouts.build_workout_table())

    assert path.resolve() == (paths.processed_dir("jra") / "jrdb_workouts.parquet").resolve()
    frame = pl.read_parquet(path)
    missing = [c for c in ["race_id", "race_date", "horse_no", "horse_id", *STORED_DATA]
               if c not in frame.columns]  # fmt: skip
    assert not missing, missing
    got = sorted(
        (r["race_id"], r["race_date"], r["horse_no"], r["horse_id"], r["workout_date"],
         r["centre"], r["course"], round(r["final_furlong_s"], 6))
        for r in frame.to_dicts()
    )  # fmt: skip
    want = []
    for race in BUILD_RACES:  # matched rows only: B3 is not in the store, B1's no 4 is no runner
        for h in race.printed:
            if not race.in_store or h not in race.horses:
                continue
            day, place, last = syn_values(race, h)
            centre, course = SYN_PLACES[place]
            want.append((race.race_id, race.day, h, race.horses[h], day, centre, course,
                         round(last / 10.0, 6)))  # fmt: skip
    assert got == sorted(want)
    assert len(want) == 6

    table = workouts.workout_table(B2.race_id, as_of=at_jst(B2.day, 10))
    by_horse = {
        no: table.filter(pl.col("horse_no") == no)["workout_race_id"].to_list() for no in (1, 5, 6)
    }
    assert by_horse == {1: [B2.race_id], 5: [B2.race_id, B1.race_id], 6: [B2.race_id]}


@pytest.mark.local
def test_real_cha_delivery_maps_into_the_vocabulary(real_fixtures):
    """The real `cha_workouts` over one real CHA delivery: vocabulary and plausibility only."""
    data = paths.data_dir()
    folder = paths.raw_dir("jrdb") / "CHA"
    if not (data / "jrdb" / "specs" / "cha.csv").is_file() or not folder.is_dir():
        pytest.skip("no CHA spec or deliveries on this data dir")
    before_2026 = re.compile(r"CHA_?20(0\d|1\d|2[0-5])\.zip")
    packs = sorted(p for p in folder.iterdir() if before_2026.fullmatch(p.name))
    if not packs:
        pytest.skip("no CHA year pack before 2026")
    name, body = sorted(parse.members(packs[-1]))[0]
    parsed = parse.parse_bytes("CHA", body)
    out = workouts.cha_workouts(parsed)
    assert len(out) == len(parsed) > 0
    assert {"race_key", "horse_no", *STORED_DATA} <= set(out.columns)
    centres, courses = set(out["centre"].dropna()), set(out["course"].dropna())
    assert centres <= set(workouts.CENTRES) and {"miho", "ritto"} <= centres
    assert courses <= set(workouts.COURSES) and len(courses - {"other"}) >= 2
    efforts = set(out["effort"].dropna())
    assert efforts <= set(workouts.EFFORTS) and efforts & set(workouts.HARD_EFFORTS)
    ff = pd.to_numeric(out["final_furlong_s"], errors="coerce").dropna()
    assert len(ff) > 0 and 10.0 <= ff.median() <= 15.0
    assert ff.between(9.0, 20.0).mean() > 0.95
    stem = Path(name).stem.upper()
    if re.fullmatch(r"CHA\d{6}", stem):
        delivered = datetime.strptime(stem[3:], "%y%m%d").date()
        days = pd.to_datetime(out["workout_date"], errors="coerce").dropna().dt.date
        assert len(days) > 0 and all(d <= delivered for d in days)
    pre_race = paths.processed_dir("jra") / "jrdb_pre_race.parquet"
    if pre_race.is_file():
        assert set(REAL_CYB_COUNT_COLUMNS) <= set(pq.read_schema(pre_race).names)


# ---------------------------------------------------------------------------------------------
# DoD: the log-loss gain report, on development days only, aggregates only, under the data dir
# ---------------------------------------------------------------------------------------------

REPORT_DAYS = [date(2025, m, d) for m in range(1, 6) for d in (4, 11, 18, 25)]
SCHEME = FoldScheme(
    unit="month", test_length=1, step=1, min_train=1, window="expanding", embargo_days=0,
    anchor_month=1,
)  # fmt: skip
FORBIDDEN_KEYS = {"race_id", "horse_id", "horse_no", "horse_name", "jrdb_horse_id", "race_key"}
RACE_ID = re.compile(r"jra:\d{4}-\d{2}-\d{2}:")


def build_report_world() -> World:
    rng = random.Random(137)
    horses = [f"h:r{i:02d}" for i in range(48)]
    w = World()
    for day in REPORT_DAYS:
        for race_no in range(1, 5):
            race_id = rid(day, "TOK", race_no)
            field_horses = rng.sample(horses, 8)
            strength = [rng.lognormvariate(0.0, 0.8) for _ in field_horses]
            total = sum(strength)
            q = [s / total for s in strength]
            winner = rng.choices(range(8), weights=q)[0]
            places = [p for p in range(2, 9)]
            rng.shuffle(places)
            runner = {}
            for i, horse_id in enumerate(field_horses):
                pos = 1 if i == winner else places.pop()
                runner[horse_id] = {
                    "win_odds": max(1.1, round(0.8 / q[i], 1)), "finish_pos": pos,
                    "placing_raw": str(pos),
                }  # fmt: skip
                w.add(horse_id, race_id, i + 1, day - timedelta(days=rng.randint(3, 6)),
                      rng.choice(["miho", "ritto"]), rng.choice(["hill", "wood"]),
                      round(rng.uniform(11.6, 13.2) - 0.15 * math.log(strength[i]), 1),
                      rng.choice(["H", "E"]), partner=rng.choice(["ahead", "level", None]),
                      furlongs=rng.choice([4, 5, 6]))  # fmt: skip
                w.count(race_id, i + 1, horse_id, rng.randint(0, 3), rng.randint(0, 3))
            w.race(race_id, {h: i + 1 for i, h in enumerate(field_horses)}, fields=runner)
    return w


def expected_report(w: World, hidden=frozenset()) -> dict:
    """The baseline from #114's own out-of-fold run on the same races and folds."""
    rows = [r for r in w.runners if r["race_date"] not in hidden]
    frame = pd.DataFrame(
        {
            "race_id": [r["race_id"] for r in rows],
            "race_date": pd.to_datetime([r["race_date"] for r in rows]),
            "horse_no": [r["horse_no"] for r in rows],
            "win_odds": [r["win_odds"] for r in rows],
            "won": [r["finish_pos"] == 1 for r in rows],
        }
    ).sort_values(["race_date", "race_id", "horse_no"], ignore_index=True)
    folds = walk_forward(frame["race_date"], scheme=SCHEME)
    oof = oof_predict(frame, folds=folds).merge(
        frame[["race_id", "horse_no", "won"]], on=["race_id", "horse_no"]
    )
    winners = oof[oof["won"]]
    losses = {r.race_id: -math.log(r.p_cal) for r in winners.itertuples()}
    per_fold = []
    for k, fold in enumerate(folds):
        races = set(winners.loc[winners["fold"] == k, "race_id"])
        per_fold.append(
            {
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "n_races": len(races),
                "baseline_logloss": statistics.fmean(losses[r] for r in races),
            }
        )
    return {
        "n_races": len(losses),
        "baseline_logloss": statistics.fmean(losses.values()),
        "folds": per_fold,
    }


def assert_aggregates_only(node, horse_ids) -> None:
    if isinstance(node, dict):
        assert len(node) < 40, "a mapping as long as a per-race table"
        for key, value in node.items():
            assert key not in FORBIDDEN_KEYS, key
            assert_aggregates_only(value, horse_ids)
    elif isinstance(node, list):
        assert len(node) < 40, "a list as long as a per-race table"
        for value in node:
            assert_aggregates_only(value, horse_ids)
    elif isinstance(node, str):
        assert not RACE_ID.search(node), node
        assert node not in horse_ids and "h:r" not in node, node


def check_report(path: Path, w: World, hidden=frozenset()) -> dict:
    report = json.loads(Path(path).read_text())
    want = expected_report(w, hidden)
    assert report["features"] == list(FEATURES)
    assert report["n_races"] == want["n_races"]
    assert report["baseline_logloss"] == pytest.approx(want["baseline_logloss"], rel=1e-6)
    assert len(report["folds"]) == len(want["folds"]) == 3
    total = 0
    for got, exp in zip(report["folds"], want["folds"], strict=True):
        for key in ("test_start", "test_end", "n_races"):
            assert got[key] == exp[key], key
        assert got["baseline_logloss"] == pytest.approx(exp["baseline_logloss"], rel=1e-6)
        assert math.isfinite(got["workout_logloss"]) and got["workout_logloss"] > 0
        assert got["logloss_gain"] == pytest.approx(
            got["baseline_logloss"] - got["workout_logloss"], abs=1e-9
        )
        total += got["n_races"]
    assert total == report["n_races"]
    pooled = sum(f["workout_logloss"] * f["n_races"] for f in report["folds"]) / total
    assert report["workout_logloss"] == pytest.approx(pooled, rel=1e-9)
    assert report["logloss_gain"] == pytest.approx(
        report["baseline_logloss"] - report["workout_logloss"], abs=1e-9
    )
    assert_aggregates_only(report, {r["horse_id"] for r in w.runners})
    return report


@pytest.fixture
def report_world(tmp_path, monkeypatch) -> World:
    point_data_dir(tmp_path, monkeypatch)
    w = build_report_world()
    w.write()
    return w


def test_report_is_written_under_the_data_dir_with_aggregates_only(report_world):
    path = Path(workouts.write_logloss_report(scheme=SCHEME)).resolve()
    assert path.is_file()
    assert path.is_relative_to(paths.reports_dir().resolve())
    assert not path.is_relative_to(REPO)
    check_report(path, report_world)


def test_report_reads_development_days_only(report_world, monkeypatch, tmp_path):
    hidden = date(2025, 4, 11)
    patch_holdout(monkeypatch, [hidden])
    out = tmp_path / "elsewhere"
    path = Path(workouts.write_logloss_report(scheme=SCHEME, out_dir=out)).resolve()
    assert path.is_relative_to(out.resolve())
    report = check_report(path, report_world, hidden={hidden})
    assert report["n_races"] == expected_report(report_world)["n_races"] - 4
    assert hidden.isoformat() not in path.read_text()


def test_report_refuses_an_out_dir_inside_the_repository(report_world):
    inside = REPO / "reports-137-must-not-exist"
    with pytest.raises(paths.DataDirInsideRepo):
        workouts.write_logloss_report(scheme=SCHEME, out_dir=inside)
    assert not inside.exists()
