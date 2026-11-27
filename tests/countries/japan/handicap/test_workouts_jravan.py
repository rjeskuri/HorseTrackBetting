"""JRA-VAN 坂路 (HC) and ウッドチップ (WC) workouts as of decision time, and #137's workout
features from this source (ticket #229).

Source and licence: JRA-VAN Data Lab, private use in the owner's own programs only (D21): the
table and anything built on it stay under the data dir. Nothing in this file is a JRA-VAN figure
or layout: the HC and WC sections below use the workbook's item names (as the other jravan tests
do) with invented widths, order and companions, and every record is made up.

Contract pinned here (`horsetrack.countries.japan.handicap.workouts_jravan`, beside #137's
`workouts`):

- Constants: `N_RECENT == 3`; `FEATURE_COLUMNS == ("ff_best_vs_par_s", "ff_trend_s",
  "days_since_hard")`; `PAR_MIN_WORKOUTS` (int), looked up as a module attribute at call time
  (the tests monkeypatch it).
- `published_at(workout_date) -> datetime` (Ruling, see below): a workout counts as published
  at the end of its training day, (workout_date + 1 day) 00:00 Asia/Tokyo. It does not depend on
  データ作成年月日: in the stored export that date is a later re-issue (a bulk date for old
  years), not the first publication.
- `build_workout_table(*, raw=None, out=None) -> records.BuildReport` (name `workouts`): `raw` is
  the export root (default `paths.raw_dir("jravan")`); it reads every `<raw>/stored/SLOP/HC/
  <YYYY>.jvd` and `<raw>/stored/WOOD/WC/<YYYY>.jvd` with the layouts `spec.record_spec("HC")` and
  `spec.record_spec("WC")`, keeps one version per key (`records.latest_versions` on the layout's
  key items トレセン区分, 調教年月日, 調教時刻, 血統登録番号; a key whose latest version has
  データ区分 `0` is a deletion and has no row) and writes `<out>/workouts.parquet` (`out` default
  `paths.processed_dir("jravan")`; an `out` inside the repository raises
  `paths.DataDirInsideRepo` and writes nothing). The same inputs give the same bytes. A record of
  the wrong width or type raises `records.RecordLengthError` / `records.RecordTypeError` and
  writes no table. The report holds counts only (no horse id). Stored columns (others allowed,
  e.g. the scoped reader's `race_date`, which would hold the workout date):
  - `horse_id`: 血統登録番号 as printed, the store's `horse_id` format (ten digits); a blank or
    all-zero id is null and the row is kept (it still enters a par, never a runner's history).
  - `workout_date`: 調教年月日; `workout_time`: 調教時刻 as `HH:MM`.
  - `centre`: トレセン区分 `0` miho, `1` ritto, anything else `other`; `course`: `hill` for HC,
    `wood` for WC.
  - times in seconds from tenths (`0523` -> 52.3, whatever the width); all zeros (no measure) or
    all nines (over the limit) is null. `furlongs`: the largest n whose n-furlong total
    (`nハロンタイム合計`, the workbook spells the 8-furlong one `8ロンタイム合計`) is not null,
    null when none is; `split_first_s`: that total; `split_mid_s`: the 3-furlong total
    (600M～0M); `final_furlong_s`: ラップタイム(200M～0M).
  - `effort` and `partner`: null (the source has neither).
- `workout_table(race_id, *, as_of) -> polars.DataFrame`: one row per counted workout of each
  declared (not scratched, store `runners` of `processed_dir("jra")`) runner of the race,
  ordered by horse_no, then workout_date and workout_time newest first. Columns `race_id`,
  `horse_no`, `horse_id`, the stored data columns above and `published_at` (a UTC Datetime,
  = published_at(workout_date)). A workout counts when it is the horse's (horse_id equal), its
  workout_date is before the race day (a race-day or later workout never counts, whatever
  `as_of`) and published_at(workout_date) is strictly before `as_of`. A runner without
  workouts has no row; every call has the same columns. A naive `as_of` raises ValueError.
- `workout_features(race_id, *, as_of) -> polars.DataFrame`: one row per declared runner,
  ordered by horse_no: `race_id, horse_no, horse_id`, then `FEATURE_COLUMNS`, computed as #137
  does: par(centre, course) the median final_furlong_s of every record (any horse, null ids
  included) with workout_date before the race day and published before `as_of`, null with
  fewer than `PAR_MIN_WORKOUTS`; recent the newest `N_RECENT` counted workouts with a par;
  `ff_best_vs_par_s` min(final - par) over recent; `ff_trend_s` (newest - oldest) / 2 when recent
  holds exactly 3; `days_since_hard` an integer column, always null (no effort in this source).
  It reads no JRDB table and ignores `HORSETRACK_JRDB_FEATURES`.
- Holdout: every store read goes through #114's `read_store_table` (the workouts by their
  workout date), so a workout dated on a holdout day is never read, and a holdout race day has
  no entries. The tests patch `is_holdout_day` on the holdout module, on v0's store and on the
  module, as test_workouts does.

Data: synthetic only, under tmp_path; HORSETRACK_DATA_DIR points there (monkeypatched). One test
is marked `local`: it builds one real year of the owner's export into tmp_path and checks only
vocabulary and plausibility, never a figure.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import importlib.util
import random
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core import paths
from horsetrack.core.eval import holdout
from horsetrack.core.store import schemas
from horsetrack.core.store import tables as store_tables
from horsetrack.countries.japan.handicap import workouts_jravan as wj
from horsetrack.countries.japan.jravan import records
from horsetrack.countries.japan.models.v0 import store as v0_store


def _workbook_helper():
    """tests/countries/japan/jravan_workbook.py, the shared synthetic-workbook writer."""
    name = "jravan_workbook"
    if name not in sys.modules:
        location = Path(__file__).resolve().parents[1] / f"{name}.py"
        loader = importlib.util.spec_from_file_location(name, location)
        module = importlib.util.module_from_spec(loader)
        sys.modules[name] = module
        loader.loader.exec_module(module)
    return sys.modules[name]


wb = _workbook_helper()

DATA_VAR = "HORSETRACK_DATA_DIR"
JRDB_FLAG = "HORSETRACK_JRDB_FEATURES"
JST = ZoneInfo("Asia/Tokyo")
UTC = timezone.utc
REPO = Path(__file__).resolve().parents[4]
FEATURES = ("ff_best_vs_par_s", "ff_trend_s", "days_since_hard")
KEYS = ["race_id", "horse_no", "horse_id"]
STORED_DATA = [
    "workout_date", "workout_time", "centre", "course", "furlongs", "split_first_s",
    "split_mid_s", "final_furlong_s", "effort", "partner",
]  # fmt: skip
TABLE_COLUMNS = [*KEYS, *STORED_DATA, "published_at"]
PAR_MIN = 3
CENTRE_CODES = {"0": "miho", "1": "ritto"}
FINAL_LAP = "ラップタイム(200M～0M)"


def rid(day: date, venue: str, race_no: int) -> str:
    return f"jra:{day.isoformat()}:{venue}:{race_no:02d}"


def day_of(race_id: str) -> date:
    return date.fromisoformat(race_id.split(":")[1])


def at_jst(day: date, hh: int, mm: int = 0, ss: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm, ss), tzinfo=JST)


def pub(workout_date: date) -> datetime:
    """The pinned publication instant: the end of the training day, JST."""
    return datetime.combine(workout_date + timedelta(days=1), time(0), tzinfo=JST)


# ---------------------------------------------------------------------------------------------
# Made-up HC and WC layouts: the workbook's item names, invented widths, order and companions
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Field:
    name: str
    width: int
    key: bool = False


def _total(n: int) -> str:
    stem = "ロンタイム合計" if n == 8 else "ハロンタイム合計"  # the workbook's own spelling
    return f"{n}{stem}({200 * n}M～0M)"


def _lap(n: int) -> str:
    return f"ラップタイム({200 * n}M～{200 * (n - 1)}M)"


HEAD = (
    Field("レコード種別ID", 2),
    Field("データ区分", 1),
    Field("データ作成年月日", 8),
    Field("トレセン区分", 1, key=True),
    Field("調教年月日", 8, key=True),
)
HC_LAYOUT = (
    *HEAD,
    Field("血統登録番号", 10, key=True),  # before 調教時刻 here, after it in WC
    Field("調教時刻", 4, key=True),
    *(f for n in (4, 3, 2) for f in (Field(_total(n), 5), Field(_lap(n), 4))),
    Field(FINAL_LAP, 4),
    Field("予備", 3),
    Field("レコード区切", 2),
)
WC_LAYOUT = (
    *HEAD,
    Field("調教時刻", 4, key=True),
    Field("血統登録番号", 10, key=True),
    Field("コース", 1),
    Field("馬場周り", 1),
    Field("予備", 2),
    *(f for n in range(10, 1, -1) for f in (Field(_total(n), 6), Field(_lap(n), 5))),
    Field(FINAL_LAP, 5),
    Field("レコード区切", 2),
)
LAYOUTS = {"HC": HC_LAYOUT, "WC": WC_LAYOUT}
FOLDERS = {"HC": ("stored", "SLOP", "HC"), "WC": ("stored", "WOOD", "WC")}
_FURLONGS = re.compile(r"^(\d+)(?:ハ)?ロンタイム合計")


def section(number: int, record_type: str, title: str, layout: tuple[Field, ...]) -> wb.Section:
    rows: list[wb.Row] = []
    pos = 1
    for no, f in enumerate(layout, start=1):
        if no == 1:
            rows.append(
                wb.item(1, f.name, 1, f.width, initial=record_type, desc=wb.id_desc(record_type))
            )
        else:
            rows.append(
                wb.item(
                    no, f.name, pos, f.width, key="○" if f.key else "", desc=f"{f.name}（架空）"
                )
            )
        pos += f.width
    return wb.Section(number, title, pos - 1, tuple(rows))


def write_workbook(data: Path) -> Path:
    other = wb.simple_section(1, "XQ", "架空その他", [("架空値", 4), ("血統登録番号", 10)])
    return wb.write_workbook(
        data / "specs" / "jravan" / wb.WORKBOOK_NAME,
        [
            other,
            section(2, "HC", "架空坂路", HC_LAYOUT),
            section(3, "WC", "架空ウッド", WC_LAYOUT),
        ],
        [wb.Table("2001", "架空コード", 1, (("0", "架空ア"), ("1", "架空イ")))],
    )


def record_length(record_type: str) -> int:
    return sum(f.width for f in LAYOUTS[record_type])


# ---------------------------------------------------------------------------------------------
# Synthetic workouts and their expected stored rows (derived here, never by the module)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Work:
    """One HC (course hill) or WC (course wood) record."""

    horse: str | None  # 血統登録番号 as printed; None is blank
    day: date  # 調教年月日
    centre: str  # トレセン区分 code
    course: str  # "hill" or "wood"
    ff: float | None  # the final furlong in seconds; None is written as zeros
    clock: str = "0700"  # 調教時刻 HHMM
    totals: tuple[tuple[int, float | None], ...] | None = None  # None: derived from ff
    created: date | None = None  # データ作成年月日 (default: three days after the workout)
    kubun: str = "1"
    nines: bool = False  # the final furlong written as all nines (over the limit)

    @property
    def record_type(self) -> str:
        return "HC" if self.course == "hill" else "WC"

    def made(self) -> date:
        return self.created or self.day + timedelta(days=3)

    def key(self) -> tuple:
        return (self.centre, self.day, self.clock, self.horse or "")

    def split_totals(self) -> dict[int, float | None]:
        if self.totals is not None:
            return dict(self.totals)
        span = (4, 3, 2) if self.course == "hill" else (6, 5, 4, 3, 2)
        base = 12.5 if self.ff is None else self.ff
        return {n: round(base * n + 0.8 * (n - 1), 1) for n in span}

    def horse_id(self) -> str | None:
        h = self.horse
        if h is None or not re.fullmatch(r"[0-9]{10}", h) or set(h) == {"0"}:
            return None
        return h

    def stored(self) -> dict:
        totals = {n: s for n, s in self.split_totals().items() if s is not None}
        far = max(totals) if totals else None
        return {
            "horse_id": self.horse_id(),
            "workout_date": self.day,
            "workout_time": f"{self.clock[:2]}:{self.clock[2:]}",
            "centre": CENTRE_CODES.get(self.centre, "other"),
            "course": self.course,
            "furlongs": far,
            "split_first_s": totals[far] if far is not None else None,
            "split_mid_s": totals.get(3),
            "final_furlong_s": None if self.nines else self.ff,
            "effort": None,
            "partner": None,
        }


def _tenths(seconds: float | None, width: int) -> bytes:
    if seconds is None:
        return b"0" * width
    return f"{round(seconds * 10):0{width}d}".encode("ascii")


def jvd(w: Work) -> bytes:
    """`w` as one cp932 record of its layout, CRLF included."""
    totals = w.split_totals()
    out = b""
    for f in LAYOUTS[w.record_type]:
        name = f.name
        if name == "レコード種別ID":
            part = w.record_type.encode("ascii")
        elif name == "データ区分":
            part = w.kubun.encode("ascii")
        elif name == "データ作成年月日":
            part = f"{w.made():%Y%m%d}".encode("ascii")
        elif name == "トレセン区分":
            part = w.centre.encode("ascii")
        elif name == "調教年月日":
            part = f"{w.day:%Y%m%d}".encode("ascii")
        elif name == "調教時刻":
            part = w.clock.encode("ascii")
        elif name == "血統登録番号":
            part = (w.horse or "").encode("ascii").ljust(f.width)
        elif name == "コース":
            part = b"2"
        elif name == "馬場周り":
            part = b"1"
        elif name == "予備":
            part = b" " * f.width
        elif name == "レコード区切":
            part = b"\r\n"
        elif name == FINAL_LAP:
            part = b"9" * f.width if w.nines else _tenths(w.ff, f.width)
        elif m := _FURLONGS.match(name):
            part = _tenths(totals.get(int(m.group(1))), f.width)
        else:  # the other laps: written, never stored
            part = _tenths(13.1, f.width)
        assert len(part) == f.width, (name, part)
        out += part
    return out


# ---------------------------------------------------------------------------------------------
# Writing a synthetic data dir
# ---------------------------------------------------------------------------------------------


def point_data_dir(tmp_path: Path, monkeypatch) -> Path:
    data = (tmp_path / "data").resolve()
    data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(DATA_VAR, str(data))
    return data


def write_store_table(name: str, rows: list[dict]) -> None:
    """A jra store table with every declared column (None where the test has no value)."""
    spec = schemas.TABLES[name].columns
    schema = pa.schema([(c, store_tables.ARROW_TYPES[k]) for c, k in spec.items()])
    shuffled = list(rows)
    random.Random(229).shuffle(shuffled)
    columns = {c: [r.get(c) for r in shuffled] for c in spec}
    path = paths.processed_dir("jra") / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns, schema=schema), path)


@dataclass
class World:
    works: list[Work] = field(default_factory=list)  # in file order
    races: list[dict] = field(default_factory=list)
    runners: list[dict] = field(default_factory=list)

    def race(self, race_id: str, entries: dict[str, int], scratched=frozenset()) -> None:
        day = day_of(race_id)
        _, _, venue, no = race_id.split(":")
        keys = {
            "jurisdiction": "jra", "race_id": race_id, "meeting_id": f"jra:{day}:{venue}",
            "race_date": day, "venue": venue, "race_no": int(no),
        }  # fmt: skip
        self.races.append({**keys, "status": "run"})
        for horse_id, horse_no in entries.items():
            self.runners.append(
                {**keys, "horse_id": horse_id, "horse_no": horse_no,
                 "is_scratched": horse_id in scratched, "dead_heat": False}
            )  # fmt: skip

    def kept(self) -> list[Work]:
        """One version per key, the latest データ作成年月日 (a tie: the later record); a kept
        deletion has no row."""
        latest: dict[tuple, Work] = {}
        for w in self.works:
            old = latest.get(w.key())
            if old is None or w.made() >= old.made():
                latest[w.key()] = w
        return [w for w in latest.values() if w.kubun != "0"]

    def write_inputs(self, data: Path) -> None:
        write_workbook(data)
        write_store_table("races", self.races)
        write_store_table("runners", self.runners)
        files: dict[tuple[str, int], list[bytes]] = {}
        for w in self.works:
            files.setdefault((w.record_type, w.day.year), []).append(jvd(w))
        for (record_type, year), recs in files.items():
            folder = paths.raw_dir("jravan").joinpath(*FOLDERS[record_type])
            wb.write_jvd(folder / f"{year}.jvd", recs)


def patch_holdout(monkeypatch, days) -> None:
    marked = set(days)

    def fake(country, day):
        return day in marked

    monkeypatch.setattr(holdout, "is_holdout_day", fake)
    monkeypatch.setattr(v0_store, "is_holdout_day", fake, raising=False)
    monkeypatch.setattr(wj, "is_holdout_day", fake, raising=False)


@pytest.fixture(autouse=True)
def hooks(monkeypatch):
    monkeypatch.delenv(JRDB_FLAG, raising=False)
    monkeypatch.setattr(wj, "PAR_MIN_WORKOUTS", PAR_MIN, raising=False)


# ---------------------------------------------------------------------------------------------
# The main synthetic world: race day T (a Sunday), target race TOK 11.
#   alpha (no 3, Miho): workouts 02-26 hill, 04-09 wood (corrected: the later version is the
#     one), 05-14 hill 06:30 and wood 08:10 (two the same day), 06-11 wood; one on race day and
#     one after it (never counted).
#   bravo (no 7): no workout.  charlie (no 1, Ritto): 04-30 wood (thin par), 05-21 hill.
#   delta (no 5, Ritto): 06-12 hill and 06-14 hill (published at T 00:00 JST: in on race
#     morning, out the evening before).  echo (no 9): scratched.
# Par pools: filler horses before T (one with a blank id, one all zeros, one in the 2024 file, one
# deleted), and same-day and later workouts that would move every median if they leaked in.
# ---------------------------------------------------------------------------------------------

T = date(2025, 6, 15)
TARGET = rid(T, "TOK", 11)
EMPTY_RACE = rid(T, "TOK", 2)  # its runners have no workout at all
ALPHA, BRAVO, CHARLIE, DELTA, ECHO = (f"20211040{i:02d}" for i in range(1, 6))
NAMES = {ALPHA: "alpha", BRAVO: "bravo", CHARLIE: "charlie", DELTA: "delta", ECHO: "echo"}
ENTRIES = {CHARLIE: 1, ALPHA: 3, DELTA: 5, BRAVO: 7, ECHO: 9}
DECLARED = {h: n for h, n in ENTRIES.items() if h != ECHO}
AS_OF = at_jst(T, 10)
EVE = at_jst(T - timedelta(days=2), 23, 59, 59)
MID = at_jst(date(2025, 5, 20), 12)
LATE = at_jst(T + timedelta(days=30), 12)
MIHO, RITTO = "0", "1"


def build_main_world() -> World:
    w = World()
    w.race(TARGET, ENTRIES, scratched={ECHO})
    w.race(EMPTY_RACE, {"2019300001": 1, "2019300002": 2})
    d = date
    w.works += [
        Work(ALPHA, d(2025, 2, 26), MIHO, "hill", 12.8, clock="0640"),
        Work(ALPHA, d(2025, 4, 9), MIHO, "wood", 12.1, clock="0715", created=d(2025, 4, 20)),
        Work(ALPHA, d(2025, 5, 14), MIHO, "hill", 12.4, clock="0630"),
        Work(ALPHA, d(2025, 5, 14), MIHO, "wood", 12.0, clock="0810"),
        Work(ALPHA, d(2025, 6, 11), MIHO, "wood", 11.7),
        Work(ALPHA, T, MIHO, "hill", 10.9, clock="0600"),  # race day: never counts
        Work(ALPHA, d(2025, 6, 25), MIHO, "hill", 10.8),  # after the race
        Work(ALPHA, d(2025, 4, 9), MIHO, "wood", 13.5, clock="0715", created=d(2025, 4, 10)),
        Work(CHARLIE, d(2025, 4, 30), RITTO, "wood", 12.9),
        Work(CHARLIE, d(2025, 5, 21), RITTO, "hill", 12.0),
        Work(DELTA, d(2025, 6, 12), RITTO, "hill", 12.3),
        Work(DELTA, d(2025, 6, 14), RITTO, "hill", 12.2, clock="0550"),
        Work(ECHO, d(2025, 6, 11), RITTO, "wood", 11.5),
        # filler horses before T
        Work("2020200000", d(2025, 5, 7), MIHO, "hill", 12.0),
        Work("2020200001", d(2025, 5, 28), MIHO, "hill", 12.2),
        Work(None, d(2025, 4, 23), MIHO, "hill", 12.6),  # blank id
        Work("2020200003", d(2025, 6, 4), MIHO, "hill", 13.0),
        Work("2020200004", d(2024, 12, 18), MIHO, "hill", 12.2),  # the 2024 file
        Work("2020200005", d(2025, 4, 30), MIHO, "wood", 11.9),
        Work("2020200006", d(2025, 5, 28), MIHO, "wood", 12.3),
        Work("0000000000", d(2025, 5, 14), MIHO, "wood", 12.5),  # all-zero id
        Work("2020200008", d(2025, 4, 16), RITTO, "hill", 12.1),
        Work("2020200009", d(2025, 5, 21), RITTO, "hill", 12.5),
        Work("2020200010", d(2025, 6, 4), RITTO, "hill", 12.7),
        # a deletion, written before the version it deletes
        Work("2020200099", d(2025, 5, 7), MIHO, "hill", None, created=d(2025, 5, 20), kubun="0"),
        Work("2020200099", d(2025, 5, 7), MIHO, "hill", 8.0, created=d(2025, 5, 8)),
    ]
    for i, (centre, course) in enumerate(
        [(MIHO, "hill"), (MIHO, "wood"), (RITTO, "hill"), (RITTO, "wood")]
    ):
        w.works.append(Work(f"20203000{i}0", T, centre, course, 9.5))  # same day
        w.works.append(Work(f"20203000{i}1", d(2025, 6, 18), centre, course, 9.0))  # later
    # after T, so they never count: the encodings the build must map
    w.works += [
        Work("2020400001", d(2025, 7, 2), "2", "hill", 12.6),  # another centre code
        Work("2020400002", d(2025, 7, 2), RITTO, "wood", 11.8,
             totals=tuple((n, round(11.8 * n + 1.0 * (n - 1), 1)) for n in range(10, 1, -1))),
        Work("2020400003", d(2025, 7, 3), MIHO, "hill", 12.9,
             totals=((4, None), (3, 39.1), (2, 26.0))),  # no 4-furlong measure
        Work("2020400004", d(2025, 7, 3), MIHO, "wood", 12.0, nines=True),
        Work("2020400005", d(2025, 7, 4), RITTO, "hill", None),  # no final-furlong measure
        Work("2020400006", d(2025, 7, 4), RITTO, "wood", 12.4,
             totals=((6, None), (5, None), (4, None), (3, None), (2, None))),
    ]  # fmt: skip
    return w


def build(**kwargs) -> records.BuildReport:
    return wj.build_workout_table(**kwargs)


@pytest.fixture
def main_world(tmp_path, monkeypatch) -> World:
    data = point_data_dir(tmp_path, monkeypatch)
    w = build_main_world()
    w.write_inputs(data)
    build()
    return w


# ---------------------------------------------------------------------------------------------
# The reference: the contract's rules applied to the fixture's own workouts
# ---------------------------------------------------------------------------------------------


class Expect:
    def __init__(self, world, race_id, *, as_of, hidden=frozenset(), par_min=PAR_MIN):
        self.kept = world.kept()
        self.race_id, self.day = race_id, day_of(race_id)
        self.as_of, self.hidden, self.par_min = as_of, set(hidden), par_min

    def _seen(self, w: Work) -> bool:
        return w.day < self.day and w.day not in self.hidden and pub(w.day) < self.as_of

    def history(self, horse_id: str) -> list[Work]:
        mine = [w for w in self.kept if w.horse_id() == horse_id and self._seen(w)]
        return sorted(mine, key=lambda w: (w.day, w.clock), reverse=True)

    def par(self, centre: str, course: str) -> float | None:
        times = [
            w.ff
            for w in self.kept
            if w.stored()["centre"] == centre
            and w.course == course
            and w.stored()["final_furlong_s"] is not None
            and self._seen(w)
        ]
        return statistics.median(times) if len(times) >= self.par_min else None

    def table(self) -> list[dict]:
        rows = []
        for horse_id, horse_no in sorted(DECLARED.items(), key=lambda kv: kv[1]):
            for w in self.history(horse_id):
                stored = w.stored()
                rows.append(
                    {"race_id": self.race_id, "horse_no": horse_no, "horse_id": horse_id,
                     **{c: stored[c] for c in STORED_DATA}}
                )  # fmt: skip
        return norm(rows)

    def features(self, horse_id: str) -> dict:
        rel = []
        for w in self.history(horse_id):
            s = w.stored()
            par = self.par(s["centre"], s["course"])
            if s["final_furlong_s"] is not None and par is not None:
                rel.append(s["final_furlong_s"] - par)
        recent = rel[:3]  # newest first
        return {
            "ff_best_vs_par_s": min(recent) if recent else None,
            "ff_trend_s": (recent[0] - recent[2]) / 2 if len(recent) == 3 else None,
            "days_since_hard": None,
        }


def norm(rows: list[dict]) -> list[dict]:
    return [{k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()} for r in rows]


def table_rows(frame: pl.DataFrame) -> list[dict]:
    missing = [c for c in TABLE_COLUMNS if c not in frame.columns]
    assert not missing, f"workout_table lacks columns {missing}"
    return norm(frame.select([c for c in TABLE_COLUMNS if c != "published_at"]).to_dicts())


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
# The build: HC and WC records -> processed/jravan/workouts.parquet
# ---------------------------------------------------------------------------------------------

STORED_KEYS = ("horse_id", "workout_date", "workout_time", "centre", "course")


def stored_rows(path: Path) -> list[dict]:
    frame = pl.read_parquet(path)
    missing = [c for c in ("horse_id", *STORED_DATA) if c not in frame.columns]
    assert not missing, f"the stored table lacks columns {missing}"
    rows = norm(frame.select(["horse_id", *STORED_DATA]).to_dicts())
    return sorted(rows, key=lambda r: tuple(str(r[k]) for k in STORED_KEYS))


def test_build_maps_every_kept_record_into_the_workout_columns(main_world):
    path = paths.processed_dir("jravan") / "workouts.parquet"
    assert path.is_file()
    assert path.resolve().is_relative_to(paths.data_dir().resolve())
    assert not path.resolve().is_relative_to(REPO)
    want = sorted(
        norm([w.stored() for w in main_world.kept()]),
        key=lambda r: tuple(str(r[k]) for k in STORED_KEYS),
    )
    assert len(want) == len(main_world.works) - 3  # a correction and a deleted pair drop out
    assert stored_rows(path) == want
    frame = pl.read_parquet(path)
    assert frame.schema["workout_date"] == pl.Date
    assert frame.schema["furlongs"].is_integer()
    for c in ("split_first_s", "split_mid_s", "final_furlong_s"):
        assert frame.schema[c].is_float(), c
    assert frame["effort"].null_count() == frame.height
    assert frame["partner"].null_count() == frame.height


def test_build_report_holds_counts_only(tmp_path, monkeypatch):
    data = point_data_dir(tmp_path, monkeypatch)
    w = build_main_world()
    w.write_inputs(data)
    report = build()
    assert isinstance(report, records.BuildReport)
    assert report.name == "workouts"
    text = report.to_json()
    for work in w.works:
        if work.horse_id():
            assert work.horse_id() not in text


def test_build_gives_the_same_bytes_twice(tmp_path, monkeypatch):
    data = point_data_dir(tmp_path, monkeypatch)
    build_main_world().write_inputs(data)
    first, second = tmp_path / "out1", tmp_path / "out2"
    build(out=first)
    build(out=second)
    a = (first / "workouts.parquet").read_bytes()
    assert a == (second / "workouts.parquet").read_bytes()


def test_build_refuses_an_out_dir_inside_the_repository(tmp_path, monkeypatch):
    data = point_data_dir(tmp_path, monkeypatch)
    build_main_world().write_inputs(data)
    inside = REPO / "jravan-workouts-229-must-not-exist"
    with pytest.raises(paths.DataDirInsideRepo):
        build(out=inside)
    assert not inside.exists()


@pytest.mark.parametrize("bad", ["short HC record", "HC record in the WC file"])
def test_a_corrupt_or_wrong_type_record_raises_and_writes_nothing(tmp_path, monkeypatch, bad):
    data = point_data_dir(tmp_path, monkeypatch)
    build_main_world().write_inputs(data)
    if bad == "short HC record":
        target = paths.raw_dir("jravan").joinpath(*FOLDERS["HC"], "2025.jvd")
        extra, error = wb.jvd_record("HC", record_length("HC") - 3), records.RecordLengthError
    else:
        target = paths.raw_dir("jravan").joinpath(*FOLDERS["WC"], "2025.jvd")
        extra, error = wb.jvd_record("HC", record_length("WC")), records.RecordTypeError
    target.write_bytes(target.read_bytes() + extra)
    with pytest.raises(error):
        build()
    assert not (paths.processed_dir("jravan") / "workouts.parquet").exists()


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


@pytest.mark.parametrize(
    "day",
    [T, T - timedelta(days=1), date(2024, 12, 31), date(2025, 1, 1), date(2016, 2, 29),
     date(2026, 11, 28), date(2009, 7, 4)],
)  # fmt: skip
def test_published_at_is_the_end_of_the_training_day(day):
    p = wj.published_at(day)
    assert p.tzinfo is not None and p.utcoffset() is not None
    assert p == pub(day)
    assert p.astimezone(UTC) == datetime.combine(day, time(15), tzinfo=UTC)
    assert wj.published_at(day + timedelta(days=1)) > p


@pytest.mark.parametrize("as_of", list(AS_OFS.values()), ids=list(AS_OFS))
def test_workout_table_holds_each_runners_published_workouts(main_world, as_of):
    frame = wj.workout_table(TARGET, as_of=as_of)
    assert isinstance(frame, pl.DataFrame)
    assert table_rows(frame) == Expect(main_world, TARGET, as_of=as_of).table()
    zone = frame.schema["published_at"]
    assert isinstance(zone, pl.Datetime) and zone.time_zone == "UTC"
    for row in frame.select("workout_date", "published_at").to_dicts():
        assert row["published_at"] == pub(row["workout_date"])
        assert row["published_at"] < as_of


def test_the_cut_is_live(main_world):
    """Each as_of gives a different table: the cut follows the training date, not a constant
    and not the record's creation date."""
    counts = {
        k: {
            NAMES[h]: n
            for h, n in wj.workout_table(TARGET, as_of=v).group_by("horse_id").len().iter_rows()
        }
        for k, v in (("mid", MID), ("eve", EVE), ("morning", AS_OF))
    }
    assert counts["mid"] == {"alpha": 4, "charlie": 1}
    assert counts["eve"] == {"alpha": 5, "charlie": 2, "delta": 1}
    assert counts["morning"] == {"alpha": 5, "charlie": 2, "delta": 2}


def test_publication_boundary_is_strictly_before(main_world):
    """Delta's workout of T - 1 is not in at exactly its publication instant, and is one
    microsecond later."""
    p = pub(T - timedelta(days=1))
    for as_of, inside in ((p, False), (p + timedelta(microseconds=1), True)):
        frame = wj.workout_table(TARGET, as_of=as_of)
        assert table_rows(frame) == Expect(main_world, TARGET, as_of=as_of).table()
        delta = frame.filter(pl.col("horse_id") == DELTA)["workout_date"].to_list()
        assert (T - timedelta(days=1) in delta) is inside
        got = feature_rows(wj.workout_features(TARGET, as_of=as_of))
        expect = Expect(main_world, TARGET, as_of=as_of)
        for horse_id in DECLARED:
            assert_features(got[horse_id], expect.features(horse_id), (as_of, horse_id))


@pytest.mark.parametrize("as_of", [AS_OF, LATE, at_jst(T + timedelta(days=400), 12)])
def test_race_day_and_later_workouts_never_appear(main_world, as_of):
    """Alpha's race-day and later workouts and every filler's same-day and later ones are in the
    table and, with a late as_of, published: none is a workout of T's race or moves a par."""
    frame = wj.workout_table(TARGET, as_of=as_of)
    assert all(d < T for d in frame["workout_date"].to_list())
    assert table_rows(frame) == Expect(main_world, TARGET, as_of=AS_OF).table()
    got = feature_rows(wj.workout_features(TARGET, as_of=as_of))
    assert got == feature_rows(wj.workout_features(TARGET, as_of=AS_OF))


def test_a_race_without_workouts_has_the_same_columns(main_world):
    frame = wj.workout_table(EMPTY_RACE, as_of=AS_OF)
    assert frame.height == 0
    assert set(TABLE_COLUMNS) <= set(frame.columns)
    assert frame.columns == wj.workout_table(TARGET, as_of=AS_OF).columns


@pytest.mark.parametrize("call", ["workout_table", "workout_features"])
def test_a_naive_as_of_is_refused(main_world, call):
    with pytest.raises(ValueError):
        getattr(wj, call)(TARGET, as_of=datetime(2025, 6, 15, 10, 0))


# ---------------------------------------------------------------------------------------------
# The features: hand-computed values, then the reference over every as_of
# ---------------------------------------------------------------------------------------------

HAND = {  # at AS_OF: pars miho hill 12.4 (7 times, the blank-id one in), miho wood 12.05 (6, the
    # all-zero id in, the corrected 12.1 not 13.5), ritto hill 12.25 (6), ritto wood none (2);
    # alpha's newest three: 06-11 wood, 05-14 08:10 wood, 05-14 06:30 hill
    ALPHA: {"ff_best_vs_par_s": -0.35, "ff_trend_s": -0.175, "days_since_hard": None},
    BRAVO: dict.fromkeys(FEATURES),
    CHARLIE: {"ff_best_vs_par_s": -0.25, "ff_trend_s": None, "days_since_hard": None},
    DELTA: {"ff_best_vs_par_s": -0.05, "ff_trend_s": None, "days_since_hard": None},
}


def test_vocabulary_and_feature_names():
    assert wj.N_RECENT == 3
    assert tuple(wj.FEATURE_COLUMNS) == FEATURES


def test_features_by_hand(main_world):
    expect = Expect(main_world, TARGET, as_of=AS_OF)
    for horse_id in DECLARED:  # the reference agrees with the hand values
        assert_features(expect.features(horse_id), HAND[horse_id], ("reference", horse_id))
    frame = wj.workout_features(TARGET, as_of=AS_OF)
    assert isinstance(frame, pl.DataFrame)
    assert frame["race_id"].to_list() == [TARGET] * len(DECLARED)
    assert dict(zip(frame["horse_id"], frame["horse_no"], strict=True)) == DECLARED
    for c in ("ff_best_vs_par_s", "ff_trend_s"):
        assert frame.schema[c].is_float(), c
    assert frame.schema["days_since_hard"].is_integer()
    assert frame["days_since_hard"].null_count() == frame.height
    got = feature_rows(frame)
    for horse_id in DECLARED:
        assert_features(got[horse_id], HAND[horse_id], NAMES[horse_id])


@pytest.mark.parametrize("as_of", list(AS_OFS.values()), ids=list(AS_OFS))
def test_features_follow_the_as_of_rules(main_world, as_of):
    got = feature_rows(wj.workout_features(TARGET, as_of=as_of))
    expect = Expect(main_world, TARGET, as_of=as_of)
    for horse_id in DECLARED:
        assert_features(got[horse_id], expect.features(horse_id), NAMES[horse_id])


PAR_MIN_HAND = {  # (best, trend); pool sizes at AS_OF: miho hill 7, miho wood 6, ritto hill 6,
    # ritto wood 2
    2: {ALPHA: (-0.35, -0.175), CHARLIE: (-0.25, None), DELTA: (-0.05, None)},
    7: {ALPHA: (0.0, None), CHARLIE: (None, None), DELTA: (None, None)},
    8: {ALPHA: (None, None), CHARLIE: (None, None), DELTA: (None, None)},
}


@pytest.mark.parametrize("par_min", sorted(PAR_MIN_HAND))
def test_a_par_needs_par_min_workouts(main_world, monkeypatch, par_min):
    monkeypatch.setattr(wj, "PAR_MIN_WORKOUTS", par_min)
    got = feature_rows(wj.workout_features(TARGET, as_of=AS_OF))
    expect = Expect(main_world, TARGET, as_of=AS_OF, par_min=par_min)
    for horse_id, (best, trend) in PAR_MIN_HAND[par_min].items():
        want = expect.features(horse_id)
        hand = {**want, "ff_best_vs_par_s": best, "ff_trend_s": trend}
        assert_features(hand, want, ("reference", horse_id))
        assert_features(got[horse_id], want, NAMES[horse_id])


def test_the_features_do_not_depend_on_jrdb(main_world, monkeypatch):
    """With the JRDB tables unreadable and the JRDB flag off, this source's features are whole."""
    for name in ("jrdb_workouts", "jrdb_pre_race"):
        (paths.processed_dir("jra") / f"{name}.parquet").write_bytes(b"not a parquet file")
    monkeypatch.setenv(JRDB_FLAG, "0")
    got = feature_rows(wj.workout_features(TARGET, as_of=AS_OF))
    for horse_id in DECLARED:
        assert_features(got[horse_id], HAND[horse_id], NAMES[horse_id])


# ---------------------------------------------------------------------------------------------
# Holdout: no workout dated on a holdout day is read
# ---------------------------------------------------------------------------------------------


def test_a_holdout_day_is_never_read(main_world, monkeypatch):
    """05-14 marked holdout: alpha's two workouts that day and the all-zero-id pool one vanish.
    By hand: miho hill par 12.4, miho wood par 12.0; alpha's rel -0.3 (06-11), 0.1 (04-09),
    0.4 (02-26)."""
    hidden = date(2025, 5, 14)
    patch_holdout(monkeypatch, [hidden])
    frame = wj.workout_table(TARGET, as_of=AS_OF)
    assert hidden not in frame["workout_date"].to_list()
    assert table_rows(frame) == Expect(main_world, TARGET, as_of=AS_OF, hidden={hidden}).table()
    got = feature_rows(wj.workout_features(TARGET, as_of=AS_OF))
    want = {"ff_best_vs_par_s": -0.3, "ff_trend_s": -0.35, "days_since_hard": None}
    assert_features(got[ALPHA], want, "alpha")
    expect = Expect(main_world, TARGET, as_of=AS_OF, hidden={hidden})
    for horse_id in DECLARED:
        assert_features(got[horse_id], expect.features(horse_id), NAMES[horse_id])


def test_the_races_own_holdout_day_reads_nothing(main_world, monkeypatch):
    patch_holdout(monkeypatch, [T])
    assert wj.workout_table(TARGET, as_of=AS_OF).height == 0
    assert wj.workout_features(TARGET, as_of=AS_OF).height == 0


def test_committed_holdout_days_are_never_read(tmp_path, monkeypatch):
    """Unpatched: a workout dated on a day #71's committed rule marks holdout (2026-11-28 on) is
    not a workout of a later development race."""
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
    dev = date(2026, 11, 25)
    assert not holdout.is_holdout_day("japan", dev)
    data = point_data_dir(tmp_path, monkeypatch)
    golf = "2022100007"
    target = rid(race_day, "TOK", 11)
    w = World()
    w.race(target, {golf: 2, "2022100008": 4})
    w.works += [
        Work(golf, dev, MIHO, "wood", 12.0),
        Work(golf, hold, MIHO, "wood", 11.0),
    ]
    w.write_inputs(data)
    build()
    frame = wj.workout_table(target, as_of=at_jst(race_day, 10))
    assert frame["workout_date"].to_list() == [dev]


# ---------------------------------------------------------------------------------------------
# One real year of the owner's export (local only): vocabulary and plausibility, no figure
# ---------------------------------------------------------------------------------------------


@pytest.mark.local
def test_one_real_year_maps_into_the_vocabulary(tmp_path):
    export, _ = wb.real_export_or_skip()
    years = {
        rt: {p.stem for p in export.joinpath(*FOLDERS[rt]).glob("*.jvd")} for rt in ("HC", "WC")
    }
    both = sorted(y for y in years["HC"] & years["WC"] if y.isdigit() and int(y) < 2026)
    if not both:
        pytest.skip("no year with both HC and WC files before 2026")
    year = both[-1]
    raw = tmp_path / "raw"
    for rt in ("HC", "WC"):
        link = raw.joinpath(*FOLDERS[rt], f"{year}.jvd")
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(export.joinpath(*FOLDERS[rt], f"{year}.jvd"))
    out = tmp_path / "out"
    report = wj.build_workout_table(raw=raw, out=out)
    assert report.name == "workouts"
    frame = pl.read_parquet(out / "workouts.parquet")
    assert frame.height > 0
    centres = set(frame["centre"].drop_nulls().to_list())
    assert centres <= {"miho", "ritto", "other"} and {"miho", "ritto"} <= centres
    assert set(frame["course"].to_list()) == {"hill", "wood"}
    assert {d.year for d in frame["workout_date"].to_list()} == {int(year)}
    ids = frame["horse_id"].drop_nulls()
    assert ids.len() > 0.99 * frame.height
    assert ids.str.contains(r"^[0-9]{10}$").all()
    ff = frame["final_furlong_s"].drop_nulls()
    assert ff.len() > 0 and 10.0 <= ff.median() <= 15.0
    assert ff.is_between(9.0, 20.0).mean() > 0.95
    furlongs = frame.group_by("course").agg(pl.col("furlongs").max()).rows()
    assert dict(furlongs)["hill"] <= 4 and dict(furlongs)["wood"] <= 10
    assert frame["effort"].null_count() == frame["partner"].null_count() == frame.height
