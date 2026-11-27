"""JV-Data RA (race detail) records -> the jravan store's races and meetings (ticket #364), and
the weather per race (ticket #402).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jravan.races`:
- `races_from_records(cols) -> DataFrame`: one `races` row per record of `cols` (as
  `records.decode` returns RA records: JRA venues, one version per race), with the columns of
  `schemas.TABLES['races']`.
- `meetings_from_races(races) -> DataFrame`: one `meetings` row per `meeting_id`, `day_night` and
  `surface_mix` null, `status` `run` when any of its races ran, else `abandoned`.
- `build_races(*, years=None, raw=None, out=None) -> records.BuildReport` (name `races`): reads
  `<raw>/stored/RACE/RA/<YYYY>.jvd` for `years` (default: every year file), the layout from
  `spec.record_spec('RA')`; keeps one version per race (`records.latest_versions` on the layout's
  key items), keeps JRA venues (01-10) and counts the others; writes `<out>/races.parquet` and
  `<out>/meetings.parquet` (defaults `paths.raw_dir('jravan')`, `paths.processed_dir('jravan')`)
  in `TABLES` column order, `tables.ARROW_TYPES` and sort keys, through `records.write_parquet`.
- `horsetrack japan jravan races`: `build_races()` with the defaults, the report written by
  `records.write_build_report` (`<reports dir>/jravan-races.json`).

The weather (#402; orchestrator rulings, reversible):
- The jra store's schema is unchanged: `schemas.TABLES['races']` gets no weather column. The
  weather is a JRA-VAN-only table `race_weather` (like `final_odds` and `pool_votes`, not in
  `schemas.TABLES`), columns in order `race_id`, `weather`, `weather_raw`, all strings (the last
  two nullable): one row per race `races_from_records` writes, same races, same order.
- `weather_from_records(cols) -> DataFrame`: the `race_weather` rows for the RA records
  `races_from_records` takes. `build_races` also writes `<out>/race_weather.parquet`, sorted by
  `race_id`, for the races it writes to `races.parquet`; the same inputs give the same bytes.
- Item 天候コード (the constant `WEATHER`); its values (the constant `WEATHERS`, from the 2011
  code table): `1` FINE, `2` CLOUDY, `3` RAIN, `4` LIGHT RAIN, `5` SNOW, `6` LIGHT SNOW.
  `weather_raw` is the code as printed (trimmed); a blank or all-zero code is unset, both null. A
  code outside the table: `weather` null, `weather_raw` kept, counted in the report's
  `unmapped['weather']` on races that were run only. Whether an unset code on a run race is
  counted is not pinned here.

Each column and the RA item (the `records.decode` key) it comes from, as named in the JV-Data
workbook on the owner's Mac:
- `jurisdiction`: `jra`.
- `race_id`: `keys.race_id(開催年, 開催月日, 競馬場コード, レース番号)`; `meeting_id`:
  `keys.meeting_id(開催年, 開催月日, 競馬場コード)`; `race_date`: 開催年 + 開催月日; `venue`:
  the short code of 競馬場コード (`config/jra.yaml`); `race_no`: レース番号; `source_race_key`:
  `keys.source_race_key(開催年, 競馬場コード, 開催回[第N回], 開催日目[N日目], レース番号)`.
- `season`: from `race_date`, the calendar-year label (`core.store.build._season_of`, start month
  1): 2026-11-22 -> `2026/27`.
- `race_name`: 競走名本題 (`records.text`; blank is null).
- `race_class`: from 競走条件コード最若年条件 (the condition of the youngest age allowed in), then
  グレードコード for a race whose class is OP or unknown, as `parse_results.race_class` does;
  `race_class_raw`: 競走条件コード最若年条件 as printed (trimmed text).
- `distance_m`: 距離. `field_size`: 出走頭数. `post_time_local`: `keys.post_time_local(発走時刻)`.
- `surface`, `course_config`: トラックコード.
- `going`, `going_raw`: 芝馬場状態コード for a turf or jump race, ダート馬場状態コード for a dirt
  race.
- `prize`: the first occurrence of 本賞金 x 100 (yen).
- `split_times_s`: ラップタイム, tenths of a second per lap (`records.scaled(raw, 1)`), blank and
  zero laps dropped, cumulative; empty (or null) for a race without laps.
- `status`: データ区分 `9` is `abandoned`, any other kept record `run`.
- `weight_type`: 重量種別コード. `rail_setting`: コース区分 (a turf race's letter A-D).
- Versions: データ作成年月日 and データ区分 (`0` a deletion), keyed by the layout's key items
  開催年, 開催月日, 競馬場コード, 開催回[第N回], 開催日目[N日目], レース番号.
- `race_index`, `rating_hi`, `rating_lo`, `finish_time_s`, `result_published_utc`,
  `result_unpublished_utc`: null.

Code values pinned (named constants in `races.py`):
- condition: `701` DEBUT, `702` and `703` MAIDEN, `005` C1, `010` C2, `016` C3, `999` OP.
- grade (only for a race whose class is OP): `A` and `F` G1, `B` and `G` G2, `C` and `H` G3, `L` L;
  blank, `D` and `E` keep the class.
- weight type: `1` handicap, `2` allowance, `3` age, `4` set.
- going: `1` GOOD, `2` GOOD TO YIELDING, `3` SOFT, `4` HEAVY.
- track: `10`-`22` turf, `23`-`26` and `29` dirt, `51`-`59` jump; `course_config` is `inner` for
  a code that names only the inner course (`15`, `21`, `25`, `58`), `outer` for one that names only
  the outer (`12`, `16`, `18`, `22`, `26`, `55`, `59`), null otherwise (no course named, or both).
- course letter: `A`-`D` on a turf race; anything else, a blank or a non-turf race is null.

The report: `by_year[<year>]` holds `records` (every record read), `versions_dropped` (records
superseded by a later version of their race), `deletions` (races whose kept version is a deletion),
`races_written`, `non_jra_nar` and `non_jra_overseas` (by `keys.jurisdiction`); `unmapped` is
store column (`race_class`, `weight_type`, `surface`, `going`; `weather` for `race_weather`) ->
code -> count, over the races written only. Counts and codes only.

The RA section built here (`jravan_workbook`) uses those item names with invented positions,
widths, repeats and order, and every value is made up (D21: the workbook and the export are
licensed and stay under `$HORSETRACK_DATA_DIR`). Expected values are written out by hand.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.core.store import schemas, tables
from horsetrack.core.store.asof import load_availability, read_asof
from horsetrack.countries.japan.jravan import keys, records, spec

# isort: split
# (its own block: the sibling helper module, then the module this ticket creates)
import jravan_workbook as wb

from horsetrack.countries.japan.jravan import races

CP = "cp932"
SOURCE = "jravan"
UTC = timezone.utc
US = timedelta(microseconds=1)
NULL_COLUMNS = (
    "race_index", "rating_hi", "rating_lo", "finish_time_s", "result_published_utc",
    "result_unpublished_utc",
)  # fmt: skip

# ------------------------------------------------------------------------------------------------
# A made-up RA layout: the workbook's item names, invented widths, repeats and order

# (item name, width, repeat, key item)
LAYOUT: tuple[tuple[str, int, int, bool], ...] = (
    ("レコード種別ID", 2, 1, False),
    ("データ区分", 1, 1, False),
    ("データ作成年月日", 8, 1, False),
    ("開催年", 4, 1, True),
    ("開催月日", 4, 1, True),
    ("競馬場コード", 2, 1, True),
    ("開催回[第N回]", 2, 1, True),
    ("開催日目[N日目]", 2, 1, True),
    ("レース番号", 2, 1, True),
    ("予備", 3, 1, False),
    ("競走名本題", 24, 1, False),
    ("グレードコード", 2, 1, False),
    ("重量種別コード", 2, 1, False),
    ("競走条件コード3歳条件", 4, 1, False),
    ("競走条件コード4歳条件", 4, 1, False),
    ("競走条件コード最若年条件", 4, 1, False),
    ("距離", 5, 1, False),
    ("トラックコード", 3, 1, False),
    ("コース区分", 3, 1, False),
    ("本賞金", 9, 3, False),
    ("発走時刻", 4, 1, False),
    ("出走頭数", 3, 1, False),
    ("天候コード", 2, 1, False),
    ("芝馬場状態コード", 2, 1, False),
    ("ダート馬場状態コード", 2, 1, False),
    ("ラップタイム", 4, 6, False),
    ("レコード区切", 2, 1, False),
)
ALIASES = {
    "kubun": "データ区分",
    "created": "データ作成年月日",
    "year": "開催年",
    "mmdd": "開催月日",
    "venue": "競馬場コード",
    "kai": "開催回[第N回]",
    "nichi": "開催日目[N日目]",
    "race": "レース番号",
    "name": "競走名本題",
    "grade": "グレードコード",
    "weight": "重量種別コード",
    "cond3": "競走条件コード3歳条件",
    "cond4": "競走条件コード4歳条件",
    "cond": "競走条件コード最若年条件",
    "distance": "距離",
    "track": "トラックコード",
    "course": "コース区分",
    "prize": "本賞金",
    "post": "発走時刻",
    "field": "出走頭数",
    "weather": "天候コード",
    "turf": "芝馬場状態コード",
    "dirt": "ダート馬場状態コード",
    "laps": "ラップタイム",
}
DEFAULTS: dict[str, Any] = {
    "kubun": "7",
    "created": "20250602",
    "year": "2025",
    "mmdd": "0601",
    "venue": "05",
    "kai": "03",
    "nichi": "02",
    "race": "11",
    "name": "架空記念",
    "grade": "",
    "weight": "3",
    "cond": "005",
    "distance": 1600,
    "track": "11",
    "course": "A",
    "prize": (18000, 7200, 4500),
    "post": "1540",
    "field": 16,
    "weather": "1",
    "turf": "1",
    "dirt": "0",
    "laps": (12.3, 11.0, 11.5),
}


def ra_section() -> wb.Section:
    rows: list[wb.Row] = []
    pos = 1
    for no, (name, width, repeat, is_key) in enumerate(LAYOUT, start=1):
        if no == 1:
            rows.append(wb.item(1, name, 1, width, initial="RA", desc=wb.id_desc("RA")))
        else:
            rows.append(
                wb.item(
                    no,
                    name,
                    pos,
                    width,
                    repeat=repeat if repeat > 1 else None,
                    key="○" if is_key else "",
                    desc=f"{name}（架空）",
                )
            )
        pos += width * repeat
    return wb.Section(2, "架空レース詳細", pos - 1, tuple(rows))


def write_ra_workbook(path: Path) -> Path:
    other = wb.simple_section(1, "XQ", "架空その他", [("架空値", 4), ("競走名本題", 6)])
    return wb.write_workbook(path, [other, ra_section()], [])


def _field(value: Any, width: int) -> bytes:
    if value is None:
        out = b" " * width
    elif isinstance(value, bool):
        raise TypeError(value)
    elif isinstance(value, int):
        out = f"{value:0{width}d}".encode("ascii")
    elif isinstance(value, float):  # a lap, in tenths of a second
        out = f"{round(value * 10):0{width}d}".encode("ascii")
    else:
        out = str(value).encode(CP).ljust(width)
    assert len(out) == width, (value, width)
    return out


def ra(**over: Any) -> bytes:
    """One RA record (CRLF included) of the made-up layout: DEFAULTS, then `over` by alias."""
    values = {**DEFAULTS, **over}
    values.setdefault("cond3", values["cond"])
    values.setdefault("cond4", values["cond"])
    by_item = {ALIASES[k]: v for k, v in values.items()}
    out = b""
    for name, width, repeat, _ in LAYOUT:
        if name == "レコード種別ID":
            out += b"RA"
        elif name == "レコード区切":
            out += b"\r\n"
        elif name == "予備":
            out += b"0" * width
        elif repeat > 1:
            given = list(by_item[name])
            assert len(given) <= repeat, (name, given)
            given += [0] * (repeat - len(given))
            out += b"".join(_field(v, width) for v in given)
        else:
            out += _field(by_item[name], width)
    assert len(out) == ra_section().length
    return out


@pytest.fixture(scope="module")
def ra_spec(tmp_path_factory: pytest.TempPathFactory) -> spec.RecordSpec:
    path = write_ra_workbook(tmp_path_factory.mktemp("jravan-spec") / wb.WORKBOOK_NAME)
    return spec.record_spec("RA", path)


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp data dir with the made-up workbook in place."""
    d = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    write_ra_workbook(d / "specs" / "jravan" / wb.WORKBOOK_NAME)
    return d


def write_year(data: Path, year: int, recs: list[bytes]) -> Path:
    return wb.write_jvd(data / "raw" / "jravan" / "stored" / "RACE" / "RA" / f"{year}.jvd", recs)


def mapped(spec_ra: spec.RecordSpec, *recs: bytes) -> pd.DataFrame:
    return races.races_from_records(records.decode(b"".join(recs), spec_ra))


def plain(v: Any) -> Any:
    """A frame value as plain Python: None for any null, dates as `date`, arrays as lists."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [plain(x) for x in v]
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, np.datetime64):
        return pd.Timestamp(v).date()
    if isinstance(v, np.generic):
        return plain(v.item())
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, pd.Timestamp) and v.tzinfo is None and v == v.normalize():
        return v.date()  # a naive midnight: a race date held as a timestamp
    return v


def by_id(frame: pd.DataFrame, column: str = "race_id") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        key = row[column]
        assert key not in out, f"{column} {key} twice"
        out[key] = {k: plain(v) for k, v in row.items()}
    return out


def assert_subset(row: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    got = {k: row.get(k, "<missing>") for k in expected}
    assert got == expected, label


def nonempty(unmapped: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {field: dict(codes) for field, codes in unmapped.items() if codes}


def no_laps(value: Any) -> bool:
    return value is None or len(value) == 0


# ------------------------------------------------------------------------------------------------
# One record, one row


ID_CASES = [  # (record overrides, expected columns)
    (
        dict(year="2026", mmdd="1122", venue="08", kai="05", nichi="07", race="11",
             distance=1600, post="1540", field=18, name="架空杯", prize=(200000, 80000, 50000)),
        dict(race_id="jra:2026-11-22:KYO:11", meeting_id="jra:2026-11-22:KYO",
             race_date=date(2026, 11, 22), venue="KYO", race_no=11,
             source_race_key="202608050711", distance_m=1600, post_time_local="15:40",
             field_size=18, season="2026/27", race_name="架空杯", prize=20_000_000),
    ),
    (
        dict(year="2010", mmdd="0105", venue="06", kai="01", nichi="01", race="01",
             distance=1200, post="0950", field=16, name="", prize=(4800, 1900, 1200)),
        dict(race_id="jra:2010-01-05:NAK:01", meeting_id="jra:2010-01-05:NAK",
             race_date=date(2010, 1, 5), venue="NAK", race_no=1,
             source_race_key="201006010101", distance_m=1200, post_time_local="09:50",
             field_size=16, season="2010/11", race_name=None, prize=480_000),
    ),
    (
        dict(year="2019", mmdd="0728", venue="01", kai="02", nichi="04", race="07",
             distance=2000, post="1305", field=9, name="架空特別", prize=(14500, 5800, 3600)),
        dict(race_id="jra:2019-07-28:SAP:07", meeting_id="jra:2019-07-28:SAP",
             race_date=date(2019, 7, 28), venue="SAP", race_no=7,
             source_race_key="201901020407", distance_m=2000, post_time_local="13:05",
             field_size=9, season="2019/20", race_name="架空特別", prize=1_450_000),
    ),
    (
        dict(year="2024", mmdd="0229", venue="09", kai="01", nichi="06", race="12",
             distance=1400, post="1625", field=14, name="架空Ｓ", prize=(10000, 4000, 2500)),
        dict(race_id="jra:2024-02-29:HSN:12", meeting_id="jra:2024-02-29:HSN",
             race_date=date(2024, 2, 29), venue="HSN", race_no=12,
             source_race_key="202409010612", distance_m=1400, post_time_local="16:25",
             field_size=14, season="2024/25", race_name="架空Ｓ", prize=1_000_000),
    ),
    (
        dict(year="2016", mmdd="0213", venue="10", kai="02", nichi="03", race="03",
             distance=1700, track="24", post="1050", field=11, prize=(5000, 2000, 1300)),
        dict(race_id="jra:2016-02-13:KOK:03", meeting_id="jra:2016-02-13:KOK",
             race_date=date(2016, 2, 13), venue="KOK", race_no=3,
             source_race_key="201610020303", distance_m=1700, post_time_local="10:50",
             field_size=11, season="2016/17", race_name="架空記念", prize=500_000),
    ),
    (
        dict(year="2025", mmdd="1228", venue="06", kai="05", nichi="09", race="10",
             distance=3570, track="52", post="1215", field=12, laps=(),
             prize=(66000, 26000, 17000)),
        dict(race_id="jra:2025-12-28:NAK:10", meeting_id="jra:2025-12-28:NAK",
             race_date=date(2025, 12, 28), venue="NAK", race_no=10,
             source_race_key="202506050910", distance_m=3570, post_time_local="12:15",
             field_size=12, season="2025/26", race_name="架空記念", prize=6_600_000),
    ),
]  # fmt: skip


def test_a_race_record_becomes_one_races_row_with_the_jra_store_id(
    ra_spec: spec.RecordSpec,
) -> None:
    frame = mapped(ra_spec, *(ra(**over) for over, _ in ID_CASES))
    assert len(frame) == len(ID_CASES)
    missing = [c for c in schemas.TABLES["races"].columns if c not in frame.columns]
    assert not missing, missing
    rows = by_id(frame)
    assert set(rows) == {expected["race_id"] for _, expected in ID_CASES}
    for over, expected in ID_CASES:
        row = rows[expected["race_id"]]
        assert_subset(row, {**expected, "jurisdiction": "jra"}, expected["race_id"])
        for column in NULL_COLUMNS:
            assert row[column] is None, (expected["race_id"], column)
        # the id agrees with keys.race_id on the same key text
        assert row["race_id"] == keys.race_id(
            over["year"], over["mmdd"], over["venue"], over["race"]
        )


# ------------------------------------------------------------------------------------------------
# The build


BUILD_2024 = [  # written out of order on purpose
    ra(year="2024", mmdd="1201", venue="06", kai="05", nichi="02", race="02", created="20241203"),
    ra(year="2024", mmdd="1130", venue="08", kai="05", nichi="01", race="11", created="20241202",
       distance=2200, laps=(12.5, 10.9, 11.4, 11.8, 12.0)),
    ra(year="2024", mmdd="1201", venue="06", kai="05", nichi="02", race="01", created="20241203",
       track="24", turf="0", dirt="2", course=""),
]  # fmt: skip
BUILD_2025 = [
    ra(year="2025", mmdd="0601", venue="05", kai="03", nichi="02", race="11"),
    ra(year="2025", mmdd="0105", venue="06", kai="01", nichi="01", race="11", created="20250107"),
    ra(year="2025", mmdd="0601", venue="05", kai="03", nichi="02", race="05", post="1225"),
]
IDS_2024 = ["jra:2024-11-30:KYO:11", "jra:2024-12-01:NAK:01", "jra:2024-12-01:NAK:02"]
IDS_2025 = ["jra:2025-01-05:NAK:11", "jra:2025-06-01:TOK:05", "jra:2025-06-01:TOK:11"]
MEETINGS = [  # in (race_date, venue) order
    ("jra:2024-11-30:KYO", date(2024, 11, 30), "KYO", "2024/25"),
    ("jra:2024-12-01:NAK", date(2024, 12, 1), "NAK", "2024/25"),
    ("jra:2025-01-05:NAK", date(2025, 1, 5), "NAK", "2025/26"),
    ("jra:2025-06-01:TOK", date(2025, 6, 1), "TOK", "2025/26"),
]


def assert_store_types(path: Path, table: str) -> None:
    declared = schemas.TABLES[table].columns
    schema = pq.read_schema(path)
    assert schema.names == list(declared), (table, schema.names)
    for column, kind in declared.items():
        assert schema.field(column).type == tables.ARROW_TYPES[kind], (table, column)


def test_build_writes_races_and_meetings_in_store_types_and_order(data: Path) -> None:
    write_year(data, 2024, BUILD_2024)
    write_year(data, 2025, BUILD_2025)
    report = races.build_races()
    assert isinstance(report, records.BuildReport)
    assert report.name == "races"

    for table in ("races", "meetings"):
        path = tables.table_path(SOURCE, table)
        assert path == data / "processed" / "jravan" / f"{table}.parquet"
        assert_store_types(path, table)

    got = tables.read_table(SOURCE, "races")
    assert list(got.columns) == list(schemas.TABLES["races"].columns)
    assert got["race_id"].tolist() == IDS_2024 + IDS_2025  # sorted by race_id
    rows = by_id(got)
    assert rows["jra:2024-11-30:KYO:11"]["race_date"] == date(2024, 11, 30)
    assert rows["jra:2024-11-30:KYO:11"]["distance_m"] == 2200
    assert rows["jra:2024-11-30:KYO:11"]["prize"] == 1_800_000
    assert rows["jra:2024-11-30:KYO:11"]["split_times_s"] == pytest.approx(
        [12.5, 23.4, 34.8, 46.6, 58.6], abs=1e-6
    )
    assert rows["jra:2024-12-01:NAK:01"]["surface"] == "dirt"
    assert rows["jra:2025-06-01:TOK:05"]["post_time_local"] == "12:25"
    assert {r["jurisdiction"] for r in rows.values()} == {"jra"}

    meetings = tables.read_table(SOURCE, "meetings")
    assert list(meetings.columns) == list(schemas.TABLES["meetings"].columns)
    got_meetings = [
        (r.meeting_id, r.race_date, r.venue, r.season) for r in meetings.itertuples(index=False)
    ]
    assert got_meetings == MEETINGS
    assert set(meetings["jurisdiction"]) == {"jra"}
    assert set(meetings["status"]) == {"run"}
    assert meetings["day_night"].isna().all() and meetings["surface_mix"].isna().all()

    assert report.by_year[2024]["records"] == 3 and report.by_year[2024]["races_written"] == 3
    assert report.by_year[2025]["records"] == 3 and report.by_year[2025]["races_written"] == 3

    # `years` limits what is read, `out` where it goes
    only = data / "elsewhere"
    part = races.build_races(years=[2025], out=only)
    assert set(part.by_year) == {2025}
    assert pq.read_table(only / "races.parquet")["race_id"].to_pylist() == IDS_2025
    assert pq.read_table(only / "meetings.parquet")["meeting_id"].to_pylist() == [
        m[0] for m in MEETINGS[2:]
    ]
    assert_store_types(only / "races.parquet", "races")


def test_the_races_view_derives_post_utc_and_gates_the_going(data: Path) -> None:
    tok11 = "jra:2025-06-01:TOK:11"  # turf, GOOD, 15:40 JST = 06:40 UTC
    hsn01 = "jra:2025-06-01:HSN:01"  # dirt, SOFT, 10:05 JST = 01:05 UTC
    tok12 = "jra:2025-06-01:TOK:12"  # no post time: never shows its going
    write_year(
        data,
        2025,
        [
            ra(race="11", post="1540", turf="1"),
            ra(venue="09", kai="02", nichi="04", race="01", post="1005", track="24",
               turf="0", dirt="3", course=""),
            ra(race="12", post="0000", turf="2"),
        ],
    )  # fmt: skip
    races.build_races()
    rules = load_availability("japan")
    con = tables.connect(SOURCE)
    try:

        def view(t: datetime) -> dict[str, dict[str, Any]]:
            return by_id(read_asof(con, "races", t, rules=rules))

        day_start = datetime(2025, 5, 31, 15, 0, tzinfo=UTC)  # 00:00 JST of the race date
        assert view(day_start - US) == {}
        at = view(day_start)
        assert set(at) == {tok11, hsn01, tok12}
        assert pd.Timestamp(at[tok11]["post_utc"]) == pd.Timestamp("2025-06-01 06:40", tz=UTC)
        assert pd.Timestamp(at[hsn01]["post_utc"]) == pd.Timestamp("2025-06-01 01:05", tz=UTC)
        assert at[tok12]["post_utc"] is None
        assert all(r["going"] is None and r["going_raw"] is None for r in at.values())

        hsn_open = datetime(2025, 6, 1, 0, 5, tzinfo=UTC)  # its post - 60 min
        assert view(hsn_open - US)[hsn01]["going"] is None
        at = view(hsn_open)
        assert (at[hsn01]["going"], at[hsn01]["going_raw"]) == ("SOFT", "3")
        assert at[tok11]["going"] is None

        tok_open = datetime(2025, 6, 1, 5, 40, tzinfo=UTC)
        assert view(tok_open - US)[tok11]["going"] is None
        at = view(tok_open)
        assert (at[tok11]["going"], at[tok11]["going_raw"]) == ("GOOD", "1")
        assert at[tok12]["going"] is None
        assert view(datetime(2025, 6, 3, tzinfo=UTC))[tok12]["going"] is None
    finally:
        con.close()


def test_the_latest_version_of_a_race_is_kept_and_a_deletion_drops_it(data: Path) -> None:
    write_year(
        data,
        2025,
        [
            ra(race="01", created="20250602", kubun="7", distance=1800),  # A: newest first
            ra(race="02", created="20250530", kubun="2", distance=2400),  # B: then deleted
            ra(race="01", created="20250530", kubun="2", distance=1600),
            ra(race="03", created="20250501", kubun="0", distance=1000),  # C: deleted, re-sent
            ra(race="04", created="20250601", kubun="7", distance=1400),  # D: one version
            ra(race="02", created="20250603", kubun="0", distance=2400),
            ra(race="03", created="20250601", kubun="7", distance=2000),
            ra(race="05", created="20250601", kubun="7", distance=1200),  # E: a tie, later wins
            ra(race="05", created="20250601", kubun="7", distance=1150),
        ],
    )
    report = races.build_races()
    got = {rid: row["distance_m"] for rid, row in by_id(tables.read_table(SOURCE, "races")).items()}
    assert got == {
        "jra:2025-06-01:TOK:01": 1800,
        "jra:2025-06-01:TOK:03": 2000,
        "jra:2025-06-01:TOK:04": 1400,
        "jra:2025-06-01:TOK:05": 1150,
    }
    year = report.by_year[2025]
    expected = {"records": 9, "versions_dropped": 4, "deletions": 1, "races_written": 4}
    assert {k: year.get(k) for k in expected} == expected


def test_an_abandoned_race_is_abandoned_and_its_meeting_runs_if_another_race_ran(
    ra_spec: spec.RecordSpec, data: Path
) -> None:
    recs = [
        ra(venue="05", race="01", kubun="9"),  # TOK: one abandoned, one run
        ra(venue="05", race="02", kubun="7"),
        ra(venue="09", kai="02", nichi="04", race="01", kubun="9"),  # HSN: all abandoned
        ra(venue="09", kai="02", nichi="04", race="02", kubun="9"),
        ra(venue="06", kai="04", nichi="01", race="03", kubun="7"),  # NAK: run
    ]
    frame = mapped(ra_spec, *recs)
    status = {rid: row["status"] for rid, row in by_id(frame).items()}
    assert status == {
        "jra:2025-06-01:TOK:01": "abandoned",
        "jra:2025-06-01:TOK:02": "run",
        "jra:2025-06-01:HSN:01": "abandoned",
        "jra:2025-06-01:HSN:02": "abandoned",
        "jra:2025-06-01:NAK:03": "run",
    }
    expected = {
        "jra:2025-06-01:TOK": ("TOK", "run"),
        "jra:2025-06-01:HSN": ("HSN", "abandoned"),
        "jra:2025-06-01:NAK": ("NAK", "run"),
    }
    meetings = by_id(races.meetings_from_races(frame), "meeting_id")
    assert set(meetings) == set(expected)
    for mid, (venue, state) in expected.items():
        assert_subset(
            meetings[mid],
            {"jurisdiction": "jra", "race_date": date(2025, 6, 1), "venue": venue,
             "season": "2025/26", "day_night": None, "surface_mix": None, "status": state},
            mid,
        )  # fmt: skip

    # the build writes the abandoned races (an abandonment is not a deletion)
    write_year(data, 2025, recs)
    races.build_races()
    built = by_id(tables.read_table(SOURCE, "races"))
    assert {rid: row["status"] for rid, row in built.items()} == status
    built_meetings = by_id(tables.read_table(SOURCE, "meetings"), "meeting_id")
    assert {mid: row["status"] for mid, row in built_meetings.items()} == {
        mid: state for mid, (_, state) in expected.items()
    }


# ------------------------------------------------------------------------------------------------
# The vocabulary


GOING_CASES = [  # (track, turf going, dirt going) -> (surface, going, going_raw)
    (("11", "1", "3"), ("turf", "GOOD", "1")),
    (("18", "2", "4"), ("turf", "GOOD TO YIELDING", "2")),
    (("17", "4", "1"), ("turf", "HEAVY", "4")),
    (("24", "1", "3"), ("dirt", "SOFT", "3")),
    (("23", "3", "2"), ("dirt", "GOOD TO YIELDING", "2")),
    (("26", "0", "4"), ("dirt", "HEAVY", "4")),
    (("29", "4", "1"), ("dirt", "GOOD", "1")),
    (("54", "3", "1"), ("jump", "SOFT", "3")),
    (("52", "1", "4"), ("jump", "GOOD", "1")),
]
TRACK_CASES = [  # track -> (surface, course_config)
    ("10", ("turf", None)), ("11", ("turf", None)), ("12", ("turf", "outer")),
    ("13", ("turf", None)), ("15", ("turf", "inner")), ("16", ("turf", "outer")),
    ("17", ("turf", None)), ("18", ("turf", "outer")), ("20", ("turf", None)),
    ("21", ("turf", "inner")), ("22", ("turf", "outer")), ("23", ("dirt", None)),
    ("24", ("dirt", None)), ("25", ("dirt", "inner")), ("26", ("dirt", "outer")),
    ("29", ("dirt", None)), ("51", ("jump", None)), ("52", ("jump", None)),
    ("54", ("jump", None)), ("55", ("jump", "outer")), ("56", ("jump", None)),
    ("58", ("jump", "inner")), ("59", ("jump", "outer")),
]  # fmt: skip


def race_no(i: int) -> str:
    return f"{i + 1:02d}"


def rid(i: int) -> str:
    return f"jra:2025-06-01:TOK:{race_no(i)}"


def test_going_comes_from_the_races_own_surface(ra_spec: spec.RecordSpec) -> None:
    frame = mapped(
        ra_spec,
        *(ra(race=race_no(i), track=t, turf=tg, dirt=dg) for i, ((t, tg, dg), _) in
          enumerate(GOING_CASES)),
    )  # fmt: skip
    rows = by_id(frame)
    for i, (given, (surface, going, raw)) in enumerate(GOING_CASES):
        row = rows[rid(i)]
        assert (row["surface"], row["going"], row["going_raw"]) == (surface, going, raw), given

    frame = mapped(ra_spec, *(ra(race=race_no(i), track=t) for i, (t, _) in enumerate(TRACK_CASES)))
    rows = by_id(frame)
    for i, (track, expected) in enumerate(TRACK_CASES):
        assert (rows[rid(i)]["surface"], rows[rid(i)]["course_config"]) == expected, track


RAIL_CASES = [  # (track, course letter) -> rail_setting
    (("17", "A"), "A"),
    (("18", "B"), "B"),
    (("11", "C"), "C"),
    (("12", "D"), "D"),
    (("10", "B"), "B"),
    (("17", "E"), None),
    (("11", ""), None),
    (("24", "A"), None),
    (("23", "C"), None),
    (("54", "B"), None),
    (("52", "A"), None),
]


def test_rail_setting_is_a_turf_course_letter_and_null_otherwise(
    ra_spec: spec.RecordSpec,
) -> None:
    frame = mapped(
        ra_spec,
        *(ra(race=race_no(i), track=t, course=c) for i, ((t, c), _) in enumerate(RAIL_CASES)),
    )
    rows = by_id(frame)
    for i, (given, expected) in enumerate(RAIL_CASES):
        assert rows[rid(i)]["rail_setting"] == expected, given


SPLIT_CASES = [  # laps -> cumulative leader times
    ((12.3, 11.0, 11.5), [12.3, 23.3, 34.8]),
    ((12.5, 10.9, 11.4, 11.8, 12.0), [12.5, 23.4, 34.8, 46.6, 58.6]),
    ((7.1, 11.2, 12.0, 11.9, 12.4, 12.8), [7.1, 18.3, 30.3, 42.2, 54.6, 67.4]),
    ((12.9, 12.1, None, None), [12.9, 25.0]),  # blank laps dropped
]


def test_split_times_are_cumulative_leader_times(ra_spec: spec.RecordSpec) -> None:
    recs = [ra(race=race_no(i), laps=laps) for i, (laps, _) in enumerate(SPLIT_CASES)]
    n = len(SPLIT_CASES)
    recs.append(ra(race=race_no(n), track="54", laps=()))  # a jump race: every lap zero
    recs.append(ra(race=race_no(n + 1), laps=(None,) * 6))  # every lap blank
    rows = by_id(mapped(ra_spec, *recs))
    for i, (laps, expected) in enumerate(SPLIT_CASES):
        got = rows[rid(i)]["split_times_s"]
        assert got is not None and len(got) == len(expected), laps
        assert got == pytest.approx(expected, abs=1e-6), laps
    assert no_laps(rows[rid(n)]["split_times_s"])
    assert no_laps(rows[rid(n + 1)]["split_times_s"])


CLASS_CASES = [  # (youngest-age condition, grade, track, other ages' condition) -> race_class
    (("701", "", "11", None), "DEBUT"),
    (("702", "", "17", None), "MAIDEN"),
    (("703", "", "24", None), "MAIDEN"),
    (("005", "", "11", "010"), "C1"),  # summer: the older horses' condition is not the class
    (("010", "", "18", None), "C2"),
    (("016", "", "23", None), "C3"),
    (("016", "E", "17", None), "C3"),
    (("999", "", "11", None), "OP"),
    (("999", "E", "24", None), "OP"),
    (("999", "D", "18", None), "OP"),
    (("999", "A", "11", None), "G1"),
    (("999", "B", "17", None), "G2"),
    (("999", "C", "24", None), "G3"),
    (("999", "L", "18", None), "L"),
    (("999", "F", "54", None), "G1"),
    (("999", "G", "52", None), "G2"),
    (("999", "H", "54", None), "G3"),
    (("703", "", "54", None), "MAIDEN"),
]
WEIGHT_CASES = [("1", "handicap"), ("2", "allowance"), ("3", "age"), ("4", "set")]


def test_class_grade_and_weight_type_use_the_jra_store_vocabulary(
    ra_spec: spec.RecordSpec,
) -> None:
    recs = [
        ra(race=race_no(i), cond=c, grade=g, track=t, cond4=other or c,
           weight=WEIGHT_CASES[i % 4][0])
        for i, ((c, g, t, other), _) in enumerate(CLASS_CASES)
    ]  # fmt: skip
    rows = by_id(mapped(ra_spec, *recs))
    for i, ((cond, grade, track, _), expected) in enumerate(CLASS_CASES):
        row = rows[rid(i)]
        assert row["race_class"] == expected, (cond, grade, track)
        assert row["race_class_raw"] == cond, (cond, grade)
        assert row["weight_type"] == WEIGHT_CASES[i % 4][1], WEIGHT_CASES[i % 4]


def test_races_outside_jra_venues_are_counted_not_written(data: Path) -> None:
    other = dict(track="00", cond="000", weight="0", grade="", turf="0", dirt="0", laps=(),
                 course="")  # fmt: skip
    write_year(
        data,
        2025,
        [
            ra(venue="30", kai="01", nichi="01", race="01", kubun="A", **other),
            ra(venue="05", race="01"),
            ra(venue="30", kai="01", nichi="01", race="02", kubun="A", **other),
            ra(venue="44", kai="03", nichi="02", race="11", kubun="A", **other),
            ra(venue="A4", kai="00", nichi="00", race="09", kubun="B", **other),
            ra(venue="05", race="02"),
        ],
    )
    report = races.build_races()
    assert tables.read_table(SOURCE, "races")["race_id"].tolist() == [
        "jra:2025-06-01:TOK:01",
        "jra:2025-06-01:TOK:02",
    ]
    assert tables.read_table(SOURCE, "meetings")["meeting_id"].tolist() == ["jra:2025-06-01:TOK"]
    year = report.by_year[2025]
    expected = {"records": 6, "races_written": 2, "non_jra_nar": 3, "non_jra_overseas": 1}
    assert {k: year.get(k) for k in expected} == expected
    # a race not written is never an unmapped code
    assert nonempty(report.unmapped) == {}


def test_an_unmapped_code_is_null_and_counted(ra_spec: spec.RecordSpec, data: Path) -> None:
    recs = [
        ra(race="01", cond="777"),
        ra(race="02", cond="777", track="24", turf="0", dirt="2", course=""),
        ra(race="03", cond="888"),
        ra(race="04", weight="8"),
        ra(race="05", track="99", turf="1", dirt="1", course=""),
        ra(race="06", track="17", turf="7"),
        ra(race="07"),
        ra(race="08", cond="703", weight="1", track="23", turf="0", dirt="4", course=""),
    ]
    rows = by_id(mapped(ra_spec, *recs))
    tok = "jra:2025-06-01:TOK:"
    assert (rows[tok + "01"]["race_class"], rows[tok + "01"]["race_class_raw"]) == (None, "777")
    assert rows[tok + "03"]["race_class"] is None
    assert rows[tok + "04"]["weight_type"] is None
    assert (rows[tok + "05"]["surface"], rows[tok + "05"]["course_config"]) == (None, None)
    assert rows[tok + "06"]["going"] is None and rows[tok + "06"]["surface"] == "turf"
    assert_subset(
        rows[tok + "07"],
        {"race_class": "C1", "weight_type": "age", "surface": "turf", "going": "GOOD"},
        "mapped",
    )
    assert_subset(
        rows[tok + "08"],
        {"race_class": "MAIDEN", "weight_type": "handicap", "surface": "dirt", "going": "HEAVY"},
        "mapped",
    )

    write_year(data, 2025, recs)
    report = races.build_races()
    assert nonempty(report.unmapped) == {
        "race_class": {"777": 2, "888": 1},
        "weight_type": {"8": 1},
        "surface": {"99": 1},
        "going": {"7": 1},
    }
    assert report.by_year[2025]["races_written"] == 8
    written = by_id(tables.read_table(SOURCE, "races"))
    assert written[tok + "01"]["race_class"] is None and written[tok + "04"]["weight_type"] is None


def test_two_builds_write_identical_bytes(data: Path) -> None:
    write_year(data, 2024, BUILD_2024)
    write_year(data, 2025, BUILD_2025)
    races.build_races()
    first = {t: tables.table_path(SOURCE, t).read_bytes() for t in ("races", "meetings")}
    races.build_races(out=data / "again")
    for table, blob in first.items():
        assert (data / "again" / f"{table}.parquet").read_bytes() == blob, table

    # the command rebuilds the same bytes and writes the report under the data dir
    result = CliRunner().invoke(app, ["japan", "jravan", "races"])
    assert result.exit_code == 0, result.output
    for table, blob in first.items():
        assert tables.table_path(SOURCE, table).read_bytes() == blob, table
    report_path = paths.reports_dir() / "jravan-races.json"
    text = report_path.read_text(encoding="utf-8")
    parsed = json.loads(text)
    assert parsed["name"] == "races"
    assert parsed["by_year"]["2024"]["races_written"] == 3
    assert parsed["by_year"]["2025"]["races_written"] == 3
    assert "架空記念" not in text and "架空記念" not in result.output  # counts only


# ------------------------------------------------------------------------------------------------
# The weather (#402): a jravan-only race_weather table, the jra store's races unchanged


WEATHER_COLUMNS = ["race_id", "weather", "weather_raw"]
# the jra store's races columns before #402, written out: the weather must not join them
RACES_COLUMNS = [
    "jurisdiction", "race_id", "meeting_id", "race_date", "venue", "race_no", "season",
    "race_index", "race_name", "race_class", "rating_hi", "rating_lo", "distance_m", "surface",
    "course_config", "going", "prize", "post_time_local", "field_size", "finish_time_s",
    "split_times_s", "status", "weight_type", "race_class_raw", "going_raw", "source_race_key",
    "rail_setting", "result_published_utc", "result_unpublished_utc",
]  # fmt: skip
WEATHER_CASES = [  # (weather code as printed, track, data kubun) -> (weather, weather_raw)
    (("1", "11", "7"), ("FINE", "1")),
    (("2", "24", "7"), ("CLOUDY", "2")),
    (("3", "54", "7"), ("RAIN", "3")),
    (("4", "17", "7"), ("LIGHT RAIN", "4")),
    (("5", "23", "7"), ("SNOW", "5")),
    (("6", "11", "9"), ("LIGHT SNOW", "6")),  # an abandoned race keeps its weather
    (("3", "18", "7"), ("RAIN", "3")),
    (("1", "52", "9"), ("FINE", "1")),
    ((None, "11", "7"), (None, None)),  # blank: unset
    (("0", "24", "7"), (None, None)),  # all zero: unset
    (("00", "11", "7"), (None, None)),
    (("7", "11", "7"), (None, "7")),  # outside the table: raw kept
    (("9", "24", "7"), (None, "9")),
    (("A", "52", "7"), (None, "A")),
    (("15", "17", "7"), (None, "15")),
]


def test_the_weather_code_is_mapped(ra_spec: spec.RecordSpec) -> None:
    recs = [
        ra(race=race_no(i), weather=w, track=t, kubun=k)
        for i, ((w, t, k), _) in enumerate(WEATHER_CASES)
    ]
    cols = records.decode(b"".join(reversed(recs)), ra_spec)  # written in reverse on purpose
    weather = races.weather_from_records(cols)
    assert list(weather.columns) == WEATHER_COLUMNS
    race_rows = races.races_from_records(cols)
    # one row per race the races mapping writes, same races in the same order
    assert weather["race_id"].tolist() == race_rows["race_id"].tolist()
    assert len(weather) == len(WEATHER_CASES)
    # the jra store's races keep their columns: no weather there
    assert list(race_rows.columns) == RACES_COLUMNS

    rows = by_id(weather)
    for i, (given, expected) in enumerate(WEATHER_CASES):
        row = rows[rid(i)]
        assert (row["weather"], row["weather_raw"]) == expected, given


def weather_rows(path: Path) -> dict[str, tuple[Any, Any]]:
    got = pq.read_table(path).to_pylist()
    return {r["race_id"]: (r["weather"], r["weather_raw"]) for r in got}


def assert_weather_schema(path: Path) -> None:
    schema = pq.read_schema(path)
    assert schema.names == WEATHER_COLUMNS, schema.names
    for column in WEATHER_COLUMNS:
        assert schema.field(column).type == pa.string(), (column, schema.field(column).type)


def test_build_writes_race_weather_for_the_races_written(data: Path) -> None:
    other = dict(track="00", cond="000", weight="0", grade="", turf="0", dirt="0", laps=(),
                 course="")  # fmt: skip
    write_year(
        data,
        2024,
        [
            ra(year="2024", mmdd="1201", venue="06", kai="05", nichi="02", race="01",
               created="20241203", weather="5", track="24", turf="0", dirt="2", course=""),
            ra(year="2024", mmdd="1130", venue="08", kai="05", nichi="01", race="11",
               created="20241202", weather="3"),
        ],
    )  # fmt: skip
    write_year(
        data,
        2025,
        [
            ra(race="01", created="20250530", kubun="2", weather="2"),  # superseded
            ra(race="08", weather="00"),
            ra(race="02", weather="7"),
            ra(race="06", created="20250501", kubun="7", weather="6"),  # then deleted
            ra(race="04", weather="8", kubun="9"),  # abandoned: an unknown code, not counted
            ra(race="03", weather="7", track="24", turf="0", dirt="1", course=""),
            ra(race="01", created="20250602", kubun="7", weather="4"),
            ra(race="05", weather=None),
            ra(venue="30", kai="01", nichi="01", race="01", kubun="A", weather="9", **other),
            ra(race="06", created="20250603", kubun="0", weather="6"),
            ra(race="07", weather="6"),
        ],
    )
    report = races.build_races()

    tok = "jra:2025-06-01:TOK:"
    expected = {
        "jra:2024-11-30:KYO:11": ("RAIN", "3"),
        "jra:2024-12-01:NAK:01": ("SNOW", "5"),
        tok + "01": ("LIGHT RAIN", "4"),
        tok + "02": (None, "7"),
        tok + "03": (None, "7"),
        tok + "04": (None, "8"),
        tok + "05": (None, None),
        tok + "07": ("LIGHT SNOW", "6"),
        tok + "08": (None, None),
    }
    path = data / "processed" / "jravan" / "race_weather.parquet"
    assert path.is_file()
    assert_weather_schema(path)
    ids = pq.read_table(path)["race_id"].to_pylist()
    assert ids == sorted(expected)  # sorted by race_id, one row per race written
    assert weather_rows(path) == expected

    # the races written are the same races, and the jra store's races keep their columns
    assert_store_types(tables.table_path(SOURCE, "races"), "races")
    assert list(schemas.TABLES["races"].columns) == RACES_COLUMNS
    assert tables.read_table(SOURCE, "races")["race_id"].tolist() == ids
    assert report.by_year[2025]["races_written"] == 7

    # unknown codes are counted under `weather`, on run races only; the rest stay mapped
    assert "weather" in report.unmapped
    counted = {code: n for code, n in report.unmapped["weather"].items() if code.strip("0")}
    assert counted == {"7": 2}
    assert nonempty({k: v for k, v in report.unmapped.items() if k != "weather"}) == {}

    # `out` and `years` apply; the same inputs give the same bytes
    again = data / "again"
    races.build_races(out=again)
    assert (again / "race_weather.parquet").read_bytes() == path.read_bytes()
    only = data / "only"
    races.build_races(years=[2024], out=only)
    assert weather_rows(only / "race_weather.parquet") == {
        k: v for k, v in expected.items() if k.startswith("jra:2024")
    }

    # a year whose weather is all unset still writes string columns
    write_year(
        data,
        2023,
        [ra(year="2023", race="01", weather=None), ra(year="2023", race="02", weather="0")],
    )
    unset = data / "unset"
    races.build_races(years=[2023], out=unset)
    assert_weather_schema(unset / "race_weather.parquet")
    assert weather_rows(unset / "race_weather.parquet") == {
        "jra:2023-06-01:TOK:01": (None, None),
        "jra:2023-06-01:TOK:02": (None, None),
    }


# ------------------------------------------------------------------------------------------------
# The real export (local: needs $HORSETRACK_DATA_DIR with the export and the workbook)


@pytest.mark.local
def test_a_real_year_maps_with_unique_race_ids(tmp_path: Path) -> None:
    export, workbook = wb.real_export_or_skip()
    year = 2025  # a whole recent year, before the holdout
    source = export / "stored" / "RACE" / "RA" / f"{year}.jvd"
    if not source.is_file():
        pytest.skip(f"no RA file for {year}")

    report = races.build_races(years=[year], out=tmp_path)
    assert set(report.by_year) == {year}
    for table in ("races", "meetings"):
        assert_store_types(tmp_path / f"{table}.parquet", table)
    got = pq.read_table(tmp_path / "races.parquet").to_pandas()
    ids = got["race_id"].tolist()
    assert len(ids) == len(set(ids)), "duplicate race ids"
    assert ids == sorted(ids)

    # the races expected, derived from the raw records with the existing decoder and keys
    layout = spec.record_spec("RA", workbook)
    cols = records.decode(source.read_bytes(), layout)
    frame = pd.DataFrame({item.name: records.text(cols[item.name]) for item in layout.items
                          if item.is_key})  # fmt: skip
    for name in ("データ作成年月日", "データ区分"):
        frame[name] = records.text(cols[name])
    kept = records.latest_versions(frame, [i.name for i in layout.items if i.is_key])
    kept = kept[[keys.jurisdiction(v) == "jra" for v in kept["競馬場コード"]]]
    expected = {
        keys.race_id(r["開催年"], r["開催月日"], r["競馬場コード"], r["レース番号"])
        for r in kept.to_dict("records")
    }
    assert None not in expected
    assert set(ids) == expected, f"{len(set(ids) ^ expected)} race ids differ"
    assert report.by_year[year]["races_written"] == len(ids)
    assert report.by_year[year]["records"] == len(frame)

    assert set(got["jurisdiction"]) == {"jra"}
    assert got["surface"].notna().all(), f"{int(got['surface'].isna().sum())} races lack a surface"
    assert (got["distance_m"] > 0).all()
    assert set(got["status"]) <= {"run", "abandoned"}
    meetings = pq.read_table(tmp_path / "meetings.parquet").to_pandas()
    assert set(meetings["meeting_id"]) == set(got["meeting_id"])
    assert meetings["meeting_id"].is_unique
