"""JV-Data SE (runner per race) records -> the jravan store's runners, and each race's winner's
time on races (ticket #365).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jravan.runners`:
- `runners_from_records(cols: Mapping[str, numpy.ndarray]) -> pandas.DataFrame`: one `runners`
  row per record of `cols` (as `records.decode` returns SE records: JRA venues, one version per
  runner), with the columns of `schemas.TABLES['runners']`. Beaten lengths are summed per race
  over the records of `cols`.
- `build_runners(*, years=None, raw=None, out=None) -> records.BuildReport` (name `runners`):
  reads `<raw>/stored/RACE/SE/<YYYY>.jvd` for `years` (default: every year file) a year at a
  time, the layout from `spec.record_spec('SE')`; keeps one version per runner
  (`records.latest_versions` on the layout's key items) and JRA venues (01-10), counting the
  others; writes `<out>/runners.parquet` (defaults `paths.raw_dir('jravan')`,
  `paths.processed_dir('jravan')`) in `TABLES` column order, `tables.ARROW_TYPES` and the sort key
  (race_id, horse_no, horse_id), through `records.write_parquet`. Then it sets
  `finish_time_s` in `<out>/races.parquet` (written by `races.build_races`) to the time of each
  race's `finish_pos` 1 runner, the lower horse number on a dead heat, null for a race without
  one, and rewrites that file whole (every other column unchanged, the same bytes for the same
  input).
- `horsetrack japan jravan runners`: `build_runners()` with the defaults, the report written by
  `records.write_build_report` (`<reports dir>/jravan-runners.json`, counts and codes only).

Each column and the SE item (the `records.decode` key) it comes from, as named in the JV-Data
workbook on the owner's Mac:
- `jurisdiction`: `jra`. `race_id`: `keys.race_id(開催年, 開催月日, 競馬場コード, レース番号)`;
  `meeting_id`, `race_date`, `venue`, `race_no` and `season` as on the jravan races (the same
  keys, `core.store.build._season_of` with start month 1).
- `horse_id`: the top-level 血統登録番号, the ten digits as printed (never the one inside
  1着馬(相手馬)情報). `horse_no`: 馬番; `draw`: = `horse_no`. `bracket`: 枠番.
- `horse_name`: 馬名 (`records.text`). `brand_raw` (馬記号コード is not used), `rating` and
  `gear_raw` (ブリンカー使用区分 is not used): null.
- `jockey_code`: 騎手コード as printed; `jockey_name`: 騎手名略称; `jockey_key`:
  `jockey:<code>`. 変更前騎手コード and 変更前騎手名略称 are not used.
- `trainer_code`: 調教師コード; `trainer_name`: 調教師名略称; `trainer_key`: `trainer:<code>`.
- `actual_wt_kg`: 負担重量, tenths of a kg; `actual_wt_lb`: `round(kg * 2.20462)`
  (`parse_results._lb`). 変更前負担重量 is not used.
- `horse_wt_kg`: 馬体重 (999, not weighed, and 000 are null); `declared_horse_wt_lb`:
  `round(horse_wt_kg * 2.20462)`; `horse_wt_change_kg`: 増減差 signed by 増減符号 (`-` negative;
  999 and blank are null, 000 is 0).
- `finish_pos`: 確定着順, null when 00 and for 異常区分コード 1-5 (入線順位 is not used).
- `placing_raw`, `placing_code`, `is_scratched` from 異常区分コード: `0` a plain place
  (`placing_raw` the place as text, `"3"`); `1` `取` WV; `2` and `3` `除` WX; `4` `中` PU;
  `5` `失` DISQ; `7` (demoted) keeps the official place, `placing_raw` `<n>(降)`,
  `placing_code` null. `is_scratched` is true for WV and WX only.
- `dead_heat`: 同着区分 `1`; such a runner's `placing_code` is DH.
- `lbw_raw`: 着差コード as `records.text` gives it (` 12` -> `12`, blank null). `lbw_lengths`: the
  running sum of the margins down the placed order (by finish_pos, then horse number), 0 for the
  winner whatever its code, null for a runner without a place, and null from an unreadable margin
  on (`parse_results._beaten_lengths`). Margin codes (the values `_NAMED_MARGINS` gives the same
  meaning): `H` 0.05, `A` 0.1, `K` 0.25, `D` 0 (dead heat), `T` and `Z` 10, ` 14` 0.25,
  ` 12` 0.5, ` 34` 0.75, `1`-`9` that many lengths, `114` 1.25, `112` 1.5, `134` 1.75,
  `212` 2.5, `214` 2.25, `312` 3.5, `734` 7.75. ＋着差コード and ＋＋着差コード are not used.
- `running_positions`: 1コーナーでの順位, 2コーナーでの順位, 3コーナーでの順位, 4コーナーでの順位
  in that order, the unused corners (00) dropped.
- `finish_time_s`: 走破タイム, minutes then seconds to a tenth (1345 -> 94.5), 0000 null.
- `last_3f_s`: 後3ハロンタイム, tenths (000 null; 後4ハロンタイム is not used).
- `win_odds`: 単勝オッズ, tenths (0 null). `win_fav_rank`: 単勝人気順 (00 null).
- `sex`: 性別コード `1` H, `2` M, `3` G. `age`: 馬齢 as printed (no conversion before 2001).
- Versions: データ作成年月日 and データ区分 (`0` a deletion), keyed by the layout's key items
  開催年, 開催月日, 競馬場コード, 開催回[第N回], 開催日目[N日目], レース番号, 馬番, 血統登録番号.

The report: `by_year[<year>]` holds `records` (every record read), `versions_dropped` (records
superseded by a later version of their runner), `deletions` (runners whose kept version is a
deletion), `runners_written`, `non_jra_nar` and `non_jra_overseas` (by `keys.jurisdiction`);
`unmapped` is field -> code -> count over the runners written (the field names are the coder's;
the tests read the codes and counts). A blank or unset code (a winner's or an unplaced runner's
blank margin) is not unmapped.

The SE and RA sections built here (`jravan_workbook`) use the workbook's item names with invented
widths, repeats and order, and every value is made up (D21: the workbook and the export are
licensed and stay under `$HORSETRACK_DATA_DIR`). Expected values are written out by hand.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.core.store import schemas, tables
from horsetrack.core.store.asof import load_availability, read_asof
from horsetrack.countries.japan.jravan import keys, races, records, spec

# isort: split
# (its own block: the sibling helper module, then the module this ticket creates)
import jravan_workbook as wb

from horsetrack.countries.japan.jravan import runners

CP = "cp932"
SOURCE = "jravan"
UTC = timezone.utc
US = timedelta(microseconds=1)
COLUMNS = list(schemas.TABLES["runners"].columns)
RUNNER_LABELS = {
    "placing_raw", "placing_code", "lbw_raw", "lbw_lengths", "running_positions", "last_3f_s",
    "win_odds", "win_fav_rank",
}  # fmt: skip
WEIGHTS = ("horse_wt_kg", "horse_wt_change_kg", "declared_horse_wt_lb")
FINISH = ("finish_pos", "dead_heat", "finish_time_s")

# ------------------------------------------------------------------------------------------------
# Made-up layouts: the workbook's item names, invented widths, repeats and order


@dataclass(frozen=True)
class Field:
    name: str
    width: int  # one occurrence (a group: the sum of its subs)
    repeat: int = 1
    key: bool = False
    subs: tuple[tuple[str, int], ...] = ()


SE_LAYOUT: tuple[Field, ...] = (
    Field("レコード種別ID", 2),
    Field("データ区分", 1),
    Field("データ作成年月日", 8),
    Field("開催年", 4, key=True),
    Field("開催月日", 4, key=True),
    Field("競馬場コード", 2, key=True),
    Field("開催回[第N回]", 2, key=True),
    Field("開催日目[N日目]", 2, key=True),
    Field("レース番号", 2, key=True),
    Field("馬番", 3, key=True),
    Field("血統登録番号", 10, key=True),
    Field("枠番", 2),
    Field("予備", 4),
    Field("馬名", 24),
    Field("馬記号コード", 2),
    Field("性別コード", 1),
    Field("馬齢", 3),
    Field("騎手コード", 5),
    Field("騎手名略称", 10),
    Field("変更前騎手コード", 5),
    Field("変更前騎手名略称", 10),
    Field("調教師コード", 5),
    Field("調教師名略称", 12),
    Field("負担重量", 4),
    Field("変更前負担重量", 4),
    Field("ブリンカー使用区分", 1),
    Field("馬体重", 3),
    Field("増減符号", 1),
    Field("増減差", 3),
    Field("予備", 2),
    Field("異常区分コード", 1),
    Field("入線順位", 3),
    Field("確定着順", 3),
    Field("同着区分", 1),
    Field("同着頭数", 1),
    Field("走破タイム", 4),
    Field("着差コード", 3),
    Field("＋着差コード", 3),
    Field("1コーナーでの順位", 3),
    Field("2コーナーでの順位", 3),
    Field("3コーナーでの順位", 3),
    Field("4コーナーでの順位", 3),
    Field("単勝オッズ", 5),
    Field("単勝人気順", 3),
    Field("後4ハロンタイム", 3),
    Field("後3ハロンタイム", 3),
    Field("1着馬(相手馬)情報", 34, 2, subs=(("血統登録番号", 10), ("馬名", 24))),
    Field("タイム差", 4),
    Field("レコード区切", 2),
)
SE_ALIASES = {
    "kubun": "データ区分",
    "created": "データ作成年月日",
    "year": "開催年",
    "mmdd": "開催月日",
    "venue": "競馬場コード",
    "kai": "開催回[第N回]",
    "nichi": "開催日目[N日目]",
    "race": "レース番号",
    "horse_no": "馬番",
    "horse_id": "血統登録番号",
    "bracket": "枠番",
    "name": "馬名",
    "brand": "馬記号コード",
    "sex": "性別コード",
    "age": "馬齢",
    "jockey": "騎手コード",
    "jockey_name": "騎手名略称",
    "prev_jockey": "変更前騎手コード",
    "prev_jockey_name": "変更前騎手名略称",
    "trainer": "調教師コード",
    "trainer_name": "調教師名略称",
    "carried": "負担重量",
    "prev_carried": "変更前負担重量",
    "blinkers": "ブリンカー使用区分",
    "body": "馬体重",
    "sign": "増減符号",
    "change": "増減差",
    "abnormal": "異常区分コード",
    "arrived": "入線順位",
    "place": "確定着順",
    "dh": "同着区分",
    "dh_n": "同着頭数",
    "time": "走破タイム",
    "margin": "着差コード",
    "margin2": "＋着差コード",
    "c1": "1コーナーでの順位",
    "c2": "2コーナーでの順位",
    "c3": "3コーナーでの順位",
    "c4": "4コーナーでの順位",
    "odds": "単勝オッズ",
    "fav": "単勝人気順",
    "last4": "後4ハロンタイム",
    "last3": "後3ハロンタイム",
    "rival": "1着馬(相手馬)情報",
    "time_diff": "タイム差",
}
SE_DEFAULTS: dict[str, Any] = {
    "kubun": "7",
    "created": "20250602",
    "year": "2025",
    "mmdd": "0601",
    "venue": "05",
    "kai": "03",
    "nichi": "02",
    "race": "11",
    "horse_no": 1,
    "horse_id": "2021105001",
    "bracket": 1,
    "name": "カクウノウマ",
    "brand": "01",
    "sex": "1",
    "age": 4,
    "jockey": "01126",
    "jockey_name": "架空騎手",
    "prev_jockey": None,
    "prev_jockey_name": None,
    "trainer": "01061",
    "trainer_name": "架空調教師",
    "carried": 57.0,
    "prev_carried": None,
    "blinkers": "1",
    "body": 486,
    "sign": "-",
    "change": 4,
    "abnormal": "0",
    "place": 1,
    "dh": "0",
    "dh_n": "0",
    "time": "1345",
    "margin": "",
    "margin2": "",
    "corners": (0, 0, 3, 2),
    "odds": 12.3,
    "fav": 5,
    "last4": 46.0,
    "last3": 33.8,
    "rival": (("2021109999", "アイテノウマ"),),
    "time_diff": "-002",
}

RA_LAYOUT: tuple[Field, ...] = (
    Field("レコード種別ID", 2),
    Field("データ区分", 1),
    Field("データ作成年月日", 8),
    Field("開催年", 4, key=True),
    Field("開催月日", 4, key=True),
    Field("競馬場コード", 2, key=True),
    Field("開催回[第N回]", 2, key=True),
    Field("開催日目[N日目]", 2, key=True),
    Field("レース番号", 2, key=True),
    Field("競走名本題", 20),
    Field("グレードコード", 1),
    Field("重量種別コード", 1),
    Field("競走条件コード最若年条件", 3),
    Field("距離", 4),
    Field("トラックコード", 2),
    Field("コース区分", 2),
    Field("本賞金", 8, 2),
    Field("発走時刻", 4),
    Field("出走頭数", 2),
    Field("芝馬場状態コード", 1),
    Field("ダート馬場状態コード", 1),
    Field("ラップタイム", 3, 4),
    Field("レコード区切", 2),
)
RA_ALIASES = {
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
    "cond": "競走条件コード最若年条件",
    "distance": "距離",
    "track": "トラックコード",
    "course": "コース区分",
    "prize": "本賞金",
    "post": "発走時刻",
    "field": "出走頭数",
    "turf": "芝馬場状態コード",
    "dirt": "ダート馬場状態コード",
    "laps": "ラップタイム",
}
RA_DEFAULTS: dict[str, Any] = {
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
    "prize": (18000, 7200),
    "post": "1540",
    "field": 16,
    "turf": "1",
    "dirt": "0",
    "laps": (12.3, 11.0, 11.5),
}


def section(number: int, record_type: str, title: str, layout: tuple[Field, ...]) -> wb.Section:
    rows: list[wb.Row] = []
    pos = 1
    for no, f in enumerate(layout, start=1):
        if no == 1:
            rows.append(
                wb.item(1, f.name, 1, f.width, initial=record_type, desc=wb.id_desc(record_type))
            )
        elif f.subs:
            rows.append(wb.group(no, f"<{f.name}>", pos, f.width, f.repeat, desc="（架空）"))
            rel = 1
            for i, (name, length) in enumerate(f.subs):
                rows.append(wb.sub(wb.letter_code(i), name, rel, length, desc="値（架空）"))
                rel += length
        else:
            rows.append(
                wb.item(
                    no,
                    f.name,
                    pos,
                    f.width,
                    repeat=f.repeat if f.repeat > 1 else None,
                    key="○" if f.key else "",
                    desc=f"{f.name}（架空）",
                )
            )
        pos += f.width * f.repeat
    return wb.Section(number, title, pos - 1, tuple(rows))


def write_workbook(path: Path) -> Path:
    other = wb.simple_section(1, "XQ", "架空その他", [("架空値", 4), ("馬名", 6)])
    return wb.write_workbook(
        path,
        [
            other,
            section(2, "RA", "架空レース詳細", RA_LAYOUT),
            section(3, "SE", "架空馬毎レース情報", SE_LAYOUT),
        ],
        [],
    )


def _field(value: Any, width: int) -> bytes:
    if value is None:
        out = b" " * width
    elif isinstance(value, bool):
        raise TypeError(value)
    elif isinstance(value, int):
        out = f"{value:0{width}d}".encode("ascii")
    elif isinstance(value, float):  # tenths
        out = f"{round(value * 10):0{width}d}".encode("ascii")
    else:
        out = str(value).encode(CP).ljust(width)
    assert len(out) == width, (value, width)
    return out


def _record(record_type: str, layout: tuple[Field, ...], by_item: dict[str, Any]) -> bytes:
    out = b""
    for f in layout:
        if f.name == "レコード種別ID":
            out += record_type.encode("ascii")
        elif f.name == "レコード区切":
            out += b"\r\n"
        elif f.name == "予備":
            out += b"0" * (f.width * f.repeat)
        elif f.subs:
            given = list(by_item[f.name])
            assert len(given) <= f.repeat, (f.name, given)
            given += [(None,) * len(f.subs)] * (f.repeat - len(given))
            for occurrence in given:
                out += b"".join(_field(v, w) for v, (_, w) in zip(occurrence, f.subs, strict=True))
        elif f.repeat > 1:
            given = list(by_item[f.name])
            assert len(given) <= f.repeat, (f.name, given)
            given += [0] * (f.repeat - len(given))
            out += b"".join(_field(v, f.width) for v in given)
        else:
            out += _field(by_item[f.name], f.width)
    assert len(out) == sum(f.width * f.repeat for f in layout)
    return out


def se(**over: Any) -> bytes:
    """One SE record (CRLF included) of the made-up layout: SE_DEFAULTS, then `over` by alias.
    `corners` is the four corner positions (0 = unused); `arrived` defaults to `place`."""
    values = {**SE_DEFAULTS, **over}
    corners = values.pop("corners")
    for i, position in enumerate(corners, start=1):
        values[f"c{i}"] = position
    values.setdefault("arrived", values["place"])
    return _record("SE", SE_LAYOUT, {SE_ALIASES[k]: v for k, v in values.items()})


def ra(**over: Any) -> bytes:
    """One RA record of the made-up RA layout: RA_DEFAULTS, then `over` by alias."""
    values = {**RA_DEFAULTS, **over}
    return _record("RA", RA_LAYOUT, {RA_ALIASES[k]: v for k, v in values.items()})


@pytest.fixture(scope="module")
def se_spec(tmp_path_factory: pytest.TempPathFactory) -> spec.RecordSpec:
    path = write_workbook(tmp_path_factory.mktemp("jravan-spec") / wb.WORKBOOK_NAME)
    return spec.record_spec("SE", path)


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp data dir with the made-up workbook in place."""
    d = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    write_workbook(d / "specs" / "jravan" / wb.WORKBOOK_NAME)
    return d


def write_se(data: Path, year: int, recs: list[bytes]) -> Path:
    return wb.write_jvd(data / "raw" / "jravan" / "stored" / "RACE" / "SE" / f"{year}.jvd", recs)


def write_ra(data: Path, year: int, recs: list[bytes]) -> Path:
    return wb.write_jvd(data / "raw" / "jravan" / "stored" / "RACE" / "RA" / f"{year}.jvd", recs)


def mapped(layout: spec.RecordSpec, *recs: bytes) -> pd.DataFrame:
    return runners.runners_from_records(records.decode(b"".join(recs), layout))


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
        return v.date()
    return v


RunnerKey = tuple[str, int]


def by_runner(frame: pd.DataFrame) -> dict[RunnerKey, dict[str, Any]]:
    out: dict[RunnerKey, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        key = (row["race_id"], plain(row["horse_no"]))
        assert key not in out, f"{key} twice"
        out[key] = {k: plain(v) for k, v in row.items()}
    return out


def by_race(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        assert row["race_id"] not in out
        out[row["race_id"]] = {k: plain(v) for k, v in row.items()}
    return out


def assert_subset(row: dict[str, Any], expected: dict[str, Any], label: Any) -> None:
    got = {k: row.get(k, "<missing>") for k in expected}
    assert got == expected, label


def assert_store_types(path: Path, table: str) -> None:
    declared = schemas.TABLES[table].columns
    schema = pq.read_schema(path)
    assert schema.names == list(declared), (table, schema.names)
    for column, kind in declared.items():
        assert schema.field(column).type == tables.ARROW_TYPES[kind], (table, column)


def flat_unmapped(report: records.BuildReport) -> dict[str, int]:
    """Every unmapped code (trimmed) with its count, whatever field the report files it under."""
    out: dict[str, int] = {}
    for codes in report.unmapped.values():
        for code, n in codes.items():
            if n:
                key = str(code).strip()
                out[key] = out.get(key, 0) + n
    return out


def no_positions(value: Any) -> bool:
    return value is None or len(value) == 0


TOK = "jra:2025-06-01:TOK:"


# ------------------------------------------------------------------------------------------------
# One record, one row


ROW_CASES = [  # (record overrides, expected columns)
    (
        dict(year="2026", mmdd="1122", venue="08", kai="05", nichi="07", race="11", horse_no=7,
             horse_id="2022104567", bracket=4, name="カクウノホシ", sex="1", age=4,
             jockey="01126", jockey_name="架空一郎", prev_jockey="05339",
             prev_jockey_name="架空次郎", trainer="01061", trainer_name="架空調教", place=1),
        dict(race_id="jra:2026-11-22:KYO:11", meeting_id="jra:2026-11-22:KYO",
             race_date=date(2026, 11, 22), venue="KYO", race_no=11, season="2026/27",
             horse_id="2022104567", horse_no=7, draw=7, bracket=4, horse_name="カクウノホシ",
             sex="H", age=4, jockey_code="01126", jockey_name="架空一郎",
             jockey_key="jockey:01126", trainer_code="01061", trainer_name="架空調教",
             trainer_key="trainer:01061", finish_pos=1, placing_raw="1"),
    ),
    (
        dict(year="2010", mmdd="0105", venue="06", kai="01", nichi="01", race="01", horse_no=12,
             horse_id="2007100001", bracket=8, name="カクウ　", sex="2", age=3, jockey="00666",
             jockey_name="架空三郎", trainer="00420", trainer_name="架空師", place=1),
        dict(race_id="jra:2010-01-05:NAK:01", meeting_id="jra:2010-01-05:NAK",
             race_date=date(2010, 1, 5), venue="NAK", race_no=1, season="2010/11",
             horse_id="2007100001", horse_no=12, draw=12, bracket=8, horse_name="カクウ",
             sex="M", age=3, jockey_code="00666", jockey_name="架空三郎",
             jockey_key="jockey:00666", trainer_code="00420", trainer_name="架空師",
             trainer_key="trainer:00420", finish_pos=1, placing_raw="1"),
    ),
    (
        dict(year="2010", mmdd="0105", venue="06", kai="01", nichi="01", race="01", horse_no=3,
             horse_id="2007101234", bracket=2, name="ベツノウマ", sex="3", age=5, jockey="01170",
             jockey_name="架空四郎", trainer="01002", trainer_name="架空厩舎", place=10,
             margin="2  "),
        dict(race_id="jra:2010-01-05:NAK:01", meeting_id="jra:2010-01-05:NAK",
             race_date=date(2010, 1, 5), venue="NAK", race_no=1, season="2010/11",
             horse_id="2007101234", horse_no=3, draw=3, bracket=2, horse_name="ベツノウマ",
             sex="G", age=5, jockey_code="01170", jockey_name="架空四郎",
             jockey_key="jockey:01170", trainer_code="01002", trainer_name="架空厩舎",
             trainer_key="trainer:01002", finish_pos=10, placing_raw="10"),
    ),
    (
        dict(year="1998", mmdd="0503", venue="10", kai="01", nichi="04", race="05", horse_no=9,
             horse_id="1992103344", bracket=6, name="ムカシノウマ", sex="3", age=6,
             jockey="00301", jockey_name="昔騎手", trainer="00102", trainer_name="昔調教師",
             place=1),
        dict(race_id="jra:1998-05-03:KOK:05", meeting_id="jra:1998-05-03:KOK",
             race_date=date(1998, 5, 3), venue="KOK", race_no=5, season="1998/99",
             horse_id="1992103344", horse_no=9, draw=9, bracket=6, horse_name="ムカシノウマ",
             sex="G", age=6, jockey_code="00301", jockey_name="昔騎手",
             jockey_key="jockey:00301", trainer_code="00102", trainer_name="昔調教師",
             trainer_key="trainer:00102", finish_pos=1, placing_raw="1"),
    ),
    (
        dict(year="2019", mmdd="0728", venue="01", kai="02", nichi="04", race="07", horse_no=1,
             horse_id="2017102020", bracket=1, name="サッポロノウマ", sex="2", age=2,
             jockey="05212", jockey_name="北騎手", trainer="01155", trainer_name="北調教",
             place=1),
        dict(race_id="jra:2019-07-28:SAP:07", meeting_id="jra:2019-07-28:SAP",
             race_date=date(2019, 7, 28), venue="SAP", race_no=7, season="2019/20",
             horse_id="2017102020", horse_no=1, draw=1, bracket=1, horse_name="サッポロノウマ",
             sex="M", age=2, jockey_code="05212", jockey_name="北騎手",
             jockey_key="jockey:05212", trainer_code="01155", trainer_name="北調教",
             trainer_key="trainer:01155", finish_pos=1, placing_raw="1"),
    ),
]  # fmt: skip


def test_a_runner_record_becomes_a_jra_store_runner_row(se_spec: spec.RecordSpec) -> None:
    frame = mapped(se_spec, *(se(**over) for over, _ in ROW_CASES))
    assert len(frame) == len(ROW_CASES)
    missing = [c for c in COLUMNS if c not in frame.columns]
    assert not missing, missing
    rows = by_runner(frame)
    assert set(rows) == {(e["race_id"], e["horse_no"]) for _, e in ROW_CASES}
    for over, expected in ROW_CASES:
        key = (expected["race_id"], expected["horse_no"])
        row = rows[key]
        assert_subset(
            row,
            {**expected, "jurisdiction": "jra", "brand_raw": None, "rating": None,
             "gear_raw": None, "placing_code": None, "dead_heat": False, "is_scratched": False},
            key,
        )  # fmt: skip
        # the id agrees with keys.race_id on the same key text
        assert row["race_id"] == keys.race_id(
            over["year"], over["mmdd"], over["venue"], over["race"]
        )


# ------------------------------------------------------------------------------------------------
# The build


RA_2024 = [
    ra(year="2024", mmdd="1130", venue="08", kai="05", nichi="01", race="11", created="20241202"),
    ra(year="2024", mmdd="1201", venue="06", kai="05", nichi="02", race="01", created="20241203",
       post="1005"),
    ra(year="2024", mmdd="1201", venue="06", kai="05", nichi="02", race="02", created="20241203",
       post="1035"),
]  # fmt: skip
RA_2025 = [
    ra(race="11", post="1540"),
    ra(race="05", post="1225"),
    ra(year="2025", mmdd="0105", venue="06", kai="01", nichi="01", race="11", created="20250107"),
]
KYO = dict(year="2024", mmdd="1130", venue="08", kai="05", nichi="01", race="11",
           created="20241202")  # fmt: skip
NAK1 = dict(year="2024", mmdd="1201", venue="06", kai="05", nichi="02", race="01",
            created="20241203")  # fmt: skip
NAK11 = dict(year="2025", mmdd="0105", venue="06", kai="01", nichi="01", race="11",
             created="20250107")  # fmt: skip
SE_2024 = [  # written out of order on purpose
    se(**KYO, horse_no=3, horse_id="2021100003", place=2, time="1461", margin="112"),
    se(**NAK1, horse_no=6, horse_id="2022100006", place=1, dh="1", dh_n="1", time="1124",
       margin="D  "),
    se(**KYO, horse_no=1, horse_id="2021100001", place=1, time="1458"),
    se(**NAK1, horse_no=4, horse_id="2022100004", place=3, time="1127", margin=" 34"),
    se(**KYO, horse_no=5, horse_id="2021100005", place=3, time="1462", margin="K  "),
    se(**NAK1, horse_no=2, horse_id="2022100002", place=1, dh="1", dh_n="1", time="1125"),
]  # fmt: skip
SE_2025 = [
    se(race="11", horse_no=9, horse_id="2021100009", place=1, time="2013"),
    se(race="05", horse_no=2, horse_id="2022100102", abnormal="4", place=0, time="0000",
       corners=(0, 0, 4, 0)),
    se(**NAK11, horse_no=11, horse_id="2020100011", abnormal="3", place=0, time="0000",
       corners=(0, 0, 0, 0), odds=0, fav=0, last3=0),
    se(race="11", horse_no=4, horse_id="2021100004", place=2, time="2015", margin="1  "),
    se(race="05", horse_no=1, horse_id="2022100101", abnormal="1", place=0, time="0000",
       corners=(0, 0, 0, 0), odds=0, fav=0, body=0, sign="", change=None, last3=0),
    se(**NAK11, horse_no=8, horse_id="2020100008", place=1, time="0589"),
]  # fmt: skip
IDS_2024 = [
    ("jra:2024-11-30:KYO:11", 1), ("jra:2024-11-30:KYO:11", 3), ("jra:2024-11-30:KYO:11", 5),
    ("jra:2024-12-01:NAK:01", 2), ("jra:2024-12-01:NAK:01", 4), ("jra:2024-12-01:NAK:01", 6),
]  # fmt: skip
IDS_2025 = [
    ("jra:2025-01-05:NAK:11", 8), ("jra:2025-01-05:NAK:11", 11), ("jra:2025-06-01:TOK:05", 1),
    ("jra:2025-06-01:TOK:05", 2), ("jra:2025-06-01:TOK:11", 4), ("jra:2025-06-01:TOK:11", 9),
]  # fmt: skip
WINNERS_TIME = {
    "jra:2024-11-30:KYO:11": 105.8,
    "jra:2024-12-01:NAK:01": 72.5,  # dead heat: horse 2's time, not the lower time (horse 6)
    "jra:2024-12-01:NAK:02": None,  # no runner records
    "jra:2025-01-05:NAK:11": 58.9,
    "jra:2025-06-01:TOK:05": None,  # no placed winner
    "jra:2025-06-01:TOK:11": 121.3,
}


def write_build_inputs(data: Path) -> None:
    write_ra(data, 2024, RA_2024)
    write_ra(data, 2025, RA_2025)
    write_se(data, 2024, SE_2024)
    write_se(data, 2025, SE_2025)


def runner_ids(frame: pd.DataFrame) -> list[RunnerKey]:
    return [(r, int(n)) for r, n in zip(frame["race_id"], frame["horse_no"], strict=True)]


def test_build_writes_runners_and_fills_the_winners_time_on_races(data: Path) -> None:
    write_build_inputs(data)
    races.build_races()
    before = tables.read_table(SOURCE, "races")
    assert before["finish_time_s"].isna().all()

    report = runners.build_runners()
    assert isinstance(report, records.BuildReport)
    assert report.name == "runners"

    path = tables.table_path(SOURCE, "runners")
    assert path == data / "processed" / "jravan" / "runners.parquet"
    assert_store_types(path, "runners")
    got = tables.read_table(SOURCE, "runners")
    assert list(got.columns) == COLUMNS
    assert runner_ids(got) == IDS_2024 + IDS_2025  # sorted by race_id, horse_no
    rows = by_runner(got)
    assert {r["jurisdiction"] for r in rows.values()} == {"jra"}
    assert rows[("jra:2024-11-30:KYO:11", 3)]["finish_time_s"] == pytest.approx(106.1)
    assert rows[("jra:2025-06-01:TOK:11", 9)]["finish_time_s"] == pytest.approx(121.3)
    assert rows[("jra:2025-06-01:TOK:05", 1)]["is_scratched"] is True
    assert rows[("jra:2025-06-01:TOK:05", 2)]["placing_code"] == "PU"
    assert report.by_year[2024]["records"] == 6 and report.by_year[2024]["runners_written"] == 6
    assert report.by_year[2025]["records"] == 6 and report.by_year[2025]["runners_written"] == 6

    # every runner's race is in races
    after = tables.read_table(SOURCE, "races")
    assert set(got["race_id"]) <= set(after["race_id"])

    # races: the winner's time filled in, every other column as races.build_races wrote it
    assert_store_types(tables.table_path(SOURCE, "races"), "races")
    times = {rid: row["finish_time_s"] for rid, row in by_race(after).items()}
    assert set(times) == set(WINNERS_TIME)
    for rid, expected in WINNERS_TIME.items():
        if expected is None:
            assert times[rid] is None, rid
        else:
            assert times[rid] == pytest.approx(expected, abs=1e-6), rid
    pd.testing.assert_frame_equal(
        after.drop(columns="finish_time_s"), before.drop(columns="finish_time_s")
    )

    # `years` limits what is read, `out` where it goes (and which races file is filled)
    only = data / "elsewhere"
    races.build_races(years=[2025], out=only)
    part = runners.build_runners(years=[2025], out=only)
    assert set(part.by_year) == {2025}
    assert_store_types(only / "runners.parquet", "runners")
    assert runner_ids(pq.read_table(only / "runners.parquet").to_pandas()) == IDS_2025
    only_races = pq.read_table(only / "races.parquet").to_pandas()
    only_times = {rid: row["finish_time_s"] for rid, row in by_race(only_races).items()}
    assert only_times["jra:2025-06-01:TOK:11"] == pytest.approx(121.3)
    assert only_times["jra:2025-01-05:NAK:11"] == pytest.approx(58.9)
    assert only_times["jra:2025-06-01:TOK:05"] is None


def test_the_runners_view_gates_weights_and_finish_as_on_the_jra_store(data: Path) -> None:
    tok11 = "jra:2025-06-01:TOK:11"  # 15:40 JST = 06:40 UTC
    hsn01 = "jra:2025-06-01:HSN:01"  # 10:05 JST = 01:05 UTC
    tok12 = "jra:2025-06-01:TOK:12"  # no post time: never shows weights or the finish
    hsn = dict(venue="09", kai="02", nichi="04", race="01")
    write_ra(data, 2025, [ra(race="11", post="1540"), ra(**hsn, post="1005"),
                          ra(race="12", post="0000")])  # fmt: skip
    write_se(
        data,
        2025,
        [
            se(race="11", horse_no=3, horse_id="2021100003", body=486, sign="-", change=4,
               place=1, time="1345"),
            se(race="11", horse_no=7, horse_id="2021100007", abnormal="1", place=0, time="0000",
               body=0, sign="", change=None, odds=0, fav=0, corners=(0, 0, 0, 0), last3=0),
            se(**hsn, horse_no=5, horse_id="2021100005", body=452, sign="+", change=12, place=1,
               dh="1", dh_n="1", time="1112"),
            se(**hsn, horse_no=6, horse_id="2021100006", body=470, sign="", change=0, place=1,
               dh="1", dh_n="1", time="1112", margin="D  "),
            se(race="12", horse_no=1, horse_id="2021100001", body=500, sign="+", change=2,
               place=1, time="1201"),
        ],
    )  # fmt: skip
    races.build_races()
    runners.build_runners()
    weights = {
        (tok11, 3): (486, -4, 1071),
        (tok11, 7): (None, None, None),
        (hsn01, 5): (452, 12, 996),
        (hsn01, 6): (470, 0, 1036),
        (tok12, 1): (500, 2, 1102),  # gated forever: no post time
    }
    finish = {
        (tok11, 3): (1, False, 94.5),
        (tok11, 7): (None, False, None),
        (hsn01, 5): (1, True, 71.2),
        (hsn01, 6): (1, True, 71.2),
        (tok12, 1): (1, False, 72.1),  # gated forever: no post time
    }
    scratched = {(tok11, 3): False, (tok11, 7): True, (hsn01, 5): False, (hsn01, 6): False,
                 (tok12, 1): False}  # fmt: skip
    nothing = (None, None, None)
    rules = load_availability("japan")
    con = tables.connect(SOURCE)
    try:

        def view(t: datetime) -> dict[RunnerKey, dict[str, Any]]:
            frame = read_asof(con, "runners", t, rules=rules)
            assert not RUNNER_LABELS & set(frame.columns), t
            return by_runner(frame)

        def seen(t: datetime, columns: tuple[str, ...]) -> dict[RunnerKey, tuple]:
            return {k: tuple(row[c] for c in columns) for k, row in view(t).items()}

        def expect(table: dict[RunnerKey, tuple], open_for: set[str]) -> dict[RunnerKey, tuple]:
            return {k: v if k[0] in open_for else nothing for k, v in table.items()}

        day_start = datetime(2025, 5, 31, 15, 0, tzinfo=UTC)  # 00:00 JST of the race date
        assert view(day_start - US) == {}
        at = view(day_start)
        assert set(at) == set(weights)
        assert {k: row["is_scratched"] for k, row in at.items()} == scratched  # accepted leak
        assert seen(day_start, WEIGHTS) == expect(weights, set())
        assert seen(day_start, FINISH) == expect(finish, set())

        hsn_weights = datetime(2025, 6, 1, 0, 5, tzinfo=UTC)  # its post - 60 min
        assert seen(hsn_weights - US, WEIGHTS) == expect(weights, set())
        assert seen(hsn_weights, WEIGHTS) == expect(weights, {hsn01})
        tok_weights = datetime(2025, 6, 1, 5, 40, tzinfo=UTC)
        assert seen(tok_weights - US, WEIGHTS) == expect(weights, {hsn01})
        assert seen(tok_weights, WEIGHTS) == expect(weights, {hsn01, tok11})
        assert seen(tok_weights, FINISH) == expect(finish, {hsn01})

        hsn_finish = datetime(2025, 6, 1, 1, 35, tzinfo=UTC)  # its post + 30 min
        assert seen(hsn_finish - US, FINISH) == expect(finish, set())
        assert seen(hsn_finish, FINISH) == expect(finish, {hsn01})
        tok_finish = datetime(2025, 6, 1, 7, 10, tzinfo=UTC)
        assert seen(tok_finish - US, FINISH) == expect(finish, {hsn01})
        assert seen(tok_finish, FINISH) == expect(finish, {hsn01, tok11})
        later = datetime(2025, 6, 5, tzinfo=UTC)
        assert seen(later, WEIGHTS) == expect(weights, {hsn01, tok11})
        assert seen(later, FINISH) == expect(finish, {hsn01, tok11})
    finally:
        con.close()


# ------------------------------------------------------------------------------------------------
# The vocabulary


ABNORMAL_CASES = [  # (abnormal code, 入線, 確定, margin) -> (finish_pos, raw, code, scratched)
    (("0", 1, 1, ""), (1, "1", None, False)),
    (("0", 2, 2, " 12"), (2, "2", None, False)),
    (("7", 3, 5, "K  "), (5, "5(降)", None, False)),  # demoted from 3rd to 5th
    (("0", 4, 3, "1  "), (3, "3", None, False)),
    (("0", 5, 4, "H  "), (4, "4", None, False)),
    (("1", 0, 0, ""), (None, "取", "WV", True)),
    (("2", 0, 0, ""), (None, "除", "WX", True)),
    (("3", 0, 0, ""), (None, "除", "WX", True)),
    (("4", 0, 0, ""), (None, "中", "PU", False)),
    (("5", 0, 0, ""), (None, "失", "DISQ", False)),
    (("0", 6, 6, "2  "), (6, "6", None, False)),
    (("7", 7, 12, "312"), (12, "12(降)", None, False)),  # demoted from 7th to 12th
]


def test_abnormal_codes_map_to_the_jra_store_placing_codes(
    se_spec: spec.RecordSpec, data: Path
) -> None:
    recs = [
        se(race="11", horse_no=i + 1, horse_id=f"20211000{i + 1:02d}", abnormal=code,
           arrived=arrived, place=place, margin=margin,
           time="1345" if place else "0000")
        for i, ((code, arrived, place, margin), _) in enumerate(ABNORMAL_CASES)
    ]  # fmt: skip
    rows = by_runner(mapped(se_spec, *recs))
    for i, (given, (pos, raw, code, scratched)) in enumerate(ABNORMAL_CASES):
        row = rows[(TOK + "11", i + 1)]
        got = (row["finish_pos"], row["placing_raw"], row["placing_code"], row["is_scratched"])
        assert got == (pos, raw, code, scratched), given
        assert row["dead_heat"] is False, given
        if pos is None:
            assert row["lbw_lengths"] is None, given

    # an abnormal code the mapping lacks: no placing code, counted
    write_ra(data, 2025, [ra(race="11"), ra(race="12")])
    write_se(
        data,
        2025,
        [
            *recs,
            se(race="12", horse_no=1, horse_id="2021200001", place=1),
            se(race="12", horse_no=2, horse_id="2021200002", abnormal="8", arrived=0, place=0,
               time="0000"),
        ],
    )  # fmt: skip
    races.build_races()
    report = runners.build_runners()
    built = by_runner(tables.read_table(SOURCE, "runners"))
    odd = built[(TOK + "12", 2)]
    assert odd["placing_code"] is None and odd["is_scratched"] is False
    assert flat_unmapped(report) == {"8": 1}
    assert report.by_year[2025]["runners_written"] == len(recs) + 2


def test_dead_heats_are_flagged(se_spec: spec.RecordSpec) -> None:
    recs = [
        # race 01: two winners
        se(race="01", horse_no=2, horse_id="2021000102", place=1, dh="1", dh_n="1"),
        se(race="01", horse_no=5, horse_id="2021000105", place=1, dh="1", dh_n="1",
           margin="D  "),
        se(race="01", horse_no=1, horse_id="2021000101", place=3, margin=" 12"),
        # race 02: three dead-heated for 2nd, then a plain 5th
        se(race="02", horse_no=1, horse_id="2021000201", place=1),
        se(race="02", horse_no=3, horse_id="2021000203", place=2, dh="1", dh_n="2",
           margin="A  "),
        se(race="02", horse_no=4, horse_id="2021000204", place=2, dh="1", dh_n="2",
           margin="D  "),
        se(race="02", horse_no=8, horse_id="2021000208", place=2, dh="1", dh_n="2",
           margin="D  "),
        se(race="02", horse_no=6, horse_id="2021000206", place=5, margin="1  "),
        # race 03: no dead heat
        se(race="03", horse_no=1, horse_id="2021000301", place=2, margin="K  "),
        se(race="03", horse_no=2, horse_id="2021000302", place=1),
    ]  # fmt: skip
    expected = {  # (race, horse) -> (finish_pos, dead_heat, placing_code, placing_raw, lbw)
        ("01", 2): (1, True, "DH", "1", 0.0),
        ("01", 5): (1, True, "DH", "1", 0.0),
        ("01", 1): (3, False, None, "3", 0.5),
        ("02", 1): (1, False, None, "1", 0.0),
        ("02", 3): (2, True, "DH", "2", 0.1),
        ("02", 4): (2, True, "DH", "2", 0.1),
        ("02", 8): (2, True, "DH", "2", 0.1),
        ("02", 6): (5, False, None, "5", 1.1),
        ("03", 1): (2, False, None, "2", 0.25),
        ("03", 2): (1, False, None, "1", 0.0),
    }
    rows = by_runner(mapped(se_spec, *recs))
    assert len(rows) == len(expected)
    for (race, horse), (pos, dh, code, raw, lbw) in expected.items():
        row = rows[(TOK + race, horse)]
        got = (row["finish_pos"], row["dead_heat"], row["placing_code"], row["placing_raw"])
        assert got == (pos, dh, code, raw), (race, horse)
        assert row["is_scratched"] is False, (race, horse)
        assert row["lbw_lengths"] == pytest.approx(lbw, abs=1e-6), (race, horse)


UNIT_CASES = [  # (record overrides) -> expected columns
    (dict(time="1345", carried=57.0, body=486, sign="-", change=4, odds=12.3, fav=5, last3=33.8),
     dict(finish_time_s=94.5, actual_wt_kg=57.0, actual_wt_lb=126, horse_wt_kg=486,
          horse_wt_change_kg=-4, declared_horse_wt_lb=1071, win_odds=12.3, win_fav_rank=5,
          last_3f_s=33.8)),
    (dict(time="2013", carried=55.0, body=452, sign="+", change=12, odds=1.5, fav=1, last3=35.1),
     dict(finish_time_s=121.3, actual_wt_kg=55.0, actual_wt_lb=121, horse_wt_kg=452,
          horse_wt_change_kg=12, declared_horse_wt_lb=996, win_odds=1.5, win_fav_rank=1,
          last_3f_s=35.1)),
    (dict(time="0589", carried=58.5, body=530, sign="", change=0, odds=123.4, fav=12,
          last3=40.0),
     dict(finish_time_s=58.9, actual_wt_kg=58.5, actual_wt_lb=129, horse_wt_kg=530,
          horse_wt_change_kg=0, declared_horse_wt_lb=1168, win_odds=123.4, win_fav_rank=12,
          last_3f_s=40.0)),
    (dict(time="3256", carried=52.5, body=470, sign="", change=None, odds=45.0, fav=9,
          last3=36.7),  # a first start: no change
     dict(finish_time_s=205.6, actual_wt_kg=52.5, actual_wt_lb=116, horse_wt_kg=470,
          horse_wt_change_kg=None, declared_horse_wt_lb=1036, win_odds=45.0, win_fav_rank=9,
          last_3f_s=36.7)),
    (dict(time="1060", carried=60.0, body=999, sign="", change=999, odds=999.9, fav=18,
          last3=34.0),  # not weighed
     dict(finish_time_s=66.0, actual_wt_kg=60.0, actual_wt_lb=132, horse_wt_kg=None,
          horse_wt_change_kg=None, declared_horse_wt_lb=None, win_odds=999.9, win_fav_rank=18,
          last_3f_s=34.0)),
    (dict(time="0000", carried=54.0, body=0, sign="", change=None, odds=0, fav=0, last3=0),
     dict(finish_time_s=None, actual_wt_kg=54.0, actual_wt_lb=119, horse_wt_kg=None,
          horse_wt_change_kg=None, declared_horse_wt_lb=None, win_odds=None, win_fav_rank=None,
          last_3f_s=None)),
]  # fmt: skip


def test_times_weights_and_odds_are_in_store_units(se_spec: spec.RecordSpec) -> None:
    recs = [
        se(race="11", horse_no=i + 1, horse_id=f"20211000{i + 1:02d}", place=i + 1,
           margin="" if i == 0 else "1  ", prev_carried=50.0, last4=48.2, **over)
        for i, (over, _) in enumerate(UNIT_CASES)
    ]  # fmt: skip
    rows = by_runner(mapped(se_spec, *recs))
    for i, (over, expected) in enumerate(UNIT_CASES):
        row = rows[(TOK + "11", i + 1)]
        for column, want in expected.items():
            got = row[column]
            if want is None or isinstance(want, int):
                assert got == want, (over, column)
            else:
                assert got == pytest.approx(want, abs=1e-6), (over, column)


LBW_CASES = {  # race -> [(horse_no, 異常区分コード, 確定着順, margin, lbw_raw, lbw_lengths)]
    "01": [  # records in horse order; the placed order is 2, 4, 1, 5, 3, 6
        (1, "0", 3, "K  ", "K", 0.75),
        (2, "0", 1, "   ", None, 0.0),
        (3, "0", 5, "112", "112", 2.3),
        (4, "0", 2, " 12", "12", 0.5),
        (5, "0", 4, "H  ", "H", 0.8),
        (6, "0", 6, "T  ", "T", 12.3),
        (7, "4", 0, "   ", None, None),
        (8, "1", 0, "   ", None, None),
    ],
    "02": [  # a promoted winner keeps a margin code and is still 0
        (1, "0", 1, "A  ", "A", 0.0),
        (2, "0", 2, " 34", "34", 0.75),
        (3, "0", 3, "114", "114", 2.0),
        (4, "0", 4, "134", "134", 3.75),
        (5, "0", 5, "2  ", "2", 5.75),
        (6, "0", 6, "212", "212", 8.25),
        (7, "0", 7, "214", "214", 10.5),
        (8, "0", 8, "3  ", "3", 13.5),
        (9, "0", 9, "312", "312", 17.0),
        (10, "0", 10, "4  ", "4", 21.0),
        (11, "0", 11, "5  ", "5", 26.0),
        (12, "0", 12, "6  ", "6", 32.0),
        (13, "0", 13, "7  ", "7", 39.0),
        (14, "0", 14, "734", "734", 46.75),
        (15, "0", 15, "8  ", "8", 54.75),
        (16, "0", 16, "9  ", "9", 63.75),
        (17, "0", 17, "Z  ", "Z", 73.75),
        (18, "0", 18, " 14", "14", 74.0),
    ],
    "03": [  # a dead heat for 2nd, then 1 length and a neck
        (1, "0", 1, "   ", None, 0.0),
        (2, "0", 2, "1  ", "1", 1.0),
        (3, "0", 2, "D  ", "D", 1.0),
        (4, "0", 4, "K  ", "K", 1.25),
        (5, "0", 5, "1  ", "1", 2.25),
    ],
    "04": [  # an unreadable margin: unknown from there on
        (1, "0", 1, "   ", None, 0.0),
        (2, "0", 2, " 12", "12", 0.5),
        (3, "0", 3, "Q  ", "Q", None),
        (4, "0", 4, "1  ", "1", None),
    ],
}


def lbw_records() -> list[bytes]:
    out = []
    for race, field in LBW_CASES.items():
        for horse, code, place, margin, _, _ in field:
            dh = "1" if race == "03" and place == 2 else "0"
            out.append(
                se(race=race, horse_no=horse, horse_id=f"2021{race}{horse:04d}", abnormal=code,
                   place=place, margin=margin, dh=dh, dh_n=dh,
                   time="1345" if place else "0000")
            )  # fmt: skip
    return out


def test_beaten_lengths_sum_the_margins_down_the_placed_order(
    se_spec: spec.RecordSpec, data: Path
) -> None:
    rows = by_runner(mapped(se_spec, *lbw_records()))
    assert len(rows) == sum(len(f) for f in LBW_CASES.values())
    for race, field in LBW_CASES.items():
        for horse, _, _, margin, raw, lbw in field:
            row = rows[(TOK + race, horse)]
            assert row["lbw_raw"] == raw, (race, horse, margin)
            if lbw is None:
                assert row["lbw_lengths"] is None, (race, horse, margin)
            else:
                assert row["lbw_lengths"] == pytest.approx(lbw, abs=1e-6), (race, horse, margin)

    # the unreadable margin is counted; blank margins (winners, unplaced) are not
    write_ra(data, 2025, [ra(race=race) for race in LBW_CASES])
    write_se(data, 2025, lbw_records())
    races.build_races()
    report = runners.build_runners()
    assert flat_unmapped(report) == {"Q": 1}


CORNER_CASES = [
    ((0, 0, 3, 2), [3, 2]),
    ((5, 5, 4, 1), [5, 5, 4, 1]),
    ((0, 12, 10, 8), [12, 10, 8]),
    ((7, 0, 0, 0), [7]),
    ((0, 0, 9, 0), [9]),
    ((3, 4, 0, 0), [3, 4]),
    ((16, 15, 14, 13), [16, 15, 14, 13]),
    ((0, 0, 0, 11), [11]),
    ((2, 0, 6, 0), [2, 6]),
]


def test_unused_corners_are_dropped_from_running_positions(se_spec: spec.RecordSpec) -> None:
    recs = [
        se(race="11", horse_no=i + 1, horse_id=f"20211000{i + 1:02d}", place=i + 1,
           margin="" if i == 0 else " 12", corners=corners)
        for i, (corners, _) in enumerate(CORNER_CASES)
    ]  # fmt: skip
    recs.append(
        se(race="11", horse_no=len(CORNER_CASES) + 1, horse_id="2021109901", abnormal="1",
           place=0, time="0000", corners=(0, 0, 0, 0))
    )  # fmt: skip
    rows = by_runner(mapped(se_spec, *recs))
    for i, (corners, expected) in enumerate(CORNER_CASES):
        assert rows[(TOK + "11", i + 1)]["running_positions"] == expected, corners
    assert no_positions(rows[(TOK + "11", len(CORNER_CASES) + 1)]["running_positions"])


def test_the_latest_version_of_a_runner_is_kept(data: Path) -> None:
    write_ra(data, 2025, [ra(race="11")])
    h = dict(race="11")
    write_se(
        data,
        2025,
        [
            # horse 1: the result (newest) first, then the earlier card version
            se(**h, horse_no=1, horse_id="2021100001", created="20250603", kubun="7", place=2,
               margin="1  ", odds=4.4),
            se(**h, horse_no=1, horse_id="2021100001", created="20250530", kubun="2", place=0,
               time="0000", odds=0),
            # horse 2: sent, then deleted
            se(**h, horse_no=2, horse_id="2021100002", created="20250602", kubun="7", place=3,
               margin="2  "),
            se(**h, horse_no=2, horse_id="2021100002", created="20250604", kubun="0", place=3,
               margin="2  "),
            # horse 3: one version
            se(**h, horse_no=3, horse_id="2021100003", created="20250602", kubun="7", place=1,
               odds=2.1),
            # horse 4: a tie on the creation date, the later record wins
            se(**h, horse_no=4, horse_id="2021100004", created="20250603", kubun="7", place=4,
               margin=" 12", odds=5.0),
            se(**h, horse_no=4, horse_id="2021100004", created="20250603", kubun="7", place=4,
               margin=" 12", odds=6.1),
        ],
    )  # fmt: skip
    races.build_races()
    report = runners.build_runners()
    got = {
        k: (row["finish_pos"], row["win_odds"])
        for k, row in by_runner(tables.read_table(SOURCE, "runners")).items()
    }
    assert got == {
        (TOK + "11", 1): (2, pytest.approx(4.4)),
        (TOK + "11", 3): (1, pytest.approx(2.1)),
        (TOK + "11", 4): (4, pytest.approx(6.1)),
    }
    year = report.by_year[2025]
    expected = {"records": 7, "versions_dropped": 3, "deletions": 1, "runners_written": 3}
    assert {k: year.get(k) for k in expected} == expected


def test_runners_outside_jra_venues_are_counted_not_written(data: Path) -> None:
    other = dict(sex="0", abnormal="0", margin="   ", odds=0, fav=0, place=0, time="0000",
                 trainer=None, trainer_name=None)  # fmt: skip
    write_ra(data, 2025, [ra(race="01"), ra(race="02")])
    write_se(
        data,
        2025,
        [
            se(venue="30", kai="01", nichi="01", race="01", kubun="A", horse_no=1,
               horse_id="2021190001", **other),
            se(race="01", horse_no=1, horse_id="2021100001", place=1),
            se(venue="30", kai="01", nichi="01", race="01", kubun="A", horse_no=2,
               horse_id="2021190002", **other),
            se(venue="44", kai="03", nichi="02", race="11", kubun="A", horse_no=5,
               horse_id="2021190005", **other),
            se(venue="A4", kai="00", nichi="00", race="09", kubun="B", horse_no=3,
               horse_id="2021190003", **other),
            se(race="02", horse_no=4, horse_id="2021100004", place=1),
        ],
    )  # fmt: skip
    races.build_races()
    report = runners.build_runners()
    assert runner_ids(tables.read_table(SOURCE, "runners")) == [(TOK + "01", 1), (TOK + "02", 4)]
    year = report.by_year[2025]
    expected = {"records": 6, "runners_written": 2, "non_jra_nar": 3, "non_jra_overseas": 1}
    assert {k: year.get(k) for k in expected} == expected
    # a runner not written is never an unmapped code
    assert flat_unmapped(report) == {}


def test_two_builds_write_identical_bytes(data: Path) -> None:
    write_build_inputs(data)
    races.build_races()
    runners.build_runners()
    first = {t: tables.table_path(SOURCE, t).read_bytes() for t in ("runners", "races")}

    # into another folder from scratch
    again = data / "again"
    races.build_races(out=again)
    runners.build_runners(out=again)
    for table, blob in first.items():
        assert (again / f"{table}.parquet").read_bytes() == blob, table

    # a second fill over an already-filled races file changes nothing
    runners.build_runners()
    for table, blob in first.items():
        assert tables.table_path(SOURCE, table).read_bytes() == blob, table

    # the command rebuilds the same bytes and writes the report under the data dir
    result = CliRunner().invoke(app, ["japan", "jravan", "runners"])
    assert result.exit_code == 0, result.output
    for table, blob in first.items():
        assert tables.table_path(SOURCE, table).read_bytes() == blob, table
    report_path = paths.reports_dir() / "jravan-runners.json"
    text = report_path.read_text(encoding="utf-8")
    parsed = json.loads(text)
    assert parsed["name"] == "runners"
    assert parsed["by_year"]["2024"]["runners_written"] == 6
    assert parsed["by_year"]["2025"]["runners_written"] == 6
    for secret in ("カクウノウマ", "架空騎手", "架空調教師", "2021100009"):
        assert secret not in text and secret not in result.output  # counts only


# ------------------------------------------------------------------------------------------------
# The real export (local: needs $HORSETRACK_DATA_DIR with the export and the workbook)

REAL_YEAR = 2024  # a whole recent year, before the holdout
PLACING_CODES = {"WV", "WX", "PU", "DISQ", "DH", None}


@pytest.mark.local
def test_a_real_year_maps_with_every_runner_in_a_race(tmp_path: Path) -> None:
    export, workbook = wb.real_export_or_skip()
    source = export / "stored" / "RACE" / "SE" / f"{REAL_YEAR}.jvd"
    race_source = export / "stored" / "RACE" / "RA" / f"{REAL_YEAR}.jvd"
    if not source.is_file() or not race_source.is_file():
        pytest.skip(f"no SE or RA file for {REAL_YEAR}")

    races.build_races(years=[REAL_YEAR], out=tmp_path)
    report = runners.build_runners(years=[REAL_YEAR], out=tmp_path)
    assert set(report.by_year) == {REAL_YEAR}
    for table in ("runners", "races"):
        assert_store_types(tmp_path / f"{table}.parquet", table)
    got = pq.read_table(tmp_path / "runners.parquet").to_pandas()
    race_rows = pq.read_table(tmp_path / "races.parquet").to_pandas()

    # the runners expected, derived from the raw records with the existing decoder and keys
    layout = spec.record_spec("SE", workbook)
    key = [item.name for item in layout.items if item.is_key]
    cols = records.decode(source.read_bytes(), layout)
    frame = pd.DataFrame({name: records.text(cols[name]) for name in key})
    for name in ("データ作成年月日", "データ区分"):
        frame[name] = records.text(cols[name])
    kept = records.latest_versions(frame, key)
    kept = kept[[keys.jurisdiction(v) == "jra" for v in kept["競馬場コード"]]]
    expected = {
        (keys.race_id(r["開催年"], r["開催月日"], r["競馬場コード"], r["レース番号"]),
         int(r["馬番"]), r["血統登録番号"])
        for r in kept.to_dict("records")
    }  # fmt: skip
    ids = list(zip(got["race_id"], got["horse_no"].astype(int), got["horse_id"], strict=True))
    assert len(ids) == len(set(ids)), "duplicate runners"
    assert ids == sorted(ids)
    assert set(ids) == expected, f"{len(set(ids) ^ expected)} runners differ"
    assert report.by_year[REAL_YEAR]["runners_written"] == len(ids)
    assert report.by_year[REAL_YEAR]["records"] == len(frame)

    # every runner is in a race, and every run race's starters are its field
    assert set(got["race_id"]) <= set(race_rows["race_id"])
    assert set(got["jurisdiction"]) == {"jra"}
    assert set(got["placing_code"].where(got["placing_code"].notna(), None)) <= PLACING_CODES
    assert set(got["sex"].dropna()) <= {"H", "M", "G"}
    starters = got[~got["is_scratched"].astype(bool)].groupby("race_id").size()
    run = race_rows[race_rows["status"] == "run"].set_index("race_id")["field_size"]
    off = [rid for rid, size in run.items() if int(starters.get(rid, 0)) != int(size)]
    assert not off, f"{len(off)} run races whose starters are not the field, e.g. {off[:3]}"

    # winners are 0 lengths behind; the races carry the winner's time
    placed = got[got["finish_pos"] == 1].sort_values(["race_id", "horse_no"])
    assert (placed["lbw_lengths"] == 0).all()
    first = placed.drop_duplicates("race_id").set_index("race_id")["finish_time_s"].dropna()
    filled = race_rows.set_index("race_id")["finish_time_s"]
    assert len(first) > 0
    assert set(filled.dropna().index) == set(first.index)
    assert np.allclose(filled[first.index].to_numpy(dtype=float), first.to_numpy(dtype=float))
