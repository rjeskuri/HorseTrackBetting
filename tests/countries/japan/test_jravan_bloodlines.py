"""JV-Data HN (breeding horse), SK (produce) and BT (lineage) records into the jravan tables
`breeding_horses`, `produce` and `lineages` (ticket #371).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jravan.bloodlines`:
- `breeding_horses_from_records(cols)` (HN), `produce_from_records(cols)` (SK) and
  `lineages_from_records(cols)` (BT), each `cols: Mapping[str, numpy.ndarray]` as `records.decode`
  returns it -> `pandas.DataFrame`, one row per record of `cols` (the build has already kept one
  version per key), its columns the names of the table's schema, in order. The colour meaning is
  read with `spec.code_table` from the workbook at `spec.spec_path()` at run time.
- `BREEDING_HORSES_SCHEMA`, `PRODUCE_SCHEMA`, `LINEAGES_SCHEMA: pyarrow.Schema`, exactly the
  columns and types of `BREEDING_TYPES`, `PRODUCE_TYPES` and `LINEAGE_TYPES` below, in that order.
- `build_bloodlines(*, raw=None, out=None) -> records.BuildReport` (name `bloodlines`): `raw` is
  the export root (default `paths.raw_dir('jravan')`), read at `<raw>/stored/BLDN/HN/all.jvd`,
  `.../SK/all.jvd` and `.../BT/all.jvd` with the layouts `spec.record_spec('HN'|'SK'|'BT')`; one
  version per key (`records.latest_versions` on the layout's key item: 繁殖登録番号 for HN and BT,
  血統登録番号 for SK; a key whose latest record has データ区分 `0` is a deletion and has no row).
  Writes, in `out` (default `paths.processed_dir('jravan')`), through `records.write_parquet`:
  `breeding_horses.parquet` sorted by `breeding_id`, `produce.parquet` sorted by `horse_id` and
  `lineages.parquet` sorted by `breeding_id`, with the module's schemas. The same inputs give the
  same bytes.
- The report's `counts` (ints, counts only: no id, name or code-table text) hold at least:
  `breeding_horses`, `produce`, `lineages` (rows written); `sire_ids` (non-null `sire_breeding_id`
  values written) and `sire_ids_resolved` (those equal to a written `breeding_horses.breeding_id`);
  `pedigree_ids` (non-null values in every written `pedigree_breeding_ids` list, each value
  counted, repeats too) and `pedigree_ids_resolved` (those equal to a written breeding id);
  `lineages_resolved` (written `lineages` rows whose `breeding_id` is a written breeding id).
- `horsetrack japan jravan bloodlines`: `build_bloodlines()` with the defaults, the report written
  by `records.write_build_report` (`<reports dir>/jravan-bloodlines.json`).
- None of the three tables is in `schemas.TABLES`, so the store refuses them. Their availability
  rule is `static` (#376, the owner's request of 2026-11-26: pedigree is known at any t): every
  row is in every view at any t, with the same values, `asof_join` matches on its key alone, and
  `read_label` refuses them.

Each column and the workbook item it comes from (the `records.decode` key; names as
`spec.record_spec` returns them from the JV-Data 4.9.0.1 workbook on the owner's Mac). Every text
value is `records.text` (cp932, both ends trimmed of half- and full-width spaces, blank null).
An id (any breeding id, a 血統登録番号, 生産者コード) is the trimmed text as printed, null when
blank or all zeros. A year is an integer, null when blank or all zeros; a date is `YYYYMMDD`,
null when blank or `00000000`.
- breeding_horses (HN): `breeding_id` 繁殖登録番号; `horse_id` 血統登録番号; `name` 馬名; `name_en`
  馬名欧字 (馬名半角ｶﾅ is not used); `birth_year` 生年; `sex` 性別コード `1` H, `2` M, `3` G (code
  table 2202), anything else null; `colour` the meaning of 毛色コード in code table 2203
  (`spec.code_table('2203').values[code]`, its first content column), null for the unset code
  `00`, a blank code and a code the table lacks; `import_type_code` 繁殖馬持込区分 as printed (`0`
  stays `0`); `import_year` 輸入年; `origin` 産地名; `sire_breeding_id` 父馬繁殖登録番号;
  `dam_breeding_id` 母馬繁殖登録番号. 品種コード is not used.
- produce (SK): `horse_id` 血統登録番号; `birth_date` 生年月日; `sex` and `colour` as on HN
  (性別コード, 毛色コード); `import_type_code` 産駒持込区分 as printed; `import_year` 輸入年;
  `breeder_code` 生産者コード; `origin` 産地名; `pedigree_breeding_ids` 3代血統繁殖登録番号, the
  item repeated 14 times without sub-items (`records.decode` shape (n, 14)): a list of the 14
  values in the record's order, each an id as above (null in place, the list always 14 long).
  品種コード is not used.
- lineages (BT): `breeding_id` 繁殖登録番号; `lineage_id` 系統ID; `lineage_name` 系統名;
  `lineage_description` 系統説明 (trimmed at both ends only).

The HN, SK and BT sections built here (`jravan_workbook`) use the workbook's item names with
invented widths, order and companions, so nothing may be sliced by hand; every value and every
code-table meaning is made up (D21: the workbook and the export are licensed and stay under
`$HORSETRACK_DATA_DIR`). Expected values are written out by hand.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.core.store import schemas, tables
from horsetrack.core.store.asof import asof_join, load_availability, read_asof, read_label
from horsetrack.countries.japan.jravan import records, spec

# isort: split
# (its own block: the sibling helper module, then the module this ticket creates)
import jravan_workbook as wb

from horsetrack.countries.japan.jravan import bloodlines

CP = "cp932"
SOURCE = "jravan"
UTC = timezone.utc
TABLE_NAMES = ("breeding_horses", "produce", "lineages")
BREEDING_TYPES = [
    ("breeding_id", pa.string()),
    ("horse_id", pa.string()),
    ("name", pa.string()),
    ("name_en", pa.string()),
    ("birth_year", pa.int64()),
    ("sex", pa.string()),
    ("colour", pa.string()),
    ("import_type_code", pa.string()),
    ("import_year", pa.int64()),
    ("origin", pa.string()),
    ("sire_breeding_id", pa.string()),
    ("dam_breeding_id", pa.string()),
]
PRODUCE_TYPES = [
    ("horse_id", pa.string()),
    ("birth_date", pa.date32()),
    ("sex", pa.string()),
    ("colour", pa.string()),
    ("import_type_code", pa.string()),
    ("import_year", pa.int64()),
    ("breeder_code", pa.string()),
    ("origin", pa.string()),
    ("pedigree_breeding_ids", pa.list_(pa.string())),
]
LINEAGE_TYPES = [
    ("breeding_id", pa.string()),
    ("lineage_id", pa.string()),
    ("lineage_name", pa.string()),
    ("lineage_description", pa.string()),
]
TYPES = {"breeding_horses": BREEDING_TYPES, "produce": PRODUCE_TYPES, "lineages": LINEAGE_TYPES}
SORT_KEY = {"breeding_horses": "breeding_id", "produce": "horse_id", "lineages": "breeding_id"}
ZERO = "0000000000"
COUNT_KEYS = (
    "breeding_horses", "produce", "lineages", "sire_ids", "sire_ids_resolved", "pedigree_ids",
    "pedigree_ids_resolved", "lineages_resolved",
)  # fmt: skip

# ------------------------------------------------------------------------------------------------
# Made-up layouts: the workbook's item names, invented widths, order and companions


@dataclass(frozen=True)
class Field:
    name: str
    width: int
    repeat: int = 1
    key: bool = False


PEDIGREE = "3代血統繁殖登録番号"
HN_LAYOUT: tuple[Field, ...] = (
    Field("レコード種別ID", 2),
    Field("データ区分", 1),
    Field("データ作成年月日", 8),
    Field("繁殖登録番号", 10, key=True),
    Field("血統登録番号", 10),
    Field("予備", 3),
    Field("馬名", 20),
    Field("馬名半角ｶﾅ", 14),
    Field("馬名欧字", 24),
    Field("性別コード", 1),
    Field("生年", 4),
    Field("毛色コード", 2),
    Field("品種コード", 1),
    Field("輸入年", 4),
    Field("繁殖馬持込区分", 1),
    Field("産地名", 12),
    Field("母馬繁殖登録番号", 10),
    Field("父馬繁殖登録番号", 10),
    Field("レコード区切", 2),
)
HN_ALIASES = {
    "kubun": "データ区分", "created": "データ作成年月日", "breeding_id": "繁殖登録番号",
    "horse_id": "血統登録番号", "name": "馬名", "kana": "馬名半角ｶﾅ", "latin": "馬名欧字",
    "sex": "性別コード", "born": "生年", "colour": "毛色コード", "breed": "品種コード",
    "imported": "輸入年", "import_type": "繁殖馬持込区分", "origin": "産地名",
    "dam": "母馬繁殖登録番号", "sire": "父馬繁殖登録番号",
}  # fmt: skip
HN_DEFAULTS: dict[str, Any] = {
    "kubun": "1", "created": "20250602", "breeding_id": "1110000001", "horse_id": "2010100001",
    "name": "カクウシュボバ", "kana": "ｶｸｳ", "latin": "KAKUU SIRE", "sex": "1", "born": "2010",
    "colour": "01", "breed": "1", "imported": "0000", "import_type": "0", "origin": "架空町",
    "dam": "1120000002", "sire": "1110000003",
}  # fmt: skip

SK_LAYOUT: tuple[Field, ...] = (
    Field("レコード種別ID", 2),
    Field("データ区分", 1),
    Field("データ作成年月日", 8),
    Field("血統登録番号", 10, key=True),
    Field("性別コード", 1),
    Field("生年月日", 8),
    Field("毛色コード", 2),
    Field("品種コード", 1),
    Field("予備", 2),
    Field("産駒持込区分", 1),
    Field("輸入年", 4),
    Field("産地名", 12),
    Field("生産者コード", 8),
    Field(PEDIGREE, 10, repeat=14),
    Field("レコード区切", 2),
)
SK_ALIASES = {
    "kubun": "データ区分", "created": "データ作成年月日", "horse_id": "血統登録番号",
    "sex": "性別コード", "born": "生年月日", "colour": "毛色コード", "breed": "品種コード",
    "import_type": "産駒持込区分", "imported": "輸入年", "origin": "産地名",
    "breeder": "生産者コード", "pedigree": PEDIGREE,
}  # fmt: skip


def ancestry(seed: int) -> tuple[str, ...]:
    """14 distinct made-up breeding ids for one foal, each telling its position apart."""
    return tuple(f"1{seed:03d}{pos:02d}{(seed * 7 + pos) % 10000:04d}" for pos in range(1, 15))


SK_DEFAULTS: dict[str, Any] = {
    "kubun": "1", "created": "20250602", "horse_id": "2021105001", "sex": "1",
    "born": "20210315", "colour": "05", "breed": "1", "import_type": "0", "imported": "0000",
    "origin": "架空村", "breeder": "00123456", "pedigree": ancestry(1),
}  # fmt: skip

BT_LAYOUT: tuple[Field, ...] = (
    Field("レコード種別ID", 2),
    Field("データ区分", 1),
    Field("データ作成年月日", 8),
    Field("繁殖登録番号", 10, key=True),
    Field("系統ID", 18),
    Field("予備", 4),
    Field("系統名", 24),
    Field("系統説明", 120),
    Field("レコード区切", 2),
)
BT_ALIASES = {
    "kubun": "データ区分", "created": "データ作成年月日", "breeding_id": "繁殖登録番号",
    "lineage_id": "系統ID", "lineage_name": "系統名", "description": "系統説明",
}  # fmt: skip
BT_DEFAULTS: dict[str, Any] = {
    "kubun": "1", "created": "20250602", "breeding_id": "1110000001", "lineage_id": "0102",
    "lineage_name": "架空系", "description": "架空の系統の説明。",
}  # fmt: skip

# ------------------------------------------------------------------------------------------------
# Made-up code tables: the workbook's table numbers and names, invented meanings. The decoys carry
# the same codes, so a wrong table gives a wrong meaning.

COLOURS = {"01": "架空色ア", "02": "架空色イ", "05": "架空色オ", "09": "架空色ケ"}
OTHER_COLOURS = {"01": "別の色ア色", "02": "別の色イ色", "05": "別の色オ色", "09": "別の色ケ色"}


def code_tables(colours: dict[str, str]) -> list[wb.Table]:
    colour_rows = (("00", "架空初期値", "未設定（架空）"),) + tuple(
        (code, text, f"副{code}") for code, text in colours.items()
    )
    decoy = tuple((code, "誤りの表") for code in ("00", "01", "02", "05", "09"))
    return [
        wb.Table("2201", "架空別表", 2, decoy),
        wb.Table("2202", "性別コード", 1, (("1", "架空牡"), ("2", "架空牝"), ("3", "架空騸"))),
        wb.Table("2203", "毛色コード", 2, colour_rows),
        wb.Table("2204", "馬記号コード", 2, decoy),
    ]


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


def write_workbook(path: Path, colours: dict[str, str] = COLOURS) -> Path:
    other = wb.simple_section(1, "XQ", "架空その他", [("架空値", 4), ("馬名", 6)])
    return wb.write_workbook(
        path,
        [
            other,
            section(2, "HN", "架空繁殖馬マスタ", HN_LAYOUT),
            section(3, "SK", "架空産駒マスタ", SK_LAYOUT),
            section(4, "BT", "架空系統情報", BT_LAYOUT),
        ],
        code_tables(colours),
    )


def _field(value: Any, width: int) -> bytes:
    if value is None:
        out = b" " * width
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
            out += b" " * (f.width * f.repeat)
        elif f.repeat > 1:
            given = list(by_item[f.name])
            assert len(given) == f.repeat, (f.name, given)
            out += b"".join(_field(v, f.width) for v in given)
        else:
            out += _field(by_item[f.name], f.width)
    assert len(out) == sum(f.width * f.repeat for f in layout)
    return out


def hn(**over: Any) -> bytes:
    """One HN record (CRLF included): HN_DEFAULTS, then `over` by alias."""
    values = {**HN_DEFAULTS, **over}
    return _record("HN", HN_LAYOUT, {HN_ALIASES[k]: v for k, v in values.items()})


def sk(**over: Any) -> bytes:
    values = {**SK_DEFAULTS, **over}
    return _record("SK", SK_LAYOUT, {SK_ALIASES[k]: v for k, v in values.items()})


def bt(**over: Any) -> bytes:
    values = {**BT_DEFAULTS, **over}
    return _record("BT", BT_LAYOUT, {BT_ALIASES[k]: v for k, v in values.items()})


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp data dir with the made-up workbook in place."""
    d = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    write_workbook(d / "specs" / "jravan" / wb.WORKBOOK_NAME)
    return d


def write_bldn(root: Path, record_type: str, recs: Sequence[bytes]) -> Path:
    """`recs` as `<root>/stored/BLDN/<type>/all.jvd` (`root` the export root)."""
    return wb.write_jvd(root / "stored" / "BLDN" / record_type / "all.jvd", list(recs))


def write_inputs(root: Path, hns: Sequence[bytes], sks: Sequence[bytes], bts: Sequence[bytes]):
    write_bldn(root, "HN", hns)
    write_bldn(root, "SK", sks)
    write_bldn(root, "BT", bts)


def decoded(record_type: str, *recs: bytes) -> dict[str, np.ndarray]:
    return records.decode(b"".join(recs), spec.record_spec(record_type))


def plain(v: Any) -> Any:
    """A frame value as plain Python: None for any null, dates as `date`, lists as lists."""
    if isinstance(v, (list, tuple, np.ndarray, pa.Array)):
        return [plain(x) for x in list(v)]
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, np.datetime64):
        return pd.Timestamp(v).date()
    if isinstance(v, np.generic):
        return plain(v.item())
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, pd.Timestamp) and v.tzinfo is None and v == v.normalize():
        return v.date()
    if isinstance(v, datetime) and v.tzinfo is None and v.time() == datetime.min.time():
        return v.date()
    return v


def by_key(frame: pd.DataFrame, key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        k = row[key]
        assert k not in out, f"{k} twice"
        out[k] = {c: plain(v) for c, v in row.items()}
    return out


def check_frame(frame: pd.DataFrame, table: str, n: int) -> None:
    assert isinstance(frame, pd.DataFrame), table
    assert list(frame.columns) == [name for name, _ in TYPES[table]], table
    assert len(frame) == n, table


def table_path(out: Path, table: str) -> Path:
    return out / f"{table}.parquet"


def read_rows(out: Path, table: str) -> list[dict[str, Any]]:
    return pq.read_table(table_path(out, table)).to_pylist()


def keyed(rows: list[dict[str, Any]], table: str) -> dict[str, dict[str, Any]]:
    return {r[SORT_KEY[table]]: r for r in rows}


# ------------------------------------------------------------------------------------------------
# breeding_horses (HN)


HN_CASES = [  # (HN overrides, expected breeding_horses row)
    (
        dict(breeding_id="1110000001", horse_id="2010100001", name="カクウシュボバ",
             latin="KAKUU SIRE", sex="1", born="2010", colour="01", import_type="0",
             imported="0000", origin="架空町", sire="1110000003", dam="1120000002"),
        dict(breeding_id="1110000001", horse_id="2010100001", name="カクウシュボバ",
             name_en="KAKUU SIRE", birth_year=2010, sex="H", colour="架空色ア",
             import_type_code="0", import_year=None, origin="架空町",
             sire_breeding_id="1110000003", dam_breeding_id="1120000002"),
    ),
    (
        dict(breeding_id="1120000002", horse_id="2004104321", name="ベツノハハ　",
             latin="Betsu no Haha", sex="2", born="2004", colour="09", import_type="3",
             imported="2007", origin="架空国", sire="1110000777", dam="1120000888"),
        dict(breeding_id="1120000002", horse_id="2004104321", name="ベツノハハ",
             name_en="Betsu no Haha", birth_year=2004, sex="M", colour="架空色ケ",
             import_type_code="3", import_year=2007, origin="架空国",
             sire_breeding_id="1110000777", dam_breeding_id="1120000888"),
    ),
    (
        dict(breeding_id="1110000003", horse_id="1998100042", name="センノチチ",
             latin="  SEN NO CHICHI", sex="3", born="1998", colour="05", import_type="1",
             imported="2001", origin="ｶｸｳ牧場", sire="1110000005", dam="1120000006"),
        dict(breeding_id="1110000003", horse_id="1998100042", name="センノチチ",
             name_en="SEN NO CHICHI", birth_year=1998, sex="G", colour="架空色オ",
             import_type_code="1", import_year=2001, origin="ｶｸｳ牧場",
             sire_breeding_id="1110000005", dam_breeding_id="1120000006"),
    ),
    (
        dict(breeding_id="1110000005", horse_id="1985100777", name="フルイソセン",
             latin="Furui Sosen", sex="1", born="1985", colour="00", import_type="9",
             imported="1990", origin="", sire="1110000999", dam="1120000998"),
        dict(breeding_id="1110000005", horse_id="1985100777", name="フルイソセン",
             name_en="Furui Sosen", birth_year=1985, sex="H", colour=None,
             import_type_code="9", import_year=1990, origin=None,
             sire_breeding_id="1110000999", dam_breeding_id="1120000998"),
    ),
    (
        dict(breeding_id="1120000006", horse_id="1990100500", name="ナゾノハハ",
             latin="Nazo", sex="0", born="1990", colour="77", import_type="2",
             imported="0000", origin="架空郡", sire="1110000005", dam="1120000501"),
        dict(breeding_id="1120000006", horse_id="1990100500", name="ナゾノハハ",
             name_en="Nazo", birth_year=1990, sex=None, colour=None,
             import_type_code="2", import_year=None, origin="架空郡",
             sire_breeding_id="1110000005", dam_breeding_id="1120000501"),
    ),
]  # fmt: skip


def test_a_breeding_horse_record_maps_with_its_parents(data: Path) -> None:
    check_schema = [(f.name, f.type) for f in bloodlines.BREEDING_HORSES_SCHEMA]
    assert check_schema == BREEDING_TYPES
    recs = [hn(**over) for over, _ in HN_CASES]
    frame = bloodlines.breeding_horses_from_records(decoded("HN", *recs))
    check_frame(frame, "breeding_horses", len(HN_CASES))
    rows = by_key(frame, "breeding_id")
    assert rows == {e["breeding_id"]: e for _, e in HN_CASES}

    # The parents are breeding ids: follow the sire line through the rows of the same records
    # (1110000001 -> its sire 1110000003 -> its sire 1110000005, whose sire is not among them).
    line, at = [], "1110000001"
    while at in rows:
        line.append(rows[at]["name"])
        at = rows[at]["sire_breeding_id"]
    assert line == ["カクウシュボバ", "センノチチ", "フルイソセン"]
    assert at == "1110000999"
    # ... and a dam is a breeding id too: 1110000003's dam has its own row and sire.
    dam = rows[rows["1110000003"]["dam_breeding_id"]]
    assert (dam["name"], dam["sire_breeding_id"]) == ("ナゾノハハ", "1110000005")

    # The colour meaning comes from the workbook: the same records under another workbook.
    write_workbook(data / "specs" / "jravan" / wb.WORKBOOK_NAME, OTHER_COLOURS)
    frame = bloodlines.breeding_horses_from_records(decoded("HN", *recs))
    got = {k: r["colour"] for k, r in by_key(frame, "breeding_id").items()}
    codes = {over["breeding_id"]: over["colour"] for over, _ in HN_CASES}
    assert got == {k: OTHER_COLOURS.get(code) for k, code in codes.items()}


# ------------------------------------------------------------------------------------------------
# produce (SK)


def shuffled(seed: int) -> tuple[str, ...]:
    """14 distinct ids in an order that is neither sorted nor reversed."""
    ids = ancestry(seed)
    order = [5, 0, 13, 2, 9, 1, 12, 7, 3, 11, 4, 8, 10, 6]
    return tuple(ids[i] for i in order)


SK_CASES = [  # (SK overrides, expected produce row)
    (
        dict(horse_id="2021105001", sex="1", born="20210315", colour="05", import_type="0",
             imported="0000", breeder="00123456", origin="架空村", pedigree=ancestry(1)),
        dict(horse_id="2021105001", birth_date=date(2021, 3, 15), sex="H", colour="架空色オ",
             import_type_code="0", import_year=None, breeder_code="00123456", origin="架空村",
             pedigree_breeding_ids=list(ancestry(1))),
    ),
    (
        dict(horse_id="2019104444", sex="2", born="20190228",
             colour="02", import_type="3", imported="2020", breeder="70000001",
             origin="架空国　", pedigree=tuple(reversed(ancestry(2)))),
        dict(horse_id="2019104444", birth_date=date(2019, 2, 28), sex="M", colour="架空色イ",
             import_type_code="3", import_year=2020, breeder_code="70000001", origin="架空国",
             pedigree_breeding_ids=list(reversed(ancestry(2)))),
    ),
    (
        dict(horse_id="2016100808", sex="3", born="20160501", colour="09", import_type="1",
             imported="2017", breeder="00000777", origin="ｶｸｳ牧場", pedigree=shuffled(3)),
        dict(horse_id="2016100808", birth_date=date(2016, 5, 1), sex="G", colour="架空色ケ",
             import_type_code="1", import_year=2017, breeder_code="00000777", origin="ｶｸｳ牧場",
             pedigree_breeding_ids=list(shuffled(3))),
    ),
    (
        dict(horse_id="2024100001", sex="4", born="20240229", colour="00", import_type="2",
             imported="0000", breeder="12345678", origin="架空郡", pedigree=ancestry(4)),
        dict(horse_id="2024100001", birth_date=date(2024, 2, 29), sex=None, colour=None,
             import_type_code="2", import_year=None, breeder_code="12345678", origin="架空郡",
             pedigree_breeding_ids=list(ancestry(4))),
    ),
]  # fmt: skip


def test_produce_keeps_the_fourteen_pedigree_ids_in_order(data: Path) -> None:
    assert [(f.name, f.type) for f in bloodlines.PRODUCE_SCHEMA] == PRODUCE_TYPES
    recs = [sk(**over) for over, _ in SK_CASES]
    frame = bloodlines.produce_from_records(decoded("SK", *recs))
    check_frame(frame, "produce", len(SK_CASES))
    rows = by_key(frame, "horse_id")
    assert rows == {e["horse_id"]: e for _, e in SK_CASES}
    for horse_id, row in rows.items():
        ids = row["pedigree_breeding_ids"]
        assert len(ids) == 14 and len(set(ids)) == 14, horse_id
    # position k of every list is the k-th printed id (the ids encode their position)
    first = rows["2021105001"]["pedigree_breeding_ids"]
    assert [int(i[4:6]) for i in first] == list(range(1, 15))

    # The same foal printed with its ancestors rotated: the list follows the record.
    rotated = ancestry(1)[3:] + ancestry(1)[:3]
    frame = bloodlines.produce_from_records(decoded("SK", sk(pedigree=rotated)))
    assert plain(frame["pedigree_breeding_ids"].iloc[0]) == list(rotated)

    # Colour, as on HN, from the workbook at run time.
    write_workbook(data / "specs" / "jravan" / wb.WORKBOOK_NAME, OTHER_COLOURS)
    frame = bloodlines.produce_from_records(decoded("SK", *recs))
    got = {k: r["colour"] for k, r in by_key(frame, "horse_id").items()}
    assert got == {over["horse_id"]: OTHER_COLOURS.get(over["colour"]) for over, _ in SK_CASES}


# ------------------------------------------------------------------------------------------------
# lineages (BT)


BT_CASES = [  # (BT overrides, expected lineages row)
    (
        dict(breeding_id="1110000001", lineage_id="01", lineage_name="架空系",
             description="架空の大系統。"),
        dict(breeding_id="1110000001", lineage_id="01", lineage_name="架空系",
             lineage_description="架空の大系統。"),
    ),
    (
        dict(breeding_id="1110000003", lineage_id="0102", lineage_name="架空分岐系　",
             description="　前に空白　中にも　空白。　"),
        dict(breeding_id="1110000003", lineage_id="0102", lineage_name="架空分岐系",
             lineage_description="前に空白　中にも　空白。"),
    ),
    (
        dict(breeding_id="1110000777", lineage_id="010203040506", lineage_name="Kakuu Line",
             description="Line text, two sentences. The second one."),
        dict(breeding_id="1110000777", lineage_id="010203040506", lineage_name="Kakuu Line",
             lineage_description="Line text, two sentences. The second one."),
    ),
    (
        dict(breeding_id="1110000005", lineage_id="0399", lineage_name="ベツ系",
             description=""),
        dict(breeding_id="1110000005", lineage_id="0399", lineage_name="ベツ系",
             lineage_description=None),
    ),
]  # fmt: skip


def test_lineages_map_breeding_ids_to_lineage_ids(data: Path) -> None:
    assert [(f.name, f.type) for f in bloodlines.LINEAGES_SCHEMA] == LINEAGE_TYPES
    recs = [bt(**over) for over, _ in BT_CASES]
    frame = bloodlines.lineages_from_records(decoded("BT", *recs))
    check_frame(frame, "lineages", len(BT_CASES))
    assert by_key(frame, "breeding_id") == {e["breeding_id"]: e for _, e in BT_CASES}
    # a lineage id names its parent line by its prefix: 0102 is under 01
    ids = dict(zip(frame["breeding_id"], frame["lineage_id"], strict=True))
    assert ids["1110000003"].startswith(ids["1110000001"])


# ------------------------------------------------------------------------------------------------
# The build


HN_RECORDS = [
    # two versions, the newer first in the file: the newer one is kept
    hn(breeding_id="1110000003", created="20250602", kubun="2", name="センノチチ",
       sire="1110000005", dam="1120000006"),
    hn(breeding_id="1110000003", created="20200101", kubun="1", name="フルイナマエ",
       sire="1110000099", dam="1120000099"),
    hn(breeding_id="1110000001", created="20240101", name="カクウシュボバ", sire="1110000003",
       dam="1120000006"),
    # a deletion: no row
    hn(breeding_id="1110000444", created="20190101", kubun="1", name="ケサレタウマ"),
    hn(breeding_id="1110000444", created="20190505", kubun="0", name="ケサレタウマ"),
    hn(breeding_id="1120000006", created="20230101", name="ナゾノハハ", sex="2",
       sire="1110000005", dam=ZERO),
]  # fmt: skip
BREEDING_IDS = ["1110000001", "1110000003", "1120000006"]
SK_RECORDS = [
    sk(horse_id="2022100002", created="20250602", kubun="2", pedigree=ancestry(5)),
    sk(horse_id="2022100002", created="20230101", kubun="1", pedigree=ancestry(6)),
    sk(horse_id="2021105001", created="20250101", pedigree=ancestry(1)),
    sk(horse_id="2020109999", created="20210101", kubun="1"),
    sk(horse_id="2020109999", created="20220101", kubun="0"),
]  # fmt: skip
HORSE_IDS = ["2021105001", "2022100002"]
BT_RECORDS = [
    bt(breeding_id="1110000003", created="20250101", lineage_id="0102", lineage_name="架空分岐系"),
    bt(breeding_id="1110000001", created="20250101", lineage_id="01", lineage_name="架空系"),
    bt(breeding_id="1110000001", created="20240101", lineage_id="09", lineage_name="旧系統"),
    bt(breeding_id="1110000444", created="20200101", kubun="0", lineage_id="03"),
]  # fmt: skip
LINEAGE_IDS = ["1110000001", "1110000003"]


def write_build_inputs(root: Path) -> None:
    write_inputs(root, HN_RECORDS, SK_RECORDS, BT_RECORDS)


def assert_written_schema(path: Path, table: str, module_schema: pa.Schema) -> None:
    written = pq.read_schema(path).remove_metadata()
    assert written.equals(module_schema.remove_metadata()), (table, written)
    assert [(f.name, f.type) for f in written] == TYPES[table], table


def test_build_writes_the_three_tables_with_their_schemas(data: Path) -> None:
    write_build_inputs(data / "raw" / SOURCE)
    report = bloodlines.build_bloodlines()
    assert isinstance(report, records.BuildReport)
    assert report.name == "bloodlines"

    processed = data / "processed" / SOURCE
    module_schemas = {
        "breeding_horses": bloodlines.BREEDING_HORSES_SCHEMA,
        "produce": bloodlines.PRODUCE_SCHEMA,
        "lineages": bloodlines.LINEAGES_SCHEMA,
    }
    for table, schema in module_schemas.items():
        assert_written_schema(table_path(processed, table), table, schema)

    # every key's latest version, deletions dropped, sorted by the key
    horses = read_rows(processed, "breeding_horses")
    produce = read_rows(processed, "produce")
    lineages = read_rows(processed, "lineages")
    assert [r["breeding_id"] for r in horses] == BREEDING_IDS
    assert [r["horse_id"] for r in produce] == HORSE_IDS
    assert [r["breeding_id"] for r in lineages] == LINEAGE_IDS
    h = keyed(horses, "breeding_horses")
    assert (h["1110000003"]["name"], h["1110000003"]["sire_breeding_id"]) == (
        "センノチチ", "1110000005",
    )  # fmt: skip  # the newer version, though first in the file
    assert h["1110000001"]["colour"] == "架空色ア" and h["1120000006"]["sex"] == "M"
    assert h["1120000006"]["dam_breeding_id"] is None
    p = keyed(produce, "produce")
    assert p["2022100002"]["pedigree_breeding_ids"] == list(ancestry(5))
    assert p["2021105001"]["pedigree_breeding_ids"] == list(ancestry(1))
    assert p["2021105001"]["birth_date"] == date(2021, 3, 15)
    lin = keyed(lineages, "lineages")
    assert (lin["1110000001"]["lineage_id"], lin["1110000001"]["lineage_name"]) == ("01", "架空系")
    assert report.counts["breeding_horses"] == 3
    assert report.counts["produce"] == 2
    assert report.counts["lineages"] == 2

    # `raw` is the export root and `out` the folder written; the defaults stay as they were
    elsewhere = data / "elsewhere" / "export"
    write_inputs(
        elsewhere,
        [hn(breeding_id="1110005555", name="ホカノウマ")],
        [sk(horse_id="2023100055", pedigree=ancestry(9))],
        [bt(breeding_id="1110005555", lineage_id="07")],
    )
    out = data / "elsewhere" / "out"
    other = bloodlines.build_bloodlines(raw=elsewhere, out=out)
    assert [r["breeding_id"] for r in read_rows(out, "breeding_horses")] == ["1110005555"]
    assert [r["horse_id"] for r in read_rows(out, "produce")] == ["2023100055"]
    assert [r["lineage_id"] for r in read_rows(out, "lineages")] == ["07"]
    assert (other.counts["breeding_horses"], other.counts["produce"]) == (1, 1)
    for table, schema in module_schemas.items():
        assert_written_schema(table_path(out, table), table, schema)
    assert [r["breeding_id"] for r in read_rows(processed, "breeding_horses")] == BREEDING_IDS

    # None of the three is a store table: the store refuses them (their views: below).
    for table in TABLE_NAMES:
        assert table not in schemas.TABLES, table
        with pytest.raises(ValueError, match="not a store table"):
            tables.read_table(SOURCE, table)
    con = tables.connect(SOURCE)
    try:
        listed = {
            row[0]
            for row in con.execute("SELECT table_name FROM information_schema.tables").fetchall()
        }
        assert not listed & set(TABLE_NAMES), listed
    finally:
        con.close()


# Before every birth year and date below, inside them, after the export (2025-06-02), long after.
STATIC_TIMES = [
    datetime(1900, 1, 1, tzinfo=UTC),
    datetime(2009, 3, 1, tzinfo=UTC),
    datetime(2021, 3, 14, 15, 0, tzinfo=UTC),  # 2021-03-15 00:00 JST, a foal's birth date
    datetime(2025, 6, 1, 6, 40, tzinfo=UTC),
    datetime(2100, 1, 1, tzinfo=UTC),
]


def test_the_bloodlines_tables_are_static_in_every_view_at_any_t(data: Path) -> None:
    write_build_inputs(data / "raw" / SOURCE)
    bloodlines.build_bloodlines()
    processed = data / "processed" / SOURCE
    stored = {
        table: by_key(pq.read_table(table_path(processed, table)).to_pandas(), SORT_KEY[table])
        for table in TABLE_NAMES
    }
    written = {"breeding_horses": BREEDING_IDS, "produce": HORSE_IDS, "lineages": LINEAGE_IDS}

    rules = load_availability("japan")
    for table in TABLE_NAMES:
        rule = rules[table]
        assert rule.kind == "static", table
        assert (rule.label_columns, rule.label_prefixes) == ((), ()), table

    con = duckdb.connect(":memory:")
    try:
        for table in TABLE_NAMES:
            path = table_path(processed, table)
            con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path}')")
        for t in STATIC_TIMES:
            for table in TABLE_NAMES:
                view = read_asof(con, table, t, rules=rules)
                assert list(view.columns) == [name for name, _ in TYPES[table]], (table, t)
                rows = by_key(view, SORT_KEY[table])
                assert sorted(rows) == written[table], (table, t)
                assert rows == stored[table], (table, t)  # every row, the same values at every t
            produce = by_key(read_asof(con, "produce", t, rules=rules), "horse_id")
            assert produce["2022100002"]["pedigree_breeding_ids"] == list(ancestry(5)), t
            assert produce["2021105001"]["birth_date"] == date(2021, 3, 15), t

        # asof_join matches on the key alone, whatever the key's t; a key without a row (a
        # deletion, an unknown id) gets nulls
        keys = pd.DataFrame(
            {
                "breeding_id": ["1110000003", "1120000006", "1110000444", "1110000001"],
                "t_utc": pd.to_datetime(STATIC_TIMES[:4], utc=True),
            }
        )
        got = asof_join(keys, "breeding_horses", on=["breeding_id"], rules=rules, con=con)
        assert [plain(v) for v in got["name"]] == [
            "センノチチ", "ナゾノハハ", None, "カクウシュボバ",
        ]  # fmt: skip
        assert [plain(v) for v in got["sire_breeding_id"]] == [
            "1110000005", "1110000005", None, "1110000003",
        ]  # fmt: skip
        got = asof_join(keys, "lineages", on=["breeding_id"], rules=rules, con=con)
        assert [plain(v) for v in got["lineage_id"]] == ["0102", None, None, "01"]
        keys = pd.DataFrame(
            {
                "horse_id": ["2021105001", "2020109999", "2022100002"],
                "t_utc": pd.to_datetime([STATIC_TIMES[0], STATIC_TIMES[3], STATIC_TIMES[4]],
                                        utc=True),
            }
        )  # fmt: skip
        got = asof_join(keys, "produce", on=["horse_id"], rules=rules, con=con)
        assert [plain(v) for v in got["birth_date"]] == [date(2021, 3, 15), None, date(2021, 3, 15)]
        assert [plain(v) for v in got["pedigree_breeding_ids"]] == [
            list(ancestry(1)), None, list(ancestry(5)),
        ]  # fmt: skip

        # pedigree, not labels: read_label refuses them
        for table in TABLE_NAMES:
            with pytest.raises(ValueError):
                read_label(con, table, rules=rules)
    finally:
        con.close()


# ------------------------------------------------------------------------------------------------
# The report


A, B, C, D, F = "1000000001", "1000000002", "1000000003", "1000000004", "1000000006"
E = "1000000005"
GONE = "1000000099"  # an HN key whose latest record is a deletion
REPORT_HN = [
    hn(breeding_id=A, sire=ZERO, dam=None),
    hn(breeding_id=B, sire=A, dam="1999999999"),
    hn(breeding_id=C, sire=B, dam=A),
    hn(breeding_id=D, sire=GONE, dam=B),
    hn(breeding_id=GONE, created="20200101", kubun="1", sire=ZERO),
    hn(breeding_id=GONE, created="20210101", kubun="0", sire=ZERO),
    # the older version's sire resolves, the kept one's does not
    hn(breeding_id=E, created="20200101", sire=A),
    hn(breeding_id=E, created="20250101", sire="3000000000"),
    hn(breeding_id=F, sire=A, dam=C),
]
REPORT_SK = [
    sk(horse_id="2020100001",
       pedigree=(A, B, ZERO, None, C, GONE, "4000000001", A, A, D, E, F, None, "4000000002")),
    sk(horse_id="2020100002", created="20250101", pedigree=(F, "4000000003", *[ZERO] * 12)),
    sk(horse_id="2020100002", created="20200101", pedigree=(A,) * 14),  # superseded
    sk(horse_id="2020100003", created="20200101", kubun="1", pedigree=(A,) * 14),
    sk(horse_id="2020100003", created="20210101", kubun="0", pedigree=(A,) * 14),  # deleted
]  # fmt: skip
REPORT_BT = [
    bt(breeding_id=A, lineage_id="01"),
    bt(breeding_id=GONE, lineage_id="0101"),
    bt(breeding_id="5000000001", lineage_id="02"),
    bt(breeding_id=C, lineage_id="0102"),
    bt(breeding_id=F, created="20200101", lineage_id="03"),
    bt(breeding_id=F, created="20250101", lineage_id="0301"),
    bt(breeding_id=B, created="20200101", kubun="1", lineage_id="04"),
    bt(breeding_id=B, created="20210101", kubun="0", lineage_id="04"),
]
# Counted by hand from the records above (values, repeats included; nulls never count):
# HN rows A B C D E F; sires B->A, C->B, D->GONE, E->3000000000, F->A (A's is zeros).
# SK lists: 11 ids in the first (8 written breeding ids), 2 in the second (1).
# BT rows A GONE 5000000001 C F (B deleted); A, C and F resolve.
REPORT_COUNTS = {
    "breeding_horses": 6, "produce": 2, "lineages": 5,
    "sire_ids": 5, "sire_ids_resolved": 3,
    "pedigree_ids": 13, "pedigree_ids_resolved": 9,
    "lineages_resolved": 3,
}  # fmt: skip
# The same SK and BT records with the absent ids added to HN (their own sires zeros): every
# count that follows a breeding id moves.
ABSENT = ("4000000001", "4000000002", "4000000003", "5000000001", "3000000000")
MORE_HN = [hn(breeding_id=i, sire=ZERO, dam=ZERO) for i in ABSENT]
MORE_COUNTS = {
    "breeding_horses": 11, "produce": 2, "lineages": 5,
    "sire_ids": 5, "sire_ids_resolved": 4,
    "pedigree_ids": 13, "pedigree_ids_resolved": 12,
    "lineages_resolved": 4,
}  # fmt: skip


def counted(report: records.BuildReport) -> dict[str, int]:
    missing = [k for k in COUNT_KEYS if k not in report.counts]
    assert not missing, missing
    assert all(type(v) is int for v in report.counts.values()), report.counts
    return {k: report.counts[k] for k in COUNT_KEYS}


def recount(out: Path) -> dict[str, int]:
    """The counts from the written tables alone."""
    ids = {r["breeding_id"] for r in read_rows(out, "breeding_horses")}
    sires = [r["sire_breeding_id"] for r in read_rows(out, "breeding_horses")]
    sires = [s for s in sires if s is not None]
    pedigree = [i for r in read_rows(out, "produce") for i in r["pedigree_breeding_ids"]]
    pedigree = [i for i in pedigree if i is not None]
    lineages = [r["breeding_id"] for r in read_rows(out, "lineages")]
    return {
        "breeding_horses": len(ids),
        "produce": len(read_rows(out, "produce")),
        "lineages": len(lineages),
        "sire_ids": len(sires),
        "sire_ids_resolved": sum(s in ids for s in sires),
        "pedigree_ids": len(pedigree),
        "pedigree_ids_resolved": sum(i in ids for i in pedigree),
        "lineages_resolved": sum(b in ids for b in lineages),
    }


def test_the_report_counts_resolved_ids(data: Path, tmp_path: Path) -> None:
    root = data / "raw" / SOURCE
    write_inputs(root, REPORT_HN, REPORT_SK, REPORT_BT)
    report = bloodlines.build_bloodlines()
    assert report.name == "bloodlines"
    assert counted(report) == REPORT_COUNTS
    assert recount(data / "processed" / SOURCE) == REPORT_COUNTS

    other = tmp_path / "more"
    write_inputs(other, [*REPORT_HN, *MORE_HN], REPORT_SK, REPORT_BT)
    out = tmp_path / "more-out"
    more = bloodlines.build_bloodlines(raw=other, out=out)
    assert counted(more) == MORE_COUNTS
    assert recount(out) == MORE_COUNTS

    # The command writes the report through write_build_report: the counts, no id or name.
    result = CliRunner().invoke(app, ["japan", "jravan", "bloodlines"])
    assert result.exit_code == 0, result.output
    written = paths.reports_dir() / "jravan-bloodlines.json"
    assert written.is_file()
    text = written.read_text(encoding="utf-8")
    parsed = json.loads(text)
    assert parsed["name"] == "bloodlines"
    assert {k: parsed["counts"][k] for k in COUNT_KEYS} == REPORT_COUNTS
    for secret in (A, C, GONE, "4000000001", "5000000001", "カクウシュボバ", "架空色ア",
                   "架空系"):  # fmt: skip
        assert secret not in text and secret not in result.output, secret


# ------------------------------------------------------------------------------------------------
# Blank ids


BLANK_HN = [  # (breeding_id, 血統登録番号, sire, dam) -> (horse_id, sire, dam)
    ("1110000101", ZERO, "1110000003", "1120000006"),
    ("1110000102", None, ZERO, "1120000006"),
    ("1110000103", "2001100103", None, ZERO),
    ("1110000104", "  ", ZERO, None),
    ("1110000105", "0002100105", "0000000003", "1120000006"),
]
BLANK_HN_EXPECTED = {
    "1110000101": (None, "1110000003", "1120000006"),
    "1110000102": (None, None, "1120000006"),
    "1110000103": ("2001100103", None, None),
    "1110000104": (None, None, None),
    "1110000105": ("0002100105", "0000000003", "1120000006"),  # leading zeros are an id
}
BLANK_SK = {  # horse_id -> 14 printed values (zeros and blank are null, in place)
    "2020100701": (ZERO, *ancestry(7)[1:]),
    "2020100702": (*ancestry(8)[:6], None, ZERO, *ancestry(8)[8:]),
    "2020100703": (*ancestry(9)[:13], None),
    "2020100704": (None,) * 7 + (ZERO,) * 7,
}
BLANK_BT = [("1110000101", "01"), ("1110000102", None)]  # (breeding_id, 系統ID)


def nulled(values: Sequence[str | None]) -> list[str | None]:
    return [None if v is None or set(v) <= {"0", " "} else v for v in values]


def test_blank_ids_are_null(data: Path) -> None:
    hns = [hn(breeding_id=b, horse_id=h, sire=s, dam=d) for b, h, s, d in BLANK_HN]
    frame = bloodlines.breeding_horses_from_records(decoded("HN", *hns))
    got = {
        k: (r["horse_id"], r["sire_breeding_id"], r["dam_breeding_id"])
        for k, r in by_key(frame, "breeding_id").items()
    }
    assert got == BLANK_HN_EXPECTED

    sks = [sk(horse_id=h, pedigree=ids) for h, ids in BLANK_SK.items()]
    frame = bloodlines.produce_from_records(decoded("SK", *sks))
    got = {k: r["pedigree_breeding_ids"] for k, r in by_key(frame, "horse_id").items()}
    assert got == {h: nulled(ids) for h, ids in BLANK_SK.items()}
    assert got["2020100704"] == [None] * 14
    assert got["2020100702"][6:8] == [None, None] and got["2020100702"][8] == ancestry(8)[8]

    # breeder code: blank or zeros null; years and dates likewise
    unset = [
        sk(horse_id="2020100801", breeder="00000000", imported="0000", born="00000000"),
        sk(horse_id="2020100802", breeder=None, imported=None, born=None),
        sk(horse_id="2020100803", breeder="00000042", imported="2021", born="20200401"),
    ]
    frame = bloodlines.produce_from_records(decoded("SK", *unset))
    got = {
        k: (r["breeder_code"], r["import_year"], r["birth_date"])
        for k, r in by_key(frame, "horse_id").items()
    }
    assert got == {
        "2020100801": (None, None, None),
        "2020100802": (None, None, None),
        "2020100803": ("00000042", 2021, date(2020, 4, 1)),
    }
    frame = bloodlines.breeding_horses_from_records(
        decoded("HN", hn(breeding_id="1110000201", born="0000", imported="    "))
    )
    assert (plain(frame["birth_year"].iloc[0]), plain(frame["import_year"].iloc[0])) == (None, None)

    # through the build: nulls in the files, not empty strings or zeros, and never resolved
    bts = [bt(breeding_id=b, lineage_id=lineage) for b, lineage in BLANK_BT]
    write_inputs(data / "raw" / SOURCE, hns, sks, bts)
    report = bloodlines.build_bloodlines()
    processed = data / "processed" / SOURCE
    horses = keyed(read_rows(processed, "breeding_horses"), "breeding_horses")
    assert {
        k: (r["horse_id"], r["sire_breeding_id"], r["dam_breeding_id"]) for k, r in horses.items()
    } == BLANK_HN_EXPECTED
    produce = keyed(read_rows(processed, "produce"), "produce")
    assert {k: r["pedigree_breeding_ids"] for k, r in produce.items()} == {
        h: nulled(ids) for h, ids in BLANK_SK.items()
    }
    assert all(len(r["pedigree_breeding_ids"]) == 14 for r in produce.values())
    lineages = keyed(read_rows(processed, "lineages"), "lineages")
    assert lineages["1110000102"]["lineage_id"] is None
    non_null = sum(v is not None for ids in BLANK_SK.values() for v in nulled(ids))
    assert report.counts["pedigree_ids"] == non_null == 13 + 12 + 13
    assert report.counts["sire_ids"] == 2  # 1110000003 and 0000000003
    assert report.counts["sire_ids_resolved"] == 0


# ------------------------------------------------------------------------------------------------
# Byte stability


def test_two_builds_write_identical_bytes(data: Path) -> None:
    write_inputs(data / "raw" / SOURCE, REPORT_HN, REPORT_SK, REPORT_BT)
    first_report = bloodlines.build_bloodlines()
    processed = data / "processed" / SOURCE
    first = {t: table_path(processed, t).read_bytes() for t in TABLE_NAMES}

    again = data / "again"
    second_report = bloodlines.build_bloodlines(out=again)
    for table, blob in first.items():
        assert table_path(again, table).read_bytes() == blob, table
    assert second_report.to_json() == first_report.to_json()

    # the same inputs rewritten (new mtimes), built over the existing files
    write_inputs(data / "raw" / SOURCE, REPORT_HN, REPORT_SK, REPORT_BT)
    bloodlines.build_bloodlines()
    for table, blob in first.items():
        assert table_path(processed, table).read_bytes() == blob, table
    assert sorted(q.name for q in processed.iterdir()) == sorted(
        f"{t}.parquet" for t in TABLE_NAMES
    )

    result = CliRunner().invoke(app, ["japan", "jravan", "bloodlines"])
    assert result.exit_code == 0, result.output
    for table, blob in first.items():
        assert table_path(processed, table).read_bytes() == blob, table


# ------------------------------------------------------------------------------------------------
# The real export (local: needs $HORSETRACK_DATA_DIR with the export and the workbook)


@pytest.mark.local
def test_the_real_bloodlines_map(tmp_path: Path) -> None:
    export, workbook = wb.real_export_or_skip()
    bldn = export / "stored" / "BLDN"
    if not all((bldn / rt / "all.jvd").is_file() for rt in ("HN", "SK", "BT")):
        pytest.skip("no HN, SK or BT all.jvd in the export")

    report = bloodlines.build_bloodlines(raw=export, out=tmp_path)
    assert report.name == "bloodlines"
    module_schemas = {
        "breeding_horses": bloodlines.BREEDING_HORSES_SCHEMA,
        "produce": bloodlines.PRODUCE_SCHEMA,
        "lineages": bloodlines.LINEAGES_SCHEMA,
    }
    for table, schema in module_schemas.items():
        assert_written_schema(table_path(tmp_path, table), table, schema)

    # the keys, the sire ids and the pedigree lists expected, from the raw records with the
    # existing decoder and version rule
    def kept(rt: str, key: str) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
        cols = records.decode((bldn / rt / "all.jvd").read_bytes(), spec.record_spec(rt, workbook))
        index = pd.DataFrame(
            {k: records.text(cols[k]) for k in (key, "データ作成年月日", "データ区分")}
        )
        return cols, records.latest_versions(index, [key]).sort_values(key, kind="stable")

    def ids_of(raw: np.ndarray) -> np.ndarray:
        """Ids as the tables hold them: trimmed text, blank or all zeros None."""
        values = records.text(raw)
        flat = values.reshape(-1)
        flat[np.array([v is not None and set(v) == {"0"} for v in flat], dtype=bool)] = None
        return values

    horses = pq.read_table(table_path(tmp_path, "breeding_horses")).to_pandas()
    produce = pq.read_table(table_path(tmp_path, "produce"))
    lineages = pq.read_table(table_path(tmp_path, "lineages")).to_pandas()
    cols, latest = kept("HN", "繁殖登録番号")
    assert list(horses["breeding_id"]) == list(latest["繁殖登録番号"])
    sires = ids_of(cols["父馬繁殖登録番号"])[latest.index.to_numpy()]
    assert [plain(v) for v in horses["sire_breeding_id"]] == list(sires)
    del cols
    cols, latest = kept("SK", "血統登録番号")
    assert produce["horse_id"].to_pylist() == list(latest["血統登録番号"])
    lists = produce["pedigree_breeding_ids"].combine_chunks()
    assert (lists.value_lengths().to_numpy(zero_copy_only=False) == 14).all()
    got = np.array(lists.flatten().to_pylist(), dtype=object).reshape(-1, 14)
    expected = ids_of(cols[PEDIGREE])[latest.index.to_numpy()]
    assert got.shape == expected.shape
    assert bool((got == expected).all())
    del cols, got, expected, lists
    _, latest = kept("BT", "繁殖登録番号")
    assert list(lineages["breeding_id"]) == list(latest["繁殖登録番号"])
    assert lineages["lineage_id"].notna().all()

    # the vocabulary, and meanings that are the workbook's own
    colours = set(spec.code_table("2203", workbook).values.values())
    produce = produce.drop_columns(["pedigree_breeding_ids"]).to_pandas()
    for frame in (horses, produce):
        assert set(frame["sex"].dropna()) <= {"H", "M", "G"}
        assert set(frame["colour"].dropna()) <= colours
        assert frame["colour"].notna().any()
    assert produce["colour"].notna().mean() > 0.99
    ids = pd.concat([horses["horse_id"], horses["sire_breeding_id"], horses["dam_breeding_id"]])
    assert ids.dropna().str.fullmatch(r"\d{10}").all()
    assert not ids.dropna().str.fullmatch(r"0+").any()
    years = horses["birth_year"].dropna()
    assert ((years >= 1600) & (years <= 2026)).all()

    # the report's counts, independently from the written tables
    assert counted(report) == recount(tmp_path)
    c = report.counts
    assert 0 < c["sire_ids_resolved"] <= c["sire_ids"] < c["breeding_horses"]
    assert 0 < c["pedigree_ids_resolved"] <= c["pedigree_ids"] <= 14 * c["produce"]
    assert 0 < c["lineages_resolved"] <= c["lineages"]

    # the repository holds no code-table text: no colour meaning in the jravan sources
    sources = sorted(Path(bloodlines.__file__).parent.glob("*.py")) + [Path(__file__)]
    meanings = {m for m in colours if m and len(m) >= 2}
    for path in sources:
        text = path.read_text(encoding="utf-8")
        leaked = sorted(m for m in meanings if m in text)
        assert not leaked, f"{path.name} holds {len(leaked)} code-table meanings"
