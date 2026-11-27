"""JV-Data UM (horse), KS (jockey) and CH (trainer) masters into the jravan store's `horses` and
`horse_pedigree` and new `jockeys` and `trainers` tables (ticket #370).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jravan.masters`:
- `horses_from_records(cols)`, `pedigree_from_records(cols)` (UM), `jockeys_from_records(cols)`
  (KS) and `trainers_from_records(cols)` (CH), each `cols: Mapping[str, numpy.ndarray]` as
  `records.decode` returns it -> `pandas.DataFrame`, one row per record of `cols` (the build has
  already kept one version per key). `horses` and `horse_pedigree` rows carry the columns of
  `schemas.TABLES[...]`; `jockeys` and `trainers` rows the names of `JOCKEYS_SCHEMA` and
  `TRAINERS_SCHEMA`. The colour and mark meanings are read with `spec.code_table` from the
  workbook at `spec.spec_path()` (under the data dir) at run time.
- `JOCKEYS_SCHEMA`, `TRAINERS_SCHEMA: pyarrow.Schema`: the columns below, dates `date32`,
  `is_retired` `bool`, every other column `string` (the column order is the coder's).
- `build_masters(*, raw=None, out=None) -> records.BuildReport` (name `masters`): `raw` is the
  export root (default `paths.raw_dir('jravan')`), read at `<raw>/stored/DIFN/UM/all.jvd`,
  `.../KS/all.jvd` and `.../CH/all.jvd` with the layouts `spec.record_spec('UM'|'KS'|'CH')`; one
  version per key (`records.latest_versions` on the layout's key item: 血統登録番号, 騎手コード,
  調教師コード; a key whose latest record has データ区分 `0` is a deletion and has no row; UM
  データ区分 `9`, a deregistration, is an ordinary version and is kept). Writes, in `out`
  (default `paths.processed_dir('jravan')`), through `records.write_parquet`:
  `horses.parquet` and `horse_pedigree.parquet` in `TABLES` column order and `tables.ARROW_TYPES`,
  sorted by `horse_id`; `jockeys.parquet` with `JOCKEYS_SCHEMA` sorted by `jockey_code`;
  `trainers.parquet` with `TRAINERS_SCHEMA` sorted by `trainer_code`. The same inputs give the
  same bytes.
- `horsetrack japan jravan masters`: `build_masters()` with the defaults, the report written by
  `records.write_build_report` (`<reports dir>/jravan-masters.json`, counts only: no name or id).

Each column and the workbook item it comes from (the `records.decode` key; names as
`spec.record_spec` returns them from the JV-Data 4.9.0.1 workbook on the owner's Mac). Every text
value is `records.text` (cp932, both ends trimmed of half- and full-width spaces, blank null).
- horses (UM): `jurisdiction` `jra`. `horse_id`: 血統登録番号 as printed (ten digits). `name`:
  馬名 (馬名半角ｶﾅ and 馬名欧字 are not used). `origin`: 産地名. `colour`: the meaning of 毛色コード
  in code table 2203 (`spec.code_table('2203').values[code]`, its first content column).
  `import_type`: the meaning of 馬記号コード in code table 2204, the same way. Both are null for
  the unset code (`00`), a blank code and a code the table lacks. `sex`: 性別コード `1` H, `2` M,
  `3` G (code table 2202), anything else null. `sire`, `dam`, `dam_sire`: 3代血統情報/馬名 of
  occurrences 1, 2 and 5 of the 14 (the order is 父, 母, 父父, 父母, 母父, 母母, ...).
  `is_retired`: 競走馬抹消区分 `1` true, `0` false. `profile_scraped_utc`: null.
- horse_pedigree (UM): `jurisdiction` `jra`; `horse_id` as on horses; `sire_name`, `dam_name`,
  `dam_sire_name`: the same three names; `*_name_norm`: `unicodedata.normalize('NFKC', name)
  .casefold()` (as `jra/netkeiba/parse_pedigree_card._norm`), null for a null name; `sire_id`,
  `dam_id`, `dam_sire_id`: 3代血統情報/繁殖登録番号 of the same occurrences, as printed, null when
  blank or all zeros. `source_race_id` and `scraped_utc`: null.
- jockeys (KS): `jurisdiction` `jra`; `jockey_code`: 騎手コード as printed; `jockey_key`
  `jockey:<code>` (as `runners.jockey_key`); `name`: 騎手名; `name_short`: 騎手名略称;
  `birth_date`: 生年月日; `licence_issued`: 騎手免許交付年月日; `licence_revoked`:
  騎手免許抹消年月日 (dates `YYYYMMDD`, null when blank or `00000000`); `is_retired`: 騎手抹消区分
  `1`; `affiliation`: 騎手東西所属コード (code table 2301) `1` miho, `2` ritto, `3` nar, `4`
  overseas, anything else null; `trainer_code`: 所属調教師コード (the jockey's stable), null when
  blank or `00000` (a freelance jockey); `apprentice_code`: 騎手見習コード as printed (`0` stays
  `0`; code table 2303), blank null. 騎手名半角ｶﾅ, 騎手名欧字, 性別区分, 騎乗資格コード,
  招待地域名, 所属調教師名略称 and the record groups are not used.
- trainers (CH): `jurisdiction` `jra`; `trainer_code`: 調教師コード; `trainer_key`
  `trainer:<code>`; `name`: 調教師名; `name_short`: 調教師名略称; `birth_date`: 生年月日;
  `licence_issued`: 調教師免許交付年月日; `licence_revoked`: 調教師免許抹消年月日; `is_retired`:
  調教師抹消区分 `1`; `affiliation`: 調教師東西所属コード, as for jockeys.

Store rules (`countries/japan/config/availability.yaml`): `horses` is `current_value` (never in a
view), `horse_pedigree` is `static` with labels `source_race_id` and `scraped_utc`. `jockeys` and
`trainers` are `current_value` with `in_view` (#376, the owner's request of 2026-11-26: "the
master row as exported is used, with no history"): every master row is in every view at any t,
unchanged, `asof_join` attaches it by key alone, and `read_label` refuses them. The row carries
the export's look-ahead: a jockey retired since shows as retired at a t before his licence.

The UM, KS and CH sections built here (`jravan_workbook`) use the workbook's item names with
invented widths, order and companions; every value and every code-table meaning is made up (D21:
the workbook and the export are licensed and stay under `$HORSETRACK_DATA_DIR`). Expected values
are written out by hand.
"""

from __future__ import annotations

import json
import unicodedata
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
from horsetrack.core.store.asof import (
    CurrentValueTable,
    asof_join,
    load_availability,
    read_asof,
    read_label,
)
from horsetrack.countries.japan.jravan import records, runners, spec

# isort: split
# (its own block: the sibling helper module, then the module this ticket creates)
import jravan_workbook as wb

from horsetrack.countries.japan.jravan import masters

CP = "cp932"
SOURCE = "jravan"
UTC = timezone.utc
HORSE_COLUMNS = list(schemas.TABLES["horses"].columns)
PEDIGREE_COLUMNS = list(schemas.TABLES["horse_pedigree"].columns)
DATES = ("birth_date", "licence_issued", "licence_revoked")
JOCKEY_TYPES = {
    "jurisdiction": pa.string(), "jockey_code": pa.string(), "jockey_key": pa.string(),
    "name": pa.string(), "name_short": pa.string(), "birth_date": pa.date32(),
    "licence_issued": pa.date32(), "licence_revoked": pa.date32(), "is_retired": pa.bool_(),
    "affiliation": pa.string(), "trainer_code": pa.string(), "apprentice_code": pa.string(),
}  # fmt: skip
TRAINER_TYPES = {
    "jurisdiction": pa.string(), "trainer_code": pa.string(), "trainer_key": pa.string(),
    "name": pa.string(), "name_short": pa.string(), "birth_date": pa.date32(),
    "licence_issued": pa.date32(), "licence_revoked": pa.date32(), "is_retired": pa.bool_(),
    "affiliation": pa.string(),
}  # fmt: skip
AFFILIATIONS = {"miho", "ritto", "nar", "overseas"}

# ------------------------------------------------------------------------------------------------
# Made-up layouts: the workbook's item names, invented widths, order and companions


@dataclass(frozen=True)
class Field:
    name: str
    width: int  # one occurrence (a group: the sum of its subs)
    repeat: int = 1
    key: bool = False
    subs: tuple[tuple[str, int], ...] = ()


PEDIGREE = "3代血統情報"
UM_LAYOUT: tuple[Field, ...] = (
    Field("レコード種別ID", 2),
    Field("データ区分", 1),
    Field("データ作成年月日", 8),
    Field("血統登録番号", 10, key=True),
    Field("競走馬抹消区分", 1),
    Field("競走馬登録年月日", 8),
    Field("競走馬抹消年月日", 8),
    Field("生年月日", 8),
    Field("馬名", 24),
    Field("馬名半角ｶﾅ", 12),
    Field("馬名欧字", 20),
    Field("予備", 3),
    Field("毛色コード", 2),
    Field("性別コード", 1),
    Field("馬記号コード", 2),
    Field(PEDIGREE, 46, 14, subs=(("繁殖登録番号", 10), ("馬名", 36))),
    Field("東西所属コード", 1),
    Field("調教師コード", 5),
    Field("産地名", 12),
    Field("馬主名(法人格無)", 16),
    Field("登録レース数", 3),
    Field("レコード区切", 2),
)
UM_ALIASES = {
    "kubun": "データ区分", "created": "データ作成年月日", "horse_id": "血統登録番号",
    "retired": "競走馬抹消区分", "registered": "競走馬登録年月日",
    "deregistered": "競走馬抹消年月日",
    "born": "生年月日", "name": "馬名", "kana": "馬名半角ｶﾅ", "latin": "馬名欧字",
    "colour": "毛色コード", "sex": "性別コード", "mark": "馬記号コード", "pedigree": PEDIGREE,
    "east_west": "東西所属コード", "trainer": "調教師コード", "origin": "産地名",
    "owner": "馬主名(法人格無)", "races": "登録レース数",
}  # fmt: skip


def ancestors(
    sire: tuple[str | None, str | None] = ("1110000001", "カクウチチ"),
    dam: tuple[str | None, str | None] = ("1120000002", "カクウハハ"),
    dam_sire: tuple[str | None, str | None] = ("1110000005", "カクウハハチチ"),
) -> tuple[tuple[str | None, str | None], ...]:
    """The 14 (繁殖登録番号, 馬名) occurrences: sire, dam, then 父父, 父母 as decoys, dam_sire at
    the 5th, the rest decoys too (a wrong occurrence picks a 祖先 name)."""
    out: list[tuple[str | None, str | None]] = [sire, dam]
    out += [(f"13000000{i:02d}", f"祖先{i:02d}") for i in (3, 4)]
    out.append(dam_sire)
    out += [(f"13000000{i:02d}", f"祖先{i:02d}") for i in range(6, 15)]
    return tuple(out)


UM_DEFAULTS: dict[str, Any] = {
    "kubun": "1", "created": "20250602", "horse_id": "2021105001", "retired": "0",
    "registered": "20230401", "deregistered": "00000000", "born": "20210315",
    "name": "カクウノウマ", "kana": "ｶｸｳ", "latin": "KAKUU", "colour": "03", "sex": "1",
    "mark": "00", "pedigree": ancestors(), "east_west": "1", "trainer": "01061",
    "origin": "架空町", "owner": "架空馬主", "races": "012",
}  # fmt: skip

KS_LAYOUT: tuple[Field, ...] = (
    Field("レコード種別ID", 2),
    Field("データ区分", 1),
    Field("データ作成年月日", 8),
    Field("騎手コード", 5, key=True),
    Field("騎手抹消区分", 1),
    Field("騎手免許交付年月日", 8),
    Field("騎手免許抹消年月日", 8),
    Field("生年月日", 8),
    Field("騎手名", 20),
    Field("予備", 4),
    Field("騎手名半角ｶﾅ", 10),
    Field("騎手名略称", 8),
    Field("騎手名欧字", 16),
    Field("性別区分", 1),
    Field("騎乗資格コード", 1),
    Field("騎手見習コード", 1),
    Field("騎手東西所属コード", 1),
    Field("招待地域名", 8),
    Field("所属調教師コード", 5),
    Field("所属調教師名略称", 8),
    Field("初勝利情報", 38, 2, subs=(("年月日場回日R", 16), ("血統登録番号", 10), ("馬名", 12))),
    Field("レコード区切", 2),
)
KS_ALIASES = {
    "kubun": "データ区分", "created": "データ作成年月日", "code": "騎手コード",
    "retired": "騎手抹消区分", "issued": "騎手免許交付年月日", "revoked": "騎手免許抹消年月日",
    "born": "生年月日", "name": "騎手名", "kana": "騎手名半角ｶﾅ", "name_short": "騎手名略称",
    "latin": "騎手名欧字", "gender": "性別区分", "licence": "騎乗資格コード",
    "apprentice": "騎手見習コード", "east_west": "騎手東西所属コード", "invited": "招待地域名",
    "stable": "所属調教師コード", "stable_name": "所属調教師名略称", "first_win": "初勝利情報",
}  # fmt: skip
KS_DEFAULTS: dict[str, Any] = {
    "kubun": "1", "created": "20250602", "code": "01126", "retired": "0", "issued": "20100301",
    "revoked": "00000000", "born": "19900105", "name": "架空　一郎", "kana": "ｶｸｳ",
    "name_short": "架空一", "latin": "KAKUU", "gender": "1", "licence": "1", "apprentice": "0",
    "east_west": "2", "invited": "", "stable": "00000", "stable_name": "",
    "first_win": (("2010040105010101", "2007100009", "ハツショウリ"),),
}  # fmt: skip

CH_LAYOUT: tuple[Field, ...] = (
    Field("レコード種別ID", 2),
    Field("データ区分", 1),
    Field("データ作成年月日", 8),
    Field("調教師コード", 5, key=True),
    Field("調教師抹消区分", 1),
    Field("調教師免許交付年月日", 8),
    Field("調教師免許抹消年月日", 8),
    Field("生年月日", 8),
    Field("調教師名", 20),
    Field("調教師名半角ｶﾅ", 10),
    Field("調教師名略称", 8),
    Field("調教師名欧字", 16),
    Field("性別区分", 1),
    Field("調教師東西所属コード", 1),
    Field("招待地域名", 8),
    Field(
        "最近重賞勝利情報",
        50,
        2,
        subs=(("年月日場回日R", 16), ("競走名本題", 12), ("血統登録番号", 10), ("馬名", 12)),
    ),
    Field("レコード区切", 2),
)
CH_ALIASES = {
    "kubun": "データ区分", "created": "データ作成年月日", "code": "調教師コード",
    "retired": "調教師抹消区分", "issued": "調教師免許交付年月日",
    "revoked": "調教師免許抹消年月日", "born": "生年月日", "name": "調教師名",
    "kana": "調教師名半角ｶﾅ", "name_short": "調教師名略称", "latin": "調教師名欧字",
    "gender": "性別区分", "east_west": "調教師東西所属コード", "invited": "招待地域名",
    "graded": "最近重賞勝利情報",
}  # fmt: skip
CH_DEFAULTS: dict[str, Any] = {
    "kubun": "1", "created": "20250602", "code": "01061", "retired": "0", "issued": "20050301",
    "revoked": "00000000", "born": "19700720", "name": "架空　調教", "kana": "ｶｸｳ",
    "name_short": "架空調", "latin": "KAKUU", "gender": "1", "east_west": "1", "invited": "",
    "graded": (("2024060105010111", "架空記念", "2021100001", "ジュウショウ"),),
}  # fmt: skip

# ------------------------------------------------------------------------------------------------
# Made-up code tables: the workbook's table numbers and names, invented meanings. The decoys carry
# the same codes, so a wrong table gives a wrong meaning.

COLOURS = {"01": "架空色ア", "03": "架空色ウ", "07": "架空色キ", "11": "架空色サ"}
MARKS = {"03": "(架父)", "06": "(架外)", "21": "[架地]"}
OTHER_COLOURS = {"01": "別の色ア色", "03": "別の色ウ色", "07": "別の色キ色", "11": "別の色サ色"}
OTHER_MARKS = {"03": "(別父)父", "06": "(別外)外", "21": "[別地]地"}


def code_tables(colours: dict[str, str], marks: dict[str, str]) -> list[wb.Table]:
    def rows(meanings: dict[str, str], unset: str) -> tuple[tuple[str, ...], ...]:
        return (("00", unset, "未設定（架空）"),) + tuple(
            (code, text, f"副{code}") for code, text in meanings.items()
        )

    return [
        wb.Table(
            "2201", "架空別表", 2, (("01", "誤りの表"), ("03", "誤りの表"), ("06", "誤りの表"))
        ),
        wb.Table("2202", "性別コード", 1, (("1", "架空牡"), ("2", "架空牝"), ("3", "架空騸"))),
        wb.Table("2203", "毛色コード", 2, rows(colours, "架空初期値")),
        wb.Table("2204", "馬記号コード", 2, rows(marks, "架空無印")),
        wb.Table("2301", "東西所属コード", 1, (("1", "架空東"), ("2", "架空西"), ("3", "架空地"))),
        wb.Table("2303", "騎手見習コード", 1, (("1", "架☆"), ("3", "架▲"))),
    ]


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


def write_workbook(
    path: Path, colours: dict[str, str] = COLOURS, marks: dict[str, str] = MARKS
) -> Path:
    other = wb.simple_section(1, "XQ", "架空その他", [("架空値", 4), ("馬名", 6)])
    return wb.write_workbook(
        path,
        [
            other,
            section(2, "UM", "架空競走馬マスタ", UM_LAYOUT),
            section(3, "KS", "架空騎手マスタ", KS_LAYOUT),
            section(4, "CH", "架空調教師マスタ", CH_LAYOUT),
        ],
        code_tables(colours, marks),
    )


def _field(value: Any, width: int) -> bytes:
    if value is None:
        out = b" " * width
    elif isinstance(value, bool):
        raise TypeError(value)
    elif isinstance(value, int):
        out = f"{value:0{width}d}".encode("ascii")
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
        elif f.subs:
            given = list(by_item[f.name])
            assert len(given) <= f.repeat, (f.name, given)
            given += [(None,) * len(f.subs)] * (f.repeat - len(given))
            for occurrence in given:
                out += b"".join(_field(v, w) for v, (_, w) in zip(occurrence, f.subs, strict=True))
        else:
            out += _field(by_item[f.name], f.width)
    assert len(out) == sum(f.width * f.repeat for f in layout)
    return out


def um(**over: Any) -> bytes:
    """One UM record (CRLF included): UM_DEFAULTS, then `over` by alias."""
    values = {**UM_DEFAULTS, **over}
    return _record("UM", UM_LAYOUT, {UM_ALIASES[k]: v for k, v in values.items()})


def ks(**over: Any) -> bytes:
    values = {**KS_DEFAULTS, **over}
    return _record("KS", KS_LAYOUT, {KS_ALIASES[k]: v for k, v in values.items()})


def ch(**over: Any) -> bytes:
    values = {**CH_DEFAULTS, **over}
    return _record("CH", CH_LAYOUT, {CH_ALIASES[k]: v for k, v in values.items()})


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp data dir with the made-up workbook in place."""
    d = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    write_workbook(d / "specs" / "jravan" / wb.WORKBOOK_NAME)
    return d


def write_master(root: Path, record_type: str, recs: list[bytes]) -> Path:
    """`recs` as `<root>/stored/DIFN/<type>/all.jvd` (`root` the export root)."""
    return wb.write_jvd(root / "stored" / "DIFN" / record_type / "all.jvd", recs)


def decoded(record_type: str, *recs: bytes) -> dict[str, np.ndarray]:
    return records.decode(b"".join(recs), spec.record_spec(record_type))


def plain(v: Any) -> Any:
    """A frame value as plain Python: None for any null, dates as `date`."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, np.datetime64):
        return pd.Timestamp(v).date()
    if isinstance(v, np.generic):
        return plain(v.item())
    if isinstance(v, float) and np.isnan(v):
        return None
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


def assert_subset(row: dict[str, Any], expected: dict[str, Any], label: Any) -> None:
    got = {k: row.get(k, "<missing>") for k in expected}
    assert got == expected, label


def assert_store_types(path: Path, table: str) -> None:
    declared = schemas.TABLES[table].columns
    schema = pq.read_schema(path)
    assert schema.names == list(declared), (table, schema.names)
    for column, kind in declared.items():
        assert schema.field(column).type == tables.ARROW_TYPES[kind], (table, column)


def norm(name: str | None) -> str | None:
    return None if name is None else unicodedata.normalize("NFKC", name).casefold()


# ------------------------------------------------------------------------------------------------
# horses


HORSE_CASES = [  # (UM overrides, expected horses columns)
    (
        dict(horse_id="2021105001", name="カクウノウマ", origin="架空町", colour="03", mark="00",
             sex="1", retired="0",
             pedigree=ancestors(("1110000001", "カクウチチ"), ("1120000002", "カクウハハ"),
                                ("1110000005", "カクウハハチチ"))),
        dict(horse_id="2021105001", name="カクウノウマ", origin="架空町", colour="架空色ウ",
             import_type=None, sex="H", sire="カクウチチ", dam="カクウハハ",
             dam_sire="カクウハハチチ", is_retired=False),
    ),
    (
        dict(horse_id="2018101234", name="ベツノウマ　", origin="架空村", colour="07", mark="06",
             sex="2", retired="1",
             pedigree=ancestors(("1110009001", "ＫＡＫＵＵ　Ｓｉｒｅ"), ("1120009002", "ベツハハ"),
                                ("1110009005", "Foreign Damsire"))),
        dict(horse_id="2018101234", name="ベツノウマ", origin="架空村", colour="架空色キ",
             import_type="(架外)", sex="M", sire="ＫＡＫＵＵ　Ｓｉｒｅ", dam="ベツハハ",
             dam_sire="Foreign Damsire", is_retired=True),
    ),
    (
        dict(horse_id="2009100777", name="センノウマ", origin="ｶｸｳ牧場", colour="11", mark="21",
             sex="3", retired="1",
             pedigree=ancestors(("1110000777", "センチチ"), ("1120000778", "センハハ"),
                                ("1110000779", "センハハチチ"))),
        dict(horse_id="2009100777", name="センノウマ", origin="ｶｸｳ牧場", colour="架空色サ",
             import_type="[架地]", sex="G", sire="センチチ", dam="センハハ",
             dam_sire="センハハチチ", is_retired=True),
    ),
    (
        dict(horse_id="2022100002", name="ムジルシ", origin="", colour="01", mark="03", sex="1",
             retired="0",
             pedigree=ancestors(("1110000021", "ムジチチ"), ("1120000022", "ムジハハ"),
                                ("1110000025", "ムジハハチチ"))),
        dict(horse_id="2022100002", name="ムジルシ", origin=None, colour="架空色ア",
             import_type="(架父)", sex="H", sire="ムジチチ", dam="ムジハハ",
             dam_sire="ムジハハチチ", is_retired=False),
    ),
]  # fmt: skip


def test_a_horse_record_maps_to_horses_with_sire_dam_and_dam_sire(data: Path) -> None:
    frame = masters.horses_from_records(decoded("UM", *(um(**over) for over, _ in HORSE_CASES)))
    assert len(frame) == len(HORSE_CASES)
    missing = [c for c in HORSE_COLUMNS if c not in frame.columns]
    assert not missing, missing
    rows = by_key(frame, "horse_id")
    assert set(rows) == {e["horse_id"] for _, e in HORSE_CASES}
    for _, expected in HORSE_CASES:
        row = rows[expected["horse_id"]]
        assert_subset(
            row, {**expected, "jurisdiction": "jra", "profile_scraped_utc": None}, row["horse_id"]
        )


# ------------------------------------------------------------------------------------------------
# horse_pedigree


PEDIGREE_CASES = [  # (horse_id, sire, dam, dam_sire as (id, name)) -> expected ids
    ("2021105001", ("1110000001", "カクウチチ"), ("1120000002", "カクウハハ"),
     ("1110000005", "カクウハハチチ"), ("1110000001", "1120000002", "1110000005")),
    ("2018101234", ("1110009001", "ＫＡＫＵＵ　Ｓｉｒｅ"), ("1120009002", "ﾍﾞﾂﾊﾊ"),
     ("0000000000", "Ｆｏｒｅｉｇｎ　ＤＡＭＳＩＲＥ"), ("1110009001", "1120009002", None)),
    ("2009100777", ("0000000000", "Some Sire"), (None, "センハハ"), ("1110000779", "ｶﾀｶﾅ ﾁﾁ"),
     (None, None, "1110000779")),
    ("2022100002", ("1110000021", "ムジチチ"), ("1120000022", None), ("1110000025", "ムジハハチチ"),
     ("1110000021", "1120000022", "1110000025")),
]  # fmt: skip


def test_horse_pedigree_carries_breeding_ids_and_normalised_names(data: Path) -> None:
    recs = [um(horse_id=h, pedigree=ancestors(s, d, ds)) for h, s, d, ds, _ in PEDIGREE_CASES]
    frame = masters.pedigree_from_records(decoded("UM", *recs))
    assert len(frame) == len(PEDIGREE_CASES)
    missing = [c for c in PEDIGREE_COLUMNS if c not in frame.columns]
    assert not missing, missing
    rows = by_key(frame, "horse_id")
    for horse_id, sire, dam, dam_sire, ids in PEDIGREE_CASES:
        names = {"sire": sire[1], "dam": dam[1], "dam_sire": dam_sire[1]}
        expected: dict[str, Any] = {"jurisdiction": "jra", "source_race_id": None,
                                    "scraped_utc": None}  # fmt: skip
        for (column, name), breeding_id in zip(names.items(), ids, strict=True):
            expected[f"{column}_name"] = name
            expected[f"{column}_name_norm"] = norm(name)
            expected[f"{column}_id"] = breeding_id
        assert_subset(rows[horse_id], expected, horse_id)
    # the normal forms, written out: full-width letters and spaces fold, half-width kana widen
    assert rows["2018101234"]["sire_name_norm"] == "kakuu sire"
    assert rows["2018101234"]["dam_name_norm"] == "ベツハハ"
    assert rows["2018101234"]["dam_sire_name_norm"] == "foreign damsire"
    assert rows["2009100777"]["sire_name_norm"] == "some sire"
    assert rows["2009100777"]["dam_sire_name_norm"] == "カタカナ チチ"


# ------------------------------------------------------------------------------------------------
# The build


UM_RECORDS = [
    # a horse with two versions, the newer first in the file: the newer one is kept
    um(horse_id="2021105001", created="20250602", kubun="4", name="カクウノウマ", retired="0"),
    um(horse_id="2021105001", created="20240101", kubun="1", name="フルイナマエ", retired="0"),
    # deregistered (データ区分 9) is a version like any other
    um(horse_id="2009100777", created="20180301", kubun="9", name="センノウマ", retired="1",
       sex="3", colour="11", mark="21"),
    um(horse_id="2009100777", created="20110301", kubun="1", name="センノウマ", retired="0",
       sex="1", colour="11", mark="21"),
    # a deletion: no row
    um(horse_id="2019100404", created="20200101", kubun="1", name="ケサレルウマ"),
    um(horse_id="2019100404", created="20200202", kubun="0", name="ケサレルウマ"),
    um(horse_id="2018101234", created="20250101", kubun="4", name="ベツノウマ", sex="2",
       colour="07", mark="06", retired="1"),
]  # fmt: skip
HORSE_IDS = ["2009100777", "2018101234", "2021105001"]
KS_RECORDS = [
    ks(code="05339", created="20250602", kubun="2", name="架空　次郎", name_short="架空次",
       east_west="1", stable="01061", apprentice="3"),
    ks(code="01126", created="20250602", kubun="2", name="架空　一郎", name_short="架空一",
       east_west="2", retired="0"),
    ks(code="01126", created="20200101", kubun="1", name="架空　一郎", name_short="旧略称",
       east_west="1", retired="0"),
    ks(code="00666", created="20150101", kubun="2", name="架空　三郎", name_short="架空三",
       east_west="3", retired="1", revoked="20141231"),
    ks(code="00999", created="20150101", kubun="1", name="消えた騎手", name_short="消騎"),
    ks(code="00999", created="20160101", kubun="0", name="消えた騎手", name_short="消騎"),
]  # fmt: skip
JOCKEY_CODES = ["00666", "01126", "05339"]
CH_RECORDS = [
    ch(code="01061", created="20250602", kubun="2", name="架空　調教", name_short="架空調",
       east_west="1"),
    ch(code="00420", created="20250602", kubun="2", name="架空　師匠", name_short="架空師",
       east_west="2", retired="1", revoked="20200229"),
    ch(code="00420", created="20100101", kubun="1", name="架空　師匠", name_short="旧師",
       east_west="1", retired="0"),
]  # fmt: skip
TRAINER_CODES = ["00420", "01061"]


def write_build_inputs(root: Path) -> None:
    write_master(root, "UM", UM_RECORDS)
    write_master(root, "KS", KS_RECORDS)
    write_master(root, "CH", CH_RECORDS)


def assert_schema(path: Path, expected: pa.Schema, types: dict[str, pa.DataType]) -> None:
    written = pq.read_schema(path).remove_metadata()
    assert written.equals(expected.remove_metadata()), (path.name, written, expected)
    assert set(written.names) == set(types), (path.name, written.names)
    for name, kind in types.items():
        assert written.field(name).type == kind, (path.name, name)


def test_build_writes_the_four_tables_with_their_schemas(data: Path) -> None:
    write_build_inputs(data / "raw" / "jravan")
    report = masters.build_masters()
    assert isinstance(report, records.BuildReport)
    assert report.name == "masters"

    # the module's schemas are the columns pinned above
    assert set(masters.JOCKEYS_SCHEMA.names) == set(JOCKEY_TYPES)
    assert set(masters.TRAINERS_SCHEMA.names) == set(TRAINER_TYPES)
    for name, kind in JOCKEY_TYPES.items():
        assert masters.JOCKEYS_SCHEMA.field(name).type == kind, name
    for name, kind in TRAINER_TYPES.items():
        assert masters.TRAINERS_SCHEMA.field(name).type == kind, name

    processed = data / "processed" / "jravan"
    for table in ("horses", "horse_pedigree"):
        assert tables.table_path(SOURCE, table) == processed / f"{table}.parquet"
        assert_store_types(processed / f"{table}.parquet", table)
    assert_schema(processed / "jockeys.parquet", masters.JOCKEYS_SCHEMA, JOCKEY_TYPES)
    assert_schema(processed / "trainers.parquet", masters.TRAINERS_SCHEMA, TRAINER_TYPES)

    # every UM horse, one version each, deletions dropped, sorted by horse_id
    horses = tables.read_table(SOURCE, "horses")
    pedigree = tables.read_table(SOURCE, "horse_pedigree")
    assert list(horses["horse_id"]) == HORSE_IDS
    assert list(pedigree["horse_id"]) == HORSE_IDS
    h = by_key(horses, "horse_id")
    assert h["2021105001"]["name"] == "カクウノウマ"  # the newer version, though first in the file
    assert_subset(h["2009100777"], {"is_retired": True, "sex": "G", "colour": "架空色サ",
                                    "import_type": "[架地]"}, "deregistered")  # fmt: skip
    assert_subset(h["2018101234"], {"sex": "M", "colour": "架空色キ", "import_type": "(架外)"},
                  "2018101234")  # fmt: skip

    jockeys = pq.read_table(processed / "jockeys.parquet").to_pandas()
    trainers = pq.read_table(processed / "trainers.parquet").to_pandas()
    assert list(jockeys["jockey_code"]) == JOCKEY_CODES
    assert list(trainers["trainer_code"]) == TRAINER_CODES
    j = by_key(jockeys, "jockey_code")
    assert_subset(j["01126"], {"name_short": "架空一", "affiliation": "ritto"}, "01126 latest")
    assert_subset(j["05339"], {"trainer_code": "01061", "apprentice_code": "3"}, "05339")
    t = by_key(trainers, "trainer_code")
    assert_subset(t["00420"], {"name_short": "架空師", "is_retired": True,
                               "licence_revoked": date(2020, 2, 29)}, "00420")  # fmt: skip

    # `raw` is the export root and `out` the folder written
    elsewhere = data / "elsewhere" / "export"
    write_master(elsewhere, "UM", [um(horse_id="2020100001", name="ホカノウマ")])
    write_master(elsewhere, "KS", [ks(code="07777")])
    write_master(elsewhere, "CH", [ch(code="08888")])
    out = data / "elsewhere" / "out"
    masters.build_masters(raw=elsewhere, out=out)
    other = pq.read_table(out / "horses.parquet").to_pandas()
    assert list(other["horse_id"]) == ["2020100001"]
    assert list(pq.read_table(out / "jockeys.parquet").to_pandas()["jockey_code"]) == ["07777"]
    assert list(pq.read_table(out / "trainers.parquet").to_pandas()["trainer_code"]) == ["08888"]
    assert_store_types(out / "horse_pedigree.parquet", "horse_pedigree")
    # the default outputs were not touched by the second build
    assert list(tables.read_table(SOURCE, "horses")["horse_id"]) == HORSE_IDS


# ------------------------------------------------------------------------------------------------
# The views


def test_horses_stay_out_of_every_view_and_pedigree_is_static(data: Path) -> None:
    write_build_inputs(data / "raw" / "jravan")
    masters.build_masters()
    horses = tables.read_table(SOURCE, "horses")
    pedigree = tables.read_table(SOURCE, "horse_pedigree")
    assert horses["profile_scraped_utc"].isna().all()
    assert pedigree["source_race_id"].isna().all() and pedigree["scraped_utc"].isna().all()

    rules = load_availability("japan")
    # #376 leaves horses refused (the owner's #80 decision): no in_view
    assert (rules["horses"].kind, rules["horses"].in_view) == ("current_value", False)
    times = [
        datetime(1990, 1, 1, tzinfo=UTC),
        datetime(2009, 3, 1, 12, 0, tzinfo=UTC),
        datetime(2025, 6, 1, 6, 40, tzinfo=UTC),
        datetime(2031, 12, 31, tzinfo=UTC),
    ]
    expected = by_key(pedigree.drop(columns=["source_race_id", "scraped_utc"]), "horse_id")
    con = tables.connect(SOURCE)
    try:
        for t in times:
            with pytest.raises(CurrentValueTable):
                read_asof(con, "horses", t, rules=rules)
            view = read_asof(con, "horse_pedigree", t, rules=rules)
            assert "source_race_id" not in view.columns and "scraped_utc" not in view.columns, t
            assert by_key(view, "horse_id") == expected, (
                t
            )  # every horse at every t, the same values
    finally:
        con.close()


# Before any birth date or licence of the masters below, between them, after the 2014 and 2020
# revocations, after the export (created 2025-06-02), long after.
MASTER_TIMES = [
    datetime(1960, 1, 1, tzinfo=UTC),
    datetime(2012, 6, 1, 3, 0, tzinfo=UTC),
    datetime(2021, 1, 1, tzinfo=UTC),
    datetime(2025, 6, 1, 6, 40, tzinfo=UTC),
    datetime(2031, 12, 31, tzinfo=UTC),
]


def test_jockey_and_trainer_masters_are_in_every_view_as_exported(data: Path) -> None:
    write_build_inputs(data / "raw" / "jravan")
    masters.build_masters()
    processed = data / "processed" / "jravan"
    stored = {
        "jockeys": by_key(pq.read_table(processed / "jockeys.parquet").to_pandas(), "jockey_code"),
        "trainers": by_key(
            pq.read_table(processed / "trainers.parquet").to_pandas(), "trainer_code"
        ),
    }
    assert sorted(stored["jockeys"]) == JOCKEY_CODES and sorted(stored["trainers"]) == TRAINER_CODES

    rules = load_availability("japan")
    for table in ("jockeys", "trainers"):
        rule = rules[table]
        assert (rule.kind, rule.in_view) == ("current_value", True), table
        assert (rule.label_columns, rule.label_prefixes) == ((), ()), table
    assert "jockeys" not in schemas.TABLES and "trainers" not in schemas.TABLES

    con = duckdb.connect(":memory:")
    try:
        for table in ("jockeys", "trainers"):
            con.execute(
                f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{processed / table}.parquet')"
            )
        names = {"jockeys": set(JOCKEY_TYPES), "trainers": set(TRAINER_TYPES)}
        keys_of = {"jockeys": "jockey_code", "trainers": "trainer_code"}
        for t in MASTER_TIMES:
            for table in ("jockeys", "trainers"):
                view = read_asof(con, table, t, rules=rules)
                assert set(view.columns) == names[table], (table, t)
                # every master row at every t, unchanged: no history, no label column
                assert by_key(view, keys_of[table]) == stored[table], (table, t)
            # the export's look-ahead: 00666 retired with his licence revoked on 2014-12-31, and
            # 00420's revoked on 2020-02-29, show so at every t, before either too
            j = by_key(read_asof(con, "jockeys", t, rules=rules), "jockey_code")
            assert_subset(j["00666"], {"is_retired": True, "licence_revoked": date(2014, 12, 31),
                                       "affiliation": "nar"}, ("00666", t))  # fmt: skip
            assert_subset(j["01126"], {"is_retired": False, "licence_revoked": None,
                                       "name_short": "架空一"}, ("01126", t))  # fmt: skip
            tr = by_key(read_asof(con, "trainers", t, rules=rules), "trainer_code")
            assert_subset(tr["00420"], {"is_retired": True, "licence_revoked": date(2020, 2, 29),
                                        "affiliation": "ritto"}, ("00420", t))  # fmt: skip

        # asof_join attaches the master row by key alone, whatever the key's t (the runner
        # tables carry jockey_key and trainer_key); an unknown key gets nulls
        keys = pd.DataFrame(
            {
                "jockey_key": ["jockey:05339", "jockey:00666", "jockey:09999", "jockey:01126"],
                "t_utc": pd.to_datetime(MASTER_TIMES[:4], utc=True),
            }
        )
        got = asof_join(keys, "jockeys", on=["jockey_key"], rules=rules, con=con)
        assert list(got["jockey_key"]) == list(keys["jockey_key"])
        assert [plain(v) for v in got["jockey_code"]] == ["05339", "00666", None, "01126"]
        assert [plain(v) for v in got["name_short"]] == ["架空次", "架空三", None, "架空一"]
        assert [plain(v) for v in got["is_retired"]] == [False, True, None, False]
        assert [plain(v) for v in got["trainer_code"]] == ["01061", None, None, None]
        keys = pd.DataFrame(
            {
                "trainer_key": ["trainer:00420", "trainer:01061", "trainer:00000"],
                "t_utc": pd.to_datetime([MASTER_TIMES[0], MASTER_TIMES[4], MASTER_TIMES[2]],
                                        utc=True),
            }
        )  # fmt: skip
        got = asof_join(keys, "trainers", on=["trainer_key"], rules=rules, con=con)
        assert [plain(v) for v in got["name_short"]] == ["架空師", "架空調", None]
        assert [plain(v) for v in got["licence_revoked"]] == [date(2020, 2, 29), None, None]
        assert [plain(v) for v in got["affiliation"]] == ["ritto", "miho", None]

        # masters, not labels: read_label refuses them
        for table in ("jockeys", "trainers"):
            with pytest.raises(ValueError):
                read_label(con, table, rules=rules)
    finally:
        con.close()


# ------------------------------------------------------------------------------------------------
# Keys shared with runners


def se_cols(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    """SE arrays as `records.decode` gives them, for `runners.runners_from_records`: a plain
    placed runner per row at one race, with the row's jockey and trainer code and short names."""
    defaults = {
        runners.KUBUN: "7", runners.CREATED: "20250602", runners.YEAR: "2025",
        runners.MMDD: "0601", runners.VENUE: "05", runners.KAI: "03", runners.NICHI: "02",
        runners.RACE_NO: "11", runners.BRACKET: "1", runners.HORSE_NAME: "カクウノウマ",
        runners.SEX: "1", runners.AGE: "004", runners.CARRIED: "0570", runners.BODY_WT: "486",
        runners.CHANGE_SIGN: "+", runners.CHANGE: "002", runners.ABNORMAL: "0",
        runners.DEAD_HEAT: "0", runners.TIME: "1345", runners.MARGIN: "", runners.WIN_ODDS: "0123",
        runners.FAV_RANK: "01", runners.LAST_3F: "338", **dict.fromkeys(runners.CORNERS, "00"),
    }  # fmt: skip
    out: dict[str, list[bytes]] = {name: [] for name in runners.ITEMS}
    for i, row in enumerate(rows, start=1):
        values = {
            **defaults,
            runners.HORSE_NO: f"{i:02d}",
            runners.PLACE: f"{i:02d}",
            runners.HORSE_ID: f"20211000{i:02d}",
            runners.JOCKEY: row["jockey"],
            runners.JOCKEY_NAME: row["jockey_name"],
            runners.TRAINER: row["trainer"],
            runners.TRAINER_NAME: row["trainer_name"],
        }
        for name in runners.ITEMS:
            out[name].append(values[name].encode(CP))
    return {name: np.array(v) for name, v in out.items()}


def test_jockey_and_trainer_keys_match_the_runner_keys(data: Path) -> None:
    write_build_inputs(data / "raw" / "jravan")
    masters.build_masters()
    processed = data / "processed" / "jravan"
    jockeys = pq.read_table(processed / "jockeys.parquet").to_pandas()
    trainers = pq.read_table(processed / "trainers.parquet").to_pandas()
    for row in jockeys.to_dict("records"):
        assert row["jockey_key"] == f"jockey:{row['jockey_code']}", row["jockey_code"]
    for row in trainers.to_dict("records"):
        assert row["trainer_key"] == f"trainer:{row['trainer_code']}", row["trainer_code"]

    # runners built by the runners module from the same codes join on their keys
    rides = [
        dict(jockey="01126", jockey_name="架空一", trainer="01061", trainer_name="架空調"),
        dict(jockey="05339", jockey_name="架空次", trainer="00420", trainer_name="架空師"),
        dict(jockey="00666", jockey_name="架空三", trainer="01061", trainer_name="架空調"),
    ]
    field = runners.runners_from_records(se_cols(rides))
    assert len(field) == len(rides)
    joined = field.merge(jockeys, on="jockey_key", how="left", suffixes=("", "_master"))
    assert joined["jockey_code_master"].notna().all(), joined["jockey_key"].tolist()
    assert list(joined["name_short"]) == list(joined["jockey_name"])
    joined = field.merge(trainers, on="trainer_key", how="left", suffixes=("", "_master"))
    assert joined["trainer_code_master"].notna().all(), joined["trainer_key"].tolist()
    assert list(joined["name_short"]) == list(joined["trainer_name"])

    # a jockey's stable is a trainer code
    j = by_key(jockeys, "jockey_code")
    assert j["05339"]["trainer_code"] in set(trainers["trainer_code"])


# ------------------------------------------------------------------------------------------------
# The vocabulary


SEX_CASES = [("1", "H"), ("2", "M"), ("3", "G"), ("0", None)]
RETIRED_CASES = [("0", False), ("1", True)]
AFFILIATION_CASES = [("1", "miho"), ("2", "ritto"), ("3", "nar"), ("4", "overseas"), ("0", None)]


def test_sex_retirement_and_affiliation_use_the_store_vocabulary(data: Path) -> None:
    # horses: sex and retirement
    recs = []
    expected: dict[str, tuple[str | None, bool]] = {}
    n = 0
    for sex, want_sex in SEX_CASES:
        for retired, want_retired in RETIRED_CASES:
            n += 1
            horse_id = f"20201000{n:02d}"
            recs.append(um(horse_id=horse_id, sex=sex, retired=retired))
            expected[horse_id] = (want_sex, want_retired)
    rows = by_key(masters.horses_from_records(decoded("UM", *recs)), "horse_id")
    assert {k: (r["sex"], r["is_retired"]) for k, r in rows.items()} == expected

    # jockeys and trainers: affiliation and retirement
    for build, record, key in (
        (masters.jockeys_from_records, ks, "jockey_code"),
        (masters.trainers_from_records, ch, "trainer_code"),
    ):
        recs, want = [], {}
        n = 0
        for east_west, affiliation in AFFILIATION_CASES:
            for retired, is_retired in RETIRED_CASES:
                n += 1
                code = f"02{n:03d}"
                recs.append(record(code=code, east_west=east_west, retired=retired))
                want[code] = (affiliation, is_retired, "jra")
        rt = "KS" if record is ks else "CH"
        got = by_key(build(decoded(rt, *recs)), key)
        assert {k: (r["affiliation"], r["is_retired"], r["jurisdiction"])
                for k, r in got.items()} == want, rt  # fmt: skip
        assert {r["affiliation"] for r in got.values()} - {None} <= AFFILIATIONS

    # jockeys: the stable (00000 is freelance: null) and the apprentice code as printed
    stable_cases = [("01061", "01061", "0", "0"), ("00000", None, "1", "1"),
                    ("00420", "00420", "9", "9"), (None, None, None, None)]  # fmt: skip
    recs = [ks(code=f"03{i:03d}", stable=stable, apprentice=apprentice)
            for i, (stable, _, apprentice, _) in enumerate(stable_cases)]  # fmt: skip
    got = by_key(masters.jockeys_from_records(decoded("KS", *recs)), "jockey_code")
    for i, (_, trainer_code, _, apprentice_code) in enumerate(stable_cases):
        row = got[f"03{i:03d}"]
        assert (row["trainer_code"], row["apprentice_code"]) == (trainer_code, apprentice_code), i


# ------------------------------------------------------------------------------------------------
# Code tables from the workbook


def test_colour_and_mark_meanings_come_from_the_workbook(data: Path) -> None:
    colour_codes = ["01", "03", "07", "11", "00", "  ", "99"]
    mark_codes = ["03", "06", "21", "00", "  ", "99", "03"]
    recs = [um(horse_id=f"20191000{i:02d}", colour=c, mark=m)
            for i, (c, m) in enumerate(zip(colour_codes, mark_codes, strict=True))]  # fmt: skip
    write_master(data / "raw" / "jravan", "UM", recs)
    write_master(data / "raw" / "jravan", "KS", [ks()])
    write_master(data / "raw" / "jravan", "CH", [ch()])

    def meanings(colours: dict[str, str], marks: dict[str, str]) -> dict[str, tuple]:
        return {
            f"20191000{i:02d}": (colours.get(c), marks.get(m))  # unset, blank, unknown: null
            for i, (c, m) in enumerate(zip(colour_codes, mark_codes, strict=True))
        }

    # in memory and through the build, with the first workbook's meanings
    frame = masters.horses_from_records(decoded("UM", *recs))
    got = {k: (r["colour"], r["import_type"]) for k, r in by_key(frame, "horse_id").items()}
    assert got == meanings(COLOURS, MARKS)
    masters.build_masters()
    built = by_key(tables.read_table(SOURCE, "horses"), "horse_id")
    assert {k: (r["colour"], r["import_type"]) for k, r in built.items()} == meanings(
        COLOURS, MARKS
    )

    # the same records under a workbook whose tables say otherwise: the new meanings
    write_workbook(data / "specs" / "jravan" / wb.WORKBOOK_NAME, OTHER_COLOURS, OTHER_MARKS)
    frame = masters.horses_from_records(decoded("UM", *recs))
    got = {k: (r["colour"], r["import_type"]) for k, r in by_key(frame, "horse_id").items()}
    assert got == meanings(OTHER_COLOURS, OTHER_MARKS)
    masters.build_masters()
    built = by_key(tables.read_table(SOURCE, "horses"), "horse_id")
    assert {k: (r["colour"], r["import_type"]) for k, r in built.items()} == meanings(
        OTHER_COLOURS, OTHER_MARKS
    )


# ------------------------------------------------------------------------------------------------
# Dates


DATE_CASES = [  # (生年月日, 免許交付, 免許抹消) -> expected dates
    (("19900105", "20100301", "00000000"), (date(1990, 1, 5), date(2010, 3, 1), None)),
    (("00000000", "19700301", "19791230"), (None, date(1970, 3, 1), date(1979, 12, 30))),
    ((None, "20250101", None), (None, date(2025, 1, 1), None)),
    (("19551231", "00000000", "20251220"), (date(1955, 12, 31), None, date(2025, 12, 20))),
    (("20000229", None, "20240229"), (date(2000, 2, 29), None, date(2024, 2, 29))),
]


def test_blank_dates_are_null(data: Path) -> None:
    for build, record, rt, key in (
        (masters.jockeys_from_records, ks, "KS", "jockey_code"),
        (masters.trainers_from_records, ch, "CH", "trainer_code"),
    ):
        recs = [record(code=f"04{i:03d}", born=b, issued=s, revoked=r)
                for i, ((b, s, r), _) in enumerate(DATE_CASES)]  # fmt: skip
        got = by_key(build(decoded(rt, *recs)), key)
        for i, (_, want) in enumerate(DATE_CASES):
            row = got[f"04{i:03d}"]
            assert tuple(row[c] for c in DATES) == want, (rt, i)

    # through the build: date32 columns, null where blank
    root = data / "raw" / "jravan"
    write_master(root, "UM", [um()])
    write_master(root, "KS", [ks(code=f"04{i:03d}", born=b, issued=s, revoked=r)
                              for i, ((b, s, r), _) in enumerate(DATE_CASES)])  # fmt: skip
    write_master(root, "CH", [ch(code=f"04{i:03d}", born=b, issued=s, revoked=r)
                              for i, ((b, s, r), _) in enumerate(DATE_CASES)])  # fmt: skip
    masters.build_masters()
    for table in ("jockeys", "trainers"):
        arrow = pq.read_table(data / "processed" / "jravan" / f"{table}.parquet")
        for column in DATES:
            assert arrow.schema.field(column).type == pa.date32(), (table, column)
        got = [tuple(arrow[c][i].as_py() for c in DATES) for i in range(arrow.num_rows)]
        assert got == [want for _, want in DATE_CASES], table


# ------------------------------------------------------------------------------------------------
# Byte stability and the command


def test_two_builds_write_identical_bytes(data: Path) -> None:
    write_build_inputs(data / "raw" / "jravan")
    masters.build_masters()
    processed = data / "processed" / "jravan"
    names = ("horses", "horse_pedigree", "jockeys", "trainers")
    first = {t: (processed / f"{t}.parquet").read_bytes() for t in names}

    again = data / "again"
    masters.build_masters(out=again)
    for table, blob in first.items():
        assert (again / f"{table}.parquet").read_bytes() == blob, table

    masters.build_masters()  # over the existing files
    for table, blob in first.items():
        assert (processed / f"{table}.parquet").read_bytes() == blob, table

    result = CliRunner().invoke(app, ["japan", "jravan", "masters"])
    assert result.exit_code == 0, result.output
    for table, blob in first.items():
        assert (processed / f"{table}.parquet").read_bytes() == blob, table
    text = (paths.reports_dir() / "jravan-masters.json").read_text(encoding="utf-8")
    parsed = json.loads(text)
    assert parsed["name"] == "masters"
    for secret in ("カクウノウマ", "架空一", "架空師", "2021105001", "カクウチチ", "1110000001",
                   "架空色ウ", "(架外)"):  # fmt: skip
        assert secret not in text and secret not in result.output, secret  # counts only


# ------------------------------------------------------------------------------------------------
# The real export (local: needs $HORSETRACK_DATA_DIR with the export and the workbook)


@pytest.mark.local
def test_the_real_masters_map(tmp_path: Path) -> None:
    export, workbook = wb.real_export_or_skip()
    difn = export / "stored" / "DIFN"
    if not all((difn / rt / "all.jvd").is_file() for rt in ("UM", "KS", "CH")):
        pytest.skip("no UM, KS or CH all.jvd in the export")

    report = masters.build_masters(out=tmp_path)
    assert report.name == "masters"
    for table in ("horses", "horse_pedigree"):
        assert_store_types(tmp_path / f"{table}.parquet", table)
    assert_schema(tmp_path / "jockeys.parquet", masters.JOCKEYS_SCHEMA, JOCKEY_TYPES)
    assert_schema(tmp_path / "trainers.parquet", masters.TRAINERS_SCHEMA, TRAINER_TYPES)
    horses = pq.read_table(tmp_path / "horses.parquet").to_pandas()
    pedigree = pq.read_table(tmp_path / "horse_pedigree.parquet").to_pandas()
    jockeys = pq.read_table(tmp_path / "jockeys.parquet").to_pandas()
    trainers = pq.read_table(tmp_path / "trainers.parquet").to_pandas()

    # the keys expected, from the raw records with the existing decoder and version rule
    def kept_keys(rt: str, key: str) -> list[str]:
        layout = spec.record_spec(rt, workbook)
        parts = [
            pd.DataFrame({k: records.text(c[k]) for k in (key, "データ作成年月日", "データ区分")})
            for c in records.iter_file(difn / rt / "all.jvd", layout)
        ]
        return sorted(records.latest_versions(pd.concat(parts, ignore_index=True), [key])[key])

    assert list(horses["horse_id"]) == kept_keys("UM", "血統登録番号")
    assert list(pedigree["horse_id"]) == list(horses["horse_id"])
    assert list(jockeys["jockey_code"]) == kept_keys("KS", "騎手コード")
    assert list(trainers["trainer_code"]) == kept_keys("CH", "調教師コード")

    # the vocabulary, and meanings that are the workbook's own
    colours = set(spec.code_table("2203", workbook).values.values())
    marks = set(spec.code_table("2204", workbook).values.values())
    assert set(horses["sex"].dropna()) <= {"H", "M", "G"}
    assert set(horses["colour"].dropna()) <= colours
    assert set(horses["import_type"].dropna()) <= marks
    assert horses["colour"].notna().mean() > 0.99
    assert set(horses["jurisdiction"]) == {"jra"}
    for frame in (jockeys, trainers):
        assert set(frame["affiliation"].dropna()) <= AFFILIATIONS
        assert set(frame["jurisdiction"]) == {"jra"}
    names = pedigree["sire_name"].where(pedigree["sire_name"].notna(), None)
    assert [norm(n) for n in names] == [
        plain(v) for v in pedigree["sire_name_norm"].where(pedigree["sire_name_norm"].notna(), None)
    ]
    assert (horses["sire"].fillna("") == pedigree["sire_name"].fillna("")).all()

    # the repository holds no code-table text: no colour or mark meaning in the jravan sources
    sources = sorted((Path(masters.__file__).parent).glob("*.py")) + [Path(__file__)]
    meanings = {m for m in colours | marks if m and len(m) >= 2}
    for path in sources:
        text = path.read_text(encoding="utf-8")
        leaked = sorted(m for m in meanings if m in text)
        assert not leaked, f"{path.name} holds {len(leaked)} code-table meanings"

    # every runner's horse is a horse and its jockey and trainer keys join (the unset code
    # 00000 aside)
    runners_path = paths.processed_dir(SOURCE) / "runners.parquet"
    if not runners_path.is_file():
        pytest.skip("no jravan runners built yet: the joins are not checked")
    ran = pq.read_table(runners_path, columns=["horse_id", "jockey_key", "trainer_key"]).to_pandas()
    assert ran["horse_id"].isin(set(horses["horse_id"])).all()
    jockey_keys = ran["jockey_key"].dropna()
    jockey_keys = jockey_keys[jockey_keys != "jockey:00000"]
    assert jockey_keys.isin(set(jockeys["jockey_key"])).all()
    trainer_keys = ran["trainer_key"].dropna()
    trainer_keys = trainer_keys[trainer_keys != "trainer:00000"]
    assert trainer_keys.isin(set(trainers["trainer_key"])).all()
