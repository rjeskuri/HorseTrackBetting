"""JV-Data H1 (votes, Win to Trio) and H6 (votes, Trifecta) records into the jravan
`pool_votes` table (ticket #368).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jravan.votes`:
- `POOL_VOTES_SCHEMA: pyarrow.Schema`: `jurisdiction` string, `race_id` string, `race_date`
  date32, `venue` string, `race_no` int64, `pool` string, `combination` string, `votes` int64,
  `refund_votes` int64, `fav_rank` int64, `data_kubun` string, in that order.
- `votes_rows(record_type: str, cols: Mapping[str, numpy.ndarray]) -> pyarrow.Table`:
  `record_type` is `H1` or `H6`, `cols` is `records.decode` of JRA records of that type; the
  rows of those records, schema `POOL_VOTES_SCHEMA` (row order not pinned here).
- `build_pool_votes(*, years=None, raw=None, out=None, chunk_records: int = 500)
  -> records.BuildReport`: reads `<raw>/stored/RACE/H1/<YYYY>.jvd` and
  `<raw>/stored/RACE/H6/<YYYY>.jvd` (`raw` is the export root, default `paths.raw_dir('jravan')`;
  `years` None reads every year file of either folder; a year with no H6 file, every year before
  2004, reads H1 alone) with the layouts `spec.record_spec('H1')` and `spec.record_spec('H6')`,
  streamed `chunk_records` records at a time; keeps one version per race and record type and JRA
  venues only; writes `<out>/pool_votes.parquet` (`out` a folder, default
  `paths.processed_dir('jravan')`): schema `POOL_VOTES_SCHEMA`, one row group per race day
  (ascending, #502), rows sorted by `race_id, pool, combination` (plain string order),
  byte-stable. The report's name is
  `votes` and `counts['rows']` is the rows written; it holds counts only.
- `horsetrack japan jravan votes`: the same build, then `records.write_build_report`
  (`<reports dir>/jravan-votes.json`).
- `pool_votes` is not in `schemas.TABLES`. Its availability rule is `label_only` (#376, the
  owner's request of 2026-11-26: a label or a conditioner, never a feature at t): `read_asof` and
  `asof_join` raise `LabelOnlyTable` at any t, and `read_label` returns it whole.

The workbook items each column comes from (H1 `５．票数１`, 28,955 bytes, and H6
`６．票数6（3連単）`, 102,890 bytes, in the JV-Data 4.9.0.1 workbook on this Mac; names as
`spec.record_spec` returns them, keys as `records.decode` builds them):
- `jurisdiction`: `jra`. Only records whose `競馬場コード` is a JRA venue (01-10) are kept.
- `race_id`: `keys.race_id(開催年, 開催月日, 競馬場コード, レース番号)`.
- `race_date`: `開催年` + `開催月日`; `venue`: the short code of `競馬場コード` (`config/jra.yaml`,
  as in the race id); `race_no`: `レース番号` as an integer.
- one version per race and record type: `records.latest_versions` on `開催年`, `開催月日`,
  `競馬場コード`, `開催回[第N回]`, `開催日目[N日目]`, `レース番号` (`データ作成年月日`,
  `データ区分`; a race whose latest record of a type is a deletion, `データ区分` 0, has no rows
  of that type, the other type's rows stay).
- `data_kubun`: `データ区分` of the record kept (2 前日売最終, 4 確定, 5 確定(月曜), 9 レース中止).
- `pool` and `combination`: one row per registered slot of each votes group, the group and its
  combination item:
    H1 `単勝票数/馬番` WIN, `複勝票数/馬番` PLACE, `枠連票数/組番` BRACKET_QUINELLA,
    `馬連票数/組番` QUINELLA, `ワイド票数/組番` QUINELLA_PLACE, `馬単票数/組番` FORECAST,
    `3連複票数/組番` TRIO; H6 `3連単票数/組番` TIERCE.
  A `組番` holds the legs (1, 2 or 3) as equal-width numbers (the item's width / legs; a bracket
  is one digit, a horse two). `combination` joins them with `,`, leading zeros dropped (the form of
  `docs/specs/odds-log.md`): FORECAST and TIERCE in the printed (finishing) order, the others
  ascending by number (`1,1` is a same-bracket pair).
- `votes`: on a combination row `<group>/票数` (units of 100 yen, as printed); `refund_votes` null.
- `fav_rank`: `<group>/人気順` as an integer; `--`/`---`/`----` (cancelled before sale),
  `**`/`***`/`****` (cancelled after sale, refunded) and blank are null.
- `__pool__` row, one per sold pool: `combination` `__pool__`, `votes` the pool's `<P>票数合計`
  (refunded votes included), `refund_votes` its `<P>返還票数合計` (blank when nothing was
  refunded: null), `fav_rank` null. P is 単勝 複勝 枠連 馬連 ワイド 馬単 3連複 (H1), 3連単 (H6).
- Sold: `発売フラグ<P>` `7`. A pool flagged `0` (not sold: its slots and totals are blank) or `1`
  (cancelled before sale: its slots print zeros and `--`, its totals blank) writes no row at all.
  (`3`, cancelled after sale, never occurs in the export and is not pinned here.)
- A slot is a row when every leg of its combination is a positive integer, whatever its votes:
  a blank slot (a horse number past the field, an unregistered combination) or a zero or partly
  blank combination writes nothing. A registered slot of a sold pool with zero votes is a row
  (0 votes; its `人気順` is a number when sold, `--` when the horse was scratched before sale).
  A count (`票数`, `票数合計`, `返還票数合計`) or a rank that is not a number is null.
- Read, writing no row: `登録頭数`, `出走頭数`, `複勝着払キー`, `返還馬番情報(馬番01～28)`
  (H1), `返還馬番情報(馬番01～18)` (H6), `返還枠番情報(枠番1～8)`, `返還同枠情報(枠番1～8)`.

Every layout here is made up in the workbook's shape (`jravan_workbook.write_workbook`): the item
names are the workbook's, the widths of the counts and ranks, every repeat count and so every
position are invented, so nothing may be sliced by hand. Every record and value is invented too.
Expected rows are written out by hand.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from functools import cache
from pathlib import Path

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
    LabelOnlyTable,
    asof_join,
    load_availability,
    read_asof,
    read_label,
)

# isort: split
# (its own block: the sibling helper module, then the module this ticket creates)
import jravan_workbook as wb

from horsetrack.countries.japan.jravan import records, spec, votes

SOURCE = "jravan"
NAME = "pool_votes"
FILENAME = "pool_votes.parquet"
POOL_ROW = "__pool__"
UTC = timezone.utc
EXPECTED_SCHEMA = [
    ("jurisdiction", pa.string()),
    ("race_id", pa.string()),
    ("race_date", pa.date32()),
    ("venue", pa.string()),
    ("race_no", pa.int64()),
    ("pool", pa.string()),
    ("combination", pa.string()),
    ("votes", pa.int64()),
    ("refund_votes", pa.int64()),
    ("fav_rank", pa.int64()),
    ("data_kubun", pa.string()),
]
COLUMNS = [name for name, _ in EXPECTED_SCHEMA]
H1_CODES = ("WIN", "PLACE", "BRACKET_QUINELLA", "QUINELLA", "QUINELLA_PLACE", "FORECAST", "TRIO")
CODES = (*H1_CODES, "TIERCE")
VENUES = {  # number -> short code, as config/jra.yaml lists them
    "01": "SAP", "02": "HAK", "03": "FKS", "04": "NII", "05": "TOK",
    "06": "NAK", "07": "CKY", "08": "KYO", "09": "HSN", "10": "KOK",
}  # fmt: skip

# ------------------------------------------------------------------------------------------------
# Made-up H1 and H6 layouts: the workbook's item names, invented widths and repeats


VOTE = 9  # bytes of 票数 (invented)
TOTAL = 10  # bytes of 票数合計 and 返還票数合計 (invented)
H1_HORSES = 12  # repeat of H1's refunded-horse item (invented)
H6_HORSES = 14  # repeat of H6's refunded-horse item (invented)


@dataclass(frozen=True)
class PoolLayout:
    code: str
    jv: str  # the pool's name in the item names
    number: str  # the group's combination item
    digits: int  # digits of one leg
    legs: int
    repeat: int  # slots (invented)
    pop: int  # bytes of 人気順 (invented)

    @property
    def group(self) -> str:
        return f"{self.jv}票数"

    @property
    def width(self) -> int:
        return self.digits * self.legs


H1_POOLS = (
    PoolLayout("WIN", "単勝", "馬番", 2, 1, 6, 3),
    PoolLayout("PLACE", "複勝", "馬番", 2, 1, 6, 3),
    PoolLayout("BRACKET_QUINELLA", "枠連", "組番", 1, 2, 5, 2),
    PoolLayout("QUINELLA", "馬連", "組番", 2, 2, 6, 3),
    PoolLayout("QUINELLA_PLACE", "ワイド", "組番", 2, 2, 6, 4),
    PoolLayout("FORECAST", "馬単", "組番", 2, 2, 8, 3),
    PoolLayout("TRIO", "3連複", "組番", 2, 3, 6, 3),
)
H6_POOLS = (PoolLayout("TIERCE", "3連単", "組番", 2, 3, 10, 4),)
POOLS_OF = {"H1": H1_POOLS, "H6": H6_POOLS}

HEAD = (
    ("レコード種別ID", 2),
    ("データ区分", 1),
    ("データ作成年月日", 8),
    ("開催年", 4),
    ("開催月日", 4),
    ("競馬場コード", 2),
    ("開催回[第N回]", 2),
    ("開催日目[N日目]", 2),
    ("レース番号", 2),
    ("登録頭数", 2),
    ("出走頭数", 2),
)


@dataclass(frozen=True)
class Field:
    name: str
    length: int
    repeat: int = 1
    subs: tuple[tuple[str, int], ...] = ()

    @property
    def width(self) -> int:
        return sum(n for _, n in self.subs) if self.subs else self.length


@cache
def layout(rt: str) -> tuple[Field, ...]:
    pools = POOLS_OF[rt]
    out = [Field(name, length) for name, length in HEAD]
    out += [Field(f"発売フラグ{p.jv}", 1) for p in pools]
    if rt == "H1":
        out += [
            Field("複勝着払キー", 1),
            Field("予備", 3),
            Field("返還馬番情報(馬番01～28)", 1, H1_HORSES),
            Field("返還枠番情報(枠番1～8)", 1, 8),
            Field("返還同枠情報(枠番1～8)", 1, 8),
        ]
    else:
        out.append(Field("返還馬番情報(馬番01～18)", 1, H6_HORSES))
    for p in pools:
        subs = ((p.number, p.width), ("票数", VOTE), ("人気順", p.pop))
        out.append(Field(p.group, 0, p.repeat, subs))
    if rt == "H1":
        out.append(Field("予備", 5))
    out += [Field(f"{p.jv}票数合計", TOTAL) for p in pools]
    out += [Field(f"{p.jv}返還票数合計", TOTAL) for p in pools]
    out.append(Field("レコード区切", 2))
    return tuple(out)


@cache
def section(rt: str) -> wb.Section:
    number = {"H1": 5, "H6": 6}[rt]
    rows: list[wb.Row] = []
    pos = 1
    for no, f in enumerate(layout(rt), start=1):
        if f.subs:
            rows.append(wb.group(no, f"<{f.name}>", pos, f.width, f.repeat, desc="票数（架空）"))
            rel = 1
            for i, (name, length) in enumerate(f.subs):
                rows.append(wb.sub(wb.letter_code(i), name, rel, length, desc="値（架空）"))
                rel += length
            pos += f.width * f.repeat
            continue
        if no == 1:
            rows.append(wb.item(1, f.name, 1, f.length, key="●", initial=rt, desc=wb.id_desc(rt)))
        else:
            repeat = f.repeat if f.repeat > 1 else None
            rows.append(wb.item(no, f.name, pos, f.length, repeat=repeat, desc="（架空）"))
        pos += f.length * f.repeat
    return wb.Section(number, f"票数{rt}（架空）", pos - 1, tuple(rows))


def write_votes_workbook(path: Path) -> Path:
    decoy = wb.simple_section(1, "XQ", "架空その他", [("架空項目", 6)])
    return wb.write_workbook(path, [decoy, section("H1"), section("H6")], [])


Slot = tuple[str, "int | str | None", "int | str | None"]  # (組番 as printed, 票数, 人気順)


@dataclass(frozen=True)
class Sale:
    """One pool of a record: its slots in record order, its totals, its sale flag."""

    slots: Sequence[Slot] = ()
    total: int | str | None = None
    refund: int | str | None = None
    flag: str = "7"


def count(value: int | str | None, width: int) -> str:
    """A count as printed: an int zero-padded, a str as is, None blank."""
    if value is None:
        return ""
    if isinstance(value, int):
        return f"{value:0{width}d}"
    return value


def rank(value: int | str | None, width: int) -> str:
    """人気順 as printed: an int zero-padded, `-` or `*` repeated to the width, None blank."""
    if value is None:
        return ""
    if isinstance(value, int):
        return f"{value:0{width}d}"
    if value in ("-", "*"):
        return value * width
    return value


def fit(value: str, length: int) -> str:
    """A field as printed: `value` at the field's exact width, or blank."""
    if value and len(value) != length:
        raise ValueError(f"{value!r} is not {length} wide")
    return value or " " * length


@dataclass
class Rec:
    """One made-up H1 or H6 record; a pool not in `sales` is unsold (flag 0, all blank)."""

    rt: str = "H1"
    year: str = "2024"
    mmdd: str = "0601"
    venue: str = "05"
    race: str = "11"
    kai: str = "03"
    nichi: str = "02"
    kubun: str = "5"
    created: str = "20240603"
    sales: Mapping[str, Sale] = field(default_factory=dict)

    def record(self) -> bytes:
        pools = POOLS_OF[self.rt]
        unknown = set(self.sales) - {p.code for p in pools}
        assert not unknown, (self.rt, unknown)
        flat = dict(
            zip(
                [name for name, _ in HEAD],
                (self.rt, self.kubun, self.created, self.year, self.mmdd, self.venue, self.kai,
                 self.nichi, self.race, "16", "15"),
                strict=True,
            )
        )  # fmt: skip
        flat["複勝着払キー"] = "3"
        for p in pools:
            sale = self.sales.get(p.code)
            flat[f"発売フラグ{p.jv}"] = sale.flag if sale else "0"
            flat[f"{p.jv}票数合計"] = count(sale.total, TOTAL) if sale else ""
            flat[f"{p.jv}返還票数合計"] = count(sale.refund, TOTAL) if sale else ""
        by_group = {p.group: p for p in pools}
        out: list[str] = []
        for f in layout(self.rt)[:-1]:
            if f.subs:
                p = by_group[f.name]
                sale = self.sales.get(p.code)
                slots = list(sale.slots) if sale else []
                assert len(slots) <= f.repeat, (f.name, slots)
                for r in range(f.repeat):
                    combo, n, pop = slots[r] if r < len(slots) else ("", None, None)
                    out += [
                        fit(combo, p.width),
                        fit(count(n, VOTE), VOTE),
                        fit(rank(pop, p.pop), p.pop),
                    ]
            elif f.name == "予備":
                out.append(" " * f.length)
            elif f.repeat > 1:
                out.append("0" * f.repeat)
            else:
                out.append(fit(flat[f.name], f.length))
        body = "".join(out).encode("cp932")
        assert len(body) == section(self.rt).length - 2
        return body + b"\r\n"


def h6(**kwargs) -> Rec:
    return Rec(rt="H6", **kwargs)


# ------------------------------------------------------------------------------------------------
# Helpers


Row = tuple[str, str, str, "int | None", "int | None", "int | None"]


def c(race_id: str, pool: str, combination: str, n: int | None, fav: int | None) -> Row:
    """An expected combination row."""
    return (race_id, pool, combination, n, None, fav)


def p(race_id: str, pool: str, total: int | None, refund: int | None = None) -> Row:
    """An expected `__pool__` row."""
    return (race_id, pool, POOL_ROW, total, refund, None)


def key(row: Row) -> tuple[str, str, str]:
    return row[0], row[1], row[2]


def check_schema(schema: pa.Schema) -> None:
    assert [(f.name, f.type) for f in schema] == EXPECTED_SCHEMA


def rows_of(table: pa.Table) -> list[Row]:
    """(race_id, pool, combination, votes, refund_votes, fav_rank) per row, in table order, after
    checking the columns derived from the race id."""
    assert table.schema.names == COLUMNS
    out = []
    for r in table.to_pylist():
        jurisdiction, day, venue, race_no = r["race_id"].split(":")
        assert r["jurisdiction"] == "jra" == jurisdiction, r
        assert r["race_date"] == date.fromisoformat(day), r
        assert r["venue"] == venue, r
        assert r["race_no"] == int(race_no), r
        out.append(
            (r["race_id"], r["pool"], r["combination"], r["votes"], r["refund_votes"],
             r["fav_rank"])
        )  # fmt: skip
    return out


def kubun_of(table: pa.Table) -> dict[str, set[str]]:
    """`data_kubun` values per (race_id + record type) as `race_id/H1` or `race_id/H6`."""
    out: dict[str, set[str]] = {}
    for r in table.to_pylist():
        rt = "H6" if r["pool"] == "TIERCE" else "H1"
        out.setdefault(f"{r['race_id']}/{rt}", set()).add(r["data_kubun"])
    return out


@pytest.fixture(scope="module")
def layouts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, spec.RecordSpec]:
    path = write_votes_workbook(tmp_path_factory.mktemp("workbook") / wb.WORKBOOK_NAME)
    return {rt: spec.record_spec(rt, path) for rt in ("H1", "H6")}


def from_records(layouts: Mapping[str, spec.RecordSpec], rt: str, recs: Sequence[Rec]) -> pa.Table:
    assert {r.rt for r in recs} <= {rt}
    cols = records.decode(b"".join(r.record() for r in recs), layouts[rt])
    table = votes.votes_rows(rt, cols)
    assert isinstance(table, pa.Table)
    check_schema(table.schema)
    assert table.schema == votes.POOL_VOTES_SCHEMA
    return table


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data dir with the made-up workbook and an empty export."""
    root = tmp_path / "data"
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(root))
    write_votes_workbook(root / "specs" / "jravan" / wb.WORKBOOK_NAME)
    for rt in ("H1", "H6"):
        (root / "raw" / SOURCE / "stored" / "RACE" / rt).mkdir(parents=True)
    return root


def write_year(data: Path, rt: str, year: int, recs: Sequence[Rec]) -> Path:
    assert {r.rt for r in recs} <= {rt}
    path = data / "raw" / SOURCE / "stored" / "RACE" / rt / f"{year}.jvd"
    return wb.write_jvd(path, [r.record() for r in recs])


def store_path(data: Path) -> Path:
    return data.resolve() / "processed" / SOURCE / FILENAME


def store_rows(data: Path) -> list[Row]:
    return rows_of(pq.read_table(store_path(data)))


# ------------------------------------------------------------------------------------------------
# votes_rows


def test_win_to_trio_votes_become_one_row_per_combination(
    layouts: Mapping[str, spec.RecordSpec],
) -> None:
    check_schema(votes.POOL_VOTES_SCHEMA)
    a = Rec(
        year="2024", mmdd="0601", venue="05", race="11",
        sales={
            "WIN": Sale([("01", 30391, 1), ("02", 28655, 2), ("03", 784, 11)], total=59830),
            "PLACE": Sale([("01", 12000, 1), ("02", 9100, 2), ("03", 450, 3)], total=21550),
            "BRACKET_QUINELLA": Sale([("11", 120, 3), ("12", 5400, 1), ("23", 880, 2)],
                                     total=6400),
            "QUINELLA": Sale([("0102", 15320, 1), ("0103", 2210, 2), ("0203", 1990, 3)],
                             total=19520),
            "QUINELLA_PLACE": Sale([("0102", 8800, 1), ("0103", 3100, 2), ("0203", 2900, 3)],
                                   total=14800),
            "FORECAST": Sale([("0102", 7000, 1), ("0103", 900, 4), ("0201", 6100, 2),
                              ("0203", 1200, 3), ("0301", 450, 6), ("0302", 520, 5)],
                             total=16170),
            "TRIO": Sale([("010203", 4321, 1)], total=4321),
        },
    )  # fmt: skip
    # Printed out of numeric order: the unordered pools sort their legs, Exacta keeps them.
    b = Rec(
        year="2019", mmdd="1228", venue="06", race="09", kai="05", nichi="09", kubun="4",
        created="20191228",
        sales={
            "WIN": Sale([("16", 125, 9), ("09", 33000, 1)], total=33125),
            "PLACE": Sale([("16", 4000, 3), ("09", 11800, 1)], total=15800),
            "BRACKET_QUINELLA": Sale([("77", 640, 2), ("53", 2100, 1)], total=2740),
            "QUINELLA": Sale([("1203", 1500, 2), ("0916", 9800, 1)], total=11300),
            "QUINELLA_PLACE": Sale([("1810", 777, 1)], total=777),
            "FORECAST": Sale([("1203", 610, 2), ("0912", 3300, 1)], total=3910),
            "TRIO": Sale([("120309", 250, 2), ("011805", 1250, 1)], total=1500),
        },
    )  # fmt: skip
    ra, rb = "jra:2024-06-01:TOK:11", "jra:2019-12-28:NAK:09"
    expected = [
        c(ra, "WIN", "1", 30391, 1), c(ra, "WIN", "2", 28655, 2), c(ra, "WIN", "3", 784, 11),
        p(ra, "WIN", 59830),
        c(ra, "PLACE", "1", 12000, 1), c(ra, "PLACE", "2", 9100, 2), c(ra, "PLACE", "3", 450, 3),
        p(ra, "PLACE", 21550),
        c(ra, "BRACKET_QUINELLA", "1,1", 120, 3), c(ra, "BRACKET_QUINELLA", "1,2", 5400, 1),
        c(ra, "BRACKET_QUINELLA", "2,3", 880, 2), p(ra, "BRACKET_QUINELLA", 6400),
        c(ra, "QUINELLA", "1,2", 15320, 1), c(ra, "QUINELLA", "1,3", 2210, 2),
        c(ra, "QUINELLA", "2,3", 1990, 3), p(ra, "QUINELLA", 19520),
        c(ra, "QUINELLA_PLACE", "1,2", 8800, 1), c(ra, "QUINELLA_PLACE", "1,3", 3100, 2),
        c(ra, "QUINELLA_PLACE", "2,3", 2900, 3), p(ra, "QUINELLA_PLACE", 14800),
        c(ra, "FORECAST", "1,2", 7000, 1), c(ra, "FORECAST", "1,3", 900, 4),
        c(ra, "FORECAST", "2,1", 6100, 2), c(ra, "FORECAST", "2,3", 1200, 3),
        c(ra, "FORECAST", "3,1", 450, 6), c(ra, "FORECAST", "3,2", 520, 5),
        p(ra, "FORECAST", 16170),
        c(ra, "TRIO", "1,2,3", 4321, 1), p(ra, "TRIO", 4321),
        c(rb, "WIN", "16", 125, 9), c(rb, "WIN", "9", 33000, 1), p(rb, "WIN", 33125),
        c(rb, "PLACE", "16", 4000, 3), c(rb, "PLACE", "9", 11800, 1), p(rb, "PLACE", 15800),
        c(rb, "BRACKET_QUINELLA", "7,7", 640, 2), c(rb, "BRACKET_QUINELLA", "3,5", 2100, 1),
        p(rb, "BRACKET_QUINELLA", 2740),
        c(rb, "QUINELLA", "3,12", 1500, 2), c(rb, "QUINELLA", "9,16", 9800, 1),
        p(rb, "QUINELLA", 11300),
        c(rb, "QUINELLA_PLACE", "10,18", 777, 1), p(rb, "QUINELLA_PLACE", 777),
        c(rb, "FORECAST", "12,3", 610, 2), c(rb, "FORECAST", "9,12", 3300, 1),
        p(rb, "FORECAST", 3910),
        c(rb, "TRIO", "3,9,12", 250, 2), c(rb, "TRIO", "1,5,18", 1250, 1), p(rb, "TRIO", 1500),
    ]  # fmt: skip
    # Every JRA venue, each race with another horse, count and rank.
    recs = [a, b]
    kubun = {f"{ra}/H1": {"5"}, f"{rb}/H1": {"4"}}
    for i, (number, code) in enumerate(VENUES.items()):
        n = 100 + 70 * i
        recs.append(
            Rec(year="2010", mmdd="0815", venue=number, race=f"{i + 1:02d}", created="20100816",
                kubun="5" if i % 2 else "2",
                sales={"WIN": Sale([(f"{i + 3:02d}", n, i + 1)], total=n + i)})
        )  # fmt: skip
        rid = f"jra:2010-08-15:{code}:{i + 1:02d}"
        expected += [c(rid, "WIN", str(i + 3), n, i + 1), p(rid, "WIN", n + i)]
        kubun[f"{rid}/H1"] = {"5" if i % 2 else "2"}

    table = from_records(layouts, "H1", recs)
    got = rows_of(table)
    assert sorted(got, key=key) == sorted(expected, key=key)
    assert {row[1] for row in got} == set(H1_CODES)
    assert kubun_of(table) == kubun


TIERCE_CASES = [  # (組番 as printed, combination)
    ("120309", "12,3,9"),
    ("100201", "10,2,1"),
    ("011805", "1,18,5"),
    ("180317", "18,3,17"),
    ("030201", "3,2,1"),
    ("010203", "1,2,3"),
    ("091011", "9,10,11"),
    ("111009", "11,10,9"),
]


def test_trifecta_votes_keep_finishing_order(layouts: Mapping[str, spec.RecordSpec]) -> None:
    rc = "jra:2024-06-02:KYO:12"
    slots = [(printed, 40 * i + 7, i + 1) for i, (printed, _) in enumerate(TIERCE_CASES)]
    # A sold combination nobody bought: 0 votes, still ranked.
    slots.append(("170102", 0, 4002))
    total = sum(n for _, n, _ in slots)
    expected = [
        c(rc, "TIERCE", combination, 40 * i + 7, i + 1)
        for i, (_, combination) in enumerate(TIERCE_CASES)
    ]
    expected += [c(rc, "TIERCE", "17,1,2", 0, 4002), p(rc, "TIERCE", total)]
    other = h6(
        year="2005", mmdd="1127", venue="05", race="10", created="20051128",
        sales={"TIERCE": Sale([("020103", 999, 2), ("010203", 1000, 1)], total=1999)},
    )  # fmt: skip
    rd = "jra:2005-11-27:TOK:10"
    expected += [
        c(rd, "TIERCE", "2,1,3", 999, 2), c(rd, "TIERCE", "1,2,3", 1000, 1),
        p(rd, "TIERCE", 1999),
    ]  # fmt: skip

    recs = [
        h6(mmdd="0602", venue="08", race="12", sales={"TIERCE": Sale(slots, total=total)}),
        other,
    ]
    table = from_records(layouts, "H6", recs)
    got = rows_of(table)
    assert sorted(got, key=key) == sorted(expected, key=key)
    assert {row[1] for row in got} == {"TIERCE"}
    assert kubun_of(table) == {f"{rc}/H6": {"5"}, f"{rd}/H6": {"5"}}


def test_pool_totals_and_refunds_are_pool_rows(layouts: Mapping[str, spec.RecordSpec]) -> None:
    # Horse 5 scratched after sales opened: its combinations are refunded ('**'), their votes
    # stay in the total. The pool row carries the item's total, not a sum of the slots (TRIO).
    e = Rec(
        year="2023", mmdd="0415", venue="09", race="11", created="20230417",
        sales={
            "WIN": Sale([("01", 5000, 1), ("05", 420, "*"), ("07", 2600, 2)], total=8020,
                        refund=420),
            "PLACE": Sale([("01", 3000, 1), ("05", 0, "*"), ("07", 1200, 2)], total=4200,
                          refund=0),
            "BRACKET_QUINELLA": Sale([("11", 100, 2), ("14", 900, 1)], total=1000),
            "QUINELLA": Sale([("0105", 300, "*"), ("0107", 4100, 1), ("0507", 150, "*")],
                             total=4550, refund=450),
            "TRIO": Sale([("010507", 20, "*")], total=123456, refund=20),
        },
    )  # fmt: skip
    # Horse 3 scratched before sales: its slots print zeros and '--'.
    g = Rec(
        year="2024", mmdd="1124", venue="05", race="12", created="20241126",
        sales={"WIN": Sale([("03", 0, "-"), ("04", 700, 1)], total=700),
               "QUINELLA_PLACE": Sale([("0304", 0, "-"), ("0408", 61, 1)], total=61, refund=7)},
    )  # fmt: skip
    re_, rg = "jra:2023-04-15:HSN:11", "jra:2024-11-24:TOK:12"
    expected_h1 = [
        c(re_, "WIN", "1", 5000, 1), c(re_, "WIN", "5", 420, None), c(re_, "WIN", "7", 2600, 2),
        p(re_, "WIN", 8020, 420),
        c(re_, "PLACE", "1", 3000, 1), c(re_, "PLACE", "5", 0, None),
        c(re_, "PLACE", "7", 1200, 2), p(re_, "PLACE", 4200, 0),
        c(re_, "BRACKET_QUINELLA", "1,1", 100, 2), c(re_, "BRACKET_QUINELLA", "1,4", 900, 1),
        p(re_, "BRACKET_QUINELLA", 1000, None),
        c(re_, "QUINELLA", "1,5", 300, None), c(re_, "QUINELLA", "1,7", 4100, 1),
        c(re_, "QUINELLA", "5,7", 150, None), p(re_, "QUINELLA", 4550, 450),
        c(re_, "TRIO", "1,5,7", 20, None), p(re_, "TRIO", 123456, 20),
        c(rg, "WIN", "3", 0, None), c(rg, "WIN", "4", 700, 1), p(rg, "WIN", 700, None),
        c(rg, "QUINELLA_PLACE", "3,4", 0, None), c(rg, "QUINELLA_PLACE", "4,8", 61, 1),
        p(rg, "QUINELLA_PLACE", 61, 7),
    ]  # fmt: skip
    got = rows_of(from_records(layouts, "H1", [e, g]))
    assert sorted(got, key=key) == sorted(expected_h1, key=key)

    f = h6(
        year="2023", mmdd="0415", venue="09", race="11", created="20230417",
        sales={"TIERCE": Sale([("010705", 30, "*"), ("070105", 12, "*"), ("010702", 900, 1)],
                              total=942, refund=42)},
    )  # fmt: skip
    expected_h6 = [
        c(re_, "TIERCE", "1,7,5", 30, None), c(re_, "TIERCE", "7,1,5", 12, None),
        c(re_, "TIERCE", "1,7,2", 900, 1), p(re_, "TIERCE", 942, 42),
    ]  # fmt: skip
    got = rows_of(from_records(layouts, "H6", [f]))
    assert sorted(got, key=key) == sorted(expected_h6, key=key)

    # Exactly one pool row per sold pool; refunds only there, ranks only on combinations.
    for rows in (expected_h1, expected_h6):
        pools = [(r[0], r[1]) for r in rows if r[2] == POOL_ROW]
        assert len(pools) == len(set(pools))
        assert all(r[4] is None for r in rows if r[2] != POOL_ROW)
        assert all(r[5] is None for r in rows if r[2] == POOL_ROW)


def test_unsold_pools_and_blank_slots_write_no_rows(
    layouts: Mapping[str, spec.RecordSpec],
) -> None:
    # A year before Quinella, Wide, Exacta and Trio: those pools are flagged 0, all blank; a
    # blank slot between two horses.
    h = Rec(
        year="1990", mmdd="0506", venue="05", race="10", created="19900507",
        sales={
            "WIN": Sale([("05", 420, 1), ("", None, None), ("09", 380, 2)], total=800),
            "PLACE": Sale([("05", 160, 1), ("09", 220, 2), ("02", 180, 3)], total=560),
            "BRACKET_QUINELLA": Sale([("35", 990, 1)], total=990),
        },
    )  # fmt: skip
    # Place cancelled before sale (flag 1): registered slots of zeros and '--', blank totals.
    i = Rec(
        year="2004", mmdd="0717", venue="08", race="03", created="20040719",
        sales={
            "WIN": Sale([("01", 500, 1), ("02", 300, 2)], total=800),
            "PLACE": Sale([("01", 0, "-"), ("02", 0, "-")], flag="1"),
        },
    )  # fmt: skip
    # Counts that are not numbers are null; combinations that are zero or partly blank are no row.
    j = Rec(
        year="2015", mmdd="0301", venue="07", race="05", created="20150302",
        sales={
            "WIN": Sale([("03", "12A456789", 1), ("04", 50, "x1 "), ("06", "         ", 3)],
                        total="garbled!!!", refund="  12 34   "),
            "QUINELLA": Sale([("0000", 10, 1), ("07  ", 20, 2), ("0708", 30, 3), ("    ", 40, 4)],
                             total=100),
        },
    )  # fmt: skip
    # A race with nothing sold.
    nothing = Rec(year="2015", mmdd="0301", venue="07", race="06", created="20150302")
    rh, ri, rj = "jra:1990-05-06:TOK:10", "jra:2004-07-17:KYO:03", "jra:2015-03-01:CKY:05"
    expected = [
        c(rh, "WIN", "5", 420, 1), c(rh, "WIN", "9", 380, 2), p(rh, "WIN", 800),
        c(rh, "PLACE", "5", 160, 1), c(rh, "PLACE", "9", 220, 2), c(rh, "PLACE", "2", 180, 3),
        p(rh, "PLACE", 560),
        c(rh, "BRACKET_QUINELLA", "3,5", 990, 1), p(rh, "BRACKET_QUINELLA", 990),
        c(ri, "WIN", "1", 500, 1), c(ri, "WIN", "2", 300, 2), p(ri, "WIN", 800),
        c(rj, "WIN", "3", None, 1), c(rj, "WIN", "4", 50, None), c(rj, "WIN", "6", None, 3),
        p(rj, "WIN", None, None),
        c(rj, "QUINELLA", "7,8", 30, 3), p(rj, "QUINELLA", 100),
    ]  # fmt: skip
    got = rows_of(from_records(layouts, "H1", [h, i, j, nothing]))
    assert sorted(got, key=key) == sorted(expected, key=key)

    # H6: Trifecta unsold (flag 0) or cancelled before sale (flag 1) writes nothing; blank slots
    # around a sold combination are no rows.
    unsold = h6(year="2010", mmdd="0105", venue="06", race="01", created="20100106")
    cancelled = h6(year="2010", mmdd="0105", venue="06", race="02", created="20100106",
                   sales={"TIERCE": Sale([("010203", 0, "-")], flag="1")})  # fmt: skip
    sparse = h6(year="2010", mmdd="0105", venue="06", race="03", created="20100106",
                sales={"TIERCE": Sale([("", None, None), ("020301", 77, 1), ("", None, None)],
                                      total=77)})  # fmt: skip
    rs = "jra:2010-01-05:NAK:03"
    got = rows_of(from_records(layouts, "H6", [unsold, cancelled, sparse]))
    assert sorted(got, key=key) == sorted(
        [c(rs, "TIERCE", "2,3,1", 77, 1), p(rs, "TIERCE", 77)], key=key
    )

    # Only unsold records: an empty table with the schema.
    for rt, recs in (("H1", [nothing]), ("H6", [unsold, cancelled])):
        empty = from_records(layouts, rt, recs)
        assert empty.num_rows == 0


# ------------------------------------------------------------------------------------------------
# build_pool_votes


def build_races() -> dict[tuple[str, int], list[Rec]]:
    """H1 for 2023 and 2024, H6 for 2024 only; the races out of race_id order; a NAR race."""
    return {
        ("H1", 2024): [
            Rec(year="2024", mmdd="0602", venue="05", race="11", created="20240604",
                sales={"WIN": Sale([("04", 350, 1), ("12", 90, 2)], total=440),
                       "PLACE": Sale([("04", 140, 1), ("12", 310, 2)], total=450)}),
            Rec(year="2024", mmdd="0602", venue="08", race="02", created="20240604",
                sales={"QUINELLA_PLACE": Sale([("0612", 990, 1), ("0206", 15, 2)], total=1005)}),
            Rec(year="2024", mmdd="0602", venue="30", race="05", created="20240604",
                sales={"WIN": Sale([("03", 500, 1)], total=500)}),
            Rec(year="2024", mmdd="0601", venue="05", race="01", created="20240603",
                sales={"WIN": Sale([("01", 180, 1)], total=180),
                       "TRIO": Sale([("010203", 1500, 1)], total=1500, refund=3)}),
        ],
        ("H6", 2024): [
            h6(year="2024", mmdd="0602", venue="05", race="11", created="20240604",
               sales={"TIERCE": Sale([("041207", 35, 1), ("120407", 12, 2)], total=47)}),
            h6(year="2024", mmdd="0602", venue="30", race="05", created="20240604",
               sales={"TIERCE": Sale([("030102", 5, 1)], total=5)}),
            h6(year="2024", mmdd="0601", venue="05", race="01", created="20240603",
               sales={"TIERCE": Sale([("010203", 66, 1)], total=66)}),
        ],
        ("H1", 2023): [
            Rec(year="2023", mmdd="1228", venue="06", race="11", created="20231229",
                sales={"WIN": Sale([("05", 670, 1)], total=670),
                       "FORECAST": Sale([("0905", 4530, 1), ("0509", 210, 2)], total=4740)}),
            Rec(year="2023", mmdd="0107", venue="06", race="01", created="20230108",
                sales={"WIN": Sale([("02", 33, 1)], total=33)}),
        ],
    }  # fmt: skip


R23A, R23B = "jra:2023-01-07:NAK:01", "jra:2023-12-28:NAK:11"
R24A, R24B, R24C = "jra:2024-06-01:TOK:01", "jra:2024-06-02:KYO:02", "jra:2024-06-02:TOK:11"
ROWS_2023 = [  # sorted by race_id, pool, combination: plain string order ('9,5' after '5,9')
    c(R23A, "WIN", "2", 33, 1), p(R23A, "WIN", 33),
    c(R23B, "FORECAST", "5,9", 210, 2), c(R23B, "FORECAST", "9,5", 4530, 1),
    p(R23B, "FORECAST", 4740),
    c(R23B, "WIN", "5", 670, 1), p(R23B, "WIN", 670),
]  # fmt: skip
ROWS_2024 = [  # '12' before '4', 'TIERCE' before 'TRIO' before 'WIN'
    c(R24A, "TIERCE", "1,2,3", 66, 1), p(R24A, "TIERCE", 66),
    c(R24A, "TRIO", "1,2,3", 1500, 1), p(R24A, "TRIO", 1500, 3),
    c(R24A, "WIN", "1", 180, 1), p(R24A, "WIN", 180),
    c(R24B, "QUINELLA_PLACE", "2,6", 15, 2), c(R24B, "QUINELLA_PLACE", "6,12", 990, 1),
    p(R24B, "QUINELLA_PLACE", 1005),
    c(R24C, "PLACE", "12", 310, 2), c(R24C, "PLACE", "4", 140, 1), p(R24C, "PLACE", 450),
    c(R24C, "TIERCE", "12,4,7", 12, 2), c(R24C, "TIERCE", "4,12,7", 35, 1),
    p(R24C, "TIERCE", 47),
    c(R24C, "WIN", "12", 90, 2), c(R24C, "WIN", "4", 350, 1), p(R24C, "WIN", 440),
]  # fmt: skip


def write_build_races(data: Path) -> None:
    for (rt, year), recs in build_races().items():
        write_year(data, rt, year, recs)


def test_build_writes_pool_votes_year_by_year_in_race_order(data: Path, tmp_path: Path) -> None:
    write_build_races(data)
    report = votes.build_pool_votes()
    assert isinstance(report, records.BuildReport)
    assert report.name == "votes"
    path = store_path(data)
    assert path.is_file()
    assert pq.read_schema(path) == votes.POOL_VOTES_SCHEMA
    check_schema(pq.read_schema(path))
    # Every JRA race's votes in race_id, pool, combination order; no NAR row.
    assert store_rows(data) == ROWS_2023 + ROWS_2024
    assert report.counts["rows"] == len(ROWS_2023) + len(ROWS_2024)
    assert all(isinstance(v, int) for v in report.counts.values())
    # One row group per race day, in day order (#502): 2023-01-07, 2023-12-28, 2024-06-01 and
    # 2024-06-02 (two races).
    meta = pq.ParquetFile(path)
    rows = ROWS_2023 + ROWS_2024
    days = sorted({r[0].split(":")[1] for r in rows})
    assert meta.metadata.num_row_groups == len(days) == 4
    for group, day in enumerate(days):
        ids = meta.read_row_group(group, columns=["race_id"]).column("race_id").to_pylist()
        assert ids == [r[0] for r in rows if r[0].split(":")[1] == day], day

    # One year only, a year with no H6 file.
    votes.build_pool_votes(years=[2023])
    assert store_rows(data) == ROWS_2023

    # Another export and another output folder; the store's file is left alone.
    other = tmp_path / "elsewhere" / "jravan"
    shutil.copytree(data / "raw" / SOURCE, other)
    out = tmp_path / "out"
    report = votes.build_pool_votes(years=[2024], raw=other, out=out)
    assert report.counts["rows"] == len(ROWS_2024)
    assert rows_of(pq.read_table(out / FILENAME)) == ROWS_2024
    assert store_rows(data) == ROWS_2023

    # The command builds every year and writes the report, counts only.
    result = CliRunner().invoke(app, ["japan", "jravan", "votes"])
    assert result.exit_code == 0, result.output
    assert store_rows(data) == ROWS_2023 + ROWS_2024
    written = paths.reports_dir() / "jravan-votes.json"
    assert written.is_file()
    parsed = json.loads(written.read_text(encoding="utf-8"))
    assert parsed["name"] == "votes"
    assert parsed["counts"]["rows"] == len(ROWS_2023) + len(ROWS_2024)


def plain(v: object) -> object:
    """A frame cell as plain Python: None for any null, a naive timestamp (a DuckDB DATE) as a
    date."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):
        return v.date() if v.tzinfo is None else v.tz_convert("UTC").to_pydatetime()
    if isinstance(v, np.generic):
        return plain(v.item())
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def test_pool_votes_is_label_only_so_views_raise_and_read_label_returns_it_whole(
    data: Path,
) -> None:
    write_build_races(data)
    votes.build_pool_votes()
    path = store_path(data)
    assert path.is_file()

    assert NAME not in schemas.TABLES
    rules = load_availability("japan")
    assert rules[NAME].kind == "label_only"
    with pytest.raises(ValueError, match="not a store table"):
        tables.read_table(SOURCE, NAME)

    con = tables.connect(SOURCE)
    query = "SELECT table_name FROM information_schema.tables"
    listed = {row[0] for row in con.execute(query).fetchall()}
    assert NAME not in listed
    con.execute(f"CREATE VIEW {NAME} AS SELECT * FROM read_parquet('{path}')")
    assert con.execute(f"SELECT count(*) FROM {NAME}").fetchone()[0] == len(ROWS_2023 + ROWS_2024)
    # Never in a view: before the data, between the races, after them, long after.
    times = [
        datetime(2000, 1, 1, tzinfo=UTC),
        datetime(2023, 12, 28, 6, 0, tzinfo=UTC),
        datetime(2024, 6, 1, 15, 0, tzinfo=UTC),
        datetime(2024, 6, 30, tzinfo=UTC),
        datetime(2030, 1, 1, tzinfo=UTC),
    ]
    keys = pd.DataFrame(
        {
            "race_id": [R24C, R23B, R24A, R24B, R23A],
            "pool": ["WIN", "FORECAST", "TIERCE", "QUINELLA_PLACE", "WIN"],
            "combination": ["4", "9,5", POOL_ROW, "2,6", "2"],
            "t_utc": pd.to_datetime(times, utc=True),
        }
    )
    for t in times:
        with pytest.raises(LabelOnlyTable):
            read_asof(con, NAME, t, rules=rules)
    with pytest.raises(LabelOnlyTable):
        asof_join(keys, NAME, on=["race_id", "pool", "combination"], rules=rules, con=con)
    with pytest.raises(LabelOnlyTable):
        asof_join(keys[["race_id", "t_utc"]], NAME, on=["race_id"], rules=rules, con=con)

    # As a label the table is whole: every row and column, the votes included.
    labels = read_label(con, NAME, rules=rules)
    assert list(labels.columns) == COLUMNS
    whole = [{c: plain(v) for c, v in r.items()} for r in labels.to_dict("records")]
    assert sorted(whole, key=lambda r: (r["race_id"], r["pool"], r["combination"])) == [
        {"jurisdiction": "jra", "race_id": rid, "race_date": date.fromisoformat(rid.split(":")[1]),
         "venue": rid.split(":")[2], "race_no": int(rid.split(":")[3]), "pool": pool,
         "combination": combination, "votes": n, "refund_votes": refund, "fav_rank": fav,
         "data_kubun": "5"}  # build_races keeps Rec's default データ区分
        for rid, pool, combination, n, refund, fav in ROWS_2023 + ROWS_2024
    ]  # fmt: skip


def test_the_latest_version_per_race_is_kept(data: Path) -> None:
    def kyo(rt: str, race: str, kubun: str, created: str, sales: dict) -> Rec:
        return Rec(rt=rt, mmdd="0608", venue="08", race=race, kubun=kubun, created=created,
                   sales=sales)  # fmt: skip

    def win(*slots: Slot) -> dict:
        return {"WIN": Sale(slots, total=sum(n for _, n, _ in slots))}

    def tierce(*slots: Slot) -> dict:
        return {"TIERCE": Sale(slots, total=sum(n for _, n, _ in slots))}

    write_year(
        data, "H1", 2024,
        [
            kyo("H1", "10", "2", "20240607", win(("06", 1000, 1))),
            kyo("H1", "10", "5", "20240610", win(("06", 4810, 1), ("11", 2600, 2))),
            kyo("H1", "11", "5", "20240608", win(("13", 990, 1))),
            kyo("H1", "10", "4", "20240608", win(("06", 4800, 1), ("11", 2590, 2))),
            kyo("H1", "12", "4", "20240608", win(("01", 220, 1))),
            kyo("H1", "11", "0", "20240611", {}),
            # Same day and race number at another venue: another race.
            Rec(mmdd="0608", venue="05", race="10", kubun="2", created="20240607",
                sales=win(("09", 330, 1))),
        ],
    )  # fmt: skip
    write_year(
        data, "H6", 2024,
        [
            kyo("H6", "10", "5", "20240609", tierce(("061102", 45, 1), ("110602", 12, 2))),
            kyo("H6", "10", "4", "20240608", tierce(("061102", 40, 1))),
            # H1's deletion of race 11 leaves its H6 record alone.
            kyo("H6", "11", "5", "20240608", tierce(("130102", 9, 1))),
        ],
    )  # fmt: skip
    votes.build_pool_votes()
    k10, k11, k12 = "jra:2024-06-08:KYO:10", "jra:2024-06-08:KYO:11", "jra:2024-06-08:KYO:12"
    t10 = "jra:2024-06-08:TOK:10"
    assert store_rows(data) == [
        c(k10, "TIERCE", "11,6,2", 12, 2), c(k10, "TIERCE", "6,11,2", 45, 1),
        p(k10, "TIERCE", 57),
        c(k10, "WIN", "11", 2600, 2), c(k10, "WIN", "6", 4810, 1), p(k10, "WIN", 7410),
        c(k11, "TIERCE", "13,1,2", 9, 1), p(k11, "TIERCE", 9),
        c(k12, "WIN", "1", 220, 1), p(k12, "WIN", 220),
        c(t10, "WIN", "9", 330, 1), p(t10, "WIN", 330),
    ]  # fmt: skip
    assert kubun_of(pq.read_table(store_path(data))) == {
        f"{k10}/H1": {"5"}, f"{k10}/H6": {"5"}, f"{k11}/H6": {"5"}, f"{k12}/H1": {"4"},
        f"{t10}/H1": {"2"},
    }  # fmt: skip


def test_a_year_read_in_chunks_equals_one_read(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Five records per file, the two versions of race 10 in different chunks of two.
    def r(rt: str, race: str, kubun: str, created: str, slots: list[Slot]) -> Rec:
        code = "WIN" if rt == "H1" else "TIERCE"
        total = sum(n for _, n, _ in slots)
        return Rec(rt=rt, year="2022", mmdd="1009", venue="04", race=race, kubun=kubun,
                   created=created, sales={code: Sale(slots, total=total)})  # fmt: skip

    write_year(
        data, "H1", 2022,
        [
            r("H1", "10", "4", "20221009", [("03", 700, 1)]),
            r("H1", "01", "5", "20221011", [("01", 50, 1), ("02", 40, 2)]),
            r("H1", "02", "5", "20221011", [("08", 12, 1)]),
            r("H1", "10", "5", "20221011", [("03", 720, 1), ("05", 80, 2)]),
            r("H1", "03", "5", "20221011", [("11", 9, 1)]),
        ],
    )  # fmt: skip
    write_year(
        data, "H6", 2022,
        [
            r("H6", "03", "5", "20221011", [("110102", 3, 1)]),
            r("H6", "10", "4", "20221009", [("030501", 20, 1)]),
            r("H6", "01", "5", "20221011", [("010203", 6, 1)]),
            r("H6", "02", "5", "20221011", [("080102", 4, 1)]),
            r("H6", "10", "5", "20221011", [("030501", 21, 1), ("050301", 8, 2)]),
        ],
    )  # fmt: skip
    decoded: list[int] = []
    original = records._decode_lines

    def counting(lines, layout_, first=1):
        decoded.append(len(lines))
        return original(lines, layout_, first)

    monkeypatch.setattr(records, "_decode_lines", counting)
    chunked = votes.build_pool_votes(out=tmp_path / "chunked", chunk_records=2)
    assert decoded and max(decoded) <= 2, decoded
    decoded.clear()
    whole = votes.build_pool_votes(out=tmp_path / "whole")

    a = pq.read_table(tmp_path / "chunked" / FILENAME)
    b = pq.read_table(tmp_path / "whole" / FILENAME)
    assert a.equals(b)
    whole_bytes = (tmp_path / "whole" / FILENAME).read_bytes()
    assert (tmp_path / "chunked" / FILENAME).read_bytes() == whole_bytes
    assert chunked.to_json() == whole.to_json()
    n01, n02, n03, n10 = (f"jra:2022-10-09:NII:{n}" for n in ("01", "02", "03", "10"))
    assert rows_of(a) == [
        c(n01, "TIERCE", "1,2,3", 6, 1), p(n01, "TIERCE", 6),
        c(n01, "WIN", "1", 50, 1), c(n01, "WIN", "2", 40, 2), p(n01, "WIN", 90),
        c(n02, "TIERCE", "8,1,2", 4, 1), p(n02, "TIERCE", 4),
        c(n02, "WIN", "8", 12, 1), p(n02, "WIN", 12),
        c(n03, "TIERCE", "11,1,2", 3, 1), p(n03, "TIERCE", 3),
        c(n03, "WIN", "11", 9, 1), p(n03, "WIN", 9),
        c(n10, "TIERCE", "3,5,1", 21, 1), c(n10, "TIERCE", "5,3,1", 8, 2), p(n10, "TIERCE", 29),
        c(n10, "WIN", "3", 720, 1), c(n10, "WIN", "5", 80, 2), p(n10, "WIN", 800),
    ]  # fmt: skip


def test_two_builds_write_identical_bytes(data: Path) -> None:
    write_build_races(data)
    extra = Rec(year="2010", mmdd="0105", venue="06", race="01", created="20100106",
                sales={"WIN": Sale([("08", 1110, 1)], total=1110)})  # fmt: skip
    write_year(data, "H1", 2010, [extra])
    first_report = votes.build_pool_votes()
    path = store_path(data)
    first = path.read_bytes()
    assert pq.read_metadata(path).num_rows == len(ROWS_2023) + len(ROWS_2024) + 2

    # The same inputs rewritten (new mtimes): the same bytes and the same report.
    write_build_races(data)
    second_report = votes.build_pool_votes()
    assert path.read_bytes() == first
    assert second_report.to_json() == first_report.to_json()
    assert sorted(q.name for q in path.parent.iterdir()) == [FILENAME]


# ------------------------------------------------------------------------------------------------
# The real export (local: needs $HORSETRACK_DATA_DIR with the export and the workbook)

REAL_YEAR = 2004  # the first year with H6 (Trifecta from autumn), every H1 pool sold
REAL_RECORDS = 60  # the first records of each type's year file (a whole year took ~7 min, #502)
LEGS ={"WIN": 1, "PLACE": 1, "BRACKET_QUINELLA": 2, "QUINELLA": 2, "QUINELLA_PLACE": 2,
        "FORECAST": 2, "TRIO": 3, "TIERCE": 3}  # fmt: skip
ORDERED = {"FORECAST", "TIERCE"}
H1_NAMES = ("単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複")
H1_JV = dict(zip(H1_CODES, H1_NAMES, strict=True))


def _expected_counts(cols: Mapping[str, np.ndarray], jv: str, number: str) -> dict[str, int]:
    """From the decoded records alone: the combination rows, pool rows and refunds expected."""
    jra = np.isin(cols["競馬場コード"], [v.encode() for v in VENUES])
    sold = jra & (cols[f"発売フラグ{jv}"] == b"7")
    combo = cols[f"{jv}票数/{number}"]
    valid = np.char.isdigit(combo) & (records.integer(combo) > 0)
    refunds = ~np.isnan(records.integer(cols[f"{jv}返還票数合計"]))
    return {
        "combinations": int((valid & sold[:, None]).sum()),
        "pools": int(sold.sum()),
        "refunds": int((refunds & sold).sum()),
    }


@pytest.mark.local
def test_a_real_year_maps(tmp_path: Path) -> None:
    export, workbook = wb.real_export_or_skip()
    real = {rt: export / "stored" / "RACE" / rt / f"{REAL_YEAR}.jvd" for rt in ("H1", "H6")}
    if not all(s.is_file() for s in real.values()):
        pytest.skip(f"no H1 or H6 file for {REAL_YEAR}")
    raw = tmp_path / "export"
    sources = {}
    for rt, source in real.items():
        with open(source, "rb") as handle:
            lines = [line for _, line in zip(range(REAL_RECORDS), handle, strict=False)]
        sources[rt] = raw / "stored" / "RACE" / rt / f"{REAL_YEAR}.jvd"
        sources[rt].parent.mkdir(parents=True)
        sources[rt].write_bytes(b"".join(lines))
    out = tmp_path / "out"
    report = votes.build_pool_votes(years=[REAL_YEAR], raw=raw, out=out)
    got = pq.read_table(out / FILENAME)
    assert got.schema == votes.POOL_VOTES_SCHEMA
    check_schema(got.schema)
    df = got.to_pandas()
    n = len(df)
    assert n > 0 and report.counts["rows"] == n
    assert set(df["pool"]) == set(CODES)
    assert (df["jurisdiction"] == "jra").all()
    parts = df["race_id"].str.split(":", expand=True)
    venues = "|".join(VENUES.values())
    assert df["race_id"].str.fullmatch(rf"jra:{REAL_YEAR}-\d\d-\d\d:(?:{venues}):\d\d").all()
    assert (df["venue"] == parts[2]).all()
    assert (df["race_no"] == parts[3].astype(int)).all()
    assert (pd.to_datetime(df["race_date"]).dt.strftime("%Y-%m-%d") == parts[1]).all()
    assert set(df["data_kubun"]) <= {"2", "4", "5", "9"}
    # Sorted by race_id, pool, combination and unique on them.
    triples = list(zip(df["race_id"], df["pool"], df["combination"], strict=True))
    assert triples == sorted(triples)
    assert len(set(triples)) == n

    is_pool = df["combination"] == POOL_ROW
    combos, pools = df[~is_pool], df[is_pool]
    assert combos["refund_votes"].isna().all()
    assert pools["fav_rank"].isna().all()
    assert combos["votes"].notna().all() and (combos["votes"] >= 0).all()
    assert pools["votes"].notna().all() and (pools["votes"] >= 0).all()
    # Every pool with combinations has its pool row, and its slots add up to the total (the
    # export's totals include the refunded votes, as do the slots).
    sums = combos.groupby(["race_id", "pool"])["votes"].sum().sort_index()
    totals = pools.set_index(["race_id", "pool"])["votes"].sort_index()
    assert sums.index.equals(totals.index)
    assert (sums == totals).all()

    # The combination forms.
    for pool, legs in LEGS.items():
        mine = combos.loc[combos["pool"] == pool, "combination"]
        numbers = mine.str.split(",", expand=True)
        assert numbers.shape[1] == legs, pool
        assert numbers.apply(lambda s: s.str.fullmatch(r"[1-9][0-9]?")).all().all(), pool
        numbers = numbers.astype(int).to_numpy()
        top = 8 if pool == "BRACKET_QUINELLA" else 18
        assert ((numbers >= 1) & (numbers <= top)).all(), pool
        if legs == 1:
            continue
        steps = np.diff(numbers, axis=1)
        if pool in ORDERED:
            pairs = [(0, 1)] if legs == 2 else [(0, 1), (0, 2), (1, 2)]
            assert all((numbers[:, i] != numbers[:, j]).all() for i, j in pairs), pool
        elif pool == "BRACKET_QUINELLA":
            assert (steps >= 0).all(), pool
        else:
            assert (steps > 0).all(), pool

    # The counts, independently of the module, from the decoded records of the same files.
    del df, got, combos, pools
    expected = {}
    for rt, path in sources.items():
        cols = records.decode(path.read_bytes(), spec.record_spec(rt, workbook))
        layout_ = H1_JV if rt == "H1" else {"TIERCE": "3連単"}
        for code, jv in layout_.items():
            number = "馬番" if code in ("WIN", "PLACE") else "組番"
            expected[code] = _expected_counts(cols, jv, number)
        del cols
    wanted = ["pool", "combination", "refund_votes"]
    table = pq.read_table(out / FILENAME, columns=wanted).to_pandas()
    for code in CODES:
        mine = table[table["pool"] == code]
        on_pool = mine["combination"] == POOL_ROW
        assert {
            "combinations": int((~on_pool).sum()),
            "pools": int(on_pool.sum()),
            "refunds": int(mine.loc[on_pool, "refund_votes"].notna().sum()),
        } == expected[code], code
