"""JV-Data O1-O6 (final odds) records into the jravan `final_odds` table (ticket #367).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jravan.odds`:
- `odds_rows(record_type: str, cols: Mapping[str, numpy.ndarray]) -> pyarrow.Table`: `cols` is
  `records.decode` of `record_type` (O1..O6) records; one row per record and filled slot of each
  sold pool, plus one pool row per sold pool, in the columns `ROW_COLUMNS` (in that order) with
  the types of `FINAL_ODDS_SCHEMA`; the row order is free.
- `build_final_odds(*, years=None, raw=None, out=None) -> records.BuildReport`: reads
  `<raw>/stored/RACE/<O1..O6>/<YYYY>.jvd` (`raw` the export root, default
  `paths.raw_dir('jravan')`; `years` None reads every year file of every type, and a type with no
  file for a year adds nothing), with the layouts `spec.record_spec(<type>)`; keeps one version
  per race and record type and JRA venues only; writes `<out>/final_odds.parquet` (`out` a
  folder, default `paths.processed_dir('jravan')`) with `FINAL_ODDS_SCHEMA`, one row group per
  race day (ascending; each group's `race_date` statistics record that one day, #502), sorted by
  `race_id, pool, combination` (plain string order), the same bytes for the same inputs, through
  a tmp file and `os.replace`.
- `FINAL_ODDS_SCHEMA`: equal to `EXPECTED_SCHEMA` below. `final_odds` is not in
  `schemas.TABLES`. Its availability rule is `label_only` (#376, the owner's request of
  2026-11-26: a label or a conditioner, never a feature at t): `read_asof` and `asof_join` on a
  view of it raise `LabelOnlyTable` at any t, and `read_label` returns it whole.
- `horsetrack japan jravan final-odds [--year Y ...]`: the same build, then
  `records.write_build_report`, written as `<reports dir>/jravan-final-odds.json` (counts only;
  `by_year` keyed by the years that have rows).

The workbook items each column comes from (O1-O6, the JV-Data 4.9.0.1 workbook on this Mac; names
as `spec.record_spec` returns them, keys as `records.decode` builds them):
- `jurisdiction`: `jra`. Only records whose `競馬場コード` is a JRA venue (01-10) are kept.
- `race_id`: `keys.race_id(開催年, 開催月日, 競馬場コード, レース番号)`; `race_date`, `venue`
  (the short code of config/jra.yaml), `race_no`: the same race's date, venue and number.
- one version per race and record type: `records.latest_versions` on `開催年`, `開催月日`,
  `競馬場コード`, `開催回[第N回]`, `開催日目[N日目]`, `レース番号` (`データ作成年月日`,
  `データ区分`; a race whose latest record of a type is a deletion, `データ区分` 0, has no rows of
  that type). `data_kubun`: the kept record's `データ区分`.
- `published_utc`: `keys.announced_utc(開催年, 開催月日, 発表月日時分)` (null for `00000000`), on
  every row of the record, pool rows included.
- `pool`, per record type, sale flag and repeat group (the `pool` codes of config/jra.yaml):
    O1 `発売フラグ単勝` `単勝オッズ` WIN, `発売フラグ複勝` `複勝オッズ` PLACE,
       `発売フラグ枠連` `枠連オッズ` BRACKET_QUINELLA (`複勝着払キー` unused);
    O2 `発売フラグ馬連` `馬連オッズ` QUINELLA; O3 `発売フラグワイド` `ワイドオッズ` QUINELLA_PLACE;
    O4 `発売フラグ馬単` `馬単オッズ` FORECAST; O5 `発売フラグ3連複` `3連複オッズ` TRIO;
    O6 `発売フラグ3連単` `3連単オッズ` TIERCE.
  A pool is sold when its sale flag is `7`; flag `0` (not sold) writes no row, pool row included,
  whatever its slots and total hold.
- `combination`: the group's `馬番` (WIN, PLACE) or `組番` (the others), the legs (1, 2 or 3) as
  equal-width numbers (a bracket one digit, a horse two), joined with `,`, leading zeros dropped,
  FORECAST and TIERCE in the printed order and the others ascending (`0704` is `4,7` on O2 and
  `7,4` on O4; `070401` is `7,4,1` on O6). A slot whose number item is blank writes no row.
- `odds`: `<group>/オッズ`, or `<group>/最低オッズ` for PLACE and QUINELLA_PLACE, the decimal price
  (stake included) with one implied decimal place at any field width; `odds_hi`:
  `<group>/最高オッズ` the same way for those two pools, null for the others. A price that is not
  a positive number (`0000` no votes, `----` cancelled before sale, `****` cancelled after sale)
  is null and its raw text is `status_raw`; `status_raw` is null for a positive price.
- `fav_rank`: `<group>/人気順` as an integer, null when it is not one (`--`, `**`, blank).
- `investment`: null on slot rows. Each sold pool's vote total, `単勝票数合計`, `複勝票数合計`,
  `枠連票数合計`, `馬連票数合計`, `ワイド票数合計`, `馬単票数合計`, `3連複票数合計`,
  `3連単票数合計` (in units of 100 yen), is a row with `combination` `__pool__`, `investment` the
  total times 100 (yen), and null `odds`, `odds_hi`, `fav_rank` and `status_raw`.

Every layout here is made up in the workbook's shape (`jravan_workbook.write_workbook`): the item
names are the workbook's, the widths of the prices, ranks and totals, every repeat count and so
every position are invented, so nothing may be sliced by hand. Every record and value is invented
too. Expected rows are written out by hand.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.core.store import schemas
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

from horsetrack.countries.japan.jravan import odds, records, spec

SOURCE = "jravan"
UTC = timezone.utc
JST = timezone(timedelta(hours=9))
POOL_ROW = "__pool__"
FILENAME = "final_odds.parquet"

EXPECTED_SCHEMA = pa.schema(
    [
        ("jurisdiction", pa.string()),
        ("race_id", pa.string()),
        ("race_date", pa.date32()),
        ("venue", pa.string()),
        ("race_no", pa.int64()),
        ("pool", pa.string()),
        ("combination", pa.string()),
        ("odds", pa.float64()),
        ("odds_hi", pa.float64()),
        ("fav_rank", pa.int64()),
        ("investment", pa.float64()),
        ("status_raw", pa.string()),
        ("published_utc", pa.timestamp("us", tz="UTC")),
        ("data_kubun", pa.string()),
    ]
)
ROW_COLUMNS = [
    "race_id", "race_date", "venue", "race_no", "pool", "combination", "odds", "odds_hi",
    "fav_rank", "investment", "status_raw", "published_utc", "data_kubun",
]  # fmt: skip
CODES = (
    "WIN", "PLACE", "BRACKET_QUINELLA", "QUINELLA", "QUINELLA_PLACE", "FORECAST", "TRIO", "TIERCE",
)  # fmt: skip
RECORD_TYPES = ("O1", "O2", "O3", "O4", "O5", "O6")
VENUES = {  # number -> short code, as config/jra.yaml lists them
    "01": "SAP", "02": "HAK", "03": "FKS", "04": "NII", "05": "TOK",
    "06": "NAK", "07": "CKY", "08": "KYO", "09": "HSN", "10": "KOK",
}  # fmt: skip

# ------------------------------------------------------------------------------------------------
# Made-up O1-O6 layouts: the workbook's item names, invented widths and repeats


TOTAL = 12  # bytes of a vote total (invented)


@dataclass(frozen=True)
class PoolDef:
    code: str
    record_type: str
    sale: str  # the sale flag item
    group: str  # the repeat group
    number: str  # the group's combination item
    number_width: int  # legs x digits
    prices: tuple[str, ...]  # one price item, or the low and high items
    price_width: int  # invented
    rank_width: int  # invented
    total: str  # the vote total item
    repeat: int  # slots (invented)
    legs: int


LOW_HIGH = ("最低オッズ", "最高オッズ")
POOLS = {
    p.code: p
    for p in (
        PoolDef("WIN", "O1", "発売フラグ単勝", "単勝オッズ", "馬番", 2, ("オッズ",), 5, 3,
                "単勝票数合計", 6, 1),
        PoolDef("PLACE", "O1", "発売フラグ複勝", "複勝オッズ", "馬番", 2, LOW_HIGH, 5, 3,
                "複勝票数合計", 6, 1),
        PoolDef("BRACKET_QUINELLA", "O1", "発売フラグ枠連", "枠連オッズ", "組番", 2, ("オッズ",), 6,
                3, "枠連票数合計", 8, 2),
        PoolDef("QUINELLA", "O2", "発売フラグ馬連", "馬連オッズ", "組番", 4, ("オッズ",), 7, 4,
                "馬連票数合計", 6, 2),
        PoolDef("QUINELLA_PLACE", "O3", "発売フラグワイド", "ワイドオッズ", "組番", 4, LOW_HIGH,
                6, 4, "ワイド票数合計", 6, 2),
        PoolDef("FORECAST", "O4", "発売フラグ馬単", "馬単オッズ", "組番", 4, ("オッズ",), 7, 4,
                "馬単票数合計", 6, 2),
        PoolDef("TRIO", "O5", "発売フラグ3連複", "3連複オッズ", "組番", 6, ("オッズ",), 7, 4,
                "3連複票数合計", 6, 3),
        PoolDef("TIERCE", "O6", "発売フラグ3連単", "3連単オッズ", "組番", 6, ("オッズ",), 8, 5,
                "3連単票数合計", 7, 3),
    )
}  # fmt: skip
TYPE_POOLS = {rt: [p for p in POOLS.values() if p.record_type == rt] for rt in RECORD_TYPES}
GROUP_POOL = {p.group: p.code for p in POOLS.values()}
TOTAL_POOL = {p.total: p.code for p in POOLS.values()}
SALE_POOL = {p.sale: p.code for p in POOLS.values()}

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
    ("発表月日時分", 8),
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


def layout(rt: str) -> tuple[Field, ...]:
    pools = TYPE_POOLS[rt]
    out = [Field(name, length) for name, length in HEAD]
    out += [Field(p.sale, 1) for p in pools]
    if rt == "O1":
        out.append(Field("複勝着払キー", 1))
    for p in pools:
        subs = (
            (p.number, p.number_width),
            *((name, p.price_width) for name in p.prices),
            ("人気順", p.rank_width),
        )
        out.append(Field(p.group, 0, p.repeat, subs))
    out += [Field(p.total, TOTAL) for p in pools]
    out.append(Field("予備", 3))
    out.append(Field("レコード区切", 2))
    return tuple(out)


def section(rt: str, number: int) -> wb.Section:
    rows: list[wb.Row] = []
    pos = 1
    for no, f in enumerate(layout(rt), start=1):
        if f.subs:
            rows.append(wb.group(no, f"<{f.name}>", pos, f.width, f.repeat, desc="オッズ（架空）"))
            rel = 1
            for i, (name, length) in enumerate(f.subs):
                rows.append(wb.sub(wb.letter_code(i), name, rel, length, desc="値（架空）"))
                rel += length
            pos += f.width * f.repeat
            continue
        if no == 1:
            rows.append(wb.item(1, f.name, 1, f.length, key="●", initial=rt, desc=wb.id_desc(rt)))
        else:
            rows.append(wb.item(no, f.name, pos, f.length, desc="（架空）"))
        pos += f.length * f.repeat
    return wb.Section(number, f"オッズ{rt[1]}（架空）", pos - 1, tuple(rows))


SECTIONS = {rt: section(rt, i + 2) for i, rt in enumerate(RECORD_TYPES)}


def write_odds_workbook(path: Path) -> Path:
    decoy = wb.simple_section(1, "XQ", "架空その他", [("架空項目", 6)])
    return wb.write_workbook(path, [decoy, *SECTIONS.values()], [])


Price = float | str | None  # a decimal price, a raw text of the field's width, or blank
Rank = int | str | None


def tenths(value: float, width: int) -> str:
    return f"{round(value * 10):0{width}d}"


def fit(value: str, length: int) -> str:
    """A field as printed: `value` at the field's exact width, or blank."""
    if value and len(value) != length:
        raise ValueError(f"{value!r} is not {length} wide")
    return value or " " * length


def s(number: str, price: Price, rank: Rank) -> tuple:
    """A one-price slot (WIN, BRACKET_QUINELLA, QUINELLA, FORECAST, TRIO, TIERCE)."""
    return (number, price, rank)


def r(number: str, low: Price, high: Price, rank: Rank) -> tuple:
    """A ranged slot (PLACE, QUINELLA_PLACE)."""
    return (number, low, high, rank)


@dataclass
class Odds:
    """One made-up O1-O6 record; unused slots and totals are blank. A pool's sale flag is `7`
    when it has slots or a total, else `0`, unless `sold` says otherwise."""

    rt: str
    year: str = "2024"
    mmdd: str = "0601"
    venue: str = "05"
    race: str = "11"
    kai: str = "03"
    nichi: str = "02"
    kubun: str = "4"
    created: str = "20240603"
    announced: str = "00000000"
    slots: Mapping[str, Sequence[tuple]] = field(default_factory=dict)
    totals: Mapping[str, int] = field(default_factory=dict)
    sold: Mapping[str, str] = field(default_factory=dict)

    def flag(self, code: str) -> str:
        default = "7" if code in self.slots or code in self.totals else "0"
        return self.sold.get(code, default)

    def record(self) -> bytes:
        mine = {p.code for p in TYPE_POOLS[self.rt]}
        assert set(self.slots) <= mine and set(self.totals) <= mine, (self.rt, self.slots)
        head = dict(
            zip(
                [name for name, _ in HEAD],
                (self.rt, self.kubun, self.created, self.year, self.mmdd, self.venue, self.kai,
                 self.nichi, self.race, self.announced, "16", "16"),
                strict=True,
            )
        )  # fmt: skip
        out: list[str] = []
        for f in layout(self.rt)[:-1]:
            if f.subs:
                pool = POOLS[GROUP_POOL[f.name]]
                slots = list(self.slots.get(pool.code, ()))
                assert len(slots) <= f.repeat, (f.name, slots)
                for k in range(f.repeat):
                    slot = slots[k] if k < len(slots) else ("",) + (None,) * (len(f.subs) - 1)
                    assert len(slot) == len(f.subs), (f.name, slot)
                    number, *prices, rank = slot
                    out.append(fit(number, pool.number_width))
                    for price in prices:
                        if isinstance(price, float | int) and not isinstance(price, bool):
                            out.append(tenths(float(price), pool.price_width))
                        else:
                            out.append(fit(price or "", pool.price_width))
                    if isinstance(rank, int):
                        out.append(f"{rank:0{pool.rank_width}d}")
                    else:
                        out.append(fit(rank or "", pool.rank_width))
            elif f.name in SALE_POOL:
                out.append(self.flag(SALE_POOL[f.name]))
            elif f.name in TOTAL_POOL:
                votes = self.totals.get(TOTAL_POOL[f.name])
                out.append(" " * TOTAL if votes is None else f"{votes:0{TOTAL}d}")
            elif f.name == "複勝着払キー":
                out.append("3" if self.flag("PLACE") == "7" else "0")
            elif f.name == "予備":
                out.append(" " * f.length)
            else:
                out.append(fit(head[f.name], f.length))
        body = "".join(out).encode("cp932")
        assert len(body) == SECTIONS[self.rt].length - 2
        return body + b"\r\n"


# ------------------------------------------------------------------------------------------------
# Helpers


@pytest.fixture(scope="module")
def layouts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, spec.RecordSpec]:
    path = write_odds_workbook(tmp_path_factory.mktemp("workbook") / wb.WORKBOOK_NAME)
    return {rt: spec.record_spec(rt, path) for rt in RECORD_TYPES}


def rows_table(layouts: Mapping[str, spec.RecordSpec], rt: str, recs: Sequence[Odds]) -> pa.Table:
    assert all(x.rt == rt for x in recs)
    cols = records.decode(b"".join(x.record() for x in recs), layouts[rt])
    table = odds.odds_rows(rt, cols)
    assert isinstance(table, pa.Table)
    assert table.column_names == ROW_COLUMNS
    for name in ROW_COLUMNS:
        assert table.schema.field(name).type == EXPECTED_SCHEMA.field(name).type, name
    return table


def _num(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


Row = tuple  # (race_id, pool, combination, odds, odds_hi, fav_rank, investment, status_raw)


def rows_of(table: pa.Table) -> list[Row]:
    return [
        (d["race_id"], d["pool"], d["combination"], _num(d["odds"]), _num(d["odds_hi"]),
         d["fav_rank"], _num(d["investment"]), d["status_raw"])
        for d in table.to_pylist()
    ]  # fmt: skip


def bag(rows: Sequence[tuple]) -> list[tuple]:
    """Rows in a canonical order, for comparing without the row order."""
    return sorted(rows, key=repr)


def row(race: str, pool: str, combination: str, price: float | None, hi: float | None = None,
        rank: int | None = None, status: str | None = None) -> Row:  # fmt: skip
    return (race, pool, combination, _num(price), _num(hi), rank, None, status)


def pool_row(race: str, pool: str, votes: int) -> Row:
    return (race, pool, POOL_ROW, None, None, None, float(votes * 100), None)


def race_keys(race: str) -> tuple[date, str, int]:
    _, day, venue, number = race.split(":")
    return date.fromisoformat(day), venue, int(number)


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data dir with the made-up workbook and an empty export."""
    root = tmp_path / "data"
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(root))
    write_odds_workbook(root / "specs" / "jravan" / wb.WORKBOOK_NAME)
    for rt in RECORD_TYPES:
        (root / "raw" / SOURCE / "stored" / "RACE" / rt).mkdir(parents=True)
    return root


def write_year(data: Path, year: int, recs: Sequence[Odds]) -> None:
    """One `<type>/<year>.jvd` per record type present in `recs`, records in the given order."""
    for rt in RECORD_TYPES:
        mine = [x.record() for x in recs if x.rt == rt]
        if mine:
            wb.write_jvd(data / "raw" / SOURCE / "stored" / "RACE" / rt / f"{year}.jvd", mine)


def store_path() -> Path:
    return paths.processed_dir(SOURCE) / FILENAME


FullRow = tuple  # every column of EXPECTED_SCHEMA, in order


def full_rows(table: pa.Table) -> list[FullRow]:
    out = []
    for d in table.to_pylist():
        values = [d[c] for c in EXPECTED_SCHEMA.names]
        for c in ("odds", "odds_hi", "investment"):
            i = EXPECTED_SCHEMA.names.index(c)
            values[i] = _num(values[i])
        out.append(tuple(values))
    return out


def full(r_: Row, *, published: datetime | None = None, kubun: str = "4") -> FullRow:
    race, pool, combination, price, hi, rank, investment, status = r_
    day, venue, number = race_keys(race)
    return ("jra", race, day, venue, number, pool, combination, price, hi, rank, investment,
            status, published, kubun)  # fmt: skip


def in_store_order(rows: Sequence[FullRow]) -> list[FullRow]:
    return sorted(rows, key=lambda x: (x[1], x[5], x[6]))


# ------------------------------------------------------------------------------------------------
# odds_rows


def test_win_place_and_bracket_odds_become_long_rows(layouts: dict[str, spec.RecordSpec]) -> None:
    a = Odds(
        "O1", year="2024", mmdd="0601", venue="05", race="11", kubun="4",
        slots={
            "WIN": [s("01", 5.6, 2), s("04", 3.5, 1), s("07", 12.3, 3), s("12", 999.9, 4)],
            "PLACE": [r("01", 1.8, 2.6, 2), r("04", 1.2, 1.5, 1), r("07", 3.1, 4.4, 3),
                      r("12", 48.0, 61.7, 4)],
            "BRACKET_QUINELLA": [s("12", 15.2, 3), s("14", 6.1, 1), s("44", 88.8, 5),
                                 s("62", 120.5, 4), s("11", 2345.6, 6)],
        },
        totals={"WIN": 123456, "PLACE": 54321, "BRACKET_QUINELLA": 9876},
    )  # fmt: skip
    b = Odds(
        "O1", year="2019", mmdd="1228", venue="06", race="09", kai="05", nichi="09", kubun="5",
        created="20191230",
        slots={
            "WIN": [s("03", 2.2, 1), s("16", 45.6, 9)],
            "PLACE": [r("03", 1.1, 1.3, 1), r("16", 9.9, 14.0, 9)],
            "BRACKET_QUINELLA": [s("18", 30.0, 7), s("88", 250.0, 20), s("53", 9.1, 2)],
        },
        totals={"WIN": 500001, "PLACE": 250002, "BRACKET_QUINELLA": 30003},
    )  # fmt: skip
    ra, rb = "jra:2024-06-01:TOK:11", "jra:2019-12-28:NAK:09"
    expected = [
        row(ra, "WIN", "1", 5.6, rank=2),
        row(ra, "WIN", "4", 3.5, rank=1),
        row(ra, "WIN", "7", 12.3, rank=3),
        row(ra, "WIN", "12", 999.9, rank=4),
        row(ra, "PLACE", "1", 1.8, 2.6, 2),
        row(ra, "PLACE", "4", 1.2, 1.5, 1),
        row(ra, "PLACE", "7", 3.1, 4.4, 3),
        row(ra, "PLACE", "12", 48.0, 61.7, 4),
        row(ra, "BRACKET_QUINELLA", "1,2", 15.2, rank=3),
        row(ra, "BRACKET_QUINELLA", "1,4", 6.1, rank=1),
        row(ra, "BRACKET_QUINELLA", "4,4", 88.8, rank=5),
        row(ra, "BRACKET_QUINELLA", "2,6", 120.5, rank=4),
        row(ra, "BRACKET_QUINELLA", "1,1", 2345.6, rank=6),
        pool_row(ra, "WIN", 123456),
        pool_row(ra, "PLACE", 54321),
        pool_row(ra, "BRACKET_QUINELLA", 9876),
        row(rb, "WIN", "3", 2.2, rank=1),
        row(rb, "WIN", "16", 45.6, rank=9),
        row(rb, "PLACE", "3", 1.1, 1.3, 1),
        row(rb, "PLACE", "16", 9.9, 14.0, 9),
        row(rb, "BRACKET_QUINELLA", "1,8", 30.0, rank=7),
        row(rb, "BRACKET_QUINELLA", "8,8", 250.0, rank=20),
        row(rb, "BRACKET_QUINELLA", "3,5", 9.1, rank=2),
        pool_row(rb, "WIN", 500001),
        pool_row(rb, "PLACE", 250002),
        pool_row(rb, "BRACKET_QUINELLA", 30003),
    ]
    keys = {ra: (date(2024, 6, 1), "TOK", 11, "4"), rb: (date(2019, 12, 28), "NAK", 9, "5")}
    # Every JRA venue, each race with another horse at another price; Place and Bracket unsold.
    recs = [a, b]
    for i, (number, code) in enumerate(VENUES.items()):
        recs.append(
            Odds("O1", year="2010", mmdd="0815", venue=number, race=f"{i + 1:02d}",
                 created="20100816", kubun="4",
                 slots={"WIN": [s(f"{i + 3:02d}", 1.5 + i, i + 1)]}, totals={"WIN": 1000 + i})
        )  # fmt: skip
        race = f"jra:2010-08-15:{code}:{i + 1:02d}"
        expected += [
            row(race, "WIN", str(i + 3), 1.5 + i, rank=i + 1),
            pool_row(race, "WIN", 1000 + i),
        ]
        keys[race] = (date(2010, 8, 15), code, i + 1, "4")

    table = rows_table(layouts, "O1", recs)
    assert bag(rows_of(table)) == bag(expected)
    for d in table.to_pylist():
        assert (d["race_date"], d["venue"], d["race_no"], d["data_kubun"]) == keys[d["race_id"]]
        assert d["published_utc"] is None


ORDER_CASES = {  # pool -> (number item as printed, combination)
    "QUINELLA": [("0704", "4,7"), ("1203", "3,12"), ("0109", "1,9"), ("1018", "10,18")],
    "QUINELLA_PLACE": [("0704", "4,7"), ("1602", "2,16"), ("0510", "5,10"), ("0911", "9,11")],
    "FORECAST": [("0704", "7,4"), ("1203", "12,3"), ("0102", "1,2"), ("1009", "10,9")],
    "TRIO": [("070401", "1,4,7"), ("120309", "3,9,12"), ("011805", "1,5,18"), ("100201", "1,2,10")],
    "TIERCE": [("070401", "7,4,1"), ("100201", "10,2,1"), ("011805", "1,18,5"),
               ("120309", "12,3,9"), ("010203", "1,2,3")],
}  # fmt: skip
EXOTIC_PRICES = [12.3, 456.7, 8.9, 1.0, 12345.6]


def test_each_exotic_record_type_maps_to_its_pool_and_combination_order(
    layouts: dict[str, spec.RecordSpec],
) -> None:
    for rt in ("O2", "O3", "O4", "O5", "O6"):
        (pool,) = TYPE_POOLS[rt]
        ranged = len(pool.prices) == 2
        recs, expected = [], []
        for v, (venue, code) in enumerate((("08", "KYO"), ("10", "KOK"))):
            race = f"jra:2024-06-02:{code}:10"
            slots = []
            for j, (printed, combination) in enumerate(ORDER_CASES[pool.code]):
                price = EXOTIC_PRICES[j] + 10 * v
                rank = j + 1 + v
                if ranged:
                    slots.append(r(printed, price, price + 2.5, rank))
                    expected.append(row(race, pool.code, combination, price, price + 2.5, rank))
                else:
                    slots.append(s(printed, price, rank))
                    expected.append(row(race, pool.code, combination, price, rank=rank))
            votes = 7000 + 111 * v + len(rt)
            recs.append(
                Odds(rt, mmdd="0602", venue=venue, race="10", slots={pool.code: slots},
                     totals={pool.code: votes})
            )  # fmt: skip
            expected.append(pool_row(race, pool.code, votes))
        got = rows_of(rows_table(layouts, rt, recs))
        assert {x[1] for x in got} == {pool.code}, rt
        assert bag(got) == bag(expected), rt


def test_a_price_that_is_not_a_number_is_null_with_its_raw_text(
    layouts: dict[str, spec.RecordSpec],
) -> None:
    printed = {"WIN": "05", "PLACE": "05", "BRACKET_QUINELLA": "35", "QUINELLA": "0305",
               "QUINELLA_PLACE": "0305", "FORECAST": "0503", "TRIO": "030508",
               "TIERCE": "050308"}  # fmt: skip
    combo = {"WIN": "5", "PLACE": "5", "BRACKET_QUINELLA": "3,5", "QUINELLA": "3,5",
             "QUINELLA_PLACE": "3,5", "FORECAST": "5,3", "TRIO": "3,5,8",
             "TIERCE": "5,3,8"}  # fmt: skip
    valid = {"WIN": "02", "PLACE": "02", "BRACKET_QUINELLA": "12", "QUINELLA": "0102",
             "QUINELLA_PLACE": "0102", "FORECAST": "0201", "TRIO": "010204",
             "TIERCE": "020104"}  # fmt: skip
    valid_combo = {"WIN": "2", "PLACE": "2", "BRACKET_QUINELLA": "1,2", "QUINELLA": "1,2",
                   "QUINELLA_PLACE": "1,2", "FORECAST": "2,1", "TRIO": "1,2,4",
                   "TIERCE": "2,1,4"}  # fmt: skip
    # (price text as a character repeated to the field's width, rank: None for a numeric rank)
    kinds = [("0", None), ("-", "-"), ("*", "*")]
    for rt in RECORD_TYPES:
        recs, expected = [], []
        for k, (char, rank_char) in enumerate(kinds):
            race_no = f"{k + 1:02d}"
            race = f"jra:2024-06-01:TOK:{race_no}"
            slots: dict[str, list[tuple]] = {}
            for pool in TYPE_POOLS[rt]:
                ranged = len(pool.prices) == 2
                raw = char * pool.price_width
                rank: Rank = 9 + k if rank_char is None else rank_char * pool.rank_width
                want_rank = 9 + k if rank_char is None else None
                bad = (
                    (printed[pool.code], raw, raw, rank)
                    if ranged
                    else (printed[pool.code], raw, rank)
                )
                good = r(valid[pool.code], 3.4, 5.6, 1) if ranged else s(valid[pool.code], 3.4, 1)
                slots[pool.code] = [good, bad]
                expected.append(row(race, pool.code, combo[pool.code], None, None, want_rank, raw))
                expected.append(
                    row(race, pool.code, valid_combo[pool.code], 3.4, 5.6 if ranged else None, 1)
                )
                expected.append(pool_row(race, pool.code, 500 + k))
            recs.append(
                Odds(rt, mmdd="0601", venue="05", race=race_no, slots=slots,
                     totals={p.code: 500 + k for p in TYPE_POOLS[rt]})
            )  # fmt: skip
        assert bag(rows_of(rows_table(layouts, rt, recs))) == bag(expected), rt


def test_unsold_pools_and_blank_slots_write_no_rows(layouts: dict[str, spec.RecordSpec]) -> None:
    race = "jra:2024-06-01:TOK:11"
    # Win and Place sold with blank slots between and after the filled ones, a price under a
    # blank number; Bracket Quinella not sold although its slots and total hold numbers.
    o1 = Odds(
        "O1",
        slots={
            "WIN": [s("02", 4.1, 2), s("  ", 9.9, 5), s("05", 2.0, 1), s("  ", None, None)],
            "PLACE": [r("  ", 1.3, 1.8, 2), r("02", 1.4, 2.0, 2), r("  ", None, None, None),
                      r("05", 1.1, 1.2, 1)],
            "BRACKET_QUINELLA": [s("12", 8.8, 1), s("25", 14.0, 2)],
        },
        totals={"WIN": 4000, "PLACE": 2000, "BRACKET_QUINELLA": 999},
        sold={"BRACKET_QUINELLA": "0"},
    )  # fmt: skip
    got = rows_of(rows_table(layouts, "O1", [o1]))
    assert bag(got) == bag(
        [
            row(race, "WIN", "2", 4.1, rank=2),
            row(race, "WIN", "5", 2.0, rank=1),
            row(race, "PLACE", "2", 1.4, 2.0, 2),
            row(race, "PLACE", "5", 1.1, 1.2, 1),
            pool_row(race, "WIN", 4000),
            pool_row(race, "PLACE", 2000),
        ]
    )

    # Not sold, with slots and a total that hold numbers: no row of any kind.
    unsold_tierce = Odds(
        "O6", slots={"TIERCE": [s("010203", 55.5, 1)]}, totals={"TIERCE": 77}, sold={"TIERCE": "0"}
    )
    assert rows_table(layouts, "O6", [unsold_tierce]).num_rows == 0
    # Not sold and blank, as an old year prints Wide: no row.
    assert rows_table(layouts, "O3", [Odds("O3")]).num_rows == 0
    # Sold with every slot blank: the pool row only.
    sold_blank = Odds("O2", slots={"QUINELLA": [s("    ", 6.6, 1)]}, totals={"QUINELLA": 4321})
    assert rows_of(rows_table(layouts, "O2", [sold_blank])) == [pool_row(race, "QUINELLA", 4321)]
    # In one batch with a sold record, the unsold one still adds nothing.
    sold_tierce = Odds("O6", race="12", slots={"TIERCE": [s("030201", 4567.8, 1)]},
                       totals={"TIERCE": 88})  # fmt: skip
    r12 = "jra:2024-06-01:TOK:12"
    assert bag(rows_of(rows_table(layouts, "O6", [unsold_tierce, sold_tierce]))) == bag(
        [row(r12, "TIERCE", "3,2,1", 4567.8, rank=1), pool_row(r12, "TIERCE", 88)]
    )


def test_pool_totals_are_pool_rows_in_yen(layouts: dict[str, spec.RecordSpec]) -> None:
    votes_by_pool = {  # pool -> the totals of the two races (units of 100 yen)
        "WIN": (98765, 12), "PLACE": (43210, 7), "BRACKET_QUINELLA": (1234, 56789),
        "QUINELLA": (765432, 3), "QUINELLA_PLACE": (222222, 101), "FORECAST": (909090, 44),
        "TRIO": (3141592, 2718), "TIERCE": (98765432109, 5),
    }  # fmt: skip
    numbers = {1: "07", 2: "0103", 3: "010307"}
    for rt in RECORD_TYPES:
        recs, expected = [], []
        for n, (venue, code) in enumerate((("05", "TOK"), ("09", "HSN"))):
            race = f"jra:2024-06-01:{code}:07"
            slots, totals = {}, {}
            for pool in TYPE_POOLS[rt]:
                votes = votes_by_pool[pool.code][n]
                totals[pool.code] = votes
                expected.append(pool_row(race, pool.code, votes))
                number = "13" if pool.code == "BRACKET_QUINELLA" else numbers[pool.legs]
                if len(pool.prices) == 2:
                    slots[pool.code] = [r(number, 2.0, 3.0, 1)]
                else:
                    slots[pool.code] = [s(number, 2.0, 1)]
            recs.append(Odds(rt, venue=venue, race="07", slots=slots, totals=totals))
        table = rows_table(layouts, rt, recs)
        got = rows_of(table)
        pool_rows = [x for x in got if x[2] == POOL_ROW]
        assert bag(pool_rows) == bag(expected), rt
        # One pool row per race and sold pool; the slot rows carry no investment.
        assert all(x[6] is None for x in got if x[2] != POOL_ROW), rt
        assert len(got) == 2 * len(pool_rows), rt


ANNOUNCED = [  # (開催年, 開催月日, 発表月日時分, published_utc)
    ("2024", "0601", "06011540", datetime(2024, 6, 1, 6, 40, tzinfo=UTC)),
    ("2024", "0601", "05311830", datetime(2024, 5, 31, 9, 30, tzinfo=UTC)),
    ("2024", "0601", "06010005", datetime(2024, 5, 31, 15, 5, tzinfo=UTC)),
    ("2025", "0105", "12281500", datetime(2024, 12, 28, 6, 0, tzinfo=UTC)),
    ("2016", "1127", "11271623", datetime(2016, 11, 27, 7, 23, tzinfo=UTC)),
    ("2024", "0601", "00000000", None),
]


def test_the_announcement_time_is_published_utc(layouts: dict[str, spec.RecordSpec]) -> None:
    # The expected instants follow from JST (UTC+9), independently of the module.
    for year, mmdd, stamp, utc in ANNOUNCED:
        if utc is not None:
            when_year = int(year) - 1 if int(stamp[:2]) > int(mmdd[:2]) else int(year)
            local = datetime(when_year, int(stamp[:2]), int(stamp[2:4]), int(stamp[4:6]),
                             int(stamp[6:]), tzinfo=JST)  # fmt: skip
            assert local.astimezone(UTC) == utc

    for rt in ("O1", "O5"):
        recs, want = [], {}
        for i, (year, mmdd, stamp, utc) in enumerate(ANNOUNCED):
            if rt == "O5" and utc is not None:  # another record type keeps its own time
                stamp = stamp[:6] + f"{int(stamp[6:]) + 1:02d}"
                utc = utc + timedelta(minutes=1)
            slots = (
                {"WIN": [s("03", 4.4, 1)], "PLACE": [r("03", 1.5, 2.2, 1)]}
                if rt == "O1"
                else {"TRIO": [s("030507", 88.8, 1)]}
            )
            recs.append(
                Odds(rt, year=year, mmdd=mmdd, venue="05", race=f"{i + 1:02d}", announced=stamp,
                     slots=slots, totals={k: 100 + i for k in slots})
            )  # fmt: skip
            day = date(int(year), int(mmdd[:2]), int(mmdd[2:]))
            want[f"jra:{day.isoformat()}:TOK:{i + 1:02d}"] = utc
        table = rows_table(layouts, rt, recs)
        got: dict[str, set] = {}
        for d in table.to_pylist():
            got.setdefault(d["race_id"], set()).add(d["published_utc"])
        # Every row of a record, its pool rows too, carries the record's time.
        assert got == {race: {utc} for race, utc in want.items()}, rt
        per_race = {k: sum(1 for d in table.to_pylist() if d["race_id"] == k) for k in want}
        assert set(per_race.values()) == {4 if rt == "O1" else 2}, rt


# ------------------------------------------------------------------------------------------------
# build_final_odds

BUILD_PRINTED = {  # pool -> (number item as printed, combination)
    "WIN": ("04", "4"), "PLACE": ("11", "11"), "BRACKET_QUINELLA": ("36", "3,6"),
    "QUINELLA": ("0904", "4,9"), "QUINELLA_PLACE": ("1102", "2,11"),
    "FORECAST": ("0904", "9,4"), "TRIO": ("100307", "3,7,10"), "TIERCE": ("100307", "10,3,7"),
}  # fmt: skip


def race_set(year: str, mmdd: str, venue: str, race: str, k: int,
             types: Sequence[str] = RECORD_TYPES) -> tuple[list[Odds], list[FullRow]]:  # fmt: skip
    """One record per type for a race, one slot and a total per pool, the values set by `k`;
    with the rows it gives (none outside JRA)."""
    recs, rows = [], []
    code = VENUES.get(venue)
    day = date(int(year), int(mmdd[:2]), int(mmdd[2:]))
    race_id = f"jra:{day.isoformat()}:{code}:{race}" if code else None
    for rt in types:
        slots, totals = {}, {}
        for i, pool in enumerate(TYPE_POOLS[rt]):
            printed, combination = BUILD_PRINTED[pool.code]
            price = round(2.0 + k + 0.1 * CODES.index(pool.code) + 0.1 * i, 1)
            votes = 1000 * k + CODES.index(pool.code) + 1
            ranged = len(pool.prices) == 2
            slots[pool.code] = [
                r(printed, price, price + 1.5, k) if ranged else s(printed, price, k)
            ]
            totals[pool.code] = votes
            if race_id:
                rows.append(full(row(race_id, pool.code, combination, price,
                                     price + 1.5 if ranged else None, k)))  # fmt: skip
                rows.append(full(pool_row(race_id, pool.code, votes)))
        created = f"{year}{mmdd}"
        recs.append(Odds(rt, year=year, mmdd=mmdd, venue=venue, race=race, created=created,
                         slots=slots, totals=totals))  # fmt: skip
    return recs, rows


def build_inputs() -> dict[int, tuple[list[Odds], list[FullRow]]]:
    """Three years, races out of race_id order, one NAR race; 2001 has only O1 and O2 files."""
    out = {}
    spec_ = {
        2024: [("0602", "05", "11", 1), ("0602", "08", "02", 2), ("0602", "30", "05", 3),
               ("0601", "05", "01", 4)],
        2023: [("1228", "06", "11", 5), ("0105", "06", "01", 6)],
        2001: [("0610", "09", "03", 7)],
    }  # fmt: skip
    for year, races in spec_.items():
        recs, rows = [], []
        types = ("O1", "O2") if year == 2001 else RECORD_TYPES
        for mmdd, venue, race, k in races:
            more, expected = race_set(str(year), mmdd, venue, race, k, types)
            recs += more
            rows += expected
        out[year] = (recs, rows)
    return out


def test_build_writes_final_odds_year_by_year_in_race_order(data: Path, tmp_path: Path) -> None:
    inputs = build_inputs()
    for year, (recs, _) in inputs.items():
        write_year(data, year, recs)
    expected = {year: in_store_order(rows) for year, (_, rows) in inputs.items()}
    assert len(expected[2001]) == 8 and len(expected[2023]) == 32 and len(expected[2024]) == 48

    assert odds.FINAL_ODDS_SCHEMA == EXPECTED_SCHEMA
    report = odds.build_final_odds()
    assert isinstance(report, records.BuildReport)
    path = data.resolve() / "processed" / SOURCE / FILENAME
    assert path.is_file()
    assert pq.read_schema(path) == EXPECTED_SCHEMA
    # One row group per race day (#502), in day order: 2001-06-10, 2023-01-05, 2023-12-28,
    # 2024-06-01, 2024-06-02.
    whole = expected[2001] + expected[2023] + expected[2024]
    days = sorted({r[2] for r in whole})
    assert len(days) == 5
    file = pq.ParquetFile(path)
    assert file.metadata.num_row_groups == len(days)
    for i, day in enumerate(days):
        assert full_rows(file.read_row_group(i)) == [r for r in whole if r[2] == day], day
    assert full_rows(pq.read_table(path)) == whole
    # Column- and row-scoped reads.
    scoped = pq.read_table(path, columns=["race_id", "odds"], filters=[("pool", "=", "TIERCE")])
    assert scoped.column_names == ["race_id", "odds"]
    assert scoped.num_rows == 2 * 5  # five JRA races in 2023-2024, a slot and a pool row each
    assert sorted(p.name for p in path.parent.iterdir()) == [FILENAME]

    # One year only: its two race days, a group each.
    odds.build_final_odds(years=[2023])
    assert pq.ParquetFile(path).metadata.num_row_groups == 2
    assert full_rows(pq.read_table(path)) == expected[2023]

    # Another export and another output folder; the store's file is left alone.
    other = tmp_path / "elsewhere" / "jravan"
    shutil.copytree(data / "raw" / SOURCE, other)
    out = tmp_path / "out"
    odds.build_final_odds(years=[2024], raw=other, out=out)
    assert full_rows(pq.read_table(out / FILENAME)) == expected[2024]
    assert full_rows(pq.read_table(path)) == expected[2023]

    # The command builds the same table and writes the report, counts only.
    result = CliRunner().invoke(app, ["japan", "jravan", "final-odds"])
    assert result.exit_code == 0, result.output
    assert full_rows(pq.read_table(path)) == expected[2001] + expected[2023] + expected[2024]
    written = paths.reports_dir() / "jravan-final-odds.json"
    assert written.is_file()
    parsed = json.loads(written.read_text(encoding="utf-8"))
    assert set(parsed["by_year"]) == {"2001", "2023", "2024"}
    result = CliRunner().invoke(app, ["japan", "jravan", "final-odds", "--year", "2024"])
    assert result.exit_code == 0, result.output
    assert full_rows(pq.read_table(path)) == expected[2024]


def plain(v: object) -> object:
    """A frame cell as plain Python: None for any null, a naive timestamp (a DuckDB DATE) as a
    date, an aware one as a UTC datetime."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):
        return v.date() if v.tzinfo is None else v.tz_convert("UTC").to_pydatetime()
    if isinstance(v, np.generic):
        return plain(v.item())
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def label_rows(frame: pd.DataFrame) -> list[FullRow]:
    """`full_rows` of a frame read back through DuckDB."""
    out = []
    for d in frame.to_dict("records"):
        values = [plain(d[c]) for c in EXPECTED_SCHEMA.names]
        for c in ("odds", "odds_hi", "investment"):
            i = EXPECTED_SCHEMA.names.index(c)
            values[i] = _num(values[i])
        out.append(tuple(values))
    return out


ANNOUNCED_RACE = "jra:2024-06-01:TOK:12"  # announced 2024-06-01 15:40 JST = 06:40 UTC
ANNOUNCED_UTC = datetime(2024, 6, 1, 6, 40, tzinfo=UTC)


def test_final_odds_is_label_only_so_views_raise_and_read_label_returns_it_whole(
    data: Path,
) -> None:
    # Three years of races, and one race whose odds carry an announcement time.
    inputs = build_inputs()
    announced = Odds("O1", year="2024", mmdd="0601", venue="05", race="12", announced="06011540",
                     slots={"WIN": [s("03", 4.4, 1)]}, totals={"WIN": 100})  # fmt: skip
    for year, (recs, _) in inputs.items():
        write_year(data, year, [*recs, announced] if year == 2024 else recs)
    expected = in_store_order(
        [row_ for _, rows in inputs.values() for row_ in rows]
        + [full(row(ANNOUNCED_RACE, "WIN", "3", 4.4, rank=1), published=ANNOUNCED_UTC),
           full(pool_row(ANNOUNCED_RACE, "WIN", 100), published=ANNOUNCED_UTC)]
    )  # fmt: skip
    odds.build_final_odds()
    path = store_path()
    assert pq.read_metadata(path).num_rows == len(expected)

    assert "final_odds" not in schemas.TABLES
    rules = load_availability("japan")
    assert rules["final_odds"].kind == "label_only"
    con = duckdb.connect(":memory:")
    try:
        con.execute(f"CREATE VIEW final_odds AS SELECT * FROM read_parquet('{path}')")
        assert con.execute("SELECT count(*) FROM final_odds").fetchone()[0] == len(expected)
        # before the data, at the announcement, after a race day's post, long after
        times = [
            datetime(2000, 1, 1, tzinfo=UTC),
            ANNOUNCED_UTC,
            datetime(2024, 6, 2, 7, 0, tzinfo=UTC),
            datetime(2030, 1, 1, tzinfo=UTC),
        ]
        keys = pd.DataFrame(
            {
                "race_id": [ANNOUNCED_RACE, "jra:2023-12-28:NAK:11"] * 2,
                "pool": ["WIN", "TIERCE"] * 2,
                "combination": ["3", POOL_ROW] * 2,
                "t_utc": pd.to_datetime(times, utc=True),
            }
        )
        for t in times:
            with pytest.raises(LabelOnlyTable):
                read_asof(con, "final_odds", t, rules=rules)
        with pytest.raises(LabelOnlyTable):
            asof_join(keys, "final_odds", on=["race_id", "pool", "combination"], rules=rules,
                      con=con)  # fmt: skip
        with pytest.raises(LabelOnlyTable):
            asof_join(keys[["race_id", "t_utc"]], "final_odds", on=["race_id"], rules=rules,
                      con=con)  # fmt: skip

        # As a label the table is whole: every row and every column, the odds included.
        labels = read_label(con, "final_odds", rules=rules)
        assert list(labels.columns) == EXPECTED_SCHEMA.names
        assert in_store_order(label_rows(labels)) == expected
    finally:
        con.close()


def test_the_latest_version_per_race_is_kept(data: Path) -> None:
    def kyo(rt: str, race: str, kubun: str, created: str, slots: dict, totals: dict) -> Odds:
        return Odds(rt, mmdd="0608", venue="08", race=race, kubun=kubun, created=created,
                    slots=slots, totals=totals)  # fmt: skip

    write_year(
        data,
        2024,
        [
            kyo("O1", "10", "4", "20240610", {"WIN": [s("06", 4.8, 2)]}, {"WIN": 100}),
            kyo("O1", "11", "4", "20240608", {"WIN": [s("13", 9.9, 1)]}, {"WIN": 200}),
            kyo("O1", "10", "3", "20240608", {"WIN": [s("06", 4.7, 2), s("11", 2.6, 1)]},
                {"WIN": 90}),
            kyo("O1", "12", "4", "20240610", {"WIN": [s("01", 2.2, 1)]}, {"WIN": 300}),
            kyo("O1", "11", "0", "20240611", {}, {}),
            # Same day and race number at another venue: another race.
            Odds("O1", mmdd="0608", venue="05", race="10", kubun="3", created="20240608",
                 slots={"WIN": [s("09", 3.3, 1)]}, totals={"WIN": 400}),
            # Another record type of the same races keeps its own version.
            kyo("O2", "10", "3", "20240608", {"QUINELLA": [s("0611", 7.1, 1)]},
                {"QUINELLA": 500}),
            kyo("O2", "11", "4", "20240608", {"QUINELLA": [s("1301", 5.5, 1)]},
                {"QUINELLA": 600}),
        ],
    )  # fmt: skip
    odds.build_final_odds()
    k10, k11, k12 = (f"jra:2024-06-08:KYO:{n}" for n in ("10", "11", "12"))
    t10 = "jra:2024-06-08:TOK:10"
    expected = [
        full(row(k10, "WIN", "6", 4.8, rank=2), kubun="4"),
        full(pool_row(k10, "WIN", 100), kubun="4"),
        full(row(k10, "QUINELLA", "6,11", 7.1, rank=1), kubun="3"),
        full(pool_row(k10, "QUINELLA", 500), kubun="3"),
        full(row(k11, "QUINELLA", "1,13", 5.5, rank=1), kubun="4"),
        full(pool_row(k11, "QUINELLA", 600), kubun="4"),
        full(row(k12, "WIN", "1", 2.2, rank=1), kubun="4"),
        full(pool_row(k12, "WIN", 300), kubun="4"),
        full(row(t10, "WIN", "9", 3.3, rank=1), kubun="3"),
        full(pool_row(t10, "WIN", 400), kubun="3"),
    ]
    assert full_rows(pq.read_table(store_path())) == in_store_order(expected)


def test_two_builds_write_identical_bytes(data: Path) -> None:
    inputs = build_inputs()
    for year, (recs, _) in inputs.items():
        write_year(data, year, recs)
    first_report = odds.build_final_odds()
    path = store_path()
    first = path.read_bytes()

    # The same inputs rewritten (new mtimes): the same bytes and the same report.
    for year, (recs, _) in inputs.items():
        write_year(data, year, recs)
    second_report = odds.build_final_odds()
    assert path.read_bytes() == first
    assert second_report.to_json() == first_report.to_json()
    assert sorted(p.name for p in path.parent.iterdir()) == [FILENAME]


# ------------------------------------------------------------------------------------------------
# The real export (local: needs $HORSETRACK_DATA_DIR with the export and the workbook)

REAL_YEAR = 2024
REAL_RECORDS = 60  # the first records of each type's year file (an O6 record holds 4,896 slots)
HORSE = r"(?:[1-9]|1[0-8])"


@pytest.mark.local
def test_a_real_year_of_each_odds_type_maps(tmp_path: Path) -> None:
    export, workbook = wb.real_export_or_skip()
    raw = tmp_path / "export"
    cols: dict[str, dict[str, np.ndarray]] = {}
    for rt in RECORD_TYPES:
        source = export / "stored" / "RACE" / rt / f"{REAL_YEAR}.jvd"
        if not source.is_file():
            pytest.skip(f"no {rt} file for {REAL_YEAR}")
        with open(source, "rb") as handle:
            lines = [line for _, line in zip(range(REAL_RECORDS), handle, strict=False)]
        part = b"".join(lines)
        target = raw / "stored" / "RACE" / rt / f"{REAL_YEAR}.jvd"
        target.parent.mkdir(parents=True)
        target.write_bytes(part)
        cols[rt] = records.decode(part, spec.record_spec(rt, workbook))

    out = tmp_path / "out"
    odds.build_final_odds(years=[REAL_YEAR], raw=raw, out=out)
    path = out / FILENAME
    assert pq.read_schema(path) == EXPECTED_SCHEMA
    table = pq.read_table(path)
    # One row group per race day (#502): each group's statistics record one day, days ascending.
    meta = pq.ParquetFile(path).metadata
    at = EXPECTED_SCHEMA.get_field_index("race_date")
    group_days = []
    for g in range(meta.num_row_groups):
        stats = meta.row_group(g).column(at).statistics
        assert stats is not None and stats.has_min_max, f"row group {g} has no race_date min/max"
        assert stats.min == stats.max, f"row group {g} holds {stats.min} to {stats.max}"
        group_days.append(stats.min)
    assert group_days == sorted(set(table.column("race_date").to_pylist()))
    got = {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}
    assert len(got["race_id"]) > 0
    assert set(got["pool"]) == set(CODES)
    assert (got["jurisdiction"] == "jra").all()
    venues = "|".join(VENUES.values())
    race_re = re.compile(rf"jra:{REAL_YEAR}-\d\d-\d\d:(?:{venues}):\d\d")
    assert all(race_re.fullmatch(x) for x in set(got["race_id"]))
    order = list(zip(got["race_id"], got["pool"], got["combination"], strict=True))
    assert order == sorted(order)
    assert len(set(order)) == len(order)

    for code, pool in POOLS.items():
        c = cols[pool.record_type]
        jra = np.isin(c["競馬場コード"], [v.encode() for v in VENUES])
        race_keys_ = np.char.add(np.char.add(c["開催月日"], c["競馬場コード"]), c["レース番号"])
        assert len(np.unique(race_keys_[jra])) == int(jra.sum()), "one version per race expected"
        sold = jra & (c[pool.sale] == b"7")
        number = c[f"{pool.group}/{pool.number}"].reshape(len(jra), -1)
        filled = sold[:, None] & (np.char.strip(number) != b"")
        low = records.integer(c[f"{pool.group}/{pool.prices[0]}"].reshape(len(jra), -1))
        rank = records.integer(c[f"{pool.group}/人気順"].reshape(len(jra), -1))
        totals = records.integer(c[pool.total])

        mine = got["pool"] == code
        slot = mine & (got["combination"] != POOL_ROW)
        pools_ = mine & (got["combination"] == POOL_ROW)
        assert int(slot.sum()) == int(filled.sum()), code
        assert int(pools_.sum()) == int(sold.sum()), code
        inv = got["investment"][pools_].astype(float)
        assert np.isclose(inv.sum(), np.nansum(totals[sold]) * 100), code
        assert np.isnan(got["investment"][slot].astype(float)).all(), code

        price = got["odds"][slot].astype(float)
        positive = filled & (low > 0)
        assert int((~np.isnan(price)).sum()) == int(positive.sum()), code
        assert np.isclose(np.nansum(price), low[positive].sum() / 10), code
        status = got["status_raw"][slot]
        assert sum(x is not None for x in status) == int((filled & ~(low > 0)).sum()), code
        ranks = got["fav_rank"][slot].astype(float)
        assert int((~np.isnan(ranks)).sum()) == int((filled & (rank > 0)).sum()), code

        high = got["odds_hi"][slot].astype(float)
        if len(pool.prices) == 2:
            hi_raw = records.integer(c[f"{pool.group}/{pool.prices[1]}"].reshape(len(jra), -1))
            assert int((~np.isnan(high)).sum()) == int((filled & (hi_raw > 0)).sum()), code
            both = ~np.isnan(high) & ~np.isnan(price)
            assert (high[both] >= price[both]).all(), code
        else:
            assert np.isnan(high).all(), code

        if (c["発表月日時分"] == b"00000000").all():
            assert all(x is None or x != x for x in got["published_utc"][mine]), code
        kubun = set(records.text(c["データ区分"])[sold])
        assert set(got["data_kubun"][mine]) <= kubun, code

        bad = 0
        token = r"[1-8]" if code == "BRACKET_QUINELLA" else HORSE
        for combination in set(got["combination"][slot]):
            parts = combination.split(",")
            if len(parts) != pool.legs or not all(re.fullmatch(token, p) for p in parts):
                bad += 1
                continue
            numbers = [int(p) for p in parts]
            if code == "BRACKET_QUINELLA":
                bad += numbers != sorted(numbers)
            elif code in ("QUINELLA", "QUINELLA_PLACE", "TRIO"):
                bad += numbers != sorted(set(numbers))
            else:
                bad += len(set(numbers)) != len(numbers)
        assert bad == 0, f"{code}: {bad} malformed combinations"
