"""JV-Data 0B41/0B42 time-series odds into the jravan `odds_timeseries` day files (ticket #369).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jravan.timeseries`:
- `build_day(race_date: datetime.date, *, raw=None, out=None) -> records.BuildReport` reads
  `<raw>/timeseries/0B41/<YYYY>/<YYYYMMDD>.jvd` (O1 records) and `0B42/...` (O2 records), either
  of which may be missing, with the layouts `spec.record_spec('O1'|'O2')`, decodes them with
  `odds.odds_rows` and writes `processed/jravan/odds_timeseries/<YYYY-MM-DD>.parquet` (`raw`
  default `paths.raw_dir('jravan')`) in exactly `logger.compact.SCHEMA`.
- Columns: `race_id`, `venue` (the short code), `race_no`, `pool`, `combination` as `odds_rows`
  gives them, pool rows `__pool__` with `investment` in yen; `snapshot_utc` the record's
  発表月日時分 as UTC (JST is UTC+9), an evening-before snapshot in the race day's file;
  `fetched_utc` null; `post_utc` the race's `post_time_local` (JST, on the race date) from
  `processed/jravan/races.parquet`, null without a race row or a post time; `minutes_to_post`
  (`post_utc` - `snapshot_utc`) in minutes; `odds`, `odds_hi`, `status_raw`; `is_final_capture`
  on the latest `snapshot_utc` per (race_id, pool, combination), pool rows included; `dividend`,
  `is_refund` a left join on (race_id, pool, combination) to `processed/jravan/dividends.parquet`,
  null without a match or without the table. The same (race_id, pool, combination, snapshot_utc)
  twice is one row, the later record's.
- Rows sorted on content (here: by race_id, pool, combination, snapshot_utc), the same bytes for
  the same inputs whatever their record order; the folder holds the day files and `_build.json`
  only. A day whose inputs and workbook are unchanged is not rewritten; a day whose file is gone
  is written again.
- `build_all(*, years=None, raw=None, out=None) -> records.BuildReport`: every day of the
  requested years found under either family; a day that fails is in `failures` (naming the day)
  and the other days are written.
- `horsetrack japan jravan timeseries [--year YYYY ...] [--day YYYY-MM-DD]`: the same, the report
  `<reports dir>/jravan-timeseries.json` through `records.write_build_report`; exit 1 when a day
  failed, else 0.

Not pinned: the report's count names, the `_build.json` content, how `out` lays out its files.

Every layout here is made up in the workbook's shape (`jravan_workbook.write_workbook`): the item
names are the workbook's, the widths, repeats and so every position are invented. Every record and
value is invented. Expected rows follow from the fixtures by hand (JST = UTC+9), never from the
module under test.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.core.store.asof import load_availability, read_asof
from horsetrack.core.store.build import arrow_schema
from horsetrack.countries.japan.logger import compact

# isort: split
# (its own block: the sibling helper module, then the module this ticket creates)
import jravan_workbook as wb

from horsetrack.countries.japan.jravan import records, timeseries

SOURCE = "jravan"
UTC = timezone.utc
JST = timezone(timedelta(hours=9))
POOL_ROW = "__pool__"
REPORT = "jravan-timeseries.json"
BUILD_FILE = "_build.json"
VENUES = {"05": "TOK", "06": "NAK", "08": "KYO", "09": "HSN"}  # number -> config/jra.yaml code
FAMILY = {"O1": "0B41", "O2": "0B42"}

# ------------------------------------------------------------------------------------------------
# Made-up O1/O2 layouts: the workbook's item names, invented widths and repeats

TOTAL = 11  # bytes of a vote total (invented)


@dataclass(frozen=True)
class PoolDef:
    code: str
    record_type: str
    sale: str
    group: str
    number: str
    number_width: int
    prices: tuple[str, ...]
    price_width: int
    rank_width: int
    total: str
    repeat: int


LOW_HIGH = ("最低オッズ", "最高オッズ")
POOLS = {
    p.code: p
    for p in (
        PoolDef("WIN", "O1", "発売フラグ単勝", "単勝オッズ", "馬番", 2, ("オッズ",), 4, 2,
                "単勝票数合計", 8),
        PoolDef("PLACE", "O1", "発売フラグ複勝", "複勝オッズ", "馬番", 2, LOW_HIGH, 4, 2,
                "複勝票数合計", 8),
        PoolDef("BRACKET_QUINELLA", "O1", "発売フラグ枠連", "枠連オッズ", "組番", 2, ("オッズ",),
                5, 2, "枠連票数合計", 6),
        PoolDef("QUINELLA", "O2", "発売フラグ馬連", "馬連オッズ", "組番", 4, ("オッズ",), 6, 3,
                "馬連票数合計", 10),
    )
}  # fmt: skip
TYPE_POOLS = {rt: [p for p in POOLS.values() if p.record_type == rt] for rt in ("O1", "O2")}
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
    return wb.Section(number, f"時系列オッズ{rt[1]}（架空）", pos - 1, tuple(rows))


SECTIONS = {rt: section(rt, i + 2) for i, rt in enumerate(("O1", "O2"))}


def write_ts_workbook(path: Path, *, decoys: int = 1) -> Path:
    """The made-up workbook; `decoys` other sections before O1/O2 (changes the bytes only)."""
    others = [
        wb.simple_section(10 + i, f"X{chr(ord('A') + i)}", f"架空その他{i}", [("架空項目", 6 + i)])
        for i in range(decoys)
    ]
    return wb.write_workbook(path, [*others, *SECTIONS.values()], [])


# ------------------------------------------------------------------------------------------------
# Made-up time-series records

Price = float | str  # a decimal price, or a raw text at the field's width ("----")


def w(printed: str, combination: str, price: Price) -> tuple:
    """A one-price slot (WIN, BRACKET_QUINELLA, QUINELLA): as printed, the combination it is."""
    return (printed, combination, price)


def p(printed: str, combination: str, low: Price, high: float) -> tuple:
    """A PLACE slot."""
    return (printed, combination, low, high)


@dataclass
class Snap:
    """One made-up O1 or O2 record: a race's board as announced at `at` (JST wall clock)."""

    rt: str
    at: datetime  # 発表月日時分, JST
    race_day: date = date(2024, 6, 8)
    venue: str = "05"
    race: str = "11"
    kai: str = "03"
    nichi: str = "02"
    kubun: str = "1"
    slots: Mapping[str, Sequence[tuple]] = field(default_factory=dict)
    totals: Mapping[str, int] = field(default_factory=dict)

    @property
    def race_id(self) -> str:
        return f"jra:{self.race_day.isoformat()}:{VENUES[self.venue]}:{self.race}"

    @property
    def snapshot_utc(self) -> datetime:
        return self.at.astimezone(UTC)

    def record(self) -> bytes:
        assert self.at.tzinfo is JST
        mine = {q.code for q in TYPE_POOLS[self.rt]}
        assert set(self.slots) <= mine and set(self.totals) == set(self.slots), self
        head = dict(
            zip(
                [name for name, _ in HEAD],
                (self.rt, self.kubun, self.at.strftime("%Y%m%d"), f"{self.race_day.year:04d}",
                 self.race_day.strftime("%m%d"), self.venue, self.kai, self.nichi, self.race,
                 self.at.strftime("%m%d%H%M"), "16", "16"),
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
                    if k >= len(slots):
                        out.append(" " * f.width)
                        continue
                    printed, _, *prices = slots[k]
                    assert len(prices) == len(pool.prices), (f.name, slots[k])
                    out.append(fit(printed, pool.number_width))
                    for price in prices:
                        if isinstance(price, str):
                            out.append(fit(price, pool.price_width))
                        else:
                            out.append(f"{round(price * 10):0{pool.price_width}d}")
                    out.append(f"{k + 1:0{pool.rank_width}d}")
            elif f.name in SALE_POOL:
                out.append("7" if SALE_POOL[f.name] in self.slots else "0")
            elif f.name in TOTAL_POOL:
                votes = self.totals.get(TOTAL_POOL[f.name])
                out.append(" " * TOTAL if votes is None else f"{votes:0{TOTAL}d}")
            elif f.name == "複勝着払キー":
                out.append("3" if "PLACE" in self.slots else "0")
            elif f.name == "予備":
                out.append(" " * f.length)
            else:
                out.append(fit(head[f.name], f.length))
        body = "".join(out).encode("cp932")
        assert len(body) == SECTIONS[self.rt].length - 2
        return body + b"\r\n"


def fit(value: str, length: int) -> str:
    if len(value) != length:
        raise ValueError(f"{value!r} is not {length} wide")
    return value


def o1(
    at: datetime,
    *,
    win: Sequence[tuple] = (),
    place: Sequence[tuple] = (),
    bracket: Sequence[tuple] = (),
    votes: tuple[int, int, int] = (0, 0, 0),
    **kw,
) -> Snap:
    slots, totals = {}, {}
    for code, given, total in zip(("WIN", "PLACE", "BRACKET_QUINELLA"), (win, place, bracket),
                                  votes, strict=True):  # fmt: skip
        if given:
            slots[code], totals[code] = list(given), total
    return Snap("O1", at, slots=slots, totals=totals, **kw)


def o2(at: datetime, quinella: Sequence[tuple], votes: int, **kw) -> Snap:
    return Snap("O2", at, slots={"QUINELLA": list(quinella)}, totals={"QUINELLA": votes}, **kw)


def jst(y: int, m: int, d: int, hh: int, mm: int) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=JST)


def utc(y: int, m: int, d: int, hh: int, mm: int, us: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, 0, us, tzinfo=UTC)


# ------------------------------------------------------------------------------------------------
# Expected rows, from the fixtures

COLUMNS = compact.SCHEMA.names
FLOATS = ("minutes_to_post", "odds", "odds_hi", "investment", "dividend")


def logged_rows(x: Snap) -> list[dict]:
    """The rows one record gives: its slots, then a pool row per sold pool."""
    out = []
    venue, race_no = VENUES[x.venue], int(x.race)
    base = {"race_id": x.race_id, "venue": venue, "race_no": race_no,
            "snapshot_utc": x.snapshot_utc, "fetched_utc": None}  # fmt: skip
    for pool in TYPE_POOLS[x.rt]:
        if pool.code not in x.slots:
            continue
        for _, combination, *prices in x.slots[pool.code]:
            low = prices[0]
            high = prices[1] if len(prices) == 2 else None
            out.append(
                {**base, "pool": pool.code, "combination": combination,
                 "odds": None if isinstance(low, str) else float(low),
                 "odds_hi": None if high is None else float(high),
                 "investment": None, "status_raw": low if isinstance(low, str) else None}
            )  # fmt: skip
        out.append(
            {**base, "pool": pool.code, "combination": POOL_ROW, "odds": None, "odds_hi": None,
             "investment": float(x.totals[pool.code] * 100), "status_raw": None}
        )  # fmt: skip
    return out


def expected(
    snaps: Sequence[Snap],
    *,
    posts: Mapping[str, datetime] | None = None,
    dividends: Mapping[tuple, tuple[float, bool]] | None = None,
) -> list[tuple]:
    """The day file's rows, in SCHEMA order, sorted on (race_id, pool, combination, snapshot);
    `snaps` in file order (the later record wins a repeated key)."""
    posts, dividends = posts or {}, dividends or {}
    rows: dict[tuple, dict] = {}
    for x in snaps:
        for r in logged_rows(x):
            rows[(r["race_id"], r["pool"], r["combination"], r["snapshot_utc"])] = r
    latest: dict[tuple, datetime] = {}
    for race, pool, combination, snapshot in rows:
        k = (race, pool, combination)
        latest[k] = max(latest.get(k, snapshot), snapshot)
    out = []
    for (race, pool, combination, snapshot), r in sorted(rows.items(), key=lambda kv: kv[0]):
        post = posts.get(race)
        r = dict(r)
        r["post_utc"] = post
        r["minutes_to_post"] = None if post is None else (post - snapshot) / timedelta(minutes=1)
        r["is_final_capture"] = snapshot == latest[(race, pool, combination)]
        r["dividend"], r["is_refund"] = dividends.get((race, pool, combination), (None, None))
        out.append(normal(r))
    return out


def normal(r: Mapping) -> tuple:
    values = []
    for name in COLUMNS:
        v = r[name]
        if name in FLOATS and v is not None:
            v = round(float(v), 6)
        values.append(v)
    return tuple(values)


def col(name: str) -> int:
    return COLUMNS.index(name)


# ------------------------------------------------------------------------------------------------
# Data-dir helpers


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data dir with the made-up workbook and an empty time-series export."""
    return make_data(tmp_path / "data", monkeypatch)


def make_data(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(root))
    write_ts_workbook(root / "specs" / "jravan" / wb.WORKBOOK_NAME)
    (root / "raw" / SOURCE / "timeseries").mkdir(parents=True)
    return root


def jvd_path(raw: Path, family: str, day: date) -> Path:
    return raw / "timeseries" / family / f"{day.year:04d}" / f"{day:%Y%m%d}.jvd"


def write_day(day: date, snaps: Sequence[Snap], *, raw: Path | None = None) -> None:
    """One 0B41 file (the O1 records) and one 0B42 file (the O2), each only when it has records,
    records in the given order."""
    raw = raw if raw is not None else paths.raw_dir(SOURCE)
    for rt, family in FAMILY.items():
        mine = [x.record() for x in snaps if x.rt == rt]
        if mine:
            wb.write_jvd(jvd_path(raw, family, day), mine)


def day_file(day: date) -> Path:
    return paths.processed_dir(SOURCE) / "odds_timeseries" / f"{day.isoformat()}.parquet"


def read_rows(path: Path) -> list[tuple]:
    return [normal(r) for r in pq.read_table(path).to_pylist()]


def sort_key(row: tuple) -> tuple:
    return tuple(row[col(c)] for c in ("race_id", "pool", "combination", "snapshot_utc"))


def post_at(day: date, hhmm: str) -> datetime:
    """`HH:MM` JST on `day`, as UTC."""
    hh, mm = (int(x) for x in hhmm.split(":"))
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=JST).astimezone(UTC)


def write_races(rows: Sequence[tuple[str, str | None]]) -> Path:
    """A synthetic `processed/jravan/races.parquet` in the store's races schema: (race_id,
    post_time_local) per race, the other columns null."""
    schema = arrow_schema("races")
    n = len(rows)
    given: dict[str, list] = {
        "jurisdiction": ["jra"] * n,
        "race_id": [r for r, _ in rows],
        "meeting_id": [r.rsplit(":", 1)[0] for r, _ in rows],
        "race_date": [date.fromisoformat(r.split(":")[1]) for r, _ in rows],
        "venue": [r.split(":")[2] for r, _ in rows],
        "race_no": [int(r.split(":")[3]) for r, _ in rows],
        "post_time_local": [t for _, t in rows],
    }
    columns = [
        pa.array(given[f.name], f.type) if f.name in given else pa.nulls(n, f.type) for f in schema
    ]
    path = paths.processed_dir(SOURCE) / "races.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns, schema=schema), path)
    return path


def write_dividends(rows: Sequence[tuple[str, str, str, float, bool]]) -> Path:
    """A synthetic `processed/jravan/dividends.parquet` in the store's dividends schema."""
    schema = arrow_schema("dividends")
    given = {
        "jurisdiction": ["jra"] * len(rows),
        "race_id": [r[0] for r in rows],
        "pool": [r[1] for r in rows],
        "combination": [r[2] for r in rows],
        "dividend": [r[3] for r in rows],
        "is_refund": [r[4] for r in rows],
    }
    columns = [
        pa.array(given[f.name], f.type) if f.name in given else pa.nulls(len(rows), f.type)
        for f in schema
    ]
    path = paths.processed_dir(SOURCE) / "dividends.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns, schema=schema), path)
    return path


# ------------------------------------------------------------------------------------------------
# A made-up race day: two venues, three races, both families, snapshots from the evening before

DAY = date(2024, 6, 8)
TOK11 = "jra:2024-06-08:TOK:11"
TOK01 = "jra:2024-06-08:TOK:01"
KYO11 = "jra:2024-06-08:KYO:11"


def day_snaps() -> list[Snap]:
    """Records in a file order that is not time order; prices change from snapshot to snapshot."""
    tok11 = dict(venue="05", race="11")
    tok01 = dict(venue="05", race="01", kai="03", nichi="02")
    kyo11 = dict(venue="08", race="11", kai="02", nichi="04")
    return [
        # TOK 11: the evening before, the morning, T-10, and an announcement after the post.
        o1(
            jst(2024, 6, 7, 18, 30),
            win=[w("01", "1", 5.6), w("04", "4", 3.5), w("12", "12", 99.9)],
            place=[p("01", "1", 1.8, 2.6), p("04", "4", 1.2, 1.5)],
            bracket=[w("12", "1,2", 15.2), w("44", "4,4", 88.8)],
            votes=(1200, 800, 300),
            **tok11,
        ),
        o1(
            jst(2024, 6, 8, 15, 30),
            win=[w("01", "1", 4.9), w("04", "4", 3.9), w("12", "12", "----")],
            place=[p("01", "1", 1.7, 2.4), p("04", "4", 1.3, 1.6)],
            bracket=[w("12", "1,2", 14.0), w("44", "4,4", 91.0)],
            votes=(45678, 23456, 9876),
            **tok11,
        ),
        o1(
            jst(2024, 6, 8, 9, 15),
            win=[w("01", "1", 5.1), w("04", "4", 3.7), w("12", "12", 120.5)],
            place=[p("01", "1", 1.9, 2.8), p("04", "4", 1.2, 1.4)],
            bracket=[w("12", "1,2", 16.6), w("44", "4,4", 80.1)],
            votes=(9000, 5000, 2000),
            **tok11,
        ),
        o1(
            jst(2024, 6, 8, 15, 42),
            win=[w("01", "1", 4.8), w("04", "4", 4.0)],
            place=[p("01", "1", 1.7, 2.3), p("04", "4", 1.3, 1.7)],
            bracket=[w("12", "1,2", 13.9)],
            votes=(50001, 25002, 10003),
            **tok11,
        ),
        o2(
            jst(2024, 6, 7, 18, 30), [w("0104", "1,4", 7.7), w("1204", "4,12", 345.6)], 700, **tok11
        ),
        o2(
            jst(2024, 6, 8, 15, 30),
            [w("0104", "1,4", 6.5), w("1204", "4,12", 410.2), w("0112", "1,12", 999.9)],
            88888,
            **tok11,
        ),  # fmt: skip
        # TOK 1: one morning board.
        o1(
            jst(2024, 6, 8, 9, 50),
            win=[w("03", "3", 2.2), w("16", "16", 45.6)],
            place=[p("03", "3", 1.1, 1.3)],
            votes=(3100, 1500, 0),
            **tok01,
        ),
        o2(jst(2024, 6, 8, 9, 50), [w("1603", "3,16", 55.5)], 1234, **tok01),
        # KYO 11: the same race number at another venue, its own boards.
        o1(
            jst(2024, 6, 8, 15, 20),
            win=[w("07", "7", 12.3), w("02", "2", 2.6)],
            bracket=[w("36", "3,6", 30.0)],
            votes=(30303, 0, 4040),
            **kyo11,
        ),
        o1(
            jst(2024, 6, 7, 19, 5),
            win=[w("07", "7", 10.0), w("02", "2", 3.0)],
            bracket=[w("36", "3,6", 25.5)],
            votes=(1111, 0, 222),
            **kyo11,
        ),
        o2(
            jst(2024, 6, 8, 15, 20), [w("0702", "2,7", 18.8), w("0307", "3,7", 44.4)], 5555, **kyo11
        ),
    ]


# ------------------------------------------------------------------------------------------------
# The named tests


def test_a_day_of_snapshots_becomes_odds_timeseries_rows_in_the_compact_schema(data: Path) -> None:
    snaps = day_snaps()
    write_day(DAY, snaps)
    report = timeseries.build_day(DAY)
    assert isinstance(report, records.BuildReport)
    assert report.failures == []

    path = day_file(DAY)
    assert path.is_file()
    assert pq.read_schema(path) == compact.SCHEMA
    got = read_rows(path)
    want = expected(snaps)
    assert got == want
    assert got == sorted(got, key=sort_key)
    assert len({sort_key(r) for r in got}) == len(got)

    # A spread the fixture fixes: every pool, both venues, slot and pool rows per snapshot.
    pools = {(r[col("race_id")], r[col("pool")]) for r in got}
    assert pools == {
        (TOK11, "WIN"), (TOK11, "PLACE"), (TOK11, "BRACKET_QUINELLA"), (TOK11, "QUINELLA"),
        (TOK01, "WIN"), (TOK01, "PLACE"), (TOK01, "QUINELLA"),
        (KYO11, "WIN"), (KYO11, "BRACKET_QUINELLA"), (KYO11, "QUINELLA"),
    }  # fmt: skip
    tok11_win = [r for r in got if r[col("race_id")] == TOK11 and r[col("pool")] == "WIN"]
    assert len(tok11_win) == 3 + 3 + 3 + 2 + 4  # slots per snapshot + one pool row each
    pool_rows = [r for r in tok11_win if r[col("combination")] == POOL_ROW]
    assert [r[col("investment")] for r in pool_rows] == [120000.0, 900000.0, 4567800.0, 5000100.0]
    cancelled = [r for r in tok11_win if r[col("status_raw")] is not None]
    assert [(r[col("combination")], r[col("odds")], r[col("status_raw")]) for r in cancelled] == [
        ("12", None, "----")
    ]
    assert {r[col("venue")] for r in got} == {"TOK", "KYO"}
    assert {r[col("race_no")] for r in got} == {1, 11}
    # No races or dividends table here: those columns are null, the schema unchanged.
    for name in ("post_utc", "minutes_to_post", "dividend", "is_refund"):
        assert {r[col(name)] for r in got} == {None}, name
    assert sorted(p.name for p in path.parent.iterdir()) == sorted([BUILD_FILE, path.name])


def test_the_board_at_t_holds_the_snapshots_announced_by_t(data: Path) -> None:
    snaps = day_snaps()
    write_day(DAY, snaps)
    timeseries.build_day(DAY)
    rows = expected(snaps)
    keys = [
        (r[col("race_id")], r[col("pool")], r[col("combination")], r[col("snapshot_utc")])
        for r in rows
    ]
    instants = sorted({k[3] for k in keys})
    assert len(instants) == 7

    con = duckdb.connect(":memory:")
    con.execute(f"CREATE VIEW odds_timeseries AS SELECT * FROM read_parquet('{day_file(DAY)}')")
    rules = load_availability("japan")

    def board(t: datetime) -> list[tuple]:
        view = read_asof(con, "odds_timeseries", t, rules=rules)
        for label in ("dividend", "is_refund", "is_final_capture"):
            assert label not in view.columns
        return sorted(
            (race, pool, combination, pd.Timestamp(snap).to_pydatetime())
            for race, pool, combination, snap in zip(
                view["race_id"],
                view["pool"],
                view["combination"],
                view["snapshot_utc"],
                strict=True,
            )
        )

    assert board(instants[0] - timedelta(microseconds=1)) == []
    for instant in instants:
        assert board(instant) == sorted(k for k in keys if k[3] <= instant), instant
        before = instant - timedelta(microseconds=1)
        assert board(before) == sorted(k for k in keys if k[3] < instant), before
    assert board(utc(2030, 1, 1, 0, 0)) == sorted(keys)


SNAPSHOT_CASES = [  # (race day, announced JST, the UTC instant written out by hand)
    (date(2024, 6, 1), jst(2024, 5, 31, 18, 30), utc(2024, 5, 31, 9, 30)),  # the month before
    (date(2024, 6, 1), jst(2024, 6, 1, 0, 0), utc(2024, 5, 31, 15, 0)),  # UTC is the day before
    (date(2024, 6, 1), jst(2024, 6, 1, 8, 59), utc(2024, 5, 31, 23, 59)),
    (date(2024, 6, 1), jst(2024, 6, 1, 9, 0), utc(2024, 6, 1, 0, 0)),
    (date(2024, 6, 1), jst(2024, 6, 1, 15, 39), utc(2024, 6, 1, 6, 39)),
    (date(2024, 6, 1), jst(2024, 6, 1, 23, 59), utc(2024, 6, 1, 14, 59)),
    (date(2019, 12, 28), jst(2019, 12, 27, 19, 1), utc(2019, 12, 27, 10, 1)),
    (date(2019, 12, 28), jst(2019, 12, 28, 16, 25), utc(2019, 12, 28, 7, 25)),
]


def test_snapshot_utc_is_the_announcement_minute_and_fetched_utc_is_null(data: Path) -> None:
    days: dict[date, list[Snap]] = {}
    want: dict[date, dict[float, datetime]] = {}
    for i, (day, at, instant) in enumerate(SNAPSHOT_CASES):
        price = 2.0 + i  # identifies the snapshot
        venue = "06" if day.year == 2019 else "09"
        days.setdefault(day, []).append(
            o1(at, win=[w("05", "5", price)], place=[p("05", "5", 1.1 + i, 1.5 + i)],
               votes=(100 + i, 50 + i, 0), race_day=day, venue=venue, race="09")
        )  # fmt: skip
        days[day].append(
            o2(at, [w("0105", "1,5", 10.0 + i)], 70 + i, race_day=day, venue=venue, race="09")
        )
        want.setdefault(day, {})[price] = instant
    for day, snaps in days.items():
        write_day(day, snaps)
        timeseries.build_day(day)
        rows = pq.read_table(day_file(day)).to_pylist()
        assert len(rows) == 6 * len(want[day])  # WIN, PLACE, QUINELLA: a slot and a pool row each
        assert all(r["fetched_utc"] is None for r in rows)
        win = {r["odds"]: r["snapshot_utc"] for r in rows if r["pool"] == "WIN" and r["odds"]}
        assert win == want[day]
        # Every row of a record, its pool rows and the O2 record's too, carries the record's time.
        assert {r["snapshot_utc"] for r in rows} == set(want[day].values())
        per_instant = {}
        for r in rows:
            per_instant[r["snapshot_utc"]] = per_instant.get(r["snapshot_utc"], 0) + 1
        assert set(per_instant.values()) == {6}


def test_an_evening_before_snapshot_belongs_to_the_races_days_file(data: Path) -> None:
    race_day = date(2026, 11, 28)
    snaps = [
        o1(jst(2026, 11, 27, 18, 31), win=[w("03", "3", 6.6)], votes=(4321, 0, 0),
           race_day=race_day, venue="05", race="10", kai="05", nichi="07"),
        o1(jst(2026, 11, 28, 14, 55), win=[w("03", "3", 5.9)], votes=(65432, 0, 0),
           race_day=race_day, venue="05", race="10", kai="05", nichi="07"),
        o2(jst(2026, 11, 27, 18, 31), [w("0803", "3,8", 21.0)], 999, race_day=race_day,
           venue="05", race="10", kai="05", nichi="07"),
    ]  # fmt: skip
    write_day(race_day, snaps)
    timeseries.build_day(race_day)
    folder = paths.processed_dir(SOURCE) / "odds_timeseries"
    assert sorted(p.name for p in folder.iterdir()) == sorted([BUILD_FILE, "2026-11-28.parquet"])
    rows = pq.read_table(folder / "2026-11-28.parquet").to_pylist()
    evening = utc(2026, 11, 27, 9, 31)
    by = {(r["pool"], r["combination"], r["snapshot_utc"]): r for r in rows}
    assert set(by) == {
        ("WIN", "3", evening), ("WIN", POOL_ROW, evening),
        ("WIN", "3", utc(2026, 11, 28, 5, 55)), ("WIN", POOL_ROW, utc(2026, 11, 28, 5, 55)),
        ("QUINELLA", "3,8", evening), ("QUINELLA", POOL_ROW, evening),
    }  # fmt: skip
    assert {r["race_id"] for r in rows} == {"jra:2026-11-28:TOK:10"}
    assert by[("WIN", "3", evening)]["odds"] == pytest.approx(6.6)
    assert by[("QUINELLA", POOL_ROW, evening)]["investment"] == pytest.approx(99900.0)


def test_minutes_to_post_uses_the_jravan_post_time_and_is_null_without_it(data: Path) -> None:
    kyo12 = "jra:2024-06-08:KYO:12"
    snaps = [
        *day_snaps(),
        o1(jst(2024, 6, 8, 16, 0), win=[w("05", "5", 3.3)], votes=(777, 0, 0), venue="08",
           race="12", kai="02", nichi="04"),
    ]  # fmt: skip
    write_day(DAY, snaps)
    write_races(
        [
            (TOK11, "15:40"),
            (TOK01, "10:05"),
            (KYO11, None),  # a race row without a post time
            # kyo12 has no race row; these are other races on other days or venues
            ("jra:2024-06-08:HSN:12", "16:10"),
            ("jra:2024-06-09:KYO:12", "16:25"),
            ("jra:2024-06-01:TOK:11", "15:45"),
        ]
    )
    timeseries.build_day(DAY)
    got = read_rows(day_file(DAY))
    posts = {TOK11: post_at(DAY, "15:40"), TOK01: post_at(DAY, "10:05")}
    assert posts[TOK11] == utc(2024, 6, 8, 6, 40)
    assert got == expected(snaps, posts=posts)

    def minutes(race: str, at: datetime) -> set:
        return {
            r[col("minutes_to_post")]
            for r in got
            if r[col("race_id")] == race and r[col("snapshot_utc")] == at.astimezone(UTC)
        }

    # Written out by hand: post 15:40 JST on 06-08.
    assert minutes(TOK11, jst(2024, 6, 7, 18, 30)) == {1270.0}  # 21 h 10 min, the evening before
    assert minutes(TOK11, jst(2024, 6, 8, 9, 15)) == {385.0}
    assert minutes(TOK11, jst(2024, 6, 8, 15, 30)) == {10.0}
    assert minutes(TOK11, jst(2024, 6, 8, 15, 42)) == {-2.0}  # announced after the post
    assert minutes(TOK01, jst(2024, 6, 8, 9, 50)) == {15.0}
    tok11 = [r for r in got if r[col("race_id")] == TOK11]
    assert {r[col("post_utc")] for r in tok11} == {utc(2024, 6, 8, 6, 40)}
    for race in (KYO11, kyo12):
        mine = [r for r in got if r[col("race_id")] == race]
        assert mine, race
        assert {(r[col("post_utc")], r[col("minutes_to_post")]) for r in mine} == {(None, None)}


def test_the_latest_snapshot_is_the_final_capture(data: Path) -> None:
    race = dict(venue="09", race="07", kai="02", nichi="06")
    s1, s2, s3 = jst(2024, 6, 8, 10, 0), jst(2024, 6, 8, 12, 30), jst(2024, 6, 8, 13, 1)
    snaps = [  # written out of time order
        o1(s3, win=[w("01", "1", 2.0)], place=[p("01", "1", 1.1, 1.2)], votes=(300, 30, 0),
           **race),
        o1(s1, win=[w("01", "1", 2.4), w("09", "9", 30.0)], place=[p("09", "9", 5.0, 7.0)],
           votes=(100, 10, 0), **race),
        o1(s2, win=[w("01", "1", 2.2), w("09", "9", 33.0)], place=[p("01", "1", 1.2, 1.3)],
           votes=(200, 20, 0), **race),
        o2(s2, [w("0109", "1,9", 40.0)], 2000, **race),
        o2(s1, [w("0109", "1,9", 44.0)], 1000, **race),
        # Another race: its own latest.
        o1(jst(2024, 6, 8, 9, 0), win=[w("01", "1", 8.0)], votes=(50, 0, 0), venue="09",
           race="01", kai="02", nichi="06"),
    ]  # fmt: skip
    write_day(DAY, snaps)
    timeseries.build_day(DAY)
    got = read_rows(day_file(DAY))
    assert got == expected(snaps)
    rid = "jra:2024-06-08:HSN:07"
    final = {
        (r[col("pool")], r[col("combination")]): r[col("snapshot_utc")]
        for r in got
        if r[col("race_id")] == rid and r[col("is_final_capture")]
    }
    t1, t2, t3 = (x.astimezone(UTC) for x in (s1, s2, s3))
    assert final == {
        ("WIN", "1"): t3, ("WIN", "9"): t2, ("WIN", POOL_ROW): t3,
        ("PLACE", "1"): t3, ("PLACE", "9"): t1, ("PLACE", POOL_ROW): t3,
        ("QUINELLA", "1,9"): t2, ("QUINELLA", POOL_ROW): t2,
    }  # fmt: skip
    counts: dict[tuple, int] = {}
    for r in got:
        k = (r[col("race_id")], r[col("pool")], r[col("combination")])
        counts[k] = counts.get(k, 0) + bool(r[col("is_final_capture")])
    assert set(counts.values()) == {1}  # exactly one final row per key
    assert all(r[col("is_final_capture")] in (True, False) for r in got)


def test_dividends_join_on_race_pool_and_combination(data: Path) -> None:
    # A day built without a dividends table: the columns are null.
    other_day = date(2024, 6, 1)
    first = [o1(jst(2024, 6, 1, 11, 0), win=[w("04", "4", 3.0)], votes=(10, 0, 0),
                race_day=other_day)]  # fmt: skip
    write_day(other_day, first)
    timeseries.build_day(other_day)
    plain = read_rows(day_file(other_day))
    assert plain == expected(first)
    assert {(r[col("dividend")], r[col("is_refund")]) for r in plain} == {(None, None)}

    snaps = day_snaps()
    write_day(DAY, snaps)
    paid = {
        (TOK11, "WIN", "4"): (350.0, False),
        (TOK11, "PLACE", "4"): (140.0, False),  # the same combination in another pool
        (TOK11, "PLACE", "1"): (190.0, False),
        (TOK11, "QUINELLA", "1,4"): (1230.0, False),
        (TOK11, "BRACKET_QUINELLA", "4,4"): (100.0, True),  # a refund keeps its flag
        (KYO11, "WIN", "7"): (1230.0, False),
        (KYO11, "QUINELLA", "2,7"): (1650.0, False),
    }
    unmatched = [
        (TOK01, "WIN", "4", 999.0, False),  # same pool and combination, another race
        (KYO11, "PLACE", "7", 410.0, False),  # same race and combination, a pool not sold
        ("jra:2024-06-08:HSN:11", "WIN", "1", 220.0, False),
    ]
    write_dividends([(*k, *v) for k, v in paid.items()] + unmatched)
    timeseries.build_day(DAY)
    got = read_rows(day_file(DAY))
    assert got == expected(snaps, dividends=paid)
    joined = {
        (r[col("race_id")], r[col("pool")], r[col("combination")]):
            (r[col("dividend")], r[col("is_refund")])
        for r in got
        if r[col("dividend")] is not None
    }  # fmt: skip
    assert joined == paid
    # Every capture of a paid combination carries it, the evening before's too.
    tok11_win4 = [
        r for r in got if (r[col("race_id")], r[col("pool")], r[col("combination")]) == (
            TOK11, "WIN", "4")
    ]  # fmt: skip
    assert len(tok11_win4) == 4
    assert {r[col("dividend")] for r in tok11_win4} == {350.0}
    pool_rows = [r for r in got if r[col("combination")] == POOL_ROW]
    assert {(r[col("dividend")], r[col("is_refund")]) for r in pool_rows} == {(None, None)}


def test_a_repeated_snapshot_is_one_row(data: Path) -> None:
    race = dict(venue="06", race="05", kai="05", nichi="01")
    at, later = jst(2024, 6, 8, 11, 20), jst(2024, 6, 8, 11, 40)
    snaps = [
        o1(at, win=[w("02", "2", 4.4), w("06", "6", 9.9)], votes=(400, 0, 0), **race),
        o1(later, win=[w("02", "2", 4.0), w("06", "6", 11.1)], votes=(800, 0, 0), **race),
        # The 11:20 board again (a resend): its rows replace the first record's; WIN 6 is not in
        # it, so the first record's WIN 6 row stays.
        o1(at, win=[w("02", "2", 4.5)], place=[p("02", "2", 1.4, 1.9)], votes=(410, 120, 0),
           **race),
        o2(at, [w("0206", "2,6", 15.0)], 300, **race),
        o2(at, [w("0206", "2,6", 15.5)], 310, **race),
        o2(at, [w("0206", "2,6", 15.7)], 320, **race),
    ]  # fmt: skip
    write_day(DAY, snaps)
    timeseries.build_day(DAY)
    got = read_rows(day_file(DAY))
    assert got == expected(snaps)
    rid = "jra:2024-06-08:NAK:05"
    t = at.astimezone(UTC)
    at_t = {
        (r[col("pool")], r[col("combination")]): (r[col("odds")], r[col("investment")])
        for r in got
        if r[col("race_id")] == rid and r[col("snapshot_utc")] == t
    }
    assert at_t == {
        ("WIN", "2"): (4.5, None),
        ("WIN", "6"): (9.9, None),
        ("WIN", POOL_ROW): (None, 41000.0),
        ("PLACE", "2"): (1.4, None),
        ("PLACE", POOL_ROW): (None, 12000.0),
        ("QUINELLA", "2,6"): (15.7, None),
        ("QUINELLA", POOL_ROW): (None, 32000.0),
    }
    assert len(got) == len(at_t) + 3  # the 11:40 board: two WIN slots and the WIN pool row
    keys = [sort_key(r) for r in got]
    assert len(keys) == len(set(keys))


def test_a_day_with_only_one_family_is_written(data: Path) -> None:
    only_o1, only_o2 = date(2024, 6, 1), date(2024, 6, 2)
    first = [
        o1(jst(2024, 5, 31, 18, 40), win=[w("08", "8", 7.1)], place=[p("08", "8", 2.0, 3.1)],
           bracket=[w("45", "4,5", 22.2)], votes=(70, 20, 9), race_day=only_o1),
        o1(jst(2024, 6, 1, 10, 10), win=[w("08", "8", 6.4)], place=[p("08", "8", 1.9, 2.8)],
           bracket=[w("45", "4,5", 20.0)], votes=(700, 200, 90), race_day=only_o1),
    ]  # fmt: skip
    second = [
        o2(jst(2024, 6, 2, 12, 0), [w("0311", "3,11", 66.0), w("0511", "5,11", 70.0)], 404,
           race_day=only_o2, venue="06", race="03"),
        o2(jst(2024, 6, 2, 12, 5), [w("0311", "3,11", 60.0)], 505, race_day=only_o2, venue="06",
           race="03"),
    ]  # fmt: skip
    write_day(only_o1, first)
    write_day(only_o2, second)
    raw = paths.raw_dir(SOURCE)
    assert not jvd_path(raw, "0B42", only_o1).exists()
    assert not jvd_path(raw, "0B41", only_o2).exists()

    timeseries.build_day(only_o1)
    timeseries.build_day(only_o2)
    for day, snaps, pools in (
        (only_o1, first, {"WIN", "PLACE", "BRACKET_QUINELLA"}),
        (only_o2, second, {"QUINELLA"}),
    ):
        path = day_file(day)
        assert pq.read_schema(path) == compact.SCHEMA, day
        got = read_rows(path)
        assert got == expected(snaps), day
        assert {r[col("pool")] for r in got} == pools, day


def test_an_unchanged_day_is_not_rebuilt(data: Path) -> None:
    a, b = date(2024, 6, 1), DAY
    first = [
        o1(jst(2024, 6, 1, 10, 0), win=[w("02", "2", 3.3)], votes=(10, 0, 0), race_day=a),
        o2(jst(2024, 6, 1, 10, 0), [w("0102", "1,2", 9.0)], 20, race_day=a),
    ]
    second = day_snaps()
    write_day(a, first)
    write_day(b, second)
    timeseries.build_all()
    paths_ = {a: day_file(a), b: day_file(b)}
    assert read_rows(paths_[a]) == expected(first)
    assert read_rows(paths_[b]) == expected(second)

    old = 1_000_000_000_000_000_000  # 2001-09-09: any rewrite gives the file a new mtime

    def age() -> dict[date, tuple[int, int]]:
        for path in paths_.values():
            os.utime(path, ns=(old, old))
        return {d: stat(path) for d, path in paths_.items()}

    def stat(path: Path) -> tuple[int, int]:
        s = path.stat()
        return (s.st_ino, s.st_mtime_ns)

    # Nothing changed: neither day is written again, by build_all or build_day.
    before = age()
    bytes_before = {d: path.read_bytes() for d, path in paths_.items()}
    timeseries.build_all()
    timeseries.build_day(a)
    timeseries.build_day(b)
    assert {d: stat(path) for d, path in paths_.items()} == before
    assert {d: path.read_bytes() for d, path in paths_.items()} == bytes_before

    # One day's 0B42 input changes: that day is rebuilt with the new snapshot, the other is not.
    added = o2(jst(2024, 6, 1, 10, 30), [w("0102", "1,2", 8.5)], 45, race_day=a)
    write_day(a, [*first, added])
    before = age()
    timeseries.build_all()
    assert stat(paths_[b]) == before[b]
    assert stat(paths_[a]) != before[a]
    assert read_rows(paths_[a]) == expected([*first, added])

    # The workbook changes (its bytes, not the O1/O2 layouts): every day is rebuilt.
    before = age()
    write_ts_workbook(paths.data_dir() / "specs" / "jravan" / wb.WORKBOOK_NAME, decoys=2)
    timeseries.build_all()
    for d, path in paths_.items():
        assert stat(path) != before[d], d
    assert read_rows(paths_[a]) == expected([*first, added])
    assert read_rows(paths_[b]) == expected(second)

    # A day file that is gone is written again, though its inputs match the build record.
    gone = paths_[b].read_bytes()
    paths_[b].unlink()
    timeseries.build_day(b)
    assert paths_[b].read_bytes() == gone


def test_two_builds_write_identical_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snaps = day_snaps()  # no repeated key, so the record order cannot change a value
    posts = [(TOK11, "15:40"), (TOK01, "10:05"), (KYO11, "15:35")]
    paid = [(TOK11, "WIN", "4", 350.0, False), (KYO11, "QUINELLA", "2,7", 1650.0, False)]
    builds = {}
    for name, order in (("a", snaps), ("b", list(reversed(snaps)))):
        make_data(tmp_path / name, monkeypatch)
        write_day(DAY, order)
        write_races(posts)
        write_dividends(paid)
        report = timeseries.build_day(DAY)
        builds[name] = (day_file(DAY).read_bytes(), report.to_json())
    assert builds["a"] == builds["b"]
    assert len(builds["a"][0]) > 0

    # The same inputs rewritten (new mtimes, the build record gone): the same bytes.
    folder = day_file(DAY).parent
    (folder / BUILD_FILE).unlink()
    day_file(DAY).unlink()
    write_day(DAY, list(reversed(snaps)))
    timeseries.build_day(DAY)
    assert day_file(DAY).read_bytes() == builds["a"][0]
    assert sorted(p.name for p in folder.iterdir()) == sorted([BUILD_FILE, day_file(DAY).name])


def test_a_broken_day_is_reported_and_the_others_are_written(data: Path) -> None:
    good_a, broken, good_b, last_year = (
        date(2024, 6, 1),
        date(2024, 6, 8),
        date(2024, 6, 15),
        date(2023, 12, 24),
    )
    boards = {
        day: [
            o1(jst(day.year, day.month, day.day, 10, i), win=[w("03", "3", 3.0 + i)],
               votes=(100 + i, 0, 0), race_day=day),
            o2(jst(day.year, day.month, day.day, 10, i), [w("0103", "1,3", 9.0 + i)], 50 + i,
               race_day=day),
        ]
        for i, day in enumerate((good_a, broken, good_b))
    }  # fmt: skip
    for day, snaps in boards.items():
        write_day(day, snaps)
    # 2023 has only a 0B42 file.
    boards[last_year] = [
        o2(jst(2023, 12, 24, 13, 0), [w("0512", "5,12", 31.0)], 77, race_day=last_year,
           venue="06", race="11")
    ]  # fmt: skip
    write_day(last_year, boards[last_year])
    # The broken day's 0B41 file: its second record is cut short.
    raw = paths.raw_dir(SOURCE)
    good = boards[broken][0].record()
    cut = good[:-12] + b"\r\n"
    wb.write_jvd(jvd_path(raw, "0B41", broken), [good, cut, good])

    report = timeseries.build_all(years=[2024])
    assert isinstance(report, records.BuildReport)
    assert len(report.failures) == 1
    failure = json.dumps(report.failures[0], default=str)
    assert "2024-06-08" in failure or "20240608" in failure
    for day in (good_a, good_b):
        assert read_rows(day_file(day)) == expected(boards[day]), day
    assert not day_file(broken).exists()
    assert not day_file(last_year).exists()  # not a requested year

    runner = CliRunner()
    result = runner.invoke(app, ["japan", "jravan", "timeseries", "--year", "2023"])
    assert result.exit_code == 0, result.output
    assert read_rows(day_file(last_year)) == expected(boards[last_year])

    result = runner.invoke(app, ["japan", "jravan", "timeseries", "--day", "2024-06-15"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["japan", "jravan", "timeseries", "--day", "2024-06-08"])
    assert result.exit_code == 1, result.output
    assert not day_file(broken).exists()

    # Every day: the broken one fails, the run exits 1, the others are written, the report lists it.
    for day in (good_a, good_b, last_year):
        day_file(day).unlink()
    (day_file(good_a).parent / BUILD_FILE).unlink(missing_ok=True)
    result = runner.invoke(app, ["japan", "jravan", "timeseries"])
    assert result.exit_code == 1, result.output
    for day in (good_a, good_b, last_year):
        assert read_rows(day_file(day)) == expected(boards[day]), day
    assert not day_file(broken).exists()
    written = paths.reports_dir() / REPORT
    assert written.is_file()
    parsed = json.loads(written.read_text(encoding="utf-8"))
    assert parsed["name"] == "timeseries"
    assert len(parsed["failures"]) == 1
    text = json.dumps(parsed["failures"])
    assert "2024-06-08" in text or "20240608" in text


# ------------------------------------------------------------------------------------------------
# The real export (local: needs $HORSETRACK_DATA_DIR with the export, the workbook and the races)

REAL_YEAR = 2025
REAL_POOLS = {"WIN", "PLACE", "BRACKET_QUINELLA", "QUINELLA"}


def _repo_status() -> str:
    root = Path(__file__).resolve().parents[3]
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root, capture_output=True, text=True, check=True,
    )  # fmt: skip
    return out.stdout


@pytest.mark.local
def test_a_real_day_builds(tmp_path: Path) -> None:
    export, _ = wb.real_export_or_skip()
    files = sorted((export / "timeseries" / "0B41" / f"{REAL_YEAR}").glob("*.jvd"))
    if not files:
        pytest.skip(f"no 0B41 files for {REAL_YEAR}")
    races_path = paths.processed_dir(SOURCE) / "races.parquet"
    if not races_path.is_file():
        pytest.skip(f"jravan races missing: {races_path}")
    day = datetime.strptime(files[0].stem, "%Y%m%d").date()
    store_file = paths.processed_dir(SOURCE) / "odds_timeseries" / f"{day.isoformat()}.parquet"
    store_before = store_file.stat() if store_file.exists() else None
    repo_before = _repo_status()

    out = tmp_path / "out"
    report = timeseries.build_day(day, raw=export, out=out)
    assert isinstance(report, records.BuildReport)
    assert report.failures == []

    written = list(out.rglob(f"{day.isoformat()}.parquet"))
    assert len(written) == 1, written
    assert pq.read_schema(written[0]) == compact.SCHEMA
    table = pq.read_table(written[0])
    assert table.num_rows > 0
    got = table.to_pydict()
    assert set(got["pool"]) <= REAL_POOLS
    assert all(v is None for v in got["fetched_utc"])
    race_ids = set(got["race_id"])
    assert all(r.startswith(f"jra:{day.isoformat()}:") for r in race_ids)
    races = pq.read_table(races_path, columns=["race_id"], filters=[("race_date", "=", day)])
    assert race_ids <= set(races["race_id"].to_pylist())
    keys = list(zip(got["race_id"], got["pool"], got["combination"], got["snapshot_utc"],
                    strict=True))  # fmt: skip
    assert len(set(keys)) == len(keys)
    never = datetime.min.replace(tzinfo=UTC)
    order = [(r, p_, c, s is None, s or never) for r, p_, c, s in keys]
    assert order == sorted(order)

    # Nothing was written into the repository or the real store.
    assert _repo_status() == repo_before
    after = store_file.stat() if store_file.exists() else None
    if store_before is None:
        assert after is None
    else:
        assert (after.st_ino, after.st_mtime_ns) == (store_before.st_ino, store_before.st_mtime_ns)
    shutil.rmtree(out)
