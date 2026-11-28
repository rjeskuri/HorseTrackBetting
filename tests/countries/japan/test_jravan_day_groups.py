"""Ticket #502: the jravan price stores keep each race day in its own row group, so a development
read of a store with interleaved holdout days opens no holdout day's bytes (D14).

Contract pinned here (the coder follows it):

- Builders. `jravan.odds.build_final_odds` (`final_odds.parquet`), `jravan.votes.build_pool_votes`
  (`pool_votes.parquet`), `jravan.races.build_races` (`races.parquet`, also once
  `jravan.runners.build_runners` has rewritten it with the winners' times) and
  `jravan.runners.build_runners` (`runners.parquet`) -- the jravan tables `read_store_table` serves
  with `touches_price=True` -- write exactly one row group per race day: every row group's
  `race_date` column chunk records statistics (min and max, and a null count of 0) with min ==
  max, and the groups' days strictly ascend, so a day is never split across groups nor shares one
  with another day. The rows, their order (each table's sort keys, race_id first, so a day's rows
  are contiguous), the schema and the atomic write (a tmp file and `os.replace`: a failed swap
  leaves the old file whole and no stray file) are unchanged.
- Reader. `models.v0.store.read_store_table` on a store whose row groups each hold one race day
  (with statistics) reads no byte of a row group holding a holdout day (#71's `is_holdout_day`,
  looked up at call time), whatever the range (holdout days inside it, open-ended to `date.max`,
  from `date.min`, one day), with or without a `where`, `touches_price` True or False. It opens no
  row group outside the read's development days either, so a one-day read opens that day's group
  only. The holdout filter stays in the scan as the second guard.
- How the reads are seen. `read_store_table` opens the file with `pyarrow.memory_map(path)` or
  `pyarrow.OSFile(path)` (looked up on the `pyarrow` module at call time), and every byte the
  Parquet reader takes comes through that handle, by byte range. The spy serves the handle from a
  copy of the file that records each read's byte range (`pa.PythonFile`). A read that overlaps a
  row group's column chunks opens that group: a coalesced (pre-buffered) range across a holdout
  group opens it, and so does a whole-file buffer. The library's own footer reads (calibrated on
  each file: the reads `pq.ParquetFile(...).metadata` and a dataset fragment's `metadata` make)
  are not row-group reads. Every development group holding a row of the result must show each
  requested column's chunk read through the handle, so a read that reached the file by any other
  route (a path, another opener, a buffer of the whole file) fails.
- A store not yet rebuilt (the old one-group-per-year layout, each group mixing development and
  holdout days) reads as before: exactly the development rows in range. No byte-level claim is
  made there: such a group has to be opened. (Small shuffled mixed groups are
  `test_development_scan`'s.)

All data is synthetic and written under tmp_path; HORSETRACK_DATA_DIR points there. The builders'
inputs are the made-up exports of the builders' own test modules (`test_jravan_odds`,
`test_jravan_votes`, `test_jravan_runners`), with their hand-written expected rows. Holdout days
are a stand-in for `is_holdout_day` (an explicit set of days, interleaved with development days),
and in a few reads the committed rule over synthetic days from its start (2026-11-28), recomputed
here with hashlib. The holdout is never opened.
"""

from __future__ import annotations

import hashlib
import io
import os
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import yaml

import horsetrack.core.eval.holdout as holdout_module
from horsetrack.core import paths
from horsetrack.core.store.build import arrow_schema
from horsetrack.countries.japan.jravan import odds, races, runners, votes
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table

# isort: split
# (its own block: the sibling helper modules, whose made-up exports and expected rows are reused)
import jravan_workbook as wb
import test_jravan_odds as od
import test_jravan_runners as rn
import test_jravan_votes as vt

SOURCE = "jravan"
COUNTRY = "japan"
POOL_ROW = "__pool__"
TABLES = ("final_odds", "pool_votes", "races", "runners")
REPO_ROOT = Path(__file__).resolve().parents[3]
JAPAN_HOLDOUT = REPO_ROOT / "src" / "horsetrack" / "countries" / "japan" / "config" / "holdout.yaml"


def day_of(race_id: str) -> date:
    return date.fromisoformat(race_id.split(":")[1])


def plain(value: object) -> object:
    """A cell for comparing: floats rounded to 6 places, everything else as is."""
    return round(value, 6) if isinstance(value, float) else value


def values(table: pa.Table, columns: Sequence[str]) -> list[tuple]:
    """The rows of `table` as tuples of `columns`, in table order."""
    cells = [table.column(c).to_pylist() for c in columns]
    return [tuple(plain(v) for v in row) for row in zip(*cells, strict=True)]


# ---------------------------------------------------------------------------------------------
# The byte-range spy
# ---------------------------------------------------------------------------------------------


class CountingFile(io.BytesIO):
    """A Parquet file in memory that records the byte range of every read made of it (through
    ``pa.PythonFile``)."""

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.data = data
        self.reads: list[tuple[int, int]] = []

    def read(self, size=-1):
        at = self.tell()
        out = super().read(size)
        self.reads.append((at, at + len(out)))
        return out

    def readinto(self, buffer):
        at = self.tell()
        n = super().readinto(buffer)
        self.reads.append((at, at + n))
        return n


def footer_reads(data: bytes) -> frozenset[tuple[int, int]]:
    """The reads pyarrow makes to open the file's footer (its speculative tail read, and the rest
    of a long footer), by the two routes a reader opens metadata through. They read no row
    group: in a small file the tail read is the whole file, which is why the tests also ask for
    the development chunks to be read through the handle."""
    seen: set[tuple[int, int]] = set()
    openers = (
        lambda f: pq.ParquetFile(f).metadata,
        lambda f: ds.ParquetFileFormat().make_fragment(f).metadata,
    )
    for metadata_of in openers:
        probe = CountingFile(data)
        metadata_of(pa.PythonFile(probe, mode="r"))
        seen.update(probe.reads)
    return frozenset(seen)


@dataclass(frozen=True)
class FileLayout:
    """Each row group's column chunks (byte ranges, from the footer), its byte extent and the race
    days it holds; and the library's footer reads of the file."""

    chunks: tuple[dict[str, tuple[int, int]], ...]
    extents: tuple[tuple[int, int], ...]
    days: tuple[frozenset[date], ...]
    footer: frozenset[tuple[int, int]]

    def touched(self, reads: Sequence[tuple[int, int]]) -> set[int]:
        return {
            g for g, (lo, hi) in enumerate(self.extents) if any(a < hi and b > lo for a, b in reads)
        }


def file_layout(data: bytes) -> FileLayout:
    file = pq.ParquetFile(io.BytesIO(data))
    meta = file.metadata
    chunks, extents, days = [], [], []
    for g in range(meta.num_row_groups):
        group = meta.row_group(g)
        mine = {}
        for c in range(group.num_columns):
            chunk = group.column(c)
            first = chunk.data_page_offset
            if chunk.has_dictionary_page and chunk.dictionary_page_offset is not None:
                first = min(first, chunk.dictionary_page_offset)
            mine[chunk.path_in_schema] = (first, first + chunk.total_compressed_size)
        chunks.append(mine)
        extents.append((min(lo for lo, _ in mine.values()), max(hi for _, hi in mine.values())))
        dates = file.read_row_group(g, columns=["race_date"]).column("race_date").to_pylist()
        days.append(frozenset(d for d in dates if d is not None))
    return FileLayout(tuple(chunks), tuple(extents), tuple(days), footer_reads(data))


def describe(layout: FileLayout, groups: Sequence[int]) -> str:
    return ", ".join(
        f"group {g} ({', '.join(d.isoformat() for d in sorted(layout.days[g]))})" for g in groups
    )


class ByteSpy:
    """Serves every read-only open of a file in `folder` by `pyarrow.memory_map` or
    `pyarrow.OSFile` from a copy that records the byte range of each read (`opened[file name]`,
    one copy per open)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, folder: Path) -> None:
        self.folder = folder.resolve()
        self.opened: dict[str, list[CountingFile]] = {}
        for name in ("memory_map", "OSFile"):
            monkeypatch.setattr(pa, name, self._serve(getattr(pa, name)))

    def _serve(self, real):
        def open_(path, mode="r", *args, **kwargs):
            if not isinstance(path, (str, os.PathLike)) or mode not in ("r", "rb"):
                return real(path, mode, *args, **kwargs)
            target = Path(path).resolve()
            if target.parent != self.folder:
                return real(path, mode, *args, **kwargs)
            counting = CountingFile(target.read_bytes())
            self.opened.setdefault(target.name, []).append(counting)
            return pa.PythonFile(counting, mode="r")

        return open_

    def assert_isolated(
        self,
        filename: str,
        *,
        held: frozenset[date],
        start: date,
        end: date,
        columns: Sequence[str],
        result_days: Counter,
        only_day: date | None = None,
    ) -> None:
        """No read of `filename` overlapped a row group holding a holdout day, or a group outside
        the development days of `start..end`; every development group holding a result row
        (`result_days`: rows per day) had each of `columns`' chunks read through the handle; with
        `only_day`, the groups read are exactly that day's."""
        copies = self.opened.get(filename)
        assert copies, (
            f"read_store_table did not open {filename} through pyarrow.memory_map or "
            "pyarrow.OSFile: its reads cannot be seen"
        )
        layout = file_layout(copies[0].data)
        reads = [r for f in copies for r in f.reads if r[1] > r[0] and r not in layout.footer]
        opened = layout.touched(reads)

        holdout = {g for g, days in enumerate(layout.days) if days & held}
        leaked = sorted(opened & holdout)
        assert not leaked, (
            "the development read opened row groups holding holdout days: "
            + describe(layout, leaked)
        )
        in_range = {
            g
            for g, days in enumerate(layout.days)
            if g not in holdout and any(start <= d <= end for d in days)
        }
        stray = sorted(opened - in_range)
        assert not stray, (
            f"the read of {start}..{end} opened row groups outside its development days: "
            + describe(layout, stray)
        )
        for g, days in enumerate(layout.days):
            if not any(result_days[d] for d in days):
                continue
            for column in columns:
                lo, hi = layout.chunks[g][column]
                assert any(a < hi and b > lo for a, b in reads), (
                    f"{describe(layout, [g])} holds rows of the result, but its {column!r} chunk "
                    "was not read through the handle opened with pyarrow.memory_map/OSFile: the "
                    "rows reached the reader by another route (a whole-file buffer, a path, "
                    "another opener)"
                )
        if only_day is not None:
            assert opened == {g for g, days in enumerate(layout.days) if days == {only_day}}, (
                f"a read of {only_day} alone opened {describe(layout, sorted(opened))}"
            )


def stand_in(monkeypatch: pytest.MonkeyPatch, held: frozenset[date]) -> list[date]:
    """Make exactly `held` the holdout days (development_filter looks `is_holdout_day` up at call
    time); the days asked about are recorded."""
    asked: list[date] = []

    def is_holdout_day(country: str, day: date) -> bool:
        asked.append(day)
        return country == COUNTRY and day in held

    monkeypatch.setattr(holdout_module, "is_holdout_day", is_holdout_day)
    return asked


# ---------------------------------------------------------------------------------------------
# The built store: the four tables written by their builders from the made-up exports
# ---------------------------------------------------------------------------------------------

# The stand-in's holdout days: inside each built table's span, each between development days of
# its table, so a group per year mixes them with development days.
BUILT_HELD = frozenset({date(2023, 12, 28), date(2024, 6, 1), date(2024, 12, 1), date(2025, 1, 5)})


@dataclass(frozen=True)
class Built:
    """A built table as its builder's test module writes it out: per row its race day and the
    values of `columns`, in the store's order; and a `where` with the same predicate on those
    values (`keep`)."""

    name: str
    rows: tuple[tuple[date, tuple], ...]
    columns: tuple[str, ...]
    where: ds.Expression
    keep: Callable[[tuple], bool]

    @property
    def days(self) -> list[date]:
        return sorted({d for d, _ in self.rows})


def _final_odds() -> Built:
    # od's FullRow: jurisdiction, race_id, race_date, venue, race_no, pool, combination, odds, ...
    rows = od.in_store_order([r for _, rows in od.build_inputs().values() for r in rows])
    return Built(
        "final_odds",
        tuple((r[2], (r[1], r[5], r[6], r[7])) for r in rows),
        ("race_id", "pool", "combination", "odds"),
        ds.field("pool") == "WIN",
        lambda v: v[1] == "WIN",
    )


def _pool_votes() -> Built:
    # vt's Row: race_id, pool, combination, votes, refund_votes, fav_rank
    rows = vt.ROWS_2023 + vt.ROWS_2024
    return Built(
        "pool_votes",
        tuple((day_of(r[0]), (r[0], r[1], r[2], r[3])) for r in rows),
        ("race_id", "pool", "combination", "votes"),
        ds.field("combination") != POOL_ROW,
        lambda v: v[2] != POOL_ROW,
    )


def _races() -> Built:
    # every race of rn's export, with the winner's time build_runners writes on it
    rows = sorted(rn.WINNERS_TIME.items())
    return Built(
        "races",
        tuple((day_of(rid), (rid, None if t is None else plain(float(t)))) for rid, t in rows),
        ("race_id", "finish_time_s"),
        ds.field("race_no") >= 5,
        lambda v: int(v[0].split(":")[3]) >= 5,
    )


def _runners() -> Built:
    rows = rn.IDS_2024 + rn.IDS_2025  # sorted by race_id, horse_no
    return Built(
        "runners",
        tuple((day_of(rid), (rid, n)) for rid, n in rows),
        ("race_id", "horse_no"),
        ds.field("horse_no") <= 4,
        lambda v: v[1] <= 4,
    )


BUILT = {b.name: b for b in (_final_odds(), _pool_votes(), _races(), _runners())}


def write_export(name: str, root: Path) -> None:
    """The made-up workbook and export `name`'s builder reads, under the data dir `root`."""
    workbook = root / "specs" / SOURCE / wb.WORKBOOK_NAME
    if name == "final_odds":
        od.write_odds_workbook(workbook)
        for year, (recs, _) in od.build_inputs().items():
            od.write_year(root, year, recs)
    elif name == "pool_votes":
        vt.write_votes_workbook(workbook)
        for (rt, year), recs in vt.build_races().items():
            vt.write_year(root, rt, year, recs)
    else:  # races and runners: one RA and SE export
        rn.write_workbook(workbook)
        rn.write_build_inputs(root)


def build(name: str, out: Path, years: list[int] | None = None) -> None:
    builders = {
        "final_odds": odds.build_final_odds,
        "pool_votes": votes.build_pool_votes,
        "races": races.build_races,
        "runners": runners.build_runners,
    }
    builders[name](years=years, out=out)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The data dir of a jravan store whose four tables the builders wrote, each from its own
    made-up export and workbook, into one processed folder (races rewritten by build_runners)."""
    store = tmp_path_factory.mktemp("built").resolve()
    out = store / "processed" / SOURCE
    with pytest.MonkeyPatch.context() as mp:
        for name in ("final_odds", "pool_votes", "races"):
            root = tmp_path_factory.mktemp(f"export-{name}").resolve()
            mp.setenv(paths.DATA_DIR_VAR, str(root))
            write_export(name, root)
            build(name, out)
            if name == "races":
                build("runners", out)
    return store


def store_schema(name: str) -> pa.Schema:
    if name == "final_odds":
        return od.EXPECTED_SCHEMA
    if name == "pool_votes":
        return pa.schema(vt.EXPECTED_SCHEMA)
    return arrow_schema(name)  # the store's races and runners


def statistics_days(meta: pq.FileMetaData) -> list[date]:
    """The day of each row group, from its race_date statistics, which must record one day and no
    null."""
    days = []
    for g in range(meta.num_row_groups):
        group = meta.row_group(g)
        chunk = next(
            group.column(c)
            for c in range(group.num_columns)
            if group.column(c).path_in_schema == "race_date"
        )
        stats = chunk.statistics
        assert stats is not None and stats.has_min_max, f"row group {g} has no race_date min/max"
        assert stats.has_null_count and stats.null_count == 0, f"row group {g}: race_date nulls"
        assert stats.min == stats.max, (
            f"row group {g} holds the days {stats.min} to {stats.max}, not one race day"
        )
        days.append(stats.min)
    return days


# ---------------------------------------------------------------------------------------------
# DoD: the builders write one row group per race day (rows, order and schema unchanged)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", TABLES)
def test_each_builder_writes_one_row_group_per_race_day(built: Path, name: str) -> None:
    want = BUILT[name]
    folder = built / "processed" / SOURCE
    path = folder / f"{name}.parquet"
    assert pq.read_schema(path) == store_schema(name)

    meta = pq.read_metadata(path)
    days = statistics_days(meta)
    assert len(want.days) >= 4  # several days over two years or more, so a group per year mixes
    assert days == want.days, (
        f"{name}: the row groups hold the days {days}, not one group per race day {want.days}"
    )
    per_day = Counter(d for d, _ in want.rows)
    file = pq.ParquetFile(path)
    for g, day in enumerate(days):
        dates = file.read_row_group(g, columns=["race_date"]).column("race_date").to_pylist()
        assert dates == [day] * per_day[day], (name, g, day)

    # The rows and their order, as the builder's own tests write them out.
    assert values(pq.read_table(path, columns=list(want.columns)), want.columns) == [
        v for _, v in want.rows
    ]
    # The atomic write leaves no stray file.
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        f"{n}.parquet" for n in (*TABLES, "meetings")
    )


@pytest.mark.parametrize("name", TABLES)
def test_a_failed_swap_leaves_the_day_grouped_file_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    root = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(root))
    out = tmp_path / "out"
    write_export(name, root)
    if name == "runners":
        build("races", out)
    build(name, out)
    if name == "races":
        build("runners", out)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    days = statistics_days(pq.read_metadata(out / f"{name}.parquet"))
    assert days == BUILT[name].days

    def refuse(*args, **kwargs):
        raise OSError("refused")

    # A rebuild of one year only (other bytes) whose swap fails: the old file stays whole.
    with monkeypatch.context() as m:
        m.setattr(os, "replace", refuse)
        with pytest.raises(OSError):
            build(name, out, years=[BUILT[name].days[-1].year])
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


# ---------------------------------------------------------------------------------------------
# DoD: a development read of the built store opens no row group holding a holdout day
# ---------------------------------------------------------------------------------------------

BUILT_READS = (
    "price-era-to-date-max",
    "date-min-to-date-max-with-where",
    "first-holdout-day-to-last-day-with-where",
    "last-development-day",
)


def built_read(want: Built, read: str) -> tuple[date, date, bool, ds.Expression | None]:
    days = want.days
    dev = [d for d in days if d not in BUILT_HELD]
    held = [d for d in days if d in BUILT_HELD]
    assert dev and held and dev[0] < held[0] < dev[-1], (want.name, days)  # interleaved
    return {
        "price-era-to-date-max": (PRICE_ERA_START, date.max, True, None),
        "date-min-to-date-max-with-where": (date.min, date.max, False, want.where),
        "first-holdout-day-to-last-day-with-where": (held[0], days[-1], True, want.where),
        "last-development-day": (dev[-1], dev[-1], True, None),
    }[read]


@pytest.mark.parametrize("read", BUILT_READS)
@pytest.mark.parametrize("name", TABLES)
def test_read_store_table_opens_no_holdout_day_group_of_the_built_store(
    built: Path, monkeypatch: pytest.MonkeyPatch, name: str, read: str
) -> None:
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(built))
    asked = stand_in(monkeypatch, BUILT_HELD)
    spy = ByteSpy(monkeypatch, built / "processed" / SOURCE)
    want = BUILT[name]
    start, end, touches_price, where = built_read(want, read)

    table = read_store_table(
        name,
        columns=list(want.columns),
        start=start,
        end=end,
        touches_price=touches_price,
        where=where,
        source=SOURCE,
    )

    expected = [
        (d, v)
        for d, v in want.rows
        if start <= d <= end and d not in BUILT_HELD and (where is None or want.keep(v))
    ]
    assert expected, "every read returns development rows"
    assert table.column_names == list(want.columns)
    assert Counter(values(table, want.columns)) == Counter(v for _, v in expected)
    assert asked, "the holdout rule was not consulted"
    spy.assert_isolated(
        f"{name}.parquet",
        held=BUILT_HELD,
        start=start,
        end=end,
        columns=want.columns,
        result_days=Counter(d for d, _ in expected),
        only_day=start if start == end else None,
    )


# ---------------------------------------------------------------------------------------------
# DoD: the same over a larger store with interleaved holdout days, by range, where and price flag
# ---------------------------------------------------------------------------------------------

SCHEMA = od.EXPECTED_SCHEMA  # final_odds' columns
POOLS = ("WIN", "PLACE", "QUINELLA", "TRIO")
COMBINATIONS = 6

# 36 race days, the Saturdays and Sundays from 2024-12-07 to 2025-04-06: D a development day, H a
# holdout day (the first a development day, the last a holdout day, runs of both). A holdout day
# runs one race, a development day one or two, so every row group is a few KiB and a pre-buffer
# that coalesces reads across small holes would bridge a holdout group between development groups;
# the many groups keep the file well over pyarrow's footer read.
PATTERN = "DHDHHDDHDHDDDHHDHDDHDHHHDDHDHDHDDHDH"
SPAN = tuple(
    (
        date(2024, 12, 7) + timedelta(days=7 * (k // 2) + k % 2),
        mark == "H",
        1 + k % 2 * (mark == "D"),
    )
    for k, mark in enumerate(PATTERN)
)  # (day, holdout, races)
SPAN_HELD = frozenset(d for d, held, _ in SPAN if held)


def span_day(k: int) -> date:
    return SPAN[k][0]


# The reads below rely on these marks.
assert len(PATTERN) == 36 and PATTERN[0] == "D" and PATTERN[-1] == "H"
assert PATTERN[3] == PATTERN[23] == PATTERN[22] == "H" and PATTERN[7] == PATTERN[9] == "H"
assert PATTERN[8] == PATTERN[31] == "D" and span_day(35) == date(2025, 4, 6)


def _committed_rule() -> tuple[date, str]:
    section = yaml.safe_load(JAPAN_HOLDOUT.read_text(encoding="utf-8"))["holdout"]
    start = section["start"]
    start = start if isinstance(start, date) else date.fromisoformat(str(start))
    return start, str(section["seed"])


def committed_held(day: date) -> bool:
    """The committed holdout rule, recomputed: a day from the start whose seeded hash is even."""
    start, seed = _committed_rule()
    digest = hashlib.sha256(f"{seed}:{day.isoformat()}".encode()).hexdigest()
    return day >= start and int(digest, 16) % 2 == 0


COMMITTED_DAYS = tuple(date(2026, 11, 27) + timedelta(days=k) for k in range(36))


def committed_span() -> tuple[tuple[date, bool, int], ...]:
    span = tuple(
        (d, committed_held(d), 1 if committed_held(d) else 1 + k % 2)
        for k, d in enumerate(COMMITTED_DAYS)
    )
    held = [d for d, h, _ in span if h]
    assert held and len(held) < len(span) and min(held) > COMMITTED_DAYS[0], held
    return span


def span_rows(span) -> list[dict]:
    """Store-shaped final_odds rows: per race of each day, every pool's combinations and its pool
    row, the values set by the day and the race."""
    rows = []
    for k, (day, _, n_races) in enumerate(span):
        for r in range(1, n_races + 1):
            race_id = f"jra:{day.isoformat()}:TOK:{r:02d}"
            base = {"jurisdiction": "jra", "race_id": race_id, "race_date": day, "venue": "TOK",
                    "race_no": r, "published_utc": None, "data_kubun": "4"}  # fmt: skip
            for p, pool in enumerate(POOLS):
                for c in range(1, COMBINATIONS + 1):
                    price = round(1.0 + c * 0.7 + k + 0.1 * r + 0.01 * p, 2)
                    rows.append({**base, "pool": pool, "combination": str(c), "odds": price,
                                 "odds_hi": price + 0.5 if pool == "PLACE" else None,
                                 "fav_rank": c, "investment": None,
                                 "status_raw": None})  # fmt: skip
                rows.append({**base, "pool": pool, "combination": POOL_ROW, "odds": None,
                             "odds_hi": None, "fav_rank": None,
                             "investment": float(1000 * (k + 1) + 100 * r + p),
                             "status_raw": None})  # fmt: skip
    return rows


def write_day_groups(path: Path, rows: list[dict]) -> bytes:
    """`rows` with one row group per race day, uncompressed and without dictionaries, so the
    file is larger than pyarrow's footer read: a buffer of the whole file is told apart from it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    days = sorted({r["race_date"] for r in rows})
    with pq.ParquetWriter(path, SCHEMA, compression="none", use_dictionary=False) as writer:
        for day in days:
            part = table.filter(ds.field("race_date") == pa.scalar(day, pa.date32()))
            writer.write_table(part, row_group_size=part.num_rows)
    data = path.read_bytes()
    assert statistics_days(pq.read_metadata(path)) == days
    layout = file_layout(data)
    assert min(lo for lo, _ in layout.footer) > layout.extents[0][1], (
        "the footer read must not cover the first row group"
    )
    assert max(hi - lo for lo, hi in layout.extents) < 8 * 1024, "every row group is small"
    return data


def write_year_groups(path: Path, rows: list[dict]) -> None:
    """`rows` with one row group per year, as the builders wrote the stores before #502 (and the
    stores stay until they are rebuilt): each group mixes development and holdout days."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    with pq.ParquetWriter(path, SCHEMA, compression="zstd") as writer:
        for year in sorted({r["race_date"].year for r in rows}):
            mine = (ds.field("race_date") >= pa.scalar(date(year, 1, 1), pa.date32())) & (
                ds.field("race_date") <= pa.scalar(date(year, 12, 31), pa.date32())
            )
            part = table.filter(mine)
            writer.write_table(part, row_group_size=part.num_rows)
    groups = file_layout(path.read_bytes()).days
    assert len(groups) == 2 and all(days & SPAN_HELD and days - SPAN_HELD for days in groups)


def written_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    return data / "processed" / SOURCE / "final_odds.parquet"


def expected_rows(rows, held, start, end, place_only, columns) -> list[tuple]:
    return [
        tuple(plain(r[c]) for c in columns)
        for r in rows
        if start <= r["race_date"] <= end
        and r["race_date"] not in held
        and (not place_only or r["pool"] == "PLACE")
    ]


D = date
SPAN_READS = [  # (id, start, end); every one over the stand-in's span
    ("whole-span", span_day(0), span_day(35)),
    ("holdout-day-to-holdout-day", span_day(3), span_day(23)),
    ("price-era-to-date-max", PRICE_ERA_START, date.max),
    ("development-day-to-date-max", span_day(31), date.max),
    ("one-development-day-between-holdout-days", span_day(8), span_day(8)),
    ("one-holdout-day", span_day(22), span_day(22)),
]
READ_COLUMNS = (
    ["race_id", "combination", "odds"],
    ["odds", "race_date", "pool"],
    ["investment"],
)
COMMITTED_READS = [  # (id, start, end); over the committed rule's synthetic days
    ("whole-span", COMMITTED_DAYS[0], COMMITTED_DAYS[-1]),
    ("holdout-start-to-date-max", D(2026, 11, 28), date.max),
    ("one-development-day-between-holdout-days", D(2026, 11, 30), D(2026, 11, 30)),
]


def case(rule, name, start, end, touches_price, place_only, columns):
    price = "price" if touches_price else "no-price"
    where = "-where" if place_only else ""
    return pytest.param(
        rule, start, end, touches_price, place_only, columns, id=f"{rule}-{name}-{price}{where}"
    )


CASES = [
    case("stand-in", name, start, end, price, place, READ_COLUMNS[k % len(READ_COLUMNS)])
    for k, (name, start, end) in enumerate(SPAN_READS)
    for price in (True, False)
    for place in (False, True)
]
CASES += [
    case(
        "stand-in",
        "date-min-to-date-max",
        date.min,
        date.max,
        False,
        place,
        ["race_id", "fav_rank"],
    )
    for place in (False, True)
]
CASES += [
    case("committed", name, start, end, True, place, ["race_id", "odds"])
    for name, start, end in COMMITTED_READS
    for place in (False, True)
]


@pytest.mark.parametrize(("rule", "start", "end", "touches_price", "place_only", "columns"), CASES)
def test_read_store_table_opens_no_holdout_day_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rule: str,
    start: date,
    end: date,
    touches_price: bool,
    place_only: bool,
    columns: list[str],
) -> None:
    if rule == "stand-in":
        span, held = SPAN, SPAN_HELD
        stand_in(monkeypatch, held)
    else:
        span = committed_span()
        held = frozenset(d for d, h, _ in span if h)
        assert not committed_held(D(2026, 11, 30)) and committed_held(D(2026, 11, 29))
        assert committed_held(D(2026, 12, 1))
    rows = span_rows(span)
    path = written_store(tmp_path, monkeypatch)
    write_day_groups(path, rows)
    spy = ByteSpy(monkeypatch, path.parent)

    table = read_store_table(
        "final_odds",
        columns=columns,
        start=start,
        end=end,
        touches_price=touches_price,
        where=(ds.field("pool") == "PLACE") if place_only else None,
        source=SOURCE,
    )

    want = expected_rows(rows, held, start, end, place_only, columns)
    assert table.column_names == columns
    assert Counter(values(table, columns)) == Counter(want)
    result_days = Counter(
        r["race_date"]
        for r in rows
        if start <= r["race_date"] <= end
        and r["race_date"] not in held
        and (not place_only or r["pool"] == "PLACE")
    )
    spy.assert_isolated(
        "final_odds.parquet",
        held=held,
        start=start,
        end=end,
        columns=columns,
        result_days=result_days,
        only_day=start if start == end and start not in held else None,
    )


# ---------------------------------------------------------------------------------------------
# DoD: existing readers' results are unchanged on a store not yet rebuilt (a row group per year)
# (a regression guard, green before the change; shuffled mixed groups: test_development_scan)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "touches_price", "place_only"),
    [
        pytest.param(span_day(0), span_day(35), True, False, id="whole-span"),
        pytest.param(span_day(8), span_day(8), True, True, id="one-development-day-where"),
        pytest.param(date.min, date.max, False, False, id="date-min-to-date-max"),
    ],
)
def test_read_store_table_reads_a_store_with_a_row_group_per_year_as_before(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    start: date,
    end: date,
    touches_price: bool,
    place_only: bool,
) -> None:
    stand_in(monkeypatch, SPAN_HELD)
    rows = span_rows(SPAN)
    path = written_store(tmp_path, monkeypatch)
    write_year_groups(path, rows)
    columns = ["race_id", "pool", "combination", "odds", "investment"]

    table = read_store_table(
        "final_odds",
        columns=columns,
        start=start,
        end=end,
        touches_price=touches_price,
        where=(ds.field("pool") == "PLACE") if place_only else None,
        source=SOURCE,
    )

    want = expected_rows(rows, SPAN_HELD, start, end, place_only, columns)
    assert want
    assert table.column_names == columns
    assert Counter(values(table, columns)) == Counter(want)
