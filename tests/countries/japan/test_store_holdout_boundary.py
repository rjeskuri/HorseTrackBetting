"""Ticket #510: no shared study reader decodes a holdout-day row (D14): the access boundary.

Contract pinned here (the coder follows it):

- Readers: the shared development readers that scope a store table by race_date.
  `core.eval.holdout.read_development` (on a handle the caller opens with `pyarrow.OSFile`, as
  `research.spread_scan` does), `core.store.tables.read_table(..., development="japan")` (a range,
  when a study wants one, goes in `filters` beside its `where`), `models.v0.store.read_store_table`
  and `research.pool_depth._scan` (the pool-depth study's scan of pool_votes, final_odds, races
  and runners).
- Boundary: on a store whose row groups each hold one race day (race_date statistics, min == max,
  no null), with holdout days (#71's `is_holdout_day`, looked up at call time) interleaved with
  development days, a development read (open-ended to `date.max`, a range with holdout days
  inside it or at its ends, one development day, one holdout day, with or without a `where`)
  reads no byte of a holdout day's row group. The handle refuses (raises on) every read that
  overlaps a holdout group's column chunks, so a coalesced (pre-buffered) range across one, or a
  buffer of the whole file, fails. The library's own footer reads, calibrated on each file, are
  not row-group reads.
- The reads are seen: every development group holding a row of the result had each requested
  column's chunk read through that handle (the one the reader opens with `pyarrow.memory_map` or
  `pyarrow.OSFile`, looked up on the `pyarrow` module at call time), so a reader that reaches the
  file by another route fails.
- Output: exactly the development rows in range that satisfy `where`, the columns asked for in
  their order.
- The jra store (fix round). The same boundary holds on `processed/jra/{races,runners}.parquet`
  as the core build writes them (`core.store.build._write_tables`, from frames shaped as the build
  hands them over) for the readers that read it: `models.v0.store.read_store_table` (its default
  source, `jra`) and `core.store.tables.read_table("jra", ..., development="japan")`; in the
  build's race_id order (a day's rows contiguous, venues interleaved within it) and in venue
  order (a day's rows in one run per venue, so a holdout day spans several groups among the
  development days'). Each file must show a row group per race day (statistics min == max, no
  null): the build's writer is what gives the readers groups to skip. A jra store written the old
  way (one row group per table) reads as before, exactly the development rows; no byte-level
  claim is made there (`test_store_build_day_groups` pins the writer itself).

Not covered here: `studies.avoidance.read_development`'s dividends read (no race_date column, and
its writer does not keep a row group per day).

All data is synthetic, written under tmp_path, with HORSETRACK_DATA_DIR pointing there. Holdout
days are a stand-in for `is_holdout_day` (an explicit set of days), and in a few reads the
committed rule over synthetic days from its start, recomputed here with hashlib. The holdout is
never opened.
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

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import yaml

import horsetrack.core.eval.holdout as holdout_module
from horsetrack.core import paths
from horsetrack.core.store import build, schemas, tables
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table
from horsetrack.countries.japan.research import pool_depth

COUNTRY = "japan"
SOURCE = "jravan"
TABLE = "runners"
REPO_ROOT = Path(__file__).resolve().parents[3]
JAPAN_HOLDOUT = REPO_ROOT / "src" / "horsetrack" / "countries" / "japan" / "config" / "holdout.yaml"
FOOTER_READ = 64 * 1024  # pyarrow's speculative footer read
COALESCE_HOLE = 8 * 1024  # pyarrow's pre-buffer bridges holes up to this size

# The store's columns, typed as the store declares them (so `tables.read_table` reads them as is).
COLUMNS = (
    "jurisdiction", "race_id", "race_date", "venue", "race_no", "horse_id", "horse_no",
    "finish_pos", "win_odds", "finish_time_s",
)  # fmt: skip
SCHEMA = pa.schema([(c, tables.ARROW_TYPES[schemas.TABLES[TABLE].columns[c]]) for c in COLUMNS])
VENUES = ("TOK", "NAK", "KYO")  # race r runs at VENUES[r - 1]
HORSES = 10


def plain(value: object) -> object:
    return round(value, 6) if isinstance(value, float) else value


# ---------------------------------------------------------------------------------------------
# The synthetic store: one row group per race day
# ---------------------------------------------------------------------------------------------


def day_rows(day: date, k: int, races: int) -> list[dict]:
    """Store-shaped runner rows of one race day (the k-th of its span), values set by the day,
    the race and the horse."""
    rows = []
    for r in range(1, races + 1):
        venue = VENUES[r - 1]
        race_id = f"jra:{day.isoformat()}:{venue}:{r:02d}"
        for h in range(1, HORSES + 1):
            rows.append({"jurisdiction": "jra", "race_id": race_id,
                         "race_date": day, "venue": venue, "race_no": r,
                         "horse_id": f"H{k:03d}{r}{h:02d}", "horse_no": h,
                         "finish_pos": (h + k + r) % HORSES + 1,
                         "win_odds": round(1.5 + 0.9 * h + 0.01 * k + 0.1 * r, 2),
                         "finish_time_s": round(60.0 + 0.1 * h + k + 0.5 * r, 1)})  # fmt: skip
    return rows


@dataclass(frozen=True)
class Layout:
    """Each row group's column chunks (byte ranges, from the footer), its byte extent and its race
    days; and the library's own footer reads of the file."""

    chunks: tuple[dict[str, tuple[int, int]], ...]
    extents: tuple[tuple[int, int], ...]
    days: tuple[frozenset[date], ...]
    footer: frozenset[tuple[int, int]]

    def overlapping(self, lo: int, hi: int, groups: Sequence[int]) -> list[int]:
        return [g for g in groups if lo < self.extents[g][1] and hi > self.extents[g][0]]

    def describe(self, groups: Sequence[int]) -> str:
        return ", ".join(
            f"group {g} ({', '.join(d.isoformat() for d in sorted(self.days[g]))})" for g in groups
        )


class CountingFile(io.BytesIO):
    """A Parquet file in memory that records the byte range of every read made of it. With a
    `guard`, a read the guard refuses raises before its bytes are handed back."""

    def __init__(self, data: bytes, guard: Callable[[int, int], None] | None = None) -> None:
        super().__init__(data)
        self.data = data
        self.guard = guard
        self.reads: list[tuple[int, int]] = []

    def _seen(self, lo: int, hi: int) -> None:
        self.reads.append((lo, hi))
        if self.guard is not None and hi > lo:
            self.guard(lo, hi)

    def read(self, size=-1):
        at = self.tell()
        out = super().read(size)
        self._seen(at, at + len(out))
        return out

    def readinto(self, buffer):
        at = self.tell()
        n = super().readinto(buffer)
        self._seen(at, at + n)
        return n


def footer_reads(data: bytes) -> frozenset[tuple[int, int]]:
    """The reads pyarrow makes to open the file's footer, by the two routes the readers open
    metadata through (a ParquetFile, a dataset fragment). They read no row group's chunks for a
    reader: the tests below also ask for each development chunk of the result to be read."""
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


def file_layout(data: bytes) -> Layout:
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
    return Layout(tuple(chunks), tuple(extents), tuple(days), footer_reads(data))


@dataclass(frozen=True)
class Store:
    """A data dir holding `processed/<source>/<table>.parquet` (by default jravan's runners), its
    rows (store order), its layout and its holdout days (`held`)."""

    root: Path
    rows: tuple[dict, ...]
    held: frozenset[date]
    days: tuple[date, ...]
    layout: Layout
    source: str = SOURCE
    table: str = TABLE

    @property
    def folder(self) -> Path:
        return self.root / "processed" / self.source

    @property
    def path(self) -> Path:
        return self.folder / f"{self.table}.parquet"


def write_store(root: Path, span: Sequence[tuple[date, bool, int]]) -> Store:
    """`span` (day, holdout, races) written as the store's runners table, one row group per race
    day, uncompressed and without dictionaries: every group a few KiB (under pyarrow's pre-buffer
    hole), the file well over pyarrow's footer read."""
    rows = [row for k, (day, _, races) in enumerate(span) for row in day_rows(day, k, races)]
    path = root / "processed" / SOURCE / f"{TABLE}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    days = [day for day, _, _ in span]
    with pq.ParquetWriter(path, SCHEMA, compression="none", use_dictionary=False) as writer:
        for day in days:
            part = table.filter(ds.field("race_date") == pa.scalar(day, pa.date32()))
            writer.write_table(part, row_group_size=part.num_rows)
    data = path.read_bytes()
    layout = file_layout(data)
    held = frozenset(day for day, holdout, _ in span if holdout)

    # The store the boundary is claimed on: a day per row group, recorded in its statistics.
    meta = pq.read_metadata(path)
    assert meta.num_row_groups == len(days)
    for g, day in enumerate(days):
        stats = meta.row_group(g).column(COLUMNS.index("race_date")).statistics
        assert stats.has_min_max and stats.min == stats.max == day, (g, day)
        assert stats.has_null_count and stats.null_count == 0, (g, day)
        assert layout.days[g] == {day}
    sizes = [hi - lo for lo, hi in layout.extents]
    assert max(sizes) < COALESCE_HOLE // 2, "every row group is a few KiB"
    assert len(data) > 2 * FOOTER_READ, "the file is well over pyarrow's footer read"
    assert min(lo for lo, _ in layout.footer) > layout.extents[0][1], (
        "the footer read must not cover the first row group"
    )
    return Store(root, tuple(rows), held, tuple(days), layout)


# The stand-in's span: 40 consecutive days from 2025-01-04, D a development day, H a holdout day
# (the first a development day, the last a holdout day, runs of both). A development day runs two
# or three races, a holdout day one or two.
PATTERN = "DHDHHDDHDHDDDHHDHDDHDHHHDDHDHDHDDHDHDDHH"
STAND_IN_SPAN = tuple(
    (date(2025, 1, 4) + timedelta(days=k), mark == "H", (2 if mark == "D" else 1) + k % 2)
    for k, mark in enumerate(PATTERN)
)


def day_at(k: int) -> date:
    return STAND_IN_SPAN[k][0]


# The reads below rely on these marks.
assert len(PATTERN) == 40 and PATTERN[0] == "D" and PATTERN[-1] == "H"
assert PATTERN[3] == PATTERN[23] == PATTERN[21] == "H" and PATTERN[7] == PATTERN[9] == "H"
assert PATTERN[2] == PATTERN[8] == PATTERN[20] == PATTERN[31] == "D"


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


COMMITTED_START = date(2026, 11, 28)
COMMITTED_DAYS = tuple(COMMITTED_START - timedelta(days=1) + timedelta(days=k) for k in range(40))


def committed_span() -> tuple[tuple[date, bool, int], ...]:
    span = tuple(
        (d, committed_held(d), (1 if committed_held(d) else 2) + k % 2)
        for k, d in enumerate(COMMITTED_DAYS)
    )
    marks = [h for _, h, _ in span]
    assert _committed_rule()[0] == COMMITTED_START
    assert any(marks) and not all(marks) and not marks[0], "holdout days interleave"
    assert any(not a and b and not c for a, b, c in zip(marks, marks[1:], marks[2:], strict=False))
    return span


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Store]:
    return {
        "stand-in": write_store(tmp_path_factory.mktemp("stand-in").resolve(), STAND_IN_SPAN),
        "committed": write_store(tmp_path_factory.mktemp("committed").resolve(), committed_span()),
    }


# ---------------------------------------------------------------------------------------------
# The handle spy: every read-only open of the store file, served from a guarded copy
# ---------------------------------------------------------------------------------------------


class HoldoutBytesRead(OSError):
    """A read reached a row group holding a holdout day."""


class ByteSpy:
    """Serves every read-only open of the store's file by `pyarrow.memory_map` or
    `pyarrow.OSFile` from an in-memory copy that records each read's byte range and refuses
    (raises on) every read, other than the library's footer reads, overlapping a holdout day's
    row group."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, store: Store) -> None:
        self.store = store
        self.layout = store.layout
        self.forbidden = [g for g, days in enumerate(self.layout.days) if days & store.held]
        assert self.forbidden, "the store holds holdout days"
        self.copies: list[CountingFile] = []
        self.refused: list[tuple[tuple[int, int], list[int]]] = []
        for name in ("memory_map", "OSFile"):
            monkeypatch.setattr(pa, name, self._serve(getattr(pa, name)))

    def _guard(self, lo: int, hi: int) -> None:
        if (lo, hi) in self.layout.footer:
            return
        hit = self.layout.overlapping(lo, hi, self.forbidden)
        if hit:
            self.refused.append(((lo, hi), hit))
            raise HoldoutBytesRead(
                f"bytes {lo}..{hi} read from {self.layout.describe(hit)}, holding holdout days"
            )

    def _serve(self, real):
        def open_(path, mode="r", *args, **kwargs):
            if not isinstance(path, (str, os.PathLike)) or mode not in ("r", "rb"):
                return real(path, mode, *args, **kwargs)
            target = Path(path).resolve()
            if target != self.store.path:
                return real(path, mode, *args, **kwargs)
            copy = CountingFile(target.read_bytes(), guard=self._guard)
            self.copies.append(copy)
            return pa.PythonFile(copy, mode="r")

        return open_

    def assert_boundary(
        self,
        error: BaseException | None,
        columns: Sequence[str],
        result_days: Counter,
        result_groups: set[int] | None = None,
    ) -> None:
        """No read reached a holdout group; the reader opened the file through the spied
        openers; every development group holding a result row (`result_groups`, or else every
        group holding a day of `result_days`) had each of `columns`' chunks read through the
        handle."""
        refused = "; ".join(
            f"bytes {lo}..{hi} over {self.layout.describe(groups)}"
            for (lo, hi), groups in self.refused
        )
        assert not self.refused, (
            f"the development read reached row groups holding holdout days: {refused}"
            + (f" (the refusal reached the caller as {type(error).__name__})" if error else "")
        )
        if error is not None:
            raise error
        assert self.copies, (
            f"the reader did not open {self.store.path.name} through pyarrow.memory_map or "
            "pyarrow.OSFile: its reads cannot be seen"
        )
        reads = [
            r for f in self.copies for r in f.reads if r[1] > r[0] and r not in self.layout.footer
        ]
        for g, days in enumerate(self.layout.days):
            holds = (
                any(result_days[d] for d in days) if result_groups is None else g in result_groups
            )
            if not holds:
                continue
            for column in columns:
                lo, hi = self.layout.chunks[g][column]
                assert any(a < hi and b > lo for a, b in reads), (
                    f"{self.layout.describe([g])} holds rows of the result, but its {column!r} "
                    "chunk was not read through the handle opened with pyarrow.memory_map/OSFile: "
                    "the rows reached the reader by another route"
                )


def stand_in(monkeypatch: pytest.MonkeyPatch, held: frozenset[date]) -> list[date]:
    """Make exactly `held` the holdout days (`development_filter` looks `is_holdout_day` up at
    call time); the days asked about are recorded."""
    asked: list[date] = []

    def is_holdout_day(country: str, day: date) -> bool:
        asked.append(day)
        return country == COUNTRY and day in held

    monkeypatch.setattr(holdout_module, "is_holdout_day", is_holdout_day)
    return asked


# ---------------------------------------------------------------------------------------------
# The readers, each called as a study calls it
# ---------------------------------------------------------------------------------------------


def _table_rows(table: pa.Table, columns: Sequence[str]) -> tuple[list[str], list[tuple]]:
    cells = [table.column(c).to_pylist() for c in columns]
    return table.column_names, [tuple(plain(v) for v in row) for row in zip(*cells, strict=True)]


def _frame_rows(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[list[str], list[tuple]]:
    cells = [frame[c].tolist() for c in columns]
    return list(frame.columns), [tuple(plain(v) for v in row) for row in zip(*cells, strict=True)]


def _span(start: date, end: date) -> ds.Expression | None:
    day = ds.field("race_date")
    parts = []
    if start > date.min:
        parts.append(day >= pa.scalar(start, pa.date32()))
    if end < date.max:
        parts.append(day <= pa.scalar(end, pa.date32()))
    out = None
    for part in parts:
        out = part if out is None else out & part
    return out


def via_read_development(store, columns, start, end, where):
    # as research.spread_scan reads final_odds: a handle opened with pyarrow.OSFile
    with pa.OSFile(str(store.path)) as handle:
        table = holdout_module.read_development(
            handle, COUNTRY, columns=list(columns), start=start, end=end, where=where
        )
    return _table_rows(table, columns)


def via_store_tables(store, columns, start, end, where):
    span = _span(start, end)
    filters = where if span is None else (span if where is None else span & where)
    frame = tables.read_table(SOURCE, TABLE, list(columns), filters=filters, development=COUNTRY)
    return _frame_rows(frame, columns)


def via_read_store_table(store, columns, start, end, where):
    table = read_store_table(
        TABLE,
        columns=list(columns),
        start=start,
        end=end,
        touches_price=start >= PRICE_ERA_START,
        where=where,
        source=SOURCE,
    )
    return _table_rows(table, columns)


def via_pool_depth_scan(store, columns, start, end, where):
    start = max(start, PRICE_ERA_START)  # the scan refuses a span before D10's price era
    table = pool_depth._scan(TABLE, list(columns), start, end, where=where)
    return _table_rows(table, columns)


READERS = {
    "holdout.read_development": via_read_development,
    "store.tables.read_table": via_store_tables,
    "v0.store.read_store_table": via_read_store_table,
    "pool_depth._scan": via_pool_depth_scan,
}


# ---------------------------------------------------------------------------------------------
# DoD: no shared study reader reads a holdout day's row group; it returns the development rows
# ---------------------------------------------------------------------------------------------

WHERES: dict[str, tuple[ds.Expression | None, Callable[[dict], bool]]] = {
    "none": (None, lambda r: True),
    "finish-pos-le-3": (ds.field("finish_pos") <= 3, lambda r: r["finish_pos"] <= 3),
    "venue-nak": (ds.field("venue") == "NAK", lambda r: r["venue"] == "NAK"),
    "venue-tok": (ds.field("venue") == "TOK", lambda r: r["venue"] == "TOK"),
    "finish-pos-le-5": (ds.field("finish_pos") <= 5, lambda r: r["finish_pos"] <= 5),
    "race-no-ge-3": (ds.field("race_no") >= 3, lambda r: r["race_no"] >= 3),
}


@dataclass(frozen=True)
class Read:
    rule: str  # "stand-in" or "committed"
    start: date
    end: date
    where: str
    columns: tuple[str, ...]


READS = {
    "open-ended": Read("stand-in", date.min, date.max, "none", ("race_id", "horse_no", "win_odds")),
    "open-ended-where-unread-column": Read(
        "stand-in", date.min, date.max, "finish-pos-le-3", ("race_id", "finish_time_s")
    ),
    "holdout-day-to-holdout-day-where": Read(
        "stand-in", day_at(3), day_at(23), "venue-nak", ("horse_id", "race_date", "finish_pos")
    ),
    "development-day-to-development-day": Read(
        "stand-in", day_at(2), day_at(20), "none", ("race_no", "win_odds")
    ),
    "development-day-to-date-max-where": Read(
        "stand-in", day_at(31), date.max, "finish-pos-le-5", ("finish_time_s",)
    ),
    "one-development-day-between-holdout-days": Read(
        "stand-in", day_at(8), day_at(8), "none", ("race_id", "venue", "horse_no")
    ),
    "one-holdout-day": Read("stand-in", day_at(21), day_at(21), "none", ("race_id",)),
    "committed-rule-whole-span": Read(
        "committed", COMMITTED_DAYS[0], COMMITTED_DAYS[-1], "none", ("race_id", "win_odds")
    ),
    "committed-rule-holdout-start-to-date-max-where": Read(
        "committed", COMMITTED_START, date.max, "venue-tok", ("horse_id", "finish_pos")
    ),
}


@pytest.mark.parametrize("read_id", list(READS))
@pytest.mark.parametrize("reader", list(READERS))
def test_development_read_touches_no_holdout_row_group(
    stores: dict[str, Store], monkeypatch: pytest.MonkeyPatch, reader: str, read_id: str
) -> None:
    read = READS[read_id]
    store = stores[read.rule]
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(store.root))
    asked = stand_in(monkeypatch, store.held) if read.rule == "stand-in" else None
    where, keep = WHERES[read.where]
    spy = ByteSpy(monkeypatch, store)

    error = None
    try:
        names, got = READERS[reader](store, read.columns, read.start, read.end, where)
    except Exception as exc:  # judged below: a refused holdout read first
        error = exc

    kept = [
        r
        for r in store.rows
        if read.start <= r["race_date"] <= read.end and r["race_date"] not in store.held and keep(r)
    ]
    spy.assert_boundary(error, read.columns, Counter(r["race_date"] for r in kept))
    assert names == list(read.columns)
    assert Counter(got) == Counter(tuple(plain(r[c]) for c in read.columns) for r in kept)
    if read_id != "one-holdout-day":
        assert kept, "the read returns development rows"
    if asked is not None:
        assert asked, "the holdout rule was not consulted"


# ---------------------------------------------------------------------------------------------
# The jra store (netkeiba): races and runners as the core build's writer writes them
# ---------------------------------------------------------------------------------------------

JRA = "jra"  # models.v0.store.read_store_table's default source
JRA_TABLES = ("races", "runners")
JRA_VENUE_SETS = (("KYO", "TOK"), ("KYO", "NAK", "TOK"), ("NAK",), ("NAK", "TOK"))  # by k % 4
JRA_HORSES = 6
JRA_ORDERS: dict[str, Callable[[dict], tuple]] = {
    # the build's order (the declared sort keys): a day's rows contiguous, its venues within it
    "race-id": lambda r: (r["race_id"], r.get("horse_no", 0)),
    # venue first: a day's rows in one run per venue, so a day spans several row groups
    "venue": lambda r: (r["venue"], r["race_date"], r["race_no"], r.get("horse_no", 0)),
}
_PANDAS_TYPES = {pa.int64(): pd.Int64Dtype(), pa.bool_(): pd.BooleanDtype()}


def declared_schema(name: str) -> pa.Schema:
    return pa.schema([(c, tables.ARROW_TYPES[k]) for c, k in schemas.TABLES[name].columns.items()])


def hexed(*parts: object) -> str:
    """64 hex characters no compression shrinks: the files stay well over the footer read."""
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def jra_rows(span: Sequence[tuple[date, bool, int]]) -> dict[str, list[dict]]:
    """Store-shaped races and runners of `span`'s days (the k-th day races at
    JRA_VENUE_SETS[k % 4], two to four races a venue), values set by the day, venue, race and
    horse."""
    races, runners = [], []
    for k, (day, _, _) in enumerate(span):
        for v, venue in enumerate(JRA_VENUE_SETS[k % len(JRA_VENUE_SETS)]):
            for r in range(1, 2 + (k + v) % 3 + 1):
                rid = f"{JRA}:{day.isoformat()}:{venue}:{r:02d}"
                keys = {"jurisdiction": JRA, "race_id": rid,
                        "meeting_id": f"{JRA}:{day.isoformat()}:{venue}", "race_date": day,
                        "venue": venue, "race_no": r, "season": str(day.year)}  # fmt: skip
                races.append({
                    **keys, "race_index": 100 * k + 10 * v + r, "race_name": hexed("name", rid),
                    "race_class_raw": hexed("class", rid), "going_raw": hexed("going", rid),
                    "source_race_key": hexed("key", rid),
                    "distance_m": 1000 + 200 * ((k + v + r) % 8),
                    "surface": ("turf", "dirt")[(k + r) % 2], "field_size": JRA_HORSES,
                    "finish_time_s": round(58.0 + 0.7 * r + 0.13 * k + 0.05 * v, 2),
                    "split_times_s": [round(11.0 + ((7 * k + 3 * r + i) % 17) / 10, 1)
                                      for i in range(6)],
                    "status": "run",
                })  # fmt: skip
                for h in range(1, JRA_HORSES + 1):
                    runners.append({
                        **keys, "horse_id": f"J{k:03d}{v}{r}{h}", "horse_no": h,
                        "horse_name": hexed("horse", rid, h),
                        "jockey_name": hexed("jockey", rid, h),
                        "finish_pos": (h + k + r) % JRA_HORSES + 1,
                        "win_odds": round(1.5 + 0.9 * h + 0.01 * k + 0.1 * r, 2),
                        "finish_time_s": round(60.0 + 0.1 * h + 0.3 * k + 0.5 * r, 1),
                        "horse_wt_kg": 440 + (7 * h + k) % 60,
                        "running_positions": [(h + i) % JRA_HORSES + 1 for i in range(4)],
                    })  # fmt: skip
    return {"races": races, "runners": runners}


def frame_of(name: str, rows: Sequence[dict]) -> pd.DataFrame:
    """A table's frame as the build hands it to its writer: declared columns in order, pandas
    Int64 and boolean."""
    return pa.Table.from_pylist(list(rows), schema=declared_schema(name)).to_pandas(
        types_mapper=_PANDAS_TYPES.get
    )


def write_jra_store(
    root: Path, span: Sequence[tuple[date, bool, int]], order: str
) -> dict[str, Store]:
    """`span`'s races and runners, in `order`, written by the core build's writer into
    `processed/jra`. No layout is asserted here: the tests do."""
    rows = {name: sorted(got, key=JRA_ORDERS[order]) for name, got in jra_rows(span).items()}
    folder = root / "processed" / JRA
    build._write_tables({name: frame_of(name, rows[name]) for name in JRA_TABLES}, folder)
    held = frozenset(day for day, holdout, _ in span if holdout)
    days = tuple(day for day, _, _ in span)
    return {
        name: Store(
            root,
            tuple(rows[name]),
            held,
            days,
            file_layout((folder / f"{name}.parquet").read_bytes()),
            source=JRA,
            table=name,
        )  # fmt: skip
        for name in JRA_TABLES
    }


@pytest.fixture(scope="module")
def jra_stores(tmp_path_factory: pytest.TempPathFactory) -> dict[tuple[str, str], dict]:
    spans = {"stand-in": STAND_IN_SPAN, "committed": committed_span()}
    return {
        (rule, order): write_jra_store(
            tmp_path_factory.mktemp(f"jra-{rule}-{order}").resolve(), span, order
        )
        for rule, span in spans.items()
        for order in JRA_ORDERS
    }


def assert_a_row_group_per_day(store: Store) -> None:
    """Every row group of the store's file holds one race day, recorded in its race_date
    statistics (min == max, no null)."""
    meta = pq.read_metadata(store.path)
    at = meta.schema.names.index("race_date")
    name = f"processed/{store.source}/{store.path.name}"
    for g in range(meta.num_row_groups):
        days = sorted(store.layout.days[g])
        assert len(days) == 1, (
            f"{name}: row group {g} of {meta.num_row_groups} holds {len(days)} race days "
            f"({days[0]} .. {days[-1]}), not one: the core build's writer must write a row group "
            "per race day for a reader to skip a holdout day's rows unread"
        )
        stats = meta.row_group(g).column(at).statistics
        assert stats is not None and stats.has_min_max, f"{name}: group {g} has no min/max"
        assert stats.min == stats.max == days[0], (name, g, stats.min, stats.max)
        assert stats.has_null_count and stats.null_count == 0, (name, g, stats.null_count)
    assert set().union(*store.layout.days) == set(store.days), f"{name}: not every day written"


def assert_reads_can_be_seen(store: Store) -> None:
    """The synthetic file is large enough for the claim: the library's footer reads leave
    holdout and development row groups wholly unread, so the spy sees a reader's reads of them."""
    layout = store.layout
    assert store.path.stat().st_size > 2 * FOOTER_READ
    first_footer = min(lo for lo, _ in layout.footer)
    clear = [g for g, (_, hi) in enumerate(layout.extents) if hi <= first_footer]
    assert any(layout.days[g] & store.held for g in clear), "no holdout group before the footer"
    assert any(not layout.days[g] & store.held for g in clear), "no development group before it"


def via_jra_read_store_table(store, columns, start, end, where):
    table = read_store_table(  # the default source: jra
        store.table,
        columns=list(columns),
        start=start,
        end=end,
        touches_price=start >= PRICE_ERA_START,
        where=where,
    )
    return _table_rows(table, columns)


def via_jra_store_tables(store, columns, start, end, where):
    span = _span(start, end)
    filters = where if span is None else (span if where is None else span & where)
    frame = tables.read_table(JRA, store.table, list(columns), filters=filters, development=COUNTRY)
    return _frame_rows(frame, columns)


JRA_READERS = {
    "v0.store.read_store_table": via_jra_read_store_table,
    "store.tables.read_table": via_jra_store_tables,
}


@dataclass(frozen=True)
class JraRead:
    rule: str  # "stand-in" or "committed"
    start: date
    end: date
    where: str
    columns: dict[str, tuple[str, ...]]  # per table


JRA_READS = {
    "open-ended": JraRead(
        "stand-in", date.min, date.max, "none",
        {"races": ("race_id", "distance_m", "finish_time_s"),
         "runners": ("race_id", "horse_no", "win_odds")},
    ),
    "holdout-day-to-holdout-day-where": JraRead(
        "stand-in", day_at(3), day_at(23), "venue-nak",
        {"races": ("race_id", "race_date", "race_name"),
         "runners": ("horse_id", "race_date", "finish_pos")},
    ),
    "development-day-to-date-max-where-unread-column": JraRead(
        "stand-in", day_at(31), date.max, "race-no-ge-3",
        {"races": ("source_race_key", "field_size"), "runners": ("horse_name", "finish_time_s")},
    ),
    "one-development-day-between-holdout-days": JraRead(
        "stand-in", day_at(8), day_at(8), "none",
        {"races": ("race_id", "venue", "race_no"), "runners": ("race_id", "venue", "horse_no")},
    ),
    "one-holdout-day": JraRead(
        "stand-in", day_at(21), day_at(21), "none", {"races": ("race_id",), "runners": ("race_id",)}
    ),
    "committed-rule-holdout-start-to-date-max-where": JraRead(
        "committed", COMMITTED_START, date.max, "venue-tok",
        {"races": ("race_id", "distance_m"), "runners": ("horse_id", "win_odds")},
    ),
}  # fmt: skip


def wanted(store: Store, read: JraRead) -> Callable[[dict], bool]:
    """A row of the read's result: a development day in range that satisfies `where`."""
    keep = WHERES[read.where][1]
    return lambda r: (
        read.start <= r["race_date"] <= read.end and r["race_date"] not in store.held and keep(r)
    )


def development_rows(store: Store, read: JraRead) -> list[dict]:
    return [r for r in store.rows if wanted(store, read)(r)]


def groups_holding(store: Store, read: JraRead) -> set[int]:
    """The row groups of the file holding a row of the read's result, from the file's own rows
    (a day may span several groups, of which a `where` can rule some out)."""
    file = pq.ParquetFile(io.BytesIO(store.path.read_bytes()))
    row_of = wanted(store, read)
    return {
        g
        for g in range(file.metadata.num_row_groups)
        if any(row_of(r) for r in file.read_row_group(g).to_pylist())
    }


@pytest.mark.parametrize("read_id", list(JRA_READS))
@pytest.mark.parametrize("table", JRA_TABLES)
@pytest.mark.parametrize("order", list(JRA_ORDERS))
@pytest.mark.parametrize("reader", list(JRA_READERS))
def test_jra_development_read_touches_no_holdout_row_group(
    jra_stores: dict, monkeypatch: pytest.MonkeyPatch, reader: str, order: str, table: str,
    read_id: str,
) -> None:  # fmt: skip
    read = JRA_READS[read_id]
    store = jra_stores[(read.rule, order)][table]
    columns = read.columns[table]
    assert_a_row_group_per_day(store)
    assert_reads_can_be_seen(store)
    if order == "venue":  # a holdout day in several groups, among development days' groups
        spans = Counter(d for days in store.layout.days for d in days)
        assert any(spans[d] > 1 for d in store.held)
    result_groups = groups_holding(store, read)
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(store.root))
    asked = stand_in(monkeypatch, store.held) if read.rule == "stand-in" else None
    spy = ByteSpy(monkeypatch, store)

    error = None
    try:
        names, got = JRA_READERS[reader](
            store, columns, read.start, read.end, WHERES[read.where][0]
        )
    except Exception as exc:  # judged below: a refused holdout read first
        error = exc

    kept = development_rows(store, read)
    spy.assert_boundary(error, columns, Counter(r["race_date"] for r in kept), result_groups)
    assert names == list(columns)
    assert Counter(got) == Counter(tuple(plain(r[c]) for c in columns) for r in kept)
    if read_id != "one-holdout-day":
        assert kept, "the read returns development rows"
    if asked is not None:
        assert asked, "the holdout rule was not consulted"


@pytest.mark.parametrize("table", JRA_TABLES)
@pytest.mark.parametrize("reader", list(JRA_READERS))
def test_a_jra_store_written_a_row_group_per_table_reads_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str, table: str
) -> None:
    """The jra store as the build wrote it before #510 (`pq.write_table`'s default: one row group
    holding every day): exactly the development rows; no byte-level claim."""
    root = tmp_path.resolve()
    rows = sorted(jra_rows(STAND_IN_SPAN)[table], key=JRA_ORDERS["race-id"])
    path = root / "processed" / JRA / f"{table}.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=declared_schema(table)), path)
    assert pq.read_metadata(path).num_row_groups == 1
    held = frozenset(day for day, holdout, _ in STAND_IN_SPAN if holdout)
    store = Store(root, tuple(rows), held, (), file_layout(path.read_bytes()), JRA, table)
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(root))
    stand_in(monkeypatch, held)

    read = JRA_READS["holdout-day-to-holdout-day-where"]
    columns = read.columns[table]
    names, got = JRA_READERS[reader](store, columns, read.start, read.end, WHERES[read.where][0])
    kept = development_rows(store, read)
    assert kept and {r["race_date"] for r in kept}.isdisjoint(held)
    assert names == list(columns)
    assert Counter(got) == Counter(tuple(plain(r[c]) for c in columns) for r in kept)
