"""Ticket #415: holdout days are excluded inside the data scan, not dropped after it.

Contract pinned here, for the two development readers:

- ``horsetrack.countries.japan.models.v0.store.read_store_table`` and
  ``horsetrack.countries.japan.studies.speed_run.read_inputs`` return no row on a holdout day
  (#71's committed rule, ``core.eval.holdout``) and every development row in range, the read
  range inclusive at both ends, with the columns asked for in their order and the store copy
  named by ``source``.
- No holdout row is materialized by the scan either: every table or record batch the pyarrow
  scan produces already excludes the holdout days, so the predicate is applied by the scan
  (built from the calendar, not from the rows read). The spy below wraps the two pyarrow entry
  points a reader scans through, ``pyarrow.dataset.dataset`` (with what it returns: its
  ``to_table``, ``to_batches``, ``head``, ``take``, ``scanner``, ``filter``, ``get_fragments``,
  ``Scanner.from_dataset``) and ``pyarrow.parquet.read_table`` (which ``pandas.read_parquet``
  uses), and checks every table and batch they hand back. It also wraps the row reads of
  ``pyarrow.parquet.ParquetFile`` (``read``, ``read_row_group(s)``, ``iter_batches``),
  ``pyarrow.parquet.ParquetDataset.read(_pandas)`` and ``pyarrow.parquet.read_pandas``, so finding
  a table's date span may not materialize holdout dates either (footer statistics or a row count
  are not rows). A read that scans by another route is not seen and fails the "a scan was seen"
  check.
- An open-ended read (``end=date.max``, the pattern ten callers use, or ``start=date.min``) asks
  the holdout rule about the days the table can hold, not about every day of the requested range
  (PR #427 review): its development rows come back within a small budget of ``is_holdout_day``
  calls, and fast. A Parquet file written without race_date statistics is still read in full and
  without a leak.
- ``read_inputs`` reads one snapshot of each table: a table atomically rebuilt while the read is
  under way comes back either as it was or as it is, never a mixture of the two.

The store is synthetic: every calendar day from 2026-11-20 to 2026-12-21, so development and
holdout days interleave from the holdout start (2026-11-28, a development day) on, and the rows
are shuffled into small, mixed row groups. The oracle recomputes the rule from the committed
holdout.yaml with hashlib; it never calls the code under test. The holdout is never opened.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import yaml

import horsetrack.core.eval.holdout as holdout_module
from horsetrack.core.store import schemas, tables
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table
from horsetrack.countries.japan.studies.speed_run import RACE_COLUMNS, RUNNER_COLUMNS, read_inputs

REPO_ROOT = Path(__file__).resolve().parents[3]
JAPAN_YAML = REPO_ROOT / "src" / "horsetrack" / "countries" / "japan" / "config" / "holdout.yaml"
DATA_VAR = "HORSETRACK_DATA_DIR"


def _definition() -> tuple[date, str]:
    section = yaml.safe_load(JAPAN_YAML.read_text(encoding="utf-8"))["holdout"]
    start = section["start"]
    start = start if isinstance(start, date) else date.fromisoformat(str(start))
    return start, str(section["seed"])


HOLDOUT_START, SEED = _definition()


def held(day: date) -> bool:
    digest = hashlib.sha256(f"{SEED}:{day.isoformat()}".encode()).hexdigest()
    return day >= HOLDOUT_START and int(digest, 16) % 2 == 0


DAYS = [date(2026, 11, 20) + timedelta(days=k) for k in range(32)]  # 2026-11-20 .. 2026-12-21


# The fixture's ranges rely on these classifications of the committed rule.
assert HOLDOUT_START == date(2026, 11, 28) and not held(HOLDOUT_START)
assert all(held(date(2026, m, d)) for m, d in [(11, 29), (12, 1), (12, 3), (12, 4), (12, 9)])
assert not any(held(date(2026, m, d)) for m, d in [(11, 23), (11, 30), (12, 5), (12, 12)])
assert held(DAYS[-1])
# 2026-11-23 hashes even: it is development only because it is before the start
assert int(hashlib.sha256(f"{SEED}:2026-11-23".encode()).hexdigest(), 16) % 2 == 0


# ---------------------------------------------------------------------------------------------
# The scan spy
# ---------------------------------------------------------------------------------------------


def day_of(race_id: str) -> date:
    return date.fromisoformat(race_id.split(":")[1])


class ScanSpy:
    """Every table and record batch the wrapped pyarrow scans hand back."""

    def __init__(self) -> None:
        self.outputs: list[pa.Table | pa.RecordBatch] = []

    def record(self, out):
        if isinstance(out, (pa.Table, pa.RecordBatch)):
            self.outputs.append(out)
        return out

    def leaks(self, n_development: int) -> list[str]:
        """What each scan output shows of holdout rows: their days, or a row count above the
        development rows when the output holds neither race_date nor race_id."""
        found = []
        for out in self.outputs:
            columns = out.to_pydict()
            if "race_date" in columns:
                days = [d for d in columns["race_date"] if d is not None]
            elif "race_id" in columns:
                days = [day_of(r) for r in columns["race_id"] if r is not None]
            else:
                if out.num_rows > n_development:
                    found.append(f"{out.num_rows} rows scanned for {n_development} development")
                continue
            found += sorted({d.isoformat() for d in days if held(d)})
        return found


class Spied:
    """A pyarrow dataset, scanner or fragment whose materializations the spy records."""

    _TABLES = ("to_table", "head", "take")
    _BATCHES = ("to_batches",)
    _WRAPPED = ("scanner", "filter", "replace_schema")

    def __init__(self, real, spy: ScanSpy) -> None:
        self._real = real
        self._spy = spy

    def __getattr__(self, name):
        attr = getattr(self._real, name)
        spy = self._spy
        if name in self._TABLES:
            return lambda *a, **kw: spy.record(attr(*a, **kw))
        if name in self._BATCHES:
            return lambda *a, **kw: (spy.record(b) for b in attr(*a, **kw))
        if name in self._WRAPPED:
            return lambda *a, **kw: Spied(attr(*a, **kw), spy)
        if name == "get_fragments":
            return lambda *a, **kw: [Spied(f, spy) for f in attr(*a, **kw)]
        if name == "to_reader":

            def reader(*a, **kw):
                real = attr(*a, **kw)
                return pa.RecordBatchReader.from_batches(real.schema, (spy.record(b) for b in real))

            return reader
        if name == "scan_batches":

            def tagged(*a, **kw):
                for item in attr(*a, **kw):
                    spy.record(item.record_batch)
                    yield item

            return tagged
        return attr


def _unwrap(obj):
    return obj._real if isinstance(obj, Spied) else obj


@pytest.fixture
def spy(monkeypatch) -> ScanSpy:
    watcher = ScanSpy()
    real_dataset, real_read_table, real_scanner = ds.dataset, pq.read_table, ds.Scanner

    def dataset(*a, **kw):
        return Spied(real_dataset(*a, **kw), watcher)

    def read_table(*a, **kw):
        return watcher.record(real_read_table(*a, **kw))

    class Scanner:
        @staticmethod
        def from_dataset(dataset, *a, **kw):
            return Spied(real_scanner.from_dataset(_unwrap(dataset), *a, **kw), watcher)

        @staticmethod
        def from_fragment(fragment, *a, **kw):
            return Spied(real_scanner.from_fragment(_unwrap(fragment), *a, **kw), watcher)

        @staticmethod
        def from_batches(*a, **kw):
            return Spied(real_scanner.from_batches(*a, **kw), watcher)

    real_file, real_pq_dataset, real_read_pandas = pq.ParquetFile, pq.ParquetDataset, pq.read_pandas

    class ParquetFile(real_file):
        def read(self, *a, **kw):
            return watcher.record(super().read(*a, **kw))

        def read_row_group(self, *a, **kw):
            return watcher.record(super().read_row_group(*a, **kw))

        def read_row_groups(self, *a, **kw):
            return watcher.record(super().read_row_groups(*a, **kw))

        def iter_batches(self, *a, **kw):
            return (watcher.record(b) for b in super().iter_batches(*a, **kw))

    class ParquetDataset(real_pq_dataset):
        def read(self, *a, **kw):
            return watcher.record(super().read(*a, **kw))

        def read_pandas(self, *a, **kw):
            return watcher.record(super().read_pandas(*a, **kw))

    def read_pandas(*a, **kw):
        return watcher.record(real_read_pandas(*a, **kw))

    monkeypatch.setattr(ds, "dataset", dataset)
    monkeypatch.setattr(ds, "Scanner", Scanner)
    monkeypatch.setattr(pq, "read_table", read_table)
    monkeypatch.setattr(pq, "ParquetFile", ParquetFile)
    monkeypatch.setattr(pq, "ParquetDataset", ParquetDataset)
    monkeypatch.setattr(pq, "read_pandas", read_pandas)
    return watcher


# A read asks the holdout rule about at most this many days: the synthetic tables span 32 days,
# an open-ended request up to ~2.9M (date.min or 1900 to date.max).
DAY_BUDGET = 5_000


@pytest.fixture
def day_budget(monkeypatch) -> Counter:
    """Counts the days ``is_holdout_day`` is asked about (development_filter looks it up at call
    time) and fails the read once they pass ``DAY_BUDGET``."""
    asked: Counter = Counter()
    real = holdout_module.is_holdout_day

    def counted(country, day):
        asked["days"] += 1
        if asked["days"] > DAY_BUDGET:
            raise AssertionError(
                f"the read asked is_holdout_day about more than {DAY_BUDGET} days: it enumerates "
                "the requested range instead of the days the table can hold"
            )
        return real(country, day)

    monkeypatch.setattr(holdout_module, "is_holdout_day", counted)
    return asked


def statistics_for(mode: str, names: list[str]):
    """write_statistics for pq.write_table: 'all', 'none', or every column but race_date."""
    return {"all": True, "none": False}.get(mode, [n for n in names if n != "race_date"])


def assert_scan_excluded_holdout(spy: ScanSpy, n_development: int) -> None:
    assert spy.outputs, "no pyarrow scan was seen: the read must scan through pyarrow"
    leaks = spy.leaks(n_development)
    assert not leaks, f"the scan materialized holdout rows: {leaks}"


# ---------------------------------------------------------------------------------------------
# read_store_table (v0's scoped reader)
# ---------------------------------------------------------------------------------------------

V0_SCHEMA = pa.schema(
    [
        ("race_id", pa.string()),
        ("race_date", pa.date32()),
        ("horse_no", pa.int32()),
        ("win_odds", pa.float64()),
        ("finish_pos", pa.int32()),
    ]
)
SOURCE_SHAPES = {  # source -> (race_id tag, races on day k, runners in race r of day k)
    "jra": ("jra", lambda k: 1 + k % 3, lambda k, r: 4 + (k + r) % 3),
    "jravan": ("jvn", lambda k: 2 + k % 2, lambda k, r: 3 + (k * r) % 4),
}


def v0_rows(source: str) -> pd.DataFrame:
    tag, n_races, n_runners = SOURCE_SHAPES[source]
    rows = []
    for k, day in enumerate(DAYS):
        for r in range(n_races(k)):
            race_id = f"{tag}:{day.isoformat()}:TOK:{r + 1:02d}"
            for h in range(1, n_runners(k, r) + 1):
                rows.append(
                    {
                        "race_id": race_id,
                        "race_date": day,
                        "horse_no": h,
                        "win_odds": 1.5 + 2.5 * h + k,
                        "finish_pos": h,
                    }
                )
    return pd.DataFrame(rows)


def write_v0_store(tmp_path, monkeypatch, statistics: str = "all") -> dict[str, pd.DataFrame]:
    data = (tmp_path / "data").resolve()
    monkeypatch.setenv(DATA_VAR, str(data))
    frames = {}
    for source in SOURCE_SHAPES:
        frame = v0_rows(source)
        folder = data / "processed" / source
        folder.mkdir(parents=True)
        shuffled = frame.sample(frac=1.0, random_state=415).reset_index(drop=True)
        pq.write_table(
            pa.Table.from_pandas(shuffled, schema=V0_SCHEMA, preserve_index=False),
            folder / "runners.parquet",
            row_group_size=16,  # small, mixed row groups: each holds development and holdout days
            write_statistics=statistics_for(statistics, V0_SCHEMA.names),
        )
        frames[source] = frame
    return frames


COLUMN_SETS = [
    ["race_id", "race_date", "finish_pos"],
    ["win_odds", "race_id"],
    ["horse_no"],
    ["finish_pos", "race_date", "horse_no"],
]
_RANGES = [  # (id, start, end); each range holds at least one holdout day
    ("pre-start-even-day-to-dev-day", date(2026, 11, 23), date(2026, 12, 5)),
    ("holdout-day-to-holdout-day", date(2026, 11, 29), date(2026, 12, 4)),
    ("holdout-start-to-dev-day", date(2026, 11, 28), date(2026, 12, 12)),
    ("holdout-start-to-first-holdout", date(2026, 11, 28), date(2026, 11, 29)),
    ("dev-day-to-holdout-day", date(2026, 12, 5), date(2026, 12, 9)),
    ("one-holdout-day", date(2026, 12, 3), date(2026, 12, 3)),
    ("whole-store", DAYS[0], DAYS[-1]),
]
RANGES = [
    pytest.param(start, end, COLUMN_SETS[k % len(COLUMN_SETS)], k % 2 == 0, id=name)
    for k, (name, start, end) in enumerate(_RANGES)
]


def expected_v0(frame, start, end, where_max_horse=None):
    mask = frame["race_date"].map(lambda d: start <= d <= end and not held(d))
    if where_max_horse is not None:
        mask &= frame["horse_no"] <= where_max_horse
    return frame[mask]


@pytest.mark.parametrize("source", [None, "jra", "jravan"])
@pytest.mark.parametrize(("start", "end", "columns", "touches_price"), RANGES)
def test_read_store_table_scans_no_holdout_row(
    tmp_path, monkeypatch, spy, source, start, end, columns, touches_price
):
    frames = write_v0_store(tmp_path, monkeypatch)
    frame = frames[source or "jra"]
    kwargs = {} if source is None else {"source": source}

    table = read_store_table(
        "runners", columns=columns, start=start, end=end, touches_price=touches_price, **kwargs
    )

    want = expected_v0(frame, start, end)
    assert table.column_names == columns
    assert table.num_rows == len(want)
    got = table.to_pandas()
    for column in columns:  # every development row in range, both endpoints included
        assert Counter(got[column].tolist()) == Counter(want[column].tolist()), column
    assert_scan_excluded_holdout(spy, len(want))


_OPEN_ENDED = [  # (id, start, end, touches_price): the callers' open-ended reads
    ("1900-to-date-max", date(1900, 1, 1), date.max, False),
    ("price-era-to-date-max", PRICE_ERA_START, date.max, True),
    ("holdout-day-to-date-max", date(2026, 11, 29), date.max, True),
    ("dev-day-to-date-max", date(2026, 12, 5), date.max, False),
    ("date-min-to-holdout-day", date.min, date(2026, 12, 3), False),
    ("date-min-to-date-max", date.min, date.max, False),
    ("after-the-store-to-date-max", date(2027, 1, 1), date.max, True),
]
OPEN_ENDED = [
    pytest.param(start, end, price, COLUMN_SETS[k % len(COLUMN_SETS)], id=name)
    for k, (name, start, end, price) in enumerate(_OPEN_ENDED)
]


@pytest.mark.parametrize("statistics", ["all", "none", "no-race-date"])
@pytest.mark.parametrize(("start", "end", "touches_price", "columns"), OPEN_ENDED)
def test_read_store_table_open_ended_read_asks_only_the_tables_days(
    tmp_path, monkeypatch, spy, day_budget, start, end, touches_price, columns, statistics
):
    """PR #427 review: ``end=date.max`` (or ``start=date.min``) asks the holdout rule about the
    days the table can hold, not the ~2.9M days of the request; with or without race_date
    statistics, every development row in range comes back and no holdout row is scanned."""
    frames = write_v0_store(tmp_path, monkeypatch, statistics)
    source = "jravan" if statistics == "none" else "jra"

    table = read_store_table(
        "runners",
        columns=columns,
        start=start,
        end=end,
        touches_price=touches_price,
        source=source,
    )

    want = expected_v0(frames[source], start, end)
    assert table.column_names == columns
    assert table.num_rows == len(want)
    got = table.to_pandas()
    for column in columns:
        assert Counter(got[column].tolist()) == Counter(want[column].tolist()), column
    assert day_budget["days"] <= DAY_BUDGET
    if len(want):
        assert_scan_excluded_holdout(spy, len(want))
    else:
        assert not spy.leaks(0), f"the scan materialized holdout rows: {spy.leaks(0)}"


def test_read_store_table_to_date_max_is_fast(tmp_path, monkeypatch):
    """The callers' ``start=HISTORY_START (1900), end=date.max`` read of a 32-day table finishes
    well inside 2 s (it took ~15 s when every requested day was asked about)."""
    frames = write_v0_store(tmp_path, monkeypatch)
    start = date(1900, 1, 1)

    began = time.perf_counter()
    table = read_store_table(
        "runners", columns=["race_id", "race_date"], start=start, end=date.max, touches_price=False
    )
    elapsed = time.perf_counter() - began

    want = expected_v0(frames["jra"], start, date.max)
    assert Counter(table.column("race_id").to_pylist()) == Counter(want["race_id"].tolist())
    assert elapsed < 2.0, f"read_store_table(..., end=date.max) took {elapsed:.1f} s"


@pytest.mark.parametrize("source", ["jra", "jravan"])
def test_read_store_table_scan_keeps_the_where_narrowing(tmp_path, monkeypatch, spy, source):
    frames = write_v0_store(tmp_path, monkeypatch)
    start, end = date(2026, 11, 27), date(2026, 12, 13)
    table = read_store_table(
        "runners",
        columns=["horse_no", "race_id"],
        start=start,
        end=end,
        touches_price=True,
        where=ds.field("horse_no") <= 2,
        source=source,
    )
    want = expected_v0(frames[source], start, end, where_max_horse=2)
    assert table.column_names == ["horse_no", "race_id"]
    assert Counter(zip(*table.to_pydict().values(), strict=True)) == Counter(
        zip(want["horse_no"], want["race_id"], strict=True)
    )
    assert_scan_excluded_holdout(spy, len(want))


# ---------------------------------------------------------------------------------------------
# speed_run.read_inputs (the speed study's races and runners)
# ---------------------------------------------------------------------------------------------


def _store_frame(name: str, rows: list[dict]) -> pa.Table:
    spec = schemas.TABLES[name].columns
    schema = pa.schema([(c, tables.ARROW_TYPES[kind]) for c, kind in spec.items()])
    full = [{c: row.get(c) for c in spec} for row in rows]
    return pa.Table.from_pylist(full, schema=schema)


def speed_store(days: list[date], tag: str) -> tuple[list[dict], list[dict]]:
    races, runners = [], []
    for k, day in enumerate(days):
        for r in range(1 + k % 2):
            race_id = f"{tag}:{day.isoformat()}:TOK:{r + 1:02d}"
            races.append(
                {
                    "race_id": race_id,
                    "meeting_id": f"{tag}:{day.isoformat()}:TOK",
                    "race_date": day,
                    "venue": "TOK",
                    "race_no": r + 1,
                    "surface": "turf",
                    "distance_m": 1600 + 200 * r,
                    "course_config": "A",
                    "race_class": "1win",
                    "post_time_local": f"{10 + r}:30",
                }
            )
            for h in range(1, 4 + (k + r) % 3):
                runners.append(
                    {
                        "race_id": race_id,
                        "race_date": day,
                        "horse_no": h,
                        "finish_pos": h,
                        "finish_time_s": 95.0 + h + k,
                    }
                )
    return races, runners


def write_speed_store(
    tmp_path, monkeypatch, days, statistics: str = "all"
) -> tuple[list[dict], list[dict]]:
    data = (tmp_path / "data").resolve()
    monkeypatch.setenv(DATA_VAR, str(data))
    for source, tag in (("jra", "jra"), ("jravan", "jvn")):
        races, runners = speed_store(days, tag)
        folder = data / "processed" / source
        folder.mkdir(parents=True)
        for name, rows in (("races", races), ("runners", runners)):
            rows = pd.DataFrame(rows).sample(frac=1.0, random_state=7).to_dict("records")
            rows = [{c: (None if pd.isna(v) else v) for c, v in row.items()} for row in rows]
            frame = _store_frame(name, rows)
            pq.write_table(
                frame,
                folder / f"{name}.parquet",
                row_group_size=8,
                write_statistics=statistics_for(statistics, frame.column_names),
            )
        if source == "jra":
            jra = (races, runners)
    return jra


CALENDARS = [
    pytest.param(
        [
            date(2015, 3, 1),
            *(date(2026, 11, d) for d in (27, 28, 29, 30)),
            date(2026, 12, 1),
            date(2026, 12, 5),
        ],
        id="old-dev-day-to-dev-day",
    ),  # fmt: skip
    pytest.param(
        [
            date(2026, 11, 29),
            date(2026, 11, 30),
            date(2026, 12, 2),
            date(2026, 12, 3),
            date(2026, 12, 12),
            date(2026, 12, 21),
        ],
        id="holdout-day-to-holdout-day",
    ),  # fmt: skip
    pytest.param(DAYS, id="every-day"),
]


@pytest.mark.parametrize("statistics", ["all", "none", "no-race-date"])
@pytest.mark.parametrize("days", CALENDARS)
def test_read_inputs_scans_no_holdout_row(tmp_path, monkeypatch, spy, days, statistics):
    """Also without race_date statistics in the footer (PR #427 review): the span may not be
    found by reading the race_date column, which holds the holdout days."""
    races, runners = write_speed_store(tmp_path, monkeypatch, days, statistics)
    dev_races = [r for r in races if not held(r["race_date"])]
    dev_runners = [r for r in runners if not held(r["race_date"])]
    assert dev_races and len(dev_races) < len(races)

    got_races, got_runners = read_inputs()

    assert list(got_races.columns) == list(RACE_COLUMNS)
    assert list(got_runners.columns) == [*RUNNER_COLUMNS, "race_date"]
    assert Counter(got_races["race_id"]) == Counter(r["race_id"] for r in dev_races)
    got_pairs = zip(got_runners["race_id"], got_runners["finish_pos"].astype(int), strict=True)
    assert Counter(got_pairs) == Counter((r["race_id"], r["finish_pos"]) for r in dev_runners)
    kept_days = {pd.Timestamp(d).date() for d in got_races["race_date"]}
    assert kept_days == {r["race_date"] for r in dev_races}  # the first and last dev days too
    assert_scan_excluded_holdout(spy, max(len(dev_races), len(dev_runners)))


# ---------------------------------------------------------------------------------------------
# read_inputs reads one snapshot (PR #427 review: the span and the scan opened the file apart)
# ---------------------------------------------------------------------------------------------


class RebuildOnFirstOpen:
    """Atomically replaces ``target`` with ``rebuilt`` (os.replace, as a store rebuild does) right
    after the first pyarrow call that opens ``target`` by its path returns; once only."""

    def __init__(self, monkeypatch, target: Path, rebuilt: Path) -> None:
        self.target, self.rebuilt, self.replaced = target.resolve(), rebuilt, False
        names = {
            pq: ("read_table", "read_metadata", "read_schema", "read_pandas"),
            ds: ("dataset",),
            pa: ("memory_map", "input_stream"),
        }
        for module, funcs in names.items():
            for name in funcs:
                monkeypatch.setattr(module, name, self._wrap(getattr(module, name)))
        hook = self

        class ParquetFile(pq.ParquetFile):
            def __init__(self, source, *a, **kw):
                super().__init__(source, *a, **kw)
                hook._after(source)

        class ParquetDataset(pq.ParquetDataset):
            def __init__(self, path_or_paths, *a, **kw):
                super().__init__(path_or_paths, *a, **kw)
                hook._after(path_or_paths)

        monkeypatch.setattr(pq, "ParquetFile", ParquetFile)
        monkeypatch.setattr(pq, "ParquetDataset", ParquetDataset)

    def _opens_target(self, source) -> bool:
        items = source if isinstance(source, (list, tuple)) else [source]
        return any(
            isinstance(x, (str, os.PathLike)) and Path(x).resolve() == self.target for x in items
        )

    def _after(self, source) -> None:
        if not self.replaced and self._opens_target(source):
            os.replace(self.rebuilt, self.target)
            self.replaced = True

    def _wrap(self, real):
        def opened(*a, **kw):
            out = real(*a, **kw)
            source = (
                a[0]
                if a
                else next(
                    (kw[k] for k in ("source", "path", "where", "path_or_paths") if k in kw), None
                )
            )
            self._after(source)
            return out

        return opened


def test_read_inputs_reads_one_snapshot_of_a_table_rebuilt_mid_read(tmp_path, monkeypatch):
    """The runners table is rebuilt (every finish time changed, one development day added after
    its last day) between the read's first open of it and the rest of the read. The runners come
    back as the table was or as it is, all of one: never the new times with the added day cut."""
    _, runners = write_speed_store(tmp_path, monkeypatch, DAYS)
    folder = tmp_path / "data" / "processed" / "jra"
    added = next(d for d in (DAYS[-1] + timedelta(days=k) for k in range(1, 30)) if not held(d))
    new_rows = [{**r, "finish_time_s": r["finish_time_s"] + 100.0} for r in runners] + [
        {
            "race_id": f"jra:{added.isoformat()}:TOK:01",
            "race_date": added,
            "horse_no": h,
            "finish_pos": h,
            "finish_time_s": 90.0 + h,
        }
        for h in range(1, 5)
    ]
    rebuilt = folder / "runners.rebuilt"
    pq.write_table(_store_frame("runners", new_rows), rebuilt, row_group_size=8)

    def dev(rows):
        return Counter((r["race_id"], r["finish_time_s"]) for r in rows if not held(r["race_date"]))

    before, after = dev(runners), dev(new_rows)
    assert before != after and (f"jra:{added.isoformat()}:TOK:01", 91.0) in after

    RebuildOnFirstOpen(monkeypatch, folder / "runners.parquet", rebuilt)
    _, got = read_inputs()

    pairs = Counter(zip(got["race_id"], got["finish_time_s"].astype(float), strict=True))
    assert pairs in (before, after), (
        "the runners mix two versions of the table: "
        f"{sum((pairs & after).values())} rows of the rebuilt one, "
        f"{sum((pairs & before).values())} of the old, of {sum(pairs.values())}"
    )
