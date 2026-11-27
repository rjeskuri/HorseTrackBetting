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
  uses), and checks every table and batch they hand back. A read that scans by another route is
  not seen and fails the "a scan was seen" check.

The store is synthetic: every calendar day from 2026-11-20 to 2026-12-21, so development and
holdout days interleave from the holdout start (2026-11-28, a development day) on, and the rows
are shuffled into small, mixed row groups. The oracle recomputes the rule from the committed
holdout.yaml with hashlib; it never calls the code under test. The holdout is never opened.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import yaml

from horsetrack.core.store import schemas, tables
from horsetrack.countries.japan.models.v0.store import read_store_table
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

    monkeypatch.setattr(ds, "dataset", dataset)
    monkeypatch.setattr(ds, "Scanner", Scanner)
    monkeypatch.setattr(pq, "read_table", read_table)
    return watcher


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


def write_v0_store(tmp_path, monkeypatch) -> dict[str, pd.DataFrame]:
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


def write_speed_store(tmp_path, monkeypatch, days) -> tuple[list[dict], list[dict]]:
    data = (tmp_path / "data").resolve()
    monkeypatch.setenv(DATA_VAR, str(data))
    for source, tag in (("jra", "jra"), ("jravan", "jvn")):
        races, runners = speed_store(days, tag)
        folder = data / "processed" / source
        folder.mkdir(parents=True)
        for name, rows in (("races", races), ("runners", runners)):
            rows = pd.DataFrame(rows).sample(frac=1.0, random_state=7).to_dict("records")
            rows = [{c: (None if pd.isna(v) else v) for c, v in row.items()} for row in rows]
            pq.write_table(_store_frame(name, rows), folder / f"{name}.parquet", row_group_size=8)
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


@pytest.mark.parametrize("days", CALENDARS)
def test_read_inputs_scans_no_holdout_row(tmp_path, monkeypatch, spy, days):
    races, runners = write_speed_store(tmp_path, monkeypatch, days)
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
