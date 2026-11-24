"""Read access to the store's Parquet tables (#46).

Ported from HKTrack tests/unit/store/test_tables.py. Layout pinned here (D = data_dir()): a
source's table <name> lives at D/processed/<source>/<name>.parquet, i.e.
`paths.processed_dir(source) / f"{name}.parquet"`. Every table is tiny synthetic Parquet written in
tmp_path from the declared schema; no real data.

dtypes out of `read_table`: int -> pandas Int64, float -> float64, bool -> pandas boolean,
str -> str values (object or string dtype), date -> datetime.date values, utc -> tz-aware UTC
timestamps, int_list / float_list -> list-like values.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core import paths

# isort: split
# A block of its own: ruff counts horsetrack.core.store as third-party until it exists.
from horsetrack.core.store import schemas, tables

VAR = "HORSETRACK_DATA_DIR"
TABLE_NAMES = (
    "meetings", "races", "runners", "dividends", "sectionals", "incidents", "running_comments",
    "horses", "horse_runs", "horse_pedigree", "jrdb_pre_race", "jrdb_post_race",
)  # fmt: skip

# kind -> the Arrow type the store writes it as (independent of the code under test)
ARROW = {
    schemas.INT: pa.int64(),
    schemas.FLOAT: pa.float64(),
    schemas.STR: pa.string(),
    schemas.BOOL: pa.bool_(),
    schemas.DATE: pa.date32(),
    schemas.UTC: pa.timestamp("us", tz="UTC"),
    schemas.INT_LIST: pa.list_(pa.int32()),
    schemas.FLOAT_LIST: pa.list_(pa.float64()),
}


@pytest.fixture
def data(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    monkeypatch.setenv(VAR, str(d))
    return d


def _value(kind: str, column: str, source: str, i: int, salt: int) -> Any:
    if kind == schemas.INT:
        return 100 * salt + 7 * i + len(column)
    if kind == schemas.FLOAT:
        return salt + i + 0.25
    if kind == schemas.STR:
        return f"{source}-{column}-{i}"
    if kind == schemas.BOOL:
        return (i + salt) % 2 == 0
    if kind == schemas.DATE:
        return date(2010 + salt, 1 + i % 12, 1 + i)
    if kind == schemas.UTC:
        return datetime(2010 + salt, 2, 1 + i, 3, 4, 5, tzinfo=timezone.utc)
    if kind == schemas.INT_LIST:
        return [i, i + salt, 3]
    if kind == schemas.FLOAT_LIST:
        return [i + 0.5, float(salt)]
    raise AssertionError(f"unknown kind {kind!r}")


def _rows(table: str, source: str, n: int, /, salt: int, **overrides: list) -> dict[str, list]:
    """n rows of every declared column; every third row null outside sort key and jurisdiction."""
    spec = schemas.TABLES[table]
    unknown = set(overrides) - set(spec.columns)
    assert not unknown, f"{table} declares no column {sorted(unknown)}"
    keep = {"jurisdiction", *spec.sort_by}
    out: dict[str, list] = {}
    for column, kind in spec.columns.items():
        if column in overrides:
            out[column] = list(overrides[column])
            continue
        out[column] = [
            None if (i % 3 == 1 and column not in keep) else _value(kind, column, source, i, salt)
            for i in range(n)
        ]
    if "jurisdiction" not in overrides:
        out["jurisdiction"] = [source] * n
    return out


def _write(source: str, name: str, rows: dict[str, list], order: list[str] | None = None) -> Path:
    spec = schemas.TABLES[name]
    order = order or list(spec.columns)
    schema = pa.schema([(column, ARROW[spec.columns[column]]) for column in order])
    path = paths.processed_dir(source) / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({c: rows[c] for c in order}, schema=schema), path)
    return path


def _py(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_py(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _assert_kind(series: pd.Series, kind: str, where: str) -> None:
    present = [v for v in series if _py(v) is not None]
    assert present, f"{where}: test data has no non-null value"
    if kind == schemas.INT:
        assert series.dtype == pd.Int64Dtype(), where
    elif kind == schemas.FLOAT:
        assert series.dtype == np.float64, where
    elif kind == schemas.BOOL:
        assert series.dtype == pd.BooleanDtype(), where
    elif kind == schemas.STR:
        assert series.dtype == object or pd.api.types.is_string_dtype(series.dtype), where
        assert all(isinstance(v, str) for v in present), where
    elif kind == schemas.DATE:
        assert all(isinstance(v, date) and not isinstance(v, datetime) for v in present), where
    elif kind == schemas.UTC:
        assert isinstance(series.dtype, pd.DatetimeTZDtype), where
        assert str(series.dtype.tz) == "UTC", where
    elif kind in (schemas.INT_LIST, schemas.FLOAT_LIST):
        scalar = int if kind == schemas.INT_LIST else float
        for v in present:
            items = _py(v)
            assert isinstance(items, list) and all(isinstance(x, scalar) for x in items), where
    else:
        raise AssertionError(f"{where}: unknown kind {kind!r}")


def _assert_frame(frame: pd.DataFrame, name: str, rows: dict, columns: list[str]) -> None:
    spec = schemas.TABLES[name]
    assert list(frame.columns) == columns, name
    for column in columns:
        _assert_kind(frame[column], spec.columns[column], f"{name}.{column}")
        assert [_py(v) for v in frame[column]] == rows[column], f"{name}.{column}"


def test_every_schema_has_a_sort_key_and_a_jurisdiction_column():
    assert set(schemas.TABLES) == set(TABLE_NAMES)
    for name in TABLE_NAMES:
        spec = schemas.TABLES[name]
        assert isinstance(spec, schemas.TableSpec)
        assert spec.name == name
        assert spec.columns.get("jurisdiction") == schemas.STR, name
        assert spec.sort_by and all(isinstance(c, str) for c in spec.sort_by), name
        assert set(spec.sort_by) <= set(spec.columns), name
        assert set(spec.columns.values()) <= set(ARROW), name


@pytest.mark.parametrize("name", TABLE_NAMES)
def test_read_table_returns_the_declared_columns_and_dtypes(data, name):
    """Each source reads its own file: two sources, different row counts and values."""
    written = {}
    for salt, (source, n) in enumerate((("jra", 4), ("nar", 2)), start=1):
        written[source] = _rows(name, source, n, salt)
        _write(source, name, written[source])
    declared = list(schemas.TABLES[name].columns)
    for source, rows in written.items():
        frame = tables.read_table(source, name)
        assert len(frame) == len(rows["jurisdiction"])
        _assert_frame(frame, name, rows, declared)


@pytest.mark.parametrize("name", TABLE_NAMES)
def test_read_table_returns_declared_order_whatever_the_file_order(data, name):
    rows = _rows(name, "jra", 3, salt=2)
    declared = list(schemas.TABLES[name].columns)
    _write("jra", name, rows, order=declared[::-1])
    _assert_frame(tables.read_table("jra", name), name, rows, declared)


@pytest.mark.parametrize("name", TABLE_NAMES)
def test_read_table_column_subset(data, name):
    spec = schemas.TABLES[name]
    rows = _rows(name, "jra", 4, salt=3)
    _write("jra", name, rows)
    declared = list(spec.columns)
    for subset in (declared[-3:][::-1], [spec.sort_by[0]], ["jurisdiction", declared[-1]]):
        subset = list(dict.fromkeys(subset))
        frame = tables.read_table("jra", name, columns=subset)
        _assert_frame(frame, name, rows, subset)
    assert list(tables.read_table("jra", name, columns=None).columns) == declared


def test_runners_subset_reads_nullable_ints(data):
    """HKTrack's own case: a subset of runners keeps `rating` as Int64 with its nulls."""
    assert schemas.TABLES["runners"].columns["rating"] == schemas.INT
    rows = _rows("runners", "jra", 5, salt=1, rating=[31, None, 40, None, 12])
    _write("jra", "runners", rows)
    some = tables.read_table("jra", "runners", columns=["race_id", "horse_id", "rating"])
    assert list(some.columns) == ["race_id", "horse_id", "rating"]
    assert some["rating"].dtype == pd.Int64Dtype()
    assert [_py(v) for v in some["rating"]] == [31, None, 40, None, 12]


def test_missing_table_raises_a_clear_error(data):
    _write("jra", "races", _rows("races", "jra", 2, salt=1))
    for source, name in (("jra", "runners"), ("jra", "horse_pedigree"), ("nar", "races")):
        with pytest.raises(FileNotFoundError) as exc:
            tables.read_table(source, name)
        message = str(exc.value)
        assert name in message and source in message, message


def test_unknown_table_name_is_refused(data):
    # a file of that name on disk does not make it a table
    folder = paths.processed_dir("jra")
    folder.mkdir(parents=True)
    pq.write_table(pa.table({"jurisdiction": ["jra"]}), folder / "racecard_runners.parquet")
    for name in ("racecard_runners", "nope", "../manifest", "", "Runners"):
        with pytest.raises(ValueError) as exc:
            tables.read_table("jra", name)
        if name:
            assert name in str(exc.value), str(exc.value)


def test_read_table_follows_the_data_dir_at_call_time(tmp_path, monkeypatch):
    for n, folder in ((2, "one"), (5, "two")):
        monkeypatch.setenv(VAR, str(tmp_path / folder))
        _write("jra", "races", _rows("races", "jra", n, salt=n))
    for n, folder in ((2, "one"), (5, "two")):
        monkeypatch.setenv(VAR, str(tmp_path / folder))
        assert len(tables.read_table("jra", "races")) == n


def _views(conn) -> set[str]:
    found = conn.execute("SELECT table_name FROM information_schema.tables").fetchall()
    return {row[0] for row in found}


def test_duckdb_views_read_the_parquet(tmp_path, monkeypatch):
    # a quote in the data dir: the view's path must be escaped, not spliced into the SQL
    monkeypatch.setenv(VAR, str(tmp_path / "o'data"))
    races = _rows(
        "races", "jra", 3, salt=1,
        race_id=["jra:R3", "jra:R1", "jra:R2"],
        race_date=[date(2024, 6, 2), date(2019, 1, 5), date(2026, 11, 22)],
    )  # fmt: skip
    runners = _rows(
        "runners", "jra", 5, salt=2,
        race_id=["jra:R1", "jra:R1", "jra:R2", "jra:R3", "jra:R3"],
        horse_id=["H1", "H2", "H1", "H3", "H2"],
        rating=[31, None, 44, 12, 50],
        running_positions=[[3, 2, 1], [5, 6, 7], None, [1, 1, 1], [9, 8, 4]],
    )  # fmt: skip
    horses = _rows("horses", "jra", 2, salt=3, horse_id=["H1", "H2"], name=["Alpha", "Beta"])
    for name, rows in (("races", races), ("runners", runners), ("horses", horses)):
        _write("jra", name, rows)
    _write("nar", "dividends", _rows("dividends", "nar", 6, salt=4))

    conn = tables.connect("jra")
    assert _views(conn) == {"races", "runners", "horses"}
    span = "SELECT count(*), min(race_date), max(race_date) FROM races"
    n, first, last = conn.execute(span).fetchone()
    assert (n, first, last) == (3, date(2019, 1, 5), date(2026, 11, 22))
    assert conn.execute("SELECT count(*) FROM runners").fetchone()[0] == 5
    joined = conn.execute(
        "SELECT r.race_id, r.rating, h.name FROM runners r JOIN horses h USING (horse_id)"
        " ORDER BY r.race_id, h.name"
    ).fetchall()
    assert joined == [
        ("jra:R1", 31, "Alpha"),
        ("jra:R1", None, "Beta"),
        ("jra:R2", 44, "Alpha"),
        ("jra:R3", 50, "Beta"),
    ]
    query = "SELECT running_positions FROM runners WHERE rating = 50"
    positions = conn.execute(query).fetchone()[0]
    assert list(positions) == [9, 8, 4] and all(isinstance(p, int) for p in positions)
    conn.close()

    other = tables.connect("nar")
    assert _views(other) == {"dividends"}
    assert other.execute("SELECT count(*) FROM dividends").fetchone()[0] == 6
    assert other.execute("SELECT DISTINCT jurisdiction FROM dividends").fetchall() == [("nar",)]
    other.close()

    assert _views(tables.connect("hkjc")) == set()
