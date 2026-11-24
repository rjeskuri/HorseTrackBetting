"""scripts/check_store_rebuild.py (ticket #59): compare a rebuilt store table with the copied one.

Every test writes small synthetic Parquet files under tmp_path; no test reads the data dir.

Contract pinned here (the names the implementation must follow):

* `compare_tables(a: Path, b: Path, key: Sequence[str] | None = None) -> TableDiff` is pure: it
  reads the two Parquet files and writes nothing (both files keep their bytes, no file appears
  next to them). `key` is the table's key, the script passing `schemas.TABLES[name].sort_by`;
  `None` sorts by every column. Rows are compared as a set ordered by the key, so the order the
  rows are stored in never matters.
* `TableDiff` has attributes:
  - `equal: bool`, True exactly when the row counts, the column sets, the column types and the
    content hashes all agree;
  - `rows_a`, `rows_b: int`, the row count of each side;
  - `missing_columns`, the columns of `a` that `b` lacks, and `extra_columns`, the columns of `b`
    that `a` lacks (any collection of names; empty when the column sets agree);
  - `dtype_diffs`, a mapping keyed by each column both sides have whose Arrow type differs (the
    values are the script's to choose, e.g. the two type names); empty when the types agree;
  - `hash_a`, `hash_b`, the content hash of each side's rows sorted by the key; equal for equal
    content in any row order, different when any value (a null, a list element too) differs.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_store_rebuild.py"
MODULE = "check_store_rebuild"

SCHEMA = pa.schema(
    [
        ("race_id", pa.string()),
        ("horse_no", pa.int64()),
        ("race_date", pa.date32()),
        ("horse_name", pa.string()),
        ("odds", pa.float64()),
        ("split_times_s", pa.list_(pa.float64())),
    ]
)
SCALAR_SCHEMA = pa.schema([f for f in SCHEMA if f.name != "split_times_s"])
KEY = ("race_id", "horse_no")


def _script():
    """The script loaded as a module (scripts/ is not a package)."""
    if not SCRIPT.is_file():
        pytest.fail(f"{SCRIPT.relative_to(REPO)} does not exist yet", pytrace=False)
    if MODULE not in sys.modules:
        spec = importlib.util.spec_from_file_location(MODULE, SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE] = module  # dataclasses look their module up here
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[MODULE]
            raise
    return sys.modules[MODULE]


def _rows(n: int, *, lists: bool = True) -> list[dict]:
    """n distinct runners, unique on KEY; a null in each nullable column somewhere."""
    rows = []
    for i in range(n):
        row = {
            "race_id": f"2025{i // 3:04d}",
            "horse_no": i % 3 + 1,
            "race_date": dt.date(2025, 1, 4) + dt.timedelta(days=i // 3),
            "horse_name": None if i == 1 else f"Horse {i}",
            "odds": None if i == 2 else round(1.5 + i * 0.7, 1),
        }
        if lists:
            row["split_times_s"] = [12.0 + i / 10, 11.5 + i / 10] if i != 3 else None
        rows.append(row)
    return rows


def _write(path: Path, rows: list[dict], schema: pa.Schema = SCHEMA, **kwargs) -> Path:
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, **kwargs)
    return path


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


@pytest.mark.parametrize("n", [1, 4, 9])
def test_identical_tables_compare_equal(tmp_path, n):
    rows = _rows(n)
    a = _write(tmp_path / "a.parquet", rows)
    b = _write(tmp_path / "b.parquet", rows, row_group_size=2)  # same rows, other file layout
    before = _snapshot(tmp_path)

    diff = _script().compare_tables(a, b, key=KEY)

    assert diff.equal is True
    assert (diff.rows_a, diff.rows_b) == (n, n)
    assert not diff.missing_columns and not diff.extra_columns
    assert not diff.dtype_diffs
    assert diff.hash_a == diff.hash_b
    assert _snapshot(tmp_path) == before, "compare_tables wrote to or beside its inputs"


@pytest.mark.parametrize(
    ("key", "lists"),
    [(KEY, True), (("race_id", "horse_no"), False), (None, False)],
    ids=["key", "key-scalar", "all-columns"],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_row_order_does_not_matter(tmp_path, key, lists, seed):
    schema = SCHEMA if lists else SCALAR_SCHEMA
    rows = _rows(9, lists=lists)
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    assert shuffled != rows
    a = _write(tmp_path / "a.parquet", rows, schema)
    b = _write(tmp_path / "b.parquet", shuffled, schema)

    diff = _script().compare_tables(a, b, key=key)

    assert diff.equal is True
    assert (diff.rows_a, diff.rows_b) == (9, 9)
    assert diff.hash_a == diff.hash_b


CHANGES = [
    (0, "horse_name", "Horse 0x"),
    (8, "horse_name", "Horse 9"),  # another row's value: still a change
    (4, "odds", 4.3000001),
    (1, "horse_name", "Horse 1"),  # null -> value
    (5, "odds", None),  # value -> null
    (6, "race_date", dt.date(2025, 1, 7)),
]
LIST_CHANGES = [
    (2, "split_times_s", [12.2, 11.8]),  # one list element
    (7, "split_times_s", [12.7]),  # a shorter list
    (3, "split_times_s", [12.3, 11.8]),  # null -> list
]


@pytest.mark.parametrize(
    ("row", "column", "value", "key"),
    [(*c, KEY) for c in CHANGES + LIST_CHANGES] + [(*c, None) for c in CHANGES],
)
def test_a_changed_value_is_reported(tmp_path, row, column, value, key):
    schema = SCHEMA if key is not None else SCALAR_SCHEMA  # key=None: sort by scalar columns
    rows = _rows(9, lists=key is not None)
    changed = [dict(r) for r in rows]
    assert changed[row][column] != value
    changed[row][column] = value
    a = _write(tmp_path / "a.parquet", rows, schema)
    b = _write(tmp_path / "b.parquet", changed, schema)

    diff = _script().compare_tables(a, b, key=key)

    assert diff.equal is False
    assert (diff.rows_a, diff.rows_b) == (9, 9)
    assert not diff.missing_columns and not diff.extra_columns
    assert not diff.dtype_diffs
    assert diff.hash_a != diff.hash_b


@pytest.mark.parametrize("column", ["race_date", "horse_name", "odds", "split_times_s"])
@pytest.mark.parametrize("side", ["b-lacks", "a-lacks"])
def test_a_missing_column_is_reported(tmp_path, column, side):
    rows = _rows(6)
    full = _write(tmp_path / "full.parquet", rows)
    narrow_schema = pa.schema([f for f in SCHEMA if f.name != column])
    narrow_rows = [{k: v for k, v in r.items() if k != column} for r in rows]
    narrow = _write(tmp_path / "narrow.parquet", narrow_rows, narrow_schema)
    a, b = (full, narrow) if side == "b-lacks" else (narrow, full)

    diff = _script().compare_tables(a, b, key=KEY)

    assert diff.equal is False
    assert (diff.rows_a, diff.rows_b) == (6, 6)
    if side == "b-lacks":
        assert set(diff.missing_columns) == {column}
        assert not diff.extra_columns
    else:
        assert set(diff.extra_columns) == {column}
        assert not diff.missing_columns
    assert not diff.dtype_diffs


@pytest.mark.parametrize(
    ("column", "new_type"),
    [
        ("horse_no", pa.float64()),
        ("odds", pa.float32()),
        ("race_date", pa.string()),
        ("split_times_s", pa.list_(pa.float32())),
    ],
)
def test_a_dtype_change_is_reported(tmp_path, column, new_type):
    rows = _rows(6)
    a = _write(tmp_path / "a.parquet", rows)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    idx = SCHEMA.get_field_index(column)
    table = table.set_column(idx, pa.field(column, new_type), table.column(column).cast(new_type))
    b = tmp_path / "b.parquet"
    pq.write_table(table, b)

    diff = _script().compare_tables(a, b, key=KEY)

    assert diff.equal is False
    assert (diff.rows_a, diff.rows_b) == (6, 6)
    assert set(diff.dtype_diffs) == {column}
    assert not diff.missing_columns and not diff.extra_columns


@pytest.mark.parametrize(("n_a", "n_b"), [(6, 5), (5, 6), (6, 0)])
def test_a_row_count_difference_is_reported(tmp_path, n_a, n_b):
    rows = _rows(6)
    a = _write(tmp_path / "a.parquet", rows[:n_a])
    b = _write(tmp_path / "b.parquet", rows[:n_b])

    diff = _script().compare_tables(a, b, key=KEY)

    assert diff.equal is False
    assert (diff.rows_a, diff.rows_b) == (n_a, n_b)
    assert not diff.missing_columns and not diff.extra_columns
