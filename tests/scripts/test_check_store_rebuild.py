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


# ------------------------------------------------ optional columns that are new and all null (#215)
#
# Contract pinned here for the optional-column rule. It folds into the accepted-differences
# mechanism of #210 (tests/scripts/test_check_store_rebuild_accepted.py), it does not sit beside it:
#
# * `horsetrack.core.store.schemas.OPTIONAL_COLUMNS: dict[str, tuple[str, ...]]`: for each
#   table that has optional columns, their names (a subset of `TABLES[name].columns`). Tables
#   without optional columns need no entry (readers use `.get(name, ())`). races has 7, runners 8.
# * `compare_tables(a, b, key=None, *, expected_extra=(), optional=())`: `optional` is the table's
#   optional column names. A column in `optional` that `b` has and `a` lacks, holding only nulls in
#   `b`, is not an extra column: it is left out of `extra_columns` and recorded as
#   `diff.accepted[column] = OPTIONAL` (module constant `OPTIONAL = "optional"`, beside ADDED and
#   ALL_NULL), and `equal` can be True. Every other difference is reported as before. There is no
#   `TableDiff.forgiven_columns`.
# * `compare_stores(copied, rebuilt, tables, optional=None) -> (lines, ok)`: `optional` maps a table
#   name to its optional columns (None: no table has any). A table whose only differences are
#   accepted ones reads `<table>: match rows A / B (accepted: optional all-null: a, b)`; the
#   optional part comes after the `added` and `all-null` parts, parts joined with "; ".
# * `main` passes `schemas.OPTIONAL_COLUMNS` to `compare_stores`.

RACES_OPTIONAL = (
    "weight_type", "race_class_raw", "going_raw", "source_race_key",
    "rail_setting", "result_published_utc", "result_unpublished_utc",
)  # fmt: skip
# the three that came into the races schema from main (#222, #225): absent from HKTrack's copy and
# null on every HK row, so the real HK rebuild reports them as new unless they are optional
RACES_FROM_MAIN = ("rail_setting", "result_published_utc", "result_unpublished_utc")
RUNNERS_OPTIONAL = (
    "bracket", "actual_wt_kg", "horse_wt_kg", "horse_wt_change_kg", "last_3f_s", "win_fav_rank",
    "sex", "age",
)  # fmt: skip

ARROW_TYPES = {
    "string": pa.string(),
    "int64": pa.int64(),
    "float64": pa.float64(),
    "bool": pa.bool_(),
    "float-list": pa.list_(pa.float64()),
    "timestamp-utc": pa.timestamp("us", tz="UTC"),
}
SAMPLE_VALUES = {
    "string": "x",
    "int64": 3,
    "float64": 2.5,
    "bool": True,
    "float-list": [1.0, 2.0],
    "timestamp-utc": dt.datetime(2026, 11, 28, 6, 30, tzinfo=dt.timezone.utc),
}


def _store_table(path: Path, n: int = 4, extra: dict | None = None, **kwargs) -> Path:
    """A small table: race_id, horse_no, odds, plus `extra` columns (name -> (type, values))."""
    data = {
        "race_id": [f"r{i // 2}" for i in range(n)],
        "horse_no": [i % 2 + 1 for i in range(n)],
        "odds": [1.5 + i for i in range(n)],
    }
    fields = [("race_id", pa.string()), ("horse_no", pa.int64()), ("odds", pa.float64())]
    for name, (typ, values) in (extra or {}).items():
        data[name] = values
        fields.append((name, typ))
    pq.write_table(pa.table(data, schema=pa.schema(fields)), path, **kwargs)
    return path


def _nulls(n: int) -> list:
    return [None] * n


def test_schema_optional_columns_are_the_documented_ones():
    from horsetrack.core.store import schemas

    assert set(schemas.OPTIONAL_COLUMNS["races"]) == set(RACES_OPTIONAL)
    assert set(schemas.OPTIONAL_COLUMNS["runners"]) == set(RUNNERS_OPTIONAL)
    assert len(schemas.OPTIONAL_COLUMNS["races"]) == 7
    assert set(RACES_FROM_MAIN) <= set(schemas.OPTIONAL_COLUMNS["races"])
    assert len(schemas.OPTIONAL_COLUMNS["runners"]) == 8
    for table, names in schemas.OPTIONAL_COLUMNS.items():
        assert set(names) <= set(schemas.TABLES[table].columns), table
        assert len(set(names)) == len(names)
    # the other tables, and the key columns, are never optional
    others = set(schemas.OPTIONAL_COLUMNS) - {"races", "runners"}
    assert not others, others
    assert not {"race_id", "horse_no", "horse_id"} & set(schemas.OPTIONAL_COLUMNS["runners"])


def test_optional_is_its_own_accepted_reason():
    script = _script()

    assert script.OPTIONAL == "optional"
    assert len({script.OPTIONAL, script.ADDED, script.ALL_NULL}) == 3


@pytest.mark.parametrize("type_name", list(ARROW_TYPES))
@pytest.mark.parametrize("n", [1, 4])
def test_new_all_null_optional_column_is_accepted(tmp_path, type_name, n):
    a = _store_table(tmp_path / "a.parquet", n)
    b = _store_table(
        tmp_path / "b.parquet", n, {"gear": (ARROW_TYPES[type_name], _nulls(n))}, row_group_size=2
    )
    script = _script()

    diff = script.compare_tables(a, b, key=KEY, optional=("gear",))

    assert diff.equal is True
    assert diff.failed_checks() == []
    assert (diff.rows_a, diff.rows_b) == (n, n)
    assert not diff.extra_columns and not diff.missing_columns and not diff.dtype_diffs
    assert diff.accepted == {"gear": script.OPTIONAL}
    assert not hasattr(diff, "forgiven_columns")  # folded into `accepted`
    assert diff.hash_a == diff.hash_b


@pytest.mark.parametrize("type_name", list(ARROW_TYPES))
@pytest.mark.parametrize("position", [0, 2, 3])
def test_new_optional_column_with_any_value_is_still_new(tmp_path, type_name, position):
    n = 4
    values = _nulls(n)
    values[position] = SAMPLE_VALUES[type_name]
    a = _store_table(tmp_path / "a.parquet", n)
    b = _store_table(tmp_path / "b.parquet", n, {"gear": (ARROW_TYPES[type_name], values)})

    diff = _script().compare_tables(a, b, key=KEY, optional=("gear",))

    assert diff.equal is False
    assert diff.failed_checks() == ["columns"]
    assert set(diff.extra_columns) == {"gear"}
    assert not diff.accepted


def test_without_optional_an_all_null_new_column_is_still_new(tmp_path):
    a = _store_table(tmp_path / "a.parquet")
    b = _store_table(tmp_path / "b.parquet", extra={"gear": (pa.string(), _nulls(4))})
    script = _script()

    for diff in (
        script.compare_tables(a, b, key=KEY),
        script.compare_tables(a, b, key=KEY, optional=()),
        script.compare_tables(a, b, key=KEY, optional=("other",)),  # another column's allowance
        script.compare_tables(a, b, key=KEY, expected_extra=("other",)),
    ):
        assert diff.equal is False
        assert set(diff.extra_columns) == {"gear"}
        assert not diff.accepted


def test_only_the_optional_columns_are_accepted(tmp_path):
    a = _store_table(tmp_path / "a.parquet")
    new = {"gear": (pa.string(), _nulls(4)), "extra": (pa.int64(), _nulls(4))}
    b = _store_table(tmp_path / "b.parquet", extra=new)
    script = _script()

    diff = script.compare_tables(a, b, key=KEY, optional=("gear",))

    assert diff.equal is False
    assert set(diff.extra_columns) == {"extra"}
    assert diff.accepted == {"gear": script.OPTIONAL}


def test_an_optional_column_missing_from_the_rebuild_is_still_reported(tmp_path):
    for nulls in (True, False):  # even when the copy's column is all null: only "new" is accepted
        values = _nulls(4) if nulls else [1, 2, 3, 4]
        a = _store_table(tmp_path / f"a{nulls}.parquet", extra={"gear": (pa.int64(), values)})
        b = _store_table(tmp_path / f"b{nulls}.parquet")

        diff = _script().compare_tables(a, b, key=KEY, optional=("gear",))

        assert diff.equal is False
        assert set(diff.missing_columns) == {"gear"}
        assert not diff.extra_columns and not diff.accepted


def test_an_optional_column_both_sides_have_is_compared_as_usual(tmp_path):
    a = _store_table(tmp_path / "a.parquet", extra={"gear": (pa.int64(), [1, 2, 3, 4])})
    same = _store_table(tmp_path / "same.parquet", extra={"gear": (pa.int64(), [1, 2, 3, 4])})
    other = _store_table(tmp_path / "other.parquet", extra={"gear": (pa.int64(), [1, 2, 3, 5])})
    nulls = _store_table(tmp_path / "nulls.parquet", extra={"gear": (pa.int64(), _nulls(4))})
    script = _script()

    assert script.compare_tables(a, same, key=KEY, optional=("gear",)).equal is True
    assert not script.compare_tables(a, same, key=KEY, optional=("gear",)).accepted
    assert script.compare_tables(a, other, key=KEY, optional=("gear",)).equal is False
    assert script.compare_tables(a, nulls, key=KEY, optional=("gear",)).equal is False


@pytest.mark.parametrize("change", ["rows", "content", "dtype", "missing-column"])
def test_other_differences_still_fail_next_to_an_accepted_optional_column(tmp_path, change):
    new = {"gear": (pa.string(), _nulls(4))}
    a = _store_table(tmp_path / "a.parquet")
    b = _store_table(tmp_path / "b.parquet", extra=new)
    if change == "rows":
        b = _store_table(tmp_path / "b.parquet", 3, {"gear": (pa.string(), _nulls(3))})
    elif change == "content":
        table = pq.read_table(b)
        odds = table.column("odds").to_pylist()
        odds[0] += 1.0
        table = table.set_column(2, "odds", pa.array(odds, pa.float64()))
        pq.write_table(table, b)
    elif change == "dtype":
        table = pq.read_table(b)
        table = table.set_column(1, "horse_no", table.column("horse_no").cast(pa.float64()))
        pq.write_table(table, b)
    else:
        pq.write_table(pq.read_table(b).drop_columns(["odds"]), b)
    script = _script()

    diff = script.compare_tables(a, b, key=KEY, optional=("gear",))

    assert diff.equal is False
    assert not diff.extra_columns  # the optional column is not what fails
    assert diff.accepted == {"gear": script.OPTIONAL}


def test_optional_sits_beside_the_other_accepted_reasons(tmp_path):
    a = _store_table(tmp_path / "a.parquet", extra={"cyb": (pa.null(), _nulls(4))})
    new = {
        "cyb": (pa.string(), _nulls(4)),
        "jurisdiction": (pa.string(), ["x"] * 4),
        "gear": (pa.string(), _nulls(4)),
    }
    b = _store_table(tmp_path / "b.parquet", extra=new)
    script = _script()

    diff = script.compare_tables(
        a, b, key=KEY, expected_extra=("jurisdiction",), optional=("gear",)
    )

    assert diff.equal is True
    assert diff.accepted == {
        "cyb": script.ALL_NULL,
        "jurisdiction": script.ADDED,
        "gear": script.OPTIONAL,
    }


def _two_stores(tmp_path: Path) -> tuple[Path, Path]:
    copied, rebuilt = tmp_path / "copied", tmp_path / "rebuilt"
    copied.mkdir()
    rebuilt.mkdir()
    return copied, rebuilt


def _line(lines: list[str], table: str) -> str:
    (line,) = [x for x in lines if x.startswith(f"{table}:")]
    return line


def test_compare_stores_notes_accepted_optional_columns_and_matches(tmp_path):
    copied, rebuilt = _two_stores(tmp_path)
    race_new = {c: (pa.string(), _nulls(4)) for c in RACES_OPTIONAL}
    _store_table(copied / "races.parquet")
    _store_table(rebuilt / "races.parquet", extra=race_new)
    _store_table(copied / "runners.parquet", 6)  # a table with no new columns beside it
    _store_table(rebuilt / "runners.parquet", 6)
    tables = {"races": KEY, "runners": KEY}

    lines, ok = _script().compare_stores(
        copied, rebuilt, tables, optional={"races": RACES_OPTIONAL, "runners": RUNNERS_OPTIONAL}
    )

    assert ok is True
    assert _line(lines, "races") == (
        "races: match rows 4 / 4 (accepted: optional all-null: "
        "weight_type, race_class_raw, going_raw, source_race_key, "
        "rail_setting, result_published_utc, result_unpublished_utc)"
    )
    assert _line(lines, "runners") == "runners: match rows 6 / 6"  # no note when nothing accepted


def test_hk_races_columns_from_main_are_accepted_when_absent_and_all_null(tmp_path):
    """The real HK run: HKTrack's races copy lacks the three columns, the rebuild has them null."""
    from horsetrack.core.store import schemas

    copied, rebuilt = _two_stores(tmp_path)
    types = {c: pa.timestamp("us", tz="UTC") for c in RACES_FROM_MAIN}
    types["rail_setting"] = pa.string()
    for c in RACES_FROM_MAIN:  # the names are really in the races table
        assert c in schemas.TABLES["races"].columns
    _store_table(copied / "races.parquet", 5)
    _store_table(rebuilt / "races.parquet", 5, {c: (t, _nulls(5)) for c, t in types.items()})

    lines, ok = _script().compare_stores(
        copied, rebuilt, {"races": KEY}, optional=schemas.OPTIONAL_COLUMNS
    )

    assert ok is True
    assert _line(lines, "races") == (
        "races: match rows 5 / 5 (accepted: optional all-null: "
        "rail_setting, result_published_utc, result_unpublished_utc)"
    )


@pytest.mark.parametrize("column", RACES_FROM_MAIN)
def test_hk_races_columns_from_main_still_fail_when_they_hold_a_value(tmp_path, column):
    from horsetrack.core.store import schemas

    copied, rebuilt = _two_stores(tmp_path)
    typ = pa.string() if column == "rail_setting" else pa.timestamp("us", tz="UTC")
    value = SAMPLE_VALUES["string" if column == "rail_setting" else "timestamp-utc"]
    extra = {
        c: (pa.string() if c == "rail_setting" else pa.timestamp("us", tz="UTC"), _nulls(4))
        for c in RACES_FROM_MAIN
    }
    extra[column] = (typ, [None, value, None, None])
    _store_table(copied / "races.parquet")
    _store_table(rebuilt / "races.parquet", extra=extra)

    lines, ok = _script().compare_stores(
        copied, rebuilt, {"races": KEY}, optional=schemas.OPTIONAL_COLUMNS
    )

    assert ok is False
    races = _line(lines, "races")
    assert "MISMATCH" in races and f"new: {column}" in races
    for other in set(RACES_FROM_MAIN) - {column}:
        assert other not in races.split("new:")[1]


def test_compare_stores_names_only_the_accepted_columns(tmp_path):
    copied, rebuilt = _two_stores(tmp_path)
    _store_table(copied / "races.parquet", 3)
    extra = {"going_raw": (pa.string(), _nulls(3)), "weight_type": (pa.string(), _nulls(3))}
    _store_table(rebuilt / "races.parquet", 3, extra)

    lines, ok = _script().compare_stores(
        copied, rebuilt, {"races": KEY}, optional={"races": RACES_OPTIONAL}
    )

    assert ok is True
    assert _line(lines, "races") == (
        "races: match rows 3 / 3 (accepted: optional all-null: going_raw, weight_type)"
    )


def test_compare_stores_puts_the_optional_part_after_the_other_accepted_parts(tmp_path):
    copied, rebuilt = _two_stores(tmp_path)
    _store_table(copied / "jrdb_pre_race.parquet", extra={"cyb": (pa.null(), _nulls(4))})
    new = {
        "cyb": (pa.string(), _nulls(4)),
        "jurisdiction": (pa.string(), ["x"] * 4),
        "gear": (pa.string(), _nulls(4)),
    }
    _store_table(rebuilt / "jrdb_pre_race.parquet", extra=new)

    lines, ok = _script().compare_stores(
        copied, rebuilt, {"jrdb_pre_race": KEY}, optional={"jrdb_pre_race": ("gear",)}
    )

    assert ok is True
    assert _line(lines, "jrdb_pre_race") == (
        "jrdb_pre_race: match "
        "(accepted: jurisdiction added; all-null: cyb; optional all-null: gear)"
    )


def test_compare_stores_reports_a_non_null_optional_column_as_new(tmp_path):
    copied, rebuilt = _two_stores(tmp_path)
    _store_table(copied / "races.parquet")
    values = _nulls(4)
    values[2] = "A"
    extra = {"weight_type": (pa.string(), values), "going_raw": (pa.string(), _nulls(4))}
    _store_table(rebuilt / "races.parquet", extra=extra)

    lines, ok = _script().compare_stores(
        copied, rebuilt, {"races": KEY}, optional={"races": RACES_OPTIONAL}
    )

    assert ok is False
    races = _line(lines, "races")
    assert "MISMATCH" in races
    assert "failed columns" in races
    assert "new: weight_type" in races
    assert "going_raw" not in races  # the all-null one is accepted, not named as a failure


def test_compare_stores_without_optional_is_as_before(tmp_path):
    copied, rebuilt = _two_stores(tmp_path)
    _store_table(copied / "races.parquet")
    _store_table(rebuilt / "races.parquet", extra={"weight_type": (pa.string(), _nulls(4))})
    script = _script()

    for kwargs in (
        {},
        {"optional": None},
        {"optional": {}},
        {"optional": {"runners": ("weight_type",)}},
    ):
        lines, ok = script.compare_stores(copied, rebuilt, {"races": KEY}, **kwargs)
        assert ok is False, kwargs
        assert _line(lines, "races") == (
            "races: MISMATCH rows 4 / 4; failed columns; new: weight_type"
        ), kwargs


def test_compare_stores_scopes_optional_columns_per_table(tmp_path):
    copied, rebuilt = _two_stores(tmp_path)
    for name in ("races", "runners"):
        _store_table(copied / f"{name}.parquet")
        extra = {"weight_type": (pa.string(), _nulls(4))}
        _store_table(rebuilt / f"{name}.parquet", extra=extra)

    lines, ok = _script().compare_stores(
        copied, rebuilt, {"races": KEY, "runners": KEY}, optional={"races": ("weight_type",)}
    )

    assert ok is False
    assert "MISMATCH" not in _line(lines, "races")
    assert "MISMATCH" in _line(lines, "runners") and "new: weight_type" in _line(lines, "runners")


def test_compare_stores_still_reports_not_rebuilt_and_other_failures(tmp_path):
    copied, rebuilt = _two_stores(tmp_path)
    _store_table(copied / "races.parquet", extra={"weight_type": (pa.string(), _nulls(4))})
    _store_table(rebuilt / "races.parquet")
    _store_table(copied / "runners.parquet", 4)
    _store_table(rebuilt / "runners.parquet", 5, {"bracket": (pa.int64(), _nulls(5))})
    optional = {"races": RACES_OPTIONAL, "runners": RUNNERS_OPTIONAL}

    lines, ok = _script().compare_stores(
        copied, rebuilt, {"races": KEY, "runners": KEY}, optional=optional
    )

    assert ok is False
    assert "MISMATCH" in _line(lines, "races") and "not rebuilt: weight_type" in _line(
        lines, "races"
    )
    assert "MISMATCH" in _line(lines, "runners") and "rows 4 / 5" in _line(lines, "runners")


@pytest.mark.parametrize(
    ("table", "optional"), [("races", RACES_OPTIONAL), ("runners", RUNNERS_OPTIONAL)]
)
@pytest.mark.parametrize("filled", [False, True])
def test_main_compare_only_exit_code(tmp_path, capsys, table, optional, filled):
    """The whole script on synthetic dirs: the schema's optional columns drive the exit status."""
    source = "hkjc"
    data, work = tmp_path / "data", tmp_path / "work"
    copied, rebuilt = data / "processed" / source, work / "processed" / source
    copied.mkdir(parents=True)
    rebuilt.mkdir(parents=True)
    columns = {c: (pa.int64(), _nulls(4)) for c in optional}
    if filled:
        columns[optional[-1]] = (pa.int64(), [None, None, 7, None])
    _store_table(copied / f"{table}.parquet")
    _store_table(rebuilt / f"{table}.parquet", extra=columns)

    code = _script().main(
        ["--source", source, "--data-dir", str(data), "--work-dir", str(work), "--compare-only"]
    )

    out = capsys.readouterr().out
    line = _line(out.splitlines(), table)
    if filled:
        assert code == 1
        assert "MISMATCH" in line and "new:" in line
        assert optional[-1] in line.split("new:")[1]
    else:
        assert code == 0
        assert "MISMATCH" not in line
        assert f"(accepted: optional all-null: {', '.join(optional)})" in line


# --- prepare_work_dir exposes the upcoming cards (ticket #210 fix round) ---------------------
#
# The JRDB build (`jrdb/tables.py::_store`) also reads `processed/jra/upcoming/<meeting>/` when it
# holds `races.parquet` and `racecard_runners.parquet`. For source jra the work dir therefore links
# `processed/jra/upcoming` to the data dir's folder (when it exists) while `processed/jra` itself
# stays a real folder inside the work dir, so the build writes only there.

MEETINGS = ("2026_0405_tokyo", "2026_1128_kyoto", "2026_1129_hanshin")


def _data_dir(tmp_path: Path, source: str, meetings: tuple[str, ...] | None) -> Path:
    """A minimal data dir: a manifest for `source` and, for jra, upcoming meeting folders."""
    data = tmp_path / "data"
    (data / "raw" / source).mkdir(parents=True)
    (data / "raw" / source / "manifest.sqlite").write_bytes(b"synthetic")
    (data / "processed" / source).mkdir(parents=True)
    for meeting in meetings or ():
        folder = data / "processed" / "jra" / "upcoming" / meeting
        folder.mkdir(parents=True)
        for name in ("races.parquet", "racecard_runners.parquet"):
            _write(folder / name, _rows(3), SCHEMA)
    return data


@pytest.mark.parametrize("meetings", [MEETINGS[:1], MEETINGS])
def test_jra_work_dir_shows_the_upcoming_meetings_and_keeps_processed_jra_its_own(
    tmp_path, meetings
):
    data = _data_dir(tmp_path, "jra", meetings)
    work = tmp_path / "work"

    _script().prepare_work_dir(data, work, "jra")

    upcoming = work / "processed" / "jra" / "upcoming"
    assert upcoming.is_dir()
    assert sorted(p.name for p in upcoming.iterdir()) == sorted(meetings)
    for meeting in meetings:
        for name in ("races.parquet", "racecard_runners.parquet"):
            seen = upcoming / meeting / name
            assert (
                seen.read_bytes() == (data / "processed/jra/upcoming" / meeting / name).read_bytes()
            )
    # the build writes into processed/jra: that folder must be the work dir's own
    jra = work / "processed" / "jra"
    assert jra.resolve().is_relative_to(work.resolve())
    assert not jra.resolve().is_relative_to(data.resolve())
    assert not jra.is_symlink()
    assert not (work / "processed").is_symlink()
    # a file the build writes there does not reach the data dir
    (jra / "races.parquet").write_bytes(b"built")
    assert not (data / "processed" / "jra" / "races.parquet").exists()


def test_jra_work_dir_without_upcoming_in_data_links_nothing(tmp_path):
    data = _data_dir(tmp_path, "jra", None)
    work = tmp_path / "work"

    _script().prepare_work_dir(data, work, "jra")

    assert not (work / "processed" / "jra" / "upcoming").exists()
    assert not (work / "processed" / "jra" / "upcoming").is_symlink()
    assert (work / "processed").is_dir()


def test_non_jra_work_dir_links_no_upcoming(tmp_path):
    data = _data_dir(tmp_path, "nar", None)
    folder = data / "processed" / "jra" / "upcoming" / MEETINGS[0]
    folder.mkdir(parents=True)
    _write(folder / "races.parquet", _rows(3), SCHEMA)
    work = tmp_path / "work"

    _script().prepare_work_dir(data, work, "nar")

    assert not (work / "processed" / "jra").exists()
    assert not any(p.name == "upcoming" for p in (work / "processed").rglob("*"))


def test_a_build_writing_through_the_upcoming_link_is_caught_by_the_data_dir_guard(
    tmp_path, monkeypatch, capsys
):
    """The link must not weaken the guard: the fingerprint covers processed/jra incl. upcoming."""
    data = _data_dir(tmp_path, "jra", MEETINGS)
    work = tmp_path / "work"
    script = _script()
    saw = {}

    def build(work_dir, source, workers):
        upcoming = work_dir / "processed" / "jra" / "upcoming"
        saw["meetings"] = sorted(p.name for p in upcoming.iterdir())
        (upcoming / MEETINGS[0] / "stray.parquet").write_bytes(b"x")  # writes into the data dir
        return []

    monkeypatch.setattr(script, "rebuild", build)
    monkeypatch.setattr(script, "compare_stores", lambda *a, **k: ([], True))

    code = script.main(["--source", "jra", "--data-dir", str(data), "--work-dir", str(work)])

    assert saw["meetings"] == sorted(MEETINGS)
    assert code == 2
    assert "data dir changed" in capsys.readouterr().out


def test_a_build_that_only_reads_upcoming_leaves_the_data_dir_unchanged(tmp_path, monkeypatch):
    data = _data_dir(tmp_path, "jra", MEETINGS)
    work = tmp_path / "work"
    script = _script()

    def build(work_dir, source, workers):
        upcoming = work_dir / "processed" / "jra" / "upcoming"
        for folder in sorted(upcoming.iterdir()):
            pq.read_table(folder / "races.parquet")
        (work_dir / "processed" / "jra" / "jrdb_pre_race.parquet").write_bytes(b"built")
        return []

    monkeypatch.setattr(script, "rebuild", build)
    monkeypatch.setattr(script, "compare_stores", lambda *a, **k: ([], True))

    code = script.main(["--source", "jra", "--data-dir", str(data), "--work-dir", str(work)])

    assert code == 0
    assert not (data / "processed" / "jra" / "jrdb_pre_race.parquet").exists()
