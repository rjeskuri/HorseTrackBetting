"""scripts/check_store_rebuild.py (ticket #210): accepted differences in the jrdb_* tables.

Every test writes small synthetic Parquet files under tmp_path; no test reads the data dir and no
value here comes from JRDB.

Contract pinned here (the names the implementation must follow):

* `compare_tables(a, b, key=None, *, expected_extra=())`: `expected_extra` names columns that `b`
  (the rebuild) may have and `a` (the copy) lacks. Such a column is left out of
  `TableDiff.extra_columns` and does not fail the `columns` check. A column in `b` only that is not
  listed still counts, and a listed column that `a` has but `b` lacks stays in `missing_columns`.
* `compare_tables` also accepts, for every table: a column whose type in `a` is Arrow `null` and
  whose values in `b` are all null (`b`'s type is any other type). It is left out of
  `dtype_diffs` and does not fail the `dtypes` check. A `b` column with one value is still a
  dtype difference, and so is a column typed `null` in `b` only.
* `TableDiff.accepted` is a mapping keyed by each accepted column (both kinds above; the values,
  a reason, are the script's to choose; an empty mapping when nothing was accepted). A table whose
  only differences are accepted ones has `equal` True and `failed_checks() == []`; every other
  check (rows, content hash, other columns and types) behaves as before.
* `compare_stores(copied, rebuilt, tables)` takes the expectation from a module-level mapping
  `EXPECTED_ADDITIONS = {"jrdb_pre_race": ("jurisdiction",), "jrdb_post_race": ("jurisdiction",)}`
  and passes it as `expected_extra` per table, so `jurisdiction` is accepted for the jrdb_* tables
  only. The line of a table with accepted differences starts `<name>: match`, names each accepted
  column, holds no `MISMATCH` and, for a jrdb_* table, no row counts; the `ok` flag stays True.
* `main(["--source", "jra", "--compare-only", ...])` exits 0 for such tables, 1 for a real
  difference.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import random
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_store_rebuild.py"
MODULE = "check_store_rebuild"

KEY = ("race_id", "horse_no")
JRDB = ["jrdb_pre_race", "jrdb_post_race"]
ROW_COUNTS = re.compile(r"\d+\s*/\s*\d+")

COPY_SCHEMA = pa.schema(
    [
        ("race_id", pa.string()),
        ("horse_no", pa.int64()),
        ("race_date", pa.date32()),
        ("score", pa.float64()),
        ("cyb_a", pa.null()),
        ("sed_remarks", pa.null()),
    ]
)


def _script():
    if not SCRIPT.is_file():
        pytest.fail(f"{SCRIPT.relative_to(REPO)} does not exist", pytrace=False)
    if MODULE not in sys.modules:
        spec = importlib.util.spec_from_file_location(MODULE, SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[MODULE]
            raise
    return sys.modules[MODULE]


def _rows(n: int) -> list[dict]:
    return [
        {
            "race_id": f"2026{i // 3:04d}",
            "horse_no": i % 3 + 1,
            "race_date": dt.date(2026, 3, 1) + dt.timedelta(days=i // 3),
            "score": None if i == 2 else round(10.0 + i * 1.3, 1),
            "cyb_a": None,
            "sed_remarks": None,
        }
        for i in range(n)
    ]


def _write(path: Path, rows: list[dict], schema: pa.Schema) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def _with(schema: pa.Schema, **types: pa.DataType | None) -> pa.Schema:
    """`schema` with the given columns retyped (a None type drops the column)."""
    fields = []
    for f in schema:
        if f.name in types:
            if types[f.name] is not None:
                fields.append(pa.field(f.name, types[f.name]))
        else:
            fields.append(f)
    return pa.schema(fields)


def _rebuild_schema(null_type: pa.DataType | None = None, *, jurisdiction: bool = True):
    null_type = null_type or pa.string()
    schema = _with(COPY_SCHEMA, cyb_a=null_type, sed_remarks=null_type)
    if jurisdiction:
        schema = pa.schema([("jurisdiction", pa.string()), *schema])
    return schema


def _rebuild_rows(rows: list[dict], *, jurisdiction: bool = True) -> list[dict]:
    return [({"jurisdiction": "JP"} if jurisdiction else {}) | r for r in rows]


def _stores(tmp_path: Path, name: str, copy_rows, copy_schema, rebuilt_rows, rebuilt_schema):
    copied, rebuilt = tmp_path / "copied", tmp_path / "rebuilt"
    _write(copied / f"{name}.parquet", copy_rows, copy_schema)
    _write(rebuilt / f"{name}.parquet", rebuilt_rows, rebuilt_schema)
    return copied, rebuilt


def _line(lines: list[str], name: str) -> str:
    (line,) = [ln for ln in lines if ln.startswith(f"{name}:")]
    return line


# ------------------------------------------------------------ jurisdiction, an expected addition


@pytest.mark.parametrize("name", JRDB)
@pytest.mark.parametrize("n", [1, 5, 9])
def test_jrdb_table_with_added_jurisdiction_matches(tmp_path, name, n):
    rows = _rows(n)
    copied, rebuilt = _stores(
        tmp_path, name, rows, COPY_SCHEMA, _rebuild_rows(rows), _rebuild_schema()
    )

    lines, ok = _script().compare_stores(copied, rebuilt, {name: KEY})

    assert ok is True
    line = _line(lines, name)
    assert line.startswith(f"{name}: match")
    assert "MISMATCH" not in line
    assert not ROW_COUNTS.search(line), "jrdb_* lines carry no row counts"
    assert "jurisdiction" in line, "the line names the accepted difference"


def test_compare_tables_accepts_a_listed_extra_column_only_when_told(tmp_path):
    rows = _rows(6)
    # same-typed content columns, so only jurisdiction differs
    a = _write(
        tmp_path / "a.parquet",
        rows,
        _with(COPY_SCHEMA, cyb_a=pa.string(), sed_remarks=pa.string()),
    )
    b = _write(
        tmp_path / "b.parquet",
        _rebuild_rows(rows),
        _rebuild_schema(pa.string()),
    )

    plain = _script().compare_tables(a, b, key=KEY)
    expected = _script().compare_tables(a, b, key=KEY, expected_extra=("jurisdiction",))

    assert plain.equal is False
    assert set(plain.extra_columns) == {"jurisdiction"}
    assert not plain.accepted

    assert expected.equal is True
    assert expected.failed_checks() == []
    assert not expected.extra_columns
    assert set(expected.accepted) == {"jurisdiction"}
    assert expected.hash_a == expected.hash_b
    assert (expected.rows_a, expected.rows_b) == (6, 6)


@pytest.mark.parametrize("name", ["races", "results", "horses"])
def test_jurisdiction_added_to_a_non_jrdb_table_is_a_mismatch(tmp_path, name):
    rows = _rows(5)
    copied, rebuilt = _stores(
        tmp_path, name, rows, COPY_SCHEMA, _rebuild_rows(rows), _rebuild_schema()
    )

    lines, ok = _script().compare_stores(copied, rebuilt, {name: KEY})

    assert ok is False
    line = _line(lines, name)
    assert "MISMATCH" in line
    assert "jurisdiction" in line


@pytest.mark.parametrize("name", JRDB)
@pytest.mark.parametrize("extra", ["venue_code", "jurisdictions", "Jurisdiction"])
def test_another_extra_column_on_a_jrdb_table_is_a_mismatch(tmp_path, name, extra):
    rows = _rows(5)
    rebuilt_schema = pa.schema([*_rebuild_schema(), (extra, pa.string())])
    rebuilt_rows = [{**r, extra: "x"} for r in _rebuild_rows(rows)]
    copied, rebuilt = _stores(tmp_path, name, rows, COPY_SCHEMA, rebuilt_rows, rebuilt_schema)

    lines, ok = _script().compare_stores(copied, rebuilt, {name: KEY})

    assert ok is False
    line = _line(lines, name)
    assert "MISMATCH" in line
    assert extra in line


@pytest.mark.parametrize("name", JRDB)
def test_jurisdiction_missing_from_the_rebuild_is_a_mismatch(tmp_path, name):
    rows = _rows(5)
    copy_schema = pa.schema([("jurisdiction", pa.string()), *COPY_SCHEMA])
    copy_rows = _rebuild_rows(rows)
    copied, rebuilt = _stores(
        tmp_path, name, copy_rows, copy_schema, rows, _rebuild_schema(jurisdiction=False)
    )

    lines, ok = _script().compare_stores(copied, rebuilt, {name: KEY})
    diff = _script().compare_tables(
        copied / f"{name}.parquet",
        rebuilt / f"{name}.parquet",
        key=KEY,
        expected_extra=("jurisdiction",),
    )

    assert ok is False
    assert "MISMATCH" in _line(lines, name)
    assert diff.equal is False
    assert set(diff.missing_columns) == {"jurisdiction"}
    assert "columns" in diff.failed_checks()


# ------------------------------------------------------------ all-null columns typed null


@pytest.mark.parametrize("rebuilt_type", [pa.string(), pa.date32()], ids=["string", "date32"])
@pytest.mark.parametrize("columns", [["cyb_a"], ["sed_remarks"], ["cyb_a", "sed_remarks"]])
def test_null_typed_copy_column_against_all_null_rebuild_column_is_no_difference(
    tmp_path, rebuilt_type, columns
):
    rows = _rows(7)
    # only `columns` change type in the rebuild; the other null column keeps the copy's type
    types = {c: rebuilt_type for c in columns}
    b_schema = _with(COPY_SCHEMA, **types)
    a = _write(tmp_path / "a.parquet", rows, COPY_SCHEMA)
    b = _write(tmp_path / "b.parquet", rows, b_schema)

    diff = _script().compare_tables(a, b, key=KEY)

    assert diff.equal is True
    assert diff.failed_checks() == []
    assert not diff.dtype_diffs
    assert set(diff.accepted) == set(columns)
    assert not diff.missing_columns and not diff.extra_columns
    assert (diff.rows_a, diff.rows_b) == (7, 7)
    assert diff.hash_a == diff.hash_b


def test_a_plain_equal_pair_accepts_nothing(tmp_path):
    rows = _rows(4)
    a = _write(tmp_path / "a.parquet", rows, COPY_SCHEMA)
    b = _write(tmp_path / "b.parquet", rows, COPY_SCHEMA)

    diff = _script().compare_tables(a, b, key=KEY)

    assert diff.equal is True
    assert not diff.accepted


@pytest.mark.parametrize("name", JRDB)
@pytest.mark.parametrize("rebuilt_type", [pa.string(), pa.date32()], ids=["string", "date32"])
def test_null_columns_line_reads_as_match(tmp_path, name, rebuilt_type):
    rows = _rows(6)
    copied, rebuilt = _stores(
        tmp_path,
        name,
        rows,
        COPY_SCHEMA,
        _rebuild_rows(rows),
        _rebuild_schema(rebuilt_type),
    )

    lines, ok = _script().compare_stores(copied, rebuilt, {name: KEY})

    assert ok is True
    line = _line(lines, name)
    assert line.startswith(f"{name}: match")
    assert "MISMATCH" not in line
    assert not ROW_COUNTS.search(line)
    assert "cyb_a" in line and "sed_remarks" in line


@pytest.mark.parametrize("rebuilt_type", [pa.string(), pa.date32()], ids=["string", "date32"])
@pytest.mark.parametrize("row", [0, 3, 6])
def test_a_rebuild_column_with_a_value_is_still_a_mismatch(tmp_path, rebuilt_type, row):
    rows = _rows(7)
    value = "ok" if rebuilt_type == pa.string() else dt.date(2026, 3, 2)
    rebuilt_rows = [dict(r) for r in rows]
    rebuilt_rows[row]["cyb_a"] = value
    a = _write(tmp_path / "a.parquet", rows, COPY_SCHEMA)
    b = _write(
        tmp_path / "b.parquet",
        rebuilt_rows,
        _with(COPY_SCHEMA, cyb_a=rebuilt_type, sed_remarks=rebuilt_type),
    )

    diff = _script().compare_tables(a, b, key=KEY)

    assert diff.equal is False
    assert "dtypes" in diff.failed_checks()
    assert set(diff.dtype_diffs) == {"cyb_a"}  # sed_remarks, all null, is accepted
    assert set(diff.accepted) == {"sed_remarks"}


@pytest.mark.parametrize("name", JRDB)
def test_a_rebuild_value_in_a_null_column_fails_the_jrdb_line(tmp_path, name):
    rows = _rows(6)
    rebuilt_rows = _rebuild_rows(rows)
    rebuilt_rows[4]["sed_remarks"] = "x"
    copied, rebuilt = _stores(
        tmp_path, name, rows, COPY_SCHEMA, rebuilt_rows, _rebuild_schema(pa.string())
    )

    lines, ok = _script().compare_stores(copied, rebuilt, {name: KEY})

    assert ok is False
    line = _line(lines, name)
    assert "MISMATCH" in line
    assert "sed_remarks" in line


def test_a_null_typed_rebuild_column_against_a_typed_copy_column_is_still_a_mismatch(tmp_path):
    # guard (passes today): only a null-typed COPY column is accepted, not the reverse
    rows = _rows(5)
    a = _write(tmp_path / "a.parquet", rows, _with(COPY_SCHEMA, cyb_a=pa.string()))
    b = _write(tmp_path / "b.parquet", rows, COPY_SCHEMA)

    diff = _script().compare_tables(a, b, key=KEY)

    assert diff.equal is False
    assert set(diff.dtype_diffs) == {"cyb_a"}
    assert "cyb_a" not in diff.accepted


def test_a_non_null_type_change_between_all_null_columns_is_still_a_mismatch(tmp_path):
    # guard (passes today): string vs date32, both all null, is not the accepted null-vs-typed case
    rows = _rows(5)
    a = _write(tmp_path / "a.parquet", rows, _with(COPY_SCHEMA, cyb_a=pa.string()))
    b = _write(tmp_path / "b.parquet", rows, _with(COPY_SCHEMA, cyb_a=pa.date32()))

    diff = _script().compare_tables(a, b, key=KEY)

    assert diff.equal is False
    assert set(diff.dtype_diffs) == {"cyb_a"}


# ------------------------------------------------------------ accepted differences hide nothing


def _row_count_change(rows):
    return rows[:-1]


def _value_change(rows):
    changed = [dict(r) for r in rows]
    changed[1]["score"] = 99.5
    return changed


def _value_to_null(rows):
    changed = [dict(r) for r in rows]
    changed[3]["score"] = None
    return changed


def _date_change(rows):
    changed = [dict(r) for r in rows]
    changed[0]["race_date"] = dt.date(2027, 1, 1)
    return changed


@pytest.mark.parametrize("name", JRDB)
@pytest.mark.parametrize(
    ("change", "failed"),
    [
        (_row_count_change, "rows"),
        (_value_change, "content"),
        (_value_to_null, "content"),
        (_date_change, "content"),
    ],
    ids=["rows", "value", "to-null", "date"],
)
def test_accepted_differences_do_not_hide_a_content_or_row_difference(
    tmp_path, name, change, failed
):
    rows = _rows(8)
    copied, rebuilt = _stores(
        tmp_path,
        name,
        rows,
        COPY_SCHEMA,
        _rebuild_rows(change(rows)),
        _rebuild_schema(pa.date32()),
    )

    lines, ok = _script().compare_stores(copied, rebuilt, {name: KEY})
    diff = _script().compare_tables(
        copied / f"{name}.parquet",
        rebuilt / f"{name}.parquet",
        key=KEY,
        expected_extra=("jurisdiction",),
    )

    assert ok is False
    line = _line(lines, name)
    assert "MISMATCH" in line
    assert not ROW_COUNTS.search(line)
    assert diff.failed_checks() == [failed]
    assert set(diff.accepted) == {"jurisdiction", "cyb_a", "sed_remarks"}


@pytest.mark.parametrize("name", JRDB)
@pytest.mark.parametrize("kind", ["dtype", "missing"])
def test_accepted_differences_do_not_hide_a_schema_difference(tmp_path, name, kind):
    rows = _rows(6)
    rebuilt_schema = _rebuild_schema(pa.string())
    rebuilt_rows = _rebuild_rows(rows)
    if kind == "dtype":
        rebuilt_schema = _with(rebuilt_schema, score=pa.float32())
        rebuilt_rows = [{**r, "score": r["score"]} for r in rebuilt_rows]
        expected_check = "dtypes"
    else:
        rebuilt_schema = _with(rebuilt_schema, score=None)
        rebuilt_rows = [{k: v for k, v in r.items() if k != "score"} for r in rebuilt_rows]
        expected_check = "columns"
    copied, rebuilt = _stores(tmp_path, name, rows, COPY_SCHEMA, rebuilt_rows, rebuilt_schema)

    lines, ok = _script().compare_stores(copied, rebuilt, {name: KEY})
    diff = _script().compare_tables(
        copied / f"{name}.parquet",
        rebuilt / f"{name}.parquet",
        key=KEY,
        expected_extra=("jurisdiction",),
    )

    assert ok is False
    assert "MISMATCH" in _line(lines, name)
    assert expected_check in diff.failed_checks()
    assert set(diff.dtype_diffs) <= {"score"}
    assert "jurisdiction" not in diff.extra_columns


@pytest.mark.parametrize("name", JRDB)
@pytest.mark.parametrize("seed", [0, 1])
def test_both_accepted_differences_together_with_shuffled_rows_match(tmp_path, name, seed):
    rows = _rows(9)
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    assert shuffled != rows
    copied, rebuilt = _stores(
        tmp_path,
        name,
        rows,
        COPY_SCHEMA,
        _rebuild_rows(shuffled),
        _rebuild_schema(pa.string()),
    )

    lines, ok = _script().compare_stores(copied, rebuilt, {name: KEY})

    assert ok is True
    assert _line(lines, name).startswith(f"{name}: match")


def test_compare_stores_mixes_accepted_and_failing_tables(tmp_path):
    rows = _rows(5)
    copied, rebuilt = tmp_path / "copied", tmp_path / "rebuilt"
    _write(copied / "jrdb_pre_race.parquet", rows, COPY_SCHEMA)
    _write(rebuilt / "jrdb_pre_race.parquet", _rebuild_rows(rows), _rebuild_schema())
    _write(copied / "races.parquet", rows, COPY_SCHEMA)
    _write(rebuilt / "races.parquet", rows[:-1], COPY_SCHEMA)

    lines, ok = _script().compare_stores(
        copied, rebuilt, {"jrdb_pre_race": KEY, "races": KEY, "jrdb_post_race": KEY}
    )

    assert ok is False
    assert _line(lines, "jrdb_pre_race").startswith("jrdb_pre_race: match")
    assert "MISMATCH" in _line(lines, "races")
    assert not any(ln.startswith("jrdb_post_race:") for ln in lines)  # neither side has it


# ------------------------------------------------------------ the command's exit status


@pytest.mark.parametrize("break_content", [False, True], ids=["accepted-only", "real-difference"])
def test_compare_only_exit_status(tmp_path, capsys, break_content):
    rows = _rows(6)
    data, work = tmp_path / "data", tmp_path / "work"
    for name in JRDB:
        rebuilt_rows = _rebuild_rows(rows)
        if break_content and name == "jrdb_post_race":
            rebuilt_rows[2]["score"] = 77.7
        _write(data / "processed" / "jra" / f"{name}.parquet", rows, COPY_SCHEMA)
        _write(work / "processed" / "jra" / f"{name}.parquet", rebuilt_rows, _rebuild_schema())
    (data / "raw" / "jra").mkdir(parents=True)

    code = _script().main(
        ["--source", "jra", "--data-dir", str(data), "--work-dir", str(work), "--compare-only"]
    )
    out = capsys.readouterr().out

    assert code == (1 if break_content else 0), out
    assert ("MISMATCH" in out) is break_content
    assert "jrdb_pre_race: match" in out
    assert not ROW_COUNTS.search(out)
