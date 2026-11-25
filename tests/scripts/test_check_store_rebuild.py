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
