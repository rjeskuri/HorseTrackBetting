"""Rebuild a source's store from the copied raw cache and compare it with the copied tables (#59).

    uv run python3 scripts/check_store_rebuild.py --source jra [--data-dir DIR] [--work-dir DIR]
        [--workers N] [--compare-only]

The data dir (default: HORSETRACK_DATA_DIR) is only read. The build runs in a work dir (default: a
new temporary one; a given one must be empty) laid out like a data dir: `raw/<source>/` is a real
folder holding a copy of the manifest and a symlink to each page folder, so the parse cache, the
tables and the reports all land in the work dir and opening the manifest creates no -shm file
beside the original. For jra the JRDB build runs too, on `raw/jrdb` and `jrdb/specs` linked the
same way. Then every table of `processed/<source>` is compared with its rebuild by
`compare_tables`: row count, column set, Arrow types and a content hash of the rows sorted by the
table's key (`schemas.TABLES[name].sort_by`). A work dir inside the data dir (or the reverse) is
refused, and the data dir's `raw/<source>`, `processed/<source>` (and the JRDB folders) are
fingerprinted (path, size, mtime) before and after: a change fails the run.

Prints one line per table and no row contents; jrdb_* lines carry no row counts (licensed data).
`--compare-only` compares an earlier run's work dir again without building. Exit status: 0 when
every table matches, 1 when one differs, 2 when the run is refused or the data dir changed.

Accepted differences (#210), settled against HKTrack's copy and named on the table's line
(`jrdb_pre_race: match (accepted: jurisdiction added; all-null: cyb_a, sed_remarks)`); they do not
fail a table, and every other difference still does:
- `EXPECTED_ADDITIONS`: a column the rebuild has and the copy lacks (`jurisdiction` on jrdb_pre_race
  and jrdb_post_race, added after the copy was made).
- every table: a column the copy types as Arrow `null` (it was all null when written) that the
  rebuild types otherwise but holds only nulls in. One value in the rebuild is still a difference.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

MANIFEST = "manifest.sqlite"
JRDB_LINKS = ("raw/jrdb", "jrdb")  # the JRDB archives and the field-spec tables, for jra
# columns the rebuild may have and HKTrack's copy lacks, per table (see the module docstring)
EXPECTED_ADDITIONS: dict[str, tuple[str, ...]] = {
    "jrdb_pre_race": ("jurisdiction",),
    "jrdb_post_race": ("jurisdiction",),
}
ADDED = "added"
ALL_NULL = "all-null"
_BITS_AS_INT = {16: pa.int16(), 32: pa.int32(), 64: pa.int64()}


class Refused(Exception):
    """The work dir would let the build write into the data dir, or would not start fresh."""


@dataclass
class TableDiff:
    rows_a: int
    rows_b: int
    missing_columns: list[str] = field(default_factory=list)  # in a, not in b
    extra_columns: list[str] = field(default_factory=list)  # in b, not in a
    dtype_diffs: dict[str, tuple[str, str]] = field(default_factory=dict)  # column -> (a, b)
    hash_a: str = ""
    hash_b: str = ""
    accepted: dict[str, str] = field(default_factory=dict)  # column -> why it is not a difference

    @property
    def equal(self) -> bool:
        return not self.failed_checks()

    def failed_checks(self) -> list[str]:
        rows_differ = self.rows_a != self.rows_b
        checks = [
            ("rows", rows_differ),
            ("columns", bool(self.missing_columns or self.extra_columns)),
            ("dtypes", bool(self.dtype_diffs)),
            # the hashes differ by construction when the row counts do: the difference is named once
            ("content", not rows_differ and self.hash_a != self.hash_b),
        ]
        return [name for name, failed in checks if failed]


# ---------------------------------------------------------------- comparing two tables


def _sortable(dtype: pa.DataType) -> bool:
    return not (pa.types.is_nested(dtype) or pa.types.is_null(dtype))


def _array(column: pa.ChunkedArray | pa.Array) -> pa.Array:
    return column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column


def _digest_array(h: hashlib._Hash, arr: pa.Array) -> None:
    """Feed the logical content of `arr` to `h`: validity, then values with every null slot
    normalised, so neither the file layout nor what a null slot happens to hold matters."""
    dtype = arr.type
    h.update(str(dtype).encode())
    h.update(np.asarray(pc.is_valid(arr), dtype=bool).tobytes())
    if pa.types.is_null(dtype) or not len(arr):
        return
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        lengths = pc.fill_null(pc.list_value_length(arr), 0)
        h.update(np.asarray(lengths, dtype=np.int64).tobytes())
        _digest_array(h, arr.flatten())  # the logical values only: offsets and nulls applied
        return
    if pa.types.is_string(dtype) or pa.types.is_binary(dtype):
        filled = _array(pc.fill_null(arr, pa.scalar(b"" if pa.types.is_binary(dtype) else "")))
        offsets = np.frombuffer(filled.buffers()[1], dtype=np.int32)
        offsets = offsets[filled.offset : filled.offset + len(filled) + 1]
        h.update((offsets - offsets[0]).astype(np.int64).tobytes())
        data = filled.buffers()[2]
        if data is not None:
            h.update(data[int(offsets[0]) : int(offsets[-1])].to_pybytes())
        return
    if pa.types.is_boolean(dtype):
        arr = pc.cast(arr, pa.int8())
    elif pa.types.is_temporal(dtype) or pa.types.is_floating(dtype):
        arr = arr.view(_BITS_AS_INT[dtype.bit_width])  # the exact bit patterns
    if pa.types.is_integer(arr.type):
        h.update(np.asarray(pc.fill_null(arr, 0)).tobytes())
        return
    h.update(repr(arr.to_pylist()).encode())  # any other type: slow, but exact


def _content_hash(table: pa.Table, key: Sequence[str]) -> str:
    """sha256 over the columns of `table` in order, its rows sorted by `key`, then by every other
    sortable column (so rows tied on a non-unique key still come out in one order)."""
    rest = [c for c in table.column_names if c not in key and _sortable(table.schema.field(c).type)]
    order = [(c, "ascending") for c in (*key, *rest)]
    if order and table.num_rows:
        table = table.take(pc.sort_indices(table, sort_keys=order))  # nulls last
    h = hashlib.sha256()
    for name in table.column_names:
        h.update(name.encode() + b"\0")
        _digest_array(h, _array(table.column(name)))
    return h.hexdigest()


def _all_null(path: Path, name: str) -> bool:
    column = pq.read_table(path, columns=[name]).column(name)
    return column.null_count == len(column)


def compare_tables(
    a: Path, b: Path, key: Sequence[str] | None = None, *, expected_extra: Sequence[str] = ()
) -> TableDiff:
    """Compare two Parquet tables; reads them (one after the other) and writes nothing.

    The content hashes cover the columns both sides have with the same type, so a schema
    difference does not hide whether the rest of the content matches. `key` None: sort by every
    column. Two kinds of difference are accepted and recorded in `TableDiff.accepted`: a column of
    `expected_extra` that only b has, and a column typed `null` in a whose values in b are all
    null (b's type is any other)."""
    schema_a, schema_b = pq.read_schema(a), pq.read_schema(b)
    names_a, names_b = set(schema_a.names), set(schema_b.names)
    diff = TableDiff(
        rows_a=pq.ParquetFile(a).metadata.num_rows,
        rows_b=pq.ParquetFile(b).metadata.num_rows,
        missing_columns=[c for c in schema_a.names if c not in names_b],
        extra_columns=[c for c in schema_b.names if c not in names_a and c not in expected_extra],
        accepted={c: ADDED for c in schema_b.names if c not in names_a and c in expected_extra},
    )
    shared = []
    for name in schema_a.names:
        if name not in names_b:
            continue
        type_a, type_b = schema_a.field(name).type, schema_b.field(name).type
        if type_a == type_b:
            shared.append(name)
        elif pa.types.is_null(type_a) and not pa.types.is_null(type_b) and _all_null(b, name):
            diff.accepted[name] = ALL_NULL
        else:
            diff.dtype_diffs[name] = (str(type_a), str(type_b))
    sort_key = [c for c in (key if key is not None else shared) if c in shared]
    diff.hash_a = _content_hash(pq.read_table(a, columns=shared), sort_key)
    diff.hash_b = _content_hash(pq.read_table(b, columns=shared), sort_key)
    return diff


# ---------------------------------------------------------------- the work dir and the guard


def _inside(path: Path, other: Path) -> bool:
    return path == other or path.is_relative_to(other)


def fingerprint(root: Path, subdirs: Sequence[str]) -> dict[str, tuple[int, int]]:
    """(size, mtime_ns) of every file and folder under root/<subdir>, without following links."""
    out: dict[str, tuple[int, int]] = {}
    for sub in subdirs:
        top = root / sub
        if not top.exists():
            continue
        for folder, dirs, files in os.walk(top):
            for name in [*dirs, *files]:
                st = (Path(folder) / name).lstat()
                out[str((Path(folder) / name).relative_to(root))] = (st.st_size, st.st_mtime_ns)
        st = top.lstat()
        out[sub] = (st.st_size, st.st_mtime_ns)
    return out


def prepare_work_dir(data: Path, work: Path, source: str) -> None:
    """Lay out `work` as a data dir that reads the raw cache of `data` and writes only to itself."""
    data, work = data.resolve(), work.resolve()
    if _inside(work, data) or _inside(data, work):
        raise Refused(f"work dir {work} and data dir {data} must not contain one another")
    if work.exists() and any(work.iterdir()):
        raise Refused(f"work dir {work} is not empty: the build needs a fresh parse cache")
    raw_src = data / "raw" / source
    if not (raw_src / MANIFEST).is_file():
        raise Refused(f"no manifest at {raw_src / MANIFEST}")
    raw_dst = work / "raw" / source
    raw_dst.mkdir(parents=True)
    for entry in sorted(raw_src.iterdir()):
        if entry.name.startswith(MANIFEST):  # the database and its -wal: copied, never opened
            if not entry.name.endswith("-shm"):
                shutil.copyfile(entry, raw_dst / entry.name)
        elif entry.is_dir():
            (raw_dst / entry.name).symlink_to(entry, target_is_directory=True)
    if source == "jra":
        for rel in JRDB_LINKS:
            if (data / rel).is_dir():
                (work / rel).parent.mkdir(parents=True, exist_ok=True)
                (work / rel).symlink_to(data / rel, target_is_directory=True)
    for rel in ("interim", "processed", "reports"):  # where the builds write: real folders here
        (work / rel).mkdir(parents=True, exist_ok=True)
        if _inside((work / rel).resolve(), data):
            raise Refused(f"{work / rel} resolves into the data dir")


def _guarded(source: str) -> list[str]:
    return [f"raw/{source}", f"processed/{source}", *(JRDB_LINKS if source == "jra" else ())]


def rebuild(work: Path, source: str, workers: int) -> list[str]:
    """Run the store build (and for jra the JRDB build) with the work dir as the data dir."""
    os.environ["HORSETRACK_DATA_DIR"] = str(work)  # read on every call; the workers inherit it
    from horsetrack.core.store.build import build_tables

    lines = build_tables(source, workers=workers).lines()
    if source == "jra":
        from horsetrack.countries.japan.jrdb import tables as jrdb_tables

        outcome = jrdb_tables.run()
        if outcome.error:
            raise RuntimeError(f"jrdb build failed: {outcome.error}")
        lines.append(f"jrdb build: {', '.join(jrdb_tables.TABLES)} written")
    return lines


def _accepted_note(diff: TableDiff) -> str:
    """` (accepted: a added; all-null: b, c)` for the accepted columns, or nothing."""
    parts = []
    for reason, label in ((ADDED, "{} added"), (ALL_NULL, "all-null: {}")):
        names = [c for c, why in diff.accepted.items() if why == reason]
        if names:
            parts.append(label.format(", ".join(names)))
    return f" (accepted: {'; '.join(parts)})" if parts else ""


def compare_stores(
    copied: Path, rebuilt: Path, tables: Mapping[str, Sequence[str]]
) -> tuple[list[str], bool]:
    """One line per table, match / MISMATCH with the failed checks (no row contents), and whether
    every table matched. A table neither side has is skipped."""
    lines, ok = [], True
    for name, key in tables.items():
        a, b = copied / f"{name}.parquet", rebuilt / f"{name}.parquet"
        if not a.exists() or not b.exists():
            if a.exists() or b.exists():
                lines.append(
                    f"{name}: MISMATCH, no table in the {'rebuild' if a.exists() else 'copy'}"
                )
                ok = False
            continue
        diff = compare_tables(a, b, key=key, expected_extra=EXPECTED_ADDITIONS.get(name, ()))
        rows = "" if name.startswith("jrdb_") else f" rows {diff.rows_a} / {diff.rows_b}"
        if diff.equal:
            lines.append(f"{name}: match{rows}{_accepted_note(diff)}")
            continue
        ok = False
        detail = [f"failed {','.join(diff.failed_checks())}"]
        if diff.missing_columns:
            detail.append(f"not rebuilt: {','.join(diff.missing_columns)}")
        if diff.extra_columns:
            detail.append(f"new: {','.join(diff.extra_columns)}")
        if diff.dtype_diffs:
            detail.append(" ".join(f"{c}:{x}->{y}" for c, (x, y) in diff.dtype_diffs.items()))
        lines.append(f"{name}: MISMATCH{rows}; {'; '.join(detail)}")
    return lines, ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the store and compare it with the copy.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--data-dir", type=Path, help="default: HORSETRACK_DATA_DIR")
    parser.add_argument("--work-dir", type=Path, help="default: a new temporary dir")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compare-only", action="store_true", help="reuse --work-dir's build")
    args = parser.parse_args(argv)
    if args.compare_only and args.work_dir is None:
        parser.error("--compare-only needs --work-dir")

    from horsetrack.core import paths
    from horsetrack.core.store import schemas

    data = (args.data_dir or paths.data_dir()).expanduser().resolve()
    work = (args.work_dir or Path(tempfile.mkdtemp(prefix="store-rebuild-"))).expanduser().resolve()
    guarded = _guarded(args.source)
    before = fingerprint(data, guarded)
    result = None
    try:
        if not args.compare_only:
            prepare_work_dir(data, work, args.source)
            print(f"work dir {work}")
            for line in rebuild(work, args.source, args.workers):
                print(line)
        tables = {name: spec.sort_by for name, spec in schemas.TABLES.items()}
        copied, rebuilt = data / "processed" / args.source, work / "processed" / args.source
        result = compare_stores(copied, rebuilt, tables)
    except Refused as exc:
        print(f"ERROR {exc}")
    finally:  # also when the build raised
        changed = fingerprint(data, guarded) != before
        if changed:
            print(f"ERROR the data dir changed under {', '.join(guarded)}")
    if changed or result is None:
        return 2
    lines, ok = result
    for line in lines:
        print(line)
    print(f"data dir unchanged under {', '.join(guarded)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
