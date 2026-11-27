"""A scoped reader over the copied store for the v0 models (#114).

`read_store_table` reads `processed_dir(source)/<name>.parquet` through one open handle (`source`
defaults to `SOURCE`, 'jra'; `source='jravan'` reads the JRA-VAN copy): only the named columns,
with the inclusive date range, clamped to the file's own first and last race_date, and the
holdout exclusion pushed into the scan as one dataset expression
(`core.eval.holdout.file_development_filter`, #415: #71's `is_holdout_day`, looked up at call
time, so an open-ended `end=date.max` asks it only about the file's days). The scan is a
`ds.dataset` of the file's path served from that handle (`_Snapshot`, a one-file read-only
filesystem), so the bounds and the rows come from one snapshot and the read is a dataset scan
with the predicate as its filter. No holdout row is materialized;
Parquet prunes row groups only by their min/max statistics, so a row group mixing development and
holdout days is still read from disk and its holdout rows are dropped by the scan's filter. A
read that touches prices may not start before D10's price era (2014-06-07), whichever the source:
the requested start is checked, before any clamping.

When #46 lands, this may delegate to its `read_table`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs

from horsetrack.core.eval.holdout import file_development_filter
from horsetrack.core.paths import processed_dir

PRICE_ERA_START = date(2014, 6, 7)
COUNTRY = "japan"
SOURCE = "jra"
DATE_COLUMN = "race_date"


class PreReformPriceError(ValueError):
    """A price read or fit reaches before D10's price era (2014-06-07)."""


def _as_date(value: object) -> date:
    return pd.Timestamp(value).date()


def read_store_table(
    name: str,
    *,
    columns: Sequence[str],
    start: date,
    end: date,
    touches_price: bool,
    where: ds.Expression | None = None,
    source: str = SOURCE,
) -> pa.Table:
    """Rows of the store table `name` with `start <= race_date <= end`, holding exactly `columns`
    in the order asked; holdout days are dropped. `where`, a dataset expression (it may name
    columns not read), narrows the rows further inside the scan. `source` names the store copy under
    `processed_dir` (default `SOURCE`, 'jra')."""
    if columns is None or len(columns) == 0:
        raise ValueError("columns are required: name the columns to read (no whole-table read)")
    columns = list(columns)
    start, end = _as_date(start), _as_date(end)
    if touches_price and start < PRICE_ERA_START:
        raise PreReformPriceError(
            f"a price read may not start before {PRICE_ERA_START.isoformat()} (D10), "
            f"got start {start.isoformat()}"
        )

    path = str(processed_dir(source) / f"{name}.parquet")
    with pa.memory_map(path) as handle:  # the file's bounds and its scan: one snapshot
        snapshot = pafs.PyFileSystem(_Snapshot(path, handle))
        dataset = ds.dataset(path, filesystem=snapshot, format="parquet")
        available = set(dataset.schema.names)
        missing = [c for c in columns if c not in available]
        if missing:
            raise KeyError(f"{name} has no column(s): {', '.join(missing)}")
        if DATE_COLUMN not in available:
            raise ValueError(f"{name} has no {DATE_COLUMN} column to scope the read by")
        keep = file_development_filter(
            handle, COUNTRY, start=start, end=end, where=where, column=DATE_COLUMN
        )
        return dataset.to_table(columns=columns, filter=keep)


class _Snapshot(pafs.FileSystemHandler):
    """A read-only filesystem holding the one file `path`, served zero-copy from the memory map
    `handle`: a dataset of `path` on it reads the snapshot the handle opened, even after the file
    is atomically replaced on disk."""

    def __init__(self, path: str, handle: pa.MemoryMappedFile) -> None:
        self.path = path
        handle.seek(0)
        self.buffer = handle.read_buffer()  # the whole map, not a copy

    def get_type_name(self) -> str:
        return "snapshot"

    def normalize_path(self, path: str) -> str:
        return path

    def get_file_info(self, paths: list[str]) -> list[pafs.FileInfo]:
        return [
            pafs.FileInfo(p, pafs.FileType.File, size=self.buffer.size)
            if p == self.path
            else pafs.FileInfo(p, pafs.FileType.NotFound)
            for p in paths
        ]

    def get_file_info_selector(self, selector: pafs.FileSelector) -> list[pafs.FileInfo]:
        raise NotImplementedError("a snapshot holds one file and lists no directory")

    def open_input_file(self, path: str) -> pa.NativeFile:
        if path != self.path:
            raise FileNotFoundError(path)
        return pa.BufferReader(self.buffer)

    open_input_stream = open_input_file

    def _read_only(self, *args: object, **kwargs: object) -> None:
        raise PermissionError("a store snapshot is read-only")

    create_dir = delete_dir = delete_dir_contents = delete_root_dir_contents = _read_only
    delete_file = move = copy_file = open_output_stream = open_append_stream = _read_only
