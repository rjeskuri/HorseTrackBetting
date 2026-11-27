"""Read access to the store's tables (docs/specs/store.md): pandas frames or a DuckDB connection.

A source's table lives at `paths.processed_dir(source) / f"{name}.parquet"`, resolved on every call.
dtypes out of `read_table`: ints are pandas `Int64`, bools `boolean`, floats `float64`, strings
`object` (None for missing), dates `object` holding `datetime.date`, `utc` columns tz-aware UTC
timestamps, list columns `object` holding numpy arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from horsetrack.core.paths import processed_dir
from horsetrack.core.store import schemas

ARROW_TYPES: dict[str, pa.DataType] = {
    schemas.INT: pa.int64(),
    schemas.FLOAT: pa.float64(),
    schemas.STR: pa.string(),
    schemas.BOOL: pa.bool_(),
    schemas.DATE: pa.date32(),
    schemas.UTC: pa.timestamp("us", tz="UTC"),
    schemas.INT_LIST: pa.list_(pa.int32()),
    schemas.FLOAT_LIST: pa.list_(pa.float64()),
}
_PANDAS_TYPES = {pa.int64(): pd.Int64Dtype(), pa.bool_(): pd.BooleanDtype()}


def _spec(name: str) -> schemas.TableSpec:
    spec = schemas.TABLES.get(name)
    if spec is None:
        raise ValueError(f"not a store table: {name!r} (tables: {', '.join(schemas.TABLE_NAMES)})")
    return spec


def table_path(source: str, name: str) -> Path:
    """The Parquet file of a declared table for a source (it need not exist)."""
    return processed_dir(source) / f"{_spec(name).name}.parquet"


def read_table(source: str, name: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """A source's table as a frame: the requested (default: all declared) columns, in that order,
    cast to their declared types."""
    spec = _spec(name)
    wanted = list(spec.columns) if columns is None else list(columns)
    unknown = [c for c in wanted if c not in spec.columns]
    if unknown:
        raise ValueError(f"table {name!r} declares no column {unknown}")
    path = table_path(source, name)
    if not path.exists():
        raise FileNotFoundError(f"table {name!r} of source {source!r}: {path} does not exist")
    missing = [c for c in wanted if c not in pq.read_schema(path).names]
    if missing:
        raise ValueError(f"table {name!r} of source {source!r}: {path} has no column {missing}")
    return to_frame(name, pq.read_table(path, columns=wanted).select(wanted))


def to_frame(name: str, data: pa.Table) -> pd.DataFrame:
    """An Arrow table of the declared table `name`'s columns as a frame with `read_table`'s
    dtypes (the module docstring)."""
    spec = _spec(name)
    wanted = list(data.column_names)
    target = pa.schema([(c, ARROW_TYPES[spec.columns[c]]) for c in wanted])
    frame = data.cast(target).to_pandas(types_mapper=_PANDAS_TYPES.get)
    for column in wanted:
        if spec.columns[column] == schemas.STR:  # pandas 3 reads strings as `str`, NaN for missing
            values = frame[column].astype(object)
            frame[column] = values.where(values.notna(), None)
    return frame


def row_counts(source: str) -> dict[str, int]:
    """Rows per declared table whose file exists, from the Parquet footers (no data is read)."""
    return {
        name: pq.read_metadata(path).num_rows
        for name in schemas.TABLE_NAMES
        if (path := table_path(source, name)).exists()
    }


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def connect(source: str) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with one view per declared table whose file exists for the source."""
    conn = duckdb.connect(":memory:")
    for name in schemas.TABLE_NAMES:
        path = table_path(source, name)
        if path.exists():
            conn.execute(
                f'CREATE VIEW "{name}" AS SELECT * FROM read_parquet({_sql_string(str(path))})'
            )
    return conn
