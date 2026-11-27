"""A scoped reader over the copied store for the v0 models (#114).

`read_store_table` reads `processed_dir(source)/<name>.parquet` through pyarrow.dataset (`source`
defaults to `SOURCE`, 'jra'; `source='jravan'` reads the JRA-VAN copy): only the named columns,
with the inclusive date range and the holdout exclusion pushed into the scan as one dataset
expression (`core.eval.holdout.development_filter`, #415: #71's `is_holdout_day`, looked up at
call time). No holdout row is materialized; Parquet prunes row groups only by their min/max
statistics, so a row group mixing development and holdout days is still read from disk and its
holdout rows are dropped by the scan's filter. A read that touches prices may not start before
D10's price era (2014-06-07), whichever the source.

When #46 lands, this may delegate to its `read_table`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from horsetrack.core.eval.holdout import development_filter
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

    dataset = ds.dataset(processed_dir(source) / f"{name}.parquet", format="parquet")
    available = set(dataset.schema.names)
    missing = [c for c in columns if c not in available]
    if missing:
        raise KeyError(f"{name} has no column(s): {', '.join(missing)}")
    if DATE_COLUMN not in available:
        raise ValueError(f"{name} has no {DATE_COLUMN} column to scope the read by")

    keep = development_filter(COUNTRY, start, end, column=DATE_COLUMN)
    if where is not None:
        keep = keep & where
    return dataset.to_table(columns=columns, filter=keep)
