"""`races.result_published_utc` (ticket #225): when the race's official result was first seen
published, null for history without a logged time.

Contract pinned here (the coder follows it, and docs/specs/store.md says it):
- `schemas.TABLES["races"]` declares `result_published_utc` of kind UTC (timestamp[us, UTC]).
- The build fills it from the adapter's optional `result_published_utc()` hook, a mapping
  race_id -> aware datetime (any zone; stored as that instant in UTC), the way
  `season_start_month()` is an optional hook. A race the hook does not name is null; a race_id the
  hook names but no page holds adds no row. An adapter without the hook gets an all-null column.

Synthetic pages and the fake adapter of store_fakes.py; no real page.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from store_fakes import (
    PAGES,
    RACE_2010,
    RACE_2026,
    RACE_ABANDONED,
    RACE_DEAD_HEAT,
    SOURCE,
    FakeStoreAdapter,
    populate,
    register,
)

from horsetrack.core import paths
from horsetrack.core.store import build, schemas, tables

UTC = timezone.utc
HK = timezone(timedelta(hours=8))
ALL_RACES = {RACE_2010, RACE_ABANDONED, RACE_DEAD_HEAT, RACE_2026}
COLUMN = "result_published_utc"


@pytest.fixture
def data(tmp_path, monkeypatch) -> Path:
    path = (tmp_path / "data").resolve()
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(path))
    return path


@pytest.fixture
def adapter(monkeypatch) -> FakeStoreAdapter:
    return register(monkeypatch, FakeStoreAdapter())


def built_races() -> dict[str, datetime | None]:
    populate(PAGES)
    build.build_tables(SOURCE, workers=1)
    races = tables.read_table(SOURCE, "races", ["race_id", COLUMN])
    return {
        rid: (None if pd.isna(t) else t.to_pydatetime())
        for rid, t in zip(races["race_id"], races[COLUMN], strict=True)
    }


def test_races_declare_a_utc_publication_column():
    assert schemas.TABLES["races"].columns.get(COLUMN) == schemas.UTC


@pytest.mark.parametrize(
    "logged",
    [
        {},
        {RACE_2026: datetime(2026, 9, 16, 12, 5, 30, tzinfo=UTC)},
        {
            RACE_2010: datetime(2010, 1, 1, 22, 41, tzinfo=HK),  # 14:41 UTC
            RACE_DEAD_HEAT: datetime(2020, 12, 9, 13, 2, 15, tzinfo=UTC),
            f"{SOURCE}:2030-01-01:ST:01": datetime(2030, 1, 1, 6, 0, tzinfo=UTC),  # no such race
        },
    ],
    ids=["none-logged", "one", "zones-and-an-unknown-race"],
)
def test_the_build_fills_publication_times_from_the_adapter_hook(data, adapter, logged):
    adapter.result_published_utc = lambda: dict(logged)
    got = built_races()
    assert set(got) == ALL_RACES, "a logged time added or dropped a race"
    for rid in ALL_RACES:
        want = logged.get(rid)
        assert got[rid] == want, rid
        if want is not None:
            assert got[rid].utcoffset() == timedelta(0), rid
    schema = pq.read_schema(paths.processed_dir(SOURCE) / "races.parquet")
    assert schema.field(COLUMN).type == pa.timestamp("us", tz="UTC")


def test_an_adapter_without_the_hook_gets_an_all_null_column(data, adapter):
    assert not hasattr(adapter, "result_published_utc")
    got = built_races()
    assert set(got) == ALL_RACES
    assert all(t is None for t in got.values())
