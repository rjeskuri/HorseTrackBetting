"""The store build (#48): manifest + raw page cache -> the store's Parquet tables, byte-stable.

Ported from HKTrack tests/unit/store/test_build.py and the build cases of test_adversarial.py, on
the synthetic pages and fake adapter of store_fakes.py (no real page). Seams pinned, S = source:

* `horsetrack.core.store.build.build_tables(S, *, page_types=None, workers=None, limit=None)`
  returns a `BuildSummary` with HKTrack's fields: snapshot_id, seconds, n_selected, n_included,
  n_parsed, n_from_cache, n_not_parsed, n_empty, failed and unreadable ({page_type, key, error}),
  key_mismatch, excluded_with_mismatch, duplicate_pages ("page_type/key"), skipped_page_types,
  extra_columns, parser_versions, rows, enrichment, date_min, date_max, build_path, report_path.
* It writes `processed_dir(S)/<table>.parquet` (declared columns in declared order, the store's
  Arrow types = `build.arrow_schema(table)`), `processed_dir(S)/_build.json` (`build.BUILD_FILE`:
  the summary's fields plus `built_utc`; `built_utc` and `seconds` are its only wall-clock values)
  and a report at `reports_dir() / f"store-build-{S}.md"`.
* Page types feed tables as in HKTrack: fixture -> meetings; results -> meetings, races, runners,
  dividends, incidents; sectional -> sectionals; corunning -> running_comments; horse -> horses,
  horse_runs; pedigree_card -> horse_pedigree. A page's rows for any other table are ignored; a page
  type with no table, or whose adapter has no parser version, is skipped and counted.
* The nine tables fed by fixture..horse are written on every build; horse_pedigree only when the
  manifest holds pedigree_card pages; jrdb_pre_race / jrdb_post_race (built elsewhere) never.
* Seasons are "YYYY/YY" labels of race_date, starting in the month the adapter's optional
  `season_start_month()` hook returns; an adapter without the hook gets calendar-year seasons.
* `build.read_manifest(S)`: the fetched / abandoned manifest rows that carry a sha256, as dicts, in
  (page_type, key) order; the manifest is only ever read. `build.default_workers()`:
  min(6, cpus - 2), at least 1. With workers >= 2 and at least 64 stale pages (HKTrack's threshold)
  the pages are
  parsed in spawned worker processes running `parse_worker.parse_job` (looked up when the build
  runs, as HKTrack's tests patch it); a dying worker is a RuntimeError, never a hang.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import threading
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import store_fakes
from store_fakes import (
    FETCHED_CELL,
    GOOD_PAGES,
    HORSE_RECENT,
    HORSE_RETIRED,
    KEY_2010,
    KEY_2026,
    MISMATCHED_KEY,
    PAGES,
    RACE_2010,
    RACE_2026,
    RACE_ABANDONED,
    RACE_DEAD_HEAT,
    RESULTS_2026,
    SECTIONAL_2026,
    SOURCE,
    TEST_PID_VAR,
    VERSIONS,
    FakeStoreAdapter,
    Page,
    add,
    add_page,
    die_in_worker,
    dividend,
    horse,
    horse_run,
    meeting,
    page,
    parse_job_in_fresh_worker,
    populate,
    register,
    results_page,
    runner,
    sectional,
)

from horsetrack.core import paths
from horsetrack.core.store import parse_worker, schemas, tables
from horsetrack.core.store.cache import ParseCache, cache_path

# isort: split
# A block of its own: while the module is missing, it is the import error a red run reports.
from horsetrack.core.store import build

VAR = "HORSETRACK_DATA_DIR"
NINE = (
    "meetings", "races", "runners", "dividends", "sectionals", "incidents", "running_comments",
    "horses", "horse_runs",
)  # fmt: skip
# page type -> the tables its rows feed (HKTrack's schemas.TableSpec.sources, read the other way)
FEEDS = {
    "fixture": ("meetings",),
    "results": ("meetings", "races", "runners", "dividends", "incidents"),
    "sectional": ("sectionals",),
    "corunning": ("running_comments",),
    "horse": ("horses", "horse_runs"),
    "pedigree_card": ("horse_pedigree",),
}
# kind -> the Arrow type the store writes (docs/specs/store.md), independent of the code under test
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
# what makes a row the same row from one build to the next
IDENTITY = {
    "meetings": ("meeting_id",),
    "races": ("race_id",),
    "runners": ("race_id", "horse_no"),
    "dividends": ("race_id", "pool", "combination"),
    "sectionals": ("race_id", "horse_no", "section_no"),
    "incidents": ("race_id", "horse_no", "text"),
    "running_comments": ("race_id", "horse_no"),
    "horses": ("horse_id",),
    "horse_runs": ("horse_id", "race_date", "race_index"),
}
WITHDRAWN = ("WV", "WX", "WR")  # placings of a horse_runs row that is a withdrawal, not a run
CLOCK_KEYS = ("built_utc", "seconds")
PATH_KEYS = ("build_path", "report_path")
COUNT_KEYS = (
    "n_selected", "n_included", "n_parsed", "n_from_cache", "n_not_parsed", "n_empty", "failed",
    "unreadable", "key_mismatch", "excluded_with_mismatch", "duplicate_pages", "rows",
)  # fmt: skip
RACE_ID = re.compile(r"^(([^:]+):(\d{4}-\d{2}-\d{2}):([^:]+)):(\d+)$")


# ---------------------------------------------------------------- fixtures and helpers


def use_data_dir(monkeypatch, path: Path) -> Path:
    path = path.resolve()
    monkeypatch.setenv(VAR, str(path))
    return path


@pytest.fixture
def data(tmp_path, monkeypatch) -> Path:
    return use_data_dir(monkeypatch, tmp_path / "data")


@pytest.fixture
def adapter(monkeypatch) -> FakeStoreAdapter:
    return register(monkeypatch, FakeStoreAdapter())


@pytest.fixture
def raw_root(data, adapter) -> Path:
    """Raw cache + manifest of PAGES only: nothing parsed, nothing built."""
    populate(PAGES)
    return data


@pytest.fixture
def built_root(raw_root, adapter) -> Path:
    """PAGES built once (parse cache, tables, _build.json, report); the adapter's calls cleared."""
    build.build_tables(SOURCE, workers=1)
    adapter.calls.clear()
    return raw_root


def build_now(**kwargs):
    return build.build_tables(SOURCE, workers=1, **kwargs)


def read(name: str) -> pd.DataFrame:
    return tables.read_table(SOURCE, name)


def written() -> set[str]:
    return {path.stem for path in paths.processed_dir(SOURCE).glob("*.parquet")}


def table_bytes() -> dict[str, str]:
    return {
        path.stem: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths.processed_dir(SOURCE).glob("*.parquet"))
    }


def record() -> dict[str, Any]:
    return json.loads((paths.processed_dir(SOURCE) / "_build.json").read_text(encoding="utf-8"))


def without(mapping: dict[str, Any], keys) -> str:
    """The mapping minus `keys`, as JSON text in its own key order."""
    return json.dumps({k: v for k, v in mapping.items() if k not in keys})


def report_path() -> Path:
    return paths.reports_dir() / f"store-build-{SOURCE}.md"


def expected_schema(table: str) -> pa.Schema:
    return pa.schema([(c, ARROW[kind]) for c, kind in schemas.TABLES[table].columns.items()])


def season(day: date, start_month: int) -> str:
    start = day.year if day.month >= start_month else day.year - 1
    return f"{start}/{(start + 1) % 100:02d}"


def race_keys(race_id: str) -> tuple[str, date, str, int]:
    """(meeting_id, race_date, venue, race_no) as the race id spells them."""
    match = RACE_ID.match(race_id)
    assert match, race_id
    return match[1], date.fromisoformat(match[3]), match[4], int(match[5])


def cell(value: Any) -> Any:
    """A table cell as plain Python: None for missing, lists for arrays, ISO text for dates."""
    if isinstance(value, np.ndarray):
        return [cell(v) for v in value.tolist()]
    if isinstance(value, list | tuple):
        return [cell(v) for v in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    return value


def frame_rows(frame: pd.DataFrame, columns: list[str]) -> list[tuple]:
    return [tuple(cell(v) for v in row) for row in frame[columns].itertuples(index=False)]


def fixture_rows(rows: list[dict], columns: list[str]) -> list[tuple]:
    return [tuple(cell(row.get(c)) for c in columns) for row in rows]


def expected_rows(table: str, pages: list[Page] = GOOD_PAGES) -> list[dict]:
    """The rows of one table as the pages hold them, pages in (page_type, key) order."""
    out: list[dict] = []
    for item in sorted(pages, key=lambda p: (p.page_type, p.key)):
        if table in FEEDS.get(item.page_type, ()):
            out += [dict(row) for row in item.tables.get(table, [])]
    return out


def in_store_order(table: str, rows: list[dict]) -> list[dict]:
    """Stable sort by the table's sort key, missing values last."""
    sort_by = schemas.TABLES[table].sort_by

    def key(row):
        return tuple(
            (row.get(c) is None, row.get(c) if row.get(c) is not None else 0) for c in sort_by
        )

    return sorted(rows, key=key)


def expected_counts(pages: list[Page]) -> dict[str, int]:
    """Rows per written table when exactly `pages` are in the tables."""
    counts = dict.fromkeys(NINE, 0)
    meeting_ids: set[str] = set()
    pedigree_ids: set[str] | None = None
    for item in pages:
        for table in FEEDS.get(item.page_type, ()):
            rows = item.tables.get(table, [])
            if table == "meetings":
                meeting_ids |= {row["meeting_id"] for row in rows}
            elif table == "horse_pedigree":
                pedigree_ids = (pedigree_ids or set()) | {row["horse_id"] for row in rows}
            else:
                counts[table] += len(rows)
        if item.page_type == "results":  # a meeting only the results know about is added
            meeting_ids |= {race_keys(r["race_id"])[0] for r in item.tables.get("races", [])}
    counts["meetings"] = len(meeting_ids)
    if pedigree_ids is not None:
        counts["horse_pedigree"] = len(pedigree_ids)
    return counts


def page_rows(item: Page) -> int:
    """Rows a page holds for the tables its page type feeds."""
    return sum(len(item.tables.get(table, [])) for table in FEEDS.get(item.page_type, ()))


def run_dates(pages: list[Page]) -> tuple[str | None, str | None]:
    days = [
        race_keys(r["race_id"])[1].isoformat()
        for item in pages
        if item.page_type == "results"
        for r in item.tables.get("races", [])
        if r["status"] == "run"
    ]
    return (min(days), max(days)) if days else (None, None)


def identities(table: str) -> set[tuple]:
    frame = read(table)
    return set(frame_rows(frame, list(IDENTITY[table])))


def withdrawn(placing: str | None) -> bool:
    return str(placing or "").strip().upper().startswith(WITHDRAWN)


def worker_rows() -> list[dict]:
    """The `_worker` rows the fake adapter put into every cached payload."""
    out = []
    with ParseCache(cache_path(SOURCE)) as cache:
        for page_type in sorted(FEEDS):
            for _key, _sha, _version, payload in cache.payloads(page_type):
                part = parse_worker.decode_tables(payload)["_worker"]
                out += [dict(zip(part["columns"], row, strict=True)) for row in part["rows"]]
    return out


RACE_NEW_DAY = f"{SOURCE}:2026-09-23:HV:01"
NEW_DAY = [  # the race day after the cache was first built
    results_page(
        "2026-09-23_HV_R01",
        2026,
        {
            "races": [
                {"race_id": RACE_NEW_DAY, "race_index": 35, "race_name": "SYNTH NEW DAY",
                 "distance_m": 1650, "field_size": 2, "status": "run"}
            ],
            "runners": [
                runner(RACE_NEW_DAY, 1, HORSE_RECENT, 1, win_odds=4.1),
                runner(RACE_NEW_DAY, 2, "H_2023_J308", 2, win_odds=1.9),
            ],
            "dividends": [dividend(RACE_NEW_DAY, "WIN", "1", 41.0)],
            "incidents": [],
        },
    ),
    Page(
        "sectional",
        "2026-09-23_HV_R01",
        2026,
        {"sectionals": [sectional(RACE_NEW_DAY, 1, HORSE_RECENT, 1, 1, 0.0, 24.0, [12.0, 12.0])]},
    ),
]  # fmt: skip
# enough extra pages that the pool is used (HKTrack sends 64 stale pages and more to the pool)
FILLER = [
    Page(
        "horse",
        f"H_2030_F{i:03d}",
        2030,
        {"horses": [horse(f"H_2030_F{i:03d}", f"FILLER {i}", False)], "horse_runs": []},
    )
    for i in range(66)
]


def pedigree(horse_id, sire, dam, dam_sire, race_id, *, sire_id=None) -> dict:
    def norm(name):
        return None if name is None else name.casefold()

    return {
        "horse_id": horse_id, "sire_name": sire, "sire_name_norm": norm(sire), "sire_id": sire_id,
        "dam_name": dam, "dam_name_norm": norm(dam), "dam_id": None, "dam_sire_name": dam_sire,
        "dam_sire_name_norm": norm(dam_sire), "dam_sire_id": None, "source_race_id": race_id,
        "scraped_utc": FETCHED_CELL,
    }  # fmt: skip


CARD_X_RACE = f"{SOURCE}:2025-11-23:ST:11"
CARD_Y_RACE = f"{SOURCE}:2026-02-01:ST:05"
CARD_X = {  # P001 with its dam sire, without a sire id
    "horse_pedigree": [
        pedigree("P001", "Sire One", "Dam One", "Damsire One", CARD_X_RACE),
        pedigree("P002", "Sire Two", "Dam Two", "Damsire Two", CARD_X_RACE),
    ]
}
CARD_Y = {  # P001 without its dam sire, with a sire id
    "horse_pedigree": [
        pedigree("P001", "Sire One", "Dam One", None, CARD_Y_RACE, sire_id="S0001"),
        pedigree("P003", "Sire Three", "Dam Three", "Damsire Three", CARD_Y_RACE),
    ]
}
EARLY, LATE = "2026-11-18T14:20:00+00:00", "2026-11-18T15:00:00+00:00"


def cards(x_is_newer: bool) -> list[Page]:
    """The two cards; the newer one has the later key and the later fetch time."""
    old_key, new_key = "2025-11-23_ST_R11", "2026-02-01_ST_R05"
    x_key, y_key = (new_key, old_key) if x_is_newer else (old_key, new_key)
    return [
        Page(
            "pedigree_card",
            x_key,
            int(x_key[:4]),
            CARD_X,
            fetched_utc=LATE if x_is_newer else EARLY,
        ),
        Page(
            "pedigree_card",
            y_key,
            int(y_key[:4]),
            CARD_Y,
            fetched_utc=EARLY if x_is_newer else LATE,
        ),
    ]


# ---------------------------------------------------------------- definition of done


def test_build_is_byte_stable(tmp_path, monkeypatch, adapter):
    """Two builds of the same inputs give identical bytes: from the parse cache, from scratch, and
    in another data dir whose manifest and raw files were written in the reverse order (other gzip
    headers). `_build.json` differs only in its wall-clock values (and its paths in another dir)."""
    use_data_dir(monkeypatch, tmp_path / "a" / "data")
    populate(PAGES)
    build_now()
    first_tables, first_record = table_bytes(), record()
    assert set(first_tables) == set(NINE)

    again = build_now()  # everything served from the parse cache
    assert table_bytes() == first_tables
    assert again.snapshot_id == first_record["snapshot_id"] and again.rows == first_record["rows"]

    shutil.rmtree(paths.interim_dir(SOURCE))
    shutil.rmtree(paths.processed_dir(SOURCE))
    build_now()  # from scratch
    assert table_bytes() == first_tables
    assert without(record(), CLOCK_KEYS) == without(first_record, CLOCK_KEYS)

    use_data_dir(monkeypatch, tmp_path / "b" / "data")
    populate(list(reversed(PAGES)), mtime=1_700_000_000)
    build_now()
    assert table_bytes() == first_tables
    other = CLOCK_KEYS + PATH_KEYS
    assert without(record(), other) == without(first_record, other)

    # the bytes follow the inputs: a refetched results page with one new price rewrites runners only
    refetched = copy.deepcopy(RESULTS_2026)
    refetched["runners"][1]["win_odds"] = 3.4
    add(results_page(KEY_2026, 2026, refetched))
    build_now()
    after = table_bytes()
    assert {name for name, digest in first_tables.items() if after[name] != digest} == {"runners"}


# each run: what it is asked to parse, and the pages that reached the cache before it
PARTIAL_RUNS = [
    ({"page_types": ["results"]}, []),
    ({"page_types": ["sectional", "horse"], "limit": 1}, []),
    ({"limit": 1}, []),
    ({"page_types": []}, NEW_DAY),
    ({"page_types": ["corunning", "fixture"]}, []),
    ({"limit": 1}, []),
    ({}, []),
]


def test_a_partial_run_never_drops_rows(raw_root, adapter):
    """`page_types` and `limit` restrict what a run parses, never what it assembles: every row an
    earlier run put in the tables is still there after any partial run, new race days included."""
    in_manifest = list(PAGES)
    good = {(p.page_type, p.key) for p in GOOD_PAGES + NEW_DAY}
    parsed: set[tuple[str, str]] = set()
    seen: dict[str, set[tuple]] = {name: set() for name in NINE}
    for kwargs, added in PARTIAL_RUNS:
        populate(added)
        in_manifest += added
        adapter.calls.clear()
        summary = build_now(**kwargs)

        # the run parsed only what it was asked to
        if kwargs.get("page_types") is not None:
            assert {pt for pt, _ in adapter.calls} <= set(kwargs["page_types"]), kwargs
        if kwargs.get("limit") is not None:
            assert (
                max(Counter(pt for pt, _ in adapter.calls).values(), default=0) <= kwargs["limit"]
            )
        parsed |= set(adapter.calls)

        # nothing an earlier run put in the tables is gone
        for name in NINE:
            now = identities(name)
            assert seen[name] <= now, (kwargs, name, sorted(seen[name] - now))
            seen[name] = now
        # and the tables hold every parsed page that belongs there
        included = [p for p in in_manifest if (p.page_type, p.key) in parsed & good]
        assert summary.rows == expected_counts(included), kwargs

    assert summary.n_not_parsed == 0 and parsed == {(p.page_type, p.key) for p in in_manifest}
    assert summary.rows == expected_counts(GOOD_PAGES + NEW_DAY)
    meetings = read("meetings").set_index("meeting_id")
    assert (
        meetings.loc[f"{SOURCE}:2026-09-23:HV", "status"] == "run"
    )  # the new day reached the store


def _clean(adapter) -> list[Page]:
    return GOOD_PAGES


def _results_only(adapter) -> list[Page]:
    return [p for p in GOOD_PAGES if p.page_type == "results"]


def _newest_of_each_type(adapter) -> list[Page]:
    newest = {}
    for item in PAGES:
        if item.key > newest.get(item.page_type, ("", ""))[1]:
            newest[item.page_type] = (item, item.key)
    return [item for item, _ in newest.values()]


EMPTY_PAGE = results_page("2026-09-16_HV_R09", 2026, {"races": [], "runners": []})
OLDER_PROFILE = Page(  # a second page for HORSE_RECENT with an older profile: it loses
    "horse",
    "H_2023_J087_OLD",
    2023,
    {
        "horses": [horse(HORSE_RECENT, "SYNTH SPARK OLD", False)],
        "horse_runs": [horse_run(HORSE_RECENT, "2026-01-01", 5, rating=20, placing_raw="2")],
    },
    fetched_utc="2025-01-01T00:00:00+00:00",
)


def _trouble(adapter) -> list[Page]:
    """A raising parser, an unreadable file, a duplicate horse page and a page without rows."""
    adapter.overrides["sectional", KEY_2026] = ValueError("parser broke")
    add_page("results", "2026-09-16_HV_R08", 2026, b"never readable", raw=b"\x1f\x8b not gzip")
    add(OLDER_PROFILE)
    add(EMPTY_PAGE)
    return [p for p in GOOD_PAGES if (p.page_type, p.key) != ("sectional", KEY_2026)] + [EMPTY_PAGE]


def _with_pedigree_cards(adapter) -> list[Page]:
    populate(cards(x_is_newer=True))
    return GOOD_PAGES + cards(x_is_newer=True)


BUILDS = {
    "full": (_clean, {}),
    "results only": (_results_only, {"page_types": ["results"]}),
    "newest page of each type": (_newest_of_each_type, {"limit": 1}),
    "failures, a duplicate and an empty page": (_trouble, {}),
    "with pedigree cards": (_with_pedigree_cards, {}),
}


@pytest.mark.parametrize("scenario", list(BUILDS))
def test_build_json_counts_match_the_written_tables(raw_root, adapter, scenario):
    setup, kwargs = BUILDS[scenario]
    included = setup(adapter)
    summary = build_now(**kwargs)
    stored = record()
    on_disk = {
        path.stem: pq.read_metadata(path).num_rows
        for path in paths.processed_dir(SOURCE).glob("*.parquet")
    }
    assert stored["rows"] == on_disk == expected_counts(included)
    for key in COUNT_KEYS:
        assert stored[key] == json.loads(json.dumps(getattr(summary, key))), key
    # every selected page is accounted for exactly once
    assert stored["n_included"] == len(included)
    assert stored["n_selected"] == (
        stored["n_included"] + len(stored["duplicate_pages"]) + len(stored["failed"])
        + len(stored["unreadable"]) + len(stored["key_mismatch"])
        + len(stored["excluded_with_mismatch"]) + stored["n_not_parsed"]
    )  # fmt: skip
    assert stored["n_empty"] == sum(1 for p in included if not page_rows(p))
    assert (stored["date_min"], stored["date_max"]) == run_dates(included)


def test_build_record_names_versions_snapshot_and_report(built_root):
    stored = record()
    assert build.BUILD_FILE == "_build.json"
    assert stored["parser_versions"] == {
        t: VERSIONS[t] for t in ("corunning", "fixture", "horse", "results", "sectional")
    }
    assert re.fullmatch(r"[0-9a-f]{40}", stored["snapshot_id"])
    built = datetime.fromisoformat(stored["built_utc"])
    assert stored["built_utc"].endswith("+00:00") and built.utcoffset() == timedelta(0)
    assert (stored["failed"], stored["unreadable"], stored["duplicate_pages"]) == ([], [], [])
    assert stored["build_path"] == str(paths.processed_dir(SOURCE) / "_build.json")
    assert stored["report_path"] == str(report_path()) and report_path().is_file()


# ---------------------------------------------------------------- layout and schema


def test_every_table_is_written_with_the_contract_schema(built_root):
    assert written() == set(NINE)  # no horse_pedigree without pedigree cards, never a jrdb table
    for name in NINE:
        stored = pq.read_schema(tables.table_path(SOURCE, name))
        assert stored.names == list(schemas.TABLES[name].columns), name
        assert stored.remove_metadata().equals(expected_schema(name)), name
        assert set(read(name)["jurisdiction"]) <= {SOURCE}, name
    assert not [p for p in paths.processed_dir(SOURCE).iterdir() if "tmp" in p.name]


def test_arrow_schema_is_the_declared_one():
    for name in schemas.TABLE_NAMES:
        assert build.arrow_schema(name).equals(expected_schema(name)), name


@pytest.mark.parametrize("page_type", sorted(FEEDS))
def test_each_page_type_feeds_only_its_tables(data, adapter, page_type):
    race = f"{SOURCE}:2026-01-04:ST:01"
    one_of_each = {
        "meetings": [meeting("2026-01-04", "ST", "day", "turf")],
        "races": [{"race_id": race, "distance_m": 1000, "status": "run"}],
        "runners": [runner(race, 1, "H_2025_Z001", 1, win_odds=2.0)],
        "dividends": [dividend(race, "WIN", "1", 20.0)],
        "sectionals": [sectional(race, 1, "H_2025_Z001", 1, 1, 0.0, 24.0, [])],
        "incidents": [
            {"race_id": race, "horse_id": "H_2025_Z001", "horse_no": 1, "text": "Bumped."}
        ],
        "running_comments": [
            {"race_id": race, "horse_id": "H_2025_Z001", "horse_no": 1, "comment": "Won easily."}
        ],
        "horses": [horse("H_2025_Z001", "SYNTH ZED", False)],
        "horse_runs": [horse_run("H_2025_Z001", "2026-01-04", 1, rating=12, placing_raw="1")],
        "horse_pedigree": [pedigree("H_2025_Z001", "Sire Z", "Dam Z", "Damsire Z", race)],
        "jrdb_pre_race": [{"race_id": race, "horse_no": 1, "horse_id": "H_2025_Z001"}],
        "jrdb_post_race": [{"race_id": race, "horse_no": 1, "horse_id": "H_2025_Z001"}],
    }
    add(Page(page_type, "2026-01-04_ST_R01", 2026, one_of_each))
    summary = build_now()
    expected = {name: int(name in FEEDS[page_type]) for name in NINE}
    if page_type == "pedigree_card":
        expected["horse_pedigree"] = 1
    assert summary.rows == expected and written() == set(expected)
    if "meetings" in FEEDS[page_type]:  # the page's own meeting row, not one made from a race id
        assert read("meetings")["day_night"].tolist() == ["day"]


def test_parser_columns_outside_the_schema_are_reported(raw_root):
    assert build_now().extra_columns == {}
    extra = copy.deepcopy(RESULTS_2026)
    extra["runners"][0]["cap_colour"] = "red"
    extra["races"][0].update(weather="fine", rail="A+3")
    add(results_page(KEY_2026, 2026, extra))
    summary = build_now()
    assert summary.extra_columns == {"races": ["rail", "weather"], "runners": ["cap_colour"]}
    assert record()["extra_columns"] == summary.extra_columns
    assert "cap_colour" not in pq.read_schema(tables.table_path(SOURCE, "runners")).names


# ---------------------------------------------------------------- contents


@pytest.mark.parametrize(
    "start_month, season_2010, season_2026",
    [(9, "2009/10", "2026/27"), (1, "2010/11", "2026/27"), (11, "2009/10", "2025/26"),
     (None, "2010/11", "2026/27")],
    ids=["september", "january", "november", "no-hook"],
)  # fmt: skip
def test_races_carry_the_denormalised_keys(
    raw_root, adapter, start_month, season_2010, season_2026
):
    adapter.set_start_month(start_month)
    build_now()
    month = 1 if start_month is None else start_month
    races = read("races").set_index("race_id")
    assert list(races.index) == sorted([RACE_2010, RACE_ABANDONED, RACE_DEAD_HEAT, RACE_2026])
    for race_id, row in races.iterrows():
        meeting_id, day, venue, race_no = race_keys(race_id)
        got = (row["meeting_id"], row["race_date"], row["venue"], row["race_no"], row["season"])
        assert got == (meeting_id, day, venue, race_no, season(day, month)), race_id
    assert (races.loc[RACE_2010, "season"], races.loc[RACE_2026, "season"]) == (
        season_2010,
        season_2026,
    )
    assert dict(races["status"]) == {
        RACE_2010: "run",
        RACE_ABANDONED: "abandoned",
        RACE_DEAD_HEAT: "run",
        RACE_2026: "run",
    }
    (source_row,) = RESULTS_2026["races"]
    columns = [c for c in source_row if c != "race_id"]
    assert frame_rows(races.loc[[RACE_2026]], columns) == fixture_rows([source_row], columns)
    for name in ("runners", "meetings"):
        frame = read(name)
        expected = [season(date.fromisoformat(cell(d)), month) for d in frame["race_date"]]
        assert frame["season"].tolist() == expected, name


def test_runner_rows_match_the_parsers_one_to_one(built_root):
    runners = read("runners")
    expected = in_store_order("runners", expected_rows("runners"))
    assert len(runners) == len(expected) == 14
    columns = list(expected[0])
    assert frame_rows(runners, columns) == fixture_rows(expected, columns)
    for row in runners.itertuples(index=False):  # keys denormalised from the race id
        meeting_id, day, venue, race_no = race_keys(row.race_id)
        assert (row.meeting_id, row.race_date, row.venue, row.race_no, row.season) == (
            meeting_id,
            day,
            venue,
            race_no,
            season(day, 9),
        )


def test_dead_heat_winners_are_both_kept(built_root):
    runners = read("runners")
    winners = runners[(runners["race_id"] == RACE_DEAD_HEAT) & (runners["finish_pos"] == 1)]
    assert sorted(winners["horse_no"]) == [3, 5] and winners["dead_heat"].all()


@pytest.mark.parametrize(
    "table", ["dividends", "sectionals", "incidents", "running_comments", "horses", "horse_runs"]
)
def test_other_tables_hold_exactly_the_parsed_rows(built_root, table):
    frame = read(table)
    expected = in_store_order(table, expected_rows(table))
    assert len(expected) > 0
    columns = [c for c in expected[0] if c != "profile_scraped_utc"]
    assert frame_rows(frame, columns) == fixture_rows(expected, columns)
    assert frame["jurisdiction"].eq(SOURCE).all()


def test_dividends_keep_page_order_within_a_race(built_root):
    dividends = read("dividends")
    got = dividends.loc[dividends["race_id"] == RACE_2026, ["pool", "combination", "dividend"]]
    assert [tuple(r) for r in got.itertuples(index=False)] == [
        (d["pool"], d["combination"], d["dividend"]) for d in RESULTS_2026["dividends"]
    ]


def test_meetings_start_from_the_fixture_and_are_upgraded_by_results(built_root):
    meetings = read("meetings")
    assert meetings["meeting_id"].is_unique
    got = {
        row.meeting_id.split(":", 1)[1]: (row.status, row.day_night, row.surface_mix)
        for row in meetings.itertuples(index=False)
    }
    assert got == {
        "2010-01-01:ST": ("run", "day", "turf"),
        "2010-01-06:HV": ("scheduled", "night", "turf"),  # no results page
        "2019-11-13:HV": ("abandoned", "night", "turf"),  # all its result pages abandoned
        "2019-11-17:ST": ("scheduled", "day", "turf+awt"),
        "2020-12-09:HV": ("run", None, None),  # only the results know it: its month is not cached
        "2026-09-13:ST": ("scheduled", "day", "turf"),  # its only results page was a mismatch
        "2026-09-16:HV": ("run", "night", "turf"),
        "2026-09-23:HV": ("scheduled", "day", "turf"),  # the newer calendar page wins
        "2026-10-04:ST": ("scheduled", "day", "turf+awt"),
    }
    order = [(cell(d), v) for d, v in zip(meetings["race_date"], meetings["venue"], strict=True)]
    assert order == sorted(order)
    for row in meetings.itertuples(index=False):
        _, day, venue = row.meeting_id.split(":")
        assert (row.race_date, row.venue, row.season) == (
            date.fromisoformat(day),
            venue,
            season(date.fromisoformat(day), 9),
        )


def test_horses_are_static_attributes_with_the_manifest_fetch_time(built_root):
    horses = read("horses").set_index("horse_id")
    assert list(horses.index) == [HORSE_RETIRED, HORSE_RECENT]
    assert horses.loc[HORSE_RECENT, "name"] == "SYNTH SPARK"
    assert not horses.loc[HORSE_RECENT, "is_retired"] and horses.loc[HORSE_RETIRED, "is_retired"]
    # the manifest's fetched_utc in UTC, not the raw file's mtime: reproducible across machines
    assert horses.loc[HORSE_RECENT, "profile_scraped_utc"] == datetime(
        2026, 9, 21, 4, 7, 12, tzinfo=timezone.utc
    )
    assert horses.loc[HORSE_RETIRED, "profile_scraped_utc"] == datetime(
        2026, 8, 2, 23, 15, tzinfo=timezone.utc
    )


@pytest.mark.parametrize("x_is_newer", [True, False], ids=["card-x-newer", "card-y-newer"])
def test_pedigree_cards_give_one_row_per_horse_first_non_null_newest_first(
    raw_root, adapter, x_is_newer
):
    jrdb = paths.processed_dir(SOURCE) / "jrdb_pre_race.parquet"
    jrdb.parent.mkdir(parents=True, exist_ok=True)
    jrdb.write_bytes(b"written by the JRDB matcher, not by the store build")
    populate(cards(x_is_newer))
    summary = build_now()
    assert written() == {*NINE, "horse_pedigree", "jrdb_pre_race"}
    assert jrdb.read_bytes() == b"written by the JRDB matcher, not by the store build"
    assert set(summary.rows) == {*NINE, "horse_pedigree"}
    pedigree_rows = read("horse_pedigree").set_index("horse_id")
    assert list(pedigree_rows.index) == ["P001", "P002", "P003"]
    p1 = pedigree_rows.loc["P001"]
    # a card that misses a value never blanks out one that another card printed
    assert [p1["sire_name"], p1["dam_name"], p1["dam_sire_name"], p1["sire_id"]] == [
        "Sire One",
        "Dam One",
        "Damsire One",
        "S0001",
    ]
    assert p1["dam_sire_name_norm"] == "damsire one"
    # where both cards print a value, the newer card's wins
    assert p1["source_race_id"] == (CARD_X_RACE if x_is_newer else CARD_Y_RACE)
    assert p1["scraped_utc"] == datetime.fromisoformat(LATE)
    assert pedigree_rows.loc["P002", "dam_sire_name"] == "Damsire Two"
    assert pedigree_rows.loc["P003", "sire_name"] == "Sire Three"
    assert pedigree_rows["jurisdiction"].eq(SOURCE).all()


# ---------------------------------------------------------------- enrichment


def test_runners_get_rating_and_gear_at_that_run_from_horse_runs(built_root):
    runners = read("runners").set_index(["race_id", "horse_id"])
    assert runners.loc[(RACE_2026, HORSE_RECENT), ["rating", "gear_raw"]].tolist() == [31, "H/TT"]
    assert runners.loc[(RACE_2010, HORSE_RETIRED), ["rating", "gear_raw"]].tolist() == [54, "B"]
    assert runners["rating"].notna().sum() == 2 and runners["gear_raw"].notna().sum() == 2
    # the join is on (horse_id, race_date) only: the form table prints class and venue otherwise
    races = read("races").set_index("race_id")
    assert races.loc[RACE_2026, "race_class"] == "Class 5"
    assert races.loc[RACE_2010, "venue"] == "ST"
    runs = read("horse_runs").set_index(["horse_id", "race_index"])
    assert runs.loc[(HORSE_RECENT, 29), "race_class"] == "5"
    assert runs.loc[(HORSE_RETIRED, 301), "venue"] == "HV"


def test_enrichment_stats_are_reported(raw_root):
    summary = build_now()
    races = {r["race_id"]: r["status"] for r in expected_rows("races")}
    starters = [
        r
        for r in expected_rows("runners")
        if races[r["race_id"]] == "run" and not r["is_scratched"] and r["horse_id"]
    ]
    runs = expected_rows("horse_runs")
    genuine = {(r["horse_id"], r["race_date"]) for r in runs if not withdrawn(r["placing_raw"])}
    matched = [
        s for s in starters if (s["horse_id"], race_keys(s["race_id"])[1].isoformat()) in genuine
    ]
    profiled = [s for s in starters if s["horse_id"] in {r["horse_id"] for r in runs}]
    e = summary.enrichment
    assert e["eligible"] == len(starters) == 10
    assert (e["matched"], e["with_profile"]) == (len(matched), len(profiled)) == (2, 2)
    assert e["match_rate"] == pytest.approx(2 / 10)
    assert e["match_rate_profiled"] == 1.0
    assert e["rating_rate"] == pytest.approx(2 / 10)
    assert e["withdrawal_rows_ignored"] == sum(withdrawn(r["placing_raw"]) for r in runs) == 2
    assert e["duplicate_run_keys_dropped"] == 0
    assert record()["enrichment"] == json.loads(json.dumps(e))


def test_withdrawals_never_feed_the_join_and_the_join_never_multiplies_rows(raw_root, adapter):
    night = "2026-09-16"
    adapter.overrides["horse", HORSE_RECENT] = {
        "horses": [],
        "horse_runs": [
            horse_run(HORSE_RECENT, night, 30, rating=99, gear_raw="XX", placing_raw="WV"),
            horse_run(HORSE_RECENT, night, 29, rating=31, gear_raw="H/TT", placing_raw="1"),
            horse_run(HORSE_RECENT, night, 31, rating=32, gear_raw="B", placing_raw="5"),  # twin
            horse_run("H_2023_J308", night, 29, rating=40, placing_raw="WV-A"),
            horse_run("H_2023_J504", night, 29, rating=41, gear_raw="TT", placing_raw="WX"),
            horse_run("H_2024_B111", night, 29, rating=42, gear_raw="V", placing_raw=" wr "),
        ],
    }
    summary = build_now()
    runners = read("runners")
    assert len(runners) == 14 and not runners.duplicated(["race_id", "horse_id"]).any()
    night_runners = runners[runners["race_id"] == RACE_2026].set_index("horse_id")
    assert night_runners.loc[HORSE_RECENT, ["rating", "gear_raw"]].tolist() == [31, "H/TT"]
    for withdrawn_horse in ("H_2023_J308", "H_2023_J504", "H_2024_B111"):
        assert pd.isna(night_runners.loc[withdrawn_horse, "rating"]), withdrawn_horse
        assert night_runners.loc[withdrawn_horse, "gear_raw"] is None, withdrawn_horse
    assert summary.enrichment["duplicate_run_keys_dropped"] == 1
    assert summary.enrichment["withdrawal_rows_ignored"] == 4 + 1  # + the retired horse's WX-A
    assert len(read("horse_runs")) == 6 + 2


# ---------------------------------------------------------------- exclusions and failures


def test_key_mismatch_pages_are_excluded_and_listed(raw_root):
    summary = build_now()
    assert summary.key_mismatch == [f"results/{MISMATCHED_KEY}"]
    assert summary.excluded_with_mismatch == [f"sectional/{MISMATCHED_KEY}"]
    for name in ("races", "runners", "dividends", "sectionals", "incidents"):
        assert not read(name)["race_id"].str.contains("2026-09-13").any(), name
    assert (read("races")["race_id"] == RACE_2026).sum() == 1  # the same body under the right key
    assert len(read("sectionals")) == len(SECTIONAL_2026["sectionals"])
    assert record()["key_mismatch"] == [f"results/{MISMATCHED_KEY}"]
    assert MISMATCHED_KEY in report_path().read_text(encoding="utf-8")


def test_corrupt_gzip_and_a_raising_parser_are_reported_never_fatal(raw_root, adapter):
    add_page("results", "2026-09-16_HV_R02", 2026, b"never readable", raw=b"\x1f\x8b\x08 not gzip")
    add_page("results", "2026-09-16_HV_R03", 2026, b"zero bytes on disk", raw=b"")
    whole = gzip.compress(page("results", KEY_2026).body)
    add_page("results", "2026-09-16_HV_R04", 2026, b"truncated", raw=whole[: len(whole) // 2])
    adapter.overrides["sectional", KEY_2026] = ValueError("boom\nsecond line")
    summary = build_now()
    assert [(f["page_type"], f["key"]) for f in summary.unreadable] == [
        ("results", "2026-09-16_HV_R02"),
        ("results", "2026-09-16_HV_R03"),
        ("results", "2026-09-16_HV_R04"),
    ]
    assert summary.failed == [
        {"page_type": "sectional", "key": KEY_2026, "error": "ValueError: boom second line"}
    ]
    assert summary.rows["races"] == 4 and summary.rows["sectionals"] == 0
    report = report_path().read_text(encoding="utf-8")
    assert f"sectional/{KEY_2026}" in report and "results/2026-09-16_HV_R02" in report
    # failures are not cached: once the parser is fixed the very next build picks the page up
    adapter.overrides.clear()
    adapter.calls.clear()
    again = build_now()
    assert adapter.calls == [("sectional", KEY_2026)]  # unreadable files never reach the parser
    assert again.failed == [] and len(again.unreadable) == 3
    assert again.rows["sectionals"] == len(SECTIONAL_2026["sectionals"])


def test_a_missing_raw_file_is_unreadable_not_fatal(raw_root):
    paths.raw_page_path(SOURCE, "horse", 2023, HORSE_RECENT).unlink()
    summary = build_now()
    assert [f["key"] for f in summary.unreadable] == [HORSE_RECENT]
    assert list(read("horses")["horse_id"]) == [HORSE_RETIRED]


def test_a_raw_file_that_is_not_what_the_manifest_recorded_is_skipped_and_reported(raw_root):
    """Valid gzip, wrong content (a half-finished refetch, a mixed-up restore): filing it under the
    manifest's sha256 would poison the parse cache and make the snapshot id lie."""
    raw = paths.raw_page_path(SOURCE, "results", 2026, KEY_2026)
    raw.write_bytes(gzip.compress(page("results", KEY_2010).body, mtime=0))
    summary = build_now()
    assert [(f["page_type"], f["key"]) for f in summary.unreadable] == [("results", KEY_2026)]
    races = read("races")
    assert RACE_2026 not in set(races["race_id"]) and not races["race_id"].duplicated().any()
    # nothing was cached for it: once the right file is back the next build picks it up
    raw.write_bytes(gzip.compress(page("results", KEY_2026).body, mtime=0))
    again = build_now()
    assert again.unreadable == [] and again.n_parsed == 1
    assert RACE_2026 in set(read("races")["race_id"])


def test_unknown_and_upcoming_page_types_are_skipped(raw_root, adapter):
    upcoming = f"{SOURCE}:2026-09-30:HV:01"
    add(Page("racecard", "2026-09-30_HV_R01", 2026, {"races": [{"race_id": upcoming}]}))
    add_page("telemetry", KEY_2026, 2026, b'{"tables": {}}')
    adapter.versions["corunning"] = None  # a known page type whose parser is gone
    summary = build_now()
    assert summary.skipped_page_types == {"racecard": 1, "telemetry": 1, "corunning": 1}
    assert summary.n_selected == len(PAGES) - 1 and summary.failed == []
    assert summary.parser_versions == {
        t: VERSIONS[t] for t in ("fixture", "horse", "results", "sectional")
    }
    assert not {pt for pt, _ in adapter.calls} & {"racecard", "telemetry", "corunning"}
    assert upcoming not in set(read("races")["race_id"])  # race cards live apart
    assert summary.rows["running_comments"] == 0


def test_pending_and_empty_manifest_rows_are_ignored(raw_root):
    add_page("results", "2026-09-16_HV_R05", 2026, page("results", KEY_2026).body, status="pending")
    add_page("results", "2026-09-16_HV_R06", 2026, b"No information.", status="empty")
    assert build_now().n_selected == len(PAGES)
    rows = build.read_manifest(SOURCE)
    assert [(r["page_type"], r["key"]) for r in rows] == sorted((p.page_type, p.key) for p in PAGES)
    (retired,) = [r for r in rows if r["key"] == HORSE_RETIRED]
    assert (retired["year"], retired["sha256"], retired["fetched_utc"]) == (
        2008,
        page("horse", HORSE_RETIRED).sha256,
        page("horse", HORSE_RETIRED).fetched_utc,
    )


def test_abandoned_pages_without_content_are_ignored_not_unreadable(raw_root):
    """A page dropped before it was ever fetched (abandoned, no raw file, no sha256) has nothing to
    read: not selected, not unreadable. An abandoned page that was fetched still flows through."""
    conn = sqlite3.connect(paths.manifest_path(SOURCE))
    with conn:
        conn.execute(
            "INSERT INTO fetch_queue (jurisdiction, page_type, key, url, year, status, attempts)"
            " VALUES (?, 'horse', 'H_2099_Z999', 'https://x.test/z', 2099, 'abandoned', 0)",
            (SOURCE,),
        )
    conn.close()
    summary = build_now()
    assert summary.n_selected == len(PAGES) and summary.unreadable == []
    assert RACE_ABANDONED in set(read("races")["race_id"])


@pytest.mark.parametrize("old_key, new_key", [("DUP_A", "DUP_B"), ("DUP_B", "DUP_A")])
def test_two_pages_claiming_one_horse_latest_profile_wins(raw_root, old_key, new_key):
    for key, name, fetched in (
        (old_key, "OLD NAME", "2026-09-01T00:00:00+00:00"),
        (new_key, "NEW NAME", "2026-09-20T00:00:00+00:00"),
    ):
        tables_ = {
            "horses": [horse("H_2020_E001", name, False)],
            "horse_runs": [
                horse_run("H_2020_E001", "2021-01-01", 7, gear_raw=name, placing_raw="3")
            ],
        }
        add(Page("horse", key, 2020, tables_, fetched_utc=fetched))
    summary = build_now()
    horses = read("horses")
    assert horses["horse_id"].is_unique
    assert horses.set_index("horse_id").loc["H_2020_E001", "name"] == "NEW NAME"
    runs = read("horse_runs")
    assert list(runs.loc[runs["horse_id"] == "H_2020_E001", "gear_raw"]) == ["NEW NAME"]
    assert summary.duplicate_pages == [f"horse/{old_key}"]


@pytest.mark.parametrize("copy_key", ["zz-same-race", "0000-same-race"])
def test_a_second_page_for_the_same_race_is_dropped_whole(raw_root, copy_key):
    """Two pages claiming one race: the smaller page key wins and the other page's rows leave every
    table its page type feeds."""
    second = copy.deepcopy(RESULTS_2026)
    second["races"][0]["race_name"] = "SYNTH SPRINT (COPY)"
    second["dividends"] = second["dividends"][:2]
    add(results_page(copy_key, 2026, second))
    summary = build_now()
    winner, loser = sorted([KEY_2026, copy_key])
    assert summary.duplicate_pages == [f"results/{loser}"]
    races = read("races")
    assert races["race_id"].is_unique
    won = RESULTS_2026 if winner == KEY_2026 else second
    assert races.set_index("race_id").loc[RACE_2026, "race_name"] == won["races"][0]["race_name"]
    assert len(read("runners")) == 14
    others = [p for p in GOOD_PAGES if p.page_type == "results" and p.key != KEY_2026]
    assert len(read("dividends")) == len(won["dividends"]) + sum(
        len(p.tables["dividends"]) for p in others
    )


# (horse_no in RACE_2026, column, what the parser hands over, what the table holds)
BAD_CELLS = [
    (4, "draw", "seven", None),
    (4, "win_odds", "n/a", None),
    (4, "actual_wt_lb", 126.0, 126),
    (4, "running_positions", [1, None, 2], [1, 2]),
    (2, "draw", 10**30, None),  # beyond int64
    (2, "actual_wt_lb", 2**70, None),
    (2, "running_positions", [1, 2**40, 3], [1, 3]),  # beyond int32: a hole, like a null
    (7, "running_positions", [2**31], []),
    (7, "win_odds", float("inf"), None),  # missing is null, never inf / NaN
    (7, "finish_time_s", float("nan"), None),
    (1, "draw", "11", 11),
]


def test_bad_cells_become_null_instead_of_breaking_the_build(raw_root, adapter):
    """Scraped values are untrusted: a bad cell costs that cell, never the build or its column."""
    hostile = copy.deepcopy(RESULTS_2026)
    by_no = {row["horse_no"]: row for row in hostile["runners"]}
    for horse_no, column, value, _ in BAD_CELLS:
        by_no[horse_no][column] = value
    adapter.overrides["results", KEY_2026] = hostile
    summary = build_now()
    assert summary.failed == []
    runners = read("runners").set_index(["race_id", "horse_no"])
    for horse_no, column, _, stored in BAD_CELLS:
        assert cell(runners.loc[(RACE_2026, horse_no), column]) == stored, (horse_no, column)
    touched = {(RACE_2026, horse_no, column) for horse_no, column, *_ in BAD_CELLS}
    clean = {(r["race_id"], r["horse_no"]): r for r in expected_rows("runners")}
    for column in {column for _, column, *_ in BAD_CELLS}:
        for (race_id, horse_no), value in runners[column].items():
            if (race_id, horse_no, column) not in touched:
                assert cell(value) == cell(clean[race_id, horse_no][column]), (race_id, column)
    stored_floats = pq.read_table(
        tables.table_path(SOURCE, "runners"), columns=["win_odds", "finish_time_s"]
    )
    for column in ("win_odds", "finish_time_s"):
        values = stored_floats[column].to_pylist()
        assert all(math.isfinite(v) for v in values if v is not None), column
        nulls = sum(r[column] is None for r in clean.values()) + sum(
            1 for _, c, _, stored in BAD_CELLS if c == column and stored is None
        )
        assert stored_floats[column].null_count == nulls, column  # real Parquet nulls


# ---------------------------------------------------------------- incremental behaviour


def test_second_build_parses_nothing_and_writes_the_same_tables(raw_root, adapter):
    first = build_now()
    assert first.n_parsed == len(adapter.calls) == len(PAGES)
    before, frames = table_bytes(), {name: read(name) for name in NINE}
    adapter.calls.clear()
    second = build_now()
    assert adapter.calls == [] and second.n_parsed == 0
    assert second.n_from_cache == second.n_included == len(GOOD_PAGES)
    assert second.snapshot_id == first.snapshot_id and second.rows == first.rows
    assert table_bytes() == before
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(read(name), frame)


def test_changed_sha256_reparses_only_that_page(built_root, adapter):
    refetched = copy.deepcopy(RESULTS_2026)
    refetched["races"][0]["going"] = "GOOD TO YIELDING"
    add(results_page(KEY_2026, 2026, refetched))
    summary = build_now()
    assert adapter.calls == [("results", KEY_2026)]
    assert (summary.n_parsed, summary.n_from_cache) == (1, len(GOOD_PAGES) - 1)
    assert read("races").set_index("race_id").loc[RACE_2026, "going"] == "GOOD TO YIELDING"
    assert record()["snapshot_id"] == summary.snapshot_id != ""


def _snapshot(pages: list[Page]) -> str:
    """sha1 over the sorted (page_type, key, sha256) of every page whose rows are in the tables."""
    included = sorted((p.page_type, p.key, p.sha256) for p in pages)
    return hashlib.sha1("\n".join("\t".join(item) for item in included).encode()).hexdigest()


def test_snapshot_id_follows_the_page_hashes(built_root):
    assert record()["snapshot_id"] == _snapshot(GOOD_PAGES)
    changed = results_page(KEY_2026, 2026, {**RESULTS_2026, "incidents": []})
    add(changed)
    now = [changed if (p.page_type, p.key) == ("results", KEY_2026) else p for p in GOOD_PAGES]
    assert build_now().snapshot_id == _snapshot(now) != _snapshot(GOOD_PAGES)


def test_parser_version_bump_reparses_that_page_type_only(built_root, adapter):
    adapter.versions["horse"] = "6"
    summary = build_now()
    assert sorted(adapter.calls) == [("horse", HORSE_RETIRED), ("horse", HORSE_RECENT)]
    assert summary.parser_versions["horse"] == "6" and summary.n_parsed == 2
    adapter.calls.clear()
    assert build_now().n_parsed == 0 and adapter.calls == []


def test_page_types_and_limit_only_restrict_parsing(raw_root, adapter):
    first = build_now(page_types=["results"])
    assert {page_type for page_type, _ in adapter.calls} == {"results"}
    assert first.rows["races"] == 4 and first.rows["horses"] == 0
    assert first.n_not_parsed == len([p for p in PAGES if p.page_type != "results"])
    adapter.calls.clear()
    second = build_now(limit=1)  # one page per type, newest key first
    assert sorted(adapter.calls) == [
        ("corunning", KEY_2010), ("fixture", "2026-10"), ("horse", HORSE_RECENT),
        ("sectional", KEY_2026),
    ]  # fmt: skip
    assert second.rows["races"] == 4  # results parsed by the first run are still there
    assert second.n_not_parsed == 5
    adapter.calls.clear()
    third = build_now()
    assert ("results", KEY_2026) not in adapter.calls and third.n_not_parsed == 0
    assert third.rows["horses"] == 2 and third.rows["meetings"] == 9


def test_process_pool_gives_the_same_tables_as_in_process_parsing(raw_root, monkeypatch):
    populate(FILLER)
    inline = build_now()
    before = table_bytes()
    shutil.rmtree(paths.interim_dir(SOURCE))
    shutil.rmtree(paths.processed_dir(SOURCE))
    monkeypatch.setattr(store_fakes, "WORKER_STATE", {})
    monkeypatch.setattr(parse_worker, "parse_job", parse_job_in_fresh_worker)
    pooled = build.build_tables(SOURCE, workers=2)
    assert pooled.n_parsed == inline.n_parsed == len(PAGES) + len(FILLER)
    assert pooled.snapshot_id == inline.snapshot_id and table_bytes() == before
    workers = worker_rows()
    assert len(workers) == len(PAGES) + len(FILLER)
    assert os.getpid() not in {w["pid"] for w in workers}  # every page parsed in a worker
    # fresh interpreters (spawn, the macOS default), not forks of the test process
    assert {w["registry_inherited"] for w in workers} == {False}


def test_a_parser_process_that_dies_is_an_error_not_a_hang(raw_root, monkeypatch):
    """A segfault in the HTML parser or an OOM kill takes the worker with it; an unattended build
    must fail loudly instead of waiting for the lost pages for ever."""
    populate(FILLER)
    monkeypatch.setenv(TEST_PID_VAR, str(os.getpid()))
    monkeypatch.setattr(parse_worker, "parse_job", die_in_worker)
    outcome: list[object] = []

    def run() -> None:
        try:
            outcome.append(build.build_tables(SOURCE, workers=2))
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(60)
    assert not thread.is_alive(), "the build hangs when a worker process dies"
    assert len(outcome) == 1 and isinstance(outcome[0], RuntimeError), outcome


def test_pages_parsed_before_a_crash_are_kept_in_the_cache(raw_root, adapter):
    adapter.interrupt_at = 3
    with pytest.raises(KeyboardInterrupt):
        build_now()
    kept = adapter.calls[:2]
    with ParseCache(cache_path(SOURCE)) as cache:
        index = cache.index()
    assert sorted(index) == sorted(kept)
    for page_type, key in kept:
        assert index[page_type, key][:2] == (page(page_type, key).sha256, VERSIONS[page_type])
    adapter.interrupt_at = None
    adapter.calls.clear()
    summary = build_now()
    assert set(adapter.calls) == {(p.page_type, p.key) for p in PAGES} - set(kept)
    assert summary.n_parsed == len(PAGES) - 2
    assert summary.n_from_cache == len(set(kept) & {(p.page_type, p.key) for p in GOOD_PAGES})


def test_default_workers_leave_two_cores_free(monkeypatch):
    for cpus, expected in ((16, 6), (8, 6), (4, 2), (3, 1), (1, 1), (None, 1)):
        monkeypatch.setattr(os, "cpu_count", lambda cpus=cpus: cpus)
        assert build.default_workers() == expected, cpus


# ---------------------------------------------------------------- the manifest belongs to ingest


def test_manifest_is_only_ever_read(raw_root):
    manifest = paths.manifest_path(SOURCE)
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    os.chmod(manifest, 0o444)
    try:
        build_now()
    finally:
        os.chmod(manifest, 0o644)
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == before
    raw_files = sorted(p.name for p in manifest.parent.rglob("*") if p.is_file())
    assert all(name.endswith(".html.gz") or name == "manifest.sqlite" for name in raw_files)


def test_wal_mode_manifest_with_a_live_writer_can_be_read(raw_root):
    writer = sqlite3.connect(paths.manifest_path(SOURCE))
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("UPDATE fetch_queue SET attempts = attempts + 1")
        writer.commit()
        assert build_now().n_selected == len(PAGES)
    finally:
        writer.close()


def test_no_manifest_is_a_clear_error(data, adapter):
    with pytest.raises(FileNotFoundError, match="manifest"):
        build_now()
