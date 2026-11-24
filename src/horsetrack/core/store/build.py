"""Manifest + raw page cache -> the store's Parquet tables (ported from HKTrack's store/build.py;
contract: docs/specs/store.md, "Build").

    manifest (read-only) -> pages to parse (new sha256 or parser version) -> parse cache
    parse cache -> one frame per table -> keys denormalised, runners enriched from horse_runs
    -> processed_dir(source)/<table>.parquet (atomic) + _build.json + reports_dir()/store-build-*.md

`page_types` and `limit` only restrict what is PARSED in this run; the tables are always assembled
from every cached page that still matches the manifest, so a partial run never drops rows an
earlier run parsed. Row order is fixed by page key and then by each table's sort key, and no
wall-clock value enters a table, which makes the output byte-stable. Source-agnostic: the source is
always an argument, and the only thing asked of its adapter here is the season's start month.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from horsetrack.core.ingest.adapters import get_adapter
from horsetrack.core.paths import manifest_path, processed_dir, reports_dir
from horsetrack.core.store import parse_worker, schemas
from horsetrack.core.store.cache import ParseCache, cache_path
from horsetrack.core.store.parse_worker import ParseJob, ParseResult
from horsetrack.core.store.tables import ARROW_TYPES

MANIFEST_STATUSES = ("fetched", "abandoned")
BUILD_FILE = "_build.json"
PAGE_KEY = "_page_key"  # transient column: which page a row came from (dropped before writing)

_PANDAS_TYPES = {pa.int64(): pd.Int64Dtype(), pa.bool_(): pd.BooleanDtype()}
_UTC_DTYPE = "datetime64[us, UTC]"  # a `utc` column as pandas holds it (tables.py)

# horse_runs lists withdrawals as rows (placing WV, WV-A, WX, ...): not runs, never in the join
_WITHDRAWN_PREFIXES = ("WV", "WX", "WR")
_RACE_ID = re.compile(r"^([^:]+:(\d{4}-\d{2}-\d{2}):([^:]+)):(\d+)$")

_POOL_MIN_PAGES = 64  # below this, starting worker processes costs more than parsing in-process
_POOL_CHUNK = 16
_CACHE_BATCH = 200
_FLUSH_ROWS = 50_000
_REPORT_MAX_ITEMS = 200
_INT64_MAX = 2**63 - 1
_INT32_MAX = 2**31 - 1


@dataclass
class BuildSummary:
    jurisdiction: str  # the source key, as in the tables' `jurisdiction` column
    snapshot_id: str = ""
    seconds: float = 0.0
    n_selected: int = 0  # manifest pages of known page types with status fetched/abandoned
    n_included: int = 0  # pages whose rows are in the tables
    n_parsed: int = 0  # parsed in this run
    n_from_cache: int = 0  # included without parsing
    n_not_parsed: int = 0  # left for a later run (page_types / limit filter)
    n_empty: int = 0  # parsed fine, no data rows
    failed: list[dict[str, str]] = field(default_factory=list)  # the parser raised
    unreadable: list[dict[str, str]] = field(default_factory=list)  # raw file missing / corrupt
    key_mismatch: list[str] = field(default_factory=list)  # "page_type/key" flagged by the parser
    excluded_with_mismatch: list[str] = field(default_factory=list)  # others sharing such a key
    duplicate_pages: list[str] = field(default_factory=list)  # a 2nd page claiming a race / horse
    skipped_page_types: dict[str, int] = field(default_factory=dict)  # no table, or no parser
    extra_columns: dict[str, list[str]] = field(default_factory=dict)  # dropped parser columns
    parser_versions: dict[str, str] = field(default_factory=dict)
    rows: dict[str, int] = field(default_factory=dict)
    enrichment: dict[str, Any] = field(default_factory=dict)
    date_min: str | None = None
    date_max: str | None = None
    build_path: str = ""
    report_path: str = ""

    def lines(self) -> list[str]:
        """The short summary a CLI prints."""
        rows = ", ".join(f"{name} {n}" for name, n in self.rows.items())
        e = self.enrichment
        out = [
            f"store build {self.jurisdiction}: snapshot {self.snapshot_id[:12]}"
            f" in {self.seconds:.1f} s",
            f"pages: {self.n_selected} selected, {self.n_included} in tables"
            f" ({self.n_parsed} parsed now, {self.n_from_cache} from cache),"
            f" {self.n_empty} without rows",
            f"excluded: {len(self.failed)} parse failures, {len(self.unreadable)} unreadable,"
            f" {len(self.key_mismatch)} key mismatches"
            f" (+{len(self.excluded_with_mismatch)} sharing a key),"
            f" {len(self.duplicate_pages)} duplicates, {self.n_not_parsed} not parsed yet",
            f"rows: {rows}",
            f"race dates: {self.date_min} .. {self.date_max}",
        ]
        if e:
            out.append(
                f"runners enriched from horse_runs: {e['matched']} of {e['eligible']} starters"
                f" matched ({_pct(e['match_rate'])}); {_pct(e['match_rate_profiled'])} of the"
                f" {e['with_profile']} whose horse page is cached; rating present on"
                f" {_pct(e['rating_rate'])}"
            )
        if self.skipped_page_types:
            skipped = ", ".join(f"{t} {n}" for t, n in sorted(self.skipped_page_types.items()))
            out.append(f"page types skipped: {skipped}")
        out.append(f"report {self.report_path}")
        return out


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


# ---------------------------------------------------------------- manifest and work list


def read_manifest(source: str) -> list[dict[str, Any]]:
    """Fetched / abandoned pages with a sha256, in (page_type, key) order. The manifest belongs to
    ingest and a backfill may be writing to it right now: it is opened read-only and closed again
    as soon as the rows are in memory."""
    path = manifest_path(source)
    if not path.exists():
        raise FileNotFoundError(f"no manifest at {path}: nothing has been scraped for {source!r}")
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=30.0)
    try:
        conn.row_factory = sqlite3.Row
        marks = ", ".join("?" * len(MANIFEST_STATUSES))
        rows = conn.execute(
            "SELECT page_type, key, year, url, final_url, sha256, fetched_utc FROM fetch_queue"
            f" WHERE jurisdiction = ? AND status IN ({marks})"
            # no sha256, no raw body (a page abandoned before it was ever fetched): selecting it
            # would only report a phantom "unreadable"
            " AND sha256 IS NOT NULL AND sha256 <> '' ORDER BY page_type, key",
            (source, *MANIFEST_STATUSES),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def default_workers() -> int:
    """Leave two cores free, at most 6 workers, at least 1."""
    return max(1, min(6, (os.cpu_count() or 2) - 2))


def season_start_month(source: str) -> int:
    """The month a racing season starts in: the adapter's optional `season_start_month()` hook,
    else 1 (calendar-year seasons)."""
    hook = getattr(get_adapter(source), "season_start_month", None)
    month = int(hook()) if callable(hook) else 1
    if not 1 <= month <= 12:
        raise ValueError(f"season_start_month() of {source!r} is {month}, not a month")
    return month


def _race_instants(source: str, hook_name: str) -> dict[str, datetime]:
    """race_id -> instant in UTC from the adapter's optional `<hook_name>()` hook, else none."""
    hook = getattr(get_adapter(source), hook_name, None)
    logged = dict(hook()) if callable(hook) else {}
    return {str(rid): _coerce_value(at, schemas.UTC) for rid, at in logged.items()}


def result_published_utc(source: str) -> dict[str, datetime]:
    """race_id -> when the race's official result was first seen published, in UTC: the adapter's
    optional `result_published_utc()` hook (ticket #225), else none."""
    return _race_instants(source, "result_published_utc")


def result_unpublished_utc(source: str) -> dict[str, datetime]:
    """race_id -> the race's lower bound, the last logger poll that did not show its result, in
    UTC: the adapter's optional `result_unpublished_utc()` hook (ticket #225), else none."""
    return _race_instants(source, "result_unpublished_utc")


def _season_of(race_date: date, start_month: int) -> str:
    """Season label, e.g. 2026-09-23 -> '2026/27' when seasons start in September."""
    start_year = race_date.year if race_date.month >= start_month else race_date.year - 1
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def _job(source: str, row: dict[str, Any], version: str) -> ParseJob:
    return ParseJob(
        source, row["page_type"], row["key"], int(row["year"]), row["url"] or "",
        row["final_url"], row["fetched_utc"], row["sha256"] or "", version,
    )  # fmt: skip


def _stale_jobs(
    source: str,
    pages: list[dict[str, Any]],
    wanted: dict[tuple[str, str], tuple[str, str]],
    index: dict[tuple[str, str], tuple[str, str, bool]],
    page_types: list[str] | None,
    limit: int | None,
) -> list[ParseJob]:
    """Pages whose cached parse is missing or was made from another sha256 / parser version."""
    taken: Counter[str] = Counter()
    jobs = []
    for row in sorted(pages, key=lambda r: r["key"], reverse=True):  # newest first, for `limit`
        page = (row["page_type"], row["key"])
        if index.get(page, ("", ""))[:2] == wanted[page]:
            continue
        if page_types is not None and row["page_type"] not in page_types:
            continue
        if limit is not None and taken[row["page_type"]] >= limit:
            continue
        taken[row["page_type"]] += 1
        jobs.append(_job(source, row, wanted[page][1]))
    return sorted(jobs, key=lambda job: (job.page_type, job.key))


def _run_jobs(jobs: list[ParseJob], workers: int) -> Iterator[ParseResult]:
    if workers <= 1 or len(jobs) < _POOL_MIN_PAGES:
        yield from (parse_worker.parse_job(job) for job in jobs)
        return
    # spawn (the macOS default) everywhere: workers import only the light parse_worker module.
    # concurrent.futures, not multiprocessing.Pool: when a worker dies (a segfault in the HTML
    # parser, an OOM kill) Pool waits for the lost task for ever, the executor raises.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=context) as pool:
        try:
            yield from pool.map(parse_worker.parse_job, jobs, chunksize=_POOL_CHUNK)
        except BrokenProcessPool as exc:
            raise RuntimeError(
                "a parser process died (crash or out of memory); pages parsed so far are cached:"
                " run again with one worker to find the page"
            ) from exc


def _parse_pages(
    jobs: list[ParseJob], workers: int, cache: ParseCache, summary: BuildSummary
) -> None:
    batch: list[ParseResult] = []
    try:
        for result in _run_jobs(jobs, workers):
            if result.status == parse_worker.STATUS_OK:
                batch.append(result)
                summary.n_parsed += 1
            else:
                failed = result.status == parse_worker.STATUS_FAILED
                target = summary.failed if failed else summary.unreadable
                target.append(
                    {"page_type": result.page_type, "key": result.key, "error": result.error or ""}
                )
            if len(batch) >= _CACHE_BATCH:
                cache.put(batch)
                batch = []
    finally:  # an interrupted or crashed run keeps what it has parsed
        cache.put(batch)
    for items in (summary.failed, summary.unreadable):
        items.sort(key=lambda item: (item["page_type"], item["key"]))


# ---------------------------------------------------------------- parse cache -> typed frames


def _coerce_value(value: Any, kind: str) -> Any:
    if value is None:
        return None
    try:
        if kind == schemas.INT:
            number = int(value) if isinstance(value, int) else float(value)
            fits = number == int(number) and abs(int(number)) <= _INT64_MAX
            return int(number) if fits else None
        if kind == schemas.FLOAT:
            number = float(value)
            return number if math.isfinite(number) else None  # missing is null, never inf / NaN
        if kind == schemas.STR:
            return value if isinstance(value, str) else str(value)
        if kind == schemas.BOOL:
            return bool(value)
        if kind == schemas.DATE:
            return date.fromisoformat(str(value)[:10])
        if kind == schemas.UTC:
            stamp = datetime.fromisoformat(str(value))
            if stamp.tzinfo is None:
                return stamp.replace(tzinfo=timezone.utc)
            return stamp.astimezone(timezone.utc)
        if kind == schemas.INT_LIST:  # list<int32>: a number that does not fit is a hole
            return [int(v) for v in value if v is not None and abs(int(v)) <= _INT32_MAX]
        return [float(v) for v in value if v is not None]
    except (TypeError, ValueError, OverflowError):
        return None


def _to_arrow(values: list[Any], kind: str) -> pa.Array:
    """Typed array from JSON values. The fast path trusts the parsers; one bad cell sends the
    column through the per-value path, where what cannot be converted becomes null."""
    target = ARROW_TYPES[kind]
    try:
        if kind in (schemas.DATE, schemas.UTC):
            return pa.array(values, type=pa.string()).cast(target)
        array = pa.array(values, type=target)
        if pa.types.is_list(target) and array.flatten().null_count:
            raise ValueError("null inside a list")  # a list holds values only: drop the holes
        if kind == schemas.FLOAT and pc.all(pc.is_finite(array), min_count=0).as_py() is False:
            raise ValueError("inf / NaN in a float column")  # nulled below
        return array
    except (pa.ArrowException, TypeError, ValueError, OverflowError):
        return pa.array([_coerce_value(v, kind) for v in values], type=target)


class _TableBuffer:
    """Column lists of one table, flushed to Arrow in chunks so Python objects never pile up."""

    def __init__(self, spec: schemas.TableSpec):
        self.spec = spec
        self.kinds = {**spec.columns, PAGE_KEY: schemas.STR}
        self.extra: set[str] = set()
        self._chunks: list[pa.Table] = []
        self._reset()

    def _reset(self) -> None:
        self._columns: dict[str, list[Any]] = {name: [] for name in self.kinds}
        self._n = 0

    def add(self, page_key: str, columns: list[str], rows: list[list[Any]]) -> None:
        at = {name: i for i, name in enumerate(columns)}
        for name, out in self._columns.items():
            i = at.get(name)
            if name == PAGE_KEY:
                out.extend([page_key] * len(rows))
            elif i is None:
                out.extend([None] * len(rows))
            else:
                out.extend(row[i] for row in rows)
        self.extra.update(name for name in columns if name not in self.kinds)
        self._n += len(rows)
        if self._n >= _FLUSH_ROWS:
            self._flush()

    def _flush(self) -> None:
        arrays = {
            name: _to_arrow(values, self.kinds[name]) for name, values in self._columns.items()
        }
        self._chunks.append(pa.table(arrays))
        self._reset()

    def frame(self) -> pd.DataFrame:
        self._flush()
        return pa.concat_tables(self._chunks).to_pandas(types_mapper=_PANDAS_TYPES.get)


def _collect(
    cache: ParseCache, wanted: dict[tuple[str, str], tuple[str, str]], summary: BuildSummary
) -> tuple[dict[str, pd.DataFrame], list[tuple[str, str, str]]]:
    """Frames per table from every cached page that still matches the manifest (`wanted`: page ->
    sha256, parser version), plus the (page_type, key, sha256) list that defines the snapshot."""
    index = cache.index()
    current = {page: entry for page, entry in index.items() if wanted.get(page) == entry[:2]}
    mismatch_keys = {key for (_, key), entry in current.items() if entry[2]}
    buffers = {name: _TableBuffer(schemas.TABLES[name]) for name in schemas.FED_BY}
    included: list[tuple[str, str, str]] = []
    for page_type in schemas.PAGE_TYPES:
        targets = [buffers[name] for name, types in schemas.FED_BY.items() if page_type in types]
        for key, sha256, _version, payload in cache.payloads(page_type):
            entry = current.get((page_type, key))
            if entry is None:  # cached from an older sha256 / parser version, not re-parsed yet
                continue
            if key in mismatch_keys:  # the site served another meeting under this key: drop all
                listed = summary.key_mismatch if entry[2] else summary.excluded_with_mismatch
                listed.append(f"{page_type}/{key}")
                continue
            tables = parse_worker.decode_tables(payload)
            n_rows = 0
            for buffer in targets:
                part = tables.get(buffer.spec.name)
                if part and part["rows"]:
                    buffer.add(key, part["columns"], part["rows"])
                    n_rows += len(part["rows"])
            summary.n_empty += n_rows == 0
            included.append((page_type, key, sha256))
    summary.extra_columns = {name: sorted(b.extra) for name, b in buffers.items() if b.extra}
    return {name: buffer.frame() for name, buffer in buffers.items()}, included


# ---------------------------------------------------------------- table assembly


def _nulls(series: pd.Series) -> pd.Series:
    """Object column with None (not NaN / NaT) for missing, what the Arrow conversion expects."""
    out = series.astype(object)
    return out.where(out.notna(), None)


def _race_keys(race_id: pd.Series) -> pd.DataFrame:
    """race_id -> meeting_id, race_date, venue, race_no (nulls when malformed)."""
    parts = race_id.astype(object).str.extract(_RACE_ID)
    dates = {text: _coerce_value(text, schemas.DATE) for text in parts[1].dropna().unique()}
    return pd.DataFrame(
        {
            "meeting_id": _nulls(parts[0]),
            "race_date": _nulls(parts[1].map(dates)),
            "venue": _nulls(parts[2]),
            "race_no": pd.to_numeric(parts[3], errors="coerce").astype("Int64"),
        }
    )


def _season(race_date: pd.Series, start_month: int) -> pd.Series:
    labels = {d: _season_of(d, start_month) for d in race_date.dropna().unique()}
    return _nulls(race_date.map(labels))


def _drop_duplicate_pages(
    frames: dict[str, pd.DataFrame],
    table: str,
    id_column: str,
    summary: BuildSummary,
    newest: str | None = None,
) -> None:
    """Two pages claiming the same race (or horse): one page wins, the latest `newest` value when
    given, else the smallest page key, and the loser's rows leave every table its page type
    feeds."""
    order, ascending = ([newest, PAGE_KEY], [False, True]) if newest else ([PAGE_KEY], [True])
    frame = frames[table].sort_values(order, ascending=ascending, kind="stable", na_position="last")
    losers = frame.loc[
        frame[id_column].notna() & frame.duplicated(id_column, keep="first"), PAGE_KEY
    ]
    dropped = set(losers)
    fed_by = schemas.FED_BY[table]
    summary.duplicate_pages += [f"{fed_by[0]}/{key}" for key in sorted(dropped)]
    for name, types in schemas.FED_BY.items():
        if types == fed_by:
            frames[name] = frames[name][~frames[name][PAGE_KEY].isin(dropped)]


def _meetings(fixtures: pd.DataFrame, races: pd.DataFrame, start_month: int) -> pd.DataFrame:
    """The scheduled programme, upgraded by what the results say. A meeting is `run` when any of
    its races was run and `abandoned` when all its result pages are abandoned; without results it
    stays `scheduled`. Meetings only the results know about (a calendar page not cached) are
    added."""
    meetings = fixtures.drop_duplicates("meeting_id", keep="last")
    known = set(meetings["meeting_id"])
    seen = races.loc[races["meeting_id"].notna(), ["meeting_id", "race_date", "venue", "status"]]
    extra = seen.loc[~seen["meeting_id"].isin(known)].drop_duplicates("meeting_id")
    extra = extra.drop(columns="status").assign(day_night=None, surface_mix=None)
    meetings = pd.concat([meetings, extra], ignore_index=True)
    any_run = seen["status"].eq("run").groupby(seen["meeting_id"]).any()
    outcome = meetings["meeting_id"].map(any_run.map({True: "run", False: "abandoned"}))
    meetings["status"] = _nulls(outcome.astype(object).fillna("scheduled"))  # all-NaN is float64
    meetings["season"] = _season(meetings["race_date"], start_month)
    return meetings


def _share(part: int, whole: int) -> float | None:
    return float(part) / float(whole) if whole else None


def _enrich_runners(
    runners: pd.DataFrame, horse_runs: pd.DataFrame, races: pd.DataFrame
) -> dict[str, Any]:
    """Left-join rating-at-run and gear onto `runners` IN PLACE on (horse_id, race_date); returns
    the stats. The right side is cut to one genuine run per key first, so the join cannot multiply
    runner rows."""
    placing = horse_runs["placing_raw"].astype(object).fillna("").str.strip().str.upper()
    withdrawn = placing.str.startswith(_WITHDRAWN_PREFIXES)
    usable = ~withdrawn & horse_runs["horse_id"].notna() & horse_runs["race_date"].notna()
    runs = horse_runs.loc[usable, ["horse_id", "race_date", "rating", "gear_raw"]]
    duplicated = runs.duplicated(["horse_id", "race_date"], keep="first")
    runs = runs.loc[~duplicated].assign(_matched=True)

    left = runners[["horse_id", "race_date"]].reset_index(drop=True)
    joined = left.merge(
        runs, on=["horse_id", "race_date"], how="left", validate="many_to_one", sort=False
    )
    if len(joined) != len(runners):
        raise AssertionError("horse_runs join changed the number of runner rows")
    runners["rating"] = joined["rating"].astype("Int64").to_numpy()
    runners["gear_raw"] = _nulls(joined["gear_raw"]).to_numpy()

    run_races = set(races.loc[races["status"].eq("run"), "race_id"])
    starter = (
        runners["horse_id"].notna() & ~runners["is_scratched"].fillna(False).astype(bool)
        & runners["race_id"].isin(run_races)
    ).to_numpy()  # fmt: skip
    matched = joined["_matched"].notna().to_numpy() & starter
    profiled = runners["horse_id"].isin(set(horse_runs["horse_id"].dropna())).to_numpy() & starter
    eligible = int(starter.sum())
    return {
        "eligible": eligible,
        "matched": int(matched.sum()),
        "match_rate": _share(matched.sum(), eligible),
        "with_profile": int(profiled.sum()),
        "match_rate_profiled": _share((matched & profiled).sum(), profiled.sum()),
        "rating_rate": _share((runners["rating"].notna().to_numpy() & starter).sum(), eligible),
        "withdrawal_rows_ignored": int(withdrawn.sum()),
        "duplicate_run_keys_dropped": int(duplicated.sum()),
    }


def _assemble(
    frames: dict[str, pd.DataFrame], source: str, summary: BuildSummary, present_types: set[str]
) -> dict[str, pd.DataFrame]:
    """`present_types`: the page types the manifest holds; a SOURCE_TABLES table is written only
    when one of its page types is among them."""
    start_month = season_start_month(source)
    _drop_duplicate_pages(frames, "races", "race_id", summary)
    _drop_duplicate_pages(frames, "horses", "horse_id", summary, newest="profile_scraped_utc")
    races, runners = frames["races"].copy(), frames["runners"].copy()

    keys = _race_keys(races["race_id"])
    races["race_date"], races["venue"] = keys["race_date"], keys["venue"]
    races["meeting_id"] = _nulls(
        races["meeting_id"].where(races["meeting_id"].notna(), keys["meeting_id"])
    )
    races["race_no"] = races["race_no"].fillna(keys["race_no"])
    races["season"] = _season(races["race_date"], start_month)
    for column, instants in (
        ("result_published_utc", result_published_utc(source)),
        ("result_unpublished_utc", result_unpublished_utc(source)),
    ):
        races[column] = pd.Series(
            [instants.get(rid) for rid in races["race_id"]], index=races.index, dtype=_UTC_DTYPE
        )

    keys = _race_keys(runners["race_id"])
    for column in ("meeting_id", "race_date", "venue", "race_no"):
        runners[column] = keys[column]
    runners["season"] = _season(runners["race_date"], start_month)
    summary.enrichment = _enrich_runners(runners, frames["horse_runs"], races)

    frames = {**frames, "races": races, "runners": runners}
    frames["meetings"] = _meetings(frames["meetings"], races, start_month)
    names = [name for name in schemas.FED_BY if name not in schemas.SOURCE_TABLES]
    for name in schemas.SOURCE_TABLES:
        if present_types & set(schemas.FED_BY[name]):
            frames[name] = _first_non_null(frames[name], schemas.TABLES[name].sort_by[0])
            names.append(name)
    out = {}
    for name in names:
        spec = schemas.TABLES[name]
        frame = frames[name].assign(jurisdiction=source)
        frame = frame.sort_values(
            list(spec.sort_by), kind="stable", na_position="last", ignore_index=True
        )
        out[name] = frame[list(spec.columns)]
    return out


def _first_non_null(frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """One row per id (e.g. a horse on many race cards): pages in descending key order (newest race
    first for date-led keys), and for every column the first non-null value, so a page that misses
    a value never blanks out one that another page printed."""
    frame = frame.loc[frame[id_column].notna()].sort_values(
        PAGE_KEY, ascending=False, kind="stable"
    )
    first = frame.groupby(id_column, sort=False, as_index=False).first()
    return first.astype(object).where(first.notna(), None)


# ---------------------------------------------------------------- output


def arrow_schema(table: str) -> pa.Schema:
    """The Arrow schema a store table is written with: declared columns, declared order."""
    return pa.schema(
        [(name, ARROW_TYPES[kind]) for name, kind in schemas.TABLES[table].columns.items()]
    )


def _replace_atomically(tmp: Path, path: Path) -> None:
    try:
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_tables(tables: dict[str, pd.DataFrame], folder: Path) -> None:
    """Every table goes to a tmp file first; only when all are written are they moved into place."""
    folder.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        for name, frame in tables.items():
            tmp = folder / f"{name}.parquet.{os.getpid()}.tmp"
            staged.append((tmp, folder / f"{name}.parquet"))
            table = pa.Table.from_pandas(frame, schema=arrow_schema(name), preserve_index=False)
            pq.write_table(table, tmp, compression="zstd")
    except BaseException:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)
        raise
    for tmp, path in staged:
        _replace_atomically(tmp, path)


def write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text through a tmp file in the same folder, then move it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    _replace_atomically(tmp, path)


def snapshot_id(included: Iterable[tuple[str, str, str]]) -> str:
    """sha1 over the sorted (page_type, key, sha256) of every page whose rows are in the tables."""
    lines = ("\t".join(item) for item in sorted(included))
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()


def _listing(title: str, items: list[str]) -> list[str]:
    if not items:
        return []
    more = len(items) - _REPORT_MAX_ITEMS
    tail = [f"- ... and {more} more"] if more > 0 else []
    return ["", f"## {title} ({len(items)})", *(f"- {i}" for i in items[:_REPORT_MAX_ITEMS]), *tail]


def _report(summary: BuildSummary) -> str:
    lines = [f"# Store build: {summary.jurisdiction}", ""]
    lines += [f"- {line}" for line in summary.lines()[:-1]]
    versions = ", ".join(f"{t} {v}" for t, v in summary.parser_versions.items())
    lines.append(f"- parser versions: {versions}")
    lines += _listing(
        "Parse failures", [f"{f['page_type']}/{f['key']}: {f['error']}" for f in summary.failed]
    )
    lines += _listing(
        "Unreadable raw files", [f"{f['page_type']}/{f['key']}" for f in summary.unreadable]
    )
    lines += _listing("Key mismatch (excluded)", summary.key_mismatch)
    lines += _listing(
        "Excluded because a page with the same key is mismatched", summary.excluded_with_mismatch
    )
    lines += _listing("Duplicate pages (excluded)", summary.duplicate_pages)
    lines += _listing(
        "Parser columns not in the table schema",
        [f"{t}: {', '.join(c)}" for t, c in summary.extra_columns.items()],
    )
    if summary.enrichment:
        lines += [
            "",
            "## Runner enrichment from horse_runs",
            *(f"- {k}: {v}" for k, v in summary.enrichment.items()),
        ]
    return "\n".join(lines) + "\n"


def _account(
    summary: BuildSummary,
    pages: list[dict[str, Any]],
    jobs: list[ParseJob],
    included: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Page counts for the summary; returns `included` without the pages that lost a duplicate
    check."""
    attempted = {(job.page_type, job.key) for job in jobs}
    done = {(page_type, key) for page_type, key, _ in included}
    left_out = len(summary.key_mismatch) + len(summary.excluded_with_mismatch)
    left_out += len(summary.failed) + len(summary.unreadable)
    duplicates = set(summary.duplicate_pages)
    included = [item for item in included if f"{item[0]}/{item[1]}" not in duplicates]
    summary.n_included = len(included)
    summary.n_from_cache = len(done - attempted)
    summary.n_not_parsed = len(pages) - len(done) - left_out
    return included


def build_tables(
    source: str,
    *,
    page_types: list[str] | None = None,
    workers: int | None = None,
    limit: int | None = None,
) -> BuildSummary:
    """Parse what is new, then rewrite every table. `page_types` / `limit` (pages per type, newest
    keys first) restrict the parsing of this run only. Safe to run while a backfill adds pages."""
    started = time.monotonic()
    summary = BuildSummary(source)
    manifest = read_manifest(source)

    by_type = Counter(row["page_type"] for row in manifest)
    versions = {
        t: parse_worker.parser_version(source, t) for t in schemas.PAGE_TYPES if t in by_type
    }
    summary.parser_versions = {t: v for t, v in versions.items() if v is not None}
    summary.skipped_page_types = {t: n for t, n in by_type.items() if versions.get(t) is None}
    pages = [row for row in manifest if versions.get(row["page_type"]) is not None]
    wanted = {
        (r["page_type"], r["key"]): (r["sha256"] or "", versions[r["page_type"]]) for r in pages
    }
    summary.n_selected = len(pages)

    with ParseCache(cache_path(source)) as cache:
        jobs = _stale_jobs(source, pages, wanted, cache.index(), page_types, limit)
        _parse_pages(jobs, default_workers() if workers is None else int(workers), cache, summary)
        frames, included = _collect(cache, wanted, summary)

    tables = _assemble(frames, source, summary, {page_type for page_type, _ in wanted})
    _write_tables(tables, processed_dir(source))

    included = _account(summary, pages, jobs, included)
    summary.snapshot_id = snapshot_id(included)
    summary.rows = {name: len(frame) for name, frame in tables.items()}
    races = tables["races"]
    run_dates = races.loc[races["status"].eq("run"), "race_date"].dropna()
    summary.date_min = min(run_dates).isoformat() if len(run_dates) else None
    summary.date_max = max(run_dates).isoformat() if len(run_dates) else None
    summary.build_path = str(processed_dir(source) / BUILD_FILE)
    summary.report_path = str(reports_dir() / f"store-build-{source}.md")
    summary.seconds = round(time.monotonic() - started, 2)

    # the build time lives in _build.json only: no wall-clock value ever enters a table
    built = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_text_atomic(
        Path(summary.build_path),
        json.dumps({**asdict(summary), "built_utc": built}, indent=1) + "\n",
    )
    write_text_atomic(Path(summary.report_path), _report(summary))
    return summary
