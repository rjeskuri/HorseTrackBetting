"""Fetching JMA hourly downloads through `horsetrack.core.ingest` and storing them in the `weather`
table (ticket #89): `horsetrack japan jma-backfill`. No race-day command calls it.

The 過去の気象データ・ダウンロード page (https://www.data.jma.go.jp/risk/obsdl/) posts a form to
`show/table` and gets one CSV back (Shift_JIS). `month_task` builds that post for one station:
hourly 気温, 降水量, 日照時間 and 風向・風速 over one calendar month, with the quality columns and
dates as literals (24時 as 0時 of the next day). The form's names were read from the page's script
(top.bundle.js, 2026-11-25); JMA documents none of them. The first wired fetch (府中, 2024-06, on
2026-11-25) confirmed them: one POST, no session or cookie, returned the CSV. A response without
the CSV header counts as blocked.

The Fetcher (cache-first, rate-limited per host, circuit breaker; the http block of
`config/weather.yaml`) keeps each month under `raw/jma/amedas_hourly/<year>/<station>_<YYYY-MM>`.
JMA serves data up to the day before, so a month that has not ended in JST is refetched
(`mutable`), and a cached copy whose last hourly line is not the month's 24時 (taken before the
month ended) is fetched again; a complete month is never requested twice.

`backfill` fetches every month of a span for the venues' stations and the sunshine stations they
borrow (`stations.yaml` `sunshine_station`), stamps `available_utc` with the owner's lag and
writes the `weather` table of the `jra` store (docs/specs/weather.md).
"""

from __future__ import annotations

import calendar
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import typer

from horsetrack.core.config import load_country_yaml
from horsetrack.core.ingest.fetcher import CircuitOpen, Fetcher, FetchError
from horsetrack.core.ingest.types import STATUS_BLOCKED, STATUS_FETCHED, FetchTask, RawPage
from horsetrack.core.store import build, schemas, tables
from horsetrack.countries.japan.jma import parse

SOURCE = "jma"
PAGE_TYPE = "amedas_hourly"
TABLE_URL = "https://www.data.jma.go.jp/risk/obsdl/show/table"
HOURLY = "9"  # aggrgPeriod 時別値
# 気温, 降水量 (前1時間), 日照時間 (前1時間), 風向・風速
ELEMENTS = (("201", ""), ("101", ""), ("401", ""), ("301", ""))
ENCODING = "cp932"  # Shift_JIS as JMA sends it
# the download time on a file's first line, JST: ダウンロードした時刻：2026/11/25 17:02:51
DOWNLOADED = re.compile(
    r"ダウンロードした時刻：(\d{4})/(\d{1,2})/(\d{1,2}) (\d{1,2}):(\d{2}):(\d{2})"
)
STORE = "jra"  # the store whose venues the stations serve: processed/jra/weather.parquet
TABLE = "weather"


def make_fetcher(**kwargs: Any) -> Fetcher:
    """The ingest Fetcher with weather.yaml's per-host rate and circuit breaker."""
    return Fetcher(SOURCE, http=load_country_yaml(parse.COUNTRY, parse.CONFIG)["http"], **kwargs)


def stations(venues: Sequence[str] | None = None) -> dict[str, dict]:
    """station_id -> stations.yaml entry: each venue's station, then the sunshine station it
    borrows, each once, in venue order (default: every JRA venue). KeyError for another code."""
    out: dict[str, dict] = {}
    for venue in venues or list(parse._stations()):
        entry = parse.station(venue)
        for each in (entry, entry.get("sunshine_station")):
            if each:
                out.setdefault(str(each["station_id"]), dict(each))
    return out


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))  # as JSON.stringify


def form(entry: dict, year: int, month: int) -> dict[str, str]:
    """The download form for one station (a stations.yaml entry) over one calendar month (JST).
    The station is named kind + block_no (`s47412`, `a1133`), the form's own pattern."""
    last = calendar.monthrange(year, month)[1]
    return {
        "stationNumList": _json([f"{entry['kind']}{entry['station_id']}"]),
        "aggrgPeriod": HOURLY,
        "elementNumList": _json([list(e) for e in ELEMENTS]),
        "interAnnualType": "1",  # one continuous period
        "ymdList": _json([str(year), str(year), str(month), str(month), "1", str(last)]),
        "option": _json({"list": ["obs"], "nyear": 0}),
        "downloadFlag": "true",
        "rmkFlag": "1",  # 品質情報, 現象なし情報, 均質番号
        "disconnectFlag": "1",  # every value, across homogeneity breaks
        "youbiFlag": "0",
        "fukenFlag": "0",
        "kijiFlag": "0",
        "csvFlag": "1",  # numbers, not symbols
        "jikantaiFlag": "0",
        "jikantaiList": _json([1, 24]),
        "ymdLiteral": "1",
    }


def month_task(entry: dict, year: int, month: int, *, today: date | None = None) -> FetchTask:
    """The fetch of one station's month; refetched (`mutable`) until the month has ended in JST."""
    today = today or datetime.now(parse.JST).date()
    if date(year, month, 1) > today:
        raise ValueError(f"{year}-{month:02d} has not started: JMA has no data for it yet")
    last = date(year, month, calendar.monthrange(year, month)[1])
    key = f"{entry['station_id']}_{year:04d}-{month:02d}"
    body = urlencode(form(entry, year, month))
    return FetchTask(SOURCE, PAGE_TYPE, key, TABLE_URL, year, method="POST", body=body,
                     mutable=last >= today)  # fmt: skip


def classify(body: bytes) -> str:
    """A download is a CSV with the 年月日時 header row; an error or block page is not."""
    try:
        text = body.decode(ENCODING)
    except UnicodeDecodeError:
        return STATUS_BLOCKED
    lines = text.splitlines()
    return STATUS_FETCHED if any(line.startswith("年月日時,") for line in lines) else STATUS_BLOCKED


def complete(body: bytes, year: int, month: int) -> bool:
    """Whether a download holds the whole month: its last hourly line is the month's 24時."""
    end = datetime(year, month, calendar.monthrange(year, month)[1], tzinfo=parse.JST)
    try:
        lines = [line for line in body.decode(ENCODING).splitlines() if line.strip()]
        last = parse._obs_utc(lines[-1].split(",", 1)[0]) if lines else None
    except (UnicodeDecodeError, ValueError):
        return False
    return last is not None and last >= end + timedelta(days=1)


def fetch_month(
    fetcher: Fetcher, entry: dict, year: int, month: int, *, today: date | None = None
) -> RawPage:
    """One station's month, cache-first; a cached copy that stops short of the month's end is
    fetched again. A body that is not a download is moved aside (quarantined) and raised as a
    FetchError, so the next attempt goes back to JMA."""
    task = month_task(entry, year, month, today=today)
    page = fetcher.fetch(task)
    if page.meta.get("from_cache") and not complete(page.body, year, month):
        page = fetcher.fetch(task, force=True)
    if classify(page.body) != STATUS_FETCHED:
        fetcher.quarantine(task)
        raise FetchError(f"not a JMA download for {PAGE_TYPE}/{task.key}")
    return page


def parse_page(page: RawPage, station_id: str) -> pd.DataFrame:
    """The page's hourly rows (`parse.parse_amedas_csv`) plus `ingested_utc`, the fetch time: the
    download time JMA prints on the first line (JST, to the second), so a rerun from the cache
    writes the same value; the page's own fetch time where that line is missing. No
    `available_utc`: `parse.with_publication_time` adds it with the owner's lag."""
    text = page.body.decode(ENCODING)
    stamp = DOWNLOADED.match(text)
    if stamp:
        fetched = datetime(*map(int, stamp.groups()), tzinfo=parse.JST)
    elif page.fetched_utc is not None:
        fetched = page.fetched_utc
    else:
        raise ValueError(f"{page.page_type}/{page.key} has no fetch time")
    frame = parse.parse_amedas_csv(text, station_id=station_id)
    frame["ingested_utc"] = pd.Timestamp(fetched).tz_convert("UTC").as_unit("us")
    return frame


# --------------------------------------------------------------------------- the backfill


@dataclass
class BackfillSummary:
    planned: int = 0  # station-months in the span
    stored: int = 0  # station-months parsed into the table
    requested: int = 0  # requests sent to JMA
    from_cache: int = 0
    failed: list[str] = field(default_factory=list)
    table_rows: int | None = None  # rows in the weather table after the write
    stopped: str | None = None  # why the run stopped early


def months(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Every (year, month) from start to end, both included."""
    first, last = start[0] * 12 + start[1] - 1, end[0] * 12 + end[1] - 1
    return [(i // 12, i % 12 + 1) for i in range(first, last + 1)]


def write_weather(frames: Sequence[pd.DataFrame], lag) -> int:
    """Merge parsed months into the `weather` table through the store's writer: a row of the
    same (station_id, obs_utc) is replaced, every other row kept, `available_utc` = obs_utc +
    the configured lag on every row. Returns the table's row count."""
    spec = schemas.TABLES[TABLE]
    path = tables.table_path(STORE, TABLE)
    parts = [tables.read_table(STORE, TABLE)] if path.exists() else []
    merged = pd.concat([*parts, *frames], ignore_index=True)
    merged["jurisdiction"] = STORE
    merged = (
        merged.drop(columns="available_utc", errors="ignore")
        .drop_duplicates(subset=["station_id", "obs_utc"], keep="last")
        .sort_values(list(spec.sort_by), kind="stable")
        .reset_index(drop=True)
    )
    merged = parse.with_publication_time(merged, lag)[list(spec.columns)]
    build._write_tables({TABLE: merged}, path.parent)
    return len(merged)


def backfill(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    venues: Sequence[str] | None = None,
    fetcher: Fetcher | None = None,
    echo: Callable[[str], None] = print,
    today: date | None = None,
) -> BackfillSummary:
    """Fetch every month of [start, end] (the current JST month at most) for the venues' stations
    and their sunshine stations, then merge what was fetched into the weather table. Stops early,
    still writing what it has, when the circuit opens (403 or 429, or failed requests in a row),
    after as many failed months in a row, or on Ctrl-C."""
    lag = parse.publication_lag()  # before any request: refuses while the owner's lag is unset
    today = today or datetime.now(parse.JST).date()
    span = months(start, min(end, (today.year, today.month)))
    if not span:
        raise ValueError(f"no month from {start} to {end} has started")
    chosen = stations(venues)
    plan = [(sid, entry, ym) for sid, entry in chosen.items() for ym in span]
    breaker = load_country_yaml(parse.COUNTRY, parse.CONFIG)["http"]["circuit_breaker"]
    limit = int(breaker["max_consecutive_failures"])
    summary = BackfillSummary(planned=len(plan))
    echo(f"jma backfill: {len(plan)} station-months, {span[0][0]}-{span[0][1]:02d} to "
         f"{span[-1][0]}-{span[-1][1]:02d}, stations {', '.join(chosen)}")  # fmt: skip
    own = fetcher is None
    fetcher = fetcher or make_fetcher()
    frames: list[pd.DataFrame] = []
    streak = 0
    try:
        for i, (sid, entry, (year, month)) in enumerate(plan, 1):
            sent = fetcher.requests_sent
            label = f"[{i}/{len(plan)}] {entry['name']} {sid} {year}-{month:02d}"
            try:
                frame = parse_page(fetch_month(fetcher, entry, year, month, today=today), sid)
            except CircuitOpen as exc:
                summary.stopped = f"circuit open: {exc}"
                break
            except (FetchError, ValueError) as exc:
                summary.failed.append(f"{sid} {year}-{month:02d}: {exc}")
                echo(f"{label}: FAILED {exc}")
                streak += 1
                if streak >= limit:
                    summary.stopped = f"{streak} failed months in a row"
                    break
                continue
            finally:
                summary.requested += fetcher.requests_sent - sent
            frames.append(frame)
            summary.stored += 1
            if fetcher.requests_sent > sent:
                streak = 0  # a cache hit says nothing about JMA
                echo(f"{label}: {len(frame)} rows")
            else:
                summary.from_cache += 1
    except KeyboardInterrupt:
        summary.stopped = "interrupted"
    finally:
        if own:
            fetcher.close()
    if frames:
        summary.table_rows = write_weather(frames, lag)
    return summary


def _month(text: str) -> tuple[int, int]:
    try:
        value = datetime.strptime(text, "%Y-%m")
    except ValueError:
        raise typer.BadParameter(f"not a month YYYY-MM: {text!r}") from None
    return value.year, value.month


def jma_backfill_command(
    start: str | None = typer.Option(
        None, "--start", help="First month, YYYY-MM; default: the price era's (2014-06, D10)."
    ),
    end: str | None = typer.Option(None, "--end", help="Last month, YYYY-MM; default: this one."),
    venues: str | None = typer.Option(
        None, "--venues", help="JRA venue codes, e.g. TOK,NII; default: all ten."
    ),
) -> None:
    """Fetch JMA's hourly observations for the JRA venues' stations into the weather table
    (cache-first, one request per 5 s; a stop on 403 or 429). Safe to rerun: it resumes."""
    from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START

    today = datetime.now(parse.JST).date()
    first = _month(start) if start else (PRICE_ERA_START.year, PRICE_ERA_START.month)
    last = _month(end) if end else (today.year, today.month)
    codes = [v.strip() for v in (venues or "").split(",") if v.strip()] or None
    summary = backfill(first, last, venues=codes, echo=typer.echo, today=today)
    table = "not written" if summary.table_rows is None else f"{summary.table_rows} rows"
    typer.echo(f"jma backfill: {summary.stored} of {summary.planned} station-months stored "
               f"({summary.requested} requests, {summary.from_cache} from cache), "
               f"{len(summary.failed)} failed; weather table {table}")  # fmt: skip
    for line in summary.failed[:20]:
        typer.echo(f"  failed {line}")
    if summary.stopped:
        typer.echo(f"stopped early: {summary.stopped}; rerun the same command to resume")
        raise typer.Exit(1)
