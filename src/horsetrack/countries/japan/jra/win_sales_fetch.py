"""Fetching a race day's JRADB 単勝・複勝 pages for their Win sales (ticket #278), through the
`core.ingest` fetcher (rate-limited, circuit breaker), so the store build fills `win_sales`.

The オッズ menu, each meeting's race list, then each race's 単勝・複勝 page: every request a
FetchTask (`jra.capture.task`: POST, mutable), every (path, cname) read off a page already
fetched. Each
単勝・複勝 page is kept under page type `win_sales` with a key of its own per fetch,
`<race date>_<VENUE>_<RR>_<fetch time UTC>Z`, and recorded in the JRA manifest, so every
publication fetched stays (a live figure, later the final one) and the build reads each. A
publication fetched twice is one row of the table (the latest fetch). Forward only: JRADB shows the
current week. Pages are untrusted data, only parsed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import typer

from horsetrack.core.config import load_source
from horsetrack.core.ingest.fetcher import Fetcher
from horsetrack.core.ingest.queue import FetchQueue
from horsetrack.core.ingest.types import STATUS_FETCHED
from horsetrack.core.paths import manifest_path
from horsetrack.countries.japan.jra import capture
from horsetrack.countries.japan.jra import official_pages as op
from horsetrack.countries.japan.jra.adapter import ADAPTER
from horsetrack.countries.japan.jra.win_sales import PAGE_TYPE

SOURCE = "jra"
RACE_LIST_PAGE_TYPE = "official_racelist"  # as `jra.capture` keeps it


def page_key(race_date: date, venue: str, race_no: int, fetched: datetime) -> str:
    """<race date>_<VENUE>_<RR>_<YYYYMMDDTHHMMSS>Z: one cache file per race and fetch."""
    return f"{race_date.isoformat()}_{venue}_{race_no:02d}_{fetched:%Y%m%dT%H%M%S}Z"


def fetch_win_sales(
    race_date: date,
    *,
    http: dict[str, Any],
    fetcher: Fetcher | None = None,
    venues: Sequence[str] | None = None,
    now: Callable[[], datetime] = capture.utc_now,
) -> list[str]:
    """Fetch the 単勝・複勝 page of every race of every meeting of `race_date` on the オッズ menu
    (or of `venues` only) and record each in the manifest as page type `win_sales` (status
    `fetched`, or `blocked` for a page that names no race). Returns the keys recorded `fetched`.
    Raises capture.NotListed for a venue the menu does not list (before any race list is
    fetched), capture.WrongMeeting for a race list of another meeting, and the fetcher's errors."""
    own = fetcher is None
    fetcher = fetcher if fetcher is not None else Fetcher(SOURCE, http=http)
    try:
        menu = capture._menu(fetcher, capture.ODDS_MENU, race_date)
        listed = op.meetings_on(menu, race_date, op.ODDS_PATH)
        wanted = list(listed) if venues is None else list(dict.fromkeys(venues))
        missing = [venue for venue in wanted if venue not in listed]
        if missing:
            raise capture.NotListed(f"{race_date} {', '.join(missing)}: not on the オッズ menu")
        keys: list[str] = []
        with FetchQueue(manifest_path(SOURCE)) as queue:
            for venue in wanted:
                keys += _fetch_meeting(fetcher, queue, race_date, venue, listed[venue], now)
        return keys
    finally:
        if own:
            fetcher.close()


def _fetch_meeting(
    fetcher: Fetcher,
    queue: FetchQueue,
    race_date: date,
    venue: str,
    link: op.Link,
    now: Callable[[], datetime],
) -> list[str]:
    race_list, _ = capture._get(fetcher, RACE_LIST_PAGE_TYPE, link, race_date.year)
    capture._check(race_list, race_date, venue, f"race list {link.cname}")
    keys = []
    for number, row in op.race_rows(race_list):
        win_link = op.pool_links(row).get("WIN")
        if win_link is None:  # Win not sold on this race
            continue
        key = page_key(race_date, venue, number, now())
        task = replace(capture.task(PAGE_TYPE, win_link, race_date.year), key=key)
        page = fetcher.fetch(task)
        status = ADAPTER.classify(PAGE_TYPE, page.body)
        error = None if status == STATUS_FETCHED else "no race named on the page (an error page?)"
        queue.mark(task, status, http_status=page.meta.get("http_status"),
                   final_url=page.meta.get("final_url"), sha256=page.meta.get("sha256"),
                   n_bytes=len(page.body), error=error)  # fmt: skip
        if status == STATUS_FETCHED:
            keys.append(key)
    return keys


def fetch_win_sales_command(
    race_date: str | None = typer.Option(
        None, "--date", help="Race date (YYYY-MM-DD); default: today in Asia/Tokyo."
    ),
    venues: str | None = typer.Option(
        None, "--venues", help="JRA venue codes, e.g. TOK,KYO; default: every meeting of the day."
    ),
) -> None:
    """Fetch the day's JRADB 単勝・複勝 pages into the raw cache and the manifest; `store build`
    then fills the win_sales table."""
    from horsetrack.core.ingest.cli import _fetcher_http, source_config

    tz = ZoneInfo(load_source("japan", SOURCE).timezone)
    day = date.fromisoformat(race_date) if race_date else datetime.now(tz).date()
    wanted = [v.strip() for v in (venues or "").split(",") if v.strip()] or None
    keys = fetch_win_sales(day, http=_fetcher_http(source_config(SOURCE)["http"]), venues=wanted)
    typer.echo(f"win_sales pages for {day}: {len(keys)} fetched")
