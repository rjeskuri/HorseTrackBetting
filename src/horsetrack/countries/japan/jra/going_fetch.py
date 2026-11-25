"""Fetching the JRA going pages (馬場情報) through the ingest runner (ticket #253), so the store
build can fill `races.rail_setting` from them (`going.attach_rail_setting`).

https://www.jra.go.jp/keiba/baba/ shows the going of the meeting now on (or just past), one venue
per tab: index.html (the first venue, also the folder URL), index2.html, ... Only the latest race
day is on the site, so a page is kept under `going/<race date>_<VENUE>`: the day it names and its
venue, read off the page itself, and a later fetch of the same day replaces it (a going page
changes during a meeting). Two page types:

- `going_menu`, key `index` or `index2` ...: a venue tab as fetched. Its follow-ups are the going
  page for the day and venue it names (the same URL again, under its own key) and the other tabs
  it links. Never read by the store build.
- `going`: the page the store build reads. Pages are Shift_JIS (cp932) and untrusted data, only
  parsed; the runner records them in the manifest, so the build finds them there.

`fetch_going` runs these two page types and nothing else off the manifest's queue: cache-first
and rate-limited by `core.ingest`, the host's circuit breaker stops it.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from horsetrack.core.config import load_source
from horsetrack.core.ingest.types import (
    STATUS_BLOCKED,
    STATUS_FETCHED,
    FetchTask,
    ParsedPage,
    RawPage,
)
from horsetrack.countries.japan.jra.official_pages import soup

# The adapter imports this module and parse workers import the adapter, so what a worker does not
# need (pandas through `going`, the runner, typer) is imported inside the functions that do.

SOURCE = "jra"
GOING_PAGE_TYPE = "going"
MENU_PAGE_TYPE = "going_menu"
PAGE_TYPES = (MENU_PAGE_TYPE, GOING_PAGE_TYPE)
FIRST_SLOT = "index"
BABA_PATH = "/keiba/baba/"
_SLOT = re.compile(r"index[0-9]*")
_TABS = ".kaisai_tab .nav a[href]"


def slot_url(slot: str) -> str:
    """The URL of a venue tab; ValueError for anything that is not `index` / `index<N>`."""
    if not _SLOT.fullmatch(slot):
        raise ValueError(f"not a going page tab: {slot!r}")
    host = load_source("japan", SOURCE).hosts.official.rstrip("/")
    return host + BABA_PATH + ("" if slot == FIRST_SLOT else f"{slot}.html")


def menu_task(slot: str, year: int) -> FetchTask:
    return FetchTask(SOURCE, MENU_PAGE_TYPE, slot, slot_url(slot), year, mutable=True)


def classify(body: bytes) -> str:
    """A going page names its race day and venue; a block or error page does not."""
    from horsetrack.countries.japan.jra import going

    try:
        going.meeting(body)
    except ValueError:
        return STATUS_BLOCKED
    return STATUS_FETCHED


def parse_menu(page: RawPage) -> ParsedPage:
    """The tab's race day and venue and the tabs it links (their slots), as `_meta`."""
    from horsetrack.countries.japan.jra import going

    race_date, venue = going.meeting(page.body)
    tabs = []
    for link in soup(page.body).select(_TABS):
        slot = link["href"].rsplit("/", 1)[-1].removesuffix(".html")
        if _SLOT.fullmatch(slot) and slot not in tabs:
            tabs.append(slot)
    meta = {"page_type": MENU_PAGE_TYPE, "slot": page.key, "race_date": race_date,
            "venue": venue, "tabs": tabs}  # fmt: skip
    return {"_meta": [meta]}


def followups(parsed: ParsedPage) -> list[FetchTask]:
    meta = (parsed.get("_meta") or [{}])[0]
    if meta.get("page_type") != MENU_PAGE_TYPE:
        return []
    race_date, slot = meta["race_date"], meta["slot"]
    key = f"{race_date.isoformat()}_{meta['venue']}"
    tasks = [FetchTask(SOURCE, GOING_PAGE_TYPE, key, slot_url(slot), race_date.year, mutable=True)]
    tasks += [menu_task(tab, race_date.year) for tab in meta["tabs"] if tab != slot]
    return tasks


def fetch_going(*, year: int, http: dict[str, Any], max_requests: int = 8, fetcher: Any = None):
    """Queue the first tab and work the going page types off the queue: the tab, its going page
    and the other tabs (2 requests per venue on now). Returns the runner's RunSummary."""
    from horsetrack.core.ingest import runner
    from horsetrack.core.ingest.queue import FetchQueue
    from horsetrack.core.paths import manifest_path

    with FetchQueue(manifest_path(SOURCE)) as queue:
        queue.enqueue([menu_task(FIRST_SLOT, year)])
        return runner.run_backfill(SOURCE, max_requests=max_requests, page_types=list(PAGE_TYPES),
                                   queue=queue, http=http, fetcher=fetcher)  # fmt: skip


def fetch_going_command(max_requests: int = 8) -> None:
    """Fetch the going pages (馬場情報) on the JRA site now into the raw cache and the manifest
    (`--max-requests`: stop after N requests); `store build` then fills races.rail_setting."""
    import typer

    from horsetrack.core.ingest.cli import _fetcher_http, source_config

    summary = fetch_going(
        year=datetime.now(ZoneInfo(load_source("japan", SOURCE).timezone)).year,
        max_requests=max_requests,
        http=_fetcher_http(source_config(SOURCE)["http"]),
    )
    typer.echo(f"going pages: network {summary.requested}, fetched {summary.fetched},"
               f" failed {summary.failed}, stopped ({summary.stopped_reason})")  # fmt: skip
    typer.echo(f"log {summary.log_path}")
