"""Per-race Win sales (単勝 発売票数) off a JRADB 単勝・複勝 odds page, with their publication time
(ticket #278, for E4's low-turnover segment). Pure: no network; `jra.win_sales_fetch` fetches the
pages through `horsetrack.core.ingest` and the store build reads them into `win_sales`.

The publication time follows the owner's answer on #278 (rule A, 2026-11-25): a live page's own
clock (`12時10分現在`), dated as `official_pages.source_update_utc` does; a page without a clock
(最終オッズ) shows the final figure, published at the post time the page prints (発走時刻, Tokyo,
race date) plus 30 minutes, or at `fetched_utc` when it prints no post time. Pages are cp932 and
untrusted data, only parsed.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from bs4 import BeautifulSoup

from horsetrack.core.config import load_source
from horsetrack.countries.japan.jra.netkeiba.common import store_race_id
from horsetrack.countries.japan.jra.official_pages import (
    clean,
    local_clock,
    page_meeting,
    race_number,
    soup,
    source_update_utc,
)

PAGE_TYPE = "win_sales"  # the manifest page type of a 単勝・複勝 page kept for its Win sales
PARSER_VERSION = "1"
FINAL = "最終オッズ"
POST_TO_FINAL = timedelta(minutes=30)  # a final figure is published at its post + 30 min (#278)
WIN = "WIN"

_REFRESH_TIME = ".refresh_line .time"
_POST_TIME = ".race_header .date_line .time"
_POST_LABEL = "発走時刻"
_VOTES_ROW = "#votes tr"


def parse_win_sales(body: bytes, *, fetched_utc: datetime) -> dict[str, Any] | None:
    """The race's Win sales and their publication time (the module docstring); None when the
    page's 発売票数 block has no whole-number 単勝 figure. ValueError for a naive `fetched_utc` and
    for a page that does not say which JRA race it is for."""
    if fetched_utc.tzinfo is None or fetched_utc.utcoffset() is None:
        raise ValueError(f"fetched_utc must be timezone-aware, got {fetched_utc!r}")
    fetched = fetched_utc.astimezone(timezone.utc)
    doc = soup(body)
    race_date, venue, race_no = page_race(doc)
    tickets = _win_tickets(doc)
    if tickets is None:
        return None
    node = doc.select_one(_REFRESH_TIME)
    state = clean(node.get_text(" ")) if node is not None else ""
    published = source_update_utc(state, fetched)
    if published is None:  # no clock: the final figure, from post + 30 min
        post = _post_utc(doc, race_date)
        published = fetched if post is None else post + POST_TO_FINAL
    return {
        "race_id": store_race_id(race_date, venue, race_no),
        "race_date": race_date,
        "venue": venue,
        "race_no": race_no,
        "win_tickets": tickets,
        "win_sales_yen": tickets * load_source("japan", "jra").unit_stake,
        "is_final": FINAL in state,
        "published_utc": published,
        "fetched_utc": fetched,
    }


def page_race(doc: BeautifulSoup) -> tuple[date, str, int]:
    """(race date, venue code, race number) the page says it is for; ValueError when it does not
    say (no date line, a venue not in jra.yaml, no race number: an error page)."""
    shown = page_meeting(doc)
    if shown is None:
        raise ValueError("not a JRADB race page: no date line (an error page?)")
    race_date, venue = shown
    if venue is None:
        raise ValueError(f"the page's meeting on {race_date} is not a JRA venue (jra.yaml)")
    race_no = race_number(doc)
    if race_no is None:
        raise ValueError(f"the page for {race_date} {venue} names no race number")
    return race_date, venue, race_no


def _win_tickets(doc: BeautifulSoup) -> int | None:
    """The 単勝 row of the 発売票数 block (`#votes`), comma thousands; None when there is none or
    its cell is not a whole number. A 単勝 row anywhere else on the page is never read."""
    config = load_source("japan", "jra")
    for tr in doc.select(_VOTES_ROW):
        pool = config.pool_by_label(clean(tr.th.get_text())) if tr.th is not None else None
        if pool is None or pool.code != WIN:
            continue
        text = clean(tr.td.get_text()).replace(",", "") if tr.td is not None else ""
        return int(text) if text.isascii() and text.isdigit() else None
    return None


def _post_utc(doc: BeautifulSoup, race_date: date) -> datetime | None:
    """The post time the race header prints (発走時刻：15時40分) on the race date, Tokyo, as UTC."""
    for node in doc.select(_POST_TIME):
        text = clean(node.get_text(" "))
        if _POST_LABEL in text:
            local = local_clock(race_date, text.split(_POST_LABEL, 1)[1])
            if local is not None:
                return local.astimezone(timezone.utc)
    return None
