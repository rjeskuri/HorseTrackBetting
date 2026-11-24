"""Capturing a NAR meeting from nar.netkeiba.com through the ingest fetcher (ticket #38), with the
signatures and refusals of `jra/capture.py`, so `logger.schedule.run_day` drives it as it drives
JRA's.

Every request is a FetchTask: GET, source "nar", mutable, to nar.netkeiba.com (nar.yaml
`hosts.race`, at nar.yaml's rate). Each call starts at the day's race list
(`/top/race_list_sub.html?kaisai_date=YYYYMMDD`); the odds pages are
`/odds/?race_id=<12 digits>&type=<t>`, the race id read off the race list: `b1` holds WIN and
PLACE, `b4` QUINELLA, `b5` QUINELLA_PLACE. Pages are untrusted data, only parsed (`nar/pages.py`).

Refusals, as JRA's: a meeting the race list does not list raises NotListed; an odds page whose own
race is not the race asked for raises WrongMeeting. A page's own race is what its race navigation
names (`_OWN_RACE`): the active race-number entry, the active venue tab and the page menu's links,
never every `race_id=` on the page (a real page links to every race of its meeting and to the
day's other venues). A page that names no race of its own is a ValueError. A race whose
odds page shows netkeiba's 予想オッズ (`pages.NotOnSale`: sales not open yet) gives no rows and
its other pages are not asked for; the capture goes on with the other races.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from bs4 import BeautifulSoup

from horsetrack.core.config import load_country_yaml
from horsetrack.core.ingest.fetcher import Fetcher
from horsetrack.core.ingest.types import FetchTask
from horsetrack.countries.japan.jra import capture as jra_capture
from horsetrack.countries.japan.jra.capture import MeetingCapture
from horsetrack.countries.japan.nar import pages

log = logging.getLogger(__name__)

SOURCE = pages.SOURCE
RACE_LIST_PATH = "/top/race_list_sub.html"
ODDS_PATH = "/odds/"
# pool -> the odds page type that prints it
PAGE_OF = {"WIN": "b1", "PLACE": "b1", "QUINELLA": "b4", "QUINELLA_PLACE": "b5"}
ODDS_POOLS = tuple(PAGE_OF)


class WrongMeeting(jra_capture.WrongMeeting):
    """An odds page the site served belongs to another race than the one asked for."""


class NotListed(jra_capture.NotListed):
    """The day's race list has no meeting of the venue asked for."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def host() -> str:
    """https://nar.netkeiba.com, from nar.yaml."""
    return str(load_country_yaml("japan", SOURCE)["hosts"]["race"]).rstrip("/")


def _task(page_type: str, path: str, query: dict[str, str], key: str, year: int) -> FetchTask:
    url = f"{host()}{path}?{urlencode(query)}"
    return FetchTask(SOURCE, page_type, key, url, year, method="GET", mutable=True)


def _race_list(fetcher: Fetcher, race_date: date) -> BeautifulSoup:
    day = f"{race_date:%Y%m%d}"
    page = fetcher.fetch(_task("race_list", RACE_LIST_PATH, {"kaisai_date": day}, day,
                               race_date.year))  # fmt: skip
    return pages.soup(page.body)


def _meeting(race_list: BeautifulSoup, race_date: date, venue: str) -> pages.Meeting:
    meeting = pages.meetings_on(race_list, race_date).get(venue)
    if meeting is None:
        raise NotListed(f"{race_date} {venue} is not on the NAR race list")
    return meeting


# The links where an odds page names its own race: the active race number, the active venue tab,
# the page menu (出馬表, オッズ, ...).
_OWN_RACE = (
    "div.RaceNumWrap ul.fc li.Active a[href]",
    "div.RaceKaisaiWrap ul.Col li.Active a[href]",
    'ul.RaceMainMenu li[class*="race_navi_"] a[href]',
)


def _page_race_ids(doc: BeautifulSoup) -> set[str]:
    """The `race_id=` values of the page's own-race links (`_OWN_RACE`)."""
    out: set[str] = set()
    for a in doc.select(", ".join(_OWN_RACE)):
        out.update(parse_qs(urlsplit(str(a["href"])).query).get("race_id", []))
    return out


def _check_race(doc: BeautifulSoup, race_id: str, what: str) -> None:
    shown = _page_race_ids(doc)
    if not shown:
        raise ValueError(f"{what}: the page names no race of its own (an error page?)")
    if shown != {race_id}:
        raise WrongMeeting(f"{what} is for race {','.join(sorted(shown))}, asked for {race_id}")


def discover_venues(fetcher: Fetcher, race_date: date) -> list[str]:
    """nar.yaml codes of every meeting of `race_date` on the day's race list (1 request)."""
    return list(pages.meetings_on(_race_list(fetcher, race_date), race_date))


def read_post_times(fetcher: Fetcher, race_date: date, venue: str) -> dict[int, datetime]:
    """Race number -> post time (aware UTC) of the meeting, from the race list (1 request)."""
    race_list = _race_list(fetcher, race_date)
    _meeting(race_list, race_date, venue)
    return pages.post_times(race_list, race_date, venue)


def capture_meeting(
    fetcher: Fetcher,
    race_date: date,
    venue: str,
    pools: Sequence[str],
    *,
    races: Sequence[int] | None = None,
    now: Callable[[], datetime] = utc_now,
) -> MeetingCapture:
    """The odds rows of `pools` for every race of the meeting (or `races` only), in race-list
    order: the race list, then one request per distinct odds page of each race. Raises NotListed,
    WrongMeeting, or ValueError (an unknown pool, before any request)."""
    wanted = list(dict.fromkeys(pools))
    unknown = [p for p in wanted if p not in PAGE_OF]
    if unknown or not wanted:
        raise ValueError(f"no NAR live odds for pool(s) {unknown or wanted}; known: {ODDS_POOLS}")
    on_page: dict[str, list[str]] = {}
    for pool in wanted:
        on_page.setdefault(PAGE_OF[pool], []).append(pool)
    snapshot_utc = now()
    start = fetcher.requests_sent
    meeting = _meeting(_race_list(fetcher, race_date), race_date, venue)
    rows: list[dict[str, Any]] = []
    for number, race_id in meeting.races.items():
        if races is not None and number not in races:
            continue
        for page_type in sorted(on_page):
            query = {"race_id": race_id, "type": page_type}
            page = fetcher.fetch(_task(f"odds_{page_type}", ODDS_PATH, query, race_id,
                                       race_date.year))  # fmt: skip
            doc = pages.soup(page.body)
            what = f"odds page {page_type} of {race_id}"
            _check_race(doc, race_id, what)
            try:
                rows += pages.odds_rows(doc, race_id, on_page[page_type], snapshot_utc=snapshot_utc,
                                        fetched_utc=page.fetched_utc or snapshot_utc)  # fmt: skip
            except pages.NotOnSale:
                log.info("%s %s race %d: not on sale yet", race_date, venue, number)
                break  # its other pages show the forecast too
    return MeetingCapture(rows, snapshot_utc, fetcher.requests_sent - start)
