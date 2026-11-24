"""Capturing a JRA meeting from JRADB through the ingest fetcher (ported from HKTrack's
`sources/jra/official.py`: `_race_list`, `_check`, `live_odds`, `card_tasks`).

Only the two entry menus are fixed (出馬表 and オッズ); every other (path, cname) is read off a page
already fetched. Every request is a FetchTask: POST, form body `cname=<code>`, mutable. Each call
starts at its menu: nothing is remembered between calls. Pages are untrusted data, only parsed.

Change from HKTrack, fixing (d): a page whose date line names another meeting raises WrongMeeting,
and a meeting the menu does not list raises NotListed. HKTrack's `live_odds` swallowed both into
an empty list, so a loop kept writing 0-row captures and its stop rule never fired.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import urlencode

from bs4 import BeautifulSoup

from horsetrack.core.config import load_source
from horsetrack.core.ingest.fetcher import Fetcher
from horsetrack.core.ingest.types import FetchTask, RawPage
from horsetrack.countries.japan.jra import official_pages as op

SOURCE = "jra"
CARD_MENU = op.Link("/JRADB/accessD.html", "pw01dli00/F3")  # 出馬表: this week's meetings
ODDS_MENU = op.Link(op.ODDS_PATH, "pw15oli00/6D")  # オッズ: this week's meetings
ODDS_POOLS = ("WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE")


class WrongMeeting(RuntimeError):
    """A page the site served belongs to another meeting (or race) than the one asked for."""


class NotListed(RuntimeError):
    """The menu does not list the meeting asked for (not published yet, or no racing)."""


@dataclass(frozen=True)
class MeetingCapture:
    rows: list[dict[str, Any]]  # official_pages.odds_rows dicts, all with this snapshot_utc
    snapshot_utc: datetime
    requests: int  # HTTP requests this capture sent


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def race_id(race_date: date, venue: str, number: int) -> str:
    """jra:<YYYY-MM-DD>:<VENUE>:<NN> (docs/specs/odds-log.md)."""
    return f"{SOURCE}:{race_date.isoformat()}:{venue}:{number:02d}"


def task(page_type: str, link: op.Link, year: int) -> FetchTask:
    path, cname = link
    host = load_source("japan", SOURCE).hosts.official.rstrip("/")
    return FetchTask(SOURCE, page_type, cname.split("/")[0], host + path, year, method="POST",
                     body=urlencode({"cname": cname}), mutable=True)  # fmt: skip


def _get(
    fetcher: Fetcher, page_type: str, link: op.Link, year: int
) -> tuple[BeautifulSoup, RawPage]:
    page = fetcher.fetch(task(page_type, link, year))
    return op.soup(page.body), page


def _menu(fetcher: Fetcher, menu_link: op.Link, race_date: date) -> BeautifulSoup:
    """A menu, refused as an error page when it lists no meeting at all."""
    menu, _ = _get(fetcher, "official_menu", menu_link, race_date.year)
    if not op._meeting_links(menu, menu_link.path):
        raise ValueError(f"menu {menu_link.cname} lists no meeting at all (an error page?)")
    return menu


def _check(doc: BeautifulSoup, race_date: date, venue: str, what: str) -> None:
    shown = op.page_meeting(doc)
    if shown is None:
        raise ValueError(f"{what}: not a JRADB page with a date line (an error page?)")
    if shown != (race_date, venue):
        raise WrongMeeting(
            f"{what} is for {shown[0]} {shown[1] or '?'}, asked for {race_date} {venue}: refused"
        )


def _race_list(
    fetcher: Fetcher, menu_link: op.Link, page_type: str, race_date: date, venue: str
) -> BeautifulSoup:
    """The meeting's race list off the menu, its date line checked."""
    menu = _menu(fetcher, menu_link, race_date)
    link = op.meetings_on(menu, race_date, menu_link.path).get(venue)
    if link is None:
        raise NotListed(f"{race_date} {venue} is not on the menu {menu_link.cname}")
    race_list, _ = _get(fetcher, page_type, link, race_date.year)
    _check(race_list, race_date, venue, f"race list {link.cname}")
    return race_list


def discover_venues(fetcher: Fetcher, race_date: date) -> list[str]:
    """Venue codes of every meeting under `race_date`'s heading of the 出馬表 menu (1 request)."""
    return list(op.meetings_on(_menu(fetcher, CARD_MENU, race_date), race_date, CARD_MENU.path))


def read_post_times(fetcher: Fetcher, race_date: date, venue: str) -> dict[int, datetime]:
    """Race number -> post time (aware UTC): the 出馬表 menu, then the card race list
    (2 requests)."""
    race_list = _race_list(fetcher, CARD_MENU, "official_card_racelist", race_date, venue)
    return op.post_times(race_list, race_date)


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
    order: the オッズ menu, the race list, then one request per distinct odds page of each race.
    Raises NotListed, WrongMeeting, or ValueError (an unknown pool, before any request)."""
    wanted = list(dict.fromkeys(pools))
    unknown = [p for p in wanted if p not in ODDS_POOLS]
    if unknown or not wanted:
        raise ValueError(f"no JRA live odds for pool(s) {unknown or wanted}; known: {ODDS_POOLS}")
    snapshot_utc = now()
    start = fetcher.requests_sent
    race_list = _race_list(fetcher, ODDS_MENU, "official_racelist", race_date, venue)
    rows: list[dict[str, Any]] = []
    for number, row in op.race_rows(race_list):
        if races is not None and number not in races:
            continue
        offered = op.pool_links(row)
        on_page: dict[op.Link, list[str]] = {}
        for pool in wanted:
            if pool in offered:
                on_page.setdefault(offered[pool], []).append(pool)
        for link, page_pools in on_page.items():
            doc, page = _get(fetcher, "official_odds", link, race_date.year)
            what = f"odds page {link.cname}"
            _check(doc, race_date, venue, what)
            shown = op.race_number(doc)
            if shown != number:
                raise WrongMeeting(f"{what} is for race {shown}, asked for race {number}: refused")
            rows += op.odds_rows(
                doc,
                race_id(race_date, venue, number),
                page_pools,
                snapshot_utc=snapshot_utc,
                fetched_utc=page.fetched_utc,
            )
    return MeetingCapture(rows, snapshot_utc, fetcher.requests_sent - start)
