"""Capturing a JRA meeting from JRADB through the ingest fetcher (ported from HKTrack's
`sources/jra/official.py`: `_race_list`, `_check`, `live_odds`, `card_tasks`).

Only the entry menus are fixed (出馬表, オッズ and レース結果); every other (path, cname) is
read off a page already fetched. Every request is a FetchTask: POST, form body `cname=<code>`,
mutable. Each call starts at its menu: nothing is remembered between calls. Pages are untrusted
data, only parsed.

Change from HKTrack, fixing (d): a page whose date line names another meeting raises WrongMeeting,
and a meeting the menu does not list raises NotListed. HKTrack's `live_odds` swallowed both into
an empty list, so a loop kept writing 0-row captures and its stop rule never fired.
"""

from __future__ import annotations

import gzip
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from bs4 import BeautifulSoup

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml, load_source
from horsetrack.core.ingest.fetcher import Fetcher, FetchError
from horsetrack.core.ingest.types import FetchTask, RawPage
from horsetrack.countries.japan.jra import exotic_pages as ep
from horsetrack.countries.japan.jra import official_pages as op

log = logging.getLogger(__name__)

SOURCE = "jra"
RAW_PAGES = ("all", "none")
CARD_MENU = op.Link("/JRADB/accessD.html", "pw01dli00/F3")  # 出馬表: this week's meetings
ODDS_MENU = op.Link(op.ODDS_PATH, "pw15oli00/6D")  # オッズ: this week's meetings
RESULTS_MENU = op.Link(op.RESULTS_PATH, "pw01sli00/AF")  # レース結果: this week's meetings (#225)
# Tier 1, every round: the four of #23 and, since #358, 枠連 on a page of its own
ODDS_POOLS = ("WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE", "BRACKET_QUINELLA")
TIER2_POOLS = ("FORECAST", "TRIO", "TIERCE")  # ticket #130 (枠連 moved to tier 1 in #358)
_BOARD_POOLS = ("BRACKET_QUINELLA", *TIER2_POOLS)  # pages read by exotic_pages, a pool each


class WrongMeeting(RuntimeError):
    """A page the site served belongs to another meeting (or race) than the one asked for."""


class NotListed(RuntimeError):
    """The menu does not list the meeting asked for (not published yet, or no racing)."""


@dataclass(frozen=True)
class MeetingCapture:
    rows: list[dict[str, Any]]  # official_pages.odds_rows dicts, all with this snapshot_utc
    snapshot_utc: datetime
    requests: int  # HTTP requests this capture sent


@dataclass
class Win5:
    """The day's WIN5 legs (race_ids) once a tier-2 capture read the WIN5 page; None before."""

    legs: list[str] | None = None


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
    _verdict(op.page_meeting(doc), race_date, venue, what)


def _verdict(shown: tuple[date, str | None] | None, race_date: date, venue: str, what: str) -> None:
    if shown is None:
        raise ValueError(f"{what}: not a JRADB page with a date line (an error page?)")
    if shown != (race_date, venue):
        raise WrongMeeting(
            f"{what} is for {shown[0]} {shown[1] or '?'}, asked for {race_date} {venue}: refused"
        )


def _check_race(shown: int | None, race_no: int, what: str) -> None:
    if shown != race_no:
        raise WrongMeeting(f"{what} is for race {shown}, asked for race {race_no}: refused")


def _race_list(
    fetcher: Fetcher, menu_link: op.Link, page_type: str, race_date: date, venue: str
) -> tuple[BeautifulSoup, BeautifulSoup]:
    """The menu, and the meeting's race list off it, its date line checked."""
    menu = _menu(fetcher, menu_link, race_date)
    link = op.meetings_on(menu, race_date, menu_link.path).get(venue)
    if link is None:
        raise NotListed(f"{race_date} {venue} is not on the menu {menu_link.cname}")
    race_list, _ = _get(fetcher, page_type, link, race_date.year)
    _check(race_list, race_date, venue, f"race list {link.cname}")
    return menu, race_list


def sales_open(fetcher: Fetcher, race_date: date) -> bool:
    """Whether the オッズ menu lists a meeting under `race_date`'s heading (ticket #358): the race
    date's sales are open, the day-before sales from the evening before (1 request). A menu that
    lists no meeting at all raises ValueError (an error page?), as for the other menus."""
    return bool(op.meetings_on(_menu(fetcher, ODDS_MENU, race_date), race_date, ODDS_MENU.path))


def discover_venues(fetcher: Fetcher, race_date: date) -> list[str]:
    """Venue codes of every meeting under `race_date`'s heading of the 出馬表 menu (1 request)."""
    return list(op.meetings_on(_menu(fetcher, CARD_MENU, race_date), race_date, CARD_MENU.path))


def read_post_times(fetcher: Fetcher, race_date: date, venue: str) -> dict[int, datetime]:
    """Race number -> post time (aware UTC): the 出馬表 menu, then the card race list
    (2 requests)."""
    _, race_list = _race_list(fetcher, CARD_MENU, "official_card_racelist", race_date, venue)
    return op.post_times(race_list, race_date)


def published_results(fetcher: Fetcher, race_date: date, venue: str) -> set[int]:
    """Race numbers whose official result is out (ticket #225): those the meeting's レース結果 race
    list links a result page for. The レース結果 menu, then the race list (2 requests); raises
    NotListed or WrongMeeting like the other race lists."""
    _, race_list = _race_list(fetcher, RESULTS_MENU, "official_results_racelist", race_date, venue)
    return op.result_races(race_list)


def raw_odds_path(
    race_date: date, venue: str, race_no: int, code: str, fetched_utc: datetime
) -> Path:
    """Where a capture keeps one odds page: raw/jra/official_odds/<race date>/<VENUE>/
    <HHMMSS>Z-<RR>-<code>.html.gz, the time being the fetch time in UTC."""
    stamp = fetched_utc.astimezone(timezone.utc)
    name = f"{stamp:%H%M%S}Z-{race_no:02d}-{code}.html.gz"
    return paths.raw_dir(SOURCE) / "official_odds" / race_date.isoformat() / venue / name


def _keep_raw(path: Path, body: bytes) -> None:
    """gzip `body` to `path`: a temp file beside it, then a rename; no temp file survives."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                gz.write(body)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def capture_meeting(
    fetcher: Fetcher,
    race_date: date,
    venue: str,
    pools: Sequence[str],
    *,
    races: Sequence[int] | None = None,
    now: Callable[[], datetime] = utc_now,
    raw_pages: str | None = None,
) -> MeetingCapture:
    """The odds rows of `pools` (of ODDS_POOLS) for every race of the meeting (or `races` only),
    in race-list order: the オッズ menu, the race list, then one request per distinct odds page of
    each race. The BRACKET_QUINELLA page is read by `exotic_pages`, its rows in tier 2's shape
    (#358). `raw_pages` "all" keeps each odds page body (see raw_odds_path), "none" nothing; None
    reads logger.yaml. A page that cannot be written never fails the capture: one warning is
    logged. Raises NotListed, WrongMeeting, or ValueError (an unknown pool or raw_pages, before
    any request)."""

    def read(page: RawPage, what: str, number: int, page_pools: list[str], snapshot_utc) -> list:
        boards = [p for p in page_pools if p in _BOARD_POOLS]
        rows = _board_rows(page, what, race_date, venue, number, boards, snapshot_utc)
        classic = [p for p in page_pools if p not in boards]
        if not classic:
            return rows
        doc = op.soup(page.body)
        _check(doc, race_date, venue, what)
        _check_race(op.race_number(doc), number, what)
        rid = race_id(race_date, venue, number)
        return rows + op.odds_rows(
            doc, rid, classic, snapshot_utc=snapshot_utc, fetched_utc=page.fetched_utc
        )

    return _capture(fetcher, race_date, venue, pools, ODDS_POOLS, races, now, raw_pages, read)


def _board_rows(
    page: RawPage,
    what: str,
    race_date: date,
    venue: str,
    number: int,
    pools: Sequence[str],
    snapshot_utc: datetime,
) -> list[dict[str, Any]]:
    """The rows of `pools` off an odds page read by `exotic_pages` (one pool per page on JRADB; a
    shared page is read per pool), each board's date line and race number checked."""
    rows: list[dict[str, Any]] = []
    for pool in pools:
        board = ep.read_board(page.body, pool)
        _verdict(board.meeting, race_date, venue, what)
        _check_race(board.race_no, number, what)
        rid = race_id(race_date, venue, number)
        at = {"snapshot_utc": snapshot_utc, "fetched_utc": page.fetched_utc}
        rows += ep.board_rows(board, rid, pool, **at)
    return rows


def capture_tier2(
    fetcher: Fetcher,
    race_date: date,
    venue: str,
    pools: Sequence[str],
    *,
    races: Sequence[int],
    now: Callable[[], datetime] = utc_now,
    raw_pages: str | None = None,
    win5: Win5 | None = None,
) -> MeetingCapture:
    """capture_meeting for the tier-2 pools (TIER2_POOLS, a page each, read by `exotic_pages`
    without holding a page's tree). With `win5`, the WIN5 page too (1 request more) when the オッズ
    menu links one (a WIN5 day) and `win5.legs` is None (not read yet) or names a race captured;
    its legs are kept in `win5`, its row added when the page is for `race_date`."""

    def read(page: RawPage, what: str, number: int, page_pools: list[str], snapshot_utc) -> list:
        return _board_rows(page, what, race_date, venue, number, page_pools, snapshot_utc)

    def win5_rows(menu: BeautifulSoup, snapshot_utc: datetime) -> list[dict[str, Any]]:
        asked = {race_id(race_date, venue, n) for n in races}
        if win5 is None or (win5.legs is not None and not asked & set(win5.legs)):
            return []
        named = (op.link_of(a) for a in menu.find_all("a") if "WIN5" in op.clean(a.get_text()))
        link = next((x for x in named if x is not None and x.path == op.ODDS_PATH), None)
        if link is None:
            return []
        try:
            page = fetcher.fetch(task("official_odds", link, race_date.year))
            at = {"snapshot_utc": snapshot_utc, "fetched_utc": page.fetched_utc}
            rows = ep.win5_rows(page.body, **at)
        except (FetchError, ValueError) as exc:  # the boards stand; unread legs: tried no more
            log.warning("WIN5 page %s not read: %s", link.cname, exc)
            win5.legs = win5.legs or []
            return []
        win5.legs = [leg for row in rows for leg in row["legs"]]
        today = f"{SOURCE}:{race_date.isoformat()}:{ep.WIN5}"
        return [row for row in rows if row["race_id"] == today]  # not another day's WIN5

    return _capture(
        fetcher, race_date, venue, pools, TIER2_POOLS, races, now, raw_pages, read, win5_rows
    )


def _capture(
    fetcher: Fetcher,
    race_date: date,
    venue: str,
    pools: Sequence[str],
    known: Sequence[str],
    races: Sequence[int] | None,
    now: Callable[[], datetime],
    raw_pages: str | None,
    read: Callable[..., list[dict[str, Any]]],
    win5: Callable[[BeautifulSoup, datetime], list[dict[str, Any]]] | None = None,
) -> MeetingCapture:
    """The オッズ menu, the race list, then each distinct odds page of `pools` (of `known`) for the
    races asked, `read(page, what, race_no, page_pools, snapshot_utc)` into rows; then `win5`."""
    wanted = list(dict.fromkeys(pools))
    unknown = [p for p in wanted if p not in known]
    if unknown or not wanted:
        raise ValueError(f"no JRA live odds for pool(s) {unknown or wanted}; known: {known}")
    if raw_pages is None:
        raw_pages = load_country_yaml("japan", "logger")["raw_pages"]
    if raw_pages not in RAW_PAGES:
        raise ValueError(f"raw_pages must be one of {RAW_PAGES}, not {raw_pages!r}")
    snapshot_utc = now()
    start = fetcher.requests_sent
    menu, race_list = _race_list(fetcher, ODDS_MENU, "official_racelist", race_date, venue)
    rows: list[dict[str, Any]] = []
    unwritten: list[OSError] = []
    for number, row in op.race_rows(race_list):
        if races is not None and number not in races:
            continue
        offered = op.pool_links(row)
        on_page: dict[op.Link, list[str]] = {}
        for pool in wanted:
            if pool in offered:
                on_page.setdefault(offered[pool], []).append(pool)
        for link, page_pools in on_page.items():
            page = fetcher.fetch(task("official_odds", link, race_date.year))
            if raw_pages == "all":  # kept before it is read: a page a parser fails on included
                code = link.cname.split("/")[0]
                try:
                    _keep_raw(raw_odds_path(race_date, venue, number, code, page.fetched_utc),
                              page.body)  # fmt: skip
                except OSError as err:
                    unwritten.append(err)
            rows += read(page, f"odds page {link.cname}", number, page_pools, snapshot_utc)
    if win5 is not None:
        rows += win5(menu, snapshot_utc)
    if unwritten:
        log.warning(
            "raw odds pages not kept: %d of this capture's pages failed to write (first: %s)",
            len(unwritten),
            unwritten[0],
        )
    return MeetingCapture(rows, snapshot_utc, fetcher.requests_sent - start)
