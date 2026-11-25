"""JRADB page parsing: menus, race lists and odds pages (ported from HKTrack's
`sources/jra/official.py`). Pure: no network; fetching goes through `horsetrack.core.ingest`.

JRADB pages are POSTs of `cname=<page code>/<2-hex checksum>` to /JRADB/access<X>.html. Checksums
are never computed here: every (path, cname) pair is read off a page already fetched, from its
links (`doAction('<path>', '<code>')` or `href="<path>?CNAME=<code>"`). Pages are Shift_JIS
(cp932) and untrusted data, only parsed. Venue names and pool labels come from
`countries/japan/config/jra.yaml`.

Odds rows: Place and Wide print a min-max range per ticket, `odds` = min, `odds_hi` = max; Win
and Quinella print one figure. A scratched horse (取消 / 除外) keeps its Win and Place line with
`status_raw` "SCR" and no price; pair tickets naming it are left out. 発売票数 (tickets sold, one
per unit stake) becomes a pool-total row: `combination` "__pool__", `investment` = tickets x unit
stake (yen). Changes from HKTrack:
(a) `source_update_utc` is the page's clock (HH時MM分) on the Tokyo date of `fetched_utc`, or the
    day before when that would be later than `fetched_utc`; null when the page prints no clock
    (最終オッズ). HKTrack dated it by the race day, a day off for captures made the day before.
(b) `snapshot_utc` is the caller's, one per capture; `fetched_utc` is the page's own fetch time.
(c) `status_raw` is the cell's own status only; the refresh-line text is `page_state`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
from typing import Any, NamedTuple
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup, Tag

from horsetrack.core.config import SourceConfig, load_source

POOL_TOTAL = "__pool__"  # `combination` of a pool-total row (as in odds_log)
ODDS_PATH = "/JRADB/accessO.html"  # the odds pages (and the オッズ menu)
RESULTS_PATH = "/JRADB/accessS.html"  # the result pages (and the レース結果 menu, race lists)
SCRATCHED = "SCR"  # status_raw of a scratched horse's line
SCRATCH_WORDS = ("取消", "除外")
PAIR_POOLS = ("QUINELLA", "QUINELLA_PLACE")

_DO_ACTION = re.compile(
    r"doAction\(\s*'(/JRADB/[A-Za-z0-9]+\.html)'\s*,\s*'([0-9A-Za-z]+/[0-9A-F]{2})'\s*\)"
)
_HREF = re.compile(r"(/JRADB/[A-Za-z0-9]+\.html)\?CNAME=([0-9A-Za-z]+/[0-9A-F]{2})")
_PAGE_MEETING = re.compile(
    r"([0-9]{4})年([0-9]{1,2})月([0-9]{1,2})日[^0-9]*?[0-9]+回(\D+?)[0-9]+日"
)
_MENU_DAY = re.compile(r"([0-9]{1,2})月([0-9]{1,2})日")
_MENU_MEETING = re.compile(r"[0-9]+回(\D+?)[0-9]+日")
_RACE_NO = re.compile(r"([0-9]{1,2})レース")
_CLOCK = re.compile(r"([0-9]{1,2})時([0-9]{1,2})分")

# Every CSS selector the parser reads, whole-page, by the kind of page it is read on (the kinds of
# tests/countries/japan/jradb_pages.PAGE_KINDS). The parser passes only these to `select` /
# `select_one`; test_official_pages.py checks that, and the `local` test_official_real.py checks
# that each one finds a node in the real page of its kind, so a page redesign shows up there.
_TITLE = "h1, h2"  # the date line of a race list (it has no .race_header)
_DATE_LINE = ".race_header .date_line .date"
_RACE_NUMBER = ".race_header .race_number img[alt]"
_POST_TIME = "td.time"
_REFRESH_TIME = ".refresh_line .time"
_ODDS_LIST = "#odds_list"
_NUM = "td.num"
_WIN_CELL = "td.odds_tan"
_PLACE_CELL = "td.odds_fuku"
_LOW = ".min"  # inside a Place or Wide cell; a Win or Quinella cell has none
_HIGH = ".max"
_VOTES_ROW = "#votes tr"
_PAIR_CAPTION = "#odds_list table caption"
_PAIR_HEAD = "#odds_list table tr th"

_PAGE_HEAD = (_DATE_LINE, _RACE_NUMBER, _REFRESH_TIME, _VOTES_ROW)
SELECTORS: Mapping[str, tuple[str, ...]] = {
    "card_menu": ("h3",),
    "odds_menu": ("h3",),
    "card_race_list": (_TITLE, _POST_TIME),
    "odds_race_list": (_TITLE, "tr th img[alt]"),
    "win_place": (*_PAGE_HEAD, _NUM, _WIN_CELL, _PLACE_CELL, _LOW, _HIGH),
    "quinella": (*_PAGE_HEAD, _ODDS_LIST, _PAIR_CAPTION, _PAIR_HEAD),
    "wide": (*_PAGE_HEAD, _ODDS_LIST, _PAIR_CAPTION, _PAIR_HEAD, _LOW, _HIGH),
}


class Link(NamedTuple):
    """Where a JRADB link POSTs: equal to the plain tuple (path, cname)."""

    path: str
    cname: str


def _config() -> SourceConfig:
    return load_source("japan", "jra")


def _zone() -> ZoneInfo:
    return ZoneInfo(_config().timezone)


# ---------------------------------------------------------------- pages and links


def soup(body: bytes) -> BeautifulSoup:
    """JRADB pages declare Shift_JIS; cp932 is the superset they are really written in."""
    return BeautifulSoup(body.decode("cp932", errors="replace"), "lxml")


def clean(text: str | None) -> str:
    return " ".join((text or "").split())


def link_of(a: Tag) -> Link | None:
    """The (path, cname) a link POSTs to, as printed on the page; None for any other link."""
    m = _DO_ACTION.search(str(a.get("onclick") or "")) or _HREF.fullmatch(str(a.get("href") or ""))
    return Link(m.group(1), m.group(2)) if m else None


def venue_code(name: str) -> str | None:
    venue = _config().venue_by_name(name)
    return venue.code if venue is not None else None


def page_meeting(doc: BeautifulSoup) -> tuple[date, str | None] | None:
    """(date, venue code) a race list or race page says it is for, from its date line
    ('2026年11月15日（日曜） 5回京都4日'); None when the page has none (an error page)."""
    nodes = [*doc.select(_DATE_LINE), *doc.select(_TITLE)]
    for node in nodes:
        m = _PAGE_MEETING.search(clean(node.get_text(" ")))
        if m:
            try:
                shown = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                continue
            return shown, venue_code(m.group(4))
    return None


def race_number(doc: BeautifulSoup) -> int | None:
    """The race number of a race page, from its race-number image ('10レース')."""
    img = doc.select_one(_RACE_NUMBER)
    m = _RACE_NO.fullmatch(clean(str(img.get("alt")))) if img is not None else None
    return int(m.group(1)) if m else None


def local_clock(local_date: date, text: str | None) -> datetime | None:
    """'15時01分' on `local_date` in Tokyo -> an aware datetime; None when `text` has no clock."""
    m = _CLOCK.search(text or "")
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    try:
        return datetime(
            local_date.year, local_date.month, local_date.day, hour, minute, tzinfo=_zone()
        )
    except ValueError:  # 24時 or 60分: not a clock
        return None


def meetings_on(menu: BeautifulSoup, race_date: date, path: str) -> dict[str, Link]:
    """Venue code -> race-list link of every meeting of `race_date` on a 出馬表 or オッズ menu:
    under the headings of its day, the links to `path` named like 5回京都4日. Meetings whose
    venue jra.yaml does not know are left out."""
    out: dict[str, Link] = {}
    for heading in menu.find_all("h3"):
        day = _MENU_DAY.search(heading.get_text())
        if not day or (int(day.group(1)), int(day.group(2))) != (race_date.month, race_date.day):
            continue
        if heading.parent is None:
            continue
        for name, link in _meeting_links(heading.parent, path):
            venue = venue_code(name)
            if venue is not None:
                out.setdefault(venue, link)
    return out


def _meeting_links(node: Tag, path: str) -> list[tuple[str, Link]]:
    """(venue name, link) of every meeting link under `node`: links to `path` named like
    5回京都4日."""
    out = []
    for a in node.find_all("a"):
        link, m = link_of(a), _MENU_MEETING.search(clean(a.get_text(" ")))
        if link and link.path == path and m:
            out.append((m.group(1), link))
    return out


def race_rows(race_list: BeautifulSoup) -> list[tuple[int, Tag]]:
    """(race number, table row) of every race on a race list, by the row's race-number image."""
    out = []
    for row in race_list.find_all("tr"):
        head = row.find("th")
        img = head.find("img", alt=True) if head is not None else None
        m = _RACE_NO.fullmatch(clean(str(img.get("alt")))) if img is not None else None
        if m:
            out.append((int(m.group(1)), row))
    return out


def pool_links(row: Tag) -> dict[str, Link]:
    """Pool code -> odds page linked from a race-list row, by the link text (単勝 複勝, 馬連,
    ワイド...) matched against the pool labels of jra.yaml."""
    config = _config()
    out: dict[str, Link] = {}
    for a in row.find_all("a"):
        link = link_of(a)
        if link is None or link.path != ODDS_PATH:
            continue
        for word in re.split(r"[\s・/]+", clean(a.get_text(" "))):
            pool = config.pool_by_label(word) if word else None
            if pool is not None:
                out.setdefault(pool.code, link)
    return out


def result_races(results_race_list: BeautifulSoup) -> set[int]:
    """Race numbers whose row on a レース結果 race list links a page on RESULTS_PATH: the race's
    result is out (ticket #225). A row linking only other pages (its card), or no row, is not; a
    link outside the race rows (the site's navigation) marks nothing."""
    out = set()
    for race_no, row in race_rows(results_race_list):
        links = (link_of(a) for a in row.find_all("a"))
        if any(link is not None and link.path == RESULTS_PATH for link in links):
            out.add(race_no)
    return out


def post_times(card_race_list: BeautifulSoup, race_date: date) -> dict[int, datetime]:
    """Race number -> post time (aware UTC) from a 出馬表 race list's time cells; cells that are
    not a clock (発走済, empty) are left out."""
    out = {}
    for race_no, row in race_rows(card_race_list):
        cell = row.select_one(_POST_TIME)
        local = local_clock(race_date, cell.get_text(" ")) if cell is not None else None
        if local is not None:
            out[race_no] = local.astimezone(timezone.utc)
    return out


# ---------------------------------------------------------------- odds pages


def source_update_utc(state: str | None, fetched_utc: datetime) -> datetime | None:
    """The page's clock (HH時MM分) on the Tokyo date of `fetched_utc`, or the day before when
    that would be later than the fetch (a page seen just after midnight); None with no clock."""
    fetched = (
        fetched_utc if fetched_utc.tzinfo is not None else fetched_utc.replace(tzinfo=timezone.utc)
    )
    local = local_clock(fetched.astimezone(_zone()).date(), state)
    if local is None:
        return None
    if local > fetched:
        local -= timedelta(days=1)  # wall-clock arithmetic; Tokyo has no DST
    return local.astimezone(timezone.utc)


def _price(cell: Tag | None) -> tuple[float | None, float | None, str | None]:
    """(odds or min, max or None, status): status SCRATCHED for 取消 / 除外, the printed text when
    not a number."""
    text = clean(cell.get_text(" ")) if cell is not None else ""
    if any(word in text for word in SCRATCH_WORDS):
        return None, None, SCRATCHED
    low, high = (
        (cell.select_one(_LOW), cell.select_one(_HIGH)) if cell is not None else (None, None)
    )
    if low is not None:
        return _number(low.get_text()), _number(high.get_text()) if high is not None else None, None
    value = _number(text)
    return value, None, None if value is not None else (text or None)


def _number(text: str) -> float | None:
    try:
        return float(clean(text).replace(",", ""))
    except ValueError:
        return None


def _int(text: str) -> int | None:
    t = clean(text)
    return int(t) if t.isascii() and t.isdigit() else None


def odds_rows(
    doc: BeautifulSoup,
    race_id: str,
    pools: list[str],
    *,
    snapshot_utc: datetime,
    fetched_utc: datetime,
) -> list[dict[str, Any]]:
    """The rows of `pools` on one odds page (単勝・複勝, 馬連 or ワイド) plus their pool totals.
    `snapshot_utc` is the capture's (one for every page of it), `fetched_utc` this page's."""
    state_node = doc.select_one(_REFRESH_TIME)
    state = clean(state_node.get_text(" ")) if state_node is not None else ""
    base = {
        "snapshot_utc": snapshot_utc,
        "race_id": race_id,
        "source_update_utc": source_update_utc(state, fetched_utc),
        "fetched_utc": fetched_utc,
        "page_state": state or None,
    }

    def row(pool: str, combination: str, low: float | None, high: float | None, status: str | None):
        out = {**base, "pool": pool, "combination": combination, "odds": low, "status_raw": status}
        if high is not None:
            out["odds_hi"] = high
        return out

    rows = []
    odds_list = doc.select_one(_ODDS_LIST) or doc
    for tr in odds_list.find_all("tr"):
        num = tr.select_one(_NUM)
        number = _int(num.get_text()) if num is not None else None
        if number is None or tr.select_one(_WIN_CELL) is None:
            continue
        for pool, cell in (("WIN", _WIN_CELL), ("PLACE", _PLACE_CELL)):
            if pool in pools:
                rows.append(row(pool, str(number), *_price(tr.select_one(cell))))
    pair_pools = [p for p in pools if p in PAIR_POOLS]
    for table in odds_list.find_all("table") if pair_pools else []:
        first = _int(table.caption.get_text()) if table.caption is not None else None
        for tr in table.find_all("tr") if first is not None else []:
            second = _int(tr.th.get_text()) if tr.th is not None else None
            if second is None:
                continue
            low, high, status = _price(tr.find("td"))
            if status != SCRATCHED:
                pair = f"{min(first, second)},{max(first, second)}"
                rows += [row(pool, pair, low, high, status) for pool in pair_pools]
    return rows + _pool_totals(doc, base, pools)


def _pool_totals(doc: BeautifulSoup, base: dict[str, Any], pools: list[str]) -> list[dict]:
    """発売票数: tickets sold per pool (one ticket = one unit stake) -> money in the pool."""
    config = _config()
    out = []
    for tr in doc.select(_VOTES_ROW):
        pool = config.pool_by_label(clean(tr.th.get_text())) if tr.th is not None else None
        tickets = _int(tr.td.get_text().replace(",", "")) if tr.td is not None else None
        if pool is not None and pool.code in pools and tickets is not None:
            out.append(
                {
                    **base,
                    "pool": pool.code,
                    "combination": POOL_TOTAL,
                    "odds": None,
                    "status_raw": None,
                    "investment": float(tickets * config.unit_stake),
                }
            )
    return out
