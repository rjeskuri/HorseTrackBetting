"""nar.netkeiba.com page parsing: the day's race list and the odds pages, per the "Logger spec for
NAR" in docs/research/nar-data-access.md. Pure: no network; fetching goes through
`horsetrack.core.ingest`. Pages are UTF-8 and untrusted data, only parsed. Venues and pool labels
come from `countries/japan/config/nar.yaml`.

Race ids: the site's 12 digits `YYYY VV MMDD RR` (year, venue number, month and day, race number)
become the store id `nar:<YYYY-MM-DD>:<VENUE>:<NN>`; the 12 digits stay on each row as
`source_race_key`. A venue number nar.yaml does not know is an error, never skipped.

Odds rows follow the JRA conventions of `jra/official_pages.py` and the keys of
docs/specs/odds-log.md: PLACE and QUINELLA_PLACE print a range `lo - hi` (`odds`, `odds_hi`); a
scratched runner (取消 / 除外) keeps its WIN and PLACE rows with odds None and status_raw "SCR",
pairs naming it are left out; tickets sold become a "__pool__" row (`investment` = tickets x 100
yen); `source_update_utc` is the page's HH:MM on the Tokyo date of `fetched_utc`, or the day before
when that would be later than the fetch. A block's pool is read from its heading, so a page never
yields another pool's rows. The before-sales 予想オッズ table is netkeiba's forecast, not the board:
it raises `NotOnSale`.

Not done here: recognising the final board (the captured finished-race page prints no 最終 or 確定
marker) and checking the page's own race id against the one asked for (the logger's job, as JRA's
WrongMeeting is the capture's).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup, Tag

from horsetrack.core.config import load_country_yaml
from horsetrack.countries.japan.jra.official_pages import SCRATCH_WORDS, SCRATCHED, clean
from horsetrack.countries.japan.odds_log import POOL_TOTAL

SOURCE = "nar"
RUNNER_POOLS = ("WIN", "PLACE")
PAIR_POOLS = ("QUINELLA", "QUINELLA_PLACE")

_NETKEIBA_ID = re.compile(r"[0-9]{12}")
_RACE_NO = re.compile(r"([0-9]{1,2})R")
_CLOCK = re.compile(r"([0-9]{1,2}):([0-9]{2})")
_RANGE = re.compile(r"([0-9][0-9.,]*)\s*-\s*([0-9][0-9.,]*)")
_TICKETS = re.compile(r"([0-9][0-9,]*)\s*票?")

# Selectors seen on the captured pages (research doc, "Selectors").
_MEETING = "dl.RaceList_DataList"
_MEETING_TITLE = "p.RaceList_DataTitle"
_NIGHTER = "small.icon-nighter"
_RACE_LINK = 'li.RaceList_DataItem > a[href*="shutuba.html?race_id="]'
_RACE_NUM = ".Race_Num"
_POST_TIME = ".RaceData span"  # the first span is the post time
_HORSE_TABLE = "table.RaceOdds_HorseList_Table"
_FORECAST_TABLE = "table.RaceOdds_HorseList_Table.Ninki"  # before sales: 予想オッズ
_PRICE = "span.Odds"
_NUMBER_COLUMN = "馬番"
_ODDS_COLUMN = "オッズ"
_FORECAST_COLUMN = "予想オッズ"

# GUESSES, unverified: no page of a race on sale was captured. Revise these, and only these, when
# one is seen (tests/countries/japan/nar_pages.py builds the same guesses).
_ODDS_BLOCKS = "#odds_tan_block, #odds_fuku_block, #odds_pair_block"  # pair block: a guess
_UPDATE_TIME = "#odds_update_time"  # text 'HH:MM現在' (JST); absent on the finished page captured
_SCRATCHED_ROW = "Cancel"  # class of a scratched runner's row, besides the 取消 / 除外 word
_VOTE_ROWS = "#odds_vote_block table.RaceOdds_Vote_Table tr"  # th label, td '12,345票'
_PAIR_TABLE = "table.Odds_Table"  # one per first runner: th its number
_PAIR_NUMBER = "td.Num"  # per row: the second runner
_PAIR_PRICE = "td.Odds"


class NotOnSale(Exception):
    """The odds page shows netkeiba's forecast (予想オッズ), not the board: sales are not open."""


@dataclass(frozen=True)
class Meeting:
    """One venue's racing on a day of the race list."""

    venue: str  # the nar.yaml short code
    kaisai_id: str  # 10 digits: the race id without its race number
    nighter: bool
    races: dict[int, str] = field(default_factory=dict)  # race number -> 12-digit race id


def _config() -> dict[str, Any]:
    return load_country_yaml("japan", SOURCE)


def _zone() -> ZoneInfo:
    return ZoneInfo(_config()["timezone"])


def soup(body: bytes) -> BeautifulSoup:
    """nar.netkeiba.com pages are UTF-8 (header and meta)."""
    return BeautifulSoup(body.decode("utf-8", errors="replace"), "lxml")


def _text(node: Tag | None) -> str:
    return clean(node.get_text(" ")) if node is not None else ""


# ---------------------------------------------------------------- race ids


def _venue_of(number: str) -> str:
    for code, venue in _config()["venues"].items():
        if str(venue["number"]) == number:
            return code
    raise ValueError(f"venue number {number!r} is not in nar.yaml")


def _parse_id(netkeiba_id: str) -> tuple[date, str, int]:
    """(JST date, venue code, race number) of a 12-digit race id; ValueError when malformed."""
    if not _NETKEIBA_ID.fullmatch(netkeiba_id):
        raise ValueError(f"not a 12-digit NAR race id: {netkeiba_id!r}")
    venue = _venue_of(netkeiba_id[4:6])
    day = date(int(netkeiba_id[:4]), int(netkeiba_id[6:8]), int(netkeiba_id[8:10]))
    race_no = int(netkeiba_id[10:])
    if race_no < 1:
        raise ValueError(f"race number 00 in {netkeiba_id!r}")
    return day, venue, race_no


def nar_race_id(netkeiba_id: str) -> str:
    """'202642112511' -> 'nar:2026-11-25:URA:11'."""
    day, venue, race_no = _parse_id(netkeiba_id)
    return f"{SOURCE}:{day.isoformat()}:{venue}:{race_no:02d}"


# ---------------------------------------------------------------- race list


def _href_race_id(href: str) -> str:
    ids = parse_qs(urlsplit(href).query).get("race_id", [])
    if len(ids) != 1:
        raise ValueError(f"race link without one race_id: {href!r}")
    return ids[0]


def _races(race_list: BeautifulSoup, race_date: date) -> Iterator[tuple[str, bool, str, Tag]]:
    """(venue, nighter, race id, race item) of every race of `race_date` on the list. Every race
    link is checked, whatever its date: an unknown venue or a malformed id raises ValueError."""
    for meeting in race_list.select(_MEETING):
        title = meeting.select_one(_MEETING_TITLE)
        nighter = title is not None and title.select_one(_NIGHTER) is not None
        for a in meeting.select(_RACE_LINK):
            race_id = _href_race_id(str(a.get("href") or ""))
            day, venue, race_no = _parse_id(race_id)
            shown = _RACE_NO.fullmatch(_text(a.select_one(_RACE_NUM)))
            if shown is not None and int(shown.group(1)) != race_no:
                raise ValueError(f"race {race_id} is labelled {shown.group(0)}")
            if day == race_date:
                yield venue, nighter, race_id, a.parent if a.parent is not None else a


def meetings_on(race_list: BeautifulSoup, race_date: date) -> dict[str, Meeting]:
    """Venue code -> the meeting of `race_date` on a race_list_sub page, with its races."""
    races: dict[str, dict[int, str]] = {}
    nighters: dict[str, bool] = {}
    for venue, nighter, race_id, _ in _races(race_list, race_date):
        races.setdefault(venue, {})[int(race_id[10:])] = race_id
        nighters[venue] = nighters.get(venue, False) or nighter
    return {
        venue: Meeting(venue, next(iter(ids.values()))[:10], nighters[venue], ids)
        for venue, ids in races.items()
    }


def post_times(race_list: BeautifulSoup, race_date: date, venue: str) -> dict[int, datetime]:
    """Race number -> post time (aware UTC) of `venue`'s races of `race_date`; the page prints
    JST HH:MM. A race whose first .RaceData span is not a clock is left out."""
    out = {}
    for code, _, race_id, item in _races(race_list, race_date):
        local = local_clock(race_date, _text(item.select_one(_POST_TIME)), whole=True)
        if code == venue and local is not None:
            out[int(race_id[10:])] = local.astimezone(timezone.utc)
    return out


def local_clock(local_date: date, text: str | None, *, whole: bool = False) -> datetime | None:
    """'14:27' (the whole text, or anywhere in it) on `local_date` in Tokyo -> an aware datetime;
    None when there is no valid clock."""
    m = (_CLOCK.fullmatch if whole else _CLOCK.search)(text or "")
    if m is None or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        return None
    return datetime(
        local_date.year,
        local_date.month,
        local_date.day,
        int(m.group(1)),
        int(m.group(2)),
        tzinfo=_zone(),
    )


# ---------------------------------------------------------------- odds pages


def source_update_utc(state: str | None, fetched_utc: datetime) -> datetime | None:
    """The page's clock (HH:MM) on the Tokyo date of `fetched_utc`, or the day before when that
    would be later than the fetch (a page seen just after midnight); None with no clock."""
    fetched = (
        fetched_utc if fetched_utc.tzinfo is not None else fetched_utc.replace(tzinfo=timezone.utc)
    )
    local = local_clock(fetched.astimezone(_zone()).date(), state)
    if local is None:
        return None
    if local > fetched:
        local -= timedelta(days=1)  # wall-clock arithmetic; Tokyo has no DST
    return local.astimezone(timezone.utc)


def _pool_of(label: str) -> str | None:
    """The pool code a printed label names (nar.yaml labels), else None."""
    for code, pool in _config()["pools"].items():
        if label in pool["labels"]:
            return code
    return None


def _number(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _int(node: Tag | None) -> int | None:
    t = _text(node)
    return int(t) if t.isascii() and t.isdigit() else None


def _price(cell: Tag, *, scratched: bool = False) -> tuple[float | None, float | None, str | None]:
    """(odds or min, max or None, status) of a price cell: SCRATCHED for 取消 / 除外, the printed
    text when it is not a number."""
    text = _text(cell.select_one(_PRICE) or cell)
    if scratched or any(word in text for word in SCRATCH_WORDS):
        return None, None, SCRATCHED
    ranged = _RANGE.fullmatch(text)
    if ranged is not None:
        return _number(ranged.group(1)), _number(ranged.group(2)), None
    value = _number(text)
    return value, None, None if value is not None else (text or None)


def _before_sales(doc: BeautifulSoup) -> bool:
    if doc.select_one(_FORECAST_TABLE) is not None:
        return True
    return any(
        _text(th) == _FORECAST_COLUMN for table in doc.select(_HORSE_TABLE) for th in table("th")
    )


def _runner_prices(
    block: Tag,
) -> Iterator[tuple[int, tuple[float | None, float | None, str | None]]]:
    """(runner number, price) of each row of a 単勝 or 複勝 block, columns keyed by header label."""
    table = block.select_one(_HORSE_TABLE)
    if table is None:
        return
    labels = [_text(th) for th in table("th")]
    if _NUMBER_COLUMN not in labels or _ODDS_COLUMN not in labels:
        raise ValueError(f"odds table without {_NUMBER_COLUMN} and {_ODDS_COLUMN}: {labels}")
    number_at, odds_at = labels.index(_NUMBER_COLUMN), labels.index(_ODDS_COLUMN)
    for tr in table("tr"):
        cells = tr.find_all("td", recursive=False)
        if len(cells) != len(labels):
            continue
        number = _int(cells[number_at])
        if number is not None:
            scratched = _SCRATCHED_ROW in (tr.get("class") or [])
            yield number, _price(cells[odds_at], scratched=scratched)


def _pair_prices(block: Tag) -> Iterator[tuple[str, tuple[float | None, float | None, str | None]]]:
    """("a,b" sorted ascending, price) of each pair of a pair-pool block; pairs naming a scratched
    runner are left out."""
    for table in block.select(_PAIR_TABLE):
        first = _int(table.find("th"))
        for tr in table("tr") if first is not None else []:
            second, cell = _int(tr.select_one(_PAIR_NUMBER)), tr.select_one(_PAIR_PRICE)
            if second is None or cell is None:
                continue
            price = _price(cell)
            if price[2] != SCRATCHED:
                yield f"{min(first, second)},{max(first, second)}", price


def odds_rows(
    doc: BeautifulSoup,
    race_id: str,
    pools: Collection[str],
    *,
    snapshot_utc: datetime,
    fetched_utc: datetime,
) -> list[dict[str, Any]]:
    """The rows of `pools` on one odds page (`b1` 単勝・複勝, `b4` 馬連, `b5` ワイド) plus
    their pool totals. `race_id` is the 12-digit netkeiba id; `snapshot_utc` is the capture's (one
    for every page of it), `fetched_utc` this page's. NotOnSale before sales open."""
    store_id = nar_race_id(race_id)
    if _before_sales(doc):
        raise NotOnSale(f"race {race_id}: the page shows {_FORECAST_COLUMN}, not the board")
    state = _text(doc.select_one(_UPDATE_TIME)) or None
    base = {
        "snapshot_utc": snapshot_utc,
        "fetched_utc": fetched_utc,
        "race_id": store_id,
        "source_race_key": race_id,
        "source_update_utc": source_update_utc(state, fetched_utc),
        "page_state": state,
    }

    def row(pool: str, combination: str, low: float | None, high: float | None, status: str | None):
        out = {**base, "pool": pool, "combination": combination, "odds": low, "status_raw": status}
        if high is not None:
            out["odds_hi"] = high
        return out

    rows = []
    for block in doc.select(_ODDS_BLOCKS):
        pool = _pool_of(_text(block.find("h2")))
        if pool is None or pool not in pools:
            continue
        if pool in RUNNER_POOLS:
            rows += [row(pool, str(n), *price) for n, price in _runner_prices(block)]
        elif pool in PAIR_POOLS:
            rows += [row(pool, pair, *price) for pair, price in _pair_prices(block)]
    return rows + _pool_totals(doc, base, pools)


def _pool_totals(doc: BeautifulSoup, base: dict[str, Any], pools: Collection[str]) -> list[dict]:
    """Tickets sold per pool (one ticket = one unit stake) -> money in the pool."""
    unit = float(_config()["unit_stake"])
    out = []
    for tr in doc.select(_VOTE_ROWS):
        pool = _pool_of(_text(tr.find("th")))
        tickets = _TICKETS.fullmatch(_text(tr.find("td")))
        if pool is not None and pool in pools and tickets is not None:
            out.append(
                {
                    **base,
                    "pool": pool,
                    "combination": POOL_TOTAL,
                    "odds": None,
                    "status_raw": None,
                    "investment": int(tickets.group(1).replace(",", "")) * unit,
                }
            )
    return out
