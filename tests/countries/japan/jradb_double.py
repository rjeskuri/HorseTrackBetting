"""Offline JRADB double for the capture tests (ticket #23), ported from HKTrack's
`tests/unit/sources/jra/official_site.py` (the mechanism, not the pages).

`JradbSite` is an httpx.MockTransport handler that answers like JRADB: a POST to
www.jra.go.jp at the path a page was linked under, with the form body `cname=<code>`, gets that
page (cp932 bytes, built by `jradb_pages`); anything else gets an error page with no date line.
Every request is recorded in `site.seen`. `RecordingFetcher` is the real ingest Fetcher that also
keeps every FetchTask it was handed, so tests can read page types without a network.

`World` builds a synthetic week: per day, the meetings racing (venue code, meeting name, number
of races); both menus (出馬表 and オッズ, with the fixed entry codes), each meeting's two race
lists and three odds pages per race (単勝・複勝, 馬連, ワイド). Every other code is invented and
carries a checksum the code under test cannot derive: it is only readable off the pages.
`scramble` swaps every checksum for another, in the pages and the routing alike.

Every value is invented. No real page, name or scraped value is copied here.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs

import httpx
import jradb_pages as pages
from jradb_pages import CARD_PATH, ODDS_PATH

from horsetrack.core.ingest.fetcher import Fetcher
from horsetrack.core.ingest.types import FetchTask

HOST = "www.jra.go.jp"
SOURCE = "jra"
CARD_MENU = (CARD_PATH, "pw01dli00/F3")  # 出馬表: the week's meetings (fixed entry point)
ODDS_MENU = (ODDS_PATH, "pw15oli00/6D")  # オッズ: the week's meetings (fixed entry point)
ENTRY_CNAMES = {CARD_MENU[1], ODDS_MENU[1]}
LIVE_STATE = "14時00分現在"
JST = timezone(timedelta(hours=9))
UTC = timezone.utc

# The fetcher's http block for tests: a high rate and no jitter, so nothing waits.
HTTP: dict[str, Any] = {
    "user_agent": "horsetrack-tests/1.0",
    "http2": False,
    "timeout_s": 5,
    "max_retries": 0,
    "backoff_base_s": 0.0,
    "circuit_breaker": {"stop_on_status": [403, 429], "max_consecutive_failures": 50},
    "hosts": {"default": {"requests_per_second": 1000.0, "jitter": 0.0}},
}

VENUES = {  # code -> (number, name as printed), as in countries/japan/config/jra.yaml
    "SAP": ("01", "札幌"),
    "HAK": ("02", "函館"),
    "FKS": ("03", "福島"),
    "NII": ("04", "新潟"),
    "TOK": ("05", "東京"),
    "NAK": ("06", "中山"),
    "CKY": ("07", "中京"),
    "KYO": ("08", "京都"),
    "HSN": ("09", "阪神"),
    "KOK": ("10", "小倉"),
}
KINDS = ("winplace", "quinella", "wide")  # the odds pages of a race, one per kind
KIND_POOLS = {"winplace": ("WIN", "PLACE"), "quinella": ("QUINELLA",), "wide": ("QUINELLA_PLACE",)}
POOL_KIND = {pool: kind for kind, pools in KIND_POOLS.items() for pool in pools}

Link = tuple[str, str]  # (path, cname)

_CNAME = re.compile(r"(pw[0-9A-Za-z]+)/([0-9A-F]{2})(?![0-9A-Za-z])")


def meeting_name(venue: str, kai: int = 5, nichi: int = 7) -> str:
    """5回東京7日"""
    return f"{kai}回{VENUES[venue][1]}{nichi}日"


def _checksum(code: str) -> str:
    """An arbitrary 2-hex checksum: nothing outside this file can derive it from the code."""
    return f"{hashlib.sha256(code.encode()).digest()[0]:02X}"


def _cname(code: str) -> str:
    return f"{code}/{_checksum(code)}"


# ---------------------------------------------------------------- the races


@dataclass(frozen=True)
class Race:
    """One race's invented board: prices as the pages print them."""

    day: date
    venue: str
    number: int
    win: dict[int, float]
    place: dict[int, tuple[float, float]]
    quinella: dict[tuple[int, int], float]
    wide: dict[tuple[int, int], tuple[float, float]]
    post_minutes: int  # minutes after local midnight in Tokyo

    @property
    def race_id(self) -> str:
        return f"jra:{self.day.isoformat()}:{self.venue}:{self.number:02d}"

    @property
    def post_text(self) -> str:
        return f"{self.post_minutes // 60}時{self.post_minutes % 60:02d}分"

    @property
    def post_utc(self) -> datetime:
        local = datetime(self.day.year, self.day.month, self.day.day, tzinfo=JST)
        return (local + timedelta(minutes=self.post_minutes)).astimezone(UTC)

    def expected(self, pools: Sequence[str]) -> list[tuple]:
        """(race_id, pool, combination, odds, odds_hi) of every price row of `pools`."""
        out = []
        for h, price in self.win.items():
            if "WIN" in pools:
                out.append((self.race_id, "WIN", str(h), price, None))
        for h, (low, high) in self.place.items():
            if "PLACE" in pools:
                out.append((self.race_id, "PLACE", str(h), low, high))
        for (a, b), price in self.quinella.items():
            if "QUINELLA" in pools:
                out.append((self.race_id, "QUINELLA", f"{a},{b}", price, None))
        for (a, b), (low, high) in self.wide.items():
            if "QUINELLA_PLACE" in pools:
                out.append((self.race_id, "QUINELLA_PLACE", f"{a},{b}", low, high))
        return out


def make_race(day: date, venue: str, number: int) -> Race:
    k = day.toordinal() % 7 + list(VENUES).index(venue) * 3 + number
    runners = range(1, 5 + (number + k) % 6 + 1)  # 5 to 10 runners
    win = {h: round(1.5 + ((h * 7 + k) % 40) * 1.3, 1) for h in runners}
    place = {}
    for h in runners:
        low = round(1.1 + ((h * 3 + k) % 9) * 0.4, 1)
        place[h] = (low, round(low + 0.8, 1))
    quinella, wide = {}, {}
    for a in runners:
        for b in runners:
            if a < b:
                quinella[(a, b)] = round(3.0 + ((a * 11 + b * 5 + k) % 97) * 1.7, 1)
                low = round(1.5 + ((a * 5 + b * 3 + k) % 50) * 0.9, 1)
                wide[(a, b)] = (low, round(low + 1.2, 1))
    post = 9 * 60 + 30 + 30 * (number - 1) + 5 * (k % 3)
    return Race(day, venue, number, win, place, quinella, wide, post)


# ---------------------------------------------------------------- the week


@dataclass(frozen=True)
class Meeting:
    venue: str
    races: int = 12
    kai: int = 5
    nichi: int = 7

    @property
    def name(self) -> str:
        return meeting_name(self.venue, self.kai, self.nichi)


class World:
    """The pages of a synthetic week, keyed by (path, cname) as JRADB routes them."""

    def __init__(self, days: Mapping[date, Sequence[Meeting]]):
        self.days = {day: list(meetings) for day, meetings in days.items()}
        self.pages: dict[Link, bytes] = {}
        self._card_lists: dict[tuple[date, str], Link] = {}
        self._odds_lists: dict[tuple[date, str], Link] = {}
        self._odds: dict[tuple[date, str, int, str], Link] = {}
        self._meetings: dict[tuple[date, str], Meeting] = {}
        card_menu: dict[date, list[tuple[str, str]]] = {}
        odds_menu: dict[date, list[tuple[str, str]]] = {}
        for day, meetings in self.days.items():
            card_menu[day], odds_menu[day] = [], []
            for m in meetings:
                self._meetings[(day, m.venue)] = m
                card, odds = self._build_meeting(day, m)
                card_menu[day].append((m.name, card[1]))
                odds_menu[day].append((m.name, odds[1]))
        self.pages[CARD_MENU] = pages.menu(card_menu, CARD_PATH)
        self.pages[ODDS_MENU] = pages.menu(odds_menu, ODDS_PATH)

    def _codes(self, day: date, m: Meeting) -> str:
        return f"{VENUES[m.venue][0]}{day.year}{m.kai:02d}{m.nichi:02d}"

    def _build_meeting(self, day: date, m: Meeting) -> tuple[Link, Link]:
        mid, ymd = self._codes(day, m), f"{day:%Y%m%d}"
        card = (CARD_PATH, _cname(f"pw01drl10{mid}{ymd}"))
        odds = (ODDS_PATH, _cname(f"pw15orl10{mid}{ymd}"))
        self._card_lists[(day, m.venue)] = card
        self._odds_lists[(day, m.venue)] = odds
        buttons: dict[int, dict[str, str]] = {}
        for n in range(1, m.races + 1):
            race = make_race(day, m.venue, n)
            links = {
                kind: (ODDS_PATH, _cname(f"pw15{digit}ou10{mid}{n:02d}{ymd}Z"))
                for kind, digit in zip(KINDS, "145", strict=True)
            }
            for kind, link in links.items():
                self._odds[(day, m.venue, n, kind)] = link
                self.pages[link] = self.odds_page(race, kind, day, m.name)
            # The race list offers more pools than the four: those pages must never be asked for.
            extra = {
                label: _cname(f"pw15{digit}ou10{mid}{n:02d}{ymd}Z")
                for label, digit in (("枠連", "3"), ("馬単", "6"), ("3連複", "7"))
            }
            buttons[n] = {
                "単勝 複勝": links["winplace"][1],
                "馬連": links["quinella"][1],
                "ワイド": links["wide"][1],
                **extra,
            }
        times = {n: make_race(day, m.venue, n).post_text for n in range(1, m.races + 1)}
        self.pages[card] = pages.card_race_list(day, m.name, times)
        self.pages[odds] = pages.odds_race_list(day, m.name, buttons, times)
        return card, odds

    @staticmethod
    def odds_page(race: Race, kind: str, day: date, meeting: str) -> bytes:
        """The `kind` odds page of `race`, its date line saying `day` and `meeting`."""
        if kind == "winplace":
            runners = {h: (race.win[h], race.place[h]) for h in race.win}
            return pages.win_place(day, meeting, race.number, runners, state=LIVE_STATE)
        odds = race.quinella if kind == "quinella" else race.wide
        return pages.pairs(day, meeting, race.number, dict(odds), state=LIVE_STATE)

    # ------------------------------------------------------------ lookups

    def meeting(self, day: date, venue: str) -> Meeting:
        return self._meetings[(day, venue)]

    def race(self, day: date, venue: str, number: int) -> Race:
        return make_race(day, venue, number)

    def card_list(self, day: date, venue: str) -> Link:
        return self._card_lists[(day, venue)]

    def odds_list(self, day: date, venue: str) -> Link:
        return self._odds_lists[(day, venue)]

    def odds_link(self, day: date, venue: str, number: int, kind: str) -> Link:
        return self._odds[(day, venue, number, kind)]

    def odds_links(self, day: date, venue: str, number: int) -> dict[str, Link]:
        return {kind: self.odds_link(day, venue, number, kind) for kind in KINDS}

    def expected_rows(
        self, day: date, venue: str, pools: Sequence[str], races: Sequence[int] | None = None
    ) -> list[tuple]:
        m = self.meeting(day, venue)
        numbers = races if races is not None else range(1, m.races + 1)
        out = []
        for n in numbers:
            out += self.race(day, venue, n).expected(pools)
        return sorted(out, key=_sort_key)

    def expected_post_times(self, day: date, venue: str) -> dict[int, datetime]:
        m = self.meeting(day, venue)
        return {n: self.race(day, venue, n).post_utc for n in range(1, m.races + 1)}


def _sort_key(row: tuple) -> tuple:
    return tuple("" if v is None else str(v) for v in row)


def price_rows(rows: Sequence[Mapping[str, Any]]) -> list[tuple]:
    """(race_id, pool, combination, odds, odds_hi) of the price rows (pool totals left out)."""
    out = [
        (r["race_id"], r["pool"], r["combination"], r["odds"], r.get("odds_hi"))
        for r in rows
        if r["combination"] != "__pool__"
    ]
    return sorted(out, key=_sort_key)


def relabel(body: bytes, day: date, meeting: str, new_day: date, new_meeting: str) -> bytes:
    """The same page with its date line saying another meeting (the site answering for another
    meeting than the one asked for); everything else, its links included, unchanged."""
    text = body.decode("cp932")
    old, new = pages.date_line(day, meeting), pages.date_line(new_day, new_meeting)
    assert old in text and old != new
    return text.replace(old, new).encode("cp932")


def scramble(site_pages: Mapping[Link, bytes]) -> tuple[dict[Link, bytes], dict[str, str]]:
    """Every checksum replaced by another 2-hex value, in the pages and in the routing alike
    (entry menus kept): only code that reads the codes off the pages can still navigate.
    Returns (pages, {old cname: new cname})."""

    def new(cname: str) -> str:
        body, checksum = cname.rsplit("/", 1)
        return cname if cname in ENTRY_CNAMES else f"{body}/{int(checksum, 16) ^ 0xA5:02X}"

    mapping: dict[str, str] = {}

    def swap(m: re.Match[str]) -> str:
        mapping[m.group(0)] = new(m.group(0))
        return mapping[m.group(0)]

    out = {}
    for (path, cname), body in site_pages.items():
        text = _CNAME.sub(swap, body.decode("cp932"))
        out[(path, new(cname))] = text.encode("cp932")
    return out, mapping


# ---------------------------------------------------------------- the site


@dataclass
class Seen:
    method: str
    host: str
    path: str
    cname: str | None
    content_type: str | None
    served: bool

    @property
    def link(self) -> Link:
        return (self.path, self.cname or "")


@dataclass
class JradbSite:
    pages: dict[Link, bytes]
    seen: list[Seen] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = request.read().decode("ascii", errors="replace") if request.method == "POST" else ""
        form = {k.lower(): v for k, v in parse_qs(body).items()}
        cname = (form.get("cname") or [None])[0]
        page = None
        if request.method == "POST" and request.url.host == HOST and cname is not None:
            page = self.pages.get((request.url.path, cname))
        self.seen.append(
            Seen(
                request.method,
                request.url.host,
                request.url.path,
                cname,
                request.headers.get("content-type"),
                page is not None,
            )
        )
        content = page if page is not None else pages.no_header_page()
        return httpx.Response(200, content=content, headers={"Content-Type": "text/html"})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def links(self) -> list[Link]:
        return [s.link for s in self.seen]


class RecordingFetcher(Fetcher):
    """The real Fetcher, also keeping every task it was handed (fetch goes through request)."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.tasks: list[FetchTask] = []

    def request(self, task: FetchTask) -> httpx.Response:
        self.tasks.append(task)
        return super().request(task)


def make_fetcher(site: JradbSite) -> RecordingFetcher:
    return RecordingFetcher(SOURCE, http=HTTP, transport=site.transport, sleep=lambda _s: None)
