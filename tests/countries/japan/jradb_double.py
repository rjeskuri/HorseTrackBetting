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

Results (ticket #225): the week also has the レース結果 menu (RESULTS_MENU, a fixed entry code like
the other two) and one results race list per meeting (`jradb_pages.results_race_list`). A World's
`published` gives, per (day, venue), the races whose result is out (default: none);
`results_page(day, venue, races)` builds the list as it reads with `races` out, so a test can
serve it changing over time. Result pages themselves are linked, never served.

Tier 2 (ticket #130): every race list also links 枠連, 馬単, 3連複 and 3連単. Their pages exist only
for the pools a World is given in `tier2` (built with `jradb_pages.legs`, prices from
`exotic_board`), so a request for any other pool's page gets the error page (`served` False). A
`win5` day adds a WIN5 link to that day's オッズ menu and its page (`jradb_pages.win5`).
Since #358 bracket quinella (枠連) is captured every round with tier 1: its page always exists
(TIER1_EXOTIC), whatever `tier2` says.

The evening before (#358): `ClockedSite` answers as of a fake clock. Before `opens_at` its オッズ
menu is `menu_before` (the race date's meetings not on it yet: the day-before sales are not
open), and from `block_from` on (until `block_until`, when given) every request is refused with
`status` (403: the breaker opens).
Every request's fake time is kept in `at`, in the order of `seen`.

Every value is invented. No real page, name or scraped value is copied here.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs

import httpx
import jradb_pages as pages
from jradb_pages import CARD_PATH, ODDS_PATH, RESULTS_MENU_CNAME, RESULTS_PATH

from horsetrack.core.ingest.fetcher import Fetcher
from horsetrack.core.ingest.types import FetchTask

HOST = "www.jra.go.jp"
SOURCE = "jra"
CARD_MENU = (CARD_PATH, "pw01dli00/F3")  # 出馬表: the week's meetings (fixed entry point)
ODDS_MENU = (ODDS_PATH, "pw15oli00/6D")  # オッズ: the week's meetings (fixed entry point)
RESULTS_MENU = (RESULTS_PATH, RESULTS_MENU_CNAME)  # レース結果: the week's meetings (#225)
ENTRY_CNAMES = {CARD_MENU[1], ODDS_MENU[1], RESULTS_MENU[1]}
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
# Tier 2 (#130): pool -> (race-list label, the code's page digit)
TIER2 = {
    "BRACKET_QUINELLA": ("枠連", "3"),
    "FORECAST": ("馬単", "6"),
    "TRIO": ("3連複", "7"),
    "TIERCE": ("3連単", "8"),
}
TIER1_EXOTIC = ("BRACKET_QUINELLA",)  # #358: captured every round with tier 1, its page served
UNIT_STAKE = 100  # yen per ticket, jra.yaml

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


# ---------------------------------------------------------------- tier 2 (#130)


def _brackets(runners: int) -> dict[int, int]:
    """Runner -> bracket: one runner a bracket up to 8 runners; past 8, the last brackets take
    two runners each (9 runners: 1-7 alone, 8 and 9 in bracket 8)."""
    single = 8 - max(0, runners - 8)
    return {h: h if h <= single else single + (h - single + 1) // 2 for h in range(1, runners + 1)}


def _seed(race: Race) -> int:
    return race.day.toordinal() % 11 + list(VENUES).index(race.venue) * 5 + race.number * 3


def exotic_board(race: Race, pool: str) -> dict[tuple[int, ...], float]:
    """The invented tier-2 board of `race`: combination as the page lays it out (finishing order
    for FORECAST and TIERCE, ascending for TRIO and BRACKET_QUINELLA, a same-bracket pair where a
    bracket holds two runners) -> odds."""
    runners = sorted(race.win)
    k = _seed(race)
    if pool == "BRACKET_QUINELLA":
        held: dict[int, int] = {}
        for b in _brackets(len(runners)).values():
            held[b] = held.get(b, 0) + 1
        pairs = [(a, b) for a in held for b in held if a < b or (a == b and held[a] > 1)]
        return {c: round(4.0 + ((c[0] * 13 + c[1] * 7 + k) % 61) * 1.1, 1) for c in pairs}
    if pool == "FORECAST":
        pairs = [(a, b) for a in runners for b in runners if a != b]
        return {c: round(6.0 + ((c[0] * 17 + c[1] * 5 + k) % 89) * 2.3, 1) for c in pairs}
    triples = [(a, b, c) for a in runners for b in runners for c in runners if len({a, b, c}) == 3]
    if pool == "TRIO":
        return {
            c: round(9.0 + ((c[0] * 7 + c[1] * 11 + c[2] * 3 + k) % 97) * 3.1, 1)
            for c in triples
            if c[0] < c[1] < c[2]
        }
    if pool == "TIERCE":
        return {
            c: round(20.0 + ((c[0] * 19 + c[1] * 7 + c[2] * 3 + k) % 211) * 9.7, 1) for c in triples
        }
    raise ValueError(pool)


def exotic_expected(race: Race, pool: str) -> dict[str, float]:
    """The log's {combination: odds} for `race`'s `pool` board: the page lays every combination
    out in the log's order already (ordered pools as run, unordered ones ascending)."""
    return {",".join(map(str, c)): odds for c, odds in exotic_board(race, pool).items()}


def exotic_tickets(race: Race, pool: str) -> int:
    """発売票数 the `pool` page of `race` prints."""
    return 1000 + _seed(race) * 37 + list(TIER2).index(pool) * 4111


@dataclass(frozen=True)
class Win5Day:
    """A WIN5 day: the five legs (venue code, race number), tickets sold and carryover in yen."""

    legs: tuple[tuple[str, int], ...]
    tickets: int
    carryover_yen: int

    def race_ids(self, day: date) -> list[str]:
        return [f"jra:{day.isoformat()}:{v}:{n:02d}" for v, n in self.legs]


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

    def __init__(
        self,
        days: Mapping[date, Sequence[Meeting]],
        *,
        tier2: Sequence[str] = (),
        win5: Mapping[date, Win5Day] | None = None,
        published: Mapping[tuple[date, str], Iterable[int]] | None = None,
    ):
        self.days = {day: list(meetings) for day, meetings in days.items()}
        unknown = set(tier2) - set(TIER2)
        assert not unknown, f"no tier-2 page for {unknown}"
        self.tier2 = tuple(tier2)
        self.pages: dict[Link, bytes] = {}
        self.win5_links: dict[date, Link] = {}
        self._card_lists: dict[tuple[date, str], Link] = {}
        self._odds_lists: dict[tuple[date, str], Link] = {}
        self._results_lists: dict[tuple[date, str], Link] = {}
        self._result_cnames: dict[tuple[date, str], dict[int, str]] = {}
        self._odds: dict[tuple[date, str, int, str], Link] = {}
        self._meetings: dict[tuple[date, str], Meeting] = {}
        card_menu: dict[date, list[tuple[str, str]]] = {}
        odds_menu: dict[date, list[tuple[str, str]]] = {}
        results_menu: dict[date, list[tuple[str, str]]] = {}
        for day, meetings in self.days.items():
            card_menu[day], odds_menu[day], results_menu[day] = [], [], []
            for m in meetings:
                self._meetings[(day, m.venue)] = m
                card, odds = self._build_meeting(day, m)
                card_menu[day].append((m.name, card[1]))
                odds_menu[day].append((m.name, odds[1]))
                results = self._build_results(day, m, (published or {}).get((day, m.venue), ()))
                results_menu[day].append((m.name, results[1]))
        extra: dict[date, list[tuple[str, str]]] = {}
        for day, w5 in (win5 or {}).items():
            link = (ODDS_PATH, _cname(f"pw15w5ou10{day:%Y%m%d}Z"))
            self.win5_links[day] = link
            extra[day] = [("WIN5", link[1])]
            printed = [(VENUES[v][1], n) for v, n in w5.legs]
            self.pages[link] = pages.win5(
                day, printed, state=LIVE_STATE, tickets=w5.tickets, carryover_yen=w5.carryover_yen
            )
        self.pages[CARD_MENU] = pages.menu(card_menu, CARD_PATH)
        self.pages[ODDS_MENU] = pages.menu(odds_menu, ODDS_PATH, extra=extra)
        self.pages[RESULTS_MENU] = pages.menu(results_menu, RESULTS_PATH)

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
            # The race list offers more pools than the four (tier 2, #130): their pages exist
            # only for the World's `tier2` pools and bracket quinella (tier 1 since #358); any
            # other is the error page (never served).
            extra = {}
            for pool, (label, digit) in TIER2.items():
                link = (ODDS_PATH, _cname(f"pw15{digit}ou10{mid}{n:02d}{ymd}Z"))
                extra[label] = link[1]
                self._odds[(day, m.venue, n, pool)] = link
                if pool in self.tier2 or pool in TIER1_EXOTIC:
                    self.pages[link] = pages.legs(
                        day,
                        m.name,
                        n,
                        exotic_board(race, pool),
                        state=LIVE_STATE,
                        votes=[(label, exotic_tickets(race, pool))],
                    )
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

    def _build_results(self, day: date, m: Meeting, published: Iterable[int]) -> Link:
        mid, ymd = self._codes(day, m), f"{day:%Y%m%d}"
        link = (RESULTS_PATH, _cname(f"pw01srl10{mid}{ymd}"))
        self._results_lists[(day, m.venue)] = link
        self._result_cnames[(day, m.venue)] = {
            n: _cname(f"pw01sde10{mid}{n:02d}{ymd}") for n in range(1, m.races + 1)
        }
        self.pages[link] = self.results_page(day, m.venue, published)
        return link

    def results_page(self, day: date, venue: str, published: Iterable[int]) -> bytes:
        """The meeting's results race list with the results of `published` out (#225)."""
        out = set(published)
        cnames = self._result_cnames[(day, venue)]
        rows = {n: (cname if n in out else None) for n, cname in cnames.items()}
        return pages.results_race_list(day, self.meeting(day, venue).name, rows)

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

    def results_list(self, day: date, venue: str) -> Link:
        return self._results_lists[(day, venue)]

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


@dataclass
class ClockedSite(JradbSite):
    """The double as of a fake clock (#358). `clock()` is read at each request and kept in `at`.
    Before `opens_at` (or always, when it is None) the オッズ menu is `menu_before` when given: the
    race date's meetings not listed yet, the day-before sales not open. From `block_from` on
    (until `block_until`, when given: #387, a block that lifts), every request is answered
    `status` with an empty body."""

    clock: Callable[[], datetime] | None = None
    opens_at: datetime | None = None
    menu_before: bytes | None = None
    block_from: datetime | None = None
    status: int = 403
    at: list[datetime] = field(default_factory=list)
    block_until: datetime | None = None

    def __post_init__(self) -> None:
        self._menu_open = self.pages.get(ODDS_MENU)

    def refused(self, t: datetime) -> bool:
        """Whether a request at `t` is refused."""
        if self.block_from is None or t < self.block_from:
            return False
        return self.block_until is None or t < self.block_until

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert self.clock is not None, "a ClockedSite needs its clock"
        t = self.clock()
        self.at.append(t)
        if self.refused(t):
            url = request.url
            self.seen.append(Seen(request.method, url.host, url.path, None, None, False))
            return httpx.Response(self.status, content=b"", headers={"Content-Type": "text/html"})
        is_open = self.opens_at is not None and t >= self.opens_at
        if self.menu_before is not None and not is_open:
            self.pages[ODDS_MENU] = self.menu_before
        elif self._menu_open is not None:
            self.pages[ODDS_MENU] = self._menu_open
        return super().handler(request)

    def times_of(self, link: Link) -> list[datetime]:
        """The fake time of every request for `link`, in order."""
        return [t for s, t in zip(self.seen, self.at, strict=True) if s.link == link]


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
