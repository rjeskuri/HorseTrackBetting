"""A synthetic nar.netkeiba.com behind an httpx.MockTransport (ticket #38), serving the pages of
`nar_pages.py` through the real ingest Fetcher.

`NarWorld` holds the days' meetings (venue code, post times as printed, nighter flag) and invents
every race's board (`make_race`: prices, ranges and tickets sold, all made up). `NarSite` answers
like the site, GET only, to nar.netkeiba.com:
- `/top/race_list_sub.html?kaisai_date=YYYYMMDD`: that day's race list (`nar_pages.race_list`),
  a list with no meeting on a day without racing;
- `/odds/?race_id=<12 digits>&type=<t>` (or `/odds/index.html?...`): `b1` the 単勝・複勝 page,
  `b4` the 馬連 page, `b5` the ワイド page of that race, each with its pools' tickets sold; for a
  race not on sale yet (`before_sales`) every type gets the 予想オッズ page;
- anything else: 404.
`site.replace[(race_id, type)] = body` serves another page at an odds URL (another race's, for the
wrong-meeting tests); `refuse_from=n` answers 403 from the n-th request on (1-based). Every
request is kept in `site.seen`. Expected values come from the invented boards here, never from
the code under test. Every name, id and number is invented.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
import nar_pages as pg
from jradb_double import HTTP, RecordingFetcher

HOST = "nar.netkeiba.com"
SOURCE = "nar"
POOL = "__pool__"
JST = timezone(timedelta(hours=9))
UTC = timezone.utc
RACE_LIST_PATH = "/top/race_list_sub.html"
ODDS_PATHS = ("/odds/", "/odds/index.html")

# code -> (netkeiba venue number, name as printed), as in countries/japan/config/nar.yaml
VENUES = {
    "URA": ("42", "浦和"),
    "FUN": ("43", "船橋"),
    "OOI": ("44", "大井"),
    "NGY": ("48", "名古屋"),
    "SON": ("50", "園田"),
}
# pool -> the odds page type that prints it, and the label of its tickets-sold row
PAGE_OF = {"WIN": "b1", "PLACE": "b1", "QUINELLA": "b4", "QUINELLA_PLACE": "b5"}
LABEL_OF = {"WIN": "単勝", "PLACE": "複勝", "QUINELLA": "馬連", "QUINELLA_PLACE": "ワイド"}
ALL_POOLS = ("WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE")


def page_types(pools: Sequence[str]) -> list[str]:
    """The distinct odds pages `pools` are printed on, b1 before b4 before b5."""
    return sorted({PAGE_OF[p] for p in pools})


@dataclass(frozen=True)
class NarMeeting:
    venue: str  # short code (VENUES)
    posts: tuple[str, ...]  # race 1's post first, JST HH:MM as printed
    nighter: bool = False


@dataclass(frozen=True)
class NarRace:
    day: date
    venue: str
    number: int
    post: str  # JST HH:MM
    win: dict[int, float]
    place: dict[int, tuple[float, float]]
    quinella: dict[tuple[int, int], float]
    wide: dict[tuple[int, int], tuple[float, float]]
    tickets: dict[str, int]  # pool -> tickets sold

    @property
    def netkeiba_id(self) -> str:
        return pg.netkeiba_id(self.day, VENUES[self.venue][0], self.number)

    @property
    def race_id(self) -> str:
        return f"{SOURCE}:{self.day.isoformat()}:{self.venue}:{self.number:02d}"

    @property
    def post_utc(self) -> datetime:
        h, m = (int(x) for x in self.post.split(":"))
        return datetime(self.day.year, self.day.month, self.day.day, h, m, tzinfo=JST).astimezone(
            UTC
        )

    def board(self, pool: str) -> set[tuple]:
        """(combination, odds, odds_hi, investment) of every row of `pool`, its total included."""
        if pool == "WIN":
            out = {(str(h), p, None, None) for h, p in self.win.items()}
        elif pool == "PLACE":
            out = {(str(h), lo, hi, None) for h, (lo, hi) in self.place.items()}
        elif pool == "QUINELLA":
            out = {(f"{a},{b}", p, None, None) for (a, b), p in self.quinella.items()}
        else:
            out = {(f"{a},{b}", lo, hi, None) for (a, b), (lo, hi) in self.wide.items()}
        return out | {(POOL, None, None, self.tickets[pool] * 100.0)}

    def expected(self, pools: Sequence[str]) -> list[tuple]:
        """(race_id, source_race_key, pool, combination, odds, odds_hi, investment), sorted."""
        rows = [
            (self.race_id, self.netkeiba_id, pool, *row)
            for pool in dict.fromkeys(pools)
            for row in self.board(pool)
        ]
        return sorted(rows, key=repr)


def make_race(day: date, venue: str, number: int, post: str) -> NarRace:
    k = day.toordinal() % 7 + list(VENUES).index(venue) * 3 + number
    runners = range(1, 5 + (number + k) % 4 + 1)  # 5 to 8 runners
    win = {h: round(1.4 + ((h * 7 + k) % 30) * 1.3, 1) for h in runners}
    place = {}
    for h in runners:
        low = round(1.1 + ((h * 3 + k) % 9) * 0.4, 1)
        place[h] = (low, round(low + 0.7, 1))
    quinella, wide = {}, {}
    for a in runners:
        for b in runners:
            if a < b:
                quinella[(a, b)] = round(2.5 + ((a * 11 + b * 5 + k) % 90) * 1.7, 1)
                low = round(1.3 + ((a * 5 + b * 3 + k) % 50) * 0.9, 1)
                wide[(a, b)] = (low, round(low + 1.1, 1))
    tickets = {pool: 1_000 + 37 * k + 11 * number + 503 * i for i, pool in enumerate(ALL_POOLS)}
    return NarRace(day, venue, number, post, win, place, quinella, wide, tickets)


def rows_of(rows: Sequence[Mapping[str, Any]]) -> list[tuple]:
    """Capture or log rows in NarRace.expected's shape, sorted."""
    out = [
        (
            r["race_id"],
            r.get("source_race_key"),
            r["pool"],
            r["combination"],
            r.get("odds"),
            r.get("odds_hi"),
            r.get("investment"),
        )
        for r in rows
    ]
    return sorted(out, key=repr)


class NarWorld:
    def __init__(
        self,
        days: Mapping[date, Sequence[NarMeeting]],
        *,
        before_sales: Sequence[tuple[date, str, int]] = (),
    ):
        self.days = {d: list(ms) for d, ms in days.items()}
        self.before_sales = set(before_sales)

    def meeting(self, day: date, venue: str) -> NarMeeting:
        return next(m for m in self.days.get(day, []) if m.venue == venue)

    def races(self, day: date, venue: str) -> list[NarRace]:
        m = self.meeting(day, venue)
        return [make_race(day, venue, n, post) for n, post in enumerate(m.posts, start=1)]

    def race(self, day: date, venue: str, number: int) -> NarRace:
        return self.races(day, venue)[number - 1]

    def by_id(self, netkeiba_id: str) -> NarRace | None:
        for day, meetings in self.days.items():
            for m in meetings:
                for race in self.races(day, m.venue):
                    if race.netkeiba_id == netkeiba_id:
                        return race
        return None

    def expected_post_times(self, day: date, venue: str) -> dict[int, datetime]:
        return {r.number: r.post_utc for r in self.races(day, venue)}

    def race_list(self, day: date) -> bytes:
        meetings = [
            pg.Meeting(
                VENUES[m.venue][0],
                VENUES[m.venue][1],
                [pg.Race(n, post) for n, post in enumerate(m.posts, start=1)],
                nighter=m.nighter,
            )
            for m in self.days.get(day, [])
        ]
        return pg.race_list(day, meetings)


def odds_page(race: NarRace, page_type: str, *, as_id: str | None = None) -> bytes:
    """The race's `b1`, `b4` or `b5` page; `as_id` prints another race id on it (its links)."""
    rid = as_id or race.netkeiba_id
    if page_type == "b1":
        runners = {h: (race.win[h], race.place[h]) for h in race.win}
        votes = [(LABEL_OF[p], race.tickets[p]) for p in ("WIN", "PLACE")]
        return pg.odds_b1(rid, runners, votes=votes)
    if page_type == "b4":
        votes = [(LABEL_OF["QUINELLA"], race.tickets["QUINELLA"])]
        return pg.odds_pairs(rid, dict(race.quinella), label="馬連", votes=votes)
    if page_type == "b5":
        votes = [(LABEL_OF["QUINELLA_PLACE"], race.tickets["QUINELLA_PLACE"])]
        return pg.odds_pairs(rid, dict(race.wide), label="ワイド", votes=votes)
    raise ValueError(page_type)


@dataclass
class Seen:
    method: str
    host: str
    path: str
    params: dict[str, str]
    status: int

    @property
    def served(self) -> bool:
        return self.status == 200

    @property
    def key(self) -> tuple:
        """('race_list', YYYYMMDD) | ('odds', race id, type) | ('other', path)."""
        if self.path == RACE_LIST_PATH:
            return ("race_list", self.params.get("kaisai_date"))
        if self.path in ODDS_PATHS:
            return ("odds", self.params.get("race_id"), self.params.get("type"))
        return ("other", self.path)


@dataclass
class NarSite:
    world: NarWorld
    refuse_from: int | None = None
    replace: dict[tuple[str, str], bytes] = field(default_factory=dict)
    seen: list[Seen] = field(default_factory=list)

    def _answer(self, request: httpx.Request, params: dict[str, str]) -> tuple[int, bytes]:
        if self.refuse_from is not None and len(self.seen) + 1 >= self.refuse_from:
            return 403, b"<html><body>403 Forbidden</body></html>"
        if request.method != "GET" or request.url.host != HOST:
            return 404, b"not found"
        if request.url.path == RACE_LIST_PATH:
            try:
                day = datetime.strptime(params.get("kaisai_date", ""), "%Y%m%d").date()
            except ValueError:
                return 404, b"not found"
            return 200, self.world.race_list(day)
        if request.url.path in ODDS_PATHS:
            rid, page_type = params.get("race_id", ""), params.get("type", "")
            if (rid, page_type) in self.replace:
                return 200, self.replace[(rid, page_type)]
            race = self.world.by_id(rid)
            if race is None or page_type not in ("b1", "b4", "b5"):
                return 404, b"not found"
            if (race.day, race.venue, race.number) in self.world.before_sales:
                forecast = {h: price for h, price in race.win.items()}
                return 200, pg.odds_b1_before_sales(rid, forecast)
            return 200, odds_page(race, page_type)
        return 404, b"not found"

    def handler(self, request: httpx.Request) -> httpx.Response:
        params = {k: v for k, v in request.url.params.multi_items()}
        status, body = self._answer(request, params)
        self.seen.append(Seen(request.method, request.url.host, request.url.path, params, status))
        return httpx.Response(
            status, content=body, headers={"Content-Type": "text/html; charset=UTF-8"}
        )

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def keys(self) -> list[tuple]:
        return [s.key for s in self.seen]


def make_fetcher(site: NarSite) -> RecordingFetcher:
    """The real Fetcher for source "nar", over the double; nothing waits."""
    return RecordingFetcher(SOURCE, http=HTTP, transport=site.transport, sleep=lambda _s: None)
