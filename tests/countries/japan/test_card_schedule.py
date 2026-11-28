"""The cadence of the race-card (出馬表) captures in the logger's day (ticket #426).

Contract pinned here (the coder follows it), in `horsetrack.countries.japan.logger.schedule`:

- The schedule's `cards` block (logger.yaml, JRA only, like `tier2`):
  `{grid_min: [minutes before post, ...], requests: {navigation: N, per_race: M}}`. Without it,
  no card is captured and the day runs exactly as before.
- Site gains `capture_cards(venue, races) -> CardCapture` and Store `write_cards(venue, capture)
  -> rows written`. A site or store without its method (one from before #426): no card captures,
  no error, the day as before.
- Grid point g of a race falls due at post - g and belongs to the stretch until the next smaller
  grid value (the post after the last), as tier 2's. Each race's card is captured once per grid
  point, by a capture that starts within its stretch; a venue's races due in a round go in one
  `capture_cards` call. Every successful capture is written with `store.write_cards` (never
  `store.write`). The grid applies in the day-before window's rounds (#358) too, so a grid value
  whose stretch reaches back into the evening before gives each race a card captured that night.
- Card captures spend only what is left of a round's budget (every_s x rps) after tier 1, the
  results polls and tier 2, costing navigation + per_race x races; so tier 1's and tier 2's
  captures (their calls, races and times) and every VenueSummary count are the same with the
  `cards` block as without it.
- A failed card capture (WrongMeeting, NotListed, FetchError, ValueError) is logged and never
  touches the odds: no failure or wrong_meeting counted, no venue stopped. A CircuitOpen from it
  is a block, as from any site call: no site call after it and every venue stopped
  'circuit_open'.

Not pinned: whether a failed card capture is retried in a later round of its stretch, the order of
the card calls in a round, logging, DaySummary.requests. Everything is synthetic: no HTTP, no real
sleep; the fake clock moves only on sleep().
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from horsetrack.core.ingest.fetcher import CircuitOpen, FetchError
from horsetrack.countries.japan.jra.capture import MeetingCapture, NotListed, WrongMeeting

# isort: split
# The names under test get a block of their own: while they are missing, they are the import error
# a red run reports, and the order lints the same before and after they exist.
from horsetrack.countries.japan.jra.capture import CardCapture
from horsetrack.countries.japan.logger.schedule import run_day

UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
SAT = date(2026, 11, 28)
EVE = SAT - timedelta(days=1)
HOST = "www.jra.go.jp"
EPS = 1e-6

CONFIG = {
    "pools": ["WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE", "BRACKET_QUINELLA"],
    "min_poll_interval_s": 20,
    "windows": [{"from": "07:00", "every_s": 300}, {"from": "09:30", "every_s": 20}],
    "stop": "16:45",
    "after_last_race_min": 10,
    "hot_races": {"within_min": 20, "after_post_min": 5, "all_every_s": 300},
    "requests": {"navigation": 2, "per_race": 4},
    "max_wrong_meeting": 3,
    "on_block": {"cooldown_min": None},
}
TIER2 = {
    "pools": ["FORECAST", "TRIO", "TIERCE"],
    "grid_min": [60, 30, 16, 5, 2, 1],
    "requests": {"navigation": 2, "per_race": 3},
}
CARDS = {"grid_min": [120, 45, 10, 3], "requests": {"navigation": 2, "per_race": 1}}
DAY_BEFORE = {"probe_from": "17:00", "probe_until": "21:00", "probe_every_s": 60, "every_s": 300}


def config(*, cards=CARDS, tier2=None, day_before=None) -> dict:
    cfg = copy.deepcopy(CONFIG)
    for key, value in (("cards", cards), ("tier2", tier2), ("day_before", day_before)):
        if value is not None:
            cfg[key] = copy.deepcopy(value)
    return cfg


def jst(h: int, m: int = 0, s: int = 0, day: date = SAT) -> datetime:
    return datetime(day.year, day.month, day.day, h, m, s, tzinfo=JST).astimezone(UTC)


def every_s_at(t: datetime, cfg: dict) -> float | None:
    current = None
    for w in sorted(cfg["windows"], key=lambda w: w["from"]):
        h, m = (int(x) for x in w["from"].split(":"))
        if jst(h, m) <= t:
            current = w["every_s"]
    return current


def meetings(firsts: dict[str, tuple[int, int]], races: int) -> dict[str, dict[int, datetime]]:
    """Each venue's `races` races 30 minutes apart from its first post (Tokyo time)."""
    return {
        v: {n: jst(*first) + timedelta(minutes=30 * (n - 1)) for n in range(1, races + 1)}
        for v, first in firsts.items()
    }


THREE = meetings({"TOK": (10, 0), "KYO": (10, 10), "FKS": (10, 20)}, 4)
TWO = meetings({"NAK": (10, 5), "HSN": (11, 0)}, 6)


def stretch(post: datetime, t: datetime, grid: list[int]) -> int | None:
    """The grid value whose stretch [post - g, post - next smaller g) holds `t`, or None."""
    values = sorted(set(grid), reverse=True)
    for i, g in enumerate(values):
        nxt = values[i + 1] if i + 1 < len(values) else 0
        if post - timedelta(minutes=g) <= t < post - timedelta(minutes=nxt):
            return g
    return None


# ---------------------------------------------------------------- fakes


class Runaway(BaseException):
    """run_day kept going where a day cannot."""


class Clock:
    def __init__(self, start: datetime):
        self.t = start
        self.limit = start + timedelta(hours=40)
        self.reads = 0

    def now(self) -> datetime:
        self.reads += 1
        if self.reads > 200_000:
            raise Runaway("now() read 200000 times without a sleep: a busy loop")
        return self.t

    def sleep(self, seconds: float) -> None:
        if seconds < -EPS:
            raise Runaway(f"sleep({seconds}): a negative wait")
        self.reads = 0
        self.t += timedelta(seconds=max(seconds, 0.0))
        if self.t > self.limit:
            raise Runaway(f"run_day still sleeping at {self.t.isoformat()}")


@dataclass
class Call:
    t: datetime
    kind: str  # discover | post_times | sales_open | capture | tier2 | cards
    venue: str | None
    races: tuple[int, ...] | None
    requests: int
    ok: bool = True


def rid(venue: str, n: int) -> str:
    return f"jra:{SAT.isoformat()}:{venue}:{n:02d}"


@dataclass
class OddsSite:
    """A site from before #426: odds, tier 2, post times, the day-before probe; no cards."""

    clock: Clock
    posts: dict[str, dict[int, datetime]]
    cfg: dict
    requests_per_second: float
    sales_from: datetime | None = None
    source: str = "jra"
    host: str = HOST
    calls: list[Call] = field(default_factory=list)

    def _note(self, kind, venue, races, requests, ok=True) -> datetime:
        t = self.clock.now()
        self.calls.append(
            Call(t, kind, venue, None if races is None else tuple(races), requests, ok)
        )
        return t

    def discover(self) -> list[str]:
        self._note("discover", None, None, 1)
        return list(self.posts)

    def post_times(self, venue: str) -> dict[int, datetime]:
        self._note("post_times", venue, None, 2)
        return dict(self.posts[venue])

    def sales_open(self) -> bool:
        t = self._note("sales_open", None, None, 1)
        return self.sales_from is not None and t >= self.sales_from

    def capture(self, venue: str, races: list[int] | None) -> MeetingCapture:
        races = sorted(self.posts[venue]) if races is None else list(races)
        req = self.cfg["requests"]
        cost = req["navigation"] + req["per_race"] * len(races)
        t = self._note("capture", venue, races, cost)
        rows = [{"race_id": rid(venue, n), "pool": "WIN", "combination": "1", "odds": 2.0,
                 "snapshot_utc": t} for n in races]  # fmt: skip
        return MeetingCapture(rows, t, cost)

    def capture_tier2(self, venue: str, races: list[int]) -> MeetingCapture:
        req = self.cfg["tier2"]["requests"]
        cost = req["navigation"] + req["per_race"] * len(races)
        t = self._note("tier2", venue, races, cost)
        rows = [{"race_id": rid(venue, n), "pool": "TRIO", "combination": "1,2,3", "odds": 9.0,
                 "snapshot_utc": t} for n in races]  # fmt: skip
        return MeetingCapture(rows, t, cost)

    def of(self, *kinds: str) -> list[Call]:
        return [c for c in self.calls if c.kind in kinds]


@dataclass
class CardSite(OddsSite):
    """The site with capture_cards (#426); `fail(t, venue)` gives an exception to raise."""

    fail: object = None
    returned: list[CardCapture] = field(default_factory=list)

    def capture_cards(self, venue: str, races: list[int]) -> CardCapture:
        req = self.cfg["cards"]["requests"]
        cost = req["navigation"] + req["per_race"] * len(races)
        exc = self.fail(self.clock.now(), venue) if self.fail is not None else None
        t = self._note("cards", venue, races, cost, ok=exc is None)
        if exc is not None:
            raise exc
        at = {"fetched_utc": t, "snapshot_utc": t}
        cap = CardCapture(
            races=[{"race_id": rid(venue, n), "field_size": 8, **at} for n in races],
            runners=[
                {"race_id": rid(venue, n), "horse_no": h, "is_scratched": False, **at}
                for n in races
                for h in range(1, 9)
            ],  # fmt: skip
            snapshot_utc=t,
            requests=cost,
        )
        self.returned.append(cap)
        return cap


@dataclass
class OddsStore:
    """A store from before #426."""

    writes: list[tuple[str, object]] = field(default_factory=list)
    post_writes: list[str] = field(default_factory=list)

    def write(self, venue: str, capture) -> int:
        self.writes.append((venue, capture))
        return len(capture.rows)

    def write_post_times(self, venue: str, times) -> None:
        self.post_writes.append(venue)


@dataclass
class CardStore(OddsStore):
    cards: list[tuple[str, CardCapture]] = field(default_factory=list)

    def write_cards(self, venue: str, capture: CardCapture) -> int:
        self.cards.append((venue, capture))
        return len(capture.races) + len(capture.runners)


def run(
    posts,
    cfg: dict,
    rps: float,
    marker: Path,
    *,
    site_cls=CardSite,
    store_cls=CardStore,
    start: datetime | None = None,
    **site_kw,
):
    clock = Clock(start or jst(6, 30))
    site = site_cls(clock, posts, cfg, rps, **site_kw)
    store = store_cls()
    summary = run_day(SAT, site, store, config=cfg, now=clock.now, sleep=clock.sleep,
                      block_marker=marker, venues=list(posts))  # fmt: skip
    return summary, site, store


def odds_calls(site: OddsSite) -> list[tuple]:
    """Tier 1's and tier 2's captures: (time, kind, venue, races), in order."""
    return [(c.t, c.kind, c.venue, c.races) for c in site.of("capture", "tier2")]


def counts(summary) -> dict:
    return {
        v: (s.snapshots, s.rows, s.failures, s.wrong_meeting, s.stopped, s.tier2_skipped)
        for v, s in summary.venues.items()
    }


def card_points(site: CardSite, posts, grid) -> Counter:
    """(venue, race, grid value) of every race of every successful card capture."""
    out: Counter = Counter()
    for c in site.of("cards"):
        if not c.ok:
            continue
        assert c.races and len(set(c.races)) == len(c.races), c
        for n in c.races:
            g = stretch(posts[c.venue][n], c.t, grid)
            assert g is not None, f"{c.venue} race {n}'s card captured at {c.t}, in no stretch"
            out[(c.venue, n, g)] += 1
    return out


def every_point(posts, grid) -> Counter:
    return Counter({(v, n, g): 1 for v, races in posts.items() for n in races for g in grid})


# ---------------------------------------------------------------- the cadence


@pytest.mark.parametrize(
    "posts, grid",
    [
        (THREE, [120, 45, 10, 3]),
        (THREE, [60, 5]),
        (TWO, [200, 30, 2]),
        (TWO, [90, 20, 8, 4, 1]),
    ],
    ids=["three-4-points", "three-2-points", "two-3-points", "two-5-points"],
)
def test_each_race_card_is_captured_once_in_each_grid_stretch(tmp_path, posts, grid):
    cards = {**CARDS, "grid_min": grid}
    summary, site, store = run(posts, config(cards=cards), 10.0, tmp_path / "block.txt")
    assert card_points(site, posts, grid) == every_point(posts, grid)
    # each successful capture written once, to the card log, never to the odds log
    assert [c for _, c in store.cards] == site.returned
    assert [v for v, _ in store.cards] == [c.venue for c in site.of("cards")]
    assert all(isinstance(c, MeetingCapture) for _, c in store.writes)
    assert {v.stopped for v in summary.venues.values()} == {"ended"}


@pytest.mark.parametrize("rps", [0.5, 1.0, 10.0])
@pytest.mark.parametrize("tier2", [None, TIER2], ids=["tier1", "tier1+tier2"])
def test_the_odds_captures_are_the_same_with_the_card_captures(tmp_path, rps, tier2):
    without = config(cards=None, tier2=tier2)
    assert "cards" not in without
    off, off_site, off_store = run(THREE, without, rps, tmp_path / "off.txt")
    on, on_site, on_store = run(THREE, config(tier2=tier2), rps, tmp_path / "on.txt")
    assert off_site.of("cards") == [] and off_store.cards == []
    assert on_site.of("cards"), "the cards block on, and no card captured"
    assert odds_calls(on_site) == odds_calls(off_site)
    assert counts(on) == counts(off)
    assert on.deferred == off.deferred
    assert len(on_store.writes) == len(off_store.writes)


@pytest.mark.parametrize("rps", [0.5, 1.0])
def test_card_captures_spend_only_what_is_left_of_the_round(tmp_path, rps):
    cfg = config(tier2=TIER2)
    _, site, _ = run(THREE, cfg, rps, tmp_path / "block.txt")
    assert site.of("cards")
    by_round: dict[datetime, int] = defaultdict(int)
    for c in site.of("capture", "tier2", "cards"):  # nothing moves the clock inside a round
        by_round[c.t] += c.requests
    for t, spent in by_round.items():
        budget = every_s_at(t, cfg) * rps
        assert spent <= budget + EPS, f"round at {t}: {spent} requests, budget {budget}"


# ---------------------------------------------------------------- failures and old sites


@pytest.mark.parametrize(
    "make",
    [
        lambda: WrongMeeting("a card page of another meeting"),
        lambda: NotListed("not on the card menu"),
        lambda: FetchError("HTTP 500", http_status=500),
        lambda: ValueError("not a JRADB page with a date line"),
    ],
    ids=["wrong-meeting", "not-listed", "fetch-error", "error-page"],
)
@pytest.mark.parametrize("who", ["TOK", "every venue"])
def test_a_failed_card_capture_never_touches_the_odds(tmp_path, make, who):
    def fail(t, venue):
        return make() if who in ("every venue", venue) else None

    cfg = config(tier2=TIER2)
    off, off_site, _ = run(THREE, config(cards=None, tier2=TIER2), 1.0, tmp_path / "off.txt")
    on, on_site, store = run(THREE, cfg, 1.0, tmp_path / "on.txt", fail=fail)
    assert [c for c in on_site.of("cards") if not c.ok], "no card capture failed"
    assert odds_calls(on_site) == odds_calls(off_site)
    assert counts(on) == counts(off)
    assert all(v.failures == 0 and v.wrong_meeting == 0 for v in on.venues.values())
    failed = {c.venue for c in on_site.of("cards") if not c.ok}
    assert not failed & {v for v, _ in store.cards}, "a failed card capture was written"


def test_a_block_on_a_card_capture_ends_the_day(tmp_path):
    blocked_at: list[datetime] = []

    def fail(t, venue):
        if not blocked_at:
            blocked_at.append(t)
        return CircuitOpen("HTTP 403", http_status=403, host=HOST)

    marker = tmp_path / "block.txt"
    summary, site, _ = run(THREE, config(), 1.0, marker, fail=fail)
    (first,) = [i for i, c in enumerate(site.calls) if c.kind == "cards" and not c.ok][:1]
    assert site.calls[first + 1 :] == [], "a site call after the block"
    assert {v.stopped for v in summary.venues.values()} == {"circuit_open"}
    text = marker.read_text(encoding="utf-8")
    assert HOST in text and SAT.isoformat() in text


@pytest.mark.parametrize(
    "site_cls, store_cls", [(OddsSite, CardStore), (CardSite, OddsStore)],
    ids=["site-without-capture-cards", "store-without-write-cards"],
)  # fmt: skip
def test_a_site_or_store_from_before_the_cards_logs_the_odds_as_before(
    tmp_path, site_cls, store_cls
):
    off, off_site, _ = run(THREE, config(cards=None, tier2=TIER2), 1.0, tmp_path / "off.txt")
    on, on_site, _ = run(THREE, config(tier2=TIER2), 1.0, tmp_path / "on.txt",
                         site_cls=site_cls, store_cls=store_cls)  # fmt: skip
    assert on_site.of("cards") == [], "a card captured that could not be written"
    assert odds_calls(on_site) == odds_calls(off_site)
    assert counts(on) == counts(off)


# ---------------------------------------------------------------- the evening before (#358)


def test_a_grid_stretch_reaching_the_evening_before_gives_each_race_a_card_that_night(tmp_path):
    grid = [900, 10]  # 15 h before the off: from 19:00 the evening before for a 10:00 race
    cfg = config(cards={**CARDS, "grid_min": grid}, day_before=DAY_BEFORE)
    opens = jst(18, 30, day=EVE)
    first_window = jst(7, 0)
    summary, site, _ = run(THREE, cfg, 2.0, tmp_path / "block.txt", start=jst(16, 50, day=EVE),
                           sales_from=opens)  # fmt: skip
    assert summary.sales_open_utc == opens
    assert card_points(site, THREE, grid) == every_point(THREE, grid)
    for c in site.of("cards"):
        assert c.t >= opens, f"a card captured at {c.t}, before the sales opened"
    night = {(c.venue, n) for c in site.of("cards") if c.t < first_window for n in c.races}
    assert night == {(v, n) for v, races in THREE.items() for n in races}
