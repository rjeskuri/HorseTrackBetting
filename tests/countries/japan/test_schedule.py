"""Scheduling every venue's captures under one request budget (ticket #24), fixing HKTrack bugs
(f) (after a 403/429 the supervisor retried about every 30 s until the stop time) and (g) (a full
capture is 2 + 3 x races requests and nothing budgeted them across venues).

Contract pinned here (the coder follows it):

Config
- `countries/japan/config/logger.yaml`, read with `load_country_yaml("japan", "logger")`, holds the
  outline's values (times quoted: "07:00" is a string). The `config` that plan_round and run_day
  take is a mapping of that shape: that dict, or a copy with some values changed. The tests change
  values (windows, stop, hot_races, requests, max_wrong_meeting, on_block, after_last_race_min), so
  none of them may be a constant in the code.

Time
- `now()` returns aware UTC datetimes and post times are aware UTC. Window `from` and `stop` are
  Asia/Tokyo wall-clock times on the race date (for plan_round: the Tokyo date of `now`). plan_round
  takes any aware `now`: the same instant gives the same plan.

plan_round(now, venues, config, rps) -> list[Job]   (horsetrack.countries.japan.logger.schedule)
- `venues`: a list of `VenueState(code=..., post_times=..., last_capture=None)`: post_times maps
  race number -> aware UTC post; last_capture is when this venue's latest capture started (None:
  never). Any other field has a default meaning "no full cycle run yet, not stopped", so a fresh
  VenueState has its whole full cycle (every race of the meeting) due.
- A Job has `venue`, `races` (a non-empty list of race numbers, no repeats; None means every race
  and is only for a venue whose post times are unknown) and `requests` == navigation + per_race x
  len(races). At most one Job per venue per round: a venue's hot and cycle races share one capture,
  so navigation is charged once.
- Budget = every_s of the window in force at `now` x rps. Before the first window: [] always.
- A race is hot when post - within_min <= now <= post + after_post_min (closed at both ends).
  With a fresh VenueState (no race captured yet), and always without `hot_races.every_s` and
  `floor_s` (this file's CONFIG): hot races are taken first, nearest post first, race by race
  across venues, each one that still fits (a hot race no plan includes would not have fit on top
  of the planned hot races). Then the spare budget fills the venues' due full cycles race by race
  until no further race fits.
- The owner's rule (#387, 2026-11-29; logger.yaml `hot_races.every_s` 60, `floor_s` 300,
  `venue_rates`, `on_block`): each hot race at least every 60 s, every other race at least every
  300 s, a faster JRA rate on three-venue days, and a block that backs off instead of ending the
  day. test_five_minute_floor.py pins it (VenueState.captured, the must races, the race day).
- The sum of Job.requests is <= budget, except when the budget is below one hot race's cost
  (navigation + per_race): then the plan is that single race alone ("one hot job always runs").
- A venue whose last_capture is less than min_poll_interval_s before `now` gets no Job (exactly
  min_poll_interval_s is allowed). The Job holding the nearest hot race comes first.

site (FakeSite here): `host`, `requests_per_second`, `discover() -> list[str]`,
`post_times(venue) -> dict[int, datetime]` ({} = unknown),
`capture(venue, races) -> MeetingCapture`; each may raise WrongMeeting, NotListed, CircuitOpen or
FetchError.
store (FakeStore here): `write(venue, capture) -> rows written`, `write_post_times(venue, times)`.

run_day(race_date, site, store, *, config, now, sleep, block_marker, venues=None) -> DaySummary
- `now()` is the only clock and `sleep(seconds)` the only wait (the fakes: sleep moves the clock, a
  capture takes no time). Rounds start every_s of the window in force apart (start to start); each
  round runs plan_round's jobs as `site.capture(job.venue, job.races)`.
- Nothing is sent before the first window, discover() included. venues=None: discover() once;
  venues given: no discover(). No venue: return at once.
- post_times(venue) is read before the venue's first capture and again with each full cycle (not
  every round), and each read is written with store.write_post_times; every successful capture is
  written with store.write.
- A re-read adds to what is known, it does not replace it: the real reader leaves out races already
  off (a 発走済 cell), so a race missing from a re-read keeps its known post time and is still
  captured through post + after_post_min, and still counts for the venue's end. A time the re-read
  does give replaces the known one (a post moved later).
- Full cycles run back to back: the next starts all_every_s after the last one started, or in the
  round after it completes if that is later.
- A venue ends at its last post (latest post times) + after_last_race_min when post times are known,
  even past `stop`; else at `stop`. Nothing is sent for it from then on; run_day returns once every
  venue has ended or stopped.
- WrongMeeting: counted; max_wrong_meeting in a row stops that venue (stopped 'wrong_meeting'), the
  others go on. NotListed before the venue's first post: retried in later rounds, not a failure.
  After it, NotListed (like FetchError) is a failure and the venue goes on.
- CircuitOpen from any site call: with on_block.cooldown_min null, no further site call in this run,
  every venue's stopped == 'circuit_open', and `block_marker` (a file path) is written: a text file
  naming the host and the race's Tokyo date (ISO, e.g. 2026-11-28). A run that finds a marker for
  this host and today's date sends nothing (no discover either); a marker for another date or host
  does not block. With cooldown_min N: nothing for N minutes, then the run resumes once; the next
  CircuitOpen ends it. (#387: logger.yaml's JRA on_block is a back-off ladder, its own key, under
  which no block ends the day and a marker holds a run only for the pause after its latest
  block; test_five_minute_floor.py pins it. This file's config() keeps the rules above.)
- DaySummary: `venues` {code: VenueSummary}, `requests` (at least the captures' requests, at most
  what the site sent), `deferred` (race captures a round left out for budget: 0 when the budget
  never binds, > 0 when it does). VenueSummary: snapshots (successful captures), rows (sum of
  store.write's returns), failures, wrong_meeting, stopped, longest_cycle_s (the longest full
  cycle, first capture start to last capture end).

Not pinned: logging, the marker's format beyond the host and date in its text, the stopped value of
a venue that ran to its end, which venue's cycle gets spare budget, the race order inside a cycle,
the order of jobs after the first, whether post-time reads are charged to a round, and CircuitOpen
without an HTTP status. Everything is synthetic: no HTTP, no real sleep, no scraped page.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from horsetrack.core.config import load_country_yaml
from horsetrack.core.ingest.fetcher import CircuitOpen, FetchError
from horsetrack.countries.japan.jra.capture import (
    ODDS_POOLS,
    MeetingCapture,
    NotListed,
    WrongMeeting,
)
from horsetrack.countries.japan.logger.health import MAX_GAP_S

# isort: split
# The module under test gets a block of its own: while it is missing, it is the import error a
# red run reports, and the order lints the same before and after it exists.
from horsetrack.countries.japan.logger.schedule import VenueState, plan_round, run_day

UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
SAT, SUN = date(2026, 11, 28), date(2026, 11, 29)
HOST = "www.jra.go.jp"
EPS = 1e-6

CONFIG = {
    "pools": list(ODDS_POOLS),
    "min_poll_interval_s": 20,
    "windows": [{"from": "07:00", "every_s": 300}, {"from": "09:30", "every_s": 20}],
    "stop": "16:45",
    "after_last_race_min": 10,
    "hot_races": {"within_min": 20, "after_post_min": 5, "all_every_s": 300},
    "requests": {"navigation": 2, "per_race": 3},
    "max_wrong_meeting": 3,
    "on_block": {"cooldown_min": None},
}


def config(**changes) -> dict:
    """The outline's config with some values changed (a dict value is merged into the default)."""
    cfg = copy.deepcopy(CONFIG)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **copy.deepcopy(value)}
        else:
            cfg[key] = copy.deepcopy(value)
    return cfg


# ---------------------------------------------------------------- time and expected values


def jst(h: int, m: int = 0, s: int = 0, day: date = SAT) -> datetime:
    """The aware UTC instant of a Tokyo wall-clock time on `day`."""
    return datetime(day.year, day.month, day.day, h, m, s, tzinfo=JST).astimezone(UTC)


def wall(hhmm: str, day: date) -> datetime:
    h, m = (int(part) for part in hhmm.split(":"))
    return jst(h, m, day=day)


def every_s_at(t: datetime, cfg: dict) -> float | None:
    """every_s of the window in force at `t` (on t's Tokyo date); None before the first window."""
    day = t.astimezone(JST).date()
    current = None
    for window in sorted(cfg["windows"], key=lambda w: wall(w["from"], day)):
        if wall(window["from"], day) <= t:
            current = window["every_s"]
    return current


def budget_at(t: datetime, cfg: dict, rps: float) -> float | None:
    every = every_s_at(t, cfg)
    return None if every is None else every * rps


def cost(n_races: int, cfg: dict) -> int:
    return cfg["requests"]["navigation"] + cfg["requests"]["per_race"] * n_races


def is_hot(post: datetime, t: datetime, cfg: dict) -> bool:
    hot = cfg["hot_races"]
    before = timedelta(minutes=hot["within_min"])
    after = timedelta(minutes=hot["after_post_min"])
    return post - before <= t <= post + after


def secs(delta: timedelta) -> float:
    return delta.total_seconds()


def gaps_s(times: list[datetime]) -> list[float]:
    """Seconds between consecutive times."""
    return [secs(times[i + 1] - times[i]) for i in range(len(times) - 1)]


def hhmm(t: datetime) -> str:
    return t.astimezone(JST).strftime("%H:%M:%S JST")


def staggered(races: int = 12, day: date = SAT) -> dict[str, dict[int, datetime]]:
    """TOK, KYO, FKS: `races` races 30 minutes apart, first posts 10:00, 10:10, 10:20 Tokyo time."""
    firsts = {"TOK": (10, 0), "KYO": (10, 10), "FKS": (10, 20)}
    return {
        venue: {
            n: jst(h, m, day=day) + timedelta(minutes=30 * (n - 1)) for n in range(1, races + 1)
        }
        for venue, (h, m) in firsts.items()
    }


# ---------------------------------------------------------------- fakes


class Runaway(BaseException):
    """run_day kept going where a day cannot (a BaseException: no `except Exception` hides it)."""


class Clock:
    """The fake clock: only sleep() moves it."""

    def __init__(self, start: datetime):
        self.t = start
        self.limit = start + timedelta(hours=24)
        self.reads_since_sleep = 0
        self.zero_sleeps = 0

    def now(self) -> datetime:
        self.reads_since_sleep += 1
        if self.reads_since_sleep > 200_000:
            raise Runaway("now() read 200000 times without a sleep: a busy loop")
        return self.t

    def sleep(self, seconds: float) -> None:
        if seconds < -EPS:
            raise Runaway(f"sleep({seconds}): a negative wait")
        self.reads_since_sleep = 0
        self.zero_sleeps = self.zero_sleeps + 1 if seconds <= 0 else 0
        if self.zero_sleeps > 10_000:
            raise Runaway("10000 zero-length sleeps in a row")
        self.t += timedelta(seconds=max(seconds, 0.0))
        if self.t > self.limit:
            raise Runaway(f"run_day still sleeping at {self.t.isoformat()}: it never returned")


@dataclass
class Call:
    t: datetime
    kind: str  # discover | post_times | capture
    venue: str | None
    races: tuple[int, ...] | None = None  # capture: the races asked for (None = every race)
    error: str | None = None  # class name of the exception the fake raised
    requests: int = 0


Rule = Callable[[str, str | None, int, datetime], BaseException | None]


class FakeSite:
    """One host's JRADB as the scheduler sees it. Every call is recorded at the fake clock's time.
    A capture costs navigation + per_race x races requests. `fail(kind, venue, n, t)`, n being the
    1-based count of that kind of call for that venue, returns an exception to raise, or None."""

    def __init__(
        self,
        clock: Clock,
        meetings: dict[str, dict[int, datetime]],
        *,
        rps: float = 1.0,
        nav: int = 2,
        per_race: int = 3,
        host: str = HOST,
        fail: Rule | None = None,
        post_times: Callable[[str, datetime], dict[int, datetime]] | None = None,
    ):
        self.clock = clock
        self.meetings = meetings
        self.host = host
        self.requests_per_second = rps
        self.nav, self.per_race = nav, per_race
        self.fail = fail or (lambda kind, venue, n, t: None)
        self.shown_post_times = post_times or (lambda venue, t: dict(self.meetings[venue]))
        self.calls: list[Call] = []
        self.requests = 0
        self.defects: list[str] = []
        self._counts: Counter = Counter()

    def _call(self, kind: str, venue: str | None, races=None) -> tuple[Call, BaseException | None]:
        call = Call(self.clock.t, kind, venue, races)
        self.calls.append(call)
        self._counts[(kind, venue)] += 1
        if venue is not None and venue not in self.meetings:
            self.defects.append(f"{kind} for {venue}, which has no meeting today")
        return call, self.fail(kind, venue, self._counts[(kind, venue)], call.t)

    def _spend(self, call: Call, requests: int, exc: BaseException | None = None) -> None:
        call.requests = requests
        self.requests += requests
        if exc is not None:
            call.error = type(exc).__name__
            raise exc

    def discover(self) -> list[str]:
        call, exc = self._call("discover", None)
        self._spend(call, 1, exc)
        return list(self.meetings)

    def post_times(self, venue: str) -> dict[int, datetime]:
        call, exc = self._call("post_times", venue)
        self._spend(call, self.nav, exc)
        return self.shown_post_times(venue, call.t)

    def capture(self, venue: str, races) -> MeetingCapture:
        asked = None if races is None else tuple(races)
        call, exc = self._call("capture", venue, asked)
        every = sorted(self.meetings.get(venue, {}))
        if asked is not None:
            if not asked:
                self.defects.append(f"{hhmm(call.t)} capture of {venue} asked for no race")
            if len(set(asked)) != len(asked):
                self.defects.append(f"{hhmm(call.t)} capture of {venue} repeats a race: {asked}")
            if set(asked) - set(every):
                self.defects.append(f"{hhmm(call.t)} capture of {venue} asks for races {asked}")
        if exc is not None:
            self._spend(call, 1 if isinstance(exc, (NotListed, CircuitOpen)) else self.nav, exc)
        wanted = every if asked is None else sorted(set(asked) & set(every))
        spent = self.nav + self.per_race * len(wanted)
        self._spend(call, spent)
        day = call.t.astimezone(JST).date().isoformat()
        rows = [
            {"race_id": f"jra:{day}:{venue}:{n:02d}", "pool": pool, "snapshot_utc": call.t}
            for n in wanted
            for pool in ODDS_POOLS
        ]
        return MeetingCapture(rows, call.t, spent)


class FakeStore:
    def __init__(self):
        self.writes: list[tuple[str, MeetingCapture]] = []
        self.post_times: list[tuple[str, dict[int, datetime]]] = []

    def write(self, venue: str, capture: MeetingCapture) -> int:
        self.writes.append((venue, capture))
        return len(capture.rows)

    def write_post_times(self, venue: str, times: dict[int, datetime]) -> None:
        self.post_times.append((venue, dict(times)))


def on_calls(kind: str, venue: str | None, numbers: set[int], make: Callable[[], BaseException]):
    """Raise make() on the calls of `kind` for `venue` whose 1-based count is in `numbers`."""

    def rule(k, v, n, t):
        return make() if (k, v) == (kind, venue) and n in numbers else None

    return rule


def from_time(kind: str, venue: str | None, start: datetime, *excs: BaseException):
    """Raise excs[0], excs[1], ... on the first calls of `kind` for `venue` at or after `start`."""
    pending = list(excs)

    def rule(k, v, n, t):
        if (k, v) == (kind, venue) and t >= start and pending:
            return pending.pop(0)
        return None

    return rule


def rules(*parts):
    def fail(k, v, n, t):
        for part in parts:
            exc = part(k, v, n, t)
            if exc is not None:
                return exc
        return None

    return fail


def blocked(status: int) -> Callable[[], CircuitOpen]:
    return lambda: CircuitOpen(f"HTTP {status}", http_status=status, host=HOST)


def run(site: FakeSite, clock: Clock, marker: Path, *, cfg=None, day=SAT, venues=None):
    store = FakeStore()
    summary = run_day(
        day,
        site,
        store,
        config=cfg if cfg is not None else config(),
        now=clock.now,
        sleep=clock.sleep,
        block_marker=marker,
        venues=venues,
    )
    assert site.defects == [], site.defects
    return summary, store


# ---------------------------------------------------------------- reading what the fake saw


def caps(site: FakeSite, venue: str | None = None, ok: bool | None = None) -> list[Call]:
    return [
        c
        for c in site.calls
        if c.kind == "capture"
        and (venue is None or c.venue == venue)
        and (ok is None or (c.error is None) == ok)
    ]


def asked(call: Call, site: FakeSite) -> list[int]:
    return sorted(site.meetings[call.venue]) if call.races is None else sorted(call.races)


def assert_single_hot_exception(group: list[Call], site: FakeSite, cfg: dict, where: str) -> None:
    rps = site.requests_per_second
    budget = budget_at(group[0].t, cfg, rps)
    assert budget < cost(1, cfg), f"{where}: over the budget of {budget}"
    assert len(group) == 1, f"{where}: over the budget of {budget} with {len(group)} captures"
    (call,) = group
    races = asked(call, site)
    assert len(races) == 1, f"{where}: over the budget of {budget} with races {races}"
    post = site.meetings[call.venue][races[0]]
    assert is_hot(post, call.t, cfg), f"{where}: over the budget for a race that is not hot"


def assert_rounds_fit_budget(site: FakeSite, cfg: dict) -> None:
    """Every round (the captures at one instant) fits the budget of the window in force, or is the
    single hot job; and no every_s span inside one window holds more than one round's budget."""
    rps = site.requests_per_second
    rounds: dict[datetime, list[Call]] = defaultdict(list)
    for call in caps(site):
        rounds[call.t].append(call)
    instants = sorted(rounds)
    for t in instants:
        group = rounds[t]
        where = f"round at {hhmm(t)} (rps {rps})"
        budget = budget_at(t, cfg, rps)
        assert budget is not None, f"{where}: a capture before the first window"
        venues = [c.venue for c in group]
        assert len(venues) == len(set(venues)), f"{where}: a venue captured twice: {venues}"
        planned = sum(cost(len(asked(c, site)), cfg) for c in group)
        if planned > budget + EPS:
            assert_single_hot_exception(group, site, cfg, where)
    for i, start in enumerate(instants):
        every = every_s_at(start, cfg)
        end = start + timedelta(seconds=every)
        if every_s_at(end, cfg) != every:
            continue  # the span runs into the next window
        span = [c for t in instants[i:] if t < end for c in rounds[t]]
        planned = sum(cost(len(asked(c, site)), cfg) for c in span)
        if planned > every * rps + EPS:
            where = f"{every} s from {hhmm(start)} (rps {rps}): {planned} requests"
            assert_single_hot_exception(span, site, cfg, where)


def assert_hot_races_followed(site: FakeSite, cfg: dict, venues, every: float) -> None:
    """Each race of `venues` is captured through its hot window at most `every` seconds apart."""
    hot = cfg["hot_races"]
    for venue in venues:
        ok = caps(site, venue, ok=True)
        for n, post in site.meetings[venue].items():
            lo = post - timedelta(minutes=hot["within_min"])
            hi = post + timedelta(minutes=hot["after_post_min"])
            times = [c.t for c in ok if n in asked(c, site) and lo <= c.t <= hi]
            where = f"{venue} race {n} (post {hhmm(post)})"
            assert times, f"{where}: never captured in its hot window"
            assert secs(times[0] - lo) <= every + EPS, (
                f"{where}: first hot capture {hhmm(times[0])}"
            )
            assert secs(hi - times[-1]) <= every + EPS, (
                f"{where}: last hot capture {hhmm(times[-1])}"
            )
            gaps = gaps_s(times)
            assert max(gaps, default=0.0) <= every + EPS, f"{where}: a {max(gaps)} s gap while hot"


def assert_logged_until(site: FakeSite, venue: str, end: datetime) -> None:
    """No call for `venue` at or after `end`, and it was being captured up to one cycle before."""
    late = [c for c in site.calls if c.venue == venue and c.t >= end]
    assert late == [], f"{venue}: {len(late)} calls at or after its end {hhmm(end)}: {late[:3]}"
    ok = caps(site, venue, ok=True)
    assert ok, f"{venue}: never captured"
    assert secs(end - ok[-1].t) <= 320 + EPS, f"{venue}: last capture {hhmm(ok[-1].t)}"


def assert_summary_matches(summary, site: FakeSite, store: FakeStore) -> None:
    for venue, vs in summary.venues.items():
        ok = caps(site, venue, ok=True)
        written = [cap for v, cap in store.writes if v == venue]
        assert vs.snapshots == len(ok) == len(written), (venue, vs.snapshots, len(ok))
        assert vs.rows == sum(len(cap.rows) for cap in written), venue
    captured = sum(c.requests for c in caps(site, ok=True))
    assert captured <= summary.requests <= site.requests, (captured, summary.requests)


# ---------------------------------------------------------------- the shipped config


def test_logger_yaml_ships_the_outline_values():
    shipped = load_country_yaml("japan", "logger")
    # #358: bracket quinella is captured every round with tier 1, on a page of its own; #387: the
    # owner's 60 s hot grain and 300 s floor beside the hot window and the cycles as they were.
    # on_block is the owner's back-off since #387 (test_five_minute_floor.py pins it).
    want = config(
        pools=["WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE", "BRACKET_QUINELLA"],
        requests={"navigation": 2, "per_race": 4},
        hot_races={"every_s": 60, "floor_s": 300},
    )
    for key, value in want.items():
        if key == "on_block":
            continue
        assert shipped.get(key) == value, f"logger.yaml {key}: {shipped.get(key)!r}, not {value!r}"


# ---------------------------------------------------------------- plan_round


def test_hot_window_is_post_minus_20_to_post_plus_5_in_utc():
    posts = {1: jst(12, 0), 2: jst(12, 30), 3: jst(13, 0)}  # 03:00, 03:30, 04:00 UTC
    assert posts[2] == datetime(2026, 11, 28, 3, 30, tzinfo=UTC)
    # 20 s x 1/16 rps = 1.25 requests: no capture fits, so a plan can only be the one hot job
    rps = 0.0625
    cases = [
        (config(), (11, 39, 59), None),
        (config(), (11, 40, 0), 1),
        (config(), (12, 5, 0), 1),
        (config(), (12, 5, 1), None),
        (config(), (12, 9, 59), None),
        (config(), (12, 10, 0), 2),
        (config(), (12, 21, 30), 2),
        (config(), (12, 35, 0), 2),
        (config(), (12, 35, 1), None),
        (config(), (12, 40, 0), 3),
        (config(), (13, 5, 0), 3),
        (config(), (13, 5, 1), None),
        (config(hot_races={"within_min": 8, "after_post_min": 2}), (12, 21, 59), None),
        (config(hot_races={"within_min": 8, "after_post_min": 2}), (12, 22, 0), 2),
        (config(hot_races={"within_min": 8, "after_post_min": 2}), (12, 32, 0), 2),
        (config(hot_races={"within_min": 8, "after_post_min": 2}), (12, 32, 1), None),
    ]
    for cfg, hms, race in cases:
        at = jst(*hms)
        for now in (at, at.astimezone(JST)):
            plan = plan_round(now, [VenueState(code="TOK", post_times=posts)], cfg, rps)
            got = [(job.venue, list(job.races), job.requests) for job in plan]
            want = [] if race is None else [("TOK", [race], cost(1, cfg))]
            assert got == want, f"{now.isoformat()} {cfg['hot_races']}: {got}, expected {want}"


def _spaced(now: datetime, anchor: int, minutes: float) -> dict[int, datetime]:
    """12 races 30 minutes apart, race `anchor` posting `minutes` after `now`."""
    return {n: now + timedelta(minutes=minutes + 30 * (n - anchor)) for n in range(1, 13)}


def test_hot_races_come_first_nearest_post_first_when_the_budget_is_short():
    for cfg in (config(), config(hot_races={"every_s": 60, "floor_s": 300})):
        # With the owner's grain and floor (#387) too: no VenueState here has a capture, so every
        # hot race is due and #24's order stands (test_five_minute_floor.py has the captures).
        _hot_races_come_first(cfg)


def _hot_races_come_first(cfg: dict) -> None:
    now = jst(12, 0, 7)
    # (rps, hot races planned, cycle races planned); hot jobs cost 5, a cycle race at a venue
    # already in the round 3 (navigation paid), at another venue 5
    cases = [
        (0.1875, 1, 0),  # budget 3.75: the one hot job alone
        (0.25, 1, 0),  # 5
        (0.5, 2, 0),  # 10
        (0.625, 2, 0),  # 12.5: 2.5 spare, no race fits
        (0.6875, 2, 1),  # 13.75: the third hot race (5) does not fit, a cycle race (3) does
        (0.75, 3, 0),  # 15
        (1.0, 3, 1),  # 20
    ]
    assignments = [
        [("TOK", 3), ("KYO", 9), ("FKS", 16)],
        [("FKS", 3), ("TOK", 16), ("KYO", 9)],
    ]
    for assignment in assignments:
        posts = {venue: _spaced(now, 5, minutes) for venue, minutes in assignment}
        nearest = [venue for venue, _ in sorted(assignment, key=lambda a: a[1])]
        for order in (assignment, assignment[::-1]):
            states = [VenueState(code=venue, post_times=posts[venue]) for venue, _ in order]
            for rps, n_hot, n_cycle in cases:
                plan = plan_round(now, states, cfg, rps)
                where = f"rps {rps}, posts in {assignment}, given as {[v for v, _ in order]}"
                hot = {
                    (j.venue, r)
                    for j in plan
                    for r in j.races
                    if is_hot(posts[j.venue][r], now, cfg)
                }
                assert hot == {(venue, 5) for venue in nearest[:n_hot]}, f"{where}: {hot}"
                cycle = [
                    r for j in plan for r in j.races if not is_hot(posts[j.venue][r], now, cfg)
                ]
                assert len(cycle) == n_cycle, f"{where}: cycle races {cycle}"
                assert plan[0].venue == nearest[0] and 5 in plan[0].races, f"{where}: {plan[0]}"
                total = sum(cost(len(j.races), cfg) for j in plan)
                assert total <= 20 * rps or (len(plan) == 1 and total == cost(1, cfg)), where

    # One venue with two hot races: the order is by post, not by venue
    tok = {n: now + timedelta(minutes=2 + 30 * (n - 5)) for n in range(1, 6)}
    tok |= {n: now + timedelta(minutes=18 + 30 * (n - 6)) for n in range(6, 13)}
    kyo = _spaced(now, 5, 8)
    states = [VenueState(code="TOK", post_times=tok), VenueState(code="KYO", post_times=kyo)]
    for rps, want in [
        (0.25, {("TOK", 5)}),
        (0.5, {("TOK", 5), ("KYO", 5)}),  # not TOK 5 + TOK 6 (8), which leaves KYO out
        (0.6875, {("TOK", 5), ("KYO", 5), ("TOK", 6)}),  # 5 + 5 + 3 = 13 of 13.75
    ]:
        plan = plan_round(now, states, cfg, rps)
        got = {(j.venue, r) for j in plan for r in j.races}
        assert got == want, f"rps {rps}: {got}, expected {want}"


def _check_plan(plan, t: datetime, states: list, cfg: dict, rps: float) -> None:
    posts = {s.code: s.post_times for s in states}
    last = {s.code: s.last_capture for s in states}
    nav, per = cfg["requests"]["navigation"], cfg["requests"]["per_race"]
    floor = timedelta(seconds=cfg["min_poll_interval_s"])
    where = f"plan at {hhmm(t)}, rps {rps}, {cfg['requests']}"
    budget = budget_at(t, cfg, rps)
    if budget is None:
        assert plan == [], f"{where}: planned before the first window"
        return
    venues = [job.venue for job in plan]
    assert len(venues) == len(set(venues)), f"{where}: two jobs for one venue: {venues}"
    eligible = {v for v in posts if last[v] is None or t - last[v] >= floor}
    assert set(venues) <= eligible, f"{where}: a venue inside the floor was planned: {venues}"
    planned: dict[str, set[int]] = {}
    for job in plan:
        races = list(job.races)
        assert races and len(set(races)) == len(races), f"{where}: {job}"
        assert set(races) <= set(posts[job.venue]), f"{where}: {job}"
        assert job.requests == cost(len(races), cfg), f"{where}: {job}"
        planned[job.venue] = set(races)
    total = sum(cost(len(races), cfg) for races in planned.values())
    hot_eligible = [v for v in eligible if any(is_hot(p, t, cfg) for p in posts[v].values())]
    if budget < nav + per and hot_eligible:
        assert len(plan) == 1, f"{where}: one hot job always runs, planned {plan}"
    if total > budget + EPS:
        assert budget < nav + per, f"{where}: {total} requests for a budget of {budget}"
        assert len(plan) == 1 and len(plan[0].races) == 1, f"{where}: over budget: {plan}"
        (job,) = plan
        assert is_hot(posts[job.venue][job.races[0]], t, cfg), f"{where}: over budget, not hot"
        return
    # hot races first: none left out that fits on top of the planned hot races alone
    hot_in = {v: {r for r in rs if is_hot(posts[v][r], t, cfg)} for v, rs in planned.items()}
    hot_cost = sum(cost(len(rs), cfg) for rs in hot_in.values() if rs)
    for v in eligible:
        for r, post in posts[v].items():
            if is_hot(post, t, cfg) and r not in hot_in.get(v, set()):
                extra = per if hot_in.get(v) else nav + per
                assert hot_cost + extra > budget + EPS, f"{where}: hot {v} race {r} left out"
    # then the spare fills the due full cycles (every VenueState here is fresh: all races due)
    spare = budget - total
    for v in eligible:
        if v in planned and planned[v] != set(posts[v]):
            assert spare < per - EPS, f"{where}: {spare} spare while {v}'s cycle has races to go"
        if v not in planned:
            assert spare < nav + per - EPS, f"{where}: {spare} spare while {v} gets nothing"


def test_a_round_never_exceeds_the_budget_except_one_hot_job(tmp_path):
    meetings = staggered()
    # plan_round across the day, before any venue's end (TOK's is 15:40): two request costs, four
    # rates, venues fresh or inside the floor
    for requests in ({"navigation": 2, "per_race": 3}, {"navigation": 1, "per_race": 4}):
        cfg = config(requests=requests)
        for rps in (0.1875, 0.5, 1.0, 2.0):
            t = jst(6, 50, 7)
            while t < jst(15, 35):
                fresh = [VenueState(code=v, post_times=p) for v, p in meetings.items()]
                _check_plan(plan_round(t, fresh, cfg, rps), t, fresh, cfg, rps)
                mixed = [
                    VenueState(
                        code="TOK",
                        post_times=meetings["TOK"],
                        last_capture=t - timedelta(seconds=10),
                    ),
                    VenueState(
                        code="KYO",
                        post_times=meetings["KYO"],
                        last_capture=t - timedelta(seconds=20),
                    ),
                    VenueState(
                        code="FKS", post_times=meetings["FKS"], last_capture=t - timedelta(hours=1)
                    ),
                ]
                _check_plan(plan_round(t, mixed, cfg, rps), t, mixed, cfg, rps)
                t += timedelta(seconds=433)

    # run_day: what was sent, round by round, for a whole day at each rate
    for rps in (0.1875, 0.5, 1.0, 2.0, 64.0):
        clock = Clock(jst(6, 30))
        site = FakeSite(clock, staggered(), rps=rps)
        summary, store = run(site, clock, tmp_path / f"blocked-{rps}")
        assert caps(site, ok=True), f"rps {rps}: nothing captured"
        assert_rounds_fit_budget(site, CONFIG)
        assert_summary_matches(summary, site, store)
        if rps == 0.1875:
            assert summary.deferred > 0, "three venues' hot races cannot share 3.75 requests"
        if rps == 64.0:
            assert summary.deferred == 0, "nothing is deferred when the budget never binds"


def test_a_full_cycle_is_spread_over_rounds_race_by_race(tmp_path):
    # One venue, posts from 13:00 (hot from 12:40), run from 09:31: until 12:30 only full cycles.
    # (rps, navigation, per_race, races): k races fit a 20 s round, so a cycle takes ceil(n / k)
    cases = [(1.0, 2, 3, 12), (0.5, 2, 3, 12), (1.0, 1, 4, 11), (2.0, 2, 3, 12), (0.25, 2, 3, 10)]
    for rps, nav, per, n in cases:
        cfg = config(requests={"navigation": nav, "per_race": per})
        posts = {r: jst(13, 0) + timedelta(minutes=10 * (r - 1)) for r in range(1, n + 1)}
        clock = Clock(jst(9, 31))
        site = FakeSite(clock, {"KYO": posts}, rps=rps, nav=nav, per_race=per)
        summary, _ = run(site, clock, tmp_path / f"blocked-{rps}-{nav}-{n}", cfg=cfg)
        k = int((20 * rps - nav) // per)
        rounds = -(-n // k)
        sizes_due = [k] * (rounds - 1) + [n - k * (rounds - 1)]
        where = f"rps {rps}, {nav}+{per}/race, {n} races ({k} a round, {rounds} rounds)"

        cycles: list[list[Call]] = []
        covered: set[int] = set()
        for call in [c for c in caps(site, "KYO", ok=True) if c.t < jst(12, 30)]:
            races = set(asked(call, site))
            if not covered:
                cycles.append([])
            assert not races & covered, f"{where}: {hhmm(call.t)} repeats {races & covered}"
            cycles[-1].append(call)
            covered |= races
            if covered == set(posts):
                covered = set()
        complete = cycles if not covered else cycles[:-1]
        assert len(complete) >= 5, f"{where}: {len(complete)} full cycles from 09:31 to 12:30"
        for cycle in cycles:
            sizes = [len(asked(c, site)) for c in cycle]
            due = sizes_due if cycle in complete else sizes_due[: len(sizes)]
            assert sizes == due, f"{where}: cycle from {hhmm(cycle[0].t)} took {sizes}, not {due}"
            gaps = gaps_s([c.t for c in cycle])
            assert all(abs(g - 20) < 1e-3 for g in gaps), f"{where}: chunks {gaps} s apart"
        starts = [cycle[0].t for cycle in cycles]
        for gap in gaps_s(starts):
            assert 300 - 1e-3 <= gap < 320, f"{where}: cycles start {gap} s apart"
        assert summary.venues["KYO"].longest_cycle_s >= 20 * (rounds - 1) - 1e-3, where


def test_no_venue_is_captured_faster_than_the_20_s_floor(tmp_path):
    cfg = config()
    now = jst(12, 0, 7)
    hot = _spaced(now, 5, 5)  # race 5 posts in 5 minutes
    near = _spaced(now, 5, 2)
    ago = lambda seconds: now - timedelta(seconds=seconds)  # noqa: E731
    for rps in (0.1875, 1.0, 100.0):
        alone = plan_round(
            now, [VenueState(code="TOK", post_times=near, last_capture=ago(10))], cfg, rps
        )
        assert alone == [], f"rps {rps}: captured 10 s after the last capture"
        at_19 = plan_round(
            now, [VenueState(code="TOK", post_times=near, last_capture=ago(19.5))], cfg, rps
        )
        assert at_19 == [], f"rps {rps}: captured 19.5 s after the last capture"
        at_20 = plan_round(
            now, [VenueState(code="TOK", post_times=near, last_capture=ago(20))], cfg, rps
        )
        assert [j.venue for j in at_20] == ["TOK"], f"rps {rps}: 20 s is the floor, not below it"
        states = [
            VenueState(code="TOK", post_times=near, last_capture=ago(10)),
            VenueState(code="KYO", post_times=hot, last_capture=None),
        ]
        assert [j.venue for j in plan_round(now, states, cfg, rps)] == ["KYO"], f"rps {rps}"

    # A window faster than the floor: rounds every 5 s, each venue still at most every 20 s
    cfg = config(windows=[{"from": "07:00", "every_s": 300}, {"from": "09:30", "every_s": 5}])
    clock = Clock(jst(9, 31))
    site = FakeSite(clock, staggered(), rps=4.0)
    run(site, clock, tmp_path / "blocked", cfg=cfg)
    for venue in site.meetings:
        times = [c.t for c in caps(site, venue)]
        gaps = gaps_s(times)
        assert gaps and min(gaps) >= 20 - EPS, f"{venue}: gaps {min(gaps, default=None)} s"
    assert_hot_races_followed(site, cfg, site.meetings, every=25)


# ---------------------------------------------------------------- run_day


def test_three_venues_are_logged_until_after_their_last_races(tmp_path):
    meetings = staggered()
    clock = Clock(jst(6, 30))
    shipped = copy.deepcopy(load_country_yaml("japan", "logger"))
    # The race date's schedule: the day-before window (#358) has its own tests
    # (test_day_before_schedule.py), and this fake site has no sales_open().
    shipped.pop("day_before", None)
    site = FakeSite(clock, meetings, rps=1.0, per_race=shipped["requests"]["per_race"])
    summary, store = run(site, clock, tmp_path / "blocked", cfg=shipped)

    ends = {v: max(p.values()) + timedelta(minutes=10) for v, p in meetings.items()}
    assert set(summary.venues) == set(meetings)
    for venue, end in ends.items():
        ok = caps(site, venue, ok=True)
        assert ok and ok[0].t < jst(7, 5), f"{venue}: not captured from the morning"
        assert_logged_until(site, venue, end)
        vs = summary.venues[venue]
        assert (vs.failures, vs.wrong_meeting) == (0, 0), venue
        assert vs.stopped not in ("circuit_open", "wrong_meeting"), (venue, vs.stopped)
        # (#387: with the shipped floor the 300 s captures may take the full cycles' place from
        # 09:30; test_a_full_cycle_is_spread_over_rounds_race_by_race keeps #24's cycles.)
        reads = [c for c in site.calls if c.kind == "post_times" and c.venue == venue]
        assert 1 <= len(reads) <= secs(end - jst(7, 0)) / 300 + 2, f"{venue}: {len(reads)} reads"
        assert (venue, meetings[venue]) in store.post_times, f"{venue}: post times not stored"
    # FKS ends last, alone: it is still captured after its last race's hot window closes
    last_post = max(meetings["FKS"].values())
    after_hot = [c for c in caps(site, "FKS", ok=True) if last_post + timedelta(minutes=5) < c.t]
    assert after_hot, "FKS not captured between its last post + 5 and + 10 minutes"
    # Three venues at jra.yaml's 1 req/s (no `resume` here, so no three-venue rate, #387): the
    # hot races keep at least the health line's T-20 criterion, MAX_GAP_S. The owner's 60 s grain
    # at the three-venue rate is test_five_minute_floor.py's.
    assert_hot_races_followed(site, shipped, meetings, every=MAX_GAP_S)
    assert_summary_matches(summary, site, store)
    assert clock.t <= max(ends.values()) + timedelta(seconds=300), f"returned at {hhmm(clock.t)}"


@pytest.mark.parametrize(
    ("after_min", "first_post", "delay", "end"),
    [
        (10, (10, 0), None, (15, 40)),
        (25, (10, 0), None, (15, 55)),
        (10, (10, 0), 20, (16, 0)),  # race 12 moved from 15:30 to 15:50 at 11:00
        (10, (11, 20), None, (17, 0)),  # last post 16:50, after the 16:45 stop
    ],
    ids=["10-min", "25-min", "post-moved", "past-stop"],
)
def test_a_venue_is_logged_until_its_last_post_plus_after_last_race_min(
    tmp_path, after_min, first_post, delay, end
):
    posts = {n: jst(*first_post) + timedelta(minutes=30 * (n - 1)) for n in range(1, 13)}
    moved = dict(posts)
    if delay:
        moved[12] = posts[12] + timedelta(minutes=delay)

    def shown(venue, t):
        return dict(moved if t >= jst(11, 0) else posts)

    clock = Clock(jst(9, 31))
    site = FakeSite(clock, {"TOK": posts}, post_times=shown)
    summary, store = run(
        site, clock, tmp_path / "blocked", cfg=config(after_last_race_min=after_min)
    )
    end_t = jst(*end)
    assert_logged_until(site, "TOK", end_t)
    last_post = max(moved.values())
    after_hot = [c for c in caps(site, "TOK", ok=True) if last_post + timedelta(minutes=5) < c.t]
    assert after_hot, f"not captured between {hhmm(last_post)} + 5 minutes and {hhmm(end_t)}"
    if delay:
        assert ("TOK", moved) in store.post_times, "the moved post time was not stored"


def started_left_out(times_at: Callable[[str, datetime], dict[int, datetime]], lag: timedelta):
    """A post_times reader that, like official_pages.post_times (発走済 cells), leaves out every
    race whose post is `lag` or more in the past."""

    def shown(venue: str, t: datetime) -> dict[int, datetime]:
        return {n: post for n, post in times_at(venue, t).items() if t < post + lag}

    return shown


@pytest.mark.parametrize(
    ("lag_s", "after_post_min", "after_min", "delay"),
    [
        (0, 5, 10, None),
        (90, 5, 10, None),
        (0, 8, 15, None),
        (60, 5, 10, 20),  # TOK race 12 moved from 15:30 to 15:50 at 11:00
    ],
    ids=["left-out-at-post", "left-out-90-s-after-post", "8-min-tail", "post-moved"],
)
def test_a_race_left_off_a_post_time_reread_after_its_off_is_still_captured_to_post_plus_5(
    tmp_path, lag_s, after_post_min, after_min, delay
):
    first = staggered()
    final = copy.deepcopy(first)
    if delay:
        final["TOK"][12] = first["TOK"][12] + timedelta(minutes=delay)

    def times_at(venue, t):
        return dict(final[venue] if t >= jst(11, 0) else first[venue])

    cfg = config(hot_races={"after_post_min": after_post_min}, after_last_race_min=after_min)
    clock = Clock(jst(6, 30))
    reader = started_left_out(times_at, timedelta(seconds=lag_s))
    site = FakeSite(clock, final, rps=1.0, post_times=reader)
    summary, store = run(site, clock, tmp_path / "blocked", cfg=cfg)

    # every race is followed after its off to post + after_post_min, at the 20 s round cadence
    tail = timedelta(minutes=after_post_min)
    cut_short = []
    for venue, posts in final.items():
        ok = caps(site, venue, ok=True)
        for n, post in posts.items():
            off = [c.t for c in ok if n in asked(c, site) and post <= c.t <= post + tail]
            if not off or secs(post + tail - off[-1]) > 20 + EPS:
                last = hhmm(off[-1]) if off else "none"
                cut_short.append(f"{venue} {n} (post {hhmm(post)}, last after it {last})")
    races = sum(len(posts) for posts in final.values())
    assert cut_short == [], f"{len(cut_short)} of {races} races cut short: {cut_short[:4]}"
    assert_hot_races_followed(site, cfg, final, every=20)
    # and each venue still ends at its last post (the moved one too) + after_last_race_min
    for venue, posts in final.items():
        assert_logged_until(site, venue, max(posts.values()) + timedelta(minutes=after_min))
        vs = summary.venues[venue]
        assert (vs.failures, vs.wrong_meeting) == (0, 0), venue
    assert_summary_matches(summary, site, store)


def test_three_wrong_meetings_in_a_row_stop_that_venue_only(tmp_path):
    def wrong():
        return WrongMeeting("the race list is for another meeting")

    # (KYO captures that answer for another meeting, max_wrong_meeting, KYO stops after capture n)
    cases = [({5, 6, 7}, 3, 7), ({5, 6}, 2, 6), ({5, 6, 8, 9}, 3, None)]
    meetings = staggered()
    ends = {v: max(p.values()) + timedelta(minutes=10) for v, p in meetings.items()}
    for numbers, limit, stops_after in cases:
        clock = Clock(jst(6, 30))
        site = FakeSite(clock, meetings, fail=on_calls("capture", "KYO", numbers, wrong))
        cfg = config(max_wrong_meeting=limit)
        summary, store = run(site, clock, tmp_path / f"blocked-{limit}-{len(numbers)}", cfg=cfg)
        where = f"wrong meeting on KYO captures {sorted(numbers)}, max {limit}"
        kyo = summary.venues["KYO"]
        assert kyo.wrong_meeting == len(numbers), where
        if stops_after:
            assert kyo.stopped == "wrong_meeting", f"{where}: stopped {kyo.stopped!r}"
            kyo_caps = caps(site, "KYO")
            assert len(kyo_caps) == stops_after, f"{where}: {len(kyo_caps)} KYO captures"
            last = site.calls.index(kyo_caps[-1])
            after = [c for c in site.calls[last + 1 :] if c.venue == "KYO"]
            assert after == [], f"{where}: KYO called again: {after[:3]}"
        else:
            assert kyo.stopped not in ("circuit_open", "wrong_meeting"), f"{where}: {kyo.stopped}"
            assert_logged_until(site, "KYO", ends["KYO"])
        for venue in ("TOK", "FKS"):
            vs = summary.venues[venue]
            assert vs.wrong_meeting == 0 and vs.stopped not in ("circuit_open", "wrong_meeting")
            assert_logged_until(site, venue, ends[venue])
        assert_summary_matches(summary, site, store)


def test_not_listed_before_the_first_post_is_retried_quietly(tmp_path):
    meetings = staggered()
    listed_from = jst(8, 2)  # TOK's odds menu lists it from 08:02; its first post is 10:00

    def tok_unlisted(kind, venue, n, t):
        if (kind, venue) == ("capture", "TOK") and t < listed_from:
            return NotListed("TOK is not on the odds menu yet")
        return None

    kyo_after_post = from_time(
        "capture", "KYO", jst(11, 10), NotListed("KYO left the menu"), FetchError("timeout")
    )
    clock = Clock(jst(6, 30))
    site = FakeSite(clock, meetings, fail=rules(tok_unlisted, kyo_after_post))
    summary, store = run(site, clock, tmp_path / "blocked")
    ends = {v: max(p.values()) + timedelta(minutes=10) for v, p in meetings.items()}

    tok = summary.venues["TOK"]
    unlisted = [c for c in caps(site, "TOK") if c.error == "NotListed"]
    assert len(unlisted) >= 2, f"TOK tried {len(unlisted)} times before it was listed"
    assert (tok.failures, tok.wrong_meeting) == (0, 0), "a NotListed before the first post counted"
    assert tok.stopped not in ("circuit_open", "wrong_meeting"), tok.stopped
    first_ok = caps(site, "TOK", ok=True)[0].t
    assert listed_from <= first_ok < listed_from + timedelta(seconds=320), hhmm(first_ok)
    assert_logged_until(site, "TOK", ends["TOK"])

    kyo = summary.venues["KYO"]
    assert kyo.failures == 2, f"KYO failures {kyo.failures}: NotListed after the post is one"
    assert kyo.stopped not in ("circuit_open", "wrong_meeting"), kyo.stopped
    assert_logged_until(site, "KYO", ends["KYO"])
    assert summary.venues["FKS"].failures == 0
    assert_summary_matches(summary, site, store)


def test_circuit_open_stops_every_venue_and_writes_the_block_marker(tmp_path):
    cases = {
        "kyo-capture-403": on_calls("capture", "KYO", {3}, blocked(403)),
        # 08:30 in Tokyo is 23:30 UTC on the day before: the marker carries the Tokyo date
        "tok-capture-429-0830": from_time("capture", "TOK", jst(8, 30), blocked(429)()),
        "fks-capture-403-racing": from_time("capture", "FKS", jst(13, 0), blocked(403)()),
        "fks-post-times-403": on_calls("post_times", "FKS", {1}, blocked(403)),
        "discover-429": on_calls("discover", None, {1}, blocked(429)),
    }
    for name, fail in cases.items():
        marker = tmp_path / f"{name}.blocked"
        clock = Clock(jst(6, 30))
        site = FakeSite(clock, staggered(), fail=fail)
        summary, _ = run(site, clock, marker)
        opened = [i for i, c in enumerate(site.calls) if c.error == "CircuitOpen"]
        assert len(opened) == 1, f"{name}: the scripted block never came"
        assert opened[0] == len(site.calls) - 1, (
            f"{name}: called after the block: {site.calls[-3:]}"
        )
        if name != "discover-429":
            assert set(summary.venues) == set(site.meetings), name
        stopped = {v: vs.stopped for v, vs in summary.venues.items()}
        assert all(s == "circuit_open" for s in stopped.values()), f"{name}: {stopped}"
        assert summary.requests <= site.requests, name
        assert marker.is_file(), f"{name}: no block marker"
        text = marker.read_text(encoding="utf-8")
        assert HOST in text and "2026-11-28" in text, f"{name}: marker says {text!r}"


def test_a_block_marker_for_today_means_no_request(tmp_path):
    marker = tmp_path / "jra.blocked"
    clock = Clock(jst(6, 30))
    site = FakeSite(
        clock, staggered(), fail=from_time("capture", "TOK", jst(8, 30), blocked(403)())
    )
    run(site, clock, marker)
    assert marker.is_file(), "the 403 wrote no marker"

    # restarts the same day, whenever and however they are started: nothing goes out
    for start, venues in ((jst(10, 0), None), (jst(10, 0), ["TOK"]), (jst(6, 0), None)):
        clock = Clock(start)
        site = FakeSite(clock, staggered())
        summary, store = run(site, clock, marker, venues=venues)
        assert site.calls == [], f"restart at {hhmm(start)}: {site.calls[:3]}"
        assert summary.requests == 0 and store.writes == [], hhmm(start)
    assert marker.is_file(), "a restart removed today's marker"

    # another host the same day, and this host the next day, are not blocked
    short = {"NAK": {1: jst(10, 30), 2: jst(11, 0)}}
    clock = Clock(jst(10, 0))
    site = FakeSite(clock, short, host="odds.example.test")
    summary, _ = run(site, clock, marker)
    assert caps(site, ok=True), "a marker for www.jra.go.jp blocked another host"
    sunday = {"NAK": {1: jst(10, 30, day=SUN), 2: jst(11, 0, day=SUN)}}
    clock = Clock(jst(9, 40, day=SUN))
    site = FakeSite(clock, sunday)
    summary, _ = run(site, clock, marker, day=SUN)
    assert [c.kind for c in site.calls][:1] == ["discover"], site.calls[:2]
    assert summary.venues["NAK"].snapshots > 0, "yesterday's marker blocked today"


def test_a_day_without_meetings_returns_at_once(tmp_path):
    for start, first in ((jst(10, 0), jst(10, 0)), (jst(6, 0), jst(7, 0))):
        clock = Clock(start)
        site = FakeSite(clock, {})
        summary, store = run(site, clock, tmp_path / "blocked")
        assert [c.kind for c in site.calls] == ["discover"], site.calls
        found = site.calls[0].t
        assert first <= found < first + timedelta(seconds=300), hhmm(found)
        assert secs(clock.t - found) <= 1, f"started {hhmm(start)}, returned at {hhmm(clock.t)}"
        assert dict(summary.venues) == {} and store.writes == [] and store.post_times == []


def test_nothing_is_requested_before_the_first_window(tmp_path):
    early = {1: jst(7, 5), 2: jst(7, 35)}  # race 1 is hot from 06:45
    for rps in (1.0, 100.0):
        for t in (jst(6, 45), jst(6, 59, 59)):
            plan = plan_round(t, [VenueState(code="TOK", post_times=early)], config(), rps)
            assert plan == [], f"rps {rps}: planned at {hhmm(t)}: {plan}"
        assert plan_round(jst(7, 0), [VenueState(code="TOK", post_times=early)], config(), rps)

    # run from 05:00 Tokyo time, 20:00 UTC the day before
    late_start = [{"from": "08:15", "every_s": 300}, {"from": "09:30", "every_s": 20}]
    for windows, first in ((CONFIG["windows"], jst(7, 0)), (late_start, jst(8, 15))):
        clock = Clock(jst(5, 0))
        site = FakeSite(clock, staggered(races=2))
        run(site, clock, tmp_path / "blocked", cfg=config(windows=windows))
        assert site.calls, "nothing was requested all day"
        assert min(c.t for c in site.calls) >= first, f"first request at {hhmm(site.calls[0].t)}"
        assert site.calls[0].t < first + timedelta(seconds=300), hhmm(site.calls[0].t)
        assert caps(site, ok=True), "no capture"


def test_stop_falls_back_to_16_45_without_post_times(tmp_path):
    meetings = {v: dict(list(p.items())[:4]) for v, p in staggered().items()}
    for stop in ("16:45", "15:30"):
        stop_t = wall(stop, SAT)
        clock = Clock(jst(9, 31))
        site = FakeSite(clock, meetings, rps=6.0, post_times=lambda venue, t: {})
        run(site, clock, tmp_path / "blocked", cfg=config(stop=stop))
        late = [c for c in site.calls if c.t >= stop_t]
        assert late == [], f"stop {stop}: {len(late)} calls at or after it: {late[:3]}"
        for venue in meetings:
            times = [c.t for c in caps(site, venue, ok=True)]
            assert times and secs(stop_t - times[-1]) <= 320 + EPS, f"{venue}: stopped early"
            gaps = gaps_s(times)
            assert gaps and min(gaps) >= 300 - 1e-3, f"{venue}: gaps {min(gaps, default=None)} s"
            reads = [c for c in site.calls if c.kind == "post_times" and c.venue == venue]
            assert len(reads) >= 2, f"{venue}: post times never read again"
        assert clock.t <= stop_t + timedelta(seconds=300), f"returned at {hhmm(clock.t)}"


def test_a_cooldown_resumes_once_after_the_set_minutes(tmp_path):
    fail = rules(
        from_time("capture", "TOK", jst(10, 5), blocked(403)()),
        from_time("capture", "TOK", jst(11, 0), blocked(429)()),
    )
    clock = Clock(jst(9, 31))
    site = FakeSite(clock, {"TOK": staggered()["TOK"]}, fail=fail)
    summary, _ = run(site, clock, tmp_path / "blocked", cfg=config(on_block={"cooldown_min": 30}))
    opened = [i for i, c in enumerate(site.calls) if c.error == "CircuitOpen"]
    assert len(opened) == 2, f"{len(opened)} blocks: the run did not resume, or not once"
    first, second = (site.calls[i].t for i in opened)
    between = site.calls[opened[0] + 1 : opened[1]]
    assert between and between[0].t >= first + timedelta(minutes=30), hhmm(between[0].t)
    assert between[0].t < first + timedelta(minutes=30, seconds=320), hhmm(between[0].t)
    assert any(c.kind == "capture" and c.error is None for c in between), "nothing captured"
    assert opened[1] == len(site.calls) - 1, f"called after the second block: {site.calls[-3:]}"
    assert summary.venues["TOK"].stopped == "circuit_open"


def test_given_venues_are_logged_without_discovery(tmp_path):
    clock = Clock(jst(9, 40))
    site = FakeSite(clock, staggered(races=3))
    summary, _ = run(site, clock, tmp_path / "blocked", venues=["KYO"])
    assert all(c.kind != "discover" for c in site.calls), "discovered although venues were given"
    assert {c.venue for c in site.calls} == {"KYO"}
    assert set(summary.venues) == {"KYO"} and summary.venues["KYO"].snapshots > 0
