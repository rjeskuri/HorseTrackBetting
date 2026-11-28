"""The 5-minute floor on two- and three-venue days at jra.yaml's 1 request a second (ticket #387).

#358 (the owner's rule 2) asks for every JRA race to be captured at least every 5 minutes until its
T-15, and keeps the 09:30 window (20 s rounds) and the T-20 hot rule. At 1 req/s three hot races at
per_race 4 cost 18 of a round's 20 requests, so the other races' full cycles starve: #358's
simulation measured gaps up to 1,140 s (three venues) and 404 s (two) from 09:30 to T-15.

The owner's decision (2026-11-28, option (a) on #387): "Keep the 5-minute floor first. The
20-second hot grain is second priority: once a race's last capture is 300 s old, it outranks the
other hot races, and those take the slower grain (about 30-40 s, inside MAX_GAP_S 120). The JRA
rate stays at 1 req/s, and all five tier-1 pools stay every round."

Contract pinned here (the coder follows it), on top of test_schedule.py's and
test_day_before_schedule.py's:

Config
- `hot_races.floor_s` (seconds): the floor. logger.yaml's JRA schedule (its top level) ships
  `floor_s: 300`; NAR's schedule (`source_config(config, "nar")`) has its own `hot_races` block
  without it, so NAR is unchanged. A schedule without `floor_s` plans and runs exactly as before
  (#24's order). The tests vary floor_s and the windows' every_s: neither may be a constant.
- Unchanged, as the owner decided: jra.yaml's www.jra.go.jp at 1.0 requests a second; the five
  tier-1 pools at `requests` {navigation 2, per_race 4}; the windows (09:30 every 20 s); the hot
  window (within_min 20, after_post_min 5); all_every_s 300; min_poll_interval_s 20; and the
  health line's `MAX_GAP_S` 120 (horsetrack.countries.japan.logger.health).

VenueState (horsetrack.countries.japan.logger.schedule)
- gains `captured: dict[int, datetime]`, default empty: race number -> the start of the round
  whose successful capture of that race is the latest (run_day stamps it, as it stamps
  last_capture with the round's start; a capture of every race, races None, stamps every race of
  the meeting). A race missing from it has never been captured: it is never stale (the full cycle
  covers it), so a fresh VenueState plans as test_schedule.py says.

Stale (a round starting at `now`, every_s e of the window in force, floor_s F)
- A race of `captured` whose post is still ahead (now < post) is stale when now - captured[race]
  >= F - e (one round before its capture would be F old), and not stale when that age is <= F / 2.
  Between the two the threshold is the coder's (a Ruling line on the PR). Races already off are
  not part of the floor (the rule is "until its T-15"; from T-20 the hot window applies).

plan_round(now, venues, config, rps) with floor_s, inside the budget (every_s x rps):
1. the nearest hot race of all: the hot race (post - within_min <= now <= post + after_post_min)
   with the smallest |post - now| over the eligible venues (ties: either); its Job comes first;
2. then every stale race that still fits (their order among themselves is not pinned);
3. then the other hot races, nearest post first, each one that still fits;
4. then the due full cycles race by race, from what is left (as before).
One Job per venue, Job.requests, min_poll_interval_s and the single hot job exception (a budget
below one hot race) are test_schedule.py's. With no stale race the plan is test_schedule.py's.

run_day, the race-day simulation (the DoD): logger.yaml's JRA schedule as shipped (tier 2 and
results polling included; the run starts at 07:00, the first window, so it never probes for the
day-before sales), jra.yaml's rate, synthetic 12-race cards: #358's (first posts 09:50, 10:00,
10:10, every 34 minutes) on two and three venues, the same five minutes later on three, and a
30-minute card (10:00, 10:10) on two. Two clocks:
- 1 s a request: each site call moves the fake clock by its requests / rps, so a round lasts as
  long as what it sends (results polls, post-time reads and tier 2 included).
- exact rounds: site calls take no time, so every capture of a round is at the round's start.
Both pin, for every race:
- (i) the floor: no gap over floor_s between 09:30, its successful tier-1 captures from 09:30 to
  its T-15, and its T-15;
- (ii) the hot window [post - within_min, post + after_post_min]: no gap over MAX_GAP_S between
  the window's start, the race's captures in it and the window's end; and the health line's own
  metric (health.gaps_inside of the written tier-1 rows, [T-20, post]) at most MAX_GAP_S.
The exact rounds also pin:
- (iii) every round from 09:30 while a race is hot captures the nearest hot race of all (at the
  round's start, smallest |post - start|; either one of a tie);
- (iv) every round fits its budget, or is the single hot job (test_schedule.py's
  assert_rounds_fit_budget).

Not pinned: the order of jobs after the first, the order of stale races among themselves, which
venue's cycle gets spare budget, tier 2's share (it takes what tier 1 leaves), races already off,
and a three-venue 30-minute card (three hot races half the time: denser than #358's simulation,
which the DoD names). Everything is synthetic: no HTTP, no real sleep, no scraped page.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from horsetrack.core.config import load_country_yaml
from horsetrack.countries.japan.jra.capture import MeetingCapture
from horsetrack.countries.japan.logger.health import MAX_GAP_S, gaps_inside

# isort: split
from horsetrack.countries.japan.logger.schedule import (
    VenueState,
    plan_round,
    run_day,
    source_config,
)

UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
DAY = date(2026, 12, 5)  # a Saturday: three-venue days start in December
HOST = "www.jra.go.jp"
EPS = 1e-6
TIER1 = ["WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE", "BRACKET_QUINELLA"]
TIER2 = ["FORECAST", "TRIO", "TIERCE"]
FLOOR_END_MIN = 15  # the owner's rule 2 (#358): the floor holds until the race's T-15
OWNER_FLOOR_S = 300  # the owner's 5 minutes (logger.yaml hot_races.floor_s)

# The outline's schedule as #358 ships it, with the owner's floor (plan_round tests).
CONFIG = {
    "pools": list(TIER1),
    "min_poll_interval_s": 20,
    "windows": [{"from": "07:00", "every_s": 300}, {"from": "09:30", "every_s": 20}],
    "stop": "16:45",
    "after_last_race_min": 10,
    "hot_races": {"within_min": 20, "after_post_min": 5, "all_every_s": 300, "floor_s": 300},
    "requests": {"navigation": 2, "per_race": 4},
    "max_wrong_meeting": 3,
    "on_block": {"cooldown_min": None},
}


def config(base: dict | None = None, **changes) -> dict:
    """`base` (CONFIG by default) with some values changed (a dict value is merged in)."""
    cfg = copy.deepcopy(CONFIG if base is None else base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **copy.deepcopy(value)}
        else:
            cfg[key] = copy.deepcopy(value)
    return cfg


def jra_schedule() -> dict:
    """logger.yaml's JRA schedule, as log-day hands it to run_day."""
    return copy.deepcopy(source_config(load_country_yaml("japan", "logger"), "jra"))


def jra_rate() -> float:
    http = load_country_yaml("japan", "jra")["http"]
    return float(http["hosts"].get(HOST, http["default"])["requests_per_second"])


# ---------------------------------------------------------------- time


def jst(h: int, m: int = 0, s: int = 0, day: date = DAY) -> datetime:
    """The aware UTC instant of a Tokyo wall-clock time on `day`."""
    return datetime(day.year, day.month, day.day, h, m, s, tzinfo=JST).astimezone(UTC)


def wall(hhmm: str, day: date = DAY) -> datetime:
    h, m = (int(part) for part in str(hhmm).split(":"))
    return jst(h, m, day=day)


def secs(delta: timedelta) -> float:
    return delta.total_seconds()


def hhmm(t: datetime) -> str:
    return t.astimezone(JST).strftime("%H:%M:%S JST")


def every_s_at(t: datetime, cfg: dict) -> float | None:
    current = None
    for window in sorted(cfg["windows"], key=lambda w: wall(w["from"])):
        if wall(window["from"]) <= t:
            current = float(window["every_s"])
    return current


def is_hot(post: datetime, t: datetime, cfg: dict) -> bool:
    hot = cfg["hot_races"]
    return (
        post - timedelta(minutes=hot["within_min"])
        <= t
        <= post + timedelta(minutes=hot["after_post_min"])
    )


def cost(n_races: int, cfg: dict) -> int:
    return cfg["requests"]["navigation"] + cfg["requests"]["per_race"] * n_races


# ---------------------------------------------------------------- race cards


def card(venues: int, firsts: list[tuple[int, int]], every_min: int, races: int = 12):
    """`venues` meetings of `races` races, `every_min` minutes apart, first posts `firsts`."""
    codes = ["TOK", "KYO", "FKS"][:venues]
    return {
        code: {n: jst(*first) + timedelta(minutes=every_min * (n - 1)) for n in range(1, races + 1)}
        for code, first in zip(codes, firsts, strict=False)
    }


CARDS = {
    "358": ([(9, 50), (10, 0), (10, 10)], 34),  # #358's simulation
    "358-later": ([(9, 55), (10, 5), (10, 15)], 34),  # the same density, five minutes later
    "30-min": ([(10, 0), (10, 10), (10, 20)], 30),  # two venues only: see the module docstring
}


# ---------------------------------------------------------------- fakes


class Runaway(BaseException):
    """run_day kept going where a day cannot (a BaseException: no `except Exception` hides it)."""


class Clock:
    """The fake clock: sleep() moves it, and so does the fake site's request time."""

    def __init__(self, start: datetime):
        self.t = start
        self.limit = start + timedelta(hours=24)
        self.reads = 0

    def now(self) -> datetime:
        self.reads += 1
        if self.reads > 200_000:
            raise Runaway("now() read 200000 times while the clock stood still: a busy loop")
        return self.t

    def advance(self, seconds: float) -> None:
        if seconds < -EPS:
            raise Runaway(f"sleep({seconds}): a negative wait")
        if seconds > 0:
            self.reads = 0
        self.t += timedelta(seconds=max(seconds, 0.0))
        if self.t > self.limit:
            raise Runaway(f"run_day still running at {self.t.isoformat()}: it never returned")

    sleep = advance


@dataclass
class Call:
    t: datetime
    kind: str  # sales_open | discover | post_times | capture | tier2 | results
    venue: str | None
    races: tuple[int, ...] | None = None
    requests: int = 0


class FakeSite:
    """The JRA site of race date DAY as the scheduler sees it: every call recorded at the fake
    clock's time, then the clock moved by its requests x `request_s` (1 / rps: one request a
    second; 0: calls take no time). A tier-1 capture costs navigation + per_race x races, a tier-2
    one its own, a results poll 2 (the レース結果 menu and the race list); a race's result is
    published 10 minutes after its post."""

    source = "jra"
    host = HOST

    def __init__(self, clock: Clock, meetings, cfg: dict, *, rps: float, request_s: float):
        self.clock, self.meetings, self.cfg = clock, meetings, cfg
        self.requests_per_second, self.request_s = rps, request_s
        self.nav, self.per_race = cfg["requests"]["navigation"], cfg["requests"]["per_race"]
        tier2 = cfg.get("tier2") or {"requests": {"navigation": 2, "per_race": 3}}
        self.t2_nav, self.t2_per = tier2["requests"]["navigation"], tier2["requests"]["per_race"]
        self.calls: list[Call] = []
        self.defects: list[str] = []

    def _call(self, kind: str, venue: str | None, races, requests: int) -> Call:
        call = Call(self.clock.t, kind, venue, races, requests)
        self.calls.append(call)
        if venue is not None and venue not in self.meetings:
            self.defects.append(f"{kind} for {venue}, which has no meeting on {DAY}")
        self.clock.advance(requests * self.request_s)
        return call

    def sales_open(self) -> bool:
        self._call("sales_open", None, None, 1)
        return True

    def discover(self) -> list[str]:
        self._call("discover", None, None, 1)
        return list(self.meetings)

    def post_times(self, venue: str) -> dict[int, datetime]:
        self._call("post_times", venue, None, self.nav)
        return dict(self.meetings[venue])

    def _asked(self, venue: str, races, kind: str) -> list[int]:
        every = sorted(self.meetings[venue])
        if races is None:
            return every
        asked = list(races)
        if not asked or len(set(asked)) != len(asked) or set(asked) - set(every):
            self.defects.append(f"{hhmm(self.clock.t)} {kind} of {venue} asks for {asked}")
        return sorted(set(asked) & set(every))

    def capture(self, venue: str, races) -> MeetingCapture:
        wanted = self._asked(venue, races, "capture")
        spent = self.nav + self.per_race * len(wanted)
        call = self._call("capture", venue, tuple(wanted), spent)
        rows = [
            {"race_id": f"jra:{DAY.isoformat()}:{venue}:{n:02d}", "pool": pool,
             "snapshot_utc": call.t}
            for n in wanted
            for pool in self.cfg["pools"]
        ]  # fmt: skip
        return MeetingCapture(rows, call.t, spent)

    def capture_tier2(self, venue: str, races) -> MeetingCapture:
        wanted = self._asked(venue, races, "tier-2 capture")
        spent = self.t2_nav + self.t2_per * len(wanted)
        call = self._call("tier2", venue, tuple(wanted), spent)
        rows = [
            {"race_id": f"jra:{DAY.isoformat()}:{venue}:{n:02d}", "pool": pool,
             "snapshot_utc": call.t}
            for n in wanted
            for pool in TIER2
        ]  # fmt: skip
        return MeetingCapture(rows, call.t, spent)

    def published_results(self, venue: str) -> set[int]:
        call = self._call("results", venue, None, 2)
        out = timedelta(minutes=10)
        return {n for n, post in self.meetings[venue].items() if call.t >= post + out}


class FakeStore:
    def __init__(self):
        self.writes: list[tuple[str, MeetingCapture]] = []
        self.post_times: list[tuple[str, dict[int, datetime]]] = []
        self.published: list[tuple[str, list[int]]] = []

    def write(self, venue: str, capture: MeetingCapture) -> int:
        self.writes.append((venue, capture))
        return len(capture.rows)

    def write_post_times(self, venue: str, times: dict[int, datetime]) -> None:
        self.post_times.append((venue, dict(times)))

    def write_published(self, venue, races, seen_utc, unpublished_utc=None) -> None:
        self.published.append((venue, sorted(races)))


def simulate(meetings, cfg: dict, rps: float, request_s: float, marker: Path):
    """A race day from 07:00 (the first window: no day-before probe) to its end."""
    clock = Clock(jst(7, 0))
    site = FakeSite(clock, meetings, cfg, rps=rps, request_s=request_s)
    store = FakeStore()
    summary = run_day(DAY, site, store, config=cfg, now=clock.now, sleep=clock.sleep,
                      block_marker=marker)  # fmt: skip
    assert site.defects == [], site.defects[:5]
    return site, store, summary


# ---------------------------------------------------------------- what the fake saw


def captured_at(site: FakeSite, venue: str, race: int) -> list[datetime]:
    """When `race` of `venue` was captured in tier 1."""
    return [c.t for c in site.calls if c.kind == "capture" and c.venue == venue and race in c.races]


def longest_gap(times: list[datetime], lo: datetime, hi: datetime) -> tuple[float, datetime]:
    """The longest gap between lo, the times inside [lo, hi] and hi; and where it starts."""
    edges = [lo, *sorted(t for t in times if lo <= t <= hi), hi]
    gap, at = max((secs(b - a), a) for a, b in zip(edges, edges[1:], strict=False))
    return gap, at


def floor_misses(site: FakeSite, cfg: dict) -> list[str]:
    """(i): races with a gap over floor_s from 09:30 to their T-15 (the owner's 300 s while a
    schedule has no floor_s yet: the yaml test pins the key)."""
    floor_s = float(cfg["hot_races"].get("floor_s", OWNER_FLOOR_S))
    w = jst(9, 30)
    out = []
    for venue, races in site.meetings.items():
        for n, post in races.items():
            t15 = post - timedelta(minutes=FLOOR_END_MIN)
            if t15 <= w:
                continue
            gap, at = longest_gap(captured_at(site, venue, n), w, t15)
            if gap > floor_s + EPS:
                out.append(f"{venue} {n} (post {hhmm(post)}): {gap:.0f} s from {hhmm(at)}")
    return out


def hot_misses(site: FakeSite, cfg: dict) -> list[str]:
    """(ii): races with a gap over MAX_GAP_S inside their hot window, its edges included."""
    hot = cfg["hot_races"]
    out = []
    for venue, races in site.meetings.items():
        for n, post in races.items():
            lo = post - timedelta(minutes=hot["within_min"])
            hi = post + timedelta(minutes=hot["after_post_min"])
            gap, at = longest_gap(captured_at(site, venue, n), lo, hi)
            if gap > MAX_GAP_S + EPS:
                out.append(f"{venue} {n} (post {hhmm(post)}): {gap:.0f} s from {hhmm(at)}")
    return out


def health_misses(site: FakeSite, store: FakeStore, cfg: dict) -> list[str]:
    """(ii), the health line's metric: health.gaps_inside over the written tier-1 rows."""
    rows = [r for _, cap in store.writes for r in cap.rows if r["pool"] in cfg["pools"]]
    posts = {
        f"jra:{DAY.isoformat()}:{venue}:{n:02d}": post
        for venue, races in site.meetings.items()
        for n, post in races.items()
    }
    gaps = gaps_inside(rows, posts, cfg["hot_races"]["within_min"])
    return [f"{rid}: {gap:.0f} s" for rid, gap in sorted(gaps.items()) if gap > MAX_GAP_S + EPS]


def assert_day_ended(summary, site: FakeSite) -> None:
    for venue, vs in summary.venues.items():
        assert vs.stopped == "ended", (venue, vs.stopped)
        assert (vs.failures, vs.wrong_meeting) == (0, 0), (venue, vs)
    assert set(summary.venues) == set(site.meetings)


def assert_no_misses(misses: dict[str, list[str]], site: FakeSite) -> None:
    """Every kind of miss at once, so a failure shows the floor and the hot window together."""
    races = sum(len(r) for r in site.meetings.values())
    found = {name: found for name, found in misses.items() if found}
    assert not found, " | ".join(
        f"{name}: {len(found)} (of {races} races): {found[:4]}" for name, found in found.items()
    )


def race_misses(site: FakeSite, store: FakeStore, cfg: dict) -> dict[str, list[str]]:
    floor_s = cfg["hot_races"].get("floor_s", OWNER_FLOOR_S)
    return {
        f"(i) gap over floor_s {floor_s} s from 09:30 to T-15": floor_misses(site, cfg),
        f"(ii) gap over MAX_GAP_S {MAX_GAP_S} s while hot": hot_misses(site, cfg),
        "(ii) health.gaps_inside over MAX_GAP_S": health_misses(site, store, cfg),
    }


# ---------------------------------------------------------------- logger.yaml


def test_logger_yaml_ships_the_floor_and_keeps_the_rate_pools_and_hot_rule():
    shipped = copy.deepcopy(load_country_yaml("japan", "logger"))
    jra = source_config(shipped, "jra")
    hot = jra["hot_races"]
    assert hot.get("floor_s") == 300, f"JRA hot_races.floor_s {hot.get('floor_s')!r}: the 5 minutes"
    # what the owner kept: the hot window, the cycles, the windows, the rate, the pools, the costs
    assert {k: v for k, v in hot.items() if k != "floor_s"} == {
        "within_min": 20,
        "after_post_min": 5,
        "all_every_s": 300,
    }, hot
    assert jra["windows"] == CONFIG["windows"], jra["windows"]
    assert jra["min_poll_interval_s"] == 20
    assert sorted(jra["pools"]) == sorted(TIER1), jra["pools"]
    assert jra["requests"] == {"navigation": 2, "per_race": 4}, jra["requests"]
    assert jra_rate() == 1.0, f"jra.yaml {HOST} at {jra_rate()} requests a second"
    assert MAX_GAP_S == 120, "the health line's T-20 criterion is the owner's (#34)"
    # NAR's schedule keeps its own hot_races: no floor
    nar = source_config(shipped, "nar")
    assert "floor_s" not in nar["hot_races"], f"NAR's hot_races {nar['hot_races']}"


# ---------------------------------------------------------------- plan_round


def _spaced(now: datetime, minutes: float) -> dict[int, datetime]:
    """12 races 30 minutes apart, race 5 posting `minutes` after `now`: race 5 is hot, races 1-4
    are off and out of their hot window, races 6-12 are not hot yet."""
    return {n: now + timedelta(minutes=minutes + 30 * (n - 5)) for n in range(1, 13)}


def _states(now, posts, every: float, ages: dict[tuple[str, int], float | None], order):
    """VenueStates captured in the previous round (every race fresh, no full cycle due), but
    `ages`: (venue, race) -> seconds since its latest capture's round (None: never captured)."""
    out = []
    for venue in order:
        captured = {n: now - timedelta(seconds=every) for n in posts[venue]}
        for (v, n), age in ages.items():
            if v != venue:
                continue
            if age is None:
                captured.pop(n)
            else:
                captured[n] = now - timedelta(seconds=age)
        state = VenueState(
            code=venue,
            post_times=posts[venue],
            last_capture=now - timedelta(seconds=every),
            cycle_start=now - timedelta(seconds=100),
            cycle_left=None,
        )
        state.captured = captured  # the field #387 adds; set after construction
        out.append(state)
    return out


def _check(plan, now, posts, cfg, rps, want: set[tuple[str, int]], nearest, where) -> None:
    got = {(job.venue, n) for job in plan for n in job.races}
    venues = [job.venue for job in plan]
    assert len(venues) == len(set(venues)), f"{where}: two jobs for one venue: {plan}"
    for job in plan:
        assert job.races and len(set(job.races)) == len(job.races), f"{where}: {job}"
        assert job.requests == cost(len(job.races), cfg), f"{where}: {job}"
    budget = every_s_at(now, cfg) * rps
    total = sum(job.requests for job in plan)
    assert total <= budget + EPS, f"{where}: {total} requests for a budget of {budget}"
    assert plan and plan[0].venue == nearest and 5 in plan[0].races, f"{where}: first {plan[:1]}"
    hot_out = sorted(
        (v, n)
        for v, races in posts.items()
        for n, post in races.items()
        if is_hot(post, now, cfg) and (v, n) not in got
    )
    assert got == want, (
        f"{where}: planned {sorted(got)}, expected {sorted(want)}; hot left out {hot_out}"
    )


def test_a_stale_race_outranks_the_farther_hot_races_but_never_the_nearest():
    now = jst(12, 0, 7)
    cfg = config()
    floor_s = cfg["hot_races"]["floor_s"]
    # Three venues, one hot race each (race 5), budget 20 x 1.0: the three hot races cost 18, so
    # under #24's order no other race fits. With a race whose latest capture is floor_s old, the
    # plan is: the nearest hot race (6), the stale race (4 at a venue in the round, else 6), then
    # the other hot races nearest first while they fit.
    # (rps, stale races as (rank of their venue by hot post: 0 nearest), race), expected plan as
    # (rank, race); a venue whose hot race is left out still gets its stale race.
    cases = [
        (1.0, [(0, 8)], [(0, 5), (0, 8), (1, 5)]),  # 6 + 4 + 6 = 16; the third hot race: 22
        (1.0, [(1, 8)], [(0, 5), (1, 8), (1, 5)]),  # 6 + 6 + 4 = 16; then 22
        (1.0, [(2, 8)], [(0, 5), (2, 8), (1, 5)]),  # 6 + 6 + 6 = 18; its own hot race: 22
        (1.0, [(1, 9), (2, 8)], [(0, 5), (1, 9), (2, 8)]),  # 18: both farther hot races out
        (1.0, [(0, 8), (0, 9)], [(0, 5), (0, 8), (0, 9), (1, 5)]),  # 6 + 4 + 4 + 6 = 20
        (1.25, [(2, 8)], [(0, 5), (2, 8), (1, 5), (2, 5)]),  # budget 25: 22, every hot race fits
        (0.5, [(1, 8)], [(0, 5)]),  # budget 10: the nearest hot race, then no 6 fits beside it
    ]
    assignments = [
        [("TOK", 3), ("KYO", 9), ("FKS", 16)],
        [("FKS", -2), ("TOK", 7), ("KYO", 19)],  # the nearest is off 2 minutes ago, still hot
    ]
    for assignment in assignments:
        posts = {venue: _spaced(now, minutes) for venue, minutes in assignment}
        ranked = [venue for venue, _ in sorted(assignment, key=lambda a: abs(a[1]))]
        for order in ([v for v, _ in assignment], [v for v, _ in assignment][::-1]):
            for rps, stale, want in cases:
                ages = {(ranked[rank], n): float(floor_s) for rank, n in stale}
                states = _states(now, posts, 20, ages, order)
                plan = plan_round(now, states, cfg, rps)
                where = f"rps {rps}, posts {assignment} given as {order}, stale {sorted(ages)}"
                expected = {(ranked[rank], n) for rank, n in want}
                _check(plan, now, posts, cfg, rps, expected, ranked[0], where)

    # The field itself: empty by default, kept as given
    assert VenueState(code="TOK").captured == {}
    at = {3: now}
    assert VenueState(code="TOK", captured=at).captured == {3: now}


def test_a_race_is_stale_from_floor_s_minus_every_s_and_not_at_half_floor_s():
    now = jst(12, 0, 7)
    assignment = [("TOK", 3), ("KYO", 9), ("FKS", 16)]
    posts = {venue: _spaced(now, minutes) for venue, minutes in assignment}
    three_hot = {("TOK", 5), ("KYO", 5), ("FKS", 5)}
    # FKS race 8 (not hot) is the candidate. Budgets are 20 (every_s x rps). Stale: it goes
    # in beside the nearest hot race (6 + 6), KYO's hot race (6) follows and FKS's own (4) does
    # not fit (22). Not stale: the three hot races (18) leave 2, and no full cycle is due.
    stale_plan = {("TOK", 5), ("FKS", 8), ("KYO", 5)}
    # (floor_s, every_s, rps, age of FKS 8's latest capture in s (None: never captured), stale)
    cases = [
        (300, 20, 1.0, 280, True),  # floor_s - every_s
        (300, 20, 1.0, 300, True),
        (300, 20, 1.0, 1000, True),
        (300, 20, 1.0, 150, False),  # floor_s / 2
        (300, 20, 1.0, 40, False),
        (300, 20, 1.0, None, False),  # never captured: the full cycle's, not the floor's
        (240, 20, 1.0, 220, True),
        (240, 20, 1.0, 120, False),
        (600, 20, 1.0, 580, True),
        (600, 20, 1.0, 300, False),  # 300 s is not stale when the floor is 600 s
        (300, 30, 2 / 3, 270, True),  # a 30 s window: stale one round (30 s) before 300 s
        (300, 30, 2 / 3, 150, False),
    ]
    for floor_s, every, rps, age, stale in cases:
        windows = [{"from": "07:00", "every_s": 300}, {"from": "09:30", "every_s": every}]
        cfg = config(windows=windows, hot_races={"floor_s": floor_s})
        for order in (["TOK", "KYO", "FKS"], ["FKS", "KYO", "TOK"]):
            states = _states(now, posts, every, {("FKS", 8): age}, order)
            plan = plan_round(now, states, cfg, rps)
            where = f"floor_s {floor_s}, every_s {every}, FKS 8 captured {age} s ago"
            _check(plan, now, posts, cfg, rps, stale_plan if stale else three_hot, "TOK", where)


# ---------------------------------------------------------------- run_day: the race day


@pytest.mark.parametrize(
    ("venues", "card_name"),
    [(2, "358"), (3, "358"), (2, "30-min"), (3, "358-later")],
    ids=["2-venues-358", "3-venues-358", "2-venues-30-min", "3-venues-358-later"],
)
def test_at_one_request_a_second_every_race_keeps_the_floor_and_hot_races_max_gap_s(
    tmp_path, venues, card_name
):
    cfg = jra_schedule()
    rps = jra_rate()
    assert rps == 1.0 and cfg["requests"] == {"navigation": 2, "per_race": 4}, (rps, cfg)
    assert sorted(cfg["pools"]) == sorted(TIER1) and cfg.get("tier2") and cfg.get("results")
    meetings = card(venues, *CARDS[card_name])
    site, store, summary = simulate(meetings, cfg, rps, 1.0 / rps, tmp_path / "jra-block.txt")
    assert_day_ended(summary, site)
    kinds = {c.kind for c in site.calls}
    assert {"results", "tier2"} <= kinds, f"results polls and tier 2 ride on the rounds: {kinds}"
    assert_no_misses(race_misses(site, store, cfg), site)


# ---------------------------------------------------------------- run_day: exact rounds


def assert_rounds_fit_budget(site: FakeSite, cfg: dict) -> None:
    """test_schedule.py's: every round (the tier-1 captures at one instant) fits the budget of
    the window in force, or is the single hot job."""
    rps = site.requests_per_second
    rounds: dict[datetime, list[Call]] = defaultdict(list)
    for call in site.calls:
        if call.kind == "capture":
            rounds[call.t].append(call)
    for t, group in sorted(rounds.items()):
        where = f"round at {hhmm(t)} (rps {rps})"
        every = every_s_at(t, cfg)
        assert every is not None, f"{where}: a capture before the first window"
        venues = [c.venue for c in group]
        assert len(venues) == len(set(venues)), f"{where}: a venue captured twice: {venues}"
        planned = sum(cost(len(c.races), cfg) for c in group)
        if planned > every * rps + EPS:
            (call,) = group
            post = site.meetings[call.venue][call.races[0]]
            assert len(call.races) == 1 and is_hot(post, t, cfg), f"{where}: {planned} requests"
            assert every * rps < cost(1, cfg), f"{where}: over the budget of {every * rps}"


def nearest_hot(site: FakeSite, cfg: dict, t: datetime) -> set[tuple[str, int]]:
    """The hot races at `t` with the smallest |post - t| (more than one: a tie)."""
    hot = [
        (abs(secs(post - t)), venue, n)
        for venue, races in site.meetings.items()
        for n, post in races.items()
        if is_hot(post, t, cfg)
    ]
    if not hot:
        return set()
    best = min(d for d, *_ in hot)
    return {(venue, n) for d, venue, n in hot if d <= best + EPS}


def nearest_misses(site: FakeSite, cfg: dict) -> list[str]:
    """(iii): rounds from 09:30 (every every_s, start to start) in which some race is hot and the
    nearest hot race of all was not captured at the round's start."""
    at: dict[datetime, set[tuple[str, int]]] = defaultdict(set)
    for c in site.calls:
        if c.kind == "capture":
            at[c.t] |= {(c.venue, n) for n in c.races}
    hot = cfg["hot_races"]
    last = max(
        post + timedelta(minutes=hot["after_post_min"])
        for races in site.meetings.values()
        for post in races.values()
    )
    out = []
    t = jst(9, 30)
    while t <= last:
        nearest = nearest_hot(site, cfg, t)
        if nearest and not nearest & at.get(t, set()):
            got = sorted(at.get(t, set()))
            out.append(f"{hhmm(t)}: nearest {sorted(nearest)}, captured {got}")
        t += timedelta(seconds=every_s_at(t, cfg))
    return out


@pytest.mark.parametrize(
    ("venues", "card_name", "changes"),
    [
        (3, "358", {}),
        (3, "358-later", {}),
        (2, "358", {"hot_races": {"floor_s": 240}}),
        (3, "358", {"windows": [{"from": "07:00", "every_s": 300},
                                {"from": "09:30", "every_s": 30}]}),
    ],
    ids=["3-venues-358", "3-venues-358-later", "2-venues-floor-240", "3-venues-30-s-rounds"],
)  # fmt: skip
def test_in_exact_rounds_the_nearest_hot_race_keeps_every_round_under_the_floor(
    tmp_path, venues, card_name, changes
):
    cfg = config(jra_schedule(), **changes)
    rps = jra_rate()
    meetings = card(venues, *CARDS[card_name])
    site, store, summary = simulate(meetings, cfg, rps, 0.0, tmp_path / "jra-block.txt")
    assert_day_ended(summary, site)
    assert_rounds_fit_budget(site, cfg)
    misses = {"(iii) rounds without the nearest hot race": nearest_misses(site, cfg)}
    assert_no_misses(misses | race_misses(site, store, cfg), site)
