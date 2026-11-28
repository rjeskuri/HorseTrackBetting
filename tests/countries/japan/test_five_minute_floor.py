"""The 60 s hot grain, the 300 s floor and the three-venue JRA rate (ticket #387).

#358 (the owner's rule 2) asks for every JRA race at least every 5 minutes until its T-15, while
the 09:30 window (20 s rounds) captured each hot race every round. At jra.yaml's 1 request a second
three venues' hot races starve the other races: #358's simulation measured gaps up to 1,140 s
(three venues) and 404 s (two) from 09:30 to T-15.

The owner's decision (2026-11-29 01:20 JST on #387, replacing the 2026-11-28 answer (a)): "go faster
than 1 req/second to get as close as possible. Since our data is mostly looking at 1 minute
intervals for training we don't need the 20 seconds logger, we want to make sure we aren't slow on
all the venues."
1. Hot races (the existing hot window) are captured every 60 s; every other race keeps the 300 s
   floor from 09:30 to its hot window (measured to its T-15, as #358). No venue may fall behind.
2. On three-venue days the JRA rate goes above 1 req/s, to the lowest rate at which the simulation
   holds both, under 2 req/s.
3. The simulation is re-run on #358's cards (the PR and docs/ledger.md carry the figures).
4. A 403/429 does not end the day's logging: back off to 1 req/s, then to the 300 s floor only, and
   alert through the health line.

Contract pinned here (the coder follows it), on top of test_schedule.py's:

Config (logger.yaml's JRA schedule, its top level: `source_config(config, "jra")`)
- `hot_races.every_s` (s): the hot grain. A hot race (post - within_min <= now <= post +
  after_post_min) is captured at least every every_s. Shipped: 60. `hot_races.floor_s` (s): the
  floor. A race not hot whose post is still ahead is captured at least every floor_s. Shipped: 300.
  A schedule without every_s captures its hot races every round (#24; NAR's, and the CONFIG of
  test_schedule.py and test_day_before_schedule.py); one without floor_s has no floor (the full
  cycles alone, as before). The tests vary both: neither may be a constant in the code.
- `venue_rates`: a list of `{min_venues: N, requests_per_second: R}`. A day that logs V venues
  (given or discovered) runs at the R of the entry with the greatest N <= V; a day no entry
  applies to runs at its site's own rate (jra.yaml's www.jra.go.jp). Shipped: an entry that gives
  three venues a rate above jra.yaml's and under 2; none for one venue (Ruling: two venues stay at
  jra.yaml's rate unless the simulation fails there, and then an entry of their own, the lowest
  rate on the same grid). The tests change the rate: it comes from this key, never a constant.
- `on_block`: `cooldown_min` a number, 0 < N <= 30 (Ruling: NAR's 15 is the precedent; a longer
  pause would end the day by other means), and `resume_requests_per_second` 1.0 (the owner's 1
  req/s). Any other key (the floor-only step) is the coder's.
- Unchanged: the windows (07:00 every 300 s, 09:30 every 20 s), within_min 20, after_post_min 5,
  all_every_s 300, min_poll_interval_s 20, the five tier-1 pools at {navigation 2, per_race 4},
  jra.yaml's 1.0 requests a second, and the health line's MAX_GAP_S 120.
- NAR's schedule (`source_config(config, "nar")`) inherits none of venue_rates, every_s, floor_s.

VenueState.captured (horsetrack.countries.japan.logger.schedule): dict[int, datetime], default
empty: race number -> when its latest successful capture started (run_day keeps it; whether it
stamps the round's start or the call's own is the coder's). A race missing from it has never been
captured: a hot one is due, a race not hot is the full cycle's; so a fresh VenueState plans as
test_schedule.py says.

plan_round(now, venues, config, rps) with every_s H, floor_s F and e the every_s of the window in
force (a race's age: now - captured[race]):
- must: a hot race whose age is >= H - e (the next round would be H late); a race not hot whose post
  is ahead (now < post) and whose age is >= F - e.
- fresh: a hot race whose age is <= H / 2; a race not hot whose age is <= F / 2. Between the two the
  coder decides (a Ruling line on the PR), so a race may be captured early to absorb its offset in
  the round.
- Must races come first: a must race left out of the plan would not fit on top of the plan's must
  races alone (navigation once per venue), so a fresh race never takes budget a must race needed,
  and when every must race fits, every must race is planned. Their order among themselves and what
  fills the spare budget are not pinned. Budget (every_s x rps), one Job per venue, Job.requests,
  min_poll_interval_s and the single hot job are test_schedule.py's.

run_day(..., resume=factory): `resume(requests_per_second)` returns a fresh site at that rate (as
for NAR's cooldown; log-day: a site over `make_fetcher(source, requests_per_second=rate)`).
run_day uses it:
- for venue_rates: once the day's venues are known, a day whose rate (above) is not its site's own
  runs on `resume(rate)`, so every capture from 09:30 is sent at that rate. A day at its site's own
  rate makes no new site. Without `resume` a day runs on the site it was given.
- after a block (a CircuitOpen; the site that raised it is never called again: its breaker is open):
  no call for on_block.cooldown_min minutes, then the day goes on, from within 320 s, on
  `resume(resume_requests_per_second)`, with the full schedule (the hot grain included) and no
  later switch back to the venue_rates rate. After a second block and its cooldown the day goes on
  at no more than that rate with the floor only: every race at least every floor_s up to its post,
  hot races included, and no race captured more than twice in any floor_s span (no hot grain).
  Neither block ends the day: every venue still ends ('ended'). Ruling: a third block ends the
  day as a block did before #387 (no further call, every venue still running 'circuit_open', the
  marker names the host and the race date): the owner's ladder has two steps, and a host that
  refuses the floor-only step after two cooldowns is not polled on until the stop.
- the health line alerts a block: health.health_lines gives every venue of a day with a block an
  ALERT whose reasons name it (the word "block", or the status 403/429). The venue line's fields
  are unchanged (test_log_day_cli.py's pattern).

The race-day simulation (the DoD): logger.yaml's JRA schedule as shipped (tier 2 and results polling
included), a run from 07:00 (the first window: no day-before probe), synthetic 12-race cards:
#358's (first posts 09:50, 10:00, 10:10, every 34 minutes) and the same five minutes later, on two
and three venues. The site starts at jra.yaml's rate and `resume` makes sites at any rate. Every
site call moves the fake clock by its requests / its site's rate (1 s a request at 1 req/s;
jra.yaml's +-10 % jitter is not modelled, as in #358's), so a round lasts as long as what it sends
(results polls, post-time reads and tier 2 included). The bounds are on the instants the captures
started, with no slack (Ruling: the models read 1-minute boards, and the rate is the owner's
lever):
- the floor: for every race no gap over floor_s between 09:30, its tier-1 captures from 09:30 to its
  T-15, and its T-15;
- the hot grain: for every race no gap over every_s between its hot window's start, its captures in
  the window and the window's end; the health line's own metric (health.gaps_inside of the written
  tier-1 rows) at most MAX_GAP_S, and no ALERT line for the day;
- the rate: every capture from 09:30 until the first venue ends is sent at the day's rate;
- the lowest rate (Ruling: a 0.05 req/s grid): for a venue count whose rate is above jra.yaml's
  (three venues always), the simulation at that rate minus 0.05, 0.10, ... and at jra.yaml's rate
  itself misses a bound on at least one of the two cards.

Not pinned: the order of jobs, which race fills spare budget, tier 2's share, races already off,
the cooldown's length within its range, the marker lines of a block the day recovers from.
Everything is synthetic: no HTTP, no real sleep, no scraped page.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from horsetrack.core.config import load_country_yaml
from horsetrack.core.ingest.fetcher import CircuitOpen
from horsetrack.countries.japan.jra.capture import MeetingCapture
from horsetrack.countries.japan.logger.health import MAX_GAP_S, gaps_inside, health_lines

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
FLOOR_END_MIN = 15  # #358's rule 2: the floor holds until the race's T-15
OWNER_HOT_S = 60  # the owner's hot grain (logger.yaml hot_races.every_s)
OWNER_FLOOR_S = 300  # the owner's 5 minutes (logger.yaml hot_races.floor_s)
OWNER_RESUME_RPS = 1.0  # "back off to 1 req/s" (logger.yaml on_block.resume_requests_per_second)
MAX_RATE = 2.0  # "keep it under 2 req/s"
STEP = 0.05  # Ruling: the grid the lowest rate is found on
COOLDOWN_MAX_MIN = 30  # Ruling: the longest pause after a block
CARD_NAMES = ("358", "358-later")

# The schedule the plan_round tests start from: the shipped one's shape, with the owner's keys.
CONFIG = {
    "pools": list(TIER1),
    "min_poll_interval_s": 20,
    "windows": [{"from": "07:00", "every_s": 300}, {"from": "09:30", "every_s": 20}],
    "stop": "16:45",
    "after_last_race_min": 10,
    "hot_races": {"within_min": 20, "after_post_min": 5, "all_every_s": 300, "every_s": 60,
                  "floor_s": 300},
    "requests": {"navigation": 2, "per_race": 4},
    "max_wrong_meeting": 3,
    "on_block": {"cooldown_min": None},
}  # fmt: skip


def config(**changes) -> dict:
    """CONFIG with some values changed (a dict value is merged in)."""
    cfg = copy.deepcopy(CONFIG)
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
    """jra.yaml's rate for the odds host."""
    http = load_country_yaml("japan", "jra")["http"]
    return float(http["hosts"].get(HOST, http["default"])["requests_per_second"])


def day_rate(cfg: dict, venues: int, base: float) -> float:
    """The venue_rates rule, from first principles: the entry with the greatest min_venues <=
    venues, else the site's own rate."""
    entries = [e for e in cfg.get("venue_rates") or [] if int(e["min_venues"]) <= venues]
    if not entries:
        return base
    return float(max(entries, key=lambda e: int(e["min_venues"]))["requests_per_second"])


def with_rate(cfg: dict, venues: int, rate: float) -> dict:
    """`cfg` with the rate a day of `venues` venues runs at set to `rate`."""
    cfg = copy.deepcopy(cfg)
    entries = list(cfg.get("venue_rates") or [])
    applies = [e for e in entries if int(e["min_venues"]) <= venues]
    if applies:
        max(applies, key=lambda e: int(e["min_venues"]))["requests_per_second"] = rate
    else:
        entries.append({"min_venues": venues, "requests_per_second": rate})
    cfg["venue_rates"] = entries
    return cfg


def hot_grain(cfg: dict) -> float:
    return float(cfg["hot_races"].get("every_s", OWNER_HOT_S))


def floor_grain(cfg: dict) -> float:
    return float(cfg["hot_races"].get("floor_s", OWNER_FLOOR_S))


def cooldown_of(cfg: dict) -> float:
    c = cfg["on_block"].get("cooldown_min")
    assert isinstance(c, (int, float)) and not isinstance(c, bool) and c > 0, (
        f"on_block.cooldown_min {c!r}: a block must pause, then resume (#387)"
    )
    return float(c)


# ---------------------------------------------------------------- time


def jst(h: int, m: int = 0, s: int = 0, day: date = DAY) -> datetime:
    """The aware UTC instant of a Tokyo wall-clock time on `day`."""
    return datetime(day.year, day.month, day.day, h, m, s, tzinfo=JST).astimezone(UTC)


def secs(delta: timedelta) -> float:
    return delta.total_seconds()


def hhmm(t: datetime) -> str:
    return t.astimezone(JST).strftime("%H:%M:%S JST")


def is_hot(post: datetime, t: datetime, cfg: dict) -> bool:
    hot = cfg["hot_races"]
    return (
        post - timedelta(minutes=hot["within_min"])
        <= t
        <= post + timedelta(minutes=hot["after_post_min"])
    )


def cost(n_races: int, cfg: dict) -> int:
    return cfg["requests"]["navigation"] + cfg["requests"]["per_race"] * n_races


def set_cost(races: set[tuple[str, int]], cfg: dict) -> int:
    """What capturing `races` costs in one round: navigation once per venue."""
    per_venue: dict[str, int] = {}
    for venue, _ in races:
        per_venue[venue] = per_venue.get(venue, 0) + 1
    return sum(cost(k, cfg) for k in per_venue.values())


# ---------------------------------------------------------------- logger.yaml


def test_logger_yaml_ships_the_hot_grain_the_floor_the_three_venue_rate_and_the_back_off():
    shipped = copy.deepcopy(load_country_yaml("japan", "logger"))
    jra = source_config(shipped, "jra")
    hot = jra["hot_races"]
    assert hot.get("every_s") == OWNER_HOT_S, f"JRA hot_races.every_s {hot.get('every_s')!r}"
    assert hot.get("floor_s") == OWNER_FLOOR_S, f"JRA hot_races.floor_s {hot.get('floor_s')!r}"
    rest = {k: v for k, v in hot.items() if k not in ("every_s", "floor_s")}
    assert rest == {"within_min": 20, "after_post_min": 5, "all_every_s": 300}, hot
    assert jra["windows"] == CONFIG["windows"], jra["windows"]
    assert jra["min_poll_interval_s"] == 20
    assert sorted(jra["pools"]) == sorted(TIER1), jra["pools"]
    assert jra["requests"] == {"navigation": 2, "per_race": 4}, jra["requests"]
    assert MAX_GAP_S == 120, "the health line's T-20 criterion is the owner's (#34)"

    base = jra_rate()
    assert base == 1.0, f"jra.yaml {HOST} at {base} requests a second"
    entries = jra.get("venue_rates")
    assert isinstance(entries, list) and entries, f"JRA venue_rates {entries!r}"
    for entry in entries:
        assert set(entry) == {"min_venues", "requests_per_second"}, entry
        assert int(entry["min_venues"]) >= 2, f"{entry}: a one-venue day keeps jra.yaml's rate"
        assert base < float(entry["requests_per_second"]) < MAX_RATE, entry
    r3, r2 = day_rate(jra, 3, base), day_rate(jra, 2, base)
    assert base < r3 < MAX_RATE, f"three venues at {r3} requests a second"
    assert base <= r2 <= r3, f"two venues at {r2}, three at {r3}"
    assert day_rate(jra, 1, base) == base

    on_block = jra["on_block"]
    assert 0 < cooldown_of(jra) <= COOLDOWN_MAX_MIN, on_block
    assert on_block.get("resume_requests_per_second") == OWNER_RESUME_RPS, on_block

    nar = source_config(shipped, "nar")
    assert "venue_rates" not in nar, f"NAR inherited JRA's venue_rates: {nar.get('venue_rates')}"
    assert not {"every_s", "floor_s"} & set(nar["hot_races"]), f"NAR's {nar['hot_races']}"


# ---------------------------------------------------------------- plan_round


def _spaced(now: datetime, minutes: float) -> dict[int, datetime]:
    """12 races 30 minutes apart, race 5 posting `minutes` after `now`: race 5 is hot, races 1-4
    are off and out of their hot window, races 6-12 are not hot and post later."""
    return {n: now + timedelta(minutes=minutes + 30 * (n - 5)) for n in range(1, 13)}


def _states(now, posts, every: float, ages: dict, order, default_age: float) -> list[VenueState]:
    """VenueStates with every race captured `default_age` s ago, but `ages` ((venue, race) -> s);
    each venue last captured one round ago, its full cycle not due."""
    out = []
    for venue in order:
        captured = {n: now - timedelta(seconds=default_age) for n in posts[venue]}
        for (v, n), age in ages.items():
            if v == venue:
                captured[n] = now - timedelta(seconds=age)
        state = VenueState(
            code=venue,
            post_times=posts[venue],
            last_capture=now - timedelta(seconds=every),
            cycle_start=now - timedelta(seconds=100),
            cycle_left=None,
        )
        state.captured = captured  # the field #387 adds
        out.append(state)
    return out


def must_races(now, states, cfg: dict, every: float) -> set[tuple[str, int]]:
    """The contract's must races, from first principles."""
    h, f = hot_grain(cfg), floor_grain(cfg)
    out = set()
    for s in states:
        for n, post in s.post_times.items():
            age = secs(now - s.captured[n])
            if is_hot(post, now, cfg):
                if age >= h - every - EPS:
                    out.add((s.code, n))
            elif now < post and age >= f - every - EPS:
                out.add((s.code, n))
    return out


def check_must_first(plan, now, states, cfg: dict, rps: float, every: float, where: str) -> set:
    """The plan is well formed, within its budget, and no must race left out fits on top of the
    plan's must races. Returns the planned races."""
    venues = [job.venue for job in plan]
    assert len(venues) == len(set(venues)), f"{where}: two jobs for one venue: {plan}"
    posts = {s.code: s.post_times for s in states}
    for job in plan:
        assert job.races and len(set(job.races)) == len(job.races), f"{where}: {job}"
        assert set(job.races) <= set(posts[job.venue]), f"{where}: {job}"
        assert job.requests == cost(len(job.races), cfg), f"{where}: {job}"
    budget = every * rps
    total = sum(job.requests for job in plan)
    assert total <= budget + EPS, f"{where}: {total} requests for a budget of {budget}"
    got = {(job.venue, n) for job in plan for n in job.races}
    must = must_races(now, states, cfg, every)
    kept = got & must
    for race in sorted(must - got):
        assert set_cost(kept | {race}, cfg) > budget + EPS, (
            f"{where}: must race {race} left out, though it fits beside the planned must races "
            f"{sorted(kept)} (planned {sorted(got)}, budget {budget})"
        )
    return got


# (hot every_s H, floor_s F, the 09:30 window's every_s e, rps, ages by (rank of the venue by
# |post - now| of its race 5, race), the must races by (rank, race))
PLAN_CASES = [
    (60, 300, 20, 1.0, {(0, 5): 40, (2, 8): 280}, {(0, 5), (2, 8)}),  # 12 of 20
    # 22 of 20: three hot musts and a floor must do not all fit
    (60, 300, 20, 1.0, {(0, 5): 40, (1, 5): 45, (2, 5): 60, (2, 8): 290},
     {(0, 5), (1, 5), (2, 5), (2, 8)}),
    (60, 300, 20, 0.3, {(1, 5): 40}, {(1, 5)}),  # 6 of 6: the must, not the nearest (fresh)
    (60, 300, 20, 0.5, {(0, 8): 280, (1, 9): 300}, {(0, 8), (1, 9)}),  # the fresh hot races wait
    (60, 300, 20, 1.0, {(0, 8): 285, (0, 9): 290, (1, 10): 299, (2, 11): 300},
     {(0, 8), (0, 9), (1, 10), (2, 11)}),  # 22 of 20
    (60, 300, 20, 0.8, {(0, 5): 40, (0, 8): 280, (2, 8): 280}, {(0, 5), (0, 8), (2, 8)}),  # 16/16
    (90, 300, 20, 0.6, {(0, 5): 40, (1, 5): 70, (2, 8): 280}, {(1, 5), (2, 8)}),  # H 90: 40 fresh
    (60, 600, 20, 0.6, {(0, 5): 40, (2, 8): 290, (1, 9): 580}, {(0, 5), (1, 9)}),  # F 600
    (90, 300, 30, 0.4, {(0, 5): 60, (2, 8): 270}, {(0, 5), (2, 8)}),  # 30 s rounds: 12 of 12
    (90, 300, 30, 0.4, {(0, 5): 45, (1, 5): 60, (2, 8): 150}, {(1, 5)}),  # 45 and 150 are fresh
]  # fmt: skip
ASSIGNMENTS = [
    [("TOK", 3), ("KYO", 9), ("FKS", 16)],
    [("FKS", -2), ("TOK", 7), ("KYO", 19)],  # the nearest is off 2 minutes ago, still hot
]


def test_must_races_come_first_and_a_fresh_race_never_takes_their_budget():
    now = jst(12, 0, 7)
    for assignment in ASSIGNMENTS:
        posts = {venue: _spaced(now, minutes) for venue, minutes in assignment}
        ranked = [venue for venue, _ in sorted(assignment, key=lambda a: abs(a[1]))]
        for h, f, e, rps, by_rank, want in PLAN_CASES:
            windows = [{"from": "07:00", "every_s": 300}, {"from": "09:30", "every_s": e}]
            cfg = config(windows=windows, hot_races={"every_s": h, "floor_s": f})
            ages = {(ranked[rank], n): float(age) for (rank, n), age in by_rank.items()}
            expected = {(ranked[rank], n) for rank, n in want}
            for order in ([v for v, _ in assignment], [v for v, _ in assignment][::-1]):
                states = _states(now, posts, e, ages, order, default_age=e)
                where = f"H {h}, F {f}, e {e}, rps {rps}, posts {assignment}, ages {ages}"
                assert must_races(now, states, cfg, e) == expected, where  # the table itself
                got = check_must_first(plan_round(now, states, cfg, rps), now, states, cfg, rps,
                                       e, where)  # fmt: skip
                if abs(set_cost(expected, cfg) - e * rps) < EPS:  # the musts fill the budget
                    assert got == expected, f"{where}: planned {sorted(got)}"

    # Without hot_races.every_s and floor_s (NAR's schedule, the older tests' CONFIG) hot races
    # come first every round (#24), however recent their latest capture.
    cfg = config()
    cfg["hot_races"] = {
        k: v for k, v in cfg["hot_races"].items() if k not in ("every_s", "floor_s")
    }
    for assignment in ASSIGNMENTS:
        posts = {venue: _spaced(now, minutes) for venue, minutes in assignment}
        order = [v for v, _ in assignment]
        states = _states(now, posts, 20, {}, order, default_age=20)
        plan = plan_round(now, states, cfg, 1.0)  # 20 requests: the three hot races cost 18
        got = {(job.venue, n) for job in plan for n in job.races}
        assert got >= {(v, 5) for v in order}, f"without every_s, {assignment}: {sorted(got)}"

    # The field itself: empty by default, kept as given
    assert getattr(VenueState(code="TOK"), "captured", None) == {}, "VenueState.captured"
    assert VenueState(code="TOK", captured={3: now}).captured == {3: now}


def test_at_the_three_venue_rate_a_round_holds_every_venue_s_hot_race():
    shipped = jra_schedule()
    rate = day_rate(shipped, 3, jra_rate())
    assert rate > jra_rate(), f"three venues at {rate} requests a second"
    every = float(shipped["windows"][-1]["every_s"])
    h = hot_grain(shipped)
    now = jst(12, 0, 7)
    for assignment in ASSIGNMENTS:
        posts = {venue: _spaced(now, minutes) for venue, minutes in assignment}
        order = [v for v, _ in assignment]
        ages = {(v, 5): h - every for v in order}  # all three hot races must, the rest fresh
        states = _states(now, posts, every, ages, order, default_age=every)
        where = f"shipped schedule at {rate} req/s, posts {assignment}"
        got = check_must_first(plan_round(now, states, shipped, rate), now, states, shipped, rate,
                               every, where)  # fmt: skip
        assert got >= {(v, 5) for v in order}, f"{where}: planned {sorted(got)}"


# ---------------------------------------------------------------- the simulated race day


def card(venues: int, first: list[tuple[int, int]], every_min: int, races: int = 12):
    """`venues` meetings of `races` races, `every_min` minutes apart, first posts `first`."""
    codes = ["TOK", "KYO", "FKS"][:venues]
    return {
        code: {n: jst(*f) + timedelta(minutes=every_min * (n - 1)) for n in range(1, races + 1)}
        for code, f in zip(codes, first, strict=False)
    }


CARDS = {
    "358": ([(9, 50), (10, 0), (10, 10)], 34),  # #358's simulation
    "358-later": ([(9, 55), (10, 5), (10, 15)], 34),  # the same density, five minutes later
}


class Runaway(BaseException):
    """run_day kept going where a day cannot (a BaseException: no `except Exception` hides it)."""


class Clock:
    """The fake clock: sleep() moves it, and so does each site call's request time."""

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
    races: tuple[int, ...] | None
    requests: int
    rps: float  # the rate of the site it went to
    status: int | None = None  # the 403/429 a scripted block answered
    refused: bool = False  # sent to a site whose breaker was open: never sent


@dataclass
class Net:
    """What every site of one run shares: the clock, the card, the call log, the scripted blocks
    ((instant, status): the first call at or after the instant is refused with that status) and
    the sites `resume` made, as (when, rate)."""

    clock: Clock
    meetings: dict[str, dict[int, datetime]]
    cfg: dict
    blocks: list[tuple[datetime, int]] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    made: list[tuple[datetime, float]] = field(default_factory=list)
    defects: list[str] = field(default_factory=list)

    def resume(self, rate: float) -> FakeSite:
        self.made.append((self.clock.t, float(rate)))
        return FakeSite(self, float(rate))


class FakeSite:
    """The JRA site of race date DAY as the scheduler sees it, at one rate: each call is recorded
    at the fake clock's time, then the clock moves by its requests / the rate. A tier-1 capture
    costs navigation + per_race x races, a tier-2 one its own, a results poll 2; a race's result is
    published 10 minutes after its post. After a block its breaker is open: every later call to it
    is refused (CircuitOpen) without being sent."""

    source = "jra"
    host = HOST

    def __init__(self, net: Net, rps: float):
        self.net, self._rps, self.broken = net, rps, False
        cfg = net.cfg
        self.nav, self.per_race = cfg["requests"]["navigation"], cfg["requests"]["per_race"]
        tier2 = cfg.get("tier2") or {"requests": {"navigation": 2, "per_race": 3}}
        self.t2_nav, self.t2_per = tier2["requests"]["navigation"], tier2["requests"]["per_race"]

    @property
    def requests_per_second(self) -> float:  # read-only: a new rate is a new site (Fetcher)
        return self._rps

    def _call(self, kind: str, venue: str | None, races, requests: int) -> Call:
        net = self.net
        t = net.clock.t
        if self.broken:
            net.calls.append(Call(t, kind, venue, races, 0, self._rps, refused=True))
            raise CircuitOpen("the circuit breaker is open", host=HOST)
        if net.blocks and net.blocks[0][0] <= t:
            status = net.blocks.pop(0)[1]
            self.broken = True
            net.calls.append(Call(t, kind, venue, races, 1, self._rps, status=status))
            net.clock.advance(1.0 / self._rps)
            raise CircuitOpen(f"HTTP {status}", http_status=status, host=HOST)
        if venue is not None and venue not in net.meetings:
            net.defects.append(f"{kind} for {venue}, which has no meeting on {DAY}")
        call = Call(t, kind, venue, races, requests, self._rps)
        net.calls.append(call)
        net.clock.advance(requests / self._rps)
        return call

    def sales_open(self) -> bool:
        self._call("sales_open", None, None, 1)
        return True

    def discover(self) -> list[str]:
        self._call("discover", None, None, 1)
        return list(self.net.meetings)

    def post_times(self, venue: str) -> dict[int, datetime]:
        self._call("post_times", venue, None, self.nav)
        return dict(self.net.meetings[venue])

    def _asked(self, venue: str, races, kind: str) -> list[int]:
        every = sorted(self.net.meetings.get(venue, {}))
        if races is None:
            return every
        asked = list(races)
        if not asked or len(set(asked)) != len(asked) or set(asked) - set(every):
            self.net.defects.append(f"{hhmm(self.net.clock.t)} {kind} of {venue} asks {asked}")
        return sorted(set(asked) & set(every))

    def capture(self, venue: str, races) -> MeetingCapture:
        wanted = self._asked(venue, races, "capture")
        spent = self.nav + self.per_race * len(wanted)
        call = self._call("capture", venue, tuple(wanted), spent)
        rows = [
            {"race_id": f"jra:{DAY.isoformat()}:{venue}:{n:02d}", "pool": pool,
             "snapshot_utc": call.t}
            for n in wanted
            for pool in self.net.cfg["pools"]
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
        return {n for n, post in self.net.meetings[venue].items() if call.t >= post + out}


class FakeStore:
    def __init__(self):
        self.writes: list[tuple[str, MeetingCapture]] = []

    def write(self, venue: str, capture: MeetingCapture) -> int:
        self.writes.append((venue, capture))
        return len(capture.rows)

    def write_post_times(self, venue: str, times: dict[int, datetime]) -> None:
        pass

    def write_published(self, venue, races, seen_utc, unpublished_utc=None) -> None:
        pass


@dataclass
class Run:
    net: Net
    store: FakeStore
    summary: object
    cfg: dict
    marker: Path

    @property
    def meetings(self) -> dict[str, dict[int, datetime]]:
        return self.net.meetings


def simulate(meetings, cfg: dict, marker: Path, blocks=()) -> Run:
    """A race day from 07:00 (the first window: no day-before probe) to its end, on a site at
    jra.yaml's rate, with `resume` making a site at any rate."""
    clock = Clock(jst(7, 0))
    net = Net(clock, meetings, cfg, blocks=sorted(blocks))
    store = FakeStore()
    site = FakeSite(net, jra_rate())
    summary = run_day(DAY, site, store, config=cfg, now=clock.now, sleep=clock.sleep,
                      block_marker=marker, resume=net.resume)  # fmt: skip
    assert net.defects == [], net.defects[:5]
    return Run(net, store, summary, cfg, marker)


class Sims:
    """One simulation per (venues, card, rate, hot_races changes), shared by the tests."""

    def __init__(self, root: Path):
        self.root, self.cache = root, {}

    def run(self, venues: int, card_name: str, rate: float | None = None, hot=None) -> Run:
        key = (venues, card_name, rate, tuple(sorted((hot or {}).items())))
        if key not in self.cache:
            cfg = jra_schedule()
            if rate is not None:
                cfg = with_rate(cfg, venues, rate)
            cfg["hot_races"].update(hot or {})
            marker = self.root / f"jra-block-{len(self.cache)}.txt"
            self.cache[key] = simulate(card(venues, *CARDS[card_name]), cfg, marker)
        return self.cache[key]


@pytest.fixture(scope="module")
def sims(tmp_path_factory) -> Sims:
    return Sims(tmp_path_factory.mktemp("sims"))


# ---------------------------------------------------------------- what the fake saw


def captured_at(run: Run, venue: str, race: int, since: datetime | None = None) -> list[datetime]:
    """When `race` of `venue` was captured in tier 1 (sent and answered), from `since` on."""
    return [
        c.t
        for c in run.net.calls
        if c.kind == "capture" and c.venue == venue and c.races and race in c.races
        and c.status is None and not c.refused and (since is None or c.t >= since)
    ]  # fmt: skip


def longest_gap(times: list[datetime], lo: datetime, hi: datetime) -> tuple[float, datetime]:
    """The longest gap between lo, the times inside [lo, hi] and hi; and where it starts."""
    edges = [lo, *sorted(t for t in times if lo <= t <= hi), hi]
    return max((secs(b - a), a) for a, b in zip(edges, edges[1:], strict=False))


def floor_misses(run: Run) -> list[str]:
    """Races with a gap over floor_s from 09:30 to their T-15."""
    floor_s, w, out = floor_grain(run.cfg), jst(9, 30), []
    for venue, races in run.meetings.items():
        for n, post in races.items():
            t15 = post - timedelta(minutes=FLOOR_END_MIN)
            if t15 <= w:
                continue
            gap, at = longest_gap(captured_at(run, venue, n), w, t15)
            if gap > floor_s + EPS:
                out.append(f"{venue} {n} (post {hhmm(post)}): {gap:.1f} s from {hhmm(at)}")
    return out


def hot_misses(run: Run) -> list[str]:
    """Races with a gap over every_s inside their hot window, its edges included."""
    hot, grain, out = run.cfg["hot_races"], hot_grain(run.cfg), []
    for venue, races in run.meetings.items():
        for n, post in races.items():
            lo = post - timedelta(minutes=hot["within_min"])
            hi = post + timedelta(minutes=hot["after_post_min"])
            gap, at = longest_gap(captured_at(run, venue, n), lo, hi)
            if gap > grain + EPS:
                out.append(f"{venue} {n} (post {hhmm(post)}): {gap:.1f} s from {hhmm(at)}")
    return out


def post_times(run: Run) -> dict[str, datetime]:
    return {
        f"jra:{DAY.isoformat()}:{venue}:{n:02d}": post
        for venue, races in run.meetings.items()
        for n, post in races.items()
    }


def tier1_rows(run: Run) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {venue: [] for venue in run.meetings}
    for venue, capture in run.store.writes:
        out[venue] += [r for r in capture.rows if r["pool"] in TIER1]
    return out


def health_misses(run: Run) -> list[str]:
    """The health line's own metric: health.gaps_inside over the written tier-1 rows."""
    rows = [r for rs in tier1_rows(run).values() for r in rs]
    gaps = gaps_inside(rows, post_times(run), run.cfg["hot_races"]["within_min"])
    return [f"{rid}: {gap:.1f} s" for rid, gap in sorted(gaps.items()) if gap > MAX_GAP_S + EPS]


def alerts(run: Run) -> list[str]:
    lines = health_lines(run.summary, tier1_rows(run), post_times(run))
    return [line for line in lines if line.startswith("ALERT")]


def first_end(run: Run) -> datetime:
    after = timedelta(minutes=run.cfg["after_last_race_min"])
    return min(max(races.values()) + after for races in run.meetings.values())


def rate_misses(run: Run, rate: float) -> list[str]:
    """Captures from 09:30 until the first venue ends that were not sent at `rate`."""
    lo, hi = jst(9, 30), first_end(run)
    return [
        f"{hhmm(c.t)} {c.venue} at {c.rps} req/s"
        for c in run.net.calls
        if c.kind == "capture" and lo <= c.t < hi and abs(c.rps - rate) > EPS
    ]


def misses(run: Run) -> dict[str, list[str]]:
    return {
        f"gap over floor_s {floor_grain(run.cfg):g} s from 09:30 to T-15": floor_misses(run),
        f"gap over every_s {hot_grain(run.cfg):g} s while hot": hot_misses(run),
        f"health.gaps_inside over MAX_GAP_S {MAX_GAP_S}": health_misses(run),
    }


def describe(found: dict[str, list[str]], run: Run) -> str:
    races = sum(len(r) for r in run.meetings.values())
    return " | ".join(f"{name}: {len(v)} of {races} races: {v[:4]}" for name, v in found.items())


def assert_day_ended(run: Run) -> None:
    assert set(run.summary.venues) == set(run.meetings), sorted(run.summary.venues)
    for venue, vs in run.summary.venues.items():
        assert vs.stopped == "ended", (venue, vs.stopped)
        assert (vs.failures, vs.wrong_meeting) == (0, 0), (venue, vs)


# ---------------------------------------------------------------- the DoD: both bounds


@pytest.mark.parametrize(
    ("venues", "card_name"),
    [(2, "358"), (2, "358-later"), (3, "358"), (3, "358-later")],
    ids=["2-venues-358", "2-venues-358-later", "3-venues-358", "3-venues-358-later"],
)
def test_every_race_keeps_the_floor_and_every_hot_race_the_hot_grain_at_the_day_s_rate(
    sims, venues, card_name
):
    run = sims.run(venues, card_name)
    base = jra_rate()
    rate = day_rate(run.cfg, venues, base)
    if venues == 3:
        assert rate > base, f"three venues at {rate} requests a second: venue_rates (#387)"
    assert_day_ended(run)
    kinds = {c.kind for c in run.net.calls}
    assert {"results", "tier2"} <= kinds, f"results polls and tier 2 ride on the rounds: {kinds}"
    assert rate_misses(run, rate) == [], rate_misses(run, rate)[:4]
    if rate == base:
        assert run.net.made == [], f"a day at its site's rate made new sites: {run.net.made}"
    found = {name: v for name, v in misses(run).items() if v}
    assert not found, f"at {rate} req/s: " + describe(found, run)
    assert alerts(run) == [], alerts(run)


def grid_below(rate: float, base: float) -> list[float]:
    """rate - STEP, rate - 2 STEP, ... while above base, then base."""
    out, k = [], 1
    while rate - k * STEP > base + EPS:
        out.append(round(rate - k * STEP, 6))
        k += 1
    return [*out, base]


def test_the_day_s_rate_is_the_lowest_on_the_grid_that_holds_both_bounds(sims):
    shipped, base = jra_schedule(), jra_rate()
    r3 = day_rate(shipped, 3, base)
    assert base < r3 < MAX_RATE, f"three venues at {r3} requests a second: above {base}, under 2"
    for venues in (3, 2):
        rate = day_rate(shipped, venues, base)
        if rate <= base + EPS:
            continue  # two venues at jra.yaml's rate: the test above holds them there
        for lower in grid_below(rate, base):
            held = []
            for card_name in CARD_NAMES:
                run = sims.run(venues, card_name, lower)
                # the day ran at the rate venue_rates gave it, not at a constant
                assert rate_misses(run, lower) == [], rate_misses(run, lower)[:4]
                held.append(not any(misses(run).values()))
            assert not all(held), (
                f"{venues} venues at {lower} req/s hold both bounds on every card: logger.yaml's "
                f"{rate} is not the lowest rate on the {STEP} grid"
            )


def test_the_floor_and_the_hot_grain_come_from_the_schedule(sims):
    # Two venues at 2 req/s with a 50 s hot grain and a 240 s floor: a constant 60 or 300 misses.
    hot = {"every_s": 50, "floor_s": 240}
    run = sims.run(2, "358", 2.0, hot=hot)
    assert_day_ended(run)
    assert rate_misses(run, 2.0) == [], rate_misses(run, 2.0)[:4]
    found = {name: v for name, v in misses(run).items() if v}
    assert not found, f"hot_races {hot}: " + describe(found, run)


# ---------------------------------------------------------------- a block does not end the day


BLOCK_ALERT = re.compile(r"block|403|429", re.IGNORECASE)


def blocked_calls(run: Run) -> list[Call]:
    return [c for c in run.net.calls if c.status is not None]


def after_cooldown(run: Run, block: Call, cooldown_min: float) -> list[Call]:
    """The calls after `block`, checked: none inside the cooldown, the first within 320 s of
    its end."""
    later = [c for c in run.net.calls if c.t > block.t]
    assert later, f"nothing after the block at {hhmm(block.t)}: the day ended"
    end = block.t + timedelta(minutes=cooldown_min)
    assert later[0].t >= end, f"a call at {hhmm(later[0].t)}, inside the cooldown to {hhmm(end)}"
    assert later[0].t < end + timedelta(seconds=320), f"resumed only at {hhmm(later[0].t)}"
    return later


def assert_block_alerts(run: Run) -> None:
    lines = alerts(run)
    for venue in run.meetings:
        named = [x for x in lines if x.split()[1:2] == [venue]]
        assert any(BLOCK_ALERT.search(x) for x in named), (
            f"{venue}: no ALERT names the block: {lines}"
        )


@pytest.mark.parametrize(
    ("status", "changes"),
    [(403, {}), (429, {"cooldown_min": 10, "resume_requests_per_second": 1.2})],
    ids=["403-as-shipped", "429-other-cooldown-and-rate"],
)
def test_a_first_block_pauses_then_the_day_goes_on_at_the_resume_rate_and_alerts(
    tmp_path, status, changes
):
    cfg = jra_schedule()
    cfg["on_block"] = {**cfg["on_block"], **changes}
    cooldown = cooldown_of(cfg)
    resume_rps = float(cfg["on_block"].get("resume_requests_per_second") or 0.0)
    assert resume_rps > 0, f"on_block {cfg['on_block']}: no resume rate"
    meetings = card(3, *CARDS["358"])
    run = simulate(meetings, cfg, tmp_path / "jra-block.txt", blocks=[(jst(8, 0), status)])

    (block,) = blocked_calls(run)
    assert [c for c in run.net.calls if c.refused] == [], "a call to the site that was blocked"
    later = after_cooldown(run, block, cooldown)
    made = [(t, r) for t, r in run.net.made if t >= block.t]
    assert len(made) == 1 and abs(made[0][1] - resume_rps) < EPS, f"sites made after it: {made}"
    slow = [c for c in later if abs(c.rps - resume_rps) > EPS]
    assert slow == [], f"after the block, calls not at {resume_rps} req/s: {slow[:3]}"

    # the day goes on, every venue to its end, with the full schedule: the hot grain still runs
    assert run.summary.stopped is None, run.summary.stopped
    assert_day_ended(run)
    hot = cfg["hot_races"]
    thin = []
    for venue, races in meetings.items():
        for n, post in races.items():
            lo = post - timedelta(minutes=hot["within_min"])
            if lo < later[0].t + timedelta(seconds=300):
                continue
            hi = post + timedelta(minutes=hot["after_post_min"])
            got = [t for t in captured_at(run, venue, n) if lo <= t <= hi]
            if len(got) < 10:
                thin.append(f"{venue} {n}: {len(got)} captures while hot")
    assert thin == [], f"the hot grain stopped after the block: {thin[:4]}"
    assert_block_alerts(run)


def test_a_second_block_drops_to_the_floor_only_and_the_day_still_goes_on(tmp_path):
    cfg = jra_schedule()
    cooldown = cooldown_of(cfg)
    resume_rps = float(cfg["on_block"].get("resume_requests_per_second") or 0.0)
    floor_s = floor_grain(cfg)
    meetings = card(3, *CARDS["358"])
    blocks = [(jst(8, 0), 403), (jst(10, 30), 429)]
    run = simulate(meetings, cfg, tmp_path / "jra-block.txt", blocks=blocks)

    first, second = blocked_calls(run)
    assert [c for c in run.net.calls if c.refused] == [], "a call to a site that was blocked"
    after_cooldown(run, first, cooldown)
    later = after_cooldown(run, second, cooldown)
    made = [r for t, r in run.net.made if t >= first.t]
    assert len(made) == 2 and made[0] == pytest.approx(resume_rps), f"sites made: {made}"
    assert made[1] <= resume_rps + EPS, f"the floor-only step at {made[1]} req/s"
    fast = [c for c in later if c.rps > resume_rps + EPS]
    assert fast == [], f"after the second block, calls over {resume_rps} req/s: {fast[:3]}"
    assert run.summary.stopped is None, run.summary.stopped
    assert_day_ended(run)

    # the floor only: every race at least every floor_s up to its post, none at the hot grain
    resumed = later[0].t
    span = timedelta(seconds=floor_s)
    over, crowded = [], []
    for venue, races in meetings.items():
        for n, post in races.items():
            times = captured_at(run, venue, n, since=resumed)
            for i, t in enumerate(times):
                inside = [u for u in times[i:] if u <= t + span]
                if len(inside) > 2:
                    crowded.append(f"{venue} {n}: {len(inside)} from {hhmm(t)}")
                    break
            if post > resumed:
                gap, at = longest_gap(times, resumed, post)
                if gap > floor_s + EPS:
                    over.append(f"{venue} {n} (post {hhmm(post)}): {gap:.1f} s from {hhmm(at)}")
    assert crowded == [], f"captured at the hot grain after the second block: {crowded[:4]}"
    assert over == [], f"the floor missed after the second block: {over[:4]}"
    assert_block_alerts(run)


def test_a_third_block_ends_the_day_as_before(tmp_path):
    cfg = jra_schedule()
    cooldown_of(cfg)
    meetings = card(3, *CARDS["358"])
    blocks = [(jst(8, 0), 403), (jst(10, 30), 429), (jst(12, 0), 403)]
    marker = tmp_path / "jra-block.txt"
    run = simulate(meetings, cfg, marker, blocks=blocks)

    third = blocked_calls(run)[-1]
    assert len(blocked_calls(run)) == 3, blocked_calls(run)
    assert [c for c in run.net.calls if c.t > third.t] == [], "a call after the third block"
    assert run.summary.stopped == "circuit_open", run.summary.stopped
    stops = {v: vs.stopped for v, vs in run.summary.venues.items()}
    assert set(stops.values()) == {"circuit_open"}, stops  # no venue had ended by 12:00
    text = marker.read_text(encoding="utf-8")
    assert HOST in text and DAY.isoformat() in text, text
