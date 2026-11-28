"""The day-before window (ticket #358): the JRA odds logger matches JV-Link's time-series grain. It
starts when the day-before sales open the evening before the race date, read off the オッズ menu,
captures every race at least every 5 minutes through the night until each race's T-15, and from
T-15 (hot from T-20) keeps the race date's fine grain (20 s) as before.

Contract pinned here (the coder follows it), on top of test_schedule.py's for run_day:

Config
- A schedule (the mapping run_day takes) may have a `day_before` block:
  `{probe_from: "HH:MM", probe_until: "HH:MM", probe_every_s: N, every_s: N}`, probe_from and
  probe_until being Asia/Tokyo wall-clock times on the day before the race date. Without the block
  run_day is unchanged. logger.yaml's JRA schedule (its top level) has it; NAR's schedule
  (`source_config(config, "nar")`) never inherits it, as with tier2 and results.
- logger.yaml: tier 1 `pools` are WIN, PLACE, QUINELLA, QUINELLA_PLACE and BRACKET_QUINELLA (its
  own page: `requests.per_race` 4); tier 2 keeps FORECAST, TRIO, TIERCE (`tier2.requests.per_race`
  3). The windows (07:00 every 300 s, 09:30 every 20 s), the stop and the hot races are unchanged,
  but for `hot_races.floor_s` 300: the owner's 5-minute floor (#387, option (a), 2026-11-28),
  which test_five_minute_floor.py pins on the race date (a stale race outranks the hot races but
  the nearest; those keep inside the health line's MAX_GAP_S). This file's CONFIG has no floor_s.

Site
- With `day_before`, a site also has `sales_open() -> bool`: whether the オッズ menu lists the race
  date's meetings (the day-before sales are open). JRA: one request, the オッズ menu. It may raise
  what any site call raises.

run_day with `day_before` (race date D; W = the earliest `windows` start on D)
- Probing: nothing is sent before probe_from on D-1 (a run started earlier sleeps until then).
  From max(start, probe_from), sales_open() is called at once and then every probe_every_s (start
  to start) while before probe_until, until one returns True: the sales open.
  `DaySummary.sales_open_utc` is that probe's instant (None when no probe of the run returned
  True). No probe after it.
- Ruling: a run started at or after probe_until on D-1 and before W calls sales_open() once, at
  its start (so a restart in the night resumes the window); True opens the window there, False
  leaves it shut. A run started at or after W never probes.
- No open by probe_until (and no late start that finds it): nothing more is sent until W, and the
  race date runs as it does without `day_before` (discovery at W, ...).
- Before the open the only calls are probes: no discovery, post-time read or capture (Ruling:
  the race date's discovery comes with the open, or at W as before when there is none).
- From the open: every race of every venue is captured at least every `day_before.every_s` until
  W, each round within its budget (day_before.every_s x rps), and at least every 300 s on from W
  to the race's T-15 (the 07:00 window's grain; across the switch at W too). From W the race
  date's windows and hot races run as before, so from T-20 (hot) to post + after_post_min a race
  is captured every round of the window in force (20 s from 09:30): the fine grain from T-15.
- Blocks: a CircuitOpen from a probe is a block as from any site call: the block marker gets a
  line `<host> <D> ...` (the race date, not the day of the probe), nothing more is sent, every
  venue stops 'circuit_open', and a run that finds the line (that evening, or on D) sends nothing.
  A probe raising FetchError or ValueError is retried at the next probe and counts as no venue's
  failure.

Not pinned: logging, post-time reads' charging, the order of calls inside a round, which venue's
cycle gets spare budget, how the rounds are aligned. Everything is synthetic: no HTTP, no real
sleep, no scraped page.
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
from horsetrack.countries.japan.jra.capture import MeetingCapture, NotListed

# isort: split
from horsetrack.countries.japan.logger.schedule import run_day, source_config

UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
SAT, SUN = date(2026, 11, 28), date(2026, 11, 29)
DAY, EVE = SUN, SAT  # the race date and the evening before it
HOST = "www.jra.go.jp"
EPS = 1e-6
POOLS = ["WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE", "BRACKET_QUINELLA"]
GRAIN_S = 300  # the owner's rule: every race at least every 5 minutes until its T-15

CONFIG = {
    "pools": list(POOLS),
    "min_poll_interval_s": 20,
    "windows": [{"from": "07:00", "every_s": 300}, {"from": "09:30", "every_s": 20}],
    "stop": "16:45",
    "after_last_race_min": 10,
    "hot_races": {"within_min": 20, "after_post_min": 5, "all_every_s": 300},
    "requests": {"navigation": 2, "per_race": 4},
    "max_wrong_meeting": 3,
    "on_block": {"cooldown_min": None},
    "day_before": {"probe_from": "17:00", "probe_until": "21:00", "probe_every_s": 60,
                   "every_s": 300},
}  # fmt: skip


def config(**changes) -> dict:
    """CONFIG with some values changed (a dict value is merged into the default)."""
    cfg = copy.deepcopy(CONFIG)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **copy.deepcopy(value)}
        else:
            cfg[key] = copy.deepcopy(value)
    return cfg


# ---------------------------------------------------------------- time


def jst(h: int, m: int = 0, s: int = 0, day: date = DAY) -> datetime:
    """The aware UTC instant of a Tokyo wall-clock time on `day`."""
    return datetime(day.year, day.month, day.day, h, m, s, tzinfo=JST).astimezone(UTC)


def wall(hhmm: str, day: date) -> datetime:
    h, m = (int(part) for part in str(hhmm).split(":"))
    return jst(h, m, day=day)


def first_window(cfg: dict, day: date = DAY) -> datetime:
    return min(wall(w["from"], day) for w in cfg["windows"])


def secs(delta: timedelta) -> float:
    return delta.total_seconds()


def hhmm(t: datetime) -> str:
    return t.astimezone(JST).strftime("%m-%d %H:%M:%S JST")


def probe_grid(start: datetime, cfg: dict) -> list[datetime]:
    """The probes a run started at `start` makes while the menu does not list the race date: at
    max(start, probe_from), then every probe_every_s while before probe_until; a run started at
    or after probe_until (and before W) probes once. A probe exactly at probe_until may be made
    or not: it is the last of the list when it falls there."""
    db = cfg["day_before"]
    t0 = max(start, wall(db["probe_from"], EVE))
    until = wall(db["probe_until"], EVE)
    if t0 >= first_window(cfg):
        return []
    if t0 >= until:
        return [t0]
    out, t = [], t0
    while t <= until:
        out.append(t)
        t += timedelta(seconds=db["probe_every_s"])
    return out


def expected_open(start: datetime, cfg: dict, listed_from: datetime | None) -> datetime | None:
    """The instant of the first probe that finds the race date on the menu, from first
    principles; None when no probe does."""
    if listed_from is None:
        return None
    grid = probe_grid(start, cfg)
    until = wall(cfg["day_before"]["probe_until"], EVE)
    for t in grid:
        if t >= listed_from and (t < until or len(grid) == 1):
            return t
    return None


def posts(first: tuple[int, int], races: int, every_min: int = 30) -> dict[int, datetime]:
    return {n: jst(*first) + timedelta(minutes=every_min * (n - 1)) for n in range(1, races + 1)}


# ---------------------------------------------------------------- fakes


class Runaway(BaseException):
    """run_day kept going where a day cannot (a BaseException: no `except Exception` hides it)."""


class Clock:
    """The fake clock: only sleep() moves it."""

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
            raise Runaway(f"run_day still sleeping at {self.t.isoformat()}: it never returned")


@dataclass
class Call:
    t: datetime
    kind: str  # sales_open | discover | post_times | capture
    venue: str | None
    races: tuple[int, ...] | None = None
    error: str | None = None
    requests: int = 0
    answer: bool | None = None  # sales_open's


Rule = Callable[[str, str | None, int, datetime], BaseException | None]


class FakeSite:
    """One host's JRADB for race date DAY as the scheduler sees it, every call recorded at the
    fake clock's time. The オッズ menu lists DAY's meetings from `listed_from` (None: never): before
    that sales_open() says False and a capture is NotListed. sales_open() and discover() cost a
    request, post_times() `nav`, a capture nav + per_race x races. `fail(kind, venue, n, t)`, n
    being the 1-based count of that kind of call for that venue, returns an exception to raise."""

    def __init__(
        self,
        clock: Clock,
        meetings: dict[str, dict[int, datetime]],
        *,
        listed_from: datetime | None,
        rps: float = 1.0,
        nav: int = 2,
        per_race: int = 4,
        fail: Rule | None = None,
    ):
        self.clock, self.meetings, self.listed_from = clock, meetings, listed_from
        self.source, self.host, self.requests_per_second = "jra", HOST, rps
        self.nav, self.per_race = nav, per_race
        self.fail = fail or (lambda kind, venue, n, t: None)
        self.calls: list[Call] = []
        self.requests = 0
        self.defects: list[str] = []
        self._counts: Counter = Counter()

    def _listed(self, t: datetime) -> bool:
        return self.listed_from is not None and t >= self.listed_from

    def _call(self, kind: str, venue: str | None, races=None) -> tuple[Call, BaseException | None]:
        call = Call(self.clock.t, kind, venue, races)
        self.calls.append(call)
        self._counts[(kind, venue)] += 1
        if venue is not None and venue not in self.meetings:
            self.defects.append(f"{kind} for {venue}, which has no meeting on {DAY}")
        return call, self.fail(kind, venue, self._counts[(kind, venue)], call.t)

    def _spend(self, call: Call, requests: int, exc: BaseException | None = None) -> None:
        call.requests = requests
        self.requests += requests
        if exc is not None:
            call.error = type(exc).__name__
            raise exc

    def sales_open(self) -> bool:
        call, exc = self._call("sales_open", None)
        self._spend(call, 1, exc)
        call.answer = self._listed(call.t)
        return call.answer

    def discover(self) -> list[str]:
        call, exc = self._call("discover", None)
        self._spend(call, 1, exc)
        return list(self.meetings)

    def post_times(self, venue: str) -> dict[int, datetime]:
        call, exc = self._call("post_times", venue)
        self._spend(call, self.nav, exc)
        return dict(self.meetings[venue])

    def capture(self, venue: str, races) -> MeetingCapture:
        asked = None if races is None else tuple(races)
        call, exc = self._call("capture", venue, asked)
        every = sorted(self.meetings.get(venue, {}))
        if asked is not None and (not asked or set(asked) - set(every)):
            self.defects.append(f"{hhmm(call.t)} capture of {venue} asks for races {asked}")
        if exc is None and not self._listed(call.t):
            exc = NotListed(f"{DAY} {venue} is not on the odds menu yet")
        if exc is not None:
            self._spend(call, 1, exc)
        wanted = every if asked is None else sorted(set(asked) & set(every))
        spent = self.nav + self.per_race * len(wanted)
        self._spend(call, spent)
        rows = [
            {"race_id": f"jra:{DAY.isoformat()}:{venue}:{n:02d}", "pool": pool,
             "snapshot_utc": call.t}
            for n in wanted
            for pool in POOLS
        ]  # fmt: skip
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


def run(site: FakeSite, clock: Clock, marker: Path, cfg: dict, venues=None):
    store = FakeStore()
    summary = run_day(DAY, site, store, config=cfg, now=clock.now, sleep=clock.sleep,
                      block_marker=marker, venues=venues)  # fmt: skip
    assert site.defects == [], site.defects
    return summary, store


# ---------------------------------------------------------------- reading what the fake saw


def calls_of(site: FakeSite, kind: str) -> list[Call]:
    return [c for c in site.calls if c.kind == kind]


def captured(site: FakeSite, venue: str, race: int) -> list[datetime]:
    """When `race` of `venue` was captured (successful captures that asked for it)."""
    return [
        c.t
        for c in calls_of(site, "capture")
        if c.venue == venue and c.error is None and (c.races is None or race in c.races)
    ]


def cost(call: Call, site: FakeSite, cfg: dict) -> int:
    races = sorted(site.meetings[call.venue]) if call.races is None else call.races
    return cfg["requests"]["navigation"] + cfg["requests"]["per_race"] * len(races)


def assert_probes(site: FakeSite, start: datetime, cfg: dict, opened: datetime | None) -> None:
    """The probes are the grid's, up to the open (all of the grid's when it never opens)."""
    got = [c.t for c in calls_of(site, "sales_open")]
    grid = probe_grid(start, cfg)
    want = [t for t in grid if opened is None or t <= opened]
    until = wall(cfg["day_before"]["probe_until"], EVE)
    if opened is None and len(grid) > 1 and grid[-1] == until and len(got) == len(want) - 1:
        want = want[:-1]  # the probe at probe_until itself: either way
    where = f"probes from {hhmm(start)}"
    assert len(got) == len(want), f"{where}: {len(got)} probes, expected {len(want)}"
    assert all(abs(secs(g - w)) <= EPS for g, w in zip(got, want, strict=True)), (
        f"{where}: first off the grid: "
        + next(f"{hhmm(g)} for {hhmm(w)}" for g, w in zip(got, want, strict=True) if g != w)
    )


def assert_quiet_before(site: FakeSite, opened: datetime) -> None:
    """Before the open: probes only (no capture, discovery or post-time read)."""
    kinds = Counter(c.kind for c in site.calls if c.t < opened)
    assert kinds["capture"] == 0, f"a capture before the open at {hhmm(opened)}"
    assert set(kinds) <= {"sales_open"}, f"before the open at {hhmm(opened)}: {dict(kinds)}"


def past_the_switch(w: datetime) -> Callable[[datetime], datetime]:
    """assert_grain's end: the race's T-15, or GRAIN_S past W (the 07:00 switch) when sooner."""
    return lambda post: min(post - timedelta(minutes=15), w + timedelta(seconds=GRAIN_S))


def assert_grain(site: FakeSite, cfg: dict, opened: datetime, until_t: Callable) -> None:
    """Every race: first capture within day_before.every_s of the open; then no gap over
    day_before.every_s before W, and none over GRAIN_S from the open to `until_t(post)`."""
    every = cfg["day_before"]["every_s"]
    w = first_window(cfg)
    for venue, races in site.meetings.items():
        for n, post in races.items():
            end = until_t(post)
            times = [t for t in captured(site, venue, n) if opened <= t <= end]
            where = f"{venue} race {n} (post {hhmm(post)})"
            assert times, f"{where}: not captured from the open {hhmm(opened)} to {hhmm(end)}"
            assert secs(times[0] - opened) <= every + EPS, f"{where}: first at {hhmm(times[0])}"
            for a, b in zip(times, times[1:], strict=False):
                limit = every if b < w else GRAIN_S
                assert secs(b - a) <= limit + EPS, (
                    f"{where}: {secs(b - a):.0f} s from {hhmm(a)} to {hhmm(b)} (limit {limit} s)"
                )
            assert secs(end - times[-1]) <= GRAIN_S + EPS, f"{where}: last at {hhmm(times[-1])}"


# ---------------------------------------------------------------- logger.yaml


def test_logger_yaml_ships_the_358_rules():
    shipped = copy.deepcopy(load_country_yaml("japan", "logger"))
    pools = shipped["pools"]
    assert len(pools) == len(set(pools)), pools
    assert set(pools) == set(POOLS), f"tier 1 pools {pools}: bracket quinella every round"
    assert shipped["requests"] == {"navigation": 2, "per_race": 4}, shipped["requests"]
    tier2 = shipped["tier2"]
    assert sorted(tier2["pools"]) == ["FORECAST", "TIERCE", "TRIO"], tier2["pools"]
    assert tier2["requests"] == {"navigation": 2, "per_race": 3}, tier2["requests"]
    # unchanged: the race date's windows, hot races and stop; #387 adds the owner's floor
    hot = dict(shipped["hot_races"])
    assert hot.pop("floor_s", None) == GRAIN_S, f"hot_races.floor_s: {shipped['hot_races']}"
    assert hot == CONFIG["hot_races"], f"logger.yaml hot_races: {shipped['hot_races']!r}"
    for key in ("windows", "stop", "after_last_race_min", "min_poll_interval_s"):
        assert shipped[key] == CONFIG[key], f"logger.yaml {key}: {shipped[key]!r}"

    db = shipped.get("day_before")
    assert isinstance(db, dict), "logger.yaml has no day_before block"
    assert set(db) >= {"probe_from", "probe_until", "probe_every_s", "every_s"}, db
    stop, probe_from, until = (wall(x, EVE) for x in (shipped["stop"], db["probe_from"],
                                                       db["probe_until"]))  # fmt: skip
    # the evening before, after the race day's stop, early enough for an 18:30 open
    assert stop < probe_from <= jst(18, 30, day=EVE), db["probe_from"]
    assert jst(19, 30, day=EVE) <= until <= jst(23, 59, day=EVE), db["probe_until"]
    assert 0 < float(db["probe_every_s"]) <= GRAIN_S, db["probe_every_s"]
    assert 0 < float(db["every_s"]) <= GRAIN_S, db["every_s"]
    # a full cycle of a three-venue, 12-race day fits one day-before round at jra.yaml's rate
    http = load_country_yaml("japan", "jra")["http"]
    rps = float(http["hosts"].get(HOST, http["default"])["requests_per_second"])
    cycle = 3 * (shipped["requests"]["navigation"] + shipped["requests"]["per_race"] * 12)
    assert cycle <= float(db["every_s"]) * rps + EPS, (cycle, db["every_s"], rps)


def test_nar_never_inherits_the_day_before_block():
    shipped = copy.deepcopy(load_country_yaml("japan", "logger"))
    with_block = {**shipped, "day_before": copy.deepcopy(CONFIG["day_before"])}
    for cfg in (shipped, with_block):
        nar = source_config(cfg, "nar")
        assert "day_before" not in nar, f"NAR inherited day_before: {nar.get('day_before')}"
        assert source_config(cfg, "jra").get("day_before") == cfg.get("day_before")
    assert source_config(with_block, "jra")["day_before"] == CONFIG["day_before"]


# ---------------------------------------------------------------- the window


TWO_VENUES = {"TOK": posts((10, 0), 3), "KYO": posts((10, 10), 3)}


@pytest.mark.parametrize(
    ("start", "listed", "changes"),
    [
        (jst(16, 0, day=EVE), jst(18, 30, day=EVE), {}),
        (jst(17, 0, day=EVE), jst(18, 31, 40, day=EVE), {}),
        (jst(17, 13, 20, day=EVE), jst(18, 30, day=EVE), {"probe_every_s": 120}),
        (
            jst(16, 0, day=EVE),
            jst(19, 58, 30, day=EVE),
            {"probe_from": "17:30", "probe_until": "20:00", "probe_every_s": 45, "every_s": 240},
        ),
    ],
    ids=["before-the-probe-start", "open-off-the-grid", "mid-span-120-s", "other-span-240-s"],
)
def test_the_window_opens_at_the_first_probe_that_lists_the_race_date(
    tmp_path, start, listed, changes
):
    cfg = config(day_before=changes)
    clock = Clock(start)
    # 0.125 req/s: a 300 s round holds 37.5 requests, a 240 s one 30: both venues' full cycles
    # (2 x (2 + 4 x 3) = 28) fit one round, and no more.
    site = FakeSite(clock, copy.deepcopy(TWO_VENUES), listed_from=listed, rps=0.125)
    summary, store = run(site, clock, tmp_path / "blocked", cfg)

    opened = expected_open(start, cfg, listed)
    assert opened is not None and opened < wall(cfg["day_before"]["probe_until"], EVE)
    assert min(c.t for c in site.calls) >= wall(cfg["day_before"]["probe_from"], EVE), (
        f"a call at {hhmm(min(c.t for c in site.calls))}, before the probe start"
    )
    assert_probes(site, start, cfg, opened)
    assert getattr(summary, "sales_open_utc", "missing") == opened, (
        f"sales_open_utc {getattr(summary, 'sales_open_utc', 'missing')!r}, opened {opened}"
    )
    assert_quiet_before(site, opened)
    assert [c for c in calls_of(site, "sales_open") if c.t > opened] == [], "probed after the open"

    # every race, every venue, from the open through the night and across the 07:00 switch
    w = first_window(cfg)
    assert_grain(site, cfg, opened, past_the_switch(w))
    night = [jst(0, 0), jst(5, 0)]
    for venue, races in site.meetings.items():
        for n in races:
            assert any(night[0] <= t <= night[1] for t in captured(site, venue, n)), (venue, n)

    # each day-before round within its budget (day_before.every_s x rps), through the site only
    budget = cfg["day_before"]["every_s"] * site.requests_per_second
    rounds: dict[datetime, int] = defaultdict(int)
    for c in calls_of(site, "capture"):
        if opened <= c.t < w and c.error is None:
            rounds[c.t] += cost(c, site, cfg)
    assert rounds, "no capture between the open and W"
    over = {hhmm(t): n for t, n in rounds.items() if n > budget + EPS}
    assert over == {}, f"rounds over the {budget} budget: {list(over.items())[:3]}"
    ok = [c for c in calls_of(site, "capture") if c.error is None]
    assert sum(c.requests for c in ok) <= summary.requests <= site.requests
    assert len(store.writes) == len(ok), "a successful capture not written"


@pytest.mark.parametrize(
    ("fine_every", "within", "after"), [(20, 20, 5), (30, 25, 3)], ids=["20-s", "30-s-t25"]
)
def test_from_t15_each_race_has_the_fine_grain_and_later_races_keep_5_minutes(
    tmp_path, fine_every, within, after
):
    cfg = config(
        windows=[{"from": "07:00", "every_s": 300}, {"from": "09:30", "every_s": fine_every}],
        hot_races={"within_min": within, "after_post_min": after},
    )
    meetings = {
        "TOK": {1: jst(10, 0), 2: jst(12, 40), 3: jst(15, 10)},
        "KYO": {1: jst(10, 20), 2: jst(13, 0)},
    }
    clock = Clock(jst(17, 0, day=EVE))
    site = FakeSite(clock, meetings, listed_from=jst(18, 30, day=EVE), rps=2.0)
    summary, _ = run(site, clock, tmp_path / "blocked", cfg)
    opened = jst(18, 30, day=EVE)
    assert getattr(summary, "sales_open_utc", None) == opened

    # until T-15: at least every 5 minutes, overnight and while earlier races are hot
    assert_grain(site, cfg, opened, lambda post: post - timedelta(minutes=15))
    # from T-15, inside the hot window [T-within, post + after]: every round of the fine window
    for venue, races in meetings.items():
        for n, post in races.items():
            lo, hi = post - timedelta(minutes=within), post + timedelta(minutes=after)
            times = [t for t in captured(site, venue, n) if lo <= t <= hi]
            where = f"{venue} race {n} (post {hhmm(post)})"
            assert times, f"{where}: never captured while hot"
            edges = [lo, *times, hi]
            gaps = [secs(b - a) for a, b in zip(edges, edges[1:], strict=False)]
            assert max(gaps) <= fine_every + EPS, f"{where}: a {max(gaps):.0f} s gap while hot"
            t15 = [t for t in times if t >= post - timedelta(minutes=15)]
            assert len(t15) >= (15 + after) * 60 // fine_every - 1, f"{where}: {len(t15)} from T-15"
        ends = max(races.values()) + timedelta(minutes=cfg["after_last_race_min"])
        assert summary.venues[venue].stopped == "ended", (venue, summary.venues[venue].stopped)
        assert [c for c in site.calls if c.venue == venue and c.t >= ends] == [], venue


# ---------------------------------------------------------------- no open, late starts


@pytest.mark.parametrize(
    ("changes", "meetings", "listed"),
    [
        ({}, {}, None),  # no racing on the race date: the menu never lists it
        ({"probe_every_s": 70, "probe_until": "20:30"}, {}, None),
        ({}, {"TOK": posts((10, 0), 3)}, jst(22, 15, day=EVE)),  # listed after probe_until
    ],
    ids=["no-racing", "no-racing-70-s", "listed-after-the-span"],
)
def test_without_an_open_in_the_probe_span_the_race_date_runs_as_before(
    tmp_path, changes, meetings, listed
):
    cfg = config(day_before=changes)
    start = jst(17, 0, day=EVE)
    clock = Clock(start)
    site = FakeSite(clock, meetings, listed_from=listed, rps=1.0)
    summary, _ = run(site, clock, tmp_path / "blocked", cfg)

    assert getattr(summary, "sales_open_utc", "missing") is None
    assert_probes(site, start, cfg, None)
    until, w = wall(cfg["day_before"]["probe_until"], EVE), first_window(cfg)
    between = [c for c in site.calls if until < c.t < w]
    assert between == [], f"{len(between)} calls between probe_until and W: {between[:3]}"
    assert [c for c in site.calls if c.t < w and c.kind == "capture"] == [], "captured before W"
    discovered = calls_of(site, "discover")
    assert len(discovered) == 1 and w <= discovered[0].t < w + timedelta(seconds=300), discovered
    if not meetings:
        assert dict(summary.venues) == {}
        assert site.calls[-1].kind == "discover", "a call after a discovery that found no venue"
    else:
        for venue, races in meetings.items():
            for n in races:
                times = captured(site, venue, n)
                assert times and times[0] < w + timedelta(seconds=300), (venue, n)


@pytest.mark.parametrize(
    ("start", "meetings", "listed", "opens"),
    [
        (jst(22, 30, day=EVE), {"TOK": posts((10, 0), 3)}, jst(18, 30, day=EVE), True),
        (jst(5, 0), {"TOK": posts((10, 0), 3)}, jst(18, 30, day=EVE), True),
        (jst(5, 0), {}, None, False),  # a night start before a day without racing
        (jst(7, 30), {"TOK": posts((10, 0), 3)}, jst(18, 30, day=EVE), False),  # after W
    ],
    ids=["restart-at-22h30", "restart-at-05h00", "no-racing-at-05h00", "after-the-first-window"],
)
def test_a_start_after_the_probe_span_reads_the_menu_once(tmp_path, start, meetings, listed, opens):
    cfg = config()
    clock = Clock(start)
    site = FakeSite(clock, meetings, listed_from=listed, rps=1.0)
    summary, _ = run(site, clock, tmp_path / "blocked", cfg)
    w = first_window(cfg)

    probes = [c.t for c in calls_of(site, "sales_open")]
    if start >= w:
        assert probes == [], f"probed at {[hhmm(t) for t in probes]}, after the first window"
        assert getattr(summary, "sales_open_utc", "missing") is None
        return
    assert probes == [start], f"probes {[hhmm(t) for t in probes]}: one, at the start"
    if opens:
        assert getattr(summary, "sales_open_utc", None) == start
        assert_grain(site, cfg, start, past_the_switch(w))
    else:
        assert getattr(summary, "sales_open_utc", "missing") is None
        rest = [c for c in site.calls if c.kind != "sales_open"]
        assert rest and all(c.t >= w for c in rest), f"a call before W: {rest[:2]}"
        assert [c.kind for c in rest] == ["discover"], rest


# ---------------------------------------------------------------- blocks


def blocked_on(kind: str, n: int, status: int = 403):
    def rule(k, v, count, t):
        if k == kind and count == n:
            return CircuitOpen(f"HTTP {status}", http_status=status, host=HOST)
        return None

    return rule


@pytest.mark.parametrize(
    ("venues", "status"), [(None, 403), (["TOK"], 429)], ids=["discovered", "given-venues"]
)
def test_a_block_on_a_probe_is_a_block_for_the_race_date(tmp_path, venues, status):
    marker = tmp_path / "jra-block.txt"
    start = jst(18, 0, day=EVE)
    clock = Clock(start)
    site = FakeSite(clock, copy.deepcopy(TWO_VENUES), listed_from=jst(18, 30, day=EVE),
                    fail=blocked_on("sales_open", 3, status))  # fmt: skip
    summary, store = run(site, clock, marker, config(), venues=venues)

    probes = calls_of(site, "sales_open")
    assert len(probes) == 3 and probes[-1].error == "CircuitOpen", probes
    assert site.calls[-1] is probes[-1], f"called after the block: {site.calls[-3:]}"
    assert summary.stopped == "circuit_open", summary.stopped
    assert all(v.stopped == "circuit_open" for v in summary.venues.values()), summary.venues
    if venues:
        assert set(summary.venues) == set(venues)
    assert store.writes == []
    assert getattr(summary, "sales_open_utc", "missing") is None
    lines = [line.split()[:2] for line in marker.read_text(encoding="utf-8").splitlines()]
    assert lines == [[HOST, DAY.isoformat()]], f"block lines {lines}: the race date's"

    # restarts that evening and on the race date send nothing
    for again in (jst(19, 30, day=EVE), jst(9, 0)):
        clock = Clock(again)
        site = FakeSite(clock, copy.deepcopy(TWO_VENUES), listed_from=jst(18, 30, day=EVE))
        summary, store = run(site, clock, marker, config(), venues=venues)
        assert site.calls == [], f"restart at {hhmm(again)}: {site.calls[:3]}"
        assert store.writes == [] and summary.requests == 0


def test_a_failed_probe_is_retried_at_the_next_probe(tmp_path):
    errors = {2: FetchError("timeout"), 3: ValueError("menu lists no meeting at all")}

    def fail(kind, venue, n, t):
        return errors.get(n) if kind == "sales_open" else None

    start = jst(18, 27, day=EVE)
    clock = Clock(start)
    site = FakeSite(clock, copy.deepcopy(TWO_VENUES), listed_from=jst(18, 30, day=EVE), fail=fail)
    summary, _ = run(site, clock, tmp_path / "blocked", config())

    probes = calls_of(site, "sales_open")
    assert [c.t for c in probes] == [start + timedelta(minutes=k) for k in range(4)], probes
    assert [c.error for c in probes] == [None, "FetchError", "ValueError", None]
    assert getattr(summary, "sales_open_utc", None) == jst(18, 30, day=EVE)
    assert summary.stopped is None
    for venue, vs in summary.venues.items():
        assert (vs.failures, vs.stopped) == (0, "ended"), (venue, vs)
    assert_grain(site, config(), jst(18, 30, day=EVE), lambda post: first_window(config()))
