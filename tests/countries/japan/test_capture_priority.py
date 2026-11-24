"""Tier 2 (枠連, 馬単, 3連複, 3連単) captured at the D2 grid points, below tier 1 in the request
budget (ticket #130).

Contract pinned here (the coder follows it), on top of test_schedule.py's:

Config
- logger.yaml gains `tier2`: `pools` (the jra.yaml codes BRACKET_QUINELLA, FORECAST, TRIO,
  TIERCE), `grid_min` (minutes before post; the shipped value is decision.yaml's D2 grid) and
  `requests` ({navigation, per_race}: what a tier-2 capture costs, as `requests` for tier 1).
  `pools` (tier 1) is unchanged. A config without `tier2` logs tier 1 only, exactly as before:
  the site's `capture_tier2` is never called and every VenueSummary.tier2_skipped is None.

run_day with `tier2`
- The site gains `capture_tier2(venue, races) -> MeetingCapture` (races: a non-empty list of
  race numbers, no repeats); a successful one is written with store.write, as tier 1 is.
- Grid point g of race r (post p) falls due at p - g minutes; its capture belongs to the interval
  [p - g, p - g') where g' is the next smaller grid value (the post itself after the last). A
  capture of r in that interval is r's capture for g; no race is captured by tier 2 outside its
  grid intervals, nor twice in one.
- Each round plans tier 1 exactly as without tier 2 (same captures, same instants, same
  DaySummary.deferred), runs them first, and gives tier 2 only what is left of the round's budget
  (every_s x rps): tier 1 + tier 2 requests of a round never exceed it (the single hot job is
  tier 1's exception alone). A tier-2 capture whose grid point is due is planned whenever what is
  left covers it (navigation + per_race, or per_race at a venue already in the round's tier 2).
- VenueSummary.tier2_skipped: the venue's grid points (race x grid value) whose capture never
  ran. Captured + skipped = races x grid values.
- A tier-2 capture that fails (FetchError, WrongMeeting, ValueError, or any parser error such as
  KeyError) changes nothing for tier 1 and does not end the venue.
- health_lines: a venue line carries `tier2_skipped=N` when its VenueSummary.tier2_skipped is not
  None; when it is None the line is the #25 line, unchanged.

Not pinned: whether a grid point that did not fit is retried later in its interval or dropped at
once, how tier-2 races are grouped into captures, tier-2 rows' pools in the fake, WIN5 capture
scheduling, the token's place in the line. Everything is synthetic: no HTTP, no real sleep.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from horsetrack.core.config import load_country_yaml
from horsetrack.core.ingest.fetcher import FetchError
from horsetrack.countries.japan.jra.capture import ODDS_POOLS, MeetingCapture, WrongMeeting
from horsetrack.countries.japan.logger.health import health_lines
from horsetrack.countries.japan.logger.schedule import DaySummary, VenueSummary, run_day

UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
SAT = date(2026, 11, 28)
HOST = "www.jra.go.jp"
EPS = 1e-6
TIER2_POOLS = ["BRACKET_QUINELLA", "FORECAST", "TRIO", "TIERCE"]
D2_GRID = [60, 30, 16, 5, 2, 1]

BASE = {
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


def config(grid=D2_GRID, nav: int = 2, per: int = 4) -> dict:
    cfg = copy.deepcopy(BASE)
    cfg["tier2"] = {
        "pools": list(TIER2_POOLS),
        "grid_min": list(grid),
        "requests": {"navigation": nav, "per_race": per},
    }
    return cfg


def without_tier2(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    out.pop("tier2", None)
    return out


def jst(h: int, m: int = 0, s: int = 0, day: date = SAT) -> datetime:
    return datetime(day.year, day.month, day.day, h, m, s, tzinfo=JST).astimezone(UTC)


def hhmm(t: datetime) -> str:
    return t.astimezone(JST).strftime("%H:%M:%S JST")


def posts_from(first: datetime, races: int, every_min: int = 30) -> dict[int, datetime]:
    return {n: first + timedelta(minutes=every_min * (n - 1)) for n in range(1, races + 1)}


# ---------------------------------------------------------------- fakes


class Runaway(BaseException):
    """run_day kept going where a day cannot."""


class Clock:
    """The fake clock: only sleep() moves it."""

    def __init__(self, start: datetime):
        self.t = start
        self.limit = start + timedelta(hours=24)
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
    kind: str  # discover | post_times | capture | tier2
    venue: str | None
    races: tuple[int, ...] | None = None
    requests: int = 0
    error: str | None = None


class FakeSite:
    """One host as the scheduler sees it, with the tier-2 capture. A capture costs navigation +
    per_race x races of its tier. `fail_tier2(n)` (n: the 1-based count of tier-2 calls) returns
    an exception for that call to raise, or None."""

    def __init__(
        self,
        clock: Clock,
        meetings: dict[str, dict[int, datetime]],
        *,
        rps: float,
        tier2_requests: dict[str, int],
        fail_tier2: Callable[[int], BaseException | None] | None = None,
    ):
        self.clock, self.meetings = clock, meetings
        self.host, self.requests_per_second = HOST, rps
        self.nav, self.per = BASE["requests"]["navigation"], BASE["requests"]["per_race"]
        self.nav2, self.per2 = tier2_requests["navigation"], tier2_requests["per_race"]
        self.fail_tier2 = fail_tier2 or (lambda n: None)
        self.calls: list[Call] = []
        self.defects: list[str] = []
        self.tier2_count = 0

    def _check(self, call: Call, races) -> None:
        if races is None:
            return
        every = set(self.meetings[call.venue])
        if not races:
            self.defects.append(f"{hhmm(call.t)} {call.kind} of {call.venue} asked for no race")
        if len(set(races)) != len(races):
            self.defects.append(f"{hhmm(call.t)} {call.kind} of {call.venue} repeats a race")
        if set(races) - every:
            self.defects.append(f"{hhmm(call.t)} {call.kind} of {call.venue} asks for {races}")

    def discover(self) -> list[str]:
        self.calls.append(Call(self.clock.t, "discover", None, requests=1))
        return list(self.meetings)

    def post_times(self, venue: str) -> dict[int, datetime]:
        self.calls.append(Call(self.clock.t, "post_times", venue, requests=self.nav))
        return dict(self.meetings[venue])

    def _rows(self, venue: str, races, pools, t: datetime) -> list[dict]:
        return [
            {"race_id": f"jra:{SAT.isoformat()}:{venue}:{n:02d}", "pool": p, "snapshot_utc": t}
            for n in races
            for p in pools
        ]

    def capture(self, venue: str, races) -> MeetingCapture:
        asked = None if races is None else tuple(races)
        call = Call(self.clock.t, "capture", venue, asked)
        self.calls.append(call)
        self._check(call, asked)
        wanted = sorted(self.meetings[venue]) if asked is None else sorted(asked)
        call.requests = self.nav + self.per * len(wanted)
        rows = self._rows(venue, wanted, ODDS_POOLS, call.t)
        return MeetingCapture(rows, call.t, call.requests)

    def capture_tier2(self, venue: str, races) -> MeetingCapture:
        asked = tuple(races) if races is not None else None
        call = Call(self.clock.t, "tier2", venue, asked)
        self.calls.append(call)
        if asked is None:
            self.defects.append(f"{hhmm(call.t)} tier-2 capture of {venue} without races")
            asked = tuple(sorted(self.meetings[venue]))
        self._check(call, asked)
        self.tier2_count += 1
        exc = self.fail_tier2(self.tier2_count)
        call.requests = self.nav2 + self.per2 * len(asked)
        if exc is not None:
            call.error = type(exc).__name__
            raise exc
        rows = self._rows(venue, asked, TIER2_POOLS, call.t)
        return MeetingCapture(rows, call.t, call.requests)


class FakeStore:
    def __init__(self):
        self.writes: list[tuple[str, MeetingCapture]] = []

    def write(self, venue: str, capture: MeetingCapture) -> int:
        self.writes.append((venue, capture))
        return len(capture.rows)

    def write_post_times(self, venue: str, times: dict[int, datetime]) -> None:
        pass


def run(meetings, cfg: dict, rps: float, marker: Path, tier2_requests=None, fail_tier2=None):
    clock = Clock(jst(6, 30))
    t2 = tier2_requests or cfg.get("tier2", config())["requests"]
    site = FakeSite(clock, meetings, rps=rps, tier2_requests=t2, fail_tier2=fail_tier2)
    store = FakeStore()
    summary = run_day(
        SAT,
        site,
        store,
        config=cfg,
        now=clock.now,
        sleep=clock.sleep,
        block_marker=marker,
        venues=list(meetings),
    )
    assert site.defects == [], site.defects
    return summary, site, store


# ---------------------------------------------------------------- reading what the fake saw


def tier1(site: FakeSite) -> list[tuple]:
    return [(c.t, c.venue, c.races) for c in site.calls if c.kind == "capture"]


def tier2_ok(site: FakeSite, venue: str | None = None) -> list[Call]:
    return [
        c
        for c in site.calls
        if c.kind == "tier2" and c.error is None and (venue is None or c.venue == venue)
    ]


def every_s_at(t: datetime, cfg: dict) -> float | None:
    current = None
    for w in sorted(cfg["windows"], key=lambda w: w["from"]):
        h, m = (int(x) for x in w["from"].split(":"))
        if jst(h, m) <= t:
            current = w["every_s"]
    return current


def round_starts(cfg: dict, until: datetime) -> list[datetime]:
    """Every round's start from the first window: every_s apart, restarting at each window."""
    starts_of = sorted(jst(*(int(x) for x in w["from"].split(":"))) for w in cfg["windows"])
    out, t = [], starts_of[0]
    while t < until:
        out.append(t)
        nxt = t + timedelta(seconds=every_s_at(t, cfg))
        later = [s for s in starts_of if t < s < nxt]
        t = later[0] if later else nxt
    return out


def grid_intervals(post: datetime, grid: list[int]) -> list[tuple[int, datetime, datetime]]:
    """(g, due, end) per grid value: [post - g, post - next smaller g), the last one to the post."""
    values = sorted(grid, reverse=True)
    out = []
    for i, g in enumerate(values):
        nxt = values[i + 1] if i + 1 < len(values) else 0
        out.append((g, post - timedelta(minutes=g), post - timedelta(minutes=nxt)))
    return out


def grid_captures(site: FakeSite, venue: str, grid: list[int]) -> dict[tuple[int, int], list]:
    """(race, g) -> instants of the tier-2 captures of that race inside g's interval. Asserts no
    tier-2 capture of a race falls outside every grid interval."""
    posts = site.meetings[venue]
    out: dict[tuple[int, int], list[datetime]] = defaultdict(list)
    for call in tier2_ok(site, venue):
        for race in call.races:
            spans = [(g, a, b) for g, a, b in grid_intervals(posts[race], grid) if a <= call.t < b]
            assert spans, f"{venue} race {race}: tier-2 capture at {hhmm(call.t)}, off the grid"
            out[(race, spans[0][0])].append(call.t)
    return out


def spend_by_round(site: FakeSite) -> dict[datetime, dict[str, int]]:
    out: dict[datetime, dict[str, int]] = defaultdict(lambda: {"capture": 0, "tier2": 0})
    for c in site.calls:
        if c.kind in ("capture", "tier2"):
            out[c.t][c.kind] += c.requests
    return out


def assert_tier1_as_without_tier2(summary, site, base_summary, base_site, where: str) -> None:
    got, want = tier1(site), tier1(base_site)
    assert len(got) == len(want), f"{where}: {len(got)} tier-1 captures, {len(want)} without tier 2"
    for g, w in zip(got, want, strict=True):
        assert g == w, f"{where}: tier-1 capture {g[1]} {g[2]} at {hhmm(g[0])}, not at {hhmm(w[0])}"
    assert summary.deferred == base_summary.deferred, where
    for venue, vs in summary.venues.items():
        assert vs.stopped == base_summary.venues[venue].stopped, f"{where}: {venue} stopped"


def assert_rounds_fit(site: FakeSite, cfg: dict, rps: float, where: str) -> None:
    nav, per = cfg["requests"]["navigation"], cfg["requests"]["per_race"]
    for t, spent in spend_by_round(site).items():
        budget = every_s_at(t, cfg) * rps
        total = spent["capture"] + spent["tier2"]
        if spent["capture"] > budget + EPS:  # tier 1's single hot job: tier 2 gets nothing
            assert spent["capture"] == nav + per, f"{where}: round at {hhmm(t)}"
            assert spent["tier2"] == 0, f"{where}: tier 2 over the budget at {hhmm(t)}"
        else:
            assert total <= budget + EPS, f"{where}: {total} requests at {hhmm(t)}, budget {budget}"


def assert_tier1_first(site: FakeSite, where: str) -> None:
    seen_tier2: set[datetime] = set()
    for c in site.calls:
        if c.kind == "tier2":
            seen_tier2.add(c.t)
        elif c.kind == "capture":
            assert c.t not in seen_tier2, f"{where}: tier 1 after tier 2 in the round {hhmm(c.t)}"


def skipped_in_health(summary: DaySummary, venue: str) -> int:
    lines = health_lines(summary, {}, {})
    line = next(x for x in lines if not x.startswith("ALERT") and f"{venue} snapshots=" in x)
    m = re.search(r"\btier2_skipped=(\d+)\b", line)
    assert m, f"no tier2_skipped in the health line: {line!r}"
    return int(m.group(1))


# ---------------------------------------------------------------- the shipped config


def test_logger_yaml_ships_tier2_at_the_d2_grid_after_tier1():
    shipped = load_country_yaml("japan", "logger")
    assert shipped["pools"] == list(ODDS_POOLS), "tier 1's pools are unchanged"
    tier2 = shipped["tier2"]
    assert sorted(tier2["pools"]) == sorted(TIER2_POOLS)
    assert tier2["grid_min"] == load_country_yaml("japan", "decision")["grid_min"] == D2_GRID
    # one capture: the odds menu and race list, then one page per pool of each race
    assert tier2["requests"] == {"navigation": 2, "per_race": len(TIER2_POOLS)}


# ---------------------------------------------------------------- an ample budget


@pytest.mark.parametrize(
    ("grid", "nav", "per"), [(D2_GRID, 2, 4), ([45, 10, 3], 1, 5)], ids=["d2-grid", "other-grid"]
)
def test_with_budget_to_spare_tier2_is_captured_once_per_grid_point(tmp_path, grid, nav, per):
    meetings = {
        "TOK": posts_from(jst(10, 40), 8),
        "KYO": posts_from(jst(10, 55), 8, every_min=35),
    }
    cfg, rps = config(grid, nav, per), 64.0
    where = f"grid {grid}, tier 2 {nav}+{per}/race"
    summary, site, store = run(meetings, cfg, rps, tmp_path / "a")
    base_summary, base_site, _ = run(meetings, without_tier2(cfg), rps, tmp_path / "b")

    assert_tier1_as_without_tier2(summary, site, base_summary, base_site, where)
    assert_rounds_fit(site, cfg, rps, where)
    assert_tier1_first(site, where)
    for venue, posts in meetings.items():
        captured = grid_captures(site, venue, grid)
        for race, post in posts.items():
            for g, due, _ in grid_intervals(post, grid):
                times = captured.get((race, g), [])
                assert len(times) == 1, f"{where}: {venue} {race} T-{g}: {len(times)} captures"
                late = (times[0] - due).total_seconds()
                assert 0 <= late <= 20 + EPS, f"{where}: {venue} {race} T-{g} {late} s late"
        assert summary.venues[venue].tier2_skipped == 0, f"{where}: {venue}"
        assert skipped_in_health(summary, venue) == 0, f"{where}: {venue}"

    # every capture is written, tier 2's with its own pools
    ok = [c for c in site.calls if c.kind == "capture"] + tier2_ok(site)
    assert len(store.writes) == len(ok)
    written = {row["pool"] for _, cap in store.writes for row in cap.rows}
    assert written == set(ODDS_POOLS) | set(TIER2_POOLS)


# ---------------------------------------------------------------- a short budget

# (rps, tier-2 navigation, per_race, grid, expectation). A hot race costs 2 + 3 = 5; budgets are
# every_s (20) x rps: 5 leaves nothing a tier-2 capture of 6 fits in; 10 leaves at most 5 beside
# a hot race, so every grid point inside a race's hot window (g <= within_min) is skipped.
SHORT = [
    (0.25, 2, 4, D2_GRID, "all"),
    (0.25, 2, 4, [45, 10, 3], "all"),
    (0.5, 2, 4, D2_GRID, "hot"),
    (0.5, 1, 5, [45, 10, 3], "hot"),
    (0.75, 2, 4, D2_GRID, None),
    (0.25, 1, 3, D2_GRID, None),
]


@pytest.mark.parametrize(("rps", "nav", "per", "grid", "expect"), SHORT)
def test_under_a_short_budget_tier1_keeps_its_schedule_and_tier2_skips_are_counted(
    tmp_path, rps, nav, per, grid, expect
):
    venue = "KYO"
    meetings = {venue: posts_from(jst(10, 40), 10)}
    cfg = config(grid, nav, per)
    where = f"rps {rps}, grid {grid}, tier 2 {nav}+{per}/race"
    summary, site, _ = run(meetings, cfg, rps, tmp_path / "a")
    base_summary, base_site, _ = run(meetings, without_tier2(cfg), rps, tmp_path / "b")

    assert_tier1_as_without_tier2(summary, site, base_summary, base_site, where)
    assert_rounds_fit(site, cfg, rps, where)
    assert_tier1_first(site, where)

    posts = meetings[venue]
    captured = grid_captures(site, venue, grid)
    assert all(len(t) == 1 for t in captured.values()), f"{where}: a grid point captured twice"
    due = len(posts) * len(grid)
    skipped = summary.venues[venue].tier2_skipped
    assert skipped is not None, where
    assert len(captured) + skipped == due, f"{where}: {len(captured)} + {skipped} != {due}"
    assert skipped_in_health(summary, venue) == skipped, where
    if expect == "all":
        assert tier2_ok(site) == [] and skipped == due, where
    elif expect == "hot":
        inside_hot = len(posts) * len([g for g in grid if g <= BASE["hot_races"]["within_min"]])
        assert skipped >= inside_hot, f"{where}: {skipped} skipped, {inside_hot} inside T-20"

    # a grid point is skipped only when its first round had no room left for it
    spent = spend_by_round(site)
    starts = round_starts(cfg, jst(17, 0))
    tier2_rounds = {c.t for c in tier2_ok(site, venue)}  # the venue already in tier 2 there
    for race, post in posts.items():
        for g, due_at, _ in grid_intervals(post, grid):
            if (race, g) in captured:
                continue
            t0 = next(t for t in starts if t >= due_at)
            left = every_s_at(t0, cfg) * rps - spent[t0]["capture"] - spent[t0]["tier2"]
            marginal = per if t0 in tier2_rounds else nav + per
            assert left < marginal - EPS, (
                f"{where}: race {race} T-{g} skipped with {left} of the budget left at {hhmm(t0)}"
            )


# ---------------------------------------------------------------- tier 1 alone, and failures


def test_without_tier2_in_the_config_nothing_changes(tmp_path):
    meetings = {"TOK": posts_from(jst(10, 40), 6), "KYO": posts_from(jst(11, 5), 6)}
    summary, site, store = run(meetings, without_tier2(config()), 1.0, tmp_path / "a")

    assert [c for c in site.calls if c.kind == "tier2"] == []
    assert tier1(site), "tier 1 captured nothing"
    for venue, vs in summary.venues.items():
        assert vs.tier2_skipped is None, venue
    for line in health_lines(summary, {}, {}):
        assert "tier2" not in line, line
    assert {row["pool"] for _, cap in store.writes for row in cap.rows} == set(ODDS_POOLS)


def test_the_health_line_counts_tier2_skips():
    cases = [(0, "ended"), (7, "ended"), (60, "circuit_open")]
    for skipped, stopped in cases:
        summary = DaySummary()
        summary.venues["NAK"] = VenueSummary(
            snapshots=12, rows=480, stopped=stopped, tier2_skipped=skipped
        )
        summary.venues["HSN"] = VenueSummary(snapshots=9, rows=300, stopped="ended")
        lines = health_lines(summary, {}, {})
        nak = [x for x in lines if not x.startswith("ALERT") and "NAK snapshots=12" in x]
        hsn = [x for x in lines if not x.startswith("ALERT") and "HSN snapshots=9" in x]
        assert len(nak) == 1 and len(hsn) == 1, lines
        assert re.findall(r"\btier2_skipped=(\d+)\b", nak[0]) == [str(skipped)], nak[0]
        assert "tier2" not in hsn[0], "a venue without tier 2 keeps the #25 line"


@pytest.mark.parametrize(
    "make",
    [
        lambda: FetchError("timeout"),
        lambda: WrongMeeting("odds page for another race"),
        lambda: ValueError("not a JRADB page"),
        lambda: KeyError("caption"),
    ],
    ids=["fetch-error", "wrong-meeting", "value-error", "parser-key-error"],
)
def test_a_failing_tier2_capture_never_touches_tier1(tmp_path, make):
    meetings = {"TOK": posts_from(jst(10, 40), 6), "FKS": posts_from(jst(10, 50), 6)}
    cfg, rps = config(), 64.0
    # every other tier-2 call fails
    summary, site, _ = run(
        meetings, cfg, rps, tmp_path / "a", fail_tier2=lambda n: make() if n % 2 else None
    )
    base_summary, base_site, _ = run(meetings, without_tier2(cfg), rps, tmp_path / "b")

    assert [c for c in site.calls if c.kind == "tier2" and c.error], "no tier-2 call failed"
    assert_tier1_as_without_tier2(summary, site, base_summary, base_site, "failing tier 2")
    assert summary.stopped is None
    assert all(vs.stopped == "ended" for vs in summary.venues.values()), summary.venues
