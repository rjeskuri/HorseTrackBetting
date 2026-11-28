"""Logging JRA and NAR in one process through the same scheduler (ticket #38): each host with its
own rate budget, and a block on one host stopping that source only.

Contract pinned here (the coder follows it), on top of test_schedule.py's for run_day:

- A site (schedule.Site) also has `source`: its source key, "jra" or "nar". `host` and
  `requests_per_second` stay the site's own: a round's budget for a source is its window's every_s
  x its own site's requests_per_second, never the other source's.
- `schedule.Source(site=..., store=..., config=..., block_marker=..., venues=None)`: one source's
  run of the day, `config` being that source's schedule (the mapping run_day takes).
- `schedule.run_sources(race_date, sources, *, now, sleep) -> {source key: DaySummary}` logs every
  source's day at once: each source runs as run_day would run it alone (its own config, windows,
  stop, budget, venues and block marker), on a thread of its own for the whole day, all of them
  concurrently; `now` and `sleep` are called from that thread. It returns once every source's
  day is over.
- A CircuitOpen from one source's site stops that source only (its venues stopped
  'circuit_open', its DaySummary.stopped 'circuit_open', its block marker written with its host
  and the Tokyo race date); every other source logs its whole day, and its marker is not written.
  A later run that day sends nothing for the blocked source and logs the others.
- Block policy per source (fix round, review of PR #255), from its schedule's `on_block`. This
  file's JRA_CONFIG has `{cooldown_min: null}`: the first block stops it for the day, as above
  (logger.yaml's JRA on_block backs off since #387: test_five_minute_floor.py). NAR's follows the
  owner decision in docs/research/nar-data-access.md: `{cooldown_min: 15,
  resume_requests_per_second: 0.05}` (half nar.yaml's 0.1). On the first block that source alone
  pauses `cooldown_min` minutes (no call to it meanwhile; the other sources keep logging), then
  resumes once on a fresh site (so a fresh Fetcher) at `resume_requests_per_second`, its rounds'
  budgets at that rate; a second block stops it for the day (venues 'circuit_open', its block
  marker has the host and race date) and a later run that day sends it nothing. Whether the first
  block already writes a marker line is not pinned.
  Ruling: the fresh-site seam is `Source.resume`, a callable taking the resume rate
  (`resume(requests_per_second)`, positional) and returning the site the rest of the day uses;
  run_day takes the same as `resume=`. Without one a cooldown resumes on the same site. log-day
  builds it from `make_fetcher(source, requests_per_second=rate)`.
- An error other than CircuitOpen escaping one source's day (a FetchError at discovery, a parser
  bug in a capture) stops that source only: its DaySummary.stopped is 'error', each of its venues
  not ended yet is stopped 'error', no marker is written; every other source logs its whole day.
  run_sources returns every source's summary and does not raise for it.
  Ruling: `run_sources(..., on_error=None)`; `on_error(source_key, exception)` is called once per
  failed source, as soon as its day fails (from its own thread), while the others still log; so
  log-day reports it at once, not after the other sources' days.
- `schedule.source_config(config, source) -> mapping`: one source's schedule out of logger.yaml.
  JRA's is the outline's values as before; NAR's (night racing, docs/research/nar-data-access.md)
  has a window in force from 10:20 JST (the earliest race's hot window, a 10:40 post) to 20:50
  (the last race), a `stop` no earlier than 21:00, `requests` matching what nar.capture sends (1
  navigation request, one per distinct odds page of its `pools` per race) and a round that fits at
  least one race at nar.yaml's rate; and no tier 2 (there is no NAR tier-2 capture).

The fake clock (thread_clock.py) keeps one virtual time per thread: a source on its own thread
lives through its own day. Each fake site records its calls at its thread's time. Everything is
synthetic: no HTTP, no real sleep, no scraped page.
"""

from __future__ import annotations

import copy
import threading
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from horsetrack.core.config import load_country_yaml
from horsetrack.core.ingest.fetcher import CircuitOpen, FetchError
from horsetrack.countries.japan.jra.capture import ODDS_POOLS, MeetingCapture, NotListed

# isort: split
# The names under test get a block of their own: while they are missing, the import error is what
# a red run reports, and the order lints the same before and after they exist.
from horsetrack.countries.japan.logger.schedule import Source, run_sources, source_config

# isort: split
from thread_clock import EPS, Runaway, ThreadClock

UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
SAT = date(2026, 11, 28)
JRA_HOST, NAR_HOST = "www.jra.go.jp", "nar.netkeiba.com"

JRA_CONFIG = {
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
# NAR's own schedule here: a later first window, a night stop, its own request costs.
NAR_CONFIG = {
    **copy.deepcopy(JRA_CONFIG),
    "pools": ["WIN", "PLACE"],
    "windows": [{"from": "10:00", "every_s": 60}, {"from": "10:15", "every_s": 20}],
    "stop": "21:00",
    "requests": {"navigation": 1, "per_race": 1},
    # the owner's NAR block policy: pause 15 minutes, resume once at half rate
    "on_block": {"cooldown_min": 15, "resume_requests_per_second": 0.05},
}


# ---------------------------------------------------------------- time and expected values


def jst(h: int, m: int = 0, s: int = 0, day: date = SAT) -> datetime:
    return datetime(day.year, day.month, day.day, h, m, s, tzinfo=JST).astimezone(UTC)


def wall(hhmm: str, day: date) -> datetime:
    h, m = (int(part) for part in str(hhmm).split(":"))
    return jst(h, m, day=day)


def every_s_at(t: datetime, cfg) -> float | None:
    day = t.astimezone(JST).date()
    current = None
    for window in sorted(cfg["windows"], key=lambda w: wall(w["from"], day)):
        if wall(window["from"], day) <= t:
            current = float(window["every_s"])
    return current


def cost(n_races: int, cfg) -> int:
    return cfg["requests"]["navigation"] + cfg["requests"]["per_race"] * n_races


def is_hot(post: datetime, t: datetime, cfg) -> bool:
    hot = cfg["hot_races"]
    return (
        post - timedelta(minutes=hot["within_min"])
        <= t
        <= post + timedelta(minutes=hot["after_post_min"])
    )


def hhmm(t: datetime) -> str:
    return t.astimezone(JST).strftime("%H:%M:%S JST")


def card(first: tuple[int, int], races: int, every_min: int = 30) -> dict[int, datetime]:
    h, m = first
    return {n: jst(h, m) + timedelta(minutes=every_min * (n - 1)) for n in range(1, races + 1)}


def jra_meetings() -> dict[str, dict[int, datetime]]:
    """TOK, KYO, FKS: 12 races 30 minutes apart, first posts 10:00, 10:10, 10:20."""
    return {"TOK": card((10, 0), 12), "KYO": card((10, 10), 12), "FKS": card((10, 20), 12)}


def nar_meetings() -> dict[str, dict[int, datetime]]:
    """Urawa by day (8 races from 10:40) and a Sonoda nighter (12 races from 15:00 to 20:30)."""
    return {"URA": card((10, 40), 8), "SON": card((15, 0), 12)}


# ---------------------------------------------------------------- fakes


@dataclass
class Call:
    t: datetime
    kind: str  # discover | post_times | capture
    venue: str | None
    races: tuple[int, ...] | None = None
    error: str | None = None
    requests: int = 0


Rule = Callable[[str, str | None, int, datetime], BaseException | None]


class SourceSite:
    """One source's site as the scheduler sees it, every call recorded at the calling thread's
    time. A capture costs navigation + per_race x races requests. `fail(kind, venue, n, t)` (n:
    the 1-based count of that kind of call for that venue) returns an exception to raise, or None.
    With `barrier`, the first call waits (real time, at most 10 s) until every site sharing it has
    made its first call: the sources are being logged at once."""

    def __init__(
        self,
        clock: ThreadClock,
        source: str,
        host: str,
        meetings: dict[str, dict[int, datetime]],
        *,
        rps: float,
        nav: int,
        per_race: int,
        fail: Rule | None = None,
        barrier: threading.Barrier | None = None,
    ):
        self.clock, self.meetings = clock, meetings
        self.source, self.host, self.requests_per_second = source, host, rps
        self.nav, self.per_race = nav, per_race
        self.fail = fail or (lambda kind, venue, n, t: None)
        self.barrier = barrier
        self.calls: list[Call] = []
        self.requests = 0
        self.defects: list[str] = []
        self._counts: Counter = Counter()

    def _call(self, kind: str, venue: str | None, races=None) -> tuple[Call, BaseException | None]:
        if self.barrier is not None and not self.calls:
            try:
                self.barrier.wait()
            except threading.BrokenBarrierError:
                msg = f"{self.source}: the other source made no call while this one waited 10 s"
                self.defects.append(msg)
                raise Runaway(msg) from None
        call = Call(self.clock.now(), kind, venue, races)
        self.calls.append(call)
        self._counts[(kind, venue)] += 1
        if venue is not None and venue not in self.meetings:
            self.defects.append(f"{self.source}: {kind} for {venue}, not one of its venues")
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
        return dict(self.meetings[venue])

    def capture(self, venue: str, races) -> MeetingCapture:
        asked = None if races is None else tuple(races)
        call, exc = self._call("capture", venue, asked)
        every = sorted(self.meetings.get(venue, {}))
        if asked is not None and (not asked or set(asked) - set(every)):
            self.defects.append(f"{self.source}: capture of {venue} asks for races {asked}")
        if exc is not None:
            self._spend(call, 1 if isinstance(exc, (NotListed, CircuitOpen)) else self.nav, exc)
        wanted = every if asked is None else sorted(set(asked) & set(every))
        spent = self.nav + self.per_race * len(wanted)
        self._spend(call, spent)
        rows = [
            {"race_id": f"{self.source}:{SAT.isoformat()}:{venue}:{n:02d}", "pool": "WIN",
             "snapshot_utc": call.t}
            for n in wanted
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


def blocked(host: str, status: int) -> Callable[[], CircuitOpen]:
    return lambda: CircuitOpen(f"HTTP {status}", http_status=status, host=host)


def on_calls(kind: str, venue: str | None, numbers: set[int], make):
    def rule(k, v, n, t):
        return make() if (k, v) == (kind, venue) and n in numbers else None

    return rule


def from_time(kind: str, venue: str | None, start: datetime, make):
    pending = [make()]

    def rule(k, v, n, t):
        if (k, v) == (kind, venue) and t >= start and pending:
            return pending.pop(0)
        return None

    return rule


def first_call(make):
    """The site's first call, whatever it is, raises make(); later calls are served."""
    pending = [make()]

    def rule(k, v, n, t):
        return pending.pop(0) if pending else None

    return rule


class Resumer:
    """A Source.resume for the tests: each call notes (the calling thread's time, the rate asked)
    and returns a fresh site from `make(rate)`, kept in `sites`."""

    def __init__(self, clock: ThreadClock, make: Callable[[float], SourceSite]):
        self.clock, self.make = clock, make
        self.calls: list[tuple[datetime, float]] = []
        self.sites: list[SourceSite] = []

    def __call__(self, requests_per_second: float) -> SourceSite:
        self.calls.append((self.clock.t, float(requests_per_second)))
        site = self.make(float(requests_per_second))
        self.sites.append(site)
        return site


def resumers(clock: ThreadClock, *, nar_fail: Rule | None = None) -> dict[str, Resumer]:
    """Per source a Resumer making a fresh site of that source at the rate asked (NAR's failing
    by `nar_fail`)."""
    return {
        "jra": Resumer(clock, lambda rps: SourceSite(clock, "jra", JRA_HOST, jra_meetings(),
                                                     rps=rps, nav=2, per_race=3)),
        "nar": Resumer(clock, lambda rps: SourceSite(clock, "nar", NAR_HOST, nar_meetings(),
                                                     rps=rps, nav=1, per_race=1, fail=nar_fail)),
    }  # fmt: skip


def held_after(after: datetime, reported: threading.Event, late: list[str]) -> Rule:
    """For the healthy site: its first call at or after `after` (its own time) waits, in real time
    up to 5 s, for the other source's failure to be reported; a wait that runs out is noted in
    `late` (the report was held back until this source's day ended)."""
    waited: list[datetime] = []

    def rule(k, v, n, t):
        if t >= after and not waited:
            waited.append(t)
            if not reported.wait(5.0):
                late.append(f"no failure reported by {hhmm(t)} of the healthy source's day")
        return None

    return rule


def make_sites(clock, *, jra_rps=1.0, nar_rps=0.1, jra_fail=None, nar_fail=None, barrier=None):
    jra = SourceSite(clock, "jra", JRA_HOST, jra_meetings(), rps=jra_rps, nav=2, per_race=3,
                     fail=jra_fail, barrier=barrier)  # fmt: skip
    nar = SourceSite(clock, "nar", NAR_HOST, nar_meetings(), rps=nar_rps, nav=1, per_race=1,
                     fail=nar_fail, barrier=barrier)  # fmt: skip
    return {"jra": jra, "nar": nar}


CONFIGS = {"jra": JRA_CONFIG, "nar": NAR_CONFIG}


def run(
    clock: ThreadClock,
    sites: dict[str, SourceSite],
    markers: dict[str, Path],
    *,
    configs=CONFIGS,
    resume: dict[str, Resumer] | None = None,
    on_error=None,
):
    stores = {key: FakeStore() for key in sites}
    sources = [
        Source(
            site=sites[k],
            store=stores[k],
            config=configs[k],
            block_marker=markers[k],
            **({"resume": resume[k]} if resume else {}),
        )  # fmt: skip
        for k in sites
    ]
    extra = {} if on_error is None else {"on_error": on_error}
    summaries = run_sources(SAT, sources, now=clock.now, sleep=clock.sleep, **extra)
    assert clock.errors == [], clock.errors
    for site in [*sites.values(), *(s for r in (resume or {}).values() for s in r.sites)]:
        assert site.defects == [], site.defects
    assert set(summaries) == set(sites), f"summaries for {sorted(summaries)}"
    return summaries, stores


# ---------------------------------------------------------------- reading what a fake saw


def caps(site: SourceSite, venue: str | None = None, ok: bool | None = None) -> list[Call]:
    return [
        c
        for c in site.calls
        if c.kind == "capture"
        and (venue is None or c.venue == venue)
        and (ok is None or (c.error is None) == ok)
    ]


def asked(call: Call, site: SourceSite) -> list[int]:
    return sorted(site.meetings[call.venue]) if call.races is None else sorted(call.races)


def assert_rounds_fit_budget(site: SourceSite, cfg) -> None:
    """Every round of this site (its captures at one instant), and every every_s span inside one
    window, costs at most its window's every_s x this site's rate, or is one hot race alone."""
    rps = site.requests_per_second
    rounds: dict[datetime, list[Call]] = defaultdict(list)
    for call in caps(site):
        rounds[call.t].append(call)
    instants = sorted(rounds)

    def check(group: list[Call], budget: float, where: str) -> None:
        planned = sum(cost(len(asked(c, site)), cfg) for c in group)
        if planned <= budget + EPS:
            return
        assert budget < cost(1, cfg), f"{where}: {planned} requests over a budget of {budget}"
        assert len(group) == 1, f"{where}: {len(group)} captures over a budget of {budget}"
        races = asked(group[0], site)
        assert len(races) == 1, f"{where}: races {races} over a budget of {budget}"
        post = site.meetings[group[0].venue][races[0]]
        assert is_hot(post, group[0].t, cfg), f"{where}: over the budget for a race not hot"

    for i, start in enumerate(instants):
        every = every_s_at(start, cfg)
        where = f"{site.source} round at {hhmm(start)} (rps {rps})"
        assert every is not None, f"{where}: a capture before the source's first window"
        check(rounds[start], every * rps, where)
        end = start + timedelta(seconds=every)
        if every_s_at(end, cfg) == every:
            span = [c for t in instants[i:] if t < end for c in rounds[t]]
            check(span, every * rps, f"{site.source}: {every} s from {hhmm(start)} (rps {rps})")


def assert_every_race_captured_while_hot(site: SourceSite, cfg) -> None:
    for venue, posts in site.meetings.items():
        ok = caps(site, venue, ok=True)
        for n, post in posts.items():
            hot = [c for c in ok if n in asked(c, site) and is_hot(post, c.t, cfg)]
            assert hot, f"{site.source} {venue} race {n} (post {hhmm(post)}): never captured hot"


def assert_hot_races_followed(site: SourceSite, cfg, every: float) -> None:
    """Each race is captured through its hot window at most `every` seconds apart."""
    hot = cfg["hot_races"]
    for venue, posts in site.meetings.items():
        ok = caps(site, venue, ok=True)
        for n, post in posts.items():
            lo = post - timedelta(minutes=hot["within_min"])
            hi = post + timedelta(minutes=hot["after_post_min"])
            times = [c.t for c in ok if n in asked(c, site) and lo <= c.t <= hi]
            where = f"{site.source} {venue} race {n} (post {hhmm(post)})"
            assert times, f"{where}: never captured in its hot window"
            edges = [lo, *times, hi]
            gaps = [(b - a).total_seconds() for a, b in zip(edges, edges[1:], strict=False)]
            assert max(gaps) <= every + EPS, f"{where}: a {max(gaps)} s gap while hot"


def assert_logged_its_day(site: SourceSite, cfg) -> None:
    """Every race captured while hot, and nothing sent for a venue from its end on."""
    assert_every_race_captured_while_hot(site, cfg)
    for venue, posts in site.meetings.items():
        end = max(posts.values()) + timedelta(minutes=cfg["after_last_race_min"])
        late = [c for c in site.calls if c.venue == venue and c.t >= end]
        assert late == [], f"{site.source} {venue}: {len(late)} calls after its end {hhmm(end)}"


# ---------------------------------------------------------------- the DoD


@pytest.mark.parametrize(
    ("jra_rps", "nar_rps"), [(1.0, 0.1), (2.0, 0.25)], ids=["jra-1-nar-0.1", "jra-2-nar-0.25"]
)
def test_two_sources_keep_their_own_rate_budgets(tmp_path, jra_rps, nar_rps):
    clock = ThreadClock(jst(6, 30))
    barrier = threading.Barrier(2, timeout=10)  # both sources in flight at once
    sites = make_sites(clock, jra_rps=jra_rps, nar_rps=nar_rps, barrier=barrier)
    markers = {k: tmp_path / f"{k}-block.txt" for k in sites}

    summaries, stores = run(clock, sites, markers)
    jra, nar = sites["jra"], sites["nar"]

    for key, site in sites.items():
        cfg = CONFIGS[key]
        summary = summaries[key]
        assert set(summary.venues) == set(site.meetings), key
        assert summary.stopped is None, (key, summary.stopped)
        assert {v for v, _ in stores[key].writes} == set(site.meetings), key
        # each source's rounds are held to its own window x its own site's rate
        assert_rounds_fit_budget(site, cfg)
        assert_logged_its_day(site, cfg)
        ok = caps(site, ok=True)
        assert sum(c.requests for c in ok) <= summary.requests <= site.requests, key
        assert not markers[key].exists(), f"{key}: a block marker without a block"

    # Each source keeps its own windows: NAR sends nothing before its 10:00, JRA starts at 07:00.
    assert min(c.t for c in nar.calls) >= jst(10, 0), hhmm(min(c.t for c in nar.calls))
    first_jra = min(c.t for c in jra.calls)
    assert jst(7, 0) <= first_jra < jst(7, 5), hhmm(first_jra)
    # JRA's budget is its own, not NAR's: three venues' hot races (15 requests a round) are each
    # captured every 20 s, which a budget of NAR's 20 s x nar_rps (2 or 5 requests) cannot hold.
    assert 20 * nar_rps < 3 * cost(1, JRA_CONFIG)
    assert_hot_races_followed(jra, JRA_CONFIG, every=20)
    peak = max(
        sum(cost(len(asked(c, jra)), JRA_CONFIG) for c in caps(jra) if c.t == t)
        for t in {c.t for c in caps(jra)}
    )
    assert peak > 20 * nar_rps, f"JRA's largest round ({peak}) fits NAR's budget"
    # and NAR's is its own, not JRA's: it has to leave races out of its rounds
    assert summaries["nar"].deferred > 0, "NAR at its own rate never had to defer a race"


@pytest.mark.parametrize(
    ("which", "first", "second"),
    [
        ("nar", lambda: from_time("capture", "URA", jst(11, 0), blocked(NAR_HOST, 403)),
         lambda: first_call(blocked(NAR_HOST, 403))),
        ("nar", lambda: on_calls("post_times", "SON", {1}, blocked(NAR_HOST, 429)),
         lambda: from_time("capture", "URA", jst(12, 0), blocked(NAR_HOST, 429))),
        ("nar", lambda: on_calls("discover", None, {1}, blocked(NAR_HOST, 403)),
         lambda: first_call(blocked(NAR_HOST, 403))),
        ("jra", lambda: from_time("capture", "KYO", jst(12, 0), blocked(JRA_HOST, 429)), None),
    ],
    ids=["nar-capture-403", "nar-post-times-429", "nar-discover-403", "jra-capture-429"],
)  # fmt: skip
def test_a_block_on_one_source_leaves_the_other_logging(tmp_path, which, first, second):
    """JRA: the first block stops it for the day. NAR: the first block pauses it, it resumes once
    on a fresh site, and the second block (`second`, on that site) stops it for the day. Either
    way the other source logs its whole day."""
    other = "jra" if which == "nar" else "nar"
    clock = ThreadClock(jst(6, 30))
    sites = make_sites(clock, **{f"{which}_fail": first()})
    markers = {k: tmp_path / f"{k}-block.txt" for k in sites}
    resume = resumers(clock, nar_fail=second() if second else None)

    summaries, _ = run(clock, sites, markers, resume=resume)
    hit, spared = sites[which], sites[other]

    # The blocked source's sites in turn: the first, then (NAR only) the one its resume gave.
    chain = [hit, *resume[which].sites]
    assert len(chain) == (2 if which == "nar" else 1), (
        f"{which}: resumed {len(chain) - 1} times; on_block {CONFIGS[which]['on_block']}"
    )
    assert resume[other].calls == [], f"{other} resumed after a block on {which}"
    # On each site, nothing after its block.
    for site in chain:
        opened = [i for i, c in enumerate(site.calls) if c.error == "CircuitOpen"]
        assert len(opened) == 1, f"{which}: a scripted block never came: {site.calls[-3:]}"
        assert opened[0] == len(site.calls) - 1, f"{which}: called after a block: {site.calls[-3:]}"
    if which == "nar":
        policy = CONFIGS["nar"]["on_block"]
        [(_, rate)] = resume["nar"].calls
        assert rate == pytest.approx(policy["resume_requests_per_second"]), rate
        pause_end = hit.calls[-1].t + timedelta(minutes=policy["cooldown_min"])
        fresh = chain[1]
        assert fresh.calls[0].t >= pause_end - timedelta(seconds=EPS), (
            f"NAR resumed at {hhmm(fresh.calls[0].t)}, before the pause ended {hhmm(pause_end)}"
        )
        assert_rounds_fit_budget(fresh, CONFIGS["nar"])  # the fresh site's rate, 0.05
        # JRA never paused while NAR did: its hot races followed every 20 s all day.
        assert_hot_races_followed(spared, JRA_CONFIG, every=20)
    stops = {v: vs.stopped for v, vs in summaries[which].venues.items()}
    assert all(s == "circuit_open" for s in stops.values()), stops
    assert summaries[which].stopped == "circuit_open"
    if hit.calls[-1].kind == "capture":
        assert caps(hit, ok=True), f"{which}: nothing captured before the block"
    text = markers[which].read_text(encoding="utf-8")
    assert hit.host in text and SAT.isoformat() in text, text

    # The other source: its whole day, as if alone; no marker of its own.
    assert summaries[other].stopped is None
    assert set(summaries[other].venues) == set(spared.meetings)
    assert all(vs.stopped != "circuit_open" for vs in summaries[other].venues.values())
    assert_logged_its_day(spared, CONFIGS[other])
    assert_rounds_fit_budget(spared, CONFIGS[other])
    assert not markers[other].exists(), f"{other}'s marker written for a block on {which}"

    # A later run the same day: nothing for the blocked source, the other logs again.
    clock = ThreadClock(jst(13, 0))
    sites = make_sites(clock)
    again = resumers(clock)
    summaries, _ = run(clock, sites, markers, resume=again)
    assert sites[which].calls == [], f"{which} after its block: {sites[which].calls[:3]}"
    assert again[which].calls == [], f"{which} resumed in a run after its day's last block"
    assert summaries[which].requests == 0
    assert caps(sites[other], ok=True), f"{other} logged nothing after {which}'s block"
    assert summaries[other].stopped is None


@pytest.mark.parametrize(
    "policy",
    [
        {"cooldown_min": 15, "resume_requests_per_second": 0.05},
        {"cooldown_min": 40, "resume_requests_per_second": 0.02},
    ],
    ids=["15min-at-0.05", "40min-at-0.02"],
)
@pytest.mark.parametrize(
    "first",
    [
        lambda: from_time("capture", "URA", jst(11, 0), blocked(NAR_HOST, 403)),
        lambda: on_calls("discover", None, {1}, blocked(NAR_HOST, 429)),
    ],
    ids=["capture-403", "discover-429"],
)
def test_a_first_nar_block_pauses_nar_alone_then_resumes_at_the_slower_rate(
    tmp_path, policy, first
):
    nar_cfg = {**NAR_CONFIG, "on_block": policy}
    configs = {"jra": JRA_CONFIG, "nar": nar_cfg}
    clock = ThreadClock(jst(6, 30))
    sites = make_sites(clock, nar_fail=first())
    markers = {k: tmp_path / f"{k}-block.txt" for k in sites}
    resume = resumers(clock)

    summaries, _ = run(clock, sites, markers, configs=configs, resume=resume)
    hit, jra = sites["nar"], sites["jra"]

    # The block: the first site's last call; then one resume, at the policy's rate.
    blocks = [c for c in hit.calls if c.error == "CircuitOpen"]
    assert blocks == [hit.calls[-1]], f"NAR's first site after its block: {hit.calls[-3:]}"
    assert len(resume["nar"].calls) == 1, f"NAR resumed {len(resume['nar'].calls)} times"
    [(_, rate)] = resume["nar"].calls
    assert rate == pytest.approx(policy["resume_requests_per_second"]), rate
    [fresh] = resume["nar"].sites

    # The pause: nothing to NAR for cooldown_min, then the rest of its day on the fresh site.
    resumed_at = blocks[0].t + timedelta(minutes=policy["cooldown_min"])
    assert fresh.calls, "NAR never resumed"
    assert fresh.calls[0].t >= resumed_at - timedelta(seconds=EPS), (
        f"NAR resumed at {hhmm(fresh.calls[0].t)}, before its pause ended {hhmm(resumed_at)}"
    )
    assert [c for c in fresh.calls if c.error] == [], fresh.calls[:3]
    assert_rounds_fit_budget(fresh, nar_cfg)  # at the resume rate: fresh.requests_per_second
    within = timedelta(minutes=nar_cfg["hot_races"]["within_min"])
    for venue, posts in fresh.meetings.items():
        for n, post in posts.items():
            if post - within < resumed_at:
                continue  # hot before the resume, maybe during the pause
            hot = [c for c in caps(fresh, venue, ok=True)
                   if n in asked(c, fresh) and is_hot(post, c.t, nar_cfg)]  # fmt: skip
            assert hot, f"NAR {venue} race {n} (post {hhmm(post)}): never captured after resume"
        end = max(posts.values()) + timedelta(minutes=nar_cfg["after_last_race_min"])
        assert [c for c in fresh.calls if c.venue == venue and c.t >= end] == [], venue
    assert summaries["nar"].stopped is None, summaries["nar"].stopped
    stops = {v: vs.stopped for v, vs in summaries["nar"].venues.items()}
    assert set(stops) == set(nar_meetings()) and set(stops.values()) == {"ended"}, stops

    # JRA: never paused, never resumed, no marker; its whole day at its own rate.
    assert resume["jra"].calls == []
    assert summaries["jra"].stopped is None
    assert_logged_its_day(jra, JRA_CONFIG)
    assert_rounds_fit_budget(jra, JRA_CONFIG)
    assert_hot_races_followed(jra, JRA_CONFIG, every=20)
    assert not markers["jra"].exists()


@pytest.mark.parametrize(
    ("which", "fail", "error"),
    [
        ("nar", lambda e: on_calls("discover", None, {1}, lambda: e),
         FetchError("HTTP 500 for race_list/20261128", http_status=500)),
        ("nar", lambda e: from_time("capture", "URA", jst(11, 0), lambda: e),
         KeyError("odds_tan_block")),
        ("jra", lambda e: from_time("capture", "KYO", jst(12, 0), lambda: e),
         RuntimeError("an odds table the parser does not know")),
    ],
    ids=["nar-discover-fetch-error", "nar-capture-key-error", "jra-capture-runtime-error"],
)  # fmt: skip
def test_an_error_in_one_source_is_reported_at_once_and_the_other_logs_its_day(
    tmp_path, which, fail, error
):
    other = "jra" if which == "nar" else "nar"
    clock = ThreadClock(jst(6, 30))
    reported_now = threading.Event()
    late: list[str] = []
    # the healthy source waits, part way through its day, for the failure to be reported
    hold = held_after(jst(14, 0) if other == "jra" else jst(15, 30), reported_now, late)
    sites = make_sites(clock, **{f"{which}_fail": fail(error), f"{other}_fail": hold})
    markers = {k: tmp_path / f"{k}-block.txt" for k in sites}
    reported: list[tuple[str, BaseException]] = []

    def on_error(key, exc):
        reported.append((key, exc))
        reported_now.set()

    summaries, _ = run(clock, sites, markers, on_error=on_error)  # returns: does not raise
    failed, healthy = sites[which], sites[other]

    # The failure: reported once, naming the source and its error, while the other still logged.
    assert [(k, type(e), str(e)) for k, e in reported] == [(which, type(error), str(error))]
    assert late == [], late
    assert failed.calls[-1].error == type(error).__name__, failed.calls[-3:]
    assert [c for c in failed.calls if c.error] == [failed.calls[-1]], "called after its error"
    assert summaries[which].stopped == "error", summaries[which].stopped
    stops = {v: vs.stopped for v, vs in summaries[which].venues.items()}
    assert all(s in ("error", "ended") for s in stops.values()), stops
    if failed.calls[-1].venue is not None:
        assert stops[failed.calls[-1].venue] == "error", stops

    # The other source: its whole day, as if alone.
    assert summaries[other].stopped is None
    assert set(summaries[other].venues) == set(healthy.meetings)
    assert all(vs.stopped == "ended" for vs in summaries[other].venues.values())
    assert_logged_its_day(healthy, CONFIGS[other])
    assert_rounds_fit_budget(healthy, CONFIGS[other])
    # An error is not a block.
    assert not markers[which].exists() and not markers[other].exists()


# ---------------------------------------------------------------- logger.yaml


def test_logger_yaml_ships_a_schedule_per_source():
    shipped = copy.deepcopy(load_country_yaml("japan", "logger"))
    assert {"jra", "nar"} >= set(shipped["sources"]) and "jra" in shipped["sources"]

    jra = source_config(shipped, "jra")
    # #358: bracket quinella every round with tier 1, on a page of its own; #387: the owner's
    # 60 s hot grain and 5-minute floor in JRA's hot_races, JRA's venue_rates and its back-off
    # on_block (test_five_minute_floor.py); NAR inherits none of them (checked below)
    shipped_jra = {
        **JRA_CONFIG,
        "pools": ["WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE", "BRACKET_QUINELLA"],
        "requests": {"navigation": 2, "per_race": 4},
        "hot_races": {**JRA_CONFIG["hot_races"], "every_s": 60, "floor_s": 300},
    }
    shipped_jra.pop("on_block")
    for key, value in shipped_jra.items():
        assert jra.get(key) == value, f"JRA's {key} changed: {jra.get(key)!r}"
    assert jra["on_block"].get("cooldown_min") is not None, f"JRA's on_block {jra['on_block']}"
    assert jra.get("venue_rates"), "JRA has no venue_rates (#387)"
    assert jra.get("day_before") == shipped.get("day_before") is not None, "JRA's day_before"

    nar = source_config(shipped, "nar")
    for t in (jst(10, 20), jst(14, 0), jst(20, 50)):
        assert every_s_at(t, nar) is not None, f"no NAR window in force at {hhmm(t)}"
    assert wall(nar["stop"], SAT) >= jst(21, 0), f"NAR stops at {nar['stop']}"
    assert set(nar["pools"]) <= {"WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE"}, nar["pools"]
    pages = {"WIN": "b1", "PLACE": "b1", "QUINELLA": "b4", "QUINELLA_PLACE": "b5"}
    assert nar["requests"]["navigation"] == 1
    assert nar["requests"]["per_race"] == len({pages[p] for p in nar["pools"]})
    http = load_country_yaml("japan", "nar")["http"]["hosts"]
    rps = float(http.get(NAR_HOST, http["default"])["requests_per_second"])
    for t in (jst(10, 20), jst(14, 0), jst(20, 50)):
        assert every_s_at(t, nar) * rps >= cost(1, nar) - EPS, (
            f"at {hhmm(t)} a NAR round ({every_s_at(t, nar)} s x {rps}) fits no race"
        )
    assert not nar.get("tier2"), "NAR has no tier-2 capture"
    assert not {"every_s", "floor_s"} & set(nar["hot_races"]), (
        f"NAR inherited JRA's hot grain or floor (#387): {nar['hot_races']}"
    )
    assert "venue_rates" not in nar, f"NAR inherited JRA's venue_rates (#387): {nar}"
    assert "day_before" not in nar, "NAR has no day-before window (#358)"
    # The owner's NAR block policy: pause 15 minutes, resume once at half nar.yaml's rate. JRA's
    # is its own since #387 (a pause, then 1 req/s, then the floor only).
    assert nar["on_block"]["cooldown_min"] == 15, nar["on_block"]
    assert nar["on_block"]["resume_requests_per_second"] == pytest.approx(rps / 2), nar["on_block"]
