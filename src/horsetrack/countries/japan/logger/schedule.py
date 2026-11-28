"""Scheduling every JRA venue's odds captures under one request budget (ticket #24).

Ported from HKTrack's `ingest/live.py` (`poll_odds`, `_poll_loop`, `hot_races`) and
`scripts/oddsloop.sh`, which ran one loop per venue. Changes, fixing:

(f) after a 403/429 opened the breaker, the supervisor retried about every 30 s until the stop
    time. Here every CircuitOpen appends a block marker line (host, Tokyo race date) and the
    schedule's `on_block` decides the rest (below): never a retry loop.
(g) a full capture is navigation + per_race x races requests, and nothing budgeted them across
    venues. Here every round has a budget, the window's every_s x the host's requests per second:
    hot races first, nearest post first, then the venues' full cycles race by race.

The hot grain and the floor (ticket #387, logger.yaml `hot_races.every_s` and `floor_s`, JRA only):
each race is planned by its own deadline (VenueState.captured: when its latest capture started):
a hot race at least every every_s, every other race whose post is ahead at least every floor_s
(counted from the window in force). A round plans the races its next round would be late for
first, then those the next round might miss (it may start them anywhere in its every_s), then
the races a venue in the round takes along, then the floor's work ahead at a pace, then the full
cycles of races never captured. In run_day a round's captures spend what is left of its every_s
less the results polls due in it, so rounds keep their time; a venue's post-time refresh waits
for a round with room, and a capture inside min_poll_interval_s lets another venue's go first
when that still starts it by its deadline. `venue_rates`: a day of N venues runs at the rate of
the entry with the greatest min_venues <= N, through `resume(rate)`, once its venues are known.

Tier 2 (ticket #130, logger.yaml `tier2`): each race's tier-2 pools once per grid point. Grid
point g of a race falls due at post - g and belongs to the stretch until the next smaller grid
value (the post after the last). Each round runs tier 1 first, exactly as without tier 2, then
spends only what is left of its budget on the grid points due, the stretch ending soonest first;
one that does not fit is tried again in a later round of its stretch, and one with no successful
capture when its stretch ends counts in VenueSummary.tier2_skipped. A failed tier-2 capture is
logged and never touches tier 1 (a CircuitOpen is still a block, per `on_block`).

Results (ticket #225, logger.yaml `results`, JRA only): each venue with post times polls the
site's `published_results` from its first post to its last post + until_min, at most every
every_s, and writes each race the first time a poll shows its official result, at the instant that
poll returned, with a lower bound: the instant the run's previous successful poll returned (None, a
late sighting, for a race shown on the run's first successful poll). Polls ride on the rounds
while odds are logged (after tier 1, ahead of tier 2's share of the budget) and set their own pace
once they end; a venue stops polling once every race with a post time is logged. A failed poll is
retried at the next one, is never a bound and is not an odds failure; a CircuitOpen is a block,
as from any site call.

The day before (ticket #358, logger.yaml `day_before`, JRA only): the race date's sales open the
evening before (about 18:30 Tokyo), and from then on every race is logged at least at JV-Link's
time-series grain. Nothing is sent before probe_from on the day before (a run started earlier
sleeps until then). From max(start, probe_from) the site's `sales_open()` is called every
probe_every_s (start to start) while before probe_until, until one says True; a run started at or
after probe_until and before the first window W calls it once, at its start (a restart in the
night resumes the window); a run started at or after W never probes. A probe that raises a
CircuitOpen is a block for the race date, as from any site call; any other failed probe is retried
at the next and is no venue's failure. The first probe that says True opens the window
(DaySummary.sales_open_utc): discovery, post times and rounds start there, every day_before.every_s,
each round capturing every race of every venue (a full cycle due every round, whatever
hot_races.all_every_s says) within its budget (every_s x the rate), until W. At the first round
from W every venue's cycle falls due at once, so no race waits longer than the night's grain across
the switch; from W the race date runs as before (its windows, hot races, tier 2, results). Without
an open (no racing that day, or the sales not open by probe_until) only the probes are sent before
W, and the day runs as without the block.

Several sources (ticket #38, logger.yaml `sources`): `run_sources` runs each source's run_day on
a thread of its own, all at once, each with its own site (so its own host, rate and budget), store,
schedule (`source_config`), block marker and race date (#358: JRA's may be tomorrow while NAR's is
today). A block on one source's host stops that source only. Any other error escaping a source's
day stops that source only (stopped 'error'): `on_error` hears of it at once, and run_sources still
returns every source's summary.

The block policy is the schedule's `on_block`. Every block appends a marker line. Without
`ladder`, a run that finds a line for its host and date sends nothing; `cooldown_min: null` stops
the day at the first block; N pauses that source N minutes, then resumes once: on
`resume(rate)`'s site when run_day is given `resume` (NAR: a fresh Fetcher at
`resume_requests_per_second`), else on the same site. A second block stops the day. With
`ladder: true` (#387, JRA: the owner's "a 403 or 429 must not end the day's logging") no block
ends the day: every block pauses cooldown_min, then the day goes on on `resume(rate)` at
resume_requests_per_second, after the first block with the full schedule and after the second
and every later one with the floor only (hot races at floor_s too, in the finest window's
rounds). A run that finds lines for its host and date waits out the pause after the latest, then
logs at the level they give. DaySummary.blocks holds every block's status, the marker's too.

`plan_round` is pure. `run_day` drives it with an injected clock (`now`, `sleep`) and reaches the
site and the store only through the protocols below. `config` is a mapping shaped like
`countries/japan/config/logger.yaml`. Window and stop times are Asia/Tokyo wall-clock times; every
instant is an aware datetime, and post times are UTC.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from horsetrack.core.ingest.fetcher import CircuitOpen, FetchError
from horsetrack.countries.japan.jra.capture import MeetingCapture, NotListed, WrongMeeting

log = logging.getLogger(__name__)

JST = ZoneInfo("Asia/Tokyo")
# A JRA meeting's usual card: the cost basis of a capture whose races are not known yet.
RACES_IF_UNKNOWN = 12
_EPS = 1e-9
# What a failed site call can raise without ending the day (CircuitOpen does end it). ValueError is
# the capture's verdict on an error page.
_CALL_ERRORS = (WrongMeeting, NotListed, FetchError, ValueError)
_RESULTS_REQUESTS = 2  # a results poll: the レース結果 menu and the meeting's race list


class Site(Protocol):
    source: str  # its source key, "jra" or "nar" (run_sources keys its summaries by it)
    host: str
    requests_per_second: float

    def discover(self) -> list[str]: ...

    def post_times(self, venue: str) -> dict[int, datetime]: ...  # {} = unknown

    def capture(self, venue: str, races: list[int] | None) -> MeetingCapture: ...

    def capture_tier2(self, venue: str, races: list[int]) -> MeetingCapture: ...  # with `tier2`

    def published_results(self, venue: str) -> set[int]: ...  # with `results`

    def sales_open(self) -> bool: ...  # with `day_before`: the race date's sales are open


class Store(Protocol):
    def write(self, venue: str, capture: MeetingCapture) -> int: ...  # rows written

    def write_post_times(self, venue: str, times: dict[int, datetime]) -> None: ...

    # with `results`: the races first seen published at `seen_utc`, bounded below by
    # `unpublished_utc` (None: a late sighting, shown on the run's first successful poll)
    def write_published(
        self,
        venue: str,
        races: Iterable[int],
        seen_utc: datetime,
        unpublished_utc: datetime | None,
    ) -> None: ...


@dataclass
class VenueState:
    """What plan_round knows of one venue. run_day stamps `last_capture` and `cycle_start` with the
    start of the round that ran the capture, so rounds every_s apart compare exactly."""

    code: str
    post_times: dict[int, datetime] = field(default_factory=dict)  # race -> post (UTC); {} unknown
    last_capture: datetime | None = None  # when the latest capture started; None: never
    cycle_start: datetime | None = None  # when the current (or last) full cycle started
    cycle_left: frozenset[int] | None = None  # races the cycle in progress still needs
    races_seen: tuple[int, ...] = ()  # race numbers captures returned (used without post times)
    stopped: str | None = None
    # race -> when its latest successful capture started (#387; run_day stamps the call's own
    # start); a race missing has never been captured
    captured: dict[int, datetime] = field(default_factory=dict)


@dataclass
class Job:
    venue: str
    races: list[int] | None  # None: every race, for a venue whose races are not known yet
    requests: int  # navigation + per_race x races
    # #387: the earliest deadline of its races (None: none, as without every_s and floor_s)
    due: datetime | None = field(default=None, compare=False, repr=False)


@dataclass
class VenueSummary:
    snapshots: int = 0  # successful captures
    rows: int = 0  # rows the store wrote
    failures: int = 0
    wrong_meeting: int = 0
    stopped: str | None = None  # 'ended', 'wrong_meeting', 'circuit_open' or 'error'
    longest_cycle_s: float = 0.0  # longest full cycle: first capture start to last capture end
    tier2_skipped: int | None = None  # grid points (race x grid value) not captured; None: off


@dataclass
class DaySummary:
    venues: dict[str, VenueSummary] = field(default_factory=dict)
    requests: int = 0  # requests the successful captures sent
    deferred: int = 0  # race captures a round left out for budget
    stopped: str | None = None  # 'circuit_open' when a block ended the day, 'error' (run_sources)
    sales_open_utc: datetime | None = None  # the probe that found the sales open (#358); None: none
    # the HTTP status (None: unknown) of every block of the day, the marker's included (#387)
    blocks: list[int | None] = field(default_factory=list)


# ---------------------------------------------------------------- time


def _secs(delta: timedelta) -> float:
    return delta.total_seconds()


def _wall(hhmm: str, day: date) -> datetime:
    """The UTC instant of a Tokyo wall-clock "HH:MM" on `day`."""
    at = datetime.strptime(str(hhmm), "%H:%M").time()
    return datetime.combine(day, at, tzinfo=JST).astimezone(timezone.utc)


def _windows(day: date, config: Mapping[str, Any]) -> list[tuple[datetime, float]]:
    return sorted((_wall(w["from"], day), float(w["every_s"])) for w in config["windows"])


def _every_s(now: datetime, config: Mapping[str, Any], day: date | None = None) -> float | None:
    """every_s of the window in force at `now` (windows of `day`, by default now's Tokyo date);
    None before the first window."""
    current = None
    for start, every in _windows(day or now.astimezone(JST).date(), config):
        if start <= now:
            current = every
    return current


def _is_hot(post: datetime, now: datetime, config: Mapping[str, Any]) -> bool:
    hot = config["hot_races"]
    before = timedelta(minutes=hot["within_min"])
    return post - before <= now <= post + timedelta(minutes=hot["after_post_min"])


# ---------------------------------------------------------------- cycles


def _races(state: VenueState) -> list[int]:
    """The meeting's race numbers: from the post times, else from captures; [] when unknown."""
    return sorted(state.post_times) or list(state.races_seen)


def _cycle_left(state: VenueState) -> list[int]:
    """The races the full cycle in progress still needs ([]: no cycle in progress)."""
    left = state.cycle_left or frozenset()
    return [n for n in _races(state) if n in left]


def _new_cycle_due(state: VenueState, now: datetime, config: Mapping[str, Any]) -> bool:
    every = timedelta(seconds=config["hot_races"]["all_every_s"])
    return not _cycle_left(state) and (
        state.cycle_start is None or now - state.cycle_start >= every
    )


def _cycle_due(state: VenueState, now: datetime, config: Mapping[str, Any]) -> list[int] | None:
    """The races the venue's full cycle needs at `now`: [] when none is due, None for every race of
    a meeting whose races are not known yet."""
    left = _cycle_left(state)
    if left:
        return left
    if not _new_cycle_due(state, now, config):
        return []
    races = _races(state)
    if races and config["hot_races"].get("floor_s") is not None:
        # with the floor (#387) a race with a post time and a capture is kept by its deadline
        return [n for n in races if n not in state.post_times or n not in state.captured]
    return races or None


def _race_numbers(rows: Iterable[Mapping[str, Any]]) -> set[int]:
    """Race numbers of capture rows (race_id jra:<date>:<venue>:<NN>)."""
    out = set()
    for row in rows:
        try:
            out.add(int(str(row["race_id"]).rsplit(":", 1)[-1]))
        except (KeyError, ValueError):
            continue
    return out


# ---------------------------------------------------------------- planning


def plan_round(
    now: datetime, venues: list[VenueState], config: Mapping[str, Any], rps: float
) -> list[Job]:
    """The captures of one round starting at `now`, within its budget (every_s x rps)."""
    return _plan(now, venues, config, rps)[0]


def _grain(config: Mapping[str, Any]) -> bool:
    """Whether a schedule keeps its races by deadline (#387: hot_races.every_s or floor_s)."""
    hot = config["hot_races"]
    return hot.get("every_s") is not None or hot.get("floor_s") is not None


def _window_start(now: datetime, config: Mapping[str, Any]) -> datetime | None:
    """The start of the window in force at `now` (None before the first)."""
    starts = [w for w, _ in _windows(now.astimezone(JST).date(), config) if w <= now]
    return starts[-1] if starts else None


@dataclass
class _Deadlines:
    """A round's races with hot_races.every_s or floor_s (#387), as (venue index, race)."""

    first: list[tuple[int, int]]  # the must races, then those the next round might miss
    along: list[tuple[int, int]]  # races not hot due within two rounds more: with their venue
    ahead: list[tuple[int, int]]  # races not hot and not fresh, the rest: paced
    hot: set[tuple[int, int]]
    floor_races: int  # races kept by the floor (not hot, post ahead, captured)
    dues: dict[tuple[int, int], datetime]  # each race's deadline


def _deadlines(
    now: datetime, eligible: list[VenueState], config: Mapping[str, Any], every: float
) -> _Deadlines:
    """A race's deadline: a hot race's latest capture + every_s (counted from the hot window's
    start when it was captured before it), at most its latest + floor_s; a race not hot whose post
    is ahead, its latest + floor_s (counted from the window's start: the floor runs from the
    window in force). A hot race whose deadline is past its window's end is done. A must race
    (the next round would be late: age >= every_s - e, or >= floor_s - e) comes first; then a
    race whose deadline falls before the end of the next round (which may start it anywhere in
    its every_s). Each group goes by deadline, then by post. A hot race never captured, or every
    hot race without every_s, is a must at once; a race not hot never captured is the full
    cycle's."""
    hot = config["hot_races"]
    grain, floor = hot.get("every_s"), hot.get("floor_s")
    e = timedelta(seconds=every)
    # the floor counts from the window's start, so the hot grain comes first as it starts; with
    # no grain finer than the floor (the floor only) each race keeps its own deadline
    since = _window_start(now, config) if grain != floor else None
    before, after = timedelta(minutes=hot["within_min"]), timedelta(minutes=hot["after_post_min"])
    musts, soon, along, ahead, hot_set, kept = [], [], [], [], set(), 0
    dues: dict[tuple[int, int], datetime] = {}
    for i, v in enumerate(eligible):
        last = max(v.post_times.values(), default=None)
        for n, post in v.post_times.items():
            at = v.captured.get(n)
            lo, hi = post - before, post + after
            floor_due = None
            if at is not None and floor is not None:
                floor_due = max(at, since or at) + timedelta(seconds=float(floor))
            if lo <= now <= hi:
                hot_set.add((i, n))
                if at is None or grain is None:
                    musts.append((now, abs(post - now), i, n))
                    dues[(i, n)] = now
                    continue
                due = max(at, lo) + timedelta(seconds=float(grain))
                due = due if floor_due is None else min(due, floor_due)
                if due >= hi:
                    continue
                must = _secs(now - at) >= float(grain) - every - _EPS
            elif (now < post or post == last) and floor_due is not None:
                # the venue's last race stays at the floor after its window: every venue is
                # captured at least every floor_s until its end
                kept += 1
                due = floor_due
                must = _secs(now - at) >= float(floor) - every - _EPS
            else:
                continue
            item = (due, abs(post - now), i, n)
            dues[(i, n)] = due
            rounds = 1 if (i, n) in hot_set else _FLOOR_ROUNDS
            if must:
                musts.append(item)
            elif due < now + (1 + rounds) * e:
                soon.append(item)
            elif due < now + (2 + rounds) * e and ((i, n) in hot_set or _not_fresh(now, at, floor)):
                along.append(item)
            if (i, n) not in hot_set and not must and _not_fresh(now, at, floor):
                ahead.append(item)
    order = [(i, n) for *_, i, n in sorted(musts) + sorted(soon)]
    return _Deadlines(order, [(i, n) for *_, i, n in sorted(along)],
                      [(i, n) for *_, i, n in sorted(ahead)], hot_set, kept, dues)  # fmt: skip


def _not_fresh(now: datetime, at: datetime, floor: float | None) -> bool:
    """Whether a race not hot is past half its floor (#387: younger, it is fresh)."""
    return floor is not None and _secs(now - at) > float(floor) / 2


# The rounds a race kept by the floor is taken ahead of its deadline (#387): a hot race is taken
# when the next round might miss it (one round); a floor race one round earlier still, so the
# races a venue's capture took together, falling due together, spread over two rounds.
_FLOOR_ROUNDS = 2

# The pace of the floor's work ahead (#387): each round may capture races not yet due, earliest
# deadline first, up to this many times its share of the floor's work (the races kept by the
# floor x per_race x every_s / floor_s), so the floor's captures spread over the rounds instead
# of falling due together and crowding out the hot races.
_FLOOR_PACE = 1.25


class _Spend:
    """The requests a round has planned, by venue, against its budget."""

    def __init__(self, budget: float, nav: int, per: int):
        self.budget, self.nav, self.per = budget, nav, per
        self.venues: set[str] = set()
        self.spent = 0.0

    def extra(self, code: str) -> int:
        """What one race more at the venue costs: navigation too for a venue not in the round."""
        return self.per if code in self.venues else self.nav + self.per

    def take(self, code: str, requests: float) -> bool:
        """Add `requests` at the venue when they fit the budget; whether they did."""
        if self.spent + requests > self.budget + _EPS:
            return False
        self.spent += requests
        self.venues.add(code)
        return True


def _work_ahead(
    plan: _Deadlines,
    eligible: list[VenueState],
    picked: dict[str, list[int] | None],
    spend: _Spend,
    every: float,
    config: Mapping[str, Any],
) -> None:
    """After the races planned first: a venue already in the round takes along its races due
    within two rounds more (its navigation paid); then races not hot and not fresh, earliest
    deadline first, up to _FLOOR_PACE times the floor's share of a round."""
    per, floor = spend.per, config["hot_races"].get("floor_s")
    for i, n in plan.along:
        code = eligible[i].code
        if code in picked and n not in picked[code] and spend.take(code, per):
            picked[code].append(n)
    if not floor:
        return
    quota = _FLOOR_PACE * plan.floor_races * per * every / float(floor)
    index = {v.code: i for i, v in enumerate(eligible)}
    done = sum(per for code, races in picked.items() for n in races or []
               if (index[code], n) not in plan.hot)  # fmt: skip
    for i, n in plan.ahead:
        code = eligible[i].code
        if code in picked and n in picked[code]:
            continue
        if done + per <= quota + _EPS and spend.take(code, spend.extra(code)):
            picked.setdefault(code, []).append(n)
            done += per


def _plan(
    now: datetime,
    venues: list[VenueState],
    config: Mapping[str, Any],
    rps: float,
    every: float | None = None,
    budget: float | None = None,
) -> tuple[list[Job], int]:
    """plan_round's jobs, and how many race captures were due at the venues it could capture.
    `every`: the round's every_s (None: the window in force at `now`); `budget`: the requests the
    round may spend (None: every x rps)."""
    every = _every_s(now, config) if every is None else every
    if every is None:
        return [], 0
    budget = every * rps if budget is None else budget
    nav, per = config["requests"]["navigation"], config["requests"]["per_race"]
    floor = timedelta(seconds=config["min_poll_interval_s"])
    eligible = [
        v
        for v in venues
        if v.stopped is None and (v.last_capture is None or now - v.last_capture >= floor)
    ]
    due = {v.code: _cycle_due(v, now, config) for v in eligible}
    plan = None
    if _grain(config):
        plan = _deadlines(now, eligible, config, every)
        first, hot = plan.first, plan.hot
    else:  # #24: every hot race, nearest post first
        ranked = sorted(
            (abs(post - now), post, i, n)
            for i, v in enumerate(eligible)
            for n, post in v.post_times.items()
            if _is_hot(post, now, config)
        )
        first = [(i, n) for *_, i, n in ranked]
        hot = set(first)
    wanted = len(  # every race of a meeting whose races are unknown counts once, as n None
        {(eligible[i].code, n) for i, n in first}
        | {
            (code, n)
            for code, races in due.items()
            for n in (races if races is not None else [None])
        }
    )

    spend = _Spend(budget, nav, per)
    picked: dict[str, list[int] | None] = {}
    for i, n in first:
        code = eligible[i].code
        if spend.take(code, spend.extra(code)):
            picked.setdefault(code, []).append(n)
    if not picked and hot & set(first):  # the budget is below one hot race: that race alone
        i, n = next(r for r in first if r in hot)
        return [Job(eligible[i].code, [n], nav + per)], wanted
    if plan is not None:
        _work_ahead(plan, eligible, picked, spend, every, config)

    def urgency(item: tuple[int, VenueState]) -> tuple:
        # most overdue first: never cycled, then by when the cycle started or fell due
        i, v = item
        if v.cycle_start is None:
            at = None
        elif _cycle_left(v):
            at = v.cycle_start
        else:
            at = v.cycle_start + timedelta(seconds=config["hot_races"]["all_every_s"])
        return (at is not None, at or now, v.code not in picked, i)

    for _, v in sorted(enumerate(eligible), key=urgency):
        races = due[v.code]
        if races is None:  # races unknown: every race in one capture, if it fits
            if spend.take(v.code, nav + per * RACES_IF_UNKNOWN):
                picked[v.code] = None
            continue
        for n in races:
            have = picked.get(v.code)
            if have is not None and n in have:
                continue
            if not spend.take(v.code, spend.extra(v.code)):
                break
            picked.setdefault(v.code, []).append(n)

    jobs = [
        Job(code, None, nav + per * RACES_IF_UNKNOWN)
        if races is None
        else Job(code, sorted(races), nav + per * len(races))
        for code, races in picked.items()
    ]
    if plan is not None:  # #387: each job's earliest deadline, for the order it runs in
        index = {v.code: i for i, v in enumerate(eligible)}
        for job in jobs:
            dues = [plan.dues[(index[job.venue], n)] for n in job.races or []
                    if (index[job.venue], n) in plan.dues]  # fmt: skip
            job.due = min(dues, default=None)
    return jobs, wanted


# ---------------------------------------------------------------- the block marker


def blocked(marker: Path, host: str, race_date: date) -> bool:
    """Whether the marker holds a line for this host and race date."""
    return bool(block_lines(marker, host, race_date))


def block_lines(
    marker: Path, host: str, race_date: date
) -> list[tuple[int | None, datetime | None]]:
    """(status, at) of every marker line for this host and race date, in order; None where the
    line has no readable value."""
    if not marker.is_file():
        return []
    out = []
    for line in marker.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields[:2] != [host, race_date.isoformat()]:
            continue
        values = dict(f.partition("=")[::2] for f in fields[2:])
        status = values.get("status", "")
        try:
            at = datetime.fromisoformat(values.get("at", ""))
        except ValueError:
            at = None
        if at is not None and at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        out.append((int(status) if status.isdigit() else None, at))
    return out


def backs_off(on_block: Mapping[str, Any]) -> bool:
    """Whether a schedule's `on_block` is #387's ladder, under which no block ends the day."""
    return bool(on_block.get("ladder")) and on_block.get("cooldown_min") is not None


def pause_end(
    marker: Path, host: str, race_date: date, on_block: Mapping[str, Any]
) -> datetime | None:
    """When the pause after the latest marker line for this host and race date ends (#387);
    None when no line has a readable time."""
    times = [at for _, at in block_lines(marker, host, race_date) if at is not None]
    if not times:
        return None
    return max(times) + timedelta(minutes=float(on_block["cooldown_min"]))


def _write_marker(marker: Path, host: str, race_date: date, exc: CircuitOpen, at: datetime) -> None:
    """Append `<host> <race date> status=<status> at=<UTC instant>`."""
    marker.parent.mkdir(parents=True, exist_ok=True)
    status = exc.http_status if exc.http_status is not None else "-"
    with open(marker, "a", encoding="utf-8") as fh:
        fh.write(f"{host} {race_date.isoformat()} status={status} at={at.isoformat()}\n")


# ---------------------------------------------------------------- the day


@dataclass
class _Venue:
    state: VenueState
    summary: VenueSummary = field(default_factory=VenueSummary)
    read_at: datetime | None = None  # round of the latest post-time read
    started_at: datetime | None = None  # when the latest capture really started (the floor)
    cycle_first: datetime | None = None  # when the cycle in progress's first capture started
    wrong_in_row: int = 0
    tier2_tried: set[tuple[int, int]] = field(default_factory=set)  # (race, grid value)
    tier2_got: set[tuple[int, int]] = field(default_factory=set)  # tried and written
    published: set[int] = field(default_factory=set)  # races whose result this run has logged
    results_polled: datetime | None = None  # when the latest results poll started
    results_ok: datetime | None = None  # when the latest successful results poll returned


def run_day(
    race_date: date,
    site: Site,
    store: Store,
    *,
    config: Mapping[str, Any],
    now: Callable[[], datetime],
    sleep: Callable[[float], None],
    block_marker: Path | str,
    venues: list[str] | None = None,
    resume: Callable[[float], Site] | None = None,
) -> DaySummary:
    """Log every venue of `race_date` (discovered when `venues` is None) from the first window, or
    from the day-before sales open with `day_before`, to its end, round by round, and return what
    was done. `resume(requests_per_second)`: the site the day goes on with after a block's cooldown,
    or at its `venue_rates` rate (#387) (None: the same site, at its own rate)."""
    day = _Day(race_date, site, store, config, now, sleep, Path(block_marker), resume)
    return day.run(venues)


# ---------------------------------------------------------------- several sources

SOURCES = ("jra", "nar")


def source_config(config: Mapping[str, Any], source: str) -> dict[str, Any]:
    """One source's schedule (the mapping run_day takes) out of logger.yaml. JRA's is the top level,
    as before #38; another source's is its own block (`nar:`) over the top level's keys, with no
    tier 2, results polling (#225), day-before window (#358) or venue_rates (#387) unless its block
    has them."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; known: {', '.join(SOURCES)}")
    out = {k: v for k, v in config.items() if k not in SOURCES and k != "sources"}
    if source == "jra":
        return out
    block = config.get(source)
    if not isinstance(block, Mapping):
        raise ValueError(f"logger.yaml has no {source!r} block")
    for key in ("tier2", "results", "day_before", "venue_rates"):
        out.pop(key, None)
    return {**out, **block}


@dataclass
class Source:
    """One source's day for run_sources: run_day's arguments but the clock."""

    site: Site
    store: Store
    config: Mapping[str, Any]
    block_marker: Path | str
    venues: list[str] | None = None
    resume: Callable[[float], Site] | None = None  # run_day's `resume`
    race_date: date | None = None  # its own race date (#358); None: run_sources' race_date


def run_sources(
    race_date: date,
    sources: Sequence[Source],
    *,
    now: Callable[[], datetime],
    sleep: Callable[[float], None],
    on_error: Callable[[str, Exception], None] | None = None,
) -> dict[str, DaySummary]:
    """Every source's run_day at once, each on a thread of its own (one source runs in the calling
    thread), keyed by its site's source, each for its own `race_date` (`race_date` for a source
    without one). Returns once every day is over. An error escaping one source's day stops that
    source only: `on_error(source, exc)` is called once, from that source's thread, as soon as
    its day fails; its summary is stopped 'error', as is each of its venues still running.
    Nothing is raised for it."""
    keys = [s.site.source for s in sources]
    if len(set(keys)) != len(keys):
        raise ValueError(f"a source given twice: {keys}")
    days = {}
    for k, s in zip(keys, sources, strict=True):
        day = s.race_date or race_date
        days[k] = _Day(day, s.site, s.store, s.config, now, sleep, Path(s.block_marker), s.resume)

    def target(key: str, s: Source) -> None:
        day = days[key]
        try:
            day.run(s.venues)
        except Exception as exc:
            log.error("%s: the day's run failed: %r", key, exc)
            day._stop_running("error")
            if on_error is not None:
                on_error(key, exc)

    if len(sources) == 1:
        target(keys[0], sources[0])
    else:
        threads = [
            threading.Thread(target=target, args=(k, s), name=f"log-day-{k}", daemon=True)
            for k, s in zip(keys, sources, strict=True)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    return {k: days[k].summary for k in keys}


class _Day:
    def __init__(self, race_date, site, store, config, now, sleep, marker: Path, resume=None):
        self.race_date = race_date
        self.site, self.store, self.config = site, store, config
        self.now, self.sleep, self.marker = now, sleep, marker
        self.resume = resume
        self.summary = DaySummary()
        self.venues: dict[str, _Venue] = {}
        self.tier2 = config.get("tier2")
        self.results = config.get("results")
        able = callable(getattr(site, "published_results", None)) and callable(
            getattr(store, "write_published", None)
        )
        if self.results and not able:  # a site or store from before #225: odds only
            log.warning("results polling off: the site or store has no results methods")
            self.results = None
        self.grid = sorted(set(self.tier2["grid_min"]), reverse=True) if self.tier2 else []
        self.first = _windows(race_date, config)[0][0]  # W, the race date's first window
        self.day_before = config.get("day_before")
        if self.day_before and not callable(getattr(site, "sales_open", None)):
            log.warning("day-before window off: the site has no sales_open")  # a site before #358
            self.day_before = None
        self.opened: datetime | None = None  # the day-before window's open (#358)
        self.switched = False  # the first round from W after a day-before window has run
        self.level = 0  # #387's back-off: 0 none, 1 the resume rate, 2 the floor only
        self.rated = False  # venue_rates has been applied (or did not apply)

    def run(self, codes: list[str] | None) -> DaySummary:
        on_block = self.config["on_block"]
        ladder = backs_off(on_block)
        if ladder:
            self._resume_marker(on_block)
        elif blocked(self.marker, self.site.host, self.race_date):
            log.warning("%s is blocked for %s: nothing sent", self.site.host, self.race_date)
            self._add(codes or [])
            self._stop_running("circuit_open")
            return self.summary
        self._add(codes or [])  # given venues: stopped like the others by a block on a probe
        cooldown = on_block["cooldown_min"]
        resumes = 0 if cooldown is None else 1
        while True:
            try:
                if self.opened is None:
                    self._probe()
                if self.opened is None:
                    self._sleep_until(self.first)
                if codes is None:
                    if self.level and self.now() >= _wall(self.config["stop"], self.race_date):
                        return self.summary  # blocked until the stop, no venue known
                    codes = list(self.site.discover())
                self._add(codes)
                self._venue_rate()
                self._rounds()
                return self.summary
            except CircuitOpen as exc:
                at = self.now()
                _write_marker(self.marker, self.site.host, self.race_date, exc, at)
                self.summary.blocks.append(exc.http_status)
                log.warning("%s blocked (HTTP %s) at %s", self.site.host, exc.http_status, at)
                if ladder:  # #387: no block ends the day
                    self.sleep(60.0 * cooldown)
                    self._back_off(self.level + 1)
                    continue
                if resumes == 0:
                    self._stop_running("circuit_open")
                    return self.summary
                resumes -= 1
                self.sleep(60.0 * cooldown)
                if self.resume is not None:
                    rate = on_block.get("resume_requests_per_second")
                    rate = self.site.requests_per_second if rate is None else float(rate)
                    log.info("%s: resuming at %g requests a second", self.site.host, rate)
                    self.site = self.resume(rate)

    # ------------------------------------------------ rates and blocks (ticket #387)

    def _venue_rate(self) -> None:
        """Once the venues are known: a day whose venue_rates entry (the greatest min_venues not
        over its venue count) gives another rate than its site's goes on on resume(rate)."""
        if self.rated or self.level or self.resume is None:
            return
        self.rated = True
        base = self.site.requests_per_second
        entries = [
            e
            for e in self.config.get("venue_rates") or []
            if int(e["min_venues"]) <= len(self.venues)
        ]
        if not entries:
            return
        rate = float(max(entries, key=lambda e: int(e["min_venues"]))["requests_per_second"])
        if abs(rate - base) > _EPS:
            log.info("%s: %d venues at %g requests a second", self.site.host, len(self.venues),
                     rate)  # fmt: skip
            self.site = self.resume(rate)

    def _back_off(self, level: int) -> None:
        """The ladder's step after `level` blocks: the resume rate, from the second the floor only
        (every race every floor_s, hot ones included)."""
        self.level = level
        rate = self.config["on_block"].get("resume_requests_per_second")
        if self.resume is not None and rate is not None:
            log.info("%s: block %d, resuming at %g requests a second%s", self.site.host, level,
                     float(rate), ", the floor only" if level >= 2 else "")  # fmt: skip
            self.site = self.resume(float(rate))

    def _resume_marker(self, on_block: Mapping[str, Any]) -> None:
        """A restart on a day with block lines: nothing until the pause after the latest is over,
        then the level the lines give."""
        lines = block_lines(self.marker, self.site.host, self.race_date)
        if not lines:
            return
        self.summary.blocks.extend(status for status, _ in lines)
        end = pause_end(self.marker, self.site.host, self.race_date, on_block)
        log.warning("%s: %d block(s) on %s, resuming after %s", self.site.host, len(lines),
                    self.race_date, end)  # fmt: skip
        if end is not None:
            self._sleep_until(end)
        self._back_off(len(lines))

    def _add(self, codes: Iterable[str]) -> None:
        for code in dict.fromkeys(codes):
            if code not in self.venues:
                self.venues[code] = _Venue(VenueState(code=code))
                self.summary.venues[code] = self.venues[code].summary
                if self.tier2:
                    self.venues[code].summary.tier2_skipped = 0

    def _stop(self, v: _Venue, reason: str) -> None:
        v.state.stopped = v.summary.stopped = reason

    def _stop_running(self, reason: str) -> None:
        for v in self.venues.values():
            if v.state.stopped is None:
                self._stop(v, reason)
        self.summary.stopped = reason

    def _sleep_until(self, t: datetime) -> None:
        wait = _secs(t - self.now())
        if wait > 0:
            self.sleep(wait)

    # ------------------------------------------------ the day before (ticket #358)

    def _probe(self) -> None:
        """The day-before probes: from probe_from on the day before (or now, when later), every
        probe_every_s while before probe_until, or once now when that is past; never from W on.
        The first that finds the sales open opens the window (`opened`). A CircuitOpen is raised
        (a block); any other failed probe is retried at the next."""
        db = self.day_before
        if not db:
            return
        eve = self.race_date - timedelta(days=1)
        until = _wall(db["probe_until"], eve)
        every = timedelta(seconds=float(db["probe_every_s"]))
        self._sleep_until(_wall(db["probe_from"], eve))
        t = self.now()
        while t < self.first:
            at = self.now()
            try:
                if self.site.sales_open():
                    self.opened = self.summary.sales_open_utc = at
                    log.info("%s: the %s sales are open at %s", self.site.host, self.race_date, at)
                    return
            except _CALL_ERRORS as exc:
                log.info("%s: odds menu probe failed, retried at the next: %s", self.site.host, exc)
            t += every
            if t >= until:
                log.info("%s: the %s sales not open by %s", self.site.host, self.race_date, until)
                return
            self._sleep_until(t)

    def _night(self, t: datetime) -> bool:
        """Whether a round starting at `t` is in the day-before window (from the open to W)."""
        return self.opened is not None and self.opened <= t < self.first

    def _every(self, t: datetime) -> float | None:
        """every_s of a round starting at `t`: the day-before window's, else the race date's
        window in force; None before both."""
        if self._night(t):
            return float(self.day_before["every_s"])
        every = _every_s(t, self.config, self.race_date)
        if self.level >= 2 and every is not None:  # #387: the floor only, each race by its
            return min(e for _, e in _windows(self.race_date, self.config))  # own deadline
        return every

    def _schedule(self, t: datetime) -> Mapping[str, Any]:
        """The schedule of a round starting at `t`: in the day-before window, every venue's full
        cycle falls due every round (all_every_s is the window's every_s); from a second block
        (#387) the floor only: hot races too at floor_s."""
        config = self.config
        if self.level >= 2:
            hot = config["hot_races"]
            floor = hot.get("floor_s") or hot["all_every_s"]
            config = {**config, "hot_races": {**hot, "every_s": floor, "floor_s": floor}}
        if not self._night(t):
            return config
        hot = {**config["hot_races"], "all_every_s": float(self.day_before["every_s"])}
        return {**config, "hot_races": hot}

    def _switch(self, t: datetime) -> None:
        """At the first round from W after a day-before window, every venue's cycle falls due at
        once: the night's cycles ran at its every_s, the race date's run at all_every_s."""
        if self.opened is None or self.switched or t < self.first:
            return
        self.switched = True
        for v in self.venues.values():
            v.state.cycle_start, v.state.cycle_left = None, None

    # ------------------------------------------------ rounds

    def _end(self, v: _Venue) -> datetime:
        if v.state.post_times:
            after = timedelta(minutes=self.config["after_last_race_min"])
            return max(v.state.post_times.values()) + after
        return _wall(self.config["stop"], self.race_date)

    def _running(self, v: _Venue, t: datetime) -> bool:
        """Whether `v` still logs at `t`; one past its end stops as 'ended'. A venue whose post
        times were never read has no end yet."""
        if v.state.stopped is None and v.read_at is not None and t >= self._end(v):
            self._stop(v, "ended")
        return v.state.stopped is None

    def _active(self, t: datetime) -> list[_Venue]:
        return [v for v in self.venues.values() if self._running(v, t)]

    def _next_round(self, start: datetime) -> datetime:
        """every_s after `start`, or the next window's start when that is sooner."""
        windows = _windows(self.race_date, self.config)
        every = self._every(start)
        later = [w for w, _ in windows if w > start][:1]
        return min(later + ([start + timedelta(seconds=every)] if every is not None else []))

    def _rounds(self) -> None:
        all_every = timedelta(seconds=self.config["hot_races"]["all_every_s"])
        while True:
            start = self.now()
            self._switch(start)
            schedule = self._schedule(start)
            active = self._active(start)
            if not active and self._results_wake() is None:
                return
            grain = _grain(schedule)
            later = []  # #387: a venue's refresh read waits for the round's captures
            for v in active:  # post times refresh as each full cycle starts
                fresh = v.read_at is None or start - v.read_at >= all_every
                if fresh and _new_cycle_due(v.state, start, schedule):
                    if grain and v.state.post_times:
                        later.append(v)
                    else:
                        self._read(v, start)
            active = self._active(self.now())
            jobs: list[Job] = []
            if active:
                rps, every = self.site.requests_per_second, self._every(start)
                budget = self._budget(start, every, rps, schedule)
                states = [v.state for v in active]
                jobs, wanted = _plan(start, states, schedule, rps, every, budget)
                self.summary.deferred += max(0, wanted - sum(
                    1 if j.races is None else len(j.races) for j in jobs
                ))  # fmt: skip
                self._run_jobs(jobs, start, schedule, grain)
            polled = self._results_round()
            read = self._late_reads(later, start, all_every)
            if self.tier2 and active:
                self._tier2_round(start, sum(job.requests for job in jobs) + polled + read)
            # while odds are logged, results polls ride on the rounds; after, they set the pace
            wake = self._next_round(start) if self._active(self.now()) else self._results_wake()
            if wake is None:
                return
            self._sleep_until(wake)

    def _run_jobs(self, jobs: list[Job], start: datetime, schedule, grain: bool) -> None:
        """The round's captures in plan order; with #387's grain, see _next_job."""
        pending = list(jobs)
        while pending:
            job = self._next_job(pending) if grain else pending[0]
            pending.remove(job)
            v = self.venues[job.venue]
            if self._running(v, self.now()):
                self._capture(v, job, start, schedule)

    def _next_job(self, pending: list[Job]) -> Job:
        """The first job, unless its venue must wait for min_poll_interval_s: then the later job
        whose venue is free soonest, before the first's, when running it still lets the first
        start by its deadline (#387), so a round idles as little as it can on such a wait."""
        now, rps = self.now(), self.site.requests_per_second
        gap = timedelta(seconds=self.config["min_poll_interval_s"])

        def ready(job: Job) -> datetime:
            at = self.venues[job.venue].started_at
            return now if at is None else max(now, at + gap)

        first = pending[0]
        if ready(first) <= now:
            return first
        fill = [
            job
            for job in pending[1:]
            if ready(job) < ready(first)
            and (first.due is None
                 or max(ready(job) + timedelta(seconds=job.requests / rps), ready(first))
                 <= first.due)
        ]  # fmt: skip
        return min(fill, key=ready, default=first)

    def _late_reads(self, later: list[_Venue], start: datetime, all_every: timedelta) -> int:
        """#387: the post-time reads that waited for the round's captures, each when it ends
        before the next round starts, or once it has waited a full cycle more; the requests
        sent (navigation each)."""
        nav, sent = self.config["requests"]["navigation"], 0
        for v in later:
            ends = self.now() + timedelta(seconds=nav / self.site.requests_per_second)
            late = v.read_at is not None and start - v.read_at >= 2 * all_every
            if v.state.stopped is None and (ends <= self._next_round(start) or late):
                self._read(v, start)
                sent += nav
        return sent

    def _budget(self, start: datetime, every: float | None, rps: float, schedule) -> float | None:
        """With #387's grain, what the round may spend on captures so that it ends on time: the
        round's time left, less the results polls due in it. None (every_s x rps) otherwise."""
        if every is None or not _grain(schedule):
            return None
        end = start + timedelta(seconds=every)
        polls = sum(
            _RESULTS_REQUESTS
            for v in self.venues.values()
            if (due := self._results_due(v, start)) is not None and due[0] <= end
        )
        return max(0.0, _secs(end - self.now())) * rps - polls

    def _read(self, v: _Venue, start: datetime) -> None:
        v.read_at = start
        try:
            times = dict(self.site.post_times(v.state.code))
        except _CALL_ERRORS as exc:
            self._failed(v, exc, self.now(), "post times")
            return
        if times:  # {} is unknown: keep what was known, and the store's sidecar
            # merged, not replaced: the reader leaves out races already off (発走済), and those
            # keep their known time; a time the read does give wins (a post moved)
            merged = {**v.state.post_times, **times}
            v.state.post_times = merged
            self.store.write_post_times(v.state.code, merged)
            self._count_tier2(v)

    def _capture(self, v: _Venue, job: Job, start: datetime, schedule: Mapping[str, Any]) -> None:
        if v.started_at is not None:  # the floor between real capture starts
            wait = self.config["min_poll_interval_s"] - _secs(self.now() - v.started_at)
            if wait > 0:
                self.sleep(wait)
        due = _cycle_due(v.state, start, schedule)
        v.state.last_capture = start
        t0 = v.started_at = self.now()
        try:
            capture = self.site.capture(v.state.code, job.races)
        except _CALL_ERRORS as exc:
            self._failed(v, exc, t0, "capture")
            return
        t1 = self.now()
        v.summary.rows += self.store.write(v.state.code, capture)
        v.summary.snapshots += 1
        self.summary.requests += capture.requests
        v.wrong_in_row = 0
        v.state.races_seen = tuple(sorted(set(v.state.races_seen) | _race_numbers(capture.rows)))
        for n in job.races if job.races is not None else _races(v.state):
            v.state.captured[n] = t0
        if due == []:  # hot races only: no cycle was due
            return
        if not _cycle_left(v.state):  # this capture starts a cycle
            v.state.cycle_start, v.cycle_first = start, t0
        done = due is None or job.races is None
        left = frozenset() if done else frozenset(due) - frozenset(job.races)
        v.state.cycle_left = left or None
        if not left:
            cycle_s = _secs(t1 - v.cycle_first)
            v.summary.longest_cycle_s = max(v.summary.longest_cycle_s, cycle_s)

    # ------------------------------------------------ results (ticket #225)

    def _results_due(self, v: _Venue, t: datetime) -> tuple[datetime, datetime] | None:
        """(when v's next results poll falls due, the last instant it may poll) at `t`; None when
        it polls no more: no `results` block or no post times, every race logged, past its
        window (last post + until_min), or its odds stopped otherwise than as 'ended' (a block,
        or answers for another meeting)."""
        posts = v.state.post_times.values()
        if not self.results or not posts or v.state.stopped not in (None, "ended"):
            return None
        if set(v.state.post_times) <= v.published:
            return None
        first = min(posts)
        until = max(posts) + timedelta(minutes=self.results["until_min"])
        due = first
        if v.results_polled is not None:
            due = max(first, v.results_polled + timedelta(seconds=self.results["every_s"]))
        return (due, until) if t <= until and due <= until else None

    def _results_wake(self) -> datetime | None:
        """The earliest instant a venue's results poll falls due; None when none polls again."""
        t = self.now()
        dues = [d[0] for v in self.venues.values() if (d := self._results_due(v, t)) is not None]
        return min(dues, default=None)

    def _results_round(self) -> int:
        """Poll every venue whose results poll is due; the requests the polls cost."""
        sent = 0
        for v in self.venues.values():
            t = self.now()
            due = self._results_due(v, t)
            if due is not None and due[0] <= t <= due[1]:
                self._poll_results(v, t)
                sent += _RESULTS_REQUESTS
        return sent

    def _poll_results(self, v: _Venue, t: datetime) -> None:
        """One results poll: the races shown and not yet logged are written at the instant the
        poll returned, bounded below by the instant the run's previous successful poll returned
        (it did not show them). On the run's first successful poll there is no such poll: the
        races are late sightings (bound None), whatever came out while nothing ran (a restart, a
        late start, a wake). A failed poll is retried at the next one, is never a bound and is no
        failure of the odds (a CircuitOpen is still a block, per `on_block`)."""
        v.results_polled = t
        try:
            shown = {int(n) for n in self.site.published_results(v.state.code)}
        except _CALL_ERRORS as exc:
            log.info("%s results poll failed, retried at the next poll: %s", v.state.code, exc)
            return
        seen = self.now()
        new = shown - v.published
        if new:
            self.store.write_published(
                v.state.code, sorted(new), seen, unpublished_utc=v.results_ok
            )
            v.published |= new
        v.results_ok = seen

    # ------------------------------------------------ tier 2

    def _grid_point(self, post: datetime, t: datetime) -> int | None:
        """The grid value whose stretch [post - g, post - next smaller g) holds `t`, or None."""
        for i, g in enumerate(self.grid):
            nxt = self.grid[i + 1] if i + 1 < len(self.grid) else 0
            if post - timedelta(minutes=g) <= t < post - timedelta(minutes=nxt):
                return g
        return None

    def _tier2_round(self, start: datetime, spent: float) -> None:
        """The grid points due at `start`, in what tier 1 left of the round's budget (`spent`: its
        planned cost); a capture that would run past the next round's start is left for later."""
        every = self._every(start)
        if every is None:
            return
        rps = self.site.requests_per_second
        left = every * rps - spent
        nav, per = self.tier2["requests"]["navigation"], self.tier2["requests"]["per_race"]
        due = []
        for v in self.venues.values():
            if v.state.stopped is not None:
                continue
            for race, post in v.state.post_times.items():
                g = self._grid_point(post, start)
                if g is not None and (race, g) not in v.tier2_tried:
                    due.append((post - timedelta(minutes=g), v.state.code, race, g))
        picked: dict[str, list[tuple[int, int]]] = {}
        for _, code, race, g in sorted(due):  # by due time: the stretch ending soonest
            cost = per if code in picked else nav + per
            if cost <= left + _EPS:
                picked.setdefault(code, []).append((race, g))
                left -= cost
        deadline = self._next_round(start)
        for code, points in picked.items():
            v, cost = self.venues[code], nav + per * len(points)
            if v.state.stopped is None and self.now() + timedelta(seconds=cost / rps) <= deadline:
                self._capture_tier2(v, points)

    def _capture_tier2(self, v: _Venue, points: list[tuple[int, int]]) -> None:
        v.tier2_tried.update(points)
        races = sorted(race for race, _ in points)
        try:
            capture = self.site.capture_tier2(v.state.code, races)
            self.store.write(v.state.code, capture)
        except CircuitOpen:
            raise
        except Exception as exc:  # parser errors included: tier 2 never stops tier 1
            log.warning("%s tier-2 capture of races %s failed: %r", v.state.code, races, exc)
            return
        v.tier2_got.update(points)
        self.summary.requests += capture.requests
        self._count_tier2(v)

    def _count_tier2(self, v: _Venue) -> None:
        if self.tier2:
            got = {(race, g) for race, g in v.tier2_got if race in v.state.post_times}
            v.summary.tier2_skipped = len(v.state.post_times) * len(self.grid) - len(got)

    def _failed(self, v: _Venue, exc: Exception, t: datetime, what: str) -> None:
        if isinstance(exc, WrongMeeting):
            v.summary.wrong_meeting += 1
            v.wrong_in_row += 1
            if v.wrong_in_row >= self.config["max_wrong_meeting"]:
                self._stop(v, "wrong_meeting")
                log.warning(
                    "%s stopped: %d answers for another meeting", v.state.code, v.wrong_in_row
                )
                return
        elif isinstance(exc, NotListed) and v.state.post_times:
            if t < min(v.state.post_times.values()):
                return  # not on the menu before its first post: retried in a later round
            v.summary.failures += 1
        else:
            v.summary.failures += 1
        log.warning("%s %s failed: %s", v.state.code, what, exc)
