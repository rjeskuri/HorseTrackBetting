"""Scheduling every JRA venue's odds captures under one request budget (ticket #24).

Ported from HKTrack's `ingest/live.py` (`poll_odds`, `_poll_loop`, `hot_races`) and
`scripts/oddsloop.sh`, which ran one loop per venue. Changes, fixing:

(f) after a 403/429 opened the breaker, the supervisor retried about every 30 s until the stop
    time. Here any CircuitOpen stops every venue on the host and appends a block marker line
    (host, Tokyo race date); a run that finds today's line for its host sends nothing.
(g) a full capture is navigation + per_race x races requests, and nothing budgeted them across
    venues. Here every round has a budget, the window's every_s x the host's requests per second:
    hot races first, nearest post first, then the venues' full cycles race by race.

Tier 2 (ticket #130, logger.yaml `tier2`): each race's tier-2 pools once per grid point. Grid
point g of a race falls due at post - g and belongs to the stretch until the next smaller grid
value (the post after the last). Each round runs tier 1 first, exactly as without tier 2, then
spends only what is left of its budget on the grid points due, the stretch ending soonest first;
one that does not fit is tried again in a later round of its stretch, and one with no successful
capture when its stretch ends counts in VenueSummary.tier2_skipped. A failed tier-2 capture is
logged and never touches tier 1 (a CircuitOpen still ends the day: the host blocked us).

Race cards (ticket #426, logger.yaml `cards`, JRA only): each race's 出馬表 once per grid point of
`cards.grid_min`, stretches as tier 2's (the day-before window's rounds included), a venue's due
races in one `capture_cards` call written with `write_cards`. They spend only what is left of a
round's budget after tier 1, the results polls and tier 2, so the odds captures are the same with
the block as without. A grid point is tried once; a failed card capture is logged and is no
venue's failure (a CircuitOpen still ends the day). A site without `capture_cards` or a store
without `write_cards` (one from before #426): no cards.

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

The block policy is the schedule's `on_block`. Every block appends a marker line, and a run that
finds one for its host and date sends nothing. `cooldown_min: null` stops the day at the first
block; N pauses that source N minutes, then resumes once: on `resume(rate)`'s site when run_day is
given `resume` (NAR: a fresh Fetcher at `resume_requests_per_second`), else on the same site. A
second block stops the day.

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
from horsetrack.countries.japan.jra.capture import (
    CardCapture,
    MeetingCapture,
    NotListed,
    WrongMeeting,
)

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

    def capture_cards(self, venue: str, races: list[int]) -> CardCapture: ...  # with `cards`


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

    def write_cards(self, venue: str, capture: CardCapture) -> int: ...  # with `cards`: rows


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


@dataclass
class Job:
    venue: str
    races: list[int] | None  # None: every race, for a venue whose races are not known yet
    requests: int  # navigation + per_race x races


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
    return _races(state) or None


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


def _plan(
    now: datetime,
    venues: list[VenueState],
    config: Mapping[str, Any],
    rps: float,
    every: float | None = None,
) -> tuple[list[Job], int]:
    """plan_round's jobs, and how many race captures were due at the venues it could capture.
    `every`: the round's every_s (None: the window in force at `now`)."""
    every = _every_s(now, config) if every is None else every
    if every is None:
        return [], 0
    budget = every * rps
    nav, per = config["requests"]["navigation"], config["requests"]["per_race"]
    floor = timedelta(seconds=config["min_poll_interval_s"])
    eligible = [
        v
        for v in venues
        if v.stopped is None and (v.last_capture is None or now - v.last_capture >= floor)
    ]
    due = {v.code: _cycle_due(v, now, config) for v in eligible}
    hot = sorted(
        (abs(post - now), post, i, n)
        for i, v in enumerate(eligible)
        for n, post in v.post_times.items()
        if _is_hot(post, now, config)
    )
    wanted = len(  # every race of a meeting whose races are unknown counts once, as n None
        {(eligible[i].code, n) for *_, i, n in hot}
        | {
            (code, n)
            for code, races in due.items()
            for n in (races if races is not None else [None])
        }
    )

    picked: dict[str, list[int] | None] = {}
    spent = 0.0
    for *_, i, n in hot:
        code = eligible[i].code
        cost = per if code in picked else nav + per
        if spent + cost <= budget + _EPS:
            picked.setdefault(code, []).append(n)
            spent += cost
    if hot and not picked:  # the budget is below one hot race: that race alone
        *_, i, n = hot[0]
        return [Job(eligible[i].code, [n], nav + per)], wanted

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
            cost = nav + per * RACES_IF_UNKNOWN
            if spent + cost <= budget + _EPS:
                picked[v.code] = None
                spent += cost
            continue
        for n in races:
            have = picked.get(v.code)
            if have is not None and n in have:
                continue
            cost = per if v.code in picked else nav + per
            if spent + cost > budget + _EPS:
                break
            picked.setdefault(v.code, []).append(n)
            spent += cost

    jobs = [
        Job(code, None, nav + per * RACES_IF_UNKNOWN)
        if races is None
        else Job(code, sorted(races), nav + per * len(races))
        for code, races in picked.items()
    ]
    return jobs, wanted


# ---------------------------------------------------------------- the block marker


def blocked(marker: Path, host: str, race_date: date) -> bool:
    """Whether the marker holds a line for this host and race date."""
    if not marker.is_file():
        return False
    wanted = [host, race_date.isoformat()]
    lines = marker.read_text(encoding="utf-8").splitlines()
    return any(line.split()[:2] == wanted for line in lines)


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
    cards_tried: set[tuple[int, int]] = field(default_factory=set)  # (race, card grid value)
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
    was done. `resume(requests_per_second)`: the site the day goes on with after a block's cooldown
    (None: the same site)."""
    day = _Day(race_date, site, store, config, now, sleep, Path(block_marker), resume)
    return day.run(venues)


# ---------------------------------------------------------------- several sources

SOURCES = ("jra", "nar")


def source_config(config: Mapping[str, Any], source: str) -> dict[str, Any]:
    """One source's schedule (the mapping run_day takes) out of logger.yaml. JRA's is the top level,
    as before #38; another source's is its own block (`nar:`) over the top level's keys, with no
    tier 2, results polling (#225) or day-before window (#358) unless its block has them."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; known: {', '.join(SOURCES)}")
    out = {k: v for k, v in config.items() if k not in SOURCES and k != "sources"}
    if source == "jra":
        return out
    block = config.get(source)
    if not isinstance(block, Mapping):
        raise ValueError(f"logger.yaml has no {source!r} block")
    for key in ("tier2", "results", "day_before", "cards"):
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
        self.cards = config.get("cards")
        able = callable(getattr(site, "capture_cards", None)) and callable(
            getattr(store, "write_cards", None)
        )
        if self.cards and not able:  # a site or store from before #426: no cards
            log.warning("race cards off: the site or store has no card methods")
            self.cards = None
        self.card_grid = sorted(set(self.cards["grid_min"]), reverse=True) if self.cards else []
        self.first = _windows(race_date, config)[0][0]  # W, the race date's first window
        self.day_before = config.get("day_before")
        if self.day_before and not callable(getattr(site, "sales_open", None)):
            log.warning("day-before window off: the site has no sales_open")  # a site before #358
            self.day_before = None
        self.opened: datetime | None = None  # the day-before window's open (#358)
        self.switched = False  # the first round from W after a day-before window has run

    def run(self, codes: list[str] | None) -> DaySummary:
        if blocked(self.marker, self.site.host, self.race_date):
            log.warning("%s is blocked for %s: nothing sent", self.site.host, self.race_date)
            self._add(codes or [])
            self._stop_running("circuit_open")
            return self.summary
        self._add(codes or [])  # given venues: stopped like the others by a block on a probe
        on_block = self.config["on_block"]
        cooldown = on_block["cooldown_min"]
        resumes = 0 if cooldown is None else 1
        while True:
            try:
                if self.opened is None:
                    self._probe()
                if self.opened is None:
                    self._sleep_until(self.first)
                if codes is None:
                    codes = list(self.site.discover())
                self._add(codes)
                self._rounds()
                return self.summary
            except CircuitOpen as exc:
                at = self.now()
                _write_marker(self.marker, self.site.host, self.race_date, exc, at)
                log.warning("%s blocked (HTTP %s) at %s", self.site.host, exc.http_status, at)
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
        return _every_s(t, self.config, self.race_date)

    def _schedule(self, t: datetime) -> Mapping[str, Any]:
        """The schedule of a round starting at `t`: in the day-before window, every venue's full
        cycle falls due every round (all_every_s is the window's every_s)."""
        if not self._night(t):
            return self.config
        hot = {**self.config["hot_races"], "all_every_s": float(self.day_before["every_s"])}
        return {**self.config, "hot_races": hot}

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
            for v in active:  # post times refresh as each full cycle starts
                fresh = v.read_at is None or start - v.read_at >= all_every
                if fresh and _new_cycle_due(v.state, start, schedule):
                    self._read(v, start)
            active = self._active(self.now())
            jobs: list[Job] = []
            if active:
                rps, every = self.site.requests_per_second, self._every(start)
                jobs, wanted = _plan(start, [v.state for v in active], schedule, rps, every)
                self.summary.deferred += wanted - sum(
                    1 if j.races is None else len(j.races) for j in jobs
                )
                for job in jobs:
                    v = self.venues[job.venue]
                    if self._running(v, self.now()):
                        self._capture(v, job, start, schedule)
            spent = sum(job.requests for job in jobs) + self._results_round()
            if self.tier2 and active:
                spent += self._tier2_round(start, spent)
            if self.cards and active:
                self._cards_round(start, spent)
            # while odds are logged, results polls ride on the rounds; after, they set the pace
            wake = self._next_round(start) if self._active(self.now()) else self._results_wake()
            if wake is None:
                return
            self._sleep_until(wake)

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

    def _grid_point(self, post: datetime, t: datetime, grid: Sequence[int]) -> int | None:
        """The value of `grid` (descending) whose stretch [post - g, post - next smaller g) holds
        `t`, or None."""
        for i, g in enumerate(grid):
            nxt = grid[i + 1] if i + 1 < len(grid) else 0
            if post - timedelta(minutes=g) <= t < post - timedelta(minutes=nxt):
                return g
        return None

    def _tier2_round(self, start: datetime, spent: float) -> float:
        """The tier-2 grid points due at `start`, in what tier 1 left of the round's budget
        (`spent`: its planned cost); the planned cost of the captures run."""
        tried = {code: v.tier2_tried for code, v in self.venues.items()}
        return self._grid_round(
            start, spent, self.grid, tried, self.tier2["requests"], self._capture_tier2
        )

    def _grid_round(
        self,
        start: datetime,
        spent: float,
        grid: Sequence[int],
        tried: Mapping[str, set[tuple[int, int]]],
        requests: Mapping[str, Any],
        capture: Callable[[_Venue, list[tuple[int, int]]], None],
    ) -> float:
        """The points of `grid` due at `start` and not `tried`, in what is left of the round's
        budget after `spent`, a venue's in one `capture(v, points)`; a capture that would run past
        the next round's start is left for later. Returns the planned cost of the captures run."""
        every = self._every(start)
        if every is None:
            return 0.0
        rps = self.site.requests_per_second
        left = every * rps - spent
        nav, per = requests["navigation"], requests["per_race"]
        due = []
        for v in self.venues.values():
            if v.state.stopped is not None:
                continue
            for race, post in v.state.post_times.items():
                g = self._grid_point(post, start, grid)
                if g is not None and (race, g) not in tried[v.state.code]:
                    due.append((post - timedelta(minutes=g), v.state.code, race, g))
        picked: dict[str, list[tuple[int, int]]] = {}
        for _, code, race, g in sorted(due):  # by due time: the stretch ending soonest
            cost = per if code in picked else nav + per
            if cost <= left + _EPS:
                picked.setdefault(code, []).append((race, g))
                left -= cost
        deadline = self._next_round(start)
        sent = 0.0
        for code, points in picked.items():
            v, cost = self.venues[code], nav + per * len(points)
            if v.state.stopped is None and self.now() + timedelta(seconds=cost / rps) <= deadline:
                capture(v, points)
                sent += cost
        return sent

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

    # ------------------------------------------------ race cards (ticket #426)

    def _cards_round(self, start: datetime, spent: float) -> None:
        """The card grid points due at `start`, in what the odds left of the round's budget
        (`spent`: tier 1's, the results polls' and tier 2's)."""
        tried = {code: v.cards_tried for code, v in self.venues.items()}
        self._grid_round(
            start, spent, self.card_grid, tried, self.cards["requests"], self._capture_cards
        )

    def _capture_cards(self, v: _Venue, points: list[tuple[int, int]]) -> None:
        v.cards_tried.update(points)
        races = sorted(race for race, _ in points)
        try:
            capture = self.site.capture_cards(v.state.code, races)
            self.store.write_cards(v.state.code, capture)
        except CircuitOpen:
            raise
        except Exception as exc:  # parser errors included: cards never touch the odds
            log.warning("%s card capture of races %s failed: %r", v.state.code, races, exc)
            return
        self.summary.requests += capture.requests

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
