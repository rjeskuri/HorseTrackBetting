"""Scheduling every JRA venue's odds captures under one request budget (ticket #24).

Ported from HKTrack's `ingest/live.py` (`poll_odds`, `_poll_loop`, `hot_races`) and
`scripts/oddsloop.sh`, which ran one loop per venue. Changes, fixing:

(f) after a 403/429 opened the breaker, the supervisor retried about every 30 s until the stop
    time. Here any CircuitOpen stops every venue on the host and appends a block marker line
    (host, Tokyo race date); a run that finds today's line for its host sends nothing.
(g) a full capture is navigation + per_race x races requests, and nothing budgeted them across
    venues. Here every round has a budget, the window's every_s x the host's requests per second:
    hot races first, nearest post first, then the venues' full cycles race by race.

`plan_round` is pure. `run_day` drives it with an injected clock (`now`, `sleep`) and reaches the
site and the store only through the protocols below. `config` is a mapping shaped like
`countries/japan/config/logger.yaml`. Window and stop times are Asia/Tokyo wall-clock times; every
instant is an aware datetime, and post times are UTC.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
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


class Site(Protocol):
    host: str
    requests_per_second: float

    def discover(self) -> list[str]: ...

    def post_times(self, venue: str) -> dict[int, datetime]: ...  # {} = unknown

    def capture(self, venue: str, races: list[int] | None) -> MeetingCapture: ...


class Store(Protocol):
    def write(self, venue: str, capture: MeetingCapture) -> int: ...  # rows written

    def write_post_times(self, venue: str, times: dict[int, datetime]) -> None: ...


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
    stopped: str | None = None  # 'ended', 'wrong_meeting' or 'circuit_open'
    longest_cycle_s: float = 0.0  # longest full cycle: first capture start to last capture end


@dataclass
class DaySummary:
    venues: dict[str, VenueSummary] = field(default_factory=dict)
    requests: int = 0  # requests the successful captures sent
    deferred: int = 0  # race captures a round left out for budget
    stopped: str | None = None  # 'circuit_open' when a block ended the day


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
    now: datetime, venues: list[VenueState], config: Mapping[str, Any], rps: float
) -> tuple[list[Job], int]:
    """plan_round's jobs, and how many race captures were due at the venues it could capture."""
    every = _every_s(now, config)
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


def _blocked(marker: Path, host: str, race_date: date) -> bool:
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
) -> DaySummary:
    """Log every venue of `race_date` (discovered when `venues` is None) from the first window to
    its end, round by round, and return what was done."""
    return _Day(race_date, site, store, config, now, sleep, Path(block_marker)).run(venues)


class _Day:
    def __init__(self, race_date, site, store, config, now, sleep, marker: Path):
        self.race_date = race_date
        self.site, self.store, self.config = site, store, config
        self.now, self.sleep, self.marker = now, sleep, marker
        self.summary = DaySummary()
        self.venues: dict[str, _Venue] = {}

    def run(self, codes: list[str] | None) -> DaySummary:
        if _blocked(self.marker, self.site.host, self.race_date):
            log.warning("%s is blocked for %s: nothing sent", self.site.host, self.race_date)
            self._add(codes or [])
            self._stop_running("circuit_open")
            return self.summary
        first = _windows(self.race_date, self.config)[0][0]
        self._sleep_until(first)
        cooldown = self.config["on_block"]["cooldown_min"]
        resumes = 0 if cooldown is None else 1
        while True:
            try:
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

    def _add(self, codes: Iterable[str]) -> None:
        for code in dict.fromkeys(codes):
            if code not in self.venues:
                self.venues[code] = _Venue(VenueState(code=code))
                self.summary.venues[code] = self.venues[code].summary

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
        every = _every_s(start, self.config, self.race_date)
        later = [w for w, _ in windows if w > start][:1]
        return min(later + ([start + timedelta(seconds=every)] if every is not None else []))

    def _rounds(self) -> None:
        all_every = timedelta(seconds=self.config["hot_races"]["all_every_s"])
        while True:
            start = self.now()
            active = self._active(start)
            if not active:
                return
            for v in active:  # post times refresh as each full cycle starts
                fresh = v.read_at is None or start - v.read_at >= all_every
                if fresh and _new_cycle_due(v.state, start, self.config):
                    self._read(v, start)
            active = self._active(self.now())
            if not active:
                return
            rps = self.site.requests_per_second
            jobs, wanted = _plan(start, [v.state for v in active], self.config, rps)
            self.summary.deferred += wanted - sum(
                1 if j.races is None else len(j.races) for j in jobs
            )
            for job in jobs:
                v = self.venues[job.venue]
                if self._running(v, self.now()):
                    self._capture(v, job, start)
            self._sleep_until(self._next_round(start))

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

    def _capture(self, v: _Venue, job: Job, start: datetime) -> None:
        if v.started_at is not None:  # the floor between real capture starts
            wait = self.config["min_poll_interval_s"] - _secs(self.now() - v.started_at)
            if wait > 0:
                self.sleep(wait)
        due = _cycle_due(v.state, start, self.config)
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
