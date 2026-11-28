"""`horsetrack japan log-day` (ticket #25): every JRA venue of a day logged from one command, then a
health line per venue and an alert for a venue that logged nothing, stopped early or left a gap.

It wires the JRA site (one ingest Fetcher, so one rate limit for www.jra.go.jp, and the capture
module) and the odds log to `schedule.run_day`, with logger.yaml as the schedule; its `tier2` block
adds the tier-2 pools and WIN5 (ticket #130; `--pools` picks tier 1's only, bracket quinella among
them since #358), its `results` block the result-publication polls (ticket #225), kept in the odds
log's `<VENUE>.results.json`, its `day_before` block the evening before the race date, from the
day-before sales open (ticket #358: the JRA site's `sales_open` reads the オッズ menu; see
`schedule`), its `cards` block each race's 出馬表 on a grid of its own (ticket #426), every capture
appended to the card log `<VENUE>.cards.jsonl.gz` beside the odds log. Log records go to
`logs_dir('japan')/log-day-<date>.log` (the earliest race date of the run's sources); the
terminal gets a few lines. At the end one line per venue is appended to
`logs_dir('japan')/health.log` (see `health.py`), each ALERT also on stderr.

The race date (#358): `--date D` is D for every source. Without it each source has its own: today
in Tokyo, or tomorrow for a source with a `day_before` block (JRA's) once today is over for it
(past today's `stop`, or today's block, no-meeting or day-done line) and the time is at or after
tomorrow's probe start (day_before.probe_from, today). So a start the evening before a race date
logs that date's night, and with `--sources jra,nar` one process can log JRA's tomorrow and NAR's
today, each with its own odds log, health lines (dated with its date) and markers.

Sources (ticket #38): `--sources jra,nar` (default: logger.yaml's `sources`) logs each source with
its own site, Fetcher (so its own per-host rate and circuit breaker), schedule
(`schedule.source_config`) and markers (`<source>-block.txt`, ...), all at once in one process
(`schedule.run_sources`). NAR's snapshots go to `odds_snapshots/nar/<date>/<VENUE>.jsonl.gz`.
`--venues` codes go to the source whose yaml lists them (jra.yaml, nar.yaml); a source none of
whose venues is named sends nothing. Every health line ends ` source=<key>`, and an ALERT reads
`ALERT <date> <venue> source=<key> <reasons>`. A block on one host stops that source only, after
its schedule's `on_block` (NAR: a 15-minute pause, then once on a fresh Fetcher at
`resume_requests_per_second`; see `schedule`). An error escaping one source's day stops that
source only: an `ERROR source=<key> ...` line on stderr at once, and the other sources log their
day and get their health lines and markers.

Exit codes: 0 clean, 2 with any ALERT, 1 on error (one `ERROR` line on stderr; a bad option is an
error too, so 2 always means an alert). A failed source exits 1 even with alerts.

Every run that sends requests, attended or not, holds a non-blocking fcntl.flock on
`logs_dir('japan')/log-day.lock` until it exits, so two log-days never share the per-host rate.
An attended log-day that finds it held sends nothing and exits 1 with one ERROR line; `--dry-run`
takes no lock.

`--unattended` (ticket #27) is how launchd starts it every 30 minutes (see `launchd.py`), with no
restart loop against the site (HKTrack's bug f: its supervisor relaunched 30 s after every exit).
The start exits 0 at once, sending nothing and waiting for nothing, when another log-day holds
the lock, or when every source has nothing to do on its race date (above): that date's `stop`
(that source's schedule) has passed, which for JRA happens only between today's stop and
tomorrow's probe start; the block marker (`jra-block.txt` for JRA) has the date's line for the
source's host; or the no-meeting marker `jra-no-meeting.txt` or the day-done marker
`jra-day-done.txt` (`nar-...` for NAR) has a line for the date. A source with nothing to do is
left out of a start that runs the others. The markers are read under the lock. Otherwise the day
runs as above. Once a source's day is over, the start appends `<race date> at=<UTC instant>` to
its no-meeting marker when its one discovery found no venue (a blocked discovery or probe writes
the block line instead), or to its day-done marker when that discovery found a venue and every
venue stopped as `ended`. A venue stopped by a block or by answers for another meeting leaves the
day open, as does a run that raised or was terminated, so the next start resumes the day (a start
in the night reads the オッズ menu once and resumes the day-before window). So after a block, on a
day without racing, or once the day's run has finished, every later start for that race date
sends nothing for that source. Only `--unattended` writes these markers: an attended log-day may
cover other venues, pools or another date, so it never marks a day for the agent.

Seams, looked up at call time: `utc_now` (the only clock), `sleep` (the only wait) and
`make_fetcher` (the Fetcher every request goes through). `sleep` waits on `utc_now`, in
`time.sleep` slices of at most 60 s, because time.sleep's clock stops while the Mac sleeps: a
night start's wait (for the next night round, or for the first window) then ends when the Mac
wakes, not hours later.
"""

from __future__ import annotations

import fcntl
import logging
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import typer

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml, load_source
from horsetrack.core.ingest.fetcher import Fetcher
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.jra.capture import (
    ODDS_POOLS,
    SOURCE,
    CardCapture,
    MeetingCapture,
    Win5,
    capture_cards,
    capture_meeting,
    capture_tier2,
    discover_venues,
    published_results,
    read_post_times,
    sales_open,
)
from horsetrack.countries.japan.logger.health import health_lines
from horsetrack.countries.japan.logger.schedule import (
    SOURCES,
    Source,
    blocked,
    run_sources,
    source_config,
)
from horsetrack.countries.japan.nar import capture as nar_capture

log = logging.getLogger(__name__)

TIMEZONE = "Asia/Tokyo"
JST = ZoneInfo(TIMEZONE)
EXIT_ALERT = 2
EXIT_ERROR = 1
# The owner's rule (#358): every race at least every 5 minutes until its T-15, at the race date's
# fine grain from there. The hot window (hot_races.within_min, 20) is what gives the fine grain.
FINE_FROM_MIN = 15


class _Usage(ValueError):
    """A bad option: one ERROR line, exit 1."""


# ---------------------------------------------------------------- seams


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sleep(seconds: float) -> None:
    """Wait `seconds` on the wall clock `utc_now()`: time.sleep's clock stops while the Mac sleeps
    (a closed lid), so a long wait sleeps in slices of at most 60 s, each up to the deadline by
    utc_now(), and ends after the slice in which the wall clock reaches it."""
    deadline = utc_now() + timedelta(seconds=seconds)
    while (left := (deadline - utc_now()).total_seconds()) > 0:
        time.sleep(min(60.0, left))


def make_fetcher(source: str = SOURCE, *, requests_per_second: float | None = None) -> Fetcher:
    """The ingest Fetcher for `source` (JRA by default), with its yaml's per-host rates and
    circuit breaker (nar.yaml's http block is already in the Fetcher's format). A
    `requests_per_second` replaces the rate of every host of that block (a resume after a block);
    jitter and the User-Agent stay."""
    http = load_country_yaml("japan", source)["http"]
    if source == SOURCE:
        http = {
            "hosts": {"default": http["default"], **http.get("hosts", {})},
            "circuit_breaker": http["circuit_breaker"],
        }
    if requests_per_second is not None:
        rate = float(requests_per_second)
        hosts = {h: {**entry, "requests_per_second": rate} for h, entry in http["hosts"].items()}
        http = {**http, "hosts": hosts}
    return Fetcher(source, http=http)


# ---------------------------------------------------------------- the site and the store


def _official_host() -> str:
    return urlparse(load_source("japan", SOURCE).hosts.official).hostname or ""


def _requests_per_second(host: str) -> float:
    http = load_source("japan", SOURCE).http
    return http.hosts.get(host, http.default).requests_per_second


def _nar_host() -> str:
    return urlparse(nar_capture.host()).hostname or ""


def _nar_requests_per_second(host: str) -> float:
    hosts = load_country_yaml("japan", nar_capture.SOURCE)["http"]["hosts"]
    return float(hosts.get(host, hosts["default"])["requests_per_second"])


def _host(source: str) -> str:
    return _official_host() if source == SOURCE else _nar_host()


def _rate(source: str, host: str) -> float:
    return _requests_per_second(host) if source == SOURCE else _nar_requests_per_second(host)


@dataclass
class JraSite:
    """schedule.Site over JRADB: every call goes through one Fetcher."""

    fetcher: Fetcher
    race_date: date
    pools: list[str]
    now: Callable[[], datetime]
    host: str
    requests_per_second: float
    tier2_pools: Sequence[str] = ()  # logger.yaml tier2.pools
    win5: Win5 = field(default_factory=Win5)  # the day's WIN5 legs, once read
    source: str = SOURCE

    def discover(self) -> list[str]:
        return discover_venues(self.fetcher, self.race_date)

    def post_times(self, venue: str) -> dict[int, datetime]:
        return read_post_times(self.fetcher, self.race_date, venue)

    def capture(self, venue: str, races: list[int] | None) -> MeetingCapture:
        return capture_meeting(
            self.fetcher, self.race_date, venue, self.pools, races=races, now=self.now
        )

    def capture_tier2(self, venue: str, races: list[int]) -> MeetingCapture:
        return capture_tier2(
            self.fetcher, self.race_date, venue, self.tier2_pools, races=races, now=self.now,
            win5=self.win5,
        )  # fmt: skip

    def capture_cards(self, venue: str, races: list[int]) -> CardCapture:
        return capture_cards(self.fetcher, self.race_date, venue, races=races, now=self.now)

    def published_results(self, venue: str) -> set[int]:
        return published_results(self.fetcher, self.race_date, venue)

    def sales_open(self) -> bool:
        return sales_open(self.fetcher, self.race_date)


@dataclass
class NarSite:
    """schedule.Site over nar.netkeiba.com: every call goes through one Fetcher. No tier 2."""

    fetcher: Fetcher
    race_date: date
    pools: list[str]
    now: Callable[[], datetime]
    host: str
    requests_per_second: float
    source: str = nar_capture.SOURCE

    def discover(self) -> list[str]:
        return nar_capture.discover_venues(self.fetcher, self.race_date)

    def post_times(self, venue: str) -> dict[int, datetime]:
        return nar_capture.read_post_times(self.fetcher, self.race_date, venue)

    def capture(self, venue: str, races: list[int] | None) -> MeetingCapture:
        return nar_capture.capture_meeting(
            self.fetcher, self.race_date, venue, self.pools, races=races, now=self.now
        )


def _log_path(race_date: date, venue: str, source: str = SOURCE) -> Path:
    return paths.odds_snapshot_path(source, race_date.isoformat(), venue)


def _sidecar_path(race_date: date, venue: str, source: str = SOURCE) -> Path:
    return _log_path(race_date, venue, source).with_name(f"{venue}.races.json")


def _results_path(race_date: date, venue: str, source: str = SOURCE) -> Path:
    return _log_path(race_date, venue, source).with_name(f"{venue}.results.json")


def _cards_path(race_date: date, venue: str, source: str = SOURCE) -> Path:
    return _log_path(race_date, venue, source).with_name(f"{venue}.cards.jsonl.gz")


@dataclass
class OddsLogStore:
    """schedule.Store over the odds log: one gzip member per capture, post times in the sidecar,
    first-seen result publication times in the results sidecar (ticket #225), race-card captures
    in the card log, one gzip member per capture (ticket #426)."""

    race_date: date
    now: Callable[[], datetime]
    source: str = SOURCE

    def write(self, venue: str, capture: MeetingCapture) -> int:
        return odds_log.append_capture(_log_path(self.race_date, venue, self.source), capture.rows)

    def write_cards(self, venue: str, capture: CardCapture) -> int:
        """Append the capture's race and runner rows, each with its `table`, as one member of the
        card log: earlier captures are never rewritten. Returns the rows written."""
        rows = [{**r, "table": "races"} for r in capture.races]
        rows += [{**r, "table": "racecard_runners"} for r in capture.runners]
        return odds_log.append_capture(_cards_path(self.race_date, venue, self.source), rows)

    def write_post_times(self, venue: str, times: dict[int, datetime]) -> None:
        sidecar = _sidecar_path(self.race_date, venue, self.source)
        odds_log.write_post_times(sidecar, times, self.now())

    def write_published(
        self,
        venue: str,
        races: Iterable[int],
        seen_utc: datetime,
        unpublished_utc: datetime | None = None,
    ) -> None:
        odds_log.record_published(
            _results_path(self.race_date, venue, self.source),
            races,
            seen_utc,
            unpublished_utc=unpublished_utc,
        )


def _logged(
    race_date: date, venue: str, source: str = SOURCE
) -> tuple[list[dict[str, Any]], dict[str, datetime]]:
    """The venue's capture instants per race, read back from the day's log (every run that day),
    one member at a time; and its post times from the sidecar, as race_id -> post. Tier-2 rows
    are left out: the gap metric is tier 1's."""
    seen = {
        (row.get("race_id"), row.get("snapshot_utc"))
        for member in odds_log.iter_members(_log_path(race_date, venue, source))
        for row in member
        if row.get("pool") in ODDS_POOLS
    }
    rows = [{"race_id": race_id, "snapshot_utc": at} for race_id, at in seen]
    posts = odds_log.read_post_times(_sidecar_path(race_date, venue, source))
    prefix = f"{source}:{race_date.isoformat()}:{venue}"
    return rows, {f"{prefix}:{n:02d}": at for n, at in posts.items()}


def _block_marker(logs: Path, source: str = SOURCE) -> Path:
    """Where schedule.run_day appends `<host> <race date> ...` when a 403/429 opens the breaker."""
    return logs / f"{source}-block.txt"


def _no_meeting_marker(logs: Path, source: str = SOURCE) -> Path:
    """Dated lines (`_mark`), one per day whose discovery found no venue."""
    return logs / f"{source}-no-meeting.txt"


def _day_done_marker(logs: Path, source: str = SOURCE) -> Path:
    """Dated lines (`_mark`), one per day whose run finished."""
    return logs / f"{source}-day-done.txt"


def _marked(marker: Path, race_date: date) -> bool:
    """Whether a dated marker has a line for `race_date`."""
    if not marker.is_file():
        return False
    lines = marker.read_text(encoding="utf-8").splitlines()
    return any(line.split()[:1] == [race_date.isoformat()] for line in lines)


def _mark(marker: Path, race_date: date, at: datetime) -> None:
    """Append `<race date> at=<UTC instant>` to a dated marker."""
    marker.parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "a", encoding="utf-8") as fh:
        fh.write(f"{race_date.isoformat()} at={at.astimezone(timezone.utc).isoformat()}\n")


class _Held(RuntimeError):
    """Another log-day holds the lock."""


@contextmanager
def _lock(logs: Path) -> Iterator[None]:
    """Hold fcntl.flock(LOCK_EX | LOCK_NB) on `logs/log-day.lock` for the block (never a blocking
    wait); _Held at once when another open file holds it. Closing the file lets go."""
    path = logs / "log-day.lock"
    logs.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise _Held(f"another log-day holds {path.name} ({path})") from None
        yield


# ---------------------------------------------------------------- options


def _day(value: str | None) -> date | None:
    """--date; None when it is not given (each source then takes its default date)."""
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise _Usage(f"--date {value!r} is not a YYYY-MM-DD date") from None


def _local(hhmm: str, day: date) -> datetime:
    """The instant of a Tokyo wall-clock "HH:MM" on `day`."""
    return datetime.combine(day, datetime.strptime(str(hhmm), "%H:%M").time(), tzinfo=JST)


def _default_date(key: str, schedule: dict[str, Any]) -> date:
    """A source's race date without --date: today in Tokyo; tomorrow for a source with a
    day-before window (#358) when today is over for it (past today's stop, or today's block,
    no-meeting or day-done line) and the time is at or after tomorrow's probe start
    (day_before.probe_from, today)."""
    now = utc_now()
    today = now.astimezone(JST).date()
    db = schedule.get("day_before")
    if not db or now < _local(db["probe_from"], today):
        return today
    over = now >= _local(schedule["stop"], today)
    if not over:
        over = _nothing_to_do(today, key, paths.logs_dir("japan")) is not None
    return today + timedelta(days=1) if over else today


def _codes(value: str | None) -> list[str]:
    return list(dict.fromkeys(p.strip().upper() for p in (value or "").split(",") if p.strip()))


def _sources(value: str | None, config: dict[str, Any]) -> list[str]:
    given = value if value is not None else ",".join(str(s) for s in config["sources"])
    keys = [k.lower() for k in _codes(given)]
    unknown = [k for k in keys if k not in SOURCES]
    if unknown or not keys:
        known = ",".join(SOURCES)
        raise _Usage(f"--sources {given!r}: unknown source(s) {unknown}; known: {known}")
    return keys


def _known_venues(source: str) -> list[str]:
    if source == SOURCE:
        return list(load_source("japan", SOURCE).venues)
    return list(load_country_yaml("japan", source)["venues"])


def _venues(value: str | None, sources: Sequence[str]) -> dict[str, list[str] | None]:
    """Source -> its venues of `--venues` (None: discovered), only for the sources that run: with
    `--venues`, a source none of whose venues is named is left out."""
    if value is None:
        return {s: None for s in sources}
    codes = _codes(value)
    known = {s: _known_venues(s) for s in sources}
    unknown = [c for c in codes if not any(c in k for k in known.values())]
    if unknown or not codes:
        listed = ",".join(c for k in known.values() for c in k)
        raise _Usage(f"--venues {value!r}: unknown venue(s) {unknown}; known: {listed}")
    out = {s: [c for c in codes if c in known[s]] for s in sources}
    return {s: v for s, v in out.items() if v}


def _pools(value: str | None, config: dict[str, Any]) -> list[str]:
    codes = _codes(value) if value is not None else [str(p) for p in config["pools"]]
    unknown = [c for c in codes if c not in ODDS_POOLS]
    if unknown or not codes:
        raise _Usage(f"--pools {value!r}: unknown pool(s) {unknown}; known: {','.join(ODDS_POOLS)}")
    return codes


@dataclass
class _SourceRun:
    """One source's part of a log-day: its venues (None: discovered), pools, schedule and race
    date."""

    key: str
    venues: list[str] | None
    pools: list[str]
    config: dict[str, Any]
    race_date: date


def _runs(
    sources: str | None,
    venues: str | None,
    pools: str | None,
    config: dict[str, Any],
    race_date: date | None,
) -> list[_SourceRun]:
    """The sources' runs; `race_date` (--date) for each, or each source's default date."""
    runs = []
    for key, codes in _venues(venues, _sources(sources, config)).items():
        schedule = source_config(config, key)
        day = race_date if race_date is not None else _default_date(key, schedule)
        runs.append(_SourceRun(key, codes, _pools(pools, schedule), schedule, day))
    return runs


def _dates(runs: Sequence[_SourceRun]) -> str:
    """The runs' race dates, distinct, in order: what a line names for the whole log-day."""
    return ",".join(dict.fromkeys(run.race_date.isoformat() for run in runs))


# ---------------------------------------------------------------- output


def _number(x: float) -> str:
    return f"{x:g}"


def _plan(run: _SourceRun) -> list[str]:
    """The dry run's lines of one source, after the run's first line."""
    venues, pools, config = run.venues, run.pools, run.config
    host = _host(run.key)
    hot = config["hot_races"]
    windows = ", ".join(f"from {w['from']} every {w['every_s']} s" for w in config["windows"])
    menu = "race-card menu" if run.key == SOURCE else "race list"
    db = config.get("day_before")
    when = "at the sales open (else the first window)" if db else "at the first window"
    given = ",".join(venues) if venues else f"discovered from the {menu} {when}"
    day_before, fine = [], ""
    if db:  # #358: the night from the day-before sales open, the fine grain from T-15
        eve = (run.race_date - timedelta(days=1)).isoformat()
        day_before = [
            f"day before: from {eve} {db['probe_from']} the odds menu every"
            f" {_number(db['probe_every_s'])} s until {db['probe_until']} (a later start before the"
            f" first window: once); from the sales open every race every"
            f" {_number(db['every_s'])} s, then the windows, until each race's T-{FINE_FROM_MIN}"
        ]
        last = max(config["windows"], key=lambda w: _local(w["from"], run.race_date))
        fine = (
            f"; from T-{FINE_FROM_MIN} (hot from T-{hot['within_min']}) each race every round,"
            f" every {_number(last['every_s'])} s from {last['from']}"
        )
    tier2 = config.get("tier2")
    tier2_line = (
        [f"tier 2: {','.join(tier2['pools'])} at T-{',T-'.join(map(str, tier2['grid_min']))} min,"
         " from what tier 1 leaves of each round"] if tier2 else []
    )  # fmt: skip
    cards = config.get("cards")
    cards_line = (
        [f"race cards: each race's card at T-{',T-'.join(map(str, cards['grid_min']))} min,"
         " after the hot races, results and tier 2, before spare full-cycle refreshes"]
        if cards else []
    )  # fmt: skip
    results = config.get("results")
    results_line = (
        [f"results: every {results['every_s']} s from first post to last post +"
         f" {results['until_min']} min, first sighting per race"] if results else []
    )  # fmt: skip
    return [
        f"time zone: {TIMEZONE} (window and stop times are local)",
        *day_before,
        f"windows: {windows}",
        f"stop: last post + {config['after_last_race_min']} min;"
        f" {config['stop']} for a venue without post times",
        f"hot races: post - {hot['within_min']} min to post + {hot['after_post_min']} min first;"
        f" a full cycle at most every {hot['all_every_s']} s{fine}",
        f"pools: {','.join(pools)}",
        *tier2_line,
        *cards_line,
        *results_line,
        f"rate: {_number(_rate(run.key, host))} req/s to {host}"
        " (a round's budget is every_s x the rate)",
        f"venues: {given}",
    ]


def _file_log(path: Path) -> logging.Handler:
    """Send the package's log records to `path` (UTC timestamps); the terminal stays quiet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(name)s %(message)s")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    package = logging.getLogger("horsetrack")
    package.addHandler(handler)
    package.setLevel(logging.INFO)
    return handler


def _site(run: _SourceRun, fetcher: Fetcher, rate: float | None = None) -> JraSite | NarSite:
    """The source's site over `fetcher`; `rate`: the fetcher's rate when it was given one."""
    host = _host(run.key)
    rps = _rate(run.key, host) if rate is None else rate
    if run.key == SOURCE:
        tier2 = run.config.get("tier2") or {}
        return JraSite(fetcher, run.race_date, run.pools, utc_now, host, rps,
                       tier2.get("pools", ()))  # fmt: skip
    return NarSite(fetcher, run.race_date, run.pools, utc_now, host, rps)


def _resume(run: _SourceRun, stack: ExitStack) -> Callable[[float], JraSite | NarSite] | None:
    """schedule.Source.resume for a source whose on_block names a resume rate (NAR's): a site over
    a fresh Fetcher at that rate, closed with the others. None otherwise (the same site resumes)."""
    if run.config["on_block"].get("resume_requests_per_second") is None:
        return None

    def resume(rate: float) -> JraSite | NarSite:
        fetcher = stack.enter_context(make_fetcher(run.key, requests_per_second=rate))
        return _site(run, fetcher, rate)

    return resume


def _run(runs: Sequence[_SourceRun], *, markers: bool = False) -> int:
    """Log the day of every source of `runs` at once, each its own race date, write the health
    lines; the exit code (0 clean, 2 with an alert). The caller holds the lock. With `markers`,
    once the sources' days are over, for each source whose venues were discovered: a discovery
    that found no venue appends its race date's line to its no-meeting marker, and venues that all
    stopped as 'ended' append it to its day-done marker. The log records go to the log of the
    earliest race date."""
    logs = paths.logs_dir("japan")
    first = min(run.race_date for run in runs)
    day_log = logs / f"log-day-{first.isoformat()}.log"
    handler = _file_log(day_log)
    try:
        for run in runs:
            given = ",".join(run.venues) if run.venues else "discover"
            typer.echo(f"log-day {run.race_date.isoformat()} venues={given}"
                       f" pools={','.join(run.pools)} source={run.key}")  # fmt: skip
            log.info("log-day %s source=%s venues=%s pools=%s", run.race_date, run.key, given,
                     ",".join(run.pools))  # fmt: skip
        typer.echo(f"log: {day_log}")
        with ExitStack() as stack:
            sources = []
            for run in runs:
                fetcher = stack.enter_context(make_fetcher(run.key))
                sources.append(
                    Source(
                        site=_site(run, fetcher),
                        store=OddsLogStore(run.race_date, utc_now, run.key),
                        config=run.config,
                        block_marker=_block_marker(logs, run.key),
                        venues=run.venues,
                        resume=_resume(run, stack),
                        race_date=run.race_date,
                    )
                )
            failed: list[str] = []
            dates = {run.key: run.race_date for run in runs}

            def on_error(key: str, exc: Exception) -> None:
                # from the failed source's thread, while the others go on logging
                failed.append(key)
                log.error("log-day %s source=%s failed", dates[key], key, exc_info=exc)
                text = " ".join(str(exc).split())[:300] or type(exc).__name__
                typer.echo(f"ERROR source={key} {type(exc).__name__}: {text}", err=True)

            summaries = run_sources(first, sources, now=utc_now, sleep=sleep,
                                    on_error=on_error)  # fmt: skip
        returned = utc_now()
        lines: list[str] = []
        for run in runs:
            summary, day = summaries[run.key], run.race_date
            if markers and run.venues is None and run.key not in failed:
                _mark_day(day, run.key, summary, logs, returned)
            rows_by_venue, post_times = {}, {}
            for venue in summary.venues:
                rows_by_venue[venue], posts = _logged(day, venue, run.key)
                post_times.update(posts)
            lines += health_lines(summary, rows_by_venue, post_times, race_date=day, source=run.key)
        health = logs / "health.log"
        with open(health, "a", encoding="utf-8") as fh:
            fh.write("".join(line + "\n" for line in lines))
        alerts = [line for line in lines if line.startswith("ALERT")]
        for line in lines:
            (log.warning if line in alerts else log.info)("%s", line)
            typer.echo(line, err=line in alerts)
        requests = sum(s.requests for s in summaries.values())
        deferred = sum(s.deferred for s in summaries.values())
        typer.echo(
            f"done: requests={requests} deferred={deferred} alerts={len(alerts)} health: {health}"
        )
        if failed:
            return EXIT_ERROR
        return EXIT_ALERT if alerts else 0
    except Exception:
        log.exception("log-day %s failed", _dates(runs))
        raise
    finally:
        logging.getLogger("horsetrack").removeHandler(handler)
        handler.close()


def _mark_day(race_date: date, source: str, summary, logs: Path, returned: datetime) -> None:
    """The unattended markers of one source whose venues were discovered, once its day is over."""
    # No venue and no stop reason: discovery answered with none (a block stops the day as
    # circuit_open, an error page raises).
    if not summary.venues and not summary.stopped:
        no_meeting = _no_meeting_marker(logs, source)
        _mark(no_meeting, race_date, returned)
        typer.echo(f"no meeting on {race_date.isoformat()}: noted in {no_meeting}")
    # run_day returned with every venue ended: the day's run is over. A venue stopped by a
    # block (its own marker: clearing that line by hand resumes the day) or by answers for
    # another meeting (one bad stretch of the site) leaves the day open for a later start.
    stops = [v.stopped for v in summary.venues.values()]
    if stops and all(stop == "ended" for stop in stops):
        day_done = _day_done_marker(logs, source)
        _mark(day_done, race_date, returned)
        typer.echo(f"day done on {race_date.isoformat()}: noted in {day_done}")


def _attended(runs: Sequence[_SourceRun]) -> int:
    """The owner's log-day: the day under the lock; _Held (an error, exit 1) when it is taken."""
    with _lock(paths.logs_dir("japan")):
        return _run(runs)


def _nothing_to_do(race_date: date, source: str, logs: Path) -> str | None:
    """Why an unattended start sends nothing for this source on its race date, read under the
    lock; None when it has a day to log."""
    host = _host(source)
    if blocked(_block_marker(logs, source), host, race_date):
        return f"{host} blocked on {race_date} ({_block_marker(logs, source).name})"
    no_meeting, day_done = _no_meeting_marker(logs, source), _day_done_marker(logs, source)
    if _marked(no_meeting, race_date):
        return f"no meeting on {race_date} ({no_meeting.name})"
    if _marked(day_done, race_date):
        return f"the run of {race_date} has finished ({day_done.name})"
    return None


def _unattended(runs: Sequence[_SourceRun]) -> int:
    """A start by launchd: 0 at once when no source has anything to do, else the day of those that
    have, each on its own race date, under the lock."""

    def skip(reasons: list[str]) -> int:
        typer.echo(f"log-day {_dates(runs)} --unattended: {'; '.join(reasons)}; nothing sent")
        return 0

    reasons, pending = [], []
    for run in runs:
        stop = str(run.config["stop"])
        if utc_now() >= _local(stop, run.race_date):
            reasons.append(f"{run.key} past the stop ({stop} {TIMEZONE})")
        else:
            pending.append(run)
    if not pending:
        return skip(reasons)
    logs = paths.logs_dir("japan")
    try:
        with _lock(logs):
            # Under the lock, so a marker the previous holder wrote before letting go is seen.
            todo = []
            for run in pending:
                reason = _nothing_to_do(run.race_date, run.key, logs)
                if reason is None:
                    todo.append(run)
                else:
                    reasons.append(f"{run.key} {reason}")
            if not todo:
                return skip(reasons)
            return _run(todo, markers=True)
    except _Held as held:
        return skip([str(held)])


# ---------------------------------------------------------------- the command


def log_day(
    race_date: str | None = typer.Option(
        None,
        "--date",
        help="Race date (YYYY-MM-DD); default: today in Asia/Tokyo, or tomorrow from the evening"
        " before (logger.yaml day_before.probe_from) once today is over.",
    ),
    venues: str | None = typer.Option(
        None,
        "--venues",
        help="Venue codes of any source, e.g. KYO,TOK,URA; default: every venue of the day.",
    ),
    sources: str | None = typer.Option(
        None, "--sources", help="Sources, e.g. jra,nar; default: logger.yaml's."
    ),
    pools: str | None = typer.Option(
        None, "--pools", help="Pools, e.g. WIN,QUINELLA; default: logger.yaml's."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan; send no request."),
    unattended: bool = typer.Option(
        False,
        "--unattended",
        help="As launchd starts it: exit 0 at once past the stop, while another log-day runs,"
        " or when the race date is blocked, has no meeting or has finished.",
    ),
) -> None:
    """Log every venue of a day's odds (JRA, NAR or both), then append health lines."""
    try:
        config = load_country_yaml("japan", "logger")
        runs = _runs(sources, venues, pools, config, _day(race_date))
        if dry_run:
            typer.echo(f"log-day {_dates(runs)}: dry run, no request sent")
            for run in runs:
                if len(runs) > 1:
                    typer.echo(f"source {run.key}: {run.race_date.isoformat()}")
                for line in _plan(run):
                    typer.echo(line)
            return
        code = (_unattended if unattended else _attended)(runs)
    except Exception as exc:
        typer.echo(f"ERROR {' '.join(str(exc).split())[:300] or type(exc).__name__}", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    if code:
        raise typer.Exit(code)
