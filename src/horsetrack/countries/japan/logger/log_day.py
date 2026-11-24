"""`horsetrack japan log-day` (ticket #25): every JRA venue of a day logged from one command, then a
health line per venue and an alert for a venue that logged nothing, stopped early or left a gap.

It wires the JRA site (one ingest Fetcher, so one rate limit for www.jra.go.jp, and the capture
module) and the odds log to `schedule.run_day`, with logger.yaml as the schedule; its `tier2` block
adds the tier-2 pools and WIN5 (ticket #130; `--pools` picks tier 1's only), its `results` block
the result-publication polls (ticket #225), kept in the odds log's `<VENUE>.results.json`. Log
records go to `logs_dir('japan')/log-day-<date>.log`; the terminal gets a few lines. At the end one
line per venue is appended to `logs_dir('japan')/health.log` (see `health.py`), each ALERT also on
stderr.

Exit codes: 0 clean, 2 with any ALERT, 1 on error (one `ERROR` line on stderr; a bad option is an
error too, so 2 always means an alert).

Every run that sends requests, attended or not, holds a non-blocking fcntl.flock on
`logs_dir('japan')/log-day.lock` until it exits, so two log-days never share the per-host rate.
An attended log-day that finds it held sends nothing and exits 1 with one ERROR line; `--dry-run`
takes no lock.

`--unattended` (ticket #27) is how launchd starts it every 30 minutes (see `launchd.py`), with no
restart loop against the site (HKTrack's bug f: its supervisor relaunched 30 s after every exit).
The start exits 0 at once, sending nothing and waiting for nothing, when today's `stop`
(logger.yaml) has passed; when another log-day holds the lock; when the block marker
`jra-block.txt` has today's line for the JRA host; or when the no-meeting marker
`jra-no-meeting.txt` or the day-done marker `jra-day-done.txt` has a line for today. The markers
are read under the lock. Otherwise the day runs as above. Once run_day has returned, the start
appends `<race date> at=<UTC instant>` to the no-meeting marker when its one discovery found no
venue (a blocked discovery writes the block line instead), or to the day-done marker when that
discovery found a venue and every venue stopped as `ended`. A venue stopped by a block or by
answers for another meeting leaves the day open, as does a run that raised or was terminated, so
the next start resumes the day. So after a block, on a day without racing, or once the day's run
has finished, every later start that day sends nothing. Only `--unattended` writes these markers:
an attended log-day may cover other venues, pools or another date, so it never marks a day for the
agent.

Seams, looked up at call time: `utc_now` (the only clock), `sleep` (the only wait) and
`make_fetcher` (the Fetcher every request goes through). `sleep` waits on `utc_now`, in
`time.sleep` slices of at most 60 s, because time.sleep's clock stops while the Mac sleeps: the
00:30 start's wait for the first window then ends when the Mac wakes, not hours later.
"""

from __future__ import annotations

import fcntl
import logging
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
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
    MeetingCapture,
    Win5,
    capture_meeting,
    capture_tier2,
    discover_venues,
    published_results,
    read_post_times,
)
from horsetrack.countries.japan.logger.health import health_lines
from horsetrack.countries.japan.logger.schedule import blocked, run_day

log = logging.getLogger(__name__)

TIMEZONE = "Asia/Tokyo"
JST = ZoneInfo(TIMEZONE)
EXIT_ALERT = 2
EXIT_ERROR = 1


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


def make_fetcher() -> Fetcher:
    """The ingest Fetcher for JRA, with jra.yaml's per-host rates and circuit breaker."""
    http = load_country_yaml("japan", SOURCE)["http"]
    return Fetcher(
        SOURCE,
        http={
            "hosts": {"default": http["default"], **http.get("hosts", {})},
            "circuit_breaker": http["circuit_breaker"],
        },
    )


# ---------------------------------------------------------------- the site and the store


def _official_host() -> str:
    return urlparse(load_source("japan", SOURCE).hosts.official).hostname or ""


def _requests_per_second(host: str) -> float:
    http = load_source("japan", SOURCE).http
    return http.hosts.get(host, http.default).requests_per_second


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

    def published_results(self, venue: str) -> set[int]:
        return published_results(self.fetcher, self.race_date, venue)


def _log_path(race_date: date, venue: str) -> Path:
    return paths.odds_snapshot_path(SOURCE, race_date.isoformat(), venue)


def _sidecar_path(race_date: date, venue: str) -> Path:
    return _log_path(race_date, venue).with_name(f"{venue}.races.json")


def _results_path(race_date: date, venue: str) -> Path:
    return _log_path(race_date, venue).with_name(f"{venue}.results.json")


@dataclass
class OddsLogStore:
    """schedule.Store over the odds log: one gzip member per capture, post times in the sidecar,
    first-seen result publication times in the results sidecar (ticket #225)."""

    race_date: date
    now: Callable[[], datetime]

    def write(self, venue: str, capture: MeetingCapture) -> int:
        return odds_log.append_capture(_log_path(self.race_date, venue), capture.rows)

    def write_post_times(self, venue: str, times: dict[int, datetime]) -> None:
        odds_log.write_post_times(_sidecar_path(self.race_date, venue), times, self.now())

    def write_published(
        self,
        venue: str,
        races: Iterable[int],
        seen_utc: datetime,
        unpublished_utc: datetime | None = None,
    ) -> None:
        odds_log.record_published(
            _results_path(self.race_date, venue), races, seen_utc, unpublished_utc=unpublished_utc
        )


def _logged(race_date: date, venue: str) -> tuple[list[dict[str, Any]], dict[str, datetime]]:
    """The venue's capture instants per race, read back from the day's log (every run that day),
    one member at a time; and its post times from the sidecar, as race_id -> post. Tier-2 rows
    are left out: the gap metric is tier 1's."""
    seen = {
        (row.get("race_id"), row.get("snapshot_utc"))
        for member in odds_log.iter_members(_log_path(race_date, venue))
        for row in member
        if row.get("pool") in ODDS_POOLS
    }
    rows = [{"race_id": race_id, "snapshot_utc": at} for race_id, at in seen]
    posts = odds_log.read_post_times(_sidecar_path(race_date, venue))
    prefix = f"{SOURCE}:{race_date.isoformat()}:{venue}"
    return rows, {f"{prefix}:{n:02d}": at for n, at in posts.items()}


def _block_marker(logs: Path) -> Path:
    """Where schedule.run_day appends `<host> <race date> ...` when a 403/429 opens the breaker."""
    return logs / f"{SOURCE}-block.txt"


def _no_meeting_marker(logs: Path) -> Path:
    """Dated lines (`_mark`), one per day whose discovery found no venue."""
    return logs / f"{SOURCE}-no-meeting.txt"


def _day_done_marker(logs: Path) -> Path:
    """Dated lines (`_mark`), one per day whose run finished."""
    return logs / f"{SOURCE}-day-done.txt"


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


def _day(value: str | None) -> date:
    if value is None:
        return utc_now().astimezone(JST).date()
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise _Usage(f"--date {value!r} is not a YYYY-MM-DD date") from None


def _codes(value: str | None) -> list[str]:
    return list(dict.fromkeys(p.strip().upper() for p in (value or "").split(",") if p.strip()))


def _venues(value: str | None) -> list[str] | None:
    if value is None:
        return None
    codes = _codes(value)
    known = load_source("japan", SOURCE).venues
    unknown = [c for c in codes if c not in known]
    if unknown or not codes:
        raise _Usage(f"--venues {value!r}: unknown venue(s) {unknown}; known: {','.join(known)}")
    return codes


def _pools(value: str | None, config: dict[str, Any]) -> list[str]:
    codes = _codes(value) if value is not None else [str(p) for p in config["pools"]]
    unknown = [c for c in codes if c not in ODDS_POOLS]
    if unknown or not codes:
        raise _Usage(f"--pools {value!r}: unknown pool(s) {unknown}; known: {','.join(ODDS_POOLS)}")
    return codes


# ---------------------------------------------------------------- output


def _number(x: float) -> str:
    return f"{x:g}"


def _plan(race_date: date, venues: Sequence[str] | None, pools: Sequence[str], config) -> list[str]:
    host = _official_host()
    hot = config["hot_races"]
    windows = ", ".join(f"from {w['from']} every {w['every_s']} s" for w in config["windows"])
    given = ",".join(venues) if venues else "discovered from the race-card menu at the first window"
    tier2 = config.get("tier2")
    tier2_line = (
        [f"tier 2: {','.join(tier2['pools'])} at T-{',T-'.join(map(str, tier2['grid_min']))} min,"
         " from what tier 1 leaves of each round"] if tier2 else []
    )  # fmt: skip
    results = config.get("results")
    results_line = (
        [f"results: every {results['every_s']} s from first post to last post +"
         f" {results['until_min']} min, first sighting per race"] if results else []
    )  # fmt: skip
    return [
        f"log-day {race_date.isoformat()}: dry run, no request sent",
        f"time zone: {TIMEZONE} (window and stop times are local)",
        f"windows: {windows}",
        f"stop: last post + {config['after_last_race_min']} min;"
        f" {config['stop']} for a venue without post times",
        f"hot races: post - {hot['within_min']} min to post + {hot['after_post_min']} min first;"
        f" a full cycle at most every {hot['all_every_s']} s",
        f"pools: {','.join(pools)}",
        *tier2_line,
        *results_line,
        f"rate: {_number(_requests_per_second(host))} req/s to {host}"
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


def _run(
    race_date: date,
    venues: list[str] | None,
    pools: list[str],
    config,
    *,
    no_meeting: Path | None = None,
    day_done: Path | None = None,
) -> int:
    """Log the day, write the health lines; the exit code (0 clean, 2 with an alert). The caller
    holds the lock. Once run_day has returned: with `no_meeting`, a discovery that found no venue
    appends the day's line to that marker; with `day_done`, a day of discovered venues that all
    stopped as 'ended' appends the day's line to that one."""
    logs = paths.logs_dir("japan")
    day_log = logs / f"log-day-{race_date.isoformat()}.log"
    handler = _file_log(day_log)
    try:
        host = _official_host()
        given = ",".join(venues) if venues else "discover"
        typer.echo(f"log-day {race_date.isoformat()} venues={given} pools={','.join(pools)}")
        typer.echo(f"log: {day_log}")
        log.info("log-day %s venues=%s pools=%s", race_date, given, ",".join(pools))
        with make_fetcher() as fetcher:
            tier2 = config.get("tier2") or {}
            site = JraSite(fetcher, race_date, pools, utc_now, host, _requests_per_second(host),
                           tier2.get("pools", ()))  # fmt: skip
            summary = run_day(
                race_date,
                site,
                OddsLogStore(race_date, utc_now),
                config=config,
                now=utc_now,
                sleep=sleep,
                block_marker=_block_marker(logs),
                venues=venues,
            )
        returned = utc_now()
        # No venue and no stop reason: discovery answered with none (a block stops the day as
        # circuit_open, an error page raises).
        if no_meeting is not None and venues is None and not summary.venues and not summary.stopped:
            _mark(no_meeting, race_date, returned)
            typer.echo(f"no meeting on {race_date.isoformat()}: noted in {no_meeting}")
        # run_day returned with every venue ended: the day's run is over. A venue stopped by a
        # block (its own marker: clearing that line by hand resumes the day) or by answers for
        # another meeting (one bad stretch of the site) leaves the day open for a later start.
        stops = [v.stopped for v in summary.venues.values()]
        finished = bool(stops) and all(stop == "ended" for stop in stops)
        if day_done is not None and venues is None and finished:
            _mark(day_done, race_date, returned)
            typer.echo(f"day done on {race_date.isoformat()}: noted in {day_done}")
        rows_by_venue, post_times = {}, {}
        for venue in summary.venues:
            rows_by_venue[venue], posts = _logged(race_date, venue)
            post_times.update(posts)
        lines = health_lines(summary, rows_by_venue, post_times, race_date=race_date)
        health = logs / "health.log"
        with open(health, "a", encoding="utf-8") as fh:
            fh.write("".join(line + "\n" for line in lines))
        alerts = [line for line in lines if line.startswith("ALERT")]
        for line in lines:
            (log.warning if line in alerts else log.info)("%s", line)
            typer.echo(line, err=line in alerts)
        typer.echo(
            f"done: requests={summary.requests} deferred={summary.deferred}"
            f" alerts={len(alerts)} health: {health}"
        )
        return EXIT_ALERT if alerts else 0
    except Exception:
        log.exception("log-day %s failed", race_date)
        raise
    finally:
        logging.getLogger("horsetrack").removeHandler(handler)
        handler.close()


def _attended(race_date: date, venues: list[str] | None, pools: list[str], config) -> int:
    """The owner's log-day: the day under the lock; _Held (an error, exit 1) when it is taken."""
    with _lock(paths.logs_dir("japan")):
        return _run(race_date, venues, pools, config)


def _unattended(race_date: date, venues: list[str] | None, pools: list[str], config) -> int:
    """A start by launchd: 0 at once when there is nothing to do, else the day under the lock."""

    def skip(reason: str) -> int:
        typer.echo(f"log-day {race_date.isoformat()} --unattended: {reason}; nothing sent")
        return 0

    at = datetime.strptime(str(config["stop"]), "%H:%M").time()
    if utc_now() >= datetime.combine(race_date, at, tzinfo=JST):
        return skip(f"past the stop ({config['stop']} {TIMEZONE})")
    logs = paths.logs_dir("japan")
    try:
        with _lock(logs):
            # Under the lock, so a marker the previous holder wrote before letting go is seen.
            host = _official_host()
            if blocked(_block_marker(logs), host, race_date):
                return skip(f"{host} blocked today ({_block_marker(logs).name})")
            no_meeting, day_done = _no_meeting_marker(logs), _day_done_marker(logs)
            if _marked(no_meeting, race_date):
                return skip(f"no meeting today ({no_meeting.name})")
            if _marked(day_done, race_date):
                return skip(f"today's run has finished ({day_done.name})")
            return _run(race_date, venues, pools, config, no_meeting=no_meeting, day_done=day_done)
    except _Held as held:
        return skip(str(held))


# ---------------------------------------------------------------- the command


def log_day(
    race_date: str | None = typer.Option(
        None, "--date", help="Race date (YYYY-MM-DD); default: today in Asia/Tokyo."
    ),
    venues: str | None = typer.Option(
        None, "--venues", help="Venue codes, e.g. KYO,TOK; default: every venue of the day."
    ),
    pools: str | None = typer.Option(
        None, "--pools", help="Pools, e.g. WIN,QUINELLA; default: logger.yaml's."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan; send no request."),
    unattended: bool = typer.Option(
        False,
        "--unattended",
        help="As launchd starts it: exit 0 at once past the stop, while another log-day runs,"
        " or when today is blocked, has no meeting or has finished.",
    ),
) -> None:
    """Log every JRA venue of a day's odds, then append health lines to health.log."""
    try:
        config = load_country_yaml("japan", "logger")
        day = _day(race_date)
        codes = _venues(venues)
        wanted = _pools(pools, config)
        if dry_run:
            for line in _plan(day, codes, wanted, config):
                typer.echo(line)
            return
        code = (_unattended if unattended else _attended)(day, codes, wanted, config)
    except Exception as exc:
        typer.echo(f"ERROR {' '.join(str(exc).split())[:300] or type(exc).__name__}", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    if code:
        raise typer.Exit(code)
