"""`horsetrack japan log-day` (ticket #25): every JRA venue of a day logged from one command, then a
health line per venue and an alert for a venue that logged nothing, stopped early or left a gap.

It wires the JRA site (one ingest Fetcher, so one rate limit for www.jra.go.jp, and the capture
module) and the odds log to `schedule.run_day`, with logger.yaml as the schedule. Log records go to
`logs_dir('japan')/log-day-<date>.log`; the terminal gets a few lines. At the end one line per venue
is appended to `logs_dir('japan')/health.log` (see `health.py`), each ALERT also on stderr.

Exit codes: 0 clean, 2 with any ALERT, 1 on error (one `ERROR` line on stderr; a bad option is an
error too, so 2 always means an alert).

Seams, looked up at call time: `utc_now` (the only clock), `sleep` (the only wait) and
`make_fetcher` (the Fetcher every request goes through).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
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
    capture_meeting,
    discover_venues,
    read_post_times,
)
from horsetrack.countries.japan.logger.health import health_lines
from horsetrack.countries.japan.logger.schedule import run_day

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
    time.sleep(seconds)


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

    def discover(self) -> list[str]:
        return discover_venues(self.fetcher, self.race_date)

    def post_times(self, venue: str) -> dict[int, datetime]:
        return read_post_times(self.fetcher, self.race_date, venue)

    def capture(self, venue: str, races: list[int] | None) -> MeetingCapture:
        return capture_meeting(
            self.fetcher, self.race_date, venue, self.pools, races=races, now=self.now
        )


def _log_path(race_date: date, venue: str) -> Path:
    return paths.odds_snapshot_path(SOURCE, race_date.isoformat(), venue)


def _sidecar_path(race_date: date, venue: str) -> Path:
    return _log_path(race_date, venue).with_name(f"{venue}.races.json")


@dataclass
class OddsLogStore:
    """schedule.Store over the odds log: one gzip member per capture, post times in the sidecar."""

    race_date: date
    now: Callable[[], datetime]

    def write(self, venue: str, capture: MeetingCapture) -> int:
        return odds_log.append_capture(_log_path(self.race_date, venue), capture.rows)

    def write_post_times(self, venue: str, times: dict[int, datetime]) -> None:
        odds_log.write_post_times(_sidecar_path(self.race_date, venue), times, self.now())


def _logged(race_date: date, venue: str) -> tuple[list[dict[str, Any]], dict[str, datetime]]:
    """The venue's capture instants per race, read back from the day's log (every run that day),
    one member at a time; and its post times from the sidecar, as race_id -> post."""
    seen = {
        (row.get("race_id"), row.get("snapshot_utc"))
        for member in odds_log.iter_members(_log_path(race_date, venue))
        for row in member
    }
    rows = [{"race_id": race_id, "snapshot_utc": at} for race_id, at in seen]
    posts = odds_log.read_post_times(_sidecar_path(race_date, venue))
    prefix = f"{SOURCE}:{race_date.isoformat()}:{venue}"
    return rows, {f"{prefix}:{n:02d}": at for n, at in posts.items()}


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
    return [
        f"log-day {race_date.isoformat()}: dry run, no request sent",
        f"time zone: {TIMEZONE} (window and stop times are local)",
        f"windows: {windows}",
        f"stop: last post + {config['after_last_race_min']} min;"
        f" {config['stop']} for a venue without post times",
        f"hot races: post - {hot['within_min']} min to post + {hot['after_post_min']} min first;"
        f" a full cycle at most every {hot['all_every_s']} s",
        f"pools: {','.join(pools)}",
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


def _run(race_date: date, venues: list[str] | None, pools: list[str], config) -> int:
    """Log the day, write the health lines; the exit code (0 clean, 2 with an alert)."""
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
            site = JraSite(fetcher, race_date, pools, utc_now, host, _requests_per_second(host))
            summary = run_day(
                race_date,
                site,
                OddsLogStore(race_date, utc_now),
                config=config,
                now=utc_now,
                sleep=sleep,
                block_marker=logs / f"{SOURCE}-block.txt",
                venues=venues,
            )
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
        code = _run(day, codes, wanted, config)
    except Exception as exc:
        typer.echo(f"ERROR {' '.join(str(exc).split())[:300] or type(exc).__name__}", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    if code:
        raise typer.Exit(code)
