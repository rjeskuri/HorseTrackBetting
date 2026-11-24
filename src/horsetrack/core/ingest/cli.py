"""`horsetrack scrape ...`: plan, run, status, expand, retry-failed. Ported from HKTrack.

Everything a command needs about a source comes from its config (`source_config`): the rates per
host, the circuit breaker and the quiet hours from `http`, "today" from `timezone`, plan's first
day from `backfill.start_date`. Output is a few lines; progress and failures go to
logs/<source>/scrape-*.log under the data directory. Exit codes for `run`: 0 clean stop (budget,
empty queue, signal, quiet hours), 2 circuit breaker opened, 1 anything else that went wrong.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping
from datetime import date, datetime, time, timezone
from typing import Annotated, Any, NoReturn
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import typer

from horsetrack.core.config import load_country_yaml
from horsetrack.core.ingest import adapters, runner
from horsetrack.core.ingest.adapters import SourceAdapter
from horsetrack.core.ingest.queue import STATUSES, FetchQueue
from horsetrack.core.paths import logs_dir, manifest_path

scrape_app = typer.Typer(help="Historical backfill.", no_args_is_help=True, add_completion=False)

EXIT_ERROR = 1
EXIT_CIRCUIT_OPEN = 2
# `retry-failed` runs unattended between restarts; failed tasks are the newest, so they run first.
# Without a limit, five pages that always fail would open the breaker at every restart, all night.
DEFAULT_RETRY_MAX_ATTEMPTS = 5
# keeps `status` within 20 lines: header + rows + total + up to 4 summary lines
_MAX_STATUS_ROWS = 14
# quiet_hours day names, in datetime.weekday() order
_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
# source -> the country folder holding its config/<source>.yaml
SOURCE_COUNTRIES = {"jra": "japan", "hkjc": "hongkong"}

SourceOpt = Annotated[
    str | None,
    typer.Option("--source", "-s", help="Source slug; default: the only one registered."),
]
PageTypesOpt = Annotated[
    str | None,
    typer.Option("--page-types", help="Comma-separated page types, e.g. fixture,results."),
]
DateOpt = Annotated[datetime | None, typer.Option(formats=["%Y-%m-%d"])]


def source_config(source: str) -> Mapping[str, Any]:
    """The source's config as a plain mapping (shared and cached: do not mutate)."""
    country = SOURCE_COUNTRIES.get(source)
    if country is None:
        raise ValueError(f"no config for source {source!r}")
    return load_country_yaml(country, source)


def _utc_now() -> datetime:
    """The wall clock for quiet hours; tests replace it."""
    return datetime.now(timezone.utc)


def _die(message: str, code: int = EXIT_ERROR) -> NoReturn:
    typer.echo(f"ERROR {message}")
    raise typer.Exit(code)


def _source(value: str | None) -> str:
    if value:
        return value
    if len(adapters.ADAPTERS) == 1:
        return next(iter(adapters.ADAPTERS))
    _die(f"--source is required: one of {', '.join(sorted(adapters.ADAPTERS))}")


def _adapter(source: str) -> SourceAdapter:
    try:
        return adapters.get_adapter(source)
    except (ValueError, ImportError, AttributeError) as exc:
        _die(f"no source adapter for {source!r}: {exc}")


def _config(source: str) -> Mapping[str, Any]:
    try:
        return source_config(source)
    except (ValueError, OSError) as exc:
        _die(str(exc))


def _fetcher_http(http: Mapping[str, Any]) -> dict[str, Any]:
    """The config's http block in the Fetcher's shape: `default` becomes `hosts.default`."""
    out = {k: v for k, v in http.items() if k not in ("default", "hosts", "quiet_hours")}
    out["hosts"] = {"default": http["default"], **(http.get("hosts") or {})}
    return out


def _split(page_types: str | None) -> list[str] | None:
    parts = [p.strip() for p in (page_types or "").split(",") if p.strip()]
    return parts or None


def _default_start(source: str, config: Mapping[str, Any]) -> date:
    value = (config.get("backfill") or {}).get("start_date")
    if isinstance(value, date):
        return value
    if value:
        try:
            return date.fromisoformat(str(value))
        except ValueError:
            _die(f"backfill.start_date for {source} is not a date: {value!r}")
    _die(f"no --start given and no backfill.start_date in the {source} config")


def _today(config: Mapping[str, Any]) -> date:
    """Race dates are local to the source, so 'today' is too."""
    try:
        return datetime.now(ZoneInfo(str(config["timezone"]))).date()
    except Exception:
        return date.today()


def _pending(source: str) -> int:
    with FetchQueue(manifest_path(source)) as queue:
        return sum(n for (_, status), n in queue.counts().items() if status == "pending")


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _quiet_windows(source: str, http: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    """The http.quiet_hours entries, validated: (zone, weekdays, start, end, hosts, label) each.
    Each entry is {days: [sat, sun], start: "HH:MM", end: "HH:MM", timezone: Area/City,
    hosts: [...]}, start inclusive, end exclusive; an entry without hosts covers every host."""
    windows = []
    for spec in http.get("quiet_hours") or []:
        try:
            zone = ZoneInfo(str(spec["timezone"]))
            if isinstance(spec["days"], str):
                raise ValueError("days must be a list")
            names = [str(d).strip().lower() for d in spec["days"]]
            unknown = [d for d in names if d not in _WEEKDAYS]
            if unknown or not names:
                raise ValueError(f"unknown days {unknown}; use {', '.join(_WEEKDAYS)}")
            days = {_WEEKDAYS.index(d) for d in names}
            start = time.fromisoformat(str(spec["start"]))
            end = time.fromisoformat(str(spec["end"]))
            if start >= end:
                raise ValueError("start must be before end")
            hosts = frozenset(str(h).lower() for h in spec.get("hosts") or [])
        except Exception as exc:
            _die(f"bad http.quiet_hours entry for {source}: {spec!r} ({exc})")
        day_names = "/".join(_WEEKDAYS[d].title() for d in sorted(days))
        covered = ", ".join(sorted(hosts)) or "every host"
        label = f"{day_names} {start:%H:%M}-{end:%H:%M} {zone.key} for {covered}"
        windows.append((zone, days, start, end, hosts, label))
    return windows


def _quiet_check(windows: list[tuple[Any, ...]], queue: FetchQueue,
                 page_types: list[str] | None) -> Callable[[], str | None] | None:  # fmt: skip
    """A check asked before every task: the window `_utc_now()` falls in that covers the host of
    the next pending task, or None. None when there are no windows."""
    if not windows:
        return None

    def check() -> str | None:
        tasks = queue.next_pending(1, page_types)
        if not tasks:
            return None
        host = _host(tasks[0].url)
        now = _utc_now()
        for zone, days, start, end, hosts, label in windows:
            local = now.astimezone(zone)
            if (not hosts or host in hosts) and local.weekday() in days:
                if start <= local.time() < end:
                    return label
        return None

    return check


def _column_sums(rows: list[list[int]]) -> list[int]:
    return [sum(row[i] for row in rows) for i in range(len(STATUSES))]


def status_lines(
    counts: dict[tuple[str, str], int], rate: float, type_order: list[str], parse_errors: int = 0
) -> list[str]:
    """The `status` table: counts per page_type x status, totals, % done, ETA. At most 20 lines."""
    per_type = {t: [counts.get((t, s), 0) for s in STATUSES] for t in type_order}
    shown = list(type_order)
    if len(shown) > _MAX_STATUS_ROWS:
        biggest = sorted(shown, key=lambda t: -sum(per_type[t]))[: _MAX_STATUS_ROWS - 1]
        shown = [t for t in type_order if t in biggest]
    rows = [(t, per_type[t]) for t in shown]
    hidden = [per_type[t] for t in type_order if t not in shown]
    if hidden:
        rows.append((f"(other:{len(hidden)})", _column_sums(hidden)))
    totals = _column_sums(list(per_type.values()))
    rows.append(("total", totals))

    width = max(len("page_type"), *(len(name) for name, _ in rows))
    lines = [f"{'page_type':<{width}}" + "".join(f"{s:>10}" for s in (*STATUSES, "total"))]
    lines += [
        f"{name:<{width}}" + "".join(f"{n:>10}" for n in (*cells, sum(cells)))
        for name, cells in rows
    ]

    total = sum(totals)
    pending, failed = totals[STATUSES.index("pending")], totals[STATUSES.index("failed")]
    done = total - pending - failed
    pct = 100.0 * done / total if total else 0.0
    lines.append(f"done {pct:.1f}% ({done} of {total} known tasks; follow-ups are added as pages"
                 " are parsed)")  # fmt: skip
    lines.append(f"ETA {pending / rate / 3600:.1f} h for {pending} pending at {rate:.1f} req/s")
    if failed:
        lines.append(f"{failed} failed: see the manifest error column, then `scrape retry-failed`")
    if parse_errors:
        lines.append(f"{parse_errors} kept pages with a parse error (follow-ups missing): fix,"
                     " then `scrape expand`")  # fmt: skip
    return lines


@scrape_app.command()
def plan(source: SourceOpt = None, start: DateOpt = None, end: DateOpt = None) -> None:
    """Enqueue the fixture pages for a date range (idempotent). Defaults: config start .. today."""
    name = _source(source)
    adapter = _adapter(name)
    config = _config(name)
    first = start.date() if start else _default_start(name, config)
    last = end.date() if end else _today(config)
    if first > last:
        _die(f"--start {first} is after --end {last}")
    added = runner.plan_backfill(name, first, last, adapter=adapter)
    typer.echo(f"plan {name} {first}..{last}: {added} new fixture tasks,"
               f" {_pending(name)} pending in total")  # fmt: skip


@scrape_app.command()
def run(
    source: SourceOpt = None,
    max_requests: Annotated[int | None, typer.Option(min=0, help="Stop after N requests.")] = None,
    max_minutes: Annotated[float | None, typer.Option(min=0.0, help="Stop after M min.")] = None,
    page_types: PageTypesOpt = None,
) -> None:
    """Work through the queue politely; resumable. Exit code 2 means the circuit breaker opened."""
    name = _source(source)
    adapter = _adapter(name)
    http = _config(name)["http"]
    windows = _quiet_windows(name, http)
    types = _split(page_types)
    try:
        with FetchQueue(manifest_path(name)) as queue:
            quiet = _quiet_check(windows, queue, types)
            if quiet is not None and (window := quiet()):
                typer.echo(f"scrape run {name}: quiet hours ({window}), no request sent;"
                           f" {_pending(name)} pending")  # fmt: skip
                return
            summary = runner.run_backfill(
                name,
                max_requests=max_requests,
                max_minutes=max_minutes,
                page_types=types,
                adapter=adapter,
                queue=queue,
                http=_fetcher_http(http),
                quiet=quiet,
            )
    except Exception as exc:
        folder = logs_dir(name)
        folder.mkdir(parents=True, exist_ok=True)
        crash = folder / f"scrape-crash-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.log"
        crash.write_text(traceback.format_exc(), encoding="utf-8")
        _die(f"scrape run crashed: {type(exc).__name__}: {' '.join(str(exc).split())[:200]}"
             f" (see {crash})")  # fmt: skip

    if summary.stopped_reason.startswith(runner.REASON_ALREADY_RUNNING):
        _die(summary.stopped_reason)
    if summary.stopped_reason.startswith(runner.REASON_QUIET_HOURS):
        typer.echo(f"scrape run {name}: stopped for {summary.stopped_reason} after"
                   f" {summary.requested} requests; {_pending(name)} pending,"
                   f" log {summary.log_path}")  # fmt: skip
        return
    rate = summary.requested / summary.elapsed_s if summary.elapsed_s > 0 else 0.0
    reason = runner.REASON_CIRCUIT if summary.circuit_opened else summary.stopped_reason
    typer.echo(f"scrape run {name}: stopped ({reason}) after {_duration(summary.elapsed_s)}")
    typer.echo(f"tasks {summary.processed}: network {summary.requested},"
               f" from cache {summary.from_cache}, {rate:.2f} req/s")  # fmt: skip
    typer.echo(f"fetched {summary.fetched}, empty {summary.empty},"
               f" abandoned {summary.abandoned}, failed {summary.failed}")  # fmt: skip
    typer.echo(f"follow-ups enqueued {summary.enqueued_followups}; pending now {_pending(name)}")
    typer.echo(f"log {summary.log_path}")
    if summary.circuit_opened:
        _die(summary.stopped_reason, EXIT_CIRCUIT_OPEN)


@scrape_app.command()
def status(source: SourceOpt = None) -> None:
    """Counts per page type and status, % done and an ETA at the configured default rate."""
    name = _source(source)
    path = manifest_path(name)
    if not path.exists():
        typer.echo(f"no manifest for {name} yet ({path}): run `scrape plan` first")
        return
    rate = float(_config(name)["http"]["default"]["requests_per_second"])
    with FetchQueue(path) as queue:
        counts, parse_errors = queue.counts(), queue.count_parse_errors()
    for line in status_lines(counts, rate, sorted({t for t, _ in counts}), parse_errors):
        typer.echo(line)


@scrape_app.command()
def expand(source: SourceOpt = None, page_types: PageTypesOpt = None) -> None:
    """Re-derive follow-up tasks from already-fetched pages in the raw cache (no network)."""
    name = _source(source)
    adapter = _adapter(name)
    http = _fetcher_http(_config(name)["http"])
    added = runner.expand_followups(name, _split(page_types), adapter=adapter, http=http)
    typer.echo(f"expand {name}: {added} follow-up tasks enqueued from the raw cache,"
               f" {_pending(name)} pending now")  # fmt: skip


@scrape_app.command("retry-failed")
def retry_failed(
    source: SourceOpt = None,
    page_types: PageTypesOpt = None,
    max_attempts: Annotated[
        int,
        typer.Option(min=0, help="Leave tasks attempted this often as failed; 0 = no limit."),
    ] = DEFAULT_RETRY_MAX_ATTEMPTS,
) -> None:
    """Put failed tasks back to pending, except those that keep failing (see --max-attempts)."""
    name = _source(source)
    path = manifest_path(name)
    if not path.exists():
        typer.echo(f"no manifest for {name} yet ({path}): run `scrape plan` first")
        return
    types = _split(page_types)
    with FetchQueue(path) as queue:
        reset = queue.reset_failed(types, max_attempts=max_attempts or None)
        left = sum(
            n
            for (t, status), n in queue.counts().items()
            if status == "failed" and (not types or t in types)
        )
    note = f"; {left} left as failed after {max_attempts}+ attempts (--max-attempts 0 retries them)"
    typer.echo(f"retry-failed {name}: {reset} failed tasks reset to pending{note if left else ''}")
