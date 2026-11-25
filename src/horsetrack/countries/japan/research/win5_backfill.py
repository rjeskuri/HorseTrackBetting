"""`horsetrack japan win5-history`: backfill the WIN5 history table (ticket #135).

Source: netkeiba's WIN5 pages (private use; aggregates may be quoted, rows never enter git). One
year list per year (`/top/win5_results.html?year=YYYY`) names the WIN5 days; one page per day
(`/top/win5.html?date=YYYYMMDD`) prints its sales, winning tickets, dividend and carryovers. Every
request goes through `core.ingest` (cache-first, rate-limited, circuit breaker); past pages are
cached for good, so a rerun only asks for what is new. A year list is final only when fetched after
its year ended (1 January JST); one cached earlier is fetched again. No request while the
jra.yaml quiet hours cover the host. The table goes to `processed_dir('jra') / 'win5.parquet'`;
the command prints aggregate counts only.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timezone
from typing import Annotated
from zoneinfo import ZoneInfo

import typer

from horsetrack.core.config import load_country_yaml, load_source
from horsetrack.core.ingest.cli import _quiet_windows
from horsetrack.core.ingest.fetcher import CircuitOpen, Fetcher, FetchError
from horsetrack.core.ingest.types import FetchTask, RawPage
from horsetrack.countries.japan.logger.log_day import make_fetcher
from horsetrack.countries.japan.research import win5

YEAR_PAGE, DAY_PAGE = "win5_year", "win5_day"
HOST = "race.netkeiba.com"
JST = ZoneInfo("Asia/Tokyo")
DEFAULT_RATE = 0.2  # one request per 5 s: a fifth of jra.yaml's assumed 1 a second for this host
EXIT_CIRCUIT_OPEN = 2


class QuietHours(RuntimeError):
    """A request would fall in a jra.yaml quiet window for the host."""


def _race_host() -> str:
    return load_source("japan", win5.SOURCE).hosts.race.rstrip("/")


def year_url(year: int) -> str:
    return f"{_race_host()}/top/win5_results.html?select=win5_results&year={year:d}"


def day_url(race_date: date) -> str:
    return f"{_race_host()}/top/win5.html?date={race_date:%Y%m%d}"


def _quiet_check(now: Callable[[], datetime]) -> Callable[[], str | None]:
    windows = _quiet_windows(win5.SOURCE, load_country_yaml("japan", win5.SOURCE)["http"])

    def check() -> str | None:
        for zone, days, start, end, hosts, label in windows:
            local = now().astimezone(zone)
            if (not hosts or HOST in hosts) and local.weekday() in days:
                if start <= local.time() < end:
                    return label
        return None

    return check


def _get(fetcher: Fetcher, task: FetchTask, quiet: Callable[[], str | None]) -> RawPage:
    if not task.mutable:
        cached = fetcher.read_cached(task.page_type, task.year, task.key)
        if cached is not None:
            return cached
    window = quiet()
    if window is not None:
        raise QuietHours(window)
    return fetcher.fetch(task)


def _stale_year_list(fetcher: Fetcher, year: int) -> bool:
    """A cached year list fetched before its year ended (1 January JST of the next year) misses
    the year's last WIN5 days, so it is fetched again; one fetched after is final."""
    cached = fetcher.read_cached(YEAR_PAGE, year, str(year))
    return cached is not None and cached.fetched_utc < datetime(year + 1, 1, 1, tzinfo=JST)


def backfill(
    *,
    years: range,
    today: date,
    fetcher: Fetcher,
    quiet: Callable[[], str | None],
    echo: Callable[[str], None] = typer.echo,
) -> tuple[list[dict], list[date], int]:
    """(rows parsed, the WIN5 days the year lists name before `today`, day pages that failed)."""
    expected: list[date] = []
    for year in years:
        task = FetchTask(win5.SOURCE, YEAR_PAGE, str(year), year_url(year), year,
                         mutable=year >= today.year or _stale_year_list(fetcher, year))  # fmt: skip
        days = [d for d in win5.parse_year_list(_get(fetcher, task, quiet).body) if d < today]
        expected += [d for d in days if d.year == year]
    rows, failed = [], 0
    for n, day in enumerate(expected, 1):
        task = FetchTask(win5.SOURCE, DAY_PAGE, f"{day:%Y%m%d}", day_url(day), day.year)
        try:
            page = _get(fetcher, task, quiet)
        except FetchError as exc:
            failed += 1
            echo(f"{day}: not fetched ({exc})")
            continue
        try:
            rows.append(win5.parse_day_page(page.body, day))
        except ValueError as exc:
            failed += 1
            fetcher.quarantine(task)  # asked again on the next run
            echo(f"{day}: not parsed ({exc})")
        if n % 100 == 0:
            echo(f"{n}/{len(expected)} days read")
    return rows, expected, failed


def win5_history_command(
    start_year: Annotated[int, typer.Option(help="First year to read.")] = win5.WIN5_START.year,
    end_year: Annotated[int | None, typer.Option(help="Last year; default: this year.")] = None,
    requests_per_second: Annotated[
        float, typer.Option(help="Requests per second to netkeiba.")
    ] = DEFAULT_RATE,
) -> None:
    """Backfill the WIN5 history (every WIN5 day since 2011-04-24) into the data dir table."""
    today = datetime.now(JST).date()
    years = range(max(start_year, win5.WIN5_START.year), (end_year or today.year) + 1)
    code = 0
    rows: list[dict] = []
    expected: list[date] = []
    failed = 0
    with make_fetcher(win5.SOURCE, requests_per_second=requests_per_second) as fetcher:
        try:
            rows, expected, failed = backfill(
                years=years,
                today=today,
                fetcher=fetcher,
                quiet=_quiet_check(lambda: datetime.now(timezone.utc)),
            )
        except (CircuitOpen, QuietHours) as exc:
            typer.echo(f"stopped: {exc}; run the command again to resume (pages are cached)")
            code = EXIT_CIRCUIT_OPEN if isinstance(exc, CircuitOpen) else 0
        sent = fetcher.requests_sent
    table = win5.build_history(rows)
    if len(table):
        typer.echo(f"wrote {win5.write_history(table)}")
    counts = win5.history_counts(table)
    gaps = win5.missing_days(table, expected)
    typer.echo(" ".join(f"{k}={v}" for k, v in counts.items()))
    breaks = win5.carryover_chain_breaks(table)
    typer.echo(f"expected_days={len(expected)} missing_days={len(gaps)} failed_pages={failed} "
               f"carryover_breaks={len(breaks)} requests_sent={sent}")  # fmt: skip
    if gaps:
        typer.echo("missing: " + ", ".join(d.isoformat() for d in gaps[:20]))
    if breaks:
        typer.echo("carryover breaks: " + ", ".join(d.isoformat() for d in breaks[:20]))
    raise typer.Exit(code)
