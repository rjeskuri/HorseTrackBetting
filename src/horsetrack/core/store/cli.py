"""`horsetrack store ...`: build the normalised tables, check them, show what is there.

Every command prints at most 15 lines (build: 20, as it runs the checks too); details go to
reports_dir()/store-build-<source>.md and store-check-<source>.md. A failure prints one line
starting with `ERROR` and exits non-zero. Ported from HKTrack.
"""

from __future__ import annotations

import json
from typing import Annotated, NoReturn

import typer

from horsetrack.core.config import SourceConfig, load_source
from horsetrack.core.ingest import adapters
from horsetrack.core.ingest.cli import SOURCE_COUNTRIES
from horsetrack.core.paths import processed_dir

store_app = typer.Typer(help="Normalised tables.", no_args_is_help=True, add_completion=False)

EXIT_ERROR = 1
_MAX_CHECK_LINES = 10  # build summary (<= 8) + checks (<= 10 + 1) + verdict: <= 20 lines

SourceOpt = Annotated[
    str | None,
    typer.Option("--source", "-s", help="Source slug; default: the only one registered."),
]


def source_config(source: str) -> SourceConfig:
    """The typed config of `source`, from the country that owns it (the scrape CLI's mapping)."""
    country = SOURCE_COUNTRIES.get(source)
    if country is None:
        raise ValueError(f"no config for source {source!r}")
    return load_source(country, source)


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {' '.join(str(message).split())[:400]}")
    raise typer.Exit(EXIT_ERROR)


def _source(value: str | None) -> str:
    if value is None:
        if len(adapters.ADAPTERS) != 1:
            _die(f"--source is required: one of {', '.join(sorted(adapters.ADAPTERS))}")
        return next(iter(adapters.ADAPTERS))
    if value not in adapters.ADAPTERS:
        _die(f"unknown source {value!r}; known sources: {', '.join(sorted(adapters.ADAPTERS))}")
    return value


def _config(source: str) -> SourceConfig:
    try:
        return source_config(source)
    except (ValueError, OSError) as exc:
        _die(str(exc))


def _split(value: str | None) -> list[str] | None:
    parts = [p.strip() for p in (value or "").split(",") if p.strip()]
    return parts or None


@store_app.command()
def build(
    source: SourceOpt = None,
    page_types: Annotated[
        str | None,
        typer.Option("--page-types", help="Only parse these page types now, e.g. results,horse."),
    ] = None,
    workers: Annotated[
        int | None, typer.Option(min=1, help="Parser processes (default min(6, cpus - 2)).")
    ] = None,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Parse at most N new pages per page type.")
    ] = None,
) -> None:
    """Parse new or changed pages, rewrite every table, then run the checks."""
    from horsetrack.core.store.build import build_tables  # pandas / pyarrow only when needed

    name = _source(source)
    config = _config(name)
    try:
        summary = build_tables(name, page_types=_split(page_types), workers=workers, limit=limit)
    except Exception as exc:
        _die(f"store build failed: {type(exc).__name__}: {exc}")
    for line in summary.lines():
        typer.echo(line)
    _echo_checks(name, config)


def _echo_checks(name: str, config: SourceConfig) -> None:
    """The checks on the tables just written: those that are not OK (failures first, at most
    _MAX_CHECK_LINES), then one verdict line; a failure ends in the ERROR line and exit 1."""
    from horsetrack.core.store.checks import FAIL, OK, run_checks

    try:
        result = run_checks(name, config=config)
    except Exception as exc:
        _die(f"store check could not run: {type(exc).__name__}: {exc}")
    flagged = sorted(
        (check for check in [*result.checks, *result.integrity] if check.level != OK),
        key=lambda check: check.level != FAIL,
    )
    for check in flagged[:_MAX_CHECK_LINES]:
        typer.echo(check.line())
    if len(flagged) > _MAX_CHECK_LINES:
        typer.echo(f"... {len(flagged) - _MAX_CHECK_LINES} more in {result.report_path}")
    if not result.ok:
        _die(result.error_line() or "store check failed")
    typer.echo(f"store check {name}: OK, {len(flagged)} warnings")


@store_app.command()
def check(source: SourceOpt = None) -> None:
    """Reconciliation checks on the built tables; exits 1 on a failed check."""
    from horsetrack.core.store.checks import run_checks

    name = _source(source)
    config = _config(name)
    try:
        result = run_checks(name, config=config)
    except Exception as exc:
        _die(f"store check could not run: {type(exc).__name__}: {exc}")
    for line in result.lines():
        typer.echo(line)
    if not result.ok:
        _die(result.error_line() or "store check failed")


@store_app.command()
def info(source: SourceOpt = None) -> None:
    """Row counts per table, the race-date range and the snapshot id of the last build."""
    from horsetrack.core.store import tables
    from horsetrack.core.store.build import BUILD_FILE

    name = _source(source)
    try:
        counts = tables.row_counts(name)
        if not counts:
            _die(f"no tables under {processed_dir(name)}: run `horsetrack store build` first")
        build_file = processed_dir(name) / BUILD_FILE
        last = json.loads(build_file.read_text(encoding="utf-8")) if build_file.exists() else {}
    except typer.Exit:
        raise
    except Exception as exc:
        _die(f"store info failed: {type(exc).__name__}: {exc}")
    typer.echo(f"store {name}: {processed_dir(name)}")
    width = max(len(table) for table in counts)
    for table, n in counts.items():
        typer.echo(f"  {table:<{width}} {n:>9}")
    typer.echo(f"race dates {last.get('date_min')} .. {last.get('date_max')}")
    snapshot = str(last.get("snapshot_id", ""))[:12] or "unknown"
    typer.echo(f"snapshot {snapshot} built {last.get('built_utc', 'unknown')}")
