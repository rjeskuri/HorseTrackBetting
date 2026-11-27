"""The jravan store's `horse_runs`, every run of a horse, from its `races` and `runners` (#403).

`build_horse_runs` reads `processed/jravan/races.parquet` and `runners.parquet` (from
`races.build_races` and `runners.build_runners`) through the scoped reader
`models.v0.store.read_store_table(..., source='jravan')`, one calendar year at a time over the
years the two tables span, so holdout days are never read. Each runner with a `horse_id` is one
row, a scratched or excluded one too (its placing mark kept, as a form table lists it), in the jra
store's vocabulary, which the jravan races and runners already write: the race's `race_index`,
`venue`, `surface`, `course_config`, `distance_m`, `going` and `race_class` (on `race_id`), the
rest from the runner. `win_odds` follows D10: the runner's Win odds for a run on or after
2014-06-07, read with `touches_price=True`, and null before it, the run itself kept. The table is
written to `processed/jravan/horse_runs.parquet` through `records.write_parquet` (tmp file and
`os.replace`), in the columns and types of `schemas.TABLES['horse_runs']`, without schema
metadata, sorted by its sort key and then `race_id`, so the same inputs give the same bytes. The
report holds counts only.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import NoReturn

import pyarrow as pa
import pyarrow.compute as pc
import typer

from horsetrack.core import paths
from horsetrack.core.store.build import arrow_schema
from horsetrack.core.store.schemas import TABLES
from horsetrack.countries.japan.jravan import records
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table

SOURCE = records.SOURCE
NAME = "horse_runs"
FILENAME = f"{NAME}.parquet"
# the span the year scan looks within (the reader's dates must fit a pandas Timestamp)
EARLIEST, LATEST = date(1900, 1, 1), date(2199, 12, 31)
FROM_RACE = ("race_index", "venue", "surface", "course_config", "distance_m", "going", "race_class")
FROM_RUNNER = (
    "jurisdiction", "horse_id", "race_date", "race_id", "draw", "rating", "trainer_name",
    "jockey_name", "lbw_raw", "actual_wt_lb", "running_positions", "finish_time_s",
    "declared_horse_wt_lb", "gear_raw", "placing_raw",
)  # fmt: skip
PRICE = "win_odds"
YEAR_COUNTS = ("runners", "without_horse_id", "runs_written", "before_price_era", "without_race")


def _years() -> list[int]:
    """Every calendar year from the first to the last race day of the jravan races and runners
    (holdout days dropped by the reader)."""
    first, last = [], []
    for table in ("races", "runners"):
        days = read_store_table(
            table, columns=["race_date"], start=EARLIEST, end=LATEST, touches_price=False,
            source=SOURCE,
        )["race_date"]  # fmt: skip
        span = pc.min_max(days).as_py()
        if span["min"] is not None:
            first.append(span["min"])
            last.append(span["max"])
    return list(range(min(first).year, max(last).year + 1)) if first else []


def _runners(lo: date, hi: date) -> pa.Table:
    """The runners of `lo..hi` with `FROM_RUNNER` and `win_odds`: the price read only inside D10's
    era, null before it."""
    parts = []
    if lo < PRICE_ERA_START:
        before = read_store_table(
            "runners", columns=FROM_RUNNER, start=lo,
            end=min(hi, PRICE_ERA_START - timedelta(days=1)), touches_price=False, source=SOURCE,
        )  # fmt: skip
        parts.append(before.append_column(PRICE, pa.nulls(before.num_rows, pa.float64())))
    if hi >= PRICE_ERA_START:
        parts.append(
            read_store_table(
                "runners",
                columns=[*FROM_RUNNER, PRICE],
                start=max(lo, PRICE_ERA_START),
                end=hi,
                touches_price=True,
                source=SOURCE,
            )  # fmt: skip
        )
    schema = pa.schema([arrow_schema("runners").field(c) for c in (*FROM_RUNNER, PRICE)])
    return pa.concat_tables([part.cast(schema) for part in parts])


def _year(lo: date, hi: date, counts: dict[str, int]) -> pa.Table:
    """The `horse_runs` rows of `lo..hi`, unsorted."""
    runners = _runners(lo, hi)
    counts["runners"] = runners.num_rows
    runners = runners.filter(pc.is_valid(runners["horse_id"]))
    counts["without_horse_id"] = counts["runners"] - runners.num_rows
    counts["before_price_era"] = pc.sum(pc.less(runners["race_date"], PRICE_ERA_START)).as_py() or 0
    races = read_store_table(
        "races", columns=["race_id", *FROM_RACE], start=lo, end=hi, touches_price=False,
        source=SOURCE,
    )  # fmt: skip
    at = pc.index_in(runners["race_id"], value_set=races["race_id"].combine_chunks())
    counts["without_race"] = at.null_count
    race = races.take(at)
    counts["runs_written"] = runners.num_rows
    schema = arrow_schema(NAME)
    return pa.table(
        {
            c: (race if c in FROM_RACE else runners)[c].cast(schema.field(c).type)
            for c in schema.names
        },
        schema=schema,
    )


def build_horse_runs() -> records.BuildReport:
    """Build `processed/jravan/horse_runs.parquet` from the jravan races and runners, one year at
    a time; see the module docstring. The report holds counts only."""
    out = paths.processed_dir(SOURCE)
    schema = arrow_schema(NAME)
    by_year: dict[int, dict[str, int]] = {}
    tables = []
    for year in _years():
        counts = dict.fromkeys(YEAR_COUNTS, 0)
        tables.append(_year(date(year, 1, 1), date(year, 12, 31), counts))
        by_year[year] = counts
    table = pa.concat_tables(tables) if tables else schema.empty_table()
    keys = [*TABLES[NAME].sort_by, "race_id"]  # race_id breaks a tie, so no order is left open
    table = table.take(pc.sort_indices(table, [(c, "ascending") for c in keys])).combine_chunks()
    records.write_parquet(table.replace_schema_metadata(None), out / FILENAME)
    totals = {k: sum(c[k] for c in by_year.values()) for k in YEAR_COUNTS}
    return records.BuildReport(
        name=NAME, counts={"years": len(by_year), **totals}, by_year=by_year, unmapped={},
        failures=[],
    )  # fmt: skip


def horse_runs_command() -> None:
    """Build the jravan store's horse_runs (every run of a horse) from its races and runners,
    which `jravan races` and `jravan runners` write first; the report (counts only) goes to
    <reports dir>/jravan-horse_runs.json."""
    try:
        report = build_horse_runs()
        target = records.write_build_report(report)
    except (paths.DataDirNotSet, ValueError, LookupError, OSError) as exc:
        _die(f"jravan horse-runs failed: {type(exc).__name__}: {exc}")
    c = report.counts
    typer.echo(
        f"horse_runs: {c['years']} years, {c['runs_written']} runs written, "
        f"{c['before_price_era']} before the price era (win_odds null, D10), "
        f"{c['without_horse_id']} runners without a horse id, {c['without_race']} without a race"
    )
    typer.echo(f"report: {target}")


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)
