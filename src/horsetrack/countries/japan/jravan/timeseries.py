"""JV-Data 0B41/0B42 time-series odds -> the jravan `odds_timeseries` day files (`horsetrack japan
jravan timeseries`).

`build_day` reads one race day's `<raw>/timeseries/0B41/<YYYY>/<YYYYMMDD>.jvd` (O1 records: WIN,
PLACE, BRACKET_QUINELLA) and `0B42/...` (O2: QUINELLA), either of which may be missing, streams
them with `records.iter_file`, decodes every record with the O1-O6 mapping (`odds.odds_rows`) and
writes `<out>/odds_timeseries/<YYYY-MM-DD>.parquet` (`out` default `processed/jravan`) in exactly
`logger.compact.SCHEMA`, the schema the odds logger's days use:

- `snapshot_utc` is the record's 発表月日時分 (JST) as UTC (`keys.announced_utc`); the file is the
  race day's, so the evening-before boards are in it. A record without an announcement time
  (`00000000`, blank or invalid) has no place in a time series and writes no row; the report
  counts those records. `fetched_utc` is null: the export has no fetch time.
- `post_utc` is the race's scheduled `post_time_local` (JST, on the race date) from
  `processed/jravan/races.parquet`, null without a race row or a post time; `minutes_to_post` is
  (`post_utc` - `snapshot_utc`) in minutes, negative after the post.
- The same (race_id, pool, combination, snapshot_utc) twice is one row, the later record's (O1
  records before O2, each file in its record order).
- `is_final_capture` marks the latest `snapshot_utc` per (race_id, pool, combination), pool rows
  included; `dividend` and `is_refund` are a left join on (race_id, pool, combination) to
  `processed/jravan/dividends.parquet`, null without a match or without the table.

Rows are sorted on (race_id, pool, combination, snapshot_utc), unique after the dedupe, so the
same inputs give the same bytes whatever their record order (`records.write_parquet`, no
wall-clock metadata). `_build.json` beside the day files records, per day, what the file was built
from: both inputs' size and mtime, the workbook's sha256, `PARSER_VERSION` and a sha256 of the
day's races and dividends rows. A day whose record matches and whose file exists is not rebuilt.

`build_all` runs every day of the requested years, one at a time in one process; a day that fails
is in the report's `failures` and the other days are written. The report holds counts only.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import statistics
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Annotated, Any, NoReturn

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import typer

from horsetrack.core import paths
from horsetrack.countries.japan.jravan import keys, odds, records, spec
from horsetrack.countries.japan.logger import compact

SOURCE = records.SOURCE
NAME = "timeseries"  # the report's name: reports/jravan-timeseries.json
FAMILIES = {"0B41": "O1", "0B42": "O2"}  # in record order: O1 records before O2 on a tie
FOLDER = "odds_timeseries"
BUILD_FILE = "_build.json"
# Bumped whenever the rows a day gives change, so every day is rebuilt once.
PARSER_VERSION = 1
SAVE_EVERY = 50  # days written between two saves of _build.json in build_all
POOLS = tuple(p.code for p in odds.POOLS if p.record_type in FAMILIES.values())
KEY = ("race_id", "pool", "combination", "snapshot_utc")
_UTC = compact.SCHEMA.field("snapshot_utc").type
_LOGGED = ("race_id", "venue", "race_no", "pool", "combination", "published_utc", "odds",
           "odds_hi", "investment", "status_raw")  # fmt: skip
_SEP = "\x1f"
_US_PER_MINUTE = 60_000_000


# ------------------------------------------------------------------------------------------------
# What a run reads once: the layouts, the workbook's digest, the races and dividends to join


@dataclass(frozen=True)
class _Context:
    raw: Path
    folder: Path
    layouts: dict[str, spec.RecordSpec]
    workbook: str
    races: pa.Table  # race_date (string), race_id, post_utc
    dividends: pa.Table  # race_date (string), key, dividend, is_refund


def _context(raw: Path | None, out: Path | None) -> _Context:
    root = Path(raw) if raw is not None else paths.raw_dir(SOURCE)
    folder = (Path(out) if out is not None else paths.processed_dir(SOURCE)) / FOLDER
    workbook = spec.spec_path()
    layouts = {rt: spec.record_spec(rt) for rt in FAMILIES.values()}
    digest = hashlib.sha256(workbook.read_bytes()).hexdigest()
    store = paths.processed_dir(SOURCE)
    return _Context(
        root, folder, layouts, digest, _races(store / "races.parquet"),
        _dividends(store / "dividends.parquet"),
    )  # fmt: skip


def _races(path: Path) -> pa.Table:
    schema = pa.schema([("race_date", pa.string()), ("race_id", pa.string()), ("post_utc", _UTC)])
    if not path.is_file():
        return schema.empty_table()
    table = pq.read_table(path, columns=["race_id", "race_date", "post_time_local"])
    posts = [
        _post_utc(d, t)
        for d, t in zip(
            table["race_date"].to_pylist(), table["post_time_local"].to_pylist(), strict=True
        )
    ]
    return pa.table(
        [pc.cast(table["race_date"], pa.string()), table["race_id"], pa.array(posts, _UTC)],
        schema=schema,
    )


def _post_utc(day: date | None, hhmm: str | None) -> datetime | None:
    if day is None or not hhmm:
        return None
    hours, minutes = (int(x) for x in hhmm.split(":"))
    return datetime.combine(day, time(hours, minutes), keys.JST).astimezone(timezone.utc)


def _dividends(path: Path) -> pa.Table:
    schema = pa.schema(
        [("race_date", pa.string()), ("key", pa.string()), ("dividend", pa.float64()),
         ("is_refund", pa.bool_())]
    )  # fmt: skip
    if not path.is_file():
        return schema.empty_table()
    table = pq.read_table(path, columns=["race_id", "pool", "combination", "dividend", "is_refund"])
    return pa.table(
        [
            pc.utf8_slice_codeunits(table["race_id"], 4, 14),  # jra:<YYYY-MM-DD>:...
            _join_key(table),
            pc.cast(table["dividend"], pa.float64()),
            pc.cast(table["is_refund"], pa.bool_()),
        ],
        schema=schema,
    )


def _join_key(table: pa.Table) -> pa.ChunkedArray:
    return pc.binary_join_element_wise(table["race_id"], table["pool"], table["combination"], _SEP)


def _day_joins(ctx: _Context, day: date) -> tuple[pa.Table, pa.Table, str]:
    """The day's races (race_id, post_utc) and dividends (key, dividend, is_refund), and a sha256
    of both: a change to either rebuilds the day."""
    iso = day.isoformat()
    races = ctx.races.filter(pc.equal(ctx.races["race_date"], iso)).sort_by("race_id")
    paid = ctx.dividends.filter(pc.equal(ctx.dividends["race_date"], iso)).sort_by("key")
    if len(set(paid["key"].to_pylist())) != paid.num_rows:
        raise ValueError(f"dividends hold one (race_id, pool, combination) twice on {iso}")
    digest = hashlib.sha256()
    for table in (races.drop_columns("race_date"), paid.drop_columns("race_date")):
        digest.update(json.dumps(table.to_pylist(), default=str).encode())
    return races, paid, digest.hexdigest()


# ------------------------------------------------------------------------------------------------
# One day


def _jvd(raw: Path, family: str, day: date) -> Path:
    return raw / "timeseries" / family / f"{day.year:04d}" / f"{day:%Y%m%d}.jvd"


def _stamp(path: Path) -> list[int] | None:
    try:
        s = path.stat()
    except FileNotFoundError:
        return None
    return [s.st_size, s.st_mtime_ns]


def _read(ctx: _Context, day: date) -> tuple[pa.Table, dict[str, Any]]:
    """The day's logged rows (`_LOGGED` with `published_utc` as `snapshot_utc`, and `seq`, the
    record's position over both files), and the read's counts."""
    parts: list[pa.Table] = []
    counts: collections.Counter = collections.Counter()
    offset = 0
    for family, rt in FAMILIES.items():
        path = _jvd(ctx.raw, family, day)
        if not path.is_file():
            continue
        for cols in records.iter_file(path, ctx.layouts[rt]):
            n = len(cols["開催年"])
            counts["records"] += n
            counts.update(f"kubun_{k}" for k in records.text(cols["データ区分"]))
            table, record = odds._rows(rt, cols)
            other = pc.not_equal(table["race_date"], pa.scalar(day, pa.date32()))
            if pc.any(other).as_py():
                raise ValueError(f"{path.name}: records of a race on another day")
            no_time = pc.is_null(table["published_utc"]).to_numpy(zero_copy_only=False)
            counts["no_snapshot_records"] += int(np.unique(record[no_time]).size)
            venues = records.text(cols["競馬場コード"])
            counts["non_jra_records"] += sum(keys.jurisdiction(v) != "jra" for v in venues)
            table = table.select(list(_LOGGED)).rename_columns([*_LOGGED[:5], "snapshot_utc",
                                                                *_LOGGED[6:]])  # fmt: skip
            table = table.append_column("seq", pa.array(record + offset, pa.int64()))
            parts.append(table.filter(pa.array(~no_time)))
            offset += n
    if not parts:
        raise FileNotFoundError(f"no 0B41 or 0B42 file for {day.isoformat()}")
    return pa.concat_tables(parts), dict(counts)


def _changes(table: pa.Table, columns: Iterable[str]) -> np.ndarray:
    """Per row of a sorted table: True where any of `columns` differs from the row before (the
    first row True)."""
    n = table.num_rows
    new = np.ones(n, dtype=bool)
    if n > 1:
        same = np.ones(n - 1, dtype=bool)
        for name in columns:
            column = table[name]
            equal = pc.equal(column.slice(1), column.slice(0, n - 1))
            same &= pc.fill_null(equal, False).to_numpy(zero_copy_only=False)
        new[1:] = ~same
    return new


def _rows(table: pa.Table, races: pa.Table, paid: pa.Table) -> pa.Table:
    """The day file: deduplicated, sorted, with the derived columns, in `compact.SCHEMA`."""
    order = [(k, "ascending") for k in KEY] + [("seq", "descending")]
    table = table.sort_by(order)
    table = table.filter(pa.array(_changes(table, KEY)))  # the later record of a repeated key
    n = table.num_rows
    final = np.ones(n, dtype=bool)
    final[:-1] = _changes(table, KEY[:3])[1:]  # the last snapshot of each (race, pool, combination)

    post = races["post_utc"].take(pc.index_in(table["race_id"], value_set=races["race_id"]))
    minutes = pc.divide(
        pc.cast(pc.subtract(pc.cast(post, pa.int64()), pc.cast(table["snapshot_utc"], pa.int64())),
                pa.float64()),
        float(_US_PER_MINUTE),
    )  # fmt: skip
    match = pc.index_in(_join_key(table), value_set=paid["key"])
    columns = {
        "fetched_utc": pa.nulls(n, _UTC),
        "post_utc": post,
        "minutes_to_post": minutes,
        "is_final_capture": pa.array(final, pa.bool_()),
        "dividend": paid["dividend"].take(match),
        "is_refund": paid["is_refund"].take(match),
    }
    out = [columns[f.name] if f.name in columns else table[f.name] for f in compact.SCHEMA]
    return pa.table([pc.cast(c, f.type) for c, f in zip(out, compact.SCHEMA, strict=True)],
                    schema=compact.SCHEMA)  # fmt: skip


def _stats(table: pa.Table, counts: dict[str, Any], races: pa.Table) -> dict[str, Any]:
    """The day's counts for the build record and the report (no record content)."""
    per_race = table.group_by("race_id").aggregate([("snapshot_utc", "count_distinct")])
    per_race = per_race.sort_by("race_id")
    race_ids = per_race["race_id"]
    with_post = races.filter(pc.is_valid(races["post_utc"]))["race_id"]
    pools = {c["values"]: int(c["counts"]) for c in pc.value_counts(table["pool"]).to_pylist()}
    return {
        **counts,
        "rows": table.num_rows,
        **{f"rows_{code}": pools.get(code, 0) for code in POOLS},
        "races": len(race_ids),
        "races_without_post": int(pc.sum(pc.invert(pc.is_in(race_ids, with_post))).as_py() or 0),
        "rows_with_dividend": int(pc.sum(pc.is_valid(table["dividend"])).as_py() or 0),
        "snapshots_per_race": per_race["snapshot_utc_count_distinct"].to_pylist(),
    }


def _build_one(ctx: _Context, day: date, built: dict[str, dict]) -> tuple[str, dict | None]:
    """Build or skip `day`; ('written' | 'skipped', its record) or ('failed', the failure)."""
    iso = day.isoformat()
    target = ctx.folder / f"{iso}.parquet"
    try:
        races, paid, joins = _day_joins(ctx, day)
        fingerprint = {
            "parser": PARSER_VERSION,
            "workbook": ctx.workbook,
            "inputs": {f: _stamp(_jvd(ctx.raw, f, day)) for f in FAMILIES},
            "joins": joins,
        }
        entry = built.get(iso)
        if entry and entry.get("fingerprint") == fingerprint and target.is_file():
            return "skipped", entry
        logged, counts = _read(ctx, day)
        removed = logged.num_rows
        table = _rows(logged, races, paid)
        del logged
        counts["repeated_rows"] = removed - table.num_rows
        stats = _stats(table, counts, races)
        records.write_parquet(table, target)
    except (ValueError, LookupError, OSError, pa.ArrowException) as exc:
        built.pop(iso, None)  # retried on the next run; a previous file is left as it was
        return "failed", {"day": iso, "error": f"{type(exc).__name__}: {exc}"}
    entry = {"fingerprint": fingerprint, "stats": stats}
    built[iso] = entry
    return "written", entry


# ------------------------------------------------------------------------------------------------
# The build record and the report


def _load_built(folder: Path) -> dict[str, dict]:
    try:
        data = json.loads((folder / BUILD_FILE).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    days = data.get("days") if isinstance(data, dict) else None
    return days if isinstance(days, dict) else {}


def _save_built(folder: Path, built: dict[str, dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / BUILD_FILE
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        tmp.write_text(json.dumps({"days": built}, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def _report(outcomes: list[tuple[date, str, dict | None]]) -> records.BuildReport:
    counts: collections.Counter = collections.Counter()
    grain: dict[int, list[int]] = collections.defaultdict(list)
    by_year: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    failures = []
    for day, status, entry in outcomes:
        counts[f"days_{status}"] += 1
        if status == "failed":
            failures.append(entry)
            continue
        stats = dict(entry["stats"])
        grain[day.year].extend(stats.pop("snapshots_per_race"))
        counts["days"] += 1
        counts.update(stats)
        year = by_year[day.year]
        year["days"] += 1
        year.update({k: stats[k] for k in ("records", "rows", "races") if k in stats})
    out_years: dict[int, dict[str, Any]] = {}
    for year in sorted(by_year):
        snaps = grain[year]
        out_years[year] = dict(by_year[year])
        if snaps:
            out_years[year].update(
                snapshots_per_race_median=float(statistics.median(snaps)),
                snapshots_per_race_min=min(snaps),
                snapshots_per_race_max=max(snaps),
            )
    for key in ("days", "days_written", "days_skipped", "days_failed", "records", "rows",
                *(f"rows_{code}" for code in POOLS)):  # fmt: skip
        counts.setdefault(key, 0)
    return records.BuildReport(
        name=NAME,
        counts=dict(sorted(counts.items())),
        by_year=out_years,
        unmapped={},
        failures=failures,
    )


def _run(days: list[date], raw: Path | None, out: Path | None) -> records.BuildReport:
    ctx = _context(raw, out)
    built = _load_built(ctx.folder)
    outcomes: list[tuple[date, str, dict | None]] = []
    unsaved = 0
    try:
        for day in days:
            status, entry = _build_one(ctx, day, built)
            outcomes.append((day, status, entry))
            unsaved += status != "skipped"
            if unsaved >= SAVE_EVERY:
                _save_built(ctx.folder, built)
                unsaved = 0
    finally:
        if unsaved or not (ctx.folder / BUILD_FILE).exists():
            _save_built(ctx.folder, built)
    return _report(outcomes)


def build_day(
    race_date: date, *, raw: Path | None = None, out: Path | None = None
) -> records.BuildReport:
    """Build `<out>/odds_timeseries/<race_date>.parquet` (default `processed/jravan`) from the
    day's 0B41 and 0B42 files under `raw` (default `paths.raw_dir('jravan')`), unless its build
    record matches. A day that cannot be built is in the report's `failures`."""
    return _run([race_date], raw, out)


def _days(raw: Path, years: list[int] | None) -> list[date]:
    found: set[date] = set()
    for family in FAMILIES:
        base = raw / "timeseries" / family
        if not base.is_dir():
            continue
        for folder in base.iterdir():
            if not (folder.is_dir() and folder.name.isdigit() and len(folder.name) == 4):
                continue
            if years is not None and int(folder.name) not in years:
                continue
            for path in folder.glob("*.jvd"):
                try:
                    day = datetime.strptime(path.stem, "%Y%m%d").date()
                except ValueError:
                    continue
                if day.year == int(folder.name):
                    found.add(day)
    return sorted(found)


def build_all(
    *, years: list[int] | None = None, raw: Path | None = None, out: Path | None = None
) -> records.BuildReport:
    """`build_day` for every race day of `years` (None: every year) with a 0B41 or 0B42 file,
    one day at a time; a day that fails is in `failures` and the others are written."""
    root = Path(raw) if raw is not None else paths.raw_dir(SOURCE)
    wanted = None if years is None else {int(y) for y in years}
    return _run(_days(root, wanted), raw, out)


# ------------------------------------------------------------------------------------------------
# The command


def timeseries_command(
    year: Annotated[
        list[int] | None, typer.Option("--year", help="A year to build (repeatable).")
    ] = None,
    day: Annotated[
        list[str] | None, typer.Option("--day", help="A race day YYYY-MM-DD (repeatable).")
    ] = None,
) -> None:
    """Parse the JRA-VAN export's 0B41/0B42 time-series odds into
    processed/jravan/odds_timeseries/<YYYY-MM-DD>.parquet; the counts in
    reports/jravan-timeseries.json. Exits 1 when a day failed."""
    if year and day:
        raise typer.BadParameter("give --year or --day, not both")
    try:
        if day:
            days = [_parse_day(d) for d in day]
            report = _run(sorted(set(days)), None, None)
        else:
            report = build_all(years=year or None)
        written = records.write_build_report(report)
    except (paths.DataDirNotSet, ValueError, LookupError, OSError) as exc:
        _die(f"jravan timeseries failed: {type(exc).__name__}: {exc}")
    c = report.counts
    typer.echo(
        f"odds_timeseries: {c['days_written']} days written, {c['days_skipped']} unchanged, "
        f"{c['days_failed']} failed; {c['rows']} rows, {c['records']} records; "
        + ", ".join(f"{code} {c[f'rows_{code}']}" for code in POOLS)
    )
    for failure in report.failures:
        typer.echo(f"failed {failure['day']}: {failure['error']}", err=True)
    typer.echo(f"report: {written}")
    if report.failures:
        raise typer.Exit(1)


def _parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"{value!r} is not a YYYY-MM-DD date", param_hint="--day") from exc


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)
