"""Miho and Ritto 坂路 and ウッドチップ workouts from JRA-VAN, as of decision time, and #137's
workout features from this source (#229).

Source and licence: JRA-VAN Data Lab (JV-Data HC, the 坂路 automatic times, and WC, the ウッド
ones), private use in the owner's own programs only (D21). The table, its build report and
anything built on it stay under the data dir; the repository holds no layout and no figure: the
layouts are read from the workbook at run time (`jravan.spec`), the items by name.

`build_workout_table()` reads every `<raw>/stored/SLOP/HC/<YYYY>.jvd` and
`<raw>/stored/WOOD/WC/<YYYY>.jvd` (`raw` default `paths.raw_dir('jravan')`), keeps one version
per key (`records.latest_versions` on the layout's key items, across every year file of the
record type; a key whose latest version is a deletion has no row) and writes
`<out>/workouts.parquet` (default `paths.processed_dir('jravan')`), sorted, so the same inputs
give the same bytes. Its columns are #137's: `horse_id` (血統登録番号 as printed; blank or all
zeros null), `workout_date` and `race_date` (both 調教年月日, the latter for the scoped reader),
`workout_time` (調教時刻 `HH:MM`), `centre` (トレセン区分 0 miho, 1 ritto, else other), `course`
(hill for HC, wood for WC), `furlongs` (the longest n-furlong total measured), `split_first_s`
(that total), `split_mid_s` (the 3-furlong total), `final_furlong_s` (the 200 m to 0 m lap); times
are tenths of a second, all zeros (no measure) or all nines (over the limit) null. `effort` and
`partner` are null: the source has neither. The totals are matched by their distance in the item
name, not by the name's spelling.

Publication (owner decision on #229, 2026-11-27): a workout counts from the end of its training
day, (workout_date + 1 day) 00:00 Asia/Tokyo (`published_at`), strictly before the decision time,
and only for a race on a later day; データ作成年月日 is ignored (in the stored export it is a later
re-issue). The latest stored version is used at every decision time, an accepted look-ahead as
for the masters.

`workout_table(race_id, as_of=...)` and `workout_features(race_id, as_of=...)` are #137's with
this source's rules: par (the median final furlong per centre and course of every record before
race day published before `as_of`, with at least `PAR_MIN_WORKOUTS`), best and trend over the
newest `N_RECENT` workouts with a par (#137's code), and `days_since_hard` always null. They read
no JRDB table and ignore `HORSETRACK_JRDB_FEATURES`. Every store read goes through #114's
`read_store_table`, the workouts by their workout date, so a workout on a holdout day is never
read and a holdout race day reads nothing.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import typer

from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.reports import write_report
from horsetrack.countries.japan.handicap import workouts as w137
from horsetrack.countries.japan.handicap.pp import race_day_of
from horsetrack.countries.japan.jravan import records, spec
from horsetrack.countries.japan.models.v0.store import read_store_table

SOURCE = records.SOURCE
NAME = TABLE = "workouts"
COUNTRY = "japan"
JST, UTC = w137.JST, w137.UTC
HISTORY_START = w137.HISTORY_START

N_RECENT = w137.N_RECENT
PAR_MIN_WORKOUTS = w137.PAR_MIN_WORKOUTS
FEATURE_COLUMNS = ("ff_best_vs_par_s", "ff_trend_s", "days_since_hard")

CREATED, KUBUN = "データ作成年月日", "データ区分"
CENTRE_ITEM, DATE_ITEM, TIME_ITEM, HORSE_ITEM = (
    "トレセン区分",
    "調教年月日",
    "調教時刻",
    "血統登録番号",
)
CENTRES = {"0": "miho", "1": "ritto"}  # トレセン区分; any other code is "other"
SOURCES = (  # record type, folder under the export root, course
    ("HC", ("stored", "SLOP", "HC"), "hill"),
    ("WC", ("stored", "WOOD", "WC"), "wood"),
)
_TOTAL = re.compile(r"合計[(（](\d+)M[～〜~]0M[)）]$")  # an n-furlong total, by its distance
_FINAL_LAP = re.compile(r"^ラップタイム[(（]200M[～〜~]0M[)）]$")
_YEAR_FILE = re.compile(r"^\d{4}$")
_EPOCH = date(1970, 1, 1)

KEYS = w137.KEYS
STORED_DATA = (
    "workout_date", "workout_time", "centre", "course", "furlongs", "split_first_s",
    "split_mid_s", "final_furlong_s", "effort", "partner",
)  # fmt: skip
STORED_SCHEMA = pa.schema(
    [
        ("horse_id", pa.string()), ("race_date", pa.date32()), ("workout_date", pa.date32()),
        ("workout_time", pa.string()), ("centre", pa.string()), ("course", pa.string()),
        ("furlongs", pa.int64()), ("split_first_s", pa.float64()),
        ("split_mid_s", pa.float64()), ("final_furlong_s", pa.float64()),
        ("effort", pa.string()), ("partner", pa.string()),
    ]
)  # fmt: skip
_SORT = ("workout_date", "centre", "course", "workout_time", "horse_id")
_TYPES = {
    "race_id": pl.String, "horse_no": pl.Int64, "horse_id": pl.String,
    "workout_date": pl.Date, "workout_time": pl.String, "centre": pl.String,
    "course": pl.String, "furlongs": pl.Int64, "split_first_s": pl.Float64,
    "split_mid_s": pl.Float64, "final_furlong_s": pl.Float64, "effort": pl.String,
    "partner": pl.String, "published_at": pl.Datetime("us", "UTC"),
}  # fmt: skip


def published_at(workout_date: date) -> datetime:
    """When a workout on `workout_date` counts as published: the end of its training day, JST."""
    return datetime.combine(workout_date + timedelta(days=1), time(0), tzinfo=JST)


# ---------------------------------------------------------------------------------------------
# HC and WC records -> workouts
# ---------------------------------------------------------------------------------------------


def _distinct(raw: np.ndarray, convert: Callable[[bytes], object], dtype) -> np.ndarray:
    """`convert` applied once per distinct value of the 1-d `raw`."""
    values, inverse = np.unique(raw, return_inverse=True)
    converted = np.empty(len(values), dtype=dtype)
    converted[:] = [convert(bytes(v)) for v in values]
    return converted[inverse.reshape(-1)]


def _code(raw: bytes) -> str:
    return raw.decode(records.ENCODING, errors="replace").strip(records.TRIM)


def _horse_id(raw: bytes) -> str | None:
    text = _code(raw)
    return None if not text or set(text) == {"0"} else text


def _day_number(raw: bytes) -> int:
    """YYYYMMDD -> days since 1970-01-01; -1 when blank or not a date."""
    if len(raw) != 8 or not raw.isdigit():
        return -1
    try:
        return (date(int(raw[:4]), int(raw[4:6]), int(raw[6:])) - _EPOCH).days
    except ValueError:
        return -1


def _clock(raw: bytes) -> str | None:
    text = raw.decode("ascii", errors="replace")
    return f"{text[:2]}:{text[2:]}" if len(text) == 4 and text.isdigit() else None


def _seconds(raw: np.ndarray) -> np.ndarray:
    """Tenths of a second -> seconds; blank, all zeros (no measure) or all nines (over the limit)
    NaN."""
    values = records.scaled(raw, 1)
    values[(values == 0) | (raw == b"9" * raw.dtype.itemsize)] = np.nan
    return values


def _floats(values: np.ndarray) -> pa.Array:
    return pa.array(values, type=pa.float64(), from_pandas=True)


def workouts_from_records(cols: Mapping[str, np.ndarray], course: str) -> pa.Table:
    """One stored row per HC or WC record of `cols` (a `records.decode` dict), STORED_SCHEMA."""
    n = len(cols[KUBUN])
    totals = {}
    for name, raw in cols.items():
        if (m := _TOTAL.search(name)) and int(m.group(1)) % 200 == 0:
            totals[int(m.group(1)) // 200] = _seconds(raw)
    final = [name for name in cols if _FINAL_LAP.match(name)]
    if not totals or len(final) != 1:
        raise ValueError(f"{course}: the layout has no furlong totals or no single final lap")
    ns = sorted(totals, reverse=True)
    matrix = np.column_stack([totals[k] for k in ns])
    known = ~np.isnan(matrix)
    measured = known.any(axis=1)
    longest = known.argmax(axis=1)
    days = _distinct(cols[DATE_ITEM], _day_number, np.int64)
    dates = pa.array(days, type=pa.int64(), mask=days < 0).cast(pa.int32()).cast(pa.date32())
    centre = _distinct(cols[CENTRE_ITEM], lambda b: CENTRES.get(_code(b), "other"), object)
    columns = {
        "horse_id": pa.array(_distinct(cols[HORSE_ITEM], _horse_id, object), pa.string()),
        "race_date": dates,
        "workout_date": dates,
        "workout_time": pa.array(_distinct(cols[TIME_ITEM], _clock, object), pa.string()),
        "centre": pa.array(centre, pa.string()),
        "course": pa.array(np.full(n, course, dtype=object), pa.string()),
        "furlongs": pa.array(np.array(ns)[longest], type=pa.int64(), mask=~measured),
        "split_first_s": _floats(np.where(measured, matrix[np.arange(n), longest], np.nan)),
        "split_mid_s": _floats(totals.get(3, np.full(n, np.nan))),
        "final_furlong_s": _floats(_seconds(cols[final[0]])),
        "effort": pa.nulls(n, pa.string()),
        "partner": pa.nulls(n, pa.string()),
    }
    return pa.table(columns, schema=STORED_SCHEMA)


def _year_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.jvd") if _YEAR_FILE.match(p.stem))


def _read_type(files: list[Path], record_type: str, course: str) -> tuple[pa.Table, dict, dict]:
    """The kept records of every year file of one record type, mapped; their counts; and the
    count of each トレセン区分 code other than miho's and ritto's (all versions)."""
    layout = spec.record_spec(record_type)
    keys = [item.name for item in layout.items if item.is_key]
    parts: list[pa.Table] = []
    raw: dict[str, list[np.ndarray]] = {name: [] for name in (*keys, CREATED, KUBUN)}
    centre_codes: Counter[str] = Counter()
    for path in files:
        for cols in records.iter_file(path, layout):
            parts.append(workouts_from_records(cols, course))
            for name in raw:
                raw[name].append(cols[name])
            codes, counts = np.unique(cols[CENTRE_ITEM], return_counts=True)
            for code, count in zip(codes, counts, strict=True):
                centre_codes[_code(bytes(code))] += int(count)
    if not parts:
        empty = STORED_SCHEMA.empty_table()
        counts = {"files": len(files), "records": 0, "versions_dropped": 0, "deletions": 0}
        return empty, counts, {}
    # Versions: every key item and the creation date as order-preserving integer codes (np.unique
    # sorts the bytes, the order of YYYYMMDD text), so the index stays small however many years.
    index = pd.DataFrame(
        {
            name: np.unique(np.concatenate(raw.pop(name)), return_inverse=True)[1].reshape(-1)
            for name in (*keys, CREATED)
        }
    )
    index[KUBUN] = records.text(np.concatenate(raw.pop(KUBUN)))
    distinct = len(index.drop_duplicates(keys))
    kept = records.latest_versions(index, keys).index.to_numpy()
    table = pa.concat_tables(parts).take(pa.array(kept))
    counts = {
        "files": len(files),
        "records": len(index),
        "versions_dropped": len(index) - distinct,
        "deletions": distinct - len(kept),
    }
    other = {code: n for code, n in sorted(centre_codes.items()) if code not in CENTRES}
    return table, counts, other


def build_workout_table(*, raw: Path | None = None, out: Path | None = None) -> records.BuildReport:
    """Read the export's HC and WC year files and write `<out>/workouts.parquet` (module
    docstring). Returns the counts, no record content."""
    out = paths.processed_dir(SOURCE) if out is None else paths.outside_repo(Path(out), what="out")
    root = Path(raw) if raw is not None else paths.raw_dir(SOURCE)
    found = {rt: _year_files(root.joinpath(*folder)) for rt, folder, _ in SOURCES}
    if not any(found.values()):
        folders = ", ".join(str(root.joinpath(*folder)) for _, folder, _ in SOURCES)
        raise FileNotFoundError(f"no HC or WC year file under {folders}")
    counts: dict[str, int] = {}
    unmapped: dict[str, dict[str, int]] = {}
    tables = []
    for record_type, _, course in SOURCES:
        table, read, other = _read_type(found[record_type], record_type, course)
        counts.update({f"{record_type}_{k}": v for k, v in read.items()})
        if other:
            unmapped[f"{record_type}_{CENTRE_ITEM}"] = other
        counts[course] = table.num_rows
        tables.append(table)
    table = pa.concat_tables(tables)
    table = table.sort_by([(c, "ascending") for c in _SORT])
    table = table.combine_chunks().replace_schema_metadata(None)
    counts["rows"] = table.num_rows
    counts["rows_without_horse_id"] = table.column("horse_id").null_count
    years = (
        pl.from_arrow(table.select(["workout_date", "course"]))
        .group_by(pl.col("workout_date").dt.year().alias("year"), "course")
        .len()
    )
    by_year: dict[int, dict[str, int]] = {}
    for year, course, n in sorted(years.drop_nulls("year").iter_rows()):
        by_year.setdefault(int(year), {})[course] = int(n)
    records.write_parquet(table, out / f"{TABLE}.parquet")
    return records.BuildReport(
        name=NAME, counts=counts, by_year=by_year, unmapped=unmapped, failures=[]
    )


def write_coverage_report() -> Path:
    """The stored table's coverage, per centre and course: the first and last workout date, the
    rows and the distinct horses, as `<reports dir>/jravan-workouts-coverage.json` through
    `write_report` (never the repository). Read through the scoped reader, so holdout days are not
    read."""
    frame = _read(["centre", "course", "workout_date", "horse_id"], date.max)
    rows = (
        frame.group_by("centre", "course")
        .agg(
            pl.col("workout_date").min().alias("first"),
            pl.col("workout_date").max().alias("last"),
            pl.len().alias("rows"),
            pl.col("horse_id").n_unique().alias("horses"),
        )
        .sort("centre", "course")
    )
    coverage = [
        {**r, "first": r["first"].isoformat(), "last": r["last"].isoformat()}
        for r in rows.to_dicts()
    ]
    text = json.dumps({"table": f"processed/{SOURCE}/{TABLE}.parquet", "coverage": coverage},
                      ensure_ascii=False, indent=2) + "\n"  # fmt: skip
    target = paths.reports_dir() / f"{SOURCE}-{TABLE}-coverage.json"
    return write_report(target, text, sources=(SOURCE,))


def workouts_command() -> None:
    """Parse the JRA-VAN export's HC (坂路) and WC (ウッド) records into
    processed/jravan/workouts; the counts in reports/jravan-workouts.json and the coverage per
    centre and course in reports/jravan-workouts-coverage.json."""
    try:
        report = build_workout_table()
        written = records.write_build_report(report)
        coverage = write_coverage_report()
    except (paths.DataDirNotSet, ValueError, LookupError, OSError) as exc:
        _die(f"jravan workouts failed: {type(exc).__name__}: {exc}")
    c = report.counts
    typer.echo(
        f"workouts: {c['rows']} rows ({c['hill']} hill, {c['wood']} wood); "
        f"{c['HC_versions_dropped'] + c['WC_versions_dropped']} older versions and "
        f"{c['HC_deletions'] + c['WC_deletions']} deletions dropped"
    )
    typer.echo(f"report: {written}")
    typer.echo(f"coverage: {coverage}")


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)


# ---------------------------------------------------------------------------------------------
# The as-of table and the features
# ---------------------------------------------------------------------------------------------


def _read(columns: list[str], end: date, where=None) -> pl.DataFrame:
    """The stored workouts dated HISTORY_START..end, through the scoped reader (holdout days
    dropped by workout date)."""
    table = read_store_table(
        TABLE, columns=columns, start=HISTORY_START, end=end, touches_price=False, where=where,
        source=SOURCE,
    )  # fmt: skip
    return pl.from_arrow(table)


def _with_published(frame: pl.DataFrame) -> pl.DataFrame:
    """`published_at` (UTC) of each record's workout_date."""
    days = frame.get_column("workout_date").drop_nulls().unique().to_list()
    lookup = pl.DataFrame(
        {"workout_date": days, "published_at": [published_at(d).astimezone(UTC) for d in days]},
        schema={"workout_date": pl.Date, "published_at": _TYPES["published_at"]},
    )
    return frame.join(lookup, on="workout_date", how="left")


_RECORD_COLUMNS = ["horse_id", *STORED_DATA]


def _records(horse_ids: list[str], day: date) -> pl.DataFrame:
    """The horses' stored workouts dated before race day `day`, with `published_at`."""
    if not horse_ids:
        return _with_published(pl.DataFrame(schema={c: _TYPES[c] for c in _RECORD_COLUMNS}))
    where = w137._is_in("horse_id", horse_ids)
    return _with_published(_read(_RECORD_COLUMNS, day - timedelta(days=1), where))


def _counted(records_: pl.DataFrame, entries: pl.DataFrame, day: date, as_of: datetime):
    """The entries' workouts that count on race day `day` at `as_of` (module docstring)."""
    horses = entries.get_column("horse_id").drop_nulls().unique().implode()
    return records_.filter(
        pl.col("horse_id").is_in(horses)
        & (pl.col("workout_date") < day)
        & (pl.col("published_at") < as_of)
    )


def _entries(day: date, race_id: str) -> pl.DataFrame:
    """The race's declared runners; none on a holdout race day (nothing is read)."""
    if is_holdout_day(COUNTRY, day):
        return pl.DataFrame(schema={c: _TYPES[c] for c in KEYS})
    return w137._entries(day, [race_id])


def workout_table(race_id: str, *, as_of: datetime) -> pl.DataFrame:
    """Each declared runner's counted workouts, by horse_no then newest first."""
    as_of = w137._utc(as_of)
    day = race_day_of(race_id)
    entries = _entries(day, race_id)
    horse_ids = entries.get_column("horse_id").drop_nulls().unique().to_list()
    counted = _counted(_records(horse_ids, day), entries, day, as_of)
    frame = entries.join(counted, on="horse_id", how="inner").sort(
        ["horse_no", "workout_date", "workout_time"], descending=[False, True, True]
    )
    return frame.select([pl.col(c).cast(t) for c, t in _TYPES.items()])


def workout_features(race_id: str, *, as_of: datetime) -> pl.DataFrame:
    """One row per declared runner: race_id, horse_no, horse_id and FEATURE_COLUMNS."""
    as_of = w137._utc(as_of)
    day = race_day_of(race_id)
    entries = _entries(day, race_id)
    recent = pl.DataFrame(
        schema={"horse_id": pl.String, "ff_best_vs_par_s": pl.Float64, "ff_trend_s": pl.Float64}
    )
    if entries.height:
        horse_ids = entries.get_column("horse_id").drop_nulls().unique().to_list()
        counted = _counted(_records(horse_ids, day), entries, day, as_of)
        pool = _with_published(
            _read(["workout_date", "centre", "course", "final_furlong_s"], day - timedelta(days=1))
        ).with_columns(pl.col("workout_date").alias("race_date"))
        pars = w137._pars(pool, day, as_of, PAR_MIN_WORKOUTS)
        recent = w137._best_and_trend(counted, pars, "workout_time")
    out = entries.join(recent, on="horse_id", how="left").with_columns(
        pl.lit(None, pl.Int64).alias("days_since_hard")
    )
    return out.select(
        *[pl.col(c).cast(_TYPES[c]) for c in KEYS],
        pl.col("ff_best_vs_par_s").cast(pl.Float64),
        pl.col("ff_trend_s").cast(pl.Float64),
        pl.col("days_since_hard").cast(pl.Int64),
    ).sort("horse_no")
