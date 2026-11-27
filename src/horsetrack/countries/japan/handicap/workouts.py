"""Miho and Ritto workouts as of decision time, and the workout features (#137).

Source and licence: JRDB, a paid subscription. CHA gives each runner's main pre-race workout
(本追切) for a race, CYB the per-course presence flags merged into `jrdb_pre_race` by #57. The
archives are the member downloads under `raw_dir("jrdb")`. JRDB's terms allow private use only:
no publication of the data, and predictions built on JRDB inputs may not be shown to third parties
(owner decision D21; `docs/ledger.md`). So this is an opt-in feature set behind
`HORSETRACK_JRDB_FEATURES`, its table and its log-loss report stay under the data dir, and the
repository holds no JRDB layout, code meaning or figure: `cha_workouts` maps the spec's fields by
name and reads the code meanings from `<data dir>/jrdb/codes/cha.csv` at run time.

Publication: JRDB documents CHA as delivered the evening before race day, later than its other
eve-of-race files, so the CHA and CYB records for race day D count as published at (D - 1) 20:00
Asia/Tokyo (`published_at`).

`build_workout_table()` parses every CHA delivery, joins it to the store and writes
`processed_dir("jra")/jrdb_workouts.parquet`; a delivery in an older layout than the spec's (the
earliest files lack the workout date) is skipped with a warning, and records of holdout race days
are not written (the store read drops them). `workout_table(race_id, as_of=...)` gives each
declared runner's workouts that count at `as_of`: the horse's records whose race day is on or before
the race day, whose workout date is not after it, and that were published strictly before `as_of`.
`workout_features(race_id, as_of=...)` derives, per runner:

- `ff_best_vs_par_s`: the best final-furlong time relative to its course par over the newest
  `N_RECENT` workouts that have one. Par is the median final furlong at that centre and course over
  every horse's records from earlier race days published before `as_of`, with at least
  `PAR_MIN_WORKOUTS` of them.
- `ff_trend_s`: the least-squares slope of those relative times, oldest to newest, in seconds per
  workout (null unless there are `N_RECENT`).
- `days_since_hard`: days from the newest hard workout (`HARD_EFFORTS`) to the race day.
- `course_kinds_since_last_race`: the sum of CYB's per-course presence flags of the runner's own
  row. It counts the course kinds used since the last race, not workouts (`docs/ledger.md`; a true
  workout count waits for another source, #229).

Every store read goes through #114's `read_store_table`, so holdout days are never read.
`write_logloss_report()` scores the v0 calibrated-market Win model with and without these features
out of fold on development days and writes aggregates only.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import warnings
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.optimize import minimize

from horsetrack.core import env, paths
from horsetrack.core.eval.folds import Fold, FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.store import schemas
from horsetrack.countries.japan.handicap.pp import race_day_of
from horsetrack.countries.japan.jrdb import join, parse
from horsetrack.countries.japan.jrdb import tables as jrdb_tables
from horsetrack.countries.japan.models.v0 import calibration as cal
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table

JST = ZoneInfo("Asia/Tokyo")
UTC = timezone.utc
FLAG = "HORSETRACK_JRDB_FEATURES"
TABLE = "jrdb_workouts"
REPORT = "workout-features-logloss.json"
HISTORY_START = date(1900, 1, 1)
PUBLISH_TIME = time(20, 0)  # on the day before race day, Asia/Tokyo
REPORT_AS_OF = time(9, 30)  # race morning: after publication, before the first post

CENTRES = ("miho", "ritto", "other")
COURSES = ("wood", "hill", "poly", "turf", "dirt", "other")
EFFORTS = ("all_out", "strong", "easy", "other")
HARD_EFFORTS = frozenset({"all_out", "strong"})
PARTNERS = ("ahead", "level", "behind")
N_RECENT = 3
PAR_MIN_WORKOUTS = 30
FEATURE_COLUMNS = (
    "ff_best_vs_par_s",
    "ff_trend_s",
    "days_since_hard",
    "course_kinds_since_last_race",
)
CYB_COUNT_COLUMNS = tuple(
    f"cyb_course_{c}" for c in ("slope", "wood", "dirt", "turf", "pool", "jump", "poly")
)

KEYS = ("race_id", "horse_no", "horse_id")
STORED_DATA = (
    "workout_date", "centre", "course", "furlongs", "split_first_s", "split_mid_s",
    "final_furlong_s", "effort", "partner",
)  # fmt: skip
_TYPES = {
    "race_id": pl.String, "horse_no": pl.Int64, "horse_id": pl.String,
    "workout_race_id": pl.String, "workout_date": pl.Date, "centre": pl.String,
    "course": pl.String, "furlongs": pl.Int64, "split_first_s": pl.Float64,
    "split_mid_s": pl.Float64, "final_furlong_s": pl.Float64, "effort": pl.String,
    "partner": pl.String, "published_at": pl.Datetime("us", "UTC"),
}  # fmt: skip
STORED_SCHEMA = pa.schema(
    [
        ("jurisdiction", pa.string()), ("race_id", pa.string()), ("race_date", pa.date32()),
        ("horse_no", pa.int64()), ("horse_id", pa.string()), ("workout_date", pa.date32()),
        ("centre", pa.string()), ("course", pa.string()), ("furlongs", pa.int64()),
        ("split_first_s", pa.float64()), ("split_mid_s", pa.float64()),
        ("final_furlong_s", pa.float64()), ("effort", pa.string()), ("partner", pa.string()),
    ]
)  # fmt: skip
_ON, _OFF = {"1", "true", "on", "yes"}, {"0", "false", "off", "no"}


def published_at(race_date: date) -> datetime:
    """When the CHA and CYB records delivered for race day `race_date` count as published."""
    return datetime.combine(race_date - timedelta(days=1), PUBLISH_TIME, tzinfo=JST)


def jrdb_features_enabled() -> bool:
    """`HORSETRACK_JRDB_FEATURES` (environment, else the repo-root `.env`); on when unset."""
    raw = env.get_setting(FLAG)
    value = (raw or "").strip().lower()
    if not value or value in _ON:
        return True
    if value in _OFF:
        return False
    raise ValueError(f"{FLAG}={raw!r}: use 1/true/on/yes or 0/false/off/no")


# ---------------------------------------------------------------------------------------------
# CHA -> jrdb_workouts
# ---------------------------------------------------------------------------------------------


def codes_path() -> Path:
    """The CHA code meanings (`field,code,value,label`), kept with the licensed specs."""
    return paths.data_dir() / "jrdb" / "codes" / "cha.csv"


def _codes() -> dict[str, dict[str, str]]:
    text = codes_path().read_text(encoding="utf-8")
    out: dict[str, dict[str, str]] = {}
    for row in csv.DictReader(line for line in text.splitlines() if not line.startswith("#")):
        out.setdefault(row["field"], {})[row["code"].strip()] = row["value"].strip()
    for code, value in out.get("course_code", {}).items():
        centre, _, course = value.partition(":")
        if centre not in CENTRES or course not in COURSES:
            raise ValueError(f"{codes_path()}: course code {code} maps to {value!r}")
    vocab = {"workout_kind": EFFORTS, "partner_result": PARTNERS}
    for field, allowed in vocab.items():
        bad = set(out.get(field, {}).values()) - set(allowed)
        if bad:
            raise ValueError(f"{codes_path()}: {field} maps to {sorted(bad)}")
    return out


def _code_key(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(int(value)) if isinstance(value, float) else str(value).strip()


def _seconds(values: pd.Series) -> pd.Series:
    """JRDB clock fields in tenths of a second -> seconds; blank or zero -> NaN."""
    tenths = pd.to_numeric(values, errors="coerce").astype("float64")
    return (tenths / 10.0).where(tenths > 0)


def cha_workouts(parsed: pd.DataFrame) -> pd.DataFrame:
    """Parsed CHA records -> race_key, horse_no and the stored data columns (module docstring)."""
    codes = _codes()
    places = [
        codes["course_code"].get(_code_key(c) or "", "other:other").split(":")
        for c in parsed["course_code"]
    ]
    efforts, partners = codes.get("workout_kind", {}), codes.get("partner_result", {})
    return pd.DataFrame(
        {
            "race_key": parsed["race_key"],
            "horse_no": parsed["horse_no"],
            "workout_date": parsed["workout_date"],
            "centre": pd.Series([p[0] for p in places], index=parsed.index, dtype=object),
            "course": pd.Series([p[1] for p in places], index=parsed.index, dtype=object),
            "furlongs": pd.to_numeric(parsed["furlongs"], errors="coerce").round().astype("Int64"),
            "split_first_s": _seconds(parsed["ten_f"]),
            "split_mid_s": _seconds(parsed["mid_f"]),
            "final_furlong_s": _seconds(parsed["last_f"]),
            "effort": parsed["workout_kind"].map(
                lambda v: efforts.get(_code_key(v) or "", "other")
            ),
            "partner": parsed["partner_result"].map(lambda v: partners.get(_code_key(v) or "")),
        },
        index=parsed.index,
    )


def _store_keys() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The store's races and runners, the join's columns only, through the scoped reader."""
    race_columns = [
        c
        for c in ("race_id", "race_date", "venue", "race_no", "kai", "nichi", "source_race_key")
        if c in schemas.TABLES["races"].columns
    ]
    scope = {"start": HISTORY_START, "end": date.max, "touches_price": False}
    races = read_store_table("races", columns=race_columns, **scope).to_pandas()
    runners = read_store_table(
        "runners", columns=["race_id", "horse_no", "horse_id"], **scope
    ).to_pandas()
    return races, runners


def build_workout_table() -> Path:
    """Every CHA delivery parsed, mapped, joined to the store and written (module docstring)."""
    archives = jrdb_tables.collect(["CHA"])
    if archives.lzh_only:
        raise ValueError("CHA: only .lzh archives, which cannot be read")
    frames, older = [parse.parse_bytes("CHA", b"")], []
    for delivery in archives.deliveries.get("CHA", []):
        try:
            frames.append(parse.parse_bytes("CHA", delivery.data))
        except parse.RecordLengthError:  # an edition before the spec's (no workout date)
            older.append(delivery.name)
    if older:
        message = f"CHA: skipped {len(older)} deliveries in an older layout, e.g. {older[0]}"
        warnings.warn(message, stacklevel=2)
    rows = cha_workouts(pd.concat(frames, ignore_index=True))
    races, runners = _store_keys()
    joined, _ = join.join_store(rows, races, runners)
    kept = joined[joined["matched_by"].notna()].copy()
    kept["jurisdiction"] = join.SOURCE
    kept["horse_no"] = pd.to_numeric(kept["horse_no"]).round().astype("Int64")
    kept["furlongs"] = pd.to_numeric(kept["furlongs"], errors="coerce").round().astype("Int64")
    kept = (
        kept.drop_duplicates(["race_id", "horse_no"], keep="last")  # the later delivery
        .sort_values(["race_id", "horse_no"], kind="stable")
        .reindex(columns=STORED_SCHEMA.names)
    )
    table = pa.Table.from_pandas(kept, schema=STORED_SCHEMA, preserve_index=False)
    path = paths.processed_dir(join.SOURCE) / f"{TABLE}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


# ---------------------------------------------------------------------------------------------
# The as-of table and the features
# ---------------------------------------------------------------------------------------------


def _utc(as_of: datetime) -> datetime:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    return as_of.astimezone(UTC)


def _read(name, columns, start, end, where=None) -> pl.DataFrame:
    table = read_store_table(
        name, columns=columns, start=start, end=end, touches_price=False, where=where
    )
    return pl.from_arrow(table)


def _is_in(column: str, values: list) -> ds.Expression:
    return ds.field(column).isin(pa.array(values, pa.string()))


def _entries(day: date, race_ids: list[str] | None = None) -> pl.DataFrame:
    """The declared (not scratched) runners of the day, or of the races named."""
    where = None if race_ids is None else _is_in("race_id", race_ids)
    frame = _read("runners", [*KEYS, "is_scratched"], day, day, where)
    return (
        frame.filter(~pl.col("is_scratched").fill_null(False))
        .select([pl.col(c).cast(_TYPES[c]) for c in KEYS])
        .sort(["race_id", "horse_no"])
    )


def _with_published(frame: pl.DataFrame) -> pl.DataFrame:
    """`published_at` (UTC) of each record's race_date."""
    days = frame.get_column("race_date").drop_nulls().unique().to_list()
    lookup = pl.DataFrame(
        {"race_date": days, "published_at": [published_at(d).astimezone(UTC) for d in days]},
        schema={"race_date": pl.Date, "published_at": _TYPES["published_at"]},
    )
    return frame.join(lookup, on="race_date", how="left")


_RECORD_COLUMNS = ["race_id", "race_date", "horse_id", *STORED_DATA]


def _records(horse_ids: list[str], end: date) -> pl.DataFrame:
    if not horse_ids:
        return _with_published(pl.DataFrame(schema=_record_schema()))
    frame = _read(TABLE, _RECORD_COLUMNS, HISTORY_START, end, _is_in("horse_id", horse_ids))
    return _with_published(frame.rename({"race_id": "workout_race_id"}))


def _record_schema() -> dict:
    out = {c: _TYPES.get(c, pl.Date) for c in _RECORD_COLUMNS}
    out["workout_race_id"] = out.pop("race_id")
    return out


def _counted(records: pl.DataFrame, entries: pl.DataFrame, day: date, as_of: datetime):
    """The entries' workouts that count on race day `day` at `as_of` (module docstring)."""
    horses = entries.get_column("horse_id").drop_nulls().unique().implode()
    return records.filter(
        pl.col("horse_id").is_in(horses)
        & (pl.col("race_date") <= day)
        & (pl.col("workout_date") <= day)
        & (pl.col("published_at") < as_of)
    )


def workout_table(race_id: str, *, as_of: datetime) -> pl.DataFrame:
    """Each declared runner's counted workouts, by horse_no then newest first."""
    as_of = _utc(as_of)
    day = race_day_of(race_id)
    entries = _entries(day, [race_id])
    horse_ids = entries.get_column("horse_id").drop_nulls().unique().to_list()
    counted = _counted(_records(horse_ids, day), entries, day, as_of)
    frame = entries.join(counted.drop("race_date"), on="horse_id", how="inner")
    frame = frame.sort(
        ["horse_no", "workout_date", "workout_race_id"], descending=[False, True, True]
    )
    return frame.select([pl.col(c).cast(t) for c, t in _TYPES.items()])


def _pars(pool: pl.DataFrame, day: date, as_of: datetime, min_workouts=None) -> pl.DataFrame:
    """Median final furlong per (centre, course) of `pool`'s records from before race day `day`
    published before `as_of`, where at least `min_workouts` (default `PAR_MIN_WORKOUTS`) have
    one."""
    if min_workouts is None:
        min_workouts = PAR_MIN_WORKOUTS
    return (
        pool.filter(
            (pl.col("race_date") < day)
            & (pl.col("published_at") < as_of)
            & pl.col("final_furlong_s").is_not_null()
            & pl.col("final_furlong_s").is_not_nan()
        )
        .group_by("centre", "course")
        .agg(pl.col("final_furlong_s").median().alias("par"), pl.len().alias("_n"))
        .filter(pl.col("_n") >= min_workouts)
        .select("centre", "course", "par")
    )


def _best_and_trend(counted: pl.DataFrame, pars: pl.DataFrame, tie_break: str) -> pl.DataFrame:
    """Per horse_id of `counted`: `ff_best_vs_par_s` and `ff_trend_s` over its newest `N_RECENT`
    workouts with a par in `pars` (newest by workout_date, then `tie_break`)."""
    n, xbar = N_RECENT, (N_RECENT - 1) / 2
    sxx = sum((k - xbar) ** 2 for k in range(n))
    recent = (
        counted.join(pars, on=["centre", "course"], how="inner")
        .with_columns((pl.col("final_furlong_s") - pl.col("par")).alias("rel"))
        .filter(pl.col("rel").is_not_null() & pl.col("rel").is_not_nan())
        .sort(["horse_id", "workout_date", tie_break], descending=[False, True, True])
        .with_columns(pl.int_range(pl.len()).over("horse_id").alias("_k"))
        .filter(pl.col("_k") < n)
        .with_columns((n - 1 - pl.col("_k") - xbar).alias("_x"))  # 0 = oldest
    )
    slope = (pl.col("_x") * pl.col("rel")).sum() / sxx if sxx > 0 else pl.lit(None)
    return recent.group_by("horse_id").agg(
        pl.col("rel").min().alias("ff_best_vs_par_s"),
        pl.when(pl.len() == n).then(slope).otherwise(None).alias("ff_trend_s"),
    )


def _features_from(entries, records, pool, counts, day: date, as_of: datetime) -> pl.DataFrame:
    """The features of `entries` (one race day) from unfiltered records, par pool and CYB rows."""
    counted = _counted(records, entries, day, as_of)
    recent = _best_and_trend(counted, _pars(pool, day, as_of), "race_date")
    hard = (
        counted.filter(pl.col("effort").is_in(sorted(HARD_EFFORTS)))
        .group_by("horse_id")
        .agg(pl.col("workout_date").max().alias("_hard"))
        .with_columns((pl.lit(day) - pl.col("_hard")).dt.total_days().alias("days_since_hard"))
    )
    columns = list(CYB_COUNT_COLUMNS)
    if published_at(day) < as_of:
        values = [pl.col(c).cast(pl.Float64).fill_nan(None) for c in columns]
        known = [v.is_not_null() for v in values]
        total = pl.sum_horizontal(values)
        cyb = counts.select(
            "race_id",
            pl.col("horse_no").cast(pl.Int64),
            pl.when(pl.any_horizontal(known))
            .then(total.round())
            .otherwise(None)
            .alias("course_kinds_since_last_race"),
        ).unique(["race_id", "horse_no"], keep="last")
    else:
        cyb = pl.DataFrame(
            schema={
                "race_id": pl.String,
                "horse_no": pl.Int64,
                "course_kinds_since_last_race": pl.Int64,
            }
        )
    out = (
        entries.join(recent, on="horse_id", how="left")
        .join(hard.select("horse_id", "days_since_hard"), on="horse_id", how="left")
        .join(cyb, on=["race_id", "horse_no"], how="left")
    )
    types = {"ff_best_vs_par_s": pl.Float64, "ff_trend_s": pl.Float64}
    return out.select(
        *[pl.col(c).cast(_TYPES[c]) for c in KEYS],
        *[pl.col(c).cast(types.get(c, pl.Int64)) for c in FEATURE_COLUMNS],
    ).sort(["race_id", "horse_no"])


def _counts(start: date, end: date, where=None) -> pl.DataFrame:
    return _read("jrdb_pre_race", ["race_id", "horse_no", *CYB_COUNT_COLUMNS], start, end, where)


def workout_features(race_id: str, *, as_of: datetime, enabled: bool | None = None):
    """One row per declared runner: its keys and, when the flag is on, FEATURE_COLUMNS."""
    as_of = _utc(as_of)
    day = race_day_of(race_id)
    if enabled is None:
        enabled = jrdb_features_enabled()
    entries = _entries(day, [race_id])
    if not enabled:
        return entries.sort("horse_no")
    horse_ids = entries.get_column("horse_id").drop_nulls().unique().to_list()
    records = _records(horse_ids, day)
    pool = _with_published(
        _read(TABLE, ["race_date", "centre", "course", "final_furlong_s"], HISTORY_START, day)
    )
    counts = _counts(day, day, ds.field("race_id") == race_id)
    return _features_from(entries, records, pool, counts, day, as_of).sort("horse_no")


# ---------------------------------------------------------------------------------------------
# The development-data log-loss report
# ---------------------------------------------------------------------------------------------

N_KNOTS = 5
RIDGE = 1e-4  # on the workout coefficients only, against singular fits


def _report_features(frame: pd.DataFrame) -> np.ndarray:
    """FEATURE_COLUMNS for every row of `frame`, as of each race morning, NaN where null."""
    if frame.empty:
        return np.empty((0, len(FEATURE_COLUMNS)))
    first, last = frame["race_date"].min().date(), frame["race_date"].max().date()
    runners = _read("runners", list(KEYS), first, last).with_columns(
        pl.col("horse_no").cast(pl.Int64)
    )
    records = _with_published(
        _read(TABLE, _RECORD_COLUMNS, HISTORY_START, last).rename({"race_id": "workout_race_id"})
    )
    counts = _counts(first, last)
    keys = pl.DataFrame(
        {
            "_i": np.arange(len(frame)),
            "race_id": frame["race_id"].astype(str).to_numpy(),
            "horse_no": frame["horse_no"].to_numpy().astype(np.int64),
            "_d": frame["race_date"].dt.date.to_numpy(),
        }
    )
    entries = keys.join(runners, on=["race_id", "horse_no"], how="left")
    parts = []
    for (day,), group in entries.group_by("_d", maintain_order=True):
        as_of = datetime.combine(day, REPORT_AS_OF, tzinfo=JST).astimezone(UTC)
        races = group.get_column("race_id").unique().implode()
        day_counts = counts.filter(pl.col("race_id").is_in(races))
        got = _features_from(group.select(KEYS), records, records, day_counts, day, as_of)
        parts.append(
            group.select("_i", "race_id", "horse_no").join(got, on=["race_id", "horse_no"])
        )
    feats = pl.concat(parts).sort("_i")
    return feats.select(FEATURE_COLUMNS).to_numpy().astype(float)


def _design(ln_q, raw, knots, stats) -> np.ndarray:
    """Spline basis in ln q, then each feature standardised (nulls at 0) and its null flag."""
    cols = [cal._basis(ln_q, knots)]
    for j, (mean, sd) in enumerate(stats):
        value = raw[:, j]
        missing = np.isnan(value)
        cols.append(np.where(missing, 0.0, (value - mean) / sd)[:, None])
        cols.append(missing.astype(float)[:, None])
    return np.hstack(cols)


def _transform(raw: np.ndarray) -> np.ndarray:
    out = raw.copy()
    out[:, 0] = np.clip(out[:, 0], -3.0, 3.0)
    out[:, 1] = np.clip(out[:, 1], -3.0, 3.0)
    out[:, 2] = np.log1p(np.clip(out[:, 2], 0.0, 365.0))
    return out


def _fit_logit(x, won, codes, n_free) -> np.ndarray:
    order = np.argsort(codes, kind="stable")
    x, won, codes = x[order], won[order], codes[order]
    starts, sizes = cal._segments(codes)
    n_races = len(starts)
    w = np.add.reduceat(won.astype(float), starts)
    w_rows = np.repeat(w, sizes)
    penalty = np.r_[np.zeros(n_free), np.full(x.shape[1] - n_free, RIDGE)]

    def fun(beta):
        value, grad, _ = cal._nll(beta, x, won, w, w_rows, starts, sizes, n_races)
        return value + 0.5 * float(penalty @ beta**2), grad + penalty * beta

    def hess(beta):
        _, _, p = cal._nll(beta, x, won, w, w_rows, starts, sizes, n_races)
        xp = x * p[:, None]
        mean = np.add.reduceat(xp, starts, axis=0)
        h = ((xp * w_rows[:, None]).T @ x - (mean * w[:, None]).T @ mean) / n_races
        return h + np.diag(penalty)

    beta0 = np.zeros(x.shape[1])
    beta0[0] = 1.0
    result = minimize(fun, beta0, jac=True, hess=hess, method="trust-exact", options={"gtol": 1e-9})
    if not (result.success or np.max(np.abs(result.jac)) < cal.GRAD_TOL):
        raise RuntimeError(f"the workout-feature fit did not converge: {result.message}")
    return result.x


def _workout_oof(frame: pd.DataFrame, raw: np.ndarray, folds: list[Fold]) -> np.ndarray:
    """Out-of-fold P(win) of the v0 model with the workout features added (NaN outside folds)."""
    codes = pd.factorize(frame["race_id"])[0]
    ln_q = np.log(cal._q_by_race(frame["win_odds"].to_numpy(dtype=float), codes))
    won = frame["won"].to_numpy(dtype=bool)
    raw = _transform(raw)
    p = np.full(len(frame), np.nan)
    for fold in folds:
        train, test = np.asarray(fold.train_idx), np.asarray(fold.test_idx)
        if len(test) == 0:
            continue
        knots = np.quantile(ln_q[train], np.linspace(*cal.QUANTILE_RANGE, N_KNOTS))
        stats = []  # training mean and sd of each feature's known values
        for j in range(raw.shape[1]):
            seen = raw[train, j][~np.isnan(raw[train, j])]
            sd = float(seen.std()) if len(seen) > 1 else 0.0
            stats.append((float(seen.mean()) if len(seen) else 0.0, sd if sd > 0 else 1.0))
        n_free = knots.size - 1
        beta = _fit_logit(_design(ln_q[train], raw[train], knots, stats), won[train],
                          codes[train], n_free)  # fmt: skip
        score = _design(ln_q[test], raw[test], knots, stats) @ beta
        p[test] = cal._group_softmax(score, codes[test])
    return p


def _race_losses(frame: pd.DataFrame, p: np.ndarray) -> pd.Series:
    winners = frame["won"].to_numpy(dtype=bool) & ~np.isnan(p)
    return pd.Series(-np.log(p[winners]), index=frame["race_id"].to_numpy()[winners])


def write_logloss_report(*, scheme: FoldScheme | None = None, out_dir: Path | None = None) -> Path:
    """The v0 Win model's out-of-fold log loss with and without the workout features on
    development days, written as aggregates only (module docstring)."""
    out = (
        paths.reports_dir()
        if out_dir is None
        else paths.outside_repo(Path(out_dir), what="out_dir")
    )
    scheme = scheme or load_country_scheme("japan")
    table = read_store_table(
        "runners", columns=cal.STORE_COLUMNS, start=PRICE_ERA_START, end=date.max,
        touches_price=True,
    )  # fmt: skip
    frame, _, _ = cal._development_frame(table)
    del table
    frame["race_id"] = frame["race_id"].astype(str)
    frame = frame.sort_values(["race_date", "race_id", "horse_no"], ignore_index=True)
    folds = walk_forward(frame["race_date"], scheme=scheme)

    oof = cal.oof_predict(frame, folds=folds, n_knots=N_KNOTS)
    keys = frame[["race_id", "horse_no"]]
    base = keys.merge(oof[["race_id", "horse_no", "p_cal"]], on=["race_id", "horse_no"], how="left")
    base = base["p_cal"].to_numpy(dtype=float)
    workout = _workout_oof(frame, _report_features(frame), folds)
    base_loss, work_loss = _race_losses(frame, base), _race_losses(frame, workout)

    def entry(races: pd.Index) -> dict:
        b, w = base_loss.loc[races], work_loss.loc[races]
        bl, wl = (float(b.mean()), float(w.mean())) if len(races) else (None, None)
        gain = bl - wl if len(races) else None
        return {"n_races": int(len(races)), "baseline_logloss": bl, "workout_logloss": wl,
                "logloss_gain": gain}  # fmt: skip

    fold_entries = []
    for fold in folds:
        races = pd.Index(frame["race_id"].to_numpy()[np.asarray(fold.test_idx)]).unique()
        races = races[races.isin(base_loss.index)]
        fold_entries.append(
            {"test_start": fold.test_start.isoformat(), "test_end": fold.test_end.isoformat(),
             **entry(races)}
        )  # fmt: skip
    report = {
        **entry(base_loss.index),
        "features": list(FEATURE_COLUMNS),
        "folds": fold_entries,
        "model": "win_calibration_v0 + workout features (conditional logit)",
        "fold_scheme": dataclasses.asdict(scheme),
        "as_of": f"race day {REPORT_AS_OF.isoformat('minutes')} Asia/Tokyo",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / REPORT
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path
