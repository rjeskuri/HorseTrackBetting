"""Each runner's past-performance (PP) lines as of decision time (#128).

`pp_lines(race_id, as_of=..., n=10)` gives one row per prior start of each declared (not scratched)
runner of a race, newest first, at most `n` per runner; a runner with none gets one row with
`reason` "debut". A prior start is a not-scratched `runners` row of the same horse on an earlier
race day whose result was published before `as_of`. The store holds no publication time, so a
result counts as published at (race_date + 1 day) 00:00 Asia/Tokyo, the end of its local race day.

Every read goes through #114's scoped reader (`read_store_table`): named columns, a date range and
a row filter pushed into the scan, holdout days dropped. History is scoped to the race's horses,
never the whole table. Odds and popularity are read only inside D10's price era (2014-06-07 on);
an older start keeps its line with those two null. Both functions take `source` (#404), the store
copy read: 'jra' by default, 'jravan' the JRA-VAN copy.

`build_pp_table(date_from, date_to)` writes one Parquet file per race day under the data dir,
`features/<source>/pp_lines/<YYYY-MM-DD>.parquet` (`features/jra/...` by default), building one
day at a time.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds

from horsetrack.core.paths import data_dir
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, SOURCE, read_store_table

JST = ZoneInfo("Asia/Tokyo")
DEFAULT_N = 10
HISTORY_START = date(1900, 1, 1)  # no lower bound on a horse's history
_RACE_DAY = re.compile(r"^jra:(\d{4}-\d{2}-\d{2}):")

RACE_COLUMNS = (
    "venue", "surface", "distance_m", "course_config", "going", "race_class", "field_size",
)  # fmt: skip
RUNNER_COLUMNS = (
    "draw", "actual_wt_kg", "horse_wt_kg", "horse_wt_change_kg", "jockey_key", "jockey_name",
    "trainer_key", "trainer_name", "finish_pos", "lbw_lengths", "finish_time_s", "last_3f_s",
    "running_positions",
)  # fmt: skip
PRICE_COLUMNS = ("win_odds", "win_fav_rank")

_STR, _INT, _FLOAT = pl.String, pl.Int64, pl.Float64
SCHEMA: dict[str, pl.DataType] = {
    "race_id": _STR, "horse_no": _INT, "horse_id": _STR, "line_no": _INT, "reason": _STR,
    "pp_race_id": _STR, "pp_date": pl.Date, "venue": _STR, "surface": _STR, "distance_m": _INT,
    "course_config": _STR, "going": _STR, "race_class": _STR, "field_size": _INT, "draw": _INT,
    "actual_wt_kg": _FLOAT, "horse_wt_kg": _INT, "horse_wt_change_kg": _INT, "jockey_key": _STR,
    "jockey_name": _STR, "trainer_key": _STR, "trainer_name": _STR, "finish_pos": _INT,
    "lbw_lengths": _FLOAT, "finish_time_s": _FLOAT, "last_3f_s": _FLOAT, "last_3f_rank": _INT,
    "running_positions": pl.List(_INT), "win_odds": _FLOAT, "win_fav_rank": _INT,
    "days_since_prev": _INT, "winner_time_s": _FLOAT,
}  # fmt: skip
ENTRY_KEYS = ("race_id", "horse_no", "horse_id")
LINE_COLUMNS = tuple(c for c in SCHEMA if c not in (*ENTRY_KEYS, "line_no", "reason"))


def race_day_of(race_id: str) -> date:
    """The race day in a store race id, `jra:<YYYY-MM-DD>:<VENUE>:<NN>`."""
    match = _RACE_DAY.match(race_id)
    if not match:
        raise ValueError(f"not a JRA store race id: {race_id!r}")
    return date.fromisoformat(match.group(1))


def published_at(race_date: date) -> datetime:
    """When a start's result counts as published: the end of its local race day."""
    return datetime.combine(race_date + timedelta(days=1), time(0), tzinfo=JST)


def last_published_day(as_of: datetime) -> date:
    """The latest race day whose result is published strictly before `as_of`."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    local = as_of.astimezone(JST)
    day = local.date() - timedelta(days=1)
    return day if published_at(day) < as_of else day - timedelta(days=1)


def pp_lines(
    race_id: str, *, as_of: datetime, n: int = DEFAULT_N, source: str = SOURCE
) -> pl.DataFrame:
    """The PP lines of every declared runner of `race_id` as of `as_of`, newest first, read from
    the store copy `source`."""
    day = race_day_of(race_id)
    last_day = min(day - timedelta(days=1), last_published_day(as_of))
    entries = _entries(day, where=ds.field("race_id") == race_id, source=source)
    return _lines_for(entries, last_day=last_day, n=n, source=source)


def build_pp_table(date_from: date, date_to: date, *, source: str = SOURCE) -> dict[date, Path]:
    """One Parquet file of `pp_lines` (default n, as of the race day) per non-holdout race day in
    the inclusive range of the store copy `source`, written under the data dir's
    `features/<source>/pp_lines` one day at a time."""
    races = read_store_table(
        "races",
        columns=["race_date"],
        start=date_from,
        end=date_to,
        touches_price=False,
        source=source,
    )
    days = sorted({d for d in races.column("race_date").to_pylist() if d is not None})
    out_dir = data_dir() / "features" / source / "pp_lines"
    written: dict[date, Path] = {}
    for day in days:
        frame = _lines_for(
            _entries(day, source=source),
            last_day=day - timedelta(days=1),
            n=DEFAULT_N,
            source=source,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{day.isoformat()}.parquet"
        tmp = path.with_name(path.name + ".tmp")
        frame.write_parquet(tmp)
        tmp.replace(path)
        written[day] = path
        del frame
    return written


def _read(
    name: str,
    columns: Sequence[str],
    start: date,
    end: date,
    *,
    where: ds.Expression | None = None,
    touches_price: bool = False,
    source: str = SOURCE,
) -> pl.DataFrame:
    table = read_store_table(
        name,
        columns=columns,
        start=start,
        end=end,
        touches_price=touches_price,
        where=where,
        source=source,
    )
    return pl.from_arrow(table)


def _entries(
    day: date, *, where: ds.Expression | None = None, source: str = SOURCE
) -> pl.DataFrame:
    """The day's declared runners (not scratched), optionally narrowed to one race."""
    frame = _read("runners", [*ENTRY_KEYS, "is_scratched"], day, day, where=where, source=source)
    return frame.filter(~pl.col("is_scratched").fill_null(False)).select(ENTRY_KEYS)


def _is_in(column: str, values: Sequence[str]) -> ds.Expression:
    return ds.field(column).isin(pa.array(list(values), pa.string()))


def _lines_for(
    entries: pl.DataFrame, *, last_day: date, n: int, source: str = SOURCE
) -> pl.DataFrame:
    """The entries joined to their horses' starts on or before `last_day`, at most n each."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    lines = _empty_lines()
    horse_ids = entries.get_column("horse_id").drop_nulls().unique().to_list()
    if horse_ids and last_day >= HISTORY_START:
        lines = _history(horse_ids, last_day=last_day, n=n, source=source)
    frame = (
        entries.join(lines, on="horse_id", how="left")
        .with_columns(
            pl.when(pl.col("line_no").is_null())
            .then(pl.lit("debut"))
            .otherwise(None)
            .alias("reason")
        )
        .select([pl.col(c).cast(t) for c, t in SCHEMA.items()])
    )
    return frame.sort(["race_id", "horse_no", "line_no"], nulls_last=True)


def _history(horse_ids: list[str], *, last_day: date, n: int, source: str = SOURCE) -> pl.DataFrame:
    """Each horse's newest n starts on or before `last_day`, with line_no and days_since_prev."""
    starts = (
        _read(
            "runners",
            ["horse_id", "race_id", "race_date", "is_scratched", *RUNNER_COLUMNS],
            HISTORY_START,
            last_day,
            where=_is_in("horse_id", horse_ids),
            source=source,
        )
        .filter(~pl.col("is_scratched").fill_null(False))
        .sort(["horse_id", "race_date", "race_id"], descending=[False, True, True])
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("horse_id").alias("line_no"),
            (pl.col("race_date") - pl.col("race_date").shift(-1).over("horse_id"))
            .dt.total_days()
            .alias("days_since_prev"),
        )
        .filter(pl.col("line_no") <= n)
        .rename({"race_id": "pp_race_id", "race_date": "pp_date"})
        .drop("is_scratched")
    )
    if starts.height == 0:
        return _empty_lines()
    race_ids = starts.get_column("pp_race_id").unique().to_list()
    first, last = starts.get_column("pp_date").min(), starts.get_column("pp_date").max()
    in_races = _is_in("race_id", race_ids)

    races = _read(
        "races",
        ["race_id", *RACE_COLUMNS, "finish_time_s"],
        first,
        last,
        where=in_races,
        source=source,
    ).rename({"race_id": "pp_race_id", "finish_time_s": "winner_time_s"})
    ranks = (
        _read(
            "runners",
            ["race_id", "horse_id", "last_3f_s", "is_scratched"],
            first,
            last,
            where=in_races,
            source=source,
        )  # fmt: skip
        .filter(~pl.col("is_scratched").fill_null(False) & pl.col("last_3f_s").is_not_null())
        .with_columns(pl.col("last_3f_s").rank("min").over("race_id").alias("last_3f_rank"))
        .select(pl.col("race_id").alias("pp_race_id"), "horse_id", "last_3f_rank")
    )
    starts = starts.join(races, on="pp_race_id", how="left").join(
        ranks, on=["pp_race_id", "horse_id"], how="left"
    )
    price_first = max(first, PRICE_ERA_START)
    if last >= price_first:
        prices = _read(
            "runners",
            ["race_id", "horse_id", *PRICE_COLUMNS],
            price_first,
            last,
            where=in_races & _is_in("horse_id", horse_ids),
            touches_price=True,
            source=source,
        ).rename({"race_id": "pp_race_id"})
        starts = starts.join(prices, on=["pp_race_id", "horse_id"], how="left")
    else:
        starts = starts.with_columns(pl.lit(None).alias(c) for c in PRICE_COLUMNS)
    return starts.select(
        "horse_id", *[pl.col(c).cast(SCHEMA[c]) for c in ("line_no", *LINE_COLUMNS)]
    )


def _empty_lines() -> pl.DataFrame:
    return pl.DataFrame(
        schema={"horse_id": pl.String, **{c: SCHEMA[c] for c in ("line_no", *LINE_COLUMNS)}}
    )
