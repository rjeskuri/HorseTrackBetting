"""Compact a day's odds logs into one Parquet time series (ticket #32).

Reads every `<VENUE>.jsonl.gz` log and `<VENUE>.races.json` post-time sidecar under
`odds_snapshots/<source>/<YYYY-MM-DD>/` (docs/specs/odds-log.md) and writes
`processed/<source>/odds_timeseries/<YYYY-MM-DD>.parquet`. The logs are the record; the Parquet is
derived and rebuilt whole on every run.

Columns, one row per logged row (pool-total `__pool__` rows, tier-2 and WIN5 rows included). The
logs are streamed member by member and the day is held as one Arrow table per member, not one
Python dict per row (a tier-2 day runs to millions of rows):

- race_id, venue, race_no, pool, combination: venue is the log file's; race_no from the race_id.
- snapshot_utc, fetched_utc: as logged, tz-aware UTC; fetched_utc null when absent or null.
- post_utc: the *scheduled* post time from the venue's own sidecar, by race number; not the actual
  off. Where the sidecar has no entry for the race (or there is no sidecar), the meeting's stored
  race card (`processed/jra/upcoming/jra_<date>_<VENUE>/races.parquet`, `post_time_local`, Tokyo
  time without an offset; `board.snapshot.card_post_times`; source "jra" only). Null when
  neither gives one.
- minutes_to_post: (post_utc - (fetched_utc or snapshot_utc)) in minutes, against that scheduled
  time; negative after the scheduled post, null without a post time.
- odds, odds_hi, investment, status_raw: as logged; null when absent.
- is_final_capture: the row with the latest snapshot_utc per (race_id, pool, combination), pool
  totals included; at the same instant the later logged row wins, as `odds_log.latest_prices`.
- dividend, is_refund: left join on exact (race_id, pool, combination) to
  `processed/<source>/dividends.parquet`; null without a match or without the table.
- is_final_state (`DAY_SCHEMA` only): the logged row shows the page's final state
  (docs/specs/board-at-t.md, `board.snapshot._is_final`: `source_update_utc` null and 最終オッズ
  in `page_state` or `status_raw`), the two fields compaction does not keep (#411). The day file
  is `DAY_SCHEMA`, `SCHEMA` (shared with the JRA-VAN series, `jravan/timeseries.py`, which records
  no page state) plus this column; a reader of `SCHEMA`'s columns reads either.

Identical logs give identical bytes: rows are sorted on their content, the schema is fixed, and the
file carries no wall-clock metadata.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import typer

from horsetrack.core import paths
from horsetrack.countries.japan.board.snapshot import _is_final, card_post_times
from horsetrack.countries.japan.odds_log import _instant, iter_members, read_post_times

_LOG_SUFFIX = ".jsonl.gz"
_SIDECAR_SUFFIX = ".races.json"
_JOIN_KEY = ("race_id", "pool", "combination")
_UTC = pa.timestamp("us", tz="UTC")

SCHEMA = pa.schema(
    [
        ("race_id", pa.string()),
        ("venue", pa.string()),
        ("race_no", pa.int64()),
        ("pool", pa.string()),
        ("combination", pa.string()),
        ("snapshot_utc", _UTC),
        ("fetched_utc", _UTC),
        ("post_utc", _UTC),
        ("minutes_to_post", pa.float64()),  # to the scheduled post, not the actual off
        ("odds", pa.float64()),
        ("odds_hi", pa.float64()),
        ("investment", pa.float64()),
        ("status_raw", pa.string()),
        ("is_final_capture", pa.bool_()),
        ("dividend", pa.float64()),
        ("is_refund", pa.bool_()),
    ]
)
# Our logged day files: SCHEMA plus the row's final-state flag (#411).
DAY_SCHEMA = SCHEMA.append(pa.field("is_final_state", pa.bool_()))
# The columns a logged row gives on its own; the rest need the whole day (or the dividends).
_LOGGED = pa.schema(
    [f for f in DAY_SCHEMA if f.name not in ("is_final_capture", "dividend", "is_refund")]
)
# A total order on the row's content, so the bytes do not depend on file or log order.
_SORT_COLUMNS = (
    "race_id",
    "pool",
    "combination",
    "snapshot_utc",
    "venue",
    "fetched_utc",
    "is_final_capture",
    "odds",
    "odds_hi",
    "investment",
    "status_raw",
    "is_final_state",
)


class NoLogs(FileNotFoundError):
    """The day has no odds logs to compact."""


def day_file(race_date: str, source: str = "jra") -> Path:
    return paths.processed_dir(source) / "odds_timeseries" / f"{race_date}.parquet"


def compact_day(race_date: str, source: str = "jra") -> Path:
    """Rebuild the day's Parquet time series from its logs and return its path.

    Raises NoLogs (a FileNotFoundError) and writes nothing when the day has no logs, and ValueError
    when the dividends table holds one (race_id, pool, combination) twice for a race of the day."""
    day = date.fromisoformat(str(race_date)).isoformat()
    day_dir = paths.odds_snapshot_path(source, day, "_").parent
    logs = sorted(day_dir.glob(f"*{_LOG_SUFFIX}")) if day_dir.is_dir() else []
    if not logs:
        raise NoLogs(f"no odds logs for {day} under {day_dir}")

    # One Arrow table per member, so the day is held column-wise (tier-2 days run to millions of
    # rows); only the latest (instant, log position) per key and the race ids are kept in Python.
    parts: list[pa.Table] = []
    latest: dict[tuple, tuple[datetime, int]] = {}
    race_ids: set[Any] = set()
    position = 0
    for log_path in logs:
        venue = log_path.name[: -len(_LOG_SUFFIX)]
        post_times = _post_times(day, venue, source, log_path.with_name(venue + _SIDECAR_SUFFIX))
        for member in iter_members(log_path):
            records = [_record(row, venue, post_times) for row in member]
            for rec in records:
                _note_latest(latest, rec, position)
                race_ids.add(rec["race_id"])
                position += 1
            if records:
                parts.append(pa.Table.from_pylist(records, schema=_LOGGED))
    race_ids.discard(None)

    finals = {at_position for _, at_position in latest.values()}
    del latest
    dividends = _read_dividends(paths.processed_dir(source) / "dividends.parquet", race_ids)
    start = 0
    for i, part in enumerate(parts):
        parts[i] = _with_derived(part, start, finals, dividends)
        start += part.num_rows
    table = pa.concat_tables(parts) if parts else DAY_SCHEMA.empty_table()
    del parts
    # null last and strings in code-point order, as a Python sort on (is None, value) would give
    table = table.sort_by([(name, "ascending") for name in _SORT_COLUMNS])
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="snappy")
    target = day_file(day, source)
    _write_atomic(target, sink.getvalue().to_pybytes())
    return target


def _post_times(day: str, venue: str, source: str, sidecar: Path) -> dict[int, datetime]:
    """Race number -> scheduled post (aware UTC): the venue's sidecar entry, else (JRA only) the
    meeting's stored race card."""
    card = card_post_times(date.fromisoformat(day), venue) if source == "jra" else {}
    return card | read_post_times(sidecar)


def _race_no(race_id: Any) -> int | None:
    try:
        return int(str(race_id).rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


def _record(row: dict[str, Any], venue: str, post_times: dict[int, datetime]) -> dict[str, Any]:
    snapshot = _instant(row.get("snapshot_utc"))
    fetched = _instant(row.get("fetched_utc"))
    race_no = _race_no(row.get("race_id"))
    post = post_times.get(race_no) if race_no is not None else None
    origin = fetched or snapshot
    minutes = (post - origin) / timedelta(minutes=1) if post and origin else None
    return {
        "race_id": row.get("race_id"),
        "venue": venue,
        "race_no": race_no,
        "pool": row.get("pool"),
        "combination": row.get("combination"),
        "snapshot_utc": snapshot,
        "fetched_utc": fetched,
        "post_utc": post,
        "minutes_to_post": minutes,
        "odds": _number(row.get("odds")),
        "odds_hi": _number(row.get("odds_hi")),
        "investment": _number(row.get("investment")),
        "status_raw": row.get("status_raw"),
        "is_final_state": _is_final(row),
    }


def _note_latest(latest: dict[tuple, tuple[datetime, int]], rec: dict[str, Any], at: int) -> None:
    """Keep the latest row per key by its log position; rows come in log order (venues by name),
    so `>=` lets the later logged row win a tie. Rows without a readable snapshot_utc are never
    final."""
    snapshot = rec["snapshot_utc"]
    if snapshot is None:
        return
    key = tuple(rec[field] for field in _JOIN_KEY)
    current = latest.get(key)
    if current is None or snapshot >= current[0]:
        latest[key] = (snapshot, at)


_Dividends = dict[tuple, tuple[float | None, bool | None]]


def _read_dividends(path: Path, race_ids: set[Any]) -> _Dividends:
    """(race_id, pool, combination) -> (dividend, is_refund) for the day's races; ValueError when a
    key appears twice."""
    if not path.exists() or not race_ids:
        return {}
    table = pq.read_table(path, columns=[*_JOIN_KEY, "dividend", "is_refund"])
    table = table.filter(pc.is_in(table["race_id"], value_set=pa.array(sorted(race_ids))))
    dividends: _Dividends = {}
    for row in table.to_pylist():
        key = tuple(row[field] for field in _JOIN_KEY)
        if key in dividends:
            raise ValueError(f"{path}: (race_id, pool, combination) {key} appears more than once")
        refund = row["is_refund"]
        dividends[key] = (_number(row["dividend"]), None if refund is None else bool(refund))
    return dividends


def _with_derived(part: pa.Table, start: int, finals: set[int], dividends: _Dividends) -> pa.Table:
    """A member's table (rows at log positions start...) with is_final_capture and the dividend
    join added, in DAY_SCHEMA's column order."""
    final = [start + i in finals for i in range(part.num_rows)]
    if dividends:
        keys = zip(*(part[field].to_pylist() for field in _JOIN_KEY), strict=True)
        matches = [dividends.get(key, (None, None)) for key in keys]
        dividend, refund = [m[0] for m in matches], [m[1] for m in matches]
    else:
        dividend = refund = [None] * part.num_rows
    columns = {
        "is_final_capture": pa.chunked_array([pa.array(final, pa.bool_())]),
        "dividend": pa.chunked_array([pa.array(dividend, pa.float64())]),
        "is_refund": pa.chunked_array([pa.array(refund, pa.bool_())]),
    }
    return pa.table(
        [part[name] if name in part.column_names else columns[name] for name in DAY_SCHEMA.names],
        schema=DAY_SCHEMA,
    )


def _write_atomic(target: Path, data: bytes) -> None:
    """A temporary file in the target's directory, fsynced, then os.replace; a failure leaves the
    previous file as it was and no temporary file behind."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------- the CLI


def _parse_day(value: str, option: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"{value!r} is not a YYYY-MM-DD date", param_hint=option) from exc


def compact(
    race_date: str | None = typer.Option(None, "--date", help="Compact one day (YYYY-MM-DD)."),
    first: str | None = typer.Option(None, "--from", help="First day of a range (YYYY-MM-DD)."),
    last: str | None = typer.Option(None, "--to", help="Last day of a range (YYYY-MM-DD)."),
) -> None:
    """Compact a day's odds logs into processed/jra/odds_timeseries/<date>.parquet."""
    in_range = first is not None or last is not None
    if (race_date is None) == (not in_range):
        raise typer.BadParameter("give either --date or both --from and --to")
    if race_date is not None:
        day = _parse_day(race_date, "--date")
        try:
            typer.echo(compact_day(day.isoformat()))
        except NoLogs as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1) from exc
        return
    if first is None or last is None:
        raise typer.BadParameter("--from and --to go together")
    start, end = _parse_day(first, "--from"), _parse_day(last, "--to")
    if start > end:
        raise typer.BadParameter(f"--from {start} is after --to {end}")
    day = start
    while day <= end:
        try:
            typer.echo(compact_day(day.isoformat()))
        except NoLogs:
            typer.echo(f"skipped {day}: no logs")
        day += timedelta(days=1)
