"""Compact a day's odds logs into one Parquet time series (ticket #32).

Reads every `<VENUE>.jsonl.gz` log and `<VENUE>.races.json` post-time sidecar under
`odds_snapshots/<source>/<YYYY-MM-DD>/` (docs/specs/odds-log.md) and writes
`processed/<source>/odds_timeseries/<YYYY-MM-DD>.parquet`. The logs are the record; the Parquet is
derived and rebuilt whole on every run.

Columns, one row per logged row (pool-total `__pool__` rows included):

- race_id, venue, race_no, pool, combination: venue is the log file's; race_no from the race_id.
- snapshot_utc, fetched_utc: as logged, tz-aware UTC; fetched_utc null when absent or null.
- post_utc: the *scheduled* post time from the venue's own sidecar, by race number; not the actual
  off. Null when the venue has no sidecar or the sidecar has no entry for the race.
- minutes_to_post: (post_utc - (fetched_utc or snapshot_utc)) in minutes, against that scheduled
  time; negative after the scheduled post, null without a post time.
- odds, odds_hi, investment, status_raw: as logged; null when absent.
- is_final_capture: the row with the latest snapshot_utc per (race_id, pool, combination), pool
  totals included; at the same instant the later logged row wins, as `odds_log.latest_prices`.
- dividend, is_refund: left join on exact (race_id, pool, combination) to
  `processed/<source>/dividends.parquet`; null without a match or without the table.

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
from horsetrack.countries.japan.odds_log import _instant, read_post_times, read_rows

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

    records: list[dict[str, Any]] = []
    for log_path in logs:
        venue = log_path.name[: -len(_LOG_SUFFIX)]
        post_times = read_post_times(log_path.with_name(venue + _SIDECAR_SUFFIX))
        for row in read_rows(log_path):
            records.append(_record(row, venue, post_times))

    _flag_final_captures(records)
    _join_dividends(records, paths.processed_dir(source) / "dividends.parquet")
    records.sort(key=_sort_key)

    table = pa.Table.from_pylist(
        [{name: rec[name] for name in SCHEMA.names} for rec in records], schema=SCHEMA
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="snappy")
    target = day_file(day, source)
    _write_atomic(target, sink.getvalue().to_pybytes())
    return target


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
        "is_final_capture": False,
        "dividend": None,
        "is_refund": None,
    }


def _flag_final_captures(records: list[dict[str, Any]]) -> None:
    """Flag the latest row per key; records are in log order (venues by name), so `>=` lets the
    later logged row win a tie. Rows without a readable snapshot_utc are never final."""
    best: dict[tuple, dict[str, Any]] = {}
    for rec in records:
        at = rec["snapshot_utc"]
        if at is None:
            continue
        key = tuple(rec[field] for field in _JOIN_KEY)
        current = best.get(key)
        if current is None or at >= current["snapshot_utc"]:
            best[key] = rec
    for rec in best.values():
        rec["is_final_capture"] = True


def _join_dividends(records: list[dict[str, Any]], path: Path) -> None:
    if not path.exists() or not records:
        return
    table = pq.read_table(path, columns=[*_JOIN_KEY, "dividend", "is_refund"])
    race_ids = pa.array(sorted({rec["race_id"] for rec in records if rec["race_id"] is not None}))
    table = table.filter(pc.is_in(table["race_id"], value_set=race_ids))
    dividends: dict[tuple, tuple[float | None, bool | None]] = {}
    for row in table.to_pylist():
        key = tuple(row[field] for field in _JOIN_KEY)
        if key in dividends:
            raise ValueError(f"{path}: (race_id, pool, combination) {key} appears more than once")
        refund = row["is_refund"]
        dividends[key] = (_number(row["dividend"]), None if refund is None else bool(refund))
    for rec in records:
        match = dividends.get(tuple(rec[field] for field in _JOIN_KEY))
        if match is not None:
            rec["dividend"], rec["is_refund"] = match


def _sort_key(rec: dict[str, Any]) -> tuple:
    """A total order on the row's content, so the bytes do not depend on file or log order."""

    def part(value: Any) -> tuple:
        return (value is None, "" if value is None else value)

    return tuple(
        part(rec[name])
        for name in (
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
        )
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
