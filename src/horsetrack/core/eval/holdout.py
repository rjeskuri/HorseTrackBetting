"""The locked final holdout per country, and the log of every use (#71, #126).

``split_off`` divides a frame into development rows and a ``LockedHoldout``. The holdout is
defined in ``<countries_dir>/<country>/config/holdout.yaml`` (with the default root, read through
``horsetrack.core.config.load_country_yaml(country, "holdout")``)::

    holdout:
      start: 2026-11-28
      seed: "<hex string>"
      timezone: Asia/Tokyo   # the country's race-day zone; needed once the holdout is opened

A race day (its local date) on or after ``start`` is holdout when
``int(sha256(f"{seed}:{YYYY-MM-DD}").hexdigest(), 16) % 2 == 0``; every other day is
development. ``is_holdout_day(country, day)`` applies that rule to one day. A missing or invalid
definition raises ``HoldoutConfigError``; there is no default.

``development_filter(country, start, end, *, column="race_date")`` is the same rule as a
``pyarrow.dataset`` expression for a scan (#415): true for the rows whose ``column`` lies in
``start <= day <= end`` (inclusive) and is not a holdout day; a null day is not a development
row. It is built from the calendar alone (every day from ``start`` to ``end`` is asked
``is_holdout_day``, looked up at call time), so a reader passes it to the scan and no
holdout row, outcomes included, reaches a table. Pruning is limited by what the file records:
Parquet skips a row group only from its min/max statistics, so a row group whose days mix
development and holdout days is still read from disk and decoded, and the scan's filter drops
its holdout rows before any table is returned. A row group holding one day (min == max, no
null), as the jravan builders write ``final_odds``, ``pool_votes``, ``races`` and ``runners``
(#502) and the core store build writes ``meetings``, ``races``, ``runners`` and ``win_sales``
(#510: a group per run of one day in the table's order), is pruned whole, never decoded, when
that day is a holdout day or out of range.

``read_development(handle, country, *, columns, start, end, where, column, dated)`` is the
development scan of one Parquet file through one open handle (#415, PR #427 review): it first
narrows ``start..end`` to the file's own first and last ``column`` day (from the row groups'
footer statistics; the row groups without them are scanned once, together, by one Acero plan of
a scan, a projection and a min/max aggregate with a null count on the same handle, so only the
two bounds and the count reach Python), so an open-ended read (``end=date.max``) asks
``is_holdout_day`` only about the days the file can hold, then scans the same handle with
``development_filter`` over that narrowed range. With ``dated=True`` a file with a null
``column`` on any row raises ``ValueError`` before any row is read, instead of the undated rows
being dropped. The bounds and the scan read one snapshot of the file even when it is
atomically replaced meanwhile. Both read each column chunk they need on its own, with no
pre-buffering, whose coalesced reads could span a skipped group (#510): a row group pruned by
its statistics has no byte read, so a development read of a store written a day per row group
reads nothing of a holdout day (D14). ``split_off`` and ``LockedHoldout.open`` below are
unchanged and remain the only way to the holdout rows.

The holdout rows are reached only through ``LockedHoldout.open(ticket=..., purpose=...,
family=...)``, which appends one JSON line to ``<countries_dir>/<country>/holdout_log.jsonl``
before it returns them. An opening is gated (#126, docs/problem-statement.md D12, D15):

- the family must be pre-registered with its power target in ``config/families.yaml``::

      families:
        <name>:
          power_target_races: <int >= 1>   # other keys are allowed and not read here

- the split must name its ``race_col``: a race is one distinct (local race day, ``race_col``)
  pair, and the block must hold at least ``power_target_races`` races;
- a family opens once (any log line naming it counts), and a block opens once: a holdout split
  before another opening refuses.

A refused opening raises ``HoldoutOpenError`` and logs nothing. Every log line is an opening,
and its ``period`` records the block it opened: ``start`` and ``end``, the last race day
(YYYY-MM-DD) among the rows it returned. A race day an opening returned never belongs to a later
block, so the current block starts on the latest of:

- the configured ``start``;
- the local date (``timezone``) of the latest logged ``utc``;
- the day after the latest ``period.end`` over every log line.

An opening made before its day's races are in the frame leaves that day in the next block; one
made after pushes the next block to the following day. A line without ``period.end`` (absent or
null, as before this rule) is placed by its local date alone. The days of an opened block stay
holdout days: development rows and ``is_holdout_day`` never change, and only the
``LockedHoldout`` moves to the current block. A log line that cannot be read, an unparseable
``period.end`` included, refuses.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import pyarrow as pa
import pyarrow.acero as acero
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

from horsetrack.core.config import load_country_yaml

DEFAULT_COUNTRIES_DIR = Path(__file__).resolve().parents[2] / "countries"
CONFIG_STEM = "holdout"
FAMILIES_STEM = "families"
CONFIG_NAME = Path("config") / f"{CONFIG_STEM}.yaml"
LOG_NAME = "holdout_log.jsonl"

_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
# Each column chunk read on its own: a pre-buffer coalesces nearby ranges, across a skipped group.
_NO_PRE_BUFFER = ds.ParquetFragmentScanOptions(pre_buffer=False)


class HoldoutConfigError(ValueError):
    """The country's holdout definition is missing or invalid."""


class HoldoutOpenError(RuntimeError):
    """The holdout may not be opened: no registered power target, too few races, the family or
    the block already opened, or a log that cannot be read."""


def _is_holdout_day(seed: str, day: str) -> bool:
    return int(hashlib.sha256(f"{seed}:{day}".encode()).hexdigest(), 16) % 2 == 0


def _read_yaml(countries_dir: Path, country: str, stem: str) -> tuple[Any, Path]:
    """The parsed ``config/<stem>.yaml`` of ``country`` and its path; the default root goes
    through the shared ``load_country_yaml`` (cached: the result is only read, never mutated).
    Raises ``FileNotFoundError`` when the file is absent."""
    path = countries_dir / country / "config" / f"{stem}.yaml"
    try:
        default = countries_dir is DEFAULT_COUNTRIES_DIR  # skips resolve(): it is called per day
        if default or countries_dir.resolve() == DEFAULT_COUNTRIES_DIR.resolve():
            return load_country_yaml(country, stem), path
        if not path.is_file():
            raise FileNotFoundError(path)
        return yaml.safe_load(path.read_text(encoding="utf-8")), path
    except yaml.YAMLError as exc:
        raise HoldoutConfigError(f"{path} is not valid YAML: {exc}") from exc


def _read_config(countries_dir: Path, country: str) -> tuple[Any, Path]:
    try:
        return _read_yaml(countries_dir, country, CONFIG_STEM)
    except OSError as exc:
        raise HoldoutConfigError(
            f"no holdout definition: {countries_dir / country / CONFIG_NAME} cannot be read ({exc})"
        ) from exc


def _load_definition(countries_dir: Path, country: str) -> tuple[date, str, ZoneInfo | None]:
    config, config_path = _read_config(countries_dir, country)
    section = config.get("holdout") if isinstance(config, dict) else None
    if not isinstance(section, dict):
        raise HoldoutConfigError(f"{config_path} has no holdout section")

    start = section.get("start")
    if isinstance(start, str) and _ISO_DAY.fullmatch(start.strip()):
        try:
            start = date.fromisoformat(start.strip())
        except ValueError:
            pass
    if not isinstance(start, date) or isinstance(start, datetime):
        raise HoldoutConfigError(
            f"{config_path}: holdout.start must be a YYYY-MM-DD date, got {start!r}"
        )

    seed = section.get("seed")
    if not isinstance(seed, str) or not seed.strip():
        raise HoldoutConfigError(f"{config_path}: holdout.seed must be a non-empty string")

    zone = section.get("timezone")
    if zone is not None:
        try:
            zone = ZoneInfo(zone) if isinstance(zone, str) and zone.strip() else None
        except (ZoneInfoNotFoundError, ValueError):
            zone = None
        if zone is None:
            raise HoldoutConfigError(
                f"{config_path}: holdout.timezone must be an IANA zone name, "
                f"got {section.get('timezone')!r}"
            )
    return start, seed, zone


def _power_targets(countries_dir: Path, country: str) -> dict[str, int]:
    """Each pre-registered family's power target, from ``config/families.yaml`` (none when the
    file is absent). A malformed file or target raises ``HoldoutConfigError``."""
    try:
        config, path = _read_yaml(countries_dir, country, FAMILIES_STEM)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise HoldoutConfigError(f"the families of {country} cannot be read ({exc})") from exc
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise HoldoutConfigError(f"{path} must be a mapping with a families section")
    families = config.get("families")
    if families is None:
        return {}
    if not isinstance(families, dict):
        raise HoldoutConfigError(f"{path}: families must map each family to its entry")
    targets = {}
    for name, entry in families.items():
        target = entry.get("power_target_races") if isinstance(entry, dict) else None
        if not isinstance(target, int) or isinstance(target, bool) or target < 1:
            raise HoldoutConfigError(
                f"{path}: families.{name}.power_target_races must be a positive int, got {target!r}"
            )
        targets[str(name)] = target
    return targets


def _read_log(log_path: Path) -> list[str]:
    """The log's non-blank lines; an absent log has none. Other read errors propagate."""
    if not log_path.exists():
        return []
    return [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _period_end(record: dict) -> date | None:
    """The logged ``period.end`` (the last race day an opening returned), or None when the line
    has none. A value that is not a YYYY-MM-DD date raises ``ValueError``."""
    period = record.get("period")
    if period is None:
        return None
    if not isinstance(period, dict):
        raise ValueError(f"period must be a mapping, got {period!r}")
    end = period.get("end")
    if end is None:
        return None
    if not isinstance(end, str) or not _ISO_DAY.fullmatch(end):
        raise ValueError(f"period.end must be a YYYY-MM-DD date, got {end!r}")
    return date.fromisoformat(end)


def _parse_openings(lines: list[str], log_path: Path) -> list[tuple[datetime, Any, date | None]]:
    """(utc, family, period.end) of every logged opening. A line that cannot be read could hide
    an opening, so it raises ``HoldoutOpenError``."""
    openings = []
    for number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
            stamp = datetime.fromisoformat(record["utc"].replace("Z", "+00:00"))
            if stamp.utcoffset() is None:
                raise ValueError("utc has no offset")
            end = _period_end(record)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise HoldoutOpenError(
                f"{log_path} line {number} cannot be read ({exc}); the current block cannot be "
                "placed"
            ) from exc
        openings.append((stamp, record.get("family"), end))
    return openings


def _block_start(
    start: date,
    openings: list[tuple[datetime, Any, date | None]],
    zone: ZoneInfo | None,
    log_path: Path,
) -> date:
    """The latest of ``start``, the local date of the latest logged opening and the day after
    the latest race day any logged opening returned."""
    if not openings:
        return start
    if zone is None:
        raise HoldoutConfigError(
            f"{log_path} records openings but holdout.timezone is not set; the current block "
            "cannot be placed"
        )
    latest = max(stamp for stamp, _, _ in openings).astimezone(zone).date()
    block_start = max(start, latest)
    ends = [end for _, _, end in openings if end is not None]
    if ends:
        block_start = max(block_start, max(ends) + timedelta(days=1))
    return block_start


def is_holdout_day(country: str, day: date) -> bool:
    """Whether the race day ``day`` (its local date) is in ``country``'s locked holdout, by the
    committed definition in its holdout.yaml. Raises ``HoldoutConfigError`` when there is none.
    Days of an opened block stay holdout days."""
    if not isinstance(day, date) or isinstance(day, datetime):
        raise TypeError(f"day must be a date, got {day!r}")
    start, seed, _ = _load_definition(DEFAULT_COUNTRIES_DIR, country)
    return day >= start and _is_holdout_day(seed, day.isoformat())


def development_filter(
    country: str, start: date, end: date, *, column: str = "race_date"
) -> ds.Expression:
    """A dataset expression true exactly for the development days of ``country`` in ``start <=
    column <= end`` (both inclusive; a null day is excluded), for a scan to apply before any row
    is materialized. Built from the calendar, not the data: every day from ``start`` to ``end``
    is asked ``is_holdout_day`` (looked up at call time, so a stand-in for it is honoured on
    every day, before the holdout start too). Raises ``HoldoutConfigError`` when the country has
    no valid holdout definition."""
    for value in (start, end):
        if not isinstance(value, date) or isinstance(value, datetime):
            raise TypeError(f"start and end must be dates, got {value!r}")
    _load_definition(DEFAULT_COUNTRIES_DIR, country)  # no definition: refuse, even for no days
    day = ds.field(column)
    keep = (day >= pa.scalar(start, pa.date32())) & (day <= pa.scalar(end, pa.date32()))
    calendar = (start + timedelta(days=k) for k in range((end - start).days + 1))
    held = [d for d in calendar if is_holdout_day(country, d)]
    if held:
        keep = keep & ~day.isin(pa.array(held, pa.date32()))
    return keep


def _scanned_bounds(
    fragment: ds.ParquetFileFragment, column: str, start: date, end: date
) -> tuple[tuple[date, date] | None, int]:
    """The first and last ``column`` day in ``start..end`` of ``fragment`` (None when it has
    none), and how many of its rows have a null ``column``: one Acero scan of that column feeding
    a min/max aggregate of the in-range days and a null count, so the fragment is read once and
    only the two bounds and the count (``lo``, ``hi``, ``nulls``) reach Python."""
    day = ds.field(column)
    span = (day >= pa.scalar(start, pa.date32())) & (day <= pa.scalar(end, pa.date32()))
    in_span = pc.if_else(span, day, pa.scalar(None, pa.date32()))
    source = ds.FileSystemDataset([fragment], fragment.physical_schema, fragment.format)
    plan = acero.Declaration.from_sequence(
        [
            acero.Declaration("scan", acero.ScanNodeOptions(source, columns={column: day})),
            acero.Declaration(
                "project", acero.ProjectNodeOptions([day, in_span], [column, "in_span"])
            ),
            acero.Declaration(
                "aggregate",
                acero.AggregateNodeOptions(
                    [
                        ("in_span", "min", None, "lo"),
                        ("in_span", "max", None, "hi"),
                        (column, "count", pc.CountOptions(mode="only_null"), "nulls"),
                    ]
                ),
            ),
        ]
    )
    out = plan.to_table()
    low, high = out.column("lo")[0].as_py(), out.column("hi")[0].as_py()
    return (None if low is None else (low, high)), out.column("nulls")[0].as_py()


def _file_day_bounds(
    fragment: ds.ParquetFileFragment, column: str, start: date, end: date
) -> tuple[tuple[date, date] | None, int]:
    """A range holding every ``column`` day in ``start..end`` of the Parquet file ``fragment``
    (None when it holds none), and how many of its rows have a null ``column``. From each row
    group's footer statistics where it records them (min/max and null count); the row groups
    that do not are scanned once, together, for their bounds and null count."""
    meta = fragment.metadata
    lows, highs, unrecorded, nulls = [], [], [], 0
    for g in range(meta.num_row_groups):
        group = meta.row_group(g)
        chunk = next(
            (
                group.column(c)
                for c in range(group.num_columns)
                if group.column(c).path_in_schema == column
            ),
            None,
        )
        if chunk is None:
            raise ValueError(f"the Parquet file has no {column!r} column")
        stats = chunk.statistics
        if group.num_rows == 0:
            continue
        counted = stats is not None and stats.has_null_count
        if not counted or not (stats.has_min_max or stats.null_count == group.num_rows):
            unrecorded.append(g)
            continue
        nulls += stats.null_count
        if stats.has_min_max and stats.max >= start and stats.min <= end:
            lows.append(max(stats.min, start))
            highs.append(min(stats.max, end))
    if unrecorded:
        scanned, missing = _scanned_bounds(
            fragment.subset(row_group_ids=unrecorded), column, start, end
        )
        nulls += missing
        if scanned is not None:
            lows.append(scanned[0])
            highs.append(scanned[1])
    return ((min(lows), max(highs)) if lows else None), nulls


def file_development_filter(
    handle: pa.NativeFile,
    country: str,
    *,
    start: date = date.min,
    end: date = date.max,
    where: ds.Expression | None = None,
    column: str = "race_date",
    dated: bool = False,
) -> ds.Expression:
    """The scan predicate of ``read_development`` for the Parquet file open on ``handle``:
    ``development_filter`` over ``start..end`` narrowed to the file's own first and last day
    (footer statistics, or one min/max scan of the row groups without them), and ``where``; a
    range outside the file keeps no row. For a reader that scans the same snapshot by another
    route (``models.v0.store``). With ``dated``, a file with a null ``column`` on any row is
    refused (``ValueError``) instead of its undated rows being dropped: the null count comes from
    the same footer statistics and bounds scan, so no row is read for it. The bounds scan reads
    only the ``column`` chunks of the row groups without statistics, each on its own (no
    pre-buffer reaching a group between them). Raises ``HoldoutConfigError`` when the country has
    no valid holdout definition."""
    fragment = ds.ParquetFileFormat(default_fragment_scan_options=_NO_PRE_BUFFER).make_fragment(
        handle
    )
    bounds, nulls = _file_day_bounds(fragment, column, start, end)
    if dated and nulls:
        raise ValueError(f"{column} has a missing value: {nulls} row(s) of the file have no day")
    if bounds is None:
        _load_definition(DEFAULT_COUNTRIES_DIR, country)  # no definition: refuse, even for no days
        keep = ds.scalar(False)
    else:
        keep = development_filter(country, *bounds, column=column)
    return keep if where is None else keep & where


def read_development(
    handle: pa.NativeFile,
    country: str,
    *,
    columns: Sequence[str],
    start: date = date.min,
    end: date = date.max,
    where: ds.Expression | None = None,
    column: str = "race_date",
    dated: bool = False,
) -> pa.Table:
    """The development rows of the Parquet file open on ``handle`` with ``start <= column <=
    end``: exactly ``columns``, in that order, with ``where`` (a dataset expression, which may
    name columns not read) narrowing them further inside the scan. The range is first narrowed to
    the file's own first and last day (footer statistics, or one min/max scan of the row groups
    without them), so ``development_filter`` asks ``is_holdout_day`` only about days the file
    can hold; a range outside the file returns no row. Bounds and scan read the one open handle,
    so a file atomically replaced at its path meanwhile is read as it was. The scan skips each
    row group whose ``column`` statistics rule the predicate out and reads the column chunks of
    the others one by one (no pre-buffer, whose coalesced reads could span a skipped group): a
    group holding one holdout day (min == max, no null) has no byte read, while a group mixing
    days, or without statistics, is read and its holdout rows dropped by the filter. With
    ``dated``, a file with a null ``column`` on any row raises ``ValueError`` before any row is
    read (see ``file_development_filter``). Raises ``HoldoutConfigError`` when the country has no
    valid holdout definition."""
    keep = file_development_filter(
        handle, country, start=start, end=end, where=where, column=column, dated=dated
    )
    return pq.read_table(handle, columns=list(columns), filters=keep, pre_buffer=False)


def _race_days(frame: pd.DataFrame, date_col: str) -> pd.Series:
    """Each row's race day as YYYY-MM-DD, from naive local dates, timestamps or strings."""
    stamps = pd.to_datetime(frame[date_col])
    if stamps.dt.tz is not None:
        raise ValueError(f"{date_col!r} is timezone-aware; pass the local race date (naive)")
    if stamps.isna().any():
        raise ValueError(f"{date_col!r} has missing dates; every row needs a race day")
    days = stamps.dt.normalize()
    labels = {day: day.strftime("%Y-%m-%d") for day in days.unique()}
    return days.map(labels)


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot read the git commit for the holdout log") from exc
    return result.stdout.strip()


class LockedHoldout:
    """The current block of one country's holdout, reachable only through ``open``."""

    __slots__ = (
        "__rows",
        "__log_path",
        "__countries_dir",
        "__country",
        "__start",
        "__seed",
        "__zone",
        "__n_days",
        "__n_races",
        "__end",
        "__seen_lines",
    )

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        log_path: Path,
        countries_dir: Path,
        country: str,
        start: date,
        seed: str,
        zone: ZoneInfo | None,
        n_days: int,
        n_races: int | None,
        end: str | None,
        seen_lines: list[str],
    ) -> None:
        self.__rows = rows
        self.__log_path = log_path
        self.__countries_dir = countries_dir
        self.__country = country
        self.__start = start
        self.__seed = seed
        self.__zone = zone
        self.__n_days = n_days
        self.__n_races = n_races
        self.__end = end
        self.__seen_lines = tuple(seen_lines)

    def open(self, *, ticket: int, purpose: str, family: str) -> pd.DataFrame:
        """Log this opening of the current block for ``family``, then return its rows.

        Raises ``HoldoutOpenError`` and logs nothing when the family has no registered power
        target, the split named no ``race_col``, the block has fewer races than the target, the
        family has opened before, or the log changed since this holdout was split. One JSON line
        (utc, ticket, purpose, family, git_sha, n_rows, n_races, power_target, period) is
        appended and synced to the country's holdout log before any row is returned; if it
        cannot be written, no rows are returned.
        """
        if not isinstance(ticket, int) or isinstance(ticket, bool):
            raise TypeError(f"ticket must be an issue number (int), got {ticket!r}")
        if ticket <= 0:
            raise ValueError(f"ticket must be a positive issue number, got {ticket!r}")
        if not isinstance(purpose, str):
            raise TypeError(f"purpose must be a string, got {purpose!r}")
        if not purpose.strip():
            raise ValueError("purpose must say why the holdout is opened")
        if not isinstance(family, str):
            raise TypeError(f"family must be a string, got {family!r}")
        if not family.strip():
            raise ValueError("family must name the pre-registered family being claimed")

        targets = _power_targets(self.__countries_dir, self.__country)
        lines = _read_log(self.__log_path)
        openings = _parse_openings(lines, self.__log_path)
        if tuple(lines) != self.__seen_lines:
            raise HoldoutOpenError(
                f"{self.__log_path} changed since this holdout was split: its block is spent; "
                "split again for the current block"
            )
        if any(logged == family for _, logged, _ in openings):
            raise HoldoutOpenError(f"family {family!r} has already opened the holdout")
        if family not in targets:
            raise HoldoutOpenError(
                f"family {family!r} has no registered power target for {self.__country}"
            )
        target = targets[family]
        if self.__n_races is None:
            raise HoldoutOpenError("the split named no race_col, so its races cannot be counted")
        if self.__n_races < target:
            raise HoldoutOpenError(
                f"the block from {self.__start.isoformat()} has {self.__n_races} races, below "
                f"the power target of {target} for family {family!r}"
            )
        if self.__zone is None:
            raise HoldoutConfigError(
                "holdout.timezone is not set; an opening could not be placed after it"
            )

        record = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "ticket": ticket,
            "purpose": purpose,
            "family": family,
            "git_sha": _git_sha(),
            "n_rows": len(self.__rows),
            "n_races": self.__n_races,
            "power_target": target,
            "period": {
                "start": self.__start.isoformat(),
                "end": self.__end,
                "seed_sha256": hashlib.sha256(self.__seed.encode()).hexdigest(),
            },
        }
        line = json.dumps(record) + "\n"
        with open(self.__log_path, "a", encoding="utf-8") as log:
            log.write(line)
            log.flush()
            os.fsync(log.fileno())
        return self.__rows.copy()

    def __reduce__(self):
        raise TypeError("a LockedHoldout cannot be pickled or copied; use open()")

    def __repr__(self) -> str:
        races = "" if self.__n_races is None else f", n_races={self.__n_races}"
        return (
            f"LockedHoldout(country={self.__country!r}, start={self.__start.isoformat()}, "
            f"n_rows={len(self.__rows)}, n_days={self.__n_days}{races})"
        )


def split_off(
    frame: pd.DataFrame,
    date_col: str,
    country: str,
    *,
    countries_dir: Path | str = DEFAULT_COUNTRIES_DIR,
    race_col: str | None = None,
) -> tuple[pd.DataFrame, LockedHoldout]:
    """Return the development rows and the country's locked holdout (its current block).

    ``race_col`` identifies a race within its day; without it the holdout cannot be opened.
    Raises ``HoldoutConfigError`` when the country has no valid holdout definition and
    ``HoldoutOpenError`` when its log cannot be read.
    """
    countries_dir = Path(countries_dir)
    log_path = countries_dir / country / LOG_NAME
    start, seed, zone = _load_definition(countries_dir, country)
    lines = _read_log(log_path)
    block_start = _block_start(start, _parse_openings(lines, log_path), zone, log_path)

    days = _race_days(frame, date_col)
    start_label, block_label = start.isoformat(), block_start.isoformat()
    held_days = {d for d in days.unique() if d >= start_label and _is_holdout_day(seed, d)}
    mask = days.isin(held_days).to_numpy()
    block_days = {d for d in held_days if d >= block_label}
    in_block = days.isin(block_days).to_numpy()

    n_races = None
    if race_col is not None:
        races = frame[race_col]
        if races.isna().any():
            raise ValueError(f"{race_col!r} has missing values; every row needs its race")
        n_races = len(
            pd.DataFrame({"day": days[in_block], "race": races[in_block]}).drop_duplicates()
        )

    development = frame.loc[~mask].copy()
    holdout = LockedHoldout(
        frame.loc[in_block].copy(),
        log_path=log_path,
        countries_dir=countries_dir,
        country=country,
        start=block_start,
        seed=seed,
        zone=zone,
        n_days=len(block_days),
        n_races=n_races,
        end=max(block_days) if block_days else None,
        seen_lines=lines,
    )
    return development, holdout
