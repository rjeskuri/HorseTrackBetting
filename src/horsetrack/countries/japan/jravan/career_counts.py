"""JV-Data CK (出走別着度数) records -> the jravan `career_counts` table
(`horsetrack japan jravan career-counts`, #407).

A CK record is one runner in one race: its placings so far (1st to 5th and out of the first five)
under each condition (surface, distance, going, venue...), and the same for its jockey and trainer,
this year and in total, as of that race. `build_career_counts` writes them in long form to
`<out>/career_counts.parquet`, one row per race, horse, subject, period and condition.

The export holds every CK record in one multi-GB file, `<raw>/stored/SNPN/CK/all.jvd`, so the build
runs in bounded memory in two steps:

1. One pass streams the file through `records.iter_lines`, one chunk at a time. Each record's type
   is checked (`records.check_types`), then its race date is read from the bytes of 開催年 and
   開催月日 alone: a record on a holdout day (`is_holdout_day('japan', day)`) is read no further,
   whatever its width, and counted only in `holdout_skipped`. Every other record must be the
   layout's width (else `records.RecordLengthError` with the file's record number); one without a
   valid race date (`undated`) or outside the JRA venues (`non_jra`) is dropped, and the rest are
   spilled, undecoded, into one temporary file per race month next to the output.
2. Each month, in order, is decoded on its own: one version per key (`records.latest_versions` on
   the race key and 血統登録番号, a key whose latest version is a deletion has no rows; a key's
   versions share a race date, so they share a month), its records sorted by race id and horse id,
   and its rows written as one row group. Race ids start with the date, so the file is sorted by
   `race_id, horse_id, subject, period, condition` (plain string order) throughout, and the same
   records, in any order and at any chunk size, give the same bytes.

The rows come from the layout (`spec.record_spec('CK')`), never a hand-written list: a leaf whose
name ends in `着回数` with six repeats is a placing-counts item, its `condition` the name without
`着回数` and a trailing `・` (or the half-width `･`). A top-level item is a `horse` row
(`subject_id` 血統登録番号, `period` `career`); a sub-item of the group whose name holds `騎手` a
`jockey` row (`subject_id` the top-level 騎手コード), of the group whose name holds `調教師` a
`trainer` row (調教師コード); the group's first occurrence is `this_year`, the second `career`.
The owner's and breeder's groups give no rows. A count that is not a number is null.
See docs/specs/jravan.md, "CK: career counts (#407)".
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.reports import LicensedOutputInRepo
from horsetrack.countries.japan.jravan import keys, records, spec

SOURCE = records.SOURCE
COUNTRY = "japan"
NAME = "career_counts"
FILENAME = "career_counts.parquet"
RECORD_TYPE = "CK"
JURISDICTION = "jra"

YEAR, MMDD, VENUE, RACE_NO = "開催年", "開催月日", "競馬場コード", "レース番号"
HORSE_ID, CREATED, KUBUN = "血統登録番号", "データ作成年月日", "データ区分"
VERSION_KEY = [YEAR, MMDD, VENUE, "開催回[第N回]", "開催日目[N日目]", RACE_NO, HORSE_ID]

COUNT_SUFFIX = "着回数"
CONDITION_TRIM = ("・", "･")  # full- and half-width middle dots
PLACES = ("first", "second", "third", "fourth", "fifth", "unplaced")
# group name fragment -> (subject, its code item); the occurrences in period order
SUBJECT_GROUPS = (("騎手", "jockey", "騎手コード"), ("調教師", "trainer", "調教師コード"))
HORSE, CAREER, THIS_YEAR = "horse", "career", "this_year"
GROUP_PERIODS = (THIS_YEAR, CAREER)  # 本年, then 累計
SUBJECTS = (HORSE, *(s for _, s, _ in SUBJECT_GROUPS))

REPORT_KEYS = (
    "records", "holdout_skipped", "undated", "non_jra", "versions_dropped", "deletions", "kept",
    "rows",
)  # fmt: skip

CAREER_COUNTS_SCHEMA = pa.schema(
    [
        ("race_id", pa.string()),
        ("race_date", pa.date32()),
        ("horse_id", pa.string()),
        ("subject", pa.string()),
        ("subject_id", pa.string()),
        ("period", pa.string()),
        ("condition", pa.string()),
        *((place, pa.int64()) for place in PLACES),
    ]
)


# ------------------------------------------------------------------------------------------------
# The layout -> one record's rows


@dataclass(frozen=True)
class _Count:
    """One row of every record: the `records.decode` key it reads, and which occurrence of its
    group (None for a top-level item)."""

    key: str
    subject: str
    period: str
    condition: str
    occurrence: int | None


def _condition(name: str) -> str:
    name = name.removesuffix(COUNT_SUFFIX)
    for dot in CONDITION_TRIM:
        if name.endswith(dot):
            return name[: -len(dot)]
    return name


def _is_count(item: spec.Item) -> bool:
    return not item.items and item.name.endswith(COUNT_SUFFIX) and item.repeat == len(PLACES)


def _named(items: tuple[spec.Item, ...]) -> Iterator[tuple[str, spec.Item]]:
    """Each item but the reserves under its `records.decode` name (`name#k` for a repeat)."""
    seen: Counter[str] = Counter()
    for item in items:
        if item.name == records.RESERVE:
            continue
        seen[item.name] += 1
        k = seen[item.name]
        yield (item.name if k == 1 else f"{item.name}#{k}"), item


def _subject_of(group: str) -> tuple[str, str] | None:
    for fragment, subject, code in SUBJECT_GROUPS:
        if fragment in group:
            return subject, code
    return None


def _template(layout: spec.RecordSpec) -> tuple[tuple[_Count, ...], dict[str, str]]:
    """Every record's rows, sorted by subject, period and condition (layout order on a tie), and
    each subject's id item."""
    counts: list[_Count] = []
    ids = {HORSE: HORSE_ID}
    for key, item in _named(layout.items[:-1]):  # the last item is the separator
        if _is_count(item):
            counts.append(_Count(key, HORSE, CAREER, _condition(item.name), None))
            continue
        found = _subject_of(item.name) if item.items else None
        if found is None:
            continue
        subject, code = found
        if subject in ids:
            raise ValueError(f"{layout.record_type}: a second {subject} group, {item.name}")
        if item.repeat != len(GROUP_PERIODS):
            raise ValueError(
                f"{layout.record_type}: group {item.name} repeats {item.repeat} times, "
                f"not {len(GROUP_PERIODS)} (this year, career)"
            )
        ids[subject] = code
        for sub_key, sub in _named(item.items):
            if _is_count(sub):
                for occurrence, period in enumerate(GROUP_PERIODS):
                    counts.append(
                        _Count(f"{key}/{sub_key}", subject, period, _condition(sub.name),
                               occurrence)
                    )  # fmt: skip
    leaves = records.leaf_axes(layout)
    for subject, code in ids.items():
        if code not in leaves:
            raise ValueError(f"{layout.record_type}: no item {code} for the {subject} rows")
    order = sorted(range(len(counts)), key=lambda i: (counts[i].subject, counts[i].period,
                                                       counts[i].condition, i))  # fmt: skip
    return tuple(counts[i] for i in order), ids


def _numbers(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`records.integer` of an `S<n>` array as (int64 values, is a number), shaped as `raw`:
    plain digits parsed in bulk, anything else (blank, padded, signed) by `records.integer`."""
    raw = np.ascontiguousarray(raw)
    shape, width = raw.shape, raw.dtype.itemsize
    flat = raw.reshape(-1)
    values = np.zeros(flat.size, dtype=np.int64)
    if flat.size == 0:
        return values.reshape(shape), np.zeros(shape, dtype=bool)
    u8 = flat.view(np.uint8).reshape(flat.size, width)
    ok = ((u8 >= ord("0")) & (u8 <= ord("9"))).all(axis=1)
    if ok.any():
        powers = 10 ** np.arange(width - 1, -1, -1, dtype=np.int64)
        values[ok] = (u8[ok].astype(np.int64) - ord("0")) @ powers
    other = np.flatnonzero(~ok)
    if other.size:
        parsed = records.integer(flat[other])
        good = ~np.isnan(parsed)
        values[other[good]] = parsed[good].astype(np.int64)
        ok[other[good]] = True
    return values.reshape(shape), ok.reshape(shape)


def _table(
    cols: Mapping[str, np.ndarray],
    use: np.ndarray,
    race_ids: list[str],
    template: tuple[_Count, ...],
    ids: dict[str, str],
) -> pa.Table:
    """The rows of the records `use` of `cols` (already in table order), each with its race id."""
    n, t = len(use), len(template)
    values = np.zeros((n, t, len(PLACES)), dtype=np.int64)
    ok = np.zeros((n, t, len(PLACES)), dtype=bool)
    for j, c in enumerate(template):
        raw = cols[c.key][use]
        if c.occurrence is not None:
            raw = raw[:, c.occurrence]
        values[:, j], ok[:, j] = _numbers(raw)
    values, ok = values.reshape(n * t, -1), ok.reshape(n * t, -1)
    rec = pa.array(np.repeat(np.arange(n, dtype=np.int64), t))
    slot = pa.array(np.tile(np.arange(t, dtype=np.int64), n))
    subject_index = np.array([SUBJECTS.index(c.subject) for c in template], dtype=np.int64)
    subject_ids = np.empty((n, len(SUBJECTS)), dtype=object)
    for s, subject in enumerate(SUBJECTS):
        if subject in ids:
            subject_ids[:, s] = records.text(cols[ids[subject]][use])
    who = np.repeat(np.arange(n, dtype=np.int64), t) * len(SUBJECTS) + np.tile(subject_index, n)
    columns = [
        pa.array(race_ids, pa.string()).take(rec),
        pa.array([date.fromisoformat(r.split(":")[1]) for r in race_ids], pa.date32()).take(rec),
        pa.array(subject_ids[:, 0].tolist(), pa.string()).take(rec),
        pa.array([c.subject for c in template], pa.string()).take(slot),
        pa.array(subject_ids.reshape(-1).tolist(), pa.string()).take(pa.array(who)),
        pa.array([c.period for c in template], pa.string()).take(slot),
        pa.array([c.condition for c in template], pa.string()).take(slot),
        *(pa.array(values[:, p], pa.int64(), mask=~ok[:, p]) for p in range(len(PLACES))),
    ]
    return pa.Table.from_arrays(columns, schema=CAREER_COUNTS_SCHEMA)


# ------------------------------------------------------------------------------------------------
# Step 1: the file -> one spill file per race month


def _day(raw: bytes) -> date | None:
    if len(raw) != 8 or not raw.isdigit():
        return None
    try:
        return date(int(raw[:4]), int(raw[4:6]), int(raw[6:]))
    except ValueError:
        return None


def _spill(
    path: Path, layout: spec.RecordSpec, chunk_records: int, spill: Path, counts: Counter[str]
) -> list[Path]:
    """Stream `path`, skip the holdout days, check the rest's width, drop the undated and
    non-JRA records and append the others (undecoded, one per line) to `<spill>/<YYYYMM>`; the
    month files, in order."""
    year, mmdd = records.leaf_bytes(layout, YEAR), records.leaf_bytes(layout, MMDD)
    venue = records.leaf_bytes(layout, VENUE)
    date_end = max(year.stop, mmdd.stop)
    width = layout.length - spec.SEPARATOR_BYTES
    days: dict[bytes, date | None] = {}
    held: dict[date, bool] = {}
    jra: dict[bytes, bool] = {}
    months: set[str] = set()
    for first, lines in records.iter_lines(path, layout, chunk_records=chunk_records):
        records.check_types(lines, layout, first)  # before any date is trusted
        counts["records"] += len(lines)
        by_month: dict[str, list[bytes]] = {}
        for i, line in enumerate(lines):
            stamp = line[year] + line[mmdd] if len(line) >= date_end else b""  # all read first
            if stamp not in days:
                days[stamp] = _day(stamp)
            day = days[stamp]
            if day is not None:
                if day not in held:
                    held[day] = is_holdout_day(COUNTRY, day)
                if held[day]:
                    counts["holdout_skipped"] += 1
                    continue
            if len(line) != width:
                raise records.RecordLengthError(layout.record_type, first + i, len(line), width)
            if day is None:
                counts["undated"] += 1
                continue
            code = line[venue]
            if code not in jra:
                text = code.decode(records.ENCODING, errors="replace").strip(records.TRIM)
                jra[code] = keys.jurisdiction(text) == JURISDICTION
            if not jra[code]:
                counts["non_jra"] += 1
                continue
            by_month.setdefault(stamp[:6].decode("ascii"), []).append(line)
        for month, kept in by_month.items():
            with open(spill / month, "ab") as handle:
                handle.write(b"\n".join(kept) + b"\n")
            months.add(month)
        del lines, by_month
    return [spill / m for m in sorted(months)]


# ------------------------------------------------------------------------------------------------
# Step 2: one month -> its rows


def _month(
    path: Path,
    layout: spec.RecordSpec,
    template: tuple[_Count, ...],
    ids: dict[str, str],
    counts: Counter[str],
) -> pa.Table | None:
    """One spill file's rows, one version per key, in table order (None when it has none)."""
    cols = records.decode(path.read_bytes(), layout)
    frame = pd.DataFrame({k: records.text(cols[k]) for k in (*VERSION_KEY, CREATED, KUBUN)})
    distinct = len(frame.drop_duplicates(VERSION_KEY))
    latest = records.latest_versions(frame, VERSION_KEY)
    counts["versions_dropped"] += len(frame) - distinct
    counts["deletions"] += distinct - len(latest)
    rows = []
    for i, *key in zip(latest.index, *(latest[k] for k in VERSION_KEY), strict=True):
        race_id = keys.race_id(key[0], key[1], key[2], key[5])
        if race_id is None:  # a JRA venue with an invalid race number
            counts["non_jra"] += 1
            continue
        rows.append((race_id, key[6] or "", tuple(k or "" for k in key), int(i)))
    if not rows:
        return None
    rows.sort()
    counts["kept"] += len(rows)
    use = np.array([r[3] for r in rows], dtype=np.int64)
    return _table(cols, use, [r[0] for r in rows], template, ids)


# ------------------------------------------------------------------------------------------------
# Build


def build_career_counts(
    *, raw: Path | None = None, out: Path | None = None, chunk_records: int = 20_000
) -> records.BuildReport:
    """Read `<raw>/stored/SNPN/CK/all.jvd` (`raw` the export root, default
    `paths.raw_dir('jravan')`), `chunk_records` records at a time, and write
    `<out>/career_counts.parquet` (default `paths.processed_dir('jravan')`), one row group per race
    month. Returns the counts, no record content (D21)."""
    root = Path(raw) if raw is not None else paths.raw_dir(SOURCE)
    source = root / "stored" / "SNPN" / RECORD_TYPE / "all.jvd"
    if not source.is_file():
        raise FileNotFoundError(f"no CK file: {source}")
    layout = spec.record_spec(RECORD_TYPE)
    template, ids = _template(layout)
    target = (Path(out) if out is not None else paths.processed_dir(SOURCE)) / FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:12]
    tmp = target.with_name(f".{target.name}.{tag}.tmp")
    spill = target.with_name(f".{target.name}.{tag}.spill.tmp")
    counts: Counter[str] = Counter()
    by_year: dict[int, dict[str, int]] = {}
    try:
        spill.mkdir()
        months = _spill(source, layout, chunk_records, spill, counts)
        with pq.ParquetWriter(tmp, CAREER_COUNTS_SCHEMA, compression="zstd") as writer:
            for month in months:
                table = _month(month, layout, template, ids, counts)
                month.unlink()
                if table is None:
                    continue
                counts["rows"] += table.num_rows
                year = by_year.setdefault(int(month.name[:4]), {"rows": 0})
                year["rows"] += table.num_rows
                writer.write_table(table, row_group_size=table.num_rows)
                del table
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
        shutil.rmtree(spill, ignore_errors=True)
    totals = {key: int(counts[key]) for key in REPORT_KEYS}
    return records.BuildReport(name=NAME, counts=totals, by_year=by_year, unmapped={}, failures=[])


def career_counts_command() -> None:
    """Parse the JRA-VAN export's CK records (SNPN/CK/all.jvd) into
    processed/jravan/career_counts.parquet; the counts in reports/jravan-career_counts.json."""
    errors = (paths.DataDirNotSet, LicensedOutputInRepo, ValueError, LookupError, OSError)
    try:
        report = build_career_counts()
        written = records.write_build_report(report)
    except errors as exc:
        _die(f"jravan career-counts failed: {type(exc).__name__}: {exc}")
    c = report.counts
    typer.echo(
        f"career-counts: {c['rows']} rows, {c['kept']} runners, {len(report.by_year)} years; "
        f"{c['records']} records, {c['versions_dropped']} versions dropped, "
        f"{c['deletions']} deletions, {c['non_jra']} non-JRA, "
        f"{c['holdout_skipped']} on holdout days"
    )
    typer.echo(f"report: {written}")


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)
