"""JV-Data DM (time-type) and TM (match-type) data-mining predictions -> the jravan
`mining_predictions` table (`horsetrack japan jravan mining`, #406).

`build_mining_predictions` reads `<raw>/stored/MING/DM|TM/<YYYY>.jvd` year by year, streamed
through `records.iter_lines`: each record's type is checked and its race date read from its
開催年 and 開催月日 bytes alone, so a record on a holdout day (`is_holdout_day('japan', day)`, D14)
is read no further and counted only in `holdout_skipped` (as `card_versions` does). A first pass
over a file reads the version keys of the other records and keeps one version per race
(`records.latest_versions` on 開催年, 開催月日, 競馬場コード, 開催回, 開催日目, レース番号, the
latest by データ作成年月日 then データ作成時分, a tie to the later record; a race whose latest
record is a deletion has no rows of that kind); a second turns the kept records of JRA venues into
rows. Each year's rows (DM and TM) are sorted by `race_id, kind, horse_no` and written as one row
group of `<out>/mining_predictions.parquet`, through a tmp file and `os.replace`; the same inputs
give the same bytes.

One row per slot of マイニング予想 whose 馬番 is a positive integer: `kind` the record type,
`value` the DM 予想走破タイム in seconds (MMSSff; an all-zero time is unset) or the TM 予測スコア
(tenths; zero is a score), `error_plus_s` and `error_minus_s` the DM 予想誤差(信頼度)＋ and －
(hundredths of a second; null on TM), each null when blank or not a number; `rank` per race and
kind over the rows with a value (DM ascending, TM descending, ties sharing the lowest rank);
`data_kubun` and `created_utc` (データ作成年月日 + データ作成時分 JST, as UTC; null when either is
invalid) of the record kept. The table is not a
store table and has no availability rule until the owner sets one, so the as-of views refuse it.
See docs/specs/jravan.md.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Iterator, Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, NoReturn

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.countries.japan.jravan import keys, records, spec
from horsetrack.countries.japan.jravan.card_versions import _holdout_flags, _join, _line_dates

SOURCE = records.SOURCE
COUNTRY = "japan"
NAME = "mining"
FILENAME = "mining_predictions.parquet"
DATASPEC = "MING"
JURISDICTION = "jra"
RECORD_TYPES = ("DM", "TM")
VERSION_KEY = [
    "開催年", "開催月日", "競馬場コード", "開催回[第N回]", "開催日目[N日目]", "レース番号",
]  # fmt: skip
CREATED, HHMM, KUBUN, VENUE = "データ作成年月日", "データ作成時分", "データ区分", "競馬場コード"
YEAR, MMDD = "開催年", "開催月日"
RACE_KEY = ("開催年", "開催月日", "競馬場コード", "レース番号")  # keys.race_id's arguments
GROUP = "マイニング予想"
HORSE = f"{GROUP}/馬番"
DM_TIME = f"{GROUP}/予想走破タイム"
DM_PLUS = f"{GROUP}/予想誤差(信頼度)＋"
DM_MINUS = f"{GROUP}/予想誤差(信頼度)－"
TM_SCORE = f"{GROUP}/予測スコア"
REPORT_KEYS = (
    "records", "holdout_skipped", "versions_dropped", "deletions", "non_jra", "kept", "rows",
)  # fmt: skip

MINING_PREDICTIONS_SCHEMA = pa.schema(
    [
        ("jurisdiction", pa.string()),
        ("race_id", pa.string()),
        ("race_date", pa.date32()),
        ("venue", pa.string()),
        ("race_no", pa.int64()),
        ("horse_no", pa.int64()),
        ("kind", pa.string()),
        ("value", pa.float64()),
        ("error_plus_s", pa.float64()),
        ("error_minus_s", pa.float64()),
        ("rank", pa.int64()),
        ("data_kubun", pa.string()),
        ("created_utc", pa.timestamp("us", tz="UTC")),
    ]
)
_SORT = [("race_id", "ascending"), ("kind", "ascending"), ("horse_no", "ascending")]


# ------------------------------------------------------------------------------------------------
# Records -> rows


def _seconds(raw: np.ndarray) -> np.ndarray:
    """予想走破タイム (MMSSff: the last four digits SSff, the digits before them the minutes) as
    seconds; NaN when blank, not a number, negative or all zeros (an unset time)."""
    v = records.integer(raw)
    v[v <= 0] = np.nan
    minutes = np.floor(v / 10_000)
    return (minutes * 6_000 + (v - minutes * 10_000)) / 100


def _stamp(raw: bytes) -> datetime | None:
    """データ作成年月日 + データ作成時分 (YYYYMMDDHHMM, JST) as an aware UTC datetime, or None."""
    if len(raw) != 12 or not raw.isdigit():
        return None
    try:
        local = datetime(
            int(raw[:4]), int(raw[4:6]), int(raw[6:8]), int(raw[8:10]), int(raw[10:]),
            tzinfo=keys.JST,
        )  # fmt: skip
    except ValueError:
        return None
    return local.astimezone(timezone.utc)


def _stamps(cols: Mapping[str, np.ndarray]) -> list[datetime | None]:
    joined = _join(cols[CREATED], cols[HHMM])
    values, inverse = np.unique(joined, return_inverse=True)
    parsed = [_stamp(bytes(v)) for v in values]
    return [parsed[i] for i in inverse.reshape(-1)]


def _ranks(values: np.ndarray, valid: np.ndarray, ascending: bool) -> np.ndarray:
    """Per record (row) the rank of each slot among its valid slots: 1 + the valid slots strictly
    better (lower when `ascending`, else higher), so ties share the lowest rank."""
    v = np.where(valid, values, 0.0)
    better = v[:, None, :] < v[:, :, None] if ascending else v[:, None, :] > v[:, :, None]
    return 1 + (better & valid[:, None, :]).sum(axis=2)


def _table(record_type: str, cols: Mapping[str, np.ndarray], use: np.ndarray | None) -> pa.Table:
    """The rows of the records of `cols` (those of `use`, when given) at a JRA venue, unsorted."""
    n = len(cols[KUBUN])
    if n == 0:
        return MINING_PREDICTIONS_SCHEMA.empty_table()
    parts = [records.text(cols[k]) for k in RACE_KEY]
    race_ids = np.array([keys.race_id(*p) for p in zip(*parts, strict=True)], dtype=object)
    keep = np.array([r is not None for r in race_ids], dtype=bool)
    if use is not None:
        keep &= use
    horse = records.integer(cols[HORSE].reshape(n, -1))
    slot_ok = keep[:, None] & (horse > 0) & (horse == np.floor(horse))
    if record_type == "DM":
        value = _seconds(cols[DM_TIME].reshape(n, -1))
        plus = records.scaled(cols[DM_PLUS].reshape(n, -1), 2)
        minus = records.scaled(cols[DM_MINUS].reshape(n, -1), 2)
    else:
        value = records.scaled(cols[TM_SCORE].reshape(n, -1), 1)
        plus = minus = np.full(value.shape, np.nan)
    has_value = slot_ok & ~np.isnan(value)
    rank = _ranks(value, has_value, ascending=record_type == "DM")
    rec, slot = np.nonzero(slot_ok)

    kept = np.unique(rec)
    ids = [str(race_ids[i]) for i in kept]
    split = [r.split(":") for r in ids]  # jra:YYYY-MM-DD:VENUE:NN
    local = np.zeros(n, dtype=np.int64)
    local[kept] = np.arange(len(kept))
    take = pa.array(local[rec])
    kept_cols = {k: v[kept] for k, v in cols.items() if k in (KUBUN, CREATED, HHMM)}

    def measure(a: np.ndarray) -> pa.Array:
        picked = a[rec, slot]
        return pa.array(picked, type=pa.float64(), mask=np.isnan(picked))

    picked_rank = rank[rec, slot]
    columns = [
        pa.repeat(pa.scalar(JURISDICTION, pa.string()), len(rec)),
        pa.array(ids, pa.string()).take(take),
        pa.array([date.fromisoformat(s[1]) for s in split], pa.date32()).take(take),
        pa.array([s[2] for s in split], pa.string()).take(take),
        pa.array([int(s[3]) for s in split], pa.int64()).take(take),
        pa.array(horse[rec, slot].astype(np.int64), pa.int64()),
        pa.repeat(pa.scalar(record_type, pa.string()), len(rec)),
        measure(value),
        measure(plus),
        measure(minus),
        pa.array(picked_rank.astype(np.int64), pa.int64(), mask=~has_value[rec, slot]),
        pa.array(list(records.text(kept_cols[KUBUN])), pa.string()).take(take),
        pa.array(_stamps(kept_cols), MINING_PREDICTIONS_SCHEMA.field("created_utc").type).take(
            take
        ),
    ]
    return pa.Table.from_arrays(columns, schema=MINING_PREDICTIONS_SCHEMA)


def mining_rows(record_type: str, cols: Mapping[str, np.ndarray]) -> pa.Table:
    """`records.decode` of DM or TM records -> their rows (`MINING_PREDICTIONS_SCHEMA`): one per
    slot with a positive 馬番, JRA venues only, every version kept; ranks per record."""
    if record_type not in RECORD_TYPES:
        raise ValueError(f"not a mining record type: {record_type!r}")
    return _table(record_type, cols, None)


# ------------------------------------------------------------------------------------------------
# Reading a file, holdout days skipped


def _chunks(
    path: Path, layout: spec.RecordSpec, chunk_records: int, holdout
) -> Iterator[tuple[dict[str, np.ndarray] | None, int]]:
    """Per chunk of `path`: the decoded records not on a holdout day (None when there is none),
    and the records skipped as on one. A record's type is checked first and its race date read
    from its 開催年 and 開催月日 bytes alone; a holdout record is read no further."""
    year, mmdd = records.leaf_bytes(layout, YEAR), records.leaf_bytes(layout, MMDD)
    for first, lines in records.iter_lines(path, layout, chunk_records=chunk_records):
        records.check_types(lines, layout, first)
        days = _line_dates(lines, year, mmdd)
        dated = ~np.isnat(days)
        held = np.zeros(len(days), dtype=bool)
        held[dated] = _holdout_flags(days[dated], holdout)
        rest = np.flatnonzero(~held)
        if not rest.size:
            yield None, int(held.sum())
            continue
        kept = lines if rest.size == len(lines) else [lines[i] for i in rest]
        yield records.decode_lines(kept, layout, first + rest), int(held.sum())


def _versions(
    path: Path, layout: spec.RecordSpec, chunk_records: int, holdout
) -> tuple[np.ndarray, dict[str, int]]:
    """A first pass over one file: which of its records outside the holdout to keep (one version
    per race, deletions dropped, JRA venues only) and the counts of those choices."""
    parts: dict[str, list[np.ndarray]] = {k: [] for k in (*VERSION_KEY, KUBUN, "stamp")}
    skipped = 0
    for chunk, held in _chunks(path, layout, chunk_records, holdout):
        skipped += held
        if chunk is None:
            continue
        for k in (*VERSION_KEY, KUBUN):
            parts[k].append(records.text(chunk[k]))
        parts["stamp"].append(_join(chunk[CREATED], chunk[HHMM]))
    stamps = parts.pop("stamp")
    frame = pd.DataFrame(
        {k: np.concatenate(v) if v else np.array([], dtype=object) for k, v in parts.items()}
    )
    # the creation stamp's byte order (YYYYMMDDHHMM) as an integer rank
    frame["stamp"] = (
        np.unique(np.concatenate(stamps), return_inverse=True)[1].reshape(-1)
        if stamps
        else np.array([], dtype=np.int64)
    )
    races = len(frame.drop_duplicates(VERSION_KEY))
    latest = records.latest_versions(frame, VERSION_KEY, created="stamp", kubun=KUBUN)
    jra = np.array([keys.jurisdiction(v) == JURISDICTION for v in latest[VENUE]], dtype=bool)
    use = np.zeros(len(frame), dtype=bool)
    use[latest.index.to_numpy()[jra]] = True
    counts = {
        "records": len(frame),
        "holdout_skipped": skipped,
        "versions_dropped": len(frame) - races,
        "deletions": races - len(latest),
        "non_jra": int((~jra).sum()),
        "kept": int(jra.sum()),
    }
    return use, counts


def _read(
    record_type: str, path: Path, layout: spec.RecordSpec, chunk_records: int, holdout
) -> tuple[list[pa.Table], dict[str, int]]:
    use, counts = _versions(path, layout, chunk_records, holdout)
    parts = []
    start = 0
    for chunk, _ in _chunks(path, layout, chunk_records, holdout):
        if chunk is None:
            continue
        n = len(chunk[KUBUN])
        mine = use[start : start + n]
        start += n
        if mine.any():
            parts.append(_table(record_type, chunk, mine))
    return parts, counts


# ------------------------------------------------------------------------------------------------
# Build


def _year_files(root: Path, years: Iterable[int] | None) -> dict[int, dict[str, Path]]:
    """Year -> record type -> file, over the DM and TM folders of the export."""
    found: dict[int, dict[str, Path]] = {}
    for rt in RECORD_TYPES:
        for p in (root / "stored" / DATASPEC / rt).glob("*.jvd"):
            if p.stem.isdigit() and len(p.stem) == 4:
                found.setdefault(int(p.stem), {})[rt] = p
    if years is None:
        if not found:
            raise FileNotFoundError(f"no DM or TM year file under {root / 'stored' / DATASPEC}")
        return dict(sorted(found.items()))
    wanted = sorted({int(y) for y in years})
    missing = [y for y in wanted if y not in found]
    if missing:
        raise FileNotFoundError(f"no DM or TM file for {missing[0]}")
    return {y: found[y] for y in wanted}


def build_mining_predictions(
    *,
    years: Iterable[int] | None = None,
    raw: Path | None = None,
    out: Path | None = None,
    chunk_records: int = 500,
) -> records.BuildReport:
    """Read the export's DM and TM year files (`raw` the export root, default
    `paths.raw_dir('jravan')`; `years` None for every year file of either folder), `chunk_records`
    records at a time, and write `<out>/mining_predictions.parquet` (default
    `paths.processed_dir('jravan')`), one row group per year. Before anything is written, an `out`
    inside the repository raises `paths.DataDirInsideRepo` and an export with no year file (no
    `years` given) raises FileNotFoundError, so a table already there stays. Returns the counts,
    no record content."""
    root = Path(raw) if raw is not None else paths.raw_dir(SOURCE)
    folder = (
        paths.processed_dir(SOURCE) if out is None else paths.outside_repo(Path(out), what="out")
    )
    target = folder / FILENAME
    files = _year_files(root, years)
    layouts = {rt: spec.record_spec(rt) for rt in RECORD_TYPES}
    cache: dict[date, bool] = {}

    def holdout(day: date) -> bool:
        if day not in cache:
            cache[day] = is_holdout_day(COUNTRY, day)
        return cache[day]

    by_year: dict[int, dict[str, int]] = {}
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        with pq.ParquetWriter(tmp, MINING_PREDICTIONS_SCHEMA, compression="zstd") as writer:
            for year, per_type in files.items():
                counts = dict.fromkeys(REPORT_KEYS, 0)
                parts: list[pa.Table] = []
                for rt, path in sorted(per_type.items()):
                    mine, read = _read(rt, path, layouts[rt], chunk_records, holdout)
                    parts += mine
                    for key, value in read.items():
                        counts[key] += value
                if parts:
                    table = pa.concat_tables(parts).sort_by(_SORT)
                    del parts
                    counts["rows"] = table.num_rows
                    if table.num_rows:
                        writer.write_table(table, row_group_size=table.num_rows)
                    del table
                by_year[year] = counts
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    totals = {key: sum(c[key] for c in by_year.values()) for key in REPORT_KEYS}
    return records.BuildReport(name=NAME, counts=totals, by_year=by_year, unmapped={}, failures=[])


def mining_command(
    year: Annotated[
        list[int] | None, typer.Option("--year", help="A year to read (repeatable).")
    ] = None,
) -> None:
    """Parse the JRA-VAN export's DM and TM mining predictions into
    processed/jravan/mining_predictions.parquet; the counts in reports/jravan-mining.json."""
    try:
        report = build_mining_predictions(years=year or None)
        written = records.write_build_report(report)
    except (paths.DataDirNotSet, ValueError, LookupError, OSError) as exc:
        _die(f"jravan mining failed: {type(exc).__name__}: {exc}")
    c = report.counts
    typer.echo(
        f"mining: {c['rows']} rows, {len(report.by_year)} years; {c['records']} records, "
        f"{c['versions_dropped']} versions dropped, {c['deletions']} deletions, "
        f"{c['non_jra']} non-JRA, {c['holdout_skipped']} on holdout days"
    )
    typer.echo(f"report: {written}")


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)
