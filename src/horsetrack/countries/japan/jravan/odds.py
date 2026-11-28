"""JV-Data O1-O6 (final odds) records -> the jravan `final_odds` table (`horsetrack japan jravan
final-odds`).

`odds_rows` turns decoded O1-O6 arrays into long rows: one per record and filled slot of each sold
pool, plus one pool row (`combination` `__pool__`, `investment` the vote total in yen) per sold
pool. O1 holds WIN, PLACE and BRACKET_QUINELLA, O2 QUINELLA, O3 QUINELLA_PLACE, O4 FORECAST, O5
TRIO and O6 TIERCE. A pool is sold when its sale flag is `7`; any other flag (0 not sold, 1
cancelled before sale, 3 cancelled after sale) writes no row. `combination` is the printed number
split into its equal-width legs, leading zeros dropped, joined with `,`: FORECAST and TIERCE in the
printed order, the others ascending; a slot whose number is blank (or not all positive integers)
writes no row. `odds` (and `odds_hi`, the high end of PLACE and QUINELLA_PLACE) is the price with
one implied decimal place, null when it is not a positive number, the low price's raw text then in
`status_raw`.

`build_final_odds` reads `<raw>/stored/RACE/<O1..O6>/<YYYY>.jvd` one year at a time, streaming
each file in chunks (`records.iter_file`), keeps one version per race and record type
(`records.latest_versions`) and JRA venues only, sorts the year by `race_id, pool, combination`
and writes it as one row group per race day of `<out>/final_odds.parquet` (`records.write_days`,
#502), through a tmp file and `os.replace`.
Its availability rule is `label_only` (owner decision, #376; docs/specs/as-of.md): it is a label
or a conditioner, never a feature at t.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, NoReturn

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import typer

from horsetrack.core import paths
from horsetrack.countries.japan.jravan import keys, records, spec

SOURCE = records.SOURCE
NAME = "final-odds"  # the report's name: reports/jravan-final-odds.json
JURISDICTION = "jra"
FILENAME = "final_odds.parquet"
RECORD_TYPES = ("O1", "O2", "O3", "O4", "O5", "O6")
POOL_ROW = "__pool__"
SOLD = b"7"  # 発売フラグ: 0 not sold, 1 cancelled before sale, 3 cancelled after sale, 7 sold
VOTE_UNIT = 100  # yen per vote of a 票数合計
CHUNK_RECORDS = 2_000  # records per decoded chunk (fewer for O5/O6, capped by records.CHUNK_BYTES)
VERSION_KEY = [
    "開催年", "開催月日", "競馬場コード", "開催回[第N回]", "開催日目[N日目]", "レース番号",
]  # fmt: skip

FINAL_ODDS_SCHEMA = pa.schema(
    [
        ("jurisdiction", pa.string()),
        ("race_id", pa.string()),
        ("race_date", pa.date32()),
        ("venue", pa.string()),
        ("race_no", pa.int64()),
        ("pool", pa.string()),
        ("combination", pa.string()),
        ("odds", pa.float64()),
        ("odds_hi", pa.float64()),
        ("fav_rank", pa.int64()),
        ("investment", pa.float64()),
        ("status_raw", pa.string()),
        ("published_utc", pa.timestamp("us", tz="UTC")),
        ("data_kubun", pa.string()),
    ]
)
ROW_COLUMNS = FINAL_ODDS_SCHEMA.names[1:]
SORT_KEYS = [("race_id", "ascending"), ("pool", "ascending"), ("combination", "ascending")]


@dataclass(frozen=True)
class Pool:
    code: str  # the pool code of config/jra.yaml
    record_type: str
    sale: str  # the sale flag item
    group: str  # the repeat group
    number: str  # the group's combination item
    prices: tuple[str, ...]  # the price item, or the low and high items
    total: str  # the vote total item (units of 100 yen)
    legs: int
    ordered: bool  # legs in the printed order (else ascending)


_LOW_HIGH = ("最低オッズ", "最高オッズ")
POOLS = (
    Pool("WIN", "O1", "発売フラグ単勝", "単勝オッズ", "馬番", ("オッズ",), "単勝票数合計", 1,
         False),
    Pool("PLACE", "O1", "発売フラグ複勝", "複勝オッズ", "馬番", _LOW_HIGH, "複勝票数合計", 1,
         False),
    Pool("BRACKET_QUINELLA", "O1", "発売フラグ枠連", "枠連オッズ", "組番", ("オッズ",),
         "枠連票数合計", 2, False),
    Pool("QUINELLA", "O2", "発売フラグ馬連", "馬連オッズ", "組番", ("オッズ",), "馬連票数合計", 2,
         False),
    Pool("QUINELLA_PLACE", "O3", "発売フラグワイド", "ワイドオッズ", "組番", _LOW_HIGH,
         "ワイド票数合計", 2, False),
    Pool("FORECAST", "O4", "発売フラグ馬単", "馬単オッズ", "組番", ("オッズ",), "馬単票数合計", 2,
         True),
    Pool("TRIO", "O5", "発売フラグ3連複", "3連複オッズ", "組番", ("オッズ",), "3連複票数合計", 3,
         False),
    Pool("TIERCE", "O6", "発売フラグ3連単", "3連単オッズ", "組番", ("オッズ",), "3連単票数合計", 3,
         True),
)  # fmt: skip
POOL_CODES = tuple(p.code for p in POOLS)


# ------------------------------------------------------------------------------------------------
# Records -> rows


def _combinations(raw: np.ndarray, pool: Pool) -> tuple[list[str | None], np.ndarray]:
    """The distinct printed numbers of `raw` as combinations (None when blank or not all positive
    integers), and each element's index into them."""
    width = raw.dtype.itemsize
    if width % pool.legs:
        raise ValueError(f"{pool.code}: a {width}-byte number does not split into {pool.legs} legs")
    size = width // pool.legs
    values, inverse = np.unique(raw.reshape(-1), return_inverse=True)
    out: list[str | None] = []
    for value in values:
        text = bytes(value).ljust(width)
        parts = [text[i * size : (i + 1) * size] for i in range(pool.legs)]
        if not all(p.isdigit() and int(p) > 0 for p in parts):
            out.append(None)
            continue
        numbers = [int(p) for p in parts]
        out.append(",".join(map(str, numbers if pool.ordered else sorted(numbers))))
    return out, inverse.reshape(raw.shape)


def _positive(values: np.ndarray) -> pa.Array:
    return pa.array(values, type=pa.float64(), mask=~(values > 0))


def _raw_text(raw: np.ndarray, keep: np.ndarray) -> pa.Array:
    """`raw` as printed (cp932, untrimmed) where `keep`, else null."""
    out = np.full(len(raw), None, dtype=object)
    if keep.any():
        values, inverse = np.unique(raw[keep], return_inverse=True)
        decoded = np.array([bytes(v).decode(records.ENCODING, errors="replace") for v in values])
        out[keep] = decoded[inverse.reshape(-1)]
    return pa.array(out, type=pa.string())


def _pool_rows(pool: Pool, cols: Mapping[str, np.ndarray], sold: np.ndarray) -> dict:
    """The slot rows and the pool rows of `pool`, as per-row arrays; `record` the record index."""
    n = len(sold)
    numbers = cols[f"{pool.group}/{pool.number}"].reshape(n, -1)
    combos, index = _combinations(numbers, pool)
    valid = np.array([c is not None for c in combos], dtype=bool)
    rec, slot = np.nonzero(valid[index] & sold[:, None])

    def slots(item: str) -> np.ndarray:
        return cols[f"{pool.group}/{item}"].reshape(n, -1)[rec, slot]

    low_raw = slots(pool.prices[0])
    low = records.scaled(low_raw, 1)
    high = (
        _positive(records.scaled(slots(pool.prices[1]), 1))
        if len(pool.prices) == 2
        else pa.nulls(len(rec), pa.float64())
    )
    rank = records.integer(slots("人気順"))
    totals = np.flatnonzero(sold)
    investment = records.integer(cols[pool.total][totals]) * VOTE_UNIT
    m = len(totals)
    return {
        "record": np.concatenate([rec, totals]),
        "combination": pa.concat_arrays(
            [
                pa.array(combos, type=pa.string()).take(pa.array(index[rec, slot])),
                pa.array([POOL_ROW] * m, type=pa.string()),
            ]
        ),
        "odds": pa.concat_arrays([_positive(low), pa.nulls(m, pa.float64())]),
        "odds_hi": pa.concat_arrays([high, pa.nulls(m, pa.float64())]),
        "fav_rank": pa.concat_arrays(
            [
                pa.array(rank, type=pa.float64(), mask=~(rank > 0)).cast(pa.int64()),
                pa.nulls(m, pa.int64()),
            ]
        ),
        "investment": pa.concat_arrays(
            [
                pa.nulls(len(rec), pa.float64()),
                pa.array(investment, type=pa.float64(), from_pandas=True),
            ]
        ),
        "status_raw": pa.concat_arrays([_raw_text(low_raw, ~(low > 0)), pa.nulls(m, pa.string())]),
    }


def _rows(record_type: str, cols: Mapping[str, np.ndarray]) -> tuple[pa.Table, np.ndarray]:
    """`odds_rows`, with each row's record index (into `cols`)."""
    pools = [p for p in POOLS if p.record_type == record_type]
    if not pools:
        raise LookupError(f"{record_type!r} is not an O1-O6 record type")
    year, mmdd, venue, race = (
        records.text(cols[k]) for k in ("開催年", "開催月日", "競馬場コード", "レース番号")
    )
    stamp = records.text(cols["発表月日時分"])
    race_ids = np.array(
        [keys.race_id(*p) for p in zip(year, mmdd, venue, race, strict=True)], dtype=object
    )
    jra = np.array([r is not None for r in race_ids], dtype=bool)
    parts = []
    for pool in pools:
        part = _pool_rows(pool, cols, jra & (cols[pool.sale] == SOLD))
        part["pool"] = np.full(len(part["record"]), POOL_CODES.index(pool.code), dtype=np.int64)
        parts.append(part)
    record = np.concatenate([p["record"] for p in parts]).astype(np.int64)
    if len(record) == 0:
        return FINAL_ODDS_SCHEMA.empty_table().select(ROW_COLUMNS), record

    # Per-record values, taken per row.
    split = [r.split(":") if r is not None else [None, None, None, None] for r in race_ids]
    per_record = {
        "race_id": pa.array(race_ids, type=pa.string()),
        "race_date": pa.array(
            [date.fromisoformat(s[1]) if s[1] else None for s in split], type=pa.date32()
        ),
        "venue": pa.array([s[2] for s in split], type=pa.string()),
        "race_no": pa.array([int(s[3]) if s[3] else None for s in split], type=pa.int64()),
        "published_utc": pa.array(
            [
                keys.announced_utc(y, d, t) if r is not None else None
                for y, d, t, r in zip(year, mmdd, stamp, race_ids, strict=True)
            ],
            type=FINAL_ODDS_SCHEMA.field("published_utc").type,
        ),
        "data_kubun": pa.array(records.text(cols["データ区分"]), type=pa.string()),
    }
    take = pa.array(record)
    columns = {name: per_record[name].take(take) for name in ("race_id", "race_date", "venue",
                                                              "race_no")}  # fmt: skip
    pool_codes = pa.array(POOL_CODES, type=pa.string())
    columns["pool"] = pool_codes.take(pa.array(np.concatenate([p["pool"] for p in parts])))
    for name in ("combination", "odds", "odds_hi", "fav_rank", "investment", "status_raw"):
        columns[name] = pa.concat_arrays([p[name] for p in parts])
    columns["published_utc"] = per_record["published_utc"].take(take)
    columns["data_kubun"] = per_record["data_kubun"].take(take)
    schema = pa.schema([FINAL_ODDS_SCHEMA.field(c) for c in ROW_COLUMNS])
    return pa.table([columns[c] for c in ROW_COLUMNS], schema=schema), record


def odds_rows(record_type: str, cols: Mapping[str, np.ndarray]) -> pa.Table:
    """`records.decode` of `record_type` (O1..O6) records -> one row per JRA record and filled slot
    of each sold pool, plus one pool row per sold pool, in the columns `ROW_COLUMNS` with the types
    of `FINAL_ODDS_SCHEMA`; per record type, the pools in `POOLS` order, slots before pool rows."""
    return _rows(record_type, cols)[0]


# ------------------------------------------------------------------------------------------------
# Build


def _year_files(folder: Path) -> dict[int, Path]:
    if not folder.is_dir():
        return {}
    return {int(p.stem): p for p in folder.glob("*.jvd") if p.stem.isdigit() and len(p.stem) == 4}


def _read_file(path: Path, layout: spec.RecordSpec) -> tuple[pa.Table, np.ndarray]:
    """The rows of one year file's kept JRA records (one version per race), and the kept
    records' 開催年."""
    tables: list[pa.Table] = []
    record_parts: list[np.ndarray] = []
    key_frames: list[pd.DataFrame] = []
    offset = 0
    for cols in records.iter_file(path, layout, chunk_records=CHUNK_RECORDS):
        n = len(cols["開催年"])
        table, record = _rows(layout.record_type, cols)
        tables.append(table)
        record_parts.append(record + offset)
        key_frames.append(
            pd.DataFrame(
                {k: records.text(cols[k]) for k in (*VERSION_KEY, "データ作成年月日", "データ区分")}
            )
        )
        offset += n
    if not key_frames:
        return FINAL_ODDS_SCHEMA.empty_table().select(ROW_COLUMNS), np.array([], dtype=np.int64)
    frame = pd.concat(key_frames, ignore_index=True)
    jra = np.array([keys.jurisdiction(v) == JURISDICTION for v in frame["競馬場コード"]], bool)
    kept_frame = records.latest_versions(frame[jra], VERSION_KEY)
    kept = np.zeros(len(frame), dtype=bool)
    kept[kept_frame.index.to_numpy()] = True
    out = []
    for table, record in zip(tables, record_parts, strict=True):
        out.append(table.filter(pa.array(kept[record], type=pa.bool_())))
    years = np.array([int(y) for y in kept_frame["開催年"]], dtype=np.int64)
    return pa.concat_tables(out), years


def _write(target: Path, years: list[int], read_year) -> dict[int, dict[str, int]]:
    """One row group per race day of each year with rows, in `years` order, through a tmp file
    and `os.replace`; the rows per pool of each year written."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:12]}.tmp")
    rows: dict[int, dict[str, int]] = {}
    try:
        with pq.ParquetWriter(tmp, FINAL_ODDS_SCHEMA, compression="zstd") as writer:
            for year in years:
                table = read_year(year)
                if table.num_rows == 0:
                    continue
                race_years = pc.utf8_slice_codeunits(table["race_id"], 4, 8)
                other = pc.not_equal(race_years, pa.scalar(f"{year:04d}"))
                if pc.any(other).as_py():
                    raise ValueError(f"the {year} files hold races of another year")
                table = table.sort_by(SORT_KEYS)
                table = table.add_column(
                    0, "jurisdiction", pa.array([JURISDICTION] * table.num_rows, pa.string())
                ).cast(FINAL_ODDS_SCHEMA)
                records.write_days(writer, table)
                counts = pc.value_counts(table["pool"]).to_pylist()
                rows[year] = {c["values"]: int(c["counts"]) for c in counts}
                del table
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return rows


def build_final_odds(
    *, years: Iterable[int] | None = None, raw: Path | None = None, out: Path | None = None
) -> records.BuildReport:
    """Read the export's O1-O6 year files (`raw` the export root, default `paths.raw_dir('jravan')`;
    `years` None for every year file of every type, a type without a file for a year adds nothing)
    and write `<out>/final_odds.parquet` (default `paths.processed_dir('jravan')`), one row group
    per race day. Returns the counts, no record content."""
    root = Path(raw) if raw is not None else paths.raw_dir(SOURCE)
    target = (Path(out) if out is not None else paths.processed_dir(SOURCE)) / FILENAME
    files = {rt: _year_files(root / "stored" / "RACE" / rt) for rt in RECORD_TYPES}
    wanted = sorted(
        {y for f in files.values() for y in f} if years is None else {int(y) for y in years}
    )
    layouts: dict[str, spec.RecordSpec] = {}
    kept_records: dict[int, int] = {}

    def read_year(year: int) -> pa.Table:
        parts = []
        for rt in RECORD_TYPES:
            path = files[rt].get(year)
            if path is None:
                continue
            if rt not in layouts:
                layouts[rt] = spec.record_spec(rt)
            table, record_years = _read_file(path, layouts[rt])
            parts.append(table)
            for y, k in zip(*np.unique(record_years, return_counts=True), strict=True):
                kept_records[int(y)] = kept_records.get(int(y), 0) + int(k)
        if not parts:
            return FINAL_ODDS_SCHEMA.empty_table().select(ROW_COLUMNS)
        return pa.concat_tables(parts)

    rows = _write(target, wanted, read_year)
    by_year = {}
    for year, per_pool in sorted(rows.items()):
        mine = {"records": kept_records.get(year, 0), "rows": sum(per_pool.values())}
        mine.update({code: per_pool.get(code, 0) for code in POOL_CODES})
        by_year[year] = mine
    counts = {
        "records": sum(kept_records.values()),
        "rows": sum(y["rows"] for y in by_year.values()),
    }
    counts.update({code: sum(y[code] for y in by_year.values()) for code in POOL_CODES})
    return records.BuildReport(name=NAME, counts=counts, by_year=by_year, unmapped={}, failures=[])


def final_odds_command(
    year: Annotated[
        list[int] | None, typer.Option("--year", help="A year to read (repeatable).")
    ] = None,
) -> None:
    """Parse the JRA-VAN export's O1-O6 final odds into processed/jravan/final_odds.parquet; the
    counts in reports/jravan-final-odds.json."""
    try:
        report = build_final_odds(years=year or None)
        written = records.write_build_report(report)
    except (paths.DataDirNotSet, ValueError, LookupError, OSError) as exc:
        _die(f"jravan final-odds failed: {type(exc).__name__}: {exc}")
    c = report.counts
    typer.echo(
        f"final_odds: {c['rows']} rows, {c['records']} records, {len(report.by_year)} years; "
        + ", ".join(f"{code} {c[code]}" for code in POOL_CODES)
    )
    typer.echo(f"report: {written}")


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)
