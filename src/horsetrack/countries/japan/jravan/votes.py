"""JV-Data H1 (votes, Win to Trio) and H6 (votes, Trifecta) records -> the jravan `pool_votes`
table (`horsetrack japan jravan votes`, #368).

`build_pool_votes` reads `<raw>/stored/RACE/H1|H6/<YYYY>.jvd` year by year, streamed through
`records.iter_file`: a first pass over a file reads the version keys and picks one version per
race (`records.latest_versions`), a second turns the kept records of JRA venues into rows, so no
more than one chunk of records is decoded at a time. Each year's rows (H1 and H6) are sorted by
`race_id, pool, combination` (plain string order) and written as one row group of
`<out>/pool_votes.parquet`, through a tmp file and `os.replace`; the same inputs give the same
bytes.

Each registered slot of a sold pool (`発売フラグ<P>` 7) is one row: `combination` the legs of its
馬番 or 組番 joined with `,`, leading zeros dropped (FORECAST and TIERCE in finishing order, the
others ascending), `votes` its 票数 (100-yen tickets) and `fav_rank` its 人気順. A slot is a row
when its combination is digits only and every leg is positive, whatever its votes. Each sold pool
also has a `__pool__` row: `votes` its 票数合計 and `refund_votes` its 返還票数合計. A count or a
rank that is not a number is null. The table is not a store table and has no availability rule
(owner decision), so the views refuse it. See docs/specs/jravan.md.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path
from typing import Annotated, NoReturn

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from horsetrack.core import paths
from horsetrack.countries.japan.jravan import keys, records, spec

SOURCE = records.SOURCE
NAME = "votes"
FILENAME = "pool_votes.parquet"
JURISDICTION = "jra"
RECORD_TYPES = ("H1", "H6")
POOL_ROW = "__pool__"
SOLD = b"7"
VERSION_KEY = [
    "開催年", "開催月日", "競馬場コード", "開催回[第N回]", "開催日目[N日目]", "レース番号",
]  # fmt: skip
CREATED, KUBUN, VENUE = "データ作成年月日", "データ区分", "競馬場コード"
RACE_KEY = ("開催年", "開催月日", "競馬場コード", "レース番号")  # keys.race_id's arguments

POOL_VOTES_SCHEMA = pa.schema(
    [
        ("jurisdiction", pa.string()),
        ("race_id", pa.string()),
        ("race_date", pa.date32()),
        ("venue", pa.string()),
        ("race_no", pa.int64()),
        ("pool", pa.string()),
        ("combination", pa.string()),
        ("votes", pa.int64()),
        ("refund_votes", pa.int64()),
        ("fav_rank", pa.int64()),
        ("data_kubun", pa.string()),
    ]
)


@dataclass(frozen=True)
class Pool:
    code: str  # the pool code of docs/specs/odds-log.md
    name: str  # the pool's name in the item names
    number: str  # the votes group's combination item
    legs: int
    ordered: bool  # legs in finishing order (else ascending)

    @property
    def group(self) -> str:
        return f"{self.name}票数"


POOLS = (
    Pool("WIN", "単勝", "馬番", 1, False),
    Pool("PLACE", "複勝", "馬番", 1, False),
    Pool("BRACKET_QUINELLA", "枠連", "組番", 2, False),
    Pool("QUINELLA", "馬連", "組番", 2, False),
    Pool("QUINELLA_PLACE", "ワイド", "組番", 2, False),
    Pool("FORECAST", "馬単", "組番", 2, True),
    Pool("TRIO", "3連複", "組番", 3, False),
    Pool("TIERCE", "3連単", "組番", 3, True),
)
POOLS_OF = {"H1": POOLS[:7], "H6": POOLS[7:]}
_POOL_CODES = pa.array([p.code for p in POOLS], pa.string())
_POOL_RANK = np.argsort(np.argsort([p.code for p in POOLS]))  # plain string order of the codes
_LEG_BASE = 100  # a combination's legs as one integer in base 100 (a leg has at most 2 digits)
REPORT_KEYS = (
    "records", "versions_dropped", "deletions", "non_jra", "kept", "races", "rows",
    *(f"{kind}_{p.code}" for p in POOLS for kind in ("combinations", "pools", "refunds")),
)  # fmt: skip


# ------------------------------------------------------------------------------------------------
# Records -> rows


def _number(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`records.integer` of an `S<n>` array, flattened, as (int64 values, is a number): plain
    digits are parsed in bulk, anything else through `records.integer`."""
    raw = np.ascontiguousarray(raw).reshape(-1)
    m, width = raw.size, raw.dtype.itemsize
    values = np.zeros(m, dtype=np.int64)
    if m == 0:
        return values, np.zeros(0, dtype=bool)
    u8 = raw.view(np.uint8).reshape(m, width)
    ok = ((u8 >= ord("0")) & (u8 <= ord("9"))).all(axis=1)
    if ok.any():
        powers = 10 ** np.arange(width - 1, -1, -1, dtype=np.int64)
        values[ok] = (u8[ok].astype(np.int64) - ord("0")) @ powers
    other = np.flatnonzero(~ok)
    if other.size:
        parsed = records.integer(raw[other])
        good = ~np.isnan(parsed)
        values[other[good]] = parsed[good].astype(np.int64)
        ok[other[good]] = True
    return values, ok


def _combinations(raw: np.ndarray, legs: int, ordered: bool) -> tuple[np.ndarray, np.ndarray]:
    """(n, slots) combination items -> (is a combination, code), both (n, slots): every byte a
    digit and each of the `legs` equal-width legs positive; the code is the legs (ascending unless
    `ordered`) as one integer in base `_LEG_BASE`, the first leg most significant."""
    n, slots = raw.shape
    width = raw.dtype.itemsize
    if width % legs:
        raise ValueError(f"a {width}-byte combination does not split into {legs} legs")
    size = width // legs
    u8 = np.ascontiguousarray(raw).view(np.uint8).reshape(n, slots, width)
    digits = ((u8 >= ord("0")) & (u8 <= ord("9"))).all(axis=2)
    powers = 10 ** np.arange(size - 1, -1, -1, dtype=np.int32)
    numbers = (u8.reshape(n, slots, legs, size).astype(np.int32) - ord("0")) @ powers
    valid = digits & (numbers > 0).all(axis=2)
    if not ordered:
        numbers = np.sort(numbers, axis=2)
    code = numbers @ (_LEG_BASE ** np.arange(legs - 1, -1, -1, dtype=np.int32))
    return valid, code


def _combination_text(code: int) -> str:
    """A combination code as `a,b,c`; -1 is the pool row."""
    if code < 0:
        return POOL_ROW
    legs = []
    while code:
        code, leg = divmod(code, _LEG_BASE)
        legs.append(str(leg))
    return ",".join(reversed(legs))


@dataclass
class _Rows:
    """Rows as arrays: per record its race id and data kubun; per row its record, pool (a `POOLS`
    index), combination code (-1 on a pool row) and its counts with their is-a-number masks."""

    race_id: np.ndarray  # object, per record
    kubun: np.ndarray  # object, per record
    rec: np.ndarray  # int64, per row from here on
    pool: np.ndarray  # int8
    code: np.ndarray  # int32
    votes: np.ndarray  # int64
    votes_ok: np.ndarray  # bool
    refund: np.ndarray
    refund_ok: np.ndarray
    fav: np.ndarray
    fav_ok: np.ndarray

    @classmethod
    def concat(cls, parts: list[_Rows]) -> _Rows:
        offsets = np.cumsum([0] + [len(p.race_id) for p in parts[:-1]])
        out = {}
        for f in fields(cls):
            arrays = [getattr(p, f.name) for p in parts]
            if f.name == "rec":
                arrays = [a + offset for a, offset in zip(arrays, offsets, strict=True)]
            out[f.name] = np.concatenate(arrays)
        return cls(**out)


def _rows(record_type: str, cols: Mapping[str, np.ndarray], use: np.ndarray | None) -> _Rows:
    """The rows of the records of `cols` (those of `use`, when given) at a JRA venue."""
    n = len(cols[KUBUN])
    parts = [records.text(cols[k]) for k in RACE_KEY]
    race_ids = np.array([keys.race_id(*p) for p in zip(*parts, strict=True)], dtype=object)
    keep = np.array([r is not None for r in race_ids], dtype=bool)
    if use is not None:
        keep &= use
    kept = np.flatnonzero(keep)
    local = np.full(n, -1, dtype=np.int64)
    local[kept] = np.arange(len(kept))
    out: dict[str, list[np.ndarray]] = {
        f.name: [] for f in fields(_Rows) if f.name not in ("race_id", "kubun")
    }

    def add(rec: np.ndarray, pool: int, code, votes, refund, fav) -> None:
        m = len(rec)
        out["rec"].append(local[rec])
        out["pool"].append(np.full(m, pool, dtype=np.int8))
        out["code"].append(np.broadcast_to(np.asarray(code, dtype=np.int32), (m,)))
        for name, (values, ok) in (("votes", votes), ("refund", refund), ("fav", fav)):
            out[name].append(values if values is not None else np.zeros(m, dtype=np.int64))
            out[f"{name}_ok"].append(ok if ok is not None else np.zeros(m, dtype=bool))

    absent = (None, None)
    for pool in POOLS_OF[record_type]:
        index = POOLS.index(pool)
        sold = keep & (cols[f"発売フラグ{pool.name}"] == SOLD)
        combination = cols[f"{pool.group}/{pool.number}"].reshape(n, -1)
        valid, code = _combinations(combination, pool.legs, pool.ordered)
        rec, slot = np.nonzero(valid & sold[:, None])
        votes = _number(cols[f"{pool.group}/票数"].reshape(n, -1)[rec, slot])
        fav = _number(cols[f"{pool.group}/人気順"].reshape(n, -1)[rec, slot])
        add(rec, index, code[rec, slot], votes, absent, fav)
        rec = np.flatnonzero(sold)
        total = _number(cols[f"{pool.name}票数合計"][rec])
        refund = _number(cols[f"{pool.name}返還票数合計"][rec])
        add(rec, index, -1, total, refund, absent)
    return _Rows(
        race_id=race_ids[kept],
        kubun=records.text(cols[KUBUN])[kept],
        **{name: np.concatenate(arrays) for name, arrays in out.items()},
    )


def _table(rows: _Rows) -> pa.Table:
    """`rows` as a `POOL_VOTES_SCHEMA` table, sorted by race_id, pool, combination."""
    codes, combination = np.unique(rows.code, return_inverse=True)
    texts = [_combination_text(int(c)) for c in codes]
    text_rank = np.argsort(np.argsort(np.array(texts, dtype=object)))
    ids = [str(r) for r in rows.race_id]
    # a race's H1 and H6 records share a rank
    race_rank = np.unique(np.array(ids, dtype=str), return_inverse=True)[1].reshape(-1)
    order = np.lexsort(
        (text_rank[combination], _POOL_RANK[rows.pool], race_rank[rows.rec].astype(np.int64))
    )
    rec = pa.array(rows.rec[order])
    split = [r.split(":") for r in ids]  # jra:YYYY-MM-DD:VENUE:NN

    def counts(values: np.ndarray, ok: np.ndarray) -> pa.Array:
        return pa.array(values[order], type=pa.int64(), mask=~ok[order])

    columns = [
        pa.repeat(pa.scalar(JURISDICTION, pa.string()), len(order)),
        pa.array(ids, pa.string()).take(rec),
        pa.array([date.fromisoformat(s[1]) for s in split], pa.date32()).take(rec),
        pa.array([s[2] for s in split], pa.string()).take(rec),
        pa.array([int(s[3]) for s in split], pa.int64()).take(rec),
        _POOL_CODES.take(pa.array(rows.pool[order].astype(np.int64))),
        pa.array(texts, pa.string()).take(pa.array(combination.reshape(-1)[order])),
        counts(rows.votes, rows.votes_ok),
        counts(rows.refund, rows.refund_ok),
        counts(rows.fav, rows.fav_ok),
        pa.array(list(rows.kubun), pa.string()).take(rec),
    ]
    return pa.Table.from_arrays(columns, schema=POOL_VOTES_SCHEMA)


def votes_rows(record_type: str, cols: Mapping[str, np.ndarray]) -> pa.Table:
    """`records.decode` of H1 or H6 records -> their rows (`POOL_VOTES_SCHEMA`): one per
    registered slot of a sold pool and one `__pool__` row per sold pool, JRA venues only."""
    if record_type not in POOLS_OF:
        raise ValueError(f"not a votes record type: {record_type!r}")
    return _table(_rows(record_type, cols, None))


# ------------------------------------------------------------------------------------------------
# Versions


def _versions(
    path: Path, layout: spec.RecordSpec, chunk_records: int
) -> tuple[np.ndarray, dict[str, int]]:
    """A first pass over one file: which records to keep (one version per race, deletions
    dropped, JRA venues only) and the counts of those choices."""
    names = (*VERSION_KEY, CREATED, KUBUN)
    parts: dict[str, list[np.ndarray]] = {k: [] for k in names}
    for chunk in records.iter_file(path, layout, chunk_records=chunk_records):
        for k in names:
            parts[k].append(records.text(chunk[k]))
    frame = pd.DataFrame(
        {k: np.concatenate(v) if v else np.array([], dtype=object) for k, v in parts.items()}
    )
    races = len(frame.drop_duplicates(VERSION_KEY))
    latest = records.latest_versions(frame, VERSION_KEY)
    jra = np.array([keys.jurisdiction(v) == JURISDICTION for v in latest[VENUE]], dtype=bool)
    use = np.zeros(len(frame), dtype=bool)
    use[latest.index.to_numpy()[jra]] = True
    counts = {
        "records": len(frame),
        "versions_dropped": len(frame) - races,
        "deletions": races - len(latest),
        "non_jra": int((~jra).sum()),
        "kept": int(jra.sum()),
    }
    return use, counts


def _read(
    record_type: str, path: Path, layout: spec.RecordSpec, chunk_records: int
) -> tuple[list[_Rows], dict[str, int]]:
    use, counts = _versions(path, layout, chunk_records)
    parts = []
    start = 0
    for chunk in records.iter_file(path, layout, chunk_records=chunk_records):
        n = len(chunk[KUBUN])
        mine = use[start : start + n]
        start += n
        if mine.any():
            parts.append(_rows(record_type, chunk, mine))
    return parts, counts


# ------------------------------------------------------------------------------------------------
# Build


def _year_files(root: Path, years: Iterable[int] | None) -> dict[int, dict[str, Path]]:
    """Year -> record type -> file, over the H1 and H6 folders of the export."""
    found: dict[int, dict[str, Path]] = {}
    for rt in RECORD_TYPES:
        for p in (root / "stored" / "RACE" / rt).glob("*.jvd"):
            if p.stem.isdigit() and len(p.stem) == 4:
                found.setdefault(int(p.stem), {})[rt] = p
    if years is None:
        return dict(sorted(found.items()))
    wanted = sorted({int(y) for y in years})
    missing = [y for y in wanted if y not in found]
    if missing:
        raise FileNotFoundError(f"no H1 or H6 file for {missing[0]}")
    return {y: found[y] for y in wanted}


def _row_counts(rows: _Rows) -> dict[str, int]:
    """The races with a row, the rows, and per pool its combination rows, pool rows and pool rows
    with refunded votes."""
    with_rows = rows.race_id[np.unique(rows.rec)]
    counts = {"races": len(set(with_rows)), "rows": len(rows.rec)}
    on_pool = rows.code < 0
    for index, p in enumerate(POOLS):
        mine = rows.pool == index
        counts[f"combinations_{p.code}"] = int((mine & ~on_pool).sum())
        counts[f"pools_{p.code}"] = int((mine & on_pool).sum())
        counts[f"refunds_{p.code}"] = int((mine & on_pool & rows.refund_ok).sum())
    return counts


def build_pool_votes(
    *,
    years: Iterable[int] | None = None,
    raw: Path | None = None,
    out: Path | None = None,
    chunk_records: int = 500,
) -> records.BuildReport:
    """Read the export's H1 and H6 year files (`raw` the export root, default
    `paths.raw_dir('jravan')`; `years` None for every year file of either folder), `chunk_records`
    records at a time, and write `<out>/pool_votes.parquet` (default
    `paths.processed_dir('jravan')`), one row group per year. Returns the counts, no record
    content."""
    root = Path(raw) if raw is not None else paths.raw_dir(SOURCE)
    target = (Path(out) if out is not None else paths.processed_dir(SOURCE)) / FILENAME
    files = _year_files(root, years)
    layouts = {rt: spec.record_spec(rt) for rt in RECORD_TYPES}
    by_year: dict[int, dict[str, int]] = {}
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        with pq.ParquetWriter(tmp, POOL_VOTES_SCHEMA, compression="zstd") as writer:
            for year, per_type in files.items():
                counts = dict.fromkeys(REPORT_KEYS, 0)
                parts: list[_Rows] = []
                for rt, path in sorted(per_type.items()):
                    mine, read = _read(rt, path, layouts[rt], chunk_records)
                    parts += mine
                    for key, value in read.items():
                        counts[key] += value
                if parts:
                    rows = _Rows.concat(parts)
                    del parts
                    counts.update(_row_counts(rows))
                    table = _table(rows)
                    del rows
                    if table.num_rows:
                        writer.write_table(table, row_group_size=table.num_rows)
                    del table
                by_year[year] = counts
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    totals = {key: sum(c[key] for c in by_year.values()) for key in REPORT_KEYS}
    return records.BuildReport(name=NAME, counts=totals, by_year=by_year, unmapped={}, failures=[])


def votes_command(
    year: Annotated[
        list[int] | None, typer.Option("--year", help="A year to read (repeatable).")
    ] = None,
) -> None:
    """Parse the JRA-VAN export's H1 and H6 votes into processed/jravan/pool_votes.parquet; the
    counts in reports/jravan-votes.json."""
    try:
        report = build_pool_votes(years=year or None)
        written = records.write_build_report(report)
    except (paths.DataDirNotSet, ValueError, LookupError, OSError) as exc:
        _die(f"jravan votes failed: {type(exc).__name__}: {exc}")
    c = report.counts
    typer.echo(
        f"votes: {c['rows']} rows, {c['races']} races, {len(report.by_year)} years; "
        f"{c['records']} records, {c['versions_dropped']} versions dropped, "
        f"{c['deletions']} deletions, {c['non_jra']} non-JRA"
    )
    typer.echo(f"report: {written}")


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)
