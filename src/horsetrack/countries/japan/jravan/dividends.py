"""JV-Data HR (payout) records -> the jravan store's `dividends` table (`horsetrack japan jravan
dividends`).

`build_dividends` reads `<raw>/stored/RACE/HR/<YYYY>.jvd` year by year with the layout
`spec.record_spec('HR')`, keeps one version per race (`records.latest_versions`) and JRA venues
only, and writes `<out>/dividends.parquet`, stable-sorted by `race_id`. Each paid combination is
one row, in the jra store's shape (`jra.netkeiba.parse_results._dividends`): the pools in `POOLS`
order, then the record's slot order; `combination` is the legs joined with `,`, leading zeros
dropped, FORECAST and TIERCE in finishing order and the others ascending; `is_refund` is false.

A slot is a row only when every leg and its payout are positive integers, so blank and zero slots
and unsold pools write nothing. A pool flagged not established (`不成立フラグ`) or special payout
(`特払フラグ`) writes no row, refund rows included. A refund flag (`返還フラグ`) writes no row of
its own, and the pool's paid combinations are still rows.

Refund rows (#443): for a record with refunded horse numbers (`返還馬番情報`), every combination
holding one, over the horse numbers 1..`登録頭数`, is a row of each horse-number pool with
`is_refund` true and a null `dividend`, in the paid rows' text; BRACKET_QUINELLA refunds every
bracket pair holding a bracket of `返還枠番情報` and the pair b,b of each bracket of `返還同枠情報`.
They follow the pool's paid rows and are written whatever the pool's refund flag; a combination
that also paid stays its paid row only. The report counts the flags per year, over the kept JRA
records, and its `rows` include the refund rows.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Annotated, NoReturn

import numpy as np
import pandas as pd
import pyarrow as pa
import typer

from horsetrack.core import paths
from horsetrack.core.store import schemas, tables
from horsetrack.countries.japan.jravan import keys, records, spec

SOURCE = records.SOURCE
RECORD_TYPE = "HR"
NAME = "dividends"
JURISDICTION = "jra"
FILENAME = "dividends.parquet"
VERSION_KEY = [
    "開催年", "開催月日", "競馬場コード", "開催回[第N回]", "開催日目[N日目]", "レース番号",
]  # fmt: skip
REFUNDED_HORSES = "返還馬番情報(馬番01～28)"
REFUNDED_BRACKETS = "返還枠番情報(枠番1～8)"  # every bracket pair holding one is refunded
REFUNDED_SAME = "返還同枠情報(枠番1～8)"  # the same-bracket pair b,b is refunded
REGISTERED = "登録頭数"  # the horse numbers a refund enumerates
BRACKETS = 8
FLAGS = {"not_established": "不成立", "special_payout": "特払", "refund": "返還"}
DROPPING_FLAGS = ("not_established", "special_payout")  # the pool writes no row
REPORT_KEYS = (*FLAGS, "refunded_horses", "races", "rows")


@dataclass(frozen=True)
class Pool:
    code: str  # the pool code of config/jra.yaml
    name: str  # the pool's name in the flag items
    group: str  # the payout group
    number: str  # the group's combination item
    legs: int
    ordered: bool  # legs in finishing order (else ascending)


POOLS = (
    Pool("WIN", "単勝", "単勝払戻", "馬番", 1, False),
    Pool("PLACE", "複勝", "複勝払戻", "馬番", 1, False),
    Pool("BRACKET_QUINELLA", "枠連", "枠連払戻", "組番", 2, False),
    Pool("QUINELLA", "馬連", "馬連払戻", "組番", 2, False),
    Pool("QUINELLA_PLACE", "ワイド", "ワイド払戻", "組番", 2, False),
    Pool("FORECAST", "馬単", "馬単払戻", "組番", 2, True),
    Pool("TRIO", "3連複", "3連複払戻", "組番", 3, False),
    Pool("TIERCE", "3連単", "3連単払戻", "組番", 3, True),
)


# ------------------------------------------------------------------------------------------------
# Records -> rows


def _race_ids(cols: Mapping[str, np.ndarray]) -> np.ndarray:
    """Each record's race id (None outside JRA or for a malformed key)."""
    parts = [records.text(cols[k]) for k in ("開催年", "開催月日", "競馬場コード", "レース番号")]
    return np.array([keys.race_id(*p) for p in zip(*parts, strict=True)], dtype=object)


def _slots(raw: np.ndarray, n: int) -> np.ndarray:
    """`raw` as (n, slots), a repeat of 1 included."""
    return raw.reshape(n, -1)


def _legs(raw: np.ndarray, legs: int) -> np.ndarray:
    """The (n, slots) combination items split into `legs` equal-width numbers, (n, slots, legs)
    float64 (NaN where a part is not an integer)."""
    width = raw.dtype.itemsize
    if width % legs:
        raise ValueError(f"a {width}-byte combination does not split into {legs} legs")
    size = width // legs
    n, slots = raw.shape
    matrix = np.ascontiguousarray(raw).view(np.uint8).reshape(n, slots, legs, size)
    parts = np.ascontiguousarray(matrix).view(f"S{size}").reshape(n, slots, legs)
    return records.integer(parts)


def _joined(numbers: np.ndarray) -> np.ndarray:
    """(rows, legs) positive integers -> `a,b,c` text per row."""
    out = numbers[:, 0].astype(np.int64).astype(str).astype(object)
    for leg in range(1, numbers.shape[1]):
        out = out + "," + numbers[:, leg].astype(np.int64).astype(str).astype(object)
    return out


def _empty() -> pd.DataFrame:
    columns = schemas.TABLES[NAME].columns
    return pd.DataFrame({c: pd.Series(dtype=object) for c in columns})


@cache
def _every(legs: int, ordered: bool, numbers: int, same: bool = False) -> tuple[np.ndarray, ...]:
    """Every combination of `legs` of 1..numbers, as (k, legs) int64 and its `a,b,c` text:
    ordered permutations, else ascending combinations (with the pairs a,a when `same`)."""
    if ordered:
        make = itertools.permutations
    else:
        make = itertools.combinations_with_replacement if same else itertools.combinations
    combos = np.array(list(make(range(1, numbers + 1), legs)), dtype=np.int64).reshape(-1, legs)
    return combos, _joined(combos) if len(combos) else np.array([], dtype=object)


def _refunds(pool: Pool, record: int, refund: Mapping[str, np.ndarray]) -> np.ndarray:
    """The text of `pool`'s refunded combinations in `record` (see the module docstring)."""
    if pool.code == "BRACKET_QUINELLA":
        combos, text = _every(2, False, BRACKETS, same=True)
        brackets = np.flatnonzero(refund["brackets"][record]) + 1
        same = np.flatnonzero(refund["same"][record]) + 1
        holding = np.isin(combos, brackets).any(axis=1)
        holding |= (combos[:, 0] == combos[:, 1]) & np.isin(combos[:, 0], same)
        return text[holding]
    runners = refund["registered"][record]
    if not runners >= 1:  # NaN included
        return np.array([], dtype=object)
    combos, text = _every(pool.legs, pool.ordered, int(runners))
    horses = np.flatnonzero(refund["horses"][record]) + 1
    return text[np.isin(combos, horses).any(axis=1)]


def dividends_from_records(cols: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """`records.decode` of HR records -> one row per paid combination of a JRA race and one refund
    row per combination holding a refunded horse or bracket, in the columns of
    `schemas.TABLES['dividends']`; per record the pools in `POOLS` order, then the paid slots in
    slot order, then the refund rows."""
    n = len(cols["開催年"])
    if n == 0:
        return _empty()
    race_ids = _race_ids(cols)
    is_jra = np.array([r is not None for r in race_ids], dtype=bool)
    refund = {
        "horses": _slots(cols[REFUNDED_HORSES], n) == b"1",
        "brackets": _slots(cols[REFUNDED_BRACKETS], n) == b"1",
        "same": _slots(cols[REFUNDED_SAME], n) == b"1",
        "registered": records.integer(cols[REGISTERED]),
    }
    refunding = is_jra & np.column_stack(
        [refund["horses"], refund["brackets"], refund["same"]]
    ).any(axis=1)
    parts = []  # (record, pool order, refund, slot, combination, dividend) per row
    for order, pool in enumerate(POOLS):
        numbers = _legs(_slots(cols[f"{pool.group}/{pool.number}"], n), pool.legs)
        pay = records.integer(_slots(cols[f"{pool.group}/払戻金"], n))
        paid = (pay > 0) & (np.nan_to_num(numbers, nan=0.0) > 0).all(axis=2)
        dropped = np.zeros(n, dtype=bool)
        for flag in DROPPING_FLAGS:
            dropped |= cols[f"{FLAGS[flag]}フラグ{pool.name}"] == b"1"
        paid &= (is_jra & ~dropped)[:, None]
        rec, slot = np.nonzero(paid)
        legs = numbers[rec, slot]
        if not pool.ordered:
            legs = np.sort(legs, axis=1)
        parts.append(
            pd.DataFrame(
                {
                    "record": rec,
                    "order": order,
                    "slot": slot,
                    "combination": _joined(legs) if len(rec) else np.array([], dtype=object),
                    "dividend": pay[rec, slot],
                    "refund": False,
                }
            )
        )
        for record in np.flatnonzero(refunding & ~dropped):
            text = _refunds(pool, int(record), refund)
            parts.append(
                pd.DataFrame(
                    {
                        "record": int(record),
                        "order": order,
                        "slot": np.arange(len(text)),
                        "combination": text,
                        "dividend": np.nan,
                        "refund": True,
                    }
                )
            )
    rows = pd.concat(parts, ignore_index=True)
    if rows.empty:
        return _empty()
    rows = rows.sort_values(["record", "order", "refund", "slot"], kind="stable", ignore_index=True)
    # A combination both paid and refunded stays its paid row: the store's keys stay unique.
    rows = rows[~(rows["refund"] & rows.duplicated(["record", "order", "combination"]))]
    return pd.DataFrame(
        {
            "jurisdiction": JURISDICTION,
            "race_id": race_ids[rows["record"].to_numpy()],
            "pool": np.array([p.code for p in POOLS], dtype=object)[rows["order"].to_numpy()],
            "combination": rows["combination"].to_numpy(),
            "dividend": rows["dividend"].to_numpy(dtype=np.float64),
            "is_refund": rows["refund"].to_numpy(dtype=bool),
        }
    )[list(schemas.TABLES[NAME].columns)]


# ------------------------------------------------------------------------------------------------
# Versions and flag counts


def _select(cols: Mapping[str, np.ndarray], index: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[index] for k, v in cols.items()}


def _kept(cols: Mapping[str, np.ndarray]) -> np.ndarray:
    """Positions of the JRA records kept: one version per race, deletions dropped."""
    frame = pd.DataFrame(
        {k: records.text(cols[k]) for k in (*VERSION_KEY, "データ作成年月日", "データ区分")}
    )
    jra = np.array([keys.jurisdiction(v) == JURISDICTION for v in frame["競馬場コード"]], bool)
    frame = frame[jra].reset_index(drop=True)
    positions = np.flatnonzero(jra)
    latest = records.latest_versions(frame, VERSION_KEY)
    return positions[latest.index.to_numpy()]


def _flag_counts(cols: Mapping[str, np.ndarray]) -> dict[int, dict[str, int]]:
    """Per 開催年, the races x pools flagged per flag and the refunded horse numbers."""
    years = records.integer(cols["開催年"])
    out: dict[int, dict[str, int]] = {}
    for year in np.unique(years[~np.isnan(years)]):
        mine = years == year
        counts = {
            key: sum(int((cols[f"{kind}フラグ{p.name}"][mine] == b"1").sum()) for p in POOLS)
            for key, kind in FLAGS.items()
        }
        counts["refunded_horses"] = int((cols[REFUNDED_HORSES][mine] == b"1").sum())
        counts["races"] = int(mine.sum())
        out[int(year)] = counts
    return out


# ------------------------------------------------------------------------------------------------
# Build


def _year_files(folder: Path, years: Iterable[int] | None) -> list[Path]:
    if years is None:
        return sorted(p for p in folder.glob("*.jvd") if p.stem.isdigit() and len(p.stem) == 4)
    files = [folder / f"{int(y):04d}.jvd" for y in sorted(set(years))]
    missing = [p for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"no HR file: {missing[0]}")
    return files


def _to_arrow(frame: pd.DataFrame) -> pa.Table:
    declared = schemas.TABLES[NAME].columns
    return pa.table(
        {
            # from_pandas: a refund row's NaN dividend is written null
            c: pa.array(frame[c].tolist(), type=tables.ARROW_TYPES[kind], from_pandas=True)
            for c, kind in declared.items()
        }
    )


def build_dividends(
    *, years: Iterable[int] | None = None, raw: Path | None = None, out: Path | None = None
) -> records.BuildReport:
    """Read the export's HR year files (`raw` the export root, default `paths.raw_dir('jravan')`;
    `years` None for every year file) and write `<out>/dividends.parquet` (default
    `paths.processed_dir('jravan')`). Returns the counts, no record content."""
    root = Path(raw) if raw is not None else paths.raw_dir(SOURCE)
    target = (Path(out) if out is not None else paths.processed_dir(SOURCE)) / FILENAME
    layout = spec.record_spec(RECORD_TYPE)
    frames: list[pd.DataFrame] = []
    by_year: dict[int, dict[str, int]] = {}
    for path in _year_files(root / "stored" / "RACE" / RECORD_TYPE, years):
        cols = records.decode(path.read_bytes(), layout)
        kept = _select(cols, _kept(cols))
        frame = dividends_from_records(kept)
        frames.append(frame)
        for year, counts in _flag_counts(kept).items():
            mine = by_year.setdefault(year, dict.fromkeys(REPORT_KEYS, 0))
            for key, value in counts.items():
                mine[key] += value
        if len(frame):
            race_year = frame["race_id"].str.slice(4, 8).astype(int)
            for year, rows in race_year.value_counts().items():
                by_year.setdefault(int(year), dict.fromkeys(REPORT_KEYS, 0))["rows"] += int(rows)
    frame = pd.concat(frames, ignore_index=True) if frames else _empty()
    frame = frame.sort_values("race_id", kind="stable", ignore_index=True)
    records.write_parquet(_to_arrow(frame), target)
    counts = {key: sum(y[key] for y in by_year.values()) for key in REPORT_KEYS}
    counts["rows"] = len(frame)
    return records.BuildReport(
        name=NAME, counts=counts, by_year=dict(sorted(by_year.items())), unmapped={}, failures=[]
    )


def dividends_command(
    year: Annotated[
        list[int] | None, typer.Option("--year", help="A year to read (repeatable).")
    ] = None,
) -> None:
    """Parse the JRA-VAN export's HR payouts into processed/jravan/dividends.parquet; the counts,
    flags included, in reports/jravan-dividends.json."""
    try:
        report = build_dividends(years=year or None)
        written = records.write_build_report(report)
    except (paths.DataDirNotSet, ValueError, LookupError, OSError) as exc:
        _die(f"jravan dividends failed: {type(exc).__name__}: {exc}")
    c = report.counts
    typer.echo(
        f"dividends: {c['rows']} rows, {c['races']} races, {len(report.by_year)} years; "
        f"flags: {c['not_established']} not established, {c['special_payout']} special payout, "
        f"{c['refund']} refund, {c['refunded_horses']} refunded horses"
    )
    typer.echo(f"report: {written}")


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)
