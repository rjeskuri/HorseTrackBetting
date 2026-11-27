"""JV-Data JG (horses excluded from a race's entries) records -> the jravan `exclusions` table
(`horsetrack japan jravan exclusions`, #405).

`build_exclusions` reads `<raw>/stored/RACE/JG/<YYYY>.jvd` one year file at a time through
`records.iter_file`, keeps one version per entry (`records.latest_versions` on the race key,
血統登録番号 and 出馬投票受付順番; an entry whose latest record is a deletion has no row) and the
records of JRA venues, and writes `<out>/exclusions.parquet` through `records.write_parquet`,
sorted by `race_id`, `horse_id` (plain string order), then `entry_order`, so the same inputs give
the same bytes whatever the order of the records in the files.

One row per entry the record type lists, excluded or not (a consumer filters on the kinds):
`entry_kind` is 出走区分 and `exclusion_kind` 除外状態区分, translated by `ENTRY_KINDS` and
`EXCLUSION_KINDS`. An unset code (blank or 0) is null; a code outside the translation is null and
counted in the report's `unmapped`. `published_date` is データ作成年月日, null when unset. The
table is not a store table and has no availability rule yet (the owner's question, #405), so the
as-of views refuse it. The report holds counts only. See docs/specs/jravan.md.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Annotated, NoReturn

import numpy as np
import pandas as pd
import pyarrow as pa
import typer

from horsetrack.core import paths
from horsetrack.countries.japan.jravan import keys, records, spec

SOURCE = records.SOURCE
NAME = "exclusions"
FILENAME = "exclusions.parquet"
RECORD_TYPE = "JG"
JURISDICTION = "jra"
CREATED, KUBUN = "データ作成年月日", "データ区分"
YEAR, MMDD, VENUE, KAI, NICHI, RACE_NO = (
    "開催年", "開催月日", "競馬場コード", "開催回[第N回]", "開催日目[N日目]", "レース番号",
)  # fmt: skip
HORSE_ID, HORSE_NAME, ORDER = "血統登録番号", "馬名", "出馬投票受付順番"
ENTRY, EXCLUSION = "出走区分", "除外状態区分"
VERSION_KEY = [YEAR, MMDD, VENUE, KAI, NICHI, RACE_NO, HORSE_ID, ORDER]
READ = (*VERSION_KEY, CREATED, KUBUN, HORSE_NAME, ENTRY, EXCLUSION)

ENTRY_KINDS = {  # 出走区分
    "1": "declared",
    "2": "excluded_at_close",
    "4": "redeclared",
    "5": "excluded_on_redeclaration",
    "6": "scratched_before_numbering",
    "9": "scratched",
}
EXCLUSION_KINDS = {"1": "no_ballot", "2": "lost_ballot"}  # 除外状態区分
TRANSLATED = (("entry_kind", ENTRY, ENTRY_KINDS), ("exclusion_kind", EXCLUSION, EXCLUSION_KINDS))
REPORT_KEYS = ("records", "versions_dropped", "deletions", "non_jra", "no_race_id", "rows")

EXCLUSIONS_SCHEMA = pa.schema(
    [
        ("jurisdiction", pa.string()),
        ("race_id", pa.string()),
        ("race_date", pa.date32()),
        ("venue", pa.string()),
        ("race_no", pa.int64()),
        ("horse_id", pa.string()),
        ("horse_name", pa.string()),
        ("entry_order", pa.int64()),
        ("entry_kind", pa.string()),
        ("exclusion_kind", pa.string()),
        ("published_date", pa.date32()),
    ]
)


# ------------------------------------------------------------------------------------------------
# Converters


def _unset(value: str | None) -> bool:
    return value is None or set(value) == {"0"}


def _date(value: str | None) -> date | None:
    if _unset(value) or len(value) != 8 or not value.isdigit():
        return None
    try:
        return date(int(value[:4]), int(value[4:6]), int(value[6:]))
    except ValueError:
        return None


def _integer(value: str | None) -> int | None:
    return int(value) if value is not None and value.isascii() and value.isdigit() else None


def _translate(values: Iterable[str | None], meanings: dict[str, str], unmapped: dict) -> list:
    """Each code's meaning; an unset code None, an unknown one None and counted in `unmapped`."""
    out: list[str | None] = []
    for v in values:
        if _unset(v):
            out.append(None)
        elif v in meanings:
            out.append(meanings[v])
        else:
            unmapped[v] += 1
            out.append(None)
    return out


# ------------------------------------------------------------------------------------------------
# Records -> rows


def _read_year(path: Path, layout: spec.RecordSpec) -> pd.DataFrame:
    """The text of the items `READ` names, one row per record of the file, in file order."""
    parts: dict[str, list[np.ndarray]] = {k: [] for k in READ}
    for chunk in records.iter_file(path, layout):
        for k in READ:
            parts[k].append(records.text(chunk[k]))
    return pd.DataFrame(
        {k: np.concatenate(v) if v else np.array([], dtype=object) for k, v in parts.items()}
    )


def _rows(
    frame: pd.DataFrame, unmapped: dict[str, collections.Counter]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """One version per entry of `frame`, JRA venues only, as the table's columns plus the version
    key's tie-breakers (`_kai`, `_nichi`, `_order`); and the counts of those choices."""
    entries = len(frame.drop_duplicates(VERSION_KEY))
    latest = records.latest_versions(frame, VERSION_KEY)
    jra = np.array([keys.jurisdiction(v) == JURISDICTION for v in latest[VENUE]], dtype=bool)
    kept = latest[jra]
    race_ids = [
        keys.race_id(*p)
        for p in zip(kept[YEAR], kept[MMDD], kept[VENUE], kept[RACE_NO], strict=True)
    ]
    has_id = np.array([r is not None for r in race_ids], dtype=bool)
    kept = kept[has_id]
    race_ids = [r for r in race_ids if r is not None]
    counts = {
        "records": len(frame),
        "versions_dropped": len(frame) - entries,
        "deletions": entries - len(latest),
        "non_jra": int((~jra).sum()),
        "no_race_id": int((~has_id).sum()),
        "rows": len(kept),
    }
    split = [r.split(":") for r in race_ids]  # jra:YYYY-MM-DD:VENUE:NN
    rows = {
        "jurisdiction": [JURISDICTION] * len(kept),
        "race_id": race_ids,
        "race_date": [date.fromisoformat(s[1]) for s in split],
        "venue": [s[2] for s in split],
        "race_no": [int(s[3]) for s in split],
        "horse_id": list(kept[HORSE_ID]),
        "horse_name": list(kept[HORSE_NAME]),
        "entry_order": [_integer(v) for v in kept[ORDER]],
    }
    for column, item, meanings in TRANSLATED:
        rows[column] = _translate(kept[item], meanings, unmapped[column])
    rows["published_date"] = [_date(v) for v in kept[CREATED]]
    for column, item in (("_kai", KAI), ("_nichi", NICHI), ("_order", ORDER)):
        rows[column] = list(kept[item])
    return pd.DataFrame(rows), counts


def _table(frame: pd.DataFrame) -> pa.Table:
    """`frame` as an `EXCLUSIONS_SCHEMA` table sorted by race_id, horse_id, entry_order, the rest
    of the version key breaking ties."""
    by = ["race_id", "horse_id", "entry_order", "_kai", "_nichi", "_order"]
    frame = frame.sort_values(by, kind="stable", na_position="last") if len(frame) else frame
    columns = [
        pa.array(list(frame[name]) if name in frame else [], type=field.type)
        for name, field in zip(EXCLUSIONS_SCHEMA.names, EXCLUSIONS_SCHEMA, strict=True)
    ]
    return pa.Table.from_arrays(columns, schema=EXCLUSIONS_SCHEMA)


# ------------------------------------------------------------------------------------------------
# Build


def _year_files(root: Path, years: Iterable[int] | None) -> dict[int, Path]:
    """Year -> its JG file in the export."""
    found = {
        int(p.stem): p
        for p in (root / "stored" / "RACE" / RECORD_TYPE).glob("*.jvd")
        if p.stem.isdigit() and len(p.stem) == 4
    }
    if years is None:
        return dict(sorted(found.items()))
    wanted = sorted({int(y) for y in years})
    missing = [y for y in wanted if y not in found]
    if missing:
        raise FileNotFoundError(f"no {RECORD_TYPE} file for {missing[0]}")
    return {y: found[y] for y in wanted}


def build_exclusions(
    *,
    years: Iterable[int] | None = None,
    raw: Path | None = None,
    out: Path | None = None,
) -> records.BuildReport:
    """Read the export's JG year files (`raw` the export root, default `paths.raw_dir('jravan')`;
    `years` None for every year file), one at a time, and write `<out>/exclusions.parquet`
    (default `paths.processed_dir('jravan')`). Returns the counts, no record content."""
    root = Path(raw) if raw is not None else paths.raw_dir(SOURCE)
    target = (Path(out) if out is not None else paths.processed_dir(SOURCE)) / FILENAME
    files = _year_files(root, years)
    layout = spec.record_spec(RECORD_TYPE)
    unmapped = {column: collections.Counter() for column, _, _ in TRANSLATED}
    by_year: dict[int, dict[str, int]] = {}
    parts: list[pd.DataFrame] = []
    for year, path in files.items():
        frame = _read_year(path, layout)
        rows, by_year[year] = _rows(frame, unmapped)
        del frame
        parts.append(rows)
    table = _table(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())
    del parts
    records.write_parquet(table, target)
    totals = {key: sum(c[key] for c in by_year.values()) for key in REPORT_KEYS}
    return records.BuildReport(
        name=NAME,
        counts=totals,
        by_year=by_year,
        unmapped={column: dict(sorted(c.items())) for column, c in unmapped.items()},
        failures=[],
    )


def exclusions_command(
    year: Annotated[
        list[int] | None, typer.Option("--year", help="A year to read (repeatable).")
    ] = None,
) -> None:
    """Parse the JRA-VAN export's JG exclusions into processed/jravan/exclusions.parquet; the
    counts in reports/jravan-exclusions.json."""
    try:
        report = build_exclusions(years=year or None)
        written = records.write_build_report(report)
    except (paths.DataDirNotSet, ValueError, LookupError, OSError) as exc:
        _die(f"jravan exclusions failed: {type(exc).__name__}: {exc}")
    c = report.counts
    unmapped = sum(sum(codes.values()) for codes in report.unmapped.values())
    typer.echo(
        f"exclusions: {c['rows']} rows, {len(report.by_year)} years; {c['records']} records, "
        f"{c['versions_dropped']} versions dropped, {c['deletions']} deletions, "
        f"{c['non_jra']} non-JRA, {unmapped} unmapped codes"
    )
    typer.echo(f"report: {written}")


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)
