"""JV-Data HN (breeding horse), SK (produce) and BT (lineage) records -> the jravan tables
`breeding_horses`, `produce` and `lineages` (`horsetrack japan jravan bloodlines`, #371).

`build_bloodlines` reads `<raw>/stored/BLDN/HN|SK|BT/all.jvd` through `records.iter_file`, keeps
one version per key (`records.latest_versions` on 繁殖登録番号 for HN and BT, 血統登録番号 for SK;
a key whose latest record is a deletion has no row), maps the kept records and writes
`<out>/breeding_horses.parquet`, `produce.parquet` and `lineages.parquet`, each sorted by its key
(plain string order) through `records.write_parquet`, so the same inputs give the same bytes.

Every text value is `records.text`. An id (a breeding id, a 血統登録番号, 生産者コード) is the
trimmed text as printed, null when blank or all zeros; a year is an integer and a date
`YYYYMMDD`, null when blank or all zeros. `sex` is 性別コード 1/2/3 as H/M/G; `colour` the first
content column of code table 2203 for 毛色コード, read from the workbook at run time (the meanings
stay in the licensed workbook), null for an unset, blank or unknown code. None of the tables is a
store table or has an availability rule (owner decision), so the views refuse them. The report
holds counts only. See docs/specs/jravan.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd
import pyarrow as pa
import typer

from horsetrack.core import paths
from horsetrack.countries.japan.jravan import records, spec

SOURCE = records.SOURCE
NAME = "bloodlines"
CREATED, KUBUN = "データ作成年月日", "データ区分"
BREEDING_ID, HORSE_ID = "繁殖登録番号", "血統登録番号"
PEDIGREE = "3代血統繁殖登録番号"
PEDIGREE_IDS = 14
COLOUR_TABLE = "2203"
SEXES = {"1": "H", "2": "M", "3": "G"}  # 性別コード (code table 2202)

BREEDING_HORSES_SCHEMA = pa.schema(
    [
        ("breeding_id", pa.string()),
        ("horse_id", pa.string()),
        ("name", pa.string()),
        ("name_en", pa.string()),
        ("birth_year", pa.int64()),
        ("sex", pa.string()),
        ("colour", pa.string()),
        ("import_type_code", pa.string()),
        ("import_year", pa.int64()),
        ("origin", pa.string()),
        ("sire_breeding_id", pa.string()),
        ("dam_breeding_id", pa.string()),
    ]
)
PRODUCE_SCHEMA = pa.schema(
    [
        ("horse_id", pa.string()),
        ("birth_date", pa.date32()),
        ("sex", pa.string()),
        ("colour", pa.string()),
        ("import_type_code", pa.string()),
        ("import_year", pa.int64()),
        ("breeder_code", pa.string()),
        ("origin", pa.string()),
        ("pedigree_breeding_ids", pa.list_(pa.string())),
    ]
)
LINEAGES_SCHEMA = pa.schema(
    [
        ("breeding_id", pa.string()),
        ("lineage_id", pa.string()),
        ("lineage_name", pa.string()),
        ("lineage_description", pa.string()),
    ]
)


# ------------------------------------------------------------------------------------------------
# Converters


def _unset(value: str | None) -> bool:
    return value is None or set(value) == {"0"}


def _ids(raw: np.ndarray) -> np.ndarray:
    """`records.text`, blank or all zeros None, in the input shape."""
    values = records.text(raw)
    flat = values.reshape(-1)
    flat[np.array([_unset(v) for v in flat], dtype=bool)] = None
    return values


def _years(raw: np.ndarray) -> pd.array:
    numbers = records.integer(raw)
    return pd.array([None if np.isnan(v) or v == 0 else int(v) for v in numbers], dtype="Int64")


def _date(value: str | None) -> date | None:
    if _unset(value) or len(value) != 8 or not value.isdigit():
        return None
    try:
        return date(int(value[:4]), int(value[4:6]), int(value[6:]))
    except ValueError:
        return None


def _sexes(raw: np.ndarray) -> list[str | None]:
    return [SEXES.get(v or "") for v in records.text(raw)]


def _colours(raw: np.ndarray) -> list[str | None]:
    """毛色コード -> its meaning in code table 2203; an unset, blank or unknown code None."""
    meanings = spec.code_table(COLOUR_TABLE).values
    return [None if _unset(v) else meanings.get(v) or None for v in records.text(raw)]


# ------------------------------------------------------------------------------------------------
# Records -> rows


def breeding_horses_from_records(cols: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """One `breeding_horses` row per HN record of `cols`, the schema's columns in order."""
    rows = {
        "breeding_id": _ids(cols[BREEDING_ID]),
        "horse_id": _ids(cols[HORSE_ID]),
        "name": records.text(cols["馬名"]),
        "name_en": records.text(cols["馬名欧字"]),
        "birth_year": _years(cols["生年"]),
        "sex": _sexes(cols["性別コード"]),
        "colour": _colours(cols["毛色コード"]),
        "import_type_code": records.text(cols["繁殖馬持込区分"]),
        "import_year": _years(cols["輸入年"]),
        "origin": records.text(cols["産地名"]),
        "sire_breeding_id": _ids(cols["父馬繁殖登録番号"]),
        "dam_breeding_id": _ids(cols["母馬繁殖登録番号"]),
    }
    return pd.DataFrame(rows, columns=BREEDING_HORSES_SCHEMA.names)


def produce_from_records(cols: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """One `produce` row per SK record of `cols`; `pedigree_breeding_ids` the 14 printed ids in
    record order, each null in place when blank or all zeros."""
    pedigree = _ids(cols[PEDIGREE])
    n = len(cols[KUBUN])
    if pedigree.shape != (n, PEDIGREE_IDS):
        raise ValueError(f"{PEDIGREE}: shape {pedigree.shape}, expected ({n}, {PEDIGREE_IDS})")
    rows = {
        "horse_id": _ids(cols[HORSE_ID]),
        "birth_date": [_date(v) for v in records.text(cols["生年月日"])],
        "sex": _sexes(cols["性別コード"]),
        "colour": _colours(cols["毛色コード"]),
        "import_type_code": records.text(cols["産駒持込区分"]),
        "import_year": _years(cols["輸入年"]),
        "breeder_code": _ids(cols["生産者コード"]),
        "origin": records.text(cols["産地名"]),
        "pedigree_breeding_ids": [list(ids) for ids in pedigree],
    }
    return pd.DataFrame(rows, columns=PRODUCE_SCHEMA.names)


def lineages_from_records(cols: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """One `lineages` row per BT record of `cols`, the schema's columns in order."""
    rows = {
        "breeding_id": _ids(cols[BREEDING_ID]),
        "lineage_id": records.text(cols["系統ID"]),
        "lineage_name": records.text(cols["系統名"]),
        "lineage_description": records.text(cols["系統説明"]),
    }
    return pd.DataFrame(rows, columns=LINEAGES_SCHEMA.names)


# ------------------------------------------------------------------------------------------------
# Build


def _read_latest(path: Path, record_type: str, key: str) -> tuple[dict[str, np.ndarray], dict]:
    """The kept records of one all.jvd (one version per `key`, deletions dropped), and counts."""
    chunks = list(records.iter_file(path, spec.record_spec(record_type)))
    cols = {name: np.concatenate([c[name] for c in chunks]) for name in chunks[0]} if chunks else {}
    del chunks
    if not cols:
        return cols, {"records": 0, "versions_dropped": 0, "deletions": 0}
    index = pd.DataFrame({name: records.text(cols[name]) for name in (key, CREATED, KUBUN)})
    keys = index.drop_duplicates([key]).shape[0]
    kept = records.latest_versions(index, [key]).index.to_numpy()
    counts = {
        "records": len(index),
        "versions_dropped": len(index) - keys,
        "deletions": keys - len(kept),
    }
    return {name: array[kept] for name, array in cols.items()}, counts


def _table(frame: pd.DataFrame, schema: pa.Schema, sort_by: str) -> pa.Table:
    frame = frame.sort_values(sort_by, kind="stable", na_position="last")
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    return table.replace_schema_metadata(None)


def build_bloodlines(*, raw: Path | None = None, out: Path | None = None) -> records.BuildReport:
    """Read the export's HN, SK and BT all.jvd (`raw` the export root, default
    `paths.raw_dir('jravan')`) and write the three tables in `out` (default
    `paths.processed_dir('jravan')`). Returns the counts, no record content."""
    root = (Path(raw) if raw is not None else paths.raw_dir(SOURCE)) / "stored" / "BLDN"
    out = Path(out) if out is not None else paths.processed_dir(SOURCE)
    counts: dict[str, int] = {}
    built = {}
    for table, record_type, key, mapper, schema, sort_by in (
        ("breeding_horses", "HN", BREEDING_ID, breeding_horses_from_records,
         BREEDING_HORSES_SCHEMA, "breeding_id"),
        ("produce", "SK", HORSE_ID, produce_from_records, PRODUCE_SCHEMA, "horse_id"),
        ("lineages", "BT", BREEDING_ID, lineages_from_records, LINEAGES_SCHEMA, "breeding_id"),
    ):  # fmt: skip
        path = root / record_type / "all.jvd"
        if not path.is_file():
            raise FileNotFoundError(f"no {record_type} file at {path}")
        cols, read = _read_latest(path, record_type, key)
        for name, value in read.items():
            counts[f"{record_type}_{name}"] = value
        frame = mapper(cols) if cols else pd.DataFrame(columns=schema.names)
        del cols
        built[table] = _table(frame, schema, sort_by)
        counts[table] = built[table].num_rows
        del frame

    breeding_ids = set(built["breeding_horses"]["breeding_id"].to_pylist())
    sires = [s for s in built["breeding_horses"]["sire_breeding_id"].to_pylist() if s is not None]
    pedigree = built["produce"]["pedigree_breeding_ids"].combine_chunks().flatten().to_pylist()
    pedigree = [i for i in pedigree if i is not None]
    lineage_ids = built["lineages"]["breeding_id"].to_pylist()
    counts["sire_ids"] = len(sires)
    counts["sire_ids_resolved"] = sum(s in breeding_ids for s in sires)
    counts["pedigree_ids"] = len(pedigree)
    counts["pedigree_ids_resolved"] = sum(i in breeding_ids for i in pedigree)
    counts["lineages_resolved"] = sum(b in breeding_ids for b in lineage_ids)

    for table, arrow in built.items():
        records.write_parquet(arrow, out / f"{table}.parquet")
    return records.BuildReport(name=NAME, counts=counts, by_year={}, unmapped={}, failures=[])


def bloodlines_command() -> None:
    """Parse the JRA-VAN export's HN, SK and BT records into processed/jravan/breeding_horses,
    produce and lineages; the counts in reports/jravan-bloodlines.json."""
    try:
        report = build_bloodlines()
        written = records.write_build_report(report)
    except (paths.DataDirNotSet, ValueError, LookupError, OSError) as exc:
        _die(f"jravan bloodlines failed: {type(exc).__name__}: {exc}")
    c = report.counts
    typer.echo(
        f"bloodlines: {c['breeding_horses']} breeding horses, {c['produce']} produce, "
        f"{c['lineages']} lineages; sires {c['sire_ids_resolved']}/{c['sire_ids']} resolved, "
        f"pedigree ids {c['pedigree_ids_resolved']}/{c['pedigree_ids']}, "
        f"lineages {c['lineages_resolved']}/{c['lineages']}"
    )
    typer.echo(f"report: {written}")


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)
