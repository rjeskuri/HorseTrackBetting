"""JV-Data UM (horse), KS (jockey) and CH (trainer) masters -> the jravan store's `horses` and
`horse_pedigree` and the jravan-only `jockeys` and `trainers` tables (#370).

`horses_from_records` and `pedigree_from_records` map `records.decode`d UM arrays, and
`jockeys_from_records` and `trainers_from_records` KS and CH arrays, to rows. `build_masters` reads
`raw/jravan/stored/DIFN/{UM,KS,CH}/all.jvd` (only the items it maps, chunk by chunk), keeps one
version per key (`records.latest_versions` on the layout's key item; a deletion has no row) and
writes `processed/jravan/{horses,horse_pedigree,jockeys,trainers}.parquet`, sorted by key, the same
inputs giving the same bytes. The colour and horse-mark meanings are read from the workbook's code
tables at run time (`spec.code_table`), so the repository holds no code-table text; sex and
affiliation are code constants below. The masters are the export's current snapshot: retirement,
affiliation and licences are not values at t. See docs/specs/jravan.md.
"""

from __future__ import annotations

import collections
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd
import pyarrow as pa
import typer

from horsetrack.core import paths
from horsetrack.core.store.build import arrow_schema
from horsetrack.core.store.schemas import TABLES
from horsetrack.countries.japan.jra.netkeiba.parse_pedigree_card import _norm
from horsetrack.countries.japan.jravan import records, runners, spec

SOURCE = records.SOURCE
JURISDICTION = "jra"
KUBUN, CREATED = "データ区分", "データ作成年月日"
MASTERS_DIR = ("stored", "DIFN")
FILE_NAME = "all.jvd"

# UM items (the `records.decode` keys)
HORSE_ID = "血統登録番号"
HORSE_RETIRED = "競走馬抹消区分"
HORSE_NAME = "馬名"
ORIGIN = "産地名"
COLOUR = "毛色コード"
SEX = "性別コード"
MARK = "馬記号コード"
PEDIGREE = "3代血統情報"
ANCESTOR_ID, ANCESTOR_NAME = f"{PEDIGREE}/繁殖登録番号", f"{PEDIGREE}/馬名"
# the occurrences of 3代血統情報 (父, 母, 父父, 父母, 母父, ...): sire, dam, dam_sire
ANCESTORS = {"sire": 0, "dam": 1, "dam_sire": 4}
UM_ITEMS = (KUBUN, CREATED, HORSE_ID, HORSE_RETIRED, HORSE_NAME, ORIGIN, COLOUR, SEX, MARK,
            ANCESTOR_ID, ANCESTOR_NAME)  # fmt: skip
COLOUR_TABLE, MARK_TABLE = "2203", "2204"

# KS items
JOCKEY_CODE = "騎手コード"
JOCKEY_ITEMS = {
    "retired": "騎手抹消区分", "issued": "騎手免許交付年月日", "revoked": "騎手免許抹消年月日",
    "born": "生年月日", "name": "騎手名", "name_short": "騎手名略称",
    "affiliation": "騎手東西所属コード", "stable": "所属調教師コード",
    "apprentice": "騎手見習コード",
}  # fmt: skip
KS_ITEMS = (KUBUN, CREATED, JOCKEY_CODE, *JOCKEY_ITEMS.values())

# CH items
TRAINER_CODE = "調教師コード"
TRAINER_ITEMS = {
    "retired": "調教師抹消区分", "issued": "調教師免許交付年月日",
    "revoked": "調教師免許抹消年月日", "born": "生年月日", "name": "調教師名",
    "name_short": "調教師名略称", "affiliation": "調教師東西所属コード",
}  # fmt: skip
CH_ITEMS = (KUBUN, CREATED, TRAINER_CODE, *TRAINER_ITEMS.values())

SEXES = runners.SEXES
AFFILIATIONS = {"1": "miho", "2": "ritto", "3": "nar", "4": "overseas"}
RETIRED, ACTIVE = "1", "0"

JOCKEYS_SCHEMA = pa.schema([
    ("jurisdiction", pa.string()), ("jockey_code", pa.string()), ("jockey_key", pa.string()),
    ("name", pa.string()), ("name_short", pa.string()), ("birth_date", pa.date32()),
    ("licence_issued", pa.date32()), ("licence_revoked", pa.date32()),
    ("is_retired", pa.bool_()), ("affiliation", pa.string()), ("trainer_code", pa.string()),
    ("apprentice_code", pa.string()),
])  # fmt: skip
TRAINERS_SCHEMA = pa.schema([
    ("jurisdiction", pa.string()), ("trainer_code", pa.string()), ("trainer_key", pa.string()),
    ("name", pa.string()), ("name_short", pa.string()), ("birth_date", pa.date32()),
    ("licence_issued", pa.date32()), ("licence_revoked", pa.date32()),
    ("is_retired", pa.bool_()), ("affiliation", pa.string()),
])  # fmt: skip

# the fields whose unmapped codes the report counts (an unset code, blank or all zeros, is null
# and never counted)
SEX_FIELD, COLOUR_FIELD, MARK_FIELD = "sex", "colour", "mark"
RETIRED_FIELD, AFFILIATION_FIELD, DATE_FIELD = "retired", "affiliation", "date"
INVALID_DATE = "invalid"  # a date is counted under this code, never by its value


def _unset(code: str | None) -> bool:
    return code is None or set(code) == {"0"}


def _coded(
    codes: np.ndarray, meanings: Mapping[str, str], counter: collections.Counter
) -> list[str | None]:
    """Each code's meaning; an unset code is null, a code `meanings` lacks null and counted."""
    out = []
    for code in codes:
        meaning = None if _unset(code) else meanings.get(code) or None
        if meaning is None and not _unset(code):
            counter[code] += 1
        out.append(meaning)
    return out


def _flags(codes: np.ndarray, counter: collections.Counter) -> list[bool | None]:
    """`1` true, `0` false, anything else null and counted."""
    out: list[bool | None] = []
    for code in codes:
        if code == RETIRED:
            out.append(True)
        elif code == ACTIVE:
            out.append(False)
        else:
            counter[code or ""] += 1
            out.append(None)
    return out


def _dates(codes: np.ndarray, counter: collections.Counter) -> list[date | None]:
    """`YYYYMMDD` as dates; blank or all zeros null, an impossible date null and counted."""
    out: list[date | None] = []
    for code in codes:
        if _unset(code):
            out.append(None)
            continue
        try:
            out.append(date(int(code[:4]), int(code[4:6]), int(code[6:8])))
        except ValueError:
            counter[INVALID_DATE] += 1
            out.append(None)
    return out


def _ids(codes: np.ndarray) -> list[str | None]:
    return [None if _unset(code) else code for code in codes]


def _counters(*fields: str) -> dict[str, collections.Counter]:
    return {field: collections.Counter() for field in fields}


# ------------------------------------------------------------------------------------------------
# UM


def _ancestors(cols: Mapping[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per ancestor (sire, dam, dam_sire): its breeding ids and names as text."""
    ids, names = np.asarray(cols[ANCESTOR_ID]), np.asarray(cols[ANCESTOR_NAME])
    return {
        who: (records.text(ids[:, i]), records.text(names[:, i])) for who, i in ANCESTORS.items()
    }


def _horses(cols: Mapping[str, np.ndarray], unmapped: dict[str, collections.Counter]):
    text = {name: records.text(cols[name]) for name in (HORSE_ID, HORSE_RETIRED, HORSE_NAME,
                                                       ORIGIN, COLOUR, SEX, MARK)}  # fmt: skip
    colours = spec.code_table(COLOUR_TABLE).values
    marks = spec.code_table(MARK_TABLE).values
    ancestors = _ancestors(cols)
    n = len(text[HORSE_ID])
    rows = {
        "jurisdiction": [JURISDICTION] * n,
        "horse_id": text[HORSE_ID],
        "name": text[HORSE_NAME],
        "origin": text[ORIGIN],
        "colour": _coded(text[COLOUR], colours, unmapped[COLOUR_FIELD]),
        "sex": _coded(text[SEX], SEXES, unmapped[SEX_FIELD]),
        "import_type": _coded(text[MARK], marks, unmapped[MARK_FIELD]),
        "sire": ancestors["sire"][1],
        "dam": ancestors["dam"][1],
        "dam_sire": ancestors["dam_sire"][1],
        "is_retired": _flags(text[HORSE_RETIRED], unmapped[RETIRED_FIELD]),
        "profile_scraped_utc": [None] * n,
    }
    return pd.DataFrame(rows, columns=list(TABLES["horses"].columns))


def horses_from_records(cols: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """One `horses` row per UM record of `cols`, in `schemas.TABLES['horses']` column order."""
    return _horses(cols, _counters(COLOUR_FIELD, SEX_FIELD, MARK_FIELD, RETIRED_FIELD))


def pedigree_from_records(cols: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """One `horse_pedigree` row per UM record of `cols`: the sire, dam and dam sire's names, their
    normal forms and breeding ids (null when blank or all zeros)."""
    n = len(cols[HORSE_ID])
    rows: dict[str, list] = {
        "jurisdiction": [JURISDICTION] * n,
        "horse_id": list(records.text(cols[HORSE_ID])),
    }
    for who, (ids, names) in _ancestors(cols).items():
        rows[f"{who}_name"] = list(names)
        rows[f"{who}_name_norm"] = [_norm(name) for name in names]
        rows[f"{who}_id"] = _ids(ids)
    rows["source_race_id"] = [None] * n
    rows["scraped_utc"] = [None] * n
    return pd.DataFrame(rows, columns=list(TABLES["horse_pedigree"].columns))


# ------------------------------------------------------------------------------------------------
# KS and CH


def _people(
    cols: Mapping[str, np.ndarray],
    code_item: str,
    items: Mapping[str, str],
    role: str,
    unmapped: dict[str, collections.Counter],
) -> dict[str, list]:
    """The columns jockeys and trainers share, `role` naming the code and key columns."""
    codes = records.text(cols[code_item])
    text = {alias: records.text(cols[item]) for alias, item in items.items()}
    return {
        "jurisdiction": [JURISDICTION] * len(codes),
        f"{role}_code": list(codes),
        f"{role}_key": [f"{role}:{code}" if code else None for code in codes],
        "name": list(text["name"]),
        "name_short": list(text["name_short"]),
        "birth_date": _dates(text["born"], unmapped[DATE_FIELD]),
        "licence_issued": _dates(text["issued"], unmapped[DATE_FIELD]),
        "licence_revoked": _dates(text["revoked"], unmapped[DATE_FIELD]),
        "is_retired": _flags(text["retired"], unmapped[RETIRED_FIELD]),
        "affiliation": _coded(text["affiliation"], AFFILIATIONS, unmapped[AFFILIATION_FIELD]),
    }


def _jockeys(cols: Mapping[str, np.ndarray], unmapped: dict[str, collections.Counter]):
    rows = _people(cols, JOCKEY_CODE, JOCKEY_ITEMS, "jockey", unmapped)
    rows["trainer_code"] = _ids(records.text(cols[JOCKEY_ITEMS["stable"]]))
    rows["apprentice_code"] = list(records.text(cols[JOCKEY_ITEMS["apprentice"]]))
    return pd.DataFrame(rows, columns=JOCKEYS_SCHEMA.names)


def _trainers(cols: Mapping[str, np.ndarray], unmapped: dict[str, collections.Counter]):
    rows = _people(cols, TRAINER_CODE, TRAINER_ITEMS, "trainer", unmapped)
    return pd.DataFrame(rows, columns=TRAINERS_SCHEMA.names)


def _people_counters() -> dict[str, collections.Counter]:
    return _counters(DATE_FIELD, RETIRED_FIELD, AFFILIATION_FIELD)


def jockeys_from_records(cols: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """One `jockeys` row per KS record of `cols`, with the columns of `JOCKEYS_SCHEMA`."""
    return _jockeys(cols, _people_counters())


def trainers_from_records(cols: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """One `trainers` row per CH record of `cols`, with the columns of `TRAINERS_SCHEMA`."""
    return _trainers(cols, _people_counters())


# ------------------------------------------------------------------------------------------------
# The build


def _read_latest(
    path: Path, record_type: str, items: tuple[str, ...], counts: dict[str, int]
) -> dict[str, np.ndarray]:
    """The `items` of `path`'s records, one version per key (a deletion has none), in file order;
    3代血統情報 is cut to the occurrences up to the dam sire's. `counts` gets the record, dropped
    version and deletion counts."""
    layout = spec.record_spec(record_type)
    key = [item.name for item in layout.items if item.is_key]
    names = tuple(dict.fromkeys((*key, *items)))
    upto = max(ANCESTORS.values()) + 1
    chunks = []
    for part in records.iter_file(path, layout):
        chunk = {name: part[name] for name in names}
        for name in (ANCESTOR_ID, ANCESTOR_NAME):
            if name in chunk:
                chunk[name] = np.ascontiguousarray(chunk[name][:, :upto])
        chunks.append(chunk)
    if not chunks:
        counts.update(dict.fromkeys(("records", "versions_dropped", "deletions"), 0))
        layout_cols = records.decode(b"", layout)
        return {name: layout_cols[name] for name in names}
    cols = {name: np.concatenate([c[name] for c in chunks]) for name in names}
    del chunks
    index = pd.DataFrame({name: records.text(cols[name]) for name in (*key, CREATED, KUBUN)})
    kept = records.latest_versions(index, key)
    distinct = index.drop_duplicates(key).shape[0]
    counts["records"] = len(index)
    counts["versions_dropped"] = len(index) - distinct
    counts["deletions"] = distinct - len(kept)
    rows = np.flatnonzero(index.index.isin(kept.index))
    return {name: array[rows] for name, array in cols.items()}


def _write(frame: pd.DataFrame, schema: pa.Schema, sort_by: str, path: Path) -> int:
    frame = frame.sort_values(sort_by, kind="stable")
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    records.write_parquet(table.replace_schema_metadata(None), path)
    return len(frame)


def build_masters(*, raw: Path | None = None, out: Path | None = None) -> records.BuildReport:
    """Build `horses`, `horse_pedigree`, `jockeys` and `trainers` in `out` (default
    `processed/jravan`) from the UM, KS and CH masters under the export root `raw` (default
    `raw/jravan`). The report holds counts and codes only."""
    raw = Path(raw) if raw is not None else paths.raw_dir(SOURCE)
    out = Path(out) if out is not None else paths.processed_dir(SOURCE)
    folder = raw.joinpath(*MASTERS_DIR)
    counts: dict[str, int] = {}
    unmapped = _counters(COLOUR_FIELD, SEX_FIELD, MARK_FIELD, RETIRED_FIELD, AFFILIATION_FIELD,
                         DATE_FIELD)  # fmt: skip

    def read(record_type: str, items: tuple[str, ...]) -> dict[str, np.ndarray]:
        found: dict[str, int] = {}
        cols = _read_latest(folder / record_type / FILE_NAME, record_type, items, found)
        counts.update({f"{record_type}_{k}": v for k, v in found.items()})
        return cols

    cols = read("UM", UM_ITEMS)
    horses = _horses(cols, unmapped)
    pedigree = pedigree_from_records(cols)
    del cols
    counts["horses_written"] = _write(
        horses, arrow_schema("horses"), "horse_id", out / "horses.parquet"
    )
    counts["horse_pedigree_written"] = _write(
        pedigree, arrow_schema("horse_pedigree"), "horse_id", out / "horse_pedigree.parquet"
    )
    del horses, pedigree
    jockeys = _jockeys(read("KS", KS_ITEMS), unmapped)
    counts["jockeys_written"] = _write(
        jockeys, JOCKEYS_SCHEMA, "jockey_code", out / "jockeys.parquet"
    )
    trainers = _trainers(read("CH", CH_ITEMS), unmapped)
    counts["trainers_written"] = _write(
        trainers, TRAINERS_SCHEMA, "trainer_code", out / "trainers.parquet"
    )
    return records.BuildReport(
        name="masters",
        counts=counts,
        by_year={},
        unmapped={field: dict(sorted(c.items())) for field, c in unmapped.items()},
        failures=[],
    )


def masters_command() -> None:
    """Build the jravan store's horses, horse_pedigree, jockeys and trainers from the JV-Data UM,
    KS and CH masters; the report (counts and codes only) goes to
    <reports dir>/jravan-masters.json."""
    try:
        report = build_masters()
        target = records.write_build_report(report)
    except (paths.DataDirNotSet, ValueError, LookupError, OSError) as exc:
        _die(f"jravan masters failed: {type(exc).__name__}: {exc}")
    c = report.counts
    for record_type, table in (("UM", "horses"), ("KS", "jockeys"), ("CH", "trainers")):
        typer.echo(
            f"{record_type}: {c[f'{record_type}_records']} records, "
            f"{c[f'{record_type}_versions_dropped']} versions dropped, "
            f"{c[f'{record_type}_deletions']} deletions, {c[f'{table}_written']} {table} written"
        )
    for field, codes in report.unmapped.items():
        if codes:
            listed = ", ".join(f"{code} x{n}" for code, n in codes.items())
            typer.echo(f"unmapped {field}: {listed}")
    typer.echo(f"report: {target}")


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)
