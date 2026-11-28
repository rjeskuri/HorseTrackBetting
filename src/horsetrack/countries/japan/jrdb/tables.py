"""The JRDB store tables ``jrdb_pre_race`` and ``jrdb_post_race``: one row per runner matched to our
JRA store, and ``horsetrack japan jrdb build`` (ticket #57).

Ported from HKTrack's ``sources/jra/jrdb/tables.py`` without its backfill checks.

``jrdb_pre_race`` merges KYI, CYB, JOA and TYB; ``jrdb_post_race`` merges SED and SKB. A table's
families are merged on (race_key, horse_no) (outer: a runner in any one family is a row), joined to
the store's races and runners (``join.join_store``), and only rows whose race AND horse matched are
kept. The store here is the history races and runners plus every upcoming meeting folder
``processed/jra/upcoming/<meeting>/`` holding both ``races.parquet`` and
``racecard_runners.parquet``; a card race's missing ``venue`` and ``race_date`` come from its
``meeting_id`` (``with_meeting_keys``), and history wins a race_id over a card.
``jrdb_horse_id`` is the first family's (table order) that prints one. Columns: the key columns
the store declares (``schemas.TABLES[table].columns``, ``jurisdiction`` "jra" included), then
every other parsed column of family F as ``<f>_<column>`` (``kyi_idm``, ``tyb_paddock_index``).
A family that was not delivered still has its columns, all null; a ``date``-kind field is a
date32 column either way. Rows are stored in the table's sort order (race_id, horse_no).

``jrdb_pre_race`` ends with ``tyb_fetched_utc`` (ticket #603), a UTC timestamp: when the TYB
delivery behind the row was fetched, the ``fetched_utc`` that the downloader (HKTrack's, or
``jrdb fetch``) wrote for the delivery's archive in ``raw/jrdb/manifest.json`` (``{"files":
[{"file": "TYB/TYB261128.zip", "fetched_utc": "..."}]}``, ``file`` relative to raw/jrdb; a member
of a year pack takes its pack's time; a .txt with no entry of its own takes the entry of the .zip
of the same stem beside it, which ``jrdb fetch`` unpacked it from). It is null when there is no
manifest, no entry for the archive (nor, for a .txt, its sibling .zip), or the entry's time is
missing or unparseable, and for a runner TYB does not print; the file's mtime is never a
fallback. ``jrdb_post_race`` has no such column.

``run`` reads every archive under ``paths.raw_dir("jrdb") / <FAM>/`` and writes both tables to
``paths.processed_dir("jra")`` and ``paths.reports_dir() / "jrdb-build.json"`` (per table the
``JoinReport`` counts and rates plus ``table_rows``). Each delivery file (``<FAM>yymmdd``: a day
file, or a member of a year pack) is read once: from a day file before a year pack, and within
one kind from the first by name. A family whose folder holds only .lzh archives (no reader) stops
the build before anything is written; a .lzh beside a readable .zip or .txt is skipped.

The UKC horse master (ticket #209) is read the same way from ``raw/jrdb/UKC/``. When it holds
deliveries, ``pedigree.build_horse_pedigree`` of them is merged over the ``horse_pedigree`` already
in ``processed/jra`` (the netkeiba one; absent or unreadable counts as none) with
``pedigree.merge_pedigree`` and written back with ``PEDIGREE_SCHEMA``; the report gets a
``horse_pedigree`` entry (``pedigree.coverage`` plus ``ukc_rows``, ``netkeiba_rows``, ``path``).
No UKC delivery (or only .lzh ones): horse_pedigree is left alone and has no report entry.
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from horsetrack.core import paths
from horsetrack.core.store import schemas
from horsetrack.core.store import tables as store_tables
from horsetrack.countries.japan.jrdb import join, parse, pedigree

PRE_RACE, POST_RACE = "jrdb_pre_race", "jrdb_post_race"
# family order is merge order: the first family that prints a jrdb_horse_id supplies it
TABLES: dict[str, tuple[str, ...]] = {
    PRE_RACE: ("KYI", "CYB", "JOA", "TYB"),
    POST_RACE: ("SED", "SKB"),
}
REPORT = "jrdb-build.json"
MANIFEST = "manifest.json"  # the downloader's log under raw/jrdb
FETCHED = "tyb_fetched_utc"
UPCOMING_DIR = "upcoming"  # processed/jra/upcoming/<meeting_id with ':' as '_'>/
_RACE_COLUMNS = ("race_id", "race_date", "venue", "race_no", "kai", "nichi", "source_race_key")
_RUNNER_COLUMNS = ["race_id", "horse_no", "horse_id"]
MERGE_KEYS = ["race_key", "horse_no"]
_UNPREFIXED = ("race_key", "horse_no", "jrdb_horse_id")
_READABLE = (".zip", ".txt")
_DELIVERY = re.compile(r"[A-Z]+[0-9]{6}")  # a day file: KYI261128
_LETTERS = re.compile(r"[A-Z]*")
_KIND_TYPES = {"num": pa.float64(), "date": pa.date32(), "text": pa.string(), "code": pa.string()}
_BIRTH_YEARS = ("sire_birth_year", "dam_birth_year", "dam_sire_birth_year")
# The written horse_pedigree: HKTrack's (the copied store's) schema, whatever the data holds.
PEDIGREE_SCHEMA = pa.schema(
    [(c, pa.int64() if c in _BIRTH_YEARS else pa.string()) for c in pedigree.OUTPUT_COLUMNS]
)


def _family_columns(family: str) -> dict[str, pa.DataType]:
    """``<f>_<column>`` -> Arrow type, for every parsed column of ``family`` but the merge keys."""
    return {
        f"{family.lower()}_{f.name}": _KIND_TYPES[f.kind]
        for f in parse.spec(family)
        if f.kind != "skip" and f.name not in _UNPREFIXED
    }


def table_schema(table: str) -> pa.Schema:
    """The written file's schema: the store's key columns, then each family's columns."""
    keys = schemas.TABLES[table].columns
    fields = [(c, store_tables.ARROW_TYPES[kind]) for c, kind in keys.items()]
    for family in TABLES[table]:
        fields += list(_family_columns(family).items())
    if table == PRE_RACE:
        fields.append((FETCHED, pa.timestamp("us", tz="UTC")))
    return pa.schema(fields)


def _prefixed(family: str, frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["horse_no"] = pd.to_numeric(frame["horse_no"], errors="coerce").astype("float64")
    frame = frame.drop_duplicates(MERGE_KEYS, keep="last")  # a runner in two deliveries: the later
    if "jrdb_horse_id" not in frame.columns:
        frame["jrdb_horse_id"] = None
    rename = {c: f"{family.lower()}_{c}" for c in frame.columns if c not in _UNPREFIXED}
    return frame.rename(columns=rename).rename(columns={"jrdb_horse_id": f"_id_{family}"})


def merge_families(parsed: Mapping[str, pd.DataFrame], families: tuple[str, ...]) -> pd.DataFrame:
    """The families' rows merged on (race_key, horse_no); ``jrdb_horse_id`` from the first family
    that prints one."""
    merged: pd.DataFrame | None = None
    for family in families:
        rows = parsed[family] if family in parsed else parse.parse_bytes(family, b"")
        frame = _prefixed(family, rows)
        merged = frame if merged is None else merged.merge(frame, on=MERGE_KEYS, how="outer")
    assert merged is not None
    ids = [f"_id_{family}" for family in families]
    jrdb_id = merged[ids[0]].astype("object")
    for column in ids[1:]:
        jrdb_id = jrdb_id.where(jrdb_id.notna(), merged[column].astype("object"))
    merged["jrdb_horse_id"] = jrdb_id.where(jrdb_id.notna(), None)
    return merged.drop(columns=ids).reset_index(drop=True)


def build_table(
    table: str, parsed: Mapping[str, pd.DataFrame], races: pd.DataFrame, runners: pd.DataFrame
) -> tuple[pa.Table, join.JoinReport]:
    """``table`` from parsed family frames (any subset) and the store's races and runners."""
    merged = merge_families(parsed, TABLES[table])
    joined, report = join.join_store(merged, races, runners)
    kept = joined[joined["matched_by"].notna()].copy()
    kept["jurisdiction"] = join.SOURCE
    kept["horse_no"] = kept["horse_no"].round().astype("Int64")
    kept = kept.sort_values(["race_id", "horse_no"], kind="stable").reset_index(drop=True)
    schema = table_schema(table)
    frame = kept.reindex(columns=schema.names)
    if FETCHED in frame.columns:  # all null when TYB was not delivered or has no recorded time
        frame[FETCHED] = pd.to_datetime(frame[FETCHED], utc=True)
    return pa.Table.from_pandas(frame, schema=schema, preserve_index=False), report


# ------------------------------------------------------------------------------------------------
# horsetrack japan jrdb build
# ------------------------------------------------------------------------------------------------


@dataclass
class BuildOutcome:
    lines: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class Delivery:
    """One delivered file: a day file, or a member of a year pack."""

    family: str
    name: str  # the file's stem in upper case, e.g. KYI261128
    data: bytes
    archive: str  # the archive it was read from, relative to raw/jrdb


@dataclass
class Archives:
    deliveries: dict[str, list[Delivery]] = field(default_factory=dict)
    read: int = 0
    skipped: list[str] = field(default_factory=list)
    lzh_only: list[str] = field(default_factory=list)


def _rank(path: Path) -> tuple[int, str]:
    """Day files first, then year packs; within one kind by name."""
    return (0 if _DELIVERY.fullmatch(path.stem.upper()) else 1, path.name)


def _folders(families: list[str]) -> dict[str, list[Path]]:
    root = paths.raw_dir("jrdb")
    out = {}
    for family in families:
        folder = root / family
        out[family] = sorted(p for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []
    return out


def collect(families: list[str]) -> Archives:
    """Every delivery under raw/jrdb/<FAM>/, each once (see the module docstring). When any family
    holds only .lzh archives nothing is read: ``lzh_only`` names them."""
    folders = _folders(families)
    found = Archives()
    for family, files in folders.items():
        readable = [p for p in files if p.suffix.lower() in _READABLE]
        found.skipped += [p.name for p in files if p not in readable]
        if files and not readable:
            found.lzh_only.append(family)
    if found.lzh_only:
        return found

    for family, files in folders.items():
        chosen: dict[str, Delivery] = {}
        readable = [p for p in files if p.suffix.lower() in _READABLE]
        for path in sorted(readable, key=_rank):
            found.read += 1
            for member, data in parse.members(path):
                name = Path(member).stem.upper()
                if _LETTERS.match(name)[0] != family:  # type: ignore[index]
                    continue  # another family's file in a pack
                if name not in chosen:
                    chosen[name] = Delivery(family, name, data, f"{family}/{path.name}")
        if chosen:
            found.deliveries[family] = [chosen[name] for name in sorted(chosen)]
    return found


def fetch_times() -> dict[str, pd.Timestamp]:
    """Archive (relative to raw/jrdb) -> the UTC instant the downloader's manifest says it was
    fetched. An absent or unreadable manifest, or an entry with a missing or unparseable time,
    leaves the archive out."""
    try:
        files = json.loads((paths.raw_dir("jrdb") / MANIFEST).read_text(encoding="utf-8"))["files"]
        entries = [e for e in files if isinstance(e, dict)]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    out: dict[str, pd.Timestamp] = {}
    for entry in entries:
        try:
            when = pd.Timestamp(entry["fetched_utc"])
            name = entry["file"]
        except (KeyError, ValueError, TypeError):
            continue
        if isinstance(name, str) and not pd.isna(when):
            out[name] = when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")
    return out


def _fetched_at(archive: str, fetched: Mapping[str, pd.Timestamp]) -> pd.Timestamp:
    """``archive``'s instant in ``fetched``; a .txt with none of its own takes the .zip's of the
    same stem in its folder (``jrdb fetch`` unpacks a zip's members beside it), else NaT."""
    if archive in fetched:
        return fetched[archive]
    path = PurePosixPath(archive)
    if path.suffix.lower() == ".txt":
        return fetched.get(str(path.with_suffix(".zip")), pd.NaT)
    return pd.NaT


def parse_deliveries(
    deliveries: list[Delivery], fetched: Mapping[str, pd.Timestamp] | None = None
) -> pd.DataFrame:
    """The deliveries' records as one frame, in delivery-name order. With ``fetched`` (archive ->
    instant, see ``fetch_times``) each row also carries its archive's ``fetched_utc``
    (``_fetched_at``: an unpacked .txt takes its sibling .zip's), null for an archive not in it."""
    frames = []
    for d in deliveries:
        try:
            frame = parse.parse_bytes(d.family, d.data)
        except parse.RecordLengthError as exc:
            raise ValueError(f"{d.archive} {d.name}: {exc}") from exc
        if fetched is not None:
            when = _fetched_at(d.archive, fetched)
            frame["fetched_utc"] = pd.Series(when, index=frame.index, dtype="datetime64[us, UTC]")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def with_meeting_keys(races: pd.DataFrame) -> pd.DataFrame:
    """Fill ``venue`` and ``race_date`` from ``meeting_id``
    (``<jurisdiction>:<YYYY-MM-DD>:<venue>``) where a card race lacks them: the JRDB race key of a
    race without a ``source_race_key`` cannot be built without both."""
    need = [c for c in ("venue", "race_date") if c not in races.columns or races[c].isna().any()]
    if not need or "meeting_id" not in races.columns:
        return races
    parts = races["meeting_id"].astype("string").str.split(":", n=2, expand=True)
    if parts.shape[1] < 3:
        return races
    out = races.copy()
    derived = {
        "venue": parts[2].astype("object"),
        "race_date": pd.to_datetime(parts[1], errors="coerce").dt.date,
    }
    for column in need:
        value = derived[column]
        if column in out.columns:
            value = out[column].astype("object").where(out[column].notna(), value)
        out[column] = value
    return out


def _store() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The store's JRA races and runners (only the columns the join reads), plus every upcoming
    meeting's card races and runners; history wins a race_id, then the first folder by name."""
    race_columns = [c for c in _RACE_COLUMNS if c in schemas.TABLES["races"].columns]
    history = store_tables.read_table(join.SOURCE, "races", race_columns)
    races = [history]
    runners = [store_tables.read_table(join.SOURCE, "runners", _RUNNER_COLUMNS)]
    seen = set(history["race_id"].dropna())
    upcoming = paths.processed_dir(join.SOURCE) / UPCOMING_DIR
    folders = sorted(p for p in upcoming.iterdir() if p.is_dir()) if upcoming.is_dir() else []
    for folder in folders:
        card_races, card = folder / "races.parquet", folder / "racecard_runners.parquet"
        if not (card_races.is_file() and card.is_file()):
            continue
        frame = with_meeting_keys(pd.read_parquet(card_races))
        frame = frame[~frame["race_id"].isin(seen)]
        races.append(frame[[c for c in _RACE_COLUMNS if c in frame.columns]])
        entries = pd.read_parquet(card, columns=_RUNNER_COLUMNS)
        runners.append(entries[entries["race_id"].isin(set(frame["race_id"]))])
        seen |= set(frame["race_id"].dropna())
    return pd.concat(races, ignore_index=True), pd.concat(runners, ignore_index=True)


def _write_atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _rate(value: float) -> str:
    return "-" if value != value else f"{value:.1%}"


def _read_pedigree(path: Path) -> pd.DataFrame | None:
    """The horse_pedigree already on disk, or None when there is none (or it is unreadable)."""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError):
        return None


@dataclass
class PedigreeBuild:
    table: pa.Table
    ukc_rows: int
    path: Path


def _build_pedigree() -> PedigreeBuild | None:
    """UKC deliveries merged over the horse_pedigree on disk, or None without a UKC delivery."""
    ukc = collect([pedigree.UKC_FAMILY]).deliveries.get(pedigree.UKC_FAMILY)
    if not ukc:
        return None
    table = pedigree.build_horse_pedigree(parse_deliveries(ukc))
    path = store_tables.table_path(join.SOURCE, pedigree.HORSE_PEDIGREE)
    merged = pedigree.merge_pedigree(table, _read_pedigree(path))
    frame = merged.reindex(columns=PEDIGREE_SCHEMA.names)
    data = pa.Table.from_pandas(frame, schema=PEDIGREE_SCHEMA, preserve_index=False)
    return PedigreeBuild(data, len(table), path)


def run() -> BuildOutcome:
    """Parse every archive, build and write both tables and jrdb-build.json (module docstring)."""
    outcome = BuildOutcome()
    try:
        races, runners = _store()
    except FileNotFoundError as exc:
        outcome.error = f"no jra store: {exc}"
        return outcome
    archives = collect([f for families in TABLES.values() for f in families])
    if archives.lzh_only:
        outcome.error = (
            f"{', '.join(archives.lzh_only)}: only .lzh archives, which cannot be read; "
            "download the .zip versions"
        )
        return outcome
    fetched = fetch_times()
    parsed = {
        fam: parse_deliveries(d, fetched if fam == "TYB" else None)
        for fam, d in archives.deliveries.items()
    }
    counts = ", ".join(f"{k} {len(v)}" for k, v in parsed.items()) or "none"
    outcome.lines.append(
        f"jrdb build: {archives.read} archives, "
        f"{sum(len(d) for d in archives.deliveries.values())} delivery files, "
        f"{sum(len(f) for f in parsed.values())} records ({counts})"
    )
    if archives.skipped:
        outcome.lines.append(
            f"  skipped {len(archives.skipped)} files that are not .zip or .txt, "
            f"e.g. {archives.skipped[0]}"
        )

    built = {table: build_table(table, parsed, races, runners) for table in TABLES}
    ped = _build_pedigree()  # built before anything is written: a failure writes nothing
    summary: dict[str, Any] = {}
    for table, (data, report) in built.items():
        path = store_tables.table_path(join.SOURCE, table)
        _write_atomic(path, lambda tmp, data=data: pq.write_table(data, tmp))
        summary[table] = {**report.to_json(), "table_rows": data.num_rows, "path": str(path)}
        outcome.lines.append(
            f"  {table}: {data.num_rows} rows kept of {report.rows}; races not found "
            f"{_rate(report.race_unmatched_rate)}, horses not matched "
            f"{_rate(report.horse_unmatched_rate)} ({report.matched_by_horse_no} by number, "
            f"{len(report.mismatches)} id mismatches)"
        )
    if ped is not None:
        _write_atomic(ped.path, lambda tmp: pq.write_table(ped.table, tmp))
        cov = pedigree.coverage(ped.table.to_pandas())
        fallback = ped.table.num_rows - ped.ukc_rows
        summary[pedigree.HORSE_PEDIGREE] = {
            **cov,
            "ukc_rows": ped.ukc_rows,
            "netkeiba_rows": fallback,
            "path": str(ped.path),
        }
        outcome.lines.append(
            f"  horse_pedigree: {cov['rows']} rows, {cov['distinct_horse_ids']} horses, "
            f"{_rate(cov['sire_line_code_rate'])} with a sire-line code (UKC {ped.ukc_rows}, "
            f"netkeiba fallback {fallback}, paternal grandsire {cov['with_paternal_grandsire']})"
        )
    summary["archives"] = {"read": archives.read, "skipped": archives.skipped}
    detail = paths.reports_dir() / REPORT
    text = json.dumps(summary, indent=1, ensure_ascii=False, default=str) + "\n"
    _write_atomic(detail, lambda tmp: tmp.write_text(text, encoding="utf-8"))
    outcome.lines.append(f"  details {detail}")
    return outcome


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)


def build_command() -> None:
    """Parse the archives under raw/jrdb into processed/jra/jrdb_{pre,post}_race.parquet, joined to
    the jra store, and the UKC horse master into horse_pedigree.parquet; row counts and join rates
    in reports/jrdb-build.json."""
    try:
        outcome = run()
    except (ValueError, LookupError, OSError, zipfile.BadZipFile) as exc:
        _die(f"jrdb build failed: {type(exc).__name__}: {exc}")
    if outcome.error:
        _die(outcome.error)
    for line in outcome.lines:
        typer.echo(line)
