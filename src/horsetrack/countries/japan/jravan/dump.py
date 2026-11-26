"""`horsetrack japan jravan dump`: every JV-Data family no store table maps, decoded to Parquet.

Each `raw/jravan/stored/<DATASPEC>/<RT>/*.jvd` file of a folder outside `MAPPED` (or of a record
type `types` names, mapped or not) becomes `<out>/<DATASPEC>/<RT>/<stem>.parquet`: one row per
record, one column per `records.decode` key holding `records.text` (string, null when blank), a
repeated leaf `list<string>`, a nested repeat `list<list<string>>`, plus `_file` (the path relative
to the raw folder) and `_record` (1-based). It is written a row group per `records.iter_file`
chunk through a tmp file and `os.replace`, so a multi-GB file never sits in memory.

`<out>/_dump.json` records each file written: its size, mtime and the workbook's sha256. A file
that matches its entry and whose parquet exists is skipped. A file that fails to decode leaves no
parquet, loses its entry (the next run tries it again) and is a failure in the report; the other
files are still written. Dumps are not store tables and have no availability rule.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, NoReturn

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from horsetrack.core import paths
from horsetrack.core.eval.reports import LicensedOutputInRepo, write_report
from horsetrack.countries.japan.jravan import records, spec

SOURCE = records.SOURCE
NAME = "dump"
MANIFEST = "_dump.json"
MAPPED: frozenset[tuple[str, str]] = frozenset(
    {("RACE", rt) for rt in ("RA", "SE", "HR", "H1", "H6", "O1", "O2", "O3", "O4", "O5", "O6")}
    | {("DIFN", rt) for rt in ("UM", "KS", "CH")}
    | {("BLDN", rt) for rt in ("HN", "SK", "BT")}
)

_TEXT = pa.string()
_TYPES = {0: _TEXT, 1: pa.list_(_TEXT), 2: pa.list_(pa.list_(_TEXT))}


class DumpFailed(Exception):
    """A file that could not be dumped; `record` is the 1-based record number, if known."""

    def __init__(self, reason: str, record: int | None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.record = record


def run(
    *,
    types: list[str] | None = None,
    years: list[int] | None = None,
    raw: Path | None = None,
    out: Path | None = None,
) -> records.BuildReport:
    """Dump the selected stored folders (see the module docstring) and return the counts."""
    out = paths.outside_repo(
        Path(out) if out is not None else paths.interim_dir(SOURCE) / "records", what="out"
    )
    raw = Path(raw) if raw is not None else paths.raw_dir(SOURCE)
    workbook = spec.spec_path()
    digest = _sha256(workbook)
    if not (raw / "stored").is_dir():
        raise FileNotFoundError(f"JRA-VAN export not found: {raw / 'stored'}")

    manifest_path = out / MANIFEST
    done = _read_manifest(manifest_path)
    totals = {"files": 0, "records": 0, "bytes": 0, "skipped": 0, "failed": 0}
    per_folder: dict[str, dict[str, int | float]] = {}
    failures: list[dict] = []
    for ds, rt, folder in _folders(raw, types):
        started = time.monotonic()
        entry = {"files": 0, "records": 0, "bytes": 0}
        for path in _files(folder, years):
            rel = path.relative_to(raw).as_posix()
            target = out / ds / rt / f"{path.stem}.parquet"
            stat = path.stat()
            stamp = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "workbook": digest}
            if done.get(rel) == stamp and target.is_file():
                totals["skipped"] += 1
                continue
            try:
                n = _dump_file(path, rel, rt, workbook, target)
            except DumpFailed as exc:
                target.unlink(missing_ok=True)  # a stale dump of the file's earlier bytes
                failures.append({"file": rel, "record": exc.record, "reason": exc.reason})
                totals["failed"] += 1
                if done.pop(rel, None) is not None:
                    _write_manifest(manifest_path, done)
                continue
            done[rel] = stamp
            _write_manifest(manifest_path, done)
            for counts in (entry, totals):
                counts["files"] += 1
                counts["records"] += n
                counts["bytes"] += stat.st_size
        if entry["files"]:
            per_folder[f"{ds}/{rt}"] = {**entry, "seconds": round(time.monotonic() - started, 3)}
    return records.BuildReport(
        name=NAME, counts=totals, by_year={}, unmapped=per_folder, failures=failures
    )


def dump_command(
    record_types: Annotated[
        list[str] | None,
        typer.Option("--type", help="A record type to dump (repeatable), mapped or not."),
    ] = None,
    years: Annotated[
        list[int] | None,
        typer.Option(
            "--year", help="Only <YYYY>.jvd of this year (repeatable); all.jvd is always read."
        ),
    ] = None,
) -> None:
    """Decode every JV-Data family no store table maps (or the named types) to Parquet under
    interim/jravan/records, and write the counts to reports/jravan-dump.json."""
    try:
        report = run(types=record_types or None, years=years or None)
        target = records.write_build_report(report)
    except (
        paths.DataDirNotSet,
        paths.DataDirInsideRepo,
        LicensedOutputInRepo,
        OSError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        _die(str(exc))
    c = report.counts
    typer.echo(
        f"{c['files']} files dumped ({c['records']} records, {c['bytes']} bytes), "
        f"{c['skipped']} unchanged, {c['failed']} failed; report: {target}"
    )
    for failure in report.failures:
        where = f" record {failure['record']}" if failure["record"] is not None else ""
        typer.echo(f"FAILED {failure['file']}{where}: {failure['reason']}", err=True)
    if report.failures:
        raise typer.Exit(1)


# ------------------------------------------------------------------------------------------------
# Selection


def _folders(raw: Path, types: list[str] | None) -> Iterator[tuple[str, str, Path]]:
    for ds in sorted((raw / "stored").iterdir()):
        if not ds.is_dir():
            continue
        for rt in sorted(ds.iterdir()):
            if not rt.is_dir():
                continue
            wanted = (ds.name, rt.name) not in MAPPED if types is None else rt.name in types
            if wanted:
                yield ds.name, rt.name, rt


def _files(folder: Path, years: list[int] | None) -> list[Path]:
    files = sorted(p for p in folder.glob("*.jvd") if p.is_file())
    if years is None:
        return files
    keep = {f"{y:04d}" for y in years}
    return [p for p in files if p.stem == "all" or p.stem in keep]


# ------------------------------------------------------------------------------------------------
# One file


def _dump_file(path: Path, rel: str, rt: str, workbook: Path, target: Path) -> int:
    """Write `target` from `path` through a tmp file; the records written. Raises DumpFailed."""
    try:
        layout = spec.record_spec(rt, workbook)
    except LookupError:
        raise DumpFailed("record type not in the workbook", None) from None
    except ValueError:
        raise DumpFailed("invalid layout", None) from None
    axes = records.leaf_axes(layout)
    schema = pa.schema(
        [pa.field(key, _TYPES[n]) for key, n in axes.items()]
        + [pa.field("_file", _TEXT), pa.field("_record", pa.int64())]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:12]}.tmp")
    written = 0
    try:
        with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
            try:
                for chunk in records.iter_file(path, layout):
                    n = len(next(iter(chunk.values()))) if chunk else 0
                    columns = [_column(chunk[key]) for key in axes]
                    columns.append(pa.array(np.full(n, rel, dtype=object), type=_TEXT))
                    columns.append(pa.array(np.arange(written + 1, written + n + 1)))
                    writer.write_table(pa.Table.from_arrays(columns, schema=schema))
                    written += n
            except records.RecordLengthError as exc:
                raise DumpFailed("record length", exc.record) from None
            except records.RecordTypeError as exc:
                raise DumpFailed("record type", exc.record) from None
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return written


def _column(raw: np.ndarray) -> pa.Array:
    """`raw` ((n,), (n, r) or (n, r, s) bytes) as text: string, list<string>, list<list<string>>."""
    flat = pa.array(records.text(raw).reshape(-1), type=_TEXT)
    for width in reversed(raw.shape[1:]):
        offsets = pa.array(np.arange(0, len(flat) + 1, width, dtype=np.int32))
        flat = pa.ListArray.from_arrays(offsets, flat)
    return flat


# ------------------------------------------------------------------------------------------------
# The manifest


def _read_manifest(path: Path) -> dict[str, dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}
    files = data.get("files") if isinstance(data, dict) else None
    return dict(files) if isinstance(files, dict) else {}


def _write_manifest(path: Path, done: dict[str, dict]) -> None:
    text = json.dumps({"files": done}, sort_keys=True, indent=1) + "\n"
    write_report(path, text, sources=(SOURCE,))


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"JV-Data specification workbook not found: {path} (the JV-Data specification is a "
            "precondition: place JV-Data4901.xlsx there)"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)
