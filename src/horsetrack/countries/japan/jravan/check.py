"""`horsetrack japan jravan check`: the spec workbook describes every folder of the export.

For every record-type folder `raw/jravan/stored/<DATASPEC>/<RT>/`, plus the time series
`timeseries/0B41` (O1) and `timeseries/0B42` (O2) when present, the record type's layout must
validate and its length must equal the byte width, CRLF included, of the first record of the
folder's first `.jvd` file (sorted by path). One line per folder; exit 1 on any failure, a missing
workbook or a missing export. Only each first line is read, and no record content is printed.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import NoReturn

import typer

from horsetrack.core import paths
from horsetrack.countries.japan.jravan import spec

SOURCE = "jravan"
TIME_SERIES = {"0B41": "O1", "0B42": "O2"}


def export_folders(root: Path) -> list[tuple[str, str, Path]]:
    """(label, record type, folder) for every folder the check covers, in a stable order."""
    folders = [
        (f"stored/{ds.name}/{rt.name}", rt.name, rt)
        for ds in sorted((root / "stored").iterdir())
        if ds.is_dir()
        for rt in sorted(ds.iterdir())
        if rt.is_dir()
    ]
    for series, rt in TIME_SERIES.items():
        folder = root / "timeseries" / series
        if folder.is_dir():
            folders.append((f"timeseries/{series}", rt, folder))
    return folders


def check_folder(record_type: str, folder: Path, workbook: Path) -> tuple[bool, str]:
    """(ok, the line's text after the label) for one export folder."""
    files = sorted(folder.rglob("*.jvd"))
    if not files:
        return False, "no .jvd files"
    count = f"{len(files)} files"
    try:
        layout = spec.record_spec(record_type, workbook)
    except LookupError:
        return False, f"record type {record_type} is not in the workbook, {count}"
    except ValueError as exc:
        return False, f"invalid layout ({exc}), {count}"
    with files[0].open("rb") as fh:
        width = len(fh.readline())
    if width != layout.length:
        first = files[0].relative_to(folder)
        return False, f"layout {layout.length} bytes, {first} records {width} bytes, {count}"
    return True, f"ok, {count}"


def check_command() -> None:
    """Check that the JV-Data spec workbook describes every record type in the JRA-VAN export
    and that each layout's length matches its files' record width."""
    try:
        workbook = spec.spec_path()
        types = spec.record_types(workbook)
        root = paths.raw_dir(SOURCE)
    except (paths.DataDirNotSet, ValueError, OSError, zipfile.BadZipFile) as exc:
        _die(str(exc))
    if not (root / "stored").is_dir():
        _die(f"JRA-VAN export not found: {root / 'stored'}")
    typer.echo(f"JV-Data specification: {len(types)} record types")
    failed = 0
    folders = export_folders(root)
    for label, record_type, folder in folders:
        ok, text = check_folder(record_type, folder, workbook)
        failed += not ok
        typer.echo(f"{label}: {text}")
    typer.echo(f"{len(folders)} folders: {len(folders) - failed} ok, {failed} failed")
    if failed or not folders:
        raise typer.Exit(1)


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)
