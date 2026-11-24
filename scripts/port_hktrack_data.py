"""Copy HKTrack's data and real fixtures into the data directory (ticket #18).

    uv run python3 scripts/port_hktrack_data.py --from <HKTrack root> --to <data dir> [--dry-run]

Ports `data/{raw,interim,processed,odds_snapshots}` to the same names under the destination and
`tests/fixtures/{hkjc,jra,jrdb,web}` to `<dst>/fixtures/`. Skips every other data directory
(`data/features` among them), anything under `__pycache__`, `*.lock`, `*.tmp`, `.DS_Store` and the
-wal, -shm and -journal files of a `*.sqlite` database. A database is copied with the sqlite3
backup API from a read-only connection, so rows still in its -wal reach the copy and HKTrack's
files are never checkpointed, created or deleted.

Re-runs copy only what is new: a missing file is copied; a destination that is a byte prefix of
the source (an append-only log that grew) is extended; an equal one is left alone; any other
differing destination, a database whose content differs included, is a conflict, reported on every
run and left alone. Nothing in the destination is deleted or overwritten, and no partial file is
left at a final path. A destination inside a git work tree, or equal to, inside or containing the
source, is refused before anything is written. Every real run (re)writes `<dst>/PORTED.json`.

Exit status: 0 on a clean run, 1 when a conflict is reported, 2 when the destination is refused.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

DATA_DIRS = ("raw", "interim", "processed", "odds_snapshots")
FIXTURE_DIRS = ("hkjc", "jra", "jrdb", "web")
TOPS = (*DATA_DIRS, "fixtures")
SKIP_DIRS = frozenset({"__pycache__"})
SKIP_NAMES = frozenset({".DS_Store"})
SKIP_SUFFIXES = (".lock", ".tmp")
SQLITE = ".sqlite"
SIDECARS = ("-wal", "-shm", "-journal")
CHUNK = 1 << 20

# outcomes, named after the Report lists they go to
COPIED, EXTENDED, UNCHANGED, CONFLICT = "copied", "extended", "unchanged", "conflicts"
_GAP = object()  # pads the shorter of two database dumps: equal to nothing


class Refused(Exception):
    """The destination would write into a git work tree or into the source."""


@dataclass(frozen=True)
class Step:
    src: Path  # absolute
    dst: Path  # absolute
    rel: str  # dst relative to the destination root, posix
    top: str  # one of TOPS
    sqlite: bool


@dataclass
class Report:
    copied: list[str] = field(default_factory=list)
    extended: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    dirs: dict[str, dict[str, int]] = field(
        default_factory=lambda: {top: {"files": 0, "bytes": 0} for top in TOPS}
    )


# --- plan -------------------------------------------------------------------------------------


def plan(src_data: Path, src_fixtures: Path, dst: Path) -> list[Step]:
    """Every source file to port, in order; raises Refused for a destination it must not write."""
    src_data = Path(src_data).expanduser().resolve()
    src_fixtures = Path(src_fixtures).expanduser().resolve()
    dst = _checked_destination(Path(dst), (src_data, src_fixtures))
    if not src_data.is_dir():
        raise Refused(f"no data directory at {src_data}")
    steps: list[Step] = []
    for top in DATA_DIRS:
        steps += _walk(src_data / top, dst, top, top)
    for sub in FIXTURE_DIRS:
        steps += _walk(src_fixtures / sub, dst, f"fixtures/{sub}", "fixtures")
    return steps


def _checked_destination(dst: Path, sources: tuple[Path, ...]) -> Path:
    """dst resolved, after refusing a git work tree or an overlap with a source (gitignored too)."""
    dst = dst.expanduser().resolve()
    for src in sources:
        if dst.is_relative_to(src) or src.is_relative_to(dst):
            raise Refused(f"the destination {dst} overlaps the source {src}")
    for ancestor in (dst, *dst.parents):
        if os.path.lexists(ancestor / ".git"):
            raise Refused(f"the destination {dst} is inside the git work tree at {ancestor}")
    return dst


def _walk(src_root: Path, dst_root: Path, prefix: str, top: str) -> list[Step]:
    steps = []
    for base, dirs, names in os.walk(src_root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        here = Path(base)
        for name in sorted(names):
            path = here / name
            if _skipped(name) or not path.is_file():
                continue
            rel = f"{prefix}/{path.relative_to(src_root).as_posix()}"
            steps.append(Step(path, dst_root / rel, rel, top, name.endswith(SQLITE)))
    return steps


def _skipped(name: str) -> bool:
    if name in SKIP_NAMES or name.endswith(SKIP_SUFFIXES):
        return True
    return any(name.endswith(SQLITE + sidecar) for sidecar in SIDECARS)


# --- run --------------------------------------------------------------------------------------


def run(steps: list[Step], dry_run: bool = False) -> Report:
    """Port each step; a dry run writes nothing and returns the report the real run would."""
    report = Report()
    for step in steps:
        size = os.stat(step.src).st_size
        port = _port_sqlite if step.sqlite else _port_file
        getattr(report, port(step.src, step.dst, dry_run)).append(step.rel)
        counts = report.dirs[step.top]
        counts["files"] += 1
        counts["bytes"] += size
    return report


def _port_file(src: Path, dst: Path, dry_run: bool) -> str:
    if not os.path.lexists(dst) and (dry_run or _copy_new(src, dst)):
        return COPIED
    if not _regular(dst):
        return CONFLICT
    size, length = os.stat(src).st_size, os.lstat(dst).st_size
    if length > size or not _same_prefix(src, dst, length):
        return CONFLICT
    if length == size:
        return UNCHANGED
    return EXTENDED if dry_run or _extend(src, dst, length) else CONFLICT


def _port_sqlite(src: Path, dst: Path, dry_run: bool) -> str:
    if not os.path.lexists(dst) and (dry_run or _backup_new(src, dst)):
        return COPIED
    if not _regular(dst):
        return CONFLICT
    return UNCHANGED if _same_database(src, dst) else CONFLICT


def _regular(path: Path) -> bool:
    return stat.S_ISREG(os.lstat(path).st_mode)


def _same_prefix(src: Path, dst: Path, length: int) -> bool:
    """Whether the first `length` bytes of src and dst are equal, read in chunks."""
    with open(src, "rb") as a, open(dst, "rb") as b:
        left = length
        while left:
            n = min(CHUNK, left)
            chunk = a.read(n)
            if len(chunk) != n or chunk != b.read(n):
                return False
            left -= n
    return True


def _copy_new(src: Path, dst: Path) -> bool:
    """Copy src to a missing dst via a temp file; False when dst appeared meanwhile."""
    st = os.stat(src)
    with _temp_beside(dst) as tmp:
        with open(src, "rb") as inp, open(tmp, "wb") as out:
            shutil.copyfileobj(inp, out, CHUNK)
        os.chmod(tmp, stat.S_IMODE(st.st_mode))
        os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
        return _place(tmp, dst)


def _extend(src: Path, dst: Path, length: int) -> bool:
    """Append src[length:] to dst, after re-checking dst is still src's first `length` bytes."""
    st = os.stat(src)
    with open(dst, "ab") as out:
        if os.fstat(out.fileno()).st_size != length or not _same_prefix(src, dst, length):
            return False
        with open(src, "rb") as inp:
            inp.seek(length)
            shutil.copyfileobj(inp, out, CHUNK)
    os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))
    return True


def _backup_new(src: Path, dst: Path) -> bool:
    """Back the database up into a temp file, then move it to a missing dst."""
    st = os.stat(src)
    with _temp_beside(dst) as tmp:
        with _reader(src) as source:
            target = sqlite3.connect(tmp)
            try:
                source.backup(target)
            finally:
                target.close()
        os.chmod(tmp, stat.S_IMODE(st.st_mode))
        return _place(tmp, dst)


@contextmanager
def _temp_beside(dst: Path) -> Iterator[Path]:
    """A fresh temp file in dst's directory, removed (with any sqlite sidecar) afterwards."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
    finally:
        for path in (tmp, *(Path(f"{tmp}{sidecar}") for sidecar in SIDECARS)):
            path.unlink(missing_ok=True)


def _place(tmp: Path, dst: Path) -> bool:
    """Give the finished temp file dst's name, never replacing a file that appeared there."""
    try:
        os.link(tmp, dst)
    except FileExistsError:
        return False
    return True


# --- sqlite -----------------------------------------------------------------------------------


@contextmanager
def _reader(path: Path) -> Iterator[sqlite3.Connection]:
    """A read-only connection that creates, checkpoints and deletes no file next to path.

    With no rows in a -wal the database file holds every committed row, so it is read as
    immutable (no -wal or -shm is created) and must not change while it is read. With rows in the
    -wal, a plain read-only connection reads them and leaves the -wal as it was.
    """
    uri = f"{path.as_uri()}?mode=ro"
    with open(path, "rb") as f:
        header = f.read(20)
    wal, shm = Path(f"{path}-wal"), Path(f"{path}-shm")
    guarded = False
    if header[:16] == b"SQLite format 3\x00" and 2 in header[18:20]:  # WAL mode
        if not wal.exists() or wal.stat().st_size == 0:
            uri += "&immutable=1"
            guarded = True
        elif not shm.exists():
            raise RuntimeError(f"{path} has a -wal and no -shm: reading it would create one")
    before = _identity(path)
    con = sqlite3.connect(uri, uri=True)
    try:
        yield con
    finally:
        con.close()
    if guarded and _identity(path) != before:
        raise RuntimeError(f"{path} changed while it was read; run again")


def _identity(path: Path) -> tuple[int, int, int]:
    st = os.stat(path)
    return st.st_ino, st.st_size, st.st_mtime_ns


def _same_database(src: Path, dst: Path) -> bool:
    """Whether the two databases hold the same schema, user_version and rows."""
    with _reader(src) as a, _reader(dst) as b:
        pairs = zip_longest(_dump(a), _dump(b, tolerant=True), fillvalue=_GAP)
        return all(x == y for x, y in pairs)


def _dump(con: sqlite3.Connection, tolerant: bool = False) -> Iterator[object]:
    """A deterministic stream of the database's content; unreadable yields _GAP when tolerant."""
    try:
        yield con.execute("PRAGMA user_version").fetchone()
        schema = con.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        yield from schema
        for kind, name, _, sql in schema:
            if kind != "table" or (sql or "").lstrip().upper().startswith("CREATE VIRTUAL"):
                continue
            yield ("rows of", name)
            quoted = '"' + name.replace('"', '""') + '"'
            cursor = con.execute(f"SELECT * FROM {quoted} NOT INDEXED")
            while rows := cursor.fetchmany(1024):
                yield from rows
    except sqlite3.DatabaseError:
        if not tolerant:
            raise
        yield _GAP


# --- CLI --------------------------------------------------------------------------------------


def write_ported(dst: Path, source: Path, report: Report) -> Path:
    """Write <dst>/PORTED.json (via a temp file) and return its path."""
    ported = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": str(source),
        "dirs": report.dirs,
        "conflicts": sorted(report.conflicts),
    }
    path = dst / "PORTED.json"
    dst.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=dst, prefix=".PORTED.json.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(ported, out, indent=2)
            out.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copy HKTrack's data and real fixtures.")
    parser.add_argument("--from", dest="source", required=True, help="the HKTrack root")
    parser.add_argument("--to", dest="dest", required=True, help="the data directory")
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args(argv)
    root = Path(args.source).expanduser().resolve()
    try:
        steps = plan(root / "data", root / "tests" / "fixtures", Path(args.dest))
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    dst = Path(args.dest).expanduser().resolve()
    report = run(steps, dry_run=args.dry_run)

    for top, counts in report.dirs.items():
        print(f"{top}: {counts['files']} files, {counts['bytes']} bytes")
    print(
        f"copied {len(report.copied)}, extended {len(report.extended)}, "
        f"unchanged {len(report.unchanged)}, conflicts {len(report.conflicts)}"
    )
    for rel in report.extended:
        print(f"extended: {rel}")
    for rel in report.conflicts:
        print(f"conflict (left alone): {rel}")
    if args.dry_run:
        print("dry run: nothing written")
    else:
        print(f"wrote {write_ported(dst, root, report)}")
    return 1 if report.conflicts else 0


if __name__ == "__main__":
    sys.exit(main())
