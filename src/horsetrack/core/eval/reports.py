"""Report writing that keeps licensed-source outputs out of the repository (#74).

JRDB's and JRA-VAN's terms forbid publishing figures derived from their data, and this repository
is public. ``write_report`` refuses to write a report built on a source in ``WITHHELD_SOURCES``
anywhere inside the repository and writes it elsewhere (normally under ``$HORSETRACK_DATA_DIR``).
Every write is atomic: a temporary file in the target directory, then ``os.replace``.

A source counts as withheld in any spelling: letter case, width, spaces and punctuation are
ignored, and a name that contains a withheld one (``JRA-VAN``, ``jrdb-kyi``) counts too.
"""

from __future__ import annotations

import os
import re
import unicodedata
import uuid
from collections.abc import Iterable
from pathlib import Path

from horsetrack.core import env

WITHHELD_SOURCES = frozenset({"jrdb", "jravan"})

_NOT_ALNUM = re.compile(r"[^0-9a-z]")
_COLLECTIONS = (tuple, list, set, frozenset)


class LicensedOutputInRepo(RuntimeError):
    """A licensed-source output was about to be written inside the repository."""


def _key(name: str) -> str:
    return _NOT_ALNUM.sub("", unicodedata.normalize("NFKC", name).casefold())


def is_withheld(names: Iterable[str]) -> bool:
    """Whether any of ``names`` is (or contains) a withheld source, in any spelling.

    Fails closed: a bare string, a non-collection or a non-string name counts as withheld.
    """
    if isinstance(names, (str, bytes)) or not isinstance(names, _COLLECTIONS):
        return True
    withheld = {_key(source) for source in WITHHELD_SOURCES}
    for name in names:
        if not isinstance(name, str):
            return True
        key = _key(name)
        if any(source in key for source in withheld):
            return True
    return False


def is_inside(path: Path | str, root: Path | str) -> bool:
    """Whether ``path`` is ``root`` or lies under it, after ``..``, symlinks, the working
    directory and letter case: every existing ancestor of the path, taken both as written and
    with symlinks resolved, is compared with ``root`` by device and inode. ``root`` must exist."""
    root_stat = os.stat(root)
    root_id = (root_stat.st_dev, root_stat.st_ino)
    candidates = (Path(os.path.abspath(path)), Path(os.path.realpath(path)))
    for candidate in candidates:
        for ancestor in (candidate, *candidate.parents):
            try:
                found = os.stat(ancestor)
            except OSError:  # missing, or not a directory: it cannot be the root
                continue
            if (found.st_dev, found.st_ino) == root_id:
                return True
    return False


def write_report(
    path: Path | str,
    text: str,
    *,
    sources: tuple[str, ...],
    repo_root: Path | str | None = None,
) -> Path:
    """Write ``text`` as UTF-8 to ``path`` atomically and return the path.

    ``sources`` names every data source the report is built on. When one is withheld and the
    path is inside ``repo_root`` (default: the project root), ``LicensedOutputInRepo`` is raised
    and nothing is created.
    """
    if isinstance(sources, (str, bytes)) or not isinstance(sources, _COLLECTIONS):
        raise TypeError(f"sources must be a tuple of source names, got {sources!r}")
    if not all(isinstance(name, str) for name in sources):
        raise TypeError(f"every source must be a name (str), got {sources!r}")
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    target = Path(path)
    if is_withheld(sources):
        root = env.repo_root() if repo_root is None else repo_root
        if is_inside(target, root):
            raise LicensedOutputInRepo(
                f"{target} is inside the repository at {root} and the report is built on a "
                f"licensed source ({', '.join(sources)}); write it under $HORSETRACK_DATA_DIR"
            )
    data = text.encode("utf-8")  # before any file exists: an unencodable text creates nothing
    _write_atomic(target, data)
    return target


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
