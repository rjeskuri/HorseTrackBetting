"""The Japan odds snapshot log and its post-time sidecar (ported from HKTrack, format unchanged).

`odds_snapshots/jra/<YYYY-MM-DD>/<VENUE>.jsonl.gz` holds one gzip member per capture, one JSON
object per line; pool totals are rows whose `combination` is "__pool__". Each meeting's post times
sit beside the log in `<VENUE>.races.json`. The row keys and the sidecar shape are in
docs/specs/odds-log.md.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import zlib
from collections.abc import Iterable, Iterator
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

log = logging.getLogger(__name__)

POOL_TOTAL = "__pool__"
# magic + deflate: where a member written by gzip.compress starts
_GZIP_MEMBER_HEADER = b"\x1f\x8b\x08"


def _utc_iso(value: datetime) -> str:
    """ISO text in UTC; a naive datetime is read as UTC, as HKTrack did."""
    aware = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return _utc_iso(value)
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalars, without importing numpy
        return value.item()
    return str(value)


# ---------------------------------------------------------------- the log


def append_capture(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Append one capture as one gzip member, in one write, fsynced. Returns the rows written;
    no rows, no write."""
    rows = list(rows)
    if not rows:
        return 0
    payload = "".join(
        json.dumps(row, default=_json_value, separators=(",", ":")) + "\n" for row in rows
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as fh:
        fh.write(gzip.compress(payload.encode("utf-8")))  # one complete member per capture
        fh.flush()
        os.fsync(fh.fileno())
    return len(rows)


# zlib is fed bounded chunks, so a member's unused_data is at most one chunk and a scan stays linear
_CHUNK = 1 << 16


def _next_header(fh: BinaryIO, after: int) -> int | None:
    """The offset of the first member header at or after byte `after`, or None."""
    fh.seek(after)
    offset, carry = after, b""
    while True:
        chunk = fh.read(_CHUNK)
        if not chunk:
            return None
        window = carry + chunk
        found = window.find(_GZIP_MEMBER_HEADER)
        if found >= 0:
            return offset - len(carry) + found
        carry = window[-(len(_GZIP_MEMBER_HEADER) - 1) :]
        offset += len(chunk)


def _complete_members(path: Path, start: int) -> Iterator[tuple[int, bytes]]:
    """(end offset, decompressed bytes) of each gzip member from byte `start`, in order. Only whole,
    CRC-checked members count: a capture is all or nothing. A torn or damaged member costs that
    capture only: reading resumes at the next member header, since the logger is restarted and
    keeps appending after it. A torn trailing member is not yielded and no end offset passes it."""
    try:
        fh = open(path, "rb")
    except FileNotFoundError:
        return
    with fh:
        fh.seek(start)
        pos, pending = start, b""  # pending: bytes read from `pos` on, not yet fed to zlib
        while True:
            member = zlib.decompressobj(wbits=31)
            parts: list[bytes] = []
            fed = 0
            try:
                while not member.eof:
                    data = pending or fh.read(_CHUNK)
                    pending = b""
                    if not data:
                        break
                    fed += len(data)
                    parts.append(member.decompress(data))
            except zlib.error:
                pass
            if member.eof:
                end = pos + fed - len(member.unused_data)
                yield end, b"".join(parts)
                pos, pending = end, member.unused_data
                continue
            if fed == 0:
                return  # the end of the log, on a member boundary
            resume = _next_header(fh, pos + 1)
            if resume is None:
                log.warning("odds log %s: damaged or unfinished data after byte %d", path, pos)
                return
            log.warning("odds log %s: damaged member at %d skipped, on at %d", path, pos, resume)
            fh.seek(resume)
            pos, pending = resume, b""


def _member_rows(data: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def iter_members_with_offsets(
    path: Path, *, start: int = 0
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    """(end offset, rows) of each whole member from byte offset `start`, in log order. The end
    offset is the byte just after the member, where a later read can resume; nothing when there is
    no log yet. The file is read in bounded chunks and one member is held at a time."""
    for end, data in _complete_members(Path(path), start):
        yield end, _member_rows(data)


def iter_members(path: Path, *, start: int = 0) -> Iterator[list[dict[str, Any]]]:
    """The rows of each whole member (one capture) from byte offset `start`, in log order; nothing
    when there is no log yet. Only one decompressed member is held at a time, so a reader can
    filter as it goes."""
    for _, rows in iter_members_with_offsets(path, start=start):
        yield rows


class LogTail:
    """A growing log read on: each poll returns the rows of the members appended since the previous
    poll, reading from where the last whole member ended. A torn trailing member (a write still in
    progress) is read on the poll after it is complete. Polling a missing log creates nothing."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.offset = 0

    def poll(self) -> list[dict[str, Any]]:
        try:
            size = os.stat(self.path).st_size
        except FileNotFoundError:
            return []
        if size < self.offset:  # the log is append-only; a shorter file is a replaced one
            log.warning("odds log %s shrank below byte %d: read from 0", self.path, self.offset)
            self.offset = 0
        if size == self.offset:
            return []
        rows: list[dict[str, Any]] = []
        for end, member in iter_members_with_offsets(self.path, start=self.offset):
            rows.extend(member)
            self.offset = end
        return rows


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Every row of every whole member, as logged; empty when there is no log yet."""
    return [row for rows in iter_members(path) for row in rows]


def _instant(value: Any) -> datetime | None:
    """A logged timestamp as an aware UTC datetime, or None when it cannot be read."""
    if isinstance(value, str):
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            value = datetime.fromisoformat(text)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _latest(rows: Iterable[dict[str, Any]], key_fields: tuple[str, ...], totals: bool) -> dict:
    """Newest row per key by snapshot_utc; at the same instant the later row in the log wins.
    Rows without a readable snapshot_utc can never be the latest and are left out."""
    best: dict[tuple, tuple[datetime, dict[str, Any]]] = {}
    for row in rows:
        if (row.get("combination") == POOL_TOTAL) is not totals:
            continue
        at = _instant(row.get("snapshot_utc"))
        if at is None:
            continue
        key = tuple(row.get(field) for field in key_fields)
        current = best.get(key)
        if current is None or at >= current[0]:
            best[key] = (at, row)
    return {key: row for key, (_, row) in best.items()}


def latest_prices(rows: Iterable[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    """The newest price row per (race_id, pool, combination); pool totals left out."""
    return _latest(rows, ("race_id", "pool", "combination"), totals=False)


def latest_totals(rows: Iterable[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    """The newest pool-total row per (race_id, pool) that carries an investment figure."""
    with_figure = (row for row in rows if row.get("investment") is not None)
    return _latest(with_figure, ("race_id", "pool"), totals=True)


# ---------------------------------------------------------------- post-time sidecar


def write_post_times(path: Path, post_times: dict[int, datetime], updated_utc: datetime) -> None:
    """Replace the sidecar as a whole: written to a temporary file, fsynced, then os.replace. A
    failure leaves the previous sidecar as it was and no temporary file behind."""
    path = Path(path)
    document = {
        "updated_utc": _utc_iso(updated_utc),
        "post_times": {str(int(no)): _utc_iso(at) for no, at in sorted(post_times.items())},
    }
    text = json.dumps(document, indent=2) + "\n"  # serialise before touching the disk
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_post_times(path: Path) -> dict[int, datetime]:
    """Race number -> post time as an aware UTC datetime; empty when there is no sidecar."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    out: dict[int, datetime] = {}
    for no, text in document.get("post_times", {}).items():
        at = _instant(text)
        if at is None:
            raise ValueError(f"post-time sidecar {path}: race {no} has no readable time: {text!r}")
        out[int(no)] = at
    return out
