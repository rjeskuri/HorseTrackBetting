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
from typing import Any

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


def _complete_members(raw: bytes, path: Path) -> Iterator[bytes]:
    """Decompressed gzip members, in order. Only whole, CRC-checked members count: a capture is all
    or nothing. A torn or damaged member costs that capture only: reading resumes at the next member
    header, since the logger is restarted and keeps appending after it."""
    start = 0
    while start < len(raw):
        member = zlib.decompressobj(wbits=31)
        try:
            data = member.decompress(memoryview(raw)[start:])
            complete = member.eof
        except zlib.error:
            data, complete = b"", False
        if complete:
            yield data
            start = len(raw) - len(member.unused_data)
            continue
        resume = raw.find(_GZIP_MEMBER_HEADER, start + 1)
        if resume < 0:
            log.warning("odds log %s: damaged data after byte %d ignored", path, start)
            return
        log.warning("odds log %s: damaged member at %d skipped, on at %d", path, start, resume)
        start = resume


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Every row of every whole member, as logged; empty when there is no log yet."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return []
    rows: list[dict[str, Any]] = []
    for data in _complete_members(raw, path):
        for line in data.decode("utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


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
