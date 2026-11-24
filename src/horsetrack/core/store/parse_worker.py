"""One page in, one compact payload out (ported from HKTrack's store/parse_worker.py).

Runs inside spawned worker processes, so: top-level functions only, plain-tuple jobs, and no pandas
or pyarrow imports (the adapter's own imports are the whole start-up cost). Parses only through
`get_adapter(source)`; never touches the network and never writes.

Payload = zlib(JSON) of {table: {"columns": [...], "rows": [[...], ...]}}; dates are ISO strings and
datetimes UTC ISO strings (the table schema turns them back into date32 / timestamp).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import zlib
from datetime import date, datetime, timezone
from importlib import import_module
from typing import Any, NamedTuple

from horsetrack.core.ingest.adapters import ADAPTERS, get_adapter
from horsetrack.core.ingest.types import ParsedPage, RawPage
from horsetrack.core.paths import raw_page_path

STATUS_OK = "ok"
STATUS_FAILED = "failed"  # the parser raised
STATUS_UNREADABLE = "unreadable"  # raw file missing, not valid gzip, or not the manifest's body

_MAX_ERROR_CHARS = 300
_COMPRESS_LEVEL = 3


class ParseJob(NamedTuple):
    source: str
    page_type: str
    key: str
    year: int
    url: str
    final_url: str | None
    fetched_utc: str | None
    sha256: str
    parser_version: str


class ParseResult(NamedTuple):
    page_type: str
    key: str
    sha256: str
    parser_version: str
    status: str
    key_mismatch: bool
    n_rows: int
    error: str | None
    payload: bytes | None


def parser_version(source: str, page_type: str) -> str | None:
    """The adapter's `parser_version(page_type)` as a string, None meaning "no parser" (that page
    type is skipped, never an error). An adapter without the hook falls back to PARSER_VERSION of
    `parse_<page_type>` next to the adapter's module: "0" when that module declares none, None when
    there is no such module."""
    adapter_hook = getattr(get_adapter(source), "parser_version", None)
    if callable(adapter_hook):
        version = adapter_hook(page_type)
        return None if version is None else str(version)
    if not page_type.isidentifier():
        return None
    package = ADAPTERS[source].rpartition(".")[0]
    name = f"{package}.parse_{page_type}"
    try:
        module = import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name:
            return None
        raise
    return str(getattr(module, "PARSER_VERSION", "0"))


def _as_utc(value: datetime) -> datetime:
    """A naive datetime is taken as UTC; an aware one is converted."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return _as_utc(parsed)


def read_raw_page(job: ParseJob) -> RawPage | None:
    """The cached page with the manifest's URL and fetch time (not the file mtime, which is not
    reproducible). None when the file is missing or corrupt, or when its body is not the one the
    manifest recorded (a refetch in progress, a mixed-up restore): parsing that would file another
    body under this sha256 in the parse cache, where it would stay."""
    path = raw_page_path(job.source, job.page_type, int(job.year), job.key)
    try:
        with gzip.open(path, "rb") as fh:
            body = fh.read()
        if not body and path.stat().st_size == 0:  # a zero-byte file reads as an empty stream
            return None
    except (OSError, EOFError, zlib.error):
        return None
    if job.sha256 and hashlib.sha256(body).hexdigest() != job.sha256:
        return None
    meta = {
        "final_url": job.final_url,
        "sha256": job.sha256,
        "n_bytes": len(body),
        "from_cache": True,
    }
    return RawPage(
        job.source, job.page_type, job.key, job.url or "", body, _utc(job.fetched_utc), meta
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return _as_utc(value).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, set | frozenset):
        return sorted(value)
    return str(value)


def encode_tables(parsed: ParsedPage) -> tuple[bytes, int, bool]:
    """(payload, data row count, key_mismatch). Tables whose name starts with "_" are metadata."""
    tables: dict[str, Any] = {}
    n_rows, mismatch = 0, False
    for name, rows in parsed.items():
        rows = [row for row in rows or [] if isinstance(row, dict)]
        columns = list(dict.fromkeys(column for row in rows for column in row))
        tables[name] = {
            "columns": columns,
            "rows": [[row.get(column) for column in columns] for row in rows],
        }
        if name.startswith("_"):
            mismatch = mismatch or any(bool(row.get("key_mismatch")) for row in rows)
        else:
            n_rows += len(rows)
    text = json.dumps(tables, default=_json_default, separators=(",", ":"), ensure_ascii=False)
    return zlib.compress(text.encode("utf-8"), _COMPRESS_LEVEL), n_rows, mismatch


def decode_tables(payload: bytes) -> dict[str, dict[str, Any]]:
    return json.loads(zlib.decompress(payload).decode("utf-8"))


def parse_job(job: ParseJob) -> ParseResult:
    """Never raises: a page that cannot be read or parsed comes back with a status and a one-line
    error, and without a payload."""
    job = ParseJob(*job)
    head = (job.page_type, job.key, job.sha256, job.parser_version)
    try:
        page = read_raw_page(job)
        if page is None:
            return ParseResult(
                *head,
                STATUS_UNREADABLE,
                False,
                0,
                "raw file missing, not valid gzip, or not the body the manifest recorded (sha256)",
                None,
            )
        parsed = get_adapter(job.source).parse(page)
        payload, n_rows, mismatch = encode_tables(parsed)
    except Exception as exc:  # scraped input: whatever a parser trips over must not stop the build
        error = " ".join(f"{type(exc).__name__}: {exc}".split())[:_MAX_ERROR_CHARS]
        return ParseResult(*head, STATUS_FAILED, False, 0, error, None)
    return ParseResult(*head, STATUS_OK, mismatch, n_rows, None, payload)
