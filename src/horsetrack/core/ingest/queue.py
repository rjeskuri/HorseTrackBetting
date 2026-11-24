"""Fetch queue and manifest, ported from HKTrack: one SQLite row per (page_type, key).

The manifest is the resume point of a backfill. Every write is its own short transaction, so a
kill -9 loses at most the task that was in flight; the raw cache covers that one on the next run.

The on-disk schema is HKTrack's, unchanged, so a manifest HKTrack wrote opens without migration:
the column `jurisdiction` holds `FetchTask.source`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from horsetrack.core.ingest.types import FetchTask

STATUS_PENDING = "pending"
STATUS_FAILED = "failed"
STATUSES = (STATUS_PENDING, "fetched", "empty", "abandoned", STATUS_FAILED)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fetch_queue (
    jurisdiction   TEXT NOT NULL,
    page_type      TEXT NOT NULL,
    key            TEXT NOT NULL,
    url            TEXT NOT NULL,
    year           INTEGER NOT NULL,
    priority       INTEGER NOT NULL DEFAULT 0,
    method         TEXT NOT NULL DEFAULT 'GET',
    body           TEXT,
    mutable        INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'pending',
    http_status    INTEGER,
    final_url      TEXT,
    sha256         TEXT,
    bytes          INTEGER,
    fetched_utc    TEXT,
    attempts       INTEGER NOT NULL DEFAULT 0,
    error          TEXT,
    parser_version TEXT,
    PRIMARY KEY (page_type, key)
);
CREATE INDEX IF NOT EXISTS fetch_queue_next ON fetch_queue (status, priority DESC, key, page_type);
"""

_TASK_COLUMNS = "jurisdiction, page_type, key, url, year, priority, method, body, mutable"
_ITER_BATCH = 500

_SQL_INSERT = (
    f"INSERT OR IGNORE INTO fetch_queue ({_TASK_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_SQL_RESET_MUTABLE = """
UPDATE fetch_queue SET status = 'pending', url = ?, year = ?, priority = ?, method = ?, body = ?,
    mutable = 1
WHERE page_type = ? AND key = ?
"""
# A page first fetched while its task was mutable is provisional (a meeting not run yet answers "no
# data"). When the same task later arrives immutable it is owed exactly one more network fetch:
# mutable = 2 keeps it cache-bypassing (next_pending hands out mutable=True) until that fetch
# succeeds, then it is fetch-once.
_MUTABLE_FINAL_FETCH = 2
_SQL_FINAL_FETCH = f"""
UPDATE fetch_queue SET status = 'pending', url = ?, year = ?, priority = ?, method = ?, body = ?,
    mutable = {_MUTABLE_FINAL_FETCH}
WHERE page_type = ? AND key = ? AND mutable = 1
"""
_SQL_MARK = f"""
UPDATE fetch_queue SET status = ?, http_status = COALESCE(?, http_status),
    final_url = COALESCE(?, final_url), sha256 = COALESCE(?, sha256), bytes = COALESCE(?, bytes),
    fetched_utc = ?, error = ?, attempts = attempts + 1,
    mutable = CASE WHEN mutable = {_MUTABLE_FINAL_FETCH} AND ? THEN 0 ELSE mutable END
WHERE page_type = ? AND key = ?
"""
_SQL_MARK_NEW = f"""
INSERT INTO fetch_queue ({_TASK_COLUMNS}, status, http_status, final_url, sha256, bytes,
    fetched_utc, error, attempts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
"""
_SQL_ITER = """
SELECT * FROM fetch_queue WHERE page_type = ? AND status = ? AND key > ? ORDER BY key LIMIT ?
"""


def _task_values(task: FetchTask) -> tuple[Any, ...]:
    """Checked here, before SQLite sees them: a value it cannot bind raises
    sqlite3.InterfaceError, which callers must treat as a manifest fault, and INSERT OR IGNORE
    silently drops a row whose url is NULL. Both are adapter bugs and must surface as ordinary
    errors instead."""
    for name in ("source", "page_type", "key", "url", "method"):
        value = getattr(task, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"task {name} must be a non-empty string, got {value!r}")
    if task.body is not None and not isinstance(task.body, str):
        raise TypeError(f"task body must be str or None, got {type(task.body).__name__}")
    return (
        task.source, task.page_type, task.key, task.url, int(task.year), int(task.priority),
        task.method, task.body, int(task.mutable),
    )  # fmt: skip


def _task_from_row(row: sqlite3.Row) -> FetchTask:
    return FetchTask(
        source=row["jurisdiction"],
        page_type=row["page_type"],
        key=row["key"],
        url=row["url"],
        year=row["year"],
        priority=row["priority"],
        method=row["method"],
        body=row["body"],
        mutable=bool(row["mutable"]),
    )


def _type_filter(page_types: list[str] | None) -> tuple[str, list[str]]:
    if not page_types:
        return "", []
    return f" AND page_type IN ({', '.join('?' * len(page_types))})", list(page_types)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FetchQueue:
    """SQLite-backed task queue. Single writer (the backfill); readers may run alongside (WAL)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL + WAL: commits survive process death (kill -9) and the file stays consistent
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def __enter__(self) -> FetchQueue:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def enqueue(self, tasks: Iterable[FetchTask]) -> int:
        """Insert-or-ignore; returns how many rows are new. An existing *mutable* task is reset to
        pending. An existing row that was mutable when it was last planned and now arrives
        immutable is reset once more for its final fetch; rows that were always immutable are
        never touched."""
        added = 0
        # all validated before anything is written
        batch = [(task, _task_values(task)) for task in tasks]
        with self._conn:
            for task, values in batch:
                if self._conn.execute(_SQL_INSERT, values).rowcount == 1:
                    added += 1
                    continue
                url, year, priority, method, body = values[3:8]
                sql = _SQL_RESET_MUTABLE if task.mutable else _SQL_FINAL_FETCH
                self._conn.execute(
                    sql, (url, year, priority, method, body, task.page_type, task.key)
                )
        return added

    def next_pending(self, limit: int = 1, page_types: list[str] | None = None) -> list[FetchTask]:
        where, params = _type_filter(page_types)
        rows = self._conn.execute(
            f"SELECT {_TASK_COLUMNS} FROM fetch_queue WHERE status = 'pending'{where}"
            " ORDER BY priority DESC, key, page_type LIMIT ?",
            [*params, int(limit)],
        ).fetchall()
        return [_task_from_row(row) for row in rows]

    def mark(
        self,
        task: FetchTask,
        status: str,
        *,
        http_status: int | None = None,
        final_url: str | None = None,
        sha256: str | None = None,
        n_bytes: int | None = None,
        error: str | None = None,
    ) -> None:
        """Record one attempt's outcome. None for http_status/final_url/sha256/n_bytes keeps the
        old value (a cache hit knows the body hash but not the original HTTP status)."""
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}; expected one of {STATUSES}")
        outcome = (status, http_status, final_url, sha256, n_bytes, _utc_now(), error)
        # a failed final fetch is still owed
        final = int(status not in (STATUS_PENDING, STATUS_FAILED))
        with self._conn:
            cur = self._conn.execute(_SQL_MARK, (*outcome, final, task.page_type, task.key))
            if cur.rowcount == 0:
                # fetched outside the queue (e.g. a live page): still leave a manifest row
                self._conn.execute(_SQL_MARK_NEW, (*_task_values(task), *outcome))

    def counts(self) -> dict[tuple[str, str], int]:
        rows = self._conn.execute(
            "SELECT page_type, status, COUNT(*) FROM fetch_queue GROUP BY page_type, status"
        )
        return {(page_type, status): n for page_type, status, n in rows}

    def get(self, page_type: str, key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM fetch_queue WHERE page_type = ? AND key = ?", (page_type, key)
        ).fetchone()
        return dict(row) if row is not None else None

    def iter_status(self, page_type: str, status: str = "fetched") -> Iterator[dict[str, Any]]:
        """Rows of one page type in key order. Paged by key, so callers may enqueue while
        iterating."""
        last_key = ""
        while True:
            rows = self._conn.execute(
                _SQL_ITER, (page_type, status, last_key, _ITER_BATCH)
            ).fetchall()
            if not rows:
                return
            for row in rows:
                yield dict(row)
            last_key = rows[-1]["key"]

    def page_types(self) -> list[str]:
        return [
            r[0]
            for r in self._conn.execute(
                "SELECT DISTINCT page_type FROM fetch_queue ORDER BY page_type"
            )
        ]

    def count_parse_errors(self) -> int:
        """Pages that were kept but whose follow-ups could not be derived."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM fetch_queue WHERE status != 'failed' AND error LIKE 'parse:%'"
        ).fetchone()
        return int(row[0])

    def reset_failed(
        self, page_types: list[str] | None = None, max_attempts: int | None = None
    ) -> int:
        """Failed -> pending. With max_attempts, tasks already attempted that often stay failed:
        an automatic retry loop must not keep sending the same hopeless pages (which run first)
        to the site all night."""
        where, params = _type_filter(page_types)
        if max_attempts is not None and max_attempts > 0:
            where += " AND attempts < ?"
            params = [*params, int(max_attempts)]
        with self._conn:
            cur = self._conn.execute(
                f"UPDATE fetch_queue SET status = 'pending' WHERE status = 'failed'{where}", params
            )
        return cur.rowcount

    def set_error(self, page_type: str, key: str, error: str | None) -> None:
        """Update the error note without counting an attempt (used when cached pages are
        re-parsed)."""
        with self._conn:
            self._conn.execute(
                "UPDATE fetch_queue SET error = ? WHERE page_type = ? AND key = ?",
                (error, page_type, key),
            )

    def set_parser_version(self, page_type: str, key: str, version: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE fetch_queue SET parser_version = ? WHERE page_type = ? AND key = ?",
                (str(version), page_type, key),
            )
