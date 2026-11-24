"""Incremental parse cache at `interim_dir(source)/parse_cache.sqlite` (ported from HKTrack's
store/cache.py; the file format is HKTrack's, so its caches open and serve unchanged).

One row per (page_type, key) holding the page's parsed payload together with the sha256 and parser
version it was produced from. A page is parsed again only when either of those differs from what
the manifest and the parser say now. Failed parses are never stored, so a fixed parser or a
repaired file is picked up by the next build without any flag.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path

from horsetrack.core.paths import interim_dir
from horsetrack.core.store.parse_worker import ParseResult

CACHE_FORMAT = 1  # bump when the payload encoding changes: the whole cache is dropped and rebuilt

_SCHEMA = """
CREATE TABLE IF NOT EXISTS parsed (
    page_type      TEXT NOT NULL,
    key            TEXT NOT NULL,
    sha256         TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    key_mismatch   INTEGER NOT NULL DEFAULT 0,
    n_rows         INTEGER NOT NULL DEFAULT 0,
    payload        BLOB NOT NULL,
    PRIMARY KEY (page_type, key)
)
"""
_SQL_PUT = "INSERT OR REPLACE INTO parsed VALUES (?, ?, ?, ?, ?, ?, ?)"


def cache_path(source: str) -> Path:
    return interim_dir(source) / "parse_cache.sqlite"


class ParseCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30.0)
        fmt = int(CACHE_FORMAT)  # read on every open, so a bumped constant takes effect at once
        with self._conn:
            if self._conn.execute("PRAGMA user_version").fetchone()[0] != fmt:
                self._conn.execute("DROP TABLE IF EXISTS parsed")
            self._conn.execute(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version = {fmt}")

    def __enter__(self) -> ParseCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def index(self) -> dict[tuple[str, str], tuple[str, str, bool]]:
        """(page_type, key) -> (sha256, parser_version, key_mismatch) for everything cached."""
        rows = self._conn.execute(
            "SELECT page_type, key, sha256, parser_version, key_mismatch FROM parsed"
        )
        return {
            (page_type, key): (sha, version, bool(mismatch))
            for page_type, key, sha, version, mismatch in rows
        }

    def put(self, results: Iterable[ParseResult]) -> None:
        """Stores the results that carry a payload, one transaction per batch: a killed build keeps
        every batch that was committed."""
        rows = [
            (
                r.page_type,
                r.key,
                r.sha256,
                r.parser_version,
                int(r.key_mismatch),
                int(r.n_rows),
                r.payload,
            )
            for r in results
            if r.payload is not None
        ]
        with self._conn:
            self._conn.executemany(_SQL_PUT, rows)

    def payloads(self, page_type: str) -> Iterator[tuple[str, str, str, bytes]]:
        """(key, sha256, parser_version, payload) of one page type in key order."""
        yield from self._conn.execute(
            "SELECT key, sha256, parser_version, payload FROM parsed"
            " WHERE page_type = ? ORDER BY key",
            (page_type,),
        )
