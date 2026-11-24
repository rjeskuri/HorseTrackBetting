"""What a source hands the fetcher and what it gets back (kept from HKTrack's sources/base.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# table name -> list of row dicts
ParsedPage = dict[str, list[dict[str, Any]]]

# classify() results; also the terminal statuses a fetched task can take in the manifest
STATUS_FETCHED = "fetched"
STATUS_EMPTY = "empty"  # valid response, no data
STATUS_ABANDONED = "abandoned"  # meeting or race abandoned: keep the page, expect no results
STATUS_BLOCKED = "blocked"  # expected content marker missing: counts toward the circuit breaker


@dataclass(frozen=True)
class FetchTask:
    source: str
    page_type: str
    key: str
    url: str
    year: int
    priority: int = 0
    method: str = "GET"
    body: str | None = None
    mutable: bool = False  # refetched on every fetch (race cards, odds); never served from cache


@dataclass(frozen=True)
class RawPage:
    source: str
    page_type: str
    key: str
    url: str
    body: bytes
    fetched_utc: datetime | None = None
    meta: dict[str, Any] = field(default_factory=dict)
