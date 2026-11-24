"""Offline fakes for the ingest tests: a fake multi-host site, a fake monotonic clock and a fake
line-based source adapter.

Ported from HKTrack tests/unit/ingest/ingest_fakes.py. Everything here is synthetic: no real page,
no real User-Agent.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from datetime import date
from typing import Any

import httpx

# isort: split
# A block of its own: ruff counts horsetrack.core.ingest as third-party until it exists.
from horsetrack.core.ingest.types import (
    STATUS_ABANDONED,
    STATUS_BLOCKED,
    STATUS_EMPTY,
    STATUS_FETCHED,
    FetchTask,
    ParsedPage,
    RawPage,
)

SOURCE = "testsrc"
BASE = "https://site.test"
USER_AGENT = "horsetrack-tests/1.0"

# The source config's http block, in the shape the Fetcher takes (a plain mapping).
HTTP: dict[str, Any] = {
    "user_agent": USER_AGENT,
    "http2": True,
    "timeout_s": 5,
    "max_retries": 4,
    "backoff_base_s": 2.0,
    "circuit_breaker": {"stop_on_status": [403, 429], "max_consecutive_failures": 5},
    "hosts": {"default": {"requests_per_second": 1.0, "jitter": 0.3}},
}


class FakeClock:
    """Monotonic clock that only moves when someone sleeps (or a test advances it)."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSite:
    """URL -> response table behind an httpx.MockTransport, any host; counts every request."""

    def __init__(self) -> None:
        self.pages: dict[str, Any] = {}
        self.hits: Counter[str] = Counter()
        self.requests: list[httpx.Request] = []
        self.on_request: Callable[[httpx.Request], None] | None = None

    def add(self, url: str, body: bytes | str = b"", status: int = 200) -> None:
        self.pages[url] = (status, body.encode() if isinstance(body, str) else body)

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.hits[url] += 1
        self.requests.append(request)
        if self.on_request is not None:
            self.on_request(request)
        entry = self.pages.get(url, (404, b"not found"))
        if callable(entry):
            return entry(request)
        status, body = entry
        return httpx.Response(status, content=body)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    @property
    def total_hits(self) -> int:
        return sum(self.hits.values())

    def hits_on(self, host: str) -> int:
        return sum(n for url, n in self.hits.items() if httpx.URL(url).host == host)


def make_task(page_type: str, key: str, base: str = BASE, **kw: Any) -> FetchTask:
    kw.setdefault("url", f"{base}/{page_type}/{key}")
    kw.setdefault("year", 2026)
    return FetchTask(source=SOURCE, page_type=page_type, key=key, **kw)


class FakeAdapter:
    """Line-based fake source that satisfies the SourceAdapter contract.

    fixture page:  'MARKER' + lines 'meeting:<key>'   -> results tasks (via meeting_tasks)
    results page:  'MARKER' + lines 'horse:<id>'      -> horse tasks (via parse -> followup_tasks)
    sentinels:     'No information.' -> empty, 'ABANDONED' -> abandoned, no 'MARKER' -> blocked,
                   'BROKEN' -> parse() raises
    """

    source = SOURCE

    def __init__(self, months: list[str] | None = None, follow_horses: bool = True) -> None:
        self.months = months or ["2026-09"]
        self.follow_horses = follow_horses
        self.parsed: list[str] = []
        self.planned: list[tuple[date, date]] = []

    def fixture_tasks(self, start: date, end: date) -> list[FetchTask]:
        self.planned.append((start, end))
        return [make_task("fixture", m, priority=int(m.replace("-", ""))) for m in self.months]

    def meeting_tasks(self, fixture_pages: Iterable[RawPage]) -> list[FetchTask]:
        tasks = []
        for page in fixture_pages:
            for line in page.body.decode().splitlines():
                if line.startswith("meeting:"):
                    tasks.append(make_task("results", line.split(":", 1)[1], priority=50))
        return tasks

    def followup_tasks(self, parsed: ParsedPage) -> list[FetchTask]:
        if not self.follow_horses:
            return []
        return [make_task("horse", r["horse_id"], priority=10) for r in parsed.get("runners", [])]

    def classify(self, page_type: str, body: bytes) -> str:
        if b"No information." in body:
            return STATUS_EMPTY
        if b"ABANDONED" in body:
            return STATUS_ABANDONED
        if b"MARKER" not in body:
            return STATUS_BLOCKED
        return STATUS_FETCHED

    def parse(self, page: RawPage) -> ParsedPage:
        self.parsed.append(f"{page.page_type}/{page.key}")
        text = page.body.decode()
        if "BROKEN" in text:
            raise ValueError("unexpected table layout")
        runners = [
            {"horse_id": ln.split(":", 1)[1]} for ln in text.splitlines() if ln.startswith("horse:")
        ]
        return {"runners": runners}
