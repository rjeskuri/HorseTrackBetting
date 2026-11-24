"""Offline fakes for the fetcher tests: a fake multi-host site and a fake monotonic clock.

Ported from HKTrack tests/unit/ingest/ingest_fakes.py. Everything here is synthetic: no real page,
no real User-Agent. HKTrack's FakeAdapter is left for the ticket that ports the runner.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

import httpx

# isort: split
# A block of its own: ruff counts horsetrack.core.ingest as third-party until it exists.
from horsetrack.core.ingest.types import FetchTask

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
