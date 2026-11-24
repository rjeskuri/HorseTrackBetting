"""Offline wiring for the `horsetrack scrape` tests (tests/core/test_scrape_cli.py and
tests/core/test_quiet_hours.py).

Everything goes through public seams of the code that already exists, so it holds wherever the
CLI builds its Fetcher or looks up its adapter:

* every `Fetcher` built during a test talks to the FakeSite and sleeps on the FakeClock (its
  `__init__` is wrapped, so the transport, sleep and clock are injected whoever constructs it);
* adapters are registered in `horsetrack.core.ingest.adapters.ADAPTERS` (mutated in place) under
  a module put into `sys.modules`, which is how `get_adapter` finds the real ones;
* the data directory and the repo root are under tmp_path, and the User-Agent comes from the
  environment (the source configs carry none).
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fakes import USER_AGENT, FakeClock, FakeSite

from horsetrack.core.ingest import adapters
from horsetrack.core.ingest.fetcher import Fetcher
from horsetrack.core.ingest.types import STATUS_BLOCKED, STATUS_FETCHED, FetchTask, ParsedPage

UA_VAR = "HORSETRACK_USER_AGENT"


def offline_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, site: FakeSite,
                clock: FakeClock) -> Path:  # fmt: skip
    """Data dir and repo root under tmp_path; every Fetcher wired to the fake site and clock.
    Returns the data directory."""
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv(UA_VAR, USER_AGENT)

    original = Fetcher.__init__

    def offline_init(self: Fetcher, source: str, *, http: Any, rng: Any = None, **_: Any) -> None:
        original(self, source, http=http, transport=site.transport, sleep=clock.sleep,
                 clock=clock, rng=rng)  # fmt: skip

    monkeypatch.setattr(Fetcher, "__init__", offline_init)
    return tmp_path / "data"


def register_adapter(monkeypatch: pytest.MonkeyPatch, source: str, adapter: Any) -> None:
    """Make `get_adapter(source)` return `adapter` for the duration of the test."""
    module_name = f"horsetrack_tests_fake_adapter_{source}"
    module = types.ModuleType(module_name)
    module.ADAPTER = adapter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(adapters.ADAPTERS, source, module_name)


class PlainAdapter:
    """Minimal source for any slug: a page with MARKER is content, nothing has follow-ups."""

    def __init__(self, source: str) -> None:
        self.source = source

    def fixture_tasks(self, start: date, end: date) -> list[FetchTask]:
        return []

    def meeting_tasks(self, fixture_pages: Iterable[Any]) -> list[FetchTask]:
        return []

    def followup_tasks(self, parsed: ParsedPage) -> list[FetchTask]:
        return []

    def classify(self, page_type: str, body: bytes) -> str:
        return STATUS_FETCHED if b"MARKER" in body else STATUS_BLOCKED

    def parse(self, page: Any) -> ParsedPage:
        return {}


def lines(result: Any) -> list[str]:
    return [ln for ln in result.stdout.splitlines() if ln.strip()]
