"""The source-adapter contract, ported from HKTrack's sources/base.py, and the adapter registry.

Ingest and everything downstream depend only on `SourceAdapter`. Each country module named in
`ADAPTERS` exposes `ADAPTER`; `get_adapter` imports it when asked, never at import time. HKTrack's
live-only methods (upcoming_tasks, live_odds) are not part of this contract.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from importlib import import_module
from typing import Protocol, runtime_checkable

from horsetrack.core.ingest.types import FetchTask, ParsedPage, RawPage

# source -> module exposing ADAPTER
ADAPTERS: dict[str, str] = {
    "jra": "horsetrack.countries.japan.jra.adapter",
    "hkjc": "horsetrack.countries.hongkong.hkjc.adapter",
}


@runtime_checkable
class SourceAdapter(Protocol):
    source: str

    def fixture_tasks(self, start: date, end: date) -> list[FetchTask]: ...

    def meeting_tasks(self, fixture_pages: Iterable[RawPage]) -> list[FetchTask]: ...

    def followup_tasks(self, parsed: ParsedPage) -> list[FetchTask]: ...

    def classify(self, page_type: str, body: bytes) -> str: ...

    def parse(self, page: RawPage) -> ParsedPage: ...


def get_adapter(source: str) -> SourceAdapter:
    """The registered adapter for `source`; a source outside the registry imports nothing."""
    module_name = ADAPTERS.get(source)
    if module_name is None:
        known = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"unknown source {source!r}; known sources: {known}")
    return import_module(module_name).ADAPTER
