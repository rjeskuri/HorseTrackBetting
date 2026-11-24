"""Offline fakes for the parse worker and parse cache tests (#47). Everything here is synthetic.

A page body is JSON: {"tables": {table: [row, ...]}, "key_mismatch": bool}. The fake adapter parses
it into those tables plus a `_page` metadata table (key, url, n_bytes, fetched_utc, key_mismatch),
counts every parse, raises for the keys in `broken`, and answers `parser_version(page_type)` from
`versions` (the adapter hook HKTrack's parse_worker.parser_version consults first).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import types
from typing import Any

from horsetrack.core import paths
from horsetrack.core.ingest import adapters

SOURCE = "testsrc"
OTHER = "othersrc"
YEAR = 2026


class FakeParseAdapter:
    def __init__(self, source: str, versions: dict[str, Any] | None = None) -> None:
        self.source = source
        self.versions: dict[str, Any] = dict(versions or {})
        self.broken: dict[str, BaseException] = {}  # key -> what parse raises for that page
        self.pages: list[Any] = []  # every RawPage handed to parse, in order

    def fixture_tasks(self, start, end):
        return []

    def meeting_tasks(self, fixture_pages):
        return []

    def followup_tasks(self, parsed):
        return []

    def classify(self, page_type, body):
        return "fetched"

    def parser_version(self, page_type: str):
        return self.versions.get(page_type)

    def parse(self, page):
        self.pages.append(page)
        if page.key in self.broken:
            raise self.broken[page.key]
        spec = json.loads(page.body.decode("utf-8"))
        tables = {name: [dict(row) for row in rows] for name, rows in spec["tables"].items()}
        tables["_page"] = [
            {
                "key": page.key,
                "url": page.url,
                "n_bytes": len(page.body),
                "fetched_utc": page.fetched_utc,
                "key_mismatch": bool(spec.get("key_mismatch", False)),
            }
        ]
        return tables

    def parsed_keys(self) -> list[tuple[str, str]]:
        return [(p.page_type, p.key) for p in self.pages]


def register(monkeypatch, adapter: FakeParseAdapter) -> FakeParseAdapter:
    """Put `adapter` in the registry under its source, backed by a module in sys.modules."""
    module_name = f"_horsetrack_test_parse_adapters.{adapter.source}"
    module = types.ModuleType(module_name)
    module.ADAPTER = adapter
    monkeypatch.setitem(sys.modules, "_horsetrack_test_parse_adapters", types.ModuleType("_p"))
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(adapters.ADAPTERS, adapter.source, module_name)
    return adapter


def body(tables: dict[str, list[dict[str, Any]]], key_mismatch: bool = False) -> bytes:
    # Key order is kept on purpose: the payload's columns follow the order the parser emits.
    return json.dumps({"tables": tables, "key_mismatch": key_mismatch}).encode()


def write_page(source: str, page_type: str, key: str, data: bytes, year: int = YEAR) -> str:
    """Gzip `data` where the raw cache keeps it; returns the sha256 of the uncompressed body."""
    path = paths.raw_page_path(source, page_type, year, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        fh.write(data)
    return hashlib.sha256(data).hexdigest()


def expected_payload_tables(tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """HKTrack's payload shape for plain JSON rows: columns in first-seen order, missing -> None."""
    out = {}
    for name, rows in tables.items():
        columns = list(dict.fromkeys(c for row in rows for c in row))
        out[name] = {"columns": columns, "rows": [[row.get(c) for c in columns] for row in rows]}
    return out
