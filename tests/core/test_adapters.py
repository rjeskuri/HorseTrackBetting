"""The source-adapter contract and its registry (ticket #43). All synthetic.

Seams the implementation must honour (pinned here):

* `horsetrack.core.ingest.adapters.ADAPTERS`: a plain dict, source -> module path, holding
  {'jra': 'horsetrack.countries.japan.jra.adapter',
   'hkjc': 'horsetrack.countries.hongkong.hkjc.adapter'}.
* `get_adapter(source)` imports `ADAPTERS[source]` when called (not at import time) and returns its
  `ADAPTER`; a source missing from the registry is a `ValueError` naming it and the known sources.
* `SourceAdapter` is a runtime-checkable Protocol: `source` plus fixture_tasks, meeting_tasks,
  followup_tasks, classify and parse. HKTrack's live-only upcoming_tasks and live_odds are out.
"""

from __future__ import annotations

import sys
import types

import pytest

# isort: split
# The module under test gets a block of its own: while it is missing, it is the import error a red
# run reports, and the order lints the same before and after it exists.
from horsetrack.core.ingest import adapters

SourceAdapter = adapters.SourceAdapter
get_adapter = adapters.get_adapter

FIVE = ("fixture_tasks", "meeting_tasks", "followup_tasks", "classify", "parse")


def _adapter_class(source: str, methods=FIVE):
    namespace = {"source": source}
    for name in methods:
        namespace[name] = lambda self, *a, **k: []
    return type(f"Fake_{source}", (), namespace)


def test_registry_names_the_two_country_adapters():
    assert adapters.ADAPTERS == {
        "jra": "horsetrack.countries.japan.jra.adapter",
        "hkjc": "horsetrack.countries.hongkong.hkjc.adapter",
    }


def test_get_adapter_imports_the_registered_module(monkeypatch):
    # Two sources registered at run time, each backed by a module injected into sys.modules:
    # get_adapter returns that module's ADAPTER object itself, and the right one per source.
    made = {}
    for source in ("testsrc_a", "testsrc_b"):
        module_name = f"_horsetrack_test_adapters.{source}"
        module = types.ModuleType(module_name)
        module.ADAPTER = _adapter_class(source)()
        made[source] = module.ADAPTER
        monkeypatch.setitem(sys.modules, module_name, module)
        monkeypatch.setitem(adapters.ADAPTERS, source, module_name)
    monkeypatch.setitem(
        sys.modules, "_horsetrack_test_adapters", types.ModuleType("_horsetrack_test_adapters")
    )

    for source in ("testsrc_b", "testsrc_a", "testsrc_b"):
        got = get_adapter(source)
        assert got is made[source]
        assert got.source == source


def test_get_adapter_reads_the_registry_at_call_time(monkeypatch, tmp_path):
    # A real package on disk, registered after adapters was imported: get_adapter imports it then.
    pkg = tmp_path / "_horsetrack_tmp_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "late.py").write_text(
        "class _A:\n"
        "    source = 'late'\n"
        "    def fixture_tasks(self, start, end): return []\n"
        "    def meeting_tasks(self, fixture_pages): return []\n"
        "    def followup_tasks(self, parsed): return []\n"
        "    def classify(self, page_type, body): return 'fetched'\n"
        "    def parse(self, page): return {}\n"
        "ADAPTER = _A()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("_horsetrack_tmp_pkg", "_horsetrack_tmp_pkg.late"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(adapters.ADAPTERS, "late", "_horsetrack_tmp_pkg.late")

    got = get_adapter("late")
    assert got is sys.modules["_horsetrack_tmp_pkg.late"].ADAPTER
    assert got.source == "late"
    assert isinstance(got, SourceAdapter)


@pytest.mark.parametrize("source", ["nope", "JRA", "jra ", "", "os", "japan.jra"])
def test_unknown_source_is_an_error(source, monkeypatch):
    # Nothing outside the registry is ever imported, even a name that resolves to a real module.
    before = set(sys.modules)
    with pytest.raises(ValueError) as excinfo:
        get_adapter(source)
    message = str(excinfo.value)
    assert repr(source) in message or f'"{source}"' in message
    for known in adapters.ADAPTERS:
        assert known in message
    assert {m for m in set(sys.modules) - before if m.startswith("horsetrack")} == set()


def test_source_adapter_is_the_five_ingest_methods():
    # No upcoming_tasks or live_odds needed.
    assert isinstance(_adapter_class("testsrc")(), SourceAdapter)
    for missing in FIVE:
        partial = _adapter_class("testsrc", [m for m in FIVE if m != missing])()
        assert not isinstance(partial, SourceAdapter), missing
