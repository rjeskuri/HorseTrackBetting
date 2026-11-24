"""`horsetrack store build | check | info -s SOURCE` (#50), driven through the root CLI.

Ported from HKTrack tests/unit/store/test_store_cli.py (the same 8 cases, plus the options and the
failure paths the port adds) on the synthetic pages and fake adapter of store_fakes.py. No real
page, no network. Seams pinned:

* `horsetrack.core.store.cli` is a module; the root app mounts its group as `store`, so
  `horsetrack store <command> --source/-s SOURCE` runs it. With exactly one source registered in
  `horsetrack.core.ingest.adapters.ADAPTERS`, `--source` defaults to it; with several it is
  required.
* `cli.source_config(source) -> SourceConfig`: the typed config the checks need (for hkjc,
  `load_source("hongkong", "hkjc")`), looked up through the module attribute at call time so a test
  can replace it (the test source has no YAML; the synthetic dividends are per 10, so the tests
  answer with the hkjc config).
* Output is a few lines (build <= 20, as it runs the checks too; check and info <= 15); the details
  go to the reports under `reports_dir()`. A failure prints exactly one line starting with `ERROR`
  and exits 1; no traceback reaches the terminal.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from store_fakes import (
    PAGES,
    RESULTS_ABANDONED,
    SECTIONAL_2026,
    SOURCE,
    FakeStoreAdapter,
    populate,
    register,
)
from typer.testing import CliRunner

from horsetrack.core import paths
from horsetrack.core.config import load_source
from horsetrack.core.ingest import adapters
from horsetrack.core.store import build, tables

# isort: split
# A block of its own: while the module is missing, it is the import error a red run reports.
from horsetrack.core.store import cli as cli_module

# isort: split
from horsetrack.cli import app

VAR = "HORSETRACK_DATA_DIR"
HKJC = load_source("hongkong", "hkjc")
runner = CliRunner()

# PAGES with the abandoned race as a results page shows it (no field, no runners): all checks pass.
_ABANDONED = {**RESULTS_ABANDONED, "races": [{**RESULTS_ABANDONED["races"][0], "field_size": 0}]}
CHECK_PAGES = [
    p._replace(tables={**_ABANDONED, "runners": []}) if p.key == "2019-11-13_HV_R01" else p
    for p in PAGES
]


# ---------------------------------------------------------------- fixtures and helpers


@pytest.fixture
def data(tmp_path, monkeypatch) -> Path:
    path = (tmp_path / "data").resolve()
    monkeypatch.setenv(VAR, str(path))
    # the test source has no YAML: the checks run on the hkjc config (dividends quoted per 10)
    monkeypatch.setattr(cli_module, "source_config", lambda source: HKJC)
    return path


def _only_test_source(monkeypatch) -> None:
    for name in [n for n in adapters.ADAPTERS if n != SOURCE]:
        monkeypatch.delitem(adapters.ADAPTERS, name)


@pytest.fixture
def raw_root(data, monkeypatch) -> Path:
    """Raw cache + manifest of PAGES (one key mismatch, one field-size warning): nothing built."""
    register(monkeypatch, FakeStoreAdapter())
    populate(PAGES)
    return data


@pytest.fixture
def built_root(data, monkeypatch) -> Path:
    """CHECK_PAGES built: every check passes."""
    register(monkeypatch, FakeStoreAdapter())
    populate(CHECK_PAGES)
    build.build_tables(SOURCE, workers=1)
    return data


def run(*args: str):
    return runner.invoke(app, ["store", *args])


def lines(result) -> list[str]:
    return [line for line in result.output.splitlines() if line.strip()]


def errors(result) -> list[str]:
    return [line for line in lines(result) if line.startswith("ERROR")]


def build_record() -> dict:
    return json.loads((paths.processed_dir(SOURCE) / build.BUILD_FILE).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- build


def test_build_prints_a_short_summary_and_the_tables_exist(raw_root):
    result = run("build", "-s", SOURCE, "--workers", "1")
    assert result.exit_code == 0, result.output
    out = lines(result)
    assert len(out) <= 20 and out[0].startswith(f"store build {SOURCE}: snapshot ")
    # 14 runner rows, 1 key mismatch, written independently of the summary's own wording
    assert any("runners 14" in line for line in out)
    assert any("1 key mismatches" in line for line in out)
    assert out[-1].startswith(f"store check {SOURCE}: OK")
    assert not errors(result)
    assert tables.row_counts(SOURCE)["races"] == 4
    assert tables.row_counts(SOURCE)["runners"] == 14
    # the snapshot printed is the one written to _build.json
    assert out[0].split("snapshot ")[1].split()[0] == build_record()["snapshot_id"][:12]
    # details go to reports, not the terminal
    assert (paths.reports_dir() / f"store-build-{SOURCE}.md").is_file()
    assert (paths.reports_dir() / f"store-check-{SOURCE}.md").is_file()


def test_build_accepts_page_types_limit_and_workers(raw_root):
    result = run(
        "build",
        "--source",
        SOURCE,
        "--page-types",
        "results, horse",
        "--limit",
        "1",
        "--workers",
        "1",
    )
    assert result.exit_code == 0, result.output
    counts = tables.row_counts(SOURCE)
    assert counts["races"] == 1 and counts["horses"] == 1
    assert counts.get("sectionals", 0) == 0  # a page type not asked for is not parsed now
    # without the options every page is parsed: the option changed what was built
    again = run("build", "-s", SOURCE, "--workers", "1")
    assert again.exit_code == 0, again.output
    assert tables.row_counts(SOURCE)["races"] == 4 and tables.row_counts(SOURCE)["horses"] == 2


@pytest.mark.parametrize("option", [["--workers", "0"], ["--limit", "0"]])
def test_build_rejects_a_non_positive_workers_or_limit(raw_root, option):
    result = run("build", "-s", SOURCE, *option)
    assert result.exit_code != 0
    assert not tables.row_counts(SOURCE)


def test_build_without_a_manifest_is_one_error_line(raw_root):
    paths.manifest_path(SOURCE).unlink()
    result = run("build", "-s", SOURCE)
    assert result.exit_code == 1
    assert len(lines(result)) == 1
    assert lines(result)[0].startswith("ERROR store build failed: FileNotFoundError")
    assert "Traceback" not in result.output


def test_build_whose_checks_fail_is_one_error_line_and_exit_1(data, monkeypatch):
    register(monkeypatch, FakeStoreAdapter())
    ghost = copy.deepcopy(SECTIONAL_2026)
    for row in ghost["sectionals"]:
        row["race_id"] = f"{SOURCE}:2031-01-01:ST:01"  # a race no results page has
    populate([p._replace(tables=ghost) if p.page_type == "sectional" else p for p in PAGES])
    result = run("build", "-s", SOURCE, "--workers", "1")
    assert result.exit_code == 1
    assert len(errors(result)) == 1 and "orphan race ids in sectionals 1 of 1" in errors(result)[0]
    assert lines(result)[-1] == errors(result)[0]
    assert len(lines(result)) <= 20
    assert tables.row_counts(SOURCE)["races"] == 4  # the tables were still written


# ---------------------------------------------------------------- check


def test_check_passes_and_is_short(built_root):
    result = run("check", "-s", SOURCE)
    assert result.exit_code == 0, result.output
    assert len(lines(result)) <= 15 and lines(result)[0] == f"store check {SOURCE}: OK"
    assert not errors(result)
    assert "race dates 2010-01-01 .. 2026-09-16" in lines(result)
    assert (paths.reports_dir() / f"store-check-{SOURCE}.md").is_file()


def test_check_exits_non_zero_with_one_error_line_on_a_duplicate(built_root):
    runners = tables.read_table(SOURCE, "runners")
    doubled = pd.concat([runners, runners.head(1)], ignore_index=True)
    pq.write_table(
        pa.Table.from_pandas(doubled, schema=build.arrow_schema("runners"), preserve_index=False),
        tables.table_path(SOURCE, "runners"),
    )
    result = run("check", "-s", SOURCE)
    assert result.exit_code == 1
    assert len(errors(result)) == 1 and "unique (race_id, horse_id) 1 of 13" in errors(result)[0]
    assert len(lines(result)) <= 15
    assert lines(result)[0] == f"store check {SOURCE}: FAILED"


def test_check_before_any_build_is_one_error_line(raw_root):
    result = run("check", "-s", SOURCE)
    assert result.exit_code == 1 and len(lines(result)) == 1
    assert lines(result)[0].startswith("ERROR store check")


# ---------------------------------------------------------------- info


def test_info_lists_row_counts_and_the_date_range(built_root):
    result = run("info", "-s", SOURCE)
    assert result.exit_code == 0, result.output
    out = lines(result)
    assert len(out) <= 15
    assert out[0].startswith(f"store {SOURCE}: ") and str(paths.processed_dir(SOURCE)) in out[0]
    # row counts read off the Parquet footers, not through the code under test
    on_disk = {
        p.stem: pq.read_metadata(p).num_rows for p in paths.processed_dir(SOURCE).glob("*.parquet")
    }
    assert on_disk["runners"] == 12 and on_disk["races"] == 4
    for table, n in on_disk.items():
        assert any(line.split() == [table, str(n)] for line in out), (table, n)
    assert "race dates 2010-01-01 .. 2026-09-16" in out
    assert out[-1].startswith("snapshot ")
    assert out[-1].split()[1] == build_record()["snapshot_id"][:12]


def test_info_shows_what_is_on_disk_now(built_root):
    first = run("info", "-s", SOURCE)
    pq.write_table(  # shrink one table: info must follow the files, not a cached summary
        pa.Table.from_pandas(
            tables.read_table(SOURCE, "runners").head(5),
            schema=build.arrow_schema("runners"),
            preserve_index=False,
        ),
        tables.table_path(SOURCE, "runners"),
    )
    second = run("info", "-s", SOURCE)
    assert any(line.split() == ["runners", "12"] for line in lines(first))
    assert any(line.split() == ["runners", "5"] for line in lines(second))


def test_info_before_any_build_is_one_error_line(raw_root):
    result = run("info", "-s", SOURCE)
    assert result.exit_code == 1 and len(lines(result)) == 1
    assert lines(result)[0].startswith("ERROR no tables under")


# ---------------------------------------------------------------- the source option and mounting


def test_source_defaults_to_the_only_registered_one(built_root, monkeypatch):
    _only_test_source(monkeypatch)
    for command in ("check", "info"):
        result = run(command)
        assert result.exit_code == 0, result.output
        assert SOURCE in lines(result)[0]


def test_source_is_required_when_several_are_registered(built_root, monkeypatch):
    assert len(adapters.ADAPTERS) > 1
    for command in ("build", "check", "info"):
        result = run(command)
        assert result.exit_code == 1
        assert len(lines(result)) == 1 and lines(result)[0].startswith("ERROR")
        assert "--source" in lines(result)[0]


@pytest.mark.parametrize("command", ["build", "check", "info"])
def test_an_unknown_source_is_one_error_line(data, command):
    result = run(command, "-s", "nosuchsource")
    assert result.exit_code == 1
    assert len(lines(result)) == 1 and lines(result)[0].startswith("ERROR")
    assert "nosuchsource" in lines(result)[0]
    assert "Traceback" not in result.output


def test_source_without_a_config_is_one_error_line(built_root, monkeypatch):
    def no_config(source: str):
        raise ValueError(f"no config for source {source!r}")

    monkeypatch.setattr(cli_module, "source_config", no_config)
    result = run("check", "-s", SOURCE)
    assert result.exit_code == 1
    assert len(lines(result)) == 1 and lines(result)[0].startswith("ERROR")


def test_real_sources_resolve_to_their_country_config():
    # the seam the other tests replace, unreplaced: the shipped sources map to their YAML
    assert cli_module.source_config("hkjc") == load_source("hongkong", "hkjc")
    assert cli_module.source_config("jra") == load_source("japan", "jra")


def test_the_store_group_is_mounted_on_the_root_cli():
    result = runner.invoke(app, ["store", "--help"])
    assert result.exit_code == 0
    for command in ("build", "check", "info"):
        assert command in result.output
