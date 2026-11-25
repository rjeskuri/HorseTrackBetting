"""Ticket #208: the JRA store build parses pages (`horsetrack store build --source jra`).

`core.store.parse_worker.parser_version("jra", page_type)` found no parser, because the JRA parsers
live in `countries/japan/jra/netkeiba/` and `JRAAdapter` had no `parser_version` hook, so the build
skipped every page type. Contract pinned here:

- `parse_worker.parser_version("jra", t)` is `str(PARSER_VERSION)` of the netkeiba module
  `parse_<t>` for fixture, results and pedigree_card, and None for a page type without a parser
  (never an error).
- A store build over a synthetic raw cache (the real `JRAAdapter`, synthetic netkeiba pages) selects
  and parses fixture, results and pedigree_card pages: none is in `skipped_page_types`,
  `parser_versions` carries each one's version, and the results and pedigree rows reach the tables.
"""

from __future__ import annotations

import gzip
import hashlib
import sqlite3
from datetime import date
from importlib import import_module

import netkeiba_pages as pages
import netkeiba_results_pages as results_pages
import pandas as pd
import pytest

from horsetrack.core import paths

# isort: split
from horsetrack.core.store import build, parse_worker

VAR = "HORSETRACK_DATA_DIR"
PARSERS = "horsetrack.countries.japan.jra.netkeiba"
SOURCE = "jra"
FETCHED_UTC = "2026-11-23T04:07:12+00:00"
STORE_PAGE_TYPES = ("fixture", "results", "pedigree_card")

# the manifest as core.ingest.queue defines it (columns the build reads)
MANIFEST_SCHEMA = """
CREATE TABLE IF NOT EXISTS fetch_queue (
    jurisdiction TEXT NOT NULL, page_type TEXT NOT NULL, key TEXT NOT NULL, url TEXT NOT NULL,
    year INTEGER NOT NULL, priority INTEGER NOT NULL DEFAULT 0, method TEXT NOT NULL DEFAULT 'GET',
    body TEXT, mutable INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'pending',
    http_status INTEGER, final_url TEXT, sha256 TEXT, bytes INTEGER, fetched_utc TEXT,
    attempts INTEGER NOT NULL DEFAULT 0, error TEXT, parser_version TEXT,
    PRIMARY KEY (page_type, key)
)
"""

RESULTS = results_pages.PLAIN  # 2026-11-22, Kyoto, race 11, key 202608050611
RACE_ID = "jra:2026-11-22:KYO:11"
CARD_RUNNERS = [
    pages.Runner("2022101101", "シンセサイア", "シンセダム", pages.bracketed("シンセ")),
    pages.Runner("2022101102", "ミドリサイア", "ミドリダム", pages.bracketed("ミドリ")),
    pages.Runner("2022101103", "アオサイア", "アオダム", pages.bracketed("アオ")),
]


@pytest.fixture
def data(tmp_path, monkeypatch):
    d = (tmp_path / "data").resolve()
    monkeypatch.setenv(VAR, str(d))
    return d


def add_page(page_type: str, key: str, year: int, body: bytes) -> None:
    path = paths.raw_page_path(SOURCE, page_type, year, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(body, mtime=0))
    manifest = paths.manifest_path(SOURCE)
    url = f"https://synthetic.example.test/{page_type}/{key}"
    conn = sqlite3.connect(manifest)
    with conn:
        conn.execute(MANIFEST_SCHEMA)
        conn.execute(
            "INSERT OR REPLACE INTO fetch_queue (jurisdiction, page_type, key, url, year, status,"
            " final_url, sha256, bytes, fetched_utc, http_status, attempts)"
            " VALUES (?, ?, ?, ?, ?, 'fetched', ?, ?, ?, ?, 200, 1)",
            (SOURCE, page_type, key, url, year, url, hashlib.sha256(body).hexdigest(), len(body),
             FETCHED_UTC),
        )  # fmt: skip
    conn.close()


@pytest.fixture
def raw_cache(data):
    add_page("fixture", "2026-11", 2026, pages.calendar(2026, 11, [1, 7, 8, 14, 15, 21, 22]))
    add_page("results", RESULTS.key, 2026, results_pages.render(RESULTS))
    add_page(
        "pedigree_card",
        RESULTS.key,
        2026,
        pages.pedigree_card(CARD_RUNNERS, race_date=RESULTS.race_date, venue_name="京都"),
    )
    return data


# ---------------------------------------------------------------- the version hook


@pytest.mark.parametrize("page_type", STORE_PAGE_TYPES)
def test_parser_version_is_the_netkeiba_parsers_version(page_type):
    module = import_module(f"{PARSERS}.parse_{page_type}")
    version = parse_worker.parser_version(SOURCE, page_type)
    assert isinstance(version, str)
    assert version == str(module.PARSER_VERSION)


def test_parser_versions_follow_each_parser_not_one_constant():
    # the three parsers do not share a version today: a hook returning one value cannot pass
    expected = {
        t: str(import_module(f"{PARSERS}.parse_{t}").PARSER_VERSION) for t in STORE_PAGE_TYPES
    }
    got = {t: parse_worker.parser_version(SOURCE, t) for t in STORE_PAGE_TYPES}
    assert got == expected
    assert len(set(expected.values())) > 1


@pytest.mark.parametrize("page_type", ["horse", "sectional", "corunning", "no_such_page", "x y"])
def test_parser_version_of_a_page_type_without_a_parser_is_none(page_type):
    assert parse_worker.parser_version(SOURCE, page_type) is None


# ---------------------------------------------------------------- the build


def test_jra_build_selects_and_parses_every_page_type(raw_cache):
    summary = build.build_tables(SOURCE, workers=1)

    assert summary.skipped_page_types == {}
    assert summary.n_selected == 3
    assert summary.n_parsed == 3
    assert summary.failed == [] and summary.unreadable == []
    assert summary.parser_versions == {
        t: str(import_module(f"{PARSERS}.parse_{t}").PARSER_VERSION) for t in STORE_PAGE_TYPES
    }


def test_jra_build_puts_the_parsed_rows_in_the_tables(raw_cache):
    summary = build.build_tables(SOURCE, workers=1)

    n_runners = len(RESULTS.table.strip().splitlines())
    assert summary.rows["races"] == 1
    assert summary.rows["runners"] == n_runners
    assert summary.rows["horse_pedigree"] == len(CARD_RUNNERS)
    assert summary.rows["dividends"] > 0
    assert summary.date_min == summary.date_max == RESULTS.race_date.isoformat()

    processed = paths.processed_dir(SOURCE)
    races = pd.read_parquet(processed / "races.parquet")
    assert list(races.index if "race_id" not in races else races["race_id"]) == [RACE_ID]
    pedigree = pd.read_parquet(processed / "horse_pedigree.parquet")
    pedigree = pedigree.reset_index() if "horse_id" not in pedigree else pedigree
    assert sorted(pedigree["horse_id"]) == sorted(r.horse_id for r in CARD_RUNNERS)
    sires = dict(zip(pedigree["horse_id"], pedigree["sire_name"], strict=True))
    assert sires == {r.horse_id: r.sire for r in CARD_RUNNERS}


@pytest.mark.parametrize("missing", STORE_PAGE_TYPES)
def test_jra_build_parses_each_page_type_on_its_own(data, missing):
    # a cache holding one page type is parsed, not skipped: each type is wired separately
    add_page_for = {
        "fixture": lambda: add_page("fixture", "2026-11", 2026, pages.calendar(2026, 11, [7, 8])),
        "results": lambda: add_page("results", RESULTS.key, 2026, results_pages.render(RESULTS)),
        "pedigree_card": lambda: add_page(
            "pedigree_card",
            RESULTS.key,
            2026,
            pages.pedigree_card(CARD_RUNNERS, race_date=date(2026, 11, 22)),
        ),
    }
    add_page_for[missing]()

    summary = build.build_tables(SOURCE, workers=1)

    assert summary.skipped_page_types == {}
    assert summary.n_parsed == 1
    assert list(summary.parser_versions) == [missing]
