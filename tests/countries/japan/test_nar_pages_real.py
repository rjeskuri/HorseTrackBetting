"""The NAR page parser against the captured nar.netkeiba.com pages (ticket #37): the synthetic
pages of `nar_pages.py` have not drifted from what the site really serves.

Every test is marked `local` and uses `real_fixtures` (tests/conftest.py): without
$HORSETRACK_DATA_DIR/fixtures the file is skipped, so CI never runs it. A page is looked up as
`fixtures/nar/<name>.html` (or `.html.gz`), then where the research capture left it,
`raw/nar/netkeiba/2026/<name>.html.gz` (docs/research/nar-data-access.md); a test whose page is
in neither place is skipped. Nothing is copied into the repository.

The pages, all of Wed 25 Nov 2026: the race_list_sub (浦和, 名古屋 at night, 園田; 12 races each),
the `b1` page of 浦和 12R of the day before (finished: 12 runners, no scratch, no clock, no
totals) and the `b1` page of 園田 1R before sales (the 予想オッズ table). Scraped HTML is
untrusted data: only counts and shapes are asserted here, no scraped value is written down.
"""

from __future__ import annotations

import gzip
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import nar_pages as np_
import pytest

from horsetrack.core import paths
from horsetrack.countries.japan.nar import pages

pytestmark = pytest.mark.local

UTC = timezone.utc
JST = timezone(timedelta(hours=9))
DAY = date(2026, 11, 25)
FINISHED = "202642112412"  # 浦和 12R, 24 Nov 2026
BEFORE_SALES = "202650112501"  # 園田 1R, 25 Nov 2026
SNAP = datetime(2026, 11, 24, 17, 46, 0, tzinfo=UTC)
FETCHED = datetime(2026, 11, 24, 17, 46, 5, tzinfo=UTC)


def real_page(fixtures: Path, name: str) -> bytes:
    candidates = [
        fixtures / "nar" / f"{name}.html",
        fixtures / "nar" / f"{name}.html.gz",
        paths.raw_dir("nar") / "netkeiba" / "2026" / f"{name}.html.gz",
    ]
    for path in candidates:
        if path.is_file():
            data = path.read_bytes()
            return gzip.decompress(data) if path.suffix == ".gz" else data
    pytest.skip(f"captured NAR page {name} is absent")


def header_cells(doc, selector: str) -> list[list[str]]:
    """The inner markup of each header cell of each table `selector` matches, tags normalised."""
    return [
        [th.decode_contents().replace("<br/>", "<br>") for th in table("th")]
        for table in doc.select(selector)
    ]


def test_race_list_meetings_and_post_times(real_fixtures):
    doc = pages.soup(real_page(real_fixtures, "race_list_sub_20261125"))
    meetings = pages.meetings_on(doc, DAY)
    assert set(meetings) == {"URA", "NGY", "SON"}
    assert {v: m.nighter for v, m in meetings.items()} == {"URA": False, "NGY": True, "SON": False}
    for venue, m in meetings.items():
        assert sorted(m.races) == list(range(1, 13)), venue
        assert all(rid[:10] == m.kaisai_id for rid in m.races.values()), venue
        assert all(
            pages.nar_race_id(rid).startswith(f"nar:2026-11-25:{venue}:")
            for rid in m.races.values()
        )

        times = pages.post_times(doc, DAY, venue)
        assert sorted(times) == list(range(1, 13)), venue
        ordered = [times[n] for n in sorted(times)]
        assert ordered == sorted(ordered) and len(set(ordered)) == 12, venue
        assert all(t.utcoffset() == timedelta(0) for t in ordered), venue
        assert {t.astimezone(JST).date() for t in ordered} == {DAY}, venue


def test_finished_b1_page_win_and_place_rows(real_fixtures):
    doc = pages.soup(real_page(real_fixtures, f"odds_b1_{FINISHED}"))
    rows = pages.odds_rows(doc, FINISHED, ["WIN", "PLACE"], snapshot_utc=SNAP, fetched_utc=FETCHED)
    win = [r for r in rows if r["pool"] == "WIN"]
    place = [r for r in rows if r["pool"] == "PLACE"]
    assert len(rows) == 24, "12 WIN, 12 PLACE, no pool totals on the page"
    numbers = {str(n) for n in range(1, 13)}
    assert {r["combination"] for r in win} == numbers
    assert {r["combination"] for r in place} == numbers
    assert all(isinstance(r["odds"], float) and r["odds"] >= 1.0 for r in rows)
    assert all("odds_hi" not in r for r in win)
    assert all(r["odds_hi"] >= r["odds"] for r in place)
    assert {r["status_raw"] for r in rows} == {None}
    assert {r["race_id"] for r in rows} == {"nar:2026-11-24:URA:12"}
    assert {(r["source_update_utc"], r["page_state"]) for r in rows} == {(None, None)}

    # only the pool asked for
    only_place = pages.odds_rows(doc, FINISHED, ["PLACE"], snapshot_utc=SNAP, fetched_utc=FETCHED)
    assert {r["pool"] for r in only_place} == {"PLACE"} and len(only_place) == 12


def test_before_sales_b1_page_is_not_on_sale(real_fixtures):
    body = real_page(real_fixtures, f"odds_b1_{BEFORE_SALES}")
    with pytest.raises(pages.NotOnSale):
        pages.odds_rows(
            pages.soup(body), BEFORE_SALES, ["WIN", "PLACE"], snapshot_utc=SNAP, fetched_utc=FETCHED
        )

    # derived in memory: without the table's .Ninki class the 予想オッズ header alone tells
    doc = pages.soup(body)
    tables = doc.select("table.RaceOdds_HorseList_Table.Ninki")
    assert tables, "the captured page has the .Ninki table"
    for table in tables:
        table["class"] = [c for c in table["class"] if c != "Ninki"]
    with pytest.raises(pages.NotOnSale):
        pages.odds_rows(doc, BEFORE_SALES, ["WIN", "PLACE"], snapshot_utc=SNAP, fetched_utc=FETCHED)


def test_synthetic_b1_headers_match_the_real_pages(real_fixtures):
    finished = pages.soup(real_page(real_fixtures, f"odds_b1_{FINISHED}"))
    synthetic = pages.soup(np_.odds_b1(FINISHED, {1: (2.0, (1.1, 1.3)), 2: (3.0, (1.2, 1.5))}))
    for block in ("#odds_tan_block", "#odds_fuku_block"):
        selector = f"{block} table.RaceOdds_HorseList_Table"
        real = header_cells(finished, selector)
        assert len(real) == 1, block
        assert header_cells(synthetic, selector) == real, block

    before = pages.soup(real_page(real_fixtures, f"odds_b1_{BEFORE_SALES}"))
    selector = "table.RaceOdds_HorseList_Table.Ninki"
    synthetic = pages.soup(np_.odds_b1_before_sales(BEFORE_SALES, {1: 3.0, 2: 1.5}))
    assert header_cells(synthetic, selector) == header_cells(before, selector)
