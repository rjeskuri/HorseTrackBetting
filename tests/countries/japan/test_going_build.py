"""Ticket #253: the JRA store build fills `races.rail_setting` from the cached going pages.

Contract pinned here (the coder follows it):

- Going pages (馬場情報) are raw-cache pages of the JRA source with page_type `going` (any key; the
  page itself holds its date and venue): `build_tables("jra")` reads every such page that is in the
  manifest, parses it with `jra.going.parse_going_page` (`fetched_utc` = the manifest's) and fills
  `races.rail_setting` with `jra.going.attach_rail_setting` (no `decision_utc`: the store holds
  every race with the latest rail printed for its meeting day). A `going` page is consumed, so it
  is never in `summary.skipped_page_types`. A turf race of a (venue, race_date) with a rail gets
  'A'-'D'; a dirt or jump race, a day with no page, a page printing no rail, and a day only
  another venue or another day has a page for, stay null. A page that cannot be parsed (an error
  page) does not stop the build or touch other days. Going pages are applied when the tables are
  assembled, so a page added after an earlier build shows up on the next one.

Pages and results are synthetic; expected rails are written out here.
"""

from __future__ import annotations

from datetime import date, datetime

import going_pages as gp
import netkeiba_results_pages as rp
import pyarrow.parquet as pq
import pytest

from horsetrack.core import paths
from horsetrack.core.store import build

SOURCE = "jra"
VAR = "HORSETRACK_DATA_DIR"
RESULTS_FETCHED = "2026-11-23T04:07:12+00:00"

# race id -> (results page, expected store surface, expected rail)
PLAIN = rp.PLAIN  # turf, Kyoto, 2026-11-22
SCRATCHED = rp.SCRATCHED  # turf, Tokyo, 2026-10-11: only the next day has a page
DEAD_HEAT = rp.DEAD_HEAT  # turf, Niigata, 2026-08-30: only another venue has a page
DEMOTED = rp.DEMOTED  # dirt, Hanshin, 2026-06-14: the page prints a turf rail
JUMP = rp.JUMP  # jump, Fukushima, 2026-07-05: the page prints a turf rail
RESULTS = (PLAIN, SCRATCHED, DEAD_HEAT, DEMOTED, JUMP)
RACE_IDS = {
    "jra:2026-11-22:KYO:11": "B",
    "jra:2026-10-11:TOK:05": None,
    "jra:2026-08-30:NII:10": None,
    "jra:2026-06-14:HSN:07": None,
    "jra:2026-07-05:FKS:04": None,
}


def jst(*parts: int) -> datetime:
    return datetime(*parts)


@pytest.fixture
def data(tmp_path, monkeypatch):
    d = (tmp_path / "data").resolve()
    monkeypatch.setenv(VAR, str(d))
    return d


def add_results(*pages: rp.ResultPage) -> None:
    for p in pages:
        gp.add_raw(SOURCE, "results", p.key, p.race_date.year, rp.render(p), RESULTS_FETCHED)


def add_going(key, race_date, meeting, published_jst, rail, fetched_utc) -> None:
    body = gp.going_page(race_date, meeting, published_jst, rail)
    gp.add_raw(SOURCE, "going", key, race_date.year, body, fetched_utc)


def add_the_going_pages() -> None:
    # Kyoto 2026-11-22: three publications, added out of order. A (day before), B (race morning),
    # then a later page printing － that must not erase B.
    add_going("k3", date(2026, 11, 22), "5回京都6日", jst(2026, 11, 22, 11, 0), "－",
              "2026-11-22T02:05:00+00:00")  # fmt: skip
    add_going("k1", date(2026, 11, 22), "5回京都6日", jst(2026, 11, 21, 15, 0), "Aコース",
              "2026-11-21T06:10:00+00:00")  # fmt: skip
    add_going("k2", date(2026, 11, 22), "5回京都6日", jst(2026, 11, 22, 8, 30), "Ｂコース",
              "2026-11-21T23:40:00+00:00")  # fmt: skip
    # Tokyo: the day after the race; Niigata: another venue; Hanshin, Fukushima: dirt and jump
    add_going("t1", date(2026, 10, 12), "4回東京3日", jst(2026, 10, 11, 15, 0), "Dコース",
              "2026-10-11T06:10:00+00:00")  # fmt: skip
    add_going("n1", date(2026, 8, 30), "2回函館8日", jst(2026, 8, 29, 15, 0), "Aコース",
              "2026-08-29T06:10:00+00:00")  # fmt: skip
    add_going("h1", date(2026, 6, 14), "3回阪神2日", jst(2026, 6, 13, 15, 0), "Cコース",
              "2026-06-13T06:10:00+00:00")  # fmt: skip
    add_going("f1", date(2026, 7, 5), "2回福島2日", jst(2026, 7, 4, 15, 0), "Aコース",
              "2026-07-04T06:10:00+00:00")  # fmt: skip


def rails() -> dict[str, str | None]:
    races = pq.read_table(paths.processed_dir(SOURCE) / "races.parquet").to_pandas()
    assert "rail_setting" in races.columns
    return {
        r: (None if v is None or v != v else v)
        for r, v in zip(races["race_id"], races["rail_setting"], strict=True)
    }


# ---------------------------------------------------------------- the build


def test_build_fills_rail_setting_on_turf_races_from_the_going_pages(data):
    add_results(*RESULTS)
    add_the_going_pages()

    summary = build.build_tables(SOURCE, workers=1)

    assert "going" not in summary.skipped_page_types
    assert summary.failed == [] and summary.unreadable == []
    races = pq.read_table(paths.processed_dir(SOURCE) / "races.parquet").to_pandas()
    surfaces = dict(zip(races["race_id"], races["surface"], strict=True))
    assert surfaces["jra:2026-11-22:KYO:11"] == "turf"  # the cases mean what the comments say
    assert surfaces["jra:2026-06-14:HSN:07"] == "dirt"
    assert surfaces["jra:2026-07-05:FKS:04"] == "jump"
    assert rails() == RACE_IDS


@pytest.mark.parametrize(
    ("rail_printed", "expected"),
    [("Aコース", "A"), ("Ｂコース", "B"), ("Cコース", "C"), ("Dコース", "D")],
)
def test_each_course_letter_reaches_the_turf_race(data, rail_printed, expected):
    add_results(PLAIN)
    add_going("g", PLAIN.race_date, "5回京都6日", jst(2026, 11, 21, 15, 0), rail_printed,
              "2026-11-21T06:10:00+00:00")  # fmt: skip

    build.build_tables(SOURCE, workers=1)

    assert rails() == {"jra:2026-11-22:KYO:11": expected}


def test_the_later_publication_wins_whatever_order_the_pages_are_keyed_in(data):
    add_results(PLAIN)
    add_going("a", PLAIN.race_date, "5回京都6日", jst(2026, 11, 22, 8, 30), "Cコース",
              "2026-11-21T23:40:00+00:00")  # fmt: skip
    add_going("z", PLAIN.race_date, "5回京都6日", jst(2026, 11, 21, 15, 0), "Aコース",
              "2026-11-21T06:10:00+00:00")  # fmt: skip

    build.build_tables(SOURCE, workers=1)

    assert rails() == {"jra:2026-11-22:KYO:11": "C"}


def test_a_going_page_added_after_a_build_shows_on_the_next_build(data):
    add_results(PLAIN, DEMOTED)
    build.build_tables(SOURCE, workers=1)
    assert set(rails().values()) == {None}

    add_going("g", PLAIN.race_date, "5回京都6日", jst(2026, 11, 21, 15, 0), "Dコース",
              "2026-11-21T06:10:00+00:00")  # fmt: skip
    build.build_tables(SOURCE, workers=1)

    assert rails() == {"jra:2026-11-22:KYO:11": "D", "jra:2026-06-14:HSN:07": None}


def test_an_unparseable_going_page_does_not_stop_the_build(data):
    add_results(PLAIN, SCRATCHED)
    add_going("good", PLAIN.race_date, "5回京都6日", jst(2026, 11, 21, 15, 0), "Bコース",
              "2026-11-21T06:10:00+00:00")  # fmt: skip
    gp.add_raw(SOURCE, "going", "broken", 2026, "<html><body>エラー</body></html>".encode("cp932"),
               "2026-10-11T06:10:00+00:00")  # fmt: skip

    summary = build.build_tables(SOURCE, workers=1)

    assert summary.rows["races"] == 2
    assert rails() == {"jra:2026-11-22:KYO:11": "B", "jra:2026-10-11:TOK:05": None}
