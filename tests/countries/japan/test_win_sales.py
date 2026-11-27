"""Per-race Win sales (単勝 発売票数) off a JRA 単勝・複勝 odds page, with their publication time
(ticket #278, for E4's low-turnover segment, 08 §4).

Contract pinned here (the coder follows it):

- `horsetrack.countries.japan.jra.win_sales` is pure (no network) and builds on
  `jra.official_pages` (`soup`, `page_meeting`, `race_number`, `source_update_utc`).
- `parse_win_sales(body: bytes, *, fetched_utc: datetime) -> dict | None`: one record for the race
  the page is for, with the keys of RECORD_KEYS:
  - `race_id` `jra:<YYYY-MM-DD>:<VENUE>:<NN>`, `race_date` (a `datetime.date`), `venue` (the
    jra.yaml code), `race_no` (int), all from the page's own date line and race-number image;
  - `win_tickets`: the 単勝 row of the page's 発売票数 block (`#votes`), comma thousands; a 単勝
    row anywhere else on the page is never read;
  - `win_sales_yen` = `win_tickets` x jra.yaml's unit stake (100 yen), an int;
  - `is_final`: the refresh line says 最終オッズ;
  - `published_utc` (aware, UTC): the page's own clock (`12時10分現在`) when it prints one, dated as
    `official_pages.source_update_utc` does (the Tokyo date of the fetch, or the day before when
    that would be after the fetch), so a live figure is never published after its fetch; a page
    without a clock (最終オッズ) shows a final figure, so its time is the post time the page prints
    (発走時刻 on the date line, Tokyo, race date) plus 30 minutes, whatever the fetch time (the
    result tables' convention, conservative; the owner's answer on #278), or `fetched_utc` when it
    prints no post time;
  - `fetched_utc`: the caller's, aware UTC.
  - None (no record) when the page has no 単勝 figure in its votes block: no block, a block from
    another pool (馬連), or a 単勝 cell that is not a whole number.
  - ValueError for a naive `fetched_utc`, and for a page that does not say which race it is (no
    date line, a venue not in jra.yaml, no race number: an error page).

Pages are synthetic: four files under tests/fixtures/synthetic/ (stored UTF-8, encoded cp932 here
as the site sends them) and `win_sales_pages.sales_page`, which builds the same structure from
values. Expected values are written out here, never computed by the code under test.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import jradb_pages as jp
import pytest
import win_sales_pages as wsp

from horsetrack.countries.japan.jra import win_sales

UTC = timezone.utc
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "synthetic"
UNIT_STAKE = 100  # yen per ticket (countries/japan/config/jra.yaml unit_stake)
RECORD_KEYS = {
    "race_id", "race_date", "venue", "race_no", "win_tickets", "win_sales_yen", "is_final",
    "published_utc", "fetched_utc",
}  # fmt: skip
FINAL = "最終オッズ"


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_text(encoding="utf-8").encode("cp932")


def assert_utc(value, expected: datetime, what: str) -> None:
    assert value is not None, what
    assert value.utcoffset() == timedelta(0), f"{what} is not UTC: {value!r}"
    assert value == expected, what


def assert_record(rec, *, race_id, race_date, venue, race_no, tickets, final, published, fetched):
    assert rec is not None
    assert set(rec) >= RECORD_KEYS, f"missing keys {sorted(RECORD_KEYS - set(rec))}"
    assert rec["race_id"] == race_id
    assert isinstance(rec["race_date"], date) and not isinstance(rec["race_date"], datetime)
    assert rec["race_date"] == race_date
    assert rec["venue"] == venue
    assert rec["race_no"] == race_no
    assert rec["win_tickets"] == tickets
    assert rec["win_sales_yen"] == tickets * UNIT_STAKE
    assert int(rec["win_sales_yen"]) == rec["win_sales_yen"]
    assert bool(rec["is_final"]) is final
    assert_utc(rec["published_utc"], published, "published_utc")
    assert_utc(rec["fetched_utc"], fetched, "fetched_utc")


# ---------------------------------------------------------------- broad: a spread of pages

# (id, race date, meeting, venue, race no, refresh line, 発走時刻, votes, fetched, tickets,
#  final, published). Clocks are Tokyo (UTC+9).
CASES = [
    # before the post: the page's clock, on the fetch's Tokyo date
    ("live-tokyo", date(2026, 11, 28), "5回東京7日", "TOK", 5, "12時10分現在", "14時25分",
     [("単勝", "12,345"), ("複勝", "23,456"), ("総票数合計", "345,678")],
     utc(2026, 11, 28, 3, 12, 40), 12_345, False, utc(2026, 11, 28, 3, 10)),
    # after the close: no clock, the post printed on the page (15:40 JST = 06:40 UTC) + 30 min;
    # 単勝 not the first row
    ("final-kyoto", date(2026, 11, 29), "5回京都9日", "KYO", 11, FINAL, "15時40分",
     [("複勝", "2,222,222"), ("単勝", "1,234,567"), ("総票数合計", "9,876,543")],
     utc(2026, 11, 29, 8, 0), 1_234_567, True, utc(2026, 11, 29, 7, 10)),
    # an archive page of the price era, fetched years later: dated by its race, not the fetch
    # (post 10:05 JST = 01:05 UTC, + 30 min)
    ("archive-nakayama-2015", date(2015, 1, 5), "1回中山1日", "NAK", 1, FINAL, "10時05分",
     [("単勝", "8,808")], utc(2026, 11, 20, 10, 0), 8_808, True, utc(2015, 1, 5, 1, 35)),
    # post 9:50 JST: + 30 min crosses the hour (00:50 UTC -> 01:20 UTC)
    ("final-fukushima-early", date(2026, 4, 12), "1回福島6日", "FKS", 1, FINAL, "9時50分",
     [("単勝", "2,718")], utc(2026, 4, 12, 3, 0), 2_718, True, utc(2026, 4, 12, 1, 20)),
    # sold the day before the race (前日発売): the clock is on the fetch's date, not the race's
    ("day-before-kokura", date(2026, 8, 2), "2回小倉8日", "KOK", 12, "17時30分現在", "16時30分",
     [("単勝", "7"), ("複勝", "5")], utc(2026, 8, 1, 8, 31), 7, False, utc(2026, 8, 1, 8, 30)),
    # final, but no post time printed: the fetch time, never earlier
    ("final-no-post-hanshin", date(2026, 6, 14), "3回阪神2日", "HSN", 7, FINAL, None,
     [("単勝", "400,000")], utc(2026, 6, 14, 7, 45, 10), 400_000, True,
     utc(2026, 6, 14, 7, 45, 10)),
    ("live-sapporo-morning", date(2026, 7, 25), "1回札幌1日", "SAP", 3, "9時05分現在", "11時15分",
     [("単勝", "31,416")], utc(2026, 7, 25, 0, 5, 59), 31_416, False, utc(2026, 7, 25, 0, 5)),
]  # fmt: skip


@pytest.mark.parametrize(
    ("day", "meeting", "venue", "race_no", "state", "post", "votes", "fetched", "tickets",
     "final", "published"),
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)  # fmt: skip
def test_a_win_place_page_gives_its_race_win_sales_and_publication_time(
    day, meeting, venue, race_no, state, post, votes, fetched, tickets, final, published
):
    body = wsp.sales_page(day, meeting, race_no, state=state, post=post, votes=votes)

    rec = win_sales.parse_win_sales(body, fetched_utc=fetched)

    assert_record(
        rec,
        race_id=f"jra:{day.isoformat()}:{venue}:{race_no:02d}",
        race_date=day,
        venue=venue,
        race_no=race_no,
        tickets=tickets,
        final=final,
        published=published,
        fetched=fetched,
    )


def test_the_committed_pages_parse_to_the_values_they_print():
    live = win_sales.parse_win_sales(
        fixture("jra_win_sales_live.html"), fetched_utc=utc(2026, 11, 28, 3, 12, 40)
    )
    assert_record(live, race_id="jra:2026-11-28:TOK:05", race_date=date(2026, 11, 28),
                  venue="TOK", race_no=5, tickets=12_345, final=False,
                  published=utc(2026, 11, 28, 3, 10),
                  fetched=utc(2026, 11, 28, 3, 12, 40))  # fmt: skip
    final = win_sales.parse_win_sales(
        fixture("jra_win_sales_final.html"), fetched_utc=utc(2026, 11, 30, 1, 0)
    )
    # the page also prints a decoy 単勝 row (999,999) outside its votes block
    # final, post 15:40 JST (06:40 UTC): published at post + 30 min
    assert_record(final, race_id="jra:2026-11-29:KYO:11", race_date=date(2026, 11, 29),
                  venue="KYO", race_no=11, tickets=1_234_567, final=True,
                  published=utc(2026, 11, 29, 7, 10), fetched=utc(2026, 11, 30, 1, 0))  # fmt: skip
    for name in ("jra_win_sales_no_votes.html", "jra_win_sales_quinella.html"):
        assert win_sales.parse_win_sales(fixture(name), fetched_utc=utc(2026, 11, 28, 3)) is None


@pytest.mark.parametrize(
    ("fetched", "fetched_final"),
    [
        (utc(2026, 11, 28, 3, 10), utc(2026, 11, 28, 5, 30)),  # final seen 5 min after the post
        (utc(2026, 11, 28, 5, 24, 59), utc(2026, 11, 29, 2, 0)),  # final seen the next day
    ],
)
def test_a_live_figure_is_never_published_after_its_fetch_nor_a_final_one_before_post_plus_30(
    fetched, fetched_final
):
    """The same race seen live (clock 12時10分) and final: the live time is the clock whatever the
    fetch time; the final one is the post (14:25 JST = 05:25 UTC) + 30 min = 05:55 UTC, whether
    it was fetched before that or a day later."""
    day, meeting = date(2026, 11, 28), "5回東京7日"
    live = wsp.sales_page(day, meeting, 5, state="12時10分現在", post="14時25分",
                          votes=[("単勝", "12,345")])  # fmt: skip
    final = wsp.sales_page(day, meeting, 5, state=FINAL, post="14時25分",
                           votes=[("単勝", "60,002")])  # fmt: skip

    seen_live = win_sales.parse_win_sales(live, fetched_utc=fetched)
    seen_final = win_sales.parse_win_sales(final, fetched_utc=fetched_final)

    assert seen_live["published_utc"] == utc(2026, 11, 28, 3, 10) <= fetched
    assert seen_final["published_utc"] == utc(2026, 11, 28, 5, 55)
    assert seen_final["published_utc"] > seen_live["published_utc"]


# ---------------------------------------------------------------- no Win figure: no record

DAY, MEETING = date(2026, 11, 28), "5回東京7日"


@pytest.mark.parametrize(
    "votes",
    [
        None,  # no 発売票数 block at all
        [],  # an empty block
        [("複勝", "5,000"), ("総票数合計", "9,000")],  # no 単勝 row
        [("単勝", "－"), ("複勝", "5,000")],
        [("単勝", ""), ("複勝", "5,000")],
        [("単勝", "1,2a4")],
        [("単勝", "12.5")],
    ],
    ids=["no-block", "empty-block", "no-win-row", "dash", "blank", "junk", "decimal"],
)
def test_a_page_without_a_win_figure_in_its_votes_block_gives_no_record(votes):
    body = wsp.sales_page(DAY, MEETING, 5, state="12時10分現在", post="14時25分", votes=votes)

    assert win_sales.parse_win_sales(body, fetched_utc=utc(2026, 11, 28, 3, 12)) is None


def test_another_pools_page_gives_no_win_sales():
    for votes in ([("馬連", 877_724)], [("ワイド", 764_551)], [("3連単", 1_234)]):
        body = jp.pairs(DAY, MEETING, 5, {(1, 2): 32.1}, state="12時10分現在", votes=votes)
        assert win_sales.parse_win_sales(body, fetched_utc=utc(2026, 11, 28, 3, 12)) is None


@pytest.mark.parametrize("tickets", ["4,321", "88,000"])
def test_only_the_votes_block_counts_never_a_win_row_elsewhere(tickets):
    body = wsp.sales_page(DAY, MEETING, 5, state="12時10分現在", post="14時25分",
                          votes=[("単勝", tickets)], decoy="999,999")  # fmt: skip

    rec = win_sales.parse_win_sales(body, fetched_utc=utc(2026, 11, 28, 3, 12))

    assert rec["win_tickets"] == int(tickets.replace(",", ""))
    assert rec["win_sales_yen"] == int(tickets.replace(",", "")) * UNIT_STAKE


# ---------------------------------------------------------------- errors


@pytest.mark.parametrize(
    "body",
    [
        jp.no_header_page(),
        wsp.sales_page(None, MEETING, 5, state=FINAL, post="14時25分", votes=[("単勝", "100")]),
        wsp.sales_page(DAY, MEETING, None, state=FINAL, post="14時25分", votes=[("単勝", "100")]),
        wsp.sales_page(DAY, "5回大井7日", 5, state=FINAL, post="14時25分", votes=[("単勝", "100")]),
    ],
    ids=["error-page", "no-date-line", "no-race-number", "venue-not-jra"],
)
def test_a_page_that_does_not_name_its_race_is_an_error(body):
    with pytest.raises(ValueError):
        win_sales.parse_win_sales(body, fetched_utc=utc(2026, 11, 28, 6))


def test_a_naive_fetch_time_is_an_error():
    body = wsp.sales_page(DAY, MEETING, 5, state=FINAL, post="14時25分", votes=[("単勝", "100")])
    with pytest.raises(ValueError):
        win_sales.parse_win_sales(body, fetched_utc=datetime(2026, 11, 28, 6, 0))
