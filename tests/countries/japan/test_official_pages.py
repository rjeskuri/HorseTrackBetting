"""JRADB page parsing (ticket #21, ported from HKTrack's `sources/jra/official.py`, bugs a-c fixed).

Contract pinned here (the coder follows it):
- `horsetrack.countries.japan.jra.official_pages` is pure (no network). `soup(body)` decodes cp932.
- Link = (path, cname), equal to a plain tuple, e.g. ("/JRADB/accessO.html", "pw15orl.../3C").
- `page_meeting(doc) -> (date, venue code | None) | None`, `race_number(doc) -> int | None`.
- `meetings_on(menu, race_date, path) -> {venue code: Link}`: every meeting link to `path` named
  like 5回東京7日 under the day's h3 heading; links of other days, other paths or single races
  (東京11R) are not meetings.
- `race_rows(race_list) -> [(race number, row)]`, `pool_links(row) -> {pool code: Link}`.
- `post_times(card_race_list, race_date) -> {race number: aware UTC datetime}`, non-clock cells
  (発走済, empty) left out.
- `odds_rows(doc, race_id, pools, *, snapshot_utc, fetched_utc)`: HKTrack's rows (keys
  snapshot_utc, race_id, source_update_utc, pool, combination, odds, status_raw, `odds_hi` only
  on ranged prices, `investment` on "__pool__" rows) plus `fetched_utc` and `page_state` on
  every row of the page, pool-total rows included. Fixes: (a) source_update_utc is the page's
  clock on the JST date of fetched_utc, or the day before when that would be later than
  fetched_utc; (b) snapshot_utc is the caller's; (c) status_raw is the cell's own status only and
  the refresh-line text (whitespace collapsed) is page_state.

Pages are synthetic (jradb_pages.py). Expected values are written out here, never computed by
the code under test.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from itertools import combinations

import jradb_pages as pages
from jradb_pages import CARD_PATH, ODDS_PATH

from horsetrack.countries.japan.jra import official_pages as op

UTC = timezone.utc
SAT = date(2026, 11, 28)
SUN = date(2026, 11, 29)
MEETING = "5回東京7日"
RACE_ID = "jra:2026-11-28:TOK:11"
NOW_STATE = "14時00分現在"
SNAP = datetime(2026, 11, 28, 5, 0, 15, tzinfo=UTC)
FETCHED = datetime(2026, 11, 28, 5, 0, 20, tzinfo=UTC)  # 14:00:20 in Tokyo
UPDATED = datetime(2026, 11, 28, 5, 0, tzinfo=UTC)  # 14時00分 on 28 Nov in Tokyo
POOL = "__pool__"


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def rows_of(
    body: bytes,
    pools: list[str],
    *,
    snapshot: datetime = SNAP,
    fetched: datetime = FETCHED,
    race_id: str = RACE_ID,
) -> list[dict]:
    return op.odds_rows(op.soup(body), race_id, pools, snapshot_utc=snapshot, fetched_utc=fetched)


def by_key(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r["pool"], r["combination"]))


def base(
    *,
    state: str | None = NOW_STATE,
    updated: datetime | None = UPDATED,
    snapshot: datetime = SNAP,
    fetched: datetime = FETCHED,
    race_id: str = RACE_ID,
) -> dict:
    return {
        "snapshot_utc": snapshot,
        "race_id": race_id,
        "source_update_utc": updated,
        "fetched_utc": fetched,
        "page_state": state,
    }


def price_row(pool, combination, odds, odds_hi=None, status=None, **kw) -> dict:
    row = {**base(**kw), "pool": pool, "combination": combination, "odds": odds}
    row["status_raw"] = status
    if odds_hi is not None:
        row["odds_hi"] = odds_hi
    return row


def pool_row(pool: str, investment: float, **kw) -> dict:
    return {
        **base(**kw),
        "pool": pool,
        "combination": POOL,
        "odds": None,
        "status_raw": None,
        "investment": investment,
    }


def names(combination: str) -> set[int]:
    return {int(n) for n in combination.split(",")}


# ---------------------------------------------------------------- win and place

RUNNERS = {
    1: (5.5, (1.8, 2.4)),
    2: (12.3, (3.1, 5.0)),
    3: (1234.5, (88.8, 120.4)),
    4: (2.0, (1.1, 1.1)),
    10: (47.1, (7.7, 12.9)),
    12: (8.8, (2.2, 3.9)),
}


def test_win_and_place_rows_with_place_ranges():
    body = pages.win_place(SAT, MEETING, 11, RUNNERS, state=NOW_STATE)
    win = [price_row("WIN", str(n), w) for n, (w, _) in RUNNERS.items()]
    place = [price_row("PLACE", str(n), lo, hi) for n, (_, (lo, hi)) in RUNNERS.items()]

    assert by_key(rows_of(body, ["WIN", "PLACE"])) == by_key(win + place)
    # only the asked pools, and the race id is the caller's
    other = "jra:2026-11-28:TOK:03"
    assert by_key(rows_of(body, ["WIN"], race_id=other)) == by_key(
        [{**r, "race_id": other} for r in win]
    )
    assert by_key(rows_of(body, ["PLACE"])) == by_key(place)
    assert all("odds_hi" not in r for r in rows_of(body, ["WIN"])), "Win prints one figure"


# ---------------------------------------------------------------- quinella and wide

QUINELLA = {(1, 2): 12.5, (1, 3): 30.1, (2, 3): 8.4, (10, 9): 101.7, (12, 3): 1530.0, (4, 12): 66.6}
WIDE = {
    (1, 2): (3.2, 4.1),
    (1, 3): (7.7, 9.9),
    (2, 3): (2.5, 2.5),
    (10, 9): (25.0, 31.4),
    (12, 3): (210.3, 260.8),
    (4, 12): (11.1, 15.6),
}
SORTED = {
    (1, 2): "1,2",
    (1, 3): "1,3",
    (2, 3): "2,3",
    (10, 9): "9,10",
    (12, 3): "3,12",
    (4, 12): "4,12",
}


def test_quinella_and_wide_pairs_are_sorted_and_wide_is_ranged():
    quinella = pages.pairs(SAT, MEETING, 11, QUINELLA, state=NOW_STATE)
    wide = pages.pairs(SAT, MEETING, 11, WIDE, state=NOW_STATE)

    expected_q = [price_row("QUINELLA", SORTED[k], v) for k, v in QUINELLA.items()]
    expected_w = [price_row("QUINELLA_PLACE", SORTED[k], lo, hi) for k, (lo, hi) in WIDE.items()]
    assert by_key(rows_of(quinella, ["QUINELLA"])) == by_key(expected_q)
    assert by_key(rows_of(wide, ["QUINELLA_PLACE"])) == by_key(expected_w)

    for row in rows_of(quinella, ["QUINELLA"]) + rows_of(wide, ["QUINELLA_PLACE"]):
        a, b = (int(n) for n in row["combination"].split(","))
        assert a < b, row["combination"]


# ---------------------------------------------------------------- scratched runners

SCRATCHED = {3: "除外", 6: "取消"}
FIELD = range(1, 9)


def scratched_field() -> dict[int, tuple]:
    out = {}
    for n in FIELD:
        if n in SCRATCHED:
            out[n] = (SCRATCHED[n], SCRATCHED[n])
        else:
            out[n] = (float(n) + 0.5, (round(1 + n / 10, 1), round(2 + n / 10, 1)))
    return out


def scratched_pairs(ranged: bool) -> dict[tuple[int, int], object]:
    out: dict[tuple[int, int], object] = {}
    for a, b in combinations(FIELD, 2):
        word = SCRATCHED.get(a) or SCRATCHED.get(b)
        out[(a, b)] = word or ((a + b, a + b + 0.5) if ranged else float(a * 10 + b))
    return out


def test_a_scratched_runner_has_status_scr_and_no_pair_rows():
    rows = rows_of(
        pages.win_place(SAT, MEETING, 11, scratched_field(), state=NOW_STATE), ["WIN", "PLACE"]
    )
    expected = []
    for n, (win, place) in scratched_field().items():
        if n in SCRATCHED:
            expected += [price_row(p, str(n), None, status="SCR") for p in ("WIN", "PLACE")]
        else:
            expected += [price_row("WIN", str(n), win), price_row("PLACE", str(n), *place)]
    assert by_key(rows) == by_key(expected)

    starters = [n for n in FIELD if n not in SCRATCHED]
    want = {f"{a},{b}" for a, b in combinations(starters, 2)}
    for pool, ranged in (("QUINELLA", False), ("QUINELLA_PLACE", True)):
        body = pages.pairs(SAT, MEETING, 11, scratched_pairs(ranged), state=NOW_STATE)
        got = rows_of(body, [pool])
        assert {r["combination"] for r in got} == want, pool
        assert not [r for r in got if names(r["combination"]) & set(SCRATCHED)], pool
        assert all(r["odds"] is not None and r["status_raw"] is None for r in got), pool


# ---------------------------------------------------------------- pool totals


def test_pool_totals_are_pool_rows_in_yen():
    votes = [("単勝", 12_345), ("複勝", 6_789), ("総票数合計", 99_999)]
    body = pages.win_place(SAT, MEETING, 11, RUNNERS, state=NOW_STATE, votes=votes)
    totals = [r for r in rows_of(body, ["WIN", "PLACE"]) if r["combination"] == POOL]
    assert by_key(totals) == by_key([pool_row("WIN", 1_234_500.0), pool_row("PLACE", 678_900.0)])

    only_win = rows_of(body, ["WIN"])
    assert {r["pool"] for r in only_win} == {"WIN"}
    assert [r for r in only_win if r["combination"] == POOL] == [pool_row("WIN", 1_234_500.0)]

    quinella = pages.pairs(SAT, MEETING, 11, QUINELLA, state=NOW_STATE, votes=[("馬連", 54_321)])
    wide = pages.pairs(SAT, MEETING, 11, WIDE, state=NOW_STATE, votes=[("ワイド", 7)])
    q_totals = [r for r in rows_of(quinella, ["QUINELLA"]) if r["combination"] == POOL]
    w_totals = [r for r in rows_of(wide, ["QUINELLA_PLACE"]) if r["combination"] == POOL]
    assert q_totals == [pool_row("QUINELLA", 5_432_100.0)]
    assert w_totals == [pool_row("QUINELLA_PLACE", 700.0)]


# ---------------------------------------------------------------- (b) one snapshot per capture


def test_every_row_carries_the_given_snapshot_utc():
    votes = {"WIN": [("単勝", 100)], "Q": [("馬連", 200)], "W": [("ワイド", 300)]}
    captures = [
        (utc(2026, 11, 28, 4, 59, 58), [utc(2026, 11, 28, 5, 0, 1), utc(2026, 11, 28, 5, 0, 3)]),
        (utc(2026, 11, 28, 5, 0, 30), [utc(2026, 11, 28, 5, 0, 31), utc(2026, 11, 28, 5, 0, 44)]),
    ]
    for snapshot, (t_wp, t_wide) in captures:
        wp = pages.win_place(SAT, MEETING, 11, RUNNERS, state=NOW_STATE, votes=votes["WIN"])
        wide = pages.pairs(SAT, MEETING, 11, WIDE, state=NOW_STATE, votes=votes["W"])
        for body, pools, fetched in (
            (wp, ["WIN", "PLACE"], t_wp),
            (wide, ["QUINELLA_PLACE"], t_wide),
        ):
            rows = rows_of(body, pools, snapshot=snapshot, fetched=fetched)
            assert rows and any(r["combination"] == POOL for r in rows)
            assert {r["snapshot_utc"] for r in rows} == {snapshot}, "the caller's, not the clock's"
            assert {r["fetched_utc"] for r in rows} == {fetched}, "each page's own fetch time"


# ---------------------------------------------------------------- (c) status_raw vs page_state


def test_status_raw_is_null_for_a_price_and_page_state_holds_the_refresh_text():
    runners = {
        1: (5.5, (1.8, 2.4)),
        2: ("取消", "取消"),
        5: ("---.-", "---.-"),
        7: (9.9, (2.0, 3.3)),
    }
    for state, fetched, updated in (
        (NOW_STATE, FETCHED, UPDATED),
        ("9時41分現在", utc(2026, 11, 28, 0, 41, 50), utc(2026, 11, 28, 0, 41)),
    ):
        body = pages.win_place(SAT, MEETING, 11, runners, state=state, votes=[("単勝", 10)])
        rows = rows_of(body, ["WIN", "PLACE"], fetched=fetched)
        assert {r["page_state"] for r in rows} == {state}
        assert {r["source_update_utc"] for r in rows} == {updated}
        status = {(r["pool"], r["combination"]): r["status_raw"] for r in rows}
        assert status == {
            ("WIN", "1"): None,
            ("PLACE", "1"): None,
            ("WIN", "2"): "SCR",
            ("PLACE", "2"): "SCR",
            ("WIN", "5"): "---.-",
            ("PLACE", "5"): "---.-",
            ("WIN", "7"): None,
            ("PLACE", "7"): None,
            ("WIN", POOL): None,
        }
        assert all(r["odds"] is None for r in rows if r["combination"] in ("2", "5", POOL))


# ---------------------------------------------------------------- (a) source_update_utc


def source_updates(race_day: date, clock: str, fetched: datetime) -> set:
    body = pages.win_place(
        race_day, "5回東京7日", 11, RUNNERS, state=f"{clock}現在", votes=[("単勝", 1)]
    )
    return {r["source_update_utc"] for r in rows_of(body, ["WIN", "PLACE"], fetched=fetched)}


def test_source_update_is_dated_by_the_fetch_day_not_the_race_day():
    cases = [
        # (race day on the page, page clock, fetched_utc, expected source_update_utc)
        (SAT, "18時55分", utc(2026, 11, 27, 10, 0), utc(2026, 11, 27, 9, 55)),  # Fri 19:00 JST
        (SAT, "14時00分", utc(2026, 11, 28, 5, 0, 20), utc(2026, 11, 28, 5, 0)),
        (SAT, "15時30分", utc(2026, 11, 28, 6, 30), utc(2026, 11, 28, 6, 30)),  # same minute
        (SUN, "9時05分", utc(2026, 11, 28, 0, 10), utc(2026, 11, 28, 0, 5)),
        (date(2026, 12, 6), "20時10分", utc(2026, 12, 4, 11, 15), utc(2026, 12, 4, 11, 10)),
    ]
    for race_day, clock, fetched, expected in cases:
        assert source_updates(race_day, clock, fetched) == {expected}, (race_day, clock, fetched)


def test_source_update_seen_just_after_midnight_is_the_previous_day():
    cases = [
        # 00:05 JST on Sat 28 Nov, page 23時58分 -> 23:58 JST on Fri 27 Nov
        (SAT, "23時58分", utc(2026, 11, 27, 15, 5), utc(2026, 11, 27, 14, 58)),
        # 00:00:30 JST on Sun 29 Nov, page 23時59分 -> 23:59 JST on Sat 28 Nov
        (SUN, "23時59分", utc(2026, 11, 28, 15, 0, 30), utc(2026, 11, 28, 14, 59)),
        # 00:00:30 JST on Sat 28 Nov, page 0時00分 -> the same local day
        (SAT, "0時00分", utc(2026, 11, 27, 15, 0, 30), utc(2026, 11, 27, 15, 0)),
    ]
    for race_day, clock, fetched, expected in cases:
        assert source_updates(race_day, clock, fetched) == {expected}, (race_day, clock, fetched)


def test_final_odds_page_has_no_source_update():
    final = "最終オッズ"
    for fetched in (utc(2026, 11, 28, 7, 0), utc(2026, 11, 30, 12, 0)):
        wp = pages.win_place(SAT, MEETING, 11, RUNNERS, state=final, votes=[("単勝", 5)])
        wide = pages.pairs(SAT, MEETING, 11, WIDE, state=final, votes=[("ワイド", 5)])
        rows = rows_of(wp, ["WIN", "PLACE"], fetched=fetched)
        rows += rows_of(wide, ["QUINELLA_PLACE"], fetched=fetched)
        assert {r["source_update_utc"] for r in rows} == {None}
        assert {r["page_state"] for r in rows} == {final}
        assert {r["status_raw"] for r in rows} == {None}, "the page state is not a cell status"
        win = {r["combination"]: r["odds"] for r in rows if r["pool"] == "WIN"}
        assert win == {**{str(n): w for n, (w, _) in RUNNERS.items()}, POOL: None}


# ---------------------------------------------------------------- menus and race lists

SAT_MEETINGS = [
    ("5回東京7日", "pw15orl10052026050720261128/3C"),
    ("5回京都7日", "pw15orl10082026050720261128/1F"),
    ("3回福島7日", "pw15orl10032026030720261128/A0"),
]
SUN_MEETINGS = [
    ("5回東京8日", "pw15orl10052026050820261129/77"),
    ("5回京都8日", "pw15orl10082026050820261129/E4"),
    ("4回中京2日", "pw15orl10072026040220261129/09"),
]


def test_meetings_on_lists_every_venue_of_the_day_and_no_other_day():
    menu = op.soup(pages.menu({SAT: SAT_MEETINGS, SUN: SUN_MEETINGS}, ODDS_PATH))
    assert op.meetings_on(menu, SAT, ODDS_PATH) == {
        "TOK": (ODDS_PATH, SAT_MEETINGS[0][1]),
        "KYO": (ODDS_PATH, SAT_MEETINGS[1][1]),
        "FKS": (ODDS_PATH, SAT_MEETINGS[2][1]),
    }
    assert op.meetings_on(menu, SUN, ODDS_PATH) == {
        "TOK": (ODDS_PATH, SUN_MEETINGS[0][1]),
        "KYO": (ODDS_PATH, SUN_MEETINGS[1][1]),
        "CKY": (ODDS_PATH, SUN_MEETINGS[2][1]),
    }
    assert op.meetings_on(menu, date(2026, 11, 30), ODDS_PATH) == {}
    assert op.meetings_on(menu, SAT, CARD_PATH) == {}, "links to another path are not meetings"

    card = [(name, cname.replace("pw15o", "pw01d")) for name, cname in SUN_MEETINGS[:2]]
    card_menu = op.soup(pages.menu({SUN: card}, CARD_PATH))
    assert op.meetings_on(card_menu, SUN, CARD_PATH) == {
        "TOK": (CARD_PATH, card[0][1]),
        "KYO": (CARD_PATH, card[1][1]),
    }


def test_post_times_from_the_card_race_list_in_utc():
    cases = [
        (
            SAT,
            {1: "9時55分", 2: "10時25分", 7: "13時05分", 11: "15時40分", 12: "16時20分"},
            {
                1: utc(2026, 11, 28, 0, 55),
                2: utc(2026, 11, 28, 1, 25),
                7: utc(2026, 11, 28, 4, 5),
                11: utc(2026, 11, 28, 6, 40),
                12: utc(2026, 11, 28, 7, 20),
            },
        ),
        (
            date(2026, 12, 6),
            {1: "10時01分", 12: "16時30分"},
            {1: utc(2026, 12, 6, 1, 1), 12: utc(2026, 12, 6, 7, 30)},
        ),
    ]
    for day, times, expected in cases:
        got = op.post_times(op.soup(pages.card_race_list(day, "5回東京7日", times)), day)
        assert got == expected, day
        for when in got.values():
            assert when.tzinfo is not None and when.utcoffset() == timedelta(0), when


def test_post_times_leave_out_races_already_started():
    times = {1: "発走済", 2: "発走済", 3: "10時55分", 4: "11時25分", 5: ""}
    got = op.post_times(op.soup(pages.card_race_list(SAT, MEETING, times)), SAT)
    assert got == {3: utc(2026, 11, 28, 1, 55), 4: utc(2026, 11, 28, 2, 25)}


def test_page_meeting_reads_date_and_venue():
    cases = [
        (SAT, "5回東京7日", "TOK"),
        (SUN, "5回京都8日", "KYO"),
        (date(2027, 1, 5), "1回中山1日", "NAK"),
        (date(2026, 12, 6), "4回中京2日", "CKY"),
        (date(2026, 12, 6), "9回大井3日", None),  # a name jra.yaml does not know
    ]
    for day, meeting, venue in cases:
        odds_page = pages.win_place(day, meeting, 11, RUNNERS, state=NOW_STATE)
        race_list = pages.odds_race_list(day, meeting, {1: {"単勝 複勝": "pw151ou1/D7"}})
        assert op.page_meeting(op.soup(odds_page)) == (day, venue), meeting
        assert op.page_meeting(op.soup(race_list)) == (day, venue), meeting
    assert op.page_meeting(op.soup(pages.no_header_page())) is None


def test_race_number_reads_the_race_number_image():
    for n in (1, 10, 12):
        assert op.race_number(op.soup(pages.pairs(SAT, MEETING, n, QUINELLA, state=None))) == n
    race_list = pages.odds_race_list(SAT, MEETING, {1: {"単勝 複勝": "pw151ou1/D7"}})
    assert op.race_number(op.soup(race_list)) is None
    assert op.race_number(op.soup(pages.no_header_page())) is None


def test_link_of_reads_do_action_and_href_links():
    snippet = pages.page(
        pages.action_link(ODDS_PATH, "pw151ou1005202605071120261128Z/1B", "a")
        + pages.href_link(CARD_PATH, "pw01dde1005202605071120261128/D9", "b")
        + '<a href="#">c</a><a href="/elsewhere.html">d</a>'
    )
    links = [op.link_of(a) for a in op.soup(snippet).find_all("a")]
    assert links == [
        (ODDS_PATH, "pw151ou1005202605071120261128Z/1B"),
        (CARD_PATH, "pw01dde1005202605071120261128/D9"),
        None,
        None,
    ]


def test_race_rows_and_pool_links_of_the_odds_race_list():
    def buttons(race_no: int) -> dict[str, str]:
        code = f"{race_no:02d}20261128Z"
        return {
            "単勝 複勝": f"pw151ou{code}/{race_no:02X}",
            "枠連": f"pw153ou{code}/A{race_no % 10}",
            "馬連": f"pw154ou{code}/B{race_no % 10}",
            "ワイド": f"pw155ou{code}/C{race_no % 10}",
            "馬単": f"pw156ou{code}/D{race_no % 10}",
            "3連複": f"pw157ou{code}/E{race_no % 10}",
            "3連単": f"pw158ou{code}/F{race_no % 10}",
        }

    races = {n: buttons(n) for n in (1, 2, 11, 12)}
    rows = op.race_rows(op.soup(pages.odds_race_list(SAT, MEETING, races)))
    assert [n for n, _ in rows] == [1, 2, 11, 12]
    for n, row in rows:
        b = races[n]
        assert op.pool_links(row) == {
            "WIN": (ODDS_PATH, b["単勝 複勝"]),
            "PLACE": (ODDS_PATH, b["単勝 複勝"]),
            "BRACKET_QUINELLA": (ODDS_PATH, b["枠連"]),
            "QUINELLA": (ODDS_PATH, b["馬連"]),
            "QUINELLA_PLACE": (ODDS_PATH, b["ワイド"]),
            "FORECAST": (ODDS_PATH, b["馬単"]),
            "TRIO": (ODDS_PATH, b["3連複"]),
            "TIERCE": (ODDS_PATH, b["3連単"]),
        }, n
