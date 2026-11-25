"""NAR page parsing from nar.netkeiba.com (ticket #37), per the "Logger spec for NAR" in
docs/research/nar-data-access.md and the row format of docs/specs/odds-log.md.

Contract pinned here (the coder follows it):
- `horsetrack.countries.japan.nar.pages` is pure (no network). `soup(body)` decodes UTF-8.
- `nar_race_id(netkeiba_id) -> str`: the 12-digit id `YYYY VV MMDD RR` -> the store race id
  `nar:<YYYY-MM-DD>:<VENUE>:<NN>` (JST date, short venue code of config/nar.yaml); ValueError on
  a venue code nar.yaml does not know.
- `meetings_on(race_list, race_date) -> {venue code: Meeting}`; a Meeting has `venue` (short
  code), `kaisai_id` (10 digits), `nighter` (bool, `small.icon-nighter` in its title) and
  `races` ({race number: 12-digit race id}). Meetings of another date are left out; a venue
  code nar.yaml does not know raises ValueError.
- `post_times(race_list, race_date, venue) -> {race number: aware UTC datetime}` (the page
  prints JST `HH:MM`); {} for a venue not racing.
- `odds_rows(doc, race_id, pools, *, snapshot_utc, fetched_utc) -> list[dict]`, `race_id` the
  12-digit netkeiba id. Every row: snapshot_utc (the caller's), fetched_utc (this page's),
  race_id (`nar:` form), source_race_key (the 12 digits), pool, combination, odds, status_raw,
  source_update_utc, page_state; `odds_hi` only on ranged prices (PLACE, QUINELLA_PLACE);
  `investment` only on "__pool__" rows. As JRA: a scratched runner keeps WIN and PLACE rows with
  odds None and status_raw "SCR", and pairs naming it are left out; pair combinations are the two
  numbers sorted ascending, joined by ","; pool totals are tickets x 100 yen; the pool of a block
  is read from its heading label (nar.yaml labels), so a page never yields another pool's rows;
  source_update_utc is the page's HH:MM on the JST date of fetched_utc, or the day before when
  that would be later than fetched_utc, None when the page prints no clock; page_state is the
  clock's text (whitespace collapsed), None when absent.
- A `b1` page before sales (the `.Ninki` table, or any odds table headed 予想オッズ) raises
  `NotOnSale`. Header labels are read as the site prints them: `馬<br>番`, `予想<br>オッズ`.

Pages are synthetic (nar_pages.py; its docstring lists which markup is a guess). Expected values
are written out here, never computed by the code under test.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from itertools import combinations

import nar_pages as np_
import pytest
from nar_pages import Meeting, Race

from horsetrack.core.config import load_country_yaml
from horsetrack.core.ingest.fetcher import Fetcher
from horsetrack.countries.japan.nar import pages

UTC = timezone.utc
POOL = "__pool__"
WED = date(2026, 11, 25)
RACE = "202642112511"  # 浦和 11R on 2026-11-25
STORE = "nar:2026-11-25:URA:11"
SNAP = datetime(2026, 11, 25, 5, 30, 5, tzinfo=UTC)
FETCHED = datetime(2026, 11, 25, 5, 30, 9, tzinfo=UTC)  # 14:30:09 JST
CLOCK = "14:27"
UPDATED = datetime(2026, 11, 25, 5, 27, tzinfo=UTC)  # 14:27 JST on 25 Nov


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def rows_of(
    body: bytes,
    pools: list[str],
    *,
    race: str = RACE,
    snapshot: datetime = SNAP,
    fetched: datetime = FETCHED,
) -> list[dict]:
    return pages.odds_rows(
        pages.soup(body), race, pools, snapshot_utc=snapshot, fetched_utc=fetched
    )


def by_key(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r["pool"], r["combination"]))


def base(
    *,
    race: str = RACE,
    store: str = STORE,
    snapshot: datetime = SNAP,
    fetched: datetime = FETCHED,
    updated: datetime | None = UPDATED,
    state: str | None = f"{CLOCK}現在",
) -> dict:
    return {
        "snapshot_utc": snapshot,
        "fetched_utc": fetched,
        "race_id": store,
        "source_race_key": race,
        "source_update_utc": updated,
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


def prices(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["combination"] != POOL]


def totals(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["combination"] == POOL]


# ---------------------------------------------------------------- race list

# Two days, two meetings each: venue codes and short codes from the research doc's table.
DAYS = {
    WED: [
        Meeting("42", "浦和", [Race(1, "12:20"), Race(7, "15:30"), Race(12, "18:20")]),
        Meeting(
            "48",
            "名古屋",
            [Race(1, "14:40"), Race(5, "16:55"), Race(12, "20:50")],
            nighter=True,
        ),
    ],
    date(2026, 12, 3): [
        Meeting("50", "園田", [Race(1, "08:50"), Race(2, "09:20"), Race(10, "13:45")]),
        Meeting(
            "44",
            "大井",
            [Race(3, "15:10"), Race(9, "18:40"), Race(11, "20:10")],
            nighter=True,
        ),
    ],
}
# (venue code, kaisai_id, nighter, {race number: 12-digit id}, {race number: store id})
EXPECTED_MEETINGS = {
    WED: {
        "URA": (
            "2026421125",
            False,
            {1: "202642112501", 7: "202642112507", 12: "202642112512"},
            {1: "nar:2026-11-25:URA:01", 7: "nar:2026-11-25:URA:07", 12: "nar:2026-11-25:URA:12"},
        ),
        "NGY": (
            "2026481125",
            True,
            {1: "202648112501", 5: "202648112505", 12: "202648112512"},
            {1: "nar:2026-11-25:NGY:01", 5: "nar:2026-11-25:NGY:05", 12: "nar:2026-11-25:NGY:12"},
        ),
    },
    date(2026, 12, 3): {
        "SON": (
            "2026501203",
            False,
            {1: "202650120301", 2: "202650120302", 10: "202650120310"},
            {1: "nar:2026-12-03:SON:01", 2: "nar:2026-12-03:SON:02", 10: "nar:2026-12-03:SON:10"},
        ),
        "OOI": (
            "2026441203",
            True,
            {3: "202644120303", 9: "202644120309", 11: "202644120311"},
            {3: "nar:2026-12-03:OOI:03", 9: "nar:2026-12-03:OOI:09", 11: "nar:2026-12-03:OOI:11"},
        ),
    },
}


def test_meetings_of_the_day():
    for day, meetings in DAYS.items():
        doc = pages.soup(np_.race_list(day, meetings))
        got = pages.meetings_on(doc, day)
        expected = EXPECTED_MEETINGS[day]
        assert set(got) == set(expected), day
        for venue, (kaisai, nighter, races, store_ids) in expected.items():
            m = got[venue]
            assert m.venue == venue
            assert m.kaisai_id == kaisai, venue
            assert m.nighter is nighter, venue
            assert m.races == races, venue
            assert {n: pages.nar_race_id(rid) for n, rid in m.races.items()} == store_ids

        # the list of one date names no meeting of another
        other = date.fromordinal(day.toordinal() + 1)
        assert pages.meetings_on(doc, other) == {}, day

    # decoys off: the same meetings (the 払戻一覧 item and the next-day link are not races)
    plain = pages.soup(np_.race_list(WED, DAYS[WED], decoys=False))
    assert {v: m.races for v, m in pages.meetings_on(plain, WED).items()} == {
        v: races for v, (_, _, races, _) in EXPECTED_MEETINGS[WED].items()
    }


def test_post_times_in_utc():
    expected = {
        (WED, "URA"): {
            1: utc(2026, 11, 25, 3, 20),
            7: utc(2026, 11, 25, 6, 30),
            12: utc(2026, 11, 25, 9, 20),
        },
        (WED, "NGY"): {
            1: utc(2026, 11, 25, 5, 40),
            5: utc(2026, 11, 25, 7, 55),
            12: utc(2026, 11, 25, 11, 50),
        },
        # 08:50 JST is 23:50 UTC on the day before
        (date(2026, 12, 3), "SON"): {
            1: utc(2026, 12, 2, 23, 50),
            2: utc(2026, 12, 3, 0, 20),
            10: utc(2026, 12, 3, 4, 45),
        },
        (date(2026, 12, 3), "OOI"): {
            3: utc(2026, 12, 3, 6, 10),
            9: utc(2026, 12, 3, 9, 40),
            11: utc(2026, 12, 3, 11, 10),
        },
    }
    for (day, venue), want in expected.items():
        doc = pages.soup(np_.race_list(day, DAYS[day]))
        got = pages.post_times(doc, day, venue)
        assert got == want, (day, venue)
        assert all(t.utcoffset() == UTC.utcoffset(None) for t in got.values()), (day, venue)

    # a venue not racing that day has no post times
    assert pages.post_times(pages.soup(np_.race_list(WED, DAYS[WED])), WED, "SON") == {}


# ---------------------------------------------------------------- win and place

RUNNERS = {
    1: (5.5, (1.8, 2.4)),
    2: (12.3, (3.1, 5.0)),
    3: (1.4, (1.0, 1.1)),
    4: (87.6, (9.9, 15.2)),
    5: (234.5, (30.3, 48.0)),
    6: (3.0, (1.1, 1.1)),
    7: (19.8, (4.4, 7.0)),
    8: (999.9, (88.8, 120.4)),
}
# A 12-runner field (brackets differ from numbers), its columns in another order.
RUNNERS_12 = {
    n: (round(2.0 + 3.1 * n, 1), (round(1.0 + 0.4 * n, 1), round(1.5 + n, 1))) for n in range(1, 13)
}
REORDERED = ("馬番", "枠", "馬名", "印", "オッズ")


def test_win_and_place_rows():
    cases = [
        (RACE, STORE, RUNNERS, np_.TAN_COLUMNS),
        ("202650112503", "nar:2026-11-25:SON:03", RUNNERS, np_.TAN_COLUMNS),
        ("202654120309", "nar:2026-12-03:KOC:09", RUNNERS_12, REORDERED),
    ]
    for race, store, runners, columns in cases:
        body = np_.odds_b1(race, runners, updated=CLOCK, columns=columns)
        ids = {"race": race, "store": store}
        win = [price_row("WIN", str(n), w, **ids) for n, (w, _) in runners.items()]
        place = [price_row("PLACE", str(n), lo, hi, **ids) for n, (_, (lo, hi)) in runners.items()]

        assert by_key(rows_of(body, ["WIN", "PLACE"], race=race)) == by_key(win + place), race
        # only the pools asked for
        assert by_key(rows_of(body, ["WIN"], race=race)) == by_key(win), race
        assert by_key(rows_of(body, ["PLACE"], race=race)) == by_key(place), race
        assert all("odds_hi" not in r for r in rows_of(body, ["WIN"], race=race)), "one figure"


# ---------------------------------------------------------------- pair pools

PAIRS_WIDE = {
    (1, 2): (3.2, 4.1),
    (1, 3): (7.7, 9.9),
    (2, 3): (2.5, 2.5),
    (10, 9): (25.0, 31.4),
    (12, 3): (210.3, 260.8),
    (4, 12): (11.1, 15.6),
}
PAIRS_QUINELLA = {
    (1, 2): 12.5,
    (1, 3): 30.1,
    (2, 3): 8.4,
    (10, 9): 101.7,
    (12, 3): 1530.0,
    (4, 12): 66.6,
}
SORTED = {
    (1, 2): "1,2",
    (1, 3): "1,3",
    (2, 3): "2,3",
    (10, 9): "9,10",
    (12, 3): "3,12",
    (4, 12): "4,12",
}


def test_pair_pool_rows():
    wide = np_.odds_pairs(RACE, PAIRS_WIDE, label="ワイド", updated=CLOCK)
    expected_w = [
        price_row("QUINELLA_PLACE", SORTED[k], lo, hi) for k, (lo, hi) in PAIRS_WIDE.items()
    ]
    assert by_key(rows_of(wide, ["QUINELLA_PLACE"])) == by_key(expected_w)

    # 馬連 and NAR's own name 馬複 are both QUINELLA: one figure, no odds_hi
    for label in ("馬連", "馬複"):
        body = np_.odds_pairs(RACE, PAIRS_QUINELLA, label=label, updated=CLOCK)
        expected_q = [price_row("QUINELLA", SORTED[k], v) for k, v in PAIRS_QUINELLA.items()]
        assert by_key(rows_of(body, ["QUINELLA"])) == by_key(expected_q), label
        # the block's heading names the pool: a 馬連 page is never logged as ワイド
        assert rows_of(body, ["QUINELLA_PLACE"]) == [], label

    assert rows_of(wide, ["QUINELLA"]) == [], "a ワイド page is never logged as 馬連"
    assert rows_of(wide, ["WIN", "PLACE"]) == [], "no runner rows on a pair page"


# ---------------------------------------------------------------- scratched runners

FIELD = range(1, 9)


def scratched_field(scratched: dict[int, str]) -> dict[int, tuple]:
    return {
        n: (word, word)
        if (word := scratched.get(n))
        else (round(n + 0.5, 1), (round(1 + n / 10, 1), round(2 + n / 10, 1)))
        for n in FIELD
    }


def scratched_pairs(scratched: dict[int, str]) -> dict[tuple[int, int], object]:
    out: dict[tuple[int, int], object] = {}
    for a, b in combinations(FIELD, 2):
        word = scratched.get(a) or scratched.get(b)
        out[(a, b)] = word or (float(a + b), a + b + 0.5)
    return out


def test_a_scratched_runner():
    for scratched in ({5: "取消"}, {2: "除外"}, {3: "取消", 8: "除外"}):
        field = scratched_field(scratched)
        rows = rows_of(np_.odds_b1(RACE, field, updated=CLOCK), ["WIN", "PLACE"])
        expected = []
        for n, (win, place) in field.items():
            if n in scratched:
                expected += [price_row(p, str(n), None, status="SCR") for p in ("WIN", "PLACE")]
            else:
                expected += [price_row("WIN", str(n), win), price_row("PLACE", str(n), *place)]
        assert by_key(rows) == by_key(expected), scratched
        assert not [r for r in rows if r["status_raw"] == "SCR" and "odds_hi" in r], scratched

        # pairs naming a scratched runner are left out; every other pair has its price
        starters = [n for n in FIELD if n not in scratched]
        wide = rows_of(
            np_.odds_pairs(RACE, scratched_pairs(scratched), updated=CLOCK), ["QUINELLA_PLACE"]
        )
        assert {r["combination"] for r in wide} == {
            f"{a},{b}" for a, b in combinations(starters, 2)
        }
        assert all(r["odds"] is not None and r["status_raw"] is None for r in wide), scratched


# ---------------------------------------------------------------- pool totals


def test_pool_totals():
    for tan, fuku, wide_tickets in ((12_345, 6_789, 4_321), (1, 250_000, 98_765)):
        votes = [("単勝", tan), ("複勝", fuku), ("合計", tan + fuku + 1)]
        b1 = np_.odds_b1(RACE, RUNNERS, updated=CLOCK, votes=votes)
        assert by_key(totals(rows_of(b1, ["WIN", "PLACE"]))) == by_key(
            [pool_row("WIN", tan * 100.0), pool_row("PLACE", fuku * 100.0)]
        )
        only_win = rows_of(b1, ["WIN"])
        assert {r["pool"] for r in only_win} == {"WIN"}
        assert totals(only_win) == [pool_row("WIN", tan * 100.0)]

        wide = np_.odds_pairs(RACE, PAIRS_WIDE, updated=CLOCK, votes=[("ワイド", wide_tickets)])
        assert totals(rows_of(wide, ["QUINELLA_PLACE"])) == [
            pool_row("QUINELLA_PLACE", wide_tickets * 100.0)
        ]

    # a page that prints no totals has no pool rows
    assert totals(rows_of(np_.odds_b1(RACE, RUNNERS, updated=CLOCK), ["WIN", "PLACE"])) == []


# ---------------------------------------------------------------- snapshot and fetch times


def test_rows_share_the_snapshot_utc_and_carry_fetched_utc():
    captures = [
        (utc(2026, 11, 25, 5, 29, 58), utc(2026, 11, 25, 5, 30, 1), utc(2026, 11, 25, 5, 30, 12)),
        (utc(2026, 11, 25, 5, 31, 0), utc(2026, 11, 25, 5, 31, 10), utc(2026, 11, 25, 5, 31, 44)),
    ]
    for snapshot, t_b1, t_b5 in captures:
        b1 = np_.odds_b1(RACE, RUNNERS, updated=CLOCK, votes=[("単勝", 100), ("複勝", 50)])
        b5 = np_.odds_pairs(RACE, PAIRS_WIDE, updated=CLOCK, votes=[("ワイド", 300)])
        capture = []
        for body, pools, fetched in ((b1, ["WIN", "PLACE"], t_b1), (b5, ["QUINELLA_PLACE"], t_b5)):
            rows = rows_of(body, pools, snapshot=snapshot, fetched=fetched)
            assert prices(rows) and totals(rows), pools
            assert {r["fetched_utc"] for r in rows} == {fetched}, "each page's own fetch time"
            capture += rows
        assert {r["snapshot_utc"] for r in capture} == {snapshot}, "the caller's, not the clock's"
        assert {r["race_id"] for r in capture} == {STORE}
        assert {r["source_race_key"] for r in capture} == {RACE}


def source_updates(clock: str | None, fetched: datetime) -> set:
    body = np_.odds_b1(RACE, RUNNERS, updated=clock, votes=[("単勝", 1)])
    return {r["source_update_utc"] for r in rows_of(body, ["WIN", "PLACE"], fetched=fetched)}


def test_source_update_is_dated_by_the_fetch_day():
    cases = [
        # (page clock, fetched_utc, expected source_update_utc)
        ("14:27", utc(2026, 11, 25, 5, 30, 9), utc(2026, 11, 25, 5, 27)),
        ("20:48", utc(2026, 11, 25, 11, 49), utc(2026, 11, 25, 11, 48)),  # nighter
        ("10:05", utc(2026, 11, 25, 1, 5), utc(2026, 11, 25, 1, 5)),  # same minute
        ("08:41", utc(2026, 12, 2, 23, 42), utc(2026, 12, 2, 23, 41)),  # JST 3 Dec, UTC 2 Dec
        # fetched the evening before the race day: dated by the fetch day, not the race's
        ("18:55", utc(2026, 11, 24, 10, 0), utc(2026, 11, 24, 9, 55)),
        # seen just after midnight JST: the clock is the day before
        ("23:58", utc(2026, 11, 25, 15, 5), utc(2026, 11, 25, 14, 58)),
    ]
    for clock, fetched, expected in cases:
        assert source_updates(clock, fetched) == {expected}, (clock, fetched)

    # no clock on the page (as on the finished race captured): no source update, no page state
    rows = rows_of(np_.odds_b1(RACE, RUNNERS, updated=None), ["WIN", "PLACE"])
    assert {r["source_update_utc"] for r in rows} == {None}
    assert {r["page_state"] for r in rows} == {None}


# ---------------------------------------------------------------- before sales, unknown venue


def test_before_sales_page_is_not_on_sale():
    forecast = {1: 4.2, 2: 2.1, 3: 15.0, 4: 33.3, 5: 7.5, 6: 60.0, 7: 9.8, 8: 120.0}
    # ninki_class=False: only the 予想<br>オッズ header (as the site prints it) marks the page
    for in_tan_block in (False, True):
        for ninki_class in (True, False):
            body = np_.odds_b1_before_sales(
                RACE, forecast, in_tan_block=in_tan_block, ninki_class=ninki_class
            )
            for pools in (["WIN", "PLACE"], ["WIN"], ["PLACE"]):
                with pytest.raises(pages.NotOnSale):
                    rows_of(body, pools)


def test_an_unknown_venue_code_fails_loudly():
    with pytest.raises(ValueError):
        pages.nar_race_id("202699112501")
    with pytest.raises(ValueError):
        rows_of(np_.odds_b1("202699112501", RUNNERS), ["WIN"], race="202699112501")
    unknown = [*DAYS[WED], Meeting("99", "架空", [Race(1, "11:00")])]
    with pytest.raises(ValueError):
        pages.meetings_on(pages.soup(np_.race_list(WED, unknown)), WED)


# ---------------------------------------------------------------- config/nar.yaml

VENUES = {
    "OBI": "65",
    "MON": "30",
    "MOR": "35",
    "MIZ": "36",
    "URA": "42",
    "FUN": "43",
    "OOI": "44",
    "KAW": "45",
    "KNZ": "46",
    "KSM": "47",
    "NGY": "48",
    "SON": "50",
    "HIM": "51",
    "KOC": "54",
    "SAG": "55",
}
NAMES = {
    "MON": "門別",
    "MOR": "盛岡",
    "MIZ": "水沢",
    "URA": "浦和",
    "FUN": "船橋",
    "OOI": "大井",
    "KAW": "川崎",
    "KNZ": "金沢",
    "KSM": "笠松",
    "NGY": "名古屋",
    "SON": "園田",
    "HIM": "姫路",
    "KOC": "高知",
    "SAG": "佐賀",
}
LABELS = {
    "単勝": "WIN",
    "複勝": "PLACE",
    "枠連": "BRACKET_QUINELLA",
    "枠複": "BRACKET_QUINELLA",
    "馬連": "QUINELLA",
    "馬複": "QUINELLA",
    "ワイド": "QUINELLA_PLACE",
    "枠単": "BRACKET_EXACTA",
    "馬単": "FORECAST",
    "3連複": "TRIO",
    "3連単": "TIERCE",
}


def test_nar_yaml_has_the_spec_venues_pools_and_http_block():
    cfg = load_country_yaml("japan", "nar")
    assert cfg["timezone"] == "Asia/Tokyo"

    venues = cfg["venues"]
    assert {code: str(v["number"]) for code, v in venues.items()} == VENUES
    assert {code: v["name"] for code, v in venues.items() if code in NAMES} == NAMES
    assert venues["OBI"]["name"].startswith("帯広")

    label_to_pool = {lab: code for code, p in cfg["pools"].items() for lab in p["labels"]}
    assert {lab: label_to_pool.get(lab) for lab in LABELS} == LABELS

    http = cfg["http"]
    slow = {"requests_per_second": 0.1, "jitter": 0.0}
    assert http["hosts"]["default"] == slow
    assert http["hosts"]["nar.netkeiba.com"] == slow
    assert http["max_retries"] == 2
    assert http["circuit_breaker"] == {"stop_on_status": [403, 429], "max_consecutive_failures": 5}
    # the block is the Fetcher's own format (no request is made)
    with Fetcher("nar", http={**http, "user_agent": "horsetrack-tests/1.0"}):
        pass
