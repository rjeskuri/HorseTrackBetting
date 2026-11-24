"""The JRA odds parser against the real JRADB pages (ticket #31): the synthetic pages of
`jradb_pages.py` have not drifted from the structure the site really serves.

Every test is marked `local` and uses `real_fixtures` (tests/conftest.py): without
$HORSETRACK_DATA_DIR/fixtures the file is skipped. The pages are `fixtures/jra/official_*.html`:
Kyoto (京都), Sun 15 Nov 2026 (5回京都4日), fetched on Wed 18 Nov 2026, when the menus still showed
the weekend of 14-15 Nov. Race 10: 10 declared, No 6 scratched (取消), final odds (最終オッズ).
Each file starts with a comment naming the path and the POSTed cname it was fetched with; it is
re-encoded to cp932 here, as on the wire, and served through the JRADB double of the capture tests.

Expectations are ported from HKTrack's test_odds.py, test_official_nav.py and the odds half of
test_live_cli_jra.py (copied off the pages by hand there). Nothing real is saved in the repository
beyond the figures asserted below.

`test_synthetic_pages_share_the_real_selectors` takes the selectors from the parser itself
(`official_pages.SELECTORS`, by page kind); the guards that the parser reads nothing outside it
are in test_official_pages.py, unmarked, so CI runs them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import pytest

# isort: split
import jradb_double as dbl
import jradb_pages as pages

from horsetrack.countries.japan.jra import official_pages as op
from horsetrack.countries.japan.jra.capture import (
    NotListed,
    capture_meeting,
    discover_venues,
    read_post_times,
)

pytestmark = pytest.mark.local

UTC = timezone.utc
KYOTO_SUN = date(2026, 11, 15)
RACE_ID = "jra:2026-11-15:KYO:10"
ALL_POOLS = ["WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE"]
SCRATCHED = 6
SNAP = datetime(2026, 11, 18, 12, 4, 30, tzinfo=UTC)
FETCHED = datetime(2026, 11, 18, 12, 4, 36, tzinfo=UTC)
FINAL = "最終オッズ"
UNIT_STAKE = 100  # yen per ticket, jra.yaml

ACCESS_D, ACCESS_O = pages.CARD_PATH, pages.ODDS_PATH
CARD_MENU, ODDS_MENU = dbl.CARD_MENU, dbl.ODDS_MENU
CARD_LIST = (ACCESS_D, "pw01drl10082026050420261115/82")
ODDS_LIST = (ACCESS_O, "pw15orl10082026050420261115/FC")
CARD_R10 = (ACCESS_D, "pw01dde1008202605041020261115/76")
WINPLACE = (ACCESS_O, "pw151ou1008202605041020261115Z/74")
QUINELLA = (ACCESS_O, "pw154ou1008202605041020261115Z/00")
WIDE = (ACCESS_O, "pw155ou1008202605041020261115Z/84")
REAL_PAGES = {
    "official_card_2026-11-15_KYO_R10.html": CARD_R10,
    "official_card_menu_2026-11-18.html": CARD_MENU,
    "official_card_racelist_2026-11-15_KYO.html": CARD_LIST,
    "official_odds_menu_2026-11-18.html": ODDS_MENU,
    "official_odds_quinella_2026-11-15_KYO_R10.html": QUINELLA,
    "official_odds_racelist_2026-11-15_KYO.html": ODDS_LIST,
    "official_odds_wide_2026-11-15_KYO_R10.html": WIDE,
    "official_odds_winplace_2026-11-15_KYO_R10.html": WINPLACE,
}

# Every meeting on both menus: (date, venue) -> (出馬表 race list cname, オッズ race list cname).
MEETINGS = {
    ("2026-11-14", "TOK"): ("pw01drl10052026050320261114/B6", "pw15orl10052026050320261114/30"),
    ("2026-11-14", "KYO"): ("pw01drl10082026050320261114/94", "pw15orl10082026050320261114/0E"),
    ("2026-11-14", "FKS"): ("pw01drl10032026030320261114/C0", "pw15orl10032026030320261114/3A"),
    ("2026-11-15", "TOK"): ("pw01drl10052026050420261115/A4", "pw15orl10052026050420261115/1E"),
    ("2026-11-15", "KYO"): (CARD_LIST[1], ODDS_LIST[1]),
    ("2026-11-15", "FKS"): ("pw01drl10032026030420261115/AE", "pw15orl10032026030420261115/28"),
}

# Race 10 final odds, copied off the pages. No 6 is scratched: no price, no pair naming it.
WIN = {1: 37.8, 2: 82.1, 3: 30.7, 4: 12.5, 5: 32.8, 7: 5.8, 8: 3.5, 9: 1.7, 10: 24.5}
PLACE = {
    1: (3.0, 7.3), 2: (6.4, 16.0), 3: (2.9, 7.1), 4: (1.6, 3.5), 5: (3.3, 8.0),
    7: (1.5, 3.2), 8: (1.1, 1.8), 9: (1.1, 1.1), 10: (2.4, 5.5),
}  # fmt: skip
PAIR_QUINELLA = {
    (1, 2): 326.8, (1, 3): 220.9, (1, 4): 97.4, (1, 5): 239.5, (1, 7): 80.7, (1, 8): 50.0,
    (1, 9): 25.3, (1, 10): 209.4,
    (2, 3): 444.0, (2, 4): 232.0, (2, 5): 469.1, (2, 7): 161.1, (2, 8): 95.0, (2, 9): 63.6,
    (2, 10): 360.6,
    (3, 4): 82.6, (3, 5): 232.6, (3, 7): 76.9, (3, 8): 39.6, (3, 9): 25.7, (3, 10): 219.7,
    (4, 5): 95.4, (4, 7): 31.9, (4, 8): 18.7, (4, 9): 9.7, (4, 10): 85.3,
    (5, 7): 87.9, (5, 8): 54.5, (5, 9): 30.8, (5, 10): 197.9,
    (7, 8): 13.0, (7, 9): 6.0, (7, 10): 73.5, (8, 9): 2.2, (8, 10): 40.9, (9, 10): 20.9,
}  # fmt: skip
PAIR_WIDE = {
    (1, 2): (63.5, 65.9), (1, 3): (40.8, 43.2), (1, 4): (17.4, 19.7), (1, 5): (42.5, 44.8),
    (1, 7): (21.5, 24.2), (1, 8): (8.6, 11.4), (1, 9): (6.5, 8.5), (1, 10): (39.7, 42.4),
    (2, 3): (86.0, 89.3), (2, 4): (42.7, 47.0), (2, 5): (87.4, 90.1), (2, 7): (36.6, 40.3),
    (2, 8): (21.2, 27.5), (2, 9): (14.3, 18.4), (2, 10): (65.0, 68.2),
    (3, 4): (18.2, 20.6), (3, 5): (35.7, 37.6), (3, 7): (18.8, 21.2), (3, 8): (6.7, 8.8),
    (3, 9): (6.4, 8.4), (3, 10): (39.6, 42.4),
    (4, 5): (15.8, 17.7), (4, 7): (8.6, 10.3), (4, 8): (3.1, 4.3), (4, 9): (2.4, 3.2),
    (4, 10): (15.3, 17.5),
    (5, 7): (21.5, 24.1), (5, 8): (9.7, 12.9), (5, 9): (7.7, 10.0), (5, 10): (33.5, 35.7),
    (7, 8): (3.4, 4.8), (7, 9): (2.4, 3.2), (7, 10): (16.7, 19.0), (8, 9): (1.4, 1.5),
    (8, 10): (6.7, 9.0), (9, 10): (5.1, 6.7),
}  # fmt: skip
# 発売票数 printed on the pages (tickets sold); the pool total is tickets x the 100-yen stake.
TICKETS = {"WIN": 716_227, "PLACE": 648_500, "QUINELLA": 877_724, "QUINELLA_PLACE": 764_551}

# Race post times printed on the 出馬表 race list (JST), as (hour, minute); UTC is 9 hours earlier.
POST_JST = {
    1: (9, 55), 2: (10, 25), 3: (10, 55), 4: (11, 25), 5: (12, 15), 6: (12, 50),
    7: (13, 20), 8: (13, 50), 9: (14, 20), 10: (15, 1), 11: (15, 40), 12: (16, 25),
}  # fmt: skip

_HEADER = re.compile(
    r"<!-- source: POST https://www\.jra\.go\.jp(?P<path>/JRADB/[A-Za-z0-9]+\.html) "
    r"body cname=(?P<cname>\S+) \|"
)
_PRINTED_LINKS = (
    re.compile(r"doAction\(\s*'(/JRADB/[A-Za-z0-9]+\.html)'\s*,\s*'([^']+)'\s*\)"),
    re.compile(r'href="(/JRADB/[A-Za-z0-9]+\.html)\?CNAME=([^"&]+)"'),
)


# ---------------------------------------------------------------- the real pages


@pytest.fixture
def real_pages(real_fixtures: Path) -> dict[tuple[str, str], bytes]:
    """The eight official pages, as cp932 bytes keyed by the (path, cname) they were POSTed to."""
    out = {}
    for name, key in REAL_PAGES.items():
        text = (real_fixtures / "jra" / name).read_text(encoding="utf-8")
        head = _HEADER.match(text)
        assert head is not None, f"{name}: no source comment on its first line"
        assert (head["path"], head["cname"]) == key, name
        out[key] = text.encode("cp932")
    return out


@pytest.fixture
def isolated(real_pages, tmp_path, monkeypatch) -> dict[tuple[str, str], bytes]:
    """`real_pages`, with raw pages going under tmp_path and no real `.env` or variable supplying
    a setting. Built after `real_fixtures` has read the real data directory."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("HORSETRACK_USER_AGENT", raising=False)
    return real_pages


def real_site(real_pages: Mapping[tuple[str, str], bytes]):
    """The JRADB double serving only the real pages: any other cname gets the no-date error page."""
    site = dbl.JradbSite(dict(real_pages))
    return site, dbl.make_fetcher(site)


def doc(real_pages, key):
    return op.soup(real_pages[key])


def price_tuples(pool: str, prices: Mapping, *, scratched: tuple[str, ...] = ()) -> list[tuple]:
    """(race_id, pool, combination, odds, odds_hi) as `dbl.price_rows` prints them."""
    out = [(RACE_ID, pool, c, None, None) for c in scratched]
    for key, price in prices.items():
        combination = ",".join(map(str, key)) if isinstance(key, tuple) else str(key)
        low, high = price if isinstance(price, tuple) else (price, None)
        out.append((RACE_ID, pool, combination, low, high))
    return out


def expected_price_rows(pools: list[str]) -> list[tuple]:
    out = []
    out += price_tuples("WIN", WIN, scratched=(str(SCRATCHED),)) if "WIN" in pools else []
    out += price_tuples("PLACE", PLACE, scratched=(str(SCRATCHED),)) if "PLACE" in pools else []
    out += price_tuples("QUINELLA", PAIR_QUINELLA) if "QUINELLA" in pools else []
    out += price_tuples("QUINELLA_PLACE", PAIR_WIDE) if "QUINELLA_PLACE" in pools else []
    return sorted(out, key=lambda r: tuple("" if v is None else str(v) for v in r))


def pool_total(rows: list[dict], pool: str) -> list[dict]:
    return [r for r in rows if r["pool"] == pool and r["combination"] == op.POOL_TOTAL]


def odds_of(rows: list[dict], pool: str) -> dict[str, tuple]:
    """combination -> (odds, odds_hi or None, status_raw) of the price rows of `pool`."""
    return {
        r["combination"]: (r["odds"], r.get("odds_hi"), r["status_raw"])
        for r in rows
        if r["pool"] == pool and r["combination"] != op.POOL_TOTAL
    }


def parse(real_pages, key, pools, *, fetched=FETCHED, snapshot=SNAP) -> list[dict]:
    return op.odds_rows(
        doc(real_pages, key), RACE_ID, pools, snapshot_utc=snapshot, fetched_utc=fetched
    )


# ---------------------------------------------------------------- the odds pages


def test_real_win_place_page(real_pages):
    """10 declared, No 6 scratched (取消), Place ranged (min-max), Win one figure."""
    page = doc(real_pages, WINPLACE)
    assert op.page_meeting(page) == (KYOTO_SUN, "KYO")
    assert op.race_number(page) == 10
    assert len(page.select("#odds_list td.num")) == 10, "ten horses declared"
    assert "取消" in real_pages[WINPLACE].decode("cp932"), "the page prints the scratching"

    rows = parse(real_pages, WINPLACE, ["WIN", "PLACE"])
    for row in rows:
        assert row["race_id"] == RACE_ID
        assert row["snapshot_utc"] == SNAP and row["fetched_utc"] == FETCHED
        assert row["page_state"] == FINAL
        assert row["source_update_utc"] is None, "最終オッズ prints no clock"

    win, place = odds_of(rows, "WIN"), odds_of(rows, "PLACE")
    assert set(win) == set(place) == {str(n) for n in range(1, 11)}
    assert {c: v for c, v in win.items() if v[2] is None} == {
        str(n): (price, None, None) for n, price in WIN.items()
    }, "Win shows one figure and no odds_hi"
    assert {c: v for c, v in place.items() if v[2] is None} == {
        str(n): (low, high, None) for n, (low, high) in PLACE.items()
    }, "Place prints a range: odds = min, odds_hi = max"
    assert all(low <= high for low, high, _ in (v for v in place.values() if v[2] is None))
    assert win[str(SCRATCHED)] == (None, None, op.SCRATCHED)
    assert place[str(SCRATCHED)] == (None, None, op.SCRATCHED)
    assert all("odds_hi" not in r for r in rows if r["pool"] == "WIN")

    for pool in ("WIN", "PLACE"):
        (total,) = pool_total(rows, pool)
        assert total["investment"] == float(TICKETS[pool] * UNIT_STAKE)
        assert total["odds"] is None and total["status_raw"] is None
    assert len(rows) == 10 + 10 + 2


@pytest.mark.parametrize("pools", [["WIN"], ["PLACE"]])
def test_real_win_place_page_returns_only_the_asked_pools(real_pages, pools):
    rows = parse(real_pages, WINPLACE, pools)
    assert {r["pool"] for r in rows} == set(pools)
    assert len(rows) == 10 + 1
    other = "PLACE" if pools == ["WIN"] else "WIN"
    assert not [r for r in rows if r["pool"] == other]


def test_real_quinella_and_wide_pages(real_pages):
    """36 pairs among the nine runners, none naming No 6; Quinella one figure, Wide a range."""
    starters = [n for n in range(1, 11) if n != SCRATCHED]
    pairs = {f"{a},{b}" for a, b in combinations(starters, 2)}
    assert len(pairs) == 36
    assert {f"{a},{b}" for a, b in PAIR_QUINELLA} == pairs == {f"{a},{b}" for a, b in PAIR_WIDE}

    for key, pool, table in (
        (QUINELLA, "QUINELLA", PAIR_QUINELLA),
        (WIDE, "QUINELLA_PLACE", PAIR_WIDE),
    ):
        page = doc(real_pages, key)
        assert op.page_meeting(page) == (KYOTO_SUN, "KYO")
        assert op.race_number(page) == 10
        rows = parse(real_pages, key, [pool])
        assert {r["pool"] for r in rows} == {pool}
        assert all(r["page_state"] == FINAL and r["source_update_utc"] is None for r in rows)
        prices = odds_of(rows, pool)
        assert set(prices) == pairs, "every pair of starters, none naming the scratched horse"
        assert not [c for c in prices if str(SCRATCHED) in c.split(",")]
        assert all(status is None for *_, status in prices.values())
        if pool == "QUINELLA":
            assert {c: v[:2] for c, v in prices.items()} == {
                f"{a},{b}": (price, None) for (a, b), price in table.items()
            }
            assert all("odds_hi" not in r for r in rows)
        else:
            assert {c: v[:2] for c, v in prices.items()} == {
                f"{a},{b}": (low, high) for (a, b), (low, high) in table.items()
            }
            assert all(low <= high for low, high, _ in prices.values())
        (total,) = pool_total(rows, pool)
        assert total["investment"] == float(TICKETS[pool] * UNIT_STAKE)
        assert len(rows) == 36 + 1


# ---------------------------------------------------------------- menus and race lists


@pytest.mark.parametrize(
    ("menu", "path", "column"),
    [(CARD_MENU, ACCESS_D, 0), (ODDS_MENU, ACCESS_O, 1)],
    ids=["card-menu", "odds-menu"],
)
def test_real_menus_list_the_six_meetings_of_14_and_15_nov(real_pages, menu, path, column):
    page = doc(real_pages, menu)
    for day in (date(2026, 11, 14), date(2026, 11, 15)):
        expected = {
            venue: op.Link(path, codes[column])
            for (shown, venue), codes in MEETINGS.items()
            if shown == day.isoformat()
        }
        assert set(expected) == {"TOK", "KYO", "FKS"}
        assert op.meetings_on(page, day, path) == expected
    # The menu was fetched on Wed 18 Nov: no heading for any other day.
    for other in (date(2026, 11, 13), date(2026, 11, 16), date(2026, 11, 18), date(2026, 11, 21)):
        assert op.meetings_on(page, other, path) == {}
    # The other menu's path never matches this menu's links.
    wrong = ACCESS_O if path == ACCESS_D else ACCESS_D
    assert op.meetings_on(page, date(2026, 11, 15), wrong) == {}


def test_real_menus_are_read_through_the_fetcher(isolated):
    site, fetcher = real_site(isolated)
    with fetcher:
        assert sorted(discover_venues(fetcher, date(2026, 11, 14))) == ["FKS", "KYO", "TOK"]
        assert sorted(discover_venues(fetcher, date(2026, 11, 15))) == ["FKS", "KYO", "TOK"]
        assert discover_venues(fetcher, date(2026, 11, 21)) == []
    assert site.links() == [CARD_MENU] * 3
    assert all(s.served for s in site.seen)


def test_real_card_race_list_post_times(real_pages):
    """Race 1 at 9時55分 is 00:55Z; every race's clock is read, JST -> UTC."""
    page = doc(real_pages, CARD_LIST)
    assert op.page_meeting(page) == (KYOTO_SUN, "KYO")
    assert [n for n, _ in op.race_rows(page)] == list(range(1, 13))
    got = op.post_times(page, KYOTO_SUN)
    assert got[1] == datetime(2026, 11, 15, 0, 55, tzinfo=UTC)
    expected = {
        n: datetime(2026, 11, 15, h, m, tzinfo=UTC) - timedelta(hours=9)
        for n, (h, m) in POST_JST.items()
    }
    assert got == expected
    assert all(t.utcoffset() == timedelta(0) for t in got.values())
    # The same page asked for another day moves every post time with it.
    assert op.post_times(page, date(2026, 11, 14))[10] == datetime(2026, 11, 14, 6, 1, tzinfo=UTC)


def test_real_card_post_times_through_the_fetcher(isolated):
    site, fetcher = real_site(isolated)
    with fetcher:
        got = read_post_times(fetcher, KYOTO_SUN, "KYO")
    assert got[10] == datetime(2026, 11, 15, 6, 1, tzinfo=UTC)
    assert len(got) == 12
    assert site.links() == [CARD_MENU, CARD_LIST]
    assert all(s.served for s in site.seen)


def test_real_odds_race_list_offers_the_four_pools_of_race_10(real_pages):
    page = doc(real_pages, ODDS_LIST)
    assert op.page_meeting(page) == (KYOTO_SUN, "KYO")
    rows = dict(op.race_rows(page))
    assert sorted(rows) == list(range(1, 13))
    links = op.pool_links(rows[10])
    assert {pool: link for pool, link in links.items() if pool in ALL_POOLS} == {
        "WIN": op.Link(*WINPLACE),
        "PLACE": op.Link(*WINPLACE),
        "QUINELLA": op.Link(*QUINELLA),
        "QUINELLA_PLACE": op.Link(*WIDE),
    }
    for number, row in rows.items():
        offered = op.pool_links(row)
        assert {"WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE"} <= set(offered), number


# ---------------------------------------------------------------- capture through the double


@pytest.mark.parametrize(
    ("pools", "pages_asked"),
    [
        (["WIN"], [WINPLACE]),
        (["PLACE", "QUINELLA"], [WINPLACE, QUINELLA]),
        (["QUINELLA_PLACE"], [WIDE]),
        (ALL_POOLS, [WINPLACE, QUINELLA, WIDE]),
    ],
)
def test_real_capture_of_race_10_matches_the_hand_checked_odds(isolated, pools, pages_asked):
    site, fetcher = real_site(isolated)
    snapshot = datetime(2026, 11, 18, 12, 10, tzinfo=UTC)
    with fetcher:
        cap = capture_meeting(fetcher, KYOTO_SUN, "KYO", pools, races=[10], now=lambda: snapshot)
    assert cap.requests == 2 + len(pages_asked)
    assert site.links() == [ODDS_MENU, ODDS_LIST, *pages_asked]
    assert all(s.served for s in site.seen), "a request went to a code no real page linked"
    assert dbl.price_rows(cap.rows) == expected_price_rows(pools)
    assert {r["snapshot_utc"] for r in cap.rows} == {snapshot} == {cap.snapshot_utc}
    assert {r["race_id"] for r in cap.rows} == {RACE_ID}
    totals = {r["pool"]: r["investment"] for r in cap.rows if r["combination"] == op.POOL_TOTAL}
    assert totals == {p: float(TICKETS[p] * UNIT_STAKE) for p in pools}


def test_real_navigation_follows_printed_codes_not_computed_ones(isolated):
    """Every checksum on every real page is replaced by another value: only code that reads the
    codes off the pages gets to the odds and the post times."""
    scrambled, renamed = dbl.scramble(isolated)
    changed = {old: new for old, new in renamed.items() if old not in dbl.ENTRY_CNAMES}
    assert len(changed) > 100 and all(new != old for old, new in changed.items())
    site, fetcher = real_site(scrambled)
    with fetcher:
        cap = capture_meeting(fetcher, KYOTO_SUN, "KYO", ALL_POOLS, races=[10])
        times = read_post_times(fetcher, KYOTO_SUN, "KYO")
    assert dbl.price_rows(cap.rows) == expected_price_rows(ALL_POOLS)
    assert times[1] == datetime(2026, 11, 15, 0, 55, tzinfo=UTC)

    linked = {CARD_MENU, ODDS_MENU}
    for seen in site.seen:
        assert seen.method == "POST" and seen.served, seen
        assert seen.link in linked, f"{seen.link}: not printed on any page fetched before"
        body = scrambled[seen.link].decode("cp932")
        linked |= {(m[1], m[2]) for pattern in _PRINTED_LINKS for m in pattern.finditer(body)}
    assert not {s.cname for s in site.seen} & set(changed), "an unscrambled checksum was sent"
    sent = {s.link for s in site.seen}
    for path, cname in (CARD_LIST, ODDS_LIST, WINPLACE, QUINELLA, WIDE):
        assert (path, renamed[cname]) in sent


@pytest.mark.parametrize(
    ("day", "venue"),
    [(date(2026, 11, 21), "KYO"), (date(2026, 11, 15), "HSN"), (date(2026, 11, 14), "NAK")],
)
def test_real_meeting_the_menu_does_not_list_is_not_listed(isolated, day, venue):
    """On Wed 18 Nov Sat 21 Nov is not published yet, and 阪神 / 中山 are not racing on 14-15."""
    site, fetcher = real_site(isolated)
    with fetcher, pytest.raises(NotListed):
        capture_meeting(fetcher, day, venue, ALL_POOLS)
    assert site.links() == [ODDS_MENU], "nothing but the menu was requested"


# ---------------------------------------------------------------- the synthetic pages

REAL_OF_KIND = {
    "card_menu": CARD_MENU,
    "odds_menu": ODDS_MENU,
    "card_race_list": CARD_LIST,
    "odds_race_list": ODDS_LIST,
    "win_place": WINPLACE,
    "quinella": QUINELLA,
    "wide": WIDE,
}


def test_synthetic_pages_share_the_real_selectors(real_pages):
    """Every selector the parser declares in `official_pages.SELECTORS` finds a node in each
    synthetic builder's page and in the matching real page: neither has drifted from the other.
    (That the parser reads nothing outside SELECTORS is checked in test_official_pages.py.)"""
    selectors = op.SELECTORS
    assert set(selectors) == set(pages.PAGE_KINDS) == set(REAL_OF_KIND)
    synthetic = pages.sample_pages()
    missing = []
    for kind, declared in selectors.items():
        assert declared, f"{kind}: no selector declared"
        for label, page in (
            ("synthetic", op.soup(synthetic[kind])),
            ("real", doc(real_pages, REAL_OF_KIND[kind])),
        ):
            missing += [(kind, label, sel) for sel in declared if page.select_one(sel) is None]
            if kind.endswith("_menu"):
                # The meeting links sit in the heading's parent, as a doAction or href link.
                heading = page.select_one("h3")
                links = [op.link_of(a) for a in heading.parent.find_all("a")] if heading else []
                if not any(links):
                    missing.append((kind, label, "h3 parent > a with a JRADB link"))
    assert not missing, missing
