"""The tier-2 JRADB odds pages (ticket #130): 枠連, 馬単, 3連複, 3連単 and WIN5.

Contract pinned here (the coder follows it):
- `horsetrack.countries.japan.jra.exotic_pages` is pure (no network), beside `official_pages`.
- `odds_rows(body, race_id, pool, *, snapshot_utc, fetched_utc) -> Iterable[dict]`: `body` is the
  page as fetched (cp932 bytes), `pool` one of the jra.yaml codes BRACKET_QUINELLA (枠連),
  FORECAST (馬単, exacta), TRIO (3連複), TIERCE (3連単, trifecta): one pool per page. The rows
  have the shape of `official_pages.odds_rows` rows, exactly: snapshot_utc, race_id,
  source_update_utc, fetched_utc, page_state, pool, combination, odds, status_raw (no `odds_hi`:
  these pools print one figure), and 発売票数 of the page's own pool becomes the "__pool__" row
  with `investment` = tickets x 100 yen. It may be a generator: the rows are consumed once.
- Combinations are runner (or bracket) numbers joined by "," (as "1,2" for 馬連):
  FORECAST and TIERCE keep the page's order (finishing order: "2,1" is not "1,2"); TRIO and
  BRACKET_QUINELLA are sorted ascending ("3,7,12", "1,8"), and a bracket pair may name one
  bracket twice ("1,1"). A combination naming a scratched runner (取消 / 除外) gets no row; a cell
  that is not a number gets a row with odds None and the text as status_raw.
- `win5_rows(body, *, snapshot_utc, fetched_utc) -> list[dict]`: one "__pool__" row, pool
  "WIN5", race_id `jra:<YYYY-MM-DD>:WIN5` (the page's date), odds None, `investment` = tickets x
  100 yen, `carryover` (yen, float), `legs` (the five designated races' race_ids, leg order),
  and the base keys (snapshot_utc, fetched_utc, source_update_utc) as on any odds page.
- Parsing an 18-runner 3連単 page (4,896 combinations) and appending its rows to the odds log
  peaks under 200 MB RSS (in a child process, counting its own memory only: child_memory.py).

Pages are synthetic (jradb_pages.legs / jradb_pages.win5). Their layout follows the 馬連 page the
real-page tests check (one table per leading leg, a row per last leg); no real 枠連 / 馬単 / 3連複
/ 3連単 / WIN5 page is saved yet. Expected values are written out or derived from a formula here,
never computed by the code under test.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timezone
from itertools import combinations, combinations_with_replacement, permutations

import child_memory
import jradb_pages as pages
import pytest

from horsetrack.countries.japan import odds_log

# isort: split
# The module under test gets a block of its own: while it is missing, it is the import error a
# red run reports, and the order lints the same before and after it exists.
from horsetrack.countries.japan.jra import exotic_pages as ep

UTC = timezone.utc
SAT = date(2026, 11, 28)
SUN = date(2026, 11, 29)
MEETING = "5回東京7日"
RACE_ID = "jra:2026-11-28:TOK:11"
NOW_STATE = "14時00分現在"
FINAL = "最終オッズ"
SNAP = datetime(2026, 11, 28, 5, 0, 15, tzinfo=UTC)
FETCHED = datetime(2026, 11, 28, 5, 0, 20, tzinfo=UTC)  # 14:00:20 in Tokyo
UPDATED = datetime(2026, 11, 28, 5, 0, tzinfo=UTC)  # 14時00分 on 28 Nov in Tokyo
POOL = "__pool__"
UNIT_STAKE = 100  # yen per ticket, jra.yaml
EXOTIC = ("BRACKET_QUINELLA", "FORECAST", "TRIO", "TIERCE")


def rows_of(body: bytes, pool: str, *, race_id=RACE_ID, snapshot=SNAP, fetched=FETCHED):
    return list(ep.odds_rows(body, race_id, pool, snapshot_utc=snapshot, fetched_utc=fetched))


def base(*, state=NOW_STATE, updated=UPDATED, race_id=RACE_ID, snapshot=SNAP, fetched=FETCHED):
    return {
        "snapshot_utc": snapshot,
        "race_id": race_id,
        "source_update_utc": updated,
        "fetched_utc": fetched,
        "page_state": state,
    }


def price_row(pool: str, combination: str, odds, status=None, **kw) -> dict:
    row = {**base(**kw), "pool": pool, "combination": combination, "odds": odds}
    return {**row, "status_raw": status}


def pool_row(pool: str, investment: float, **kw) -> dict:
    return {
        **base(**kw),
        "pool": pool,
        "combination": POOL,
        "odds": None,
        "status_raw": None,
        "investment": investment,
    }


def by_key(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r["pool"], r["combination"]))


def prices(rows: list[dict]) -> dict[str, float | None]:
    return {r["combination"]: r["odds"] for r in rows if r["combination"] != POOL}


# ---------------------------------------------------------------- every pool, every row

# pool -> (odds as laid out on the page, expected {combination: odds or (None, status_raw)},
#          votes table, expected investment)
CASES = {
    "FORECAST": (
        {
            (1, 2): 25.3,
            (2, 1): 31.8,
            (10, 9): 412.0,
            (9, 10): 388.5,
            (3, 12): 1530.0,
            (12, 3): "---.-",
        },
        {
            "1,2": 25.3,
            "2,1": 31.8,
            "10,9": 412.0,
            "9,10": 388.5,
            "3,12": 1530.0,
            "12,3": (None, "---.-"),
        },
        [("馬単", 123_456), ("馬連", 999)],
        12_345_600.0,
    ),
    "TRIO": (
        {(1, 2, 3): 18.4, (12, 3, 7): 2045.1, (10, 2, 9): 356.7, (4, 5, 6): 88.8},
        {"1,2,3": 18.4, "3,7,12": 2045.1, "2,9,10": 356.7, "4,5,6": 88.8},
        [("3連複", 234_567), ("3連単", 11)],
        23_456_700.0,
    ),
    "TIERCE": (
        {
            (1, 2, 3): 95.2,
            (1, 3, 2): 120.4,
            (2, 1, 3): 101.7,
            (2, 3, 1): 210.9,
            (3, 1, 2): 150.0,
            (3, 2, 1): 333.3,
            (12, 3, 7): 12345.6,
        },
        {
            "1,2,3": 95.2,
            "1,3,2": 120.4,
            "2,1,3": 101.7,
            "2,3,1": 210.9,
            "3,1,2": 150.0,
            "3,2,1": 333.3,
            "12,3,7": 12345.6,
        },
        [("3連単", 345_678)],
        34_567_800.0,
    ),
    "BRACKET_QUINELLA": (
        {(1, 1): 45.6, (1, 2): 12.3, (8, 1): 77.7, (3, 3): 210.0, (7, 8): 9.9},
        {"1,1": 45.6, "1,2": 12.3, "1,8": 77.7, "3,3": 210.0, "7,8": 9.9},
        [("総票数合計", 99_999), ("枠連", 4_321)],
        432_100.0,
    ),
}


def expected_rows(pool: str, **kw) -> list[dict]:
    _, want, _, investment = CASES[pool]
    out = []
    for combination, value in want.items():
        odds, status = value if isinstance(value, tuple) else (value, None)
        out.append(price_row(pool, combination, odds, status, **kw))
    return out + [pool_row(pool, investment, **kw)]


@pytest.mark.parametrize("pool", list(CASES))
def test_each_tier2_pool_parses_into_odds_log_rows(pool):
    odds, _, votes, _ = CASES[pool]
    race_no = 3 + EXOTIC.index(pool)
    race_id = f"jra:2026-11-28:TOK:{race_no:02d}"
    live = pages.legs(SAT, MEETING, race_no, odds, state=NOW_STATE, votes=votes)
    assert by_key(rows_of(live, pool, race_id=race_id)) == by_key(
        expected_rows(pool, race_id=race_id)
    )

    # the final board: no clock on the page, so no source update; the caller's times as given
    snapshot = datetime(2026, 11, 28, 7, 10, 0, tzinfo=UTC)
    fetched = datetime(2026, 11, 28, 7, 10, 4, tzinfo=UTC)
    final = pages.legs(SAT, MEETING, race_no, odds, state=FINAL, votes=votes)
    kw = dict(race_id=race_id, state=FINAL, updated=None, snapshot=snapshot, fetched=fetched)
    got = rows_of(final, pool, race_id=race_id, snapshot=snapshot, fetched=fetched)
    assert by_key(got) == by_key(expected_rows(pool, **kw))


def test_ordered_pools_keep_the_finishing_order_and_unordered_pools_sort():
    # the same runners laid out in every order: one row per order for 馬単 / 3連単, one row in
    # all for 3連複 / 枠連, whatever order the page lists it in
    for pool, legs, ordered in (
        ("FORECAST", (4, 11), True),
        ("TIERCE", (2, 9, 14), True),
        ("TRIO", (2, 9, 14), False),
        ("BRACKET_QUINELLA", (3, 6), False),
    ):
        orders = list(permutations(legs))
        for layout in orders:
            body = pages.legs(SAT, MEETING, 11, {layout: 42.5}, state=NOW_STATE)
            want = ",".join(map(str, layout if ordered else sorted(layout)))
            assert prices(rows_of(body, pool)) == {want: 42.5}, (pool, layout)
        if ordered:  # every order on one page: one row each, each with its own price
            everything = {order: 10.0 + i for i, order in enumerate(orders)}
            body = pages.legs(SAT, MEETING, 11, everything, state=NOW_STATE)
            got = prices(rows_of(body, pool))
            assert got == {",".join(map(str, o)): v for o, v in everything.items()}, pool


def test_bracket_quinella_has_every_bracket_pair_including_same_bracket_pairs():
    pairs = list(combinations_with_replacement(range(1, 9), 2))
    assert len(pairs) == 36
    odds = {(a, b): float(a * 10 + b) + 0.5 for a, b in pairs}
    body = pages.legs(SAT, MEETING, 11, odds, state=NOW_STATE, votes=[("枠連", 50)])
    rows = rows_of(body, "BRACKET_QUINELLA")

    assert prices(rows) == {f"{a},{b}": v for (a, b), v in odds.items()}
    for n in range(1, 9):
        assert prices(rows)[f"{n},{n}"] == float(n * 11) + 0.5, f"same-bracket pair {n}-{n}"
    assert {r["pool"] for r in rows} == {"BRACKET_QUINELLA"}
    assert [r["investment"] for r in rows if r["combination"] == POOL] == [5_000.0]


def test_a_combination_naming_a_scratched_runner_has_no_row():
    field = range(1, 9)
    scratched = {6: "取消", 3: "除外"}
    # pool, legs, ordered, combinations of the 6 starters
    for pool, legs, ordered, count in (
        ("FORECAST", 2, True, 6 * 5),
        ("TRIO", 3, False, 20),
        ("TIERCE", 3, True, 6 * 5 * 4),
    ):
        layouts = permutations(field, legs) if ordered else combinations(field, legs)
        odds: dict[tuple[int, ...], object] = {}
        want = {}
        for combo in layouts:
            word = next((scratched[n] for n in combo if n in scratched), None)
            value = float("".join(f"{n}" for n in combo)) / 10
            odds[combo] = word or value
            if word is None:
                want[",".join(map(str, combo))] = value
        body = pages.legs(SAT, MEETING, 11, odds, state=NOW_STATE)
        rows = rows_of(body, pool)
        assert len(want) == count, pool
        assert prices(rows) == want, pool
        assert not [r for r in rows if {"3", "6"} & set(r["combination"].split(","))], pool
        assert all(r["status_raw"] is None for r in rows), pool


# ---------------------------------------------------------------- the 18-runner trifecta page

RUNNERS_18 = range(1, 19)


def trifecta_odds(a: int, b: int, c: int) -> float:
    """A distinct price per ordered triple, readable back to it: (3, 12, 7) -> 3120.7."""
    return (a * 10_000 + b * 100 + c) / 10


def trifecta_18() -> tuple[bytes, dict[str, float]]:
    """The 3連単 page of an 18-runner field, every ordered triple once, tables in a shuffled
    order; and the prices it prints."""
    triples = list(permutations(RUNNERS_18, 3))
    random.Random(130).shuffle(triples)
    odds = {t: trifecta_odds(*t) for t in triples}
    body = pages.legs(SAT, MEETING, 11, odds, state=NOW_STATE, votes=[("3連単", 9_876_543)])
    return body, {",".join(map(str, t)): v for t, v in odds.items()}


def test_an_18_runner_trifecta_page_has_all_4896_ordered_combinations():
    body, want = trifecta_18()
    assert len(want) == 18 * 17 * 16 == 4_896
    rows = rows_of(body, "TIERCE")

    assert len(rows) == 4_896 + 1
    assert prices(rows) == want
    assert [r["investment"] for r in rows if r["combination"] == POOL] == [987_654_300.0]
    assert {r["pool"] for r in rows} == {"TIERCE"}
    assert {r["snapshot_utc"] for r in rows} == {SNAP}
    assert {r["race_id"] for r in rows} == {RACE_ID}
    # finishing order kept: 4,896 distinct strings, most of them not sorted
    unsorted = [c for c in want if c.split(",") != sorted(c.split(","), key=int)]
    assert len(unsorted) == 4_896 - 816  # all but the C(18, 3) ascending ones


PEAK_LIMIT_BYTES = 200 * 1000 * 1000  # 200 MB
# runs after child_memory.PROBE, which defines peak_bytes() and START_PEAK
CHILD = r"""
import json
from datetime import datetime, timezone
from pathlib import Path

from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.jra import exotic_pages as ep

body = Path(sys.argv[1]).read_bytes()
at = datetime(2026, 11, 28, 5, 0, 15, tzinfo=timezone.utc)
rows = ep.odds_rows(body, "jra:2026-11-28:TOK:11", "TIERCE", snapshot_utc=at, fetched_utc=at)
written = odds_log.append_capture(Path(sys.argv[2]), rows)
print(json.dumps({"written": written, "peak": peak_bytes(), "start": START_PEAK}))
"""


def test_parsing_an_18_runner_trifecta_page_peaks_under_200_mb(tmp_path):
    body, want = trifecta_18()
    page_file, log_file = tmp_path / "tierce.html", tmp_path / "TOK.jsonl.gz"
    page_file.write_bytes(body)

    # the child's own peak, not the pytest parent's (see child_memory.py)
    out = child_memory.run_child(CHILD, str(page_file), str(log_file))
    assert out["written"] == 4_896 + 1
    peak = out["peak"]
    assert peak < PEAK_LIMIT_BYTES, f"peak RSS {peak / 1e6:.1f} MB"
    logged = odds_log.read_rows(log_file)
    assert {r["combination"]: r["odds"] for r in logged if r["combination"] != POOL} == want


# ---------------------------------------------------------------- WIN5

WIN5_CASES = [
    # (day, legs as printed, tickets, carryover yen, page clock, fetched, expected source update,
    #  expected leg race ids)
    (
        SUN,
        [("東京", 9), ("京都", 10), ("東京", 10), ("京都", 11), ("東京", 11)],
        4_567_890,
        312_450_000,
        "13時05分現在",
        datetime(2026, 11, 29, 4, 5, 30, tzinfo=UTC),
        datetime(2026, 11, 29, 4, 5, tzinfo=UTC),
        [
            "jra:2026-11-29:TOK:09",
            "jra:2026-11-29:KYO:10",
            "jra:2026-11-29:TOK:10",
            "jra:2026-11-29:KYO:11",
            "jra:2026-11-29:TOK:11",
        ],
    ),
    (
        date(2026, 12, 6),
        [("中山", 10), ("阪神", 9), ("中京", 11), ("中山", 11), ("阪神", 12)],
        1_234,
        0,
        "9時41分現在",
        datetime(2026, 12, 6, 0, 41, 50, tzinfo=UTC),
        datetime(2026, 12, 6, 0, 41, tzinfo=UTC),
        [
            "jra:2026-12-06:NAK:10",
            "jra:2026-12-06:HSN:09",
            "jra:2026-12-06:CKY:11",
            "jra:2026-12-06:NAK:11",
            "jra:2026-12-06:HSN:12",
        ],
    ),
]


@pytest.mark.parametrize("case", WIN5_CASES, ids=["carryover", "no-carryover"])
def test_win5_page_gives_the_pool_the_carryover_and_the_five_designated_races(case):
    day, legs, tickets, carryover, clock, fetched, updated, race_ids = case
    snapshot = fetched.replace(second=0)
    body = pages.win5(day, legs, state=clock, tickets=tickets, carryover_yen=carryover)

    rows = ep.win5_rows(body, snapshot_utc=snapshot, fetched_utc=fetched)

    assert len(rows) == 1
    (row,) = rows
    want = {
        "race_id": f"jra:{day.isoformat()}:WIN5",
        "pool": "WIN5",
        "combination": POOL,
        "odds": None,
        "investment": float(tickets * UNIT_STAKE),
        "carryover": float(carryover),
        "legs": race_ids,
        "snapshot_utc": snapshot,
        "fetched_utc": fetched,
        "source_update_utc": updated,
    }
    assert {key: row.get(key) for key in want} == want
