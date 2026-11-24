"""The live-board feed built from a day's snapshot logs (ticket #40).

Contract pinned here (`horsetrack.countries.japan.board.feed.build_board`):
- `build_board(race_date: date, *, venues=None, now=utc_now, morning_from='07:00') -> dict`, read
  from `$HORSETRACK_DATA_DIR/odds_snapshots/jra/<date>/<VENUE>.jsonl.gz` (every venue logged that
  day, or only `venues`); nothing is requested over the network.
- Shape: {race_date, generated_utc, venues: [{venue, races: [{race_no, race_id, post_utc,
  last_capture_utc, pools: {POOL: {investment, rows: [{combination, odds, odds_hi, status,
  morning_odds, move_morning_pct, t20_odds, move_t20_pct}]}}}]}]}. The feed is JSON-ready:
  `race_date` is ISO date text, timestamps are ISO UTC text (compared here as instants).
- Current odds: the latest row of the combination by snapshot_utc (log order does not matter).
- Morning reference: the earliest price row of the combination captured on race day at or after
  `morning_from` Japan time. T-20 reference: the latest row at or before post - 20 min; null while
  now is before post - 20 min, and without a post time. move = current / reference - 1 in percent,
  one decimal, on `odds` (the low figure of a ranged pool); null without a reference.
- 'SCR' rows carry status 'SCR', no odds and no move. Pool `investment` is the latest `__pool__`
  figure; `__pool__` is never a row. Post times: the sidecar, else HKTrack's stored card, else null.
- No key anywhere names a probability, EV or bet.

All rows are synthetic and built here; HORSETRACK_DATA_DIR points at tmp_path.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterable, Iterator
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from horsetrack.core import paths
from horsetrack.countries.japan import odds_log

UTC = timezone.utc
JST = timezone(timedelta(hours=9))
DAY = date(2026, 11, 28)
POOL = odds_log.POOL_TOTAL

ROW_KEYS = {
    "combination",
    "odds",
    "odds_hi",
    "status",
    "morning_odds",
    "move_morning_pct",
    "t20_odds",
    "move_t20_pct",
}
RACE_KEYS = {"race_no", "race_id", "post_utc", "last_capture_utc", "pools"}


# ---------------------------------------------------------------- helpers


def jst(hh: int, mm: int, ss: int = 0, *, day: date = DAY) -> datetime:
    return datetime(day.year, day.month, day.day, hh, mm, ss, tzinfo=JST)


def rid(venue: str, no: int, day: date = DAY) -> str:
    return f"jra:{day.isoformat()}:{venue}:{no:02d}"


def price(race_id, pool, combination, odds, at, *, odds_hi=None) -> dict:
    """One price row in the HKTrack shape; odds None is a scratched runner ('SCR')."""
    out = {
        "snapshot_utc": at.astimezone(UTC).isoformat(),
        "race_id": race_id,
        "pool": pool,
        "combination": combination,
        "odds": odds,
        "status_raw": "SCR" if odds is None else f"{at.hour:02d}時{at.minute:02d}分現在",
        "source_update_utc": at.astimezone(UTC).isoformat(),
    }
    if odds_hi is not None:
        out["odds_hi"] = odds_hi
    return out


def total(race_id, pool, investment, at) -> dict:
    out = price(race_id, pool, POOL, None, at)
    out["status_raw"] = None
    if investment is not None:
        out["investment"] = investment
    return out


def write_log(venue: str, rows: Iterable[dict], day: date = DAY) -> Path:
    """One gzip member per capture (rows sharing a snapshot_utc), in the order first given, so a
    test can write captures out of time order."""
    captures: dict[str, list[dict]] = {}
    for row in rows:
        captures.setdefault(row["snapshot_utc"], []).append(row)
    path = paths.odds_snapshot_path("jra", day.isoformat(), venue)
    for capture in captures.values():
        odds_log.append_capture(path, capture)
    return path


def write_sidecar(venue: str, post_times: dict[int, datetime], day: date = DAY) -> None:
    log = paths.odds_snapshot_path("jra", day.isoformat(), venue)
    updated = datetime(day.year, day.month, day.day, 0, 0, tzinfo=UTC)
    odds_log.write_post_times(log.with_name(f"{venue}.races.json"), post_times, updated)


def write_card(venue: str, races: dict[int, str], day: date) -> None:
    """HKTrack's stored card: processed/jra/upcoming/jra_<date>_<VENUE>/races.parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder = paths.processed_dir("jra") / "upcoming" / f"jra_{day.isoformat()}_{venue}"
    folder.mkdir(parents=True, exist_ok=True)
    nos = sorted(races)
    table = pa.table(
        {
            "race_id": [rid(venue, no, day) for no in nos],
            "race_no": pa.array(nos, type=pa.int64()),
            "post_time_local": [races[no] for no in nos],
        }
    )
    pq.write_table(table, folder / "races.parquet")


def build(now: datetime, day: date = DAY, **kwargs) -> dict:
    from horsetrack.countries.japan.board import feed

    return feed.build_board(day, now=lambda: now, **kwargs)


def instant(text: Any) -> datetime | None:
    if text is None:
        return None
    assert isinstance(text, str), f"timestamp {text!r} is not ISO text"
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    assert value.tzinfo is not None, f"timestamp {text!r} has no offset"
    return value.astimezone(UTC)


def venue_of(board: dict, venue: str) -> dict:
    found = [v for v in board["venues"] if v["venue"] == venue]
    assert len(found) == 1, f"venue {venue} appears {len(found)} times"
    return found[0]


def race_of(board: dict, venue: str, no: int) -> dict:
    found = [r for r in venue_of(board, venue)["races"] if r["race_no"] == no]
    assert len(found) == 1, f"{venue} race {no} appears {len(found)} times"
    return found[0]


def line(race: dict, pool: str, combination: str) -> dict:
    found = [r for r in race["pools"][pool]["rows"] if r["combination"] == combination]
    assert len(found) == 1, f"{pool} {combination} appears {len(found)} times"
    return found[0]


def walk_keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, inner in value.items():
            yield str(key)
            yield from walk_keys(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from walk_keys(inner)


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """The feed reads the logs only; any connection attempt fails the test."""

    def refuse(*args, **kwargs):
        raise AssertionError("the board feed must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


# ---------------------------------------------------------------- venues and races


def test_board_lists_every_venue_and_race_of_the_day():
    # TOK races written out of order, across captures; 9 < 10 numerically, not as text
    tok_order = [12, 2, 10, 1, 11, 9]
    tok = []
    for i, no in enumerate(tok_order):
        tok.append(price(rid("TOK", no), "WIN", "1", 3.0 + no, jst(9, 0)))
        tok.append(price(rid("TOK", no), "WIN", "2", 5.0 + i, jst(10, 0)))
    write_log("TOK", tok)
    write_sidecar("TOK", {1: jst(10, 5).astimezone(UTC)})
    write_log(
        "KYO",
        [
            price(rid("KYO", 3), "WIN", "4", 6.2, jst(9, 5)),
            price(rid("KYO", 1), "WIN", "4", 2.2, jst(9, 5)),
        ],
    )
    write_log("CHU", [price(rid("CHU", 5), "PLACE", "8", 1.4, jst(9, 10), odds_hi=1.9)])
    # another day's log is not this day's
    other = date(2026, 11, 29)
    write_log("NAK", [price(rid("NAK", 1, other), "WIN", "1", 4.0, jst(9, 0, day=other))], other)

    now = jst(10, 30)
    board = build(now)

    assert board["race_date"] == "2026-11-28"
    assert instant(board["generated_utc"]) == now.astimezone(UTC)
    assert sorted(v["venue"] for v in board["venues"]) == ["CHU", "KYO", "TOK"]
    expected = {"TOK": [1, 2, 9, 10, 11, 12], "KYO": [1, 3], "CHU": [5]}
    for venue, nos in expected.items():
        races = venue_of(board, venue)["races"]
        assert [r["race_no"] for r in races] == nos, venue
        assert [r["race_id"] for r in races] == [rid(venue, no) for no in nos], venue
        for race in races:
            assert RACE_KEYS <= set(race), (venue, race["race_no"])
            for pool in race["pools"].values():
                assert "investment" in pool and "rows" in pool
                for row in pool["rows"]:
                    assert ROW_KEYS <= set(row), (venue, race["race_no"], row)
    assert set(race_of(board, "CHU", 5)["pools"]) == {"PLACE"}
    assert set(race_of(board, "TOK", 9)["pools"]) == {"WIN"}
    json.dumps(board)  # the feed is JSON-ready

    only = build(now, venues=["KYO"])
    assert [v["venue"] for v in only["venues"]] == ["KYO"]
    assert [r["race_no"] for r in only["venues"][0]["races"]] == [1, 3]

    assert build(now, day=date(2026, 11, 27))["venues"] == []


def test_current_odds_are_the_latest_capture():
    r11, r12 = rid("TOK", 11), rid("TOK", 12)
    rows = [
        price(r11, "WIN", "7", 4.1, jst(9, 0)),
        price(r11, "WIN", "3", 9.5, jst(9, 0)),
        price(r12, "WIN", "7", 2.0, jst(9, 0)),
        # the newest capture is written before an older one
        price(r11, "WIN", "7", 3.4, jst(14, 55)),
        price(r12, "WIN", "7", 2.6, jst(14, 50)),
        price(r11, "WIN", "7", 3.6, jst(14, 40)),
        price(r11, "WIN", "3", 8.8, jst(14, 40)),
        price(r11, "PLACE", "3", 2.1, jst(13, 0), odds_hi=3.3),
        price(r11, "PLACE", "3", 1.9, jst(14, 10), odds_hi=2.7),
    ]
    write_log("TOK", rows)
    board = build(jst(14, 58))

    race11, race12 = race_of(board, "TOK", 11), race_of(board, "TOK", 12)
    assert line(race11, "WIN", "7")["odds"] == 3.4
    assert line(race11, "WIN", "3")["odds"] == 8.8
    place = line(race11, "PLACE", "3")
    assert (place["odds"], place["odds_hi"]) == (1.9, 2.7)
    assert line(race12, "WIN", "7")["odds"] == 2.6
    assert [r["combination"] for r in race12["pools"]["WIN"]["rows"]] == ["7"]
    assert instant(race11["last_capture_utc"]) == jst(14, 55).astimezone(UTC)
    assert instant(race12["last_capture_utc"]) == jst(14, 50).astimezone(UTC)


# ---------------------------------------------------------------- morning reference


def test_morning_move_uses_the_first_capture_at_or_after_07_00_local():
    race = rid("TOK", 11)
    eve = DAY - timedelta(days=1)
    rows = [
        price(race, "WIN", "5", 30.0, jst(8, 0, day=eve)),  # the day before: not race day
        price(race, "WIN", "11", 15.0, jst(6, 40)),  # before 07:00 JST only
        price(race, "WIN", "9", 20.0, jst(6, 30)),
        price(race, "WIN", "7", 5.0, jst(6, 59, 59)),
        price(race, "WIN", "7", 4.0, jst(7, 0)),  # 07:00 JST is 22:00 UTC the day before
        price(race, "WIN", "5", 25.0, jst(7, 10)),
        price(race, "WIN", "3", 8.0, jst(7, 20)),
        price(race, "WIN", "2", 2.2, jst(7, 25)),
        price(race, "WIN", "4", 55.3, jst(7, 25)),
        price(race, "WIN", "6", 1.4, jst(7, 25)),
        price(race, "WIN", "7", 4.5, jst(7, 30)),
        price(race, "WIN", "7", 4.4, jst(8, 59)),
        price(race, "WIN", "7", 4.8, jst(9, 0)),
        price(race, "WIN", "3", 9.0, jst(9, 30)),
        price(race, "WIN", "9", 18.0, jst(14, 0)),
        price(race, "WIN", "7", 3.0, jst(14, 55)),
        price(race, "WIN", "5", 20.0, jst(14, 55)),
        price(race, "WIN", "3", 10.0, jst(14, 55)),
        price(race, "WIN", "2", 1.9, jst(14, 55)),
        price(race, "WIN", "4", 61.0, jst(14, 55)),
        price(race, "WIN", "6", 1.4, jst(14, 55)),
    ]
    write_log("TOK", rows)
    board = race_of(build(jst(14, 58)), "TOK", 11)

    # combination: (morning_odds, move_morning_pct)
    expected = {
        "7": (4.0, -25.0),
        "3": (8.0, 25.0),
        "5": (25.0, -20.0),
        "2": (2.2, -13.6),
        "4": (55.3, 10.3),
        "6": (1.4, 0.0),
        "9": (18.0, 0.0),  # its only row from 07:00 on is the current one
        "11": (None, None),  # no row at or after 07:00
    }
    for combination, (ref, move) in expected.items():
        got = line(board, "WIN", combination)
        assert (got["morning_odds"], got["move_morning_pct"]) == (ref, move), combination

    later = race_of(build(jst(14, 58), morning_from="09:00"), "TOK", 11)
    expected_9 = {"7": (4.8, -37.5), "3": (9.0, 11.1), "9": (18.0, 0.0), "2": (1.9, 0.0)}
    for combination, (ref, move) in expected_9.items():
        got = line(later, "WIN", combination)
        assert (got["morning_odds"], got["move_morning_pct"]) == (ref, move), combination


# ---------------------------------------------------------------- T-20 reference


def test_t20_move_uses_the_last_capture_at_or_before_post_minus_20():
    r11, r12 = rid("TOK", 11), rid("TOK", 12)
    write_sidecar("TOK", {11: jst(15, 1).astimezone(UTC), 12: jst(14, 31).astimezone(UTC)})
    rows = []
    for race in (r11, r12):  # the same prices, two post times
        rows += [
            price(race, "WIN", "7", 4.1, jst(7, 5)),
            price(race, "WIN", "7", 3.9, jst(14, 10)),
            price(race, "WIN", "7", 3.6, jst(14, 40)),
            price(race, "WIN", "7", 3.4, jst(14, 55)),
        ]
    rows += [
        price(r11, "WIN", "3", 7.5, jst(7, 5)),
        price(r11, "WIN", "3", 6.0, jst(14, 41)),  # exactly post - 20: used
        price(r11, "WIN", "3", 7.0, jst(14, 41, 30)),  # after post - 20: not used
        price(r11, "WIN", "3", 6.6, jst(14, 55)),
        price(r11, "WIN", "5", 10.0, jst(13, 0)),
        price(r11, "WIN", "5", 12.0, jst(14, 50)),
        price(r11, "WIN", "9", 11.0, jst(14, 45)),  # first seen after post - 20
        price(r11, "WIN", "9", 9.9, jst(14, 55)),
    ]
    write_log("TOK", rows)
    board = build(jst(14, 58))
    race11, race12 = race_of(board, "TOK", 11), race_of(board, "TOK", 12)

    assert instant(race11["post_utc"]) == jst(15, 1).astimezone(UTC)
    seven = line(race11, "WIN", "7")
    assert seven["odds"] == 3.4
    assert (seven["morning_odds"], seven["move_morning_pct"]) == (4.1, -17.1)
    assert (seven["t20_odds"], seven["move_t20_pct"]) == (3.6, -5.6)

    # combination: (t20_odds, move_t20_pct)
    expected = {"3": (6.0, 10.0), "5": (10.0, 20.0), "9": (None, None)}
    for combination, (ref, move) in expected.items():
        got = line(race11, "WIN", combination)
        assert (got["t20_odds"], got["move_t20_pct"]) == (ref, move), combination

    # post 14:31: post - 20 is 14:11, so the 14:10 capture is the reference
    assert instant(race12["post_utc"]) == jst(14, 31).astimezone(UTC)
    seven12 = line(race12, "WIN", "7")
    assert (seven12["t20_odds"], seven12["move_t20_pct"]) == (3.9, -12.8)
    assert seven12["move_morning_pct"] == -17.1


def test_t20_move_is_null_before_t20_and_without_a_post_time():
    r11, r12 = rid("TOK", 11), rid("TOK", 12)
    write_sidecar("TOK", {11: jst(15, 1).astimezone(UTC)})  # no post time for race 12
    rows = []
    for race in (r11, r12):
        rows += [
            price(race, "WIN", "7", 4.1, jst(7, 5)),
            price(race, "WIN", "7", 3.6, jst(14, 0)),
            price(race, "WIN", "7", 3.4, jst(14, 25)),
        ]
    write_log("TOK", rows)

    early = build(jst(14, 30))  # before post - 20 (14:41)
    race11 = race_of(early, "TOK", 11)
    seven = line(race11, "WIN", "7")
    assert seven["odds"] == 3.4
    assert (seven["t20_odds"], seven["move_t20_pct"]) == (None, None)
    assert seven["move_morning_pct"] == -17.1

    late = build(jst(14, 45))  # after post - 20: the 14:25 capture is the latest by then
    seven = line(race_of(late, "TOK", 11), "WIN", "7")
    assert (seven["t20_odds"], seven["move_t20_pct"]) == (3.4, 0.0)

    for board in (early, late, build(jst(16, 0))):
        race12 = race_of(board, "TOK", 12)
        assert race12["post_utc"] is None
        no_post = line(race12, "WIN", "7")
        assert (no_post["t20_odds"], no_post["move_t20_pct"]) == (None, None)
        assert no_post["move_morning_pct"] == -17.1


# ---------------------------------------------------------------- pools


def test_ranged_pools_move_on_the_low_figure_and_carry_both():
    race = rid("TOK", 11)
    write_sidecar("TOK", {11: jst(15, 1).astimezone(UTC)})
    rows = [
        price(race, "PLACE", "7", 1.5, jst(7, 5), odds_hi=2.0),
        price(race, "PLACE", "7", 1.3, jst(14, 40), odds_hi=1.8),
        price(race, "PLACE", "7", 1.2, jst(14, 55), odds_hi=1.9),
        price(race, "QUINELLA_PLACE", "3,7", 4.0, jst(7, 5), odds_hi=5.5),
        price(race, "QUINELLA_PLACE", "3,7", 3.0, jst(14, 40), odds_hi=4.2),
        price(race, "QUINELLA_PLACE", "3,7", 2.7, jst(14, 55), odds_hi=4.6),
        price(race, "WIN", "7", 3.4, jst(14, 55)),
    ]
    write_log("TOK", rows)
    board = race_of(build(jst(14, 58)), "TOK", 11)

    # pool, combination: (odds, odds_hi, morning_odds, move_morning, t20_odds, move_t20)
    expected = {
        ("PLACE", "7"): (1.2, 1.9, 1.5, -20.0, 1.3, -7.7),
        ("QUINELLA_PLACE", "3,7"): (2.7, 4.6, 4.0, -32.5, 3.0, -10.0),
    }
    for (pool, combination), want in expected.items():
        got = line(board, pool, combination)
        assert (
            got["odds"],
            got["odds_hi"],
            got["morning_odds"],
            got["move_morning_pct"],
            got["t20_odds"],
            got["move_t20_pct"],
        ) == want, (pool, combination)
    assert line(board, "WIN", "7")["odds_hi"] is None


def test_a_scratched_runner_has_status_and_no_move():
    race = rid("TOK", 11)
    write_sidecar("TOK", {11: jst(15, 1).astimezone(UTC)})
    rows = [
        price(race, "WIN", "5", 12.0, jst(7, 5)),
        price(race, "WIN", "7", 4.1, jst(7, 5)),
        price(race, "PLACE", "5", 3.0, jst(7, 5), odds_hi=4.5),
        price(race, "WIN", "5", 10.0, jst(14, 40)),
        price(race, "WIN", "7", 3.6, jst(14, 40)),
        price(race, "PLACE", "5", 2.8, jst(14, 40), odds_hi=4.0),
        price(race, "WIN", "5", None, jst(14, 55)),
        price(race, "PLACE", "5", None, jst(14, 55)),
        price(race, "WIN", "7", 3.4, jst(14, 55)),
    ]
    write_log("TOK", rows)
    board = race_of(build(jst(14, 58)), "TOK", 11)

    for pool in ("WIN", "PLACE"):
        scratched = line(board, pool, "5")
        assert scratched["status"] == "SCR", pool
        assert scratched["odds"] is None, pool
        assert scratched["move_morning_pct"] is None, pool
        assert scratched["move_t20_pct"] is None, pool
    runner = line(board, "WIN", "7")
    assert runner["status"] != "SCR"
    assert (runner["move_morning_pct"], runner["move_t20_pct"]) == (-17.1, -5.6)


def test_pool_investment_is_the_latest_total():
    r11, r12 = rid("TOK", 11), rid("TOK", 12)
    rows = [
        total(r11, "WIN", 1_000_000, jst(7, 5)),
        price(r11, "WIN", "7", 4.1, jst(7, 5)),
        total(r11, "PLACE", 3_210_000, jst(14, 55)),  # newest PLACE total, written first
        total(r11, "WIN", 5_000_000, jst(14, 40)),
        total(r11, "PLACE", 900_000, jst(14, 40)),
        price(r11, "PLACE", "7", 1.3, jst(14, 40), odds_hi=1.8),
        price(r11, "QUINELLA_PLACE", "3,7", 3.0, jst(14, 40), odds_hi=4.2),
        total(r11, "WIN", 8_123_400, jst(14, 50)),
        total(r11, "WIN", None, jst(14, 56)),  # a total row without a figure
        price(r11, "WIN", "7", 3.4, jst(14, 56)),
        total(r12, "WIN", 2_500_000, jst(14, 50)),
        price(r12, "WIN", "1", 2.5, jst(14, 50)),
    ]
    write_log("TOK", rows)
    board = build(jst(14, 58))
    race11, race12 = race_of(board, "TOK", 11), race_of(board, "TOK", 12)

    assert race11["pools"]["WIN"]["investment"] == 8_123_400
    assert race11["pools"]["PLACE"]["investment"] == 3_210_000
    assert race11["pools"]["QUINELLA_PLACE"]["investment"] is None
    assert race12["pools"]["WIN"]["investment"] == 2_500_000
    for race in (race11, race12):
        for pool in race["pools"].values():
            assert POOL not in [r["combination"] for r in pool["rows"]]


# ---------------------------------------------------------------- post times


def test_post_times_fall_back_to_the_hktrack_card():
    day = date(2026, 11, 22)  # a copied HKTrack day: no sidecar for KYO
    write_card("KYO", {1: "2026-11-22T10:05:00+09:00", 11: "2026-11-22T15:25:00+09:00"}, day)
    # TOK has both: the sidecar wins over the card
    write_card("TOK", {11: "2026-11-22T15:45:00+09:00"}, day)
    write_sidecar("TOK", {11: jst(15, 1, day=day).astimezone(UTC)}, day)
    rows = {"KYO": [], "TOK": []}
    for venue in rows:
        for no in (1, 11, 12):
            race = rid(venue, no, day)
            rows[venue] += [
                price(race, "WIN", "7", 4.1, jst(7, 5, day=day)),
                price(race, "WIN", "7", 3.9, jst(14, 40, day=day)),
                price(race, "WIN", "7", 3.6, jst(15, 0, day=day)),
                price(race, "WIN", "7", 3.4, jst(15, 20, day=day)),
            ]
        write_log(venue, rows[venue], day)

    board = build(jst(15, 22, day=day), day=day)

    kyo11 = race_of(board, "KYO", 11)
    assert instant(kyo11["post_utc"]) == datetime(2026, 11, 22, 6, 25, tzinfo=UTC)
    assert instant(race_of(board, "KYO", 1)["post_utc"]) == datetime(2026, 11, 22, 1, 5, tzinfo=UTC)
    assert race_of(board, "KYO", 12)["post_utc"] is None
    # card post 15:25 JST: post - 20 is 15:05, so the 15:00 capture is the reference
    seven = line(kyo11, "WIN", "7")
    assert (seven["t20_odds"], seven["move_t20_pct"]) == (3.6, -5.6)
    assert line(race_of(board, "KYO", 12), "WIN", "7")["t20_odds"] is None

    tok11 = race_of(board, "TOK", 11)
    assert instant(tok11["post_utc"]) == datetime(2026, 11, 22, 6, 1, tzinfo=UTC)
    seven = line(tok11, "WIN", "7")  # sidecar post 15:01: post - 20 is 14:41
    assert (seven["t20_odds"], seven["move_t20_pct"]) == (3.9, -12.8)
    assert race_of(board, "TOK", 1)["post_utc"] is None  # the sidecar has no race 1


# ---------------------------------------------------------------- no probability or bet


_FORBIDDEN_TOKENS = {"p", "ev", "fair", "bet", "bets", "stake", "stakes", "prob", "kelly"}
_FORBIDDEN_PARTS = ("prob", "fair", "stake", "bet", "kelly", "expected_value")


def _forbidden(key: str) -> bool:
    lowered = key.lower()
    tokens = set(lowered.replace("-", "_").split("_"))
    return (
        lowered.startswith("p_")
        or bool(tokens & _FORBIDDEN_TOKENS)
        or any(part in lowered for part in _FORBIDDEN_PARTS)
    )


def test_the_feed_has_no_probability_or_bet_fields():
    assert _forbidden("p_win") and _forbidden("ev") and _forbidden("fair_odds")
    assert _forbidden("bet_flag") and _forbidden("stake") and _forbidden("win_prob")
    assert not any(_forbidden(k) for k in ROW_KEYS | RACE_KEYS | {"venue", "investment"})

    race = rid("TOK", 11)
    write_sidecar("TOK", {11: jst(15, 1).astimezone(UTC)})
    rows = [
        total(race, "WIN", 1_000_000, jst(7, 5)),
        price(race, "WIN", "7", 4.1, jst(7, 5)),
        price(race, "WIN", "5", 12.0, jst(7, 5)),
        price(race, "PLACE", "7", 1.5, jst(7, 5), odds_hi=2.0),
        price(race, "QUINELLA_PLACE", "5,7", 9.0, jst(7, 5), odds_hi=12.0),
        price(race, "WIN", "7", 3.6, jst(14, 40)),
        price(race, "WIN", "5", None, jst(14, 55)),
        price(race, "WIN", "7", 3.4, jst(14, 55)),
        total(race, "WIN", 8_000_000, jst(14, 55)),
    ]
    write_log("TOK", rows)
    write_log("KYO", [price(rid("KYO", 1), "WIN", "1", 2.0, jst(9, 0))])
    board = build(jst(14, 58))

    keys = set(walk_keys(board))
    assert ROW_KEYS <= keys  # the walk reaches the rows
    assert sorted(k for k in keys if _forbidden(k)) == []
