"""Readers of the per-venue odds log once tier-2 captures are in it (ticket #130, review fix).

Tier 2 (BRACKET_QUINELLA 枠連, FORECAST 馬単, TRIO 3連複, TIERCE 3連単) and the WIN5 `__pool__` row
go into the same `<VENUE>.jsonl.gz` as tier 1 (WIN, PLACE, QUINELLA, QUINELLA_PLACE). The ticket's
rule: "Stream large pages; never hold a day's trifecta boards in memory" (the owner's 16 GB Mac).

Contract pinned here (the coder follows it):
- The live board (`board.feed.build_board`, and the server's `/feed` built from it) shows tier 1
  only: over a log holding tier-2 and WIN5 rows besides tier 1, its output is exactly its output
  over the tier-1 rows alone. No tier-2 pool and no WIN5 row reaches the feed; a race with only
  tier-2 rows is not on the board; a tier-2 capture does not move `last_capture_utc`.
- Bounded memory: `build_board` over a venue-day holding 24 tier-2 captures of 18-runner races
  (4,896 3連単 prices each, ~145k tier-2 rows) does not materialise the tier-2 rows: its peak RSS
  rises by less than FEED_LIMIT_BYTES (a child process measures its own peak: child_memory.py).
- `logger.compact.compact_day` keeps tier-2 and WIN5 rows in the day's Parquet, one row per logged
  row as for tier 1 (Ruling: tier-2 studies need decision-time prices, so the Parquet carries them;
  the live board is the reader that drops them). Over the same venue-day its peak RSS rises by less
  than COMPACT_LIMIT_BYTES, so the day's rows are not held as one Python dict each (column-wise,
  e.g. one Arrow table per member, fits).

Each limit sits between streaming and materialising, from measurements on this day (see the
constants). A real day is ~6x this per venue, two venues: what materialises here at ~0.3-0.5 GB
would be several GB there. All logs are synthetic, written under tmp_path.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from itertools import combinations, combinations_with_replacement, permutations
from pathlib import Path
from typing import Any

import child_memory
import pyarrow.parquet as pq
import pytest

from horsetrack.core import paths
from horsetrack.countries.japan import odds_log

UTC = timezone.utc
JST = timezone(timedelta(hours=9))
DAY = date(2025, 11, 29)  # in the past: the server's real clock sees every capture
POOL = odds_log.POOL_TOTAL
TIER1 = ("WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE")
TIER2 = ("BRACKET_QUINELLA", "FORECAST", "TRIO", "TIERCE")
NOT_ON_BOARD = {*TIER2, "WIN5"}

# 18 runners in 8 brackets, as JRA draws them (the last brackets take the extra runners)
RUNNERS = 18
BRACKETS = {
    n: b
    for b, members in enumerate(
        [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12, 13), (14, 15, 16), (17, 18)], start=1
    )
    for n in members
}
PER_POOL = {
    "BRACKET_QUINELLA": 36,  # 8 brackets, with replacement
    "FORECAST": 18 * 17,
    "TRIO": 816,  # C(18, 3)
    "TIERCE": 4_896,  # 18 x 17 x 16
}
TIER2_ROWS_PER_CAPTURE = sum(PER_POOL.values()) + len(TIER2)  # + one __pool__ row per pool


# ---------------------------------------------------------------- rows


def jst(hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime(DAY.year, DAY.month, DAY.day, hh, mm, ss, tzinfo=JST)


def rid(venue: str, no: int) -> str:
    return f"jra:{DAY.isoformat()}:{venue}:{no:02d}"


def iso(at: datetime) -> str:
    return at.astimezone(UTC).isoformat()


def base(race_id: str, at: datetime) -> dict[str, Any]:
    """The keys every live odds row carries (official_pages / exotic_pages shape)."""
    local = at.astimezone(JST)
    return {
        "snapshot_utc": iso(at),
        "race_id": race_id,
        "source_update_utc": iso(at - timedelta(seconds=20)),
        "fetched_utc": iso(at + timedelta(seconds=2)),
        "page_state": f"{local.hour}時{local.minute:02d}分現在のオッズ",
    }


def priced(race_id, pool, combination, odds, at, **extra) -> dict[str, Any]:
    local = at.astimezone(JST)
    row = base(race_id, at) | {
        "pool": pool,
        "combination": combination,
        "odds": odds,
        "status_raw": "SCR" if odds is None else f"{local.hour}時{local.minute:02d}分現在",
    }
    return row | extra


def total(race_id, pool, investment, at) -> dict[str, Any]:
    return base(race_id, at) | {
        "pool": pool,
        "combination": POOL,
        "odds": None,
        "status_raw": None,
        "investment": investment,
    }


def tier1_capture(venue: str, no: int, at: datetime, step: int, runners: int) -> list[dict]:
    """WIN, PLACE (ranged), QUINELLA and Wide (ranged) of one race, with pool totals."""
    r = rid(venue, no)
    rows = []
    for n in range(1, runners + 1):
        rows.append(priced(r, "WIN", str(n), round(2.0 + n * 1.5 + step * 0.3, 1), at))
        lo = round(1.1 + n * 0.2 + step * 0.1, 1)
        rows.append(priced(r, "PLACE", str(n), lo, at, odds_hi=round(lo + 0.6, 1)))
    for a, b in combinations(range(1, runners + 1), 2):
        q = round(5.0 + a * 2 + b + step, 1)
        rows.append(priced(r, "QUINELLA", f"{a},{b}", q, at))
        rows.append(priced(r, "QUINELLA_PLACE", f"{a},{b}", round(q / 3, 1), at, odds_hi=q))
    for i, pool in enumerate(TIER1):
        rows.append(total(r, pool, 1_000_000.0 * (i + 1) * (step + 1) + no, at))
    return rows


def tier2_capture(venue: str, no: int, at: datetime, step: int) -> list[dict]:
    """One grid-point capture of an 18-runner race: 枠連, 馬単, 3連複, 3連単 and their totals."""
    r = rid(venue, no)
    field = range(1, RUNNERS + 1)
    rows = [
        priced(r, "BRACKET_QUINELLA", f"{a},{b}", round(4.0 + a * 3 + b + step, 1), at)
        for a, b in combinations_with_replacement(range(1, 9), 2)
    ]
    rows += [
        priced(r, "FORECAST", f"{a},{b}", round(8.0 + a * 4 + b * 2 + step, 1), at)
        for a, b in permutations(field, 2)
    ]
    rows += [
        priced(r, "TRIO", f"{a},{b},{c}", round(20.0 + a * 5 + b * 3 + c + step, 1), at)
        for a, b, c in combinations(field, 3)
    ]
    rows += [
        priced(r, "TIERCE", f"{a},{b},{c}", round(60.0 + a * 11 + b * 7 + c * 3 + step, 1), at)
        for a, b, c in permutations(field, 3)
    ]
    for i, pool in enumerate(TIER2):
        rows.append(total(r, pool, 5_000_000.0 * (i + 2) * (step + 1) + no, at))
    return rows


def win5_row(at: datetime, legs: list[str]) -> dict[str, Any]:
    return {
        "snapshot_utc": iso(at),
        "fetched_utc": iso(at + timedelta(seconds=2)),
        "source_update_utc": iso(at - timedelta(seconds=20)),
        "race_id": f"jra:{DAY.isoformat()}:WIN5",
        "pool": "WIN5",
        "combination": POOL,
        "odds": None,
        "status_raw": None,
        "investment": 812_345_600.0,
        "carryover": 123_456_700.0,
        "legs": legs,
    }


def write_members(data: Path, venue: str, members: list[list[dict]], post: dict[int, datetime]):
    log = data / "odds_snapshots" / "jra" / DAY.isoformat() / f"{venue}.jsonl.gz"
    for rows in members:
        odds_log.append_capture(log, rows)  # an empty member writes nothing
    odds_log.write_post_times(log.with_name(f"{venue}.races.json"), post, jst(7, 0))
    return log


def tier1_only(members: list[list[dict]]) -> list[list[dict]]:
    return [[row for row in rows if row["pool"] in TIER1] for rows in members]


# ---------------------------------------------------------------- the board shows tier 1 only

POSTS = {1: jst(10, 5), 2: jst(10, 35), 7: jst(13, 5)}  # race 7: tier-2 rows only


def mixed_day() -> list[list[dict]]:
    """Two tier-1 races (6 runners) with tier-2 captures around them, before and after T-20, in
    the morning window and after the last tier-1 capture; a member mixing both tiers; a race with
    tier-2 rows only; and the WIN5 row. Tier-2 prices sit far from tier-1 ones on purpose."""
    members: list[list[dict]] = []
    for no in (1, 2):
        post = POSTS[no]
        t1 = [post - timedelta(minutes=m) for m in (150, 45, 25, 12, 3)]
        t2 = [post - timedelta(minutes=m) for m in (60, 30, 16, 5, 1)]  # the D2 grid
        for step, (a, b) in enumerate(zip(t1, t2, strict=True)):
            members.append(tier1_capture("TOK", no, a, step, runners=6))
            members.append(tier2_capture("TOK", no, b, step))
        # the logger never mixes tiers in one member; the log format does not forbid it
        shared = post - timedelta(minutes=2)
        members.append(
            tier1_capture("TOK", no, shared, 9, runners=6)[:8]
            + tier2_capture("TOK", no, shared, 9)[:50]
        )
        members.append(tier2_capture("TOK", no, post + timedelta(minutes=1), 7))  # last capture
    members.append(tier2_capture("TOK", 7, POSTS[7] - timedelta(minutes=30), 0))
    members.append([win5_row(jst(9, 50), [rid("TOK", n) for n in (1, 2, 7, 8, 9)])])
    return members


def set_data(monkeypatch, data: Path) -> None:
    data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))


@pytest.mark.parametrize(
    "now",
    [jst(9, 10), jst(9, 50), jst(10, 4), jst(10, 30), jst(23, 0)],
    ids=["morning", "race1-before-t20", "race1-after-t20", "race2-after-t20", "day-end"],
)
def test_the_board_over_a_mixed_log_is_the_board_over_its_tier1_rows(tmp_path, monkeypatch, now):
    from horsetrack.countries.japan.board import feed

    members = mixed_day()
    write_members(tmp_path / "mixed", "TOK", members, POSTS)
    write_members(tmp_path / "tier1", "TOK", tier1_only(members), POSTS)

    set_data(monkeypatch, tmp_path / "mixed")
    mixed = feed.build_board(DAY, now=lambda: now)
    set_data(monkeypatch, tmp_path / "tier1")
    alone = feed.build_board(DAY, now=lambda: now)

    assert mixed == alone
    races = mixed["venues"][0]["races"]
    for race in races:
        assert not set(race["pools"]) & NOT_ON_BOARD, race["race_id"]


def test_the_board_names_tier1_pools_races_and_captures_only(tmp_path, monkeypatch):
    """Absolute checks, so an empty board cannot pass the comparison above."""
    from horsetrack.countries.japan.board import feed

    set_data(monkeypatch, tmp_path / "mixed")
    write_members(tmp_path / "mixed", "TOK", mixed_day(), POSTS)

    board = feed.build_board(DAY, now=lambda: jst(23, 0))

    (venue,) = board["venues"]
    races = venue["races"]
    assert [race["race_no"] for race in races] == [1, 2]  # race 7 has tier-2 rows only
    for race in races:
        no, post = race["race_no"], POSTS[race["race_no"]]
        assert set(race["pools"]) == set(TIER1)
        # the last tier-1 capture is the shared member at post - 2; tier 2 ran on to post + 1
        last = datetime.fromisoformat(race["last_capture_utc"])
        assert last == post - timedelta(minutes=2)
        win = {row["combination"]: row["odds"] for row in race["pools"]["WIN"]["rows"]}
        # step 9 (the shared member) holds WIN 1-4 (its first 8 rows: WIN and PLACE of 1-4)
        assert win == {
            **{str(n): round(2.0 + n * 1.5 + 9 * 0.3, 1) for n in range(1, 5)},
            **{str(n): round(2.0 + n * 1.5 + 4 * 0.3, 1) for n in range(5, 7)},
        }
        assert len(race["pools"]["QUINELLA"]["rows"]) == 15  # C(6, 2)
        assert race["pools"]["WIN"]["investment"] == 1_000_000.0 * 5 + no  # step 4 totals
    text = json.dumps(board, ensure_ascii=False)
    for pool in NOT_ON_BOARD:
        assert f'"{pool}"' not in text


def test_the_served_feed_over_a_mixed_log_is_the_tier1_feed(tmp_path, monkeypatch):
    from horsetrack.countries.japan.board import feed, server

    members = mixed_day()
    write_members(tmp_path / "tier1", "TOK", tier1_only(members), POSTS)
    set_data(monkeypatch, tmp_path / "tier1")
    want = feed.build_board(
        DAY, now=lambda: jst(23, 59)
    )  # the day is past: the real clock is later

    set_data(monkeypatch, tmp_path / "mixed")
    write_members(tmp_path / "mixed", "TOK", members, POSTS)
    body, etag = server.Feed(DAY, min_rebuild_s=0).current()

    assert body is not None and etag is not None
    got = json.loads(body)
    got.pop("generated_utc"), want.pop("generated_utc")
    assert got == json.loads(json.dumps(want, ensure_ascii=False))


# ---------------------------------------------------------------- compact keeps tier 2


def test_compact_keeps_tier2_and_win5_rows_one_per_logged_row(tmp_path, monkeypatch):
    """Ruling: tier-2 and WIN5 rows stay in the day's Parquet, with the tier-1 columns."""
    from horsetrack.countries.japan.logger.compact import compact_day

    members = mixed_day()
    set_data(monkeypatch, tmp_path / "mixed")
    write_members(tmp_path / "mixed", "TOK", members, POSTS)

    table = pq.read_table(compact_day(DAY.isoformat()))
    rows = table.to_pylist()

    logged = [row for member in members for row in member]
    assert len(rows) == len(logged)
    count = {}
    for row in rows:
        count[row["pool"]] = count.get(row["pool"], 0) + 1
    want = {}
    for row in logged:
        want[row["pool"]] = want.get(row["pool"], 0) + 1
    assert count == want and set(TIER2) <= set(count) and count["WIN5"] == 1

    # a tier-2 price at its grid point: values, the minutes to post and the final flag
    race2 = rid("TOK", 2)
    at16 = POSTS[2] - timedelta(minutes=16)
    (tierce,) = [
        r
        for r in rows
        if r["race_id"] == race2
        and r["pool"] == "TIERCE"
        and r["combination"] == "18,1,7"
        and r["snapshot_utc"] == at16
    ]
    assert tierce["odds"] == round(60.0 + 18 * 11 + 1 * 7 + 7 * 3 + 2, 1)
    fetched = at16 + timedelta(seconds=2)
    assert tierce["minutes_to_post"] == pytest.approx((POSTS[2] - fetched) / timedelta(minutes=1))
    assert tierce["is_final_capture"] is False  # captured again at post + 1
    finals = [
        r
        for r in rows
        if r["race_id"] == race2 and r["pool"] == "TIERCE" and r["combination"] == "18,1,7"
    ]
    assert [r["snapshot_utc"] for r in finals if r["is_final_capture"]] == [
        POSTS[2] + timedelta(minutes=1)
    ]
    (win5,) = [r for r in rows if r["pool"] == "WIN5"]
    assert win5["investment"] == 812_345_600.0 and win5["combination"] == POOL


# ---------------------------------------------------------------- bounded memory

MEM_RACES = 6
MEM_GRID = (60, 30, 16, 5)  # grid points captured per race
MEM_TIER2_ROWS = MEM_RACES * len(MEM_GRID) * TIER2_ROWS_PER_CAPTURE  # 145,392
MEM_TIER1_RUNNERS = 18
MEM_TIER1_STEPS = (45, 25, 12, 3)
MEM_TIER1_ROWS = MEM_RACES * len(MEM_TIER1_STEPS) * (2 * 18 + 2 * 153 + len(TIER1))  # 8,304
MEM_POSTS = {no: jst(10, 5) + timedelta(minutes=35 * (no - 1)) for no in range(1, MEM_RACES + 1)}

# Measured on this day (macOS, CPython 3): `read_rows` then build, as at 8ff0e73, rises 328 MB;
# streaming `iter_members` and keeping tier-1 rows only rises 77 MB (iter_members alone ~62 MB:
# a member's bytes, text, lines and dicts, two members overlapping at the generator's hand-over).
FEED_LIMIT_BYTES = 150 * 1000 * 1000
# Measured: 8ff0e73 (`read_rows`, then one record dict per row) rises 553 MB; a prototype that
# turns each member into an Arrow table, concatenates, sorts and writes rises 159 MB.
COMPACT_LIMIT_BYTES = 250 * 1000 * 1000


@pytest.fixture(scope="module")
def heavy_day(tmp_path_factory) -> Path:
    """One venue-day: 6 18-runner races, tier 1 at 4 captures each, tier 2 at 4 grid points each,
    one member per capture, written as the logger appends them."""
    data = tmp_path_factory.mktemp("heavy") / "data"
    members: list[list[dict]] = []
    for no, post in MEM_POSTS.items():
        for step, (m1, m2) in enumerate(zip(MEM_TIER1_STEPS, MEM_GRID, strict=True)):
            members.append(
                tier1_capture("TOK", no, post - timedelta(minutes=m1), step, MEM_TIER1_RUNNERS)
            )
            members.append(tier2_capture("TOK", no, post - timedelta(minutes=m2), step))
    write_members(data, "TOK", members, MEM_POSTS)
    return data


# runs after child_memory.PROBE, which defines peak_bytes() (the child's own peak) and START_PEAK
CHILD = r"""
import json
from datetime import date, datetime

peak = peak_bytes

mode, day = sys.argv[1], sys.argv[2]
if mode == "feed":
    from horsetrack.countries.japan.board import feed
    now = datetime.fromisoformat(sys.argv[3])
    before = peak()
    board = feed.build_board(date.fromisoformat(day), now=lambda: now)
    after = peak()
    races = [r for v in board["venues"] for r in v["races"]]
    out = {
        "pools": sorted({p for r in races for p in r["pools"]}),
        "races": [r["race_no"] for r in races],
        "lines": sum(len(p["rows"]) for r in races for p in r["pools"].values()),
    }
else:
    import pyarrow.parquet as pq
    from horsetrack.countries.japan.logger.compact import compact_day
    before = peak()
    path = compact_day(day)
    after = peak()
    pools = pq.read_table(path, columns=["pool"]).column("pool").to_pylist()
    out = {"pools": {p: pools.count(p) for p in sorted(set(pools))}}
out["rise"] = after - before
out["start"] = START_PEAK
print(json.dumps(out))
"""


def run_child(data: Path, *args: str) -> dict[str, Any]:
    return child_memory.run_child(CHILD, *args, env={**os.environ, paths.DATA_DIR_VAR: str(data)})


def test_the_board_over_a_day_of_trifecta_captures_does_not_hold_them(heavy_day):
    out = run_child(heavy_day, "feed", DAY.isoformat(), iso(jst(23, 0)))

    assert out["pools"] == sorted(TIER1)
    assert out["races"] == list(range(1, MEM_RACES + 1))
    assert out["lines"] == MEM_RACES * (2 * 18 + 2 * 153)
    assert out["rise"] < FEED_LIMIT_BYTES, (
        f"build_board rose {out['rise'] / 1e6:.0f} MB over {MEM_TIER2_ROWS:,} tier-2 rows"
    )


def test_compacting_a_day_of_trifecta_captures_stays_under_its_memory_bound(heavy_day):
    out = run_child(heavy_day, "compact", DAY.isoformat())

    per_race_grid = MEM_RACES * len(MEM_GRID)
    want = {pool: per_race_grid * (n + 1) for pool, n in PER_POOL.items()}
    want |= {
        pool: MEM_RACES * len(MEM_TIER1_STEPS) * (n + 1)
        for pool, n in {"WIN": 18, "PLACE": 18, "QUINELLA": 153, "QUINELLA_PLACE": 153}.items()
    }
    assert out["pools"] == want  # tier 2 kept, none dropped to save memory
    assert sum(want.values()) == MEM_TIER1_ROWS + MEM_TIER2_ROWS
    assert out["rise"] < COMPACT_LIMIT_BYTES, (
        f"compact_day rose {out['rise'] / 1e6:.0f} MB over {MEM_TIER2_ROWS:,} tier-2 rows"
    )
