"""The board at a decision time and the final board (ticket #113).

Contract pinned here (`horsetrack.countries.japan.board.snapshot`):
- A row is available at its `fetched_utc`, else (key absent or null, as in HKTrack logs) at its
  `snapshot_utc`. `board_at(rows, race_id, t)` takes, per (pool, combination) of that race, the
  latest row available at or before t; nothing available after t is ever used, in any pool or in
  the totals. No row of the race by t gives None. Rows of other races are ignored.
- `capture_utc` is the oldest availability time among the price rows used.
- WIN -> `win[int]`; PLACE -> `place[int] = (odds, odds_hi)`; QUINELLA_PLACE 'a,b' ->
  `wide[(min, max)] = (odds, odds_hi)`; a runner whose latest row is status_raw 'SCR' (odds null)
  is in `scratched` and unpriced; '__pool__' rows give `totals[pool] = investment`.
- A row is final when `source_update_utc` is null and '最終オッズ' is in `page_state` (#21 rows) or
  `status_raw` (HKTrack rows). `final_board` uses the first capture (by snapshot_utc) whose WIN rows
  are final; a race never final gives None.
- Split captures (#234, owner sign-off 2026-11-25: "each pool's latest final capture"). A pool's
  final capture holds at least one priced row of that pool, every one final (SCR rows do not
  count). The board starts from the anchor capture's rows (WIN, other pools' totals); PLACE and
  Wide, each with its prices and pool total, come from that pool's first final capture at or after
  the anchor, else its latest final capture before it, else nothing. So HKTrack's final Wide
  capture ~2 s after WIN+PLACE prices the board's Wide, and a one-capture round whose pools are all
  final gives that capture's board as before. `at_utc` is the latest availability among the rows
  taken, `capture_utc` the oldest price row's.
- `post_times_for(date, venue)`: the `<VENUE>.races.json` sidecar beside the odds log, else
  HKTrack's stored card `processed/jra/upcoming/jra_<date>_<VENUE>/races.parquet`, else {}; times
  are aware UTC.
- `bucket_for(minutes, grid)`: the smallest grid horizon at or above the minutes to post, the
  largest horizon for anything earlier, 0 after the post. The shipped grid is D2's.

All rows are synthetic and built here; HORSETRACK_DATA_DIR points at tmp_path.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from horsetrack.core import paths
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board import snapshot

UTC = timezone.utc
RACE = "jra:2026-11-28:TOK:11"
OTHER = "jra:2026-11-28:TOK:12"
POST = datetime(2026, 11, 28, 6, 1, tzinfo=UTC)
FINAL = "最終オッズ"


def at(hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime(2026, 11, 28, hh, mm, ss, tzinfo=UTC)


def iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def row(
    pool: str,
    combination: str,
    odds,
    *,
    fetched: datetime | None = None,
    snap: datetime | None = None,
    race_id: str = RACE,
    odds_hi=None,
    status_raw: str | None = "14時55分現在",
    page_state: str | None = None,
    source_update: datetime | None = None,
    investment=None,
    hktrack: bool = False,
) -> dict:
    """One logged row. #21 shape: fetched_utc present, snapshot_utc 30 s before it unless given.
    HKTrack shape (`hktrack=True`): no fetched_utc and no page_state keys."""
    if snap is None:
        snap = fetched - timedelta(seconds=30)
    out = {
        "snapshot_utc": iso(snap),
        "race_id": race_id,
        "pool": pool,
        "combination": combination,
        "odds": odds,
        "status_raw": "SCR" if odds is None and combination != "__pool__" else status_raw,
        "source_update_utc": iso(source_update if source_update is not None else snap),
    }
    if not hktrack:
        out["fetched_utc"] = iso(fetched)
        out["page_state"] = page_state
    if odds_hi is not None:
        out["odds_hi"] = odds_hi
    if investment is not None:
        out["investment"] = investment
    return out


def final_row(pool, combination, odds, *, snap, hktrack, race_id=RACE, odds_hi=None):
    """A row of a final capture: source_update_utc null, '最終オッズ' in status_raw (HKTrack) or
    page_state (#21, status_raw null)."""
    r = row(
        pool,
        combination,
        odds,
        fetched=None if hktrack else snap + timedelta(seconds=5),
        snap=snap,
        race_id=race_id,
        odds_hi=odds_hi,
        status_raw=FINAL if hktrack else None,
        page_state=None if hktrack else FINAL,
        hktrack=hktrack,
    )
    r["source_update_utc"] = None
    if odds is None:
        r["status_raw"] = "SCR"
    return r


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


# ---------------------------------------------------------------- board_at


def test_board_at_takes_the_latest_row_available_by_t():
    rows = [
        row("WIN", "7", 4.1, fetched=at(5, 40)),
        row("WIN", "7", 3.6, fetched=at(5, 57, 50)),
        row("WIN", "7", 3.4, fetched=at(5, 58, 40)),
        row("WIN", "3", 9.0, fetched=at(5, 40)),
        row("WIN", "3", 8.0, fetched=at(5, 58, 40)),
    ]
    later = snapshot.board_at(rows, RACE, at(5, 59), post_utc=POST)
    assert later.win[7] == 3.4
    assert later.win[3] == 8.0
    assert later.race_id == RACE
    assert later.post_utc == POST
    assert later.at_utc == at(5, 59)
    assert later.final is False

    earlier = snapshot.board_at(rows, RACE, at(5, 58), post_utc=POST)
    assert earlier.win[7] == 3.6
    assert earlier.win[3] == 9.0
    assert earlier.at_utc == at(5, 58)

    # a spread of times: the expected price is the last fetch at or before t
    fetches = [(at(5, 40), 4.1), (at(5, 57, 50), 3.6), (at(5, 58, 40), 3.4)]
    for t in [at(5, 40, 1), at(5, 50), at(5, 57, 49), at(5, 57, 51), at(5, 58, 39), at(6, 30)]:
        expected = [odds for when, odds in fetches if when <= t][-1]
        assert snapshot.board_at(rows, RACE, t).win[7] == expected, t


def test_a_row_fetched_exactly_at_t_is_used():
    rows = [
        row("WIN", "7", 4.1, fetched=at(5, 40)),
        row("WIN", "7", 3.6, fetched=at(5, 57, 50)),
    ]
    assert snapshot.board_at(rows, RACE, at(5, 57, 50)).win[7] == 3.6
    assert snapshot.board_at(rows, RACE, at(5, 40)).win[7] == 4.1
    assert snapshot.board_at(rows, RACE, at(5, 57, 49)).win[7] == 4.1


def test_no_row_before_t_gives_none():
    rows = [
        row("WIN", "7", 4.1, fetched=at(5, 40)),
        row("WIN", "7", 3.9, fetched=at(5, 30), race_id=OTHER),  # another race, earlier
    ]
    assert snapshot.board_at(rows, RACE, at(5, 39, 59)) is None
    assert snapshot.board_at(rows, RACE, at(5, 20)) is None
    assert snapshot.board_at([], RACE, at(5, 59)) is None
    assert snapshot.board_at(rows, "jra:2026-11-28:TOK:01", at(6, 30)) is None


def test_board_at_never_uses_a_row_fetched_after_t():
    """Captured (snapshot_utc) before t but fetched after: not available at t, in any pool."""
    rows = [
        row("WIN", "7", 4.1, fetched=at(5, 40)),
        row("PLACE", "7", 1.5, fetched=at(5, 40), odds_hi=1.9),
        row("QUINELLA_PLACE", "3,7", 6.0, fetched=at(5, 40), odds_hi=7.5),
        row("WIN", "__pool__", None, fetched=at(5, 40), investment=100_000_000),
        # a later capture stamped 05:57:30 whose pages arrived at 05:58:30
        row("WIN", "7", 3.4, snap=at(5, 57, 30), fetched=at(5, 58, 30)),
        row("PLACE", "7", 1.3, snap=at(5, 57, 30), fetched=at(5, 58, 30), odds_hi=1.6),
        row("QUINELLA_PLACE", "3,7", 5.0, snap=at(5, 57, 30), fetched=at(5, 58, 30), odds_hi=6.1),
        row(
            "WIN",
            "__pool__",
            None,
            snap=at(5, 57, 30),
            fetched=at(5, 58, 30),
            investment=150_000_000,
        ),
        row("WIN", "9", 12.0, snap=at(5, 57, 30), fetched=at(5, 58, 30)),  # new runner, too late
    ]
    board = snapshot.board_at(rows, RACE, at(5, 58))
    assert board.win == {7: 4.1}
    assert board.place == {7: (1.5, 1.9)}
    assert board.wide == {(3, 7): (6.0, 7.5)}
    assert board.totals == {"WIN": 100_000_000}
    assert board.capture_utc == at(5, 40)

    after = snapshot.board_at(rows, RACE, at(5, 58, 30))
    assert after.win == {7: 3.4, 9: 12.0}
    assert after.place == {7: (1.3, 1.6)}
    assert after.wide == {(3, 7): (5.0, 6.1)}
    assert after.totals == {"WIN": 150_000_000}


def test_rows_of_other_races_are_ignored():
    rows = [
        row("WIN", "7", 4.1, fetched=at(5, 40)),
        row("WIN", "7", 2.2, fetched=at(5, 50), race_id=OTHER),
        row("WIN", "8", 5.5, fetched=at(5, 50), race_id=OTHER),
        row("PLACE", "7", 1.1, fetched=at(5, 50), race_id=OTHER, odds_hi=1.3),
        row("WIN", "__pool__", None, fetched=at(5, 50), race_id=OTHER, investment=9_000_000),
        row("WIN", "4", None, fetched=at(5, 50), race_id=OTHER),
    ]
    board = snapshot.board_at(rows, RACE, at(5, 59))
    assert board.win == {7: 4.1}
    assert board.place == {}
    assert board.totals == {}
    assert board.scratched == frozenset()
    assert board.capture_utc == at(5, 40)

    other = snapshot.board_at(rows, OTHER, at(5, 59))
    assert other.win == {7: 2.2, 8: 5.5}
    assert other.scratched == frozenset({4})


@pytest.mark.parametrize("fetched_key", ["absent", "null"])
def test_hktrack_rows_without_fetched_utc_use_snapshot_utc(fetched_key):
    rows = [
        row("WIN", "7", 4.0, snap=at(5, 50), hktrack=True),
        row("WIN", "7", 3.8, snap=at(5, 55), hktrack=True),
        row("PLACE", "7", 1.4, snap=at(5, 55), odds_hi=1.8, hktrack=True),
    ]
    if fetched_key == "null":
        for r in rows:
            r["fetched_utc"] = None
    first = snapshot.board_at(rows, RACE, at(5, 52))
    assert first.win == {7: 4.0}
    assert first.place == {}
    assert first.capture_utc == at(5, 50)

    second = snapshot.board_at(rows, RACE, at(5, 55))
    assert second.win == {7: 3.8}
    assert second.place == {7: (1.4, 1.8)}
    assert second.capture_utc == at(5, 55)

    assert snapshot.board_at(rows, RACE, at(5, 49, 59)) is None


def test_place_and_wide_carry_low_and_high():
    rows = [
        row("PLACE", "1", 1.1, fetched=at(5, 50), odds_hi=1.2),
        row("PLACE", "12", 3.0, fetched=at(5, 50), odds_hi=4.7),
        row("PLACE", "5", 2.0, fetched=at(5, 50)),  # no odds_hi key
        row("QUINELLA_PLACE", "1,2", 2.6, fetched=at(5, 50), odds_hi=3.3),
        row("QUINELLA_PLACE", "10,3", 8.4, fetched=at(5, 50), odds_hi=9.9),  # given high-low
        row("QUINELLA_PLACE", "4,11", 15.0, fetched=at(5, 50), odds_hi=18.2),
        row("WIN", "1", 2.1, fetched=at(5, 50)),
    ]
    board = snapshot.board_at(rows, RACE, at(5, 55))
    assert board.place[1] == (1.1, 1.2)
    assert board.place == {1: (1.1, 1.2), 12: (3.0, 4.7), 5: (2.0, None)}
    assert board.wide[(1, 2)] == (2.6, 3.3)
    assert board.wide == {(1, 2): (2.6, 3.3), (3, 10): (8.4, 9.9), (4, 11): (15.0, 18.2)}
    assert all(a < b for a, b in board.wide)
    assert board.win == {1: 2.1}


def test_a_wide_combination_given_high_first_keys_low_first():
    rows = [
        row("QUINELLA_PLACE", "2,1", 2.6, fetched=at(5, 50), odds_hi=3.3),
        row("QUINELLA_PLACE", "1,2", 2.4, fetched=at(5, 52), odds_hi=3.0),
    ]
    assert snapshot.board_at(rows, RACE, at(5, 51)).wide == {(1, 2): (2.6, 3.3)}
    # the same pair whichever way it was written: the later row replaces the earlier
    assert snapshot.board_at(rows, RACE, at(5, 53)).wide == {(1, 2): (2.4, 3.0)}


def test_scratched_runners_are_listed_and_unpriced():
    rows = [
        row("WIN", "1", 2.5, fetched=at(5, 40)),
        row("WIN", "4", 6.0, fetched=at(5, 40)),
        row("WIN", "6", None, fetched=at(5, 40)),  # scratched from the start
        row("PLACE", "4", 1.8, fetched=at(5, 40), odds_hi=2.4),
        row("PLACE", "6", None, fetched=at(5, 40)),
        # runner 4 is scratched at 05:50
        row("WIN", "4", None, fetched=at(5, 50)),
        row("PLACE", "4", None, fetched=at(5, 50)),
        row("WIN", "1", 2.3, fetched=at(5, 50)),
    ]
    for r in rows:
        if r["odds"] is None:
            assert r["status_raw"] == "SCR"

    before = snapshot.board_at(rows, RACE, at(5, 45))
    assert before.scratched == frozenset({6})
    assert before.win == {1: 2.5, 4: 6.0}
    assert before.place == {4: (1.8, 2.4)}
    assert before.starters == 2

    after = snapshot.board_at(rows, RACE, at(5, 55))
    assert after.scratched == frozenset({4, 6})
    assert after.win == {1: 2.3}
    assert 4 not in after.place and 6 not in after.place
    assert after.starters == 1
    assert isinstance(after.scratched, frozenset)


def test_a_late_scratch_drops_its_wide_pairs_and_their_staleness():
    """HKTrack's logger stops logging Wide pairs that name a scratched runner, so their last rows
    stay behind at the pre-scratch capture. They are not on the board, so they do not set
    capture_utc; the SCR rows do count, pool totals do not."""
    early, scratch = at(5, 0), at(5, 57)
    rows = [
        row("WIN", "1", 2.5, fetched=early),
        row("WIN", "4", 6.0, fetched=early),
        row("WIN", "7", 9.0, fetched=early),
        row("PLACE", "4", 1.8, fetched=early, odds_hi=2.4),
        row("QUINELLA_PLACE", "1,4", 3.1, fetched=early, odds_hi=3.9),
        row("QUINELLA_PLACE", "4,7", 7.2, fetched=early, odds_hi=8.8),
        row("QUINELLA_PLACE", "1,7", 4.4, fetched=early, odds_hi=5.6),
        row("WIN", "__pool__", None, fetched=early, investment=90_000_000),  # never refreshed
        # runner 4 scratched at 05:57; fresh rows for everything else, no pair naming 4
        row("WIN", "1", 2.2, fetched=scratch),
        row("WIN", "4", None, fetched=scratch),
        row("WIN", "7", 8.5, fetched=scratch),
        row("PLACE", "4", None, fetched=scratch),
        row("QUINELLA_PLACE", "1,7", 4.0, fetched=scratch, odds_hi=5.1),
    ]
    before = snapshot.board_at(rows, RACE, at(5, 30))
    assert before.wide == {(1, 4): (3.1, 3.9), (4, 7): (7.2, 8.8), (1, 7): (4.4, 5.6)}
    assert before.capture_utc == early

    board = snapshot.board_at(rows, RACE, at(5, 58))
    assert board.scratched == frozenset({4})
    assert board.win == {1: 2.2, 7: 8.5}
    assert board.place == {}
    assert board.wide == {(1, 7): (4.0, 5.1)}
    assert (1, 4) not in board.wide and (4, 7) not in board.wide
    assert board.totals == {"WIN": 90_000_000}
    # every row on the board, SCR rows included, was available at 05:57
    assert board.capture_utc == scratch


def test_capture_utc_is_the_oldest_row_used():
    rows = [
        row("WIN", "7", 4.1, fetched=at(5, 50)),
        row("WIN", "3", 9.0, fetched=at(5, 55)),
        row("PLACE", "7", 1.5, fetched=at(5, 57), odds_hi=1.9),
        row("WIN", "7", 4.0, fetched=at(5, 57, 30)),
    ]
    assert snapshot.board_at(rows, RACE, at(5, 50)).capture_utc == at(5, 50)
    assert snapshot.board_at(rows, RACE, at(5, 56)).capture_utc == at(5, 50)
    assert snapshot.board_at(rows, RACE, at(5, 57, 10)).capture_utc == at(5, 50)
    # 7 refreshed at 05:57:30: the oldest row used is now runner 3's, fetched at 05:55
    assert snapshot.board_at(rows, RACE, at(5, 58)).capture_utc == at(5, 55)


def test_pool_totals_come_from_pool_rows():
    rows = [
        row("WIN", "7", 4.1, fetched=at(5, 50)),
        row("WIN", "__pool__", None, fetched=at(5, 50), investment=120_000_000),
        row("PLACE", "__pool__", None, fetched=at(5, 50), investment=34_000_000),
        row("QUINELLA_PLACE", "__pool__", None, fetched=at(5, 50), investment=56_500_000),
        row("WIN", "__pool__", None, fetched=at(5, 56), investment=131_000_000),
        row("PLACE", "__pool__", None, fetched=at(5, 59), investment=40_000_000),  # after t
    ]
    board = snapshot.board_at(rows, RACE, at(5, 57))
    assert board.totals == {"WIN": 131_000_000, "PLACE": 34_000_000, "QUINELLA_PLACE": 56_500_000}
    assert board.win == {7: 4.1}
    assert board.scratched == frozenset()
    assert board.place == {} and board.wide == {}

    early = snapshot.board_at(rows, RACE, at(5, 51))
    assert early.totals == {"WIN": 120_000_000, "PLACE": 34_000_000, "QUINELLA_PLACE": 56_500_000}


@pytest.mark.parametrize(
    "race_id, venue, race_no",
    [("jra:2026-11-28:TOK:11", "TOK", 11), ("jra:2026-11-22:KYO:01", "KYO", 1)],
)
def test_race_context_from_a_board(race_id, venue, race_no):
    runners = (1, 2, 3, 5, 8)
    rows = [row("WIN", str(n), 2.0 + n, fetched=at(5, 50), race_id=race_id) for n in runners]
    rows.append(row("WIN", "4", None, fetched=at(5, 50), race_id=race_id))
    rows.append(row("PLACE", "9", 3.0, fetched=at(5, 50), race_id=race_id, odds_hi=4.0))
    board = snapshot.board_at(rows, race_id, at(5, 55))
    assert board.starters == 5
    ctx = board.context()
    assert ctx == snapshot.RaceContext(race_id=race_id, venue=venue, race_no=race_no, starters=5)
    assert isinstance(ctx.race_no, int)

    fewer = snapshot.board_at(rows[:3], race_id, at(5, 55))
    assert fewer.context().starters == 3


def test_race_board_is_frozen():
    board = snapshot.board_at([row("WIN", "7", 4.1, fetched=at(5, 50))], RACE, at(5, 55))
    assert isinstance(board, snapshot.RaceBoard)
    with pytest.raises(AttributeError):
        board.final = True  # type: ignore[misc]


# ---------------------------------------------------------------- final_board


def _race_to_the_final(hktrack: bool, race_id: str = RACE) -> list[dict]:
    """Non-final captures at 05:58 and 06:00, the first final capture at 06:02, another at 06:05."""
    live = dict(hktrack=hktrack, race_id=race_id)
    if not hktrack:
        live["page_state"] = "発売中"
    rows = []
    for snap, odds7, odds3 in [(at(5, 58), 3.6, 8.0), (at(6, 0), 3.4, 8.2)]:
        fetched = None if hktrack else snap + timedelta(seconds=5)
        rows += [
            row("WIN", "7", odds7, snap=snap, fetched=fetched, **live),
            row("WIN", "3", odds3, snap=snap, fetched=fetched, **live),
            row("PLACE", "7", 1.4, snap=snap, fetched=fetched, odds_hi=1.7, **live),
        ]
    for snap, odds7, odds3 in [(at(6, 2), 3.2, 8.5), (at(6, 5), 3.3, 8.4)]:
        rows += [
            final_row("WIN", "7", odds7, snap=snap, hktrack=hktrack, race_id=race_id),
            final_row("WIN", "3", odds3, snap=snap, hktrack=hktrack, race_id=race_id),
            final_row("PLACE", "7", 1.3, snap=snap, hktrack=hktrack, race_id=race_id, odds_hi=1.5),
        ]
    return rows


def test_final_board_in_the_hktrack_shape():
    rows = _race_to_the_final(hktrack=True)
    assert all("fetched_utc" not in r and "page_state" not in r for r in rows)
    board = snapshot.final_board(rows, RACE, post_utc=POST)
    assert board is not None
    assert board.final is True
    assert board.win == {7: 3.2, 3: 8.5}
    assert board.place == {7: (1.3, 1.5)}
    assert board.post_utc == POST
    assert board.race_id == RACE
    # before the final capture the board is not final
    assert snapshot.board_at(rows, RACE, at(6, 1)).final is False


def test_final_board_in_the_new_shape():
    rows = _race_to_the_final(hktrack=False)
    finals = [r for r in rows if r["source_update_utc"] is None]
    assert finals and all(r["status_raw"] is None and r["page_state"] == FINAL for r in finals)
    board = snapshot.final_board(rows, RACE, post_utc=POST)
    assert board is not None
    assert board.final is True
    assert board.win == {7: 3.2, 3: 8.5}
    assert board.place == {7: (1.3, 1.5)}
    assert board.post_utc == POST
    assert snapshot.board_at(rows, RACE, at(6, 1)).final is False


@pytest.mark.parametrize("hktrack", [True, False])
def test_final_board_takes_the_first_final_capture(hktrack):
    rows = _race_to_the_final(hktrack=hktrack)
    shuffled = rows[::-1]  # log order must not decide which final capture is first
    for given in (rows, shuffled):
        board = snapshot.final_board(given, RACE)
        assert board.win == {7: 3.2, 3: 8.5}, hktrack


@pytest.mark.parametrize("hktrack", [True, False])
def test_final_board_ignores_other_races(hktrack):
    other = _race_to_the_final(hktrack=hktrack, race_id=OTHER)
    live = [row("WIN", "7", 3.6, snap=at(5, 58), fetched=at(5, 58, 5), page_state="発売中")]
    assert snapshot.final_board(other + live, RACE) is None
    assert snapshot.final_board(other + live, OTHER).win == {7: 3.2, 3: 8.5}


@pytest.mark.parametrize("hktrack", [True, False])
def test_a_scratched_runner_does_not_block_the_final_capture(hktrack):
    rows = _race_to_the_final(hktrack=hktrack)
    rows.append(final_row("WIN", "5", None, snap=at(6, 2), hktrack=hktrack))
    rows.append(final_row("WIN", "5", None, snap=at(6, 5), hktrack=hktrack))
    board = snapshot.final_board(rows, RACE)
    assert board.win == {7: 3.2, 3: 8.5}
    assert board.scratched == frozenset({5})


def test_final_board_times_come_from_the_final_capture():
    """at_utc: the latest availability in the final capture; capture_utc: the oldest among its
    price rows. Neither reads another capture."""
    rows = [r for r in _race_to_the_final(hktrack=False) if r["snapshot_utc"] != iso(at(6, 2))]
    snap = at(6, 2)
    for pool, combination, odds, fetched in [
        ("WIN", "7", 3.2, at(6, 2, 7)),
        ("WIN", "3", 8.5, at(6, 2, 3)),
        ("PLACE", "7", 1.3, at(6, 2, 11)),
    ]:
        r = final_row(pool, combination, odds, snap=snap, hktrack=False)
        r["fetched_utc"] = iso(fetched)
        rows.append(r)
    board = snapshot.final_board(rows, RACE)
    assert board.win == {7: 3.2, 3: 8.5}
    assert board.at_utc == at(6, 2, 11)
    assert board.capture_utc == at(6, 2, 3)

    hk = snapshot.final_board(_race_to_the_final(hktrack=True), RACE)
    assert hk.at_utc == at(6, 2)
    assert hk.capture_utc == at(6, 2)


def test_a_race_never_final_gives_none():
    rows = [
        row("WIN", "7", 3.6, fetched=at(5, 58), page_state="発売中"),
        row("WIN", "7", 3.4, snap=at(6, 0), hktrack=True),
        # '最終オッズ' but the source still gives an as-of time: not final
        row("WIN", "7", 3.3, snap=at(6, 2), hktrack=True, status_raw=FINAL),
        row("WIN", "7", 3.2, fetched=at(6, 3), page_state=FINAL, status_raw=None),
        # a final PLACE capture does not make the race final: WIN decides
        final_row("PLACE", "7", 1.3, snap=at(6, 4), hktrack=True, odds_hi=1.5),
    ]
    assert all(r["source_update_utc"] is not None for r in rows if r["pool"] == "WIN")
    assert snapshot.final_board(rows, RACE) is None
    assert snapshot.final_board([], RACE) is None


# ---------------------------------------------------------------- final_board over split captures
# Ticket #234. HKTrack logged a round as separate captures (WIN+PLACE at s, QUINELLA about 1 s on,
# Wide about 2 s on, the next round about 20 s later); a race keeps being captured with final prices
# for hours. Wide's rows can turn final a round before WIN's, or, as on 2026-11-22 KYO races 01, 04
# and 09, a round after (the anchor round's Wide capture still pre-final). Every capture prices
# differently (`sp` reads the capture's stamp), so each assertion names the capture a price came
# from; on the real logs every final capture of a pool repeats one price set.

WIN, PLACE, WIDE, QUINELLA = "WIN", "PLACE", "QUINELLA_PLACE", "QUINELLA"
SPLIT_RACE = "jra:2026-11-22:KYO:05"  # a development day, before the Japan holdout start
SPLIT_POST = datetime(2026, 11, 22, 6, 0, tzinfo=UTC)
POOL_ORDER = (WIN, PLACE, QUINELLA, WIDE)  # horsetrack fetches the pages in this order
FETCH_DELAY_S = {WIN: 3, PLACE: 4, QUINELLA: 5, WIDE: 6}  # new-shape fetched_utc after snapshot_utc
RUNNERS = (1, 2, 3, 4, 5)
PAIRS = [(a, b) for a in range(1, 5) for b in range(a + 1, 5)]  # Wide and QUINELLA pairs logged
WIN_BASE = {1: 2.4, 2: 3.9, 3: 6.5, 4: 12.0, 5: 30.0}
_EPOCH = datetime(2026, 11, 22, 5, 0, tzinfo=UTC)


def kt(hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime(2026, 11, 22, hh, mm, ss, tzinfo=UTC)


def sec(value: datetime, seconds: float) -> datetime:
    return value + timedelta(seconds=seconds)


def sp(pool: str, runners: tuple[int, ...], snap: datetime) -> tuple[float, float | None]:
    """(low, high) of one combination in the capture stamped `snap`: no two captures share one."""
    drift = 1.0 + 0.0001 * (snap - _EPOCH).total_seconds()
    win = [WIN_BASE[r] * drift for r in runners]
    if pool == WIN:
        return round(win[0], 6), None
    if pool == PLACE:
        lo = 1.0 + (win[0] - 1.0) / 3.0
        return round(lo, 6), round(lo * 1.3, 6)
    if pool == WIDE:
        lo = 1.0 + win[0] * win[1] / 4.0
        return round(lo, 6), round(lo * 1.25, 6)
    return round(win[0] * win[1], 6), None  # QUINELLA


def pool_total(pool: str, snap: datetime) -> float:
    return float(10_000_000 * (POOL_ORDER.index(pool) + 1) + (snap - _EPOCH).total_seconds())


def _logged(pool, combination, odds, snap, *, final, hktrack, odds_hi=None, investment=None):
    live = f"{snap.hour + 9}時{snap.minute:02d}分現在"
    if hktrack:
        status, page = (FINAL if final else live), None
    else:
        status, page = (None if final else live), (FINAL if final else "発売中")
    r = row(
        pool,
        combination,
        odds,
        fetched=None if hktrack else sec(snap, FETCH_DELAY_S[pool]),
        snap=snap,
        race_id=SPLIT_RACE,
        odds_hi=odds_hi,
        status_raw=status,
        page_state=page,
        investment=investment,
        hktrack=hktrack,
    )
    if final:
        r["source_update_utc"] = None
    return r


def capture(snap: datetime, pools: dict[str, bool], *, hktrack=True, scratched=(5,)) -> list[dict]:
    """One capture stamped `snap`: per pool (pool -> final) its rows and its pool-total row. WIN and
    PLACE list every runner, a scratched one as SCR; Wide and QUINELLA every pair of 1-4."""
    rows = []
    for pool, final in pools.items():
        keys = [(h,) for h in RUNNERS] if pool in (WIN, PLACE) else PAIRS
        for runners in keys:
            scratch = pool in (WIN, PLACE) and runners[0] in scratched
            lo, hi = (None, None) if scratch else sp(pool, runners, snap)
            combination = ",".join(str(h) for h in runners)
            rows.append(
                _logged(pool, combination, lo, snap, final=final, hktrack=hktrack, odds_hi=hi)
            )
        total = pool_total(pool, snap)
        rows.append(
            _logged(pool, "__pool__", None, snap, final=final, hktrack=hktrack, investment=total)
        )
    return rows


def split_round(start, *, final, wide_final=None, wide_at=(2,), quinella=True, scratched=(5,)):
    """An HKTrack round: WIN+PLACE at `start`, QUINELLA 1 s on, one Wide capture per offset (s)."""
    wide_final = final if wide_final is None else wide_final
    rows = capture(start, {WIN: final, PLACE: final}, scratched=scratched)
    if quinella:
        rows += capture(sec(start, 1), {QUINELLA: wide_final}, scratched=scratched)
    for offset in wide_at:
        rows += capture(sec(start, offset), {WIDE: wide_final}, scratched=scratched)
    return rows


def single_round(start, *, final, wide_final=None, hktrack):
    """A round in one capture: every pool shares one snapshot_utc."""
    wide_final = final if wide_final is None else wide_final
    pools = {WIN: final, PLACE: final, QUINELLA: wide_final, WIDE: wide_final}
    return capture(start, pools, hktrack=hktrack)


def board_prices(win_snap, wide_snap, *, scratched=(5,)):
    """(win, place, wide) as the board shows them: WIN and PLACE from the capture at `win_snap`,
    Wide from the one at `wide_snap` (None: no Wide), pairs naming a scratched runner left out."""
    win = {h: sp(WIN, (h,), win_snap)[0] for h in RUNNERS if h not in scratched}
    place = {h: sp(PLACE, (h,), win_snap) for h in RUNNERS if h not in scratched}
    wide = {}
    if wide_snap is not None:
        wide = {p: sp(WIDE, p, wide_snap) for p in PAIRS if not set(p) & set(scratched)}
    return win, place, wide


def orders(rows: list[dict]) -> list[list[dict]]:
    """The log as written, reversed and shuffled: log order never decides the board."""
    shuffled = list(rows)
    random.Random(234).shuffle(shuffled)
    return [rows, rows[::-1], shuffled]


def pre_final(rows: list[dict], pool: str, combination: str) -> list[dict]:
    """The rows with `pool`'s `combination` row set back to a pre-final state (an as-of time)."""
    out = []
    for r in rows:
        if r["pool"] == pool and r["combination"] == combination:
            r = dict(r, status_raw="15時02分現在", source_update_utc=r["snapshot_utc"])
        out.append(r)
    return out


ANCHOR = kt(6, 2)  # the first capture whose priced WIN rows are all final
LIVE = split_round(kt(5, 58), final=False)
BEFORE = kt(6, 1, 40)  # the round before the anchor's
PRE_ROUND = split_round(BEFORE, final=False)  # nothing final yet
EARLY_WIDE = split_round(BEFORE, final=False, wide_final=True)  # Wide final, WIN not yet
NEXT = split_round(kt(6, 2, 20), final=True)  # the next round: final too, other prices
HOURS_ON = split_round(kt(8, 30), final=True)
ANCHOR_ONLY = split_round(ANCHOR, final=True, quinella=False, wide_at=())  # no Wide capture
PRE_FINAL_WIDE = split_round(ANCHOR, final=True, wide_final=False)  # the KYO 01/04/09 shape
PARTLY_FINAL_WIDE = pre_final(split_round(ANCHOR, final=True), WIDE, "2,3")  # 5 of 6 pairs final
EARLY, NEXT_WIDE = sec(BEFORE, 2), sec(kt(6, 2, 20), 2)

# (rows, the capture Wide comes from (None: no Wide), at_utc, capture_utc)
SPLIT_CASES = [
    pytest.param(
        LIVE + EARLY_WIDE + split_round(ANCHOR, final=True) + NEXT + HOURS_ON,
        *(sec(ANCHOR, 2), sec(ANCHOR, 2), ANCHOR),
        id="final-wide-2s-after-the-anchor",
    ),
    pytest.param(
        LIVE
        + EARLY_WIDE
        + split_round(ANCHOR, final=True, quinella=False, wide_at=(1,))
        + NEXT
        + HOURS_ON,
        *(sec(ANCHOR, 1), sec(ANCHOR, 1), ANCHOR),
        id="final-wide-1s-after-no-quinella",
    ),
    pytest.param(
        LIVE + EARLY_WIDE + split_round(ANCHOR, final=True, wide_at=(19,)) + NEXT + HOURS_ON,
        *(sec(ANCHOR, 19), sec(ANCHOR, 19), ANCHOR),
        id="final-wide-19s-after",
    ),
    pytest.param(
        LIVE + EARLY_WIDE + split_round(ANCHOR, final=True, wide_at=(2, 9)) + NEXT + HOURS_ON,
        *(sec(ANCHOR, 2), sec(ANCHOR, 2), ANCHOR),
        id="two-final-wide-captures-in-the-round-the-first",
    ),
    pytest.param(
        LIVE
        + split_round(kt(6, 1, 20), final=False, wide_final=True)
        + EARLY_WIDE
        + split_round(ANCHOR, final=True)
        + NEXT,
        *(sec(ANCHOR, 2), sec(ANCHOR, 2), ANCHOR),
        id="two-early-final-wide-rounds",
    ),
    pytest.param(
        LIVE + split_round(ANCHOR, final=True),
        *(sec(ANCHOR, 2), sec(ANCHOR, 2), ANCHOR),
        id="the-anchor-round-ends-the-log",
    ),
    pytest.param(
        LIVE + PRE_ROUND + PRE_FINAL_WIDE + NEXT + HOURS_ON,
        *(NEXT_WIDE, NEXT_WIDE, ANCHOR),
        id="pre-final-wide-in-the-anchor-round",
    ),
    pytest.param(
        LIVE + EARLY_WIDE + PRE_FINAL_WIDE + NEXT + HOURS_ON,
        *(NEXT_WIDE, NEXT_WIDE, ANCHOR),
        id="pre-final-in-round-next-final-beats-an-early-final",
    ),
    pytest.param(
        LIVE + PRE_ROUND + PARTLY_FINAL_WIDE + NEXT + HOURS_ON,
        *(NEXT_WIDE, NEXT_WIDE, ANCHOR),
        id="partly-final-wide-in-the-anchor-round",
    ),
    pytest.param(
        LIVE + EARLY_WIDE + ANCHOR_ONLY + NEXT + HOURS_ON,
        *(NEXT_WIDE, NEXT_WIDE, ANCHOR),
        id="no-wide-capture-in-the-anchor-round",
    ),
    pytest.param(
        EARLY_WIDE + ANCHOR_ONLY + single_round(kt(6, 2, 20), final=True, hktrack=True) + HOURS_ON,
        *(kt(6, 2, 20), kt(6, 2, 20), ANCHOR),
        id="next-round-in-one-capture",
    ),
    pytest.param(
        LIVE + EARLY_WIDE + PRE_FINAL_WIDE,
        *(EARLY, ANCHOR, EARLY),
        id="an-early-final-wide-when-none-at-or-after",
    ),
    pytest.param(
        LIVE + split_round(kt(6, 1, 20), final=False, wide_final=True) + EARLY_WIDE + ANCHOR_ONLY,
        *(EARLY, ANCHOR, EARLY),
        id="the-latest-early-final-wide-when-none-after",
    ),
    pytest.param(
        LIVE + PRE_ROUND + PRE_FINAL_WIDE + split_round(kt(6, 2, 20), final=True, wide_final=False),
        *(None, ANCHOR, ANCHOR),
        id="no-final-wide-anywhere",
    ),
]


@pytest.mark.parametrize(("rows", "wide_snap", "at_utc", "capture_utc"), SPLIT_CASES)
def test_a_split_final_board_takes_wide_from_its_first_final_capture(
    rows, wide_snap, at_utc, capture_utc
):
    """WIN and PLACE (final in the anchor) from the anchor. Wide, prices and total, from its first
    final capture at or after the anchor, of whatever round; else its latest final capture before
    the anchor; else none. at_utc: when the last row taken was in hand."""
    win, place, wide = board_prices(ANCHOR, wide_snap)
    assert len(wide) == (0 if wide_snap is None else 6)
    expected_totals = {WIN: pool_total(WIN, ANCHOR), PLACE: pool_total(PLACE, ANCHOR)}
    if wide_snap is not None:
        expected_totals[WIDE] = pool_total(WIDE, wide_snap)
    for given in orders(rows):
        board = snapshot.final_board(given, SPLIT_RACE, post_utc=SPLIT_POST)
        assert board is not None
        assert board.wide == wide
        assert board.win == win
        assert board.place == place
        assert board.totals == expected_totals
        assert board.final is True
        assert board.scratched == frozenset({5})
        assert board.at_utc == at_utc
        assert board.capture_utc == capture_utc
        assert board.post_utc == SPLIT_POST
        assert board.race_id == SPLIT_RACE
        assert board.starters == 4


def test_a_split_wide_capture_still_drops_the_pairs_of_a_scratched_runner():
    """Runner 4 is scratched after the live round (SCR in WIN and PLACE); the Wide captures go on
    pricing its pairs. They are not on the final board, and the SCR rows come from the anchor."""
    late = (4, 5)
    rows = (
        LIVE
        + split_round(kt(6, 1, 40), final=False, wide_final=True, scratched=late)
        + split_round(ANCHOR, final=True, scratched=late)
        + split_round(kt(6, 2, 20), final=True, scratched=late)
    )
    in_round_wide = [
        r for r in rows if r["pool"] == WIDE and r["snapshot_utc"] == iso(sec(ANCHOR, 2))
    ]
    assert any("4" in r["combination"].split(",") and r["odds"] is not None for r in in_round_wide)
    win, place, wide = board_prices(ANCHOR, sec(ANCHOR, 2), scratched=late)
    assert set(wide) == {(1, 2), (1, 3), (2, 3)}
    for given in orders(rows):
        board = snapshot.final_board(given, SPLIT_RACE)
        assert board.wide == wide
        assert board.scratched == frozenset(late)
        assert board.win == win and board.place == place
        assert board.starters == 3
        assert board.capture_utc == ANCHOR


@pytest.mark.parametrize("hktrack", [True, False], ids=["hktrack-shape", "horsetrack-shape"])
def test_single_capture_rounds_give_the_anchor_captures_board(hktrack):
    """One snapshot_utc per round across pools (horsetrack's own log-day, and the HKTrack row shape
    in the same layout): the final board is the first final capture's alone, as before #234, though
    the Wide of the round before is already final and later final rounds price differently."""
    rows = (
        single_round(kt(5, 58), final=False, hktrack=hktrack)
        + single_round(kt(6, 1, 40), final=False, wide_final=True, hktrack=hktrack)
        + single_round(ANCHOR, final=True, hktrack=hktrack)
        + single_round(kt(6, 2, 20), final=True, hktrack=hktrack)
        + single_round(kt(8, 30), final=True, hktrack=hktrack)
    )
    assert sp(WIN, (1,), ANCHOR) != sp(WIN, (1,), kt(6, 2, 20))
    assert sp(WIDE, (1, 2), ANCHOR) != sp(WIDE, (1, 2), kt(6, 1, 40))
    win, place, wide = board_prices(ANCHOR, ANCHOR)
    expected = snapshot.RaceBoard(
        race_id=SPLIT_RACE,
        post_utc=SPLIT_POST,
        # the capture's last row in hand (Wide's page, fetched last), and its oldest price row
        at_utc=ANCHOR if hktrack else sec(ANCHOR, FETCH_DELAY_S[WIDE]),
        capture_utc=ANCHOR if hktrack else sec(ANCHOR, FETCH_DELAY_S[WIN]),
        final=True,
        win=win,
        place=place,
        wide=wide,
        totals={pool: pool_total(pool, ANCHOR) for pool in POOL_ORDER},
        scratched=frozenset({5}),
    )
    assert len(expected.wide) == 6
    for given in orders(rows):
        assert snapshot.final_board(given, SPLIT_RACE, post_utc=SPLIT_POST) == expected


@pytest.mark.parametrize("hktrack", [True, False], ids=["hktrack-shape", "horsetrack-shape"])
@pytest.mark.parametrize("pool", [WIDE, PLACE])
def test_a_pool_still_pre_final_in_the_anchor_capture_comes_from_its_next_final_capture(
    pool, hktrack
):
    """The one deliberate change from the board before #234. The anchor capture's WIN rows are
    final but `pool`'s rows in the same capture are not yet: the old board put those pre-final
    prices on the final board, the target the nowcast trains against. Now that pool, prices and
    total, comes from its first final capture after the anchor (the next round here); the rest of
    the board is the anchor capture's, as before."""
    nxt = kt(6, 2, 20)
    at_anchor = {WIN: True, PLACE: True, QUINELLA: True, WIDE: True}
    at_anchor[pool] = False
    rows = (
        single_round(kt(5, 58), final=False, hktrack=hktrack)
        + capture(ANCHOR, at_anchor, hktrack=hktrack)
        + single_round(nxt, final=True, hktrack=hktrack)
        + single_round(kt(8, 30), final=True, hktrack=hktrack)
    )
    win, place, wide = board_prices(ANCHOR, ANCHOR)
    _, next_place, next_wide = board_prices(nxt, nxt)
    totals = {p: pool_total(p, ANCHOR) for p in POOL_ORDER}
    totals[pool] = pool_total(pool, nxt)
    expected = snapshot.RaceBoard(
        race_id=SPLIT_RACE,
        post_utc=SPLIT_POST,
        at_utc=nxt if hktrack else sec(nxt, FETCH_DELAY_S[pool]),  # the replacement's rows
        capture_utc=ANCHOR if hktrack else sec(ANCHOR, FETCH_DELAY_S[WIN]),
        final=True,
        win=win,
        place=next_place if pool == PLACE else place,
        wide=next_wide if pool == WIDE else wide,
        totals=totals,
        scratched=frozenset({5}),
    )
    for given in orders(rows):
        assert snapshot.final_board(given, SPLIT_RACE, post_utc=SPLIT_POST) == expected


@pytest.mark.parametrize("pool", [WIDE, PLACE])
def test_a_pool_never_final_is_left_off_the_final_board(pool):
    """WIN is final from the anchor on but `pool` never turns final in any capture: the board shows
    none of its prices and no total for it, where the old board showed the anchor's pre-final
    ones."""
    at_anchor = {WIN: True, PLACE: True, QUINELLA: True, WIDE: True}
    at_anchor[pool] = False
    rows = (
        single_round(kt(5, 58), final=False, hktrack=True)
        + capture(ANCHOR, at_anchor)
        + capture(kt(6, 2, 20), at_anchor)
    )
    win, place, wide = board_prices(ANCHOR, ANCHOR)
    for given in orders(rows):
        board = snapshot.final_board(given, SPLIT_RACE)
        assert board.place == ({} if pool == PLACE else place)
        assert board.wide == ({} if pool == WIDE else wide)
        assert board.totals == {p: pool_total(p, ANCHOR) for p in POOL_ORDER if p != pool}
        assert board.win == win
        assert board.scratched == frozenset({5})
        assert board.final is True
        assert board.at_utc == ANCHOR


# ---------------------------------------------------------------- post times


def _sidecar(race_date: date, venue: str) -> Path:
    log = paths.odds_snapshot_path("jra", race_date.isoformat(), venue)
    return log.with_name(f"{venue}.races.json")


def _card(race_date: date, venue: str, races: dict[int, str]) -> Path:
    """HKTrack's stored card of one meeting: races.parquet (race_id, race_no, post_time_local)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder = paths.processed_dir("jra") / "upcoming" / f"jra_{race_date.isoformat()}_{venue}"
    folder.mkdir(parents=True, exist_ok=True)
    nos = sorted(races)
    table = pa.table(
        {
            "race_id": [f"jra:{race_date.isoformat()}:{venue}:{no:02d}" for no in nos],
            "race_no": pa.array(nos, type=pa.int64()),
            "post_time_local": [races[no] for no in nos],
        }
    )
    path = folder / "races.parquet"
    pq.write_table(table, path)
    return path


def _aware_utc(values) -> bool:
    return all(v.tzinfo is not None and v.utcoffset() == timedelta(0) for v in values)


def test_post_times_prefer_the_sidecar(data_dir):
    day = date(2026, 11, 28)
    sidecar = {
        1: datetime(2026, 11, 28, 1, 5, tzinfo=UTC),
        11: datetime(2026, 11, 28, 6, 40, tzinfo=UTC),
    }
    updated = datetime(2026, 11, 28, 0, 30, tzinfo=UTC)
    odds_log.write_post_times(_sidecar(day, "TOK"), sidecar, updated)
    _card(day, "TOK", {1: "2026-11-28T10:15:00+09:00", 2: "2026-11-28T10:45:00+09:00"})
    got = snapshot.post_times_for(day, "TOK")
    assert got == sidecar
    assert _aware_utc(got.values())
    assert all(isinstance(no, int) for no in got)


def test_post_times_fall_back_to_the_hktrack_card(data_dir):
    day = date(2026, 11, 22)
    card = {
        1: "2026-11-22T10:05:00+09:00",
        2: "2026-11-22T10:35:00+09:00",
        12: "2026-11-22T16:30:00+09:00",
    }
    _card(day, "KYO", card)
    got = snapshot.post_times_for(day, "KYO")
    assert got[1] == datetime(2026, 11, 22, 1, 5, tzinfo=UTC)
    assert got == {
        1: datetime(2026, 11, 22, 1, 5, tzinfo=UTC),
        2: datetime(2026, 11, 22, 1, 35, tzinfo=UTC),
        12: datetime(2026, 11, 22, 7, 30, tzinfo=UTC),
    }
    assert _aware_utc(got.values())
    assert all(isinstance(no, int) for no in got)
    # another meeting's sidecar or card is not this one's
    odds_log.write_post_times(
        _sidecar(day, "TOK"),
        {1: datetime(2026, 11, 22, 0, 50, tzinfo=UTC)},
        datetime(2026, 11, 22, 0, 0, tzinfo=UTC),
    )
    assert snapshot.post_times_for(day, "KYO") == got


def test_a_card_time_without_an_offset_is_japan_time(data_dir):
    day = date(2026, 11, 22)
    _card(day, "KYO", {1: "2026-11-22T10:05:00", 12: "2026-11-22T16:30:00+09:00"})
    got = snapshot.post_times_for(day, "KYO")
    assert got == {
        1: datetime(2026, 11, 22, 1, 5, tzinfo=UTC),
        12: datetime(2026, 11, 22, 7, 30, tzinfo=UTC),
    }
    assert _aware_utc(got.values())


def test_post_times_with_neither_source_are_empty(data_dir):
    _card(date(2026, 11, 22), "KYO", {1: "2026-11-22T10:05:00+09:00"})
    assert snapshot.post_times_for(date(2026, 11, 28), "TOK") == {}
    assert snapshot.post_times_for(date(2026, 11, 22), "TOK") == {}


# ---------------------------------------------------------------- decision grid


def test_bucket_for_maps_onto_the_grid():
    grid = snapshot.DecisionGrid(minutes=(60, 30, 16, 5, 2, 1), primary=2)
    cases = {
        70: 60,
        600: 60,
        60: 60,
        59.9: 60,
        30.01: 60,
        30: 30,
        16.5: 30,
        16: 16,
        5.01: 16,
        5: 5,
        3.5: 5,
        2.01: 5,
        2: 2,
        1.5: 2,
        1: 1,
        0.5: 1,
        0.01: 1,
        -0.1: 0,
        -5: 0,
    }
    for minutes, expected in cases.items():
        assert snapshot.bucket_for(minutes, grid) == expected, minutes

    other = snapshot.DecisionGrid(minutes=(45, 10, 3), primary=10)
    other_cases = {90: 45, 45: 45, 44: 45, 10.5: 45, 10: 10, 4: 10, 3: 3, 0.2: 3, -1: 0}
    for minutes, expected in other_cases.items():
        assert snapshot.bucket_for(minutes, other) == expected, minutes


def test_the_shipped_grid_is_d2():
    grid = snapshot.load_decision_grid()
    assert grid == snapshot.DecisionGrid(minutes=(60, 30, 16, 5, 2, 1), primary=2)
    assert tuple(grid.minutes) == (60, 30, 16, 5, 2, 1)
    assert grid.primary == 2
    assert grid.primary in grid.minutes


def test_load_decision_grid_reads_the_given_file(tmp_path):
    path = tmp_path / "decision.yaml"
    path.write_text("grid_min: [45, 15, 3]\nprimary_min: 15\n", encoding="utf-8")
    grid = snapshot.load_decision_grid(path)
    assert grid == snapshot.DecisionGrid(minutes=(45, 15, 3), primary=15)
    assert isinstance(grid.minutes, tuple)
