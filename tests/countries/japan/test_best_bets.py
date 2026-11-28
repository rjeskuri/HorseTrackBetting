"""Ticket #714: the best bets of a venue's next race, read on the nowcast's predicted final board.

Contract pinned here for `horsetrack.countries.japan.board.best_bets` (the coder implements what
these tests call; the server route is in test_best_bets_server.py, the page in
test_best_bets_page.py):

- `checkpoint_for(minutes_to_post) -> int`: 10 while more than 5 min to the post, 5 while more
  than 2, 2 from post - 2 on (post - 5 exactly is T-5, post - 2 exactly is T-2).
- `eligible(rows, checkpoint_min) -> list[LeaderboardRow]`: the rows that may be best bets at
  T-k: selection `predicted_final@T-k` (that checkpoint only) or `decision_time`, a shrunk value,
  n >= 100 (whatever its n_unit; #713's rows count races) and n_meetings >= 20. Licensed rows
  count (the page is token-gated and not in the repository). look_ahead, card_only and unknown
  rows, and the other checkpoints' rows, never.
- `match(rows, board, *, race) -> {"best_bets": [...], "unmatched": [...]}`: `rows` eligible,
  `board` the projected (predicted final) `RaceBoard`, `race` the card as a
  `predicted_final.RaceInput`. Each row whose rule is a `predicted_final.RULES` rule of its
  family is read with that rule's matcher (`predicted_final.select`) on `board`; every other row
  (no matcher for its family, an unknown rule, an unmeasured rule) is listed in `unmatched`
  ({"option_id", "family_id", ...}), never dropped; a matched row that buys nothing on the board
  is in neither list. Each ticket (pool, combination: the feed's text, "2" or "2,4", a Trio's
  horses ascending, an Exacta's or Trifecta's in finishing order) appears once, under its row of
  the highest shrunk return; the other rows buying it are its `duplicates` (each {"option_id",
  ...}, shrunk descending). Tickets are ranked by that shrunk return, descending (`rank` 1..n),
  and each carries option_id, family_id, rule, `description` (the rule in plain English,
  `RULES[rule].description`), shrunk_per_100, return_per_100, ci_lo, ci_hi, n, n_meetings,
  variants_searched and `highlight`: "strong" at a shrunk return of ¥100 or more, "muted" below.
  The page's Model % plays no part.
- `build_best_bets(race_date, *, venue, leaderboard, models, now, day_dir=None, cache=None)`: the
  venue's next race (its first race whose post time, read from the post-time sidecar as the feeds
  read it, so a delayed post counts, is after now()) as {"race": {"race_id", "race_no",
  "post_utc"} or None, "checkpoint_min", "frozen", "best_bets", "unmatched"}. Before post - 2 the
  board is the board as of now(), projected with `models.nowcast` (provisional); from post - 2 on it
  is the board as of post - 2 (frozen: a capture after post - 2 changes nothing). A day with no
  race left at the venue gives race None and no best bets.
- `LeaderboardFile(path).rows()`: the rows of the private leaderboard JSON (`to_json`'s text),
  read again when the file changes; [] while there is no file.

The rows, boards and nowcast are synthetic (best_bets_fakes.py); every expected ticket is worked
out by hand there.
"""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime

import pandas as pd
import pytest

from horsetrack.core import paths
from horsetrack.core.eval.leaderboard import to_json
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board.snapshot import RaceBoard
from horsetrack.countries.japan.research import predicted_final as pf

# isort: split
import best_bets_fakes as fx

UTC = fx.UTC


class _Module:
    """`best_bets`, imported at first use: until the ticket creates it each test fails on its
    own (ImportError) instead of the whole file failing to collect."""

    def __getattr__(self, name: str):
        import importlib

        return getattr(importlib.import_module("horsetrack.countries.japan.board.best_bets"), name)


bb = _Module()


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


def projected_board(name: str) -> RaceBoard:
    at = fx.jst(10, 0).astimezone(UTC)
    return RaceBoard(
        race_id=fx.rid("TOK", 1),
        post_utc=fx.POSTS[("TOK", 1)].astimezone(UTC),
        at_utc=at,
        capture_utc=at,
        final=False,
        win=dict(fx.PROJECTED[name]),
        place={},
        wide={},
        totals={},
        scratched=frozenset(),
    )


def card() -> pf.RaceInput:
    return pf.RaceInput(
        race_id=fx.rid("TOK", 1),
        race_date=fx.DAY,
        meeting_id=f"{fx.DAY.isoformat()}:TOK",
        race_class="MAIDEN",
        surface="turf",
        boards={},
        final=None,
        paid=pd.DataFrame(columns=["race_id", "pool", "combination", "dividend", "is_refund"]),
    )


def eligible_by_hand(k: int):
    return fx.scored_rows(k) + fx.unmatched_rows(k) + [fx.DECISION_ROW]


# --------------------------------------------------------------------------- the checkpoint


@pytest.mark.parametrize(
    ("minutes", "checkpoint"),
    [
        (45.0, 10), (10.5, 10), (10.0, 10), (7.25, 10), (5.01, 10),
        (5.0, 5), (4.0, 5), (3.3, 5), (2.01, 5),
        (2.0, 2), (1.5, 2), (0.2, 2), (0.0, 2),
    ],
)  # fmt: skip
def test_the_checkpoint_follows_the_minutes_to_post(minutes, checkpoint):
    assert bb.checkpoint_for(minutes) == checkpoint


# --------------------------------------------------------------------------- eligibility


@pytest.mark.parametrize("k", fx.CHECKPOINTS)
def test_only_measured_rows_of_the_checkpoint_over_the_thresholds_are_eligible(k):
    rows = fx.all_rows()
    got = [r.option_id for r in bb.eligible(rows, k)]
    assert len(got) == len(set(got))
    assert set(got) == fx.eligible_ids(k)
    # the licensed row and the row exactly at 100 races and 20 meetings are in
    assert fx.option_id(fx.TOP1, k) in got and fx.option_id(fx.WIDE12, k) in got


def test_eligibility_moves_with_each_threshold():
    base = fx.row(fx.TOP1, 120.0, k=5)
    cases = [
        (base, True),
        (replace(base, n=100), True),
        (replace(base, n=99), False),
        (replace(base, n=None), False),
        (replace(base, n_meetings=20), True),
        (replace(base, n_meetings=19), False),
        (replace(base, n_meetings=None), False),
        (replace(base, n=100, n_meetings=20, n_unit="bets"), True),
        (replace(base, shrunk_per_100=None), False),
        (replace(base, shrunk_per_100=40.0), True),  # a low return is still a row
        (replace(base, licensed=True), True),
        (replace(base, selection="predicted_final@T-10"), False),
        (replace(base, selection="predicted_final@T-2"), False),
        (replace(base, selection="decision_time"), True),
        (replace(base, selection="look_ahead"), False),
        (replace(base, selection="card_only"), False),
        (replace(base, selection="unknown"), False),
    ]
    for candidate, want in cases:
        got = [r.option_id for r in bb.eligible([candidate], 5)]
        assert got == ([candidate.option_id] if want else []), candidate


# --------------------------------------------------------------------------- matching


@pytest.mark.parametrize("board", ["A", "B"])
@pytest.mark.parametrize("k", fx.CHECKPOINTS)
def test_rows_are_matched_on_the_projected_board_one_ticket_once(board, k):
    found = bb.match(eligible_by_hand(k), projected_board(board), race=card())
    fx.check_bets(found["best_bets"], board, k, f"{board} T-{k}")
    assert {u["option_id"] for u in found["unmatched"]} == fx.unmatched_ids(k)
    unmatched = {u["option_id"]: u for u in found["unmatched"]}
    assert unmatched[fx.DECISION_OID]["family_id"] == fx.DECISION_FAMILY


def test_the_top_three_follow_the_hand_computed_matches():
    """Spot checks of the reference itself, so the first three are read off by hand too."""
    w2, w23, w24 = ("WIN", "2"), ("QUINELLA_PLACE", "2,3"), ("QUINELLA_PLACE", "2,4")
    f12, f13 = ("FORECAST", "1,2"), ("FORECAST", "1,3")
    wd12, wd13 = ("QUINELLA_PLACE", "1,2"), ("QUINELLA_PLACE", "1,3")
    # per rank 1, 2, 3: the tickets allowed there (a tie may come in either order), and colour
    top = {
        ("A", 10): ([{w2}, {w23, w24}, {w23, w24}], ["strong", "strong", "strong"]),
        ("B", 2): ([{f12, f13}, {f12, f13}, {wd12, wd13}], ["strong", "strong", "strong"]),
        ("B", 5): ([{("TIERCE", "1,2,3")}, {("TRIO", "1,2,3")}, {f12, f13}],
                   ["strong", "strong", "muted"]),
    }  # fmt: skip
    for (board, k), (allowed, colours) in top.items():
        found = bb.match(eligible_by_hand(k), projected_board(board), race=card())["best_bets"]
        got = [(e["pool"], e["combination"]) for e in found[:3]]
        assert len(set(got)) == 3, (board, k, got)
        assert all(t in ok for t, ok in zip(got, allowed, strict=True)), (board, k, got)
        assert [e["highlight"] for e in found[:3]] == colours, (board, k)
    # B at T-5: WIN 1 is bought by BAND35 (99.5) and TOP1 (88): listed once, TOP1 under it
    found = bb.match(eligible_by_hand(5), projected_board("B"), race=card())["best_bets"]
    win1 = [e for e in found if (e["pool"], e["combination"]) == ("WIN", "1")]
    assert len(win1) == 1
    assert win1[0]["rule"] == fx.BAND35
    assert [d["option_id"] for d in win1[0]["duplicates"]] == [fx.option_id(fx.TOP1, 5)]
    # the colour moves with the shrunk value: 99.99 is muted, 100.0 strong
    exacta = [e for e in found if e["pool"] == "FORECAST"]
    assert exacta and all(e["highlight"] == "muted" for e in exacta)


def test_a_row_without_a_matcher_is_listed_not_dropped():
    lone = [fx.DECISION_ROW, fx.row(fx.UNKNOWN_RULE, 180.0, k=10)]
    found = bb.match(lone, projected_board("A"), race=card())
    assert found["best_bets"] == []
    assert {u["option_id"] for u in found["unmatched"]} == {r.option_id for r in lone}


def test_a_matched_rule_that_buys_nothing_is_neither_a_bet_nor_unmatched():
    found = bb.match([fx.row(fx.E15, 150.0, k=10)], projected_board("A"), race=card())
    assert found["best_bets"] == [] and found["unmatched"] == []


# --------------------------------------------------------------------------- the venue's next race


def build(venue: str, at: datetime, rows=None) -> dict:
    return bb.build_best_bets(
        fx.DAY,
        venue=venue,
        leaderboard=fx.all_rows() if rows is None else rows,
        models=fx.make_models(),
        now=lambda: at.astimezone(UTC),
    )


def check_payload(got: dict, venue, no, checkpoint, frozen, board, post=None) -> None:
    where = f"{venue} {no} T-{checkpoint}"
    assert got["race"]["race_id"] == fx.rid(venue, no), where
    assert got["race"]["race_no"] == no, where
    post = post or fx.POSTS[(venue, no)]
    assert datetime.fromisoformat(got["race"]["post_utc"]) == post, where
    assert got["checkpoint_min"] == checkpoint, where
    assert got["frozen"] is frozen, where
    fx.check_bets(got["best_bets"], board, checkpoint, where)
    assert {u["option_id"] for u in got["unmatched"]} == fx.unmatched_ids(checkpoint), where


@pytest.mark.parametrize(
    ("venue", "now", "no", "checkpoint", "frozen", "board"),
    fx.TIMELINE,
    ids=[f"{v}-{t:%H%M%S}" for v, t, *_ in fx.TIMELINE],
)
def test_each_venue_shows_its_next_race_provisional_then_frozen(
    venue, now, no, checkpoint, frozen, board
):
    fx.write_day()
    check_payload(build(venue, now), venue, no, checkpoint, frozen, board)


def test_a_delayed_post_keeps_the_race_until_the_later_post():
    fx.write_day()
    sidecar = paths.odds_snapshot_path("jra", fx.DAY.isoformat(), "KYO").with_name("KYO.races.json")
    posts = {3: fx.DELAYED_KYO3.astimezone(UTC), 4: fx.POSTS[("KYO", 4)].astimezone(UTC)}
    odds_log.write_post_times(sidecar, posts, fx.jst(10, 10).astimezone(UTC))
    for venue, now, no, checkpoint, frozen, board in fx.DELAYED_TIMELINE:
        post = fx.DELAYED_KYO3 if (venue, no) == ("KYO", 3) else None
        check_payload(build(venue, now), venue, no, checkpoint, frozen, board, post)


def test_after_the_last_race_there_is_no_next_race():
    fx.write_day()
    for venue, at in (("TOK", fx.jst(10, 40)), ("TOK", fx.jst(12, 0)), ("KYO", fx.jst(10, 55))):
        got = build(venue, at)
        assert got["race"] is None, (venue, at)
        assert got["best_bets"] == [], (venue, at)


def test_the_leaderboard_decides_the_bets_not_the_model():
    """Same day, same moment, other leaderboard values: other bets; the models are unchanged."""
    fx.write_day()
    at = fx.jst(9, 59)  # TOK 1 at T-10 on board A
    flipped = [
        replace(r, shrunk_per_100=200.0) if r.rule == fx.QUIN12 and r.shrunk_per_100 else r
        for r in fx.all_rows()
    ]
    first = build("TOK", at)["best_bets"][0]
    second = build("TOK", at, flipped)["best_bets"]
    assert (first["pool"], first["combination"]) == ("WIN", "2")
    assert {(e["pool"], e["combination"]) for e in second[:2]} == {
        ("QUINELLA", "2,3"),
        ("QUINELLA", "2,4"),
    }


# --------------------------------------------------------------------------- the leaderboard file


def test_the_leaderboard_file_is_read_and_read_again_when_it_changes(tmp_path):
    path = paths.reports_dir() / "leaderboard-japan.json"
    source = bb.LeaderboardFile(path)
    assert source.rows() == []
    first = fx.scored_rows(10)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(first), encoding="utf-8")
    got = source.rows()
    assert [(r.option_id, r.shrunk_per_100) for r in got] == [
        (r.option_id, r.shrunk_per_100) for r in first
    ]
    second = fx.scored_rows(5) + [fx.DECISION_ROW]
    path.write_text(to_json(second), encoding="utf-8")
    stamp = path.stat().st_mtime_ns + 5_000_000_000
    os.utime(path, ns=(stamp, stamp))
    again = source.rows()
    assert [(r.option_id, r.shrunk_per_100) for r in again] == [
        (r.option_id, r.shrunk_per_100) for r in second
    ]
