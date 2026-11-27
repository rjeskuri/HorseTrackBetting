"""The exotic pool-choice study (#81) as #417's representative caller of the model boundary.

Contract pinned here (`horsetrack.countries.japan.studies.avoidance`, migrated in #417; the
boundary is `horsetrack.countries.japan.feature_boundary`, see tests/countries/japan/
test_feature_boundary.py):

- `view_at(con, race_id, at_utc, *, rules)` reads the board at t through the boundary
  (`feature_view`), not the whole as-of projection: a column added to odds_timeseries that no
  config classifies never reaches `RaceView.board`, while the board at t keeps its prices. Every
  board column is an approved feature of odds_timeseries (`approved_features`) or one of the
  board's identifiers (race_id, pool, combination, snapshot_utc, fetched_utc, post_utc).
- `render_report(study, spec, about)` identifies the study's predictors that carry a documented
  approximation (`feature_boundary.approximations`): the starters come from runners.is_scratched
  and the races from races.status, both results-derived card values (docs/specs/as-of.md,
  "Accepted leak: results-derived card columns"). Each is named on a report line that carries the
  `RESULTS_DERIVED` tag, with its table.
- The odds_timeseries rule's fallback (availability.yaml: `column: fetched_utc`, `fallback:
  snapshot_utc`): a capture whose `fetched_utc` is null is on the board at t when its
  `snapshot_utc` is at or before t, and `view_at` reads a board without `post_utc`. The report
  discloses the fallback: `render_report` on `compute_study`'s output has a line with
  `feature_boundary.TIMESTAMP_FALLBACK` that names `odds_timeseries.odds` and the number of races
  whose board at t (the latest capture per pool and combination) holds such a capture.

Every frame is synthetic; no date is a Japan holdout day (on or after 2026-11-28).
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

import duckdb
import pandas as pd
import pytest

from horsetrack.core.config import load_country_yaml
from horsetrack.core.store.asof import load_availability
from horsetrack.countries.japan.studies import avoidance
from horsetrack.countries.japan.studies.calibration import Study

UTC = timezone.utc
NAN = float("nan")
ODDS_LABELS = ("dividend", "is_refund", "is_final_capture")
IDENTIFIERS = {"race_id", "pool", "combination", "snapshot_utc", "fetched_utc", "post_utc"}
DAY = datetime(2026, 6, 7, tzinfo=UTC)
RACE, OTHER = "jra:2026-06-07:TOK:11", "jra:2026-06-07:KYO:11"
# New columns a later logger schema could add, one per kind of value.
NEW_COLUMNS = [
    ("pace_signal", lambda i: 0.25 * i),
    ("is_hot", lambda i: i % 2 == 1),
    ("odds_note", lambda i: f"n{i}"),
]


def at(hh: int, mm: int, ss: int = 0) -> datetime:
    return DAY + timedelta(hours=hh, minutes=mm, seconds=ss)


T, POST = at(5, 55), at(6, 2)
# (race, pool, combination, snapshot, fetched, odds, final)
CAPTURES = [
    (RACE, "TRIO", "2,9,14", at(5, 30), at(5, 30, 4), 18.5, False),
    (RACE, "TRIO", "2,9,14", at(5, 50), at(5, 50, 3), 21.0, False),
    (RACE, "TRIO", "2,9,14", at(5, 54), at(5, 56, 10), 95.0, False),  # fetched after t
    (RACE, "TRIO", "2,9,14", at(6, 1), at(6, 1, 5), 88.4, True),
    (RACE, "WIN", "9", at(5, 50), at(5, 50, 3), 3.2, False),
    (RACE, "WIN", "3", at(5, 40), at(5, 40, 1), 7.7, False),
    (OTHER, "TRIO", "2,9,14", at(5, 53), at(5, 53, 30), 150.0, False),
]
AT_T = {("TRIO", "2,9,14"): 21.0, ("WIN", "9"): 3.2, ("WIN", "3"): 7.7}  # RACE's board at T


def _board_con(new=None, captures=CAPTURES, *, drop=(), posts=None) -> duckdb.DuckDBPyConnection:
    """An in-memory odds_timeseries of `captures` (race -> post from `posts`, default POST),
    without the columns in `drop`."""
    posts = posts or {}
    frame = pd.DataFrame(
        [
            {
                "race_id": race,
                "venue": race.split(":")[2],
                "race_no": int(race.split(":")[3]),
                "pool": pool,
                "combination": combination,
                "snapshot_utc": snap,
                "fetched_utc": fetched,
                "post_utc": posts.get(race, POST),
                "minutes_to_post": (posts.get(race, POST) - snap).total_seconds() / 60,
                "odds": odds,
                "odds_hi": NAN,
                "investment": 1000.0,
                "status_raw": "",
                "is_final_capture": final,
                "dividend": 8900.0,
                "is_refund": False,
            }
            for race, pool, combination, snap, fetched, odds, final in captures
        ]
    )
    for column in ("snapshot_utc", "fetched_utc", "post_utc"):
        frame[column] = pd.Series(pd.to_datetime(list(frame[column]), utc=True))
    if new is not None:
        name, value = new
        frame[name] = [value(i) for i in range(len(frame))]
    frame = frame.drop(columns=list(drop))
    con = duckdb.connect(":memory:")
    con.register("_frame", frame)
    con.execute('CREATE TABLE "odds_timeseries" AS SELECT * FROM _frame')
    con.unregister("_frame")
    return con


@pytest.fixture(scope="module")
def rules():
    return load_availability("japan")


@pytest.mark.parametrize("new", NEW_COLUMNS, ids=[n[0] for n in NEW_COLUMNS])
def test_the_board_at_t_never_carries_a_column_no_config_classifies(rules, new):
    con = _board_con(new)
    try:
        view = avoidance.view_at(con, RACE, T, rules=rules)
    finally:
        con.close()
    assert new[0] not in view.board.columns, f"{new[0]} reached the board at t"
    for column in ODDS_LABELS:
        assert column not in view.board.columns
    assert not view.board.duplicated(["pool", "combination"]).any()
    on_board = view.board.set_index(["pool", "combination"])["odds"]
    assert {k: float(v) for k, v in on_board.items()} == AT_T
    if "race_id" in view.board.columns:
        assert set(view.board["race_id"]) == {RACE}


def test_every_board_column_is_an_approved_feature_or_an_identifier(rules):
    from horsetrack.countries.japan.feature_boundary import approved_features

    approved = approved_features("odds_timeseries", source="jra", rules=rules)
    assert "odds" in approved  # the price the study reads at t is classified
    con = _board_con()
    try:
        view = avoidance.view_at(con, RACE, T, rules=rules)
    finally:
        con.close()
    columns = set(view.board.columns)
    assert {"pool", "combination", "odds"} <= columns
    assert columns <= approved | IDENTIFIERS, sorted(columns - approved - IDENTIFIERS)


def test_the_report_names_the_results_derived_predictors():
    from horsetrack.countries.japan.feature_boundary import RESULTS_DERIVED

    minutes = int(load_country_yaml("japan", "decision")["primary_min"])
    spec = avoidance.build_spec(avoidance.owner_inputs(), avoidance.bootstrap_settings(), minutes)
    report = avoidance.render_report(Study(counts={}), spec, {"git": "0000000"})
    tagged = [line for line in report.splitlines() if RESULTS_DERIVED in line]
    for table, column in (("runners", "is_scratched"), ("races", "status")):
        assert any(table in line and column in line for line in tagged), (
            f"{table}.{column} is not named as {RESULTS_DERIVED!r}"
        )


# ---------------------------------------------------------------------------------------------
# The board rule's fallback: a capture without fetched_utc is on the board at its snapshot_utc
# ---------------------------------------------------------------------------------------------

US = timedelta(microseconds=1)


def _expected_latest(captures, race: str, t: datetime) -> dict[tuple[str, str], tuple]:
    """From first principles: a capture is available at its fetched_utc, or at its snapshot_utc
    where fetched_utc is null; the board at t holds, per (pool, combination), `race`'s latest
    snapshot among the captures available at t (each value the capture's row)."""
    latest: dict[tuple[str, str], tuple] = {}
    for row in captures:
        r, pool, combination, snap, fetched, _, _ = row
        available = snap if fetched is None else fetched
        if r != race or available > t:
            continue
        key = (pool, combination)
        if key not in latest or snap > latest[key][3]:
            latest[key] = row
    return latest


def _expected_board(captures, race: str, t: datetime) -> dict[tuple[str, str], float]:
    return {key: row[5] for key, row in _expected_latest(captures, race, t).items()}


def _prices(view) -> dict[tuple[str, str], float]:
    on_board = view.board.set_index(["pool", "combination"])["odds"]
    return {k: float(v) for k, v in on_board.items()}


# snapshot_utc of the captures without fetched_utc, and whether they are on the board at T
FALLBACK_SNAPSHOTS = [
    (at(5, 52), True),
    (T - US, True),
    (T, True),  # available at t
    (T + US, False),
    (at(5, 58), False),
]


@pytest.mark.parametrize(
    "snap,on_board",
    FALLBACK_SNAPSHOTS,
    ids=[s.strftime("%H:%M:%S.%f") for s, _ in FALLBACK_SNAPSHOTS],
)
def test_a_capture_without_fetched_utc_is_on_the_board_at_its_snapshot_utc(rules, snap, on_board):
    # a new combination, and a newer snapshot of one the board prices at 21.0 (05:50)
    captures = [
        *CAPTURES,
        (RACE, "WIN", "5", snap, None, 12.5, False),
        (RACE, "TRIO", "2,9,14", snap, None, 30.0, False),
    ]
    expected = _expected_board(captures, RACE, T)
    assert (("WIN", "5") in expected) is on_board  # the fixture is what the case says
    assert (expected[("TRIO", "2,9,14")] == 30.0) is on_board
    con = _board_con(captures=captures)
    try:
        view = avoidance.view_at(con, RACE, T, rules=rules)
    finally:
        con.close()
    assert not view.board.duplicated(["pool", "combination"]).any()
    assert _prices(view) == expected
    if on_board:
        row = view.board.loc[view.board["combination"].eq("5")].iloc[0]
        assert pd.isna(row["fetched_utc"]), "the capture's missing fetched_utc was filled in"
        assert pd.Timestamp(row["snapshot_utc"]) == pd.Timestamp(snap)


@pytest.mark.parametrize("with_fallback", [False, True], ids=["fetched", "a-capture-unfetched"])
def test_view_at_reads_a_board_without_post_utc(rules, with_fallback):
    extra = [(RACE, "WIN", "5", at(5, 51), None, 14.0, False)] if with_fallback else []
    captures = [*CAPTURES, *extra]
    con = _board_con(captures=captures, drop=("post_utc", "minutes_to_post"))
    try:
        view = avoidance.view_at(con, RACE, T, rules=rules)
    finally:
        con.close()
    assert "post_utc" not in view.board.columns
    assert _prices(view) == _expected_board(captures, RACE, T)
    assert _prices(view) == {**AT_T, **({("WIN", "5"): 14.0} if with_fallback else {})}


# compute_study over four synthetic races whose boards at t are read through view_at
STUDY_RACES = (
    "jra:2026-06-06:TOK:01",
    "jra:2026-06-06:TOK:02",
    "jra:2026-06-07:KYO:01",
    "jra:2026-06-07:KYO:02",
)
STUDY_POSTS = {  # each race's post on its own date, UTC
    race: datetime.fromisoformat(f"{race.split(':')[1]}T0{1 + 2 * i}:05:00+00:00")
    for i, race in enumerate(STUDY_RACES)
}
BOARD_COMBOS = {"WIN": ("1", "2"), "TRIO": ("1,2,3", "4,5,6"), "QUINELLA": ("1,2", "3,4")}


def _study_captures(minutes: int, fallback_races) -> list[tuple]:
    """Every race: each combination captured with its fetch time 10 min before t, and a capture
    without fetched_utc after t (never on the board). In each of `fallback_races`, the latest
    capture of a TRIO combination has no fetched_utc and a snapshot before t (on its board at t);
    in every other race, a capture without fetched_utc that a later fetched capture of the same
    combination supersedes (in the view at t, not on the board)."""
    rows = []
    for race in STUDY_RACES:
        t = STUDY_POSTS[race] - timedelta(minutes=minutes)
        snap = t - timedelta(minutes=10)
        for k, (pool, combos) in enumerate(BOARD_COMBOS.items()):
            for j, combination in enumerate(combos):
                fetched = snap + timedelta(seconds=3)
                rows.append((race, pool, combination, snap, fetched, 2.0 + 3 * k + j, False))
        rows.append((race, "WIN", "1", t + timedelta(seconds=30), None, 1.8, False))
        if race in fallback_races:
            rows.append((race, "TRIO", "1,2,3", t - timedelta(minutes=2), None, 40.0, False))
        else:
            early, late = t - timedelta(minutes=8), t - timedelta(minutes=5)
            rows.append((race, "WIN", "2", early, None, 9.0, False))
            rows.append((race, "WIN", "2", late, late + timedelta(seconds=3), 8.0, False))
    return rows


def _study_data(con, rules, minutes: int):
    days = {race: date.fromisoformat(race.split(":")[1]) for race in STUDY_RACES}
    races = pd.DataFrame(
        [(r, days[r], ":".join(r.split(":")[1:3]), "run", None) for r in STUDY_RACES],
        columns=["race_id", "race_date", "meeting_id", "status", "race_class_raw"],
    )
    runners = pd.DataFrame(
        [(r, days[r], h, h, False, 1.5 + 2.0 * h) for r in STUDY_RACES for h in range(1, 9)],
        columns=["race_id", "race_date", "horse_no", "bracket", "is_scratched", "win_odds"],
    )
    paying = [("TRIO", "1,2,3", 2500.0), ("TIERCE", "1,2,3", 9800.0), ("QUINELLA", "1,2", 760.0)]
    paying += [("FORECAST", "1,2", 1490.0), ("BRACKET_QUINELLA", "1,2", 720.0)]
    dividends = pd.DataFrame(
        [(r, pool, c, d, False) for r in STUDY_RACES for pool, c, d in paying],
        columns=["race_id", "pool", "combination", "dividend", "is_refund"],
    )
    selection = f"{avoidance.BOARD_AT_T} (T-{minutes})"

    def view(race: str):
        t = STUDY_POSTS[race] - timedelta(minutes=minutes)
        return avoidance.view_at(con, race, t, rules=rules, selection=selection)

    return avoidance.StudyData(
        races=races,
        runners=runners,
        dividends=dividends,
        logged={pool: list(STUDY_RACES) for pool in BOARD_COMBOS},
        view_at_t=view,
        board_files=[],
    )


@pytest.mark.parametrize(
    "fallback_races",
    [(), (STUDY_RACES[0],), (STUDY_RACES[0], STUDY_RACES[2], STUDY_RACES[3])],
    ids=["none", "one", "three"],
)
def test_the_report_discloses_the_board_fallback_with_its_race_count(rules, fallback_races):
    from horsetrack.countries.japan.feature_boundary import TIMESTAMP_FALLBACK

    minutes = int(load_country_yaml("japan", "decision")["primary_min"])
    boot = {**avoidance.bootstrap_settings(), "n_boot": 200}
    spec = avoidance.build_spec(avoidance.owner_inputs(), boot, minutes)
    captures = _study_captures(minutes, fallback_races)
    # From first principles: the races whose board at t holds a capture without fetched_utc.
    expected = sum(
        any(
            row[4] is None
            for row in _expected_latest(
                captures, race, STUDY_POSTS[race] - timedelta(minutes=minutes)
            ).values()
        )
        for race in STUDY_RACES
    )
    assert expected == len(fallback_races)

    con = _board_con(captures=captures, posts=STUDY_POSTS)
    try:
        study = avoidance.compute_study(_study_data(con, rules, minutes), spec)
    finally:
        con.close()
    report = avoidance.render_report(study, spec, {"git": "0000000"})
    lines = [
        line
        for line in report.splitlines()
        if TIMESTAMP_FALLBACK in line and "odds_timeseries.odds" in line
    ]
    assert lines, f"no line names odds_timeseries.odds as {TIMESTAMP_FALLBACK!r}"
    assert any(str(expected) in re.findall(r"\d+", line) for line in lines), (
        f"{expected} races on the fallback: {lines}"
    )
