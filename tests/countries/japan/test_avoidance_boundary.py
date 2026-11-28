"""The exotic pool-choice study (#81) as #417's representative caller of the model boundary.

Contract pinned here (`horsetrack.countries.japan.studies.avoidance`, migrated in #417; the
boundary is `horsetrack.countries.japan.feature_boundary`, see tests/countries/japan/
test_feature_boundary.py):

- `view_at(con, race_id, at_utc, *, rules)` reads the board at t through the boundary
  (`feature_view`), not the whole as-of projection: a column added to odds_timeseries that no
  config classifies never reaches `RaceView.board`, while the board at t keeps its prices. Every
  board column is an approved feature of odds_timeseries (`approved_features`) or one of the
  board's identifiers (race_id, pool, combination, snapshot_utc, fetched_utc, post_utc).
  `status_raw` is no such feature (main's inventory leaves it unclassified; feature inclusion is
  the owner's, docs/specs/as-of.md, "Adopting the boundary"), so it is not on the board: the
  board-at-t scratch rule (docs/specs/board-at-t.md) reads it outside `feature_view`, as a
  documented filter, into `RaceView.scratched` (a frozenset of horse numbers, empty by default):
  the runners whose latest WIN capture, or latest PLACE capture, available at t (the board's
  as-of rules and its latest-per-(pool, combination) order) has `status_raw` `SCR`.
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
- The study's real board path: `board_view(files, rules, minutes)` reads logger day files
  (`logger.compact.SCHEMA`) and registers the race's rows as Arrow data; through it the board at
  t has the prices its rules make, and `compute_study` scores the board-at-t variants (none
  fails) and counts the fallback races.
- The disclosure follows the board reader's rules: `StudyData.board_rules` (the availability
  rules its `view_at_t` reader uses; `read_development` sets the ones it gives `board_view`)
  decides the fallback `compute_study` counts and `render_report` names, from the study alone.
  Without a fallback in the reader's odds_timeseries rule (a column rule without one, or an
  offset rule) the report discloses no timestamp fallback for odds; with another fallback column
  it names that column and the races whose board at t used it.

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


def _board_con(
    new=None, captures=CAPTURES, *, drop=(), posts=None, status=None
) -> duckdb.DuckDBPyConnection:
    """An in-memory odds_timeseries of `captures` (race -> post from `posts`, default POST),
    without the columns in `drop`; each capture's `status_raw` is `status(snapshot, odds)`, or
    blank without `status`."""
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
                "status_raw": "" if status is None else status(snap, odds),
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


def _status(snap: datetime, odds) -> str:
    """A capture's status text: `SCR` for a scratch (no price), else the page's JST time text."""
    if odds is None:
        return "SCR"
    local = snap + timedelta(hours=9)
    return f"{local.hour}時{local.minute:02d}分現在"


# RACE's WIN 3, priced at 05:40, is scratched at 05:52; WIN 12 is scratched at 05:57, after T
SCRATCHES = [
    (RACE, "WIN", "3", at(5, 52), at(5, 52, 2), None, False),
    (RACE, "WIN", "12", at(5, 57), at(5, 57, 2), None, False),
]


def _expected_scratched(captures, race: str, t: datetime) -> frozenset[int]:
    """From first principles (docs/specs/board-at-t.md): the horses whose latest WIN capture, or
    latest PLACE capture, available at t (`_expected_latest`) has status `SCR` (`_status`)."""
    return frozenset(
        int(combination)
        for (pool, combination), row in _expected_latest(captures, race, t).items()
        if pool in ("WIN", "PLACE") and _status(row[3], row[5]) == "SCR"
    )


@pytest.mark.parametrize("t", [at(5, 51), T, at(6, 0)], ids=["05:51", "05:55", "06:00"])
def test_the_scratches_at_t_are_read_outside_the_board(rules, t):
    """status_raw is no approved feature (feature inclusion is the owner's, docs/specs/as-of.md,
    "Adopting the boundary"), so the board at t does not carry it; the board-at-t scratch rule
    (docs/specs/board-at-t.md) reads it through a documented filter instead: `view.scratched`
    holds the runners whose latest WIN or PLACE capture available at t is `SCR`. WIN 3 is
    scratched from its 05:52 fetch on, WIN 12 from 05:57 (after T); before, a priced capture's
    status is a time text."""
    from horsetrack.countries.japan.feature_boundary import approved_features

    assert "status_raw" not in approved_features("odds_timeseries", source="jra", rules=rules)
    captures = [*CAPTURES, *SCRATCHES]
    con = _board_con(captures=captures, status=_status)
    try:
        view = avoidance.view_at(con, RACE, t, rules=rules)
    finally:
        con.close()
    assert "status_raw" not in view.board.columns, "status_raw is on the board at t"
    for column in ODDS_LABELS:
        assert column not in view.board.columns, f"{column} is on the board at t"
    assert not view.board.duplicated(["pool", "combination"]).any()
    expected = _expected_scratched(captures, RACE, t)
    # the fixture is what the case says: none before 05:52, WIN 3 at T, WIN 12 too by 06:00
    assert expected == {at(5, 51): set(), T: {3}, at(6, 0): {3, 12}}[t]
    assert isinstance(view.scratched, frozenset), type(view.scratched)
    assert all(isinstance(h, int) for h in view.scratched), view.scratched
    assert view.scratched == expected
    # a view built without scratches has none
    assert avoidance.RaceView(RACE, view.board, view.selection).scratched == frozenset()


# Each line varies one input of the rule: (race, pool, combination, snapshot, fetched, odds, final)
SCRATCH_CASES = [
    # PLACE 5 is SCR while WIN 5 is still priced: a PLACE scratch counts
    (RACE, "WIN", "5", at(5, 44), at(5, 44, 2), 9.4, False),
    (RACE, "PLACE", "5", at(5, 46), at(5, 46, 2), None, False),
    # WIN 8 is SCR at 05:42, then priced again at 05:48: its latest capture is no scratch
    (RACE, "WIN", "8", at(5, 42), at(5, 42, 2), None, False),
    (RACE, "WIN", "8", at(5, 48), at(5, 48, 2), 15.0, False),
    # WIN 6's SCR capture is taken at 05:53 but fetched at 05:56:10, after T
    (RACE, "WIN", "6", at(5, 45), at(5, 45, 2), 11.0, False),
    (RACE, "WIN", "6", at(5, 53), at(5, 56, 10), None, False),
    # WIN 11's SCR capture has no fetched_utc: available at its snapshot (the board's fallback)
    (RACE, "WIN", "11", at(5, 41), at(5, 41, 2), 30.0, False),
    (RACE, "WIN", "11", at(5, 52), None, None, False),
    # a TRIO row is SCR: not a WIN or PLACE row, so it scratches none of 1, 4, 7
    (RACE, "TRIO", "1,4,7", at(5, 47), at(5, 47, 2), None, False),
    # the other race's WIN 10 is SCR: not this race's
    (OTHER, "WIN", "10", at(5, 43), at(5, 43, 2), None, False),
]


@pytest.mark.parametrize(
    "t", [at(5, 43), at(5, 47), T, at(5, 58)], ids=["05:43", "05:47", "05:55", "05:58"]
)
def test_the_scratched_set_follows_the_latest_win_or_place_capture_at_t(rules, t):
    """`view.scratched` follows the board's as-of rules and its latest-per-(pool, combination)
    order: a later priced capture lifts an earlier SCR, a capture fetched after t is not there
    yet, one without fetched_utc is there at its snapshot, a PLACE SCR counts as a WIN one does,
    and a row of another pool or another race scratches nobody."""
    captures = [*CAPTURES, *SCRATCHES, *SCRATCH_CASES]
    con = _board_con(captures=captures, status=_status)
    try:
        view = avoidance.view_at(con, RACE, t, rules=rules)
    finally:
        con.close()
    expected = _expected_scratched(captures, RACE, t)
    # the fixture is what the lines say
    case = {at(5, 43): {8}, at(5, 47): {5, 8}, T: {3, 5, 11}, at(5, 58): {3, 5, 6, 11, 12}}
    assert expected == case[t]
    assert view.scratched == expected
    assert "status_raw" not in view.board.columns


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


# ---------------------------------------------------------------------------------------------
# The study's real board path: day files read by `board_view`, the rows registered as Arrow data
# ---------------------------------------------------------------------------------------------

# The unfetched TRIO capture of each fallback race (snapshot t - 2 min) carries `logged_utc` =
# snapshot + this delay: on the board at t under a `logged_utc` fallback for the first and last
# race only, while the shipped `snapshot_utc` fallback puts all three on it.
LOGGED_DELAY = {
    STUDY_RACES[0]: timedelta(minutes=1),
    STUDY_RACES[2]: timedelta(minutes=3),
    STUDY_RACES[3]: timedelta(seconds=30),
}
FALLBACK_RACES = tuple(LOGGED_DELAY)


def _logged_utc(row) -> datetime:
    race, pool, _, snap, fetched, _, _ = row
    if fetched is None and pool == "TRIO":
        return snap + LOGGED_DELAY[race]
    return snap + timedelta(seconds=3)


def _write_days(folder, captures, *, logged: bool = False) -> dict:
    """The captures as logger day files (`logger.compact.SCHEMA`, one Parquet per race date,
    each race's post from STUDY_POSTS), plus a `logged_utc` column when `logged`; race -> file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from horsetrack.countries.japan.logger.compact import SCHEMA

    schema = (
        SCHEMA.append(pa.field("logged_utc", pa.timestamp("us", tz="UTC"))) if logged else SCHEMA
    )
    by_day: dict[str, list[dict]] = {}
    for row in captures:
        race, pool, combination, snap, fetched, odds, final = row
        post = STUDY_POSTS[race]
        record = {
            "race_id": race,
            "venue": race.split(":")[2],
            "race_no": int(race.split(":")[3]),
            "pool": pool,
            "combination": combination,
            "snapshot_utc": snap,
            "fetched_utc": fetched,
            "post_utc": post,
            "minutes_to_post": (post - (fetched or snap)).total_seconds() / 60,
            "odds": odds,
            "odds_hi": None,
            "investment": 1000.0,
            "status_raw": "",
            "is_final_capture": final,
            "dividend": None,
            "is_refund": None,
        }
        if logged:
            record["logged_utc"] = _logged_utc(row)
        by_day.setdefault(race.split(":")[1], []).append(record)
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for day, records in by_day.items():
        path = folder / f"{day}.parquet"
        pq.write_table(pa.Table.from_pylist(records, schema=schema), path)
        files.update({r["race_id"]: path for r in records})
    return files


def _latest_by(captures, race: str, t: datetime, available) -> dict[tuple[str, str], tuple]:
    """From first principles: per (pool, combination), `race`'s latest snapshot among the captures
    with `available(row)` (None: never) at or before t."""
    latest: dict[tuple[str, str], tuple] = {}
    for row in captures:
        when = available(row)
        if row[0] != race or when is None or when > t:
            continue
        key = (row[1], row[2])
        if key not in latest or row[3] > latest[key][3]:
            latest[key] = row
    return latest


def _shipped(row):  # fetched_utc, or snapshot_utc where it is null
    return row[3] if row[4] is None else row[4]


def _study_setup(minutes: int):
    boot = {**avoidance.bootstrap_settings(), "n_boot": 200}
    return avoidance.build_spec(avoidance.owner_inputs(), boot, minutes)


def _real_path_data(files, rules, minutes: int, *, board_rules=None):
    """The study's data with its board at t read the way `read_development` reads it:
    `board_view` over the day files under `rules`; with `board_rules`, also
    `StudyData.board_rules`, the rules that reader uses."""
    import dataclasses

    data = dataclasses.replace(
        _study_data(None, rules, minutes),
        view_at_t=avoidance.board_view(files, rules, minutes),
        board_files=sorted(set(files.values())),
    )
    return data if board_rules is None else dataclasses.replace(data, board_rules=board_rules)


def _fallback_lines(report: str) -> list[str]:
    from horsetrack.countries.japan.feature_boundary import TIMESTAMP_FALLBACK

    return [
        line
        for line in report.splitlines()
        if TIMESTAMP_FALLBACK in line and "odds_timeseries.odds" in line
    ]


def test_board_view_reads_the_board_at_t_from_registered_day_files(rules, tmp_path):
    minutes = int(load_country_yaml("japan", "decision")["primary_min"])
    captures = _study_captures(minutes, FALLBACK_RACES)
    files = _write_days(tmp_path / "odds_timeseries", captures)
    view = avoidance.board_view(files, rules, minutes)
    for race in STUDY_RACES:
        t = STUDY_POSTS[race] - timedelta(minutes=minutes)
        seen = view(race)
        assert seen.selection == f"{avoidance.BOARD_AT_T} (T-{minutes})"
        expected = {k: row[5] for k, row in _latest_by(captures, race, t, _shipped).items()}
        assert _prices(seen) == expected, race
        # the unfetched TRIO capture is on a fallback race's board, at 40.0
        assert (expected[("TRIO", "1,2,3")] == 40.0) is (race in FALLBACK_RACES)


def test_the_study_scores_its_board_at_t_variants_on_the_registered_board(rules, tmp_path):
    minutes = int(load_country_yaml("japan", "decision")["primary_min"])
    spec = _study_setup(minutes)
    captures = _study_captures(minutes, FALLBACK_RACES)
    files = _write_days(tmp_path / "odds_timeseries", captures)
    study = avoidance.compute_study(_real_path_data(files, rules, minutes), spec)

    at_t = f"{avoidance.BOARD_AT_T} (T-{minutes})"
    board_variants = [v for v in study.variants if v.components.get("selection") == at_t]
    assert board_variants, "no board-at-t variant ran"
    failed = {
        v.name: v.error
        for v in board_variants
        if v.error and not v.error.startswith(("no paired race", "data pending"))
    }
    assert not failed, f"board-at-t variants failed on the registered board: {failed}"
    rows = [r for r in study.tables.get("comparisons", []) if r["selection"] == at_t]
    assert rows and all(r["status"] != "failed" for r in rows), rows
    assert any(r["status"] == "scored" for r in rows), rows

    expected = sum(
        any(
            row[4] is None
            for row in _latest_by(
                captures, race, STUDY_POSTS[race] - timedelta(minutes=minutes), _shipped
            ).values()
        )
        for race in STUDY_RACES
    )
    assert expected == len(FALLBACK_RACES)
    lines = _fallback_lines(avoidance.render_report(study, spec, {"git": "0000000"}))
    assert lines and any(str(expected) in re.findall(r"\d+", line) for line in lines), lines


# ---------------------------------------------------------------------------------------------
# The disclosure follows the board reader's rules, not the shipped ones
# ---------------------------------------------------------------------------------------------


def _board_rules(rules, case: str):
    """The shipped rules with odds_timeseries replaced: no fallback, an offset of two minutes
    from snapshot_utc, a `logged_utc` fallback, odds gated until snapshot_utc + 1 min (a
    publication proxy), or a primary timestamp the board view drops (`<column>-primary`, with
    the shipped `snapshot_utc` fallback)."""
    import dataclasses

    from horsetrack.core.store.asof import Availability, Gate

    board = rules["odds_timeseries"]
    if case.endswith("-primary"):
        board = dataclasses.replace(board, column=case.removesuffix("-primary"))
    elif case == "gated-odds":
        gate = Gate(("odds",), anchor="snapshot_utc", offset=timedelta(minutes=1))
        board = dataclasses.replace(board, gates=(gate,))
    elif case == "no-fallback":
        board = dataclasses.replace(board, fallback=None)
    elif case == "offset-2min":
        board = Availability(
            "odds_timeseries",
            "offset",
            anchor="snapshot_utc",
            offset=timedelta(minutes=2),
            label_columns=board.label_columns,
        )
    else:
        board = dataclasses.replace(board, fallback="logged_utc")
    return {**rules, "odds_timeseries": board}


AVAILABLE = {  # case -> when a capture is available under it, from first principles
    "no-fallback": lambda row: row[4],
    "offset-2min": lambda row: row[3] + timedelta(minutes=2),
    "logged-utc-fallback": lambda row: _logged_utc(row) if row[4] is None else row[4],
}
FALLBACK_COUNT = {"no-fallback": None, "offset-2min": None, "logged-utc-fallback": 2}


@pytest.mark.parametrize("case", list(AVAILABLE))
def test_the_fallback_disclosure_follows_the_board_readers_rules(rules, tmp_path, case):
    minutes = int(load_country_yaml("japan", "decision")["primary_min"])
    spec = _study_setup(minutes)
    captures = _study_captures(minutes, FALLBACK_RACES)
    files = _write_days(tmp_path / "odds_timeseries", captures, logged=True)
    board_rules = _board_rules(rules, case)

    # the reader's boards are the ones its rules make
    view = avoidance.board_view(files, board_rules, minutes)
    for race in STUDY_RACES:
        t = STUDY_POSTS[race] - timedelta(minutes=minutes)
        latest = _latest_by(captures, race, t, AVAILABLE[case])
        assert _prices(view(race)) == {k: row[5] for k, row in latest.items()}, (case, race)

    # From first principles: the races whose board at t holds a capture on the fallback.
    on_fallback = sum(
        any(
            row[4] is None
            for row in _latest_by(
                captures, race, STUDY_POSTS[race] - timedelta(minutes=minutes), AVAILABLE[case]
            ).values()
        )
        for race in STUDY_RACES
    )
    # the fixture is what the case says: without a fallback no unfetched capture is on a board;
    # the offset rule puts three on boards, none by a fallback; `logged_utc` puts two, not three
    assert on_fallback == {"no-fallback": 0, "offset-2min": 3, "logged-utc-fallback": 2}[case]
    expected = FALLBACK_COUNT[case]  # the races the report discloses, None: no disclosure

    study = avoidance.compute_study(
        _real_path_data(files, board_rules, minutes, board_rules=board_rules), spec
    )
    lines = _fallback_lines(avoidance.render_report(study, spec, {"git": "0000000"}))
    if expected is None:
        # no column rule with a fallback: the report discloses no timestamp fallback for odds
        assert not lines, f"{case}: the report discloses the shipped fallback: {lines}"
    else:
        assert lines, f"{case}: the reader's fallback is not disclosed"
        assert all("logged_utc" in line for line in lines), lines
        assert any(str(expected) in re.findall(r"\d+", line) for line in lines), (
            f"{expected} races on the logged_utc fallback: {lines}"
        )


def _proxy_lines(report: str) -> list[str]:
    from horsetrack.countries.japan.feature_boundary import PUBLICATION_PROXY

    return [
        line
        for line in report.splitlines()
        if PUBLICATION_PROXY in line and "`odds_timeseries.odds`" in line
    ]


@pytest.mark.parametrize("case", ["gated-odds", "shipped"])
def test_the_board_features_approximation_tags_follow_the_board_readers_rules(
    rules, tmp_path, case
):
    """The odds line's tags come from the rules the board was read under: a reader whose rule
    gates odds behind an anchor + offset reports odds as a publication proxy; the shipped rule
    (no gate) does not."""
    minutes = int(load_country_yaml("japan", "decision")["primary_min"])
    spec = _study_setup(minutes)
    captures = _study_captures(minutes, FALLBACK_RACES)
    files = _write_days(tmp_path / "odds_timeseries", captures)
    board_rules = rules if case == "shipped" else _board_rules(rules, case)
    assert ("odds" in board_rules["odds_timeseries"].gated) is (case == "gated-odds")

    study = avoidance.compute_study(
        _real_path_data(files, board_rules, minutes, board_rules=board_rules), spec
    )
    report = avoidance.render_report(study, spec, {"git": "0000000"})
    lines = _proxy_lines(report)
    if case == "gated-odds":
        assert lines, "the reader's gate on odds is not reported as a publication proxy"
    else:
        assert not lines, f"the shipped rule does not gate odds: {lines}"
    # the shipped snapshot_utc fallback is still disclosed under both
    assert _fallback_lines(report), case


@pytest.mark.parametrize("primary", ["captured_utc", "logged_utc"])
def test_a_board_rule_whose_primary_timestamp_the_view_drops_is_refused(rules, tmp_path, primary):
    """Fail closed: the board view keeps only BOARD_KEYS and the features, so a fallback on a
    primary column outside them could never be counted; compute_study refuses the rules."""
    assert primary not in IDENTIFIERS
    minutes = int(load_country_yaml("japan", "decision")["primary_min"])
    spec = _study_setup(minutes)
    captures = _study_captures(minutes, FALLBACK_RACES)
    files = _write_days(tmp_path / "odds_timeseries", captures, logged=True)
    board_rules = _board_rules(rules, f"{primary}-primary")
    rule = board_rules["odds_timeseries"]
    assert (rule.column, rule.fallback) == (primary, "snapshot_utc")
    # the reader reads its boards under the shipped rules where its own cannot be read
    reader = board_rules if primary == "logged_utc" else rules
    data = _real_path_data(files, reader, minutes, board_rules=board_rules)
    with pytest.raises(ValueError, match=primary):
        avoidance.compute_study(data, spec)
