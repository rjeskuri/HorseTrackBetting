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

Every frame is synthetic; no date is a Japan holdout day (on or after 2026-11-28).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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


def _board_con(new=None) -> duckdb.DuckDBPyConnection:
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
                "post_utc": POST,
                "minutes_to_post": (POST - snap).total_seconds() / 60,
                "odds": odds,
                "odds_hi": NAN,
                "investment": 1000.0,
                "status_raw": "",
                "is_final_capture": final,
                "dividend": 8900.0,
                "is_refund": False,
            }
            for race, pool, combination, snap, fetched, odds, final in CAPTURES
        ]
    )
    for column in ("snapshot_utc", "fetched_utc", "post_utc"):
        frame[column] = pd.Series(pd.to_datetime(list(frame[column]), utc=True))
    if new is not None:
        name, value = new
        frame[name] = [value(i) for i in range(len(frame))]
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
