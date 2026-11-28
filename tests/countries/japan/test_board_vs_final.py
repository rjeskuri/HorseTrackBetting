"""Board-fitted against final-fitted terms on the JRDB board (ticket #103, epic #73; 08 §4 E2, E45,
R13).

Contract pinned here (`horsetrack.countries.japan.studies.board_vs_final`). Where the ticket left
a choice open, the test-author's choice is marked (*).

- `jrdb_board(con, race_ids, *, rules) -> pd.DataFrame`: JRDB's fixed-time board, the TYB
  family's `tyb_win_odds` taken at `tyb_odds_time` ("HHMM" on the JST clock of `race_date`), from
  the store table `jrdb_pre_race` in the DuckDB connection `con`, read through `read_asof` at the
  JRDB release time. (*) `rules` is the release-time rule as data: the `board` mapping of
  `countries/japan/config/board_vs_final.yaml` (`load_country_yaml("japan", "board_vs_final")
  ["board"]`) with (*) one `release_lag_min` (a number of minutes, >= 0) set from the config's
  `release_lags_min` list (#602: each configured lag is a counted variant of the study), which
  runs from JRDB's odds time
  to its release. A row is released at race_date, tyb_odds_time (Asia/Tokyo) + release_lag_min.
  The frame has `race_id`, `horse_no`, `tyb_win_odds` and (*) `release_utc`, that instant
  (timezone-aware; the snapshot a board-fitted term is reported with, R13), at most one row per
  runner and only for the requested races. A row with no release time (a missing or malformed
  odds time) or no board odds has no board: it is absent, or its `tyb_win_odds` is null. A board
  value is never filled from another price: the store's final `runners.win_odds`, or KYI's
  day-before `kyi_base_odds`. The owner's `jrdb_pre_race` rule in availability.yaml (tyb_ columns
  are labels) is not edited; `test_availability.py` pins it.
  `jrdb_board(..., decision_utc=offs)`: `offs` is a Series of timezone-aware instants indexed by
  race_id, each race's own decision time (its off; the run passes `races.post_utc`). A runner's
  board counts only if released at or before its own race's off (release_utc <= off, the
  boundary included), never by another race's later off on the same day; a requested race with
  no decision time (missing from `offs`, or NaT) has no board. Without `decision_utc` the board
  is read as of now.
- `cross_apply(frame, *, fit_price, score_price, fit, score) -> pd.DataFrame`: `fit_price` and
  `score_price` name the two price columns of `frame` (one row per runner, with `race_id`). Only
  the races where both prices are present on every row are used, by every fit and every score
  (matching seasons: a race, and so a season, priced on one side only is in neither fit, so a
  cell's difference comes from the price and not from the races). `fit(rows, price) -> model` is
  called once per price, and `score(model, rows, price) -> pd.Series` (per-race ΔLL indexed by
  race_id) once per fit and price; both get every column of the matched rows. The table's index
  is the price fitted on ({fit_price, score_price}); (*) its columns are a MultiIndex (price
  scored on, stat), stat in ("mean", "lo", "hi"): the plain mean over races of the score's values
  and the bounds of its CI (lo <= mean <= hi; lo == mean == hi when every race scores the same).
- (*) `tables` below is a mapping (hypothesis_id, pool code) -> a `cross_apply` table:
  ("E2", "WIN") for #87's meta-model terms (beta, gamma), ("E45", "PLACE" | "QUINELLA_PLACE" |
  "TRIO") for #99's ordering coefficients.
  - (*) `leaderboard_rows(tables) -> list[LeaderboardRow]`: one member row per fit x score cell,
    in its table's pool, each withheld (D21), option ids unique.
  - (*) `run_records(tables, *, git_sha, data_builds) -> list[RunRecord]`: records of ticket 103
    whose sources carry "jrdb", one hypothesis id per table kind present, every cell's mean, lo
    and hi a metric (value, lo, hi); they append to a registry, whose committed summary
    (`write_summary`) then shows none of those figures.
  - (*) `write_outputs(tables, *, out_dir) -> the paths written` (an iterable or a mapping of
    them): the report and the private leaderboard under `out_dir`, through `write_report` with
    JRDB among the sources, so a directory inside the repository raises LicensedOutputInRepo
    before any file is created.

All data is synthetic: frames shaped like the store's JRDB table and the study's frame, no JRDB
file or value. Expected values come from the tests' own fit and score callables and from the clock
arithmetic below, never from the code under test.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import itertools
import json
import math
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar

from horsetrack.core import env
from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.leaderboard import LeaderboardRow
from horsetrack.core.eval.registry import WITHHELD_LABEL, Registry, RunRecord, write_summary
from horsetrack.core.eval.reports import LicensedOutputInRepo
from horsetrack.countries.japan.models.exotic import PLACE, TRIO, WIDE, WIN
from horsetrack.core.eval.folds import load_country_scheme
from horsetrack.core.eval.registry import FoldMetric
from horsetrack.countries.japan.studies.board_vs_final import (
    FAMILY_ID,
    bootstrap_settings,
    build_spec,
    cross_apply,
    jrdb_board,
    jrdb_boards,
    leaderboard_rows,
    release_lags,
    run_records,
    write_outputs,
)

STATS = ("mean", "lo", "hi")
FINAL, BOARD = "q_final", "q_board"
BOARD_CONFIG = load_country_yaml("japan", "board_vs_final")["board"]
LAGS = tuple(BOARD_CONFIG.get("release_lags_min", (0, 10)))  # #602: each lag a counted variant
TABLES = [("E2", WIN), ("E45", PLACE), ("E45", WIDE), ("E45", TRIO)]
ALL_KEYS = [(h, pool, lag) for lag in LAGS for h, pool in TABLES]  # (hypothesis, pool, lag_min)
JST = ZoneInfo("Asia/Tokyo")


# ---------------------------------------------------------------------------------------------
# Synthetic races and a one-parameter model to fit on either price
# ---------------------------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def synthetic_races(
    seed: int,
    *,
    final: str = FINAL,
    board: str = BOARD,
    seasons: tuple[int, ...] = (2024, 2025),
    per_season: int = 24,
    stale: float = 0.5,
) -> pd.DataFrame:
    """One row per runner: race_id, race_date, meeting_id, horse_no, won and two prices that sum
    to 1 within each race: `final`, close to the chances that draw the winner, and `board`, a
    staler copy (shrunk toward equal chances and noisier by `stale`). Six races a meeting day."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for k in range(per_season):
            day = date(season, 4, 6) + timedelta(days=7 * (k // 6))
            meeting = f"jra:{day.isoformat()}:TOK"
            n = int(rng.integers(6, 13))
            strength = rng.normal(0.0, 1.0, n)
            winner = int(rng.choice(n, p=_softmax(strength)))
            q_final = _softmax(strength + rng.normal(0.0, 0.15, n))
            q_board = _softmax((1.0 - 0.4 * stale) * strength + rng.normal(0.0, stale, n))
            for i in range(n):
                rows.append(
                    {
                        "race_id": f"{meeting}:{k % 6 + 1:02d}",
                        "race_date": day,
                        "meeting_id": meeting,
                        "horse_no": i + 1,
                        "won": int(i == winner),
                        final: float(q_final[i]),
                        board: float(q_board[i]),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["race_date"] = pd.to_datetime(frame["race_date"])
    return frame


def winner_log_prob(frame: pd.DataFrame, price: str, beta: float) -> pd.Series:
    """Per race (indexed by race_id), ln P(winner) when p is proportional to price ** beta within
    the race; beta = 1 is the price itself. One winner a race."""
    codes, races = pd.factorize(frame["race_id"].to_numpy(dtype=object))
    u = beta * np.log(frame[price].to_numpy(dtype=float))
    top = np.full(len(races), -np.inf)
    np.maximum.at(top, codes, u)
    lse = top + np.log(np.bincount(codes, weights=np.exp(u - top[codes]), minlength=len(races)))
    won = frame["won"].to_numpy(dtype=float)
    lp = np.bincount(codes, weights=won * u, minlength=len(races)) - lse
    return pd.Series(lp, index=pd.Index(races, name="race_id"), dtype=float)


def power_fit(frame: pd.DataFrame, price: str) -> dict:
    """The maximum-likelihood power beta of p ~ price ** beta on these rows."""
    found = minimize_scalar(
        lambda b: -winner_log_prob(frame, price, b).sum(),
        bounds=(-1.0, 4.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return {"beta": float(found.x)}


def power_score(model: dict, frame: pd.DataFrame, price: str) -> pd.Series:
    """Per race, the fitted power's log-likelihood gain over the scored price itself (ΔLL)."""
    return winner_log_prob(frame, price, model["beta"]) - winner_log_prob(frame, price, 1.0)


def level_model(first: str, second: str):
    """A fit whose model is a level per price, and a score that is the same on every race:
    0.1 x level, + 0.01 when scored on `second`. Each cell has its own value."""
    levels = {first: 1.0, second: 2.0}

    def fit(frame: pd.DataFrame, price: str) -> dict:
        return {"level": levels[price]}

    def score(model: dict, frame: pd.DataFrame, price: str) -> pd.Series:
        races = pd.unique(frame["race_id"].to_numpy(dtype=object))
        value = 0.1 * model["level"] + (0.01 if price == second else 0.0)
        return pd.Series(value, index=pd.Index(races, name="race_id"), dtype=float)

    return fit, score


def cell(table: pd.DataFrame, fit_on: str, scored_on: str, stat: str) -> float:
    return float(table.loc[fit_on, (scored_on, stat)])


def assert_cross_table(table: pd.DataFrame, prices: tuple[str, str]) -> None:
    """Rows: the price fitted on; columns: (the price scored on, mean | lo | hi)."""
    assert isinstance(table, pd.DataFrame)
    assert table.shape == (2, 2 * len(STATS)), table
    assert set(table.index) == set(prices)
    assert isinstance(table.columns, pd.MultiIndex)
    assert set(table.columns) == {(s, stat) for s in prices for stat in STATS}


# ---------------------------------------------------------------------------------------------
# cross_apply
# ---------------------------------------------------------------------------------------------

CROSS_CASES = [
    # seed, the final price's column, the board's column, fit_price, the board's staleness
    pytest.param(3, "q_final", "q_board", "q_final", 0.6, id="fit-price-final"),
    pytest.param(11, "q_final", "q_board", "q_board", 0.3, id="fit-price-board"),
    pytest.param(29, "p_last", "p_t10", "p_t10", 0.9, id="other-names"),
]


@pytest.mark.parametrize(("seed", "final", "board", "fit_price", "stale"), CROSS_CASES)
def test_cross_application_scores_each_fit_on_the_other_price(seed, final, board, fit_price, stale):
    frame = synthetic_races(seed, final=final, board=board, stale=stale)
    score_price = board if fit_price == final else final
    prices = (fit_price, score_price)
    want = {
        (f, s): float(power_score(power_fit(frame, f), frame, s).mean())
        for f in prices
        for s in prices
    }
    # the fixture tells every cell apart: a copied diagonal, a transposed or a constant table fails
    assert min(abs(a - b) for a, b in itertools.combinations(want.values(), 2)) > 1e-4

    table = cross_apply(
        frame, fit_price=fit_price, score_price=score_price, fit=power_fit, score=power_score
    )
    assert_cross_table(table, prices)
    for (f, s), mean in want.items():
        assert cell(table, f, s, "mean") == pytest.approx(mean, rel=1e-6, abs=1e-9), (f, s)
        lo, hi = cell(table, f, s, "lo"), cell(table, f, s, "hi")
        assert lo <= cell(table, f, s, "mean") <= hi, (f, s, lo, hi)
        assert lo < hi, f"{f} on {s}: the races score differently, so the CI has width"

    # A score that is the same on every race: each cell is its own fit's level on its own price,
    # and its CI is that value.
    fit, score = level_model(fit_price, score_price)
    table = cross_apply(frame, fit_price=fit_price, score_price=score_price, fit=fit, score=score)
    assert_cross_table(table, prices)
    levels = {
        (fit_price, fit_price): 0.1,
        (fit_price, score_price): 0.11,
        (score_price, fit_price): 0.2,
        (score_price, score_price): 0.21,
    }
    for (f, s), value in levels.items():
        for stat in STATS:
            assert cell(table, f, s, stat) == pytest.approx(value, abs=1e-12), (f, s, stat)


def test_fits_use_matching_seasons():
    """2021 has no board (before the JRDB board) and 2024 no final price; in 2022 one race lacks
    the board of one runner and another the final price of one. Every fit and every score sees
    the same races, those priced both ways, so the same seasons."""
    frame = synthetic_races(41, seasons=(2021, 2022, 2023, 2024), per_season=12)
    season = frame["race_date"].dt.year
    frame.loc[season == 2021, BOARD] = np.nan
    frame.loc[season == 2024, FINAL] = np.nan
    races_2022 = sorted(frame.loc[season == 2022, "race_id"].unique())
    no_board, no_final = races_2022[2], races_2022[7]
    frame.loc[frame.index[frame["race_id"] == no_board][1], BOARD] = np.nan
    frame.loc[frame.index[frame["race_id"] == no_final][0], FINAL] = np.nan
    matched = set(frame.loc[season.isin([2022, 2023]), "race_id"]) - {no_board, no_final}
    assert len(matched) == 22

    seen: list[tuple[str, str, frozenset, frozenset]] = []

    def note(kind: str, rows: pd.DataFrame, price: str) -> None:
        years = frozenset(int(y) for y in pd.to_datetime(rows["race_date"]).dt.year)
        seen.append((kind, price, frozenset(str(r) for r in rows["race_id"]), years))

    def fit(rows: pd.DataFrame, price: str) -> dict:
        note("fit", rows, price)
        return power_fit(rows, price)

    def score(model: dict, rows: pd.DataFrame, price: str) -> pd.Series:
        note("score", rows, price)
        return power_score(model, rows, price)

    table = cross_apply(frame, fit_price=FINAL, score_price=BOARD, fit=fit, score=score)
    assert_cross_table(table, (FINAL, BOARD))
    assert {price for kind, price, _, _ in seen if kind == "fit"} == {FINAL, BOARD}
    assert {price for kind, price, _, _ in seen if kind == "score"} == {FINAL, BOARD}
    for kind, price, races, years in seen:
        assert years == {2022, 2023}, f"{kind} on {price} saw seasons {sorted(years)}"
        assert races == matched, (
            f"{kind} on {price}: extra {sorted(races - matched)}, missing {sorted(matched - races)}"
        )

    rows = frame[frame["race_id"].isin(matched)]
    for f in (FINAL, BOARD):
        model = power_fit(rows, f)
        for s in (FINAL, BOARD):
            want = float(power_score(model, rows, s).mean())
            assert cell(table, f, s, "mean") == pytest.approx(want, rel=1e-6, abs=1e-9), (f, s)


# ---------------------------------------------------------------------------------------------
# jrdb_board
# ---------------------------------------------------------------------------------------------

BOARD_RACES = [
    # race_id, race_date, JRDB's odds time (HHMM on the JST clock), runners
    ("jra:2025-06-01:TOK:01", date(2025, 6, 1), "0950", 5),
    ("jra:2025-06-01:TOK:11", date(2025, 6, 1), "1532", 6),
    ("jra:2025-06-08:KYO:01", date(2025, 6, 8), "0850", 4),  # before 09:00 JST: the UTC day before
    ("jra:2025-06-08:KYO:05", date(2025, 6, 8), None, 5),  # no odds time: never released
    ("jra:2025-06-08:KYO:06", date(2025, 6, 8), "1075", 5),  # not a clock time: never released
    ("jra:2025-06-08:KYO:12", date(2025, 6, 8), "1601", 5),  # in the table, not requested
]
NOT_REQUESTED = "jra:2025-06-08:KYO:12"
UNKNOWN = "jra:2025-06-08:KYO:99"  # requested, in no table
NO_ODDS = ("jra:2025-06-01:TOK:11", 3)  # the board has no odds for this runner
RELEASED = {"0950", "1532", "0850", "1601"}


def put(con, name: str, frame: pd.DataFrame, casts: dict[str, str]) -> None:
    """A real DuckDB table `name` from a pandas frame, the named columns cast to a DuckDB type."""
    con.register("_frame", frame)
    replaced = ", ".join(f'CAST("{c}" AS {kind}) AS "{c}"' for c, kind in casts.items())
    con.execute(f'CREATE TABLE "{name}" AS SELECT * REPLACE ({replaced}) FROM _frame')
    con.unregister("_frame")


def put_jrdb_store(
    con, seed: int = 5, races=BOARD_RACES, no_odds=NO_ODDS, fetched: Mapping | None = None
) -> pd.DataFrame:
    """`jrdb_pre_race` as the store holds it (the store's key columns, then KYI's day-before board
    and TYB's day-of columns, then `tyb_fetched_utc`: a UTC timestamp, null unless `fetched` maps
    the race_id to an instant, ticket #603) and `runners` with the final Win odds, every price
    different, for `races` (race_id, race_date, odds time, runners), the `no_odds` runner without
    board odds. Returns the rows, one per runner, with the final odds beside them."""
    fetched = fetched or {}
    rng = np.random.default_rng(seed)
    rows = []
    for race_id, day, odds_time, n in races:
        venue, race_no = race_id.split(":")[2:]
        for horse in range(1, n + 1):
            board = round(float(rng.uniform(1.5, 60.0)), 1)
            rows.append(
                {
                    "jurisdiction": "jra",
                    "race_id": race_id,
                    "race_date": day,
                    "horse_no": horse,
                    "horse_id": f"horse-{venue}-{race_no}-{horse}",
                    "race_key": f"{venue}{day:%y}{race_no}",
                    "jrdb_horse_id": f"jrdb-{venue}-{race_no}-{horse}",
                    "matched_by": "id",
                    "kyi_base_odds": round(board * 1.7 + 0.3, 1),
                    "tyb_odds_time": "0000" if odds_time is None else odds_time,
                    "tyb_post_time": "1700",
                    "tyb_win_odds": board,
                    "tyb_place_odds": round(1.0 + board / 4.0, 1),
                    "tyb_weight": float(rng.integers(420, 540)),
                    "tyb_fetched_utc": fetched.get(race_id),
                    "final_win_odds": round(board * 0.8 + 0.2, 1),
                }
            )
    frame = pd.DataFrame(rows)
    frame["tyb_fetched_utc"] = pd.to_datetime(frame["tyb_fetched_utc"], utc=True)
    pre_race = frame.drop(columns=["final_win_odds"])
    casts = {
        "race_date": "DATE",
        "horse_no": "BIGINT",
        "tyb_odds_time": "VARCHAR",
        "tyb_fetched_utc": "TIMESTAMPTZ",
    }
    put(con, "jrdb_pre_race", pre_race, casts)
    for race_id, _, odds_time, _ in races:
        if odds_time is None:
            con.execute(
                "UPDATE jrdb_pre_race SET tyb_odds_time = NULL WHERE race_id = ?", [race_id]
            )
    con.execute(
        "UPDATE jrdb_pre_race SET tyb_win_odds = NULL WHERE race_id = ? AND horse_no = ?",
        list(no_odds),
    )
    runners = frame[["race_id", "race_date", "horse_no", "final_win_odds"]].rename(
        columns={"final_win_odds": "win_odds"}
    )
    put(con, "runners", runners, {"race_date": "DATE", "horse_no": "BIGINT"})
    return frame


def released_at(day: date, hhmm: str, lag_min: float) -> pd.Timestamp:
    """race_date at the HHMM wall-clock time in Tokyo, as a UTC instant, + the release lag."""
    clock = datetime.combine(day, time(int(hhmm[:2]), int(hhmm[2:])), tzinfo=JST)
    return pd.Timestamp(clock.astimezone(timezone.utc) + timedelta(minutes=lag_min))


@pytest.mark.parametrize("lag", [None, 0, 7, 45], ids=["longest-configured", "0", "7", "45"])
def test_the_board_is_read_as_of_the_jrdb_release_time(lag):
    minutes = max(LAGS) if lag is None else lag
    assert isinstance(minutes, int | float) and not isinstance(minutes, bool) and minutes >= 0
    rules = {"tz": BOARD_CONFIG["tz"], "release_lag_min": minutes}

    con = duckdb.connect(":memory:")
    try:
        store = put_jrdb_store(con)
        requested = [r[0] for r in BOARD_RACES if r[0] != NOT_REQUESTED] + [UNKNOWN]
        got = jrdb_board(con, requested, rules=rules)
    finally:
        con.close()

    assert {"race_id", "horse_no", "tyb_win_odds", "release_utc"} <= set(got.columns)
    keys = [(str(r), int(h)) for r, h in zip(got["race_id"], got["horse_no"], strict=True)]
    assert len(keys) == len(set(keys)), "one row per runner at most"
    assert {r for r, _ in keys} <= set(requested) - {UNKNOWN}, "only the requested races"
    at = dict(zip(keys, range(len(got)), strict=True))

    expected = {}
    for row in store.itertuples(index=False):
        key = (row.race_id, int(row.horse_no))
        if row.race_id != NOT_REQUESTED and row.tyb_odds_time in RELEASED and key != NO_ODDS:
            expected[key] = (
                row.tyb_win_odds,
                released_at(row.race_date, row.tyb_odds_time, minutes),
            )
    assert len(expected) == 5 + 5 + 4  # TOK:01, TOK:11 but one runner, KYO:01

    for key, (board, when) in expected.items():
        assert key in at, f"{key}: its board was released, so it is read"
        row = got.iloc[at[key]]
        assert float(row["tyb_win_odds"]) == pytest.approx(board), f"{key}: the board's own odds"
        stamp = pd.Timestamp(row["release_utc"])
        assert stamp.tzinfo is not None, f"{key}: release_utc must be timezone-aware"
        assert stamp.tz_convert("UTC") == when, f"{key}: released {stamp}, want {when}"

    final = {(r.race_id, int(r.horse_no)): r.final_win_odds for r in store.itertuples(index=False)}
    for key in set(at) - set(expected):
        value = got.iloc[at[key]]["tyb_win_odds"]
        assert pd.isna(value), (
            f"{key} has no released board, yet reads {value} (final odds {final[key]})"
        )


# Each race's own off. The last field is the off: minutes after the race's release (an int, so the
# race keeps that relation to its board whatever the lag), a JST clock time that does not move
# with the lag (a str), "missing" (not in decision_utc) or "nat" (a NaT decision time).
OFF_RACES = [
    ("jra:2025-06-01:TOK:01", date(2025, 6, 1), "0950", 4, -1),  # released a minute after its off
    ("jra:2025-06-01:TOK:02", date(2025, 6, 1), "1025", 5, 0),  # released exactly at its off
    ("jra:2025-06-01:TOK:03", date(2025, 6, 1), "1105", 4, "1140"),  # a lag of 45 misses the off
    ("jra:2025-06-01:TOK:05", date(2025, 6, 1), "1200", 3, "missing"),
    ("jra:2025-06-01:TOK:06", date(2025, 6, 1), "1230", 3, "nat"),
    ("jra:2025-06-01:TOK:11", date(2025, 6, 1), "1532", 6, 13),  # the day's last off
    ("jra:2025-06-08:KYO:01", date(2025, 6, 8), "0850", 4, -2),  # UTC day before; after its off
    ("jra:2025-06-08:KYO:11", date(2025, 6, 8), "1545", 5, 5),
    ("jra:2025-06-08:KYO:12", date(2025, 6, 8), "1610", 3, 20),  # in decision_utc, not requested
]
OFF_NOT_REQUESTED = "jra:2025-06-08:KYO:12"
OFF_NO_ODDS = ("jra:2025-06-01:TOK:11", 2)


def off_of(day: date, hhmm: str, lag_min: float, off) -> pd.Timestamp:
    """The race's off as a UTC instant (NaT for "nat"), from OFF_RACES' last field."""
    if off == "nat":
        return pd.NaT
    if isinstance(off, str):
        return released_at(day, off, 0)
    return released_at(day, hhmm, lag_min) + pd.Timedelta(minutes=off)


@pytest.mark.parametrize(
    ("lag", "zone"),
    [(None, "UTC"), (0, "Asia/Tokyo"), (45, "UTC")],
    ids=["longest-configured", "0-offs-in-JST", "45"],
)
def test_each_race_reads_its_board_as_of_its_own_off(lag, zone):
    minutes = max(LAGS) if lag is None else lag
    rules = {"tz": BOARD_CONFIG["tz"], "release_lag_min": minutes}

    offs = {r[0]: off_of(r[1], r[2], minutes, r[4]) for r in OFF_RACES if r[4] != "missing"}
    decision = pd.Series(
        pd.DatetimeIndex(list(offs.values())).tz_convert(zone),
        index=pd.Index(list(offs), dtype=object),
    )
    requested = [r[0] for r in OFF_RACES if r[0] != OFF_NOT_REQUESTED]
    con = duckdb.connect(":memory:")
    try:
        races = [r[:4] for r in OFF_RACES]
        store = put_jrdb_store(con, seed=11, races=races, no_odds=OFF_NO_ODDS)
        got = jrdb_board(con, requested, rules=rules, decision_utc=decision)
    finally:
        con.close()

    release = {r[0]: released_at(r[1], r[2], minutes) for r in OFF_RACES}
    off = {rid: offs.get(rid, pd.NaT) for rid in requested}
    last_off = max(t for t in off.values() if not pd.isna(t))
    # The fixture holds what a cutoff at the day's last off, or at now, would get wrong.
    late = [r for r in requested if not pd.isna(off[r]) and off[r] < release[r] <= last_off]
    timeless = [r for r in requested if pd.isna(off[r])]
    assert {"jra:2025-06-01:TOK:01", "jra:2025-06-08:KYO:01"} <= set(late)
    assert any(off[r] == release[r] for r in requested), "a board released exactly at its off"
    assert len(timeless) == 2 and all(release[r] <= last_off for r in timeless)

    assert {"race_id", "horse_no", "tyb_win_odds", "release_utc"} <= set(got.columns)
    keys = [(str(r), int(h)) for r, h in zip(got["race_id"], got["horse_no"], strict=True)]
    assert len(keys) == len(set(keys)), "one row per runner at most"
    assert {r for r, _ in keys} <= set(requested), "only the requested races"
    at = dict(zip(keys, range(len(got)), strict=True))

    expected = {}
    for row in store.itertuples(index=False):
        key, rid = (row.race_id, int(row.horse_no)), row.race_id
        if rid in off and not pd.isna(off[rid]) and release[rid] <= off[rid] and key != OFF_NO_ODDS:
            expected[key] = (row.tyb_win_odds, release[rid])
    assert {r for r, _ in expected} >= {"jra:2025-06-01:TOK:11", "jra:2025-06-08:KYO:11"}

    for key, (board, when) in expected.items():
        assert key in at, f"{key}: released by its own off {off[key[0]]}, so it is read"
        row = got.iloc[at[key]]
        assert float(row["tyb_win_odds"]) == pytest.approx(board), f"{key}: the board's own odds"
        stamp = pd.Timestamp(row["release_utc"])
        assert stamp.tzinfo is not None, f"{key}: release_utc must be timezone-aware"
        assert stamp.tz_convert("UTC") == when, f"{key}: released {stamp}, want {when}"

    final = {(r.race_id, int(r.horse_no)): r.final_win_odds for r in store.itertuples(index=False)}
    for key in set(at) - set(expected):
        value = got.iloc[at[key]]["tyb_win_odds"]
        assert pd.isna(value), (
            f"{key}: released {release[key[0]]}, its off {off[key[0]]}: no board, yet reads "
            f"{value} (final odds {final[key]})"
        )


# ---------------------------------------------------------------------------------------------
# Ticket #603: the measured release time. `jrdb_pre_race.tyb_fetched_utc` is when the TYB file was
# fetched (a UTC timestamp, null when unrecorded). `jrdb_board` returns it as `fetched_utc` and
# takes a new optional rule `rules["release"]`: "lag" (the default when absent: the rule above) or
# "fetched" (release_utc is the fetch time; `release_lag_min` is not required; a null fetch time
# is never released). Any other value raises ValueError. The frozen config is not edited.
# ---------------------------------------------------------------------------------------------

# race_id, day, odds time, runners, off (JST clock), fetch time as seconds after the off (None: no
# fetch time), expected under "fetched". The odds time is chosen so that the lag rule (10 minutes)
# disagrees with the fetch rule on most of them.
FETCH_RACES = [
    ("jra:2025-06-01:TOK:01", date(2025, 6, 1), "0950", 4, "1000", 60, False),  # fetched after off
    ("jra:2025-06-01:TOK:02", date(2025, 6, 1), "1025", 5, "1030", -90, True),  # lag: after off
    ("jra:2025-06-01:TOK:03", date(2025, 6, 1), "1105", 4, "1140", 0, True),  # exactly at the off
    ("jra:2025-06-01:TOK:05", date(2025, 6, 1), "1200", 3, "1240", None, False),  # never fetched
    ("jra:2025-06-01:TOK:06", date(2025, 6, 1), None, 3, "1300", -1800, True),  # no odds time
    ("jra:2025-06-01:TOK:07", date(2025, 6, 1), "1075", 3, "1330", -300, True),  # bad odds time
    ("jra:2025-06-01:TOK:08", date(2025, 6, 1), "1300", 3, None, -600, False),  # no decision time
    ("jra:2025-06-01:TOK:11", date(2025, 6, 1), "1500", 6, "1532", -7200, True),  # fetched early
    ("jra:2025-06-08:KYO:01", date(2025, 6, 8), "0850", 4, "0900", -180, True),  # UTC day before
]
FETCH_NO_ODDS = ("jra:2025-06-01:TOK:11", 2)


def fetch_instant(day: date, off_hhmm: str, seconds: int | None) -> pd.Timestamp | None:
    if seconds is None:
        return None
    return released_at(day, off_hhmm, 0) + pd.Timedelta(seconds=seconds)


@pytest.mark.parametrize("zone", ["UTC", "Asia/Tokyo"])
def test_a_fetched_board_is_released_at_its_fetch_time(zone):
    """Under `release: fetched` a runner's board counts when its TYB file was fetched at or before
    its own race's off, whatever its odds time says; release_utc and fetched_utc are the fetch
    time. Fetched after the off, never fetched (null), or a race with no decision time: no board
    (absent, or null odds); never a value from another price. `release_lag_min` is not given."""
    rules = {"tz": "Asia/Tokyo", "release": "fetched"}
    fetched = {r[0]: fetch_instant(r[1], r[4] or "1200", r[5]) for r in FETCH_RACES}
    fetched = {k: v for k, v in fetched.items() if v is not None}
    offs = {r[0]: released_at(r[1], r[4], 0) for r in FETCH_RACES if r[4] is not None}
    decision = pd.Series(
        pd.DatetimeIndex(list(offs.values())).tz_convert(zone),
        index=pd.Index(list(offs), dtype=object),
    )
    requested = [r[0] for r in FETCH_RACES]
    con = duckdb.connect(":memory:")
    try:
        store = put_jrdb_store(
            con, seed=21, races=[r[:4] for r in FETCH_RACES], no_odds=FETCH_NO_ODDS, fetched=fetched
        )
        got = jrdb_board(con, requested, rules=rules, decision_utc=decision)
    finally:
        con.close()

    assert {"race_id", "horse_no", "tyb_win_odds", "release_utc", "fetched_utc"} <= set(got.columns)
    keys = [(str(r), int(h)) for r, h in zip(got["race_id"], got["horse_no"], strict=True)]
    assert len(keys) == len(set(keys)), "one row per runner at most"
    at = dict(zip(keys, range(len(got)), strict=True))

    counted = {r[0] for r in FETCH_RACES if r[6]}
    expected = {}
    for row in store.itertuples(index=False):
        key = (row.race_id, int(row.horse_no))
        if row.race_id in counted and key != FETCH_NO_ODDS:
            expected[key] = (row.tyb_win_odds, fetched[row.race_id])
    assert len(expected) == 5 + 4 + 3 + 3 + 5 + 4  # TOK:02, 03, 06, 07, 11 but one runner, KYO:01

    for key, (board, when) in expected.items():
        assert key in at, f"{key}: fetched by its own off, so it is read"
        row = got.iloc[at[key]]
        assert float(row["tyb_win_odds"]) == pytest.approx(board), f"{key}: the board's own odds"
        for column in ("release_utc", "fetched_utc"):
            stamp = pd.Timestamp(row[column])
            assert stamp.tzinfo is not None, f"{key}: {column} must be timezone-aware"
            assert stamp.tz_convert("UTC") == when, f"{key}: {column} {stamp}, want {when}"

    final = {(r.race_id, int(r.horse_no)): r.final_win_odds for r in store.itertuples(index=False)}
    for key in set(at) - set(expected):
        value = got.iloc[at[key]]["tyb_win_odds"]
        assert pd.isna(value), f"{key} was not fetched in time, yet reads {value} ({final[key]})"


def test_a_fetched_board_without_a_decision_time_is_read_as_of_now():
    """No decision_utc: the board is read as of now. A fetch time in the past counts, one in the
    future (or null) does not, whatever the odds time; the lag rule would read the opposite for
    the races whose odds time is later than their fetch."""
    past = pd.Timestamp("2025-06-01T00:40:00Z")
    fetched = {
        "jra:2025-06-01:TOK:01": past,
        "jra:2025-06-01:TOK:11": pd.Timestamp("2099-01-01T00:00:00Z"),
        "jra:2025-06-08:KYO:01": None,
        "jra:2025-06-08:KYO:05": past + pd.Timedelta(days=7),  # no odds time: still released
        "jra:2025-06-08:KYO:06": past + pd.Timedelta(days=7, seconds=5),  # bad odds time
    }
    requested = [r[0] for r in BOARD_RACES if r[0] != NOT_REQUESTED]
    con = duckdb.connect(":memory:")
    try:
        store = put_jrdb_store(con, fetched=fetched)
        got = jrdb_board(con, requested, rules={"release": "fetched"})
    finally:
        con.close()
    readable = {r for r, t in fetched.items() if t is not None and t < pd.Timestamp.now(tz="UTC")}
    assert readable == {
        "jra:2025-06-01:TOK:01",
        "jra:2025-06-08:KYO:05",
        "jra:2025-06-08:KYO:06",
    }
    final = {(r.race_id, int(r.horse_no)): r for r in store.itertuples(index=False)}
    seen = set()
    for row in got.itertuples(index=False):
        key = (str(row.race_id), int(row.horse_no))
        if pd.isna(row.tyb_win_odds):
            continue
        seen.add(key)
        assert key[0] in readable, f"{key}: not fetched yet, yet reads {row.tyb_win_odds}"
        assert float(row.tyb_win_odds) == pytest.approx(final[key].tyb_win_odds)
        assert pd.Timestamp(row.release_utc).tz_convert("UTC") == fetched[key[0]]
        assert pd.Timestamp(row.fetched_utc).tz_convert("UTC") == fetched[key[0]]
    want = {k for k, r in final.items() if r.race_id in readable and k != NO_ODDS}
    assert seen == want and len(want) == 5 + 5 + 5


@pytest.mark.parametrize("minutes", release_lags(BOARD_CONFIG), ids=lambda m: f"lag{m}")
@pytest.mark.parametrize("release", [None, "lag"], ids=["absent", "lag"])
def test_the_lag_rule_stays_the_default_and_returns_the_fetch_time(release, minutes):
    """Without `release`, or with "lag", release_utc is odds time + release_lag_min as before,
    whatever the fetch time (even one in the future, or null); `fetched_utc` is the row's own
    fetch time (NaT where unrecorded). Each lag of the configured `release_lags_min` (#602: the
    config lists the lags, `jrdb_board` takes one as `release_lag_min`) is checked."""
    configured = load_country_yaml("japan", "board_vs_final")["board"]
    assert "release" not in configured  # the frozen family's config is not edited
    assert minutes in release_lags(configured)
    rules = {**configured, "release_lag_min": minutes}
    if release is not None:
        rules["release"] = release
    fetched = {
        "jra:2025-06-01:TOK:01": pd.Timestamp("2030-01-01T00:00:00Z"),  # not yet fetched, but lag
        "jra:2025-06-08:KYO:01": pd.Timestamp("2025-06-07T23:51:07Z"),
    }
    requested = [r[0] for r in BOARD_RACES if r[0] != NOT_REQUESTED]
    con = duckdb.connect(":memory:")
    try:
        store = put_jrdb_store(con, fetched=fetched)
        got = jrdb_board(con, requested, rules=rules)
    finally:
        con.close()

    assert "fetched_utc" in got.columns
    by_race = {}
    for row in got.itertuples(index=False):
        if pd.isna(row.tyb_win_odds):
            continue
        by_race.setdefault(str(row.race_id), []).append(row)
        day = store.loc[store["race_id"] == row.race_id, "race_date"].iloc[0]
        odds_time = store.loc[store["race_id"] == row.race_id, "tyb_odds_time"].iloc[0]
        assert pd.Timestamp(row.release_utc).tz_convert("UTC") == released_at(
            day, odds_time, minutes
        )
    assert set(by_race) == {
        "jra:2025-06-01:TOK:01",
        "jra:2025-06-01:TOK:11",
        "jra:2025-06-08:KYO:01",
    }
    for race_id, rows in by_race.items():
        for row in rows:
            if race_id in fetched:
                assert pd.Timestamp(row.fetched_utc).tz_convert("UTC") == fetched[race_id]
            else:
                assert pd.isna(row.fetched_utc)


@pytest.mark.parametrize("release", ["bogus", "Fetched", "", 0, True])
def test_an_unknown_release_rule_is_refused(release):
    con = duckdb.connect(":memory:")
    try:
        put_jrdb_store(con)
        with pytest.raises(ValueError):
            jrdb_board(
                con, ["jra:2025-06-01:TOK:01"], rules={"release_lag_min": 10, "release": release}
            )
    finally:
        con.close()


def test_the_lag_rule_still_needs_its_lag_but_the_fetched_rule_does_not():
    con = duckdb.connect(":memory:")
    try:
        put_jrdb_store(con, fetched={"jra:2025-06-01:TOK:01": pd.Timestamp("2025-06-01T00:00:00Z")})
        for rules in ({"tz": "Asia/Tokyo"}, {"tz": "Asia/Tokyo", "release": "lag"}):
            with pytest.raises(ValueError):
                jrdb_board(con, ["jra:2025-06-01:TOK:01"], rules=rules)
        got = jrdb_board(con, ["jra:2025-06-01:TOK:01"], rules={"release": "fetched"})
        assert len(got) == 5 and got["tyb_win_odds"].notna().sum() == 5
    finally:
        con.close()


# ---------------------------------------------------------------------------------------------
# Outputs: the report, the private leaderboard and the registry records
# ---------------------------------------------------------------------------------------------


def cell_tables(keys) -> dict:
    """A cross_apply table per (hypothesis, pool), each from its own synthetic races."""
    tables = {}
    for k, key in enumerate(keys):
        frame = synthetic_races(100 + k, per_season=10)
        tables[key] = cross_apply(
            frame, fit_price=FINAL, score_price=BOARD, fit=power_fit, score=power_score
        )
    return tables


def cells_of(tables: Mapping) -> list[tuple]:
    out = []
    for key, table in tables.items():
        for f, s in itertools.product((FINAL, BOARD), repeat=2):
            out.append((key, f, s, *(cell(table, f, s, stat) for stat in STATS)))
    return out


def written_paths(result) -> list[Path]:
    items = result.values() if isinstance(result, Mapping) else result
    return [Path(p) for p in items]


def test_the_report_cannot_be_written_inside_the_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path / "data"))
    tables = cell_tables(ALL_KEYS)

    inside = env.repo_root() / f".tmp-board-vs-final-{uuid.uuid4().hex[:8]}"
    try:
        with pytest.raises(LicensedOutputInRepo):
            write_outputs(tables, out_dir=inside)
        created = [p for p in inside.rglob("*") if p.is_file()] if inside.exists() else []
        assert created == []
    finally:
        shutil.rmtree(inside, ignore_errors=True)

    # Outside the repository the same outputs are written, all under out_dir.
    out = tmp_path / "data" / "reports" / "board-vs-final"
    paths = written_paths(write_outputs(tables, out_dir=out))
    assert paths
    for path in paths:
        assert path.is_file(), path
        assert path.resolve().is_relative_to(out.resolve()), path


def close(a, b) -> bool:
    return a is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


@pytest.mark.parametrize(
    "keys",
    [
        pytest.param([("E2", WIN, LAGS[0])], id="e2-one-lag"),
        pytest.param(ALL_KEYS, id="e2-and-e45-every-lag"),
    ],
)
def test_each_cell_is_a_withheld_member_row_and_a_jrdb_record(tmp_path, monkeypatch, keys):
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path / "data"))
    tables = cell_tables(keys)
    cells = cells_of(tables)
    assert len(cells) == 4 * len(keys)

    rows = leaderboard_rows(tables)
    assert all(isinstance(r, LeaderboardRow) for r in rows)
    assert len(rows) == len(cells), "one member row per fit x score cell"
    assert {r.kind for r in rows} == {"member"}
    assert all(r.withheld for r in rows), "D21: every JRDB-derived row is withheld"
    assert len({r.option_id for r in rows}) == len(rows)
    assert Counter(r.pool for r in rows) == Counter(pool for _, pool, _ in keys for _ in range(4))
    # #602: every lag x table x cell is a counted variant, on every row.
    assert {r.variants_searched for r in rows} == {len(cells)}

    records = run_records(tables, git_sha="0" * 40, data_builds={"store": "synthetic@0"})
    assert records and all(isinstance(r, RunRecord) for r in records)
    assert all("jrdb" in r.sources for r in records)
    assert all(r.ticket == 103 for r in records)
    assert {r.hypothesis_id for r in records} == {h for h, _, _ in keys}

    metrics = [m for r in records for m in r.metrics]
    for key, f, s, mean, lo, hi in cells:
        assert any(close(m.value, mean) and close(m.lo, lo) and close(m.hi, hi) for m in metrics), (
            f"{key}: the cell fit on {f}, scored on {s} is in no record"
        )

    registry = Registry(
        tmp_path / "registry" / "japan.jsonl", holdout_log=tmp_path / "holdout_log.jsonl"
    )
    for record in records:
        registry.append(record)
    registry.verify()
    summary = tmp_path / "summary"
    write_summary(registry, summary)
    text = "\n".join(p.read_text(encoding="utf-8") for p in sorted(summary.glob("*.md")))
    assert WITHHELD_LABEL in text
    for *_, mean, lo, hi in cells:
        for value in (mean, lo, hi):
            assert f"{value:.6g}" not in text, "a JRDB-derived figure reached the summary"


# ---------------------------------------------------------------------------------------------
# #602: the release lag is a counted variant, not one assumed value
#
# Contract (*): config/board_vs_final.yaml's `board.release_lags_min` is a list of lags (minutes
# >= 0, at least two, distinct, the earlier assumed 10 among them) and replaces the single
# `board.release_lag_min`; the family id is `japan-board-vs-final-e2-e45-v3`. Table keys are
# (hypothesis, pool, lag_min); coefficient keys (hypothesis, price, lag_min). A record's
# components hold `release_lag_min` and its variant, a leaderboard option id and the cell's JSON
# row name the lag as `lag=<lag:g>` / `lag_min`; variants_searched counts lags x tables x cells.
# `jrdb_boards(con, race_ids, *, rules, decision_utc=None) -> {lag_min: board}` runs `jrdb_board`
# once per lag of `rules["release_lags_min"]` (ValueError unless distinct non-negative minutes,
# at least one), in the list's order; the study builds its per-lag boards through it.
# ---------------------------------------------------------------------------------------------

LAG_TOKEN = re.compile(r"(?:^|[:,;\s])lag=([0-9.]+)")


def lag_of(text: str) -> float:
    found = LAG_TOKEN.search(text)
    assert found, f"{text!r} does not name its lag as lag=<minutes>"
    return float(found.group(1))


def board_requested() -> list[str]:
    return [r[0] for r in BOARD_RACES if r[0] != NOT_REQUESTED]


LAG_PAIRS = [(0, 10), (3, 25), (10, 45)]


@pytest.mark.parametrize(("short", "long"), LAG_PAIRS, ids=[f"{a}-{b}" for a, b in LAG_PAIRS])
def test_two_lags_give_two_variants_with_distinct_release_times(short, long):
    con = duckdb.connect(":memory:")
    try:
        store = put_jrdb_store(con)
        requested = board_requested()
        rules = {"tz": BOARD_CONFIG["tz"], "release_lags_min": [short, long]}
        boards = jrdb_boards(con, requested, rules=rules)
        # An off between the two release times: the short lag's board is out by then, the long
        # lag's is not. Each race's own odds time sets its clock.
        middle = (short + long) / 2
        offs = pd.Series(
            {
                r[0]: released_at(r[1], r[2], middle)
                for r in BOARD_RACES
                if r[0] != NOT_REQUESTED and r[2] in RELEASED
            }
        )
        at_off = jrdb_boards(con, requested, rules=rules, decision_utc=offs)
        assert list(boards) == [short, long] == list(at_off), "one board per lag, in order"
    finally:
        con.close()

    def stamps(board: pd.DataFrame) -> dict:
        return {
            (str(r), int(h)): pd.Timestamp(t).tz_convert("UTC")
            for r, h, t in zip(
                board["race_id"], board["horse_no"], board["release_utc"], strict=True
            )
        }

    a, b = stamps(boards[short]), stamps(boards[long])
    assert a and set(a) == set(b), "the same runners are released, only later"
    odds_time = {
        (r.race_id, int(r.horse_no)): (r.race_date, r.tyb_odds_time) for r in store.itertuples()
    }
    for key in a:
        day, hhmm = odds_time[key]
        assert a[key] == released_at(day, hhmm, short), key
        assert b[key] == released_at(day, hhmm, long), key
        assert b[key] - a[key] == pd.Timedelta(minutes=long - short), f"{key}: distinct times"
    odds = {
        lag: dict(zip(stamps(boards[lag]), boards[lag]["tyb_win_odds"], strict=True))
        for lag in boards
    }
    for key, value in odds[short].items():  # the lag moves the release, never the odds
        assert (pd.isna(value) and pd.isna(odds[long][key])) or value == pytest.approx(
            odds[long][key]
        )

    early, late = stamps(at_off[short]), stamps(at_off[long])
    expected_races = {r for r, _ in a}
    assert {r for r, _ in early} == expected_races, "released before the off at the short lag"
    assert late == {}, "the long lag's board is not out by the same off"


@pytest.mark.parametrize("lags", [[], [10, 10], [0, -5], [5, True], [5, float("nan")], "10"])
def test_jrdb_boards_refuses_lags_that_are_not_distinct_minutes(lags):
    con = duckdb.connect(":memory:")
    try:
        put_jrdb_store(con)
        with pytest.raises(ValueError):
            jrdb_boards(
                con, board_requested(), rules={"tz": BOARD_CONFIG["tz"], "release_lags_min": lags}
            )
    finally:
        con.close()


def test_the_config_lists_the_lags_as_distinct_variants():
    board = BOARD_CONFIG
    assert "release_lag_min" not in board, "one assumed lag is replaced by the list"
    lags = board["release_lags_min"]
    assert isinstance(lags, list) and len(lags) >= 2
    assert all(isinstance(x, int | float) and not isinstance(x, bool) and x >= 0 for x in lags)
    assert len(set(lags)) == len(lags), "each lag counted once"
    assert 10 in lags, "the lag #103 assumed stays a variant"
    assert FAMILY_ID == "japan-board-vs-final-e2-e45-v3", "a changed study is a new family"


def test_the_frozen_spec_lists_every_lag():
    spec = build_spec(load_country_scheme("japan"), bootstrap_settings())

    assert spec["family_id"] == FAMILY_ID
    assert spec["board"]["release_lags_min"] == list(LAGS)
    assert "release_lag_min" not in spec["board"], "no single assumed lag beside the variants"
    json.dumps(spec)  # freezable


def expected_cells_per_lag() -> int:
    return 4 * len(TABLES)


def test_every_lag_is_its_own_record_and_leaderboard_row(tmp_path):
    tables = cell_tables(ALL_KEYS)
    coefficients = {
        (h, price, lag): [FoldMetric("fold-1", "beta", 0.1 + lag / 100 + k, 0.0, 1.0, 10)]
        for lag in LAGS
        for h in ("E2", "E45")
        for k, price in enumerate((FINAL, BOARD))
    }
    records = run_records(
        tables, git_sha="0" * 40, data_builds={"store": "synthetic@0"}, coefficients=coefficients
    )

    assert len(records) == len(ALL_KEYS) * 2 + len(coefficients)
    assert len({(r.hypothesis_id, r.variant) for r in records}) == len(records), "variants differ"
    registry = Registry(tmp_path / "japan.jsonl", holdout_log=tmp_path / "holdout_log.jsonl")
    ids = [registry.append(r) for r in records]
    registry.verify()
    assert len(set(ids)) == len(records), "one registry record per lag x cell, none collapsed"
    assert all("jrdb" in r.sources and r.family_id == FAMILY_ID for r in records)

    for r in records:
        lag = float(r.components["release_lag_min"])
        assert lag in LAGS and lag_of(r.variant) == lag, "the record names its own lag"

    for h, pool, lag in ALL_KEYS:
        for f in (FINAL, BOARD):
            got = [
                r
                for r in records
                if (r.hypothesis_id, r.components.get("pool"), r.components.get("fit_price"))
                == (h, pool, f)
                and float(r.components["release_lag_min"]) == lag
            ]
            assert len(got) == 1, f"{(h, pool, lag, f)}: one record, got {len(got)}"
            for s in (FINAL, BOARD):
                want = tuple(cell(tables[(h, pool, lag)], f, s, stat) for stat in STATS)
                assert any(
                    close(m.value, want[0]) and close(m.lo, want[1]) and close(m.hi, want[2])
                    for m in got[0].metrics
                ), f"{(h, pool, lag, f, s)}: the lag's own cell is not in its record"

    for (h, price, lag), metrics in coefficients.items():
        got = [
            r
            for r in records
            if r.hypothesis_id == h
            and r.components.get("fit_price") == price
            and "pool" not in r.components
            and float(r.components["release_lag_min"]) == lag
        ]
        assert len(got) == 1, f"{(h, price, lag)}: one coefficient record"
        assert [m.value for m in got[0].metrics] == [m.value for m in metrics]

    rows = leaderboard_rows(tables)
    assert {r.variants_searched for r in rows} == {len(LAGS) * expected_cells_per_lag()}
    assert len({r.option_id for r in rows}) == len(rows) == len(ALL_KEYS) * 4
    per_lag = Counter(lag_of(r.option_id) for r in rows)
    assert per_lag == {float(lag): expected_cells_per_lag() for lag in LAGS}
    assert all(r.withheld and "jrdb" in r.sources for r in rows)


def test_the_outputs_carry_every_lag_under_the_data_dir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(data))
    tables = cell_tables(ALL_KEYS)

    written = write_outputs(tables, out_dir=data / "reports" / "lags")

    for path in written_paths(written):
        assert path.resolve().is_relative_to(data.resolve()), path
    cells = json.loads(Path(written["tables"]).read_text(encoding="utf-8"))
    assert Counter(float(c["lag_min"]) for c in cells) == {
        float(lag): expected_cells_per_lag() for lag in LAGS
    }
    board = json.loads(Path(written["leaderboard_json"]).read_text(encoding="utf-8"))
    assert Counter(lag_of(r["option_id"]) for r in board) == {
        float(lag): expected_cells_per_lag() for lag in LAGS
    }
