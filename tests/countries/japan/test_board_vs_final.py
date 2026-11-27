"""Board-fitted against final-fitted terms on the JRDB board (ticket #103, epic #73; 08 §4 E2, E45,
R13).

Contract pinned here (`horsetrack.countries.japan.studies.board_vs_final`). Where the ticket left
a choice open, the test-author's choice is marked (*).

- `jrdb_board(con, race_ids, *, rules) -> pd.DataFrame`: JRDB's fixed-time board, the TYB
  family's `tyb_win_odds` taken at `tyb_odds_time` ("HHMM" on the JST clock of `race_date`), from
  the store table `jrdb_pre_race` in the DuckDB connection `con`, read through `read_asof` at the
  JRDB release time. (*) `rules` is the release-time rule as data: the `board` mapping of
  `countries/japan/config/board_vs_final.yaml` (`load_country_yaml("japan", "board_vs_final")
  ["board"]`), whose (*) `release_lag_min` (a number of minutes, >= 0) runs from JRDB's odds time
  to its release. A row is released at race_date, tyb_odds_time (Asia/Tokyo) + release_lag_min.
  The frame has `race_id`, `horse_no`, `tyb_win_odds` and (*) `release_utc`, that instant
  (timezone-aware; the snapshot a board-fitted term is reported with, R13), at most one row per
  runner and only for the requested races. A row with no release time (a missing or malformed
  odds time) or no board odds has no board: it is absent, or its `tyb_win_odds` is null. A board
  value is never filled from another price: the store's final `runners.win_odds`, or KYI's
  day-before `kyi_base_odds`. The owner's `jrdb_pre_race` rule in availability.yaml (tyb_ columns
  are labels) is not edited; `test_availability.py` pins it.
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
import math
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
from horsetrack.countries.japan.studies.board_vs_final import (
    cross_apply,
    jrdb_board,
    leaderboard_rows,
    run_records,
    write_outputs,
)

STATS = ("mean", "lo", "hi")
FINAL, BOARD = "q_final", "q_board"
ALL_KEYS = [("E2", WIN), ("E45", PLACE), ("E45", WIDE), ("E45", TRIO)]
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


def put_jrdb_store(con, seed: int = 5) -> pd.DataFrame:
    """`jrdb_pre_race` as the store holds it (the store's key columns, then KYI's day-before board
    and TYB's day-of columns) and `runners` with the final Win odds, every price different.
    Returns the rows, one per runner, with the final odds beside them."""
    rng = np.random.default_rng(seed)
    rows = []
    for race_id, day, odds_time, n in BOARD_RACES:
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
                    "final_win_odds": round(board * 0.8 + 0.2, 1),
                }
            )
    frame = pd.DataFrame(rows)
    pre_race = frame.drop(columns=["final_win_odds"])
    casts = {"race_date": "DATE", "horse_no": "BIGINT", "tyb_odds_time": "VARCHAR"}
    put(con, "jrdb_pre_race", pre_race, casts)
    for race_id, _, odds_time, _ in BOARD_RACES:
        if odds_time is None:
            con.execute(
                "UPDATE jrdb_pre_race SET tyb_odds_time = NULL WHERE race_id = ?", [race_id]
            )
    con.execute(
        "UPDATE jrdb_pre_race SET tyb_win_odds = NULL WHERE race_id = ? AND horse_no = ?",
        list(NO_ODDS),
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


@pytest.mark.parametrize("lag", [None, 0, 7, 45], ids=["as-configured", "0", "7", "45"])
def test_the_board_is_read_as_of_the_jrdb_release_time(lag):
    configured = load_country_yaml("japan", "board_vs_final")["board"]
    minutes = configured["release_lag_min"]
    assert isinstance(minutes, int | float) and not isinstance(minutes, bool) and minutes >= 0
    rules = dict(configured) if lag is None else {**configured, "release_lag_min": lag}
    minutes = minutes if lag is None else lag

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


@pytest.mark.parametrize(
    "keys", [pytest.param([("E2", WIN)], id="e2-only"), pytest.param(ALL_KEYS, id="e2-and-e45")]
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
    assert Counter(r.pool for r in rows) == Counter({pool: 4 for _, pool in keys})

    records = run_records(tables, git_sha="0" * 40, data_builds={"store": "synthetic@0"})
    assert records and all(isinstance(r, RunRecord) for r in records)
    assert all("jrdb" in r.sources for r in records)
    assert all(r.ticket == 103 for r in records)
    assert {r.hypothesis_id for r in records} == {h for h, _ in keys}

    def close(a, b) -> bool:
        return a is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)

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
