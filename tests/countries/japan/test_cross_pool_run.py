"""The T-2 cross-pool signal study, E25 (ticket #435, family `japan-cross-pool-t2-v1`).

Module under test: `horsetrack.countries.japan.studies.cross_pool_run`. Contract pinned here (the
ticket, its owner reframe of 2026-11-27 and the #435 design brief):

- `race_probs(offset, X, coef, race_id)`: the softmax per race of `offset + X @ coef` (rows need
  not be grouped by race). With coef = 0 and offset = ln q (+ any per-race constant) it returns q.
- `day_rows(day, *, minutes_before=MINUTES_BEFORE)`: one odds_timeseries day -> one row per Win
  starter at t with `race_id, horse_no, p_win_t, x_cross, is_fav, fav_x_cross`, built from the
  board at t (`studies.cross_pool.boards_at`, rows with minutes_to_post >= minutes_before) and
  `cross_features` (lambda2 1, zero votes unavailable): x_cross = ln(q_top2_t / h_top2_t),
  fav_x_cross = is_fav * x_cross. `MINUTES_BEFORE` is config/decision.yaml's primary_min.
- `fit_fold(train, *, features)`: the unpenalised conditional logit of `train[features]` with
  `offset=train["offset"]` and `cluster=train["meeting_id"]`, through the module global
  `fit_conditional_logit` (a test may wrap it).
- `score_folds(rows, folds)`: per fold (positions into `rows`) and per variant of `VARIANTS`
  ("<features>:<offset>"), the offset on the fold's rows (normalised: ln of 1/odds normalised per
  race; calibrated: `models.v0.calibration.fit_win_calibration` fitted on the fold's training
  rows, applied to its training and test rows), coefficients fitted on `rows.iloc[train_idx]`
  only, and one row per (variant, test race with a winner): `variant, fold` (the fold's index),
  `race_id, race_date, meeting_id, class_group, ll_offset, ll_model, delta_ll` (winner log loss,
  a dead heat's winners averaged; delta = offset - model), `coef_x_cross, coef_fav_x_cross` (NaN
  where the variant lacks the feature). `predict_folds(rows, folds)`: `variant, fold, race_id,
  horse_no, p_offset, p_model` per test-fold runner.
- `settle(bets, dividends, *, pool)`: the return per unit stake at the final dividend, dividend /
  100 for a (race_id, pool, combination) the dividends pay, else 0; never the board's odds.
- `build_spec(*, date_from, date_to, seed)`, `freeze_spec(...)` (the registry's freeze, its hash),
  `FamilyNotFrozen`; `run_cross_pool(*, date_from, date_to, out_dir, seed, registry)` refuses a
  span before 2014-06-07 (D10) and an unfrozen family before any store file is opened, never
  opens a holdout day's file (D14), writes the private report and the candidate-cell file through
  `write_report` and appends one registry record per variant, failures included.
- `render_public_report(result)` prints no figure from the result: "withheld (D21)".

Expected values are derived here from first principles or from the dependencies the brief names
(`models.condlogit`, `models.v0.calibration`, `core.eval.folds`), never from the module under test.
Every store, board, result and dividend is synthetic and built in the test; HORSETRACK_DATA_DIR
points under tmp; no real data and no holdout data are read.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import builtins
import glob
import hashlib
import io
import itertools
import json
import numbers
import os
import pathlib
import shutil
import uuid
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from horsetrack.core import env
from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.folds import FoldScheme, walk_forward
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import Registry
from horsetrack.core.eval.reports import LicensedOutputInRepo
from horsetrack.core.eval.tail import COLUMNS as TAIL_COLUMNS
from horsetrack.core.paths import DataDirInsideRepo
from horsetrack.core.store.build import arrow_schema
from horsetrack.countries.japan.jravan.odds import FINAL_ODDS_SCHEMA
from horsetrack.countries.japan.logger.compact import SCHEMA
from horsetrack.countries.japan.models import condlogit
from horsetrack.countries.japan.models.v0.calibration import fit_win_calibration
from horsetrack.countries.japan.studies import cross_pool_run
from horsetrack.countries.japan.studies.cross_pool_run import (
    FAMILY_ID,
    VARIANTS,
    CrossPoolResult,
    FamilyNotFrozen,
    build_spec,
    day_rows,
    fit_fold,
    freeze_spec,
    predict_folds,
    race_probs,
    render_public_report,
    run_cross_pool,
    score_folds,
    settle,
)

UTC = timezone.utc
SEED = 435
SPAN_START = date(2014, 6, 7)  # D10
# A short D9-shaped scheme (expanding, 7-day embargo) so a few months of synthetic days give folds.
SHORT = FoldScheme(
    unit="month",
    test_length=1,
    step=1,
    min_train=2,
    window="expanding",
    embargo_days=7,
    anchor_month=1,
)
EXPECTED_VARIANTS = {
    "x_cross:normalised",
    "x_cross:calibrated",
    "x_cross+fav:normalised",
    "x_cross+fav:calibrated",
}
FEATURES_OF = {"x_cross": ["x_cross"], "x_cross+fav": ["x_cross", "fav_x_cross"]}
POOLS = ("WIN", "QUINELLA", "FORECAST")
CLASS_GROUPS = ("newcomer", "maiden", "conditions", "open", "graded")
CELL_KEYS = {
    "study",
    "family_id",
    "cell_id",
    "pool",
    "class_group",
    "odds_band",
    "payout_rate",
    "n_bets",
    "n_race_days",
    "n_meetings",
    "mean_return",
    "se_mean_return",
    "shrunk_edge",
    "mean_odds",
}
POOL_ROW = "__pool__"


# ---------------------------------------------------------------- references (first principles)


def ref_softmax(u, race) -> np.ndarray:
    """exp(u) / sum of exp(u) over the race's rows, race by race (shifted by the race's max)."""
    u = np.asarray(u, dtype=float)
    race = np.asarray(race, dtype=object)
    out = np.empty_like(u)
    for r in pd.unique(race):
        m = race == r
        e = np.exp(u[m] - u[m].max())
        out[m] = e / e.sum()
    return out


def ref_ln_q(frame: pd.DataFrame) -> np.ndarray:
    """ln of 1/odds normalised per race."""
    inv = 1.0 / frame["win_odds"].to_numpy(dtype=float)
    total = pd.Series(inv).groupby(frame["race_id"].to_numpy()).transform("sum").to_numpy()
    return np.log(inv / total)


def ref_shares(odds: dict) -> dict:
    inv = {k: 1.0 / v for k, v in odds.items()}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}


def ref_harville(p: dict[int, float]) -> dict[tuple[int, int], float]:
    """P({a, b} first two, either order), plain Harville (lambda2 = 1)."""
    return {
        (a, b): p[a] * p[b] / (1 - p[a]) + p[b] * p[a] / (1 - p[b])
        for a, b in itertools.combinations(sorted(p), 2)
    }


def ref_top2(pairs: dict[tuple[int, int], float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for (a, b), v in pairs.items():
        out[a] = out.get(a, 0.0) + v
        out[b] = out.get(b, 0.0) + v
    return out


def ref_x_cross(win: dict[int, float], quinella: dict[tuple[int, int], float]) -> dict[int, float]:
    """x_cross = ln(q_top2 / h_top2) per runner from the boards at t."""
    h_top2 = ref_top2(ref_harville(ref_shares(win)))
    q_top2 = ref_top2(ref_shares(quinella))
    return {h: float(np.log(q_top2[h] / h_top2[h])) for h in win}


def ref_race_ll(frame: pd.DataFrame) -> pd.DataFrame:
    """Per race with a winner: the mean over its winners of -ln p_offset and -ln p_model."""
    won = frame[frame["won"].astype(bool)]
    parts = won.assign(ll_offset=-np.log(won["p_offset"]), ll_model=-np.log(won["p_model"]))
    return parts.groupby("race_id")[["ll_offset", "ll_model"]].mean()


def expected_fold(rows: pd.DataFrame, fold, variant: str) -> tuple[dict, pd.DataFrame]:
    """The variant's fit on the fold's training rows and its test rows' p_offset and p_model."""
    features, offset = variant.split(":")
    cols = FEATURES_OF[features]
    train, test = rows.iloc[fold.train_idx], rows.iloc[fold.test_idx]
    if offset == "normalised":
        off_train, off_test = ref_ln_q(train), ref_ln_q(test)
    else:
        cal = fit_win_calibration(train)
        off_train = np.log(cal.predict_many(np.exp(ref_ln_q(train)), train["race_id"].to_numpy()))
        off_test = np.log(cal.predict_many(np.exp(ref_ln_q(test)), test["race_id"].to_numpy()))
    fit = condlogit.fit_conditional_logit(
        train[cols].to_numpy(dtype=float),
        train["race_id"].to_numpy(),
        train["won"].to_numpy(dtype=bool),
        offset=off_train,
        cluster=train["meeting_id"].to_numpy(),
    )
    race = test["race_id"].to_numpy()
    frame = test[["race_id", "horse_no", "won"]].assign(
        p_offset=ref_softmax(off_test, race),
        p_model=ref_softmax(off_test + test[cols].to_numpy(dtype=float) @ fit.coef, race),
    )
    return dict(zip(cols, fit.coef, strict=True)), frame


# ---------------------------------------------------------------- in-memory study rows

STUDY_FROM, STUDY_TO = date(2021, 1, 2), date(2021, 8, 29)
CLASS_CYCLE = ("newcomer", "maiden", "conditions", "open", "graded", None)


def days_between(first: date, last: date, *, weekends: bool = True) -> list[date]:
    out, d = [], first
    while d <= last:
        if not weekends or d.weekday() >= 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def study_rows(seed: int = 7, beta: float = 0.25) -> pd.DataFrame:
    """Score_folds' input: one row per Win starter of three races per weekend day, the winner
    drawn from ln q + beta * x_cross; one dead heat in May and one void race in June."""
    rng = np.random.default_rng(seed)
    races = []
    n = 0
    for day in days_between(STUDY_FROM, STUDY_TO):
        for venue, race_no in (("TOK", 1), ("TOK", 2), ("KYO", 1)):
            rid = f"jra:{day.isoformat()}:{venue}:{race_no:02d}"
            size = int(rng.integers(5, 11))
            strength = rng.gamma(2.0, size=size)
            p = strength / strength.sum()
            odds = np.maximum(np.round(0.8 / p, 1), 1.1)
            x = rng.normal(0.0, 0.4, size=size)
            fav = int(np.argmin(odds))
            won = np.arange(size) == np.argmax(np.log(p) + beta * x + rng.gumbel(size=size))
            races.append(
                pd.DataFrame(
                    {
                        "race_id": rid,
                        "race_date": day,
                        "meeting_id": f"jra:{day.isoformat()}:{venue}",
                        "class_group": CLASS_CYCLE[n % len(CLASS_CYCLE)],
                        "horse_no": np.arange(1, size + 1),
                        "won": won,
                        "win_odds": odds,
                        "x_cross": x,
                        "fav_x_cross": np.where(np.arange(size) == fav, x, 0.0),
                    }
                )
            )
            n += 1
    order = rng.permutation(len(races))  # races in no particular order; positions still index
    frame = pd.concat([races[i] for i in order], ignore_index=True)
    may = frame.loc[pd.to_datetime(frame["race_date"]).dt.month == 5, "race_id"].iloc[0]
    june = frame.loc[pd.to_datetime(frame["race_date"]).dt.month == 6, "race_id"].iloc[0]
    second = frame.index[(frame["race_id"] == may) & ~frame["won"]][0]
    frame.loc[second, "won"] = True  # a dead heat for first
    frame.loc[frame["race_id"] == june, "won"] = False  # a void race
    return frame


def scored_race_ids(rows: pd.DataFrame, fold) -> set[str]:
    test = rows.iloc[fold.test_idx]
    return set(test.loc[test["won"].astype(bool), "race_id"])


# ---------------------------------------------------------------- race_probs


@pytest.mark.parametrize("k", [1, 2, 3])
def test_zero_coefficients_reproduce_the_offset(k):
    rng = np.random.default_rng(100 + k)
    sizes = (2, 5, 9, 16, 3, 7)
    shifts = (0.0, -5.0, 3.5, 720.0, -730.0, 41.0)  # per-race constants, far past exp's range too
    race = np.concatenate([[f"r{i}"] * n for i, n in enumerate(sizes)]).astype(object)
    q = np.concatenate([(s := rng.gamma(1.5, size=n)) / s.sum() for n in sizes])
    shift = np.concatenate([[c] * n for c, n in zip(shifts, sizes, strict=True)])
    perm = rng.permutation(len(race))  # rows interleaved, not grouped by race
    race, q, shift = race[perm], q[perm], shift[perm]
    X = rng.normal(0.0, 1.0, size=(len(race), k))
    offset = np.log(q) + shift

    got = np.asarray(race_probs(offset, X, np.zeros(k), race), dtype=float)
    np.testing.assert_allclose(got, q, rtol=1e-12, atol=0)

    coef = rng.normal(0.0, 0.7, size=k)
    want = ref_softmax(offset + X @ coef, race)
    got = np.asarray(race_probs(offset, X, coef, race), dtype=float)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-300)
    assert np.abs(want - q).max() > 1e-3  # the coefficient moves the probabilities
    sums = pd.Series(got).groupby(race).sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0, rtol=1e-12)


# ---------------------------------------------------------------- day_rows

DAY = date(2025, 6, 1)


def ts_row(rid, pool, combination, minutes, odds, post, *, investment=None, status=None):
    _, _, venue, no = rid.split(":")
    return {
        "race_id": rid,
        "venue": venue,
        "race_no": int(no),
        "pool": pool,
        "combination": combination,
        "snapshot_utc": post - timedelta(minutes=minutes),
        "fetched_utc": None,  # the JRA-VAN export has no fetch time
        "post_utc": post,
        "minutes_to_post": float(minutes),
        "odds": None if odds is None else float(odds),
        "odds_hi": None,
        "investment": investment,
        "status_raw": status,
        "is_final_capture": False,
        "dividend": None,
        "is_refund": None,
    }


def capture(rid, post, minutes, *, win=None, quinella=None) -> list[dict]:
    rows = []
    if win is not None:
        rows += [ts_row(rid, "WIN", str(h), minutes, o, post) for h, o in win.items()]
        rows.append(ts_row(rid, "WIN", POOL_ROW, minutes, None, post, investment=1.0e7))
    if quinella is not None:
        rows += [
            ts_row(rid, "QUINELLA", f"{a},{b}", minutes, o, post) for (a, b), o in quinella.items()
        ]
        rows.append(ts_row(rid, "QUINELLA", POOL_ROW, minutes, None, post, investment=2.0e7))
    return rows


def day_frame(rows: list[dict], *, seed: int = 3) -> pd.DataFrame:
    frame = pa.Table.from_pylist(rows, schema=SCHEMA).to_pandas()
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def tilted_boards(p: dict[int, float], tilt: dict[int, float]):
    """Win odds whose shares are p, and Quinella odds whose shares are p's Harville pairs with
    each runner's pairs scaled by its tilt."""
    win = {h: 0.80 / v for h, v in p.items()}
    pairs = {pr: v * tilt[pr[0]] * tilt[pr[1]] for pr, v in ref_harville(p).items()}
    z = sum(pairs.values())
    return win, {pr: 0.775 * z / v for pr, v in pairs.items()}


def normalise(values, horses) -> dict[int, float]:
    total = sum(values)
    return {h: v / total for h, v in zip(horses, values, strict=True)}


def test_day_rows_build_the_t2_features_from_the_board_at_t():
    assert cross_pool_run.MINUTES_BEFORE == load_country_yaml("japan", "decision")["primary_min"]
    a, b = f"jra:{DAY.isoformat()}:TOK:11", f"jra:{DAY.isoformat()}:KYO:10"
    post_a = datetime(2025, 6, 1, 6, 40, tzinfo=UTC)
    post_b = datetime(2025, 6, 1, 6, 15, tzinfo=UTC)
    horses_a, horses_b = (1, 2, 3, 4, 5, 6), (2, 4, 5, 7, 9)
    boards = {
        # (race, minutes) -> (win, quinella); the T-2 board is A's 2.0 / 2.5 and B's 3.0 / 2.2
        ("a", 6.0): tilted_boards(
            normalise([0.3, 0.25, 0.15, 0.12, 0.1, 0.08], horses_a), dict.fromkeys(horses_a, 1.0)
        ),
        ("a", 2.0): tilted_boards(
            normalise([0.32, 0.22, 0.17, 0.13, 0.1, 0.06], horses_a),
            {1: 0.9, 2: 1.0, 3: 1.35, 4: 1.0, 5: 0.95, 6: 1.1},
        ),
        ("b", 7.0): tilted_boards(
            normalise([0.4, 0.2, 0.2, 0.12, 0.08], horses_b),
            {2: 1.2, 4: 1.0, 5: 1.0, 7: 0.8, 9: 1.0},
        ),
        ("b", 3.0): tilted_boards(
            normalise([0.45, 0.2, 0.15, 0.12, 0.08], horses_b),
            {2: 1.25, 4: 0.9, 5: 1.0, 7: 1.0, 9: 1.15},
        ),
    }

    def build(after: float) -> pd.DataFrame:
        late_a = {h: o * after for h, o in boards[("a", 2.0)][0].items()}
        late_qa = {pr: o * (after + 0.1 * pr[0]) for pr, o in boards[("a", 2.0)][1].items()}
        late_b = {h: o / after for h, o in boards[("b", 3.0)][0].items()}
        rows = capture(a, post_a, 6.0, win=boards[("a", 6.0)][0])
        rows += capture(a, post_a, 5.0, quinella=boards[("a", 6.0)][1])
        rows += capture(a, post_a, 2.0, win=boards[("a", 2.0)][0])
        rows += capture(a, post_a, 2.5, quinella=boards[("a", 2.0)][1])
        rows += capture(a, post_a, 1.0, win=late_a)  # after t
        rows += capture(a, post_a, 0.5, quinella=late_qa)  # after t
        rows += capture(b, post_b, 7.0, win=boards[("b", 7.0)][0])
        rows += capture(b, post_b, 6.5, quinella=boards[("b", 7.0)][1])
        rows += capture(b, post_b, 3.0, win=boards[("b", 3.0)][0])
        rows += capture(b, post_b, 2.2, quinella=boards[("b", 3.0)][1])
        rows += capture(b, post_b, 1.5, win=late_b)  # after t
        rows += [ts_row(a, "PLACE", str(h), 2.0, 1.1 + 0.2 * h, post_a) for h in horses_a]
        return day_frame(rows)

    def expected(at: dict[str, float]) -> dict[tuple[str, int], dict[str, float]]:
        out = {}
        for rid, key in ((a, "a"), (b, "b")):
            win, quinella = boards[(key, at[key])]
            p = ref_shares(win)
            x = ref_x_cross(win, quinella)
            fav = max(p, key=p.get)
            for h in win:
                out[(rid, h)] = {
                    "p_win_t": p[h],
                    "x_cross": x[h],
                    "is_fav": float(h == fav),
                    "fav_x_cross": x[h] * (h == fav),
                }
        return out

    def by_runner(frame: pd.DataFrame) -> dict[tuple[str, int], dict[str, float]]:
        assert {"race_id", "horse_no", "p_win_t", "x_cross", "is_fav", "fav_x_cross"} <= set(
            frame.columns
        )
        keys = list(zip(frame["race_id"], frame["horse_no"].astype(int), strict=True))
        assert len(keys) == len(set(keys)), "one row per Win starter"
        return {
            k: {c: float(frame[c].iloc[i]) for c in ("p_win_t", "x_cross", "is_fav", "fav_x_cross")}
            for i, k in enumerate(keys)
        }

    def assert_matches(got, want):
        assert set(got) == set(want)
        for key, values in want.items():
            for column, value in values.items():
                assert got[key][column] == pytest.approx(value, abs=1e-9), (key, column)

    day = build(after=1.7)
    at_t2 = expected({"a": 2.0, "b": 3.0})
    assert_matches(by_runner(day_rows(day)), at_t2)
    assert_matches(by_runner(day_rows(day, minutes_before=2.0)), at_t2)
    assert max(abs(v["x_cross"]) for v in at_t2.values()) > 0.05  # the boards disagree

    # Rows after t never move a feature.
    assert_matches(by_runner(day_rows(build(after=0.4))), at_t2)
    # Another horizon reads another board (A's 6.0 / 5.0 and B's 7.0 / 6.5 at T-5).
    at_t5 = expected({"a": 6.0, "b": 7.0})
    assert_matches(by_runner(day_rows(day, minutes_before=5.0)), at_t5)
    assert any(abs(at_t5[k]["x_cross"] - at_t2[k]["x_cross"]) > 1e-3 for k in at_t2)


# ---------------------------------------------------------------- fit_fold: a planted signal


def planted_races(seed: int, n_races: int, beta: dict[str, float]) -> pd.DataFrame:
    """Races whose winners follow ln q_final + beta . features (Gumbel-max draws)."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(6, 15, size=n_races)
    race = np.repeat(np.arange(n_races), sizes)
    strength = rng.gamma(1.5, size=len(race))
    q = strength / np.bincount(race, strength)[race]
    x = rng.normal(0.0, 0.5, size=len(race))
    fav = (q == pd.Series(q).groupby(race).transform("max").to_numpy()).astype(float)
    columns = {"x_cross": x, "fav_x_cross": fav * x}
    u = np.log(q) + sum(b * columns[f] for f, b in beta.items())
    g = u + rng.gumbel(size=len(race))
    won = g == pd.Series(g).groupby(race).transform("max").to_numpy()
    return pd.DataFrame(
        {
            "race_id": [f"r{r:05d}" for r in race],
            "meeting_id": [f"m{r // 12:04d}" for r in race],
            "horse_no": np.concatenate([np.arange(1, n + 1) for n in sizes]),
            "won": won,
            "offset": np.log(q),
            **columns,
        }
    )


@pytest.mark.parametrize(
    "beta, seed",
    [
        ({"x_cross": 0.3}, 4351),  # the ticket's plant
        ({"x_cross": 0.3, "fav_x_cross": -0.6}, 4352),
        ({"x_cross": -0.25}, 4353),
    ],
)
def test_a_planted_signal_is_recovered(beta, seed):
    train = planted_races(seed, 6000, beta)
    features = tuple(beta)

    fit = fit_fold(train, features=features)

    coef = np.asarray(fit.coef, dtype=float)
    assert fit.se_cluster is not None, "the fit clusters on the meeting"
    se = np.asarray(fit.se_cluster, dtype=float)
    assert coef.shape == se.shape == (len(features),)
    assert fit.n_races == 6000
    for name, c, s in zip(features, coef, se, strict=True):
        assert abs(c - beta[name]) < 3 * s, (name, c, s)
        assert abs(c) > 3 * s, (name, c, s)  # detected, not merely bracketed


# ---------------------------------------------------------------- score_folds and predict_folds


def test_score_folds_score_every_test_race_against_its_offset():
    assert set(VARIANTS) == EXPECTED_VARIANTS
    rows = study_rows()
    folds = walk_forward(rows["race_date"], scheme=SHORT)
    assert len(folds) >= 4

    scored = score_folds(rows, folds)
    predicted = predict_folds(rows, folds)

    need = {"variant", "fold", "race_id", "race_date", "meeting_id", "class_group"}
    need |= {"ll_offset", "ll_model", "delta_ll", "coef_x_cross", "coef_fav_x_cross"}
    assert need <= set(scored.columns)
    assert {"variant", "fold", "race_id", "horse_no", "p_offset", "p_model"} <= set(
        predicted.columns
    )
    assert set(scored["variant"]) == set(predicted["variant"]) == EXPECTED_VARIANTS
    meta = rows.drop_duplicates("race_id").set_index("race_id")
    want_races = set().union(*(scored_race_ids(rows, f) for f in folds))

    for variant in sorted(EXPECTED_VARIANTS):
        mine = scored[scored["variant"] == variant].set_index("race_id")
        assert mine.index.is_unique, "one row per (variant, scored race)"
        assert set(mine.index) == want_races  # the void race is not scored
        guess = predicted[predicted["variant"] == variant].set_index(["race_id", "horse_no"])
        assert guess.index.is_unique
        for k, fold in enumerate(folds):
            coef, frame = expected_fold(rows, fold, variant)
            races = scored_race_ids(rows, fold)
            got = mine.loc[sorted(races)]
            assert (got["fold"] == k).all()
            for name in ("x_cross", "fav_x_cross"):
                column = got[f"coef_{name}"].to_numpy(dtype=float)
                if name in coef:
                    np.testing.assert_allclose(column, coef[name], atol=1e-6)
                else:
                    assert np.isnan(column).all()
            ll = ref_race_ll(frame).loc[sorted(races)]
            np.testing.assert_allclose(got["ll_offset"], ll["ll_offset"], atol=1e-6)
            np.testing.assert_allclose(got["ll_model"], ll["ll_model"], atol=1e-6)
            np.testing.assert_allclose(
                got["delta_ll"], got["ll_offset"] - got["ll_model"], atol=1e-12
            )
            for rid in races:
                assert pd.Timestamp(got.loc[rid, "race_date"]).date() == meta.loc[rid, "race_date"]
                assert got.loc[rid, "meeting_id"] == meta.loc[rid, "meeting_id"]
                cls = meta.loc[rid, "class_group"]
                if cls is None:  # an unmapped class stays unmapped
                    assert pd.isna(got.loc[rid, "class_group"]), rid
                else:
                    assert got.loc[rid, "class_group"] == cls, rid
            # Out-of-fold probabilities, runner by runner.
            runners = frame[frame["race_id"].isin(races)]
            keys = list(zip(runners["race_id"], runners["horse_no"], strict=True))
            probs = guess.loc[keys]
            assert (probs["fold"] == k).all()
            np.testing.assert_allclose(probs["p_offset"], runners["p_offset"], atol=1e-6)
            np.testing.assert_allclose(probs["p_model"], runners["p_model"], atol=1e-6)
            if variant.endswith(":normalised"):  # the normalised offset is q itself
                test = rows.iloc[fold.test_idx]
                keyed = pd.MultiIndex.from_frame(test[["race_id", "horse_no"]])
                q = pd.Series(np.exp(ref_ln_q(test)), index=keyed).loc[keys]
                np.testing.assert_allclose(probs["p_offset"], q, rtol=1e-12)
    # The fixture holds one dead heat (its two winners' -ln p averaged by ref_race_ll above).
    assert (rows.groupby("race_id")["won"].sum() == 2).sum() == 1
    # The feature moves the model off its offset: the ΔLL is not identically zero.
    assert scored["delta_ll"].abs().max() > 1e-4


def call_race_ids(args, kwargs) -> frozenset:
    race_id = kwargs["race_id"] if "race_id" in kwargs else args[1]
    return frozenset(np.asarray(race_id, dtype=object).tolist())


def test_folds_fit_on_earlier_days_only(monkeypatch):
    rows = study_rows(seed=11)
    folds = walk_forward(rows["race_date"], scheme=SHORT)
    date_of = dict(zip(rows["race_id"], rows["race_date"], strict=False))
    has_winner = rows.groupby("race_id")["won"].any()
    void = frozenset(has_winner.index[~has_winner])
    train_sets = {
        k: frozenset(rows.iloc[f.train_idx]["race_id"]) - void for k, f in enumerate(folds)
    }
    calls: list[frozenset] = []
    real = condlogit.fit_conditional_logit

    def spy(*args, **kwargs):
        calls.append(call_race_ids(args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(cross_pool_run, "fit_conditional_logit", spy)
    score_folds(rows, folds)

    assert calls, "score_folds fits through the module's fit_conditional_logit"
    fitted = set()
    for races in calls:
        matches = [k for k, s in train_sets.items() if races - void == s]
        assert matches, "a fit saw rows outside every fold's training days"
        k = matches[0]
        fitted.add(k)
        last = max(date_of[r] for r in races)
        assert last <= folds[k].train_end
        # D9's embargo, counted back from the fold's first scored day (core.eval.folds)
        first_scored = min(rows.iloc[folds[k].test_idx]["race_date"])
        assert (first_scored - last).days >= SHORT.embargo_days
        assert not races & frozenset(rows.iloc[folds[k].test_idx]["race_id"])
    assert fitted == set(train_sets), "every fold is fitted"

    # Nothing dated after a fold's train_end moves its out-of-fold probabilities: re-draw every
    # later winner and move the features and prices of every later row outside its test window.
    monkeypatch.setattr(cross_pool_run, "fit_conditional_logit", real)
    k = 2
    fold = folds[k]
    base = predict_folds(rows, folds)
    rng = np.random.default_rng(5)
    moved = rows.copy()
    later = (pd.to_datetime(moved["race_date"]) > pd.Timestamp(fold.train_end)).to_numpy()
    in_test = np.zeros(len(rows), dtype=bool)
    in_test[fold.test_idx] = True
    moved.loc[later, "won"] = False
    for _, idx in moved[later].groupby("race_id").groups.items():
        moved.loc[rng.choice(np.asarray(idx)), "won"] = True
    shift = later & ~in_test
    moved.loc[shift, "x_cross"] = -2.0 * moved.loc[shift, "x_cross"] + 0.3
    moved.loc[shift, "win_odds"] = np.round(
        moved.loc[shift, "win_odds"] * rng.uniform(0.6, 1.6, size=shift.sum()), 1
    ).clip(lower=1.1)
    fav = moved.groupby("race_id")["win_odds"].transform("min") == moved["win_odds"]
    moved.loc[shift, "fav_x_cross"] = np.where(fav[shift], moved.loc[shift, "x_cross"], 0.0)
    again = predict_folds(moved, folds)

    def fold_probs(frame, which):
        races = set(rows.iloc[folds[which].test_idx]["race_id"])
        part = frame[frame["race_id"].isin(races)]
        return part.set_index(["variant", "race_id", "horse_no"])[["p_offset", "p_model"]]

    before, after = fold_probs(base, k), fold_probs(again, k)
    after = after.loc[before.index]
    np.testing.assert_allclose(after.to_numpy(), before.to_numpy(), atol=1e-10)
    # The change is real: the next fold, which trains on those days, moves.
    nxt_before, nxt_after = fold_probs(base, k + 1), fold_probs(again, k + 1)
    nxt_after = nxt_after.loc[nxt_before.index]
    assert np.abs(nxt_after["p_model"] - nxt_before["p_model"]).max() > 1e-6
    calibrated = nxt_before.index.get_level_values("variant").str.endswith(":calibrated")
    assert calibrated.any()
    cal_before = nxt_before["p_offset"].to_numpy()[calibrated]
    cal_after = nxt_after["p_offset"].to_numpy()[calibrated]
    assert np.abs(cal_after - cal_before).max() > 1e-6  # the spline is refitted on training rows


# ---------------------------------------------------------------- settlement


def test_settlement_pays_final_dividends():
    paid = pd.DataFrame(
        [
            ("R1", "WIN", "7", 1230.0),
            ("R1", "PLACE", "3", 150.0),  # another pool, same text as a WIN bet
            ("R1", "QUINELLA", "3,7", 4560.0),
            ("R1", "FORECAST", "7,3", 9870.0),
            ("R2", "WIN", "2", 340.0),  # a dead heat for first: both pay
            ("R2", "WIN", "5", 410.0),
            ("R2", "QUINELLA", "2,5", 880.0),
            ("R2", "FORECAST", "2,5", 1500.0),
            ("R2", "FORECAST", "5,2", 1720.0),
            ("R3", "WIN", "3", 2210.0),  # another race, same text as R1's unpaid bet
            ("R3", "QUINELLA", "1,3", 3330.0),
            ("R3", "FORECAST", "3,1", 6130.0),
        ],
        columns=["race_id", "pool", "combination", "dividend"],
    ).assign(jurisdiction="jra", is_refund=False)
    cases = {  # pool -> [(race_id, combination, board odds, expected return)]
        "WIN": [
            ("R1", "7", 11.0, 12.3),
            ("R1", "3", 4.5, 0.0),
            ("R2", "2", 2.9, 3.4),
            ("R2", "5", 3.6, 4.1),
            ("R2", "9", 15.0, 0.0),
            ("R3", "3", 19.5, 22.1),
            ("R4", "1", 2.0, 0.0),  # a race with no dividend rows
        ],
        "QUINELLA": [
            ("R1", "3,7", 41.0, 45.6),
            ("R1", "1,3", 12.0, 0.0),
            ("R2", "2,5", 7.7, 8.8),
            ("R3", "1,3", 30.1, 33.3),
            ("R3", "3,7", 30.1, 0.0),
        ],
        "FORECAST": [
            ("R1", "7,3", 90.0, 98.7),
            ("R1", "3,7", 95.0, 0.0),  # the other order did not pay
            ("R2", "2,5", 14.0, 15.0),
            ("R2", "5,2", 16.0, 17.2),
            ("R3", "3,1", 55.0, 61.3),
            ("R3", "1,3", 58.0, 0.0),
        ],
    }
    for pool, bets in cases.items():
        frame = pd.DataFrame(bets, columns=["race_id", "combination", "odds", "want"])
        got = settle(frame[["race_id", "combination", "odds"]], paid, pool=pool)
        assert isinstance(got, pd.Series)
        assert len(got) == len(frame)
        np.testing.assert_allclose(got.to_numpy(dtype=float), frame["want"], atol=1e-12)
        priced = (frame["want"] > 0).to_numpy()
        board = frame["odds"].to_numpy(dtype=float)
        assert not np.isclose(got.to_numpy(dtype=float)[priced], board[priced]).any()


# ---------------------------------------------------------------- the spec and its freeze


def temp_registry(root: Path) -> Registry:
    return Registry(root / "registry" / "japan.jsonl", holdout_log=root / "holdout_log.jsonl")


def spec_hash(spec: dict) -> str:
    text = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_freeze_spec_pre_registers_the_family(tmp_path, monkeypatch):
    monkeypatch.setattr(cross_pool_run, "load_country_scheme", lambda country: SHORT)
    assert FAMILY_ID == "japan-cross-pool-t2-v1"
    args = {"date_from": SPAN_START, "date_to": date(2026, 10, 31), "seed": SEED}
    spec = build_spec(**args)
    text = json.dumps(spec, sort_keys=True, allow_nan=False)  # plain JSON
    assert build_spec(**args) == spec
    assert FAMILY_ID in text
    assert all(v in text for v in EXPECTED_VARIANTS)
    changes = ({"seed": SEED + 1}, {"date_to": date(2026, 9, 30)}, {"date_from": date(2015, 1, 1)})
    for change in changes:
        assert spec_hash(build_spec(**{**args, **change})) != spec_hash(spec), change

    registry = temp_registry(tmp_path / "reg")
    frozen = freeze_spec(**args, registry=registry)
    assert frozen == spec_hash(spec) == registry.frozen_spec_hash(FAMILY_ID)
    assert freeze_spec(**args, registry=registry) == frozen  # the same spec again is a no-op

    data = tmp_path / "data"
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(data))
    assert freeze_spec(**args) == frozen  # registry None: the data dir's japan registry
    assert temp_registry(data).frozen_spec_hash(FAMILY_ID) == frozen


# ---------------------------------------------------------------- a synthetic jravan store

STORE_FROM, STORE_TO = date(2026, 9, 1), date(2027, 1, 31)
HOLDOUT_DAY = date(2026, 11, 29)
EARLY_DAY = date(2014, 6, 1)  # before D10's price era
VENUES = ("TOK", "KYO")
RACES_PER_VENUE = 2
RACE_CLASSES = ("DEBUT", "MAIDEN", "C1", "OP", "G1", None, "C2", "L")
GARBAGE = b"not a parquet file: a holdout day's file is never opened\n"


class ForbiddenRead(BaseException):
    """A file the run must not open was opened. A BaseException, so `except Exception` cannot
    hide it."""


def harville_ordered(p: np.ndarray) -> np.ndarray:
    """P(i first, j second) = p_i p_j / (1 - p_i)."""
    out = np.outer(p, p) / (1.0 - p)[:, None]
    np.fill_diagonal(out, 0.0)
    return out


def synth_race(rng, day: date, venue: str, race_no: int, race_class) -> dict[str, list[dict]]:
    rid = f"jra:{day.isoformat()}:{venue}:{race_no:02d}"
    meeting = f"jra:{day.isoformat()}:{venue}"
    n = int(rng.integers(6, 10))
    horses = list(range(1, n + 1))
    p = rng.gamma(2.0, size=n)
    p /= p.sum()
    final_win = np.maximum(np.round(0.8 / p, 1), 1.1)
    p_t = p * np.exp(rng.normal(0.0, 0.1, size=n))
    p_t /= p_t.sum()
    win_t = np.maximum(np.round(0.8 / p_t, 1), 1.1)
    tilt = np.exp(rng.normal(0.0, 0.15, size=n))
    ordered_t = harville_ordered(p_t)
    pairs = list(itertools.combinations(range(n), 2))
    money = {(i, j): (ordered_t[i, j] + ordered_t[j, i]) * tilt[i] * tilt[j] for i, j in pairs}
    z = sum(money.values())
    quin_t = {(i + 1, j + 1): max(round(0.775 * z / v, 1), 1.1) for (i, j), v in money.items()}
    ordered = harville_ordered(p)
    final_quin = {(i + 1, j + 1): max(round(0.775 / (ordered[i, j] + ordered[j, i]), 1), 1.1)
                  for i, j in pairs}  # fmt: skip
    final_fc = {(i + 1, j + 1): max(round(0.75 / ordered[i, j], 1), 1.1)
                for i in range(n) for j in range(n) if i != j}  # fmt: skip
    order = np.argsort(-(np.log(p) + rng.gumbel(size=n)))
    finish = np.empty(n, dtype=int)
    finish[order] = np.arange(1, n + 1)
    first, second = horses[order[0]], horses[order[1]]

    post = datetime(day.year, day.month, day.day, 1, 0, tzinfo=UTC) + timedelta(
        minutes=40 * race_no + (0 if venue == VENUES[0] else 15)
    )

    def jitter(odds: dict, low: float, high: float) -> dict:
        return {k: max(round(v * rng.uniform(low, high), 1), 1.1) for k, v in odds.items()}

    win_board = dict(zip(horses, win_t.tolist(), strict=True))
    ts = capture(rid, post, 6.0, win=jitter(win_board, 0.9, 1.1))
    ts += capture(rid, post, 5.0, quinella=jitter(quin_t, 0.9, 1.1))
    ts += capture(rid, post, 2.0, win=win_board)
    ts += capture(rid, post, 2.5, quinella=quin_t)
    ts += capture(rid, post, 1.0, win=jitter(win_board, 0.5, 1.5))  # after t
    ts += capture(rid, post, 0.5, quinella=jitter(quin_t, 0.5, 1.5))  # after t

    base = {"jurisdiction": "jra", "race_id": rid, "race_date": day, "venue": venue,
            "race_no": race_no}  # fmt: skip

    def odds_row(pool, combination, odds, investment=None):
        return {**base, "pool": pool, "combination": combination, "odds": odds,
                "investment": investment, "published_utc": post, "data_kubun": "4"}  # fmt: skip

    final = [odds_row("WIN", str(h), float(o)) for h, o in zip(horses, final_win, strict=True)]
    final += [odds_row("PLACE", str(h), max(round(0.8 / (3 * v), 1), 1.0)) for h, v in
              zip(horses, p, strict=True)]  # fmt: skip
    final += [odds_row("QUINELLA", f"{a},{b}", o) for (a, b), o in final_quin.items()]
    final += [odds_row("FORECAST", f"{a},{b}", o) for (a, b), o in final_fc.items()]
    final += [odds_row(pool, POOL_ROW, None, 1.0e8) for pool in ("WIN", "PLACE", *POOLS[1:])]

    lo, hi = sorted((first, second))
    dividends = [
        ("WIN", str(first), final_win[first - 1] * 100 + 30),
        ("QUINELLA", f"{lo},{hi}", final_quin[(lo, hi)] * 100 + 40),
        ("FORECAST", f"{first},{second}", final_fc[(first, second)] * 100 + 50),
        *[("PLACE", str(horses[order[i]]), 150.0 + 10 * i) for i in range(3)],
    ]
    paid = [{"jurisdiction": "jra", "race_id": rid, "pool": pool, "combination": c,
             "dividend": float(round(d)), "is_refund": False}
            for pool, c, d in dividends]  # fmt: skip

    jst = post + timedelta(hours=9)
    race = {**base, "meeting_id": meeting, "season": str(day.year), "race_name": "synthetic",
            "race_class": race_class, "distance_m": 1600, "surface": "turf",
            "post_time_local": f"{jst:%H:%M}", "field_size": n, "status": "run"}  # fmt: skip
    runners = [{**base, "meeting_id": meeting, "season": str(day.year), "horse_id": f"{rid}:{h}",
                "horse_no": h, "horse_name": f"horse {h}", "placing_raw": str(int(pos)),
                "finish_pos": int(pos), "dead_heat": False, "win_odds": float(o),
                "is_scratched": False}
               for h, pos, o in zip(horses, finish, final_win, strict=True)]  # fmt: skip
    return {"ts": ts, "races": [race], "runners": runners, "final": final, "paid": paid}


def development_days() -> list[date]:
    days = [STORE_FROM, *days_between(STORE_FROM + timedelta(days=1), STORE_TO)]
    return [d for d in days if not is_holdout_day("japan", d)]


def after_day() -> date:
    d = STORE_TO + timedelta(days=1)
    while d.weekday() < 5 or is_holdout_day("japan", d):
        d += timedelta(days=1)
    return d


def build_store(data: Path) -> SimpleNamespace:
    """processed/jravan under `data`: one odds_timeseries file per day (the holdout day's is not
    parquet) and races, runners, final_odds and dividends in the store's shapes."""
    assert is_holdout_day("japan", HOLDOUT_DAY)
    rng = np.random.default_rng(2026)
    processed = data / "processed" / "jravan"
    folder = processed / "odds_timeseries"
    folder.mkdir(parents=True)
    (folder / "_build.json").write_text("{}\n")
    dev, late = development_days(), after_day()
    tables: dict[str, list[dict]] = {"races": [], "runners": [], "final": [], "paid": []}
    races_of: dict[date, list[str]] = {}
    n = 0
    for day in sorted({*dev, HOLDOUT_DAY, EARLY_DAY, late}):
        ts = []
        for venue in VENUES:
            for race_no in range(1, RACES_PER_VENUE + 1):
                race = synth_race(rng, day, venue, race_no, RACE_CLASSES[n % len(RACE_CLASSES)])
                n += 1
                ts += race["ts"]
                for name in tables:
                    tables[name] += race[name]
                races_of.setdefault(day, []).append(race["races"][0]["race_id"])
        path = folder / f"{day.isoformat()}.parquet"
        if day == HOLDOUT_DAY:
            path.write_bytes(GARBAGE)
        else:
            pq.write_table(pa.Table.from_pylist(ts, schema=SCHEMA), path)
    shapes = {
        "races": arrow_schema("races"),
        "runners": arrow_schema("runners"),
        "final_odds": FINAL_ODDS_SCHEMA,
        "dividends": arrow_schema("dividends"),
    }
    for name, key in (("races", "races"), ("runners", "runners"), ("final_odds", "final"),
                      ("dividends", "paid")):  # fmt: skip
        pq.write_table(
            pa.Table.from_pylist(tables[key], schema=shapes[name]), processed / f"{name}.parquet"
        )

    race_dates = [d for d in dev for _ in range(len(VENUES) * RACES_PER_VENUE)]
    folds = walk_forward(pd.Series(race_dates), scheme=SHORT)
    test_days = sorted({race_dates[i] for f in folds for i in f.test_idx})
    assert test_days and test_days[0] == date(2026, 11, 1)
    assert HOLDOUT_DAY > test_days[0] and any(f.train_end > HOLDOUT_DAY for f in folds)
    forbidden_days = (HOLDOUT_DAY, EARLY_DAY, late)
    return SimpleNamespace(
        data=data,
        processed=processed,
        dev_days=dev,
        test_days=test_days,
        forbidden_days=forbidden_days,
        forbidden=frozenset(
            os.path.realpath(folder / f"{d.isoformat()}.parquet") for d in forbidden_days
        ),
        races_of=races_of,
    )


# ---------------------------------------------------------------- a spy on file reads

_PATH_KEYS = ("source", "path", "where", "file", "path_or_paths", "filename")


def _arg_paths(value) -> list[Path]:
    if isinstance(value, (list, tuple)):
        return [p for v in value for p in _arg_paths(v)]
    if not isinstance(value, (str, os.PathLike)):
        return []
    text = os.fsdecode(os.fspath(value))
    if any(ch in text for ch in "*?["):
        return [Path(x) for x in glob.glob(text, recursive=True)]
    path = Path(text)
    if path.is_dir():  # a directory read reads every file under it
        return [p for p in path.rglob("*") if p.is_file()]
    return [path]


def install_file_spy(mp: pytest.MonkeyPatch, forbidden: Callable[[Path], bool], log: list) -> None:
    """Wrap the parquet readers (pyarrow, pandas, polars) and the file openers so that opening a
    forbidden path is logged and raises ForbiddenRead; every other call goes through."""

    def guard(args, kwargs) -> None:
        value = args[0] if args else next((kwargs[k] for k in _PATH_KEYS if k in kwargs), None)
        for path in _arg_paths(value):
            if forbidden(path):
                log.append(path)
                raise ForbiddenRead(str(path))

    def wrap(func):
        def spy(*args, **kwargs):
            guard(args, kwargs)
            return func(*args, **kwargs)

        return spy

    readers = [
        (pq, "read_table"),
        (pq, "read_schema"),
        (pq, "read_metadata"),
        (pq, "read_pandas"),
        (ds, "dataset"),
        (pa, "memory_map"),
        (pd, "read_parquet"),
        (pl, "read_parquet"),
        (pl, "scan_parquet"),
        (pl, "read_parquet_schema"),
        (pl, "read_parquet_metadata"),
        (builtins, "open"),
        (io, "open"),
        (os, "open"),
    ]
    for module, name in readers:
        if hasattr(module, name):
            mp.setattr(module, name, wrap(getattr(module, name)))
    for name in ("ParquetFile", "ParquetDataset"):
        original = getattr(pq, name)

        def init(self, *args, _original=original, **kwargs):
            guard(args, kwargs)
            _original.__init__(self, *args, **kwargs)

        mp.setattr(pq, name, type(name, (original,), {"__init__": init}))
    accessor = getattr(pathlib, "_NormalAccessor", None)  # Python 3.10's Path.open
    if accessor is not None and hasattr(accessor, "open"):
        mp.setattr(accessor, "open", staticmethod(wrap(accessor.open)))


def under(root: Path) -> Callable[[Path], bool]:
    real = Path(os.path.realpath(root))

    def inside(path: Path) -> bool:
        resolved = Path(os.path.realpath(path))
        return resolved == real or real in resolved.parents

    return inside


# ---------------------------------------------------------------- the run on the store


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> SimpleNamespace:
    return build_store(tmp_path_factory.mktemp("cross_pool_store") / "data")


def run_args(store: SimpleNamespace, out: str, registry: Registry, **change) -> dict:
    args = {
        "date_from": SPAN_START,
        "date_to": STORE_TO,
        "out_dir": store.data / "reports" / out,
        "seed": SEED,
        "registry": registry,
    }
    return {**args, **change}


@pytest.fixture(scope="module")
def main_run(store, tmp_path_factory) -> SimpleNamespace:
    """One run over the store, frozen first, with every read of a forbidden day file (the holdout
    day, a day before 2014-06-07, a day after date_to) raising and every fit's races recorded."""
    registry = temp_registry(tmp_path_factory.mktemp("main_registry"))
    opened: list[Path] = []
    fits: list[frozenset] = []
    real = condlogit.fit_conditional_logit
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HORSETRACK_DATA_DIR", str(store.data))
        mp.setattr(cross_pool_run, "load_country_scheme", lambda country: SHORT)

        def spy(*args, **kwargs):
            fits.append(call_race_ids(args, kwargs))
            return real(*args, **kwargs)

        mp.setattr(cross_pool_run, "fit_conditional_logit", spy)
        args = run_args(store, "main", registry)
        freeze_spec(
            date_from=args["date_from"], date_to=args["date_to"], seed=SEED, registry=registry
        )
        install_file_spy(mp, lambda p: os.path.realpath(p) in store.forbidden, opened)
        try:
            result = run_cross_pool(**args)
        except ForbiddenRead as exc:
            pytest.fail(f"the run opened a day file it must never open: {exc}")
    return SimpleNamespace(
        result=result, registry=registry, out_dir=args["out_dir"], opened=opened, fits=fits
    )


def string_cells(frame: pd.DataFrame) -> set[str]:
    return {str(v) for v in frame.to_numpy().ravel() if isinstance(v, str)}


def contains_all(cells: set[str], names) -> bool:
    return all(any(name in cell for cell in cells) for name in names)


def test_holdout_day_files_are_never_opened(store, main_run):
    result = main_run.result
    assert isinstance(result, CrossPoolResult)
    assert main_run.opened == []
    assert result.failed_variants == []

    # No fit saw a race of the holdout day, of a day before 2014-06-07 or of a day after date_to.
    forbidden_races = {r for d in store.forbidden_days for r in store.races_of[d]}
    assert main_run.fits
    assert not any(races & forbidden_races for races in main_run.fits)
    # And no result table names a race or meeting of those days.
    texts = [frame.to_csv() for frame in (result.coefficients, result.gains, result.settlement)]
    texts += [result.tail.to_csv(), json.dumps(result.cells, default=str)]
    for day in store.forbidden_days:
        assert not any(f"jra:{day.isoformat()}" in text for text in texts), day

    # The settled races are the development days of the test folds, exactly: a top1 selection
    # bets every scored race, so the largest cell of each pool covers every scored day and meeting.
    cells = json.loads(Path(result.cells_path).read_text(encoding="utf-8"))
    for pool in POOLS:
        mine = [c for c in cells if c["pool"] == pool]
        assert max(c["n_race_days"] for c in mine) == len(store.test_days), pool
        assert max(c["n_meetings"] for c in mine) == len(VENUES) * len(store.test_days), pool


def _strict(text: str):
    def refuse(constant):
        raise ValueError(f"not JSON: {constant}")

    return json.loads(text, parse_constant=refuse)


def test_run_writes_the_private_report_cells_and_registry_records(store, main_run):
    result, registry, out_dir = main_run.result, main_run.registry, main_run.out_dir

    # The family: frozen before the run, one record per pre-registered variant.
    assert result.family_id == FAMILY_ID
    assert result.spec_hash == registry.frozen_spec_hash(FAMILY_ID)
    records = registry.records(FAMILY_ID)
    assert sorted(r.variant for r in records) == sorted(EXPECTED_VARIANTS)
    assert all(r.status == "scored" for r in records)
    assert all(r.sources == ("jravan",) for r in records)
    assert all(r.hypothesis_id == "E25" and r.ticket == 435 for r in records)
    assert result.family_size == registry.family_size(FAMILY_ID) == len(EXPECTED_VARIANTS)
    lines = [json.loads(line) for line in registry.path.read_text().splitlines()]
    assert [line["kind"] for line in lines] == ["freeze"] + ["run"] * len(records)
    run_ids = [line["run_id"] for line in lines if line["kind"] == "run"]
    assert len(result.run_ids) == len(run_ids) == len(set(result.run_ids))
    assert all(any(full.startswith(short) for full in run_ids) for short in result.run_ids)

    # The private report and the cell file, under out_dir.
    assert Path(result.report_path) == out_dir / f"cross-pool-japan-{result.spec_hash[:12]}.md"
    report = Path(result.report_path).read_text(encoding="utf-8")
    assert report.strip()
    assert contains_all({report}, EXPECTED_VARIANTS)
    assert Path(result.cells_path) == out_dir / "candidate-cells" / "cross-pool.json"
    cells = _strict(Path(result.cells_path).read_text(encoding="utf-8"))
    assert isinstance(cells, list) and cells
    assert len(cells) == len(result.cells)
    assert {c["pool"] for c in cells} == set(POOLS)
    ids = [c["cell_id"] for c in cells]
    assert all(isinstance(i, str) and i for i in ids) and len(ids) == len(set(ids))
    returns = []
    for cell in cells:
        assert set(cell) == CELL_KEYS, sorted(set(cell) ^ CELL_KEYS)
        assert isinstance(cell["study"], str) and cell["study"]
        assert cell["family_id"] == FAMILY_ID
        assert cell["class_group"] is None or isinstance(cell["class_group"], str)
        assert cell["odds_band"] is None or isinstance(cell["odds_band"], str)
        assert cell["shrunk_edge"] is None
        assert 0.0 < cell["payout_rate"] < 1.0
        for key in ("n_bets", "n_race_days", "n_meetings"):
            assert isinstance(cell[key], int) and not isinstance(cell[key], bool), key
        assert 0 <= cell["n_race_days"] <= cell["n_meetings"] <= max(cell["n_bets"], 0)
        if cell["mean_return"] is not None:
            returns.append(cell["mean_return"])
            assert cell["mean_return"] >= 0.0
    assert returns
    assert all(a >= b for a, b in zip(returns, returns[1:], strict=False)), "ranked by mean_return"
    classed = {c["class_group"] for c in cells} - {None}
    assert set(CLASS_GROUPS) <= classed <= set(CLASS_GROUPS) | {"all"}  # N5 by class group

    # The tables behind the report: coefficients per variant, the gain overall and by class, the
    # N5 curve per settled pool, and counts.
    assert contains_all(string_cells(result.coefficients), EXPECTED_VARIANTS)
    assert contains_all(string_cells(result.gains), EXPECTED_VARIANTS)
    assert contains_all(string_cells(result.gains), CLASS_GROUPS)
    assert set(TAIL_COLUMNS) <= set(result.tail.columns)
    assert contains_all(string_cells(result.tail), POOLS)
    assert contains_all(string_cells(result.settlement), POOLS)
    assert result.counts
    assert all(isinstance(v, numbers.Integral) and v >= 0 for v in result.counts.values())


def test_a_failed_variant_is_recorded_and_the_others_still_run(store, tmp_path, monkeypatch):
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(store.data))
    monkeypatch.setattr(cross_pool_run, "load_country_scheme", lambda country: SHORT)
    real = condlogit.fit_conditional_logit

    def two_columns_fail(*args, **kwargs):
        X = np.asarray(args[0] if args else kwargs["X"], dtype=float)
        if X.ndim == 2 and X.shape[1] == 2:
            raise RuntimeError("planted failure of the two-feature fit")
        return real(*args, **kwargs)

    monkeypatch.setattr(cross_pool_run, "fit_conditional_logit", two_columns_fail)
    registry = temp_registry(tmp_path)
    args = run_args(store, "failed", registry)
    freeze_spec(date_from=args["date_from"], date_to=args["date_to"], seed=SEED, registry=registry)

    result = run_cross_pool(**args)

    failed = {"x_cross+fav:normalised", "x_cross+fav:calibrated"}
    assert set(result.failed_variants) == failed
    status = {r.variant: r.status for r in registry.records(FAMILY_ID)}
    assert status == {v: "failed" if v in failed else "scored" for v in EXPECTED_VARIANTS}
    assert all(r.sources == ("jravan",) for r in registry.records(FAMILY_ID))
    assert result.family_size == len(EXPECTED_VARIANTS)
    cells = _strict(Path(result.cells_path).read_text(encoding="utf-8"))
    assert {c["pool"] for c in cells} == set(POOLS)
    assert contains_all(string_cells(result.coefficients), EXPECTED_VARIANTS - failed)


def test_a_report_target_inside_the_repo_is_refused(store, tmp_path, monkeypatch):
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(store.data))
    monkeypatch.setattr(cross_pool_run, "load_country_scheme", lambda country: SHORT)
    registry = temp_registry(tmp_path)
    inside = env.repo_root() / f".tmp-cross-pool-run-{uuid.uuid4().hex[:8]}"
    args = run_args(store, "unused", registry, out_dir=inside)
    freeze_spec(date_from=args["date_from"], date_to=args["date_to"], seed=SEED, registry=registry)
    try:
        with pytest.raises((LicensedOutputInRepo, DataDirInsideRepo)):
            run_cross_pool(**args)
        written = [p for p in inside.rglob("*") if p.is_file()] if inside.exists() else []
        assert written == []
    finally:
        shutil.rmtree(inside, ignore_errors=True)


# ---------------------------------------------------------------- refusals before any store read


def test_run_refuses_an_unfrozen_family(store, tmp_path, monkeypatch):
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(store.data))
    monkeypatch.setattr(cross_pool_run, "load_country_scheme", lambda country: SHORT)
    opened: list[Path] = []

    def attempt(registry: Registry, out: str) -> None:
        with pytest.MonkeyPatch.context() as mp:
            install_file_spy(mp, under(store.processed), opened)
            run_cross_pool(**run_args(store, out, registry))

    # No freeze at all: refused, nothing frozen or recorded on its behalf.
    bare = temp_registry(tmp_path / "bare")
    with pytest.raises(FamilyNotFrozen):
        attempt(bare, "bare")
    assert opened == []
    assert bare.frozen_spec_hash(FAMILY_ID) is None
    assert bare.records(FAMILY_ID) == []

    # A freeze of another spec (another seed, another span end) is refused too.
    for name, change in (("seed", {"seed": SEED + 1}), ("to", {"date_to": date(2027, 1, 30)})):
        other = temp_registry(tmp_path / name)
        spec_args = {"date_from": SPAN_START, "date_to": STORE_TO, "seed": SEED, **change}
        frozen = freeze_spec(**spec_args, registry=other)
        with pytest.raises(FamilyNotFrozen):
            attempt(other, name)
        assert opened == [], name
        assert other.frozen_spec_hash(FAMILY_ID) == frozen
        assert other.records(FAMILY_ID) == []

    # The same arguments frozen: the check passes and the run goes on to read the store (the spy
    # stops it at its first store file).
    same = temp_registry(tmp_path / "same")
    freeze_spec(date_from=SPAN_START, date_to=STORE_TO, seed=SEED, registry=same)
    with pytest.raises(ForbiddenRead):
        attempt(same, "same")
    assert opened


def test_a_span_before_2014_06_07_is_refused_before_any_store_file_is_opened(
    store, tmp_path, monkeypatch
):
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(store.data))
    monkeypatch.setattr(cross_pool_run, "load_country_scheme", lambda country: SHORT)
    registry = temp_registry(tmp_path)
    early = date(2014, 6, 6)
    refused: tuple[type[BaseException], ...] = (ValueError,)
    try:  # frozen when it can be, so only the span can refuse the run
        freeze_spec(date_from=early, date_to=STORE_TO, seed=SEED, registry=registry)
    except ValueError:  # refusing to pre-register the span is a refusal too: no freeze can exist
        refused = (ValueError, FamilyNotFrozen)
    opened: list[Path] = []
    with pytest.MonkeyPatch.context() as mp:
        install_file_spy(mp, under(store.processed), opened)
        with pytest.raises(refused):
            run_cross_pool(**run_args(store, "early", registry, date_from=early))
    assert opened == []
    assert registry.records(FAMILY_ID) == []


# ---------------------------------------------------------------- the public report

# Five-digit integers, so a git sha or a timestamp in the page cannot hold one by chance.
FIGURES = (1.2345, 98765, -0.0731, 0.00456, 43217, 0.6789, 56789)
LEAKS = ("1.2345", "1.23", "98765", "98,765", "0.0731", "0.073", "0.00456", "43217", "43,217",
         "0.6789", "0.679", "56789", "56,789")  # fmt: skip


def test_public_report_carries_no_figures(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(data))
    a, b, c, d, e, f, g = FIGURES
    digest = "abcdef" * 10 + "abcd"  # hex without digits: no figure can hide in it
    run_ids = [s * 10 + s[:4] for s in ("fedcba", "bcadef", "cabfed", "dabcef")]
    variants = sorted(EXPECTED_VARIANTS)
    coefficients = pd.DataFrame(
        [{"variant": v, "feature": "x_cross", "fold": "pooled", "coef": a, "se_cluster": -c,
          "lower": c, "upper": f, "races": b, "meetings": e} for v in variants]
    )  # fmt: skip
    gains = pd.DataFrame(
        [{"variant": v, "class_group": cls, "delta_ll_mean": d, "lo": c, "hi": a, "races": b,
          "meetings": e} for v in variants for cls in ("all", "maiden")]
    )  # fmt: skip
    selections = [(v, pool, sel) for v in variants for pool in POOLS for sel in ("value", "top1")]
    settlement = pd.DataFrame(
        [{"variant": v, "pool": pool, "selection": sel, "n_bets": b, "mean_return": a, "lo": f,
          "hi": a, "n_meetings": e} for v, pool, sel in selections]
    )  # fmt: skip
    tail = pd.DataFrame(
        [{"variant": v, "pool": pool, "segment": "all", "band": 0, "q_lo": 0.0, "q_hi": 0.5,
          "claim_lo": f, "claim_hi": a, "n_bets": b, "n_clusters": e, "mean_claim": a,
          "realised": f, "lo": c, "hi": a} for v in variants for pool in POOLS]
    )  # fmt: skip
    cells = [
        {"study": "cross-pool", "family_id": FAMILY_ID, "cell_id": f"{v}:WIN:top1", "pool": "WIN",
         "class_group": None, "odds_band": None, "payout_rate": 0.8, "n_bets": b,
         "n_race_days": e, "n_meetings": g, "mean_return": a, "se_mean_return": -c,
         "shrunk_edge": None, "mean_odds": f}
        for v in variants
    ]  # fmt: skip
    result = CrossPoolResult(
        family_id=FAMILY_ID,
        spec_hash=digest,
        family_size=len(variants),
        coefficients=coefficients,
        gains=gains,
        settlement=settlement,
        tail=tail,
        cells=cells,
        counts={"races": b, "race_days": e, "meetings": g, "excluded_no_quinella_board": e},
        report_path=data / "reports" / f"cross-pool-japan-{digest[:12]}.md",
        cells_path=data / "reports" / "candidate-cells" / "cross-pool.json",
        run_ids=run_ids,
        failed_variants=[],
    )

    text = render_public_report(result)

    assert isinstance(text, str)
    for leak in LEAKS:
        assert leak not in text, leak
    assert "withheld (D21)" in text
    assert FAMILY_ID in text
    assert digest[:12] in text
    assert all(v in text for v in variants)
    assert "top1" in text
    assert all(run_id[:12] in text for run_id in run_ids)
    assert f"reports/cross-pool-japan-{digest[:12]}.md" in text  # relative to the data dir
    assert str(data) not in text and str(data.resolve()) not in text  # no home path
