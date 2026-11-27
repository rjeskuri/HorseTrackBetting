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

Fix round 1 (review of PR #493; the orchestrator's fix brief) adds:

- The pair model, the fifth variant `"pair:quinella"` (family size 5; `score_folds` keeps its Win
  output): `score_pair_folds(pairs, folds)`, `pairs` one row per starter pair with `race_id,
  race_date, meeting_id, class_group, combination` ("a,b" ascending), `won` (a pair the final
  QUINELLA dividends pay; a dead heat's paid pairs are co-winners), `quinella_odds` (the final
  Quinella odds) and `log_ratio` (ln q_pair_t / h_pair_t at T-2); the offset is ln of the
  normalised 1/quinella_odds per race, the fit `fit_fold` on each fold's training rows. One row
  per (fold, test race with a paid pair): `variant, fold, race_id, race_date, meeting_id,
  class_group, ll_offset, ll_model, delta_ll, coef_log_ratio`. In a run the pair model settles in
  QUINELLA only, with its baseline `offset:quinella`, and the races it drops are counted under
  `PAIR_COUNTS`' keys.
- A trial: `run_cross_pool(..., trial=True)` appends one record per variant to `TRIAL_FAMILY_ID`
  ("japan-cross-pool-t2-trial"), needs no freeze and never freezes or writes `FAMILY_ID`, folds
  by the module global `TRIAL_SCHEME` (a FoldScheme read at call time), and writes report and
  cell files whose names carry "trial". CLI `--trial`; `--to` is required by every mode.
- A settlement pool with nothing to settle (no FORECAST rows) is unavailable for that pool alone:
  named in `CrossPoolResult.unavailable_pools` (a collection or mapping keyed by pool), the private
  report and the Win variants' records; fits, gains and the other pools are kept, records "scored".
- `top1`'s tie rule: claims equal within a relative 1e-9 tie, a tie goes to the lowest final odds,
  then the lowest combination text (so `offset:normalised` Win top1 is the final favourite).
- Cells carry `"study": "cross-pool/final-price-diagnostic"`; the spec, the private and public
  reports say settlement, N5 and cells are retrospective diagnostics at the closing price.
- The spec (doc and `build_spec`'s `decisions`) cites the owner's reframe comment on #435; it names
  the offset baselines and how they are recorded (`offset_return[...]` metrics).
- `FamilyNotFrozen`'s message names the frozen spec's date_from, date_to and seed.

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
import re
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
import pyarrow.compute as pc
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
    main,
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
WIN_VARIANTS = {
    "x_cross:normalised",
    "x_cross:calibrated",
    "x_cross+fav:normalised",
    "x_cross+fav:calibrated",
}
PAIR_VARIANT = "pair:quinella"
EXPECTED_VARIANTS = WIN_VARIANTS | {PAIR_VARIANT}  # the family, size 5
WIN_BASELINES = ("offset:normalised", "offset:calibrated")
PAIR_BASELINE = "offset:quinella"
TRIAL_FAMILY = "japan-cross-pool-t2-trial"
DIAGNOSTIC_STUDY = "cross-pool/final-price-diagnostic"
PAIR_COUNTS = {  # a planted defect -> the count of the races the pair model drops for it
    "partial_board": "pair_excluded_incomplete_t2_quinella_board",
    "no_quinella_dividend": "pair_excluded_no_quinella_dividend",
    "unpriced_final_pair": "pair_excluded_unpriced_final_pair",
}
TIE = 1e-9  # claims equal within this relative gap tie
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


def ref_ln_q(frame: pd.DataFrame, column: str = "win_odds") -> np.ndarray:
    """ln of 1/odds normalised per race."""
    inv = 1.0 / frame[column].to_numpy(dtype=float)
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


def ref_pair_fit(train: pd.DataFrame):
    """The pair model's conditional logit on `train`: ln of the normalised 1/final Quinella odds
    as the offset, log_ratio the feature, clustered on the meeting."""
    return condlogit.fit_conditional_logit(
        train[["log_ratio"]].to_numpy(dtype=float),
        train["race_id"].to_numpy(),
        train["won"].to_numpy(dtype=bool),
        offset=ref_ln_q(train, "quinella_odds"),
        cluster=train["meeting_id"].to_numpy(),
    )


def expected_pair_fold(pairs: pd.DataFrame, fold):
    """The pair model's fit on the fold's training rows and its test rows' p_offset, p_model."""
    fit = ref_pair_fit(pairs.iloc[fold.train_idx])
    test = pairs.iloc[fold.test_idx]
    off = ref_ln_q(test, "quinella_odds")
    race = test["race_id"].to_numpy()
    frame = test[["race_id", "combination", "won"]].assign(
        p_offset=ref_softmax(off, race),
        p_model=ref_softmax(off + test["log_ratio"].to_numpy(dtype=float) * fit.coef[0], race),
    )
    return fit, frame


def ref_top1(bets: list[dict]) -> dict:
    """The race's largest claim; claims within a relative 1e-9 of it tie, and a tie goes to the
    lowest final odds, then the lowest combination text."""
    best = max(b["claim"] for b in bets)
    tied = [b for b in bets if b["claim"] >= best - TIE * abs(best)]
    return min(tied, key=lambda b: (b["odds"], b["combination"]))


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


def pair_study_rows(seed: int, gamma: float, races_per_day: int = 3) -> pd.DataFrame:
    """score_pair_folds' input: one row per starter pair of `races_per_day` races per weekend
    day, the paid pair drawn from ln q_final_pair + gamma * log_ratio, q_final_pair the
    normalised 1/final Quinella odds (Gumbel-max draws)."""
    rng = np.random.default_rng(seed)
    parts = []
    n = 0
    for day in days_between(STUDY_FROM, STUDY_TO):
        for r in range(races_per_day):
            venue = ("TOK", "KYO")[r % 2]
            rid = f"jra:{day.isoformat()}:{venue}:{r // 2 + 1:02d}"
            size = int(rng.integers(5, 9))
            pairs = list(itertools.combinations(range(1, size + 1), 2))
            strength = rng.gamma(1.5, size=len(pairs))
            odds = np.maximum(np.round(0.775 * strength.sum() / strength, 1), 1.1)
            q_final = (1.0 / odds) / (1.0 / odds).sum()
            log_ratio = rng.normal(0.0, 0.4, size=len(pairs))
            u = np.log(q_final) + gamma * log_ratio + rng.gumbel(size=len(pairs))
            parts.append(
                pd.DataFrame(
                    {
                        "race_id": rid,
                        "race_date": day,
                        "meeting_id": f"jra:{day.isoformat()}:{venue}",
                        "class_group": CLASS_CYCLE[n % len(CLASS_CYCLE)],
                        "combination": [f"{a},{b}" for a, b in pairs],
                        "won": np.arange(len(pairs)) == np.argmax(u),
                        "quinella_odds": odds,
                        "log_ratio": log_ratio,
                    }
                )
            )
            n += 1
    order = rng.permutation(len(parts))  # races in no particular order; positions still index
    return pd.concat([parts[i] for i in order], ignore_index=True)


def with_dead_heat_and_void(frame: pd.DataFrame) -> pd.DataFrame:
    """One race in May pays two pairs (a dead heat for second); one race in June pays none."""
    frame = frame.copy()
    month = pd.to_datetime(frame["race_date"]).dt.month
    may = frame.loc[month == 5, "race_id"].iloc[0]
    june = frame.loc[month == 6, "race_id"].iloc[0]
    second = frame.index[(frame["race_id"] == may) & ~frame["won"]][0]
    frame.loc[second, "won"] = True
    frame.loc[frame["race_id"] == june, "won"] = False
    return frame


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
    assert set(VARIANTS) == EXPECTED_VARIANTS  # the family; score_folds scores its Win variants
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
    assert set(scored["variant"]) == set(predicted["variant"]) == WIN_VARIANTS
    meta = rows.drop_duplicates("race_id").set_index("race_id")
    want_races = set().union(*(scored_race_ids(rows, f) for f in folds))

    for variant in sorted(WIN_VARIANTS):
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


# ---------------------------------------------------------------- the pair model: score_pair_folds

PAIR_SCORE_COLUMNS = {"variant", "fold", "race_id", "race_date", "meeting_id", "class_group"}
PAIR_SCORE_COLUMNS |= {"ll_offset", "ll_model", "delta_ll", "coef_log_ratio"}


@pytest.mark.parametrize("gamma, seed", [(0.4, 4361), (-0.3, 4362)])
def test_a_planted_pair_signal_is_recovered(gamma, seed):
    """Winners (paid pairs) follow ln q_final_pair + gamma * log_ratio over 6,000+ races."""
    pairs = pair_study_rows(seed, gamma, races_per_day=86)
    assert pairs["race_id"].nunique() >= 6000

    # The module's fit on every race, the final Quinella price as the offset.
    fit = fit_fold(pairs.assign(offset=ref_ln_q(pairs, "quinella_odds")), features=("log_ratio",))
    c, s = float(np.asarray(fit.coef)[0]), float(np.asarray(fit.se_cluster)[0])
    assert abs(c - gamma) < 3 * s, (c, s)
    assert abs(c) > 3 * s, (c, s)  # detected, not merely bracketed

    # Through the walk-forward: each fold's coefficient is the fold's own fit, and the fold with
    # the most training recovers the plant.
    folds = walk_forward(pairs["race_date"], scheme=SHORT)
    scored = cross_pool_run.score_pair_folds(pairs, folds)
    assert PAIR_SCORE_COLUMNS <= set(scored.columns)
    last = len(folds) - 1
    ref = ref_pair_fit(pairs.iloc[folds[last].train_idx])
    got = scored.loc[scored["fold"] == last, "coef_log_ratio"].to_numpy(dtype=float)
    assert len(got)
    np.testing.assert_allclose(got, ref.coef[0], atol=1e-6)
    assert abs(got[0] - gamma) < 3 * ref.se_cluster[0], (got[0], ref.se_cluster[0])


def test_score_pair_folds_score_each_race_against_the_final_quinella_price(monkeypatch):
    assert PAIR_VARIANT in set(VARIANTS)
    rows = with_dead_heat_and_void(pair_study_rows(seed=21, gamma=0.35))
    folds = walk_forward(rows["race_date"], scheme=SHORT)
    assert len(folds) >= 4

    scored = cross_pool_run.score_pair_folds(rows, folds)

    assert PAIR_SCORE_COLUMNS <= set(scored.columns)
    assert set(scored["variant"]) == {PAIR_VARIANT}
    mine = scored.set_index("race_id")
    assert mine.index.is_unique, "one row per scored race"
    want_races = set().union(*(scored_race_ids(rows, f) for f in folds))
    assert set(mine.index) == want_races  # the race that pays no pair is not scored
    meta = rows.drop_duplicates("race_id").set_index("race_id")
    for k, fold in enumerate(folds):
        fit, frame = expected_pair_fold(rows, fold)
        races = sorted(scored_race_ids(rows, fold))
        got = mine.loc[races]
        assert (got["fold"] == k).all()
        np.testing.assert_allclose(got["coef_log_ratio"], fit.coef[0], atol=1e-6)
        ll = ref_race_ll(frame).loc[races]  # a dead heat's paid pairs averaged
        np.testing.assert_allclose(got["ll_offset"], ll["ll_offset"], atol=1e-6)
        np.testing.assert_allclose(got["ll_model"], ll["ll_model"], atol=1e-6)
        np.testing.assert_allclose(got["delta_ll"], got["ll_offset"] - got["ll_model"], atol=1e-12)
        for rid in races:
            assert pd.Timestamp(got.loc[rid, "race_date"]).date() == meta.loc[rid, "race_date"]
            assert got.loc[rid, "meeting_id"] == meta.loc[rid, "meeting_id"]
            cls = meta.loc[rid, "class_group"]
            if cls is None:  # an unmapped class stays unmapped
                assert pd.isna(got.loc[rid, "class_group"]), rid
            else:
                assert got.loc[rid, "class_group"] == cls, rid
    assert (rows.groupby("race_id")["won"].sum() == 2).sum() == 1  # the dead heat is in
    assert scored["delta_ll"].abs().max() > 1e-4

    # gamma = 0 gives the offset: a zero coefficient scores every race at the Quinella price.
    def zero_fit(*args, **kwargs):
        X = np.asarray(args[0] if args else kwargs["X"], dtype=float)
        race_id = args[1] if len(args) > 1 else kwargs["race_id"]
        k = X.reshape(len(X), -1).shape[1]
        return condlogit.CLFit(
            coef=np.zeros(k),
            se=np.full(k, 0.1),
            loglik=0.0,
            n_races=int(pd.Series(np.asarray(race_id, dtype=object)).nunique()),
            se_cluster=np.full(k, 0.1),
        )

    monkeypatch.setattr(cross_pool_run, "fit_conditional_logit", zero_fit)
    flat = cross_pool_run.score_pair_folds(rows, folds).set_index("race_id").loc[mine.index]
    np.testing.assert_allclose(flat["ll_model"], flat["ll_offset"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(flat["delta_ll"], 0.0, atol=1e-12)
    np.testing.assert_allclose(flat["coef_log_ratio"], 0.0, atol=0)
    np.testing.assert_allclose(flat["ll_offset"], mine["ll_offset"], rtol=1e-12)


def test_pair_folds_fit_on_earlier_days_only(monkeypatch):
    rows = with_dead_heat_and_void(pair_study_rows(seed=31, gamma=0.3))
    folds = walk_forward(rows["race_date"], scheme=SHORT)
    date_of = dict(zip(rows["race_id"], rows["race_date"], strict=False))
    paid = rows.groupby("race_id")["won"].any()
    void = frozenset(paid.index[~paid])
    train_sets = {
        k: frozenset(rows.iloc[f.train_idx]["race_id"]) - void for k, f in enumerate(folds)
    }
    calls: list[frozenset] = []
    real = condlogit.fit_conditional_logit

    def spy(*args, **kwargs):
        calls.append(call_race_ids(args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(cross_pool_run, "fit_conditional_logit", spy)
    base = cross_pool_run.score_pair_folds(rows, folds)

    assert calls, "score_pair_folds fits through the module's fit_conditional_logit"
    fitted = set()
    for races in calls:
        matches = [k for k, s in train_sets.items() if races - void == s]
        assert matches, "a pair fit saw rows outside every fold's training days"
        k = matches[0]
        fitted.add(k)
        last = max(date_of[r] for r in races)
        assert last <= folds[k].train_end
        first_scored = min(rows.iloc[folds[k].test_idx]["race_date"])
        assert (first_scored - last).days >= SHORT.embargo_days
        assert not races & frozenset(rows.iloc[folds[k].test_idx]["race_id"])
    assert fitted == set(train_sets), "every fold is fitted"

    # Nothing dated after fold k's train_end, outside its test window, moves fold k.
    k = 2
    fold = folds[k]
    rng = np.random.default_rng(6)
    moved = rows.copy()
    later = (pd.to_datetime(moved["race_date"]) > pd.Timestamp(fold.train_end)).to_numpy()
    in_test = np.zeros(len(rows), dtype=bool)
    in_test[fold.test_idx] = True
    shift = later & ~in_test
    moved.loc[shift, "log_ratio"] = -2.0 * moved.loc[shift, "log_ratio"] + 0.3
    moved.loc[shift, "quinella_odds"] = np.round(
        moved.loc[shift, "quinella_odds"] * rng.uniform(0.6, 1.6, size=shift.sum()), 1
    ).clip(lower=1.1)
    moved.loc[shift, "won"] = False
    for _, idx in moved[shift].groupby("race_id").groups.items():
        moved.loc[rng.choice(np.asarray(idx)), "won"] = True
    again = cross_pool_run.score_pair_folds(moved, folds)

    def fold_rows(frame, which):
        part = frame[frame["fold"] == which].set_index("race_id").sort_index()
        return part[["ll_offset", "ll_model", "coef_log_ratio"]]

    before, after = fold_rows(base, k), fold_rows(again, k)
    assert list(after.index) == list(before.index)
    np.testing.assert_allclose(after.to_numpy(), before.to_numpy(), atol=1e-10)
    # The change is real: the next fold, which trains on those days, moves.
    nxt_before = fold_rows(base, k + 1)["coef_log_ratio"].to_numpy()
    nxt_after = fold_rows(again, k + 1)["coef_log_ratio"].to_numpy()
    assert abs(nxt_after[0] - nxt_before[0]) > 1e-6


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
    assert set(spec["variants"]) == EXPECTED_VARIANTS
    assert spec["family_size"] == len(EXPECTED_VARIANTS) == 5
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


SPEC_DOC = env.repo_root() / "docs" / "specs" / "cross-pool-prereg.md"


def flat(text: str) -> str:
    """Whitespace collapsed, so a quote wrapped across lines still reads as one."""
    return " ".join(text.split())


def test_the_spec_cites_the_owner_reframe_that_replaces_the_sign_off():
    """M4: the owner's reframe comment on #435 (2026-11-27) replaced the sign-off stop; the spec
    doc and build_spec's `decisions` cite it, with its quoted lines."""
    spec = build_spec(date_from=SPAN_START, date_to=date(2026, 10, 31), seed=SEED)
    assert "decisions" in spec
    doc = flat(SPEC_DOC.read_text(encoding="utf-8"))
    decisions = flat(json.dumps(spec["decisions"], ensure_ascii=False))
    for where, text in (("spec doc", doc), ("build_spec decisions", decisions)):
        for quote in ("Reframe, owner 2026-11-27", "#435", "No sign-off stop", "No owner sign-off"):
            assert quote in text, (where, quote)


def test_the_spec_names_the_pair_variant_and_how_the_offset_baselines_are_recorded():
    """M2 and Ruling 8: the pair model is a registered variant; the three offset baselines are
    named, and their returns are recorded as `offset_return[...]` metrics."""
    spec = build_spec(date_from=SPAN_START, date_to=date(2026, 10, 31), seed=SEED)
    whole = json.dumps(spec, ensure_ascii=False)
    doc = SPEC_DOC.read_text(encoding="utf-8")
    for name in (PAIR_VARIANT, *WIN_BASELINES, PAIR_BASELINE, "offset_return"):
        assert name in whole, ("build_spec", name)
        assert name in doc, ("spec doc", name)
    assert spec["family_size"] == 5


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


def synth_race(
    rng,
    day: date,
    venue: str,
    race_no: int,
    race_class,
    *,
    defect: str | None = None,
    tie_fav: bool = False,
) -> dict:
    """One race in every table. `defect`: "partial_board" (a starter pair off the Quinella
    board at t, priced only after t), "no_quinella_dividend" or "unpriced_final_pair" (a starter
    pair with no final Quinella price); `tie_fav`: the two shortest final Win prices tie. The
    final Quinella and Forecast odds carry noise, so some claims exceed 1."""
    rid = f"jra:{day.isoformat()}:{venue}:{race_no:02d}"
    meeting = f"jra:{day.isoformat()}:{venue}"
    n = int(rng.integers(6, 10))
    horses = list(range(1, n + 1))
    p = rng.gamma(2.0, size=n)
    p /= p.sum()
    final_win = np.maximum(np.round(0.8 / p, 1), 1.1)
    if tie_fav:
        low = np.argsort(final_win, kind="stable")[:2]
        final_win[low] = final_win[low].min()
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
    q_noise, f_noise = np.exp(rng.normal(0.0, 0.4, size=(2, n, n)))
    final_quin = {(i + 1, j + 1): max(round(0.775 * q_noise[i, j] / (ordered[i, j] + ordered[j, i]),
                                            1), 1.1)
                  for i, j in pairs}  # fmt: skip
    final_fc = {(i + 1, j + 1): max(round(0.75 * f_noise[i, j] / ordered[i, j], 1), 1.1)
                for i in range(n) for j in range(n) if i != j}  # fmt: skip
    order = np.argsort(-(np.log(p) + rng.gumbel(size=n)))
    finish = np.empty(n, dtype=int)
    finish[order] = np.arange(1, n + 1)
    first, second = horses[order[0]], horses[order[1]]
    lo, hi = sorted((first, second))
    quin_board = dict(quin_t)  # the Quinella board at t
    if defect == "partial_board":
        del quin_board[(horses[0], horses[1])]
    if defect == "unpriced_final_pair":
        del final_quin[next(pr for pr in itertools.combinations(horses, 2) if pr != (lo, hi))]

    post = datetime(day.year, day.month, day.day, 1, 0, tzinfo=UTC) + timedelta(
        minutes=40 * race_no + (0 if venue == VENUES[0] else 15)
    )

    def jitter(odds: dict, low: float, high: float) -> dict:
        return {k: max(round(v * rng.uniform(low, high), 1), 1.1) for k, v in odds.items()}

    win_board = dict(zip(horses, win_t.tolist(), strict=True))
    ts = capture(rid, post, 6.0, win=jitter(win_board, 0.9, 1.1))
    ts += capture(rid, post, 5.0, quinella=jitter(quin_board, 0.9, 1.1))
    ts += capture(rid, post, 2.0, win=win_board)
    ts += capture(rid, post, 2.5, quinella=quin_board)
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

    dividends = [
        ("WIN", str(first), final_win[first - 1] * 100 + 30),
        ("QUINELLA", f"{lo},{hi}", final_quin[(lo, hi)] * 100 + 40),
        ("FORECAST", f"{first},{second}", final_fc[(first, second)] * 100 + 50),
        *[("PLACE", str(horses[order[i]]), 150.0 + 10 * i) for i in range(3)],
    ]
    if defect == "no_quinella_dividend":
        dividends = [d for d in dividends if d[0] != "QUINELLA"]
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
    info = SimpleNamespace(
        race_id=rid,
        day=day,
        meeting=meeting,
        win_t=win_board,
        quin_board=quin_board,
        final_win=dict(zip(horses, final_win.tolist(), strict=True)),
        final_quin=dict(final_quin),
        final_fc=dict(final_fc),
        paid={(r["pool"], r["combination"]): r["dividend"] for r in paid},
    )
    return {"ts": ts, "races": [race], "runners": runners, "final": final, "paid": paid,
            "info": info}  # fmt: skip


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
    # Three defect races and a tied favourite on development days of the test months (from
    # 2026-11-01), each in a meeting that keeps its other race.
    tested = [d for d in dev if d >= date(2026, 11, 1)]
    defects = {
        kind: f"jra:{tested[i].isoformat()}:{VENUES[0]}:01"
        for kind, i in (
            ("partial_board", 1),
            ("no_quinella_dividend", 3),
            ("unpriced_final_pair", 5),
        )
    }
    defect_of = {rid: kind for kind, rid in defects.items()}
    tie_race = f"jra:{tested[2].isoformat()}:{VENUES[1]}:01"
    tables: dict[str, list[dict]] = {"races": [], "runners": [], "final": [], "paid": []}
    races_of: dict[date, list[str]] = {}
    info: dict[str, SimpleNamespace] = {}
    n = 0
    for day in sorted({*dev, HOLDOUT_DAY, EARLY_DAY, late}):
        ts = []
        for venue in VENUES:
            for race_no in range(1, RACES_PER_VENUE + 1):
                rid = f"jra:{day.isoformat()}:{venue}:{race_no:02d}"
                race = synth_race(
                    rng,
                    day,
                    venue,
                    race_no,
                    RACE_CLASSES[n % len(RACE_CLASSES)],
                    defect=defect_of.get(rid),
                    tie_fav=rid == tie_race,
                )
                n += 1
                ts += race["ts"]
                for name in tables:
                    tables[name] += race[name]
                races_of.setdefault(day, []).append(rid)
                info[rid] = race["info"]
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
    assert {info[r].day for r in (*defects.values(), tie_race)} <= set(test_days)
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
        info=info,
        defects=defects,
        tie_race=tie_race,
    )


# ---------------------------------------------------------------- references on the store


def win_scored_races(store: SimpleNamespace) -> list[str]:
    """The Win study's test-fold races: every race of a test day but the one whose T-2 Quinella
    board misses a starter pair (its x_cross is NaN)."""
    skip = store.defects["partial_board"]
    return [r for d in store.test_days for r in store.races_of[d] if r != skip]


def store_pairs(store: SimpleNamespace) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """The pair model's rows from first principles: one per starter pair of every development race
    with a complete T-2 Quinella board, a QUINELLA dividend and a final price for every starter
    pair; log_ratio = ln(q_pair_t / h_pair_t) from the boards at t; and the races dropped."""
    rows, dropped = [], {kind: set() for kind in PAIR_COUNTS}
    for day in store.dev_days:
        for rid in store.races_of[day]:
            info = store.info[rid]
            pairs = list(itertools.combinations(sorted(info.win_t), 2))
            paid = {c for pool, c in info.paid if pool == "QUINELLA"}
            if any(pr not in info.quin_board for pr in pairs):
                dropped["partial_board"].add(rid)
                continue
            if not paid:
                dropped["no_quinella_dividend"].add(rid)
                continue
            if any(pr not in info.final_quin for pr in pairs):
                dropped["unpriced_final_pair"].add(rid)
                continue
            q_t = ref_shares({pr: info.quin_board[pr] for pr in pairs})
            h_t = ref_harville(ref_shares(info.win_t))
            for a, b in pairs:
                rows.append(
                    {
                        "race_id": rid,
                        "race_date": day,
                        "meeting_id": info.meeting,
                        "combination": f"{a},{b}",
                        "won": f"{a},{b}" in paid,
                        "quinella_odds": info.final_quin[(a, b)],
                        "log_ratio": float(np.log(q_t[(a, b)] / h_t[(a, b)])),
                    }
                )
    return pd.DataFrame(rows), dropped


def ref_bets(info: SimpleNamespace, pool: str, p: dict[int, float]) -> list[dict]:
    """Every priced final combination of `pool` with its Harville probability from the Win
    probabilities p (Quinella either order; Forecast a first, then b), claim p x final odds and
    return at the final dividend."""
    if pool == "WIN":
        book = {str(h): (o, p[h]) for h, o in info.final_win.items()}
    elif pool == "QUINELLA":
        book = {f"{a},{b}": (o, p[a] * p[b] / (1 - p[a]) + p[b] * p[a] / (1 - p[b]))
                for (a, b), o in info.final_quin.items()}  # fmt: skip
    else:
        book = {f"{a},{b}": (o, p[a] * p[b] / (1 - p[a])) for (a, b), o in info.final_fc.items()}
    return [
        {"combination": c, "odds": o, "claim": pr * o, "ret": info.paid.get((pool, c), 0.0) / 100}
        for c, (o, pr) in book.items()
    ]


def cell_of(result, cell_id: str) -> dict:
    cells = {c["cell_id"]: c for c in result.cells}
    assert cell_id in cells, (cell_id, sorted(cells)[:8])
    return cells[cell_id]


def assert_cell(cell: dict, chosen: list[dict]) -> None:
    """The cell's n_bets, mean_return and mean_odds are the chosen bets'."""
    assert cell["n_bets"] == len(chosen), cell["cell_id"]
    if not chosen:
        assert cell["mean_return"] is None, cell["cell_id"]
        return
    assert cell["mean_return"] == pytest.approx(np.mean([b["ret"] for b in chosen]), abs=1e-9)
    assert cell["mean_odds"] == pytest.approx(np.mean([b["odds"] for b in chosen]), abs=1e-9)


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
        assert cell["study"] == DIAGNOSTIC_STUDY  # a retrospective diagnostic at final prices
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
    # The one-feature fits (x_cross and the pair model's log_ratio) still run.
    assert status == {v: "failed" if v in failed else "scored" for v in EXPECTED_VARIANTS}
    assert all(r.sources == ("jravan",) for r in registry.records(FAMILY_ID))
    assert result.family_size == len(EXPECTED_VARIANTS)
    cells = _strict(Path(result.cells_path).read_text(encoding="utf-8"))
    assert {c["pool"] for c in cells} == set(POOLS)
    assert contains_all(string_cells(result.coefficients), EXPECTED_VARIANTS - failed)


def test_the_run_fits_scores_and_settles_the_pair_model(store, main_run):
    """M2: the pair price at T-2 against the Win-implied pair, fitted on the final Quinella price,
    scored, recorded and settled in QUINELLA with its offset baseline."""
    result, registry = main_run.result, main_run.registry
    pairs, dropped = store_pairs(store)
    assert dropped == {kind: {rid} for kind, rid in store.defects.items()}  # the fixture's plants
    for key in PAIR_COUNTS.values():  # each dropped race is counted
        assert result.counts.get(key) == 1, (key, result.counts.get(key))

    record = [r for r in registry.records(FAMILY_ID) if r.variant == PAIR_VARIANT]
    assert len(record) == 1 and record[0].status == "scored"
    names = {m.metric for m in record[0].metrics}
    assert "coef[log_ratio]" in names
    assert "offset_return[QUINELLA:top1]" in names  # Ruling 8: its baseline's return

    # The coefficients, per fold (fitted on the fold's training rows) and pooled over the scored
    # rows: the reference conditional logit on rows built from the boards at t.
    folds = [f for f in walk_forward(pairs["race_date"], scheme=SHORT) if len(f.test_idx)]
    per_fold = [ref_pair_fit(pairs.iloc[f.train_idx]).coef[0] for f in folds]
    tested = pd.concat([pairs.iloc[f.test_idx] for f in folds], ignore_index=True)
    pooled = ref_pair_fit(tested)
    mine = result.coefficients[result.coefficients["variant"] == PAIR_VARIANT]
    assert set(mine["feature"]) == {"log_ratio"}
    row = mine[mine["fold"] == "pooled"]
    assert len(row) == 1
    np.testing.assert_allclose(row["coef"].to_numpy(dtype=float), pooled.coef[0], atol=1e-6)
    np.testing.assert_allclose(
        row["se_cluster"].to_numpy(dtype=float), pooled.se_cluster[0], atol=1e-6
    )
    got = np.sort(mine.loc[mine["fold"] != "pooled", "coef"].to_numpy(dtype=float))
    np.testing.assert_allclose(got, np.sort(per_fold), atol=1e-6)

    # The gain, overall over every scored pair race and by class group.
    gains = result.gains[result.gains["variant"] == PAIR_VARIANT]
    overall = gains[gains["class_group"] == "all"]
    assert len(overall) == 1
    assert int(overall["races"].iloc[0]) == tested["race_id"].nunique()
    assert contains_all(string_cells(gains), CLASS_GROUPS)

    # Settlement, N5 and cells in QUINELLA only, for the model and its offset baseline.
    for scorer in (PAIR_VARIANT, PAIR_BASELINE):
        settled = result.settlement[result.settlement["variant"] == scorer]
        assert len(settled) and set(settled["pool"]) == {"QUINELLA"}, scorer
        assert set(settled["selection"]) == {"value", "top1"}, scorer
        tail = result.tail[result.tail["variant"] == scorer]
        assert len(tail) and set(tail["pool"]) == {"QUINELLA"}, scorer
        cells = [c for c in result.cells if c["cell_id"].split("/")[0] == scorer]
        assert {c["pool"] for c in cells} == {"QUINELLA"}, scorer
        assert any(c["odds_band"] for c in cells), scorer  # N5 bands
        top1 = cell_of(result, f"{scorer}/QUINELLA/top1/all")
        assert top1["n_bets"] == tested["race_id"].nunique(), scorer  # one bet per scored race


def test_top1_breaks_equal_claims_by_the_lowest_final_odds(store, main_run):
    """Under an offset alone every claim of a race is mathematically equal (q x O = 1 / sum 1/O),
    so top1 is the final favourite, a tie of prices going to the lowest combination text: never
    the argmax of floating-point noise."""
    result = main_run.result
    tie = sorted(store.info[store.tie_race].final_win.values())
    assert tie[0] == tie[1] and store.tie_race in win_scored_races(store)

    # offset:normalised in WIN: the final Win favourite of every scored race.
    chosen = []
    for rid in win_scored_races(store):
        info = store.info[rid]
        bets = ref_bets(info, "WIN", ref_shares(info.final_win))
        claims = [b["claim"] for b in bets]
        assert max(claims) - min(claims) <= TIE * max(claims)  # one tie across the field
        chosen.append(ref_top1(bets))
    assert_cell(cell_of(result, "offset:normalised/WIN/top1/all"), chosen)

    # offset:quinella in QUINELLA: the final Quinella favourite of every scored pair race.
    pairs, _ = store_pairs(store)
    folds = [f for f in walk_forward(pairs["race_date"], scheme=SHORT) if len(f.test_idx)]
    tested = sorted(set().union(*(set(pairs.iloc[f.test_idx]["race_id"]) for f in folds)))
    chosen = []
    for rid in tested:
        info = store.info[rid]
        q = ref_shares(info.final_quin)
        bets = [
            {"combination": f"{a},{b}", "odds": o, "claim": q[(a, b)] * o,
             "ret": info.paid.get(("QUINELLA", f"{a},{b}"), 0.0) / 100}
            for (a, b), o in info.final_quin.items()
        ]  # fmt: skip
        chosen.append(ref_top1(bets))
    assert_cell(cell_of(result, f"{PAIR_BASELINE}/QUINELLA/top1/all"), chosen)


def test_settlement_prices_combinations_by_harville_from_the_win_probabilities(store, main_run):
    """The settlement path beyond `settle`, on offset:normalised (p = the final Win shares):
    Harville QUINELLA and FORECAST probabilities (FORECAST in leg order), claim = p x final odds,
    `value` (claim > 1) and `top1` (the largest claim), and each cell's n_bets, mean_return and
    mean_odds over the scored races (a race with no dividend in the pool left out of it)."""
    result = main_run.result
    scored = win_scored_races(store)
    for pool in POOLS:
        races = [r for r in scored if any(k[0] == pool for k in store.info[r].paid)]
        value, top1, swapped = [], [], []
        for rid in races:
            info = store.info[rid]
            p = ref_shares(info.final_win)
            bets = ref_bets(info, pool, p)
            assert all(abs(b["claim"] - 1.0) > TIE for b in bets)  # none on the value edge
            value += [b for b in bets if b["claim"] > 1.0]
            top1.append(ref_top1(bets))
            if pool == "FORECAST":  # the legs read the wrong way round: P(b first, then a)
                for bet in bets:
                    a, b = (int(h) for h in bet["combination"].split(","))
                    if p[b] * p[a] / (1 - p[b]) * bet["odds"] > 1.0:
                        swapped.append(bet)
        if pool == "WIN":
            assert value == []  # q x O = 1 / sum 1/O < 1: no Win value bet at the offset
        else:
            assert value, pool  # the fixture's noisy final prices give value bets
            assert_cell(cell_of(result, f"offset:normalised/{pool}/top1/all"), top1)
        assert_cell(cell_of(result, f"offset:normalised/{pool}/value/all"), value)
        if pool == "QUINELLA":
            assert store.defects["no_quinella_dividend"] not in races
        if pool == "FORECAST":  # the fixture tells the leg orders apart
            assert (len(swapped), np.mean([b["ret"] for b in swapped])) != (
                len(value),
                np.mean([b["ret"] for b in value]),
            )


def test_settlement_is_labelled_a_retrospective_final_price_diagnostic(main_run):
    """M5: settlement, N5 and cells use the closing price twice (offset and claim), so they are
    retrospective diagnostics, not decision-time rules: the cells' `study` says so, and the spec,
    the private report and the public report each carry the sentence."""
    result = main_run.result
    cells = _strict(Path(result.cells_path).read_text(encoding="utf-8"))
    assert {c["study"] for c in cells} == {c["study"] for c in result.cells} == {DIAGNOSTIC_STUDY}
    spec = build_spec(date_from=SPAN_START, date_to=STORE_TO, seed=SEED)
    texts = {
        "private report": Path(result.report_path).read_text(encoding="utf-8"),
        "public report": render_public_report(result),
        "spec doc": SPEC_DOC.read_text(encoding="utf-8"),
        "build_spec": json.dumps(spec, ensure_ascii=False),
    }
    for where, text in texts.items():
        assert "retrospective" in text.lower(), where


def no_forecast_store(store: SimpleNamespace, root: Path) -> Path:
    """A copy of the store with every FORECAST row gone from final_odds and dividends."""
    data = root / "data"
    shutil.copytree(store.data, data)
    for name in ("final_odds", "dividends"):
        path = data / "processed" / "jravan" / f"{name}.parquet"
        table = pq.read_table(path)
        pq.write_table(table.filter(pc.not_equal(table["pool"], "FORECAST")), path)
    return data


def test_an_unavailable_settlement_pool_keeps_the_fits_and_the_other_pools(
    store, main_run, tmp_path, monkeypatch
):
    """M3: no FORECAST price or dividend at all. FORECAST is unavailable, named as such; every
    variant is still scored, and its fits, gains and WIN and QUINELLA settlement are the run's
    with FORECAST present."""
    data = no_forecast_store(store, tmp_path)
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(data))
    monkeypatch.setattr(cross_pool_run, "load_country_scheme", lambda country: SHORT)
    registry = temp_registry(tmp_path / "reg")
    args = {"date_from": SPAN_START, "date_to": STORE_TO, "seed": SEED}
    freeze_spec(**args, registry=registry)

    result = run_cross_pool(**args, out_dir=data / "reports" / "no-forecast", registry=registry)

    assert result.failed_variants == []
    records = registry.records(FAMILY_ID)
    assert sorted(r.variant for r in records) == sorted(EXPECTED_VARIANTS)
    assert all(r.status == "scored" for r in records)
    assert "FORECAST" in result.unavailable_pools
    report = Path(result.report_path).read_text(encoding="utf-8")
    assert any("FORECAST" in line and "unavailable" in line.lower() for line in report.splitlines())
    for r in records:
        if r.variant in WIN_VARIANTS:  # the pool its record would have settled
            blob = json.dumps(r.params) + " ".join(m.metric for m in r.metrics)
            assert "FORECAST" in blob, r.variant

    full = main_run.result
    key = ["variant", "feature", "fold"]
    for got, want, by in (
        (result.coefficients, full.coefficients, key),
        (result.gains, full.gains, ["variant", "class_group"]),
    ):
        assert contains_all(string_cells(got), EXPECTED_VARIANTS)
        got = got.sort_values(by).reset_index(drop=True)
        want = want.sort_values(by).reset_index(drop=True)
        pd.testing.assert_frame_equal(got[want.columns], want, check_dtype=False, atol=1e-12)
    settled = result.settlement
    assert set(settled["pool"]) == {"WIN", "QUINELLA"}
    for scorer in (*WIN_VARIANTS, *WIN_BASELINES):
        assert set(settled.loc[settled["variant"] == scorer, "pool"]) == {"WIN", "QUINELLA"}
    for scorer in (PAIR_VARIANT, PAIR_BASELINE):
        assert set(settled.loc[settled["variant"] == scorer, "pool"]) == {"QUINELLA"}
    columns = ["variant", "pool", "selection", "class_group", "n_bets", "mean_return"]
    want = full.settlement[full.settlement["pool"] != "FORECAST"][columns]
    order = columns[:4]
    pd.testing.assert_frame_equal(
        settled[columns].sort_values(order).reset_index(drop=True),
        want.sort_values(order).reset_index(drop=True),
        check_dtype=False,
        atol=1e-12,
    )
    cells = _strict(Path(result.cells_path).read_text(encoding="utf-8"))
    assert {c["pool"] for c in cells} == {"WIN", "QUINELLA"}


def test_a_trial_run_is_recorded_in_the_trial_family(store, main_run, tmp_path, monkeypatch):
    """M1: a trial runs the same code, appends one record per variant to the trial family, never
    freezes or writes FAMILY_ID, and writes files named as a trial's; the run proper still needs
    the family frozen."""
    assert cross_pool_run.TRIAL_FAMILY_ID == TRIAL_FAMILY
    assert isinstance(cross_pool_run.TRIAL_SCHEME, FoldScheme)
    assert TRIAL_FAMILY in SPEC_DOC.read_text(encoding="utf-8")  # where the trial records go
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(store.data))
    monkeypatch.setattr(cross_pool_run, "load_country_scheme", lambda country: SHORT)
    monkeypatch.setattr(cross_pool_run, "TRIAL_SCHEME", SHORT)
    registry = temp_registry(tmp_path)
    args = run_args(store, "trial", registry)

    result = run_cross_pool(**args, trial=True)  # no freeze first

    assert result.family_id == TRIAL_FAMILY
    trial = registry.records(TRIAL_FAMILY)
    assert sorted(r.variant for r in trial) == sorted(EXPECTED_VARIANTS)
    assert all(r.status == "scored" and r.sources == ("jravan",) for r in trial)
    for r in trial:  # its span and fold scheme
        assert r.params["date_from"] == SPAN_START.isoformat()
        assert r.params["date_to"] == STORE_TO.isoformat()
        assert "month" in r.fold_scheme, r.fold_scheme
    assert registry.records(FAMILY_ID) == []
    assert registry.frozen_spec_hash(FAMILY_ID) is None
    lines = [json.loads(line) for line in registry.path.read_text().splitlines()]
    assert not [x for x in lines if x["kind"] == "freeze" and x["family_id"] == FAMILY_ID]
    assert len(result.run_ids) == len(trial)
    # The same code as the run proper: the same coefficients.
    by = ["variant", "feature", "fold"]
    want = main_run.result.coefficients.sort_values(by).reset_index(drop=True)
    got = result.coefficients.sort_values(by).reset_index(drop=True)
    pd.testing.assert_frame_equal(got[want.columns], want, check_dtype=False, atol=1e-12)

    out = args["out_dir"]
    assert Path(result.report_path).is_file() and Path(result.cells_path).is_file()
    written = [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()]
    assert written and all("trial" in name for name in written), written
    cells = _strict(Path(result.cells_path).read_text(encoding="utf-8"))
    assert {c["family_id"] for c in cells} == {TRIAL_FAMILY}

    # The run proper is still refused before any store file is opened.
    opened: list[Path] = []
    with pytest.MonkeyPatch.context() as mp:
        install_file_spy(mp, under(store.processed), opened)
        with pytest.raises(FamilyNotFrozen):
            run_cross_pool(**run_args(store, "after-trial", registry))
    assert opened == []
    assert registry.records(FAMILY_ID) == []


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


def test_family_not_frozen_names_the_frozen_spec(store, tmp_path, monkeypatch):
    """When another spec is frozen, the refusal names its date_from, date_to and seed, so the run
    can be repeated with the frozen arguments (a second freeze would raise)."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(store.data))
    monkeypatch.setattr(cross_pool_run, "load_country_scheme", lambda country: SHORT)
    registry = temp_registry(tmp_path)
    frozen = {"date_from": date(2015, 3, 7), "date_to": date(2027, 1, 17), "seed": 9137}
    freeze_spec(**frozen, registry=registry)
    opened: list[Path] = []
    with pytest.MonkeyPatch.context() as mp:
        install_file_spy(mp, under(store.processed), opened)
        with pytest.raises(FamilyNotFrozen) as refused:
            run_cross_pool(**run_args(store, "other", registry))
    message = str(refused.value)
    assert "2015-03-07" in message and "2027-01-17" in message, message
    assert re.search(r"(?<![0-9A-Za-z])9137(?![0-9A-Za-z])", message), message
    assert opened == []
    assert registry.records(FAMILY_ID) == []


class StopRun(Exception):
    """Raised by the CLI test's stand-in for run_cross_pool, so nothing is read or written."""


def test_the_cli_requires_to_for_every_mode(tmp_path, monkeypatch, capsys):
    """--to has no default: --freeze, --run and --trial refuse to start without it (a frozen
    span end of 'today' would not match a later run). --trial runs the study as a trial."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(data))
    calls: list[dict] = []

    def stand_in(**kwargs):
        calls.append(kwargs)
        raise StopRun

    monkeypatch.setattr(cross_pool_run, "run_cross_pool", stand_in)
    registry = temp_registry(data)
    for argv in (["--freeze"], ["--run"], ["--trial"], ["--freeze", "--seed", "7"]):
        with pytest.raises(SystemExit) as refused:
            main(argv)
        assert refused.value.code not in (0, None), argv
    assert calls == []
    assert not registry.path.exists() or registry.frozen_spec_hash(FAMILY_ID) is None

    assert main(["--freeze", "--to", "2026-10-31"]) == 0
    want = spec_hash(build_spec(date_from=SPAN_START, date_to=date(2026, 10, 31), seed=SEED))
    assert registry.frozen_spec_hash(FAMILY_ID) == want

    out = str(tmp_path / "out")
    for argv, trial, last in (
        (["--run", "--to", "2026-10-31", "--out", out], False, date(2026, 10, 31)),
        (["--trial", "--to", "2027-01-31", "--out", out], True, date(2027, 1, 31)),
    ):
        with pytest.raises(StopRun):
            main(argv)
        assert calls[-1]["date_to"] == last, argv
        assert bool(calls[-1].get("trial", False)) is trial, argv
    assert len(calls) == 2


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
    run_ids = [s * 10 + s[:4] for s in ("fedcba", "bcadef", "cabfed", "dabcef", "efabcd")]
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
