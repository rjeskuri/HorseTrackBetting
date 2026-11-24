"""The v0 calibrated-market Win model (ticket #114, outline 2 and 3).

Contract pinned here (`horsetrack.countries.japan.models.v0.calibration`):
- `market_q(odds)`: q_i = (1/O_i) / sum_j (1/O_j) over the priced runners; NaN odds stay NaN.
- `fit_win_calibration(frame, *, n_knots=5)`: a conditional logit, P(i wins) proportional to
  exp(f(ln q_i)), f a natural cubic spline in ln q with `n_knots` knots at quantiles of the
  training rows' ln q (`knots` are in ln q units). Frame columns race_id, race_date, win_odds,
  won (horse_no may be present). A race before 2014-06-07 is a PreReformPriceError (from
  `store`). `train_start`/`train_end` are the first and last race dates, `n_races` the races fitted.
- `WinCalibration.predict(q, race=None)`: one race; NaN entries stay NaN, the rest sum to 1;
  `race` is accepted and ignored (#118's WinModel protocol). `to_json` / `from_json` round-trip.
- `WinCalibration.predict_many(q, races)`: many races at once; `q` is a flat array of runners'
  market q and `races` a same-length array labelling each runner's race (any hashable labels, rows
  need not be contiguous); returns p aligned with `q`, each race's p equal to `predict` on it.
- `oof_predict(frame, *, folds, n_knots=5, chunk_races=2000)`: `folds` are #70 `Fold`s whose
  positions index the rows of `frame` (built with `walk_forward(frame["race_date"], ...)`). For
  fold k, the model is `fit_win_calibration(frame.iloc[fold.train_idx], n_knots=n_knots)` and every
  test row gets p_cal from it. Output columns race_id, horse_no, fold, q, p_cal, one row per test
  row; `fold` is k, the fold's position in `folds`. ALL of oof_predict's p_cal goes through
  `WinCalibration.predict_many`, called once per chunk of at most `chunk_races` whole races
  (never one race split over two calls); the chunk test spies on that method.
- `run_fit(out_dir=None)` reads runners from 2014-06-07 to the store's last date, drops scratched
  and unpriced runners before market_q, leaves out races with a dead heat for first, fits the
  live model on every remaining development row, writes out-of-fold predictions over Japan's #70
  scheme (`load_country_scheme('japan')`), and writes `win_calibration-<fit_id>.json`,
  `win_calibration_oof-<fit_id>.parquet` and `manifest-win_calibration-<fit_id>.json` under
  `out_dir`, default `data_dir()/models/japan/v0/`. The manifest (also returned) has `fit_id`,
  `git_sha`, `n_races`, `races_left_out` (with `dead_heat_for_first`: the count) and `folds` (one
  entry per fold). CLI: `python -m horsetrack.countries.japan.models.v0.calibration --fit
  [--out DIR]`. An `out_dir` or `--out` that resolves inside the repository (absolute, relative to
  the cwd, through `..` or a symlink) is refused with `paths.DataDirInsideRepo`, the error
  `data_dir()` raises for a data dir inside the repo, and nothing is written; the CLI exits
  non-zero.

Expected values come from first principles (q from odds, planted p from q^gamma) or, for the
out-of-fold contract, from the separately tested `fit_win_calibration` on the fold's training
rows. All data is synthetic; HORSETRACK_DATA_DIR points under tmp.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core import env, paths
from horsetrack.core.eval.folds import FoldScheme, walk_forward
from horsetrack.countries.japan.models.v0 import calibration
from horsetrack.countries.japan.models.v0.calibration import (
    WinCalibration,
    fit_win_calibration,
    market_q,
    oof_predict,
    run_fit,
)
from horsetrack.countries.japan.models.v0.store import PreReformPriceError

DATA_VAR = "HORSETRACK_DATA_DIR"
TAKEOUT = 0.8
PROBE = np.array([0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05])
EXTRA_PROBES = [
    np.array([0.5, 0.3, 0.2]),
    np.array([0.25, 0.25, 0.125, 0.125, 0.125, 0.125]),
    np.array([0.3, 0.15, 0.12, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]),
]
PLANTED_GAMMA = 1.15


# ---------------------------------------------------------------------------------------------
# Synthetic races and independent references
# ---------------------------------------------------------------------------------------------


def simulate_q(rng, n_races, n_runners, alpha=0.9, lo=0.01, hi=0.6):
    """Market q per race (rows sum to 1), spread over about lo..hi."""
    q = rng.dirichlet(np.full(n_runners, alpha), size=n_races)
    for _ in range(3):
        q = np.clip(q, lo, hi)
        q /= q.sum(axis=1, keepdims=True)
    return q


def power_p(q, gamma):
    p = np.asarray(q, dtype=float) ** gamma
    return p / p.sum(axis=-1, keepdims=True)


def draw_winners(rng, p):
    u = rng.random(len(p))[:, None]
    return np.minimum((np.cumsum(p, axis=1) < u).sum(axis=1), p.shape[1] - 1)


def ref_q(odds):
    inv = 1.0 / np.asarray(odds, dtype=float)
    return inv / np.nansum(inv)


def to_frame(q, winner, *, first_day=date(2015, 1, 3), races_per_day=12, first_race=0):
    """A fit frame (race_id, race_date, horse_no, win_odds, won) for simulated races."""
    n_races, n = q.shape
    r = np.arange(first_race, first_race + n_races)
    days = [first_day + timedelta(days=int(k)) for k in r // races_per_day]
    ids = [
        f"jra:{d.isoformat()}:TOK:{k % races_per_day + 1:02d}" for d, k in zip(days, r, strict=True)
    ]
    won = np.arange(n)[None, :] == np.asarray(winner)[:, None]
    return pd.DataFrame(
        {
            "race_id": np.repeat(ids, n),
            "race_date": np.repeat(np.array(days, dtype=object), n),
            "horse_no": np.tile(np.arange(1, n + 1), n_races),
            "win_odds": (TAKEOUT / q).ravel(),
            "won": won.ravel(),
        }
    )


def race_q(frame):
    """{race_id: (horse_no array, q array)} from the frame's odds, computed here."""
    out = {}
    for race_id, g in frame.groupby("race_id", sort=False):
        out[race_id] = (g["horse_no"].to_numpy(), ref_q(g["win_odds"].to_numpy()))
    return out


# ---------------------------------------------------------------------------------------------
# Heavy fits, shared: 20,000 twelve-runner races each
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def recovery_frames():
    rng = np.random.default_rng(114)
    q_id = simulate_q(rng, 20_000, 12)
    identity = to_frame(q_id, draw_winners(rng, q_id))
    q_b = simulate_q(rng, 20_000, 12)
    biased = to_frame(q_b, draw_winners(rng, power_p(q_b, PLANTED_GAMMA)))
    return identity, biased


@pytest.fixture(scope="module")
def identity_fit(recovery_frames):
    return fit_win_calibration(recovery_frames[0])


@pytest.fixture(scope="module")
def biased_fit(recovery_frames):
    return fit_win_calibration(recovery_frames[1])


# ---------------------------------------------------------------------------------------------
# market_q
# ---------------------------------------------------------------------------------------------


def test_market_q_normalises_priced_runners():
    got = market_q(np.array([2.0, 4.0, np.nan, 4.0]))
    np.testing.assert_allclose(got, [0.5, 0.25, np.nan, 0.25], rtol=0, atol=1e-12)
    assert np.isnan(got[2])

    rng = np.random.default_rng(1)
    for n in (2, 5, 9, 18):
        odds = np.round(rng.uniform(1.1, 300.0, size=n), 1)
        if n > 2:
            odds[rng.choice(n, size=n // 3, replace=False)] = np.nan
        got = market_q(odds)
        want = ref_q(odds)
        assert got.shape == odds.shape
        np.testing.assert_array_equal(np.isnan(got), np.isnan(odds))
        np.testing.assert_allclose(got[~np.isnan(odds)], want[~np.isnan(odds)], atol=1e-12)
        assert np.nansum(got) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------------------------
# DoD: recovers a calibrated market as the identity and a planted favourite-longshot bias
# ---------------------------------------------------------------------------------------------


def test_a_calibrated_market_is_recovered_as_the_identity(identity_fit):
    got = identity_fit.predict(PROBE)
    np.testing.assert_allclose(got, PROBE, rtol=0, atol=0.02)
    for probe in EXTRA_PROBES:
        np.testing.assert_allclose(identity_fit.predict(probe), probe, rtol=0, atol=0.02)


def test_a_planted_longshot_bias_is_recovered(biased_fit):
    planted = power_p(PROBE, PLANTED_GAMMA)
    # The planted bias is visibly away from the identity, so the identity cannot pass here.
    assert np.abs(planted - PROBE).max() > 0.04
    np.testing.assert_allclose(biased_fit.predict(PROBE), planted, rtol=0, atol=0.02)
    for probe in EXTRA_PROBES:
        np.testing.assert_allclose(
            biased_fit.predict(probe), power_p(probe, PLANTED_GAMMA), rtol=0, atol=0.02
        )


def test_the_fit_records_its_training_rows(recovery_frames, identity_fit, biased_fit):
    identity = recovery_frames[0]
    assert identity_fit.n_races == 20_000
    assert identity_fit.train_start == min(identity["race_date"])
    assert identity_fit.train_end == max(identity["race_date"])
    assert isinstance(identity_fit.fit_id, str) and identity_fit.fit_id
    assert identity_fit.fit_id != biased_fit.fit_id
    ln_q = np.log(ref_q_per_race(identity))
    knots = np.asarray(identity_fit.knots, dtype=float)
    assert len(knots) == 5
    assert np.all(np.diff(knots) > 0)
    assert ln_q.min() <= knots[0] and knots[-1] <= ln_q.max()
    assert all(math.isfinite(c) for c in identity_fit.coef)


def ref_q_per_race(frame):
    inv = 1.0 / frame["win_odds"].to_numpy(float)
    totals = pd.Series(inv).groupby(frame["race_id"].to_numpy()).transform("sum").to_numpy()
    return inv / totals


@pytest.mark.parametrize("n_knots", [3, 4, 7])
def test_knots_follow_n_knots_and_the_training_rows(n_knots):
    rng = np.random.default_rng(40 + n_knots)
    q_wide = simulate_q(rng, 2_000, 10, alpha=0.5, lo=0.003, hi=0.8)
    q_flat = simulate_q(rng, 2_000, 10, alpha=6.0, lo=0.03, hi=0.3)
    wide = fit_win_calibration(to_frame(q_wide, draw_winners(rng, q_wide)), n_knots=n_knots)
    flat = fit_win_calibration(to_frame(q_flat, draw_winners(rng, q_flat)), n_knots=n_knots)
    for fit, q in ((wide, q_wide), (flat, q_flat)):
        knots = np.asarray(fit.knots, dtype=float)
        assert len(knots) == n_knots
        assert np.all(np.diff(knots) > 0)
        assert np.log(q).min() <= knots[0] and knots[-1] <= np.log(q).max()
    # Knots move with the training rows' ln q: the wider market has wider outer knots.
    assert wide.knots[0] < flat.knots[0] and wide.knots[-1] > flat.knots[-1]


# ---------------------------------------------------------------------------------------------
# DoD: predict sums to 1 for any field size
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_runners", [2, 8, 18])
def test_predict_sums_to_one_for_any_field_size(identity_fit, biased_fit, n_runners):
    rng = np.random.default_rng(n_runners)
    for fit in (identity_fit, biased_fit):
        for _ in range(5):
            q = rng.dirichlet(np.full(n_runners, 1.0))
            got = fit.predict(q)
            assert got.shape == (n_runners,)
            assert np.all(np.isfinite(got)) and np.all(got > 0)
            assert got.sum() == pytest.approx(1.0, abs=1e-9)
    # A two-horse field of equal prices is a coin toss under any monotone curve.
    np.testing.assert_allclose(biased_fit.predict(np.array([0.5, 0.5])), [0.5, 0.5], atol=1e-12)


def test_nan_entries_stay_nan_and_race_is_ignored(biased_fit):
    q = np.array([0.4, np.nan, 0.35, np.nan, 0.25])
    got = biased_fit.predict(q)
    assert np.isnan(got[1]) and np.isnan(got[3])
    priced = got[[0, 2, 4]]
    assert priced.sum() == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(priced, biased_fit.predict(np.array([0.4, 0.35, 0.25])), atol=1e-12)
    np.testing.assert_allclose(priced, power_p(np.array([0.4, 0.35, 0.25]), 1.15), atol=0.02)

    probe = PROBE.copy()
    base = biased_fit.predict(probe)
    np.testing.assert_array_equal(biased_fit.predict(probe, None), base)
    np.testing.assert_array_equal(biased_fit.predict(probe, object()), base)
    np.testing.assert_array_equal(biased_fit.predict(probe, race={"race_id": "x"}), base)


def test_predict_many_matches_predict_race_by_race(biased_fit):
    rng = np.random.default_rng(7)
    sizes = [2, 5, 8, 12, 18, 7]
    labels, qs = [], []
    for k, n in enumerate(sizes):
        labels.append(np.full(n, f"race-{k}", dtype=object))
        qs.append(rng.dirichlet(np.full(n, 0.9)))
    flat_labels = np.concatenate(labels)
    flat_q = np.concatenate(qs)
    order = rng.permutation(len(flat_q))  # rows of a race need not be contiguous
    got = biased_fit.predict_many(flat_q[order], flat_labels[order])
    assert got.shape == flat_q.shape
    back = np.empty_like(got)
    back[order] = got
    start = 0
    for n, q in zip(sizes, qs, strict=True):
        np.testing.assert_allclose(back[start : start + n], biased_fit.predict(q), atol=1e-12)
        start += n


# ---------------------------------------------------------------------------------------------
# D10: pre-reform rows are refused
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("as_timestamps", [False, True])
def test_pre_reform_rows_are_refused(as_timestamps):
    rng = np.random.default_rng(66)
    q = simulate_q(rng, 400, 8)
    # one race a day from 2014-06-06: a single pre-reform race, and enough months for folds
    frame = to_frame(q, draw_winners(rng, q), first_day=date(2014, 6, 6), races_per_day=1)
    assert min(frame["race_date"]) == date(2014, 6, 6)
    if as_timestamps:
        frame["race_date"] = pd.to_datetime(frame["race_date"])
    with pytest.raises(PreReformPriceError):
        fit_win_calibration(frame)
    folds = walk_forward(frame["race_date"], scheme=MONTHLY)
    assert folds
    with pytest.raises(PreReformPriceError):
        oof_predict(frame, folds=folds)
    # From the era start the same races fit.
    later = frame[pd.to_datetime(frame["race_date"]) >= pd.Timestamp(2014, 6, 7)]
    assert fit_win_calibration(later).n_races == 400 - 1


# ---------------------------------------------------------------------------------------------
# DoD: knots and coefficients from training rows only; OOF never sees its own fold
# ---------------------------------------------------------------------------------------------

MONTHLY = FoldScheme(
    unit="month",
    test_length=1,
    step=1,
    min_train=6,
    window="expanding",
    embargo_days=0,
    anchor_month=1,
)
FOLD_J = 3


@pytest.fixture(scope="module")
def oof_case():
    rng = np.random.default_rng(2114)
    q = simulate_q(rng, 3_000, 10)
    frame = to_frame(q, draw_winners(rng, power_p(q, 1.1)), races_per_day=6)
    folds = walk_forward(frame["race_date"], scheme=MONTHLY)
    assert len(folds) >= FOLD_J + 3
    base = oof_predict(frame, folds=folds, n_knots=4)
    return frame, folds, base


def keyed(oof):
    return oof.set_index(["race_id", "horse_no"]).sort_index()


def rows_of(frame, fold):
    return frame.iloc[fold.test_idx].set_index(["race_id", "horse_no"]).index


def test_oof_output_covers_the_test_rows_with_the_fold_position(oof_case):
    frame, folds, base = oof_case
    assert {"race_id", "horse_no", "fold", "q", "p_cal"} <= set(base.columns)
    got = keyed(base)
    assert not got.index.duplicated().any()
    expected_index = pd.MultiIndex.from_tuples([], names=["race_id", "horse_no"])
    for k, fold in enumerate(folds):
        idx = rows_of(frame, fold)
        expected_index = expected_index.append(idx)
        assert (got.loc[idx, "fold"] == k).all()
    assert set(got.index) == set(expected_index)
    sums = base.groupby("race_id")["p_cal"].sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0, atol=1e-9)


def test_oof_p_cal_is_the_fold_training_fit_applied_to_each_race(oof_case):
    frame, folds, base = oof_case
    got = keyed(base)
    for k in (0, FOLD_J, len(folds) - 1):
        fold = folds[k]
        model = fit_win_calibration(frame.iloc[fold.train_idx], n_knots=4)
        assert len(model.knots) == 4
        test = frame.iloc[fold.test_idx]
        for race_id, (horses, q) in race_q(test).items():
            keys = list(zip([race_id] * len(horses), horses, strict=True))
            np.testing.assert_allclose(got.loc[keys, "q"].to_numpy(), q, atol=1e-12)
            np.testing.assert_allclose(
                got.loc[keys, "p_cal"].to_numpy(), model.predict(q), atol=1e-9
            )


def test_oof_predictions_do_not_see_their_own_fold(oof_case):
    frame, folds, base = oof_case
    flipped = frame.copy()
    test_rows = folds[FOLD_J].test_idx
    # Every race of fold j is now won by its longest shot.
    sub = flipped.iloc[test_rows]
    longest = sub.groupby("race_id")["win_odds"].transform("max").to_numpy()
    flipped.loc[flipped.index[test_rows], "won"] = sub["win_odds"].to_numpy() == longest
    assert (flipped["won"] != frame["won"]).sum() > 100

    again = oof_predict(flipped, folds=folds, n_knots=4)
    a, b = keyed(base), keyed(again)
    for k in range(FOLD_J + 1):
        idx = rows_of(frame, folds[k])
        np.testing.assert_allclose(
            b.loc[idx, "p_cal"].to_numpy(), a.loc[idx, "p_cal"].to_numpy(), rtol=0, atol=1e-12
        )
    # A later fold trains on fold j, so the flip must reach it (the test can fail).
    later = rows_of(frame, folds[FOLD_J + 1])
    assert np.abs(b.loc[later, "p_cal"].to_numpy() - a.loc[later, "p_cal"].to_numpy()).max() > 1e-6


def test_oof_knots_and_coefficients_come_from_training_rows_only(oof_case):
    frame, folds, base = oof_case
    rng = np.random.default_rng(77)
    after = pd.to_datetime(frame["race_date"]) > pd.Timestamp(folds[FOLD_J].test_end)
    changed = frame.copy()
    later_ids = changed.loc[after, "race_id"].unique()
    q_new = simulate_q(rng, len(later_ids), 10, alpha=0.15, lo=0.0005, hi=0.97)
    w_new = draw_winners(rng, power_p(q_new, 0.7))
    # rows are race-major, ten runners per race, in race order
    changed.loc[after, "win_odds"] = (TAKEOUT / q_new).ravel()
    changed.loc[after, "won"] = (np.arange(10)[None, :] == w_new[:, None]).ravel()
    assert after.sum() > len(frame) / 4

    again = oof_predict(changed, folds=folds, n_knots=4)
    a, b = keyed(base), keyed(again)
    for k in range(FOLD_J + 1):
        idx = rows_of(frame, folds[k])
        np.testing.assert_allclose(
            b.loc[idx, "p_cal"].to_numpy(), a.loc[idx, "p_cal"].to_numpy(), rtol=0, atol=1e-12
        )


# ---------------------------------------------------------------------------------------------
# DoD (memory): prediction over many races runs in chunks of at most chunk_races
# ---------------------------------------------------------------------------------------------

YEARLY = FoldScheme(
    unit="year",
    test_length=1,
    step=1,
    min_train=1,
    window="expanding",
    embargo_days=0,
    anchor_month=1,
)


@pytest.fixture(scope="module")
def chunk_case():
    rng = np.random.default_rng(5000)
    q = simulate_q(rng, 5_000, 10)
    frame = to_frame(q, draw_winners(rng, q), first_day=date(2015, 1, 1), races_per_day=7)
    folds = walk_forward(frame["race_date"], scheme=YEARLY)
    assert len(folds) == 1
    n_test = frame.iloc[folds[0].test_idx]["race_id"].nunique()
    assert n_test > 2_000
    return frame, folds, n_test


@pytest.mark.parametrize("chunk_races", [1_000, 700, None])
def test_prediction_runs_in_chunks(monkeypatch, chunk_case, chunk_races):
    """oof_predict routes every p_cal through WinCalibration.predict_many(q, races), one call per
    chunk of at most chunk_races whole races (default 2,000); the spy counts races per call."""
    frame, folds, n_test = chunk_case
    calls = []
    original = WinCalibration.predict_many

    def spy(self, q, races, *args, **kwargs):
        calls.append(len(np.unique(np.asarray(races))))
        return original(self, q, races, *args, **kwargs)

    monkeypatch.setattr(calibration.WinCalibration, "predict_many", spy)
    kwargs = {} if chunk_races is None else {"chunk_races": chunk_races}
    limit = 2_000 if chunk_races is None else chunk_races

    out = oof_predict(frame, folds=folds, **kwargs)

    assert calls, "oof_predict must predict through WinCalibration.predict_many"
    assert max(calls) <= limit
    assert sum(calls) == n_test  # every test race in exactly one call: none split, none skipped
    assert len(calls) >= math.ceil(n_test / limit)
    assert out["race_id"].nunique() == n_test
    sums = out.groupby("race_id")["p_cal"].sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------------------------
# JSON round trip
# ---------------------------------------------------------------------------------------------


def test_the_fit_round_trips_through_json(biased_fit):
    built = WinCalibration(
        knots=tuple(biased_fit.knots),
        coef=tuple(biased_fit.coef),
        fit_id="fit-0123456789ab",
        train_start=date(2015, 2, 7),
        train_end=date(2020, 12, 27),
        n_races=4321,
    )
    rng = np.random.default_rng(9)
    probes = [PROBE, *EXTRA_PROBES, rng.dirichlet(np.full(18, 0.8))]
    for original in (biased_fit, built):
        text = original.to_json()
        json.loads(text)  # valid JSON
        back = WinCalibration.from_json(text)
        assert tuple(back.knots) == tuple(original.knots)
        assert tuple(back.coef) == tuple(original.coef)
        assert back.fit_id == original.fit_id
        assert back.train_start == original.train_start
        assert back.train_end == original.train_end
        assert isinstance(back.train_start, date) and isinstance(back.train_end, date)
        assert back.n_races == original.n_races
        for probe in probes:
            assert np.array_equal(back.predict(probe), original.predict(probe))
    assert built.n_races == 4321
    np.testing.assert_allclose(built.predict(PROBE), biased_fit.predict(PROBE), atol=1e-12)


# ---------------------------------------------------------------------------------------------
# run_fit over a synthetic store: dead heats, scratches, the era guard, outputs out of the repo
# ---------------------------------------------------------------------------------------------

RUNNERS_SCHEMA = pa.schema(
    [
        ("race_id", pa.string()),
        ("race_date", pa.date32()),
        ("horse_no", pa.int32()),
        ("win_odds", pa.float64()),
        ("finish_pos", pa.int32()),
        ("is_scratched", pa.bool_()),
        ("dead_heat", pa.bool_()),
        ("win_fav_rank", pa.int32()),
    ]
)
RUN_START, RUN_END = date(2014, 6, 1), date(2021, 6, 30)
FIRST_SCORED = date(2020, 1, 1)  # Japan's scheme: min_train 22 quarters from 2014-06-07
N_FOLDS = 6  # 2020-Q1 .. 2021-Q2
N_RUN = 8


def build_store(rng, every_days, n_dh_first, n_dh_third, n_scratched, n_unpriced):
    """Store rows from RUN_START to RUN_END and the sets of races treated specially."""
    days = [RUN_START + timedelta(days=k) for k in range(0, (RUN_END - RUN_START).days + 1)]
    days = days[::every_days]
    if days[-1] != RUN_END:
        days.append(RUN_END)
    n_races = len(days)
    q = simulate_q(rng, n_races, N_RUN, alpha=1.0, lo=0.015, hi=0.6)
    winner = draw_winners(rng, power_p(q, 1.1))
    pre = np.array([d < date(2014, 6, 7) for d in days])
    winner[pre] = np.argmin(q[pre], axis=1)  # pre-reform races: a pattern that would show
    finish = np.zeros((n_races, N_RUN))
    for r in range(n_races):
        others = rng.permutation(np.setdiff1d(np.arange(N_RUN), [winner[r]]))
        finish[r, winner[r]] = 1
        finish[r, others] = np.arange(2, N_RUN + 1)
    odds = np.maximum(np.round(TAKEOUT / q, 1), 1.1)
    scratched = np.zeros((n_races, N_RUN), dtype=bool)
    dead_heat = np.zeros((n_races, N_RUN), dtype=bool)

    post = np.flatnonzero(~pre)
    picks = rng.choice(post, size=n_dh_first + n_dh_third + n_scratched + n_unpriced, replace=False)
    dh_first = picks[:n_dh_first]
    dh_third = picks[n_dh_first : n_dh_first + n_dh_third]
    scr = picks[n_dh_first + n_dh_third : n_dh_first + n_dh_third + n_scratched]
    unpriced = picks[n_dh_first + n_dh_third + n_scratched :]
    for r in dh_first:  # the two longest shots dead-heat for first
        a, b = np.argsort(q[r])[:2]
        rest = np.setdiff1d(np.arange(N_RUN), [a, b])
        finish[r, [a, b]] = 1
        finish[r, rng.permutation(rest)] = np.arange(3, N_RUN + 1)
        dead_heat[r, [a, b]] = True
    for r in dh_third:  # a dead heat for third: the race is kept
        a, b = np.flatnonzero(finish[r] == 3)[0], np.flatnonzero(finish[r] == 4)[0]
        finish[r, [a, b]] = 3
        dead_heat[r, [a, b]] = True
    for r in scr:  # a scratched runner keeps a stale price in the store
        h = int(np.flatnonzero(finish[r] == N_RUN)[0])
        scratched[r, h] = True
        finish[r, h] = np.nan
    for r in unpriced:  # a runner with no price, not scratched
        h = int(np.flatnonzero(finish[r] == N_RUN)[0])
        odds[r, h] = np.nan

    ids = [f"jra:{d.isoformat()}:HSN:01" for d in days]
    fav_rank = np.argsort(np.argsort(-q, axis=1), axis=1) + 1
    runners = pd.DataFrame(
        {
            "race_id": np.repeat(ids, N_RUN),
            "race_date": np.repeat(np.array(days, dtype=object), N_RUN),
            "horse_no": np.tile(np.arange(1, N_RUN + 1), n_races),
            "win_odds": odds.ravel(),
            "finish_pos": finish.ravel(),
            "is_scratched": scratched.ravel(),
            "dead_heat": dead_heat.ravel(),
            "win_fav_rank": fav_rank.ravel(),
        }
    )
    special = {
        "dh_first": {ids[r] for r in dh_first},
        "dh_third": {ids[r] for r in dh_third},
        "pre": {ids[r] for r in np.flatnonzero(pre)},
    }
    return runners, special


def write_store(data, runners):
    jra = data / "processed" / "jra"
    jra.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pandas(runners, schema=RUNNERS_SCHEMA, preserve_index=False),
        jra / "runners.parquet",
    )


def kept_rows(runners, special):
    """The rows the live fit uses, derived here: post-reform, priced, not scratched, and not in a
    race with a dead heat for first."""
    keep = (
        ~runners["race_id"].isin(special["pre"] | special["dh_first"])
        & ~runners["is_scratched"]
        & runners["win_odds"].notna()
    )
    rows = runners[keep].copy()
    rows["won"] = rows["finish_pos"] == 1
    return rows[["race_id", "race_date", "horse_no", "win_odds", "won"]].reset_index(drop=True)


def repo_outputs():
    """Paths git sees in the repo (untracked or ignored) that look like this ticket's outputs."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "--ignored", "--untracked-files=all"],
        cwd=env.repo_root(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {line for line in out.splitlines() if "win_calibration" in line}


@pytest.fixture(scope="module")
def run_case(tmp_path_factory):
    rng = np.random.default_rng(1230)
    runners, special = build_store(
        rng, every_days=1, n_dh_first=60, n_dh_third=30, n_scratched=50, n_unpriced=20
    )
    data = (tmp_path_factory.mktemp("run") / "data").resolve()
    write_store(data, runners)
    before = repo_outputs()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(DATA_VAR, str(data))
        manifest = run_fit()
    after = repo_outputs()
    return {
        "data": data,
        "runners": runners,
        "special": special,
        "manifest": manifest,
        "new_in_repo": after - before,
    }


def fit_files(out_dir, fit_id):
    return {
        "fit": out_dir / f"win_calibration-{fit_id}.json",
        "oof": out_dir / f"win_calibration_oof-{fit_id}.parquet",
        "manifest": out_dir / f"manifest-win_calibration-{fit_id}.json",
    }


def test_races_with_a_dead_heat_for_first_are_left_out_and_counted(run_case):
    manifest = run_case["manifest"]
    runners, special = run_case["runners"], run_case["special"]
    kept = kept_rows(runners, special)
    n_kept = kept["race_id"].nunique()
    assert manifest["races_left_out"]["dead_heat_for_first"] == len(special["dh_first"]) == 60
    assert manifest["n_races"] == n_kept

    files = fit_files(run_case["data"] / "models" / "japan" / "v0", manifest["fit_id"])
    live = WinCalibration.from_json(files["fit"].read_text())
    assert live.n_races == n_kept
    # The live fit is the fit on the kept rows: a dead heat for third stays in, for first is out.
    reference = fit_win_calibration(kept)
    for probe in (PROBE, *EXTRA_PROBES):
        np.testing.assert_allclose(live.predict(probe), reference.predict(probe), atol=1e-4)


def test_run_fit_reads_from_the_price_era_start_to_the_stores_last_date(run_case):
    manifest = run_case["manifest"]
    files = fit_files(run_case["data"] / "models" / "japan" / "v0", manifest["fit_id"])
    live = WinCalibration.from_json(files["fit"].read_text())
    assert live.train_start == date(2014, 6, 7)
    assert live.train_end == RUN_END
    assert live.fit_id == manifest["fit_id"]


def test_run_fit_writes_out_of_fold_predictions_over_japans_scheme(run_case):
    manifest = run_case["manifest"]
    runners, special = run_case["runners"], run_case["special"]
    files = fit_files(run_case["data"] / "models" / "japan" / "v0", manifest["fit_id"])
    oof = pq.read_table(files["oof"]).to_pandas()
    assert {"race_id", "horse_no", "fold", "q", "p_cal"} <= set(oof.columns)
    assert len(manifest["folds"]) == N_FOLDS

    usable = runners[~runners["is_scratched"] & runners["win_odds"].notna()]
    dates = pd.to_datetime(usable["race_date"])
    scored = usable[dates >= pd.Timestamp(FIRST_SCORED)]
    clean = scored[~scored["race_id"].isin(special["dh_first"])]
    got_keys = set(zip(oof["race_id"], oof["horse_no"].astype(int), strict=True))
    assert set(zip(clean["race_id"], clean["horse_no"].astype(int), strict=True)) <= got_keys
    assert got_keys <= set(zip(scored["race_id"], scored["horse_no"].astype(int), strict=True))

    # q is the market over priced, unscratched runners; fold is the scored quarter's position.
    got = oof.set_index(["race_id", "horse_no"]).sort_index()
    for race_id, g in clean.groupby("race_id"):
        keys = list(zip([race_id] * len(g), g["horse_no"].astype(int), strict=True))
        np.testing.assert_allclose(got.loc[keys, "q"].to_numpy(), ref_q(g["win_odds"]), atol=1e-12)
        d = pd.Timestamp(g["race_date"].iloc[0])
        quarter = (d.year - 2020) * 4 + (d.month - 1) // 3
        assert (got.loc[keys, "fold"] == quarter).all()
    np.testing.assert_allclose(oof.groupby("race_id")["p_cal"].sum().to_numpy(), 1.0, atol=1e-9)


def test_run_fit_writes_its_three_files_under_the_data_dir_and_nothing_in_the_repo(run_case):
    manifest = run_case["manifest"]
    default = run_case["data"] / "models" / "japan" / "v0"
    files = fit_files(default, manifest["fit_id"])
    assert sorted(p.name for p in default.iterdir()) == sorted(p.name for p in files.values())
    assert json.loads(files["manifest"].read_text()) == json.loads(json.dumps(manifest))
    for key in ("fit_id", "git_sha", "n_races", "races_left_out", "folds"):
        assert key in manifest
    assert run_case["new_in_repo"] == set()


@pytest.fixture
def small_store(tmp_path, monkeypatch):
    rng = np.random.default_rng(1231)
    runners, _ = build_store(
        rng, every_days=3, n_dh_first=10, n_dh_third=5, n_scratched=10, n_unpriced=5
    )
    data = (tmp_path / "data").resolve()
    write_store(data, runners)
    monkeypatch.setenv(DATA_VAR, str(data))
    return data


def test_run_fit_writes_to_out_dir_when_given(small_store, tmp_path):
    out_dir = tmp_path / "elsewhere" / "fits"
    manifest = run_fit(out_dir)
    files = fit_files(out_dir, manifest["fit_id"])
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(p.name for p in files.values())
    assert manifest["races_left_out"]["dead_heat_for_first"] == 10
    assert not (small_store / "models").exists() or not list((small_store / "models").rglob("*"))


def test_the_cli_fits_into_out(small_store, tmp_path):
    out_dir = tmp_path / "cli-out"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "horsetrack.countries.japan.models.v0.calibration",
            "--fit",
            "--out",
            str(out_dir),
        ],
        env={**os.environ, DATA_VAR: str(small_store)},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=170,
    )
    assert proc.returncode == 0, proc.stderr
    names = sorted(p.name for p in out_dir.iterdir())
    assert len(names) == 3
    fit_id = names[0].removeprefix("manifest-win_calibration-").removesuffix(".json")
    assert names == sorted(p.name for p in fit_files(Path(out_dir), fit_id).values())


def fresh_name_in_repo():
    """The resolved repo root and a directory name in it that nothing uses."""
    return env.repo_root().resolve(), f".pytest-refused-out-{uuid.uuid4().hex}"


@pytest.mark.parametrize("case", ["absolute", "relative", "dotdot", "symlink"])
def test_run_fit_refuses_an_out_dir_inside_the_repo(small_store, tmp_path, monkeypatch, case):
    root, name = fresh_name_in_repo()
    if case == "absolute":
        out_dir, target = root / name, root / name
    elif case == "relative":
        monkeypatch.chdir(root / "tests")
        out_dir, target = Path(name), root / "tests" / name
    elif case == "dotdot":
        # leaves the repo as written, then comes back into it
        out_dir, target = Path(f"{root}/../{root.name}/{name}"), root / name
    else:
        link = tmp_path / "link-to-repo"
        link.symlink_to(root, target_is_directory=True)
        out_dir, target = link / name, root / name
    before = repo_outputs()
    try:
        with pytest.raises(paths.DataDirInsideRepo):
            run_fit(out_dir)
        assert not target.exists()
        assert repo_outputs() - before == set()
    finally:
        shutil.rmtree(target, ignore_errors=True)


def test_the_cli_refuses_an_out_inside_the_repo(small_store, tmp_path):
    root, name = fresh_name_in_repo()
    target = root / name
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "horsetrack.countries.japan.models.v0.calibration",
                "--fit",
                "--out",
                str(target),
            ],
            env={**os.environ, DATA_VAR: str(small_store)},
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=170,
        )
        assert proc.returncode != 0, "the CLI wrote a fit inside the repo"
        assert not target.exists()
    finally:
        shutil.rmtree(target, ignore_errors=True)
