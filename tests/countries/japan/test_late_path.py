"""The late-money path: E23 (ex post), E24 (ex ante) and E22 (ticket #102, epic #73).

Module under test: `horsetrack.countries.japan.studies.late_path`. Contract pinned here:

The ex-post window: `late_change(captures, *, close_utc, window_min) -> pd.Series`
- `captures` holds one race's WIN captures: `horse_id`, `fetched_utc` (tz-aware UTC), `odds`, in any
  row order. The change of a horse is O_to / O_from - 1, O_from its last capture at or before
  close_utc - window_min (inclusive) and O_to its last interim capture, strictly before close_utc.
  A capture at or after the close is the final board and never enters.
- The Series is indexed by horse_id and lists every horse of `captures`: NaN when the horse has no
  capture at or before close - window or no capture before the close. When the last interim
  capture is the reference capture itself, the change is 0.
- `to="final"` (the paper's ending; `to="interim"`, the default, is the one above): O_from is the
  same reference and O_to the horse's last capture at or after close_utc, its final odds. NaN when
  the horse has no capture at or after the close, or no reference. Any other `to` is a ValueError.

E23, the paper's regression: `paper_regression(frame) -> RegressionResult(delta, se)`
- `frame` has one row per runner: `race_id`, `won` (bool), `odds_final` (decimal, stake
  included), `late_change`. The return at final odds, won * odds_final, is regressed on
  late_change with final-odds controls (the paper's base spec has the final odds linearly; more
  controls are allowed). `delta` is the late_change coefficient and `se` its standard error.
- A runner with a missing late_change is left out: the result is the one on the other runners.

The final-price offset: `offset_probabilities` and `offset_test`
- `frame` has one row per runner: `race_id`, `won`, `q_final` and the path columns. `q_final` is
  the final-price baseline's probability, or any positive per-race multiple of it (1/odds_final
  included): the final price enters as the offset ln q_final, with its coefficient fixed at 1.
- `offset_probabilities(frame, *, coef: Mapping[str, float]) -> pd.Series`, aligned to
  frame.index: softmax over each race of ln q_final + sum_c coef[c] * frame[c]. At zero path
  coefficients it is the de-vigged final.
- `offset_test(frame, *, path_cols, clusters, n_boot, level, seed) -> core.eval.bootstrap.Interval`:
  the per-race winner log-loss gain, in nats, of final + path (the path coefficients fitted)
  against final alone, positive when the path helps (the sign of `core.eval.scoring.gain`), with
  one-sided bootstrap bounds resampling `clusters` (a Series from race_id to its cluster, as in
  `paired_bootstrap`). `n_clusters` is the number of clusters resampled. The same seed gives the
  same Interval. Whether the path coefficients are fitted in sample or cross-fitted is left to the
  module: every test here holds either way.

E24, information at t: `ex_ante_features(frame, *, at_utc, nowcast) -> pd.DataFrame`
- `frame` is one race's rows in `logger.compact.SCHEMA` (a jravan day file's rows: the time of a
  row is its snapshot_utc, fetched_utc is null); pools other than WIN are ignored. Only rows at or
  before at_utc are information at t (a row stamped exactly at t counts, as in C1's `race_view`).
- One row per WIN runner on the board at t, indexed by horse_id (the WIN combination text, "7"),
  with exactly the two features the ticket names, nothing else:
  - `path_change`: the observed path up to t in the paper's 5-minute window, O_t / O_ref - 1, with
    O_t the runner's latest price at or before t and O_ref its latest at or before t - 5 min;
  - `c1_drift`: C1's predicted late drift, the predicted relative change of the runner's WIN odds
    from t to the final, E[O_final / O_t] - 1 = E[s_t / s_final] - 1 over `nowcast.draws` on the
    board at t (C1's `race_view` at tau = post_utc - at_utc), pool WIN.
- `nowcast` is a fitted C1 `Nowcast` or any forecaster with its draws signature. One fitted on the
  race's day or a later day (`nowcast.days`, "YYYY-MM-DD") holds information from after t and is
  refused.

E22: `alpha_by_tau(frame, *, buckets) -> pd.DataFrame`
- `frame` has one row per (race, snapshot, runner): `race_id`, `horse_id`, `tau_min`, `won`,
  `p_model` (the model's win probability) and `q_tau` (the de-vigged board at tau). A case is one
  (race_id, tau_min): its rows are one softmax. A row's bucket is the smallest bucket at or above
  its tau_min (C1's rule).
- One row per bucket: `tau_min` (the bucket), `alpha`, `gw_stat`, `gw_pvalue`, `n_races` (cases in
  the bucket). `alpha` is the weight on ln p_model in the Benter blend softmax(alpha ln p_model +
  beta ln q_tau), beta free, fitted per bucket. `gw_stat` and `gw_pvalue` are a Giacomini-White
  test of that bucket's blend against one fixed blend (alpha, beta) fitted on every bucket pooled.

The study: `path_files(source) -> list[tuple[str, Path]]`
- The development day files of a source, as ("YYYY-MM-DD", path) sorted by day: "jravan" reads
  `processed/jravan/odds_timeseries/<day>.parquet`, "jra-logs" the compacted logs
  `processed/jra/odds_timeseries/<day>.parquet`; any other source is a ValueError. Holdout days
  (D14, `is_holdout_day`) and days before 2014-06-07 (D10: no pre-reform price) are never listed.

The pre-registered spec, `config/late_path.yaml` (owner's answer on #102, 2026-11-28)
- No field is pending (`late_path_study.pending_fields` is empty): E24's features and E22's
  p_model read `status: approved`. `e23.ends` lists both endings of the late change, `interim`
  and `final`, every one a `to` that `late_change` accepts, and `e23.primary_end` is `interim`.

Expected values are derived here from first principles (the stated formulas, numpy least squares
and the planted parameters), never from the module under test. All data are synthetic and in
memory or under tmp; HORSETRACK_DATA_DIR points at a tmp dir.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from horsetrack.core import config, paths
from horsetrack.core.eval.bootstrap import Interval
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.countries.japan.logger.compact import SCHEMA
from horsetrack.countries.japan.models.nowcast import Draws, Nowcast
from horsetrack.countries.japan.studies import late_path, late_path_study

UTC = timezone.utc
COUNTRY = "japan"
WIN = "WIN"
POOL_ROW = "__pool__"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


# ---------------------------------------------------------------- synthetic helpers


def group_softmax(u: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Softmax of `u` within each group (integer codes)."""
    top = np.full(groups.max() + 1, -np.inf)
    np.maximum.at(top, groups, u)
    e = np.exp(u - top[groups])
    return e / np.bincount(groups, e)[groups]


def draw_winners(p: np.ndarray, sizes: np.ndarray, rng) -> np.ndarray:
    """One winner per race drawn from `p` (rows grouped by race, `sizes` runners each)."""
    race = np.repeat(np.arange(len(sizes)), sizes)
    ends = np.cumsum(sizes) - 1
    cum = np.cumsum(p)
    base = np.concatenate([[0.0], cum[ends[:-1]]])
    within = cum - base[race]
    u = rng.random(len(sizes))[race]
    won = ((within - p) < u) & (u <= within)
    missing = np.bincount(race, won, minlength=len(sizes)) == 0  # float round-off at the top
    won[ends[missing]] = True
    assert (np.bincount(race, won) == 1).all()
    return won


def clustered_ols(y: np.ndarray, X: np.ndarray, groups: np.ndarray):
    """OLS coefficients and race-clustered sandwich standard errors."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    scores = np.zeros((groups.max() + 1, X.shape[1]))
    np.add.at(scores, groups, X * resid[:, None])
    cov = bread @ (scores.T @ scores) @ bread
    return beta, np.sqrt(np.diag(cov))


# ---------------------------------------------------------------- the ex-post window


def captures_frame(paths_by_horse: dict[str, list[tuple[float, float]]], close: datetime):
    """Captures as (minutes after the close, odds) per horse; negative minutes are before it."""
    rows = [
        {"horse_id": horse, "fetched_utc": close + timedelta(minutes=m), "odds": float(o)}
        for horse, path in paths_by_horse.items()
        for m, o in path
    ]
    frame = pd.DataFrame(rows)
    frame["fetched_utc"] = pd.to_datetime(frame["fetched_utc"], utc=True)
    return frame


CLOSE = datetime(2026, 10, 4, 5, 23, tzinfo=UTC)


def test_late_change_uses_the_last_interim_before_close():
    """Sales close at T; captures at T-7, T-5, T-3, T-1.5 and a final after the close. The change
    runs from the T-5 capture to the T-1.5 capture, per horse; the final never enters."""
    path = {
        "1": [(-7, 4.0), (-5, 3.6), (-3, 3.2), (-1.5, 2.7), (2, 9.9)],
        "2": [(-7, 12.0), (-5, 15.0), (-3, 16.5), (-1.5, 18.0), (2, 1.1)],
        "3": [(-7, 30.0), (-5, 25.0), (-3, 26.0), (-1.5, 25.0), (2, 99.0)],
    }
    result = late_path.late_change(captures_frame(path, CLOSE), close_utc=CLOSE, window_min=5)
    expected = {"1": 2.7 / 3.6 - 1, "2": 18.0 / 15.0 - 1, "3": 0.0}
    assert set(result.index) == set(expected)
    for horse, value in expected.items():
        assert result.loc[horse] == pytest.approx(value, abs=1e-12), horse


def test_late_change_windows_and_missing_ends():
    """Per horse and per window: the reference is the last capture at or before close - window
    (inclusive), the end the last capture strictly before the close (a capture at the close is
    the final). No reference or no interim capture gives NaN; a reference that is also the last
    interim capture gives 0. Shuffled rows give the same result."""
    path = {
        # every window has its own reference; the capture at the close and the final are ignored
        "A": [(-12, 10.0), (-9, 9.0), (-6, 8.0), (-4, 7.5), (-2.5, 6.0), (-1, 5.0), (0, 1.5),
              (3, 2.0)],
        "B": [(-8, 3.0), (-5.5, 3.3), (-3, 3.0), (-0.5, 2.4), (3, 2.2)],
        "C": [(-4, 20.0), (-1, 22.0), (3, 30.0)],  # first priced inside the window
        "D": [(3, 7.0)],  # only the final, after the close
        "E": [(-10, 6.0), (-6, 6.6), (3, 5.0)],  # nothing after the reference before the close
    }  # fmt: skip
    nan = math.nan
    table = {
        5: {"A": 5.0 / 8.0 - 1, "B": 2.4 / 3.3 - 1, "C": nan, "D": nan, "E": 0.0},
        10: {"A": 5.0 / 10.0 - 1, "B": nan, "C": nan, "D": nan, "E": 6.6 / 6.0 - 1},
        3: {"A": 5.0 / 7.5 - 1, "B": 2.4 / 3.0 - 1, "C": 22.0 / 20.0 - 1, "D": nan, "E": 0.0},
        2.5: {"A": 5.0 / 6.0 - 1, "B": 2.4 / 3.0 - 1, "C": 22.0 / 20.0 - 1, "D": nan, "E": 0.0},
        1: {"A": 0.0, "B": 2.4 / 3.0 - 1, "C": 0.0, "D": nan, "E": 0.0},
    }
    frame = captures_frame(path, CLOSE)
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    for window, expected in table.items():
        for captures in (frame, shuffled):
            result = late_path.late_change(captures, close_utc=CLOSE, window_min=window)
            assert set(result.index) == set(expected), window
            for horse, value in expected.items():
                got = result.loc[horse]
                if math.isnan(value):
                    assert pd.isna(got), (window, horse, got)
                else:
                    assert got == pytest.approx(value, abs=1e-12), (window, horse)


# ---------------------------------------------------------------- the ex-post window to the final


def assert_changes(result: pd.Series, expected: dict[str, float], label) -> None:
    assert set(result.index) == set(expected), label
    for horse, value in expected.items():
        got = result.loc[horse]
        if math.isnan(value):
            assert pd.isna(got), (label, horse, got)
        else:
            assert got == pytest.approx(value, abs=1e-12), (label, horse, got)


def test_late_change_to_final_runs_from_the_reference_to_the_final_odds():
    """The paper's ending (owner, #102): the same T-5 reference, and the change runs to the final
    odds after the close. The default and to="interim" are today's change to the T-1.5 capture."""
    path = {
        "1": [(-7, 4.0), (-5, 3.6), (-3, 3.2), (-1.5, 2.7), (2, 9.9)],
        "2": [(-7, 12.0), (-5, 15.0), (-3, 16.5), (-1.5, 18.0), (2, 1.1)],
        "3": [(-7, 30.0), (-5, 25.0), (-3, 26.0), (-1.5, 25.0), (2, 99.0)],
    }
    captures = captures_frame(path, CLOSE)
    final = late_path.late_change(captures, close_utc=CLOSE, window_min=5, to="final")
    assert_changes(final, {"1": 9.9 / 3.6 - 1, "2": 1.1 / 15.0 - 1, "3": 99.0 / 25.0 - 1}, "final")
    interim = {"1": 2.7 / 3.6 - 1, "2": 18.0 / 15.0 - 1, "3": 0.0}
    for kw in ({}, {"to": "interim"}):
        result = late_path.late_change(captures, close_utc=CLOSE, window_min=5, **kw)
        assert_changes(result, interim, kw)


def test_late_change_ends_by_window_and_missing_ends():
    """Per horse, window (5, 10, 20) and ending: the reference is the last capture at or before
    close - window (inclusive) for both endings; the interim end is the last capture strictly
    before the close, the final end the last capture at or after it (a capture at the close is a
    final; of two after the close, the later). No reference, or no capture at or after the close
    for the final, gives NaN. Shuffled rows give the same result."""
    path = {
        # a reference exactly on close - window at every window; one capture, at the close
        "A": [(-25, 10.0), (-20, 9.0), (-12, 8.0), (-10, 7.5), (-6, 6.0), (-5, 5.5), (-2, 5.0),
              (0, 4.0)],
        "B": [(-21, 2.8), (-11, 3.3), (-6, 3.0), (-1, 2.4), (0, 2.2), (4, 2.6)],  # two finals
        "C": [(-8, 20.0), (-1, 22.0), (3, 30.0)],  # first priced inside the 10 and 20 windows
        "D": [(3, 7.0)],  # only the final: no reference
        "E": [(-30, 6.0), (-12, 6.6), (-4, 6.3)],  # nothing at or after the close: no final
        "F": [(-22, 5.0), (2, 4.0)],  # nothing between the reference and the close
    }  # fmt: skip
    nan = math.nan
    table = {
        ("interim", 5): {"A": 5.0 / 5.5 - 1, "B": 2.4 / 3.0 - 1, "C": 22.0 / 20.0 - 1, "D": nan,
                         "E": 6.3 / 6.6 - 1, "F": 0.0},
        ("interim", 10): {"A": 5.0 / 7.5 - 1, "B": 2.4 / 3.3 - 1, "C": nan, "D": nan,
                          "E": 6.3 / 6.6 - 1, "F": 0.0},
        ("interim", 20): {"A": 5.0 / 9.0 - 1, "B": 2.4 / 2.8 - 1, "C": nan, "D": nan,
                          "E": 6.3 / 6.0 - 1, "F": 0.0},
        ("final", 5): {"A": 4.0 / 5.5 - 1, "B": 2.6 / 3.0 - 1, "C": 30.0 / 20.0 - 1, "D": nan,
                       "E": nan, "F": 4.0 / 5.0 - 1},
        ("final", 10): {"A": 4.0 / 7.5 - 1, "B": 2.6 / 3.3 - 1, "C": nan, "D": nan, "E": nan,
                        "F": 4.0 / 5.0 - 1},
        ("final", 20): {"A": 4.0 / 9.0 - 1, "B": 2.6 / 2.8 - 1, "C": nan, "D": nan, "E": nan,
                        "F": 4.0 / 5.0 - 1},
    }  # fmt: skip
    frame = captures_frame(path, CLOSE)
    shuffled = frame.sample(frac=1.0, random_state=5).reset_index(drop=True)
    for (to, window), expected in table.items():
        for captures in (frame, shuffled):
            result = late_path.late_change(captures, close_utc=CLOSE, window_min=window, to=to)
            assert_changes(result, expected, (to, window))


@pytest.mark.parametrize("to", ["closing", "last", "", "post"])
def test_late_change_refuses_an_unknown_ending(to):
    captures = captures_frame({"1": [(-7, 4.0), (-5, 3.6), (-1.5, 2.7), (2, 9.9)]}, CLOSE)
    with pytest.raises(ValueError):
        late_path.late_change(captures, close_utc=CLOSE, window_min=5, to=to)


# ---------------------------------------------------------------- E23: the paper's regression

RETURN_LEVEL = 0.84  # E[won * odds_final] at zero odds and zero late change
SHARE_FLOOR = 0.025


def paper_frame(n_races: int, delta: float, *, seed: int, flb: float = 0.012) -> pd.DataFrame:
    """Runners whose expected return at final odds is exactly linear: E[won * odds_final |
    odds_final, late_change] = RETURN_LEVEL - flb * odds_final + delta * late_change.

    A race of k runners has shares s (sum 1) and final odds 1 / (book * s) with
    book = (1 + k flb) / RETURN_LEVEL; the late change leans on ln odds (longshots drift out late)
    and is centred so that sum_i s_i change_i = 0 in each race. Then
    p_i = book s_i (RETURN_LEVEL + delta change_i) - flb sums to 1 in each race, and p_i odds_i is
    the line above. The favourite-longshot bias (flb) and the lean make a regression without
    final-odds controls find a delta that is not there."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(8, 17, size=n_races)
    race = np.repeat(np.arange(n_races), sizes)
    g = rng.gamma(2.0, size=race.size)
    s = np.maximum(g / np.bincount(race, g)[race], SHARE_FLOOR)
    s = s / np.bincount(race, s)[race]
    book = (1.0 + sizes * flb) / RETURN_LEVEL
    odds = 1.0 / (book[race] * s)
    lean = -0.3 * np.log(odds / 10.0) + 0.25 * rng.standard_normal(race.size)
    lean = np.clip(lean, -0.8, 0.8)
    change = lean - np.bincount(race, s * lean)[race]
    p = book[race] * s * (RETURN_LEVEL + delta * change) - flb
    assert (p > 0).all()
    np.testing.assert_allclose(np.bincount(race, p), 1.0, rtol=0.0, atol=1e-12)
    won = draw_winners(p, sizes, rng)
    return pd.DataFrame(
        {
            "race_id": pd.Series(race).map("r{:05d}".format),
            "won": won,
            "odds_final": odds,
            "late_change": change,
        }
    )


def reference_fits(frame: pd.DataFrame):
    """(delta, se) of numpy OLS with the final odds as control, and without any control."""
    groups = pd.factorize(frame["race_id"])[0]
    y = (frame["won"].astype(float) * frame["odds_final"]).to_numpy()
    one = np.ones(len(frame))
    x = frame["late_change"].to_numpy()
    b, se = clustered_ols(y, np.c_[one, frame["odds_final"].to_numpy(), x], groups)
    nb, nse = clustered_ols(y, np.c_[one, x], groups)
    return (b[2], se[2]), (nb[1], nse[1])


def assert_recovers(frame: pd.DataFrame, planted: float) -> None:
    result = late_path.paper_regression(frame)
    assert isinstance(result, late_path.RegressionResult)
    assert math.isfinite(result.delta) and math.isfinite(result.se) and result.se > 0
    (_, ref_se), (naive, naive_se) = reference_fits(frame)
    # the fixture has teeth: without final-odds controls the planted delta is not recovered
    assert abs(naive - planted) > 4 * naive_se, (naive, naive_se)
    assert abs(result.delta - planted) < 3 * result.se, (result.delta, result.se, planted)
    assert 0.5 < result.se / ref_se < 2.0, (result.se, ref_se)


def test_paper_regression_recovers_a_planted_delta():
    """delta = -0.36 planted on 6,000 synthetic races (horses that shorten late return more at
    the same final odds) is recovered within 3 SE, with a favourite-longshot bias in the returns
    and late drift leaning on the final odds; the SE is the size a race-clustered OLS gives."""
    assert_recovers(paper_frame(6000, -0.36, seed=1), -0.36)


@pytest.mark.parametrize(("planted", "seed"), [(0.0, 4), (0.3, 4)])
def test_paper_regression_controls_for_the_final_odds(planted, seed):
    """No planted effect (and a positive one): the estimate follows the planted delta, not the
    spurious one a regression without final-odds controls finds on the same runners."""
    assert_recovers(paper_frame(6000, planted, seed=seed), planted)


def test_paper_regression_leaves_out_runners_without_a_late_change():
    """A runner with no late change (no capture in the window) is left out of the regression."""
    frame = paper_frame(2000, -0.36, seed=11)
    rng = np.random.default_rng(12)
    frame.loc[rng.random(len(frame)) < 0.05, "late_change"] = np.nan
    full = late_path.paper_regression(frame)
    kept = late_path.paper_regression(frame.dropna(subset=["late_change"]).reset_index(drop=True))
    assert full.delta == pytest.approx(kept.delta, rel=1e-9, abs=1e-12)
    assert full.se == pytest.approx(kept.se, rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------- the final-price offset


def offset_frame(
    n_races: int,
    *,
    coef: dict[str, float],
    seed: int,
    slope: float = 1.0,
    n_days: int = 30,
    spread: float = 0.25,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Runners with final odds 0.8 / q (q the fair shares), `q_final` = 1 / odds_final (the raw
    implied probability, overround included) and path columns ~ N(0, spread^2). The winner is drawn
    from softmax(slope * ln q + sum_c coef[c] * x_c). Returns the frame, the race -> day clusters
    and the per-race oracle gain ln p_true(winner) - ln q(winner)."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(8, 15, size=n_races)
    race = np.repeat(np.arange(n_races), sizes)
    g = rng.gamma(1.5, size=race.size)
    q = g / np.bincount(race, g)[race]
    odds = 0.8 / q
    columns = {c: spread * rng.standard_normal(race.size) for c in coef}
    u = slope * np.log(q) + sum(coef[c] * columns[c] for c in coef)
    p = group_softmax(u, race)
    won = draw_winners(p, sizes, rng)
    race_ids = pd.Series(race).map("r{:05d}".format)
    frame = pd.DataFrame({"race_id": race_ids, "won": won, "q_final": 1.0 / odds, **columns})
    ids = [f"r{i:05d}" for i in range(n_races)]
    clusters = pd.Series([f"d{i % n_days:02d}" for i in range(n_races)], index=ids)
    oracle = np.log(p[won]) - np.log(q[won])
    return frame, clusters, oracle


def test_final_price_offset_is_applied():
    """With zero path coefficients the fitted probabilities equal the de-vigged final (1/odds
    normalised over each race, whatever its overround); with others, the softmax of ln q_final
    plus the path terms, derived here per race."""
    frame, _, _ = offset_frame(40, coef={"late_change": 0.0, "c1_drift": 0.0}, seed=5)
    frame.loc[frame["race_id"] == "r00003", "q_final"] *= 1.6  # another overround, same de-vig
    frame = frame.sample(frac=1.0, random_state=1)  # races need not be contiguous
    devig = frame["q_final"] / frame.groupby("race_id")["q_final"].transform("sum")
    zero = late_path.offset_probabilities(frame, coef={"late_change": 0.0, "c1_drift": 0.0})
    assert zero.index.equals(frame.index)
    np.testing.assert_allclose(zero.to_numpy(), devig.to_numpy(), rtol=1e-12, atol=1e-15)
    for coef in ({"late_change": -0.45}, {"late_change": 1.2, "c1_drift": -0.8}):
        u = np.log(frame["q_final"]) + sum(v * frame[c] for c, v in coef.items())
        e = np.exp(u)
        expected = e / e.groupby(frame["race_id"]).transform("sum")
        got = late_path.offset_probabilities(frame, coef=coef)
        np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1e-10, atol=1e-15)
        assert not np.allclose(got.to_numpy(), devig.to_numpy(), rtol=1e-3)


@pytest.mark.parametrize(
    "coef", [{"late_change": 1.5}, {"late_change": 0.8}, {"late_change": 1.2, "c1_drift": -0.8}]
)
def test_offset_test_finds_a_planted_path_effect(coef):
    """A path term planted on top of the final price: the gain's lower bound is above 0 and the
    gain is the oracle's (the true path coefficients on the same races) within 25%."""
    frame, clusters, oracle = offset_frame(1500, coef=coef, seed=21)
    result = late_path.offset_test(
        frame, path_cols=list(coef), clusters=clusters, n_boot=2000, level=0.95, seed=7
    )
    assert isinstance(result, Interval)
    assert result.lo > 0, result
    assert result.lo <= result.mean <= result.hi
    assert result.mean == pytest.approx(oracle.mean(), rel=0.25), (result, oracle.mean())
    assert result.n_clusters == 30


def test_offset_test_finds_no_gain_on_a_null_path():
    """The path is noise and the final price carries a favourite-longshot bias (truth: slope 1.3
    on ln q). The final price is an offset, so the bias is not the path's to claim: no gain."""
    frame, clusters, _ = offset_frame(1500, coef={"late_change": 0.0}, seed=31, slope=1.3)
    # the fixture has teeth: a free slope on ln q_final would gain this much (the true slope)
    race = pd.factorize(frame["race_id"])[0]
    ln_q = np.log(frame["q_final"].to_numpy())
    p_slope = group_softmax(1.3 * ln_q, race)
    q = group_softmax(ln_q, race)
    won = frame["won"].to_numpy()
    assert (np.log(p_slope[won]) - np.log(q[won])).mean() > 0.01
    result = late_path.offset_test(
        frame, path_cols=["late_change"], clusters=clusters, n_boot=2000, level=0.95, seed=7
    )
    assert result.lo <= 0, result
    assert abs(result.mean) < 0.003, result


def test_offset_test_is_reproducible_and_clustered():
    """The same seed gives the same Interval, another seed other bounds; the bootstrap resamples
    the clusters it is given (days here, then races)."""
    frame, by_day, _ = offset_frame(600, coef={"late_change": 1.5}, seed=41, n_days=12)
    kw = {"path_cols": ["late_change"], "n_boot": 1000, "level": 0.95}
    first = late_path.offset_test(frame, clusters=by_day, seed=3, **kw)
    again = late_path.offset_test(frame, clusters=by_day, seed=3, **kw)
    other = late_path.offset_test(frame, clusters=by_day, seed=4, **kw)
    assert first == again
    assert (first.lo, first.hi) != (other.lo, other.hi)
    assert first.n_clusters == 12
    by_race = pd.Series(by_day.index, index=by_day.index)
    assert late_path.offset_test(frame, clusters=by_race, seed=3, **kw).n_clusters == 600


# ---------------------------------------------------------------- E24: information at t

EX_DAY = date(2026, 10, 4)
EX_RACE = "jra:2026-10-04:KYO:05"
EX_POST = datetime(2026, 10, 4, 5, 25, tzinfo=UTC)
EX_RUNNERS = tuple(str(j) for j in range(1, 9))
EX_WINNER = "4"
EX_MINUTES = (900.0, 60.0, 30.0, 21.0, 16.0, 12.0, 9.0, 7.0, 5.0, 3.5, 2.0, 1.0)
EX_FINAL = -1.5  # the post-race final board
WINDOW_MIN = 5.0


def ex_ante_odds(seed: int = 102) -> dict[float, np.ndarray]:
    """WIN odds per snapshot (minutes to post), in EX_RUNNERS order: a random walk of the shares."""
    rng = np.random.default_rng(seed)
    ln_s = np.log(rng.dirichlet(np.full(len(EX_RUNNERS), 2.0)))
    out = {}
    for minutes in (*EX_MINUTES, EX_FINAL):
        ln_s = ln_s + 0.15 * rng.standard_normal(len(EX_RUNNERS))
        s = np.exp(ln_s) / np.exp(ln_s).sum()
        out[minutes] = 0.8 / s
    return out


def compact_row(pool, combination, minutes, *, odds=None, investment=None, final=False,
                dividend=None) -> dict:  # fmt: skip
    """One row in logger.compact.SCHEMA, jravan-style: no fetch time, a snapshot at its minute."""
    return {
        "race_id": EX_RACE,
        "venue": "KYO",
        "race_no": 5,
        "pool": pool,
        "combination": combination,
        "snapshot_utc": EX_POST - timedelta(minutes=minutes),
        "fetched_utc": None,
        "post_utc": EX_POST,
        "minutes_to_post": float(minutes),
        "odds": None if odds is None else float(odds),
        "odds_hi": None,
        "investment": None if investment is None else float(investment),
        "status_raw": None,
        "is_final_capture": bool(final),
        "dividend": None if dividend is None else float(dividend),
        "is_refund": None if dividend is None else False,
    }


def ex_ante_frame(odds: dict[float, np.ndarray]) -> pd.DataFrame:
    rows = []
    for minutes, board in odds.items():
        final = minutes == EX_FINAL
        for runner, price in zip(EX_RUNNERS, board, strict=True):
            dividend = round(price * 100) if final and runner == EX_WINNER else None
            rows.append(compact_row(WIN, runner, minutes, odds=price, final=final,
                                    dividend=dividend))  # fmt: skip
        total = 1e8 * math.exp(-0.04 * max(minutes, 0.0))
        rows.append(compact_row(WIN, POOL_ROW, minutes, investment=total, final=final))
        rows.append(compact_row("QUINELLA", "1,2", minutes, odds=9.9 + minutes, final=final))
    return pa.Table.from_pylist(rows, schema=SCHEMA).to_pandas()


class PowerNowcast:
    """A deterministic stand-in with C1's draws signature: every draw is s_t ** 0.8 renormalised,
    so the expected drift is known exactly. Fitted on no day."""

    days: tuple[str, ...] = ()

    def draws(self, view, *, pool, n, seed):
        s = np.asarray(view.pools[pool].shares, dtype=float)
        s = s / s.sum()
        target = s**0.8 / (s**0.8).sum()
        total = view.pools[pool].total or 1e8
        return Draws(shares=np.tile(target, (n, 1)), v_final=np.full(n, total / 0.9))


def latest_minute(at_min: float) -> float:
    """The snapshot (minutes to post) of the latest WIN board at or before post - at_min."""
    return min(m for m in EX_MINUTES if m >= at_min)


def expected_features(odds: dict[float, np.ndarray], tau: float) -> pd.DataFrame:
    o_t = odds[latest_minute(tau)]
    o_ref = odds[latest_minute(tau + WINDOW_MIN)]
    s_t = (1.0 / o_t) / (1.0 / o_t).sum()
    target = s_t**0.8 / (s_t**0.8).sum()
    return pd.DataFrame(
        {"path_change": o_t / o_ref - 1.0, "c1_drift": s_t / target - 1.0},
        index=pd.Index(EX_RUNNERS, name="horse_id"),
    )


def features(frame: pd.DataFrame, at: datetime, nowcast=None) -> pd.DataFrame:
    got = late_path.ex_ante_features(frame, at_utc=at, nowcast=nowcast or PowerNowcast())
    assert set(got.columns) == {"path_change", "c1_drift"}, list(got.columns)
    assert set(got.index) == set(EX_RUNNERS)
    return got.loc[list(EX_RUNNERS), ["path_change", "c1_drift"]].astype(float)


def assert_features(got: pd.DataFrame, expected: pd.DataFrame) -> None:
    for column in ("path_change", "c1_drift"):
        np.testing.assert_allclose(
            got[column].to_numpy(), expected[column].to_numpy(), rtol=1e-9, atol=1e-12
        )


def after(frame: pd.DataFrame, at: datetime) -> pd.Series:
    return frame["snapshot_utc"] > pd.Timestamp(at)


@pytest.mark.parametrize("tau", [16.0, 5.0, 2.0])
def test_ex_ante_features_use_only_information_at_t(tau):
    """At t = post - tau the features are the observed path up to t and C1's predicted drift from
    the board at t (a row stamped exactly at t counts). Altering, removing or adding any row after
    t, the final board and the result included, leaves them unchanged; changing a price at t
    changes them as derived."""
    odds = ex_ante_odds()
    frame = ex_ante_frame(odds)
    at = EX_POST - timedelta(minutes=tau)
    base = features(frame, at)
    assert_features(base, expected_features(odds, tau))

    rng = np.random.default_rng(int(tau * 10))
    later = after(frame, at)
    assert later.any() and (frame.loc[later, "is_final_capture"]).any()

    altered = frame.copy()
    altered.loc[later, "odds"] *= rng.uniform(0.3, 3.0, size=int(later.sum()))
    altered.loc[later, "investment"] *= 1.7
    final_win = altered["is_final_capture"] & (altered["pool"] == WIN)
    altered.loc[final_win, "dividend"] = np.nan
    altered.loc[final_win & (altered["combination"] == "7"), "dividend"] = 4321.0  # another winner
    removed = frame[~later].reset_index(drop=True)
    extra = pd.DataFrame(
        [compact_row(WIN, r, tau - 0.5, odds=rng.uniform(1.5, 80.0)) for r in EX_RUNNERS]
        + [compact_row(WIN, POOL_ROW, tau - 0.5, investment=5e9)]
    )
    added = pd.concat([frame, extra.astype(frame.dtypes.to_dict())], ignore_index=True)
    shuffled = frame.sample(frac=1.0, random_state=9).reset_index(drop=True)
    for variant in (altered, removed, added, shuffled):
        pd.testing.assert_frame_equal(features(variant, at), base)

    moved = {m: board.copy() for m, board in odds.items()}
    moved[latest_minute(tau)][2] *= 1.5  # runner "3" at the board on t
    at_t = frame.copy()
    row_t = frame["snapshot_utc"] == pd.Timestamp(EX_POST - timedelta(minutes=latest_minute(tau)))
    at_t.loc[row_t & (frame["combination"] == "3"), "odds"] *= 1.5
    assert_features(features(at_t, at), expected_features(moved, tau))


def test_ex_ante_drift_comes_from_the_c1_nowcast():
    """With C1's own known nowcast at phi = 1 every draw is the board at t, so the predicted drift
    is 0 for every runner while the observed path is unchanged."""
    odds = ex_ante_odds()
    frame = ex_ante_frame(odds)
    at = EX_POST - timedelta(minutes=2.0)
    got = features(frame, at, Nowcast.from_params(phi=1.0, sigma=0.4))
    np.testing.assert_allclose(got["c1_drift"].to_numpy(), 0.0, rtol=0.0, atol=1e-12)
    expected = expected_features(odds, 2.0)
    np.testing.assert_allclose(
        got["path_change"].to_numpy(), expected["path_change"].to_numpy(), rtol=1e-9
    )


def test_ex_ante_features_refuse_a_nowcast_fitted_on_the_race_day_or_later():
    """A C1 fitted on the race's day (or after it) has seen this race's final: its drift is not
    information at t. One fitted on earlier days only is accepted."""
    frame = ex_ante_frame(ex_ante_odds())
    at = EX_POST - timedelta(minutes=2.0)
    known = Nowcast.from_params(phi=1.0, sigma=0.4)
    for days in (("2026-09-27", EX_DAY.isoformat()), ("2026-10-11",)):
        with pytest.raises((ValueError, RuntimeError)):
            late_path.ex_ante_features(
                frame, at_utc=at, nowcast=dataclasses.replace(known, days=days)
            )
    earlier = dataclasses.replace(known, days=("2026-09-27", "2026-10-03"))
    got = features(frame, at, earlier)
    np.testing.assert_allclose(got["c1_drift"].to_numpy(), 0.0, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------- E22: the blend weight by tau


def blend_frame(planted: dict[float, tuple[tuple[float, ...], float, float]], *, n_races: int,
                seed: int) -> pd.DataFrame:  # fmt: skip
    """Each race is seen once per bucket (the same race ids, runners and model probabilities), at
    a tau drawn from the bucket's taus, with its own board q_tau. Each (race, tau) case draws its
    winner from softmax(alpha ln p_model + beta ln q_tau), (alpha, beta) planted per bucket."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(8, 15, size=n_races)
    race = np.repeat(np.arange(n_races), sizes)
    strength = rng.standard_normal(race.size)
    ln_p = np.log(group_softmax(strength + 0.8 * rng.standard_normal(race.size), race))
    race_ids = pd.Series(race).map("r{:05d}".format).to_numpy()
    horse_ids = np.concatenate([np.arange(1, k + 1) for k in sizes]).astype(str)
    parts = []
    for taus, alpha, beta in planted.values():
        ln_q = np.log(group_softmax(strength + 0.8 * rng.standard_normal(race.size), race))
        won = draw_winners(group_softmax(alpha * ln_p + beta * ln_q, race), sizes, rng)
        tau = rng.choice(np.asarray(taus, dtype=float), size=n_races)[race]
        parts.append(
            pd.DataFrame(
                {
                    "race_id": race_ids,
                    "horse_id": horse_ids,
                    "tau_min": tau,
                    "won": won,
                    "p_model": np.exp(ln_p),
                    "q_tau": np.exp(ln_q),
                }
            )
        )
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)


def by_bucket(result: pd.DataFrame, buckets: list[float]) -> pd.DataFrame:
    assert len(result) == len(buckets)
    assert {"tau_min", "alpha", "gw_stat", "gw_pvalue", "n_races"} <= set(result.columns)
    out = result.set_index(result["tau_min"].astype(float))
    assert set(out.index) == set(buckets)
    return out


def test_alpha_by_tau_recovers_a_planted_weight_per_bucket():
    """The weight on the model falls toward the post: alpha 0.5, 0.3 and 0.14 planted in the
    buckets 16, 5 and 2, with beta free (1.0, 1.1, 1.2: a convex blend, beta = 1 - alpha, misses
    alpha by more than 0.1 here), and off-grid taus binned to the smallest bucket at or above
    them. Each bucket's alpha is recovered within 0.1 (about 4 SE) and counts its 2,000 cases."""
    planted = {
        16.0: ((16.0, 12.0, 8.0), 0.5, 1.0),
        5.0: ((5.0, 4.0, 3.0), 0.3, 1.1),
        2.0: ((2.0, 1.5), 0.14, 1.2),
    }
    buckets = [16.0, 5.0, 2.0]
    result = by_bucket(
        late_path.alpha_by_tau(blend_frame(planted, n_races=2000, seed=51), buckets=buckets),
        buckets,
    )
    for bucket, (_, alpha, _) in planted.items():
        assert result.loc[bucket, "alpha"] == pytest.approx(alpha, abs=0.1), bucket
        assert int(result.loc[bucket, "n_races"]) == 2000
        assert 0.0 <= result.loc[bucket, "gw_pvalue"] <= 1.0
        assert math.isfinite(result.loc[bucket, "gw_stat"])


@pytest.mark.parametrize(
    ("weights", "rejects"),
    [(((0.8, 0.3), (0.0, 1.0)), True), (((0.3, 0.7), (0.3, 0.7)), False)],
    ids=["weights-differ", "weights-equal"],
)
def test_alpha_by_tau_gw_rejects_a_fixed_blend_only_when_weights_differ(weights, rejects):
    """Giacomini-White per bucket against one blend fitted on both buckets: very small p-values
    when the planted weights differ (0.8 and 0.0), none below 1e-3 when they are equal."""
    (a1, b1), (a2, b2) = weights
    planted = {30.0: ((30.0,), a1, b1), 2.0: ((2.0,), a2, b2)}
    buckets = [30.0, 2.0]
    result = by_bucket(
        late_path.alpha_by_tau(blend_frame(planted, n_races=2000, seed=61), buckets=buckets),
        buckets,
    )
    for bucket in buckets:
        p = float(result.loc[bucket, "gw_pvalue"])
        assert (p < 1e-4) if rejects else (p > 1e-3), (bucket, p)


# ---------------------------------------------------------------- the study's inputs


def first_days(want_holdout: bool, n: int) -> list[date]:
    out, d = [], date(2026, 11, 28)
    while len(out) < n:
        if is_holdout_day(COUNTRY, d) == want_holdout:
            out.append(d)
        d += timedelta(days=1)
    return out


@pytest.mark.parametrize(("source", "folder"), [("jravan", "jravan"), ("jra-logs", "jra")])
def test_path_files_lists_development_days_of_the_source(data_dir, source, folder):
    """Each source reads its own odds_timeseries folder. Holdout days (D14) and days before
    2014-06-07 (D10) are never listed, nor files that are not day files; the other source's
    days are not this source's."""
    dev = [date(2014, 6, 7), date(2019, 3, 2), date(2026, 11, 22), *first_days(False, 2)]
    held = first_days(True, 2)
    early = [date(2013, 12, 28), date(2014, 6, 1)]
    other = {"jravan": "jra", "jra": "jravan"}[folder]
    base = data_dir / "processed" / folder / "odds_timeseries"
    base.mkdir(parents=True)
    for d in dev + held + early:
        (base / f"{d.isoformat()}.parquet").write_bytes(b"not read by a listing\n")
    (base / "_build.json").write_text("{}")
    elsewhere = data_dir / "processed" / other / "odds_timeseries"
    elsewhere.mkdir(parents=True)
    (elsewhere / "2020-01-05.parquet").write_bytes(b"another source\n")

    listed = late_path.path_files(source)
    assert [day for day, _ in listed] == sorted(d.isoformat() for d in dev)
    for day, path in listed:
        assert path == base / f"{day}.parquet"
    with pytest.raises(ValueError):
        late_path.path_files("netkeiba")


# ---------------------------------------------------------------- the pre-registered spec


def test_late_path_spec_is_approved_and_names_both_e23_endings():
    """The owner's answer on #102 (2026-11-28): no field is pending, so the full run may go; E24's
    features and E22's p_model are approved as proposed; E23 reports the late change to the last
    interim (primary) and to the final odds, at 5 (primary), 10 and 20 minutes."""
    plan = config.load_country_yaml(COUNTRY, "late_path")
    assert late_path_study.pending_fields(plan) == []
    features, p_model = plan["e24"]["features"], plan["e22"]["p_model"]
    assert features["status"] == "approved" and p_model["status"] == "approved"
    assert list(features["columns"]) == ["path_change", "c1_drift"]
    assert p_model["source"] == "c2_rating_oof"

    e23 = plan["e23"]
    assert sorted(e23["ends"]) == ["final", "interim"]
    assert e23["primary_end"] == "interim"
    assert sorted(e23["windows_min"]) == [5, 10, 20] and e23["primary_window_min"] == 5
    captures = captures_frame({"1": [(-7, 4.0), (-5, 3.6), (-1.5, 2.7), (2, 9.9)]}, CLOSE)
    expected = {"interim": 2.7 / 3.6 - 1, "final": 9.9 / 3.6 - 1}
    for end in e23["ends"]:  # every ending the spec names is one late_change computes
        got = late_path.late_change(captures, close_utc=CLOSE, window_min=5, to=end)
        assert got.loc["1"] == pytest.approx(expected[end], abs=1e-12), end
