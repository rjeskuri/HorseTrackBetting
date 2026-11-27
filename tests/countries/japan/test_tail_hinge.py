"""The hinge tail slope beside #285's tail beta (ticket #294; epic #72).

Contract pinned here (`horsetrack.countries.japan.studies.tail_hinge`, a sibling of
`studies.tail_beta` and `studies.calibration`, both of which stay untouched). The names mirror
`tail_beta`'s:
- `fit_tail_hinge(odds, race_id, won, *, min_odds=None) -> TailHingeFit`: the conditional-logit MLE
  of P(i wins) proportional to exp(beta * x_i + gamma * h_i), x = ln(1 / O), the knot
  k = ln(1 / min_odds) and the hinge h = max(0, k - x) (the distance past the knot, 0 at odds
  <= min_odds). The log-weight is continuous at O = min_odds (None: `tail_min_odds` of
  config/calibration.yaml, 51.0). Inputs are pandas Series on one index with
  `calibration.fit_power_beta`'s handling: odds go through `core.pricing.clean_odds` (a missing,
  non-finite or sub-1.0 figure is an unpriced runner and leaves its race's choice set), only races
  with a priced winner enter, a dead heat counts each priced co-winner.
- `TailHingeFit` carries `beta` (the body slope on x) and `se`; `gamma` (the coefficient of the
  hinge) and `gamma_se`; `tail_beta` (the slope on x past the knot, = beta - gamma, because
  gamma * (k - x) = gamma * k - gamma * x) and `tail_beta_se` (from the full covariance,
  var b + var g - 2 cov); and `n_races`. Every standard error is the square root of the diagonal
  of the inverse observed information at the MLE.
- No fitted runner priced strictly above `min_odds` (hinge identically 0): ValueError.
- `devig_hinge(odds, race_id, fit, *, min_odds) -> pd.Series`: p_i proportional to
  exp(beta x_i + gamma h_i) over each race's priced runners, NaN where unpriced.
- `build_spec(owner, scheme, bootstrap, floor) -> dict` and `FAMILY_ID`: a new family (not #77's,
  not #285's) with a plain-JSON spec frozen in the run registry before any scoring; the threshold,
  the fold scheme, the bootstrap and the comparators (`normalised`, `power`, `tail_beta`) are in
  it, `ticket` is 294, `decisions` and `owner_signoff` carry the owner decisions used.
- `compute_study(frame, spec) -> calibration.Study`, pure, on `development_frame`'s shape. Per
  subset (`all_races`; `no_floor_bound_races`: races holding a runner at or under the floor 1.1
  left out of fitting and scoring alike) a walk-forward variant `tail_hinge:walk_forward:<subset>`
  and an in-sample variant `tail_hinge:by_year:<subset>`. Metrics: `body_beta`, `hinge_gamma`,
  `tail_beta` with 95% Wald CIs, in sample overall (no suffix) and by calendar year
  (`[year=YYYY]`), per fold on training rows (`[test_year=YYYY]`); and out of fold
  `delta_ll_vs_normalised`, `delta_ll_vs_power`, `delta_ll_vs_tail_beta` (the comparator's winner
  log loss minus the hinge model's, positive: the hinge model is better; the comparators are
  normalised, power-beta and #285's two-slope tail beta, each refitted on the fold's training rows)
  with meeting-clustered bounds.
- `render_report(study, spec, about) -> str`, `run_study(out_dir=None) -> dict` (keys as
  `tail_beta.run_study`): the family is frozen before `calibration.read_development` is called,
  development rows only, one registry record per variant, the report through `write_report`.

Expected values come from an independent reference below (a two-parameter Newton solver with the
score and information computed from first principles), never from the code under test. Every
frame is synthetic and every RNG seeded: no real data and no holdout.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import dataclasses
import json
import math
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horsetrack.core import env, paths
from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval import holdout
from horsetrack.core.eval.folds import FoldScheme, walk_forward
from horsetrack.core.eval.registry import FamilyAlreadyFrozen, default_registry
from horsetrack.countries.japan.studies import calibration, tail_beta
from horsetrack.countries.japan.studies import tail_hinge as module
from horsetrack.countries.japan.studies.tail_hinge import (
    FAMILY_ID,
    TailHingeFit,
    build_spec,
    compute_study,
    devig_hinge,
    fit_tail_hinge,
    render_report,
    run_study,
)

NAN = float("nan")
TAKE = 0.8
THRESHOLD = 51.0
KNOT = -math.log(THRESHOLD)  # ln(1 / 51)
N_RACES = 12_000
WALD_Z = 1.959963984540054

# ---------------------------------------------------------------------------------------------
# Synthetic markets and the reference fit
# ---------------------------------------------------------------------------------------------


def market_odds(rng, n_races, n_max, n_min, spread, floor_every=0):
    """(n_races, n_max) final Win odds rounded to 0.1 (floor 1.0), NaN for absent runners. With
    `floor_every`, every that-many-th race's favourite is quoted exactly 1.1."""
    strength = rng.standard_normal((n_races, n_max)) * spread
    size = rng.integers(n_min, n_max + 1, size=n_races)
    absent = np.arange(n_max)[None, :] >= size[:, None]
    s = np.where(absent, 0.0, np.exp(strength))
    q = s / s.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        odds = np.maximum(np.round(TAKE / q, 1), 1.0)
    if floor_every:
        rows = np.arange(0, n_races, floor_every)
        fav = np.where(absent, np.inf, odds).argmin(axis=1)[rows]
        odds[rows, fav] = 1.1
    odds[absent] = NAN
    return odds


def log_weight(odds, body, gamma, step=0.0, min_odds=THRESHOLD):
    """beta x + gamma max(0, k - x), plus a level step (a constant below the log-weight of every
    runner priced strictly above min_odds: no slope change)."""
    x = -np.log(odds)
    return body * x + gamma * np.maximum(0.0, -math.log(min_odds) - x) + step * (odds > min_odds)


def draw_winners(rng, odds, body, gamma, step=0.0, min_odds=THRESHOLD):
    z = log_weight(odds, body, gamma, step, min_odds)
    e = np.exp(np.nan_to_num(z - np.nanmax(z, axis=1, keepdims=True), nan=-np.inf))
    p = e / e.sum(axis=1, keepdims=True)
    u = rng.random(len(p))[:, None]
    return np.minimum((np.cumsum(p, axis=1) < u).sum(axis=1), p.shape[1] - 1)


def long_rows(odds, winners, prefix="r"):
    n_races, n_max = odds.shape
    won = np.arange(n_max)[None, :] == np.asarray(winners)[:, None]
    frame = pd.DataFrame(
        {
            "race_id": np.repeat([f"{prefix}{k:05d}" for k in range(n_races)], n_max),
            "k": np.repeat(np.arange(n_races), n_max),
            "odds": odds.ravel(),
            "won": won.ravel(),
        }
    )
    return frame[~np.isnan(odds.ravel())].reset_index(drop=True)


def planted(seed, n_races, body, gamma, *, step=0.0, n_max=16, n_min=8, spread=1.4):
    rng = np.random.default_rng(seed)
    odds = market_odds(rng, n_races, n_max, n_min, spread)
    winners = draw_winners(rng, odds, body, gamma, step)
    return long_rows(odds, winners).drop(columns="k"), rng


def shuffled(frame, rng):
    """Rows shuffled across races, on a shuffled, non-contiguous integer index."""
    frame = frame.iloc[rng.permutation(len(frame))].copy()
    frame.index = pd.Index(7 * rng.permutation(len(frame)) + 101)
    return frame


def hinge_column(x, o, min_odds):
    return np.maximum(0.0, -math.log(min_odds) - x)


def slope_column(x, o, min_odds):
    """#285's second column: x on the runners priced strictly above min_odds."""
    return x * (o > min_odds)


def design(frame, column, min_odds):
    """Runner rows of races with a priced winner, sorted by race: X = [x, column(x, odds)]."""
    odds = frame["odds"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(odds) & (odds >= 1.0)
    d = frame.loc[ok]
    d = d[d.groupby("race_id")["won"].transform("any").to_numpy()]
    d = d.iloc[np.argsort(pd.factorize(d["race_id"])[0], kind="stable")]
    o = d["odds"].to_numpy(dtype=float)
    x = -np.log(o)
    codes = pd.factorize(d["race_id"])[0]
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    won = d["won"].to_numpy(dtype=bool)
    winners = np.add.reduceat(won.astype(float), starts)
    return np.column_stack([x, column(x, o, min_odds)]), won, starts, winners


def loglik_score_info(frame, theta, column, min_odds):
    X, won, starts, winners = design(frame, column, min_odds)
    sizes = np.diff(np.r_[starts, len(X)])
    s = X @ theta
    top = np.maximum.reduceat(s, starts)
    e = np.exp(s - np.repeat(top, sizes))
    tot = np.add.reduceat(e, starts)
    mean = np.add.reduceat(e[:, None] * X, starts) / tot[:, None]
    second = np.add.reduceat(e[:, None, None] * X[:, :, None] * X[:, None, :], starts)
    cov = second / tot[:, None, None] - mean[:, :, None] * mean[:, None, :]
    ll = float(s[won].sum() - (winners * (np.log(tot) + top)).sum())
    score = X[won].sum(axis=0) - (winners[:, None] * mean).sum(axis=0)
    info = (winners[:, None, None] * cov).sum(axis=0)
    return ll, score, info, len(starts)


def ref_fit(frame, min_odds=THRESHOLD, column=hinge_column):
    """The two-slope conditional-logit MLE by damped Newton: (theta, covariance, n_races)."""
    theta = np.array([1.0, 0.0])
    ll, score, info, n = loglik_score_info(frame, theta, column, min_odds)
    for _ in range(100):
        step = np.linalg.solve(info, score)
        while True:
            trial = loglik_score_info(frame, theta + step, column, min_odds)
            if trial[0] >= ll or np.abs(step).max() < 1e-12:
                break
            step = step / 2.0
        theta = theta + step
        ll, score, info, n = trial
        if np.abs(step).max() < 1e-10:
            break
    return theta, np.linalg.inv(info), n


def tail_beta_of(theta, cov):
    """(slope past the knot, its se) of the hinge parameters: beta - gamma."""
    return theta[0] - theta[1], math.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])


def assert_is_the_mle(fit, frame, min_odds=THRESHOLD):
    """The fit sits within a Newton step of 1e-4 of the root of the score, its standard errors
    are the inverse information's, and the derived fields agree."""
    theta = np.array([fit.beta, fit.gamma])
    _, score, info, n = loglik_score_info(frame, theta, hinge_column, min_odds)
    assert np.abs(np.linalg.solve(info, score)).max() < 1e-4
    cov = np.linalg.inv(info)
    assert fit.n_races == n
    assert fit.se == pytest.approx(math.sqrt(cov[0, 0]), rel=1e-3)
    assert fit.gamma_se == pytest.approx(math.sqrt(cov[1, 1]), rel=1e-3)
    assert fit.tail_beta == pytest.approx(fit.beta - fit.gamma, abs=1e-12)
    assert fit.tail_beta_se == pytest.approx(tail_beta_of(theta, cov)[1], rel=1e-3)


def fit_of(frame, **kwargs):
    return fit_tail_hinge(frame["odds"], frame["race_id"], frame["won"], **kwargs)


# ---------------------------------------------------------------------------------------------
# The hinge fit
# ---------------------------------------------------------------------------------------------


def test_the_market_has_a_real_tail():
    """Guard on the fixtures: enough runners above 51.0, and some win, for the fit to bite."""
    frame, _ = planted(3000, N_RACES, 1.0, 0.3)
    above = frame["odds"] > THRESHOLD
    assert above.mean() > 0.08
    assert frame.loc[above, "won"].sum() > 60


@pytest.mark.parametrize(
    ("seed", "body", "gamma"),
    [(3000, 1.0, 0.3), (3001, 0.9, -0.25), (3002, 1.15, 0.6)],
)
def test_recovers_each_planted_pair_of_slopes_within_three_se(seed, body, gamma):
    frame, rng = planted(seed, N_RACES, body, gamma)
    frame = shuffled(frame, rng)

    fit = fit_of(frame, min_odds=THRESHOLD)

    assert isinstance(fit, TailHingeFit)
    assert fit.n_races == N_RACES
    assert 0 < fit.se < 0.06
    assert 0 < fit.gamma_se < 0.25
    assert abs(fit.beta - body) < 3 * fit.se
    assert abs(fit.gamma - gamma) < 3 * fit.gamma_se
    # The slope past 51.0 on x is beta - gamma: the planted hinge adds -gamma there.
    assert abs(fit.tail_beta - (body - gamma)) < 3 * fit.tail_beta_se
    assert_is_the_mle(fit, frame)


@pytest.mark.parametrize(("seed", "body"), [(3010, 1.0), (3011, 0.9), (3012, 1.1)])
def test_a_one_slope_market_gives_a_hinge_increment_whose_ci_covers_zero(seed, body):
    frame, rng = planted(seed, N_RACES, body, 0.0)
    frame = shuffled(frame, rng)

    fit = fit_of(frame, min_odds=THRESHOLD)

    assert 0 < fit.gamma_se < 0.25  # an informative interval, not an unbounded one
    assert abs(fit.gamma) <= WALD_Z * fit.gamma_se  # the 95% CI covers 0
    assert abs(fit.beta - body) < 3 * fit.se
    assert abs(fit.tail_beta - body) < 3 * fit.tail_beta_se
    assert_is_the_mle(fit, frame)


def test_a_planted_hinge_is_not_covered_by_a_zero_slope_change():
    """The hinge moves with the data: a strong planted slope change is far from 0."""
    frame, _ = planted(3002, N_RACES, 1.15, 0.6)
    fit = fit_of(frame, min_odds=THRESHOLD)
    assert fit.gamma > 3 * fit.gamma_se
    negative, _ = planted(3001, N_RACES, 0.9, -0.25)
    assert fit_of(negative, min_odds=THRESHOLD).gamma < 0  # the sign follows the plant


def step_column(x, o, min_odds):
    """The reference level-step model's second column: 1 on runners priced strictly above
    min_odds (a constant shift of their log-weight, no slope change)."""
    return (o > min_odds).astype(float)


def max_loglik(frame, column):
    """Maximised conditional log-likelihood of the reference two-parameter model on `column`."""
    theta, _, _ = ref_fit(frame, THRESHOLD, column)
    return loglik_score_info(frame, theta, column, THRESHOLD)[0], theta


@pytest.mark.parametrize("seed", [3100, 3101, 3102])
@pytest.mark.parametrize(
    ("gamma", "step"),
    [(0.0, -1.0), (0.0, 1.0), (0.6, 0.0)],
    ids=["level-step-down", "level-step-up", "hinge-slope"],
)
def test_the_hinge_is_told_apart_from_a_level_step_by_likelihood(seed, gamma, step):
    """A level step with no slope change is not a hinge slope: the hinge is continuous at the
    knot, so it cannot reproduce a jump, and a step model fitted to the same market (also two
    parameters, so the log-likelihoods compare directly) fits a planted step clearly better. On a
    planted hinge slope the order reverses. Over seeds 3100-3119 (20 draws of each plant) the
    margins were at least 15 log-likelihood units against a bar of 8 (a likelihood-ratio
    chi-square of 16 on one degree of freedom). The hinge coefficient alone is not the test: a
    level step is partly absorbed by a slope change, so gamma-hat follows the step's sign."""
    frame, rng = planted(seed, N_RACES, 1.0, gamma, step=step)
    frame = shuffled(frame, rng)

    fit = fit_of(frame, min_odds=THRESHOLD)
    assert_is_the_mle(fit, frame)
    theta = np.array([fit.beta, fit.gamma])
    hinge_ll = loglik_score_info(frame, theta, hinge_column, THRESHOLD)[0]
    step_ll, (step_beta, step_shift) = max_loglik(frame, step_column)

    margin = step_ll - hinge_ll  # positive: the level-step model is the better description
    if step != 0.0:
        assert margin > 8.0
        assert math.copysign(1.0, step_shift) == math.copysign(1.0, step)
        assert math.copysign(1.0, fit.gamma) == math.copysign(1.0, step)  # absorbed, not ignored
    else:
        assert margin < -8.0
    assert abs(step_beta - 1.0) < 0.15  # the step model sees the same body slope


def test_the_log_weight_is_continuous_at_the_knot_unlike_the_step_model():
    """Runners quoted exactly 51.0 move to 51.0001: the hinge fit does not move (the hinge is 0 at
    the knot and 2e-6 just above it), while #285's two-slope fit steps by its increment."""
    rng = np.random.default_rng(3050)
    odds = market_odds(rng, 6_000, 16, 8, 1.4)
    odds[(odds >= 30.0) & (odds <= 100.0)] = THRESHOLD
    assert (odds == THRESHOLD).sum() > 1_500
    winners = draw_winners(rng, odds, 1.0, 0.3)
    at = shuffled(long_rows(odds, winners).drop(columns="k"), rng)
    above = at.assign(odds=at["odds"].where(at["odds"] != THRESHOLD, 51.0001))

    a, b = fit_of(at, min_odds=THRESHOLD), fit_of(above, min_odds=THRESHOLD)

    assert abs(a.beta - b.beta) < 1e-3
    assert abs(a.gamma - b.gamma) < 1e-3
    assert abs(a.tail_beta - b.tail_beta) < 1e-3
    assert_is_the_mle(a, at)
    assert_is_the_mle(b, above)
    # Control: the same move changes the two-slope fit, so this test would bite a discontinuity.
    ta = tail_beta.fit_tail_beta(at["odds"], at["race_id"], at["won"], min_odds=THRESHOLD)
    tb = tail_beta.fit_tail_beta(above["odds"], above["race_id"], above["won"], min_odds=THRESHOLD)
    assert abs(ta.increment - tb.increment) > 0.02


def test_devig_hinge_follows_the_continuous_log_weight():
    frame, _ = planted(3002, 6_000, 1.15, 0.6)
    fit = fit_of(frame, min_odds=THRESHOLD)
    race = pd.DataFrame(
        {
            "race_id": ["a"] * 7 + ["b"] * 3,
            "odds": [2.0, 30.0, 50.999, 51.0, 51.001, 120.0, NAN, 3.0, 9.0, 400.0],
        }
    )

    p = devig_hinge(race["odds"], race["race_id"], fit, min_odds=THRESHOLD)

    x = -np.log(race["odds"].to_numpy(dtype=float))
    w = np.exp(fit.beta * x + fit.gamma * np.maximum(0.0, KNOT - x))  # NaN where unpriced
    expected = w / np.where(race["race_id"] == "a", np.nansum(w[:7]), np.nansum(w[7:]))
    assert p.index.equals(race.index)
    assert np.isnan(p.iloc[6])
    np.testing.assert_allclose(p.to_numpy(), expected, rtol=1e-9, equal_nan=True)
    assert p.iloc[:6].sum() == pytest.approx(1.0) and p.iloc[7:].sum() == pytest.approx(1.0)
    # Continuity: the weights of 50.999 and 51.001 differ only by the body slope on a 4e-5 move
    # of x, whatever gamma is (a step at the knot would put exp(step) between them).
    assert p.iloc[4] / p.iloc[2] == pytest.approx((50.999 / 51.001) ** fit.beta, rel=1e-3)


def test_the_threshold_is_an_argument_and_defaults_to_the_owners_tail_min_odds():
    owner = load_country_yaml("japan", "calibration")["tail_min_odds"]
    assert owner == THRESHOLD
    frame, _ = planted(3059, 4_000, 1.0, 0.3)

    default = fit_of(frame)
    explicit = fit_of(frame, min_odds=owner)
    lower = fit_of(frame, min_odds=20.0)

    for name in ("beta", "se", "gamma", "gamma_se", "tail_beta", "tail_beta_se"):
        assert getattr(default, name) == pytest.approx(getattr(explicit, name), rel=1e-9), name
    assert default.n_races == explicit.n_races == 4_000
    assert_is_the_mle(lower, frame, 20.0)  # the knot moves with the threshold
    assert abs(lower.gamma - explicit.gamma) > 0.02  # the threshold changes the fit


def test_unpriced_scratched_and_winnerless_races_leave_the_fit():
    frame, rng = planted(3060, 1_500, 1.05, 0.3, n_min=5)
    clean = frame.copy()
    # Unpriced and junk-priced non-winners inside counted races leave the race's choice set.
    extra = frame.drop_duplicates("race_id").iloc[::5][["race_id"]]
    junk = pd.concat(
        [
            extra.assign(odds=NAN, won=False),
            extra.iloc[::2].assign(odds=0.0, won=False),
            extra.iloc[::3].assign(odds=0.5, won=False),
            extra.iloc[::4].assign(odds=float("inf"), won=False),
        ]
    )
    # Races with no priced winner: a void race, an unpriced winner, a junk-priced winner. Their
    # runners include a longshot, so a leak would move the tail slope.
    rows = []
    for k, (winner_odds, others) in enumerate(
        [
            (None, [2.5, 3.1, 7.7, 80.0]),
            (NAN, [1.9, 4.4, 9.0, 120.0]),
            (0.5, [2.2, 2.8, 30.0, 70.0]),
        ]
    ):
        rows += [(f"void{k}", o, False) for o in others]
        if winner_odds is not None:
            rows.append((f"void{k}", winner_odds, True))
    winnerless = pd.DataFrame(rows, columns=["race_id", "odds", "won"])
    dirty = shuffled(pd.concat([frame, junk, winnerless], ignore_index=True), rng)

    fit = fit_of(dirty, min_odds=THRESHOLD)
    base = fit_of(clean, min_odds=THRESHOLD)

    assert fit.n_races == 1_500
    assert fit.beta == pytest.approx(base.beta, abs=1e-6)
    assert fit.gamma == pytest.approx(base.gamma, abs=1e-6)
    assert fit.gamma_se == pytest.approx(base.gamma_se, rel=1e-6)
    assert_is_the_mle(fit, dirty)
    # Row order and the index labels do not matter.
    again = fit_of(shuffled(dirty, rng), min_odds=THRESHOLD)
    assert again.beta == pytest.approx(fit.beta, abs=1e-6)
    assert again.gamma == pytest.approx(fit.gamma, abs=1e-6)


def test_a_dead_heat_counts_each_priced_co_winner():
    frame, rng = planted(3061, 2_000, 1.0, 0.3)
    races = frame["race_id"].drop_duplicates().iloc[::10]
    extra = []
    for race_id in races:
        part = frame[frame["race_id"] == race_id]
        extra.append(part.index[~part["won"].to_numpy()][-1])
    frame.loc[extra, "won"] = True
    assert frame.groupby("race_id")["won"].sum().max() == 2

    fit = fit_of(shuffled(frame, rng), min_odds=THRESHOLD)

    assert fit.n_races == 2_000
    assert_is_the_mle(fit, frame)  # the reference adds one choice per co-winner


def test_without_a_runner_strictly_above_the_threshold_the_hinge_is_refused():
    frame, _ = planted(3062, 500, 1.0, 0.0)
    flat, _ = planted(3063, 500, 1.0, 0.0, spread=0.2)  # a flat market: nothing above 51.0
    assert (flat["odds"] <= THRESHOLD).all()
    with pytest.raises(ValueError):
        fit_of(flat, min_odds=THRESHOLD)
    # Runners at exactly 51.0 sit on the knot (hinge 0): still nothing identifies gamma.
    on_the_knot = flat.assign(odds=flat["odds"].where(flat["odds"] < 8.0, THRESHOLD))
    assert (on_the_knot["odds"] == THRESHOLD).sum() > 50
    with pytest.raises(ValueError):
        fit_of(on_the_knot, min_odds=THRESHOLD)
    # The same rows do have a tail at a lower threshold.
    assert (flat["odds"] > 8.0).any()
    assert fit_of(flat, min_odds=8.0).n_races == 500
    assert fit_of(frame, min_odds=THRESHOLD).n_races == 500


# ---------------------------------------------------------------------------------------------
# The family: spec, freeze, study
# ---------------------------------------------------------------------------------------------

SCHEME = FoldScheme(
    unit="quarter",
    test_length=1,
    step=1,
    min_train=2,
    window="expanding",
    embargo_days=7,
    anchor_month=1,
)
BOOTSTRAP = {
    "cluster_unit": "meeting",
    "sided": "one-sided",
    "level": 0.95,
    "n_boot": 200,
    "seed": 294,
}
FLOOR = 1.1


def owner_inputs(**override):
    return {**calibration.owner_inputs(), **override}


def make_spec(**override):
    return build_spec(owner_inputs(**override), SCHEME, BOOTSTRAP, FLOOR)


def study_frame(seed=3070, n_races=3_600, years_from="2020-01-02"):
    """Development-frame-shaped rows: 12 races a race day, two venues, from 2020-01 to 2021-08,
    a floor-bound favourite (1.1) in one race in twelve, a planted hinge slope change of 0.4."""
    rng = np.random.default_rng(seed)
    odds = market_odds(rng, n_races, 14, 8, 1.4, floor_every=12)
    winners = draw_winners(rng, odds, 1.05, 0.4)
    frame = long_rows(odds, winners)
    day = frame["k"].to_numpy() // 12
    race_date = pd.Timestamp(years_from) + pd.to_timedelta(day * 2, unit="D")
    venue = np.where((frame["k"].to_numpy() // 6) % 2 == 0, "TOK", "KYO")
    frame["race_date"] = race_date
    frame["meeting_id"] = [f"{d:%Y-%m-%d}:{v}" for d, v in zip(race_date, venue, strict=True)]
    return frame[["race_id", "race_date", "meeting_id", "odds", "won"]]


def test_the_family_is_new_and_the_spec_is_frozen_plain_json(tmp_path, monkeypatch):
    spec = make_spec()
    assert spec["family_id"] == FAMILY_ID
    assert spec["ticket"] == 294
    assert isinstance(FAMILY_ID, str) and FAMILY_ID
    assert FAMILY_ID not in (calibration.FAMILY_ID, tail_beta.FAMILY_ID)
    assert set(spec["comparators"]) == {"normalised", "power", "tail_beta"}
    assert spec["tail"]["min_odds"] == THRESHOLD
    json.dumps(spec)  # plain JSON, as the registry requires

    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    registry = default_registry("japan")
    digest = registry.freeze_family(FAMILY_ID, spec)
    assert registry.freeze_family(FAMILY_ID, make_spec()) == digest  # an equal spec: a no-op
    assert registry.records(FAMILY_ID) == []  # frozen with no scored run

    # Everything pre-registered is in the spec: another threshold, scheme, bootstrap or floor is a
    # different family, which the registry refuses under this id.
    other_tail = make_spec(tail_min_odds=40.0)
    other_scheme = build_spec(owner_inputs(), replace(SCHEME, min_train=3), BOOTSTRAP, FLOOR)
    other_boot = build_spec(owner_inputs(), SCHEME, {**BOOTSTRAP, "n_boot": 300}, FLOOR)
    other_floor = build_spec(owner_inputs(), SCHEME, BOOTSTRAP, 1.2)
    for other in (other_tail, other_scheme, other_boot, other_floor):
        assert other != spec
        with pytest.raises(ValueError):  # FamilyAlreadyFrozen
            registry.freeze_family(FAMILY_ID, other)


@pytest.fixture(scope="module")
def frame():
    return study_frame()


@pytest.fixture(scope="module")
def study(frame):
    return compute_study(frame, make_spec())


def subset_rows(frame, subset):
    if subset == "all_races":
        return frame.reset_index(drop=True)
    floor_race = (
        (frame["odds"] <= FLOOR + 1e-9).groupby(frame["race_id"].to_numpy()).transform("any")
    )
    return frame.loc[~floor_race.to_numpy()].reset_index(drop=True)


def metrics_of(study, subset):
    return [m for v in study.variants if subset in v.name for m in v.metrics]


def metric(study, subset, name):
    found = [m for m in metrics_of(study, subset) if m.metric == name]
    assert len(found) == 1, (subset, name, [m.metric for m in metrics_of(study, subset)])
    return found[0]


def softmax_by_race(z, race_id):
    z = pd.Series(np.asarray(z, dtype=float))
    e = np.exp(z - z.groupby(race_id.to_numpy()).transform("max"))
    return (e / e.groupby(race_id.to_numpy()).transform("sum")).to_numpy()


def reference_scores(rows, min_odds=THRESHOLD):
    """Out-of-fold per-race winner log loss of normalised, power, #285's tail beta and the hinge
    on `rows`, each fitted on its fold's training rows: {name: Series}, and the hinge fold fits."""
    folds = walk_forward(rows["race_date"], scheme=SCHEME)
    losses = {"normalised": [], "power": [], "tail_beta": [], "hinge": []}
    fits = []
    for fold in folds:
        train, test = rows.iloc[fold.train_idx], rows.iloc[fold.test_idx]
        theta, cov, _ = ref_fit(train, min_odds, hinge_column)
        theta_285, _, _ = ref_fit(train, min_odds, slope_column)
        fits.append((theta, cov))
        o = test["odds"].to_numpy()
        x = -np.log(o)
        power = calibration.fit_power_beta(train["odds"], train["race_id"], train["won"]).beta
        z = {
            "hinge": theta[0] * x + theta[1] * hinge_column(x, o, min_odds),
            "tail_beta": theta_285[0] * x + theta_285[1] * slope_column(x, o, min_odds),
            "power": power * x,
            "normalised": x,
        }
        for name, scores in z.items():
            p = softmax_by_race(scores, test["race_id"])
            winner = test["won"].to_numpy(dtype=bool)
            losses[name].append(
                pd.Series(-np.log(p[winner]), index=test["race_id"].to_numpy()[winner])
            )
    return {k: pd.concat(v) for k, v in losses.items()}, fits


@pytest.mark.parametrize("subset", ["all_races", "no_floor_bound_races"])
@pytest.mark.parametrize("comparator", ["normalised", "power", "tail_beta"])
def test_delta_ll_is_out_of_fold_against_each_comparator_with_cluster_bounds(
    frame, study, subset, comparator
):
    rows = subset_rows(frame, subset)
    losses, _ = reference_scores(rows)
    n_scored = len(losses["hinge"])
    assert 500 < n_scored < rows["race_id"].nunique()  # the early quarters only train

    m = metric(study, subset, f"delta_ll_vs_{comparator}")
    expected = float((losses[comparator] - losses["hinge"]).mean())
    assert m.n == n_scored
    assert m.value == pytest.approx(expected, abs=1e-6)
    assert m.lo is not None and m.hi is not None
    assert math.isfinite(m.lo) and math.isfinite(m.hi)
    assert m.lo <= m.value <= m.hi
    assert m.lo < m.hi  # a real bootstrap spread, not a point


def test_the_three_comparisons_are_three_different_numbers(frame, study):
    values = [
        metric(study, "all_races", f"delta_ll_vs_{c}").value
        for c in ("normalised", "power", "tail_beta")
    ]
    assert len({round(v, 6) for v in values}) == 3
    # The planted hinge beats the one-slope power-beta, and #285's step model is the closest.
    assert values[0] > 0 and values[1] > 0
    assert abs(values[2]) < abs(values[0])


def test_the_floor_bound_subset_leaves_whole_races_out_of_fitting_and_scoring(frame, study):
    all_races, no_floor = (subset_rows(frame, s) for s in ("all_races", "no_floor_bound_races"))
    assert 0 < all_races["race_id"].nunique() - no_floor["race_id"].nunique()
    assert (no_floor["odds"] > FLOOR + 1e-9).all()

    n_all = metric(study, "all_races", "delta_ll_vs_normalised").n
    n_no = metric(study, "no_floor_bound_races", "delta_ll_vs_normalised").n
    assert n_all == len(reference_scores(all_races)[0]["hinge"])
    assert n_no == len(reference_scores(no_floor)[0]["hinge"])
    assert n_no < n_all
    # The fit leaves them out too: overall in-sample slopes are the reference's on each subset.
    for subset, rows in (("all_races", all_races), ("no_floor_bound_races", no_floor)):
        assert metric(study, subset, "hinge_gamma").n == ref_fit(rows)[2]
    assert (
        metric(study, "all_races", "hinge_gamma").n
        > metric(study, "no_floor_bound_races", "hinge_gamma").n
    )


@pytest.mark.parametrize("subset", ["all_races", "no_floor_bound_races"])
def test_both_slopes_are_reported_overall_and_by_year_with_wald_cis(frame, study, subset):
    rows = subset_rows(frame, subset)
    years = sorted(rows["race_date"].dt.year.unique())
    assert years == [2020, 2021]
    groups = {"": rows} | {f"[year={y}]": rows[rows["race_date"].dt.year == y] for y in years}

    for suffix, part in groups.items():
        theta, cov, n = ref_fit(part)
        tail, tail_se = tail_beta_of(theta, cov)
        expected = {
            "body_beta": (theta[0], math.sqrt(cov[0, 0])),
            "hinge_gamma": (theta[1], math.sqrt(cov[1, 1])),
            "tail_beta": (tail, tail_se),
        }
        for name, (value, se) in expected.items():
            m = metric(study, subset, name + suffix)
            assert m.n == n, name + suffix
            assert m.value == pytest.approx(value, abs=1e-5), name + suffix
            assert m.lo == pytest.approx(value - WALD_Z * se, abs=1e-3), name + suffix
            assert m.hi == pytest.approx(value + WALD_Z * se, abs=1e-3), name + suffix
            assert m.lo < m.value < m.hi
    # The years differ, so a year's slope is not the overall one.
    overall = metric(study, subset, "hinge_gamma").value
    assert any(
        abs(metric(study, subset, f"hinge_gamma[year={y}]").value - overall) > 1e-3 for y in years
    )


def test_the_planted_hinge_is_recovered_in_the_study_frame(frame):
    theta, cov, _ = ref_fit(frame.reset_index(drop=True))
    assert abs(theta[1] - 0.4) < 3 * math.sqrt(cov[1, 1])  # the plant is recoverable here


def test_the_fold_slopes_are_each_folds_training_fit(frame, study):
    rows = subset_rows(frame, "all_races")
    _, fits = reference_scores(rows)
    folds = walk_forward(rows["race_date"], scheme=SCHEME)
    found = [
        m for m in metrics_of(study, "all_races") if m.metric.startswith("hinge_gamma[test_year=")
    ]
    assert len(found) == len(folds) == len(fits)
    got = sorted(m.value for m in found)
    want = sorted(float(theta[1]) for theta, _ in fits)
    np.testing.assert_allclose(got, want, atol=1e-5)
    assert all(m.lo is not None and m.lo < m.value < m.hi for m in found)


def test_every_variant_is_one_scored_record_and_none_failed(study):
    names = [(v.hypothesis_id, v.name) for v in study.variants]
    assert len(names) == len(set(names))  # one record per variant
    for subset in ("all_races", "no_floor_bound_races"):
        assert any(subset in n and "walk_forward" in n for _, n in names)
        assert any(subset in n and "by_year" in n for _, n in names)
    assert len(names) == 4
    for v in study.variants:
        assert v.error is None, (v.name, v.error)
        assert v.metrics, v.name
        assert v.fold_scheme
        assert v.components
    # Out-of-fold scoring runs on the spec's folds: the fold scheme is in a record's identity.
    scored = [v for v in study.variants if any(m.metric.startswith("delta_ll") for m in v.metrics)]
    assert len(scored) == 2
    assert all(
        json.dumps(dataclasses.asdict(SCHEME), sort_keys=True) in v.fold_scheme for v in scored
    )


def test_the_report_lists_the_owner_decisions_and_their_signoff_and_no_race_rows(study, frame):
    spec = make_spec()
    about = {"family": f"`{FAMILY_ID}`, spec sha256 `abc123`", "git": "deadbeef"}

    text = render_report(study, spec, about)

    assert spec["decisions"]  # the decisions used are pre-registered in the spec
    for key, decision in spec["decisions"].items():
        assert str(key) in text and str(decision) in text
    assert f"#{spec['owner_signoff']}" in text
    assert "abc123" in text and "deadbeef" in text
    for v in study.variants:
        assert v.name in text
    assert "no_floor_bound_races" in text
    assert frame["race_id"].iloc[0] not in text  # aggregate tables only, never a race row


# ---------------------------------------------------------------------------------------------
# run_study: freeze first, development rows only, no holdout, one record per variant
# ---------------------------------------------------------------------------------------------

BUILDS = {
    "jra_runners": "processed/jra/runners.parquet@aaaaaaaaaaaaaaaa",
    "jra_races": "processed/jra/races.parquet@bbbbbbbbbbbbbbbb",
}


class _StopHere(Exception):
    pass


def test_the_family_is_frozen_before_any_development_row_is_read(tmp_path, monkeypatch):
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(calibration, "store_builds", lambda: dict(BUILDS))
    seen = {}

    def read_development():
        registry = default_registry("japan")
        seen["frozen"] = registry.frozen_spec_hash(FAMILY_ID)
        seen["records"] = registry.records(FAMILY_ID)
        raise _StopHere

    monkeypatch.setattr(calibration, "read_development", read_development)
    with pytest.raises(_StopHere):
        run_study(tmp_path / "reports")
    assert seen["frozen"]  # the spec was in the registry when the first row was asked for
    assert seen["records"] == []  # and nothing was scored yet


def test_run_refuses_a_family_frozen_with_another_spec_before_reading_anything(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path / "data"))
    default_registry("japan").freeze_family(FAMILY_ID, {"family_id": FAMILY_ID, "another": "spec"})
    monkeypatch.setattr(calibration, "store_builds", lambda: dict(BUILDS))

    def read_development():
        raise AssertionError("a development row was read")

    monkeypatch.setattr(calibration, "read_development", read_development)
    with pytest.raises(FamilyAlreadyFrozen):
        run_study(tmp_path / "reports")
    assert default_registry("japan").records(FAMILY_ID) == []


def test_run_refuses_an_out_dir_inside_the_repository(tmp_path, monkeypatch):
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path / "data"))
    with pytest.raises(paths.DataDirInsideRepo):
        run_study(env.repo_root() / "reports-should-not-be-here")


def holdout_days(n, start=date(2026, 11, 28)):
    days, day = [], start
    while len(days) < n:
        if holdout.is_holdout_day("japan", day):
            days.append(day)
        day += timedelta(days=1)
    return days


def long_study_frame(seed=3080, n_races=6_000):
    """Race days a week apart from 2016-01 (the real folds need 22 quarters to train), 12 races a
    meeting, one venue, a planted hinge slope change; ends before the holdout starts."""
    rng = np.random.default_rng(seed)
    odds = market_odds(rng, n_races, 10, 7, 1.4, floor_every=12)
    winners = draw_winners(rng, odds, 1.05, 0.4)
    frame = long_rows(odds, winners)
    race = frame["k"].to_numpy() // 12
    day = pd.Timestamp("2016-01-02") + pd.to_timedelta(7 * race, unit="D")
    assert day.max() < pd.Timestamp("2026-11-28")
    frame["race_date"] = day
    frame["meeting_id"] = [f"{d:%Y-%m-%d}:TOK" for d in day]
    frame["race_no"] = frame["k"].to_numpy() % 12 + 1
    return frame


def write_store(data_root, frame, poison):
    """The synthetic frame (plus `poison`, rows on holdout days) as the store's two tables."""
    store = data_root / "processed" / "jra"
    store.mkdir(parents=True)
    both = pd.concat([frame, poison], ignore_index=True)
    day = both["race_date"].dt.date
    runners = pd.DataFrame(
        {
            "race_id": both["race_id"],
            "race_date": day,
            "win_odds": both["odds"],
            "finish_pos": pd.array(np.where(both["won"], 1, 2), dtype="Int64"),
            "is_scratched": False,
        }
    )
    races = both.drop_duplicates("race_id").assign(
        venue="TOK",
        race_class="OPEN",
        surface="turf",
        going="GOOD",
        weight_type="age",
        status="run",
        race_date=lambda r: r["race_date"].dt.date,
    )
    races["field_size"] = races["race_id"].map(both.groupby("race_id").size())
    races = races[
        ["race_id", "race_date", "meeting_id", "venue", "race_no", "race_class", "surface"]
        + ["going", "weight_type", "field_size", "status"]
    ]
    runners.to_parquet(store / "runners.parquet", index=False)
    races.to_parquet(store / "races.parquet", index=False)


@pytest.fixture(scope="module")
def long_frame():
    return long_study_frame()


def test_run_study_scores_development_rows_only_and_records_every_variant(
    long_frame, tmp_path, monkeypatch
):
    poison_days = holdout_days(2)
    poison = long_study_frame(seed=3081, n_races=24).assign(
        race_id=lambda f: "HOLDOUT-" + f["race_id"],
        race_date=lambda f: (
            pd.Timestamp(poison_days[0])
            + pd.to_timedelta((f["k"] // 12) * (poison_days[1] - poison_days[0]).days, unit="D")
        ),
        meeting_id=lambda f: "HOLDOUT-" + f["meeting_id"],
    )
    assert {d.date() for d in poison["race_date"]} == set(poison_days)
    root = tmp_path / "data"
    write_store(root, long_frame, poison.drop(columns="k"))
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(root))

    def refuse(*args, **kwargs):
        raise AssertionError("the holdout was opened")

    monkeypatch.setattr(holdout.LockedHoldout, "open", refuse)
    log = default_registry("japan").holdout_log
    log_before = log.read_bytes() if log.exists() else None
    scored_frames = []
    real_compute = module.compute_study

    def spy(frame_, spec_):
        scored_frames.append(frame_)
        return real_compute(frame_, spec_)

    monkeypatch.setattr(module, "compute_study", spy)

    result = run_study(tmp_path / "out")

    assert scored_frames, "run_study must score through compute_study"
    scored = scored_frames[0]
    assert not scored["race_id"].str.startswith("HOLDOUT-").any()  # no holdout race row read
    assert scored["race_id"].nunique() == long_frame["race_id"].nunique()
    assert result["family_id"] == FAMILY_ID
    assert result["failed_variants"] == []
    registry = default_registry("japan")
    registry.verify()
    assert registry.frozen_spec_hash(FAMILY_ID) == result["spec_hash"]
    records = registry.records(FAMILY_ID)
    assert len(records) == result["records_appended"] == 4
    assert result["records_already_present"] == 0
    assert all(r.status == "scored" and r.ticket == 294 and r.sources == ("jra",) for r in records)
    assert registry.frozen_spec_hash(calibration.FAMILY_ID) is None  # #77's family untouched
    assert registry.frozen_spec_hash(tail_beta.FAMILY_ID) is None  # and #285's
    names = {r.variant for r in records}
    assert any("walk_forward" in n for n in names) and any("by_year" in n for n in names)
    for r in records:
        if "walk_forward" in r.variant:
            assert {m.metric for m in r.metrics} >= {
                "delta_ll_vs_normalised",
                "delta_ll_vs_power",
                "delta_ll_vs_tail_beta",
            }
        else:
            assert {m.metric for m in r.metrics} >= {"body_beta", "hinge_gamma", "tail_beta"}

    # The report goes through write_report: outside the repository, one file, named by the spec.
    report = Path(result["report_path"])
    assert report.resolve().parent == (tmp_path / "out").resolve()
    assert list((tmp_path / "out").glob("*.md")) == [report]
    text = report.read_text(encoding="utf-8")
    assert result["spec_hash"] in text and "HOLDOUT-" not in text
    log_after = log.read_bytes() if log.exists() else None
    assert log_after == log_before  # the holdout log is untouched

    again = run_study(tmp_path / "out")  # a rerun adds no record and keeps the family
    assert again["spec_hash"] == result["spec_hash"]
    assert again["records_appended"] == 0
    assert again["records_already_present"] == 4
    assert len(default_registry("japan").records(FAMILY_ID)) == 4
