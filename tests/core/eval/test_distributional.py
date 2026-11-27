"""C1's distributional scores (ticket #85): `horsetrack.core.eval.distributional`.

Contract pinned here:
- `crps_ensemble(draws, obs) -> float`: the CRPS of the 1-D ensemble `draws` at `obs`, lower is
  better: E|X - obs| - 1/2 E|X - X'| over the ensemble. A point forecast (every draw equal) scores
  the absolute error; a large ensemble from N(mu, sd) approaches the Gaussian closed form; the score
  is translation invariant and scales with the units. Whether E|X - X'| uses 1/n^2 or the fair
  1/(n(n - 1)) is left to the module (both meet every check here).
- `pit(draws, obs) -> float`: the empirical CDF of the ensemble at `obs`, the share of draws below
  it; 0 below every draw and 1 above every draw. Ties are not pinned. An observation drawn from the
  ensemble's own distribution gives uniform PIT values.
- `gaussian_log_score(mean, sd, obs) -> float`: log N(obs; mean, sd), the log predictive density
  of D8, higher is better.

Expected values come from scipy, numpy and closed forms, never from the module under test.
"""

# Ruff reads the module under test as third-party until it exists; I001 is waived so lint passes
# before and after.
from __future__ import annotations  # noqa: I001

import math

import numpy as np
import pytest
from scipy import stats

from horsetrack.core.eval.distributional import crps_ensemble, gaussian_log_score, pit


def gaussian_crps(mu: float, sd: float, obs: float) -> float:
    """The closed-form CRPS of N(mu, sd^2) at obs (Gneiting and Raftery 2007)."""
    z = (obs - mu) / sd
    return sd * (
        z * (2.0 * stats.norm.cdf(z) - 1.0) + 2.0 * stats.norm.pdf(z) - 1.0 / math.sqrt(math.pi)
    )


# ---------------------------------------------------------------- CRPS


@pytest.mark.parametrize(
    ("value", "obs", "n"),
    [
        (0.0, 0.0, 1),
        (1.5, 0.25, 1),
        (-2.0, 3.0, 10),
        (0.3, 0.31, 100),
        (-4.6, -2.3, 7),
        (12.0, -1.0, 3),
        (-0.7, -0.7, 25),
    ],
)
def test_crps_of_a_point_forecast_is_the_absolute_error(value, obs, n):
    score = crps_ensemble(np.full(n, value), obs)
    assert isinstance(score, float)
    assert score == pytest.approx(abs(value - obs), abs=1e-12)


@pytest.mark.parametrize(
    ("mu", "sd", "obs"),
    [(0.0, 1.0, 0.0), (0.0, 1.0, 1.5), (2.0, 0.5, 1.0), (-1.0, 2.0, 3.0), (0.3, 0.05, 0.31)],
)
def test_crps_of_a_large_ensemble_approaches_the_gaussian_closed_form(mu, sd, obs):
    draws = np.random.default_rng(17).normal(mu, sd, size=2000)
    expected = gaussian_crps(mu, sd, obs)
    assert crps_ensemble(draws, obs) == pytest.approx(expected, abs=0.05 * sd)
    # the spread term matters: the score is well below the mean absolute error
    assert crps_ensemble(draws, obs) < float(np.mean(np.abs(draws - obs))) - 0.2 * sd


def test_crps_is_translation_invariant_and_scales_with_the_units():
    rng = np.random.default_rng(3)
    draws, obs = rng.standard_normal(300), 0.4
    base = crps_ensemble(draws, obs)
    assert base > 0
    for scale, shift in ((1.0, 5.0), (3.0, -2.0), (0.1, 0.0), (250.0, 1e3)):
        moved = crps_ensemble(scale * draws + shift, scale * obs + shift)
        assert moved == pytest.approx(scale * base, rel=1e-9)
    # the further the observation from the ensemble, the worse the score
    scores = [crps_ensemble(draws, y) for y in (0.0, 1.0, 2.0, 5.0, 10.0)]
    assert scores == sorted(scores)
    assert scores[-1] > scores[0]


# ---------------------------------------------------------------- PIT


def test_pit_is_the_share_of_draws_below_the_observation():
    rng = np.random.default_rng(5)
    for n in (1, 5, 40, 333):
        draws = rng.standard_normal(n)
        lo, hi = draws.min(), draws.max()
        assert pit(draws, lo - 1.0) == pytest.approx(0.0, abs=1e-12)
        assert pit(draws, hi + 1.0) == pytest.approx(1.0, abs=1e-12)
        for obs in rng.uniform(lo - 0.5, hi + 0.5, size=12):
            assert not np.any(draws == obs)
            assert pit(draws, float(obs)) == pytest.approx(float(np.mean(draws < obs)), abs=1e-12)


def _sample(family: int, a: float, b: float, rng, size: int) -> np.ndarray:
    """Draws from one of three families: normal, lognormal or gamma."""
    if family == 0:
        return rng.normal(a, b, size=size)
    if family == 1:
        return np.exp(rng.normal(a / 3.0, b, size=size))
    return rng.gamma(1.0 + b, 1.0 + abs(a), size=size)


def test_pit_of_draws_from_the_true_distribution_is_uniform():
    rng = np.random.default_rng(85)
    n_cases, n_draws = 2000, 500
    calibrated, biased, too_wide = [], [], []
    for i in range(n_cases):
        a, b = rng.normal(0.0, 3.0), rng.uniform(0.1, 2.0)
        family = i % 3
        draws = _sample(family, a, b, rng, n_draws)
        obs = float(_sample(family, a, b, rng, 1)[0])
        calibrated.append(pit(draws, obs))
        # misspecified forecasts for the same normal truth: shifted by one sd, and three times wide
        truth = a + b * rng.standard_normal()
        biased.append(pit(rng.normal(a - b, b, size=n_draws), truth))
        too_wide.append(pit(rng.normal(a, 3.0 * b, size=n_draws), truth))
    u = np.asarray(calibrated, dtype=float)
    assert ((u >= 0.0) & (u <= 1.0)).all()
    assert stats.kstest(u, "uniform").pvalue > 0.01
    # the check has power: miscalibrated ensembles are not uniform
    assert stats.kstest(biased, "uniform").pvalue < 1e-6
    assert stats.kstest(too_wide, "uniform").pvalue < 1e-6


# ---------------------------------------------------------------- log score


@pytest.mark.parametrize(
    ("mean", "sd", "obs"),
    [
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 2.0),
        (-3.2, 0.4, -3.0),
        (1.5, 0.05, 1.62),
        (10.0, 7.5, -4.0),
        (-0.69, 0.2, -1.6),
    ],
)
def test_gaussian_log_score_is_the_normal_log_density(mean, sd, obs):
    score = gaussian_log_score(mean, sd, obs)
    assert isinstance(score, float)
    assert score == pytest.approx(float(stats.norm.logpdf(obs, loc=mean, scale=sd)), rel=1e-12)
    # positively oriented: the density is highest at the mean
    assert gaussian_log_score(mean, sd, mean) >= score
    assert gaussian_log_score(mean, sd, mean + 2.0 * sd) < gaussian_log_score(mean, sd, mean)
