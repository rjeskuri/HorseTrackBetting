"""Distributional scores of an ensemble forecast (C1 of the Japan stack, ticket #85).

#69's scores are point scores; the final-price nowcast forecasts a distribution, scored as D8 signs
(docs/problem-statement.md section 3: the log predictive density of the final log share, with PIT
and interval coverage) and by CRPS:

- `crps_ensemble(draws, obs)`: the CRPS of the empirical distribution of a 1-D ensemble at `obs`,
  E|X - obs| - 1/2 E|X - X'| with both expectations over the ensemble (the spread term over all
  n^2 ordered pairs, a pair with itself included). Lower is better; a point forecast scores the
  absolute error. `crps_columns` is the same per column of an (n, k) array.
- `pit(draws, obs)`: the empirical CDF of the ensemble at `obs`, the share of draws strictly below
  it. `pit_columns` per column.
- `gaussian_log_score(mean, sd, obs)`: log N(obs; mean, sd^2), the log predictive density of a
  Gaussian forecast; higher is better. `gaussian_log_scores` is its elementwise array form.
"""

from __future__ import annotations

import math

import numpy as np

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def crps_columns(draws: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """The ensemble CRPS of each column of `draws` (n, k) at the matching entry of `obs` (k,).

    Computed on draws - obs (the score is translation invariant), with the spread term from the
    sorted draws: sum_ij |x_i - x_j| = 2 sum_i (2i - n + 1) x_(i), i from 0."""
    x = np.asarray(draws, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError(f"draws must be (n, k) with n >= 1, got shape {x.shape}")
    y = np.broadcast_to(np.asarray(obs, dtype=float), (x.shape[1],))
    z = np.sort(x - y, axis=0)
    n = z.shape[0]
    weights = (2.0 * np.arange(n) - n + 1.0)[:, None]
    return np.abs(z).mean(axis=0) - (weights * z).sum(axis=0) / (n * n)


def crps_ensemble(draws: np.ndarray, obs: float) -> float:
    """The CRPS of the 1-D ensemble `draws` at `obs` (lower is better)."""
    x = np.asarray(draws, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"draws must be 1-D, got shape {x.shape}")
    return float(crps_columns(x[:, None], np.array([float(obs)]))[0])


def pit_columns(draws: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """The share of each column's draws strictly below the matching entry of `obs`."""
    x = np.asarray(draws, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError(f"draws must be (n, k) with n >= 1, got shape {x.shape}")
    return (x < np.asarray(obs, dtype=float)[None, :]).mean(axis=0)


def pit(draws: np.ndarray, obs: float) -> float:
    """The empirical CDF of the 1-D ensemble `draws` at `obs`: the share of draws below it."""
    x = np.asarray(draws, dtype=float)
    if x.ndim != 1 or x.size < 1:
        raise ValueError(f"draws must be a non-empty 1-D array, got shape {x.shape}")
    return float(np.mean(x < float(obs)))


def gaussian_log_scores(mean, sd, obs) -> np.ndarray:
    """log N(obs; mean, sd^2), elementwise; `sd` must be positive."""
    mean, sd, obs = (np.asarray(v, dtype=float) for v in (mean, sd, obs))
    if np.any(~(sd > 0.0)):
        raise ValueError("sd must be positive")
    z = (obs - mean) / sd
    return -_HALF_LOG_2PI - np.log(sd) - 0.5 * z * z


def gaussian_log_score(mean: float, sd: float, obs: float) -> float:
    """The log density of N(mean, sd^2) at `obs`, D8's log predictive density (higher is better)."""
    return float(gaussian_log_scores(mean, sd, obs))
