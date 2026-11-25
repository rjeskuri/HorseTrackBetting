"""The crowd-calibration baseline at the final Win price (#77; 08 §8 A3, §4 E1, E3, E4).

Not the D4 spline model (`models.v0.calibration`): this module measures how the final Win price
turns into a probability under three de-vigs and how far a power beta departs from 1.

Inputs are pandas Series on one index; `race_id` labels each runner's race (rows need not be
grouped by race). Odds are decimal, stake included, and go through `core.pricing.clean_odds`: a
missing, non-finite or sub-1.0 figure is an unpriced runner, dropped before any de-vig, so it takes
no share (NaN) and each race's priced runners sum to 1.

- normalised: q_i = (1/O_i) / sum_j (1/O_j);
- power: p_i proportional to (1/O_i)^beta;
- Shin (1993): with pi_i = 1/O_i and Pi = sum_j pi_j, p_i = (sqrt(z^2 + 4(1 - z) pi_i^2 / Pi) - z)
  / (2(1 - z)), the insider share z in [0, 1) solved per race so the race sums to 1.

Memory: everything works on flat arrays (np.bincount and np.add.reduceat over race segments);
Shin's z is one vectorised bisection over every race at once, and the beta fit is a Newton
iteration whose steps each cost a few passes over the rows. Nothing is quadratic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from horsetrack.core.pricing import _EPS as FLOOR_TOLERANCE
from horsetrack.core.pricing import clean_odds

SHIN_BISECTIONS = 60  # halvings of [0, 1]: z to below 1e-17
NEWTON_MAX_STEPS = 50
NEWTON_STEP_TOL = 1e-10


@dataclass(frozen=True)
class BetaFit:
    """The power de-vig's beta (maximum likelihood), its standard error and the races fitted."""

    beta: float
    se: float
    n_races: int


# --------------------------------------------------------------------------- inputs


def _check_aligned(first: pd.Series, *others: pd.Series) -> None:
    for other in others:
        if not other.index.equals(first.index):
            raise ValueError(f"{other.name or 'a Series'} is not on the odds' index")


def _race_codes(race_id: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """0..n-1 race codes per row, in first-appearance order, and the race ids they stand for."""
    codes, uniques = pd.factorize(race_id.to_numpy())
    if (codes < 0).any():
        raise ValueError("race_id has missing values; every runner needs its race")
    return codes, np.asarray(uniques)


def _priced(
    odds: pd.Series, race_id: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cleaned odds, the priced mask, race codes and the race ids the codes stand for."""
    _check_aligned(odds, race_id)
    cleaned = clean_odds(odds.to_numpy())
    codes, races = _race_codes(race_id)
    return cleaned, ~np.isnan(cleaned), codes, races


def _on_index(values: np.ndarray, ok: np.ndarray, index: pd.Index) -> pd.Series:
    out = np.full(len(ok), np.nan)
    out[ok] = values
    return pd.Series(out, index=index, dtype=float)


# --------------------------------------------------------------------------- de-vigs


def devig_power(odds: pd.Series, race_id: pd.Series, beta: float) -> pd.Series:
    """p_i proportional to (1/O_i)^beta over each race's priced runners; NaN where unpriced."""
    cleaned, ok, codes, races = _priced(odds, race_id)
    weight = (1.0 / cleaned[ok]) ** float(beta)
    totals = np.bincount(codes[ok], weights=weight, minlength=len(races))
    return _on_index(weight / totals[codes[ok]], ok, odds.index)


def devig_normalised(odds: pd.Series, race_id: pd.Series) -> pd.Series:
    """q_i = (1/O_i) / sum_j (1/O_j) over each race's priced runners; NaN where unpriced."""
    return devig_power(odds, race_id, 1.0)


def _shin_p(c: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Shin's p_i for c_i = pi_i^2 / Pi, written without the 1 / (1 - z) that cancels badly near
    z = 1: (sqrt(z^2 + 4(1 - z)c) - z) / (2(1 - z)) = 2c / (sqrt(z^2 + 4(1 - z)c) + z)."""
    return 2.0 * c / (np.sqrt(z * z + 4.0 * (1.0 - z) * c) + z)


def devig_shin(odds: pd.Series, race_id: pd.Series) -> pd.Series:
    """Shin (1993) probabilities over each race's priced runners; NaN where unpriced.

    The race's sum of p_i falls strictly in z (each c_i <= 1), from sqrt(Pi) at z = 0 to
    sum c_i < 1 at z = 1, so a book with overround (Pi > 1) has one root in (0, 1), found by
    bisection over every race at once. A book without overround (Pi <= 1) has no root in [0, 1):
    z is clamped to 0 and the shares normalised, which is the normalised de-vig (exactly 1/O_i
    when Pi = 1).
    """
    cleaned, ok, codes, races = _priced(odds, race_id)
    codes, n_races = codes[ok], len(races)
    pi = 1.0 / cleaned[ok]
    book = np.bincount(codes, weights=pi, minlength=n_races)
    c = pi * pi / book[codes]

    lo, hi = np.zeros(n_races), np.ones(n_races)
    for _ in range(SHIN_BISECTIONS):
        mid = 0.5 * (lo + hi)
        above = np.bincount(codes, weights=_shin_p(c, mid[codes]), minlength=n_races) > 1.0
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    z = 0.5 * (lo + hi)

    over = (book > 1.0)[codes]
    p = np.where(over, _shin_p(c, z[codes]), pi / book[codes])
    return _on_index(p, ok, odds.index)


# --------------------------------------------------------------------------- the beta fit


@dataclass(frozen=True)
class _ChoiceSets:
    """Priced runners of the races with a priced winner, sorted by race."""

    x: np.ndarray  # ln(1 / O)
    starts: np.ndarray  # each race's first row
    sizes: np.ndarray  # each race's priced runners
    winners: np.ndarray  # each race's priced winners (2+ on a dead heat)
    x_won: float  # sum of x over every priced winner


def _choice_sets(odds: pd.Series, race_id: pd.Series, won: pd.Series) -> _ChoiceSets:
    _check_aligned(odds, race_id, won)
    if won.isna().any():
        raise ValueError("won has missing values")
    cleaned, ok, codes, races = _priced(odds, race_id)
    won_priced = won.to_numpy(dtype=bool) & ok
    winners = np.bincount(codes, weights=won_priced, minlength=len(races))
    rows = np.flatnonzero(ok & (winners[codes] > 0))
    if len(rows) == 0:
        raise ValueError("no race with a priced winner to fit")
    rows = rows[np.argsort(codes[rows], kind="stable")]
    race = codes[rows]
    starts = np.flatnonzero(np.r_[True, race[1:] != race[:-1]])
    x = -np.log(cleaned[rows])
    return _ChoiceSets(
        x=x,
        starts=starts,
        sizes=np.diff(np.r_[starts, len(rows)]),
        winners=winners[race[starts]],
        x_won=float(x[won_priced[rows]].sum()),
    )


def _loglik(beta: float, sets: _ChoiceSets) -> tuple[float, float, float]:
    """The conditional-logit log-likelihood, its score and the observed information at beta."""
    s = beta * sets.x
    top = np.maximum.reduceat(s, sets.starts)
    e = np.exp(s - np.repeat(top, sets.sizes))
    total = np.add.reduceat(e, sets.starts)
    mean = np.add.reduceat(e * sets.x, sets.starts) / total
    spread = sets.x - np.repeat(mean, sets.sizes)
    var = np.add.reduceat(e * spread * spread, sets.starts) / total
    w = sets.winners
    loglik = beta * sets.x_won - float((w * (np.log(total) + top)).sum())
    return loglik, sets.x_won - float((w * mean).sum()), float((w * var).sum())


def fit_power_beta(odds: pd.Series, race_id: pd.Series, won: pd.Series) -> BetaFit:
    """The maximum-likelihood beta of P(i wins) proportional to exp(beta * ln(1/O_i)) over each
    race's priced runners, a conditional logit on ln(1/odds).

    Only races with a priced winner enter, and `n_races` counts them; unpriced runners leave
    their race's choice set. A dead heat counts each priced co-winner as a winner of its race
    (as #114's fit does): the race adds one choice from its runners per co-winner. The
    log-likelihood is concave in beta; Newton steps, halved whenever one would lower it, run
    until a step is under 1e-10. `se` is the observed information at beta-hat to the power -1/2.

    Raises ValueError when no race has a priced winner or no race's runners differ in price
    (beta is not identified), and RuntimeError when Newton does not converge (no finite maximum,
    for example when every race went to its favourite).
    """
    sets = _choice_sets(odds, race_id, won)
    beta = 1.0
    loglik, score, info = _loglik(beta, sets)
    for _ in range(NEWTON_MAX_STEPS):
        if not info > 0:
            raise ValueError("beta is not identified: no fitted race's runners differ in price")
        step = score / info
        while True:
            trial = _loglik(beta + step, sets)
            if trial[0] >= loglik or abs(step) < NEWTON_STEP_TOL:
                break
            step /= 2.0
        beta += step
        loglik, score, info = trial
        if abs(step) < NEWTON_STEP_TOL:
            break
    else:
        raise RuntimeError(f"the beta fit did not converge in {NEWTON_MAX_STEPS} Newton steps")
    if not (np.isfinite(beta) and info > 0):
        raise RuntimeError(f"the beta fit stopped at beta={beta}, information={info}")
    return BetaFit(beta=float(beta), se=float(info**-0.5), n_races=len(sets.starts))


def segment_beta(frame: pd.DataFrame, segment: str) -> pd.DataFrame:
    """The beta fit on each value's races of `frame[segment]` (constant within a race): one row
    per value, in sorted order, with columns `segment`, beta, se and n_races. `frame` has
    race_id, odds and won; other columns are ignored, and rows with a missing segment value are
    left out."""
    varies = frame.groupby("race_id", observed=True)[segment].nunique(dropna=False) > 1
    if varies.any():
        raise ValueError(f"{segment!r} varies within {int(varies.sum())} race(s)")
    rows = []
    for value, part in frame.groupby(segment, sort=True, observed=True):
        fit = fit_power_beta(part["odds"], part["race_id"], part["won"])
        rows.append({segment: value, "beta": fit.beta, "se": fit.se, "n_races": fit.n_races})
    return pd.DataFrame(rows, columns=[segment, "beta", "se", "n_races"])


# --------------------------------------------------------------------------- entropy, floor


def normalised_entropy(odds: pd.Series, race_id: pd.Series) -> pd.Series:
    """Per race, indexed by race id (first-appearance order): the Shannon entropy of its
    normalised probabilities over its n priced runners divided by ln n, clipped to [0, 1]
    against rounding. NaN for a race with fewer than two priced runners (ln n = 0)."""
    cleaned, ok, codes, races = _priced(odds, race_id)
    codes = codes[ok]
    inv = 1.0 / cleaned[ok]
    q = inv / np.bincount(codes, weights=inv, minlength=len(races))[codes]
    entropy = np.bincount(codes, weights=-q * np.log(q), minlength=len(races))
    n = np.bincount(codes, minlength=len(races))
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where(n >= 2, np.clip(entropy / np.log(n), 0.0, 1.0), np.nan)
    return pd.Series(value, index=pd.Index(races, name="race_id"), name="entropy")


def flag_floor_bound(odds: pd.Series, *, floor: float) -> pd.Series:
    """True for a priced runner at or under `floor` (within `core.pricing.floor_bound`'s 1e-9):
    its real pool share is unknown. An unpriced runner is never floor-bound."""
    cleaned = clean_odds(odds.to_numpy())
    with np.errstate(invalid="ignore"):
        flagged = ~np.isnan(cleaned) & (cleaned <= floor + FLOOR_TOLERANCE)
    return pd.Series(flagged, index=odds.index, dtype=bool)
