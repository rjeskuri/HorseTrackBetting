"""The conditional logit under the C4 meta-model (ticket #87, epic #73; 08 §3.2 C4).

Contract pinned here (`horsetrack.countries.japan.models.condlogit`):
- `fit_conditional_logit(X, race_id, won, *, offset=None, ridge=0.0) -> CLFit(coef, se, loglik,
  n_races)`. `X` is a (rows, k) float array, `race_id` one label per row (any hashable; rows need
  not be grouped by race), `won` a bool per row, `offset` None or one float per row. P(i wins) =
  exp(u_i) / sum_j exp(u_j) over the rows of i's race, with u = offset + X @ coef: the offset
  enters with its coefficient fixed at 1 and has no entry in `coef`. There are no bounds.
- Races are treated as in core.eval.scoring (owner's answer on #69, 2026-11-24): a race with no
  winner is dropped (no likelihood, not in `n_races`); a dead heat with k co-winners counts each
  co-winner at weight 1/k, so every race carries one unit:
  loglik = sum over races of (1/k) * sum over its winners of (u_w - logsumexp of the race's u).
- `ridge`: the objective minimised is -loglik + ridge / 2 * |coef|^2 (the precision of a
  N(0, 1/ridge) prior on each coefficient, as in models.v0.residual and handicap.factors); 0 is
  the plain maximum likelihood.
- `coef` and `se` are float arrays of length k. At ridge 0, `se` is sqrt(diag(I^-1)) with I the
  information at `coef`: the sum over races of the p-weighted covariance of the race's X rows.
  `loglik` is the unpenalised log-likelihood at `coef` (under a ridge too). `n_races` counts the
  races that enter the likelihood.
- Missing or non-finite X or offset, inputs of different lengths and a negative ridge are a
  ValueError.

Expected values come from first principles (the score, information and log-likelihood written out
below), never from the code under test. The maximum-likelihood checks allow the fit to stop within
1% of a standard error of the root, which any decent optimiser does and no wrong objective can.
All data is synthetic and every RNG is seeded.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import numpy as np
import pandas as pd
import pytest

from horsetrack.countries.japan.models.condlogit import CLFit, fit_conditional_logit

PLANTED = np.array([0.8, -0.5])


# ---------------------------------------------------------------------------------------------
# Synthetic races and the first-principles likelihood
# ---------------------------------------------------------------------------------------------


def field(rng, n_races, sizes=(6, 14)):
    """Race codes 0..n_races-1, one per runner row, with field sizes drawn from `sizes`."""
    n = rng.integers(sizes[0], sizes[1] + 1, size=n_races)
    return np.repeat(np.arange(n_races), n)


def labels(codes, prefix="jra:r"):
    names = np.array([f"{prefix}{c:05d}" for c in range(codes.max() + 1)], dtype=object)
    return names[codes]


def market(rng, codes):
    """A Dirichlet(1) market per race, clipped away from 0 and 1 and renormalised."""
    g = rng.gamma(1.0, size=len(codes))
    q = g / np.bincount(codes, weights=g)[codes]
    for _ in range(3):
        q = np.clip(q, 0.003, 0.75)
        q = q / np.bincount(codes, weights=q)[codes]
    return q


def race_mean(v, codes):
    return (np.bincount(codes, weights=v) / np.bincount(codes))[codes]


def draw_winners(rng, u, codes):
    """One winner per race drawn from softmax(u) within the race (Gumbel-max)."""
    g = u + rng.gumbel(size=len(u))
    top = pd.Series(g).groupby(codes).transform("max").to_numpy()
    return g == top


def first_principles(X, race_id, won, coef, offset=None, *, each_winner_counts_one=False):
    """(loglik, score, information) of the conditional logit at `coef` by the contract above.

    Races without a winner are dropped. Co-winners weigh 1/k each, or 1 each when
    `each_winner_counts_one` (the rule the contract rejects, used to show a test can fail)."""
    won = np.asarray(won, dtype=bool)
    X = np.asarray(X, dtype=float).reshape(len(won), -1)
    codes = pd.factorize(np.asarray(race_id, dtype=object))[0]
    k = np.bincount(codes, weights=won)
    keep = k[codes] > 0
    X, won, k_row = X[keep], won[keep], k[codes][keep]
    codes = pd.factorize(codes[keep])[0]
    u = X @ np.asarray(coef, dtype=float)
    if offset is not None:
        u = u + np.asarray(offset, dtype=float)[keep]
    top = pd.Series(u).groupby(codes).transform("max").to_numpy()
    e = np.exp(u - top)
    total = np.bincount(codes, weights=e)
    p = e / total[codes]
    lse = np.log(total[codes]) + top
    w = won.astype(float) if each_winner_counts_one else won / k_row
    race_weight = np.bincount(codes, weights=w)  # 1 per race, or k under the rejected rule
    loglik = float(np.sum(w * (u - lse)))
    xbar = np.stack([np.bincount(codes, weights=p * X[:, j]) for j in range(X.shape[1])], axis=1)
    score = X.T @ w - (race_weight[:, None] * xbar).sum(axis=0)
    info = (X * (p * race_weight[codes])[:, None]).T @ X - (xbar * race_weight[:, None]).T @ xbar
    return loglik, score, info


def assert_is_the_mle(fit, X, race_id, won, offset=None, *, ridge=0.0, scale=None):
    """coef is a root of the (penalised) score within 1% of a standard error, and loglik is the
    unpenalised log-likelihood there."""
    coef = np.asarray(fit.coef, dtype=float)
    loglik, score, info = first_principles(X, race_id, won, coef, offset)
    k = len(coef)
    step = np.linalg.solve(info + ridge * np.eye(k), score - ridge * coef)
    se = np.sqrt(np.diag(np.linalg.inv(info))) if scale is None else scale
    assert np.all(np.abs(step) < 0.01 * se), (step.tolist(), se.tolist())
    assert fit.loglik == pytest.approx(loglik, abs=1e-6)
    return info


def as_array(v):
    return np.asarray(v, dtype=float)


# ---------------------------------------------------------------------------------------------
# DoD (outline 1): the fit recovers planted coefficients, unbounded, with honest standard errors
# ---------------------------------------------------------------------------------------------


def test_conditional_logit_recovers_planted_coefficients():
    """3,000 races of 6 to 14 runners, two covariates with coefficients (0.8, -0.5): each estimate
    is within 3 standard errors, the data can tell each from 0, and the estimate is the maximum
    likelihood with the inverse-information standard errors (an inflated se cannot pass)."""
    rng = np.random.default_rng(8701)
    codes = field(rng, 3_000)
    X = rng.normal(size=(len(codes), 2))
    won = draw_winners(rng, X @ PLANTED, codes)
    race_id = labels(codes)

    fit = fit_conditional_logit(X, race_id, won)

    assert isinstance(fit, CLFit)
    coef, se = as_array(fit.coef), as_array(fit.se)
    assert coef.shape == (2,) and se.shape == (2,)
    assert fit.n_races == 3_000
    assert np.all(3 * se < np.abs(PLANTED)), se.tolist()
    assert np.all(np.abs(coef - PLANTED) < 3 * se), (coef.tolist(), se.tolist())
    info = assert_is_the_mle(fit, X, race_id, won)
    np.testing.assert_allclose(se, np.sqrt(np.diag(np.linalg.inv(info))), rtol=1e-3)


DESIGNS = {
    "one covariate": (8711, (1.5,)),
    "a large negative coefficient": (8712, (-2.0, 1.2)),
    "a zero among three": (8713, (0.0, 0.3, -0.6)),
}


@pytest.mark.parametrize("seed, planted", list(DESIGNS.values()), ids=list(DESIGNS))
def test_the_fit_is_the_maximum_likelihood_for_varied_designs(seed, planted):
    """Fields of 2 to 18 runners, one to three covariates, coefficients of either sign (no bound
    holds -2.0 back): the fit is the MLE, its se the inverse information, and it lands within 4 se
    of what was planted."""
    planted = np.asarray(planted)
    rng = np.random.default_rng(seed)
    codes = field(rng, 700, sizes=(2, 18))
    X = rng.normal(size=(len(codes), len(planted)))
    won = draw_winners(rng, X @ planted, codes)
    race_id = labels(codes)

    fit = fit_conditional_logit(X, race_id, won)

    coef, se = as_array(fit.coef), as_array(fit.se)
    assert coef.shape == planted.shape and se.shape == planted.shape
    assert fit.n_races == 700
    info = assert_is_the_mle(fit, X, race_id, won)
    np.testing.assert_allclose(se, np.sqrt(np.diag(np.linalg.inv(info))), rtol=1e-3)
    assert np.all(np.abs(coef - planted) < 4 * se), (coef.tolist(), se.tolist())


# ---------------------------------------------------------------------------------------------
# DoD (outline 1): an optional offset, at coefficient 1
# ---------------------------------------------------------------------------------------------


def test_an_offset_shifts_utilities_without_a_coefficient():
    """u = ln q + X @ (0.5, -0.8), the first covariate correlated with ln q. With ln q as the
    offset, coef has one entry per X column, recovers the planted values and is the MLE with the
    offset at coefficient 1. Without the offset the first covariate soaks up ln q; doubling the
    offset moves the fit (it is not re-estimated away); a constant added per race changes
    nothing, since the softmax is within race."""
    rng = np.random.default_rng(8702)
    codes = field(rng, 2_500)
    ln_q = np.log(market(rng, codes))
    centred = ln_q - race_mean(ln_q, codes)
    X = np.column_stack(
        [0.6 * centred + rng.normal(0, 0.7, len(codes)), rng.normal(size=len(codes))]
    )
    planted = np.array([0.5, -0.8])
    won = draw_winners(rng, ln_q + X @ planted, codes)
    race_id = labels(codes)

    fit = fit_conditional_logit(X, race_id, won, offset=ln_q)

    coef, se = as_array(fit.coef), as_array(fit.se)
    assert coef.shape == (2,) and se.shape == (2,)
    assert np.all(np.abs(coef - planted) < 3 * se), (coef.tolist(), se.tolist())
    info = assert_is_the_mle(fit, X, race_id, won, offset=ln_q)
    np.testing.assert_allclose(se, np.sqrt(np.diag(np.linalg.inv(info))), rtol=1e-3)

    plain = fit_conditional_logit(X, race_id, won)
    assert abs(as_array(plain.coef)[0] - planted[0]) > 5 * as_array(plain.se)[0]
    assert_is_the_mle(plain, X, race_id, won)

    doubled = fit_conditional_logit(X, race_id, won, offset=2.0 * ln_q)
    assert_is_the_mle(doubled, X, race_id, won, offset=2.0 * ln_q)
    assert abs(as_array(doubled.coef)[0] - coef[0]) > 3 * se[0]

    per_race = rng.normal(0, 3.0, size=codes.max() + 1)[codes]
    shifted = fit_conditional_logit(X, race_id, won, offset=ln_q + per_race)
    np.testing.assert_allclose(as_array(shifted.coef), coef, rtol=0, atol=0.01 * se.min())
    assert shifted.loglik == pytest.approx(fit.loglik, abs=1e-5)


# ---------------------------------------------------------------------------------------------
# The softmax is within race, whatever the row order or the label type
# ---------------------------------------------------------------------------------------------


def test_the_softmax_is_within_race_whatever_the_row_order():
    """Rows shuffled so races interleave, and integer race labels in place of strings: the same
    fit, and it is the MLE of the softmax grouped by label (not by contiguous runs or all rows)."""
    rng = np.random.default_rng(8703)
    codes = field(rng, 900, sizes=(3, 16))
    ln_q = np.log(market(rng, codes))
    X = rng.normal(size=(len(codes), 2))
    won = draw_winners(rng, ln_q + X @ np.array([1.0, -0.7]), codes)
    race_id = labels(codes)

    base = fit_conditional_logit(X, race_id, won, offset=ln_q)
    se = as_array(base.se)

    perm = rng.permutation(len(codes))
    shuffled = fit_conditional_logit(X[perm], race_id[perm], won[perm], offset=ln_q[perm])
    assert_is_the_mle(shuffled, X[perm], race_id[perm], won[perm], offset=ln_q[perm])
    integer_ids = (codes * 7 + 100)[perm]
    by_int = fit_conditional_logit(X[perm], integer_ids, won[perm], offset=ln_q[perm])

    for other in (shuffled, by_int):
        np.testing.assert_allclose(as_array(other.coef), as_array(base.coef), atol=0.01 * se.min())
        np.testing.assert_allclose(as_array(other.se), se, rtol=1e-3)
        assert other.loglik == pytest.approx(base.loglik, abs=1e-5)
        assert other.n_races == base.n_races == 900


# ---------------------------------------------------------------------------------------------
# Dead heats and void races, as core.eval.scoring scores them (#69)
# ---------------------------------------------------------------------------------------------


def test_a_dead_heat_weighs_each_co_winner_one_over_k_and_a_void_race_is_dropped():
    """1,500 races, 40% of them dead heats (the lowest-x runner joins the winner, a few three-way),
    plus 60 void races with extreme covariates. n_races counts the 1,500; the fit is the MLE with
    each co-winner at 1/k; counting each co-winner at 1 is a different estimate, so the rule is
    tested."""
    rng = np.random.default_rng(8704)
    n_races = 1_500
    codes = field(rng, n_races)
    x = rng.normal(size=len(codes))
    won = draw_winners(rng, 0.7 * x, codes)

    frame = pd.DataFrame({"code": codes, "x": x, "won": won})
    heats = rng.random(n_races) < 0.4
    for r in np.flatnonzero(heats):
        rows = frame.index[(frame["code"] == r) & ~frame["won"]]
        lowest = frame.loc[rows, "x"].nsmallest(2 if r % 5 == 0 else 1).index
        frame.loc[lowest, "won"] = True
    n_void = 60
    void_codes = np.repeat(np.arange(n_races, n_races + n_void), 8)
    void = pd.DataFrame(
        {"code": void_codes, "x": rng.normal(0, 4.0, len(void_codes)), "won": False}
    )
    frame = pd.concat([frame, void], ignore_index=True).sample(frac=1.0, random_state=4)

    X = frame[["x"]].to_numpy()
    won = frame["won"].to_numpy(dtype=bool)
    race_id = labels(frame["code"].to_numpy())
    assert (pd.Series(won).groupby(race_id).sum() >= 2).sum() == heats.sum()

    fit = fit_conditional_logit(X, race_id, won)

    assert fit.n_races == n_races
    info = assert_is_the_mle(fit, X, race_id, won)
    se = as_array(fit.se)
    np.testing.assert_allclose(se, np.sqrt(np.diag(np.linalg.inv(info))), rtol=1e-3)
    _, score_1, info_1 = first_principles(X, race_id, won, fit.coef, each_winner_counts_one=True)
    assert np.abs(np.linalg.solve(info_1, score_1)).max() > se.max()


# ---------------------------------------------------------------------------------------------
# The ridge shrinks toward 0, as a prior precision
# ---------------------------------------------------------------------------------------------


def test_the_ridge_shrinks_toward_zero_as_a_prior_precision():
    """With ln q as the offset, ridge 0, 20, 400 and 1e9: each fit solves the penalised score
    (score = ridge * coef), loglik is the unpenalised log-likelihood, |coef| falls strictly with
    the ridge without changing sign, and a huge ridge leaves the offset alone."""
    rng = np.random.default_rng(8705)
    codes = field(rng, 800)
    ln_q = np.log(market(rng, codes))
    X = rng.normal(size=(len(codes), 2))
    won = draw_winners(rng, ln_q + X @ PLANTED, codes)
    race_id = labels(codes)

    ridges = (0.0, 20.0, 400.0, 1e9)
    fits = {r: fit_conditional_logit(X, race_id, won, offset=ln_q, ridge=r) for r in ridges}
    se0 = as_array(fits[0.0].se)
    for r, fit in fits.items():
        assert as_array(fit.coef).shape == (2,)
        assert_is_the_mle(fit, X, race_id, won, offset=ln_q, ridge=r, scale=se0)

    norms = [float(np.linalg.norm(as_array(fits[r].coef))) for r in ridges]
    assert norms[0] > norms[1] > norms[2] > norms[3], norms
    assert norms[2] < 0.8 * norms[0], norms
    assert np.all(np.sign(as_array(fits[400.0].coef)) == np.sign(as_array(fits[0.0].coef)))
    assert norms[3] < 1e-6
    offset_alone, _, _ = first_principles(X, race_id, won, np.zeros(2), ln_q)
    assert fits[1e9].loglik == pytest.approx(offset_alone, abs=1e-3)


# ---------------------------------------------------------------------------------------------
# Inputs that cannot be fitted are refused
# ---------------------------------------------------------------------------------------------


def _good():
    rng = np.random.default_rng(8706)
    codes = np.repeat(np.arange(4), 3)
    X = rng.normal(size=(12, 2))
    won = np.zeros(12, dtype=bool)
    won[[0, 4, 7, 11]] = True
    return dict(X=X, race_id=labels(codes), won=won, offset=np.log(np.full(12, 1 / 3)))


def _with(**changes):
    def build():
        args = _good()
        for key, change in changes.items():
            args[key] = change(args[key])
        return args

    return build


def _set(i, j, value):
    def change(X):
        X = X.copy()
        X[i, j] = value
        return X

    return change


BAD = {
    "NaN in X": _with(X=_set(3, 1, np.nan)),
    "inf in X": _with(X=_set(5, 0, np.inf)),
    "NaN in the offset": _with(offset=lambda o: np.where(np.arange(12) == 2, np.nan, o)),
    "an offset of another length": _with(offset=lambda o: o[:-1]),
    "race_id of another length": _with(race_id=lambda r: r[:-1]),
    "won of another length": _with(won=lambda w: w[:-1]),
    "X with fewer rows": _with(X=lambda X: X[:-2]),
}


@pytest.mark.parametrize("build", list(BAD.values()), ids=list(BAD))
def test_inputs_that_do_not_line_up_or_are_not_finite_are_refused(build):
    args = build()
    with pytest.raises(ValueError):
        fit_conditional_logit(args["X"], args["race_id"], args["won"], offset=args["offset"])


def test_a_negative_ridge_is_refused():
    args = _good()
    with pytest.raises(ValueError):
        fit_conditional_logit(args["X"], args["race_id"], args["won"], ridge=-1.0)
