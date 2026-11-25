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
- `cluster` (review finding on #87; D7 signs the meeting, venue x day, as the inference unit):
  None, or one label per row (any hashable), constant within a race. `CLFit.se_cluster` is then the
  meeting-clustered sandwich sqrt(diag(H^-1 B H^-1)), with H the information at `coef` plus
  diag(ridge) (the same bread as `se`) and B = sum over clusters of g_c g_c^T, g_c the sum over
  the cluster's races of sum_i (w_i - p_i) x_i (w = 1/k on each of a race's k winners; a void race
  contributes nothing). No small-sample factor. `se_cluster` is None without `cluster`, and
  `coef`, `se`, `loglik` and `n_races` do not depend on it. A race whose rows carry two labels, a
  missing label or a cluster array of another length is a ValueError.

Expected values come from first principles (the score, information and log-likelihood written out
below), never from the code under test. The maximum-likelihood checks allow the fit to stop within
1% of a standard error of the root, which any decent optimiser does and no wrong objective can.
All data is synthetic and every RNG is seeded.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math

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


# ---------------------------------------------------------------------------------------------
# Meeting-clustered standard errors (review finding on #87; D7: the inference unit is the meeting)
# ---------------------------------------------------------------------------------------------

VENUES = np.array(["TOK", "NAK", "KYO"], dtype=object)


def meeting_labels(meeting):
    """A venue x day label per meeting code: '2025-01-04:TOK', '2025-01-04:NAK', ..."""
    meeting = np.asarray(meeting)
    names = np.array(
        [
            f"{np.datetime64('2025-01-04') + m // 3}:{VENUES[m % 3]}"
            for m in range(meeting.max() + 1)
        ],
        dtype=object,
    )
    return names[meeting]


def cluster_sandwich(X, race_id, won, coef, cluster, offset=None, *, ridge=0.0):
    """sqrt(diag(H^-1 B H^-1)) by the contract: H the information at coef plus diag(ridge), B the
    sum over clusters of g_c g_c^T, g_c the sum over the cluster's races of sum_i (w_i - p_i) x_i
    with w = 1/k on each of k co-winners. Void races are dropped; no small-sample factor."""
    won = np.asarray(won, dtype=bool)
    X = np.asarray(X, dtype=float).reshape(len(won), -1)
    codes = pd.factorize(np.asarray(race_id, dtype=object))[0]
    k = np.bincount(codes, weights=won)
    keep = k[codes] > 0
    Xk = X[keep]
    race = pd.factorize(codes[keep])[0]
    u = Xk @ np.asarray(coef, dtype=float)
    if offset is not None:
        u = u + np.asarray(offset, dtype=float)[keep]
    top = pd.Series(u).groupby(race).transform("max").to_numpy()
    e = np.exp(u - top)
    p = e / np.bincount(race, weights=e)[race]
    w = won[keep] / k[codes][keep]
    per_row = (w - p)[:, None] * Xk
    unit = pd.factorize(np.asarray(cluster, dtype=object)[keep])[0]
    g = np.stack([np.bincount(unit, weights=per_row[:, j]) for j in range(Xk.shape[1])], axis=1)
    _, _, info = first_principles(X, race_id, won, coef, offset)
    penalty = np.broadcast_to(np.asarray(ridge, dtype=float), (Xk.shape[1],))
    bread = np.linalg.inv(info + np.diag(penalty))
    return np.sqrt(np.diag(bread @ (g.T @ g) @ bread))


def assert_clustering_changes_only_se_cluster(clustered, plain):
    """coef, se, loglik and n_races are those of the fit without `cluster`, which has no
    se_cluster."""
    assert plain.se_cluster is None
    scale = as_array(plain.se)
    moved = np.abs(as_array(clustered.coef) - as_array(plain.coef))
    assert np.all(moved < 1e-6 * scale), (moved.tolist(), scale.tolist())
    np.testing.assert_allclose(as_array(clustered.se), scale, rtol=1e-6)
    assert clustered.loglik == pytest.approx(plain.loglik, abs=1e-8)
    assert clustered.n_races == plain.n_races


TOK4, KYO4, TOK5 = "2025-01-04:TOK", "2025-01-04:KYO", "2025-01-05:TOK"

# (race, meeting, x1, x2, q, winners by position): r6 is a dead heat, r9 is void.
HAND_SIZED = [
    ("r1", TOK4, [1.0, 0.0, -1.0], [0.5, -0.5, 0.0], [0.5, 0.3, 0.2], [0]),
    ("r2", TOK4, [0.2, -0.4, 0.8, 0.0], [1.0, 0.0, -1.0, 0.5], [0.4, 0.3, 0.2, 0.1], [2]),
    ("r3", TOK4, [-0.5, 0.5, 0.0], [0.0, 1.0, -0.5], [0.3, 0.3, 0.4], [0]),
    ("r4", KYO4, [0.3, -0.3, 1.2], [-1.0, 0.4, 0.2], [0.2, 0.5, 0.3], [1]),
    ("r5", KYO4, [0.0, 1.0, -0.6, 0.4], [0.3, -0.2, 0.8, -0.7], [0.25] * 4, [1]),
    ("r6", KYO4, [0.7, -0.2, 0.1], [0.1, 0.9, -0.4], [0.3, 0.3, 0.4], [0, 1]),
    ("r7", TOK5, [-1.0, 0.6, 0.2], [0.6, -0.3, 0.1], [0.6, 0.25, 0.15], [2]),
    ("r8", TOK5, [0.5, 0.1, -0.3], [-0.6, 0.7, 0.2], [0.35, 0.35, 0.3], [0]),
    ("r9", TOK5, [3.0, -3.0, 2.0], [2.5, -2.0, 4.0], [0.2, 0.5, 0.3], []),
]


def test_the_clustered_se_is_the_meeting_sandwich_on_a_hand_sized_example():
    """Nine races at three meetings (a dead heat at the second, a void race with extreme
    covariates at the third), two covariates and an offset. se_cluster is sqrt(diag(H^-1 B H^-1))
    written out race by race: the void race adds nothing, the dead heat's co-winners weigh 1/2, and
    there is no small-sample factor (G/(G-1) would be 1.5 here). Clustering leaves coef, se,
    loglik and n_races as they were; without `cluster` there is no se_cluster."""
    rows = [
        (race, meeting, x1[i], x2[i], math.log(q[i]), i in winners)
        for race, meeting, x1, x2, q, winners in HAND_SIZED
        for i in range(len(x1))
    ]
    frame = pd.DataFrame(rows, columns=["race", "meeting", "x1", "x2", "ln_q", "won"])
    X = frame[["x1", "x2"]].to_numpy()
    race_id, won = frame["race"].to_numpy(dtype=object), frame["won"].to_numpy(dtype=bool)
    offset, meeting = frame["ln_q"].to_numpy(), frame["meeting"].to_numpy(dtype=object)

    fit = fit_conditional_logit(X, race_id, won, offset=offset, cluster=meeting)
    plain = fit_conditional_logit(X, race_id, won, offset=offset)

    coef = as_array(fit.coef)
    H = np.zeros((2, 2))
    g = {m: np.zeros(2) for m in frame["meeting"].unique()}
    for (_, m), race in frame.groupby(["race", "meeting"], sort=False):
        w = race["won"].to_numpy(dtype=float)
        if w.sum() == 0:
            continue
        w = w / w.sum()
        Xr = race[["x1", "x2"]].to_numpy()
        u = race["ln_q"].to_numpy() + Xr @ coef
        p = np.exp(u - u.max())
        p = p / p.sum()
        xbar = p @ Xr
        H += (Xr * p[:, None]).T @ Xr - np.outer(xbar, xbar)
        g[m] += (w - p) @ Xr
    assert np.abs(sum(g.values())).max() < 1e-6  # the score is 0 at the MLE
    B = sum(np.outer(v, v) for v in g.values())
    bread = np.linalg.inv(H)
    want = np.sqrt(np.diag(bread @ B @ bread))

    assert fit.n_races == 8
    se_cluster = as_array(fit.se_cluster)
    assert se_cluster.shape == (2,)
    np.testing.assert_allclose(se_cluster, want, rtol=1e-6)
    np.testing.assert_allclose(as_array(fit.se), np.sqrt(np.diag(bread)), rtol=1e-6)
    assert np.abs(want / as_array(fit.se) - 1).min() > 0.2  # not the model-based se
    assert_clustering_changes_only_se_cluster(fit, plain)


def _clustered_design(seed, planted, n_races, meeting_sizes, *, offset, heats, n_void):
    """Races grouped into meetings of `meeting_sizes` races (drawn per meeting), winners from
    u = offset + X @ planted, optionally dead heats (the lowest-x0 runner joins the winner) and
    void races with extreme covariates placed at existing meetings."""
    rng = np.random.default_rng(seed)
    planted = np.asarray(planted, dtype=float)
    sizes = rng.integers(meeting_sizes[0], meeting_sizes[1] + 1, size=n_races)
    meeting_of_race = np.repeat(np.arange(n_races), sizes)[:n_races]
    codes = field(rng, n_races, sizes=(4, 16))
    X = rng.normal(size=(len(codes), len(planted)))
    ln_q = np.log(market(rng, codes)) if offset else np.zeros(len(codes))
    won = draw_winners(rng, ln_q + X @ planted, codes)
    for r in np.flatnonzero(rng.random(n_races) < heats):
        rows = np.flatnonzero((codes == r) & ~won)
        won[rows[np.argmin(X[rows, 0])]] = True
    void_codes = np.repeat(np.arange(n_races, n_races + n_void), 8)
    void_meeting = rng.integers(0, meeting_of_race.max() + 1, size=n_void)
    codes = np.r_[codes, void_codes]
    X = np.r_[X, rng.normal(0, 4.0, size=(len(void_codes), len(planted)))]
    ln_q = np.r_[ln_q, np.log(np.full(len(void_codes), 1 / 8))]
    won = np.r_[won, np.zeros(len(void_codes), dtype=bool)]
    meeting = np.r_[meeting_of_race, void_meeting][codes]
    return X, labels(codes), won, (ln_q if offset else None), meeting_labels(meeting)


CLUSTER_DESIGNS = {
    "one covariate, meetings of 12 races": (8791, (1.5,), (12, 12), False, 0.0, 0),
    "two covariates and an offset, meetings of 2 to 20": (8792, (0.8, -0.5), (2, 20), True, 0.0, 0),
    "three covariates, dead heats and void races": (8793, (0.0, 0.3, -0.6), (6, 6), True, 0.3, 40),
}


@pytest.mark.parametrize(
    "seed, planted, meeting_sizes, offset, heats, n_void",
    list(CLUSTER_DESIGNS.values()),
    ids=list(CLUSTER_DESIGNS),
)
def test_the_clustered_se_is_the_sandwich_for_varied_designs(
    seed, planted, meeting_sizes, offset, heats, n_void
):
    """600 races, one to three covariates, meetings of fixed or varied size, with and without an
    offset, dead heats and void races (at existing meetings, so counting them would move B): the
    clustered se is the contract's sandwich and clustering changes nothing else."""
    X, race_id, won, ln_q, meeting = _clustered_design(
        seed, planted, 600, meeting_sizes, offset=offset, heats=heats, n_void=n_void
    )

    fit = fit_conditional_logit(X, race_id, won, offset=ln_q, cluster=meeting)
    plain = fit_conditional_logit(X, race_id, won, offset=ln_q)

    assert fit.n_races == 600
    se_cluster = as_array(fit.se_cluster)
    assert se_cluster.shape == (len(planted),)
    want = cluster_sandwich(X, race_id, won, fit.coef, meeting, ln_q)
    np.testing.assert_allclose(se_cluster, want, rtol=1e-6)
    assert_clustering_changes_only_se_cluster(fit, plain)


def test_with_every_race_its_own_cluster_the_clustered_se_is_the_race_level_robust_se():
    """3,000 correctly specified independent races: with each race its own cluster (by its label
    or an integer per race), se_cluster is the race-level robust (HC0) sandwich, the scores summed
    within each race first, and within 15% of the model-based se. A sandwich over single runners
    is a different number, so the race is the unit inside a cluster."""
    rng = np.random.default_rng(8794)
    codes = field(rng, 3_000)
    ln_q = np.log(market(rng, codes))
    X = rng.normal(size=(len(codes), 2))
    won = draw_winners(rng, ln_q + X @ PLANTED, codes)
    race_id = labels(codes)

    want = None
    for cluster in (race_id, codes * 3 + 11):
        fit = fit_conditional_logit(X, race_id, won, offset=ln_q, cluster=cluster)
        se_cluster = as_array(fit.se_cluster)
        want = cluster_sandwich(X, race_id, won, fit.coef, codes, ln_q)
        np.testing.assert_allclose(se_cluster, want, rtol=1e-6)
        assert np.all(np.abs(se_cluster / as_array(fit.se) - 1) < 0.15), (se_cluster, fit.se)

    by_runner = cluster_sandwich(X, race_id, won, fit.coef, np.arange(len(codes)), ln_q)
    assert np.abs(by_runner / want - 1).max() > 0.05


def _meeting_shock(rng, n_meetings, races_per_meeting, tau):
    """Meetings of races; u = (0.8 + b_m) x1 - 0.5 x2 with b_m ~ N(0, tau^2) shared by every race
    of meeting m (a meeting-level random effect on x1's utility)."""
    codes = field(rng, n_meetings * races_per_meeting)
    meeting = codes // races_per_meeting
    X = rng.normal(size=(len(codes), 2))
    b = rng.normal(0, tau, size=n_meetings)
    u = (PLANTED[0] + b[meeting]) * X[:, 0] + PLANTED[1] * X[:, 1]
    return codes, meeting, X, draw_winners(rng, u, codes)


def test_a_shared_meeting_shock_widens_the_clustered_se():
    """250 meetings of 12 races with a meeting-level shock on x1's coefficient (sd 0.5): the
    meeting-clustered se of x1 is more than 1.3 times the race-independent se, while x2's, which
    the shock does not touch, stays within 15% of its se. The same covariates without the shock
    give a clustered se within 15% of se for both."""
    rng = np.random.default_rng(8795)
    codes, meeting, X, won = _meeting_shock(rng, 250, 12, tau=0.5)
    race_id, cluster = labels(codes), meeting_labels(meeting)

    fit = fit_conditional_logit(X, race_id, won, cluster=cluster)

    se, se_cluster = as_array(fit.se), as_array(fit.se_cluster)
    np.testing.assert_allclose(
        se_cluster, cluster_sandwich(X, race_id, won, fit.coef, cluster), rtol=1e-6
    )
    assert se_cluster[0] > 1.3 * se[0], (se_cluster, se)
    assert abs(se_cluster[1] / se[1] - 1) < 0.15, (se_cluster, se)

    calm = draw_winners(rng, X @ PLANTED, codes)
    fit = fit_conditional_logit(X, race_id, calm, cluster=cluster)
    se, se_cluster = as_array(fit.se), as_array(fit.se_cluster)
    np.testing.assert_allclose(
        se_cluster, cluster_sandwich(X, race_id, calm, fit.coef, cluster), rtol=1e-6
    )
    assert np.all(np.abs(se_cluster / se - 1) < 0.15), (se_cluster, se)


def test_the_clustered_se_matches_the_spread_of_the_estimate_across_replicates():
    """60 replicate seasons of 120 meetings x 12 races under the meeting shock. The spread of the
    x1 estimate across replicates is the standard error's target. (A random slope pulls the
    fixed-slope estimate below 0.8, so the replicates' mean stands in for the true value.) The
    mean clustered se is within 25% of that spread and its 95% interval covers the mean in at
    least 85% of replicates; the race-independent se is under 0.8 of the spread and covers less
    often."""
    estimates, ses, clustered = [], [], []
    for r in range(60):
        rng = np.random.default_rng(88_000 + r)
        codes, meeting, X, won = _meeting_shock(rng, 120, 12, tau=0.5)
        fit = fit_conditional_logit(X, codes, won, cluster=meeting)
        estimates.append(as_array(fit.coef)[0])
        ses.append(as_array(fit.se)[0])
        clustered.append(as_array(fit.se_cluster)[0])
    estimates, ses, clustered = map(np.asarray, (estimates, ses, clustered))
    spread = estimates.std(ddof=1)
    miss = np.abs(estimates - estimates.mean())

    assert abs(clustered.mean() / spread - 1) < 0.25, (clustered.mean(), spread)
    assert ses.mean() < 0.8 * spread, (ses.mean(), spread)
    covered, naive = (miss < 1.96 * clustered).mean(), (miss < 1.96 * ses).mean()
    assert covered >= 0.85 and naive < covered, (covered, naive)


def test_the_clustered_se_ignores_row_order_and_cluster_label_type():
    """900 races at meetings of 9, with an offset: rows shuffled so races and meetings interleave,
    integer meeting labels, integer race ids and both at once give the same se_cluster."""
    rng = np.random.default_rng(8796)
    codes = field(rng, 900, sizes=(3, 16))
    meeting = codes // 9
    ln_q = np.log(market(rng, codes))
    X = rng.normal(size=(len(codes), 2))
    won = draw_winners(rng, ln_q + X @ np.array([1.0, -0.7]), codes)
    race_id, cluster = labels(codes), meeting_labels(meeting)

    base = fit_conditional_logit(X, race_id, won, offset=ln_q, cluster=cluster)
    want = as_array(base.se_cluster)
    np.testing.assert_allclose(
        want, cluster_sandwich(X, race_id, won, base.coef, cluster, ln_q), rtol=1e-6
    )

    perm = rng.permutation(len(codes))
    variants = {
        "shuffled": (race_id[perm], cluster[perm]),
        "integer meetings": (race_id[perm], (meeting * 5 + 2)[perm]),
        "integer races and meetings": ((codes * 7 + 100)[perm], (meeting * 5 + 2)[perm]),
    }
    for name, (rid, clu) in variants.items():
        fit = fit_conditional_logit(X[perm], rid, won[perm], offset=ln_q[perm], cluster=clu)
        np.testing.assert_allclose(as_array(fit.se_cluster), want, rtol=1e-6, err_msg=name)


def _split_a_race(cluster):
    """The second row of race 0 moves to the meeting of races 2 and 3."""
    cluster = cluster.copy()
    cluster[1] = cluster[6]
    return cluster


BAD_CLUSTERS = {
    "a race whose rows carry two meetings": _split_a_race,
    "a cluster array one row short": lambda c: c[:-1],
    "a cluster array one row long": lambda c: np.r_[c, c[-1:]],
    "a missing meeting label": lambda c: np.where(np.arange(len(c)) == 4, None, c),
}


@pytest.mark.parametrize("change", list(BAD_CLUSTERS.values()), ids=list(BAD_CLUSTERS))
def test_a_cluster_that_splits_a_race_or_does_not_line_up_is_refused(change):
    """Four races of three runners at two meetings (races 0 and 1 at the first): each defect is a
    ValueError, and the clean array fits."""
    args = _good()
    cluster = np.array(["2025-01-04:TOK"] * 6 + ["2025-01-04:KYO"] * 6, dtype=object)
    fit_conditional_logit(args["X"], args["race_id"], args["won"], cluster=cluster)
    with pytest.raises(ValueError):
        fit_conditional_logit(args["X"], args["race_id"], args["won"], cluster=change(cluster))


def test_under_a_ridge_the_bread_is_the_penalised_information():
    """500 races at meetings of 10, with an offset: under a scalar ridge (30) and a per-column one
    (0 and 60), se_cluster is sqrt(diag(H^-1 B H^-1)) with H = I + diag(ridge) at the penalised
    coef, and the unpenalised bread gives a different number, so the test can tell them apart."""
    rng = np.random.default_rng(8797)
    codes = field(rng, 500)
    ln_q = np.log(market(rng, codes))
    X = rng.normal(size=(len(codes), 2))
    won = draw_winners(rng, ln_q + X @ PLANTED, codes)
    race_id, cluster = labels(codes), meeting_labels(codes // 10)

    for ridge in (30.0, np.array([0.0, 60.0])):
        fit = fit_conditional_logit(X, race_id, won, offset=ln_q, ridge=ridge, cluster=cluster)
        plain = fit_conditional_logit(X, race_id, won, offset=ln_q, ridge=ridge)
        se_cluster = as_array(fit.se_cluster)
        want = cluster_sandwich(X, race_id, won, fit.coef, cluster, ln_q, ridge=ridge)
        np.testing.assert_allclose(se_cluster, want, rtol=1e-6)
        unpenalised = cluster_sandwich(X, race_id, won, fit.coef, cluster, ln_q)
        assert np.abs(unpenalised / want - 1).max() > 0.01
        assert_clustering_changes_only_se_cluster(fit, plain)
