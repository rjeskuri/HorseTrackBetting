"""The C4 meta-model at the final price (ticket #87, epic #73; 08 §3.2 C4, §4 E9, E40, R13).

Contract pinned here (`horsetrack.countries.japan.models.meta`):
- `MetaSpec(members, quadratic, segment_cols, form, ridge)`; `form` is 'stage2' or 'offset', and
  any other form is a ValueError (at construction or at `fit_meta`). A member that is not a column
  of the frame is a KeyError or ValueError.
- `fit_meta(frame, spec) -> MetaFit`. The frame has race_id, horse_id, won, ln_q_final, one
  out-of-fold log-probability column per member and the segment columns. With quadratic=False the
  utility of runner i is
      stage2:  u_i = sum_k alpha_k(z_i) * m_ki + beta * ln q_i      (alpha and beta unbounded)
      offset:  u_i = ln q_i + sum_k alpha_k(z_i) * m_ki             (q's coefficient fixed at 1)
  with alpha_k(z) = alpha_k0 + sum_j alpha_kj * z_j over `segment_cols` (raw values, an
  interaction column m_k * z_j per pair: feature-weighted stacking), and P(i wins) = softmax of u
  within the race. The fit is the conditional-logit MLE, races treated as in
  `test_condlogit.py` (void dropped, a dead heat at 1/k per co-winner).
- `MetaFit` is a dataclass whose fields include `coef` and `se`, pd.Series of floats on one
  index of term names: the member name for alpha_k0, "<member>:<segment col>" for alpha_kj, and
  "ln_q_final" for beta (stage2 only: the offset form has no q term). With quadratic=False these
  are exactly the terms; quadratic=True adds terms to them. At ridge 0, `se` is the inverse
  information at `coef`. `predict` reads its weights from `coef`.
- `ridge` shrinks the member weights (alpha_k0 and alpha_kj) toward 0. By the owner's decision of
  2026-11-24 the shrinkage target is alpha = 0, the calibrated market alone, so beta is not shrunk:
  under a huge ridge the stage-2 model is a power of q and the offset model is q.
- `MetaFit.predict(frame, q) -> np.ndarray`: one P(win | x, q) per frame row, in row order, for any
  price vector `q` (one probability per row, any positive scale within a race: only ratios within
  a race matter). It reads race_id, the members and the segment columns only; no won column is
  needed and a ln_q_final column, if present, is ignored. Each race sums to 1.
- `oof_predictions(frame, folds, fit, predict) -> pd.Series`: `folds` is a list of
  core.eval.folds.Fold (positions into `frame`), `fit(train_frame) -> model` is called on
  frame.iloc[fold.train_idx] and `predict(model, test_frame) -> array` on frame.iloc[fold.test_idx]
  (one value per test row). The Series is indexed by frame labels, unique; a row in no test window
  is NaN or absent. A fold whose training and test rows share a row, or two folds whose test
  windows share a row, is a ValueError.
- `encompassing(frame, member) -> (coef, se)`: the member's coefficient and its standard error in
  an unpenalised stage-2 fit on that member and ln_q_final alone (Fair-Shiller).
- `ceiling_table(frame, *, pi_col, q_col, thresholds) -> pd.DataFrame`: one row per threshold, in
  the order given, with columns `threshold` (as given, on the ln(pi/q) scale), `q_mass` (the share
  of the total q mass on runners with ln(pi/q) > threshold: overlays, one-sided) and `rms` (the
  q-weighted root mean square of ln(pi/q) over all runners, the same on every row). Inputs are
  per-race probabilities.
- Meeting-clustered standard errors (review finding on #87; D7's inference unit is the meeting,
  D16 reads the encompassing interval): `fit_meta(frame, spec, *, cluster_col=None)` and
  `encompassing(frame, member, *, cluster_col=None)`. With `cluster_col`, the column of that name
  (constant within a race) is the cluster of each row; `MetaFit.se_cluster` is a pd.Series on the
  index of `coef`, the sandwich sqrt(diag(H^-1 B H^-1)) of the fitted design (H the information
  plus the spec's penalty, which is 0 on the market terms; B the sum over clusters of the outer
  product of the summed per-race score, as in `test_condlogit.py`), and encompassing's second
  element is that clustered se of the member. Without `cluster_col`, `se_cluster` is None and both
  return what they return without it; `coef`, `se` and the encompassing coefficient never depend
  on it. A missing column is a KeyError or ValueError, one that splits a race a ValueError.

Expected values come from first principles (the utility, score and information written out
below, a reference Newton fit for encompassing, hand-computed ceiling figures), never from the
code under test. All data is synthetic and every RNG is seeded.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import dataclasses
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.folds import Fold, FoldScheme, walk_forward
from horsetrack.countries.japan.models.meta import (
    MetaFit,
    MetaSpec,
    ceiling_table,
    encompassing,
    fit_meta,
    oof_predictions,
)

Q = "ln_q_final"
MEMBERS = ["p_form", "p_speed"]


# ---------------------------------------------------------------------------------------------
# Synthetic races
# ---------------------------------------------------------------------------------------------


def field(rng, n_races, sizes=(6, 14)):
    n = rng.integers(sizes[0], sizes[1] + 1, size=n_races)
    return np.repeat(np.arange(n_races), n)


def market(rng, codes):
    """A Dirichlet(1) market per race, clipped away from 0 and 1 and renormalised."""
    g = rng.gamma(1.0, size=len(codes))
    q = g / np.bincount(codes, weights=g)[codes]
    for _ in range(3):
        q = np.clip(q, 0.003, 0.75)
        q = q / np.bincount(codes, weights=q)[codes]
    return q


def codes_of(race_id):
    return pd.factorize(np.asarray(race_id, dtype=object))[0]


def softmax_in_race(u, codes):
    top = pd.Series(u).groupby(codes).transform("max").to_numpy()
    e = np.exp(u - top)
    return e / np.bincount(codes, weights=e)[codes]


def log_softmax(v, codes):
    return np.log(softmax_in_race(v, codes))


def draw_winners(rng, u, codes):
    """One winner per race drawn from softmax(u) within the race (Gumbel-max)."""
    g = u + rng.gumbel(size=len(u))
    top = pd.Series(g).groupby(codes).transform("max").to_numpy()
    return g == top


def make_frame(codes, q, won=None, *, prefix="jra:r", **columns):
    names = np.array([f"{prefix}{c:05d}" for c in range(codes.max() + 1)], dtype=object)
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    first = np.zeros(codes.max() + 1, dtype=int)
    first[codes[starts]] = starts
    data = {"race_id": names[codes], "horse_id": np.arange(len(codes)) - first[codes] + 1}
    if won is not None:
        data["won"] = np.asarray(won, dtype=bool)
        data[Q] = np.log(q)
    data.update(columns)
    return pd.DataFrame(data)


def spec(members, *, form="stage2", segment_cols=(), ridge=0.0, quadratic=False):
    return MetaSpec(
        members=list(members),
        quadratic=quadratic,
        segment_cols=list(segment_cols),
        form=form,
        ridge=ridge,
    )


def expected_terms(members, segment_cols, form):
    terms = list(members) + [f"{m}:{z}" for m in members for z in segment_cols]
    return set(terms + ([Q] if form == "stage2" else []))


def design(frame, terms, q):
    """The design column of each term, by the naming in the contract."""
    cols = []
    for term in terms:
        if term == Q:
            cols.append(np.log(np.asarray(q, dtype=float)))
        elif ":" in term:
            m, z = term.split(":")
            cols.append(frame[m].to_numpy(dtype=float) * frame[z].to_numpy(dtype=float))
        else:
            cols.append(frame[term].to_numpy(dtype=float))
    return np.column_stack(cols)


def reference_predict(frame, coef, q, form):
    """softmax within race of the utility the contract writes for `coef` at price vector q."""
    terms = list(coef.index)
    u = design(frame, terms, q) @ coef.to_numpy(dtype=float)
    if form == "offset":
        u = u + np.log(np.asarray(q, dtype=float))
    return softmax_in_race(u, codes_of(frame["race_id"]))


def score_and_information(X, race_id, won, coef, offset=None):
    """Score and information of the conditional logit at coef (one unit per race, void dropped)."""
    won = np.asarray(won, dtype=bool)
    codes = codes_of(race_id)
    k = np.bincount(codes, weights=won)
    keep = k[codes] > 0
    X, won, k_row = X[keep], won[keep], k[codes][keep]
    codes = pd.factorize(codes[keep])[0]
    u = X @ np.asarray(coef, dtype=float)
    if offset is not None:
        u = u + np.asarray(offset, dtype=float)[keep]
    p = softmax_in_race(u, codes)
    w = won / k_row
    xbar = np.stack([np.bincount(codes, weights=p * X[:, j]) for j in range(X.shape[1])], axis=1)
    score = X.T @ w - xbar.sum(axis=0)
    info = (X * p[:, None]).T @ X - xbar.T @ xbar
    return score, info


def assert_mle_with_honest_se(fit, frame, form):
    """At ridge 0: coef is within 1% of a standard error of the score's root and se is the
    inverse information there, for the design the term names describe."""
    terms = list(fit.coef.index)
    q = np.exp(frame[Q].to_numpy())
    X = design(frame, terms, q)
    offset = np.log(q) if form == "offset" else None
    coef = fit.coef.to_numpy(dtype=float)
    score, info = score_and_information(X, frame["race_id"], frame["won"], coef, offset)
    se = np.sqrt(np.diag(np.linalg.inv(info)))
    step = np.linalg.solve(info, score)
    assert np.all(np.abs(step) < 0.01 * se), dict(zip(terms, step.tolist(), strict=True))
    np.testing.assert_allclose(fit.se.reindex(terms).to_numpy(dtype=float), se, rtol=1e-3)


def reference_mle(X, race_id, won, offset=None):
    """A Newton fit of the conditional logit, with step halving on the score norm."""
    coef = np.zeros(X.shape[1])
    for _ in range(100):
        score, info = score_and_information(X, race_id, won, coef, offset)
        step = np.linalg.solve(info, score)
        t = 1.0
        while t > 1e-6:
            trial = coef + t * step
            s2, _ = score_and_information(X, race_id, won, trial, offset)
            if np.linalg.norm(s2) < np.linalg.norm(score) or np.abs(step).max() < 1e-9:
                break
            t /= 2
        coef = trial
        if np.abs(t * step).max() < 1e-12:
            break
    _, info = score_and_information(X, race_id, won, coef, offset)
    return coef, np.sqrt(np.diag(np.linalg.inv(info)))


def race_sums(p, race_id):
    return pd.Series(np.asarray(p, dtype=float)).groupby(np.asarray(race_id, dtype=object)).sum()


def winner_loglik(p, frame):
    return float(np.log(np.asarray(p)[frame["won"].to_numpy(dtype=bool)]).sum())


@pytest.fixture(scope="module")
def planted():
    """1,500 races: two members (independent signals as log-probabilities) and a race-level
    segment z_class in {0, 1, 2}; winners from u = 0.9 ln q + (0.4 + 0.2 z) p_form + 0.3 p_speed."""
    rng = np.random.default_rng(8720)
    codes = field(rng, 1_500)
    q = market(rng, codes)
    p_form = log_softmax(rng.normal(size=len(codes)), codes)
    p_speed = log_softmax(rng.normal(size=len(codes)), codes)
    z = rng.integers(0, 3, size=codes.max() + 1)[codes].astype(float)
    u = 0.9 * np.log(q) + (0.4 + 0.2 * z) * p_form + 0.3 * p_speed
    won = draw_winners(rng, u, codes)
    return make_frame(codes, q, won, p_form=p_form, p_speed=p_speed, z_class=z)


def new_races(seed, n_races, sizes=(3, 12)):
    """Races to predict: race_id, horse_id, the members and z_class only, with a market q."""
    rng = np.random.default_rng(seed)
    codes = field(rng, n_races, sizes=sizes)
    q = market(rng, codes)
    frame = make_frame(
        codes,
        q,
        prefix="jra:new",
        p_form=log_softmax(rng.normal(size=len(codes)), codes),
        p_speed=log_softmax(rng.normal(size=len(codes)), codes),
        z_class=rng.integers(0, 3, size=codes.max() + 1)[codes].astype(float),
    )
    return frame, q


def three_runners():
    return pd.DataFrame(
        {
            "race_id": ["jra:2026-11-28:TOK:11"] * 3,
            "horse_id": [1, 2, 3],
            "p_form": np.log([0.2, 0.3, 0.5]),
            "p_speed": np.log([0.6, 0.25, 0.15]),
            "z_class": [2.0, 2.0, 2.0],
        }
    )


# ---------------------------------------------------------------------------------------------
# Broad: both forms are the softmax of the stated utility, fitted by maximum likelihood
# ---------------------------------------------------------------------------------------------

FORMS = {
    "stage2": ("stage2", []),
    "stage2 with a segment": ("stage2", ["z_class"]),
    "offset": ("offset", []),
    "offset with a segment": ("offset", ["z_class"]),
}


@pytest.mark.parametrize("form, segment_cols", list(FORMS.values()), ids=list(FORMS))
def test_predict_is_the_softmax_of_the_fitted_utility(planted, form, segment_cols):
    fit = fit_meta(planted, spec(MEMBERS, form=form, segment_cols=segment_cols))

    assert isinstance(fit, MetaFit)
    assert set(fit.coef.index) == expected_terms(MEMBERS, segment_cols, form)
    assert set(fit.se.index) == set(fit.coef.index)
    assert_mle_with_honest_se(fit, planted, form)

    q = np.exp(planted[Q].to_numpy())
    got = np.asarray(fit.predict(planted, q), dtype=float)
    assert got.shape == (len(planted),)
    np.testing.assert_allclose(got, reference_predict(planted, fit.coef, q, form), atol=1e-10)
    np.testing.assert_allclose(race_sums(got, planted["race_id"]), 1.0, atol=1e-12)

    fresh, q_new = new_races(8721, 60)
    got = np.asarray(fit.predict(fresh, q_new), dtype=float)
    np.testing.assert_allclose(got, reference_predict(fresh, fit.coef, q_new, form), atol=1e-10)


def test_predict_takes_any_price_vector(planted):
    """The same three runners under q = (0.5, 0.3, 0.2) and (0.2, 0.3, 0.5): two different
    predictions, each summing to 1 and each the fitted utility's softmax. Only ratios of q within
    a race matter, rows may come in any order, and a ln_q_final column in the frame is ignored."""
    runners = three_runners()
    q1, q2 = np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.3, 0.5])
    for form in ("stage2", "offset"):
        fit = fit_meta(planted, spec(MEMBERS, form=form, segment_cols=["z_class"]))
        p1 = np.asarray(fit.predict(runners, q1), dtype=float)
        p2 = np.asarray(fit.predict(runners, q2), dtype=float)
        assert p1.sum() == pytest.approx(1.0, abs=1e-12)
        assert p2.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.abs(p1 - p2).max() > 0.05
        np.testing.assert_allclose(p1, reference_predict(runners, fit.coef, q1, form), atol=1e-10)
        np.testing.assert_allclose(p2, reference_predict(runners, fit.coef, q2, form), atol=1e-10)

        stale = runners.assign(**{Q: np.log([0.9, 0.05, 0.05])})
        np.testing.assert_allclose(fit.predict(stale, q1), p1, atol=1e-12)

        frame, q = new_races(8722, 40)
        base = np.asarray(fit.predict(frame, q), dtype=float)
        scale = np.random.default_rng(1).uniform(0.5, 2.0, size=frame["race_id"].nunique())
        scaled = q * scale[codes_of(frame["race_id"])]
        np.testing.assert_allclose(fit.predict(frame, scaled), base, atol=1e-12)
        perm = np.random.default_rng(2).permutation(len(frame))
        shuffled = np.asarray(fit.predict(frame.iloc[perm], q[perm]), dtype=float)
        np.testing.assert_allclose(shuffled, base[perm], atol=1e-12)
        np.testing.assert_allclose(race_sums(base, frame["race_id"]), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------------------------
# DoD: alpha is fitted unbounded
# ---------------------------------------------------------------------------------------------


def test_alpha_is_unbounded_and_can_be_negative():
    """Truth p proportional to q * exp(s). A member ln p_m = ln q + c * s (as a log-probability)
    gives alpha = 1/c and beta = 1 - 1/c in u = alpha ln p_m + beta ln q. c = -0.5 is the ticket's
    member, the market minus half the true signal: alpha = -2 with its interval below 0, which a
    fit bounded at 0 cannot reach. c = 0.5 makes beta negative; c = 1 is the true model."""
    for seed, c in ((8731, -0.5), (8732, 0.5), (8733, 1.0)):
        rng = np.random.default_rng(seed)
        codes = field(rng, 2_000)
        q = market(rng, codes)
        s = rng.normal(0, 0.6, size=len(codes))
        won = draw_winners(rng, np.log(q) + s, codes)
        frame = make_frame(codes, q, won, p_member=log_softmax(np.log(q) + c * s, codes))

        fit = fit_meta(frame, spec(["p_member"], form="stage2"))

        assert set(fit.coef.index) == {"p_member", Q}
        assert_mle_with_honest_se(fit, frame, "stage2")
        alpha, beta = fit.coef["p_member"], fit.coef[Q]
        se_a, se_b = fit.se["p_member"], fit.se[Q]
        assert abs(alpha - 1 / c) < 3 * se_a, (c, alpha, se_a)
        assert abs(beta - (1 - 1 / c)) < 3 * se_b, (c, beta, se_b)
        if c < 0:
            assert alpha < 0 and alpha + 3 * se_a < 0
        if c == 0.5:
            assert beta < 0 and beta + 3 * se_b < 0


# ---------------------------------------------------------------------------------------------
# DoD: the offset model equals the market when delta = 0
# ---------------------------------------------------------------------------------------------


def test_offset_model_equals_market_when_delta_is_zero(planted):
    """An offset fit with every delta coefficient set to 0 predicts q itself: (0.5, 0.3, 0.2) for
    q = (0.5, 0.3, 0.2), whatever the members and segments say, for an over-round q too, and for
    many races at once. The fitted delta is not 0 on this data, so the test can fail."""
    fit = fit_meta(planted, spec(MEMBERS, form="offset", segment_cols=["z_class"]))
    assert Q not in fit.coef.index
    zero = dataclasses.replace(fit, coef=pd.Series(0.0, index=fit.coef.index))

    runners = three_runners()
    q = np.array([0.5, 0.3, 0.2])
    np.testing.assert_allclose(zero.predict(runners, q), [0.5, 0.3, 0.2], rtol=0, atol=1e-12)
    np.testing.assert_allclose(zero.predict(runners, 1.25 * q), q, rtol=0, atol=1e-12)
    assert np.abs(np.asarray(fit.predict(runners, q)) - q).max() > 1e-3

    frame, q_many = new_races(8734, 50)
    np.testing.assert_allclose(zero.predict(frame, q_many), q_many, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------------------------
# Outline 3: segment weights as interaction columns
# ---------------------------------------------------------------------------------------------


def test_segment_weights_enter_as_interaction_columns():
    """alpha(z) = 0.2 + 0.3 z planted on a member, z_class in {0, 1, 2} per race, beside a segment
    w_flag in {0, 1} with no effect: each coefficient is recovered within 3 se, in both forms
    (stage 2 with beta = 0.9; offset with q at 1), and the data can tell 0.3 from 0."""
    rng = np.random.default_rng(8740)
    codes = field(rng, 3_000)
    q = market(rng, codes)
    member = log_softmax(rng.normal(size=len(codes)), codes)
    n_races = codes.max() + 1
    z = rng.integers(0, 3, size=n_races)[codes].astype(float)
    w = rng.integers(0, 2, size=n_races)[codes].astype(float)
    alpha = 0.2 + 0.3 * z
    truths = {
        "stage2": (0.9 * np.log(q) + alpha * member, {Q: 0.9}),
        "offset": (np.log(q) + alpha * member, {}),
    }
    for form, (u, extra) in truths.items():
        won = draw_winners(rng, u, codes)
        frame = make_frame(codes, q, won, p_form=member, z_class=z, w_flag=w)

        fit = fit_meta(frame, spec(["p_form"], form=form, segment_cols=["z_class", "w_flag"]))

        want = {"p_form": 0.2, "p_form:z_class": 0.3, "p_form:w_flag": 0.0, **extra}
        assert set(fit.coef.index) == set(want)
        assert_mle_with_honest_se(fit, frame, form)
        for term, value in want.items():
            assert abs(fit.coef[term] - value) < 3 * fit.se[term], (form, term, fit.coef[term])
        assert 3 * fit.se["p_form:z_class"] < 0.3


# ---------------------------------------------------------------------------------------------
# The ridge shrinks toward the market (owner's target alpha = 0)
# ---------------------------------------------------------------------------------------------


def test_the_ridge_shrinks_member_weights_toward_the_market():
    """Truth u = ln q + 0.5 p_form. Ridge 0, 500, 1e9: |alpha| falls strictly without changing
    sign; under 1e9 the members no longer move the prediction, the offset model is q and the
    stage-2 model keeps its market weight (beta is not shrunk, so it is not the uniform field)."""
    rng = np.random.default_rng(8750)
    codes = field(rng, 1_500)
    q = market(rng, codes)
    member = log_softmax(rng.normal(size=len(codes)), codes)
    won = draw_winners(rng, np.log(q) + 0.5 * member, codes)
    frame = make_frame(codes, q, won, p_form=member)

    runners = three_runners()
    scrambled = runners.assign(p_form=np.log([0.7, 0.2, 0.1]))
    q3 = np.array([0.5, 0.3, 0.2])
    for form in ("stage2", "offset"):
        fits = {r: fit_meta(frame, spec(["p_form"], form=form, ridge=r)) for r in (0.0, 500.0, 1e9)}
        a = [fits[r].coef["p_form"] for r in (0.0, 500.0, 1e9)]
        assert abs(a[0]) > abs(a[1]) > abs(a[2]), (form, a)
        assert a[0] > 0 and a[1] > 0
        assert abs(a[1]) < 0.9 * abs(a[0])
        assert abs(a[2]) < 1e-5

        big = fits[1e9]
        p = np.asarray(big.predict(runners, q3), dtype=float)
        np.testing.assert_allclose(big.predict(scrambled, q3), p, atol=1e-5)
        assert (
            np.abs(
                np.asarray(fits[0.0].predict(scrambled, q3)) - fits[0.0].predict(runners, q3)
            ).max()
            > 1e-2
        )
        if form == "offset":
            np.testing.assert_allclose(p, q3, atol=1e-5)
        else:
            assert abs(big.coef[Q] - 1.0) < 0.25, big.coef[Q]
            assert p[0] - p[2] > 0.1


# ---------------------------------------------------------------------------------------------
# quadratic=True extends the linear model; a bad form is refused
# ---------------------------------------------------------------------------------------------


def test_quadratic_terms_extend_the_linear_model(planted):
    q = np.exp(planted[Q].to_numpy())
    for form in ("stage2", "offset"):
        linear = fit_meta(planted, spec(MEMBERS, form=form, quadratic=False))
        quad = fit_meta(planted, spec(MEMBERS, form=form, quadratic=True))
        assert set(linear.coef.index) < set(quad.coef.index)
        p = np.asarray(quad.predict(planted, q), dtype=float)
        assert np.all(np.isfinite(p)) and np.all(p > 0)
        np.testing.assert_allclose(race_sums(p, planted["race_id"]), 1.0, atol=1e-12)
        lin_ll = winner_loglik(linear.predict(planted, q), planted)
        assert winner_loglik(p, planted) >= lin_ll - 1e-6


@pytest.mark.parametrize("form", ["stage3", "", "Offset", "stage_2"])
def test_an_unknown_form_is_refused(planted, form):
    with pytest.raises(ValueError):
        fit_meta(planted, spec(["p_form"], form=form))


def test_a_member_missing_from_the_frame_is_refused(planted):
    with pytest.raises((KeyError, ValueError)):
        fit_meta(planted, spec(["p_form", "p_jockey"]))


# ---------------------------------------------------------------------------------------------
# DoD: level-1 predictions are out of fold
# ---------------------------------------------------------------------------------------------

MONTHLY = FoldScheme(
    unit="month",
    test_length=1,
    step=1,
    min_train=2,
    window="expanding",
    embargo_days=7,
    anchor_month=1,
)


def dated_frame(seed, n_races, races_per_day=2, first_day=date(2024, 1, 6)):
    """Races with dates, rows shuffled so races and dates interleave, non-default index labels."""
    rng = np.random.default_rng(seed)
    codes = field(rng, n_races, sizes=(5, 10))
    q = market(rng, codes)
    s = rng.normal(0, 0.8, size=len(codes))
    won = draw_winners(rng, np.log(q) + s, codes)
    member = log_softmax(np.log(q) + 0.5 * s, codes)
    frame = make_frame(codes, q, won, p_form=member)
    days = np.array([first_day + timedelta(days=int(c // races_per_day)) for c in codes])
    frame["race_date"] = days
    frame["row_key"] = np.arange(len(frame)) * 7 + 3
    perm = rng.permutation(len(frame))
    frame = frame.iloc[perm]
    frame.index = pd.Index(10_000 + rng.permutation(len(frame)), name="row")
    return frame


def test_out_of_fold_levels_never_see_their_own_fold():
    """A recording fit and predict: fold k's model is fitted on exactly its training rows, it
    predicts exactly its test rows, and every row's value is that model's (the value encodes the
    model and the row). No model predicts a row it saw, so fitting once on all rows, or on all
    rows but the fold, cannot pass. Rows in no test window are left empty."""
    frame = dated_frame(8770, 400)
    folds = walk_forward(frame["race_date"], scheme=MONTHLY)
    assert len(folds) >= 4
    tested = np.zeros(len(frame), dtype=bool)
    for fold in folds:
        tested[fold.test_idx] = True
    assert tested.any() and not tested.all()

    models = []
    predictions = []

    def fit(train):
        model = {"id": len(models), "seen": frozenset(train["row_key"].tolist())}
        models.append(model)
        return model

    def predict(model, test):
        keys = test["row_key"].to_numpy()
        predictions.append((model["id"], frozenset(keys.tolist())))
        return model["id"] * 1e6 + keys.astype(float)

    out = oof_predictions(frame, folds, fit, predict)

    assert isinstance(out, pd.Series)
    assert out.index.is_unique and out.index.isin(frame.index).all()
    got = out.reindex(frame.index)
    assert len(models) == len(folds)
    by_seen = {m["seen"]: m["id"] for m in models}
    assert len(by_seen) == len(models)
    for fold in folds:
        train_keys = frozenset(frame["row_key"].iloc[fold.train_idx].tolist())
        assert train_keys in by_seen, "a model was not fitted on exactly a fold's training rows"
        model_id = by_seen[train_keys]
        test = frame.iloc[fold.test_idx]
        want = model_id * 1e6 + test["row_key"].to_numpy(dtype=float)
        np.testing.assert_array_equal(got.loc[test.index].to_numpy(dtype=float), want)
    for model_id, keys in predictions:
        assert not (models[model_id]["seen"] & keys), "a model predicted a row it was fitted on"
    assert got[~tested].isna().all()


def _fold(train, test):
    return Fold(
        train_idx=np.asarray(train),
        test_idx=np.asarray(test),
        train_end=date(2024, 3, 1),
        test_start=date(2024, 3, 2),
        test_end=date(2024, 3, 31),
    )


LEAKY = {
    "a fold trains on a row it tests": [_fold(range(0, 40), range(35, 50))],
    "two folds test one row": [
        _fold(range(0, 20), range(20, 40)),
        _fold(range(0, 30), range(39, 50)),
    ],
}


@pytest.mark.parametrize("folds", list(LEAKY.values()), ids=list(LEAKY))
def test_out_of_fold_refuses_folds_that_leak_or_overlap(folds):
    frame = dated_frame(8771, 12).iloc[:60]
    assert len(frame) == 60
    with pytest.raises(ValueError):
        oof_predictions(frame, folds, lambda train: 0.0, lambda model, test: np.zeros(len(test)))


def test_out_of_fold_meta_predictions_ignore_their_own_and_later_folds_outcomes():
    """fit_meta through oof_predictions on walk-forward folds: flipping the winners of fold j (to
    each race's longest shot) leaves folds 0..j exactly as they were, and moves fold j + 1, which
    trains on fold j."""
    frame = dated_frame(8772, 1_200, races_per_day=4)
    folds = walk_forward(frame["race_date"], scheme=MONTHLY)
    assert len(folds) >= 5
    j = 2

    def fit(train):
        return fit_meta(train, spec(["p_form"], form="stage2"))

    def predict(model, test):
        return model.predict(test, np.exp(test[Q].to_numpy()))

    base = oof_predictions(frame, folds, fit, predict).reindex(frame.index)

    flipped = frame.copy()
    rows = flipped.index[folds[j].test_idx]
    sub = flipped.loc[rows]
    longest = sub.groupby("race_id")[Q].transform("min")
    flipped.loc[rows, "won"] = (sub[Q] == longest).to_numpy()
    assert (flipped["won"] != frame["won"]).sum() > 50
    again = oof_predictions(flipped, folds, fit, predict).reindex(frame.index)

    for k in range(j + 1):
        idx = frame.index[folds[k].test_idx]
        np.testing.assert_allclose(again.loc[idx], base.loc[idx], rtol=0, atol=1e-12)
        np.testing.assert_allclose(race_sums(base.loc[idx], frame.loc[idx, "race_id"]), 1.0)
    later = frame.index[folds[j + 1].test_idx]
    assert np.abs(again.loc[later].to_numpy() - base.loc[later].to_numpy()).max() > 1e-6


# ---------------------------------------------------------------------------------------------
# Encompassing: the member's weight beside the final price
# ---------------------------------------------------------------------------------------------


def test_encompassing_is_the_member_weight_beside_the_market():
    """Truth u = 0.7 ln q + 0.6 a. For each member, the coefficient and se are those of the
    unpenalised conditional logit on [member, ln q] alone (a reference Newton fit), whatever other
    members the frame holds: a is near 0.6 with its interval above 0, pure noise b is within 3 se
    of 0, and c (anti-correlated with a) is negative."""
    rng = np.random.default_rng(8760)
    codes = field(rng, 2_500)
    q = market(rng, codes)
    s = rng.normal(size=len(codes))
    a = log_softmax(s, codes)
    b = log_softmax(rng.normal(size=len(codes)), codes)
    c = log_softmax(-s + 0.5 * rng.normal(size=len(codes)), codes)
    won = draw_winners(rng, 0.7 * np.log(q) + 0.6 * a, codes)
    frame = make_frame(codes, q, won, p_a=a, p_b=b, p_c=c, z_class=np.ones(len(codes)))

    results = {}
    for member in ("p_a", "p_b", "p_c"):
        coef, se = encompassing(frame, member)
        X = np.column_stack([frame[member].to_numpy(), frame[Q].to_numpy()])
        ref, ref_se = reference_mle(X, frame["race_id"], frame["won"])
        assert abs(coef - ref[0]) < 0.01 * ref_se[0], (member, coef, ref[0])
        assert se == pytest.approx(ref_se[0], rel=1e-3)
        results[member] = (coef, se)

    coef, se = results["p_a"]
    assert abs(coef - 0.6) < 3 * se and coef - 3 * se > 0
    coef, se = results["p_b"]
    assert abs(coef) < 3 * se
    coef, se = results["p_c"]
    assert coef + 3 * se < 0


# ---------------------------------------------------------------------------------------------
# The information-ceiling table (E9)
# ---------------------------------------------------------------------------------------------


def test_ceiling_table_counts_q_mass_above_threshold():
    """q = (0.5, 0.3, 0.2), pi = (0.4, 0.3, 0.3): ln(pi/q) = (ln 0.8, 0, ln 1.5). Above ln 1.25
    only the third runner, q-mass 0.2; the q-weighted RMS is sqrt(0.5 ln(0.8)^2 + 0.2 ln(1.5)^2)
    = 0.2404. Above 0.1 also 0.2, above 0.5 nothing."""
    frame = pd.DataFrame(
        {
            "race_id": ["jra:2026-11-28:TOK:11"] * 3,
            "horse_id": [1, 2, 3],
            "p_model": [0.4, 0.3, 0.3],
            "q_win": [0.5, 0.3, 0.2],
        }
    )
    rms = math.sqrt(0.5 * math.log(0.8) ** 2 + 0.2 * math.log(1.5) ** 2)
    assert round(rms, 4) == 0.2404

    table = ceiling_table(frame, pi_col="p_model", q_col="q_win", thresholds=[math.log(1.25)])
    assert len(table) == 1
    assert table["q_mass"].iloc[0] == pytest.approx(0.2, abs=1e-12)
    assert round(float(table["rms"].iloc[0]), 4) == 0.2404
    assert table["rms"].iloc[0] == pytest.approx(rms, abs=1e-12)

    thresholds = [0.1, math.log(1.25), 0.5]
    table = ceiling_table(frame, pi_col="p_model", q_col="q_win", thresholds=thresholds)
    assert list(table["threshold"]) == pytest.approx(thresholds)
    assert list(table["q_mass"]) == pytest.approx([0.2, 0.2, 0.0], abs=1e-12)
    assert list(table["rms"]) == pytest.approx([rms] * 3, abs=1e-12)


def test_ceiling_table_pools_races_by_their_q_mass():
    """Two hand-worked races, then 200 random ones against the formula: q_mass is the share of
    the total q mass with ln(pi/q) above each threshold, rms the q-weighted RMS over every runner;
    the row order of the frame does not matter and the rows follow the thresholds' order."""
    frame = pd.DataFrame(
        {
            "race_id": ["a", "a", "a", "b", "b"],
            "horse_id": [1, 2, 3, 1, 2],
            "pi": [0.4, 0.3, 0.3, 0.3, 0.7],
            "q": [0.5, 0.3, 0.2, 0.6, 0.4],
        }
    )
    # race a: ln(pi/q) = (ln 0.8, 0, ln 1.5); race b: (ln 0.5, ln 1.75)
    thresholds = [math.log(2.0), math.log(1.25), 0.5]
    table = ceiling_table(frame, pi_col="pi", q_col="q", thresholds=thresholds)
    assert list(table["threshold"]) == pytest.approx(thresholds)
    assert list(table["q_mass"]) == pytest.approx([0.0, 0.3, 0.2], abs=1e-12)
    sq = (
        0.5 * math.log(0.8) ** 2
        + 0.2 * math.log(1.5) ** 2
        + 0.6 * math.log(0.5) ** 2
        + 0.4 * math.log(1.75) ** 2
    )
    assert list(table["rms"]) == pytest.approx([math.sqrt(sq / 2)] * 3, abs=1e-12)

    rng = np.random.default_rng(8780)
    codes = field(rng, 200, sizes=(4, 16))
    q = market(rng, codes)
    pi = softmax_in_race(np.log(q) + rng.normal(0, 0.4, size=len(codes)), codes)
    frame = make_frame(codes, q, p_model=pi, q_final=q)
    thresholds = [0.0, 0.1, math.log(1.25), 0.4, 1.0]
    x = np.log(pi / q)
    want_mass = [float(q[x > t].sum() / q.sum()) for t in thresholds]
    want_rms = math.sqrt(float((q * x**2).sum() / q.sum()))
    assert len(set(np.round(want_mass, 6))) == len(thresholds)

    perm = rng.permutation(len(frame))
    for rows in (frame, frame.iloc[perm]):
        table = ceiling_table(rows, pi_col="p_model", q_col="q_final", thresholds=thresholds)
        assert len(table) == len(thresholds)
        assert list(table["threshold"]) == pytest.approx(thresholds)
        np.testing.assert_allclose(table["q_mass"].to_numpy(dtype=float), want_mass, atol=1e-12)
        np.testing.assert_allclose(table["rms"].to_numpy(dtype=float), want_rms, atol=1e-12)


# ---------------------------------------------------------------------------------------------
# Meeting-clustered standard errors (review finding on #87; D7: the inference unit is the meeting)
# ---------------------------------------------------------------------------------------------


def with_meetings(frame, races_per_meeting=12, column="meeting"):
    """The frame with a venue x day meeting label per row, `races_per_meeting` races to each, in
    the order the races first appear."""
    meeting = codes_of(frame["race_id"]) // races_per_meeting
    names = np.array([f"2025-{m // 3:04d}:{('TOK', 'NAK', 'KYO')[m % 3]}" for m in range(400)])
    return frame.assign(**{column: names[meeting].astype(object)})


def meeting_sandwich(X, race_id, won, coef, cluster, offset=None, penalty=None):
    """sqrt(diag(H^-1 B H^-1)): H the information at coef plus diag(penalty), B the sum over
    clusters of g_c g_c^T, g_c the summed per-race score sum_i (w_i - p_i) x_i of the cluster's
    races (w = 1/k on k co-winners; void races dropped). No small-sample factor."""
    won = np.asarray(won, dtype=bool)
    codes = codes_of(race_id)
    k = np.bincount(codes, weights=won)
    keep = k[codes] > 0
    Xk = X[keep]
    race = pd.factorize(codes[keep])[0]
    u = Xk @ np.asarray(coef, dtype=float)
    if offset is not None:
        u = u + np.asarray(offset, dtype=float)[keep]
    p = softmax_in_race(u, race)
    w = won[keep] / k[codes][keep]
    per_row = (w - p)[:, None] * Xk
    unit = pd.factorize(np.asarray(cluster, dtype=object)[keep])[0]
    g = np.stack([np.bincount(unit, weights=per_row[:, j]) for j in range(X.shape[1])], axis=1)
    _, info = score_and_information(X, race_id, won, coef, offset)
    penalty = np.zeros(X.shape[1]) if penalty is None else np.asarray(penalty, dtype=float)
    bread = np.linalg.inv(info + np.diag(penalty))
    return np.sqrt(np.diag(bread @ (g.T @ g) @ bread))


CLUSTERED_SPECS = {
    "stage2": ("stage2", [], 0.0),
    "stage2 with a segment": ("stage2", ["z_class"], 0.0),
    "offset with a segment": ("offset", ["z_class"], 0.0),
    "stage2 under a ridge": ("stage2", ["z_class"], 50.0),
}


@pytest.mark.parametrize(
    "form, segment_cols, ridge", list(CLUSTERED_SPECS.values()), ids=list(CLUSTERED_SPECS)
)
def test_fit_meta_reports_the_meeting_clustered_se_of_its_design(
    planted, form, segment_cols, ridge
):
    """The planted 1,500 races at meetings of 12. With cluster_col, se_cluster is a Series on the
    index of coef, the sandwich of the fitted design (members, interactions and ln q in stage 2;
    ln q as the offset in the offset form; under a ridge the bread adds the penalty on the member
    terms only). coef and se are those of the fit without cluster_col, which has no se_cluster and
    is the same fit as cluster_col=None."""
    frame = with_meetings(planted)
    s = spec(MEMBERS, form=form, segment_cols=segment_cols, ridge=ridge)

    fit = fit_meta(frame, s, cluster_col="meeting")
    plain = fit_meta(frame, s)
    explicit = fit_meta(frame, s, cluster_col=None)

    assert isinstance(fit.se_cluster, pd.Series)
    assert list(fit.se_cluster.index) == list(fit.coef.index)
    terms = list(fit.coef.index)
    q = np.exp(frame[Q].to_numpy())
    X = design(frame, terms, q)
    offset = np.log(q) if form == "offset" else None
    penalty = [0.0 if term == Q else ridge for term in terms]
    want = meeting_sandwich(
        X,
        frame["race_id"],
        frame["won"],
        fit.coef.to_numpy(dtype=float),
        frame["meeting"],
        offset,
        penalty,
    )
    np.testing.assert_allclose(fit.se_cluster.to_numpy(dtype=float), want, rtol=1e-6)

    assert plain.se_cluster is None and explicit.se_cluster is None
    pd.testing.assert_series_equal(explicit.coef, plain.coef)
    pd.testing.assert_series_equal(explicit.se, plain.se)
    assert list(fit.coef.index) == list(plain.coef.index)
    moved = np.abs(fit.coef.to_numpy() - plain.coef.reindex(terms).to_numpy())
    assert np.all(moved < 1e-6 * plain.se.reindex(terms).to_numpy()), moved
    np.testing.assert_allclose(fit.se.to_numpy(), plain.se.reindex(terms).to_numpy(), rtol=1e-6)
    assert fit.n_races == plain.n_races == 1_500


def test_encompassing_reports_the_meeting_clustered_se_with_a_cluster_col():
    """Truth u = 0.7 ln q + (0.6 + b_m) a, with b_m ~ N(0, 0.5^2) shared by the 12 races of each
    of 200 meetings. For each member, encompassing with cluster_col gives the coefficient it gives
    without, and the meeting sandwich of the reference fit on [member, ln q]; without cluster_col
    the second element stays the inverse-information se. For a, whose weight the shock moves, the
    clustered se is more than 1.3 times that se."""
    rng = np.random.default_rng(8761)
    codes = field(rng, 2_400)
    q = market(rng, codes)
    a = log_softmax(rng.normal(size=len(codes)), codes)
    b = log_softmax(rng.normal(size=len(codes)), codes)
    shock = rng.normal(0, 0.5, size=200)[codes // 12]
    won = draw_winners(rng, 0.7 * np.log(q) + (0.6 + shock) * a, codes)
    frame = with_meetings(make_frame(codes, q, won, p_a=a, p_b=b), column="venue_day")

    for member in ("p_a", "p_b"):
        coef_c, se_c = encompassing(frame, member, cluster_col="venue_day")
        coef, se = encompassing(frame, member)
        X = np.column_stack([frame[member].to_numpy(), frame[Q].to_numpy()])
        ref, ref_se = reference_mle(X, frame["race_id"], frame["won"])
        want = meeting_sandwich(X, frame["race_id"], frame["won"], ref, frame["venue_day"])

        assert isinstance(coef_c, float) and isinstance(se_c, float)
        assert coef_c == pytest.approx(coef, abs=1e-6 * se)
        assert abs(coef_c - ref[0]) < 0.01 * ref_se[0], (member, coef_c, ref[0])
        assert se == pytest.approx(ref_se[0], rel=1e-3)
        assert se_c == pytest.approx(want[0], rel=1e-4), (member, se_c, want[0])
        if member == "p_a":
            assert se_c > 1.3 * se, (se_c, se)


REFUSED_CLUSTER_COLS = {
    "a column the frame does not have": ("venue_day", (KeyError, ValueError)),
    "a column that splits a race": ("split", ValueError),
}


@pytest.mark.parametrize(
    "column, error", list(REFUSED_CLUSTER_COLS.values()), ids=list(REFUSED_CLUSTER_COLS)
)
def test_a_cluster_col_that_is_missing_or_splits_a_race_is_refused(planted, column, error):
    frame = with_meetings(planted)
    first = frame.index[frame["race_id"] == frame["race_id"].iloc[0]]
    split = frame["meeting"].copy()
    split.loc[first[-1]] = "2025-9999:KYO"
    frame = frame.assign(split=split)
    with pytest.raises(error):
        fit_meta(frame, spec(MEMBERS), cluster_col=column)
    with pytest.raises(error):
        encompassing(frame, "p_form", cluster_col=column)
