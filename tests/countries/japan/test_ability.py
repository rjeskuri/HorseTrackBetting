"""C2 ability v0: discounted top-k Plackett-Luce with pooled jockey, trainer and sire (ticket #92,
epic #73; 08 §3.2 C2, §4 E33, E34; 05 §2.2, §2.4, §2.5).

Contract pinned here (`horsetrack.countries.japan.models.ability`):
- `pl_loglik(u, race_id, finish_pos, *, k, discounts) -> float`: the sum over races, and over the
  stages s = 1..k that have a runner placed s, of discounts[s-1] * (u of the runner placed s less
  the logsumexp of u over the race's runners not placed 1..s-1). Runners placed beyond k stay in
  every stage's choice set but are never chosen, so their order among themselves does not matter
  (positions 4+ are not used at k = 3: eased runners). `race_id` is one label per row and rows
  need not be grouped by race. `discounts` has exactly k entries, else ValueError. At k = 1 with
  discounts (1,) it is the conditional logit's log-likelihood (`models.condlogit`).
- Dead heats (a shared finish_pos, JRA style: co-winners are both 1 and the next runner is 3) stay
  in the likelihood: the race adds a finite, strictly negative term, never nothing. The tie rule
  itself is the coder's Ruling (05 §2.4) and is not pinned.
- `AbilityModel(*, k, discounts, entity_cols, features)`; `.fit(frame)` returns the fitted
  `AbilityModel`. frame: race_id, horse_id, finish_pos (1 = winner), the `features` columns and the
  `entity_cols` columns (ids; `entity_cols` may be empty); other columns are ignored. u = the
  features' linear term plus one effect per entity column, the entity effects shrunk toward 0
  (penalised MAP or a Bayesian fit; the prior and its strength are not pinned). The fit reads only
  the rows it is given: they are its training window.
- `.predict(frame) -> pd.DataFrame`: one row per input row, with race_id, horse_id, u, se_u and
  p_ability. It reads race_id, horse_id, the features and the entity columns only (no finish_pos,
  no date). p_ability is softmax(u) within the race, so it sums to 1 per race. se_u is u's
  standard error from the fit (sigma_i), finite and positive per runner. An entity the fit never
  saw gets the prior: effect 0 and the prior's uncertainty.
- `fit_discounts(frame, folds, *, features, entity_cols) -> tuple[float, float, float]`: the top-3
  discounts (w1, w2, w3), finite and non-negative with w1 > 0, chosen by fitting `AbilityModel`s
  on each fold's training rows and scoring their predictions of its test rows' winners. `folds` is
  a sequence of `core.eval.folds.Fold` (positions into `frame`, as `walk_forward` returns them):
  the inner folds a caller builds over an outer fold's training rows. A row in no fold (the outer
  test window, the holdout) is never read: changing it, or leaving it out of the frame, leaves the
  discounts unchanged. The search, its grid and the exact out-of-fold loss are not pinned.

Not pinned: the tie rule, the optimiser, the prior's variance or how it is set, whether se_u is
the standard error of u or of u less its race mean (the SE test's features are centred within
race, where the two agree), predict's row order (rows are matched on race_id and horse_id),
unplaced runners (a missing finish_pos), and the E33 and E34 runs, report and registry records,
which read the local stores. Expected values come from first principles or from
`models.condlogit.fit_conditional_logit` on the exploded rankings (ability_synth.py), never from
the code under test. Every RNG is seeded.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.folds import FoldScheme, walk_forward
from horsetrack.countries.japan.models.ability import AbilityModel, fit_discounts, pl_loglik
from horsetrack.countries.japan.models.condlogit import fit_conditional_logit

# isort: split
from ability_synth import (
    BIG_EFFECT,
    FEATURES_10,
    J_BIG,
    J_FLIP,
    J_SPARSE,
    J_ZERO,
    centre_within_race,
    discount_frame,
    exploded_fit,
    exploded_information,
    field,
    first_principles_loglik,
    labels,
    pl_positions,
    pooled_frame,
    softmax_within,
    window_frame,
)

OUT_COLUMNS = {"race_id", "horse_id", "u", "se_u", "p_ability"}
ENTITIES = ("jockey_id", "trainer_id", "sire_id")
TICKET_U = np.array([0.0, math.log(2.0), math.log(3.0)])
UNIT3 = (1.0, 1.0, 1.0)


def ints(pos) -> np.ndarray:
    return np.asarray(pos).astype(np.int64)


def reorder_beyond(rng: np.random.Generator, pos, codes, k: int) -> np.ndarray:
    """pos with the places beyond k permuted among their runners, race by race."""
    out = np.asarray(pos).copy()
    for c in np.unique(codes):
        rows = np.flatnonzero((codes == c) & (out > k))
        out[rows] = out[rng.permutation(rows)]
    return out


def race_sums(values, race_id) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).groupby(np.asarray(race_id)).sum().to_numpy()


def predict_on(model, frame: pd.DataFrame) -> pd.DataFrame:
    """model.predict(frame) checked for its shape and keyed back onto frame's (race_id, horse_id)
    rows, in frame's order."""
    out = model.predict(frame)
    assert isinstance(out, pd.DataFrame)
    assert OUT_COLUMNS <= set(out.columns), sorted(out.columns)
    assert len(out) == len(frame)
    got = out.set_index(["race_id", "horse_id"])
    assert got.index.is_unique
    want = pd.MultiIndex.from_frame(frame[["race_id", "horse_id"]])
    assert set(got.index) == set(want)
    return got.reindex(want)


def probe(model, jockeys, *, features=("x1",), entity_cols=ENTITIES) -> pd.DataFrame:
    """One race of len(jockeys) runners identical but for the jockey: every feature 0, a trainer
    and a sire the fit never saw. Returns u and se_u indexed by jockey; differences in u are
    differences in the jockeys' effects."""
    n = len(jockeys)
    frame = pd.DataFrame(
        {
            "race_id": ["probe:r1"] * n,
            "horse_id": [f"probe:h{i}" for i in range(n)],
            "jockey_id": list(jockeys),
        }
    )
    for name in features:
        frame[name] = 0.0
    if "trainer_id" in entity_cols:
        frame["trainer_id"] = "t:new"
    if "sire_id" in entity_cols:
        frame["sire_id"] = "s:new"
    out = predict_on(model, frame)
    return pd.DataFrame(
        {"u": out["u"].to_numpy(dtype=float), "se_u": out["se_u"].to_numpy(dtype=float)},
        index=list(jockeys),
    )


# ---------------------------------------------------------------------------------------------
# DoD: pl_loglik at k = 1 equals the conditional logit; the discounted top-k likelihood
# ---------------------------------------------------------------------------------------------


def test_pl_reduces_to_conditional_logit_at_k1():
    """u = (0, ln 2, ln 3) with the third runner winning: ln(3/6) = -0.6931. On varied fields of 2
    to 14 runners the k = 1 value at the conditional logit's fitted coefficients is that fit's
    log-likelihood, reordering the losers never moves it, and at any u it is the conditional
    logit by first principles."""
    one = np.array(["jra:r1"] * 3, dtype=object)
    got = pl_loglik(TICKET_U, one, np.array([3, 2, 1]), k=1, discounts=(1.0,))
    assert got == pytest.approx(math.log(0.5), abs=1e-12)
    assert round(got, 4) == -0.6931

    designs = ((9201, (0.8, -0.5)), (9202, (1.5,)), (9203, (-1.0, 0.3, 0.6)))
    for seed, planted in designs:
        rng = np.random.default_rng(seed)
        codes = field(rng, 300, sizes=(2, 14))
        X = rng.normal(size=(len(codes), len(planted)))
        pos = ints(pl_positions(rng, X @ np.asarray(planted), codes))
        race_id = labels(codes)
        fit = fit_conditional_logit(X, race_id, pos == 1)
        u = X @ np.asarray(fit.coef)
        value = pl_loglik(u, race_id, pos, k=1, discounts=(1.0,))
        assert value == pytest.approx(fit.loglik, abs=1e-8), seed
        losers_moved = reorder_beyond(rng, pos, codes, 1)
        assert (losers_moved != pos).sum() > 200
        assert pl_loglik(u, race_id, losers_moved, k=1, discounts=(1.0,)) == pytest.approx(
            fit.loglik, abs=1e-8
        )
        other = rng.normal(0.0, 1.5, size=len(codes))
        want = first_principles_loglik(other, race_id, pos, 1, (1.0,))
        assert pl_loglik(other, race_id, pos, k=1, discounts=(1.0,)) == pytest.approx(
            want, abs=1e-8
        )


def test_discounted_top3_loglik():
    """The same u, finish order third, second, first runner, discounts (1, 0.5, 0.25):
    ln(3/6) + 0.5 * ln(2/3) + 0.25 * ln(1/1) = -0.8959. At unit discounts the top-3 value at the
    exploded conditional logit's fit is that fit's log-likelihood."""
    one = np.array(["jra:r1"] * 3, dtype=object)
    got = pl_loglik(TICKET_U, one, np.array([3, 2, 1]), k=3, discounts=(1.0, 0.5, 0.25))
    assert got == pytest.approx(math.log(0.5) + 0.5 * math.log(2.0 / 3.0), abs=1e-12)
    assert round(got, 4) == -0.8959
    # the second stage's weight is what it says: (1, 1, 0.25) adds another 0.5 * ln(2/3)
    full = pl_loglik(TICKET_U, one, np.array([3, 2, 1]), k=3, discounts=(1.0, 1.0, 0.25))
    assert full == pytest.approx(math.log(0.5) + math.log(2.0 / 3.0), abs=1e-12)

    rng = np.random.default_rng(9204)
    codes = field(rng, 300, sizes=(4, 12))
    X = rng.normal(size=(len(codes), 2))
    pos = ints(pl_positions(rng, X @ np.array([0.8, -0.5]), codes))
    fit = exploded_fit(X, codes, pos, 3)
    value = pl_loglik(X @ np.asarray(fit.coef), labels(codes), pos, k=3, discounts=UNIT3)
    assert value == pytest.approx(fit.loglik, abs=1e-8)


STAGES = {
    "k=1 at weight 0.7": (1, (0.7,)),
    "k=2": (2, (1.0, 0.3)),
    "k=3, the ticket's discounts": (3, (1.0, 0.5, 0.25)),
    "k=3, a winner weight below 1": (3, (0.8, 0.6, 0.1)),
    "k=4": (4, (1.0, 0.5, 0.25, 0.125)),
}


@pytest.mark.parametrize("k, discounts", list(STAGES.values()), ids=list(STAGES))
def test_pl_loglik_matches_first_principles_over_varied_races(k, discounts):
    """80 races of 2 to 12 runners (the small ones have fewer than k stages that choose), rows
    shuffled so races interleave: the value is the sum over races and stages of the stage's
    discount times ln softmax over the runners still in."""
    rng = np.random.default_rng(9205 + k)
    codes = field(rng, 80, sizes=(2, 12))
    u = rng.normal(0.0, 1.2, size=len(codes))
    pos = ints(pl_positions(rng, u, codes))
    race_id = labels(codes)
    want = first_principles_loglik(u, race_id, pos, k, discounts)
    perm = rng.permutation(len(codes))
    got = pl_loglik(u[perm], race_id[perm], pos[perm], k=k, discounts=discounts)
    assert got == pytest.approx(want, rel=1e-10, abs=1e-10)


def test_places_beyond_k_are_not_used_but_stay_in_the_choice_sets():
    """Reordering the runners placed 4th and lower leaves the k = 3 value unchanged, while the
    same reordering moves the k = 4 value; raising the u of those runners lowers the k = 3 value,
    because they still stand in every stage's denominator."""
    rng = np.random.default_rng(9210)
    codes = field(rng, 60, sizes=(7, 14))
    u = rng.normal(size=len(codes))
    pos = ints(pl_positions(rng, u, codes))
    race_id = labels(codes)
    d3 = (1.0, 0.5, 0.25)
    d4 = d3 + (0.125,)
    base = pl_loglik(u, race_id, pos, k=3, discounts=d3)
    moved = reorder_beyond(rng, pos, codes, 3)
    assert (moved != pos).sum() > 150
    assert pl_loglik(u, race_id, moved, k=3, discounts=d3) == pytest.approx(base, abs=1e-10)
    k4 = pl_loglik(u, race_id, pos, k=4, discounts=d4)
    assert abs(pl_loglik(u, race_id, moved, k=4, discounts=d4) - k4) > 1e-3
    lifted = np.where(pos > 3, u + 1.0, u)
    assert pl_loglik(lifted, race_id, pos, k=3, discounts=d3) < base - 1.0
    assert pl_loglik(lifted, race_id, pos, k=3, discounts=d3) == pytest.approx(
        first_principles_loglik(lifted, race_id, pos, 3, d3), abs=1e-9
    )


def test_discounts_must_have_k_entries():
    one = np.array(["jra:r1"] * 3, dtype=object)
    for k, discounts in ((3, (1.0, 0.5)), (1, (1.0, 0.5)), (2, ())):
        with pytest.raises(ValueError):
            pl_loglik(TICKET_U, one, np.array([3, 2, 1]), k=k, discounts=discounts)


# ---------------------------------------------------------------------------------------------
# Outline 2: dead heats are handled in the likelihood, never by dropping races
# ---------------------------------------------------------------------------------------------

DEAD_HEATS = {
    "for first": (1, 1, 3, 4, 5, 6, 7),
    "for second": (1, 2, 2, 4, 5, 6),
    "for third": (1, 2, 3, 3, 5, 6, 7, 8),
}


@pytest.mark.parametrize("k, discounts", [(1, (1.0,)), (3, (1.0, 0.5, 0.25))], ids=["k1", "k3"])
def test_a_dead_heat_keeps_the_race_in_the_likelihood(k, discounts):
    """Thirty ordinary races plus a dead heat for first, one for second and one for third (tied
    runners share the place and the next one skips it): each tied race adds a finite, strictly
    negative term to the sum, so none is dropped. The tie rule is the coder's Ruling."""
    rng = np.random.default_rng(9220 + k)
    codes = field(rng, 30, sizes=(6, 10))
    u = rng.normal(size=len(codes))
    pos = ints(pl_positions(rng, u, codes))
    race_id = labels(codes)
    base = pl_loglik(u, race_id, pos, k=k, discounts=discounts)
    assert math.isfinite(base)
    for name, places in DEAD_HEATS.items():
        tied_u = rng.normal(size=len(places))
        tied_race = np.array([f"jra:dead-heat-{name}"] * len(places), dtype=object)
        tied_pos = np.array(places, dtype=np.int64)
        alone = pl_loglik(tied_u, tied_race, tied_pos, k=k, discounts=discounts)
        assert math.isfinite(alone) and alone < -1e-6, (name, alone)
        both = pl_loglik(
            np.r_[u, tied_u],
            np.r_[race_id, tied_race],
            np.r_[pos, tied_pos],
            k=k,
            discounts=discounts,
        )
        assert both == pytest.approx(base + alone, abs=1e-9), name


def test_the_fit_keeps_dead_heat_races():
    """240 races, every fourth a dead heat for first (the Plackett-Luce runner-up tied with the
    winner): at k = 1 and at k = 3 the model fitted with them predicts differently from the model
    fitted without them, so the fit does not drop them."""
    rng = np.random.default_rng(9225)
    codes = field(rng, 240, sizes=(6, 10))
    X = centre_within_race(rng.normal(size=(len(codes), 2)), codes)
    pos = ints(pl_positions(rng, X @ np.array([0.8, -0.5]), codes))
    heat = codes % 4 == 0
    pos = np.where(heat & (pos == 2), 1, pos)
    assert ((pos == 1) & heat).sum() == 2 * len(np.unique(codes[heat]))
    frame = pd.DataFrame(
        {
            "race_id": labels(codes),
            "horse_id": [f"h:{i:05d}" for i in range(len(codes))],
            "finish_pos": pos,
            "x1": X[:, 0],
            "x2": X[:, 1],
        }
    )
    t_codes = field(rng, 20, sizes=(6, 10))
    Xt = rng.normal(size=(len(t_codes), 2))
    test = pd.DataFrame(
        {
            "race_id": labels(t_codes, prefix="jra:t"),
            "horse_id": [f"t:{i:05d}" for i in range(len(t_codes))],
            "x1": Xt[:, 0],
            "x2": Xt[:, 1],
        }
    )
    for k, discounts in ((1, (1.0,)), (3, (1.0, 0.5, 0.25))):

        def fitted(rows, k=k, discounts=discounts):
            spec = dict(k=k, discounts=discounts, entity_cols=(), features=["x1", "x2"])
            return AbilityModel(**spec).fit(rows)

        with_heats = predict_on(fitted(frame), test)["p_ability"].to_numpy(dtype=float)
        without = predict_on(fitted(frame[~heat]), test)["p_ability"].to_numpy(dtype=float)
        assert np.isfinite(with_heats).all()
        assert np.abs(with_heats - without).max() > 1e-4, k


# ---------------------------------------------------------------------------------------------
# Outline 1: jockey, trainer and sire effects shrunk toward 0 (partial pooling)
# ---------------------------------------------------------------------------------------------

POOLED_SEED = 9226


@pytest.fixture(scope="module")
def pooled():
    """The pooled-entity design (ability_synth.pooled_frame) and a top-3 model fitted on it."""
    frame, planted = pooled_frame(POOLED_SEED)
    spec = dict(k=3, discounts=UNIT3, entity_cols=ENTITIES, features=["x1"])
    model = AbilityModel(**spec).fit(frame)
    return frame, planted, model


def test_sparse_entity_is_shrunk_toward_zero(pooled):
    """A jockey with 3 rides (placed 1st, 1st and 6th) has an effect between 0 and its
    unpenalised estimate (the maximum likelihood given the planted rest, about 2); the 500-ride
    jockey's effect is within 0.1 of its planted 0.6, so pooling shrinks the sparse one far more."""
    frame, planted, model = pooled
    assert isinstance(model, AbilityModel)
    codes, u = planted["codes"], planted["u"]
    pos = frame["finish_pos"].to_numpy()
    jockey = frame["jockey_id"].to_numpy()
    assert (jockey == J_SPARSE).sum() == 3 and (jockey == J_BIG).sum() == 500
    # J_SPARSE is planted at 0, so the planted u is the offset of everything else
    unpenalised = float(exploded_fit((jockey == J_SPARSE), codes, pos, 3, offset=u).coef[0])
    assert unpenalised > 1.0

    got = probe(model, [J_BIG, J_SPARSE, "j:new"])
    effect = got["u"] - got.loc["j:new", "u"]
    assert 0.0 < effect[J_SPARSE] < unpenalised, (effect[J_SPARSE], unpenalised)
    assert abs(effect[J_BIG] - BIG_EFFECT) < 0.1, effect[J_BIG]
    assert effect[J_SPARSE] / unpenalised < effect[J_BIG] / BIG_EFFECT


def test_an_entity_unseen_in_training_gets_the_prior(pooled):
    """Two jockeys the fit never saw get the same u and se_u (the prior); the prior's 0 is the
    level of an average jockey, so a 500-ride jockey planted at 0 sits within 0.1 of them, with a
    smaller se_u."""
    _, _, model = pooled
    got = probe(model, ["j:new", "j:new-2", J_ZERO])
    assert np.isfinite(got.to_numpy()).all()
    assert got.loc["j:new", "u"] == pytest.approx(got.loc["j:new-2", "u"], abs=1e-9)
    assert got.loc["j:new", "se_u"] == pytest.approx(got.loc["j:new-2", "se_u"], rel=1e-6)
    assert abs(got.loc[J_ZERO, "u"] - got.loc["j:new", "u"]) < 0.1
    assert got.loc[J_ZERO, "se_u"] < got.loc["j:new", "se_u"]


# ---------------------------------------------------------------------------------------------
# Outline 3: u, its SE sigma_i and p_ability per runner
# ---------------------------------------------------------------------------------------------


def test_sigma_is_exported_per_runner(pooled):
    """predict on new races of the same entities, results withheld: one row per runner with u,
    se_u and p_ability; se_u finite, positive and different across runners; p_ability the softmax
    of u within the race. sigma follows each runner's own evidence: the 500-ride jockey's runners
    have smaller se_u than the 95-ride jockeys' runners, and on otherwise identical runners se_u
    rises from the 500-ride jockey to the 3-ride one to an unseen one."""
    _, _, model = pooled
    other, _ = pooled_frame(POOLED_SEED + 1)
    races = other["race_id"].unique()[:80]
    rows = other[other["race_id"].isin(races)].drop(columns=["finish_pos"])
    out = predict_on(model, rows)

    se = out["se_u"].to_numpy(dtype=float)
    assert np.isfinite(se).all() and (se > 0).all()
    assert se.std() > 1e-3
    u = out["u"].to_numpy(dtype=float)
    p = out["p_ability"].to_numpy(dtype=float)
    race_id = rows["race_id"].to_numpy()
    codes = pd.factorize(race_id)[0]
    assert np.isfinite(u).all() and ((p > 0) & (p < 1)).all()
    np.testing.assert_allclose(race_sums(p, race_id), 1.0, atol=1e-9)
    np.testing.assert_allclose(p, softmax_within(u, codes), atol=1e-9)

    jockey = rows["jockey_id"].to_numpy()
    big = jockey == J_BIG
    many = ~np.isin(jockey, [J_BIG, J_ZERO, J_SPARSE])
    assert big.sum() > 50 and many.sum() > 200
    assert np.median(se[big]) < np.median(se[many])

    got = probe(model, [J_BIG, J_SPARSE, "j:new"])
    assert got.loc[J_BIG, "se_u"] < got.loc[J_SPARSE, "se_u"] < got.loc["j:new", "se_u"]


@pytest.fixture(scope="module")
def features_only():
    """800 races of 6 to 12 runners, two features centred within race at (0.8, -0.5), full
    Plackett-Luce orders; winner-only (k = 1) and top-3 (k = 3, unit discounts) models without
    entities, their predictions for 60 new races, and the exploded conditional logit's (u, p, se)
    for the same races: se from the inverse information at the fit."""
    rng = np.random.default_rng(9230)
    codes = field(rng, 800, sizes=(6, 12))
    X = centre_within_race(rng.normal(size=(len(codes), 2)), codes)
    pos = ints(pl_positions(rng, X @ np.array([0.8, -0.5]), codes))
    train = pd.DataFrame(
        {
            "race_id": labels(codes),
            "horse_id": [f"h:{i:05d}" for i in range(len(codes))],
            "finish_pos": pos,
            "x1": X[:, 0],
            "x2": X[:, 1],
        }
    )
    t_codes = field(rng, 60, sizes=(6, 12))
    Xt = centre_within_race(rng.normal(size=(len(t_codes), 2)), t_codes)
    test = pd.DataFrame(
        {
            "race_id": labels(t_codes, prefix="jra:t"),
            "horse_id": [f"t:{i:05d}" for i in range(len(t_codes))],
            "x1": Xt[:, 0],
            "x2": Xt[:, 1],
        }
    )
    preds, refs = {}, {}
    for k in (1, 3):
        spec = dict(k=k, discounts=(1.0,) * k, entity_cols=(), features=["x1", "x2"])
        preds[k] = predict_on(AbilityModel(**spec).fit(train), test)
        fit = exploded_fit(X, codes, pos, k)
        coef = np.asarray(fit.coef)
        cov = np.linalg.inv(exploded_information(X, codes, pos, k, coef))
        refs[k] = {
            "u": Xt @ coef,
            "p": softmax_within(Xt @ coef, t_codes),
            "se": np.sqrt(np.einsum("ij,jk,ik->i", Xt, cov, Xt)),
        }
    return preds, refs, t_codes


def test_top3_training_gives_smaller_standard_errors_than_winner_only(features_only):
    """se_u is u's standard error from the fit (within 10% of the exploded conditional logit's
    delta-method value, at k = 1 and at k = 3), and top-3 training's is below 0.8 of winner-only's
    for every runner (about 0.6 here: 2.7 times the information, 05 §2.5)."""
    preds, refs, _ = features_only
    for k in (1, 3):
        se = preds[k]["se_u"].to_numpy(dtype=float)
        np.testing.assert_allclose(se, refs[k]["se"], rtol=0.1, atol=1e-3, err_msg=f"k={k}")
    ratio = preds[3]["se_u"].to_numpy(dtype=float) / preds[1]["se_u"].to_numpy(dtype=float)
    assert np.median(ratio) < 0.8, np.median(ratio)
    assert np.quantile(ratio, 0.9) < 0.8, np.quantile(ratio, 0.9)


def test_ability_model_matches_the_exploded_conditional_logit(features_only):
    """Without entities, the k = 1 model is the conditional logit and the k = 3 unit-discount
    model the exploded one: p_ability and u (less its race mean) match their fits."""
    preds, refs, t_codes = features_only
    for k in (1, 3):
        p = preds[k]["p_ability"].to_numpy(dtype=float)
        np.testing.assert_allclose(p, refs[k]["p"], atol=1e-3, err_msg=f"k={k}")
        u = preds[k]["u"].to_numpy(dtype=float)
        centred = u - pd.Series(u).groupby(t_codes).transform("mean").to_numpy()
        np.testing.assert_allclose(centred, refs[k]["u"], atol=1e-3, err_msg=f"k={k}")


# ---------------------------------------------------------------------------------------------
# DoD: entity effects on the training window only
# ---------------------------------------------------------------------------------------------

WINDOW_SEED = 9245


def test_entity_effects_use_the_training_window_only():
    """J_FLIP is +0.8 over the training window (the first 300 races) and -0.8 over the test
    window (the last 200). Fitted on the training window, the model rates it above 0.6 (its
    maximum likelihood there is above 0.8; over both windows, below 0.45), and the model fitted
    on both windows rates it lower. Its predictions for the test window never read that window's
    outcomes: reversing every test race's order, or dropping finish_pos, changes nothing."""
    frame, planted = window_frame(WINDOW_SEED)
    codes, u, flip = planted["codes"], planted["u"], planted["flip"]
    in_train = frame["window"].eq("train").to_numpy()
    rides = (frame["jockey_id"] == J_FLIP).to_numpy()
    pos = frame["finish_pos"].to_numpy()
    rest = u - flip
    tr_codes = pd.factorize(codes[in_train])[0]
    mle_train = exploded_fit(rides[in_train], tr_codes, pos[in_train], 3, offset=rest[in_train])
    mle_all = exploded_fit(rides, codes, pos, 3, offset=rest)
    assert float(mle_train.coef[0]) > 0.8 and float(mle_all.coef[0]) < 0.45

    frame = frame.drop(columns=["window"])
    train, test = frame[in_train], frame[~in_train]
    spec = dict(k=3, discounts=UNIT3, entity_cols=("jockey_id",), features=["x1"])
    model = AbilityModel(**spec).fit(train)
    got = probe(model, [J_FLIP, "j:new"], entity_cols=("jockey_id",))
    rated = got.loc[J_FLIP, "u"] - got.loc["j:new", "u"]
    assert rated > 0.6, rated
    both = probe(AbilityModel(**spec).fit(frame), [J_FLIP, "j:new"], entity_cols=("jockey_id",))
    assert both.loc[J_FLIP, "u"] - both.loc["j:new", "u"] < rated - 0.2

    base = predict_on(model, test)
    last = test.groupby("race_id")["finish_pos"].transform("max")
    reversed_order = test.assign(finish_pos=last + 1 - test["finish_pos"])
    assert (reversed_order["finish_pos"] != test["finish_pos"]).mean() > 0.8
    for changed in (reversed_order, test.drop(columns=["finish_pos"])):
        again = predict_on(model, changed)
        for col in ("u", "se_u", "p_ability"):
            np.testing.assert_allclose(
                again[col].to_numpy(dtype=float), base[col].to_numpy(dtype=float), atol=1e-12
            )


# ---------------------------------------------------------------------------------------------
# DoD: discounts are fitted on inner folds only
# ---------------------------------------------------------------------------------------------

# D9-style walk-forward months; the inner folds of the tests below: January trains, February,
# March and April are tested in turn (expanding).
INNER = FoldScheme(
    unit="month",
    test_length=1,
    step=1,
    min_train=1,
    window="expanding",
    embargo_days=0,
    anchor_month=1,
)
INNER_SEED = 9257
OUTER_SEED = 9258
INNER_FIRST = date(2024, 1, 1)
OUTER_FIRST = date(2024, 5, 1)
OUTER_LAST = date(2024, 12, 31)


def shuffled(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Rows permuted so dates interleave, on index labels that are not positions."""
    rng = np.random.default_rng(seed)
    out = frame.iloc[rng.permutation(len(frame))]
    out.index = pd.Index(50_000 + rng.permutation(len(out)), name="row")
    return out


def check_discounts(w) -> tuple[float, ...]:
    assert isinstance(w, tuple) and len(w) == 3, w
    w = tuple(float(x) for x in w)
    assert all(math.isfinite(x) and x >= 0 for x in w) and w[0] > 0, w
    return w


def test_discounts_fit_on_inner_folds_only():
    """The inner folds are monthly walk-forward folds over January to April (positions into the
    frame, built by walk_forward with the outer window excluded, as a caller builds them over an
    outer fold's training rows); May to September, 300 races, is the outer test window and in no
    fold. Whether that window follows the planted model or has its places reversed, or is left out
    of the frame, fit_discounts returns the same discounts. A search that read it would move: a
    reference grid search goes from (1, 1, 1) to (1, 0, 0) when it also trains on the reversed
    window, and to (1, 0, 0.25) when it builds its own folds over the whole frame."""
    inner = discount_frame(INNER_SEED, 200, "pl", INNER_FIRST)
    results = []
    frame = None
    for places in ("pl", "reversed"):
        outer = discount_frame(OUTER_SEED, 300, places, OUTER_FIRST, code_from=10_000)
        frame = shuffled(pd.concat([inner, outer], ignore_index=True), 9259)
        folds = walk_forward(frame["race_date"], scheme=INNER, exclude=(OUTER_FIRST, OUTER_LAST))
        assert len(folds) == 3
        outer_rows = (pd.to_datetime(frame["race_date"]) >= pd.Timestamp(OUTER_FIRST)).to_numpy()
        in_fold = np.zeros(len(frame), dtype=bool)
        for fold in folds:
            in_fold[fold.train_idx] = True
            in_fold[fold.test_idx] = True
        assert outer_rows.sum() > 2_000 and not in_fold[outer_rows].any()
        assert in_fold[~outer_rows].all()
        w = fit_discounts(frame, folds, features=FEATURES_10, entity_cols=("jockey_id",))
        results.append(check_discounts(w))

    inner_only = frame[(pd.to_datetime(frame["race_date"]) < pd.Timestamp(OUTER_FIRST)).to_numpy()]
    folds = walk_forward(inner_only["race_date"], scheme=INNER)
    w = fit_discounts(inner_only, folds, features=FEATURES_10, entity_cols=("jockey_id",))
    results.append(check_discounts(w))
    for other in results[1:]:
        np.testing.assert_allclose(other, results[0], rtol=0, atol=1e-9)


def test_discounts_follow_the_places_evidence_on_the_inner_folds():
    """The same runners and features over the same inner folds, with places that follow the
    planted Plackett-Luce order or are drawn the wrong way round (the winner from softmax(u), the
    places from Plackett-Luce on -u; at unit discounts a reference fit loses about 120 nats out of
    fold to winner-only). The reversed places get a second-place discount below half the winner's,
    and the planted places one at least 0.25 (of w1) above that: the discounts are fitted, not
    fixed."""
    found = {}
    for places in ("pl", "reversed"):
        frame = shuffled(discount_frame(INNER_SEED, 200, places, INNER_FIRST), 9260)
        folds = walk_forward(frame["race_date"], scheme=INNER)
        assert len(folds) == 3
        w = fit_discounts(frame, folds, features=FEATURES_10, entity_cols=("jockey_id",))
        w = check_discounts(w)
        found[places] = w[1] / w[0]
    assert found["reversed"] < 0.5, found
    assert found["pl"] > found["reversed"] + 0.25, found
