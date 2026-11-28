"""The meld of the horse, speed and rating members in the C4 meta-model (ticket #98, epic #73;
08 §3.2 C4 and cross-cutting, §4 E41, E42).

Contract pinned here (`horsetrack.countries.japan.models.meld`):
- `member_frame(members, base) -> pd.DataFrame`. Each member frame has race_id, horse_id, logp and
  oof_fold (the fold its row is out of fold for: the test quarter's start, e.g. "2015-01-01");
  `base` has race_id, horse_id and `fold`, the meld's own fold of the row in the same labels. The
  result is the base rows, in base order with every base column, plus one column per member, named
  by its key, holding its logp as given. A member row is out of fold for the meld only when its
  oof_fold equals the base row's fold: one with no oof_fold (null or blank), or with another fold's
  label, refuses with `models.stacking.InvalidProvenance` naming the member. Member rows that match
  no base row are ignored, whatever their oof_fold; a base race in which some runner lacks some
  member is dropped whole; a member repeating an aligned (race_id, horse_id) is a ValueError.
- `precision_weight(sigma, *, alpha0, kappa)`: alpha0 / (1 + kappa * sigma^2), elementwise.
- `fit_meld(frame, *, members, gates, segment_cols, precision, kappa=..., form=..., ridge=...,
  quadratic=...) -> MetaFit` is exactly `models.meta.fit_meta` on an expanded design: for each
  member k and each gate cell c (the combination of the row's values of every gate column; one cell
  when there are no gates) the member column w_k * m_k on the rows of cell c and 0 elsewhere, where
  w_k = 1 / (1 + kappa[k] * sigma_k^2) with sigma_k the frame's column `precision[k]` (w_k = 1 for a
  member not in `precision`), with the same segment columns, form, ridge and quadratic. So every
  gate cell has its own alpha_k(z), the market terms are shared, and the ridge shrinks every member
  term toward alpha = 0 (the owner's target, 2026-11-24). Its coef and se are that fit's, its
  loglik and n_races too; `fit.spec` carries the form, ridge and quadratic it used.
- `MetaFit.predict(frame, q)` of a meld reads the raw columns only (race_id, the members, the gate,
  segment and sigma columns) and takes any positive price vector: one probability per row, each
  race summing to 1, only ratios of q within a race mattering, a ln_q_final column ignored.
- `marginal_ablation(frame, members, *, clusters, n_boot, level, seed) -> pd.DataFrame`: one row
  per variant, "all" first, then f"all minus {k}" per member in the order given (`variant`).
  Out of fold, walk-forward over the frame's `fold` labels: each fold after the first is scored by
  fits on the earlier folds only, and the first is never scored; `races` is the number of scored
  races, the same on every row. Per scored race, ll is the winner log loss; `delta_ll` is the mean
  of ll(market only) - ll(variant), the market-only meta fitted the same way; `gain` is the mean of
  ll(all minus k) - ll(all), the marginal gain of k (not pinned on the "all" row). `*_lo` and
  `*_hi` are the (1 - level) and level quantiles of a cluster bootstrap of each mean, `clusters`
  mapping race id to its meeting (D7), with one set of draws shared by every variant.
- `ambiguity(frame, members) -> float`: per race, each member normalised within the race (softmax
  of its column), p_bar the normalised geometric mean of the members (equal weights) and the
  race's term the mean over members of KL(p_bar || p_k); the result is the mean over races.
- `member_correlations(frame, members) -> pd.DataFrame`: the Pearson correlation matrix of the
  members' logp demeaned within each race, index and columns the members in the order given.

Expected values come from first principles or from `fit_meta` (#87's tested fit, not the code
under test) on a design the test builds itself. All data is synthetic and every RNG is seeded.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math

import numpy as np
import pandas as pd
import pytest

from horsetrack.countries.japan.models.meld import (
    ambiguity,
    fit_meld,
    marginal_ablation,
    member_correlations,
    member_frame,
    precision_weight,
)
from horsetrack.countries.japan.models.meta import MetaFit, MetaSpec, fit_meta
from horsetrack.countries.japan.models.stacking import InvalidProvenance

Q = "ln_q_final"
FOLDS = ("2015-01-01", "2015-04-01", "2015-07-01", "2015-10-01")
RACES_PER_MEETING = 12
MEMBERS = ["p_form", "p_speed"]
INTERVALS = ["delta_ll", "delta_ll_lo", "delta_ll_hi", "gain", "gain_lo", "gain_hi"]


# ---------------------------------------------------------------------------------------------
# Synthetic races
# ---------------------------------------------------------------------------------------------


def codes_of(race_id):
    return pd.factorize(np.asarray(race_id, dtype=object))[0]


def softmax_in_race(u, codes):
    u = np.asarray(u, dtype=float)
    top = pd.Series(u).groupby(codes).transform("max").to_numpy()
    e = np.exp(u - top)
    return e / np.bincount(codes, weights=e)[codes]


def log_softmax(v, codes):
    return np.log(softmax_in_race(v, codes))


def draw_winners(rng, u, codes):
    """One winner per race drawn from softmax(u) within the race (Gumbel-max)."""
    g = np.asarray(u, dtype=float) + rng.gumbel(size=len(u))
    top = pd.Series(g).groupby(codes).transform("max").to_numpy()
    return g == top


def market(rng, codes):
    """A Dirichlet(1) market per race, clipped away from 0 and 1 and renormalised."""
    g = rng.gamma(1.0, size=len(codes))
    q = g / np.bincount(codes, weights=g)[codes]
    for _ in range(3):
        q = np.clip(q, 0.003, 0.75)
        q = q / np.bincount(codes, weights=q)[codes]
    return q


def race_sums(p, race_id):
    return pd.Series(np.asarray(p, dtype=float)).groupby(np.asarray(race_id, dtype=object)).sum()


def runners(rng, n_races, sizes=(5, 14)):
    n = rng.integers(sizes[0], sizes[1] + 1, size=n_races)
    return np.repeat(np.arange(n_races), n)


def world(seed, *, meetings_per_fold, folds=FOLDS):
    """Meetings of 12 races, `meetings_per_fold` of them in each fold, in fold order: race_id,
    horse_id, meeting_id, fold and ln_q_final. Returns the RNG to draw the rest from."""
    rng = np.random.default_rng(seed)
    n_races = meetings_per_fold * len(folds) * RACES_PER_MEETING
    codes = runners(rng, n_races)
    meeting = np.arange(n_races) // RACES_PER_MEETING
    fold = np.asarray(folds, dtype=object)[meeting // meetings_per_fold]
    race = np.array(
        [
            f"jra:{fold[r]}:m{meeting[r]:03d}:{r % RACES_PER_MEETING + 1:02d}"
            for r in range(n_races)
        ],
        dtype=object,
    )
    frame = pd.DataFrame(
        {
            "race_id": race[codes],
            "horse_id": [f"h{i:06d}" for i in range(len(codes))],
            "meeting_id": np.array([f"m{m:03d}" for m in meeting], dtype=object)[codes],
            "fold": fold[codes],
            Q: np.log(market(rng, codes)),
        }
    )
    return rng, frame, codes


def meetings_of(frame):
    return frame.groupby("race_id", sort=False)["meeting_id"].first()


def meld_world(seed, n_races=1_600):
    """Races with two members, a race-level debut gate (0/1), a race-level surface ("turf" or
    "dirt"), a race-level class segment z_class and a per-runner sigma for p_speed. Winners come
    from u = ln q + alpha_form * p_form + alpha_speed * p_speed / (1 + 4 sigma^2), alpha_form 0 in
    debut races and 0.7 + 0.15 z otherwise, alpha_speed 1.0 in debut races and 0.2 otherwise, plus
    0.3 on turf."""
    rng = np.random.default_rng(seed)
    codes = runners(rng, n_races)
    n = len(codes)
    q = market(rng, codes)
    debut = (rng.random(n_races) < 0.3).astype(int)[codes]
    surface = np.array(["turf", "dirt"], dtype=object)[rng.integers(0, 2, size=n_races)][codes]
    z = rng.integers(0, 3, size=n_races)[codes].astype(float)
    sigma = rng.uniform(0.0, 1.0, size=n)
    p_form = log_softmax(rng.normal(size=n), codes)
    p_speed = log_softmax(rng.normal(size=n), codes)
    a_form = np.where(debut == 1, 0.0, 0.7 + 0.15 * z)
    a_speed = np.where(debut == 1, 1.0, 0.2) + np.where(surface == "turf", 0.3, 0.0)
    u = np.log(q) + a_form * p_form + a_speed * p_speed / (1.0 + 4.0 * sigma**2)
    won = draw_winners(rng, u, codes)
    frame = pd.DataFrame(
        {
            "race_id": np.array([f"jra:t{c:05d}" for c in range(n_races)], dtype=object)[codes],
            "horse_id": [f"h{i:06d}" for i in range(n)],
            "won": won,
            Q: np.log(q),
            "p_form": p_form,
            "p_speed": p_speed,
            "debut": debut,
            "surface": surface,
            "z_class": z,
            "sigma_speed": sigma,
        }
    )
    return frame


def fresh_races(seed, n_races=60):
    """Races to predict, with the raw columns a meld reads and nothing else, and a market q."""
    rng = np.random.default_rng(seed)
    codes = runners(rng, n_races, sizes=(3, 12))
    n = len(codes)
    frame = pd.DataFrame(
        {
            "race_id": np.array([f"jra:new{c:04d}" for c in range(n_races)], dtype=object)[codes],
            "horse_id": [f"n{i:05d}" for i in range(n)],
            "p_form": log_softmax(rng.normal(size=n), codes),
            "p_speed": log_softmax(rng.normal(size=n), codes),
            "debut": (rng.random(n_races) < 0.4).astype(int)[codes],
            "surface": np.array(["turf", "dirt"], dtype=object)[rng.integers(0, 2, n_races)][codes],
            "z_class": rng.integers(0, 3, size=n_races)[codes].astype(float),
            "sigma_speed": rng.uniform(0.0, 1.2, size=n),
        }
    )
    return frame, market(rng, codes)


@pytest.fixture(scope="module")
def planted():
    return meld_world(9810)


# ---------------------------------------------------------------------------------------------
# The reference meld: fit_meta on a design the test expands itself
# ---------------------------------------------------------------------------------------------


def cell_labels(frame, gates):
    if not gates:
        return np.full(len(frame), "all", dtype=object)
    return frame[list(gates)].astype(str).agg("|".join, axis=1).to_numpy(dtype=object)


def reference_meld(
    frame, members, *, gates, segment_cols, precision, kappa, form, ridge, quadratic
):
    """fit_meta on the expanded design of the contract, and the function expanding a frame."""
    cells = sorted(set(cell_labels(frame, gates)))

    def expand(f):
        label = cell_labels(f, gates)
        cols = {}
        for k in members:
            m = f[k].to_numpy(dtype=float)
            if k in precision:
                s = f[precision[k]].to_numpy(dtype=float)
                m = m / (1.0 + kappa[k] * s * s)
            for c in cells:
                cols[f"{k}@{c}"] = np.where(label == c, m, 0.0)
        out = pd.DataFrame({"race_id": f["race_id"].to_numpy(dtype=object)}, index=f.index)
        for z in segment_cols:
            out[z] = f[z].to_numpy(dtype=float)
        for name, values in cols.items():
            out[name] = values
        return out

    train = expand(frame)
    train["won"] = frame["won"].to_numpy(dtype=bool)
    train[Q] = frame[Q].to_numpy(dtype=float)
    names = [f"{k}@{c}" for k in members for c in cells]
    spec = MetaSpec(
        members=names,
        quadratic=quadratic,
        segment_cols=list(segment_cols),
        form=form,
        ridge=ridge,
    )
    return fit_meta(train, spec), expand


def assert_same_fit(fit, ref):
    """The meld's coefficients and standard errors are the reference's, as sets of values (the
    term names are the meld's own), and so are its likelihood and race count."""
    assert len(fit.coef) == len(ref.coef)
    np.testing.assert_allclose(np.sort(fit.coef.to_numpy(float)), np.sort(ref.coef), atol=1e-6)
    np.testing.assert_allclose(np.sort(fit.se.to_numpy(float)), np.sort(ref.se), rtol=1e-5)
    assert fit.loglik == pytest.approx(ref.loglik, abs=1e-6)
    assert fit.n_races == ref.n_races


# ---------------------------------------------------------------------------------------------
# Outline 1: member_frame aligns the out-of-fold members onto the meld's rows
# ---------------------------------------------------------------------------------------------


def provenance_world(seed=9820):
    """A base of 3 folds x 4 races (3 to 7 runners) on non-default index labels, and two members
    over it, each out of fold for the row's own fold, shuffled, with an extra column and with
    rows of races the base does not hold (no oof_fold: they are in no fold's test window)."""
    rng = np.random.default_rng(seed)
    rows = []
    for f, fold in enumerate(FOLDS[:3]):
        for r in range(4):
            for h in range(int(rng.integers(3, 8))):
                race = f"jra:{fold}:QQV:{r + 1:02d}"
                rows.append({"race_id": race, "horse_id": f"h{f}{r}{h:02d}", "fold": fold})
    base = pd.DataFrame(rows)
    base["won"] = base.groupby("race_id").cumcount().to_numpy() == 0
    base[Q] = rng.normal(size=len(base))
    base["debut"] = rng.integers(0, 2, size=len(base))
    base.index = pd.RangeIndex(500, 500 + len(base))

    def member(member_seed):
        r = np.random.default_rng(member_seed)
        own = pd.DataFrame(
            {
                "race_id": base["race_id"].to_numpy(dtype=object),
                "horse_id": base["horse_id"].to_numpy(dtype=object),
                "logp": r.normal(-2.0, 0.7, size=len(base)),
                "oof_fold": pd.array(base["fold"].tolist(), dtype="string"),
                "sigma_before": r.uniform(0.1, 1.0, size=len(base)),
            }
        )
        other = pd.DataFrame(
            {
                "race_id": ["jra:2014-10-01:QQV:01"] * 4,
                "horse_id": [f"x{i}" for i in range(4)],
                "logp": np.log([0.4, 0.3, 0.2, 0.1]),
                "oof_fold": pd.array([None] * 4, dtype="string"),
                "sigma_before": [0.5] * 4,
            }
        )
        both = pd.concat([own, other], ignore_index=True)
        return both.iloc[r.permutation(len(both))].reset_index(drop=True)

    return base, {"rating": member(seed + 1), "ability": member(seed + 2)}


def lookup(member, base):
    keys = zip(member["race_id"], member["horse_id"], strict=True)
    by_key = dict(zip(keys, member["logp"], strict=True))
    rows = zip(base["race_id"], base["horse_id"], strict=True)
    return np.array([by_key[k] for k in rows], dtype=float)


def test_member_frame_aligns_each_member_by_race_and_horse_onto_the_base_rows():
    """Each member's logp lands on its own (race_id, horse_id) row of the base, in base order,
    whatever order the member file is in; every base column is kept; rows of races the base does
    not hold are ignored even though they are in no fold; a race in which a runner lacks a member
    is dropped whole, the other races kept whole."""
    base, members = provenance_world()
    out = member_frame(members, base)

    assert len(out) == len(base)
    for col in base.columns:
        np.testing.assert_array_equal(out[col].to_numpy(), base[col].to_numpy())
    for name, member in members.items():
        np.testing.assert_allclose(out[name].to_numpy(dtype=float), lookup(member, base), rtol=0)

    gap_race = base["race_id"].unique()[5]
    lacking = members["ability"].loc[members["ability"]["race_id"] == gap_race].index[0]
    gapped = {**members, "ability": members["ability"].drop(index=lacking)}
    out = member_frame(gapped, base)
    kept = base.loc[base["race_id"] != gap_race]
    assert len(kept) < len(base)
    np.testing.assert_array_equal(out["race_id"].to_numpy(), kept["race_id"].to_numpy())
    np.testing.assert_array_equal(out["horse_id"].to_numpy(), kept["horse_id"].to_numpy())
    for name, member in members.items():
        np.testing.assert_allclose(out[name].to_numpy(dtype=float), lookup(member, kept), rtol=0)


def test_member_frame_refuses_a_member_repeating_an_aligned_runner():
    base, members = provenance_world()
    rating = members["rating"]
    aligned = rating.loc[rating["race_id"].isin(base["race_id"])]
    repeated = pd.concat([rating, aligned.iloc[[0]].assign(logp=-1.0)], ignore_index=True)
    with pytest.raises(ValueError):
        member_frame({**members, "rating": repeated}, base)


def test_in_fold_member_rows_are_refused():
    """DoD: every member is out of fold relative to the meld's own folds. A member row is out of
    fold for the meld only when its oof_fold is the base row's fold: no oof_fold (null or blank)
    means its model may have trained on the row, a later fold's label means it did (that fold
    trained through the row's quarter), and an earlier one is not the fold the meld scores the row
    in. One such row among the aligned rows refuses, naming the member."""
    base, members = provenance_world()
    member_frame(members, base)  # the clean members pass

    row = base.loc[base["fold"] == FOLDS[1]].iloc[2]
    cases = {"null": None, "blank": "", "later fold": FOLDS[2], "earlier fold": FOLDS[0]}
    for name in ("rating", "ability"):
        member = members[name]
        hit = (member["race_id"] == row["race_id"]) & (member["horse_id"] == row["horse_id"])
        assert hit.sum() == 1
        for label in cases.values():
            bad = member.copy()
            bad["oof_fold"] = bad["oof_fold"].astype("string")
            bad.loc[hit, "oof_fold"] = pd.NA if label is None else label
            with pytest.raises(InvalidProvenance, match=name):
                member_frame({**members, name: bad}, base)


# ---------------------------------------------------------------------------------------------
# Outline 1: precision weighting
# ---------------------------------------------------------------------------------------------


def test_precision_weight_falls_as_sigma_grows():
    """alpha_i = alpha0 / (1 + kappa sigma_i^2): alpha0 = 0.2, kappa = 4 and sigma = 0.5 give 0.1;
    sigma = 0 gives alpha0; the weight falls in size as sigma grows and reaches 0 as sigma goes to
    infinity, for a negative alpha0 too; the shape of sigma is kept."""
    got = np.asarray(precision_weight(np.array([0.5]), alpha0=0.2, kappa=4.0), dtype=float)
    np.testing.assert_allclose(got, [0.1], rtol=1e-12)

    sigma = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 10.0, 1e3, 1e8, np.inf])
    for alpha0, kappa in ((0.2, 4.0), (1.0, 0.5), (0.05, 25.0), (-0.3, 4.0)):
        got = np.asarray(precision_weight(sigma, alpha0=alpha0, kappa=kappa), dtype=float)
        expected = np.array([alpha0 / (1.0 + kappa * s * s) for s in sigma[:-1]] + [0.0])
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)
        assert got[0] == pytest.approx(alpha0, rel=1e-15)
        assert np.all(np.diff(np.abs(got)) < 0)
        assert got[-1] == 0.0
        assert abs(got[-2]) < 1e-15

    grid = np.array([[0.0, 0.5], [1.0, 2.0]])
    got = np.asarray(precision_weight(grid, alpha0=0.2, kappa=4.0), dtype=float)
    assert got.shape == (2, 2)
    np.testing.assert_allclose(got, [[0.2, 0.1], [0.04, 0.2 / 17.0]], rtol=1e-12)


# ---------------------------------------------------------------------------------------------
# Outline 1: fit_meld is fit_meta with gates, segment columns and precision-weighted members
# ---------------------------------------------------------------------------------------------


def test_gates_fit_separate_weights_per_gate_level(planted):
    """With gates=["debut"] the meld is fit_meta on each member split into one column per gate
    level: the same coefficients, standard errors, likelihood and predictions, here with
    p_form's weight near 0 in debut races and near 0.7 to 1 in the others. With two gates the cells
    are their combinations. The pooled (ungated) meld predicts differently."""
    q = np.exp(planted[Q].to_numpy())
    fresh, q_new = fresh_races(9811)
    raw = fresh[["race_id", "horse_id", *MEMBERS, "debut", "surface"]]

    gated = {}
    for gates in (["debut"], ["debut", "surface"]):
        fit = fit_meld(planted, members=MEMBERS, gates=gates, segment_cols=[], precision={})
        assert isinstance(fit, MetaFit)
        ref, expand = reference_meld(
            planted,
            MEMBERS,
            gates=gates,
            segment_cols=[],
            precision={},
            kappa={},
            form=fit.spec.form,
            ridge=fit.spec.ridge,
            quadratic=fit.spec.quadratic,
        )
        assert_same_fit(fit, ref)
        got = np.asarray(fit.predict(planted, q), dtype=float)
        np.testing.assert_allclose(got, ref.predict(expand(planted), q), atol=1e-6)
        got_new = np.asarray(fit.predict(raw, q_new), dtype=float)
        np.testing.assert_allclose(got_new, ref.predict(expand(raw), q_new), atol=1e-6)
        gated[len(gates)] = got

    # the planted weights differ by level, so the gated design is not the pooled one
    ref, _ = reference_meld(
        planted,
        MEMBERS,
        gates=["debut"],
        segment_cols=[],
        precision={},
        kappa={},
        form="stage2",
        ridge=0.0,
        quadratic=False,
    )
    assert abs(ref.coef["p_form@0"] - ref.coef["p_form@1"]) > 4 * math.hypot(
        ref.se["p_form@0"], ref.se["p_form@1"]
    )
    pooled = fit_meld(planted, members=MEMBERS, gates=[], segment_cols=[], precision={})
    got_pooled = np.asarray(pooled.predict(planted, q), dtype=float)
    assert np.abs(gated[1] - got_pooled).max() > 0.1
    assert np.abs(gated[2] - gated[1]).max() > 0.03


SETTINGS = {
    "stage2, ridge 2": {"form": "stage2", "ridge": 2.0, "quadratic": False},
    "offset, quadratic": {"form": "offset", "ridge": 0.0, "quadratic": True},
}


@pytest.mark.parametrize("settings", list(SETTINGS.values()), ids=list(SETTINGS))
def test_fit_meld_is_fit_meta_on_gated_precision_weighted_members(planted, settings):
    """Gates, a segment column and a precision-weighted member together, under a ridge or in the
    offset form with the quadratic term: the meld is the expanded fit_meta, which also shows the
    ridge falls on every gated member term and never on the market's."""
    kw = {"gates": ["debut"], "segment_cols": ["z_class"], "precision": {"p_speed": "sigma_speed"}}
    fit = fit_meld(planted, members=MEMBERS, kappa={"p_speed": 4.0}, **kw, **settings)
    ref, expand = reference_meld(planted, MEMBERS, kappa={"p_speed": 4.0}, **kw, **settings)
    assert isinstance(fit, MetaFit)
    assert_same_fit(fit, ref)
    q = np.exp(planted[Q].to_numpy())
    np.testing.assert_allclose(fit.predict(planted, q), ref.predict(expand(planted), q), atol=1e-6)


def test_a_precision_member_is_weighted_by_one_over_one_plus_kappa_sigma_squared(planted):
    """precision={"p_speed": "sigma_speed"} with kappa 4 fits alpha0 on p_speed / (1 + 4 sigma^2):
    the expanded fit_meta, for kappa 4 and for kappa 1. The planted truth weights p_speed so, and
    the precision-weighted meld fits it better than the constant-alpha one."""
    q = np.exp(planted[Q].to_numpy())
    fits = {}
    for kappa in (4.0, 1.0):
        fit = fit_meld(
            planted,
            members=MEMBERS,
            gates=["debut"],
            segment_cols=[],
            precision={"p_speed": "sigma_speed"},
            kappa={"p_speed": kappa},
            form="stage2",
            ridge=0.0,
        )
        ref, expand = reference_meld(
            planted,
            MEMBERS,
            gates=["debut"],
            segment_cols=[],
            precision={"p_speed": "sigma_speed"},
            kappa={"p_speed": kappa},
            form="stage2",
            ridge=0.0,
            quadratic=False,
        )
        assert_same_fit(fit, ref)
        np.testing.assert_allclose(
            fit.predict(planted, q), ref.predict(expand(planted), q), atol=1e-6
        )
        fits[kappa] = fit

    constant = fit_meld(
        planted,
        members=MEMBERS,
        gates=["debut"],
        segment_cols=[],
        precision={},
        form="stage2",
        ridge=0.0,
    )
    assert fits[4.0].loglik > constant.loglik + 20.0
    assert fits[4.0].loglik > fits[1.0].loglik + 5.0


# ---------------------------------------------------------------------------------------------
# DoD: MetaFit.predict(frame, q) still takes any price vector
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("settings", list(SETTINGS.values()), ids=list(SETTINGS))
def test_meld_predict_takes_any_price_vector(planted, settings):
    """A meld with a gate, a segment and a precision member predicts new races from their raw
    columns alone (no derived column, no won, no ln_q_final) for any positive q: the expanded
    fit's prediction, each race summing to 1, moving with q, the same under a per-race rescaling
    of q or a row shuffle, and blind to a stale ln_q_final column."""
    fit = fit_meld(
        planted,
        members=MEMBERS,
        gates=["debut"],
        segment_cols=["z_class"],
        precision={"p_speed": "sigma_speed"},
        kappa={"p_speed": 4.0},
        **settings,
    )
    ref, expand = reference_meld(
        planted,
        MEMBERS,
        gates=["debut"],
        segment_cols=["z_class"],
        precision={"p_speed": "sigma_speed"},
        kappa={"p_speed": 4.0},
        **settings,
    )
    assert isinstance(fit, MetaFit)
    fresh, q1 = fresh_races(9812)
    fresh = fresh[["race_id", "horse_id", *MEMBERS, "debut", "z_class", "sigma_speed"]]
    q2 = market(np.random.default_rng(9813), codes_of(fresh["race_id"]))

    p1 = np.asarray(fit.predict(fresh, q1), dtype=float)
    p2 = np.asarray(fit.predict(fresh, q2), dtype=float)
    assert p1.shape == (len(fresh),)
    assert np.all(np.isfinite(p1)) and np.all(p1 > 0)
    np.testing.assert_allclose(race_sums(p1, fresh["race_id"]), 1.0, atol=1e-12)
    np.testing.assert_allclose(race_sums(p2, fresh["race_id"]), 1.0, atol=1e-12)
    np.testing.assert_allclose(p1, ref.predict(expand(fresh), q1), atol=1e-6)
    np.testing.assert_allclose(p2, ref.predict(expand(fresh), q2), atol=1e-6)
    assert np.abs(p1 - p2).max() > 0.05

    scale = np.random.default_rng(9814).uniform(0.5, 2.0, size=fresh["race_id"].nunique())
    scaled = q1 * scale[codes_of(fresh["race_id"])]
    np.testing.assert_allclose(fit.predict(fresh, scaled), p1, atol=1e-12)
    perm = np.random.default_rng(9815).permutation(len(fresh))
    shuffled = np.asarray(fit.predict(fresh.iloc[perm], q1[perm]), dtype=float)
    np.testing.assert_allclose(shuffled, p1[perm], atol=1e-12)
    stale = fresh.assign(**{Q: np.log(q2)})
    np.testing.assert_allclose(fit.predict(stale, q1), p1, atol=1e-12)


# ---------------------------------------------------------------------------------------------
# Outline 1 and DoD: the marginal table, out of fold, with one set of bootstrap draws
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ablation_world():
    """4 folds x 30 meetings x 12 races. p_form carries signal at half the meetings only (weight
    1, else 0), p_speed everywhere (0.4); p_noise, a_noise and z_noise are drawn after the winners
    and carry none."""
    rng, frame, codes = world(9837, meetings_per_fold=30)
    n = len(frame)
    meeting = pd.factorize(frame["meeting_id"])[0]
    strong = rng.random(meeting.max() + 1) < 0.5
    p_form = log_softmax(rng.normal(size=n), codes)
    p_speed = log_softmax(rng.normal(size=n), codes)
    u = frame[Q].to_numpy() + np.where(strong[meeting], 1.0, 0.0) * p_form + 0.4 * p_speed
    won = draw_winners(rng, u, codes)
    noise = {k: log_softmax(rng.normal(size=n), codes) for k in ("p_noise", "a_noise", "z_noise")}
    return frame.assign(won=won, p_form=p_form, p_speed=p_speed, **noise)


def by_variant(table):
    return table.set_index("variant")


def scored_races(frame):
    return frame.loc[frame["fold"] != min(frame["fold"]), "race_id"].nunique()


def test_marginal_table_has_every_variant_out_of_fold_with_cluster_bounds(ablation_world):
    """One row per variant, all first; every row scores the same races, those after the first
    fold; the meld beats the market-only meta and each signal member adds to it, with lower bounds
    above 0; each member's gain is the all row's delta_ll less its own row's (the same races and
    the same market); the bounds are quantiles at `level` of a bootstrap over the meetings, so a
    lower level narrows them and resampling single races, which ignores p_form's meeting-level
    signal, narrows its gain's bounds a lot."""
    members = ["p_form", "p_speed", "p_noise"]
    clusters = meetings_of(ablation_world)
    kw = {"n_boot": 1_000, "level": 0.95, "seed": 98}
    table = marginal_ablation(ablation_world, members, clusters=clusters, **kw)

    assert table["variant"].tolist() == ["all", *(f"all minus {k}" for k in members)]
    t = by_variant(table)
    assert (t["races"] == scored_races(ablation_world)).all()
    assert t.loc["all", "delta_ll_lo"] > 0
    for k in ("p_form", "p_speed"):
        assert t.loc[f"all minus {k}", "gain_lo"] > 0, k
    for k in members:
        row = t.loc[f"all minus {k}"]
        assert row["gain"] == pytest.approx(t.loc["all", "delta_ll"] - row["delta_ll"], abs=1e-9)
    for v in t.index:
        row = t.loc[v]
        assert row["delta_ll_lo"] < row["delta_ll"] < row["delta_ll_hi"], v
        if v != "all":
            assert row["gain_lo"] < row["gain"] < row["gain_hi"], v

    narrow = by_variant(
        marginal_ablation(ablation_world, members, clusters=clusters, **{**kw, "level": 0.6})
    )
    for v in t.index:
        assert narrow.loc[v, "delta_ll_lo"] > t.loc[v, "delta_ll_lo"], v
        assert narrow.loc[v, "delta_ll_hi"] < t.loc[v, "delta_ll_hi"], v

    races = pd.Series(clusters.index, index=clusters.index)
    single = by_variant(marginal_ablation(ablation_world, members, clusters=races, **kw))
    width = {
        name: table_.loc["all minus p_form", "gain_hi"] - table_.loc["all minus p_form", "gain_lo"]
        for name, table_ in (("meetings", t), ("races", single))
    }
    assert width["meetings"] > 1.25 * width["races"], width


def test_marginal_gains_are_scored_walk_forward_on_later_folds():
    """p_flip's weight is +1 in the first two folds and -1 in the last two. Fitted on earlier folds
    only, it is learnt as positive and then costs in folds 3 and 4, so its gain is negative with
    its upper bound below 0; a fit on every fold would set its weight near 0 and give a gain of 0
    or more. The first fold is scored by no variant."""
    rng, frame, codes = world(9840, meetings_per_fold=25)
    n = len(frame)
    sign = np.where(frame["fold"].isin(FOLDS[:2]).to_numpy(), 1.0, -1.0)
    p_flip = log_softmax(rng.normal(size=n), codes)
    p_steady = log_softmax(rng.normal(size=n), codes)
    won = draw_winners(rng, frame[Q].to_numpy() + sign * p_flip + 0.5 * p_steady, codes)
    frame = frame.assign(won=won, p_flip=p_flip, p_steady=p_steady)

    t = by_variant(
        marginal_ablation(
            frame,
            ["p_flip", "p_steady"],
            clusters=meetings_of(frame),
            n_boot=500,
            level=0.95,
            seed=98,
        )
    )
    assert (t["races"] == scored_races(frame)).all()
    assert t.loc["all minus p_flip", "gain_hi"] < 0
    assert t.loc["all minus p_flip", "delta_ll_lo"] > 0  # p_steady alone still beats the market


def test_ablation_uses_the_same_bootstrap_indices_for_every_variant(ablation_world):
    """DoD: the marginal table uses the same bootstrap indices for every variant. The draws depend
    on the seed and the meetings alone, never on a variant's place in the list or its name: the
    same members in another order give every row the same bounds, and the variant {p_form,
    p_speed} has the same bounds whether it is "all minus a_noise" in one table or "all minus
    z_noise" in another. The bounds do come from resampling (another seed moves them and leaves
    the means), and each variant has its own."""
    clusters = meetings_of(ablation_world)
    kw = {"clusters": clusters, "n_boot": 400, "level": 0.95, "seed": 98}
    first = by_variant(marginal_ablation(ablation_world, ["p_form", "p_speed", "a_noise"], **kw))
    reordered = by_variant(
        marginal_ablation(ablation_world, ["a_noise", "p_form", "p_speed"], **kw)
    )
    swapped = by_variant(marginal_ablation(ablation_world, ["p_form", "p_speed", "z_noise"], **kw))
    reseeded = by_variant(
        marginal_ablation(ablation_world, ["p_form", "p_speed", "a_noise"], **{**kw, "seed": 99})
    )

    assert set(reordered.index) == set(first.index)
    for v in first.index:
        np.testing.assert_allclose(
            reordered.loc[v, INTERVALS].to_numpy(dtype=float),
            first.loc[v, INTERVALS].to_numpy(dtype=float),
            atol=1e-8,
            err_msg=v,
        )
    delta = ["delta_ll", "delta_ll_lo", "delta_ll_hi"]
    np.testing.assert_allclose(
        swapped.loc["all minus z_noise", delta].to_numpy(dtype=float),
        first.loc["all minus a_noise", delta].to_numpy(dtype=float),
        atol=1e-8,
    )

    np.testing.assert_allclose(reseeded["delta_ll"], first["delta_ll"], atol=1e-10)
    assert abs(reseeded.loc["all", "delta_ll_lo"] - first.loc["all", "delta_ll_lo"]) > 1e-6
    assert first["delta_ll_lo"].round(10).nunique() == len(first)


def test_a_pure_noise_member_has_a_marginal_gain_interval_covering_zero(ablation_world):
    """p_noise is drawn after the winners: out of fold its marginal gain's bounds at D7's 95%
    level straddle 0, while p_speed's, a real signal, sit above it. (Out of fold a noise member's
    expected gain is slightly negative, the cost of estimating its weight; on these races 0 sits
    near the middle of its bounds.)"""
    t = by_variant(
        marginal_ablation(
            ablation_world,
            ["p_form", "p_speed", "p_noise"],
            clusters=meetings_of(ablation_world),
            n_boot=1_000,
            level=0.95,
            seed=98,
        )
    )
    noise = t.loc["all minus p_noise"]
    assert noise["gain_lo"] <= 0.0 <= noise["gain_hi"], noise.to_dict()
    assert t.loc["all minus p_speed", "gain_lo"] > 0.0


# ---------------------------------------------------------------------------------------------
# Cross-cutting: the Heskes ambiguity term and member correlations
# ---------------------------------------------------------------------------------------------


def reference_ambiguity(frame, members):
    """Per race: members normalised, p_bar the normalised geometric mean, the mean over members of
    KL(p_bar || p_k); then the mean over races."""
    terms = []
    for _, race in frame.groupby("race_id", sort=False):
        logs = []
        for k in members:
            v = race[k].to_numpy(dtype=float)
            logs.append(v - v.max() - np.log(np.exp(v - v.max()).sum()))
        g = np.mean(logs, axis=0)
        p_bar = np.exp(g) / np.exp(g).sum()
        kl = [float(np.sum(p_bar * (np.log(p_bar) - lp))) for lp in logs]
        terms.append(np.mean(kl))
    return float(np.mean(terms))


def ambiguity_races(seed, n_races=40):
    rng = np.random.default_rng(seed)
    codes = runners(rng, n_races, sizes=(2, 16))
    n = len(codes)
    shared = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "race_id": np.array([f"jra:a{c:03d}" for c in range(n_races)], dtype=object)[codes],
            "m1": log_softmax(shared, codes),
            "m2": log_softmax(shared + 0.5 * rng.normal(size=n), codes),
            "m3": log_softmax(rng.normal(size=n), codes),
        }
    )
    return frame, codes


def test_ambiguity_is_zero_when_members_agree():
    """Members that agree have no ambiguity: identical columns, or the same distributions written
    with any per-race constant added, give 0, for two members or three. Members that disagree
    give a positive term."""
    frame, codes = ambiguity_races(9850)
    offset = np.random.default_rng(9851).normal(0, 3, size=codes.max() + 1)[codes]
    same = frame.assign(m1_copy=frame["m1"], m1_shifted=frame["m1"] + offset)
    assert ambiguity(same, ["m1", "m1_copy"]) == pytest.approx(0.0, abs=1e-12)
    assert ambiguity(same, ["m1", "m1_shifted"]) == pytest.approx(0.0, abs=1e-10)
    assert ambiguity(same, ["m1", "m1_copy", "m1_shifted"]) == pytest.approx(0.0, abs=1e-10)
    assert ambiguity(frame, ["m1", "m2"]) > 0.01
    assert ambiguity(frame, ["m1", "m3"]) > ambiguity(frame, ["m1", "m2"])


def test_ambiguity_is_the_mean_kl_from_the_combined_distribution_to_each_member():
    """Two members of one race, (0.5, 0.5) and (0.9, 0.1): p_bar is proportional to
    (sqrt .45, sqrt .05) and the term is -ln(sqrt .45 + sqrt .05). Many races of 2 to 16 runners
    and two or three members: the per-race mean over members of KL(p_bar || p_k), averaged over
    races (each race once, whatever its size), unchanged by a per-race constant on a member."""
    one = pd.DataFrame({"race_id": ["jra:x"] * 2, "a": np.log([0.5, 0.5]), "b": np.log([0.9, 0.1])})
    hand = -math.log(math.sqrt(0.45) + math.sqrt(0.05))
    assert ambiguity(one, ["a", "b"]) == pytest.approx(hand, rel=1e-10)

    frame, codes = ambiguity_races(9852)
    for members in (["m1", "m2"], ["m1", "m3"], ["m1", "m2", "m3"]):
        assert ambiguity(frame, members) == pytest.approx(
            reference_ambiguity(frame, members), rel=1e-9
        )
    shifted = frame.assign(m3=frame["m3"] + np.arange(codes.max() + 1)[codes] * 0.7)
    assert ambiguity(shifted, ["m1", "m3"]) == pytest.approx(
        reference_ambiguity(frame, ["m1", "m3"]), rel=1e-9
    )


def test_member_correlations_are_of_logp_demeaned_within_race():
    """The members' Pearson correlations after removing each race's mean of each member, as a
    matrix on the members in the order given: a per-race constant on a member (as a
    log-probability's normaliser is) leaves it unchanged, though it moves the raw correlation."""
    rng = np.random.default_rng(9860)
    codes = runners(rng, 300, sizes=(4, 16))
    n = len(codes)
    s1, s2, s3 = rng.normal(size=(3, n))
    race_level = rng.normal(0, 2.0, size=(3, codes.max() + 1))[:, codes]
    frame = pd.DataFrame(
        {
            "race_id": np.array([f"jra:c{c:04d}" for c in range(codes.max() + 1)], dtype=object)[
                codes
            ],
            "p_speed": s1 + race_level[0],
            "p_rating": 0.8 * s1 + 0.6 * s2 + race_level[1],
            "p_ability": s3 - 0.3 * s1 + race_level[2],
        }
    )
    members = ["p_speed", "p_rating", "p_ability"]
    demeaned = np.column_stack(
        [
            frame[k].to_numpy() - frame.groupby("race_id")[k].transform("mean").to_numpy()
            for k in members
        ]
    )
    expected = np.corrcoef(demeaned, rowvar=False)

    for order in (members, members[::-1]):
        got = member_correlations(frame, order)
        assert list(got.index) == order and list(got.columns) == order
        idx = [members.index(k) for k in order]
        np.testing.assert_allclose(
            got.to_numpy(dtype=float), expected[np.ix_(idx, idx)], atol=1e-10
        )

    raw = np.corrcoef(frame[members].to_numpy(), rowvar=False)
    assert np.abs(raw - expected).max() > 0.1
    shifted = frame.assign(p_rating=frame["p_rating"] + rng.normal(0, 5.0, codes.max() + 1)[codes])
    np.testing.assert_allclose(
        member_correlations(shifted, members).to_numpy(dtype=float), expected, atol=1e-10
    )
