"""The tail beta beside the crowd-calibration baseline (ticket #285; owner's answer (c) on #77).

Contract pinned here (`horsetrack.countries.japan.studies.tail_beta`, a sibling of #77's
`studies.calibration`, which stays untouched):
- `fit_tail_beta(odds, race_id, won, *, min_odds=None) -> TailBetaFit`: the conditional-logit MLE of
  P(i wins) proportional to exp(beta * x_i + increment * x_i * 1[O_i > min_odds]) over each race's
  priced runners, x = ln(1 / O). The tail is strictly above `min_odds`; None reads
  `tail_min_odds` from config/calibration.yaml (51.0). Inputs are pandas Series on one index, as in
  `calibration.fit_power_beta`; odds go through `core.pricing.clean_odds` (a missing, non-finite
  or sub-1.0 figure is an unpriced runner and leaves its race's choice set), only races with a
  priced winner enter, a dead heat counts each priced co-winner.
- `TailBetaFit` carries `beta` (the body slope) and `se`, `increment` (the added slope above
  min_odds) and `increment_se`, `tail_beta` (= beta + increment, the slope a tail runner gets) and
  `tail_beta_se` (from the full covariance, var b + var t + 2 cov), and `n_races`. Every standard
  error is the square root of the diagonal of the inverse observed information at the MLE.
- No runner above `min_odds` in any fitted race: the increment is not identified, ValueError.
- `build_spec(owner, scheme, bootstrap, floor) -> dict` (the signature of
  `calibration.build_spec`) and `FAMILY_ID`: a new family, not #77's, whose plain-JSON spec is
  frozen in the run registry before any scoring; the tail threshold, the fold scheme and the
  bootstrap are in it.
- `compute_study(frame, spec) -> calibration.Study`, pure, on `development_frame`'s shape
  (race_id, race_date datetime64, meeting_id, odds, won): every variant a `Variant` (one registry
  record each, unique names, none failed on a healthy frame), with winner-log-loss metrics scored
  out of fold on the spec's walk-forward folds, "ΔLL" meaning the comparator's loss minus the
  tail model's (positive: the tail model is better) with meeting-clustered bounds, a
  `delta_ll_vs_normalised` and a `delta_ll_vs_power` metric per subset, metrics named
  `tail_beta...` and `...increment...` (the tail slope and its increment over the body slope) with
  95% Wald CIs per subset and per year (the year in the metric's name). Subset names appear in
  variant names: `all_races` and `no_floor_bound_races` (races holding a runner at or under the
  floor, odds 1.1, left out of fitting and scoring alike), as in #77.

Expected values come from an independent reference below (a two-parameter Newton solver and the
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

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.folds import FoldScheme, walk_forward
from horsetrack.core.eval.registry import default_registry
from horsetrack.countries.japan.studies import calibration
from horsetrack.countries.japan.studies.tail_beta import (
    FAMILY_ID,
    TailBetaFit,
    build_spec,
    compute_study,
    fit_tail_beta,
)

NAN = float("nan")
TAKE = 0.8
THRESHOLD = 51.0
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


def draw_winners(rng, odds, body, increment, min_odds):
    """Winners drawn from p proportional to exp(body * x + increment * x * 1[odds > min_odds])."""
    x = -np.log(odds)
    z = body * x + increment * x * (odds > min_odds)
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


def planted(seed, n_races, body, increment, *, n_max=16, n_min=8, spread=1.4, min_odds=THRESHOLD):
    rng = np.random.default_rng(seed)
    odds = market_odds(rng, n_races, n_max, n_min, spread)
    winners = draw_winners(rng, odds, body, increment, min_odds)
    return long_rows(odds, winners).drop(columns="k"), rng


def shuffled(frame, rng):
    """Rows shuffled across races, on a shuffled, non-contiguous integer index."""
    frame = frame.iloc[rng.permutation(len(frame))].copy()
    frame.index = pd.Index(7 * rng.permutation(len(frame)) + 101)
    return frame


def design(frame, min_odds):
    """Runner rows of races with a priced winner, sorted by race: X = [x, x * tail], won, codes."""
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
    return np.column_stack([x, x * (o > min_odds)]), won, starts, winners, o


def loglik_score_info(frame, theta, min_odds):
    X, won, starts, winners, _ = design(frame, min_odds)
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


def ref_fit(frame, min_odds=THRESHOLD):
    """The two-slope conditional-logit MLE by damped Newton: (theta, covariance, n_races)."""
    theta = np.array([1.0, 0.0])
    ll, score, info, n = loglik_score_info(frame, theta, min_odds)
    for _ in range(100):
        step = np.linalg.solve(info, score)
        while True:
            trial = loglik_score_info(frame, theta + step, min_odds)
            if trial[0] >= ll or np.abs(step).max() < 1e-12:
                break
            step = step / 2.0
        theta = theta + step
        ll, score, info, n = trial
        if np.abs(step).max() < 1e-10:
            break
    return theta, np.linalg.inv(info), n


def assert_is_the_mle(fit, frame, min_odds=THRESHOLD):
    """The fit sits within a Newton step of 1e-4 of the root of the score, its standard errors
    are the inverse information's, and the derived fields agree."""
    theta = np.array([fit.beta, fit.increment])
    _, score, info, n = loglik_score_info(frame, theta, min_odds)
    assert np.abs(np.linalg.solve(info, score)).max() < 1e-4
    cov = np.linalg.inv(info)
    assert fit.n_races == n
    assert fit.se == pytest.approx(math.sqrt(cov[0, 0]), rel=1e-3)
    assert fit.increment_se == pytest.approx(math.sqrt(cov[1, 1]), rel=1e-3)
    assert fit.tail_beta == pytest.approx(fit.beta + fit.increment, abs=1e-12)
    both = cov[0, 0] + cov[1, 1] + 2 * cov[0, 1]
    assert fit.tail_beta_se == pytest.approx(math.sqrt(both), rel=1e-3)


def fit_of(frame, **kwargs):
    return fit_tail_beta(frame["odds"], frame["race_id"], frame["won"], **kwargs)


# ---------------------------------------------------------------------------------------------
# The two-slope fit
# ---------------------------------------------------------------------------------------------


def test_the_market_has_a_real_tail():
    """Guard on the fixtures: enough runners above 51.0, and some win, for the fit to bite."""
    frame, _ = planted(2851, N_RACES, 1.0, 0.3)
    above = frame["odds"] > THRESHOLD
    assert above.mean() > 0.08
    assert frame.loc[above, "won"].sum() > 60


@pytest.mark.parametrize(
    ("seed", "body", "increment"),
    [(2851, 1.0, 0.3), (2852, 0.9, -0.25), (2853, 1.15, 0.6)],
)
def test_recovers_each_planted_pair_of_slopes(seed, body, increment):
    frame, rng = planted(seed, N_RACES, body, increment)
    frame = shuffled(frame, rng)

    fit = fit_of(frame, min_odds=THRESHOLD)

    assert isinstance(fit, TailBetaFit)
    assert fit.n_races == N_RACES
    assert 0 < fit.se < 0.06
    assert 0 < fit.increment_se < 0.25
    assert abs(fit.beta - body) < 3 * fit.se
    assert abs(fit.increment - increment) < 3 * fit.increment_se
    assert abs(fit.tail_beta - (body + increment)) < 3 * fit.tail_beta_se
    assert_is_the_mle(fit, frame)


@pytest.mark.parametrize("seed", [2854, 2855, 2856])
def test_a_one_slope_market_gives_a_tail_increment_whose_ci_covers_zero(seed):
    body = 1.1
    frame, rng = planted(seed, N_RACES, body, 0.0)
    frame = shuffled(frame, rng)

    fit = fit_of(frame, min_odds=THRESHOLD)

    assert 0 < fit.increment_se < 0.25  # an informative interval, not an unbounded one
    assert abs(fit.increment) <= WALD_Z * fit.increment_se  # the 95% CI covers 0
    assert abs(fit.beta - body) < 3 * fit.se
    assert abs(fit.tail_beta - body) < 3 * fit.tail_beta_se
    assert_is_the_mle(fit, frame)


def test_a_planted_tail_is_not_covered_by_a_zero_increment():
    """The increment moves with the data: a strong planted increment is far from 0."""
    frame, _ = planted(2857, N_RACES, 1.0, 0.6)
    fit = fit_of(frame, min_odds=THRESHOLD)
    assert fit.increment > 3 * fit.increment_se


def test_the_tail_is_strictly_above_the_threshold():
    """Many runners are priced exactly 51.0: they belong to the body, not the tail."""
    rng = np.random.default_rng(2858)
    odds = market_odds(rng, 5_000, 16, 8, 1.4)
    odds[(odds >= 45.0) & (odds <= 58.0)] = THRESHOLD
    assert (odds == THRESHOLD).sum() > 300
    winners = draw_winners(rng, odds, 1.0, 0.3, THRESHOLD)
    frame = shuffled(long_rows(odds, winners).drop(columns="k"), rng)

    fit = fit_of(frame, min_odds=THRESHOLD)

    assert_is_the_mle(fit, frame, THRESHOLD)  # the reference tail is odds > 51.0
    # A threshold just under 51.0 puts those runners in the tail: another fit.
    lower = fit_of(frame, min_odds=50.99)
    assert_is_the_mle(lower, frame, 50.99)
    assert abs(lower.increment - fit.increment) > 0.02


def test_the_threshold_is_an_argument_and_defaults_to_the_owners_tail_min_odds():
    owner = load_country_yaml("japan", "calibration")["tail_min_odds"]
    assert owner == THRESHOLD
    frame, _ = planted(2859, 4_000, 1.0, 0.3)

    default = fit_of(frame)
    explicit = fit_of(frame, min_odds=owner)
    lower = fit_of(frame, min_odds=20.0)

    for name in ("beta", "se", "increment", "increment_se", "tail_beta", "tail_beta_se"):
        assert getattr(default, name) == pytest.approx(getattr(explicit, name), rel=1e-9), name
    assert default.n_races == explicit.n_races == 4_000
    assert_is_the_mle(lower, frame, 20.0)
    assert abs(lower.increment - explicit.increment) > 0.02  # the threshold changes the fit


def test_unpriced_scratched_and_winnerless_races_leave_the_fit():
    frame, rng = planted(2860, 1_500, 1.05, 0.3, n_min=5)
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
    assert fit.increment == pytest.approx(base.increment, abs=1e-6)
    assert fit.increment_se == pytest.approx(base.increment_se, rel=1e-6)
    assert_is_the_mle(fit, dirty)
    # Row order and the index labels do not matter.
    again = fit_of(shuffled(dirty, rng), min_odds=THRESHOLD)
    assert again.beta == pytest.approx(fit.beta, abs=1e-6)
    assert again.increment == pytest.approx(fit.increment, abs=1e-6)


def test_a_dead_heat_counts_each_priced_co_winner():
    frame, rng = planted(2861, 2_000, 1.0, 0.3)
    races = frame["race_id"].drop_duplicates().iloc[::10]
    extra = []
    for race_id in races:
        part = frame[frame["race_id"] == race_id]
        loser = part.index[~part["won"].to_numpy()][-1]
        extra.append(loser)
    frame.loc[extra, "won"] = True
    assert frame.groupby("race_id")["won"].sum().max() == 2

    fit = fit_of(shuffled(frame, rng), min_odds=THRESHOLD)

    assert fit.n_races == 2_000
    assert_is_the_mle(fit, frame)  # the reference adds one choice per co-winner


def test_without_a_runner_above_the_threshold_the_increment_is_refused():
    frame, _ = planted(2862, 500, 1.0, 0.0, spread=0.4)  # a flat market: nothing above 51.0
    assert (frame["odds"] <= THRESHOLD).all()
    with pytest.raises(ValueError):
        fit_of(frame, min_odds=THRESHOLD)
    # The same rows do have a tail at a lower threshold.
    assert (frame["odds"] > 8.0).any()
    assert fit_of(frame, min_odds=8.0).n_races == 500


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
    "seed": 285,
}
FLOOR = 1.1


def owner_inputs(**override):
    return {**calibration.owner_inputs(), **override}


def make_spec(**override):
    return build_spec(owner_inputs(**override), SCHEME, BOOTSTRAP, FLOOR)


def study_frame(seed=2870, n_races=2_400):
    """Development-frame-shaped rows: 12 races a race day, two venues, from 2020-01 to 2021-05,
    a floor-bound favourite (1.1) in one race in twelve, a planted tail increment of 0.4."""
    rng = np.random.default_rng(seed)
    odds = market_odds(rng, n_races, 14, 8, 1.4, floor_every=12)
    winners = draw_winners(rng, odds, 1.05, 0.4, THRESHOLD)
    frame = long_rows(odds, winners)
    day = frame["k"].to_numpy() // 12
    race_date = pd.Timestamp("2020-01-02") + pd.to_timedelta(day * 2, unit="D")
    venue = np.where((frame["k"].to_numpy() // 6) % 2 == 0, "TOK", "KYO")
    frame["race_date"] = race_date
    frame["meeting_id"] = [f"{d:%Y-%m-%d}:{v}" for d, v in zip(race_date, venue, strict=True)]
    return frame[["race_id", "race_date", "meeting_id", "odds", "won"]]


def test_the_family_is_new_and_the_spec_is_frozen_plain_json(tmp_path, monkeypatch):
    spec = make_spec()
    assert spec["family_id"] == FAMILY_ID
    assert isinstance(FAMILY_ID, str) and FAMILY_ID
    assert FAMILY_ID != calibration.FAMILY_ID
    json.dumps(spec)  # plain JSON, as the registry requires

    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    registry = default_registry("japan")
    digest = registry.freeze_family(FAMILY_ID, spec)
    assert registry.freeze_family(FAMILY_ID, make_spec()) == digest  # an equal spec: a no-op
    assert registry.records(FAMILY_ID) == []  # frozen with no scored run

    # Everything pre-registered is in the spec: another threshold, scheme or bootstrap is a
    # different family, which the registry refuses under this id.
    other_tail = make_spec(tail_min_odds=40.0)
    other_scheme = build_spec(owner_inputs(), replace(SCHEME, min_train=3), BOOTSTRAP, FLOOR)
    other_boot = build_spec(owner_inputs(), SCHEME, {**BOOTSTRAP, "n_boot": 300}, FLOOR)
    for other in (other_tail, other_scheme, other_boot):
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


def softmax_by_race(z, race_id):
    z = pd.Series(np.asarray(z, dtype=float))
    e = np.exp(z - z.groupby(race_id.to_numpy()).transform("max"))
    return (e / e.groupby(race_id.to_numpy()).transform("sum")).to_numpy()


def reference_scores(rows, min_odds=THRESHOLD):
    """Out-of-fold per-race winner log loss of normalised, power and the tail model on `rows`,
    each fitted on its fold's training rows: {name: Series}, and the fold fits."""
    folds = walk_forward(rows["race_date"], scheme=SCHEME)
    losses = {"normalised": [], "power": [], "tail": []}
    fits = []
    for fold in folds:
        train, test = rows.iloc[fold.train_idx], rows.iloc[fold.test_idx]
        theta, cov, _ = ref_fit(train, min_odds)
        fits.append((theta, cov))
        x = -np.log(test["odds"].to_numpy())
        z_tail = theta[0] * x + theta[1] * x * (test["odds"].to_numpy() > min_odds)
        power = calibration.fit_power_beta(train["odds"], train["race_id"], train["won"]).beta
        probs = {
            "tail": softmax_by_race(z_tail, test["race_id"]),
            "power": softmax_by_race(power * x, test["race_id"]),
            "normalised": softmax_by_race(x, test["race_id"]),
        }
        for name, p in probs.items():
            winner = test["won"].to_numpy(dtype=bool)
            losses[name].append(
                pd.Series(-np.log(p[winner]), index=test["race_id"].to_numpy()[winner])
            )
    return {k: pd.concat(v) for k, v in losses.items()}, fits


@pytest.mark.parametrize("subset", ["all_races", "no_floor_bound_races"])
def test_delta_ll_is_scored_out_of_fold_against_normalised_and_power_with_cluster_bounds(
    frame, study, subset
):
    rows = subset_rows(frame, subset)
    losses, _ = reference_scores(rows)
    metrics = metrics_of(study, subset)
    n_scored = len(losses["tail"])
    assert 500 < n_scored < rows["race_id"].nunique()  # the early quarters only train

    for comparator in ("normalised", "power"):
        found = [m for m in metrics if m.metric.startswith(f"delta_ll_vs_{comparator}")]
        overall = [m for m in found if m.n == n_scored]
        assert overall, (subset, comparator, [(m.metric, m.n) for m in found])
        m = overall[0]
        expected = float((losses[comparator] - losses["tail"]).mean())
        assert m.value == pytest.approx(expected, abs=1e-6), (subset, comparator)
        assert m.lo is not None and m.hi is not None
        assert math.isfinite(m.lo) and math.isfinite(m.hi)
        assert m.lo <= m.value <= m.hi
        assert m.lo < m.hi  # a real bootstrap spread, not a point

    # Normalised and power lose differently, so the two comparisons are different numbers.
    assert float((losses["normalised"] - losses["power"]).abs().max()) > 0


def test_the_floor_bound_subset_leaves_whole_races_out_of_fitting_and_scoring(frame, study):
    all_races, no_floor = (subset_rows(frame, s) for s in ("all_races", "no_floor_bound_races"))
    n_all, n_no = all_races["race_id"].nunique(), no_floor["race_id"].nunique()
    assert 0 < n_all - n_no
    assert (no_floor["odds"] > FLOOR + 1e-9).all()

    def scored(subset):
        found = [
            m for m in metrics_of(study, subset) if m.metric.startswith("delta_ll_vs_normalised")
        ]
        return max(m.n for m in found)

    losses_all, _ = reference_scores(all_races)
    losses_no, _ = reference_scores(no_floor)
    assert scored("all_races") == len(losses_all["tail"])
    assert scored("no_floor_bound_races") == len(losses_no["tail"])
    assert scored("no_floor_bound_races") < scored("all_races")


def test_the_tail_beta_and_its_increment_are_reported_with_cis_by_subset_and_year(frame, study):
    full = frame.reset_index(drop=True)
    theta, cov, _ = ref_fit(full)
    increment_se, body_se = math.sqrt(cov[1, 1]), math.sqrt(cov[0, 0])
    tail_se = math.sqrt(cov[0, 0] + cov[1, 1] + 2 * cov[0, 1])
    assert abs(theta[1] - 0.4) < 3 * increment_se  # the planted tail is recoverable here

    def is_increment(m):
        return "increment" in m.metric and not m.metric.startswith("delta")

    def is_tail_beta(m):
        return "tail_beta" in m.metric

    for subset in ("all_races", "no_floor_bound_races"):
        metrics = metrics_of(study, subset)
        for pick in (is_increment, is_tail_beta):
            found = [m for m in metrics if pick(m) and m.lo is not None and m.hi is not None]
            assert found, (subset, pick.__name__)
            for m in found:
                assert math.isfinite(m.value) and math.isfinite(m.lo) and math.isfinite(m.hi)
                assert m.lo < m.value < m.hi
        # By year: a CI per year in each subset, the year in the metric's name.
        for year in (2020, 2021):
            for pick in (is_increment, is_tail_beta):
                assert any(
                    pick(m) and str(year) in m.metric and m.lo is not None for m in metrics
                ), (subset, year, pick.__name__)

    # The numbers are the fit's: a metric equals the reference on the whole development sample
    # or on some fold's training rows, with Wald CIs beta +/- 1.96 se.
    rows = subset_rows(frame, "all_races")
    _, fold_fits = reference_scores(rows)
    candidates = [(theta[1], increment_se), (theta[0] + theta[1], tail_se)]
    candidates += [(t[1], math.sqrt(c[1, 1])) for t, c in fold_fits]
    candidates += [(t[0] + t[1], math.sqrt(c[0, 0] + c[1, 1] + 2 * c[0, 1])) for t, c in fold_fits]
    for pick in (is_increment, is_tail_beta):
        matches = []
        for m in metrics_of(study, "all_races"):
            if pick(m) and m.lo is not None:
                matches += [
                    (v, se)
                    for v, se in candidates
                    if abs(m.value - v) < 1e-5
                    and abs(m.lo - (v - WALD_Z * se)) < 1e-4
                    and abs(m.hi - (v + WALD_Z * se)) < 1e-4
                ]
        assert matches, pick.__name__
    assert body_se > 0


def test_every_variant_is_one_scored_record_and_none_failed(study):
    names = [(v.hypothesis_id, v.name) for v in study.variants]
    assert len(names) == len(set(names))  # one record per variant
    assert any("all_races" in n for _, n in names)
    assert any("no_floor_bound_races" in n for _, n in names)
    for v in study.variants:
        assert v.error is None, (v.name, v.error)
        assert v.metrics, v.name
        assert v.fold_scheme
        assert v.components
    # Out-of-fold scoring runs on the spec's folds: the fold scheme is in a record's identity.
    scored = [v for v in study.variants if any(m.metric.startswith("delta_ll") for m in v.metrics)]
    assert scored
    assert all(
        json.dumps(dataclasses.asdict(SCHEME), sort_keys=True) in v.fold_scheme for v in scored
    )
