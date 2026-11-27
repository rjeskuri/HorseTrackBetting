"""The entropy gradient in beta, conditioned on the far tail (ticket #277; follows #77's E3).

#77's E3 bins each race by its normalised market entropy and fits one power beta per bin: beta falls
from bin 1 (the most concentrated markets) to bin 5. The tail's over-pricing (#285) says a race's
tail count could produce that gradient alone, because the number of runners above `tail_min_odds`
moves with the entropy. This study separates the two.

Contract pinned here (`horsetrack.countries.japan.studies.entropy_tail`, a sibling of
`studies.calibration` and `studies.tail_beta`, both untouched):

- `METHODS = ("unconditioned", "tail_share", "tail_count_strata")`.
- `fit_gradient(odds, race_id, won, *, cut_points, method, min_odds=None, strata_edges=None,
  min_cell_races=MIN_CELL_RACES) -> GradientFit`; inputs are pandas Series on one index. Rows enter
  as in `calibration.fit_power_beta` (odds through `core.pricing.clean_odds`: a missing, non-finite
  or sub-1.0 figure is an unpriced runner and leaves its race's choice set; only races with a
  priced winner; a dead heat counts each priced co-winner) and a race also needs two or more priced
  runners (its normalised entropy is defined). Its entropy bin is
  `numpy.searchsorted(cut_points, entropy, side="right")` (0..4, "bin 1" is 0), `cut_points` four
  increasing numbers in (0, 1), else ValueError, as are an unknown method and a frame with no race
  left. `min_odds=None` reads `tail_min_odds` from config/calibration.yaml (51.0). The model is
  P(i wins) proportional to exp(slope_r * x_i), x = ln(1/O), over the race's priced runners, with
  - `unconditioned`: slope_r = beta_b (b the race's bin), one fit per bin, as #77's E3.
    delta = beta_1 - beta_5, se = sqrt(se_1^2 + se_5^2).
  - `tail_share`: slope_r = beta_b + gamma * share_r, one joint conditional logit over all five
    bins, share_r = (priced runners above `min_odds`, strictly) / (priced runners), not centred;
    `bin_betas` are the beta_b, `gamma` and `gamma_se` its coefficient. delta = beta_1 - beta_5,
    se from the full inverse observed information.
  - `tail_count_strata`: the tail count c_r (priced runners strictly above `min_odds`) cuts races
    into strata [edges[k], edges[k+1]) with the last open-ended (`strata_edges`: ints, first 0,
    strictly increasing; None reads `TAIL_COUNT_STRATA`). Within each stratum bin 1 and bin 5 get
    their own one-slope fit; a stratum is used when both cells hold at least `min_cell_races`
    races and both fits converge. delta is the inverse-variance pooled (beta_1 - beta_5) over the
    used strata, se = (sum of 1/v_s)^(-1/2) with v_s = se_1s^2 + se_5s^2; `strata_used` lists the
    used strata's lower edges. No stratum usable: ValueError.
  `GradientFit` carries `method`, `delta`, `se`, `lo` and `hi` (delta -/+ 1.959963984540054 se),
  `n_races` (the races fitted: every binned race for the first two methods, the races of the bin 1
  and bin 5 cells of the used strata for the third), `bin_betas` (five floats; () for the strata
  method), `gamma`, `gamma_se` (None unless `tail_share`) and `strata_used` (() unless strata).
- `FAMILY_ID` (a new family, not #77's and not #285's), `TAIL_COUNT_STRATA` and `MIN_CELL_RACES`
  (module constants), and `build_spec(owner, scheme, bootstrap, floor) -> dict`, plain JSON, frozen
  in the run registry before any scoring; it carries `family_id`, `ticket` (277),
  `entropy_cut_points` and `tail["min_odds"]` from the owner inputs, `tail_count_strata` (the
  strata edges) and `min_cell_races`.
- `compute_study(frame, spec) -> calibration.Study`, pure, on `development_frame`'s shape: one
  `Variant` per method named `entropy_gradient:<method>`, hypothesis id "E3", in-sample on
  development rows, none failed on a healthy frame, whose metrics hold, on fold "development",
  `delta_beta_bin1_minus_bin5` with its 95% Wald bounds and n; `tables["gradient"]` has one row per
  method in `METHODS` order with a "method" key. It uses the spec's strata edges and cell size.
- `run_study(out_dir=None) -> dict`: builds the spec from the owner inputs, freezes the family in
  the run registry, only then reads the development rows (through `calibration.store_builds`,
  `calibration.read_development` and `calibration.development_frame`; the holdout is never
  opened), scores, appends one registry record per variant and writes the report with
  `write_report` under `out_dir`. It reads the owner inputs through `calibration.owner_inputs()`
  and does not recompute the cut points (#77's own run re-verifies them). The result carries
  `family_id`, `spec_hash`, `report_path`, `records_appended`, `failed_variants` and
  `survives_conditioning`: {"tail_share": bool, "tail_count_strata": bool}, True when that method's
  95% CI of delta excludes 0.

Expected values come from an independent reference below (a general conditional-logit Newton solver
on a design matrix, the entropy, bins, share and strata computed from first principles), never from
the code under test. Every frame is synthetic and every RNG seeded: no real data, no holdout.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import functools
import json
import math

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval import holdout
from horsetrack.core.eval.registry import default_registry
from horsetrack.countries.japan.studies import calibration, tail_beta
from horsetrack.countries.japan.studies.entropy_tail import (
    FAMILY_ID,
    METHODS,
    MIN_CELL_RACES,
    TAIL_COUNT_STRATA,
    GradientFit,
    build_spec,
    compute_study,
    fit_gradient,
    run_study,
)

NAN = float("nan")
TAKE = 0.8
THRESHOLD = 51.0
FIELD = 14
WALD_Z = 1.959963984540054
CONDITIONED = ("tail_share", "tail_count_strata")
EDGES = [0, 2, 4]  # the strata the scenario tests pin; the spec carries its own default
FINE_EDGES = [0, 1, 2, 3, 4, 5]
GENUINE_BETAS = np.array([1.2, 1.1, 1.0, 0.9, 0.8])  # beta by entropy bin, tail-free
MILD_BETAS = np.array([1.1, 1.05, 1.0, 0.95, 0.9])

# ---------------------------------------------------------------------------------------------
# Synthetic markets, planted winners and the reference
# ---------------------------------------------------------------------------------------------


def markets(rng, n_races, *, smin=0.0, smax=1.4, lam0=1.0, lam1=1.2):
    """(n_races, FIELD) final Win odds rounded to 0.1 (floor 1.0). Each race has a body spread
    sigma (the main driver of its entropy) and c longshots priced 52 to 100 (c rises weakly with
    sigma), so the tail count and the entropy are correlated but not tied."""
    sigma = rng.uniform(smin, smax, size=n_races)
    lam = lam0 + lam1 * (sigma - smin) / (smax - smin)
    count = np.minimum(rng.poisson(lam), 6)
    is_tail = np.arange(FIELD)[None, :] < count[:, None]
    body = np.exp(rng.standard_normal((n_races, FIELD)) * sigma[:, None])
    body = np.where(is_tail, 0.0, body)
    q_tail = rng.uniform(0.008, 0.0155, size=(n_races, FIELD))
    body_mass = 1.0 - (q_tail * is_tail).sum(axis=1)
    q = np.where(is_tail, q_tail, body / body.sum(axis=1, keepdims=True) * body_mass[:, None])
    return np.maximum(np.round(TAKE / q, 1), 1.0)


def entropy_of(odds):
    inv = 1.0 / odds
    q = inv / inv.sum(axis=1, keepdims=True)
    return -(q * np.log(q)).sum(axis=1) / math.log(odds.shape[1])


def draw_winners(rng, odds, slope):
    z = slope[:, None] * -np.log(odds)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    u = rng.random(len(p))[:, None]
    return np.minimum((np.cumsum(p, axis=1) < u).sum(axis=1), p.shape[1] - 1)


@functools.cache
def scenario(kind, seed, n_races):
    """(frame, cut_points) of a planted world, `development_frame`'s shape. The true slope of a
    race is: tail_only 1.1 - 2.0 * its tail share, whatever its entropy; genuine GENUINE_BETAS by
    its entropy bin, whatever its tail; mixed MILD_BETAS by bin minus 1.0 * tail share."""
    rng = np.random.default_rng(seed)
    odds = markets(rng, n_races)
    cuts = [float(c) for c in np.quantile(entropy_of(odds), [0.2, 0.4, 0.6, 0.8])]
    bins = np.searchsorted(cuts, entropy_of(odds), side="right")
    share = (odds > THRESHOLD).sum(axis=1) / FIELD
    slope = {
        "tail_only": 1.1 - 2.0 * share,
        "genuine": GENUINE_BETAS[bins],
        "mixed": MILD_BETAS[bins] - 1.0 * share,
    }[kind]
    winners = draw_winners(rng, odds, slope)
    k = np.repeat(np.arange(n_races), FIELD)
    day = k // 12
    race_date = pd.Timestamp("2020-01-02") + pd.to_timedelta(day, unit="D")
    venue = np.where((k // 6) % 2 == 0, "TOK", "KYO")
    frame = pd.DataFrame(
        {
            "race_id": [f"r{i:05d}" for i in k],
            "race_date": race_date,
            "meeting_id": [f"{d:%Y-%m-%d}:{v}" for d, v in zip(race_date, venue, strict=True)],
            "odds": odds.ravel(),
            "won": (np.arange(FIELD)[None, :] == winners[:, None]).ravel(),
        }
    )
    return frame, cuts


def shuffled(frame, rng):
    frame = frame.iloc[rng.permutation(len(frame))].copy()
    frame.index = pd.Index(7 * rng.permutation(len(frame)) + 101)
    return frame


def reference_rows(frame, cuts, min_odds):
    """Runner rows of the races that enter any fit (two or more priced runners and a priced
    winner), sorted by race, with the race's entropy bin, tail count and tail share."""
    odds = frame["odds"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(odds) & (odds >= 1.0)
    d = frame.loc[ok, ["race_id", "odds", "won"]]
    code = pd.factorize(d["race_id"])[0]
    o = d["odds"].to_numpy(dtype=float)
    won = d["won"].to_numpy(dtype=bool)
    n = np.bincount(code)
    inv = 1.0 / o
    q = inv / np.bincount(code, weights=inv)[code]
    h = np.bincount(code, weights=-q * np.log(q))
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = np.where(n >= 2, np.clip(h / np.log(n), 0.0, 1.0), np.nan)
    count = np.bincount(code, weights=(o > min_odds).astype(float))
    wins = np.bincount(code, weights=won.astype(float))
    race_bin = np.searchsorted(cuts, entropy, side="right")
    keep = ((n >= 2) & (wins > 0))[code]
    order = np.argsort(code[keep], kind="stable")

    def per_runner(values):
        return values[code][keep][order]

    return {
        "x": (-np.log(o))[keep][order],
        "won": won[keep][order],
        "code": code[keep][order],
        "bin": per_runner(race_bin),
        "count": per_runner(count),
        "share": per_runner(count / n),
    }


def conditional_logit(design, code, won):
    """MLE of P(i wins in its race) proportional to exp(design_i . theta) by damped Newton from 0:
    (theta, inverse observed information, races). `code` is sorted; a dead heat counts each
    co-winner."""
    starts = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
    sizes = np.diff(np.r_[starts, len(design)])
    wins = np.add.reduceat(won.astype(float), starts)

    def evaluate(theta):
        s = design @ theta
        top = np.maximum.reduceat(s, starts)
        e = np.exp(s - np.repeat(top, sizes))
        tot = np.add.reduceat(e, starts)
        mean = np.add.reduceat(e[:, None] * design, starts) / tot[:, None]
        second = np.add.reduceat(e[:, None, None] * design[:, :, None] * design[:, None, :], starts)
        cov = second / tot[:, None, None] - mean[:, :, None] * mean[:, None, :]
        ll = float(s[won].sum() - (wins * (np.log(tot) + top)).sum())
        score = design[won].sum(axis=0) - (wins[:, None] * mean).sum(axis=0)
        return ll, score, (wins[:, None, None] * cov).sum(axis=0)

    theta = np.zeros(design.shape[1])
    ll, score, info = evaluate(theta)
    for _ in range(100):
        step = np.linalg.solve(info, score)
        while True:
            trial = evaluate(theta + step)
            if trial[0] >= ll or np.abs(step).max() < 1e-12:
                break
            step = step / 2.0
        theta = theta + step
        ll, score, info = trial
        if np.abs(step).max() < 1e-10:
            break
    return theta, np.linalg.inv(info), len(starts)


def one_slope(r, mask):
    theta, cov, races = conditional_logit(r["x"][mask][:, None], r["code"][mask], r["won"][mask])
    return float(theta[0]), float(cov[0, 0]), races


def reference_gradient(
    frame, cuts, method, *, min_odds=THRESHOLD, edges=None, min_cell=MIN_CELL_RACES
):
    r = reference_rows(frame, cuts, min_odds)
    out = {"gamma": None, "gamma_se": None, "bin_betas": (), "strata_used": ()}
    if method == "unconditioned":
        cells = [one_slope(r, r["bin"] == k) for k in range(5)]
        out |= {"delta": cells[0][0] - cells[4][0], "se": math.sqrt(cells[0][1] + cells[4][1])}
        out |= {"bin_betas": tuple(c[0] for c in cells), "n_races": len(np.unique(r["code"]))}
    elif method == "tail_share":
        columns = [r["x"] * (r["bin"] == k) for k in range(5)] + [r["x"] * r["share"]]
        theta, cov, races = conditional_logit(np.column_stack(columns), r["code"], r["won"])
        out |= {"delta": theta[0] - theta[4], "n_races": races}
        out["se"] = math.sqrt(cov[0, 0] + cov[4, 4] - 2 * cov[0, 4])
        out |= {"bin_betas": tuple(theta[:5]), "gamma": theta[5], "gamma_se": math.sqrt(cov[5, 5])}
    else:
        edges = list(TAIL_COUNT_STRATA if edges is None else edges)
        upper = edges[1:] + [np.inf]
        deltas, variances, used, races = [], [], [], 0
        for lo, hi in zip(edges, upper, strict=True):
            inside = (r["count"] >= lo) & (r["count"] < hi)
            first, last = inside & (r["bin"] == 0), inside & (r["bin"] == 4)
            n_first, n_last = (len(np.unique(r["code"][m])) for m in (first, last))
            if min(n_first, n_last) < min_cell:
                continue
            (b1, v1, _), (b5, v5, _) = one_slope(r, first), one_slope(r, last)
            deltas.append(b1 - b5)
            variances.append(v1 + v5)
            used.append(lo)
            races += n_first + n_last
        weights = 1.0 / np.array(variances)
        out |= {"delta": float((weights * np.array(deltas)).sum() / weights.sum())}
        out |= {"se": float(weights.sum() ** -0.5), "n_races": races, "strata_used": tuple(used)}
    return out


def fit_of(frame, cuts, method, **kwargs):
    return fit_gradient(
        frame["odds"], frame["race_id"], frame["won"], cut_points=cuts, method=method, **kwargs
    )


def assert_matches(fit, ref, *, delta_tol=1e-5):
    assert isinstance(fit, GradientFit)
    assert fit.delta == pytest.approx(ref["delta"], abs=delta_tol)
    assert fit.se == pytest.approx(ref["se"], rel=1e-4)
    assert fit.lo == pytest.approx(fit.delta - WALD_Z * fit.se, abs=1e-12)
    assert fit.hi == pytest.approx(fit.delta + WALD_Z * fit.se, abs=1e-12)
    assert fit.n_races == ref["n_races"]
    assert tuple(fit.strata_used) == ref["strata_used"]
    assert len(fit.bin_betas) == len(ref["bin_betas"])
    for got, want in zip(fit.bin_betas, ref["bin_betas"], strict=True):
        assert got == pytest.approx(want, abs=1e-5)
    if ref["gamma"] is None:
        assert fit.gamma is None and fit.gamma_se is None
    else:
        assert fit.gamma == pytest.approx(ref["gamma"], abs=1e-5)
        assert fit.gamma_se == pytest.approx(ref["gamma_se"], rel=1e-4)


def excludes_zero(fit_or_metric):
    return fit_or_metric.lo > 0 or fit_or_metric.hi < 0


def covers_zero(fit_or_metric):
    return fit_or_metric.lo <= 0 <= fit_or_metric.hi


# ---------------------------------------------------------------------------------------------
# The fit against the reference
# ---------------------------------------------------------------------------------------------


def test_the_fixtures_tie_the_tail_count_to_the_entropy_bin_but_leave_overlap():
    frame, cuts = scenario("tail_only", 2771, 20_000)
    r = reference_rows(frame, cuts, THRESHOLD)
    mean_first = r["count"][r["bin"] == 0].mean()
    mean_last = r["count"][r["bin"] == 4].mean()
    assert abs(mean_first - mean_last) > 2.5  # the tail count moves with the bin
    for k in range(5):  # every bin is a fifth of the races
        assert 0.15 < len(np.unique(r["code"][r["bin"] == k])) / 20_000 < 0.25
    middle = (r["count"] >= 2) & (r["count"] < 4)  # both ends of the entropy scale meet here
    assert all(len(np.unique(r["code"][middle & (r["bin"] == k)])) >= 30 for k in (0, 4))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("kind", "seed"), [("tail_only", 2771), ("genuine", 2772), ("mixed", 2773)]
)
def test_every_method_equals_the_reference_on_each_planted_world(method, kind, seed):
    frame, cuts = scenario(kind, seed, 20_000)
    ref = reference_gradient(frame, cuts, method, edges=EDGES)
    if method == "tail_count_strata":
        assert ref["strata_used"]  # the reference had something to pool
    fit = fit_of(frame, cuts, method, strata_edges=EDGES)
    assert_matches(fit, ref)
    assert fit.method == method
    assert fit.lo < fit.delta < fit.hi


def test_the_three_methods_are_three_different_numbers_on_a_tail_driven_gradient():
    frame, cuts = scenario("tail_only", 2771, 20_000)
    deltas = {m: fit_of(frame, cuts, m, strata_edges=EDGES).delta for m in METHODS}
    assert len({round(v, 3) for v in deltas.values()}) == 3
    assert abs(deltas["unconditioned"]) > 2 * max(abs(deltas[m]) for m in CONDITIONED)


@pytest.mark.parametrize("method", METHODS)
def test_unpriced_void_single_runner_and_dead_heat_rows_follow_calibrations_rules(method):
    frame, cuts = scenario("mixed", 2774, 4_000)
    rng = np.random.default_rng(2775)
    kwargs = {"strata_edges": EDGES, "min_cell_races": 10}
    base = fit_of(frame, cuts, method, **kwargs)

    first = frame.drop_duplicates("race_id").iloc[::5][["race_id"]]
    junk = pd.concat(
        [
            first.assign(odds=NAN, won=False),
            first.iloc[::2].assign(odds=0.0, won=False),
            first.iloc[::3].assign(odds=0.5, won=False),
            first.iloc[::4].assign(odds=float("inf"), won=False),
        ]
    )
    rows = []
    for k, (winner_odds, others) in enumerate(
        [(None, [2.5, 3.1, 7.7, 80.0]), (NAN, [1.9, 4.4, 9.0, 120.0]), (0.5, [2.2, 2.8, 70.0])]
    ):
        rows += [(f"void{k}", o, False) for o in others]
        if winner_odds is not None:
            rows.append((f"void{k}", winner_odds, True))
    for k in range(6):  # one priced runner who wins, one unpriced: no entropy, left out
        rows += [(f"single{k}", 3.0 + k, True), (f"single{k}", NAN, False)]
    extra = pd.DataFrame(rows, columns=["race_id", "odds", "won"])
    dirty = shuffled(pd.concat([frame, junk, extra], ignore_index=True), rng)

    again = fit_of(dirty, cuts, method, **kwargs)
    assert again.n_races == base.n_races
    assert again.delta == pytest.approx(base.delta, abs=1e-6)
    assert again.se == pytest.approx(base.se, rel=1e-6)
    assert tuple(again.strata_used) == tuple(base.strata_used)
    assert_matches(again, reference_gradient(frame, cuts, method, edges=EDGES, min_cell=10))

    # A dead heat counts each priced co-winner, as in calibration.fit_power_beta.
    heated = frame.copy()
    races = heated["race_id"].drop_duplicates().iloc[::10]
    losers = [
        heated.index[(heated["race_id"] == r).to_numpy() & ~heated["won"].to_numpy()][-1]
        for r in races
    ]
    heated.loc[losers, "won"] = True
    assert heated.groupby("race_id")["won"].sum().max() == 2
    ref = reference_gradient(heated, cuts, method, edges=EDGES, min_cell=10)
    assert_matches(fit_of(shuffled(heated, rng), cuts, method, **kwargs), ref)


def test_the_tail_threshold_is_an_argument_and_defaults_to_the_owners_tail_min_odds():
    assert load_country_yaml("japan", "calibration")["tail_min_odds"] == THRESHOLD
    frame, cuts = scenario("tail_only", 2771, 20_000)
    for method in CONDITIONED:
        kwargs = {"strata_edges": EDGES}
        default = fit_of(frame, cuts, method, **kwargs)
        explicit = fit_of(frame, cuts, method, min_odds=THRESHOLD, **kwargs)
        lower = fit_of(frame, cuts, method, min_odds=30.0, **kwargs)
        assert default.delta == pytest.approx(explicit.delta, rel=1e-9)
        assert default.se == pytest.approx(explicit.se, rel=1e-9)
        assert_matches(lower, reference_gradient(frame, cuts, method, min_odds=30.0, edges=EDGES))
        assert abs(lower.delta - default.delta) > 1e-3  # the threshold changes the answer
    # The runners priced exactly at the threshold are in the body, not the tail.
    odds = frame["odds"].to_numpy().copy()
    odds[(odds >= 45.0) & (odds <= 58.0)] = THRESHOLD
    pinned = frame.assign(odds=odds)
    assert (odds == THRESHOLD).sum() > 300
    share = fit_of(pinned, cuts, "tail_share")
    assert_matches(share, reference_gradient(pinned, cuts, "tail_share"))
    just_under = fit_of(pinned, cuts, "tail_share", min_odds=50.99)
    assert abs(just_under.gamma - share.gamma) > 0.05


def test_strata_edges_and_the_minimum_cell_choose_the_strata_used():
    frame, cuts = scenario("tail_only", 2771, 20_000)
    results = {}
    for edges, min_cell in [(EDGES, 30), (FINE_EDGES, 30), (EDGES, 100), ([0, 3], 30)]:
        ref = reference_gradient(frame, cuts, "tail_count_strata", edges=edges, min_cell=min_cell)
        fit = fit_of(frame, cuts, "tail_count_strata", strata_edges=edges, min_cell_races=min_cell)
        assert_matches(fit, ref)
        results[(tuple(edges), min_cell)] = fit
    assert len({f.strata_used for f in results.values()}) >= 3  # the choices are not idle
    assert len({round(f.delta, 4) for f in results.values()}) >= 3
    with pytest.raises(ValueError):  # no stratum holds enough races in both cells
        fit_of(frame, cuts, "tail_count_strata", strata_edges=EDGES, min_cell_races=10**9)


def test_bad_arguments_are_refused():
    frame, cuts = scenario("genuine", 2772, 1_000)
    with pytest.raises(ValueError):
        fit_of(frame, cuts, "median_split")
    for bad in (cuts[:3], cuts[::-1], [0.0, *cuts[1:]], [*cuts[:3], 1.0]):
        with pytest.raises(ValueError):
            fit_of(frame, bad, "unconditioned")
    with pytest.raises(ValueError):  # no race with a priced winner and two priced runners
        fit_gradient(
            pd.Series([3.0, NAN]),
            pd.Series(["a", "a"]),
            pd.Series([True, False]),
            cut_points=cuts,
            method="unconditioned",
        )


# ---------------------------------------------------------------------------------------------
# The question: does the gradient survive conditioning on the tail?
# ---------------------------------------------------------------------------------------------

BIG = 30_000


@pytest.fixture(scope="module")
def tail_only():
    frame, cuts = scenario("tail_only", 2776, BIG)
    fits = {m: fit_of(frame, cuts, m, strata_edges=EDGES) for m in METHODS}
    refs = {m: reference_gradient(frame, cuts, m, edges=EDGES) for m in METHODS}
    return frame, cuts, fits, refs


@pytest.fixture(scope="module")
def genuine():
    frame, cuts = scenario("genuine", 2777, BIG)
    fits = {m: fit_of(frame, cuts, m, strata_edges=EDGES) for m in METHODS}
    return frame, cuts, fits


def test_a_gradient_made_by_the_tail_count_alone_vanishes_after_conditioning(tail_only):
    _, _, fits, refs = tail_only
    for method in METHODS:
        assert_matches(fits[method], refs[method])

    raw = fits["unconditioned"]
    assert excludes_zero(raw) and abs(raw.delta) > 0.4  # the screen sees a gradient
    for method in CONDITIONED:
        fit = fits[method]
        assert covers_zero(fit), (method, fit.delta, fit.lo, fit.hi)
        assert abs(fit.delta) < 0.25 * abs(raw.delta), method
        assert fit.se < 0.2  # an informative interval, not an unbounded one
    # The share model recovers the planted tail slope: 1.1 - 2.0 * share.
    share = fits["tail_share"]
    assert abs(share.gamma - (-2.0)) < 3 * share.gamma_se
    assert share.gamma_se < 0.3
    assert all(abs(b - 1.1) < 0.2 for b in share.bin_betas)  # the bins' own betas are equal


def test_a_genuine_entropy_gradient_survives_conditioning_on_the_tail(genuine):
    _, _, fits = genuine
    planted = GENUINE_BETAS[0] - GENUINE_BETAS[4]
    for method in METHODS:
        fit = fits[method]
        assert fit.lo > 0, (method, fit.delta, fit.lo, fit.hi)  # excludes 0, on the right side
        assert abs(fit.delta - planted) < 3 * fit.se, method
    first, middle, last = (fits["unconditioned"].bin_betas[k] for k in (0, 2, 4))
    assert first > middle > last
    assert abs(fits["tail_share"].gamma) < 3 * fits["tail_share"].gamma_se  # no tail effect


# ---------------------------------------------------------------------------------------------
# The family: spec, freeze, study, run
# ---------------------------------------------------------------------------------------------


def owner_inputs(**override):
    return {**calibration.owner_inputs(), **override}


def make_spec(**override):
    from horsetrack.core.eval.folds import load_country_scheme

    return build_spec(
        owner_inputs(**override),
        load_country_scheme("japan"),
        calibration.bootstrap_settings(),
        calibration.win_floor(),
    )


def test_the_family_is_new_and_its_spec_is_frozen_plain_json(tmp_path, monkeypatch):
    spec = make_spec()
    owner = calibration.owner_inputs()
    assert spec["family_id"] == FAMILY_ID
    assert isinstance(FAMILY_ID, str) and FAMILY_ID
    assert FAMILY_ID not in (calibration.FAMILY_ID, tail_beta.FAMILY_ID)
    assert spec["ticket"] == 277
    json.dumps(spec)  # plain JSON, as the registry requires

    assert spec["entropy_cut_points"] == owner["entropy_cut_points"]
    assert spec["tail"]["min_odds"] == owner["tail_min_odds"] == THRESHOLD
    edges = spec["tail_count_strata"]
    assert edges == list(TAIL_COUNT_STRATA)
    assert all(isinstance(e, int) and not isinstance(e, bool) for e in edges)
    assert (
        edges[0] == 0
        and 3 <= len(edges) <= 8
        and all(b > a for a, b in zip(edges, edges[1:], strict=False))
    )
    assert spec["min_cell_races"] == MIN_CELL_RACES
    assert isinstance(MIN_CELL_RACES, int) and MIN_CELL_RACES >= 20

    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    registry = default_registry("japan")
    digest = registry.freeze_family(FAMILY_ID, spec)
    assert registry.freeze_family(FAMILY_ID, make_spec()) == digest  # an equal spec: a no-op
    assert registry.records(FAMILY_ID) == []  # frozen with no scored run

    # Everything pre-registered is in the spec: another cut, threshold, strata or cell is a
    # different family, which the registry refuses under this id.
    cuts = [c * 1.01 for c in owner["entropy_cut_points"]]
    others = [
        make_spec(entropy_cut_points=cuts),
        make_spec(tail_min_odds=40.0),
        {**spec, "tail_count_strata": [0, 1, 2, 3, 4, 5, 6, 7]},
        {**spec, "min_cell_races": MIN_CELL_RACES + 1},
    ]
    for other in others:
        assert other != spec
        with pytest.raises(ValueError):  # FamilyAlreadyFrozen
            registry.freeze_family(FAMILY_ID, other)


@pytest.fixture(scope="module")
def study(tail_only):
    frame, cuts, _, _ = tail_only
    spec = make_spec(entropy_cut_points=cuts)
    spec["tail_count_strata"] = list(EDGES)  # the strata the scenario tests pin
    return compute_study(frame, spec)


def delta_metric(variant):
    found = [m for m in variant.metrics if m.metric == "delta_beta_bin1_minus_bin5"]
    assert len(found) == 1, (variant.name, [m.metric for m in variant.metrics])
    return found[0]


def test_compute_study_has_one_scored_variant_per_method_with_the_reference_delta(tail_only, study):
    _, _, _, refs = tail_only
    assert [(v.hypothesis_id, v.name) for v in study.variants] == [
        ("E3", f"entropy_gradient:{m}") for m in METHODS
    ]
    for variant, method in zip(study.variants, METHODS, strict=True):
        assert variant.error is None, (variant.name, variant.error)
        assert variant.fold_scheme and variant.components and variant.params
        json.dumps(variant.params)  # a registry record's params are plain JSON
        m = delta_metric(variant)
        assert m.fold == "development"
        assert m.value == pytest.approx(refs[method]["delta"], abs=1e-5)
        half = WALD_Z * refs[method]["se"]
        assert m.lo == pytest.approx(refs[method]["delta"] - half, abs=1e-4)
        assert m.hi == pytest.approx(refs[method]["delta"] + half, abs=1e-4)
        assert m.n == refs[method]["n_races"]
    assert [row["method"] for row in study.tables["gradient"]] == list(METHODS)


def test_compute_study_shows_the_tail_driven_gradient_vanishing(study):
    raw, share, strata = (delta_metric(v) for v in study.variants)
    assert excludes_zero(raw) and abs(raw.value) > 0.4
    for conditioned in (share, strata):
        assert covers_zero(conditioned)
        assert abs(conditioned.value) < 0.25 * abs(raw.value)


def test_compute_study_takes_the_strata_and_cell_size_from_the_spec(tail_only):
    frame, cuts, _, _ = tail_only
    spec = make_spec(entropy_cut_points=cuts)
    spec["tail_count_strata"] = list(FINE_EDGES)
    spec["min_cell_races"] = 100
    ref = reference_gradient(frame, cuts, "tail_count_strata", edges=FINE_EDGES, min_cell=100)
    default = reference_gradient(frame, cuts, "tail_count_strata", edges=EDGES, min_cell=30)
    assert abs(ref["delta"] - default["delta"]) > 1e-4  # the spec's choice is visible
    other = compute_study(frame, spec)
    assert other.variants[2].error is None
    m = delta_metric(other.variants[2])
    assert m.value == pytest.approx(ref["delta"], abs=1e-5)
    assert m.n == ref["n_races"]


# ---------------------------------------------------------------------------------------------
# run_study: freeze first, development rows only, no holdout, one record per variant
# ---------------------------------------------------------------------------------------------

BUILDS = {
    "jra_runners": "processed/jra/runners.parquet@aaaaaaaaaaaaaaaa",
    "jra_races": "processed/jra/races.parquet@bbbbbbbbbbbbbbbb",
}
REAL_OWNER_INPUTS = calibration.owner_inputs


class _StopHere(Exception):
    pass


def patch_the_store(monkeypatch, tmp_path, frame, cuts):
    """Fake only what is external: the data directory, the store reads and the owner's cuts."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(calibration, "store_builds", lambda: dict(BUILDS))
    monkeypatch.setattr(calibration, "read_development", lambda: (pd.DataFrame(), pd.DataFrame()))
    counts = {"races": int(frame["race_id"].nunique()), "runners": len(frame)}
    monkeypatch.setattr(calibration, "development_frame", lambda runners, races: (frame, counts))
    monkeypatch.setattr(
        calibration, "owner_inputs", lambda: {**REAL_OWNER_INPUTS(), "entropy_cut_points": cuts}
    )

    def refuse(*args, **kwargs):
        raise AssertionError("the holdout was opened")

    monkeypatch.setattr(holdout.LockedHoldout, "open", refuse)


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


@pytest.mark.parametrize(
    ("kind", "seed", "expected"),
    [("tail_only", 2776, False), ("genuine", 2777, True)],
)
def test_run_study_writes_records_and_a_report_and_says_whether_the_gradient_survives(
    tmp_path, monkeypatch, kind, seed, expected
):
    frame, cuts = scenario(kind, seed, BIG)
    patch_the_store(monkeypatch, tmp_path, frame, cuts)
    log = default_registry("japan").holdout_log
    log_before = log.read_bytes() if log.exists() else None

    result = run_study(tmp_path / "reports")

    assert result["family_id"] == FAMILY_ID
    assert result["failed_variants"] == []
    assert result["records_appended"] == len(METHODS)
    assert result["survives_conditioning"] == {m: expected for m in CONDITIONED}

    registry = default_registry("japan")
    registry.verify()
    assert registry.frozen_spec_hash(FAMILY_ID) == result["spec_hash"]
    records = registry.records(FAMILY_ID)
    assert sorted(r.variant for r in records) == sorted(f"entropy_gradient:{m}" for m in METHODS)
    assert {r.hypothesis_id for r in records} == {"E3"} and {r.ticket for r in records} == {277}
    assert all(r.status == "scored" and r.sources == ("jra",) for r in records)
    assert all(r.data_builds == BUILDS for r in records)
    for r in records:
        assert any(m.metric == "delta_beta_bin1_minus_bin5" for m in r.metrics)

    report = tmp_path / "reports"
    files = list(report.glob("*.md"))
    assert len(files) == 1 and str(files[0]) == result["report_path"]
    text = files[0].read_text()
    assert all(m in text for m in METHODS) and "95%" in text

    after = log.read_bytes() if log.exists() else None
    assert after == log_before  # the holdout log is untouched

    again = run_study(tmp_path / "reports")  # a rerun adds no record and keeps the family
    assert again["records_appended"] == 0
    assert again["spec_hash"] == result["spec_hash"]
    assert len(default_registry("japan").records(FAMILY_ID)) == len(METHODS)
