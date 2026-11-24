"""The v0 ordering model: a discounted Harville model turning Win probabilities into Place and Wide.

P(first = i) = p_i; P(second = j | i) = p_j^l2 / sum_{l != i} p_l^l2;
P(third = k | i, j) = p_k^l3 / sum_{l not in {i, j}} p_l^l3. With (l2, l3) = (1, 1) it is Harville.

Every expected value here is derived independently: brute-force enumeration of ordered finishes,
a closed-form Harville formula, or a vectorised log-likelihood written in this file. All data is
synthetic.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import json
import math
import tracemalloc
from datetime import date, timedelta
from itertools import permutations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from horsetrack.countries.japan.models.v0 import ordering
from horsetrack.countries.japan.models.v0.ordering import (
    OrderingFit,
    fit_ordering,
    top_order_loglik,
)

FAMILY = "discounted_harville"
P4 = np.array([0.4, 0.3, 0.2, 0.1])


# ---------------------------------------------------------------------------------------------
# Independent references
# ---------------------------------------------------------------------------------------------


def brute_force(p, lambdas, places):
    """Place and pair probabilities by enumerating every ordered top-`places` finish.

    Returns (place, pair, n_orders): place[i] = P(i in the first `places`), pair[i, j] =
    P(i and j both in the first `places`), n_orders = how many ordered finishes were summed.
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    weights = [p, p ** lambdas[0], p ** lambdas[1]]
    totals = [w.sum() for w in weights]
    place = np.zeros(n)
    pair = np.zeros((n, n))
    n_orders = 0
    for order in permutations(range(n), places):
        prob = 1.0
        for stage, h in enumerate(order):
            w = weights[stage]
            prob *= w[h] / (totals[stage] - sum(w[g] for g in order[:stage]))
        n_orders += 1
        for a in order:
            place[a] += prob
            for b in order:
                if a != b:
                    pair[a, b] += prob
    return place, pair, n_orders


def harville_order_prob(p, order):
    """Classic Harville: p_i * p_j / (1 - p_i) * p_k / (1 - p_i - p_j), p summing to one."""
    prob, used = 1.0, 0.0
    for h in order:
        prob *= p[h] / (1.0 - used)
        used += p[h]
    return prob


def discounted_order_logprob(p, order, lambdas):
    """log P(first three = order) under the discounted formula, written out per stage."""
    i, j, k = order
    rest_j = [h for h in range(len(p)) if h != i]
    rest_k = [h for h in range(len(p)) if h not in (i, j)]
    l2, l3 = lambdas
    return (
        math.log(p[i] / p.sum())
        + math.log(p[j] ** l2 / sum(p[h] ** l2 for h in rest_j))
        + math.log(p[k] ** l3 / sum(p[h] ** l3 for h in rest_k))
    )


def simulate(rng, n_races, n_runners, lambdas, spread=1.0):
    """Synthetic races from the discounted model (Gumbel-max per stage), vectorised.

    Returns (p, order): p is (n_races, n_runners) Win probabilities, order is (n_races,
    n_runners) column indices in finishing order (the first three drawn from the model, the
    rest in random order).
    """
    logits = rng.normal(0.0, spread, size=(n_races, n_runners))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    logp = np.log(p)
    taken = np.zeros_like(p, dtype=bool)
    key = rng.random((n_races, n_runners)) + 3.0
    rows = np.arange(n_races)
    for stage, lam in enumerate((1.0, *lambdas)):
        score = lam * logp + rng.gumbel(size=p.shape)
        score[taken] = -np.inf
        pick = score.argmax(axis=1)
        taken[rows, pick] = True
        key[rows, pick] = stage
    order = np.argsort(key, axis=1)
    return p, order


def to_frame(rng, p, order, first_race=0):
    """The fit's input frame, rows shuffled so nothing can lean on row order."""
    n_races, n_runners = p.shape
    finish_pos = np.empty_like(order)
    rows = np.arange(n_races)[:, None]
    finish_pos[rows, order] = np.arange(1, n_runners + 1)[None, :]
    frame = pd.DataFrame(
        {
            "race_id": np.repeat([f"R{first_race + r:06d}" for r in range(n_races)], n_runners),
            "horse_no": np.tile(np.arange(1, n_runners + 1), n_races),
            "p_win": p.ravel(),
            "finish_pos": finish_pos.ravel(),
            "dead_heat": np.zeros(n_races * n_runners, dtype=bool),
        }
    )
    return frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)


def _stage_logs(p, order, stage):
    """log p for every runner and the mask of runners still in at `stage` (1 or 2)."""
    logp = np.log(p)
    rows = np.arange(len(p))
    out = np.zeros_like(p, dtype=bool)
    for s in range(stage):
        out[rows, order[:, s]] = True
    return logp, out, logp[rows, order[:, stage]]


def reference_mle(p, order):
    """Independent MLE: the log-likelihood separates into one 1-D problem per stage."""
    fitted, ses = [], []
    for stage in (1, 2):
        logp, out, chosen = _stage_logs(p, order, stage)

        def nll(lam, logp=logp, out=out, chosen=chosen):
            scores = np.where(out, -np.inf, lam * logp)
            return -(lam * chosen - logsumexp(scores, axis=1)).sum()

        lam = minimize_scalar(nll, bounds=(0.05, 4.0), method="bounded", options={"xatol": 1e-9}).x
        fitted.append(lam)
        ses.append(stage_se(p, order, stage, lam))
    return tuple(fitted), tuple(ses)


def stage_se(p, order, stage, lam):
    """Standard error from the observed information: sum over races of Var_w(log p), w ~ p^lam."""
    logp, out, _ = _stage_logs(p, order, stage)
    scores = np.where(out, -np.inf, lam * logp)
    w = np.exp(scores - logsumexp(scores, axis=1, keepdims=True))
    safe = np.where(out, 0.0, logp)
    mean = (w * safe).sum(axis=1)
    var = (w * safe**2).sum(axis=1) - mean**2
    return 1.0 / math.sqrt(var.sum())


# ---------------------------------------------------------------------------------------------
# Definition of done: with lambda = (1, 1) the probabilities equal Harville's
# ---------------------------------------------------------------------------------------------


def test_harville_place_probabilities():
    fit = OrderingFit(FAMILY, (1.0, 1.0), "harville", 0)
    np.testing.assert_allclose(fit.place_probs(P4, 3), [0.9222, 0.8702, 0.7587, 0.4488], atol=1e-4)

    # Beyond the pinned case: varied fields against the classic Harville formula.
    rng = np.random.default_rng(115)
    for n in (5, 7, 9):
        p = rng.dirichlet(np.full(n, 1.3))
        expected = np.zeros(n)
        for order in permutations(range(n), 3):
            prob = harville_order_prob(p, order)
            expected[list(order)] += prob
            assert top_order_loglik(p, order, (1.0, 1.0)) == pytest.approx(math.log(prob), abs=1e-9)
        np.testing.assert_allclose(fit.place_probs(p, 3), expected, atol=1e-10)


def test_harville_pair_probabilities():
    fit = OrderingFit(FAMILY, (1.0, 1.0), "harville", 0)
    pairs = fit.pair_probs(P4, 3)
    assert pairs.shape == (4, 4)
    assert pairs[0, 1] == pytest.approx(0.7925, abs=1e-4)
    assert pairs[2, 3] == pytest.approx(0.2075, abs=1e-4)

    rng = np.random.default_rng(1150)
    for n in (5, 8):
        p = rng.dirichlet(np.full(n, 1.3))
        expected = np.zeros((n, n))
        for order in permutations(range(n), 3):
            prob = harville_order_prob(p, order)
            for a in order:
                for b in order:
                    if a != b:
                        expected[a, b] += prob
        np.testing.assert_allclose(fit.pair_probs(p, 3), expected, atol=1e-10)


def test_two_places_paid():
    fit = OrderingFit(FAMILY, (1.0, 1.0), "harville", 0)
    np.testing.assert_allclose(fit.place_probs(P4, 2), [0.7159, 0.6083, 0.4413, 0.2345], atol=1e-4)

    # Two places use only the second-stage discount: lambda3 must not matter, lambda2 must.
    rng = np.random.default_rng(2)
    p = rng.dirichlet(np.full(9, 1.0))
    for lambdas in [(1.0, 1.0), (0.8, 0.7), (1.3, 0.5)]:
        expected_place, expected_pair, _ = brute_force(p, lambdas, 2)
        got = OrderingFit(FAMILY, lambdas, "x", 0)
        np.testing.assert_allclose(got.place_probs(p, 2), expected_place, atol=1e-10)
        np.testing.assert_allclose(got.pair_probs(p, 2), expected_pair, atol=1e-10)
    a = OrderingFit(FAMILY, (0.8, 0.3), "x", 0).place_probs(p, 2)
    b = OrderingFit(FAMILY, (0.8, 1.7), "x", 0).place_probs(p, 2)
    c = OrderingFit(FAMILY, (1.0, 0.3), "x", 0).place_probs(p, 2)
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert np.abs(a - c).max() > 1e-3


# ---------------------------------------------------------------------------------------------
# Definition of done: place_probs sums to places; pair_probs symmetric, zero diagonal,
# rows sum to (places - 1) * place_probs
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lambdas", [(1.0, 1.0), (0.8, 0.7), (1.25, 0.55)])
@pytest.mark.parametrize("places", [1, 2, 3])
def test_place_probs_sum_to_places_and_pairs_are_consistent(lambdas, places):
    rng = np.random.default_rng(14)
    p = rng.dirichlet(np.full(14, 0.9))
    fit = OrderingFit(FAMILY, lambdas, "consistency", 0)

    place = fit.place_probs(p, places)
    pairs = fit.pair_probs(p, places)

    assert place.shape == (14,)
    assert pairs.shape == (14, 14)
    assert place.sum() == pytest.approx(places, abs=1e-9)
    assert np.all(place >= 0) and np.all(place <= 1 + 1e-12)
    np.testing.assert_allclose(pairs, pairs.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(pairs), 0.0, atol=1e-15)
    np.testing.assert_allclose(pairs.sum(axis=1), (places - 1) * place, atol=1e-9)

    # And the values are the discounted model's, not just internally consistent.
    expected_place, expected_pair, _ = brute_force(p, lambdas, places)
    np.testing.assert_allclose(place, expected_place, atol=1e-10)
    np.testing.assert_allclose(pairs, expected_pair, atol=1e-10)
    for order in [(0, 1, 2), (13, 5, 7), tuple(np.argsort(-p)[:3]), tuple(np.argsort(p)[:3])]:
        assert top_order_loglik(p, order, lambdas) == pytest.approx(
            discounted_order_logprob(p, order, lambdas), abs=1e-9
        )


# ---------------------------------------------------------------------------------------------
# Definition of done: the fit recovers planted discounts
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "planted, n_races, seed",
    [((0.8, 0.7), 5000, 20261124), ((1.05, 0.6), 3000, 73)],
)
def test_planted_discounts_are_recovered(planted, n_races, seed):
    rng = np.random.default_rng(seed)
    p, order = simulate(rng, n_races, 14, planted)
    frame = to_frame(rng, p, order)

    fit = fit_ordering(frame, family=FAMILY)

    assert fit.family == FAMILY
    assert fit.n_races == n_races
    assert isinstance(fit.fit_id, str) and fit.fit_id
    ref, _ = reference_mle(p, order)
    for stage, (got, truth) in enumerate(zip(fit.lambdas, planted, strict=True), start=1):
        se = stage_se(p, order, stage, got)
        assert abs(got - truth) < 0.1, (stage, got, truth)
        assert abs(got - truth) < 3 * se, (stage, got, truth, se)
        # It is the maximum-likelihood estimate on these races, not a constant near the truth.
        assert got == pytest.approx(ref[stage - 1], abs=5e-3), (stage, got, ref)

    # top_order_loglik is the per-race term the fit maximises.
    for r in range(0, n_races, n_races // 7):
        top3 = tuple(int(h) for h in order[r, :3])
        assert top_order_loglik(p[r], top3, fit.lambdas) == pytest.approx(
            discounted_order_logprob(p[r], top3, fit.lambdas), abs=1e-9
        )


# ---------------------------------------------------------------------------------------------
# Definition of done: races with a dead heat in the first three are left out and counted
# ---------------------------------------------------------------------------------------------


def test_dead_heats_in_the_first_three_are_left_out_and_counted():
    rng = np.random.default_rng(3)
    n_clean, n_tied_top, n_tied_fourth, n = 700, 60, 25, 12
    p_clean, o_clean = simulate(rng, n_clean, n, (0.9, 0.8))
    p_tied, o_tied = simulate(rng, n_tied_top + n_tied_fourth, n, (0.9, 0.8))
    # Make the top-three ties extreme (the weakest runners fill the first places), so counting
    # them would move the fit well away from the clean-race estimate.
    for r in range(n_tied_top):
        weakest = np.argsort(p_tied[r])
        o_tied[r] = np.concatenate([weakest[:3], np.setdiff1d(np.arange(n), weakest[:3])])

    clean = to_frame(rng, p_clean, o_clean)
    tied = to_frame(rng, p_tied, o_tied, first_race=n_clean)
    tied = tied.sort_values(["race_id", "finish_pos"]).reset_index(drop=True)
    tie_at = [1, 2, 3]  # a tie for first, second or third, rotated across races
    for r, race_id in enumerate(sorted(tied["race_id"].unique())):
        rows = tied.index[tied["race_id"] == race_id]
        pos = tie_at[r % 3] if r < n_tied_top else 4
        a, b = rows[pos - 1], rows[pos]  # the horses finishing pos and pos + 1
        tied.loc[b, "finish_pos"] = pos
        tied.loc[[a, b], "dead_heat"] = True

    kept_fourth = tied[tied["race_id"].isin(sorted(tied["race_id"].unique())[n_tied_top:])]
    usable = pd.concat([clean, kept_fourth], ignore_index=True)
    everything = pd.concat([clean, tied], ignore_index=True)
    everything = everything.iloc[rng.permutation(len(everything))].reset_index(drop=True)

    fit_all = fit_ordering(everything, family=FAMILY)
    fit_usable = fit_ordering(usable, family=FAMILY)

    assert fit_all.n_races == n_clean + n_tied_fourth
    assert fit_usable.n_races == n_clean + n_tied_fourth
    np.testing.assert_allclose(fit_all.lambdas, fit_usable.lambdas, atol=1e-4)

    # The fourth-place ties are kept: the estimate is the MLE over clean + fourth-tie races.
    p_keep = np.vstack([p_clean, p_tied[n_tied_top:]])
    o_keep = np.vstack([o_clean, o_tied[n_tied_top:]])
    ref, _ = reference_mle(p_keep, o_keep)
    np.testing.assert_allclose(fit_all.lambdas, ref, atol=5e-3)
    # Had the top-three ties been counted, the estimate would sit clearly elsewhere.
    ref_with_ties, _ = reference_mle(
        np.vstack([p_keep, p_tied[:n_tied_top]]), np.vstack([o_keep, o_tied[:n_tied_top]])
    )
    assert np.abs(np.subtract(ref_with_ties, ref)).max() > 0.02


# ---------------------------------------------------------------------------------------------
# Definition of done: enumeration is per race; nothing builds triples for more than one race
# ---------------------------------------------------------------------------------------------


def test_an_18_runner_race_enumerates_4896_ordered_triples():
    rng = np.random.default_rng(18)
    p = rng.dirichlet(np.full(18, 0.8))
    lambdas = (0.85, 0.65)
    expected_place, expected_pair, n_orders = brute_force(p, lambdas, 3)
    assert n_orders == 18 * 17 * 16 == 4896

    fit = OrderingFit(FAMILY, lambdas, "eighteen", 0)
    np.testing.assert_allclose(fit.place_probs(p, 3), expected_place, atol=1e-10)
    np.testing.assert_allclose(fit.pair_probs(p, 3), expected_pair, atol=1e-10)

    # Fitting many 18-runner races must not hold every race's triples at once. 3,000 races of
    # 4,896 triples would need about 117 MB for float64 probabilities alone; per-race work
    # stays far below the bound (the input frame itself is about 2 MB).
    n_races = 3000
    p_many, o_many = simulate(rng, n_races, 18, lambdas)
    frame = to_frame(rng, p_many, o_many)
    tracemalloc.start()
    try:
        fitted = fit_ordering(frame, family=FAMILY)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert fitted.n_races == n_races
    assert peak < 40 * 2**20, f"peak {peak / 2**20:.1f} MiB while fitting {n_races} races"


# ---------------------------------------------------------------------------------------------
# Outline 4: only 'discounted_harville' is implemented
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["plackett_luce", "henery_gamma"])
def test_an_unimplemented_family_is_refused(family):
    rng = np.random.default_rng(4)
    p, order = simulate(rng, 50, 10, (1.0, 1.0))
    frame = to_frame(rng, p, order)

    with pytest.raises(NotImplementedError, match=family):
        fit_ordering(frame, family=family)
    with pytest.raises(NotImplementedError, match=family):
        OrderingFit(family, (1.0, 1.0), "x", 0).place_probs(P4, 3)
    with pytest.raises(NotImplementedError, match=family):
        OrderingFit(family, (1.0, 1.0), "x", 0).pair_probs(P4, 3)


# ---------------------------------------------------------------------------------------------
# Outline 2: a JSON round trip
# ---------------------------------------------------------------------------------------------


def test_the_fit_round_trips_through_json():
    rng = np.random.default_rng(5)
    p, order = simulate(rng, 400, 12, (0.85, 0.75))
    fitted = fit_ordering(to_frame(rng, p, order), family=FAMILY)
    built = OrderingFit(FAMILY, (0.8312345678901234, 0.7098765432109876), "fit-abc123", 4321)

    probe = rng.dirichlet(np.full(16, 1.0))
    for original in (fitted, built):
        text = original.to_json()
        json.loads(text)  # valid JSON
        back = OrderingFit.from_json(text)
        assert back.family == original.family
        assert tuple(back.lambdas) == tuple(original.lambdas)
        assert back.fit_id == original.fit_id
        assert back.n_races == original.n_races
        for places in (2, 3):
            assert np.array_equal(
                back.place_probs(probe, places), original.place_probs(probe, places)
            )
            assert np.array_equal(
                back.pair_probs(probe, places), original.pair_probs(probe, places)
            )
    assert built.n_races == 4321 and fitted.n_races == 400


# ---------------------------------------------------------------------------------------------
# Outline 3: run_fit reads #114's OOF (race_id, horse_no, fold, q, p_cal) and joins finish_pos,
# dead_heat and is_scratched from processed_dir('jra')/runners.parquet on (race_id, horse_no)
# ---------------------------------------------------------------------------------------------

DATA_VAR = "HORSETRACK_DATA_DIR"
OOF_NAME = "win_calibration_oof-0123456789ab.parquet"
RUNNERS_SCHEMA = pa.schema(
    [
        ("race_id", pa.string()),
        ("race_date", pa.date32()),
        ("horse_no", pa.int32()),
        ("win_odds", pa.float64()),
        ("finish_pos", pa.int32()),  # null: scratched or did not finish
        ("is_scratched", pa.bool_()),
        ("dead_heat", pa.bool_()),
        ("win_fav_rank", pa.int32()),
    ]
)
OOF_SCHEMA = pa.schema(
    [
        ("race_id", pa.string()),
        ("horse_no", pa.int64()),
        ("fold", pa.int64()),
        ("q", pa.float64()),
        ("p_cal", pa.float64()),
    ]
)


def race_key(r):
    """A store-style race id and date for synthetic race number r."""
    day = date(2020, 1, 4) + timedelta(days=r // 12)
    return f"jra:{day.isoformat()}:TOK:{r % 12 + 1:02d}", day


def race_tables(p, order, first_race=0):
    """#114's OOF rows and the store's runner rows for simulated races.

    p_cal is the model's Win probability; q (the market) is deliberately a different curve of
    it, so a fit on q instead of p_cal lands far from the reference.
    """
    n_races, n = p.shape
    rows = np.arange(n_races)[:, None]
    finish_pos = np.empty(order.shape, dtype=float)
    finish_pos[rows, order] = np.arange(1, n + 1)[None, :]
    ids, days = zip(*(race_key(first_race + r) for r in range(n_races)), strict=True)
    q = p**0.6
    q /= q.sum(axis=1, keepdims=True)
    horse_no = np.tile(np.arange(1, n + 1), n_races)
    oof = pd.DataFrame(
        {
            "race_id": np.repeat(ids, n),
            "horse_no": horse_no,
            "fold": np.repeat(np.arange(n_races) % 4, n),
            "q": q.ravel(),
            "p_cal": p.ravel(),
        }
    )
    runners = pd.DataFrame(
        {
            "race_id": np.repeat(ids, n),
            "race_date": np.repeat(days, n),
            "horse_no": horse_no,
            "win_odds": 0.8 / q.ravel(),
            "finish_pos": finish_pos.ravel(),
            "is_scratched": False,
            "dead_heat": False,
            "win_fav_rank": np.argsort(np.argsort(-q, axis=1), axis=1).ravel() + 1,
        }
    )
    return oof, runners


def write_inputs(tmp_path, monkeypatch, oof, runners):
    """Point the data dir at tmp_path, write the store's runners table and the OOF file."""
    data = (tmp_path / "data").resolve()
    monkeypatch.setenv(DATA_VAR, str(data))
    # read_family's default path is bound at definition and the config is an owner decision:
    # supply the family through the module seam run_fit calls.
    monkeypatch.setattr(ordering, "read_family", lambda *a, **k: FAMILY)
    rng = np.random.default_rng(len(oof) + len(runners))
    store = data / "processed" / "jra"
    store.mkdir(parents=True)
    runners = runners.iloc[rng.permutation(len(runners))].reset_index(drop=True)
    pq.write_table(
        pa.Table.from_pandas(runners, schema=RUNNERS_SCHEMA, preserve_index=False),
        store / "runners.parquet",
    )
    oof_path = data / "models" / "japan" / "v0" / OOF_NAME
    oof_path.parent.mkdir(parents=True)
    oof = oof.iloc[rng.permutation(len(oof))].reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(oof, schema=OOF_SCHEMA, preserve_index=False), oof_path)
    return data, oof_path


def assert_fit_files(out_dir, manifest):
    """The fit and the manifest are in out_dir under the ticket's names, and agree."""
    fit_id = manifest["fit_id"]
    fit_path = out_dir / f"ordering-{fit_id}.json"
    manifest_path = out_dir / f"manifest-ordering-{fit_id}.json"
    assert fit_path.is_file() and manifest_path.is_file()
    fit = OrderingFit.from_json(fit_path.read_text())
    assert fit.family == FAMILY and fit.fit_id == fit_id
    assert fit.n_races == manifest["n_races"]
    np.testing.assert_allclose(fit.lambdas, manifest["lambdas"], rtol=0, atol=1e-12)
    assert json.loads(manifest_path.read_text()) == json.loads(json.dumps(manifest))
    return fit


@pytest.mark.parametrize(
    "planted, n_races, n_runners, seed",
    [((0.8, 0.7), 2500, 12, 1142), ((1.2, 0.55), 2000, 10, 1143)],
)
def test_run_fit_fits_p_cal_from_the_oof_joined_to_the_store_results(
    tmp_path, monkeypatch, planted, n_races, n_runners, seed
):
    rng = np.random.default_rng(seed)
    p, order = simulate(rng, n_races, n_runners, planted)
    oof, runners = race_tables(p, order)
    _, oof_path = write_inputs(tmp_path, monkeypatch, oof, runners)
    out_dir = tmp_path / "out"

    manifest = ordering.run_fit(oof_path, out_dir)

    assert manifest["n_races"] == n_races
    assert manifest["family"] == FAMILY
    fit = assert_fit_files(out_dir, manifest)
    ref, _ = reference_mle(p, order)
    for stage, (got, truth) in enumerate(zip(fit.lambdas, planted, strict=True), start=1):
        se = stage_se(p, order, stage, got)
        assert abs(got - truth) < max(0.15, 4 * se), (stage, got, truth, se)
        # The MLE on p_cal (not q) over exactly these races.
        assert got == pytest.approx(ref[stage - 1], abs=5e-3), (stage, got, ref)


def test_run_fit_drops_scratched_runners_and_adds_no_runner_outside_the_oof(tmp_path, monkeypatch):
    rng = np.random.default_rng(1144)
    n_races, n = 1500, 10
    p, order = simulate(rng, n_races, n, (0.85, 0.7))
    oof, runners = race_tables(p, order)
    ids = runners["race_id"].unique()

    # A scratched runner in every race, carried in the OOF with a large p_cal: kept in the field
    # it would move both discounts well away from the clean estimate (checked below).
    scratched_p = 0.5
    scr_oof = pd.DataFrame(
        {"race_id": ids, "horse_no": n + 1, "fold": 0, "q": 0.01, "p_cal": scratched_p}
    )
    scr_run = pd.DataFrame(
        {
            "race_id": ids,
            "race_date": [race_key(r)[1] for r in range(n_races)],
            "horse_no": n + 1,
            "win_odds": np.nan,
            "finish_pos": np.nan,
            "is_scratched": True,
            "dead_heat": False,
            "win_fav_rank": 0,
        }
    )
    # A runner the store has and the OOF does not (unpriced, finished last): not in the field.
    extra_run = scr_run.assign(horse_no=n + 2, finish_pos=float(n + 1), is_scratched=False)
    # Whole races the store has and the OOF does not: they are not fitted and not counted.
    p_out, o_out = simulate(rng, 300, n, (0.3, 2.0))
    _, out_run = race_tables(p_out, o_out, first_race=n_races)

    _, oof_path = write_inputs(
        tmp_path,
        monkeypatch,
        pd.concat([oof, scr_oof], ignore_index=True),
        pd.concat([runners, scr_run, extra_run, out_run], ignore_index=True),
    )

    manifest = ordering.run_fit(oof_path, tmp_path / "out")

    assert manifest["n_races"] == n_races
    left_out = manifest["races_left_out"]
    assert left_out["dead_heat_in_first_three"] == 0
    assert left_out["incomplete_first_three"] == 0
    ref, _ = reference_mle(p, order)
    np.testing.assert_allclose(manifest["lambdas"], ref, atol=5e-3)
    # The scratched runner matters: fitted as part of the field it would give another answer.
    with_scratched, _ = reference_mle(np.hstack([p, np.full((n_races, 1), scratched_p)]), order)
    assert np.abs(np.subtract(with_scratched, ref)).max() > 0.05


def test_the_manifest_counts_races_left_out_for_dead_heats_and_incomplete_results(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(1145)
    n = 11
    n_clean, n_dead_heat, n_missing_placer, n_no_third = 900, 37, 14, 9
    p, order = simulate(rng, n_clean, n, (0.9, 0.75))
    oof, runners = race_tables(p, order)

    def bad_races(first_race, n_dh, n_missing, n_dnf):
        """Races with a dead heat in the first three, a placed runner absent from the OOF, or
        no third finisher; returns their OOF and store rows."""
        total = n_dh + n_missing + n_dnf
        pb, ob = simulate(rng, total, n, (0.9, 0.75))
        b_oof, b_run = race_tables(pb, ob, first_race=first_race)
        drop = []
        for r, race_id in enumerate(b_run["race_id"].unique()):
            in_race = b_run["race_id"] == race_id
            pos = b_run["finish_pos"]
            if r < n_dh:
                tie = r % 3 + 1  # a tie for first, second or third
                pair = b_run.index[in_race & pos.isin([tie, tie + 1])]
                b_run.loc[pair, "finish_pos"] = tie
                b_run.loc[pair, "dead_heat"] = True
            elif r < n_dh + n_missing:
                placed = int(b_run.loc[in_race & (pos == r % 3 + 1), "horse_no"].iloc[0])
                drop.append((race_id, placed))
            else:
                b_run.loc[in_race & (pos >= 3), "finish_pos"] = np.nan
        keys = set(drop)
        mask = [(a, b) in keys for a, b in zip(b_oof["race_id"], b_oof["horse_no"], strict=True)]
        return b_oof[~np.array(mask, dtype=bool)], b_run

    bad_oof, bad_run = bad_races(n_clean, n_dead_heat, n_missing_placer, n_no_third)
    # The same kinds of race in the store but not in the OOF: none of them is counted.
    _, stray_run = bad_races(n_clean + 500, 11, 5, 4)

    _, oof_path = write_inputs(
        tmp_path,
        monkeypatch,
        pd.concat([oof, bad_oof], ignore_index=True),
        pd.concat([runners, bad_run, stray_run], ignore_index=True),
    )
    out_dir = tmp_path / "out"

    manifest = ordering.run_fit(oof_path, out_dir)

    assert manifest["n_races"] == n_clean
    left_out = manifest["races_left_out"]
    assert left_out["dead_heat_in_first_three"] == n_dead_heat
    assert left_out["incomplete_first_three"] == n_missing_placer + n_no_third
    ref, _ = reference_mle(p, order)
    np.testing.assert_allclose(manifest["lambdas"], ref, atol=5e-3)
    assert_fit_files(out_dir, manifest)


def test_the_fit_and_its_manifest_are_written_to_the_out_dir(tmp_path, monkeypatch):
    rng = np.random.default_rng(1146)
    p, order = simulate(rng, 400, 9, (0.8, 0.8))
    oof, runners = race_tables(p, order)
    data, oof_path = write_inputs(tmp_path, monkeypatch, oof, runners)
    out_dir = tmp_path / "elsewhere" / "fits"

    manifest = ordering.run_fit(oof_path, out_dir)

    fit_id = manifest["fit_id"]
    assert isinstance(fit_id, str) and fit_id
    assert sorted(x.name for x in out_dir.iterdir()) == sorted(
        [f"ordering-{fit_id}.json", f"manifest-ordering-{fit_id}.json"]
    )
    fit = assert_fit_files(out_dir, manifest)
    assert fit.n_races == 400
    ref, _ = reference_mle(p, order)
    np.testing.assert_allclose(fit.lambdas, ref, atol=5e-3)
    # Nothing fitted lands next to the OOF when an out dir is given.
    assert not list((data / "models" / "japan" / "v0").glob("*ordering-*"))


def test_the_default_out_dir_is_models_japan_v0_under_the_data_dir(tmp_path, monkeypatch):
    rng = np.random.default_rng(1147)
    p, order = simulate(rng, 300, 8, (0.9, 0.7))
    oof, runners = race_tables(p, order)
    data, oof_path = write_inputs(tmp_path, monkeypatch, oof, runners)

    manifest = ordering.run_fit(oof_path)

    default = data / "models" / "japan" / "v0"
    fit = assert_fit_files(default, manifest)
    assert fit.n_races == 300
    assert sorted(x.name for x in default.glob("*ordering-*.json")) == sorted(
        [f"ordering-{fit.fit_id}.json", f"manifest-ordering-{fit.fit_id}.json"]
    )
