"""Per-race scoring as signed in docs/problem-statement.md section 3 (D6).

A Win claim is scored on winner log loss per race (-ln p of the winner), pseudo-R2 is reported
against a reference over the same races (equal chances, 1/field size), and the gain over the
baseline is in nats per race, positive when the model is better.

Expected values are computed here from math.log, never from the code under test.
"""

import inspect  # noqa: I001 (isort counts horsetrack.core.eval third-party until it exists)
import math

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.scoring import gain, per_race_log_loss, pseudo_r2


def _runner_rows(races, seed):
    """(p, won, race_id) Series sharing one index, rows shuffled so races are not contiguous.

    `races` maps race id -> (probabilities, winner position)."""
    rows = []
    for race_id, (probs, winner) in races.items():
        for i, p in enumerate(probs):
            rows.append({"race_id": race_id, "p": float(p), "won": i == winner})
    frame = pd.DataFrame(rows)
    frame = frame.iloc[np.random.default_rng(seed).permutation(len(frame))]
    return frame["p"], frame["won"], frame["race_id"]


def _assert_keyed(result, expected):
    """`result` is a Series with exactly the race ids of `expected` (a dict) and those values."""
    assert isinstance(result, pd.Series)
    assert len(result) == len(expected)
    assert set(result.index) == set(expected)
    for race_id, value in expected.items():
        assert result.loc[race_id] == pytest.approx(value, rel=1e-12, abs=1e-12), race_id


FIELD_SIZES = [2, 3, 5, 7, 8, 9, 12, 14, 16, 18]


def test_a_uniform_forecast_scores_log_field_size():
    races = {
        f"2026-01-{k + 5:02d}-R{k + 1}": ([1.0 / n] * n, (3 * k + 1) % n)
        for k, n in enumerate(FIELD_SIZES)
    }
    p, won, race_id = _runner_rows(races, seed=1)

    result = per_race_log_loss(p, won, race_id)

    _assert_keyed(result, {rid: math.log(len(probs)) for rid, (probs, _) in races.items()})


def test_log_loss_is_minus_log_of_the_winners_probability():
    rng = np.random.default_rng(2)
    races = {}
    for k in range(25):
        n = int(rng.integers(2, 19))
        races[f"R{k:03d}"] = (rng.dirichlet(np.full(n, 0.7)), int(rng.integers(0, n)))
    p, won, race_id = _runner_rows(races, seed=3)

    result = per_race_log_loss(p, won, race_id)

    _assert_keyed(result, {rid: -math.log(probs[w]) for rid, (probs, w) in races.items()})


def test_pseudo_r2_of_the_reference_against_itself_is_zero():
    rng = np.random.default_rng(4)
    for n_races in (1, 3, 40):
        sizes = rng.integers(2, 19, size=n_races)
        reference = pd.Series(np.log(sizes.astype(float)), index=[f"R{i}" for i in range(n_races)])
        assert pseudo_r2(reference, reference.copy()) == pytest.approx(0.0, abs=1e-12)

    # end to end: a uniform forecast's losses against the uniform reference
    races = {f"R{k}": ([1.0 / n] * n, k % n) for k, n in enumerate(FIELD_SIZES)}
    p, won, race_id = _runner_rows(races, seed=5)
    ll = per_race_log_loss(p, won, race_id)
    reference = pd.Series({rid: math.log(len(probs)) for rid, (probs, _) in races.items()})
    assert pseudo_r2(ll, reference) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("ll", "ref"),
    [
        ([0.5, 2.0, 1.0], [math.log(2), math.log(10), math.log(5)]),
        ([0.1, 0.1, 3.0, 2.2], [math.log(14), math.log(3), math.log(16), math.log(8)]),
        ([0.0, 0.0], [math.log(2), math.log(7)]),
        ([1.5], [math.log(4)]),
    ],
)
def test_pseudo_r2_is_one_minus_the_ratio_of_summed_losses(ll, ref):
    index = [f"R{i}" for i in range(len(ll))]
    expected = 1.0 - sum(ll) / sum(ref)

    assert pseudo_r2(pd.Series(ll, index=index), pd.Series(ref, index=index)) == pytest.approx(
        expected, rel=1e-12, abs=1e-12
    )


def test_gain_is_positive_when_the_model_beats_the_baseline():
    model = {"R1": 1.20, "R2": 2.50, "R3": 0.40, "R4": 1.90, "R5": 3.10, "R6": 0.75}
    baseline = {"R1": 1.50, "R2": 2.10, "R3": 0.40, "R4": 2.65, "R5": 2.00, "R6": 1.05}
    ll_model = pd.Series(model)
    ll_baseline = pd.Series(baseline).iloc[::-1]  # same races, another order: paired by race id

    result = gain(ll_model, ll_baseline)

    expected = {rid: baseline[rid] - model[rid] for rid in model}
    _assert_keyed(result, expected)
    for rid in model:
        assert (result.loc[rid] > 0) == (model[rid] < baseline[rid]), rid

    # end to end from probabilities: the gain per race is ln p_model(winner) - ln p_baseline(winner)
    rng = np.random.default_rng(6)
    model_races, baseline_races, expected = {}, {}, {}
    for k in range(20):
        n = int(rng.integers(3, 17))
        w = int(rng.integers(0, n))
        pm, pb = rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(n))
        model_races[f"R{k}"], baseline_races[f"R{k}"] = (pm, w), (pb, w)
        expected[f"R{k}"] = math.log(pm[w]) - math.log(pb[w])
    ll_m = per_race_log_loss(*_runner_rows(model_races, seed=7))
    ll_b = per_race_log_loss(*_runner_rows(baseline_races, seed=8))
    _assert_keyed(gain(ll_m, ll_b), expected)


@pytest.mark.parametrize("func", [per_race_log_loss, pseudo_r2, gain])
def test_scoring_parameters_have_no_defaults(func):
    params = inspect.signature(func).parameters
    assert params
    assert all(p.default is inspect.Parameter.empty for p in params.values()), params
