"""Per-race scoring as signed in docs/problem-statement.md section 3 (D6), with the owner's rules.

A Win claim is scored on winner log loss per race (-ln p of the winner), pseudo-R2 is reported
against a reference over the same races (equal chances, 1/field size), and the gain over the
baseline is in nats per race, positive when the model is better.

The owner's answer on #69 (2026-11-24) settles the cases section 3 leaves open, each counted in
the score's report so nothing disappears silently:
- a dead heat with k co-winners scores each co-winner at weight 1/k: -(1/k) * sum ln p;
- a race with no winner (void) is dropped and counted;
- model and baseline (or reference) over different races: only the common races are scored, and
  the races only on each side are counted;
- scratched runners are removed and the probabilities renormalised over the starters.

Expected values are computed here from math.log, never from the code under test.
"""

import dataclasses
import inspect  # noqa: I001 (isort counts horsetrack.core.eval third-party until it exists)
import math

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.scoring import gain, per_race_log_loss, pseudo_r2


def _runner_rows(races, seed):
    """(p, won, race_id, scratched) Series sharing one index, rows shuffled across races.

    `races` maps race id -> (probabilities, winners) or (probabilities, winners, scratched):
    `winners` is one runner position or a collection of them (empty for a void race), and
    `scratched` a collection of runner positions."""
    rows = []
    for race_id, spec in races.items():
        probs, winners = spec[0], spec[1]
        winners = {int(winners)} if isinstance(winners, (int, np.integer)) else set(winners)
        scratched = set(spec[2]) if len(spec) > 2 else set()
        for i, p in enumerate(probs):
            won, out = i in winners, i in scratched
            rows.append({"race_id": race_id, "p": float(p), "won": won, "scratched": out})
    frame = pd.DataFrame(rows)
    frame = frame.iloc[np.random.default_rng(seed).permutation(len(frame))]
    return frame["p"], frame["won"], frame["race_id"], frame["scratched"]


def _score(races, seed):
    p, won, race_id, scratched = _runner_rows(races, seed)
    return per_race_log_loss(p, won, race_id, scratched=scratched)


def _assert_keyed(result, expected):
    """`result` is a Series with exactly the race ids of `expected` (a dict) and those values."""
    assert isinstance(result, pd.Series)
    assert len(result) == len(expected)
    assert set(result.index) == set(expected)
    for race_id, value in expected.items():
        assert result.loc[race_id] == pytest.approx(value, rel=1e-12, abs=1e-12), race_id


def _assert_counts(result, **counts):
    for name, value in counts.items():
        assert getattr(result, name) == value, name


FIELD_SIZES = [2, 3, 5, 7, 8, 9, 12, 14, 16, 18]


def test_a_uniform_forecast_scores_log_field_size():
    races = {
        f"2026-01-{k + 5:02d}-R{k + 1}": ([1.0 / n] * n, (3 * k + 1) % n)
        for k, n in enumerate(FIELD_SIZES)
    }

    result = _score(races, seed=1)

    _assert_keyed(result.per_race, {rid: math.log(len(probs)) for rid, (probs, _) in races.items()})
    _assert_counts(result, n_races=10, n_dead_heats=0, n_void_dropped=0, n_scratched_removed=0)


def test_log_loss_is_minus_log_of_the_winners_probability():
    rng = np.random.default_rng(2)
    races = {}
    for k in range(25):
        n = int(rng.integers(2, 19))
        races[f"R{k:03d}"] = (rng.dirichlet(np.full(n, 0.7)), int(rng.integers(0, n)))

    result = _score(races, seed=3)

    _assert_keyed(result.per_race, {rid: -math.log(probs[w]) for rid, (probs, w) in races.items()})
    assert result.n_races == 25


def test_a_dead_heat_scores_each_co_winner_at_weight_one_over_k():
    # by hand: co-winners at 0.5 and 0.125 score -(ln 0.5 + ln 0.125) / 2 = 2 ln 2
    hand = _score({"DH": ([0.5, 0.125, 0.25, 0.125], [0, 1])}, seed=30)
    _assert_keyed(hand.per_race, {"DH": 2 * math.log(2)})
    _assert_counts(hand, n_races=1, n_dead_heats=1, n_void_dropped=0, n_scratched_removed=0)

    rng = np.random.default_rng(31)
    races, expected, n_dead = {}, {}, 0
    for j in range(24):
        n = int(rng.integers(4, 17))
        k = (1, 2, 3, 4, 2, 1)[j % 6]
        winners = [int(w) for w in rng.choice(n, size=k, replace=False)]
        probs = rng.dirichlet(np.full(n, 0.8))
        races[f"R{j:02d}"] = (probs, winners)
        expected[f"R{j:02d}"] = -sum(math.log(probs[w]) for w in winners) / k
        n_dead += k > 1

    result = _score(races, seed=32)

    _assert_keyed(result.per_race, expected)
    _assert_counts(result, n_races=24, n_dead_heats=n_dead, n_void_dropped=0, n_scratched_removed=0)


@pytest.mark.parametrize("void", [(), (0,), (2, 5, 6, 11)])
def test_a_race_with_no_winner_is_dropped_and_counted(void):
    rng = np.random.default_rng(41 + len(void))
    races, expected = {}, {}
    for j in range(15):
        n = int(rng.integers(3, 15))
        probs = rng.dirichlet(np.ones(n))
        w = int(rng.integers(0, n))
        races[f"R{j:02d}"] = (probs, [] if j in void else w)
        if j not in void:
            expected[f"R{j:02d}"] = -math.log(probs[w])

    result = _score(races, seed=42)

    _assert_keyed(result.per_race, expected)
    _assert_counts(
        result,
        n_races=15 - len(void),
        n_dead_heats=0,
        n_void_dropped=len(void),
        n_scratched_removed=0,
    )


def test_scratched_runners_are_removed_and_the_starters_renormalised():
    # by hand: 0.4 scratched, the winner's 0.3 over the starters' 0.6 is 0.5, so the loss is ln 2
    hand = _score({"S": ([0.4, 0.3, 0.2, 0.1], 1, [0])}, seed=50)
    _assert_keyed(hand.per_race, {"S": math.log(2)})
    _assert_counts(hand, n_races=1, n_dead_heats=0, n_void_dropped=0, n_scratched_removed=1)

    rng = np.random.default_rng(51)
    races, expected, n_scratched = {}, {}, 0
    for j in range(20):
        n = int(rng.integers(5, 17))
        out = [int(s) for s in rng.choice(n, size=j % 4, replace=False)]
        starters = [i for i in range(n) if i not in out]
        w = int(rng.choice(starters))
        probs = rng.dirichlet(np.ones(n))
        races[f"R{j:02d}"] = (probs, w, out)
        expected[f"R{j:02d}"] = -math.log(probs[w] / sum(probs[i] for i in starters))
        n_scratched += len(out)

    result = _score(races, seed=52)

    _assert_keyed(result.per_race, expected)
    _assert_counts(
        result, n_races=20, n_dead_heats=0, n_void_dropped=0, n_scratched_removed=n_scratched
    )


def test_scratches_dead_heats_and_void_races_are_scored_and_counted_together():
    races = {
        # 0.4 scratched; co-winners 0.3 and 0.2 over 0.6: -(ln 1/2 + ln 1/3) / 2
        "A": ([0.4, 0.3, 0.2, 0.1], [1, 2], [0]),
        # void: dropped
        "B": ([0.25] * 4, []),
        # 0.125 and 0.125 scratched; the winner's 0.25 over 0.75 is 1/3
        "C": ([0.5, 0.25, 0.125, 0.125], 1, [2, 3]),
        "D": ([0.2] * 5, 0),
    }

    result = _score(races, seed=60)

    _assert_keyed(
        result.per_race,
        {"A": (math.log(2) + math.log(3)) / 2, "C": math.log(3), "D": math.log(5)},
    )
    _assert_counts(result, n_races=3, n_dead_heats=1, n_void_dropped=1, n_scratched_removed=3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.n_races = 0


def test_a_scratched_runner_marked_as_the_winner_is_refused():
    p, won, race_id, scratched = _runner_rows(
        {"R1": ([0.25] * 4, 1, [1]), "R2": ([0.5, 0.5], 0)}, seed=70
    )
    with pytest.raises(ValueError):
        per_race_log_loss(p, won, race_id, scratched=scratched)


def test_pseudo_r2_of_the_reference_against_itself_is_zero():
    rng = np.random.default_rng(4)
    for n_races in (1, 3, 40):
        sizes = rng.integers(2, 19, size=n_races)
        reference = pd.Series(np.log(sizes.astype(float)), index=[f"R{i}" for i in range(n_races)])
        result = pseudo_r2(reference, reference.copy())
        assert result.value == pytest.approx(0.0, abs=1e-12)
        _assert_counts(result, n_common=n_races, n_only_ll=0, n_only_reference=0)

    # end to end: a uniform forecast's losses against the uniform reference
    races = {f"R{k}": ([1.0 / n] * n, k % n) for k, n in enumerate(FIELD_SIZES)}
    ll = _score(races, seed=5).per_race
    reference = pd.Series({rid: math.log(len(probs)) for rid, (probs, _) in races.items()})
    assert pseudo_r2(ll, reference).value == pytest.approx(0.0, abs=1e-12)


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

    result = pseudo_r2(pd.Series(ll, index=index), pd.Series(ref, index=index))

    assert result.value == pytest.approx(expected, rel=1e-12, abs=1e-12)
    _assert_counts(result, n_common=len(ll), n_only_ll=0, n_only_reference=0)


DIFFERING_RACES = [
    (
        {"R1": 0.5, "R2": 2.0, "R3": 1.0, "R4": 0.7},
        {"R2": math.log(10), "R3": math.log(5), "R4": math.log(3), "R5": 2.1, "R6": 0.4},
    ),
    (
        {"A": 0.1, "B": 0.2, "C": 3.0},
        {"D": math.log(4), "C": math.log(16), "B": math.log(3), "A": math.log(14)},
    ),
    (
        {"X": 1.0, "Y": 1.5, "Z": 0.3, "W": 2.0, "V": 0.9},
        {"Y": math.log(6), "W": math.log(9)},
    ),
]


@pytest.mark.parametrize(("ll", "ref"), DIFFERING_RACES)
def test_pseudo_r2_over_differing_races_uses_the_common_races_and_counts_the_rest(ll, ref):
    common = [rid for rid in ll if rid in ref]
    expected = 1.0 - sum(ll[r] for r in common) / sum(ref[r] for r in common)

    result = pseudo_r2(pd.Series(ll), pd.Series(ref))

    assert result.value == pytest.approx(expected, rel=1e-12, abs=1e-12)
    _assert_counts(
        result,
        n_common=len(common),
        n_only_ll=len(set(ll) - set(ref)),
        n_only_reference=len(set(ref) - set(ll)),
    )


def test_pseudo_r2_with_no_common_races_is_refused():
    with pytest.raises(ValueError):
        pseudo_r2(pd.Series({"A": 1.0, "B": 0.5}), pd.Series({"C": math.log(2)}))


def test_gain_is_positive_when_the_model_beats_the_baseline():
    model = {"R1": 1.20, "R2": 2.50, "R3": 0.40, "R4": 1.90, "R5": 3.10, "R6": 0.75}
    baseline = {"R1": 1.50, "R2": 2.10, "R3": 0.40, "R4": 2.65, "R5": 2.00, "R6": 1.05}
    ll_model = pd.Series(model)
    ll_baseline = pd.Series(baseline).iloc[::-1]  # same races, another order: paired by race id

    result = gain(ll_model, ll_baseline)

    expected = {rid: baseline[rid] - model[rid] for rid in model}
    _assert_keyed(result.per_race, expected)
    for rid in model:
        assert (result.per_race.loc[rid] > 0) == (model[rid] < baseline[rid]), rid
    _assert_counts(result, n_common=6, n_only_model=0, n_only_baseline=0)

    # end to end from probabilities: the gain per race is ln p_model(winner) - ln p_baseline(winner)
    rng = np.random.default_rng(6)
    model_races, baseline_races, expected = {}, {}, {}
    for k in range(20):
        n = int(rng.integers(3, 17))
        w = int(rng.integers(0, n))
        pm, pb = rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(n))
        model_races[f"R{k}"], baseline_races[f"R{k}"] = (pm, w), (pb, w)
        expected[f"R{k}"] = math.log(pm[w]) - math.log(pb[w])
    ll_m = _score(model_races, seed=7).per_race
    ll_b = _score(baseline_races, seed=8).per_race
    _assert_keyed(gain(ll_m, ll_b).per_race, expected)


@pytest.mark.parametrize(("ll_model", "ll_baseline"), DIFFERING_RACES)
def test_gain_over_differing_races_scores_the_common_races_and_counts_the_rest(
    ll_model, ll_baseline
):
    expected = {rid: ll_baseline[rid] - v for rid, v in ll_model.items() if rid in ll_baseline}

    result = gain(pd.Series(ll_model), pd.Series(ll_baseline))

    _assert_keyed(result.per_race, expected)
    _assert_counts(
        result,
        n_common=len(expected),
        n_only_model=len(set(ll_model) - set(ll_baseline)),
        n_only_baseline=len(set(ll_baseline) - set(ll_model)),
    )


@pytest.mark.parametrize("func", [per_race_log_loss, pseudo_r2, gain])
def test_scoring_parameters_have_no_defaults(func):
    params = inspect.signature(func).parameters
    assert params
    assert all(p.default is inspect.Parameter.empty for p in params.values()), params


def test_a_missing_probability_on_a_scratched_runner_is_removed_with_it():
    # the final-price baseline has no price for a scratched horse, so its p is missing (NaN)
    nan = math.nan
    # by hand: the NaN runner is scratched, the winner's 0.3 over the starters' 0.6 is 0.5: ln 2
    hand = _score({"S": ([nan, 0.3, 0.2, 0.1], 1, [0])}, seed=80)
    _assert_keyed(hand.per_race, {"S": math.log(2)})
    _assert_counts(hand, n_races=1, n_dead_heats=0, n_void_dropped=0, n_scratched_removed=1)

    rng = np.random.default_rng(81)
    races, expected, n_scratched = {}, {}, 0
    for j in range(18):
        n = int(rng.integers(5, 17))
        out = [int(s) for s in rng.choice(n, size=1 + j % 3, replace=False)]
        starters = [i for i in range(n) if i not in out]
        w = int(rng.choice(starters))
        probs = rng.dirichlet(np.ones(n))
        # starters keep prices that do not sum to 1 once the scratched ones are gone
        races[f"R{j:02d}"] = ([nan if i in out else probs[i] for i in range(n)], w, out)
        expected[f"R{j:02d}"] = -math.log(probs[w] / sum(probs[i] for i in starters))
        n_scratched += len(out)
    # a race whose every runner is scratched, none priced: void, its rows counted as scratched
    races["ALL_OUT"] = ([nan] * 6, [], range(6))
    n_scratched += 6

    result = _score(races, seed=82)

    _assert_keyed(result.per_race, expected)
    _assert_counts(
        result, n_races=18, n_dead_heats=0, n_void_dropped=1, n_scratched_removed=n_scratched
    )


@pytest.mark.parametrize(
    ("column", "row"),
    [
        ("p", "winner"),  # a starter's p is still required, the winner's included
        ("p", "starter"),
        ("race_id", "scratched"),  # won, race_id and scratched stay required on every row
        ("won", "scratched"),
        ("scratched", "starter"),
    ],
)
def test_a_missing_value_where_one_is_needed_is_refused(column, row):
    p, won, race_id, scratched = _runner_rows(
        {"R1": ([0.4, 0.3, 0.2, 0.1], 1, [0]), "R2": ([0.5, 0.25, 0.25], 2, [1])}, seed=90
    )
    cols = {"p": p, "won": won, "race_id": race_id, "scratched": scratched}
    pick = {
        "winner": won & ~scratched,
        "starter": ~won & ~scratched,
        "scratched": scratched,
    }[row]
    target = pick[pick].index[0]
    cols[column] = cols[column].astype(object if column != "p" else float)
    cols[column].loc[target] = None if column != "p" else math.nan
    # the scratched rows carry no price, as from the final-price baseline
    if column != "p":
        cols["p"] = cols["p"].where(~scratched, math.nan)

    with pytest.raises(ValueError):
        per_race_log_loss(cols["p"], cols["won"], cols["race_id"], scratched=cols["scratched"])


@pytest.mark.parametrize("scale", [0.5, 0.8, 1.25, 1.18, 3.0])
def test_a_race_whose_probabilities_do_not_sum_to_one_is_renormalised(scale):
    # no scratches: -ln(p_w / sum p); e.g. bookmaker-style prices that sum above or below 1
    rng = np.random.default_rng(int(scale * 100))
    races, expected = {}, {}
    for j in range(12):
        n = int(rng.integers(2, 19))
        probs = rng.dirichlet(np.ones(n)) * scale * (1 + 0.1 * j)
        w = int(rng.integers(0, n))
        races[f"R{j:02d}"] = (probs, w)
        expected[f"R{j:02d}"] = -math.log(probs[w] / sum(probs))

    result = _score(races, seed=91)

    _assert_keyed(result.per_race, expected)
    _assert_counts(result, n_races=12, n_dead_heats=0, n_void_dropped=0, n_scratched_removed=0)


@pytest.mark.parametrize("field", [1, 2, 9])
def test_a_race_whose_runners_are_all_scratched_is_dropped_as_void(field):
    races = {
        "OUT": ([1.0 / field] * field, [], range(field)),
        "A": ([0.5, 0.25, 0.25], 1),  # ln 4
        "B": ([0.6, 0.2, 0.1, 0.1], 0, [3]),  # 0.6 over 0.9: ln 1.5
    }

    result = _score(races, seed=92)

    _assert_keyed(result.per_race, {"A": math.log(4), "B": math.log(1.5)})
    _assert_counts(
        result, n_races=2, n_dead_heats=0, n_void_dropped=1, n_scratched_removed=field + 1
    )
