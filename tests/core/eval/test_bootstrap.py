"""The paired bootstrap as signed in docs/problem-statement.md section 3.

The interval is on the per-race difference (model and baseline resampled with the same indices),
the resampling unit is the meeting (venue x day, D7), and the level is 95% one-sided (D7). A gain
claim rests on the lower bound: `lo` is the one-sided lower bound at `level`.

Expected bounds come from first principles (the bootstrap distribution's support, or the normal
approximation to it with the cluster standard error), never from the code under test.
"""

import inspect  # noqa: I001 (isort counts horsetrack.core.eval third-party until it exists)
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.scoring import gain, per_race_log_loss


def _three_meetings():
    """60 races in 3 meetings, every race's diff 0, 1 or 2 by its meeting, rows interleaved.

    Resampling whole meetings, a replicate's mean is k/3 for some k, P(mean = 0) = 1/27 < 5%
    and P(mean <= 1/3) = 4/27 > 5%: the 5% quantile is exactly 1/3. Resampling races instead
    puts it near 1 - 1.645 * 0.816 / sqrt(60) = 0.83."""
    meetings = ["Tokyo|2026-01-05", "Kyoto|2026-01-05", "Tokyo|2026-01-06"]
    race_ids = [f"R{i:02d}" for i in range(60)]
    meeting_of = {rid: meetings[i % 3] for i, rid in enumerate(race_ids)}
    diff = pd.Series({rid: float(meetings.index(meeting_of[rid])) for rid in race_ids})
    clusters = pd.Series(meeting_of)
    return diff, clusters


def _synthetic(mu, n_meetings=240, races_per_meeting=10, seed=11):
    """diff = mu + meeting effect + race noise, centred so its per-race mean is exactly mu.

    Returns (diff, clusters, se): se is the standard error of the per-race mean under meeting
    resampling, sqrt(mean((meeting mean - mu)^2) / n_meetings), for equal-size meetings."""
    rng = np.random.default_rng(seed)
    effect = np.repeat(rng.normal(0.0, 0.03, n_meetings), races_per_meeting)
    values = effect + rng.normal(0.0, 0.1, n_meetings * races_per_meeting)
    values = values - values.mean() + mu
    meeting = np.repeat([f"V{k % 6}|day{k // 6:03d}" for k in range(n_meetings)], races_per_meeting)
    race_ids = [f"R{i:05d}" for i in range(values.size)]
    diff = pd.Series(values, index=race_ids)
    clusters = pd.Series(meeting, index=race_ids)
    meeting_means = diff.groupby(clusters).mean().to_numpy()
    se = float(np.sqrt(np.mean((meeting_means - mu) ** 2) / n_meetings))
    return diff, clusters, se


def test_the_bootstrap_is_reproducible_with_a_seed():
    diff, clusters, _ = _synthetic(0.01, n_meetings=60)

    first = paired_bootstrap(diff, clusters, n_boot=500, level=0.95, seed=123)
    again = paired_bootstrap(diff, clusters, n_boot=500, level=0.95, seed=123)
    assert (first.mean, first.lo, first.hi, first.n_clusters) == (
        again.mean,
        again.lo,
        again.hi,
        again.n_clusters,
    )

    los = {paired_bootstrap(diff, clusters, n_boot=500, level=0.95, seed=s).lo for s in (1, 2, 3)}
    assert len(los | {first.lo}) > 1, "the seed must drive the resampling"


def test_whole_clusters_are_resampled():
    diff, clusters = _three_meetings()

    result = paired_bootstrap(diff, clusters, n_boot=5000, level=0.95, seed=5)

    assert result.n_clusters == 3
    assert result.mean == pytest.approx(1.0)
    assert result.lo == pytest.approx(1.0 / 3.0, abs=1e-9)

    # the same races, each its own cluster: now races are the unit and the bound is far tighter
    per_race = pd.Series(diff.index, index=diff.index)
    by_race = paired_bootstrap(diff, per_race, n_boot=5000, level=0.95, seed=5)
    assert by_race.n_clusters == 60
    assert 0.7 < by_race.lo < 0.95


def test_clusters_are_matched_to_races_by_label():
    diff, clusters = _three_meetings()
    shuffled = clusters.iloc[np.random.default_rng(9).permutation(len(clusters))]

    result = paired_bootstrap(diff, shuffled, n_boot=5000, level=0.95, seed=5)

    assert result.n_clusters == 3
    assert result.lo == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_a_race_without_a_meeting_is_refused():
    diff, clusters = _three_meetings()

    with pytest.raises((ValueError, KeyError)):
        paired_bootstrap(diff, clusters.drop(clusters.index[7]), n_boot=200, level=0.95, seed=1)


def test_identical_forecasts_give_a_zero_width_interval():
    rng = np.random.default_rng(21)
    rows = []
    for m in range(12):
        for r in range(8):
            n = int(rng.integers(4, 17))
            probs = rng.dirichlet(np.ones(n))
            winner = int(rng.integers(0, n))
            rows += [
                {"race_id": f"M{m}-R{r}", "meeting": f"M{m}", "p": p, "won": i == winner}
                for i, p in enumerate(probs)
            ]
    frame = pd.DataFrame(rows)
    ll_model = per_race_log_loss(frame["p"], frame["won"], frame["race_id"])
    ll_baseline = per_race_log_loss(frame["p"].copy(), frame["won"], frame["race_id"])
    clusters = frame.drop_duplicates("race_id").set_index("race_id")["meeting"]

    diff = gain(ll_model, ll_baseline)
    result = paired_bootstrap(diff, clusters, n_boot=1000, level=0.95, seed=3)

    assert result.n_clusters == 12
    assert result.mean == pytest.approx(0.0, abs=1e-12)
    assert result.lo == pytest.approx(0.0, abs=1e-12)
    assert result.hi == pytest.approx(0.0, abs=1e-12)

    # a model better by the same 0.0123 nats in every race: the interval collapses on that value
    constant = pd.Series(0.0123, index=clusters.index)
    shifted = paired_bootstrap(constant, clusters, n_boot=1000, level=0.95, seed=3)
    assert shifted.hi - shifted.lo == pytest.approx(0.0, abs=1e-12)
    assert shifted.lo == pytest.approx(0.0123, abs=1e-12)


def test_level_and_n_boot_are_required():
    diff, clusters = _three_meetings()
    full = {"n_boot": 200, "level": 0.95, "seed": 1}
    paired_bootstrap(diff, clusters, **full)

    for name in full:
        partial = {k: v for k, v in full.items() if k != name}
        with pytest.raises(TypeError):
            paired_bootstrap(diff, clusters, **partial)
    with pytest.raises(TypeError):
        paired_bootstrap(diff, clusters, 200, 0.95, 1)

    params = inspect.signature(paired_bootstrap).parameters
    assert all(p.default is inspect.Parameter.empty for p in params.values()), params
    for name in full:
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name


@pytest.mark.parametrize("bad", [{"level": 95}, {"level": 0.0}, {"level": 1.0}, {"n_boot": 0}])
def test_a_level_outside_zero_one_or_no_resamples_is_refused(bad):
    diff, clusters = _three_meetings()

    with pytest.raises(ValueError):
        paired_bootstrap(diff, clusters, **{"n_boot": 200, "level": 0.95, "seed": 1, **bad})


@pytest.mark.parametrize("mu", [0.02, 0.0, -0.01])
def test_the_interval_covers_a_known_mean_in_a_synthetic_case(mu):
    diff, clusters, se = _synthetic(mu)

    result = paired_bootstrap(diff, clusters, n_boot=4000, level=0.95, seed=17)

    assert result.n_clusters == 240
    assert result.mean == pytest.approx(mu, abs=1e-12)
    assert result.lo <= mu <= result.hi
    # a useful bound, not an arbitrarily wide one: about 1.645 standard errors below the mean
    assert mu - 2.5 * se < result.lo < mu - 1.0 * se
    # the positive gain (about 7 standard errors) is detected; no gain is not
    assert (result.lo > 0) == (mu > 0)


@pytest.mark.parametrize("level", [0.80, 0.90, 0.95, 0.99])
def test_lo_is_the_one_sided_lower_bound_at_level(level):
    mu = 0.02
    diff, clusters, se = _synthetic(mu)

    result = paired_bootstrap(diff, clusters, n_boot=4000, level=level, seed=29)

    # one-sided: the (1 - level) quantile, z = 1.645 at 95% (two-sided 95% would sit at 1.96)
    expected = mu - NormalDist().inv_cdf(level) * se
    assert abs(result.lo - expected) < 0.15 * se


def test_the_mean_is_the_per_race_mean_difference():
    # meetings of 1, 4 and 10 races: the per-race mean differs from the mean of meeting means
    sizes, values = [1, 4, 10], [0.9, 0.2, -0.05]
    diff = pd.Series(
        [v for v, n in zip(values, sizes, strict=True) for _ in range(n)],
        index=[f"R{i}" for i in range(sum(sizes))],
    )
    clusters = pd.Series([f"M{k}" for k, n in enumerate(sizes) for _ in range(n)], index=diff.index)

    result = paired_bootstrap(diff, clusters, n_boot=500, level=0.95, seed=2)

    expected = sum(v * n for v, n in zip(values, sizes, strict=True)) / sum(sizes)
    assert result.mean == pytest.approx(expected, rel=1e-12)
    assert result.n_clusters == 3
