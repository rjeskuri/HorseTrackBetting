"""Ticket #75: max-statistic tests for rule families (SPA and StepM) over whole meetings.

Contract pinned here (the coder implements what these tests call):

- ``cluster_losses(losses, clusters, order)`` sums the per-race losses of each column to one row
  per cluster, the rows ordered by the clusters' dates (``order``), whatever the input row order.
- ``spa(benchmark, losses, *, clusters, order, block_size, n_boot, seed, registry, family_id)``
  and ``stepm(..., level, ...)`` run the stationary bootstrap on that cluster series, so the
  result depends on the per-meeting sums only: moving race values between races of the same
  meeting, or collapsing each meeting to one row holding its sums, leaves it unchanged for a
  fixed seed. Lower loss is better; the null is "no rule beats the benchmark".
- The loss columns must be exactly the family's non-failed variant labels in the registry (any
  order), or ``FamilySizeMismatch`` (a ``ValueError``); ``family_size`` is
  ``Registry.family_size(family_id)``, failed variants included.
- The decision parameters (``clusters``, ``order``, ``block_size``, ``n_boot``, and ``level`` for
  stepm) have no default.

Every value is synthetic and generated here; expected sums come from numpy reshapes of the
generator's arrays, never from the code under test. The values passed for block size, level and
number of replicates are test inputs, not the owner's choices.
"""

from __future__ import annotations  # noqa: I001 (snooping sorts as third-party until it exists)

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.registry import FoldMetric, Registry, RunRecord
from horsetrack.core.eval.snooping import FamilySizeMismatch, cluster_losses, spa, stepm

RULES = [f"r{k}" for k in range(10)]
FAMILY = "P75-rules"
FIRST_DATE = pd.Timestamp("2025-01-04")


# --------------------------------------------------------------------------- helpers


def make_registry(tmp_path: Path, variants: dict[str, str], family_id: str = FAMILY) -> Registry:
    """A registry under tmp_path holding one run per ``variant -> status``."""
    log = tmp_path / "countries" / "japan" / "holdout_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    registry = Registry(tmp_path / "data" / "registry" / "japan.jsonl", holdout_log=log)
    for variant, status in variants.items():
        registry.append(record(family_id, variant, status))
    return registry


def record(family_id: str, variant: str, status: str) -> RunRecord:
    metrics = () if status == "failed" else (FoldMetric("wf-1", "log_loss", 0.69, 0.6, 0.8, 600),)
    return RunRecord(
        hypothesis_id="P1",
        family_id=family_id,
        variant=variant,
        ticket=75,
        git_sha="0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c",
        data_builds={"results": "synthetic-build"},
        sources=("synthetic",),
        fold_scheme="walk-forward-6",
        components={"rule": variant},
        params={"rule": variant},
        metrics=metrics,
        status=status,
    )


def race_index(n_meetings: int, races: int) -> pd.Index:
    return pd.Index([f"M{m:03d}-R{r}" for m in range(n_meetings) for r in range(races)])


def meeting_labels(n_meetings: int, races: int) -> np.ndarray:
    return np.repeat([f"M{m:03d}" for m in range(n_meetings)], races)


def meeting_dates(n_meetings: int, races: int) -> np.ndarray:
    """One meeting per date, three days apart, meeting m on the m-th date."""
    days = pd.to_timedelta(np.arange(n_meetings) * 3, unit="D")
    return np.repeat((FIRST_DATE + days).to_numpy(), races)


def simulate(
    rng, n_meetings: int, races: int, n_rules: int, shifts: dict[int, float] | None = None
):
    """Per-race losses, shape (n_meetings, races, 1 + n_rules); column 0 is the benchmark.

    loss = meeting effect shared by all + race effect shared by all + meeting x rule effect +
    race x rule noise. Every column has the same mean unless ``shifts`` lowers rule k's per-race
    loss by ``shifts[k]`` (in units of the SD of a column's per-meeting summed loss)."""
    k = 1 + n_rules
    shared_meeting = rng.normal(0.0, 0.5, (n_meetings, 1, 1))
    shared_race = rng.normal(0.0, 1.0, (n_meetings, races, 1))
    meeting_rule = rng.normal(0.0, 0.5, (n_meetings, 1, k))
    noise = rng.normal(0.0, 1.0, (n_meetings, races, k))
    values = shared_meeting + shared_race + meeting_rule + noise
    if shifts:
        sd_meeting_sum = values[:, :, 0].sum(axis=1).std(ddof=1)
        for rule, size in shifts.items():
            values[:, :, 1 + rule] -= size * sd_meeting_sum / races
    return values


def frames(values: np.ndarray, rules: list[str]):
    """(benchmark, losses, clusters, order) from a (meetings, races, 1 + rules) array."""
    n_meetings, races, _ = values.shape
    index = race_index(n_meetings, races)
    flat = values.reshape(n_meetings * races, -1)
    benchmark = pd.Series(flat[:, 0], index=index, name="final_price")
    losses = pd.DataFrame(flat[:, 1:], index=index, columns=rules)
    clusters = pd.Series(meeting_labels(n_meetings, races), index=index)
    order = pd.Series(meeting_dates(n_meetings, races), index=index)
    return benchmark, losses, clusters, order


def run_spa(benchmark, losses, clusters, order, registry, *, seed=7, n_boot=200, block_size=4):
    return spa(
        benchmark,
        losses,
        clusters=clusters,
        order=order,
        block_size=block_size,
        n_boot=n_boot,
        seed=seed,
        registry=registry,
        family_id=FAMILY,
    )


def run_stepm(benchmark, losses, clusters, order, registry, *, seed=7, level=0.05):
    return stepm(
        benchmark,
        losses,
        clusters=clusters,
        order=order,
        block_size=4,
        n_boot=500,
        seed=seed,
        level=level,
        registry=registry,
        family_id=FAMILY,
    )


# --------------------------------------------------------------------------- size and power


def test_spa_holds_size_under_a_synthetic_global_null(tmp_path):
    n_datasets, n_meetings, races = 200, 300, 6
    registry = make_registry(tmp_path, {rule: "scored" for rule in RULES})
    rng = np.random.default_rng(20261125)
    pvalues = []
    for dataset in range(n_datasets):
        values = simulate(rng, n_meetings, races, len(RULES))
        benchmark, losses, clusters, order = frames(values, RULES)
        result = run_spa(benchmark, losses, clusters, order, registry, seed=dataset, n_boot=200)
        if dataset == 0:
            assert (result.n_rules, result.family_size, result.n_clusters) == (10, 10, 300)
        assert 0.0 <= result.pvalue_lower <= result.pvalue <= result.pvalue_upper <= 1.0
        pvalues.append(result.pvalue)

    rejection_rate = float(np.mean(np.asarray(pvalues) < 0.05))
    assert 0.01 <= rejection_rate <= 0.10, rejection_rate


def test_spa_rejects_when_one_rule_beats_the_benchmark(tmp_path):
    registry = make_registry(tmp_path, {rule: "scored" for rule in RULES})
    for winner, data_seed in ((3, 101), (8, 202)):
        values = simulate(np.random.default_rng(data_seed), 300, 6, len(RULES), {winner: 0.3})
        benchmark, losses, clusters, order = frames(values, RULES)
        assert run_spa(benchmark, losses, clusters, order, registry).pvalue < 0.05

        # A dataset drawn the same way with no edge is not significant.
        null = simulate(np.random.default_rng(data_seed + 1), 300, 6, len(RULES))
        benchmark, losses, clusters, order = frames(null, RULES)
        assert run_spa(benchmark, losses, clusters, order, registry).pvalue > 0.05


def test_stepm_finds_a_planted_winner(tmp_path):
    registry = make_registry(tmp_path, {rule: "scored" for rule in RULES})
    cases = [({3: 0.3}, 33), ({7: 0.3}, 37), ({2: 0.3, 6: 0.3}, 41), ({}, 43)]
    for planted, data_seed in cases:
        values = simulate(np.random.default_rng(data_seed), 300, 6, len(RULES), planted)
        benchmark, losses, clusters, order = frames(values, RULES)

        result = run_stepm(benchmark, losses, clusters, order, registry, level=0.05)

        assert sorted(result.superior) == sorted(f"r{k}" for k in planted), planted
        assert all(isinstance(name, str) for name in result.superior)
        assert (result.family_size, result.n_clusters) == (10, 300)


# --------------------------------------------------------------------------- whole clusters


def test_cluster_losses_sums_each_meeting_in_time_order():
    # Meeting ids sort differently from their dates, and rows arrive shuffled.
    dates = {
        "Kyoto-a": "2025-03-02",
        "Tokyo-b": "2025-01-05",
        "Hanshin-c": "2025-02-09",
        "Chukyo-d": "2025-01-19",
    }
    races = {"Kyoto-a": 3, "Tokyo-b": 5, "Hanshin-c": 2, "Chukyo-d": 4}
    rng = np.random.default_rng(5)
    rows, expected = [], {}
    for meeting, n in races.items():
        block = rng.normal(0.5, 0.2, (n, 3))
        expected[meeting] = block.sum(axis=0)
        for r in range(n):
            rows.append((f"{meeting}-R{r}", meeting, pd.Timestamp(dates[meeting]), *block[r]))
    table = pd.DataFrame(rows, columns=["race", "meeting", "date", "a", "b", "c"])
    table = table.sample(frac=1.0, random_state=3).set_index("race")

    out = cluster_losses(table[["a", "b", "c"]], table["meeting"], table["date"])

    in_time = sorted(dates, key=lambda m: dates[m])
    assert list(out.index) == in_time
    assert list(out.columns) == ["a", "b", "c"]
    np.testing.assert_allclose(out.to_numpy(), np.vstack([expected[m] for m in in_time]))


def test_bootstrap_resamples_whole_meetings(tmp_path):
    n_meetings, races = 300, 6
    rules = RULES[:4]
    registry = make_registry(tmp_path, {rule: "scored" for rule in rules})
    values = simulate(np.random.default_rng(78), n_meetings, races, len(rules), {1: 0.03})
    benchmark, losses, clusters, order = frames(values, rules)

    # cluster_losses: one row per meeting holding the per-meeting sums.
    sums = cluster_losses(losses, clusters, order)
    assert len(sums) == n_meetings
    np.testing.assert_allclose(sums.to_numpy(), values[:, :, 1:].sum(axis=1))

    base = run_spa(benchmark, losses, clusters, order, registry, seed=11)
    assert 0.0 < base.pvalue < 1.0  # informative: neither bound
    assert base.n_clusters == n_meetings

    # Permute race values within each meeting (benchmark and rules move together), race ids fixed.
    rng = np.random.default_rng(9)
    shuffled = np.stack([values[m, rng.permutation(races)] for m in range(n_meetings)])
    sb, sl, sc, so = frames(shuffled, rules)
    assert not np.allclose(sl.to_numpy(), losses.to_numpy())
    within = run_spa(sb, sl, sc, so, registry, seed=11)
    assert (within.pvalue, within.pvalue_lower, within.pvalue_upper) == (
        base.pvalue,
        base.pvalue_lower,
        base.pvalue_upper,
    )

    # Shuffle the row order of the whole frame too: still the same cluster series.
    perm = rng.permutation(len(sl))
    across = run_spa(sb.iloc[perm], sl.iloc[perm], sc.iloc[perm], so.iloc[perm], registry, seed=11)
    assert across.pvalue == base.pvalue

    # One row per meeting holding its sums is the same cluster series, so the same result.
    collapsed = np.concatenate(
        [values[:, :, :1].sum(axis=1), values[:, :, 1:].sum(axis=1)], axis=1
    )[:, None, :]
    cb, cl, cc, co = frames(collapsed, rules)
    one_row = run_spa(cb, cl, cc, co, registry, seed=11)
    assert (one_row.pvalue, one_row.n_clusters) == (base.pvalue, n_meetings)

    # StepM sees the same cluster series as well.
    assert sorted(run_stepm(sb, sl, sc, so, registry, seed=11).superior) == sorted(
        run_stepm(benchmark, losses, clusters, order, registry, seed=11).superior
    )


# --------------------------------------------------------------------------- the registry family


def test_family_size_must_match_the_registry(tmp_path):
    registry = make_registry(tmp_path, {"a": "scored", "b": "scored", "c": "scored", "d": "failed"})
    registry.append(record("another-family", "x", "scored"))
    values = simulate(np.random.default_rng(3), 60, 6, 4)

    def call(columns, fn=run_spa):
        benchmark, losses, clusters, order = frames(values[:, :, : 1 + len(columns)], columns)
        return fn(benchmark, losses, clusters, order, registry)

    with pytest.raises(FamilySizeMismatch):
        call(["a", "b"])
    for wrong in (["a", "b", "c", "d"], ["a", "b", "c", "x"], ["a", "b", "e"]):
        with pytest.raises(FamilySizeMismatch):
            call(wrong)
    with pytest.raises(FamilySizeMismatch):
        call(["a", "b"], fn=run_stepm)
    assert issubclass(FamilySizeMismatch, ValueError)

    result = call(["a", "b", "c"])
    assert (result.family_size, result.n_rules, result.n_clusters) == (4, 3, 60)
    assert call(["c", "a", "b"]).family_size == 4
    assert call(["a", "b", "c"], fn=run_stepm).family_size == 4

    # A fifth variant tried (and failed) later raises the family size the result reports.
    registry.append(record(FAMILY, "e", "failed"))
    assert call(["a", "b", "c"]).family_size == 5


# --------------------------------------------------------------------------- required decisions


def test_decision_parameters_are_required(tmp_path):
    rules = RULES[:3]
    registry = make_registry(tmp_path, {rule: "scored" for rule in rules})
    benchmark, losses, clusters, order = frames(
        simulate(np.random.default_rng(1), 40, 6, len(rules)), rules
    )
    full = {
        "clusters": clusters,
        "order": order,
        "block_size": 3,
        "n_boot": 100,
        "seed": 1,
        "registry": registry,
        "family_id": FAMILY,
    }
    spa(benchmark, losses, **full)  # the complete call works
    stepm(benchmark, losses, level=0.05, **full)

    for missing in ("block_size", "n_boot", "clusters", "order"):
        kwargs = {k: v for k, v in full.items() if k != missing}
        with pytest.raises(TypeError):
            spa(benchmark, losses, **kwargs)
        with pytest.raises(TypeError):
            stepm(benchmark, losses, level=0.05, **kwargs)
        assert inspect.signature(spa).parameters[missing].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        stepm(benchmark, losses, **full)
    assert inspect.signature(stepm).parameters["level"].default is inspect.Parameter.empty
