"""Competition terms for the C2 per-horse model (ticket #94, epic #73; 08 §3.3; 05 §3.4, §4.2).

Contract pinned here (`horsetrack.countries.japan.models.competition`):
- `field_strength(frame, *, rating_col) -> pd.DataFrame`: one row per input row, on the input's
  index, with field_mean and field_max: the mean and max of rating_col over the OTHER runners of
  the same race_id, never the runner itself. Rows need not be grouped by race. A race of one runner
  has no other runners: both are NaN.
- `comparison_graph(results, *, before_utc) -> Graph`: results as the store and `models.rating`
  give them (race_id, race_utc tz-aware UTC, horse_id, finish_pos); horses who ran in the same race
  strictly before before_utc are joined. A race at or after before_utc is never read.
- `effective_resistance(graph, a, b) -> float`: the effective resistance between horses a and b,
  (e_a - e_b)' L^+ (e_a - e_b) on the graph's Laplacian. Horses in different components, or a horse
  absent from the graph, give +inf, never an error. Symmetric.
- `race_comparability(frame, graph) -> pd.Series`: per race_id of frame, the mean over the race's
  unordered runner pairs of their effective resistance; +inf when any pair is unconnected.

Expected resistances come from a dense pseudo-inverse of a Laplacian built here from graphs whose
every edge is one two-runner race, so each edge has unit weight whatever the weighting rule for
larger fields (not pinned). Also not pinned: the weight a larger field gives each pair, the Graph's
attributes, the solver, a one-runner race's comparability, and the real-data run (outline 4).
Every RNG is seeded.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import itertools
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from horsetrack.countries.japan.models import competition
from horsetrack.countries.japan.models.competition import (
    Graph,
    comparison_graph,
    effective_resistance,
    field_strength,
    race_comparability,
)

T = datetime(2025, 6, 1, 6, 0, tzinfo=timezone.utc)  # noqa: UP017 (the red run's Python may be 3.10)


def results_of(races) -> pd.DataFrame:
    """races: (race_id, race_utc, [horse ids in finishing order]) -> the results frame."""
    rows = [
        {"race_id": rid, "race_utc": pd.Timestamp(utc), "horse_id": h, "finish_pos": pos}
        for rid, utc, horses in races
        for pos, h in enumerate(horses, start=1)
    ]
    return pd.DataFrame(rows)


def before(days: float) -> datetime:
    return T - timedelta(days=days)


def reference_resistance(edges, horses, a, b) -> float:
    """(e_a - e_b)' L^+ (e_a - e_b) on the unit-weight simple graph `edges`; inf across components
    or for a horse not in `horses`."""
    if a == b and a in horses:
        return 0.0
    if a not in horses or b not in horses:
        return math.inf
    index = {h: i for i, h in enumerate(horses)}
    n = len(horses)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    L = np.zeros((n, n))
    for u, v in edges:
        i, j = index[u], index[v]
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= 1
        L[j, i] -= 1
        parent[root(i)] = root(j)
    if root(index[a]) != root(index[b]):
        return math.inf
    d = np.zeros(n)
    d[index[a]], d[index[b]] = 1.0, -1.0
    return float(d @ np.linalg.pinv(L) @ d)


def random_pair_graph(seed: int):
    """(results, edges, horses) for a random simple graph of two components, each edge one
    two-runner race before T, plus distractor races at or after T on random pairs (new edges and
    new horses among them) that the graph must ignore."""
    rng = np.random.default_rng(seed)
    edges, horses = [], []
    for comp in range(2):
        n = int(rng.integers(6, 16))
        names = [f"h:{seed}:{comp}:{i:02d}" for i in range(n)]
        horses += names
        comp_edges = {(names[i], names[int(rng.integers(0, i))]) for i in range(1, n)}  # a tree
        for _ in range(int(rng.integers(0, n))):  # extra edges make cycles
            i, j = rng.choice(n, 2, replace=False)
            if (names[j], names[i]) not in comp_edges:
                comp_edges.add((names[i], names[j]))
        edges += sorted(comp_edges)
    races = [
        (f"jra:r{seed}:{k:04d}", before(float(rng.uniform(1, 900))), list(e))
        for k, e in enumerate(edges)
    ]
    late_pool = horses + [f"h:{seed}:late:{i}" for i in range(4)]
    for k in range(12):
        i, j = rng.choice(len(late_pool), 2, replace=False)
        when = T if k % 3 == 0 else T + timedelta(days=float(rng.uniform(0, 30)))
        races.append((f"jra:late{seed}:{k:03d}", when, [late_pool[i], late_pool[j]]))
    results = results_of(races).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return results, edges, horses


def assert_resistance(got: float, want: float) -> None:
    assert isinstance(got, float), type(got)
    if math.isinf(want):
        assert math.isinf(got) and got > 0, got
    else:
        assert got == pytest.approx(want, rel=1e-6, abs=1e-9)


# ---------------------------------------------------------------- field strength


def test_field_strength_excludes_the_runner_itself():
    frame = pd.DataFrame(
        {"race_id": ["r1"] * 3, "horse_id": ["a", "b", "c"], "mu_before": [1.0, 2.0, 3.0]}
    )
    out = field_strength(frame, rating_col="mu_before")
    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(frame.index)
    assert out["field_mean"].to_numpy(dtype=float) == pytest.approx([2.5, 2.0, 1.5])
    assert out["field_max"].to_numpy(dtype=float) == pytest.approx([3.0, 3.0, 2.0])


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("rating_col", ["mu_before", "figure"])
def test_field_strength_over_many_races_matches_a_leave_one_out_loop(seed, rating_col):
    """Shuffled rows of many races of 2 to 12 runners, ties planted, the named rating column read
    and a decoy column ignored: each runner's mean and max of the others in its race."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(2, 13, size=25)
    race = np.repeat([f"jra:r{i:03d}" for i in range(len(sizes))], sizes)
    rating = np.round(rng.normal(0.0, 1.0, size=len(race)), 1)  # rounding plants ties
    frame = pd.DataFrame(
        {
            "race_id": race,
            "horse_id": [f"h:{i:04d}" for i in range(len(race))],
            rating_col: rating,
            "decoy": rng.normal(50.0, 10.0, size=len(race)),
        }
    )
    frame = frame.sample(frac=1.0, random_state=seed)
    frame.index = pd.Index(rng.permutation(len(frame)) + 1000)  # a non-default index
    out = field_strength(frame, rating_col=rating_col)
    assert out.index.equals(frame.index)
    want_mean, want_max = [], []
    for idx in frame.index:
        others = frame[(frame["race_id"] == frame.at[idx, "race_id"]) & (frame.index != idx)]
        want_mean.append(others[rating_col].mean())
        want_max.append(others[rating_col].max())
    assert out["field_mean"].to_numpy(dtype=float) == pytest.approx(want_mean, rel=1e-12)
    assert out["field_max"].to_numpy(dtype=float) == pytest.approx(want_max, rel=1e-12)


def test_field_strength_of_a_lone_runner_is_missing():
    frame = pd.DataFrame(
        {
            "race_id": ["solo", "pair", "pair"],
            "horse_id": ["a", "b", "c"],
            "mu_before": [5.0, 1.0, 4.0],
        }
    )
    out = field_strength(frame, rating_col="mu_before")
    assert out.loc[0, ["field_mean", "field_max"]].isna().all()
    assert out.loc[1, "field_mean"] == pytest.approx(4.0)
    assert out.loc[2, "field_max"] == pytest.approx(1.0)


# ---------------------------------------------------------------- effective resistance


def test_effective_resistance_of_a_path_is_its_length():
    results = results_of([("r1", before(20), ["a", "b"]), ("r2", before(10), ["c", "b"])])
    graph = comparison_graph(results, before_utc=T)
    assert isinstance(graph, Graph)
    assert_resistance(effective_resistance(graph, "a", "c"), 2.0)
    assert_resistance(effective_resistance(graph, "a", "b"), 1.0)


@pytest.mark.parametrize("n", [2, 4, 7, 11])
def test_effective_resistance_of_a_longer_path_and_a_cycle(n):
    """A path of n two-runner races has R_eff n end to end; closing it into a cycle of n + 1
    edges gives k(n + 1 - k)/(n + 1) between horses k steps apart."""
    names = [f"h{i}" for i in range(n + 1)]
    path = [(f"p{i}", before(100 - i), [names[i], names[i + 1]]) for i in range(n)]
    graph = comparison_graph(results_of(path), before_utc=T)
    assert_resistance(effective_resistance(graph, names[0], names[n]), float(n))
    cycle = path + [("close", before(1), [names[n], names[0]])]
    graph = comparison_graph(results_of(cycle), before_utc=T)
    for k in range(1, n + 1):
        want = k * (n + 1 - k) / (n + 1)
        assert_resistance(effective_resistance(graph, names[0], names[k]), want)


@pytest.mark.parametrize("seed", [11, 12, 13, 14])
def test_effective_resistance_matches_the_laplacian_pseudo_inverse(seed):
    results, edges, horses = random_pair_graph(seed)
    graph = comparison_graph(results, before_utc=T)
    rng = np.random.default_rng(seed + 100)
    pairs = [tuple(rng.choice(horses, 2, replace=False)) for _ in range(30)]
    pairs += [(horses[0], horses[-1])]  # across the two components
    for a, b in pairs:
        want = reference_resistance(edges, horses, a, b)
        assert_resistance(effective_resistance(graph, a, b), want)
        assert_resistance(effective_resistance(graph, b, a), want)


def test_horses_that_met_are_more_comparable_than_horses_that_did_not():
    """a met b; a reaches d only through b and c. In a race of a and b the runners are more
    comparable (lower mean R_eff) than in a race of a and d."""
    results = results_of(
        [
            ("r1", before(30), ["a", "b"]),
            ("r2", before(20), ["b", "c"]),
            ("r3", before(10), ["c", "d", "e"]),
        ]
    )
    graph = comparison_graph(results, before_utc=T)
    met = effective_resistance(graph, "a", "b")
    not_met = effective_resistance(graph, "a", "d")
    assert math.isfinite(met) and math.isfinite(not_met)
    assert met < not_met
    card = pd.DataFrame(
        {"race_id": ["met", "met", "not_met", "not_met"], "horse_id": ["a", "b", "a", "d"]}
    )
    comp = race_comparability(card, graph)
    assert comp["met"] < comp["not_met"]


def test_meeting_again_makes_two_horses_more_comparable():
    """Var(s_a - s_b) = sigma^2 R_eff: a second meeting is more information, so R_eff falls."""
    once = comparison_graph(results_of([("r1", before(9), ["a", "b"])]), before_utc=T)
    twice = comparison_graph(
        results_of([("r1", before(9), ["a", "b"]), ("r2", before(5), ["b", "a"])]), before_utc=T
    )
    assert effective_resistance(twice, "a", "b") < effective_resistance(once, "a", "b")


def test_runners_of_one_field_are_all_connected():
    horses = [f"h{i}" for i in range(8)]
    graph = comparison_graph(results_of([("r1", before(3), horses)]), before_utc=T)
    values = [effective_resistance(graph, a, b) for a, b in itertools.combinations(horses, 2)]
    assert all(math.isfinite(v) and v > 0 for v in values)
    assert max(values) == pytest.approx(min(values))  # one field: every pair alike


# ---------------------------------------------------------------- before t


def test_the_graph_uses_only_races_before_t():
    """a-b ran a minute before t; b-c ran exactly at t; c-d ran after t; e ran only after t.
    Only a-b is in the graph: adding or changing races at or after t changes nothing."""
    early = [("r1", T - timedelta(minutes=1), ["a", "b"])]
    late = [
        ("r2", T, ["b", "c"]),
        ("r3", T + timedelta(days=2), ["c", "d", "a"]),
        ("r4", T + timedelta(days=3), ["e", "b"]),
    ]
    graph = comparison_graph(results_of(early + late), before_utc=T)
    assert_resistance(effective_resistance(graph, "a", "b"), 1.0)
    for x, y in [("b", "c"), ("a", "c"), ("a", "d"), ("b", "e"), ("c", "d")]:
        assert_resistance(effective_resistance(graph, x, y), math.inf)

    later = comparison_graph(results_of(early + late), before_utc=T + timedelta(days=10))
    assert math.isfinite(effective_resistance(later, "a", "e"))  # the races exist and count later


@pytest.mark.parametrize("seed", [21, 22, 23])
def test_races_at_or_after_t_never_change_the_graph(seed):
    results, edges, horses = random_pair_graph(seed)
    only_before = results[results["race_utc"] < pd.Timestamp(T)]
    full = comparison_graph(results, before_utc=T)
    trimmed = comparison_graph(only_before, before_utc=T)
    probes = horses + [f"h:{seed}:late:{i}" for i in range(4)]
    for a, b in itertools.combinations(probes[::2], 2):
        want = reference_resistance(edges, horses, a, b)
        assert_resistance(effective_resistance(full, a, b), want)
        assert_resistance(effective_resistance(trimmed, a, b), want)


# ---------------------------------------------------------------- disconnected


def test_disconnected_horses_have_infinite_resistance():
    results = results_of(
        [
            ("r1", before(40), ["a", "b"]),
            ("r2", before(30), ["b", "c"]),
            ("r3", before(20), ["x", "y"]),
        ]
    )
    graph = comparison_graph(results, before_utc=T)
    assert_resistance(effective_resistance(graph, "a", "c"), 2.0)
    assert_resistance(effective_resistance(graph, "x", "y"), 1.0)
    for a, b in [("a", "x"), ("c", "y"), ("a", "never_ran"), ("never_ran", "b")]:
        assert_resistance(effective_resistance(graph, a, b), math.inf)


def test_an_empty_graph_gives_infinite_resistance():
    results = results_of([("r1", T + timedelta(days=1), ["a", "b"])])
    graph = comparison_graph(results, before_utc=T)
    assert_resistance(effective_resistance(graph, "a", "b"), math.inf)


# ---------------------------------------------------------------- race comparability


@pytest.mark.parametrize("seed", [31, 32])
def test_race_comparability_is_the_mean_pairwise_resistance(seed):
    """A card of races drawn from the graph's horses (and some it never saw): each race's value is
    the mean over its unordered pairs of R_eff, inf when any pair is unconnected."""
    results, edges, horses = random_pair_graph(seed)
    graph = comparison_graph(results, before_utc=T)
    rng = np.random.default_rng(seed + 7)
    first = [h for h in horses if h.split(":")[2] == "0"]  # one component
    pool = horses + [f"h:{seed}:new:{i}" for i in range(3)]
    rows = []
    for r in range(10):
        source = first if r < 5 else pool  # the first five races are fully connected
        size = int(rng.integers(2, min(8, len(source)) + 1))
        for h in rng.choice(source, size, replace=False):
            rows.append({"race_id": f"card:{r:02d}", "horse_id": str(h)})
    card = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out = race_comparability(card, graph)
    assert isinstance(out, pd.Series)
    assert set(out.index) == set(card["race_id"])
    for rid, runners in card.groupby("race_id")["horse_id"]:
        values = [
            reference_resistance(edges, horses, a, b) for a, b in itertools.combinations(runners, 2)
        ]
        want = math.inf if any(math.isinf(v) for v in values) else float(np.mean(values))
        if rid < "card:05":
            assert math.isfinite(want)
        got = float(out[rid])
        if math.isinf(want):
            assert math.isinf(got) and got > 0, (rid, got)
        else:
            assert got == pytest.approx(want, rel=1e-6)


def test_race_comparability_of_runners_that_never_met_is_infinite():
    results = results_of([("r1", before(10), ["a", "b"]), ("r2", before(5), ["c", "d"])])
    graph = comparison_graph(results, before_utc=T)
    card = pd.DataFrame(
        {
            "race_id": ["apart"] * 2 + ["debut"] * 2 + ["met"] * 2,
            "horse_id": ["a", "c", "a", "newcomer", "c", "d"],
        }
    )
    out = race_comparability(card, graph)
    assert math.isinf(out["apart"]) and out["apart"] > 0
    assert math.isinf(out["debut"]) and out["debut"] > 0
    assert out["met"] == pytest.approx(1.0)


def test_the_competition_terms_are_exported():
    names = ("field_strength", "comparison_graph", "effective_resistance", "race_comparability")
    assert set(names) <= set(competition.__all__)
