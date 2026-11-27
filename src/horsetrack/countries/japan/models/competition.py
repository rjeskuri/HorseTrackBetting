"""Competition terms for the C2 per-horse model (#94, epic #73; 08 §3.3; 05 §3.4, §4.2).

Opponent strength is implicit in Plackett-Luce, so it enters in two ways.

This module builds the terms only. Their measurement (each term's ΔLL with its CI, standalone and
at the final price, its leaderboard member row, and R_eff against the meta-model's residual by
bucket) is #531's study, split from #94's outline 4 under the size cap.

Field strength. `field_strength(frame, *, rating_col)` gives each runner the mean and max of
rating_col over the other runners of its race_id, never the runner itself (a lone runner, or one
whose race has no other rated runner, gets NaN). `interaction_columns(frame, *, rating_col)` adds
field_x_rating = field_mean * the runner's own rating, the interaction "this horse's rating in a
strong field" that #92's `models.ability.AbilityModel(features=[..., "field_x_rating"])` reads as
an ordinary feature column. Where field_mean is undefined field_x_rating is 0: such a runner's
race has no second runner, so it adds nothing to a Plackett-Luce likelihood, and AbilityModel
refuses non-finite features. rating_col must be a before-the-race rating (`models.rating`'s
mu_before), so the terms read nothing from the race itself.

Comparability. `comparison_graph(results, *, before_utc)` joins the horses who ran in the same race
strictly before before_utc (rows at or after it are never read). results holds the starters, as
the store and `models.rating` give them: race_id, race_utc (tz-aware UTC), horse_id. A race of n
starters joins each of its n(n - 1)/2 pairs with weight 1/(n - 1), the Plackett-Luce information
weighting of rank breaking (Hajek, Oh and Xu 2014), so a two-runner race is one unit edge and every
race gives each starter one unit of degree; meetings add. The Graph holds the sparse Laplacian, the
horse index and the connected components (`scipy.sparse.csgraph`).

race_utc must be each race's actual post time in UTC: the cutoff is only as point-in-time as the
stamps it compares. `studies.rating.read_results()` stamps a race at midnight JST of race_date plus
race_no minutes, a stand-in that keeps a day's races in order but is not a post time. On such
stamps only a before_utc at a JST day boundary is safe; an intraday before_utc can admit same-day
races that had not yet run at that time. Callers that cut intraday (#531, #98) must supply real
post times.

`effective_resistance(graph, a, b)` is (e_a - e_b)' L^+ (e_a - e_b), so Var(s_a - s_b) =
sigma^2 * R_eff(a, b) (05 §3.4): horses who met, or met through many short paths, are comparable.
Horses in different components, or a horse the graph never saw, give +inf, never an error. It is
solved per connected component: the component's Laplacian grounded at one node is positive
definite, factorised once by sparse LU (`scipy.sparse.linalg.splu`) and cached on the Graph, and
R(a, b) = G_aa + G_bb - 2 G_ab for the grounded inverse G (0 at the ground). No dense inverse of
the store's graph is formed.

`race_comparability(frame, graph)` is the export for the meld's precision weighting (E41, #98):
per race_id of frame (race_id, horse_id), the mean effective resistance over the race's unordered
runner pairs, +inf when any pair is unconnected or a runner is unknown, NaN for a race of one
runner. A higher value means a less comparable field; 05 H12 (tested in #98) is that the model's
error against the market rises with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

FIELD_COLUMNS = ["field_mean", "field_max"]
INTERACTION = "field_x_rating"

__all__ = [
    "INTERACTION",
    "Graph",
    "comparison_graph",
    "effective_resistance",
    "field_strength",
    "interaction_columns",
    "race_comparability",
]


# ---------------------------------------------------------------- field strength


def field_strength(frame: pd.DataFrame, *, rating_col: str) -> pd.DataFrame:
    """field_mean and field_max of rating_col over each runner's other runners in its race_id,
    on frame's index. A missing rating counts as no runner."""
    x = pd.to_numeric(frame[rating_col]).astype(float)
    race = frame["race_id"]
    group = x.groupby(race)
    total = group.transform("sum")
    valid = group.transform("count")
    own = x.notna()
    others = valid - own.astype(int)
    mean = (total - x.fillna(0.0)) / others.where(others > 0)

    top = group.transform("max")
    at_top = x.eq(top)
    n_top = at_top.groupby(race).transform("sum")
    second = x.where(~at_top).groupby(race).transform("max")
    best = top.where(~(at_top & n_top.eq(1)), second)
    best = best.where(others > 0)
    return pd.DataFrame({"field_mean": mean, "field_max": best}, index=frame.index)[FIELD_COLUMNS]


def interaction_columns(frame: pd.DataFrame, *, rating_col: str) -> pd.DataFrame:
    """field_mean, field_max and field_x_rating (field_mean * own rating, 0 where undefined), on
    frame's index, for `AbilityModel(features=[...])`."""
    out = field_strength(frame, rating_col=rating_col)
    own = pd.to_numeric(frame[rating_col]).astype(float)
    out[INTERACTION] = (out["field_mean"] * own).fillna(0.0)
    return out


# ---------------------------------------------------------------- comparison graph


@dataclass(frozen=True)
class Graph:
    """The comparison graph before a time: its Laplacian over `index` (horse ids, row order) and
    each horse's connected component. Component factorisations are cached on first use."""

    laplacian: sp.csr_matrix
    index: pd.Index
    component: np.ndarray
    before_utc: datetime
    _solvers: dict = field(default_factory=dict, repr=False, compare=False)

    def _solver(self, comp: int):
        """(the component's nodes, the LU of its Laplacian grounded at nodes[0], or None for a
        single node)."""
        if comp not in self._solvers:
            nodes = np.flatnonzero(self.component == comp)
            lu = None
            if len(nodes) > 1:
                rest = nodes[1:]
                reduced = self.laplacian[rest][:, rest].tocsc()
                lu = splu(reduced, permc_spec="MMD_AT_PLUS_A")
            self._solvers[comp] = (nodes, lu)
        return self._solvers[comp]

    def _resistances(self, codes: np.ndarray) -> np.ndarray:
        """The pairwise effective resistance among the horses at codes (all in one component)."""
        nodes, lu = self._solver(int(self.component[codes[0]]))
        m = len(codes)
        if lu is None:
            return np.zeros((m, m))
        local = np.searchsorted(nodes, codes) - 1  # -1: the ground
        free = np.flatnonzero(local >= 0)
        green = np.zeros((m, m))
        if len(free):
            rhs = np.zeros((len(nodes) - 1, len(free)))
            rhs[local[free], np.arange(len(free))] = 1.0
            sol = lu.solve(rhs)
            green[np.ix_(free, free)] = sol[local[free]]
        diag = np.diag(green)
        return diag[:, None] + diag[None, :] - green - green.T


def comparison_graph(results: pd.DataFrame, *, before_utc: datetime) -> Graph:
    """The graph of horses who ran in the same race strictly before before_utc."""
    cutoff = pd.Timestamp(before_utc)
    rows = results.loc[results["race_utc"] < cutoff, ["race_id", "horse_id"]]
    rows = rows.dropna().drop_duplicates()
    horses = pd.Index(pd.unique(rows["horse_id"].to_numpy(dtype=object)))
    n = len(horses)
    race = pd.factorize(rows["race_id"])[0]
    horse = horses.get_indexer(rows["horse_id"].to_numpy(dtype=object))
    size = np.bincount(race) if len(race) else np.zeros(0, dtype=int)
    keep = size[race] > 1
    weight = 1.0 / (size[race[keep]] - 1)
    incidence = sp.csr_matrix((np.sqrt(weight), (race[keep], horse[keep])), shape=(len(size), n))
    adjacency = (incidence.T @ incidence).tolil()
    adjacency.setdiag(0.0)
    adjacency = adjacency.tocsr()
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = (sp.diags(degree) - adjacency).tocsr()
    _, component = connected_components(adjacency, directed=False)
    return Graph(laplacian=laplacian, index=horses, component=component, before_utc=before_utc)


def effective_resistance(graph: Graph, a: str, b: str) -> float:
    """R_eff(a, b) on graph; +inf across components or for a horse the graph never saw."""
    codes = graph.index.get_indexer([a, b])
    if (codes < 0).any() or graph.component[codes[0]] != graph.component[codes[1]]:
        return math.inf
    if codes[0] == codes[1]:
        return 0.0
    return float(graph._resistances(codes)[0, 1])


def race_comparability(frame: pd.DataFrame, graph: Graph) -> pd.Series:
    """Per race_id of frame: the mean R_eff over its unordered runner pairs (+inf when any pair is
    unconnected, NaN for a single runner)."""
    out = {}
    for rid, horses in frame.groupby("race_id", sort=False)["horse_id"]:
        codes = graph.index.get_indexer(pd.unique(horses.to_numpy(dtype=object)))
        if len(codes) < 2:
            out[rid] = math.nan
        elif (codes < 0).any() or len(np.unique(graph.component[codes])) > 1:
            out[rid] = math.inf
        else:
            r = graph._resistances(codes)
            out[rid] = float(r[np.triu_indices(len(codes), k=1)].mean())
    return pd.Series(out, dtype=float, name="race_comparability")
