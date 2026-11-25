"""The paired cluster bootstrap as signed in docs/problem-statement.md, section 3 (Metric), D7.

Section 3 signs a paired bootstrap: the model and the baseline are resampled with the same
indices, so the interval is on the per-race difference. Passing that difference (the `per_race`
of `horsetrack.core.eval.scoring.gain`) and resampling it is the same as resampling both sides
with shared indices. D7 signs the resampling unit as the meeting (venue x day) and the level as 95%,
one-sided. Neither is defaulted here: callers pass them from the country's evaluation config.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

# Upper bound on the replicate-by-cluster index array built at once, to keep memory flat.
_MAX_DRAWS_PER_CHUNK = 2_000_000


@dataclass(frozen=True)
class Interval:
    """The mean per-race difference and its one-sided bootstrap bounds.

    `lo` is the (1 - level) quantile of the bootstrap distribution of the mean: the one-sided
    lower bound a gain claim rests on (D7, one-sided). `hi` is the `level` quantile, the mirror
    one-sided upper bound, reported for symmetry; [lo, hi] is not a two-sided interval at `level`.
    `n_clusters` is the number of clusters resampled.
    """

    mean: float
    lo: float
    hi: float
    n_clusters: int


def paired_bootstrap(
    diff: pd.Series, clusters: pd.Series, *, n_boot: int, level: float, seed: int
) -> Interval:
    """One-sided bootstrap bounds on the mean per-race difference, resampling whole clusters.

    `diff` is the per-race paired difference (model against baseline, the same races; section 3),
    indexed by race id. `clusters` maps race id to its resampling unit; D7 signs the meeting
    (venue x day). It is matched to `diff` by label and may list races not in `diff`; a race in
    `diff` without a cluster raises a ValueError.

    Each replicate draws as many clusters as there are, with replacement, and takes the per-race
    mean over the races of the drawn clusters. `mean` is the per-race mean of `diff` itself.
    `lo` is the (1 - level) quantile of the replicates (the one-sided lower bound at `level`,
    D7: 95%, one-sided) and `hi` the `level` quantile (the mirror one-sided upper bound).

    `n_boot` (the number of replicates, >= 1), `level` (in (0, 1)) and `seed` are keyword-only
    and have no defaults: the level is an owner decision (D7) and the others set precision and
    reproducibility.
    """
    if isinstance(level, bool) or not isinstance(level, Real) or not 0.0 < level < 1.0:
        raise ValueError(f"level must be a fraction in (0, 1), got {level!r}")
    if isinstance(n_boot, bool) or not isinstance(n_boot, Integral) or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}")
    if len(diff) == 0:
        raise ValueError("diff is empty")
    if not diff.index.is_unique:
        raise ValueError("diff has duplicate race ids")
    if not clusters.index.is_unique:
        raise ValueError("clusters has duplicate race ids")
    if diff.isna().any():
        raise ValueError("diff has missing values")
    missing = diff.index.difference(clusters.index)
    if len(missing):
        raise ValueError(f"{len(missing)} race(s) have no cluster, e.g. {missing[0]!r}")
    cluster_of = clusters.reindex(diff.index)
    if cluster_of.isna().any():
        raise ValueError("a race in diff has a missing cluster")

    values = diff.to_numpy(dtype=float)
    codes, _ = pd.factorize(cluster_of, sort=True)
    n_clusters = int(codes.max()) + 1
    sums = np.bincount(codes, weights=values, minlength=n_clusters)
    sizes = np.bincount(codes, minlength=n_clusters).astype(float)

    rng = np.random.default_rng(seed)
    chunk = max(1, _MAX_DRAWS_PER_CHUNK // n_clusters)
    means = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        drawn = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
        means[start:stop] = sums[drawn].sum(axis=1) / sizes[drawn].sum(axis=1)

    return Interval(
        mean=float(values.mean()),
        lo=float(np.quantile(means, 1.0 - level)),
        hi=float(np.quantile(means, level)),
        n_clusters=n_clusters,
    )
