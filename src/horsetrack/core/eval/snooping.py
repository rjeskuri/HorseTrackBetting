"""Max-statistic tests for a family of rules: Hansen's SPA and Romano-Wolf StepM (08 §5.2, §8 A2).

Each rule's per-race loss (lower is better) is compared with a benchmark on the same races. The
races are summed to one row per cluster (the resampling unit, e.g. a meeting) and the cluster
series, in time order, goes to ``arch.bootstrap``'s stationary bootstrap. So the bootstrap draws
whole clusters, and the result depends on the per-cluster sums only.

The rules tested must be exactly the family's non-failed variants in the run registry, and the
result reports the registry family size, failed variants included, so a family cannot be tested
on a hand-picked subset of what was tried. The one exception is ``reference_variants``: variants
the family's frozen spec names as references (a re-run of an earlier result the rules are compared
with, not a rule of the test). Each must be recorded in the family, and it leaves the test only.

The cluster unit (through ``clusters``), ``block_size``, ``n_boot`` and StepM's ``level`` are owner
decisions (08 §5.2, §9.6, §9.9) and have no defaults: callers pass them from the country's
evaluation config.
"""

from collections.abc import Collection
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd
from arch.bootstrap import SPA, StepM

from horsetrack.core.eval.registry import Registry


class FamilySizeMismatch(ValueError):
    """The loss columns are not exactly the family's non-failed variants in the registry."""


@dataclass(frozen=True)
class SPAResult:
    """SPA p-values (arch's lower, consistent and upper) for "no rule beats the benchmark".

    `pvalue` is the consistent p-value. `n_rules` is the number of rules tested, `family_size`
    the registry family size (failed variants included) and `n_clusters` the bootstrap's length.
    """

    pvalue: float
    pvalue_lower: float
    pvalue_upper: float
    n_rules: int
    family_size: int
    n_clusters: int


@dataclass(frozen=True)
class StepMResult:
    """The rules StepM finds superior to the benchmark, with the family size and cluster count."""

    superior: list[str]
    family_size: int
    n_clusters: int


def _cluster_codes(index: pd.Index, clusters: pd.Series, order: pd.Series) -> tuple:
    """Per-row cluster codes, numbered in time order, and the cluster labels in that order.

    Clusters are ordered by their earliest ``order`` value, ties broken by label.
    """
    if not index.is_unique:
        raise ValueError("losses have duplicate row ids")
    for name, series in (("clusters", clusters), ("order", order)):
        if not series.index.is_unique:
            raise ValueError(f"{name} has duplicate row ids")
        missing = index.difference(series.index)
        if len(missing):
            raise ValueError(f"{len(missing)} row(s) have no {name} value, e.g. {missing[0]!r}")
    cluster_of = clusters.reindex(index)
    when = order.reindex(index)
    if cluster_of.isna().any():
        raise ValueError("a row has a missing cluster")
    if when.isna().any():
        raise ValueError("a row has a missing order value")
    first = when.groupby(cluster_of.to_numpy()).min()
    labels = sorted(first.index, key=lambda label: (first[label], label))
    position = {label: k for k, label in enumerate(labels)}
    codes = cluster_of.map(position).to_numpy(dtype=np.int64)
    return codes, pd.Index(labels)


def _sums(values: np.ndarray, codes: np.ndarray, n_clusters: int) -> np.ndarray:
    """Per-cluster column sums, independent of the row order within a cluster.

    Each column's values are summed in sorted order inside each cluster, so the floating-point
    result is the same whatever order the rows arrive in.
    """
    if np.isnan(values).any():
        raise ValueError("losses have missing values")
    out = np.empty((n_clusters, values.shape[1]))
    for j in range(values.shape[1]):
        rows = np.lexsort((values[:, j], codes))
        starts = np.searchsorted(codes[rows], np.arange(n_clusters))
        out[:, j] = np.add.reduceat(values[rows, j], starts)
    return out


def cluster_losses(losses: pd.DataFrame, clusters: pd.Series, order: pd.Series) -> pd.DataFrame:
    """Sum the per-race losses of each column to one row per cluster, in time order.

    ``losses`` is indexed by race (or bet) id with one column per variant; ``clusters`` maps each
    id to its cluster and ``order`` to its date, matched by label. The result is indexed by the
    cluster labels, ordered by each cluster's earliest date with ties broken by label.
    """
    codes, labels = _cluster_codes(losses.index, clusters, order)
    sums = _sums(losses.to_numpy(dtype=float), codes, len(labels))
    return pd.DataFrame(sums, index=labels, columns=losses.columns)


def _check_family(
    losses: pd.DataFrame,
    registry: Registry,
    family_id: str,
    reference_variants: Collection[str] = (),
) -> int:
    """Refuse loss columns that are not the family's non-failed variants less its reference
    variants (each recorded in the family); return the family size."""
    columns = list(losses.columns)
    if len(set(columns)) != len(columns):
        raise FamilySizeMismatch("losses have duplicate columns")
    records = registry.records(family_id)
    reference = set(reference_variants)
    unknown = sorted(reference - {r.variant for r in records}, key=str)
    if unknown:
        raise FamilySizeMismatch(
            f"reference variants {unknown} are not recorded in family {family_id!r}"
        )
    expected = {r.variant for r in records if r.status != "failed" and r.variant not in reference}
    if not expected:
        raise FamilySizeMismatch(f"family {family_id!r} has no non-failed variant in the registry")
    if set(columns) != expected:
        extra = sorted(set(columns) - expected, key=str)
        absent = sorted(expected - set(columns), key=str)
        raise FamilySizeMismatch(
            f"loss columns differ from family {family_id!r}'s non-failed variants: "
            f"not in the family {extra}, missing {absent}"
        )
    return registry.family_size(family_id)


def _block_size(block_size: float) -> int:
    """The block size as an int, refusing a fractional value rather than rounding it."""
    if isinstance(block_size, bool) or not isinstance(block_size, Real):
        raise ValueError(f"block_size must be a whole number, got {block_size!r}")
    if not isinstance(block_size, Integral) and not float(block_size).is_integer():
        raise ValueError(f"block_size must be a whole number of clusters, got {block_size!r}")
    return int(block_size)


def _cluster_series(
    benchmark: pd.Series, losses: pd.DataFrame, clusters: pd.Series, order: pd.Series
) -> tuple[np.ndarray, pd.DataFrame]:
    """The benchmark and rule losses summed per cluster, in time order."""
    if not benchmark.index.is_unique:
        raise ValueError("benchmark has duplicate row ids")
    if not benchmark.index.sort_values().equals(losses.index.sort_values()):
        raise ValueError("benchmark and losses must cover the same rows")
    codes, labels = _cluster_codes(losses.index, clusters, order)
    values = np.column_stack(
        [benchmark.reindex(losses.index).to_numpy(dtype=float), losses.to_numpy(dtype=float)]
    )
    sums = _sums(values, codes, len(labels))
    return sums[:, 0], pd.DataFrame(sums[:, 1:], index=labels, columns=losses.columns)


def spa(
    benchmark: pd.Series,
    losses: pd.DataFrame,
    *,
    clusters: pd.Series,
    order: pd.Series,
    block_size: float,
    n_boot: int,
    seed: int,
    registry: Registry,
    family_id: str,
    reference_variants: Collection[str] = (),
) -> SPAResult:
    """Hansen's SPA test of "no rule in the family beats the benchmark", over whole clusters.

    ``benchmark`` and ``losses`` are per-race losses (lower is better) on the same race ids.
    Both are summed per cluster (``cluster_losses``) and arch's stationary bootstrap, with mean
    block length ``block_size`` clusters and ``n_boot`` replicates seeded by ``seed``, runs on
    that series. The loss columns must be exactly ``family_id``'s non-failed variants in
    ``registry`` less ``reference_variants``, each recorded in the family (else
    ``FamilySizeMismatch``).
    """
    family_size = _check_family(losses, registry, family_id, reference_variants)
    bench, rules = _cluster_series(benchmark, losses, clusters, order)
    test = SPA(
        bench,
        rules,
        block_size=_block_size(block_size),
        reps=n_boot,
        bootstrap="stationary",
        seed=seed,
    )
    test.compute()
    p = test.pvalues
    return SPAResult(
        pvalue=float(p["consistent"]),
        pvalue_lower=float(p["lower"]),
        pvalue_upper=float(p["upper"]),
        n_rules=rules.shape[1],
        family_size=family_size,
        n_clusters=len(rules),
    )


def stepm(
    benchmark: pd.Series,
    losses: pd.DataFrame,
    *,
    clusters: pd.Series,
    order: pd.Series,
    block_size: float,
    n_boot: int,
    seed: int,
    level: float,
    registry: Registry,
    family_id: str,
    reference_variants: Collection[str] = (),
) -> StepMResult:
    """Romano-Wolf StepM: the rules that beat the benchmark at family-wise error ``level``.

    The inputs, the cluster series and the registry check are as for ``spa``.
    """
    family_size = _check_family(losses, registry, family_id, reference_variants)
    bench, rules = _cluster_series(benchmark, losses, clusters, order)
    test = StepM(
        bench,
        rules,
        size=level,
        block_size=_block_size(block_size),
        reps=n_boot,
        bootstrap="stationary",
        seed=seed,
    )
    test.compute()
    return StepMResult(
        superior=[str(name) for name in test.superior_models],
        family_size=family_size,
        n_clusters=len(rules),
    )
