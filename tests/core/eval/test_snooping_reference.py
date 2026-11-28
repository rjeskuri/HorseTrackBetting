"""#218 review: a reference variant leaves the SPA / StepM family only where the frozen spec says.

Module under test: `horsetrack.core.eval.snooping` (`spa`, `stepm`, `reference_variants=`).

Contract pinned here: `reference_variants` names variants that are recorded in the family but are
not rules of its family test. A name is accepted only when the family's frozen spec
(`Registry.freeze_family(family_id, spec)`) names it as a reference: it is a key of
`spec["variants"]` and is absent from `spec["inference"]["family"]["rules"]`, the rules the
frozen spec registers for the family test. This is the layout `studies.form_bands.build_spec`
already froze for the real #218 family (P0 among `variants`, the 11 others as `rules`), so its
spec hash does not change. Anything else raises `FamilySizeMismatch` before a test is run: an
ordinary candidate (one of the frozen rules) passed as a reference, a family whose frozen spec
declares no family-test rules, a family that is not frozen at all, or a name the frozen spec does
not list among its variants. Without `reference_variants` nothing changes (test_snooping.py).

Data: synthetic per-race losses under a tmp registry; the values do not matter, only whether
the call is accepted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.registry import FoldMetric, Registry, RunRecord
from horsetrack.core.eval.snooping import FamilySizeMismatch, spa, stepm

FAMILY = "P218-reference"
REFERENCE = "P0"
RULES = ["r1", "r2", "r3"]
VARIANTS = [REFERENCE, *RULES]


def spec_with_rules(rules: list[str] | None, variants: list[str] = VARIANTS) -> dict:
    spec = {"family_id": FAMILY, "variants": {v: f"variant {v}" for v in variants}}
    if rules is not None:
        spec["inference"] = {"family": {"tests": ["SPA", "StepM"], "rules": list(rules)}}
    return spec


def record(variant: str) -> RunRecord:
    return RunRecord(
        hypothesis_id="E218",
        family_id=FAMILY,
        variant=variant,
        ticket=218,
        git_sha="0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c",
        data_builds={"results": "synthetic-build"},
        sources=("synthetic",),
        fold_scheme="walk-forward-3",
        components={"rule": variant},
        params={"rule": variant},
        metrics=(FoldMetric("development", "delta_ll_vs_v0", 0.001, -0.01, None, 300),),
        status="scored",
    )


def make_registry(tmp_path: Path, spec: dict | None) -> Registry:
    registry = Registry(
        tmp_path / "registry" / "japan.jsonl", holdout_log=tmp_path / "holdout_log.jsonl"
    )
    if spec is not None:
        registry.freeze_family(FAMILY, spec)
    for variant in VARIANTS:
        registry.append(record(variant))
    return registry


def losses_for(columns: list[str], seed: int = 218):
    rng = np.random.default_rng(seed)
    n_meetings, races = 24, 6
    index = pd.Index([f"M{m:02d}-R{r}" for m in range(n_meetings) for r in range(races)])
    clusters = pd.Series(np.repeat([f"M{m:02d}" for m in range(n_meetings)], races), index=index)
    order = pd.Series(
        np.repeat(pd.date_range("2025-01-04", periods=n_meetings, freq="7D"), races), index=index
    )
    benchmark = pd.Series(rng.normal(2.0, 0.3, len(index)), index=index)
    losses = pd.DataFrame(
        {c: benchmark + rng.normal(0.0, 0.05, len(index)) for c in columns}, index=index
    )
    return benchmark, losses, clusters, order


def call(test: str, registry: Registry, columns: list[str], reference: tuple[str, ...]):
    benchmark, losses, clusters, order = losses_for(columns)
    kwargs = dict(
        clusters=clusters,
        order=order,
        block_size=1,
        n_boot=100,
        seed=218,
        registry=registry,
        family_id=FAMILY,
        reference_variants=reference,
    )
    if test == "spa":
        return spa(benchmark, losses, **kwargs)
    return stepm(benchmark, losses, level=0.05, **kwargs)


@pytest.mark.parametrize("test", ["spa", "stepm"])
def test_the_reference_the_frozen_spec_names_leaves_the_family_test(tmp_path, test):
    registry = make_registry(tmp_path, spec_with_rules(RULES))
    result = call(test, registry, RULES, (REFERENCE,))
    assert result.family_size == 4
    if test == "spa":
        assert result.n_rules == len(RULES)


@pytest.mark.parametrize("test", ["spa", "stepm"])
@pytest.mark.parametrize("candidate", RULES)
def test_an_ordinary_candidate_passed_as_a_reference_is_refused(tmp_path, test, candidate):
    registry = make_registry(tmp_path, spec_with_rules(RULES))
    kept = [r for r in RULES if r != candidate]
    with pytest.raises(FamilySizeMismatch):
        call(test, registry, [REFERENCE, *kept], (candidate,))
    # also when the true reference is passed beside it
    with pytest.raises(FamilySizeMismatch):
        call(test, registry, kept, (REFERENCE, candidate))


@pytest.mark.parametrize("test", ["spa", "stepm"])
@pytest.mark.parametrize(
    "spec",
    [
        spec_with_rules(None),  # the frozen spec registers no family-test rules
        None,  # the family is not frozen at all
        spec_with_rules(RULES, variants=RULES),  # P0 is not a variant the spec names
    ],
    ids=["no-rules-in-spec", "not-frozen", "reference-not-a-spec-variant"],
)
def test_a_reference_the_frozen_spec_does_not_name_is_refused(tmp_path, test, spec):
    registry = make_registry(tmp_path, spec)
    with pytest.raises(FamilySizeMismatch):
        call(test, registry, RULES, (REFERENCE,))


def test_without_a_reference_an_unfrozen_family_is_tested_as_before(tmp_path):
    registry = make_registry(tmp_path, None)
    result = call("spa", registry, VARIANTS, ())
    assert result.n_rules == len(VARIANTS)
