"""The model boundary (#417): only explicitly approved columns enter a feature frame.

`core.store.asof.read_asof` returns every column of a table outside its label exclusions, so a
column added to a known table would reach a model by default. At this boundary a column is a
feature at t only when both configs say so:

- the input inventory (`config/model_inputs.yaml`, `research.coverage.load_inputs`) names it in a
  supply `{source, table, column}` of an input whose role is `feature_at_t`, and
- the availability rules (`config/availability.yaml`, `core.store.asof.load_availability`) let
  the table into the views and do not make the column a label (`Availability.is_label`).

Nothing else is approved, and no list is kept in code. A supply's `filter` (a pool, say) does not
narrow the approval: the boundary is per column. `feature_view` reads a table at t through
`read_asof` and returns only the caller's keys and approved features; a column neither config
names is `UnclassifiedColumn`, a classified one that is not a feature at t `FeatureNotApproved`.
`approximations` names the documented approximations a column's availability carries, from the
availability rules only. docs/specs/as-of.md, "The model boundary", says how a study adopts it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime

import pandas as pd

from horsetrack.core.store.asof import (
    Availability,
    NoAvailabilityRule,
    load_availability,
    read_asof,
)
from horsetrack.countries.japan.research.coverage import ModelInput, load_inputs

COUNTRY = "japan"
FEATURE_AT_T = "feature_at_t"
# a card column taken from the results page, in the views from 00:00 JST of the race date: the
# accepted leak of docs/specs/as-of.md (an availability rule's `results_derived` list)
RESULTS_DERIVED = "results-derived"
# a gated column: in the view from an anchor + offset (post_utc - 60 min, say) that stands in for
# its unrecorded publication time (an availability rule's `gated_columns`)
PUBLICATION_PROXY = "publication-proxy"

__all__ = [
    "PUBLICATION_PROXY",
    "RESULTS_DERIVED",
    "FeatureNotApproved",
    "UnclassifiedColumn",
    "approved_features",
    "approximations",
    "feature_view",
]


class FeatureNotApproved(ValueError):
    """A requested feature is not approved at t: a label, or an input that is no feature at t."""


class UnclassifiedColumn(FeatureNotApproved):
    """A requested feature that neither the inventory nor the availability rules name."""


def _rules(rules: Mapping[str, Availability] | None) -> Mapping[str, Availability]:
    return load_availability(COUNTRY) if rules is None else rules


def _rule(table: str, rules: Mapping[str, Availability]) -> Availability:
    rule = rules.get(table)
    if rule is None:
        raise NoAvailabilityRule(f"table {table!r} has no availability rule")
    return rule


def _supplied(table: str, source: str, inputs: Iterable[ModelInput]) -> dict[str, set[str]]:
    """Each column of `table` the inventory supplies from `source`, with the roles it has."""
    roles: dict[str, set[str]] = {}
    for inp in inputs:
        for s in inp.supplies:
            if s.source == source and s.table == table:
                roles.setdefault(s.column, set()).add(inp.role)
    return roles


def approved_features(
    table: str,
    *,
    source: str = "jra",
    rules: Mapping[str, Availability] | None = None,
    inputs: Sequence[ModelInput] | None = None,
) -> frozenset[str]:
    """The columns of `table` approved as features at t for `source`: supplied by a
    `feature_at_t` input of the inventory, in a table that enters the views, and no label column
    of its rule. A table that never enters a view has none."""
    rule = _rule(table, _rules(rules))
    if not rule.enters_view:
        return frozenset()
    supplied = _supplied(table, source, load_inputs() if inputs is None else inputs)
    return frozenset(
        c for c, roles in supplied.items() if FEATURE_AT_T in roles and not rule.is_label(c)
    )


def feature_view(
    con,
    table: str,
    at_utc: datetime,
    *,
    features: Sequence[str],
    keys: Sequence[str] = (),
    source: str = "jra",
    rules: Mapping[str, Availability] | None = None,
    inputs: Sequence[ModelInput] | None = None,
) -> pd.DataFrame:
    """The as-of view of `table` at `at_utc` (`read_asof`) with exactly `[*keys, *features]`.

    `keys` are the caller's identifiers, returned as they are; a label column as a key is a
    ValueError. Every feature must be approved (`approved_features`), or nothing is read: a column
    neither config names is `UnclassifiedColumn`, a classified one that is not a feature at t
    `FeatureNotApproved`; the message names every refused column as `table.column`."""
    if isinstance(features, str) or isinstance(keys, str):
        raise TypeError("features and keys are sequences of column names, not a string")
    features, keys = list(features), list(keys)
    rules = _rules(rules)
    inputs = load_inputs() if inputs is None else inputs
    rule = _rule(table, rules)
    labelled = [k for k in keys if rule.is_label(k)]
    if labelled:
        named = ", ".join(f"{table}.{k}" for k in labelled)
        raise ValueError(f"label columns cannot enter a feature frame as keys: {named}")
    both = sorted(set(keys) & set(features))
    if both or len(set(keys)) != len(keys) or len(set(features)) != len(features):
        raise ValueError(f"{table}: a column is asked for twice (keys {keys}, features {features})")

    approved = approved_features(table, source=source, rules=rules, inputs=inputs)
    supplied = _supplied(table, source, inputs)
    refused = [f for f in features if f not in approved]
    unclassified = [f for f in refused if f not in supplied and not rule.is_label(f)]
    if refused:
        parts = []
        if unclassified:
            parts.append(
                "unclassified (in no inventory supply of "
                f"{source} and no label column): {', '.join(f'{table}.{f}' for f in unclassified)}"
            )
        classified = [f for f in refused if f not in unclassified]
        if classified:
            parts.append(
                "classified, not a feature at t (a label, or supplied only as a label or a "
                f"conditioner): {', '.join(f'{table}.{f}' for f in classified)}"
            )
        error = UnclassifiedColumn if unclassified else FeatureNotApproved
        raise error(f"refused at the model boundary: {'; '.join(parts)}")

    frame = read_asof(con, table, at_utc, rules=rules)
    missing = [c for c in [*keys, *features] if c not in frame.columns]
    if missing:
        raise ValueError(f"the view of {table!r} has no columns {missing}")
    return frame[[*keys, *features]]


def approximations(
    table: str, columns: Iterable[str], *, rules: Mapping[str, Availability] | None = None
) -> dict[str, frozenset[str]]:
    """Per column, the documented approximations its availability carries, read from the
    availability rules only: RESULTS_DERIVED for a column of the rule's `results_derived` list,
    PUBLICATION_PROXY for a gated column; an empty set for neither."""
    rule = _rule(table, _rules(rules))
    out = {}
    for column in columns:
        tags = set()
        if column in rule.results_derived:
            tags.add(RESULTS_DERIVED)
        if column in rule.gated:
            tags.add(PUBLICATION_PROXY)
        out[column] = frozenset(tags)
    return out
