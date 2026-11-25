"""`horsetrack japan win5-study freeze|score`: the WIN5 carryover study (ticket #135).

The spec is config/win5_study.yaml (docs/specs/win5-carryover-study.md). `freeze` freezes it, as
loaded, in the japan registry (`default_registry("japan")`); `score` scores the WIN5 history
(`win5.read_history()`) with it on development days only and appends one RunRecord. A scoring run
that fails after the freeze is recorded as "failed". Scoring the same spec on the same history at
the same commit gives the same record, which the registry holds once: the command says so and
exits 0. The command prints aggregates only. The count of carryover days scored at gross sales
(before 2014-06-07, takeout unknown) is printed and recorded as one more metric
(`GROSS_POOL_FOLD`), its value and n the count, next to one metric per non-empty bin.
"""

from __future__ import annotations

import hashlib
from typing import Any

import typer

from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.holdout import _git_sha
from horsetrack.core.eval.registry import (
    FamilyAlreadyFrozen,
    FoldMetric,
    RecordRefused,
    Registry,
    RunRecord,
    _spec_hash,
    default_registry,
)
from horsetrack.countries.japan.research import win5

SPEC_NAME = "win5_study"
TICKET = 135
HYPOTHESIS_ID = "P1"
SOURCES = ("netkeiba",)
FOLD_SCHEME = "development-days"
METRIC = "mean_day_value"
GROSS_POOL_FOLD = "carryover days before 2014-06-07"
GROSS_POOL_METRIC = "gross_pool_carryover_days"

win5_study_app = typer.Typer(no_args_is_help=True, help="The WIN5 carryover study (#135).")


def _spec() -> tuple[str, dict[str, Any]]:
    spec = dict(load_country_yaml(win5.COUNTRY, SPEC_NAME))
    family = spec.get("family_id")
    if not isinstance(family, str) or not family.strip():
        raise ValueError(f"config/{SPEC_NAME}.yaml names no family_id")
    return family, spec


def _history_build() -> str:
    """The history table's build identity: its name, then the sha256 of its bytes after the `@`
    (a refresh of the same table is the same variant in the registry's family size)."""
    path = win5.history_path()
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        digest = "unavailable"
    return f"{path.name}@{digest}"


def _commit() -> str:
    try:
        return _git_sha()
    except RuntimeError:
        return "unknown"


def _record(
    family: str, spec: dict[str, Any], metrics: tuple[FoldMetric, ...], status: str
) -> RunRecord:
    return RunRecord(
        hypothesis_id=HYPOTHESIS_ID,
        family_id=family,
        variant=spec["coverage"],
        ticket=TICKET,
        git_sha=_commit(),
        data_builds={"win5_history": _history_build()},
        sources=SOURCES,
        fold_scheme=FOLD_SCHEME,
        components={},
        params=spec,
        metrics=metrics,
        status=status,
    )


def _append(registry: Registry, record: RunRecord) -> bool:
    """Append the record; False when the registry already holds this exact run."""
    if record in registry.records(record.family_id):
        return False
    registry.append(record)
    return True


def _metrics(result: dict[str, Any]) -> tuple[FoldMetric, ...]:
    out = []
    for b in result["bins"]:
        if not b["n_days"]:
            continue
        hi = "inf" if b["hi_ratio"] is None else f"{b['hi_ratio']}"
        out.append(FoldMetric(fold=f"ratio {b['lo_ratio']}..{hi}", metric=METRIC,
                              value=b["mean_value"], lo=b["lower_bound"], hi=None,
                              n=b["n_days"]))  # fmt: skip
    gross = result["gross_pool_carryover_days"]
    out.append(FoldMetric(fold=GROSS_POOL_FOLD, metric=GROSS_POOL_METRIC, value=float(gross),
                          lo=None, hi=None, n=gross))  # fmt: skip
    return tuple(out)


@win5_study_app.command("freeze")
def freeze_command() -> None:
    """Freeze config/win5_study.yaml in the japan registry (irreversible; again is a no-op)."""
    try:
        family, spec = _spec()
        win5.check_study_spec(spec)
        spec_hash = default_registry(win5.COUNTRY).freeze_family(family, spec)
    except (ValueError, FamilyAlreadyFrozen) as exc:
        typer.echo(f"not frozen: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"frozen family={family} spec_hash={spec_hash}")


@win5_study_app.command("score")
def score_command() -> None:
    """Score the frozen study on the WIN5 history (development days only) and record the run."""
    family, spec = _spec()
    registry = default_registry(win5.COUNTRY)
    if registry.frozen_spec_hash(family) != _spec_hash(spec):
        typer.echo(f"not scored: family {family} is not frozen with this spec; run "
                   "`horsetrack japan win5-study freeze` first", err=True)  # fmt: skip
        raise typer.Exit(1)
    try:
        result = win5.score_study(win5.read_history(), spec, registry=registry, family_id=family)
        record = _record(family, spec, _metrics(result), "scored")
    except Exception as exc:  # every failure after the freeze is a registry record
        failed = _record(family, spec, (), "failed")
        try:
            new = _append(registry, failed)
        except RecordRefused as refused:
            typer.echo(f"the failed run was not recorded: {refused}", err=True)
            new = False
        typer.echo(f"scoring failed ({type(exc).__name__}: {exc}); "
                   f"{'recorded' if new else 'already recorded'} as failed", err=True)  # fmt: skip
        raise typer.Exit(1) from None
    typer.echo(f"family={family} spec_hash={result['spec_hash']} "
               f"holdout_days_excluded={result['holdout_days_excluded']} "
               f"gross_pool_carryover_days={result['gross_pool_carryover_days']}")  # fmt: skip
    for b in result["bins"]:
        typer.echo(" ".join(f"{k}={v}" for k, v in b.items()))
    if _append(registry, record):
        typer.echo("recorded as scored")
    else:
        typer.echo("this run is already recorded (same spec, history and commit); nothing added")
