"""The WIN5 no-hit (rollover) rate and the rollover-adjusted EV of the carryover bins (ticket #323).

Follow-up of #135 (docs/specs/win5-rollover-study.md, spec config/win5_rollover.yaml): its day
bootstrap cannot see a rollover that never occurred in the sample. A no-hit day pays 0 and passes
all the money on, so with `won_value` the mean `win5.day_value` of the won days of a bin and `q`
the no-hit rate:

    rollover-adjusted EV = (1 - q) x won_value
    break-even no-hit rate q* = 1 - 1 / won_value      (0.0 when won_value <= 1: no q helps)

`no_hit_interval` is the exact two-sided Clopper-Pearson interval of a no-hit count.
`score_rollover` scores the frozen spec on development days only (D14), on the bins of #135. The
commands `horsetrack japan win5-rollover freeze|score` live at the bottom of this module.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import typer
from scipy.stats import beta

from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day
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
from horsetrack.countries.japan.research.win5 import (
    NotPreRegistered,
    _bin_of,
    _non_negative,
    _number,
    _plain,
    _pool,
    day_value,
    net_pool,
)
from horsetrack.countries.japan.research.win5_study import _history_build

INTERVAL = "clopper-pearson"
SPEC_NAME = "win5_rollover"
TICKET = 323
HYPOTHESIS_ID = "P1"
SOURCES = ("netkeiba",)
FOLD_SCHEME = "development-days"
METRIC = "no_hit_rate"
ALL_DAYS_FOLD = "all development days"


# ---------------------------------------------------------------- the arithmetic


def _count(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return value


def _real(value: Any, name: str) -> float:
    """A finite number; a string is refused (win5._number would parse "1.2")."""
    if isinstance(value, str):
        raise ValueError(f"{name} is not a number: {value!r}")
    return _number(value, name)


def _won_value(value: Any) -> float:
    out = _real(value, "won_value")
    if out < 0:
        raise ValueError(f"won_value must not be negative, got {value!r}")
    return out


def rollover_adjusted_ev(won_value: Any, no_hit_rate: Any) -> float:
    """(1 - q) x won_value. ValueError for a rate outside [0, 1] or a value that is not a finite
    non-negative number."""
    value = _won_value(won_value)
    rate = _real(no_hit_rate, "no_hit_rate")
    if not 0 <= rate <= 1:
        raise ValueError(f"no_hit_rate must be in [0, 1], got {no_hit_rate!r}")
    return (1 - rate) * value


def break_even_no_hit_rate(won_value: Any) -> float:
    """The no-hit rate at which the adjusted EV reaches 1: 1 - 1 / won_value, 0.0 for a won value
    of 1 or less."""
    value = _won_value(won_value)
    return 0.0 if value <= 1 else 1 - 1 / value


def is_no_hit(row: dict[str, Any]) -> bool:
    """A history row with no winning ticket. A capped win that passed the excess on paid, so it is
    a hit."""
    return int(row["winning_tickets"]) == 0


# ---------------------------------------------------------------- the interval


def no_hit_interval(n_no_hit: Any, n_days: Any, *, confidence: Any = 0.95) -> tuple[float, float]:
    """The exact two-sided Clopper-Pearson interval of `n_no_hit` in `n_days` (lower 0.0 at 0,
    upper 1.0 at n_days). ValueError for counts that make no sense or a confidence outside
    (0, 1)."""
    k = _count(n_no_hit, "n_no_hit")
    n = _count(n_days, "n_days")
    level = _real(confidence, "confidence")
    if n < 1:
        raise ValueError(f"n_days must be at least 1, got {n!r}")
    if not 0 <= k <= n:
        raise ValueError(f"n_no_hit must be in [0, {n}], got {k!r}")
    if not 0 < level < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    tail = (1 - level) / 2
    lower = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
    return lower, upper


# ---------------------------------------------------------------- the study


def _params(spec: dict[str, Any]) -> dict[str, Any]:
    """The figures the scorer reads from the spec; ValueError for a spec it cannot score."""
    if not isinstance(spec, dict):
        raise ValueError(f"a rollover spec is a mapping, got {type(spec).__name__}")
    if spec.get("interval") != INTERVAL:
        raise ValueError(
            f"the scorer knows interval {INTERVAL!r} only, got {spec.get('interval')!r}"
        )
    if spec.get("pool") != win5.NET_POOL:
        raise ValueError(f"the scorer knows pool {win5.NET_POOL!r} only, got {spec.get('pool')!r}")
    confidence = _number(spec.get("confidence"), "confidence")
    if isinstance(spec.get("confidence"), str) or not 0 < confidence < 1:
        raise ValueError(f"confidence must be a number in (0, 1), got {spec.get('confidence')!r}")
    edges = spec.get("ratio_edges")
    if not isinstance(edges, list) or not edges:
        raise ValueError("the spec names no ratio_edges")
    values = [_non_negative(e, "ratio edge") for e in edges]
    if any(b <= a for a, b in zip(values, values[1:], strict=False)):
        raise ValueError(f"ratio_edges must be strictly increasing, got {edges!r}")
    break_even = _number(spec.get("break_even_takeout"), "break_even_takeout")
    if not 0 < break_even < 1:
        raise ValueError(f"break_even_takeout must be in (0, 1), got {break_even!r}")
    _pool(spec.get("payout_cap_yen"), "payout_cap_yen")
    return {"edges": list(edges), "confidence": confidence, "break_even": break_even}


def check_rollover_spec(spec: dict[str, Any]) -> None:
    """ValueError unless `score_rollover` can score `spec` (checked before it is frozen)."""
    _params(spec)


def _rate(k: int, n: int, confidence: float) -> dict[str, Any]:
    lower, upper = no_hit_interval(k, n, confidence=confidence)
    return {"n_days": n, "n_no_hit": k, "no_hit_rate": k / n, "no_hit_lower": lower,
            "no_hit_upper": upper}  # fmt: skip


def score_rollover(
    history: pd.DataFrame,
    spec: dict[str, Any],
    *,
    registry: Registry,
    family_id: str,
) -> dict[str, Any]:
    """Score the frozen spec on development days. NotPreRegistered unless `spec` is the spec frozen
    for `family_id`; ValueError for a spec the scorer cannot score. Holdout days are dropped first
    and counted. Days are binned as `win5.score_study` bins them (carryover in over the
    break-even carryover on the net pool), no-hit days included; the `overall` rate covers every
    development day, with or without carryover. A bin reports its no-hit rate and interval, the
    mean day value of its won days, the break-even rate, and the adjusted EV at its own rate and
    at the upper bound, and at the overall rate and its upper bound (the pooled figures)."""
    spec_hash = _spec_hash(spec)
    if registry.frozen_spec_hash(family_id) != spec_hash:
        raise NotPreRegistered(f"family {family_id!r} has no frozen spec with hash {spec_hash}")
    p = _params(spec)
    edges, confidence = p["edges"], p["confidence"]
    won: list[list[float]] = [[] for _ in edges]
    no_hit = [0] * len(edges)
    excluded = n_days = n_no_hit = 0
    for row in history.to_dict("records"):
        row = _plain(row)
        if is_holdout_day(win5.COUNTRY, row["race_date"]):
            excluded += 1
            continue
        n_days += 1
        n_no_hit += is_no_hit(row)
        carry_in = float(row["carryover_in_yen"])
        where = _bin_of(carry_in / (p["break_even"] * net_pool(row)), edges)
        if where is None:
            continue
        if is_no_hit(row):
            no_hit[where] += 1
        else:
            won[where].append(day_value(row))
    overall = _rate(n_no_hit, n_days, confidence) if n_days else {
        "n_days": 0, "n_no_hit": 0, "no_hit_rate": None, "no_hit_lower": None,
        "no_hit_upper": None}  # fmt: skip
    bins = []
    for i, lo in enumerate(edges):
        n = len(won[i]) + no_hit[i]
        b: dict[str, Any] = {
            "lo_ratio": lo,
            "hi_ratio": edges[i + 1] if i + 1 < len(edges) else None,
        }
        keys = ("won_mean_value", "break_even_no_hit_rate", "adjusted_ev", "adjusted_ev_at_upper",
                "adjusted_ev_pooled", "adjusted_ev_pooled_at_upper")  # fmt: skip
        if n:
            b |= _rate(no_hit[i], n, confidence)
        else:
            b |= {"n_days": 0, "n_no_hit": 0, "no_hit_rate": None, "no_hit_lower": None,
                  "no_hit_upper": None}  # fmt: skip
        b |= dict.fromkeys(keys)
        if won[i]:
            mean = sum(won[i]) / len(won[i])
            b["won_mean_value"] = mean
            b["break_even_no_hit_rate"] = break_even_no_hit_rate(mean)
            b["adjusted_ev"] = rollover_adjusted_ev(mean, b["no_hit_rate"])
            b["adjusted_ev_at_upper"] = rollover_adjusted_ev(mean, b["no_hit_upper"])
            b["adjusted_ev_pooled"] = rollover_adjusted_ev(mean, overall["no_hit_rate"])
            b["adjusted_ev_pooled_at_upper"] = rollover_adjusted_ev(mean, overall["no_hit_upper"])
        bins.append(b)
    return {
        "family_id": family_id,
        "spec_hash": spec_hash,
        "holdout_days_excluded": excluded,
        "overall": overall,
        "bins": bins,
    }


# ---------------------------------------------------------------- the commands

win5_rollover_app = typer.Typer(no_args_is_help=True, help="The WIN5 rollover risk (#323).")


def _spec() -> tuple[str, dict[str, Any]]:
    spec = dict(load_country_yaml(win5.COUNTRY, SPEC_NAME))
    family = spec.get("family_id")
    if not isinstance(family, str) or not family.strip():
        raise ValueError(f"config/{SPEC_NAME}.yaml names no family_id")
    return family, spec


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
        variant=f"{spec['interval']} {spec['confidence']}",
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
    def metric(fold: str, b: dict[str, Any]) -> FoldMetric:
        return FoldMetric(fold=fold, metric=METRIC, value=b["no_hit_rate"], lo=b["no_hit_lower"],
                          hi=b["no_hit_upper"], n=b["n_days"])  # fmt: skip

    out = []
    if result["overall"]["n_days"]:
        out.append(metric(ALL_DAYS_FOLD, result["overall"]))
    for b in result["bins"]:
        if b["n_days"]:
            hi = "inf" if b["hi_ratio"] is None else f"{b['hi_ratio']}"
            out.append(metric(f"ratio {b['lo_ratio']}..{hi}", b))
    return tuple(out)


@win5_rollover_app.command("freeze")
def freeze_command() -> None:
    """Freeze config/win5_rollover.yaml in the japan registry (irreversible; again is a no-op)."""
    try:
        family, spec = _spec()
        check_rollover_spec(spec)
        spec_hash = default_registry(win5.COUNTRY).freeze_family(family, spec)
    except (ValueError, FamilyAlreadyFrozen) as exc:
        typer.echo(f"not frozen: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"frozen family={family} spec_hash={spec_hash}")


@win5_rollover_app.command("score")
def score_command() -> None:
    """Score the frozen study on the WIN5 history (development days only) and record the run."""
    family, spec = _spec()
    registry = default_registry(win5.COUNTRY)
    if registry.frozen_spec_hash(family) != _spec_hash(spec):
        typer.echo(f"not scored: family {family} is not frozen with this spec; run "
                   "`horsetrack japan win5-rollover freeze` first", err=True)  # fmt: skip
        raise typer.Exit(1)
    try:
        result = score_rollover(win5.read_history(), spec, registry=registry, family_id=family)
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
               f"holdout_days_excluded={result['holdout_days_excluded']}")  # fmt: skip
    typer.echo("overall " + " ".join(f"{k}={v}" for k, v in result["overall"].items()))
    for b in result["bins"]:
        typer.echo(" ".join(f"{k}={v}" for k, v in b.items()))
    if _append(registry, record):
        typer.echo("recorded as scored")
    else:
        typer.echo("this run is already recorded (same spec, history and commit); nothing added")
