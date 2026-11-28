"""A carry-over variant prior for a day's first races, on #305's folds (#328, epic #73).

At T-2, 26% of #305's scored races had no result of their (day, track, surface) before t, so #96's
same-day variant was its N(0, tau^2) prior there and its gain 0 by construction. This study injects
`models.speed_prior.CarryOverPrior` in each of the owner's three pre-registered kinds (#328,
2026-11-28: previous_day, other_surface, both) and asks whether it predicts the top-3 time better
than #305's variant_on. The null is that it does not.

- `predict_prior_folds(table, *, spec, scheme, decision_min, priors)`: #305's fold loop (holdout
  days dropped first, #70's walk-forward folds, t = post_utc - decision_min), with one fit per
  fold and prediction as there. par_only, variant_on and variant_off are `speed_run.predict_folds`'s
  exactly; f"{kind}:variant_on" and f"{kind}:variant_off" are the same fitted model with the kind's
  prior at the race's t injected. The prior reads the fold's fitted pars, sigma and tau and the
  results of every development race up to the fold's last test day; it keeps only those available
  strictly before t (post + 30 min), never the target's own (day, track, surface), a later day or
  another track, and no holdout day is in the table it is given.
- `compute_study`: per error model, prior kind, prediction and baseline ("par_only", or #305's
  "variant_on" from the same fold loop) one variant, failures included: the overall time-RMSE gain
  (gain = rmse_baseline - rmse_model) with D7's bounds, the gain per number of results seen before
  t (capped at 6, `speed_run.by_results_seen`) with bounds, both RMSEs and the gain per fold; the
  by-cell and by-margin tables go to the report.

The study (`run_study`; CLI `python -m horsetrack.countries.japan.studies.speed_prior_run --run
[--out DIR]`, an `--out` inside the repository refused) freezes its spec before the store is read,
appends one registry record per variant, rewrites the committed registry summary and writes the
full report through `write_report`. #305's family is not written. Nothing is adopted (D16), and the
holdout is never opened.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from horsetrack.core import paths
from horsetrack.core.eval.folds import FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha
from horsetrack.core.eval.registry import (
    FoldMetric,
    RunRecord,
    default_registry,
    summary_dir,
    write_summary,
)
from horsetrack.core.eval.reports import write_report
from horsetrack.countries.japan.models.speed import SpeedModel, SpeedSpec
from horsetrack.countries.japan.models.speed_prior import PRIOR_KINDS, CarryOverPrior
from horsetrack.countries.japan.studies import speed_run
from horsetrack.countries.japan.studies.speed_run import (
    BASELINE,
    ERROR_MODELS,
    ESTIMANDS,
    MARGINS,
    PAR_KEYS,
    PREDICTIONS,
    VARIANT_KEYS,
    Variant,
    markdown_table,
    read_inputs,
    rmse_table,
    store_builds,
)

COUNTRY = speed_run.COUNTRY
SOURCE = speed_run.SOURCE  # the store's JRA results (netkeiba); no JRDB, no JRA-VAN
TICKET = 328
FAMILY_ID = "japan-speed-top3-c3-prior"
BASE_FAMILY_ID = speed_run.FAMILY_ID  # #305's
HYPOTHESIS_ID = speed_run.HYPOTHESIS_ID
BOOTSTRAP_SEED = 328  # reproducibility only; scoring.yaml signs no seed
BASELINES = (BASELINE, "variant_on")  # the par alone, and #305's variant_on
SEEN_CAP = 6
# the pre-registered parameters, each kind's CarryOverPrior keywords (frozen in the spec)
HALF_LIFE_DAYS = 7.0
SURFACE_WEIGHT = 0.5
PRIORS = {
    "previous_day": {"half_life_days": HALF_LIFE_DAYS},
    "other_surface": {"surface_weight": SURFACE_WEIGHT},
    "both": {"half_life_days": HALF_LIFE_DAYS, "surface_weight": SURFACE_WEIGHT},
}


# --------------------------------------------------------------------------- the fold loop


class _Switch:
    """The variant prior a fold's model reads: #305's N(0, tau^2) when `prior` is None, else
    that prior at `at_utc`. The fit never reads it (#96), so one fit serves every prior."""

    def __init__(self) -> None:
        self.tau: float | None = None
        self.prior: CarryOverPrior | None = None
        self.at_utc: Any = None

    def __call__(self, race_date, track, surface) -> tuple[float, float]:
        if self.prior is None:
            return 0.0, self.tau
        return self.prior(race_date, track, surface, at_utc=self.at_utc)


def prior_column(kind: str, prediction: str) -> str:
    return f"{kind}:{prediction}"


def predict_prior_folds(
    table: pd.DataFrame,
    *,
    spec: SpeedSpec,
    scheme: FoldScheme,
    decision_min: float,
    priors: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """One row per scored development test race: #305's columns (`speed_run.predict_folds`) and
    f"{kind}:{prediction}" per kind in `priors` (see the module docstring). attrs as #305's."""
    if (
        isinstance(decision_min, bool)
        or not isinstance(decision_min, Real)
        or not math.isfinite(decision_min)
        or decision_min < 0
    ):
        raise ValueError(f"decision_min must be a non-negative number of minutes: {decision_min!r}")
    unknown = sorted(set(priors) - set(PRIOR_KINDS))
    if unknown:
        raise ValueError(f"unknown prior kind(s) {unknown}; known: {PRIOR_KINDS}")
    kinds = [k for k in PRIOR_KINDS if k in priors]
    dev = speed_run._development(table)
    dev_days = speed_run._race_days(dev["race_date"]).to_numpy()
    lead = pd.Timedelta(minutes=float(decision_min))
    rows, folds = [], []
    for k, fold in enumerate(walk_forward(dev["race_date"], scheme=scheme)):
        label = speed_run.fold_label(k, fold.test_start, fold.test_end)
        train, test = dev.iloc[fold.train_idx], dev.iloc[fold.test_idx]
        switches = {name: _Switch() for name in PREDICTIONS}
        models = {}
        for name in PREDICTIONS:  # one fit at a time
            variant_spec = dataclasses.replace(spec, same_day_variant=name == "variant_on")
            model = SpeedModel(variant_spec, variant_prior=switches[name]).fit(train)
            models[name] = model.observe(test)
            switches[name].tau = model.tau
        fitted = models["variant_on"]
        # every development race up to the window's last day; each prior keeps what is usable at t
        known = dev.loc[dev_days <= fold.test_end]
        carry = {kind: CarryOverPrior(fitted, known, kind=kind, **priors[kind]) for kind in kinds}
        del known
        scored, unseen = 0, 0
        for race in test.to_dict("records"):
            try:
                par = fitted.par(**{key: race[key] for key in spec.par_keys})
            except KeyError:
                unseen += 1
                continue
            t = race["post_utc"] - lead
            row = {key: race[key] for key in ("race_id", "meeting_id", "race_date")}
            row |= {"fold": label, "post_utc": race["post_utc"], "decision_utc": t}
            row |= {"top3_time_s": race["top3_time_s"], BASELINE: par}
            row |= {name: m.expected_top3_time(race, at_utc=t) for name, m in models.items()}
            for kind in kinds:
                for name, model in models.items():
                    switches[name].prior, switches[name].at_utc = carry[kind], t
                    row[prior_column(kind, name)] = model.expected_top3_time(race, at_utc=t)
                    switches[name].prior = None
            rows.append(row | {key: race[key] for key in spec.par_keys})
            scored += 1
        folds.append(
            {"fold": label, "train_end": fold.train_end.isoformat()}
            | {"n_train": fitted.n_races, "n_test": len(test), "n_scored": scored}
            | {"n_unseen_cell": unseen, "n_cells": speed_run._n_cells(train, spec.par_keys)}
            | {"sigma_s": fitted.sigma, "tau_s": fitted.tau, "n_iter": fitted.n_iter}
        )
        del models, carry, switches
    columns = ["race_id", "meeting_id", "race_date", "fold", "post_utc", "decision_utc"]
    columns += ["top3_time_s", BASELINE, *PREDICTIONS]
    columns += [prior_column(kind, p) for kind in kinds for p in PREDICTIONS]
    out = pd.DataFrame(rows, columns=[*columns, *spec.par_keys])
    for column in ("post_utc", "decision_utc"):
        out[column] = pd.to_datetime(out[column], utc=True)
    out.attrs["n_unseen_cell"] = sum(f["n_unseen_cell"] for f in folds)
    out.attrs["n_holdout_dropped"] = len(table) - len(dev)
    out.attrs["folds"] = folds
    return out


# --------------------------------------------------------------------------- the spec


def bootstrap_settings() -> dict:
    """D7's bootstrap from config/scoring.yaml, with this study's fixed seed."""
    return speed_run.bootstrap_settings() | {"seed": BOOTSTRAP_SEED}


def build_spec(boot: dict, scheme: FoldScheme, minutes: int) -> dict:
    """The family's pre-registered spec, frozen before the store is read: #305's definitions
    (estimand, races, availability, keys, folds, bootstrap, decisions) and the three priors."""
    base = speed_run.build_spec(boot, scheme, minutes)
    model = dict(base["model"]) | {
        "variant_prior": "models.speed_prior.CarryOverPrior (#328), one per kind in `priors`",
    }
    return base | {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "base_family": BASE_FAMILY_ID,
        "hypotheses": {
            "null": "no carry-over prior lowers the time RMSE of E[top-3 time] below #305's "
            "variant_on (the same-day variant from N(0, tau^2)) on the races with no same-day "
            "result before t: gain = rmse_variant_on - rmse_{kind}:variant_on <= 0 on "
            "results_seen:0",
            "prior_kinds": "each of previous_day (a), other_surface (b) and both (c) (the owner, "
            "#328, 2026-11-28) is its own pre-registered variant and ranked row",
            "claim": "a kind's claim rests on the one-sided lower bound of its results_seen:0 gain "
            "over #305's variant_on being above 0; the overall gain is reported beside it",
            "check": "on results_seen >= 1 the same-day results dominate the prior, so the gain "
            "there is expected to shrink toward 0",
        },
        "model": model,
        "priors": {kind: dict(params) for kind, params in PRIORS.items()},
        "prior_definitions": {
            "base": "N(0, tau^2), tau the fold fit's",
            "residual": "top3_time_s - par(cell) of the fold's fit; a cell the fit never saw is "
            "skipped",
            "usable": "result_available_utc (post + 30 min) strictly before t; a null time never",
            "posterior": "Post(day, track, surface; t): the fold model's own posterior from the "
            "base given that group's usable residuals, as SpeedModel's variant computes it (the "
            "normal closed form; under t5 its reweighted mode and the reweighted normal SD)",
            "previous_day": "d' the latest day before race_date with a usable result of the same "
            "(track, surface); rho = 0.5 ** ((race_date - d').days / half_life_days); mean rho m', "
            "sd^2 rho^2 s'^2 + (1 - rho^2) tau^2; none: the base",
            "other_surface": "Post of the same day's other surface (turf <-> dirt) at the same "
            "track; w = surface_weight; mean w m_o, sd^2 w^2 s_o^2 + (1 - w^2) tau^2; none, or a "
            "surface without another (e.g. unknown): the base",
            "both": "p = 1/sd_a^2 + 1/sd_b^2 - 1/tau^2, mean (m_a/sd_a^2 + m_b/sd_b^2) / p, sd "
            "p^-0.5",
            "inputs": "every development race up to the fold's last test day (training, embargo "
            "and test days; a training day's residual is against a par fitted on it, all of it "
            "published before the window); holdout days are dropped before the folds",
            "half_life_days": f"{HALF_LIFE_DAYS}: one race week. A Sunday reads its Saturday at "
            "rho 0.91 (same course, overnight maintenance only); a Saturday reads the previous "
            "Sunday at rho 0.55 (a week of weather and course work). Chosen before any scored "
            "read; no measured persistence exists (ledger)",
            "surface_weight": f"{SURFACE_WEIGHT}: the turf and dirt variants share the day's "
            "weather, wind and temperature, but moisture moves them in opposite directions (wet "
            "dirt rides faster, soft turf slower), so the coupling's sign is uncertain a priori; "
            "the midpoint of the allowed [0, 1], chosen before any scored read",
        },
        "predictions": base["predictions"]
        | {
            "{kind}:variant_on": "par + the same-day variant's mean at t, from the kind's prior at "
            "t given the day's own results available strictly before t",
            "{kind}:variant_off": "par + the kind's prior mean at t (same_day_variant=False)",
        },
        "baselines": {
            BASELINE: "the par of the race's cell, fitted on the fold's training races",
            "variant_on": "#305's variant_on (no injected prior), from the same fold loop and fit",
        },
        "metric": "time RMSE (s) over the scored races; gain = rmse_baseline - rmse_model",
        "by": {
            "results_seen": "primary split: the races of the race's (day, track, surface) with a "
            f"result available strictly before t, capped at {SEEN_CAP}; results_seen:0 first",
            "cells": list(PAR_KEYS),
            "margins": list(MARGINS),
        },
        "bootstrap": boot,
        "sources": [SOURCE],
    }


# --------------------------------------------------------------------------- scoring


def _metric(fold: str, metric: str, value: float, n: int, lo=None, hi=None) -> FoldMetric:
    return speed_run._metric(fold, metric, value, n, lo, hi)


def _rmse(err: np.ndarray) -> float:
    return math.sqrt(float(np.mean(err**2)))


def prior_metrics(
    predictions: pd.DataFrame, model: str, baseline: str, overall: pd.Series, by_seen: pd.DataFrame
) -> list[FoldMetric]:
    """The overall gain with D7's bounds and both RMSEs, the gain with bounds per results seen,
    then the gain on each fold."""
    n = int(overall["n_races"])
    metrics = [
        _metric("all", "rmse_gain_s", overall["gain"], n, overall["lo"], overall["hi"]),
        _metric("all", "rmse_s", overall["rmse_model"], n),
        _metric("all", f"rmse_{baseline}_s", overall["rmse_baseline"], n),
    ]
    for row in by_seen[by_seen["scope"] == "cell"].to_dict("records"):
        label = f"results_seen:{int(row['results_seen'])}"
        metrics.append(
            _metric(label, "rmse_gain_s", row["gain"], row["n_races"], row["lo"], row["hi"])
        )
    for label, part in predictions.groupby("fold", sort=False):
        actual = part["top3_time_s"].to_numpy(dtype=float)
        base, mine = (part[c].to_numpy(dtype=float) - actual for c in (baseline, model))
        metrics.append(_metric(label, "rmse_gain_s", _rmse(base) - _rmse(mine), len(part)))
    return metrics


def _variant(error_name: str, kind: str, prediction: str, baseline: str, params: dict, **outcome):
    components = {"model": "countries.japan.models.speed.SpeedModel"}
    components |= {"prior": kind, "prediction": prediction, "baseline": baseline}
    components |= {"error_model": error_name, "estimand": ESTIMANDS[0]}
    name = f"{error_name}:{kind}:{prediction}_vs_{baseline}"
    return Variant(name, components, params, **outcome)


def _grid():
    for kind in PRIOR_KINDS:
        for prediction in PREDICTIONS:
            for baseline in BASELINES:
                yield kind, prediction, baseline


def compute_study(
    table: pd.DataFrame, spec: dict, scheme: FoldScheme, minutes: int
) -> tuple[list[Variant], dict, dict]:
    """Per error model, the prior fold loop and its 12 variants (kind x prediction x baseline);
    the tables for the report; the run counts. A failed fold loop is 12 failed variants and a
    failed score one, never dropped."""
    boot, priors = spec["bootstrap"], spec["priors"]
    kwargs = {k: boot[k] for k in ("n_boot", "level", "seed")}
    variants, tables, counts = [], {}, {}
    for error_name, t_df in ERROR_MODELS.items():
        speed = SpeedSpec(PAR_KEYS, VARIANT_KEYS, same_day_variant=True, t_df=t_df)
        base_params = {"error_model": error_name, "t_df": t_df, "decision_time_min": minutes}
        base_params |= {"bootstrap": boot, "estimand": ESTIMANDS[0]}

        def params(kind: str, base_params: dict = base_params) -> dict:
            return base_params | {"prior": kind, "prior_params": dict(priors[kind])}

        try:
            predictions = predict_prior_folds(
                table, spec=speed, scheme=scheme, decision_min=minutes, priors=priors
            )
            if predictions.empty:
                raise ValueError("no development test race was scored")
        except Exception as exc:  # every failure is a registry record, never dropped
            for kind, prediction, baseline in _grid():
                why = speed_run._why(exc)
                variants.append(
                    _variant(error_name, kind, prediction, baseline, params(kind), error=why)
                )
            continue
        seen = speed_run.by_results_seen(predictions, table).clip(upper=SEEN_CAP)
        predictions = predictions.assign(results_seen=seen.to_numpy())
        counts[error_name] = {
            "n_scored": len(predictions),
            "n_meetings": int(predictions["meeting_id"].nunique()),
            "n_folds": len(predictions.attrs["folds"]),
            "n_unseen_cell": predictions.attrs["n_unseen_cell"],
            "n_holdout_dropped": predictions.attrs["n_holdout_dropped"],
            "n_results_seen_0": int((seen == 0).sum()),
            "first_test_day": str(min(predictions["race_date"])),
            "last_test_day": str(max(predictions["race_date"])),
        }
        tables[f"folds:{error_name}"] = predictions.attrs["folds"]
        for kind, prediction, baseline in _grid():
            column = prior_column(kind, prediction)
            key = f"{error_name}:{column}:vs_{baseline}"
            try:
                score = {"model": column, "baseline": baseline, **kwargs}
                by_seen = rmse_table(predictions, by=("results_seen",), **score)
                by_cell = rmse_table(predictions, by=PAR_KEYS, **score)
                metrics = prior_metrics(predictions, column, baseline, by_cell.iloc[0], by_seen)
                tables[f"by_results_seen:{key}"] = by_seen
                tables[f"cells:{key}"] = by_cell
                for margin in MARGINS:
                    tables[f"by_{margin}:{key}"] = rmse_table(predictions, by=(margin,), **score)
            except Exception as exc:
                why = speed_run._why(exc)
                variants.append(
                    _variant(error_name, kind, prediction, baseline, params(kind), error=why)
                )
                continue
            variants.append(
                _variant(
                    error_name, kind, prediction, baseline, params(kind), metrics=tuple(metrics)
                )
            )
        del predictions
    return variants, tables, counts


def _record(variant: Variant, git_sha: str, builds: dict[str, str], fold_scheme: str) -> RunRecord:
    params = variant.params | ({"error": variant.error} if variant.error else {})
    return RunRecord(
        hypothesis_id=HYPOTHESIS_ID,
        family_id=FAMILY_ID,
        variant=variant.name,
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=builds,
        sources=(SOURCE,),
        fold_scheme=fold_scheme,
        components=variant.components,
        params=params,
        metrics=variant.metrics,
        status="failed" if variant.error else "scored",
    )


# --------------------------------------------------------------------------- report and runner


def _find(v: Variant, fold: str, metric: str = "rmse_gain_s") -> FoldMetric | None:
    return next((m for m in v.metrics if (m.fold, m.metric) == (fold, metric)), None)


def first_race_rows(variants: list[Variant], baseline: str = "variant_on") -> list[dict]:
    """Per variant against `baseline`: its gain on the races with no same-day result before t
    (results_seen:0) with D7's bounds, then its overall gain."""
    rows = []
    for v in variants:
        if v.components["baseline"] != baseline:
            continue
        first, overall = _find(v, "results_seen:0"), _find(v, "all")
        row = {"variant": v.name, "prior": v.components["prior"]}
        row |= {"error_model": v.components["error_model"]}
        row |= {"prediction": v.components["prediction"]}
        if first:
            row |= {"n_seen_0": first.n, "gain_seen_0": first.value}
            row |= {"lo_seen_0": first.lo, "hi_seen_0": first.hi}
        if overall:
            row |= {"n_all": overall.n, "gain_all": overall.value}
            row |= {"lo_all": overall.lo, "hi_all": overall.hi}
        if v.error:
            row["error"] = v.error
        rows.append(row)
    return rows


def render_report(variants: list[Variant], tables: dict, counts: dict, about: dict) -> str:
    """Aggregates only (no race row): the first races' gains over #305's variant_on first, then
    every variant, the counts and every table."""
    lines = [f"# A carry-over variant prior for a day's first races ({COUNTRY}, #{TICKET})", ""]
    lines += [f"- {key}: {value}" for key, value in about.items()]
    lines += ["- development days only: holdout days are never read and the holdout is not opened"]
    lines += ["", "## No same-day result before t: gain over #305's variant_on", ""]
    lines += [markdown_table(first_race_rows(variants))]
    lines += [
        "",
        "## The same against par_only",
        "",
        markdown_table(first_race_rows(variants, BASELINE)),
    ]
    rows = []
    for v in variants:
        row = {"variant": v.name, "status": "failed" if v.error else "scored"}
        row |= {m.metric: m.value for m in v.metrics if m.fold == "all"}
        overall = _find(v, "all")
        row |= {"lo": overall.lo if overall else None, "hi": overall.hi if overall else None}
        rows.append(row | ({"error": v.error} if v.error else {}))
    lines += ["", "## Variants (one registry record each)", "", markdown_table(rows)]
    lines += (
        ["", "## Race table", "", markdown_table([counts["builder"]])]
        if "builder" in counts
        else []
    )
    runs = [{"error_model": k, **v} for k, v in counts.items() if k in ERROR_MODELS]
    lines += ["", "## Runs", "", markdown_table(runs)]
    for key, frame in tables.items():
        lines += ["", f"## {key.replace(':', ', ')}", "", markdown_table(frame)]
    return "\n".join(lines) + "\n"


def run_study(out_dir: Path | None = None) -> dict:
    """Freeze the spec, build the race table from the development store, run the prior fold loop
    per error model, append the registry records, rewrite the committed summary and write the
    report outside the tree."""
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    scheme, minutes = load_country_scheme(COUNTRY), speed_run.decision_min()
    spec = build_spec(bootstrap_settings(), scheme, minutes)
    git_sha = _git_sha()
    registry = default_registry(COUNTRY)
    spec_hash = registry.freeze_family(FAMILY_ID, spec)  # before any outcome is read

    builds = store_builds()
    races, runners = read_inputs()
    table, builder = speed_run.build_race_table(races, runners, estimand=ESTIMANDS[0])
    del races, runners
    builder |= {
        "first_race_day": str(min(table["race_date"])),
        "last_race_day": str(max(table["race_date"])),
    }
    variants, tables, counts = compute_study(table, spec, scheme, minutes)
    counts = {"builder": builder} | counts

    existing = registry.records(FAMILY_ID)
    fold_scheme = speed_run.scheme_label(scheme)
    run_ids = {}
    for variant in variants:
        record = _record(variant, git_sha, builds, fold_scheme)
        # only an exact match, its commit included, is already recorded
        if record not in existing:
            run_ids[variant.name] = registry.append(record)
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "base family": f"`{BASE_FAMILY_ID}` (#305), recomputed in the same fold loop",
        "git": git_sha,
        "store": ", ".join(builds.values()),
        "run ids": ", ".join(f"{k} `{v}`" for k, v in run_ids.items()) or "already recorded",
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = write_report(
        out_dir / f"speed-prior-{COUNTRY}-{spec_hash[:12]}.md",
        render_report(variants, tables, counts, about),
        sources=(SOURCE,),
    )
    # last: another ticket's family id in the shared registry can make it raise (#344, #391)
    write_summary(registry, summary_dir(COUNTRY))
    return {
        "family_id": FAMILY_ID,
        "spec_hash": spec_hash,
        "report_path": str(report),
        "registry_path": str(registry.path),
        "run_ids": run_ids,
        "counts": counts,
        "failed_variants": {v.name: v.error for v in variants if v.error},
        "first_races": first_race_rows(variants),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m horsetrack.countries.japan.studies.speed_prior_run"
    )
    parser.add_argument("--run", action="store_true", required=True, help="run the study")
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    args = parser.parse_args(argv)
    print(json.dumps(run_study(args.out), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
