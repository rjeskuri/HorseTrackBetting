"""#96's expected top-3 time on development folds against the par-only baseline (#305, epic #73).

C3's standalone time test (08 §8 B6, the time-RMSE part of E39): does the same-day variant of
`countries/japan/models/speed.py` (#96) predict a race's top-3 time better than the par of its
cell alone? The null is that it does not.

- `build_race_table(races, runners, *, estimand)`: one row per race for `SpeedModel`, from the
  store's races and runners only (no JRDB table, D21). The estimand has no default; the one it
  knows, "mean_placed_1_3", is the owner's (#96, 2026-11-25) with its dead-heat reading (#305,
  2026-11-25): each runner placed 1 or 2 counts once, the runners placed 3rd share one slot (their
  mean), and a race without a 3rd-placed runner uses the runners placed 1-2 present (partly
  placed, counted). Jump races, races without a post time and races with no placed, timed runner
  are left out and counted, in that order. post_utc is post_time_local on the Asia/Tokyo wall clock
  of race_date (as-of.md, #227); result_available_utc = post_utc + 30 min, the same-day dividends
  rule (availability.yaml; the owner's answers on #80 and #96).
- `predict_folds(table, *, spec, scheme, decision_min)`: holdout days dropped first, then #70's
  walk-forward folds over the development races. Per fold a model is fitted on the training races,
  observes the test races' results, and predicts each test race at t = post_utc - decision_min
  with the same-day variant on and off; the par of its cell is the baseline. A test race whose cell
  the fit never saw is not scored and is counted.
- `rmse_diff_bootstrap`: the RMSE difference (gain = rmse_baseline - rmse_model, positive when the
  model is better, the sign of `core.eval.scoring.gain`) with D7's bounds. `core.eval.bootstrap.
  paired_bootstrap` bounds a mean of per-race differences, which an RMSE difference is not, so each
  replicate here draws whole meetings the same way and recomputes both RMSEs over the pooled races
  of the draw, model and baseline on the same draw (paired).
- `rmse_table`: the overall row and one row per distinct `by` value, each with its own bounds.
- `read_inputs()`: the JRA store's races and runners, column-scoped, holdout days dropped.

The study (`run_study`; CLI `python -m horsetrack.countries.japan.studies.speed_run --run [--out
DIR]`, an `--out` inside the repository refused) freezes its spec in the run registry before any
outcome is read, runs the fold loop at the primary decision time (config/decision.yaml) for each
error model, appends one registry record per variant (failures included), rewrites the committed
registry summary and writes the full report (every cell) through `write_report`. Nothing is
adopted (D16), and the holdout is never opened.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.bootstrap import _MAX_DRAWS_PER_CHUNK
from horsetrack.core.eval.folds import FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day
from horsetrack.core.eval.registry import (
    FoldMetric,
    RunRecord,
    default_registry,
    summary_path,
    write_summary,
)
from horsetrack.core.eval.reports import write_report
from horsetrack.core.store.tables import read_table, table_path
from horsetrack.countries.japan.models.speed import SpeedModel, SpeedSpec

COUNTRY = "japan"
SOURCE = "jra"  # the store's JRA results (netkeiba); no JRDB, no JRA-VAN
TICKET = 305
FAMILY_ID = "japan-speed-top3-c3"
HYPOTHESIS_ID = "E39"  # 08 §4: C3's figure; this is its standalone time test (08 §8 B6)
BOOTSTRAP_SEED = 305  # reproducibility only; scoring.yaml signs no seed

ESTIMANDS = ("mean_placed_1_3",)
JST = "Asia/Tokyo"
RESULT_LAG = pd.Timedelta(minutes=30)  # the same-day dividends rule (availability.yaml, #227)
CLOCK = r"[0-9]{1,2}:[0-9]{2}"  # "H:MM" or "HH:MM", as core.store.asof derives post_utc
JUMP = "jump"
UNKNOWN = "unknown"
RACE_COLUMNS = (
    "race_id",
    "meeting_id",
    "race_date",
    "venue",
    "surface",
    "distance_m",
    "course_config",
    "rail_setting",
    "race_class",
    "post_time_local",
)
RUNNER_COLUMNS = ("race_id", "finish_pos", "finish_time_s")
# SpeedModel key <- races column
KEYS = {
    "track": "venue",
    "surface": "surface",
    "distance": "distance_m",
    "layout": "course_config",
    "rail": "rail_setting",
    "class_level": "race_class",
}
PAR_KEYS = tuple(KEYS)
VARIANT_KEYS = ("track", "surface")
ERROR_MODELS = {"t5": 5.0, "normal": None}  # #96: Student-t error; the normal one as a check
PREDICTIONS = ("variant_on", "variant_off")
BASELINE = "par_only"
MARGINS = ("surface", "track", "class_level", "layout", "distance")


# --------------------------------------------------------------------------- the race table


def _race_days(values: pd.Series) -> pd.Series:
    """Each value's race day as a `datetime.date` (from dates, naive timestamps or strings)."""
    stamps = pd.to_datetime(pd.Series(values))
    if isinstance(stamps.dtype, pd.DatetimeTZDtype):
        raise ValueError("race_date is timezone-aware; pass the local race date")
    if stamps.isna().any():
        raise ValueError("race_date has a missing value")
    return pd.Series(stamps.dt.date.to_numpy(), index=pd.Series(values).index, dtype=object)


def _post_utc(days: pd.Series, clock: pd.Series) -> pd.Series:
    """post_time_local on the Asia/Tokyo wall clock of the race day, as UTC; NaT when the post time
    is missing or malformed."""
    text = clock.astype("string")
    valid = text.str.fullmatch(CLOCK).fillna(False).astype(bool)
    day = pd.Series([d.isoformat() for d in days], index=days.index, dtype="string")
    local = pd.to_datetime(day + " " + text.where(valid), format="%Y-%m-%d %H:%M", errors="coerce")
    return local.dt.tz_localize(JST).dt.tz_convert("UTC")


def _mean_placed_1_3(runners: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per race id with a placed, timed runner: the top-3 time (each runner placed 1 or 2 once,
    the runners placed 3rd as one slot at their mean) and whether a 3rd-placed runner is present."""
    pos = pd.array(runners["finish_pos"], dtype="Int64")
    secs = pd.to_numeric(runners["finish_time_s"], errors="raise").astype(float).to_numpy()
    ok = np.asarray(pd.Series(pos).isin([1, 2, 3]).fillna(False), dtype=bool) & ~np.isnan(secs)
    frame = pd.DataFrame(
        {
            "race_id": runners["race_id"].to_numpy()[ok],
            "pos": pos[ok].to_numpy(dtype=int),
            "t": secs[ok],
        }
    )
    upper = frame[frame["pos"] <= 2].groupby("race_id")["t"].agg(["sum", "count"])
    third = frame[frame["pos"] == 3].groupby("race_id")["t"].mean()
    ids = upper.index.union(third.index)
    total = upper["sum"].reindex(ids, fill_value=0.0) + third.reindex(ids, fill_value=0.0)
    slots = upper["count"].reindex(ids, fill_value=0) + ids.isin(third.index).astype(int)
    return total / slots, pd.Series(ids.isin(third.index), index=ids)


def build_race_table(
    races: pd.DataFrame, runners: pd.DataFrame, *, estimand: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per kept race for `SpeedModel` and the counts of the races left out (see the
    module docstring). Reads only RACE_COLUMNS of `races` and RUNNER_COLUMNS of `runners`."""
    if not isinstance(estimand, str) or estimand not in ESTIMANDS:
        raise ValueError(f"estimand must be one of {ESTIMANDS}, got {estimand!r}")
    for name, frame, columns in (
        ("races", races, RACE_COLUMNS),
        ("runners", runners, RUNNER_COLUMNS),
    ):
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} lacks column(s): {', '.join(missing)}")
    races = races.loc[:, list(RACE_COLUMNS)].reset_index(drop=True)
    if not races["race_id"].is_unique:
        raise ValueError("races has duplicate race ids")
    counts = {"n_races": len(races)}

    jump = races["surface"].eq(JUMP).fillna(False).to_numpy(dtype=bool)
    counts["n_jump"] = int(jump.sum())
    races = races.loc[~jump].reset_index(drop=True)

    days = _race_days(races["race_date"])
    post = _post_utc(days, races["post_time_local"])
    timed = post.notna().to_numpy()
    counts["n_no_post_time"] = int((~timed).sum())
    races, days, post = races.loc[timed], days[timed], post[timed]

    top3, has_third = _mean_placed_1_3(runners.loc[:, list(RUNNER_COLUMNS)])
    top3_time = races["race_id"].map(top3).astype(float)
    placed = top3_time.notna().to_numpy()
    counts["n_no_placed_time"] = int((~placed).sum())
    races, days, post, top3_time = races[placed], days[placed], post[placed], top3_time[placed]
    counts["n_partly_placed"] = int((~races["race_id"].map(has_third).astype(bool)).sum())

    table = pd.DataFrame(
        {
            "race_id": races["race_id"].to_numpy(),
            "meeting_id": races["meeting_id"].to_numpy(),
            "race_date": days.to_numpy(),
            "post_utc": post.to_numpy(),
            "top3_time_s": top3_time.to_numpy(),
        }
    )
    table["post_utc"] = pd.to_datetime(table["post_utc"], utc=True)
    table.insert(4, "race_utc", table["post_utc"])
    table.insert(5, "result_available_utc", table["post_utc"] + RESULT_LAG)
    for key, column in KEYS.items():
        values = races[column].astype(object)
        table[key] = values.where(values.notna(), UNKNOWN).to_numpy()
    counts["n_kept"] = len(table)
    return table, counts


# --------------------------------------------------------------------------- the fold loop


def fold_label(k: int, test_start: date, test_end: date) -> str:
    return f"{k}:{test_start.isoformat()}..{test_end.isoformat()}"


def _development(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows of `frame` not on a holdout day (#71's `is_holdout_day`, per race day)."""
    days = _race_days(frame["race_date"])
    held = {d for d in set(days) if is_holdout_day(COUNTRY, d)}
    return frame.loc[~days.isin(held).to_numpy()].reset_index(drop=True)


def predict_folds(
    table: pd.DataFrame, *, spec: SpeedSpec, scheme: FoldScheme, decision_min: float
) -> pd.DataFrame:
    """One row per scored development test race: the top-3 time, the three predictions at
    t = post_utc - decision_min, its fold and its cell (see the module docstring). attrs:
    n_unseen_cell, n_holdout_dropped and `folds`, one dict per fold."""
    if (
        isinstance(decision_min, bool)
        or not isinstance(decision_min, Real)
        or not math.isfinite(decision_min)
        or decision_min < 0
    ):
        raise ValueError(f"decision_min must be a non-negative number of minutes: {decision_min!r}")
    dev = _development(table)
    lead = pd.Timedelta(minutes=float(decision_min))
    specs = {
        "variant_on": dataclasses.replace(spec, same_day_variant=True),
        "variant_off": dataclasses.replace(spec, same_day_variant=False),
    }
    rows, folds = [], []
    for k, fold in enumerate(walk_forward(dev["race_date"], scheme=scheme)):
        label = fold_label(k, fold.test_start, fold.test_end)
        train, test = dev.iloc[fold.train_idx], dev.iloc[fold.test_idx]
        models = {}
        for name, variant_spec in specs.items():  # one fit at a time
            models[name] = SpeedModel(variant_spec).fit(train).observe(test)
        scored, unseen = 0, 0
        for race in test.to_dict("records"):
            try:
                par = models["variant_on"].par(**{key: race[key] for key in spec.par_keys})
            except KeyError:
                unseen += 1
                continue
            t = race["post_utc"] - lead
            row = {key: race[key] for key in ("race_id", "meeting_id", "race_date")}
            row |= {"fold": label, "post_utc": race["post_utc"], "decision_utc": t}
            row |= {"top3_time_s": race["top3_time_s"], BASELINE: par}
            row |= {name: m.expected_top3_time(race, at_utc=t) for name, m in models.items()}
            rows.append(row | {key: race[key] for key in spec.par_keys})
            scored += 1
        fitted = models["variant_on"]
        folds.append(
            {"fold": label, "train_end": fold.train_end.isoformat()}
            | {"n_train": fitted.n_races, "n_test": len(test), "n_scored": scored}
            | {"n_unseen_cell": unseen, "n_cells": _n_cells(train, spec.par_keys)}
            | {"sigma_s": fitted.sigma, "tau_s": fitted.tau, "n_iter": fitted.n_iter}
        )
    columns = ["race_id", "meeting_id", "race_date", "fold", "post_utc", "decision_utc"]
    columns += ["top3_time_s", BASELINE, *PREDICTIONS, *spec.par_keys]
    out = pd.DataFrame(rows, columns=columns)
    for column in ("post_utc", "decision_utc"):
        out[column] = pd.to_datetime(out[column], utc=True)
    out.attrs["n_unseen_cell"] = sum(f["n_unseen_cell"] for f in folds)
    out.attrs["n_holdout_dropped"] = len(table) - len(dev)
    out.attrs["folds"] = folds
    return out


def _n_cells(frame: pd.DataFrame, keys: Sequence[str]) -> int:
    return int(len(frame.drop_duplicates(list(keys))))


# --------------------------------------------------------------------------- RMSE and its bounds


@dataclass(frozen=True)
class RmseDiff:
    """The two RMSEs (s) over the same races, gain = rmse_baseline - rmse_model, and D7's
    one-sided bounds on the gain: lo the (1 - level) quantile of the replicates, hi the level
    quantile ([lo, hi] is not a two-sided interval at level)."""

    rmse_model: float
    rmse_baseline: float
    gain: float
    lo: float
    hi: float
    n_races: int
    n_clusters: int


def rmse_diff_bootstrap(
    model_err: pd.Series,
    baseline_err: pd.Series,
    clusters: pd.Series,
    *,
    n_boot: int,
    level: float,
    seed: int,
) -> RmseDiff:
    """The RMSE gain of the model over the baseline with meeting-clustered paired bounds.

    The errors (prediction - actual, s) are indexed by race id and cover the same races;
    `clusters` maps race id to its resampling unit (D7: the meeting) and may list other races.
    Each replicate draws as many clusters as there are, with replacement, keeps every race of each
    drawn cluster and recomputes both RMSEs over the pooled races. `n_boot`, `level` and `seed`
    have no defaults (D7; `config/scoring.yaml`)."""
    if isinstance(level, bool) or not isinstance(level, Real) or not 0.0 < level < 1.0:
        raise ValueError(f"level must be a fraction in (0, 1), got {level!r}")
    if isinstance(n_boot, bool) or not isinstance(n_boot, Integral) or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}")
    if len(model_err) == 0:
        raise ValueError("no races to score")
    for name, s in (
        ("model_err", model_err),
        ("baseline_err", baseline_err),
        ("clusters", clusters),
    ):
        if not s.index.is_unique:
            raise ValueError(f"{name} has duplicate race ids")
    if len(model_err) != len(baseline_err) or not model_err.index.isin(baseline_err.index).all():
        raise ValueError("model_err and baseline_err must cover the same races (paired)")
    model = model_err.to_numpy(dtype=float)
    base = baseline_err.reindex(model_err.index).to_numpy(dtype=float)
    if not (np.isfinite(model).all() and np.isfinite(base).all()):
        raise ValueError("an error is missing or not finite")
    cluster_of = clusters.reindex(model_err.index)
    if cluster_of.isna().any():
        raise ValueError(f"{int(cluster_of.isna().sum())} race(s) have no cluster")

    codes, _ = pd.factorize(cluster_of, sort=True)
    n_clusters = int(codes.max()) + 1
    sq_model = np.bincount(codes, weights=model**2, minlength=n_clusters)
    sq_base = np.bincount(codes, weights=base**2, minlength=n_clusters)
    sizes = np.bincount(codes, minlength=n_clusters).astype(float)

    rng = np.random.default_rng(seed)
    chunk = max(1, _MAX_DRAWS_PER_CHUNK // n_clusters)
    gains = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        drawn = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
        n = sizes[drawn].sum(axis=1)
        gains[start:stop] = np.sqrt(sq_base[drawn].sum(axis=1) / n) - np.sqrt(
            sq_model[drawn].sum(axis=1) / n
        )
    rmse_model = math.sqrt(float(np.mean(model**2)))
    rmse_base = math.sqrt(float(np.mean(base**2)))
    return RmseDiff(
        rmse_model=rmse_model,
        rmse_baseline=rmse_base,
        gain=rmse_base - rmse_model,
        lo=float(np.quantile(gains, 1.0 - level)),
        hi=float(np.quantile(gains, level)),
        n_races=len(model),
        n_clusters=n_clusters,
    )


def rmse_table(
    predictions: pd.DataFrame,
    *,
    model: str,
    baseline: str,
    by: Sequence[str],
    n_boot: int,
    level: float,
    seed: int,
) -> pd.DataFrame:
    """scope ("overall" or "cell"), the `by` columns (null on the overall row), n_races,
    n_meetings, rmse_model, rmse_baseline, gain, lo, hi: the overall row, then one row per
    distinct `by` value, each bounded over the meetings of its own races."""
    by = list(by)
    ids = predictions["race_id"].to_numpy()
    actual = predictions["top3_time_s"].to_numpy(dtype=float)
    err_model = pd.Series(predictions[model].to_numpy(dtype=float) - actual, index=ids)
    err_base = pd.Series(predictions[baseline].to_numpy(dtype=float) - actual, index=ids)
    clusters = pd.Series(predictions["meeting_id"].to_numpy(), index=ids)
    boot = {"n_boot": n_boot, "level": level, "seed": seed}

    def row(scope: str, key: dict, idx: np.ndarray) -> dict:
        d = rmse_diff_bootstrap(err_model.iloc[idx], err_base.iloc[idx], clusters, **boot)
        return {"scope": scope, **key, "n_races": d.n_races, "n_meetings": d.n_clusters} | {
            "rmse_model": d.rmse_model,
            "rmse_baseline": d.rmse_baseline,
            "gain": d.gain,
            "lo": d.lo,
            "hi": d.hi,
        }

    rows = [row("overall", dict.fromkeys(by), np.arange(len(predictions)))]
    keys = list(zip(*(predictions[c].tolist() for c in by), strict=True))
    groups: dict[tuple, list[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    for key in sorted(groups, key=lambda k: tuple(str(v) for v in k)):
        rows.append(row("cell", dict(zip(by, key, strict=True)), np.asarray(groups[key])))
    columns = ["scope", *by, "n_races", "n_meetings", "rmse_model", "rmse_baseline"]
    out = pd.DataFrame(rows, columns=[*columns, "gain", "lo", "hi"])
    for c in by:  # keep each key's own type (an int distance stays an int)
        out[c] = pd.Series([r[c] for r in rows], dtype=object)
    return out


# --------------------------------------------------------------------------- reading


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The JRA store's races (RACE_COLUMNS) and runners (RUNNER_COLUMNS and race_date),
    column-scoped, holdout days dropped. No JRDB table is read (D21)."""
    races = read_table(SOURCE, "races", columns=list(RACE_COLUMNS))
    runners = read_table(SOURCE, "runners", columns=[*RUNNER_COLUMNS, "race_date"])
    return _development(races), _development(runners)


# --------------------------------------------------------------------------- the study


def bootstrap_settings() -> dict:
    """D7's bootstrap from config/scoring.yaml, with this study's fixed seed."""
    boot = load_country_yaml(COUNTRY, "scoring")["bootstrap"]
    if boot["cluster_unit"] != "meeting" or boot["sided"] != "one-sided":
        raise ValueError(f"this study implements meeting clusters, one-sided; scoring.yaml: {boot}")
    return {"cluster_unit": "meeting", "sided": "one-sided", "level": float(boot["level"])} | {
        "n_boot": int(boot["n_boot"]),
        "seed": BOOTSTRAP_SEED,
    }


def decision_min() -> int:
    """D2's primary decision time, minutes before post (config/decision.yaml)."""
    return int(load_country_yaml(COUNTRY, "decision")["primary_min"])


def scheme_label(scheme: FoldScheme) -> str:
    return "walk_forward:" + json.dumps(dataclasses.asdict(scheme), sort_keys=True)


def build_spec(boot: dict, scheme: FoldScheme, minutes: int) -> dict:
    """The family's pre-registered spec, frozen before any outcome is read."""
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypotheses": {
            "null": "the same-day variant does not lower the time RMSE of E[top-3 time] below the "
            "par of the race's cell: gain = rmse_par_only - rmse_variant_on <= 0",
            HYPOTHESIS_ID: "with the same-day variant from races resulted before t (post + 30 "
            "min), E[top-3 time] has a lower time RMSE than the par alone on development test "
            "folds; the claim rests on the one-sided lower bound of the overall gain being above 0",
            "check": "without an injected prior the variant off is the par: its gain is 0",
        },
        "estimand": ESTIMANDS[0] + ": the mean finish time of the first three past the post, "
        "per runner; each runner placed 1 or 2 once, the runners placed 3rd as one slot at their "
        "mean; without a 3rd-placed runner, the runners placed 1-2 present (partly placed)",
        "races": "store races and runners (jra), every development race day; excluded in order "
        "and counted: jump races, no post time, no placed and timed runner",
        "post_utc": "post_time_local (H:MM or HH:MM) on the Asia/Tokyo wall clock of race_date",
        "result_available_utc": "post_utc + 30 min (availability.yaml's same-day results rule)",
        "model": {
            "module": "countries.japan.models.speed (#96), unchanged",
            "par_keys": list(PAR_KEYS),
            "par_key_columns": dict(KEYS),
            "variant_keys": list(VARIANT_KEYS),
            "error_models": dict(ERROR_MODELS),
            "primary_error_model": "t5: #96's Student-t error, at the df of #96's tests (the df "
            "is not signed); the normal error is run beside it",
            "variant_prior": "none: no non-JRDB prior exists (#90/#230 not built)",
            "null_keys": UNKNOWN,
        },
        "decision_time_min": minutes,
        "fold_scheme": scheme_label(scheme),
        "predictions": {
            "variant_on": "par + the same-day variant's mean at t from the day's results "
            "available strictly before t (same_day_variant=True)",
            "variant_off": "par + the variant prior's mean (same_day_variant=False)",
            BASELINE: "the par of the race's cell, fitted on the fold's training races",
            "unseen_cell": "a test race whose cell its fold's fit never saw is not scored, counted",
        },
        "metric": "time RMSE (s) over the scored races; gain = rmse_par_only - rmse_model",
        "by": {
            "cells": list(PAR_KEYS),
            "margins": list(MARGINS),
            "results_seen": "diagnostic: the races of the race's (day, track, surface) with a "
            "result available strictly before t, capped at 6",
        },
        "bootstrap": boot,
        "decisions": {
            "D2": f"decision time T-{minutes}, the grid's primary point",
            "D7": "meeting clusters (venue x day), 95% one-sided",
            "D9": "#70's quarterly expanding folds, 1-week embargo (config/folds.yaml)",
            "D10": "no price is read, so rows before 2014-06-07 may train",
            "D11-D15": "Japan holdout A2 from 2026-11-28, fully locked: no holdout day is read",
            "D16": "nothing is adopted",
            "D21": "no licensed source (JRDB, JRA-VAN) is read",
        },
        "sources": [SOURCE],
    }


@dataclass(frozen=True)
class Variant:
    name: str
    components: dict[str, str]
    params: dict
    metrics: tuple[FoldMetric, ...] = ()
    error: str | None = None


def _metric(fold: str, metric: str, value: float, n: int, lo=None, hi=None) -> FoldMetric:
    value, lo, hi = (None if v is None else float(v) for v in (value, lo, hi))
    if not all(np.isfinite(v) for v in (value, lo, hi) if v is not None):
        raise ValueError(f"{metric} on {fold} is not finite: {value}, [{lo}, {hi}]")
    return FoldMetric(str(fold), metric, value, lo, hi, int(n))


def variant_metrics(predictions: pd.DataFrame, model: str, overall: pd.Series) -> list[FoldMetric]:
    """The overall gain with D7's bounds and both RMSEs (`overall`, rmse_table's overall row),
    then the gain on each fold."""
    n = int(overall["n_races"])
    metrics = [
        _metric("all", "rmse_gain_s", overall["gain"], n, overall["lo"], overall["hi"]),
        _metric("all", "rmse_s", overall["rmse_model"], n),
        _metric("all", "rmse_par_only_s", overall["rmse_baseline"], n),
    ]
    for label, part in predictions.groupby("fold", sort=False):
        actual = part["top3_time_s"].to_numpy(dtype=float)
        rmse = [
            math.sqrt(float(np.mean((part[c].to_numpy(dtype=float) - actual) ** 2)))
            for c in (BASELINE, model)
        ]
        metrics.append(_metric(label, "rmse_gain_s", rmse[0] - rmse[1], len(part)))
    return metrics


def by_results_seen(predictions: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    """Per scored race: how many races of its (day, track, surface) had a result available
    strictly before its decision time, the data the same-day variant reads (before any cell
    filter)."""
    group = ["race_date", "track", "surface"]
    avail = table[[*group, "result_available_utc"]]
    merged = predictions[["race_id", *group, "decision_utc"]].merge(avail, on=group, how="left")
    seen = merged["result_available_utc"] < merged["decision_utc"]
    return seen.groupby(merged["race_id"]).sum().reindex(predictions["race_id"]).astype(int)


def compute_study(
    table: pd.DataFrame, spec: dict, scheme: FoldScheme, minutes: int
) -> tuple[list[Variant], dict, dict]:
    """Per error model, the fold loop and its scored variants; the tables for the report; the
    run counts. A failed error model is two failed variants, never dropped."""
    boot = spec["bootstrap"]
    kwargs = {k: boot[k] for k in ("n_boot", "level", "seed")}
    variants, tables, counts = [], {}, {}
    for error_name, t_df in ERROR_MODELS.items():
        speed = SpeedSpec(PAR_KEYS, VARIANT_KEYS, same_day_variant=True, t_df=t_df)
        params = {"error_model": error_name, "t_df": t_df, "decision_time_min": minutes}
        params |= {"bootstrap": boot, "estimand": ESTIMANDS[0], "variant_prior": None}
        try:
            predictions = predict_folds(table, spec=speed, scheme=scheme, decision_min=minutes)
            if predictions.empty:
                raise ValueError("no development test race was scored")
        except Exception as exc:  # every failure is a registry record, never dropped
            for model in PREDICTIONS:
                variants.append(_variant(error_name, model, params, error=_why(exc)))
            continue
        counts[error_name] = {
            "n_scored": len(predictions),
            "n_meetings": int(predictions["meeting_id"].nunique()),
            "n_folds": len(predictions.attrs["folds"]),
            "n_unseen_cell": predictions.attrs["n_unseen_cell"],
            "n_holdout_dropped": predictions.attrs["n_holdout_dropped"],
            "first_test_day": str(min(predictions["race_date"])),
            "last_test_day": str(max(predictions["race_date"])),
        }
        tables[f"folds:{error_name}"] = predictions.attrs["folds"]
        seen = by_results_seen(predictions, table).clip(upper=6)
        for model in PREDICTIONS:
            try:
                by_cell = rmse_table(
                    predictions, model=model, baseline=BASELINE, by=PAR_KEYS, **kwargs
                )
                metrics = variant_metrics(predictions, model, by_cell.iloc[0])
                tables[f"cells:{error_name}:{model}"] = by_cell
                for margin in MARGINS:
                    tables[f"by_{margin}:{error_name}:{model}"] = rmse_table(
                        predictions, model=model, baseline=BASELINE, by=(margin,), **kwargs
                    )
                tables[f"by_results_seen:{error_name}:{model}"] = rmse_table(
                    predictions.assign(results_seen=seen.to_numpy()),
                    model=model,
                    baseline=BASELINE,
                    by=("results_seen",),
                    **kwargs,
                )
            except Exception as exc:
                variants.append(_variant(error_name, model, params, error=_why(exc)))
                continue
            variants.append(_variant(error_name, model, params, metrics=tuple(metrics)))
        del predictions
    return variants, tables, counts


def _why(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _variant(error_name: str, model: str, params: dict, **outcome: Any) -> Variant:
    components = {"model": "countries.japan.models.speed.SpeedModel", "prediction": model}
    components |= {"baseline": BASELINE, "error_model": error_name, "estimand": ESTIMANDS[0]}
    return Variant(f"{error_name}:{model}_vs_{BASELINE}", components, params, **outcome)


def _sha_prefix(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def store_builds() -> dict[str, str]:
    return {
        f"{SOURCE}_{name}": f"processed/{SOURCE}/{name}.parquet@"
        + _sha_prefix(table_path(SOURCE, name))
        for name in ("races", "runners")
    }


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


def _cell(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def markdown_table(rows: list[dict] | pd.DataFrame) -> str:
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")
    if not rows:
        return "_none_"
    head = list(dict.fromkeys(key for row in rows for key in row))
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    lines += ["| " + " | ".join(_cell(row.get(k)) for k in head) + " |" for row in rows]
    return "\n".join(lines)


def render_report(variants: list[Variant], tables: dict, counts: dict, about: dict) -> str:
    """Aggregates only (no race row): counts, folds, the overall and by-cell RMSE tables."""
    lines = [f"# Expected top-3 time against the par-only baseline ({COUNTRY}, #{TICKET})", ""]
    lines += [f"- {key}: {value}" for key, value in about.items()]
    lines += ["- development days only: holdout days are never read and the holdout is not opened"]
    lines += ["", "## Race table", "", markdown_table([counts["builder"]])]
    runs = [{"error_model": k, **v} for k, v in counts.items() if k in ERROR_MODELS]
    lines += ["", "## Runs", "", markdown_table(runs)]
    rows = [
        {"variant": v.name, "status": "failed" if v.error else "scored"}
        | {m.metric: m.value for m in v.metrics if m.fold == "all"}
        | {
            "lo": next((m.lo for m in v.metrics if m.metric == "rmse_gain_s"), None),
            "hi": next((m.hi for m in v.metrics if m.metric == "rmse_gain_s"), None),
        }
        | ({"error": v.error} if v.error else {})
        for v in variants
    ]
    lines += ["", "## Variants (one registry record each)", "", markdown_table(rows)]
    for key, frame in tables.items():
        title = key.replace(":", ", ")
        lines += ["", f"## {title}", "", markdown_table(frame)]
    return "\n".join(lines) + "\n"


def run_study(out_dir: Path | None = None) -> dict:
    """Freeze the spec, build the race table from the development store, run the fold loop per
    error model, append the registry records, rewrite the committed summary and write the
    report outside the tree."""
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    scheme, minutes = load_country_scheme(COUNTRY), decision_min()
    spec = build_spec(bootstrap_settings(), scheme, minutes)
    git_sha = _git_sha()
    registry = default_registry(COUNTRY)
    spec_hash = registry.freeze_family(FAMILY_ID, spec)  # before any outcome is read

    builds = store_builds()
    races, runners = read_inputs()
    table, builder = build_race_table(races, runners, estimand=ESTIMANDS[0])
    del races, runners
    builder |= {
        "first_race_day": str(min(table["race_date"])),
        "last_race_day": str(max(table["race_date"])),
    }
    variants, tables, counts = compute_study(table, spec, scheme, minutes)
    counts = {"builder": builder} | counts

    existing = registry.records(FAMILY_ID)
    fold_scheme = scheme_label(scheme)
    records = [_record(v, git_sha, builds, fold_scheme) for v in variants]
    run_ids = {}
    for variant, record in zip(variants, records, strict=True):
        if record not in existing:
            run_ids[variant.name] = registry.append(record)
    write_summary(registry, summary_path(COUNTRY))
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "store": ", ".join(builds.values()),
        "run ids": ", ".join(f"{k} `{v}`" for k, v in run_ids.items()) or "already recorded",
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = write_report(
        out_dir / f"speed-{COUNTRY}-{spec_hash[:12]}.md",
        render_report(variants, tables, counts, about),
        sources=(SOURCE,),
    )
    return {
        "family_id": FAMILY_ID,
        "spec_hash": spec_hash,
        "report_path": str(report),
        "registry_path": str(registry.path),
        "run_ids": run_ids,
        "counts": counts,
        "failed_variants": {v.name: v.error for v in variants if v.error},
        "overall": {
            key: frame.iloc[0].to_dict()
            for key, frame in tables.items()
            if key.startswith("cells:")
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m horsetrack.countries.japan.studies.speed_run")
    parser.add_argument("--run", action="store_true", required=True, help="run the study")
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    args = parser.parse_args(argv)
    print(json.dumps(run_study(args.out), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
