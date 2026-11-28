"""E39: #97's p_speed standalone and marginal at the final Win price (#347, epic #73).

#97's definition-of-done item 2 (08 §3.2 C3, §4 E39). The null is that p_speed adds nothing to
the final price: alpha = 0 in #87's stage-2 meta, and no out-of-fold gain over the market-only
meta. 08 §4 expects at most +0.0005 pseudo-R², plausibly 0.

- `predict_folds`: holdout days (`is_holdout_day`, the JST race day) are dropped from the runs
  and the race table first. Then #70's folds over the remaining race days. Per fold, #96's
  `SpeedModel(spec)` is fitted on the training races and observes the test races; #97's
  `HorseSpeed(drift_var_per_day, t_df)` is fitted on the training races' runs and observes the
  test races' runs in one batch (the owner on #97: per fold, test days in batches, never race by
  race; batch and incremental replay agree by HorseSpeed's contract). Each test race's runners
  are predicted at t = race_utc - decision_min. A race whose cell has no par in the fold's fit,
  or with a runner without a finite weight carried, is not predicted and is counted.
- `fit_scale`: PL's scale is 1 / the conditional-logit coefficient of the winner on -mu
  (`models.condlogit`); Henery's is its own one-parameter maximum likelihood (the winners'
  log-likelihood under `speed_horse.p_speed(..., "henery")`, computed race-vectorised).
  `score_folds` fits fold k's scale on the rows of earlier folds dated on or before fold k's
  train_end (a fold without such rows only trains) and gives p_speed per race.
- `standalone`: per race, #69's winner log loss of D4's calibrated market (out of fold) less
  p_speed's; D7's meeting bootstrap.
- `marginal`: p_speed in #87's stage-2 meta next to ln q (u = alpha ln p_speed + beta ln q,
  ridge 0, so alpha = 0 is the market alone, the shrinkage target): alpha pooled over the rows with
  its meeting-clustered se and a 95% two-sided interval; and #92's C4 marginal, the paired
  out-of-fold ΔLL of that meta against the market-only stage-2 meta (u = beta ln q), both fitted
  on the earlier folds' rows up to each fold's train_end, from the rows' second fold. A fold whose
  training rows give either meta no finite fit (singular information, no finite maximum) only
  trains and is counted; both metas are scored from their log-softmax, never an underflowed p.
- `compute_study`: holdout days dropped from every input; D4 out of fold on the market's own #70
  folds (the same calendar quarters as the speed folds); every grid point x method two variants
  (standalone, marginal), failures included. The speed model's fit per fold is shared by the grid
  points; one HorseSpeed fit at a time. spec["predict_from"] limits the test windows a point is
  predicted on (PREDICT_FROM for the run: the normal points over D4's whole window, the
  Student-t points, whose fit is about twelve times slower, over the last three quarters).

`run_study` freezes the family's spec before any store row is read, reads the JRA store (results
from netkeiba; no JRDB, no JRA-VAN: nothing here is withheld under D21) column- and date-scoped
with holdout days dropped by the scan, appends one registry record per variant, writes the report
through `write_report` and then the committed registry summary. CLI: `python -m
horsetrack.countries.japan.studies.p_speed --run [--out DIR]`. Nothing is adopted (D16); the
holdout is never opened.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import sys
from collections.abc import Sequence
from datetime import date, datetime, timezone
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr

from horsetrack.core import paths
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day
from horsetrack.core.eval.registry import RunRecord, default_registry, summary_dir, write_summary
from horsetrack.core.eval.reports import write_report
from horsetrack.core.eval.scoring import per_race_log_loss
from horsetrack.countries.japan.models import speed_runs
from horsetrack.countries.japan.models.condlogit import fit_conditional_logit
from horsetrack.countries.japan.models.meta import MetaSpec, fit_meta
from horsetrack.countries.japan.models.speed import JST, SpeedModel, SpeedSpec
from horsetrack.countries.japan.models.speed_horse import (
    HENERY_SPAN,
    HENERY_STEP,
    HorseSpeed,
    p_speed,
)
from horsetrack.countries.japan.studies import calibration, speed_run
from horsetrack.countries.japan.studies import meta as meta_study
from horsetrack.countries.japan.studies.calibration import Study, Variant, _markdown_table, _metric

COUNTRY = "japan"
TICKET = 347
HYPOTHESIS_ID = "E39"
FAMILY_ID = "japan-c3-p-speed-final-win"
SOURCE = calibration.SOURCE  # the store's JRA results, from netkeiba; no JRDB, no JRA-VAN
BOOTSTRAP_SEED = 347  # reproducibility only; scoring.yaml signs no seed
Z95 = calibration.BETA_CI_Z
METHODS = ("pl", "henery")
# The pre-registered grid (see build_spec): a static ability and #97's tests' drift, each under
# the normal error and #305's Student-t (5 df).
GRID = {"drift_var_per_day": [0.0, 0.004], "t_df": [None, 5.0]}
# The test windows predicted (runtime, measured before the freeze on the largest fold, 788,150
# training runs: HorseSpeed's normal fit 77 s, its Student-t fit 900 s over 18 reweighting
# rounds). Every fold still trains on every earlier run (D9, expanding). The normal points from
# 2019-Q4, whose scale trains D4's first scored quarter, 2020-Q1; the Student-t points on
# 2026-Q2..Q4 only: Q2 trains the scale, Q3 the meta, Q4 is the marginal's.
PREDICT_FROM = {"normal": date(2019, 10, 1), "student_t": date(2026, 4, 1)}
WINDOW_RATIONALE = (
    "runtime: a Student-t HorseSpeed fit takes about 15 min per fold at 788k training runs (a "
    "normal one about 1.3 min), so the Student-t points are predicted on the last three "
    "quarters only and the normal points on D4's whole window (from the quarter before its "
    "first); a full-window Student-t rerun waits on a faster fit"
)
SPEED_T_DF = 5.0  # #96's race-level error: #305's primary
DECISION_MIN = 0.0  # the ticket: each test race predicted at the off
MEMBER = "p_speed"
CLUSTER = "meeting_id"
STAGE2 = MetaSpec([MEMBER], quadratic=False, segment_cols=[], form="stage2", ridge=0.0)
HENERY_GRID = np.arange(-HENERY_SPAN, HENERY_SPAN + HENERY_STEP / 2, HENERY_STEP)  # p_speed's
HENERY_NORM = HENERY_STEP / math.sqrt(2.0 * math.pi)  # the grid's rule for the normal density
HENERY_BOUNDS = (0.01, 100.0)  # s: the search range of Henery's scale
HENERY_CHUNK = 20_000  # rows per vectorised block of the Henery likelihood
RESULTS_FROM = date(2000, 1, 1)  # the runs read every stored day (no price is read, D10)
PREDICTED = ["race_id", "horse_id", "fold", "train_end", "test_start", "decision_utc"]
PREDICTED += ["mu_s", "sd_s"]


# --------------------------------------------------------------------------- development rows


def _held(days: pd.Series) -> np.ndarray:
    """Whether each race day is a holdout day (#71's `is_holdout_day`, once per distinct day)."""
    held = {d for d in set(days) if is_holdout_day(COUNTRY, d)}
    return days.isin(held).to_numpy(dtype=bool)


def _jst_days(stamps: pd.Series) -> pd.Series:
    local = pd.to_datetime(stamps, utc=True).dt.tz_convert(JST).dt.date
    return pd.Series(local.to_numpy(), index=stamps.index, dtype=object)


def _development(runs: pd.DataFrame | None, table: pd.DataFrame | None, market=None):
    """The frames without their holdout days' rows: the table's and the market's by race_date,
    the runs' by the JST day of race_utc and by the table's holdout races."""
    held_races: set = set()
    if table is not None:
        held = _held(speed_run._race_days(table["race_date"]))
        held_races = set(table.loc[held, "race_id"])
        table = table.loc[~held].reset_index(drop=True)
    if runs is not None:
        held = _held(_jst_days(runs["race_utc"])) | runs["race_id"].isin(held_races).to_numpy()
        runs = runs.loc[~held].reset_index(drop=True)
    if market is not None:
        held = _held(speed_run._race_days(market["race_date"]))
        market = market.loc[~held].reset_index(drop=True)
    return runs, table, market


# --------------------------------------------------------------------------- the fold loop


def _minutes(decision_min) -> float:
    if isinstance(decision_min, bool) or not isinstance(decision_min, Real):
        raise ValueError(f"decision_min must be a non-negative number of minutes: {decision_min!r}")
    if not math.isfinite(decision_min) or decision_min < 0:
        raise ValueError(f"decision_min must be a non-negative number of minutes: {decision_min!r}")
    return float(decision_min)


def _cards(test_runs: pd.DataFrame, speed: SpeedModel, lead: pd.Timedelta, counts: dict):
    """The test races to predict, (race_id, card, t), and the skipped ones counted."""
    cards = []
    for rid, card in test_runs.groupby("race_id", sort=False):
        try:
            speed.par(**{k: card[k].iloc[0] for k in speed.spec.par_keys})
        except KeyError:
            counts["n_unseen_cell"] += 1
            continue
        if not np.isfinite(card["weight_kg"].to_numpy(dtype=float)).all():
            counts["n_no_weight"] += 1
            continue
        cards.append((rid, card, card["race_utc"].iloc[0] - lead))
    return cards


def _predict_points(runs, table, *, spec, scheme, points, decision_min, predict_from=None):
    """`predict_folds` for every (drift, t_df) of `points` (one SpeedModel fit per fold, shared;
    one HorseSpeed fit at a time): point -> its frame (attrs: counts and per-fold fits), or the
    exception that failed it. `predict_from` maps a point to the first test window it predicts
    (absent or None: every fold); the earlier folds' races still train its later folds."""
    lead = pd.Timedelta(minutes=_minutes(decision_min))
    runs, table, _ = _development(runs, table)
    first = dict(predict_from or {})
    failed: dict = {}
    for p in points:
        try:
            HorseSpeed(drift_var_per_day=p[0], t_df=p[1])
        except Exception as exc:  # a point the model refuses: its variants fail, the rest run
            failed[p] = exc
    parts: dict = {p: [] for p in points}
    fits: dict = {p: [] for p in points}
    counts = {p: {"n_unseen_cell": 0, "n_no_weight": 0, "n_folds": 0} for p in points}
    race_of_run = runs["race_id"]
    for k, fold in enumerate(walk_forward(table["race_date"], scheme=scheme)):
        if all(p in failed for p in points):
            break
        live = [
            p
            for p in points
            if p not in failed and (first.get(p) is None or fold.test_start >= first[p])
        ]
        if not live:
            continue
        try:
            train, test = table.iloc[fold.train_idx], table.iloc[fold.test_idx]
            speed = SpeedModel(spec).fit(train).observe(test)
            train_runs = runs.loc[race_of_run.isin(train["race_id"]).to_numpy()]
            test_runs = runs.loc[race_of_run.isin(test["race_id"]).to_numpy()]
            skipped = {"n_unseen_cell": 0, "n_no_weight": 0}
            cards = _cards(test_runs, speed, lead, skipped)
        except Exception as exc:  # the fold's race-level fit fails every point it predicts
            failed |= dict.fromkeys(live, exc)
            continue
        for p in live:
            counts[p]["n_folds"] += 1
            for name, n in skipped.items():
                counts[p][name] += n
            try:
                horse = HorseSpeed(drift_var_per_day=p[0], t_df=p[1]).fit(train_runs, speed)
                horse.observe(test_runs)
                for rid, card, t in cards:
                    got = horse.predict(card, at_utc=t.to_pydatetime())
                    parts[p].append(
                        pd.DataFrame(
                            {
                                "race_id": rid,
                                "horse_id": got["horse_id"].to_numpy(dtype=object),
                                "fold": k,
                                "train_end": fold.train_end,
                                "test_start": fold.test_start,
                                "decision_utc": t,
                                "mu_s": got["mu_s"].to_numpy(dtype=float),
                                "sd_s": got["sd_s"].to_numpy(dtype=float),
                            }
                        )
                    )
            except Exception as exc:
                failed[p] = exc
                continue
            fits[p].append(
                {"fold": k, "test start": fold.test_start.isoformat()}
                | {"train end": fold.train_end.isoformat(), "fit runs": horse.n_runs}
                | {"observed runs": horse.n_observed, "predicted races": len(cards)}
                | {"sigma": horse.sigma, "prior_sd": horse.prior_sd, "prior_mean": horse.prior_mean}
                | {"weight_slope": horse.weight_slope, "pace_race": horse.pace_race}
                | {"rounds": horse.n_iter, "speed_sigma": speed.sigma, "speed_tau": speed.tau}
            )
            del horse
        del speed, train_runs, test_runs, cards
    out = {}
    for p in points:
        if p in failed:
            out[p] = failed[p]
            continue
        frame = (
            pd.concat(parts[p], ignore_index=True) if parts[p] else pd.DataFrame(columns=PREDICTED)
        )
        frame["decision_utc"] = pd.to_datetime(frame["decision_utc"], utc=True)
        frame.attrs = {"counts": counts[p], "fits": fits[p]}
        out[p] = frame
    return out


def predict_folds(
    runs: pd.DataFrame,
    table: pd.DataFrame,
    *,
    spec: SpeedSpec,
    scheme: FoldScheme,
    drift_var_per_day: float,
    t_df: float | None,
    decision_min: float,
) -> pd.DataFrame:
    """One row per run of every fold's test races, as of its race's t (see the module doc):
    race_id, horse_id, fold, train_end, test_start, decision_utc, mu_s, sd_s."""
    point = (drift_var_per_day, t_df)
    got = _predict_points(
        runs, table, spec=spec, scheme=scheme, points=[point], decision_min=decision_min
    )[point]
    if isinstance(got, Exception):
        raise got
    return got


# --------------------------------------------------------------------------- the scale


def _race_order(rows: pd.DataFrame):
    """Rows of races with a winner sorted by race: positions, race starts, sizes and each
    row's winner weight (1/k on a race's k winners, else 0)."""
    codes = pd.factorize(rows["race_id"].to_numpy(dtype=object))[0]
    won = rows["won"].to_numpy(dtype=bool)
    winners = np.bincount(codes, weights=won)
    keep = np.flatnonzero(winners[codes] > 0)
    order = keep[np.argsort(codes[keep], kind="stable")]
    c = codes[order]
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]]) if len(c) else np.zeros(0, int)
    sizes = np.diff(np.r_[starts, len(c)])
    return order, starts, sizes, won[order] / winners[c]


def _henery_blocks(rows: pd.DataFrame) -> list[tuple]:
    """Henery's likelihood input in blocks of whole races: (mu - race min, starts, race code per
    row, winner rows, their weights)."""
    order, starts, sizes, weight = _race_order(rows)
    if len(order) == 0:
        raise ValueError("no race has a winner to fit Henery's scale on")
    mu = rows["mu_s"].to_numpy(dtype=float)[order]
    if not np.isfinite(mu).all():
        raise ValueError("mu_s has a missing or non-finite value")
    rel = mu - np.repeat(np.minimum.reduceat(mu, starts), sizes)
    blocks, lo = [], 0
    ends = np.r_[starts[1:], len(order)]
    while lo < len(starts):  # races [lo, hi): at most HENERY_CHUNK rows, at least one race
        hi = max(lo + 1, int(np.searchsorted(ends, starts[lo] + HENERY_CHUNK, side="right")))
        a, b = starts[lo], ends[hi - 1]
        code = np.repeat(np.arange(hi - lo), sizes[lo:hi])
        win = np.flatnonzero(weight[a:b] > 0)
        blocks.append((rel[a:b], starts[lo:hi] - a, code, win, weight[a:b][win]))
        lo = hi
    return blocks


def _henery_loglik(blocks: list[tuple], scale: float) -> float:
    """The winners' log-likelihood under `p_speed(mu, scale, "henery")`, race by race, each
    race's winners averaged (#69's dead-heat rule), on p_speed's grid. p_speed's renormalising
    sum is HENERY_STEP / sqrt(2 pi) times its integrals' sum, which is 1 to within 1e-14 (the
    rule is spectrally accurate here), so only the winners' integrals are taken."""
    total = 0.0
    for rel, starts, code, win, weight in blocks:
        z = (rel / scale)[:, None]
        log_sf = log_ndtr(z - HENERY_GRID[None, :])
        log_sf_race = np.add.reduceat(log_sf, starts, axis=0)
        d = HENERY_GRID[None, :] - z[win]
        p = np.exp(-0.5 * d * d + log_sf_race[code[win]] - log_sf[win]).sum(axis=1)
        total += float(weight @ np.log(p * HENERY_NORM))
    return total


def _pl_scale(rows: pd.DataFrame) -> float:
    fit = fit_conditional_logit(
        -rows[["mu_s"]].to_numpy(dtype=float), rows["race_id"].to_numpy(), rows["won"].to_numpy()
    )
    coef = float(fit.coef[0])
    if not coef > 0:
        raise ValueError(f"the winner's coefficient on -mu is {coef}: no positive PL scale")
    return 1.0 / coef


def _henery_scale(rows: pd.DataFrame) -> float:
    """Henery's ML scale: Brent on ln s, first in a bracket around the scale PL's implies (a
    Gumbel error's SD, pi / sqrt(6) times PL's scale), then, should the maximum fall on its edge
    or PL have no scale, over HENERY_BOUNDS."""
    blocks = _henery_blocks(rows)
    full = tuple(math.log(b) for b in HENERY_BOUNDS)
    try:
        centre = math.log(_pl_scale(rows) * math.pi / math.sqrt(6.0))
        brackets = [(max(full[0], centre - 0.5), min(full[1], centre + 0.5)), full]
    except (ValueError, RuntimeError):
        brackets = [full]
    for lo, hi in brackets:
        found = minimize_scalar(
            lambda x: -_henery_loglik(blocks, math.exp(x)),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-6},
        )
        if found.success and lo + 1e-3 < found.x < hi - 1e-3:
            return math.exp(float(found.x))
    raise ValueError(f"Henery's scale has no interior maximum in {HENERY_BOUNDS} s")


def fit_scale(rows: pd.DataFrame, *, method: str = "pl") -> float:
    """The PL or Henery scale (s) of p_speed on rows with race_id, won and mu_s (module doc)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}: {method!r}")
    return _pl_scale(rows) if method == "pl" else _henery_scale(rows)


def _p_within(rows: pd.DataFrame, scale: float, method: str) -> np.ndarray:
    """`speed_horse.p_speed` of each race's mu_s, in row order."""
    codes = pd.factorize(rows["race_id"].to_numpy(dtype=object))[0]
    mu = rows["mu_s"].to_numpy(dtype=float)
    out = np.empty(len(rows))
    order = np.argsort(codes, kind="stable")
    for idx in np.split(order, np.flatnonzero(np.diff(codes[order])) + 1):
        if idx.size:
            out[idx] = p_speed(mu[idx], scale=scale, method=method)
    return out


def score_folds(rows: pd.DataFrame, *, method: str) -> pd.DataFrame:
    """Each scored fold's rows with its `scale` and `p_speed` (module doc); fold 0 only trains."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}: {method!r}")
    rows = rows.reset_index(drop=True)
    fold = rows["fold"].to_numpy()
    day = pd.to_datetime(rows["race_date"]).to_numpy()
    parts = []
    for k in sorted(pd.unique(fold)):
        mine = fold == k
        end = pd.Timestamp(rows.loc[mine, "train_end"].iloc[0]).to_datetime64()
        train = (fold < k) & (day <= end)
        if not train.any():
            continue
        scale = fit_scale(rows.loc[train], method=method)
        test = rows.loc[mine]
        parts.append(test.assign(scale=scale, p_speed=_p_within(test, scale, method)))
    if not parts:
        return rows.iloc[:0].assign(scale=pd.Series(dtype=float), p_speed=pd.Series(dtype=float))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- scoring


def _ll(p, rows: pd.DataFrame) -> pd.Series:
    """#69's winner log loss per race (p renormalised within race, a dead heat's winners
    averaged), by race id."""
    p = pd.Series(np.asarray(p, dtype=float), index=rows.index)
    no = pd.Series(False, index=rows.index)
    return per_race_log_loss(p, rows["won"].astype(bool), rows["race_id"], no).per_race


def _interval(per_race: pd.Series, rows: pd.DataFrame, spec: dict):
    boot = spec["bootstrap"]
    clusters = rows.groupby("race_id")[CLUSTER].first()
    return paired_bootstrap(
        per_race, clusters, n_boot=boot["n_boot"], level=boot["level"], seed=boot["seed"]
    )


def _uniform(rows: pd.DataFrame) -> pd.Series:
    """ln(runners) per race: equal chances' winner log loss (D6's pseudo-R² reference)."""
    return np.log(rows.groupby("race_id").size().astype(float))


def _fold_points(per_race: pd.Series, rows: pd.DataFrame, metric: str) -> list:
    fold_of = rows.groupby("race_id")["fold"].first().reindex(per_race.index)
    return [
        _metric(f"fold:{k}", metric, part.mean(), len(part))
        for k, part in per_race.groupby(fold_of.to_numpy())
    ]


def standalone(rows: pd.DataFrame, spec: dict) -> tuple[list, dict]:
    """p_speed against D4's calibrated market out of fold: per race ll(p_cal) - ll(p_speed),
    D7's meeting bounds (module doc)."""
    ll_speed, ll_cal = _ll(rows["p_speed"], rows), _ll(rows["p_cal"], rows)
    per_race = (ll_cal - ll_speed.reindex(ll_cal.index)).rename("gain")
    iv = _interval(per_race, rows, spec)
    uniform = _uniform(rows).reindex(per_race.index)
    r2 = {
        name: 1.0 - ll.sum() / uniform.sum()
        for name, ll in (("p_speed", ll_speed), ("p_cal", ll_cal))
    }
    n = len(per_race)
    metrics = [
        _metric("all_folds", "delta_ll_vs_calibrated_market", iv.mean, n, iv.lo, iv.hi),
        _metric(
            "all_folds", "delta_pseudo_r2_vs_calibrated_market", per_race.sum() / uniform.sum(), n
        ),
        _metric("all_folds", "mean_log_loss", ll_speed.mean(), n),
        _metric("all_folds", "baseline_mean_log_loss", ll_cal.mean(), n),
        *(_metric("all_folds", f"pseudo_r2_vs_equal_chances[{k}]", v, n) for k, v in r2.items()),
        *_fold_points(per_race, rows, "delta_ll_vs_calibrated_market"),
    ]
    if "scale" in rows.columns:
        scales = rows.groupby("fold")["scale"].first()
        races = rows.groupby("fold")["race_id"].nunique()
        metrics += [_metric(f"fold:{k}", "scale_s", s, races[k]) for k, s in scales.items()]
    row = {"races": n, "meetings": iv.n_clusters, "ΔLL": iv.mean, "lower": iv.lo, "upper": iv.hi}
    row |= {"Δ pseudo-R²": per_race.sum() / uniform.sum(), "mean log loss": ll_speed.mean()}
    row |= {"calibrated market": ll_cal.mean()} | {f"pseudo-R² {k}": v for k, v in r2.items()}
    return metrics, {"standalone": [row]}


def _ln_within(p, race: np.ndarray) -> np.ndarray:
    p = pd.Series(np.asarray(p, dtype=float))
    return np.log((p / p.groupby(race).transform("sum")).to_numpy())


def _ll_u(u: np.ndarray, rows: pd.DataFrame) -> pd.Series:
    """#69's winner log loss per race of softmax(u) within race, from the log-softmax itself (a
    winner far behind is a large finite loss, never a probability underflowed to 0)."""
    race = rows["race_id"].to_numpy(dtype=object)
    codes = pd.factorize(race)[0]
    top = pd.Series(u).groupby(codes).transform("max").to_numpy()
    log_p = u - top - np.log(np.bincount(codes, weights=np.exp(u - top))[codes])
    won = rows["won"].to_numpy(dtype=bool)
    return pd.Series(-log_p[won]).groupby(race[won]).mean()


def marginal(rows: pd.DataFrame, spec: dict) -> tuple[list, dict]:
    """p_speed as a member of #87's stage-2 meta next to ln q: alpha pooled with its
    meeting-clustered bounds, and the paired out-of-fold ΔLL against the market-only meta
    (module doc)."""
    rows = rows.reset_index(drop=True)
    race = rows["race_id"].to_numpy(dtype=object)
    frame = pd.DataFrame(
        {
            "race_id": race,
            CLUSTER: rows[CLUSTER].to_numpy(dtype=object),
            "won": rows["won"].to_numpy(dtype=bool),
            MEMBER: _ln_within(rows["p_speed"], race),
            "ln_q_final": _ln_within(rows["q"], race),
        }
    )
    pooled = fit_meta(frame, STAGE2, cluster_col=CLUSTER)
    alpha, se = float(pooled.coef[MEMBER]), float(pooled.se_cluster[MEMBER])
    beta, se_beta = float(pooled.coef["ln_q_final"]), float(pooled.se_cluster["ln_q_final"])
    n_pooled = int(pooled.n_races)
    metrics = [
        _metric(
            "all_folds", f"coef[{MEMBER}]", alpha, n_pooled, alpha - Z95 * se, alpha + Z95 * se
        ),
        _metric(
            "all_folds",
            "coef[ln_q_final]",
            beta,
            n_pooled,
            beta - Z95 * se_beta,
            beta + Z95 * se_beta,
        ),
    ]

    fold = rows["fold"].to_numpy()
    day = pd.to_datetime(rows["race_date"]).to_numpy()
    member, market = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    folds = []
    for k in sorted(pd.unique(fold))[1:]:
        test = fold == k
        end = pd.Timestamp(rows.loc[test, "train_end"].iloc[0]).to_datetime64()
        train = (fold < k) & (day <= end)
        if not train.any():
            continue
        tr, te = frame[train], frame[test]
        try:
            fit = fit_meta(tr, STAGE2)
            b = fit_conditional_logit(
                tr[["ln_q_final"]].to_numpy(), tr["race_id"].to_numpy(), tr["won"].to_numpy()
            ).coef[0]
        except (ValueError, RuntimeError) as exc:  # no finite fit on the training rows alone
            folds.append({"fold": k, "skipped": f"{type(exc).__name__}: {exc}"})
            continue
        a, b_q = float(fit.coef[MEMBER]), float(fit.coef["ln_q_final"])
        ln_q = _ln_within(np.exp(te["ln_q_final"].to_numpy()), te["race_id"].to_numpy())
        member[test] = a * te[MEMBER].to_numpy() + b_q * ln_q
        market[test] = b * te["ln_q_final"].to_numpy()
        folds.append({"fold": k, "alpha": a, "beta": b_q, "market-only beta": float(b)})
        folds[-1] |= {"train races": int(fit.n_races)}
    scored = ~np.isnan(member)
    if not scored.any():
        raise ValueError("no fold of the rows has earlier rows that fit the meta")
    kept = rows.loc[scored].reset_index(drop=True)
    per_race = _ll_u(market[scored], kept) - _ll_u(member[scored], kept)
    iv = _interval(per_race, kept, spec)
    r2 = per_race.sum() / _uniform(kept).reindex(per_race.index).sum()
    n = len(per_race)
    metrics += [
        _metric("all_folds", "delta_ll_vs_market_only", iv.mean, n, iv.lo, iv.hi),
        _metric("all_folds", "delta_pseudo_r2_vs_market_only", r2, n),
        *_fold_points(per_race, kept, "delta_ll_vs_market_only"),
        *(
            _metric(f"fold:{f['fold']}", f"coef[{MEMBER}]", f["alpha"], f["train races"])
            for f in folds
            if "alpha" in f
        ),
        _metric("all_folds", "meta_folds_skipped", sum("skipped" in f for f in folds), n),
    ]
    row = {"alpha": alpha, "se cluster": se, "lower 95%": alpha - Z95 * se}
    row |= {"upper 95%": alpha + Z95 * se, "beta": beta, "pooled races": n_pooled}
    row |= {"ΔLL vs market only": iv.mean, "lower": iv.lo, "upper": iv.hi, "scored races": n}
    row |= {"meetings": iv.n_clusters, "Δ pseudo-R²": r2}
    return metrics, {"marginal": [row], "meta folds": folds}


# --------------------------------------------------------------------------- the study


def _speed_spec(spec: dict) -> SpeedSpec:
    model = spec["speed_model"]
    return SpeedSpec(
        tuple(model["par_keys"]),
        tuple(model["variant_keys"]),
        same_day_variant=bool(model["same_day_variant"]),
        t_df=model["t_df"],
    )


def _label(value) -> str:
    return "none" if value is None else f"{value:g}"


def _joined(predicted: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The predictions with the market's outcome, price and D4 baseline, whole races only: a race
    with a priced starter that has no prediction is dropped (an unpriced runner's row is not
    joined; the scorers renormalise within race)."""
    keys = ["race_id", "horse_id"]
    left = predicted.astype({k: object for k in keys})
    right = market[[*keys, "race_date", "won", CLUSTER, "q", "p_cal"]].astype(
        {k: object for k in keys}
    )
    rows = left.merge(right, on=keys, how="inner", validate="one_to_one")
    starters = right.groupby("race_id").size()
    found = rows.groupby("race_id").size()
    whole = found.index[found.to_numpy() == starters.reindex(found.index).to_numpy()]
    out = rows.loc[rows["race_id"].isin(whole)].reset_index(drop=True)
    counts = {
        "predicted races": int(predicted["race_id"].nunique()),
        "joined races": int(out["race_id"].nunique()),
        "races with an unpredicted starter": int(len(found) - len(whole)),
        "unpriced predicted runs": int(len(predicted) - len(rows)),
    }
    return out, counts


def compute_study(
    runs: pd.DataFrame, table: pd.DataFrame, market: pd.DataFrame, *, spec: dict, scheme: FoldScheme
) -> Study:
    """Every grid point x method's standalone and marginal variants (module doc); pure."""
    n_in = {"runs": len(runs), "races": len(table), "market starters": len(market)}
    runs, table, market = _development(runs, table, market)
    study = Study(
        counts={f"{k} read": v for k, v in n_in.items()}
        | {"runs": len(runs), "races": len(table), "market starters": len(market)}
        | {"market races": int(market["race_id"].nunique())}
    )
    fold_scheme = speed_run.scheme_label(scheme)
    points = list(itertools.product(spec["grid"]["drift_var_per_day"], spec["grid"]["t_df"]))
    first = {p: _window(spec, p[1]) for p in points}
    try:
        market = market.reset_index(drop=True)
        market["p_cal"] = meta_study.baseline_oof(
            market, walk_forward(market["race_date"], scheme=scheme)
        )
        predictions = _predict_points(
            runs,
            table,
            spec=_speed_spec(spec),
            scheme=scheme,
            points=points,
            decision_min=spec["decision_time_min"],
            predict_from={
                p: None if d is None else date.fromisoformat(d) for p, d in first.items()
            },
        )
    except Exception as exc:  # D4 or the inputs failed: every variant is a failed record
        predictions = dict.fromkeys(points, exc)
    del runs, table
    for point in points:
        drift, t_df = point
        params = {"drift_var_per_day": float(drift), "t_df": None if t_df is None else float(t_df)}
        params |= {"predict_from": first[point]}
        label = {"drift": drift, "t_df": _label(t_df)}

        def joined(got=predictions[point], label=label):
            if isinstance(got, Exception):
                raise got
            rows, counts = _joined(got, market)
            study.tables.setdefault("predictions", []).append(label | got.attrs["counts"] | counts)
            study.tables.setdefault("fits", []).extend(label | f for f in got.attrs["fits"])
            return rows

        rows = meta_study._try(joined)
        for method in METHODS:
            name = f"{method}:drift={_label(drift)}:t_df={_label(t_df)}"
            components = {"study": "studies/p_speed.py", "model": "models/speed_horse.py"}
            components |= {"speed_model": "models/speed.py", "method": method}

            def scored(rows=rows, method=method):
                out = score_folds(rows.get(), method=method)
                return out.loc[out["p_cal"].notna()].reset_index(drop=True)

            outcome = meta_study._try(scored)
            for kind, compute, base in (
                ("standalone", standalone, "calibrated_market (D4, out of fold)"),
                ("marginal", marginal, "stage2 market only (u = beta ln q)"),
            ):
                variant = Variant(
                    HYPOTHESIS_ID,
                    f"{name}:{kind}",
                    fold_scheme,
                    components | {"comparison": kind, "baseline": base},
                    params | {"method": method},
                )
                study.attempt(variant, lambda o=outcome, c=compute, n=name: _tagged(o, c, spec, n))
            del outcome
        del rows
    return study


def _window(spec: dict, t_df) -> str | None:
    """The first test window (ISO date) a point with this t_df is predicted from; None: every
    fold (spec["predict_from"], keyed by "normal" and "student_t")."""
    windows = spec.get("predict_from")
    return None if not windows else windows["normal" if t_df is None else "student_t"]


def _tagged(outcome, compute, spec: dict, name: str) -> tuple[list, dict]:
    metrics, tables = compute(outcome.get(), spec)
    return metrics, {key: [{"variant": name} | r for r in rows] for key, rows in tables.items()}


def build_spec(
    *,
    scheme: FoldScheme,
    bootstrap: dict,
    grid: dict | None = None,
    predict_from: dict[str, date | None] | None = None,
) -> dict:
    """The family's pre-registered spec, plain JSON, frozen before any store row is read.
    `predict_from` ("normal", "student_t" -> the first test window predicted; None: every
    fold) limits the folds a point is predicted on; `run_study` passes PREDICT_FROM."""
    grid = GRID if grid is None else grid
    windows = None
    if predict_from is not None:
        windows = {k: None if d is None else d.isoformat() for k, d in predict_from.items()}
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypotheses": {
            "null": "p_speed adds nothing to the final Win price: alpha = 0 in the stage-2 meta "
            "u = alpha ln p_speed + beta ln q, and the out-of-fold ΔLL against the market-only "
            "meta is <= 0",
            HYPOTHESIS_ID: "#97's p_speed adds to the final price: the D7 one-sided lower bound "
            "of the ΔLL against the market-only meta is above 0 (08 §4 expects at most +0.0005 "
            "pseudo-R², plausibly 0)",
            "standalone": "p_speed alone against D4's calibrated market: a measurement, no claim",
        },
        "target": "the race's winner (Win pool) at the final Win odds",
        "speed_model": {
            "module": "countries.japan.models.speed (#96)",
            "par_keys": list(speed_run.PAR_KEYS),
            "variant_keys": list(speed_run.VARIANT_KEYS),
            "same_day_variant": True,
            "t_df": SPEED_T_DF,
            "fit": "per fold on the training races, observing the test races (#305)",
            "training_window_variant": "#97 Q3 (b): a fit day's in-sample variant once that day "
            "is fully resulted before t (inside HorseSpeed)",
        },
        "horse_model": {
            "module": "countries.japan.models.speed_horse.HorseSpeed (#97)",
            "fit": "per fold on the training races' runs (models.speed_runs.build_runs); the "
            "test races' runs observed in one batch; the global terms stay the training runs'",
            "pace": "#97 Q2 (c): the field's mean early_pos_mean3 (#129) from earlier races",
        },
        "grid": grid,
        "grid_rationale": "drift 0 (a static ability) and 0.004 s^2/day (#97's tests: SD 0.35 s "
        "over a 30-day gap); the normal error and #305's Student-t, 5 df",
        "methods": {
            "pl": "softmax(-mu / s), s = 1 / the conditional-logit coefficient on -mu",
            "henery": "P(fastest) of independent N(mu, s^2) times (p_speed's grid), s its own "
            "1-d maximum likelihood",
        },
        "scale": "fold k's s from the rows of folds < k dated on or before fold k's train_end; "
        "a fold without such rows only trains",
        "decision_time_min": DECISION_MIN,
        "decision_time": "the off (post_time_local, Asia/Tokyo): every run and result published "
        "strictly before it (results at post + 30 min, availability.yaml)",
        "fold_scheme": dataclasses.asdict(scheme),
        "predict_from": windows,
        **({"predict_from_rationale": WINDOW_RATIONALE} if windows else {}),
        "rows": "priced starters (studies.meta.store_frame) of races predicted whole; scored "
        "where D4's out-of-fold baseline exists (the market's own #70 folds, price era)",
        "standalone": "per race ll(p_cal) - ll(p_speed), #69's winner log loss (D6); D7",
        "marginal": {
            "meta": "models.meta.fit_meta, stage2, ridge 0, member ln p_speed (within race)",
            "coefficient": "alpha pooled over the scored rows; 95% two-sided interval on the "
            "meeting-clustered sandwich se",
            "delta_ll": "from the rows' second fold: the member's meta and the market-only meta "
            "(u = beta ln q) fitted on earlier folds' rows to train_end; per race ll(market "
            "only) - ll(member's meta); D7",
        },
        "bootstrap": bootstrap,
        "decisions": {
            **{k: calibration.DECISIONS[k] for k in ("D4", "D6", "D7", "D9", "D14", "D15")},
            "D10": "no price is read before 2014-06-07; results from 2010 train the speed models",
            "D16": "nothing is adopted",
            "D21": "no licensed source (JRDB, JRA-VAN) is read",
        },
        "sources": [SOURCE],
    }


# --------------------------------------------------------------------------- reading and running


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The JRA store's runs (`models.speed_runs.build_runs`), race table (`studies.speed_run.
    build_race_table`) and priced starters (`studies.meta.store_frame`), each read column- and
    date-scoped with holdout days dropped by the scan; counts in each frame's attrs."""
    races, runners = speed_runs.read_inputs(start=RESULTS_FROM, end=date.today())
    runs, run_counts = speed_runs.build_runs(races, runners, decision_min=DECISION_MIN)
    del races, runners
    races, runners = speed_run.read_inputs()
    table, table_counts = speed_run.build_race_table(
        races, runners, estimand=speed_run.ESTIMANDS[0]
    )
    del races, runners
    runners, races = meta_study.read_development()
    market, market_counts = meta_study.store_frame(runners, races)
    del races, runners
    for frame, counts in ((runs, run_counts), (table, table_counts), (market, market_counts)):
        frame.attrs["counts"] = counts
    return runs, table, market


def _record(variant: Variant, git_sha: str, builds: dict[str, str]) -> RunRecord:
    return RunRecord(
        hypothesis_id=variant.hypothesis_id,
        family_id=FAMILY_ID,
        variant=variant.name,
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=builds,
        sources=(SOURCE,),
        fold_scheme=variant.fold_scheme,
        components=variant.components,
        params=variant.params | ({"error": variant.error} if variant.error else {}),
        metrics=variant.metrics,
        status="failed" if variant.error else "scored",
    )


SECTIONS = (
    ("standalone", "Standalone: p_speed against D4's calibrated market (nats per race)", ""),
    ("marginal", "Marginal: p_speed in the stage-2 meta next to ln q", ""),
    ("meta folds", "Marginal per fold: the meta's coefficients from its training rows", ""),
    ("predictions", "Predictions per grid point", ""),
    ("fits", "HorseSpeed fit per fold", ""),
)


def render_report(study: Study, spec: dict, about: dict) -> str:
    """Aggregate tables only, never a runner or race row."""
    lines = [f"# E39: p_speed standalone and marginal at the final price ({COUNTRY}, #{TICKET})"]
    lines += ["", *(f"- {key}: {value}" for key, value in about.items())]
    lines += ["- development rows only: holdout days are never read and the holdout is not opened"]
    lines += ["- adoption: none (D16): measurement only", "", "## Data", ""]
    lines.append(_markdown_table([{"count": k, "n": v} for k, v in study.counts.items()]))
    note = (
        "ΔLL is positive when p_speed (or its meta) is better; the bounds are D7's one-sided 95%."
    )
    for key, title, _ in SECTIONS:
        rows = study.tables.get(key)
        lines += ["", f"## {title}", ""]
        lines += [note, ""] if key in ("standalone", "marginal") else []
        lines.append(_markdown_table(rows) if rows else "_Not computed: see the variants below._")
    variants = [
        {"E": v.hypothesis_id, "variant": v.name, "status": "failed" if v.error else "scored"}
        | {m.metric: m.value for m in v.metrics if m.fold == "all_folds"}
        | ({"error": v.error} if v.error else {})
        for v in study.variants
    ]
    lines += ["", "## Variants (one registry record each)", "", _markdown_table(variants), ""]
    return "\n".join(lines)


def run_study(*, out_dir: Path | None = None, scheme: FoldScheme | None = None) -> dict:
    """Freeze the spec, read the development store, run every variant, record each one, write
    the report outside the tree and the committed registry summary."""
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    scheme = load_country_scheme(COUNTRY) if scheme is None else scheme
    boot = {**calibration.bootstrap_settings(), "seed": BOOTSTRAP_SEED}
    spec = build_spec(scheme=scheme, bootstrap=boot, predict_from=PREDICT_FROM)
    git_sha = _git_sha()
    registry = default_registry(COUNTRY)
    spec_hash = registry.freeze_family(FAMILY_ID, spec)  # before any store row is read

    builds = calibration.store_builds()
    runs, table, market = load_inputs()
    counts = {f"{name}:{k}": v for name, f in (("runs", runs), ("table", table), ("market", market))
              for k, v in f.attrs.get("counts", {}).items()}  # fmt: skip
    study = compute_study(runs, table, market, spec=spec, scheme=scheme)
    del runs, table, market
    study.counts |= counts

    existing = registry.records(FAMILY_ID)
    run_ids = {}
    for variant in study.variants:
        record = _record(variant, git_sha, builds)
        if record not in existing:
            run_ids[variant.name] = registry.append(record)
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "store": ", ".join(builds.values()),
        "run ids": ", ".join(f"{k} `{v[:16]}`" for k, v in run_ids.items()) or "already recorded",
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = write_report(
        out_dir / f"p-speed-{COUNTRY}-{spec_hash[:12]}.md",
        render_report(study, spec, about),
        sources=(SOURCE,),
    )
    # last: another ticket's family id in the shared registry can make it raise (#344, #391)
    write_summary(registry, summary_dir(COUNTRY))
    return {
        "family_id": FAMILY_ID,
        "spec_hash": spec_hash,
        "report_path": str(report),
        "registry_path": str(registry.path),
        "run_ids": {k: v[:16] for k, v in run_ids.items()},
        "failed_variants": {v.name: v.error for v in study.variants if v.error},
        "headline": {
            v.name: {
                m.metric: [m.value, m.lo, m.hi, m.n] for m in v.metrics if m.fold == "all_folds"
            }
            for v in study.variants
            if not v.error
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m horsetrack.countries.japan.studies.p_speed")
    parser.add_argument("--run", action="store_true", required=True, help="run the study")
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    args = parser.parse_args(argv)
    print(json.dumps(run_study(out_dir=args.out), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
