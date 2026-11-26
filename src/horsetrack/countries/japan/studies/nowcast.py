"""The C1 nowcast's real-data run (#85): `horsetrack japan nowcast-study`.

Data (owner, 2026-11-26): the JRA-VAN time-series paths in the store,
`processed/jravan/odds_timeseries/<YYYY-MM-DD>.parquet` (0B41 WIN, PLACE, bracket; 0B42 QUINELLA;
exactly `logger.compact.SCHEMA`), read one day at a time by column and with the model's pools
only. D10: race days from 2014-06-07 (`PRICE_ERA_START`) only, since the nowcast is the price
half. D14: holdout days are dropped by `is_holdout_day`; the holdout is not opened. Grades are the
jravan races table's `race_class`. The JRA-VAN series has no Wide pool; our own logged days are
counted for Wide coverage only (see the report).

Folds: D9's walk-forward (`config/folds.yaml`) over the development race days. Each fold fits on
its training days only: the structural nowcast, the same with beta = 0 (`no_features`), and #116's
drift (`NowcastV0.fit` on the same boards' (low figure at t, final low figure) pairs with
nowcast_v0.yaml's bands and prior_n; its `max_board_age_s` limit is not applied, since it was set
for a logger capturing every ~30 s and on 5-6 minute paths it would drop most examples at T-2 and
T-1, and all four forecasters are fitted and scored on the same boards). The random walk is the
structural fit's. Each test day's cases are scored by every forecaster with the same view, n and
seed (`models.nowcast.score_cases`); with `--test-days-per-fold N`, a seeded sample of N test
days per fold is scored (training is never sampled).

Report (through `write_report`; every figure is withheld, D21): per comparison and (pool, tau),
the log-score, CRPS and KL gains with one-sided 95% bounds resampling race days (the ticket) and
meetings (D7); PIT deciles and 50% / 90% coverage per forecaster; sigma_t and beta per pool and
tau across folds; the phi_t profile against the observed V_t / V_final; the board's age at t; the
Place rule's check; Wide coverage; counts. One registry record per comparison (E19), failures
included, then `write_summary` regenerates the committed summary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import resource
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from horsetrack.core import config, paths
from horsetrack.core.eval.folds import load_country_scheme
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day
from horsetrack.core.eval.registry import (
    FoldMetric,
    RunRecord,
    default_registry,
    summary_dir,
    write_summary,
)
from horsetrack.core.eval.reports import write_report
from horsetrack.countries.japan.board.snapshot import RaceBoard, load_decision_grid
from horsetrack.countries.japan.models import nowcast as nc
from horsetrack.countries.japan.models.v0.nowcast import NowcastV0
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START

COUNTRY = "japan"
TICKET = 85
SOURCE = "jravan"
HYPOTHESIS = "E19"
FAMILY_ID = "japan-c1-nowcast-final-shares"
COLUMNS = [
    "race_id",
    "venue",
    "race_no",
    "pool",
    "combination",
    "snapshot_utc",
    "post_utc",
    "odds",
    "odds_hi",
    "investment",
]
FORECASTERS = ("structural", "no_features", "random_walk", "drift_v0")
COMPARISONS = (
    ("structural", "random_walk"),
    ("structural", "drift_v0"),
    ("no_features", "random_walk"),
    ("structural", "no_features"),
    ("drift_v0", "random_walk"),
)
N_DRAWS = 200
SEED = 85
PIT_BINS = 10
_AT = datetime(2000, 1, 1, tzinfo=timezone.utc)  # the drift's RaceBoard needs a time; unused


# ---------------------------------------------------------------- inputs


def day_files(start: date, end: date | None) -> list[tuple[str, Path]]:
    folder = paths.processed_dir(SOURCE) / "odds_timeseries"
    out = []
    for path in sorted(folder.glob("*.parquet")):
        day = path.stem
        if day >= start.isoformat() and (end is None or day <= end.isoformat()):
            out.append((day, path))
    return out


def read_day(path: Path) -> pd.DataFrame:
    """One day file, by column and with the model's pools only; Arrow's pool is released after."""
    table = pq.read_table(path, columns=COLUMNS, filters=[("pool", "in", list(nc.POOLS))])
    frame = table.to_pandas()
    del table
    pa.default_memory_pool().release_unused()
    return frame


def read_grades(start: date) -> dict[str, str | None]:
    races = pq.read_table(
        paths.processed_dir(SOURCE) / "races.parquet",
        columns=["race_id", "race_class"],
        filters=[("race_date", ">=", start)],
    ).to_pandas()
    return dict(
        zip(races["race_id"], races["race_class"].where(races["race_class"].notna()), strict=True)
    )


def data_builds(files: list[tuple[str, Path]]) -> dict[str, str]:
    digest = hashlib.sha256()
    for day, path in files:
        digest.update(f"{day}:{path.stat().st_size}\n".encode())
    races = paths.processed_dir(SOURCE) / "races.parquet"
    race_digest = hashlib.sha256(races.read_bytes()).hexdigest()[:16]
    return {
        "jravan_odds_timeseries": f"processed/{SOURCE}/odds_timeseries@{digest.hexdigest()[:16]}",
        "jravan_races": f"processed/{SOURCE}/races.parquet@{race_digest}",
    }


def wide_coverage() -> dict:
    """Our own logged days (processed/jra/odds_timeseries): Wide rows and post times."""
    folder = paths.processed_dir("jra") / "odds_timeseries"
    days = []
    for path in sorted(folder.glob("*.parquet")) if folder.is_dir() else []:
        t = pq.read_table(path, columns=["race_id", "pool", "post_utc"]).to_pandas()
        wide = t[t["pool"] == nc.WIDE]
        days.append(
            {
                "day": path.stem,
                "holdout": is_holdout_day(COUNTRY, date.fromisoformat(path.stem)),
                "wide_races": int(wide["race_id"].nunique()),
                "wide_races_with_post_time": int(
                    wide.dropna(subset=["post_utc"])["race_id"].nunique()
                ),
            }
        )
    return {"own_logged_days": days}


def build_spec(grid, scheme, bootstrap: dict, v0_config: dict) -> dict:
    """The family's design, frozen before any development row is read."""
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypotheses": {
            HYPOTHESIS: "the structural nowcast (phi_t x s_late, logistic-normal late composition "
            "with velocity and favourite-rank features) forecasts the final ln-shares of the Win "
            "and Place pools better than the random walk q_final = q_t and than #116's fitted "
            "drift, per decision horizon: log score (D8, primary) and CRPS gains with one-sided "
            "95% day-clustered bounds; PIT and 50%/90% coverage by horizon",
        },
        "null": "no gain over the random walk at any horizon (the starting point)",
        "data": {
            "source": f"processed/{SOURCE}/odds_timeseries (JRA-VAN 0B41/0B42)",
            "first_day": PRICE_ERA_START.isoformat(),
            "pools": [nc.WIN, nc.PLACE, nc.WIDE],
            "snapshot_rule": "a row counts from its snapshot_utc (owner 2026-11-26: no lag)",
            "final": "each combination's latest row (is_final_capture), V_final the latest total",
            "dropped": "cases whose combinations change between t and the close (late scratches)",
            "place_shares": "the low figures inverted by the stake-back-share rule (no per-runner "
            "money in the series); the rule's money total at t is reported, and where it is not "
            "1 Place shares are not money shares and Place results are a diagnostic (ticket: "
            "ship Win)",
        },
        "grid": {"minutes": list(grid.minutes), "primary": grid.primary},
        "fold_scheme": dataclasses.asdict(scheme),
        "variants": {
            "structural": "models.nowcast.Nowcast.fit_cases(features=True)",
            "no_features": "the same with beta = 0",
            "random_walk": "the structural fit's random_walk (softmax(ln s_t + sigma_rw eps))",
            "drift_v0": "models.v0.nowcast.NowcastV0.fit on the same boards, bands and prior_n of "
            f"nowcast_v0.yaml ({v0_config['bands']}, {v0_config['prior_n']}), no board-age limit",
        },
        "comparisons": [f"{a}_vs_{b}" for a, b in COMPARISONS],
        "scores": "per case (race, pool, tau): mean over combinations of the Gaussian log score "
        "(draws' mean and sd of ln share) and of the ensemble CRPS of the final ln-share; KL of "
        "the final shares from the mean draw; same view, n and seed for every forecaster",
        "bootstrap": {**bootstrap, "units": ["day (the ticket)", "meeting (D7)"]},
        "model_constants": {
            "knots": list(nc.KNOTS),
            "profile_penalty": nc.PROFILE_PENALTY,
            "effect_ridge": nc.EFFECT_RIDGE,
            "r_floor": nc.R_FLOOR,
            "n_cal": nc.N_CAL,
            "n_cal_draws": nc.N_CAL_DRAWS,
            "cal_seed": nc.CAL_SEED,
        },
    }


# ---------------------------------------------------------------- the drift baseline


def drift_examples(cases: nc.Cases, days: list[str]) -> pd.DataFrame:
    """#116's training examples from the boards of `days`: one per combination and tau, with
    the columns `NowcastV0.fit` reads."""
    rep = cases.row_case()
    ok = np.repeat(nc._in_days(cases.day, days), cases.k) & (cases.odds_t > 0) & (cases.odds_f > 0)
    rep = rep[ok]
    return pd.DataFrame(
        {
            "day": pd.Categorical(cases.day[rep]),
            "pool": pd.Categorical(cases.pool[rep]),
            "tau_min": cases.tau[rep].astype(np.int16),
            "odds_t": cases.odds_t[ok],
            "odds_final": cases.odds_f[ok],
        }
    )


def drift_forecaster(model: NowcastV0):
    """#116's drift as a share forecaster: its odds draws, read back into shares by the pool's
    share rule (WIN 1/O, PLACE the stake-back inverse)."""

    def forecast(view: nc.RaceView, *, pool: str, n: int, seed: int) -> nc.Draws:
        board = view.pools[pool]
        keys = [int(c) for c in board.combinations]
        lo = np.asarray(board.odds, dtype=float)
        hi = np.asarray(board.odds_hi, dtype=float)
        win = dict(zip(keys, lo.tolist(), strict=True)) if pool == nc.WIN else {}
        place = {}
        if pool == nc.PLACE:
            place = {
                h: (o, None if not np.isfinite(x) else x)
                for h, o, x in zip(keys, lo.tolist(), hi.tolist(), strict=True)
            }
        race = RaceBoard(view.race_id, None, _AT, _AT, False, win, place, {}, {}, frozenset())
        eps = np.random.default_rng(seed).standard_normal((n, len(keys)))
        shocks = {h: eps[:, j] for j, h in enumerate(keys)}
        drawn = model.draws(race, pool=pool, tau_min=int(round(view.tau_min)), shocks=shocks)
        shares, _ = nc.pool_shares(pool, drawn.odds)
        return nc.Draws(shares=shares, v_final=np.full(n, np.nan))

    return forecast


# ---------------------------------------------------------------- aggregation


def _pit_summary(pit: pd.DataFrame, cases: nc.Cases, offset: int, acc: dict) -> None:
    """Add PIT decile counts and 50% / 90% coverage counts per (forecaster, pool, tau)."""
    local = pit["case"].to_numpy() - offset
    keys = pd.DataFrame(
        {
            "forecaster": pit["forecaster"].to_numpy(),
            "pool": cases.pool[local],
            "tau": cases.tau[local],
            "u": pit["pit"].to_numpy(),
        }
    )
    for (name, pool, tau), g in keys.groupby(["forecaster", "pool", "tau"]):
        u = g["u"].to_numpy()
        slot = acc.setdefault((name, pool, float(tau)), np.zeros(PIT_BINS + 3))
        slot[:PIT_BINS] += np.histogram(u, bins=PIT_BINS, range=(0.0, 1.0))[0]
        slot[PIT_BINS] += ((u >= 0.25) & (u <= 0.75)).sum()
        slot[PIT_BINS + 1] += ((u >= 0.05) & (u <= 0.95)).sum()
        slot[PIT_BINS + 2] += len(u)


def _quantiles(x: np.ndarray, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean())} | {
        f"q{int(q * 100):02d}": float(np.quantile(x, q)) for q in qs
    }


def board_age(cases: nc.Cases) -> list[dict]:
    rows = []
    for pool in nc.POOLS:
        for tau in sorted(set(cases.tau.tolist()), reverse=True):
            m = (cases.pool == pool) & (cases.tau == tau)
            if not m.any():
                continue
            age = cases.board_min[m] - tau
            rows.append(
                {
                    "pool": pool,
                    "tau_min": tau,
                    **_quantiles(age),
                    "share_over_1min": float((age > 1).mean()),
                    "share_over_3min": float((age > 3).mean()),
                    "max": float(age.max()),
                }
            )
    return rows


def place_rule_check(cases: nc.Cases) -> list[dict]:
    rows = []
    for tau in sorted(set(cases.tau.tolist()), reverse=True):
        m = (cases.pool == nc.PLACE) & (cases.tau == tau)
        x = cases.rule_sum[m]
        if not m.any():
            continue
        rows.append(
            {
                "tau_min": tau,
                **_quantiles(x),
                "share_within_2pct": float((np.abs(x - 1) <= 0.02).mean()),
                "share_within_5pct": float((np.abs(x - 1) <= 0.05).mean()),
            }
        )
    return rows


def _table(rows: list[dict], digits: int = 4) -> str:
    if not rows:
        return "(none)"
    head = list(dict.fromkeys(key for row in rows for key in row))

    def cell(v):
        if isinstance(v, float):
            return f"{v:.{digits}g}"
        return str(v)

    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    lines += ["| " + " | ".join(cell(row.get(k, "")) for k in head) + " |" for row in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------- the run


def _progress(started: float) -> str:
    """Elapsed time and the process's peak resident memory so far."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak / 2**30 if sys.platform == "darwin" else peak / 2**20  # bytes on macOS, KiB else
    return f"{time.monotonic() - started:.0f} s, peak {peak_gb:.2f} GB"


def run_study(
    *,
    end: date | None = None,
    test_days_per_fold: int | None = None,
    n_draws: int = N_DRAWS,
    trial: bool = False,
    out_dir: Path | None = None,
    log=print,
) -> dict:
    grid = load_decision_grid()
    scheme = load_country_scheme(COUNTRY)
    scoring = config.load_country_yaml(COUNTRY, "scoring")["bootstrap"]
    bootstrap = {"n_boot": int(scoring["n_boot"]), "level": float(scoring["level"]), "seed": SEED}
    v0_config = config.load_country_yaml(COUNTRY, "nowcast_v0")
    spec = build_spec(grid, scheme, bootstrap, v0_config)
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    registry = None
    if not trial:
        registry = default_registry(COUNTRY)
        spec_hash = registry.freeze_family(FAMILY_ID, spec)  # before any development row is read
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    git_sha = _git_sha()

    files = day_files(PRICE_ERA_START, end)
    held = [d for d, _ in files if is_holdout_day(COUNTRY, date.fromisoformat(d))]
    files = [(d, p) for d, p in files if d not in set(held)]
    builds = data_builds(files)
    grades = read_grades(PRICE_ERA_START)
    started = time.monotonic()
    log(f"reading {len(files)} development days")
    case_chunks, growth_chunks, case_parts, growth_parts = [], [], [], []
    counts: dict[str, int] = defaultdict(int)
    for i, (_, path) in enumerate(files):
        day_cases, day_growth, c = nc.extract(read_day(path), grades=grades)
        case_parts.append(day_cases)
        growth_parts.append(day_growth)
        for key, value in c.items():
            counts[key] += value
        if len(case_parts) == 100 or i == len(files) - 1:
            case_chunks.append(nc.Cases.concat(case_parts))
            growth_chunks.append(nc.Growth.concat(growth_parts))
            case_parts, growth_parts = [], []
            log(f"  {i + 1}/{len(files)} days read ({_progress(started)})")
    cases = nc.Cases.concat(case_chunks)
    growth = nc.Growth.concat(growth_chunks)
    del case_chunks, growth_chunks
    dev_days = sorted(set(cases.day.tolist()))
    folds = nc.fold_days(dev_days, scheme)
    log(f"{len(cases)} cases on {len(dev_days)} days, {len(folds)} folds ({_progress(started)})")

    rng = np.random.default_rng(SEED)
    scores, pit_acc, cells, phi_rows, fold_rows = [], {}, [], [], []
    offset = 0
    try:
        for train, test, fold in folds:
            label = f"{fold.test_start.year}Q{(fold.test_start.month - 1) // 3 + 1}"
            scored_days = list(test)
            if test_days_per_fold is not None and len(test) > test_days_per_fold:
                scored_days = sorted(rng.choice(test, size=test_days_per_fold, replace=False))
            structural = nc.Nowcast.fit_cases(cases, growth, features=True, days=train)
            plain = nc.Nowcast.fit_cases(cases, growth, features=False, days=train)
            drift = NowcastV0.fit(
                drift_examples(cases, train),
                bands=v0_config["bands"],
                grid=grid,
                prior_n=float(v0_config["prior_n"]),
            )
            test_cases = cases.select_days(scored_days)
            forecasters = {
                "structural": structural.draws,
                "no_features": plain.draws,
                "random_walk": structural.random_walk,
                "drift_v0": drift_forecaster(drift),
            }
            table, pit = nc.score_cases(
                forecasters, test_cases, n=n_draws, seed=SEED, keep_pit=True
            )
            table["case"] += offset
            table["fold"] = label
            pit["case"] += offset
            _pit_summary(pit, test_cases, offset, pit_acc)
            scores.append(table)
            for cell in structural.cells.values():
                cells.append({"fold": label, **cell.__dict__, "beta": list(cell.beta)})
            for i in range(len(test_cases)):
                view, pool = test_cases.view(i), str(test_cases.pool[i])
                phi_rows.append(
                    (
                        pool,
                        test_cases.tau[i],
                        structural.phi(view, pool=pool),
                        test_cases.total_t[i] / test_cases.total_f[i],
                    )
                )
            fold_rows.append(
                {
                    "fold": label,
                    "train_days": len(train),
                    "test_days": len(test),
                    "scored_days": len(scored_days),
                    "cases": len(test_cases),
                    "train_end": fold.train_end.isoformat(),
                }
            )
            offset += len(test_cases)
            log(f"  fold {label}: {len(test_cases)} cases scored ({_progress(started)})")
    except Exception as exc:
        if registry is not None:
            for a, b in COMPARISONS:
                record = _record(
                    a,
                    b,
                    (),
                    git_sha,
                    builds,
                    spec_hash,
                    test_days_per_fold,
                    n_draws,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                registry.append(record)
            write_summary(registry, summary_dir(COUNTRY))
        raise

    scores = pd.concat(scores, ignore_index=True)
    gains, records = {}, []
    for a, b in COMPARISONS:
        pair = scores[scores["forecaster"].isin([a, b])]
        by_day = nc.gain_table(
            pair,
            a,
            b,
            n_boot=bootstrap["n_boot"],
            level=bootstrap["level"],
            seed=SEED,
            cluster="day",
        )
        by_meeting = nc.gain_table(
            pair,
            a,
            b,
            n_boot=bootstrap["n_boot"],
            level=bootstrap["level"],
            seed=SEED,
            cluster="meeting",
        )
        per_fold = []
        primary = pair[pair["tau_min"] == grid.primary]
        for label, g in primary.groupby("fold", sort=True):
            per_fold.append(
                (
                    label,
                    nc.gain_table(
                        g, a, b, n_boot=200, level=bootstrap["level"], seed=SEED, cluster="day"
                    ),
                )
            )
        gains[f"{a}_vs_{b}"] = {"day": by_day, "meeting": by_meeting, "fold": per_fold}
        records.append(
            _record(
                a,
                b,
                _metrics(by_day, by_meeting, per_fold, grid.primary),
                git_sha,
                builds,
                spec_hash,
                test_days_per_fold,
                n_draws,
            )
        )

    phi = pd.DataFrame(phi_rows, columns=["pool", "tau", "phi_hat", "phi_obs"])
    report_data = {
        "counts": dict(counts)
        | {
            "development_days": len(dev_days),
            "holdout_days_dropped": len(held),
            "cases": len(cases),
            "folds": len(folds),
        },
        "folds": fold_rows,
        "board_age_min": board_age(cases),
        "place_rule_sum": place_rule_check(cases),
        "phi": _phi_diagnostics(phi),
        "cells": cells,
        "pit": _pit_rows(pit_acc),
        "wide": wide_coverage(),
    }
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "data": ", ".join(builds.values()),
        "sampling": "every test day"
        if test_days_per_fold is None
        else f"{test_days_per_fold} seeded test days per fold (seed {SEED}); training unsampled",
        "draws per case": n_draws,
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trial (no registry records)": trial,
    }
    stem = f"nowcast-{COUNTRY}-{spec_hash[:12]}" + ("-trial" if trial else "")
    report = write_report(
        out_dir / f"{stem}.md", render_report(gains, report_data, about, grid), sources=(SOURCE,)
    )
    dump = {k: v for k, v in report_data.items()} | {
        "gains": {
            k: {
                u: (
                    v[u].to_dict("records")
                    if u != "fold"
                    else [(f, t.to_dict("records")) for f, t in v[u]]
                )
                for u in v
            }
            for k, v in gains.items()
        },
        "about": about,
    }
    write_report(
        out_dir / f"{stem}.json", json.dumps(dump, indent=1, default=str), sources=(SOURCE,)
    )
    appended = 0
    if registry is not None:
        existing = registry.records(FAMILY_ID)
        for record in records:
            if record not in existing:
                registry.append(record)
                appended += 1
        write_summary(registry, summary_dir(COUNTRY))
    return {
        "report_path": str(report),
        "spec_hash": spec_hash,
        "records_appended": appended,
        "registry_path": None if registry is None else str(registry.path),
    }


def _phi_diagnostics(phi: pd.DataFrame) -> list[dict]:
    rows = []
    for (pool, tau), g in phi.groupby(["pool", "tau"], sort=False):
        ok = np.isfinite(g["phi_obs"]) & (g["phi_obs"] > 0)
        err = np.log(g["phi_obs"][ok]) - np.log(g["phi_hat"][ok])
        rows.append(
            {
                "pool": pool,
                "tau_min": tau,
                "n": int(ok.sum()),
                "mean_phi_obs": float(g["phi_obs"][ok].mean()),
                "mean_phi_hat": float(g["phi_hat"].mean()),
                "mean_ln_obs_over_hat": float(err.mean()),
                "sd_ln_obs_over_hat": float(err.std()),
                "mean_abs_phi_error": float(np.abs(g["phi_obs"][ok] - g["phi_hat"][ok]).mean()),
            }
        )
    return sorted(rows, key=lambda r: (nc.POOLS.index(r["pool"]), -r["tau_min"]))


def _pit_rows(acc: dict) -> list[dict]:
    rows = []
    for (name, pool, tau), slot in sorted(
        acc.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[0][2])
    ):
        n = slot[PIT_BINS + 2]
        rows.append(
            {
                "forecaster": name,
                "pool": pool,
                "tau_min": tau,
                "n": int(n),
                "coverage_50": slot[PIT_BINS] / n,
                "coverage_90": slot[PIT_BINS + 1] / n,
                "deciles": [round(float(c / n), 4) for c in slot[:PIT_BINS]],
            }
        )
    return rows


def _metrics(by_day, by_meeting, per_fold, primary) -> tuple[FoldMetric, ...]:
    out = []
    for unit, table in (("day", by_day), ("meeting", by_meeting)):
        for row in table.to_dict("records"):
            for gain in ("log_score_gain", "crps_gain", "kl_gain"):
                values = (row[gain], row[f"{gain}_lo"], row[f"{gain}_hi"])
                if all(np.isfinite(values)):
                    out.append(
                        FoldMetric(
                            f"pooled:{unit}",
                            f"{gain}:{row['pool']}:T-{row['tau_min']:g}",
                            float(values[0]),
                            float(values[1]),
                            float(values[2]),
                            int(row["n_cases"]),
                        )
                    )
    for label, table in per_fold:
        for row in table.to_dict("records"):
            for gain in ("log_score_gain", "crps_gain"):
                if np.isfinite(row[gain]):
                    out.append(
                        FoldMetric(
                            label,
                            f"{gain}:{row['pool']}:T-{primary}",
                            float(row[gain]),
                            None,
                            None,
                            int(row["n_cases"]),
                        )
                    )
    return tuple(out)


def _record(
    candidate,
    baseline,
    metrics,
    git_sha,
    builds,
    spec_hash,
    days_per_fold,
    n_draws,
    *,
    status: str = "scored",
    error: str | None = None,
) -> RunRecord:
    params = {
        "candidate": candidate,
        "baseline": baseline,
        "spec_sha256": spec_hash,
        "n_draws": n_draws,
        "seed": SEED,
        "test_days_per_fold": days_per_fold,
    }
    if error is not None:
        params["error"] = error
    return RunRecord(
        hypothesis_id=HYPOTHESIS,
        family_id=FAMILY_ID,
        variant=f"{candidate}_vs_{baseline}",
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=builds,
        sources=(SOURCE,),
        fold_scheme="walk_forward:"
        + json.dumps(dataclasses.asdict(load_country_scheme(COUNTRY)), sort_keys=True),
        components={
            "model": "countries/japan/models/nowcast.py",
            "drift": "countries/japan/models/v0/nowcast.py",
        },
        params=params,
        metrics=tuple(metrics),
        status=status,
    )


# ---------------------------------------------------------------- the report


def render_report(gains: dict, data: dict, about: dict, grid) -> str:
    lines = [
        f"# The C1 final-price nowcast against the random walk and #116's drift "
        f"({COUNTRY}, #{TICKET})",
        "",
        "Built on JRA-VAN data: every figure here is withheld from the repository (D21).",
        "",
        *(f"- {key}: {value}" for key, value in about.items()),
        "",
        "Gains are per case (race, pool, tau): the mean over combinations of the score on the "
        "final ln-share, candidate minus baseline for the log score (higher better), baseline "
        "minus candidate for CRPS and KL (lower better); positive means the candidate is better. "
        "`_lo` / `_hi` are the one-sided 95% bounds (the 5% and 95% quantiles of the bootstrap "
        "mean).",
        "",
        "## Counts",
        "",
        _table([data["counts"]]),
        "",
        _table(data["folds"]),
        "",
    ]
    for name, by_unit in gains.items():
        lines += [f"## {name}", "", "### Race-day clusters (the ticket)", ""]
        lines += [_table(by_unit["day"].to_dict("records")), "", "### Meeting clusters (D7)", ""]
        cols = ["pool", "tau_min", "n_cases"] + [
            c for c in by_unit["meeting"].columns if "gain" in c
        ]
        lines += [_table(by_unit["meeting"][cols].to_dict("records")), ""]
        lines += [
            f"### By fold at the primary horizon T-{grid.primary} (day clusters, 200 replicates)",
            "",
        ]
        rows = []
        for label, table in by_unit["fold"]:
            for row in table.to_dict("records"):
                rows.append(
                    {
                        "fold": label,
                        "pool": row["pool"],
                        "n_cases": row["n_cases"],
                        **{
                            c: row[c]
                            for c in (
                                "log_score_gain",
                                "log_score_gain_lo",
                                "crps_gain",
                                "crps_gain_lo",
                            )
                        },
                    }
                )
        lines += [_table(rows), ""]
    lines += [
        "## PIT deciles and 50% / 90% coverage (runner level)",
        "",
        "PIT of each combination's final ln-share under its draws, pooled over the scored folds. "
        "Combinations of one race are dependent, so no test statistic is given.",
        "",
        _table(data["pit"]),
        "",
        "## sigma_t and beta by pool and tau (structural fit, every fold)",
        "",
        _table(_cell_summary(data["cells"])),
        "",
        "## phi_t: the pool-growth profile against the observed V_t / V_final (scored cases)",
        "",
        _table(data["phi"]),
        "",
        "## Board age at t (minutes from the board's oldest row to t)",
        "",
        _table(data["board_age_min"]),
        "",
        "## Place shares: the stake-back rule's money total at t (1 when the rule holds)",
        "",
        "The series carries no per-runner Place money, so Place shares invert the low figures. "
        "Where this total is not 1 the board's low figures are not the stake-back-share rule "
        "applied to the money, Place shares are not money shares, and the Place rows above are "
        "a diagnostic, not evidence (the ticket: ship Win).",
        "",
        _table(data["place_rule_sum"]),
        "",
        "## Wide coverage",
        "",
        "The JRA-VAN series carries no Wide pool. Our own logged days:",
        "",
        _table(data["wide"]["own_logged_days"]),
        "",
    ]
    return "\n".join(lines)


def _cell_summary(cells: list[dict]) -> list[dict]:
    frame = pd.DataFrame(cells)
    if frame.empty:
        return []
    rows = []
    for (pool, tau), g in frame.groupby(["pool", "tau_min"], sort=False):
        beta = np.array(g["beta"].tolist())
        last = g.iloc[-1]
        rows.append(
            {
                "pool": pool,
                "tau_min": tau,
                "folds": len(g),
                "sigma_last": last["sigma"],
                "sigma_min": g["sigma"].min(),
                "sigma_max": g["sigma"].max(),
                "sigma_rw_last": last["sigma_rw"],
                **{f"beta_{f}_last": float(b) for f, b in zip(nc.FEATURES, beta[-1], strict=True)},
                "share_floored_last": last["share_floored"],
                "cases_last": last["n_cases"],
            }
        )
    return sorted(rows, key=lambda r: (nc.POOLS.index(r["pool"]), -r["tau_min"]))


def nowcast_study_command(
    end: Annotated[
        datetime | None,
        typer.Option("--end", formats=["%Y-%m-%d"], help="Last race day read (default: all)."),
    ] = None,
    test_days_per_fold: Annotated[
        int | None,
        typer.Option("--test-days-per-fold", help="Score a seeded sample of N test days per fold."),
    ] = None,
    n_draws: Annotated[int, typer.Option("--n-draws", help="Draws per case.")] = N_DRAWS,
    trial: Annotated[
        bool, typer.Option("--trial", help="No registry records; the report is suffixed -trial.")
    ] = False,
) -> None:
    """Fit the C1 nowcast on D9's folds of the JRA-VAN paths and write the private report."""
    result = run_study(
        end=None if end is None else end.date(),
        test_days_per_fold=test_days_per_fold,
        n_draws=n_draws,
        trial=trial,
        log=typer.echo,
    )
    typer.echo(json.dumps(result, indent=2))
