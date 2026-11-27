"""The C1 nowcast's real-data run (#85): `horsetrack japan nowcast-study`.

Data (owner, 2026-11-26): the JRA-VAN time-series paths in the store,
`processed/jravan/odds_timeseries/<YYYY-MM-DD>.parquet` (0B41 WIN, PLACE, bracket; 0B42 QUINELLA;
exactly `logger.compact.SCHEMA`), read one day at a time by column and with the model's pools
only. D10: race days from 2014-06-07 (`PRICE_ERA_START`) only, since the nowcast is the price
half. Our own compacted logged days, `processed/jra/odds_timeseries/<YYYY-MM-DD>.parquet` (owner,
#85: added to the training as they accrue; #411), are read beside them, by the same columns and
pools and with `--end` applied to both: only their rows with a post time, and for a race in both
sources only the pools the JRA-VAN rows lack (JRA-VAN wins per (race, pool); in practice our logs
add Wide, D3). D14: holdout days of either source are dropped by `is_holdout_day` on the file
name, before any read; the holdout is not opened. Grades are the jravan races table's
`race_class` (None for a race it does not hold). Wide is fitted and scored where the folds'
training days hold Wide cases; the report states plainly why when they do not, and a (pool, tau)
a fold's training gives no fit for is not scored there (counted per fold, `unscored_no_fit`).

Folds: D9's walk-forward (`config/folds.yaml`) over the development race days. Each fold fits on
its training days only: the structural nowcast, the same with beta = 0 (`no_features`), and #116's
drift (`NowcastV0.fit` on the same boards' (low figure at t, final low figure) pairs with
nowcast_v0.yaml's bands and prior_n; its `max_board_age_s` limit is not applied, since it was set
for a logger capturing every ~30 s and on 5-6 minute paths it would drop most examples at T-2 and
T-1, and all four forecasters are fitted and scored on the same boards). The random walk is the
structural fit's. Each test day's cases are scored by every forecaster with the same view, n and
seed (`models.nowcast.score_cases`); with `--test-days-per-fold N`, a seeded sample of N test
days per fold is scored (training is never sampled).

A late scratch follows the owner's #69 rule: the case is scored with every forecaster's drawn
shares renormalised over the combinations priced at the close, and counted (`renormalised`). The
fits, the drift's included, train on the cases whose board at t is the final board.

Report (through `write_report`; every figure is withheld, D21): per comparison and (pool, tau),
the log-score, CRPS and KL gains with one-sided 95% bounds resampling race days (the ticket) and
meetings (D7); PIT deciles and 50% / 90% coverage per forecaster; sigma_t and beta per pool and
tau across folds; the phi_t profile against the observed V_t / V_final; the board's age at t; the
Place rule's check; Wide coverage (development days only: a holdout day's file is never read,
D14); counts. One registry record per comparison (E19), failures included, then `write_summary`
regenerates the committed summary. Family `japan-c1-nowcast-final-shares-v2` supersedes
`japan-c1-nowcast-final-shares`, whose spec dropped late-scratch cases against #69. A `--trial`
run is a run too: its records go to the family's `-trial` family with `params["trial"] = True`,
and it never freezes or alters the family's spec.
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
FAMILY_ID = "japan-c1-nowcast-final-shares-v2"
TRIAL_FAMILY_ID = f"{FAMILY_ID}-trial"  # trial runs' records; a trial never freezes FAMILY_ID
SUPERSEDES = "japan-c1-nowcast-final-shares"
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


def day_files(start: date, end: date | None, source: str = SOURCE) -> list[tuple[str, Path]]:
    """The day files of `source` from `start` to `end`, by file name; nothing is opened."""
    folder = paths.processed_dir(source) / "odds_timeseries"
    out = []
    for path in sorted(folder.glob("*.parquet")) if folder.is_dir() else []:
        day = path.stem
        if day >= start.isoformat() and (end is None or day <= end.isoformat()):
            out.append((day, path))
    return out


def read_logged_day(path: Path, jravan: pd.DataFrame | None) -> pd.DataFrame:
    """One of our own compacted logged days, as `read_day`, keeping only rows with a post time
    (a row without one cannot be placed on the tau grid) and only the (race, pool) pairs the
    same day's JRA-VAN rows `jravan` do not hold: JRA-VAN wins where both have a pool."""
    frame = read_day(path)
    frame = frame[frame["post_utc"].notna()]
    if jravan is not None and len(jravan) and len(frame):
        theirs = pd.MultiIndex.from_frame(jravan[["race_id", "pool"]].drop_duplicates())
        frame = frame[~pd.MultiIndex.from_frame(frame[["race_id", "pool"]]).isin(theirs)]
    return frame.reset_index(drop=True)


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


def _files_digest(files: list[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for day, path in files:
        digest.update(f"{day}:{path.stat().st_size}\n".encode())
    return digest.hexdigest()[:16]


def data_builds(
    files: list[tuple[str, Path]], logged: list[tuple[str, Path]] | None = None
) -> dict[str, str]:
    """The inputs' builds: the JRA-VAN day files and races table, and our own logged development
    days when any were read (by name and size; a holdout day is not among them)."""
    races = paths.processed_dir(SOURCE) / "races.parquet"
    race_digest = hashlib.sha256(races.read_bytes()).hexdigest()[:16]
    builds = {
        "jravan_odds_timeseries": f"processed/{SOURCE}/odds_timeseries@{_files_digest(files)}",
        "jravan_races": f"processed/{SOURCE}/races.parquet@{race_digest}",
    }
    if logged:
        builds["jra_odds_timeseries"] = f"processed/jra/odds_timeseries@{_files_digest(logged)}"
    return builds


def wide_coverage() -> dict:
    """Our own logged development days (processed/jra/odds_timeseries): Wide rows and post times.
    A holdout day is fully locked (D14): it is skipped by its file name, before any read."""
    folder = paths.processed_dir("jra") / "odds_timeseries"
    days = []
    for path in sorted(folder.glob("*.parquet")) if folder.is_dir() else []:
        if is_holdout_day(COUNTRY, date.fromisoformat(path.stem)):
            continue
        t = pq.read_table(path, columns=["race_id", "pool", "post_utc"]).to_pandas()
        wide = t[t["pool"] == nc.WIDE]
        days.append(
            {
                "day": path.stem,
                "wide_races": int(wide["race_id"].nunique()),
                "wide_races_with_post_time": int(
                    wide.dropna(subset=["post_utc"])["race_id"].nunique()
                ),
            }
        )
    return {"own_logged_days": days}


def wide_status(coverage: dict, cases: nc.Cases, folds, cells: list[dict]) -> dict:
    """Whether Wide (D3) was fitted, and when not, a sentence saying why: no Wide in our logs, no
    post time on the logged Wide races, Wide cases on no fold's training day, or too few of them
    for a cell. The Wide cases and days are counted either way."""
    logged = coverage["own_logged_days"]
    wide_races = sum(e["wide_races"] for e in logged)
    with_post = sum(e["wide_races_with_post_time"] for e in logged)
    wide_days = sorted(set(cases.day[cases.pool == nc.WIDE].tolist()))
    trained = sorted({d for train, _, _ in folds for d in train} & set(wide_days))
    fitted = any(c["pool"] == nc.WIDE for c in cells)
    reasons = (
        (
            not wide_races,
            f"Wide is not fitted: none of our {len(logged)} logged development days holds a Wide "
            "race (the JRA-VAN series has no Wide pool).",
        ),
        (
            not with_post,
            f"Wide is not fitted: none of the {wide_races} Wide races on our logged development "
            "days has a post time, so no Wide board can be placed on the tau grid (recompact the "
            "days: the race card now gives the post times, #411).",
        ),
        (
            not trained,
            f"Wide is not fitted: the Wide cases are on {len(wide_days)} logged day(s) "
            f"({', '.join(wide_days) or 'none'}), none of them a training day of any fold, so no "
            "fold can fit Wide yet; too few logged days hold Wide.",
        ),
        (
            True,
            f"Wide is not fitted: the folds' training days hold Wide cases on {len(trained)} "
            "day(s), too few for a pool-growth profile or any (pool, tau) cell.",
        ),
    )
    reason = None if fitted else next(text for applies, text in reasons if applies)
    return {
        "fitted": fitted,
        "reason": reason,
        "wide_cases": int((cases.pool == nc.WIDE).sum()),
        "wide_case_days": wide_days,
        "wide_training_days": trained,
    }


def build_spec(grid, scheme, bootstrap: dict, v0_config: dict) -> dict:
    """The family's design, frozen before any development row is read."""
    return {
        "family_id": FAMILY_ID,
        "supersedes": f"{SUPERSEDES}: its spec dropped late-scratch cases, against the owner's "
        "#69 rule (core/eval/scoring.py); found in #85's review",
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
            "late_scratches": "the owner's #69 rule (core/eval/scoring.py): a combination priced "
            "on the board at t and not at the close is removed; every forecaster gets the board "
            "at t, its drawn shares are restricted to the combinations priced at the close and "
            "renormalised, and the case is scored and counted (renormalised); the fits train on "
            "the cases whose board at t is the final board",
            "dropped": "cases with no board at t or no final board, and cases whose final board "
            "prices a combination the board at t does not (added_by_close), each counted; a race "
            "without a post time gives no case",
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
    fitted = nc._in_days(cases.day, days) & (cases.scratched == 0)  # as the structural fits
    ok = np.repeat(fitted, cases.k) & (cases.odds_t > 0) & (cases.odds_f > 0)
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
    share rule (WIN 1/O, PLACE and Wide the stake-back inverse). A Wide pair's shock is v0's
    (eps_a + eps_b) / sqrt 2 of per-runner shocks."""

    def forecast(view: nc.RaceView, *, pool: str, n: int, seed: int) -> nc.Draws:
        board = view.pools[pool]
        keys = [nc._combination_key(c) for c in board.combinations]
        keys = [k if pool == nc.WIDE else k[0] for k in keys]
        lo = np.asarray(board.odds, dtype=float)
        hi = np.asarray(board.odds_hi, dtype=float)
        win = dict(zip(keys, lo.tolist(), strict=True)) if pool == nc.WIN else {}
        figures = {
            k: (o, None if not np.isfinite(x) else x)
            for k, o, x in zip(keys, lo.tolist(), hi.tolist(), strict=True)
        }
        place = figures if pool == nc.PLACE else {}
        wide = figures if pool == nc.WIDE else {}
        race = RaceBoard(view.race_id, None, _AT, _AT, False, win, place, wide, {}, frozenset())
        runners = sorted({h for k in keys for h in k}) if pool == nc.WIDE else keys
        eps = np.random.default_rng(seed).standard_normal((n, len(runners)))
        shocks = {h: eps[:, j] for j, h in enumerate(runners)}
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
    # The registry's spec hash (sorted, compact JSON), so a trial's names the spec it would freeze.
    spec_hash = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    registry = default_registry(COUNTRY)
    family = TRIAL_FAMILY_ID if trial else FAMILY_ID
    if not trial:  # a trial's spec may still change, so it is not frozen
        spec_hash = registry.freeze_family(FAMILY_ID, spec)  # before any development row is read
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    git_sha = _git_sha()

    # D14: both sources' holdout days are dropped by file name, before any of their files is read.
    files = day_files(PRICE_ERA_START, end)
    logged = day_files(PRICE_ERA_START, end, source="jra")
    held = sorted({d for d, _ in files + logged if is_holdout_day(COUNTRY, date.fromisoformat(d))})
    files = [(d, p) for d, p in files if d not in set(held)]
    logged = [(d, p) for d, p in logged if d not in set(held)]
    builds = data_builds(files, logged)
    grades = read_grades(PRICE_ERA_START)
    by_day: dict[str, list[Path | None]] = defaultdict(lambda: [None, None])
    for slot, source_files in enumerate((files, logged)):
        for d, p in source_files:
            by_day[d][slot] = p
    days = sorted(by_day.items())
    started = time.monotonic()
    log(f"reading {len(days)} development days ({len(files)} JRA-VAN, {len(logged)} our logs)")
    case_chunks, growth_chunks, case_parts, growth_parts = [], [], [], []
    counts: dict[str, int] = defaultdict(int)
    for i, (_, (jravan_path, logged_path)) in enumerate(days):
        frame = None if jravan_path is None else read_day(jravan_path)
        if logged_path is not None:
            own = read_logged_day(logged_path, frame)
            if frame is None:
                frame = own
            elif len(own):
                frame = pd.concat([frame, own], ignore_index=True)
            del own
        day_cases, day_growth, c = nc.extract(frame, grades=grades)
        del frame
        case_parts.append(day_cases)
        growth_parts.append(day_growth)
        for key, value in c.items():
            counts[key] += value
        if len(case_parts) == 100 or i == len(days) - 1:
            case_chunks.append(nc.Cases.concat(case_parts))
            growth_chunks.append(nc.Growth.concat(growth_parts))
            case_parts, growth_parts = [], []
            log(f"  {i + 1}/{len(days)} days read ({_progress(started)})")
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
            # A (pool, tau) the training days give no fit for (Wide before our logs accrue) is
            # not scored: every forecaster needs its cell.
            fitted = [
                (pool, int(tau)) in structural.cells
                and (pool, int(tau)) in plain.cells
                and (pool, int(tau)) in drift.sigma0
                for pool, tau in zip(test_cases.pool, test_cases.tau, strict=True)
            ]
            unscored = len(test_cases) - sum(fitted)
            test_cases = test_cases.select(np.array(fitted, dtype=bool))
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
                    "unscored_no_fit": unscored,
                    "train_end": fold.train_end.isoformat(),
                }
            )
            offset += len(test_cases)
            log(f"  fold {label}: {len(test_cases)} cases scored ({_progress(started)})")
    except Exception as exc:
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
                trial=trial,
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
                trial=trial,
            )
        )

    phi = pd.DataFrame(phi_rows, columns=["pool", "tau", "phi_hat", "phi_obs"])
    report_data = {
        "counts": dict(counts)
        | {
            "development_days": len(dev_days),
            "jravan_day_files": len(files),
            "own_logged_day_files": len(logged),
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
    }
    coverage = wide_coverage()
    report_data["wide"] = wide_status(coverage, cases, folds, cells) | coverage
    about = {
        "family": f"`{family}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "data": ", ".join(builds.values()),
        "sampling": "every test day"
        if test_days_per_fold is None
        else f"{test_days_per_fold} seeded test days per fold (seed {SEED}); training unsampled",
        "draws per case": n_draws,
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trial (records marked params.trial, spec not frozen)": trial,
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
    existing = registry.records(family)
    for record in records:
        if record not in existing:
            registry.append(record)
            appended += 1
    write_summary(registry, summary_dir(COUNTRY))
    return {
        "report_path": str(report),
        "family_id": family,
        "spec_hash": spec_hash,
        "records_appended": appended,
        "registry_path": str(registry.path),
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
    trial: bool = False,
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
    if trial:
        params["trial"] = True
    if error is not None:
        params["error"] = error
    return RunRecord(
        hypothesis_id=HYPOTHESIS,
        family_id=TRIAL_FAMILY_ID if trial else FAMILY_ID,
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
    wide = data["wide"]
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
        "The JRA-VAN series carries no Wide pool: Wide comes from our own logged days only (D3).",
        "",
        "Wide was fitted and scored (its rows are in the tables above)."
        if wide["fitted"]
        else wide["reason"],
        "",
        f"Wide cases: {wide['wide_cases']} on {len(wide['wide_case_days'])} day(s), of which "
        f"{len(wide['wide_training_days'])} are a fold's training day.",
        "",
        "Our own logged development days:",
        "",
        _table(wide["own_logged_days"]),
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
        bool,
        typer.Option(
            "--trial",
            help=f"Records go to {TRIAL_FAMILY_ID} marked trial; the spec is not frozen; the "
            "report is suffixed -trial.",
        ),
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
