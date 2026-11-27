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

A late scratch follows the owner's #69 rule: the case is scored with every forecaster's drawn
shares renormalised over the combinations priced at the close, and counted (`renormalised`). The
fits, the drift's included, train on the cases whose board at t is the final board.

Report (through `write_report`; every figure is withheld, D21): per comparison and (pool, tau),
the log-score, CRPS and KL gains with one-sided 95% bounds resampling race days (the ticket) and
meetings (D7); PIT deciles and 50% / 90% coverage per forecaster; sigma_t and beta per pool and
tau across folds; the phi_t profile against the observed V_t / V_final; the board's age at t; the
Place rule's check at t; the Place rule against the final votes (#410: the final PLACE figures of
`final_odds` against the H1 votes of `pool_votes` on the development days read, before the
folds); Wide coverage (development days only: a holdout day's file is never read, D14); counts.
One registry record per comparison (E19) and one for the Place rule (#410), failures included,
then `write_summary` regenerates the committed summary. Family
`japan-c1-nowcast-final-shares-v3` supersedes v2, whose Place shares inverted the low figures by a
rule that applied the takeout to the pool rather than the dividend; v2 superseded
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
import pyarrow.compute as pc
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
from horsetrack.core.pricing import stake_back_share_dividend
from horsetrack.countries.japan.board.snapshot import RaceBoard, load_decision_grid
from horsetrack.countries.japan.models import nowcast as nc
from horsetrack.countries.japan.models.v0.nowcast import NowcastV0
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START

COUNTRY = "japan"
TICKET = 85
SOURCE = "jravan"
HYPOTHESIS = "E19"
FAMILY_ID = "japan-c1-nowcast-final-shares-v3"
TRIAL_FAMILY_ID = f"{FAMILY_ID}-trial"  # trial runs' records; a trial never freezes FAMILY_ID
SUPERSEDES = "japan-c1-nowcast-final-shares-v2"
PLACE_TICKET = 410
PLACE_VARIANT = "place_rule_vs_votes"  # the Place rule measured against the final votes (#410)
FINAL_ODDS, POOL_VOTES = "final_odds.parquet", "pool_votes.parquet"
# #410's tolerance for shares read from a board's figures against its money (per runner): a
# figure truncated by up to 0.1 moves the share it implies by up to 6.8% at a figure of 2.0
# (tests/countries/japan/test_nowcast.py states the derivation).
PLACE_REL_TOL = 0.075
PLACE_ABS_TOL = 0.01
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


# ---------------------------------------------------------------- the Place rule (#410)


def _place_rows(path: Path, columns: list[str], days: pa.Array) -> pd.DataFrame:
    """The PLACE combination rows of a jravan-only table on `days` (development days only, D14),
    one batch at a time."""
    parts = []
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=1_000_000, columns=["pool", "race_date", *columns]
    ):
        keep = pc.and_(
            pc.and_(pc.equal(batch["pool"], nc.PLACE), pc.is_in(batch["race_date"], days)),
            pc.not_equal(batch["combination"], nc.POOL_ROW),
        )
        part = batch.filter(keep)
        if part.num_rows:
            parts.append(part.select(columns).to_pandas())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def _file_build(name: str) -> str:
    digest = hashlib.sha256()
    with open(paths.processed_dir(SOURCE) / name, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            digest.update(chunk)
    return f"processed/{SOURCE}/{name}@{digest.hexdigest()[:16]}"


def place_rule_builds() -> dict[str, str]:
    return {
        "jravan_final_odds": _file_build(FINAL_ODDS),
        "jravan_pool_votes": _file_build(POOL_VOTES),
    }


def jra_place_figures(share: np.ndarray, places: int) -> tuple[np.ndarray, np.ndarray]:
    """The low and high Place figures JRA prints for money shares `share`: each runner's
    `core.pricing.stake_back_share_dividend` with the P - 1 most-backed others (low) and the
    P - 1 least-backed (high), per 100 yen, over 100."""
    pool = config.load_source(COUNTRY, "jra").pools[nc.PLACE]
    unit = float(config.load_source(COUNTRY, "jra").unit_stake)
    kw = {
        "rate": 1.0 - float(pool.takeout),
        "unit": unit,
        "dividend_step": float(pool.dividend_step),
        "min_dividend": float(pool.min_dividend),
    }
    order = np.argsort(-share, kind="stable")
    lo, hi = np.empty(len(share)), np.empty(len(share))
    for i in range(len(share)):
        others = share[order[order != i]]
        lo[i] = stake_back_share_dividend(share[i], others[: places - 1].tolist(), **kw) / unit
        hi[i] = stake_back_share_dividend(share[i], others[-(places - 1) :].tolist(), **kw) / unit
    return lo, hi


def place_rule_measurement(days: list[str]) -> dict:
    """#410: the Place rule against the money on the final boards of `days` (the development days
    read): the final PLACE figures (`final_odds`) of the runners priced with both, against their
    final votes (`pool_votes`, H1). Per board: whether the rule reproduces every printed figure
    from the votes (with P from the priced runners, and from the registered slots), and how far
    the shares read from the figures (`pool_shares`, joint and low-only) are from the vote
    shares."""
    folder = paths.processed_dir(SOURCE)
    wanted = pa.array([date.fromisoformat(d) for d in days], pa.date32())
    figures = _place_rows(
        folder / FINAL_ODDS, ["race_id", "combination", "odds", "odds_hi"], wanted
    )
    votes = _place_rows(folder / POOL_VOTES, ["race_id", "combination", "votes"], wanted)
    pa.default_memory_pool().release_unused()
    slots = votes.groupby("race_id").size()
    priced = figures[(figures["odds"] > 0) & (figures["odds_hi"] > 0)]
    boards = priced.merge(votes, on=["race_id", "combination"], how="left")
    place_pool = config.load_source(COUNTRY, "jra").pools[nc.PLACE]
    counts: dict[str, int] = defaultdict(int)
    counts["races_with_place_figures"] = int(figures["race_id"].nunique())
    rows = []
    for race_id, g in boards.groupby("race_id", sort=False):
        money = g["votes"].to_numpy(dtype=float)
        k = len(money)
        places = place_pool.places_for(k)
        if not (np.isfinite(money).all() and (money > 0).all()):
            counts["boards_without_votes_on_every_priced_runner"] += 1
            continue
        if places < 2 or k < places:
            counts["boards_not_sold"] += 1
            continue
        lo, hi = g["odds"].to_numpy(dtype=float), g["odds_hi"].to_numpy(dtype=float)
        share = money / money.sum()
        alt = place_pool.places_for(int(slots.get(race_id, k)))
        row = {
            "places": places,
            "reproduced": _reproduces(share, lo, hi, places),
            "places_from_slots_differ": alt != places,
        }
        row["reproduced_with_slot_places"] = (
            row["reproduced"] if alt == places else alt >= 2 and _reproduces(share, lo, hi, alt)
        )
        for reader, hi_in in (("joint", hi), ("low_only", None)):
            got, rule_sum = nc.pool_shares(nc.PLACE, lo, odds_hi=hi_in)
            err = np.abs(np.asarray(got) - share)
            row[f"{reader}_max_rel_error"] = float((err / share).max())
            row[f"{reader}_max_abs_error"] = float(err.max())
            row[f"{reader}_rule_sum"] = rule_sum
        rows.append(row)
    counts["boards_measured"] = len(rows)
    return {"counts": dict(counts), "summary": _place_summary(pd.DataFrame(rows))}


def _reproduces(share: np.ndarray, lo: np.ndarray, hi: np.ndarray, places: int) -> bool:
    want_lo, want_hi = jra_place_figures(share, places)
    return bool(
        np.allclose(want_lo, lo, rtol=0, atol=1e-6) and np.allclose(want_hi, hi, rtol=0, atol=1e-6)
    )


def _place_summary(boards: pd.DataFrame) -> list[dict]:
    """Per reader and number of places (and over all boards): the boards, the share whose
    printed figures the rule reproduces from the votes, the share read within #410's tolerance,
    and quantiles of the worst per-runner errors and of the rule's money total."""
    if boards.empty:
        return []
    out = []
    groups = [("all", boards)] + [(f"P={p}", g) for p, g in boards.groupby("places", sort=True)]
    for reader in ("joint", "low_only"):
        for label, g in groups:
            rel = g[f"{reader}_max_rel_error"].to_numpy()
            ab = g[f"{reader}_max_abs_error"].to_numpy()
            total = g[f"{reader}_rule_sum"].to_numpy()
            out.append(
                {
                    "reader": reader,
                    "boards": label,
                    "n": len(g),
                    "share_reproduced": float(g["reproduced"].mean()),
                    "share_slot_places_differ": float(g["places_from_slots_differ"].mean()),
                    "share_reproduced_with_slot_places": float(
                        g["reproduced_with_slot_places"].mean()
                    ),
                    "share_within_tolerance": float(
                        ((rel <= PLACE_REL_TOL) & (ab <= PLACE_ABS_TOL)).mean()
                    ),
                    "max_rel_error_q50": float(np.quantile(rel, 0.5)),
                    "max_rel_error_q95": float(np.quantile(rel, 0.95)),
                    "max_rel_error_max": float(rel.max()),
                    "max_abs_error_q95": float(np.quantile(ab, 0.95)),
                    "rule_sum_q05": float(np.quantile(total, 0.05)),
                    "rule_sum_q50": float(np.quantile(total, 0.5)),
                    "rule_sum_q95": float(np.quantile(total, 0.95)),
                }
            )
    return out


def _place_metrics(measured: dict) -> tuple[FoldMetric, ...]:
    out = []
    for row in measured["summary"]:
        for key in ("share_reproduced", "share_within_tolerance", "max_rel_error_q95"):
            if np.isfinite(row[key]):
                name = f"{key}:{row['reader']}:{row['boards']}"
                out.append(FoldMetric("pooled", name, float(row[key]), None, None, int(row["n"])))
    return tuple(out)


def build_spec(grid, scheme, bootstrap: dict, v0_config: dict) -> dict:
    """The family's design, frozen before any development row is read."""
    return {
        "family_id": FAMILY_ID,
        "supersedes": f"{SUPERSEDES}: its Place shares inverted the low figures alone by a rule "
        "that applied the takeout to the pool, not to the dividend, so they were not money "
        "shares (#410); v2 superseded japan-c1-nowcast-final-shares, whose spec dropped "
        "late-scratch cases against the owner's #69 rule",
        "ticket": TICKET,
        "place_ticket": PLACE_TICKET,
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
            "place_shares": "no per-runner money in the series, so the figures are read by JRA's "
            "stake-back-share rule (core.pricing.stake_back_share_dividend: rate x (1 + D / "
            "(P x share)), truncated to 10 yen, floored at the minimum dividend): a board with a "
            "high figure on every runner by models.nowcast.place_money_fit (least squares over "
            "both figures at their truncation midpoints, floored figures left out), else the "
            "low figures alone in closed form (owner, #410); the rule's money total at t is "
            "reported; #116's drift draws are printed as JRA prints (truncated, floored) and "
            "read by the same rule",
            "place_rule_measurement": "before the folds, on the final boards of the development "
            f"days read: processed/{SOURCE}/{FINAL_ODDS} PLACE figures against processed/"
            f"{SOURCE}/{POOL_VOTES} PLACE votes (H1) of the runners priced with both figures, "
            "boards with positive votes on every priced runner; P = places_for(priced runners), "
            "and again with P from the registered slots; per board, whether the rule reproduces "
            "every printed figure from the votes, and the share read (joint and low-only) "
            f"against the vote shares within {PLACE_REL_TOL} relative and {PLACE_ABS_TOL} "
            "absolute per runner, its worst errors and the rule's money total; one registry "
            "record, failed when it cannot run",
        },
        "grid": {"minutes": list(grid.minutes), "primary": grid.primary},
        "fold_scheme": dataclasses.asdict(scheme),
        "variants": {
            "structural": "models.nowcast.Nowcast.fit_cases(features=True)",
            "no_features": "the same with beta = 0",
            "random_walk": "the structural fit's random_walk (softmax(ln s_t + sigma_rw eps))",
            "drift_v0": "models.v0.nowcast.NowcastV0.fit on the same boards, bands and prior_n of "
            f"nowcast_v0.yaml ({v0_config['bands']}, {v0_config['prior_n']}), no board-age limit",
            PLACE_VARIANT: "data.place_rule_measurement",
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
    share rule (WIN 1/O; PLACE its low and high draws printed as JRA prints them, then read by
    the Place rule as a board is)."""

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
        odds, odds_hi = drawn.odds, drawn.odds_hi
        if pool == nc.PLACE:  # read as the boards are: printed figures by the Place rule
            odds, odds_hi = nc.printed_figures(pool, odds), nc.printed_figures(pool, odds_hi)
        shares, _ = nc.pool_shares(pool, odds, odds_hi=odds_hi)
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

    files = day_files(PRICE_ERA_START, end)
    held = [d for d, _ in files if is_holdout_day(COUNTRY, date.fromisoformat(d))]
    files = [(d, p) for d, p in files if d not in set(held)]
    builds = data_builds(files)
    started = time.monotonic()
    log("measuring the Place rule on the final boards (#410)")
    try:
        place_builds = builds | place_rule_builds()
        place_measured = place_rule_measurement([d for d, _ in files])
        place_record = _place_record(
            _place_metrics(place_measured), git_sha, place_builds, spec_hash, trial=trial
        )
    except Exception as exc:  # recorded as a failed run; the folds still run
        place_measured = {"error": f"{type(exc).__name__}: {exc}"}
        place_record = _place_record(
            (),
            git_sha,
            builds,
            spec_hash,
            trial=trial,
            status="failed",
            error=place_measured["error"],
        )
    if place_record not in registry.records(family):
        registry.append(place_record)
    log(f"  Place rule measured ({_progress(started)})")
    grades = read_grades(PRICE_ERA_START)
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
            "holdout_days_dropped": len(held),
            "cases": len(cases),
            "folds": len(folds),
        },
        "folds": fold_rows,
        "board_age_min": board_age(cases),
        "place_rule_sum": place_rule_check(cases),
        "place_measurement": place_measured,
        "phi": _phi_diagnostics(phi),
        "cells": cells,
        "pit": _pit_rows(pit_acc),
        "wide": wide_coverage(),
    }
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


def _place_record(
    metrics, git_sha, builds, spec_hash, *, trial: bool, status: str = "scored", error=None
) -> RunRecord:
    params = {"spec_sha256": spec_hash, "rel_tol": PLACE_REL_TOL, "abs_tol": PLACE_ABS_TOL}
    if trial:
        params["trial"] = True
    if error is not None:
        params["error"] = error
    return RunRecord(
        hypothesis_id=HYPOTHESIS,
        family_id=TRIAL_FAMILY_ID if trial else FAMILY_ID,
        variant=PLACE_VARIANT,
        ticket=PLACE_TICKET,
        git_sha=git_sha,
        data_builds=builds,
        sources=(SOURCE,),
        fold_scheme="none: the final boards of the development days read",
        components={
            "rule": "core/pricing.py stake_back_share_dividend",
            "reader": "countries/japan/models/nowcast.py pool_shares",
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
        "## Place shares: the rule against the final votes (#410)",
        "",
        "The final PLACE figures against the final PLACE votes on the development days read. "
        "`share_reproduced`: boards on which `core.pricing.stake_back_share_dividend` gives "
        "every printed low and high figure from the votes (P from the priced runners; "
        "`_with_slot_places` with P from the registered slots, which differ on "
        "`share_slot_places_differ`). `share_within_tolerance`: boards whose shares read from "
        f"the figures are within {PLACE_REL_TOL} relative and {PLACE_ABS_TOL} absolute of the "
        "vote shares on every runner (`joint`: both figures, the board path; `low_only`: the "
        "fallback). Errors are each board's worst runner; `rule_sum` the read money's total.",
        "",
        _table(data["place_measurement"].get("summary", []))
        if "error" not in data["place_measurement"]
        else f"Not measured: {data['place_measurement']['error']}",
        "",
        _table([data["place_measurement"].get("counts", {})]),
        "",
        "## Place shares: the rule's money total at t (1 when the rule holds)",
        "",
        "The series carries no per-runner Place money, so Place shares are read from the figures "
        "at t by the rule measured above.",
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
