"""N1: the pre-registered final-board segment scan of six JRA pools on the jravan store (#430).

Per pool (WIN, PLACE, QUINELLA, QUINELLA_PLACE = Wide, FORECAST = exacta, TRIO) every
pre-registered cell (`cell_ids`) is ranked by its shrunk expected return per unit staked at the
final price, takeout included (owner's reframe, 2026-11-27: none is closed). A cell's bets are a
flat box: one unit on every priced combination of the pool's final board in the cell's races (and
odds band). The spec is `prereg_spec()`; docs/specs/pool-scan-prereg.md quotes it. It is frozen in
the registry before the run, and `run_scan` reads no store row unless the registry holds exactly
that spec for `FAMILY_ID`.

`run_scan`, in order: the family check; the output directory (never inside the repository); the
store, one year at a time from 2014-06-07, development days only (`models.v0.store.
read_store_table` drops holdout days and refuses an earlier price read; `dividends`, which has no
race_date, is filtered by the year's development race ids); #429's race sums and grouped bets per
pool (`research.pool_scan`); per pool and span the shrinkage (#78's `fit_segment_roi` for the
single-winner pools, a normal empirical-Bayes shrink of the realised return for Place and Wide,
whose dividends are not their odds) and both omega readings; one registry record per pool, reading
and span in `FAMILY_ID`, failures included, and one per cell rule in the pool's SPA family
(`spa_family`); SPA and StepM per pool beside the ranking; the private report and the
candidate-cell file through `write_report`; for the default registry, `write_summary`.

D21: every figure comes from the jravan store, so both outputs go under the data directory and
`render_public_report` names the run without a figure. Selection hygiene (#88): odds bands,
favourite strength and pool size are read from the final board, so a cell is a calibration target
at the final price, not a betting rule.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import typer
from scipy import special

from horsetrack.core import config, env, paths
from horsetrack.core.eval import snooping
from horsetrack.core.eval.holdout import _git_sha
from horsetrack.core.eval.registry import (
    FoldMetric,
    Registry,
    RunRecord,
    _spec_hash,
    default_registry,
    run_id,
    summary_dir,
    write_summary,
)
from horsetrack.core.eval.reports import LicensedOutputInRepo, is_inside, write_report
from horsetrack.core.eval.segment_roi import fit_segment_roi
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table
from horsetrack.countries.japan.research import pool_scan
from horsetrack.countries.japan.takeout import (
    BoostCalendar,
    RaceConditionsNeeded,
    race_takeout,
    shipped_boosts,
)

COUNTRY = "japan"
TICKET = 430
HYPOTHESIS_ID = "E8"
SOURCE = "jravan"
STUDY = "pool-scan"
FAMILY_ID = "japan-n1-pool-scan"
POOLS = pool_scan.POOLS
DEFAULT_SEED = 20261127
RECENT_START = date(2024, 1, 1)
SPANS = {"from-2014-06-07": PRICE_ERA_START, "from-2024-01-01": RECENT_START}
MAIN_SPAN = "from-2014-06-07"
READINGS = ("omega_full", "segment_effect")
EB_POOLS = ("PLACE", "QUINELLA_PLACE")
FOLD_SCHEME = "in-sample:development-days"
BET = "flat_box_final_board"

BAND_EDGES = {
    "WIN": [2.0, 3.0, 5.0, 10.0, 20.0, 50.0],
    "PLACE": [1.2, 1.5, 2.0, 3.0, 5.0, 10.0],
    "QUINELLA": [5.0, 10.0, 20.0, 50.0, 100.0, 300.0],
    "QUINELLA_PLACE": [2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
    "FORECAST": [10.0, 20.0, 50.0, 100.0, 300.0, 1000.0],
    "TRIO": [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0],
}
SEGMENTS = ("class", "field", "fav", "pool_size", "surface", "venue")
LEVELS = {
    "class": ["newcomer", "maiden", "conditions", "open", "graded"],
    "field": ["le4", "5-10", "11-14", "15-18"],
    "fav": ["le1.9", "2.0-2.9", "3.0-4.9", "ge5.0"],
    "pool_size": ["low", "mid", "high"],
    "surface": ["turf", "dirt"],
    "venue": ["SAP", "HAK", "FKS", "NII", "TOK", "NAK", "CKY", "KYO", "HSN", "KOK"],
}
CROSSED = ("pool_size", "fav", "band")  # each crossed with class
FIT_COLUMNS = (*SEGMENTS, *(f"class_x_{other}" for other in CROSSED))
N_DRAWS = 1000
N_BOOT = 1000
LEVEL = 0.95
SPA_BLOCK = 12
SPA_N_BOOT = 1000
STEPM_LEVEL = 0.05
ODDS_EXACT_BELOW = 10.0
ODDS_BIN_RATIO = 1.1
Z = float(special.ndtri(LEVEL))

RACE_COLUMNS = ["race_id", "race_date", "meeting_id", "venue", "race_class", "field_size"]
RACE_COLUMNS += ["surface", "status"]
BOARD_COLUMNS = ["race_id", "pool", "combination", "odds", "status_raw", "investment"]
PAID_COLUMNS = ["race_id", "pool", "combination", "dividend", "is_refund"]
RACE_LEVELS = ("class", "field", "fav", "surface", "venue")  # the pool-independent segments


class FamilyNotFrozen(ValueError):
    """The registry does not hold `prereg_spec()` frozen for FAMILY_ID: nothing was read."""


def spa_family(pool: str) -> str:
    """The registry family of `pool`'s cell rules, the members SPA and StepM test."""
    return f"{FAMILY_ID}-cells-{pool.lower()}"


def band_labels(pool: str) -> list[str]:
    """The pool's odds-band labels, lowest first (`pool_scan.odds_band` over BAND_EDGES)."""
    edges = BAND_EDGES[pool]
    return [
        pool_scan.odds_band(edges[0] / 2, edges),
        *(pool_scan.odds_band(e, edges) for e in edges),
    ]


def cell_ids(pool: str) -> list[str]:
    """The pool's pre-registered cells in spec order: 'all', each odds band, each level of each
    segment, then class crossed with pool size, favourite strength and odds band."""
    bands = band_labels(pool)
    cells = ["all", *(f"band:{b}" for b in bands)]
    cells += [f"{segment}:{level}" for segment in SEGMENTS for level in LEVELS[segment]]
    for other in CROSSED:
        levels = bands if other == "band" else LEVELS[other]
        cells += [f"class:{g}&{other}:{level}" for g in LEVELS["class"] for level in levels]
    return cells


def _parts(cell: str) -> dict[str, str]:
    return {} if cell == "all" else dict(part.split(":", 1) for part in cell.split("&"))


def _family_of(cell: str) -> str | None:
    """The segment family a cell is shrunk within ('band' and 'all' are not shrunk)."""
    parts = list(_parts(cell))
    return None if not parts or parts == ["band"] else "&".join(parts)


def _query(cell: str) -> tuple[dict[str, str], str | None]:
    """The #78 query of a cell: its segment levels (a cross adds its interaction level) and band."""
    parts = _parts(cell)
    band = parts.pop("band", None)
    if band is not None and "class" in parts:
        parts["class_x_band"] = f"{parts['class']}|{band}"
    elif len(parts) == 2:
        other = next(k for k in parts if k != "class")
        parts[f"class_x_{other}"] = f"{parts['class']}|{parts[other]}"
    return parts, band


def prereg_spec() -> dict[str, Any]:
    """The pre-registered spec of N1 (plain JSON), frozen in the registry before the run."""
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypothesis_id": HYPOTHESIS_ID,
        "study": STUDY,
        "sources": [SOURCE],
        "registration": "registered and frozen by Claude under the owner's reframe of 2026-11-27 "
        "on #430 (every cell ranked, none closed, no owner sign-off); every alternative run is "
        "a counted variant",
        "hypothesis": "E8: in some pre-registered cell of a JRA pool's final board the "
        "hierarchical omega (realised wins over takeout-adjusted implied probability; for Place "
        "and Wide the realised return over the payout rate) differs from 1 once the "
        "favourite-longshot bias is taken out as odds-band fixed effects, and some cell's "
        "expected return per unit staked at the final price reaches break-even (omega at or "
        "above 1/payout rate)",
        "null": "every cell is priced like its pool: omega 1, flat rho 1, expected return the "
        "payout rate, below break-even. The null is the starting point.",
        "pools": {
            "WIN": "Win",
            "PLACE": "Place (banded and priced at the low end of the final range)",
            "QUINELLA": "Quinella",
            "QUINELLA_PLACE": "Wide (banded and priced at the low end of the final range)",
            "FORECAST": "Exacta",
            "TRIO": "Trio",
        },
        "pools_left_out": "TIERCE (trifecta) and BRACKET_QUINELLA: a follow-up after N1 that runs "
        "anyway (owner, 2026-11-27)",
        "span": {
            "rule": "development days from max(date_from, 2014-06-07) (D10) to the run's date_to; "
            "holdout days (core.eval.holdout.is_holdout_day, D14) are never read",
            "variants": {
                "from-2014-06-07": "the full span (the ranking, the rows and the candidate file)",
                "from-2024-01-01": "2024-01-01 onward, reported apart: untouched by the earlier "
                "pattern search (problem statement 5.1)",
            },
        },
        "store": "processed/jravan: final_odds (the six pools and their __pool__ rows), dividends "
        "(filtered by the year's development race ids) and races, read one calendar year at a "
        "time through models.v0.store.read_store_table(source='jravan')",
        "exclusions": [
            "a holdout day (D14) and any day before 2014-06-07 (D10)",
            "a race whose status is not 'run'",
            "a jump race (surface 'jump'): flat races only (owner, 2026-11-27)",
            "a race whose surface is not turf or dirt, whose class is not in "
            "pool_scan.CLASS_GROUPS, whose field size is not 1-18 or whose venue is not a JRA "
            "venue",
            "a race with no priced WIN combination (no favourite strength)",
            "a race and pool whose payout may be withheld (takeout.race_takeout "
            "boost_may_be_withheld, a premium boost; as in #131)",
            "a race and pool an age-restricted boost entry matches: the jravan conditions code "
            "cannot show the age restriction, so race_takeout(..., race_class_raw=None) raises "
            "RaceConditionsNeeded and the rate is unknown",
            "a race and pool with no __pool__ total (pool not sold)",
            "a refunded combination is staked nowhere (pool_scan.combination_returns)",
        ],
        "bet": "flat box: one unit on every priced combination of the pool's final board in the "
        "cell; a combination returns its dividend / 100 when it paid, else 0",
        "metric": {
            "flat_rho": "sum(return) / sum(payout rate) over the cell's staked combinations: 1 is "
            "priced like the pool; break-even is 1 / payout rate",
            "mean_return": "sum(return) / n_bets: the return per unit staked at the final dividend",
            "payout_rate": "the mean of the race and pool payout rates "
            "(takeout.race_takeout(race_id, pool), boosts included) over the cell's bets",
            "se_mean_return": "the D7 meeting bootstrap: n_boot resamples of the span's meetings "
            "with replacement, the sd of the ratio sum(return) / sum(bets)",
        },
        "odds_bands": {pool: list(BAND_EDGES[pool]) for pool in POOLS},
        "odds_band_rule": "pool_scan.odds_band: [e_i, e_i+1), lower edge inclusive; Place and Wide "
        "band on the low end of the final range",
        "segments": {
            "class": "pool_scan.class_group of races.race_class: newcomer (DEBUT), maiden, "
            "conditions (C1-C3), open (OP, L), graded (G1-G3)",
            "field": "races.field_size: le4, 5-10, 11-14, 15-18 (pool_scan.race_cells)",
            "fav": "the race's WIN favourite: the lowest priced final WIN odds; le1.9, 2.0-2.9, "
            "3.0-4.9, ge5.0, lower edge inclusive (pool_scan.race_cells)",
            "pool_size": "the pool's __pool__ total; tertiles per pool and calendar year: edges at "
            "the 1/3 and 2/3 quantiles (numpy, linear) of that year's scored races of the pool; "
            "low below the first edge, mid to the second, high from it (pool_scan.race_cells)",
            "surface": "turf, dirt",
            "venue": "the ten JRA venues",
            "levels": {segment: list(levels) for segment, levels in LEVELS.items()},
        },
        "cells": {
            "rule": "per pool, in this order: 'all'; 'band:<label>' for each odds band; "
            "'<segment>:<level>' for each segment and level; then 'class:<group>&<other>:<level>' "
            "for other in pool_size, fav and band, every class group and level; a cell with no "
            "bets is a row with null figures",
            "crossed_with_class": list(CROSSED),
        },
        "shrinkage": {
            "WIN, QUINELLA, FORECAST, TRIO": {
                "model": "#78, core.eval.segment_roi.fit_segment_roi: ln omega = mu[odds band] + "
                "the sum of crossed random effects, one N(0, tau_c^2) per column, the module's "
                "priors (MU_PRIOR_SD 1.0, TAU_PRIOR_SCALE 0.1)",
                "columns": list(FIT_COLUMNS),
                "interactions": "class_x_<other> is the level '<class group>|<level>'; a cross "
                "cell's query names class, its other level and the interaction",
                "draws": N_DRAWS,
                "seed": "numpy SeedSequence([run seed, pool index, span index]).generate_state(1)",
                "grouped_bets": "pool_scan.grouped_bets by race, then per (segment levels, band, "
                "takeout, odds group): odds below 10 exact, from 10 on a log grid of ratio 1.1 "
                "(group k holds [10 x 1.1^k, 10 x 1.1^(k+1))); a group's odds are its harmonic "
                "mean, so its mean implied probability is exact (only losers' terms see the grid)",
            },
            "PLACE, QUINELLA_PLACE": {
                "why": "a Place or Wide dividend is not its odds (it falls in the final low-high "
                "range and depends on the co-placers), so #78's likelihood, exact for one-winner "
                "pools, would not measure the return; the shrink is on the realised return",
                "model": "normal empirical Bayes per segment family. Band baselines rho_b = "
                "sum(return) / sum(rate) of the pool's band b on the span (fixed effects, not "
                "shrunk). A cell's effect e = sum(return) / sum(rate x rho_b) over its bets, with "
                "its meeting-bootstrap se (rho_b held at its estimate); tau^2 of a family = max(0, "
                "mean over its cells with bets of (e - 1)^2 - se^2); B = tau^2 / (tau^2 + se^2) "
                "(0 when both are 0); shrunk effect 1 + B (e - 1), sd sqrt(B) se",
                "families": "class, field, fav, pool_size, surface, venue, class&pool_size, "
                "class&fav, class&band",
            },
        },
        "readings": {
            "omega_full": "the cell's omega with its band effects: #78's mean over the cell's bets "
            "of exp(mu[band] + every random effect of the bet's levels); empirical Bayes: "
            "(sum(rate x rho_b) / sum(rate)) x the shrunk effect, and rho itself for 'all' and the "
            "band cells. The ranking, the rows and the candidate file use it",
            "segment_effect": "the segment effect alone: #78's exp(sum of the cell's named random "
            "effects); empirical Bayes: the shrunk effect. Undefined for 'all' and band cells",
        },
        "bounds": "omega is the posterior mean (#78: of the draws); one-sided 95% bounds each: "
        "omega_lo the 5% and omega_hi the 95% posterior quantile (#78 draws; empirical Bayes: "
        "omega -/+ 1.6449 sd); P(omega > 1/rate) is the posterior probability that omega x "
        "payout_rate exceeds 1 (#78: the share of draws; empirical Bayes: the normal tail)",
        "ranking": "every cell of every pool, full span, omega_full: shrunk_edge = omega x "
        "payout_rate - 1 (expected return per unit at the final price, takeout included, minus 1), "
        "descending; cells without one last; ties in pool and spec order",
        "bootstrap": {
            "unit": "meeting (races.meeting_id), D7",
            "n_boot": N_BOOT,
            "level": LEVEL,
            "draws": "multinomial counts of the span's meetings, shared by the pool's cells",
            "seed": "numpy default_rng(SeedSequence([run seed, pool index, span index]))",
        },
        "spa_stepm": {
            "families": "per pool, <family_id>-cells-<pool>: one record per pre-registered cell, "
            "its flat-box rule on the full span, scored with bets and failed without; SPA and "
            "StepM test exactly the family's non-failed variants (core.eval.snooping)",
            "loss": "per meeting, the sum over the cell's bets of stake minus return",
            "benchmark": "no bet (loss 0): a rule beats it when it returns more than it stakes",
            "clusters": "meetings in time order, the stationary bootstrap of arch",
            "block_size": SPA_BLOCK,
            "n_boot": SPA_N_BOOT,
            "stepm_level": STEPM_LEVEL,
            "seed": "the run seed",
            "reported": "beside the ranking; they rank and close nothing",
        },
        "variants": {
            "family": "one record per pool, reading and span, variant '<pool>/<reading>/<span>': "
            "6 x 2 x 2; scored when the pool's shrinkage fitted on the span, failed otherwise; its "
            "metrics are every cell's omega with bounds, P(omega > 1/rate), flat rho and mean "
            "return",
            "family_size": "registry.family_size(family_id) after the run is variants_searched on "
            "every row; each SPA family's size is reported beside its SPA p",
            "failures": "a pool with a data error in any year (pool_scan refuses a malformed board "
            "or dividend row) is failed on both spans, its cell rules too; a shrinkage that does "
            "not fit fails that pool and span; the private report gives each reason",
        },
        "outputs": {
            "private_report": "$HORSETRACK_DATA_DIR/reports/pool-scan-japan-<spec hash>.md",
            "candidate_cells": "$HORSETRACK_DATA_DIR/reports/candidate-cells/pool-scan.json, every "
            "cell in rank order in the candidate-cell schema plus omega, omega_lo, omega_hi and "
            "variants_searched",
            "public": "docs/experiments/pool-scan-japan.md: the run, no figure (D21)",
        },
        "selection_hygiene": "odds band, favourite strength and pool size are read from the final "
        "board: look-ahead for selection (#88). A cell is a calibration target at the final "
        "price and a candidate for a decision-time rule, never a betting rule",
        "decisions": {
            "D7": "the meeting is the resampling unit; 95% one-sided",
            "D10": "no price before 2014-06-07",
            "D14": "development days only; LockedHoldout is never opened",
            "D21": "every figure is withheld; outputs through core.eval.reports.write_report",
        },
    }


# --------------------------------------------------------------------------- reading the store


@dataclass
class _Store:
    """The scored races and, per pool, the race-band sums and the binned grouped bets."""

    races: list[pd.DataFrame] = field(default_factory=list)
    sums: dict[str, list[pd.DataFrame]] = field(default_factory=lambda: {p: [] for p in POOLS})
    bets: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)  # (pool, span)
    pool_size: dict[str, dict[str, str]] = field(default_factory=lambda: {p: {} for p in POOLS})
    excluded: Counter = field(default_factory=Counter)
    pool_excluded: dict[str, Counter] = field(default_factory=lambda: {p: Counter() for p in POOLS})
    errors: dict[str, list[str]] = field(default_factory=lambda: {p: [] for p in POOLS})


def _race_levels(race: dict, total: float = 1.0, edges: tuple[float, float] = (0.0, 0.0)) -> dict:
    """A race's level of each segment (`pool_scan.race_cells`); pool_size from `total`, `edges`."""
    cells = pool_scan.race_cells(
        race_class=race["race_class"],
        field_size=int(race["field_size"]),
        fav_odds=float(race["fav_odds"]),
        pool_total=total,
        pool_edges=edges,
        surface=race["surface"],
        venue=race["venue"],
    )
    return dict(c.split(":", 1) for c in cells if c != "all")


def _scored_races(races: pd.DataFrame, fav: pd.Series, store: _Store) -> pd.DataFrame:
    """The year's races that every pool may score, with their race-level segment levels; the
    others are counted by reason."""
    races = races.assign(fav_odds=races["race_id"].map(fav))
    group = races["race_class"].map(pool_scan.CLASS_GROUPS)
    size = pd.to_numeric(races["field_size"], errors="coerce")
    checks = (
        ("not run", races["status"].ne("run")),
        ("jump race", races["surface"].eq("jump")),
        ("surface not turf or dirt", ~races["surface"].isin(LEVELS["surface"])),
        ("class unknown", group.isna()),
        ("field size not 1-18", ~size.between(1, 18)),
        ("venue unknown", ~races["venue"].isin(LEVELS["venue"])),
        ("no priced WIN favourite", ~(races["fav_odds"] >= 1)),
    )
    reason = pd.Series(None, index=races.index, dtype=object)
    for label, bad in checks:
        reason = reason.mask(reason.isna() & bad.to_numpy(dtype=bool), label)
    store.excluded.update(reason.dropna().tolist())
    races = races.loc[reason.isna()].reset_index(drop=True)
    levels = pd.DataFrame([_race_levels(r) for r in races.to_dict("records")], index=races.index)
    return races.assign(**{s: levels[s] for s in RACE_LEVELS}) if len(races) else races


FIT_KEYS = [*SEGMENTS, "band", "takeout", "group", "exact"]


def _regroup(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(FIT_KEYS, as_index=False, sort=False, observed=True)[
        ["count", "won", "inv"]
    ].sum()


def _binned(pool: str, bets: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    """Grouped bets by race -> by (segment levels, band, takeout, odds group) for the fit."""
    odds = bets["odds"].to_numpy(dtype=float)
    big = odds >= ODDS_EXACT_BELOW
    with np.errstate(divide="ignore"):
        k = np.floor(np.log(odds / ODDS_EXACT_BELOW) / np.log(ODDS_BIN_RATIO))
    frame = levels.loc[bets["race_id"].to_numpy()].reset_index(drop=True)
    frame = frame.assign(
        band=bets["band"].to_numpy(),
        takeout=bets["takeout"].to_numpy(dtype=float),
        group=np.where(big, k, -1).astype(int),
        exact=np.where(big, 0.0, odds),
        count=bets["count"].to_numpy(dtype=int),
        won=bets["won"].to_numpy(dtype=int),
        inv=bets["count"].to_numpy(dtype=float) / odds,
    )
    categories = {**LEVELS, "band": band_labels(pool)}
    for name in [*SEGMENTS, "band"]:
        frame[name] = pd.Categorical(frame[name], categories=categories[name])
    return _regroup(frame)


def _accumulate(store: _Store, pool: str, year: int, binned: pd.DataFrame) -> None:
    """Fold a year's binned bets into each span that holds the year (spans start on Jan 1 or
    on the first day read)."""
    for name, first in SPANS.items():
        if year >= first.year:
            held = store.bets.get((pool, name))
            store.bets[pool, name] = binned if held is None else _regroup(pd.concat([held, binned]))


def _score_pool(
    pool: str,
    races: pd.DataFrame,
    board: pa.Table,
    paid: pd.DataFrame,
    totals: dict[tuple[str, str], float],
    calendars: dict[date, BoostCalendar],
    year: int,
    store: _Store,
) -> None:
    """One year of one pool: rates, pool-size tertiles, returns, race sums and grouped bets."""
    counted = store.pool_excluded[pool]
    rates = {}
    for rid, day in zip(races["race_id"], races["race_date"], strict=True):
        try:
            rate = race_takeout(rid, pool, calendar=calendars[day])
        except RaceConditionsNeeded:
            counted["age-restricted boost (rate unknown)"] += 1
            continue
        if rate.boost_may_be_withheld:
            counted["boost may be withheld"] += 1
        elif not (totals.get((rid, pool), 0.0) > 0):
            counted["pool not sold or no pool total"] += 1
        else:
            rates[rid] = rate.payout_rate
    if not rates:
        return
    mine = races.loc[races["race_id"].isin(list(rates))].reset_index(drop=True)
    total = np.array([totals[(rid, pool)] for rid in mine["race_id"]], dtype=float)
    edges = tuple(float(e) for e in np.quantile(total, [1 / 3, 2 / 3]))
    levels = [
        _race_levels(r, t, edges) for r, t in zip(mine.to_dict("records"), total, strict=True)
    ]
    levels = pd.DataFrame(levels, index=mine["race_id"].to_numpy())[list(SEGMENTS)]
    store.pool_size[pool].update(levels["pool_size"].to_dict())

    keep = pc.and_(
        pc.equal(board["pool"], pool),
        pc.is_in(board["race_id"], value_set=pa.array(list(rates), type=pa.string())),
    )
    rows = pool_scan.combination_returns(board.filter(keep).to_pandas(), paid, pool=pool)
    rows["rate"] = rows["race_id"].map(rates)
    for rate, part in rows.groupby("rate", sort=True):
        edges_ = BAND_EDGES[pool]
        sums = pool_scan.race_returns(part, payout_rate=rate, band_edges=edges_)
        bets = pool_scan.grouped_bets(part, by=["race_id"], payout_rate=rate, band_edges=edges_)
        odds = bets.assign(odds_sum=bets["odds"] * bets["count"])
        odds = odds.groupby(["race_id", "band"], as_index=False)["odds_sum"].sum()
        sums = sums.merge(odds, on=["race_id", "band"], how="left").fillna({"odds_sum": 0.0})
        store.sums[pool].append(sums)
        if len(bets):
            _accumulate(store, pool, year, _binned(pool, bets, levels))


def _read_year(start: date, end: date, store: _Store, calendar: BoostCalendar) -> None:
    races = read_store_table(
        "races", columns=RACE_COLUMNS, start=start, end=end, touches_price=False, source=SOURCE
    ).to_pandas()
    if races.empty:
        return
    board = read_store_table(
        "final_odds",
        columns=BOARD_COLUMNS,
        start=start,
        end=end,
        touches_price=True,
        where=ds.field("pool").isin(list(POOLS)),
        source=SOURCE,
    )
    is_total = pc.equal(board["combination"], pool_scan.POOL_ROW)
    win = board.filter(pc.and_(pc.equal(board["pool"], "WIN"), pc.invert(is_total)))
    fav = win.group_by("race_id").aggregate([("odds", "min")]).to_pandas()
    races["race_date"] = pd.to_datetime(races["race_date"]).dt.date
    races = _scored_races(races, fav.set_index("race_id")["odds_min"], store)
    if races.empty:
        return
    store.races.append(races)
    totals = board.filter(is_total).select(["race_id", "pool", "investment"]).to_pandas()
    totals = totals.dropna().set_index(["race_id", "pool"])["investment"].to_dict()
    ids = pa.array(races["race_id"].tolist(), type=pa.string())
    paid = (
        ds.dataset(paths.processed_dir(SOURCE) / "dividends.parquet", format="parquet")
        .to_table(
            columns=PAID_COLUMNS,
            filter=ds.field("race_id").isin(ids) & ds.field("pool").isin(list(POOLS)),
        )
        .to_pandas()
    )
    calendars = {
        day: BoostCalendar(
            tuple(e for e in calendar.entries if e.date <= day <= (e.until or e.date))
        )
        for day in races["race_date"].unique()
    }
    for pool in POOLS:
        try:
            _score_pool(pool, races, board, paid, totals, calendars, start.year, store)
        except Exception as exc:  # a data error fails the pool's variants, not the scan
            store.errors[pool].append(f"{start.year}: {type(exc).__name__}: {exc}")


def _read_store(start: date, end: date, log: Callable[[str], None]) -> _Store:
    store, calendar = _Store(), shipped_boosts()
    for year in range(start.year, end.year + 1):
        log(f"reading {year}")
        _read_year(max(start, date(year, 1, 1)), min(end, date(year, 12, 31)), store, calendar)
        pa.default_memory_pool().release_unused()  # one year's board at a time
    return store


# --------------------------------------------------------------------------- the cells


def _draw_summary(draws: np.ndarray, rate: float) -> dict[str, float] | None:
    """Omega (the posterior mean), its one-sided bounds and P(omega x rate > 1) from draws."""
    if not np.isfinite(draws).all():
        return None
    lo, hi = (float(q) for q in np.quantile(draws, [1 - LEVEL, LEVEL]))
    omega = min(max(float(draws.mean()), lo), hi)
    return {"omega": omega, "lo": lo, "hi": hi, "p": float(np.mean(draws * rate > 1))}


def _normal_summary(omega: float, sd: float, rate: float) -> dict[str, float] | None:
    """The same from a normal posterior with mean `omega` and sd `sd`."""
    if not (np.isfinite(omega) and np.isfinite(sd)):
        return None
    omega, sd = float(omega), float(sd)
    if sd > 0:
        p = float(special.ndtr((omega - 1 / rate) / sd))
    else:
        p = float(omega * rate > 1)
    return {"omega": omega, "lo": omega - Z * sd, "hi": omega + Z * sd, "p": p}


def _ratio_sd(w: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Per column, the bootstrap sd of sum(num) / sum(den) (NaN draws, an empty den, skipped)."""
    top, bottom = w @ num, w @ den
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(bottom > 0, top / bottom, np.nan)
    out = np.full(num.shape[1], np.nan)
    for j in range(num.shape[1]):
        good = ratio[np.isfinite(ratio[:, j]), j]
        out[j] = float(good.std(ddof=1)) if len(good) > 1 else 0.0
    return out


@dataclass
class _Span:
    """One pool on one span: per cell its sums and readings, and its per-meeting losses."""

    cells: dict[str, dict]
    meetings: pd.Series  # meeting id -> first race day, in code order
    losses: np.ndarray  # meetings x cells: stake - return
    fitted: bool
    note: str | None = None
    fit_info: dict = field(default_factory=dict)


def _span_sums(pool: str, start: date, store: _Store, races: pd.DataFrame, seed_seq) -> _Span:
    cells = cell_ids(pool)
    sums = pd.concat(store.sums[pool], ignore_index=True) if store.sums[pool] else None
    if sums is not None:
        sums = sums.loc[sums["race_id"].map(races["race_date"]).to_numpy() >= start]
    note = "data error " + "; ".join(store.errors[pool]) if store.errors[pool] else None
    if note or sums is None or not len(sums):
        empty = {c: {"n_bets": 0, "n_race_days": 0, "n_meetings": 0} for c in cells}
        zeros = np.zeros((0, len(cells)))
        return _Span(empty, pd.Series(dtype=object), zeros, False, note or "no bets")
    rid = sums["race_id"].to_numpy()
    attrs = {s: races.loc[rid, s].to_numpy() for s in RACE_LEVELS}
    attrs["pool_size"] = sums["race_id"].map(store.pool_size[pool]).to_numpy()
    attrs["band"] = sums["band"].to_numpy()
    meeting, meetings = pd.factorize(races.loc[rid, "meeting_id"].to_numpy(), sort=True)
    day = races.loc[rid, "race_date"].to_numpy()
    first = pd.Series(day).groupby(meeting).min()
    day = pd.factorize(day)[0]  # race days as integer codes, for counting
    n, ret = sums["n_staked"].to_numpy(float), sums["ret_sum"].to_numpy(float)
    rate, odds = sums["rate_sum"].to_numpy(float), sums["odds_sum"].to_numpy(float)
    rho_band = sums.groupby("band")["ret_sum"].sum() / sums.groupby("band")["rate_sum"].sum()
    expected = rate * sums["band"].map(rho_band.fillna(1.0)).to_numpy(float)

    masks: dict[tuple[str, str], np.ndarray] = {}
    columns = {k: np.zeros((len(meetings), len(cells))) for k in ("n", "ret", "rate", "exp")}
    out = {}
    for j, cell in enumerate(cells):
        m = np.ones(len(sums), dtype=bool)
        for key in _parts(cell).items():
            if key not in masks:
                masks[key] = attrs[key[0]] == key[1]
            m = m & masks[key]
        staked = m & (n > 0)
        for name, values in (("n", n), ("ret", ret), ("rate", rate), ("exp", expected)):
            columns[name][:, j] = np.bincount(
                meeting[m], weights=values[m], minlength=len(meetings)
            )
        bets = int(n[m].sum())
        info = {
            "n_bets": bets,
            "n_race_days": int(len(np.unique(day[staked]))),
            "n_meetings": int(len(np.unique(meeting[staked]))),
        }
        if bets:
            returned, due = float(ret[m].sum()), float(expected[m].sum())
            info |= {
                "payout_rate": float(rate[m].sum() / bets),
                "mean_return": returned / bets,
                "flat_rho": returned / float(rate[m].sum()),
                "mean_odds": float(odds[m].sum() / bets),
                "effect": returned / due if due > 0 else float("nan"),
                "baseline": due / float(rate[m].sum()),
            }
        out[cell] = info
    rng = np.random.default_rng(seed_seq)
    w = rng.multinomial(len(meetings), np.full(len(meetings), 1 / len(meetings)), size=N_BOOT)
    w = w.astype(float)
    se = _ratio_sd(w, columns["ret"], columns["n"])
    se_effect = _ratio_sd(w, columns["ret"], columns["exp"])
    se_rho = _ratio_sd(w, columns["ret"], columns["rate"])
    for j, cell in enumerate(cells):
        if out[cell]["n_bets"]:
            out[cell] |= {
                "se_mean_return": float(se[j]),
                "se_effect": float(se_effect[j]),
                "sd_rho": float(se_rho[j]),
            }
    first.index = meetings
    return _Span(out, first, columns["n"] - columns["ret"], False)


def _eb_readings(span: _Span) -> None:
    """Place and Wide: the normal empirical-Bayes shrink of each family's cell effects."""
    tau2: dict[str, float] = {}
    for family in {_family_of(c) for c in span.cells} - {None}:
        e = [
            (c["effect"], c["se_effect"])
            for k, c in span.cells.items()
            if _family_of(k) == family
            and c["n_bets"]
            and np.isfinite(c["effect"])
            and np.isfinite(c["se_effect"])
        ]
        tau2[family] = max(0.0, float(np.mean([(x - 1) ** 2 - s**2 for x, s in e]))) if e else 0.0
    for cell, c in span.cells.items():
        if not c["n_bets"]:
            continue
        rate, family = c["payout_rate"], _family_of(cell)
        if family is None:
            c["omega_full"] = _normal_summary(c["flat_rho"], c["sd_rho"], rate)
            continue
        se = c["se_effect"] if np.isfinite(c["se_effect"]) else 0.0
        t2 = tau2[family]
        b = t2 / (t2 + se**2) if t2 + se**2 > 0 else 0.0
        effect, sd = 1 + b * (c["effect"] - 1), float(np.sqrt(b) * se)
        c["omega_full"] = _normal_summary(c["baseline"] * effect, c["baseline"] * sd, rate)
        c["segment_effect"] = _normal_summary(effect, sd, rate)
    span.fitted, span.fit_info = True, {"tau2": tau2}


def _fit_bets(frame: pd.DataFrame) -> pd.DataFrame:
    """The fit's input: segment and interaction levels (categoricals, no per-row strings), band,
    harmonic-mean odds, counts."""
    out = pd.DataFrame({name: frame[name] for name in [*SEGMENTS, "band"]})
    group = out["class"].cat
    for other in CROSSED:
        levels = out[other].cat.categories
        codes = group.codes.to_numpy(dtype=np.int64) * len(levels) + out[other].cat.codes
        names = [f"{g}|{level}" for g in group.categories for level in levels]
        out[f"class_x_{other}"] = pd.Categorical.from_codes(codes, categories=names)
    out["odds"] = frame["count"].to_numpy(dtype=float) / frame["inv"].to_numpy(dtype=float)
    out["takeout"] = frame["takeout"].to_numpy(dtype=float)
    out["count"] = frame["count"].to_numpy(dtype=int)
    out["won"] = frame["won"].to_numpy(dtype=int)
    return out[[*FIT_COLUMNS, "band", "odds", "takeout", "count", "won"]]


def _fit_readings(pool: str, span: _Span, bets: pd.DataFrame, seed: int) -> None:
    """#78's hierarchical fit of one pool and span, and both readings of every cell."""
    fit = fit_segment_roi(
        bets,
        segment_cols=list(FIT_COLUMNS),
        odds_band_col="band",
        takeout_col="takeout",
        seed=seed,
        draws=N_DRAWS,
        count_col="count",
    )
    for cell, c in span.cells.items():
        if not c["n_bets"]:
            continue
        segment, band = _query(cell)
        try:
            c["omega_full"] = _draw_summary(fit.omega_draws(segment, band=band), c["payout_rate"])
            if segment:
                c["segment_effect"] = _draw_summary(fit.effect_draws(segment), c["payout_rate"])
        except (KeyError, ValueError):
            continue
    span.fitted = True
    span.fit_info = {"tau": fit.tau, "tau_ess": fit.tau_ess, "tau_rejected": fit.tau_rejected}


def _scan_pool(pool: str, store: _Store, races: pd.DataFrame, seed: int) -> dict[str, _Span]:
    out = {}
    for s, (name, first) in enumerate(SPANS.items()):
        seq = np.random.SeedSequence([seed, POOLS.index(pool), s])
        span = _span_sums(pool, first, store, races, seq)
        out[name] = span
        if span.note:
            continue
        try:
            if pool in EB_POOLS:
                _eb_readings(span)
            else:
                fit_seed = int(seq.generate_state(1)[0])
                _fit_readings(pool, span, _fit_bets(store.bets.pop((pool, name))), fit_seed)
        except Exception as exc:  # every failure is a failed record, and the scan goes on
            span.note = f"{type(exc).__name__}: {exc}"
    return out


# --------------------------------------------------------------------------- records and tests


def _builds() -> dict[str, str]:
    """Each store file read: its path under the data directory @ a size-and-mtime fingerprint."""
    out = {}
    for name in ("final_odds", "dividends", "races"):
        path = paths.processed_dir(SOURCE) / f"{name}.parquet"
        try:
            stat = path.stat()
            stamp = hashlib.sha256(f"{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]
        except OSError:
            stamp = "unavailable"
        out[name] = f"processed/{SOURCE}/{name}.parquet@{stamp}"
    return out


def _commit() -> str:
    try:
        return _git_sha()
    except RuntimeError:
        return "unknown"


def _record(family: str, variant: str, params: dict, metrics: list, scored: bool, base: dict):
    return RunRecord(
        hypothesis_id=HYPOTHESIS_ID,
        family_id=family,
        variant=variant,
        params=params,
        metrics=tuple(metrics),
        status="scored" if scored else "failed",
        **base,
    )


def _finite(value: Any) -> bool:
    return isinstance(value, float) and bool(np.isfinite(value))


def _reading_metrics(span_name: str, cells: dict[str, dict], reading: str) -> list[FoldMetric]:
    out = []
    for cell, c in cells.items():
        r = c.get(reading)
        if r is None or not all(_finite(r[k]) for k in ("omega", "lo", "hi", "p")):
            continue
        n = c["n_bets"]
        out += [
            FoldMetric(span_name, f"omega[{cell}]", r["omega"], r["lo"], r["hi"], n),
            FoldMetric(span_name, f"p_beats_takeout[{cell}]", r["p"], None, None, n),
            FoldMetric(span_name, f"flat_rho[{cell}]", c["flat_rho"], None, None, n),
            FoldMetric(span_name, f"mean_return[{cell}]", c["mean_return"], None, None, n),
        ]
    return out


def _cell_metrics(c: dict) -> list[FoldMetric]:
    if not c["n_bets"]:
        return []
    out = [
        FoldMetric(MAIN_SPAN, "mean_return", c["mean_return"], None, None, c["n_bets"]),
        FoldMetric(MAIN_SPAN, "flat_rho", c["flat_rho"], None, None, c["n_bets"]),
        FoldMetric(MAIN_SPAN, "n_meetings", float(c["n_meetings"]), None, None, c["n_meetings"]),
    ]
    if _finite(c["se_mean_return"]):
        out.append(FoldMetric(MAIN_SPAN, "se_mean_return", c["se_mean_return"], None, None, 0))
    return out


def _append(registry: Registry, records: list[RunRecord]) -> list[str]:
    """Append, at once, the records the registry does not already hold; every record's run id."""
    held = {run_id(r) for r in registry.records() if r.family_id.startswith(FAMILY_ID)}
    ids = [run_id(r) for r in records]
    registry.append_all([r for r, i in zip(records, ids, strict=True) if i not in held])
    return ids


def _snoop(pool: str, span: _Span, registry: Registry, seed: int) -> dict[str, Any]:
    """SPA and StepM over the pool's cell rules (its SPA family's non-failed variants)."""
    names = [c for c, info in span.cells.items() if info["n_bets"]]
    if not names:
        return {"error": "no cell has bets"}
    index = pd.Index(span.meetings.index, name="meeting_id")
    j = [list(span.cells).index(c) for c in names]
    losses = pd.DataFrame(span.losses[:, j], index=index, columns=names)
    args = {
        "clusters": pd.Series(index, index=index),
        "order": pd.Series(pd.to_datetime(span.meetings.to_numpy()), index=index),
        "block_size": SPA_BLOCK,
        "n_boot": SPA_N_BOOT,
        "seed": seed,
        "registry": registry,
        "family_id": spa_family(pool),
    }
    bench = pd.Series(0.0, index=index)
    out: dict[str, Any] = {}
    try:
        res = snooping.spa(bench, losses, **args)
        out["spa"] = {
            "pvalue": res.pvalue,
            "pvalue_lower": res.pvalue_lower,
            "pvalue_upper": res.pvalue_upper,
            "n_rules": res.n_rules,
            "family_size": res.family_size,
            "n_clusters": res.n_clusters,
        }
        out["stepm"] = snooping.stepm(bench, losses, level=STEPM_LEVEL, **args).superior
    except Exception as exc:  # reported beside the ranking; the scan goes on
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# --------------------------------------------------------------------------- the result


@dataclass
class ScanResult:
    family_id: str
    spec_hash: str
    start: date
    end: date
    seed: int
    git_sha: str
    rows: list[dict]
    run_ids: list[tuple[str, str]]  # (variant, run id) of FAMILY_ID, in spec order
    cell_run_ids: dict[str, list[str]]  # SPA family -> run ids
    family_size: int
    report_path: Path
    report_name: str
    candidates_path: Path
    candidates_name: str
    failed: dict[str, str]  # FAMILY_ID variant -> why
    snooping: dict[str, dict]
    excluded: dict[str, int]
    pool_excluded: dict[str, dict[str, int]]
    fit_info: dict[str, dict]
    summary_error: str | None = None  # why write_summary could not regenerate the summary


def _row(pool: str, cell: str, c: dict, base_rate: float, size: int) -> dict[str, Any]:
    parts = _parts(cell)
    full = c.get("omega_full")
    rate = c.get("payout_rate", base_rate)
    return {
        "study": STUDY,
        "family_id": FAMILY_ID,
        "cell_id": cell,
        "pool": pool,
        "class_group": parts.get("class"),
        "odds_band": parts.get("band"),
        "payout_rate": rate,
        "n_bets": c["n_bets"],
        "n_race_days": c["n_race_days"],
        "n_meetings": c["n_meetings"],
        "mean_return": c.get("mean_return"),
        "se_mean_return": c.get("se_mean_return") if _finite(c.get("se_mean_return")) else None,
        "shrunk_edge": None if full is None else full["omega"] * rate - 1,
        "mean_odds": c.get("mean_odds"),
        "omega": None if full is None else full["omega"],
        "omega_lo": None if full is None else full["lo"],
        "omega_hi": None if full is None else full["hi"],
        "variants_searched": size,
    }


def _ranked(rows: list[dict]) -> list[dict]:
    with_edge = [r for r in rows if r["shrunk_edge"] is not None]
    return sorted(with_edge, key=lambda r: -r["shrunk_edge"]) + [
        r for r in rows if r["shrunk_edge"] is None
    ]


def _display(path: Path) -> str:
    """`path` relative to the data directory when under it, else its file name alone."""
    try:
        return str(path.resolve().relative_to(paths.data_dir()))
    except (paths.DataDirNotSet, paths.DataDirInsideRepo, ValueError):
        return path.name


def run_scan(
    *,
    date_from: date,
    date_to: date,
    out_dir: Path,
    seed: int,
    registry: Registry | None = None,
    log: Callable[[str], None] | None = None,
) -> ScanResult:
    """Run the frozen N1 scan on the jravan store's development days (module docstring).

    Raises FamilyNotFrozen (a ValueError) unless the registry holds `prereg_spec()` frozen for
    FAMILY_ID, and LicensedOutputInRepo for an `out_dir` inside the repository, both before any
    store file is opened. `registry` None is the japan registry, whose committed summary is then
    regenerated."""
    log = log or (lambda message: None)
    own = registry is None
    registry = default_registry(COUNTRY) if own else registry
    spec_hash = _spec_hash(prereg_spec())
    if registry.frozen_spec_hash(FAMILY_ID) != spec_hash:
        raise FamilyNotFrozen(f"{FAMILY_ID} is not frozen with prereg_spec() (sha256 {spec_hash})")
    out_dir = Path(out_dir)
    if is_inside(out_dir, env.repo_root()):
        raise LicensedOutputInRepo(f"{out_dir} is inside the repository: N1 reads jravan (D21)")
    start = max(date_from, PRICE_ERA_START)
    if date_to < start:
        raise ValueError(f"date_to {date_to} is before the scan's first day {start}")

    store = _read_store(start, date_to, log)
    races = pd.concat(store.races, ignore_index=True) if store.races else pd.DataFrame()
    races = races.set_index("race_id") if len(races) else pd.DataFrame(columns=["race_date"])
    pools: dict[str, dict[str, _Span]] = {}
    for pool in POOLS:
        log(f"scoring {pool}")
        pools[pool] = _scan_pool(pool, store, races, seed)

    git_sha = _commit()
    base = {
        "ticket": TICKET,
        "git_sha": git_sha,
        "data_builds": _builds(),
        "sources": (SOURCE,),
        "fold_scheme": FOLD_SCHEME,
    }
    dates = {"start": start.isoformat(), "end": date_to.isoformat(), "seed": seed}
    cell_records = {
        pool: [
            _record(
                spa_family(pool),
                cell,
                {"pool": pool, "cell": cell, "span": MAIN_SPAN, **dates},
                _cell_metrics(c),
                bool(c["n_bets"]),
                {**base, "components": {"bet": BET}},
            )
            for cell, c in pools[pool][MAIN_SPAN].cells.items()
        ]
        for pool in POOLS
    }
    variants, failed = [], {}
    for pool in POOLS:
        shrink = "empirical_bayes_normal" if pool in EB_POOLS else "segment_roi_78"
        for reading in READINGS:
            for name, span in pools[pool].items():
                variant = f"{pool}/{reading}/{name}"
                metrics = _reading_metrics(name, span.cells, reading) if span.fitted else []
                if not metrics:
                    failed[variant] = span.note or "no cell has this reading"
                params = {"pool": pool, "reading": reading, "span": name, **dates}
                components = {"bet": BET, "shrinkage": shrink}
                variants.append(
                    _record(
                        FAMILY_ID,
                        variant,
                        params,
                        metrics,
                        bool(metrics),
                        {**base, "components": components},
                    )
                )
    ids = _append(registry, [*(r for p in POOLS for r in cell_records[p]), *variants])
    cell_ids_, k = {}, 0
    for p in POOLS:
        cell_ids_[spa_family(p)], k = ids[k : k + len(cell_records[p])], k + len(cell_records[p])
    run_ids = list(zip([v.variant for v in variants], ids[k:], strict=True))
    size = registry.family_size(FAMILY_ID)
    snoop = {pool: _snoop(pool, pools[pool][MAIN_SPAN], registry, seed) for pool in POOLS}

    jra = config.load_source(COUNTRY, "jra")
    rows = _ranked(
        [
            _row(pool, cell, c, round(1 - jra.pools[pool].takeout_on(date_to), 6), size)
            for pool in POOLS
            for cell, c in pools[pool][MAIN_SPAN].cells.items()
        ]
    )
    report = out_dir / f"pool-scan-japan-{spec_hash}.md"
    candidates = out_dir / "candidate-cells" / "pool-scan.json"
    result = ScanResult(
        family_id=FAMILY_ID,
        spec_hash=spec_hash,
        start=start,
        end=date_to,
        seed=seed,
        git_sha=git_sha,
        rows=rows,
        run_ids=run_ids,
        cell_run_ids=cell_ids_,
        family_size=size,
        report_path=report,
        report_name=_display(report),
        candidates_path=candidates,
        candidates_name=_display(candidates),
        failed=failed,
        snooping=snoop,
        excluded=dict(store.excluded),
        pool_excluded={p: dict(c) for p, c in store.pool_excluded.items()},
        fit_info={
            f"{p}/{n}": s.fit_info | ({"note": s.note} if s.note else {})
            for p in POOLS
            for n, s in pools[p].items()
        },
    )
    write_report(report, render_private_report(result, pools), sources=(SOURCE,))
    text = json.dumps(rows, indent=1, allow_nan=False) + "\n"
    write_report(candidates, text, sources=(SOURCE,))
    if own:
        try:
            write_summary(registry, summary_dir(COUNTRY))
        except ValueError as exc:  # e.g. another family's id is no file name; the run stands
            result.summary_error = str(exc)
    return result


# --------------------------------------------------------------------------- reports


def _num(value: Any, digits: int = 4) -> str:
    return "-" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    return out + ["| " + " | ".join(r) + " |" for r in rows]


def _bounds(r: dict | None) -> list[str]:
    if r is None:
        return ["-", "-", "-", "-"]
    return [_num(r["omega"]), _num(r["lo"]), _num(r["hi"]), _num(r["p"], 3)]


def render_private_report(result: ScanResult, pools: dict[str, dict[str, _Span]]) -> str:
    """The full report, every figure included: for the data directory only (D21)."""
    lines = [
        f"# N1 final-board segment scan, Japan (#{TICKET}): private, every figure withheld (D21)",
        "",
        f"- family `{result.family_id}`, spec sha256 `{result.spec_hash}`, git `{result.git_sha}`",
        f"- development days from {result.start} to {result.end}; recent span from {RECENT_START}",
        f"- seed {result.seed}; variants searched (family size): {result.family_size}",
        f"- written {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- selection hygiene: bands, favourite and pool size come from the final board (#88); "
        "a cell is a calibration target at the final price, not a betting rule",
        "",
        "## Races left out",
        "",
        *_table(["reason", "races"], [[k, str(v)] for k, v in sorted(result.excluded.items())]),
        "",
        *_table(
            ["pool", "reason", "races"],
            [[p, k, str(v)] for p in POOLS for k, v in sorted(result.pool_excluded[p].items())],
        ),
        "",
        "## Ranking: every cell, full span, omega_full",
        "",
        "edge = omega x payout rate - 1, the shrunk expected return per unit at the final price "
        "minus 1; lo and hi are one-sided 95% bounds of omega; break-even is omega = 1 / rate.",
        "",
    ]
    header = ["rank", "pool", "cell", "n", "days", "meetings", "rate", "rho", "mean return"]
    header += ["se", "mean odds", "omega", "lo", "hi", "P(>1/rate)", "edge"]
    ranked = []
    for k, r in enumerate(result.rows, start=1):
        c = pools[r["pool"]][MAIN_SPAN].cells[r["cell_id"]]
        ranked.append(
            [str(k), r["pool"], r["cell_id"], str(r["n_bets"]), str(r["n_race_days"])]
            + [str(r["n_meetings"]), _num(r["payout_rate"]), _num(c.get("flat_rho"))]
            + [_num(r["mean_return"]), _num(r["se_mean_return"]), _num(r["mean_odds"], 2)]
            + _bounds(c.get("omega_full"))
            + [_num(r["shrunk_edge"])]
        )
    lines += [*_table(header, ranked), ""]
    for pool in POOLS:
        main, recent = pools[pool][MAIN_SPAN], pools[pool]["from-2024-01-01"]
        snoop = result.snooping[pool]
        lines += [f"## {pool}", ""]
        lines += [f"- shrinkage {'empirical Bayes' if pool in EB_POOLS else '#78'}"]
        for name, span in pools[pool].items():
            info = {k: v for k, v in span.fit_info.items()}
            lines.append(f"- {name}: {'fitted' if span.fitted else 'not fitted'} {info}")
            if span.note:
                lines.append(f"- {name} failed: {span.note}")
        spa = snoop.get("spa")
        lines.append(
            f"- SPA over `{spa_family(pool)}` (benchmark: no bet): "
            + (
                f"p {spa['pvalue']:.4f} (lower {spa['pvalue_lower']:.4f}, upper "
                f"{spa['pvalue_upper']:.4f}), {spa['n_rules']} rules, family size "
                f"{spa['family_size']}, {spa['n_clusters']} meetings"
                if spa
                else f"not run ({snoop.get('error')})"
            )
        )
        if "stepm" in snoop:
            lines.append(f"- StepM survivors at {STEPM_LEVEL}: {snoop['stepm'] or 'none'}")
        rank = {r["cell_id"]: k for k, r in enumerate(result.rows, 1) if r["pool"] == pool}
        header = ["rank", "cell", "n", "meetings", "rho", "mean return", "omega", "lo", "hi"]
        header += ["P(>1/rate)", "effect", "lo", "hi", "P(>1/rate)", "2024+ n", "2024+ rho"]
        header += ["2024+ omega", "lo", "hi", "2024+ effect"]
        table = []
        for cell, c in main.cells.items():
            r = recent.cells[cell]
            effect = (r.get("segment_effect") or {}).get("omega")
            table.append(
                [str(rank[cell]), cell, str(c["n_bets"]), str(c["n_meetings"])]
                + [_num(c.get("flat_rho")), _num(c.get("mean_return"))]
                + _bounds(c.get("omega_full"))
                + _bounds(c.get("segment_effect"))
                + [str(r["n_bets"]), _num(r.get("flat_rho"))]
                + _bounds(r.get("omega_full"))[:3]
                + [_num(effect)]
            )
        lines += ["", *_table(header, table), ""]
    lines += ["## Registry", ""]
    lines += [f"- `{v}`: `{i}`" + (f" (failed: {result.failed[v]})" if v in result.failed else "")
              for v, i in result.run_ids]  # fmt: skip
    for family, ids in result.cell_run_ids.items():
        lines.append(f"- `{family}`: {len(ids)} records, run ids {', '.join(ids)}")
    return "\n".join(lines) + "\n"


def _digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("ascii")).hexdigest()


def render_public_report(result: ScanResult) -> str:
    """The public report: what ran and where its figures are, with no figure (D21)."""
    withheld = "withheld (D21)"
    lines = [
        f"# Experiment report: N1, the final-board segment scan of six pools (Japan, #{TICKET})",
        "",
        f"Ticket #{TICKET} (epic #72). **Every figure is {withheld}**: the scan reads the JRA-VAN "
        "store, a licensed source, so this report says what ran and where its figures are, and "
        "gives no count, rate, span, sign, bound or rank derived from that data. Nothing is "
        "adopted, no bet follows from it, and the holdout is not opened.",
        "",
        f"- Registry family `{result.family_id}`, spec sha256 `{result.spec_hash}` "
        "(`docs/specs/pool-scan-prereg.md` quotes the spec). Claude registered and froze it under "
        "the owner's reframe of 2026-11-27 before the run read any row.",
        f"- Requested range: development days from {result.start} to {result.end} (the run's "
        "arguments, not the store's measured span); holdout days are never read and "
        "`LockedHoldout` is never opened.",
        f"- Seed {result.seed}; code at `{result.git_sha[:12]}`; command `uv run horsetrack japan "
        "pool-scan run`.",
        f"- Variants searched (the family size on every row): {result.family_size}.",
        f"- Private report: `$HORSETRACK_DATA_DIR/{result.report_name}`; candidate cells: "
        f"`$HORSETRACK_DATA_DIR/{result.candidates_name}`; both written through `write_report` "
        "with the source `jravan`.",
        "",
        "## Registry records",
        "",
        f"One record per pool, reading and span in `{result.family_id}`, in spec order:",
        "",
        *_table(["variant", "run id"], [[f"`{v}`", f"`{i}`"] for v, i in result.run_ids]),
        "",
        "The cell rules SPA and StepM test are one record per pre-registered cell in each pool's "
        "family; their run ids are in the private report, and the sha256 of each family's sorted "
        "run ids is:",
        "",
        *_table(
            ["SPA family", "run ids sha256"],
            [[f"`{f}`", f"`{_digest(ids)}`"] for f, ids in result.cell_run_ids.items()],
        ),
        "",
        "## Hypothesis",
        "",
        "E8, as frozen: in some pre-registered cell of a JRA pool's final board the hierarchical "
        "omega differs from 1 once the favourite-longshot bias is taken out as odds-band fixed "
        "effects, and some cell's expected return per unit at the final price reaches break-even "
        "(omega at or above 1 / payout rate). The null, the starting point: every cell is priced "
        "like its pool.",
        "",
        "## Variants tried",
        "",
        "Per pool (WIN, PLACE, QUINELLA, QUINELLA_PLACE = Wide, FORECAST = exacta, TRIO): two "
        "omega readings (`omega_full`, the band effects included; `segment_effect`, the segment "
        "effect alone) on two spans (from 2014-06-07; from 2024-01-01, reported apart). The "
        "shrinkage is #78's hierarchical fit for the one-winner pools and a normal empirical-Bayes "
        "shrink of the realised return for Place and Wide. Every cell of every pool is ranked by "
        f"its shrunk expected return; the ranks, the omegas and their bounds are {withheld}. "
        f"Which variants failed is in the registry summary; their figures are {withheld}.",
        "",
        "## Alternatives rejected and why",
        "",
        "- #78's fit on Place and Wide at the low end of the range: a Place or Wide dividend is "
        "not its odds, so its omega would describe the placing probability, not the return.",
        "- A kill rule (close a pool with no open cell): replaced by the owner's reframe; every "
        "cell is ranked and none is closed.",
        "- Exact odds in the fit for every price: a Trio board's long tail makes the grouped rows "
        "too many for the 16 GB limit; from odds 10 a 1.1 log grid with the harmonic-mean price "
        "keeps the mean implied probability exact.",
        "- Trifecta and bracket quinella: a follow-up that runs after N1 (owner, 2026-11-27).",
        "",
        "## What this rules out",
        "",
        f"{withheld}: the private report gives, per pool, which cells' one-sided bounds cover "
        "break-even, the SPA p and the StepM survivors. Whatever it shows, an N1 cell is a "
        "calibration target at the final price (bands, favourite and pool size are read from the "
        "final board, look-ahead for selection, #88), not a betting rule.",
        "",
        "## Follow-ups",
        "",
        "- Trifecta and bracket quinella on the same scan (owner, 2026-11-27).",
        "- The top-ranked cells as decision-time rules (Family P) and in the power table (N6).",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- the command

pool_scan_app = typer.Typer(no_args_is_help=True, help="N1, the final-board segment scan (#430).")


@pool_scan_app.command("freeze")
def freeze_command() -> None:
    """Freeze prereg_spec() in the japan registry (irreversible; again is a no-op)."""
    spec_hash = default_registry(COUNTRY).freeze_family(FAMILY_ID, prereg_spec())
    typer.echo(f"frozen family={FAMILY_ID} spec_hash={spec_hash}")


@pool_scan_app.command("run")
def run_command(
    date_to: Annotated[datetime, typer.Option("--to", formats=["%Y-%m-%d"], help="Last day.")],
    date_from: Annotated[
        datetime, typer.Option("--from", formats=["%Y-%m-%d"], help="First day (clamped to D10).")
    ] = datetime(2014, 6, 7),
    seed: Annotated[int, typer.Option("--seed", help="The run seed.")] = DEFAULT_SEED,
    out: Annotated[Path | None, typer.Option("--out", help="Report directory.")] = None,
    public: Annotated[
        Path | None, typer.Option("--public", help="Write the public report (no figure) here.")
    ] = None,
) -> None:
    """Run the frozen scan on the jravan store and write the private report (no figure printed)."""
    result = run_scan(
        date_from=date_from.date(),
        date_to=date_to.date(),
        out_dir=paths.reports_dir() if out is None else out,
        seed=seed,
        log=lambda message: print(message, file=sys.stderr, flush=True),
    )
    if public is not None:
        public.write_text(render_public_report(result), encoding="utf-8")
    summary = {
        "family_id": result.family_id,
        "spec_hash": result.spec_hash,
        "report": result.report_name,
        "candidates": result.candidates_name,
        "run_ids": [i for _, i in result.run_ids],
        "summary_error": result.summary_error,
    }
    typer.echo(json.dumps(summary, indent=2))
