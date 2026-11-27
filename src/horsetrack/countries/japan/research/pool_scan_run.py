"""N1: the pre-registered final-board segment scan of six JRA pools on the jravan store (#430).

Per pool (WIN, PLACE, QUINELLA, QUINELLA_PLACE = Wide, FORECAST = exacta, TRIO) every
pre-registered cell (`cell_ids`) is ranked by its shrunk expected return per unit staked at the
final price, takeout included (owner's reframe, 2026-11-27: none is closed). A cell's bets are a
flat box: one unit on every priced combination of the pool's final board in the cell's races (and
odds band). The spec is `prereg_spec()`; docs/specs/pool-scan-prereg.md quotes it. It is frozen in
the registry before the run, and `run_scan` reads no store row unless the registry holds exactly
that spec for `FAMILY_ID`.

`run_scan`, in order: the family check; the output directory (never inside the repository); the
store, one year at a time from 2014-06-07, development days only. Prices are read only on the day
windows that no holdout-spanning row group of `final_odds` touches (footer statistics; each skipped
group is counted under ROW_GROUP_REASON, #502), each window through `models.v0.store.
read_store_table`, which drops holdout days and refuses an earlier price read; `races` is read on
the same windows and `dividends` filtered by the windows' development race ids. A race and pool
with no paying dividends row, or one paying a combination off the priced board, is unsettled and
left out. Then #429's race sums and grouped bets per pool (`research.pool_scan`); per pool and span
the shrinkage (#78's `fit_segment_roi` for the single-winner pools, its omega weighted by payout
rate; for Place and Wide an empirical-Bayes shrink of the realised return over band baselines that
carry their posterior, which ranks them, and #78's fit on a representative price, a variant beside
it) and both omega readings; one registry record per pool, shrinkage, reading and span in
`FAMILY_ID`, failures included, and one per cell rule in the pool's SPA family (`spa_family`); SPA
and StepM per pool over this scan's cell records, beside the ranking; the private report and the
candidate-cell file through `write_report`; for the default registry, `write_summary`. `FAMILY_ID`
is v3: v1 and v2 (`SUPERSEDED_FAMILIES`) stay in the registry, superseded, and their variants count
toward `variants_searched`, reported apart from the family size.

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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import typer
from scipy import special

from horsetrack.core import config, env, paths
from horsetrack.core.eval import holdout, snooping
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
FAMILY_ID = "japan-n1-pool-scan-v3"
# v1 and v2, each frozen and run on 2026-11-27 and superseded after review (prereg_spec
# 'supersedes'); their records stay in the registry and their variants count toward variants
# searched.
SUPERSEDED_FAMILIES = ("japan-n1-pool-scan", "japan-n1-pool-scan-v2")
SUPERSEDED_SPECS = {
    "japan-n1-pool-scan": "74da5fd2789d242fef9da1fbc9ecb7335c5ccfd5869ed8498c64e319d34bddd9",
    "japan-n1-pool-scan-v2": "ac3050eb8d4191f4bf9c2bfb0a4978e69438544129d209dffe9f789e33359fb6",
}
UNSETTLED_REASON = "unsettled: no paying dividends row"
OFF_BOARD_REASON = "unsettled: dividend not on the priced board"
ROW_GROUP_REASON = "row group spans a holdout day (#502)"
ROI78 = "roi78"  # the variant name and reading prefix of #78's fit on Place and Wide
FOLLOW_UPS = {
    501: "Trifecta and bracket quinella on the same scan (proposed)",
    502: "date-isolated row groups in the jravan store, so no holdout day is decoded (proposed)",
    462: "Family P on development folds: the top-ranked cells as decision-time rules (proposed)",
    436: "N6: the A2 holdout power table for the top-ranked cells",
}
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
CROSSED = ("band", "field", "fav", "pool_size", "surface", "venue")  # each crossed with class
FIT_COLUMNS = (*SEGMENTS, *(f"class_x_{other}" for other in CROSSED))
N_DRAWS = 1000
JEFFREYS = 0.5  # the Jeffreys prior's shape for a Poisson hit rate: Gamma(1/2, 0)
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


def spa_family(pool: str, family: str = FAMILY_ID) -> str:
    """The registry family of `pool`'s cell rules, the members SPA and StepM test."""
    return f"{family}-cells-{pool.lower()}"


def band_labels(pool: str) -> list[str]:
    """The pool's odds-band labels, lowest first (`pool_scan.odds_band` over BAND_EDGES)."""
    edges = BAND_EDGES[pool]
    return [
        pool_scan.odds_band(edges[0] / 2, edges),
        *(pool_scan.odds_band(e, edges) for e in edges),
    ]


def cell_ids(pool: str) -> list[str]:
    """The pool's pre-registered cells in spec order: 'all', each odds band, each level of each
    segment, then class crossed with each other dimension (CROSSED: odds band, field size,
    favourite strength, pool size, surface, venue)."""
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
        "registration": "registered and frozen by Claude in the japan registry before any scored "
        "read, with no owner sign-off, under the owner's reframe of 2026-11-27 on #430 "
        "('authority'); every alternative run is a counted variant",
        "authority": {
            "where": "the owner's comment 'Reframe, owner 2026-11-27' on #430, which amends the "
            "ticket body; quoted as written, its arrows as ->",
            "sign_off": "Outline 2: \"Stop with `harness-board needs-input N` for the owner's "
            'sign-off. Resume on the answer." -> removed. Claude freezes the spec in the '
            'registry. | Rules: "The cell list and metrics are pre-registered in a spec the owner '
            'approves before the run: a needs-input step" -> "Claude pre-registers the cell '
            "list and metrics and freezes them in the registry before the run. No owner "
            'sign-off."',
            "ranking_not_a_kill_rule": 'Request: "It names the cells a model should be aimed '
            'at, or closes them." -> "It ranks every cell by expected return per 100 yen at the '
            "final price (takeout included, CI beside it). None is closed, and models aim at the "
            'top-ranked cells first." | Context: "Kill rule (plan review, section 6...)" -> '
            'removed. | Definition of done: "the candidate-cell file lists every cell the kill '
            'rule leaves open, and is empty when none is open" -> "the file lists every cell, '
            'ranked."',
            "items_for_the_spec": 'Context: "Items for the spec, the owner\'s to decide" -> '
            '"Claude decides and registers them; each alternative is a counted variant."',
        },
        "supersedes": [
            {
                "family_id": SUPERSEDED_FAMILIES[0],
                "spec_sha256": SUPERSEDED_SPECS[SUPERSEDED_FAMILIES[0]],
                "why": "v1. Review of PR #496 found that it scored a sold board with no dividend "
                "row as if every ticket lost, took a cell's expected return as its mean omega "
                "times its mean payout rate (wrong when ordinary and boosted rates mix), checked "
                "SPA's columns against every cell any earlier run scored, and gave no Place or "
                "Wide reading for a cell whose odds band returned nothing; v2 fixed each",
            },
            {
                "family_id": SUPERSEDED_FAMILIES[1],
                "spec_sha256": SUPERSEDED_SPECS[SUPERSEDED_FAMILIES[1]],
                "why": "v2. The second review of PR #496 found that it scored a race whose paying "
                "dividend named a combination off the priced, non-refunded board as if every "
                "ticket lost ('exclusions'), read a Place or Wide band that paid nothing as a "
                "known zero with no uncertainty ('shrinkage', 'band_baseline'), reported variants "
                "searched as the family size ('variants'), decoded final_odds row groups that "
                "span a holdout day before filtering them ('store_reader_guarantee'), crossed "
                "class with three dimensions only ('cells') and gave Place and Wide no #78 fit "
                "('shrinkage'); v3 fixes each",
            },
        ],
        "superseded_records": "v1's and v2's records stay in the registry, superseded: nothing is "
        "deleted, and their variants count toward variants searched ('variants')",
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
        "anyway (owner, 2026-11-27), proposed ticket #501",
        "span": {
            "rule": "development days from max(date_from, 2014-06-07) (D10) to the run's date_to; "
            "holdout days (core.eval.holdout.is_holdout_day, D14) are never read",
            "variants": {
                "from-2014-06-07": "the full span (the ranking, the rows and the candidate file)",
                "from-2024-01-01": "2024-01-01 onward, reported apart: untouched by the earlier "
                "pattern search (problem statement 5.1)",
            },
        },
        "store": "processed/jravan, one calendar year at a time: final_odds (the six pools and "
        "their __pool__ rows) and races through models.v0.store.read_store_table(source="
        "'jravan'), one read per price window of the year ('store_reader_guarantee'); dividends "
        "filtered inside its scan to the development race ids those windows read",
        "store_reader_guarantee": {
            "final_odds": "no row group of final_odds whose race_date [min, max] (its footer "
            "statistics, read without decoding a page) spans a holdout day (is_holdout_day, D14) "
            "is decoded. The scan reads prices only on the price windows: the days of the run's "
            "range that no such row group's [min, max] covers, each window one read_store_table "
            "call, whose Parquet scan prunes by the same statistics every row group outside the "
            "window and drops holdout days in its filter. Each spanning row group that meets the "
            f"range is skipped and counted ('{ROW_GROUP_REASON}'); its development days, and "
            "those of any other row group inside its range, are not scored. A row group without "
            "race_date statistics stops the run before any page is read",
            "per_year_groups": "the jravan builders write one final_odds row group per year, so "
            "until #502 rebuilds the store with day-isolated row groups a year whose row group "
            "spans a holdout day is skipped whole",
            "races_and_dividends": "races (no price or result: class, field size, surface, "
            "status) is read on the same windows through read_store_table and dividends (the "
            "results) filtered in its scan to the windows' development race ids: their row "
            "groups are filtered, not skipped ('alternatives_not_run'), so one that meets a "
            "window and spans a holdout day (races by race_date; dividends by the day each race "
            "id jra:<YYYY-MM-DD>:<VENUE>:<NN> encodes; or one without statistics) is decoded and "
            "its holdout rows dropped inside the scan; the private report counts such row groups "
            "per table until #502 isolates them too",
        },
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
            "a race and pool sold on the final board with no paying dividends row: not "
            "established or a special payout (the jravan dividends builder writes no row for "
            "either, jravan/dividends.py) or not yet delivered; it is excluded and counted, never "
            "scored as if every ticket lost",
            "a race and pool whose paying (non-refund) dividends row names a combination that "
            "is not on the pool's priced, non-refunded final board (absent, printed without a "
            "price as 0000 or cancelled as ****, refunded, or not a combination of the board): "
            "unsettled, excluded and counted, never scored as if every ticket lost and never a "
            "data error of the pool",
            "the development days of a final_odds row group whose race_date range spans a "
            "holdout day, and of any other row group inside that range ('store_reader_guarantee')",
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
            "expected_return": "the cell's mean over its bets of payout rate x omega: the shrunk "
            "expected return per unit at the final price, takeout included; omega_full x "
            "payout_rate equals it exactly, the omega being payout-rate weighted ('readings')",
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
            "for other in crossed_with_class, in that order, every class group and level; a cell "
            "with no bets is a row with null figures",
            "crossed_with_class": list(CROSSED),
            "per_pool": "1 + 7 bands + 28 segment levels + 5 classes x (7 bands + 4 field + 4 fav "
            "+ 3 pool size + 2 surface + 10 venue) = 36 + 150 = 186 cells, the same list for the "
            "ranking, the reports, the candidate file and SPA/StepM",
        },
        "shrinkage": {
            "WIN, QUINELLA, FORECAST, TRIO (and PLACE, QUINELLA_PLACE as roi78)": {
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
                "decided_by": "Claude, under 'authority' (the ticket's Context lists the "
                "shrinkage for Place and Wide among the items for the spec)",
                "why": "a Place or Wide dividend is not its odds (it falls in the final low-high "
                "range and depends on the co-placers), so #78's likelihood, exact for one-winner "
                "pools, does not measure the return by itself; the ranking's shrink is on the "
                "realised return, and #78's fit runs beside it on a representative price "
                "('roi78')",
                "model": "normal empirical Bayes per segment family over odds-band baselines. "
                "A band's baseline rho_b = sum(return) / sum(rate) of the pool's band b on the "
                "span is a fixed effect, not shrunk, with a posterior ('band_baseline'). A cell's "
                "effect e = sum(return) / sum(rate x rho_b) over its bets, rho_b at its point "
                "estimate (0 for a band that paid nothing, so e is over the cell's bets in bands "
                "that paid), with its meeting-bootstrap se; tau^2 of a family = max(0, mean over "
                "its cells with a finite e of (e - 1)^2 - se^2); B = tau^2 / (tau^2 + se^2) (0 "
                "when both are 0); shrunk effect normal, mean 1 + B (e - 1), sd sqrt(B) se",
                "band_baseline": "each band's rho_b carries its posterior, N_DRAWS draws per band "
                "shared by the pool's cells on the span: for a band that paid, a gamma with rho_b "
                "as mean and the band cell's meeting-bootstrap sd of rho as sd (a point at rho_b "
                "when that sd is 0); for a band that paid nothing, where the bootstrap has no "
                "spread, the Jeffreys posterior of a zero-hit Poisson hit rate per combination, "
                "Gamma(1/2, rate n_b), times the band's mean odds over its mean payout rate (a "
                "paid Place or Wide combination pays at least about its low-end odds): "
                "Gamma(shape 1/2, scale sum(odds) / (n_b x sum(rate))), so a sparse band that "
                "paid nothing keeps a positive upper bound",
                "zero_return_band": "a cell whose bets all sit in bands that paid nothing has no "
                "expected return under the point baselines (e is 0 / 0): it holds no information "
                "on its effect, which is the family's prior, N(1, tau^2), and it enters no "
                "family's tau^2; its omega_full keeps its bands' baseline posterior",
                "draws": "numpy default_rng(SeedSequence([run seed, pool index, span index, 1])): "
                "the band draws, then each cell's effect draws in spec order",
                "families": "class, field, fav, pool_size, surface, venue, and class&<other> for "
                "each other in crossed_with_class",
            },
            "roi78": {
                "pools": "PLACE, QUINELLA_PLACE",
                "model": "#78's fit_segment_roi with the one-winner pools' columns, draws and "
                "seed, on the pool's grouped bets: a combination's outcome is whether it paid; "
                "its price is a representative final odds o = kappa_b x its low-end odds, "
                "kappa_b = sum(return) / sum(low-end odds) over the band's paid combinations on "
                "the span (1 for a band that paid nothing): the band's realised return per unit "
                "of low-end odds when it pays. Then p = omega x rate / o, and omega x rate is the "
                "expected return per unit at the final dividend wherever a paid combination of "
                "the band returns kappa_b x its low end, which kappa_b's definition makes exact in "
                "each band's aggregate; the scaling moves only the band effects (mu[b] + ln "
                "kappa_b), never a segment effect. Like #78 on the other pools, the likelihood "
                "takes a race's combinations as independent bets",
                "variants": "'<pool>/roi78-<reading>/<span>', both readings and spans, reported "
                "beside the empirical-Bayes readings in the private report; the ranking, the rows "
                "and the candidate file keep the empirical-Bayes omega_full",
                "failure": "a fit that cannot apply (a band whose bets all paid, a mode on p = 1) "
                "is a failed record with its reason; the empirical-Bayes variants are scored "
                "apart",
            },
        },
        "readings": {
            "omega_full": "the cell's omega with its band effects, weighted by payout rate: #78's "
            "mean over the cell's bets of exp(mu[band] + every random effect of the bet's levels), "
            "each bet weighing by its race and pool payout rate (SegmentROI.omega_draws(..., "
            "rate_weighted=True)), so that a draw times the cell's payout_rate is the cell's mean "
            "expected return per unit however it mixes ordinary and boosted rates; empirical "
            "Bayes: draws of (sum over bands of w_b rho_b) x the shrunk effect, w_b the cell's "
            "share of payout rate in band b, rho_b's draws its band baseline posterior and the "
            "effect's draws normal, censored at 0 (an effect is a ratio of returns); a band cell "
            "reads its band's baseline posterior and 'all' its flat rho, normal with its "
            "bootstrap sd. The ranking, the rows and the candidate file use it (for Place and "
            "Wide, the empirical-Bayes one)",
            "segment_effect": "the segment effect alone: #78's exp(sum of the cell's named random "
            "effects); empirical Bayes: the shrunk effect, normal. Undefined for 'all' and band "
            "cells",
            "roi78": "for Place and Wide, both readings again from #78's fit on the "
            "representative price ('shrinkage', 'roi78'), recorded as their own variants",
        },
        "bounds": "omega is the posterior mean (of the draws, held within the bounds); one-sided "
        "95% bounds each: omega_lo the 5% and omega_hi the 95% posterior quantile (of the draws; "
        "a normal reading, the empirical-Bayes segment_effect and 'all', omega -/+ 1.6449 sd); "
        "P(omega > 1/rate) is the posterior probability that omega x payout_rate exceeds 1 (the "
        "share of draws, or the normal tail): for omega_full, that the cell's mean expected "
        "return per unit exceeds 1",
        "ranking": "every cell of every pool, full span, omega_full: shrunk_edge = omega x "
        "payout_rate - 1 (expected return per unit at the final price, takeout included, minus 1), "
        "descending; cells without one last; ties in pool and spec order. Every cell is ranked "
        "and none is closed: no kill rule ('authority')",
        "bootstrap": {
            "unit": "meeting (races.meeting_id), D7",
            "n_boot": N_BOOT,
            "level": LEVEL,
            "draws": "multinomial counts of the span's meetings, shared by the pool's cells",
            "seed": "numpy default_rng(SeedSequence([run seed, pool index, span index]))",
        },
        "spa_stepm": {
            "families": "per pool, <family_id>-cells-<pool>: one record per pre-registered cell "
            "per run, its flat-box rule on the full span, scored with bets and failed without; "
            "SPA and StepM (core.eval.snooping) test exactly this run's non-failed cell records, "
            "so a narrower rerun is tested on its own cells and not on every cell an earlier run "
            "scored; reported with the SPA family's registry size (family_size) and, apart, its "
            "variants searched: that plus the same SPA family of each superseded spec",
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
            "family": "one record per pool, shrinkage, reading and span: '<pool>/<reading>/<span>' "
            "(6 x 2 x 2) and, for Place and Wide, '<pool>/roi78-<reading>/<span>' (2 x 2 x 2), 32 "
            "in all; scored when that shrinkage fitted on the span, failed otherwise; its metrics "
            "are every cell's omega with bounds, P(omega > 1/rate), flat rho and mean return",
            "family_size": "registry.family_size(family_id) after the run, on every row and in "
            "the result",
            "variants_searched": "family_size + registry.family_size(f) for each superseded "
            "family f (v1, v2) after the run, on every row and in the result, reported apart from "
            "family_size: their variants ran on the same days and count too; each SPA family's "
            "size and its variants searched (the same sum over its v1 and v2 SPA families) are "
            "reported beside its SPA p",
            "failures": "a pool with a data error in any year (pool_scan refuses a malformed board "
            "or dividend row) is failed on both spans, its cell rules too; a shrinkage that does "
            "not fit fails that pool and span; the private report gives each reason",
        },
        "alternatives_not_run": {
            "joint_cross": "the full joint cross odds band x class x field x favourite x pool "
            "size x surface x venue, 7 x 5 x 4 x 4 x 3 x 2 x 10 = 33600 cells per pool: most "
            "would hold a handful of races or none, no cell could show an effect at any useful "
            "power, and SPA over that many rules would test nothing. The outline registers task "
            "1's marginal cells plus stated crosses; the crosses run are class, the owner's axis "
            "(maidens and debuts against the big graded races), with each other dimension, 150 "
            "of the 186 cells",
            "segment_roi_78_at_the_low_end": "#78's fit on Place or Wide at the bare low end of "
            "the range: its omega would describe the probability of placing at that price, not "
            "the return, which depends on the co-placers; roi78 scales each band's price by its "
            "realised return when it pays ('shrinkage')",
            "point_band_baseline": "v2's point baseline for a Place or Wide band that paid "
            "nothing: a sparse band can pay nothing by chance, so a known zero would rule out a "
            "return the data cannot rule out ('band_baseline')",
            "skipping_races_and_dividends_row_groups": "skipping a races or dividends row group "
            "that spans a holdout day as final_odds's are skipped: their row groups do not follow "
            "the price windows (the jravan builders write races in one row group and dividends in "
            "row groups by row count), so a skipped one would drop the development days of every "
            "window it meets; they hold no price, and are filtered and counted instead until #502 "
            "('store_reader_guarantee')",
            "kill_rule": "closing a pool with no open cell: removed by the owner's reframe "
            "('authority'); every cell is ranked",
            "exact_odds_in_the_fit": "exact odds for every price: a Trio board's long tail makes "
            "too many grouped rows for the 16 GB limit; the 1.1 log grid at the harmonic-mean "
            "price keeps each group's mean implied probability exact",
            "pool_size_basis": "a fixed yen basis for the pool-size tertiles: pool totals move "
            "with sales from year to year, so a fixed basis could sort the cells by year",
            "jump_races": "jump races as a surface level: flat races only (owner, 2026-11-27)",
            "trifecta_and_bracket_quinella": "in N1: a follow-up after N1 (owner, 2026-11-27), "
            "proposed #501",
        },
        "follow_ups": {f"#{number}": what for number, what in FOLLOW_UPS.items()},
        "outputs": {
            "private_report": "$HORSETRACK_DATA_DIR/reports/pool-scan-japan-<spec hash>.md",
            "candidate_cells": "$HORSETRACK_DATA_DIR/reports/candidate-cells/pool-scan.json, every "
            "cell in rank order in the candidate-cell schema plus omega, omega_lo, omega_hi, "
            "family_size and variants_searched",
            "public": "docs/experiments/pool-scan-japan.md: the run, no figure (D21)",
            "registry_summary": "docs/experiments/registry/japan/, by write_summary: each record's "
            "E id and variant, its status and values withheld (licensed source)",
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
    spanning: dict[str, int] = field(default_factory=dict)  # _spanning_decoded


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


def _off_board(pool: str, board: pd.DataFrame, paid: pd.DataFrame) -> set[str]:
    """The races of `board` (one pool's final_odds rows) in which `paid` pays, without refund, a
    combination that is not on the pool's priced, non-refunded board: absent from it, printed
    without a price ('0000', or '****' cancelled after sale), refunded, or not a combination the
    board has. Such a race and pool cannot be scored (prereg_spec 'exclusions')."""
    keys = pool_scan.KEYS
    mine = paid.loc[
        (paid["pool"].to_numpy() == pool) & paid["race_id"].isin(board["race_id"]).to_numpy()
    ]
    refund = mine["is_refund"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    slots = board.loc[board["combination"].to_numpy() != pool_scan.POOL_ROW]
    priced = pd.to_numeric(slots["odds"], errors="coerce").notna().to_numpy()
    staked = pd.MultiIndex.from_frame(slots[keys])
    staked = staked[priced & ~staked.isin(pd.MultiIndex.from_frame(mine.loc[refund, keys]))]
    paying = mine.loc[~refund]
    off = ~pd.MultiIndex.from_frame(paying[keys]).isin(staked)
    return set(paying.loc[off, "race_id"])


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
    """One year of one pool: rates, pool-size tertiles, returns, race sums and grouped bets. A
    race whose pool has no paying dividends row (not established, a special payout, or not yet
    delivered), or pays a combination off the priced, non-refunded board, is unsettled: excluded
    and counted, not scored as if every ticket lost."""
    counted = store.pool_excluded[pool]
    mine_paid = paid["pool"].to_numpy() == pool
    paying = ~paid["is_refund"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    settled = set(paid.loc[mine_paid & paying, "race_id"])
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
        elif rid not in settled:
            counted[UNSETTLED_REASON] += 1
        else:
            rates[rid] = rate.payout_rate
    if not rates:
        return
    keep = pc.and_(
        pc.equal(board["pool"], pool),
        pc.is_in(board["race_id"], value_set=pa.array(list(rates), type=pa.string())),
    )
    rows = board.filter(keep).to_pandas()
    for rid in _off_board(pool, rows, paid):
        counted[OFF_BOARD_REASON] += 1
        del rates[rid]
    if not rates:
        return
    rows = rows.loc[rows["race_id"].isin(list(rates)).to_numpy()]
    mine = races.loc[races["race_id"].isin(list(rates))].reset_index(drop=True)
    total = np.array([totals[(rid, pool)] for rid in mine["race_id"]], dtype=float)
    edges = tuple(float(e) for e in np.quantile(total, [1 / 3, 2 / 3]))
    levels = [
        _race_levels(r, t, edges) for r, t in zip(mine.to_dict("records"), total, strict=True)
    ]
    levels = pd.DataFrame(levels, index=mine["race_id"].to_numpy())[list(SEGMENTS)]
    store.pool_size[pool].update(levels["pool_size"].to_dict())

    rows = pool_scan.combination_returns(rows, paid, pool=pool)
    rows["rate"] = rows["race_id"].map(rates)
    for rate, part in rows.groupby("rate", sort=True):
        edges_ = BAND_EDGES[pool]
        sums = pool_scan.race_returns(part, payout_rate=rate, band_edges=edges_)
        bets = pool_scan.grouped_bets(part, by=["race_id"], payout_rate=rate, band_edges=edges_)
        odds = bets.assign(
            odds_sum=bets["odds"] * bets["count"], hit_odds=bets["odds"] * bets["won"]
        )
        odds = odds.groupby(["race_id", "band"], as_index=False)[["odds_sum", "hit_odds"]].sum()
        sums = sums.merge(odds, on=["race_id", "band"], how="left")
        store.sums[pool].append(sums.fillna({"odds_sum": 0.0, "hit_odds": 0.0}))
        if len(bets):
            _accumulate(store, pool, year, _binned(pool, bets, levels))


def _race_id_day(value: Any) -> date:
    """The day a jravan race id encodes (jra:<YYYY-MM-DD>:<VENUE>:<NN>); ValueError otherwise."""
    parts = str(value).split(":")
    if len(parts) != 4 or parts[0] != "jra":
        raise ValueError(f"race id {value!r} is not jra:<YYYY-MM-DD>:<VENUE>:<NN>")
    return date.fromisoformat(parts[1])


def _row_group_days(name: str, column: str) -> list[tuple[date, date] | None]:
    """Each non-empty row group's first and last day of the store table `name`, from its footer
    statistics alone (no page is read): `race_date` as stored, or, for `race_id`, the day the race
    id encodes. None for a row group whose statistics cannot say."""
    meta = pq.read_metadata(paths.processed_dir(SOURCE) / f"{name}.parquet")
    index = meta.schema.to_arrow_schema().get_field_index(column)
    out: list[tuple[date, date] | None] = []
    for g in range(meta.num_row_groups):
        group = meta.row_group(g)
        if group.num_rows == 0:
            continue
        stats = group.column(index).statistics if index >= 0 else None
        if stats is None or not stats.has_min_max:
            out.append(None)
            continue
        try:
            if column == "race_id":
                out.append((_race_id_day(stats.min), _race_id_day(stats.max)))
            else:
                out.append((pd.Timestamp(stats.min).date(), pd.Timestamp(stats.max).date()))
        except ValueError:
            out.append(None)
    return out


def _spans_holdout(first: date, last: date) -> bool:
    """Whether a holdout day (`is_holdout_day`, looked up at call time) lies in first..last."""
    days = (first + timedelta(days=k) for k in range((last - first).days + 1))
    return any(holdout.is_holdout_day(COUNTRY, day) for day in days)


def _price_windows(start: date, end: date, store: _Store) -> list[tuple[date, date]]:
    """The day windows of start..end on which the scan reads prices (prereg_spec 'store'): every
    day that no `final_odds` row group spanning a holdout day covers. A read of a window prunes, by
    the same statistics, every row group outside it, so no spanning group is decoded; each one in
    start..end is counted under ROW_GROUP_REASON. A row group without race_date statistics cannot
    be shown clear of the holdout: ValueError, before any page is read."""
    skipped = []
    for days in _row_group_days("final_odds", "race_date"):
        if days is None:
            raise ValueError(
                "final_odds has a row group without race_date statistics: it cannot be shown "
                "clear of the holdout (D14), so nothing is read"
            )
        if days[1] >= start and days[0] <= end and _spans_holdout(*days):
            skipped.append(days)
    if skipped:
        store.excluded[ROW_GROUP_REASON] += len(skipped)
    windows, day = [], start
    for first, last in sorted(skipped):
        if first > day:
            windows.append((day, first - timedelta(days=1)))
        day = max(day, last + timedelta(days=1))
    if day <= end:
        windows.append((day, end))
    return windows


def _spanning_decoded(windows: list[tuple[date, date]]) -> dict[str, int]:
    """Per table read by filter rather than by window (`races`, `dividends`), its row groups that
    meet a window and span a holdout day (or whose statistics cannot say): their holdout rows are
    decoded and dropped inside the scan until #502 isolates row groups by day."""
    out = {}
    for name, column in (("races", "race_date"), ("dividends", "race_id")):
        out[name] = sum(
            days is None
            or (
                any(days[0] <= b and days[1] >= a for a, b in windows)
                and _spans_holdout(days[0], days[1])
            )
            for days in _row_group_days(name, column)
        )
    return out


def _read_windows(name: str, columns: list[str], windows, *, touches_price: bool, where=None):
    tables = [
        read_store_table(
            name,
            columns=columns,
            start=a,
            end=b,
            touches_price=touches_price,
            where=where,
            source=SOURCE,
        )
        for a, b in windows
    ]
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]


def _read_year(windows: list[tuple[date, date]], store: _Store, calendar: BoostCalendar) -> None:
    """One calendar year's price windows: races, the board and the dividends, then every pool."""
    races = _read_windows("races", RACE_COLUMNS, windows, touches_price=False).to_pandas()
    if races.empty:
        return
    board = _read_windows(
        "final_odds",
        BOARD_COLUMNS,
        windows,
        touches_price=True,
        where=ds.field("pool").isin(list(POOLS)),
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
    year = windows[0][0].year
    for pool in POOLS:
        try:
            _score_pool(pool, races, board, paid, totals, calendars, year, store)
        except Exception as exc:  # a data error fails the pool's variants, not the scan
            store.errors[pool].append(f"{year}: {type(exc).__name__}: {exc}")


def _read_store(start: date, end: date, log: Callable[[str], None]) -> _Store:
    store, calendar = _Store(), shipped_boosts()
    windows = _price_windows(start, end, store)
    store.spanning = _spanning_decoded(windows)
    for year in range(start.year, end.year + 1):
        first, last = date(year, 1, 1), date(year, 12, 31)
        mine = [(max(a, first), min(b, last)) for a, b in windows if a <= last and b >= first]
        if not mine:
            continue
        log(f"reading {year}")
        _read_year(mine, store, calendar)
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
    bands: list[str] = field(default_factory=list)  # band_labels(pool)
    band_totals: dict[str, np.ndarray] = field(default_factory=dict)  # per band: n, ret, rate, ...
    band_rate: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # cells x bands
    roi78_fitted: bool = False  # Place and Wide: #78's fit on the representative price
    roi78_note: str | None = None
    roi78_info: dict = field(default_factory=dict)


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
    bands = band_labels(pool)
    code = np.asarray(pd.Categorical(sums["band"], categories=bands).codes)  # unpriced: -1
    banded = code >= 0
    band_totals = {
        name: np.bincount(code[banded], weights=values[banded], minlength=len(bands))
        for name, values in (
            ("n", n),
            ("ret", ret),
            ("rate", rate),
            ("odds", odds),
            ("hit_odds", sums["hit_odds"].to_numpy(float)),
        )
    }
    band_rate = np.zeros((len(cells), len(bands)))

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
        live = m & banded
        band_rate[j] = np.bincount(code[live], weights=rate[live], minlength=len(bands))
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
    span = _Span(out, first, columns["n"] - columns["ret"], False)
    span.bands, span.band_totals, span.band_rate = bands, band_totals, band_rate
    return span


def _band_draws(span: _Span, rng: np.random.Generator) -> np.ndarray:
    """Posterior draws (bands x N_DRAWS) of each odds band's baseline rho_b, its flat rho
    (prereg_spec 'shrinkage', 'band_baseline'): a band that paid, a gamma with its rho_b as mean
    and its meeting-bootstrap sd as sd (a point at rho_b when that sd is 0); a band that paid
    nothing, the Jeffreys posterior of a zero-hit Poisson hit rate times its mean odds over its
    mean payout rate, Gamma(1/2, scale sum(odds) / (n sum(rate)))."""
    totals = span.band_totals
    out = np.zeros((len(span.bands), N_DRAWS))
    for k, band in enumerate(span.bands):
        n, ret, rate, odds = (float(totals[name][k]) for name in ("n", "ret", "rate", "odds"))
        if not n > 0:
            continue
        if ret > 0:
            rho, sd = ret / rate, span.cells[f"band:{band}"].get("sd_rho", 0.0)
            if sd > 0 and np.isfinite(sd):
                out[k] = rng.gamma((rho / sd) ** 2, sd**2 / rho, N_DRAWS)
            else:
                out[k] = rho
        else:
            out[k] = rng.gamma(JEFFREYS, odds / (n * rate), N_DRAWS)
    return out


def _eb_readings(span: _Span, seed_seq) -> None:
    """Place and Wide: the normal empirical-Bayes shrink of each family's cell effects over band
    baselines that carry their posterior (`_band_draws`). A cell's omega_full draws are its
    rate-weighted baseline draws times its effect's draws (normal, censored at 0); a cell whose
    bets all sit in bands that paid nothing has no expected return under the point baselines (its
    effect is 0 / 0): its effect is the family's prior, N(1, tau^2), and it enters no tau^2
    (prereg_spec 'zero_return_band')."""
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
    rng = np.random.default_rng(seed_seq)
    rho = _band_draws(span, rng)
    for j, (cell, c) in enumerate(span.cells.items()):
        if not c["n_bets"]:
            continue
        rate, family = c["payout_rate"], _family_of(cell)
        if cell == "all":
            c["omega_full"] = _normal_summary(c["flat_rho"], c["sd_rho"], rate)
            continue
        weights = span.band_rate[j] / span.band_rate[j].sum()
        baseline = weights @ rho
        if family is None:  # a band cell: its band's baseline
            c["omega_full"] = _draw_summary(baseline, rate)
            continue
        if np.isfinite(c["effect"]):
            se = c["se_effect"] if np.isfinite(c["se_effect"]) else 0.0
            t2 = tau2[family]
            b = t2 / (t2 + se**2) if t2 + se**2 > 0 else 0.0
            effect, sd = 1 + b * (c["effect"] - 1), float(np.sqrt(b) * se)
        else:  # no expected return under the point baselines: the family's prior
            effect, sd = 1.0, float(np.sqrt(tau2[family]))
        draws = np.maximum(effect + sd * rng.standard_normal(N_DRAWS), 0.0)
        c["omega_full"] = _draw_summary(baseline * draws, rate)
        c["segment_effect"] = _normal_summary(effect, sd, rate)
    span.fitted, span.fit_info = True, {"tau2": tau2}


def _kappa(span: _Span) -> dict[str, float]:
    """Per odds band, a paid combination's realised return per unit of its low-end odds on the
    span: sum(return) / sum(low-end odds) over the band's paid combinations, 1 for a band that
    paid nothing (Place and Wide's representative price, prereg_spec 'shrinkage')."""
    ret, hit = span.band_totals["ret"], span.band_totals["hit_odds"]
    return {
        band: float(ret[k] / hit[k]) if hit[k] > 0 else 1.0 for k, band in enumerate(span.bands)
    }


def _fit_bets(frame: pd.DataFrame, scale: dict[str, float] | None = None) -> pd.DataFrame:
    """The fit's input: segment and interaction levels (categoricals, no per-row strings), band,
    harmonic-mean odds (times `scale[band]` when given), counts."""
    out = pd.DataFrame({name: frame[name] for name in [*SEGMENTS, "band"]})
    group = out["class"].cat
    for other in CROSSED:
        levels = out[other].cat.categories
        codes = group.codes.to_numpy(dtype=np.int64) * len(levels) + out[other].cat.codes
        names = [f"{g}|{level}" for g in group.categories for level in levels]
        out[f"class_x_{other}"] = pd.Categorical.from_codes(codes, categories=names)
    odds = frame["count"].to_numpy(dtype=float) / frame["inv"].to_numpy(dtype=float)
    if scale is not None:
        band = out["band"].cat
        factor = np.array([scale.get(b, 1.0) for b in band.categories], dtype=float)
        odds = odds * factor[band.codes.to_numpy()]
    out["odds"] = odds
    out["takeout"] = frame["takeout"].to_numpy(dtype=float)
    out["count"] = frame["count"].to_numpy(dtype=int)
    out["won"] = frame["won"].to_numpy(dtype=int)
    return out[[*FIT_COLUMNS, "band", "odds", "takeout", "count", "won"]]


def _fit_readings(pool: str, span: _Span, bets: pd.DataFrame, seed: int, prefix: str = "") -> dict:
    """#78's hierarchical fit of one pool and span, and both readings of every cell (under
    `prefix` + the reading's name). omega_full weighs each bet by its payout rate, so omega x the
    cell's mean rate is its mean expected return per unit even where boosted and ordinary rates
    mix. Returns the fit's tau summary."""
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
            draws = fit.omega_draws(segment, band=band, rate_weighted=True)
            c[prefix + "omega_full"] = _draw_summary(draws, c["payout_rate"])
            if segment:
                effect = _draw_summary(fit.effect_draws(segment), c["payout_rate"])
                c[prefix + "segment_effect"] = effect
        except (KeyError, ValueError):
            continue
    return {"tau": fit.tau, "tau_ess": fit.tau_ess, "tau_rejected": fit.tau_rejected}


def _scan_pool(pool: str, store: _Store, races: pd.DataFrame, seed: int) -> dict[str, _Span]:
    """Both spans of one pool, one fit at a time; each span's grouped bets are freed after it."""
    out = {}
    for s, (name, first) in enumerate(SPANS.items()):
        seq = np.random.SeedSequence([seed, POOLS.index(pool), s])
        span = _span_sums(pool, first, store, races, seq)
        out[name] = span
        bets = store.bets.pop((pool, name), None)
        if span.note:
            continue
        fit_seed = int(seq.generate_state(1)[0])
        try:  # every failure is a failed record, and the scan goes on
            if pool in EB_POOLS:
                _eb_readings(span, np.random.SeedSequence([seed, POOLS.index(pool), s, 1]))
            else:
                span.fit_info = _fit_readings(pool, span, _fit_bets(bets), fit_seed)
                span.fitted = True
        except Exception as exc:
            span.note = f"{type(exc).__name__}: {exc}"
        if pool in EB_POOLS:
            try:
                if bets is None:
                    raise ValueError("no grouped bets for the fit")
                scaled = _fit_bets(bets, scale=_kappa(span))
                span.roi78_info = _fit_readings(pool, span, scaled, fit_seed, prefix=f"{ROI78}_")
                span.roi78_fitted = True
            except Exception as exc:
                span.roi78_note = f"{type(exc).__name__}: {exc}"
        del bets
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


def variants_searched(registry: Registry, family: str = FAMILY_ID) -> int:
    """The variants searched for `family` (FAMILY_ID or one of its SPA families): its registry
    family size plus that of the same family of each superseded spec (SUPERSEDED_FAMILIES), whose
    variants ran on the same days (prereg_spec, 'variants')."""
    suffix = family[len(FAMILY_ID) :]
    return registry.family_size(family) + sum(
        registry.family_size(old + suffix) for old in SUPERSEDED_FAMILIES
    )


class _ThisScan:
    """The registry as SPA and StepM see one pool's cell rules: the SPA family's records are this
    scan's cell records (every pre-registered cell, failed where it had no bets), not every cell
    an earlier, longer run scored. Its family size, and everything else, reads through to the
    registry."""

    def __init__(self, registry: Registry, family: str, records: list[RunRecord]) -> None:
        self._registry, self._family, self._records = registry, family, list(records)

    def records(self, family_id: str | None = None) -> list[RunRecord]:
        if family_id == self._family:
            return list(self._records)
        return self._registry.records(family_id)

    def family_size(self, family_id: str) -> int:
        return self._registry.family_size(family_id)


def _snoop(
    pool: str, span: _Span, registry: Registry, records: list[RunRecord], seed: int
) -> dict[str, Any]:
    """SPA and StepM over the pool's cell rules: this scan's non-failed cell records."""
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
        "registry": _ThisScan(registry, spa_family(pool), records),
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
            "variants_searched": variants_searched(registry, spa_family(pool)),
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
    family_size: int  # registry.family_size(FAMILY_ID) after the run
    variants_searched: int  # that plus the superseded families' sizes
    report_path: Path
    report_name: str
    candidates_path: Path
    candidates_name: str
    failed: dict[str, str]  # FAMILY_ID variant -> why
    snooping: dict[str, dict]
    excluded: dict[str, int]
    pool_excluded: dict[str, dict[str, int]]
    fit_info: dict[str, dict]
    spanning: dict[str, int] = field(default_factory=dict)  # races/dividends, _spanning_decoded
    summary_error: str | None = None  # why write_summary could not regenerate the summary


def _row(
    pool: str, cell: str, c: dict, base_rate: float, size: int, searched: int
) -> dict[str, Any]:
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
        "family_size": size,
        "variants_searched": searched,
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
        fits = [(None, "segment_roi_78")]
        if pool in EB_POOLS:
            fits = [(None, "empirical_bayes"), (ROI78, "segment_roi_78_range")]
        for alt, shrink in fits:
            for reading in READINGS:
                for name, span in pools[pool].items():
                    variant = f"{pool}/{reading}/{name}"
                    key, fitted, note = reading, span.fitted, span.note
                    if alt is not None:
                        variant = f"{pool}/{alt}-{reading}/{name}"
                        key, fitted = f"{alt}_{reading}", span.roi78_fitted
                        note = span.roi78_note or span.note
                    metrics = _reading_metrics(name, span.cells, key) if fitted else []
                    if not metrics:
                        failed[variant] = note or "no cell has this reading"
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
    size, searched = registry.family_size(FAMILY_ID), variants_searched(registry)
    snoop = {
        pool: _snoop(pool, pools[pool][MAIN_SPAN], registry, cell_records[pool], seed)
        for pool in POOLS
    }

    jra = config.load_source(COUNTRY, "jra")
    rows = _ranked(
        [
            _row(pool, cell, c, round(1 - jra.pools[pool].takeout_on(date_to), 6), size, searched)
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
        variants_searched=searched,
        report_path=report,
        report_name=_display(report),
        candidates_path=candidates,
        candidates_name=_display(candidates),
        failed=failed,
        snooping=snoop,
        excluded=dict(store.excluded),
        pool_excluded={p: dict(c) for p, c in store.pool_excluded.items()},
        fit_info={
            f"{p}/{n}": s.fit_info
            | ({"note": s.note} if s.note else {})
            | (
                {ROI78: s.roi78_info | ({"note": s.roi78_note} if s.roi78_note else {})}
                if p in EB_POOLS
                else {}
            )
            for p in POOLS
            for n, s in pools[p].items()
        },
        spanning=dict(store.spanning),
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
    superseded = ", ".join(f"`{f}`" for f in SUPERSEDED_FAMILIES)
    lines = [
        f"# N1 final-board segment scan, Japan (#{TICKET}): private, every figure withheld (D21)",
        "",
        f"- family `{result.family_id}`, spec sha256 `{result.spec_hash}`, git `{result.git_sha}`",
        f"- development days from {result.start} to {result.end}; recent span from {RECENT_START}",
        f"- seed {result.seed}; family size {result.family_size}; variants searched (that plus "
        f"superseded {superseded}): {result.variants_searched}",
        f"- written {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- selection hygiene: bands, favourite and pool size come from the final board (#88); "
        "a cell is a calibration target at the final price, not a betting rule",
        "- row groups of races and dividends that meet a price window and span a holdout day "
        f"(decoded and filtered, #502): {result.spanning}",
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
        eb = pool in EB_POOLS
        lines += [f"## {pool}", ""]
        lines += [
            "- shrinkage "
            + ("empirical Bayes (ranks); #78 on the representative price (roi78)" if eb else "#78")
        ]
        for name, span in pools[pool].items():
            lines.append(f"- {name}: {'fitted' if span.fitted else 'not fitted'} {span.fit_info}")
            if span.note:
                lines.append(f"- {name} failed: {span.note}")
            if eb:
                state = "fitted" if span.roi78_fitted else "not fitted"
                lines.append(f"- {name} {ROI78}: {state} {span.roi78_info}")
                if span.roi78_note:
                    lines.append(f"- {name} {ROI78} failed: {span.roi78_note}")
        spa = snoop.get("spa")
        lines.append(
            f"- SPA over `{spa_family(pool)}` (benchmark: no bet): "
            + (
                f"p {spa['pvalue']:.4f} (lower {spa['pvalue_lower']:.4f}, upper "
                f"{spa['pvalue_upper']:.4f}), {spa['n_rules']} rules, family size "
                f"{spa['family_size']}, variants searched {spa['variants_searched']}, "
                f"{spa['n_clusters']} meetings"
                if spa
                else f"not run ({snoop.get('error')})"
            )
        )
        if "stepm" in snoop:
            lines.append(f"- StepM survivors at {STEPM_LEVEL}: {snoop['stepm'] or 'none'}")
        rank = {r["cell_id"]: k for k, r in enumerate(result.rows, 1) if r["pool"] == pool}
        header = ["rank", "cell", "n", "meetings", "rho", "mean return", "omega", "lo", "hi"]
        header += ["P(>1/rate)", "effect", "lo", "hi", "P(>1/rate)"]
        if eb:
            header += [f"{ROI78} omega", "lo", "hi", "P(>1/rate)", f"{ROI78} effect"]
        header += ["2024+ n", "2024+ rho", "2024+ omega", "lo", "hi", "2024+ effect"]
        table = []
        for cell, c in main.cells.items():
            r = recent.cells[cell]
            effect = (r.get("segment_effect") or {}).get("omega")
            row = [str(rank[cell]), cell, str(c["n_bets"]), str(c["n_meetings"])]
            row += [_num(c.get("flat_rho")), _num(c.get("mean_return"))]
            row += _bounds(c.get("omega_full")) + _bounds(c.get("segment_effect"))
            if eb:
                alt = (c.get(f"{ROI78}_segment_effect") or {}).get("omega")
                row += _bounds(c.get(f"{ROI78}_omega_full")) + [_num(alt)]
            row += [str(r["n_bets"]), _num(r.get("flat_rho"))]
            row += _bounds(r.get("omega_full"))[:3] + [_num(effect)]
            table.append(row)
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
    v1, v2 = SUPERSEDED_FAMILIES
    lines = [
        f"# Experiment report: N1, the final-board segment scan of six pools (Japan, #{TICKET})",
        "",
        f"Ticket #{TICKET} (epic #72). **Every figure is {withheld}**: the scan reads the JRA-VAN "
        "store, a licensed source, so this report says what ran and where its figures are, and "
        "gives no count, rate, span, sign, bound or rank derived from that data. Nothing is "
        "adopted, no bet follows from it, and the holdout is not opened.",
        "",
        f"- Registry family `{result.family_id}`, spec sha256 `{result.spec_hash}` "
        "(`docs/specs/pool-scan-prereg.md` quotes the spec). Claude registered and froze it "
        "before the run read any row.",
        "- Authority: the owner's reframe of 2026-11-27 on #430 removed the sign-off (\"Claude "
        "pre-registers the cell list and metrics and freezes them in the registry before the "
        'run. No owner sign-off.") and the kill rule ("It ranks every cell by expected return '
        "per 100 yen at the final price (takeout included, CI beside it). None is closed, and "
        'models aim at the top-ranked cells first."). The frozen spec quotes it in full '
        "(`authority`).",
        f"- It supersedes v2, `{v2}` (spec sha256 `{SUPERSEDED_SPECS[v2]}`), and v1, `{v1}` (spec "
        f"sha256 `{SUPERSEDED_SPECS[v1]}`), both run the same day. The second review of PR #496 "
        "found that v2 scored a race whose dividend named a combination off the priced board as "
        "if every ticket lost, read a Place or Wide band that paid nothing as a known zero, "
        "reported variants searched as the family size, decoded price row groups that span a "
        "holdout day, crossed class with three dimensions only, and gave Place and Wide no #78 "
        "fit. v1's defects are in the spec (`supersedes`). Their records stay in the registry, "
        "superseded; nothing is deleted.",
        f"- Requested range: development days from {result.start} to {result.end} (the run's "
        "arguments, not the store's measured span); holdout days are never read and "
        "`LockedHoldout` is never opened.",
        "- The price rows (D14, #502): from the footer statistics of `final_odds`, a row group "
        "whose race_date range spans a holdout day is never decoded. Prices are read only on the "
        "day windows no such row group touches, each through `read_store_table`, whose scan "
        "prunes by the same statistics every row group outside the window and drops holdout "
        "days. The jravan builders write one row group per year, so until #502 isolates row "
        "groups by day a year whose row group spans a holdout day is skipped whole, its "
        "development days unscored; the private report counts the row groups skipped. `races` "
        "(no price or result) is read on the same windows and `dividends` filtered in its scan "
        "to the windows' development race ids: their row groups are filtered, not skipped, so "
        "one spanning a holdout day would still be decoded, its holdout rows dropped inside the "
        "scan, until #502 isolates them too; the private report counts them.",
        f"- Seed {result.seed}; code at `{result.git_sha[:12]}`; command `uv run horsetrack japan "
        f"pool-scan run --from {result.start} --to {result.end} --seed {result.seed}`.",
        f"- Family size {result.family_size} (this family's distinct variants in the registry); "
        f"variants searched {result.variants_searched} (that plus v1's and v2's, whose variants "
        "ran on the same days). Both are on every row.",
        f"- Private report: `$HORSETRACK_DATA_DIR/{result.report_name}`; candidate cells: "
        f"`$HORSETRACK_DATA_DIR/{result.candidates_name}`; both written through `write_report` "
        "with the source `jravan`.",
        "",
        "## Registry records",
        "",
        f"One record per pool, shrinkage, reading and span in `{result.family_id}`, in spec order:",
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
        "cells are `all`, each odds band, each level of class, field size, favourite strength, "
        "pool size, surface and venue, and class crossed with each of the other six. The "
        "shrinkage is #78's hierarchical fit for the one-winner pools, its omega weighted by "
        "payout rate. For Place and Wide it is an empirical-Bayes shrink of the realised return "
        "over odds-band baselines that carry their posterior (a band that paid nothing keeps a "
        "Jeffreys posterior, not a known zero), which ranks them; beside it, as variants of their "
        f"own (`{ROI78}`), #78's fit on a representative price: each combination's low-end odds "
        "times its band's realised return per unit of low-end odds when it pays, the outcome "
        "being whether it paid. Every cell of every pool is ranked by its shrunk expected return "
        f"(the owner's reframe: none is closed); the ranks, the omegas and their bounds are "
        f"{withheld}. Which variants failed, and their figures, are {withheld}: the private "
        "report gives each reason, and the registry summary shows no status of a licensed-source "
        f"run. v1's and v2's variants, superseded, are records of `{v1}` and `{v2}` and count "
        "toward variants searched.",
        "",
        "## Alternatives rejected and why",
        "",
        "- The full joint cross (odds band x class x field x favourite x pool size x surface x "
        "venue, 33600 cells per pool): most cells would hold a handful of races or none, so none "
        "could show an effect at any useful power, and SPA over that many rules would test "
        "nothing. The cells cross class, the owner's axis (maidens and debuts against the big "
        "graded races), with each other dimension: 150 of each pool's 186 cells.",
        "- #78's fit on Place and Wide at the bare low end of the range: its omega would describe "
        "the probability of placing at that price, not the return; the representative price "
        "scales each band by its realised return when it pays.",
        "- A point baseline for a Place or Wide band that paid nothing (v2): a sparse band can "
        "pay nothing by chance, so a known zero would rule out a return it cannot rule out.",
        "- A kill rule (close a pool with no open cell): removed by the owner's reframe; every "
        "cell is ranked and none is closed.",
        "- Exact odds in the fit for every price: a Trio board's long tail makes the grouped rows "
        "too many for the 16 GB limit; from odds 10 a 1.1 log grid with the harmonic-mean price "
        "keeps the mean implied probability exact.",
        "- A fixed yen basis for the pool-size tertiles: pool totals move with sales from year to "
        "year, so a fixed basis could sort the cells by year.",
        "- Trifecta and bracket quinella: a follow-up that runs after N1 (owner, 2026-11-27).",
        "",
        "## What this rules out",
        "",
        f"{withheld}: the private report gives, per pool, which cells' one-sided bounds cover "
        "break-even, the SPA p and the StepM survivors. Whatever it shows, an N1 cell is a "
        "calibration target at the final price (bands, favourite and pool size are read from the "
        "final board, look-ahead for selection, #88), not a betting rule.",
        "",
        "## Follow-ups filed",
        "",
        *(f"- #{number}: {what}." for number, what in FOLLOW_UPS.items()),
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
        "family_size": result.family_size,
        "variants_searched": result.variants_searched,
        "report": result.report_name,
        "candidates": result.candidates_name,
        "run_ids": [i for _, i in result.run_ids],
        "summary_error": result.summary_error,
    }
    typer.echo(json.dumps(summary, indent=2))
