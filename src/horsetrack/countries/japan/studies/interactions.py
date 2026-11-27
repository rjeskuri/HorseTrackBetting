"""The C2 interaction blocks' real-data run (#93; 08 §3.3, §4 E35, E36, §8 B5; 05 §4.1-4.3):
`horsetrack japan interaction-study`.

Level 1 (results only, no price; D10). Every starter of the JRA store's run races with a winner
from 2010 (`level1_frame`): race_id, race_date, race_utc (the post time, JST, in UTC), horse_id,
finish_pos (none for a starter without a placing: it stays in every choice set and is never
chosen), jockey_key, trainer_key, venue, distance_m, surface, meeting_id and race_class. The base
member is #92's model without the sire (the JRA-VAN pedigree would withhold every figure, D21; the
sire's terms are #95's): `AbilityModel(k=3)` at the fixed discounts DISCOUNTS on #91's rating
member at the (drift, scale) #91's rule picks per outer fold (`studies.ability.select_pairs`),
with jockey and trainer effects shrunk toward 0. Each block member is the base with its blocks
added (`models.interactions.fit_with_blocks`):

- horse_x_jockey (E35, the negative control): a variance component per (horse, jockey) cell;
- trainer_x_track (E36): a variance component per (trainer, venue) cell plus the one-factor
  east/west term, a rank-one (trainer, venue) block;
- horse_aptitude (E93): the structured a_h + b_h z(d) + c_h z(d)^2 + e_h 1{dirt} plus <U_h,
  W_track>, a rank-one (horse, venue) block (08 §3.3's form, without its pedigree prior: #95);
- trainer_debut (E93): a debut indicator and a variance component per (trainer, debut) cell, so
  a debut runner's level is shrunk toward its trainer's debut mean beyond the trainer's main
  effect (which the base already uses as every runner's prior mean);
- transfer_step (E93): an indicator of a run after a stable change of trainer
  (`models.interactions.transfer_step`, STABLE_RUNS runs with the new trainer).

Horse x trainer is not fitted: where a horse keeps one trainer its cell is the horse (aliased,
08 §3.3), and the base has no static horse term for it to be told from; the registry records it
as a variant that could not be fitted, with its cell counts. The cell-size table (prior runs only,
`cell_sizes`) is measured first, per interaction, over the out-of-fold starters.

Splits: D9's quarters (expanding, 1-week embargo) over the level-1 dates; level 1 is out of fold
from 2015-Q1 (`studies.ability.LEVEL1_FROM`), each quarter fitted on every earlier row to its
embargo, so level 2 trains on five seasons of members from 2020-Q1. #91's rating pair is picked
per outer fold on the inner races from 2011-Q1 inside its training rows. tau per block is the
empirical-Bayes value of each fold's fit; its profile-evidence bounds are computed once, on the
last outer fold's fit (the most training rows).

Level 2 (the final Win price): `studies.meta.store_frame`'s priced starters, D4's calibrated
market out of fold, #87's stage-2 meta (u = alpha * m + beta * ln q, unbounded, ridge 0) per member
on each D9 fold's training rows that have every member, and the market-only stage-2 meta. Scored:
every D9 fold, from 2020-Q1. Per block member: its standalone paired ΔLL against the base alone;
its ΔLL at the final price, the paired ΔLL of its stage-2 meta against the base's stage-2 meta; its
C4 marginal against the market-only meta; and its tail return at the final price, the realised
Win return per unit stake of every scored runner banded by its stage-2 claim p x O_final
(`core.eval.tail.tail_curve`, D7's meeting bounds), the top band a leaderboard row. A runner is
settled at the store's recorded Win dividend for its race and horse number (`win_paid`): a JRA
dead heat pays each winner its own dividend, and a dividend paid before a later disqualification
stands. A race with no WIN row at all is unsettled (an interrupted store refresh), never a race
every starter lost: its starters' return is NaN, read only on the scored rows, so a scored one
fails each tail variant (a failed record) and one that only trains is never read.

Every variant, failures included, is a registry record; the JRA store is netkeiba results (no
JRDB, no JRA-VAN), so the records carry the source `jra` and their figures are public. Holdout
days are never read (`read_store_table`, `split_off`); the holdout is not opened; nothing is
adopted (D16, D19: ranked, not adopted). A horse x jockey estimate is predictive, never causal
(booking endogeneity, 05 §4.2), and never a bet trigger. A `--trial` run's records go to the
family's `-trial` family with `params.trial` set, and it never freezes the family's spec.

The level-1 fits are checkpointed (`save_checkpoint`, one file published by rename) under a key of
the spec's level1, tau and folds sections and the store builds they read (`level1_key`), so a
stopped launch resumes and a level-2 amendment re-scores the saved out-of-fold predictions. Each
member-fold keeps the git revision it was fitted at, and every record after level 1 names them
(`params.fit_revisions`); every failed member-fold fit, an earlier launch's included, is a failed
record of its own (`fit_failures`); a stopped or failed launch records every planned variant as
failed. A completed launch's artifacts (the report, its JSON, the export) are named by the spec,
the builds and a digest of its results (`artifact_stem`, `results_digest`), and its records cite
them; an existing artifact is never rewritten. So a launch that changes a result (a retry that
recovers a failed fold) writes beside the earlier launch's files, whose records keep citing
them, and a record equal to a held one but for git_sha and its artifacts is not appended again.

Family `japan-c2-interactions-final-win-v2` supersedes `japan-c2-interactions-final-win` (#93's
review, round 1), whose records stay in the registry: v1 settled a winner at its final odds over
the race's number of winners, and its records named one git revision for fits made at two.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import resource
import tempfile
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import typer

from horsetrack.core import paths
from horsetrack.core.config import load_source
from horsetrack.core.eval.folds import Fold, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha, split_off
from horsetrack.core.eval.registry import RunRecord, default_registry, summary_dir, write_summary
from horsetrack.core.eval.reports import write_report
from horsetrack.core.eval.tail import tail_curve
from horsetrack.countries.japan.models import interactions as ix
from horsetrack.countries.japan.models.ability import AbilityModel, _softmax, _stages
from horsetrack.countries.japan.models.meta import MetaSpec
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table
from horsetrack.countries.japan.studies import rating as c2_rating
from horsetrack.countries.japan.studies.ability import (
    INNER_FROM,
    LEVEL1_FROM,
    _comparison,
    _market_only,
    level1_quarters,
    select_pairs,
)
from horsetrack.countries.japan.studies.calibration import (
    Study,
    Variant,
    _markdown_table,
    _metric,
    bootstrap_settings,
    store_builds,
)
from horsetrack.countries.japan.studies.meta import (
    CLUSTERED,
    Scored,
    _clustered_metrics,
    _meta_metrics,
    _meta_oof,
    _pooled,
    _run_id,
    _try,
    baseline_oof,
    read_development,
    restrict,
    store_frame,
)

COUNTRY = "japan"
TICKET = 93
FAMILY_ID = "japan-c2-interactions-final-win-v2"
TRIAL_FAMILY_ID = f"{FAMILY_ID}-trial"
SUPERSEDES = "japan-c2-interactions-final-win"
SUPERSEDED_SPEC = "3060dfbf9626cf7d46186153de6ee4b6c0b65111acadbefc464d5aca300bded0"
ARTIFACTS = "interaction_artifacts"
SOURCE = "jra"
SOURCES = (SOURCE,)
WIN = "WIN"  # the dividends table's Win pool code
DIVIDEND_COLUMNS = ["race_id", "pool", "combination", "dividend", "is_refund"]
LEVEL1_SECTIONS = ("level1", "tau", "folds")  # the spec sections the level-1 fits depend on
META = "__meta__"  # the checkpoint's metadata entry
BOOTSTRAP_SEED = 93
RESULTS_FROM = c2_rating.RATING_FROM  # 2010-01-01
DISCOUNTS = (1.0, 0.5, 0.25)
STABLE_RUNS = 2
INFO_PER_RUN = 0.176  # 08 §3.3: nats per run of top-3 training
TAIL_QUANTILES = (0.5, 0.9, 0.99)
ENTITIES = ("jockey_key", "trainer_key")
RATING = "rating"
BASE = "base"
MARKET_ONLY = "market_only"
VC, STRUCTURED, LOW_RANK = ix.VARIANCE_COMPONENT, ix.STRUCTURED, ix.LOW_RANK
B = ix.InteractionBlock
MEMBERS: dict[str, tuple[list, list[str]]] = {  # member -> (its blocks, its extra features)
    BASE: ([], []),
    "horse_x_jockey": ([B("horse_x_jockey", ("horse_id", "jockey_key"), VC)], []),
    "trainer_x_track": (
        [
            B("trainer_x_track", ("trainer_key", "venue"), VC),
            B("trainer_x_track_factor", ("trainer_key", "venue"), LOW_RANK),
        ],
        [],
    ),
    "horse_aptitude": (
        [
            B("distance_aptitude", ("horse_id", "distance_m", "surface"), STRUCTURED),
            B("horse_x_track", ("horse_id", "venue"), LOW_RANK),
        ],
        [],
    ),
    "trainer_debut": ([B("trainer_debut", ("trainer_key", "debut_flag"), VC)], ["debut"]),
    "transfer_step": ([], ["transfer"]),
}
HYPOTHESIS = {"horse_x_jockey": "E35", "trainer_x_track": "E36"}  # E93 otherwise
CELLS = {  # the cell-size table's interactions
    "horse_x_jockey": ("horse_id", "jockey_key"),
    "horse_x_trainer": ("horse_id", "trainer_key"),
    "trainer_x_track": ("trainer_key", "venue"),
    "horse_x_track": ("horse_id", "venue"),
    "horse_x_course": ("horse_id", "venue", "surface", "distance_m"),
    "trainer_x_debut": ("trainer_key", "debut_flag"),
    "horse": ("horse_id",),
    "jockey": ("jockey_key",),
    "trainer": ("trainer_key",),
}
CELL_BINS = (0, 1, 5, 10, 50, 100, 500)  # n_prior bands: [0, 1), [1, 5), ... [500, inf)
RUNNER_COLUMNS = [
    "race_id",
    "race_date",
    "horse_id",
    "finish_pos",
    "is_scratched",
    "jockey_key",
    "trainer_key",
]
RACE_COLUMNS = [
    "race_id",
    "race_date",
    "race_no",
    "meeting_id",
    "race_class",
    "status",
    "venue",
    "distance_m",
    "surface",
    "post_time_local",
]
STAGE2 = {
    m: MetaSpec([m], quadratic=False, segment_cols=[], form="stage2", ridge=0.0) for m in MEMBERS
}


def _hypothesis(member: str) -> str:
    return HYPOTHESIS.get(member, "E93")


def option_id(member: str) -> str:
    """The leaderboard option of a member's tail: Win at the final price, its stage-2 meta's
    top claim band."""
    return f"{COUNTRY}:WIN:final:c2-interactions:{member}:stage2:claim-top-1pct"


# --------------------------------------------------------------------------- the spec


def build_spec(scheme, bootstrap: dict) -> dict:
    """The family's pre-registered spec, frozen in the registry before any outcome is read."""
    blocks = {
        member: [{"name": b.name, "cols": list(b.cols), "kind": b.kind} for b in blocks]
        for member, (blocks, _) in MEMBERS.items()
    }
    return {
        "family_id": FAMILY_ID,
        "supersedes": f"{SUPERSEDES} (spec sha256 {SUPERSEDED_SPEC}): its tail settled a winner "
        "at its final Win odds over the race's number of winners, not at the recorded dividend "
        "(a JRA dead heat pays each winner its own dividend, and a dividend paid before a later "
        "disqualification stands), and its records named one git revision for level-1 fits made "
        "at two; found in #93's review (round 1). Its level1, tau and folds sections are kept "
        "unchanged, so its saved level-1 fits are re-scored, not refitted (adopt_v1_checkpoint: "
        "a member-fold's revision is the stopped second launch's where its u equals that "
        "launch's saved u bit for bit, else the completing launch's)",
        "ticket": TICKET,
        "hypotheses": {
            "E35": "horse x jockey: tau_hj < 0.1 and its cells add nothing at the final price "
            "(08 §4 E35; the negative control; booking endogeneity makes any estimate "
            "predictive, never causal, and it is never a bet trigger)",
            "E36": "trainer x track (east/west travel): tau >= 0.05, learnable, with a small "
            "standalone gain that is probably priced (08 §4 E36)",
            "E93": "horse x track/course/distance in the structured form, the trainer debut "
            "prior and the transfer step (08 §3.3; §8 B5): each measured standalone and at the "
            "final price; horse x trainer is aliased with the horse where the trainer is "
            "constant and is not fitted",
        },
        "null": "no block carries anything the base or the final price lacks: tau = 0 and a "
        "paired ΔLL of 0 standalone and at the final price; the tail returns the takeout",
        "decision_rules": {
            "ranking": "nothing is adopted or rejected (D16 as amended, D19): each block member "
            "is a ranked member row by its tail return per ¥100 at the final price (the top "
            "claim band of its stage-2 meta), with D7's meeting bounds, beside its ΔLLs",
            "tau": "descriptive: tau's empirical-Bayes value per fold and its profile-evidence "
            "bounds (each one-sided 95%) on the last outer fold's fit; learnable reads tau's "
            "lower bound against 08 §3.3's 0.05",
            "standalone": "descriptive: the paired per-race winner ΔLL of the block member "
            "alone against the base member alone over the scored races (renormalised over the "
            "priced starters), meeting bootstrap, one-sided 95% bounds",
            "final price": "descriptive: the paired per-race winner ΔLL of the block member's "
            "stage-2 meta against the base member's stage-2 meta (the same market term, rows "
            "and folds), out of fold over the scored races, with the same bounds",
            "C4 marginal": "descriptive: each member's stage-2 meta against the market-only "
            "stage-2 meta, paired, the same bounds",
            "tail": "descriptive: realised Win return per unit stake (the store's recorded Win "
            "dividend for the runner's race and horse number over the unit stake, 0 without "
            "one) of every scored runner by band of its stage-2 claim p x O_final, quantile "
            f"bands at {list(TAIL_QUANTILES)}, D7's meeting bounds; the top band (claims from "
            "the 99th percentile) is the leaderboard row",
            "cells": "descriptive: n_prior (cell_sizes) per out-of-fold starter, by band and "
            "quantile, per interaction; n_half = 1 / (I tau^2) at 08 §3.3's I = "
            f"{INFO_PER_RUN} and at the base's own mean information per run on the last fold, "
            "and the share of the out-of-fold starters with a cell whose cell holds at least "
            "n_half prior runs (structured and low-rank: the row entity's prior runs)",
        },
        "level1": {
            "frame": f"every not-scratched starter of the JRA store's run races with a winner "
            f"from {RESULTS_FROM.isoformat()}; no price is read; race_utc is the race's "
            "post_time_local in Asia/Tokyo on its race_date (race_no minutes after midnight "
            "when it has none); a starter without a placing has no finish_pos",
            "base": "models.ability.AbilityModel(k=3, discounts=DISCOUNTS): u = beta * rating "
            "+ jockey_key + trainer_key effects, each N(0, tau_e^2), tau_e^2 by empirical "
            "Bayes; #92's model without the sire (D21: the JRA-VAN pedigree; lineage is #95)",
            "discounts": list(DISCOUNTS),
            "rating": "studies.rating: DynamicRating over every starter from 2010 (#91, its "
            "fixed parameters and grids), the member logp at the (drift, scale) #91's rule picks "
            "per outer fold (studies.ability.select_pairs: the lowest mean winner log loss of "
            f"the member alone over the fold's inner races from {INNER_FROM.isoformat()})",
            "rating_fixed": c2_rating.FIXED,
            "drift_grid": list(c2_rating.DRIFT_GRID),
            "scale_grid": list(c2_rating.SCALE_GRID),
            "members": blocks,
            "extra_features": {m: f for m, (_, f) in MEMBERS.items() if f},
            "fit": "models.interactions.fit_with_blocks: the penalised MAP with each prior "
            "group's tau^2 by empirical Bayes (MacKay's fixed point, models.ability's rule); "
            f"blocks of more than {ix.DENSE_MAX} parameters take block-Jacobi steps (exact MAP, "
            "block-diagonal posterior variances); a low-rank block is rank one with W centred "
            "at unit scale; z(d) standardised over the fold's training rows; each fold's fixed "
            "point and MAP start from the previous fold's fit (fit_with_blocks' start: its prior "
            "sds, coefficients on shared levels and low-rank W), the first fold's from 0 and "
            "TAU2_START (a resumed run from the saved prior sds only)",
            "debut": "a horse's first start in the level-1 frame (from 2010); debut_flag is "
            "'debut' on that run and missing otherwise (no cell)",
            "transfer": f"models.interactions.transfer_step(stable_runs={STABLE_RUNS}) over the "
            "level-1 frame, trainer_key as the trainer",
            "horse_x_trainer": "not fitted (aliased with the horse where the horse keeps one "
            "trainer, 08 §3.3; the base has no static horse term): a record with status failed "
            "and its cell counts; the trainer debut prior and the transfer step stand in",
        },
        "tau": {
            "point": "each fold's empirical-Bayes tau; structured: the prior sd of the term "
            "averaged over the training rows; low-rank: tau of U (W at unit scale)",
            "bounds": "the profile of the Laplace evidence over the block's prior scale (the "
            "other groups at their empirical-Bayes values), each bound one-sided at 95% "
            f"(chi-square(1)); on the last outer fold's fit only; level {ix.TAU_LEVEL}",
        },
        "cells": {
            "interactions": {k: list(v) for k, v in CELLS.items()},
            "bins": list(CELL_BINS),
            "rows": f"the level-1 starters from {LEVEL1_FROM.isoformat()} (counts from every "
            "earlier row of the frame); a missing key is no cell",
            "info_per_run": INFO_PER_RUN,
        },
        "level2": {
            "frame": "studies.meta.store_frame: the priced starters of run races with a priced "
            "winner, from the price era; q the normalised final Win price",
            "baseline": "D4 calibrated market (models.v0.calibration), in-fold on D9's folds",
            "member": "ln p of each level-1 member renormalised over the race's priced starters; "
            "a race with a starter lacking a member is dropped",
            "meta": "models.meta.fit_meta stage2 on [member] and ln q, ridge 0, unbounded, on "
            "each D9 fold's training rows that have every member; pooled in-sample over the "
            "scored rows",
            "market_only": "studies.ability.market_only_oof: u = beta * ln q on the same rows "
            "and folds",
            "dividends": "the store's dividends rows of pool WIN for the frame's races "
            "(development days only), joined by race_id and the runners' horse_no; each "
            "combination one horse number, at most one row per race and number",
            "tail": "claim = stage-2 p x win_odds (cleaned final decimal odds); return = the "
            "runner's WIN dividend / the source's unit stake (config jra.yaml), the stake on a "
            "refund row, 0 without a row; clusters meeting_id",
        },
        "provenance": "each level-1 member-fold keeps the git revision (HEAD) it was fitted at, "
        "in the checkpoint and in every later record's params.fit_revisions (member -> revision "
        "-> outer folds); the checkpoint is keyed by the level1, tau and folds sections and the "
        "runners and races builds, so a change elsewhere re-scores the saved fits; every failed "
        "member-fold fit, an earlier launch's included, is a failed record "
        "fit:<member>:<fold>@<revision>; a stopped or failed launch records every planned "
        "variant as failed",
        "folds": {
            "level1": "D9's quarters (studies.ability.level1_quarters) over the level-1 dates; "
            "each quarter trains on every earlier row to its embargo",
            "level1_out_of_fold_from": LEVEL1_FROM.isoformat(),
            "inner_from": INNER_FROM.isoformat(),
            "level2": dataclasses.asdict(scheme),
            "scored": "every level-2 D9 fold",
        },
        "bootstrap": bootstrap,
        "price_era_start": PRICE_ERA_START.isoformat(),
        "leaderboard": {m: option_id(m) for m in [*MEMBERS, MARKET_ONLY]},
        "export": "per level-1 out-of-fold runner (from 2015-Q1): each member's u, under the "
        "data dir",
        "adoption": "none (D16, D19): ranked rows only; no holdout is opened",
        "sources": list(SOURCES),
    }


# --------------------------------------------------------------------------- inputs


def read_level1() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The store's runners and races from RESULTS_FROM, column- and date-scoped, no price,
    development days only."""
    out = []
    for name, columns in (("runners", RUNNER_COLUMNS), ("races", RACE_COLUMNS)):
        table = read_store_table(
            name, columns=columns, start=RESULTS_FROM, end=date.max, touches_price=False
        )
        development, _ = split_off(table.to_pandas(date_as_object=False), "race_date", COUNTRY)
        out.append(development)
    return out[0], out[1]


def race_utc(races: pd.DataFrame) -> pd.Series:
    """Each race's post time in UTC: post_time_local (HH:MM, JST) on its race_date, else race_no
    minutes after the day's midnight (JST)."""
    day = pd.to_datetime(races["race_date"]).dt.tz_localize("Asia/Tokyo")
    hhmm = races["post_time_local"].astype("string")
    ok = hhmm.str.fullmatch(r"\d{1,2}:\d{2}").fillna(False).to_numpy(dtype=bool)
    parts = hhmm.where(ok, "0:00").str.split(":", expand=True).astype(int)
    post = day + pd.to_timedelta(parts[0] * 60 + parts[1], unit="min")
    fallback = day + pd.to_timedelta(pd.to_numeric(races["race_no"]).fillna(0), unit="min")
    return post.where(ok, fallback).dt.tz_convert("UTC")


def level1_frame(runners: pd.DataFrame, races: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Every not-scratched starter of a run race with a winner, sorted by race_utc and race,
    with the race's columns, race_utc, won, debut, debut_flag and transfer."""
    scratched = runners["is_scratched"].eq(True).to_numpy(dtype=bool, na_value=False)
    base = runners.loc[~scratched].drop(columns="is_scratched")
    base["finish_pos"] = pd.to_numeric(base["finish_pos"]).astype(float)
    run = races.loc[races["status"].eq("run")].copy()
    run["race_utc"] = race_utc(run)
    keep = ["race_id", "race_no", "race_utc", "meeting_id", "race_class", "venue"]
    keep += ["distance_m", "surface"]
    frame = base.merge(run[keep], on="race_id", how="inner", validate="many_to_one")
    frame["won"] = frame["finish_pos"].eq(1).to_numpy()
    winners = frame.groupby("race_id")["won"].transform("sum")
    frame = frame.loc[winners > 0]
    frame = frame.sort_values(["race_utc", "race_id"], kind="stable").reset_index(drop=True)
    frame["distance_m"] = pd.to_numeric(frame["distance_m"]).astype(float)
    first = frame.groupby("horse_id")["race_utc"].transform("min")
    frame["debut"] = frame["race_utc"].eq(first).astype(float)
    frame["debut_flag"] = pd.Series(np.where(frame["debut"] > 0, "debut", None), dtype=object)
    step = ix.transfer_step(
        frame.rename(columns={"trainer_key": "trainer_id"}), stable_runs=STABLE_RUNS
    )
    frame["transfer"] = step.astype(float).to_numpy()
    per_race = frame.groupby("race_id")["won"].sum()
    counts = {
        "level1_runners_read": len(runners),
        "level1_scratched": int(scratched.sum()),
        "level1_races": len(per_race),
        "level1_starters": len(frame),
        "level1_races_without_a_post_time": int(
            (~run["post_time_local"].astype("string").str.fullmatch(r"\d{1,2}:\d{2}"))
            .fillna(True)
            .sum()
        ),
        "level1_starters_without_a_placing": int(frame["finish_pos"].isna().sum()),
        "level1_dead_heat_races": int((per_race >= 2).sum()),
        "level1_debut_starters": int(frame["debut"].sum()),
        "level1_transfer_starters": int(frame["transfer"].sum()),
    }
    return frame, counts


# --------------------------------------------------------------------------- cells


def cell_tables(frame: pd.DataFrame, oof: np.ndarray) -> tuple[pd.DataFrame, dict[str, list]]:
    """Per interaction, n_prior per starter (all rows), and its distribution over the rows in
    `oof`: one table row per band and a quantile row."""
    keyed = frame[["race_id", "horse_id", "race_utc"]].copy()
    counts, rows = {}, {}
    for name, cols in CELLS.items():
        sub = keyed.assign(**{c: frame[c] for c in cols if c not in keyed})
        n = ix.cell_sizes(sub, block=cols)["n_prior"].to_numpy()
        has = frame[list(cols)].notna().all(axis=1).to_numpy()
        counts[name] = np.where(has, n, np.nan)  # NaN: the row is in no cell
        x = n[oof & has]
        table = []
        edges = [*CELL_BINS, np.inf]
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            share = float(((x >= lo) & (x < hi)).mean()) if len(x) else float("nan")
            table.append({"band": f"[{lo}, {hi})", "share": share})
        q = np.quantile(x, [0.25, 0.5, 0.75, 0.9]) if len(x) else [np.nan] * 4
        cells = frame.loc[has, list(cols)].drop_duplicates()
        rows[name] = {
            "starters": int(len(x)),
            "starters_without_a_cell": int((oof & ~has).sum()),
            "cells_in_frame": int(len(cells)),
            "mean": float(x.mean()) if len(x) else float("nan"),
            "p25": float(q[0]),
            "median": float(q[1]),
            "p75": float(q[2]),
            "p90": float(q[3]),
            "bands": table,
        }
    return pd.DataFrame(counts), rows


def horse_trainer_counts(frame: pd.DataFrame, oof: np.ndarray) -> dict:
    """The horse x trainer cells: horses, cells, horses with one trainer (whose cell is the
    horse) and the share of out-of-fold starters in such a horse's cell; and drop_aliased's
    answer on those horses' rows."""
    rows = frame.loc[frame[["horse_id", "trainer_key"]].notna().all(axis=1)]
    per = rows.groupby("horse_id")["trainer_key"].nunique()
    one = per.index[per.eq(1)]
    in_one = frame["horse_id"].isin(one).to_numpy()
    block = B("horse_x_trainer", ("horse_id", "trainer_key"), VC)
    constant = rows.loc[rows["horse_id"].isin(one)]
    return {
        "horses": int(len(per)),
        "cells": int(len(rows[["horse_id", "trainer_key"]].drop_duplicates())),
        "horses_with_one_trainer": int(len(one)),
        "share_of_horses_with_one_trainer": float(len(one) / max(len(per), 1)),
        "oof_starters": int(oof.sum()),
        "share_of_oof_starters_in_an_aliased_cell": float(in_one[oof].mean()),
        "aliased_on_constant_trainer_rows": bool(not ix.drop_aliased([block], constant)),
        "aliased_on_every_row": bool(not ix.drop_aliased([block], rows)),
    }


def info_per_run(u: np.ndarray, frame: pd.DataFrame) -> float:
    """The mean information per run of the discounted top-3 likelihood at u over frame's rows:
    the diagonal of the information in u (sum over a runner's stages of w p (1 - p))."""
    codes = pd.factorize(frame["race_id"].to_numpy(dtype=object))[0]
    st = _stages(codes, frame["finish_pos"].to_numpy(dtype=float), DISCOUNTS)
    p, _ = _softmax(np.asarray(u)[st.rows], st)
    dp = np.repeat(st.discount, st.sizes) * p
    return float(np.bincount(st.rows, weights=dp * (1.0 - p), minlength=len(frame)).mean())


# --------------------------------------------------------------------------- level 1


@dataclasses.dataclass
class Level1:
    """Each member's out-of-fold u (NaN off the outer folds), each fold's tau points and extra
    feature coefficients, the last fold's tau intervals, timings, each member-fold's fitting
    revision (`fits`: fold -> member -> git sha), every failed member-fold fit of this and earlier
    launches (`failures`), and the members still without a prediction on a fold (`errors`)."""

    u: dict[str, np.ndarray]
    taus: list[dict]
    intervals: dict[str, dict]
    info: float
    errors: dict[str, str]
    seconds: dict[str, float]
    local: dict[str, list[str]]
    prior_sd: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)
    fits: dict[str, dict[str, str]] = dataclasses.field(default_factory=dict)
    failures: list[dict] = dataclasses.field(default_factory=list)


def _json(value) -> str:
    """`value` as JSON, a numpy scalar as its Python value."""
    return json.dumps(value, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))


def _plain(value):
    """`value` as it reads back from JSON."""
    return json.loads(_json(value))


def _pairs(picks: list[dict]) -> list[tuple]:
    return [(p["fold"], p["drift_var_per_day"], p["scale"]) for p in picks]


def save_checkpoint(path: Path, out: Level1, done: set[str], picks: list[dict]) -> None:
    """The level-1 state as one .npz (each member's u and the metadata), written to a temporary
    file beside it and published by rename: a stop mid-save leaves the last complete one."""
    meta = {
        "taus": out.taus,
        "done": sorted(done),
        "seconds": out.seconds,
        "local": out.local,
        "prior_sd": out.prior_sd,
        "fits": out.fits,
        "failures": out.failures,
        "intervals": out.intervals,
        "info": out.info,
        "picks": _plain(picks),
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, **{META: np.array(_json(meta))}, **out.u)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_checkpoint(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """A checkpoint's metadata and each member's u."""
    with np.load(path) as arrays:
        meta = json.loads(arrays[META].item())
        return meta, {m: np.array(arrays[m]) for m in MEMBERS}


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """The fit's columns with ids as codes (a missing id stays missing), so copies are small."""

    def codes(col: str) -> np.ndarray:
        c = pd.factorize(frame[col].to_numpy(dtype=object))[0].astype(float)
        c[c < 0] = np.nan
        return c

    return pd.DataFrame(
        {
            "race_id": pd.factorize(frame["race_id"].to_numpy(dtype=object))[0],
            "horse_id": codes("horse_id"),
            "finish_pos": frame["finish_pos"].to_numpy(dtype=float),
            "jockey_key": codes("jockey_key"),
            "trainer_key": codes("trainer_key"),
            "venue": codes("venue"),
            "distance_m": frame["distance_m"].to_numpy(dtype=float),
            "surface": frame["surface"].to_numpy(dtype=object),
            "debut_flag": frame["debut_flag"].to_numpy(dtype=object),
            "debut": frame["debut"].to_numpy(dtype=float),
            "transfer": frame["transfer"].to_numpy(dtype=float),
        }
    )


def _base_model(member: str) -> AbilityModel:
    return AbilityModel(
        k=3, discounts=DISCOUNTS, entity_cols=ENTITIES, features=[RATING, *MEMBERS[member][1]]
    )


def level1_members(
    frame: pd.DataFrame,
    rating: dict,
    outer: list[Fold],
    picks: list[dict],
    *,
    revision: str,
    intervals: bool = True,
    checkpoint: Path | None = None,
    log=print,
) -> Level1:
    """Each member fitted on every outer fold's training rows and predicting its test rows, the
    fold's rating pair as the rating feature; tau intervals on the last fold's fits. A fit is
    stamped with `revision`; a checkpoint's fits, failures and intervals are carried forward."""
    n = len(frame)
    data = _numeric(frame)
    out = Level1(
        u={m: np.full(n, np.nan) for m in MEMBERS},
        taus=[],
        intervals={},
        info=float("nan"),
        errors={},
        seconds={m: 0.0 for m in MEMBERS},
        local={},
    )
    done, warm = set(), {}  # warm: each member's last fit, to start the next fold's from
    if checkpoint is not None and checkpoint.exists():
        saved, out.u = load_checkpoint(checkpoint)
        if _pairs(saved["picks"]) != _pairs(_plain(picks)):
            raise ValueError(f"{checkpoint.name} was fitted at other rating pairs than this run's")
        out.taus, done = saved["taus"], set(saved["done"])
        out.seconds, out.local, out.prior_sd = saved["seconds"], saved["local"], saved["prior_sd"]
        out.fits, out.failures = saved["fits"], saved["failures"]
        out.intervals, out.info = saved["intervals"], float(saved["info"])
        # a member-fold without a prediction (one that failed) is fitted again; its earlier
        # failures stay, each a record
        log(f"  resumed {len(done)} folds from {checkpoint.name}; {len(out.failures)} failed fits")
    started = time.monotonic()
    for i, (fold, pick) in enumerate(zip(outer, picks, strict=True)):
        label = fold.test_start.isoformat()
        last = i == len(outer) - 1
        todo = {m for m in MEMBERS if np.isnan(out.u[m][fold.test_idx]).any()}
        if last and intervals:  # the tau intervals are the last fold's fits'
            todo |= {
                m
                for m, (blocks, _) in MEMBERS.items()
                if any(b.name not in out.intervals for b in blocks)
            }
        if last and not np.isfinite(out.info):  # the base's information per run
            todo.add(BASE)
        if label in done and not todo:
            continue
        pair = (pick["drift_var_per_day"], pick["scale"])
        fold_data = data.assign(**{RATING: rating[pair].to_numpy()})
        train, test = fold_data.iloc[fold.train_idx], fold_data.iloc[fold.test_idx]
        row = next((dict(t) for t in out.taus if t["fold"] == label), {"fold": label})
        row |= {"train_rows": len(train), "test_rows": len(test)}
        for member, (blocks, _) in MEMBERS.items():
            if member not in todo:
                continue
            t0 = time.monotonic()
            try:
                if not blocks:
                    model = _base_model(member).fit(train)
                    u = model.predict(test)["u"].to_numpy()
                    for feat in MEMBERS[member][1]:
                        row[f"{member}:coef[{feat}]"] = float(model.coef_[feat])
                    if member == BASE and last:
                        out.info = info_per_run(model.predict(train)["u"].to_numpy(), train)
                else:
                    start = warm.get(member) or {"prior_sd": out.prior_sd.get(member, {})}
                    fit = ix.fit_with_blocks(train, _base_model(member), blocks, start=start)
                    out.prior_sd[member] = dict(fit.prior_sd_)
                    warm[member] = fit.warm_start()
                    u = fit.predict(test)["u"].to_numpy()
                    out.local[member] = [t.label for t in fit._model.terms if t.local]
                    for b in blocks:
                        row[f"{member}:tau[{b.name}]"] = fit.tau_point(b.name)
                    for feat in MEMBERS[member][1]:
                        row[f"{member}:coef[{feat}]"] = float(fit.coef_[feat])
                    if last and intervals:
                        for b in blocks:
                            t1 = time.monotonic()
                            iv = fit.tau(b.name)
                            out.intervals[b.name] = {
                                "member": member,
                                "fold": label,
                                "tau": iv.mean,
                                "lower": iv.lo,
                                "upper": iv.hi,
                                "levels": iv.n_clusters,
                                "seconds": round(time.monotonic() - t1),
                            }
                    del fit
                out.u[member][fold.test_idx] = u
                out.fits.setdefault(label, {})[member] = revision
            except Exception as exc:  # a record of its own (fit_failures), never dropped
                failure = {"fold": label, "member": member, "revision": revision}
                failure["error"] = f"{type(exc).__name__}: {exc}"
                if failure not in out.failures:
                    out.failures.append(failure)
            out.seconds[member] = out.seconds.get(member, 0.0) + time.monotonic() - t0
        out.taus = [t for t in out.taus if t["fold"] != label] + [row]
        done.add(label)
        log(f"  fold {label} ({i + 1}/{len(outer)}): {_progress(started)}")
        if checkpoint is not None:
            save_checkpoint(checkpoint, out, done, picks)
    for member in MEMBERS:  # each member's variants fail while a fold has no prediction
        gaps = [
            f.test_start.isoformat() for f in outer if np.isnan(out.u[member][f.test_idx]).any()
        ]
        if gaps:
            why = [
                f["error"] for f in out.failures if (f["member"], f["fold"]) == (member, gaps[0])
            ]
            out.errors[member] = f"{gaps[0]}: {why[-1] if why else 'no prediction'}"
    return out


def fit_revisions(fits: dict[str, dict[str, str]], order: list[str]) -> dict[str, dict]:
    """member -> fitting revision -> the outer folds (of `order`) it fitted, as runs of
    consecutive folds, 'first..last' (one fold as itself)."""
    out = {}
    for member in MEMBERS:
        by: dict[str, list[int]] = {}
        for i, label in enumerate(order):
            if (rev := fits.get(label, {}).get(member)) is not None:
                by.setdefault(rev, []).append(i)
        spans = {}
        for rev, pos in by.items():
            runs = [[pos[0], pos[0]]]
            for p in pos[1:]:
                if p == runs[-1][1] + 1:
                    runs[-1][1] = p
                else:
                    runs.append([p, p])
            spans[rev] = [order[a] if a == b else f"{order[a]}..{order[b]}" for a, b in runs]
        out[member] = spans
    return out


def fit_failures(failures: list[dict], fold_scheme: str) -> list[Variant]:
    """A failed variant per failed member-fold fit (`Level1.failures`), the same whichever launch
    records it."""
    return [
        Variant(
            _hypothesis(f["member"]),
            f"fit:{f['member']}:{f['fold']}@{f['revision'][:7]}",
            fold_scheme,
            {"model": "models/interactions.py", "study": "studies/interactions.py"}
            | {"member": f["member"]},
            {"discounts": list(DISCOUNTS), "stable_runs": STABLE_RUNS}
            | {"fold": f["fold"], "fit_revision": f["revision"]},
            error=f["error"],
        )
        for f in failures
    ]


# --------------------------------------------------------------------------- level 2


def read_win_dividends(race_ids) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The development starters' horse numbers from the price era (race_id, horse_id,
    horse_no) and the store's WIN dividend rows of `race_ids`, column-scoped."""
    table = read_store_table(
        "runners",
        columns=["race_id", "race_date", "horse_id", "horse_no"],
        start=PRICE_ERA_START,
        end=date.max,
        touches_price=False,
    )
    numbers, _ = split_off(table.to_pandas(date_as_object=False), "race_date", COUNTRY)
    scope = ds.field("pool").isin([WIN]) & ds.field("race_id").isin(list(race_ids))
    dataset = ds.dataset(paths.processed_dir(SOURCE) / "dividends.parquet", format="parquet")
    dividends = dataset.to_table(columns=DIVIDEND_COLUMNS, filter=scope).to_pandas()
    return numbers.drop(columns="race_date"), dividends


def win_paid(
    price: pd.DataFrame, numbers: pd.DataFrame, dividends: pd.DataFrame, unit: float
) -> tuple[np.ndarray, dict]:
    """Each priced starter's Win return per unit stake at the recorded dividends: its race's WIN
    row for its horse number pays dividend / unit (a refund row, the stake); no row pays 0.

    A race with no WIN row at all is unsettled, not a race every starter lost (the store
    publishes runners before dividends, so an interrupted refresh leaves such races): each of its
    starters pays NaN, never 0, and `races_without_a_win_dividend` counts them. Only a scored
    row's return is read (`_tail`, which fails on a NaN), so an unsettled race that only trains
    stops nothing."""
    keyed = price[["race_id", "horse_id"]].astype(object)
    numbers = numbers.astype({"race_id": object, "horse_id": object})
    rows = keyed.merge(numbers, on=["race_id", "horse_id"], how="left", validate="one_to_one")
    horse_no = pd.to_numeric(rows["horse_no"]).astype(float)
    if horse_no.isna().any():
        raise ValueError(f"{int(horse_no.isna().sum())} priced starters have no horse number")
    win = dividends.loc[dividends["pool"].eq(WIN)]
    refund = win["is_refund"].eq(True).to_numpy(dtype=bool, na_value=False)
    pays = np.where(refund, 1.0, pd.to_numeric(win["dividend"]).to_numpy(dtype=float) / unit)
    paying = pd.DataFrame(
        {
            "race_id": win["race_id"].astype(object).to_numpy(),
            "horse_no": pd.to_numeric(win["combination"], errors="coerce").astype(float),
            "pays": pays,
        }
    )
    if paying["horse_no"].isna().any() or not np.isfinite(pays).all():
        raise ValueError("a WIN dividend row is not one horse number with a finite dividend")
    races = pd.Series(price["race_id"].astype(object).unique())
    unsettled = races.loc[~races.isin(set(paying["race_id"]))]
    joined = pd.DataFrame({"race_id": rows["race_id"], "horse_no": horse_no}).merge(
        paying, on=["race_id", "horse_no"], how="left", validate="many_to_one"
    )
    settled = ~joined["race_id"].isin(set(unsettled)).to_numpy(dtype=bool)
    paid = np.where(settled, joined["pays"].fillna(0.0).to_numpy(dtype=float), np.nan)
    won = price["won"].to_numpy(dtype=bool)
    counts = {
        "win_dividend_rows": len(paying),
        "win_dividend_rows_without_a_priced_starter": int(
            len(paying) - joined["pays"].notna().sum()
        ),
        "races_without_a_win_dividend": len(unsettled),  # its starters pay NaN
        "paid_starters": int((paid > 0).sum()),
        "paid_starters_not_placed_first": int(((paid > 0) & ~won).sum()),
        "first_placed_starters_without_a_dividend": int((won & (paid == 0)).sum()),
        "starters_paid_otherwise_than_odds_over_winners": _settled_otherwise(price, paid),
    }
    return paid, counts


def _settled_otherwise(frame: pd.DataFrame, paid) -> int:
    """How many of the frame's starters `paid` settles otherwise than v1's rule did (a placed
    first at its final odds over the race's number of them, others 0); an unsettled starter
    (NaN) is not counted."""
    won = frame["won"].to_numpy(dtype=bool)
    m = frame.groupby("race_id")["won"].transform("sum").to_numpy(dtype=float)
    v1 = np.where(won, frame["win_odds"].to_numpy(dtype=float) / np.maximum(m, 1.0), 0.0)
    return int((np.abs(np.asarray(paid, dtype=float) - v1) > 1e-9).sum())


def member_frame(price: pd.DataFrame, level1: pd.DataFrame, u: dict, first) -> tuple:
    """The price frame's rows with every member's ln p renormalised over the race's priced
    starters; a race with a starter lacking a member is dropped (counted from `first`)."""
    keys = level1[["race_id", "horse_id"]].assign(**{m: u[m] for m in MEMBERS})
    joined = price.reset_index().merge(
        keys, on=["race_id", "horse_id"], how="left", validate="one_to_one"
    )
    has = joined[list(MEMBERS)].notna().all(axis=1)
    whole = has.groupby(joined["race_id"]).transform("all")
    rows = joined.loc[whole].set_index("index").rename_axis(None)
    race = rows["race_id"].to_numpy()
    for m in MEMBERS:
        v = rows[m].to_numpy()
        top = pd.Series(v).groupby(race).transform("max").to_numpy()
        lse = top + np.log(pd.Series(np.exp(v - top)).groupby(race).transform("sum").to_numpy())
        rows[m] = v - lse
    rows["ln_q"] = np.log(rows["q"].to_numpy())
    counts = {
        "level2_races_with_every_member": int(rows["race_id"].nunique()),
        "level2_races_dropped_for_a_missing_member": int(
            joined.loc[~whole & joined["race_date"].ge(pd.Timestamp(first)), "race_id"].nunique()
        ),
    }
    return rows, counts


def _tail(p: pd.Series, scored: Scored, spec: dict, name: str) -> tuple[list, dict]:
    """The tail curve over the scored rows, each settled at its recorded dividend (`win_paid`);
    a scored race with no WIN row (NaN paid) raises, so the variant is a failed record."""
    rows = scored.rows
    ret = rows["paid"].to_numpy(dtype=float)
    if np.isnan(ret).any():
        races = rows.loc[np.isnan(ret), "race_id"].drop_duplicates()
        raise ValueError(
            f"{len(races)} scored races have no WIN dividend row (unsettled, e.g. "
            f"{', '.join(map(str, races.head(3)))}): refresh the store's dividends"
        )
    odds = rows["win_odds"].to_numpy(dtype=float)
    bets = pd.DataFrame(
        {
            "claim": p.reindex(rows.index).to_numpy() * odds,
            "ret": ret,  # at the recorded dividend (win_paid)
            "meeting_id": rows["meeting_id"].to_numpy(),
        }
    )
    boot = spec["bootstrap"]
    curve = tail_curve(
        bets,
        claim_col="claim",
        return_col="ret",
        cluster_col="meeting_id",
        quantiles=TAIL_QUANTILES,
        n_boot=boot["n_boot"],
        level=boot["level"],
        seed=boot["seed"],
    )
    metrics, table = [], []
    for r in curve.itertuples(index=False):
        band = f"claim_q[{r.q_lo:g},{r.q_hi:g}]"
        if r.n_bets:
            metrics.append(
                _metric(
                    "all_folds",
                    f"return_per_100[{band}]",
                    100 * r.realised,
                    r.n_bets,
                    100 * r.lo,
                    100 * r.hi,
                )
            )
            metrics.append(_metric("all_folds", f"mean_claim[{band}]", r.mean_claim, r.n_bets))
        table.append(
            {"forecast": name, "band": band, "claims from": r.claim_lo, "to": r.claim_hi}
            | {"bets": r.n_bets, "meetings": r.n_clusters, "mean claim": r.mean_claim}
            | {"return per ¥100": 100 * r.realised, "lower": 100 * r.lo, "upper": 100 * r.hi}
        )
    return metrics, {"tail": table}


# --------------------------------------------------------------------------- the pure study


@dataclasses.dataclass
class Inputs:
    level1: pd.DataFrame
    results: pd.DataFrame
    price: pd.DataFrame
    counts: dict


def compute_study(
    inputs: Inputs,
    spec: dict,
    scheme,
    *,
    checkpoint: Path | None = None,
    revision: str | None = None,
    log=print,
) -> tuple[Study, pd.DataFrame]:
    """Every variant, and the per-runner export. Pure but for the optional fit checkpoint; the
    fits are stamped with `revision` (default: git HEAD)."""
    revision = revision or _git_sha()
    level1, price = inputs.level1, inputs.price
    study = Study(counts=dict(inputs.counts))
    fold_scheme = "walk_forward:" + json.dumps(spec["folds"], sort_keys=True)
    comp = {"model": "models/interactions.py", "study": "studies/interactions.py"}
    params = {"discounts": list(DISCOUNTS), "stable_runs": STABLE_RUNS}

    quarters = level1_quarters(level1["race_date"], scheme)
    outer = [q for q in quarters if q.test_start >= LEVEL1_FROM]
    if quarters[0].test_start != INNER_FROM or outer[0].test_start != LEVEL1_FROM:
        raise ValueError("the level-1 quarters do not start at the signed dates")
    study.counts |= {"level1_quarters": len(quarters), "level1_outer_folds": len(outer)}
    oof = np.zeros(len(level1), dtype=bool)
    for fold in outer:
        oof[fold.test_idx] = True

    # cells first (prior runs only; no outcome is read)
    n_prior, cells = cell_tables(level1, oof)
    ht = horse_trainer_counts(level1, oof)
    study.tables["cells"] = [
        {"interaction": k} | {c: v for c, v in row.items() if c != "bands"}
        for k, row in cells.items()
    ]
    study.tables["cell_bands"] = [
        {"interaction": k} | {b["band"]: b["share"] for b in row["bands"]}
        for k, row in cells.items()
    ]
    study.tables["horse_x_trainer"] = [ht]
    for name, row in cells.items():
        hyp = "E35" if name == "horse_x_jockey" else "E36" if name == "trainer_x_track" else "E93"

        def cell_metrics(row=row):
            n = row["starters"]
            metrics = [
                _metric("oof_starters", f"n_prior[{q}]", row[q], n)
                for q in ("mean", "p25", "median", "p75", "p90")
            ]
            metrics += [
                _metric("oof_starters", f"share_n_prior{b['band']}", b["share"], n)
                for b in row["bands"]
            ]
            metrics.append(_metric("frame", "cells", row["cells_in_frame"], row["cells_in_frame"]))
            return metrics, {}

        study.attempt(Variant(hyp, f"cells:{name}", fold_scheme, comp, params), cell_metrics)
    ht_metrics = (
        _metric("frame", "horses", ht["horses"], ht["horses"]),
        _metric("frame", "cells", ht["cells"], ht["cells"]),
        _metric("frame", "horses_with_one_trainer", ht["horses_with_one_trainer"], ht["horses"]),
        _metric(
            "oof_starters",
            "share_in_an_aliased_cell",
            ht["share_of_oof_starters_in_an_aliased_cell"],
            ht["oof_starters"],
        ),
    )
    study.variants.append(
        Variant(
            "E93",
            "horse_x_trainer:not_fitted",
            fold_scheme,
            comp | {"block": "horse_x_trainer", "kind": VC},
            params | {"reason": "aliased"},
            metrics=ht_metrics,
            error=(
                f"not fitted: aliased with the horse where the trainer is constant within horse "
                f"({ht['horses_with_one_trainer']} of {ht['horses']} horses, "
                f"{ht['cells']} cells); the trainer debut prior and the transfer step stand in"
            ),
        )
    )

    # level 1
    log(f"rating grid over {len(inputs.results)} starters")
    rating, _ = c2_rating.member_grid(inputs.results, level1)
    picks = select_pairs(level1, rating, outer)
    needed = {(p["drift_var_per_day"], p["scale"]) for p in picks}
    rating = {pair: rating[pair] for pair in needed}
    study.tables["selection"] = picks
    study.attempt(
        Variant("E93", "selection:rating_pair", fold_scheme, comp, params),
        lambda: (
            [
                _metric(p["fold"], f"selected[{k}]", p[k], 1)
                for p in picks
                for k in ("drift_var_per_day", "scale")
            ],
            {},
        ),
    )
    log(f"level-1 fits: {len(outer)} folds x {len(MEMBERS)} members, pairs {sorted(needed)}")
    l1 = level1_members(
        level1, rating, outer, picks, revision=revision, checkpoint=checkpoint, log=log
    )
    del rating
    revisions = fit_revisions(l1.fits, [f.test_start.isoformat() for f in outer])
    params = params | {"fit_revisions": revisions}  # every record from here on names them
    study.tables["fit_revisions"] = [
        {"member": m, "revision": rev, "folds": ", ".join(spans)}
        | {"n": sum(by_member.get(m) == rev for by_member in l1.fits.values())}
        for m, by in revisions.items()
        for rev, spans in by.items()
    ]
    study.counts["level1_fit_revisions"] = len({rev for by in revisions.values() for rev in by})
    study.counts["level1_failed_fits"] = len(l1.failures)
    study.variants += fit_failures(l1.failures, fold_scheme)
    study.counts |= {f"level1_seconds[{m}]": round(s) for m, s in l1.seconds.items()}
    study.tables["tau_by_fold"] = l1.taus
    study.tables["tau"] = []
    for member, (blocks, feats) in MEMBERS.items():
        if member == BASE:
            continue

        def tau_metrics(member=member, blocks=blocks, feats=feats):
            if member in l1.errors:
                raise RuntimeError(l1.errors[member])
            metrics = []
            for row in l1.taus:
                for b in blocks:
                    metrics.append(
                        _metric(
                            row["fold"],
                            f"tau[{b.name}]",
                            row[f"{member}:tau[{b.name}]"],
                            row["train_rows"],
                        )
                    )
                for f in feats:
                    metrics.append(
                        _metric(
                            row["fold"], f"coef[{f}]", row[f"{member}:coef[{f}]"], row["train_rows"]
                        )
                    )
            table = []
            for b in blocks:
                iv = l1.intervals.get(b.name)
                if iv is None:
                    raise RuntimeError(f"no tau interval for {b.name}")
                metrics.append(
                    _metric(
                        iv["fold"],
                        f"tau[{b.name}]",
                        iv["tau"],
                        iv["levels"],
                        iv["lower"],
                        iv["upper"],
                    )
                )
                cell = _cell_of(b)
                n = n_prior[cell].to_numpy()[oof] if cell else None
                n = None if n is None else n[~np.isnan(n)]  # the starters with a cell
                row = {"block": b.name, "member": member, "kind": b.kind} | iv
                for label, info in (("08", INFO_PER_RUN), ("measured", l1.info)):
                    if iv["tau"] > 0 and np.isfinite(info) and info > 0:
                        nh = ix.n_half(iv["tau"], info)
                        row[f"n_half ({label} I)"] = nh
                        metrics.append(_metric(iv["fold"], f"n_half[{label}][{b.name}]", nh, 1))
                        if n is not None:
                            share = float((n >= nh).mean())
                            row[f"share at n_half ({label} I)"] = share
                            metrics.append(
                                _metric(
                                    "oof_starters",
                                    f"share_at_n_half[{label}][{b.name}]",
                                    share,
                                    len(n),
                                )
                            )
                table.append(row)
            return metrics, {"tau": table}

        study.attempt(
            Variant(
                _hypothesis(member),
                f"tau:{member}",
                fold_scheme,
                comp | {"member": member},
                params | {"info_per_run_measured": l1.info},
            ),
            tau_metrics,
        )
    study.counts["level1_info_per_run_measured_last_fold"] = l1.info
    export = level1_export(level1, l1.u, outer)

    # level 2
    folds2 = walk_forward(price["race_date"], scheme=scheme)
    log(f"D4 baseline on {len(folds2)} level-2 folds")
    price = price.assign(p_cal=baseline_oof(price, folds2))
    ok = {m: np.where(np.isnan(v), 0.0, v) if m in l1.errors else v for m, v in l1.u.items()}
    rows, member_counts = member_frame(price, level1, ok, outer[0].test_start)
    study.counts |= member_counts
    rfolds = restrict(folds2, rows.index.to_numpy(), len(price))
    rows = rows.reset_index(drop=True)
    scored = Scored.of(rows, rfolds)
    study.counts |= {"scored_folds": len(rfolds), "scored_races": scored.n_races}
    sr = scored.rows
    study.counts |= {  # an unsettled scored race (NaN paid) fails each tail variant (_tail)
        "scored_races_without_a_win_dividend": int(sr.loc[sr["paid"].isna(), "race_id"].nunique()),
        "scored_paid_starters_not_placed_first": int((sr["paid"].gt(0) & ~sr["won"]).sum()),
        "scored_starters_paid_otherwise_than_odds_over_winners": _settled_otherwise(sr, sr["paid"]),
    }
    base = comp | {"score": "winner_log_loss", "baseline": "calibrated_market"}
    meta_params = params | {"ridge": 0.0, "bounds": "none", "quadratic": False, "segments": []}

    def need(*members):
        for m in members:
            if m in l1.errors:
                raise RuntimeError(f"the level-1 member {m} failed: {l1.errors[m]}")

    p = {}
    for member in MEMBERS:
        outcome = _try(lambda m=member: (need(m), _meta_oof(rows, rfolds, STAGE2[m], "ln_q"))[1])
        pooled = _try(lambda m=member: _pooled(scored, STAGE2[m], "ln_q"))
        name = f"stage2:q:{member}"
        components = base | {"member": member, "form": "stage2", "market": "ln_q"}
        study.attempt(
            Variant(_hypothesis(member), name, fold_scheme, components, meta_params),
            lambda o=outcome, pl=pooled, nm=name: _meta_metrics(o, pl, rfolds, scored, spec, nm),
        )
        study.attempt(
            Variant(
                _hypothesis(member),
                f"{CLUSTERED}:{name}",
                fold_scheme,
                components | {"se": CLUSTERED},
                meta_params,
            ),
            lambda o=outcome, pl=pooled: _clustered_metrics(o, pl, rfolds, spec),
        )
        if outcome.error is None:
            p[member] = outcome.value[0]
    market = _try(lambda: _market_only(rows, rfolds, "ln_q"))
    study.attempt(
        Variant(
            "E93",
            f"stage2:q:{MARKET_ONLY}",
            fold_scheme,
            base | {"member": "none", "form": "stage2", "market": "ln_q"},
            meta_params,
        ),
        lambda: _meta_metrics(market, None, rfolds, scored, spec, f"stage2:q:{MARKET_ONLY}"),
    )
    if market.error is None:
        p[MARKET_ONLY] = market.value[0]

    def have(*names):
        missing = [n for n in names if n not in p]
        if missing:
            raise RuntimeError(f"the stage-2 meta of {missing} failed")

    paired = base | {"score": "paired_winner_log_loss"}
    r = scored.rows
    for member in MEMBERS:
        hyp = _hypothesis(member)
        if member != BASE:

            def standalone(m=member):
                need(m, BASE)
                return _comparison(
                    f"{m}: member alone vs base alone", np.exp(r[m]), np.exp(r[BASE]), scored, spec
                )

            def final(m=member):
                have(m, BASE)
                return _comparison(
                    f"{m}: stage-2 vs base stage-2 (at the final price)",
                    p[m],
                    p[BASE],
                    scored,
                    spec,
                )

            study.attempt(
                Variant(
                    hyp,
                    f"level1:{member}_vs_{BASE}",
                    fold_scheme,
                    paired | {"comparator": BASE},
                    params,
                ),
                standalone,
            )
            study.attempt(
                Variant(
                    hyp,
                    f"marginal:stage2:{member}_vs_{BASE}",
                    fold_scheme,
                    paired | {"comparator": f"stage2:q:{BASE}"},
                    params,
                ),
                final,
            )

        def c4(m=member):
            have(m, MARKET_ONLY)
            return _comparison(
                f"{m}: stage-2 vs stage-2 market only", p[m], p[MARKET_ONLY], scored, spec
            )

        study.attempt(
            Variant(
                hyp,
                f"marginal:stage2:{member}_vs_{MARKET_ONLY}",
                fold_scheme,
                paired | {"comparator": f"stage2:q:{MARKET_ONLY}"},
                params,
            ),
            c4,
        )
    tail_params = params | {"quantiles": list(TAIL_QUANTILES), "stake": "flat, unit"}
    for member in [*MEMBERS, MARKET_ONLY]:

        def tail(m=member):
            have(m)
            return _tail(p[m], scored, spec, f"stage2:q:{m}")

        study.attempt(
            Variant(
                _hypothesis(member) if member in MEMBERS else "E93",
                f"tail:stage2:{member}",
                fold_scheme,
                base | {"score": "tail_return", "option": option_id(member)},
                tail_params,
            ),
            tail,
        )
    study.attempt(
        Variant(
            "E93",
            "tail:calibrated_market",
            fold_scheme,
            base | {"score": "tail_return", "option": "calibrated_market"},
            tail_params,
        ),
        lambda: _tail(r["p_cal"], scored, spec, "calibrated market"),
    )
    return study, export


def _cell_of(block: ix.InteractionBlock) -> str | None:
    """The cell-size table column of a block's levels."""
    if block.kind == VC:
        for name, cols in CELLS.items():
            if tuple(cols) == block.cols or (
                block.cols == ("trainer_key", "debut_flag") and name == "trainer_x_debut"
            ):
                return name
        return None
    return {"horse_id": "horse", "trainer_key": "trainer"}.get(block.cols[0])


def level1_export(level1: pd.DataFrame, u: dict, outer: list[Fold]) -> pd.DataFrame:
    """Each member's u per level-1 out-of-fold runner: race_id, horse_id, race_date, oof_fold."""
    fold_of = np.full(len(level1), None, dtype=object)
    for fold in outer:
        fold_of[fold.test_idx] = fold.test_start.isoformat()
    keep = fold_of != None  # noqa: E711
    out = level1.loc[keep, ["race_id", "horse_id", "race_date"]].reset_index(drop=True)
    out["oof_fold"] = pd.Series(fold_of[keep], dtype="string")
    for m, v in u.items():
        out[f"u_{m}"] = v[keep]
    return out


# --------------------------------------------------------------------------- the run


def _progress(started: float) -> str:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    return f"{time.monotonic() - started:.0f} s, peak RSS {rss:.2f} GB"


def load_inputs(log=print) -> Inputs:
    runners, races = read_level1()
    level1, counts = level1_frame(runners, races)
    del runners, races
    results = c2_rating.read_results()
    p_runners, p_races = read_development()
    price, price_counts = store_frame(p_runners, p_races)
    del p_runners, p_races
    numbers, dividends = read_win_dividends(price["race_id"].unique())
    paid, paid_counts = win_paid(
        price, numbers, dividends, float(load_source(COUNTRY, SOURCE).unit_stake)
    )
    price["paid"] = paid
    del numbers, dividends
    counts |= {f"level2_{k}": v for k, v in (price_counts | paid_counts).items()}
    log(f"read {len(level1)} level-1 starters, {len(price)} priced starters")
    return Inputs(level1=level1, results=results, price=price, counts=counts)


SECTIONS = (
    ("fit_revisions", "Level-1 fitting revision (git HEAD at the fit) per member and outer fold"),
    ("cells", "Cell sizes: prior runs per out-of-fold starter's cell (from 2015-Q1)"),
    ("cell_bands", "Cell sizes by band of n_prior (share of out-of-fold starters)"),
    ("horse_x_trainer", "Horse x trainer (not fitted: aliased where the trainer is constant)"),
    ("selection", "#91's rating pair per outer fold"),
    ("tau", "tau per block on the last outer fold (profile-evidence bounds, one-sided 95% each)"),
    ("tau_by_fold", "tau and extra-feature coefficients per outer fold (empirical Bayes)"),
    ("delta", "Stage-2 ΔLL against the D4 calibrated market (out of fold; meeting bootstrap)"),
    ("comparisons", "Paired ΔLL (positive: the first beats the second; meeting bootstrap)"),
    ("tail", "Tail return by band of stage-2 claim (per ¥100, flat, at the recorded dividends)"),
    ("coefficients", "Stage 2 per fold (unbounded; Wald 95% two-sided and meeting one-sided 95%)"),
    ("pooled", "Stage-2 alpha pooled over the scored rows (in-sample)"),
)


def render_report(study: Study, spec: dict, about: dict, record_ids: dict[str, str]) -> str:
    """The full report: aggregate tables only, never a runner or race row."""
    lines = [f"# C2 interactions at the final Win price ({COUNTRY}, #{TICKET})", ""]
    lines += [*(f"- {k}: {v}" for k, v in about.items()), ""]
    lines += ["## Decision rules (frozen)", ""]
    lines += [*(f"- {k}: {v}" for k, v in spec["decision_rules"].items())]
    lines += [
        "",
        "## Data",
        "",
        _markdown_table([{"count": k, "n": v} for k, v in study.counts.items()]),
    ]
    for key, title in SECTIONS:
        rows = study.tables.get(key)
        if key == "pooled" and rows:
            rows = [{("alpha" if k == "delta" else k): v for k, v in r.items()} for r in rows]
        lines += ["", f"## {title}", "", _markdown_table(rows) if rows else "_Not computed._"]
    variants = [
        {"E": v.hypothesis_id, "variant": v.name, "status": "failed" if v.error else "scored"}
        | {"record": record_ids.get(v.name, "")[:16]}
        | ({"error": v.error} if v.error else {})
        for v in study.variants
    ]
    return "\n".join(
        [*lines, "", "## Variants (one registry record each)", "", _markdown_table(variants), ""]
    )


def artifact_stem(spec_hash: str, builds: dict, trial: bool, results: str) -> str:
    """A launch's artifacts (the report, its JSON, the export): the spec, the data builds and
    `results`, the digest of what the launch computed (`results_digest`). A launch that changes
    a result (a retry recovering a failed fold) names new files; one that reproduces every
    result names the same ones."""
    digest = hashlib.sha256(
        json.dumps(dict(builds), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    stem = f"interactions-{COUNTRY}-{spec_hash[:12]}-{digest[:12]}-{results[:12]}"
    return stem + ("-trial" if trial else "")


def results_digest(records: list[RunRecord], study: Study, export: pd.DataFrame) -> str:
    """The sha256 of a launch's results: its records but for git_sha and the artifacts they
    cite (`_identity`), the report's counts and tables, and the export's columns and rows."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(_identity(record).encode())
    shown = {"counts": study.counts, "tables": study.tables}
    digest.update(json.dumps(shown, separators=(",", ":"), default=str).encode())
    digest.update(json.dumps([str(c) for c in export.columns]).encode())
    digest.update(pd.util.hash_pandas_object(export, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _publish_parquet(frame: pd.DataFrame, path: Path) -> None:
    """`frame` as a parquet at `path`, written beside it and published by rename, so a stop
    mid-write never leaves a partial file under a results digest."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def level1_key(spec: dict, builds: dict) -> str:
    """The level-1 checkpoint's key: the spec's level1, tau and folds sections and the store
    builds level 1 reads. A change to either starts the fits afresh; a change elsewhere in the
    spec (level 2, the tail) re-scores the saved fits."""
    part = {"spec": {k: spec[k] for k in LEVEL1_SECTIONS}, "builds": dict(builds)}
    text = json.dumps(part, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def checkpoint_path(spec: dict, builds: dict, trial: bool) -> Path:
    key = level1_key(spec, builds)
    name = f"interactions-{COUNTRY}-level1-{key}" + ("-trial" if trial else "")
    return paths.data_dir() / "features" / COUNTRY / f"{name}.npz"


def _file_build(name: str) -> str:
    """A store table's build, '<path>@<sha256 prefix>' (`store_builds`' form)."""
    digest = hashlib.sha256()
    with open(paths.processed_dir(SOURCE) / f"{name}.parquet", "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return f"processed/{SOURCE}/{name}.parquet@{digest.hexdigest()[:16]}"


def _record(variant, git_sha, builds, spec_hash, trial, *, stem: str | None = None) -> RunRecord:
    """The variant's registry record; with `stem`, it cites the artifacts its launch wrote."""
    params = dict(variant.params) | {"spec_sha256": spec_hash}
    if trial:
        params["trial"] = True
    if variant.error:
        params["error"] = variant.error
    data_builds = dict(builds)
    if stem is not None:
        data_builds[ARTIFACTS] = f"reports+features/{COUNTRY}@{stem}"
    return RunRecord(
        hypothesis_id=variant.hypothesis_id,
        family_id=TRIAL_FAMILY_ID if trial else FAMILY_ID,
        variant=variant.name,
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=data_builds,
        sources=SOURCES,
        fold_scheme=variant.fold_scheme,
        components=variant.components,
        params=params,
        metrics=variant.metrics,
        status="failed" if variant.error else "scored",
    )


def _identity(record: RunRecord) -> str:
    """A record's result: its run id but for git_sha and the artifacts it cites."""
    builds = {k: v for k, v in record.data_builds.items() if k != ARTIFACTS}
    return _run_id(dataclasses.replace(record, git_sha="", data_builds=builds))


def append_new(registry, records: list[RunRecord]) -> tuple[list[str], list[str]]:
    """Append each record its family does not hold yet (`_identity`: a held record equal but
    for git_sha and the artifacts it cites keeps citing its own launch's, which hold the same
    result); each record's run id (the held one's when held), and the variants appended."""
    held = {}
    for family in {r.family_id for r in records}:
        held |= {_identity(r): r for r in registry.records(family)}
    ids, appended = [], []
    for record in records:
        key = _identity(record)
        if key in held:
            ids.append(_run_id(held[key]))
            continue
        ids.append(registry.append(record))
        held[key] = record
        appended.append(record.variant)
    return ids, appended


class _Families:
    """The registry restricted to some families, for `write_summary`."""

    def __init__(self, registry, families: set[str]) -> None:
        self._registry, self._families = registry, families

    def records(self, family_id: str | None = None):
        return [r for r in self._registry.records(family_id) if r.family_id in self._families]


_INDEX_ENTRY = re.compile(r"- \[([^\]]+)\]\(\1\.md\)")


PUBLIC_FILE_LIMIT = 256 * 1024  # scripts/check_public_tree.py's `size` rule (#16)


def write_committed_summary(registry, log=print) -> list[str]:
    """This study's family files by `write_summary`, and the committed index with their lines
    added (`studies.ability.write_committed_summary`'s rule). A family file over the public
    tree's size cap is written under the data dir's reports instead, unchanged, and gets no
    index line (the index lists exactly the committed files) until `write_summary` can split
    it (#330). Returns the families kept out of the tree."""
    directory = summary_dir(COUNTRY)
    mine = _Families(registry, {FAMILY_ID, TRIAL_FAMILY_ID, SUPERSEDES})
    if not mine.records():
        return []
    held = []
    with tempfile.TemporaryDirectory() as scratch:
        write_summary(mine, scratch)
        for path in sorted(Path(scratch).glob("*.md")):
            if path.stem == "README":
                continue
            text = path.read_text("utf-8")
            if len(text.encode("utf-8")) > PUBLIC_FILE_LIMIT:
                held.append(path.stem)
                out = paths.reports_dir() / f"registry-summary-{path.name}"
                write_report(out, text, sources=SOURCES)
                log(f"{path.stem}: summary over the public tree's size cap, written to {out}")
                (directory / path.name).unlink(missing_ok=True)
                continue
            write_report(directory / path.name, text, sources=())
        fresh = [
            line
            for line in Path(scratch, "README.md").read_text("utf-8").splitlines()
            if not ((m := _INDEX_ENTRY.fullmatch(line)) and m[1] in held)
        ]
    index = directory / "README.md"
    committed = index.read_text("utf-8").splitlines() if index.exists() else []
    entries = {m[1]: line for line in committed + fresh if (m := _INDEX_ENTRY.fullmatch(line))}
    head = [line for line in fresh if not _INDEX_ENTRY.fullmatch(line)]
    lines = [*head, *(entries[family] for family in sorted(entries) if family not in held)]
    write_report(index, "\n".join(lines) + "\n", sources=())
    return held


def planned() -> list[tuple[str, str]]:
    """Every variant a completed run records, for the failure records of an aborted run."""
    out = [
        (
            "E35" if n == "horse_x_jockey" else "E36" if n == "trainer_x_track" else "E93",
            f"cells:{n}",
        )
        for n in CELLS
    ]
    out += [("E93", "horse_x_trainer:not_fitted"), ("E93", "selection:rating_pair")]
    out += [(_hypothesis(m), f"tau:{m}") for m in MEMBERS if m != BASE]
    for m in MEMBERS:
        out += [(_hypothesis(m), f"stage2:q:{m}"), (_hypothesis(m), f"{CLUSTERED}:stage2:q:{m}")]
    out.append(("E93", f"stage2:q:{MARKET_ONLY}"))
    for m in MEMBERS:
        if m != BASE:
            out += [
                (_hypothesis(m), f"level1:{m}_vs_{BASE}"),
                (_hypothesis(m), f"marginal:stage2:{m}_vs_{BASE}"),
            ]
        out.append((_hypothesis(m), f"marginal:stage2:{m}_vs_{MARKET_ONLY}"))
    out += [(_hypothesis(m), f"tail:stage2:{m}") for m in MEMBERS]
    out += [("E93", f"tail:stage2:{MARKET_ONLY}"), ("E93", "tail:calibrated_market")]
    return out


def run_study(*, trial: bool = False, out_dir: Path | None = None, log=print) -> dict:
    """Run the pre-registered family on Japan's development data; write the report, the export
    and the registry records."""
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    scheme = load_country_scheme(COUNTRY)
    spec = build_spec(scheme, {**bootstrap_settings(), "seed": BOOTSTRAP_SEED})
    spec_hash = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    registry = default_registry(COUNTRY)
    if not trial:  # before any outcome is read
        spec_hash = registry.freeze_family(FAMILY_ID, spec)
    family = TRIAL_FAMILY_ID if trial else FAMILY_ID
    git_sha = _git_sha()
    level1_builds = store_builds()
    builds = level1_builds | {f"{SOURCE}_dividends": _file_build("dividends")}
    checkpoint = checkpoint_path(spec, level1_builds, trial)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    fold_scheme = "walk_forward:" + json.dumps(spec["folds"], sort_keys=True)
    started = time.monotonic()
    try:
        inputs = load_inputs(log=log)
        study, export = compute_study(
            inputs, spec, scheme, checkpoint=checkpoint, revision=git_sha, log=log
        )
        bare = [_record(v, git_sha, builds, spec_hash, trial) for v in study.variants]
        stem = artifact_stem(spec_hash, builds, trial, results_digest(bare, study, export))
    except (Exception, KeyboardInterrupt) as exc:  # a launch stopped by hand is a run too
        stopped = datetime.now(timezone.utc).isoformat(timespec="seconds")
        failed = [
            Variant(
                hypothesis,
                name,
                fold_scheme,
                {"study": "studies/interactions.py"},
                {"aborted_utc": stopped},
                error=f"{type(exc).__name__}: {exc}",
            )
            for hypothesis, name in planned()
        ]
        records = [_record(v, git_sha, builds, spec_hash, trial) for v in failed]
        append_new(registry, records)
        if checkpoint.exists():  # its failed fits, in case no launch resumes it
            saved, _ = load_checkpoint(checkpoint)
            fits = fit_failures(saved["failures"], fold_scheme)
            append_new(registry, [_record(v, git_sha, builds, spec_hash, trial) for v in fits])
        write_committed_summary(registry)
        raise
    log(f"computed ({_progress(started)})")

    report_path, json_path = out_dir / f"{stem}.md", out_dir / f"{stem}.json"
    export_path = paths.data_dir() / "features" / COUNTRY / f"{stem}.parquet"
    kept = {p for p in (report_path, json_path, export_path) if p.exists()}
    for path in sorted(kept):  # an earlier launch with these results wrote it: never replaced
        log(f"{path.name} exists (the same results digest): kept, not rewritten")
    records = []
    for v in study.variants:  # a failed fit's record cites no artifact (as an aborted run's)
        cites = None if v.name.startswith("fit:") else stem
        records.append(_record(v, git_sha, builds, spec_hash, trial, stem=cites))
    ids, appended = append_new(registry, records)
    record_ids = {r.variant: run_id for r, run_id in zip(records, ids, strict=True)}
    by_revision: dict[str, int] = {}
    for row in study.tables["fit_revisions"]:
        by_revision[row["revision"]] = by_revision.get(row["revision"], 0) + row["n"]
    fits = "level-1 fit revisions" + (" (a mixed run)" if len(by_revision) > 1 else "")
    about = {
        "family": f"`{family}`, spec sha256 `{spec_hash}`",
        "supersedes": f"`{SUPERSEDES}`, spec sha256 `{SUPERSEDED_SPEC}`",
        "git": git_sha,
        fits: ", ".join(f"`{r[:7]}` ({n} member-folds)" for r, n in sorted(by_revision.items())),
        "level-1 checkpoint": f"`{checkpoint.name}`",
        "data": ", ".join(builds.values()),
        "artifacts (report, JSON, export; the spec, the builds, the results' digest)": f"`{stem}`",
        "records": f"{len(appended)} appended; {len(record_ids) - len(appended)} already held",
        "trial (records marked params.trial, spec not frozen)": trial,
        "run time": _progress(started),
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if report_path not in kept:
        write_report(report_path, render_report(study, spec, about, record_ids), sources=SOURCES)
    if json_path not in kept:
        shown = {"about": about, "counts": study.counts, "tables": study.tables}
        write_report(json_path, json.dumps(shown, indent=1, default=str), sources=SOURCES)
    if export_path not in kept:
        _publish_parquet(export, export_path)
    write_committed_summary(registry)
    return {
        "family_id": family,
        "spec_hash": spec_hash,
        "artifact_stem": stem,
        "report_path": str(report_path),
        "export_path": str(export_path),
        "registry_path": str(registry.path),
        "records_appended": len(appended),
        "failed_variants": [v.name for v in study.variants if v.error],
    }


# --------------------------------------------------------------------------- v1's launches
# Run once by #93's review (round 1), kept as the code behind the records and the checkpoint
# they wrote: `uv run python -c "from horsetrack.countries.japan.studies import interactions as
# s; s.adopt_v1_checkpoint(); s.record_v1_launches()"`.

V1_STEM = "interactions-japan-3060dfbf9626-faff16a8cb38"  # v1's artifacts and checkpoints
V1_STOPPED = "56d0669b82969ffe6c96d06ae8300c389ba810b9"  # the second launch, stopped by hand
V1_COMPLETING = "b5a726cc8ae34f8b349e8bfa2be01cb57abb8d13"  # the third, resumed the second's
V1_LAUNCHES = {  # the stopped launches' errors; neither wrote a record
    "022d4fc8281f479e5b1418d890a98a06a8914046": "KeyboardInterrupt: stopped by hand after 5 of "
    "48 level-1 folds were saved (2015-01-01..2016-01-01): trainer_x_track's dense fit "
    "alternated U and W; the next launch fitted every fold afresh",
    V1_STOPPED: "KeyboardInterrupt: stopped by hand after 40 of 48 level-1 folds were saved "
    "(2015-01-01..2024-10-01): horse_aptitude's empirical Bayes did not converge from "
    "2024-04-01; the next launch resumed its checkpoint",
}
V1_NOTE = "appended 2026-11-28 by #93's review (round 1); the launch wrote no record"


def adopt_v1_checkpoint(log=print) -> Path:
    """v1's completed level-1 checkpoint (an .npz and a .json, no revisions) as this spec's. A
    member-fold whose u equals, bit for bit, the stopped second launch's saved u was fitted at
    V1_STOPPED; every other at V1_COMPLETING, which resumed that checkpoint and fitted only the
    member-folds without a prediction and the last fold. The second launch's failed fits are its
    saved folds without a prediction (it kept each member's first error only); the last fold's
    tau intervals, the base's information per run and the rating pairs are v1's report's."""
    features = paths.data_dir() / "features" / COUNTRY
    scheme = load_country_scheme(COUNTRY)
    spec = build_spec(scheme, {**bootstrap_settings(), "seed": BOOTSTRAP_SEED})
    target = checkpoint_path(spec, store_builds(), trial=False)
    if target.exists():
        raise FileExistsError(f"{target} exists")
    final = json.loads((features / f"{V1_STEM}-level1.json").read_text("utf-8"))
    stopped = json.loads((features / f"{V1_STEM}-level1-at-56d0669.json").read_text("utf-8"))
    report = json.loads((paths.reports_dir() / f"{V1_STEM}.json").read_text("utf-8"))
    u = {}
    for name in ("", "-at-56d0669"):
        with np.load(features / f"{V1_STEM}-level1{name}.npz") as arrays:
            u[name] = {m: np.array(arrays[m]) for m in MEMBERS}
    level1, _ = level1_frame(*read_level1())
    outer = [q for q in level1_quarters(level1["race_date"], scheme) if q.test_start >= LEVEL1_FROM]
    labels = [f.test_start.isoformat() for f in outer]
    if len(level1) != len(u[""][BASE]) or sorted(final["done"]) != labels:
        raise ValueError("v1's checkpoint does not match the level-1 frame and its folds")
    out = Level1(
        u=u[""],
        taus=final["taus"],
        intervals={},
        info=float(report["counts"]["level1_info_per_run_measured_last_fold"]),
        errors={},
        seconds=final["seconds"],
        local=final["local"],
        prior_sd=final["prior_sd"],
    )
    for fold, label in zip(outer, labels, strict=True):
        for m in MEMBERS:
            now, then = u[""][m][fold.test_idx], u["-at-56d0669"][m][fold.test_idx]
            carried = not np.isnan(then).any() and np.array_equal(now, then)
            out.fits.setdefault(label, {})[m] = V1_STOPPED if carried else V1_COMPLETING
            if label in stopped["done"] and np.isnan(then).any():
                kept = stopped["errors"].get(m, "")
                error = kept.split(": ", 1)[1] if kept.startswith(f"{label}: ") else None
                error = error or f"no error kept (the launch kept {m}'s first only: {kept})"
                out.failures.append(
                    {"fold": label, "member": m, "revision": V1_STOPPED, "error": error}
                )
    keys = ("member", "fold", "tau", "lower", "upper", "levels", "seconds")
    out.intervals = {r["block"]: {k: r[k] for k in keys} for r in report["tables"]["tau"]}
    save_checkpoint(target, out, set(final["done"]), report["tables"]["selection"])
    log(f"{target.name}: {len(out.failures)} failed fits; fits {fit_revisions(out.fits, labels)}")
    return target


def record_v1_launches(log=print) -> list[str]:
    """The records v1's stopped launches did not write, in v1's family under v1's spec: each
    launch's planned variants as failed at its revision, and the second launch's failed fits
    (from the adopted checkpoint). Returns their run ids."""
    registry = default_registry(COUNTRY)
    if registry.frozen_spec_hash(SUPERSEDES) != SUPERSEDED_SPEC:
        raise ValueError(f"{SUPERSEDES} is not frozen with {SUPERSEDED_SPEC}")
    scheme = load_country_scheme(COUNTRY)
    spec = build_spec(scheme, {**bootstrap_settings(), "seed": BOOTSTRAP_SEED})
    fold_scheme = "walk_forward:" + json.dumps(spec["folds"], sort_keys=True)
    builds = store_builds()  # v1's records' builds
    for r in registry.records(SUPERSEDES):
        if r.fold_scheme != fold_scheme or not r.data_builds.items() >= builds.items():
            raise ValueError(f"{SUPERSEDES}'s records have another fold scheme or store build")

    def v1(variant: Variant, sha: str) -> RunRecord:
        variant = dataclasses.replace(variant, params=variant.params | {"backfilled": V1_NOTE})
        record = _record(variant, sha, builds, SUPERSEDED_SPEC, False)
        return dataclasses.replace(record, family_id=SUPERSEDES)

    records = [
        v1(Variant(h, name, fold_scheme, {"study": "studies/interactions.py"}, {}, error=e), sha)
        for sha, e in V1_LAUNCHES.items()
        for h, name in planned()
    ]
    saved, _ = load_checkpoint(checkpoint_path(spec, builds, trial=False))
    stopped = [f for f in saved["failures"] if f["revision"] == V1_STOPPED]
    records += [v1(v, V1_STOPPED) for v in fit_failures(stopped, fold_scheme)]
    ids, appended = append_new(registry, records)
    log(f"{SUPERSEDES}: {len(appended)} of {len(records)} records appended")
    write_committed_summary(registry, log=log)
    return ids


def interaction_study_command(
    trial: Annotated[
        bool,
        typer.Option(
            "--trial",
            help=f"Records go to {TRIAL_FAMILY_ID} marked trial; the spec is not frozen; the "
            "report is suffixed -trial.",
        ),
    ] = False,
) -> None:
    """Fit the C2 interaction blocks (E35, E36, #93) on D9's folds and write the report."""
    typer.echo(json.dumps(run_study(trial=trial, log=typer.echo), indent=2))
