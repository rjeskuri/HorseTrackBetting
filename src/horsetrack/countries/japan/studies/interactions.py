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
(`core.eval.tail.tail_curve`, D7's meeting bounds), the top band a leaderboard row.

Every variant, failures included, is a registry record; the JRA store is netkeiba results (no
JRDB, no JRA-VAN), so the records carry the source `jra` and their figures are public. Holdout
days are never read (`read_store_table`, `split_off`); the holdout is not opened; nothing is
adopted (D16, D19: ranked, not adopted). A horse x jockey estimate is predictive, never causal
(booking endogeneity, 05 §4.2), and never a bet trigger. A `--trial` run's records go to the
family's `-trial` family with `params.trial` set, and it never freezes the family's spec.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import resource
import tempfile
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from horsetrack.core import paths
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
FAMILY_ID = "japan-c2-interactions-final-win"
TRIAL_FAMILY_ID = f"{FAMILY_ID}-trial"
ARTIFACTS = "interaction_artifacts"
SOURCES = ("jra",)
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
            "tail": "descriptive: realised Win return per unit stake (O_final x won, a dead "
            "heat's co-winner at 1/m) of every scored runner by band of its stage-2 claim "
            f"p x O_final, quantile bands at {list(TAIL_QUANTILES)}, D7's meeting bounds; the "
            "top band (claims from the 99th percentile) is the leaderboard row",
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
            "tail": "claim = stage-2 p x win_odds (cleaned final decimal odds); return = "
            "win_odds x won / m (m co-winners); clusters meeting_id",
        },
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
    feature coefficients, the last fold's tau intervals, errors and timings."""

    u: dict[str, np.ndarray]
    taus: list[dict]
    intervals: dict[str, dict]
    info: float
    errors: dict[str, str]
    seconds: dict[str, float]
    local: dict[str, list[str]]
    prior_sd: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)


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
    intervals: bool = True,
    checkpoint: Path | None = None,
    log=print,
) -> Level1:
    """Each member fitted on every outer fold's training rows and predicting its test rows, the
    fold's rating pair as the rating feature; tau intervals on the last fold's fits."""
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
        saved = json.loads(checkpoint.with_suffix(".json").read_text("utf-8"))
        arrays = np.load(checkpoint)
        out.u = {m: arrays[m] for m in MEMBERS}
        out.taus, out.errors, done = saved["taus"], saved["errors"], set(saved["done"])
        out.seconds, out.local, out.prior_sd = saved["seconds"], saved["local"], saved["prior_sd"]
        log(f"  resumed {len(done)} folds from {checkpoint.name}")
    started = time.monotonic()
    for i, (fold, pick) in enumerate(zip(outer, picks, strict=True)):
        label = fold.test_start.isoformat()
        last = i == len(outer) - 1
        if label in done and not (last and intervals):
            continue
        pair = (pick["drift_var_per_day"], pick["scale"])
        fold_data = data.assign(**{RATING: rating[pair].to_numpy()})
        train, test = fold_data.iloc[fold.train_idx], fold_data.iloc[fold.test_idx]
        row = {"fold": label, "train_rows": len(train), "test_rows": len(test)}
        for member, (blocks, _) in MEMBERS.items():
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
            except Exception as exc:  # recorded by the member's variants, never dropped
                out.errors.setdefault(member, f"{label}: {type(exc).__name__}: {exc}")
            out.seconds[member] = out.seconds.get(member, 0.0) + time.monotonic() - t0
        out.taus = [t for t in out.taus if t["fold"] != label] + [row]
        done.add(label)
        log(f"  fold {label} ({i + 1}/{len(outer)}): {_progress(started)}")
        if checkpoint is not None:
            np.savez(checkpoint, **out.u)
            checkpoint.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "taus": out.taus,
                        "errors": out.errors,
                        "done": sorted(done),
                        "seconds": out.seconds,
                        "local": out.local,
                        "prior_sd": out.prior_sd,
                    }
                ),
                "utf-8",
            )
    return out


# --------------------------------------------------------------------------- level 2


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
    rows = scored.rows
    won = rows["won"].to_numpy(dtype=float)
    m = rows.groupby("race_id")["won"].transform("sum").to_numpy(dtype=float)
    odds = rows["win_odds"].to_numpy(dtype=float)
    bets = pd.DataFrame(
        {
            "claim": p.reindex(rows.index).to_numpy() * odds,
            "ret": np.where(won > 0, odds / np.maximum(m, 1.0), 0.0),
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
    inputs: Inputs, spec: dict, scheme, *, checkpoint: Path | None = None, log=print
) -> tuple[Study, pd.DataFrame]:
    """Every variant, and the per-runner export. Pure but for the optional fit checkpoint."""
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
    l1 = level1_members(level1, rating, outer, picks, checkpoint=checkpoint, log=log)
    del rating
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
    counts |= {f"level2_{k}": v for k, v in price_counts.items()}
    log(f"read {len(level1)} level-1 starters, {len(price)} priced starters")
    return Inputs(level1=level1, results=results, price=price, counts=counts)


SECTIONS = (
    ("cells", "Cell sizes: prior runs per out-of-fold starter's cell (from 2015-Q1)"),
    ("cell_bands", "Cell sizes by band of n_prior (share of out-of-fold starters)"),
    ("horse_x_trainer", "Horse x trainer (not fitted: aliased where the trainer is constant)"),
    ("selection", "#91's rating pair per outer fold"),
    ("tau", "tau per block on the last outer fold (profile-evidence bounds, one-sided 95% each)"),
    ("tau_by_fold", "tau and extra-feature coefficients per outer fold (empirical Bayes)"),
    ("delta", "Stage-2 ΔLL against the D4 calibrated market (out of fold; meeting bootstrap)"),
    ("comparisons", "Paired ΔLL (positive: the first beats the second; meeting bootstrap)"),
    ("tail", "Tail return at the final price by band of stage-2 claim (per ¥100, flat stakes)"),
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


def artifact_stem(spec_hash: str, builds: dict, trial: bool) -> str:
    digest = hashlib.sha256(
        json.dumps(dict(builds), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"interactions-{COUNTRY}-{spec_hash[:12]}-{digest[:12]}" + ("-trial" if trial else "")


def _record(variant, git_sha, builds, spec_hash, trial, *, artifacts=True) -> RunRecord:
    params = dict(variant.params) | {"spec_sha256": spec_hash}
    if trial:
        params["trial"] = True
    if variant.error:
        params["error"] = variant.error
    data_builds = dict(builds)
    if artifacts:
        data_builds[ARTIFACTS] = (
            f"reports+features/{COUNTRY}@{artifact_stem(spec_hash, builds, trial)}"
        )
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


class _Families:
    """The registry restricted to some families, for `write_summary`."""

    def __init__(self, registry, families: set[str]) -> None:
        self._registry, self._families = registry, families

    def records(self, family_id: str | None = None):
        return [r for r in self._registry.records(family_id) if r.family_id in self._families]


_INDEX_ENTRY = re.compile(r"- \[([^\]]+)\]\(\1\.md\)")


def write_committed_summary(registry) -> None:
    """This study's family files by `write_summary`, and the committed index with their lines
    added (`studies.ability.write_committed_summary`'s rule)."""
    directory = summary_dir(COUNTRY)
    mine = _Families(registry, {FAMILY_ID, TRIAL_FAMILY_ID})
    if not mine.records():
        return
    with tempfile.TemporaryDirectory() as scratch:
        write_summary(mine, scratch)
        for path in sorted(Path(scratch).glob("*.md")):
            if path.stem != "README":
                write_report(directory / path.name, path.read_text("utf-8"), sources=())
        fresh = Path(scratch, "README.md").read_text("utf-8").splitlines()
    index = directory / "README.md"
    committed = index.read_text("utf-8").splitlines() if index.exists() else []
    entries = {m[1]: line for line in committed + fresh if (m := _INDEX_ENTRY.fullmatch(line))}
    head = [line for line in fresh if not _INDEX_ENTRY.fullmatch(line)]
    lines = [*head, *(entries[family] for family in sorted(entries))]
    write_report(index, "\n".join(lines) + "\n", sources=())


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
    builds = store_builds()
    stem = artifact_stem(spec_hash, builds, trial)
    checkpoint = paths.data_dir() / "features" / COUNTRY / f"{stem}-level1.npz"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        inputs = load_inputs(log=log)
        study, export = compute_study(inputs, spec, scheme, checkpoint=checkpoint, log=log)
    except Exception as exc:
        fold_scheme = "walk_forward:" + json.dumps(spec["folds"], sort_keys=True)
        for hypothesis, name in planned():
            failed = Variant(
                hypothesis,
                name,
                fold_scheme,
                {"study": "studies/interactions.py"},
                {},
                error=f"{type(exc).__name__}: {exc}",
            )
            registry.append(_record(failed, git_sha, builds, spec_hash, trial, artifacts=False))
        write_committed_summary(registry)
        raise
    log(f"computed ({_progress(started)})")

    blank = {_run_id(dataclasses.replace(r, git_sha="")): r for r in registry.records(family)}
    record_ids, appended = {}, []
    for variant in study.variants:
        record = _record(variant, git_sha, builds, spec_hash, trial)
        same = blank.get(_run_id(dataclasses.replace(record, git_sha="")))
        if same is None:
            record_ids[variant.name] = registry.append(record)
            appended.append(variant.name)
        else:
            record_ids[variant.name] = _run_id(same)
    about = {
        "family": f"`{family}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "data": ", ".join(builds.values()),
        "artifacts (report, JSON, export)": f"`{stem}`",
        "records": f"{len(appended)} appended; {len(record_ids) - len(appended)} already held",
        "trial (records marked params.trial, spec not frozen)": trial,
        "run time": _progress(started),
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = write_report(
        out_dir / f"{stem}.md", render_report(study, spec, about, record_ids), sources=SOURCES
    )
    write_report(
        out_dir / f"{stem}.json",
        json.dumps(
            {"about": about, "counts": study.counts, "tables": study.tables}, indent=1, default=str
        ),
        sources=SOURCES,
    )
    export_path = paths.data_dir() / "features" / COUNTRY / f"{stem}.parquet"
    export.to_parquet(export_path, index=False)
    write_committed_summary(registry)
    return {
        "family_id": family,
        "spec_hash": spec_hash,
        "artifact_stem": stem,
        "report_path": str(report),
        "export_path": str(export_path),
        "registry_path": str(registry.path),
        "records_appended": len(appended),
        "failed_variants": [v.name for v in study.variants if v.error],
    }


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
