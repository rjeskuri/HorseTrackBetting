"""The C2 ability model's real-data run (#92; 08 §3.2 C2, §4 E33, E34): `horsetrack japan
ability-study`.

Level 1 (results only, no price; D10). Every starter of the JRA store's run races with a winner
from 2010 (`level1_frames`): race_id, race_date, horse_id, finish_pos (a starter without a
placing has none: it stays in every choice set and is never chosen), jockey_key and trainer_key,
and the sire, `horse_pedigree.sire_id` of the JRA-VAN store (the owner's answer on #92,
2026-11-26: the JRA-VAN pedigree, `static` in every view). Features (owner, 2026-11-26): #91's
rating member (`studies.rating.member_grid`) at the (drift, scale) #91's rule picks for the fold,
plus jockey, trainer and sire effects shrunk toward 0 (`models.ability.AbilityModel`). The
comparator of E34 is the same model with #129's connection rates in place of the pooled effects:
ln of the strike rate over the race days in [day - 365, day - 1] (#129's window at cut = the
race's day - 1), shrunk RATE_PRIOR_RUNS runs toward that window's field rate, for jockey, trainer
and sire (`connection_features`).

Splits (owner, 2026-11-26): D9's quarters (expanding, 1-week embargo) over the results from
2010; level 1 is out of fold from 2015-Q1 (LEVEL1_FROM: five seasons, 2010-2014, the store's
first quarter from 2010-01-05) and its inner folds are every quarter inside a fold's training
rows from 2011-Q1 (INNER_FROM). The dates are set explicitly: walk_forward's min_train does not
count a partial first quarter. Each fold's discounts (w2, w3 on `DISCOUNT_GRID`, w1 = 1) and
rating pair are selected on its inner folds; the pair by #91's rule (the lowest mean winner log
loss of the rating member alone, `select_pairs`). An inner fold's fit is the same for every
outer fold whose training rows hold it (expanding windows from one origin) at a given pair, so
each (pair, quarter, discounts) fit is made once (`level1_grid`) and the selection sums its winner
log loss over the fold's inner races: `models.ability.fit_discounts`' rule (ties to the earlier
grid entry) from shared fits. The winner-only member is the grid's (1, 0, 0) at the fold's pair
(E33); the rates member is selected the same way on its own fits (E34).

Level 2 (the final Win price). The store's priced starters of the price era (`studies.meta.
store_frame`), D4's calibrated market out of fold on D9's folds (`baseline_oof`), and #87's
stage-2 meta (u = alpha * m + beta * ln q, unbounded, ridge 0) on each D9 fold's training rows
that have a level-1 member (from 2015-Q1, so 2020-Q1 trains on five seasons), predicting the
fold's test quarter; the market-only stage-2 meta (u = beta * ln q, `market_only_oof`) on the same
rows and folds. Scored: every D9 fold, from 2020-Q1.

Variants (one registry record each, failures included): the selection; the member's correlation
with ln q_final (raw and within race, meeting-bootstrap bounds); the member alone against the
calibrated market; the stage-2 meta of each member (pooled, winner-only, rates) and the
market-only meta against the calibrated market, with each fold's coefficients and the pooled
alpha (Wald and D7 meeting-clustered bounds); the pooled member's C4 marginal, the paired ΔLL of
its stage-2 meta against the market-only meta (`c4_marginal`: the same market term and training
rows, so only the member differs); E33, the paired ΔLL of the pooled top-3 meta against the
winner-only meta, at level 2
and (descriptive) at level 1; E34, the paired ΔLL of the pooled meta against the rates meta on
races with a thin-history runner (a jockey, trainer or sire with fewer than THIN_STARTS starts in
the store on earlier race days since 2010) and over every race; and the by-class table (owner,
2026-11-27). The N5 disagreement-tail curve is added when N5 lands; it is not part of this run.

Built on JRA-VAN data (the sire), so every figure is withheld (D21): the report and the
per-runner export (u, se_u, p_ability) go under the data dir through `write_report`, the
records carry the source `jravan` and the committed summary withholds them. Holdout days are
never read (`read_store_table`, `split_off`); the holdout is not opened; nothing is adopted
(D16). A `--trial` run's records go to the family's `-trial` family with `params.trial` set, and
it never freezes the family's spec. The report, its JSON and the export are named by
`artifact_stem` (the spec and the data builds), which each record names, so a rerun on refreshed
data writes new files beside the earlier ones.

Family `japan-c2-ability-final-win-v2` supersedes `japan-c2-ability-final-win` (#92's review,
round 1), whose records stay in the registry: v1 reported the C4 gain only against D4 (a spline
in ln q, fitted on another sample), which cannot isolate the member, and anchored the rates'
window at the last race day before the race rather than #129's day - 1.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import resource
import tempfile
import time
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import typer

from horsetrack.core import paths
from horsetrack.core.eval.folds import Fold, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha, split_off
from horsetrack.core.eval.registry import RunRecord, default_registry, summary_dir, write_summary
from horsetrack.core.eval.reports import write_report
from horsetrack.core.eval.scoring import gain
from horsetrack.countries.japan.handicap.factors import RATE_PRIOR_RUNS, RATE_WINDOW_DAYS
from horsetrack.countries.japan.models.ability import DISCOUNT_GRID, AbilityModel
from horsetrack.countries.japan.models.condlogit import fit_conditional_logit
from horsetrack.countries.japan.models.meta import MetaSpec, Q, oof_predictions
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table
from horsetrack.countries.japan.studies import rating as c2_rating
from horsetrack.countries.japan.studies.calibration import (
    Study,
    Variant,
    _bounds,
    _markdown_table,
    _metric,
    bootstrap_settings,
    store_builds,
)
from horsetrack.countries.japan.studies.meta import (
    CLUSTER,
    CLUSTERED,
    Scored,
    _clustered_metrics,
    _delta,
    _ll,
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
TICKET = 92
FAMILY_ID = "japan-c2-ability-final-win-v2"
TRIAL_FAMILY_ID = f"{FAMILY_ID}-trial"  # a trial's records; a trial never freezes FAMILY_ID
SUPERSEDES = "japan-c2-ability-final-win"
ARTIFACTS = "ability_artifacts"  # the data_builds entry naming a run's report and export stem
SOURCES = ("jra", "jravan")
BOOTSTRAP_SEED = 92  # reproducibility only; scoring.yaml signs no seed
# Splits signed by the owner on #92 (2026-11-26), never Rulings:
RESULTS_FROM = c2_rating.RATING_FROM  # 2010-01-01: level 1 trains on results from here
INNER_FROM = date(2011, 1, 1)  # inner folds: every quarter from 2011-Q1
LEVEL1_FROM = date(2015, 1, 1)  # level 1 out of fold from 2015-Q1
THIN_STARTS = 30  # E34's thin history: fewer prior starts than this (signed, 2026-11-26)
ENTITIES = ("jockey_key", "trainer_key", "sire_id")
RATING = "rating"
RATES = ("ln_jockey_rate", "ln_trainer_rate", "ln_sire_rate")
PRIOR_STARTS = ("prior_starts_jockey", "prior_starts_trainer", "prior_starts_sire")
WINNER_ONLY = (1.0, 0.0, 0.0)
POOLED, WIN_ONLY, RATED = "ability", "ability_winner_only", "ability_rates"
MARKET_ONLY = "market_only"  # the stage-2 meta without a member: u = beta * ln q
MEMBERS = {
    POOLED: "top-3 Plackett-Luce, rating + pooled jockey, trainer and sire, discounts selected",
    WIN_ONLY: "the same at discounts (1, 0, 0): winner-only training (E33's comparator)",
    RATED: "top-3 Plackett-Luce, rating + #129's ln connection rates, discounts selected (E34)",
}
STAGE2 = {
    m: MetaSpec([m], quadratic=False, segment_cols=[], form="stage2", ridge=0.0) for m in MEMBERS
}
RUNNER_COLUMNS = [
    "race_id",
    "race_date",
    "horse_id",
    "finish_pos",
    "is_scratched",
    "jockey_key",
    "trainer_key",
]
RACE_COLUMNS = ["race_id", "race_date", "meeting_id", "race_class", "status"]


# --------------------------------------------------------------------------- the spec


def level1_quarters(dates, scheme) -> list[Fold]:
    """D9's quarter folds over the level-1 dates from INNER_FROM: each quarter's training rows
    (every earlier row to its embargo) and its test rows."""
    folds = walk_forward(dates, scheme=dataclasses.replace(scheme, min_train=1))
    return [f for f in folds if f.test_start >= INNER_FROM]


def build_spec(scheme, bootstrap: dict) -> dict:
    """The family's pre-registered spec, frozen in the registry before any outcome is read."""
    return {
        "family_id": FAMILY_ID,
        "supersedes": f"{SUPERSEDES}: its C4 gain compared the stage-2 meta (linear in ln q, "
        "with the member) only with D4's calibrated market (a spline in ln q on another sample), "
        "so it could not isolate the member, and its rates' window ended on the last race day "
        "before the race, not #129's day - 1; found in #92's review (round 1)",
        "ticket": TICKET,
        "hypotheses": {
            "E33": "discounted top-3 Plackett-Luce training beats winner-only training on the "
            "same features: Win ΔLL at the final price after stage 2 (08 §4 E33; expected "
            "+0.001 to +0.004 standalone)",
            "E34": "partial pooling of jockey, trainer and sire beats #129's as-of rates on "
            "races with thin entity history but adds nothing at the final price overall (08 §4 "
            "E34; expected 0.001 or less, concentrated in debut and maiden races; a null at the "
            "final price expected)",
            "C2": "the pooled top-3 member earns stage-2 weight beside ln q_final (standalone "
            "and marginal in C4; 08 §3.2 C2)",
        },
        "null": "the member carries nothing the final price lacks (alpha = 0); top-3 training "
        "is no better than winner-only; pooling is no better than the rates",
        "hypothesis_ids": "E33 on the selection, the pooled member's correlation, standalone "
        "and stage-2 variants, the winner-only and market-only metas, the C4 marginal and E33's "
        "comparisons; E34 on the rates meta, E34's comparisons and the by-class table",
        "decision_rules": {
            "E33": "top-3 beats winner-only when the meeting-bootstrap one-sided 95% lower bound "
            "of the paired per-race winner ΔLL of the stage-2 meta with the pooled top-3 member "
            "against the stage-2 meta with the winner-only member (out of fold) over the scored "
            "races lies above 0",
            "E34 thin": "pooling beats the rates on thin history when the same bound for the "
            "stage-2 meta with the pooled member against the stage-2 meta with the rates member, "
            "over the scored races with at least one thin-history runner, lies above 0",
            "E34 overall": "descriptive: the same ΔLL over every scored race (the marginal "
            "ablation in C4), with its bounds",
            "stage-2 weight": "alpha > 0 when the meeting-clustered one-sided 95% lower bound of "
            "the pooled member's alpha pooled over the scored rows (in-sample) lies above 0",
            "stage-2 gain": "the pooled member's stage-2 meta has an out-of-fold ΔLL against the "
            "calibrated market whose meeting-bootstrap lower bound lies above 0",
            "C4 marginal": "the pooled member adds to the market term in C4 when the "
            "meeting-bootstrap one-sided 95% lower bound of the paired per-race winner ΔLL of the "
            "stage-2 meta with the pooled member against the market-only stage-2 meta (the same "
            "market term, training rows and folds, no member), out of fold over the scored races, "
            "lies above 0",
            "market only": "descriptive: the market-only stage-2 meta against the calibrated "
            "market (out-of-fold ΔLL with meeting-bootstrap bounds) and its beta per fold (Wald "
            "95% two-sided; meeting-clustered one-sided 95% in the report)",
            "distinct": "descriptive: the pooled member's Pearson correlation with ln q_final "
            "over the scored runners, raw and within race, with meeting-bootstrap one-sided 95% "
            "bounds",
            "standalone": "descriptive: the pooled member alone against the calibrated market "
            "(out-of-fold ΔLL with meeting-bootstrap bounds) and each member's pseudo-R2 "
            "against equal chances beside q's and the calibrated market's",
            "E33 level 1": "descriptive: the top-3 member alone against the winner-only member "
            "alone, the paired ΔLL over the scored races with meeting-bootstrap bounds",
        },
        "level1": {
            "frame": f"every starter (not scratched) of the JRA store's run races with a winner, "
            f"from {RESULTS_FROM.isoformat()}; no price is read; a starter without a placing has "
            "no finish_pos: it stays in every choice set and is never chosen",
            "model": "models.ability.AbilityModel(k=3): u = beta * rating + jockey + trainer + "
            "sire effects, each N(0, tau_e^2) with tau_e^2 by empirical Bayes (penalised MAP); "
            "dead heats at 1/m per co-placed runner in the shared stage; positions 4+ unused",
            "rating": "studies.rating: DynamicRating over every starter of the store's run races "
            "from 2010 (a starter without a placing tied last), its member logp (normalised "
            "over the level-1 starters) at the (drift, scale) #91's rule picks per outer fold: "
            "the lowest mean winner log loss of the member alone over the fold's inner races "
            "(dead heats at 1/k), the first grid pair on a tie",
            "rating_fixed": c2_rating.FIXED,
            "drift_grid": list(c2_rating.DRIFT_GRID),
            "scale_grid": list(c2_rating.SCALE_GRID),
            "entities": {
                "jockey_key": "runners.jockey_key (JRA store)",
                "trainer_key": "runners.trainer_key (JRA store)",
                "sire_id": "the JRA-VAN horse_pedigree.sire_id of runners.horse_id (static); a "
                "horse without one has no sire level (effect 0, the prior)",
            },
            "discount_grid": [list(w) for w in DISCOUNT_GRID],
            "discount_selection": "per outer fold, the grid entry with the lowest summed "
            "winner log loss (k = 1, dead heats at 1/m) over the fold's inner races, each "
            "predicted by the fit on its inner fold's training rows at the fold's rating pair; "
            "ties go to the earlier entry (models.ability.fit_discounts' rule)",
            "winner_only": list(WINNER_ONLY),
            "rates": f"#129's connection rates (handicap.factors._connection_rates at cut = the "
            f"race's day - 1): over the not-scratched runners of the store on the race days in "
            f"[day - {RATE_WINDOW_DAYS}, day - 1], (wins + {RATE_PRIOR_RUNS} * p0) / "
            f"(runs + {RATE_PRIOR_RUNS}) with p0 the window's win rate, for jockey_key, "
            "trainer_key and sire_id; a level the window lacks gets p0 as the formula gives; a "
            "missing key gets p0 too (where #129 gives null: the model needs a finite feature); "
            "features ln(rate); a race day whose window holds no runner (the store's first) gets "
            "ln rate 0 for every runner, a constant within race",
            "rates_member": "AbilityModel(k=3) on [rating, ln_jockey_rate, ln_trainer_rate, "
            "ln_sire_rate], no entity effects, discounts selected the same way",
        },
        "thin_history": f"a runner whose jockey, trainer or sire has fewer than {THIN_STARTS} "
        "not-scratched starts in the store on race days before the race's day, since "
        f"{RESULTS_FROM.isoformat()}; a missing sire counts 0; a thin race has at least one "
        "such runner among its priced starters",
        "level2": {
            "frame": "studies.meta.store_frame: the priced starters of run races with a priced "
            "winner, from the price era; q the normalised final Win price",
            "baseline": "D4 calibrated market (models.v0.calibration), in-fold on D9's folds",
            "member": "ln p of the level-1 member, renormalised over the race's priced starters",
            "meta": "models.meta.fit_meta stage2 on [member] and ln q, ridge 0, unbounded, on "
            "each D9 fold's training rows that have a level-1 member (from 2015-Q1); pooled "
            "in-sample over the scored rows",
            "market_only": "u = beta * ln q (q normalised within race), beta unbounded, by "
            "conditional logit (models.condlogit) on the same rows and folds as the members' "
            "metas, predicting each fold's test quarter: the stage-2 meta without a member",
        },
        "by_class": "per races.race_class over the scored races: the stage-2 and member-alone "
        "ΔLL against the calibrated market, E33's and E34's ΔLL, each with meeting-bootstrap "
        "bounds, and the share of thin races",
        "folds": {
            "level1": "D9's quarters (unit, anchor, expanding window and embargo of level2) "
            f"over the level-1 dates from {RESULTS_FROM.isoformat()}; each quarter trains on "
            "every earlier row to its embargo",
            "level1_out_of_fold_from": LEVEL1_FROM.isoformat(),
            "inner_from": INNER_FROM.isoformat(),
            "inner": "every quarter from inner_from inside an outer fold's training rows",
            "level2": dataclasses.asdict(scheme),
            "scored": "every level-2 D9 fold",
        },
        "bootstrap": bootstrap,
        "price_era_start": PRICE_ERA_START.isoformat(),
        "export": "per level-1 out-of-fold runner (from 2015-Q1): u, se_u and p_ability of the "
        "pooled member at its fold's selection, under the data dir",
        "artifacts": "the report, its JSON and the export are named by the spec hash and a "
        "digest of the data builds (artifact_stem), and each record names that stem in "
        f"data_builds.{ARTIFACTS}, so a rerun on refreshed data keeps the earlier files",
        "n5": "the N5 disagreement-tail curve is added when N5 lands (owner, 2026-11-27); not "
        "in this family",
        "owner_signoff": "#92: features, sire source, E34 comparator and splits (2026-11-26); "
        "the by-class table (2026-11-27)",
        "adoption": "none (D16): no holdout is opened and nothing is adopted",
        "sources": list(SOURCES),
    }


# --------------------------------------------------------------------------- inputs


def read_runners() -> tuple[pd.DataFrame, pd.DataFrame]:
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


def read_sires() -> pd.DataFrame:
    """horse_id -> sire_id from the JRA-VAN store's horse_pedigree (static: no date, no
    outcome, no price), two columns only."""
    path = paths.processed_dir("jravan") / "horse_pedigree.parquet"
    table = pq.read_table(path, columns=["horse_id", "sire_id"]).to_pandas()
    return table.dropna(subset=["horse_id"]).drop_duplicates("horse_id")


def pedigree_build() -> dict[str, str]:
    digest = hashlib.sha256()
    with open(paths.processed_dir("jravan") / "horse_pedigree.parquet", "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    path = "processed/jravan/horse_pedigree.parquet"
    return {"jravan_horse_pedigree": f"{path}@{digest.hexdigest()[:16]}"}


def level1_frames(
    runners: pd.DataFrame, races: pd.DataFrame, sires: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(base, level1, counts). base: every not-scratched runner (#129's rate window, any race),
    with sire_id. level1: base's starters of run races with a winner, sorted by date and race,
    with meeting_id, race_class and won."""
    scratched = runners["is_scratched"].eq(True).to_numpy(dtype=bool, na_value=False)
    base = runners.loc[~scratched].drop(columns="is_scratched")
    base = base.merge(sires, on="horse_id", how="left", validate="many_to_one")
    base["finish_pos"] = pd.to_numeric(base["finish_pos"]).astype(float)
    base["won"] = base["finish_pos"].eq(1).to_numpy()
    run = races.loc[races["status"].eq("run"), ["race_id", "meeting_id", "race_class"]]
    level1 = base.merge(run, on="race_id", how="inner", validate="many_to_one")
    winners = level1.groupby("race_id")["won"].transform("sum")
    level1 = level1.loc[winners > 0]
    level1 = level1.sort_values(["race_date", "race_id"], kind="stable").reset_index(drop=True)
    per_race = level1.groupby("race_id")["won"].sum()
    counts = {
        "level1_runners_read": len(runners),
        "level1_scratched": int(scratched.sum()),
        "level1_races": len(per_race),
        "level1_starters": len(level1),
        "level1_starters_without_a_placing": int(level1["finish_pos"].isna().sum()),
        "level1_starters_without_a_sire": int(level1["sire_id"].isna().sum()),
        "level1_dead_heat_races": int((per_race >= 2).sum()),
    }
    return base, level1, counts


def connection_features(base: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Per row of `rows` (race_date and the ENTITIES): #129's ln connection rate per entity over
    the race days in [day - RATE_WINDOW_DAYS, day - 1] (`_connection_rates` at cut = day - 1),
    and its prior starts (starts in `base` on every race day before the row's day). A missing key
    gets p0, the window's field rate (#129 gives null). Index: rows'."""
    days = np.sort(pd.unique(base["race_date"].to_numpy()))
    b_day = np.searchsorted(days, base["race_date"].to_numpy())
    r_date = rows["race_date"].to_numpy()
    r_day = np.searchsorted(days, r_date)
    if not np.array_equal(days[np.minimum(r_day, len(days) - 1)], r_date):
        raise ValueError("a row's race day is not a race day of the base")
    won = base["won"].to_numpy(dtype=float)
    cum_runs = np.r_[0.0, np.cumsum(np.bincount(b_day, minlength=len(days)))]
    cum_wins = np.r_[0.0, np.cumsum(np.bincount(b_day, weights=won, minlength=len(days)))]
    hi = r_day  # window race days are lo .. hi - 1: before the row's day, so to day - 1
    lo = np.searchsorted(days, r_date - np.timedelta64(RATE_WINDOW_DAYS, "D"), side="left")
    runs = cum_runs[hi] - cum_runs[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        p0 = (cum_wins[hi] - cum_wins[lo]) / runs
    out = pd.DataFrame(index=rows.index)
    for key, rate_col, starts_col in zip(ENTITIES, RATES, PRIOR_STARTS, strict=True):
        levels = pd.Index(pd.unique(base[key].dropna().to_numpy()))
        b_code = levels.get_indexer(base[key].to_numpy())
        r_code = levels.get_indexer(rows[key].to_numpy())
        keep = b_code >= 0
        cr = np.zeros((len(levels), len(days) + 1))
        cw = np.zeros((len(levels), len(days) + 1))
        np.add.at(cr, (b_code[keep], b_day[keep] + 1), 1.0)
        np.add.at(cw, (b_code[keep], b_day[keep] + 1), won[keep])
        cr, cw = np.cumsum(cr, axis=1), np.cumsum(cw, axis=1)
        known = r_code >= 0
        c = np.where(known, r_code, 0)
        k_runs = np.where(known, cr[c, hi] - cr[c, lo], 0.0)
        k_wins = np.where(known, cw[c, hi] - cw[c, lo], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = (k_wins + RATE_PRIOR_RUNS * p0) / (k_runs + RATE_PRIOR_RUNS)
            ln_rate = np.where(runs > 0, np.log(rate), 0.0)
        out[rate_col] = ln_rate
        out[starts_col] = np.where(known, cr[c, hi], 0.0)
    return out


# --------------------------------------------------------------------------- level 1


@dataclasses.dataclass
class Grid:
    """Level-1 predictions on the frame's rows: u (and se_u for the pooled model) per (model,
    pair, discounts), NaN on rows no quarter predicted; the quarters fitted per pair."""

    u: dict[tuple, np.ndarray]
    se: dict[tuple, np.ndarray]
    quarters: dict[tuple, int]
    seconds: float = 0.0
    fits: int = 0


def race_index(frame: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
    codes, labels = pd.factorize(frame["race_id"].to_numpy(dtype=object))
    return codes, pd.Index(labels)


def race_winner_loss(u: np.ndarray, codes: np.ndarray, won: np.ndarray, n_races: int) -> np.ndarray:
    """Per race code: the winner log loss of softmax(u) over the race's rows (dead heats at
    1/m), NaN where u is NaN."""
    missing = np.bincount(codes, weights=np.isnan(u), minlength=n_races) > 0
    v = np.where(np.isnan(u), 0.0, u)
    top = np.full(n_races, -np.inf)
    np.maximum.at(top, codes, v)
    shifted = v - top[codes]
    total = np.bincount(codes, weights=np.exp(shifted), minlength=n_races)
    m = np.bincount(codes, weights=won, minlength=n_races)
    chosen = np.bincount(codes, weights=np.where(won > 0, shifted, 0.0), minlength=n_races)
    with np.errstate(divide="ignore", invalid="ignore"):
        loss = np.log(total) - chosen / m
    return np.where(missing | (m == 0), np.nan, loss)


def _codes(ids: pd.Series) -> np.ndarray:
    """Each id's integer code as a float, NaN for a missing id."""
    codes = pd.factorize(ids.to_numpy(dtype=object))[0].astype(float)
    codes[codes < 0] = np.nan
    return codes


def level1_grid(
    frame: pd.DataFrame,
    members: dict,
    quarters: list[Fold],
    needed: dict[tuple, int],
    log: Callable[[str], None] = print,
) -> Grid:
    """For each rating pair in `needed` and each quarter up to its index there: the pooled and
    the rates AbilityModel at every DISCOUNT_GRID entry, fitted on the quarter's training rows
    and predicting its test rows."""
    n = len(frame)
    grid = Grid(u={}, se={}, quarters=dict(needed))
    started = time.monotonic()
    numeric = pd.DataFrame(  # ids as codes (a missing entity id stays missing), so copies are small
        {col: pd.factorize(frame[col].to_numpy(dtype=object))[0] for col in ("race_id", "horse_id")}
        | {"finish_pos": frame["finish_pos"].to_numpy(dtype=float)}
        | {col: _codes(frame[col]) for col in ENTITIES}
        | {col: frame[col].to_numpy(dtype=float) for col in RATES}
    )
    for pair, last in needed.items():
        data = numeric.assign(**{RATING: members[pair].to_numpy()})
        for w in DISCOUNT_GRID:
            grid.u[("pooled", pair, w)] = np.full(n, np.nan)
            grid.se[("pooled", pair, w)] = np.full(n, np.nan, dtype=np.float32)
            grid.u[("rates", pair, w)] = np.full(n, np.nan)
        for qi, quarter in enumerate(quarters[: last + 1]):
            train, test = data.iloc[quarter.train_idx], data.iloc[quarter.test_idx]
            for w in DISCOUNT_GRID:
                pooled = AbilityModel(k=3, discounts=w, entity_cols=ENTITIES, features=[RATING])
                out = pooled.fit(train).predict(test)
                grid.u[("pooled", pair, w)][quarter.test_idx] = out["u"].to_numpy()
                grid.se[("pooled", pair, w)][quarter.test_idx] = out["se_u"].to_numpy()
                rates = AbilityModel(k=3, discounts=w, features=[RATING, *RATES])
                grid.u[("rates", pair, w)][quarter.test_idx] = (
                    rates.fit(train).predict(test)["u"].to_numpy()
                )
                grid.fits += 2
            log(
                f"  pair {pair} quarter {quarter.test_start} ({qi + 1}/{last + 1}): "
                f"{_progress(started)}"
            )
    grid.seconds = time.monotonic() - started
    return grid


def _inner(race_date: np.ndarray, fold: Fold) -> np.ndarray:
    """The races of a fold's inner folds: from INNER_FROM to the fold's training end."""
    return (race_date >= np.datetime64(INNER_FROM)) & (race_date <= np.datetime64(fold.train_end))


def select_pairs(frame: pd.DataFrame, members: dict, outer: list[Fold]) -> list[dict]:
    """#91's rule per outer fold: the (drift, scale) whose member alone has the lowest mean
    winner log loss over the fold's inner races (the first grid pair on a tie)."""
    losses = pd.DataFrame({k: _ll(np.exp(m), frame) for k, m in members.items()})
    race_date = frame.groupby("race_id")["race_date"].first().reindex(losses.index).to_numpy()
    out = []
    for fold in outer:
        mean = losses.loc[_inner(race_date, fold)].mean()
        drift, scale = mean.idxmin()
        out.append(
            {"fold": fold.test_start.isoformat(), "drift_var_per_day": drift, "scale": scale}
            | {"rating inner log loss": float(mean.min()), "rating worst pair": float(mean.max())}
        )
    return out


def select_discounts(
    frame: pd.DataFrame, grid: Grid, outer: list[Fold], picks: list[dict]
) -> list[dict]:
    """Per outer fold (its `select_pairs` row): the pooled and the rates model's discounts with
    the lowest summed inner-fold winner log loss at the fold's rating pair (ties to the earlier
    grid entry)."""
    codes, labels = race_index(frame)
    won = frame["won"].to_numpy(dtype=float)
    race_date = frame.groupby(codes)["race_date"].first().to_numpy()
    losses = {key: race_winner_loss(u, codes, won, len(labels)) for key, u in grid.u.items()}
    out = []
    for fold, pick in zip(outer, picks, strict=True):
        pair = (pick["drift_var_per_day"], pick["scale"])
        inner = _inner(race_date, fold)
        row = dict(pick) | {"inner races": int(inner.sum())}
        for model in ("pooled", "rates"):
            totals = []
            for w in DISCOUNT_GRID:
                values = losses[(model, pair, w)][inner]
                if np.isnan(values).any():
                    raise ValueError(f"an inner race of {row['fold']} has no {model} prediction")
                totals.append(float(values.sum()))
            best = int(np.argmin(totals))  # the first minimum: ties to the earlier entry
            row |= {
                f"{model} w2": DISCOUNT_GRID[best][1],
                f"{model} w3": DISCOUNT_GRID[best][2],
                f"{model} inner log loss": totals[best] / max(int(inner.sum()), 1),
                f"{model} winner-only inner log loss": totals[DISCOUNT_GRID.index(WINNER_ONLY)]
                / max(int(inner.sum()), 1),
            }
        out.append(row)
    return out


def level1_members(frame: pd.DataFrame, grid: Grid, outer: list[Fold], picks: list[dict]):
    """Each member's u on the outer folds' test rows (NaN elsewhere) at the fold's selection
    (`select_discounts`' rows), the pooled member's se_u and the fold each row is out of fold
    for."""
    n = len(frame)
    u = {m: np.full(n, np.nan) for m in MEMBERS}
    se = np.full(n, np.nan)
    fold_of = np.full(n, None, dtype=object)
    for fold, pick in zip(outer, picks, strict=True):
        pair = (pick["drift_var_per_day"], pick["scale"])
        pooled_w = (1.0, pick["pooled w2"], pick["pooled w3"])
        rates_w = (1.0, pick["rates w2"], pick["rates w3"])
        idx = fold.test_idx
        u[POOLED][idx] = grid.u[("pooled", pair, pooled_w)][idx]
        se[idx] = grid.se[("pooled", pair, pooled_w)][idx]
        u[WIN_ONLY][idx] = grid.u[("pooled", pair, WINNER_ONLY)][idx]
        u[RATED][idx] = grid.u[("rates", pair, rates_w)][idx]
        fold_of[idx] = fold.test_start.isoformat()
    return u, se, fold_of


def needed_pairs(outer: list[Fold], picks: list[dict], quarters: list[Fold]) -> dict[tuple, int]:
    """Per rating pair some outer fold picked: the index (into quarters) of the last quarter its
    folds need, its latest outer fold's test quarter."""
    starts = [q.test_start for q in quarters]
    needed: dict[tuple, int] = {}
    for fold, pick in zip(outer, picks, strict=True):
        pair = (pick["drift_var_per_day"], pick["scale"])
        needed[pair] = max(needed.get(pair, -1), starts.index(fold.test_start))
    return needed


# --------------------------------------------------------------------------- level 2


def member_frame(price: pd.DataFrame, level1: pd.DataFrame, u: dict, thin: np.ndarray, first):
    """The price frame's rows with every level-1 member: each member's ln p renormalised over
    the race's priced starters, and the thin-history flag. A race with a starter lacking a
    member is dropped (counted from `first`, the first out-of-fold day). Index: price frame
    positions."""
    keys = level1[["race_id", "horse_id"]].assign(**{m: u[m] for m in MEMBERS}, thin=thin)
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


def _paired(p_a: pd.Series, p_b: pd.Series, scored: Scored, spec: dict, races=None):
    """The per-race winner ΔLL of a against b (positive: a better) and its meeting bounds, over
    the scored races (or `races`)."""
    rows = scored.rows
    g = gain(_ll(p_a, rows), _ll(p_b, rows)).per_race
    if races is not None:
        g = g.loc[g.index.isin(races)]
    return g, _bounds(g, scored.clusters, spec)


def _fold_points(g: pd.Series, scored: Scored, name: str) -> list:
    fold_of = scored.rows["fold"].groupby(scored.rows["race_id"].to_numpy()).first()
    return [
        _metric(fold, name, part.mean(), len(part))
        for fold, part in g.groupby(fold_of.reindex(g.index).to_numpy())
    ]


def _comparison(label, p_a, p_b, scored, spec, races=None, metric="delta_ll") -> tuple[list, dict]:
    g, iv = _paired(p_a, p_b, scored, spec, races)
    metrics = _fold_points(g, scored, metric)
    metrics.append(_metric("all_folds", metric, iv.mean, len(g), iv.lo, iv.hi))
    row = {"comparison": label, "races": len(g), "meetings": iv.n_clusters}
    row |= {"ΔLL": iv.mean, "lower": iv.lo, "upper": iv.hi}
    return metrics, {"comparisons": [row]}


@dataclasses.dataclass(frozen=True)
class MarketFit:
    """The market-only stage-2 meta on one fold's training rows: beta of u = beta * ln q, under
    `models.meta`'s term name Q, with its se, meeting-clustered se and race count (the fields
    `studies.meta._meta_metrics` reads)."""

    coef: pd.Series
    se: pd.Series
    se_cluster: pd.Series
    n_races: int


def _within_race_ln_q(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(race codes, ln q normalised within race) of a frame with race_id and Q."""
    codes = pd.factorize(frame["race_id"].to_numpy(dtype=object))[0]
    v = frame[Q].to_numpy(dtype=float)
    top = pd.Series(v).groupby(codes).transform("max").to_numpy()
    return codes, v - top - np.log(np.bincount(codes, weights=np.exp(v - top))[codes])


def _market_only(rows: pd.DataFrame, folds: list[Fold], market: str = "ln_q"):
    """The market-only stage-2 meta out of fold on `rows` (P(win) per fold's test rows, NaN on
    rows no fold tests) and each fold's fit, in fold order: the members' metas' market term on
    the same training rows, with no member."""
    data = rows[["race_id", "won", CLUSTER]].assign(**{Q: rows[market]})
    fits: list[MarketFit] = []

    def fit(train: pd.DataFrame) -> MarketFit:
        codes, ln_q = _within_race_ln_q(train)
        out = fit_conditional_logit(
            ln_q[:, None],
            codes,
            train["won"].to_numpy(dtype=bool),
            cluster=train[CLUSTER].to_numpy(dtype=object),
        )
        fits.append(
            MarketFit(
                *(pd.Series(v, index=[Q], dtype=float) for v in (out.coef, out.se, out.se_cluster)),
                n_races=out.n_races,
            )
        )
        return fits[-1]

    def predict(model: MarketFit, test: pd.DataFrame) -> np.ndarray:
        codes, ln_q = _within_race_ln_q(test)
        u = float(model.coef[Q]) * ln_q
        e = np.exp(u - pd.Series(u).groupby(codes).transform("max").to_numpy())
        return e / np.bincount(codes, weights=e)[codes]

    return oof_predictions(data, folds, fit, predict), fits


def market_only_oof(rows: pd.DataFrame, folds: list[Fold], market: str = "ln_q") -> pd.Series:
    """The market-only stage-2 meta's out-of-fold P(win) on rows' index: u = beta * ln q
    (`market`, normalised within race), beta unbounded, fitted by conditional logit on each
    fold's training rows and predicting its test rows; NaN on rows no fold tests."""
    return _market_only(rows, folds, market)[0]


def c4_marginal(
    p_member: pd.Series, rows: pd.DataFrame, folds: list[Fold], scored: Scored, spec: dict
) -> tuple[list, dict]:
    """A member's marginal in C4: the paired per-race winner ΔLL of `p_member` (the stage-2
    meta with the member, out of fold on `rows` and `folds`) against the market-only stage-2
    meta on the same rows, folds and market term, positive when the member's meta is better: per
    scored fold and over `scored`'s races with D7's meeting bounds (`spec["bootstrap"]`)."""
    return _comparison(
        "C4 marginal: stage-2 with the member vs stage-2 market only",
        p_member,
        market_only_oof(rows, folds),
        scored,
        spec,
    )


def _cluster_corr(x: np.ndarray, y: np.ndarray, cluster: np.ndarray, spec: dict):
    """Pearson's r of x and y and its meeting-bootstrap one-sided bounds (resampling clusters'
    sufficient statistics)."""
    codes, _ = pd.factorize(cluster, sort=True)
    k = int(codes.max()) + 1
    stats = np.column_stack(
        [
            np.bincount(codes, weights=w, minlength=k)
            for w in (np.ones_like(x), x, y, x * x, y * y, x * y)
        ]
    )

    def corr(s: np.ndarray) -> np.ndarray:
        n, sx, sy, sxx, syy, sxy = (s[..., j] for j in range(6))
        cov = sxy - sx * sy / n
        return cov / np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))

    boot = spec["bootstrap"]
    rng = np.random.default_rng(boot["seed"])
    reps, chunk = [], max(1, 2_000_000 // k)
    for start in range(0, boot["n_boot"], chunk):
        draws = rng.integers(0, k, size=(min(chunk, boot["n_boot"] - start), k))
        reps.append(corr(stats[draws].sum(axis=1)))
    reps = np.concatenate(reps)
    level = boot["level"]
    return (
        float(corr(stats.sum(axis=0))),
        float(np.quantile(reps, 1 - level)),
        float(np.quantile(reps, level)),
        k,
    )


def _correlation(scored: Scored, spec: dict) -> tuple[list, dict]:
    rows = scored.rows
    m, q = rows[POOLED].to_numpy(), rows["ln_q"].to_numpy()
    race = rows["race_id"].to_numpy()
    centre = lambda v: v - pd.Series(v).groupby(race).transform("mean").to_numpy()  # noqa: E731
    meeting = rows[CLUSTER].to_numpy()
    metrics, table = [], []
    for label, (a, b) in (
        ("pearson_ln_q", (m, q)),
        ("pearson_ln_q_within_race", (centre(m), centre(q))),
    ):
        r, lo, hi, k = _cluster_corr(a, b, meeting, spec)
        metrics.append(_metric("all_folds", label, r, scored.n_races, lo, hi))
        table.append(
            {"measure": label, "runners": len(a), "races": scored.n_races, "meetings": k}
            | {"r": r, "lower": lo, "upper": hi}
        )
    for fold, part in rows.groupby("fold"):
        raw = float(np.corrcoef(part[POOLED], part["ln_q"])[0, 1])
        metrics.append(_metric(fold, "pearson_ln_q", raw, part["race_id"].nunique()))
    return metrics, {"correlation": table}


def _alone(scored: Scored, spec: dict) -> tuple[list, dict]:
    rows = scored.rows
    metrics, tables = _delta(np.exp(rows[POOLED]), scored, spec, f"member_alone:{POOLED}")
    uniform = scored.uniform.sum()
    table = []
    for name, p in (
        ("member (pooled top-3)", np.exp(rows[POOLED])),
        ("member (winner-only)", np.exp(rows[WIN_ONLY])),
        ("member (rates)", np.exp(rows[RATED])),
        ("q", rows["q"]),
        ("calibrated market", rows["p_cal"]),
    ):
        ll = _ll(p, rows)
        r2 = 1.0 - ll.sum() / uniform
        metrics.append(_metric("all_folds", f"pseudo_r2_vs_equal_chances[{name}]", r2, len(ll)))
        table.append(
            {"forecast": name, "races": len(ll), "mean log loss": ll.mean(), "pseudo-R2": r2}
        )
    return metrics, tables | {"pseudo_r2": table}


def _by_class(scored: Scored, p: dict, spec: dict, thin_races: pd.Index) -> tuple[list, dict]:
    rows = scored.rows
    race = rows["race_id"].to_numpy()
    klass = rows["race_class"].fillna("unknown").groupby(race).first()
    pairs = {
        "stage2_vs_calibrated_market": (p[POOLED], rows["p_cal"], None),
        "member_alone_vs_calibrated_market": (np.exp(rows[POOLED]), rows["p_cal"], None),
        "E33_top3_vs_winner_only": (p[POOLED], p[WIN_ONLY], None),
        "E34_pooled_vs_rates_thin": (p[POOLED], p[RATED], thin_races),
    }
    metrics, table = [], []
    for label in sorted(klass.unique()):
        races = klass.index[klass.eq(label)]
        row = {"class": label, "races": len(races), "thin races": int(races.isin(thin_races).sum())}
        for name, (a, b, subset) in pairs.items():
            keep = races if subset is None else races[races.isin(subset)]
            if not len(keep):
                continue
            g, iv = _paired(a, b, scored, spec, keep)
            metrics.append(_metric(label, f"delta_ll[{name}]", iv.mean, len(g), iv.lo, iv.hi))
            row |= {f"{name}": iv.mean, f"{name} lower": iv.lo, f"{name} upper": iv.hi}
        metrics.append(
            _metric(label, "thin_race_share", row["thin races"] / len(races), len(races))
        )
        table.append(row)
    return metrics, {"by_class": table}


# --------------------------------------------------------------------------- the pure study


@dataclasses.dataclass
class Inputs:
    level1: pd.DataFrame  # with RATES, PRIOR_STARTS and won
    results: pd.DataFrame  # the rating's input (studies.rating.read_results)
    price: pd.DataFrame  # studies.meta.store_frame
    counts: dict


def compute_study(inputs: Inputs, spec: dict, scheme, log=print) -> tuple[Study, pd.DataFrame]:
    """Every variant, and the per-runner export. Pure: no file or registry I/O."""
    level1, price = inputs.level1, inputs.price
    study = Study(counts=dict(inputs.counts))
    fold_scheme = "walk_forward:" + json.dumps(spec["folds"], sort_keys=True)
    comp = {"model": "models/ability.py", "study": "studies/ability.py", "member": POOLED}
    params = {"discount_grid": spec["level1"]["discount_grid"], "thin_starts": THIN_STARTS}

    quarters = level1_quarters(level1["race_date"], scheme)
    outer = [q for q in quarters if q.test_start >= LEVEL1_FROM]
    if quarters[0].test_start != INNER_FROM or outer[0].test_start != LEVEL1_FROM:
        raise ValueError("the level-1 quarters do not start at the signed dates")
    study.counts |= {"level1_quarters": len(quarters), "level1_outer_folds": len(outer)}
    log(f"rating grid over {len(inputs.results)} starters")
    members, _ = c2_rating.member_grid(inputs.results, level1)
    picks = select_pairs(level1, members, outer)
    needed = needed_pairs(outer, picks, quarters)
    members = {pair: members[pair] for pair in needed}
    log(f"level-1 grid: pairs {list(needed)}, {len(DISCOUNT_GRID)} discounts")
    grid = level1_grid(level1, members, quarters, needed, log=log)
    study.counts |= {"level1_fits": grid.fits, "level1_grid_seconds": round(grid.seconds)}
    selection = select_discounts(level1, grid, outer, picks)
    study.tables["selection"] = selection

    def selection_metrics() -> tuple[list, dict]:
        metrics = []
        for row in selection:
            n = row["inner races"]
            for key in (
                "drift_var_per_day",
                "scale",
                "pooled w2",
                "pooled w3",
                "rates w2",
                "rates w3",
            ):
                metrics.append(_metric(row["fold"], f"selected[{key}]", row[key], n))
            metrics.append(
                _metric(row["fold"], "inner_log_loss[pooled]", row["pooled inner log loss"], n)
            )
            metrics.append(
                _metric(row["fold"], "inner_log_loss[rates]", row["rates inner log loss"], n)
            )
        return metrics, {}

    study.attempt(
        Variant(
            "E33",
            "selection:inner_fold",
            fold_scheme,
            comp | {"selection": "inner_fold_log_loss"},
            params,
        ),
        selection_metrics,
    )
    u, se, fold_of = level1_members(level1, grid, outer, selection)
    del grid
    thin = (level1[list(PRIOR_STARTS)].to_numpy() < THIN_STARTS).any(axis=1)
    study.counts["level1_thin_starters"] = int(thin.sum())
    export = level1_export(level1, u, se, fold_of, selection)

    folds2 = walk_forward(price["race_date"], scheme=scheme)
    log(f"D4 baseline on {len(folds2)} level-2 folds")
    price = price.assign(p_cal=baseline_oof(price, folds2))
    rows, member_counts = member_frame(price, level1, u, thin, outer[0].test_start)
    study.counts |= member_counts
    rfolds = restrict(folds2, rows.index.to_numpy(), len(price))
    rows = rows.reset_index(drop=True)
    rows["race_class"] = rows["race_id"].map(level1.groupby("race_id")["race_class"].first())
    scored = Scored.of(rows, rfolds)
    study.counts |= {"scored_folds": len(rfolds), "scored_races": scored.n_races}
    thin_races = pd.Index(scored.rows.loc[scored.rows["thin"].astype(bool), "race_id"].unique())
    study.counts |= {"scored_thin_races": len(thin_races)}
    base = comp | {"score": "winner_log_loss", "baseline": "calibrated_market"}

    study.attempt(
        Variant("E33", "correlation:ln_q", fold_scheme, comp, params),
        lambda: _correlation(scored, spec),
    )
    study.attempt(
        Variant("E33", f"member_alone:{POOLED}", fold_scheme, base, params),
        lambda: _alone(scored, spec),
    )
    meta_params = params | {"ridge": 0.0, "bounds": "none", "quadratic": False, "segments": []}
    outcomes, p = {}, {}
    for member in MEMBERS:
        outcomes[member] = _try(lambda m=member: _meta_oof(rows, rfolds, STAGE2[m], "ln_q"))
        pooled = _try(lambda m=member: _pooled(scored, STAGE2[m], "ln_q"))
        name = f"stage2:q:{member}"
        components = base | {"member": member, "form": "stage2", "market": "ln_q"}
        hypothesis = "E34" if member == RATED else "E33"
        study.attempt(
            Variant(hypothesis, name, fold_scheme, components, meta_params),
            lambda o=outcomes[member], pl=pooled, nm=name: _meta_metrics(
                o, pl, rfolds, scored, spec, nm
            ),
        )
        clustered = meta_params | {
            "inference": "meeting-clustered sandwich (models.condlogit), no small-sample factor",
            "level": "95% one-sided per bound (D7)",
        }
        study.attempt(
            Variant(
                hypothesis,
                f"{CLUSTERED}:{name}",
                fold_scheme,
                components | {"se": CLUSTERED},
                clustered,
            ),
            lambda o=outcomes[member], pl=pooled: _clustered_metrics(o, pl, rfolds, spec),
        )
        if outcomes[member].error is None:
            p[member] = outcomes[member].value[0]

    market = _try(lambda: _market_only(rows, rfolds, "ln_q"))
    market_name = f"stage2:q:{MARKET_ONLY}"
    study.attempt(
        Variant(
            "E33",
            market_name,
            fold_scheme,
            base | {"member": "none", "form": "stage2", "market": "ln_q"},
            meta_params,
        ),
        lambda: _meta_metrics(market, None, rfolds, scored, spec, market_name),
    )

    def need(*names):
        missing = [n for n in names if n not in p]
        if missing:
            raise RuntimeError(f"the stage-2 meta of {missing} failed")

    def e33():
        need(POOLED, WIN_ONLY)
        return _comparison(
            "E33: stage-2 top-3 vs stage-2 winner-only", p[POOLED], p[WIN_ONLY], scored, spec
        )

    def e33_level1():
        r = scored.rows
        return _comparison(
            "E33 (level 1): member top-3 vs member winner-only",
            np.exp(r[POOLED]),
            np.exp(r[WIN_ONLY]),
            scored,
            spec,
        )

    def e34_thin():
        need(POOLED, RATED)
        return _comparison(
            "E34: stage-2 pooled vs stage-2 rates, thin races",
            p[POOLED],
            p[RATED],
            scored,
            spec,
            thin_races,
        )

    def e34_all():
        need(POOLED, RATED)
        return _comparison(
            "E34: stage-2 pooled vs stage-2 rates, every race", p[POOLED], p[RATED], scored, spec
        )

    def marginal():
        need(POOLED)
        return c4_marginal(p[POOLED], rows, rfolds, scored, spec)

    paired = base | {"score": "paired_winner_log_loss"}
    study.attempt(
        Variant(
            "E33",
            f"marginal:stage2:{POOLED}_vs_{MARKET_ONLY}",
            fold_scheme,
            paired | {"comparator": f"stage2:q:{MARKET_ONLY}"},
            params,
        ),
        marginal,
    )
    study.attempt(
        Variant("E33", "E33:stage2:top3_vs_winner_only", fold_scheme, paired, params), e33
    )
    study.attempt(
        Variant("E33", "E33:level1:top3_vs_winner_only", fold_scheme, paired, params), e33_level1
    )
    study.attempt(
        Variant("E34", "E34:stage2:pooled_vs_rates:thin", fold_scheme, paired, params), e34_thin
    )
    study.attempt(
        Variant("E34", "E34:stage2:pooled_vs_rates:all", fold_scheme, paired, params), e34_all
    )

    def by_class():
        need(POOLED, WIN_ONLY, RATED)
        return _by_class(scored, p, spec, thin_races)

    study.attempt(Variant("E34", "by_class", fold_scheme, paired, params), by_class)
    return study, export


def level1_export(level1, u, se, fold_of, selection) -> pd.DataFrame:
    """The pooled member per level-1 out-of-fold runner: race_id, horse_id, race_date,
    oof_fold, the fold's rating pair and discounts, u, se_u and p_ability (softmax of u within
    the race's starters)."""
    keep = ~np.isnan(u[POOLED])
    out = level1.loc[keep, ["race_id", "horse_id", "race_date"]].reset_index(drop=True)
    out["oof_fold"] = pd.Series(fold_of[keep], dtype="string")
    picks = pd.DataFrame(selection).set_index("fold")
    for col in ("drift_var_per_day", "scale", "pooled w2", "pooled w3"):
        out[col.replace("pooled ", "")] = out["oof_fold"].map(picks[col]).astype(float)
    v = u[POOLED][keep]
    race = out["race_id"].to_numpy()
    top = pd.Series(v).groupby(race).transform("max").to_numpy()
    e = np.exp(v - top)
    out["u"] = v
    out["se_u"] = se[keep]
    out["p_ability"] = e / pd.Series(e).groupby(race).transform("sum").to_numpy()
    return out


# --------------------------------------------------------------------------- the run


def _progress(started: float) -> str:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    return f"{time.monotonic() - started:.0f} s, peak RSS {rss:.2f} GB"


def load_inputs(log=print) -> Inputs:
    runners, races = read_runners()
    base, level1, counts = level1_frames(runners, races, read_sires())
    del runners, races
    level1 = pd.concat([level1, connection_features(base, level1)], axis=1)
    del base
    results = c2_rating.read_results()
    p_runners, p_races = read_development()
    price, price_counts = store_frame(p_runners, p_races)
    del p_runners, p_races
    counts |= {f"level2_{k}": v for k, v in price_counts.items()}
    log(f"read {len(level1)} level-1 starters, {len(price)} priced starters")
    return Inputs(level1=level1, results=results, price=price, counts=counts)


SECTIONS = (
    ("selection", "Selection per outer fold (inner-fold winner log loss per race)"),
    ("correlation", "Correlation of the pooled member's ln p with ln q_final (scored runners)"),
    ("pseudo_r2", "Standalone pseudo-R2 against equal chances, beside the final price"),
    (
        "delta",
        "ΔLL against the D4 calibrated market (out of fold; meeting bootstrap, 95% one-sided)",
    ),
    (
        "comparisons",
        "The C4 marginal, E33 and E34: paired ΔLL (positive: the first beats the second)",
    ),
    ("coefficients", "Stage 2 per fold (unbounded; Wald 95% two-sided and meeting one-sided 95%)"),
    ("pooled", "Stage-2 alpha pooled over the scored rows (in-sample)"),
    ("by_class", "By race class (scored races; meeting bootstrap, 95% one-sided)"),
)


def render_report(study: Study, spec: dict, about: dict, record_ids: dict[str, str]) -> str:
    """The full report: aggregate tables only, never a runner or race row."""
    lines = [
        "# C2 ability (top-3 Plackett-Luce, pooled entities) at the final Win price "
        f"({COUNTRY}, #{TICKET})",
        "",
    ]
    lines += [
        "Built on JRA-VAN data (the sire): every figure here is withheld from the repository "
        "(D21).",
        "",
    ]
    lines += [*(f"- {k}: {v}" for k, v in about.items()), ""]
    lines += [
        "## Decision rules (frozen)",
        "",
        *(f"- {k}: {v}" for k, v in spec["decision_rules"].items()),
    ]
    lines += [
        "",
        "## Data",
        "",
        _markdown_table([{"count": k, "n": v} for k, v in study.counts.items()]),
    ]
    for key, title in SECTIONS:
        rows = study.tables.get(key)
        if key == "pooled" and rows:  # studies.meta labels the pooled member coefficient delta
            rows = [{("alpha" if k == "delta" else k): v for k, v in r.items()} for r in rows]
        lines += ["", f"## {title}", "", _markdown_table(rows) if rows else "_Not computed._"]
    lines += [
        "",
        "## N5",
        "",
        "The disagreement-tail curve (N5) is added when N5 lands (owner, 2026-11-27).",
    ]
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
    """The file stem of a run's report, its JSON and its export: the spec hash and a digest of
    the data builds (as a mapping: key order does not matter), "-trial" for a trial."""
    digest = hashlib.sha256(
        json.dumps(dict(builds), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"ability-{COUNTRY}-{spec_hash[:12]}-{digest[:12]}" + ("-trial" if trial else "")


def _record(
    variant: Variant,
    git_sha: str,
    builds: dict,
    spec_hash: str,
    trial: bool,
    *,
    artifacts: bool = True,
) -> RunRecord:
    """The variant's registry record; `artifacts=False` for a run that aborted before writing
    its report and export, whose record names no files (a retry must not point to an earlier
    run's)."""
    params = dict(variant.params) | {"spec_sha256": spec_hash}
    if trial:
        params["trial"] = True
    if variant.error:
        params["error"] = variant.error
    data_builds = dict(builds)
    if artifacts:  # the run's files after the '@': a data refresh is another run (#74)
        stem = artifact_stem(spec_hash, builds, trial)
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


class _Families:
    """The registry restricted to some families, for `write_summary`."""

    def __init__(self, registry, families: set[str]) -> None:
        self._registry, self._families = registry, families

    def records(self, family_id: str | None = None):
        return [r for r in self._registry.records(family_id) if r.family_id in self._families]


_INDEX_ENTRY = re.compile(r"- \[([^\]]+)\]\(\1\.md\)")  # a family's line in write_summary's index


def write_committed_summary(registry) -> None:
    """This study's family files, by `write_summary` over this study's families of `registry`,
    and the committed index with their lines added: every family the index lists stays, and no
    other family's file is written, so a fresh local registry drops nothing and no other
    ticket's uncommitted family enters (nor #131's '/' ids, which `write_summary` refuses)."""
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


PLANNED = (
    ("E33", "selection:inner_fold"),
    ("E33", "correlation:ln_q"),
    ("E33", f"member_alone:{POOLED}"),
    *(
        (h, f"{pre}stage2:q:{m}")
        for m, h in ((POOLED, "E33"), (WIN_ONLY, "E33"), (RATED, "E34"))
        for pre in ("", f"{CLUSTERED}:")
    ),
    ("E33", f"stage2:q:{MARKET_ONLY}"),
    ("E33", f"marginal:stage2:{POOLED}_vs_{MARKET_ONLY}"),
    ("E33", "E33:stage2:top3_vs_winner_only"),
    ("E33", "E33:level1:top3_vs_winner_only"),
    ("E34", "E34:stage2:pooled_vs_rates:thin"),
    ("E34", "E34:stage2:pooled_vs_rates:all"),
    ("E34", "by_class"),
)


def run_study(*, trial: bool = False, out_dir: Path | None = None, log=print) -> dict:
    """Run the pre-registered family on Japan's development data; write the private report,
    the export and the registry records."""
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    scheme = load_country_scheme(COUNTRY)
    spec = build_spec(scheme, {**bootstrap_settings(), "seed": BOOTSTRAP_SEED})
    spec_hash = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    registry = default_registry(COUNTRY)
    if not trial:  # before any outcome is read; a trial's spec may still change
        spec_hash = registry.freeze_family(FAMILY_ID, spec)
    family = TRIAL_FAMILY_ID if trial else FAMILY_ID
    git_sha = _git_sha()
    builds = {**store_builds(), **pedigree_build()}
    stem = artifact_stem(spec_hash, builds, trial)  # named in every record of a completed run
    started = time.monotonic()
    try:
        inputs = load_inputs(log=log)
        study, export = compute_study(inputs, spec, scheme, log=log)
    except Exception as exc:
        fold_scheme = "walk_forward:" + json.dumps(spec["folds"], sort_keys=True)
        for hypothesis, name in PLANNED:
            failed = Variant(
                hypothesis,
                name,
                fold_scheme,
                {"study": "studies/ability.py"},
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
        else:  # the same run reproduced at another commit
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
    export_path.parent.mkdir(parents=True, exist_ok=True)
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
        "record_ids": {name: rid[:16] for name, rid in record_ids.items()},
        "failed_variants": [v.name for v in study.variants if v.error],
    }


def ability_study_command(
    trial: Annotated[
        bool,
        typer.Option(
            "--trial",
            help=f"Records go to {TRIAL_FAMILY_ID} marked trial; the spec is not frozen; the "
            "report is suffixed -trial.",
        ),
    ] = False,
) -> None:
    """Fit the C2 ability model (E33, E34) on D9's folds and write the private report."""
    typer.echo(json.dumps(run_study(trial=trial, log=typer.echo), indent=2))
