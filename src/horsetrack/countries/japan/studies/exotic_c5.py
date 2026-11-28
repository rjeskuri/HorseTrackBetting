"""The C5 exotic layer against the pools' own dividends (#459, epic #73; 08 §4 E43, E44, E16):
`horsetrack japan exotic-c5`.

C5 (`models/exotic.py`, #99) turns one race's Win probabilities into every pool's combination
probabilities through an ordering. This study scores it, per pool, against the dividend-implied
market of the pool itself (08 R15): the share of the pool's money on each paid combination,
`exotic.dividend_shares` on the store's dividends at the race's own payout rate (boosts in), the
statute de-vig for several winning combinations (owner answer 3(a) on #99). Scoring is the log
loss at 1/k per winning combination (#69).

- Inputs (owner answers 2 and 4): the normalised final Win price (1/odds over the race's priced
  starters), D4's calibrated market (`models.v0.calibration.fit_win_calibration`, refitted on each
  fold's training rows), and #87's `stage2:q:form` member (`load_form`, rebuilt from #87's own
  code). The member reads JRDB, so every result built on it is a licensed record: withheld from
  the committed summary and the public report (D21), written in full only to the private report.
- Orderings: every family of `exotic.FAMILIES`, each a variant; the fitted ones (quadratic
  context, discounted Harville) are refitted on each fold's training races (`Ordering.fit`),
  through the input's own Win probabilities.
- Blends (owner answer 6): none; 08 E43's linear mixture alpha q + (1 - alpha) P_C5 on each paid
  combination, alpha in [0, 1] fitted per pool, input and ordering on each fold's training races;
  and `exotic.pool_blend`'s log-linear pool, recorded as not runnable (no full final boards).
- Place and Wide: each de-vig of PLACE_WIDE_DEVIGS crossed with the floor treatments 'drop' and
  'bound' (the owner's comment on #459, 2026-11-27): at the 110-yen minimum the share is censored.
- E43: ΔLL against the dividends per pool, D7's one-sided meeting-bootstrap bounds. E16: the same
  grid on the race-pools that paid the super_premium rate (owner answer 5). E44 (owner answer 1):
  Trio and Trifecta, one ordering fitted once on the market and serving the market and the model;
  the joint and the Win gain each with D7 bounds, their ratio a point estimate.
- Leaderboard: every no-blend E43 and E16 variant carries `params['leaderboard']` and the return
  per ¥100 of a flat ¥100 on the model's most probable combination, settled at the paid dividends
  (`docs/specs/leaderboard.md`'s convention), so each pool's result is a ranked row.

- The second family, `PER_TICKET_FAMILY_ID` (`build_per_ticket_spec`; the owner's reframe on #99
  of 2026-11-27 registers each candidate de-vig, the review of PR #668): Place and Wide alone
  against `PER_TICKET`, each paid ticket's own implied share min(1, R*100/(D_c+5)), with every
  input, ordering, blend and floor treatment; its public report sets each variant beside its
  statute twin of the first family (`statute_twins`). The first family's spec is unchanged.

`run_study` checks the family and the output folder, freezes the family's spec in the run
registry before either seam reads a row, reads development days only (`is_holdout_day`; the
holdout is never opened), scores, appends one registry record per variant (failures included)
and writes the private report through `write_report`. With the default registry it also writes
the family's committed summary and its public report (`render_public_report`), which hold no
licensed figure. The pure part is `compute_study`. Pre-registration (both families):
`docs/specs/exotic-c5-prereg.md`.

Memory and time: one ordering family at a time; per race one trifecta per input and fold
parameters, kept only while that family is scored; the blend's alpha is fitted on at most
`BLEND_MAX_TRAIN` training races per fold (a fixed hash sample), so Henery's quadrature is not
run on every training race of every fold.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import time as clock
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import typer
from scipy.optimize import brentq

from horsetrack.core import env, paths
from horsetrack.core.eval.bootstrap import Interval, paired_bootstrap
from horsetrack.core.eval.folds import Fold, FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day
from horsetrack.core.eval.registry import (
    RunRecord,
    default_registry,
    is_licensed_record,
    summary_dir,
    write_summary,
)
from horsetrack.core.eval.reports import write_report
from horsetrack.core.pricing import clean_odds
from horsetrack.countries.japan import takeout
from horsetrack.countries.japan.models import exotic
from horsetrack.countries.japan.models.exotic import Ordering
from horsetrack.countries.japan.models.meta import MetaSpec
from horsetrack.countries.japan.models.v0.calibration import WinCalibration, fit_win_calibration
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table
from horsetrack.countries.japan.research.leaderboard_backfill import payout_rate as board_rate
from horsetrack.countries.japan.studies import ability as c2_ability
from horsetrack.countries.japan.studies import meta as c4
from horsetrack.countries.japan.studies.calibration import (
    Study,
    Variant,
    _markdown_table,
    _metric,
)
from horsetrack.countries.japan.studies.calibration import (
    bootstrap_settings as d7_bootstrap,
)

COUNTRY = "japan"
TICKET = 459
FAMILY_ID = "japan-c5-exotic-dividends"
SEED = TICKET  # the bootstrap's seed: reproducibility only, scoring.yaml signs none
WIN, PLACE, QUINELLA, WIDE = exotic.WIN, exotic.PLACE, exotic.QUINELLA, exotic.WIDE
EXACTA, TRIO, TIERCE = exotic.EXACTA, exotic.TRIO, exotic.TRIFECTA
# Pinned to the frozen spec's pools, in its order: not exotic.POOLS, which grows. The bracket
# quinella (#513) is a later C5 family's pool.
POOLS = (WIN, PLACE, QUINELLA, WIDE, EXACTA, TRIO, TIERCE)
ORDERINGS = exotic.FAMILIES
NORMALISED, CALIBRATED, FORM = "normalised", "calibrated", "stage2:q:form"
INPUTS = (NORMALISED, CALIBRATED, FORM)
E44_POOLS = (TRIO, TIERCE)
STATUTE, PER_TICKET = "statute", "per_ticket"
PLACE_WIDE_DEVIGS = (STATUTE,)  # the first family's (frozen in default_spec)
DEVIGS = (STATUTE, PER_TICKET)  # every Place and Wide de-vig compute_study knows
PER_TICKET_FAMILY_ID = "japan-c5-exotic-dividends-per-ticket"
FLOOR_TREATMENTS = ("drop", "bound")
NO_BLEND, LINEAR, LOG_LINEAR = "none", "linear", "log_linear"
BLENDS = (NO_BLEND, LINEAR, LOG_LINEAR)
MULTI = (PLACE, WIDE)  # several winning combinations on every result
FITTED = ("quadratic_context", "discounted_harville")  # the families with parameters
UNORDERED = (QUINELLA, WIDE, TRIO)
SIZE = {WIN: 1, PLACE: 1, QUINELLA: 2, WIDE: 2, EXACTA: 2, TRIO: 3, TIERCE: 3}
N_KNOTS = 5  # fit_win_calibration's default, D4's live model
BLEND_MAX_TRAIN = 2000  # training races per fold the blend's alpha is fitted on, at most
STORE_SOURCES = ("jra",)  # the store's JRA results and dividends, from netkeiba
FORM_SOURCES = ("jra", "jrdb")  # #87's member reads JRDB's indices (D21)
SOURCE = "jra"
ALL_FOLDS, DEVELOPMENT = "all_folds", "development"
DELTA, ALPHA, RETURN = "delta_ll_vs_dividends", "alpha", "return_per_100"
SUPER = "super_premium"
FAILED_ORDERING = "races_ordering_failed"
MEMBER_FILE = c4.MEMBER_FILE
PUBLIC_REPORT = Path("docs") / "experiments" / "exotic-c5-japan.md"
PER_TICKET_PUBLIC_REPORT = Path("docs") / "experiments" / "exotic-c5-per-ticket-japan.md"
SPEC_DOC = "docs/specs/exotic-c5-prereg.md"
LOG_LINEAR_ERROR = (
    "not runnable yet: exotic.pool_blend's log-linear pool needs the full final board of the "
    "pool's combinations, which only the odds logs of #4/#21 hold; the store holds the paid "
    "dividends alone (owner answer 6 on #99)"
)
TAG = {
    WIN: "win",
    PLACE: "place",
    QUINELLA: "quinella",
    WIDE: "wide",
    EXACTA: "exacta",
    TRIO: "trio",
    TIERCE: "trifecta",
    NORMALISED: "norm",
    CALIBRATED: "cal",
    FORM: "form",
    None: "-",
    "harville": "harv",
    "henery": "hen",
    "quadratic_context": "qc",
    "discounted_harville": "dh",
    NO_BLEND: "none",
    LINEAR: "lin",
    LOG_LINEAR: "loglin",
    STATUTE: "stat",
    PER_TICKET: "tick",
}
RACE_COLUMNS = ["race_id", "race_date", "meeting_id", "race_class_raw", "field_size", "status"]
RUNNER_COLUMNS = [
    "race_id",
    "race_date",
    "horse_no",
    "win_odds",
    "finish_pos",
    "dead_heat",
    "is_scratched",
]
PAID_COLUMNS = ["race_id", "pool", "combination", "dividend", "is_refund"]

DECISIONS = {
    "D4": "the calibrated market is D4's spline in ln q, refitted on each fold's training rows",
    "D5": "the exotic market baseline is the pool's own paid dividends (08 R15, R16)",
    "D7": "meeting (venue x day) clusters, 95% one-sided (config/scoring.yaml)",
    "D9": "walk-forward quarters, expanding, 1-week embargo (config/folds.yaml)",
    "D10": "no price before 2014-06-07",
    "D11-D15": "holdout A2 from 2026-11-28: no holdout day is read; the holdout is not opened",
    "D16-D19": "nothing is adopted: every variant is a ranked row (reframe of 2026-11-27)",
    "D21": "stage2:q:form reads JRDB: its records are licensed and its figures withheld",
    "#99 answer 1": "E44 option (b): the joint and the Win gain each with D7 bounds, the ratio a "
    "point estimate; one ordering fitted once serves both inputs; Trio and Trifecta only",
    "#99 answer 2": "the publishable pair is D4's calibrated market against normalised 1/odds; "
    "E44 on the JRDB-fed member is reported as ran, withheld",
    "#99 answer 3": "several winning combinations: the statute's de-vig "
    "x_c = 1/((1+K) P (r_c - 1)); scoring at 1/k per winning combination",
    "#99 answer 4": "stage2:q:form alone feeds C5 from the meta-model",
    "#99 answer 5": "E16 on super_premium race-pools only",
    "#99 answer 6": "the blend is 08 E43's linear mixture on the realised combination, alpha in "
    "[0, 1] fitted per pool on each fold's training races; the log-linear pool is recorded as "
    "not runnable yet",
    "#459 comment": "Place and Wide at the 110-yen floor are censored: 'drop' and 'bound' are "
    "separate variants (owner, 2026-11-27)",
}


# --------------------------------------------------------------------------- the spec


def bootstrap_settings() -> dict:
    """D7's bootstrap from config/scoring.yaml, with this study's seed."""
    return {**d7_bootstrap(), "seed": SEED}


def build_spec(scheme: FoldScheme, bootstrap: Mapping) -> dict:
    """The family's pre-registered spec (plain JSON), frozen before any outcome is read. Its grid
    lists are what `compute_study` runs."""
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypotheses": {
            "E43": "C5 (an input's Win probabilities through an ordering, optionally blended with "
            "the pool) beats the dividend-implied share of the pool's own money in log loss on "
            "the paid combinations, per pool: mean ΔLL (dividends minus model) above 0",
            "E16": "the E43 table on the race-pools that paid the super_premium rate (80%)",
            "E44": "through one ordering fitted once on the market, the model's gain over the "
            "market on Trio and Trifecta is a multiple of its Win gain (08: 2-3x)",
        },
        "null": "the dividends are at least as good as C5 in every pool (ΔLL <= 0), and the joint "
        "gain is no larger than the Win gain: C5 adds nothing to the pools' own prices",
        "pools": list(POOLS),
        "orderings": list(ORDERINGS),
        "inputs": list(INPUTS),
        "blends": list(BLENDS),
        "e44_pools": list(E44_POOLS),
        "place_wide_devigs": list(PLACE_WIDE_DEVIGS),
        "floor_treatments": list(FLOOR_TREATMENTS),
        "rows": "run races of development days with at least 3 priced starters (not scratched, "
        "a decimal Win price of at least 1.0, a horse number); unpriced starters are left out "
        "and counted; a race-pool is scored when it pays, holds no refund row, has no dividend "
        "under 100 yen, and each of its paid combinations names starters of the race",
        "folds": "walk_forward over the races' dates under fold_scheme; a scored race is in a "
        "fold's test window; a fitted model (D4's spline, an ordering, alpha) sees only the "
        "fold's training races (dates up to its train_end)",
        "baseline": {
            "share": "exotic.dividend_shares on the race-pool's dividends at the race's own payout "
            "rate (takeout.race_takeout, boosts and race_class_raw in): R*100/(D+5) for one "
            "winning combination, the statute's de-vig for several",
            "single_winner_pools": "the chance of a paid combination is its share",
            "place_wide": {
                "statute": "the chance a paid ticket pays is P x its statute share (P the paid "
                "combinations, the places on a clean result), capped at 1: the money on a "
                "ticket taken as proportional to its chance of paying",
            },
            "floor": {
                "flag": "exotic.dividend_floor_bound: a race-pool with a dividend at or under its "
                "pool's 110-yen minimum",
                "drop": "flagged Place and Wide race-pools are left out of fitting and scoring",
                "bound": "every race-pool is scored, the floored share as computed",
            },
        },
        "score": "per scored race-pool, the log loss -(1/k) sum ln p(c) over its k paid "
        "combinations (#69); ΔLL = the baseline's log loss minus the model's",
        "inputs_defined": {
            NORMALISED: "q_i = (1/O_i) / sum_j (1/O_j) over the race's priced starters",
            CALIBRATED: "models.v0.calibration.fit_win_calibration (n_knots "
            f"{N_KNOTS}) on each fold's training starters (won: finish_pos 1, each co-winner of "
            "a dead heat), applied to q; in-sample on the training races where an ordering or "
            "alpha is fitted",
            FORM: "#87's stage2:q:form out-of-fold P(win) (studies.meta: HKTrack's form member "
            f"{MEMBER_FILE} with ln q, stage 2, ridge 0; scored from 2024-01-01), renormalised "
            "over the race's starters; a race with a starter it lacks is not scored for it",
        },
        "model": "P_C5: the input's Win probabilities (renormalised per race) through the "
        "ordering's trifecta (models.exotic.Ordering), pool marginals by exotic.pool_array, "
        "Place's places by the field at sale (the race's declared runners, at least "
        "races.field_size); WIN is the input itself (no ordering)",
        "ordering_fit": "quadratic_context and discounted_harville: Ordering.fit on each fold's "
        "training races through the input's own Win probabilities (04 §2.3); a fold in which the "
        "input has no training race is not scored for those families and is counted; harville "
        "and henery have no parameters. A race whose ordering raises is left out of that "
        f"variant and counted ({FAILED_ORDERING}); a variant with no scored race fails",
        "blend": {
            "linear": "p = alpha q + (1 - alpha) P_C5 on each paid combination, alpha in [0, 1] "
            "minimising the mean log loss over the fold's training race-pools of that pool, "
            "input, ordering, de-vig and floor treatment (P_C5 with the fold's parameters), by "
            "the convex objective's derivative (0 or 1 at the bounds); one alpha per fold",
            "max_train_races": BLEND_MAX_TRAIN,
            "sample": "when a fold has more training races, the BLEND_MAX_TRAIN whose "
            "sha256(race_id) is smallest",
            "log_linear": LOG_LINEAR_ERROR,
        },
        "e16": "the E43 per-race ΔLL (each fold's fits and alpha as in E43) restricted to the "
        "race-pools whose payout rate was exactly super_premium (super_premium_subset: "
        "takeout.race_takeout(...).boost == 'super_premium'; ultra_premium and premium out)",
        "e44": "for each ordering and each input but normalised: market normalised. The "
        "ordering's parameters are fitted once per fold on the market (normalised) and serve "
        "both inputs. Races: scored for both inputs, WIN and every e44 pool paying. win_gain: "
        "the market's Win log loss minus the model's; joint_gain[pool]: the market's pool log "
        "loss minus the model's; each with D7 bounds; gain_ratio[pool] = joint / win, a point "
        "estimate",
        "leaderboard": {
            "bet": "a flat ¥100 per scored race-pool on the model's most probable combination "
            "of the pool (the first in horse-number order on a tie), returning the race-pool's "
            "paid dividend on it when it is a paying combination, else 0",
            "return_per_100": "the mean over the variant's scored race-pools (E16: "
            "return_per_100[super_premium]), D7 meeting-bootstrap one-sided bounds, n in bets",
            "selection": "look_ahead: the inputs are the final Win prices",
            "linear_blend": "no return: the blend is defined only on the paid combinations, so "
            "it has no most probable combination without the full board",
            "payout_rate": "the pool's base payout rate as the leaderboard backfill reads it",
        },
        "fold_scheme": dataclasses.asdict(scheme),
        "bootstrap": dict(bootstrap),
        "price_era_start": PRICE_ERA_START.isoformat(),
        "decisions": DECISIONS,
        "sources": {"store": list(STORE_SOURCES), FORM: list(FORM_SOURCES)},
        "adoption": "none: every variant is a leaderboard row",
    }


def default_spec() -> dict:
    """The spec `run_study` freezes: Japan's fold scheme, D7's bootstrap, the full grid."""
    return build_spec(load_country_scheme(COUNTRY), bootstrap_settings())


def build_per_ticket_spec(scheme: FoldScheme, bootstrap: Mapping) -> dict:
    """The second family's spec (plain JSON, `build_spec`'s keys): Place and Wide against the
    per-ticket de-vig, every ordering, input, blend and floor treatment, no E44. The owner's
    reframe on #99 (2026-11-27) registers each candidate Place and Wide de-vig as a variant; the
    first family's spec is frozen, so this de-vig is a family of its own (review of PR #668)."""
    spec = build_spec(scheme, bootstrap)
    chance = (
        "the chance a paid ticket c pays is its own implied share min(1, R*100/(D_c+5)) (R the "
        "race's payout rate, takeout.race_takeout, boosts and race_class_raw in; D_c its dividend "
        "per 100 yen, the 5-yen mean breakage back): the money on c against what c pays, as in a "
        "pool with one winner; P times #99 question 3(b)'s equal split R*100/(P (D_c+5))"
    )
    spec.update(
        family_id=PER_TICKET_FAMILY_ID,
        hypotheses={
            "E43": "C5 (an input's Win probabilities through an ordering, optionally blended with "
            "the pool) beats the per-ticket dividend baseline in log loss on the paid "
            "combinations of Place and Wide, per pool: mean ΔLL (dividends minus model) above 0",
            "E16": "the E43 table on the race-pools that paid the super_premium rate (80%)",
        },
        null="the per-ticket dividends are at least as good as C5 in Place and in Wide (ΔLL <= "
        "0): C5 adds nothing to the pools' own prices",
        pools=list(MULTI),
        e44_pools=[],
        place_wide_devigs=[PER_TICKET],
        e44=f"none in this family: E44 (Trio and Trifecta) is {FAMILY_ID}'s",
        decisions=DECISIONS
        | {
            "#99 reframe": "each candidate Place and Wide de-vig is a registered variant (owner, "
            f"2026-11-27): {FAMILY_ID} holds the statute's, this family the per-ticket share, "
            "with the first family's spec unchanged (review of PR #668)"
        },
    )
    spec["baseline"] = {**spec["baseline"], "place_wide": {PER_TICKET: chance}}
    spec["leaderboard"] = {
        **spec["leaderboard"],
        "devig": "the bet and its return do not depend on the de-vig: each no-blend row's return "
        f"equals its statute twin's in {FAMILY_ID}",
    }
    return spec


def default_per_ticket_spec() -> dict:
    """The spec `run_study(family=PER_TICKET_FAMILY_ID)` freezes."""
    return build_per_ticket_spec(load_country_scheme(COUNTRY), bootstrap_settings())


def family_spec(family: str) -> dict:
    """The spec `run_study` freezes for `family`; ValueError for a family it does not run."""
    if family == FAMILY_ID:
        return default_spec()
    if family == PER_TICKET_FAMILY_ID:
        return default_per_ticket_spec()
    raise ValueError(f"unknown family {family!r}: {FAMILY_ID} or {PER_TICKET_FAMILY_ID}")


# --------------------------------------------------------------------------- the store, prepared


@dataclass(frozen=True)
class _Paid:
    """One race-pool's paid combinations: positions into the race's starters, dividends, the
    dividend baseline's chance of each and its log loss (per Place and Wide de-vig; None for a
    single-winner pool), and whether the 110-yen floor bound it."""

    keys: tuple[tuple[int, ...], ...]
    index: tuple[np.ndarray, ...]
    dividends: np.ndarray
    base: dict[str | None, np.ndarray]
    base_ll: dict[str | None, float]
    floor: bool

    def payout(self, key: tuple[int, ...]) -> float:
        for k, d in zip(self.keys, self.dividends, strict=True):
            if k == key:
                return float(d)
        return 0.0


@dataclass
class _Data:
    ids: np.ndarray  # race_id per race, sorted by (race_date, race_id)
    days: list[date]
    meetings: np.ndarray
    labels: list[str | None]
    fields: np.ndarray  # runners when sales opened
    lo: np.ndarray  # each race's starter rows [lo, hi)
    hi: np.ndarray
    starters: pd.DataFrame  # race_id, race_date, horse_no, win_odds, finish_pos, dead_heat, won
    q: np.ndarray  # normalised 1/odds per starter
    paid: list[dict[str, _Paid]]
    folds: list[Fold]
    form: np.ndarray | None  # stage2:q:form per starter, renormalised; NaN where missing
    form_ok: np.ndarray | None  # per race
    hashes: np.ndarray
    clusters: pd.Series
    counts: dict[str, int]
    form_counts: dict[str, int] = field(default_factory=dict)

    def horses(self, pos: int) -> np.ndarray:
        return self.starters["horse_no"].to_numpy()[self.lo[pos] : self.hi[pos]]


def _as_dates(values) -> list[date]:
    return [pd.Timestamp(v).date() for v in values]


def _flags(series: pd.Series) -> np.ndarray:
    return series.astype("boolean").fillna(False).to_numpy(dtype=bool)


def _objects(series: pd.Series) -> np.ndarray:
    return np.array([None if pd.isna(x) else str(x) for x in series], dtype=object)


def _floats(series: pd.Series) -> np.ndarray:
    """A column as float64, a missing or unparseable value NaN (nullable integers included)."""
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(values, dtype="Float64").to_numpy(dtype=float, na_value=np.nan)


def _prepare(races: pd.DataFrame, runners: pd.DataFrame, dividends: pd.DataFrame, spec: dict,
             form: pd.DataFrame | None) -> _Data:  # fmt: skip
    """The study's races, starters and race-pools from store-shaped tables (module docstring)."""
    counts: dict[str, int] = {"races_read": len(races), "runners_read": len(runners)}
    status = _objects(races["status"]) if "status" in races else np.full(len(races), "run")
    run = status == "run"
    counts["races_not_run"] = int((~run).sum())
    r = races.loc[run]
    rid = _objects(r["race_id"])
    rday = _as_dates(r["race_date"])
    labels = (
        _objects(r["race_class_raw"]) if "race_class_raw" in r else np.full(len(r), None, object)
    )
    sizes = _floats(r["field_size"]) if "field_size" in r else np.full(len(r), np.nan)
    meeting = _objects(r["meeting_id"])
    order = sorted(range(len(r)), key=lambda i: (rday[i], rid[i]))
    rid, labels, sizes, meeting = rid[order], labels[order], sizes[order], meeting[order]
    rday = [rday[i] for i in order]
    pos_of = {x: i for i, x in enumerate(rid)}
    if len(pos_of) < len(rid):
        raise ValueError("a race_id is given twice in races")

    ru_id = _objects(runners["race_id"])
    race_pos = np.array([pos_of.get(x, -1) for x in ru_id], dtype=np.int64)
    known = race_pos >= 0
    counts["runners_without_a_run_race"] = int((~known).sum())
    declared = np.bincount(race_pos[known], minlength=len(rid))
    scratched = _flags(runners["is_scratched"])
    odds = clean_odds(_floats(runners["win_odds"]))
    horse = _floats(runners["horse_no"])
    numbered = np.isfinite(horse) & (horse >= 1) & (horse == np.round(np.nan_to_num(horse)))
    starting = known & ~scratched
    counts["scratched_runners"] = int((known & scratched).sum())
    counts["unpriced_starters"] = int((starting & np.isnan(odds)).sum())
    counts["starters_without_a_horse_number"] = int((starting & ~numbered).sum())
    keep = starting & ~np.isnan(odds) & numbered
    s = pd.DataFrame(
        {
            "race_pos": race_pos[keep],
            "horse_no": horse[keep].astype(np.int64),
            "win_odds": odds[keep],
            "finish_pos": _floats(runners["finish_pos"])[keep],
            "dead_heat": _flags(runners["dead_heat"])[keep],
        }
    )
    s = s.iloc[np.lexsort((s["horse_no"].to_numpy(), s["race_pos"].to_numpy()))]
    twice = s.duplicated(["race_pos", "horse_no"], keep=False).to_numpy()
    bad = set(s["race_pos"].to_numpy()[twice].tolist())
    n_start = np.bincount(s["race_pos"].to_numpy(), minlength=len(rid))
    few = set(np.flatnonzero(n_start < 3).tolist())
    counts["races_with_a_horse_number_twice"] = len(bad)
    counts["races_under_3_priced_starters"] = len(few - bad)
    kept = np.array([i not in bad and i not in few for i in range(len(rid))], dtype=bool)
    new = np.cumsum(kept) - 1
    s = s.loc[kept[s["race_pos"].to_numpy()]].reset_index(drop=True)
    s["race_pos"] = new[s["race_pos"].to_numpy()]
    rid, labels, sizes, meeting, declared = (
        rid[kept],
        labels[kept],
        sizes[kept],
        meeting[kept],
        declared[kept],
    )
    rday = [d for d, k in zip(rday, kept, strict=True) if k]
    n = len(rid)
    counts["races"] = n
    counts["starters"] = len(s)
    rp = s["race_pos"].to_numpy()
    lo = np.searchsorted(rp, np.arange(n), side="left")
    hi = np.searchsorted(rp, np.arange(n), side="right")
    inv = 1.0 / s["win_odds"].to_numpy()
    total = np.add.reduceat(inv, lo) if n else np.zeros(0)
    q = inv / np.repeat(total, hi - lo)
    fields = np.maximum(declared, np.nan_to_num(sizes, nan=0.0)).astype(np.int64)
    fields = np.maximum(fields, hi - lo)
    s["race_id"] = rid[rp]
    s["race_date"] = np.array(rday, dtype=object)[rp] if n else np.array([], dtype=object)
    s["won"] = s["finish_pos"].to_numpy() == 1.0
    data = _Data(
        ids=rid,
        days=rday,
        meetings=meeting,
        labels=list(labels),
        fields=fields,
        lo=lo,
        hi=hi,
        starters=s,
        q=q,
        paid=[{} for _ in range(n)],
        folds=[],
        form=None,
        form_ok=None,
        hashes=np.array([int(hashlib.sha256(x.encode()).hexdigest()[:15], 16) for x in rid]),
        clusters=pd.Series(meeting, index=pd.Index(rid, dtype=object), dtype=object),
        counts=counts,
    )
    _attach_paid(data, dividends, spec)
    if n:
        data.folds = walk_forward(
            pd.Series(rday), scheme=FoldScheme.from_mapping(spec["fold_scheme"])
        )
    counts["folds"] = len(data.folds)
    counts["scored_races"] = int(sum(len(f.test_idx) for f in data.folds))
    if form is not None:
        _attach_form(data, form)
    return data


def _race_pool_shares(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`exotic.dividend_shares` and `exotic.dividend_floor_bound` of every row, on the whole
    frame or, when one race-pool makes it raise, group by group: a group that raises is marked
    unusable (NaN share)."""
    try:
        shares = exotic.dividend_shares(frame).to_numpy()
        floors = exotic.dividend_floor_bound(frame).to_numpy()
        return shares, floors, np.zeros(len(frame), dtype=bool)
    except Exception:  # one bad race-pool: settle the frame group by group
        pass
    shares = np.full(len(frame), np.nan)
    floors = np.zeros(len(frame), dtype=bool)
    failed = np.zeros(len(frame), dtype=bool)
    keys = list(zip(frame["race_id"], frame["pool"], strict=True))
    groups: dict[tuple, list[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    for rows in groups.values():
        part = frame.iloc[rows]
        try:
            shares[rows] = exotic.dividend_shares(part).to_numpy()
            floors[rows] = exotic.dividend_floor_bound(part).to_numpy()
        except Exception:
            failed[rows] = True
    return shares, floors, failed


def _attach_paid(data: _Data, dividends: pd.DataFrame, spec: dict) -> None:
    """Each race's paid race-pools (`_Paid`), with the counts of the race-pools left out."""
    counts = data.counts
    pos_of = {x: i for i, x in enumerate(data.ids)}
    d_id = _objects(dividends["race_id"])
    d_pool = _objects(dividends["pool"])
    rpos = np.array([pos_of.get(x, -1) for x in d_id], dtype=np.int64)
    keep = (rpos >= 0) & np.isin(d_pool, list(POOLS))
    refund = (
        _flags(dividends["is_refund"]) if "is_refund" in dividends else np.zeros(len(d_id), bool)
    )
    frame = pd.DataFrame(
        {
            "race_id": d_id[keep],
            "pool": d_pool[keep],
            "combination": _objects(dividends["combination"])[keep],
            "dividend": _floats(dividends["dividend"])[keep],
            "race_class_raw": np.array([data.labels[i] for i in rpos[keep]], dtype=object),
        }
    )
    rpos, refund = rpos[keep], refund[keep]
    shares, floors, failed = _race_pool_shares(frame)
    devigs = [d for d in spec["place_wide_devigs"] if d in DEVIGS]  # another fails its variants
    combos = frame["combination"].to_numpy()
    values = frame["dividend"].to_numpy()
    pools = frame["pool"].to_numpy()
    groups: dict[tuple[int, str], list[int]] = {}
    for i, key in enumerate(zip(rpos.tolist(), pools.tolist(), strict=True)):
        groups.setdefault(key, []).append(i)
    left = {"refund": 0, "special_payout": 0, "unpriced": 0, "not_in_field": 0}
    for (pos, pool), rows in groups.items():
        if refund[rows].any():
            left["refund"] += 1
            continue
        if failed[rows].any() or not np.all(np.isfinite(values[rows])):
            left["unpriced"] += 1
            continue
        if not np.all(np.isfinite(shares[rows])):
            left["special_payout"] += 1  # a dividend under 100 yen: no share of winning money
            continue
        horses = data.horses(pos)
        where = {int(h): i for i, h in enumerate(horses)}
        keys = []
        for text in combos[rows]:
            try:
                numbers = [int(x) for x in str(text).split(",")]
            except ValueError:
                numbers = []
            spots = [where.get(h, -1) for h in numbers]
            if (
                len(spots) != SIZE[pool]
                or min(spots, default=-1) < 0
                or len(set(spots)) < len(spots)
            ):
                keys = None
                break
            keys.append(tuple(sorted(spots)) if pool in UNORDERED else tuple(spots))
        if keys is None or len(set(keys)) < len(keys):
            left["not_in_field"] += 1
            continue
        paid = values[rows].astype(float)
        if pool in MULTI:
            base = {d: _ticket_chances(d, shares[rows], paid, data.ids[pos], pool,
                                       data.labels[pos]) for d in devigs}  # fmt: skip
        else:
            base = {None: shares[rows]}
        index = tuple(np.array([k[a] for k in keys], dtype=np.int64) for a in range(SIZE[pool]))
        data.paid[pos][pool] = _Paid(
            keys=tuple(keys),
            index=index,
            dividends=paid,
            base=base,
            base_ll={d: float(-np.log(b).mean()) for d, b in base.items()},
            floor=bool(floors[rows].any()),
        )
    counts["race_pools_paid"] = len(groups)
    for why, k in left.items():
        counts[f"race_pools_left_out_{why}"] = k
    counts["race_pools_scoreable"] = sum(len(p) for p in data.paid)


def _ticket_chances(devig: str, shares: np.ndarray, dividends: np.ndarray, race_id: str,
                    pool: str, label: str | None) -> np.ndarray:  # fmt: skip
    """The chance each paid ticket of a Place or Wide race-pool pays, under a de-vig of the
    spec's baseline.place_wide: P times its statute share, or its own implied share
    R*100/(D_c+5) at the race's payout rate; either capped at 1."""
    if devig == STATUTE:
        return np.minimum(1.0, len(shares) * shares)
    rate = takeout.race_takeout(str(race_id), pool, race_class_raw=label).payout_rate
    return np.minimum(1.0, rate * exotic.UNIT_STAKE / (dividends + exotic.MEAN_BREAKAGE))


def _attach_form(data: _Data, form: pd.DataFrame) -> None:
    """stage2:q:form on each starter, renormalised per race; a race with a starter it lacks (or
    a value that is not a positive number) has none. Its counts are private (D21)."""
    f = pd.DataFrame(
        {
            "race_id": _objects(form["race_id"]),
            "horse_no": _floats(form["horse_no"]),
            "p_form": _floats(form["p_form"]),
        }
    ).dropna(subset=["horse_no"])
    f["horse_no"] = f["horse_no"].astype(np.int64)
    twice = f.duplicated(["race_id", "horse_no"], keep=False)
    f = f.loc[~twice.to_numpy()]
    s = pd.DataFrame(
        {
            "race_id": np.asarray(data.starters["race_id"].to_numpy(), dtype=object),
            "horse_no": data.starters["horse_no"].to_numpy(dtype=np.int64),
        }
    )
    p = s.merge(f, on=["race_id", "horse_no"], how="left", validate="many_to_one")["p_form"]
    p = p.to_numpy(dtype=float)
    ok = np.isfinite(p) & (p > 0)
    n = len(data.ids)
    race_ok = np.logical_and.reduceat(ok, data.lo) & (data.hi > data.lo) if n else np.zeros(0, bool)
    p = np.where(ok, p, np.nan)
    total = np.add.reduceat(np.nan_to_num(p), data.lo) if n else np.zeros(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = p / np.repeat(total, data.hi - data.lo)
    data.form, data.form_ok = p, race_ok
    data.form_counts = {
        "form_rows": len(form),
        "form_keys_given_twice": int(twice.sum()),
        "races_with_the_member": int(race_ok.sum()),
    }


# --------------------------------------------------------------------------- probabilities


def _top(arr: np.ndarray, pool: str) -> tuple[int, ...]:
    key = tuple(int(i) for i in np.unravel_index(int(np.argmax(arr)), arr.shape))
    return tuple(sorted(key)) if pool in UNORDERED else key


def _evaluate(data: _Data, order: Ordering | None, p: np.ndarray, pos: int,
              pools: Sequence[str]) -> dict:  # fmt: skip
    """Per pool of `pools` the race pays: (P_C5 of each paid combination, the flat ¥100's return
    on the most probable combination), or the exception the ordering raised."""
    paid = data.paid[pos]
    want = [pool for pool in pools if pool in paid]
    out: dict[str, tuple | Exception] = {}
    trifecta: np.ndarray | Exception | None = None
    if order is not None and any(pool != WIN for pool in want):
        try:
            trifecta = order.trifecta(p)
        except Exception as exc:  # recorded per race and variant, never dropped silently
            trifecta = exc
    for pool in want:
        x = paid[pool]
        try:
            if pool == WIN:
                arr = p
            elif isinstance(trifecta, Exception):
                raise trifecta
            else:
                arr = exotic.pool_array(trifecta, pool, field_at_sale=int(data.fields[pos]))
            out[pool] = (np.asarray(arr[x.index], dtype=float), x.payout(_top(arr, pool)))
        except Exception as exc:
            out[pool] = exc
    return out


class _Calibrated:
    """D4's spline per fold, fitted on the fold's training starters (cached)."""

    def __init__(self, data: _Data) -> None:
        self.data = data
        self.models: dict[int, WinCalibration] = {}

    def model(self, k: int) -> WinCalibration:
        if k not in self.models:
            s = self.data.starters
            rows = _rows_of(self.data, self.data.folds[k].train_idx)
            frame = pd.DataFrame(
                {
                    "race_id": np.asarray(s["race_id"].to_numpy()[rows], dtype=object),
                    "race_date": np.asarray(s["race_date"].to_numpy()[rows], dtype=object),
                    "win_odds": s["win_odds"].to_numpy()[rows],
                    "won": s["won"].to_numpy()[rows],
                }
            )
            self.models[k] = fit_win_calibration(frame, n_knots=N_KNOTS)
        return self.models[k]

    def flat(self, k: int, positions: np.ndarray) -> np.ndarray:
        """The fold's calibrated P(win) on the starters of `positions`, NaN elsewhere."""
        rows = _rows_of(self.data, positions)
        out = np.full(len(self.data.q), np.nan)
        out[rows] = self.model(k).predict_many(
            self.data.q[rows], np.asarray(self.data.starters["race_id"].to_numpy()[rows])
        )
        return out


def _rows_of(data: _Data, positions: np.ndarray) -> np.ndarray:
    """The starter rows of the races at `positions`, in that order."""
    positions = np.asarray(positions, dtype=np.int64)
    if not len(positions):
        return np.zeros(0, dtype=np.int64)
    lo, hi = data.lo[positions], data.hi[positions]
    sizes = hi - lo
    return np.repeat(lo - np.cumsum(sizes) + sizes, sizes) + np.arange(int(sizes.sum()))


def _fit_frame(data: _Data, rows: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    s = data.starters
    return pd.DataFrame(
        {
            "race_id": np.asarray(s["race_id"].to_numpy()[rows], dtype=object),
            "horse_no": s["horse_no"].to_numpy()[rows],
            "p_win": p[rows],
            "finish_pos": s["finish_pos"].to_numpy()[rows],
            "dead_heat": s["dead_heat"].to_numpy()[rows],
        }
    )


# --------------------------------------------------------------------------- the scorer


@dataclass
class _Point:
    """One grid point's per-race results over every fold."""

    ids: list[str] = field(default_factory=list)
    pos: list[int] = field(default_factory=list)
    delta: list[float] = field(default_factory=list)
    model_ll: list[float] = field(default_factory=list)
    base_ll: list[float] = field(default_factory=list)
    ret: list[float] = field(default_factory=list)
    alphas: list[tuple[str, float, int]] = field(default_factory=list)
    failures: int = 0
    first_error: str | None = None
    folds_without_fit: int = 0


def fit_alpha(q: Sequence[np.ndarray], p: Sequence[np.ndarray]) -> float:
    """alpha in [0, 1] minimising the mean over races of -(1/k) sum ln(alpha q + (1 - alpha) p)
    over each race's k paid combinations: the objective is convex, so alpha is 0 when its
    derivative at 0 is not negative, 1 when it is not positive at 1, else the derivative's root."""
    if not len(q):
        raise ValueError("no training race-pool to fit alpha on")
    w = np.concatenate([np.full(len(x), 1.0 / (len(x) * len(q))) for x in q])
    qq, pp = np.concatenate(list(q)), np.concatenate(list(p))
    diff = qq - pp

    def slope(a: float) -> float:
        return float(-(w * diff / (a * qq + (1.0 - a) * pp)).sum())

    if slope(0.0) >= 0.0:
        return 0.0
    if slope(1.0) <= 0.0:
        return 1.0
    return float(brentq(slope, 0.0, 1.0, xtol=1e-14, rtol=4 * np.finfo(float).eps, maxiter=500))


class _Scorer:
    def __init__(self, data: _Data, spec: dict, form_error: str | None) -> None:
        self.data, self.spec = data, spec
        self.boot = spec["bootstrap"]
        self.cap = int(spec["blend"]["max_train_races"])
        self.pools = list(dict.fromkeys([*spec["pools"], WIN, *spec["e44_pools"]]))
        self.cal = _Calibrated(data)
        self.form_error = form_error
        self.memo: dict[tuple, dict] = {}
        self.fits: dict[tuple, dict[int, dict] | Exception] = {}
        self.samples: dict[tuple, np.ndarray] = {}
        self.cal_flat: tuple[int, np.ndarray] | None = None
        test = [np.asarray(f.test_idx, dtype=np.int64) for f in data.folds]
        self.test = test
        self.super = self._super_premium(np.concatenate(test) if test else np.zeros(0, int))

    # -- inputs ---------------------------------------------------------------------------

    def input_error(self, name: str) -> str | None:
        if name == FORM and self.data.form is None:
            return self.form_error or "the stage2:q:form member was not given"
        if name not in INPUTS:
            return f"unknown input {name!r}"
        return None

    def available(self, name: str, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.int64)
        if name == FORM:
            return positions[self.data.form_ok[positions]]
        return positions

    def p_race(self, name: str, k: int, pos: int) -> np.ndarray:
        lo, hi = self.data.lo[pos], self.data.hi[pos]
        if name == NORMALISED:
            p = self.data.q[lo:hi]
        elif name == FORM:
            p = self.data.form[lo:hi]
        else:
            p = self.cal.model(k).predict(self.data.q[lo:hi])
        return p / p.sum()

    def train_flat(self, name: str, k: int) -> np.ndarray:
        """The input's P(win) on every starter (CALIBRATED: fold k's spline, NaN off fold k)."""
        if name == NORMALISED:
            return self.data.q
        if name == FORM:
            return self.data.form
        if self.cal_flat is None or self.cal_flat[0] != k:
            fold = self.data.folds[k]
            both = np.concatenate([fold.train_idx, fold.test_idx]).astype(np.int64)
            self.cal_flat = (k, self.cal.flat(k, both))
        return self.cal_flat[1]

    def sample(self, name: str, k: int) -> np.ndarray:
        """The fold's training races the blend's alpha is fitted on (at most `cap`)."""
        key = (name, k)
        if key not in self.samples:
            train = self.available(name, self.data.folds[k].train_idx)
            if len(train) > self.cap:
                smallest = np.argsort(self.data.hashes[train], kind="stable")[: self.cap]
                train = np.sort(train[smallest])
            self.samples[key] = train
        return self.samples[key]

    # -- orderings -----------------------------------------------------------------------

    def params(self, name: str, family: str | None, k: int) -> dict | None | Exception:
        """The ordering's parameters for fold k: {} for a family without any, None when the
        input has no training race in the fold, or the fit's exception."""
        if family is None or family not in FITTED:
            return {}
        key = (name, family)
        if key not in self.fits:
            self.fits[key] = self._fit(name, family)
        fits = self.fits[key]
        if isinstance(fits, Exception):
            return fits
        return fits.get(k)

    def _fit(self, name: str, family: str) -> dict[int, dict] | Exception:
        data = self.data
        try:
            if name == CALIBRATED:
                out = {}
                for k, fold in enumerate(data.folds):
                    train = self.available(name, fold.train_idx)
                    if not len(train):
                        continue
                    rows = _rows_of(data, train)
                    frame = _fit_frame(data, rows, self.train_flat(name, k)).reset_index(drop=True)
                    one = dataclasses.replace(
                        fold, train_idx=np.arange(len(frame)), test_idx=np.zeros(0, np.int64)
                    )
                    out[k] = dict(Ordering(family).fit(frame, [one]).fold_params[0])
                return out
            ok = np.ones(len(data.ids), dtype=bool) if name != FORM else data.form_ok
            races = np.flatnonzero(ok)
            rows = _rows_of(data, races)
            frame = _fit_frame(data, rows, self.train_flat(name, 0)).reset_index(drop=True)
            where = np.full(len(data.q), -1, dtype=np.int64)
            where[rows] = np.arange(len(rows))
            used, folds = [], []
            for k, fold in enumerate(data.folds):
                train = self.available(name, fold.train_idx)
                if not len(train):
                    continue
                test = self.available(name, fold.test_idx)
                used.append(k)
                folds.append(
                    dataclasses.replace(
                        fold,
                        train_idx=where[_rows_of(data, train)],
                        test_idx=where[_rows_of(data, test)],
                    )
                )
            fitted = Ordering(family).fit(frame, folds) if folds else None
            return {k: dict(fitted.fold_params[i]) for i, k in enumerate(used)} if folds else {}
        except Exception as exc:  # every variant of this input and family records it
            return exc

    def order(self, family: str | None, params: dict) -> Ordering | None:
        if family is None:
            return None
        return Ordering(family, params or None)

    def values(self, name: str, family: str | None, k: int, pos: int, *,
               market: bool = False) -> dict | None:  # fmt: skip
        """P_C5 and the top pick's return of race `pos` in fold k for (input, family), with the
        fold's own parameters, or with the market's (`market`: E44's shared ordering)."""
        fitted = family in FITTED
        fit_input = NORMALISED if market else name
        tag = ("market" if market and fitted else "own", k if fitted or name == CALIBRATED else -1)
        key = (name, family, tag, pos)
        hit = self.memo.get(key)
        if hit is not None:
            return hit
        params = self.params(fit_input, family, k)
        if params is None or isinstance(params, Exception):
            return None
        out = _evaluate(self.data, self.order(family, params), self.p_race(name, k, pos), pos,
                        self.pools)  # fmt: skip
        self.memo[key] = out
        return out

    # -- E16's race-pools -----------------------------------------------------------------

    def _super_premium(self, positions: np.ndarray) -> dict[str, set[int]]:
        frame = pd.DataFrame(
            {
                "race_id": self.data.ids[positions] if len(positions) else np.zeros(0, object),
                "race_class_raw": np.array([self.data.labels[i] for i in positions], dtype=object),
                "pos": positions,
            }
        )
        out = {}
        for pool in self.pools:
            try:
                kept = super_premium_subset(frame, pool)
            except Exception:  # a race without its conditions label: settle race by race
                ok = []
                for row in frame.itertuples(index=False):
                    try:
                        ok.append(_boost(row.race_id, pool, row.race_class_raw) == SUPER)
                    except Exception:
                        ok.append(False)
                kept = frame.loc[np.array(ok, dtype=bool)] if len(ok) else frame
            out[pool] = set(kept["pos"].tolist())
        return out

    # -- one grid point -------------------------------------------------------------------

    def point(self, name: str, family: str | None, pool: str, blend: str, devig: str | None,
              floor: str | None) -> _Point:  # fmt: skip
        out = _Point()
        drop = floor == "drop"

        def usable(pos: int) -> _Paid | None:
            x = self.data.paid[pos].get(pool)
            return None if x is None or (drop and x.floor) else x

        for k, fold in enumerate(self.data.folds):
            test = [int(i) for i in self.available(name, self.test[k]) if usable(int(i))]
            if not test:
                continue
            params = self.params(name, family, k)
            if isinstance(params, Exception):
                raise params
            if params is None:
                out.folds_without_fit += 1
                continue
            alpha = 0.0
            if blend == LINEAR:
                qs, ps = [], []
                for i in self.sample(name, k):
                    x = usable(int(i))
                    v = None if x is None else self.values(name, family, k, int(i))
                    if v is None or isinstance(v.get(pool), Exception):
                        continue
                    qs.append(x.base[devig])
                    ps.append(v[pool][0])
                if not qs:
                    out.folds_without_fit += 1
                    continue
                alpha = fit_alpha(qs, ps)
                out.alphas.append((fold.test_start.isoformat(), alpha, len(qs)))
            for i in test:
                x = usable(i)
                v = self.values(name, family, k, i)
                got = v.get(pool) if v is not None else None
                if isinstance(got, Exception) or got is None:
                    out.failures += 1
                    if out.first_error is None and isinstance(got, Exception):
                        out.first_error = f"{type(got).__name__}: {got}"
                    continue
                q, base_ll = x.base[devig], x.base_ll[devig]
                p = got[0] if blend == NO_BLEND else alpha * q + (1.0 - alpha) * got[0]
                ll = float(-np.log(p).mean())
                out.ids.append(self.data.ids[i])
                out.pos.append(i)
                out.delta.append(base_ll - ll)
                out.model_ll.append(ll)
                out.base_ll.append(base_ll)
                out.ret.append(got[1])
        if not out.ids:
            why = f"; {out.failures} race(s) failed, first: {out.first_error}"
            raise ValueError(f"no race-pool of {pool} was scored{why if out.failures else ''}")
        return out

    # -- metrics --------------------------------------------------------------------------

    def bounds(self, ids: Sequence[str], values: Sequence[float]) -> Interval:
        series = pd.Series(np.asarray(values, dtype=float), index=pd.Index(list(ids), dtype=object))
        return paired_bootstrap(
            series,
            self.data.clusters,
            n_boot=self.boot["n_boot"],
            level=self.boot["level"],
            seed=self.boot["seed"],
        )

    def meetings(self, ids: Sequence[str]) -> int:
        return int(self.data.clusters.reindex(pd.Index(list(ids), dtype=object)).nunique())


def _boost(race_id: str, pool: str, label: str | None) -> str | None:
    return takeout.race_takeout(str(race_id), pool, race_class_raw=label).boost


def super_premium_subset(frame: pd.DataFrame, pool: str) -> pd.DataFrame:
    """The rows of `frame` (race_id and, for the age-restricted boosts, race_class_raw) whose
    race-pool paid exactly the super_premium rate: `takeout.race_takeout(...).boost ==
    'super_premium'`. Ultra_premium (80% + 5%), premium and unboosted race-pools are left out;
    the rows keep their order, index and columns."""
    labels = (
        [None if pd.isna(x) else str(x) for x in frame["race_class_raw"]]
        if "race_class_raw" in frame
        else [None] * len(frame)
    )
    cache: dict[tuple, bool] = {}
    keep = []
    for race_id, label in zip(frame["race_id"], labels, strict=True):
        key = (race_id, label)
        if key not in cache:
            cache[key] = _boost(race_id, pool, label) == SUPER
        keep.append(cache[key])
    return frame.loc[np.array(keep, dtype=bool)] if len(frame) else frame


# --------------------------------------------------------------------------- the pure study


def _name(*parts) -> str:
    return ":".join(TAG.get(p, str(p)) for p in parts)


def _leaderboard(pool: str, rule: str, level: float) -> dict:
    return {
        "kind": "model",
        "pool": pool,
        "rule": rule,
        "selection": "look_ahead",
        "payout_rate": board_rate(pool),
        "ci_level": float(level),
        "n_unit": "bets",
    }


def _rule(pool: str, name: str, family: str | None, floor: str | None, segment: str) -> str:
    via = "its Win marginal" if family is None else family.replace("_", " ")
    cut = " (floor-bound race-pools left out)" if floor == "drop" else ""
    return (
        f"C5 on {name} through {via}: ¥100 on its most probable {pool} combination at the final "
        f"Win price, {segment}{cut}"
    )


def compute_study(races: pd.DataFrame, runners: pd.DataFrame, dividends: pd.DataFrame,
                  spec: dict, *, form: pd.DataFrame | None = None,
                  form_error: str | None = None,
                  log: Callable[[str], None] | None = None) -> Study:  # fmt: skip
    """Every variant of `spec`'s grid on store-shaped tables (the module docstring and the
    tests' contract): E43 and E16 per pool, input, ordering, blend (and Place and Wide de-vig and
    floor treatment), the log-linear blend as failed records, and E44 per ordering and model
    input. `form` holds stage2:q:form (race_id, horse_no, p_form); None, or `form_error`, fails
    every variant that needs it. `log` gets one progress line per stage, with no figure. Pure:
    no I/O."""
    started = clock.monotonic()

    def note(what: str) -> None:
        if log is not None:
            log(f"{what} ({clock.monotonic() - started:.0f} s)")

    data = _prepare(races, runners, dividends, spec, form)
    note("prepared the races")
    study = Study(counts=dict(data.counts))
    study.tables["form_coverage"] = [data.form_counts] if data.form_counts else []
    scorer = _Scorer(data, spec, form_error)
    scheme = "walk_forward:" + json.dumps(spec["fold_scheme"], sort_keys=True)
    blends = [b for b in spec["blends"] if b in (NO_BLEND, LINEAR)]
    level = spec["bootstrap"]["level"]
    families: list[str | None] = []
    if WIN in spec["pools"]:
        families.append(None)
    families += list(spec["orderings"])
    for family in families:
        pools = [WIN] if family is None else [p for p in spec["pools"] if p != WIN]
        for name in spec["inputs"]:
            for pool in pools:
                treatments = (
                    [(d, f) for d in spec["place_wide_devigs"] for f in spec["floor_treatments"]]
                    if pool in MULTI
                    else [(None, None)]
                )
                for devig, floor in treatments:
                    for blend in blends:
                        _grid_point(study, scorer, scheme, level, name, family, pool, blend,
                                    devig, floor)  # fmt: skip
        if family is not None and spec["e44_pools"]:
            for name in spec["inputs"]:
                if name != NORMALISED:
                    _e44(study, scorer, scheme, family, name)
        scorer.memo.clear()
        note(f"scored ordering {family or 'none (Win)'}")
    if LOG_LINEAR in spec["blends"]:
        for name in spec["inputs"]:
            for pool in spec["pools"]:
                for family in [None] if pool == WIN else spec["orderings"]:
                    params = {"pool": pool, "input": name, "ordering": family, "blend": LOG_LINEAR}
                    study.variants.append(
                        Variant(
                            "E43",
                            _name(pool, name, family, LOG_LINEAR),
                            scheme,
                            _components(name, family, LOG_LINEAR),
                            params,
                            error=LOG_LINEAR_ERROR,
                        )  # fmt: skip
                    )
    return study


def _components(name: str, family: str | None, blend: str) -> dict[str, str]:
    return {
        "study": "countries/japan/studies/exotic_c5.py",
        "model": "models.exotic.Ordering" if family is not None else "the input's Win marginal",
        "baseline": "models.exotic.dividend_shares",
        "input": name,
        "ordering": family or "none",
        "blend": blend,
    }


def _grid_point(study: Study, scorer: _Scorer, scheme: str, level: float, name: str,
                family: str | None, pool: str, blend: str, devig: str | None,
                floor: str | None) -> None:  # fmt: skip
    """The E43 and E16 variants of one grid point, scored from one pass over the folds."""
    params = {"pool": pool, "input": name, "ordering": family, "blend": blend}
    if pool in MULTI:
        params |= {"devig": devig, "floor": floor}
    tail = (devig, floor) if pool in MULTI else ()
    e43 = Variant("E43", _name(pool, name, family, blend, *tail), scheme,
                  _components(name, family, blend), dict(params))  # fmt: skip
    e16 = Variant("E16", "sp:" + _name(pool, name, family, blend, *tail), scheme,
                  _components(name, family, blend), dict(params))  # fmt: skip
    if blend == NO_BLEND:
        e43.params["leaderboard"] = _leaderboard(
            pool, _rule(pool, name, family, floor, "every scored race"), level
        )
        e16.params["leaderboard"] = _leaderboard(
            pool, _rule(pool, name, family, floor, "super_premium race-pools"), level
        )
    error = scorer.input_error(name)
    if error is None and devig not in (None, *DEVIGS):
        error = f"unknown Place and Wide de-vig {devig!r}"
    if error is not None:
        study.variants += [dataclasses.replace(v, error=error) for v in (e43, e16)]
        return
    try:
        got = scorer.point(name, family, pool, blend, devig, floor)
    except Exception as exc:
        study.variants += [
            dataclasses.replace(v, error=f"{type(exc).__name__}: {exc}") for v in (e43, e16)
        ]
        return
    row = {"pool": pool, "input": name, "ordering": family or "-", "blend": blend}
    row |= {"devig": devig or "-", "floor": floor or "-"}

    def scored(keep: Sequence[int], hyp: str, ret_metric: str) -> tuple[list, dict]:
        ids = [got.ids[j] for j in keep]
        if not ids:
            raise ValueError(f"no {SUPER} race-pool of {pool} was scored")
        delta = [got.delta[j] for j in keep]
        iv = scorer.bounds(ids, delta)
        metrics = [_metric(ALL_FOLDS, DELTA, iv.mean, len(ids), iv.lo, iv.hi)]
        table = row | {"E": hyp, "races": len(ids), "meetings": iv.n_clusters}
        table |= {"model LL": float(np.mean([got.model_ll[j] for j in keep]))}
        table |= {"dividend LL": float(np.mean([got.base_ll[j] for j in keep]))}
        table |= {"ΔLL": iv.mean, "lower": iv.lo, "upper": iv.hi}
        if hyp == "E43":
            metrics += [_metric(a, ALPHA, v, n) for a, v, n in got.alphas]
            if got.alphas:
                values = [v for _, v, _ in got.alphas]
                table |= {"alpha (median of folds)": float(np.median(values))}
            if got.failures:
                metrics.append(_metric(ALL_FOLDS, FAILED_ORDERING, got.failures, got.failures))
                table |= {"races failed": got.failures}
            if got.folds_without_fit:
                table |= {"folds without a fit": got.folds_without_fit}
        if blend == NO_BLEND:
            ret = scorer.bounds(ids, [got.ret[j] for j in keep])
            metrics.append(_metric(DEVELOPMENT, ret_metric, ret.mean, len(ids), ret.lo, ret.hi))
            table |= {"return per ¥100": ret.mean, "return lower": ret.lo, "return upper": ret.hi}
        return metrics, {hyp.lower(): [table | {"variant": variant_name[hyp]}]}

    variant_name = {"E43": e43.name, "E16": e16.name}
    everything = list(range(len(got.ids)))
    supers = scorer.super.get(pool, set())
    subset = [j for j, i in enumerate(got.pos) if i in supers]
    study.attempt(e43, lambda: scored(everything, "E43", RETURN))
    study.attempt(e16, lambda: scored(subset, "E16", f"{RETURN}[{SUPER}]"))
    for label, value, n in got.alphas:
        study.tables.setdefault("alpha", []).append(
            row | {"variant": e43.name, "fold": label, "alpha": value, "training race-pools": n}
        )


def _e44(study: Study, scorer: _Scorer, scheme: str, family: str, name: str) -> None:
    """E44 for one ordering and model input: one ordering, fitted on the market per fold,
    serving the market (normalised) and the model on the same races."""
    pools = list(scorer.spec["e44_pools"])
    params = {"ordering": family, "input": name, "market": NORMALISED, "pools": pools}
    components = _components(name, family, NO_BLEND) | {"market": NORMALISED}
    variant = Variant("E44", _name("e44", family, name), scheme, components, params)
    error = scorer.input_error(name)
    if error is not None:
        study.variants.append(dataclasses.replace(variant, error=error))
        return

    def compute() -> tuple[list, dict]:
        data = scorer.data
        ids, win, joint = [], [], {pool: [] for pool in pools}
        failures = 0
        for k in range(len(data.folds)):
            params_k = scorer.params(NORMALISED, family, k)
            if isinstance(params_k, Exception):
                raise params_k
            if params_k is None:
                continue
            for i in scorer.available(name, scorer.test[k]):
                i = int(i)
                if not all(pool in data.paid[i] for pool in [WIN, *pools]):
                    continue
                mkt = scorer.values(NORMALISED, family, k, i)
                mod = scorer.values(name, family, k, i, market=True)
                if mkt is None or mod is None:
                    continue
                got = [(mkt.get(pool), mod.get(pool)) for pool in [WIN, *pools]]
                if any(isinstance(x, Exception) or isinstance(y, Exception) for x, y in got):
                    failures += 1
                    continue
                ll = [(-np.log(x[0]).mean(), -np.log(y[0]).mean()) for x, y in got]
                ids.append(data.ids[i])
                win.append(float(ll[0][0] - ll[0][1]))
                for pool, (a, b) in zip(pools, ll[1:], strict=True):
                    joint[pool].append(float(a - b))
        if not ids:
            raise ValueError(f"no race was scored for E44 ({failures} failed)")
        iv = scorer.bounds(ids, win)
        metrics = [_metric(ALL_FOLDS, "win_gain", iv.mean, len(ids), iv.lo, iv.hi)]
        row = {"ordering": family, "input": name, "market": NORMALISED, "races": len(ids)}
        row |= {"meetings": iv.n_clusters, "win gain": iv.mean, "win lower": iv.lo}
        row |= {"win upper": iv.hi}
        for pool in pools:
            jv = scorer.bounds(ids, joint[pool])
            metrics += [
                _metric(ALL_FOLDS, f"joint_gain[{pool}]", jv.mean, len(ids), jv.lo, jv.hi),
                _metric(ALL_FOLDS, f"gain_ratio[{pool}]", jv.mean / iv.mean, len(ids)),
            ]
            row |= {f"{pool} gain": jv.mean, f"{pool} lower": jv.lo, f"{pool} upper": jv.hi}
            row |= {f"{pool} ratio": jv.mean / iv.mean}
        if failures:
            metrics.append(_metric(ALL_FOLDS, FAILED_ORDERING, failures, failures))
        return metrics, {"e44": [row | {"variant": variant.name}]}

    study.attempt(variant, compute)


# --------------------------------------------------------------------------- the seams


def development_days(date_from: date, date_to: date) -> list[date]:
    """Every day in [date_from, date_to] that is not a holdout day (#71's `is_holdout_day`)."""
    days, day = [], date_from
    while day <= date_to:
        if not is_holdout_day(COUNTRY, day):
            days.append(day)
        day += timedelta(days=1)
    return days


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def store_builds() -> dict[str, str]:
    """Each store table read, as '<path>@<sha256 prefix>' of its file."""
    folder = paths.processed_dir(SOURCE)
    return {
        f"{SOURCE}_{name}": f"processed/{SOURCE}/{name}.parquet@"
        f"{_digest(folder / f'{name}.parquet')[:16]}"
        for name in ("races", "runners", "dividends")
    }


def _on_days(table: pa.Table, days: set[date]) -> pd.DataFrame:
    frame = table.to_pandas()
    day = _as_dates(frame["race_date"])
    frame["race_date"] = np.array(day, dtype=object)
    return frame.loc[np.array([d in days for d in day], dtype=bool)].reset_index(drop=True)


def load_store(days: Sequence[date]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """The store's races, runners and dividends of `days` (development days): column- and
    date-scoped reads (`read_store_table`, holdout days dropped in the scan), the dividends by the
    races' ids; and the builds read."""
    wanted = set(days)
    if not wanted:
        raise ValueError("no development day to read")
    common = {"start": min(wanted), "end": max(wanted), "touches_price": True}
    races = _on_days(read_store_table("races", columns=RACE_COLUMNS, **common), wanted)
    runners = _on_days(read_store_table("runners", columns=RUNNER_COLUMNS, **common), wanted)
    ids = [str(x) for x in races["race_id"]]
    keep = ds.field("race_id").isin(pa.array(ids, pa.string()))
    path = paths.processed_dir(SOURCE) / "dividends.parquet"
    dividends = pq.read_table(path, columns=PAID_COLUMNS, filters=keep).to_pandas()
    return races, runners, dividends, store_builds()


def load_form(days: Sequence[date], members: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """#87's stage2:q:form out-of-fold P(win) on `days`, rebuilt by #87's own code
    (`studies.meta`): the store frame of development rows, HKTrack's member file (`members`,
    read-only, outside the repository), D9's folds from 2024-01-01, stage 2 on the form member
    and ln q, ridge 0. Returns (race_id, horse_no, p_form) and the member build. #418's level-2
    guard (`studies.meta.check_level2`, D9's seasons) runs before any fit, as in #87's own run
    without its opt-in: `models.stacking`'s `IneligibleFold` or `InvalidProvenance` refuses."""
    if members is None:
        raise FileNotFoundError(
            f"no stage2:q:form member file: pass --members (HKTrack's outputs/oof/{MEMBER_FILE})"
        )
    path = paths.outside_repo(Path(members), what="members")
    runners, races = c4.read_development()
    frame, _ = c4.store_frame(runners, races)
    del runners, races
    folds = walk_forward(frame["race_date"], scheme=load_country_scheme(COUNTRY))
    scored = [f for f in folds if f.test_start >= c4.SCORED_FROM]
    held = c4.read_members(path)
    level2 = c4.check_level2(frame, scored, held, c4.race_calendar(), min_seasons=c4.D9_MIN_SEASONS)
    rows, _ = c4.member_frame(level2, frame, held)
    del held
    rows = rows.assign(ln_q=np.log(rows["q"].to_numpy())).reset_index(drop=True)
    spec = MetaSpec(["form"], quadratic=False, segment_cols=[], form="stage2", ridge=0.0)
    oof, _ = c4._meta_oof(rows, level2.folds, spec, "ln_q")
    found = oof.notna().to_numpy()
    out = pd.DataFrame(
        {
            "race_id": np.asarray(rows["race_id"].to_numpy()[found], dtype=object),
            "horse_id": np.asarray(rows["horse_id"].to_numpy()[found], dtype=object),
            "p_form": oof.to_numpy()[found],
        }
    )
    wanted = set(days)
    numbers = read_store_table(
        "runners",
        columns=["race_id", "race_date", "horse_id", "horse_no"],
        start=PRICE_ERA_START,
        end=max(wanted),
        touches_price=False,
    ).to_pandas()
    numbers = numbers.loc[np.array([d in wanted for d in _as_dates(numbers["race_date"])], bool)]
    numbers = pd.DataFrame(
        {
            "race_id": np.asarray(numbers["race_id"].to_numpy(), dtype=object),
            "horse_id": np.asarray(numbers["horse_id"].to_numpy(), dtype=object),
            "horse_no": numbers["horse_no"].to_numpy(),
        }
    ).drop_duplicates(["race_id", "horse_id"], keep=False)
    out = out.merge(numbers, on=["race_id", "horse_id"], how="inner", validate="one_to_one")
    builds = {"hktrack_member_oof": f"HKTrack:outputs/oof/{MEMBER_FILE}@{_digest(path)[:16]}"}
    return out[["race_id", "horse_no", "p_form"]].reset_index(drop=True), builds


# --------------------------------------------------------------------------- the run


@dataclass
class C5Result:
    spec: dict
    spec_hash: str
    git_sha: str
    study: Study
    run: dict
    builds: dict[str, str]
    run_ids: dict[str, str] = field(default_factory=dict)
    report_path: str | None = None
    public_path: str | None = None
    seconds: float = 0.0
    twins: dict[str, dict] = field(default_factory=dict)  # per-ticket variant: its statute twin's


def statute_twins(registry, study: Study) -> dict[str, dict]:
    """For each per-ticket E43 and E16 variant of `study`, its statute twin's ΔLL and bounds: the
    first family's latest scored record with the same hypothesis, pool, input, ordering, blend
    and floor treatment and the statute de-vig. Publishable inputs only (D21)."""
    keys = ("pool", "input", "ordering", "blend", "floor")
    found = {}
    for r in registry.records(FAMILY_ID):
        if r.status != "scored" or is_licensed_record(r) or r.params.get("devig") != STATUTE:
            continue
        for m in r.metrics:
            if m.metric == DELTA and m.fold == ALL_FOLDS:
                found[(r.hypothesis_id, *(r.params.get(k) for k in keys))] = m
    out = {}
    for v in study.variants:
        m = found.get((v.hypothesis_id, *(v.params.get(k) for k in keys)))
        if v.params.get("devig") == PER_TICKET and not _licensed(v) and m is not None:
            out[v.name] = {"statute ΔLL": m.value, "statute lower": m.lo, "statute upper": m.hi}
    return out


def _run_id(record: RunRecord) -> str:
    canonical = json.dumps(
        dataclasses.asdict(record), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _licensed(variant: Variant) -> bool:
    return variant.params.get("input") == FORM


def _record(variant: Variant, family: str, git_sha: str, builds: dict,
            form_builds: dict) -> RunRecord:  # fmt: skip
    licensed = _licensed(variant)
    params = dict(variant.params) | ({"error": variant.error} if variant.error else {})
    return RunRecord(
        hypothesis_id=variant.hypothesis_id,
        family_id=family,
        variant=variant.name,
        ticket=TICKET,
        git_sha=git_sha,
        data_builds={**builds, **form_builds} if licensed else dict(builds),
        sources=FORM_SOURCES if licensed else STORE_SOURCES,
        fold_scheme=variant.fold_scheme,
        components=dict(variant.components),
        params=params,
        metrics=variant.metrics,
        status="failed" if variant.error else "scored",
    )


def write_committed_summary(registry, family: str = FAMILY_ID) -> None:
    """`family`'s summary file and the committed index with its line, by `write_summary` over
    that family alone (#92's rule: no other family's file is written)."""
    directory = summary_dir(COUNTRY)
    mine = c2_ability._Families(registry, {family})
    if not mine.records():
        return
    with tempfile.TemporaryDirectory() as scratch:
        write_summary(mine, scratch)
        write_report(
            directory / f"{family}.md",
            Path(scratch, f"{family}.md").read_text("utf-8"),
            sources=(),
        )
        fresh = Path(scratch, "README.md").read_text("utf-8").splitlines()
    index = directory / "README.md"
    committed = index.read_text("utf-8").splitlines() if index.exists() else []
    found = c2_ability._index_family
    entries = {family: line for line in committed + fresh if (family := found(line))}
    head = [line for line in fresh if not found(line)]
    write_report(index, "\n".join([*head, *(entries[f] for f in sorted(entries))]) + "\n",
                 sources=())  # fmt: skip


def run_study(*, date_from: date = PRICE_ERA_START, date_to: date | None = None,
              out_dir: Path | None = None, registry=None, members: Path | None = None,
              family: str = FAMILY_ID,
              log: Callable[[str], None] = lambda _: None) -> C5Result:  # fmt: skip
    """Run a pre-registered family (`FAMILY_ID`, or `PER_TICKET_FAMILY_ID`: Place and Wide
    against the per-ticket de-vig) on Japan's development days (module docstring). Raises
    `ValueError` for another family, `paths.DataDirInsideRepo` for an `out_dir` inside the
    repository and `FamilyAlreadyFrozen` when the family is frozen with another spec, all before
    any row is read."""
    spec = family_spec(family)
    out_dir = (
        paths.reports_dir() if out_dir is None else paths.outside_repo(Path(out_dir), what="out")
    )
    date_to = date.today() if date_to is None else date_to
    committed = registry is None
    registry = default_registry(COUNTRY) if registry is None else registry
    spec_hash = registry.freeze_family(family, spec)  # before either seam reads a row
    git_sha = _git_sha()
    started = clock.monotonic()
    days = development_days(date_from, date_to)
    run = {"date_from": date_from.isoformat(), "date_to": date_to.isoformat(), "days": len(days)}
    races, runners, dividends, builds = load_store(days)
    log(f"read the store: {len(races)} races ({clock.monotonic() - started:.0f} s)")
    form_error = None
    try:
        form, form_builds = load_form(days) if members is None else load_form(days, members)
    except Exception as exc:  # every stage2:q:form variant is a failed record; the rest run
        form, form_builds, form_error = None, {}, f"{type(exc).__name__}: {exc}"
    log(f"read the member ({clock.monotonic() - started:.0f} s)")
    study = compute_study(
        races, runners, dividends, spec, form=form, form_error=form_error, log=log
    )
    del races, runners, dividends, form
    seconds = clock.monotonic() - started
    log(f"scored {len(study.variants)} variants ({seconds:.0f} s)")

    result = C5Result(spec, spec_hash, git_sha, study, run, builds, seconds=seconds)
    if family == PER_TICKET_FAMILY_ID:
        result.twins = statute_twins(registry, study)
    held = {_run_id(dataclasses.replace(r, git_sha="")): r for r in registry.records(family)}
    for variant in study.variants:
        record = _record(variant, family, git_sha, builds, form_builds)
        same = held.get(_run_id(dataclasses.replace(record, git_sha="")))
        result.run_ids[variant.name + "@" + variant.hypothesis_id] = (
            registry.append(record) if same is None else _run_id(same)
        )
    tag = hashlib.sha256(json.dumps([run, builds, form_builds], sort_keys=True).encode())
    stem = f"exotic-c5-{COUNTRY}-{spec_hash[:12]}-{tag.hexdigest()[:8]}"
    about = {
        "family": f"`{family}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "run": json.dumps(run),
        "store": ", ".join(builds.values()),
        "member": ", ".join(form_builds.values()) or f"not read ({form_error})",
        "run time (s)": round(seconds),
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    result.report_path = str(
        write_report(out_dir / f"{stem}.md", render_report(result, about), sources=FORM_SOURCES)
    )
    write_report(
        out_dir / f"{stem}.json",
        json.dumps({"about": about, "counts": study.counts, "tables": study.tables}, indent=1,
                   default=str),
        sources=FORM_SOURCES,
    )  # fmt: skip
    if committed:
        write_committed_summary(registry, family)
        report = PUBLIC_REPORT if family == FAMILY_ID else PER_TICKET_PUBLIC_REPORT
        public = env.repo_root() / report
        result.public_path = str(write_report(public, render_public_report(result), sources=()))
    return result


# --------------------------------------------------------------------------- reports


def _status(v: Variant) -> str:
    return "failed" if v.error else "scored"


def render_report(result: C5Result, about: Mapping) -> str:
    """The private report: every table, the stage2:q:form rows included (D21: never in the
    repository)."""
    study = result.study
    lines = [
        f"# C5 against the pools' own dividends ({COUNTRY}, #{TICKET}): private report",
        "",
        "Holds stage2:q:form's figures (JRDB-fed, D21): written outside the repository only.",
        "",
        *(f"- {k}: {v}" for k, v in about.items()),
        "",
        "## Data",
        "",
        _markdown_table([{"count": k, "n": v} for k, v in study.counts.items()]),
    ]
    if study.tables.get("form_coverage"):
        lines += ["", f"## {FORM} coverage", "", _markdown_table(study.tables["form_coverage"])]
    for key, title in (
        ("e43", "E43: ΔLL against the dividends per pool (all folds)"),
        ("e16", "E16: the E43 table on super_premium race-pools"),
        ("e44", "E44: joint gain against Win gain through one ordering"),
        ("alpha", "The linear blend's alpha per fold"),
    ):
        rows = study.tables.get(key, [])
        lines += ["", f"## {title}", "", _markdown_table(rows) if rows else "_none scored_"]
    variants = [
        {"E": v.hypothesis_id, "variant": v.name, "status": _status(v), "error": v.error or ""}
        for v in study.variants
    ]
    lines += ["", "## Variants (one registry record each)", "", _markdown_table(variants), ""]
    return "\n".join(lines)


def _public_rows(study: Study, key: str, twins: Mapping[str, dict]) -> list[dict]:
    """The table's publishable rows, each with its statute twin's ΔLL when `twins` has one."""
    rows = [r for r in study.tables.get(key, []) if r.get("input") != FORM]
    return [r | twins.get(r.get("variant"), {}) for r in rows]


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.4g}" if isinstance(value, float) else str(value)


def _reading(rows: list[dict]) -> list[str]:
    """One mechanical line per publishable no-blend E43 variant: what its bounds exclude."""
    out = []
    for r in rows:
        if r.get("blend") != NO_BLEND:
            continue
        lo, hi = r["lower"], r["upper"]
        verdict = (
            "C5 beats the dividends (lower bound above 0)"
            if lo > 0
            else "the dividends beat C5 (upper bound below 0)"
            if hi < 0
            else "no difference excluded (the bounds straddle 0)"
        )
        twin = (
            f"; its statute twin's ΔLL {_fmt(r['statute ΔLL'])} [{_fmt(r['statute lower'])}, "
            f"{_fmt(r['statute upper'])}]"
            if "statute ΔLL" in r
            else ""
        )
        out.append(
            f"- `{r['variant']}`: ΔLL {_fmt(r['ΔLL'])} [{_fmt(lo)}, {_fmt(hi)}]: {verdict}{twin}; "
            f"return per ¥100 {_fmt(r.get('return per ¥100'))} "
            f"[{_fmt(r.get('return lower'))}, {_fmt(r.get('return upper'))}]"
        )
    return out


FOLLOW_UPS = (
    "#671: run the log-linear pool (`exotic.pool_blend`) once #4/#21's odds logs hold full final "
    "boards of the exotic pools, and settle the leaderboard bet at the full board's "
    "expected value rather than the most probable combination.",
    "#672: score C5 at the decision time (D2: T-2) on logged boards instead of the final Win "
    "price, so its leaderboard rows are decision-time selections, not look-ahead.",
    "#673: feed C5 the D9-compliant meta-model once #91's and #98's members land (stage2:q:form "
    "is #87's documented D9 shortfall).",
    "#674: append a run's registry records in one batch (`Registry.append` re-verifies the hash "
    "chain on every record, so a run of hundreds of variants spends most of its wall time "
    "appending).",
)

ALTERNATIVES = (
    "Fitting the E43 orderings once on the market for every input: rejected for E43, where 04 "
    "§2.3 refits the ordering on whatever probabilities feed it; kept for E44, where owner answer "
    "1 makes one ordering, fitted once on the market, serve both inputs.",
    "The equal split R*100/(P*(D_c+5)) for several winning combinations (#99 question 3(b)) as a "
    "second Place and Wide de-vig: the owner chose the statute's de-vig (answer 3(a)) for every "
    "race-pool with several winning combinations; Place's and Wide's censoring at the floor is "
    "registered instead as the 'drop' and 'bound' variants (#459 comment of 2026-11-27). Under "
    "the reframe (each candidate de-vig a variant) its ticket chance, R*100/(D_c+5), is "
    f"registered since as the second family, `{PER_TICKET_FAMILY_ID}` "
    f"(`{PER_TICKET_PUBLIC_REPORT.as_posix()}`).",
    "Fitting alpha on every training race of every fold: Henery's quadrature on every training "
    "race of 28 expanding folds is hours of compute for one parameter per fold; the spec caps "
    f"the fit at {BLEND_MAX_TRAIN} hash-sampled training races per fold.",
    "A bootstrap interval on the E44 ratio (#99 question 1 options (a) and (c)): owner answer "
    "1(b), the ratio is a point estimate beside the two bounded gains.",
)


PER_TICKET_ALTERNATIVES = (
    f"Adding the per-ticket de-vig to {FAMILY_ID}'s grid: that spec was frozen before its run "
    "read a row, and a changed spec is a new family, so the second de-vig is a family of its own "
    "and the first family's spec and records stand.",
    "Running Win, Quinella, Exacta, Trio, Trifecta and E44 here too: a pool that pays one "
    "combination per result has one share, R*100/(D+5), under either de-vig, so their records "
    "would repeat the first family's.",
    "A de-vig read from the Place and Wide pools' own final board or votes (the chance the money "
    "gives each ticket): not computable from the paid dividends, which are all the store holds; "
    "it needs #4/#21's logs of the full final boards.",
    "A per-pool scale on either de-vig fitted on the training races: a parameter fitted on "
    "outcomes makes the baseline a model; both de-vigs here are fixed formulas of the dividends.",
)

INTROS = {
    FAMILY_ID: "Ticket #459 (epic #73; 08 §4 E43, E44, E16; owner answers 1-7 on #99 of "
    "2026-11-26, the reframe of 2026-11-27). C5 (`models/exotic.py`, #99) turns an input's Win "
    "probabilities into every pool's combination probabilities through an ordering; each pool's "
    "result is scored against the pool's own paid dividends (the dividend-implied share of its "
    "money, statute de-vig for several winning combinations) by the log loss at 1/k per paid "
    "combination, out of fold on D9's walk-forward quarters. ",
    PER_TICKET_FAMILY_ID: "Ticket #459's second family (epic #73; 08 §4 E43, E16; the review of "
    "PR #668 under the owner's reframe on #99 of 2026-11-27: each candidate Place and Wide de-vig "
    f"is a registered variant). The first family, `{FAMILY_ID}` "
    f"(`{PUBLIC_REPORT.as_posix()}`), scored Place and Wide against one baseline, the chance a "
    "paid ticket pays taken as P × its statute money share. This family scores the same C5 grid "
    "on Place and Wide against a second baseline computable from the paid dividends alone: each "
    "paid ticket's own implied share, min(1, R·100/(D_c + 5)) (R the race's payout rate, boosts "
    "in; D_c its dividend per 100 yen), the money on it against what it pays, as in a pool with "
    "one winner (P × #99 question 3(b)'s equal split). Same races, folds, inputs, orderings, "
    "blends and floor treatments, by the log loss at 1/k per paid combination; only the "
    "baseline differs. So a no-blend variant's ΔLL minus its statute twin's (the `statute` "
    "columns, from the first family's registry records) is the statute baseline's log loss "
    "minus this one's, and the leaderboard bet, which never reads the baseline, returns what "
    "its twin returns. ",
}
TITLES = {
    FAMILY_ID: "C5 against the pools' own dividends",
    PER_TICKET_FAMILY_ID: "C5 against the Place and Wide dividends, per-ticket de-vig",
}
TWIN_COLUMNS = ("statute ΔLL", "statute lower", "statute upper")


def render_public_report(result: C5Result) -> str:
    """The public experiment report (docs/experiment-report.md's sections) of the result's
    family: the publishable pair's figures (normalised and calibrated, from the store alone; for
    the per-ticket family beside each statute twin's ΔLL) and every stage2:q:form variant by name
    and status only (D21). Nothing here depends on the member's values."""
    study, spec = result.study, result.spec
    fid = spec["family_id"]
    e43, e16, e44 = (_public_rows(study, k, result.twins) for k in ("e43", "e16", "e44"))
    form = [v for v in study.variants if _licensed(v)]
    public = [v for v in study.variants if not _licensed(v)]
    failed = [v for v in public if v.error]
    cols43 = [
        *("variant", "races", "meetings", "model LL", "dividend LL", "ΔLL", "lower", "upper"),
        *(TWIN_COLUMNS if result.twins else ()),
        *("alpha (median of folds)", "return per ¥100", "return lower", "return upper"),
    ]
    cols44 = ["variant", "races", "meetings", "win gain", "win lower", "win upper"]
    for pool in spec["e44_pools"]:
        cols44 += [f"{pool} gain", f"{pool} lower", f"{pool} upper", f"{pool} ratio"]
    flag = "" if fid == FAMILY_ID else f" --family {fid}"

    def table(rows: list[dict], cols: list[str]) -> str:
        shown = [{c: r.get(c) for c in cols} for r in rows]
        return _markdown_table(shown) if shown else "_none scored_"

    e44_section = [
        "### E44: joint gain against Win gain (calibrated against normalised)",
        "",
        table(e44, cols44),
        "",
    ]
    lines = [
        f"# Experiment report: {TITLES[fid]} (Japan, #459)",
        "",
        INTROS[fid] + "**Nothing is adopted** "
        "(D16-D19): every variant is a ranked row. The holdout is not opened and no holdout day "
        f"is read. **{FORM} is JRDB-fed (D21)**: its variants ran and are withheld; this report "
        "gives no figure, count or sign built on it. The figures below are the publishable "
        "pair's, normalised 1/odds and D4's calibrated market, from the store's results and "
        "dividends alone.",
        "",
        f"- Registry family `{fid}`, spec sha256 `{result.spec_hash}`, frozen before the "
        f"run read any store row; the pre-registration is `{SPEC_DOC}`.",
        f"- Code: `countries/japan/studies/exotic_c5.py` at `{result.git_sha[:12]}`. Reproduce: "
        f"`uv run horsetrack japan exotic-c5{flag} --members <HKTrack>/outputs/oof/"
        f"{MEMBER_FILE}`. #418's level-2 guard (`studies.meta.check_level2`) came after this "
        f"family's first run, whose {FORM} records stand: under it, this command refuses this "
        "member file before any fit, because D9's five seasons are met at level 2 on none of "
        f"the scored folds (#87's documented shortfall), and records every {FORM} variant "
        "failed.",
        f"- Run window: {result.run['date_from']} to {result.run['date_to']}, development days "
        "only.",
        f"- Records: `docs/experiments/registry/{COUNTRY}/{fid}.md` (the {FORM} records "
        "withheld). The private report, with every table, is under `$HORSETRACK_DATA_DIR/"
        "reports/` (`exotic-c5-japan-<spec hash>-<run digest>.md`).",
        "",
        "## Hypothesis",
        "",
        *(f"- **{k}.** {v}" for k, v in spec["hypotheses"].items()),
        f"- **Null**, the starting point: {spec['null']}.",
        "",
        "## Data",
        "",
        _markdown_table([{"count": k, "n": v} for k, v in study.counts.items()]),
        "",
        "## Variants tried",
        "",
        "One registry record each, failures included. ΔLL is the dividends' log loss minus the "
        "model's, in nats per race-pool (positive: C5 better); lower and upper are D7's "
        "one-sided 95% meeting-bootstrap bounds. The return is a flat ¥100 on the model's most "
        "probable combination per race-pool at the paid dividends (look-ahead: the inputs are "
        "the final Win prices), with the same bounds."
        + (
            " The statute columns are the statute twin's ΔLL and bounds, the first family's "
            "record of the same variant (empty where it holds none)."
            if result.twins
            else ""
        ),
        "",
        "### E43: per pool, against the dividends (every scored race-pool)",
        "",
        table(e43, cols43),
        "",
        "### E16: the E43 table on super_premium race-pools",
        "",
        table(e16, [c for c in cols43 if c != "alpha (median of folds)"]),
        "",
        *(e44_section if spec["e44_pools"] else []),
        f"### {FORM}: ran, withheld (D21)",
        "",
        f"{len(form)} variants read the member; each is a registry record whose figures are "
        "withheld. By name and status:",
        "",
        _markdown_table(
            [{"E": v.hypothesis_id, "variant": f"`{v.name}`", "status": _status(v)} for v in form]
        )
        if form
        else "_none_",  # fmt: skip
        "",
        "### Failed variants (publishable inputs)",
        "",
        _markdown_table(
            [{"E": v.hypothesis_id, "variant": f"`{v.name}`", "error": v.error} for v in failed]
        )
        if failed
        else "_none_",  # fmt: skip
        "",
        "## Alternatives rejected and why",
        "",
        *(f"- {x}" for x in (ALTERNATIVES if fid == FAMILY_ID else PER_TICKET_ALTERNATIVES)),
        "",
        "## What this rules out",
        "",
        "A mechanical reading of each publishable no-blend E43 variant's bounds; a measurement, "
        "not a verdict (nothing is adopted or dropped).",
        "",
        *(_reading(e43) or ["_no publishable variant scored_"]),
        "",
        "## Leaderboard rows",
        "",
        f"Every record is a row of `horsetrack japan leaderboard` under `{fid}/<variant>`: "
        "each no-blend E43 variant a measured row (its `return_per_100`), each no-blend E16 "
        f"variant a `[{SUPER}]` segment row, every other variant (the linear and log-linear "
        f"blends, E44, failures) unmeasured and unranked. The {FORM} rows are withheld: listed by "
        "option id only. Ranks are the leaderboard's (regenerated by the command).",
        "",
        *(
            f"- `{fid}/{v.name}`"
            + (f"[{SUPER}]" if v.hypothesis_id == "E16" and "leaderboard" in v.params else "")
            + (" (withheld, D21)" if _licensed(v) else "")
            + ("" if "leaderboard" in v.params and not v.error else " (unmeasured)")
            for v in study.variants
        ),
        "",
        "## Follow-ups filed",
        "",
        "Proposed (`status:proposed` under epic #73; not linked as its sub-issues, since #73 is at "
        "GitHub's cap of 100):",
        "",
        *(f"- {x}" for x in FOLLOW_UPS),
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- the command


def exotic_c5_command(
    date_from: Annotated[
        str, typer.Option("--from", help="First race day read (D10: not before 2014-06-07).")
    ] = PRICE_ERA_START.isoformat(),
    date_to: Annotated[
        str | None, typer.Option("--to", help="Last race day read (default: today).")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Private report folder, outside the repository.")
    ] = None,
    members: Annotated[
        Path | None,
        typer.Option("--members", help=f"#87's member file (HKTrack's outputs/oof/{MEMBER_FILE})."),
    ] = None,
    family: Annotated[
        str,
        typer.Option(
            "--family",
            help=f"The family to run: {FAMILY_ID} (every pool, the statute de-vig, E44) or "
            f"{PER_TICKET_FAMILY_ID} (Place and Wide against the per-ticket de-vig).",
        ),
    ] = FAMILY_ID,
) -> None:
    """Score C5 against the pools' own dividends (#459): E43 and E16 per pool, E44, every
    ordering, input and blend a registry record; private report outside the repository."""
    result = run_study(
        date_from=date.fromisoformat(date_from),
        date_to=None if date_to is None else date.fromisoformat(date_to),
        out_dir=out,
        members=members,
        family=family,
        log=lambda line: typer.echo(line, err=True),
    )
    failed = [v.name + "@" + v.hypothesis_id for v in result.study.variants if v.error]
    typer.echo(
        json.dumps(
            {
                "family_id": result.spec["family_id"],
                "spec_hash": result.spec_hash,
                "report_path": result.report_path,
                "public_path": result.public_path,
                "variants": len(result.study.variants),
                "failed_variants": len(failed),
                "seconds": round(result.seconds),
            },
            indent=2,
        )
    )
