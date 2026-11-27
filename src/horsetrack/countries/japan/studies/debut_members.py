"""Newcomer and maiden members at the final Win price (#438, epic #73; N7, E34, E37):
`horsetrack japan debut-members`.

The owner's hypothesis ("probably in maidens and debuts versus the big G1 races") tested member by
member. The races are the JRA store's DEBUT and MAIDEN races (`select_races`) on development days
from 2014-06-07 (D10: the final price enters every scored race). Each runner's members are read as
of its decision time t = post - 2 min (D2's primary, `config/decision.yaml`) under the
availability rules of `config/availability.yaml` (`member_features`):

- `ability` (#92): the C2 ability member's per-runner u and its Laplace sigma (se_u), out of fold,
  from #92's v2 export under the data dir; known from the race day's card (00:00 JST).
- `pedigree_prior` (moved here from #95, E37): the sire's offspring record in the race's surface
  and distance band, from earlier runs only (`pedigree_prior`, `prior_at`); known from 00:00 JST.
- `workouts_jrdb` (#137, private): #137's features from JRDB's CHA and CYB, published at
  (race day - 1) 20:00 JST.
- `workouts_jravan` (#229): #229's best and trend over the newest workouts with a par, a workout
  counting when dated before race day and published, at (workout date + 1 day) 00:00 JST,
  strictly before t; computed in bulk per race day (`jravan_workouts`).
- `jvdata_mining` (#406): not landed, so listed and left out; it is added as a counted variant
  when it lands.
- `body_weight` (#227): the race-day weight and its change, in the views from post - 60 min.
- `late_win` (0B41, #369): the Win share's move to T-2 from T-20 and from T-5 (`late_win`). The
  board at T-x is each runner's latest WIN row available (fetched_utc, else snapshot_utc) at or
  before the study's own post_utc - x, post_utc the JRA store's; the day file's post_utc and
  minutes_to_post are JRA-VAN's and never set a cutoff.

The fit is a conditional logit with the final-price offset: u = ln q + sum w x, q the normalised
final Win price, so w = 0 is the market exactly (`offset_probabilities`). Each member's columns
are standardised on the training rows (a missing value at the training mean, beside a missing
flag); a column that varies within no training race is dropped. It is fitted per class (DEBUT,
MAIDEN) on each D9 fold's training rows (`config/folds.yaml`), one member at a time and all built
members together, and predicts the fold's test quarter. Per variant: the per-race winner ΔLL
against the market with D7's one-sided meeting bounds, each column's weight pooled over the scored
rows (meeting-clustered, one-sided 95%), the Fair-Shiller encompassing weight of its out-of-fold
ln p beside ln q (`models.meta.encompassing`), and N5's curve (`core.eval.tail.tail_curve`),
every scored runner a flat Win bet settled by `settle_win`: a winner at its own WIN dividend, a
loser at 0. A race any of whose winners has no WIN dividend row is left out of the curve (never
settled at the final odds, nor a loss) and counted as the ranked row's `n5_excluded_races`; it
still scores ΔLL. Place waits for #410.

Every figure is built on JRA-VAN or JRDB data, or comes from the run that reads them, so every
figure is withheld (D21). A figure reaches a file only two ways. The private report, its JSON and
the candidate-cell file go under the data dir through `write_report`, which refuses the
repository, each named per run (spec hash, date range and input builds) so no run overwrites
another. The registry (`default_registry`, `$HORSETRACK_DATA_DIR/registry/japan.jsonl`, outside
the tree) is the private record of every run CLAUDE.md requires, failures included, with each
variant's metrics as every study passes them; every record names the run's sources (`jra`,
`jravan`, `jrdb`), so `write_summary` shows each one as withheld in the committed summary. The
public report (`render_public_report`) shows no figure. `run_debut_members` freezes the spec
(`build_spec`) in the registry before any store row is read, appends one record per variant
(failures included, a failed read too), and regenerates the committed summary and the public
report; it returns its result even when no variant scored, and the command then exits nonzero.
Holdout days are never read and the holdout is not opened. Nothing is adopted: each variant is a
ranked row.

Family v1 (`japan-debut-maiden-members-v1`) is kept in the record, superseded by v2: v1 cut the
late Win boards by the JRA-VAN post's minutes_to_post, and settled a race without a WIN dividend
row at the final odds.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import time as clock
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Annotated
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import typer

from horsetrack.core import env, paths
from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.folds import Fold, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day, split_off
from horsetrack.core.eval.registry import RunRecord, default_registry, summary_dir, write_summary
from horsetrack.core.eval.reports import write_report
from horsetrack.core.eval.scoring import gain as paired_gain
from horsetrack.core.eval.tail import tail_curve
from horsetrack.core.pricing import clean_odds
from horsetrack.countries.japan.handicap import workouts as w137
from horsetrack.countries.japan.models.condlogit import fit_conditional_logit
from horsetrack.countries.japan.models.meta import encompassing
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table
from horsetrack.countries.japan.studies import ability as c2_ability
from horsetrack.countries.japan.studies.calibration import (
    Study,
    Variant,
    _bounds,
    _markdown_table,
    _metric,
    bootstrap_settings,
    store_builds,
)
from horsetrack.countries.japan.studies.cross_pool import boards_at, win_shares
from horsetrack.countries.japan.studies.meta import _ll, _run_id, file_digest, restrict

COUNTRY = "japan"
TICKET = 438
FAMILY_ID = "japan-debut-maiden-members-v2"
SUPERSEDES = {
    "family_id": "japan-debut-maiden-members-v1",
    "reason": "v1 cut the late Win boards by the day file's minutes_to_post, measured to the "
    "JRA-VAN post rather than the study's JRA post, and settled a race whose winner had no WIN "
    "dividend row at the final odds in the N5 curve; v1's records are kept",
}
SOURCES = ("jra", "jravan", "jrdb")
SEED = 438
CLASSES = ("DEBUT", "MAIDEN")  # the store's codes: research/pool_scan's newcomer and maiden
JST = ZoneInfo("Asia/Tokyo")
DECISION_MIN = int(load_country_yaml(COUNTRY, "decision")["primary_min"])  # D2: T-2
RESULTS_FROM = date(2010, 1, 1)  # the pedigree prior's runs (results, no price): the store's start
SURFACES = ("turf", "dirt")
BAND_EDGES = (1400, 1800, 2200)  # a band holds the distances up to and including its edge
BAND_LABELS = ("sprint", "mile", "intermediate", "long")
PRIOR_PSEUDO_RUNS = 10  # the prior's shrinkage: pseudo runs at the field's midpoint
WORKOUT_LOOKBACK_DAYS = 365  # #229's newest workouts are read from the year before race day
LATE_FROM_MIN = (20, 5)  # the late Win path: the move to T-2 from T-20 and from T-5
RIDGE = 1e-3  # on the standardised member columns, against near-singular fits; the offset is fixed
N5_QUANTILES = (0.5, 0.8, 0.9, 0.95, 0.99)
ABILITY_FAMILY = c2_ability.FAMILY_ID
CELLS_DIR = "candidate-cells"
PUBLIC_REPORT = Path("docs") / "experiments" / "debut-members-japan.md"
SPEC_DOC = "docs/specs/debut-members-prereg.md"


@dataclass(frozen=True)
class Member:
    """A member: the ticket that provides it, whether it is built, the sources it reads, when its
    values are known (`gate`) and its inputs, as (frame column, feature name) pairs."""

    ticket: int
    status: str
    sources: tuple[str, ...]
    gate: str  # card | post | jrdb_eve | workout | none
    inputs: tuple[tuple[str, str], ...]
    about: str
    lead_min: int = 0  # gate "post": known from post_utc - lead_min


MEMBERS = {
    "ability": Member(
        92,
        "built",
        ("jra", "jravan"),
        "card",
        (("ability_u", "u"), ("ability_se_u", "se_u")),
        "#92's per-runner u and its Laplace sigma se_u (pooled jockey, trainer and sire on #91's "
        "rating), out of fold, from the v2 export named by the family's records",
    ),
    "pedigree_prior": Member(
        TICKET,
        "built",
        ("jra", "jravan"),
        "card",
        (("pedigree_prior", "prior"),),
        "the sire's offspring record in the race's surface and distance band from earlier runs "
        "only (moved here from #95)",
    ),
    "workouts_jrdb": Member(
        137,
        "built",
        ("jra", "jrdb"),
        "jrdb_eve",
        tuple((f"jrdb_{name}", name) for name in w137.FEATURE_COLUMNS),
        "#137's workout features from JRDB's CHA and CYB (private, D21)",
    ),
    "workouts_jravan": Member(
        229,
        "built",
        ("jra", "jravan"),
        "workout",
        (("ff_best_vs_par_s", "ff_best_vs_par_s"), ("ff_trend_s", "ff_trend_s")),
        "#229's best final furlong against par and its trend from JRA-VAN's 坂路 and ウッド times",
    ),
    "jvdata_mining": Member(
        406,
        "missing",
        ("jravan",),
        "none",
        (),
        "the JV-Data mining predictions (#406): not landed; listed and left out, added as a "
        "counted variant when it lands",
    ),
    "body_weight": Member(
        227,
        "built",
        ("jra",),
        "post",
        (("horse_wt_kg", "horse_wt_kg"), ("horse_wt_change_kg", "horse_wt_change_kg")),
        "the race-day body weight and its change, in the views from post - 60 min",
        lead_min=60,
    ),
    "late_win": Member(
        TICKET,
        "built",
        ("jra", "jravan"),
        "post",
        tuple(
            (f"late_win_move_t{m}", f"ln_share_move_t{m}_t{DECISION_MIN}") for m in LATE_FROM_MIN
        ),
        "the Win share's move to T-2 from T-20 and from T-5 on the JRA-VAN 0B41 boards (#369), "
        "each board cut at the JRA post",
        lead_min=DECISION_MIN,
    ),
}
BUILT = tuple(name for name, m in MEMBERS.items() if m.status == "built")
COMBINED = "combined"
HYPOTHESES = {
    "E34": "members that carry information the final Win price lacks for runners with thin "
    "history, concentrated in debut and maiden races (08 §4 E34 as #438 reads it: the pooled "
    "ability, workouts, body weight and the late Win path, one at a time and together)",
    "E37": "a pedigree prior (sire x surface x distance band, earlier runs only) adds to the "
    "final Win price in debut and maiden races",
}


def _hypothesis(member: str) -> str:
    return "E37" if member == "pedigree_prior" else "E34"


def variant_names() -> list[str]:
    """Every planned variant, in run order: each built member alone, then all together, per
    class."""
    return [f"member:{m}:{c}" for c in CLASSES for m in BUILT] + [
        f"{COMBINED}:{c}" for c in CLASSES
    ]


# --------------------------------------------------------------------------- races and members


def select_races(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows of DEBUT and MAIDEN races on development days from 2014-06-07, unchanged."""
    days = pd.to_datetime(frame["race_date"])
    known = days.notna().to_numpy()
    day = days.dt.date
    dev = {d: d >= PRICE_ERA_START and not is_holdout_day(COUNTRY, d) for d in set(day[known])}
    on_dev = known & day.map(dev).fillna(False).to_numpy(dtype=bool)
    in_class = frame["race_class"].isin(CLASSES).to_numpy(dtype=bool, na_value=False)
    return frame.loc[in_class & on_dev]


def _jst_midnight(day: pd.Series, plus_days: int = 0) -> pd.Series:
    local = pd.to_datetime(day).dt.normalize() + pd.Timedelta(days=plus_days)
    return local.dt.tz_localize(JST).dt.tz_convert("UTC")


def _known(frame: pd.DataFrame, member: Member, t: pd.Series) -> np.ndarray:
    """Whether each row's member values are known at its t (NaT never is)."""
    if member.gate == "card":  # the race day's card: from 00:00 JST
        known = _jst_midnight(frame["race_date"]) <= t
    elif member.gate == "post":
        post = pd.to_datetime(frame["post_utc"], utc=True)
        known = post - pd.Timedelta(minutes=member.lead_min) <= t
    elif member.gate == "jrdb_eve":  # #137: (race day - 1) 20:00 JST, strictly before t
        publish = w137.PUBLISH_TIME
        eve = _jst_midnight(frame["race_date"], -1) + pd.Timedelta(
            hours=publish.hour, minutes=publish.minute
        )
        known = eve < t
    elif member.gate == "workout":  # #229: dated before race day, published strictly before t
        day = pd.to_datetime(frame["workout_date"])
        known = (day < pd.to_datetime(frame["race_date"])) & (_jst_midnight(day, 1) < t)
    else:
        raise ValueError(f"member gate {member.gate!r} builds no feature")
    return known.to_numpy(dtype=bool, na_value=False)


def member_features(
    frame: pd.DataFrame, *, at_utc_col: str, members: Sequence[str]
) -> pd.DataFrame:
    """Each built member's features as of t = frame[at_utc_col], columns `<member>__<name>` on
    frame.index; a value not known at t is NaN. A missing member builds no column; an unknown
    member is a ValueError."""
    unknown = [m for m in members if m not in MEMBERS]
    if unknown:
        raise ValueError(f"unknown member(s) {unknown}; the members are {list(MEMBERS)}")
    t = pd.to_datetime(frame[at_utc_col], utc=True)
    out = pd.DataFrame(index=frame.index)
    for name in members:
        member = MEMBERS[name]
        if member.status != "built":
            continue
        known = _known(frame, member, t)
        for column, feature in member.inputs:
            values = pd.to_numeric(frame[column], errors="coerce").astype("float64").to_numpy()
            out[f"{name}__{feature}"] = np.where(known, values, np.nan)
    return out


def feature_columns(member: str) -> list[str]:
    return [f"{member}__{feature}" for _, feature in MEMBERS[member].inputs]


# --------------------------------------------------------------------------- the pedigree prior


def distance_band(distance_m: pd.Series) -> pd.Series:
    """The band label of each distance (BAND_EDGES inclusive above), None where missing."""
    d = pd.to_numeric(distance_m, errors="coerce").to_numpy(dtype=float)
    labels = np.array(BAND_LABELS, dtype=object)[np.searchsorted(BAND_EDGES, np.nan_to_num(d))]
    return pd.Series(np.where(np.isnan(d), None, labels), index=distance_m.index, dtype=object)


def _scored_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """The runs a prior counts: a sire, turf or dirt, a distance and a placing in a field of 2 or
    more, each scored by the share of its field it beat, (field - finish) / (field - 1)."""
    finish = pd.to_numeric(runs["finish_pos"], errors="coerce").astype(float).to_numpy()
    size = pd.to_numeric(runs["field_size"], errors="coerce").astype(float).to_numpy()
    band = distance_band(runs["distance_m"])
    ok = (
        runs["sire_id"].notna().to_numpy()
        & runs["surface"].isin(SURFACES).to_numpy(dtype=bool, na_value=False)
        & band.notna().to_numpy()
        & (size >= 2)
        & (finish >= 1)
        & (finish <= size)
    )
    return pd.DataFrame(
        {
            "sire_id": runs["sire_id"].to_numpy()[ok],
            "surface": runs["surface"].to_numpy()[ok],
            "distance_band": band.to_numpy()[ok],
            "race_date": pd.to_datetime(runs["race_date"]).to_numpy()[ok],
            "beaten": ((size - finish) / (size - 1))[ok],
        }
    )


def _shrunk(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    """The logit of the share beaten, shrunk PRIOR_PSEUDO_RUNS runs toward the midpoint (0)."""
    k = PRIOR_PSEUDO_RUNS
    share = (np.asarray(total, dtype=float) + 0.5 * k) / (np.asarray(count, dtype=float) + k)
    return np.log(share / (1.0 - share))


def pedigree_prior(runs: pd.DataFrame, *, as_of: date) -> pd.DataFrame:
    """One row per (sire_id, surface, distance_band) cell with a run strictly before `as_of`: its
    `prior` (the shrunk logit of the share of the field its offspring beat) and `runs`."""
    scored = _scored_runs(runs)
    before = scored.loc[scored["race_date"] < pd.Timestamp(as_of)]
    cells = before.groupby(["sire_id", "surface", "distance_band"], sort=True)["beaten"]
    out = cells.agg(["sum", "count"]).reset_index()
    out["prior"] = _shrunk(out["sum"].to_numpy(), out["count"].to_numpy())
    return out.rename(columns={"count": "runs"})[
        ["sire_id", "surface", "distance_band", "prior", "runs"]
    ]


def prior_at(runs: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """`pedigree_prior` at each row's own race day, vectorised: rows hold sire_id, surface,
    distance_m and race_date. 0 where the cell has no earlier run; NaN without a sire, on a
    surface other than turf or dirt, or without a distance. Index: rows'."""
    scored = _scored_runs(runs)
    band = distance_band(rows["distance_m"])
    keyed = (
        rows["sire_id"].notna().to_numpy()
        & rows["surface"].isin(SURFACES).to_numpy(dtype=bool, na_value=False)
        & band.notna().to_numpy()
    )

    def key(sire, surface, bands) -> np.ndarray:
        return np.array([f"{a}|{b}|{c}" for a, b, c in zip(sire, surface, bands, strict=True)])

    run_key = key(scored["sire_id"], scored["surface"], scored["distance_band"])
    row_key = key(rows["sire_id"].to_numpy()[keyed], rows["surface"].to_numpy()[keyed], band[keyed])
    codes = pd.factorize(np.concatenate([run_key, row_key]))[0].astype(np.int64)
    span = np.int64(1 << 24)  # day numbers since 1970 sit far below it

    def day(values) -> np.ndarray:
        return pd.to_datetime(values).to_numpy().astype("datetime64[D]").astype(np.int64)

    run_comp = codes[: len(run_key)] * span + day(scored["race_date"])
    order = np.argsort(run_comp, kind="stable")
    run_comp = run_comp[order]
    cum_total = np.r_[0.0, np.cumsum(scored["beaten"].to_numpy()[order])]
    row_code = codes[len(run_key) :]
    hi = np.searchsorted(run_comp, row_code * span + day(rows["race_date"].to_numpy()[keyed]))
    lo = np.searchsorted(run_comp, row_code * span)
    out = np.full(len(rows), np.nan)
    out[keyed] = _shrunk(cum_total[hi] - cum_total[lo], (hi - lo).astype(float))
    return pd.Series(out, index=rows.index)


# --------------------------------------------------------------------------- the offset


def offset_probabilities(
    frame: pd.DataFrame, weights: dict[str, float], *, offset_col: str
) -> pd.Series:
    """P(win) per row: the softmax within race_id of offset + sum of weight x column (a missing
    value contributes nothing)."""
    u = frame[offset_col].to_numpy(dtype=float).copy()
    for column, weight in weights.items():
        if weight != 0.0:
            u += float(weight) * np.nan_to_num(frame[column].to_numpy(dtype=float), nan=0.0)
    codes = pd.factorize(frame["race_id"].to_numpy(dtype=object))[0]
    top = np.full(codes.max() + 1 if len(codes) else 0, -np.inf)
    np.maximum.at(top, codes, u)
    e = np.exp(u - top[codes])
    return pd.Series(e / np.bincount(codes, weights=e)[codes], index=frame.index)


# --------------------------------------------------------------------------- the spec


def build_spec(*, seed: int) -> dict:
    """The family's pre-registered spec (plain JSON): the repository's config and this module's
    constants only, no data dir. Frozen in the registry before any store row is read."""
    level = bootstrap_settings()["level"]
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "epic": 73,
        "spec_doc": SPEC_DOC,
        "hypotheses": HYPOTHESES,
        "null": "every member weight 0: the final Win price already holds what the member says "
        "about a debut or maiden runner (ΔLL 0, encompassing weight 0)",
        "races": "the JRA store's run races of race_class DEBUT or MAIDEN on development days "
        f"from {PRICE_ERA_START.isoformat()} (D10), with a priced winner; every priced starter",
        "target": "the race's winner (Win pool), a dead heat's winners at 1/k (D6)",
        "decision_time": f"post_utc - {DECISION_MIN} min (D2's primary, config/decision.yaml); "
        "post_utc from races.post_time_local on the JST wall clock of race_date (#227)",
        "offset": "ln q, q the final Win odds' 1/O normalised within race; coefficient 1 (D1)",
        "members": {
            name: {
                "status": m.status,
                "ticket": m.ticket,
                "sources": list(m.sources),
                "features": feature_columns(name),
                "availability": m.gate,
                "about": m.about,
            }
            for name, m in MEMBERS.items()
        },
        "availability": {
            "card": "known from 00:00 JST of the race day (the card; a past run from the next "
            "day, config/availability.yaml horse_runs)",
            "post": "known from post_utc - lead (body weight 60 min, #227; the late Win path at "
            f"its own T-{DECISION_MIN})",
            "jrdb_eve": "known strictly after (race day - 1) 20:00 JST (#137's CHA and CYB)",
            "workout": "the newest workout drawn on is dated before race day and published, at "
            "(workout date + 1 day) 00:00 JST, strictly before t (#229)",
        },
        "pedigree_prior": {
            "runs": "every not-scratched starter of the JRA store's run races from "
            f"{RESULTS_FROM.isoformat()} (results, no price), its sire the JRA-VAN "
            "horse_pedigree.sire_id (#92's source), field = the race's starters",
            "score": "share of the field beaten, (field - finish) / (field - 1); a run without "
            "a placing, a field under 2 or a jump race is not counted",
            "cell": "sire x surface (turf, dirt) x distance band, bands up to and including "
            f"{list(BAND_EDGES)} m: {list(BAND_LABELS)}",
            "prior": f"logit((sum + {PRIOR_PSEUDO_RUNS} x 0.5) / (runs + {PRIOR_PSEUDO_RUNS})) "
            "over the cell's runs strictly before the race day; 0 with none",
        },
        "workouts_jravan": "#229's ff_best_vs_par_s and ff_trend_s (par: the median final "
        "furlong per centre and course of every workout dated before race day, at least "
        f"{w137.PAR_MIN_WORKOUTS}; the newest {w137.N_RECENT} with a par) over the horse's "
        f"workouts dated in [race day - {WORKOUT_LOOKBACK_DAYS}, race day - 1]",
        "workouts_jrdb": "#137's workout_features at race morning, identical at t (#137's "
        "records are all published at (race day - 1) 20:00 JST)",
        "late_win": f"ln s(T-{DECISION_MIN}) - ln s(T-m) per runner for m in "
        f"{list(LATE_FROM_MIN)}, s the Win board's normalised 1/odds at T-x (studies.cross_pool."
        "boards_at on the jravan odds_timeseries day file): each runner's latest WIN row "
        "available (fetched_utc, else snapshot_utc) at or before the study's post_utc - x, "
        "post_utc the JRA store's (decision_time); the file's own post_utc and minutes_to_post "
        "(JRA-VAN's post) never set a cutoff; NaN where either board lacks the runner or the "
        "study has no post time",
        "ability": f"u and se_u of #92's export (family {ABILITY_FAMILY}); out of fold from "
        "2015-Q1, missing before",
        "design": "per training rows: each feature standardised by its known values' mean and "
        "sd, a missing value at 0 beside a missing flag (when a training row lacks it); a column "
        "that varies within no training race is dropped",
        "fit": f"conditional logit u = ln q + X w (models.condlogit), ridge {RIDGE} on w, per "
        "class (DEBUT, MAIDEN) and D9 fold, on the fold's training rows of the class; each built "
        "member alone and every built member together (combined)",
        "variants": variant_names(),
        "folds": dataclasses.asdict(load_country_scheme(COUNTRY)),
        "scored": "every D9 fold's test quarter (walk_forward over the study's race days)",
        "metrics": {
            "gain": "per-race winner ΔLL, ll(q) - ll(model), out of fold over the scored races; "
            f"D7's meeting bootstrap, one-sided {level:.0%}",
            "weight": "each design column's weight, fitted on every scored row of the class "
            f"(in-sample), meeting-clustered sandwich se, one-sided {level:.0%} bounds",
            "encompassing": "the Fair-Shiller weight of the variant's out-of-fold ln p in u = a "
            "ln p + b ln q over the scored rows (models.meta.encompassing), meeting-clustered, "
            "one-sided bounds; a column beside each ranked row",
            "rank": "the ranked rows order the variants by their ΔLL, each with its weight, "
            "bounds and encompassing weight; nothing is adopted or rejected",
        },
        "n5": {
            "claim": "p (out of fold) x final Win odds per scored runner",
            "return": "settle_win: a winner's own WIN dividend row (not a refund) / 100, a "
            "dead heat's winners each their own; 0 for a loser (flat stakes)",
            "excluded": "a race any of whose winners (a dead heat's too) has no WIN dividend row "
            "is left out of the curve: not settled at the final odds, nor by another pool's "
            "row, nor counted a loss; it still scores ΔLL, and the ranked row counts such "
            "scored races as n5_excluded_races (private report only)",
            "quantiles": list(N5_QUANTILES),
            "settled": ["WIN"],
            "place": "after #410",
        },
        "bootstrap": {**bootstrap_settings(), "seed": int(seed)},
        "decisions": {
            "D1": "condition on the final price: the members are measured beside the final Win q",
            "D6": "winner log loss per race, a dead heat at 1/k per co-winner",
            "D7": "meeting (venue x day) clusters, 95% one-sided (config/scoring.yaml)",
            "D9": "walk-forward quarters, expanding, 1-week embargo (config/folds.yaml)",
            "D10": "no price row before 2014-06-07",
            "D11-D15": "holdout A2 from 2026-11-28, fully locked; no holdout row is read; not "
            "opened",
            "D16-D19": "nothing is adopted; every variant is a ranked row, the encompassing "
            "weight beside it",
            "D21": "every figure is withheld: built on JRA-VAN or JRDB, or read in their run; a "
            "figure reaches a file only through write_report (under the data dir, never the "
            "repository) and the registry's private run record (under the data dir), whose "
            "committed summary write_summary withholds",
            "#438": "the owner's reframe, 2026-11-27: no result gate, no sign-off stop; a "
            "member whose ticket has not landed is listed and added as a counted variant when "
            "it lands; each JRDB feature a ranked variant with private results",
        },
        "adoption": "none: ranked rows only (D16-D19)",
        "sources": list(SOURCES),
        "supersedes": SUPERSEDES,
        "outputs": "the private report, its JSON and the candidate-cell file under the data dir "
        "through write_report, named by the spec hash and a digest of the run's date range and "
        "input builds (a later run never overwrites an earlier one); one registry record per "
        "planned variant, failures included",
    }


# --------------------------------------------------------------------------- the fit


@dataclass(frozen=True)
class Design:
    """A training sample's design: per feature its mean and sd, and the kept columns."""

    stats: dict[str, tuple[float, float]]
    flags: tuple[str, ...]
    names: tuple[str, ...]

    @classmethod
    def of(cls, rows: pd.DataFrame, features: Sequence[str]) -> Design:
        stats, flags = {}, []
        for f in features:
            values = rows[f].to_numpy(dtype=float)
            known = ~np.isnan(values)
            if known.any():
                sd = float(values[known].std())
                stats[f] = (float(values[known].mean()), sd if sd > 0 else 1.0)
            if (~known).any() and known.any():
                flags.append(f)
        full = cls(stats, tuple(flags), ())
        matrix = full._matrix(rows)
        codes = pd.factorize(rows["race_id"].to_numpy(dtype=object))[0]
        by_race = pd.DataFrame(matrix).groupby(codes)
        spread = (by_race.max() - by_race.min()).max().to_numpy()
        seen: set[bytes] = set()
        kept = []
        for j, name in enumerate(full._all_names()):
            column = matrix[:, j].tobytes()
            if spread[j] <= 1e-12 or column in seen:  # no information, or a repeated flag
                continue
            seen.add(column)
            kept.append(name)
        names = tuple(kept)
        if not names:
            raise ValueError("no member column varies within a training race")
        return cls(stats, tuple(flags), names)

    def _all_names(self) -> list[str]:
        return [*self.stats, *(f"{f}:missing" for f in self.flags)]

    def _matrix(self, rows: pd.DataFrame) -> np.ndarray:
        cols = []
        for f, (mean, sd) in self.stats.items():
            cols.append(np.nan_to_num((rows[f].to_numpy(dtype=float) - mean) / sd, nan=0.0))
        for f in self.flags:
            cols.append(np.isnan(rows[f].to_numpy(dtype=float)).astype(float))
        return np.column_stack(cols) if cols else np.empty((len(rows), 0))

    def frame(self, rows: pd.DataFrame) -> pd.DataFrame:
        """The design columns of `rows`, with race_id and ln_q, on rows' index."""
        full = pd.DataFrame(self._matrix(rows), columns=self._all_names(), index=rows.index)
        return full[list(self.names)].assign(race_id=rows["race_id"], ln_q=rows["ln_q"])


def fit_offset(rows: pd.DataFrame, features: Sequence[str]):
    """(design, fit): the conditional logit u = ln q + X w on `rows` (see the module doc)."""
    design = Design.of(rows, features)
    x = design.frame(rows)[list(design.names)].to_numpy()
    fit = fit_conditional_logit(
        x,
        rows["race_id"].to_numpy(dtype=object),
        rows["won"].to_numpy(dtype=bool),
        offset=rows["ln_q"].to_numpy(dtype=float),
        ridge=RIDGE,
        cluster=rows["meeting_id"].to_numpy(dtype=object),
    )
    return design, fit


def predict_offset(design: Design, fit, rows: pd.DataFrame) -> pd.Series:
    weights = dict(zip(design.names, (float(c) for c in fit.coef), strict=True))
    return offset_probabilities(design.frame(rows), weights, offset_col="ln_q")


def oof_probabilities(rows: pd.DataFrame, folds: Sequence[Fold], features: Sequence[str]):
    """Out-of-fold P(win) on rows' index (NaN on rows no fold tests)."""
    out = pd.Series(np.nan, index=rows.index)
    for fold in folds:
        if len(fold.test_idx) == 0:
            continue
        design, fit = fit_offset(rows.iloc[fold.train_idx], features)
        test = rows.iloc[fold.test_idx]
        out.loc[test.index] = predict_offset(design, fit, test).to_numpy()
    return out


# --------------------------------------------------------------------------- the pure study


def _z(spec: dict) -> float:
    return NormalDist().inv_cdf(spec["bootstrap"]["level"])


def _variant_metrics(name, rows, folds, features, spec) -> tuple[list, dict]:
    """Every metric of one variant on one class's rows (the module doc)."""
    klass = name.rsplit(":", 1)[-1]
    tested = np.zeros(len(rows), dtype=bool)
    fold_of = np.full(len(rows), None, dtype=object)
    for fold in folds:
        tested[fold.test_idx] = True
        fold_of[fold.test_idx] = fold.test_start.isoformat()
    if not tested.any():
        raise ValueError("no scored fold has a test row")
    p = oof_probabilities(rows, folds, features)
    scored = rows.loc[tested].assign(p=p[tested], fold=fold_of[tested])
    race = scored["race_id"].to_numpy()
    clusters = scored["meeting_id"].groupby(race).first()
    gain = paired_gain(_ll(scored["p"], scored), _ll(scored["q"], scored)).per_race
    iv = _bounds(gain, clusters, spec)
    folds_of_race = scored["fold"].groupby(race).first().reindex(gain.index).to_numpy()
    metrics = [
        _metric(f, "delta_ll", part.mean(), len(part)) for f, part in gain.groupby(folds_of_race)
    ]
    metrics.append(_metric("all_folds", "delta_ll", iv.mean, len(gain), iv.lo, iv.hi))

    z = _z(spec)
    design, pooled = fit_offset(scored, features)
    weights = []
    for column, c, s in zip(design.names, pooled.coef, pooled.se_cluster, strict=True):
        metrics.append(
            _metric("all_folds", f"weight[{column}]", c, pooled.n_races, c - z * s, c + z * s)
        )
        weights.append(
            {"variant": name, "column": column, "weight": c, "lower": c - z * s, "upper": c + z * s}
        )
    data = scored[["race_id", "won", "meeting_id"]].assign(
        member=np.log(scored["p"].to_numpy()), ln_q_final=scored["ln_q"].to_numpy()
    )
    a, se = encompassing(data, "member", cluster_col="meeting_id")
    metrics.append(_metric("all_folds", "encompassing", a, len(gain), a - z * se, a + z * se))

    # N5: a race settle_win left unsettled (a winner without a WIN dividend row) is no bet
    unsettled = scored["payout"].isna().groupby(race).transform("any").to_numpy(dtype=bool)
    settled = scored.loc[~unsettled]
    bets = pd.DataFrame(
        {
            "claim": settled["p"].to_numpy() * settled["win_odds"].to_numpy(),
            "ret": settled["payout"].to_numpy(dtype=float),
            "meeting_id": settled["meeting_id"].to_numpy(),
        }
    )
    boot = spec["bootstrap"]
    n5: list[dict] = []
    top = {"realised": np.nan, "lo": np.nan, "n_bets": 0}
    if len(bets):
        curve = tail_curve(
            bets,
            claim_col="claim",
            return_col="ret",
            cluster_col="meeting_id",
            quantiles=spec["n5"]["quantiles"],
            n_boot=boot["n_boot"],
            level=boot["level"],
            seed=boot["seed"],
        )
        n5 = curve.assign(variant=name).to_dict("records")
        top = curve.iloc[-1]
    for band in n5:
        if band["n_bets"]:
            metrics.append(
                _metric(
                    "all_folds",
                    f"n5_realised[band{band['band']}]",
                    band["realised"],
                    band["n_bets"],
                    band["lo"],
                    band["hi"],
                )  # fmt: skip
            )
    ranked = {
        "variant": name,
        "class": klass,
        "races": len(gain),
        "meetings": iv.n_clusters,
        "delta_ll": iv.mean,
        "delta_ll_lo": iv.lo,
        "delta_ll_hi": iv.hi,
        "encompassing_weight": a,
        "encompassing_lo": a - z * se,
        "encompassing_hi": a + z * se,
        "n5_top_band_realised": float(top["realised"]),
        "n5_top_band_lo": float(top["lo"]),
        "n5_top_band_bets": int(top["n_bets"]),
        "n5_excluded_races": int(pd.unique(race[unsettled]).size),
    }
    return metrics, {"ranked": [ranked], "weights": weights, "n5_curve": n5}


def compute_study(
    frame: pd.DataFrame,
    folds: Sequence[Fold],
    spec: dict,
    failed_members: dict[str, str] | None = None,
) -> Study:
    """Every planned variant on the study frame (race_id, race_date, meeting_id, race_class,
    won, q, ln_q, win_odds, payout and the member feature columns), positions per `folds`.
    A member in `failed_members` (its inputs did not load) fails its variants and the combined
    ones. Pure: no file or registry I/O."""
    failed_members = failed_members or {}
    study = Study(counts={})
    fold_scheme = "walk_forward:" + json.dumps(spec["folds"], sort_keys=True)
    for klass in CLASSES:
        positions = np.flatnonzero(frame["race_class"].eq(klass).to_numpy(dtype=bool))
        rows = frame.iloc[positions].reset_index(drop=True)
        class_folds = restrict(list(folds), positions, len(frame))
        study.counts[f"{klass}_races"] = int(rows["race_id"].nunique())
        study.counts[f"{klass}_starters"] = len(rows)
        plans = [(m, f"member:{m}:{klass}", [m]) for m in BUILT]
        plans.append((COMBINED, f"{COMBINED}:{klass}", list(BUILT)))
        for member, name, used in plans:
            components = {
                "study": "studies/debut_members.py",
                "fit": "conditional logit, offset ln q_final (models.condlogit)",
                "member": member,
            }
            params = {"race_class": klass, "members": used, "ridge": RIDGE}
            variant = Variant(_hypothesis(member), name, fold_scheme, components, params)
            broken = [m for m in used if m in failed_members]

            def compute(name=name, used=used, broken=broken, rows=rows, class_folds=class_folds):
                if broken:
                    raise RuntimeError(
                        "; ".join(f"{m} inputs: {failed_members[m]}" for m in broken)
                    )
                features = [c for m in used for c in feature_columns(m)]
                return _variant_metrics(name, rows, class_folds, features, spec)

            study.attempt(variant, compute)
    ranked = study.tables.get("ranked", [])
    ranked.sort(key=lambda r: -r["delta_ll"])
    for k, row in enumerate(ranked, start=1):
        row["rank"] = k
    return study


# --------------------------------------------------------------------------- inputs

RUNNER_COLUMNS = [
    "race_id",
    "race_date",
    "horse_id",
    "horse_no",
    "win_odds",
    "finish_pos",
    "is_scratched",
    "horse_wt_kg",
    "horse_wt_change_kg",
]
RACE_COLUMNS = [
    "race_id",
    "race_date",
    "meeting_id",
    "race_class",
    "status",
    "post_time_local",
    "surface",
    "distance_m",
]


def _development(name, columns, start, end, *, touches_price, source="jra", where=None):
    table = read_store_table(
        name,
        columns=columns,
        start=start,
        end=end,
        touches_price=touches_price,
        where=where,
        source=source,
    )
    frame, _ = split_off(table.to_pandas(date_as_object=False), "race_date", COUNTRY)
    return frame


def _post_utc(race_date: pd.Series, hhmm: pd.Series) -> pd.Series:
    text = pd.to_datetime(race_date).dt.strftime("%Y-%m-%d") + " "
    text = text + hhmm.astype("string").str.strip().str.zfill(5)
    local = pd.to_datetime(text, format="%Y-%m-%d %H:%M", errors="coerce")
    return local.dt.tz_localize(JST).dt.tz_convert("UTC")


def study_frame(date_from: date, date_to: date) -> tuple[pd.DataFrame, dict]:
    """The priced starters of the study's run races with a priced winner, sorted by date, race
    and horse number: keys, race attributes, post_utc, decision_utc, q, ln_q, won, win_odds,
    the body-weight inputs and the Win payout per unit stake."""
    races = select_races(
        _development("races", RACE_COLUMNS, date_from, date_to, touches_price=False)
    )
    races = races.loc[races["status"].eq("run")]
    ids = pa.array(races["race_id"].tolist(), pa.string())
    runners = _development(
        "runners",
        RUNNER_COLUMNS,
        date_from,
        date_to,
        touches_price=True,
        where=ds.field("race_id").isin(ids),
    )
    scratched = runners["is_scratched"].eq(True).to_numpy(dtype=bool, na_value=False)
    odds = clean_odds(runners["win_odds"].to_numpy())
    starting = ~scratched & ~np.isnan(odds)
    starters = runners.loc[starting].assign(win_odds=odds[starting])
    starters["won"] = starters["finish_pos"].eq(1).to_numpy(dtype=bool, na_value=False)
    has_winner = starters.groupby("race_id")["won"].transform("any")
    frame = starters.loc[has_winner].drop(columns=["finish_pos", "is_scratched", "race_date"])
    frame = frame.merge(races.drop(columns="status"), on="race_id", validate="many_to_one")
    frame["horse_no"] = frame["horse_no"].astype("int64")
    frame = frame.sort_values(["race_date", "race_id", "horse_no"], kind="stable")
    frame = frame.reset_index(drop=True)
    inv = 1.0 / frame["win_odds"]
    frame["q"] = inv / inv.groupby(frame["race_id"]).transform("sum")
    frame["ln_q"] = np.log(frame["q"].to_numpy())
    frame["post_utc"] = _post_utc(frame["race_date"], frame["post_time_local"])
    frame["decision_utc"] = frame["post_utc"] - pd.Timedelta(minutes=DECISION_MIN)

    path = paths.processed_dir("jra") / "dividends.parquet"
    scope = ds.field("pool").isin(pa.array(["WIN"])) & ds.field("race_id").isin(
        pa.array(frame["race_id"].unique().tolist(), pa.string())
    )
    div = (
        ds.dataset(path, format="parquet")
        .to_table(columns=["race_id", "pool", "combination", "dividend", "is_refund"], filter=scope)
        .to_pandas()
    )
    frame["payout"] = settle_win(frame, div).to_numpy()
    counts = {
        "races": int(frame["race_id"].nunique()),
        "starters": len(frame),
        "races_without_a_post_time": int(frame.loc[frame["post_utc"].isna(), "race_id"].nunique()),
        "races_without_a_win_dividend": int(frame.loc[frame["payout"].isna(), "race_id"].nunique()),
    }
    return frame, counts


def settle_win(starters: pd.DataFrame, dividends: pd.DataFrame) -> pd.Series:
    """The Win return per unit stake per starter row (on starters.index): a winner its own WIN
    dividend row / 100 (a refund row is no dividend), a loser 0.0. Every runner of a race any of
    whose winners (a dead heat's too) has no WIN dividend row is NaN: the race is excluded, never
    settled at the final odds, by another pool's row, or as a loss."""
    win = dividends.loc[dividends["pool"].astype(object).eq("WIN").to_numpy(dtype=bool)]
    win = win.loc[~win["is_refund"].astype(object).eq(True).to_numpy(dtype=bool)]
    paid = pd.DataFrame(
        {
            "race_id": win["race_id"].astype(object).to_numpy(),
            "horse_no": pd.to_numeric(win["combination"], errors="coerce").to_numpy(dtype=float),
            "dividend": pd.to_numeric(win["dividend"], errors="coerce").to_numpy(dtype=float),
        }
    ).dropna()
    paid = paid.groupby(["race_id", "horse_no"], as_index=False)["dividend"].max()
    race = starters["race_id"].astype(object).to_numpy()
    runners = pd.DataFrame({"race_id": race, "horse_no": starters["horse_no"].to_numpy(float)})
    dividend = runners.merge(paid, on=["race_id", "horse_no"], how="left", validate="many_to_one")[
        "dividend"
    ].to_numpy(dtype=float)  # a left merge keeps the starters' order
    won = starters["won"].to_numpy(dtype=bool)
    unpaid = pd.Series(won & np.isnan(dividend)).groupby(race).transform("any")
    out = np.where(won, dividend / 100.0, 0.0)
    out[unpaid.to_numpy(dtype=bool)] = np.nan
    return pd.Series(out, index=starters.index, dtype=float)


def pedigree_runs(sires: pd.DataFrame) -> pd.DataFrame:
    """Every not-scratched starter of the store's run races from RESULTS_FROM (no price), with
    its sire from `sires` (horse_id, sire_id), the race's surface and distance, and field_size
    its starters."""
    runners = _development(
        "runners",
        ["race_id", "race_date", "horse_id", "finish_pos", "is_scratched"],
        RESULTS_FROM,
        date.max,
        touches_price=False,
    )
    races = _development(
        "races", ["race_id", "race_date", "surface", "distance_m", "status"], RESULTS_FROM,
        date.max, touches_price=False,
    )  # fmt: skip
    runners = runners.loc[~runners["is_scratched"].eq(True).to_numpy(dtype=bool, na_value=False)]
    run = races.loc[races["status"].eq("run"), ["race_id", "surface", "distance_m"]]
    runs = runners.drop(columns="is_scratched").merge(run, on="race_id", validate="many_to_one")
    runs["field_size"] = runs.groupby("race_id")["horse_id"].transform("size")
    return runs.merge(sires, on="horse_id", how="left", validate="many_to_one")


def ability_export() -> tuple[pd.DataFrame, str]:
    """#92's per-runner export (race_id, horse_id, ability_u, ability_se_u), the one the latest
    scored record of its family names, and that stem."""
    records = [r for r in default_registry(COUNTRY).records(ABILITY_FAMILY) if r.status == "scored"]
    if not records:
        raise FileNotFoundError(f"no scored record of {ABILITY_FAMILY} names an export")
    stem = records[-1].data_builds[c2_ability.ARTIFACTS].rpartition("@")[2]
    path = paths.data_dir() / "features" / COUNTRY / f"{stem}.parquet"
    table = pq.read_table(path, columns=["race_id", "horse_id", "u", "se_u"]).to_pandas()
    return table.rename(columns={"u": "ability_u", "se_u": "ability_se_u"}), stem


def jrdb_workouts(frame: pd.DataFrame) -> pd.DataFrame:
    """#137's features per frame row at race morning (equal at t): `jrdb_<feature>`."""
    if not w137.jrdb_features_enabled():
        raise RuntimeError(f"{w137.FLAG} is off: the JRDB workout features are disabled")
    values = w137._report_features(frame[["race_id", "horse_no", "race_date"]])
    columns = [f"jrdb_{name}" for name in w137.FEATURE_COLUMNS]
    return pd.DataFrame(values, columns=columns, index=frame.index)


def _cumulative_pars(pool: pl.DataFrame, days: list[date]) -> dict[date, pl.DataFrame]:
    """#229's par per race day: the median final furlong per (centre, course) over every
    workout dated before the day, where at least PAR_MIN_WORKOUTS have one (#229's `_pars`,
    from cumulative counts of the distinct values)."""
    pool = pool.filter(
        pl.col("workout_date").is_not_null()
        & pl.col("final_furlong_s").is_not_null()
        & pl.col("final_furlong_s").is_not_nan()
    )
    race_days = np.array(days, dtype="datetime64[D]")
    out_rows: list[list] = [[] for _ in days]
    for (c, k), part in pool.group_by(["centre", "course"]):
        wd = part.get_column("workout_date").to_numpy().astype("datetime64[D]")
        values = part.get_column("final_furlong_s").to_numpy()
        bins, inverse = np.unique(values, return_inverse=True)
        step = np.searchsorted(race_days, wd, side="right")  # counts from this race day on
        hist = np.zeros((len(days) + 1, len(bins)), dtype=np.int64)
        np.add.at(hist, (step, inverse), 1)
        cum = np.cumsum(hist, axis=0)[: len(days)]
        n = cum.sum(axis=1)
        cdf = np.cumsum(cum, axis=1)
        for j in np.flatnonzero(n >= w137.PAR_MIN_WORKOUTS):
            lo = bins[np.searchsorted(cdf[j], (n[j] - 1) // 2, side="right")]
            hi = bins[np.searchsorted(cdf[j], n[j] // 2, side="right")]
            out_rows[j].append((c, k, (lo + hi) / 2.0))
    schema = {"centre": pl.String, "course": pl.String, "par": pl.Float64}
    return {
        d: pl.DataFrame(rows, schema=schema, orient="row")
        for d, rows in zip(days, out_rows, strict=True)
    }


def jravan_workouts(frame: pd.DataFrame, log: Callable[[str], None] = print) -> pd.DataFrame:
    """#229's features per frame row on its race day (module doc): workout_date (the newest
    workout drawn on), ff_best_vs_par_s and ff_trend_s; NaN where none counts."""
    out = pd.DataFrame(
        {"workout_date": pd.NaT, "ff_best_vs_par_s": np.nan, "ff_trend_s": np.nan},
        index=frame.index,
    )
    days = sorted({ts.date() for ts in pd.to_datetime(frame["race_date"])})
    pool = pl.from_arrow(
        read_store_table(
            "workouts",
            columns=["workout_date", "centre", "course", "final_furlong_s"],
            start=w137.HISTORY_START,
            end=days[-1] - timedelta(days=1),
            touches_price=False,
            source="jravan",
        )  # fmt: skip
    )
    pars = _cumulative_pars(pool, days)
    del pool
    columns = ["horse_id", "workout_date", "workout_time", "centre", "course", "final_furlong_s"]
    by_day = pd.to_datetime(frame["race_date"]).dt.date
    for year in sorted({d.year for d in days}):
        ydays = [d for d in days if d.year == year]
        horses = frame.loc[by_day.isin(ydays).to_numpy(), "horse_id"].dropna().unique().tolist()
        records = pl.from_arrow(
            read_store_table(
                "workouts",
                columns=columns,
                start=ydays[0] - timedelta(days=WORKOUT_LOOKBACK_DAYS),
                end=ydays[-1] - timedelta(days=1),
                touches_price=False,
                source="jravan",
                where=ds.field("horse_id").isin(pa.array(horses, pa.string())),
            )  # fmt: skip
        )
        for day in ydays:
            mine = by_day.eq(day).to_numpy()
            ids = frame.loc[mine, "horse_id"].dropna().unique().tolist()
            counted = records.filter(
                pl.col("horse_id").is_in(ids)
                & (pl.col("workout_date") < day)
                & (pl.col("workout_date") >= day - timedelta(days=WORKOUT_LOOKBACK_DAYS))
            )
            recent = w137._best_and_trend(counted, pars[day], "workout_time")
            newest = (
                counted.join(pars[day], on=["centre", "course"], how="inner")
                .filter(
                    pl.col("final_furlong_s").is_not_null() & pl.col("final_furlong_s").is_not_nan()
                )
                .group_by("horse_id")
                .agg(pl.col("workout_date").max())
            )
            got = recent.join(newest, on="horse_id", how="left").to_pandas().set_index("horse_id")
            rows = frame.loc[mine, "horse_id"]
            for col in ("ff_best_vs_par_s", "ff_trend_s", "workout_date"):
                out.loc[rows.index, col] = rows.map(got[col]).to_numpy()
        log(f"  jravan workouts {year}: {len(ydays)} race days")
    out["workout_date"] = pd.to_datetime(out["workout_date"])
    return out.astype({"ff_best_vs_par_s": float, "ff_trend_s": float})


def late_win(frame: pd.DataFrame) -> pd.DataFrame:
    """Per frame row: ln s(T-2) - ln s(T-m) for each m of LATE_FROM_MIN from the day's jravan
    odds_timeseries Win rows (`boards_at`), the board at T-x cut at the frame's own post_utc - x
    (the JRA store's post): the file's post_utc and minutes_to_post are JRA-VAN's and are
    replaced by the minutes from each row's availability (fetched_utc, else snapshot_utc) to the
    frame's post. NaN where either board lacks the runner, the race has no post time, or the
    day has no file."""
    columns = [f"late_win_move_t{m}" for m in LATE_FROM_MIN]
    out = pd.DataFrame(np.nan, index=frame.index, columns=columns)
    folder = paths.processed_dir("jravan") / "odds_timeseries"
    read = ["race_id", "pool", "combination", "snapshot_utc", "fetched_utc", "odds", "status_raw"]
    for day, rows in frame.groupby(pd.to_datetime(frame["race_date"]).dt.date, sort=True):
        path = folder / f"{day.isoformat()}.parquet"
        if is_holdout_day(COUNTRY, day) or not path.exists():  # a holdout day is never opened
            continue
        ids = rows["race_id"].astype(object).unique().tolist()
        board = pq.read_table(
            path, columns=read, filters=[("pool", "==", "WIN"), ("race_id", "in", ids)]
        ).to_pandas()
        post = dict(
            zip(
                rows["race_id"].astype(object),
                pd.to_datetime(rows["post_utc"], utc=True),
                strict=True,
            )
        )
        available = pd.to_datetime(board["fetched_utc"], utc=True).fillna(
            pd.to_datetime(board["snapshot_utc"], utc=True)
        )
        race_post = pd.to_datetime(board["race_id"].astype(object).map(post), utc=True)
        board["minutes_to_post"] = (race_post - available) / pd.Timedelta(minutes=1)
        at = {m: boards_at(board, minutes_before=m) for m in (DECISION_MIN, *LATE_FROM_MIN)}
        for rid, runners in rows.groupby("race_id"):
            now = at[DECISION_MIN].get(rid)
            if now is None or not now.win:
                continue
            s_now = win_shares(now.win)
            for m, col in zip(LATE_FROM_MIN, columns, strict=True):
                then = at[m].get(rid)
                if then is None or not then.win:
                    continue
                s_then = win_shares(then.win)
                out.loc[runners.index, col] = [
                    np.log(s_now[h] / s_then[h]) if h in s_now and h in s_then else np.nan
                    for h in runners["horse_no"]
                ]
    return out


def load_inputs(date_from, date_to, log=print) -> tuple[pd.DataFrame, dict, dict, dict]:
    """(frame with every loaded member's features, counts, failed members, data builds)."""
    frame, counts = study_frame(date_from, date_to)
    log(f"study frame: {counts['races']} races")
    failed: dict[str, str] = {}
    builds = {
        **store_builds(),
        "jra_dividends": f"processed/jra/dividends.parquet@"
        f"{file_digest(paths.processed_dir('jra') / 'dividends.parquet')[:16]}",
    }

    def attempt(member: str, load: Callable[[], None]) -> None:
        started = clock.monotonic()
        try:
            load()
        except Exception as exc:  # the member's variants fail; the run goes on
            failed[member] = f"{type(exc).__name__}: {exc}"
        log(f"member {member}: {'failed' if member in failed else 'loaded'} "
            f"({clock.monotonic() - started:.0f} s)")  # fmt: skip

    def ability() -> None:
        nonlocal frame
        export, stem = ability_export()
        frame = frame.merge(export, on=["race_id", "horse_id"], how="left", validate="many_to_one")
        builds["ability_export"] = f"features/{COUNTRY}/ability@{stem}"

    def pedigree() -> None:
        sires = c2_ability.read_sires()
        rows = frame[["race_date", "surface", "distance_m"]].assign(
            sire_id=frame["horse_id"].map(sires.set_index("horse_id")["sire_id"])
        )
        frame["pedigree_prior"] = prior_at(pedigree_runs(sires), rows).to_numpy()
        builds.update(c2_ability.pedigree_build())

    def jrdb() -> None:
        nonlocal frame
        frame = pd.concat([frame, jrdb_workouts(frame)], axis=1)
        path = paths.processed_dir("jra") / f"{w137.TABLE}.parquet"
        builds["jrdb_workouts"] = f"processed/jra/{w137.TABLE}.parquet@{file_digest(path)[:16]}"

    def jravan() -> None:
        nonlocal frame
        frame = pd.concat([frame, jravan_workouts(frame, log)], axis=1)
        path = paths.processed_dir("jravan") / "workouts.parquet"
        builds["jravan_workouts"] = f"processed/jravan/workouts.parquet@{file_digest(path)[:16]}"

    def late() -> None:
        nonlocal frame
        frame = pd.concat([frame, late_win(frame)], axis=1)
        path = paths.processed_dir("jravan") / "odds_timeseries" / "_build.json"
        builds["jravan_odds_timeseries"] = (
            f"processed/jravan/odds_timeseries@{file_digest(path)[:16]}"
        )

    for member, load in (
        ("ability", ability),
        ("pedigree_prior", pedigree),
        ("workouts_jrdb", jrdb),
        ("workouts_jravan", jravan),
        ("late_win", late),
    ):
        attempt(member, load)
    loaded = [m for m in BUILT if m not in failed]
    features = member_features(frame, at_utc_col="decision_utc", members=loaded)
    keep = ["race_id", "race_date", "meeting_id", "race_class", "horse_no", "won", "q", "ln_q",
            "win_odds", "payout"]  # fmt: skip
    frame = pd.concat([frame[keep], features], axis=1)
    for member in loaded:
        for column in feature_columns(member):
            counts[f"known[{column}]"] = int(frame[column].notna().sum())
    return frame, counts, failed, builds


# --------------------------------------------------------------------------- results and reports


@dataclass
class DebutResult:
    spec: dict
    study: Study
    spec_hash: str = ""
    report_path: str = ""
    cells_path: str = ""
    run_ids: dict[str, str] = field(default_factory=dict)
    failed_members: dict[str, str] = field(default_factory=dict)


def _spec_hash(spec: dict) -> str:
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


REPORT_STEM = f"debut-members-{COUNTRY}-<spec hash>-<run digest>"


def report_stem(spec_hash: str, run: dict, builds: dict) -> str:
    """The private report's stem: the spec hash and a digest of the run (its date range) and its
    input builds, so a run that differs in either never overwrites another's report."""
    identity = json.dumps({"run": run, "builds": builds}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"debut-members-{COUNTRY}-{spec_hash[:12]}-{digest[:12]}"


def scored_any(result: DebutResult) -> bool:
    """Whether any variant scored (the command's exit status)."""
    return any(v.error is None for v in result.study.variants)


def render_public_report(result: DebutResult) -> str:
    """The public report: what ran and where its results live, with no figure (D21). It reads
    the spec and the variants' names and statuses only."""
    spec = result.spec
    spec_hash = _spec_hash(spec)
    members = [
        {
            "member": f"`{name}`",
            "ticket": f"#{m['ticket']}",
            "status": m["status"] if m["status"] == "built" else "missing: listed, left out",
            "sources": ", ".join(m["sources"]),
            "about": m["about"],
        }
        for name, m in spec["members"].items()
    ]
    variants = [
        {
            "E": v.hypothesis_id,
            "variant": f"`{v.name}`",
            "status": "failed" if v.error else "scored",
            "ΔLL, bounds, weight, encompassing weight, N5": "withheld (D21)",
        }
        for v in result.study.variants
    ]
    lines = [
        "# Experiment report: newcomer and maiden members at the final Win price (Japan, #438)",
        "",
        "Ticket #438 (epic #73; N7, E34, E37). Each member that could carry signal for a debut or "
        "maiden runner is fitted beside the final Win price, one at a time and all together, per "
        "class (DEBUT, MAIDEN), as a conditional logit with ln q as its offset on D9's folds. "
        "**Nothing is adopted** (D16-D19): every variant is a ranked row with its encompassing "
        "weight beside it. The holdout is not opened. **Every figure is withheld (D21)**: the "
        "members read JRA-VAN and JRDB data, so this report gives no count, span, score, rate, "
        "sign or bound; each figure is in the private report and the registry records it names.",
        "",
        f"- Registry family `{FAMILY_ID}`, spec sha256 `{spec_hash}`, frozen before the run read "
        f"any store row. The pre-registration is `{SPEC_DOC}`.",
        "- Code: `countries/japan/studies/debut_members.py`. To reproduce, run "
        "`uv run horsetrack japan debut-members`.",
        f"- Private report: `$HORSETRACK_DATA_DIR/reports/{REPORT_STEM}.md` (the spec hash, then "
        "a digest of the run's date range and input builds, so no run overwrites another), its "
        f"tables as JSON beside it, and the ranked rows in `$HORSETRACK_DATA_DIR/reports/"
        f"{CELLS_DIR}/` under the same stem, all written through `write_report` with the sources "
        f"{', '.join(f'`{s}`' for s in spec['sources'])}.",
        f"- Records: `docs/experiments/registry/{COUNTRY}/{FAMILY_ID}.md`; every one is withheld.",
        f"- Supersedes `{spec['supersedes']['family_id']}`, kept in the record with its records "
        f"and summary: {spec['supersedes']['reason']}.",
        "",
        "## Hypothesis",
        "",
        *(f"- **{k}.** {v}" for k, v in spec["hypotheses"].items()),
        f"- **Null**, the starting point: {spec['null']}.",
        "",
        "## Members",
        "",
        _markdown_table(members),
        "",
        "## Variants tried",
        "",
        "One registry record each, failures included. Each is scored by its per-race winner ΔLL "
        "against the final price with D7's one-sided meeting bounds, its weights, its "
        "encompassing weight and N5's curve settled at each winner's own Win dividend (a race "
        "with a winner that has no Win dividend row is left out of the curve and counted in "
        "the private report; Place after #410).",
        "",
        _markdown_table(variants) if variants else "_No variant has run._",
        "",
        "## Alternatives rejected and why",
        "",
        "- A fit pooled over both classes: the members' meaning differs for a debut runner (no "
        "run, no weight change), so each class has its own fit.",
        "- A member's raw values without a missing flag: a debut runner lacks some members by "
        "construction, and the flag keeps that apart from a value at the mean.",
        "- The JV-Data mining member (#406): not landed; it is listed and enters as a counted "
        "variant when it lands.",
        "- Settling a race without a Win dividend row at its final odds, and cutting the late "
        "Win boards by the JRA-VAN post's minutes to post: both were v1's, replaced in this "
        "family (the board cut at the study's JRA post; such a race left out of the curve).",
        "",
        "## What this rules out",
        "",
        "Withheld (D21): the measured gains, weights and curves are in the private report.",
        "",
        "## Leaderboard rows",
        "",
        "One row per variant above, by its name; built on licensed sources, so no rank or figure "
        "is shown here (D21).",
        "",
        "## Follow-ups filed",
        "",
        "- #406: the JV-Data mining member, a counted variant when it lands.",
        "- #410: the Place settlement of the N5 curve.",
        "",
    ]
    return "\n".join(lines)


def render_report(result: DebutResult, about: dict) -> str:
    """The private report: aggregate tables only, never a runner or race row."""
    study, spec = result.study, result.spec
    lines = [
        f"# Newcomer and maiden members at the final Win price ({COUNTRY}, #{TICKET})",
        "",
        "Built on JRA-VAN and JRDB data: every figure here is withheld from the repository (D21).",
        "",
        *(f"- {k}: {v}" for k, v in about.items()),
        "",
        "## Pre-registered (frozen)",
        "",
        *(f"- {k}: {spec[k]}" for k in ("races", "decision_time", "offset", "design", "fit")),
        *(f"- {k}: {v}" for k, v in spec["metrics"].items()),
        f"- n5: {spec['n5']}",
        "",
        "## Data",
        "",
        _markdown_table([{"count": k, "n": v} for k, v in study.counts.items()]),
    ]
    for key, title in (
        ("ranked", "Ranked rows (ΔLL; D7 one-sided bounds; encompassing weight beside)"),
        ("weights", "Weights pooled over the scored rows (meeting-clustered, one-sided bounds)"),
        ("n5_curve", "N5: realised Win return per unit stake by band of p x O"),
    ):
        rows = study.tables.get(key)
        lines += ["", f"## {title}", "", _markdown_table(rows) if rows else "_Not computed._"]
    variants = [
        {"E": v.hypothesis_id, "variant": v.name, "status": "failed" if v.error else "scored"}
        | {"record": result.run_ids.get(v.name, "")[:16]}
        | ({"error": v.error} if v.error else {})
        for v in study.variants
    ]
    lines += ["", "## Variants (one registry record each)", "", _markdown_table(variants), ""]
    return "\n".join(lines)


def _record(variant: Variant, git_sha: str, builds: dict, spec_hash: str, run: dict) -> RunRecord:
    params = dict(variant.params) | run | {"spec_sha256": spec_hash}
    if variant.error:
        params["error"] = variant.error
    return RunRecord(
        hypothesis_id=variant.hypothesis_id,
        family_id=FAMILY_ID,
        variant=variant.name,
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=dict(builds),
        sources=SOURCES,
        fold_scheme=variant.fold_scheme,
        components=variant.components,
        params=params,
        metrics=variant.metrics,
        status="failed" if variant.error else "scored",
    )


def write_committed_summary(registry) -> None:
    """This family's summary file and the committed index with its line, by `write_summary`
    over this family alone (#92's `write_committed_summary` rule: every family the index lists
    stays and no other family's file is written)."""
    directory = summary_dir(COUNTRY)
    mine = c2_ability._Families(registry, {FAMILY_ID})
    if not mine.records():
        return
    with tempfile.TemporaryDirectory() as scratch:
        write_summary(mine, scratch)
        write_report(
            directory / f"{FAMILY_ID}.md",
            Path(scratch, f"{FAMILY_ID}.md").read_text("utf-8"),
            sources=(),
        )
        fresh = Path(scratch, "README.md").read_text("utf-8").splitlines()
    index = directory / "README.md"
    committed = index.read_text("utf-8").splitlines() if index.exists() else []
    found = c2_ability._index_family
    entries = {family: line for line in committed + fresh if (family := found(line))}
    head = [line for line in fresh if not found(line)]
    lines = [*head, *(entries[family] for family in sorted(entries))]
    write_report(index, "\n".join(lines) + "\n", sources=())


def run_debut_members(
    *,
    date_from: date = PRICE_ERA_START,
    date_to: date | None = None,
    out_dir: Path | None = None,
    seed: int = SEED,
    registry=None,
    log: Callable[[str], None] = print,
) -> DebutResult:
    """Run the pre-registered family on Japan's development data: freeze, read, score, append
    one record per variant (failures included), write the private report, its JSON and the
    candidate-cell file. With the default registry, also regenerate the committed summary and
    the public report. A failed read or fold split fails every planned variant, which is
    recorded and reported like any other failure; the result is returned, not raised."""
    date_to = date.today() if date_to is None else date_to
    out_dir = (
        paths.reports_dir() if out_dir is None else paths.outside_repo(Path(out_dir), what="out")
    )
    committed = registry is None
    registry = default_registry(COUNTRY) if registry is None else registry
    spec = build_spec(seed=seed)
    spec_hash = registry.freeze_family(FAMILY_ID, spec)  # before any store row is read
    git_sha = _git_sha()
    run = {"date_from": date_from.isoformat(), "date_to": date_to.isoformat()}
    builds: dict[str, str] = {}
    failed: dict[str, str] = {}
    started = clock.monotonic()
    try:
        frame, counts, failed, builds = load_inputs(date_from, date_to, log)
        folds = walk_forward(frame["race_date"], scheme=load_country_scheme(COUNTRY))
        study = compute_study(frame, folds, spec, failed)
        study.counts = {**counts, "folds": len(folds), **study.counts}
        del frame
    except Exception as exc:  # every planned variant fails, and is recorded and reported
        error = f"{type(exc).__name__}: {exc}"
        fold_scheme = "walk_forward:" + json.dumps(spec["folds"], sort_keys=True)
        study = Study(counts={})
        for name in variant_names():
            member = name.split(":")[1] if name.startswith("member:") else COMBINED
            components = {"study": "studies/debut_members.py"}
            study.variants.append(
                Variant(_hypothesis(member), name, fold_scheme, components, {}, error=error)
            )
    log(f"computed ({clock.monotonic() - started:.0f} s)")

    held = {_run_id(dataclasses.replace(r, git_sha="")): r for r in registry.records(FAMILY_ID)}
    result = DebutResult(spec=spec, study=study, spec_hash=spec_hash, failed_members=failed)
    for variant in study.variants:
        record = _record(variant, git_sha, builds, spec_hash, run)
        same = held.get(_run_id(dataclasses.replace(record, git_sha="")))
        result.run_ids[variant.name] = registry.append(record) if same is None else _run_id(same)
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "data": ", ".join(builds.values()),
        "run": json.dumps(run),
        "failed members": json.dumps(failed) if failed else "none",
        "run time (s)": round(clock.monotonic() - started),
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    stem = report_stem(spec_hash, run, builds)
    result.report_path = str(
        write_report(out_dir / f"{stem}.md", render_report(result, about), sources=SOURCES)
    )
    write_report(
        out_dir / f"{stem}.json",
        json.dumps({"about": about, "counts": study.counts, "tables": study.tables}, indent=1,
                   default=str),
        sources=SOURCES,
    )  # fmt: skip
    cells = [
        {"study": "debut-members", "family_id": FAMILY_ID, "cell_id": row["variant"]} | row
        for row in study.tables.get("ranked", [])
    ]
    result.cells_path = str(
        write_report(
            out_dir / CELLS_DIR / f"{stem}.json",
            json.dumps(cells, indent=1, default=str),
            sources=SOURCES,
        )
    )
    if committed:
        write_committed_summary(registry)
        write_report(env.repo_root() / PUBLIC_REPORT, render_public_report(result), sources=())
    return result


def debut_members_command(
    date_from: Annotated[
        str, typer.Option("--from", help="First race day read (D10: not before 2014-06-07).")
    ] = PRICE_ERA_START.isoformat(),
    date_to: Annotated[str | None, typer.Option("--to", help="Last race day read.")] = None,
    seed: Annotated[int, typer.Option(help="The bootstrap seed the spec freezes.")] = SEED,
) -> None:
    """Fit the newcomer and maiden members at the final Win price (#438) and write the private
    report. Exits nonzero when no variant scored (the failure records and the report are still
    written)."""
    result = run_debut_members(
        date_from=date.fromisoformat(date_from),
        date_to=None if date_to is None else date.fromisoformat(date_to),
        seed=seed,
        log=typer.echo,
    )
    typer.echo(
        json.dumps(
            {
                "family_id": FAMILY_ID,
                "spec_hash": result.spec_hash,
                "report_path": result.report_path,
                "cells_path": result.cells_path,
                "run_ids": {k: v[:16] for k, v in result.run_ids.items()},
                "failed_variants": [v.name for v in result.study.variants if v.error],
                "failed_members": result.failed_members,
            },
            indent=2,
        )
    )
    if not scored_any(result):
        typer.echo("no variant scored", err=True)
        raise typer.Exit(1)
