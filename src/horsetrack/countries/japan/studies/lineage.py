"""E37 (#95, #538, epic #73; 08 §4 E37): the lineage prior fed into #92's ability member for
debut and thin-history runners, scored on debut, first-time-surface and first-time-distance races,
standalone and at the final price: `horsetrack japan lineage-study`.

Scoring (#95). `run_segments` flags each run from that horse's earlier runs only (strictly
earlier race_utc): `debut` when it has none; `first_surface` when it has some but none on this
surface; `first_distance` when it has some but none in this dist_band (the caller's band label,
whose cut points belong to the study's frozen spec). A run with a missing surface or band is
never first on it. `segment_delta_ll` scores two forecasts' P(win) on the same rows by #69's
per-race winner log loss (each race renormalised over its rows, dead heats at 1/k): the per-race
ΔLL is ln p_a(winner) - ln p_b(winner), positive when a is better. A race is in a segment when
any of its rows is flagged; a segment's bounds are D7's meeting bootstrap of its per-race ΔLL
(`core.eval.bootstrap.paired_bootstrap` at spec["bootstrap"]). The same function scores both of
E37's comparisons: standalone (the ability member with the lineage prior against it without) and
at the final price (the stage-2 meta with the lineage member against the one without, or
against the market).

The as-of feature (#538). `lineage_feature` gives each run the prior mean of
`models.lineage.LineagePrior` fitted on the runs before its quarter's start (00:00 UTC of the D9
quarter's first day; no trainer effect) at its horse, surface and band, when the horse has fewer
than THIN_STARTS prior starts (its runs in the frame on earlier race days: a debut has 0, a run
without a placing counts), and 0 otherwise; a run before the first quarter start gets 0. One prior
is fitted per quarter, so a row's feature, training row or test row alike, reads only runs before
its quarter. `netkeiba_pedigree` keys the netkeiba `horse_pedigree` (whose id columns are empty)
on its normalised names and refuses a table carrying any UKC (JRDB) column;
`read_netkeiba_pedigree` reads the store's file through it.

The run (`run_study`; the spec `build_spec`, frozen before any outcome is read). Family
`japan-c2-lineage-e37-netkeiba`: JRA results, the netkeiba pedigree and the JRA final prices
only, so its figures may be public. Level 1 is #92's (`studies.ability`: the same frame, folds,
rating member, discount and pair selection) with the netkeiba sire name in place of the JRA-VAN
sire (the base member). Each lineage variant (tau_cell x damsire_weight, the band scheme of #438)
adds its `lineage_feature` as an unpenalised feature at the base member's selection per fold.
Level 2 is #92's: the store frame, D4's baseline out of fold, the stage-2 meta of each member and
the market-only stage-2 meta, on the D9 folds of the netkeiba coverage window. Every comparison
is a registry record (failures included), each lineage variant x segment one ranked member row;
the report goes under the data dir through `write_report`, and the committed summary is
`write_summary`'s. Holdout days are never read; nothing is adopted. A `--trial` run uses the
shorter plan of the spec, records to the `-trial` family and never freezes the family's spec.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import typer

from horsetrack.core import paths
from horsetrack.core.eval.folds import load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha
from horsetrack.core.eval.registry import RunRecord, default_registry
from horsetrack.core.eval.reports import write_report
from horsetrack.core.eval.scoring import gain
from horsetrack.core.store.schemas import TABLES
from horsetrack.countries.japan.jrdb.pedigree import OUTPUT_COLUMNS as UKC_OUTPUT_COLUMNS
from horsetrack.countries.japan.models.ability import DISCOUNT_GRID, AbilityModel
from horsetrack.countries.japan.models.lineage import LineagePrior
from horsetrack.countries.japan.models.meta import MetaSpec
from horsetrack.countries.japan.studies import ability as c2_ability
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
from horsetrack.countries.japan.studies.debut_members import (
    BAND_EDGES,
    BAND_LABELS,
    SURFACES,
    distance_band,
)
from horsetrack.countries.japan.studies.meta import (
    Scored,
    _ll,
    _meta_metrics,
    _meta_oof,
    _pooled,
    _try,
    baseline_oof,
    read_development,
    restrict,
    store_frame,
)

SEGMENTS = ("debut", "first_surface", "first_distance")
OUT_COLUMNS = ["segment", "races", "meetings", "delta_ll", "lower", "upper"]

COUNTRY = "japan"
TICKET = 538
FAMILY_ID = "japan-c2-lineage-e37-netkeiba"
TRIAL_FAMILY_ID = f"{FAMILY_ID}-trial"  # a trial's records; a trial never freezes FAMILY_ID
SOURCES = ("jra",)  # JRA results and prices and the netkeiba pedigree: no JRDB, no JRA-VAN
BOOTSTRAP_SEED = 538  # reproducibility only; scoring.yaml signs no seed
THIN_STARTS = c2_ability.THIN_STARTS  # fewer prior starts than this: the feature is on
PEDIGREE_FILE = "processed/jra/horse_pedigree.parquet"
# The columns a UKC (JRDB) merge adds to horse_pedigree beyond the netkeiba store schema.
UKC_COLUMNS = tuple(c for c in UKC_OUTPUT_COLUMNS if c not in TABLES["horse_pedigree"].columns)
NAME_COLUMNS = ("sire_name_norm", "dam_name_norm", "dam_sire_name_norm")

# The pre-registered grid (frozen in the spec; 08 §3.3's learnable tau range, #438's bands).
TAU_CELL_GRID = (0.1, 0.3)
DAMSIRE_WEIGHTS = (0.0, 0.25)
PRIOR_K = 3  # LineagePrior's top-3 stages at its default discounts (1, 1, 1)
BAND_SCHEME = "b4"  # #438's: up to and including 1400, 1800, 2200 m, and longer
COVERAGE_MIN = 0.8  # the window: quarters whose starters have a netkeiba sire at this share
BASE = "ability_netkeiba"  # #92's pooled member with the netkeiba sire
LINEAGE = "lineage"  # the feature column a lineage member's AbilityModel reads
ALL = "all"  # the every-race row beside the segments
COMPARISONS = ("standalone", "stage2_vs_base", "stage2_vs_market_only")


def variant_key(tau_cell: float, damsire_weight: float) -> str:
    return f"lineage_t{tau_cell:g}_d{damsire_weight:g}"


VARIANTS = {
    variant_key(t, w): {"tau_cell": t, "damsire_weight": w, "k": PRIOR_K}
    for t in TAU_CELL_GRID
    for w in DAMSIRE_WEIGHTS
}


@dataclasses.dataclass(frozen=True)
class Plan:
    """The level-1 dates and discount grid of a run: the full run's are #92's."""

    results_from: date
    inner_from: date
    level1_from: date
    discount_grid: tuple

    def as_spec(self) -> dict:
        return {
            "results_from": self.results_from.isoformat(),
            "inner_from": self.inner_from.isoformat(),
            "level1_out_of_fold_from": self.level1_from.isoformat(),
            "discount_grid": [list(w) for w in self.discount_grid],
        }


FULL = Plan(c2_ability.RESULTS_FROM, c2_ability.INNER_FROM, c2_ability.LEVEL1_FROM, DISCOUNT_GRID)
TRIAL = Plan(
    date(2021, 1, 1),
    date(2022, 1, 1),
    date(2023, 1, 1),
    (c2_ability.WINNER_ONLY, (1.0, 0.5, 0.25), (1.0, 1.0, 1.0)),
)

__all__ = [
    "FAMILY_ID",
    "OUT_COLUMNS",
    "SEGMENTS",
    "SOURCES",
    "lineage_feature",
    "lineage_study_command",
    "netkeiba_pedigree",
    "read_netkeiba_pedigree",
    "run_segments",
    "run_study",
    "segment_delta_ll",
]


# --------------------------------------------------------------------------- #95's scoring


def _none_earlier(times: pd.Series, keys: list[pd.Series]) -> np.ndarray:
    """Per row: True when no row with the same keys has an earlier time (False for a missing
    key)."""
    rank = times.groupby(keys, sort=False, dropna=True).rank(method="min")
    return rank.eq(1).to_numpy(dtype=bool)


def run_segments(runs: pd.DataFrame) -> pd.DataFrame:
    """One bool column per SEGMENTS entry, on runs' index; see the module docstring."""
    missing = [c for c in ("race_utc", "horse_id", "surface", "dist_band") if c not in runs.columns]
    if missing:
        raise ValueError(f"runs lacks column(s) {missing}")
    times = pd.Series(pd.to_datetime(runs["race_utc"], utc=True).to_numpy(), index=runs.index)
    horse = runs["horse_id"]
    debut = _none_earlier(times, [horse])
    seen = ~debut & horse.notna().to_numpy()
    return pd.DataFrame(
        {
            "debut": debut,
            "first_surface": seen & _none_earlier(times, [horse, runs["surface"]]),
            "first_distance": seen & _none_earlier(times, [horse, runs["dist_band"]]),
        },
        index=runs.index,
    )[list(SEGMENTS)]


def segment_delta_ll(
    p_a: pd.Series, p_b: pd.Series, rows: pd.DataFrame, spec: dict
) -> pd.DataFrame:
    """Per segment: races, meetings, the mean per-race winner ΔLL of p_a against p_b and its
    meeting-bootstrap bounds (spec["bootstrap"]); see the module docstring."""
    missing = [c for c in ("race_id", "meeting_id", "won", *SEGMENTS) if c not in rows.columns]
    if missing:
        raise ValueError(f"rows lacks column(s) {missing}")
    per_race = gain(_ll(p_a, rows), _ll(p_b, rows)).per_race
    clusters = rows.groupby(rows["race_id"].to_numpy())["meeting_id"].first()
    out = []
    for segment in SEGMENTS:
        flagged = rows[segment].fillna(False).to_numpy(dtype=bool)
        races = pd.unique(rows["race_id"].to_numpy()[flagged])
        part = per_race.loc[per_race.index.isin(races)]
        row = {"segment": segment, "races": len(part), "meetings": 0}
        row |= {"delta_ll": np.nan, "lower": np.nan, "upper": np.nan}
        if len(part):
            iv = _bounds(part, clusters, spec)
            row |= {"meetings": iv.n_clusters, "delta_ll": iv.mean, "lower": iv.lo, "upper": iv.hi}
        out.append(row)
    return pd.DataFrame(out, columns=OUT_COLUMNS)


# --------------------------------------------------------------------------- the as-of feature


def _days(values) -> np.ndarray:
    return pd.to_datetime(pd.Series(values)).to_numpy().astype("datetime64[D]")


def quarter_of(race_date, quarter_starts) -> np.ndarray:
    """Per row: the index of the last quarter start on or before its race date (-1 before the
    first)."""
    edges = _days([pd.Timestamp(s) for s in quarter_starts])
    if len(edges) and (np.diff(edges.astype(np.int64)) <= 0).any():
        raise ValueError("quarter_starts must be strictly ascending")
    return np.searchsorted(edges, _days(race_date), side="right") - 1


def prior_starts(runs: pd.DataFrame) -> np.ndarray:
    """Per row: the horse's rows in runs on earlier race days (0 on a debut or a missing
    horse)."""
    day = pd.Series(_days(runs["race_date"]).astype(np.int64))
    horse = pd.Series(runs["horse_id"].to_numpy(dtype=object))
    rank = day.groupby(horse, sort=False, dropna=True).rank(method="min")
    return (rank.fillna(1.0).to_numpy() - 1.0).astype(np.int64)


def _lineage_feature(
    runs: pd.DataFrame,
    pedigree: pd.DataFrame,
    quarter_starts,
    prior: Mapping,
    log: Callable[[str], None] | None = None,
) -> np.ndarray:
    starts = [pd.Timestamp(s).date() for s in quarter_starts]
    q = quarter_of(runs["race_date"], starts)
    on = (prior_starts(runs) < THIN_STARTS) & (q >= 0)
    out = np.zeros(len(runs))
    started = time.monotonic()
    for k in np.unique(q[on]):
        fitted = LineagePrior(**prior).fit(
            runs, pedigree, before_utc=pd.Timestamp(starts[k]).tz_localize("UTC")
        )
        rows = np.flatnonzero(on & (q == k))
        cells = runs.iloc[rows][["horse_id", "surface", "dist_band"]]
        out[rows] = fitted.prior_means(cells).to_numpy()
        if log is not None:
            log(
                f"    prior at {starts[k]}: {len(fitted.sire_effects_)} sires, "
                f"{len(fitted.cell_effects_)} cells ({time.monotonic() - started:.0f} s)"
            )
    return out


def lineage_feature(
    runs: pd.DataFrame, pedigree: pd.DataFrame, quarter_starts, **prior
) -> pd.Series:
    """Per row of runs, on its index: the prior mean of LineagePrior(**prior) fitted before its
    quarter's start, for a horse with fewer than THIN_STARTS prior starts, else 0.0; see the
    module docstring."""
    values = _lineage_feature(runs, pedigree, quarter_starts, prior)
    return pd.Series(values, index=runs.index, dtype=float, name=LINEAGE)


def _refuse_ukc(columns) -> None:
    merged = [c for c in UKC_COLUMNS if c in set(columns)]
    if merged:
        raise ValueError(
            f"the pedigree carries UKC (JRDB) column(s) {merged}: E37's netkeiba family reads "
            "the netkeiba horse_pedigree only (D21)"
        )


def _name(values: pd.Series) -> np.ndarray:
    """Names as objects, a missing or blank name None."""
    out = values.to_numpy(dtype=object).copy()  # pandas 3 hands out a read-only view
    blank = pd.isna(out) | np.array([isinstance(v, str) and not v.strip() for v in out], bool)
    out[blank] = None
    return out


def netkeiba_pedigree(table: pd.DataFrame) -> pd.DataFrame:
    """LineagePrior's pedigree of a netkeiba horse_pedigree frame: one row per horse_id, sire_id
    its sire_name_norm, dam_id its dam_name_norm and dam_sire_id its dam_sire_name_norm (a
    missing name None). ValueError for a table with a UKC (JRDB) column or a horse twice."""
    _refuse_ukc(table.columns)
    missing = [c for c in ("horse_id", *NAME_COLUMNS) if c not in table.columns]
    if missing:
        raise ValueError(f"the pedigree lacks column(s) {missing}")
    rows = table.loc[table["horse_id"].notna().to_numpy(dtype=bool)]
    if rows["horse_id"].duplicated().any():
        raise ValueError("the pedigree has more than one row for a horse")
    return pd.DataFrame(
        {
            "horse_id": rows["horse_id"].to_numpy(dtype=object),
            "sire_id": _name(rows["sire_name_norm"]),
            "dam_id": _name(rows["dam_name_norm"]),
            "dam_sire_id": _name(rows["dam_sire_name_norm"]),
        },
        dtype=object,
    )


def read_netkeiba_pedigree() -> pd.DataFrame:
    """`netkeiba_pedigree` of the store's processed/jra/horse_pedigree.parquet (static: no date,
    outcome or price), its name columns only; a file a UKC merge wrote back is refused."""
    path = paths.processed_dir("jra") / "horse_pedigree.parquet"
    _refuse_ukc(pq.read_schema(path).names)
    return netkeiba_pedigree(pq.read_table(path, columns=["horse_id", *NAME_COLUMNS]).to_pandas())


def pedigree_build() -> dict[str, str]:
    digest = hashlib.sha256()
    with open(paths.processed_dir("jra") / "horse_pedigree.parquet", "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return {"jra_horse_pedigree": f"{PEDIGREE_FILE}@{digest.hexdigest()[:16]}"}


# --------------------------------------------------------------------------- the spec


def build_spec(scheme, bootstrap: dict, plan: Plan = FULL) -> dict:
    """The family's pre-registered spec, frozen in the registry before any outcome is read."""
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypotheses": {
            "E37": "a sire x surface x distance-band lineage prior (the damsire at a weight) fed "
            "into #92's ability member for debut and thin-history runners improves its winner "
            "log loss on debut, first-time-surface and first-time-distance races, standalone "
            "and at the final price (08 §4 E37; expected small: pedigree-only correlation about "
            "0.08 and a strong market in debut and maiden races, 08 §3.3)",
        },
        "null": "the lineage feature adds nothing: ΔLL <= 0 on every segment, standalone and at "
        "the final price",
        "decision_rules": {
            "standalone": "per variant and segment: the lineage member alone beats the base "
            "member alone when the meeting-bootstrap one-sided 95% lower bound of the paired "
            "per-race winner ΔLL (lineage - base, both renormalised over the race's priced "
            "starters) over the segment's scored races lies above 0",
            "final price": "per variant and segment: the same bound for the stage-2 meta with "
            "the lineage member against the stage-2 meta with the base member",
            "against the market": "descriptive: the stage-2 meta with the lineage member against "
            "the market-only stage-2 meta, per segment, with its bounds",
            "base marginal": "descriptive: the stage-2 meta with the base member against the "
            "market-only stage-2 meta, per segment",
            "stage-2": "descriptive: each member's stage-2 meta and the market-only meta against "
            "D4's calibrated market (out-of-fold ΔLL with meeting-bootstrap bounds), with each "
            "fold's coefficients (Wald 95% two-sided; meeting-clustered one-sided in the report)",
            "level 1": "descriptive: each lineage member's coefficient on the feature per outer "
            "fold",
            "ranking": "nothing is adopted (D16-D19): each lineage variant x segment (debut, "
            "first_surface, first_distance, all) is one ranked member row; the family size "
            "counts every variant searched; no multiplicity correction is applied",
        },
        "segments": {
            "flags": "studies.lineage.run_segments over the level-1 frame (every starter since "
            "results_from): debut, first_surface (an earlier run, none on this surface), "
            "first_distance (an earlier run, none in this band); a race is in a segment when any "
            "of its priced starters is flagged",
            "segments": [*SEGMENTS, ALL],
            "comparisons": list(COMPARISONS),
        },
        "level1": {
            "frame": "studies.ability.level1_frames: every not-scratched starter of the JRA "
            "store's run races with a winner from results_from; no price is read",
            "sire": f"the netkeiba {PEDIGREE_FILE} sire_name_norm of runners.horse_id (keyed on "
            "names: its id columns are empty); a horse without a row has no sire level; a table "
            "carrying a UKC (JRDB) column is refused; no JRA-VAN column is read",
            "base_member": "#92's pooled member (models.ability.AbilityModel k = 3: rating + "
            "jockey, trainer and sire effects shrunk by empirical Bayes) with the netkeiba sire; "
            "#91's rating pair and the discounts selected per outer fold on its inner folds by "
            "#92's rules (studies.ability.select_pairs, select_discounts)",
            "lineage_member": "the base member at the fold's selected rating pair and discounts "
            f"(not reselected) with the unpenalised feature `{LINEAGE}`: an ablation of one "
            "feature",
            "folds": "D9's quarters over the level-1 dates (expanding, the scheme's embargo); "
            "inner folds every quarter from inner_from; out of fold from "
            "level1_out_of_fold_from",
            **plan.as_spec(),
        },
        "lineage": {
            "prior": "models.lineage.LineagePrior: u = g(sire, cell) + damsire_weight * "
            "g(damsire, cell), g(s, c) = g_sire[s] + d[s, c], d ~ N(0, tau_cell^2), tau_sire^2 by "
            f"empirical Bayes; top-{PRIOR_K} Plackett-Luce at discounts (1, 1, 1); no trainer "
            "effect",
            "pedigree": "studies.lineage.netkeiba_pedigree: sire_id = sire_name_norm, dam_id = "
            "dam_name_norm, dam_sire_id = dam_sire_name_norm",
            "as_of": "one prior per D9 quarter (starts from results_from), fitted on the level-1 "
            "runs before 00:00 UTC of the quarter's first day; every row reads its quarter's "
            "prior (studies.lineage.lineage_feature)",
            "feature": f"the prior mean at the row's horse, surface and band when the horse has "
            f"fewer than {THIN_STARTS} prior starts (its level-1 runs on earlier race days; a "
            "debut has 0, a run without a placing counts), else 0",
            "surfaces": list(SURFACES),
            "band_scheme": {
                "name": BAND_SCHEME,
                "edges_m": list(BAND_EDGES),
                "labels": list(BAND_LABELS),
                "rule": "a band holds the distances up to and including its edge (#438's "
                "studies.debut_members.distance_band); a race on another surface (jump) has "
                "neither a surface nor a band label: in no cell, never first on either",
            },
            "race_utc": "the race day at 12:00 JST: every use here reads only the race day (a "
            "horse runs once a day; JRA posts are after 00:00 UTC of the day)",
            "variants": {key: dict(v) for key, v in VARIANTS.items()},
        },
        "window": {
            "rule": f"the scored folds are the level-2 D9 folds from the first quarter Q such "
            f"that Q and every later quarter have at least {COVERAGE_MIN:.0%} of their level-1 "
            "starters with a netkeiba sire (a count: no outcome read), and whose training rows "
            "hold a level-1 member",
            "coverage_min": COVERAGE_MIN,
        },
        "level2": {
            "frame": "studies.meta.store_frame: the priced starters of run races with a priced "
            "winner from the price era; q the normalised final Win price; a race with a starter "
            "lacking any member is dropped",
            "baseline": "D4 calibrated market (studies.meta.baseline_oof) on D9's folds",
            "member": "ln p of each level-1 member renormalised over the race's priced starters",
            "meta": "models.meta.fit_meta stage2 on [member] and ln q, ridge 0, unbounded, on "
            "each scored fold's training rows that have every member (#92's)",
            "market_only": "studies.ability's market-only stage-2 meta (u = beta * ln q) on the "
            "same rows and folds",
            "folds": dataclasses.asdict(scheme),
        },
        "bootstrap": bootstrap,
        "jravan_variant": "not run in this family: the JRA-VAN pedigree over the full window "
        "would be a separate, withheld family (D21); a follow-up",
        "trial": "a --trial run uses results from 2021-01-01, inner folds from 2022-01-01, out of "
        "fold from 2023-01-01 and the discounts (1, 0, 0), (1, 0.5, 0.25), (1, 1, 1); its records "
        "go to the -trial family and its spec is never frozen",
        "adoption": "none (D16-D19): no holdout is opened and nothing is adopted",
        "sources": list(SOURCES),
    }


# --------------------------------------------------------------------------- inputs


@dataclasses.dataclass
class Inputs:
    level1: pd.DataFrame  # #92's frame + surface, dist_band, race_utc
    pedigree: pd.DataFrame  # netkeiba_pedigree
    results: pd.DataFrame  # the rating's input
    price: pd.DataFrame  # studies.meta.store_frame
    counts: dict


def add_cells(level1: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """level1 with surface (turf or dirt, else None), dist_band (#438's bands, None off those
    surfaces) and race_utc (the race day at 12:00 JST)."""
    by_race = races.drop_duplicates("race_id").set_index("race_id")
    surface = level1["race_id"].map(by_race["surface"]).to_numpy(dtype=object)
    flat = pd.Series(surface).isin(SURFACES).to_numpy(dtype=bool)
    band = distance_band(level1["race_id"].map(by_race["distance_m"])).to_numpy(dtype=object)
    day = pd.to_datetime(level1["race_date"]).dt.tz_localize("UTC")
    return level1.assign(
        surface=np.where(flat, surface, None),
        dist_band=np.where(flat, band, None),
        race_utc=day + pd.Timedelta(hours=3),
    )


def load_inputs(plan: Plan = FULL, log=print) -> Inputs:
    columns = [*c2_ability.RACE_COLUMNS, "surface", "distance_m"]
    runners, races = c2_ability.read_runners(race_columns=columns)
    pedigree = read_netkeiba_pedigree()
    sires = pedigree[["horse_id", "sire_id"]]
    _, level1, counts = c2_ability.level1_frames(runners, races, sires)
    del runners
    level1 = level1.loc[level1["race_date"].ge(pd.Timestamp(plan.results_from))]
    level1 = add_cells(level1.reset_index(drop=True), races)
    counts |= {"level1_starters_in_plan": len(level1), "pedigree_horses": len(pedigree)}
    results = c2_rating.read_results()
    p_runners, p_races = read_development()
    price, price_counts = store_frame(p_runners, p_races)
    counts |= {f"level2_{k}": v for k, v in price_counts.items()}
    log(f"read {len(level1)} level-1 starters, {len(price)} priced starters")
    return Inputs(level1=level1, pedigree=pedigree, results=results, price=price, counts=counts)


# --------------------------------------------------------------------------- level 1


def quarter_starts(level1: pd.DataFrame, scheme, first: date) -> list[date]:
    """Every D9 quarter start from `first` (a quarter start) to the last level-1 date."""
    if scheme.unit != "quarter" or first.day != 1 or (first.month - scheme.anchor_month) % 3:
        raise ValueError(f"{first} is not a start of the scheme's quarters")
    last = pd.Timestamp(level1["race_date"].max()).date()
    out, at = [], first
    while at <= last:
        out.append(at)
        months = at.month + 2  # zero-based month three months on
        at = date(at.year + months // 12, months % 12 + 1, 1)
    return out


def coverage(level1: pd.DataFrame, starts: list[date]) -> tuple[date | None, list[dict]]:
    """Per quarter: the level-1 starters and the share with a netkeiba sire; the window's first
    quarter (the spec's rule), None when no quarter qualifies."""
    q = quarter_of(level1["race_date"], starts)
    known = level1["sire_id"].notna().to_numpy(dtype=bool)[q >= 0]
    q = q[q >= 0]
    n = np.bincount(q, minlength=len(starts))
    share = np.bincount(q, weights=known, minlength=len(starts)) / np.maximum(n, 1)
    holds = (share >= COVERAGE_MIN) | (n == 0)
    tail = np.flip(np.cumprod(np.flip(holds)).astype(bool))  # this and every later quarter hold
    first = next((starts[i] for i in range(len(starts)) if tail[i] and n[i]), None)
    table = [
        {"quarter": s.isoformat(), "starters": int(c), "netkeiba sire share": float(v)}
        for s, c, v in zip(starts, n, share, strict=True)
        if c
    ]
    return first, table


def base_members(level1, grid, outer, selection) -> np.ndarray:
    """The base member's u on the outer folds' test rows at each fold's selection."""
    u = np.full(len(level1), np.nan)
    for fold, pick in zip(outer, selection, strict=True):
        pair = (pick["drift_var_per_day"], pick["scale"])
        w = (1.0, pick["pooled w2"], pick["pooled w3"])
        u[fold.test_idx] = grid.u[("pooled", pair, w)][fold.test_idx]
    return u


def lineage_member(level1, feature, members, outer, selection, log=print):
    """A lineage member's u on the outer folds' test rows (the base member's fold selection plus
    the feature) and its coefficient on the feature per fold."""
    numeric = c2_ability.numeric_frame(level1, columns=()).assign(**{LINEAGE: feature})
    u, coefs = np.full(len(level1), np.nan), []
    by_pair = {}
    for fold, pick in zip(outer, selection, strict=True):
        pair = (pick["drift_var_per_day"], pick["scale"])
        if pair not in by_pair:
            by_pair[pair] = numeric.assign(**{c2_ability.RATING: members[pair].to_numpy()})
        data = by_pair[pair]
        model = AbilityModel(
            k=3,
            discounts=(1.0, pick["pooled w2"], pick["pooled w3"]),
            entity_cols=c2_ability.ENTITIES,
            features=[c2_ability.RATING, LINEAGE],
        ).fit(data.iloc[fold.train_idx])
        u[fold.test_idx] = model.predict(data.iloc[fold.test_idx])["u"].to_numpy()
        coefs.append(
            {"fold": fold.test_start.isoformat(), "train rows": len(fold.train_idx)}
            | {"coef": float(model.coef_[LINEAGE])}
            | {"rating coef": float(model.coef_[c2_ability.RATING])}
        )
    log(f"  lineage member: {len(outer)} folds")
    return u, coefs


# --------------------------------------------------------------------------- level 2


def _renormalised(v: np.ndarray, race: np.ndarray) -> np.ndarray:
    top = pd.Series(v).groupby(race).transform("max").to_numpy()
    return v - top - np.log(pd.Series(np.exp(v - top)).groupby(race).transform("sum").to_numpy())


def member_rows(price, level1, u: dict, flags: pd.DataFrame, first) -> tuple[pd.DataFrame, dict]:
    """The price frame's rows with each member's ln p renormalised over the race's priced
    starters and the segment flags; a race with a starter lacking a member is dropped (counted
    from `first`). Index: price frame positions."""
    keys = level1[["race_id", "horse_id"]].assign(**u, **{s: flags[s] for s in SEGMENTS})
    keys = keys.astype({"race_id": object, "horse_id": object})
    left = price.reset_index().astype({"race_id": object, "horse_id": object})
    joined = left.merge(keys, on=["race_id", "horse_id"], how="left", validate="one_to_one")
    has = joined[list(u)].notna().all(axis=1)
    whole = has.groupby(joined["race_id"].to_numpy()).transform("all")
    rows = joined.loc[whole.to_numpy()].set_index("index").rename_axis(None)
    race = rows["race_id"].to_numpy()
    for m in u:
        rows[m] = _renormalised(rows[m].to_numpy(dtype=float), race)
    for s in SEGMENTS:
        rows[s] = rows[s].eq(True).to_numpy(dtype=bool)
    rows["ln_q"] = np.log(rows["q"].to_numpy())
    counts = {
        "level2_races_with_every_member": int(rows["race_id"].nunique()),
        "level2_races_dropped_for_a_missing_member": int(
            joined.loc[
                ~whole.to_numpy() & joined["race_date"].ge(pd.Timestamp(first)), "race_id"
            ].nunique()
        ),
    }
    return rows, counts


def segment_rows(p_a: pd.Series, p_b: pd.Series, scored: Scored, spec: dict) -> list[dict]:
    """`segment_delta_ll` over the scored rows, then the every-race row."""
    table = segment_delta_ll(p_a, p_b, scored.rows, spec).to_dict("records")
    g = gain(_ll(p_a, scored.rows), _ll(p_b, scored.rows)).per_race
    iv = _bounds(g, scored.clusters, spec)
    table.append(
        {"segment": ALL, "races": len(g), "meetings": iv.n_clusters}
        | {"delta_ll": iv.mean, "lower": iv.lo, "upper": iv.hi}
    )
    return table


def _segment_metrics(tables: dict[str, list[dict]], segment: str) -> list:
    metrics = []
    for comparison, table in tables.items():
        row = next(r for r in table if r["segment"] == segment)
        if row["races"]:
            metrics.append(
                _metric(
                    "all_folds",
                    f"delta_ll[{comparison}]",
                    row["delta_ll"],
                    row["races"],
                    row["lower"],
                    row["upper"],
                )
            )
    if not metrics:
        raise ValueError(f"no scored race in segment {segment}")
    return metrics


# --------------------------------------------------------------------------- the pure study


def stage2(member: str) -> MetaSpec:
    return MetaSpec([member], quadratic=False, segment_cols=[], form="stage2", ridge=0.0)


def fold_scheme_of(spec: dict) -> str:
    folds = {"level1": spec["level1"]["folds"], "level2": spec["level2"]["folds"]}
    return "walk_forward:" + json.dumps(folds, sort_keys=True)


def _raise(exc: Exception):
    raise exc


def comparisons_of(member: str) -> list[str]:
    return ["stage2_vs_market_only"] if member == BASE else list(COMPARISONS)


def compute_study(inputs: Inputs, spec: dict, scheme, plan: Plan = FULL, log=print) -> Study:
    """Every variant. Pure: no file or registry I/O."""
    level1, price = inputs.level1, inputs.price
    study = Study(counts=dict(inputs.counts))
    fold_scheme = fold_scheme_of(spec)
    comp = {"model": "models/ability.py", "prior": "models/lineage.py"}
    comp |= {"study": "studies/lineage.py", "sire": "netkeiba"}
    params = {"thin_starts": THIN_STARTS, "band_scheme": BAND_SCHEME}

    starts = quarter_starts(level1, scheme, plan.results_from)
    window, cover = coverage(level1, starts)
    study.tables["coverage"] = cover
    if window is None:
        raise ValueError("no quarter meets the netkeiba coverage rule")
    study.counts |= {"window_from": window.isoformat(), "prior_quarters": len(starts)}
    quarters = c2_ability.level1_quarters(level1["race_date"], scheme)
    quarters = [q for q in quarters if q.test_start >= plan.inner_from]
    outer = [q for q in quarters if q.test_start >= plan.level1_from]
    if quarters[0].test_start != plan.inner_from or outer[0].test_start != plan.level1_from:
        raise ValueError("the level-1 quarters do not start at the plan's dates")
    study.counts |= {"level1_quarters": len(quarters), "level1_outer_folds": len(outer)}

    log(f"rating grid over {len(inputs.results)} starters")
    grid_members, _ = c2_rating.member_grid(inputs.results, level1)
    picks = c2_ability.select_pairs(level1, grid_members, outer, plan.inner_from)
    needed = c2_ability.needed_pairs(outer, picks, quarters)
    members = {pair: grid_members[pair] for pair in needed}
    del grid_members
    log(f"level-1 base grid: pairs {list(needed)}, {len(plan.discount_grid)} discounts")
    grid = c2_ability.level1_grid(
        level1, members, quarters, needed, log=log, models=("pooled",),
        discount_grid=plan.discount_grid,
    )  # fmt: skip
    study.counts |= {"level1_base_fits": grid.fits, "level1_grid_seconds": round(grid.seconds)}
    selection = c2_ability.select_discounts(
        level1, grid, outer, picks, models=("pooled",), discount_grid=plan.discount_grid,
        inner_from=plan.inner_from,
    )  # fmt: skip
    study.tables["selection"] = selection
    u = {BASE: base_members(level1, grid, outer, selection)}
    del grid

    def selection_metrics() -> tuple[list, dict]:
        metrics = []
        for row in selection:
            n = row["inner races"]
            for key in ("drift_var_per_day", "scale", "pooled w2", "pooled w3"):
                metrics.append(_metric(row["fold"], f"selected[{key}]", row[key], n))
            metrics.append(
                _metric(row["fold"], "inner_log_loss[pooled]", row["pooled inner log loss"], n)
            )
        return metrics, {}

    base_params = params | {"discount_grid": spec["level1"]["discount_grid"]}
    study.attempt(
        Variant("E37", "selection:inner_fold", fold_scheme, comp | {"member": BASE}, base_params),
        selection_metrics,
    )

    failed: dict[str, str] = {}
    for key, prior in VARIANTS.items():
        log(f"lineage feature {key}")
        started = time.monotonic()
        vparams = params | {"prior": dict(prior)}
        try:
            feature = _lineage_feature(level1, inputs.pedigree, starts, prior, log=log)
            study.counts[f"feature_seconds[{key}]"] = round(time.monotonic() - started)
            study.counts[f"feature_nonzero[{key}]"] = int((feature != 0).sum())
            u[key], coefs = lineage_member(level1, feature, members, outer, selection, log)
        except Exception as exc:  # the variant's every record fails, never dropped
            failed[key] = f"{type(exc).__name__}: {exc}"
            study.attempt(
                Variant("E37", f"level1:{key}", fold_scheme, comp | {"member": key}, vparams),
                lambda exc=exc: _raise(exc),
            )
            continue
        study.tables.setdefault("lineage_coef", []).extend({"variant": key} | c for c in coefs)
        study.attempt(
            Variant("E37", f"level1:{key}", fold_scheme, comp | {"member": key}, vparams),
            lambda coefs=coefs: (
                [_metric(c["fold"], f"coef[{LINEAGE}]", c["coef"], c["train rows"]) for c in coefs],
                {},
            ),
        )
        log(f"  {key}: {time.monotonic() - started:.0f} s ({c2_ability._progress(started)})")
    del members

    flags = run_segments(level1)
    study.counts |= {f"level1_{s}_runs": int(flags[s].sum()) for s in SEGMENTS}
    folds2 = walk_forward(price["race_date"], scheme=scheme)
    log(f"D4 baseline on {len(folds2)} level-2 folds")
    price = price.assign(p_cal=baseline_oof(price, folds2))
    rows, member_counts = member_rows(price, level1, u, flags, outer[0].test_start)
    study.counts |= member_counts
    rfolds = restrict(folds2, rows.index.to_numpy(), len(price))
    rows = rows.reset_index(drop=True)
    scored_folds = [
        f for f in rfolds if f.test_start >= window and len(f.train_idx) and len(f.test_idx)
    ]
    scored = Scored.of(rows, scored_folds)
    study.counts |= {"scored_folds": len(scored_folds), "scored_races": scored.n_races}
    study.counts |= {
        f"scored_{s}_races": int(scored.rows.loc[scored.rows[s], "race_id"].nunique())
        for s in SEGMENTS
    }

    base = comp | {"score": "winner_log_loss", "baseline": "calibrated_market"}
    meta_params = params | {"ridge": 0.0, "bounds": "none", "quadratic": False, "segments": []}
    p = {}
    for member in u:
        outcome = _try(lambda m=member: _meta_oof(rows, scored_folds, stage2(m), "ln_q"))
        pooled = _try(lambda m=member: _pooled(scored, stage2(m), "ln_q"))
        name = f"stage2:q:{member}"
        prior = {"prior": dict(VARIANTS[member])} if member in VARIANTS else {}
        study.attempt(
            Variant("E37", name, fold_scheme, base | {"member": member}, meta_params | prior),
            lambda o=outcome, pl=pooled, nm=name: _meta_metrics(
                o, pl, scored_folds, scored, spec, nm
            ),
        )
        if outcome.error is None:
            p[member] = outcome.value[0]
    market = _try(lambda: c2_ability._market_only(rows, scored_folds, "ln_q"))
    study.attempt(
        Variant("E37", "stage2:q:market_only", fold_scheme, base | {"member": "none"}, meta_params),
        lambda: _meta_metrics(market, None, scored_folds, scored, spec, "stage2:q:market_only"),
    )

    paired = comp | {"score": "paired_winner_log_loss"}
    r = scored.rows

    def need(*names):
        missing = [m for m in names if m not in p]
        if missing:
            raise RuntimeError(f"the stage-2 meta of {missing} failed")

    def tables_of(key: str) -> dict[str, list[dict]]:
        if key in failed:
            raise RuntimeError(f"the lineage member failed: {failed[key]}")
        need(key, BASE)
        q_only = market.get()[0]
        if key == BASE:
            return {"stage2_vs_market_only": segment_rows(p[BASE], q_only, scored, spec)}
        return {
            "standalone": segment_rows(np.exp(r[key]), np.exp(r[BASE]), scored, spec),
            "stage2_vs_base": segment_rows(p[key], p[BASE], scored, spec),
            "stage2_vs_market_only": segment_rows(p[key], q_only, scored, spec),
        }

    for key in (BASE, *VARIANTS):
        tables = _try(lambda key=key: tables_of(key))
        if tables.error is None:
            for comparison, table in tables.value.items():
                study.tables.setdefault("segments", []).extend(
                    {"member": key, "comparison": comparison} | row for row in table
                )
        prior = {"prior": dict(VARIANTS[key])} if key in VARIANTS else {}
        for segment in (*SEGMENTS, ALL):
            study.attempt(
                Variant(
                    "E37",
                    f"e37:{key}:{segment}",
                    fold_scheme,
                    paired | {"member": key, "segment": segment},
                    params | prior | {"comparisons": comparisons_of(key)},
                ),
                lambda t=tables, s=segment: (_segment_metrics(t.get(), s), {}),
            )
    return study


# --------------------------------------------------------------------------- the run


def planned() -> list[tuple[str, str]]:
    """Every variant a run records, in order (a failed run records each as failed)."""
    names = ["selection:inner_fold", *(f"level1:{k}" for k in VARIANTS)]
    names += [f"stage2:q:{m}" for m in (BASE, *VARIANTS, "market_only")]
    names += [f"e37:{m}:{s}" for m in (BASE, *VARIANTS) for s in (*SEGMENTS, ALL)]
    return [("E37", n) for n in names]


SECTIONS = (
    ("coverage", "The netkeiba coverage of the level-1 starters per quarter"),
    ("selection", "The base member's selection per outer fold (inner-fold winner log loss)"),
    ("lineage_coef", "Each lineage member's coefficients per outer fold"),
    ("segments", "Segment ΔLL (positive: the member beats the comparator; one-sided 95%)"),
    ("delta", "ΔLL against D4's calibrated market (out of fold; meeting bootstrap, one-sided)"),
    ("coefficients", "Stage 2 per fold (unbounded; Wald 95% two-sided and meeting one-sided 95%)"),
    ("pooled", "Stage-2 weight pooled over the scored rows (in-sample)"),
)


def render_report(study: Study, spec: dict, about: dict, record_ids: dict[str, str]) -> str:
    """The full report: aggregate tables only, never a runner or race row."""
    lines = [f"# E37: the lineage prior in the ability member ({COUNTRY}, #{TICKET})", ""]
    lines += [*(f"- {k}: {v}" for k, v in about.items()), ""]
    lines += ["## Decision rules (frozen)", ""]
    lines += [f"- {k}: {v}" for k, v in spec["decision_rules"].items()]
    lines += ["", "## Data", ""]
    lines += [_markdown_table([{"count": k, "n": v} for k, v in study.counts.items()])]
    for key, title in SECTIONS:
        rows = study.tables.get(key)
        if key == "pooled" and rows:  # studies.meta labels the member's weight delta
            rows = [{("alpha" if k == "delta" else k): v for k, v in r.items()} for r in rows]
        lines += ["", f"## {title}", "", _markdown_table(rows) if rows else "_Not computed._"]
    variants = [
        {"E": v.hypothesis_id, "variant": v.name, "status": "failed" if v.error else "scored"}
        | {"record": record_ids.get(v.name, "")[:16]}
        | ({"error": v.error} if v.error else {})
        for v in study.variants
    ]
    lines += ["", "## Variants (one registry record each)", "", _markdown_table(variants), ""]
    return "\n".join(lines)


def artifact_stem(spec_hash: str, builds: dict, trial: bool) -> str:
    """The report's file stem: the spec hash and a digest of the data builds."""
    digest = hashlib.sha256(
        json.dumps(dict(builds), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"lineage-{COUNTRY}-{spec_hash[:12]}-{digest[:12]}" + ("-trial" if trial else "")


def _record(variant: Variant, git_sha: str, builds: dict, spec_hash: str, trial: bool):
    params = dict(variant.params) | {"spec_sha256": spec_hash}
    if trial:
        params["trial"] = True
    if variant.error:
        params["error"] = variant.error
    return RunRecord(
        hypothesis_id=variant.hypothesis_id,
        family_id=TRIAL_FAMILY_ID if trial else FAMILY_ID,
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


def _spec_hash(spec: dict) -> str:
    text = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def run_study(*, trial: bool = False, out_dir: Path | None = None, log=print) -> dict:
    """Run the pre-registered family on Japan's development data; write the report and the
    registry records, then the committed summary."""
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    scheme = load_country_scheme(COUNTRY)
    plan = TRIAL if trial else FULL
    spec = build_spec(scheme, {**bootstrap_settings(), "seed": BOOTSTRAP_SEED}, plan)
    spec_hash = _spec_hash(spec)
    registry = default_registry(COUNTRY)
    if not trial:  # before any outcome is read; a trial's spec may still change
        spec_hash = registry.freeze_family(FAMILY_ID, spec)
    family = TRIAL_FAMILY_ID if trial else FAMILY_ID
    git_sha = _git_sha()
    builds = {**store_builds(), **pedigree_build()}
    stem = artifact_stem(spec_hash, builds, trial)
    started = time.monotonic()
    families = (FAMILY_ID, TRIAL_FAMILY_ID)
    try:
        inputs = load_inputs(plan, log=log)
        study = compute_study(inputs, spec, scheme, plan, log=log)
    except Exception as exc:
        fold_scheme = fold_scheme_of(spec)
        for hypothesis, name in planned():
            failed = Variant(
                hypothesis, name, fold_scheme, {"study": "studies/lineage.py"}, {},
                error=f"{type(exc).__name__}: {exc}",
            )  # fmt: skip
            registry.append(_record(failed, git_sha, builds, spec_hash, trial))
        c2_ability.write_committed_summary(registry, families)
        raise
    log(f"computed ({c2_ability._progress(started)})")

    held = [dataclasses.replace(r, git_sha="") for r in registry.records(family)]
    record_ids, appended = {}, []
    for variant in study.variants:
        record = _record(variant, git_sha, builds, spec_hash, trial)
        if dataclasses.replace(record, git_sha="") in held:  # the same run at another commit
            record_ids[variant.name] = "already held"
            continue
        record_ids[variant.name] = registry.append(record)
        appended.append(variant.name)
    about = {
        "family": f"`{family}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "data": ", ".join(builds.values()),
        "records": f"{len(appended)} appended; {len(record_ids) - len(appended)} already held",
        "trial (records marked params.trial, spec not frozen)": trial,
        "run time": c2_ability._progress(started),
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
    c2_ability.write_committed_summary(registry, families)
    return {
        "family_id": family,
        "spec_hash": spec_hash,
        "report_path": str(report),
        "registry_path": str(registry.path),
        "records_appended": len(appended),
        "failed_variants": {v.name: v.error for v in study.variants if v.error},
    }


def lineage_study_command(
    trial: Annotated[
        bool,
        typer.Option(
            "--trial",
            help=f"The spec's short trial plan; records go to {TRIAL_FAMILY_ID} marked trial; "
            "the spec is not frozen; the report is suffixed -trial.",
        ),
    ] = False,
) -> None:
    """Feed the lineage prior into the ability member (E37) and write the report."""
    typer.echo(json.dumps(run_study(trial=trial, log=typer.echo), indent=2, default=str))
