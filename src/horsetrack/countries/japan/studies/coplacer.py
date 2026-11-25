"""Co-placer Place behind heavy favourites: reconciliation and decision-time test (#83, E15).

08 R11: two research reports measured Place on the runners behind a heavy favourite with
different definitions (composition by co-runner rank, period, field size, favourite cutoff, raw
ROI against return over an odds-matched expectation). This module puts both definitions on one
period and one filter set (step 1), selects ranks 2-3 on the board at the decision time t
(step 2), and prices the stake-back-share mechanism (step 3).

The study frame has one row per starter, in store column names (docs/specs/store.md): race_id,
race_date, horse_no, win_odds (final, decimal incl. stake), finish_pos, dead_heat, is_scratched,
field_size, place_dividend (the PLACE dividend per the 100-yen unit, NaN when it did not pay),
place_is_refund, and meeting_id where the store gives it. A runner's Place return is
place_dividend / unit_stake per unit staked, 0 when it did not pay. Rank is 1 + the runners of its
race at strictly lower win_odds; the favourite is rank 1. Places paid follow `jra.yaml`.

Pure: `common_filter`, `odds_ventile`, `odds_matched_expectation`, `coplacer_table`,
`place_payout_scenarios`, `select_at_t`. Bounds are D7's: meeting (venue x day) clusters, 95%
one-sided, replicates from `config/scoring.yaml`; a frame without `meeting_id` takes the meeting
from the race id by the store's own rule (`<source>:<date>:<venue>`).

The study (`run_study`; CLI `python -m horsetrack.countries.japan.studies.coplacer --run [--out
DIR]`, an `--out` inside the repository refused) freezes its spec in the run registry before any
outcome is read, reads the development rows of the price era (`read_store_table`, holdout days
dropped, then `split_off`; the holdout is never opened), reads the odds logs of non-holdout days,
appends one registry record per variant (failures and data-pending variants included) and writes
its report through `write_report`. Nothing enters Family P here: that is the owner's (08 §5.4).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml, load_source
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day, split_off
from horsetrack.core.eval.registry import (
    FoldMetric,
    RunRecord,
    default_registry,
    summary_path,
    write_summary,
)
from horsetrack.core.eval.reports import write_report
from horsetrack.core.pricing import _EPS, stake_back_share_dividend
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board import snapshot
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table

COUNTRY = "japan"
SOURCE = "jra"  # the store's JRA results (netkeiba) and the JRA odds logs; no JRDB, no JRA-VAN
TICKET = 83
FAMILY_ID = "japan-coplacer-place-e15"
BOOTSTRAP_SEED = 83  # reproducibility only; scoring.yaml signs no seed

FAV_BANDS = ("≤ 1.3", "1.4–1.5", "1.6–2.0", "> 2.0")
FAV_EDGES = (1.3, 1.5, 2.0)  # each band closed on the right
RANK_BANDS = ("2", "3", "4–5", "6–8", "9+")
RANK_EDGES = (2, 3, 5, 8)  # the last rank of each band but 9+
VENTILE_QUANTILES = tuple(k / 20 for k in range(1, 20))
METRICS = ("raw", "odds_matched")


def _place_pool():
    return load_source(COUNTRY, SOURCE).pools["PLACE"]


def _unit() -> float:
    return float(load_source(COUNTRY, SOURCE).unit_stake)


def places_paid(field_size: pd.Series | Iterable) -> np.ndarray:
    """Places the PLACE pool pays for each field size (`jra.yaml`); 0 where it is not sold or
    the field size is missing."""
    pool = _place_pool()
    sizes = pd.Series(field_size).astype("Int64")
    lookup = {int(s): pool.places_for(int(s)) for s in sizes.dropna().unique()}
    return sizes.map(lookup).fillna(0).astype(int).to_numpy()


def place_return(frame: pd.DataFrame) -> pd.Series:
    """Place return per unit staked: place_dividend / unit_stake, 0 when it did not pay."""
    return frame["place_dividend"].astype(float).fillna(0.0) / _unit()


def _rank(frame: pd.DataFrame) -> pd.Series:
    """1 + the runners of the race at strictly lower win_odds."""
    return frame.groupby("race_id")["win_odds"].rank(method="min")


def meeting_of(frame: pd.DataFrame) -> pd.Series:
    """The meeting (venue x day, D7's cluster): `meeting_id` when the frame has it, else the race
    id without its race number (the store's rule, `core.store.build._race_keys`)."""
    if "meeting_id" in frame.columns:
        return frame["meeting_id"].astype(str)
    return frame["race_id"].astype(str).str.rsplit(":", n=1).str[0]


# --------------------------------------------------------------------------- the common filter


def race_flags(runners: pd.DataFrame) -> pd.DataFrame:
    """Per race (indexed by race_id): field_size and why the common filter would drop it."""
    odds = runners["win_odds"].astype(float)
    race = runners["race_id"]
    fav_odds = odds.groupby(race).transform("min")
    at_fav = (odds - fav_odds).abs() <= _EPS
    placed = runners["finish_pos"].astype("Int64").le(3).fillna(False).astype(bool)
    dead_heat = runners["dead_heat"].astype("boolean").fillna(False).astype(bool)
    refund = runners["place_is_refund"].astype("boolean").fillna(False).astype(bool)
    flags = pd.DataFrame(
        {
            "field_size": runners["field_size"].astype("Int64"),
            "fav_count": at_fav.astype(int),
            "dead_heat_top3": dead_heat & placed,
            "refund": refund,
        }
    ).groupby(race.to_numpy(), sort=False)
    out = flags.agg(
        field_size=("field_size", "first"),
        fav_count=("fav_count", "sum"),
        dead_heat_top3=("dead_heat_top3", "any"),
        refund=("refund", "any"),
    )
    out["place_not_sold"] = places_paid(out["field_size"]) == 0
    out["favourite_not_unique"] = out["fav_count"] != 1
    return out.drop(columns="fav_count")


def common_filter(runners: pd.DataFrame, *, min_field: int | None) -> pd.DataFrame:
    """Whole races of `runners`, rows unchanged: Place sold, a unique favourite, no dead heat in
    the top three, no refund, and field_size >= min_field (None: every field)."""
    flags = race_flags(runners)
    keep = ~(
        flags["place_not_sold"]
        | flags["favourite_not_unique"]
        | flags["dead_heat_top3"]
        | flags["refund"]
    )
    if min_field is not None:
        keep &= flags["field_size"].fillna(0).astype(int) >= int(min_field)
    return runners.loc[runners["race_id"].isin(flags.index[keep.to_numpy(dtype=bool)])]


# --------------------------------------------------------------------------- odds matching


def odds_ventile(odds: pd.Series, cut_points: Sequence[float]) -> pd.Series:
    """The number of cut points strictly below each price (bins closed on the right), on the
    input's index; missing for missing odds."""
    values = odds.astype(float).to_numpy()
    k = np.searchsorted(np.asarray(cut_points, dtype=float), values, side="left")
    return pd.Series(
        pd.array(np.where(np.isnan(values), pd.NA, k), dtype="Int64"), index=odds.index
    )


def _cell_codes(frame: pd.DataFrame, cuts: np.ndarray) -> np.ndarray:
    ventile = odds_ventile(frame["win_odds"], cuts).to_numpy(dtype=float, na_value=np.nan)
    paid = places_paid(frame["field_size"])
    return np.where(np.isnan(ventile), -1, ventile * 100 + paid).astype(int)


def odds_matched_expectation(runners: pd.DataFrame, *, reference: pd.DataFrame) -> pd.Series:
    """On `runners`' index: the mean Place return of the `reference` rows in the runner's (odds
    ventile, places paid) cell, the ventiles cut at the reference's win_odds quantiles 0.05 to
    0.95 (numpy, linear). NaN where the reference does not fill the cell."""
    ref = reference.loc[reference["win_odds"].astype(float).notna()]
    if ref.empty:
        raise ValueError("the reference has no priced runner")
    cuts = np.quantile(ref["win_odds"].astype(float).to_numpy(), VENTILE_QUANTILES)
    means = pd.Series(place_return(ref).to_numpy()).groupby(_cell_codes(ref, cuts)).mean()
    means = means.drop(index=-1, errors="ignore")
    codes = pd.Series(_cell_codes(runners, cuts), index=runners.index)
    return codes.map(means).astype(float)


# --------------------------------------------------------------------------- the table


def fav_band(odds: pd.Series) -> np.ndarray:
    k = np.searchsorted(np.asarray(FAV_EDGES) + _EPS, odds.astype(float).to_numpy(), side="left")
    return np.asarray(FAV_BANDS, dtype=object)[k]


def rank_band(rank: pd.Series) -> np.ndarray:
    k = np.searchsorted(np.asarray(RANK_EDGES), rank.astype(float).to_numpy(), side="left")
    return np.asarray(RANK_BANDS, dtype=object)[k]


def bootstrap_settings() -> dict:
    """D7's bootstrap from config/scoring.yaml, with this study's fixed seed."""
    boot = load_country_yaml(COUNTRY, "scoring")["bootstrap"]
    if boot["cluster_unit"] != "meeting" or boot["sided"] != "one-sided":
        raise ValueError(f"this study implements meeting clusters, one-sided; scoring.yaml: {boot}")
    return {
        "cluster_unit": "meeting",
        "sided": "one-sided",
        "level": float(boot["level"]),
        "n_boot": int(boot["n_boot"]),
        "seed": BOOTSTRAP_SEED,
    }


def _starters(frame: pd.DataFrame) -> pd.DataFrame:
    scratched = frame["is_scratched"].astype("boolean").fillna(False).astype(bool)
    return frame.loc[~scratched & frame["win_odds"].astype(float).notna()]


def coplacer_cells(
    runners: pd.DataFrame, *, metric: str, reference: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per co-runner (rank > 1) of the priced starters: fav_band, rank_band, rank, value
    (Place return for 'raw'; minus its odds-matched expectation for 'odds_matched', the
    reference defaulting to the input's priced starters, favourites included) and meeting."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    starters = _starters(runners)
    value = place_return(starters)
    if metric == "odds_matched":
        ref = starters if reference is None else _starters(reference)
        value = value - odds_matched_expectation(starters, reference=ref)
        if value.isna().any():
            raise ValueError(
                f"{int(value.isna().sum())} runner(s) fall in a cell the reference lacks"
            )
    rank = _rank(starters)
    fav = starters.groupby("race_id")["win_odds"].transform("min")
    co = (rank > 1).to_numpy()
    return pd.DataFrame(
        {
            "fav_band": fav_band(fav[co]),
            "rank_band": rank_band(rank[co]),
            "rank": rank[co].astype(int).to_numpy(),
            "value": value[co].to_numpy(dtype=float),
            "meeting": meeting_of(starters).to_numpy()[co],
        }
    )


def summarise(cells: pd.DataFrame, bootstrap: dict) -> pd.DataFrame:
    """Per (fav_band, rank_band) group of `cells`, in first-label order: n, the mean value as
    `return`, and the meeting-clustered one-sided bounds lo and hi with the meetings resampled."""
    rows = []
    for (fb, rb), part in cells.groupby(["fav_band", "rank_band"], sort=False):
        idx = pd.RangeIndex(len(part))
        iv = paired_bootstrap(
            pd.Series(part["value"].to_numpy(), index=idx),
            pd.Series(part["meeting"].to_numpy(), index=idx),
            n_boot=bootstrap["n_boot"],
            level=bootstrap["level"],
            seed=bootstrap["seed"],
        )
        rows.append(
            {"fav_band": fb, "rank_band": rb, "n": len(part), "return": iv.mean}
            | {"lo": iv.lo, "hi": iv.hi, "meetings": iv.n_clusters}
        )
    columns = ["fav_band", "rank_band", "n", "return", "lo", "hi", "meetings"]
    return pd.DataFrame(rows, columns=columns)


def _ordered(cells: pd.DataFrame, fav_order: Sequence[str], rank_order: Sequence[str]):
    key = cells["fav_band"].map({b: i for i, b in enumerate(fav_order)}) * 100
    key = key + cells["rank_band"].map({b: i for i, b in enumerate(rank_order)})
    return cells.iloc[np.argsort(key.to_numpy(), kind="stable")]


def coplacer_table(
    runners: pd.DataFrame,
    *,
    metric: str,
    reference: pd.DataFrame | None = None,
    bootstrap: dict | None = None,
) -> pd.DataFrame:
    """Co-runners of an already filtered frame by favourite band x co-runner rank band: one row
    per cell holding runners (fav_band, rank_band, n, return, lo, hi, meetings). metric 'raw': the
    mean Place return; 'odds_matched': the mean of Place return minus its odds-matched
    expectation (`reference` defaults to the input frame). Bounds are D7's (`bootstrap_settings`
    unless `bootstrap` is given)."""
    cells = coplacer_cells(runners, metric=metric, reference=reference)
    boot = bootstrap_settings() if bootstrap is None else bootstrap
    return summarise(_ordered(cells, FAV_BANDS, RANK_BANDS), boot)


# --------------------------------------------------------------------------- the mechanism


def place_payout_scenarios(
    place_money: dict[int, float],
    pool_total: float,
    rate: float,
    *,
    runner: int,
    favourite: int,
) -> dict[str, float]:
    """The dividend (yen per unit) of `runner` placing under JRA's stake-back-share rule, with the
    favourite among its co-placers ('fav_places') and unplaced ('fav_misses'). Shares are money /
    pool_total; places paid come from the number of starters. The remaining co-placers are the
    most-backed other runners (the low end of the runner's displayed Place range)."""
    if runner == favourite:
        raise ValueError("the runner is the favourite")
    for horse in (runner, favourite):
        if horse not in place_money:
            raise ValueError(f"horse {horse} has no Place money")
    if not pool_total > 0:
        raise ValueError(f"pool_total must be positive, got {pool_total!r}")
    pool = _place_pool()
    paid = pool.places_for(len(place_money))
    if paid < 2:
        raise ValueError(f"Place is not sold with {len(place_money)} starters")
    share = {h: float(m) / pool_total for h, m in place_money.items()}
    others = sorted(
        (h for h in share if h not in (runner, favourite)), key=lambda h: (-share[h], h)
    )

    def dividend(co: list[int]) -> float:
        return stake_back_share_dividend(
            share[runner],
            [share[h] for h in co],
            rate=rate,
            unit=_unit(),
            dividend_step=pool.dividend_step,
            min_dividend=pool.min_dividend,
        )

    return {
        "fav_places": dividend([favourite, *others[: paid - 2]]),
        "fav_misses": dividend(others[: paid - 1]),
    }


# --------------------------------------------------------------------------- decision time


def ranked_at_t(
    rows: Iterable[dict[str, Any]], race_id: str, at_utc: datetime, *, post_utc: datetime
) -> list[tuple[int, float, int]]:
    """(horse_no, WIN odds, rank) on `board_at(rows, race_id, at_utc)`, in rank order (ties by
    horse number); scratched runners are not on the board. Empty when there is no board."""
    board = snapshot.board_at(rows, race_id, at_utc, post_utc=post_utc)
    if board is None:
        return []
    prices = sorted(board.win.items(), key=lambda item: (item[1], item[0]))
    return [(h, o, 1 + sum(1 for p in board.win.values() if p < o)) for h, o in prices]


def select_at_t(
    rows: Iterable[dict[str, Any]],
    race_id: str,
    *,
    post_utc: datetime,
    minutes_before: float | None = None,
    ranks: Sequence[int] = (2, 3),
) -> list[int]:
    """Horse numbers whose WIN-odds rank on the board at t = post_utc - minutes_before is in
    `ranks`, in rank order; `minutes_before` defaults to the owner's primary decision time (D2,
    `config/decision.yaml`). No board by t gives []."""
    if minutes_before is None:
        minutes_before = snapshot.load_decision_grid().primary
    at_utc = post_utc - timedelta(minutes=float(minutes_before))
    wanted = set(ranks)
    return [
        h for h, _, rank in ranked_at_t(rows, race_id, at_utc, post_utc=post_utc) if rank in wanted
    ]


# =========================================================================== the study

FIELD_SETS = {"all_fields": None, "8_plus": 8}  # 06's slice (every field) and 04's (8+ runners)
POOLED_FAV = {"≤ 1.5": ("≤ 1.3", "1.4–1.5")}
POOLED_RANK = {"2–3": ("2", "3"), "all others": RANK_BANDS}
STEP2_FAV_MAX = 1.5  # 08 §4 E15 and P2: "behind a ≤ 1.5 favourite"
STEP2_MIN_MEETINGS = 2  # a meeting-clustered bound needs two meetings to resample
RUNNER_COLUMNS = ["race_id", "race_date", "meeting_id", "horse_no", "win_odds", "finish_pos"]
RUNNER_COLUMNS += ["dead_heat", "is_scratched"]
RACE_COLUMNS = ["race_id", "race_date", "field_size", "status"]
DIVIDEND_COLUMNS = ["race_id", "combination", "dividend", "is_refund"]
DECISIONS = {
    "D2": "decision grid T-60/30/16/5/2/1, T-2 primary (config/decision.yaml); step 2 "
    "selects at T-2",
    "D5": "Place's baseline is its own final prices; here the odds-matched expectation of the "
    "final Win price is 06's comparator, measured, not adopted",
    "D7": "meeting (venue x day) clusters, 95% one-sided (config/scoring.yaml)",
    "D10": "races before 2014-06-07 excluded wherever the price enters: the one period is the "
    "price era",
    "D11-D15": "Japan holdout A2 from 2026-11-28, fully locked: no holdout row or log is read",
    "D21": "no licensed source (JRDB, JRA-VAN) is read",
    "08 §5.4 Decision": "entry into Family P is the owner's; nothing enters it from this family",
}


def build_spec(bootstrap: dict, grid: snapshot.DecisionGrid) -> dict:
    """The family's pre-registered spec, frozen before any outcome is read."""
    pool = _place_pool()
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypotheses": {
            "E15_step1": "on one period and one filter set, both definitions in one table: raw "
            "Place return (04) and return minus odds-matched expectation (06), by favourite band x "
            "co-runner rank band, for all fields and 8+ runners. 08's prediction: the gain sits in "
            "ranks 2-3 on the odds-matched metric and vanishes in an all-others raw average",
            "E15_step2": "ranks 2-3 selected on the board at t behind a favourite at or under "
            f"{STEP2_FAV_MAX} at t: Place return minus odds-matched expectation above 0 "
            "(one-sided lower bound); secondary: absolute Place return",
            "E15_step3": "with logged Place pool totals and per-runner Place money, the "
            "stake-back-share dividend of ranks 2-3 under 'favourite places' vs 'favourite misses'",
        },
        "period": f"development race days from {PRICE_ERA_START.isoformat()} (D10) to the "
        "store's last day",
        "common_filter": "Place sold (field_size >= 5); a unique favourite (lowest final "
        "win_odds); no dead heat among finishers 1-3; no refunded Place ticket; min_field "
        "none or 8",
        "rows": "starters (not scratched) with a final Win price; races with an unpriced starter "
        "are left out and counted",
        "fav_bands": {"labels": list(FAV_BANDS), "right_closed_edges": list(FAV_EDGES)},
        "rank_bands": {"labels": list(RANK_BANDS), "last_rank_of_each": list(RANK_EDGES)},
        "pooled": {"fav": POOLED_FAV, "rank": {k: list(v) for k, v in POOLED_RANK.items()}},
        "rank": "1 + the starters of the race at strictly lower final win_odds",
        "place_return": f"PLACE dividend / {_unit():g} (unit stake), 0 when it did not pay",
        "odds_matched": "Place return minus the mean Place return of the reference's (final "
        "win_odds ventile, places paid) cell; ventile cut points at the reference's win_odds "
        "quantiles 0.05..0.95 (numpy linear); the reference is the same filtered frame, "
        "favourites included",
        "places_paid": [r.model_dump() for r in pool.places_paid],
        "step2": {
            "decision_time_min": grid.primary,
            "board": "countries.japan.board.snapshot.board_at on the odds log, rows available at "
            "or before t = post - decision_time_min",
            "post_times": "snapshot.post_times_for (the races.json sidecar, else HKTrack's card)",
            "qualifying_race": f"five or more priced starters at t; a unique favourite at t with "
            f"WIN odds at or under {STEP2_FAV_MAX}",
            "selection": "WIN-odds ranks 2 and 3 at t",
            "primary": "per selected runner, Place return at the final dividend minus the "
            "odds-matched expectation of its final win_odds and field_size (reference: the "
            "all-fields step-1 frame); mean with D7 bounds; one-sided: the claim rests on lo > 0",
            "secondary": "the absolute Place return of the same runners, with D7 bounds",
            "outcome_filter": "none: no race is dropped for its outcome",
            "scored_when": f"selections with a result in the store span {STEP2_MIN_MEETINGS}+ "
            "meetings; otherwise the result is 'data pending'",
            "evidence_standard": "the owner's (D16-D18); this family reports, it does not decide",
        },
        "step3": {
            "needs": "per race, the final PLACE pool total and the Place money on each starter",
            "dividend": "core.pricing.stake_back_share_dividend at rate 1 - PLACE takeout, unit "
            f"{_unit():g}, step {pool.dividend_step:g}, floor {pool.min_dividend:g}",
            "co_placers": "the favourite plus the most-backed others ('fav_places'), or the "
            "most-backed others ('fav_misses')",
            "otherwise": "data pending",
        },
        "bootstrap": bootstrap,
        "decisions": DECISIONS,
        "family_p": "none: entry is the owner's decision (08 §5.4 Decision)",
        "sources": [SOURCE],
    }


# --------------------------------------------------------------------------- reading


def read_development() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Development runners, races and PLACE dividends of the price era. Runners and races are
    column- and date-scoped store reads with holdout days dropped, then `split_off`'s development
    rows; the dividends table has no date column, so its PLACE rows are read column-scoped and
    kept only for the development races."""
    frames = []
    for name, columns in (("runners", RUNNER_COLUMNS), ("races", RACE_COLUMNS)):
        table = read_store_table(
            name, columns=columns, start=PRICE_ERA_START, end=date.max, touches_price=True
        )
        development, _ = split_off(table.to_pandas(date_as_object=False), "race_date", COUNTRY)
        frames.append(development)
    runners, races = frames
    dataset = ds.dataset(paths.processed_dir(SOURCE) / "dividends.parquet", format="parquet")
    place = dataset.to_table(
        columns=DIVIDEND_COLUMNS,
        filter=(ds.field("pool") == pa.scalar("PLACE"))
        & ds.field("race_id").isin(pa.array(races["race_id"].unique())),
    ).to_pandas()
    return runners, races, place


def development_frame(
    runners: pd.DataFrame, races: pd.DataFrame, place: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """The study frame (one row per priced starter of a run race with no unpriced starter) and
    the counts each rule dropped."""
    run = races.loc[races["status"].eq("run"), ["race_id", "field_size"]]
    known = runners["race_id"].isin(run["race_id"])
    kept = runners.loc[known]
    scratched = kept["is_scratched"].astype("boolean").fillna(False).astype(bool)
    starters = kept.loc[~scratched]
    unpriced = starters["win_odds"].astype(float).isna()
    bad_races = set(starters.loc[unpriced, "race_id"])
    starters = starters.loc[~starters["race_id"].isin(bad_races)]

    dup = place.duplicated(["race_id", "combination"], keep=False)
    if dup.any():
        raise ValueError(f"{int(dup.sum())} PLACE dividend rows repeat a (race, runner)")
    place = place.rename(columns={"dividend": "place_dividend", "is_refund": "place_is_refund"})
    frame = (
        starters.assign(combination=starters["horse_no"].astype("Int64").astype(str))
        .merge(run, on="race_id", how="left", validate="many_to_one")
        .merge(place, on=["race_id", "combination"], how="left", validate="many_to_one")
        .drop(columns="combination")
    )
    frame["place_is_refund"] = frame["place_is_refund"].astype("boolean").fillna(False)
    counts = {
        "races_read": len(races),
        "races_not_run": int((~races["status"].eq("run")).sum()),
        "runners_read": len(runners),
        "runners_without_a_run_race": int((~known).sum()),
        "scratched_runners": int(scratched.sum()),
        "races_with_an_unpriced_starter": len(bad_races),
        "place_dividend_rows": len(place),
        "races": int(frame["race_id"].nunique()),
        "starters": len(frame),
        "meetings": int(frame["meeting_id"].nunique()),
        "first_race_day": str(frame["race_date"].min().date()),
        "last_race_day": str(frame["race_date"].max().date()),
    }
    return frame, counts


@dataclass
class LoggedDay:
    day: date
    venue: str
    path: Path


def logged_days(root: Path | None = None) -> tuple[list[LoggedDay], list[str]]:
    """The JRA odds logs on disk (`odds_snapshots/jra/<date>/<VENUE>.jsonl.gz`); holdout days
    (#71's `is_holdout_day`, D11/D14: outcomes and prices locked) are left unread and listed."""
    root = paths.data_dir() / "odds_snapshots" / SOURCE if root is None else root
    found, skipped = [], []
    for folder in sorted(p for p in root.glob("*") if p.is_dir()):
        try:
            day = date.fromisoformat(folder.name)
        except ValueError:
            continue
        if is_holdout_day(COUNTRY, day):
            skipped.append(folder.name)
            continue
        for log in sorted(folder.glob("*.jsonl.gz")):
            found.append(LoggedDay(day, log.name.split(".")[0], log))
    return found, skipped


def read_log(path: Path) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """WIN rows per race (the board's selection pool), streamed one capture at a time, and the
    counts step 3 needs: races with a PLACE pool total, PLACE runner rows carrying money."""
    by_race: dict[str, list[dict]] = {}
    totals: set[str] = set()
    runner_money = 0
    for member in odds_log.iter_members(path):
        for row in member:
            pool, combination = row.get("pool"), row.get("combination")
            if pool == "PLACE":
                if combination == odds_log.POOL_TOTAL:
                    if row.get("investment") is not None:
                        totals.add(row.get("race_id"))
                elif row.get("investment") is not None:
                    runner_money += 1
            elif pool == "WIN":
                by_race.setdefault(row.get("race_id"), []).append(row)
    return by_race, {"races_with_place_total": len(totals), "place_runner_money_rows": runner_money}


# --------------------------------------------------------------------------- the pure study


@dataclass(frozen=True)
class Variant:
    hypothesis_id: str
    name: str
    fold_scheme: str
    components: dict[str, str]
    params: dict
    metrics: tuple[FoldMetric, ...] = ()
    error: str | None = None


@dataclass
class Study:
    counts: dict[str, Any]
    tables: dict[str, list[dict]] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    variants: list[Variant] = field(default_factory=list)

    def attempt(
        self, variant: Variant, compute: Callable[[], tuple[list, dict, str | None]]
    ) -> None:
        """Run one variant: its metrics and table rows; or, when `compute` names a reason (data
        pending), its metrics with that reason as a failed variant; or its error."""
        try:
            metrics, tables, pending = compute()
        except Exception as exc:  # every failure is a registry record, never dropped
            self.variants.append(dataclasses.replace(variant, error=f"{type(exc).__name__}: {exc}"))
            return
        self.variants.append(dataclasses.replace(variant, metrics=tuple(metrics), error=pending))
        if pending is not None:
            self.notes[variant.name] = pending
        for key, rows in tables.items():
            self.tables.setdefault(key, []).extend(rows)


def _metric(fold: str, metric: str, value: float, n: int, lo=None, hi=None) -> FoldMetric:
    value, lo, hi = (None if v is None else float(v) for v in (value, lo, hi))
    if not all(np.isfinite(v) for v in (value, lo, hi) if v is not None):
        raise ValueError(f"{metric} on {fold} is not finite: {value}, [{lo}, {hi}]")
    return FoldMetric(str(fold), metric, value, lo, hi, int(n))


def reconciliation(frame: pd.DataFrame, metric: str, boot: dict) -> pd.DataFrame:
    """coplacer_table's cells plus the pooled rows R11 compares: favourite at or under 1.5, and
    ranks 2-3 and all others, each with D7 bounds."""
    cells = coplacer_cells(frame, metric=metric)
    fav_groups = {b: (b,) for b in FAV_BANDS} | POOLED_FAV
    rank_groups = {b: (b,) for b in RANK_BANDS} | POOLED_RANK
    out = []
    for rank_label, rank_members in rank_groups.items():
        in_rank = cells["rank_band"].isin(rank_members)
        for fav_label, fav_members in fav_groups.items():
            part = cells.loc[in_rank & cells["fav_band"].isin(fav_members)]
            if len(part):
                out.append(summarise(part.assign(fav_band=fav_label, rank_band=rank_label), boot))
    return pd.concat(out, ignore_index=True)


def _step1(frame: pd.DataFrame, fields: str, metric: str, boot: dict):
    table = reconciliation(frame, metric, boot).to_dict("records")
    metrics = [
        _metric(
            "development",
            f"{metric}[fav {r['fav_band']}|rank {r['rank_band']}]",
            r["return"],
            r["n"],
            r["lo"],
            r["hi"],
        )
        for r in table
    ]
    rows = [
        {"fields": fields, "metric": metric, "fav band": r["fav_band"]}
        | {"rank band": r["rank_band"], "n": int(r["n"]), "meetings": int(r["meetings"])}
        | {"return": r["return"], "lo": r["lo"], "hi": r["hi"]}
        for r in table
    ]
    return metrics, {"step1": rows}, None


def step2_selections(
    logs: dict[tuple[date, str], dict[str, list[dict]]],
    post_times: dict[tuple[date, str], dict[int, datetime]],
    minutes_before: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Ranks 2-3 at t behind a favourite at or under STEP2_FAV_MAX at t, from the logged WIN rows
    of each (day, venue); one row per selection (race_id, horse_no, rank_at_t, odds_at_t)."""
    counts = {"logged_races": 0, "races_without_a_post_time": 0, "races_without_a_board_at_t": 0}
    counts |= {"races_place_not_sold_at_t": 0, "races_fav_not_unique_at_t": 0}
    counts |= {"races_fav_above_cutoff_at_t": 0, "qualifying_races": 0}
    rows = []
    for key, races in logs.items():
        posts = post_times.get(key, {})
        for race_id, race_rows in sorted(races.items()):
            counts["logged_races"] += 1
            post = posts.get(int(race_id.rsplit(":", 1)[1]))
            if post is None:
                counts["races_without_a_post_time"] += 1
                continue
            at_utc = post - timedelta(minutes=float(minutes_before))
            ranked = ranked_at_t(race_rows, race_id, at_utc, post_utc=post)
            if not ranked:
                counts["races_without_a_board_at_t"] += 1
            elif _place_pool().places_for(len(ranked)) == 0:
                counts["races_place_not_sold_at_t"] += 1
            elif sum(1 for _, _, r in ranked if r == 1) != 1:
                counts["races_fav_not_unique_at_t"] += 1
            elif ranked[0][1] > STEP2_FAV_MAX + _EPS:
                counts["races_fav_above_cutoff_at_t"] += 1
            else:
                counts["qualifying_races"] += 1
                rows += [
                    {"race_id": race_id, "horse_no": h, "rank_at_t": r, "odds_at_t": o}
                    for h, o, r in ranked
                    if r in (2, 3)
                ]
    columns = ["race_id", "horse_no", "rank_at_t", "odds_at_t"]
    return pd.DataFrame(rows, columns=columns), counts


def _step2(
    selections: pd.DataFrame, frame: pd.DataFrame, reference: pd.DataFrame, boot: dict
) -> tuple[list, dict, str | None]:
    """The primary and secondary metric on the selections with a result in the study frame;
    the data-pending reason when they span fewer than STEP2_MIN_MEETINGS meetings."""
    columns = ["race_id", "horse_no", "meeting_id", "win_odds", "field_size", "place_dividend"]
    results = frame[columns].astype({"horse_no": int})
    joined = selections.astype({"horse_no": int}).merge(
        results, on=["race_id", "horse_no"], how="inner"
    )
    metrics = [
        _metric("logged_days", "selections", len(selections), len(selections)),
        _metric("logged_days", "selections_with_a_result", len(joined), len(joined)),
    ]
    meetings = int(joined["meeting_id"].nunique())
    if meetings < STEP2_MIN_MEETINGS:
        reason = (
            f"data pending: {len(selections)} selection(s) at t, {len(joined)} with a result in "
            f"the store, over {meetings} meeting(s); {STEP2_MIN_MEETINGS}+ meetings needed"
        )
        return metrics, {}, reason
    ret = place_return(joined)
    gap = ret - odds_matched_expectation(joined, reference=reference)
    rows = []
    for name, values in (("return_minus_odds_matched", gap), ("place_return", ret)):
        values = values.dropna()
        idx = pd.RangeIndex(len(values))
        iv = paired_bootstrap(
            pd.Series(values.to_numpy(), index=idx),
            pd.Series(joined.loc[values.index, "meeting_id"].to_numpy(), index=idx),
            n_boot=boot["n_boot"],
            level=boot["level"],
            seed=boot["seed"],
        )
        metrics.append(_metric("logged_days", name, iv.mean, len(values), iv.lo, iv.hi))
        rows.append(
            {"metric": name, "n": len(values), "meetings": iv.n_clusters}
            | {"mean": iv.mean, "lo": iv.lo, "hi": iv.hi}
        )
    return metrics, {"step2": rows}, None


def step3_mechanism(found: dict[str, int]) -> tuple[list, dict, str | None]:
    """Step 3 needs the final PLACE pool total and each starter's Place money; the odds log
    (docs/specs/odds-log.md) carries money on pool-total rows only."""
    metrics = [_metric("logged_days", k, v, v) for k, v in found.items()]
    if found["races_with_place_total"] == 0 or found["place_runner_money_rows"] == 0:
        reason = (
            f"data pending: Place pool totals (#4) are logged for {found['races_with_place_total']}"
            " race(s), per-runner Place money for none (the odds log carries money on pool-total"
            " rows only)"
        )
        return metrics, {}, reason
    raise NotImplementedError(
        "per-runner Place money is in the log, but no row format for it is specified "
        "(docs/specs/odds-log.md): specify it before step 3 reads it"
    )


def compute_reconciliation(frame: pd.DataFrame, spec: dict, study: Study) -> dict:
    """Step 1 on the study frame: the common filter per field set, then both metrics."""
    boot = spec["bootstrap"]
    flags = race_flags(frame)
    study.counts |= {
        f"races_dropped_{name}": int(flags[name].sum())
        for name in ("place_not_sold", "favourite_not_unique", "dead_heat_top3", "refund")
    }
    filtered = {}
    for fields, min_field in FIELD_SETS.items():
        part = common_filter(frame, min_field=min_field)
        filtered[fields] = part
        study.counts[f"{fields}_races"] = int(part["race_id"].nunique())
        study.counts[f"{fields}_starters"] = len(part)
        for metric in METRICS:
            variant = Variant(
                "E15",
                f"step1:{fields}:{metric}",
                "in_sample:development",
                {"step": "reconciliation", "metric": metric, "filter": "common"},
                {"min_field": min_field},
            )
            study.attempt(variant, lambda p=part, f=fields, m=metric: _step1(p, f, m, boot))
    return filtered


# --------------------------------------------------------------------------- report and runner

SECTIONS = (
    (
        "step1",
        "Step 1: the reconciliation table (development days, common filter)",
        "return is the mean Place return per unit staked for 'raw' (ROI = return - 1) and the mean"
        " of Place return minus its odds-matched expectation for 'odds_matched'. lo and hi are the"
        " (1 - level) and level quantiles of the bootstrap over meetings (D7: one-sided). Pooled"
        " rows: fav '≤ 1.5' joins the first two bands; rank '2–3' and 'all others'.",
    ),
    ("step2", "Step 2: decision-time selection (ranks 2-3 at t)", ""),
)


def _cell(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return "" if value is None else str(value).replace("|", "\\|")


def _markdown_table(rows: list[dict]) -> str:
    head = list(dict.fromkeys(key for row in rows for key in row))
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    lines += ["| " + " | ".join(_cell(row.get(k)) for k in head) + " |" for row in rows]
    return "\n".join(lines)


def render_report(study: Study, spec: dict, about: dict) -> str:
    """The report: aggregates only, never a runner or race row."""
    lines = [
        f"# Co-placer Place behind heavy favourites ({COUNTRY}, #{TICKET}, E15)",
        "",
        *(f"- {key}: {value}" for key, value in about.items()),
        "- development rows only: holdout days are never read and the holdout is not opened",
        "- Family P: nothing enters it from this family; entry is the owner's (08 §5.4 Decision)",
        "",
        "## Owner decisions used",
        "",
        *(f"- {key}: {value}" for key, value in spec["decisions"].items()),
        "",
        "## Data",
        "",
        _markdown_table([{"count": k, "n": v} for k, v in study.counts.items()]),
    ]
    for key, title, note in SECTIONS:
        rows = study.tables.get(key)
        lines += ["", f"## {title}", ""]
        lines += [note, ""] if note else []
        lines.append(_markdown_table(rows) if rows else "_Not computed: see the notes below._")
    lines += ["", "## Notes", "", *(f"- {k}: {v}" for k, v in study.notes.items())]
    variants = [
        {"E": v.hypothesis_id, "variant": v.name, "status": "failed" if v.error else "scored"}
        | ({"error": v.error} if v.error else {})
        for v in study.variants
    ]
    lines += ["", "## Variants (one registry record each)", "", _markdown_table(variants), ""]
    return "\n".join(lines)


def _sha_prefix(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def store_builds() -> dict[str, str]:
    return {
        f"{SOURCE}_{name}": f"processed/{SOURCE}/{name}.parquet@"
        + _sha_prefix(paths.processed_dir(SOURCE) / f"{name}.parquet")
        for name in ("runners", "races", "dividends")
    }


def _record(variant: Variant, git_sha: str, builds: dict[str, str]) -> RunRecord:
    return RunRecord(
        hypothesis_id=variant.hypothesis_id,
        family_id=FAMILY_ID,
        variant=variant.name,
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=builds,
        sources=(SOURCE,),
        fold_scheme=variant.fold_scheme,
        components=variant.components,
        params=variant.params,
        metrics=variant.metrics,
        status="failed" if variant.error else "scored",
    )


def run_study(out_dir: Path | None = None) -> dict:
    """Freeze the spec, run steps 1-3 on Japan's development data and logs, append the registry
    records, rewrite the committed registry summary and write the report outside the tree."""
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    grid = snapshot.load_decision_grid()
    spec = build_spec(bootstrap_settings(), grid)
    git_sha = _git_sha()
    registry = default_registry(COUNTRY)
    spec_hash = registry.freeze_family(FAMILY_ID, spec)  # before any outcome is read

    builds = store_builds()
    runners, races, place = read_development()
    frame, counts = development_frame(runners, races, place)
    del runners, races, place
    study = Study(counts=counts)
    filtered = compute_reconciliation(frame, spec, study)

    days, skipped = logged_days()
    logs, post_times, log_builds = {}, {}, {}
    found = {"races_with_place_total": 0, "place_runner_money_rows": 0}
    for logged in days:
        key = (logged.day, logged.venue)
        logs[key], counted = read_log(logged.path)
        post_times[key] = snapshot.post_times_for(logged.day, logged.venue)
        found = {k: found[k] + counted[k] for k in found}
        name = f"{logged.day.isoformat()}/{logged.path.name}"
        log_builds[f"{SOURCE}_odds_log_{logged.day.isoformat()}_{logged.venue}"] = (
            f"odds_snapshots/{SOURCE}/{name}@{_sha_prefix(logged.path)}"
        )
    study.counts |= {
        "logged_days": len({d.day for d in days}),
        "logged_meetings": len(days),
        "log_days_not_read_holdout": len(skipped),
    }
    selections, step2_counts = step2_selections(logs, post_times, grid.primary)
    del logs
    study.counts |= step2_counts

    def step2() -> tuple[list, dict, str | None]:
        metrics, tables, reason = _step2(
            selections, frame, filtered["all_fields"], spec["bootstrap"]
        )
        metrics += [_metric("logged_days", k, v, v) for k, v in step2_counts.items()]
        return metrics, tables, reason

    variant = Variant(
        "E15",
        f"step2:decision_time:T-{grid.primary}:ranks_2_3",
        "prospective:logged_days",
        {"step": "decision_time", "selection": "board_at_t", "metric": "return_minus_odds_matched"},
        {"minutes_before": grid.primary, "ranks": [2, 3], "fav_max_at_t": STEP2_FAV_MAX},
    )
    study.attempt(variant, step2)
    variant = Variant(
        "E15",
        "step3:mechanism:stake_back_share",
        "prospective:logged_days",
        {"step": "mechanism", "dividend": "stake_back_share", "co_placers": "most_backed"},
        {"scenarios": ["fav_places", "fav_misses"], "ranks": [2, 3]},
    )
    study.attempt(variant, lambda: step3_mechanism(found))

    existing = registry.records(FAMILY_ID)
    records = [
        _record(v, git_sha, builds if v.name.startswith("step1") else builds | log_builds)
        for v in study.variants
    ]
    appended = [registry.append(r) for r in records if r not in existing]
    write_summary(registry, summary_path(COUNTRY))
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "store and logs": ", ".join((builds | log_builds).values()),
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = write_report(
        out_dir / f"coplacer-{COUNTRY}-{spec_hash[:12]}.md",
        render_report(study, spec, about),
        sources=(SOURCE,),
    )
    return {
        "family_id": FAMILY_ID,
        "spec_hash": spec_hash,
        "report_path": str(report),
        "registry_path": str(registry.path),
        "counts": study.counts,
        "notes": study.notes,
        "records_appended": len(appended),
        "records_already_present": len(records) - len(appended),
        "failed_variants": [v.name for v in study.variants if v.error],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m horsetrack.countries.japan.studies.coplacer")
    parser.add_argument("--run", action="store_true", required=True, help="run the study")
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    args = parser.parse_args(argv)
    print(json.dumps(run_study(args.out), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
