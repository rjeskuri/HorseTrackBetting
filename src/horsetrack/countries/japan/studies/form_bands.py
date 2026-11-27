"""#218: does the meld's negative weight on form hold within T-2 odds bands and favourite ranks?

The family the owner approved on #218 (2026-11-25, bands and ranks read from the T-2 board, the
JRA-VAN time series as their source; reframed 2026-11-27). Given the v0 calibrated-market Win
probability p_cal and #129's out-of-fold handicap probability h, the meld is

    P(i wins) ∝ exp(a ln p_cal_i + sum_g b_g 1[i in g] ln h_i)     within the race,

fitted by unpenalised conditional-logit maximum likelihood (`models.condlogit`). The groups come
from each race's Win board at T-2 (D2's primary decision time): five odds bands, [1, 3), [3, 7),
[7, 15), [15, 50), [50, inf), and four favourite-rank groups, 1, 2-3, 4-6, 7+, ranked over every
runner priced on the T-2 board (a tie shares the better rank; a runner scratched after T-2 still
counts). Twelve variants: P0, one b over every runner (#129's V2, the reference); B1-B5 and R1-R4,
one b for one group and none elsewhere; BA and RA, all five band or four rank coefficients
together.

Walk-forward (#70): fold k is fitted on the rows of folds < k up to folds[k].train_end and scores
fold k's races; the earliest fold only trains. A race is scored as D6 (#69): winner log loss, a
dead heat's winners at 1/k, scratched runners removed and p_cal renormalised over the starters; a
race with no winner is dropped and counted, and so is a race with a starter off the T-2 board.
The gain over v0 is ll_v0 - ll_variant in nats per race with D7's meeting bootstrap (one-sided
95% lower bound, two-sided 95% interval). The family claim is Hansen's SPA over the 11 non-reference
variants against v0, then Romano-Wolf StepM at 5% family-wise, on the meetings resampled. The
family test reads this run's records only (`_RunView`): a rerun in which a variant fails tests the
variants this run scored, and the history keeps every record.

Each gain is the gain of the registered meld over v0: b_g on a masked ln h also acts as a group
intercept, and every variant refits the market weight a (`READING`). The frozen spec defines no
favourite-vs-longshot comparison rule, so the report answers the primary question "inconclusive"
and reads each group on its own (`PRIMARY`). v0's file leaves out first-place dead heats, races
without a priced winner and starters without a final price; `read_rows` refits v0's fold fits
from the store (checked to reproduce the file) to score first-place dead heats, checks every v0
row against the store, counts each race it leaves out once by reason and never reads a missing
v0 row as a scratch. Each run writes its own private report.

Families: v1 (`FAMILY_ID_V1`, `build_spec` / `data_description`) is the owner's pre-registration
as frozen and first run; v2 (`FAMILY_ID`, `build_spec_v2` / `data_description_v2`) is the same
design with the corrected eligibility (reviews of PR #473), the family the study records under.

T-2 bands come from JRA-VAN, a licensed source (D21): every record carries the source `jravan`,
the committed registry summary withholds its figures, and the report goes through `write_report`
under the data directory. Nothing is adopted here; adoption is by rank (owner, 2026-11-27).

CLI (the real run, on this machine's data directory):
`uv run python -m horsetrack.countries.japan.studies.form_bands [--out DIR]`.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import Fold, FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day
from horsetrack.core.eval.registry import FoldMetric, Registry, RunRecord, default_registry
from horsetrack.core.eval.reports import write_report
from horsetrack.core.eval.snooping import spa, stepm
from horsetrack.core.pricing import clean_odds
from horsetrack.countries.japan.handicap.factors import handicap_dir
from horsetrack.countries.japan.models.condlogit import fit_conditional_logit
from horsetrack.countries.japan.models.v0.calibration import (
    STORE_COLUMNS,
    _development_frame,
    fit_win_calibration,
)
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table
from horsetrack.countries.japan.studies.calibration import _markdown_table, store_builds
from horsetrack.countries.japan.studies.cross_pool import boards_at

COUNTRY = "japan"
TICKET = 218
HYPOTHESIS_ID = "E218"  # no 08 §4 number: the ticket's, as #256's E256
FAMILY_ID_V1 = "japan-form-bands-t2-win"  # the pre-registered family, frozen and run first
FAMILY_ID = "japan-form-bands-t2-win-v2"  # the corrected eligibility (review of PR #473)
BOOTSTRAP_SEED = 218
N_BOOT = 2000  # the approved pre-registration's meeting resamples
FWE_LEVEL = 0.05
BLOCK_SIZE = 1  # meetings: the stationary bootstrap's block 1 is D7's i.i.d. meeting resampling
MIN_MEETINGS = 3  # arch's SPA / StepM need at least three observations (meetings)
SOURCES = ("jra", "jravan")  # the store's results (netkeiba) and the JRA-VAN T-2 boards
REFERENCE = "P0"
VARIANTS = ("P0", "B1", "B2", "B3", "B4", "B5", "R1", "R2", "R3", "R4", "BA", "RA")
NON_REFERENCE = tuple(v for v in VARIANTS if v != REFERENCE)
BAND_EDGES = (3.0, 7.0, 15.0, 50.0)  # left-closed: [1, 3), [3, 7), [7, 15), [15, 50), [50, inf)
RANK_UPPER = (1, 3, 6)  # rank groups 1, 2-3, 4-6, 7+
T2_MINUTES = float(load_country_yaml(COUNTRY, "decision")["primary_min"])
DATE_FROM = date(2020, 1, 1)  # 2020-Q1, the first quarter v0 and #129 score out of fold
DATE_TO = date(2026, 11, 15)  # the last development race day on disk (D14)
V0_OOF = "models/japan/v0/win_calibration_oof-aac501f927b0.parquet"  # #114's v0, #129's V2 input
ODDS_COLUMNS = [
    "race_id",
    "pool",
    "combination",
    "snapshot_utc",
    "fetched_utc",
    "minutes_to_post",
    "odds",
    "status_raw",
]


def columns_of(variant: str) -> list[str]:
    """The design column names of `variant`."""
    if variant == "P0":
        return ["all"]
    if variant == "BA":
        return [f"band{j}" for j in range(1, 6)]
    if variant == "RA":
        return [f"rank{j}" for j in range(1, 5)]
    if variant in VARIANTS:
        kind = "band" if variant[0] == "B" else "rank"
        return [f"{kind}{variant[1:]}"]
    raise ValueError(f"not a variant of the family: {variant!r}")


# --------------------------------------------------------------------------- the T-2 board


def t2_win_odds(day: pd.DataFrame, *, minutes_before: float = T2_MINUTES) -> pd.DataFrame:
    """One row per runner on each race's Win board at T-2 (`cross_pool.boards_at`): race_id,
    horse_no (int) and win_odds_t2 (float). Scratched and unpriced runners are off the board."""
    boards = boards_at(day, minutes_before=minutes_before)
    rows = [
        (rid, int(horse), float(odds))
        for rid, board in boards.items()
        for horse, odds in sorted(board.win.items())
    ]
    frame = pd.DataFrame(rows, columns=["race_id", "horse_no", "win_odds_t2"])
    return frame.astype({"race_id": object, "horse_no": "int64", "win_odds_t2": "float64"})


def assign_groups(board: pd.DataFrame) -> pd.DataFrame:
    """`board` with `band`, `fav_rank` and `rank_group` added (see the module doc); a row without
    a T-2 price has all three missing. Index, row order and other columns are kept."""
    out = board.copy()
    odds = pd.to_numeric(out["win_odds_t2"], errors="coerce").astype(float)
    priced = odds.notna().to_numpy()
    band = 1 + np.searchsorted(np.array(BAND_EDGES), odds.to_numpy(), side="right")
    rank = odds.groupby(out["race_id"].to_numpy()).rank(method="min")
    group = 1 + sum((rank > upper).astype(int) for upper in RANK_UPPER)
    out["band"] = pd.array(np.where(priced, band, 0), dtype="Int64")
    out["fav_rank"] = pd.array(np.where(priced, rank.fillna(0), 0).astype(int), dtype="Int64")
    out["rank_group"] = pd.array(np.where(priced, group, 0), dtype="Int64")
    out.loc[~priced, ["band", "fav_rank", "rank_group"]] = pd.NA
    return out


# --------------------------------------------------------------------------- the meld


@dataclass(frozen=True)
class MeldFit:
    """The meld's market weight `a`, its coefficient per design column `b` and the races fitted."""

    a: float
    b: dict[str, float]
    n_races: int
    se: dict[str, float] = field(default_factory=dict)


def _in_group(rows: pd.DataFrame, column: str) -> np.ndarray:
    if column == "all":
        return np.ones(len(rows), dtype=bool)
    kind, number = ("band", column[4:]) if column.startswith("band") else ("rank_group", column[4:])
    values = pd.to_numeric(rows[kind], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return values == int(number)


def design(rows: pd.DataFrame, variant: str) -> pd.DataFrame:
    """The ln h columns of `variant`: column g is 1[runner in g] * ln h, exactly 0.0 outside g."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_h = np.log(rows["h"].to_numpy(dtype=float))
    return pd.DataFrame(
        {c: np.where(_in_group(rows, c), ln_h, 0.0) for c in columns_of(variant)},
        index=rows.index,
    )


def fit_meld(rows: pd.DataFrame, variant: str) -> MeldFit:
    """The unpenalised conditional-logit MLE of u = a ln p_cal + design(rows, variant) @ b.
    A design column with no runner in `rows` is a ValueError: the group cannot be fitted."""
    X = design(rows, variant)
    empty = [c for c in X.columns if not X[c].to_numpy().any()]
    if empty:
        raise ValueError(f"{variant}: no runner in group(s) {empty} to fit")
    ln_p = np.log(rows["p_cal"].to_numpy(dtype=float))
    fit = fit_conditional_logit(
        np.column_stack([ln_p, X.to_numpy(dtype=float)]),
        rows["race_id"].to_numpy(),
        rows["won"].to_numpy(dtype=bool),
    )
    names = list(X.columns)
    return MeldFit(
        a=float(fit.coef[0]),
        b={n: float(c) for n, c in zip(names, fit.coef[1:], strict=True)},
        n_races=int(fit.n_races),
        se={"a": float(fit.se[0]), **{n: float(s) for n, s in zip(names, fit.se[1:], strict=True)}},
    )


def score_races(rows: pd.DataFrame, variant: str, fit: MeldFit) -> pd.DataFrame:
    """Per race with a winner: race_id, meeting_id, ll_v0, ll_variant and gain (D6, #69)."""
    starters = rows[~rows["scratched"].fillna(False).to_numpy(dtype=bool)]
    codes, races = pd.factorize(starters["race_id"].to_numpy())
    n = len(races)
    p_cal = starters["p_cal"].to_numpy(dtype=float)
    p_cal = p_cal / np.bincount(codes, weights=p_cal, minlength=n)[codes]
    X = design(starters, variant).to_numpy(dtype=float)
    b = np.array([fit.b[c] for c in columns_of(variant)])
    u = fit.a * np.log(p_cal) + X @ b
    top = pd.Series(u).groupby(codes).transform("max").to_numpy()
    e = np.exp(u - top)
    p = e / np.bincount(codes, weights=e, minlength=n)[codes]
    won = starters["won"].fillna(False).to_numpy(dtype=bool)
    winners = np.bincount(codes, weights=won, minlength=n)
    w = np.where(won, 1.0 / np.maximum(winners[codes], 1.0), 0.0)
    ll_v0 = np.bincount(codes, weights=w * -np.log(np.where(won, p_cal, 1.0)), minlength=n)
    ll_var = np.bincount(codes, weights=w * -np.log(np.where(won, p, 1.0)), minlength=n)
    meeting = starters["meeting_id"].groupby(codes).first().to_numpy()
    has = winners > 0
    out = pd.DataFrame(
        {
            "race_id": races[has],
            "meeting_id": meeting[has],
            "ll_v0": ll_v0[has],
            "ll_variant": ll_var[has],
        }
    )
    out["gain"] = out["ll_v0"] - out["ll_variant"]
    return out.sort_values("race_id", kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class MeldRun:
    """The scored races (`score_races` plus `fold`), each scored fold's fit, and the drops."""

    per_race: pd.DataFrame
    fits: dict[int, MeldFit]
    n_unpriced_dropped: int
    n_void_dropped: int


def _kept(rows: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """The rows of the races with every starter on the T-2 board and a winner, and the counts of
    races dropped for each reason (unpriced first)."""
    race = rows["race_id"].to_numpy()
    starter = ~rows["scratched"].fillna(False).to_numpy(dtype=bool)
    unpriced = set(race[starter & rows["win_odds_t2"].isna().to_numpy()])
    keep = ~rows["race_id"].isin(unpriced).to_numpy()
    won = rows["won"].fillna(False).to_numpy(dtype=bool) & starter
    with_winner = set(race[keep & won])
    void = set(race[keep]) - with_winner
    keep &= ~rows["race_id"].isin(void).to_numpy()
    return rows[keep], len(unpriced), len(void)


def walk_forward_meld(rows: pd.DataFrame, folds: Sequence[Fold], variant: str) -> MeldRun:
    """Fit fold k on the starters of folds < k up to folds[k].train_end and score fold k's
    races; the earliest fold present only trains (see the module doc)."""
    kept, n_unpriced, n_void = _kept(rows)
    starters = kept[~kept["scratched"].fillna(False).to_numpy(dtype=bool)]
    days = pd.to_datetime(starters["race_date"])
    parts, fits = [], {}
    for k in sorted(int(f) for f in kept["fold"].unique()):
        train = starters[
            (starters["fold"] < k).to_numpy()
            & (days <= pd.Timestamp(folds[k].train_end)).to_numpy()
        ]
        if train.empty:
            continue
        fits[k] = fit = fit_meld(train, variant)
        parts.append(score_races(kept[kept["fold"] == k], variant, fit).assign(fold=k))
    columns = ["race_id", "meeting_id", "ll_v0", "ll_variant", "gain", "fold"]
    per_race = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    return MeldRun(per_race[columns], fits, n_unpriced, n_void)


@dataclass(frozen=True)
class GainInterval:
    """The per-race mean gain, D7's one-sided 95% lower bound, the two-sided 95% interval and
    the meetings resampled."""

    mean: float
    lower: float
    ci95: tuple[float, float]
    n_meetings: int


def gain_interval(per_race: pd.DataFrame, *, n_boot: int, seed: int) -> GainInterval:
    """D7's meeting bootstrap of the mean gain: `lower` at level 0.95, `ci95` at level 0.975."""
    gain = per_race.set_index("race_id")["gain"]
    meeting = per_race.set_index("race_id")["meeting_id"]
    one = paired_bootstrap(gain, meeting, n_boot=n_boot, level=0.95, seed=seed)
    two = paired_bootstrap(gain, meeting, n_boot=n_boot, level=0.975, seed=seed)
    return GainInterval(
        mean=float(one.mean), lower=float(one.lo), ci95=(float(two.lo), float(two.hi)),
        n_meetings=int(one.n_clusters),
    )  # fmt: skip


# --------------------------------------------------------------------------- the spec


def build_spec(n_boot: int, data: Mapping | None = None) -> dict:
    """v1: the approved pre-registration as frozen in the registry (owner, #218, 2026-11-25).
    Kept byte-identical: the v1 family's frozen hash is built from it."""
    return {
        "family_id": FAMILY_ID_V1,
        "ticket": TICKET,
        "hypothesis_id": HYPOTHESIS_ID,
        "question": "does #129's meld weight on form, negative given the market in most folds, "
        "hold within T-2 odds bands and favourite ranks? Primary, stated before looking: is b < 0 "
        "and the gain > 0 in the favourite groups (rank 1 and 2-3) more than in the longshot "
        "bands? A negative b only at [50, inf) reads as the favourite-longshot bias already in "
        "v0, not as form being overweighted",
        "null": "b_g = 0 and no gain over v0 in every group (the starting point)",
        "meld": "within race p ∝ exp(a ln p_cal + sum_g b_g 1[runner in g] ln h), unpenalised "
        "conditional-logit MLE (models.condlogit)",
        "inputs": {
            "p_cal": f"v0, the calibrated-market Win probability out of fold ({V0_OOF}, #114)",
            "h": "#129's out-of-fold handicap_score (features/jra/handicap/<day>.parquet)",
            "groups": "each race's Win board at T-2 (D2's primary), the JRA-VAN time series "
            "(processed/jravan/odds_timeseries, #369), read as studies.cross_pool.boards_at: the "
            "latest row per runner at minutes_to_post >= 2; scratched and unpriced runners leave",
        },
        "bands": {"edges": list(BAND_EDGES), "closed": "left", "groups": 5},
        "rank_groups": {"cuts": "1, 2-3, 4-6, 7+", "ties": "share the better rank", "groups": 4},
        "variants": {
            "P0": "#129's V2: one b over every runner (the reference, re-run as is)",
            **{f"B{j}": f"one b for odds band {j}, b = 0 elsewhere" for j in range(1, 6)},
            **{f"R{j}": f"one b for rank group {j}, b = 0 elsewhere" for j in range(1, 5)},
            "BA": "all five band coefficients together",
            "RA": "all four rank coefficients together",
        },
        "no_other": "no other band edges, cuts, alphas or features; any change is a new ticket",
        "walk_forward": "#70's folds (config/folds.yaml); fold k fitted on rows of folds < k with "
        "race_date <= folds[k].train_end, scored on fold k; 2020-Q1 only trains",
        "scoring": "D6 (#69): winner log loss per race, a dead heat's winners at 1/k, void races "
        "dropped and counted, scratched runners removed and p_cal renormalised over the starters; "
        "a race with a starter off the T-2 board dropped and counted; gain = ll_v0 - ll_variant "
        "in nats per race on the common races",
        "per_group": "b_g by fold (median and range), the group's runners and races",
        "inference": {
            "per_variant": {
                "bootstrap": "paired, meetings (venue x day) resampled (D7)",
                "n_boot": n_boot,
                "seed": BOOTSTRAP_SEED,
                "lower": "one-sided 95% (level 0.95)",
                "ci95": "two-sided 95% (levels 0.025 and 0.975)",
            },
            "family": {
                "tests": ["Hansen SPA", "Romano-Wolf StepM"],
                "rules": list(NON_REFERENCE),
                "benchmark": "v0",
                "fwe": FWE_LEVEL,
                "bootstrap": "arch stationary bootstrap over per-meeting loss sums in date "
                f"order, block_size {BLOCK_SIZE} (i.i.d. meetings, D7)",
                "n_boot": n_boot,
                "seed": BOOTSTRAP_SEED,
            },
        },
        "rulings": [
            "favourite ranks count every runner priced on the T-2 board, a later scratch "
            "included; a runner off the board has no rank",
            "a variant whose group holds no runner is a failed record, not a b = 0 fit",
            "ci95 is the 2.5% / 97.5% quantile pair of D7's bootstrap with the same seed",
        ],
        "data": dict(data or {"rows": "as passed to run_family"}),
        "holdout": "development days only (D14); A2 from 2026-11-28 is never read or opened",
        "approved": "owner on #218: the draft of 2026-11-24 with the T-2 correction (2026-11-25), "
        "T-2 boards from JRA-VAN (2026-11-25), reframed 2026-11-27",
        "adoption": "by rank (owner, 2026-11-27); nothing is adopted here",
        "sources": list(SOURCES),
    }


def build_spec_v2(n_boot: int, data: Mapping | None = None) -> dict:
    """v2: v1's design exactly (variants, groups, meld, walk-forward, scoring, inference), with
    the corrected eligibility of the rows in `data` (`data_description_v2`)."""
    spec = build_spec(n_boot, data)
    spec["inputs"] = {
        **spec["inputs"],
        "p_cal": f"v0, the calibrated-market Win probability out of fold ({V0_OOF}, #114); a "
        "first-place dead heat, which that file leaves out, gets its fold's v0 fit refitted "
        "from the store as models.v0.calibration.run_fit makes it, the refit checked to "
        "reproduce the file on every race the file holds and the run keeps",
    }
    spec["rulings"] = [
        *spec["rulings"],
        "a race v0 cannot score (void, no priced winner) is dropped and counted by reason",
        "every v0 row is checked against the store's current starter fields; a race where they "
        "disagree, a confirmed starter without a v0 row, an unnumbered confirmed starter or a "
        "T-2 runner the store does not hold takes its race out, each race counted once",
    ]
    return {
        **spec,
        "family_id": FAMILY_ID,
        "supersedes": f"{FAMILY_ID_V1}: its data.rows rule read a T-2 runner missing from v0 as "
        "a scratch and its rows left out first-place dead heats (round-1 and round-2 reviews of "
        "PR #473); the design is unchanged",
        "approved": "v1's pre-registration (owner on #218, 2026-11-25; reframed 2026-11-27), "
        "with the eligibility corrected and frozen by Claude under the 2026-11-27 reframe (later "
        "variants added by Claude and counted, without a new sign-off)",
    }


# --------------------------------------------------------------------------- the run


def _refuse_holdout(rows: pd.DataFrame) -> None:
    days = sorted(set(pd.to_datetime(rows["race_date"]).dt.date))
    held = [d for d in days if is_holdout_day(COUNTRY, d)]
    if held:
        raise ValueError(
            f"{len(held)} holdout race day(s) in the rows, the first {held[0].isoformat()} (D14)"
        )


def _group_stats(groups: pd.DataFrame, run: MeldRun, variant: str) -> dict:
    scored = set(run.per_race["race_id"])
    starters = groups[
        ~groups["scratched"].fillna(False).to_numpy(dtype=bool)
        & groups["race_id"].isin(scored).to_numpy()
    ]
    out = {}
    for column in columns_of(variant):
        b = [fit.b[column] for fit in run.fits.values()]
        inside = starters[_in_group(starters, column)]
        out[column] = {
            "b_median": float(np.median(b)),
            "b_min": float(min(b)),
            "b_max": float(max(b)),
            "runners": len(inside),
            "races": int(inside["race_id"].nunique()),
        }
    return out


def _metric(fold: str, metric: str, value: float, n: int, lo=None, hi=None) -> FoldMetric:
    values = [v for v in (value, lo, hi) if v is not None]
    if not all(np.isfinite(values)):
        raise ValueError(f"{metric} on {fold} is not finite: {value}, [{lo}, {hi}]")
    return FoldMetric(
        str(fold),
        metric,
        float(value),
        None if lo is None else float(lo),
        None if hi is None else float(hi),
        int(n),
    )


def _metrics(result: dict, run: MeldRun, folds: Sequence[Fold]) -> tuple[FoldMetric, ...]:
    out = [
        _metric("development", "delta_ll_vs_v0", result["gain"], result["n_races"],
                result["lower"], None),
        _metric("development", "delta_ll_vs_v0_ci95", result["gain"], result["n_races"],
                *result["ci95"]),
    ]  # fmt: skip
    for column, g in result["groups"].items():
        out.append(
            _metric("development", f"b_median:{column}", g["b_median"], g["runners"],
                    g["b_min"], g["b_max"])
        )  # fmt: skip
    for k, fit in sorted(run.fits.items()):
        mine = run.per_race[run.per_race["fold"] == k]
        label = folds[k].test_start.isoformat()
        out.append(_metric(label, "delta_ll_vs_v0", mine["gain"].mean(), len(mine)))
        out.append(_metric(label, "a", fit.a, fit.n_races))
        out += [_metric(label, f"b:{c}", v, fit.n_races) for c, v in fit.b.items()]
    return tuple(out)


def _run_id(record: RunRecord) -> str:
    """The registry's run id: the sha256 of the record's canonical JSON."""
    canonical = json.dumps(
        dataclasses.asdict(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


class _RunView:
    """The registry as this run's family test reads it: the run's own records (each one in the
    registry, appended or reproduced), with the family's size and frozen spec from the whole
    history. A variant that scored in an earlier run and fails now is tested as failed."""

    def __init__(self, registry: Registry, records: Sequence[RunRecord]) -> None:
        self._registry, self._records = registry, list(records)

    def records(self, family_id: str | None = None) -> list[RunRecord]:
        return [r for r in self._records if family_id is None or r.family_id == family_id]

    def family_size(self, family_id: str) -> int:
        return self._registry.family_size(family_id)

    def frozen_spec(self, family_id: str) -> dict | None:
        return self._registry.frozen_spec(family_id)


def _free_stem(out_dir: Path, stem: str) -> str:
    """`stem`, or `stem-<n>` when a report of that name exists: a report is never overwritten."""
    candidate, n = stem, 1
    while (out_dir / f"{candidate}.md").exists() or (out_dir / f"{candidate}.json").exists():
        n += 1
        candidate = f"{stem}-{n}"
    return candidate


def _family_tests(runs: dict[str, MeldRun], rows: pd.DataFrame, registry, n_boot: int):
    """SPA and StepM over the scored non-reference variants against v0, on their common races,
    and why they did not run (None when they did)."""
    names = [v for v in NON_REFERENCE if v in runs]
    if not names:
        return None, None, "no non-reference variant scored"
    common = set.intersection(*(set(runs[v].per_race["race_id"]) for v in names))
    frames = {v: runs[v].per_race.set_index("race_id").loc[sorted(common)] for v in names}
    first = frames[names[0]]
    meetings = first["meeting_id"].nunique()
    if meetings < MIN_MEETINGS:
        return None, None, f"{meetings} meeting(s) scored; the bootstrap needs {MIN_MEETINGS}"
    losses = pd.DataFrame({v: frames[v]["ll_variant"] for v in names})
    day = rows.drop_duplicates("race_id").set_index("race_id")["race_date"]
    kwargs = dict(
        clusters=first["meeting_id"],
        order=pd.to_datetime(day.loc[first.index]),
        block_size=BLOCK_SIZE,
        n_boot=n_boot,
        seed=BOOTSTRAP_SEED,
        registry=registry,
        family_id=FAMILY_ID,
        reference_variants=(REFERENCE,),
    )
    s = spa(first["ll_v0"], losses, **kwargs)
    m = stepm(first["ll_v0"], losses, level=FWE_LEVEL, **kwargs)
    return (
        {"pvalue": s.pvalue, "pvalue_lower": s.pvalue_lower, "pvalue_upper": s.pvalue_upper,
         "n_rules": s.n_rules, "family_size": s.family_size, "n_meetings": s.n_clusters,
         "races": len(common)},
        {"superior": sorted(m.superior, key=VARIANTS.index), "family_size": m.family_size,
         "n_meetings": m.n_clusters},
        None,
    )  # fmt: skip


def run_family(
    rows: pd.DataFrame,
    folds: Sequence[Fold],
    *,
    registry: Registry,
    out_dir: Path | str,
    n_boot: int = N_BOOT,
    data: Mapping | None = None,
    builds: Mapping[str, str] | None = None,
    scheme: FoldScheme | None = None,
    counts: Mapping[str, int] | None = None,
) -> dict:
    """Freeze the spec, score the 12 variants, record each, run SPA and StepM and write the
    report (see the module doc). Refuses holdout-day rows (ValueError) and an `out_dir` inside
    the repository before anything is written."""
    _refuse_holdout(rows)
    out_dir = paths.outside_repo(Path(out_dir), what="out_dir")
    spec = build_spec_v2(n_boot, data)
    spec_hash = registry.freeze_family(FAMILY_ID, spec)

    groups = assign_groups(rows)
    fold_scheme = "walk_forward:" + (
        json.dumps(dataclasses.asdict(scheme), sort_keys=True) if scheme else "folds passed in"
    )
    git_sha = _git_sha()
    variants, runs, records, errors, drops = {}, {}, [], {}, {}
    for variant in VARIANTS:
        try:
            run = walk_forward_meld(groups, folds, variant)
            interval = gain_interval(run.per_race, n_boot=n_boot, seed=BOOTSTRAP_SEED)
            result = {
                "status": "scored",
                "gain": interval.mean,
                "lower": interval.lower,
                "ci95": list(interval.ci95),
                "n_races": len(run.per_race),
                "n_meetings": interval.n_meetings,
                "groups": _group_stats(groups, run, variant),
            }
            metrics = _metrics(result, run, folds)
            runs[variant] = run
            drops = {"unpriced_races": run.n_unpriced_dropped, "void_races": run.n_void_dropped}
        except Exception as exc:  # every failure is a record, never dropped
            result, metrics = {"status": "failed"}, ()
            errors[variant] = f"{type(exc).__name__}: {exc}"
        variants[variant] = result
        records.append(
            RunRecord(
                hypothesis_id=HYPOTHESIS_ID,
                family_id=FAMILY_ID,
                variant=variant,
                ticket=TICKET,
                git_sha=git_sha,
                data_builds=dict(builds or {}),
                sources=SOURCES,
                fold_scheme=fold_scheme,
                components={"meld": spec["meld"], "groups": spec["inputs"]["groups"]},
                params={"design_columns": columns_of(variant), "reference": variant == REFERENCE},
                metrics=metrics,
                status=result["status"],
            )
        )
    existing = registry.records(FAMILY_ID)
    appended = [registry.append(r) for r in records if r not in existing]
    run_ids = [_run_id(r) for r in records]  # this run's records, new or reproduced
    if not set(appended) <= set(run_ids):
        raise RuntimeError("an appended record id differs from the run's own record ids")
    view = _RunView(registry, records)
    spa_result, stepm_result, not_run = _family_tests(runs, rows, view, n_boot)

    result = {
        "family_id": FAMILY_ID,
        "spec_hash": spec_hash,
        "records_appended": len(appended),
        "records_reproduced": len(records) - len(appended),
        "run_ids": [run_id[:16] for run_id in run_ids],
        "family_size": registry.family_size(FAMILY_ID),
        "variants": variants,
        "errors": errors,
        "spa": spa_result,
        "stepm": stepm_result,
        "family_test_not_run": not_run,
        "counts": {**dict(counts or {}), **drops, "folds": len(folds)},
    }
    now = datetime.now(timezone.utc)
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "builds": ", ".join(f"{k}={v}" for k, v in (builds or {}).items()) or "-",
        "run ids (this run's records)": ", ".join(result["run_ids"]),
        "records appended / reproduced": f"{len(appended)} / {result['records_reproduced']}",
        "written (UTC)": now.isoformat(timespec="seconds"),
    }
    stem = _free_stem(out_dir, f"form-bands-{COUNTRY}-{spec_hash[:12]}-{now:%Y%m%dT%H%M%S%fZ}")
    report = write_report(out_dir / f"{stem}.md", render_report(result, runs, folds, about),
                          sources=SOURCES)  # fmt: skip
    write_report(out_dir / f"{stem}.json", json.dumps(result, indent=2, default=str) + "\n",
                 sources=SOURCES)  # fmt: skip
    return {**result, "report_path": str(report), "registry_path": str(registry.path)}


# --------------------------------------------------------------------------- the report


READING = (
    "- Every gain here is the gain of the registered meld over v0, not a measure of form alone. "
    "Two confounds are built into the frozen design (round-1 review of PR #473):",
    "  - Group intercept: b_g multiplies ln h masked to group g, and ln h within a race shares a "
    "normaliser and a typical level, so b_g also acts as a group intercept. Since ln h <= 0, a "
    "b_g < 0 adds b_g ln h >= 0 to each of the group's runners: against b_g = 0 it raises (or "
    "keeps) the group's share of the race, as an under-priced group would need, whether or not "
    "it uses the form differences between the group's runners.",
    "  - Market refit: every variant refits the market weight a, while v0 is unchanged, so each "
    "gain and the family test (SPA / StepM against v0) also credit the market refit of a, not "
    "only the form terms.",
    "  - A comparison that holds group intercepts and the market refit fixed is a new family "
    "for the owner to approve (#474); none is run or implied here.",
    "- v0's out-of-fold file leaves out first-place dead heats, races without a priced winner and "
    "starters without a final price. A first-place dead heat with every starter priced is scored "
    "here with its fold's v0 fit refitted from the store (dead_heat_races_refit); the refit "
    "reproduces the file on every race the file holds and the run keeps, or the run stops. The "
    "other races are dropped and counted, each once, in the data table (dropped_void, "
    "dropped_no_priced_winner, dropped_starter_without_v0, dropped_v0_disagrees_with_store, "
    "dropped_unnumbered_starter and the rest).",
)
PRIMARY = (
    "Asked before the run: is b < 0 and the gain > 0 in the favourite groups (R1, R2) more than "
    "in the longshot bands? The frozen spec defines no favourite-vs-longshot comparison rule (no "
    "measure or threshold that decides which effect is stronger), and none is chosen after the "
    "run. **Answer under the frozen spec: inconclusive.** The table gives each group's own "
    "reading only (its b by fold and its gain of the registered meld over v0); a deciding rule "
    "needs a new, owner-approved family.",
)


def _ranked(variants: dict) -> list[dict]:
    scored = [v for v in VARIANTS if variants[v]["status"] == "scored"]
    order = sorted(scored, key=lambda v: -variants[v]["lower"])
    rows = []
    for rank, v in enumerate(order, start=1):
        r = variants[v]
        rows.append(
            {"rank (by lower)": rank, "variant": v, "ΔLL mean": r["gain"],
             "lower 95% one-sided": r["lower"], "95% two-sided lo": r["ci95"][0],
             "95% two-sided hi": r["ci95"][1], "races": r["n_races"],
             "meetings": r["n_meetings"]}
        )  # fmt: skip
    return rows


def render_report(result: dict, runs: dict[str, MeldRun], folds: Sequence[Fold], about) -> str:
    """Aggregate tables only, never a runner or race row."""
    variants = result["variants"]
    lines = [
        f"# Form within the market by T-2 odds band and favourite rank ({COUNTRY}, #{TICKET})",
        "",
        "PRIVATE: built on JRA-VAN T-2 boards (D21). Never copy a figure into the repository.",
        "",
        *(f"- {key}: {value}" for key, value in about.items()),
        f"- variants searched (registry family size): {result['family_size']}",
        "- development days only: holdout days are never read and the holdout is not opened",
        "- adoption: by rank (owner, 2026-11-27); nothing is adopted here",
        "",
        "## Data",
        "",
        _markdown_table([{"count": k, "n": v} for k, v in result["counts"].items()]),
        "",
        "## Reading these results",
        "",
        *READING,
        "",
        "## Leaderboard: gain of the registered meld over v0 (nats per race, D7 bootstrap)",
        "",
        "Ranked by the one-sided 95% lower bound; P0 is #129's V2, the reference. Each figure is "
        "the gain of the registered meld over v0, not a measure of form alone (above).",
        "",
        _markdown_table(_ranked(variants)) if runs else "_No variant scored._",
        "",
        "## Group coefficients over the scored folds' fits, and the group sizes",
        "",
    ]
    group_rows = [
        {"variant": v, "column": c, **g}
        for v in VARIANTS
        if variants[v]["status"] == "scored"
        for c, g in variants[v]["groups"].items()
    ]
    lines.append(_markdown_table(group_rows) if group_rows else "_None._")
    lines += ["", "## Primary question (stated before the run)", "", *PRIMARY, ""]
    primary = []
    for v in ("R1", "R2", "R3", "R4", "B1", "B2", "B3", "B4", "B5"):
        r = variants[v]
        if r["status"] != "scored":
            primary.append({"variant": v, "status": "failed"})
            continue
        b = r["groups"][columns_of(v)[0]]
        primary.append(
            {"variant": v, "status": "scored", "b median": b["b_median"], "b min": b["b_min"],
             "b max": b["b_max"], "gain mean": r["gain"], "lower 95%": r["lower"],
             "b median < 0": b["b_median"] < 0, "lower > 0": r["lower"] > 0}
        )  # fmt: skip
    lines.append(_markdown_table(primary))
    n_rules = result["spa"]["n_rules"] if result["spa"] else 0
    lines += ["", f"## Family test over this run's {n_rules} scored non-reference variants", ""]
    if result["spa"] is None:
        lines.append(f"_Not run: {result['family_test_not_run']}._")
    else:
        lines += [
            _markdown_table([{"test": "SPA", **result["spa"]}]),
            "",
            f"StepM at {FWE_LEVEL:.0%} family-wise: superior "
            f"{', '.join(result['stepm']['superior']) or 'none'}.",
        ]
    lines += ["", "## Per fold", ""]
    fold_rows = []
    for v, run in runs.items():
        for k, fit in sorted(run.fits.items()):
            mine = run.per_race[run.per_race["fold"] == k]
            fold_rows.append(
                {"variant": v, "test start": folds[k].test_start.isoformat(),
                 "train end": folds[k].train_end.isoformat(), "races": len(mine),
                 "ΔLL mean": mine["gain"].mean(), "a": fit.a,
                 **{f"b:{c}": b for c, b in fit.b.items()}}
            )  # fmt: skip
    lines.append(_markdown_table(fold_rows) if fold_rows else "_None._")
    status = [
        {"variant": v, "status": variants[v]["status"], "error": result["errors"].get(v, "")}
        for v in VARIANTS
    ]
    lines += ["", "## Variants (one registry record each)", "", _markdown_table(status), ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- real data


def data_description(date_from: date, date_to: date, scheme: FoldScheme) -> dict:
    return {
        "days": f"development race days {date_from.isoformat()} to {date_to.isoformat()}",
        "p_cal": V0_OOF,
        "h": "features/jra/handicap/<day>.parquet: handicap_score",
        "t2": "processed/jravan/odds_timeseries/<day>.parquet: WIN rows only",
        "store": "processed/jra runners (finish_pos, is_scratched), races (meeting_id, race_date)",
        "rows": "the v0 starters of each race; a runner on the T-2 board but not among them is "
        "a scratched row (it ranks, it is not scored); a race with a starter without h dropped "
        "and counted",
        "fold_scheme": dataclasses.asdict(scheme),
    }


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _files_digest(files: list[Path]) -> str:
    h = hashlib.sha256()
    for p in files:
        h.update(f"{p.name}:{p.stat().st_size}\n".encode())
    return h.hexdigest()[:16]


def data_description_v2(date_from: date, date_to: date, scheme: FoldScheme) -> dict:
    """v2's rows: v1's sources, with the eligibility the reviews of PR #473 corrected."""
    return {
        "days": f"development race days {date_from.isoformat()} to {date_to.isoformat()}",
        "population": "the store's races on those days (processed/jra races and runners)",
        "p_cal": f"{V0_OOF}; for a first-place dead heat with every starter priced (which that "
        "file leaves out) the fold's v0 fit refitted from the store as "
        "models.v0.calibration.run_fit makes it (fit_win_calibration on the fold's training "
        "rows from 2014-06-07), predict_many on the race's normalised 1/odds, each co-winner "
        "won; the refit must reproduce the file on every race it holds and the run keeps",
        "h": "features/jra/handicap/<day>.parquet: handicap_score",
        "t2": "processed/jravan/odds_timeseries/<day>.parquet: WIN rows only",
        "store": "processed/jra runners (race_date, horse_no, win_odds, finish_pos, "
        "is_scratched, dead_heat; horse_no may be null), races (meeting_id, race_date)",
        "rows": "each race's starters are the store's confirmed starters (is_scratched not "
        "True); a runner the store scratches but the T-2 board prices is a scratched row (it "
        "ranks, it is not scored); a missing v0 row is never read as a scratch",
        "dropped": "each race once, counted under its first reason: void (no winner); no "
        "priced winner; a dead heat with an unpriced starter; not in v0 for another reason; an "
        "unnumbered confirmed starter; a v0 row the store's current starter fields contradict "
        "(runner absent, scratched or unpriced, or its normalised 1/odds changed); a confirmed "
        "starter without a v0 row; a T-2 runner the store does not hold; a starter without h",
        "fold_scheme": dataclasses.asdict(scheme),
    }


def _per_race(store: pd.DataFrame) -> pd.DataFrame:
    """Per store race: its winners, dead-heat winners, a priced winner, and whether it has an
    unnumbered or an unpriced numbered confirmed starter."""
    starter = store["starter"].to_numpy()
    numbered = store["horse_no"].notna().to_numpy()
    priced = np.isfinite(store["win_odds"].to_numpy(dtype=float))
    first = store["finish_pos"].eq(1).fillna(False).to_numpy(dtype=bool) & starter
    flags = pd.DataFrame(
        {
            "race_id": store["race_id"].to_numpy(),
            "winners": first,
            "dh_winners": first & store["dead_heat"].fillna(False).to_numpy(dtype=bool),
            "priced_winner": first & priced,
            "unnumbered": starter & ~numbered,
            "unpriced": starter & numbered & ~priced,
        }
    )
    return flags.groupby("race_id").agg(
        winners=("winners", "sum"),
        dh_winners=("dh_winners", "sum"),
        priced_winner=("priced_winner", "any"),
        unnumbered=("unnumbered", "any"),
        unpriced=("unpriced", "any"),
    )


def _store_q(store: pd.DataFrame) -> pd.DataFrame:
    """The store's numbered, priced confirmed starters with q, their normalised 1/odds."""
    usable = store[
        store["starter"].to_numpy()
        & store["horse_no"].notna().to_numpy()
        & np.isfinite(store["win_odds"].to_numpy(dtype=float))
    ][["race_id", "horse_no", "win_odds"]].copy()
    usable["horse_no"] = usable["horse_no"].astype("int64")
    inv = 1.0 / usable["win_odds"].to_numpy(dtype=float)
    totals = pd.Series(inv).groupby(usable["race_id"].to_numpy()).transform("sum").to_numpy()
    usable["q"] = inv / totals
    return usable[["race_id", "horse_no", "q"]]


def _refit_p_cal(
    frame: pd.DataFrame, v0_folds: Sequence[Fold], fold_rows: dict[int, pd.DataFrame]
) -> dict[int, np.ndarray]:
    """For each fold k, v0's fold fit refitted (`fit_win_calibration` on its training rows, as
    `run_fit`) and its p_cal for `fold_rows[k]` (race_id, q)."""
    out = {}
    for k, part in sorted(fold_rows.items()):
        model = fit_win_calibration(frame.iloc[v0_folds[k].train_idx])
        out[k] = model.predict_many(part["q"].to_numpy(dtype=float), part["race_id"].to_numpy())
        del model
    return out


def read_rows(date_from: date, date_to: date, scheme: FoldScheme) -> tuple:
    """The run's rows, the folds, the counts and the data builds (see `data_description_v2`).

    The population is the store's races on the development days read; each race left out is
    counted once, under its first reason (`data_description_v2`'s "dropped"). v0's fold fits are
    refitted from the store as `models.v0.calibration.run_fit` makes them; they must reproduce
    the v0 file's p_cal on every race the file holds and the run keeps (else ValueError), and
    they give a first-place dead heat its p_cal. One day of odds is read at a time, WIN rows
    only; the store is read column-scoped."""
    data = paths.data_dir()
    oof = pq.read_table(data / V0_OOF, columns=["race_id", "horse_no", "fold", "q", "p_cal"])
    oof = oof.to_pandas()
    oof["race_date"] = pd.to_datetime(oof["race_id"].str.slice(4, 14)).dt.date
    oof = oof[(oof["race_date"] >= date_from) & (oof["race_date"] <= date_to)]
    oof = oof[~oof["race_date"].map(lambda d: is_holdout_day(COUNTRY, d))]
    oof = oof.astype({"horse_no": "int64"})

    table = read_store_table(
        "runners", columns=STORE_COLUMNS, start=PRICE_ERA_START, end=date_to, touches_price=True
    )
    frame, _, _ = _development_frame(table)  # v0's development frame, as run_fit builds it
    store = table.filter(
        pc.and_(
            pc.greater_equal(table["race_date"], pa.scalar(date_from, pa.date32())),
            pc.less_equal(table["race_date"], pa.scalar(date_to, pa.date32())),
        )
    ).to_pandas()
    del table
    folds = walk_forward(frame["race_date"], scheme=scheme)  # v0's folds (run_fit)

    def fold_of(day: date) -> int | None:
        for k, fold in enumerate(folds):
            if fold.test_start <= day <= fold.test_end:
                return k
        return None

    day_fold = {d: fold_of(d) for d in sorted(set(oof["race_date"]) | set(store["race_date"]))}
    mine = oof["race_date"].map(day_fold)
    if mine.isna().any() or not (mine.astype(int) == oof["fold"]).all():
        raise ValueError("the v0 OOF folds differ from #70's folds over v0's development frame")

    races = read_store_table(
        "races", columns=["race_id", "race_date", "meeting_id"], start=date_from, end=date_to,
        touches_price=False,
    ).to_pandas()  # fmt: skip
    store = store[~store["race_date"].map(lambda d: is_holdout_day(COUNTRY, d))]
    store["horse_no"] = store["horse_no"].astype("Int64")
    store["win_odds"] = clean_odds(store["win_odds"].to_numpy())
    store["starter"] = ~store["is_scratched"].eq(True).to_numpy(dtype=bool, na_value=False)
    per = _per_race(store)

    v0_races = set(oof["race_id"])
    order: list[tuple[str, set[str]]] = []
    absent = per.loc[~per.index.isin(v0_races)]
    void = set(absent.index[absent["winners"] == 0])
    dead_heat = set(absent.index[(absent["winners"] >= 2) & (absent["dh_winners"] >= 2)]) - void
    unpriced_winner = set(absent.index[~absent["priced_winner"]]) - void - dead_heat
    dh_unpriced = {r for r in dead_heat if per.at[r, "unpriced"]}
    other = set(absent.index) - void - dead_heat - unpriced_winner
    order += [("dropped_void", void), ("dropped_no_priced_winner", unpriced_winner),
              ("dropped_dead_heat_unpriced_starter", dh_unpriced),
              ("dropped_not_in_v0_other", other)]  # fmt: skip
    candidates = (set(per.index) & v0_races) | (dead_heat - dh_unpriced)
    unnumbered = {r for r in candidates if per.at[r, "unnumbered"]}
    order.append(("dropped_unnumbered_starter", unnumbered))
    candidates -= unnumbered

    # B: every v0 row against the store's current starter fields
    store_q = _store_q(store)
    v0 = oof.merge(store_q, on=["race_id", "horse_no"], how="left", suffixes=("", "_store"))
    agrees = np.isclose(v0["q"], v0["q_store"], rtol=1e-9, atol=1e-12) & v0["q_store"].notna()
    disagree = (set(v0.loc[~agrees.to_numpy(), "race_id"]) & candidates) | (
        set(v0["race_id"]) - set(per.index)
    )
    in_v0_store = store_q[store_q["race_id"].isin(v0_races)]
    extra = in_v0_store.merge(oof[["race_id", "horse_no"]], how="left", indicator=True)
    disagree |= set(extra.loc[extra["_merge"] == "left_only", "race_id"]) & candidates
    order.append(("dropped_v0_disagrees_with_store", disagree))
    candidates -= disagree
    confirmed = store[store["starter"].to_numpy() & store["horse_no"].notna().to_numpy()]
    confirmed = confirmed[confirmed["race_id"].isin(v0_races & candidates)]
    keys = pd.MultiIndex.from_arrays(
        [confirmed["race_id"], confirmed["horse_no"].astype("int64")]
    )  # fmt: skip
    without_v0 = set(
        confirmed.loc[~keys.isin(pd.MultiIndex.from_frame(oof[["race_id", "horse_no"]])),
                      "race_id"]
    )  # fmt: skip
    order.append(("dropped_starter_without_v0", without_v0))
    candidates -= without_v0

    # A: v0's fold fits refitted; they must reproduce the file, and they price the dead heats
    kept_v0 = oof[oof["race_id"].isin(candidates)]
    dh_rows = store_q[store_q["race_id"].isin(candidates - v0_races)].copy()
    dh_rows["race_date"] = pd.to_datetime(dh_rows["race_id"].str.slice(4, 14)).dt.date
    dh_rows["fold"] = dh_rows["race_date"].map(day_fold)
    no_fold = set(dh_rows.loc[dh_rows["fold"].isna(), "race_id"])
    order.append(("dropped_dead_heat_no_fold", no_fold))
    candidates -= no_fold
    dh_rows = dh_rows[~dh_rows["race_id"].isin(no_fold)].astype({"fold": "int64"})
    fold_rows = {}
    for k in sorted(set(kept_v0["fold"].astype(int)) | set(dh_rows["fold"])):
        fold_rows[k] = pd.concat(
            [kept_v0.loc[kept_v0["fold"] == k, ["race_id", "q"]],
             dh_rows.loc[dh_rows["fold"] == k, ["race_id", "q"]]], ignore_index=True,
        )  # fmt: skip
    refit = _refit_p_cal(frame, folds, fold_rows)
    del frame
    for k, p in refit.items():
        n_v0 = int((kept_v0["fold"] == k).sum())
        file_p = kept_v0.loc[kept_v0["fold"] == k, "p_cal"].to_numpy(dtype=float)
        if not np.allclose(p[:n_v0], file_p, rtol=1e-6, atol=1e-9):
            worst = float(np.max(np.abs(p[:n_v0] - file_p)))
            raise ValueError(
                f"v0's fold {k} refitted from the store does not reproduce {V0_OOF} "
                f"(largest p_cal difference {worst:.3g}): the store's training rows changed"
            )
        dh_rows.loc[dh_rows["fold"] == k, "p_cal"] = p[n_v0:]
    dh_rows = dh_rows[["race_id", "horse_no", "fold", "q", "p_cal", "race_date"]]
    oof = pd.concat([kept_v0, dh_rows], ignore_index=True)
    days = sorted(set(oof["race_date"]))
    runners = store[store["horse_no"].notna().to_numpy()].copy()
    runners["horse_no"] = runners["horse_no"].astype("int64")
    runners["won"] = (
        runners["finish_pos"].eq(1).fillna(False).to_numpy(dtype=bool)
        & runners["starter"].to_numpy()
    )
    runners["store_scratched"] = ~runners["starter"]

    h_parts, t2_parts, odds_files, no_odds, no_h = [], [], [], 0, 0
    for day in days:
        hpath = handicap_dir() / f"{day.isoformat()}.parquet"
        if hpath.is_file():
            h_parts.append(
                pl.read_parquet(hpath, columns=["race_id", "horse_no", "handicap_score"])
                .with_columns(pl.col("horse_no").cast(pl.Int64))
                .to_pandas()
            )
        else:
            no_h += 1
        opath = paths.processed_dir("jravan") / "odds_timeseries" / f"{day.isoformat()}.parquet"
        if not opath.is_file():
            no_odds += 1
            continue
        odds_files.append(opath)
        table = pq.read_table(opath, columns=ODDS_COLUMNS, filters=[("pool", "=", "WIN")])
        t2_parts.append(t2_win_odds(table.to_pandas()))
        del table

    h = pd.concat(h_parts, ignore_index=True) if h_parts else pd.DataFrame(
        columns=["race_id", "horse_no", "handicap_score"])  # fmt: skip
    h = h.rename(columns={"handicap_score": "h"}).astype({"horse_no": "int64"})
    t2 = pd.concat(t2_parts, ignore_index=True) if t2_parts else pd.DataFrame(
        {"race_id": [], "horse_no": pd.Series(dtype="int64"), "win_odds_t2": []})  # fmt: skip
    t2 = t2[t2["race_id"].isin(set(oof["race_id"]))]

    starters = (
        oof.drop(columns="q")
        .merge(h, on=["race_id", "horse_no"], how="left", validate="one_to_one")
        .merge(runners[["race_id", "horse_no", "won"]], on=["race_id", "horse_no"], how="left")
        .merge(races[["race_id", "meeting_id"]], on="race_id", how="left", validate="many_to_one")
        .merge(t2, on=["race_id", "horse_no"], how="left", validate="one_to_one")
        .assign(scratched=False)
    )
    starters["won"] = starters["won"].fillna(False).astype(bool)
    extra = t2.merge(starters[["race_id", "horse_no"]], how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"].drop(columns="_merge")
    extra = extra.merge(
        runners[["race_id", "horse_no", "store_scratched"]], on=["race_id", "horse_no"],
        how="left", validate="one_to_one",
    )  # fmt: skip
    not_in_store = set(extra.loc[extra["store_scratched"].isna(), "race_id"])
    no_h_races = set(starters.loc[starters["h"].isna(), "race_id"]) - not_in_store
    dropped = not_in_store | no_h_races
    starters = starters[~starters["race_id"].isin(dropped)]
    race_info = starters.drop_duplicates("race_id")[["race_id", "race_date", "fold", "meeting_id"]]
    extra = extra[extra["store_scratched"].eq(True).to_numpy(dtype=bool, na_value=False)]
    extra = extra.drop(columns="store_scratched").merge(race_info, on="race_id", how="inner")
    extra = extra.assign(p_cal=np.nan, h=np.nan, won=False, scratched=True)
    rows = pd.concat([starters, extra[starters.columns]], ignore_index=True)
    rows = rows.sort_values(["race_date", "race_id", "horse_no"], kind="stable")
    order += [("dropped_t2_runner_not_in_store", not_in_store),
              ("races_with_a_starter_without_h", no_h_races)]  # fmt: skip
    counts = {
        "store_races": len(per),
        **{key: len(found) for key, found in order},
        "dead_heat_races_refit": int(
            starters.loc[~starters["race_id"].isin(v0_races), "race_id"].nunique()
        ),  # fmt: skip
        "race_days": len(days),
        "days_without_odds_file": no_odds,
        "days_without_handicap_file": no_h,
        "races_read": int(starters["race_id"].nunique()),
        "starters_read": len(starters),
        "t2_rows_scratched_in_store": len(extra),
    }
    handicap_files = sorted(handicap_dir().glob("*.parquet"))
    builds = {
        **store_builds(),
        "v0_oof": f"{V0_OOF}@{_digest(data / V0_OOF)}",
        "jra_handicap": f"features/jra/handicap@{_files_digest(handicap_files)}",
        "jravan_odds_timeseries": f"processed/jravan/odds_timeseries@{_files_digest(odds_files)}",
    }
    return rows.reset_index(drop=True), folds, counts, builds


def run_study(
    out_dir: Path | None = None,
    *,
    date_from: date = DATE_FROM,
    date_to: date = DATE_TO,
    n_boot: int = N_BOOT,
) -> dict:
    """The real run: freeze the spec before any row is read, build the rows and run the family."""
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    scheme = load_country_scheme(COUNTRY)
    data = data_description_v2(date_from, date_to, scheme)
    registry = default_registry(COUNTRY)
    registry.freeze_family(FAMILY_ID, build_spec_v2(n_boot, data))  # before any row is read
    rows, folds, counts, builds = read_rows(date_from, date_to, scheme)
    return run_family(
        rows, folds, registry=registry, out_dir=out_dir, n_boot=n_boot, data=data,
        builds=builds, scheme=scheme, counts=counts,
    )  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m horsetrack.countries.japan.studies.form_bands")
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    args = parser.parse_args(argv)
    result = run_study(args.out)
    summary = {k: result[k] for k in ("family_id", "spec_hash", "records_appended", "run_ids",
                                      "family_size", "report_path", "registry_path")}  # fmt: skip
    print(json.dumps(summary, indent=2))  # figures stay in the private report
    return 0


if __name__ == "__main__":
    sys.exit(main())
