"""E25: the T-2 cross-pool signal at the final Win price, settled in Win, Quinella, Exacta (#435).

N4 of epic #73, family `japan-cross-pool-t2-v1`, pre-registered in `docs/specs/cross-pool-prereg.md`
(Claude pre-registers under the owner's reframe of 2026-11-27: no sign-off stop; `--freeze` writes
the spec to the registry before any scored read). Hypothesis E25: at t = post - 2 min (D2's
primary), the Quinella pool's top-2 support against the Win-implied (Harville) top-2 support,
x_cross (#434, `studies.cross_pool`), adds to the final Win price: u = offset + beta . x with
beta != 0 and a walk-forward winner log-loss gain over the offset. Null: beta = 0, gain 0.

Variants (the family, size 4): feature set `x_cross` or `x_cross+fav` (x_cross and fav_x_cross =
is_fav . x_cross, the T-2 favourite's support across pools) times offset `normalised` (ln of
normalised 1/final Win odds) or `calibrated` (D4's natural cubic spline in ln q,
`models.v0.calibration.fit_win_calibration`, refitted on each fold's training rows). Each is
fitted per D9 fold on that fold's training rows only (`fit_fold`, the unpenalised conditional logit
of `models.condlogit`, clustered on the meeting) and scored on its test races: ΔLL = ll_offset -
ll_model per race (winner log loss, a dead heat's winners averaged), D7's one-sided meeting
bootstrap, overall and by class group; coefficients per fold and pooled over the scored rows with
the meeting-clustered sandwich se. Settlement: each variant and each offset alone (beta = 0) bets
WIN, QUINELLA and FORECAST (Exacta) at the final dividend (`settle`), pair probabilities by
Harville from the Win probabilities, claim = p x final odds; selections `value` (claim > 1) and
`top1` (the race's largest claim); N5 (`core.eval.tail.tail_curve`) over every priced combination.
Every settled selection and N5 band is a candidate cell (#430's schema), ranked by mean return.

Order (`run_cross_pool`): refuse a span before 2014-06-07 (D10) and an unfrozen family (the
registry's frozen hash must equal `build_spec`'s) before any store file is opened; list the
`odds_timeseries` day files and drop every holdout day by its name (D14: never opened, never
stat'd); read the rest one at a time; join the jravan `races`, `runners`, `final_odds` and
`dividends`; fold, fit, score, settle; append one registry record per variant (failures included:
a variant whose fit or scoring raises is a failed record and the others still run); write the
private report and the cell file through `write_report`. The CLI (`--freeze`, `--run`) then
regenerates the registry summary and `docs/experiments/cross-pool-japan.md`, whose figures are all
withheld (D21). Measurement only: nothing is adopted.

Memory: one day file at a time (WIN and QUINELLA rows at or before t only, the columns the board
needs); `final_odds` one pool and year at a time, its year row group picked by the race_date
statistics and decoded in batches filtered by pool and development race id (a whole-row-group scan
of a year holds every pool, Trifecta included); settlement holds one pool's combinations as flat
numeric arrays and one scorer's claims at a time; the per-runner Win probabilities are a dense
(race, horse number) array per scorer.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import resource
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from horsetrack.core import env, paths
from horsetrack.core.config import load_country_yaml, load_source
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import Fold, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day
from horsetrack.core.eval.registry import (
    FoldMetric,
    Registry,
    RunRecord,
    _spec_hash,
    default_registry,
    summary_dir,
    write_summary,
)
from horsetrack.core.eval.reports import write_report
from horsetrack.core.eval.tail import ALL, tail_curve
from horsetrack.countries.japan.models.condlogit import CLFit, fit_conditional_logit
from horsetrack.countries.japan.models.v0.calibration import _group_softmax, fit_win_calibration
from horsetrack.countries.japan.models.v0.store import (
    PRICE_ERA_START,
    PreReformPriceError,
    read_store_table,
)
from horsetrack.countries.japan.research.pool_scan import CLASS_GROUPS, POOL_ROW
from horsetrack.countries.japan.studies.cross_pool import QUINELLA, WIN, boards_at, cross_features

COUNTRY = "japan"
TICKET = 435
SOURCE = "jravan"
FAMILY_ID = "japan-cross-pool-t2-v1"
HYPOTHESIS = "E25"
STUDY = "cross-pool"
MINUTES_BEFORE = load_country_yaml(COUNTRY, "decision")["primary_min"]  # D2's primary, T-2
SPAN_START = PRICE_ERA_START  # D10
POOLS = ("WIN", "QUINELLA", "FORECAST")  # FORECAST is the store's name for Exacta
FEATURES = {"x_cross": ("x_cross",), "x_cross+fav": ("x_cross", "fav_x_cross")}
OFFSETS = ("normalised", "calibrated")
VARIANTS = tuple(f"{features}:{offset}" for features in FEATURES for offset in OFFSETS)
BASELINES = tuple(f"offset:{offset}" for offset in OFFSETS)  # each offset alone, beta = 0
SELECTIONS = ("value", "top1")
N5_QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)
N_KNOTS = 5  # fit_win_calibration's default, D4's live model
LAMBDA2 = 1.0  # plain Harville
ZERO_VOTE = "unavailable"  # #434's default
Z95 = 1.959963984540054  # the coefficient's two-sided 95% Wald interval
GROUPS = tuple(dict.fromkeys(CLASS_GROUPS.values()))  # newcomer, maiden, conditions, open, graded
DAY_COLUMNS = ["race_id", "pool", "combination", "snapshot_utc", "fetched_utc"]
DAY_COLUMNS += ["minutes_to_post", "odds", "status_raw"]
ROW_COLUMNS = ["race_id", "horse_no", "p_win_t", "x_cross", "is_fav", "fav_x_cross"]
_DAY_FILE = re.compile(r"(\d{4}-\d{2}-\d{2})\.parquet")


class FamilyNotFrozen(RuntimeError):
    """The registry holds no freeze of this run's pre-registered spec."""


# --------------------------------------------------------------------------- the model


def race_probs(offset, X, coef, race_id) -> np.ndarray:
    """The softmax per race of `offset + X @ coef` (rows need not be grouped by race)."""
    offset = np.asarray(offset, dtype=float)
    coef = np.asarray(coef, dtype=float).reshape(-1)
    u = offset.copy()
    if coef.size:
        X = np.asarray(X, dtype=float).reshape(len(offset), -1)
        u = u + X @ coef
    return _group_softmax(u, pd.factorize(np.asarray(race_id, dtype=object))[0])


def day_rows(day: pd.DataFrame, *, minutes_before: float = MINUTES_BEFORE) -> pd.DataFrame:
    """One odds_timeseries day -> one row per Win starter at t: `race_id, horse_no, p_win_t,
    x_cross, is_fav, fav_x_cross` from the boards at t (#434), and `quinella_board`, whether the
    race's Quinella board at t prices any starter pair."""
    parts = []
    for race_id, boards in boards_at(day, minutes_before=minutes_before).items():
        if not boards.win:
            continue
        feats = cross_features(boards, lambda2=LAMBDA2, zero_vote=ZERO_VOTE)
        x = feats["x_cross"].to_numpy(dtype=float)
        fav = feats["is_fav"].to_numpy(dtype=float)
        parts.append(
            pd.DataFrame(
                {
                    "race_id": race_id,
                    "horse_no": feats["horse_no"].to_numpy(dtype=np.int64),
                    "p_win_t": feats["p_win_t"].to_numpy(dtype=float),
                    "x_cross": x,
                    "is_fav": fav,
                    "fav_x_cross": fav * x,
                    "quinella_board": bool(boards.quinella),
                }
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else _no_rows()


def _no_rows() -> pd.DataFrame:
    empty = {c: pd.Series(dtype=float) for c in ROW_COLUMNS}
    empty |= {"race_id": pd.Series(dtype=object), "horse_no": pd.Series(dtype=np.int64)}
    return pd.DataFrame({**empty, "quinella_board": pd.Series(dtype=bool)})


def fit_fold(train: pd.DataFrame, *, features: Sequence[str]) -> CLFit:
    """The unpenalised conditional logit u = offset + X @ coef on `train`, clustered on meeting."""
    return fit_conditional_logit(
        train[list(features)].to_numpy(dtype=float),
        train["race_id"].to_numpy(),
        train["won"].to_numpy(dtype=bool),
        offset=train["offset"].to_numpy(dtype=float),
        cluster=train["meeting_id"].to_numpy(),
    )


def _market_q(rows: pd.DataFrame) -> np.ndarray:
    """1/odds normalised over each race's rows."""
    inv = 1.0 / rows["win_odds"].to_numpy(dtype=float)
    codes = pd.factorize(rows["race_id"].to_numpy(dtype=object))[0]
    return inv / np.bincount(codes, weights=inv)[codes]


def _offsets(name: str, rows: pd.DataFrame, q: np.ndarray, fold: Fold):
    """The offset (a log score per row) on the fold's training and test rows; the calibrated
    spline is fitted on the training rows only."""
    train_idx, test_idx = fold.train_idx, fold.test_idx
    if name == "normalised":
        return np.log(q[train_idx]), np.log(q[test_idx])
    race = rows["race_id"].to_numpy(dtype=object)
    train = rows.iloc[train_idx][["race_id", "race_date", "win_odds", "won"]]
    cal = fit_win_calibration(train, n_knots=N_KNOTS)
    p_train = cal.predict_many(q[train_idx], race[train_idx])
    p_test = cal.predict_many(q[test_idx], race[test_idx])
    return np.log(p_train), np.log(p_test)


@dataclass
class _Walk:
    """Per scorer (a variant or an offset baseline): one frame per fold of its test rows
    (`fold`, `pos` into rows, `p_offset`, `p_model`), the fits per fold, and the errors."""

    probs: dict[str, list[pd.DataFrame]] = field(default_factory=dict)
    fits: dict[str, list[tuple[int, CLFit]]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def fail(self, scorer: str, error: str) -> None:
        self.errors.setdefault(scorer, error)
        self.probs.pop(scorer, None)
        self.fits.pop(scorer, None)


def _walk(rows: pd.DataFrame, folds: Sequence[Fold], *, strict: bool) -> _Walk:
    """Every variant and offset baseline on every fold. `strict` raises a variant's error;
    otherwise the variant fails alone and the others go on."""
    walk = _Walk()
    q = _market_q(rows) if len(rows) else np.zeros(0)
    race = rows["race_id"].to_numpy(dtype=object)
    design = {name: rows[list(cols)].to_numpy(dtype=float) for name, cols in FEATURES.items()}
    for k, fold in enumerate(folds):
        if len(fold.test_idx) == 0:
            continue
        test_race = race[fold.test_idx]
        for offset in OFFSETS:
            mine = [v for v in VARIANTS if v.endswith(f":{offset}")]
            baseline = f"offset:{offset}"
            if all(s in walk.errors for s in (*mine, baseline)):
                continue
            try:
                off_train, off_test = _offsets(offset, rows, q, fold)
            except Exception as exc:
                if strict:
                    raise
                for scorer in (*mine, baseline):
                    walk.fail(scorer, f"fold {k}, {offset} offset: {exc!r}")
                continue
            p_offset = race_probs(off_test, None, (), test_race)
            if baseline not in walk.errors:
                frame = pd.DataFrame({"fold": k, "pos": fold.test_idx, "p_offset": p_offset})
                walk.probs.setdefault(baseline, []).append(frame.assign(p_model=p_offset))
            for variant in mine:
                if variant in walk.errors:
                    continue
                features = variant.split(":")[0]
                cols = list(FEATURES[features])
                try:
                    train = rows[["race_id", "won", "meeting_id", *cols]].iloc[fold.train_idx]
                    fit = fit_fold(train.assign(offset=off_train), features=cols)
                    X = design[features][fold.test_idx]
                    p_model = race_probs(off_test, X, fit.coef, test_race)
                except Exception as exc:
                    if strict:
                        raise
                    walk.fail(variant, f"fold {k}: {exc!r}")
                    continue
                frame = pd.DataFrame(
                    {"fold": k, "pos": fold.test_idx, "p_offset": p_offset, "p_model": p_model}
                )
                walk.probs.setdefault(variant, []).append(frame)
                walk.fits.setdefault(variant, []).append((k, fit))
    return walk


def _predictions(rows: pd.DataFrame, walk: _Walk, scorers: Sequence[str]) -> pd.DataFrame:
    race = rows["race_id"].to_numpy(dtype=object)
    horse = rows["horse_no"].to_numpy()
    parts = [
        pd.DataFrame(
            {
                "variant": scorer,
                "fold": frame["fold"].to_numpy(),
                "race_id": race[frame["pos"].to_numpy()],
                "horse_no": horse[frame["pos"].to_numpy()],
                "p_offset": frame["p_offset"].to_numpy(),
                "p_model": frame["p_model"].to_numpy(),
            }
        )
        for scorer in scorers
        for frame in walk.probs.get(scorer, [])
    ]
    columns = ["variant", "fold", "race_id", "horse_no", "p_offset", "p_model"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


SCORE_COLUMNS = ["variant", "fold", "race_id", "race_date", "meeting_id", "class_group"]
SCORE_COLUMNS += ["ll_offset", "ll_model", "delta_ll", "coef_x_cross", "coef_fav_x_cross"]


def _scores(rows: pd.DataFrame, walk: _Walk) -> pd.DataFrame:
    """One row per (variant, scored race with a winner)."""
    won = rows["won"].to_numpy(dtype=bool)
    race = rows["race_id"].to_numpy(dtype=object)
    meta = rows.drop_duplicates("race_id").set_index("race_id")
    parts = []
    for variant in VARIANTS:
        cols = FEATURES[variant.split(":")[0]]
        coefs = {k: dict(zip(cols, fit.coef, strict=True)) for k, fit in walk.fits.get(variant, [])}
        for frame in walk.probs.get(variant, []):
            pos = frame["pos"].to_numpy()
            w = won[pos]
            if not w.any():
                continue
            k = int(frame["fold"].iloc[0])
            ll = (
                pd.DataFrame(
                    {
                        "race_id": race[pos][w],
                        "ll_offset": -np.log(frame["p_offset"].to_numpy()[w]),
                        "ll_model": -np.log(frame["p_model"].to_numpy()[w]),
                    }
                )
                .groupby("race_id", sort=True)[["ll_offset", "ll_model"]]
                .mean()
            )
            about = meta.loc[ll.index, ["race_date", "meeting_id", "class_group"]]
            parts.append(
                pd.DataFrame(
                    {
                        "variant": variant,
                        "fold": k,
                        "race_id": ll.index.to_numpy(dtype=object),
                        "race_date": about["race_date"].to_numpy(),
                        "meeting_id": about["meeting_id"].to_numpy(),
                        "class_group": about["class_group"].to_numpy(dtype=object),
                        "ll_offset": ll["ll_offset"].to_numpy(),
                        "ll_model": ll["ll_model"].to_numpy(),
                        "delta_ll": (ll["ll_offset"] - ll["ll_model"]).to_numpy(),
                        "coef_x_cross": coefs[k].get("x_cross", np.nan),
                        "coef_fav_x_cross": coefs[k].get("fav_x_cross", np.nan),
                    }
                )
            )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SCORE_COLUMNS)


def predict_folds(rows: pd.DataFrame, folds: Sequence[Fold]) -> pd.DataFrame:
    """One row per (variant, test-fold runner): `variant, fold, race_id, horse_no, p_offset,
    p_model`, out of fold (see the module doc)."""
    return _predictions(rows, _walk(rows, folds, strict=True), VARIANTS)


def score_folds(rows: pd.DataFrame, folds: Sequence[Fold]) -> pd.DataFrame:
    """One row per (variant, scored race with a winner): `variant, fold, race_id, race_date,
    meeting_id, class_group, ll_offset, ll_model, delta_ll` and the fold's `coef_x_cross`,
    `coef_fav_x_cross` (NaN where the variant lacks the feature)."""
    return _scores(rows, _walk(rows, folds, strict=True))


# --------------------------------------------------------------------------- settlement


def settle(bets: pd.DataFrame, dividends: pd.DataFrame, *, pool: str) -> pd.Series:
    """The return per unit stake of each bet (`race_id`, `combination`) at the final dividend:
    dividend / 100 when `dividends` has that (race_id, pool, combination), else 0."""
    paid = dividends.loc[(dividends["pool"] == pool).fillna(False).to_numpy(dtype=bool)]
    paid = pd.DataFrame(
        {
            "race_id": paid["race_id"].astype(object).to_numpy(),
            "combination": paid["combination"].astype(object).to_numpy(),
            "dividend": paid["dividend"].to_numpy(dtype=float, na_value=np.nan),
        }
    ).drop_duplicates(["race_id", "combination"], keep="first")
    keys = pd.DataFrame(
        {
            "race_id": bets["race_id"].astype(object).to_numpy(),
            "combination": bets["combination"].astype(object).to_numpy(),
        }
    )
    got = keys.merge(paid, on=["race_id", "combination"], how="left", sort=False)
    values = np.nan_to_num(got["dividend"].to_numpy(dtype=float), nan=0.0) / 100.0
    return pd.Series(values, index=bets.index, name="return")


def _pair_probs(pool: str, P: np.ndarray, code, a, b) -> np.ndarray:
    """Each combination's probability from the per-runner Win probabilities P[race, horse_no],
    by Harville (lambda2 = 1): Win p_a; Quinella P({a, b} first two); Forecast P(a, then b).
    NaN where a leg is not a scored starter."""

    def leg(horse: np.ndarray) -> np.ndarray:
        inside = (horse >= 0) & (horse < P.shape[1])
        return np.where(inside, P[code, np.where(inside, horse, 0)], np.nan)

    pa_ = leg(a)
    if pool == "WIN":
        return pa_
    pb_ = leg(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        ab = pa_ * pb_ / (1.0 - pa_)
        if pool == "FORECAST":
            return ab
        return ab + pb_ * pa_ / (1.0 - pb_)


@dataclass
class _Races:
    """The scored races, indexed by code: meeting, day and class codes and the year."""

    index: pd.Index
    meeting: np.ndarray
    day: np.ndarray
    group: np.ndarray  # index into GROUPS, -1 unmapped
    year: np.ndarray


def _scored_races(rows: pd.DataFrame, folds: Sequence[Fold]) -> _Races:
    pos = np.concatenate([f.test_idx for f in folds] or [np.zeros(0, dtype=int)])
    meta = rows.iloc[np.sort(pos)].drop_duplicates("race_id")
    group = meta["class_group"].map({g: i for i, g in enumerate(GROUPS)})
    return _Races(
        index=pd.Index(meta["race_id"].to_numpy(dtype=object)),
        meeting=pd.factorize(meta["meeting_id"].to_numpy(dtype=object))[0].astype(np.int32),
        day=pd.factorize(meta["race_date"].to_numpy(dtype=object))[0].astype(np.int32),
        group=group.fillna(-1).to_numpy(dtype=np.int8),
        year=np.array([d.year for d in meta["race_date"]], dtype=np.int64),
    )


def _final_odds(pool: str, start: date, end: date, race_ids: Sequence[str]) -> pa.Table:
    """The priced final combinations (`race_id, combination, odds`) of `pool` in `race_ids` with
    `start <= race_date <= end`. A year of `final_odds` is one row group holding every pool, so
    the row groups are picked by their race_date statistics and decoded in batches, each filtered
    before the next is read; `race_ids` are development races, and a kept row on a holdout day
    is refused (D14)."""
    if start < SPAN_START:
        raise PreReformPriceError(f"a final_odds read may not start before {SPAN_START} (D10)")
    columns = ["race_id", "combination", "odds"]
    schema = pa.schema(
        [("race_id", pa.string()), ("combination", pa.string()), ("odds", pa.float64())]
    )
    if not len(race_ids):
        return schema.empty_table()
    source = pq.ParquetFile(paths.processed_dir(SOURCE) / "final_odds.parquet")
    meta = source.metadata
    day_col = source.schema_arrow.names.index("race_date")
    groups = []
    for k in range(meta.num_row_groups):
        stats = meta.row_group(k).column(day_col).statistics
        if stats is None or not stats.has_min_max or (stats.min <= end and stats.max >= start):
            groups.append(k)
    wanted = pa.array(list(race_ids), pa.string())
    parts = []
    for batch in source.iter_batches(
        batch_size=65_536,
        row_groups=groups,
        columns=["race_id", "race_date", "pool", "combination", "odds"],
        use_threads=False,
    ):
        keep = pc.and_kleene(pc.equal(batch["pool"], pool), pc.greater(batch["odds"], 0.0))
        keep = pc.and_kleene(keep, pc.not_equal(batch["combination"], POOL_ROW))
        keep = pc.and_kleene(keep, pc.greater_equal(batch["race_date"], pa.scalar(start)))
        keep = pc.and_kleene(keep, pc.less_equal(batch["race_date"], pa.scalar(end)))
        keep = pc.and_kleene(keep, pc.is_in(batch["race_id"], value_set=wanted))
        kept = batch.filter(pc.fill_null(keep, False))
        held = [d for d in pc.unique(kept["race_date"]).to_pylist() if is_holdout_day(COUNTRY, d)]
        if held:
            raise RuntimeError(f"a holdout day's final odds reached the read: {held[0]} (D14)")
        parts.append(kept.select(columns))
    del source
    pa.default_memory_pool().release_unused()
    return pa.Table.from_batches(parts, schema=schema) if parts else schema.empty_table()


def _pool_bets(
    pool: str,
    races: _Races,
    paid: pd.DataFrame,
    date_from: date,
    date_to: date,
    counts: dict[str, int],
) -> dict[str, np.ndarray]:
    """Every priced final combination of `pool` in the scored races, read a year at a time and
    settled at the final dividend, as flat arrays (race code, legs, odds, return)."""
    paid_races = set(paid.loc[paid["pool"] == pool, "race_id"])
    keep = np.array([r in paid_races for r in races.index], dtype=bool)
    counts[f"{pool.lower()}_races_without_dividend"] = int((~keep).sum())
    chunks = []
    for year in sorted(set(races.year[keep].tolist())):
        ids = races.index[keep & (races.year == year)].tolist()
        start, end = max(date_from, date(year, 1, 1)), min(date_to, date(year, 12, 31))
        table = _final_odds(pool, start, end, ids)
        legs = pc.split_pattern(table["combination"], ",")
        a = pc.cast(pc.list_element(legs, 0), pa.int64()).to_numpy()
        b = (
            pc.cast(pc.list_element(legs, 1), pa.int64()).to_numpy()
            if pool != "WIN"
            else np.full(len(a), -1)
        )
        frame = table.select(["race_id", "combination"]).to_pandas()
        code = races.index.get_indexer(frame["race_id"].to_numpy(dtype=object))
        ok = code >= 0
        chunks.append(
            {
                "code": code[ok].astype(np.int32),
                "a": a[ok].astype(np.int16),
                "b": b[ok].astype(np.int16),
                "odds": table["odds"].to_numpy()[ok],
                "ret": settle(frame, paid, pool=pool).to_numpy()[ok],
            }
        )
        del table, frame, legs
        pa.default_memory_pool().release_unused()
    if not chunks:
        return {k: np.zeros(0) for k in ("code", "a", "b", "odds", "ret")}
    return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}


def _prob_matrix(rows: pd.DataFrame, races: _Races, frames: list[pd.DataFrame]) -> np.ndarray:
    """The scorer's Win probabilities as P[race code, horse_no]; NaN where no runner."""
    width = int(rows["horse_no"].max()) + 1 if len(rows) else 1
    P = np.full((len(races.index), width), np.nan)
    race = rows["race_id"].to_numpy(dtype=object)
    horse = rows["horse_no"].to_numpy(dtype=np.int64)
    for frame in frames:
        pos = frame["pos"].to_numpy()
        code = races.index.get_indexer(race[pos])
        P[code, horse[pos]] = frame["p_model"].to_numpy()
    return P


def _z(level: float) -> float:
    return NormalDist().inv_cdf(level)


def _stats(idx, bets, races: _Races, boot: dict) -> dict:
    """One selection's return per unit stake with D7's one-sided meeting bootstrap bounds."""
    n = len(idx)
    out = {"n_bets": n, "n_races": 0, "n_race_days": 0, "n_meetings": 0}
    out |= dict.fromkeys(("mean_return", "lower", "upper", "se_mean_return", "mean_odds"), np.nan)
    if n == 0:
        return out | {"mean_claim": np.nan}
    code = bets["code"][idx]
    ret = bets["ret"][idx]
    iv = paired_bootstrap(
        pd.Series(ret),
        pd.Series(races.meeting[code]),
        n_boot=boot["n_boot"],
        level=boot["level"],
        seed=boot["seed"],
    )
    return out | {
        "n_races": len(np.unique(code)),
        "n_race_days": len(np.unique(races.day[code])),
        "n_meetings": iv.n_clusters,
        "mean_return": float(ret.mean()),
        "lower": iv.lo,
        "upper": iv.hi,
        "se_mean_return": (iv.hi - iv.lo) / (2.0 * _z(boot["level"])),
        "mean_odds": float(bets["odds"][idx].mean()),
        "mean_claim": float(bets["claim"][idx].mean()),
    }


def _num(value) -> float | None:
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _cell(cell_id: str, pool: str, group: str, odds_band: str | None, rate: float, s: dict):
    """A candidate cell in #430's schema; an empty cell has no return."""
    return {
        "study": STUDY,
        "family_id": FAMILY_ID,
        "cell_id": cell_id,
        "pool": pool,
        "class_group": None if group == ALL else group,
        "odds_band": odds_band,
        "payout_rate": rate,
        "n_bets": int(s["n_bets"]),
        "n_race_days": int(s["n_race_days"]),
        "n_meetings": int(s["n_meetings"]),
        "mean_return": _num(s["mean_return"]),
        "se_mean_return": _num(s["se_mean_return"]),
        "shrunk_edge": None,
        "mean_odds": _num(s["mean_odds"]),
    }


def _segments(bets: dict, groups: np.ndarray) -> dict[str, np.ndarray]:
    every = np.arange(len(bets["code"]), dtype=np.int32)
    return {ALL: every, **{g: every[groups == i] for i, g in enumerate(GROUPS)}}


def _settle_scorer(scorer, pool, bets, races, rate, boot):
    """One scorer in one pool: its selections' and N5 bands' tables and cells."""
    groups = races.group[bets["code"]]
    segments = _segments(bets, groups)
    table, cells = [], []
    order = np.lexsort((-bets["claim"], bets["code"]))
    first = order[np.r_[True, bets["code"][order][1:] != bets["code"][order][:-1]]]
    del order
    chosen = {"value": bets["claim"] > 1.0, "top1": np.zeros(len(bets["code"]), dtype=bool)}
    chosen["top1"][first] = True
    for selection in SELECTIONS:
        for group, idx in segments.items():
            s = _stats(idx[chosen[selection][idx]], bets, races, boot)
            label = {"variant": scorer, "pool": pool, "selection": selection, "class_group": group}
            table.append(label | s)
            cells.append(_cell(f"{scorer}/{pool}/{selection}/{group}", pool, group, None, rate, s))
    frame = pd.DataFrame(
        {
            "claim": bets["claim"],
            "ret": bets["ret"],
            "meeting": races.meeting[bets["code"]],
            "class_group": pd.Categorical.from_codes(groups, categories=list(GROUPS)),
        }
    )
    tail = tail_curve(
        frame,
        claim_col="claim",
        return_col="ret",
        cluster_col="meeting",
        quantiles=N5_QUANTILES,
        by="class_group",
        n_boot=boot["n_boot"],
        level=boot["level"],
        seed=boot["seed"],
    )
    del frame, chosen
    z = _z(boot["level"])
    for segment, part in tail.groupby("segment", sort=False):
        idx = segments[segment]
        part = part.sort_values("band")
        edges = np.r_[part["claim_lo"].to_numpy(float), part["claim_hi"].to_numpy(float)[-1:]]
        band_of = np.searchsorted(edges[1:-1], bets["claim"][idx], side="right")
        for row in part.itertuples(index=False):
            member = idx[band_of == row.band]
            code = bets["code"][member]
            s = {
                "n_bets": len(member),
                "n_race_days": len(np.unique(races.day[code])),
                "n_meetings": len(np.unique(races.meeting[code])),
                "mean_return": row.realised,
                "se_mean_return": (row.hi - row.lo) / (2.0 * z),
                "mean_odds": bets["odds"][member].mean() if len(member) else np.nan,
            }
            band = f"claim q{row.q_lo:g}-{row.q_hi:g}"
            cells.append(
                _cell(f"{scorer}/{pool}/n5/{segment}/{band}", pool, segment, band, rate, s)
            )
    tail.insert(0, "pool", pool)
    tail.insert(0, "variant", scorer)
    return table, tail, cells


def _payout_rate(pool: str) -> float:
    """The pool's base payout rate (1 - takeout, jra.yaml) in force since 2014-06-07."""
    return round(1.0 - load_source(COUNTRY, "jra").pools[pool].takeout_on(SPAN_START), 6)


def _settle_all(rows, folds, walk, date_from, date_to, boot, counts, log):
    """Settle every scorer that did not fail, pool by pool."""
    races = _scored_races(rows, folds)
    scorers = [s for s in (*VARIANTS, *BASELINES) if s in walk.probs]
    tables, tails, cells = [], [], []
    if not scorers or not len(races.index):
        return tables, tails, cells
    paid = pq.read_table(
        paths.processed_dir(SOURCE) / "dividends.parquet",
        columns=["race_id", "pool", "combination", "dividend"],
        filters=[("pool", "in", list(POOLS)), ("race_id", "in", races.index.tolist())],
    ).to_pandas()
    for pool in POOLS:
        bets = _pool_bets(pool, races, paid, date_from, date_to, counts)
        rate = _payout_rate(pool)
        log(f"  {pool}: settling {len(scorers)} scorers (peak {_peak_gb():.2f} GB)")
        for scorer in scorers:
            if scorer in walk.errors:
                continue
            try:
                P = _prob_matrix(rows, races, walk.probs[scorer])
                p = _pair_probs(pool, P, bets["code"], bets["a"], bets["b"])
                del P
                ok = np.isfinite(p)
                counts[f"{pool.lower()}_combinations_with_an_unmodelled_leg"] = int((~ok).sum())
                claim = p * bets["odds"]
                del p
                if ok.all():  # the usual case: no copy of the pool's arrays
                    mine = {**bets, "claim": claim}
                else:
                    mine = {k: v[ok] for k, v in bets.items()} | {"claim": claim[ok]}
                del claim
                if not len(mine["code"]):
                    raise ValueError(f"no priced {pool} combination to settle")
                table, tail, got = _settle_scorer(scorer, pool, mine, races, rate, boot)
            except Exception as exc:
                walk.fail(scorer, f"settlement {pool}: {exc!r}")
                continue
            tables += table
            tails.append(tail)
            cells += got
        del bets
    return tables, tails, cells


# --------------------------------------------------------------------------- the spec


def _bootstrap(seed: int) -> dict:
    boot = load_country_yaml(COUNTRY, "scoring")["bootstrap"]
    if boot["cluster_unit"] != "meeting" or boot["sided"] != "one-sided":
        raise ValueError(f"this study implements meeting clusters, one-sided; scoring.yaml: {boot}")
    return {
        "cluster_unit": "meeting",
        "sided": "one-sided",
        "level": float(boot["level"]),
        "n_boot": int(boot["n_boot"]),
        "seed": int(seed),
    }


def _check_span(date_from: date, date_to: date) -> None:
    if date_from < SPAN_START:
        raise PreReformPriceError(
            f"the span may not start before {SPAN_START.isoformat()} (D10), got {date_from}"
        )
    if date_to < date_from:
        raise ValueError(f"the span ends ({date_to}) before it starts ({date_from})")


def build_spec(*, date_from: date, date_to: date, seed: int) -> dict:
    """The family's pre-registered spec (docs/specs/cross-pool-prereg.md), plain JSON."""
    _check_span(date_from, date_to)
    scheme = load_country_scheme(COUNTRY)
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "status": "pre-registered by Claude under the owner's reframe of 2026-11-27 (#435): no "
        "sign-off stop; frozen in the registry (--freeze) before any scored read",
        "hypothesis": {
            HYPOTHESIS: "at t, the Quinella pool's top-2 support against the Win-implied "
            "(Harville) top-2 support, x_cross, adds to the final Win price: beta != 0 in u = "
            "offset + beta . x, and walk-forward ΔLL > 0 against the offset",
        },
        "null": "beta = 0 and ΔLL = 0: the final Win price already holds the cross-pool signal "
        "(the starting point)",
        "target": "the race's winner (Win pool) at the final Win odds",
        "decision_time": {
            "minutes_before_post": MINUTES_BEFORE,
            "rule": "D2's primary (config/decision.yaml primary_min); the boards are the latest "
            "rows per combination with minutes_to_post >= minutes_before_post (#434 boards_at); "
            "no row after t is read",
        },
        "features": {
            "x_cross": "ln(q_top2_t / h_top2_t): the runner's Quinella top-2 support at t over "
            "its plain-Harville top-2 support from the Win shares at t (#434 cross_features)",
            "fav_x_cross": "is_fav . x_cross, is_fav the T-2 Win favourite (the lowest horse "
            "number on a tie)",
            "lambda2": LAMBDA2,
            "zero_vote": f"{ZERO_VOTE}: a zero-vote Quinella quote reads as missing (#434)",
        },
        "feature_sets": {name: list(cols) for name, cols in FEATURES.items()},
        "offsets": {
            "normalised": "ln q, q = (1/O) / sum (1/O) over the race's priced starters, O the "
            "final Win odds (final_odds WIN)",
            "calibrated": "ln of D4's calibrated market: models.v0.calibration."
            f"fit_win_calibration ({N_KNOTS} knots, natural cubic spline in ln q) fitted on "
            "each fold's training rows and applied to its training and test rows",
            "ranking": "both offsets are registered variants and each is ranked",
        },
        "variants": list(VARIANTS),
        "family_size": len(VARIANTS),
        "fit": "conditional logit u = offset + X @ beta within race, unpenalised MLE "
        "(models.condlogit.fit_conditional_logit), a dead heat's winners at 1/k, a race with no "
        "winner dropped; se_cluster the meeting sandwich; 95% two-sided Wald interval",
        "folds": {
            "scheme": dataclasses.asdict(scheme),
            "rule": "core.eval.folds.walk_forward over the study rows' race dates (D9): each "
            "fold's beta and spline fitted on its training rows only, the embargo counted back "
            "from the fold's first scored day; test races scored out of fold",
        },
        "span": {
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
            "price_era_start": SPAN_START.isoformat(),
            "D10": "no row before 2014-06-07",
            "D14": "development days only: a holdout day's file is never opened (dropped by its "
            "name by core.eval.holdout.is_holdout_day); LockedHoldout is never opened",
        },
        "exclusions": [
            "a race with no row in the jravan races table",
            "a race with fewer than 2 priced final Win starters",
            "a race with no T-2 Quinella board",
            "a race in which a priced final starter has no T-2 row or a NaN feature",
            "a race with no winner (void)",
            "a runner without a final Win price, or marked scratched, is dropped and the race "
            "renormalised",
            "an unmapped race class counts in 'all' only",
            "a race whose pool has no dividend row is left out of that pool's settlement",
            "every exclusion is counted in the private report",
        ],
        "metric": {
            "delta_ll": "per scored race ll_offset - ll_model, winner log loss, a dead heat's "
            "winners averaged (D6); positive: the model helps",
            "bounds": "D7 meeting bootstrap, one-sided 95% (lower and upper)",
            "segments": ["all", *GROUPS],
            "class_groups": dict(CLASS_GROUPS),
            "coefficients": "per fold, and pooled over the scored rows with their out-of-fold "
            "offset; meeting-clustered sandwich se and 95% interval",
        },
        "bootstrap": _bootstrap(seed),
        "settlement": {
            "pools": list(POOLS),
            "scorers": [*VARIANTS, *BASELINES],
            "baselines": "each offset alone (beta = 0)",
            "probabilities": "Win p from the scorer; Quinella P({a, b}) = p_a p_b / (1 - p_a) + "
            "p_b p_a / (1 - p_b); FORECAST (Exacta) P(a, b) = p_a p_b / (1 - p_a) (Harville, "
            "lambda2 = 1)",
            "claim": "p x the pool's final odds (final_odds), over every priced combination "
            "both of whose legs are scored starters",
            "selections": {
                "value": "every priced combination with claim > 1",
                "top1": "per race the combination with the largest claim",
            },
            "return": "dividend / 100 at the final dividend (dividends), else 0; a dead heat's "
            "paid combinations all pay; never the board's odds",
            "bounds": "D7 meeting bootstrap, one-sided 95%; se_mean_return = (upper - lower) / "
            "(2 z_level)",
            "segments": ["all", *GROUPS],
        },
        "n5": {
            "function": "core.eval.tail.tail_curve",
            "quantiles": list(N5_QUANTILES),
            "claim": "the scorer's claim over every priced combination",
            "by": "class_group",
            "cluster": "meeting",
        },
        "candidate_cells": {
            "file": "reports/candidate-cells/cross-pool.json under the data dir",
            "schema": [
                "study",
                "family_id",
                "cell_id",
                "pool",
                "class_group",
                "odds_band",
                "payout_rate",
                "n_bets",
                "n_race_days",
                "n_meetings",
                "mean_return",
                "se_mean_return",
                "shrunk_edge",
                "mean_odds",
            ],
            "cells": "every settled selection (scorer x pool x selection x segment) and every "
            "N5 band (scorer x pool x segment x band); class_group null for 'all'; odds_band null "
            "for a selection, the claim-quantile band for an N5 band; shrunk_edge null; "
            "payout_rate the pool's base rate (jra.yaml)",
            "ranking": "mean_return descending, an empty cell last",
        },
        "adoption": "none: a measurement",
        "sources": [SOURCE],
        "seed": int(seed),
    }


def freeze_spec(
    *, date_from: date, date_to: date, seed: int, registry: Registry | None = None
) -> str:
    """Freeze the family's spec for these arguments in `registry` and return its hash."""
    spec = build_spec(date_from=date_from, date_to=date_to, seed=seed)
    registry = default_registry(COUNTRY) if registry is None else registry
    return registry.freeze_family(FAMILY_ID, spec)


# --------------------------------------------------------------------------- reading


def _peak_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**3 if sys.platform == "darwin" else 1024**2)


def _day_files(date_from: date, date_to: date) -> tuple[list[tuple[date, Path]], int]:
    """The development day files in the span; a holdout day is dropped by its name, before any
    stat or read (D14)."""
    folder = paths.processed_dir(SOURCE) / "odds_timeseries"
    files, held = [], 0
    for name in sorted(os.listdir(folder)) if folder.is_dir() else []:
        match = _DAY_FILE.fullmatch(name)
        if match is None:
            continue
        try:
            day = date.fromisoformat(match.group(1))
        except ValueError:
            continue
        if not date_from <= day <= date_to:
            continue
        if is_holdout_day(COUNTRY, day):
            held += 1
            continue
        files.append((day, folder / name))
    return files, held


def _read_day(path: Path) -> pd.DataFrame:
    """One day file: the board's columns, WIN and QUINELLA rows at or before t only."""
    table = pq.read_table(
        path,
        columns=DAY_COLUMNS,
        filters=[
            ("pool", "in", [WIN, QUINELLA]),
            ("combination", "!=", POOL_ROW),
            ("minutes_to_post", ">=", float(MINUTES_BEFORE)),
        ],
    )
    frame = table.to_pandas()
    del table
    pa.default_memory_pool().release_unused()
    return frame


def _builds(files: list[tuple[date, Path]]) -> dict[str, str]:
    digest = hashlib.sha256()
    for day, path in files:
        digest.update(f"{day.isoformat()}:{path.stat().st_size}\n".encode())
    folder = paths.processed_dir(SOURCE)
    builds = {"jravan_odds_timeseries": f"processed/{SOURCE}/odds_timeseries@"}
    builds["jravan_odds_timeseries"] += digest.hexdigest()[:16]
    for name in ("races", "runners", "final_odds", "dividends"):
        stat = (folder / f"{name}.parquet").stat()
        stamp = hashlib.sha256(f"{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]
        builds[f"jravan_{name}"] = f"processed/{SOURCE}/{name}.parquet@{stamp}"
    return builds


def _features(files, log) -> tuple[pd.DataFrame, float]:
    parts, seconds, started = [], 0.0, time.monotonic()
    for i, (day, path) in enumerate(files):
        tick = time.monotonic()
        frame = day_rows(_read_day(path))
        seconds += time.monotonic() - tick
        if len(frame):
            parts.append(frame.assign(file_day=day))
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            log(
                f"  {i + 1}/{len(files)} days read ({time.monotonic() - started:.0f} s, "
                f"peak {_peak_gb():.2f} GB)"
            )
    feats = pd.concat(parts, ignore_index=True) if parts else _no_rows().assign(file_day=None)
    return feats, seconds / max(len(files), 1)


def _study_rows(feats: pd.DataFrame, date_from: date, date_to: date) -> tuple[pd.DataFrame, dict]:
    """The study rows (one per priced final starter of every kept race) and the counts."""
    counts: dict[str, int] = {"races_with_a_t2_win_board": int(feats["race_id"].nunique())}
    ids = sorted(set(feats["race_id"]))
    where = ds.field("race_id").isin(ids) if ids else ds.scalar(False)
    races = read_store_table(
        "races",
        columns=["race_id", "meeting_id", "race_date", "race_class"],
        start=date_from,
        end=date_to,
        touches_price=False,
        source=SOURCE,
    ).to_pandas()
    days = set(feats["file_day"])
    counts["races_without_a_t2_win_board"] = int(
        (races["race_date"].isin(days) & ~races["race_id"].isin(set(ids))).sum()
    )
    races = races[races["race_id"].isin(set(ids))].drop_duplicates("race_id")
    counts["excluded_no_race_row"] = len(ids) - len(races)
    runners = read_store_table(
        "runners",
        columns=["race_id", "horse_no", "finish_pos", "is_scratched"],
        start=date_from,
        end=date_to,
        touches_price=False,
        where=where,
        source=SOURCE,
    ).to_pandas()
    runners = runners.drop_duplicates(["race_id", "horse_no"])
    win = []
    for year in sorted({d.year for d in races["race_date"]}):
        year_ids = races.loc[[d.year == year for d in races["race_date"]], "race_id"].tolist()
        start, end = max(date_from, date(year, 1, 1)), min(date_to, date(year, 12, 31))
        table = _final_odds("WIN", start, end, year_ids)
        win.append(
            pd.DataFrame(
                {
                    "race_id": table["race_id"].to_numpy(zero_copy_only=False),
                    "horse_no": pc.cast(table["combination"], pa.int64()).to_numpy(),
                    "win_odds": table["odds"].to_numpy(),
                }
            )
        )
        del table
    starters = (
        pd.concat(win, ignore_index=True)
        if win
        else pd.DataFrame({"race_id": [], "horse_no": [], "win_odds": []})
    )
    starters = starters.drop_duplicates(["race_id", "horse_no"]).astype({"horse_no": np.int64})
    starters = starters.merge(
        runners[["race_id", "horse_no", "finish_pos", "is_scratched"]],
        on=["race_id", "horse_no"],
        how="left",
    )
    scratched = starters["is_scratched"].fillna(False).to_numpy(dtype=bool)
    starters = starters[~scratched]
    counts["runners_scratched_with_a_final_price"] = int(scratched.sum())
    board = feats[["race_id", "horse_no", "x_cross", "fav_x_cross"]]
    frame = starters.merge(board.drop_duplicates(["race_id", "horse_no"]), how="left")
    frame["bad"] = frame[["x_cross", "fav_x_cross"]].isna().any(axis=1).to_numpy()
    frame["won"] = (frame["finish_pos"] == 1).fillna(False).to_numpy(dtype=bool)
    race_ids = races["race_id"]
    by_race = frame.groupby("race_id")
    few = by_race.size().reindex(race_ids, fill_value=0).to_numpy() < 2
    board_q = feats.groupby("race_id")["quinella_board"].any()
    no_q = ~few & ~board_q.reindex(race_ids, fill_value=False).to_numpy(dtype=bool)
    bad = by_race["bad"].any().reindex(race_ids, fill_value=True).to_numpy(dtype=bool)
    nan_feature = ~few & ~no_q & bad
    winner = by_race["won"].any().reindex(race_ids, fill_value=False).to_numpy(dtype=bool)
    void = ~few & ~no_q & ~nan_feature & ~winner
    counts["excluded_fewer_than_2_priced_starters"] = int(few.sum())
    counts["excluded_no_t2_quinella_board"] = int(no_q.sum())
    counts["excluded_nan_feature"] = int(nan_feature.sum())
    counts["excluded_void"] = int(void.sum())
    keep = set(race_ids[~(few | no_q | nan_feature | void)])
    on_board = feats[["race_id", "horse_no"]].merge(
        starters[["race_id", "horse_no"]], how="left", indicator=True
    )
    dropped = (on_board["_merge"] == "left_only") & on_board["race_id"].isin(keep)
    counts["runners_without_a_final_price"] = int(dropped.sum())
    groups = [CLASS_GROUPS.get(c) if isinstance(c, str) else None for c in races["race_class"]]
    races = races.assign(class_group=pd.Series(groups, index=races.index, dtype=object))
    rows = frame[frame["race_id"].isin(keep)].merge(
        races[["race_id", "race_date", "meeting_id", "class_group"]], on="race_id"
    )
    rows = rows.sort_values(["race_date", "race_id", "horse_no"], kind="mergesort")
    columns = ["race_id", "race_date", "meeting_id", "class_group", "horse_no", "won"]
    rows = rows[[*columns, "win_odds", "x_cross", "fav_x_cross"]].reset_index(drop=True)
    counts["races"] = int(rows["race_id"].nunique())
    counts["runners"] = len(rows)
    counts["race_days"] = int(rows["race_date"].nunique())
    counts["meetings"] = int(rows["meeting_id"].nunique())
    counts["races_unmapped_class"] = int(
        rows.drop_duplicates("race_id")["class_group"].isna().sum()
    )
    return rows, counts


# --------------------------------------------------------------------------- the result


@dataclass(frozen=True, eq=False)
class CrossPoolResult:
    family_id: str
    spec_hash: str
    family_size: int
    coefficients: pd.DataFrame
    gains: pd.DataFrame
    settlement: pd.DataFrame
    tail: pd.DataFrame
    cells: list[dict]
    counts: dict[str, int]
    report_path: Path
    cells_path: Path
    run_ids: list[str]
    failed_variants: list[str]
    date_from: date | None = None
    date_to: date | None = None
    seed: int | None = None


def _coef_rows(variant: str, label: str, fit: CLFit, **extra) -> list[dict]:
    cols = FEATURES[variant.split(":")[0]]
    se = fit.se_cluster if fit.se_cluster is not None else np.full(len(cols), np.nan)
    return [
        {"variant": variant, "feature": name, "fold": label, **extra}
        | {"coef": float(c), "se_cluster": float(s), "lower": c - Z95 * s, "upper": c + Z95 * s}
        | {"races": int(fit.n_races)}
        for name, c, s in zip(cols, fit.coef, se, strict=True)
    ]


def _gain(part: pd.DataFrame, boot: dict) -> dict:
    """The mean ΔLL of `part`'s races with D7's one-sided meeting bootstrap bounds."""
    out = {"races": len(part), "meetings": 0, "delta_ll_mean": np.nan}
    out |= {"lower_95_one_sided": np.nan, "upper_95_one_sided": np.nan}
    if not len(part):
        return out
    keyed = part.set_index("race_id")
    iv = paired_bootstrap(
        keyed["delta_ll"],
        keyed["meeting_id"],
        n_boot=boot["n_boot"],
        level=boot["level"],
        seed=boot["seed"],
    )
    return (
        out
        | {"meetings": iv.n_clusters, "delta_ll_mean": iv.mean}
        | {
            "lower_95_one_sided": iv.lo,
            "upper_95_one_sided": iv.hi,
        }
    )


def _fits(rows, folds, walk, scores, boot):
    """Per variant: the coefficients (per fold and pooled) and the gains by class group."""
    coefficients, gains = [], []
    for variant in VARIANTS:
        if variant in walk.errors:
            continue
        try:
            cols = list(FEATURES[variant.split(":")[0]])
            table = []
            for k, fit in walk.fits[variant]:
                table += _coef_rows(
                    variant, folds[k].test_start.isoformat(), fit, train_end=folds[k].train_end
                )
            pos = np.concatenate([f["pos"].to_numpy() for f in walk.probs[variant]])
            p_offset = np.concatenate([f["p_offset"].to_numpy() for f in walk.probs[variant]])
            scored = rows.iloc[pos][["race_id", "won", "meeting_id", *cols]]
            pooled = fit_fold(scored.assign(offset=np.log(p_offset)), features=cols)
            table += _coef_rows(variant, "pooled", pooled, train_end=None)
            mine = scores[scores["variant"] == variant]
            got = []
            for group in (ALL, *GROUPS):
                part = mine if group == ALL else mine[(mine["class_group"] == group).to_numpy()]
                got.append({"variant": variant, "class_group": group} | _gain(part, boot))
        except Exception as exc:
            walk.fail(variant, f"coefficients and gains: {exc!r}")
            continue
        coefficients += table
        gains += got
    return pd.DataFrame(coefficients), pd.DataFrame(gains)


def _metric(fold: str, name: str, value, n: int, lo=None, hi=None) -> FoldMetric | None:
    """A registry metric, or None when the value is not finite."""
    value, lo, hi = (None if v is None else _num(v) for v in (value, lo, hi))
    if value is None:
        return None
    return FoldMetric(str(fold), name, value, lo, hi, int(n))


def _variant_metrics(variant, coefficients, gains, scores, settlement) -> tuple[FoldMetric, ...]:
    out = []
    for row in coefficients[coefficients["variant"] == variant].itertuples(index=False):
        out.append(
            _metric(row.fold, f"coef[{row.feature}]", row.coef, row.races, row.lower, row.upper)
        )
    for row in gains[gains["variant"] == variant].itertuples(index=False):
        label = "all_folds" if row.class_group == ALL else f"class:{row.class_group}"
        out.append(
            _metric(
                label,
                "delta_ll",
                row.delta_ll_mean,
                row.races,
                row.lower_95_one_sided,
                row.upper_95_one_sided,
            )
        )
    mine = scores[scores["variant"] == variant]
    for k, part in mine.groupby("fold", sort=True):
        out.append(_metric(f"fold:{k}", "delta_ll", part["delta_ll"].mean(), len(part)))
    baseline = f"offset:{variant.split(':')[1]}"
    whole = settlement[settlement["class_group"] == ALL] if len(settlement) else settlement
    for scorer, prefix in ((variant, "return"), (baseline, "offset_return")):
        for row in whole[whole["variant"] == scorer].itertuples(index=False) if len(whole) else []:
            out.append(
                _metric(
                    "all_folds",
                    f"{prefix}[{row.pool}:{row.selection}]",
                    row.mean_return,
                    row.n_bets,
                    row.lower,
                    row.upper,
                )
            )
    return tuple(m for m in out if m is not None)


def _record(variant, spec, spec_hash, git_sha, builds, metrics, failed) -> RunRecord:
    features, offset = variant.split(":")
    return RunRecord(
        hypothesis_id=HYPOTHESIS,
        family_id=FAMILY_ID,
        variant=variant,
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=builds,
        sources=(SOURCE,),
        fold_scheme="walk_forward:" + json.dumps(spec["folds"]["scheme"], sort_keys=True),
        components={
            "features": ",".join(FEATURES[features]),
            "offset": offset,
            "fit": "models.condlogit.fit_conditional_logit",
            "features_from": "studies.cross_pool.cross_features at t",
        },
        params={
            "minutes_before": MINUTES_BEFORE,
            "lambda2": LAMBDA2,
            "zero_vote": ZERO_VOTE,
            "date_from": spec["span"]["date_from"],
            "date_to": spec["span"]["date_to"],
            "seed": spec["seed"],
            "spec_hash": spec_hash,
        },
        metrics=metrics,
        status="failed" if failed else "scored",
    )


def _append(registry: Registry, records: list[RunRecord]) -> list[str]:
    existing = registry.records(FAMILY_ID)
    return [registry.append(r) for r in records if r not in existing]


def run_cross_pool(
    *,
    date_from: date,
    date_to: date,
    out_dir: Path,
    seed: int,
    registry: Registry | None = None,
    log: Callable[[str], None] | None = None,
) -> CrossPoolResult:
    """Run the frozen family on the development days of the span (see the module doc).

    Raises `PreReformPriceError` for a span before 2014-06-07, `FamilyNotFrozen` unless the
    registry froze this spec, and `paths.DataDirInsideRepo` for an `out_dir` inside the
    repository, each before any store file is opened."""
    log = log or (lambda message: None)
    _check_span(date_from, date_to)
    registry = default_registry(COUNTRY) if registry is None else registry
    spec = build_spec(date_from=date_from, date_to=date_to, seed=seed)
    spec_hash = _spec_hash(spec)
    if registry.frozen_spec_hash(FAMILY_ID) != spec_hash:
        raise FamilyNotFrozen(
            f"family {FAMILY_ID!r} is not frozen with this spec ({spec_hash[:12]}): run --freeze "
            "with the same --from, --to and --seed first"
        )
    paths.outside_repo(Path(out_dir), what="out")
    out_dir = Path(out_dir)
    started = time.monotonic()
    git_sha = _git_sha()
    boot = spec["bootstrap"]
    files, held = _day_files(date_from, date_to)
    builds = _builds(files)
    try:
        log(f"reading {len(files)} development day files")
        feats, per_day = _features(files, log)
        rows, counts = _study_rows(feats, date_from, date_to)
        del feats
        counts = {"development_day_files": len(files), "holdout_day_files_skipped": held} | counts
        folds = walk_forward(rows["race_date"], scheme=load_country_scheme(COUNTRY))
        counts["folds"] = len(folds)
        log(f"fitting {len(folds)} folds (peak {_peak_gb():.2f} GB)")
        walk = _walk(rows, folds, strict=False)
        if not folds:
            for scorer in (*VARIANTS, *BASELINES):
                walk.fail(scorer, "the fold scheme gives no walk-forward fold on these dates")
        scores = _scores(rows, walk)
        coefficients, gains = _fits(rows, folds, walk, scores, boot)
        log(f"settling (peak {_peak_gb():.2f} GB)")
        tables, tails, cells = _settle_all(rows, folds, walk, date_from, date_to, boot, counts, log)
    except Exception:  # the run is a record too: every variant failed
        _append(
            registry, [_record(v, spec, spec_hash, git_sha, builds, (), True) for v in VARIANTS]
        )
        raise
    failed = [v for v in VARIANTS if v in walk.errors]

    def keep(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[~frame["variant"].isin(walk.errors)] if len(frame) else frame

    coefficients, gains = keep(coefficients), keep(gains)
    settlement = keep(pd.DataFrame(tables))
    tail = keep(pd.concat(tails, ignore_index=True) if tails else pd.DataFrame())
    cells = [c for c in cells if c["cell_id"].split("/")[0] not in walk.errors]
    cells.sort(key=lambda c: (c["mean_return"] is None, -(c["mean_return"] or 0.0)))
    scores = scores[~scores["variant"].isin(walk.errors)]
    records = [
        _record(
            v,
            spec,
            spec_hash,
            git_sha,
            builds,
            () if v in failed else _variant_metrics(v, coefficients, gains, scores, settlement),
            v in failed,
        )
        for v in VARIANTS
    ]
    run_ids = _append(registry, records)
    report_path = out_dir / f"cross-pool-japan-{spec_hash[:12]}.md"
    cells_path = out_dir / "candidate-cells" / "cross-pool.json"
    result = CrossPoolResult(
        family_id=FAMILY_ID,
        spec_hash=spec_hash,
        family_size=registry.family_size(FAMILY_ID),
        coefficients=coefficients.reset_index(drop=True),
        gains=gains.reset_index(drop=True),
        settlement=settlement.reset_index(drop=True),
        tail=tail.reset_index(drop=True),
        cells=cells,
        counts={k: int(v) for k, v in counts.items()},
        report_path=report_path,
        cells_path=cells_path,
        run_ids=run_ids,
        failed_variants=failed,
        date_from=date_from,
        date_to=date_to,
        seed=seed,
    )
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`, family size {result.family_size}",
        "git": git_sha,
        "store": ", ".join(builds.values()),
        "span": f"{date_from.isoformat()} to {date_to.isoformat()} (development days)",
        "run ids": ", ".join(r[:16] for r in run_ids) or "none appended",
        "wall time": f"{time.monotonic() - started:.0f} s; read per day file {per_day:.2f} s",
        "peak RSS": f"{_peak_gb():.2f} GB",
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_report(report_path, render_report(result, walk.errors, about), sources=(SOURCE,))
    text = json.dumps(cells, indent=1, allow_nan=False, ensure_ascii=False) + "\n"
    write_report(cells_path, text, sources=(SOURCE,))
    log(f"done: {report_path}")
    return result


# --------------------------------------------------------------------------- reports


def _cell_text(value) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return " ".join(str(value).split()).replace("|", "\\|")


def _table(frame: pd.DataFrame | list[dict]) -> str:
    frame = pd.DataFrame(frame)
    if not len(frame):
        return "_None: see the variants above._"
    head = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_cell_text(v) for v in row) + " |")
    return "\n".join(lines)


def render_report(result: CrossPoolResult, errors: dict[str, str], about: dict) -> str:
    """The private report: aggregate tables only, never a runner or race row."""
    variants = [
        {"variant": v, "status": "failed" if v in result.failed_variants else "scored"}
        | {"error": errors.get(v, "")}
        for v in VARIANTS
    ]
    baselines = [{"baseline": b, "error": errors.get(b, "")} for b in BASELINES]
    lines = [
        f"# E25: the T-2 cross-pool signal at the final Win price ({COUNTRY}, #{TICKET})",
        "",
        *(f"- {key}: {value}" for key, value in about.items()),
        "- development days only: holdout day files are never opened; the holdout is not opened",
        "- adoption: none (a measurement); every figure here is withheld from the repo (D21)",
        "",
        "## Variants (one registry record each)",
        "",
        _table(variants),
        "",
        _table(baselines),
        "",
        "## Counts",
        "",
        _table([{"count": k, "n": v} for k, v in result.counts.items()]),
        "",
        "## Coefficients (per fold, and pooled over the scored rows; meeting-clustered 95% CI)",
        "",
        _table(result.coefficients),
        "",
        "## Walk-forward ΔLL over the offset (nats per race; D7 one-sided 95% bounds)",
        "",
        _table(result.gains),
        "",
        "## Settlement at the final dividend (return per unit stake)",
        "",
        _table(result.settlement),
        "",
        "## N5: realised return by claim band (core.eval.tail)",
        "",
        _table(result.tail),
        "",
        "## Candidate cells, ranked by mean return (the first 60; all in the cell file)",
        "",
        _table(result.cells[:60]),
        "",
    ]
    return "\n".join(lines)


def _data_relative(path: Path) -> str:
    """`path` relative to the data dir as `$HORSETRACK_DATA_DIR/...`, else its name."""
    try:
        root = paths.data_dir()
    except Exception:
        return Path(path).name
    for candidate in (Path(path), Path(path).resolve()):
        try:
            return f"$HORSETRACK_DATA_DIR/{candidate.relative_to(root).as_posix()}"
        except ValueError:
            continue
    return f"{Path(path).name} (outside the data dir)"


def render_public_report(result: CrossPoolResult) -> str:
    """The public page: what ran and where its figures live; every figure withheld (D21)."""
    withheld = "withheld (D21)"
    failed = set(result.failed_variants)
    run_ids = ", ".join(f"`{r[:16]}`" for r in result.run_ids) or "none appended"
    span = ""
    if result.date_from is not None and result.date_to is not None:
        span = f" --from {result.date_from.isoformat()} --to {result.date_to.isoformat()}"
    seed = "" if result.seed is None else f" --seed {result.seed}"
    rows = [
        f"| `{v}` | {', '.join(FEATURES[v.split(':')[0]])} | {v.split(':')[1]} | "
        f"{'failed' if v in failed else 'scored'} | {withheld} | {withheld} |"
        for v in VARIANTS
    ]
    settled = [
        f"| `{s}` | {pool} | {sel} | {withheld} |"
        for s in (*VARIANTS, *BASELINES)
        for pool in POOLS
        for sel in SELECTIONS
    ]
    lines = [
        f"# Experiment report: E25, the T-2 cross-pool signal at the final Win price (Japan, "
        f"#{TICKET})",
        "",
        f"Ticket #{TICKET} (epic #73, N4). **Nothing is adopted.** **Every figure is withheld "
        "(D21).** The study reads the JRA-VAN time-series boards and the jravan final odds, "
        "results and dividends, a licensed source, so this page says what ran and where its "
        "figures live. It gives no count, span, coefficient, gain, return, rate, sign or bound "
        "derived from that data.",
        "",
        f"- Registry family `{result.family_id}`, spec sha256 `{result.spec_hash}`, frozen in "
        "the registry (`--freeze`) before any scored read. The spec is "
        "`docs/specs/cross-pool-prereg.md` (PRE-REGISTERED).",
        f"- Family size: {result.family_size} pre-registered fit variants. One registry record "
        f"each: {run_ids}. Failed: {', '.join(f'`{v}`' for v in sorted(failed)) or 'none'}.",
        f"- Private report: `{_data_relative(result.report_path)}`. Candidate cells (every "
        f"settled selection and N5 band, ranked): `{_data_relative(result.cells_path)}`.",
        "- Code: `countries/japan/studies/cross_pool_run.py`. Reproduce with `uv run python -m "
        f"horsetrack.countries.japan.studies.cross_pool_run --run{span}{seed}`.",
        "- Holdout: development days only. A holdout day's file is dropped by its name and never "
        "opened (D14); `LockedHoldout` is not opened; nothing before 2014-06-07 enters (D10).",
        "",
        "## Hypothesis",
        "",
        "E25: at t = post - 2 min (D2's primary), the Quinella pool's top-2 support against the "
        "Win-implied (Harville) top-2 support, x_cross, adds to the final Win price: beta != 0 "
        "in u = offset + beta . x, and the walk-forward winner log-loss gain over the offset is "
        "positive. **Null:** beta = 0 and no gain, the starting point.",
        "",
        "## Variants tried",
        "",
        "| Variant | Features | Offset | Status | Coefficient (95% CI) | ΔLL (one-sided bounds) |",
        "| --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        f"The gain by class group (newcomer, maiden, conditions, open, graded): {withheld}.",
        "",
        "## Settlement at the final dividend",
        "",
        "Each variant and each offset alone (beta = 0) is settled in WIN, QUINELLA and FORECAST "
        "(Exacta); `value` bets every priced combination with claim p x O > 1, `top1` the "
        "race's largest claim.",
        "",
        "| Scorer | Pool | Selection | Return per unit stake (bounds) |",
        "| --- | --- | --- | --- |",
        *settled,
        "",
        f"N5 tail-return curves per scorer and pool, by class group: {withheld}. The ranked "
        f"candidate cells: {withheld}. Counts and exclusions: {withheld}.",
        "",
        "## What the result rules out",
        "",
        f"{withheld}: the reading rests on the bounds in the private report.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m horsetrack.countries.japan.studies.cross_pool_run"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true", help="freeze the spec (before the run)")
    mode.add_argument("--run", action="store_true", help="run the frozen family")
    parser.add_argument(
        "--from",
        dest="date_from",
        type=date.fromisoformat,
        default=SPAN_START,
        help="first day (default 2014-06-07)",
    )
    parser.add_argument(
        "--to", dest="date_to", type=date.fromisoformat, default=None, help="last day (today)"
    )
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    parser.add_argument("--seed", type=int, default=TICKET, help=f"bootstrap seed ({TICKET})")
    args = parser.parse_args(argv)
    date_to = date.today() if args.date_to is None else args.date_to
    registry = default_registry(COUNTRY)
    span = {"date_from": args.date_from, "date_to": date_to, "seed": args.seed}
    if args.freeze:
        spec_hash = freeze_spec(**span, registry=registry)
        print(json.dumps({"family_id": FAMILY_ID, "spec_hash": spec_hash}, indent=2))
        return 0
    out_dir = paths.reports_dir() if args.out is None else args.out
    result = run_cross_pool(
        **span,
        out_dir=out_dir,
        registry=registry,
        log=lambda message: print(message, file=sys.stderr),
    )
    write_summary(registry, summary_dir(COUNTRY))
    public = env.repo_root() / "docs" / "experiments" / "cross-pool-japan.md"
    write_report(public, render_public_report(result), sources=())
    print(
        json.dumps(
            {
                "family_id": result.family_id,
                "spec_hash": result.spec_hash,
                "report_path": str(result.report_path),
                "cells_path": str(result.cells_path),
                "run_ids": [r[:16] for r in result.run_ids],
                "failed_variants": result.failed_variants,
                "public_report": str(public),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
