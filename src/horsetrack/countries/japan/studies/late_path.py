"""The late-money path: E23 (ex post), E24 (ex ante) and E22 (#102, epic #73; 08 §8 B9).

The paper is Hanyu, Ishii, Otani & Teramoto, arXiv 2509.14645: horses whose odds shorten late
return more at the same final odds. Everything here is pure but `path_files`'s directory listing;
the study that reads the data is `studies.late_path_study` (`horsetrack japan late-path-study`).

E23, ex post
- `late_change(captures, close_utc=, window_min=)`: per horse, O_to / O_from - 1, O_from its last
  capture at or before close - window (inclusive), O_to its last capture strictly before the
  close; a capture at or after the close is the final board and never enters. NaN when either end
  is missing, 0 when both ends are the same capture.
- `paper_regression(frame)`: OLS of won * odds_final on [1, odds_final, late_change] (the paper's
  base spec, the final odds linearly); a runner without a late change is left out. `se` is the
  cluster sandwich (CR0, no small-sample factor) over races, or over `clusters` when given.

The final-price offset
- `offset_probabilities(frame, coef=)`: softmax over each race of ln q_final + sum_c coef_c * x_c.
  q_final is any positive per-race multiple of the baseline's probability: the study passes D4's
  calibrated market, and this module never chooses a de-vig.
- `fit_offset`: the path coefficients by conditional-logit maximum likelihood with ln q_final as an
  offset at coefficient 1 (`models.condlogit`; a dead heat at 1/k, a race without a winner out).
- `offset_test`: the per-race winner log-loss gain in nats of final + path over the final alone
  (`core.eval.scoring.gain`'s sign: positive when the path helps), with `paired_bootstrap` bounds
  resampling `clusters`. Without `folds` the path coefficients are fitted on the frame itself (in
  sample). With `folds` (race_id -> an ordered label) each label's races are scored with the
  coefficients fitted on the races of the earlier labels only (walk-forward, expanding); the first
  label's races only train.

E24, information at t
- `ex_ante_features(frame, at_utc=, nowcast=)`: `frame` is one race's rows in
  `logger.compact.SCHEMA`; only WIN rows whose snapshot_utc is at or before t are read (a row at t
  counts). Per runner on C1's board at t (`models.nowcast.race_view`, tau = post - t):
  `path_change` = O_t / O_ref - 1, O_t its price on the board and O_ref its latest price at or
  before t - 5 min; `c1_drift` = E[s_t / s_final] - 1 over `nowcast.draws(view, pool=WIN)`. A
  nowcast fitted on the race's day or a later one (`nowcast.days`) has seen the race: ValueError.
  No WIN board at t (no row at or before t, or none priced) is a KeyError, as in `race_view`.

E22, the blend weight by tau
- `alpha_by_tau(frame, buckets=)`: a case is one (race_id, tau_min) softmax, and a row's bucket the
  smallest bucket at or above its tau (C1's rule). Per bucket, the Benter blend
  softmax(alpha ln p_model + beta ln q_tau) by conditional-logit maximum likelihood, beta free.
  Giacomini-White: each case's log-loss differential, one fixed blend fitted on every bucket pooled
  minus the bucket's blend, both cross-fitted over `CROSS_FIT` groups of races (a case is scored
  by blends fitted without its race); its mean over the bucket against the race-clustered variance
  of that mean, chi-squared with one degree of freedom, two-sided.

The study's inputs
- `path_files(source)`: the development day files of a path source, "jravan"
  (`processed/jravan/odds_timeseries`) or "jra-logs" (the compacted logs,
  `processed/jra/odds_timeseries`), sorted by day; holdout days (D14, by file name, never read)
  and days before 2014-06-07 (D10) are left out.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from horsetrack.core import paths
from horsetrack.core.eval.bootstrap import Interval, paired_bootstrap
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.scoring import gain, per_race_log_loss
from horsetrack.countries.japan.models import nowcast as nc
from horsetrack.countries.japan.models.condlogit import fit_conditional_logit
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START

COUNTRY = "japan"
PATH_SOURCES = {"jravan": "jravan", "jra-logs": "jra"}  # a path source and its processed folder
FEATURES = ("path_change", "c1_drift")
PATH_WINDOW_MIN = 5.0  # E24's observed path: the paper's 5-minute window, ending at t
N_DRAWS = 200
DRAW_SEED = 102
CROSS_FIT = 5


# ---------------------------------------------------------------- E23, ex post


def late_change(captures: pd.DataFrame, *, close_utc: datetime, window_min: float) -> pd.Series:
    """O_to / O_from - 1 per horse of one race's captures (horse_id, fetched_utc, odds)."""
    close = pd.Timestamp(close_utc)
    rows = captures.assign(_t=pd.to_datetime(captures["fetched_utc"], utc=True))
    rows = rows.sort_values("_t", kind="stable")
    horses = pd.Index(pd.unique(captures["horse_id"]), name="horse_id")

    def last(keep: pd.Series) -> pd.Series:
        latest = rows[keep].drop_duplicates("horse_id", keep="last").set_index("horse_id")
        return latest["odds"].astype(float).reindex(horses)

    o_from = last(rows["_t"] <= close - pd.Timedelta(minutes=float(window_min)))
    o_to = last(rows["_t"] < close)
    return (o_to / o_from - 1.0).rename("late_change")


@dataclass(frozen=True)
class RegressionResult:
    delta: float  # the late_change coefficient
    se: float
    n_runners: int
    n_races: int
    n_clusters: int


def _cluster_codes(race: np.ndarray, clusters: pd.Series | None) -> np.ndarray:
    unit = race if clusters is None else clusters.reindex(race).to_numpy()
    codes = pd.factorize(unit)[0]
    if (codes < 0).any():
        raise ValueError("a race has no cluster")
    return codes


def paper_regression(frame: pd.DataFrame, *, clusters: pd.Series | None = None) -> RegressionResult:
    """The paper's return regression; `clusters` maps race_id to its unit (default: the race)."""
    rows = frame.dropna(subset=["late_change"])
    odds = rows["odds_final"].to_numpy(dtype=float)
    y = rows["won"].to_numpy(dtype=bool) * odds
    X = np.column_stack([np.ones(len(rows)), odds, rows["late_change"].to_numpy(dtype=float)])
    race = rows["race_id"].to_numpy()
    codes = _cluster_codes(race, clusters)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    moments = X * (y - X @ beta)[:, None]
    scores = np.column_stack([np.bincount(codes, weights=m) for m in moments.T])
    bread = np.linalg.inv(X.T @ X)
    cov = bread @ (scores.T @ scores) @ bread
    return RegressionResult(
        delta=float(beta[2]),
        se=float(np.sqrt(cov[2, 2])),
        n_runners=len(rows),
        n_races=int(pd.unique(race).size),
        n_clusters=int(codes.max()) + 1,
    )


# ---------------------------------------------------------------- the final-price offset


def _softmax_by(u: np.ndarray, groups: np.ndarray) -> np.ndarray:
    codes = pd.factorize(groups)[0]
    top = pd.Series(u).groupby(codes).transform("max").to_numpy()
    e = np.exp(u - top)
    return e / np.bincount(codes, weights=e)[codes]


def offset_probabilities(frame: pd.DataFrame, *, coef: Mapping[str, float]) -> pd.Series:
    """softmax over each race of ln q_final + sum_c coef[c] * frame[c], on frame.index."""
    u = np.log(frame["q_final"].to_numpy(dtype=float))
    for column, value in coef.items():
        u = u + float(value) * frame[column].to_numpy(dtype=float)
    return pd.Series(_softmax_by(u, frame["race_id"].to_numpy()), index=frame.index, name="p")


def fit_offset(frame: pd.DataFrame, *, path_cols: Sequence[str]) -> dict[str, float]:
    """The path coefficients with ln q_final as the offset (conditional-logit MLE)."""
    cols = list(path_cols)
    fit = fit_conditional_logit(
        frame[cols].to_numpy(dtype=float),
        frame["race_id"].to_numpy(),
        frame["won"].to_numpy(dtype=bool),
        offset=np.log(frame["q_final"].to_numpy(dtype=float)),
    )
    return {c: float(v) for c, v in zip(cols, fit.coef, strict=True)}


def _race_log_loss(p: pd.Series, frame: pd.DataFrame) -> pd.Series:
    no_scratch = pd.Series(False, index=frame.index)
    won = frame["won"].astype(bool)
    return per_race_log_loss(p, won, frame["race_id"], no_scratch).per_race


def offset_test(
    frame: pd.DataFrame,
    *,
    path_cols: Sequence[str],
    clusters: pd.Series,
    n_boot: int,
    level: float,
    seed: int,
    folds: pd.Series | None = None,
) -> Interval:
    """The gain of ln q_final + the fitted path terms over ln q_final, per race in nats."""
    cols = list(path_cols)
    if folds is None:
        scored = frame
        p = offset_probabilities(frame, coef=fit_offset(frame, path_cols=cols))
    else:
        label = frame["race_id"].map(folds)
        if label.isna().any():
            raise ValueError("a race has no fold label")
        order = sorted(label.unique())
        if len(order) < 2:
            raise ValueError("a walk-forward offset test needs two fold labels or more")
        parts = []
        for i, current in enumerate(order[1:], start=1):
            coef = fit_offset(frame[label.isin(order[:i])], path_cols=cols)
            parts.append(offset_probabilities(frame[label == current], coef=coef))
        p = pd.concat(parts)
        scored = frame.loc[p.index]
    base = offset_probabilities(scored, coef={})
    per_race = gain(_race_log_loss(p, scored), _race_log_loss(base, scored)).per_race
    return paired_bootstrap(per_race, clusters, n_boot=n_boot, level=level, seed=seed)


# ---------------------------------------------------------------- E24, information at t


def ex_ante_features(
    frame: pd.DataFrame,
    *,
    at_utc: datetime,
    nowcast,
    grades: Mapping[str, str | None] | None = None,
    n_draws: int = N_DRAWS,
    seed: int = DRAW_SEED,
) -> pd.DataFrame:
    """`path_change` and `c1_drift` per WIN runner on the board at t, indexed by horse_id."""
    at = pd.Timestamp(at_utc)
    rows = frame[frame["pool"] == nc.WIN]
    rows = rows[pd.to_datetime(rows["snapshot_utc"], utc=True) <= at]
    races = pd.unique(rows["race_id"])
    if not len(races):  # as race_view without rows: there is no board at t
        raise KeyError(f"no WIN row at or before {at.isoformat()}")
    if len(races) > 1:
        raise ValueError(f"one race's rows are needed, got {len(races)} races")
    race_id = str(races[0])
    day = nc._race_day(race_id)
    seen = sorted(d for d in getattr(nowcast, "days", ()) if str(d)[:10] >= day)
    if seen:
        raise ValueError(f"the nowcast was fitted on {seen[0]}, on or after {race_id}'s day")
    posts = pd.to_datetime(rows["post_utc"], utc=True).dropna()
    if posts.empty:
        raise ValueError(f"{race_id} has no post time")
    tau = (posts.iloc[0] - at).total_seconds() / 60.0
    view = nc.race_view(rows, race_id, tau_min=tau, grades=grades)
    board = view.pools[nc.WIN]  # KeyError when no WIN runner is priced at t
    combos = list(board.combinations)

    prices = rows[rows["combination"] != nc.POOL_ROW]
    ref_at = at - pd.Timedelta(minutes=PATH_WINDOW_MIN)
    ref = prices[pd.to_datetime(prices["snapshot_utc"], utc=True) <= ref_at]
    ref = ref.sort_values("snapshot_utc", kind="stable").drop_duplicates("combination", keep="last")
    o_ref = ref.set_index("combination")["odds"].astype(float).reindex(combos)
    o_ref = o_ref.where(o_ref > 0).to_numpy()

    s_t = np.asarray(board.shares, dtype=float)
    s_t = s_t / s_t.sum()
    draws = nowcast.draws(view, pool=nc.WIN, n=n_draws, seed=seed)
    drift = (s_t[None, :] / np.asarray(draws.shares, dtype=float)).mean(axis=0) - 1.0
    return pd.DataFrame(
        {"path_change": np.asarray(board.odds, dtype=float) / o_ref - 1.0, "c1_drift": drift},
        index=pd.Index(combos, name="horse_id"),
    )


# ---------------------------------------------------------------- E22, the blend weight by tau


def _case_losses(u: np.ndarray, codes: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Each case's winner log loss under softmax(u) within the case (codes 0..n-1)."""
    top = pd.Series(u).groupby(codes).transform("max").to_numpy()
    e = np.exp(u - top)
    log_p = u - top - np.log(np.bincount(codes, weights=e))[codes]
    return -np.bincount(codes, weights=w * log_p)


def _blend(X: np.ndarray, codes: np.ndarray, won: np.ndarray) -> np.ndarray:
    return fit_conditional_logit(X, codes, won).coef


def alpha_by_tau(frame: pd.DataFrame, *, buckets: Sequence[float]) -> pd.DataFrame:
    """Per bucket: alpha and beta of the blend, and the Giacomini-White test against one blend."""
    grid = list(buckets)
    tau = frame["tau_min"].astype(float)
    bucket = tau.map({t: nc._bucket(t, grid) for t in tau.unique()}).to_numpy()
    codes = frame.groupby(["race_id", "tau_min"], sort=False).ngroup().to_numpy()
    won = frame["won"].to_numpy(dtype=bool)
    winners = np.bincount(codes, weights=won)
    keep = winners[codes] > 0  # a case without a winner carries no likelihood
    codes = pd.factorize(codes[keep])[0]
    bucket, won = bucket[keep], won[keep]
    w = won / np.bincount(codes, weights=won)[codes]
    race = frame["race_id"].to_numpy()[keep]
    ln_p, ln_q = (np.log(frame[c].to_numpy(dtype=float)) for c in ("p_model", "q_tau"))
    X = np.column_stack([ln_p, ln_q])[keep]
    case_bucket = np.zeros(codes.max() + 1)
    case_bucket[codes] = bucket
    case_race = pd.factorize(pd.Series(race).groupby(codes).first().to_numpy())[0]
    group = np.array([zlib.crc32(str(r).encode()) % CROSS_FIT for r in race])

    fixed = _blend(X, codes, won)
    diff = np.full(codes.max() + 1, np.nan)
    for j in range(CROSS_FIT):
        train, test = group != j, group == j
        pooled = _blend(X[train], codes[train], won[train])
        for b in grid:
            rows = test & (bucket == b)
            fit_rows = train & (bucket == b)
            if not rows.any():
                continue
            own = _blend(X[fit_rows], codes[fit_rows], won[fit_rows])
            local = pd.factorize(codes[rows])
            loss_fixed = _case_losses(X[rows] @ pooled, local[0], w[rows])
            loss_own = _case_losses(X[rows] @ own, local[0], w[rows])
            diff[local[1]] = loss_fixed - loss_own

    out = []
    for b in grid:
        in_bucket = bucket == b
        cases = np.flatnonzero(case_bucket == b)
        row = {"tau_min": b, "n_races": len(cases)}
        if not len(cases):
            out.append(row | dict.fromkeys(("alpha", "beta", "gw_stat", "gw_pvalue"), np.nan))
            continue
        fit = fit_conditional_logit(X[in_bucket], codes[in_bucket], won[in_bucket])
        d = diff[cases]
        mean = float(d.mean())
        by_race = pd.Series(d).groupby(case_race[cases])
        var = float(((by_race.sum() - by_race.size() * mean) ** 2).sum()) / len(cases) ** 2
        stat = mean * mean / var if var > 0 else np.nan
        out.append(
            row
            | {"alpha": float(fit.coef[0]), "beta": float(fit.coef[1])}
            | {"alpha_se": float(fit.se[0]), "beta_se": float(fit.se[1])}
            | {"alpha_fixed": float(fixed[0]), "beta_fixed": float(fixed[1]), "gw_mean": mean}
            | {"gw_stat": stat, "gw_pvalue": float(stats.chi2.sf(stat, 1))}
        )
    return pd.DataFrame(out)


# ---------------------------------------------------------------- the study's inputs


def path_files(source: str) -> list[tuple[str, Path]]:
    """("YYYY-MM-DD", path) of each development day file of `source`, sorted by day."""
    if source not in PATH_SOURCES:
        raise ValueError(f"no path source {source!r}; one of {sorted(PATH_SOURCES)}")
    folder = paths.processed_dir(PATH_SOURCES[source]) / "odds_timeseries"
    out = []
    for path in sorted(folder.glob("*.parquet")) if folder.is_dir() else []:
        try:
            day = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if day >= PRICE_ERA_START and not is_holdout_day(COUNTRY, day):
            out.append((day.isoformat(), path))
    return out
