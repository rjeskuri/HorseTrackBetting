"""The C1 final-price nowcast v0: the structural phi_t x s_late model (#85, epic #73; 08 §3.2 C1).

Not the drift nowcast of #116 (`models/v0/nowcast.py`), which stays as it is; `ShadowNowcast`
puts this model behind #116's draws interface so the shadow values can switch by config.

Shares (per pool, over the combinations priced on a board)
- A combination's share is its money over the pool. With per-combination money on every priced
  row (`investment`), that money; else WIN shares are 1/O normalised, and the stake-back-share
  pools (PLACE, QUINELLA_PLACE; `payout_rule` in config/jra.yaml) invert the low figures by that
  rule: a combination's low figure prices it with the P - 1 most-backed others, so in units of
  the pool m_i (P lo_i - P + 1) = (1 - takeout) - (the P - 1 largest other m_j), P from
  `places_for(starters)`. That is solved in closed form (`stake_back_money`); low figures below
  the pool's minimum dividend are read at it. The money sums to 1 when the rule holds, which the
  study reports as the rule's check (`rule_sum`).
- `late_composition(v1, s1, v2, s2)` = (v2 s2 - v1 s1) / (v2 - v1), the money bet in between.

The board at t (`race_view`; rows in `logger.compact.SCHEMA`)
- t = post_utc - tau. A row counts from its snapshot_utc (owner, 2026-11-26: no lag); each
  combination takes its latest row at or before t, and a latest row without a positive price
  leaves the combination off the board. `total` is the latest `__pool__` investment at or before
  t; the lag boards are read the same way at t - 5 and t - 15 minutes, aligned to the board's
  combinations (NaN where a combination had no price then). `board_min` is the minutes from the
  oldest row on the board to the post: on 5-minute paths the board at t can be minutes old.
- The final board is each combination's latest row (the store's `is_final_capture`), V_final the
  latest pool total.

The model
- s_final = phi s_t + (1 - phi) s_late, phi = V_t / V_final, and
  s_late = softmax(ln s_t + beta . f + sigma eps), eps iid N(0, 1) per combination: logistic-normal
  around the board, with f = (ln s_t - ln s_{t-5}, ln s_t - ln s_{t-15}, ln rank) (a missing lag
  gives 0). Draws have shares >= phi s_t, so their spread scales with 1 - phi.
- `PoolGrowth` gives phi: ln(V_final / V_m) = h(m) exp(theta . x), with h piecewise linear over
  `KNOTS` in the board's minutes to post m and x one-hot venue, race number and grade (ridge
  `EFFECT_RIDGE`), fitted on every pool-total snapshot of the training days. phi is that profile's
  prediction whether or not the board prints a total (08 R20: a board without one still has its
  minutes to post); V_final = V_t / phi, or the training mean of ln V_final without a total.
- `LateComposition` cells (pool, tau): beta by least squares within the case on
  ln max(r, R_FLOOR), r = (s_final / s_t - phi) / (1 - phi) with the observed phi where both totals
  exist; then sigma, and the random walk's sigma, each maximise the mean training log score of
  the final ln-shares (`case_scores`) over a seeded sample of at most `N_CAL` cases with
  `N_CAL_DRAWS` common draws. A view's tau uses the smallest grid tau at or above it.
- `random_walk`: q_final = q_t with its fitted noise, softmax(ln s_t + sigma_rw eps).
- `Nowcast.from_params(phi=, sigma=)`: the known nowcast (beta = 0, phi fixed, V_final = V_t / phi).

Folds and scores
- `Nowcast.fit` and `compare` refuse a holdout day (`HoldoutDayError`, D14); `fit_folds` drops
  holdout days, then fits each `walk_forward` fold on its training days only.
- `compare` scores every (race, pool, tau of the decision grid) case with both forecasters on the
  same view, n and a seed derived from (seed, race, pool, tau); per case the score is the mean over
  combinations of the Gaussian log score and CRPS of the final ln-share (and the KL of the final
  shares from the mean draw). Gains are positive when the candidate is better, with one-sided
  bootstrap bounds (`core.eval.bootstrap.paired_bootstrap`) resampling race days. A case whose
  combinations change between t and the close (a late scratch) is not scored.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from functools import cache

import numpy as np
import pandas as pd
from scipy import optimize

from horsetrack.core import config
from horsetrack.core.eval import distributional
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import FoldScheme, walk_forward
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.countries.japan.board import snapshot
from horsetrack.countries.japan.board.snapshot import RaceBoard
from horsetrack.countries.japan.models.v0.nowcast import HoldoutDayError
from horsetrack.countries.japan.models.v0.shadow_values import PoolDraws

__all__ = [
    "Draws",
    "FoldFit",
    "HoldoutDayError",
    "Nowcast",
    "PoolBoard",
    "PoolGrowth",
    "RaceView",
    "ShadowNowcast",
    "compare",
    "fit_folds",
    "late_composition",
    "race_view",
]

COUNTRY = "japan"
MODEL_NAME = "nowcast_c1"
WIN, PLACE, WIDE = snapshot.WIN, snapshot.PLACE, snapshot.WIDE
POOLS = (WIN, PLACE, WIDE)
POOL_ROW = "__pool__"
LAGS_MIN = (5.0, 15.0)
FEATURES = ("velocity_5", "velocity_15", "ln_rank")
# Piecewise-linear knots of the growth profile, minutes from the board to the post.
KNOTS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 21, 25, 30, 35, 40, 50, 60, 75, 90, 120)
PROFILE_PENALTY = 1e-2  # second divided differences, per observation per knot
EFFECT_RIDGE = 100.0  # observations' worth of shrinkage of each venue, race-number, grade effect
R_FLOOR = 0.02  # the late-money ratio r is read at this floor in the beta fit
PHI_MIN = 1e-4
N_CAL = 400
N_CAL_DRAWS = 64
CAL_SEED = 85
SIGMA_BOUNDS = (1e-3, 20.0)
SD_FLOOR = 1e-6  # the Gaussian log score's sd floor, in ln-share units
TINY = 1e-300
_US_PER_MIN = 60_000_000
_FAR_FUTURE = np.iinfo(np.int64).max


# ---------------------------------------------------------------- shares


@cache
def _pool_config(pool: str):
    return config.load_source(COUNTRY, "jra").pools[pool]


def _stake_back(pool: str) -> bool:
    return _pool_config(pool).payout_rule == "stake_back_share"


def _floor(pool: str) -> float:
    return _pool_config(pool).min_dividend / 100.0


def late_composition(v1: float, s1: np.ndarray, v2: float, s2: np.ndarray) -> np.ndarray:
    """The composition of the money bet between two snapshots: (v2 s2 - v1 s1) / (v2 - v1)."""
    if not v2 > v1:
        raise ValueError(f"the pool must grow between the snapshots, got {v1} then {v2}")
    s1, s2 = np.asarray(s1, dtype=float), np.asarray(s2, dtype=float)
    return (v2 * s2 - v1 * s1) / (v2 - v1)


def stake_back_money(lo: np.ndarray, places: int, payout: float) -> np.ndarray:
    """Each combination's money in units of the pool from its low figure, by the stake-back-share
    rule with `places` winning tickets and `payout` = 1 - takeout.

    With a_i = 1 / (P (lo_i - 1) + 1) the rule reads m_i = a_i (payout - T_i), T_i the P - 1
    largest other m_j. The P largest a form the top set S: for i in S, m_i = (payout - M_S)
    a_i / (1 - a_i) with M_S = payout B / (1 + B), B = sum_S a / (1 - a); outside S,
    m_i = a_i (payout - M_S + m_min(S)). Rows of a 2-D `lo` are boards (or draws)."""
    lo = np.asarray(lo, dtype=float)
    a = 1.0 / (places * (np.atleast_2d(lo) - 1.0) + 1.0)
    top = np.argsort(-a, axis=1, kind="stable")[:, :places]
    a_top = np.take_along_axis(a, top, axis=1)
    ratio = a_top / (1.0 - a_top)
    b = ratio.sum(axis=1, keepdims=True)
    m_top = payout * b / (1.0 + b)
    in_top = (payout - m_top) * ratio
    money = a * (payout - m_top + in_top[:, -1:])
    np.put_along_axis(money, top, in_top, axis=1)
    return money.reshape(lo.shape)


def stake_back_odds(shares: np.ndarray, places: int, payout: float) -> tuple[np.ndarray, ...]:
    """The low and high figures the stake-back-share rule prints for money `shares` (rows sum to
    1): low with the P - 1 most-backed others, high with the P - 1 least-backed. Rows are draws."""
    m = np.atleast_2d(np.asarray(shares, dtype=float))
    desc, asc = -np.sort(-m, axis=1), np.sort(m, axis=1)
    p1 = places - 1
    big, small = desc[:, :p1].sum(axis=1, keepdims=True), asc[:, :p1].sum(axis=1, keepdims=True)
    big_p = big + desc[:, p1 : p1 + 1]
    small_p = small + asc[:, p1 : p1 + 1]
    t_big = np.where(m >= desc[:, p1 - 1 : p1], big_p - m, big)
    t_small = np.where(m <= asc[:, p1 - 1 : p1], small_p - m, small)
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = 1.0 + (payout - m - t_big) / (places * m)
        hi = 1.0 + (payout - m - t_small) / (places * m)
    return lo, hi


def pool_shares(
    pool: str, odds: np.ndarray, investment: np.ndarray | None = None
) -> tuple[np.ndarray | None, float]:
    """(shares, rule_sum) of one board's priced combinations (or, for 2-D `odds`, of each row).
    `rule_sum` is the stake-back money's total in units of the pool (1 when the rule holds), NaN
    when the rule was not used; shares are None when the pool is not sold for this many runners
    (Wide counts its pairs, which gives its three places from 4 runners, as `places_for` does)."""
    odds = np.asarray(odds, dtype=float)
    if investment is not None:
        money = np.asarray(investment, dtype=float)
        if money.size and np.all(np.isfinite(money) & (money > 0)):
            return money / money.sum(axis=-1, keepdims=True), math.nan
    if not _stake_back(pool):
        inverse = 1.0 / odds
        return inverse / inverse.sum(axis=-1, keepdims=True), math.nan
    places = _pool_config(pool).places_for(odds.shape[-1])
    if places == 0 or odds.shape[-1] < places:
        return None, math.nan
    payout = 1.0 - float(_pool_config(pool).takeout)
    money = stake_back_money(np.maximum(odds, _floor(pool)), places, payout)
    total = money.sum(axis=-1, keepdims=True)
    return money / total, float(total[0]) if odds.ndim == 1 else math.nan


# ---------------------------------------------------------------- the board at t


@dataclass(frozen=True, eq=False)
class PoolBoard:
    combinations: tuple[str, ...]
    shares: np.ndarray
    total: float | None
    shares_lag5: np.ndarray | None
    shares_lag15: np.ndarray | None
    board_min: float | None = None  # minutes from the board's oldest row to the post
    odds: np.ndarray | None = None  # the low figure at t
    odds_hi: np.ndarray | None = None


@dataclass(frozen=True, eq=False)
class RaceView:
    race_id: str
    venue: str
    race_no: int
    grade: str | None
    tau_min: float
    pools: Mapping[str, PoolBoard]


@dataclass(frozen=True, eq=False)
class Draws:
    shares: np.ndarray  # (n, k), columns in the board's combination order
    v_final: np.ndarray  # (n,)


def _combination_key(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in text.split(","))
    except ValueError:
        return (1 << 30,)


def _micros(values: pd.Series) -> np.ndarray:
    """UTC instants as int64 microseconds; NaT as the int64 minimum."""
    stamps = pd.to_datetime(values, utc=True)
    out = stamps.to_numpy(dtype="datetime64[us]").astype(np.int64)
    return np.where(stamps.isna().to_numpy(), np.iinfo(np.int64).min, out)


@dataclass(frozen=True, eq=False)
class _Series:
    """One race's rows of one pool as a step function of time."""

    pool: str
    combinations: tuple[str, ...]
    times: np.ndarray  # (T,) ascending
    latest: np.ndarray  # (T, C): the time index of each combination's latest row, -1 for none
    lo: np.ndarray  # (T, C), the row's own values where it has a row
    hi: np.ndarray
    inv: np.ndarray
    total_times: np.ndarray
    total_values: np.ndarray


@dataclass(frozen=True, eq=False)
class _State:
    index: np.ndarray  # positions in the series' combinations
    lo: np.ndarray
    hi: np.ndarray
    inv: np.ndarray
    capture: int  # the oldest row on the board, microseconds


def _series(pool: str, rows: pd.DataFrame) -> _Series | None:
    prices = rows[rows["combination"] != POOL_ROW]
    totals = rows[(rows["combination"] == POOL_ROW) & rows["investment"].notna()]
    totals = totals.sort_values("snap", kind="stable").drop_duplicates("snap", keep="last")
    if prices.empty:
        return None
    prices = prices.sort_values("snap", kind="stable").drop_duplicates(
        ["combination", "snap"], keep="last"
    )
    combinations = tuple(sorted(prices["combination"].unique(), key=_combination_key))
    column = {c: j for j, c in enumerate(combinations)}
    times = np.unique(prices["snap"].to_numpy())
    ti = np.searchsorted(times, prices["snap"].to_numpy())
    ci = prices["combination"].map(column).to_numpy()
    shape = (len(times), len(combinations))
    lo, hi, inv = (np.full(shape, np.nan) for _ in range(3))
    lo[ti, ci] = prices["odds"].to_numpy(dtype=float, na_value=np.nan)
    hi[ti, ci] = prices["odds_hi"].to_numpy(dtype=float, na_value=np.nan)
    inv[ti, ci] = prices["investment"].to_numpy(dtype=float, na_value=np.nan)
    present = np.zeros(shape, dtype=bool)
    present[ti, ci] = True
    latest = np.maximum.accumulate(np.where(present, np.arange(len(times))[:, None], -1), axis=0)
    return _Series(
        pool,
        combinations,
        times,
        latest,
        lo,
        hi,
        inv,
        totals["snap"].to_numpy(),
        totals["investment"].to_numpy(dtype=float),
    )


def _state_at(series: _Series, at: int) -> _State | None:
    ti = int(np.searchsorted(series.times, at, side="right")) - 1
    if ti < 0:
        return None
    rows = series.latest[ti]
    c = np.arange(len(series.combinations))
    has = rows >= 0
    safe = np.where(has, rows, 0)
    lo = np.where(has, series.lo[safe, c], np.nan)
    priced = has & np.isfinite(lo) & (lo > 0)
    if not priced.any():
        return None
    index = np.flatnonzero(priced)
    return _State(
        index,
        lo[index],
        series.hi[safe[index], index],
        series.inv[safe[index], index],
        int(series.times[rows[index]].min()),
    )


def _total_at(series: _Series, at: int) -> float | None:
    j = int(np.searchsorted(series.total_times, at, side="right")) - 1
    return float(series.total_values[j]) if j >= 0 else None


def _shares_of(pool: str, state: _State) -> tuple[np.ndarray | None, float]:
    return pool_shares(pool, state.lo, state.inv)


def _pool_board(series: _Series, post: int, tau_min: float) -> tuple[PoolBoard, float] | None:
    at = post - int(round(tau_min * _US_PER_MIN))
    state = _state_at(series, at)
    if state is None:
        return None
    shares, rule_sum = _shares_of(series.pool, state)
    if shares is None:
        return None
    lags = []
    for lag in LAGS_MIN:
        then = _state_at(series, at - int(round(lag * _US_PER_MIN)))
        lag_shares = None if then is None else _shares_of(series.pool, then)[0]
        if lag_shares is None:
            lags.append(None)
            continue
        full = np.full(len(series.combinations), np.nan)
        full[then.index] = lag_shares
        lags.append(full[state.index])
    board = PoolBoard(
        combinations=tuple(series.combinations[i] for i in state.index),
        shares=shares,
        total=_total_at(series, at),
        shares_lag5=lags[0],
        shares_lag15=lags[1],
        board_min=(post - state.capture) / _US_PER_MIN,
        odds=state.lo,
        odds_hi=state.hi,
    )
    return board, rule_sum


def _final_board(series: _Series) -> tuple[tuple[str, ...], np.ndarray, np.ndarray] | None:
    state = _state_at(series, _FAR_FUTURE)
    if state is None:
        return None
    shares, _ = _shares_of(series.pool, state)
    if shares is None:
        return None
    return tuple(series.combinations[i] for i in state.index), shares, state.lo


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame[frame["pool"].isin(POOLS)]
    rows = rows.assign(snap=_micros(rows["snapshot_utc"]), post=_micros(rows["post_utc"]))
    return rows[rows["snap"] > np.iinfo(np.int64).min]


def _race_day(race_id: str) -> str:
    return str(race_id).split(":")[1][:10]


def _in_days(values: np.ndarray, days: Iterable[str]) -> np.ndarray:
    """Whether each "YYYY-MM-DD" in `values` is one of `days` (a hash lookup)."""
    return pd.Series(values, dtype=object).isin({str(d)[:10] for d in days}).to_numpy()


@dataclass(frozen=True, eq=False)
class _Race:
    race_id: str
    day: str
    venue: str
    race_no: int
    grade: str | None
    post: int
    series: dict[str, _Series]


def _races(frame: pd.DataFrame, grades: Mapping[str, str | None] | None) -> Iterable[_Race]:
    """The races of `frame` with a post time, in race-id order."""
    rows = _prepare(frame)
    for race_id, g in rows.groupby("race_id", sort=True):
        posts = g["post"].to_numpy()
        posts = posts[posts > np.iinfo(np.int64).min]
        if not len(posts):
            continue
        series = {}
        for pool, pg in g.groupby("pool", sort=False):
            built = _series(str(pool), pg)
            if built is not None:
                series[str(pool)] = built
        yield _Race(
            str(race_id),
            _race_day(str(race_id)),
            str(g["venue"].iloc[0]),
            int(g["race_no"].iloc[0]),
            None if grades is None else grades.get(str(race_id)),
            int(posts[0]),
            series,
        )


def race_view(
    frame: pd.DataFrame,
    race_id: str,
    *,
    tau_min: float,
    grades: Mapping[str, str | None] | None = None,
) -> RaceView:
    """The board of each pool of `race_id` at post - tau_min, from rows in compact.SCHEMA."""
    races = list(_races(frame[frame["race_id"] == race_id], grades))
    if not races:
        raise KeyError(f"{race_id} has no rows with a post time")
    race = races[0]
    pools = {}
    for pool, series in race.series.items():
        built = _pool_board(series, race.post, float(tau_min))
        if built is not None:
            pools[pool] = built[0]
    return RaceView(race.race_id, race.venue, race.race_no, race.grade, float(tau_min), pools)


# ---------------------------------------------------------------- cases


def _grid() -> tuple[int, ...]:
    return tuple(sorted(snapshot.load_decision_grid().minutes, reverse=True))


@dataclass(frozen=True, eq=False)
class Cases:
    """Columnar (race, pool, tau) cases: case-level arrays and runner rows `start:start + k`."""

    race_id: np.ndarray
    day: np.ndarray
    venue: np.ndarray
    race_no: np.ndarray
    grade: np.ndarray
    pool: np.ndarray
    tau: np.ndarray
    start: np.ndarray
    k: np.ndarray
    total_t: np.ndarray  # NaN without a total on the board
    total_f: np.ndarray
    board_min: np.ndarray
    rule_sum: np.ndarray  # the stake-back money's total at t (NaN when the rule was not used)
    combination: np.ndarray  # runner rows from here
    s_t: np.ndarray
    lag5: np.ndarray
    lag15: np.ndarray
    s_f: np.ndarray
    odds_t: np.ndarray
    odds_hi_t: np.ndarray
    odds_f: np.ndarray

    CASE = (
        "race_id", "day", "venue", "race_no", "grade", "pool", "tau", "k", "total_t", "total_f",
        "board_min", "rule_sum",
    )  # fmt: skip
    ROW = ("combination", "s_t", "lag5", "lag15", "s_f", "odds_t", "odds_hi_t", "odds_f")

    def __len__(self) -> int:
        return len(self.race_id)

    @classmethod
    def empty(cls) -> Cases:
        return cls.build([], [])

    @classmethod
    def build(cls, case_rows: list[dict], runner_rows: list[dict]) -> Cases:
        """From one dict per case (CASE fields) and one dict of row arrays per case (ROW fields)."""
        obj = {"race_id", "day", "venue", "grade", "pool", "combination"}

        def column(name, values):
            if name in obj:
                return np.array(values, dtype=object)
            if name in ("race_no", "k"):
                return np.array(values, dtype=np.int64)
            return np.array(values, dtype=float)

        fields = {name: column(name, [c[name] for c in case_rows]) for name in cls.CASE}
        for name in cls.ROW:
            parts = [
                np.asarray(r[name], dtype=object if name in obj else float) for r in runner_rows
            ]
            fields[name] = (
                np.concatenate(parts) if parts else column(name, [])  # type: ignore[arg-type]
            )
        fields["start"] = np.concatenate([[0], np.cumsum(fields["k"])[:-1]]).astype(np.int64)
        return cls(**fields)

    @classmethod
    def concat(cls, parts: Sequence[Cases]) -> Cases:
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        fields = {
            f.name: np.concatenate([getattr(p, f.name) for p in parts])
            for f in dataclasses.fields(cls)
            if f.name != "start"
        }
        fields["start"] = np.concatenate([[0], np.cumsum(fields["k"])[:-1]]).astype(np.int64)
        return cls(**fields)

    def row_case(self) -> np.ndarray:
        """The case of each runner row."""
        return np.repeat(np.arange(len(self)), self.k)

    def select(self, mask: np.ndarray) -> Cases:
        mask = np.asarray(mask, dtype=bool)
        rows = np.repeat(mask, self.k)
        fields = {name: getattr(self, name)[mask] for name in self.CASE}
        fields |= {name: getattr(self, name)[rows] for name in self.ROW}
        fields["start"] = np.concatenate([[0], np.cumsum(fields["k"])[:-1]]).astype(np.int64)
        return Cases(**fields)

    def select_days(self, days: Iterable[str]) -> Cases:
        return self.select(_in_days(self.day, days))

    def rows(self, i: int) -> slice:
        return slice(int(self.start[i]), int(self.start[i] + self.k[i]))

    def board(self, i: int) -> PoolBoard:
        r = self.rows(i)
        lag5, lag15 = self.lag5[r], self.lag15[r]
        total = self.total_t[i]
        return PoolBoard(
            combinations=tuple(self.combination[r]),
            shares=self.s_t[r],
            total=None if not np.isfinite(total) else float(total),
            shares_lag5=None if np.isnan(lag5).all() else lag5,
            shares_lag15=None if np.isnan(lag15).all() else lag15,
            board_min=float(self.board_min[i]),
            odds=self.odds_t[r],
            odds_hi=self.odds_hi_t[r],
        )

    def view(self, i: int) -> RaceView:
        return RaceView(
            str(self.race_id[i]),
            str(self.venue[i]),
            int(self.race_no[i]),
            self.grade[i],
            float(self.tau[i]),
            {str(self.pool[i]): self.board(i)},
        )


@dataclass(frozen=True, eq=False)
class Growth:
    """Pool-total snapshots (minutes to post, ln V_final / V_m) and each race-pool's V_final."""

    pool: np.ndarray
    day: np.ndarray
    venue: np.ndarray
    race_no: np.ndarray
    grade: np.ndarray
    minutes: np.ndarray
    log_growth: np.ndarray
    final_pool: np.ndarray
    final_day: np.ndarray
    log_v_final: np.ndarray

    @classmethod
    def build(cls, obs: list[tuple], finals: list[tuple]) -> Growth:
        o = list(zip(*obs, strict=True)) if obs else [[] for _ in range(7)]
        f = list(zip(*finals, strict=True)) if finals else [[] for _ in range(3)]
        return cls(
            np.array(o[0], dtype=object),
            np.array(o[1], dtype=object),
            np.array(o[2], dtype=object),
            np.array(o[3], dtype=np.int64),
            np.array(o[4], dtype=object),
            np.array(o[5], dtype=float),
            np.array(o[6], dtype=float),
            np.array(f[0], dtype=object),
            np.array(f[1], dtype=object),
            np.array(f[2], dtype=float),
        )

    @classmethod
    def concat(cls, parts: Sequence[Growth]) -> Growth:
        return cls(
            **{
                f.name: np.concatenate([getattr(p, f.name) for p in parts])
                for f in dataclasses.fields(cls)
            }
        )

    def select_days(self, days: Iterable[str]) -> Growth:
        keep = list({str(d)[:10] for d in days})
        obs, fin = _in_days(self.day, keep), _in_days(self.final_day, keep)
        names = [f.name for f in dataclasses.fields(self)]
        return Growth(
            **{n: getattr(self, n)[fin if n.startswith("final") or n == "log_v_final" else obs]
               for n in names}
        )  # fmt: skip


def extract(
    frame: pd.DataFrame,
    *,
    taus: Sequence[float] | None = None,
    grades: Mapping[str, str | None] | None = None,
) -> tuple[Cases, Growth, dict]:
    """The cases at every tau (default: the decision grid) and the growth observations of every
    race of `frame`, with counts of what could not be used."""
    taus = _grid() if taus is None else tuple(taus)
    case_rows, runner_rows, obs, finals = [], [], [], []
    counts = {"races": 0, "no_final": 0, "no_board": 0, "changed_by_close": 0}
    for race in _races(frame, grades):
        counts["races"] += 1
        for pool, series in race.series.items():
            final = _final_board(series)
            v_final = float(series.total_values[-1]) if len(series.total_values) else math.nan
            if np.isfinite(v_final) and v_final > 0:
                finals.append((pool, race.day, math.log(v_final)))
                minutes = (race.post - series.total_times) / _US_PER_MIN
                keep = (minutes >= KNOTS[0]) & (minutes <= KNOTS[-1]) & (series.total_values > 0)
                for m, v in zip(minutes[keep], series.total_values[keep], strict=True):
                    obs.append(
                        (pool, race.day, race.venue, race.race_no, race.grade, float(m),
                         max(math.log(v_final / v), 0.0))
                    )  # fmt: skip
            if final is None:
                counts["no_final"] += len(taus)
                continue
            f_combos, f_shares, f_odds = final
            f_index = {c: j for j, c in enumerate(f_combos)}
            for tau in taus:
                built = _pool_board(series, race.post, float(tau))
                if built is None:
                    counts["no_board"] += 1
                    continue
                board, rule_sum = built
                if set(board.combinations) != set(f_combos):
                    counts["changed_by_close"] += 1
                    continue
                order = [f_index[c] for c in board.combinations]
                case_rows.append(
                    {
                        "race_id": race.race_id,
                        "day": race.day,
                        "venue": race.venue,
                        "race_no": race.race_no,
                        "grade": race.grade,
                        "pool": pool,
                        "tau": float(tau),
                        "k": len(board.combinations),
                        "total_t": math.nan if board.total is None else board.total,
                        "total_f": v_final,
                        "board_min": board.board_min,
                        "rule_sum": rule_sum,
                    }
                )
                k = len(board.combinations)
                nan = np.full(k, np.nan)
                runner_rows.append(
                    {
                        "combination": board.combinations,
                        "s_t": board.shares,
                        "lag5": nan if board.shares_lag5 is None else board.shares_lag5,
                        "lag15": nan if board.shares_lag15 is None else board.shares_lag15,
                        "s_f": f_shares[order],
                        "odds_t": board.odds,
                        "odds_hi_t": board.odds_hi,
                        "odds_f": f_odds[order],
                    }
                )
    return Cases.build(case_rows, runner_rows), Growth.build(obs, finals), counts


# ---------------------------------------------------------------- pool growth


def _hat(minutes: np.ndarray, knots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.clip(np.asarray(minutes, dtype=float), knots[0], knots[-1])
    j = np.minimum(np.searchsorted(knots, m, side="right") - 1, len(knots) - 2)
    return j, (m - knots[j]) / (knots[j + 1] - knots[j])


def _fit_profile(minutes: np.ndarray, y: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Least squares on hat functions over `knots`, with a light penalty on second divided
    differences (whose null space is linear in minutes)."""
    k = len(knots)
    j, w = _hat(minutes, knots)
    btb, bty = np.zeros((k, k)), np.zeros(k)
    np.add.at(btb, (j, j), (1 - w) ** 2)
    np.add.at(btb, (j, j + 1), (1 - w) * w)
    np.add.at(btb, (j + 1, j), (1 - w) * w)
    np.add.at(btb, (j + 1, j + 1), w**2)
    np.add.at(bty, j, (1 - w) * y)
    np.add.at(bty, j + 1, w * y)
    d = np.zeros((k - 2, k))
    for i in range(1, k - 1):
        left, right = knots[i] - knots[i - 1], knots[i + 1] - knots[i]
        d[i - 1, i - 1], d[i - 1, i], d[i - 1, i + 1] = 1 / left, -1 / left - 1 / right, 1 / right
    lam = PROFILE_PENALTY * max(len(y) / k, 1.0)
    return np.linalg.solve(btb + lam * d.T @ d + 1e-12 * np.eye(k), bty)


def _effect_keys(venue, race_no, grade) -> tuple[str, str, str]:
    return f"venue={venue}", f"race_no={int(race_no)}", f"grade={grade or 'none'}"


@dataclass(frozen=True)
class PoolGrowth:
    """phi_t = V_t / V_final from the board's minutes to post, venue, race number and grade."""

    pool: str
    knots: tuple[float, ...]
    profile: tuple[float, ...]  # E ln(V_final / V_m) at each knot
    effects: dict[str, float]  # multiplicative, on the log scale
    log_v_final: float  # mean ln V_final, for boards without a total
    n_obs: int

    def log_growth(self, minutes: float, venue: str, race_no: int, grade: str | None) -> float:
        h = float(np.interp(minutes, self.knots, self.profile))
        scale = sum(self.effects.get(key, 0.0) for key in _effect_keys(venue, race_no, grade))
        return max(h, 0.0) * math.exp(scale)

    def phi(self, minutes: float, venue: str, race_no: int, grade: str | None) -> float:
        return min(max(math.exp(-self.log_growth(minutes, venue, race_no, grade)), PHI_MIN), 1.0)

    @classmethod
    def fit(cls, pool: str, growth: Growth) -> PoolGrowth | None:
        obs = growth.pool == pool
        finals = growth.log_v_final[growth.final_pool == pool]
        if obs.sum() < 2 or len(np.unique(growth.minutes[obs])) < 2 or not len(finals):
            return None
        knots = np.asarray(KNOTS, dtype=float)
        m, y = growth.minutes[obs], growth.log_growth[obs]
        profile = _fit_profile(m, y, knots)
        h = np.interp(m, knots, profile)
        use = (y > 0.01) & (h > 0.01) & (m >= 1.0)
        effects: dict[str, float] = {}
        if use.any():
            labels = list(
                zip(
                    *map(
                        _effect_keys,
                        growth.venue[obs][use],
                        growth.race_no[obs][use],
                        growth.grade[obs][use],
                    ),
                    strict=True,
                )
            )
            codes, names, offset = [], [], 0
            for part in labels:
                c, uniques = pd.factorize(np.asarray(part, dtype=object), sort=True)
                codes.append(c + offset)
                names += list(uniques)
                offset += len(uniques)
            xtx, xtr = np.zeros((offset, offset)), np.zeros(offset)
            r = np.log(y[use]) - np.log(h[use])
            for a in codes:
                np.add.at(xtr, a, r)
                for b in codes:
                    np.add.at(xtx, (a, b), 1.0)
            theta = np.linalg.solve(xtx + EFFECT_RIDGE * np.eye(offset), xtr)
            effects = {str(n): float(t) for n, t in zip(names, theta, strict=True)}
        return cls(
            pool,
            tuple(float(x) for x in knots),
            tuple(float(x) for x in profile),
            effects,
            float(np.mean(finals)),
            int(obs.sum()),
        )


# ---------------------------------------------------------------- the late composition


def _features(s, lag5, lag15, case_of_row, start) -> np.ndarray:
    """(velocity over [t-5, t], velocity over [t-15, t], ln rank of s_t) per runner row."""
    s = np.asarray(s, dtype=float)
    out = np.zeros((len(s), len(FEATURES)))
    for col, lag in enumerate((lag5, lag15)):
        lag = np.asarray(lag, dtype=float)
        ok = np.isfinite(lag) & (lag > 0) & (s > 0)
        out[ok, col] = np.log(s[ok]) - np.log(lag[ok])
    order = np.lexsort((-s, case_of_row))
    rank = np.empty(len(s))
    rank[order] = np.arange(len(s)) - start[case_of_row[order]] + 1
    out[:, 2] = np.log(rank)
    return out


def _board_features(board: PoolBoard) -> np.ndarray:
    k = len(board.shares)
    nan = np.full(k, np.nan)
    return _features(
        board.shares,
        nan if board.shares_lag5 is None else board.shares_lag5,
        nan if board.shares_lag15 is None else board.shares_lag15,
        np.zeros(k, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
    )


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _structural(s, mu, phi, sigma, eps) -> np.ndarray:
    """phi s + (1 - phi) softmax(ln s + mu + sigma eps); s, mu (..., k), eps (..., n, k)."""
    with np.errstate(divide="ignore"):
        z = np.log(s)[..., None, :] + mu[..., None, :] + sigma * eps
    phi = np.asarray(phi, dtype=float)[..., None, None]
    return phi * s[..., None, :] + (1.0 - phi) * _softmax(z)


def _random_walk(s, sigma, eps) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return _softmax(np.log(s)[..., None, :] + sigma * eps)


def case_scores(shares: np.ndarray, s_final: np.ndarray) -> dict:
    """One case's scores on the final ln-shares: the mean over combinations of the Gaussian log
    score (draws' mean and sd of ln share) and of the ensemble CRPS, the KL of the final shares
    from the mean draw, and each combination's PIT."""
    log_draws = np.log(np.maximum(np.asarray(shares, dtype=float), TINY))
    y = np.log(np.asarray(s_final, dtype=float))
    sd = np.maximum(log_draws.std(axis=0), SD_FLOOR)
    mean_share = np.asarray(shares, dtype=float).mean(axis=0)
    return {
        "log_score": float(distributional.gaussian_log_scores(log_draws.mean(0), sd, y).mean()),
        "crps": float(distributional.crps_columns(log_draws, y).mean()),
        "kl": float(np.sum(s_final * (y - np.log(np.maximum(mean_share, TINY))))),
        "pit": distributional.pit_columns(log_draws, y),
    }


def _batch_log_score(shares: np.ndarray, ln_final: np.ndarray, mask: np.ndarray) -> float:
    """The mean over cases of the mean over combinations of the Gaussian log score; shares
    (N, n, K) padded, ln_final and mask (N, K)."""
    log_draws = np.log(np.maximum(shares, TINY))
    sd = np.maximum(log_draws.std(axis=1), SD_FLOOR)
    z = (np.where(mask, ln_final, 0.0) - log_draws.mean(axis=1)) / sd
    score = np.where(mask, -0.5 * math.log(2 * math.pi) - np.log(sd) - 0.5 * z * z, 0.0)
    return float((score.sum(axis=1) / mask.sum(axis=1)).mean())


@dataclass(frozen=True)
class Cell:
    """The late composition of one (pool, tau) and the random walk's noise there."""

    pool: str
    tau_min: int
    n_cases: int
    n_calibration: int
    share_floored: float  # runner rows whose late ratio r was read at R_FLOOR
    beta: tuple[float, ...]
    sigma: float
    sigma_rw: float
    train_log_score: float
    train_log_score_rw: float


def _pad(cases: Cases, idx: np.ndarray, values: np.ndarray, fill: float) -> np.ndarray:
    k_max = int(cases.k[idx].max())
    out = np.full((len(idx), k_max), fill)
    for row, i in enumerate(idx):
        out[row, : cases.k[i]] = values[cases.rows(i)]
    return out


def _calibrate(objective: Callable[[float], float]) -> float:
    lo, hi = (math.log(b) for b in SIGMA_BOUNDS)
    best = optimize.minimize_scalar(
        lambda x: -objective(math.exp(x)), bounds=(lo, hi), method="bounded",
        options={"xatol": 1e-3},
    )  # fmt: skip
    return float(math.exp(best.x))


def _fit_cell(
    cases: Cases, idx: np.ndarray, phi_hat: np.ndarray, pool: str, tau: int, features: bool
) -> Cell:
    sub = cases.select(np.isin(np.arange(len(cases)), idx))
    case_of_row = sub.row_case()
    phi_obs = sub.total_t / sub.total_f
    phi_fit = np.where(np.isfinite(phi_obs) & (phi_obs > 0), np.minimum(phi_obs, 1.0), phi_hat)
    f = _features(sub.s_t, sub.lag5, sub.lag15, case_of_row, sub.start)
    phi_row = phi_fit[case_of_row]
    usable = (1.0 - phi_row) > 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (sub.s_f / sub.s_t - phi_row) / (1.0 - phi_row)
    floored = usable & ~(r > R_FLOOR)
    beta = np.zeros(len(FEATURES))
    if features and usable.any():
        y = np.log(np.where(usable, np.maximum(r, R_FLOOR), 1.0))
        weight = usable.astype(float)
        count = np.bincount(case_of_row, weights=weight, minlength=len(sub))
        count = np.where(count > 0, count, 1.0)
        y_bar = np.bincount(case_of_row, weights=y * weight, minlength=len(sub)) / count
        f_bar = np.stack(
            [np.bincount(case_of_row, weights=f[:, c] * weight, minlength=len(sub)) / count
             for c in range(f.shape[1])], axis=1,
        )  # fmt: skip
        yt = (y - y_bar[case_of_row]) * weight
        ft = (f - f_bar[case_of_row]) * weight[:, None]
        xtx = ft.T @ ft
        beta = np.linalg.solve(xtx + 1e-9 * max(np.trace(xtx), 1.0) * np.eye(len(beta)), ft.T @ yt)

    rng = np.random.default_rng(CAL_SEED)
    cal = np.arange(len(sub))
    if len(cal) > N_CAL:
        cal = np.sort(rng.choice(len(sub), size=N_CAL, replace=False))
    s = _pad(sub, cal, sub.s_t, 0.0)
    mu = _pad(sub, cal, f @ beta, 0.0)
    ln_final = _pad(sub, cal, np.log(sub.s_f), 0.0)
    mask = _pad(sub, cal, np.ones(len(sub.s_t)), 0.0) > 0
    eps = rng.standard_normal((len(cal), N_CAL_DRAWS, s.shape[1]))
    phi_cal = phi_hat[cal]

    def structural(sigma: float) -> float:
        return _batch_log_score(_structural(s, mu, phi_cal, sigma, eps), ln_final, mask)

    def walk(sigma: float) -> float:
        return _batch_log_score(_random_walk(s, sigma, eps), ln_final, mask)

    sigma, sigma_rw = _calibrate(structural), _calibrate(walk)
    return Cell(
        pool,
        int(tau),
        len(sub),
        len(cal),
        float(floored.sum() / max(usable.sum(), 1)),
        tuple(float(b) for b in beta),
        sigma,
        sigma_rw,
        structural(sigma),
        walk(sigma_rw),
    )


# ---------------------------------------------------------------- the model


def _refuse_holdout(days: Iterable[str]) -> None:
    held = sorted(d for d in set(days) if is_holdout_day(COUNTRY, date.fromisoformat(d)))
    if held:
        raise HoldoutDayError(f"{len(held)} {COUNTRY} holdout day(s), e.g. {held[0]} (D14)")


def _frame_days(frame: pd.DataFrame) -> list[str]:
    return sorted({_race_day(r) for r in frame["race_id"].unique()})


def _bucket(tau_min: float, grid: Sequence[int]) -> int:
    above = [m for m in grid if m >= tau_min]
    return min(above) if above else max(grid)


@dataclass(frozen=True, eq=False)
class Nowcast:
    """The structural nowcast; see the module docstring."""

    days: tuple[str, ...]
    grid: tuple[int, ...]
    growth: dict[str, PoolGrowth]
    cells: dict[tuple[str, int], Cell]
    features: bool = True
    fixed: tuple[float, float] | None = None  # (phi, sigma) of a known nowcast
    fit_id: str = ""

    # ---- fitting

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        grades: Mapping[str, str | None] | None = None,
        features: bool = True,
    ) -> Nowcast:
        """Fit on every race of `frame` (rows in compact.SCHEMA); refuses a holdout day."""
        _refuse_holdout(_frame_days(frame))
        cases, growth, _ = extract(frame, grades=grades)
        return cls.fit_cases(cases, growth, features=features)

    @classmethod
    def fit_cases(
        cls,
        cases: Cases,
        growth: Growth,
        *,
        features: bool = True,
        days: Iterable[str] | None = None,
    ) -> Nowcast:
        """Fit on `cases` and `growth`, or on their rows of `days` only (no copy of the cases)."""
        in_fit = np.ones(len(cases), dtype=bool)
        if days is not None:
            keep = list(days)
            in_fit, growth = _in_days(cases.day, keep), growth.select_days(keep)
        days = tuple(sorted(set(cases.day[in_fit].tolist()) | set(growth.day.tolist())))
        _refuse_holdout(days)
        grid = _grid()
        pool_growth = {}
        for pool in POOLS:
            fitted = PoolGrowth.fit(pool, growth)
            if fitted is not None:
                pool_growth[pool] = fitted
        cells: dict[tuple[str, int], Cell] = {}
        for pool in POOLS:
            if pool not in pool_growth:
                continue
            g = pool_growth[pool]
            for tau in grid:
                idx = np.flatnonzero(in_fit & (cases.pool == pool) & (cases.tau == tau))
                if not len(idx):
                    continue
                phi_hat = np.array(
                    [g.phi(cases.board_min[i], cases.venue[i], cases.race_no[i], cases.grade[i])
                     for i in idx]
                )  # fmt: skip
                cells[(pool, int(tau))] = _fit_cell(cases, idx, phi_hat, pool, tau, features)
        model = cls(days, grid, pool_growth, cells, features)
        return dataclasses.replace(model, fit_id=_digest(model))

    @classmethod
    def from_params(cls, *, phi: float, sigma: float) -> Nowcast:
        """The known nowcast: phi fixed, s_late = softmax(ln s_t + sigma eps), V_final = V_t/phi."""
        if not 0.0 < phi <= 1.0:
            raise ValueError(f"phi must be in (0, 1], got {phi}")
        if not sigma >= 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        model = cls((), _grid(), {}, {}, False, (float(phi), float(sigma)))
        return dataclasses.replace(model, fit_id=_digest(model))

    # ---- parameters at a view

    def cell(self, pool: str, tau_min: float) -> Cell:
        tau = _bucket(tau_min, self.grid)
        found = self.cells.get((pool, tau))
        if found is None:
            raise KeyError(f"no nowcast fit for {pool} at tau {tau}")
        return found

    def phi(self, view: RaceView, *, pool: str) -> float:
        """phi_t = V_t / V_final, from the pool-growth profile at the board's minutes to post."""
        if self.fixed is not None:
            return self.fixed[0]
        growth = self.growth.get(pool)
        if growth is None:
            raise KeyError(f"no pool-growth fit for {pool}")
        board = view.pools[pool]
        minutes = view.tau_min if board.board_min is None else board.board_min
        return growth.phi(minutes, view.venue, view.race_no, view.grade)

    def _v_final(self, view: RaceView, pool: str, phi: float, n: int) -> np.ndarray:
        total = view.pools[pool].total
        if total is not None and total > 0:
            return np.full(n, float(total) / phi)
        if self.fixed is not None:
            raise ValueError(f"{view.race_id} {pool}: no pool total on the board")
        return np.full(n, math.exp(self.growth[pool].log_v_final))

    def _board_shares(self, view: RaceView, pool: str) -> np.ndarray:
        s = np.asarray(view.pools[pool].shares, dtype=float)
        return s / s.sum()

    def _draw_shares(self, view: RaceView, pool: str, eps: np.ndarray) -> tuple[np.ndarray, float]:
        s = self._board_shares(view, pool)
        phi = self.phi(view, pool=pool)
        if self.fixed is not None:
            sigma, beta = self.fixed[1], np.zeros(len(FEATURES))
        else:
            cell = self.cell(pool, view.tau_min)
            sigma, beta = cell.sigma, np.asarray(cell.beta)
        mu = _board_features(view.pools[pool]) @ beta
        return _structural(s, mu, phi, sigma, eps), phi

    def draws(self, view: RaceView, *, pool: str, n: int, seed: int) -> Draws:
        """n draws of the final shares and V_final of `pool` from the board in `view`."""
        k = len(view.pools[pool].shares)
        eps = np.random.default_rng(seed).standard_normal((n, k))
        shares, phi = self._draw_shares(view, pool, eps)
        return Draws(shares=shares, v_final=self._v_final(view, pool, phi, n))

    def random_walk(self, view: RaceView, *, pool: str, n: int, seed: int) -> Draws:
        """The q_final = q_t baseline: softmax(ln s_t + sigma_rw eps)."""
        s = self._board_shares(view, pool)
        sigma = self.fixed[1] if self.fixed is not None else self.cell(pool, view.tau_min).sigma_rw
        eps = np.random.default_rng(seed).standard_normal((n, len(s)))
        phi = self.phi(view, pool=pool)
        return Draws(shares=_random_walk(s, sigma, eps), v_final=self._v_final(view, pool, phi, n))

    # ---- persistence

    def to_json(self) -> str:
        return json.dumps(
            {
                "model": MODEL_NAME,
                "fit_id": self.fit_id,
                "days": list(self.days),
                "grid": list(self.grid),
                "features": self.features,
                "fixed": None if self.fixed is None else list(self.fixed),
                "growth": [dataclasses.asdict(g) for g in self.growth.values()],
                "cells": [dataclasses.asdict(c) for c in self.cells.values()],
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> Nowcast:
        data = json.loads(text)
        if data.get("model") != MODEL_NAME:
            raise ValueError(f"not a {MODEL_NAME} fit: {data.get('model')!r}")
        growth = {}
        for g in data["growth"]:
            g = dict(g, knots=tuple(g["knots"]), profile=tuple(g["profile"]))
            growth[g["pool"]] = PoolGrowth(**g)
        cells = {}
        for c in data["cells"]:
            cell = Cell(**dict(c, beta=tuple(c["beta"])))
            cells[(cell.pool, cell.tau_min)] = cell
        return cls(
            tuple(data["days"]),
            tuple(data["grid"]),
            growth,
            cells,
            bool(data["features"]),
            None if data["fixed"] is None else tuple(data["fixed"]),
            data["fit_id"],
        )


def _digest(model: Nowcast) -> str:
    return hashlib.sha256(dataclasses.replace(model, fit_id="").to_json().encode()).hexdigest()[:12]


# ---------------------------------------------------------------- folds


@dataclass(frozen=True, eq=False)
class FoldFit:
    model: Nowcast
    train_days: tuple[str, ...]
    test_days: tuple[str, ...]
    train_end: date
    test_start: date
    test_end: date


def development_days(days: Iterable[str]) -> list[str]:
    """`days` without the holdout days, sorted."""
    return sorted(d for d in set(days) if not is_holdout_day(COUNTRY, date.fromisoformat(d)))


def fold_days(days: Sequence[str], scheme: FoldScheme) -> list[tuple[list[str], list[str], object]]:
    """(training days, test days, fold) of each walk-forward fold over development `days`."""
    days = list(days)
    folds = walk_forward([date.fromisoformat(d) for d in days], scheme=scheme)
    return [([days[i] for i in f.train_idx], [days[i] for i in f.test_idx], f) for f in folds]


def fit_folds(
    frame: pd.DataFrame,
    *,
    scheme: FoldScheme,
    grades: Mapping[str, str | None] | None = None,
    features: bool = True,
) -> list[FoldFit]:
    """One Nowcast per walk-forward fold of the frame's development days, fitted on that fold's
    training days only; holdout days are dropped first."""
    dev = development_days(_frame_days(frame))
    rows = frame[frame["race_id"].map(_race_day).isin(dev)]
    cases, growth, _ = extract(rows, grades=grades)
    out = []
    for train, test, fold in fold_days(dev, scheme):
        model = Nowcast.fit_cases(cases, growth, features=features, days=train)
        out.append(
            FoldFit(
                model, tuple(train), tuple(test), fold.train_end, fold.test_start, fold.test_end
            )
        )
    return out


# ---------------------------------------------------------------- scores

Forecaster = Callable[..., Draws]


def case_seed(seed: int, race_id: str, pool: str, tau_min: float) -> int:
    """The seed both forecasters get for one case: from `seed` and the case's identity."""
    key = zlib.crc32(f"{race_id}|{pool}|{float(tau_min):g}".encode())
    return int(np.random.SeedSequence([int(seed), key]).generate_state(1)[0])


def score_cases(
    forecasters: Mapping[str, Forecaster],
    cases: Cases,
    *,
    n: int,
    seed: int,
    keep_pit: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Per case and forecaster: log score, CRPS and KL on the final ln-shares (and, with
    `keep_pit`, one PIT row per combination). Every forecaster gets the same view, n and seed."""
    records, pits = [], []
    for i in range(len(cases)):
        view, pool = cases.view(i), str(cases.pool[i])
        s_final = cases.s_f[cases.rows(i)]
        seed_i = case_seed(seed, view.race_id, pool, view.tau_min)
        for name, forecast in forecasters.items():
            scores = case_scores(forecast(view, pool=pool, n=n, seed=seed_i).shares, s_final)
            records.append((i, name, scores["log_score"], scores["crps"], scores["kl"]))
            if keep_pit:
                pits.append((np.full(len(s_final), i), [name] * len(s_final), scores["pit"]))
    table = pd.DataFrame.from_records(
        records, columns=["case", "forecaster", "log_score", "crps", "kl"]
    )
    meta = pd.DataFrame(
        {
            "case": np.arange(len(cases)),
            "race_id": cases.race_id,
            "day": cases.day,
            "venue": cases.venue,
            "pool": cases.pool,
            "tau_min": cases.tau,
        }
    )
    table = table.merge(meta, on="case", how="left")
    pit_table = None
    if keep_pit:
        pit_table = pd.DataFrame(
            {
                "case": np.concatenate([p[0] for p in pits]) if pits else [],
                "forecaster": np.concatenate([p[1] for p in pits]) if pits else [],
                "pit": np.concatenate([p[2] for p in pits]) if pits else [],
            }
        )
    return table, pit_table


GAINS = {"log_score_gain": ("log_score", 1.0), "crps_gain": ("crps", -1.0), "kl_gain": ("kl", -1.0)}


def gain_table(
    scores: pd.DataFrame,
    candidate: str,
    baseline: str,
    *,
    n_boot: int,
    level: float,
    seed: int,
    cluster: str = "day",
) -> pd.DataFrame:
    """Per (pool, tau): the mean per-case gain of `candidate` over `baseline` (positive when the
    candidate is better) and one-sided bootstrap bounds resampling `cluster` ("day" or
    "meeting", venue x day)."""
    metrics = [metric for metric, _ in GAINS.values()]
    cand = scores[scores["forecaster"] == candidate].set_index("case")
    base = scores[scores["forecaster"] == baseline].set_index("case")[metrics]
    joined = cand.join(base, rsuffix="_baseline", how="inner")
    rows = []
    pool_order = {p: i for i, p in enumerate(POOLS)}
    groups = joined.groupby(["pool", "tau_min"], sort=False)
    for (pool, tau), g in sorted(groups, key=lambda kv: (pool_order.get(kv[0][0], 99), -kv[0][1])):
        units = g["day"] if cluster == "day" else g["day"] + ":" + g["venue"]
        clusters = pd.Series(units.to_numpy(), index=g["race_id"].to_numpy())
        row = {"pool": pool, "tau_min": float(tau), "n_cases": len(g), "n_days": g["day"].nunique()}
        for gain, (metric, sign) in GAINS.items():
            diff = sign * (g[metric].to_numpy(float) - g[f"{metric}_baseline"].to_numpy(float))
            interval = paired_bootstrap(
                pd.Series(diff, index=g["race_id"].to_numpy()),
                clusters,
                n_boot=n_boot,
                level=level,
                seed=seed,
            )
            row |= {gain: interval.mean, f"{gain}_lo": interval.lo, f"{gain}_hi": interval.hi}
            row[f"{metric}_candidate"] = float(g[metric].mean())
            row[f"{metric}_baseline"] = float(g[f"{metric}_baseline"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def compare(
    candidate: Forecaster,
    baseline: Forecaster,
    frame: pd.DataFrame,
    *,
    n: int,
    seed: int,
    n_boot: int,
    level: float,
    grades: Mapping[str, str | None] | None = None,
) -> pd.DataFrame:
    """The candidate's per-case gain over the baseline per (pool, tau_min) of the decision grid,
    with one-sided bootstrap bounds that resample race days; refuses a holdout day."""
    _refuse_holdout(_frame_days(frame))
    cases, _, _ = extract(frame, grades=grades)
    scores, _ = score_cases({"candidate": candidate, "baseline": baseline}, cases, n=n, seed=seed)
    return gain_table(scores, "candidate", "baseline", n_boot=n_boot, level=level, seed=seed)


# ---------------------------------------------------------------- the v0 draws interface


@dataclass(frozen=True, eq=False)
class ShadowNowcast:
    """A Nowcast behind #116's draws interface (`v0.shadow_values.NowcastModel`): odds draws for
    one pool of a `RaceBoard` from shared per-runner shocks. WIN draws keep the board's sum of
    1/O; PLACE and Wide print the stake-back-share figures of the drawn money, floored at the
    minimum dividend. A RaceBoard carries no lag boards, so the velocity features are 0 here, and
    a Wide pair's shock is (eps_a + eps_b) / sqrt 2 as in v0."""

    model: Nowcast

    @property
    def fit_id(self) -> str:
        return self.model.fit_id

    def draws(
        self, board: RaceBoard, *, pool: str, tau_min: int, shocks: Mapping[int, np.ndarray]
    ) -> PoolDraws:
        if pool == WIN:
            keys = sorted(board.win)
            lo, hi = np.array([board.win[h] for h in keys], float), None
        elif pool in (PLACE, WIDE):
            prices = board.place if pool == PLACE else board.wide
            keys = sorted(prices)
            lo = np.array([prices[key][0] for key in keys], float)
            hi = np.array([np.nan if prices[key][1] is None else prices[key][1] for key in keys])
        else:
            raise ValueError(f"no nowcast for pool {pool!r}")
        combinations = tuple(f"{k[0]},{k[1]}" if pool == WIDE else str(k) for k in keys)
        n = len(np.asarray(next(iter(shocks.values()))))
        if tau_min <= 0:
            odds = np.tile(lo, (n, 1))
            return PoolDraws(pool, combinations, odds, None if hi is None else np.tile(hi, (n, 1)))
        shares, _ = pool_shares(pool, lo)
        if shares is None:
            raise KeyError(f"{pool} is not sold for {len(lo)} runners")
        minutes = float(tau_min)
        if board.post_utc is not None:
            minutes = (board.post_utc - board.capture_utc).total_seconds() / 60.0
        _, _, venue, race_no = board.race_id.split(":")
        view = RaceView(
            board.race_id,
            venue,
            int(race_no),
            None,
            float(tau_min),
            {pool: PoolBoard(combinations, shares, board.totals.get(pool), None, None, minutes)},
        )
        eps = np.empty((n, len(keys)))
        for j, key in enumerate(keys):
            if pool == WIDE:
                eps[:, j] = (np.asarray(shocks[key[0]]) + np.asarray(shocks[key[1]])) / math.sqrt(2)
            else:
                eps[:, j] = np.asarray(shocks[key])
        drawn, _ = self.model._draw_shares(view, pool, eps)
        if pool == WIN:
            return PoolDraws(pool, combinations, 1.0 / (drawn * float((1.0 / lo).sum())), None)
        places = _pool_config(pool).places_for(board.starters)
        payout = 1.0 - float(_pool_config(pool).takeout)
        low, high = stake_back_odds(drawn, places, payout)
        floor = _floor(pool)
        return PoolDraws(pool, combinations, np.maximum(low, floor), np.maximum(high, floor))


def load_shadow_nowcast(text: str) -> ShadowNowcast:
    """A saved Nowcast fit (`Nowcast.to_json`) behind the v0 draws interface."""
    return ShadowNowcast(Nowcast.from_json(text))
