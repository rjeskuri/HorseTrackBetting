"""T-2 cross-pool features from the Win and Quinella boards (#434).

Input: one day's odds_timeseries rows (`logger.compact.SCHEMA`, `docs/specs/odds-log.md`), held
in memory; nothing here reads a file, the store, the registry or the holdout.

- The board at t (`boards_at`): rows with `minutes_to_post >= minutes_before`, the latest per
  (race, pool, combination) by availability (`fetched_utc`, else `snapshot_utc`, as
  `board.snapshot.board_at`). Pool-total rows and pools other than WIN and QUINELLA are ignored. A
  runner whose latest WIN row is scratched ("SCR") or unpriced leaves both boards. A pool's
  capture time is the oldest availability among the rows its board shows (the WIN SCR rows
  included), as `board.snapshot`'s `capture_utc`; None when the pool has no board at t.
- Shares: normalised 1/odds. Harville pairs: P({i, j} first two) from the Win shares through
  `models.v0.ordering._order_probs` with second-place weights p ** lambda2.
- Features (`cross_features`, one row per Win starter): q_top2_t and h_top2_t, the runner's
  Quinella and Harville top-2 support (the sum of the pair shares holding it); x_cross =
  ln(q_top2_t / h_top2_t); fav_support, the favourite's x_cross; kl_race = sum q ln(q / h);
  gap_s = quinella_utc - win_utc in seconds. `pair_features` gives ln(q_pair / h_pair) per pair.
  Quinella pairs naming a runner off the Win board (scratched, unpriced or with no Win row) are
  dropped before the Quinella capture time is taken. A pair whose latest row at t has no price
  (blank, or a JRA-VAN `0000` / `----` / `****` quote) is off the board at t, even when an older
  row priced it. A race whose Quinella board at t misses any starter pair, or has none, gets NaN
  in every Quinella-dependent column (q_top2_t, x_cross, fav_support, kl_race, gap_s; q_pair and
  log_ratio in `pair_features`): its shares cannot be normalised over the field.
- Nullable and PyArrow-backed columns read as plain ones: a missing value never matches a mask.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from horsetrack.countries.japan.models.v0.ordering import _order_probs

WIN = "WIN"
QUINELLA = "QUINELLA"
POOL_TOTAL = "__pool__"
SCRATCHED = "SCR"
_KEY = ["race_id", "pool", "combination"]


@dataclass(frozen=True)
class RaceBoards:
    """One race's Win and Quinella boards at t, with each pool's capture time."""

    win: dict[int, float] = field(default_factory=dict)
    quinella: dict[tuple[int, int], float] = field(default_factory=dict)
    win_utc: datetime | None = None
    quinella_utc: datetime | None = None


def _mask(cond: pd.Series) -> np.ndarray:
    """A boolean mask in which a missing comparison (nullable or PyArrow dtypes) is False."""
    return cond.fillna(False).to_numpy(dtype=bool)


def _utc(column: pd.Series) -> pd.Series:
    """A datetime64[ns, UTC] column from a plain, nullable or PyArrow timestamp column."""
    return pd.to_datetime(column, utc=True).astype("datetime64[ns, UTC]")


def _latest_rows(day: pd.DataFrame, minutes_before: float) -> pd.DataFrame:
    minutes = day["minutes_to_post"].to_numpy(dtype=float, na_value=np.nan)
    keep = (
        _mask(day["pool"].isin((WIN, QUINELLA)))
        & _mask(day["combination"] != POOL_TOTAL)
        & (minutes >= minutes_before)
    )
    rows = day.loc[keep, [*_KEY, "snapshot_utc", "fetched_utc", "odds", "status_raw"]].copy()
    for column in _KEY:
        rows[column] = rows[column].astype(object)
    rows["odds"] = rows["odds"].to_numpy(dtype=float, na_value=np.nan)
    rows["available"] = _utc(rows["fetched_utc"]).fillna(_utc(rows["snapshot_utc"]))
    rows = rows.dropna(subset=["available"])
    rows = rows.sort_values("available", kind="mergesort")
    return rows.drop_duplicates(_KEY, keep="last")


def _priced(rows: pd.DataFrame) -> pd.Series:
    odds = rows["odds"].to_numpy(dtype=float)
    return pd.Series(np.isfinite(odds) & (odds > 0), index=rows.index)


def _first_utc(rows: pd.DataFrame) -> dict[str, datetime]:
    return rows.groupby("race_id")["available"].min().to_dict()


def boards_at(day: pd.DataFrame, *, minutes_before: float = 2.0) -> dict[str, RaceBoards]:
    """Each race's Win and Quinella boards at `minutes_before` minutes to post."""
    latest = _latest_rows(day, minutes_before)

    win = latest[latest["pool"] == WIN].copy()
    win["horse_no"] = win["combination"].astype(int)
    scratched = pd.Series(_mask(win["status_raw"] == SCRATCHED), index=win.index)
    starting = _priced(win) & ~scratched
    off = pd.MultiIndex.from_frame(win.loc[~starting, ["race_id", "horse_no"]])
    win_board = win[starting]
    win_shown = win[starting | scratched]

    quinella = latest[(latest["pool"] == QUINELLA) & _priced(latest)].copy()
    runners = quinella["combination"].str.split(",", expand=True)
    if quinella.empty:
        quinella["a"] = quinella["b"] = pd.Series(dtype=int)
    else:
        first, second = runners[0].astype(int), runners[1].astype(int)
        quinella["a"] = np.minimum(first, second)
        quinella["b"] = np.maximum(first, second)
    named_off = pd.MultiIndex.from_arrays([quinella["race_id"], quinella["a"]]).isin(off) | (
        pd.MultiIndex.from_arrays([quinella["race_id"], quinella["b"]]).isin(off)
    )
    # Where a race has Win starters, its Quinella board holds only their pairs.
    starters = pd.MultiIndex.from_frame(win_board[["race_id", "horse_no"]])
    has_starters = quinella["race_id"].isin(set(win_board["race_id"])).to_numpy()
    both_start = pd.MultiIndex.from_arrays([quinella["race_id"], quinella["a"]]).isin(
        starters
    ) & pd.MultiIndex.from_arrays([quinella["race_id"], quinella["b"]]).isin(starters)
    quinella_board = quinella[~named_off & (~has_starters | both_start)]

    win_utc = _first_utc(win_shown)
    quinella_utc = _first_utc(quinella_board)
    win_by_race = {
        rid: dict(zip(g["horse_no"].astype(int), g["odds"].astype(float), strict=True))
        for rid, g in win_board.groupby("race_id")
    }
    quinella_by_race = {
        rid: {(int(a), int(b)): float(o) for a, b, o in zip(g["a"], g["b"], g["odds"], strict=True)}
        for rid, g in quinella_board.groupby("race_id")
    }
    races = sorted(set(latest["race_id"]))
    return {
        rid: RaceBoards(
            win=win_by_race.get(rid, {}),
            quinella=quinella_by_race.get(rid, {}),
            win_utc=win_utc.get(rid),
            quinella_utc=quinella_utc.get(rid),
        )
        for rid in races
    }


def _shares(odds: dict) -> dict:
    inv = {k: 1.0 / v for k, v in odds.items()}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}


def win_shares(win: dict[int, float]) -> dict[int, float]:
    """The Win board's normalised 1/odds."""
    return _shares(win)


def quinella_shares(quinella: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    """The Quinella board's normalised 1/odds over its pairs."""
    return _shares(quinella)


def harville_pairs(p: dict[int, float], *, lambda2: float = 1.0) -> dict[tuple[int, int], float]:
    """P({i, j} first two, either order) from Win shares; second-place weights p ** lambda2."""
    horses = sorted(p)
    if len(horses) < 2:
        return {}
    probs = np.array([p[h] for h in horses], dtype=float)
    first_two = _order_probs(probs, (lambda2, 1.0), 2)
    either = first_two + first_two.T
    return {
        (horses[i], horses[j]): float(either[i, j])
        for i, j in itertools.combinations(range(len(horses)), 2)
    }


def _starter_quinella(boards: RaceBoards) -> dict[tuple[int, int], float]:
    """Quinella shares over the Win starters' pairs; {} unless every starter pair is priced."""
    pairs = list(itertools.combinations(sorted(boards.win), 2))
    if not pairs or any(pr not in boards.quinella for pr in pairs):
        return {}
    return quinella_shares({pr: boards.quinella[pr] for pr in pairs})


def pair_features(boards: RaceBoards, *, lambda2: float = 1.0) -> pd.DataFrame:
    """One row per starter pair: q_pair, h_pair and log_ratio = ln(q_pair / h_pair)."""
    h = harville_pairs(win_shares(boards.win), lambda2=lambda2) if boards.win else {}
    q = _starter_quinella(boards)
    frame = pd.DataFrame(
        {
            "pair": list(h),
            "q_pair": [q.get(pr, np.nan) for pr in h],
            "h_pair": list(h.values()),
        }
    )
    frame["log_ratio"] = np.log(frame["q_pair"].astype(float) / frame["h_pair"].astype(float))
    return frame


def _top2(pairs: dict[tuple[int, int], float], horses: list[int]) -> np.ndarray:
    out = dict.fromkeys(horses, 0.0)
    for (a, b), v in pairs.items():
        out[a] += v
        out[b] += v
    return np.array([out[h] for h in horses])


def cross_features(boards: RaceBoards, *, lambda2: float = 1.0) -> pd.DataFrame:
    """One row per Win starter: the Win share and its Quinella-vs-Harville top-2 disagreement."""
    horses = sorted(boards.win)
    p = win_shares(boards.win) if horses else {}
    h = harville_pairs(p, lambda2=lambda2)
    q = _starter_quinella(boards)
    p_win = np.array([p[k] for k in horses], dtype=float)
    h_top2 = _top2(h, horses)
    nan = np.full(len(horses), np.nan)
    if q:
        q_top2 = _top2(q, horses)
        with np.errstate(divide="ignore"):
            x_cross = np.log(q_top2 / h_top2)
        kl = float(sum(v * np.log(v / h[pr]) for pr, v in q.items()))
    else:
        q_top2, x_cross, kl = nan, nan.copy(), np.nan
    fav = int(np.argmax(p_win)) if horses else -1  # ties: the lowest horse_no
    is_fav = np.arange(len(horses)) == fav
    gap = (
        (pd.Timestamp(boards.quinella_utc) - pd.Timestamp(boards.win_utc)).total_seconds()
        if q and boards.win_utc is not None and boards.quinella_utc is not None
        else np.nan
    )
    return pd.DataFrame(
        {
            "horse_no": np.array(horses, dtype=int),
            "p_win_t": p_win,
            "q_top2_t": q_top2,
            "h_top2_t": h_top2,
            "x_cross": x_cross,
            "is_fav": is_fav.astype(float),
            "fav_support": np.full(len(horses), x_cross[fav] if horses else np.nan),
            "kl_race": np.full(len(horses), kl),
            "gap_s": np.full(len(horses), gap),
        }
    )
