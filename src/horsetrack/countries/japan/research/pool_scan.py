"""The final-board return scan of every JRA pool by segment: N1's maths (ticket #429).

A flat box stakes one unit on every priced combination of a pool's final board; a combination
returns its dividend per 100 yen over 100 when it paid, else 0. Per race and odds band the scan
sums the combinations staked, their return and their payout rate (1 - takeout), so a segment's
flat rho = sum(return) / sum(payout rate) is 1 when the segment is priced like the pool. The same
rows grouped by (segment levels, odds band, odds) with a count and wins are the input of
`core.eval.segment_roi.fit_segment_roi(..., count_col='count')` (#78).

The functions take frames and return frames; nothing here reads the store, the registry or the
holdout. `combination_returns` takes the store's `final_odds` rows of one pool and its `dividends`
rows (`docs/specs/store.md`), a day's or a year's races at once, and every step is vectorised over
those rows. Place and Wide (QUINELLA_PLACE) band on the low end of the final range (`odds`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral

import numpy as np
import pandas as pd

POOLS = ("WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE", "FORECAST", "TRIO")
POOL_ROW = "__pool__"  # the board's per-pool row (`jravan/odds.py`), not a combination
KEYS = ["race_id", "combination"]

# `races.race_class` (`jravan/races.py` CONDITION_CLASSES and GRADES) -> class group
CLASS_GROUPS = {
    "DEBUT": "newcomer",
    "MAIDEN": "maiden",
    "C1": "conditions",
    "C2": "conditions",
    "C3": "conditions",
    "OP": "open",
    "L": "open",
    "G1": "graded",
    "G2": "graded",
    "G3": "graded",
}
MIN_STARTERS, MAX_STARTERS = 5, 18


def class_group(race_class: str) -> str:
    """The class group of a `races.race_class` value; ValueError for a null or unknown class."""
    if not isinstance(race_class, str) or race_class not in CLASS_GROUPS:
        raise ValueError(f"unknown race class {race_class!r} (have {sorted(CLASS_GROUPS)})")
    return CLASS_GROUPS[race_class]


def _finite(name: str, value: float, low: float = -math.inf) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, not {value!r}") from None
    if not (math.isfinite(value) and value >= low):
        raise ValueError(f"{name} must be finite and at least {low}, not {value!r}")
    return value


def race_cells(
    *,
    race_class: str,
    field_size: int,
    fav_odds: float,
    pool_total: float,
    pool_edges: tuple[float, float],
    surface: str,
    venue: str,
) -> frozenset[str]:
    """The cells a race is in: 'all' and one `segment:level` of each segment (module docstring).

    Field bands 5-10, 11-14 and 15-18 and favourite bands le1.9, 2.0-2.9, 3.0-4.9 and ge5.0 (lower
    edge inclusive) as in #131's `exacta_study.segment_cells`; the pool size is low below the
    first edge, mid from it to the second and high from the second.
    """
    group = class_group(race_class)
    if not isinstance(field_size, Integral) or not MIN_STARTERS <= field_size <= MAX_STARTERS:
        raise ValueError(f"field size {field_size!r} is outside {MIN_STARTERS}-{MAX_STARTERS}")
    field = "5-10" if field_size <= 10 else "11-14" if field_size <= 14 else "15-18"
    fav_odds = _finite("fav_odds", fav_odds, 1.0)
    if fav_odds < 2.0:
        fav = "le1.9"
    elif fav_odds < 3.0:
        fav = "2.0-2.9"
    elif fav_odds < 5.0:
        fav = "3.0-4.9"
    else:
        fav = "ge5.0"
    pool_total = _finite("pool_total", pool_total, 0.0)
    if len(pool_edges) != 2:
        raise ValueError(f"pool_edges must be two edges, not {pool_edges!r}")
    low, high = (_finite("pool_edges", e) for e in pool_edges)
    if low > high:
        raise ValueError(f"pool_edges must be ascending, not {pool_edges!r}")
    size = "low" if pool_total < low else "mid" if pool_total < high else "high"
    return frozenset(
        {
            "all",
            f"class:{group}",
            f"field:{field}",
            f"fav:{fav}",
            f"pool_size:{size}",
            f"surface:{surface}",
            f"venue:{venue}",
        }
    )


def _band_labels(edges: Sequence[float]) -> list[str]:
    """One label per interval [e_i, e_i+1) of the ascending edges, below the first edge first."""
    edges = [_finite("band edge", e) for e in edges]
    if not edges or any(a >= b for a, b in zip(edges, edges[1:], strict=False)):
        raise ValueError(f"band edges must be one or more strictly ascending values, not {edges}")
    inner = [f"{a:g}-{b:g}" for a, b in zip(edges, edges[1:], strict=False)]
    return [f"lt{edges[0]:g}", *inner, f"ge{edges[-1]:g}"]


def _bands(odds: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """`odds_band` of every odds value at once."""
    labels = np.array(_band_labels(edges), dtype=object)
    odds = np.asarray(odds, dtype=float)
    if not np.isfinite(odds).all():
        raise ValueError("odds must be finite to be banded")
    return labels[np.searchsorted(np.asarray(edges, dtype=float), odds, side="right")]


def odds_band(odds: float, edges: Sequence[float]) -> str:
    """The odds band of `odds`: the interval [e_i, e_i+1) of the ascending `edges` it is in."""
    return str(_bands(np.array([_finite("odds", odds)]), edges)[0])


def _columns(frame: pd.DataFrame, name: str, needed: Sequence[str]) -> None:
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} lacks column(s): {', '.join(missing)}")


def combination_returns(board: pd.DataFrame, paid: pd.DataFrame, *, pool: str) -> pd.DataFrame:
    """The return at the final dividend of every priced combination of `pool` on `board`.

    `board` is `final_odds` rows (only those of `pool` are read), `paid` is `dividends` rows of
    any pools; both may hold many races. A priced combination is a board row other than the pool
    row with a non-null `odds`; it matches `paid` on (race_id, combination) within `pool`, its
    text as stored. One row per priced combination: race_id, combination, odds, won (it paid),
    ret (dividend / 100 when it paid, else 0) and refunded (`paid` has an `is_refund` row for it).
    In a dead heat every paying row pays its own combination.
    """
    if pool not in POOLS:
        raise ValueError(f"unknown pool {pool!r} (have {POOLS})")
    _columns(board, "board", ["race_id", "pool", "combination", "odds"])
    _columns(paid, "paid", ["race_id", "pool", "combination", "dividend", "is_refund"])
    mine = board["pool"].to_numpy() == pool
    priced = board.loc[mine & (board["combination"].to_numpy() != POOL_ROW), [*KEYS, "odds"]]
    priced = priced[priced["odds"].notna()].reset_index(drop=True)
    if priced.duplicated(KEYS).any():
        raise ValueError(f"board has a {pool} combination twice in a race")

    paid = paid[paid["pool"].to_numpy() == pool]
    refund = paid["is_refund"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    paying = paid.loc[~refund, [*KEYS, "dividend"]]
    if paying.duplicated(KEYS).any():
        raise ValueError(f"paid has a {pool} combination paid twice in a race")
    dividend = pd.to_numeric(paying["dividend"], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(dividend) & (dividend > 0)).all():
        raise ValueError(f"paid has a {pool} dividend that is not a positive number")

    out = priced.merge(paying, on=KEYS, how="left")
    won = out["dividend"].notna().to_numpy()
    refunds = pd.MultiIndex.from_frame(paid.loc[refund, KEYS])
    return pd.DataFrame(
        {
            "race_id": out["race_id"],
            "combination": out["combination"],
            "odds": out["odds"].astype(float),
            "won": won,
            "ret": out["dividend"].fillna(0.0).to_numpy(dtype=float) / 100,
            "refunded": pd.MultiIndex.from_frame(out[KEYS]).isin(refunds),
        }
    )


def _rate(payout_rate: float) -> float:
    rate = _finite("payout_rate", payout_rate)
    if not 0 < rate <= 1:
        raise ValueError(f"payout_rate must be in (0, 1], not {payout_rate!r}")
    return rate


def race_returns(
    rows: pd.DataFrame, *, payout_rate: float, band_edges: Sequence[float]
) -> pd.DataFrame:
    """Per (race_id, band) the flat-box sums of one pool's `combination_returns` rows.

    n_staked, ret_sum and rate_sum (payout_rate per combination staked) cover the unrefunded
    combinations; n_refunded counts the refunded ones. Flat rho is ret_sum.sum() / rate_sum.sum().
    """
    rate = _rate(payout_rate)
    _columns(rows, "rows", ["race_id", "odds", "ret", "refunded"])
    refunded = rows["refunded"].astype(bool).to_numpy()
    frame = pd.DataFrame(
        {
            "race_id": rows["race_id"].to_numpy(),
            "band": _bands(rows["odds"].to_numpy(dtype=float), band_edges),
            "n_staked": (~refunded).astype(int),
            "ret_sum": np.where(refunded, 0.0, rows["ret"].to_numpy(dtype=float)),
            "n_refunded": refunded.astype(int),
        }
    )
    out = frame.groupby(["race_id", "band"], as_index=False, sort=True, observed=True)[
        ["n_staked", "ret_sum", "n_refunded"]
    ].sum()
    out.insert(4, "rate_sum", rate * out["n_staked"])
    return out


def grouped_bets(
    rows: pd.DataFrame, *, by: Sequence[str], payout_rate: float, band_edges: Sequence[float]
) -> pd.DataFrame:
    """`combination_returns` rows with the segment columns `by` as grouped bets for #78.

    One row per distinct (*by, band, odds) of the unrefunded rows with takeout = 1 - payout_rate,
    count (its combinations) and won (the wins among them): the input of
    `fit_segment_roi(..., odds_band_col='band', takeout_col='takeout', count_col='count')`.
    """
    by = list(by)
    rate = _rate(payout_rate)
    reserved = {"band", "odds", "takeout", "count", "won"}
    if len(set(by)) != len(by) or reserved & set(by):
        raise ValueError(f"by must name distinct columns other than {sorted(reserved)}")
    _columns(rows, "rows", [*by, "odds", "won", "refunded"])
    live = rows.loc[~rows["refunded"].astype(bool).to_numpy(), [*by, "odds", "won"]]
    if live[by].isna().any().any():
        raise ValueError(f"rows has missing segment levels in {', '.join(by)}")
    odds = live["odds"].to_numpy(dtype=float)
    frame = pd.DataFrame(
        {
            **{c: live[c].to_numpy() for c in by},
            "band": _bands(odds, band_edges),
            "odds": odds,
            "won": live["won"].astype(bool).to_numpy().astype(int),
        }
    )
    out = frame.groupby([*by, "band", "odds"], as_index=False, sort=True, observed=True).agg(
        count=("won", "size"), won=("won", "sum")
    )
    out.insert(len(by) + 2, "takeout", 1 - rate)
    return out
