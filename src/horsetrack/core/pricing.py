"""Pari-mutuel pool maths: facts about the pools, not model or betting choices.

Ported from HKTrack (betting/market.py, betting/pricing.py). Nothing here turns odds into
probabilities with a chosen method; display rules are arguments, never defaults.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

Detruncate = Callable[[float], float]

_EPS = 1e-9
_RULE_KEYS = ("threshold", "detruncate_below", "detruncate_at_or_above")


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _as_float_array(values: Any) -> np.ndarray:
    """1-D float array from any array-like; None and anything non-numeric become NaN."""
    if not hasattr(values, "__len__"):
        values = list(values) if isinstance(values, Iterable) else [values]
    try:
        return np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([_to_float(v) for v in values], dtype=float)


def clean_odds(odds: Any, detruncate: Detruncate | None = None) -> np.ndarray:
    """Decimal odds (stake included) as floats; missing, non-finite or below 1.0 -> NaN.

    `detruncate` maps a displayed figure to its expected true value; it is applied to valid
    figures only, after the sub-one check on the displayed figure. The input is never modified.
    """
    arr = _as_float_array(odds).copy()
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(arr) & (arr >= 1.0)
    arr[~ok] = np.nan
    if detruncate is not None and ok.any():
        arr[ok] = [float(detruncate(float(o))) for o in arr[ok]]
    return arr


def inverse_odds(odds: Any) -> np.ndarray:
    """1 / odds per valid price, NaN elsewhere. Not normalised: the shares keep the overround."""
    return 1.0 / clean_odds(odds)


def overround(odds: Any, *, winners: int = 1) -> float:
    """sum(1 / odds) / winners - 1 over valid prices; NaN without valid odds or winners < 1.

    `winners` is the number of paying combinations in an equal-split pool (places paid; 3 for
    any-2-of-top-3). De-truncate through `clean_odds` first if the board floors its figures.
    """
    arr = clean_odds(odds)
    ok = ~np.isnan(arr)
    if not ok.any() or winners < 1:
        return math.nan
    return float((1.0 / arr[ok]).sum() / winners - 1.0)


def floor_bound(odds: Any, floor: float) -> bool:
    """True when any valid price sits at or under the minimum dividend `floor`: that runner's
    real pool share is then unknown."""
    arr = clean_odds(odds)
    ok = ~np.isnan(arr)
    return bool((arr[ok] <= floor + _EPS).any())


def stake_back_share_gross(share: Any, lost: Any, *, rate: float, winners: int) -> Any:
    """The stake-back-share identity before truncation and floor, element-wise on arrays:
    rate x (1 + lost / (winners x share)) per unit staked, `lost` the losers' share of the pool
    (1 less the winning combinations' shares) and `winners` the P paying combinations. The rate
    covers the returned stake too: (W + D / P) x R in units of the winning stake W."""
    return rate * (1.0 + lost / (winners * share))


def stake_back_share_dividend(
    share: float,
    co_shares: Sequence[float],
    *,
    rate: float,
    unit: float,
    dividend_step: float,
    min_dividend: float | None,
) -> float:
    """Dividend per `unit` on a winning ticket holding `share` of the pool, co-winners `co_shares`.

    unit x rate x (1 + D / (P x share)), D = the losers' share, P = 1 + len(co_shares); truncated
    down to `dividend_step`, never below `min_dividend`. NaN when the ticket holds no share.
    """
    if not share > 0:
        return math.nan
    lost = 1.0 - share
    for c in co_shares:
        lost -= c
    raw = stake_back_share_gross(share, lost, rate=unit * rate, winners=1 + len(co_shares))
    paid = float(math.floor(raw / dividend_step + _EPS) * dividend_step)
    return paid if min_dividend is None else max(float(min_dividend), paid)


def detruncate(odds: float, rule: Mapping[str, float]) -> float:
    """A displayed (floored) figure plus its expected truncation under `rule`.

    `rule` has the config's `odds_display` shape: `detruncate_below` is added under `threshold`,
    `detruncate_at_or_above` from it up; other keys are ignored. A missing key raises ValueError.
    """
    missing = [k for k in _RULE_KEYS if k not in rule]
    if missing:
        raise ValueError(f"detruncate rule is missing {', '.join(missing)}")
    bump = rule["detruncate_below"] if odds < rule["threshold"] else rule["detruncate_at_or_above"]
    return float(odds + bump)
