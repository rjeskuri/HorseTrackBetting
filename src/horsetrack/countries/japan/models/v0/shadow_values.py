"""Shadow values E[p * O_final] over nowcast draws (C6 decision layer, shadow mode).

A value is the gross return per 1 yen at the final price: 1.0 is break-even. Nothing here names a
stake or places a bet; every runner and every Wide pair on the board gets a value, with no
selection, and the board at t enters only through the nowcast.

Per race one shock matrix (n_draws x runners) is drawn from `seed` and the same columns reach the
nowcast for the WIN, PLACE and Wide pools. For draw m, q = (1/O_WIN) / sum(1/O_WIN) over the
board's runners and p = win.predict(q, board.context()).
- WIN runner i: value_mean = mean_m p_i O_i (never mean(p) * mean(O), so the covariance of p and O
  is kept); value_q05 / value_q95 are the 5% and 95% points of p_i O_i over the draws.
- PLACE runner i: P = ordering.place_probs(p, places); value_at_low = mean_m P_i low_i and
  value_at_high = mean_m P_i high_i. A Wide pair works the same way with
  ordering.pair_probs(p, places) at the Wide places. Places come from the jra config; 0 places,
  or no ordering model, leaves both values None.

v0 limits:
- no own stake in the dividend: the final price is taken as unmoved by our ticket;
- no co-placer-exact point value: Place and Wide get a low-end and a high-end value from the
  range figures, not the dividend given who else places;
- no parameter uncertainty from refits of the win, ordering or nowcast models (#100 adds it);
- the Win-Place covariance comes only through the shared shocks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from horsetrack.core.config import load_source
from horsetrack.countries.japan.board.snapshot import PLACE, WIDE, WIN, RaceBoard, RaceContext

QUANTILES = (0.05, 0.95)


class WinModel(Protocol):
    def predict(self, q: np.ndarray, race: RaceContext | None = None) -> np.ndarray: ...


class OrderingModel(Protocol):
    def place_probs(self, p: np.ndarray, places: int) -> np.ndarray: ...

    def pair_probs(self, p: np.ndarray, places: int) -> np.ndarray: ...


@dataclass(frozen=True, eq=False)
class PoolDraws:
    """Final-price draws of one pool: `odds` is n_draws x len(combinations), column j for
    combinations[j]; `odds_hi` the same shape with NaN where there is no high figure, or None.
    For PLACE and Wide `odds` is the low figure."""

    pool: str
    combinations: tuple[str, ...]
    odds: np.ndarray
    odds_hi: np.ndarray | None


class NowcastModel(Protocol):
    def draws(
        self, board: RaceBoard, *, pool: str, tau_min: int, shocks: Mapping[int, np.ndarray]
    ) -> PoolDraws: ...


@dataclass(frozen=True)
class ShadowModels:
    win: WinModel
    ordering: OrderingModel | None
    nowcast: NowcastModel
    ids: dict[str, str]


@dataclass(frozen=True)
class ShadowValue:
    race_id: str
    pool: str
    combination: str
    tau_min: int
    board_odds: float
    board_odds_hi: float | None
    value_mean: float | None
    value_q05: float | None
    value_q95: float | None
    value_at_low: float | None
    value_at_high: float | None


def _shocks(runners: list[int], n_draws: int, seed: int) -> dict[int, np.ndarray]:
    """One standard normal column per runner, read-only so no pool's nowcast can alter another's."""
    matrix = np.random.default_rng(seed).standard_normal((n_draws, len(runners)))
    matrix.setflags(write=False)
    return {h: matrix[:, j] for j, h in enumerate(runners)}


def _columns(draws: PoolDraws, wanted: list[str], n_draws: int) -> tuple[np.ndarray, np.ndarray]:
    """(low, high) as n_draws x len(wanted), aligned to `wanted` by combination string."""
    odds = np.asarray(draws.odds, dtype=float)
    if odds.shape != (n_draws, len(draws.combinations)):
        raise ValueError(
            f"{draws.pool} draws have shape {odds.shape}, expected "
            f"({n_draws}, {len(draws.combinations)})"
        )
    hi = np.full_like(odds, np.nan) if draws.odds_hi is None else np.asarray(draws.odds_hi, float)
    if hi.shape != odds.shape:
        raise ValueError(f"{draws.pool} odds_hi has shape {hi.shape}, expected {odds.shape}")
    index = {c: j for j, c in enumerate(draws.combinations)}
    missing = [c for c in wanted if c not in index]
    if missing:
        raise ValueError(f"{draws.pool} draws lack combinations {missing}")
    cols = [index[c] for c in wanted]
    return odds[:, cols], hi[:, cols]


def _mean_or_none(prob: np.ndarray, odds: np.ndarray) -> float | None:
    """mean_m prob * odds; None unless every draw has a figure."""
    if np.any(np.isnan(odds)):
        return None
    return float(np.mean(prob * odds))


def _key(a: int, b: int) -> str:
    a, b = sorted((a, b))
    return f"{a},{b}"


def shadow_values(
    board: RaceBoard, *, tau_min: int, models: ShadowModels, n_draws: int, seed: int
) -> list[ShadowValue]:
    """One ShadowValue per WIN runner, PLACE runner and Wide pair on `board`."""
    if n_draws < 1:
        raise ValueError(f"n_draws must be positive, got {n_draws}")
    runners = sorted(board.win)
    shocks = _shocks(runners, n_draws, seed)
    context = board.context()

    def draws(pool: str) -> PoolDraws:
        return models.nowcast.draws(board, pool=pool, tau_min=tau_min, shocks=shocks)

    def row(pool: str, combination: str, figures: tuple[float, float | None], **values):
        fields = dict.fromkeys(
            ("value_mean", "value_q05", "value_q95", "value_at_low", "value_at_high")
        )
        fields.update(values)
        return ShadowValue(board.race_id, pool, combination, tau_min, *figures, **fields)

    out: list[ShadowValue] = []
    if not runners:
        return out

    win_odds, _ = _columns(draws(WIN), [str(h) for h in runners], n_draws)
    p = np.empty_like(win_odds)
    for m in range(n_draws):
        inv = 1.0 / win_odds[m]
        p_m = np.asarray(models.win.predict(inv / inv.sum(), context), dtype=float)
        if p_m.shape != (len(runners),):
            raise ValueError(f"win.predict returned shape {p_m.shape}, expected ({len(runners)},)")
        p[m] = p_m
    x = p * win_odds
    lo_q, hi_q = np.quantile(x, QUANTILES, axis=0)
    for j, h in enumerate(runners):
        out.append(
            row(
                WIN,
                str(h),
                (board.win[h], None),
                value_mean=float(x[:, j].mean()),
                value_q05=float(lo_q[j]),
                value_q95=float(hi_q[j]),
            )
        )

    pools = load_source("japan", "jra").pools
    position = {h: j for j, h in enumerate(runners)}
    place_horses = sorted(board.place)
    wide_pairs = sorted(board.wide)
    for pool, keys, figures in (
        (PLACE, [str(h) for h in place_horses], [board.place[h] for h in place_horses]),
        (WIDE, [_key(a, b) for a, b in wide_pairs], [board.wide[k] for k in wide_pairs]),
    ):
        places = pools[pool].places_for(board.starters)
        if models.ordering is None or places == 0 or not keys:
            out.extend(row(pool, c, f) for c, f in zip(keys, figures, strict=True))
            continue
        low, high = _columns(draws(pool), keys, n_draws)
        prob = np.empty_like(low)
        for m in range(n_draws):
            if pool == PLACE:
                place_m = np.asarray(models.ordering.place_probs(p[m], places), dtype=float)
                prob[m] = [place_m[position[h]] for h in place_horses]
            else:
                pair_m = np.asarray(models.ordering.pair_probs(p[m], places), dtype=float)
                prob[m] = [pair_m[position[a], position[b]] for a, b in wide_pairs]
        for j, (c, f) in enumerate(zip(keys, figures, strict=True)):
            out.append(
                row(
                    pool,
                    c,
                    f,
                    value_at_low=_mean_or_none(prob[:, j], low[:, j]),
                    value_at_high=_mean_or_none(prob[:, j], high[:, j]),
                )
            )
    return out
