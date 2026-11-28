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
- A pool whose nowcast raises KeyError (no fit for its (pool, tau) cell) gets its rows with every
  value None and `reason` set; the other pools keep their values. Without WIN draws there is no
  p, so every pool's rows are null. `reason` is for the caller's logs and is not part of the
  shadow log's record.

v0 limits:
- no own stake in the dividend: the final price is taken as unmoved by our ticket;
- no co-placer-exact point value: Place and Wide get a low-end and a high-end value from the
  range figures, not the dividend given who else places;
- no parameter uncertainty from refits of the win, ordering or nowcast models (#100 adds it);
- the Win-Place covariance comes only through the shared shocks.

`model_probabilities` (ticket #506) gives the shadow page's `p_model`, display only: every pool's
probability per combination from one Win vector and the ordering fit's ordered finishes, exact
sums over the ordered first three. It sits beside the values and never enters a ShadowValue.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import permutations
from typing import Protocol

import numpy as np

from horsetrack.core.config import load_source
from horsetrack.countries.japan.board.snapshot import PLACE, WIDE, WIN, RaceBoard, RaceContext
from horsetrack.countries.japan.models.v0.ordering import (
    OrderingFit,
    _check_family,
    _check_p,
    _order_probs,
)

QUANTILES = (0.05, 0.95)
QUINELLA, FORECAST, TRIO, TIERCE = "QUINELLA", "FORECAST", "TRIO", "TIERCE"
BRACKET_QUINELLA = "BRACKET_QUINELLA"


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
    reason: str | None = None  # why the row has no value when its pool could not be drawn


def point_seed(seed: int, race_id: str, tau: int) -> int:
    """The draws' seed of one (race, tau): a stable hash of (seed, race_id, tau), the same on every
    run and machine, so a record's values can be recomputed from its board. Shadow-day and the
    shadow feed both seed their draws with it (moved here from shadow/day.py by #564)."""
    digest = hashlib.sha256(f"{int(seed)}|{race_id}|{int(tau)}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


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

    def draws(pool: str) -> PoolDraws | str:
        """The pool's draws, or the reason the nowcast has none (a missing fit cell)."""
        try:
            return models.nowcast.draws(board, pool=pool, tau_min=tau_min, shocks=shocks)
        except KeyError as exc:
            return str(exc.args[0]) if exc.args else f"no fit for {pool} at tau {tau_min}"

    def row(pool: str, combination: str, figures: tuple[float, float | None], **values):
        fields = dict.fromkeys(
            ("value_mean", "value_q05", "value_q95", "value_at_low", "value_at_high")
        )
        fields.update(values)
        return ShadowValue(board.race_id, pool, combination, tau_min, *figures, **fields)

    out: list[ShadowValue] = []
    if not runners:
        return out

    pools = load_source("japan", "jra").pools
    place_horses = sorted(board.place)
    wide_pairs = sorted(board.wide)
    keyed = {
        PLACE: ([str(h) for h in place_horses], [board.place[h] for h in place_horses]),
        WIDE: ([_key(a, b) for a, b in wide_pairs], [board.wide[k] for k in wide_pairs]),
    }

    def null_rows(pool: str, why: str) -> list[ShadowValue]:
        if pool == WIN:
            return [row(WIN, str(h), (board.win[h], None), reason=why) for h in runners]
        keys, figures = keyed[pool]
        return [row(pool, c, f, reason=why) for c, f in zip(keys, figures, strict=True)]

    win_draws = draws(WIN)
    if isinstance(win_draws, str):  # without WIN draws there is no p for any pool
        out.extend(null_rows(WIN, win_draws))
        out.extend(null_rows(PLACE, "no win draws"))
        out.extend(null_rows(WIDE, "no win draws"))
        return out

    win_odds, _ = _columns(win_draws, [str(h) for h in runners], n_draws)
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

    position = {h: j for j, h in enumerate(runners)}
    for pool, (keys, figures) in keyed.items():
        places = pools[pool].places_for(board.starters)
        if models.ordering is None or places == 0 or not keys:
            out.extend(row(pool, c, f) for c, f in zip(keys, figures, strict=True))
            continue
        pool_draws = draws(pool)
        if isinstance(pool_draws, str):
            out.extend(null_rows(pool, pool_draws))
            continue
        low, high = _columns(pool_draws, keys, n_draws)
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


# ---------------------------------------------------------------- model probabilities (#506)


def model_probabilities(
    p: Mapping[int, float],
    ordering: OrderingFit | None,
    *,
    brackets: Mapping[int, int] | None = None,
    starters: int | None = None,
) -> dict[str, dict[str, float]]:
    """Pool -> combination text (the odds log's) -> probability, for the runners in `p`.

    `p` (horse number -> Win probability, scratched runners left out) is renormalised to sum to 1
    and WIN is that; without an ordering fit WIN is all there is. With one, P(i, j, k first three
    in that order) is the fit's discounted Harville model (`ordering._order_probs`) and every other
    pool is an exact sum over those ordered triples:
    - PLACE "7": P(7 in the first `places`), places from the jra config for `starters`;
    - QUINELLA "a,b" (a < b): P({first two} = {a, b}); FORECAST "a,b": P(a first, b second);
    - QUINELLA_PLACE "a,b" (a < b): P(a and b both in the first `places`, 3 from 4 starters);
    - TRIO "a,b,c" (a < b < c): P({first three} = {a, b, c}); TIERCE "a,b,c": P(a, b, c in order);
    - BRACKET_QUINELLA "x,y" (x <= y), only with `brackets` (horse -> bracket; horses not in `p`
      are ignored): P({bracket of the first, bracket of the second} = {x, y}).
    `starters` (default len(p)) is the runners at the start of sales: JRA fixes the places Place and
    Wide pay, and whether they are sold, then, so a runner scratched after sales opened leaves them
    unchanged (8 drawn and one out: Place still pays three places over the seven left). Only the
    places come from it; the ordered finishes are over the runners in `p` alone.
    Each pool keys every combination of the runners and no other; a pool the config does not sell
    for `starters` is left out, and so is every pool but WIN with fewer than three runners.
    One race at a time: an 18-runner race has 4,896 ordered triples.
    """
    horses = sorted(int(h) for h in p)
    if not horses:
        return {}
    q = _check_p(np.array([float(p[h]) for h in horses]))
    names = [str(h) for h in horses]
    out: dict[str, dict[str, float]] = {WIN: dict(zip(names, q.tolist(), strict=True))}
    n = len(horses)
    starters = n if starters is None else int(starters)
    if starters < n:
        raise ValueError(f"{starters} starters at the start of sales, but {n} runners in p")
    if ordering is None or n < 3:
        return out
    _check_family(ordering.family)
    triples = _order_probs(q, ordering.lambdas, 3)  # [i, j, k]: i first, j second, k third
    # two positions at a time: (first, second), (first, third), (second, third)
    pairs = {(0, 1): triples.sum(axis=2), (0, 2): triples.sum(axis=1), (1, 2): triples.sum(axis=0)}
    marginals = [triples.sum(axis=(1, 2)), triples.sum(axis=(0, 2)), triples.sum(axis=(0, 1))]
    pools = load_source("japan", "jra").pools
    every = range(n)

    places = pools[PLACE].places_for(starters)
    if places:
        place = sum(marginals[:places]).tolist()
        out[PLACE] = {names[i]: place[i] for i in every}
    first_two = pairs[(0, 1)].tolist()
    out[QUINELLA] = {
        f"{names[i]},{names[j]}": first_two[i][j] + first_two[j][i]
        for i in every
        for j in every
        if i < j
    }
    places = pools[WIDE].places_for(starters)
    if places:
        one_way = sum(m for (a, b), m in pairs.items() if b < places)
        wide = (one_way + one_way.T).tolist()
        out[WIDE] = {f"{names[i]},{names[j]}": wide[i][j] for i in every for j in every if i < j}
    out[FORECAST] = {
        f"{names[i]},{names[j]}": first_two[i][j] for i in every for j in every if i != j
    }
    unordered = sum(triples.transpose(axes) for axes in permutations(range(3))).tolist()
    out[TRIO] = {
        f"{names[i]},{names[j]},{names[k]}": unordered[i][j][k]
        for i in every
        for j in every
        for k in every
        if i < j < k
    }
    ordered = triples.tolist()
    out[TIERCE] = {
        f"{names[i]},{names[j]},{names[k]}": ordered[i][j][k]
        for i in every
        for j in every
        for k in every
        if i != j and k not in (i, j)
    }
    if brackets is not None:
        missing = [h for h in horses if h not in brackets]
        if missing:
            raise ValueError(f"runners {missing} have no bracket")
        of = [int(brackets[h]) for h in horses]
        sums: dict[tuple[int, int], float] = {}
        for i in every:
            for j in every:
                if i != j:
                    key = (min(of[i], of[j]), max(of[i], of[j]))
                    sums[key] = sums.get(key, 0.0) + first_two[i][j]
        out[BRACKET_QUINELLA] = {f"{x},{y}": v for (x, y), v in sorted(sums.items())}
    return out
