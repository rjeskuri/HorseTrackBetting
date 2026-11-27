"""The decision layer (C6, #100; `own_stake_dividend` from #432): shadow EV with our own stake.

Nothing here selects, gates or places a bet. The functions price; `ShadowDecider` returns one
`ShadowDecision` record per combination on the board, and the owner places bets by hand (owner,
2026-11-27: options are picked by leaderboard rank, stakes are flat within the 30,000-yen
race-day cap, and there is no EV threshold or uncertainty gate). The decider refuses every mode
but "shadow", and this module imports no network, process, ingest or bet-log code.

Dividends per 100 yen, with our stake s inside them (jra.yaml: broken down to 10 yen, at least
110, the Plus 10):
- `own_stake_dividend`, the single-winner identity (Win, Quinella, Exacta, Trio, Trifecta):
  floor(100 r (V + s) / (W + s)) to 10 yen, V the pool, W the money already on the winning
  combination, r the payout rate. Under 110 it pays 110, unless paying 110 to every winning
  ticket would exceed the pool, 110 (W + s) > 100 (V + s): then the Plus 10 is withheld and it
  pays 100, the stake back. Exactly the pool pays 110. #432 lifted it out of #100 for the pool
  depth study (`research/pool_depth.py`).
- Place and Wide pay by the stake-back-share rule (jra.yaml, `core.pricing`,
  `models.exotic.statute_shares`; measured in docs/experiments/pool-depth-japan.md, family
  japan-pool-depth-coplacer-v2): with P combinations paying, combination c pays
  r (1 + L / (P (W_c + s))) per yen, L = V - (the paying combinations' W), the money that lost.
  Our stake adds to V, to W_c and to that sum alike, so L does not move. Broken to 10 yen and at
  least 110, and 100 when paying c 110 would take the pool's payout (the other paying
  combinations at their own dividends, at least 110) over V + s.
- The arithmetic is exact: each value is read as the decimal it prints as, so a dividend that is
  a multiple of 10 (7,250 at a rate of 0.725) is not lost to binary floating point.

EV over nowcast draws, per yen staked (08 §3.2 C6: (1/M) sum_m sum_o P(o | x, Q^(m))
D_c(o, Q^(m), s) - 1):
- Draws are `models.nowcast.Draws`: row m one draw of the pool's final shares in the board's
  combination order, and its final pool V_m. They carry no labels, so every EV function takes
  `combinations=`, the board's `PoolBoard.combinations`. The money on combination j in draw m is
  W_j = shares[m, j] x v_final[m], the share and the pool each read as the decimal it prints as
  and multiplied exactly: a 0.14 share of 300,000 is 42,000, not the 42,000.00000000001 of
  binary floating point, so a dividend of exactly 560 is not broken to 550.
- `ev_single_winner`: the mean over draws of p_fn(q_m) x D_m / 100, less the stake: the mean of
  P x D, never P-bar x D-bar (08 R5).
- `place_outcomes`: every ordered top three of the runners. `ev_place_wide` enumerates them in
  every draw: PLACE pays the first P runners (P from jra.yaml `places_paid` by the field at sale:
  3 from 8, 2 from 5-7), Wide the pairs of the first three, and each pays the ticket by its own
  co-placers. p_triples_fn(q_m) maps ordered triples to their probability; a missing one is 0.
- `ev_distribution`: one EV per refit of C4/C5, the parameter uncertainty reported beside a row.
- `ShadowDecider(mode="shadow", meta=, nowcast=, exotic=)`: `meta[r].predict(q, view)` gives
  P(win) of the Win board's runners from Win draw q, `exotic[r].trifecta(p_win)` the ordered top
  three (`models.exotic.Ordering`'s interface), `nowcast.draws_from_shocks(view, pool=, shocks=)`
  C1's draws (`models.nowcast.Nowcast`). Draw m of the Win pool gives P, draw m of the record's
  own pool the dividend, at the race's payout rate (`takeout.race_takeout`) with
  config/betting.yaml's flat stake inside it.
- The pools are coupled through shared shocks: `decide` draws one matrix of iid N(0, 1) shocks
  per (seed, race, tau), n_draws by the Win board's runners, from a stable sha256 hash, and hands
  the same {horse number: column} to every pool, so draw m of Win, Place and Wide moves with the
  same per-runner shocks (late Win money on a horse raises its place chance and lowers its Place
  dividend in the same draw). Pairing independently seeded draws by index would estimate
  P-bar x D-bar across pools.
- `decide(view, field_at_sale=)`: the runners when sales opened set the places Place pays
  (jra.yaml `places_paid`: a race sold to 8 pays three after a scratch leaves 7); None is the Win
  board's runners, and a field below them is a ValueError.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np

from horsetrack.core import config
from horsetrack.countries.japan import takeout

if TYPE_CHECKING:
    from horsetrack.countries.japan.models.nowcast import Draws, PoolBoard, RaceView

COUNTRY = "japan"
UNIT = 100  # yen per ticket: dividends are quoted per 100 yen
STEP = 10  # breakage: dividends are multiples of 10 yen
FLOOR = 110  # the Plus-10 minimum dividend per 100 yen

WIN, PLACE, WIDE = "WIN", "PLACE", "QUINELLA_PLACE"
SINGLE_WINNER_POOLS = (WIN, "QUINELLA", "FORECAST", "TRIO", "TIERCE")  # one combination pays
COPLACER_POOLS = (PLACE, WIDE)  # several combinations pay on every result
DECIDED_POOLS = (WIN, PLACE, WIDE)  # the pools C1 nowcasts
SHADOW = "shadow"
N_DRAWS = 400  # nowcast draws per pool, as shadow/day.py
SUM_TOL = 1e-6  # how far a draw's shares, or a draw's probabilities, may sum from 1


# ---------------------------------------------------------------- dividends


def _decimal(value: float) -> Fraction:
    return Fraction(repr(float(value)))


def _exact(name: str, value: object) -> Fraction:
    """`value` as an exact fraction of the decimal it prints as; ValueError when not finite."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number, not {value!r}")
    if isinstance(value, Integral):
        return Fraction(int(value))
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, not {value!r}")
    return _decimal(value)


def _stake(value: object) -> Fraction:
    own = _exact("stake", value)
    if own < 0:
        raise ValueError(f"stake must not be negative, not {value!r}")
    return own


def _rate(value: object) -> Fraction:
    rate = _exact("payout_rate", value)
    if not 0 < rate <= 1:
        raise ValueError(f"payout_rate must be in (0, 1], not {value!r}")
    return rate


def own_stake_dividend(
    pool_total: float, combo_money: float, stake: float, payout_rate: float
) -> int:
    """The dividend in yen per 100 yen once `stake` is in the pool and on the combination.

    floor(100 x payout_rate x (V + s) / (W + s)) to a multiple of 10; under 110 it pays 110, or
    100 when 110 (W + s) > 100 (V + s) (the withheld Plus 10). V `pool_total`, W `combo_money`
    (the money already on the combination) and s `stake`, all in yen. ValueError for a pool that
    is not positive, W below 0 or above V, a negative stake, no money on the combination at all
    (W + s = 0), a payout rate outside (0, 1], or a value that is not a finite number.
    """
    pool = _exact("pool_total", pool_total)
    combo = _exact("combo_money", combo_money)
    own = _stake(stake)
    rate = _rate(payout_rate)
    if pool <= 0:
        raise ValueError(f"pool_total must be positive, not {pool_total!r}")
    if not 0 <= combo <= pool:
        raise ValueError(f"combo_money must be in [0, pool_total], not {combo_money!r}")
    return _single_dividend(pool, combo, own, rate)


def _single_dividend(pool: Fraction, combo: Fraction, own: Fraction, rate: Fraction) -> int:
    """`own_stake_dividend` on exact values it has already checked."""
    if combo + own == 0:
        raise ValueError("no money on the combination: it has no dividend")
    raw = UNIT * rate * (pool + own) / (combo + own)
    if raw >= FLOOR:
        return math.floor(raw / STEP) * STEP
    return UNIT if FLOOR * (combo + own) > UNIT * (pool + own) else FLOOR


def _coplacer_dividend(
    pool: int, held: Sequence[int], own: int, paying: tuple[int, ...], j: int, rate: Fraction
) -> int:
    """Column j's dividend per 100 yen when the columns `paying` (j among them) pay, our stake
    `own` on j: r (1 + L / (P (W_j + s))) broken to 10 yen, at least 110, and 100 when paying 110
    takes the payout over V + s. `pool`, `held` (each column's money) and `own` are integers in
    one common unit, so the arithmetic is exact."""
    count = len(paying)
    lost = pool - sum(held[i] for i in paying)

    def tens(money: int) -> int:  # floor(100 r (1 + L / (P money)) / 10)
        return (UNIT * rate.numerator * (count * money + lost)) // (
            STEP * rate.denominator * count * money
        )

    ours = held[j] + own
    if ours == 0:
        raise ValueError("no money on the combination: it has no dividend")
    paid = tens(ours) * STEP
    if paid >= FLOOR:
        return paid
    others = sum(
        held[i] * max(FLOOR, tens(held[i]) * STEP) for i in paying if i != j and held[i] > 0
    )
    return UNIT if others + FLOOR * ours > UNIT * (pool + own) else FLOOR


def _yen_units(pool: float, shares: np.ndarray, own: Fraction) -> tuple[int, list[int], int]:
    """(V, W per column, s) as integers in one unit, W_j = share_j x V with the share and V each
    read as the decimal it prints as: V = c / d and share_j = a_j / common, so one unit is
    1 / (d x common x per) yen, `per` making room for the stake's denominator."""
    v = _decimal(pool)
    parts = [_decimal(x) for x in shares]
    common = math.lcm(*(x.denominator for x in parts))
    scale = math.lcm(v.denominator * common, own.denominator)
    per = scale // (v.denominator * common)
    held = [x.numerator * (common // x.denominator) * v.numerator * per for x in parts]
    return v.numerator * common * per, held, own.numerator * (scale // own.denominator)


# ---------------------------------------------------------------- inputs


def _probability(value: object) -> float:
    try:
        p = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"a probability must be a number, not {value!r}") from None
    if not 0.0 <= p <= 1.0:  # NaN fails too
        raise ValueError(f"a probability must be in [0, 1], not {value!r}")
    return p


def _labels(combinations: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(combinations)
    if not labels or not all(isinstance(c, str) for c in labels):
        raise ValueError(f"combinations must be the board's combination texts, not {labels!r}")
    if len(set(labels)) < len(labels):
        raise ValueError(f"a combination is given twice: {labels!r}")
    return labels


def _column(labels: tuple[str, ...], combo: str) -> int:
    if combo not in labels:
        raise ValueError(f"{combo!r} is not among the board's combinations")
    return labels.index(combo)


def _checked_draws(draws: Draws, labels: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """(shares, v_final) of `draws`. ValueError unless there is at least one draw, a column per
    combination, a share row in [0, 1] summing to 1 and a finite, positive pool per draw."""
    shares = np.asarray(draws.shares, dtype=float)
    v = np.asarray(draws.v_final, dtype=float)
    if shares.ndim != 2 or shares.shape[1] != len(labels):
        raise ValueError(f"draws of shape {shares.shape} for {len(labels)} combinations")
    if not len(shares):
        raise ValueError("no draws: the EV is a mean over draws")
    if v.shape != (len(shares),):
        raise ValueError(f"{v.shape} final pools for {len(shares)} draws")
    if not np.all((shares >= 0) & (shares <= 1)):  # NaN fails too
        raise ValueError("every share must be in [0, 1]")
    if np.any(np.abs(shares.sum(axis=1) - 1) > SUM_TOL):
        raise ValueError("every draw's shares must sum to 1")
    if not np.all(np.isfinite(v) & (v > 0)):
        raise ValueError("every draw's final pool must be finite and positive")
    return shares, v


def _horses(label: str) -> tuple[int, ...]:
    """The horse numbers a combination text names ("5", "3,7")."""
    parts = label.split(",")
    if not all(p.isdigit() and str(int(p)) == p and int(p) > 0 for p in parts):
        raise ValueError(f"{label!r} is not a combination of horse numbers")
    return tuple(int(p) for p in parts)


def _field_at_sale(value: object, runners: int) -> int:
    if value is None:
        return runners
    if isinstance(value, bool) or not isinstance(value, Integral) or value < runners:
        raise ValueError(f"field_at_sale must be a whole number from {runners}, not {value!r}")
    return int(value)


# ---------------------------------------------------------------- single winner


def _single_winner_evs(
    p: np.ndarray, shares: np.ndarray, v: np.ndarray, stake: float, payout_rate: float
) -> np.ndarray:
    """EV[r, t] = mean over draws m of p[r, m, t] x D[m, t] / 100, less 1: D the dividend of the
    combination of column t of `shares` (n, T) in draw m, its money share x V with each read as
    the decimal it prints as, priced only where a refit gives it a chance. `p` is (refits, draws,
    T)."""
    own, rate = _stake(stake), _rate(payout_rate)
    pools = [_decimal(x) for x in v]
    dividend = np.zeros(shares.shape)
    for m, t in zip(*np.nonzero((p > 0).any(axis=0)), strict=True):
        pool = pools[m]
        dividend[m, t] = _single_dividend(pool, _decimal(shares[m, t]) * pool, own, rate)
    return (p * dividend / UNIT).mean(axis=1) - 1.0


def _single_winner_by_refit(
    refits: Sequence[Callable[[np.ndarray], float]],
    draws: Draws,
    combo: str,
    stake: float,
    payout_rate: float,
    combinations: Sequence[str],
) -> np.ndarray:
    labels = _labels(combinations)
    j = _column(labels, combo)
    shares, v = _checked_draws(draws, labels)
    p = np.array([[_probability(f(q)) for q in shares] for f in refits], dtype=float)
    return _single_winner_evs(p[:, :, None], shares[:, [j]], v, stake, payout_rate)[:, 0]


def ev_single_winner(
    p_fn: Callable[[np.ndarray], float],
    draws: Draws,
    combo: str,
    stake: float,
    payout_rate: float,
    *,
    combinations: Sequence[str],
) -> float:
    """EV per yen of `stake` on `combo` in a pool that pays one combination per result:
    (1/n) sum_m p_fn(q_m) x D_m / 100 - 1, q_m the 1-D share row of draw m, p_fn(q) = P(combo | x,
    q) and D_m = own_stake_dividend(V_m, W_combo,m, stake, payout_rate). ValueError for a combo
    not in `combinations`, draws that do not match them, a probability outside [0, 1], or a stake
    or payout rate that `own_stake_dividend` refuses."""
    ev = _single_winner_by_refit([p_fn], draws, combo, stake, payout_rate, combinations)
    return float(ev[0])


# ---------------------------------------------------------------- Place and Wide


def place_outcomes(runners: Iterable[int]) -> Iterator[tuple[int, int, int]]:
    """Every ordered (first, second, third) of three distinct `runners`, once each:
    n(n - 1)(n - 2) of them. ValueError for a runner given twice."""
    numbers = list(runners)
    if len(set(numbers)) < len(numbers):
        raise ValueError(f"a runner is given twice: {numbers!r}")
    return itertools.permutations(numbers, 3)


@dataclass(frozen=True)
class _Coplacers:
    """A PLACE or Wide board: its combinations, and for every ordered top three of its runners
    the columns of the combinations that outcome pays."""

    pool: str
    labels: tuple[str, ...]
    paying: Mapping[tuple[int, ...], tuple[int, ...]]


def _coplacers(
    combinations: Sequence[str], field_at_sale: object, pool: str | None = None
) -> _Coplacers:
    """PLACE when the combinations name one horse, Wide when they name an ascending pair; ValueError
    for a mix, a `pool` that does not match, fewer than three runners, a Wide board without every
    pair of its runners, or a pool not sold to the field at sale (jra.yaml `places_paid`)."""
    labels = _labels(combinations)
    horses = [_horses(c) for c in labels]
    if all(len(h) == 1 for h in horses):
        form = PLACE
    elif all(len(h) == 2 and h[0] < h[1] for h in horses):
        form = WIDE
    else:
        raise ValueError("PLACE combinations name one horse, Wide combinations an ascending pair")
    if pool is not None and pool != form:
        raise ValueError(f"{pool} is not priced from combinations like {labels[0]!r}")
    runners = sorted({h for pair in horses for h in pair})
    if len(runners) < 3:
        raise ValueError(f"an ordered top three needs three runners, got {runners!r}")
    if form == WIDE and len(labels) != math.comb(len(runners), 2):
        raise ValueError("a Wide board must price every pair of its runners")
    field = _field_at_sale(field_at_sale, len(runners))
    places = config.load_source(COUNTRY, "jra").pools[form].places_for(field)
    if places == 0:
        raise ValueError(f"{form} is not sold to a field of {field}")
    column = {h: j for j, h in enumerate(horses)}
    paying = {}
    for outcome in place_outcomes(runners):
        top = outcome[:places]
        keys = [(h,) for h in top] if form == PLACE else itertools.combinations(sorted(top), 2)
        paying[outcome] = tuple(sorted(column[k] for k in keys))
    return _Coplacers(form, labels, paying)


def _paying_chances(board: _Coplacers, probs: object) -> dict[tuple[int, ...], float]:
    """The probability of each set of paying columns under one draw's map of ordered triples.
    ValueError for a probability outside [0, 1], a triple of runners the board does not name
    given a chance, or probabilities summing over 1."""
    if not isinstance(probs, Mapping):
        raise ValueError(f"p_triples_fn must return a mapping of ordered triples, not {probs!r}")
    out: dict[tuple[int, ...], float] = {}
    total = 0.0
    for outcome, value in probs.items():
        p = _probability(value)
        if p == 0.0:
            continue
        paying = board.paying.get(outcome)
        if paying is None:
            raise ValueError(f"{outcome!r} is not an ordered top three of the board's runners")
        total += p
        out[paying] = out.get(paying, 0.0) + p
    if total > 1.0 + SUM_TOL:
        raise ValueError(f"the outcomes' probabilities sum to {total}, over 1")
    return out


def _coplacer_evs(
    board: _Coplacers,
    probs: Sequence[Sequence[object]],
    shares: np.ndarray,
    v: np.ndarray,
    targets: Iterable[int],
    stake: float,
    payout_rate: float,
) -> np.ndarray:
    """EV[r, t] = (1/n) sum_m sum_o probs[r][m](o) x D(o, m) / 100 - 1 for the combination of
    column targets[t]: o every ordered top three of the board's runners, D its dividend when o's
    paying combinations pay (0 when it is not among them), from draw m's `shares` and pool (n,).
    Each dividend is priced once per draw and shared by every refit."""
    own = _stake(stake)
    rate = _rate(payout_rate)
    target_of = {j: t for t, j in enumerate(targets)}
    total = np.zeros((len(probs), len(target_of)))
    for m in range(len(v)):
        chances = [_paying_chances(board, by_draw[m]) for by_draw in probs]
        pool, held, ours = _yen_units(v[m], shares[m], own)
        dividends: dict[tuple[tuple[int, ...], int], int] = {}
        for r, by_paying in enumerate(chances):
            for paying, p in by_paying.items():
                for j in paying:
                    t = target_of.get(j)
                    if t is None:
                        continue
                    if (paying, j) not in dividends:
                        dividends[paying, j] = _coplacer_dividend(pool, held, ours, paying, j, rate)
                    total[r, t] += p * dividends[paying, j] / UNIT
    return total / len(v) - 1.0


def _coplacer_by_refit(
    refits: Sequence[Callable[[np.ndarray], Mapping[tuple[int, int, int], float]]],
    draws: Draws,
    combo: str,
    stake: float,
    payout_rate: float,
    combinations: Sequence[str],
    field_at_sale: int | None,
    pool: str | None = None,
) -> np.ndarray:
    board = _coplacers(combinations, field_at_sale, pool)
    j = _column(board.labels, combo)
    shares, v = _checked_draws(draws, board.labels)
    probs = [[f(q) for q in shares] for f in refits]
    return _coplacer_evs(board, probs, shares, v, [j], stake, payout_rate)[:, 0]


def ev_place_wide(
    p_triples_fn: Callable[[np.ndarray], Mapping[tuple[int, int, int], float]],
    draws: Draws,
    combo: str,
    stake: float,
    payout_rate: float,
    *,
    combinations: Sequence[str],
    field_at_sale: int | None = None,
) -> float:
    """EV per yen of `stake` on `combo` in PLACE (combinations name one horse) or Wide
    (QUINELLA_PLACE: an ascending pair): (1/n) sum_m sum_o P(o | q_m) D(o, m) / 100 - 1, o every
    ordered top three of the runners the combinations name, P(o | q) = p_triples_fn(q)[o] (0 when
    it is left out) and D(o, m) the stake-back share `combo` pays with o's co-placers in draw m, or
    0. PLACE pays the first P runners of o, P from jra.yaml by `field_at_sale` (None: the runners
    named), Wide the three pairs of o. ValueError for a pool not sold to that field, a combo not
    on the board, draws that do not match it, or a probability outside [0, 1]."""
    ev = _coplacer_by_refit(
        [p_triples_fn], draws, combo, stake, payout_rate, combinations, field_at_sale
    )
    return float(ev[0])


# ---------------------------------------------------------------- refits


def ev_distribution(
    refits: Sequence[Callable[[np.ndarray], Any]],
    draws: Draws,
    combo: str,
    stake: float,
    payout_rate: float,
    *,
    combinations: Sequence[str],
    pool: str,
    field_at_sale: int | None = None,
) -> np.ndarray:
    """One EV per refit, in order: the parameter uncertainty from refits of C4/C5. Each refit is a
    p_fn (`ev_single_winner`) for a pool that pays one combination per result, and a p_triples_fn
    (`ev_place_wide`) for PLACE and QUINELLA_PLACE. ValueError without refits, for a pool code it
    does not price, or a `field_at_sale` given for a single-winner pool."""
    refits = list(refits)
    if not refits:
        raise ValueError("no refits: the EV distribution needs at least one")
    if pool in SINGLE_WINNER_POOLS:
        if field_at_sale is not None:
            raise ValueError(f"field_at_sale does not apply to {pool}")
        return _single_winner_by_refit(refits, draws, combo, stake, payout_rate, combinations)
    if pool in COPLACER_POOLS:
        return _coplacer_by_refit(
            refits, draws, combo, stake, payout_rate, combinations, field_at_sale, pool
        )
    raise ValueError(
        f"no EV for pool {pool!r}: expected one of {SINGLE_WINNER_POOLS + COPLACER_POOLS}"
    )


# ---------------------------------------------------------------- the shadow decider


@dataclass(frozen=True, eq=False)
class ShadowDecision:
    """One combination of one pool at one decision time, in shadow: its EV per yen, one per refit,
    priced with `stake_yen` inside the dividend, beside the board it was read from. A record for
    the leaderboard and the shadow log, never a bet."""

    race_id: str
    pool: str
    combination: str
    tau_min: float
    board: PoolBoard
    ev: np.ndarray
    stake_yen: int


def _flat_stake_yen() -> int:
    """config/betting.yaml's flat stake; ValueError unless it is positive and within the race-day
    cap."""
    limits = config.load_country_yaml(COUNTRY, "betting")
    flat, cap = limits.get("flat_stake_yen"), limits.get("day_cap_yen")
    whole = all(isinstance(x, int) and not isinstance(x, bool) for x in (flat, cap))
    if not whole or not 0 < flat <= cap:
        raise ValueError(f"betting.yaml: flat stake {flat!r} must be in (0, day cap {cap!r}]")
    return flat


def _shock_seed(seed: int, race_id: str, tau_min: float) -> int:
    """The shocks' seed for one (race, tau): a stable hash, the same on every run and machine (as
    shadow/day.py's point_seed), never Python's hash(), which varies with PYTHONHASHSEED."""
    digest = hashlib.sha256(f"{seed}|{race_id}|{float(tau_min)!r}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _runner_shocks(numbers: Sequence[int], n: int, seed: int) -> dict[int, np.ndarray]:
    """One column of n iid standard normal shocks per runner, read-only so no pool's nowcast can
    alter another's (as v0's shadow values)."""
    matrix = np.random.default_rng(seed).standard_normal((n, len(numbers)))
    matrix.setflags(write=False)
    return {h: matrix[:, j] for j, h in enumerate(numbers)}


def _win_chances(value: object, runners: int) -> np.ndarray:
    p = np.asarray(value, dtype=float)
    if p.shape != (runners,) or not np.all((p >= 0) & (p <= 1)) or abs(p.sum() - 1) > SUM_TOL:
        raise ValueError(f"meta must give P(win) of the {runners} Win-board runners, summing to 1")
    return p


def _ordered_top3(trifecta: object, horses: Sequence[int]) -> dict[tuple[int, int, int], float]:
    """{(first, second, third): probability} in horse numbers, from an (n, n, n) array indexed as
    the Win board."""
    n = len(horses)
    t = np.asarray(trifecta, dtype=float)
    if t.shape != (n, n, n):
        raise ValueError(f"exotic's trifecta must be ({n}, {n}, {n}), not {t.shape}")
    return {
        (horses[i], horses[j], horses[k]): float(t[i, j, k])
        for i, j, k in itertools.permutations(range(n), 3)
    }


class ShadowDecider:
    """C6 in shadow mode: every combination on the board priced, none selected, nothing placed.

    `meta` and `exotic` are refits of C4 and C5 paired by index: `meta[r].predict(q, view)` gives
    P(win) of the Win board's runners in the board's order from q, one Win draw's share row, and
    `exotic[r].trifecta(p_win)` the (n, n, n) array of P(i first, j second, k third) indexed as
    p_win. `nowcast.draws_from_shocks(view, *, pool, shocks)` is C1's interface; the EV is a mean
    over the draws it returns. `n_draws` and `seed` set the shocks every pool is drawn from (one
    matrix per race and tau, its seed a stable hash). ValueError for any mode but "shadow".
    """

    def __init__(
        self,
        *,
        mode: str,
        meta: Sequence[Any],
        nowcast: Any,
        exotic: Sequence[Any],
        n_draws: int = N_DRAWS,
        seed: int = 0,
    ) -> None:
        if not isinstance(mode, str) or mode != SHADOW:
            raise ValueError(f"the decision layer runs in {SHADOW!r} mode only, not {mode!r}")
        self.meta = tuple(meta)
        self.exotic = tuple(exotic)
        if not self.meta or len(self.exotic) != len(self.meta):
            raise ValueError("meta and exotic are refits paired by index: as many of each, from 1")
        for name, value, low in (("n_draws", n_draws, 1), ("seed", seed, 0)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < low:
                raise ValueError(f"{name} must be a whole number from {low}, not {value!r}")
        self.mode = SHADOW
        self.nowcast = nowcast
        self.n_draws = int(n_draws)
        self.seed = int(seed)
        self.stake_yen = _flat_stake_yen()

    def decide(self, view: RaceView, *, field_at_sale: int | None = None) -> list[ShadowDecision]:
        """One ShadowDecision per combination on the board of each pool in `view` (WIN, PLACE,
        QUINELLA_PLACE), in board order: no selection, no threshold. Every pool is drawn from the
        same per-runner shocks. `field_at_sale` is the runners when sales opened, from which Place
        and Wide pay (None: the Win board's runners). ValueError for a pool it does not price, a
        view without its Win board, a field at sale below the Win board's runners, a Place or
        Wide board without a Win runner that can place, or pools drawn a different number of
        times."""
        pools = tuple(view.pools)
        unpriced = [pool for pool in pools if pool not in DECIDED_POOLS]
        if unpriced:
            raise ValueError(f"the decider prices {DECIDED_POOLS}, not {unpriced}")
        if WIN not in view.pools:
            raise ValueError("the decider reads P(win) from the Win board's draws: no Win board")
        win_labels = _labels(view.pools[WIN].combinations)
        horses = [_horses(c) for c in win_labels]
        if any(len(h) != 1 for h in horses):
            raise ValueError("Win combinations name one horse each")
        numbers = [h for (h,) in horses]
        field = _field_at_sale(field_at_sale, len(numbers))
        seed = _shock_seed(self.seed, view.race_id, view.tau_min)
        shocks = _runner_shocks(numbers, self.n_draws, seed)
        draws = {
            pool: self.nowcast.draws_from_shocks(view, pool=pool, shocks=shocks) for pool in pools
        }
        win_shares, win_v = _checked_draws(draws[WIN], win_labels)
        p_win = np.array(
            [
                [_win_chances(f.predict(q, view), len(numbers)) for q in win_shares]
                for f in self.meta
            ]
        )
        top3: list[list[dict]] = []
        if PLACE in view.pools or WIDE in view.pools:
            top3 = [
                [_ordered_top3(ordering.trifecta(p), numbers) for p in by_draw]
                for ordering, by_draw in zip(self.exotic, p_win, strict=True)
            ]

        out: list[ShadowDecision] = []
        for pool in pools:
            board = view.pools[pool]
            rate = takeout.race_takeout(view.race_id, pool).payout_rate
            if pool == WIN:
                labels = win_labels
                ev = _single_winner_evs(p_win, win_shares, win_v, self.stake_yen, rate)
            else:
                spec = _coplacers(board.combinations, field, pool)
                labels = spec.labels
                shares, v = _checked_draws(draws[pool], labels)
                if len(v) != len(win_v):
                    raise ValueError(f"{len(v)} {pool} draws for {len(win_v)} Win draws")
                ev = _coplacer_evs(spec, top3, shares, v, range(len(labels)), self.stake_yen, rate)
            out += [
                ShadowDecision(
                    race_id=view.race_id,
                    pool=pool,
                    combination=combo,
                    tau_min=view.tau_min,
                    board=board,
                    ev=ev[:, j].copy(),
                    stake_yen=self.stake_yen,
                )
                for j, combo in enumerate(labels)
            ]
        return out
