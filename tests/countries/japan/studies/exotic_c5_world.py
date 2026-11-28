"""Synthetic store tables and first-principles references for the C5 run's tests (#459).

Nothing here calls the code under test (`studies.exotic_c5`). A world is a list of meetings. Each
race draws true Win chances, a noisy final Win market (decimal odds rounded to 0.1), a noisy
model (the stand-in for #87's `stage2:q:form`, drawn from its own random stream so a world can
change it alone), a finishing order from Harville on the truth, and every pool's dividends from
planted money shares through the statute payout (04 §1.2): a winning ticket on c pays
(W_c + L / P) x R per yen, P the winning combinations, L the money that lost, R the race's own
payout rate (`takeout.race_takeout`, boosts in); the dividend per 100 yen is floored to 10 yen with
the 110-yen minimum. A single-winner pool's planted share is ratio x the Harville probability of
the winning combination on the normalised market, so the dividends' implied share
R * 100 / (D + 5) is known for the reference; a `hot` race's favourite wins and its Place and Wide
dividends are pushed to the 110-yen floor (`dividend_floor_bound` territory).

The references here are written out loop by loop: Harville / discounted Harville from its
definition, pool marginals by enumeration, the implied share by its formula.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from itertools import combinations

import numpy as np
import pandas as pd

from horsetrack.countries.japan import takeout

WIN, PLACE, QUINELLA, WIDE = "WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE"
EXACTA, TRIO, TRIFECTA = "FORECAST", "TRIO", "TIERCE"
POOLS = (WIN, PLACE, QUINELLA, WIDE, EXACTA, TRIO, TRIFECTA)
SINGLE = (WIN, QUINELLA, EXACTA, TRIO, TRIFECTA)  # one winning combination per clean result
MULTI = (PLACE, WIDE)
RAW_CLASS = "3歳以上1勝クラス"  # an all-ages label: no age-restricted boost matches it
MIN_DIVIDEND = 110.0
TAKE = 0.8  # the Win market's payout, for the synthetic odds only


@dataclass(frozen=True)
class Meeting:
    day: date
    venue: str
    races: int = 3
    hot: tuple[int, ...] = ()  # race numbers whose favourite wins at a floor-bound Place/Wide


@dataclass
class RaceTruth:
    race_id: str
    race_date: date
    meeting_id: str
    horses: list[int]
    odds: dict[int, float]
    p_norm: dict[int, float]
    p_form: dict[int, float]
    order: list[int]  # finishing order, every runner finishes
    places: int  # Place's places paid for the field
    winners: dict[str, list[str]]  # pool -> winning combination texts
    dividends: dict[str, dict[str, float]]  # pool -> combination -> dividend per 100 yen
    rates: dict[str, float]  # pool -> the race's payout rate
    hot: bool


@dataclass
class World:
    races: pd.DataFrame
    runners: pd.DataFrame
    dividends: pd.DataFrame
    form: pd.DataFrame
    truth: dict[str, RaceTruth] = field(default_factory=dict)

    def store(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.races.copy(), self.runners.copy(), self.dividends.copy()


# ---------------------------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------------------------


def normalised(values: Mapping[int, float]) -> dict[int, float]:
    total = math.fsum(values.values())
    return {h: v / total for h, v in values.items()}


def triples(p: Mapping[int, float], lambdas: tuple[float, float] = (1.0, 1.0)) -> dict:
    """P(i, j, k first three in that order): discounted Harville, p renormalised, weights
    p^lambda2 for second and p^lambda3 for third over the runners left (Harville at (1, 1))."""
    q = normalised(p)
    l2, l3 = lambdas
    out = {}
    for i in q:
        second = math.fsum(q[h] ** l2 for h in q if h != i)
        for j in q:
            if j == i:
                continue
            first_two = q[i] * q[j] ** l2 / second
            third = math.fsum(q[h] ** l3 for h in q if h not in (i, j))
            for k in q:
                if k not in (i, j):
                    out[(i, j, k)] = first_two * q[k] ** l3 / third
    return out


def _text(horses: Sequence[int], ordered: bool) -> str:
    return ",".join(str(h) for h in (horses if ordered else sorted(horses)))


def pool_probabilities(
    p: Mapping[int, float], lambdas: tuple[float, float] = (1.0, 1.0), *, places: int = 3
) -> dict[str, dict[str, float]]:
    """Every pool's combination probabilities, summed from `triples`, keyed by the store's
    combination text (WIN and PLACE one horse; QUINELLA, QUINELLA_PLACE and TRIO ascending;
    FORECAST and TIERCE in finishing order). Place pays the first `places`."""
    out: dict[str, dict[str, float]] = {pool: {} for pool in POOLS}

    def add(pool: str, key: str, value: float) -> None:
        out[pool][key] = out[pool].get(key, 0.0) + value

    for (i, j, k), prob in triples(p, lambdas).items():
        add(TRIFECTA, _text((i, j, k), True), prob)
        add(TRIO, _text((i, j, k), False), prob)
        add(EXACTA, _text((i, j), True), prob)
        add(QUINELLA, _text((i, j), False), prob)
        for pair in ((i, j), (i, k), (j, k)):
            add(WIDE, _text(pair, False), prob)
        for h in (i, j, k)[:places]:
            add(PLACE, str(h), prob)
    out[WIN] = {str(h): v for h, v in normalised(p).items()}
    return out


def implied_share(dividend: float, rate: float) -> float:
    """The money share on the one winning combination of a race-pool: R * 100 / (D + 5)."""
    return rate * 100.0 / (dividend + 5.0)


def payout_rate(race_id: str, pool: str) -> float:
    return takeout.race_takeout(race_id, pool, race_class_raw=RAW_CLASS).payout_rate


def floor_bound(dividends: Mapping[str, float]) -> bool:
    """A race-pool whose lowest dividend is at or under the 110-yen minimum."""
    return min(dividends.values()) <= MIN_DIVIDEND


# ---------------------------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------------------------


def _floor10(x: float) -> float:
    return math.floor(x / 10.0 + 1e-9) * 10.0


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def _draw_order(rng: np.random.Generator, p: np.ndarray, first: int | None) -> list[int]:
    """Positions in finishing order: Harville draws without replacement on p."""
    left = list(range(len(p)))
    order = []
    if first is not None:
        order.append(first)
        left.remove(first)
    while left:
        w = p[left] / p[left].sum()
        pick = int(rng.choice(len(left), p=w))
        order.append(left.pop(pick))
    return order


def _single_dividend(share: float, rate: float) -> float:
    """P = 1: (W + L) x R / W per yen = R / x; floored to 10 yen."""
    return _floor10(100.0 * rate / share)


def _statute_dividends(shares: Sequence[float], rate: float) -> list[float]:
    """P winners: (W_c + L / P) x R / W_c per yen, L = 1 - sum of the shares; floored to 10 yen,
    then the 110-yen minimum."""
    lost = 1.0 - math.fsum(shares)
    assert lost > 0, shares
    n = len(shares)
    return [max(MIN_DIVIDEND, _floor10(100.0 * rate * (1.0 + lost / (n * x)))) for x in shares]


Ratio = Callable[[int, str, np.random.Generator], float]


def default_ratio(index: int, pool: str, rng: np.random.Generator) -> float:
    """The planted share over the market's Harville probability: lognormal around 1."""
    return float(np.exp(rng.normal(0.0, 0.35)))


def make_world(
    meetings: Sequence[Meeting],
    *,
    seed: int,
    form_seed: int | None = None,
    ratio: Ratio = default_ratio,
    field_sizes: Sequence[int] = (6, 7, 8),
) -> World:
    """The store tables (races, runners, dividends) and the form member of `meetings`.

    `ratio(index, pool, rng)` is the planted share over the market Harville probability of a
    single-winner pool's winning combination (index: the race's position in the world). Changing
    `form_seed` changes the form member's probabilities alone."""
    rng = np.random.default_rng(seed)
    rng_form = np.random.default_rng(seed + 7919 if form_seed is None else form_seed)
    races, runners, dividends, form = [], [], [], []
    truth: dict[str, RaceTruth] = {}
    index = 0
    for meeting in meetings:
        meeting_id = f"jra:{meeting.day.isoformat()}:{meeting.venue}"
        for race_no in range(1, meeting.races + 1):
            race_id = f"jra:{meeting.day.isoformat()}:{meeting.venue}:{race_no:02d}"
            n = int(rng.choice(list(field_sizes)))
            horses = list(range(1, n + 1))
            p_true = rng.dirichlet(np.full(n, 1.6))
            q_mkt = _softmax(np.log(p_true) + rng.normal(0.0, 0.3, n))
            odds = np.maximum(1.1, np.round(TAKE / q_mkt, 1))
            inv = 1.0 / odds
            p_norm = inv / inv.sum()
            p_form = _softmax(np.log(p_true) + rng_form.normal(0.0, 0.4, n))
            hot = race_no in meeting.hot
            favourite = int(np.lexsort((np.arange(n), odds))[0])
            order_pos = _draw_order(rng, p_true, favourite if hot else None)
            order = [horses[i] for i in order_pos]
            places = 3 if n >= 8 else 2
            pn = {h: float(v) for h, v in zip(horses, p_norm, strict=True)}
            probs = pool_probabilities(pn, places=places)
            winners = {
                WIN: [str(order[0])],
                QUINELLA: [_text(order[:2], False)],
                EXACTA: [_text(order[:2], True)],
                TRIO: [_text(order[:3], False)],
                TRIFECTA: [_text(order[:3], True)],
                PLACE: [str(h) for h in order[:places]],
                WIDE: [_text(pair, False) for pair in combinations(sorted(order[:3]), 2)],
            }
            rates = {pool: payout_rate(race_id, pool) for pool in POOLS}
            paid: dict[str, dict[str, float]] = {}
            for pool in SINGLE:
                (combo,) = winners[pool]
                cap = 100.0 * rates[pool] / 125.0  # dividend 120 or more: never the floor
                share = min(cap, max(1e-5, ratio(index, pool, rng) * probs[pool][combo]))
                paid[pool] = {combo: _single_dividend(share, rates[pool])}
            for pool in MULTI:
                combos = winners[pool]
                if hot:
                    fav = str(horses[favourite])
                    big = 0.6 if pool == PLACE else 0.3
                    shares = [big if fav in c.split(",") else 0.05 for c in combos]
                else:
                    total = len(combos) if pool == WIDE else places
                    shares = [
                        min(0.3, max(1e-4, probs[pool][c] / total * float(rng.uniform(0.6, 1.2))))
                        for c in combos
                    ]
                paid[pool] = dict(zip(combos, _statute_dividends(shares, rates[pool]), strict=True))
            truth[race_id] = RaceTruth(
                race_id=race_id,
                race_date=meeting.day,
                meeting_id=meeting_id,
                horses=horses,
                odds={h: float(o) for h, o in zip(horses, odds, strict=True)},
                p_norm=pn,
                p_form={h: float(v) for h, v in zip(horses, p_form, strict=True)},
                order=order,
                places=places,
                winners=winners,
                dividends=paid,
                rates=rates,
                hot=hot,
            )
            races.append(
                {
                    "race_id": race_id,
                    "race_date": meeting.day,
                    "meeting_id": meeting_id,
                    "venue": meeting.venue,
                    "race_no": race_no,
                    "race_class_raw": RAW_CLASS,
                    "field_size": n,
                    "status": "run",
                }
            )
            position = {h: k + 1 for k, h in enumerate(order)}
            for h, o, pf in zip(horses, odds, p_form, strict=True):
                runners.append(
                    {
                        "race_id": race_id,
                        "race_date": meeting.day,
                        "meeting_id": meeting_id,
                        "venue": meeting.venue,
                        "race_no": race_no,
                        "horse_id": f"h{index:04d}-{h:02d}",
                        "horse_no": h,
                        "win_odds": float(o),
                        "finish_pos": float(position[h]),
                        "dead_heat": False,
                        "is_scratched": False,
                    }
                )
                form.append({"race_id": race_id, "horse_no": h, "p_form": float(pf)})
            for pool in POOLS:
                for combo, dividend in paid[pool].items():
                    dividends.append(
                        {
                            "race_id": race_id,
                            "pool": pool,
                            "combination": combo,
                            "dividend": float(dividend),
                            "is_refund": False,
                        }
                    )
            index += 1
    return World(
        races=_frame(races, {"race_no": "int64", "field_size": "int64"}),
        runners=_frame(
            runners,
            {
                "race_no": "int64",
                "horse_no": "int64",
                "win_odds": "float64",
                "finish_pos": "float64",
            },
        ),
        dividends=_frame(dividends, {"dividend": "float64"}),
        form=_frame(form, {"horse_no": "int64", "p_form": "float64"}),
        truth=truth,
    )


def _frame(rows: list[dict], dtypes: Mapping[str, str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return frame.astype(dict(dtypes)) if len(frame) else frame


def replace_races(world: World, other: World, race_ids: Sequence[str]) -> World:
    """`world` with the rows (and truth) of `race_ids` taken from `other`: the same races with
    other outcomes, odds and dividends."""
    ids = set(race_ids)

    def swap(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        kept = a[~a["race_id"].isin(ids)]
        return pd.concat([kept, b[b["race_id"].isin(ids)]], ignore_index=True)

    truth = dict(world.truth)
    truth.update({r: other.truth[r] for r in ids})
    return World(
        races=swap(world.races, other.races),
        runners=swap(world.runners, other.runners),
        dividends=swap(world.dividends, other.dividends),
        form=swap(world.form, other.form),
        truth=truth,
    )
