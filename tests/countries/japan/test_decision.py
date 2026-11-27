"""Tickets #432 and #100: the decision layer (C6): our own stake in the dividend, and shadow EV.

Contract pinned here for `horsetrack.countries.japan.models.decision` (#432 made it; #100 extends
it, 08 §3.2 C6: EV = (1/M) sum_m sum_o P(o | x, Q^(m)) x D_c(o, Q^(m), s) - 1):

- `own_stake_dividend(pool_total, combo_money, stake, payout_rate) -> int` (#432): the dividend in
  yen per 100 yen once our `stake` is in the pool and on the combination,
  floor(100 x rate x (V + s) / (W + s)) to a multiple of 10 (breakage), and never below 110
  (the Plus-10 floor), V the pool, W the money already on the combination, s the stake, all in
  yen. The arithmetic is decimal: a dividend that is exactly a multiple of 10 (7,250 below) stays
  that multiple, whatever binary floating point makes of 0.725. The result is a Python int.
  ValueError for a pool that is not positive, money on the combination below 0 or above the
  pool, a negative stake, no money on the combination at all (W + s = 0), a payout rate outside
  (0, 1], or any value that is not a finite number.
  #100 adds the withheld Plus 10: when paying 110 to every winning ticket would exceed the pool,
  110 (W + s) > 100 (V + s), the dividend is 100 (the stake back). Exactly the pool does not
  exceed it, and pays 110.
- Draws are `models.nowcast.Draws`: `shares` (n, k), row m one draw of the final money shares in
  the board's combination order, and `v_final` (n,), that draw's final pool in yen. Draws carry no
  labels, so every EV function takes `combinations=` (keyword only): the k combination texts of
  the columns, in order (the board's `PoolBoard.combinations`). The money on combination j in
  draw m is W_j = shares[m, j] x v_final[m].
- `ev_single_winner(p_fn, draws, combo, stake, payout_rate, *, combinations) -> float`, for a pool
  that pays one combination per result (WIN, QUINELLA, ...): (1/n) sum_m p_fn(q_m) x D_m / 100 - 1,
  q_m the 1-D share row of draw m, p_fn(q) = P(combo | x, q), and
  D_m = own_stake_dividend(V_m, W_combo,m, stake, payout_rate). The mean over draws of P x D,
  never P-bar x D-bar (08 R5).
- `place_outcomes(runners) -> Iterator[tuple[int, int, int]]`: every ordered (first, second,
  third) of three distinct runners of `runners`, once each: n(n - 1)(n - 2) of them, made of the
  numbers given. ValueError for a runner given twice.
- `ev_place_wide(p_triples_fn, draws, combo, stake, payout_rate, *, combinations,
  field_at_sale=None) -> float`: PLACE when the combinations name one horse ("5"), Wide
  (QUINELLA_PLACE) when they name a pair ("3,7", ascending). p_triples_fn(q) maps ordered triples
  of horse numbers to their probability given one draw's share row q (a triple it leaves out has
  probability 0). Every outcome o of `place_outcomes` over the runners the combinations name pays
  P combinations: PLACE the first P runners of o (P = 3 when the field at sale is 8 or more, 2 at
  5-7; PLACE is not sold at 4 or fewer: ValueError), Wide the three pairs of o's three runners.
  When `combo` is among them it pays the stake-back share r (1 + L / (P (W_combo + s))) per yen,
  L = V - (the paying combinations' W), the money that lost (our stake adds to V, to W_combo and
  to that sum alike, so L stays), broken to 10 yen per 100, at least 110, and 100 when paying
  110 would take the pool's payout over the pool (V + s); otherwise it pays nothing.
  EV = (1/n) sum_m sum_o P(o | q_m) D(o, m) / 100 - 1. `field_at_sale` defaults to the number of
  runners the combinations name.
- `ev_distribution(refits, draws, combo, stake, payout_rate, *, combinations, pool,
  field_at_sale=None) -> np.ndarray`: one EV per refit, in order (parameter uncertainty from
  refits of C4/C5). Each refit is a p_fn for a single-winner `pool`, a p_triples_fn for PLACE and
  QUINELLA_PLACE. ValueError without refits or for a pool code it does not price.
- `ShadowDecider(*, mode, meta, nowcast, exotic)`: ValueError for any mode but "shadow".
  `meta`: a sequence of C4 refits, each `predict(q, view)` -> P(win) of the Win board's runners in
  the board's order, q one Win draw's share row; `exotic`: C5 refits, paired with meta by index,
  each `trifecta(p_win)` -> the (n, n, n) array of P(i first, j second, k third) indexed as p_win
  (`models.exotic.Ordering.trifecta`); `nowcast`: C1's `draws(view, *, pool, n, seed) -> Draws`,
  averaged over however many draws it returns.
- `decide(view) -> list[ShadowDecision]`: one record per combination on the board of each pool
  in the view (no selection, no EV threshold, no uncertainty gate: owner, 2026-11-27), each with
  `race_id`, `pool`, `combination`, `tau_min`, `board` (that pool's PoolBoard at t), `ev` (an
  np.ndarray, one EV per refit) and `stake_yen` (config/betting.yaml's flat stake, within its
  race-day cap). The EV is priced with that stake inside the dividend, at the race's payout rate
  (jra.yaml's on a day without a boost). Draw m of the Win pool gives P (WIN: meta's P(win); PLACE
  and Wide: meta, then exotic's trifecta); draw m of the record's own pool gives the dividend.
  Nothing here submits, places or logs a bet: the owner places bets by hand.

Place and Wide pay by the stake-back-share rule of `config/jra.yaml`, `core.pricing` and
`models.exotic.statute_shares`: a winning ticket gets r (W_i + L / P) / W_i per yen, the payout
rate on its stake back too. The form 1 + (rV - sum W) / (P W_i), frozen in japan-pool-depth-v1,
failed its own check against the paid dividends, and this rule passed it
(japan-pool-depth-coplacer-v2; docs/experiments/pool-depth-japan.md). The withheld Plus 10 of a
pool that pays several combinations is pinned only where every reading of "paying 110 would
exceed the pool" agrees: 110 on the combination's money alone is more than the pool, or the
whole payout with 110 in it is well under it.

Every figure is invented; nothing reads data.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party.
from __future__ import annotations  # noqa: I001

import ast
import itertools
import math
import re
import socket
from collections.abc import Iterator, Mapping, Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from horsetrack.core.config import load_country_yaml
from horsetrack.countries.japan.models import decision
from horsetrack.countries.japan.models.decision import own_stake_dividend
from horsetrack.countries.japan.models.nowcast import Draws, PoolBoard, RaceView

RATES = ("0.8", "0.775", "0.75", "0.725", "0.7")  # decimal payout rates, as strings
WIN, PLACE, WIDE = "WIN", "PLACE", "QUINELLA_PLACE"
# jra.yaml's payout rates (1 - takeout) on a day without a boost, as the decider must use them
POOL_RATE = {WIN: 0.8, PLACE: 0.8, WIDE: 0.775}
TOL = 1e-9


# ---------------------------------------------------------------- references, from first principles


def expected_dividend(pool: int, combo: int, stake: int, rate: str) -> int:
    """floor(100 x rate x (V + s) / (W + s)) to 10 yen, at least 110, in exact arithmetic."""
    raw = 100 * Fraction(rate) * Fraction(pool + stake, combo + stake)
    return max(110, math.floor(raw / 10) * 10)


def _exact(value: object) -> Fraction:
    """`value` as the decimal it prints as (how decision.py reads a number)."""
    if isinstance(value, str):
        return Fraction(value)
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    return Fraction(repr(float(value)))


def paid_dividend(pool: object, combo: object, stake: object, rate: object) -> int:
    """A single-winner pool's dividend per 100 yen with our stake in: breakage to 10 yen, the
    110 floor, and 100 when paying 110 on W + s would exceed V + s."""
    v, w, s, r = (_exact(x) for x in (pool, combo, stake, rate))
    raw = 100 * r * (v + s) / (w + s)
    if raw >= 110:
        return math.floor(raw / 10) * 10
    return 100 if 110 * (w + s) > 100 * (v + s) else 110


def coplacer_dividend(
    pool: float,
    money: Mapping[str, float],
    paying: Sequence[str],
    combo: str,
    stake: float,
    rate: float,
    places: int,
    *,
    withheld_ok: bool = True,
) -> int:
    """`combo`'s dividend per 100 yen when the combinations `paying` (combo among them) pay:
    r (1 + L / (P (W_combo + s))) per yen, L = V - sum W over `paying`, broken to 10 yen, at least
    110, and 100 when paying 110 would take the pool's whole payout over V + s. `money` is the
    market's money per combination, in yen, without our stake."""
    s, r = _exact(stake), _exact(rate)
    total = _exact(pool) + s
    w = {j: _exact(money[j]) for j in paying}
    w[combo] += s
    lost = total - sum(w.values())

    def raw_of(j: str) -> Fraction:
        return 100 * r * (1 + lost / (places * w[j]))

    raw = raw_of(combo)
    if raw >= 110:
        return math.floor(raw / 10) * 10
    others = sum(w[j] * max(110, math.floor(raw_of(j) / 10) * 10) for j in paying if j != combo)
    if others + 110 * w[combo] > 100 * total:
        assert withheld_ok, "fixture error: a random draw reached the withheld Plus 10"
        return 100
    return 110


def paying_combinations(outcome: tuple[int, int, int], *, wide: bool, places: int) -> list[str]:
    if wide:
        return [f"{a},{b}" for a, b in itertools.combinations(sorted(outcome), 2)]
    return [str(h) for h in outcome[:places]]


def place_places(field: int) -> int:
    """jra.yaml: PLACE pays 3 from 8 runners at sale, 2 from 5, and is not sold to 4 or fewer."""
    return 3 if field >= 8 else 2 if field >= 5 else 0


def runners_of(combinations: Sequence[str]) -> list[int]:
    return sorted({int(h) for c in combinations for h in c.split(",")})


def ref_ev_single(
    p_by_draw: Sequence[float],
    draws: Draws,
    combinations: Sequence[str],
    combo: str,
    stake: float,
    rate: float,
) -> float:
    j = list(combinations).index(combo)
    total = 0.0
    for p, q, v in zip(p_by_draw, draws.shares, draws.v_final, strict=True):
        total += p * paid_dividend(float(v), float(q[j] * v), stake, rate) / 100
    return total / len(p_by_draw) - 1


def ref_ev_coplacer(
    probs_by_draw: Sequence[Mapping[tuple[int, int, int], float]],
    draws: Draws,
    combinations: Sequence[str],
    combo: str,
    stake: float,
    rate: float,
    places: int,
    *,
    withheld_ok: bool = True,
    coplacers: bool = True,
) -> float:
    """Sum over every ordered top three o of P(o) x combo's dividend given o's co-placers, then
    the mean over draws, less the stake. `coplacers=False` is a wrong rule kept to show the
    fixtures tell it apart: the co-placers' money counted as lost."""
    wide = "," in combo
    runners = runners_of(combinations)
    total = 0.0
    for probs, q, v in zip(probs_by_draw, draws.shares, draws.v_final, strict=True):
        money = {c: float(q[j] * v) for j, c in enumerate(combinations)}
        dividends: dict[frozenset, int] = {}
        acc = 0.0
        for outcome in itertools.permutations(runners, 3):
            p = probs.get(outcome, 0.0)
            pays = paying_combinations(outcome, wide=wide, places=places)
            if not p or combo not in pays:
                continue
            key = frozenset(pays)
            if key not in dividends:
                priced = pays if coplacers else [combo]
                dividends[key] = coplacer_dividend(
                    float(v), money, priced, combo, stake, rate, places, withheld_ok=withheld_ok
                )
            acc += p * dividends[key] / 100
        total += acc
    return total / len(probs_by_draw) - 1


def harville(strength: Mapping[int, float]) -> dict[tuple[int, int, int], float]:
    """P(i, j, k) = s_i / S x s_j / (S - s_i) x s_k / (S - s_i - s_j): sums to 1."""
    total = sum(strength.values())
    return {
        (i, j, k): strength[i]
        / total
        * strength[j]
        / (total - strength[i])
        * strength[k]
        / (total - strength[i] - strength[j])
        for i, j, k in itertools.permutations(strength, 3)
    }


def win_fn(combinations: Sequence[str], combo: str, gamma: float):
    """A synthetic P(combo | x, q): q**gamma normalised, read at combo; checks q is one row."""
    j = list(combinations).index(combo)

    def p_fn(q):
        q = np.asarray(q, dtype=float)
        assert q.shape == (len(combinations),), q.shape
        w = q**gamma
        return float(w[j] / w.sum())

    return p_fn


def triples_fn(combinations: Sequence[str], gamma: float):
    """A synthetic P(o | x, q): Harville on each runner's share of q (summed over its pairs for
    Wide) to the power gamma; checks q is one row."""

    def p_triples_fn(q):
        q = np.asarray(q, dtype=float)
        assert q.shape == (len(combinations),), q.shape
        strength: dict[int, float] = {}
        for c, x in zip(combinations, q, strict=True):
            for h in c.split(","):
                strength[int(h)] = strength.get(int(h), 0.0) + float(x)
        return harville({h: s**gamma for h, s in strength.items()})

    return p_triples_fn


def certain(outcome_probs: Mapping[tuple[int, int, int], float]):
    """A p_triples_fn that ignores q."""

    def p_triples_fn(q):
        return dict(outcome_probs)

    return p_triples_fn


def random_draws(
    rng: np.random.Generator,
    n: int,
    k: int,
    *,
    alpha: float = 1.5,
    max_share: float = 1.0,
    pools: tuple[float, float] = (2e5, 5e7),
) -> Draws:
    rows: list[np.ndarray] = []
    while len(rows) < n:
        q = rng.dirichlet(np.full(k, alpha))
        if q.max() <= max_share and q.min() > 1e-6:
            rows.append(q)
    v = np.exp(rng.uniform(math.log(pools[0]), math.log(pools[1]), size=n))
    return Draws(shares=np.array(rows), v_final=v)


def money_draws(rows: Sequence[Mapping[str, float]], combinations: Sequence[str]) -> Draws:
    """Draws from explicit money per combination (yen), one mapping per draw."""
    v = np.array([float(sum(r.values())) for r in rows])
    shares = np.array([[r[c] / t for c in combinations] for r, t in zip(rows, v, strict=True)])
    return Draws(shares=shares, v_final=v)


def pair_labels(horses: Sequence[int]) -> tuple[str, ...]:
    return tuple(f"{a},{b}" for a, b in itertools.combinations(sorted(horses), 2))


# ---------------------------------------------------------------- #432: own_stake_dividend


def test_own_stake_lowers_the_dividend():
    """#100's example: pool 1,000,000 yen, 10,000 on the combination, payout rate 0.725: 7,250
    per 100 yen with no stake and 6,590 with a 1,000-yen stake."""
    assert own_stake_dividend(1_000_000, 10_000, 0, 0.725) == 7250
    assert own_stake_dividend(1_000_000, 10_000, 1_000, 0.725) == 6590

    # The same by keyword, and as a Python int.
    got = own_stake_dividend(
        pool_total=1_000_000, combo_money=10_000, stake=1_000, payout_rate=0.725
    )
    assert got == 6590 and isinstance(got, int) and not isinstance(got, bool)

    # Exact multiples of 10 survive binary floating point: 100 x 0.8 x 500,000 / 40,000 = 1,000.
    assert own_stake_dividend(500_000, 40_000, 0, 0.8) == 1000
    assert own_stake_dividend(2_000_000, 50_000, 0, 0.775) == 3100
    assert own_stake_dividend(1_200_000, 30_000, 0, 0.75) == 3000

    # A spread of pools, money and stakes: the dividend is the identity, broken to 10 yen, and it
    # falls as the stake rises.
    rng = np.random.default_rng(4321)
    for _ in range(400):
        pool = int(np.exp(rng.uniform(np.log(2e5), np.log(5e9))))
        combo = max(100, int(pool * np.exp(rng.uniform(np.log(1e-5), np.log(0.6)))) // 100 * 100)
        rate = RATES[int(rng.integers(len(RATES)))]
        stakes = sorted({0, 100, 1_000, 10_000, int(rng.integers(100, 200_000)) // 100 * 100})
        previous = None
        for stake in stakes:
            want = expected_dividend(pool, combo, stake, rate)
            got = own_stake_dividend(float(pool), float(combo), float(stake), float(rate))
            assert got == want, (pool, combo, stake, rate, got, want)
            assert isinstance(got, int) and got % 10 == 0
            if previous is not None:
                assert got <= previous, (pool, combo, stake, rate)
            previous = got


def test_own_stake_dividend_never_pays_below_110():
    """The Plus-10 floor: a dividend that breaks below 110 pays 110; one from 110 up breaks to
    its multiple of 10 as usual. (W + s) / (V + s) stays at 0.85 or less here, where 110 is paid
    whatever #100 later adds for the withheld case."""
    pool = 1_000_000
    for combo, rate, want in (
        (800_000, "0.8", 110),  # 100 x 0.8 x 1.25 = 100
        (750_000, "0.8", 110),  # 106.67
        (700_000, "0.8", 110),  # 114.29: breaks to 110
        (690_000, "0.8", 110),  # 115.94
        (640_000, "0.8", 120),  # 125.00: breaks to 120
        (600_000, "0.75", 120),  # 125.00
        (680_000, "0.75", 110),  # 110.29
        (720_000, "0.725", 110),  # 100.69
        (850_000, "0.7", 110),  # 82.35
    ):
        assert expected_dividend(pool, combo, 0, rate) == want
        assert own_stake_dividend(pool, combo, 0, float(rate)) == want, (combo, rate)

    # A stake large enough to sink a long price to the floor: never below 110, always a multiple
    # of 10, never rising with the stake.
    for pool, combo, rate in (
        (1_000_000, 10_000, "0.8"),
        (3_000_000, 250_000, "0.75"),
        (50_000_000, 400_000, "0.725"),
    ):
        previous = None
        floored = False
        for stake in np.unique(np.geomspace(100, 5 * pool, 120).astype(int) // 100 * 100):
            stake = int(stake)
            if (combo + stake) / (pool + stake) > 0.85:
                continue
            got = own_stake_dividend(pool, combo, stake, float(rate))
            assert got == expected_dividend(pool, combo, stake, rate), (pool, combo, stake)
            assert got >= 110 and got % 10 == 0
            if previous is not None:
                assert got <= previous
            previous = got
            floored = floored or got == 110
        assert floored, (pool, combo, rate)  # the sweep reached the floor


@pytest.mark.parametrize(
    "pool, combo, stake, rate",
    [
        (0, 0, 100, 0.8),  # no pool
        (-1_000_000, 10_000, 100, 0.8),
        (1_000_000, -10_000, 100, 0.8),  # negative money on the combination
        (1_000_000, 1_000_001, 100, 0.8),  # more on the combination than in the pool
        (1_000_000, 10_000, -100, 0.8),  # a negative stake
        (1_000_000, 0, 0, 0.8),  # nobody on the combination: no dividend
        (1_000_000, 10_000, 100, 0.0),
        (1_000_000, 10_000, 100, -0.2),
        (1_000_000, 10_000, 100, 1.2),
        (float("nan"), 10_000, 100, 0.8),
        (1_000_000, float("nan"), 100, 0.8),
        (1_000_000, 10_000, float("nan"), 0.8),
        (1_000_000, 10_000, 100, float("nan")),
        (float("inf"), 10_000, 100, 0.8),
        (1_000_000, 10_000, float("inf"), 0.8),
    ],
)
def test_own_stake_dividend_rejects_impossible_inputs(pool, combo, stake, rate):
    with pytest.raises(ValueError):
        own_stake_dividend(pool, combo, stake, rate)


# ---------------------------------------------------------------- #100: EV over nowcast draws


def test_ev_integrates_over_nowcast_draws():
    """EV is the mean over draws of P(c | x, q_m) x D(q_m, V_m, s), never P-bar x D-bar: the
    ticket's two equally likely draws, (P, D) = (0.2, 6) and (0.1, 12), give 0.2, where the plug-in
    would give 0.35; then a spread of fields, pools, stakes and rates against the reference."""
    combos = ("1", "2", "3", "4")
    draws = Draws(
        shares=np.array([[0.5, 0.25, 0.125, 0.125], [0.5, 0.3125, 0.0625, 0.125]]),
        v_final=np.array([800_000.0, 800_000.0]),
    )

    def p_fn(q):  # 0.2 on the first draw, 0.1 on the second: P moves with the draw
        return 1.6 * float(np.asarray(q)[2])

    # D = 100 x 0.75 x V / W: 600 at W = 100,000, 1,200 at W = 50,000 (no stake).
    assert [paid_dividend(800_000, w, 0, "0.75") for w in (100_000, 50_000)] == [600, 1200]
    ev = decision.ev_single_winner(p_fn, draws, "3", 0, 0.75, combinations=combos)
    assert isinstance(ev, float)
    assert ev == pytest.approx(0.2, abs=1e-12)
    plug_in = np.mean([0.2, 0.1]) * np.mean([6.0, 12.0]) - 1
    assert plug_in == pytest.approx(0.35) and abs(ev - plug_in) > 0.1

    # The mean over the draws it is given: their order and repeating every draw change nothing.
    flipped = Draws(shares=draws.shares[::-1].copy(), v_final=draws.v_final[::-1].copy())
    tripled = Draws(shares=np.tile(draws.shares, (3, 1)), v_final=np.tile(draws.v_final, 3))
    for other in (flipped, tripled):
        got = decision.ev_single_winner(p_fn, other, "3", 0, 0.75, combinations=combos)
        assert got == pytest.approx(0.2, abs=1e-12)

    # A spread of single-winner pools (Win numbers and Quinella pairs as labels), each draw with
    # its own pool and shares, P a function of the draw: the reference, and not the plug-in.
    rng = np.random.default_rng(100)
    apart = 0
    for case in range(36):
        k = int(rng.choice([3, 5, 8, 12, 18]))
        n = int(rng.choice([1, 2, 7, 25]))
        labels = (
            tuple(str(h) for h in range(1, k + 1))
            if case % 2
            else tuple(f"{a},{b}" for a, b in itertools.combinations(range(1, 9), 2))[:k]
        )
        draws = random_draws(rng, n, k, alpha=float(rng.choice([0.6, 1.5, 4.0])))
        combo = labels[int(rng.integers(k))]
        stake = int(rng.choice([0, 100, 1_000, 30_000]))
        rate = float(RATES[int(rng.integers(len(RATES)))])
        p_fn = win_fn(labels, combo, float(rng.uniform(0.7, 1.6)))
        p = [p_fn(q) for q in draws.shares]
        want = ref_ev_single(p, draws, labels, combo, stake, rate)
        got = decision.ev_single_winner(p_fn, draws, combo, stake, rate, combinations=labels)
        assert got == pytest.approx(want, rel=TOL, abs=TOL), (case, k, n, combo, stake, rate)
        j = labels.index(combo)
        pairs = zip(draws.shares, draws.v_final, strict=True)
        d_bar = np.mean([paid_dividend(float(v), float(q[j] * v), stake, rate) for q, v in pairs])
        apart += abs(want - (np.mean(p) * d_bar / 100 - 1)) > 1e-6
    assert apart >= 12  # the table tells the mean of P x D from the plug-in


def test_ev_takes_our_own_stake_inside_each_draw():
    """Our stake is in the dividend of every draw: with pool 1,000,000, 10,000 on the horse and a
    rate of 0.725, P = 0.02 is worth 0.02 x 72.50 - 1 = 0.45 with no stake (exactly: 7,250 is no
    7,240 of binary floating point) and 0.02 x 65.90 - 1 = 0.318 with 1,000; the EV never rises
    with the stake. The stake moves a small pool's dividend more than a deep one's."""
    combos = ("1", "2")
    one = Draws(shares=np.array([[0.01, 0.99]]), v_final=np.array([1_000_000.0]))

    def p_fn(q):
        return 0.02

    got = [
        decision.ev_single_winner(p_fn, one, "1", stake, 0.725, combinations=combos)
        for stake in (0, 1_000)
    ]
    assert got == pytest.approx([0.45, 0.318], abs=1e-12)

    previous = math.inf
    for stake in (0, 100, 1_000, 5_000, 10_000, 30_000):
        ev = decision.ev_single_winner(p_fn, one, "1", stake, 0.725, combinations=combos)
        assert ev == pytest.approx(
            0.02 * paid_dividend(1_000_000, 10_000, stake, "0.725") / 100 - 1
        )
        assert ev <= previous
        previous = ev

    # Two draws with the same shares and pools of 100,000 and 100,000,000: the stake of 10,000
    # costs the small pool's draw far more, so the EV is the mean of the two dividends.
    deep = Draws(shares=np.array([[0.05, 0.95]] * 2), v_final=np.array([100_000.0, 1e8]))
    small, large = paid_dividend(1e5, 5e3, 10_000, 0.8), paid_dividend(1e8, 5e6, 10_000, 0.8)
    assert (small, large) == (580, 1590)  # 800 and 1,600 without the stake
    want = 0.5 * 0.02 * (small + large) / 100 - 1
    got = decision.ev_single_winner(p_fn, deep, "1", 10_000, 0.8, combinations=combos)
    assert got == pytest.approx(want, abs=1e-12)

    # Place: the same outcome pays 390 with a 1,000-yen stake and 370 with 10,000 (below).
    place = money_draws([PLACE_MONEY_8], PLACE_8)
    sure = certain({(1, 2, 3): 1.0})
    for stake, dividend in ((1_000, 390), (10_000, 370)):
        ev = decision.ev_place_wide(sure, place, "1", stake, 0.8, combinations=PLACE_8)
        assert ev == pytest.approx(dividend / 100 - 1, abs=1e-12)


# Place: eight runners, 2,000,000 yen; horse 1 has 100,000 on it.
PLACE_8 = tuple(str(h) for h in range(1, 9))
PLACE_MONEY_8 = {
    "1": 100_000, "2": 300_000, "3": 400_000, "4": 50_000,
    "5": 50_000, "6": 500_000, "7": 350_000, "8": 250_000,
}  # fmt: skip
PLACE_7 = tuple(str(h) for h in range(1, 8))
PLACE_MONEY_7 = {**{c: PLACE_MONEY_8[c] for c in PLACE_7[:6]}, "7": 600_000}
# Wide: five runners, ten pairs, 1,000,000 yen; the pair 1-2 has 40,000 on it.
WIDE_5 = pair_labels(range(1, 6))
WIDE_YEN_5 = (40_000, 100_000, 60_000, 50_000, 150_000, 80_000, 70_000, 200_000, 120_000, 130_000)
WIDE_MONEY_5 = dict(zip(WIDE_5, WIDE_YEN_5, strict=True))


def test_place_and_wide_pay_the_stake_back_share_with_their_coplacers():
    """A Place or Wide ticket's dividend depends on who else pays with it. Place, 8 runners
    (3 places), rate 0.8, 1,000 yen on horse 1 (100,000 on it, pool 2,000,000):
    - 1-2-3 pays 1, 2 and 3: L = 2,000,000 - 800,000, 0.8 (1 + 1,200,000 / (3 x 101,000)) = 3.968,
      so 390 (the form 1 + (rV - sum W) / (P W) would pay 360, the single-winner identity 1,580);
    - 4-1-5 pays 4, 1 and 5: L = 1,800,000, so 0.8 (1 + 1,800,000 / 303,000) = 5.552: 550;
    - 2-3-1 pays the same three as 1-2-3: 390; with P = 0.25, 0.25 and 0.5 the EV is
      0.25 x 3.9 + 0.25 x 5.5 + 0.5 x 3.9 - 1 = 3.3.
    Seven runners pay two places: 1-2-3 pays 1 and 2 (0.8 (1 + 1,600,000 / 202,000): 710), 4-1-5
    pays 4 and 1 (810) and 2-3-1 nothing: 0.25 x 7.1 + 0.25 x 8.1 - 1 = 2.8; sold to 8 and left
    with 7 after a scratch, three places as before. Wide, 5 runners, rate 0.775, 1,000 yen on 1-2
    (40,000): 1-2-3 pays 1-2, 1-3 and 2-3 (L = 710,000: 520), 4-1-2 pays 1-4, 1-2 and 2-4
    (L = 820,000: 590), 1-3-4 not 1-2: 0.3 x 5.2 + 0.2 x 5.9 - 1 = 1.74."""

    # The reference against the hand figures above.
    def dividend(money, paying, combo, places, rate):
        return coplacer_dividend(sum(money.values()), money, paying, combo, 1_000, rate, places)

    assert dividend(PLACE_MONEY_8, ["1", "2", "3"], "1", 3, 0.8) == 390
    assert dividend(PLACE_MONEY_8, ["4", "1", "5"], "1", 3, 0.8) == 550
    assert dividend(PLACE_MONEY_7, ["1", "2"], "1", 2, 0.8) == 710
    assert dividend(PLACE_MONEY_7, ["4", "1"], "1", 2, 0.8) == 810
    assert dividend(WIDE_MONEY_5, ["1,2", "1,3", "2,3"], "1,2", 3, 0.775) == 520
    assert dividend(WIDE_MONEY_5, ["1,2", "1,4", "2,4"], "1,2", 3, 0.775) == 590

    outcomes = certain({(1, 2, 3): 0.25, (4, 1, 5): 0.25, (2, 3, 1): 0.5})
    place8 = money_draws([PLACE_MONEY_8], PLACE_8)
    place7 = money_draws([PLACE_MONEY_7], PLACE_7)
    for draws, combinations, field, want in (
        (place8, PLACE_8, None, 3.3),
        (place7, PLACE_7, None, 2.8),
        (place7, PLACE_7, 7, 2.8),
        (place7, PLACE_7, 8, 3.3),  # sold to eight: three places, whoever started
    ):
        kwargs = {} if field is None else {"field_at_sale": field}
        got = decision.ev_place_wide(
            outcomes, draws, "1", 1_000, 0.8, combinations=combinations, **kwargs
        )
        assert isinstance(got, float)
        assert got == pytest.approx(want, abs=1e-12), (len(combinations), field)

    wide = money_draws([WIDE_MONEY_5], WIDE_5)
    wide_outcomes = certain({(1, 2, 3): 0.3, (4, 1, 2): 0.2, (1, 3, 4): 0.5})
    got = decision.ev_place_wide(wide_outcomes, wide, "1,2", 1_000, 0.775, combinations=WIDE_5)
    assert got == pytest.approx(1.74, abs=1e-12)
    # 3-4-5 pays neither ticket: the EV of a sure miss is the stake lost.
    miss = certain({(3, 4, 5): 1.0})
    assert decision.ev_place_wide(miss, wide, "1,2", 1_000, 0.775, combinations=WIDE_5) == (
        pytest.approx(-1.0, abs=1e-12)
    )
    assert decision.ev_place_wide(miss, place8, "1", 1_000, 0.8, combinations=PLACE_8) == (
        pytest.approx(-1.0, abs=1e-12)
    )


def test_place_ev_enumerates_coplacer_outcomes():
    """18 runners give 4,896 ordered triples, and every field n gives each of its n(n-1)(n-2)
    once, made of the numbers given. Place and Wide EV sum over every one of them, each paying
    the ticket by its own co-placers, in every draw: the reference over fields of 5 to 18, with
    Place at 2 and 3 places and Wide at 3."""
    triples = list(decision.place_outcomes(range(1, 19)))
    assert len(triples) == 4896 == 18 * 17 * 16
    assert len(set(triples)) == 4896
    assert set(triples) == set(itertools.permutations(range(1, 19), 3))
    assert all(isinstance(t, tuple) and len(t) == 3 for t in triples)
    assert all(isinstance(h, int) for t in triples[:50] for h in t)

    fields = (
        [1, 2, 3],
        [7, 2, 11, 5],
        list(range(1, 9)),
        [3, 1, 4, 9, 12, 16, 2],
        [14, *range(1, 11)],
    )
    for numbers in fields:
        n = len(numbers)
        got = decision.place_outcomes(numbers)
        assert isinstance(got, Iterator)
        got = list(got)
        assert len(got) == len(set(got)) == n * (n - 1) * (n - 2)
        assert set(got) == set(itertools.permutations(numbers, 3)), numbers

    rng = np.random.default_rng(4896)
    for k_runners, wide, n_draws in (
        (18, False, 3),
        (10, False, 6),
        (8, False, 6),
        (7, False, 6),
        (5, False, 8),
        (8, True, 5),
        (6, True, 6),
        (4, True, 6),
    ):
        horses = list(range(1, k_runners + 1))
        combinations = pair_labels(horses) if wide else tuple(str(h) for h in horses)
        places = 3 if wide else place_places(k_runners)
        rate = POOL_RATE[WIDE if wide else PLACE]
        draws = random_draws(
            rng, n_draws, len(combinations), alpha=2.0, max_share=0.4, pools=(1e5, 2e7)
        )
        for combo in [combinations[0], combinations[-1], combinations[len(combinations) // 2]]:
            stake = int(rng.choice([0, 1_000, 10_000]))
            p_triples = triples_fn(combinations, float(rng.uniform(0.8, 1.4)))
            probs = [p_triples(q) for q in draws.shares]
            want = ref_ev_coplacer(
                probs, draws, combinations, combo, stake, rate, places, withheld_ok=False
            )
            got = decision.ev_place_wide(
                p_triples, draws, combo, stake, rate, combinations=combinations
            )
            assert got == pytest.approx(want, rel=TOL, abs=TOL), (k_runners, wide, combo, stake)

    # The co-placers matter: the same outcomes with the co-placers' money counted as lost (the
    # ticket priced as if it paid alone) are not the EV, in Place or in Wide.
    for combinations, combo, rate in (
        (PLACE_8, "3", 0.8),
        (pair_labels(range(1, 7)), "2,5", 0.775),
    ):
        draws = random_draws(rng, 4, len(combinations), alpha=2.0, max_share=0.4, pools=(1e5, 2e7))
        p_triples = triples_fn(combinations, 1.0)
        probs = [p_triples(q) for q in draws.shares]
        want = ref_ev_coplacer(probs, draws, combinations, combo, 1_000, rate, 3)
        alone = ref_ev_coplacer(probs, draws, combinations, combo, 1_000, rate, 3, coplacers=False)
        assert abs(want - alone) > 0.05
        got = decision.ev_place_wide(
            p_triples, draws, combo, 1_000, rate, combinations=combinations
        )
        assert got == pytest.approx(want, rel=TOL, abs=TOL)


def test_the_floor_and_plus10_apply_to_the_dividend():
    """Breakage to 10 yen, the 110-yen floor (Plus 10), and the withheld Plus 10: 100 yen (the
    stake back) when paying 110 to every winning ticket would exceed the pool, our stake in.
    Single-winner: 110 (W + s) > 100 (V + s); exactly the pool pays 110. Then the EV of a sure
    winner is the dividend less the stake: 0.1 at the floor, 0 when it is withheld, for Win,
    Place and Wide alike."""
    for pool, combo, stake, rate, want in (
        (3_000_000, 70_000, 0, "0.8", 3420),  # 3,428.57 breaks to 3,420
        (1_000_000, 10_000, 1_000, "0.725", 6590),
        (1_000_000, 750_000, 0, "0.8", 110),  # 106.67: the floor
        (1_000_000, 900_000, 0, "0.8", 110),  # 88.89; 110 x 900,000 is under the pool
        (1_100_000, 1_000_000, 0, "0.8", 110),  # 110 x W is exactly the pool: not over it
        (1_000_000, 900_000, 100_000, "0.8", 110),  # 110 (W + s) is exactly V + s
        (1_100_000, 1_000_100, 0, "0.8", 100),  # just over the pool: the stake back
        (1_000_000, 910_000, 0, "0.8", 100),
        (1_000_000, 1_000_000, 0, "0.8", 100),  # every yen on the winner
        (1_000_000, 900_000, 200_000, "0.8", 100),  # our own stake tips it over
        (2_000_000, 1_900_000, 1_000, "0.75", 100),
        (1_000_000, 950_000, 0, "0.7", 100),
    ):
        assert paid_dividend(pool, combo, stake, rate) == want, (pool, combo, stake, rate)
        got = own_stake_dividend(pool, combo, stake, float(rate))
        assert got == want, (pool, combo, stake, rate, got)
        assert isinstance(got, int) and not isinstance(got, bool)

    # A sweep up to every yen on the combination: the reference everywhere, and both the floor
    # and the withheld case reached.
    rng = np.random.default_rng(110)
    seen = set()
    for _ in range(600):
        pool = int(np.exp(rng.uniform(np.log(1e5), np.log(1e9)))) // 100 * 100
        combo = int(pool * rng.uniform(0.3, 1.0)) // 100 * 100
        stake = int(rng.choice([0, 100, 1_000, 30_000]))
        rate = RATES[int(rng.integers(len(RATES)))]
        want = paid_dividend(pool, combo, stake, rate)
        got = own_stake_dividend(float(pool), float(combo), float(stake), float(rate))
        assert got == want, (pool, combo, stake, rate, got, want)
        assert got == 100 or (got >= 110 and got % 10 == 0)
        seen.add(min(got, 120))
    assert {100, 110, 120} <= seen

    # Win: a sure winner at the floor and withheld, and the mean of the two.
    combos = tuple(str(h) for h in range(1, 3))
    floor_draw = [0.75, 0.25]  # 80 x 1,025,000 / 769,000 = 106.6: 110; 110 x 769,000 < V + s
    withheld_draw = [0.9375, 0.0625]  # 110 x 961,000 > 100 x 1,025,000: 100

    def sure(q):
        return 1.0

    for rows, want in (
        ([floor_draw], 0.1),
        ([withheld_draw], 0.0),
        ([floor_draw, withheld_draw], 0.05),
    ):
        draws = Draws(shares=np.array(rows), v_final=np.full(len(rows), 1_024_000.0))
        got = decision.ev_single_winner(sure, draws, "1", 1_000, 0.8, combinations=combos)
        assert got == pytest.approx(want, abs=1e-12), rows

    # Place, nine runners (3 places), 1,024,000 yen, 1,000 on horse 1, 1-2-3 for certain.
    nine = tuple(str(h) for h in range(1, 10))
    floor_money = {"1": 512_000, **{c: 64_000 for c in nine[1:]}}
    # 0.8 (1 + 384,000 / (3 x 513,000)) = 0.9996: 110; the payout, 564,300 + 2 x 64,000 x 2.40,
    # is well under the pool
    withheld_money = {"1": 960_000, **{c: 8_000 for c in nine[1:]}}
    # 0.8 (1 + 48,000 / (3 x 961,000)) = 0.813; 110 x 961,000 alone is over 1,025,000: 100
    first = certain({(1, 2, 3): 1.0})
    for rows, want in (
        ([floor_money], 0.1),
        ([withheld_money], 0.0),
        ([floor_money, withheld_money], 0.05),
    ):
        for row in rows:
            got_row = coplacer_dividend(sum(row.values()), row, ["1", "2", "3"], "1", 1_000, 0.8, 3)
            assert got_row == (110 if row is floor_money else 100)
        draws = money_draws(rows, nine)
        got = decision.ev_place_wide(first, draws, "1", 1_000, 0.8, combinations=nine)
        assert got == pytest.approx(want, abs=1e-12), len(rows)

    # Wide, four runners (sold from 4: three places), 1,000 on 1-2, 1-2-3 for certain.
    four = pair_labels(range(1, 5))  # 1,2 1,3 1,4 2,3 2,4 3,4
    floor_wide = dict(zip(four, (600_000, 100_000, 50_000, 100_000, 50_000, 100_000), strict=True))
    # 0.775 (1 + 200,000 / (3 x 601,000)) = 0.861: 110; payout 661,100 + 2 x 100,000 x 1.20
    withheld_wide = dict(zip(four, (960_000, 12_800, 12_800, 12_800, 12_800, 12_800), strict=True))
    # 0.775 (1 + 38,400 / (3 x 961,000)) = 0.785; 110 x 961,000 is over 1,025,000: 100
    for row, want in ((floor_wide, 0.1), (withheld_wide, 0.0)):
        paying = ["1,2", "1,3", "2,3"]
        assert coplacer_dividend(sum(row.values()), row, paying, "1,2", 1_000, 0.775, 3) == (
            110 if want else 100
        )
        draws = money_draws([row], four)
        got = decision.ev_place_wide(first, draws, "1,2", 1_000, 0.775, combinations=four)
        assert got == pytest.approx(want, abs=1e-12), row


def test_parameter_uncertainty_comes_from_refits():
    """One EV per refit, in refit order: two refits that differ give an EV distribution with
    positive spread, identical refits give none, for WIN, PLACE and Wide."""
    rng = np.random.default_rng(87)
    win = tuple(str(h) for h in range(1, 9))
    draws = random_draws(rng, 6, 8)
    a, b = win_fn(win, "4", 1.0), win_fn(win, "4", 1.4)
    want = [
        ref_ev_single([f(q) for q in draws.shares], draws, win, "4", 1_000, 0.8) for f in (a, b)
    ]
    assert abs(want[0] - want[1]) > 1e-3
    got = decision.ev_distribution([a, b], draws, "4", 1_000, 0.8, combinations=win, pool=WIN)
    assert isinstance(got, np.ndarray) and got.shape == (2,)
    assert np.allclose(got, want, rtol=TOL, atol=TOL)
    assert np.std(got) > 0

    same = decision.ev_distribution([a, a], draws, "4", 1_000, 0.8, combinations=win, pool=WIN)
    assert same.shape == (2,) and np.ptp(same) == 0 and same[0] == pytest.approx(want[0])

    gammas = (0.8, 1.0, 1.1, 1.25, 1.6)
    for r in range(1, 6):
        refits = [win_fn(win, "4", g) for g in gammas[:r]]
        got = decision.ev_distribution(refits, draws, "4", 1_000, 0.8, combinations=win, pool=WIN)
        want_r = [
            ref_ev_single([f(q) for q in draws.shares], draws, win, "4", 1_000, 0.8) for f in refits
        ]
        assert got.shape == (r,) and np.allclose(got, want_r, rtol=TOL, atol=TOL)

    for pool, combinations, combo in (
        (PLACE, tuple(str(h) for h in range(1, 9)), "2"),
        (WIDE, pair_labels(range(1, 7)), "2,5"),
    ):
        places = 3 if pool == WIDE else place_places(len(combinations))
        draws = random_draws(rng, 4, len(combinations), alpha=2.0, max_share=0.4)
        refits = [triples_fn(combinations, 1.0), triples_fn(combinations, 1.5)]
        rate = POOL_RATE[pool]
        want = [
            ref_ev_coplacer(
                [f(q) for q in draws.shares],
                draws,
                combinations,
                combo,
                1_000,
                rate,
                places,
                withheld_ok=False,
            )
            for f in refits
        ]
        assert abs(want[0] - want[1]) > 1e-3
        got = decision.ev_distribution(
            refits, draws, combo, 1_000, rate, combinations=combinations, pool=pool
        )
        assert got.shape == (2,) and np.allclose(got, want, rtol=TOL, atol=TOL), pool
        assert np.std(got) > 0
        same = decision.ev_distribution(
            refits[:1] * 3, draws, combo, 1_000, rate, combinations=combinations, pool=pool
        )
        assert same.shape == (3,) and np.ptp(same) == 0


# ---------------------------------------------------------------- #100: the shadow decider

HORSES = (1, 2, 3, 5, 6, 7)  # six runners, horse 4 scratched: Place pays two, Wide three
RUNNERS = tuple(str(h) for h in HORSES)
PAIRS = pair_labels(HORSES)
RACE_ID = "jra:2026-11-22:KYO:11"


class FakeMeta:
    """A C4 refit: P(win) = q**power normalised, from one Win draw's shares."""

    def __init__(self, power: float) -> None:
        self.power = power

    def predict(self, q, view):
        q = np.asarray(q, dtype=float)
        assert q.shape == (len(HORSES),), q.shape
        assert view.race_id == RACE_ID
        w = q**self.power
        return w / w.sum()


def trifecta_array(p_win, lam: float) -> np.ndarray:
    """Discounted Harville: P(i, j, k) = p_i x w_j / (W - w_i) x w_k / (W - w_i - w_j),
    w = p**lam and W its sum."""
    p = np.asarray(p_win, dtype=float)
    w = p**lam
    out = np.zeros((len(p),) * 3)
    for i, j, k in itertools.permutations(range(len(p)), 3):
        out[i, j, k] = p[i] * w[j] / (w.sum() - w[i]) * w[k] / (w.sum() - w[i] - w[j])
    return out


class FakeOrdering:
    """A C5 refit with `Ordering.trifecta`'s interface."""

    def __init__(self, lam: float) -> None:
        self.lam = lam

    def trifecta(self, p_win):
        assert np.asarray(p_win).shape == (len(HORSES),)
        return trifecta_array(p_win, self.lam)


class FakeNowcast:
    """C1's draws interface: fixed draws per pool, whatever n and seed."""

    def __init__(self, draws: Mapping[str, Draws]) -> None:
        self._draws = dict(draws)

    def draws(self, view, *, pool, n, seed):
        assert view.race_id == RACE_ID
        return self._draws[pool]


def shadow_race() -> tuple[RaceView, dict[str, Draws]]:
    """A six-runner race at T-2 with small pools (so 1,000 yen moves the dividends) and three
    draws per pool."""
    rng = np.random.default_rng(2026)

    def board(combinations, total):
        shares = rng.dirichlet(np.full(len(combinations), 2.0))
        return PoolBoard(
            combinations=combinations,
            shares=shares,
            total=total,
            shares_lag5=None,
            shares_lag15=None,
            board_min=2.0,
            odds=0.8 / shares,
        )

    pools = {
        WIN: board(RUNNERS, 300_000.0),
        PLACE: board(RUNNERS, 120_000.0),
        WIDE: board(PAIRS, 90_000.0),
    }
    view = RaceView(RACE_ID, "KYO", 11, None, 2.0, pools)
    draws = {
        WIN: random_draws(rng, 3, len(RUNNERS), alpha=2.0, max_share=0.4, pools=(2.5e5, 4e5)),
        PLACE: random_draws(rng, 3, len(RUNNERS), alpha=2.0, max_share=0.4, pools=(1e5, 1.5e5)),
        WIDE: random_draws(rng, 3, len(PAIRS), alpha=2.0, max_share=0.4, pools=(7e4, 1.1e5)),
    }
    return view, draws


def shadow_models(draws: Mapping[str, Draws]) -> dict:
    return {
        "meta": [FakeMeta(1.0), FakeMeta(1.25)],
        "nowcast": FakeNowcast(draws),
        "exotic": [FakeOrdering(1.0), FakeOrdering(0.8)],
    }


def expected_decisions(draws: Mapping[str, Draws], stake: float) -> dict[tuple[str, str], list]:
    """{(pool, combination): [EV per refit]}: Win draw m gives P, the pool's draw m the dividend."""
    metas, orderings = (1.0, 1.25), (1.0, 0.8)
    out: dict[tuple[str, str], list] = {}
    win = draws[WIN]
    for combo in RUNNERS:
        j = RUNNERS.index(combo)
        out[(WIN, combo)] = []
        for m in metas:
            p = [FakeMeta(m).predict(q, _View)[j] for q in win.shares]
            out[(WIN, combo)].append(ref_ev_single(p, win, RUNNERS, combo, stake, POOL_RATE[WIN]))
    positions = list(itertools.permutations(range(len(HORSES)), 3))
    for pool, combinations in ((PLACE, RUNNERS), (WIDE, PAIRS)):
        places = 3 if pool == WIDE else place_places(len(HORSES))
        for combo in combinations:
            out[(pool, combo)] = []
            for m, lam in zip(metas, orderings, strict=True):
                probs = []
                for q in win.shares:
                    t = trifecta_array(FakeMeta(m).predict(q, _View), lam)
                    probs.append(
                        {tuple(HORSES[x] for x in ijk): float(t[ijk]) for ijk in positions}
                    )
                ev = ref_ev_coplacer(
                    probs,
                    draws[pool],
                    combinations,
                    combo,
                    stake,
                    POOL_RATE[pool],
                    places,
                    withheld_ok=False,
                )
                out[(pool, combo)].append(ev)
    return out


class _View:
    race_id = RACE_ID


def betting_limits() -> tuple[int, int]:
    raw = load_country_yaml("japan", "betting")
    return int(raw["flat_stake_yen"]), int(raw["day_cap_yen"])


def test_shadow_decisions_carry_the_board_the_ev_and_the_flat_stake():
    """Every combination on every pool's board gets one ShadowDecision: the race, pool,
    combination, tau, the board at t, the EV per refit (priced with the flat stake inside the
    dividend; Place and Wide over every co-placer outcome) and the flat stake, within the
    race-day cap. Nothing is selected."""
    flat, cap = betting_limits()
    assert 0 < flat <= cap
    view, draws = shadow_race()
    want = expected_decisions(draws, flat)
    unstaked = expected_decisions(draws, 0)
    for pool in (WIN, PLACE, WIDE):  # the fixture tells our stake's impact apart, in every pool
        moved = [abs(want[k][r] - unstaked[k][r]) for k in want if k[0] == pool for r in (0, 1)]
        assert max(moved) > 1e-6, pool

    out = decision.ShadowDecider(mode="shadow", **shadow_models(draws)).decide(view)
    assert isinstance(out, list)
    keys = [(rec.pool, rec.combination) for rec in out]
    assert len(keys) == len(set(keys)) == len(RUNNERS) * 2 + len(PAIRS)
    assert set(keys) == set(want)
    for rec in out:
        assert isinstance(rec, decision.ShadowDecision)
        board = view.pools[rec.pool]
        assert rec.race_id == RACE_ID
        assert rec.tau_min == view.tau_min
        assert tuple(rec.board.combinations) == board.combinations
        assert np.array_equal(np.asarray(rec.board.shares), board.shares)
        assert rec.stake_yen == flat and rec.stake_yen <= cap
        ev = np.asarray(rec.ev, dtype=float)
        assert ev.shape == (2,), (rec.pool, rec.combination)
        assert np.allclose(ev, want[(rec.pool, rec.combination)], rtol=TOL, atol=TOL), (
            rec.pool,
            rec.combination,
            ev,
            want[(rec.pool, rec.combination)],
        )
        assert np.ptp(ev) > 0  # the two refits differ


# Network, browser, mail and process modules, the racing-site HTTP layer and the manual bet log.
BANNED_IMPORTS = (
    *("socket", "ssl", "http", "urllib", "urllib3", "requests", "httpx", "aiohttp"),
    *("websocket", "websockets", "selenium", "playwright", "smtplib", "ftplib"),
    *("subprocess", "webbrowser", "horsetrack.core.ingest", "horsetrack.countries.japan.bets"),
)
BET_NAME = re.compile(r"(?:^|_)(?:bet|bets|submit|wager|purchase|buy|login)(?:_|$)", re.IGNORECASE)


def test_shadow_mode_never_emits_a_bet(monkeypatch):
    """ShadowDecider refuses every mode but 'shadow' when it is built; decide() returns only
    ShadowDecision records and opens no network connection; decision.py imports nothing that
    reaches a network, a process or the bet log, and has nothing named to place a bet."""
    view, draws = shadow_race()
    for mode in ("live", "replay", "bet", "real", "", "Shadow", "SHADOW", " shadow", "shadow "):
        with pytest.raises(ValueError):
            decision.ShadowDecider(mode=mode, **shadow_models(draws))
    with pytest.raises((ValueError, TypeError)):
        decision.ShadowDecider(mode=None, **shadow_models(draws))

    def refuse(*args, **kwargs):
        raise AssertionError("decide() opened a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    out = decision.ShadowDecider(mode="shadow", **shadow_models(draws)).decide(view)
    monkeypatch.undo()
    assert type(out) is list and out
    assert all(isinstance(rec, decision.ShadowDecision) for rec in out)

    tree = ast.parse(Path(decision.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported |= {f"{node.module}.{alias.name}" for alias in node.names}
    for name in imported:
        assert not any(name == b or name.startswith(b + ".") for b in BANNED_IMPORTS), name

    for owner in (decision, decision.ShadowDecider, decision.ShadowDecision):
        for name in dir(owner):
            assert not (BET_NAME.search(name) and callable(getattr(owner, name))), (owner, name)


# ---------------------------------------------------------------- #100: inputs the EV refuses


EIGHT = tuple(str(h) for h in range(1, 9))


def _draws(shares=None, v=None) -> Draws:
    return Draws(
        shares=np.full((2, 8), 0.125) if shares is None else shares,
        v_final=np.array([1e6, 2e6]) if v is None else v,
    )


def _ev(p=0.1, *, combo="3", stake=1_000, rate=0.8, shares=None, v=None, combinations=EIGHT):
    def p_fn(q):
        return p

    draws = _draws(shares, v)
    return decision.ev_single_winner(p_fn, draws, combo, stake, rate, combinations=combinations)


def _ev_place(probs=None, *, combo="3", combinations=EIGHT, shares=None):
    k = len(combinations)
    draws = _draws(np.full((2, k), 1.0 / k) if shares is None else shares)
    p_triples = certain({(1, 2, 3): 1.0} if probs is None else probs)
    return decision.ev_place_wide(p_triples, draws, combo, 1_000, 0.8, combinations=combinations)


def _dist(refits, pool):
    return decision.ev_distribution(
        refits, _draws(), "3", 1_000, 0.8, combinations=EIGHT, pool=pool
    )


def _nan_shares():
    shares = np.full((2, 8), 0.125)
    shares[1, 4] = np.nan
    return shares


REFUSED = {
    "combo not on the board": lambda: _ev(combo="9"),
    "labels and columns differ": lambda: _ev(combinations=EIGHT[:7]),
    "no draws": lambda: _ev(shares=np.zeros((0, 8)), v=np.zeros(0)),
    "a share is NaN": lambda: _ev(shares=_nan_shares()),
    "a pool is infinite": lambda: _ev(v=np.array([1e6, np.inf])),
    "a pool is not positive": lambda: _ev(v=np.array([1e6, 0.0])),
    "pools and draws differ": lambda: _ev(v=np.array([1e6, 2e6, 3e6])),
    "P is NaN": lambda: _ev(p=float("nan")),
    "P above 1": lambda: _ev(p=1.5),
    "P below 0": lambda: _ev(p=-0.1),
    "a negative stake": lambda: _ev(stake=-100),
    "an infinite stake": lambda: _ev(stake=float("inf")),
    "a rate above 1": lambda: _ev(rate=1.2),
    "a rate of 0": lambda: _ev(rate=0.0),
    "place: combo not on the board": lambda: _ev_place(combo="9"),
    "place: a negative probability": lambda: _ev_place({(1, 2, 3): 1.2, (2, 1, 3): -0.2}),
    "place: a NaN probability": lambda: _ev_place({(1, 2, 3): float("nan")}),
    "place: a NaN share": lambda: _ev_place(shares=_nan_shares()),
    "place: not sold to four runners": lambda: _ev_place(combinations=("1", "2", "3", "4")),
    "refits: none": lambda: _dist([], WIN),
    "refits: a pool it does not price": lambda: _dist([win_fn(EIGHT, "3", 1.0)], "WIN5"),
    "outcomes: a runner twice": lambda: list(decision.place_outcomes([1, 2, 3, 3])),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_ev_rejects_impossible_inputs(case):
    with pytest.raises(ValueError):
        REFUSED[case]()
