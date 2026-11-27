"""Ticket #429: the final-board return scan for every pool by segment (N1's maths, synthetic only).

Contract pinned here for `horsetrack.countries.japan.research.pool_scan` (the coder implements
what these tests call):

- `POOLS == ('WIN', 'PLACE', 'QUINELLA', 'QUINELLA_PLACE', 'FORECAST', 'TRIO')`, the jravan pool
  codes (`jravan/odds.py`, `jravan/dividends.py`): QUINELLA_PLACE is Wide, FORECAST the exacta.
- `class_group(race_class) -> str` over the store's `races.race_class` vocabulary
  (`jravan/races.py`): DEBUT newcomer, MAIDEN maiden, C1/C2/C3 conditions, OP and L open,
  G1/G2/G3 graded. None, NaN, '' or any other value raises ValueError.
- `race_cells(*, race_class, field_size, fav_odds, pool_total, pool_edges, surface, venue)
  -> frozenset[str]`: 'all' plus exactly one `segment:level` label per segment:
  `class:<class_group>`, `field:5-10|11-14|15-18`, `fav:le1.9|2.0-2.9|3.0-4.9|ge5.0` (lower edge
  inclusive: a 2.0 favourite is 2.0-2.9), `pool_size:low|mid|high` (below the first edge, between
  the edges, above the second), `surface:<surface>` and `venue:<venue>` as given. A null or
  unknown class raises ValueError. Which level a pool total exactly on an edge gets is not
  pinned, only that it gets exactly one.
- `odds_band(odds, edges) -> str`: a step function of the odds over the ascending edges, the
  intervals [e_i, e_i+1) lower edge inclusive (as the favourite bands), so k edges give k + 1
  distinct labels. The label text is the coder's.
- `combination_returns(board, paid, *, pool) -> DataFrame`. `board` is `final_odds` rows of
  `pool` in the store's columns (race_id, pool, combination, odds, odds_hi, investment,
  status_raw), the `__pool__` row included; `paid` is `dividends` rows (race_id, pool,
  combination, dividend, is_refund) of every pool. Both may hold many races (a day's or a year's
  rows): a combination matches on (race_id, combination) within `pool`, its text as stored
  (FORECAST legs in finishing order, every other pool ascending). A priced combination is a board
  row with a non-null `odds`; the `__pool__` row and null-odds rows (scratched or unsold, the raw
  text in `status_raw`) are not. The result has one row per priced combination with race_id,
  combination, odds (the board's `odds`, i.e. the low end of the final range for PLACE and
  QUINELLA_PLACE), won (truthy when it paid), ret (dividend / 100 when it paid, else 0) and
  refunded (true when `paid` has an `is_refund` row of `pool` for it). In a dead heat every
  paying row pays its own combination.
- `race_returns(rows, *, payout_rate, band_edges) -> DataFrame`: `rows` are one pool's
  `combination_returns` rows; one row per (race_id, band), band `odds_band(odds, band_edges)`,
  with the flat-box sums of the unrefunded combinations: n_staked (combinations staked), ret_sum
  (their return), rate_sum (payout_rate per combination staked); n_refunded counts the refunded
  combinations, which are in none of the other three. Flat rho = ret_sum.sum() / rate_sum.sum().
  A band with no combination in a race may have a row of zeros or no row.
- `grouped_bets(rows, *, by, payout_rate, band_edges) -> DataFrame`, the input of
  `fit_segment_roi(..., count_col='count')`: `rows` are `combination_returns` rows with the
  segment columns `by` added; one row per distinct (*by, band, odds) of the unrefunded rows, with
  takeout = 1 - payout_rate, count (its combinations) and won (the wins among them).

Every board and dividend here is invented and built in the test: nothing reads the store, the
registry or the holdout.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party.
from __future__ import annotations  # noqa: I001

import ast
import bisect
import itertools
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval import segment_roi
from horsetrack.countries.japan.research import pool_scan as ps

POOLS = ("WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE", "FORECAST", "TRIO")
# Payout rates (1 - takeout of config/jra.yaml, in force since 2014-06-07), hard-coded here.
RATES = {
    "WIN": 0.80,
    "PLACE": 0.80,
    "QUINELLA": 0.775,
    "QUINELLA_PLACE": 0.775,
    "FORECAST": 0.75,
    "TRIO": 0.75,
}
LEGS = {"WIN": 1, "PLACE": 1, "QUINELLA": 2, "QUINELLA_PLACE": 2, "FORECAST": 2, "TRIO": 3}
RANGE_POOLS = ("PLACE", "QUINELLA_PLACE")  # the board shows a low-high range (odds, odds_hi)
POOL_ROW = "__pool__"
BOARD_COLUMNS = ["race_id", "pool", "combination", "odds", "odds_hi", "investment", "status_raw"]
PAID_COLUMNS = ["race_id", "pool", "combination", "dividend", "is_refund"]
OUT_COLUMNS = {"race_id", "combination", "odds", "won", "ret", "refunded"}
SUM_COLUMNS = ["n_staked", "ret_sum", "rate_sum", "n_refunded"]
EDGES = (2.5, 5.0)


# --- synthetic boards and dividends (independent of the module) ----------------------------------


def race_id(i: int) -> str:
    """A store-shaped race id, `jra:<date>:<venue>:<NN>`, in an invented year."""
    day = date(2030, 1, 5) + timedelta(days=i // 36)
    venue = ("TOK", "NAK", "KYO")[(i // 12) % 3]
    return f"jra:{day.isoformat()}:{venue}:{i % 12 + 1:02d}"


def joined(legs) -> str:
    return ",".join(str(int(h)) for h in legs)


def legs_of(combination: str) -> list[int]:
    return [int(x) for x in combination.split(",")]


def combinations_of(pool: str, horses) -> list[str]:
    """Every combination of `pool` over `horses`, written as the store writes it."""
    horses = sorted(int(h) for h in horses)
    if LEGS[pool] == 1:
        return [str(h) for h in horses]
    if pool == "FORECAST":
        return [joined(c) for c in itertools.permutations(horses, 2)]
    return [joined(c) for c in itertools.combinations(horses, LEGS[pool])]


def paying(pool: str, order) -> list[str]:
    """The paying combinations of `pool` for a finishing order without a dead heat."""
    first, second, third = (int(h) for h in order[:3])
    return {
        "WIN": [str(first)],
        "PLACE": [str(first), str(second), str(third)],
        "QUINELLA": [joined(sorted((first, second)))],
        "QUINELLA_PLACE": [
            joined(sorted(pair)) for pair in itertools.combinations((first, second, third), 2)
        ],
        "FORECAST": [joined((first, second))],
        "TRIO": [joined(sorted((first, second, third)))],
    }[pool]


def board(
    race: str,
    pool: str,
    odds: dict[str, float | None],
    odds_hi: dict[str, float] | None = None,
    investment: float = 1_000_000.0,
) -> pd.DataFrame:
    """One race's `final_odds` rows of `pool`: a row per combination and the `__pool__` row."""
    rows = [
        {
            "race_id": race,
            "pool": pool,
            "combination": combination,
            "odds": price,
            "odds_hi": (odds_hi or {}).get(combination),
            "investment": None,
            "status_raw": None if price is not None else "----",
        }
        for combination, price in odds.items()
    ]
    rows.append(
        {
            "race_id": race,
            "pool": pool,
            "combination": POOL_ROW,
            "odds": None,
            "odds_hi": None,
            "investment": investment,
            "status_raw": None,
        }
    )
    frame = pd.DataFrame(rows, columns=BOARD_COLUMNS)
    return frame.astype({"odds": float, "odds_hi": float, "investment": float})


def paid(rows) -> pd.DataFrame:
    """`dividends` rows from (race_id, pool, combination, dividend[, is_refund])."""
    full = [(*r, False) if len(r) == 4 else tuple(r) for r in rows]
    frame = pd.DataFrame(full, columns=PAID_COLUMNS)
    return frame.astype({"dividend": float, "is_refund": bool})


def returns_of(out: pd.DataFrame, race: str | None = None) -> dict[str, float]:
    """combination -> ret of one race's rows (the only race when `race` is None)."""
    mine = out if race is None else out[out["race_id"] == race]
    assert not mine["combination"].duplicated().any()
    return dict(zip(mine["combination"], mine["ret"].astype(float), strict=True))


def totals(rr: pd.DataFrame) -> pd.DataFrame:
    """`race_returns` summed over bands, per race."""
    return rr.groupby("race_id")[SUM_COLUMNS].sum()


def by_band(rr: pd.DataFrame) -> dict[str, tuple[float, ...]]:
    """band -> (n_staked, ret_sum, rate_sum, n_refunded) of one race, empty bands dropped."""
    assert rr["race_id"].nunique() == 1
    assert not rr["band"].duplicated().any()
    rr = rr[(rr["n_staked"] > 0) | (rr["n_refunded"] > 0)]
    return {
        band: tuple(float(row[c]) for c in SUM_COLUMNS)
        for band, row in rr.set_index("band").iterrows()
    }


def approx_bands(expected: dict[str, tuple[float, ...]]) -> dict:
    return {band: pytest.approx(values, abs=1e-9) for band, values in expected.items()}


# --- segments: race cells and class groups -------------------------------------------------------

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


def expected_cells(*, race_class, field_size, fav_odds, pool_total, pool_edges, surface, venue):
    field = "5-10" if field_size <= 10 else "11-14" if field_size <= 14 else "15-18"
    if fav_odds < 2.0:
        fav = "le1.9"
    elif fav_odds < 3.0:
        fav = "2.0-2.9"
    elif fav_odds < 5.0:
        fav = "3.0-4.9"
    else:
        fav = "ge5.0"
    low, high = pool_edges
    size = "low" if pool_total < low else "mid" if pool_total < high else "high"
    return frozenset(
        {
            "all",
            f"class:{CLASS_GROUPS[race_class]}",
            f"field:{field}",
            f"fav:{fav}",
            f"pool_size:{size}",
            f"surface:{surface}",
            f"venue:{venue}",
        }
    )


SEGMENT_PREFIXES = ("class:", "field:", "fav:", "pool_size:", "surface:", "venue:")


def test_race_cells_place_a_race_in_one_level_of_each_segment():
    """DoD: every race maps to 'all' and to exactly one level of each segment."""
    example = ps.race_cells(
        race_class="MAIDEN",
        field_size=12,
        fav_odds=2.4,
        pool_total=2_000_000,
        pool_edges=(1_000_000, 3_000_000),
        surface="turf",
        venue="TOKYO",
    )
    assert example == frozenset(
        {
            "all",
            "class:maiden",
            "field:11-14",
            "fav:2.0-2.9",
            "pool_size:mid",
            "surface:turf",
            "venue:TOKYO",
        }
    )

    # Every class, the field and favourite band edges, pool totals against moving edges.
    fields = (5, 8, 10, 11, 14, 15, 18)
    favourites = (1.0, 1.9, 2.0, 2.4, 2.9, 3.0, 4.9, 5.0, 23.4)
    pools = (
        (500_000, (1_000_000, 3_000_000)),  # low
        (2_000_000, (1_000_000, 3_000_000)),  # mid
        (4_000_000, (1_000_000, 3_000_000)),  # high
        (2_000_000, (2_500_000, 4_000_000)),  # low: the edges moved, not the total
        (2_000_000, (500_000, 1_500_000)),  # high
    )
    for race_class, field_size, fav_odds, (total, edges), surface, venue in itertools.product(
        CLASS_GROUPS, fields, favourites, pools, ("turf", "dirt", "jump"), ("TOK", "KOK")
    ):
        args = dict(
            race_class=race_class,
            field_size=field_size,
            fav_odds=fav_odds,
            pool_total=total,
            pool_edges=edges,
            surface=surface,
            venue=venue,
        )
        cells = ps.race_cells(**args)
        assert cells == expected_cells(**args), args

    # A pool total exactly on an edge still gets exactly one pool_size level.
    for total in (1_000_000, 3_000_000):
        cells = ps.race_cells(
            race_class="C2",
            field_size=16,
            fav_odds=3.7,
            pool_total=total,
            pool_edges=(1_000_000, 3_000_000),
            surface="dirt",
            venue="NAK",
        )
        assert "all" in cells
        for prefix in SEGMENT_PREFIXES:
            assert len([c for c in cells if c.startswith(prefix)]) == 1, (total, prefix, cells)
        assert len(cells) == 1 + len(SEGMENT_PREFIXES)


def test_class_groups_follow_the_jravan_vocabulary():
    """DoD: DEBUT newcomer, MAIDEN maiden, C1-C3 conditions, OP and L open, G1-G3 graded; a null
    or unknown class raises ValueError, in `class_group` and in `race_cells`."""
    for race_class, group in CLASS_GROUPS.items():
        assert ps.class_group(race_class) == group, race_class

    for bad in (None, float("nan"), "XYZ", "", "OPEN", "G4"):
        with pytest.raises(ValueError):
            ps.class_group(bad)
        with pytest.raises(ValueError):
            ps.race_cells(
                race_class=bad,
                field_size=12,
                fav_odds=2.4,
                pool_total=2_000_000,
                pool_edges=(1_000_000, 3_000_000),
                surface="turf",
                venue="TOK",
            )


# --- odds bands ----------------------------------------------------------------------------------


def test_odds_bands_are_steps_over_the_edges():
    """`odds_band`: one label per interval [e_i, e_i+1) of the edges, distinct across intervals."""
    edges = (2.0, 5.0, 10.0)
    rng = np.random.default_rng(4293)
    odds = np.round(np.exp(rng.uniform(0.0, np.log(500.0), 400)), 1)
    odds = np.concatenate([odds, [1.0, 1.9, 2.0, 4.9, 5.0, 9.9, 10.0, 9999.9]])
    label_of_interval: dict[int, str] = {}
    for o in odds:
        label = ps.odds_band(float(o), edges)
        assert isinstance(label, str)
        interval = bisect.bisect_right(edges, float(o))
        assert label_of_interval.setdefault(interval, label) == label, (o, label)
    assert sorted(label_of_interval) == [0, 1, 2, 3]
    assert len(set(label_of_interval.values())) == 4

    # The edges move the steps.
    assert ps.odds_band(2.0, (3.0,)) == ps.odds_band(1.0, (3.0,)) != ps.odds_band(3.0, (3.0,))
    assert ps.odds_band(2.0, edges) != ps.odds_band(1.9, edges)
    assert ps.odds_band(2.0, edges) == ps.odds_band(3.0, edges)


# --- returns at the final dividend ---------------------------------------------------------------


@pytest.mark.parametrize("pool", POOLS)
def test_each_pool_returns_the_dividend_over_100_on_its_paying_combinations_only(pool):
    """DoD: in each of the six pools a priced combination returns dividend/100 when it paid and 0
    otherwise. 25 races of 8-14 starters in one call, each with a scratched horse (its
    combinations unpriced), board odds unrelated to the dividends, and the dividends of every pool
    in `paid`, so the same text paid in another pool (WIN and PLACE '3', QUINELLA, Wide and
    FORECAST '4,7', TIERCE and TRIO) or another race must not count.
    """
    assert ps.POOLS == POOLS
    rng = np.random.default_rng(4290 + POOLS.index(pool))
    boards, paid_rows = [], []
    expected: dict[tuple[str, str], float] = {}
    board_odds: dict[tuple[str, str], float] = {}
    for i in range(25):
        race = race_id(i)
        n = int(rng.integers(8, 15))
        horses = list(range(1, n + 1))
        scratched = int(rng.integers(1, n + 1))
        order = [int(h) for h in rng.permutation([h for h in horses if h != scratched])]
        if i % 5 == 0:
            order[:3] = sorted(order[:3])  # the TIERCE text is then the TRIO text
        odds = {
            c: None if scratched in legs_of(c) else float(rng.integers(11, 10000)) / 10
            for c in combinations_of(pool, horses)
        }
        hi = (
            {c: round(o * 1.6, 1) for c, o in odds.items() if o is not None}
            if pool in RANGE_POOLS
            else None
        )
        boards.append(board(race, pool, odds, hi))
        dividends = {}
        for p in POOLS:
            for c in paying(p, order):
                dividends[(p, c)] = float(rng.integers(11, 5000) * 10)
        dividends[("TIERCE", joined(order[:3]))] = float(rng.integers(11, 9000) * 10)
        dividends[("BRACKET_QUINELLA", joined(sorted(order[:2])))] = float(
            rng.integers(11, 900) * 10
        )
        paid_rows += [(race, p, c, d) for (p, c), d in dividends.items()]
        for c, o in odds.items():
            if o is not None:
                expected[(race, c)] = dividends.get((pool, c), 0.0) / 100
                board_odds[(race, c)] = o
    shuffled = paid(paid_rows).sample(frac=1.0, random_state=1).reset_index(drop=True)

    out = ps.combination_returns(pd.concat(boards, ignore_index=True), shuffled, pool=pool)

    assert OUT_COLUMNS <= set(out.columns)
    assert not out.duplicated(["race_id", "combination"]).any()
    keyed = out.set_index(["race_id", "combination"])
    assert set(keyed.index) == set(expected)  # priced only: no __pool__, no scratched horse
    got = keyed["ret"].astype(float).to_dict()
    assert got == {k: pytest.approx(v, abs=1e-12) for k, v in expected.items()}
    assert keyed["won"].astype(bool).to_dict() == {k: v > 0 for k, v in expected.items()}
    assert keyed["odds"].astype(float).to_dict() == pytest.approx(board_odds)
    assert not keyed["refunded"].astype(bool).any()
    winners = out[out["ret"] > 0].groupby("race_id").size()
    assert (winners == len(paying(pool, [1, 2, 3]))).all() and len(winners) == 25


def test_win_returns_pay_the_winner_only():
    """Win board {1: 2.0, 2: 3.5, 3: 6.0}, horse 2 paid 350: returns {1: 0.0, 2: 3.5, 3: 0.0}; at
    payout rate 0.8 the race's flat rho is (3.5 / 3) / 0.8 = 1.4583. Then 12 varied races in one
    call, each rho worked from its own dividend, field and payout rate."""
    race = race_id(0)
    win = board(race, "WIN", {"1": 2.0, "2": 3.5, "3": 6.0, "4": None})  # 4 scratched
    dividends = paid(
        [(race, "WIN", "2", 350), (race, "PLACE", "2", 120), (race, "QUINELLA", "1,2", 480)]
    )
    out = ps.combination_returns(win, dividends, pool="WIN")
    assert OUT_COLUMNS <= set(out.columns)
    assert returns_of(out) == pytest.approx({"1": 0.0, "2": 3.5, "3": 0.0}, abs=1e-12)

    rr = ps.race_returns(out, payout_rate=0.8, band_edges=EDGES)
    total = totals(rr).loc[race]
    assert total["n_staked"] == 3
    assert total["n_refunded"] == 0
    assert total["ret_sum"] == pytest.approx(3.5)
    assert total["rate_sum"] == pytest.approx(2.4)
    assert total["ret_sum"] / total["rate_sum"] == pytest.approx((3.5 / 3) / 0.8)
    assert (3.5 / 3) / 0.8 == pytest.approx(1.4583, abs=1e-4)
    # Each horse sits in its own band of EDGES = (2.5, 5.0).
    assert by_band(rr) == approx_bands(
        {
            ps.odds_band(2.0, EDGES): (1, 0.0, 0.8, 0),
            ps.odds_band(3.5, EDGES): (1, 3.5, 0.8, 0),
            ps.odds_band(6.0, EDGES): (1, 0.0, 0.8, 0),
        }
    )

    # Varied races: the winner, the dividend (not the board price), the field and the rate move.
    rng = np.random.default_rng(4294)
    boards, rows, want = [], [], {}
    for i in range(1, 13):
        race = race_id(i)
        n = int(rng.integers(5, 17))
        odds = {str(h): float(rng.integers(12, 3000)) / 10 for h in range(1, n + 1)}
        winner = str(int(rng.integers(1, n + 1)))
        dividend = float(rng.integers(11, 4000) * 10)
        boards.append(board(race, "WIN", odds))
        rows.append((race, "WIN", winner, dividend))
        want[race] = (n, dividend / 100)
    for rate in (0.8, 0.725):
        out = ps.combination_returns(pd.concat(boards, ignore_index=True), paid(rows), pool="WIN")
        got = totals(ps.race_returns(out, payout_rate=rate, band_edges=EDGES))
        assert set(got.index) == set(want)
        for race, (n, ret) in want.items():
            assert got.loc[race, "n_staked"] == n, race
            assert got.loc[race, "ret_sum"] == pytest.approx(ret), race
            assert got.loc[race, "rate_sum"] == pytest.approx(rate * n), race
            rho = got.loc[race, "ret_sum"] / got.loc[race, "rate_sum"]
            assert rho == pytest.approx(ret / n / rate), race


def test_place_and_wide_pay_every_placed_combination():
    """An 8-runner race with Place dividends for 3 horses and Wide dividends for their 3 pairs:
    exactly 3 and 3 non-zero returns, each dividend/100. The odds and the odds band of a Place or
    Wide combination are the low end of its final range (`odds`), not `odds_hi`."""
    race = race_id(3)
    place_low = {1: 1.2, 2: 1.8, 3: 2.4, 4: 3.1, 5: 4.6, 6: 7.9, 7: 9.5, 8: 15.2}
    place_high = {1: 1.5, 2: 2.6, 3: 3.3, 4: 5.4, 5: 7.0, 6: 12.8, 7: 14.1, 8: 24.0}
    place_paid = {"2": 210, "5": 520, "7": 1210}
    rng = np.random.default_rng(4295)
    wide_low, wide_high = {}, {}
    for c in combinations_of("QUINELLA_PLACE", range(1, 9)):
        low = float(rng.integers(15, 900)) / 10
        wide_low[c], wide_high[c] = low, round(low * float(rng.uniform(1.3, 2.2)), 1)
    wide_paid = {"2,5": 730, "2,7": 1480, "5,7": 2260}
    for c, d in wide_paid.items():  # each paid dividend inside its final range
        wide_low[c], wide_high[c] = round(d / 100 * 0.7, 1), round(d / 100 * 1.4, 1)
    boards = {
        "PLACE": board(
            race,
            "PLACE",
            {str(h): o for h, o in place_low.items()},
            {str(h): o for h, o in place_high.items()},
        ),
        "QUINELLA_PLACE": board(race, "QUINELLA_PLACE", wide_low, wide_high),
    }
    lows = {"PLACE": {str(h): o for h, o in place_low.items()}, "QUINELLA_PLACE": wide_low}
    highs = {"PLACE": {str(h): o for h, o in place_high.items()}, "QUINELLA_PLACE": wide_high}
    pays = {"PLACE": place_paid, "QUINELLA_PLACE": wide_paid}
    dividends = paid(
        [(race, "PLACE", c, d) for c, d in place_paid.items()]
        + [(race, "QUINELLA_PLACE", c, d) for c, d in wide_paid.items()]
        + [(race, "WIN", "2", 330), (race, "QUINELLA", "2,5", 1890), (race, "TRIO", "2,5,7", 9930)]
    )
    edges = (2.0, 5.0, 10.0)

    for pool in RANGE_POOLS:
        out = ps.combination_returns(boards[pool], dividends, pool=pool)
        returns = returns_of(out)
        assert set(returns) == set(lows[pool]), pool
        want = {c: pays[pool].get(c, 0) / 100 for c in lows[pool]}
        assert returns == pytest.approx(want, abs=1e-12), pool
        assert sum(r > 0 for r in returns.values()) == 3, pool
        assert dict(zip(out["combination"], out["odds"].astype(float), strict=True)) == (
            pytest.approx(lows[pool])
        ), pool

        # Premise: banding by the high end would put some combinations in another band.
        def interval(o: float) -> int:
            return bisect.bisect_right(edges, o)

        assert any(interval(lows[pool][c]) != interval(highs[pool][c]) for c in lows[pool])

        rate = RATES[pool]
        expected: dict[str, list[float]] = {}
        for c, low in lows[pool].items():
            band = ps.odds_band(low, edges)
            n, ret, rate_sum, refunded = expected.get(band, [0, 0.0, 0.0, 0])
            expected[band] = [n + 1, ret + want[c], rate_sum + rate, refunded]
        rr = ps.race_returns(out, payout_rate=rate, band_edges=edges)
        assert by_band(rr) == approx_bands({b: tuple(v) for b, v in expected.items()}), pool


def test_exacta_returns_follow_the_finishing_order():
    """FORECAST '4,7' paid, so '7,4' returns 0 (and in a second race the reverse); TRIO
    combinations are matched as stored, ascending by horse number ('2,5,10', not '10,2,5'), and a
    TIERCE row of the same horses does not pay the Trio."""
    first, second = race_id(5), race_id(6)
    horses = range(1, 11)
    rng = np.random.default_rng(4296)

    def priced(pool: str) -> dict[str, float]:
        return {c: float(rng.integers(20, 30000)) / 10 for c in combinations_of(pool, horses)}

    forecast = pd.concat(
        [
            board(first, "FORECAST", priced("FORECAST")),
            board(second, "FORECAST", priced("FORECAST")),
        ],
        ignore_index=True,
    )
    trio = pd.concat(
        [board(first, "TRIO", priced("TRIO")), board(second, "TRIO", priced("TRIO"))],
        ignore_index=True,
    )
    dividends = paid(
        [
            (first, "FORECAST", "4,7", 2350),
            (first, "QUINELLA", "4,7", 1100),
            (first, "TRIO", "2,4,7", 6120),
            (first, "TIERCE", "4,7,2", 31870),
            (second, "FORECAST", "7,4", 4410),
            (second, "QUINELLA", "4,7", 1990),
            (second, "TRIO", "2,5,10", 15600),
            (second, "TIERCE", "5,10,2", 88760),
        ]
    )

    out = ps.combination_returns(forecast, dividends, pool="FORECAST")
    one, two = returns_of(out, first), returns_of(out, second)
    assert len(one) == len(two) == 90
    assert one["4,7"] == pytest.approx(23.5) and one["7,4"] == 0.0
    assert two["7,4"] == pytest.approx(44.1) and two["4,7"] == 0.0
    assert sum(r > 0 for r in one.values()) == sum(r > 0 for r in two.values()) == 1

    out = ps.combination_returns(trio, dividends, pool="TRIO")
    one, two = returns_of(out, first), returns_of(out, second)
    assert len(one) == len(two) == 120
    assert one["2,4,7"] == pytest.approx(61.2)
    assert two["2,5,10"] == pytest.approx(156.0)
    assert sum(r > 0 for r in one.values()) == sum(r > 0 for r in two.values()) == 1


def test_refunds_are_dropped_and_counted():
    """Horse 5 is withdrawn after the final board: its Win combination and the five Quinella
    pairs with it have `is_refund` rows. They are flagged refunded, counted in n_refunded and
    left out of n_staked, ret_sum and rate_sum (the stake that comes back is not a return)."""
    race = race_id(7)
    rng = np.random.default_rng(4297)
    horses = range(1, 7)
    win = board(
        race, "WIN", {c: float(rng.integers(15, 400)) / 10 for c in combinations_of("WIN", horses)}
    )
    quinella = board(
        race,
        "QUINELLA",
        {c: float(rng.integers(30, 2000)) / 10 for c in combinations_of("QUINELLA", horses)},
    )
    refunded_pairs = ["1,5", "2,5", "3,5", "4,5", "5,6"]
    dividends = paid(
        [(race, "WIN", "3", 520), (race, "WIN", "5", 100, True), (race, "QUINELLA", "2,3", 1230)]
        + [(race, "QUINELLA", c, 100, True) for c in refunded_pairs]
        + [(race, "PLACE", "5", 100, True)]
    )
    cases = {
        # pool: (board, refunded combinations, paying combination, its return, combinations)
        "WIN": (win, {"5"}, "3", 5.2, 6),
        "QUINELLA": (quinella, set(refunded_pairs), "2,3", 12.3, 15),
    }
    for pool, (frame, refunded, payer, ret, n) in cases.items():
        out = ps.combination_returns(frame, dividends, pool=pool)
        assert OUT_COLUMNS <= set(out.columns)
        assert len(out) == n, pool
        flags = dict(zip(out["combination"], out["refunded"].astype(bool), strict=True))
        assert {c for c, f in flags.items() if f} == refunded, pool
        live = out[~out["refunded"].astype(bool)]
        assert returns_of(live) == pytest.approx(
            {c: (ret if c == payer else 0.0) for c in live["combination"]}, abs=1e-12
        ), pool

        rate = RATES[pool]
        total = totals(ps.race_returns(out, payout_rate=rate, band_edges=EDGES)).loc[race]
        assert total["n_refunded"] == len(refunded), pool
        assert total["n_staked"] == n - len(refunded), pool
        assert total["ret_sum"] == pytest.approx(ret), pool
        assert total["rate_sum"] == pytest.approx(rate * (n - len(refunded))), pool


DEAD_HEATS = {
    "WIN": {"3": 240, "6": 410},  # dead heat for first
    "PLACE": {"1": 130, "2": 180, "3": 300, "6": 270},  # dead heat for third
    "QUINELLA": {"1,3": 950, "1,6": 1720},  # dead heat for second
    "QUINELLA_PLACE": {"1,2": 330, "1,3": 520, "2,3": 690, "1,6": 610, "2,6": 880},
    "FORECAST": {"3,6": 2870, "6,3": 3410},  # dead heat for first
    "TRIO": {"1,2,3": 4560, "1,2,6": 5290},  # dead heat for third
}


def test_a_dead_heat_pays_each_paying_combination():
    """A dead heat writes several paying rows in a pool (two winners, four placed horses, both
    exacta orders...): each paying combination returns its own dividend/100, the rest 0, and the
    race's ret_sum is their sum."""
    race = race_id(8)
    horses = range(1, 10)
    rng = np.random.default_rng(4298)
    dividends = paid(
        [(race, pool, c, d) for pool, pays in DEAD_HEATS.items() for c, d in pays.items()]
    )
    for pool, pays in DEAD_HEATS.items():
        odds = {c: float(rng.integers(11, 20000)) / 10 for c in combinations_of(pool, horses)}
        hi = {c: round(o * 1.5, 1) for c, o in odds.items()} if pool in RANGE_POOLS else None
        out = ps.combination_returns(board(race, pool, odds, hi), dividends, pool=pool)
        want = {c: pays.get(c, 0) / 100 for c in odds}
        assert returns_of(out) == pytest.approx(want, abs=1e-12), pool
        assert sum(r > 0 for r in returns_of(out).values()) == len(pays), pool

        total = totals(ps.race_returns(out, payout_rate=RATES[pool], band_edges=EDGES)).loc[race]
        assert total["n_staked"] == len(odds), pool
        assert total["ret_sum"] == pytest.approx(sum(pays.values()) / 100), pool


# --- a fairly priced pool: flat rho near 1 -------------------------------------------------------

FAIR_RACES = {8: 667, 9: 667, 10: 666}  # starters: races (2,000 a pool)
FAIR_EDGES = (2.0, 5.0, 10.0, 30.0, 100.0)
STRENGTH_SD = 0.35  # sd of the runners' log strengths (Plackett-Luce)


def fair_pools(rng: np.random.Generator, n: int, races: int):
    """`races` races of `n` starters, priced fairly: per pool the combinations (0-based legs),
    their true probabilities (races, combinations) and the paying combinations (races, k).

    Win probabilities s come from log strengths; the finishing order is a Plackett-Luce draw from
    s (Gumbel trick), so each pool's probabilities are Harville's, computed exactly here.
    """
    s = np.exp(rng.normal(0.0, STRENGTH_SD, (races, n)))
    s /= s.sum(axis=1, keepdims=True)
    order = np.argsort(-(np.log(s) + rng.gumbel(size=(races, n))), axis=1)
    top = order[:, :3]

    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    distinct = (i != j) & (j != k) & (i != k)
    si, sj, sk = s[:, :, None, None], s[:, None, :, None], s[:, None, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        trifecta = np.where(distinct, si * sj / (1 - si) * sk / (1 - si - sj), 0.0)
    exacta = trifecta.sum(axis=3)  # P(i first, j second)
    place = trifecta.sum(axis=(2, 3)) + trifecta.sum(axis=(1, 3)) + trifecta.sum(axis=(1, 2))
    pair_top3 = trifecta.sum(axis=3) + trifecta.sum(axis=2) + trifecta.sum(axis=1)

    horses = np.arange(n)
    singles = horses[:, None]
    pairs = np.array(list(itertools.combinations(horses, 2)))
    perms = np.array(list(itertools.permutations(horses, 2)))
    triples = np.array(list(itertools.combinations(horses, 3)))
    trio = sum(trifecta[:, p[0], p[1], p[2]] for p in itertools.permutations(triples.T))

    def lookup(legs: np.ndarray, paying_legs: np.ndarray) -> np.ndarray:
        index = np.full((n,) * legs.shape[1], -1)
        index[tuple(legs.T)] = np.arange(len(legs))
        return index[tuple(np.moveaxis(paying_legs, -1, 0))]

    wide_pairs = np.sort(top[:, [[0, 1], [0, 2], [1, 2]]], axis=2)
    return {
        "WIN": (singles, s, lookup(singles, top[:, :1, None])),
        "PLACE": (singles, place, lookup(singles, top[:, :3, None])),
        "QUINELLA": (
            pairs,
            exacta[:, pairs[:, 0], pairs[:, 1]] + exacta[:, pairs[:, 1], pairs[:, 0]],
            lookup(pairs, np.sort(top[:, None, :2], axis=2)),
        ),
        "QUINELLA_PLACE": (
            pairs,
            pair_top3[:, pairs[:, 0], pairs[:, 1]] + pair_top3[:, pairs[:, 1], pairs[:, 0]],
            lookup(pairs, wide_pairs),
        ),
        "FORECAST": (perms, exacta[:, perms[:, 0], perms[:, 1]], lookup(perms, top[:, None, :2])),
        "TRIO": (triples, trio, lookup(triples, np.sort(top[:, None, :3], axis=2))),
    }


def test_a_fairly_priced_pool_returns_rho_near_one():
    """DoD: 2,000 seeded synthetic races per pool, priced at the payout rate over the true
    probability (final odds on the 0.1 grid, each paying combination paid its board price),
    outcomes drawn from those probabilities: the pool's flat rho is within 0.05 of 1, and equals
    the rho worked here from the boards and dividends."""
    rng = np.random.default_rng(20261127)
    boards = {p: [] for p in POOLS}
    paid_parts = []
    staked = dict.fromkeys(POOLS, 0)
    returned = dict.fromkeys(POOLS, 0.0)
    first = 0
    for n, races in FAIR_RACES.items():
        ids = np.array([race_id(first + r) for r in range(races)], dtype=object)
        first += races
        for pool, (legs, prob, pays) in fair_pools(rng, n, races).items():
            # Premise: the probabilities are a distribution (Place and Wide: three places).
            places = 3.0 if pool in RANGE_POOLS else 1.0
            assert np.allclose(prob.sum(axis=1), places), pool
            odds = np.maximum(np.round(RATES[pool] / prob, 1), 1.0)
            text = np.array([joined(c + 1) for c in legs], dtype=object)
            n_comb = len(legs)
            frame = pd.DataFrame(
                {
                    "race_id": np.repeat(ids, n_comb),
                    "pool": pool,
                    "combination": np.tile(text, races),
                    "odds": odds.ravel(),
                    "odds_hi": np.round(odds.ravel() * 1.5, 1) if pool in RANGE_POOLS else np.nan,
                    "investment": np.nan,
                    "status_raw": None,
                }
            )
            pool_rows = pd.DataFrame(
                {
                    "race_id": ids,
                    "pool": pool,
                    "combination": POOL_ROW,
                    "odds": np.nan,
                    "odds_hi": np.nan,
                    "investment": 1e7,
                    "status_raw": None,
                }
            )
            boards[pool] += [frame, pool_rows]
            dividend = np.round(np.take_along_axis(odds, pays, axis=1) * 100)
            paid_parts.append(
                pd.DataFrame(
                    {
                        "race_id": np.repeat(ids, pays.shape[1]),
                        "pool": pool,
                        "combination": text[pays.ravel()],
                        "dividend": dividend.ravel(),
                        "is_refund": False,
                    }
                )
            )
            staked[pool] += races * n_comb
            returned[pool] += float(dividend.sum()) / 100
    dividends = pd.concat(paid_parts, ignore_index=True)

    for pool in POOLS:
        expected_rho = returned[pool] / (RATES[pool] * staked[pool])
        assert abs(expected_rho - 1) < 0.04, (pool, expected_rho)  # premise: the draw is typical

        board_rows = pd.concat(boards[pool], ignore_index=True)
        out = ps.combination_returns(board_rows, dividends, pool=pool)
        assert len(out) == staked[pool], pool
        rr = ps.race_returns(out, payout_rate=RATES[pool], band_edges=FAIR_EDGES)
        assert rr["race_id"].nunique() == sum(FAIR_RACES.values()), pool
        assert rr["n_staked"].sum() == staked[pool], pool
        assert rr["n_refunded"].sum() == 0, pool
        rho = rr["ret_sum"].sum() / rr["rate_sum"].sum()
        assert rho == pytest.approx(expected_rho, rel=1e-9), pool
        assert abs(rho - 1) < 0.05, (pool, rho)


# --- grouped bets for #78 ------------------------------------------------------------------------


def test_grouped_bets_count_each_unrefunded_combination_once():
    """`grouped_bets`: one row per (segment levels, odds band, odds) with takeout 1 - rate, the
    combinations' count and their wins; refunded combinations are left out. Expanding the rows
    back into `count` bets with `won` wins gives the input's unrefunded bets exactly."""
    rng = np.random.default_rng(4299)
    n = 600
    rows = pd.DataFrame(
        {
            "race_id": [race_id(int(r)) for r in rng.integers(0, 60, n)],
            "combination": [str(int(h)) for h in rng.integers(1, 17, n)],
            "odds": rng.choice([1.4, 2.0, 2.0, 3.5, 4.9, 7.0, 12.0, 45.5, 120.3], n),
            "won": rng.random(n) < 0.2,
            "refunded": rng.random(n) < 0.05,
            "class": rng.choice(["maiden", "conditions", "open"], n),
            "field": rng.choice(["5-10", "11-14"], n),
        }
    )
    rows["ret"] = np.where(rows["won"], rows["odds"], 0.0)
    edges = (2.0, 5.0, 10.0)
    by = ["class", "field"]

    grouped = ps.grouped_bets(rows, by=by, payout_rate=0.775, band_edges=edges)

    assert {*by, "band", "odds", "takeout", "count", "won"} <= set(grouped.columns)
    assert not grouped.duplicated([*by, "odds"]).any()
    assert np.allclose(grouped["takeout"].astype(float), 0.225)
    for odds, band in zip(grouped["odds"], grouped["band"], strict=True):
        assert band == ps.odds_band(float(odds), edges)
    assert ((grouped["won"] >= 0) & (grouped["won"] <= grouped["count"])).all()

    live = rows[~rows["refunded"]]
    want = live.groupby([*by, "odds"]).agg(count=("won", "size"), won=("won", "sum"))
    got = grouped.set_index([*by, "odds"])[["count", "won"]].astype(int).sort_index()
    assert got.to_dict("index") == want.astype(int).sort_index().to_dict("index")
    assert int(grouped["count"].sum()) == len(live) < len(rows)


# --- nothing reads the store, the registry or the holdout ----------------------------------------

FORBIDDEN_MODULES = (
    "horsetrack.core.store",
    "horsetrack.core.eval.registry",
    "horsetrack.core.eval.holdout",
    "horsetrack.core.paths",
    "horsetrack.countries.japan.models.v0.store",
    "pyarrow.parquet",
    "duckdb",
)
FORBIDDEN_NAMES = {
    "read_store_table",
    "read_asof",
    "read_table",
    "read_parquet",
    "read_csv",
    "ParquetFile",
    "connect",
    "default_registry",
    "Registry",
    "LockedHoldout",
    "is_holdout_day",
    "open",
}


def test_nothing_reads_the_store_the_registry_or_the_holdout():
    """DoD: the scan's maths and the grouped fit take frames; neither module imports the store,
    the registry, the holdout or the data paths, nor calls a reader."""
    for module in (ps, segment_roi):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported, names = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                imported.add(base)
                imported |= {f"{base}.{alias.name}" for alias in node.names}
                names |= {alias.name for alias in node.names}
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
        bad = sorted(
            m for m in imported for f in FORBIDDEN_MODULES if m == f or m.startswith(f + ".")
        )
        assert not bad, (module.__name__, bad)
        assert not names & FORBIDDEN_NAMES, (module.__name__, sorted(names & FORBIDDEN_NAMES))
