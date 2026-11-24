"""Shadow values E[p * O_final] over nowcast draws (ticket #118). No stake, no bet: a value is the
gross return per 1 yen at the final price (1.0 = break-even).

Contract pinned here (`horsetrack.countries.japan.models.v0.shadow_values`):
- Pools are the config keys "WIN", "PLACE" and "QUINELLA_PLACE" (Wide). Combinations are strings:
  "7" for WIN and PLACE, "a,b" with a < b for Wide (the odds-log format).
- `shocks` is a Mapping[int, np.ndarray]: horse number -> shock column of length n_draws, one entry
  per runner on the board (`board.win`). One such mapping is drawn per race from `seed` and the
  same values reach the nowcast for every pool.
- `PoolDraws(pool, combinations, odds, odds_hi)`: `odds` is (n_draws x len(combinations)), column
  j aligned to `combinations[j]`, in any order; `odds_hi` is the same shape with NaN where a
  combination has no high figure, or None (WIN). For PLACE and Wide `odds` is the low figure.
- Draw m: q = (1/O_WIN) / sum(1/O_WIN) over the board's runners; p = win.predict(q, context), a 1-D
  array aligned to q; P = ordering.place_probs(p, places) and the Wide pair matrix
  ordering.pair_probs(p, places_wide) (n x n, symmetric) are aligned to the same runner order.
- WIN: value_mean = mean_m p_i O_i, value_q05/value_q95 the 5% / 95% points of p_i O_i over m.
  PLACE / Wide: value_at_low = mean_m P low, value_at_high = mean_m P high (None without a high
  figure). Places paid come from the jra config: PLACE 3 at 8+, 2 at 5-7, else 0 (no values);
  Wide 3 at 4+, else 0. Without an ordering model every PLACE runner and Wide pair still gets a
  row, with both values None.

The win, ordering and nowcast models are fakes defined here; expected values are computed here
from the draws the fake nowcast returned. All data is synthetic.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import dataclasses
from datetime import datetime, timedelta, timezone
from itertools import combinations as pairs_of

import numpy as np
import pytest

from horsetrack.countries.japan.board.snapshot import RaceBoard
from horsetrack.countries.japan.models.v0.shadow_values import (
    PoolDraws,
    ShadowModels,
    ShadowValue,
    shadow_values,
)

RACE = "jra:2026-11-28:TOK:11"
WIN, PLACE, WIDE = "WIN", "PLACE", "QUINELLA_PLACE"
AT = datetime(2026, 11, 28, 5, 51, tzinfo=timezone.utc)
IDS = {"win": "win-test", "ordering": "ordering-test", "nowcast": "nowcast-test"}
TOL = 1e-9


# ---------------------------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------------------------


def make_board(win, place=None, wide=None, scratched=()) -> RaceBoard:
    return RaceBoard(
        race_id=RACE,
        post_utc=AT + timedelta(minutes=10),
        at_utc=AT,
        capture_utc=AT,
        final=False,
        win=dict(win),
        place=dict(place or {}),
        wide=dict(wide or {}),
        totals={},
        scratched=frozenset(scratched),
    )


def full_board(horses, scratched=(), place_hi_missing=()) -> RaceBoard:
    """WIN, PLACE and every Wide pair priced for `horses`; varied, synthetic prices."""
    horses = sorted(horses)
    win = {h: round(1.8 + 1.7 * i + 0.3 * (h % 3), 1) for i, h in enumerate(horses)}
    place = {
        h: (round(1.1 + 0.3 * i, 1), None if h in place_hi_missing else round(1.5 + 0.55 * i, 1))
        for i, h in enumerate(horses)
    }
    wide = {
        (a, b): (round(1.3 + 0.4 * (a + b), 1), round(1.9 + 0.7 * (a + b), 1))
        for a, b in pairs_of(horses, 2)
    }
    return make_board(win, place, wide, scratched)


def pair_key(a: int, b: int) -> str:
    a, b = sorted((a, b))
    return f"{a},{b}"


def places_paid(starters: int) -> tuple[int, int]:
    """(PLACE, Wide) places from the JRA rules: Place 8+ -> 3, 5-7 -> 2, else 0; Wide 4+ -> 3."""
    place = 3 if starters >= 8 else 2 if starters >= 5 else 0
    return place, 3 if starters >= 4 else 0


def runners_of(combination: str) -> list[int]:
    return [int(part) for part in combination.split(",")]


# ---------------------------------------------------------------------------------------------
# Fake models
# ---------------------------------------------------------------------------------------------


class PowerWin:
    """p proportional to q**k (k = 1 is the identity p = q)."""

    def __init__(self, k: float = 1.0):
        self.k = k
        self.races = []

    def predict(self, q, race=None):
        q = np.asarray(q, dtype=float)
        if q.ndim != 1:
            raise AssertionError(f"win.predict takes one draw's 1-D q, got shape {q.shape}")
        self.races.append(race)
        w = q**self.k
        return w / w.sum()


def fake_place(p, places):
    return 1.0 - (1.0 - np.asarray(p, dtype=float)) ** places


def fake_pair(pa, pb, places):
    return places * np.asarray(pa, dtype=float) * np.asarray(pb, dtype=float)


class FakeOrdering:
    """Place = 1 - (1 - p)**places; pair = places * p_a * p_b (zero diagonal). Records places."""

    def __init__(self):
        self.place_calls = []
        self.pair_calls = []

    def place_probs(self, p, places):
        p = np.asarray(p, dtype=float)
        if p.ndim != 1:
            raise AssertionError(f"place_probs takes one draw's 1-D p, got shape {p.shape}")
        self.place_calls.append(places)
        return fake_place(p, places)

    def pair_probs(self, p, places):
        p = np.asarray(p, dtype=float)
        if p.ndim != 1:
            raise AssertionError(f"pair_probs takes one draw's 1-D p, got shape {p.shape}")
        self.pair_calls.append(places)
        matrix = fake_pair(p[:, None], p[None, :], places)
        np.fill_diagonal(matrix, 0.0)
        return matrix


def _n_draws(shocks) -> int:
    return len(np.asarray(next(iter(shocks.values()))))


def _stack(columns, n):
    return np.column_stack(columns) if columns else np.empty((n, 0))


def shock_draws(board: RaceBoard, pool: str, shocks, scale: float) -> PoolDraws:
    """Board prices times exp(scale * shock); a Wide pair takes the mean of its runners' shocks.
    Columns are in descending order so the result must be aligned by combination, not position."""
    n = _n_draws(shocks)
    s = {h: np.asarray(v, dtype=float) for h, v in shocks.items()}
    if pool == WIN:
        horses = sorted(board.win, reverse=True)
        odds = _stack([board.win[h] * np.exp(scale * s[h]) for h in horses], n)
        return PoolDraws(pool, tuple(str(h) for h in horses), odds, None)
    if pool == PLACE:
        horses = sorted(board.place, reverse=True)
        lo = [board.place[h][0] * np.exp(scale * s[h]) for h in horses]
        hi = [
            np.full(n, np.nan)
            if board.place[h][1] is None
            else board.place[h][1] * np.exp(scale * s[h])
            for h in horses
        ]
        return PoolDraws(pool, tuple(str(h) for h in horses), _stack(lo, n), _stack(hi, n))
    if pool == WIDE:
        keys = sorted(board.wide, reverse=True)
        mix = {k: (s[k[0]] + s[k[1]]) / 2 for k in keys}
        lo = [board.wide[k][0] * np.exp(scale * mix[k]) for k in keys]
        hi = [
            np.full(n, np.nan)
            if board.wide[k][1] is None
            else board.wide[k][1] * np.exp(scale * mix[k])
            for k in keys
        ]
        return PoolDraws(pool, tuple(pair_key(*k) for k in keys), _stack(lo, n), _stack(hi, n))
    return PoolDraws(pool, (), np.empty((n, 0)), None)


class ShockNowcast:
    """A nowcast driven by the shocks it is given; records every call and what it returned."""

    def __init__(self, scale: float = 0.3):
        self.scale = scale
        self.calls = []
        self.returned = {}

    def draws(self, board, *, pool, tau_min, shocks):
        self.calls.append(
            {
                "pool": pool,
                "tau_min": tau_min,
                "shocks": {h: np.array(v, dtype=float, copy=True) for h, v in shocks.items()},
            }
        )
        out = shock_draws(board, pool, shocks, self.scale)
        self.returned[pool] = out
        return out


class BoardNowcast(ShockNowcast):
    """Every draw is the board itself."""

    def __init__(self):
        super().__init__(scale=0.0)


class ScriptedNowcast:
    """Returns fixed draws per pool; pools not scripted get no combinations."""

    def __init__(self, scripted: dict[str, PoolDraws]):
        self.scripted = scripted

    def draws(self, board, *, pool, tau_min, shocks):
        if pool in self.scripted:
            return self.scripted[pool]
        return PoolDraws(pool, (), np.empty((_n_draws(shocks), 0)), None)


def models(win=None, ordering="fake", nowcast=None) -> ShadowModels:
    return ShadowModels(
        win=win if win is not None else PowerWin(),
        ordering=FakeOrdering() if ordering == "fake" else ordering,
        nowcast=nowcast if nowcast is not None else ShockNowcast(),
        ids=dict(IDS),
    )


def by_key(values) -> dict[tuple[str, str], ShadowValue]:
    out = {}
    for v in values:
        key = (v.pool, v.combination)
        assert key not in out, f"{key} appears twice"
        out[key] = v
    return out


def odds_from_q(q, overround: float = 1.25):
    """WIN odds whose normalised inverse is exactly q."""
    return 1.0 / (np.asarray(q, dtype=float) * overround)


# ---------------------------------------------------------------------------------------------
# Independent reference: the draw formula applied to the draws the fake nowcast returned
# ---------------------------------------------------------------------------------------------


def column(draws: PoolDraws, combination: str, hi: bool = False):
    j = list(draws.combinations).index(combination)
    arr = draws.odds_hi if hi else draws.odds
    return None if arr is None else np.asarray(arr, dtype=float)[:, j]


def reference(board, draws, k, place_places, wide_places):
    horses = sorted(board.win)
    odds = np.column_stack([column(draws[WIN], str(h)) for h in horses])
    q = 1.0 / odds
    q = q / q.sum(axis=1, keepdims=True)
    p = q**k
    p = p / p.sum(axis=1, keepdims=True)
    ph = {h: p[:, j] for j, h in enumerate(horses)}
    oh = {h: odds[:, j] for j, h in enumerate(horses)}
    out = {}
    for h in horses:
        out[(WIN, str(h))] = {"x": ph[h] * oh[h], "plugin": ph[h].mean() * oh[h].mean()}

    def bracket(prob, lo, hi):
        high = None if hi is None or np.all(np.isnan(hi)) else float(np.mean(prob * hi))
        return {"low": float(np.mean(prob * lo)), "high": high}

    if place_places:
        for h in board.place:
            c = str(h)
            prob = fake_place(ph[h], place_places)
            out[(PLACE, c)] = bracket(
                prob, column(draws[PLACE], c), column(draws[PLACE], c, hi=True)
            )
    if wide_places:
        for a, b in board.wide:
            c = pair_key(a, b)
            prob = fake_pair(ph[a], ph[b], wide_places)
            out[(WIDE, c)] = bracket(prob, column(draws[WIDE], c), column(draws[WIDE], c, hi=True))
    return out


def assert_quantile(value, x, level):
    lo = np.quantile(x, level, method="lower")
    hi = np.quantile(x, level, method="higher")
    assert lo - TOL <= value <= hi + TOL, f"q{level}: {value} not in [{lo}, {hi}]"


# ---------------------------------------------------------------------------------------------
# Broad behaviour
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("horses", "scratched", "k", "n_draws", "seed", "tau_min"),
    [
        (range(1, 11), (), 1.0, 300, 1, 5),
        ((1, 2, 4, 5, 6, 8, 9), (3, 7), 2.0, 250, 2, 10),
        (range(1, 6), (), 0.6, 200, 3, 2),
        (range(1, 17), (), 1.3, 120, 4, 30),
    ],
)
def test_values_follow_the_draw_formula_in_every_pool(horses, scratched, k, n_draws, seed, tau_min):
    board = full_board(horses, scratched, place_hi_missing=(2,))
    nowcast = ShockNowcast(scale=0.35)
    win = PowerWin(k)
    out = by_key(
        shadow_values(
            board,
            tau_min=tau_min,
            models=models(win=win, nowcast=nowcast),
            n_draws=n_draws,
            seed=seed,
        )
    )
    place_places, wide_places = places_paid(board.starters)
    ref = reference(board, nowcast.returned, k, place_places, wide_places)

    expected_keys = {(WIN, str(h)) for h in board.win}
    expected_keys |= {(PLACE, str(h)) for h in board.place}
    expected_keys |= {(WIDE, pair_key(a, b)) for a, b in board.wide}
    assert set(out) == expected_keys  # no selection: every runner and pair gets a row
    assert all(r is not None and r == board.context() for r in win.races)

    for (pool, c), row in out.items():
        assert row.race_id == RACE and row.pool == pool and row.combination == c
        assert row.tau_min == tau_min
        if pool == WIN:
            assert row.board_odds == board.win[int(c)] and row.board_odds_hi is None
            x = ref[(pool, c)]["x"]
            assert row.value_mean == pytest.approx(float(x.mean()), abs=TOL)
            assert_quantile(row.value_q05, x, 0.05)
            assert_quantile(row.value_q95, x, 0.95)
            continue
        figures = board.place[int(c)] if pool == PLACE else board.wide[tuple(runners_of(c))]
        assert (row.board_odds, row.board_odds_hi) == figures
        assert row.value_at_low == pytest.approx(ref[(pool, c)]["low"], abs=TOL)
        if ref[(pool, c)]["high"] is None:
            assert row.value_at_high is None
        else:
            assert row.value_at_high == pytest.approx(ref[(pool, c)]["high"], abs=TOL)

    # The case discriminates the mean of the product from the plug-in mean(p) * mean(O).
    gaps = [abs(v["x"].mean() - v["plugin"]) for key, v in ref.items() if key[0] == WIN]
    assert max(gaps) > 1e-6


def test_an_identity_model_gives_the_payout_identity():
    board = make_board({1: 1.6, 2: 3.2, 3: 3.2})
    out = shadow_values(
        board, tau_min=5, models=models(nowcast=BoardNowcast()), n_draws=50, seed=11
    )
    win_rows = [v for v in out if v.pool == WIN]
    assert sorted(v.combination for v in win_rows) == ["1", "2", "3"]
    for v in win_rows:
        assert v.value_mean == pytest.approx(0.8, abs=TOL)
        assert v.value_q05 == pytest.approx(0.8, abs=TOL)
        assert v.value_q95 == pytest.approx(0.8, abs=TOL)
        assert v.board_odds == board.win[int(v.combination)]


@pytest.mark.parametrize(
    ("win", "expected"),
    [
        ({1: 2.0, 2: 2.0}, 1.0),  # no takeout: break-even
        ({1: 1.5, 2: 3.0, 3: 6.0, 4: 6.0}, 1.0 / (1 / 1.5 + 1 / 3.0 + 2 / 6.0)),
        ({4: 2.4, 7: 4.8, 9: 4.8, 12: 9.6, 13: 9.6}, 1.0 / (1 / 2.4 + 2 / 4.8 + 2 / 9.6)),
    ],
)
def test_the_identity_holds_for_other_boards(win, expected):
    out = shadow_values(
        make_board(win), tau_min=5, models=models(nowcast=BoardNowcast()), n_draws=7, seed=3
    )
    rows = by_key(out)
    for h in win:
        assert rows[(WIN, str(h))].value_mean == pytest.approx(expected, abs=TOL)


def test_value_is_the_mean_of_p_times_o_over_draws():
    # Runner 1: (p, O) = (0.2, 6.0) in draw 1 and (0.1, 12.0) in draw 2, with p = q.
    win = PoolDraws(
        WIN,
        ("3", "1", "2"),
        np.array([[3.0, 6.0, 3.0], [8.0 / 3.0, 12.0, 8.0 / 3.0]]),
        None,
    )
    board = make_board({1: 9.0, 2: 3.0, 3: 3.0})
    out = by_key(
        shadow_values(
            board,
            tau_min=5,
            models=models(nowcast=ScriptedNowcast({WIN: win})),
            n_draws=2,
            seed=0,
        )
    )
    assert out[(WIN, "1")].value_mean == pytest.approx(1.2, abs=TOL)
    assert out[(WIN, "1")].value_mean != pytest.approx(0.15 * 9.0, abs=1e-3)


def test_place_values_bracket_from_low_and_high():
    # Six starters: Place pays 2. With p = q and P = 1 - (1 - p)**2, runner 1 has P = (0.5, 0.3).
    p1 = np.array([1 - np.sqrt(0.5), 1 - np.sqrt(0.7)])
    q = np.column_stack([p1] + [(1 - p1) / 5] * 5)
    horses = (1, 2, 3, 4, 5, 6)
    board = make_board(
        {h: 2.0 + h for h in horses},
        {h: (1.2 + 0.1 * h, None if h == 2 else 1.6 + 0.2 * h) for h in horses},
    )
    win = PoolDraws(WIN, tuple(str(h) for h in horses), odds_from_q(q), None)
    low = np.array([[1.4, 1.1, 1.5, 2.0, 2.5, 3.0], [2.0, 1.3, 1.7, 2.2, 2.9, 3.4]])
    high = np.array([[1.8, np.nan, 1.9, 2.6, 3.1, 3.9], [2.6, np.nan, 2.1, 2.8, 3.5, 4.4]])
    place = PoolDraws(PLACE, tuple(str(h) for h in horses), low, high)
    out = by_key(
        shadow_values(
            board,
            tau_min=5,
            models=models(nowcast=ScriptedNowcast({WIN: win, PLACE: place})),
            n_draws=2,
            seed=0,
        )
    )
    assert out[(PLACE, "1")].value_at_low == pytest.approx(0.65, abs=TOL)
    assert out[(PLACE, "1")].value_at_high == pytest.approx(0.84, abs=TOL)
    # Runner 2 has no high figure: a low-end value only.
    p2 = q[:, 1]
    assert out[(PLACE, "2")].value_at_low == pytest.approx(
        float(np.mean(fake_place(p2, 2) * low[:, 1])), abs=TOL
    )
    assert out[(PLACE, "2")].value_at_high is None
    assert out[(PLACE, "2")].board_odds_hi is None


def test_wide_values_use_pair_probabilities():
    # Seven starters with gaps in the numbers: Place pays 2, Wide pays 3.
    horses = (1, 2, 5, 6, 8, 11, 14)
    board = full_board(horses, scratched=(3, 4))
    nowcast = ShockNowcast(scale=0.4)
    out = by_key(
        shadow_values(
            board,
            tau_min=10,
            models=models(win=PowerWin(1.5), nowcast=nowcast),
            n_draws=150,
            seed=21,
        )
    )
    ref = reference(board, nowcast.returned, 1.5, 2, 3)
    wrong_places = reference(board, nowcast.returned, 1.5, 2, 2)
    wide_keys = [key for key in ref if key[0] == WIDE]
    assert len(wide_keys) == 21
    for key in wide_keys:
        assert out[key].value_at_low == pytest.approx(ref[key]["low"], abs=TOL)
        assert out[key].value_at_high == pytest.approx(ref[key]["high"], abs=TOL)
    # The case tells the Wide places (3) from the Place places (2).
    assert max(abs(ref[k]["low"] - wrong_places[k]["low"]) for k in wide_keys) > 1e-3


# ---------------------------------------------------------------------------------------------
# Places paid, missing ordering, scratches
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("starters", [3, 4, 5, 7, 8, 12])
def test_places_paid_follow_the_field_size(starters):
    board = full_board(range(1, starters + 1))
    ordering = FakeOrdering()
    out = shadow_values(board, tau_min=5, models=models(ordering=ordering), n_draws=10, seed=5)
    place_places, wide_places = places_paid(starters)
    for pool, places, calls in (
        (PLACE, place_places, ordering.place_calls),
        (WIDE, wide_places, ordering.pair_calls),
    ):
        rows = [v for v in out if v.pool == pool]
        if places:
            assert calls and set(calls) == {places}
            expected = board.place if pool == PLACE else board.wide
            assert len(rows) == len(expected)
            assert all(v.value_at_low is not None for v in rows)
        else:
            assert not [c for c in calls if c > 0]
            assert all(v.value_at_low is None and v.value_at_high is None for v in rows)


def test_without_an_ordering_model_place_and_wide_are_null():
    board = full_board(range(1, 10))
    out = by_key(shadow_values(board, tau_min=5, models=models(ordering=None), n_draws=20, seed=2))
    for h in board.win:
        assert out[(WIN, str(h))].value_mean is not None
    for h, figures in board.place.items():
        row = out[(PLACE, str(h))]
        assert row.value_at_low is None and row.value_at_high is None
        assert (row.board_odds, row.board_odds_hi) == figures
    for (a, b), figures in board.wide.items():
        row = out[(WIDE, pair_key(a, b))]
        assert row.value_at_low is None and row.value_at_high is None
        assert (row.board_odds, row.board_odds_hi) == figures


def test_scratched_runners_get_no_value():
    horses = (1, 2, 3, 5, 6, 7, 8, 9)
    board = full_board(horses, scratched=(4, 10))
    nowcast = ShockNowcast()
    out = shadow_values(board, tau_min=5, models=models(nowcast=nowcast), n_draws=12, seed=4)
    assert all(not {4, 10} & set(runners_of(v.combination)) for v in out)
    assert sorted(int(v.combination) for v in out if v.pool == WIN) == list(horses)
    assert len([v for v in out if v.pool == WIDE]) == len(board.wide)
    for call in nowcast.calls:
        assert set(call["shocks"]) == set(horses)


# ---------------------------------------------------------------------------------------------
# Shocks and seeds
# ---------------------------------------------------------------------------------------------


def test_one_shock_matrix_is_shared_across_pools():
    horses = (1, 2, 3, 4, 6, 7, 8, 9, 10)
    board = full_board(horses, scratched=(5,))
    nowcast = ShockNowcast()
    n_draws = 37
    shadow_values(board, tau_min=15, models=models(nowcast=nowcast), n_draws=n_draws, seed=9)
    by_pool = {call["pool"]: call for call in nowcast.calls}
    assert {WIN, PLACE, WIDE} <= set(by_pool)
    assert all(call["tau_min"] == 15 for call in nowcast.calls)
    first = by_pool[WIN]["shocks"]
    assert set(first) == set(horses)
    matrix = np.column_stack([first[h] for h in horses])
    assert matrix.shape == (n_draws, len(horses))
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix.std(axis=0) > 0)  # shocks vary over draws
    assert len({tuple(col) for col in matrix.T}) == len(horses)  # and between runners
    for call in nowcast.calls:
        assert set(call["shocks"]) == set(horses)
        for h in horses:
            assert np.array_equal(call["shocks"][h], first[h])


def test_values_are_reproducible_with_a_seed():
    board = full_board(range(1, 11))

    def run(seed):
        nowcast = ShockNowcast()
        values = shadow_values(
            board,
            tau_min=5,
            models=models(win=PowerWin(1.4), nowcast=nowcast),
            n_draws=60,
            seed=seed,
        )
        return values, nowcast.calls[0]["shocks"]

    first, shocks_a = run(17)
    again, shocks_b = run(17)
    other, shocks_c = run(18)
    assert first == again
    assert all(np.array_equal(shocks_a[h], shocks_b[h]) for h in shocks_a)
    assert not all(np.array_equal(shocks_a[h], shocks_c[h]) for h in shocks_a)
    win_first = {v.combination: v.value_mean for v in first if v.pool == WIN}
    win_other = {v.combination: v.value_mean for v in other if v.pool == WIN}
    assert win_first.keys() == win_other.keys()
    assert any(abs(win_first[c] - win_other[c]) > 1e-9 for c in win_first)


# ---------------------------------------------------------------------------------------------
# Shape of a value
# ---------------------------------------------------------------------------------------------

FIELDS = {
    "race_id",
    "pool",
    "combination",
    "tau_min",
    "board_odds",
    "board_odds_hi",
    "value_mean",
    "value_q05",
    "value_q95",
    "value_at_low",
    "value_at_high",
}


def test_no_value_field_names_a_stake():
    if dataclasses.is_dataclass(ShadowValue):
        names = {f.name for f in dataclasses.fields(ShadowValue)}
    else:
        names = set(ShadowValue._fields)
    assert names == FIELDS
    for name in names:
        for word in ("stake", "bet", "kelly", "amount", "wager"):
            assert word not in name.lower(), f"{name} names a {word}"
