"""The v0 nowcast (ticket #116): a fitted drift of the final price by pool, odds band and horizon.

Contract pinned here (`horsetrack.countries.japan.models.v0.nowcast`):
- y = ln O_final - ln O_t on the low figure, for WIN, PLACE and QUINELLA_PLACE, at t = post - tau.
- band_of(odds, bands) is the number of cut points at or below the odds.
- Cells (pool, tau, band): mu = n * ybar / (n + prior_n), sigma^2 = (sum (y - ybar)^2 +
  prior_n * sigma0^2) / (n + prior_n), sigma0 the SD (ddof 0) of every y of that (pool, tau). An
  empty cell has mu = 0 and sigma = sigma0. `model.cell(pool, tau_min, band)` returns the DriftCell.
- Draws: ln O = ln O_t + mu + sigma * eps; WIN rescaled per draw to keep the board's sum of 1/O;
  PLACE uses the runner's shock, a Wide pair (a, b) (eps_a + eps_b) / sqrt(2); high figures move by
  the low figure's factor; nothing below 1.0; tau = 0 gives the board. Combinations are text: "7"
  for WIN and PLACE, "a,b" with a < b for Wide.
- pit = Phi((y - mu) / sigma), log_predictive_density = log N(y; mu, sigma), coverage = the share of
  examples inside the central `level` interval.
- training_examples refuses a holdout day (HoldoutDayError); stale boards are dropped and counted in
  `examples.attrs["dropped"]` ({tau_min: examples dropped}); the fit records its days in
  `model.days`.
- Split captures (#234): on an HKTrack day whose rounds are WIN+PLACE, QUINELLA 1 s on and Wide 2 s
  on, the final Wide price is Wide's first final capture at or after the anchor (`final_board` over
  the whole log): the anchor round's own Wide capture when it is final, else the next round's. The
  examples and a refit carry QUINELLA_PLACE at every grid tau.

Every expected value is derived here from the formulas above, never from the code under test. All
data is synthetic; HORSETRACK_DATA_DIR points at tmp_path.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import hashlib
import json
import math
from bisect import bisect_right
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from horsetrack.core import config, paths
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board.snapshot import (
    DecisionGrid,
    RaceBoard,
    final_board,
    load_decision_grid,
)
from horsetrack.countries.japan.models.v0 import nowcast
from horsetrack.countries.japan.models.v0.nowcast import (
    HoldoutDayError,
    NowcastV0,
    band_of,
    coverage,
    log_predictive_density,
    pit,
    training_examples,
)

UTC = timezone.utc
WIN, PLACE, WIDE = "WIN", "PLACE", "QUINELLA_PLACE"
BANDS = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
FINAL = "最終オッズ"
VENUE = "KYO"
DEV_DAY = date(2026, 11, 15)  # before the Japan holdout start: a development day
POST = datetime(2026, 11, 15, 3, 0, tzinfo=UTC)


# ---------------------------------------------------------------- independent references


def ref_band(odds: float, bands=BANDS) -> int:
    return bisect_right(list(bands), odds)


def ref_cells(examples: pd.DataFrame, bands, prior_n: float):
    """{(pool, tau, band): (n, mu, sigma)} and {(pool, tau): sigma0}, from the stated formulas."""
    y = np.log(examples["odds_final"].to_numpy(float)) - np.log(examples["odds_t"].to_numpy(float))
    pools = examples["pool"].to_numpy()
    taus = examples["tau_min"].to_numpy(int)
    bands_of = np.array([ref_band(o, bands) for o in examples["odds_t"].to_numpy(float)])
    sigma0 = {}
    for key in {(p, int(t)) for p, t in zip(pools, taus, strict=True)}:
        ys = y[(pools == key[0]) & (taus == key[1])]
        sigma0[key] = float(np.sqrt(np.mean((ys - ys.mean()) ** 2)))
    cells = {}
    for key in {(p, int(t), int(b)) for p, t, b in zip(pools, taus, bands_of, strict=True)}:
        ys = y[(pools == key[0]) & (taus == key[1]) & (bands_of == key[2])]
        n = len(ys)
        s0 = sigma0[key[:2]]
        mu = n * ys.mean() / (n + prior_n)
        sigma = math.sqrt((((ys - ys.mean()) ** 2).sum() + prior_n * s0**2) / (n + prior_n))
        cells[key] = (n, float(mu), sigma)
    return cells, sigma0


def ref_mu_sigma(ref, pool: str, tau: int, band: int) -> tuple[float, float]:
    cells, sigma0 = ref
    if (pool, tau, band) in cells:
        _, mu, sigma = cells[(pool, tau, band)]
        return mu, sigma
    return 0.0, sigma0[(pool, tau)]


# ---------------------------------------------------------------- synthetic examples


def frame(records) -> pd.DataFrame:
    """records: (pool, tau, odds_t, y, day). Columns as training_examples returns them."""
    rows = []
    for i, (pool, tau, odds_t, y, day) in enumerate(records):
        rows.append(
            {
                "race_id": f"jra:{day}:{VENUE}:{i % 12 + 1:02d}",
                "day": day,
                "pool": pool,
                "combination": "1,2" if pool == WIDE else str(i % 8 + 1),
                "tau_min": int(tau),
                "odds_t": float(odds_t),
                "odds_final": float(odds_t) * math.exp(y),
                "board_age_s": 10.0,
            }
        )
    return pd.DataFrame(rows)


def cell_records(rng, pool, tau, odds_t, mean, sd, n, days=("2026-11-21", "2026-11-22")):
    ys = mean + sd * rng.standard_normal(n)
    return [(pool, tau, odds_t, float(y), days[i % len(days)]) for i, y in enumerate(ys)]


DRAW_GRID = DecisionGrid(minutes=(5, 2), primary=2)
# (pool, odds_t, mean, sd) per cell at tau 2; tau 5 doubles mean and sd. Bands differ in mu, sigma
# so a wrong band, pool or horizon lookup gives a different draw.
DRAW_CELLS = [
    (WIN, 1.7, 0.02, 0.05),
    (WIN, 3.5, -0.03, 0.08),
    (WIN, 7.0, 0.05, 0.12),
    (PLACE, 1.4, -0.04, 0.06),
    (PLACE, 3.0, 0.08, 0.10),
    (PLACE, 12.0, -0.10, 0.20),
    (WIDE, 4.0, 0.05, 0.10),
    (WIDE, 15.0, -0.06, 0.15),
    (WIDE, 25.0, 0.12, 0.25),
]
PRIOR_N = 10.0


def draw_examples() -> pd.DataFrame:
    rng = np.random.default_rng(116)
    records = []
    for tau, scale in ((2, 1.0), (5, 2.0)):
        for i, (pool, odds_t, mean, sd) in enumerate(DRAW_CELLS):
            records += cell_records(rng, pool, tau, odds_t, mean * scale, sd * scale, 20 + 5 * i)
    return frame(records)


@pytest.fixture(scope="module")
def drift():
    """(model, reference) fitted on draw_examples."""
    examples = draw_examples()
    model = NowcastV0.fit(examples, bands=BANDS, grid=DRAW_GRID, prior_n=PRIOR_N)
    return model, ref_cells(examples, BANDS, PRIOR_N)


def make_board(win=None, place=None, wide=None) -> RaceBoard:
    return RaceBoard(
        race_id=f"jra:{DEV_DAY.isoformat()}:{VENUE}:01",
        post_utc=POST,
        at_utc=POST - timedelta(minutes=2),
        capture_utc=POST - timedelta(minutes=2, seconds=20),
        final=False,
        win=dict(win or {}),
        place=dict(place or {}),
        wide=dict(wide or {}),
        totals={},
        scratched=frozenset(),
    )


def column(draws, combination: str) -> int:
    return draws.combinations.index(combination)


# ---------------------------------------------------------------- band_of


def test_band_of_uses_the_cut_points():
    shipped = config.load_country_yaml("japan", "nowcast_v0")
    assert [float(b) for b in shipped["bands"]] == list(BANDS)
    named = {1.5: 0, 2.0: 1, 7.0: 2, 150.0: 6}
    table = {
        1.0: 0,
        1.99: 0,
        4.99: 1,
        5.0: 2,
        9.9: 2,
        10.0: 3,
        19.9: 3,
        20.0: 4,
        49.9: 4,
        50.0: 5,
        99.9: 5,
        100.0: 6,
        9999.9: 6,
    }
    for odds, band in {**named, **table}.items():
        assert band_of(odds, shipped["bands"]) == band, odds
    other = [3.0, 30.0]
    for odds, band in {1.2: 0, 2.9: 0, 3.0: 1, 29.0: 1, 30.0: 2, 400.0: 2}.items():
        assert band_of(odds, other) == band, odds


# ---------------------------------------------------------------- cells


@pytest.mark.parametrize(
    ("n", "prior_n", "ybar"),
    [(10, 10.0, 0.2), (30, 10.0, -0.4), (6, 2.5, 0.17)],
)
def test_a_cell_with_prior_n_examples_is_shrunk_halfway(n, prior_n, ybar):
    """Two PLACE tau-2 cells (bands 1 and 3) with the same mean and different spreads, beside WIN
    and tau-5 data of other spreads that must not enter this (pool, tau)'s sigma0."""
    base = np.linspace(-1.0, 1.0, n)
    base -= base.mean()
    records = [(PLACE, 2, 3.0, ybar + 0.3 * d, "2026-11-21") for d in base]
    records += [(PLACE, 2, 12.0, ybar + 0.9 * d, "2026-11-21") for d in base]
    records += [(WIN, 2, 3.0, 2.0 * d, "2026-11-21") for d in base]
    records += [(PLACE, 5, 3.0, -1.0 + 3.0 * d, "2026-11-21") for d in base]
    examples = frame(records)
    grid = DecisionGrid(minutes=(5, 2), primary=2)
    model = NowcastV0.fit(examples, bands=BANDS, grid=grid, prior_n=prior_n)
    ref = ref_cells(examples, BANDS, prior_n)

    expected_mu = n * ybar / (n + prior_n)
    if (n, prior_n) == (10, 10.0):
        assert expected_mu == pytest.approx(0.1)
    for band in (1, 3):
        cell = model.cell(PLACE, 2, band)
        _, mu, sigma = ref[0][(PLACE, 2, band)]
        assert (cell.pool, cell.tau_min, cell.band, cell.n) == (PLACE, 2, band, n)
        assert cell.mu == pytest.approx(expected_mu, abs=1e-12)
        assert cell.mu == pytest.approx(mu, abs=1e-12)
        assert cell.sigma == pytest.approx(sigma, rel=1e-9)
    # the wider cell keeps more of its own spread
    assert model.cell(PLACE, 2, 3).sigma > model.cell(PLACE, 2, 1).sigma
    _, mu5, sigma5 = ref[0][(PLACE, 5, 1)]
    assert model.cell(PLACE, 5, 1).mu == pytest.approx(mu5, abs=1e-12)
    assert model.cell(PLACE, 5, 1).sigma == pytest.approx(sigma5, rel=1e-9)


def test_an_empty_cell_falls_back_to_the_pooled_prior(drift):
    model, ref = drift
    _, sigma0 = ref
    for pool, tau, band in [(PLACE, 2, 5), (PLACE, 5, 6), (WIN, 2, 4), (WIDE, 5, 0)]:
        cell = model.cell(pool, tau, band)
        assert cell.n == 0, (pool, tau, band)
        assert cell.mu == 0.0
        assert cell.sigma == pytest.approx(sigma0[(pool, tau)], rel=1e-9), (pool, tau, band)
    # the fallback reaches the draws: a PLACE runner at 60.0 (band 5, empty) at tau 2
    eps = np.array([-1.0, 0.0, 0.5, 2.0])
    draws = model.draws(make_board(place={4: (60.0, 75.0)}), pool=PLACE, tau_min=2, shocks={4: eps})
    expected = 60.0 * np.exp(sigma0[(PLACE, 2)] * eps)
    np.testing.assert_allclose(draws.odds[:, column(draws, "4")], expected, rtol=1e-9)


@pytest.mark.parametrize("pools", [(PLACE, WIDE), (WIDE, PLACE)])
def test_spread_follows_the_horizon_in_the_training_data(pools):
    """The first pool has SD 0.3 at tau 60 and 0.05 at tau 2; the second the reverse order."""
    rng = np.random.default_rng(7)
    wide_first, narrow_first = pools
    records = cell_records(rng, wide_first, 60, 3.0, 0.0, 0.3, 500)
    records += cell_records(rng, wide_first, 2, 3.0, 0.0, 0.05, 500)
    records += cell_records(rng, narrow_first, 60, 3.0, 0.0, 0.04, 500)
    records += cell_records(rng, narrow_first, 2, 3.0, 0.0, 0.2, 500)
    grid = DecisionGrid(minutes=(60, 2), primary=2)
    model = NowcastV0.fit(frame(records), bands=BANDS, grid=grid, prior_n=PRIOR_N)
    far, near = model.cell(wide_first, 60, 1).sigma, model.cell(wide_first, 2, 1).sigma
    assert far > near
    assert far == pytest.approx(0.3, rel=0.1)
    assert near == pytest.approx(0.05, rel=0.1)
    far, near = model.cell(narrow_first, 60, 1).sigma, model.cell(narrow_first, 2, 1).sigma
    assert far < near
    assert far == pytest.approx(0.04, rel=0.1)
    assert near == pytest.approx(0.2, rel=0.1)


# ---------------------------------------------------------------- draws


@pytest.mark.parametrize("tau", [2, 5])
def test_win_draws_keep_the_board_payout_identity(drift, tau):
    model, ref = drift
    board_win = {1: 1.6, 2: 3.2, 3: 3.2}
    total = sum(1.0 / o for o in board_win.values())
    assert total == pytest.approx(1.25)
    rng = np.random.default_rng(1250 + tau)
    shocks = {h: rng.standard_normal(400) for h in board_win}
    draws = model.draws(make_board(win=board_win), pool=WIN, tau_min=tau, shocks=shocks)

    assert draws.pool == WIN
    assert draws.odds_hi is None
    assert draws.odds.shape == (400, 3)
    assert sorted(draws.combinations) == ["1", "2", "3"]
    np.testing.assert_allclose((1.0 / draws.odds).sum(axis=1), 1.25, rtol=0, atol=1e-9)

    raw = {}
    for h, o in board_win.items():
        mu, sigma = ref_mu_sigma(ref, WIN, tau, ref_band(o))
        raw[h] = o * np.exp(mu + sigma * shocks[h])
    scale = sum(1.0 / r for r in raw.values()) / 1.25
    for h in board_win:
        expected = raw[h] * scale
        assert expected.min() > 1.0  # the floor does not bind here
        np.testing.assert_allclose(draws.odds[:, column(draws, str(h))], expected, rtol=1e-9)
    assert np.log(draws.odds[:, column(draws, "1")]).std() > 0.01


@pytest.mark.parametrize("tau", [2, 5])
def test_a_place_draw_shifts_by_mu_plus_sigma_times_the_shock(drift, tau):
    model, ref = drift
    board_place = {1: (1.5, 1.8), 2: (3.0, 4.0), 4: (12.0, 15.0)}
    eps = np.array([-1.5, -0.5, 0.0, 0.7, 1.9])
    shocks = {1: eps, 2: eps[::-1].copy(), 3: np.full(5, 9.0), 4: eps * 0.5 + 0.3}
    draws = model.draws(make_board(place=board_place), pool=PLACE, tau_min=tau, shocks=shocks)

    assert draws.pool == PLACE
    assert sorted(draws.combinations) == ["1", "2", "4"]
    assert draws.odds.shape == draws.odds_hi.shape == (5, 3)
    for h, (lo, hi) in board_place.items():
        mu, sigma = ref_mu_sigma(ref, PLACE, tau, ref_band(lo))
        expected_lo = lo * np.exp(mu + sigma * shocks[h])
        assert expected_lo.min() > 1.0
        k = column(draws, str(h))
        np.testing.assert_allclose(draws.odds[:, k], expected_lo, rtol=1e-9)
        np.testing.assert_allclose(draws.odds_hi[:, k], hi * expected_lo / lo, rtol=1e-9)


@pytest.mark.parametrize("tau", [2, 5])
def test_a_wide_pair_uses_the_mean_shock_of_its_runners(drift, tau):
    model, ref = drift
    board_wide = {(1, 2): (4.0, 5.0), (1, 4): (15.0, 18.0), (2, 4): (25.0, 30.0)}
    shocks = {
        1: np.array([1.0, -2.0, 0.0, 0.4]),
        2: np.array([-1.0, 0.5, 1.5, 0.4]),
        4: np.array([2.0, 2.0, -0.5, -1.2]),
    }
    draws = model.draws(make_board(wide=board_wide), pool=WIDE, tau_min=tau, shocks=shocks)

    assert draws.pool == WIDE
    assert sorted(draws.combinations) == ["1,2", "1,4", "2,4"]
    for (a, b), (lo, hi) in board_wide.items():
        mu, sigma = ref_mu_sigma(ref, WIDE, tau, ref_band(lo))
        eps = (shocks[a] + shocks[b]) / math.sqrt(2.0)
        expected_lo = lo * np.exp(mu + sigma * eps)
        k = column(draws, f"{a},{b}")
        np.testing.assert_allclose(draws.odds[:, k], expected_lo, rtol=1e-9)
        np.testing.assert_allclose(draws.odds_hi[:, k], hi * expected_lo / lo, rtol=1e-9)


def test_draws_never_fall_below_one_and_high_stays_above_low(drift):
    model, _ = drift
    rng = np.random.default_rng(99)
    n = 600
    huge = np.where(np.arange(n) < n // 2, -60.0, 60.0)
    shocks = {1: huge, 2: -huge, 3: 5.0 * rng.standard_normal(n), 4: 5.0 * rng.standard_normal(n)}
    board = make_board(
        win={1: 1.6, 2: 3.2, 3: 3.2, 4: 40.0},
        place={1: (1.1, 1.2), 2: (1.5, 2.0), 3: (3.0, 4.0), 4: (9.0, 14.0)},
        wide={(1, 2): (1.3, 1.6), (1, 3): (2.0, 2.6), (2, 3): (4.0, 5.5), (3, 4): (30.0, 41.0)},
    )
    for tau in (2, 5):
        for pool in (WIN, PLACE, WIDE):
            draws = model.draws(board, pool=pool, tau_min=tau, shocks=shocks)
            assert np.isfinite(draws.odds).all(), (pool, tau)
            assert draws.odds.min() >= 1.0, (pool, tau)
            if pool != WIN:
                assert np.isfinite(draws.odds_hi).all(), (pool, tau)
                assert (draws.odds_hi >= draws.odds).all(), (pool, tau)
        # the shocks did drive prices down to the floor
        place = model.draws(board, pool=PLACE, tau_min=tau, shocks=shocks)
        assert (place.odds[:, column(place, "1")] == 1.0).any()


def test_after_the_post_the_draws_are_the_board(drift):
    model, _ = drift
    rng = np.random.default_rng(3)
    shocks = {h: rng.standard_normal(50) for h in (1, 2, 3)}
    board = make_board(
        win={1: 1.6, 2: 3.2, 3: 3.2},
        place={1: (1.2, 1.4), 2: (1.8, 2.5), 3: (2.2, 3.1)},
        wide={(1, 2): (2.0, 2.4), (1, 3): (3.0, 3.6), (2, 3): (6.5, 8.0)},
    )
    expected = {
        WIN: {str(h): (o, None) for h, o in board.win.items()},
        PLACE: {str(h): v for h, v in board.place.items()},
        WIDE: {f"{a},{b}": v for (a, b), v in board.wide.items()},
    }
    for pool, combos in expected.items():
        draws = model.draws(board, pool=pool, tau_min=0, shocks=shocks)
        assert sorted(draws.combinations) == sorted(combos)
        assert draws.odds.shape == (50, len(combos))
        for combination, (lo, hi) in combos.items():
            k = column(draws, combination)
            np.testing.assert_array_equal(draws.odds[:, k], np.full(50, lo))
            if hi is not None:
                np.testing.assert_array_equal(draws.odds_hi[:, k], np.full(50, hi))


# ---------------------------------------------------------------- training examples


GRID = DecisionGrid(minutes=(10, 5, 2), primary=2)
# minutes from post of each race's captures (the last one is the first final capture)
RACES = {
    1: {"post": POST, "runners": 4, "live": [-12.0, -6.0, -2.5, -0.5], "final": 2.0},
    2: {
        "post": POST + timedelta(minutes=30),
        "runners": 5,
        "live": [-40.0, -6.0, -2.5],
        "final": 2.5,
    },
}
FINAL_INDEX = 9


def price(race_no: int, pool: str, runners: tuple[int, ...], capture: int) -> tuple[float, float]:
    """(low, high) of one combination in one capture; distinct per capture."""
    base = [2.0, 3.5, 6.0, 11.0, 25.0]
    drift = 1.0 + 0.03 * capture + 0.01 * race_no
    win = [base[r - 1] * drift for r in runners]
    if pool == WIN:
        return round(win[0], 4), None
    if pool == PLACE:
        lo = 1.0 + (win[0] - 1.0) / 3.0
        return round(lo, 4), round(lo * 1.3, 4)
    if pool == WIDE:
        lo = 1.0 + win[0] * win[1] / 4.0
        return round(lo, 4), round(lo * 1.25, 4)
    return round(win[0] * win[1], 4), None  # QUINELLA


def combos(n_runners: int):
    singles = [(h,) for h in range(1, n_runners + 1)]
    pairs = [(a, b) for a in range(1, n_runners + 1) for b in range(a + 1, n_runners + 1)]
    return {WIN: singles, PLACE: singles, WIDE: pairs, "QUINELLA": pairs}


def capture_rows(day: date, race_no: int, n_runners: int, snap: datetime, index: int, final: bool):
    rows = []
    race_id = f"jra:{day.isoformat()}:{VENUE}:{race_no:02d}"
    for pool, keys in combos(n_runners).items():
        for runners in keys:
            lo, hi = price(race_no, pool, runners, index)
            r = {
                "snapshot_utc": snap.isoformat(),
                "race_id": race_id,
                "pool": pool,
                "combination": ",".join(str(h) for h in runners),
                "odds": lo,
                "status_raw": FINAL if final else f"{snap.hour + 9}時{snap.minute:02d}分現在",
                "source_update_utc": None if final else snap.isoformat(),
            }
            if hi is not None:
                r["odds_hi"] = hi
            rows.append(r)
        rows.append(
            {
                "snapshot_utc": snap.isoformat(),
                "race_id": race_id,
                "pool": pool,
                "combination": "__pool__",
                "odds": None,
                "status_raw": None,
                "source_update_utc": None if final else snap.isoformat(),
                "investment": 1_000_000.0 + index,
            }
        )
    return rows


def write_day(day: date) -> dict:
    """Write the synthetic log and sidecar of `day`; return the expected examples keyed by
    (race_id, pool, combination, tau) -> (odds_t, odds_final, board_age_s)."""
    shift = datetime.combine(day, POST.timetz()) - POST
    log = paths.odds_snapshot_path("jra", day.isoformat(), VENUE)
    captures = []
    for race_no, race in RACES.items():
        post = race["post"] + shift
        for index, minutes in enumerate(race["live"]):
            captures.append((post + timedelta(minutes=minutes), race_no, index, False))
        captures.append((post + timedelta(minutes=race["final"]), race_no, FINAL_INDEX, True))
    for snap, race_no, index, final in sorted(captures):
        rows = capture_rows(day, race_no, RACES[race_no]["runners"], snap, index, final)
        odds_log.append_capture(log, rows)
    posts = {no: race["post"] + shift for no, race in RACES.items()}
    odds_log.write_post_times(log.with_name(f"{VENUE}.races.json"), posts, POST + shift)

    expected = {}
    for race_no, race in RACES.items():
        race_id = f"jra:{day.isoformat()}:{VENUE}:{race_no:02d}"
        for tau in GRID.minutes:
            used = max(i for i, m in enumerate(race["live"]) if m <= -tau)
            age = (race["live"][used] + tau) * -60.0
            for pool, keys in combos(race["runners"]).items():
                if pool == "QUINELLA":
                    continue
                for runners in keys:
                    combination = ",".join(str(h) for h in runners)
                    odds_t = price(race_no, pool, runners, used)[0]
                    odds_final = price(race_no, pool, runners, FINAL_INDEX)[0]
                    expected[(race_id, pool, combination, tau)] = (odds_t, odds_final, age)
    return expected


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


def keyed(examples: pd.DataFrame) -> dict:
    out = {}
    for r in examples.itertuples(index=False):
        key = (r.race_id, r.pool, str(r.combination), int(r.tau_min))
        assert key not in out, f"duplicate example {key}"
        out[key] = (float(r.odds_t), float(r.odds_final), float(r.board_age_s))
    return out


def assert_examples_match(examples: pd.DataFrame, expected: dict) -> None:
    got = keyed(examples)
    assert set(got) == set(expected)
    for key, (odds_t, odds_final, age) in expected.items():
        assert got[key][0] == pytest.approx(odds_t, rel=1e-12), key
        assert got[key][1] == pytest.approx(odds_final, rel=1e-12), key
        assert got[key][2] == pytest.approx(age, abs=1e-6), key


def test_training_examples_from_a_synthetic_day(data_dir):
    expected = write_day(DEV_DAY)
    assert len(expected) == 3 * (4 + 4 + 6) + 3 * (5 + 5 + 10)
    examples = training_examples(
        DEV_DAY, VENUE, grid=GRID, max_board_age_s={10: 1e6, 5: 1e6, 2: 1e6}
    )
    for name in (
        "race_id",
        "day",
        "pool",
        "combination",
        "tau_min",
        "odds_t",
        "odds_final",
        "board_age_s",
    ):
        assert name in examples.columns, name
    assert set(examples["pool"]) == {WIN, PLACE, WIDE}
    assert {str(d)[:10] for d in examples["day"]} == {DEV_DAY.isoformat()}
    assert_examples_match(examples, expected)
    assert sum(examples.attrs.get("dropped", {}).values()) == 0

    # the examples fit, and the fit records the day it used
    model = NowcastV0.fit(examples, bands=BANDS, grid=GRID, prior_n=PRIOR_N)
    assert [str(d)[:10] for d in model.days] == [DEV_DAY.isoformat()]


def test_stale_boards_are_dropped_and_counted(data_dir):
    expected = write_day(DEV_DAY)
    limits = {10: 300.0, 5: 60.0, 2: 29.0}  # race 2 at tau 10 is 1800 s old; tau 2 boards 30 s
    examples = training_examples(DEV_DAY, VENUE, grid=GRID, max_board_age_s=limits)
    kept = {k: v for k, v in expected.items() if v[2] <= limits[k[3]]}
    assert len(kept) == 14 + 14 + 20  # race 1 at tau 10 and 5, race 2 at tau 5 (60 s: kept)
    assert_examples_match(examples, kept)
    dropped = {int(t): int(n) for t, n in examples.attrs["dropped"].items() if n}
    assert dropped == {10: 20, 2: 14 + 20}


def first_day_from(start: date, holdout: bool, seed: str) -> date:
    day = start
    while True:
        digest = hashlib.sha256(f"{seed}:{day.isoformat()}".encode()).hexdigest()
        if (int(digest, 16) % 2 == 0) is holdout:
            return day
        day += timedelta(days=1)


def test_a_holdout_day_is_refused(data_dir, monkeypatch):
    assert issubclass(HoldoutDayError, RuntimeError)
    write_day(DEV_DAY)
    limits = {10: 1e6, 5: 1e6, 2: 1e6}
    monkeypatch.setattr(nowcast, "is_holdout_day", lambda *a, **k: True)
    with pytest.raises(HoldoutDayError):
        training_examples(DEV_DAY, VENUE, grid=GRID, max_board_age_s=limits)


def test_the_real_holdout_rule_decides_which_days_are_refused(data_dir):
    section = config.load_country_yaml("japan", "holdout")["holdout"]
    start = section["start"]
    start = start if isinstance(start, date) else date.fromisoformat(str(start))
    held = first_day_from(start, True, str(section["seed"]))
    open_day = first_day_from(start, False, str(section["seed"]))
    limits = {10: 1e6, 5: 1e6, 2: 1e6}
    for day in (held, open_day):
        write_day(day)
    with pytest.raises(HoldoutDayError):
        training_examples(held, VENUE, grid=GRID, max_board_age_s=limits)
    examples = training_examples(open_day, VENUE, grid=GRID, max_board_age_s=limits)
    assert len(examples) == 102


# ---------------------------------------------------------------- split captures (#234)
# HKTrack logged each round as separate captures: WIN+PLACE at s, QUINELLA 1 s on, Wide 2 s on.
# Owner sign-off 2026-11-25: the final board takes each pool's final capture, so Wide's first final
# capture at or after the anchor (the first final WIN+PLACE capture) prices the final Wide: the
# anchor round's own Wide capture when it is final, else the next round's (2026-11-22 KYO races 01,
# 04 and 09: the anchor round's Wide still pre-final). The round before's final Wide never does
# while one at or after exists. Each capture's prices differ (`price` by index).

QUINELLA = "QUINELLA"
SPLIT_POOLS = (
    (WIN, PLACE),
    (QUINELLA,),
    (WIDE,),
)  # a round's captures, 0, 1 and 2 s from its start
PRE_INDEX = 8  # the round 20 s before the final one: WIN not final yet, Wide already final
SPLIT_WIDE_INDEX = 10  # the final round's Wide capture (its WIN+PLACE capture is FINAL_INDEX)
LATER_INDEXES = (11, 12)  # final rounds 20 s and 40 s after the final one
NO_LIMIT = {10: 1e6, 5: 1e6, 2: 1e6}
# the final round's Wide capture per race: "final", "pre_final" (the KYO shape) or "missing"
ANCHOR_WIDE = {1: "final", 2: "pre_final"}


def wide_final_index(mode: str) -> int:
    """The capture index Wide's final price comes from: the first final Wide at or after the
    anchor, the final round's own when it is final, else the next round's."""
    return SPLIT_WIDE_INDEX if mode == "final" else LATER_INDEXES[0]


# live rounds each within the shipped max_board_age_s of one tau of the shipped grid
SHIPPED_LIVE = [-61.0, -31.0, -17.0, -5.5, -2.5, -1.25]
SHIPPED_RACES = {
    1: {"post": POST, "runners": 4, "live": SHIPPED_LIVE, "final": 2.0},
    2: {"post": POST + timedelta(minutes=30), "runners": 5, "live": SHIPPED_LIVE, "final": 2.5},
}


def write_split_day(day: date, *, races=RACES, grid=GRID, anchor_wide=None) -> dict:
    """write_day with every round split into HKTrack's three captures, a round before the final one
    whose Wide is already final, and two later final rounds; `anchor_wide` (default ANCHOR_WIDE)
    sets each race's final-round Wide capture. Returns the expected examples as write_day does."""
    anchor_wide = ANCHOR_WIDE if anchor_wide is None else anchor_wide
    shift = datetime.combine(day, POST.timetz()) - POST
    log = paths.odds_snapshot_path("jra", day.isoformat(), VENUE)
    captures = []
    for race_no, race in races.items():
        post = race["post"] + shift
        final_at = post + timedelta(minutes=race["final"])
        mode = anchor_wide[race_no]
        wide_index = None if mode == "missing" else SPLIT_WIDE_INDEX
        rounds = [
            (post + timedelta(minutes=m), (i, i, i), (False, False, False))
            for i, m in enumerate(race["live"])
        ]
        rounds.append((final_at - timedelta(seconds=20), (PRE_INDEX,) * 3, (False, True, True)))
        rounds.append(
            (final_at, (FINAL_INDEX, FINAL_INDEX, wide_index), (True, True, mode == "final"))
        )
        for k, index in enumerate(LATER_INDEXES, start=1):
            rounds.append((final_at + timedelta(seconds=20 * k), (index,) * 3, (True, True, True)))
        for start, indexes, finals in rounds:
            parts = zip(SPLIT_POOLS, indexes, finals, strict=True)
            for offset, (pools, index, final) in enumerate(parts):
                if index is not None:
                    snap = start + timedelta(seconds=offset)
                    captures.append((snap, race_no, pools, index, final))
    for snap, race_no, pools, index, final in sorted(captures):
        rows = capture_rows(day, race_no, races[race_no]["runners"], snap, index, final)
        odds_log.append_capture(log, [r for r in rows if r["pool"] in pools])
    posts = {no: race["post"] + shift for no, race in races.items()}
    odds_log.write_post_times(log.with_name(f"{VENUE}.races.json"), posts, POST + shift)

    expected = {}
    for race_no, race in races.items():
        race_id = f"jra:{day.isoformat()}:{VENUE}:{race_no:02d}"
        live_s = [m * 60.0 for m in race["live"]]
        for tau in grid.minutes:
            at = -60.0 * tau  # seconds from the post
            used = max(i for i, s in enumerate(live_s) if s <= at)  # WIN+PLACE capture
            used_wide = max(i for i, s in enumerate(live_s) if s + 2.0 <= at)  # Wide capture
            age = at - min(live_s[used], live_s[used_wide] + 2.0)
            for pool, keys in combos(race["runners"]).items():
                if pool == QUINELLA:
                    continue
                index_t = used_wide if pool == WIDE else used
                index_final = (
                    wide_final_index(anchor_wide[race_no]) if pool == WIDE else FINAL_INDEX
                )
                for runners in keys:
                    combination = ",".join(str(h) for h in runners)
                    expected[(race_id, pool, combination, tau)] = (
                        price(race_no, pool, runners, index_t)[0],
                        price(race_no, pool, runners, index_final)[0],
                        age,
                    )
    return expected


def expected_frame(expected: dict) -> pd.DataFrame:
    """The expected examples as the columns ref_cells reads."""
    return pd.DataFrame(
        [
            {"pool": k[1], "tau_min": k[3], "odds_t": v[0], "odds_final": v[1]}
            for k, v in expected.items()
        ]
    )


def wide_per_tau(pools, taus) -> dict[int, int]:
    out: dict[int, int] = {}
    for pool, tau in zip(pools, taus, strict=True):
        if pool == WIDE:
            out[int(tau)] = out.get(int(tau), 0) + 1
    return out


def test_training_examples_take_wide_from_its_first_final_capture(data_dir):
    """Race 1's final-round Wide capture is final and prices its Wide finals; race 2's is still
    pre-final (the KYO 01/04/09 shape), so the next round's final Wide capture prices race 2's."""
    expected = write_split_day(DEV_DAY)
    assert len(expected) == 3 * (4 + 4 + 6) + 3 * (5 + 5 + 10)
    examples = training_examples(DEV_DAY, VENUE, grid=GRID, max_board_age_s=NO_LIMIT)
    assert wide_per_tau(examples["pool"], examples["tau_min"]) == {
        tau: 6 + 10 for tau in GRID.minutes
    }
    # Wide's final price: index 10 for race 1, 11 for race 2; WIN and PLACE are the anchor's
    assert_examples_match(examples, expected)
    assert examples.attrs["skipped"]["no_final_board"] == 0
    assert sum(examples.attrs["dropped"].values()) == 0


def test_the_streamed_final_board_is_final_board_over_the_whole_log(data_dir):
    """training_examples reads only some rows of the log; its final prices must be final_board's
    over every row: the first final Wide capture at or after the anchor in, the others out."""
    write_split_day(DEV_DAY)
    rows = odds_log.read_rows(paths.odds_snapshot_path("jra", DEV_DAY.isoformat(), VENUE))
    got = keyed(training_examples(DEV_DAY, VENUE, grid=GRID, max_board_age_s=NO_LIMIT))
    for race_no, race in RACES.items():
        race_id = f"jra:{DEV_DAY.isoformat()}:{VENUE}:{race_no:02d}"
        board = final_board(rows, race_id)
        keys = combos(race["runners"])
        index = wide_final_index(ANCHOR_WIDE[race_no])
        assert board.wide == {pair: price(race_no, WIDE, pair, index) for pair in keys[WIDE]}
        assert board.win == {h: price(race_no, WIN, (h,), FINAL_INDEX)[0] for (h,) in keys[WIN]}
        assert board.place == {h: price(race_no, PLACE, (h,), FINAL_INDEX) for (h,) in keys[PLACE]}
        on_board = {(WIN, str(h)): odds for h, odds in board.win.items()}
        on_board |= {(PLACE, str(h)): lo for h, (lo, _) in board.place.items()}
        on_board |= {(WIDE, f"{a},{b}"): lo for (a, b), (lo, _) in board.wide.items()}
        streamed = {(k[1], k[2]): v[1] for k, v in got.items() if k[0] == race_id}
        assert streamed == on_board, race_id


def test_a_race_without_wide_in_its_final_round_takes_wide_from_the_next_final_capture(data_dir):
    """Race 2 logs no Wide capture in its final round: its Wide finals come from the next round's
    final Wide capture (index 11), not from the final Wide capture of the round before (8) nor a
    later one (12); race 1 keeps its own (10)."""
    expected = write_split_day(DEV_DAY, anchor_wide={1: "final", 2: "missing"})
    examples = training_examples(DEV_DAY, VENUE, grid=GRID, max_board_age_s=NO_LIMIT)
    wide = examples[examples["pool"] == WIDE]
    assert set(wide["race_id"]) == {f"jra:{DEV_DAY.isoformat()}:{VENUE}:{no:02d}" for no in (1, 2)}
    race_2 = wide[wide["race_id"] == f"jra:{DEV_DAY.isoformat()}:{VENUE}:02"]
    for r in race_2.itertuples(index=False):
        pair = tuple(int(h) for h in r.combination.split(","))
        assert r.odds_final == price(2, WIDE, pair, LATER_INDEXES[0])[0], r.combination
    assert_examples_match(examples, expected)
    assert examples.attrs["skipped"]["no_final_board"] == 0


def test_a_fit_on_split_capture_examples_has_wide_cells_at_every_tau(data_dir):
    expected = write_split_day(DEV_DAY)
    examples = training_examples(DEV_DAY, VENUE, grid=GRID, max_board_age_s=NO_LIMIT)
    model = NowcastV0.fit(examples, bands=BANDS, grid=GRID, prior_n=PRIOR_N)
    assert {tau for pool, tau, _ in model.cells if pool == WIDE} == set(GRID.minutes)
    cells, sigma0 = ref_cells(expected_frame(expected), BANDS, PRIOR_N)
    assert set(model.cells) == set(cells)
    for key, (n, mu, sigma) in cells.items():
        cell = model.cell(*key)
        assert cell.n == n, key
        assert cell.mu == pytest.approx(mu, abs=1e-12), key
        assert cell.sigma == pytest.approx(sigma, rel=1e-9), key
    for tau in GRID.minutes:
        assert model.sigma0[(WIDE, tau)] == pytest.approx(sigma0[(WIDE, tau)], rel=1e-9), tau


def test_a_refit_lists_wide_cells_for_every_shipped_grid_tau(data_dir, tmp_path):
    """run_fit on a split-capture development day, the shipped grid and config: the fit and its
    manifest hold QUINELLA_PLACE cells at every grid tau."""
    grid = load_decision_grid()
    expected = write_split_day(DEV_DAY, races=SHIPPED_RACES, grid=grid)
    manifest = nowcast.run_fit([(DEV_DAY, VENUE)], out_dir=tmp_path / "fit")
    wide_n: dict[int, int] = {}
    for c in manifest["examples_per_cell"]:
        if c["pool"] == WIDE:
            wide_n[int(c["tau_min"])] = wide_n.get(int(c["tau_min"]), 0) + int(c["n"])
    assert wide_n == {tau: 6 + 10 for tau in grid.minutes}
    assert manifest["n_examples"] == len(expected)
    label = f"{DEV_DAY.isoformat()}:{VENUE}"
    assert manifest["dropped_stale_examples"] == {label: {str(t): 0 for t in grid.minutes}}
    model = NowcastV0.from_json(Path(manifest["fit_path"]).read_text())
    assert {tau for pool, tau, _ in model.cells if pool == WIDE} == set(grid.minutes)
    assert wide_per_tau([k[1] for k in expected], [k[3] for k in expected]) == wide_n


# ---------------------------------------------------------------- scores


def test_pit_is_uniform_and_coverage_nominal_on_simulated_data():
    rng = np.random.default_rng(20261124)
    grid = DecisionGrid(minutes=(5, 2), primary=2)
    records = []
    for tau, scale in ((2, 1.0), (5, 1.8)):
        for odds_t, mean, sd in ((3.0, 0.15, 0.10), (7.0, -0.2, 0.25), (14.0, 0.3, 0.4)):
            records += cell_records(rng, PLACE, tau, odds_t, mean * scale, sd * scale, 40)
    training = frame(records)
    model = NowcastV0.fit(training, bands=BANDS, grid=grid, prior_n=PRIOR_N)
    ref = ref_cells(training, BANDS, PRIOR_N)

    # 5,000 examples simulated from the fitted model, band 4 (empty) included
    n = 5000
    taus = rng.choice([2, 5], size=n)
    bands = rng.choice([1, 2, 3, 4], size=n)
    lows = np.array([2.0, 5.0, 10.0, 20.0, 50.0])
    odds_t = rng.uniform(lows[bands - 1], lows[bands])
    params = np.array(
        [ref_mu_sigma(ref, PLACE, int(t), int(b)) for t, b in zip(taus, bands, strict=True)]
    )
    mu, sigma = params[:, 0], params[:, 1]
    y = mu + sigma * rng.standard_normal(n)
    sim = frame([(PLACE, t, o, v, "2026-11-21") for t, o, v in zip(taus, odds_t, y, strict=True)])
    y = np.log(sim["odds_final"].to_numpy()) - np.log(sim["odds_t"].to_numpy())

    u = np.asarray(pit(model, sim), dtype=float)
    assert u.shape == (n,)
    np.testing.assert_allclose(u, stats.norm.cdf((y - mu) / sigma), atol=1e-9)
    assert stats.kstest(u, "uniform").pvalue > 0.01
    assert 0.87 <= coverage(model, sim, level=0.9) <= 0.93
    assert 0.46 <= coverage(model, sim, level=0.5) <= 0.54
    lpd = np.asarray(log_predictive_density(model, sim), dtype=float)
    np.testing.assert_allclose(lpd, stats.norm.logpdf(y, mu, sigma), rtol=1e-9)

    # the same examples moved 4 sigma up: neither PIT nor coverage can still look nominal
    shifted = sim.assign(odds_final=sim["odds_final"] * np.exp(4.0 * sigma))
    assert coverage(model, shifted, level=0.9) < 0.1
    assert np.asarray(pit(model, shifted), dtype=float).mean() > 0.9


def test_the_fit_round_trips_through_json(drift):
    model, ref = drift
    assert sorted(str(d)[:10] for d in model.days) == ["2026-11-21", "2026-11-22"]
    text = model.to_json()
    json.loads(text)
    back = NowcastV0.from_json(text)
    assert [str(d)[:10] for d in back.days] == [str(d)[:10] for d in model.days]

    keys = [(pool, tau, band) for pool in (WIN, PLACE, WIDE) for tau in (2, 5) for band in range(7)]
    for key in keys:
        a, b = model.cell(*key), back.cell(*key)
        assert (b.pool, b.tau_min, b.band, b.n) == (a.pool, a.tau_min, a.band, a.n), key
        assert b.mu == a.mu and b.sigma == a.sigma, key
        mu, sigma = ref_mu_sigma(ref, *key)
        assert b.mu == pytest.approx(mu, abs=1e-12) and b.sigma == pytest.approx(sigma, rel=1e-9)

    rng = np.random.default_rng(5)
    shocks = {h: rng.standard_normal(30) for h in (1, 2, 3)}
    board = make_board(
        win={1: 1.6, 2: 3.2, 3: 3.2},
        place={1: (1.2, 1.4), 2: (1.8, 2.5), 3: (2.2, 3.1)},
        wide={(1, 2): (2.0, 2.4), (1, 3): (3.0, 3.6), (2, 3): (6.5, 8.0)},
    )
    for pool in (WIN, PLACE, WIDE):
        a = model.draws(board, pool=pool, tau_min=2, shocks=shocks)
        b = back.draws(board, pool=pool, tau_min=2, shocks=shocks)
        assert a.combinations == b.combinations
        np.testing.assert_array_equal(a.odds, b.odds)
        if a.odds_hi is not None:
            np.testing.assert_array_equal(a.odds_hi, b.odds_hi)

    examples = draw_examples()
    np.testing.assert_array_equal(
        np.asarray(pit(model, examples), dtype=float), np.asarray(pit(back, examples), dtype=float)
    )
