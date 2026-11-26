"""The C1 final-price nowcast v0 for the Win and Place pools (ticket #85).

Module under test: `horsetrack.countries.japan.models.nowcast`, not the drift nowcast of #116 in
`models/v0/nowcast.py`. Contract pinned here:

Shares and the late composition
- A combination's share is its money over the pool. A WIN board's shares are 1/O_i normalised
  over the priced runners. PLACE shares come from the per-runner money or the low odds by a rule
  left to the module; here they only have to sum to 1, with the largest share on the runner with
  the lowest low odds.
- `late_composition(v1, s1, v2, s2) -> np.ndarray`: the composition of the money bet between two
  snapshots, (v2 * s2 - v1 * s1) / (v2 - v1).
- The final shares are s_final = phi * s_t + (1 - phi) * s_late, with phi = V_t / V_final.

The board at t: `RaceView`, `PoolBoard`, `race_view`
- `RaceView(race_id, venue, race_no, grade, tau_min, pools)`, where `pools` maps a pool to
  `PoolBoard(combinations, shares, total, shares_lag5, shares_lag15)`. `combinations` are the
  runner numbers as text ("7"); `shares` is s_t over them (sums to 1); `total` is V_t in yen, or
  None when the board prints no pool total; `shares_lag5` and `shares_lag15` are the shares on the
  board 5 and 15 minutes before t, in the same order, or None when there was no board then.
- `race_view(frame, race_id, *, tau_min) -> RaceView` builds it from rows in
  `logger.compact.SCHEMA` (the jravan odds_timeseries day files and the logger's compacted days).
  t = post_utc - tau_min. A row counts from its snapshot_utc (owner: no lag), only rows at or
  before t are read, and each combination takes its latest. `total` is the latest `__pool__`
  investment at or before t. The lag boards are read the same way at t - 5 and t - 15 minutes.

The model: `Nowcast`, `Draws`
- `Nowcast.fit(frame) -> Nowcast` fits on every race of `frame`: rows in compact.SCHEMA over any
  number of days, with pools other than WIN and PLACE ignored. The fit is deterministic. It records
  the race days it used as `model.days`, sorted "YYYY-MM-DD" strings taken from the race id (the
  race day, not the snapshot's date). It refuses a frame holding a holdout day with
  `HoldoutDayError` (`core.eval.holdout.is_holdout_day("japan", day)`, D14).
- `Nowcast.from_params(*, phi, sigma) -> Nowcast` is the known nowcast the tests simulate from:
  phi_t = phi for every race, pool and tau; s_late = softmax(ln s_t + sigma * eps) with eps iid
  N(0, 1) per runner (beta = 0, so the features do not enter); V_final = V_t / phi.
- `model.draws(view, *, pool, n, seed) -> Draws(shares, v_final)`. `shares` is (n, k), its columns
  in `view.pools[pool].combinations` order, each row summing to 1 with entries >= 0. `v_final` is
  (n,), positive, and never below V_t (phi <= 1). The spread of the shares scales with 1 - phi, and
  at phi = 1 every draw is the board. The same seed gives the same draws.
- `model.phi(view, *, pool) -> float` is the phi_t = V_t / V_final the draws use. Without a pool
  total on the board, it comes from the time-to-post profile fitted on the days that print totals.
- `model.random_walk(view, *, pool, n, seed) -> Draws` is the q_final = q_t baseline with its
  fitted noise: share vectors summing to 1.

Folds and scores
- `fit_folds(frame, *, scheme) -> list` returns the fold fits in date order. Each has `.model`, a
  Nowcast fitted on that fold's training days only, and `.test_days`, its test days. The folds are
  `core.eval.folds.walk_forward` over the frame's development days: holdout days are dropped
  first and are never trained or tested on.
- `compare(candidate, baseline, frame, *, n, seed, n_boot, level) -> pd.DataFrame`. Candidate and
  baseline are forecasters with the draws signature, and every case (race, pool, tau) gives both
  the same view, n and seed. There is one row per (pool, tau_min) for WIN and PLACE at every tau of
  the decision grid (config/decision.yaml). Columns `log_score_gain` and `crps_gain` are the mean
  per-case gain on the final ln-shares, positive when the candidate is better (the log score uses
  `gaussian_log_score` on the ln-share draws, CRPS uses `crps_ensemble`; how a case aggregates its
  runners is left to the module). `<gain>_lo` and `<gain>_hi` bound a bootstrap interval that
  resamples race days. A frame holding a holdout day is refused with `HoldoutDayError`.

Expected values are derived here from first principles (money, the stated formulas, scipy), never
from the module under test. All data are synthetic and in memory, and HORSETRACK_DATA_DIR points at
a tmp dir.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from scipy import stats

from horsetrack.core import config, paths
from horsetrack.core.eval.distributional import pit
from horsetrack.core.eval.folds import FoldScheme, walk_forward
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.countries.japan.board.snapshot import PLACE, WIN
from horsetrack.countries.japan.logger.compact import SCHEMA
from horsetrack.countries.japan.models.nowcast import (
    Draws,
    HoldoutDayError,
    Nowcast,
    PoolBoard,
    RaceView,
    compare,
    fit_folds,
    late_composition,
    race_view,
)

UTC = timezone.utc
COUNTRY = "japan"
VENUE = "KYO"
POOL_ROW = "__pool__"  # the pool-total row of the odds logs and the jravan day files
TAKEOUT = 0.20  # WIN and PLACE in config/jra.yaml
GRID = tuple(
    sorted(
        (int(m) for m in config.load_country_yaml(COUNTRY, "decision")["grid_min"]), reverse=True
    )
)
GAINS = ("log_score_gain", "crps_gain")
GAIN_COLUMNS = tuple(f"{g}{suffix}" for g in GAINS for suffix in ("", "_lo", "_hi"))

# Snapshots, in minutes to the scheduled post: one the evening before (900), then every grid tau
# with its t-5 and t-15 boards, and the final capture 2 minutes after the post.
MINUTES = (900, 90, 75, 65, 60, 45, 35, 31, 30, 21, 20, 17, 16, 10, 7, 6, 5, 2, 1)
FINAL_MIN = -2
SNAPSHOTS = (*MINUTES, FINAL_MIN)
# Pool growth: V(m) / V_final = exp(-rate * m) before the post, so phi at tau is exp(-rate * tau).
GROWTH = {WIN: 0.04, PLACE: 0.06}
RUNNERS = (12, 10, 14)  # field sizes of races 1, 2 and 3 of a synthetic day

TRAIN_DAYS = tuple(
    date(2026, 10, d) for d in (3, 4, 10, 11, 17, 18, 24, 25)
)  # weekends before the holdout start
TEST_DAYS = (date(2026, 11, 14), date(2026, 11, 15), date(2026, 11, 21))

# A 12-runner board with a clear favourite (runner 1).
TWELVE = np.array([0.30, 0.15, 0.12, 0.09, 0.08, 0.06, 0.05, 0.04, 0.04, 0.03, 0.02, 0.02])


# ---------------------------------------------------------------- synthetic odds paths


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def place_odds(shares: np.ndarray, pool_total: float) -> tuple[np.ndarray, np.ndarray]:
    """Low and high PLACE odds under JRA's three-place rule: the net pool less the stakes on the
    three placed runners is split in three, each third over its runner's stakes. Low odds: placed
    with the two most-backed others; high odds: with the two least-backed."""
    money = np.asarray(shares, dtype=float) * pool_total
    net = (1.0 - TAKEOUT) * pool_total
    lo, hi = np.empty(len(money)), np.empty(len(money))
    for i, own in enumerate(money):
        others = np.sort(np.delete(money, i))
        lo[i] = 1.0 + (net - own - others[-2:].sum()) / (3.0 * own)
        hi[i] = 1.0 + (net - own - others[:2].sum()) / (3.0 * own)
    return np.maximum(lo, 1.1), np.maximum(hi, 1.1)


def pool_path(rng, k: int, rate: float, sigma: float, v_final: float):
    """(V, shares) at each of SNAPSHOTS. The pool grows as V_final * exp(-rate * m) and reaches
    V_final at the close; the money added at each snapshot has the composition
    softmax(ln s_prev + sigma * z)."""
    fractions = [math.exp(-rate * m) for m in MINUTES] + [1.0]
    shares = rng.dirichlet(np.full(k, 2.0))
    money = v_final * fractions[0] * shares
    path = [(v_final * fractions[0], shares)]
    for fraction in fractions[1:]:
        total = v_final * fraction
        late = softmax(np.log(money / money.sum()) + sigma * rng.standard_normal(k))
        money = money + (total - money.sum()) * late
        path.append((total, money / money.sum()))
    return path


def post_of(day: date, race_no: int) -> datetime:
    return datetime(day.year, day.month, day.day, 1, 0, tzinfo=UTC) + timedelta(
        minutes=35 * race_no
    )


def race_id_of(day: date, race_no: int) -> str:
    return f"jra:{day.isoformat()}:{VENUE}:{race_no:02d}"


def row(
    race_id,
    race_no,
    pool,
    combination,
    post,
    minutes,
    *,
    odds=None,
    odds_hi=None,
    investment=None,
    final=False,
) -> dict:
    """One row in logger.compact.SCHEMA, jravan-style: no fetch time, a snapshot at its minute."""
    return {
        "race_id": race_id,
        "venue": VENUE,
        "race_no": race_no,
        "pool": pool,
        "combination": combination,
        "snapshot_utc": post - timedelta(minutes=minutes),
        "fetched_utc": None,
        "post_utc": post,
        "minutes_to_post": float(minutes),
        "odds": None if odds is None else float(odds),
        "odds_hi": None if odds_hi is None else float(odds_hi),
        "investment": None if investment is None else float(investment),
        "status_raw": None,
        "is_final_capture": bool(final),
        "dividend": None,
        "is_refund": None,
    }


def race_rows(rng, day, race_no, k, *, totals="all", rate_scale=1.0, sigma=0.3) -> list[dict]:
    """A race's WIN and PLACE paths. `totals`: "all" prints the pool total on every snapshot
    (jravan), "final" only at the close (08 R20, our own logs), "none" never."""
    race_id, post = race_id_of(day, race_no), post_of(day, race_no)
    rows = []
    for pool in (WIN, PLACE):
        rate = GROWTH[pool] * rate_scale * (1.0 + 0.05 * rng.uniform(-1.0, 1.0))
        v_final = 1e8 * math.exp(0.3 * rng.standard_normal())
        for minutes, (total, shares) in zip(
            SNAPSHOTS, pool_path(rng, k, rate, sigma, v_final), strict=True
        ):
            final = minutes == FINAL_MIN
            if pool == WIN:
                lo, hi = (1.0 - TAKEOUT) / shares, [None] * k
            else:
                lo, hi = place_odds(shares, total)
            for j in range(k):
                rows.append(
                    row(
                        race_id,
                        race_no,
                        pool,
                        str(j + 1),
                        post,
                        minutes,
                        odds=lo[j],
                        odds_hi=hi[j],
                        final=final,
                    )
                )
            if totals == "all" or (totals == "final" and final):
                rows.append(
                    row(
                        race_id,
                        race_no,
                        pool,
                        POOL_ROW,
                        post,
                        minutes,
                        investment=total,
                        final=final,
                    )
                )
    for minutes in SNAPSHOTS:  # a pool the nowcast does not read
        rows.append(
            row(
                race_id,
                race_no,
                "QUINELLA",
                "1,2",
                post,
                minutes,
                odds=9.9,
                final=minutes == FINAL_MIN,
            )
        )
    return rows


def day_rows(day, *, races=3, totals="all", rate_scale=1.0, sigma=0.3) -> list[dict]:
    """A synthetic race day; the same day gives the same paths whatever `totals` keeps."""
    rng = np.random.default_rng(day.toordinal())
    rows = []
    for race_no in range(1, races + 1):
        k = RUNNERS[(race_no - 1) % len(RUNNERS)]
        rows += race_rows(rng, day, race_no, k, totals=totals, rate_scale=rate_scale, sigma=sigma)
    return rows


def to_frame(rows: list[dict]) -> pd.DataFrame:
    """Rows as a day file reads back: exactly logger.compact.SCHEMA."""
    return pa.Table.from_pylist(rows, schema=SCHEMA).to_pandas()


def race_day(race_id: str) -> str:
    return race_id.split(":")[1]


def iso(days) -> list[str]:
    """Days (dates or "YYYY-MM-DD" strings) as "YYYY-MM-DD" strings."""
    return [str(d)[:10] for d in days]


def days_of(frame: pd.DataFrame, days) -> pd.DataFrame:
    keep = {str(d)[:10] for d in days}
    return frame[frame["race_id"].map(race_day).isin(keep)].reset_index(drop=True)


def final_win_shares(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Each race's final WIN shares, 1/O normalised over the final capture."""
    rows = frame[
        (frame["pool"] == WIN) & frame["is_final_capture"] & (frame["combination"] != POOL_ROW)
    ]
    out = {}
    for race_id, g in rows.groupby("race_id"):
        inverse = 1.0 / g["odds"].to_numpy(float)
        out[race_id] = dict(zip(g["combination"], inverse / inverse.sum(), strict=True))
    return out


def make_view(shares, *, pool=WIN, total=4e7, tau=5.0, lag5=None, lag15=None, race_no=1):
    shares = np.asarray(shares, dtype=float)
    board = PoolBoard(
        combinations=tuple(str(j + 1) for j in range(len(shares))),
        shares=shares,
        total=total,
        shares_lag5=lag5,
        shares_lag15=lag15,
    )
    return RaceView(
        race_id=race_id_of(TEST_DAYS[0], race_no),
        venue=VENUE,
        race_no=race_no,
        grade=None,
        tau_min=float(tau),
        pools={pool: board},
    )


def make_oracle(truth: dict[str, dict[str, float]], fallback):
    """A forecaster that knows each race's final WIN shares (draws within 1e-4 in ln-share) and
    is `fallback` on every other pool."""

    def oracle(view, *, pool, n, seed):
        if pool != WIN:
            return fallback(view, pool=pool, n=n, seed=seed)
        combinations = view.pools[pool].combinations
        s = np.array([truth[view.race_id][c] for c in combinations])
        noise = 1e-4 * np.random.default_rng(seed).standard_normal((n, len(s)))
        return Draws(shares=softmax(np.log(s) + noise), v_final=np.full(n, 1e8))

    return oracle


def assert_board(combinations, shares, expected: dict[str, float]) -> None:
    assert shares is not None
    got = dict(zip(combinations, np.asarray(shares, dtype=float), strict=True))
    assert set(got) == set(expected)
    for c, value in expected.items():
        assert got[c] == pytest.approx(value, rel=1e-9), c


def assert_share_draws(draws, n: int, k: int) -> None:
    assert draws.shares.shape == (n, k)
    assert np.isfinite(draws.shares).all()
    assert (draws.shares >= 0.0).all()
    np.testing.assert_allclose(draws.shares.sum(axis=1), 1.0, rtol=0.0, atol=1e-9)
    assert draws.v_final.shape == (n,)


# ---------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


@pytest.fixture(scope="module")
def train_frame() -> pd.DataFrame:
    assert not any(is_holdout_day(COUNTRY, d) for d in TRAIN_DAYS + TEST_DAYS)
    return to_frame([r for d in TRAIN_DAYS for r in day_rows(d)])


@pytest.fixture(scope="module")
def scored_frame() -> pd.DataFrame:
    return to_frame([r for d in TEST_DAYS for r in day_rows(d)])


@pytest.fixture(scope="module")
def fitted(train_frame, tmp_path_factory) -> Nowcast:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(paths.DATA_DIR_VAR, str(tmp_path_factory.mktemp("fit-data")))
        return Nowcast.fit(train_frame)


# ---------------------------------------------------------------- draws


def test_draws_sum_to_one(fitted, scored_frame):
    """500 draws for a 12-runner race: every row a share vector summing to 1 within 1e-9, for the
    fitted nowcast and its random walk at every grid tau and both pools, and for a known nowcast."""
    race_id = race_id_of(TEST_DAYS[1], 1)  # race 1 has 12 runners
    cases = []
    for tau in GRID:
        view = race_view(scored_frame, race_id, tau_min=tau)
        for pool in (WIN, PLACE):
            cases += [(fitted.draws, view, pool), (fitted.random_walk, view, pool)]
    known = Nowcast.from_params(phi=0.6, sigma=0.5)
    for pool in (WIN, PLACE):
        bare = make_view(TWELVE, pool=pool, total=3e7, tau=16)  # no lag boards
        cases += [(known.draws, bare, pool), (fitted.draws, bare, pool)]
    for i, (draw, view, pool) in enumerate(cases):
        assert_share_draws(draw(view, pool=pool, n=500, seed=100 + i), 500, 12)

    # the width follows the field
    for race_no, k in zip((2, 3), RUNNERS[1:], strict=True):
        view = race_view(scored_frame, race_id_of(TEST_DAYS[1], race_no), tau_min=5)
        for pool in (WIN, PLACE):
            assert_share_draws(fitted.draws(view, pool=pool, n=50, seed=race_no), 50, k)


def test_v_final_is_positive_and_never_below_the_pool_at_t(fitted, scored_frame):
    for day in TEST_DAYS:
        for race_no in (1, 2, 3):
            for tau in GRID:
                view = race_view(scored_frame, race_id_of(day, race_no), tau_min=tau)
                for pool in (WIN, PLACE):
                    total = view.pools[pool].total
                    assert total is not None and total > 0
                    v_final = fitted.draws(view, pool=pool, n=200, seed=tau).v_final
                    assert np.isfinite(v_final).all()
                    assert (v_final >= total * (1.0 - 1e-12)).all()
    # the known nowcast: V_final = V_t / phi
    for phi in (0.25, 0.5, 0.9, 1.0):
        d = Nowcast.from_params(phi=phi, sigma=0.4).draws(
            make_view(TWELVE, total=3e7), pool=WIN, n=50, seed=1
        )
        np.testing.assert_allclose(d.v_final, 3e7 / phi, rtol=1e-12)


def test_spread_shrinks_as_phi_approaches_one():
    """The favourite's share SD at phi = 0.95 is below that at phi = 0.5; more, the SD is
    proportional to 1 - phi, since s_final - phi * s_t = (1 - phi) * s_late."""
    view = make_view(TWELVE, total=4e7, tau=5)
    sd = {}
    for phi in (0.2, 0.5, 0.8, 0.95):
        d = Nowcast.from_params(phi=phi, sigma=0.4).draws(view, pool=WIN, n=4000, seed=3)
        sd[phi] = float(d.shares[:, 0].std())
    assert sd[0.95] < sd[0.5]
    assert sd[0.95] < sd[0.8] < sd[0.5] < sd[0.2]
    per_unit = [sd[phi] / (1.0 - phi) for phi in sd]
    assert max(per_unit) / min(per_unit) < 1.1


def test_draws_at_phi_one_are_the_board():
    for pool in (WIN, PLACE):
        view = make_view(TWELVE, pool=pool, total=6e7, tau=1)
        d = Nowcast.from_params(phi=1.0, sigma=0.6).draws(view, pool=pool, n=100, seed=4)
        np.testing.assert_allclose(d.shares, np.tile(TWELVE, (100, 1)), rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(d.v_final, 6e7, rtol=1e-12)


def test_draws_are_reproducible_for_a_seed(fitted, scored_frame):
    view = race_view(scored_frame, race_id_of(TEST_DAYS[0], 1), tau_min=5)
    known = Nowcast.from_params(phi=0.7, sigma=0.4)
    for draw in (fitted.draws, fitted.random_walk, known.draws):
        a = draw(view, pool=WIN, n=200, seed=42)
        b = draw(view, pool=WIN, n=200, seed=42)
        c = draw(view, pool=WIN, n=200, seed=43)
        np.testing.assert_array_equal(a.shares, b.shares)
        np.testing.assert_array_equal(a.v_final, b.v_final)
        assert not np.array_equal(a.shares, c.shares)


def test_pit_is_uniform_on_synthetic_data():
    """2,000 races simulated from a known nowcast: the PIT of each race's final share (one runner
    per race, in turn) under that nowcast's draws is uniform, KS p > 0.01; under an overconfident
    nowcast it is not."""
    rng = np.random.default_rng(2026)
    params = [(phi, sigma) for phi in (0.3, 0.6, 0.9) for sigma in (0.3, 0.7)]
    models = {p: Nowcast.from_params(phi=p[0], sigma=p[1]) for p in params}
    overconfident = {sigma: Nowcast.from_params(phi=0.97, sigma=sigma) for sigma in (0.3, 0.7)}
    calibrated, narrow = [], []
    for i in range(2000):
        k = 8 + i % 9
        s_t = rng.dirichlet(np.full(k, 1.2))
        phi, sigma = params[i % len(params)]
        s_late = softmax(np.log(s_t) + sigma * rng.standard_normal(k))
        s_final = phi * s_t + (1.0 - phi) * s_late
        view = make_view(s_t, total=5e7, tau=5)
        j = i % k
        d = models[(phi, sigma)].draws(view, pool=WIN, n=300, seed=i)
        calibrated.append(pit(d.shares[:, j], float(s_final[j])))
        if i < 600:
            wrong = overconfident[sigma].draws(view, pool=WIN, n=300, seed=i)
            narrow.append(pit(wrong.shares[:, j], float(s_final[j])))
    assert stats.kstest(calibrated, "uniform").pvalue > 0.01
    assert stats.kstest(narrow, "uniform").pvalue < 1e-6


# ---------------------------------------------------------------- shares and the board at t


def test_late_composition_from_two_snapshots():
    got = late_composition(100.0, np.array([0.5, 0.5]), 200.0, np.array([0.6, 0.4]))
    np.testing.assert_allclose(got, [0.7, 0.3], rtol=0.0, atol=1e-12)

    # from the money: the late money's composition, whatever the field and the amounts
    rng = np.random.default_rng(1)
    for k in (2, 3, 8, 12, 18):
        early = rng.gamma(2.0, size=k) * 1e6  # money on each runner at the first snapshot
        late = rng.gamma(0.7, size=k) * 3e5  # money bet between the snapshots
        v1, v2 = early.sum(), early.sum() + late.sum()
        got = late_composition(v1, early / v1, v2, (early + late) / v2)
        np.testing.assert_allclose(got, late / late.sum(), rtol=1e-9)


def test_race_view_reads_the_board_at_t():
    day, race_no, k = TEST_DAYS[0], 5, 8
    race_id, post = race_id_of(day, race_no), post_of(day, race_no)
    rng = np.random.default_rng(32)
    minutes = (20, 10, 3, 2, 1, FINAL_MIN)
    win = {m: rng.uniform(1.5, 60.0, size=k) for m in minutes}
    place = {m: rng.dirichlet(np.full(k, 3.0)) for m in minutes}
    totals = {m: 1e6 * (i + 1) ** 2 for i, m in enumerate(minutes)}
    rows = []
    for m in minutes:
        final = m == FINAL_MIN
        lo, hi = place_odds(place[m], totals[m])
        for j in range(k):
            rows.append(
                row(race_id, race_no, WIN, str(j + 1), post, m, odds=win[m][j], final=final)
            )
            rows.append(
                row(
                    race_id,
                    race_no,
                    PLACE,
                    str(j + 1),
                    post,
                    m,
                    odds=lo[j],
                    odds_hi=hi[j],
                    final=final,
                )
            )
        rows.append(
            row(race_id, race_no, WIN, POOL_ROW, post, m, investment=totals[m], final=final)
        )
        rows.append(
            row(race_id, race_no, PLACE, POOL_ROW, post, m, investment=0.4 * totals[m], final=final)
        )
    rows += race_rows(np.random.default_rng(1), day, race_no + 1, 12)  # another race, same day
    frame = to_frame(rows)

    def win_shares(m: int) -> dict[str, float]:
        inverse = 1.0 / win[m]
        return {str(j + 1): s for j, s in enumerate(inverse / inverse.sum())}

    view = race_view(frame, race_id, tau_min=2)
    assert (view.race_id, view.venue, view.race_no) == (race_id, VENUE, race_no)
    assert view.tau_min == pytest.approx(2.0)
    board = view.pools[WIN]
    assert sorted(board.combinations, key=int) == [str(j) for j in range(1, k + 1)]
    assert_board(board.combinations, board.shares, win_shares(2))  # the board at its minute
    assert_board(board.combinations, board.shares_lag5, win_shares(10))  # post-7: the 10-min board
    assert_board(board.combinations, board.shares_lag15, win_shares(20))  # post-17: the 20-min one
    assert board.total == pytest.approx(totals[2])
    place_board = view.pools[PLACE]
    assert place_board.total == pytest.approx(0.4 * totals[2])
    assert np.asarray(place_board.shares).sum() == pytest.approx(1.0, abs=1e-9)
    assert (np.asarray(place_board.shares) > 0).all()
    favourite = str(int(np.argmin(place_odds(place[2], totals[2])[0])) + 1)
    assert place_board.combinations[int(np.argmax(place_board.shares))] == favourite

    # earlier: only the 20-minute board is up, and nothing 15 minutes before t
    early = race_view(frame, race_id, tau_min=15).pools[WIN]
    assert_board(early.combinations, early.shares, win_shares(20))
    assert_board(early.combinations, early.shares_lag5, win_shares(20))
    assert early.shares_lag15 is None
    assert early.total == pytest.approx(totals[20])

    # without pool rows the board has no total
    bare = race_view(frame[frame["combination"] != POOL_ROW], race_id, tau_min=2)
    assert bare.pools[WIN].total is None and bare.pools[PLACE].total is None
    assert_board(bare.pools[WIN].combinations, bare.pools[WIN].shares, win_shares(2))


def test_without_pool_totals_phi_comes_from_the_time_to_post_profile():
    """jravan days print the pool total on every snapshot; our own logged days only at the close
    (08 R20). A board without a total takes phi_t from the time-to-post profile fitted where totals
    exist: exp(-rate * tau) by pool here."""
    rows = []
    for i, day in enumerate(TRAIN_DAYS):
        rows += day_rows(day, totals="all" if i < 5 else "final")
    model = Nowcast.fit(to_frame(rows))
    bare = to_frame(day_rows(TEST_DAYS[0], totals="none"))
    for pool in (WIN, PLACE):
        phis = []
        for tau in GRID:
            view = race_view(bare, race_id_of(TEST_DAYS[0], 1), tau_min=tau)
            assert view.pools[pool].total is None
            phi = model.phi(view, pool=pool)
            assert phi == pytest.approx(math.exp(-GROWTH[pool] * tau), abs=0.02), (pool, tau)
            phis.append(phi)
            d = model.draws(view, pool=pool, n=200, seed=tau)
            assert_share_draws(d, 200, RUNNERS[0])
            assert np.isfinite(d.v_final).all() and (d.v_final > 0).all()
        # GRID runs from the farthest tau to the post, and the pool fills towards the post
        assert all(a < b for a, b in zip(phis, phis[1:], strict=False)), (pool, phis)


# ---------------------------------------------------------------- folds and holdout


FOLD_SCHEME = FoldScheme(
    unit="week",
    test_length=1,
    step=1,
    min_train=3,
    window="expanding",
    embargo_days=0,
    anchor_month=1,
)


def test_fit_sees_only_its_training_days():
    """Weekends across the holdout start, with the real holdout rule deciding which are locked.
    Each fold's model is the fit on that fold's training days alone: not on its test days, and not
    on any holdout day, which differ sharply so that a leak would show."""
    days = [date(2026, 10, 31) + timedelta(days=i) for i in range(50)]
    days = [d for d in days if d.weekday() >= 5]
    held = [d for d in days if is_holdout_day(COUNTRY, d)]
    dev = [d for d in days if d not in held]
    assert held and min(held) < max(dev)  # some holdout day falls inside a training span
    rows = []
    for d in days:
        if d in held:
            rows += day_rows(d, races=2, rate_scale=2.5, sigma=0.8)
        else:
            rows += day_rows(
                d,
                races=2,
                rate_scale=0.75 + 0.125 * (d.toordinal() % 5),
                sigma=0.2 + 0.05 * (d.toordinal() % 4),
            )
    frame = to_frame(rows)

    expected = [
        ([dev[i] for i in f.train_idx], [dev[i] for i in f.test_idx])
        for f in walk_forward(dev, scheme=FOLD_SCHEME)
    ]
    assert len(expected) >= 3
    fits = fit_folds(frame, scheme=FOLD_SCHEME)
    assert len(fits) == len(expected)

    probe = make_view(TWELVE, total=5e7, tau=5, lag5=TWELVE, lag15=TWELVE)
    for fit, (train, test) in zip(fits, expected, strict=True):
        assert iso(fit.model.days) == iso(train)
        assert iso(fit.test_days) == iso(test)
        assert not set(iso(fit.model.days) + iso(fit.test_days)) & set(iso(held))
        alone = Nowcast.fit(days_of(frame, train))
        leaked = Nowcast.fit(days_of(frame, train + test))
        a = fit.model.draws(probe, pool=WIN, n=300, seed=21)
        b = alone.draws(probe, pool=WIN, n=300, seed=21)
        c = leaked.draws(probe, pool=WIN, n=300, seed=21)
        np.testing.assert_allclose(a.shares, b.shares, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(a.v_final, b.v_final, rtol=1e-9)
        assert not (np.allclose(a.shares, c.shares) and np.allclose(a.v_final, c.v_final))


def test_holdout_days_are_refused(fitted):
    """D14: a holdout day never trains or scores the nowcast, by the real holdout rule."""
    start = date(2026, 11, 28)
    held = next(
        start + timedelta(days=i)
        for i in range(60)
        if is_holdout_day(COUNTRY, start + timedelta(days=i))
    )
    frame = to_frame(day_rows(TEST_DAYS[0], races=1) + day_rows(held, races=1))
    with pytest.raises(HoldoutDayError):
        Nowcast.fit(frame)
    rw = fitted.random_walk
    with pytest.raises(HoldoutDayError):
        compare(rw, rw, frame, n=20, seed=1, n_boot=20, level=0.95)


# ---------------------------------------------------------------- scores


def test_random_walk_baseline_scores_zero_gain_on_itself(fitted, scored_frame):
    rw = fitted.random_walk
    table = compare(rw, rw, scored_frame, n=200, seed=5, n_boot=200, level=0.95)
    keys = set(zip(table["pool"], table["tau_min"].astype(int), strict=True))
    assert keys == {(pool, tau) for pool in (WIN, PLACE) for tau in GRID}
    assert len(table) == len(keys)
    for column in GAIN_COLUMNS:
        np.testing.assert_allclose(table[column].to_numpy(float), 0.0, rtol=0.0, atol=1e-12)

    # the same scoring tells a better forecast from the random walk, with the sign of the gain
    oracle = make_oracle(final_win_shares(scored_frame), rw)
    ahead = compare(oracle, rw, scored_frame, n=200, seed=5, n_boot=200, level=0.95)
    behind = compare(rw, oracle, scored_frame, n=200, seed=5, n_boot=200, level=0.95)
    for result, sign in ((ahead, 1.0), (behind, -1.0)):
        win, place = result[result["pool"] == WIN], result[result["pool"] == PLACE]
        assert sorted(win["tau_min"].astype(int)) == sorted(GRID)
        for gain in GAINS:
            assert (sign * win[gain] > 0).all(), (gain, sign)
            assert (win[f"{gain}_lo"] <= win[gain] + 1e-12).all()
            assert (win[gain] <= win[f"{gain}_hi"] + 1e-12).all()
            np.testing.assert_allclose(place[gain].to_numpy(float), 0.0, rtol=0.0, atol=1e-12)


def test_compare_resamples_race_days(fitted, scored_frame):
    """The interval is clustered by race day: over one day it collapses onto the mean; over
    several it does not."""
    rw = fitted.random_walk
    oracle = make_oracle(final_win_shares(scored_frame), rw)
    one_day = compare(
        oracle, rw, days_of(scored_frame, TEST_DAYS[:1]), n=100, seed=9, n_boot=300, level=0.95
    )
    win = one_day[one_day["pool"] == WIN]
    for gain in GAINS:
        np.testing.assert_allclose(
            win[f"{gain}_lo"].to_numpy(float), win[gain].to_numpy(float), rtol=1e-9, atol=1e-12
        )
        np.testing.assert_allclose(
            win[f"{gain}_hi"].to_numpy(float), win[gain].to_numpy(float), rtol=1e-9, atol=1e-12
        )
    several = compare(oracle, rw, scored_frame, n=100, seed=9, n_boot=300, level=0.95)
    win = several[several["pool"] == WIN]
    assert (win["log_score_gain_hi"] - win["log_score_gain_lo"] > 0).any()
