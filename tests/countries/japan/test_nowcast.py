"""The C1 final-price nowcast v0 for the Win and Place pools (ticket #85).

Module under test: `horsetrack.countries.japan.models.nowcast`, not the drift nowcast of #116 in
`models/v0/nowcast.py`. Contract pinned here:

Shares and the late composition
- A combination's share is its money over the pool. A WIN board's shares are 1/O_i normalised
  over the priced runners. PLACE shares come from the per-runner money or the low odds by a rule
  left to the module; here they only have to sum to 1, with the largest share on the runner with
  the lowest low odds.
- Sound Place shares (#410): `pool_shares(PLACE, odds, investment=None, *, odds_hi=None)` returns
  (shares, rule_sum). Per-runner money on every runner (finite, positive) is used as-is. Otherwise
  the shares come from the printed figures (`odds` the low, `odds_hi` the high) and must recover
  the money of a board JRA prints from known money (`jra_place_figures` below: JRA's
  stake-back-share dividend, `core.pricing.stake_back_share_dividend`, over every placed set,
  truncated to 10 yen, floored at the minimum dividend), within PLACE_ABS_TOL and
  PLACE_REL_TOL, including boards whose favourite's low figure sits at the floor. Without high
  figures the low figures alone are used (owner, #410). The rule inside is left to the module.
  WIN is unaffected.
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
- A late scratch (owner's #69 rule, core/eval/scoring.py: scratched runners are removed and the
  probabilities renormalised over the starters). The store writes a scratched runner's rows with
  no price (null odds, the raw text in `status_raw`; jravan/odds.py), so a runner priced on the
  board at t whose latest row at the close has no price is not on the final board. That case is
  kept and scored at its (pool, tau): every forecaster gets the board at t, the scratched runner
  included, and its drawn shares are restricted to the combinations on the final board and
  renormalised to sum to 1. Scored that way, a forecaster whose draws over the starters do not
  depend on the scratched runner scores what it scores on the same race without that runner.
  `extract(frame)`'s counts report the renormalised cases (one count, expected `renormalised`).

The study: `studies/nowcast.py`, `horsetrack japan nowcast-study`
- `wide_coverage()` lists our own logged days, `processed/jra/odds_timeseries/<YYYY-MM-DD>.parquet`,
  under `own_logged_days` with `day`, `wide_races` and `wide_races_with_post_time`. D14: a holdout
  day (`is_holdout_day`) is fully locked, so its file is never opened or read and the coverage has
  no entry for it; every development day is listed.
- `run_study(trial=True)` that scores folds still appends registry records (CLAUDE.md: every run,
  failures included, is a registry record), each marked as a trial; a full run's are not. The
  registry has no trial status, so a marker is "trial" in the family id or the variant,
  `params["trial"] is True`, or a params value "trial".

Expected values are derived here from first principles (money, the stated formulas, scipy), never
from the module under test. All data are synthetic and in memory or under tmp, and
HORSETRACK_DATA_DIR points at a tmp dir.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import itertools
import math
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from scipy import stats

from horsetrack.core import config, paths
from horsetrack.core.eval.distributional import pit
from horsetrack.core.eval.folds import FoldScheme, walk_forward
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import default_registry
from horsetrack.core.pricing import stake_back_share_dividend
from horsetrack.countries.japan.board.snapshot import PLACE, WIDE, WIN
from horsetrack.countries.japan.logger.compact import SCHEMA
from horsetrack.countries.japan.models.nowcast import (
    Draws,
    HoldoutDayError,
    Nowcast,
    PoolBoard,
    RaceView,
    compare,
    extract,
    fit_folds,
    late_composition,
    pool_shares,
    race_view,
    score_cases,
)
from horsetrack.countries.japan.studies import nowcast as study

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
    """Low and high PLACE odds under JRA's stake-back-share rule with three places
    (`core.pricing.stake_back_share_dividend`, untruncated): placed with two others, a runner pays
    rate x (1 + D / (3 s_i)), D the share on the runners outside the three, rate = 1 - takeout,
    at least 1.1. Low odds: placed with the two most-backed others; high odds: with the two
    least-backed. `pool_total` only scales the money, which the rule does not see."""
    money = np.asarray(shares, dtype=float) * pool_total
    share = money / money.sum()
    rate = 1.0 - TAKEOUT
    lo, hi = np.empty(len(share)), np.empty(len(share))
    for i, own in enumerate(share):
        others = np.sort(np.delete(share, i))
        lo[i] = rate * (1.0 + (1.0 - own - others[-2:].sum()) / (3.0 * own))
        hi[i] = rate * (1.0 + (1.0 - own - others[:2].sum()) / (3.0 * own))
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


# ---------------------------------------------------------------- late scratches


SCRATCHED = "7"  # the runner scratched in the late-scratch tests; every synthetic field has one
K_MAX = max(RUNNERS)


def scratch(rows: list[dict], race_id: str, *, from_minute: float) -> list[dict]:
    """`rows` with runner SCRATCHED of `race_id` scratched `from_minute` before the post: its WIN
    and PLACE rows from then on, the final capture included, carry no price, as the store writes a
    scratched runner (null odds, the raw text in status_raw)."""
    out = []
    for r in rows:
        if (
            r["race_id"] == race_id
            and r["pool"] in (WIN, PLACE)
            and r["combination"] == SCRATCHED
            and r["minutes_to_post"] <= from_minute
        ):
            r = {**r, "odds": None, "odds_hi": None, "status_raw": "----"}
        out.append(r)
    return out


def without_runner(rows: list[dict], race_id: str) -> list[dict]:
    """`rows` as if runner SCRATCHED of `race_id` had never been entered."""
    return [
        r
        for r in rows
        if not (
            r["race_id"] == race_id and r["pool"] in (WIN, PLACE) and r["combination"] == SCRATCHED
        )
    ]


def final_shares(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    """Each (race, pool)'s 1/O shares over the combinations priced at the close: the final WIN
    shares; for PLACE only a forecast's centre (from the low figures)."""
    rows = frame[
        frame["is_final_capture"]
        & frame["pool"].isin([WIN, PLACE])
        & (frame["combination"] != POOL_ROW)
        & (frame["odds"] > 0)
    ]
    out = {}
    for (race_id, pool), g in rows.groupby(["race_id", "pool"]):
        inverse = 1.0 / g["odds"].to_numpy(float)
        out[(race_id, pool)] = dict(zip(g["combination"], inverse / inverse.sum(), strict=True))
    return out


def starter_forecaster(weight, scratched_weight: float, sigma: float, races, seen=None):
    """Draws softmax(ln v + sigma eps) over the board's combinations: v = weight(race, pool, c),
    except `scratched_weight` for runner SCRATCHED of a race in `races`, and eps by runner number
    (a runner's noise is the same whatever the field). Restricted to the starters and renormalised,
    the draws are softmax(ln v + sigma eps) over the starters alone, whatever `scratched_weight`:
    exactly the draws it makes on the race without the scratched runner. `seen` collects each
    view's (race, pool, tau, combinations)."""

    def forecast(view, *, pool, n, seed):
        combinations = view.pools[pool].combinations
        if seen is not None:
            seen.append((view.race_id, pool, float(view.tau_min), combinations))
        eps = np.random.default_rng(seed).standard_normal((n, K_MAX))
        eps = eps[:, [int(c) - 1 for c in combinations]]
        v = np.array(
            [
                scratched_weight
                if c == SCRATCHED and view.race_id in races
                else weight(view.race_id, pool, c)
                for c in combinations
            ]
        )
        return Draws(shares=softmax(np.log(v) + sigma * eps), v_final=np.full(n, 1e8))

    return forecast


def assert_same_gains(got: pd.DataFrame, expected: pd.DataFrame) -> None:
    key = ["pool", "tau_min"]
    got = got.sort_values(key).reset_index(drop=True)
    expected = expected.sort_values(key).reset_index(drop=True)
    assert list(zip(got["pool"], got["tau_min"], strict=True)) == list(
        zip(expected["pool"], expected["tau_min"], strict=True)
    )
    for column in GAIN_COLUMNS:
        np.testing.assert_allclose(
            got[column].to_numpy(float),
            expected[column].to_numpy(float),
            rtol=1e-9,
            atol=1e-10,
            err_msg=column,
        )


def test_a_late_scratch_is_scored_on_the_starters_renormalised():
    """#69: a runner priced at t and scratched by the close is removed and the draws renormalised
    over the starters, for every forecaster alike; the case is scored, not dropped. Race 1 of the
    day is scratched after T-1, so its board changes by the close at every tau; race 2 from T-21,
    so at T-60 and T-30 only (from T-16 the board at t no longer prices it)."""
    late, early = race_id_of(TEST_DAYS[0], 1), race_id_of(TEST_DAYS[0], 2)
    rows = day_rows(TEST_DAYS[0]) + day_rows(TEST_DAYS[1])
    scratched = to_frame(scratch(scratch(rows, late, from_minute=0), early, from_minute=21))
    clean = to_frame(without_runner(without_runner(rows, late), early))
    races = {late, early}
    truth = final_shares(clean)
    assert set(truth[(late, WIN)]) == {str(j) for j in range(1, RUNNERS[0] + 1)} - {SCRATCHED}

    def informed(race_id, pool, c):
        return truth[(race_id, pool)][c]

    def flat(race_id, pool, c):
        return 1.0

    def pair(seen=None):
        # The candidate is centred on the final shares, the baseline is flat. The weight on the
        # scratched runner differs between them, so it cancels only once the draws are restricted
        # to the starters and renormalised.
        return (
            starter_forecaster(informed, 0.25, 0.1, races, seen),
            starter_forecaster(flat, 3.0, 0.3, races),
        )

    kw = {"n": 200, "seed": 11, "n_boot": 200, "level": 0.95}

    # The late-scratched race alone: a scored case at every (pool, tau), on the board at t.
    seen = []
    alone = compare(*pair(seen), scratched[scratched["race_id"] == late], **kw)
    views = [(pool, int(tau), combos) for race, pool, tau, combos in seen if race == late]
    assert {(pool, tau) for pool, tau, _ in views} == {(p, t) for p in (WIN, PLACE) for t in GRID}
    assert all(SCRATCHED in combos for _, _, combos in views)  # the board at t still prices it
    assert len(alone) == 2 * len(GRID)
    assert_same_gains(alone, compare(*pair(), clean[clean["race_id"] == late], **kw))

    # Two days, both scratches and unchanged races: the gains of the same races without the runner.
    reference = compare(*pair(), clean, **kw)
    assert_same_gains(compare(*pair(), scratched, **kw), reference)
    win = reference[reference["pool"] == WIN]
    assert (win["log_score_gain"] > 0).all()  # not a vacuous equality: the candidate is better

    # The study's scoring path (extract, then score_cases): per case and forecaster, the scores of
    # the same race without the runner.
    def per_case(frame):
        cases, _, _ = extract(frame)
        forecasters = dict(zip(("candidate", "baseline"), pair(), strict=True))
        table, _ = score_cases(forecasters, cases, n=200, seed=11)
        keys = ["race_id", "pool", "tau_min", "forecaster"]
        return table.set_index(keys)[["log_score", "crps"]].sort_index()

    got, expected = per_case(scratched), per_case(clean)
    assert got.index.equals(expected.index)
    np.testing.assert_allclose(got.to_numpy(float), expected.to_numpy(float), rtol=1e-9, atol=1e-10)

    # The renormalised cases are counted: race 1 at every (pool, tau), race 2 where t precedes
    # the scratch (the latest snapshot at or before t is earlier than T-21).
    early_taus = sum(1 for tau in GRID if min(m for m in MINUTES if m >= tau) > 21)
    assert early_taus == 2
    _, _, counts = extract(scratched)
    named = {k: v for k, v in counts.items() if "renormal" in k or "scratch" in k}
    assert 2 * len(GRID) + 2 * early_taus in named.values(), counts


# ---------------------------------------------------------------- the study


def own_logged_rows(day: date, wide_posts: Sequence[bool]) -> list[dict]:
    """One of our own logged days: a Wide race per entry of `wide_posts` (with a post time or
    not), each with two pairs and a pool row, and a race selling only WIN."""
    rows = []
    for race_no, has_post in enumerate(wide_posts, start=1):
        race_id, post = race_id_of(day, race_no), post_of(day, race_no)
        for combination in ("1,2", "1,3", POOL_ROW):
            r = row(
                race_id,
                race_no,
                WIDE,
                combination,
                post,
                10,
                odds=None if combination == POOL_ROW else 3.5,
                investment=2e7 if combination == POOL_ROW else None,
            )
            if not has_post:
                r = {**r, "post_utc": None, "minutes_to_post": None}
            rows.append(r)
    race_no = len(wide_posts) + 1
    rows.append(
        row(race_id_of(day, race_no), race_no, WIN, "1", post_of(day, race_no), 10, odds=2.0)
    )
    return rows


def write_day(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)


def test_wide_coverage_never_reads_a_holdout_day(data_dir, monkeypatch):
    """D14: logged holdout days are untouched. One holdout day's file is a valid logged day with
    Wide races, the other is not parquet, so any read of it raises; every parquet read is watched.
    The development days around them, two after the holdout start, are listed with their counts."""
    held = (date(2026, 11, 29), date(2026, 12, 1))
    dev = {
        date(2026, 11, 22): (True,),
        date(2026, 11, 28): (True, False, True),
        date(2026, 11, 30): (False, True),
    }
    assert all(is_holdout_day(COUNTRY, d) for d in held)
    assert not any(is_holdout_day(COUNTRY, d) for d in dev)
    folder = data_dir / "processed" / "jra" / "odds_timeseries"
    for day, posts in dev.items():
        write_day(folder / f"{day.isoformat()}.parquet", own_logged_rows(day, posts))
    write_day(folder / f"{held[0].isoformat()}.parquet", own_logged_rows(held[0], (True, True)))
    (folder / f"{held[1].isoformat()}.parquet").write_bytes(b"a locked holdout day\n")

    opened = []
    for module, name in ((pq, "read_table"), (pq, "ParquetFile"), (pq, "read_metadata"),
                         (pq, "read_schema"), (ds, "dataset")):  # fmt: skip
        real = getattr(module, name)

        def spy(source, *args, _real=real, **kwargs):
            opened.append(str(source))
            return _real(source, *args, **kwargs)

        monkeypatch.setattr(module, name, spy)

    try:
        coverage = study.wide_coverage()
    except Exception as exc:  # the second holdout file is not parquet: reading it raises
        pytest.fail(f"wide_coverage read a holdout day's file: {exc!r}")
    touched = [p for p in opened if any(d.isoformat() in p for d in held)]
    assert not touched, touched
    listed = {entry["day"]: entry for entry in coverage["own_logged_days"]}
    assert set(listed) == {d.isoformat() for d in dev}
    for day, posts in dev.items():
        entry = listed[day.isoformat()]
        assert (entry["wide_races"], entry["wide_races_with_post_time"]) == (len(posts), sum(posts))


def write_jravan_store(data_dir: Path, days, *, scratched: Sequence[str] = ()) -> None:
    """The jravan store the study reads: an odds_timeseries day file per day (two races each, with
    runner SCRATCHED of each race in `scratched` scratched at the close) and races.parquet."""
    race_ids = []
    for day in days:
        rows = day_rows(day, races=2)
        for race_id in scratched:
            rows = scratch(rows, race_id, from_minute=0)
        write_day(data_dir / "processed" / "jravan" / "odds_timeseries" / f"{day}.parquet", rows)
        race_ids += sorted({r["race_id"] for r in rows})
    races = pa.table(
        {
            "race_id": race_ids,
            "race_class": ["G3" if i % 3 == 0 else None for i in range(len(race_ids))],
            "race_date": [date.fromisoformat(race_day(r)) for r in race_ids],
        }
    )
    pq.write_table(races, data_dir / "processed" / "jravan" / "races.parquet")


def marked_trial(record) -> bool:
    """Whether a registry record says it is a trial. The registry has no trial status; expected:
    `params["trial"] is True` or a trial suffix on the variant."""
    if "trial" in record.family_id.lower() or "trial" in record.variant.lower():
        return True
    return any(
        (str(key).lower() == "trial" and value is True)
        or (isinstance(value, str) and value.lower() == "trial")
        for key, value in record.params.items()
    )


def test_a_trial_run_is_a_registry_record_marked_as_a_trial(data_dir, tmp_path, monkeypatch):
    """CLAUDE.md: every run, failures included, is a registry record. A --trial run that scores
    folds appends records marked as a trial; the full run after it appends unmarked ones. One
    fold (FOLD_SCHEME: weekly, three weeks' training) scores the last day; a late scratch on a
    training day and on the scored day keeps the run on the renormalised path."""
    late = (race_id_of(TRAIN_DAYS[2], 1), race_id_of(TEST_DAYS[0], 2))
    write_jravan_store(data_dir, TRAIN_DAYS + TEST_DAYS[:1], scratched=late)
    monkeypatch.setattr(study, "load_country_scheme", lambda country: FOLD_SCHEME)
    summary = tmp_path / "summary"  # the run regenerates the registry summary: not the repo's
    monkeypatch.setattr(study, "summary_dir", lambda country: summary)
    monkeypatch.setattr("horsetrack.core.eval.registry.summary_dir", lambda country: summary)
    kw = {
        "test_days_per_fold": 1,
        "n_draws": 30,
        "out_dir": tmp_path / "out",
        "log": lambda *a: None,
    }

    study.run_study(trial=True, **kw)
    registry = default_registry(COUNTRY)
    trial = registry.records()
    assert trial, "a --trial run appended no registry record"
    assert all(marked_trial(r) for r in trial), [(r.family_id, r.variant, r.params) for r in trial]

    study.run_study(trial=False, **kw)
    full = registry.records()[len(trial) :]
    assert full
    assert not any(marked_trial(r) for r in full), [(r.variant, r.params) for r in full]


# ---------------------------------------------------------------- sound Place shares (#410)

JRA = config.load_source(COUNTRY, "jra")
PLACE_POOL = JRA.pools[PLACE]
PLACE_RATE = 1.0 - float(PLACE_POOL.takeout)
PLACE_UNIT = float(JRA.unit_stake)  # dividends per 100 yen, in 10-yen steps: 0.1 in odds

# The tolerance comes from that truncation. Placed in a set S, a runner's figure f is
# rate (1 + D / (P s)), so its share s solves s (P (f / rate - 1) + 1) = 1 - T, T the share of the
# other runners in S. A figure truncated by up to 0.1 moves the share it implies by up to
# 0.1 P / (P f - (P - 1) rate) of itself: 6.8% at f = 2.0 with three places, 6.3% with two (rate
# 0.8), and less from a higher figure. Every board below prints a high figure of 2.0 or more on
# every runner but the most-backed (`assert_readable`), whose share is what the others leave of
# the pool, so its error is the others' errors summed, capped at a point of the pool; 7.5% leaves
# the T term its room. A least-squares fit of both figures under this rule stays under 2.5% and
# 0.004 of the pool on every board here; the low figures alone miss by 17-135% on the boards
# whose favourite is at the floor.
PLACE_REL_TOL = 0.075  # per runner: |share - money share| / money share
PLACE_ABS_TOL = 0.01  # per runner, in units of the pool


def jra_place_figures(money_yen: Sequence[int]) -> tuple[np.ndarray, np.ndarray, int]:
    """(low, high, places): JRA's printed Place figures for per-runner money in yen. For every
    placed set S of P runners (P = `places_for(starters)`), each runner i in S is paid JRA's
    stake-back-share dividend, `core.pricing.stake_back_share_dividend` (pinned by
    tests/core/test_pricing.py): 100 x rate x (1 + D / (P x s_i)) yen per 100-yen ticket, s_i its
    share of the pool, D the share on the runners outside S, rate = 1 - takeout; truncated down to
    10 yen and paid at least the minimum dividend. The low figure is the least of runner i's
    payments over the sets holding it, the high the greatest."""
    money = [int(m) for m in money_yen]
    places = PLACE_POOL.places_for(len(money))
    assert places >= 2, len(money)
    pool = float(sum(money))
    lo = [math.inf] * len(money)
    hi = [-math.inf] * len(money)
    for placed in itertools.combinations(range(len(money)), places):
        for i in placed:
            paid = stake_back_share_dividend(
                money[i] / pool,
                [money[j] / pool for j in placed if j != i],
                rate=PLACE_RATE,
                unit=PLACE_UNIT,
                dividend_step=float(PLACE_POOL.dividend_step),
                min_dividend=float(PLACE_POOL.min_dividend),
            )
            lo[i], hi[i] = min(lo[i], paid), max(hi[i], paid)
    return np.array(lo) / PLACE_UNIT, np.array(hi) / PLACE_UNIT, places


def assert_readable(money_yen, hi) -> None:
    """The boards' precondition for the tolerance: high figures of 2.0 or more but on the
    most-backed runner."""
    others = np.delete(np.asarray(hi), int(np.argmax(money_yen)))
    assert others.min() >= 2.0, others


def random_place_boards(n: int, seed: int = 410) -> list[tuple[str, np.ndarray]]:
    """Seeded boards of 5 to 16 runners with one favourite holding 25-55% of the Place money,
    in 100-yen tickets on a pool of about 80 million yen; those the tolerance covers."""
    rng = np.random.default_rng(seed)
    boards = []
    while len(boards) < n:
        k = int(rng.integers(8, 17)) if len(boards) % 3 else int(rng.integers(5, 8))
        rest = np.sort(rng.dirichlet(np.full(k - 1, 1.5)))[::-1]
        favourite = rng.uniform(0.25, 0.55)
        weights = np.concatenate([[favourite], (1.0 - favourite) * rest])
        money = (np.maximum(np.round(weights * 8e7 / 100), 1) * 100).astype(np.int64)
        money = rng.permutation(money)  # the favourite is not always runner 1
        others = np.delete(jra_place_figures(money)[1], int(np.argmax(money)))
        if others.min() >= 2.0:
            boards.append((f"random-{len(boards)}-{k}-runners", money))
    return boards


def _yen(values: Sequence[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64) * 1000


# (id, money in yen, how many low figures sit at the floor; the favourite's is the first)
PLACE_BOARDS = [
    (
        "14-runners-favourite-at-the-floor",
        _yen([4200, 900, 700, 600, 500, 450, 400, 350, 300, 250, 200, 150, 120, 80]),
        1,
    ),
    (
        "12-runners-two-lows-at-the-floor",
        _yen([1700, 6300, 250, 230, 220, 210, 200, 190, 190, 180, 170, 160]),
        2,
    ),
    (
        "16-runners-open-race",
        _yen([900, 850, 800, 700, 650, 600, 600, 550, 500, 450, 400, 350, 300, 300, 250, 200]),
        0,
    ),
    ("8-runners-heavy-favourite", _yen([1200, 1000, 3000, 900, 700, 500, 300, 200]), 1),
    ("6-runners-pay-two-places", _yen([2000, 3000, 1800, 1300, 1000, 900]), 0),
    *((name, money, None) for name, money in random_place_boards(9)),
]


def assert_recovers(shares, money_yen) -> None:
    truth = np.asarray(money_yen, dtype=float) / float(np.sum(money_yen))
    assert shares is not None
    got = np.asarray(shares, dtype=float)
    assert got.shape == truth.shape
    assert np.isfinite(got).all() and (got > 0).all(), got
    assert got.sum() == pytest.approx(1.0, abs=1e-9)
    abs_err = np.abs(got - truth)
    rel_err = abs_err / truth
    worst = int(np.argmax(rel_err))
    assert rel_err.max() <= PLACE_REL_TOL, (
        f"runner {worst + 1}: share {got[worst]:.4f} vs money {truth[worst]:.4f}"
    )
    assert abs_err.max() <= PLACE_ABS_TOL, abs_err


@pytest.mark.parametrize(
    ("money", "at_floor"), [(m, f) for _, m, f in PLACE_BOARDS], ids=[b[0] for b in PLACE_BOARDS]
)
def test_place_shares_recover_the_money_on_a_known_board(money, at_floor):
    """DoD #410: Place shares from the printed low and high figures total 1 and match the money
    behind them within PLACE_REL_TOL and PLACE_ABS_TOL, including the boards where a heavily
    backed favourite's low figure is at the floor and the low figures alone fail."""
    lo, hi, _ = jra_place_figures(money)
    assert_readable(money, hi)
    favourite = int(np.argmax(money))
    if at_floor is not None:  # the named boards are what they say
        floor = PLACE_POOL.min_dividend / PLACE_UNIT
        assert int(np.sum(lo == floor)) == at_floor, lo
        assert bool(lo[favourite] == floor) is (at_floor > 0), lo
    shares, _ = pool_shares(PLACE, lo, odds_hi=hi)
    assert_recovers(shares, money)
    # the same board, priced from the figures, whatever partial money the log carries
    partial = np.asarray(money, dtype=float)
    partial[favourite], partial[(favourite + 1) % len(partial)] = np.nan, 0.0
    assert_recovers(pool_shares(PLACE, lo, partial, odds_hi=hi)[0], money)


def test_place_shares_from_the_board_at_t_recover_the_money():
    """`race_view` reads the Place board at t and at t - 5 minutes from the logged low and high
    figures (no per-runner money in the rows, as on JRA-VAN boards), and both boards' shares match
    the money behind them."""
    day, race_no = TEST_DAYS[0], 4
    race_id, post = race_id_of(day, race_no), post_of(day, race_no)
    boards = {  # 12 runners each; the favourite changes between the two minutes
        10: _yen([1100, 5500, 800, 600, 450, 400, 300, 250, 200, 150, 150, 100]),
        5: _yen([700, 900, 650, 4300, 500, 450, 400, 300, 250, 200, 150, 120]),
        FINAL_MIN: _yen([800, 1000, 600, 4600, 500, 400, 400, 300, 250, 200, 150, 100]),
    }
    rows = []
    for minutes, money in boards.items():
        final = minutes == FINAL_MIN
        lo, hi, _ = jra_place_figures(money)
        assert_readable(money, hi)
        for j in range(len(money)):
            rows.append(
                row(
                    race_id,
                    race_no,
                    PLACE,
                    str(j + 1),
                    post,
                    minutes,
                    odds=lo[j],
                    odds_hi=hi[j],
                    final=final,
                )
            )
            rows.append(
                row(race_id, race_no, WIN, str(j + 1), post, minutes, odds=2.0 + j, final=final)
            )
        total = float(np.sum(money))
        rows.append(
            row(race_id, race_no, PLACE, POOL_ROW, post, minutes, investment=total, final=final)
        )
    view = race_view(to_frame(rows), race_id, tau_min=5)
    board = view.pools[PLACE]
    order = np.array([int(c) - 1 for c in board.combinations])
    assert sorted(order.tolist()) == list(range(12))
    assert board.total == pytest.approx(float(np.sum(boards[5])))
    assert_recovers(board.shares, boards[5][order])
    assert_recovers(board.shares_lag5, boards[10][order])


def test_place_money_on_every_runner_is_used_as_is_and_win_is_unaffected():
    """Per-runner money, finite and positive on every runner, is the shares as it is, with or
    without the figures; WIN shares stay 1/O normalised."""
    rng = np.random.default_rng(4101)
    for _, money, _ in PLACE_BOARDS[:6]:
        lo, hi, _ = jra_place_figures(money)
        noisy = np.asarray(money, dtype=float) * rng.uniform(0.5, 1.5, size=len(money))
        expected = noisy / noisy.sum()  # not the figures' money: the money given wins
        for shares in (
            pool_shares(PLACE, lo, noisy)[0],
            pool_shares(PLACE, lo, noisy, odds_hi=hi)[0],
            pool_shares(PLACE, lo, investment=noisy, odds_hi=hi)[0],
        ):
            np.testing.assert_allclose(shares, expected, rtol=1e-12, atol=0.0)
        win = rng.uniform(1.2, 150.0, size=len(money))
        shares, _ = pool_shares(WIN, win)
        np.testing.assert_allclose(shares, (1.0 / win) / (1.0 / win).sum(), rtol=1e-12, atol=0.0)


def test_place_shares_without_high_figures_fall_back_to_the_low_figures():
    """Owner (#410): without high figures, Place shares come from the low figures alone. They
    still total 1, put the largest share on the lowest low figure, and an all-missing high column
    reads as none."""
    for _, money, _ in PLACE_BOARDS[2:6]:
        lo, _, _ = jra_place_figures(money)
        bare, _ = pool_shares(PLACE, lo)
        missing, _ = pool_shares(PLACE, lo, odds_hi=np.full(len(lo), np.nan))
        for shares in (bare, missing):
            shares = np.asarray(shares, dtype=float)
            assert shares.sum() == pytest.approx(1.0, abs=1e-9)
            assert (shares > 0).all()
            assert int(np.argmax(shares)) == int(np.argmin(lo))
        np.testing.assert_allclose(missing, bare, rtol=1e-12, atol=0.0)
