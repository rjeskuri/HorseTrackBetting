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
- (#411) `run_study` reads our own compacted logged days (`processed/jra/odds_timeseries`) beside
  the JRA-VAN day files: a development logged day with post times enters the folds exactly as the
  same rows in a JRA-VAN file would. A race in both sources is read once per pool: the JRA-VAN
  rows where JRA-VAN has the pool (WIN, PLACE), our log's for the pools it alone has (Wide). A
  logged holdout day is skipped by its file name, before any read (D14).
- Wide (D3) is fitted and scored where the folds' training days hold Wide paths with post times.
  The report's JSON (`<report>.json`, beside the rendered `.md`) carries `wide["fitted"]` (bool);
  when False, `wide["reason"]` is a sentence, printed in the rendered report, that names why (it
  differs between "no post time" and "no Wide on any training day"). `own_logged_days` stays.

#411 review (the fixes' contract)
- Wide shares are pair-aware. JRA's Wide rule (config/jra.yaml `stake_back_share`): the three
  pairs among the first three win, each paid by the shared identity
  `core.pricing.stake_back_share_dividend`, rate x (1 + D / (3 m)), D the losers' money (施行規則
  付録第六, (W + D/P) x R: the rate covers the returned stake too). The pair {a,b} wins with
  {a,x} and {b,x} for one third runner x, so its low figure uses max over x of (m_ax + m_bx)
  and its high figure the min. The fixtures' Wide figures (`wide_figures`) are that identity by
  brute force over every feasible top three, without breakage and floored only where stated.
  `nc.wide_odds(shares, combinations, payout) -> (lo, hi)` is that forward rule and
  `nc.wide_money(lo, combinations, payout) -> money` (pool units) its inverse, rows of 2-D input
  being boards or draws, `combinations` the pairs as "a,b" text in column order.
  `nc.pool_shares(WIDE, odds, combinations=...)` uses the inverse
  (`rule_sum` its money total); the C1 draws (`ShadowNowcast`) and the drift forecaster
  (`studies.nowcast.drift_forecaster`) print and read Wide figures by it. WIN and PLACE keep
  their rules.
- Our logged targets need a confirmed final capture: the logger's final-state marker
  (docs/specs/board-at-t.md: `source_update_utc` null and 最終オッズ in `page_state`). A logged
  (race, pool) whose capture stopped before a final capture gives no case and no growth
  observation, and is counted (`counts["no_final_capture"]`, 0 when every logged path is final).
  Pinned end to end: the logged days here are written as odds logs and compacted by
  `logger.compact.compact_day`, so how the marker is carried is the module's choice.
- A runner scratched in the logged WIN/PLACE rows (status_raw "SCR", no price) at capture t
  removes its Wide pairs from the Wide board from t on, the final board included; JRA's log
  simply stops listing those pairs. t is the fetched_utc of the row reporting the scratch (kept
  through compaction and `read_logged_day`), its snapshot_utc where it has none: a round that
  starts before a grid cutoff and fetches the scratch after it leaves the pairs on that board.
- A fold whose test days give only unscorable cases completes with its `unscored_no_fit` kept,
  and so does a run in which no case at all is scored (its report still renders).
- The Wide reason tracks confirmed finals apart from usable grid boards and names each missing
  input: a missing final capture with "final capture", missing boards before the post with
  "pre-post board" or "board(s) before the post"; a store whose Wide paths all have a confirmed
  final does not point to recompaction.
- A run that reads our logged days registers under `japan-c1-nowcast-logged-wide-v1`, its spec
  naming both sources, and writes its reports as `nowcast-japan-logged-wide-v1-<spec hash[:12]>`;
  a JRA-VAN-only run keeps v2, its spec hash and its report.
- Each (race, pool) path is timed by its own source's post time. JRA-VAN wins a (race, pool) only
  where its path has a post time. `wide_coverage` reads the logged days the folds read (`--end`).

Expected values are derived here from first principles (money, the stated formulas, scipy), never
from the module under test. All data are synthetic and in memory or under tmp, and
HORSETRACK_DATA_DIR points at a tmp dir.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import hashlib
import itertools
import json
import math
import re
from collections import defaultdict
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
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board.snapshot import (
    FINAL_TEXT,
    PLACE,
    WIDE,
    WIN,
    RaceBoard,
    load_decision_grid,
)
from horsetrack.countries.japan.logger.compact import SCHEMA, compact_day
from horsetrack.countries.japan.models import nowcast as nc
from horsetrack.countries.japan.models.nowcast import (
    Draws,
    HoldoutDayError,
    Nowcast,
    PoolBoard,
    RaceView,
    ShadowNowcast,
    compare,
    extract,
    fit_folds,
    late_composition,
    race_view,
    score_cases,
)
from horsetrack.countries.japan.models.v0.shadow_values import PoolDraws
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


# ---------------------------------------------------------------- our own logged days (#411)

WIDE_TAKEOUT = 0.225  # config/jra.yaml
WIDE_FLOOR = 1.1  # min_dividend 110
WIDE_RUNNERS = 6
WIDE_PAIRS = tuple(
    f"{a},{b}" for a in range(1, WIDE_RUNNERS + 1) for b in range(a + 1, WIDE_RUNNERS + 1)
)
JRAVAN_DAYS = TRAIN_DAYS + (date(2026, 10, 31), date(2026, 11, 1))  # one fold on their own
STUDY_DAYS = JRAVAN_DAYS + TEST_DAYS  # three folds
LATE_ENTRY = "99"  # a WIN runner our log prices only at the close (see logged_copy)


WIDE_PAYOUT = 1.0 - WIDE_TAKEOUT
# No rounding in the fixtures' figures: core.pricing's dividend truncation step, per 1 yen, set
# far below any tolerance here (relative error under 1e-11 on figures of 1 to 1e3).
NO_BREAKAGE = 1e-12


def jra_dividend(own: float, co_winners: Sequence[float], *, floor: float | None = None) -> float:
    """JRA's stake-back-share dividend per 1 yen (施行規則 付録第六, (W + D/P) x R; HKTrack
    B-betting-rules.md §1.5): `core.pricing.stake_back_share_dividend` itself, with the Wide rate,
    no breakage and `floor` as the minimum dividend."""
    return stake_back_share_dividend(
        float(own),
        [float(c) for c in co_winners],
        rate=WIDE_PAYOUT,
        unit=1.0,
        dividend_step=NO_BREAKAGE,
        min_dividend=floor,
    )


def wide_figures(
    money: np.ndarray, pairs: Sequence[str] = WIDE_PAIRS, *, floor: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Low and high Wide figures of pair money `money` (units of the pool, summing to 1) by JRA's
    rule, found by brute force over every feasible result: for each set of three runners finishing
    first to third, the three pairs among them win, and each is paid by the shared stake-back-share
    identity (`jra_dividend`: rate x (1 + losers' money / (3 x own))), the other two winning pairs
    its co-winners. A pair's low figure is its smallest dividend over the results it wins in, its
    high figure the largest; floored at the minimum dividend (110 yen) when `floor`."""
    money = np.asarray(money, dtype=float)
    column = {p: j for j, p in enumerate(pairs)}
    runners = sorted({int(x) for p in pairs for x in p.split(",")})
    lo, hi = np.full(len(pairs), np.inf), np.full(len(pairs), -np.inf)
    for trio in itertools.combinations(runners, 3):
        winners = [column[f"{a},{b}"] for a, b in itertools.combinations(trio, 2)]
        for j in winners:
            dividend = jra_dividend(
                money[j],
                [money[w] for w in winners if w != j],
                floor=WIDE_FLOOR if floor else None,
            )
            lo[j], hi[j] = min(lo[j], dividend), max(hi[j], dividend)
    return lo, hi


def additive_wide_figures(
    money: np.ndarray, pairs: Sequence[str] = WIDE_PAIRS
) -> tuple[np.ndarray, np.ndarray]:
    """The form the review rejected: 1 + (rate - the three winning pairs' money) / (3 x own), the
    rate applied to the pool before the winning stakes are taken out (so not to the returned
    stake). It equals JRA's only when the three winning pairs hold equal money."""
    money = np.asarray(money, dtype=float)
    column = {p: j for j, p in enumerate(pairs)}
    runners = sorted({int(x) for p in pairs for x in p.split(",")})
    lo, hi = np.full(len(pairs), np.inf), np.full(len(pairs), -np.inf)
    for trio in itertools.combinations(runners, 3):
        winners = [column[f"{a},{b}"] for a, b in itertools.combinations(trio, 2)]
        pot = WIDE_PAYOUT - money[winners].sum()
        for j in winners:
            dividend = 1.0 + pot / (3.0 * money[j])
            lo[j], hi[j] = min(lo[j], dividend), max(hi[j], dividend)
    return lo, hi


def generic_low(money: np.ndarray, places: int = 3) -> np.ndarray:
    """The stake-back low figure (the shared identity) with the P - 1 most-backed *other
    combinations* as co-winners, whatever they are: right for PLACE's structure, not for Wide
    pairs (it ignores which pairs can win together)."""
    money = np.asarray(money, dtype=float)
    lo = np.empty(len(money))
    for i, own in enumerate(money):
        others = np.sort(np.delete(money, i))
        lo[i] = jra_dividend(own, others[len(others) - (places - 1) :])
    return lo


def wide_path(day: date, race_no: int) -> list[tuple[float, float, np.ndarray]]:
    """(minutes to post, pool total, pair money shares) of a race's Wide pool at each of
    SNAPSHOTS, the pool growing as the WIN and PLACE pools do."""
    rng = np.random.default_rng(day.toordinal() * 100 + race_no)
    v_final = 3e7 * math.exp(0.3 * rng.standard_normal())
    path = pool_path(rng, len(WIDE_PAIRS), 0.05, 0.3, v_final)
    return [(m, total, shares) for m, (total, shares) in zip(SNAPSHOTS, path, strict=True)]


def wide_rows(day: date, race_no: int, *, post: bool = True) -> list[dict]:
    """A race's Wide path in our own log: every pair and the pool total at every snapshot, its
    figures by JRA's pair rule (`wide_figures`). `post=False` is a logged day compacted without a
    post time (null post_utc and minutes_to_post), as our days were before #411."""
    race_id, at_post = race_id_of(day, race_no), post_of(day, race_no)
    rows = []
    for minutes, pool_total, shares in wide_path(day, race_no):
        final = minutes == FINAL_MIN
        lo, hi = wide_figures(shares)
        for j, pair in enumerate(WIDE_PAIRS):
            rows.append(
                row(race_id, race_no, WIDE, pair, at_post, minutes, odds=lo[j], odds_hi=hi[j],
                    final=final)
            )  # fmt: skip
        rows.append(
            row(race_id, race_no, WIDE, POOL_ROW, at_post, minutes, investment=pool_total,
                final=final)
        )  # fmt: skip
    if not post:
        rows = [{**r, "post_utc": None, "minutes_to_post": None} for r in rows]
    return rows


def logged_copy(rows: list[dict]) -> list[dict]:
    """Our own log of races the JRA-VAN file also holds: the same WIN and PLACE rows, plus a WIN
    runner LATE_ENTRY priced only at the final capture. Read in place of the JRA-VAN rows, or
    beside them, every WIN case of those races changes (it is dropped as added_by_close, or
    counted twice), so only a study that keeps one source per race and pool scores what the
    JRA-VAN file alone gives."""
    out = list(rows)
    for race_id in sorted({r["race_id"] for r in rows}):
        sample = next(r for r in rows if r["race_id"] == race_id)
        out.append(
            row(race_id, sample["race_no"], WIN, LATE_ENTRY, sample["post_utc"], FINAL_MIN,
                odds=150.0, final=True)
        )  # fmt: skip
    return out


def only_races(rows: list[dict], race_nos) -> list[dict]:
    return [r for r in rows if r["race_no"] in set(race_nos)]


def held_days(n: int) -> list[date]:
    start = date(2026, 11, 28)
    days = [start + timedelta(days=i) for i in range(60)]
    return [d for d in days if is_holdout_day(COUNTRY, d)][:n]


_JST = timezone(timedelta(hours=9))


def log_row(r: dict) -> dict:
    """A row in compact.SCHEMA as the logger writes it (docs/specs/odds-log.md; `fetched_utc` only
    where the row has one, as the key is optional in the log): a row of a final
    capture (`is_final_capture` in the fixtures) shows the page's final state, 最終オッズ with no
    clock (`source_update_utc` null, docs/specs/board-at-t.md); any other row the page's clock."""
    out = {
        "snapshot_utc": r["snapshot_utc"],
        **({} if r.get("fetched_utc") is None else {"fetched_utc": r["fetched_utc"]}),
        "race_id": r["race_id"],
        "pool": r["pool"],
        "combination": r["combination"],
        "odds": r["odds"],
        "status_raw": r["status_raw"],
    }
    for key in ("odds_hi", "investment"):
        if r[key] is not None:
            out[key] = r[key]
    if r["is_final_capture"]:
        out |= {"page_state": FINAL_TEXT, "source_update_utc": None}
    else:
        clock = r["snapshot_utc"].astimezone(_JST)
        out |= {
            "page_state": f"{clock.hour}時{clock.minute:02d}分現在",
            "source_update_utc": r["snapshot_utc"],
        }
    return out


def write_logged_day(day: date, rows: list[dict]) -> None:
    """One of our own logged days as the logger leaves it, then compacted: the rows as captures
    (one gzip member per snapshot instant) in the venue's odds log, the post times of the races
    that have one in its sidecar, and `compact_day` over the day."""
    log = paths.odds_snapshot_path("jra", day.isoformat(), VENUE)
    captures: dict[datetime, list[dict]] = defaultdict(list)
    for r in rows:
        captures[r["snapshot_utc"]].append(log_row(r))
    for instant in sorted(captures):
        odds_log.append_capture(log, captures[instant])
    posts = {int(r["race_no"]): r["post_utc"] for r in rows if r["post_utc"] is not None}
    updated = datetime(day.year, day.month, day.day, 0, 0, tzinfo=UTC)
    odds_log.write_post_times(log.with_name(f"{VENUE}.races.json"), posts, updated)
    compact_day(day.isoformat(), "jra")


def store_grade(race_id: str) -> str | None:
    """A race's grade in write_store's races table, from its id alone (G3 for about a third of
    the races, mixed over days and race numbers), so a store that adds races leaves every other
    race's grade as it was."""
    race_no = int(race_id.rsplit(":", 1)[1])
    return "G3" if (date.fromisoformat(race_day(race_id)).toordinal() + race_no) % 3 == 0 else None


def write_store(data: Path, *, jravan: dict, logged: dict) -> Path:
    """A data dir with JRA-VAN day files (`processed/jravan/odds_timeseries`), our own logged
    days (odds logs compacted into `processed/jra/odds_timeseries`; a bytes value is written as
    the day file as is) and the jravan races table over every race of both. Needs
    HORSETRACK_DATA_DIR at `data`."""
    race_ids = set()
    for day, rows in jravan.items():
        write_day(data / "processed" / "jravan" / "odds_timeseries" / f"{day.isoformat()}.parquet",
                  rows)  # fmt: skip
        race_ids |= {r["race_id"] for r in rows}
    for day, rows in logged.items():
        if isinstance(rows, bytes):
            path = data / "processed" / "jra" / "odds_timeseries" / f"{day.isoformat()}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(rows)
            continue
        write_logged_day(day, rows)
        race_ids |= {r["race_id"] for r in rows}
    race_ids = sorted(race_ids)
    races = pa.table(
        {
            "race_id": race_ids,
            "race_class": [store_grade(r) for r in race_ids],
            "race_date": [date.fromisoformat(race_day(r)) for r in race_ids],
        }
    )
    (data / "processed" / "jravan").mkdir(parents=True, exist_ok=True)
    pq.write_table(races, data / "processed" / "jravan" / "races.parquet")
    return data


class StudyRuns:
    """`run_on(name, jravan=..., logged=...)` -> (the report's JSON dump, the rendered report):
    one run of the study on its own data dir, D9's folds replaced by FOLD_SCHEME. `store` writes
    the data dir alone (so a test can watch the run's reads), `run` runs the study on it, and
    `result` is the last run's return value."""

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.tmp, self.mp, self.result = tmp_path, monkeypatch, None

    def store(self, name: str, *, jravan: dict, logged: dict) -> Path:
        data = self.tmp / name / "data"
        self.mp.setenv(paths.DATA_DIR_VAR, str(data))
        return write_store(data, jravan=jravan, logged=logged)

    def run(self, data: Path, *, n_draws=24, end=None, trial=True, out_dir=None):
        self.mp.setenv(paths.DATA_DIR_VAR, str(data))
        self.result = study.run_study(
            end=end,
            trial=trial,
            n_draws=n_draws,
            out_dir=data.parent / "out" if out_dir is None else out_dir,
            log=lambda *a: None,
        )
        report = Path(self.result["report_path"])
        dump = json.loads(report.with_suffix(".json").read_text(encoding="utf-8"))
        return dump, report.read_text(encoding="utf-8")

    def __call__(self, name: str, *, jravan: dict, logged: dict, **kwargs):
        return self.run(self.store(name, jravan=jravan, logged=logged), **kwargs)


@pytest.fixture
def run_on(tmp_path, monkeypatch) -> StudyRuns:
    summary = tmp_path / "summary"
    monkeypatch.setattr(study, "load_country_scheme", lambda country: FOLD_SCHEME)
    monkeypatch.setattr(study, "summary_dir", lambda country: summary)
    monkeypatch.setattr("horsetrack.core.eval.registry.summary_dir", lambda country: summary)
    return StudyRuns(tmp_path, monkeypatch)


def scored_by_pool(dump: dict, comparison: str = "structural_vs_random_walk") -> dict[str, int]:
    """Scored cases per pool, from the comparison's race-day gain table."""
    out: dict[str, int] = {}
    for r in dump["gains"][comparison]["day"]:
        out[r["pool"]] = out.get(r["pool"], 0) + int(r["n_cases"])
    return out


def fold_facts(dump: dict) -> dict:
    counts = dump["counts"]
    return {
        "development_days": counts["development_days"],
        "cases": counts["cases"],
        "added_by_close": counts["added_by_close"],
        "folds": dump["folds"],
        "scored": scored_by_pool(dump),
    }


def test_logged_development_days_join_the_folds_as_jravan_days_do(run_on):
    """#411: our own compacted logged days with post times enter the study's folds. A store whose
    last three days (TEST_DAYS) are only in our logs, and whose day `dup` is in both (races 1-2
    in the JRA-VAN file, races 1-3 in our log, race 3 absent from JRA-VAN), gives the same
    development days, cases, folds and scored cases per pool as the store holding every day as a
    JRA-VAN file; the JRA-VAN-only subset gives fewer. The duplicated races are read once, from
    JRA-VAN (their logged copy carries a late entry that would change every WIN case it
    reached); race 3, held only in our log and without the late entry, is read from the log."""
    first, last = JRAVAN_DAYS, TEST_DAYS
    dup = JRAVAN_DAYS[3]
    every_day = {d: day_rows(d, races=3 if d == dup else 2) for d in STUDY_DAYS}

    all_jravan, _ = run_on("all-jravan", jravan=every_day, logged={})
    split, _ = run_on(
        "split",
        jravan={d: only_races(every_day[d], (1, 2)) for d in first},
        logged={d: every_day[d] for d in last}
        | {dup: logged_copy(only_races(every_day[dup], (1, 2))) + only_races(every_day[dup], (3,))},
    )
    subset, _ = run_on(
        "jravan-subset", jravan={d: only_races(every_day[d], (1, 2)) for d in first}, logged={}
    )

    assert all_jravan["counts"]["development_days"] == len(STUDY_DAYS)
    assert subset["counts"]["development_days"] == len(first)
    assert fold_facts(split) == fold_facts(all_jravan)
    assert fold_facts(subset) != fold_facts(all_jravan)


def test_a_holdout_logged_day_is_never_read(run_on, monkeypatch):
    """D14: a logged holdout day is fully locked. Of two logged holdout days, one is a valid day
    file with races and post times, the other is not parquet at all, so a read of it raises; every
    parquet open is watched and none may name a holdout day. The logged development days are read
    (they enter the folds beside the JRA-VAN days), and the run completes."""
    held = held_days(2)
    assert len(held) == 2
    jravan = {d: day_rows(d, races=2) for d in JRAVAN_DAYS}
    logged = {d: day_rows(d, races=2) for d in TEST_DAYS}
    logged[held[0]] = day_rows(held[0], races=2) + wide_rows(held[0], 1)
    logged[held[1]] = b"a locked holdout day\n"
    data = run_on.store("holdout", jravan=jravan, logged=logged)

    opened: list[str] = []
    for module, name in ((pq, "read_table"), (pq, "ParquetFile"), (pq, "ParquetDataset"),
                         (pq, "read_metadata"), (pq, "read_schema"), (ds, "dataset")):  # fmt: skip
        real = getattr(module, name)

        def spy(source, *args, _real=real, **kwargs):
            opened.append(str(source))
            return _real(source, *args, **kwargs)

        monkeypatch.setattr(module, name, spy)

    try:
        dump, _ = run_on.run(data)
    except Exception as exc:  # the second holdout file is not parquet: reading it raises
        pytest.fail(f"the study read a holdout day's file: {exc!r}")

    touched = [p for p in opened if any(d.isoformat() in p for d in held)]
    assert not touched, touched
    assert dump["counts"]["development_days"] == len(STUDY_DAYS)
    logged_read = [p for p in opened if any(f"{d.isoformat()}.parquet" in p for d in TEST_DAYS)]
    assert logged_read, "the logged development days were not read"


def test_wide_is_fitted_and_scored_from_our_logged_days(run_on):
    """D3 (#411): JRA-VAN has no Wide series, so Wide comes from our own logs. With every study
    day in JRA-VAN (WIN, PLACE) and in our logs (the same races, with Wide paths and post times),
    the structural nowcast is fitted for Wide (Wide cells) and Wide cases are scored against the
    random walk with finite gains; the report says Wide was fitted. WIN and PLACE are scored on
    exactly the cases of the JRA-VAN files alone: the logged copies add Wide, not duplicates."""
    jravan = {d: day_rows(d, races=2) for d in STUDY_DAYS}
    logged = {d: logged_copy(day_rows(d, races=2)) + wide_rows(d, 1) + wide_rows(d, 2)
              for d in STUDY_DAYS}  # fmt: skip

    alone, _ = run_on("jravan-alone", jravan=jravan, logged={})
    dump, _ = run_on("with-wide", jravan=jravan, logged=logged)

    scored = scored_by_pool(dump)
    assert WIDE not in scored_by_pool(alone)
    assert scored.get(WIDE, 0) > 0, scored
    assert {p: n for p, n in scored.items() if p != WIDE} == scored_by_pool(alone)
    wide_gains = [r for r in dump["gains"]["structural_vs_random_walk"]["day"] if r["pool"] == WIDE]
    assert {r["tau_min"] for r in wide_gains} <= set(GRID)
    assert all(np.isfinite(r["log_score_gain"]) and np.isfinite(r["crps_gain"]) for r in wide_gains)
    assert any(c["pool"] == WIDE for c in dump["cells"]), "no Wide cell was fitted"
    assert dump["wide"]["fitted"] is True


def test_the_report_states_why_wide_is_not_fitted(run_on):
    """#411: where the logged days do not allow a Wide fit, the run still completes, scores no
    Wide case, and the report says so and why: `wide["fitted"]` is False and `wide["reason"]` is
    a sentence that appears in the rendered report. Two causes, two different reasons: the logged
    Wide paths have no post time; and Wide is logged only on the last (scored) day, so no fold
    trains on it. Our logged days are still listed (`own_logged_days`)."""
    jravan = {d: day_rows(d, races=2) for d in STUDY_DAYS}
    no_post = {d: wide_rows(d, 1, post=False) + wide_rows(d, 2, post=False) for d in STUDY_DAYS}
    last_only = {STUDY_DAYS[-1]: wide_rows(STUDY_DAYS[-1], 1) + wide_rows(STUDY_DAYS[-1], 2)}

    reasons = []
    for name, logged in (("no-post-time", no_post), ("scored-day-only", last_only)):
        dump, text = run_on(name, jravan=jravan, logged=logged)
        assert WIDE not in scored_by_pool(dump), name
        assert not any(c["pool"] == WIDE for c in dump["cells"]), name
        wide = dump["wide"]
        assert wide["fitted"] is False, name
        reason = wide["reason"]
        assert isinstance(reason, str) and reason.strip(), name
        assert reason in text, f"{name}: the report does not state {reason!r}"
        assert {e["day"] for e in wide["own_logged_days"]} == {d.isoformat() for d in logged}
        reasons.append(reason)
    assert reasons[0] != reasons[1], reasons


# ---------------------------------------------------------------- #411 review: Wide pair shares


def wide_board(seed: int, runners: Sequence[int], kind: str) -> tuple[tuple[str, ...], np.ndarray]:
    """A Wide board over `runners`: its pairs ("a,b") and their money (sums to 1), every low
    figure above the floor. "pairs" draws the pair money directly; "runners" builds it from runner
    strengths (m_ab proportional to p_a p_b), so a favourite's pairs carry the pool."""
    runners = sorted(runners)
    pairs = tuple(f"{a},{b}" for a, b in itertools.combinations(runners, 2))
    for attempt in range(500):
        rng = np.random.default_rng([seed, attempt])
        if kind == "pairs":
            money = rng.dirichlet(np.full(len(pairs), 3.0))
        else:
            p = dict(zip(runners, rng.dirichlet(np.full(len(runners), 2.0)), strict=True))
            money = np.array([p[int(a)] * p[int(b)] for a, b in (q.split(",") for q in pairs)])
            money = money / money.sum()
        lo, _ = wide_figures(money, pairs, floor=False)
        if lo.min() > WIDE_FLOOR + 1e-3:
            return pairs, money
    raise AssertionError(f"no board above the floor for {runners}, {kind}")


WIDE_BOARDS = tuple(
    (n * 10 + j, tuple(range(1, n + 1)), kind)
    for n in (5, 6, 8, 11, 16)
    for j, kind in enumerate(("pairs", "runners"))
) + ((71, (1, 2, 3, 5, 6, 7, 8), "runners"),)  # runner 4 scratched: its pairs are not sold


def test_wide_figures_are_priced_over_the_results_a_pair_can_win_in():
    """#411 review: `wide_odds(shares, combinations, payout)` prints each pair's low and high
    figure as JRA's rule does over every feasible top three (`wide_figures`, brute force through
    `core.pricing.stake_back_share_dividend` here). Fields of 5 to 16 runners, pair money drawn
    directly or from runner strengths, and a field with a scratched runner's pairs missing; on
    each the rule that takes the two most-backed other pairs as co-winners (`generic_low`) prints
    different figures, and so does the form with the rate applied before the winning stakes are
    taken out (`additive_wide_figures`; the winning pairs' money is unequal on every board), so
    neither can pass. Rows of a 2-D input are draws."""
    for seed, runners, kind in WIDE_BOARDS:
        pairs, money = wide_board(seed, runners, kind)
        expected_lo, expected_hi = wide_figures(money, pairs, floor=False)
        assert np.abs(generic_low(money) - expected_lo).max() > 0.05, (runners, kind)
        additive_lo, additive_hi = additive_wide_figures(money, pairs)
        assert np.abs(additive_lo - expected_lo).max() > 0.05, (runners, kind)
        assert np.abs(additive_hi - expected_hi).max() > 0.05, (runners, kind)
        lo, hi = nc.wide_odds(money, pairs, payout=WIDE_PAYOUT)
        np.testing.assert_allclose(lo, expected_lo, rtol=1e-9, err_msg=f"{runners} {kind}")
        np.testing.assert_allclose(hi, expected_hi, rtol=1e-9, err_msg=f"{runners} {kind}")

    boards = [wide_board(seed, range(1, 10), "runners") for seed in (91, 92, 93)]
    pairs = boards[0][0]
    stack = np.vstack([money for _, money in boards])
    lo, hi = nc.wide_odds(stack, pairs, payout=WIDE_PAYOUT)
    assert lo.shape == hi.shape == stack.shape
    for row, money in enumerate(stack):
        expected_lo, expected_hi = wide_figures(money, pairs, floor=False)
        np.testing.assert_allclose(lo[row], expected_lo, rtol=1e-9)
        np.testing.assert_allclose(hi[row], expected_hi, rtol=1e-9)


def test_wide_money_inverts_the_pair_rule():
    """#411 review: `wide_money(lo, combinations, payout)` recovers each pair's money (units of
    the pool) from the low figures JRA's pair rule prints (brute force through
    `core.pricing.stake_back_share_dividend`), on every board of WIDE_BOARDS and row by row for a
    2-D input; and the round trip through its own forward rule (`wide_odds`) recovers the money."""
    for seed, runners, kind in WIDE_BOARDS:
        pairs, money = wide_board(seed, runners, kind)
        lo, _ = wide_figures(money, pairs, floor=False)
        got = nc.wide_money(lo, pairs, payout=WIDE_PAYOUT)
        np.testing.assert_allclose(got, money, rtol=1e-6, atol=1e-12, err_msg=f"{runners} {kind}")
        forward, _ = nc.wide_odds(money, pairs, payout=WIDE_PAYOUT)
        back = nc.wide_money(forward, pairs, payout=WIDE_PAYOUT)
        np.testing.assert_allclose(back, money, rtol=1e-6, atol=1e-12, err_msg=f"{runners} {kind}")

    boards = [wide_board(seed, range(1, 13), "pairs") for seed in (121, 122, 123)]
    pairs = boards[0][0]
    stack = np.vstack([money for _, money in boards])
    lo = np.vstack([wide_figures(money, pairs, floor=False)[0] for money in stack])
    got = nc.wide_money(lo, pairs, payout=WIDE_PAYOUT)
    np.testing.assert_allclose(got, stack, rtol=1e-6, atol=1e-12)


def place_board(rng, k: int) -> np.ndarray:
    for _ in range(500):
        shares = rng.dirichlet(np.full(k, 4.0))
        if place_odds(shares, 1e8)[0].min() > 1.1 + 1e-3:
            return shares
    raise AssertionError(f"no {k}-runner Place board above the floor")


def test_pool_shares_reads_each_pool_by_its_own_rule():
    """#411 review: `pool_shares` gives WIN shares as 1/O normalised and PLACE shares by the
    stake-back rule with the most-backed other runners (both as before), and Wide shares by the
    pair rule, given the board's pairs (`combinations=`): the money shares, with `rule_sum` (the
    recovered money's total) 1 on Wide boards built by the shared `core.pricing` identity
    (`wide_figures`). Boards of 8 to 16 runners for WIN and PLACE, WIDE_BOARDS and a 2-D board
    stack for Wide."""
    rng = np.random.default_rng(4111)
    for k in (8, 12, 16):
        shares = place_board(rng, k)
        got, rule_sum = nc.pool_shares(WIN, (1.0 - TAKEOUT) / shares)
        np.testing.assert_allclose(got, shares, rtol=1e-9)
        assert math.isnan(rule_sum)
        got, rule_sum = nc.pool_shares(PLACE, place_odds(shares, 1e8)[0])
        np.testing.assert_allclose(got, shares, rtol=1e-9)
        assert rule_sum == pytest.approx(1.0, abs=1e-9)

    for seed, runners, kind in WIDE_BOARDS:
        pairs, money = wide_board(seed, runners, kind)
        lo, _ = wide_figures(money, pairs)
        got, rule_sum = nc.pool_shares(WIDE, lo, combinations=pairs)
        np.testing.assert_allclose(got, money, rtol=1e-6, err_msg=f"{runners} {kind}")
        assert rule_sum == pytest.approx(1.0, abs=1e-6), (runners, kind)

    boards = [wide_board(seed, range(1, 8), "runners") for seed in (171, 172)]
    pairs = boards[0][0]
    stack = np.vstack([money for _, money in boards])
    lo = np.vstack([wide_figures(money, pairs)[0] for money in stack])
    got, _ = nc.pool_shares(WIDE, lo, combinations=pairs)
    np.testing.assert_allclose(got, stack, rtol=1e-6)


AT = datetime(2026, 11, 14, 5, 0, tzinfo=UTC)


def wide_race_board(pairs: Sequence[str], lo, hi) -> RaceBoard:
    runners = sorted({int(x) for p in pairs for x in p.split(",")})
    wide = {
        tuple(int(x) for x in p.split(",")): (float(a), float(b))
        for p, a, b in zip(pairs, lo, hi, strict=True)
    }
    return RaceBoard(
        race_id_of(TEST_DAYS[0], 1), None, AT, AT, False,
        {h: 2.0 + h for h in runners}, {}, wide, {}, frozenset(),
    )  # fmt: skip


def test_the_c1_draws_print_wide_figures_by_the_pair_rule():
    """#411 review: `ShadowNowcast` (the C1 draws behind #116's interface) prints Wide figures by
    the pair rule. At phi = 1 every draw is the board's money, so each draw prints the board's own
    low and high figures; the high figure differs from the generic rule's on this board."""
    pairs, money = wide_board(4114, range(1, 7), "runners")
    lo, hi = wide_figures(money, pairs)
    rng = np.random.default_rng(7)
    shocks = {h: rng.standard_normal(6) for h in range(1, 7)}
    shadow = ShadowNowcast(Nowcast.from_params(phi=1.0, sigma=0.5))
    drawn = shadow.draws(wide_race_board(pairs, lo, hi), pool=WIDE, tau_min=5, shocks=shocks)
    assert drawn.combinations == pairs
    np.testing.assert_allclose(drawn.odds, np.tile(lo, (6, 1)), rtol=1e-6)
    np.testing.assert_allclose(drawn.odds_hi, np.tile(hi, (6, 1)), rtol=1e-6)


class EchoDrift:
    """A drift model whose every draw is the board's own figures (no drift)."""

    def draws(self, board, *, pool, tau_min, shocks):
        keys = sorted(board.wide)
        n = len(np.asarray(next(iter(shocks.values()))))
        lo = np.array([board.wide[k][0] for k in keys], dtype=float)
        hi = np.array([board.wide[k][1] for k in keys], dtype=float)
        combinations = tuple(f"{a},{b}" for a, b in keys)
        return PoolDraws(pool, combinations, np.tile(lo, (n, 1)), np.tile(hi, (n, 1)))


def test_the_drift_forecaster_reads_wide_figures_back_by_the_pair_rule():
    """#411 review: the study's drift forecaster reads its Wide odds draws back into shares by the
    pair rule: with draws that are the board's own figures, every drawn share vector is the
    board's money."""
    pairs, money = wide_board(4115, range(1, 9), "runners")
    lo, hi = wide_figures(money, pairs)
    board = PoolBoard(pairs, money, None, None, None, 5.0, odds=lo, odds_hi=hi)
    view = RaceView(race_id_of(TEST_DAYS[0], 1), VENUE, 1, None, 5.0, {WIDE: board})
    draws = study.drift_forecaster(EchoDrift())(view, pool=WIDE, n=4, seed=3)
    np.testing.assert_allclose(draws.shares, np.tile(money, (4, 1)), rtol=1e-6)


def test_logged_wide_boards_read_back_the_pair_money():
    """#411 review: Wide cases built from our logged paths (figures by JRA's pair rule) carry
    the pair money as their shares at t and at the close, and the rule's check (`rule_sum`) is 1.
    """
    day = TEST_DAYS[0]
    cases, _, _ = extract(to_frame(wide_rows(day, 1) + wide_rows(day, 2)))
    wide = np.flatnonzero(cases.pool == WIDE)
    assert len(wide) == 2 * len(GRID)
    truth = {race_id_of(day, n): {m: s for m, _, s in wide_path(day, n)} for n in (1, 2)}
    for shares in (s for path in truth.values() for s in path.values()):
        assert wide_figures(shares, floor=False)[0].min() > WIDE_FLOOR  # no figure is floored
    column = {p: j for j, p in enumerate(WIDE_PAIRS)}
    for i in wide:
        r = cases.rows(i)
        path = truth[cases.race_id[i]]
        order = [column[c] for c in cases.combination[r]]
        np.testing.assert_allclose(cases.s_t[r], path[cases.tau[i]][order], rtol=1e-6)
        np.testing.assert_allclose(cases.s_f[r], path[FINAL_MIN][order], rtol=1e-6)
        assert cases.rule_sum[i] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------- #411 review: logged scratches


def logged_scratch(rows: list[dict], race_id: str, runner: str, *, from_minute: float) -> list:
    """`rows` with `runner` of `race_id` scratched `from_minute` before the post, as JRA's pages
    are logged: its WIN and PLACE lines carry "SCR" and no price, and the Wide pairs naming it are
    simply no longer listed."""
    out = []
    for r in rows:
        if r["race_id"] != race_id or r["minutes_to_post"] > from_minute:
            out.append(r)
        elif r["pool"] in (WIN, PLACE) and r["combination"] == runner:
            out.append({**r, "odds": None, "odds_hi": None, "status_raw": "SCR"})
        elif not (r["pool"] == WIDE and runner in r["combination"].split(",")):
            out.append(r)
    return out


def test_a_logged_scratch_takes_its_wide_pairs_off_the_board():
    """#411 review: runner 6 is scratched 20 minutes before the post. From then on its five Wide
    pairs are off the Wide board (T-16 and later, and the final board), as the board reader does
    for WIN and PLACE; a board from before the scratch (T-60, T-30) keeps them, and its case is
    renormalised over the pairs priced at the close (#69): those five have no final share."""
    day = TEST_DAYS[1]
    race_id = race_id_of(day, 1)
    rows = race_rows(np.random.default_rng(4113), day, 1, WIDE_RUNNERS) + wide_rows(day, 1)
    cases, _, _ = extract(to_frame(logged_scratch(rows, race_id, "6", from_minute=20)))
    with_six = {p for p in WIDE_PAIRS if "6" in p.split(",")}
    seen = set()
    for i in np.flatnonzero(cases.pool == WIDE):
        tau, r = float(cases.tau[i]), cases.rows(i)
        combinations = list(cases.combination[r])
        seen.add(tau)
        if tau <= 20:
            assert set(combinations) == set(WIDE_PAIRS) - with_six, tau
            assert cases.scratched[i] == 0, tau
        else:
            assert set(combinations) == set(WIDE_PAIRS), tau
            assert cases.scratched[i] == len(with_six), tau
            no_final = {c for c, s in zip(combinations, cases.s_f[r], strict=True) if np.isnan(s)}
            assert no_final == with_six, tau
    assert seen == set(map(float, GRID))


FETCH_LAG_S = 20.0  # a page's fetch after its capture round starts (fetched_utc - snapshot_utc)


def fetched(rows: list[dict], lags: dict[tuple[str, float], float]) -> list[dict]:
    """`rows` with a fetched_utc: the round's snapshot_utc plus FETCH_LAG_S seconds, or plus
    `lags[(race_id, minutes to post)]` seconds for the WIN and PLACE pages of that round."""
    out = []
    for r in rows:
        lag = FETCH_LAG_S
        if r["pool"] in (WIN, PLACE):
            lag = lags.get((r["race_id"], r["minutes_to_post"]), FETCH_LAG_S)
        out.append({**r, "fetched_utc": r["snapshot_utc"] + timedelta(seconds=lag)})
    return out


def test_a_scratch_counts_on_the_wide_board_from_its_page_fetch():
    """#411 review: a scratch is known from the fetched_utc of the WIN or PLACE row that reports
    it (snapshot_utc when the row has none), not from the start of its capture round. Two races,
    runner 6 scratched in each, logged and compacted and read as the study reads a logged day
    (`read_logged_day`, then `extract` with its scratches):
    - race 1: the round at T-2 (snapshot at the T-2 cutoff) reports the scratch on pages fetched
      30 s later, after that cutoff and before the T-1 one. The T-2 Wide board still carries
      runner 6's pairs (its case renormalised over the pairs priced at the close); the T-1 board
      does not. The T-1 round's own pages are fetched 20 s after the T-1 cutoff, too late for
      it, so the T-2 round's fetch time is what takes the pairs off.
    - race 2: the round at T-6 reports it on pages fetched 70 s later, after the T-5 cutoff (the
      next round, at T-5, starts before the cutoff too). The T-5 board carries the pairs; T-2 on
      it does not.
    Each board keeps the pairs exactly when the scratch's fetch is after its cutoff."""
    day = TEST_DAYS[1]
    rng = np.random.default_rng(4116)
    rows, known = [], {}
    for race_no, round_min, lag_s in ((1, 2.0, 30.0), (2, 6.0, 70.0)):
        race_id = race_id_of(day, race_no)
        race = race_rows(rng, day, race_no, WIDE_RUNNERS) + wide_rows(day, race_no)
        race = logged_scratch(race, race_id, "6", from_minute=round_min)
        rows += fetched(race, {(race_id, round_min): lag_s})
        known[race_id] = round_min - lag_s / 60.0  # minutes before the post the scratch is known
    write_logged_day(day, rows)
    path = paths.processed_dir("jra") / "odds_timeseries" / f"{day.isoformat()}.parquet"
    own = study.read_logged_day(path, None)
    cases, _, _ = extract(own.frame, scratches=own.scratches)

    with_six = {p for p in WIDE_PAIRS if "6" in p.split(",")}
    seen = defaultdict(set)
    kept_at = defaultdict(set)
    for i in np.flatnonzero(cases.pool == WIDE):
        race_id, tau, r = str(cases.race_id[i]), float(cases.tau[i]), cases.rows(i)
        combinations = set(cases.combination[r])
        seen[race_id].add(tau)
        if tau > known[race_id]:  # the cutoff is before the scratch's fetch: the pairs stay
            kept_at[race_id].add(tau)
            assert combinations == set(WIDE_PAIRS), (race_id, tau)
            assert cases.scratched[i] == len(with_six), (race_id, tau)
            no_final = {c for c, s in zip(cases.combination[r], cases.s_f[r], strict=True)
                        if np.isnan(s)}  # fmt: skip
            assert no_final == with_six, (race_id, tau)
        else:
            assert combinations == set(WIDE_PAIRS) - with_six, (race_id, tau)
            assert cases.scratched[i] == 0, (race_id, tau)
    for race_id in known:
        assert seen[race_id] == set(map(float, GRID)), race_id
    assert 2.0 in kept_at[race_id_of(day, 1)] and 1.0 not in kept_at[race_id_of(day, 1)]
    assert 5.0 in kept_at[race_id_of(day, 2)] and 2.0 not in kept_at[race_id_of(day, 2)]


# ---------------------------------------------------------------- #411 review: the study


def stopped(rows: list[dict], *, last_minute: float) -> list[dict]:
    """A race whose capture stopped `last_minute` before the post: no capture after it, so no
    capture of it shows the final state."""
    return [{**r, "is_final_capture": False} for r in rows if r["minutes_to_post"] >= last_minute]


def test_a_logged_path_without_a_final_capture_is_no_target(run_on):
    """#411 review: a logged race whose capture stopped at T-10 (its WIN, PLACE and Wide paths
    never show the final state) gives no case and no growth observation: a store holding it, on a
    JRA-VAN day (where only our log has it) and on a logged day, runs exactly as the store without
    it (cases, folds, scored cases and gains), and the run counts it (`no_final_capture`, 0 for
    the store whose logged paths all reach a final capture)."""
    jravan = {d: day_rows(d, races=2) for d in JRAVAN_DAYS}
    final = {d: day_rows(d, races=2) + wide_rows(d, 1) + wide_rows(d, 2) for d in TEST_DAYS}

    def cut(d):
        return stopped(only_races(day_rows(d, races=3), (3,)) + wide_rows(d, 3), last_minute=10)

    with_cut = final | {TEST_DAYS[0]: final[TEST_DAYS[0]] + cut(TEST_DAYS[0]),
                        JRAVAN_DAYS[-1]: cut(JRAVAN_DAYS[-1])}  # fmt: skip

    clean, _ = run_on("all-final", jravan=jravan, logged=final)
    dump, _ = run_on("capture-stopped", jravan=jravan, logged=with_cut)

    assert fold_facts(dump) == fold_facts(clean)
    assert json.dumps(dump["gains"], sort_keys=True) == json.dumps(clean["gains"], sort_keys=True)
    assert clean["counts"]["no_final_capture"] == 0
    assert dump["counts"]["no_final_capture"] > 0


def test_a_fold_with_only_unscorable_cases_completes(run_on):
    """#411 review: the fold testing 2026-11-14/15 holds only logged Wide cases, and no earlier
    day holds Wide, so none of them can be scored. The run completes, every fold is reported, and
    that fold reports no scored case and its 2 days x 2 races x len(GRID) unscored ones."""
    jravan = {d: day_rows(d, races=2) for d in (*JRAVAN_DAYS, TEST_DAYS[2])}
    logged = {d: wide_rows(d, 1) + wide_rows(d, 2) for d in TEST_DAYS[:2]}
    dump, text = run_on("wide-only-fold", jravan=jravan, logged=logged)
    assert len(dump["folds"]) == dump["counts"]["folds"]
    empty = [f for f in dump["folds"] if f["cases"] == 0]
    assert len(empty) == 1, dump["folds"]
    assert empty[0]["test_days"] == 2
    assert empty[0]["unscored_no_fit"] == 2 * 2 * len(GRID)
    assert "unscored_no_fit" in text


def test_a_run_whose_only_fold_holds_wide_only_logs_renders_its_report(run_on):
    """#411 review: a Wide-only capture (`log-day --pools`, so our log holds Wide and nothing
    else) on the only fold's test days, before any Wide cell exists: no case of the run can be
    scored. The run still completes and renders its report: the fold is reported with no scored
    case and its 2 days x 2 races x len(GRID) unscored ones, no comparison holds a scored case,
    the report states why Wide is not fitted, and the run is in the registry."""
    jravan = {d: day_rows(d, races=2) for d in TRAIN_DAYS}
    tested = JRAVAN_DAYS[len(TRAIN_DAYS) :]  # the one fold's test week
    logged = {d: wide_rows(d, 1) + wide_rows(d, 2) for d in tested}
    dump, text = run_on("wide-only-run", jravan=jravan, logged=logged)

    assert dump["counts"]["folds"] == len(dump["folds"]) == 1, dump["folds"]
    fold = dump["folds"][0]
    assert fold["cases"] == 0 and fold["test_days"] == len(tested) == 2, fold
    assert fold["unscored_no_fit"] == 2 * 2 * len(GRID), fold
    scored = [r for comparison in dump["gains"].values() for r in comparison.get("day", [])
              if r.get("n_cases")]  # fmt: skip
    assert not scored, scored
    assert dump["wide"]["fitted"] is False
    assert dump["wide"]["reason"] in text
    records = default_registry(COUNTRY).records()
    assert records and all(r.family_id.startswith(LOGGED_FAMILY) for r in records)


FINAL_CAPTURE = re.compile(r"final capture", re.IGNORECASE)
PRE_POST_BOARD = re.compile(r"pre-post board|boards? before the post", re.IGNORECASE)


def test_the_wide_reason_names_each_missing_input(run_on):
    """#411 review: confirmed finals and usable grid boards are tracked apart, and the Wide reason
    names what is missing. Two kinds of logged Wide race, each with its pool total:
    - blocked: the logger stopped at T-5 (no capture shows the final state), so no target;
    - after the post: a confirmed final capture, taken only after the post, so no board before
      it.
    The blocked race alone gives the missing final capture (`FINAL_CAPTURE`) and no case; the
    after-post race alone gives the missing boards before the post (`PRE_POST_BOARD`), and neither
    counts a path without a final capture nor points to recompaction; both together name both.
    The three reasons differ and each is printed in its report."""
    jravan = {d: day_rows(d, races=2) for d in STUDY_DAYS}
    blocked_day, after_day = JRAVAN_DAYS[0], JRAVAN_DAYS[1]
    blocked = stopped(wide_rows(blocked_day, 1), last_minute=5)
    after_post = [r for r in wide_rows(after_day, 2) if r["minutes_to_post"] < 0]
    assert after_post and all(r["is_final_capture"] for r in after_post)

    stores = {
        "blocked": {blocked_day: blocked},
        "after-post": {after_day: after_post},
        "both": {blocked_day: blocked, after_day: after_post},
    }
    reasons, counts = {}, {}
    for name, logged in stores.items():
        dump, text = run_on(f"reason-{name}", jravan=jravan, logged=logged)
        wide = dump["wide"]
        assert wide["fitted"] is False, name
        assert wide["wide_cases"] == 0, name
        assert WIDE not in scored_by_pool(dump), name
        assert wide["reason"] in text, name
        reasons[name], counts[name] = wide["reason"], dump["counts"]["no_final_capture"]

    assert counts == {"blocked": 1, "after-post": 0, "both": 1}, counts
    assert FINAL_CAPTURE.search(reasons["blocked"]), reasons["blocked"]
    assert not PRE_POST_BOARD.search(reasons["blocked"]), reasons["blocked"]
    assert PRE_POST_BOARD.search(reasons["after-post"]), reasons["after-post"]
    assert "recompact" not in reasons["after-post"].lower(), reasons["after-post"]
    assert FINAL_CAPTURE.search(reasons["both"]), reasons["both"]
    assert PRE_POST_BOARD.search(reasons["both"]), reasons["both"]
    assert len(set(reasons.values())) == 3, reasons


V2_FAMILY = "japan-c1-nowcast-final-shares-v2"
LOGGED_FAMILY = "japan-c1-nowcast-logged-wide-v1"


def v2_spec_hash() -> str:
    """The hash of #85's v2 spec (`build_spec`) at this run's grid, folds (FOLD_SCHEME), bootstrap
    and drift config, as the registry hashes a spec (sorted, compact JSON)."""
    scoring = config.load_country_yaml(COUNTRY, "scoring")["bootstrap"]
    bootstrap = {"n_boot": int(scoring["n_boot"]), "level": float(scoring["level"]),
                 "seed": study.SEED}  # fmt: skip
    spec = study.build_spec(
        load_decision_grid(),
        FOLD_SCHEME,
        bootstrap,
        config.load_country_yaml(COUNTRY, "nowcast_v0"),
    )
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def frozen_specs(registry) -> dict[str, dict]:
    out = {}
    for line in registry.path.read_text(encoding="ascii").splitlines():
        entry = json.loads(line)
        if entry.get("kind") == "freeze":
            out[entry["family_id"]] = entry["spec"]
    return out


def test_a_run_reading_our_logs_is_its_own_family_and_report(run_on, tmp_path):
    """#411 review: a full run on JRA-VAN days alone keeps #85's family v2, its spec hash and
    its report; the same store with our logged days added registers under
    japan-c1-nowcast-logged-wide-v1, frozen with a spec of its own that names both sources, and
    writes its report as `nowcast-japan-logged-wide-v1-<spec hash[:12]>.md`, leaving v2's report
    as it was."""
    reports = tmp_path / "reports"
    data = run_on.store("families", jravan={d: day_rows(d, races=2) for d in STUDY_DAYS}, logged={})
    registry = default_registry(COUNTRY)

    run_on.run(data, trial=False, out_dir=reports)
    v2 = run_on.result
    assert v2["family_id"] == V2_FAMILY
    assert v2["spec_hash"] == v2_spec_hash() == registry.frozen_spec_hash(V2_FAMILY)
    first = registry.records()
    assert first and {r.family_id for r in first} == {V2_FAMILY}
    v2_report = Path(v2["report_path"])
    v2_files = {p: p.read_bytes() for p in (v2_report, v2_report.with_suffix(".json"))}

    for day in STUDY_DAYS:
        write_logged_day(day, wide_rows(day, 1) + wide_rows(day, 2))
    mixed, _ = run_on.run(data, trial=False, out_dir=reports)
    logged = run_on.result
    assert scored_by_pool(mixed).get(WIDE, 0) > 0  # the logged days were read
    assert logged["family_id"] == LOGGED_FAMILY
    assert logged["spec_hash"] != v2["spec_hash"]
    assert registry.frozen_spec_hash(LOGGED_FAMILY) == logged["spec_hash"]
    assert registry.frozen_spec_hash(V2_FAMILY) == v2["spec_hash"]
    added = registry.records()[len(first) :]
    assert added and {r.family_id for r in added} == {LOGGED_FAMILY}
    logged_report = Path(logged["report_path"])
    assert logged_report.name == f"nowcast-{COUNTRY}-logged-wide-v1-{logged['spec_hash'][:12]}.md"
    assert logged_report not in v2_files
    assert {p: p.read_bytes() for p in v2_files} == v2_files
    spec = frozen_specs(registry)[LOGGED_FAMILY]
    assert spec["family_id"] == LOGGED_FAMILY
    text = json.dumps(spec)
    assert "processed/jravan/odds_timeseries" in text and "processed/jra/odds_timeseries" in text


def shifted(rows: list[dict], minutes: float) -> list[dict]:
    """`rows` with the race's post and every snapshot `minutes` later."""
    d = timedelta(minutes=minutes)
    return [
        {**r, "snapshot_utc": r["snapshot_utc"] + d, "post_utc": r["post_utc"] + d} for r in rows
    ]


def test_each_path_is_timed_by_its_own_post(run_on):
    """#411 review: JRA-VAN times every race 5 minutes later than our log does (e.g. 15:40 against
    15:35), each source's snapshots at its own grid minutes. Each (race, pool) path is read at its
    own source's post minus tau: every board, WIN and PLACE from JRA-VAN and Wide from our log, is
    exactly tau old; timed by JRA-VAN's post the Wide board at T-2 would be our capture 2 minutes
    after our post (3 minutes old)."""
    jravan = {d: shifted(day_rows(d, races=2), 5) for d in STUDY_DAYS}
    logged = {d: wide_rows(d, 1) + wide_rows(d, 2) for d in STUDY_DAYS}
    dump, _ = run_on("two-posts", jravan=jravan, logged=logged)
    ages = {(r["pool"], float(r["tau_min"])): r for r in dump["board_age_min"]}
    assert {p for p, _ in ages} == {WIN, PLACE, WIDE}
    assert ages[(WIDE, 2.0)]["n"] > 0
    for key, r in ages.items():
        assert r["max"] <= 1e-6, (key, r)


def test_a_jravan_path_without_a_post_time_leaves_the_race_to_our_log(run_on):
    """#411 review: JRA-VAN wins a (race, pool) only where its path has a post time. Race 1 of one
    day has no post time in the JRA-VAN file and is in our log with one: the run scores what the
    JRA-VAN file with the post time gives; without our log the race gives no case."""
    dup = JRAVAN_DAYS[3]
    every_day = {d: day_rows(d, races=2) for d in STUDY_DAYS}
    no_post = [
        {**r, "post_utc": None, "minutes_to_post": None} if r["race_no"] == 1 else r
        for r in every_day[dup]
    ]
    all_jravan, _ = run_on("with-post", jravan=every_day, logged={})
    alone, _ = run_on("no-post-alone", jravan=every_day | {dup: no_post}, logged={})
    split, _ = run_on(
        "no-post-logged",
        jravan=every_day | {dup: no_post},
        logged={dup: only_races(every_day[dup], (1,))},
    )
    assert fold_facts(alone) != fold_facts(all_jravan)
    assert fold_facts(split) == fold_facts(all_jravan)


def test_wide_coverage_reads_the_days_the_folds_read(run_on):
    """#411 review: with `end`, the Wide coverage and its reason come from the logged days the
    folds read. Our logged Wide races up to `end` have no post time; a later logged day, with post
    times, is not read, so it changes neither the reason nor the logged days listed."""
    end = STUDY_DAYS[-2]
    later_day = STUDY_DAYS[-1]
    jravan = {d: day_rows(d, races=2) for d in STUDY_DAYS}
    logged = {d: wide_rows(d, 1, post=False) + wide_rows(d, 2, post=False) for d in STUDY_DAYS[:-1]}
    later = {later_day: wide_rows(later_day, 1) + wide_rows(later_day, 2)}
    without, _ = run_on("to-end", jravan=jravan, logged=logged, end=end)
    with_later, text = run_on("later-log", jravan=jravan, logged=logged | later, end=end)
    assert with_later["wide"]["fitted"] is False
    assert with_later["wide"]["reason"] == without["wide"]["reason"]
    assert with_later["wide"]["reason"] in text
    listed = {e["day"] for e in with_later["wide"]["own_logged_days"]}
    assert listed == {d.isoformat() for d in logged}
