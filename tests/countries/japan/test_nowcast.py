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
- `place_shares(odds, odds_hi) -> (shares, rule_sum, reader)` reads one board as `pool_shares`
  does and names the read: `joint` (both figures), `low_only`, or `undetermined` when the figures
  above the floor cannot tell the money apart (ruling on #410: e.g. two favourites at the floor
  at both ends, whose swap prints the same board); such a board is still read, not dropped.
- `ShadowNowcast` (#116's draws interface) starts a PLACE board's draws from the same shares
  `race_view` reads from the same figures, the high figures included.
- The study's Place measurement (#410): `closing_boards(frame, PLACE)` yields the final board
  each race is scored on; `place_rule_measurement(boards, votes)` reads each as scored and
  validates it against final H1 votes (data_kubun 4 or 5) on every priced runner, counting the
  rest by reason; `final_place_votes(days)` reads `pool_votes` through the scoped store reader
  only, so no holdout row reaches it; an empty measurement is a failed, unvalidated record and
  the report calls the Place rows a diagnostic.
- #410 re-check. Coverage and agreement are separate: `place_result_readers(measured)` is the
  set of determined reads (`joint`, `low_only`) whose validated boards are read within the
  tolerance at least `PLACE_MIN_WITHIN_TOLERANCE` of the time (a module constant in (0, 1]; only
  the clear cases are pinned: all within is a result, none within is not), and
  `place_validated(measured)` is whether that set is non-empty; `undetermined` is never a
  result. The run's `place_measurement` (the JSON dump beside the report) is measured on the
  race ids actually scored (the folds' test cases after test-day sampling). `score_cases`'
  table carries each case's closing-board Place read in a `reader` column (a non-read value
  for WIN); the run's gain tables (`gains`, every cluster unit) leave the `undetermined` Place
  cases out, and the dump reports them separately under `gains_undetermined[<comparison>]`
  (a day-cluster gain table, or a dict holding one under `day`).
- The rank of a Place fit is opt-in: `pool_shares` (and `race_view`, the shadow draws) never
  call a rank routine (`np.linalg.svd`, `matrix_rank`, `cond`, `eig*`); `place_shares` does.
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
- `model.draws_from_shocks(view, *, pool, shocks) -> Draws` (#100) is the same model from shocks
  the caller gives, so the decision layer draws WIN, PLACE and Wide from one set and keeps their
  covariance. `shocks` maps a horse number (int) to its (n,) column of standard normal shocks; a
  WIN or PLACE combination "h" takes shocks[h] as its eps, a Wide pair "a,b"
  (shocks[a] + shocks[b]) / sqrt 2 (as `ShadowNowcast`). n is the columns' length; runners the
  board does not price are ignored. Given the columns of `default_rng(seed).standard_normal((n,
  k))` in the board's order it is `draws(view, pool=, n=, seed=)`. ValueError for a runner of the
  board missing from `shocks`, or columns of unequal lengths.
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
  a JRA-VAN-only run keeps its own family (final-shares-v6 since #410), its spec hash and its
  report.
- Each (race, pool) path is timed by its own source's post time. JRA-VAN wins a (race, pool) only
  where its path has a post time. `wide_coverage` reads the logged days the folds read (`--end`).

Expected values are derived here from first principles (money, the stated formulas, scipy), never
from the module under test. All data are synthetic and in memory or under tmp, and
HORSETRACK_DATA_DIR points at a tmp dir.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import contextlib
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
from horsetrack.countries.japan.jravan.votes import POOL_VOTES_SCHEMA
from horsetrack.countries.japan.logger.compact import SCHEMA, compact_day
from horsetrack.countries.japan.models import nowcast as nc
from horsetrack.countries.japan.models.nowcast import (
    Draws,
    HoldoutDayError,
    Nowcast,
    PoolBoard,
    RaceView,
    ShadowNowcast,
    closing_boards,
    compare,
    extract,
    fit_folds,
    late_composition,
    place_shares,
    pool_shares,
    race_view,
    score_cases,
)
from horsetrack.countries.japan.models.v0.nowcast import DriftCell, NowcastV0
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


# ---------------------------------------------------------------- draws from shared shocks (#100)

SHOCK_HORSES = (1, 2, 3, 5, 6, 7)  # six runners, horse 4 scratched
SHOCK_PAIRS = tuple((a, b) for i, a in enumerate(SHOCK_HORSES) for b in SHOCK_HORSES[i + 1 :])


def shock_view(*, tau: float = 5.0) -> RaceView:
    """A WIN, PLACE and Wide board of six runners, each with a pool total."""
    win = np.array([0.30, 0.22, 0.16, 0.13, 0.11, 0.08])
    place = np.array([0.24, 0.21, 0.17, 0.15, 0.13, 0.10])
    strength = dict(zip(SHOCK_HORSES, place, strict=True))
    wide = np.array([strength[a] * strength[b] for a, b in SHOCK_PAIRS])

    def board(combinations, shares, total):
        return PoolBoard(tuple(combinations), shares / shares.sum(), total, None, None)

    pools = {
        WIN: board((str(h) for h in SHOCK_HORSES), win, 3e7),
        PLACE: board((str(h) for h in SHOCK_HORSES), place, 1.2e7),
        WIDE: board((f"{a},{b}" for a, b in SHOCK_PAIRS), wide, 9e6),
    }
    return RaceView(race_id_of(TEST_DAYS[0], 4), VENUE, 4, None, float(tau), pools)


def runner_shocks(n: int, seed: int, horses=SHOCK_HORSES) -> dict[int, np.ndarray]:
    matrix = np.random.default_rng(seed).standard_normal((n, len(horses)))
    return {h: matrix[:, j].copy() for j, h in enumerate(horses)}


def shock_eps(combinations: Sequence[str], shocks) -> np.ndarray:
    """eps per combination: a runner's own shock, a Wide pair's (eps_a + eps_b) / sqrt 2."""
    columns = []
    for c in combinations:
        horses = [int(h) for h in c.split(",")]
        columns.append(
            sum(np.asarray(shocks[h], dtype=float) for h in horses) / math.sqrt(len(horses))
        )
    return np.column_stack(columns)


def test_draws_from_shocks_take_each_runners_shock(fitted, scored_frame):
    """The known nowcast from given shocks: s_final = phi s_t + (1 - phi) softmax(ln s_t + sigma
    eps), eps a runner's shock (WIN, PLACE) or (eps_a + eps_b) / sqrt 2 (Wide), V_final = V_t / phi,
    one draw per shock; a runner the board does not price is ignored. Then, for the fitted
    nowcast and the known one, the shocks `draws` would draw give `draws` itself."""
    view = shock_view()
    for phi, sigma, n in ((0.6, 0.5, 300), (0.25, 1.1, 40), (1.0, 0.7, 5)):
        model = Nowcast.from_params(phi=phi, sigma=sigma)
        shocks = runner_shocks(n, seed=int(100 * phi + n))
        extra = {**shocks, 4: np.full(n, 9.0), 11: np.full(n, -9.0)}  # not on any board
        for pool in (WIN, PLACE, WIDE):
            board = view.pools[pool]
            s = np.asarray(board.shares, dtype=float)
            eps = shock_eps(board.combinations, shocks)
            want = phi * s + (1.0 - phi) * softmax(np.log(s) + sigma * eps)
            got = model.draws_from_shocks(view, pool=pool, shocks=shocks)
            assert isinstance(got, Draws)
            assert_share_draws(got, n, len(board.combinations))
            np.testing.assert_allclose(got.shares, want, rtol=1e-12, atol=1e-15, err_msg=pool)
            np.testing.assert_allclose(got.v_final, np.full(n, board.total / phi), rtol=1e-12)
            ignored = model.draws_from_shocks(view, pool=pool, shocks=extra)
            np.testing.assert_allclose(ignored.shares, got.shares, rtol=1e-12, atol=1e-15)
        # The tolerance tells (eps_a + eps_b) / sqrt 2 from their sum and their mean.
        pair = model.draws_from_shocks(view, pool=WIDE, shocks=shocks).shares
        for scale in (1.0, 0.5):
            wrong = {h: scale * math.sqrt(2) * x for h, x in shocks.items()}
            s = np.asarray(view.pools[WIDE].shares, dtype=float)
            alt = phi * s + (1 - phi) * softmax(
                np.log(s) + sigma * shock_eps(view.pools[WIDE].combinations, wrong)
            )
            assert phi == 1.0 or not np.allclose(pair, alt, rtol=1e-6, atol=1e-9)

    known = Nowcast.from_params(phi=0.7, sigma=0.4)
    fitted_view = race_view(scored_frame, race_id_of(TEST_DAYS[0], 1), tau_min=5)
    for model, at in ((fitted, fitted_view), (known, view)):
        for pool, seed in ((WIN, 17), (PLACE, 18)):
            combinations = at.pools[pool].combinations
            n = 64
            eps = np.random.default_rng(seed).standard_normal((n, len(combinations)))
            shocks = {int(c): eps[:, j].copy() for j, c in enumerate(combinations)}
            want = model.draws(at, pool=pool, n=n, seed=seed)
            got = model.draws_from_shocks(at, pool=pool, shocks=shocks)
            np.testing.assert_allclose(got.shares, want.shares, rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(got.v_final, want.v_final, rtol=1e-12)


def test_draws_from_shocks_couple_the_pools_runner_by_runner():
    """One set of shocks for every pool: a runner's WIN and PLACE draws move together, and a Wide
    pair's with each of its runners' WIN draws; `draws` with a seed per pool couples nothing."""
    view = shock_view(tau=10.0)
    model = Nowcast.from_params(phi=0.5, sigma=0.8)
    n = 2000
    shocks = runner_shocks(n, seed=85)
    win, place, wide = (
        model.draws_from_shocks(view, pool=pool, shocks=shocks).shares
        for pool in (WIN, PLACE, WIDE)
    )
    alone = model.draws(view, pool=PLACE, n=n, seed=86).shares

    def corr(x, y) -> float:
        return float(np.corrcoef(x, y)[0, 1])

    for j in range(len(SHOCK_HORSES)):
        assert corr(win[:, j], place[:, j]) > 0.8, j
        assert abs(corr(win[:, j], alone[:, j])) < 0.15, j
    for col, (a, b) in enumerate(SHOCK_PAIRS):
        for h in (a, b):
            assert corr(wide[:, col], win[:, SHOCK_HORSES.index(h)]) > 0.3, (a, b, h)


def _ragged():
    shocks = runner_shocks(30, seed=3)
    shocks[3] = shocks[3][:29]
    return shocks


def _without(horse):
    def make():
        shocks = runner_shocks(30, seed=4)
        del shocks[horse]
        return shocks

    return make


@pytest.mark.parametrize(
    "pool, make",
    [
        pytest.param(WIN, _without(7), id="win-runner-missing"),
        pytest.param(PLACE, _without(1), id="place-runner-missing"),
        pytest.param(WIDE, _without(6), id="wide-runner-missing"),  # every pair with 6
        pytest.param(WIN, _ragged, id="win-ragged"),
        pytest.param(PLACE, _ragged, id="place-ragged"),
        pytest.param(WIDE, _ragged, id="wide-ragged"),
    ],
)
def test_draws_from_shocks_refuse_a_missing_runner_or_ragged_shocks(pool, make):
    view = shock_view()
    model = Nowcast.from_params(phi=0.6, sigma=0.5)
    with pytest.raises(ValueError):
        model.draws_from_shocks(view, pool=pool, shocks=make())


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


def write_jravan_store(
    data_dir: Path, days, *, scratched: Sequence[str] = (), custom: dict | None = None
) -> None:
    """The jravan store the study reads: an odds_timeseries day file per day (two races each, with
    runner SCRATCHED of each race in `scratched` scratched at the close; `custom` maps a day to
    the rows written instead) and races.parquet."""
    race_ids = []
    for day in days:
        rows = (custom or {}).get(day) or day_rows(day, races=2)
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


def random_place_shares(rng, k: int) -> np.ndarray:
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
        shares = random_place_shares(rng, k)
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


def fetch_timed(rows: list[dict], lag_s) -> list[dict]:
    """`rows` with fetched_utc = snapshot_utc + `lag_s(row)` seconds, or no fetch time where
    `lag_s(row)` is None."""
    out = []
    for r in rows:
        lag = lag_s(r)
        out.append(
            {
                **r,
                "fetched_utc": None if lag is None else r["snapshot_utc"] + timedelta(seconds=lag),
            }
        )
    return out


PREVIOUS_ROUND = {2: 5, 5: 6}  # the capture round before the one at a grid tau (SNAPSHOTS)


def round_figures(rows: list[dict], race_id: str, pool: str, minute: float):
    """The input's figures of one capture round: {combination: (odds, odds_hi)} and the pool
    total."""
    at = [r for r in rows if r["race_id"] == race_id and r["pool"] == pool
          and r["minutes_to_post"] == minute]  # fmt: skip
    prices = {
        r["combination"]: (r["odds"], r["odds_hi"]) for r in at if r["combination"] != POOL_ROW
    }
    (total,) = [r["investment"] for r in at if r["combination"] == POOL_ROW]
    return prices, total


def test_logged_prices_and_totals_count_from_their_page_fetch():
    """#411 review (round 3): a logged price and pool total count from their page's fetched_utc,
    their snapshot_utc where they have none, as a scratch does, so a capture round that starts at
    a grid cutoff and fetches its pages after it never mixes them into that board. Each round's
    snapshot is on its minute (T-6, T-5, T-2, ...), and a store is logged, compacted and read as
    the study reads a logged day. Per store, the round whose figures the WIN and Wide boards of
    race 1 carry at T-2 and T-5 (odds, high odds and pool total, on `race_view` and on the case):
    - late: every page fetched 20 s after its round starts: each board is the previous round's
      (T-5 at T-2, T-6 at T-5);
    - on time: every page fetched as its round starts: each board is its own round's;
    - no fetch time: timed by snapshot_utc, each board is its own round's;
    - mixed: only the WIN pages of the T-2 round late (30 s): the WIN board at T-2 is the T-5
      round's, the Wide board at T-2 its own round's."""
    day = TEST_DAYS[2]
    race_id = race_id_of(day, 1)
    base = race_rows(np.random.default_rng(4117), day, 1, WIDE_RUNNERS) + wide_rows(day, 1)

    def late_win_at_two(r):
        return 30.0 if r["pool"] == WIN and r["minutes_to_post"] == 2 else 0.0

    stores = {  # name: (lag of a row, {(pool, tau): the round its board carries})
        "late": (lambda r: FETCH_LAG_S, {(WIN, 2): 5, (WIN, 5): 6, (WIDE, 2): 5, (WIDE, 5): 6}),
        "on-time": (lambda r: 0.0, {(WIN, 2): 2, (WIN, 5): 5, (WIDE, 2): 2, (WIDE, 5): 5}),
        "no-fetch-time": (lambda r: None, {(WIN, 2): 2, (WIN, 5): 5, (WIDE, 2): 2, (WIDE, 5): 5}),
        "mixed": (late_win_at_two, {(WIN, 2): 5, (WIN, 5): 5, (WIDE, 2): 2, (WIDE, 5): 5}),
    }
    root = Path(paths.data_dir())
    for name, (lag_s, expected) in stores.items():
        rows = fetch_timed(base, lag_s)
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv(paths.DATA_DIR_VAR, str(root / name))
            write_logged_day(day, rows)
            path = paths.processed_dir("jra") / "odds_timeseries" / f"{day.isoformat()}.parquet"
            own = study.read_logged_day(path, None)
        cases, _, _ = extract(own.frame, scratches=own.scratches)
        for (pool, tau), minute in expected.items():
            prices, total = round_figures(rows, race_id, pool, minute)
            other = round_figures(
                rows, race_id, pool, PREVIOUS_ROUND[tau] if minute == tau else tau
            )
            assert other[1] != total, (name, pool, tau)  # the two candidate rounds differ
            board = race_view(own.frame, race_id, tau_min=tau).pools[pool]
            (i,) = np.flatnonzero((cases.pool == pool) & (cases.tau == tau))
            r = cases.rows(i)
            case_prices = zip(
                cases.combination[r], cases.odds_t[r], cases.odds_hi_t[r], strict=True
            )
            for got, got_total in (
                (zip(board.combinations, board.odds, board.odds_hi, strict=True), board.total),
                (case_prices, cases.total_t[i]),
            ):
                got = {c: (lo, hi) for c, lo, hi in got}
                assert set(got) == set(prices), (name, pool, tau)
                for c, (lo, hi) in prices.items():
                    assert got[c][0] == pytest.approx(lo, rel=1e-9), (name, pool, tau, c)
                    if hi is not None:
                        assert got[c][1] == pytest.approx(hi, rel=1e-9), (name, pool, tau, c)
                assert got_total == pytest.approx(total, rel=1e-9), (name, pool, tau)


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


NO_FOLD = re.compile(r"\bno folds?\b", re.IGNORECASE)


def test_a_run_with_no_fold_completes_and_says_why_wide_is_not_fitted(run_on):
    """#411 review (round 3): a run whose development days end before the first test window
    (`--end` early, or too few weeks for FOLD_SCHEME's min_train of 3) has no fold. JRA-VAN on
    two days of one week and one logged Wide day (post times, final captures) the next week: the
    run completes, reports no fold and no scored case, does not fit Wide, says in its report that
    the run has no fold, and is in the registry. With the rest of JRAVAN_DAYS added the same store
    has a fold, and the reason no longer says the run has none."""
    jravan = {d: day_rows(d, races=2) for d in TRAIN_DAYS[:2]}
    logged = {TRAIN_DAYS[2]: wide_rows(TRAIN_DAYS[2], 1) + wide_rows(TRAIN_DAYS[2], 2)}
    dump, text = run_on("no-fold", jravan=jravan, logged=logged)

    assert dump["counts"]["folds"] == 0 and dump["folds"] == [], dump["folds"]
    assert dump["counts"]["development_days"] == 3
    assert dump["counts"]["own_logged_day_files"] == 1
    assert dump["counts"]["cases"] > 0  # the logged Wide day gave cases, none of them scored
    scored = [r for comparison in dump["gains"].values() for r in comparison.get("day", [])
              if r.get("n_cases")]  # fmt: skip
    assert not scored, scored
    wide = dump["wide"]
    assert wide["fitted"] is False
    assert NO_FOLD.search(wide["reason"]), wide["reason"]
    assert wide["reason"] in text
    records = default_registry(COUNTRY).records()
    assert records and all(r.family_id.startswith(LOGGED_FAMILY) for r in records)

    with_fold = jravan | {d: day_rows(d, races=2) for d in JRAVAN_DAYS[3:]}
    folded, _ = run_on("with-a-fold", jravan=with_fold, logged=logged)
    assert folded["counts"]["folds"] > 0
    assert not NO_FOLD.search(folded["wide"].get("reason") or ""), folded["wide"]


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


JRAVAN_FAMILY = "japan-c1-nowcast-final-shares-v6"  # a JRA-VAN-only run's family since #410
LOGGED_FAMILY = "japan-c1-nowcast-logged-wide-v1"


def jravan_spec_hash() -> str:
    """The hash of the JRA-VAN-only family's spec (`build_spec`) at this run's grid, folds
    (FOLD_SCHEME), bootstrap and drift config, as the registry hashes a spec (sorted, compact
    JSON)."""
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
    """#411 review: a full run on JRA-VAN days alone keeps the JRA-VAN-only family (v6 since
    #410), its spec hash and its report; the same store with our logged days added registers
    under japan-c1-nowcast-logged-wide-v1, frozen with a spec of its own that names both sources
    and extends v6, and writes its report as `nowcast-japan-logged-wide-v1-<spec hash[:12]>.md`,
    leaving the JRA-VAN-only report as it was."""
    reports = tmp_path / "reports"
    data = run_on.store("families", jravan={d: day_rows(d, races=2) for d in STUDY_DAYS}, logged={})
    registry = default_registry(COUNTRY)

    run_on.run(data, trial=False, out_dir=reports)
    jv = run_on.result
    assert jv["family_id"] == JRAVAN_FAMILY
    assert jv["spec_hash"] == jravan_spec_hash() == registry.frozen_spec_hash(JRAVAN_FAMILY)
    first = registry.records()
    assert first and {r.family_id for r in first} == {JRAVAN_FAMILY}
    jv_report = Path(jv["report_path"])
    jv_files = {p: p.read_bytes() for p in (jv_report, jv_report.with_suffix(".json"))}

    for day in STUDY_DAYS:
        write_logged_day(day, wide_rows(day, 1) + wide_rows(day, 2))
    mixed, _ = run_on.run(data, trial=False, out_dir=reports)
    logged = run_on.result
    assert scored_by_pool(mixed).get(WIDE, 0) > 0  # the logged days were read
    assert logged["family_id"] == LOGGED_FAMILY
    assert logged["spec_hash"] != jv["spec_hash"]
    assert registry.frozen_spec_hash(LOGGED_FAMILY) == logged["spec_hash"]
    assert registry.frozen_spec_hash(JRAVAN_FAMILY) == jv["spec_hash"]
    added = registry.records()[len(first) :]
    assert added and {r.family_id for r in added} == {LOGGED_FAMILY}
    logged_report = Path(logged["report_path"])
    assert logged_report.name == f"nowcast-{COUNTRY}-logged-wide-v1-{logged['spec_hash'][:12]}.md"
    assert logged_report not in jv_files
    assert {p: p.read_bytes() for p in jv_files} == jv_files
    spec = frozen_specs(registry)[LOGGED_FAMILY]
    assert spec["family_id"] == LOGGED_FAMILY
    assert spec["extends"].startswith(f"{JRAVAN_FAMILY}:"), spec["extends"][:120]
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


def jra_place_figures(
    money_yen: Sequence[int], *, places: int | None = None
) -> tuple[np.ndarray, np.ndarray, int]:
    """(low, high, places): JRA's printed Place figures for per-runner money in yen. For every
    placed set S of P runners (P = `places_for(starters)`, or `places` when given: the places of
    the field at the start of sales after a scratch), each runner i in S is paid JRA's
    stake-back-share dividend, `core.pricing.stake_back_share_dividend` (pinned by
    tests/core/test_pricing.py): 100 x rate x (1 + D / (P x s_i)) yen per 100-yen ticket, s_i its
    share of the pool, D the share on the runners outside S, rate = 1 - takeout; truncated down to
    10 yen and paid at least the minimum dividend. The low figure is the least of runner i's
    payments over the sets holding it, the high the greatest."""
    money = [int(m) for m in money_yen]
    if places is None:
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


# ---------------------------------------------------------------- #410 review

PLACE_FLOOR = PLACE_POOL.min_dividend / PLACE_UNIT


def place_race_board(money_yen, *, hi_missing: Sequence[int] = ()) -> tuple[RaceBoard, np.ndarray]:
    """A RaceBoard 5 minutes before the post whose PLACE figures JRA prints from `money_yen`
    (runners 1..k, a high figure None on the runners of `hi_missing`), with WIN priced on every
    runner and the Place pool's total; and the high figures as the board holds them (NaN)."""
    lo, hi, _ = jra_place_figures(money_yen)
    hi = hi.copy()
    hi[list(hi_missing)] = np.nan
    post = post_of(TEST_DAYS[0], 4)
    capture = post - timedelta(minutes=5)
    board = RaceBoard(
        race_id=race_id_of(TEST_DAYS[0], 4),
        post_utc=post,
        at_utc=capture,
        capture_utc=capture,
        final=False,
        win={j + 1: 2.0 + j for j in range(len(lo))},
        place={
            j + 1: (float(lo[j]), None if np.isnan(hi[j]) else float(hi[j])) for j in range(len(lo))
        },
        wide={},
        totals={PLACE: float(np.sum(money_yen))},
        scratched=frozenset(),
    )
    return board, hi


def place_view_shares(money_yen, hi) -> np.ndarray:
    """The PLACE shares `race_view` reads 5 minutes before the post from the board JRA prints
    from `money_yen` (high figures `hi`, NaN where absent), in runner order."""
    lo, _, _ = jra_place_figures(money_yen)
    race_id, post = race_id_of(TEST_DAYS[0], 4), post_of(TEST_DAYS[0], 4)
    rows = [
        row(race_id, 4, PLACE, str(j + 1), post, 5, odds=lo[j],
            odds_hi=None if np.isnan(hi[j]) else hi[j])
        for j in range(len(lo))
    ]  # fmt: skip
    rows.append(row(race_id, 4, PLACE, POOL_ROW, post, 5, investment=float(np.sum(money_yen))))
    board = race_view(to_frame(rows), race_id, tau_min=5).pools[PLACE]
    order = np.array([int(c) - 1 for c in board.combinations])
    out = np.empty(len(lo))
    out[order] = board.shares
    return out


@pytest.mark.parametrize(
    ("money", "hi_missing"),
    [
        (PLACE_BOARDS[0][1], ()),  # 14 runners, the favourite's low figure at the floor
        (PLACE_BOARDS[1][1], ()),  # 12 runners, two low figures at the floor
        (PLACE_BOARDS[3][1], ()),  # 8 runners, the favourite at the floor at both ends
        (PLACE_BOARDS[0][1], (5,)),  # a high figure missing: both read the low figures alone
    ],
    ids=["14-favourite-at-floor", "12-two-lows-at-floor", "8-heavy-favourite", "one-high-missing"],
)
def test_shadow_place_draws_start_from_the_same_shares_as_race_view(money, hi_missing):
    """#410 review item 6: the shadow adapter reads a PLACE board's figures as `race_view` does,
    the high figures included. With the known nowcast at phi = 1 every draw is the board, so each
    draw prints the stake-back-share figures (`place_odds`, untruncated, floored) of the shares
    the board is read into; those are the shares `race_view` reads from the same figures, and on
    a board whose favourite is at the floor they are not the low figures' read."""
    board, hi = place_race_board(money, hi_missing=hi_missing)
    lo, _, places = jra_place_figures(money)
    assert places == 3  # place_odds pays three places
    expected = place_view_shares(money, hi)
    np.testing.assert_allclose(
        expected, pool_shares(PLACE, lo, odds_hi=hi)[0], rtol=1e-9, atol=1e-12
    )
    k, n = len(lo), 6
    rng = np.random.default_rng(4106)
    shocks = {j + 1: rng.standard_normal(n) for j in range(k)}
    shadow = ShadowNowcast(Nowcast.from_params(phi=1.0, sigma=1e-6))
    drawn = shadow.draws(board, pool=PLACE, tau_min=5, shocks=shocks)
    assert drawn.combinations == tuple(str(j + 1) for j in range(k))
    want_lo, want_hi = place_odds(expected, 1.0)
    np.testing.assert_allclose(drawn.odds, np.tile(want_lo, (n, 1)), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(drawn.odds_hi, np.tile(want_hi, (n, 1)), rtol=1e-9, atol=1e-9)
    if not hi_missing:  # the low figures' read prints another board
        low_lo, low_hi = place_odds(pool_shares(PLACE, lo)[0], 1.0)
        gap = max(np.abs(want_lo - low_lo).max(), np.abs(want_hi - low_hi).max())
        assert gap > 0.1, gap  # precondition: the two reads differ on this board
        assert np.abs(drawn.odds_hi - low_hi).max() > 0.1 or np.abs(drawn.odds - low_lo).max() > 0.1


# Two favourites of 51% and 40% of the Place money: each is placed with the other in every set
# that prints its figures, so both print the floor at both ends, and every other runner's figures
# hold their money only as a sum; swapping the two prints the same board.
UNIDENTIFIED = _yen([5100, 4000, 250, 200, 150, 150, 100, 50])


def test_place_board_with_unidentified_favourites_is_flagged():
    """#410 review item 7, ruling: a joint read whose figures above the floor cannot tell the
    money apart is flagged `undetermined`, still read (shares total 1, positive, the other runners
    and the two favourites' sum recovered) and not dropped; the known boards, more seeded boards,
    and boards with one favourite at the floor at both ends stay `joint` and recover the money."""
    lo, hi, places = jra_place_figures(UNIDENTIFIED)
    assert places == 3
    assert (lo[:2] == PLACE_FLOOR).all() and (hi[:2] == PLACE_FLOOR).all(), (lo, hi)
    assert (lo[2:] > PLACE_FLOOR).all() and (hi[2:] > PLACE_FLOOR).all(), (lo, hi)
    swapped = UNIDENTIFIED.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    lo_s, hi_s, _ = jra_place_figures(swapped)
    np.testing.assert_array_equal(lo_s, lo)
    np.testing.assert_array_equal(hi_s, hi)

    shares, _, reader = place_shares(lo, hi)
    assert reader == "undetermined"
    assert shares is not None
    shares = np.asarray(shares, dtype=float)
    assert np.isfinite(shares).all() and (shares > 0).all(), shares
    assert shares.sum() == pytest.approx(1.0, abs=1e-9)
    truth = UNIDENTIFIED / UNIDENTIFIED.sum()
    np.testing.assert_allclose(shares[2:], truth[2:], rtol=PLACE_REL_TOL, atol=0)
    assert shares[:2].sum() == pytest.approx(truth[:2].sum(), abs=PLACE_ABS_TOL)
    np.testing.assert_allclose(pool_shares(PLACE, lo, odds_hi=hi)[0], shares, rtol=1e-12)

    known = [money for _, money, _ in PLACE_BOARDS]
    known += [money for _, money in random_place_boards(30, seed=4107)]
    both_ends = 0
    for money in known:
        lo, hi, _ = jra_place_figures(money)
        favourite = int(np.argmax(money))
        both_ends += bool(lo[favourite] == PLACE_FLOOR and hi[favourite] == PLACE_FLOOR)
        shares, _, reader = place_shares(lo, hi)
        assert reader == "joint", (money, lo, hi)
        assert_recovers(shares, money)
    assert both_ends >= 2, both_ends  # boards with one favourite at the floor at both ends


def place_votes_rows(race_id: str, money_yen, *, kubun: str = "4", extra=()) -> list[dict]:
    """A race's PLACE H1 votes (100-yen tickets) on runners 1..k, plus `extra` slots."""
    combos = [str(j + 1) for j in range(len(money_yen))] + list(extra)
    votes = [int(m) // 100 for m in money_yen] + [1000] * len(extra)
    return [
        {"race_id": race_id, "combination": c, "votes": v, "data_kubun": kubun}
        for c, v in zip(combos, votes, strict=True)
    ]


def summary_row(measured: dict, reader: str, label: str = "all") -> dict:
    found = [r for r in measured["summary"] if r["reader"] == reader and r["boards"] == label]
    assert len(found) == 1, measured["summary"]
    return found[0]


def counted(measured: dict, reason: str) -> int:
    return sum(v for k, v in measured["counts"].items() if k.split(":")[0] == reason)


def test_place_rule_measurement_validates_the_boards_as_scored():
    """#410: each closing board is read as scored (`place_shares`) and validated only against
    final H1 votes (data_kubun 4 or 5) on every priced runner; the rest are counted by reason, and
    coverage per read is validated / scored. A board printed from its votes is reproduced and,
    read jointly, within tolerance where the low figures alone are not."""
    rid = [race_id_of(TEST_DAYS[0], n) for n in range(1, 7)]
    fav_floor = PLACE_BOARDS[0][1]  # 14 runners, P = 3, the favourite's low figure at the floor
    two_places = PLACE_BOARDS[4][1]  # 6 runners, P = 2
    twelve = PLACE_BOARDS[1][1]
    boards, votes = [], []

    def add(race_id, money, *, hi_missing=(), kubun="4", extra=(), drop=None, with_votes=True):
        lo, hi, _ = jra_place_figures(money)
        hi = hi.copy()
        hi[list(hi_missing)] = np.nan
        boards.append((race_id, tuple(str(j + 1) for j in range(len(lo))), lo, hi))
        if with_votes:
            rows = place_votes_rows(race_id, money, kubun=kubun, extra=extra)
            votes.extend(r for r in rows if r["combination"] != drop)

    add(rid[0], fav_floor)  # (a) validated, joint
    add(rid[1], two_places, kubun="5", extra=("7", "8"))  # validated, joint, P as scored = 2
    add(rid[2], twelve, hi_missing=(4,))  # (b) validated, low_only, every runner kept
    add(rid[3], fav_floor, kubun="2")  # (c) the day-before close
    add(rid[4], twelve, drop="3")  # (d) a priced runner without votes
    add(rid[5], two_places, with_votes=False)  # no votes for the race
    measured = study.place_rule_measurement(boards, pd.DataFrame(votes))

    assert measured["counts"]["boards_scored"] == 6
    assert measured["counts"]["boards_validated"] == 3
    assert counted(measured, "unavailable_votes_not_final") == 1
    assert counted(measured, "unavailable_votes_missing_on_a_priced_runner") == 1
    assert counted(measured, "unavailable_no_votes") == 1
    assert study.place_validated(measured)

    joint = summary_row(measured, "joint")
    assert (joint["n_scored"], joint["n"]) == (5, 2)
    assert joint["coverage"] == pytest.approx(2 / 5)
    assert joint["share_reproduced"] == pytest.approx(1.0)
    p3 = summary_row(measured, "joint", "P=3")
    assert (p3["n_scored"], p3["n"], p3["coverage"]) == (3, 1, pytest.approx(1 / 3))
    assert p3["share_within_tolerance"] == pytest.approx(1.0)
    assert p3["low_only_share_within_tolerance"] == pytest.approx(0.0)
    p2 = summary_row(measured, "joint", "P=2")
    assert (p2["n_scored"], p2["n"]) == (2, 1)
    assert p2["share_reproduced"] == pytest.approx(1.0)  # with P as scored, not from 8 slots
    assert p2["share_slot_places_differ"] == pytest.approx(1.0)

    low = summary_row(measured, "low_only")
    assert (low["n_scored"], low["n"], low["coverage"]) == (1, 1, pytest.approx(1.0))
    assert low["share_reproduced"] == pytest.approx(1.0)
    assert summary_row(measured, "low_only", "P=3")["n"] == 1

    empty = study.place_rule_measurement(
        boards, pd.DataFrame(columns=["race_id", "combination", "votes", "data_kubun"])
    )
    assert empty["counts"]["boards_scored"] == 6
    assert empty["counts"]["boards_validated"] == 0
    assert counted(empty, "unavailable_no_votes") == 6
    assert not study.place_validated(empty)
    for r in empty["summary"]:
        assert r["n"] == 0 and r["coverage"] == 0.0, r


def test_place_rule_measurement_counts_an_undetermined_board_under_its_own_reader():
    """Ruling on #410 review item 7: an undetermined board is a diagnostic in the measurement,
    counted under its own read and not as a joint one, even with final votes on every runner."""
    lo, hi, _ = jra_place_figures(UNIDENTIFIED)
    fav_lo, fav_hi, _ = jra_place_figures(PLACE_BOARDS[0][1])
    rid = [race_id_of(TEST_DAYS[0], n) for n in (1, 2)]
    boards = [
        (rid[0], tuple(str(j + 1) for j in range(len(lo))), lo, hi),
        (rid[1], tuple(str(j + 1) for j in range(len(fav_lo))), fav_lo, fav_hi),
    ]
    votes = place_votes_rows(rid[0], UNIDENTIFIED) + place_votes_rows(rid[1], PLACE_BOARDS[0][1])
    measured = study.place_rule_measurement(boards, pd.DataFrame(votes))
    assert measured["counts"]["boards_validated"] == 2
    undetermined = summary_row(measured, "undetermined")
    assert (undetermined["n_scored"], undetermined["n"]) == (1, 1)
    assert undetermined["share_reproduced"] == pytest.approx(1.0)
    joint = summary_row(measured, "joint")
    assert (joint["n_scored"], joint["n"]) == (1, 1)


def write_pool_votes(data_dir: Path, rows: list[dict]) -> Path:
    path = data_dir / "processed" / "jravan" / "pool_votes.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=POOL_VOTES_SCHEMA), path)
    return path


def store_votes(day: date, race_no: int, pool: str, votes: dict[str, int], kubun="4") -> list[dict]:
    race_id = race_id_of(day, race_no)
    return [
        {
            "jurisdiction": "jra",
            "race_id": race_id,
            "race_date": day,
            "venue": VENUE,
            "race_no": race_no,
            "pool": pool,
            "combination": c,
            "votes": v,
            "refund_votes": 0,
            "fav_rank": None,
            "data_kubun": kubun,
        }
        for c, v in votes.items()
    ]


def test_place_votes_are_read_through_the_scoped_reader_and_drop_holdout_days(
    data_dir, monkeypatch
):
    """D14, #415: `final_place_votes` reads `pool_votes` only through the store's scoped reader,
    so a holdout day's votes never reach the study even when that day is asked for; only the
    PLACE runner slots of the days asked are returned, with their data_kubun."""
    dev, held, other = date(2026, 11, 28), date(2026, 11, 29), date(2026, 11, 22)
    assert not is_holdout_day(COUNTRY, dev) and not is_holdout_day(COUNTRY, other)
    assert is_holdout_day(COUNTRY, held)
    dev_place = {"1": 4100, "2": 900, "3": 350}
    rows = (
        store_votes(dev, 1, PLACE, dev_place)
        + store_votes(dev, 2, PLACE, {"1": 700, "2": 650}, kubun="2")
        + store_votes(dev, 1, PLACE, {POOL_ROW: 5350})
        + store_votes(dev, 1, WIN, {"1": 9000, "2": 100})
        + store_votes(held, 1, PLACE, {"1": 1234, "2": 4321})
        + store_votes(other, 1, PLACE, {"1": 55, "2": 66})
    )
    path = write_pool_votes(data_dir, rows)

    inside, outside, scoped = [0], [], []
    real_reader = study.read_store_table

    def reader(name, *args, **kwargs):
        scoped.append(name)
        inside[0] += 1
        try:
            return real_reader(name, *args, **kwargs)
        finally:
            inside[0] -= 1

    monkeypatch.setattr(study, "read_store_table", reader)
    import builtins

    for module, name in ((pq, "read_table"), (pq, "ParquetFile"), (pq, "read_metadata"),
                         (pq, "read_schema"), (ds, "dataset"), (pa, "memory_map"),
                         (pd, "read_parquet"), (builtins, "open")):  # fmt: skip
        real = getattr(module, name)

        def spy(source, *args, _real=real, **kwargs):
            if "pool_votes" in str(source) and not inside[0]:
                outside.append(str(source))
            return _real(source, *args, **kwargs)

        monkeypatch.setattr(module, name, spy)

    got = study.final_place_votes([dev.isoformat(), held.isoformat()])
    monkeypatch.undo()
    assert not outside, outside
    assert "pool_votes" in scoped
    assert path.exists()
    races = set(got["race_id"])
    assert races == {race_id_of(dev, 1), race_id_of(dev, 2)}, races
    assert not any(held.isoformat() in r for r in got["race_id"])
    assert POOL_ROW not in set(got["combination"])
    first = got[got["race_id"] == race_id_of(dev, 1)]
    assert dict(zip(first["combination"], first["votes"].astype(int), strict=True)) == dev_place
    kubun = dict(zip(got["race_id"], got["data_kubun"], strict=False))
    assert kubun == {race_id_of(dev, 1): "4", race_id_of(dev, 2): "2"}


def test_an_empty_place_measurement_is_an_unvalidated_failed_record(
    data_dir, tmp_path, monkeypatch
):
    """#410: with no votes for the scored boards the measurement validates none; its registry
    record (variant place_rule_vs_votes) is failed and marked unvalidated, `records_appended`
    counts it, and the report reads the Place rows as a diagnostic."""
    write_jravan_store(data_dir, TRAIN_DAYS + TEST_DAYS[:1])
    write_pool_votes(data_dir, [])
    monkeypatch.setattr(study, "load_country_scheme", lambda country: FOLD_SCHEME)
    summary = tmp_path / "summary"
    monkeypatch.setattr(study, "summary_dir", lambda country: summary)
    monkeypatch.setattr("horsetrack.core.eval.registry.summary_dir", lambda country: summary)
    result = study.run_study(
        trial=True, test_days_per_fold=1, n_draws=30, out_dir=tmp_path / "out", log=lambda *a: None
    )
    records = default_registry(COUNTRY).records()
    place = [r for r in records if r.variant == "place_rule_vs_votes"]
    assert len(place) == 1, [r.variant for r in records]
    assert place[0].status == "failed"
    assert place[0].params.get("unvalidated") is True
    assert result["records_appended"] == len(records)
    report = Path(result["report_path"]).read_text()
    lines = [ln for ln in report.splitlines() if "Place" in ln and "diagnostic" in ln.lower()]
    assert lines, "the report does not call the Place rows a diagnostic"


def test_closing_boards_are_the_final_boards_with_their_figures():
    """`closing_boards` yields each race's final PLACE board as scored: each combination's latest
    row, only the priced ones (a runner scratched at the close is off it), a missing high figure
    kept as NaN; races without a post time are skipped and other pools ignored."""
    day = TEST_DAYS[0]
    rows = []
    expected = {}
    for race_no, k in ((1, 8), (2, 10)):
        race_id, post = race_id_of(day, race_no), post_of(day, race_no)
        rng = np.random.default_rng(race_no)
        early = rng.uniform(1.2, 9.0, size=(k, 2))
        late = rng.uniform(1.2, 9.0, size=(k, 2))
        for j in range(k):
            c = str(j + 1)
            rows.append(row(race_id, race_no, PLACE, c, post, 10, odds=early[j, 0],
                            odds_hi=early[j, 0] + early[j, 1]))  # fmt: skip
            rows.append(row(race_id, race_no, WIN, c, post, 2, odds=3.0 + j, final=True))
        lo, hi = late[:, 0], late[:, 0] + late[:, 1]
        hi[1] = np.nan  # runner 2 prints no high figure at the close
        keep = []
        for j in range(k):
            if j == 2:  # runner 3 scratched at the close: a row with no price
                rows.append(row(race_id, race_no, PLACE, str(j + 1), post, -2, final=True))
                continue
            if j == 4:  # runner 5's latest row is the early one
                keep.append((str(j + 1), early[j, 0], early[j, 0] + early[j, 1]))
                continue
            rows.append(row(race_id, race_no, PLACE, str(j + 1), post, -2, odds=lo[j],
                            odds_hi=None if np.isnan(hi[j]) else hi[j], final=True))  # fmt: skip
            keep.append((str(j + 1), lo[j], hi[j]))
        rows.append(row(race_id, race_no, PLACE, POOL_ROW, post, -2, investment=5e7, final=True))
        expected[race_id] = sorted(keep, key=lambda t: int(t[0]))
    no_post = {**row(race_id_of(day, 3), 3, PLACE, "1", post_of(day, 3), 5, odds=2.0, odds_hi=3.0)}
    rows.append({**no_post, "post_utc": None, "minutes_to_post": None})

    got = {race_id: (combos, lo, hi) for race_id, combos, lo, hi, *_ in closing_boards(
        to_frame(rows), PLACE)}  # fmt: skip
    assert set(got) == set(expected)
    for race_id, keep in expected.items():
        combos, lo, hi = got[race_id]
        assert tuple(combos) == tuple(c for c, _, _ in keep)
        np.testing.assert_allclose(np.asarray(lo, float), [x for _, x, _ in keep], rtol=1e-12)
        np.testing.assert_allclose(
            np.asarray(hi, float), [x for _, _, x in keep], rtol=1e-12, equal_nan=True
        )
        assert np.isnan(np.asarray(hi, float)[combos.index("2")])


# ---------------------------------------------------------------- #410 re-check

PLACE_READS = ("joint", "low_only", "undetermined")


def place_board(race_id: str, money_yen, *, hi_missing: Sequence[int] = ()) -> tuple:
    """(race_id, combinations, low, high) of the closing board JRA prints from `money_yen`, a high
    figure NaN on the runners of `hi_missing`."""
    lo, hi, _ = jra_place_figures(money_yen)
    hi = hi.copy()
    hi[list(hi_missing)] = np.nan
    return race_id, tuple(str(j + 1) for j in range(len(lo))), lo, hi


def test_place_is_a_result_only_for_a_read_that_recovers_the_money():
    """Re-check 1: coverage (final votes on a determined board) is not agreement. Place stands
    as a result only for a determined read whose validated boards are read within the tolerance;
    the floored favourite with one high figure missing (read `low_only`, every figure reproduced
    from the votes, shares 17-135% off) is a diagnostic, while the same board with all its high
    figures (`joint`) and an open race read `low_only` (no floor, within) are results."""
    rid = [race_id_of(TEST_DAYS[0], n) for n in range(1, 5)]
    money = {
        rid[0]: PLACE_BOARDS[0][1],  # 14 runners, the favourite's low figure at the floor
        rid[1]: PLACE_BOARDS[0][1],
        rid[2]: PLACE_BOARDS[2][1],  # 16 runners, no figure at the floor
        rid[3]: UNIDENTIFIED,
    }
    floored_low = place_board(rid[0], money[rid[0]], hi_missing=(5,))
    joint = place_board(rid[1], money[rid[1]])
    open_low = place_board(rid[2], money[rid[2]], hi_missing=(3,))
    undetermined = place_board(rid[3], money[rid[3]])

    def measure(*boards):
        votes = [r for b in boards for r in place_votes_rows(b[0], money[b[0]])]
        return study.place_rule_measurement(list(boards), pd.DataFrame(votes))

    only = measure(floored_low)
    low = summary_row(only, "low_only")
    assert (low["n_scored"], low["n"], low["coverage"]) == (1, 1, pytest.approx(1.0))
    assert low["share_reproduced"] == pytest.approx(1.0)  # precondition: the rule holds
    assert low["share_within_tolerance"] == pytest.approx(0.0)  # and the read misses
    assert not study.place_validated(only), low
    assert "diagnostic" in study._place_status(only).lower()

    cases = [
        ((floored_low,), set()),
        ((joint,), {"joint"}),
        ((open_low,), {"low_only"}),
        ((joint, floored_low), {"joint"}),
        ((open_low, joint), {"joint", "low_only"}),
        ((undetermined,), set()),
        ((undetermined, floored_low), set()),
    ]
    for boards, want in cases:
        measured = measure(*boards)
        names = [b[0][-2:] for b in boards]
        assert study.place_validated(measured) is bool(want), (names, measured["summary"])
        assert set(study.place_result_readers(measured)) == want, names
    assert 0.0 < float(study.PLACE_MIN_WITHIN_TOLERANCE) <= 1.0


def final_place_money(day: date, races: int = 2) -> dict[str, tuple[float, np.ndarray]]:
    """(V_final, final PLACE shares) of each race `day_rows(day, races=races)` writes, by
    replaying its seeded draws in the same order."""
    rng = np.random.default_rng(day.toordinal())
    out = {}
    for race_no in range(1, races + 1):
        k = RUNNERS[(race_no - 1) % len(RUNNERS)]
        for pool in (WIN, PLACE):
            rate = GROWTH[pool] * (1.0 + 0.05 * rng.uniform(-1.0, 1.0))
            v_final = 1e8 * math.exp(0.3 * rng.standard_normal())
            path = pool_path(rng, k, rate, 0.3, v_final)
            if pool == PLACE:
                out[race_id_of(day, race_no)] = path[-1]
    return out


def final_votes_of(days) -> list[dict]:
    """Final PLACE H1 votes (100-yen tickets) of every race the store writes on `days`."""
    rows = []
    for day in days:
        for race_id, (total, shares) in final_place_money(day).items():
            votes = np.maximum(np.round(shares * total / 100.0), 1).astype(int)
            race_no = int(race_id.split(":")[-1])
            rows += store_votes(
                day, race_no, PLACE, {str(j + 1): int(v) for j, v in enumerate(votes)}
            )
    return rows


def run_trial(data_dir, tmp_path, monkeypatch, out: str) -> tuple[dict, dict]:
    """A trial run of the study (one fold, one sampled test day, 20 draws): its result and its
    JSON dump."""
    monkeypatch.setattr(study, "load_country_scheme", lambda country: FOLD_SCHEME)
    summary = tmp_path / "summary"
    monkeypatch.setattr(study, "summary_dir", lambda country: summary)
    monkeypatch.setattr("horsetrack.core.eval.registry.summary_dir", lambda country: summary)
    result = study.run_study(
        trial=True, test_days_per_fold=1, n_draws=20, out_dir=tmp_path / out, log=lambda *a: None
    )
    dump = json.loads(Path(result["report_path"]).with_suffix(".json").read_text())
    return result, dump


def place_records() -> list:
    return [r for r in default_registry(COUNTRY).records() if r.variant == "place_rule_vs_votes"]


def test_place_coverage_is_measured_on_the_scored_cases(data_dir, tmp_path, monkeypatch):
    """Re-check 2: the measurement's boards are the races actually scored, the fold's test cases
    after test-day sampling, not every extracted Place race. One fold tests two days and samples
    one (two races each). Final votes on the training days alone cover no scored board, so
    Place is not a result; votes on one test day are a result exactly when that day is the one
    sampled, and every run measures the two scored boards. The synthetic closing boards read
    `joint` within the tolerance of their money, so coverage is what decides."""
    write_jravan_store(data_dir, TRAIN_DAYS + TEST_DAYS[:2])
    runs = {}
    for name, days in (
        ("train", TRAIN_DAYS),
        ("first", TEST_DAYS[:1]),
        ("second", TEST_DAYS[1:2]),
    ):
        write_pool_votes(data_dir, final_votes_of(days))
        before = len(place_records())
        _, dump = run_trial(data_dir, tmp_path, monkeypatch, f"out-{name}")
        new = place_records()[before:]
        assert len(new) == 1, [r.params for r in new]
        measured = dump["place_measurement"]
        assert measured["counts"]["boards_scored"] == 2, (name, measured["counts"])
        scored = sum(r["n_scored"] for r in measured["summary"] if r["boards"] == "all")
        assert scored == 2, (name, measured["summary"])
        runs[name] = (new[0], measured)

    record, measured = runs["train"]
    assert measured["counts"]["boards_validated"] == 0, measured["counts"]
    assert all(r["coverage"] == 0.0 for r in measured["summary"]), measured["summary"]
    assert not study.place_validated(measured)
    assert record.status == "failed" and record.params.get("unvalidated") is True

    results = {name: runs[name][0].status == "scored" for name in ("first", "second")}
    assert sorted(results.values()) == [False, True], results
    for name, is_result in results.items():
        record, measured = runs[name]
        assert measured["counts"]["boards_validated"] == (2 if is_result else 0), measured
        assert study.place_validated(measured) is is_result
        if not is_result:
            assert record.params.get("unvalidated") is True


def undetermined_race_rows(day: date, race_no: int) -> list[dict]:
    """A race whose PLACE board prints UNIDENTIFIED's figures at every snapshot (two favourites
    at the floor at both ends: read `undetermined`), its pool growing as GROWTH[PLACE]; WIN priced
    from the same composition."""
    race_id, post = race_id_of(day, race_no), post_of(day, race_no)
    lo, hi, _ = jra_place_figures(UNIDENTIFIED)
    share = UNIDENTIFIED / UNIDENTIFIED.sum()
    rows = []
    for pool in (WIN, PLACE):
        for minutes in SNAPSHOTS:
            final = minutes == FINAL_MIN
            total = 1e8 * (1.0 if final else math.exp(-GROWTH[pool] * minutes))
            for j in range(len(share)):
                odds = (1.0 - TAKEOUT) / share[j] if pool == WIN else lo[j]
                rows.append(row(race_id, race_no, pool, str(j + 1), post, minutes, odds=odds,
                                odds_hi=None if pool == WIN else hi[j], final=final))  # fmt: skip
            rows.append(
                row(race_id, race_no, pool, POOL_ROW, post, minutes, investment=total, final=final)
            )
    return rows


def day_with_an_undetermined_race(day: date) -> list[dict]:
    return day_rows(day, races=1) + undetermined_race_rows(day, 2)


def board_forecaster(view, *, pool, n, seed) -> Draws:
    shares = np.asarray(view.pools[pool].shares, dtype=float)
    return Draws(shares=np.tile(shares, (n, 1)), v_final=np.full(n, 1e8))


def test_score_rows_carry_the_closing_board_place_read():
    """Re-check 3: every scored Place case carries the read of its race's closing board (the
    board `closing_boards` yields, read by `place_shares`); WIN cases carry no Place read."""
    day = TEST_DAYS[0]
    frame = to_frame(day_with_an_undetermined_race(day))
    closing = {rid: place_shares(lo, hi)[2] for rid, _, lo, hi, *_ in closing_boards(frame, PLACE)}
    assert closing == {race_id_of(day, 1): "joint", race_id_of(day, 2): "undetermined"}, closing
    cases, _, _ = extract(frame)
    table, _ = score_cases({"board": board_forecaster}, cases, n=4, seed=410)
    assert "reader" in table.columns, list(table.columns)
    place = table[table["pool"] == PLACE]
    assert set(place["race_id"]) == set(closing)
    for race_id, g in place.groupby("race_id"):
        assert len(g) == len(GRID)
        assert set(g["reader"]) == {closing[race_id]}, (race_id, set(g["reader"]))
    win = table[table["pool"] == WIN]
    assert len(win) == 2 * len(GRID)
    assert not win["reader"].isin(PLACE_READS).any(), set(win["reader"])


def n_cases_by_tau(rows: list[dict], pool: str) -> dict[float, int]:
    return {float(r["tau_min"]): int(r["n_cases"]) for r in rows if r["pool"] == pool}


def test_undetermined_place_cases_are_kept_out_of_the_place_gains(data_dir, tmp_path, monkeypatch):
    """Re-check 3: the Place gains reported as the result (day and meeting clusters, and by
    fold) leave out the cases whose closing board is read `undetermined`; those are reported
    separately, under `gains_undetermined`. The scored day has a joint race and an undetermined
    one, so each tau has two WIN cases, one Place case in the result and one in the diagnostic."""
    write_jravan_store(
        data_dir,
        TRAIN_DAYS + TEST_DAYS[:1],
        custom={TEST_DAYS[0]: day_with_an_undetermined_race(TEST_DAYS[0])},
    )
    write_pool_votes(data_dir, [])
    _, dump = run_trial(data_dir, tmp_path, monkeypatch, "out")
    grid = {float(t) for t in GRID}
    primary = float(study.load_decision_grid().primary)
    assert set(dump["gains"]) == {f"{a}_vs_{b}" for a, b in study.COMPARISONS}
    for name, by_unit in dump["gains"].items():
        for unit in ("day", "meeting"):
            assert n_cases_by_tau(by_unit[unit], WIN) == dict.fromkeys(grid, 2), (name, unit)
            assert n_cases_by_tau(by_unit[unit], PLACE) == dict.fromkeys(grid, 1), (name, unit)
        for _, rows in by_unit["fold"]:
            assert n_cases_by_tau([r | {"tau_min": primary} for r in rows], PLACE) == {
                primary: 1
            }, name
        diagnostic = dump["gains_undetermined"][name]
        table = diagnostic["day"] if isinstance(diagnostic, dict) else diagnostic
        assert n_cases_by_tau(table, PLACE) == dict.fromkeys(grid, 1), (name, table)
        assert not n_cases_by_tau(table, WIN), name


RANK_ROUTINES = ("svd", "matrix_rank", "cond", "eig", "eigh", "eigvals", "eigvalsh")


def test_the_place_rank_is_computed_only_when_the_read_is_asked_for(monkeypatch):
    """Re-check 4: `pool_shares` (one board or a batch of rows, as the draw repricing reads
    them), `race_view` and the shadow draws read Place shares without computing the fit's rank;
    their shares are those `place_shares` reads, which still computes it to name the read."""
    known = (PLACE_BOARDS[0][1], PLACE_BOARDS[1][1], PLACE_BOARDS[3][1], UNIDENTIFIED)
    boards = [jra_place_figures(m)[:2] for m in known]
    want = [place_shares(lo, hi) for lo, hi in boards]
    rng = np.random.default_rng(4108)
    base = PLACE_BOARDS[0][1].astype(float)
    batch = [
        jra_place_figures(
            (np.maximum(np.round(base * rng.uniform(0.8, 1.25, len(base)) / 100), 1) * 100).astype(
                np.int64
            )
        )[:2]
        for _ in range(5)
    ]
    lo2, hi2 = np.vstack([b[0] for b in batch]), np.vstack([b[1] for b in batch])
    want2 = np.vstack([place_shares(lo, hi)[0] for lo, hi in batch])
    view_money = PLACE_BOARDS[1][1]
    view_hi = jra_place_figures(view_money)[1]
    want_view = place_shares(*jra_place_figures(view_money)[:2])[0]
    board, _ = place_race_board(PLACE_BOARDS[0][1])
    shadow = ShadowNowcast(Nowcast.from_params(phi=0.8, sigma=0.3))
    shocks = {j + 1: np.random.default_rng(j).standard_normal(4) for j in range(len(base))}

    calls = []
    for name in RANK_ROUTINES:
        real = getattr(np.linalg, name)

        def spy(*args, _real=real, _name=name, **kwargs):
            calls.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(np.linalg, name, spy)

    for (lo, hi), (shares, rule_sum, _) in zip(boards, want, strict=True):
        got, got_sum = pool_shares(PLACE, lo, odds_hi=hi)
        np.testing.assert_allclose(got, shares, rtol=1e-12, atol=0)
        assert got_sum == pytest.approx(rule_sum, rel=1e-12)
    got2, _ = pool_shares(PLACE, lo2, odds_hi=hi2)
    np.testing.assert_allclose(got2, want2, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(place_view_shares(view_money, view_hi), want_view, rtol=1e-9)
    shadow.draws(board, pool=PLACE, tau_min=5, shocks=shocks)
    assert not calls, f"the numeric Place read computed a rank: {calls}"

    _, _, reader = place_shares(*boards[3])
    assert reader == "undetermined"
    assert calls, "place_shares named the read without a rank routine of np.linalg"


# ---------------------------------------------------------------- round 5 (#410 review)


def holdout_days(n: int) -> list[date]:
    """The first `n` holdout days from the holdout start, by the committed rule."""
    start = date(2026, 11, 28)
    days = [start + timedelta(days=i) for i in range(120)]
    held = [d for d in days if is_holdout_day(COUNTRY, d)][:n]
    assert len(held) == n
    return held


def registry_run_ids() -> dict[str, str]:
    """{variant: run_id} of the registry's run entries, read from its file."""
    path = default_registry(COUNTRY).path
    out = {}
    for raw in path.read_text().splitlines():
        entry = json.loads(raw)
        if entry.get("kind") == "run":
            out[entry["record"]["variant"]] = entry["run_id"]
    return out


def votes_build_run(data_dir, tmp_path, monkeypatch, name: str, rows: list[dict]) -> dict:
    """A trial run on `rows` as pool_votes, on a fresh registry: its JSON dump, Markdown, records
    and run ids by variant."""
    write_pool_votes(data_dir, rows)
    result, dump = run_trial(data_dir, tmp_path, monkeypatch, f"out-{name}")
    registry = default_registry(COUNTRY)
    records = {r.variant: r for r in registry.records()}
    ids = registry_run_ids()
    registry.path.unlink()
    return {
        "dump": dump,
        "markdown": Path(result["report_path"]).read_text(),
        "records": records,
        "run_ids": ids,
    }


def test_holdout_votes_never_enter_the_votes_build(data_dir, tmp_path, monkeypatch):
    """Round 5: the final votes' build the study records (`jravan_pool_votes` in every
    comparison record's data builds, and in the report's data line) is taken from the vote rows
    the measurement reads, not the whole pool_votes file. Adding holdout-day votes, or changing
    only them, leaves the build, the records and their registry run ids as they were; changing
    a development vote that the measurement reads changes the build."""
    write_jravan_store(data_dir, TRAIN_DAYS + TEST_DAYS[:1])
    dev_days = TRAIN_DAYS + TEST_DAYS[:1]
    base = final_votes_of(dev_days)
    held = holdout_days(2)
    held_v1 = store_votes(held[0], 1, PLACE, {"1": 4100, "2": 900, "3": 350}) + store_votes(
        held[0], 2, PLACE, {"1": 700, "2": 650, "3": 120}
    )
    held_v2 = (
        store_votes(held[0], 1, PLACE, {"1": 3999, "2": 1001, "3": 351})
        + store_votes(held[0], 2, PLACE, {"1": 701, "2": 649, "3": 121})
        + store_votes(held[1], 1, PLACE, {"1": 55, "2": 66, "3": 77})
    )
    changed = [
        r | {"votes": r["votes"] + 7} if r["combination"] == "1" and r["race_no"] == 1 else r
        for r in base
    ]
    assert changed != base
    runs = {
        name: votes_build_run(data_dir, tmp_path, monkeypatch, name, rows)
        for name, rows in (
            ("base", base),
            ("held_added", base + held_v1),
            ("held_changed", base + held_v2),
            ("dev_changed", changed),
        )
    }

    comparisons = {f"{a}_vs_{b}" for a, b in study.COMPARISONS}
    builds = {}
    for name, run in runs.items():
        assert comparisons <= set(run["records"]), (name, sorted(run["records"]))
        votes = {run["records"][c].data_builds.get("jravan_pool_votes") for c in comparisons}
        assert len(votes) == 1 and None not in votes, (name, votes)
        builds[name] = votes.pop()
        assert builds[name] in run["dump"]["about"]["data"], (name, run["dump"]["about"])
        assert builds[name] in run["markdown"], name

    for name in ("held_added", "held_changed"):
        assert builds[name] == builds["base"], (name, builds)
        assert runs[name]["dump"]["about"]["data"] == runs["base"]["dump"]["about"]["data"]
        for variant in comparisons | {study.PLACE_VARIANT}:
            assert runs[name]["records"][variant] == runs["base"]["records"][variant], variant
            assert runs[name]["run_ids"][variant] == runs["base"]["run_ids"][variant], variant
    assert builds["dev_changed"] != builds["base"], builds


def pit_rows_of(dump: dict, key: str, pool: str) -> dict[tuple, dict]:
    """{(forecaster, tau_min[, reader]): row} of a dump's PIT rows for `pool`."""
    out = {}
    for row in dump.get(key, []):
        if row["pool"] != pool:
            continue
        ident = (row["forecaster"], float(row["tau_min"]))
        if "reader" in row:
            ident += (row["reader"],)
        out[ident] = row
    return out


def assert_same_pit(got: dict, want: dict) -> None:
    assert set(got) == set(want), (sorted(got), sorted(want))
    for key, row in want.items():
        other = got[key]
        assert other["n"] == row["n"], (key, other["n"], row["n"])
        for c in ("coverage_50", "coverage_90"):
            assert other[c] == pytest.approx(row[c], abs=1e-12), (key, c)
        np.testing.assert_allclose(other["deciles"], row["deciles"], atol=1e-9, err_msg=str(key))


def markdown_tables(markdown: str) -> list[tuple[str, list[dict]]]:
    """(the heading above, rows) of every table in a report."""
    out, heading, block = [], "", []

    def close():
        if len(block) >= 2:
            head = [c.strip() for c in block[0].strip("|").split("|")]
            rows = [
                dict(zip(head, [c.strip() for c in line.strip("|").split("|")], strict=False))
                for line in block[2:]
            ]
            out.append((heading, rows))
        block.clear()

    for line in markdown.splitlines():
        if line.startswith("|"):
            block.append(line.strip())
            continue
        close()
        if line.startswith("#"):
            heading = line
    close()
    return out


def test_place_pit_is_split_by_closing_board_read(data_dir, tmp_path, monkeypatch):
    """Round 5: the report's PIT deciles and 50% / 90% coverage follow the same split as the
    gains. The scored day holds a `joint` race (its final votes validate it, so `joint` stands)
    and an `undetermined` one. The result PIT (`pit` in the JSON) holds the Place cases of the
    reads that stand only: it equals the Place PIT of the same run with the undetermined race
    removed from the scored day (every case is drawn with its own seed). The undetermined
    board's PIT is reported apart, as `pit_diagnostic` rows carrying `reader`, equal to the run
    whose scored day holds that race alone. The Markdown shows both: the result PIT table's
    Place rows have the result counts, and a PIT table under a heading marked diagnostic has
    the undetermined rows."""
    day = TEST_DAYS[0]
    total, shares = final_place_money(day, races=1)[race_id_of(day, 1)]
    joint = np.maximum(np.round(shares * total / 100.0), 1).astype(int)
    votes = final_votes_of(TRAIN_DAYS)
    votes += store_votes(day, 1, PLACE, {str(j + 1): int(x) for j, x in enumerate(joint)})
    votes += store_votes(
        day, 2, PLACE, {str(j + 1): int(m // 100) for j, m in enumerate(UNIDENTIFIED)}
    )
    layouts = {
        "both": day_with_an_undetermined_race(day),
        "joint_only": day_rows(day, races=1),
        "undetermined_only": undetermined_race_rows(day, 2),
    }
    dumps, markdowns = {}, {}
    for name, rows in layouts.items():
        write_jravan_store(data_dir, TRAIN_DAYS + (day,), custom={day: rows})
        write_pool_votes(data_dir, votes)
        result, dumps[name] = run_trial(data_dir, tmp_path, monkeypatch, f"out-{name}")
        markdowns[name] = Path(result["report_path"]).read_text()
        default_registry(COUNTRY).path.unlink()

    both = dumps["both"]
    assert both["place_result_readers"] == ["joint"], both["place_measurement"]
    result = pit_rows_of(both, "pit", PLACE)
    assert result, both.get("pit")
    assert not any(len(k) == 3 and k[2] == "undetermined" for k in result), sorted(result)
    assert_same_pit(
        {k[:2]: v for k, v in result.items()},
        {k[:2]: v for k, v in pit_rows_of(dumps["joint_only"], "pit", PLACE).items()},
    )
    # every grid tau of the scored day's joint race counts its runners
    k_joint = RUNNERS[0]
    for (forecaster, tau), row in {k[:2]: v for k, v in result.items()}.items():
        assert row["n"] % k_joint == 0, (forecaster, tau, row["n"])

    assert "pit_diagnostic" in both, sorted(both)
    diagnostic = pit_rows_of(both, "pit_diagnostic", PLACE)
    assert diagnostic, both["pit_diagnostic"]
    assert {k[2] for k in diagnostic if len(k) == 3} == {"undetermined"}, sorted(diagnostic)
    assert all(len(k) == 3 for k in diagnostic), sorted(diagnostic)
    alone = pit_rows_of(dumps["undetermined_only"], "pit_diagnostic", PLACE)
    assert_same_pit(diagnostic, alone)
    grid = {float(t) for t in GRID}
    for (forecaster, tau, _), row in diagnostic.items():
        assert row["n"] == len(UNIDENTIFIED), (forecaster, tau, row["n"])
    assert {k[1] for k in diagnostic} == grid
    assert not pit_rows_of(both, "pit_diagnostic", WIN), "WIN cases have no Place read"

    tables = markdown_tables(markdowns["both"])
    pit_tables = [(h, rows) for h, rows in tables if "PIT" in h]
    shown = [
        rows for h, rows in pit_tables if "diagnostic" not in h.lower() and rows and "n" in rows[0]
    ]
    assert len(shown) == 1, [h for h, _ in pit_tables]
    place_n = {
        (r["forecaster"], float(r["tau_min"])): int(r["n"])
        for r in shown[0]
        if r.get("pool") == PLACE
    }
    assert place_n == {k[:2]: int(v["n"]) for k, v in result.items()}, place_n
    marked = [rows for h, rows in pit_tables if "diagnostic" in h.lower()]
    assert marked, [h for h, _ in tables]
    diag_rows = [r for rows in marked for r in rows if r.get("pool") == PLACE]
    assert diag_rows and {r.get("reader") for r in diag_rows} == {"undetermined"}, diag_rows
    assert {(r["forecaster"], float(r["tau_min"])): int(r["n"]) for r in diag_rows} == {
        k[:2]: int(v["n"]) for k, v in diagnostic.items()
    }


# ---------------------------------------------------------------- round 6 (#410 restart)

# The votes build keeps the identity the registered v5 records carry (`registry._tables` counts
# everything before the last `@` toward the variant): only the digest after it changes.
VOTES_BUILD = re.compile(r"processed/jravan/pool_votes\.parquet@[0-9a-f]{16}")
# Strings as CI's pandas 3 makes them: `astype(str)` keeps a null as a missing value, not as text.
# Locally (pandas 2) the `future.infer_string` option gives the same strings; on pandas 3 it is
# already the default.
STRING_MODES = ("default", "pandas3")


@contextlib.contextmanager
def string_mode(mode: str):
    if mode == "default":
        yield
        return
    try:
        pd.get_option("future.infer_string")
    except (KeyError, pd.errors.OptionError):  # a pandas without the option infers already
        yield
        return
    with pd.option_context("future.infer_string", True):
        yield


def votes_frame(rows: list[tuple], votes_dtype: str) -> pd.DataFrame:
    """(race_id, combination, votes, data_kubun) rows, the votes in `votes_dtype` (None a null)."""
    frame = pd.DataFrame(rows, columns=["race_id", "combination", "votes", "data_kubun"])
    if votes_dtype == "Int64":
        frame["votes"] = pd.array(list(frame["votes"]), dtype="Int64")
    elif votes_dtype == "float64":
        frame["votes"] = frame["votes"].astype("float64")  # a null as NaN: pandas 2's reader
    else:
        frame["votes"] = frame["votes"].astype(object)
    return frame


def votes_build_of(rows: list[tuple], votes_dtype: str) -> str:
    builds = study.place_rule_builds(votes_frame(rows, votes_dtype))
    assert set(builds) == {"jravan_pool_votes"}, builds
    return builds["jravan_pool_votes"]


@pytest.mark.parametrize("mode", STRING_MODES)
@pytest.mark.parametrize("votes_dtype", ["Int64", "float64", "object"])
def test_the_votes_build_serialises_a_blank_count(mode, votes_dtype):
    """Restart fix 1 and 2: the JRA-VAN parser emits null for a blank vote count. The votes
    build of rows with a null count (and a null data_kubun) is computed under pandas 3's strings
    as under pandas 2's: it keeps the identity `processed/jravan/pool_votes.parquet` with a
    16-hex digest, does not depend on the rows' order, and the null is a value of its own: it
    differs from the same slot with any count, from the slot dropped, and a null data_kubun from
    a final one. A changed count elsewhere changes the build too."""
    a, b = race_id_of(TEST_DAYS[0], 1), race_id_of(TEST_DAYS[0], 2)
    rows = [
        (a, "1", 4100, "4"),
        (a, "2", 900, "4"),
        (a, "3", None, "4"),  # a blank count
        (b, "1", 700, "5"),
        (b, "2", 650, None),  # a blank data_kubun
        (b, "3", 120, "5"),
    ]
    with string_mode(mode):
        build = votes_build_of(rows, votes_dtype)
        assert VOTES_BUILD.fullmatch(build), build
        rng = np.random.default_rng(410)
        for _ in range(4):
            shuffled = [rows[i] for i in rng.permutation(len(rows))]
            assert votes_build_of(shuffled, votes_dtype) == build
        variants = {
            "count_0": [r if r[:2] != (a, "3") else (a, "3", 0, "4") for r in rows],
            "count_350": [r if r[:2] != (a, "3") else (a, "3", 350, "4") for r in rows],
            "slot_dropped": [r for r in rows if r[:2] != (a, "3")],
            "kubun_final": [r if r[:2] != (b, "2") else (b, "2", 650, "5") for r in rows],
            "other_count": [r if r[:2] != (a, "1") else (a, "1", 4101, "4") for r in rows],
        }
        got = {name: votes_build_of(v, votes_dtype) for name, v in variants.items()}
    for name, value in got.items():
        assert VOTES_BUILD.fullmatch(value), (name, value)
    assert build not in got.values(), (build, got)
    assert len(set(got.values())) == len(got), got


def test_a_blank_place_count_leaves_the_complete_boards_scored(data_dir, tmp_path, monkeypatch):
    """Restart fix 1 and 2, through the study under pandas 3's strings: one registered Place slot
    of one scored race has a blank (null) vote count, and a training day's record a null
    data_kubun. The run completes; the Place measurement has no error, its other scored board is
    validated and the blank-count board is counted unavailable for the missing vote, so the
    Place record stays scored. Every comparison record (and the Place record) carries one
    `jravan_pool_votes` build with the identity `processed/jravan/pool_votes.parquet`, and the
    blank is in it: the same store with that count filled builds a different digest."""
    dev_days = TRAIN_DAYS + TEST_DAYS[:1]
    write_jravan_store(data_dir, dev_days)
    base = final_votes_of(dev_days)
    blank = (race_id_of(TEST_DAYS[0], 2), "3")
    no_kubun = (race_id_of(TRAIN_DAYS[0], 1), "2")
    assert sum((r["race_id"], r["combination"]) == blank for r in base) == 1
    assert sum((r["race_id"], r["combination"]) == no_kubun for r in base) == 1
    blanked = [
        r | {"votes": None}
        if (r["race_id"], r["combination"]) == blank
        else r | {"data_kubun": None}
        if (r["race_id"], r["combination"]) == no_kubun
        else r
        for r in base
    ]
    filled = [r | {"data_kubun": None} if (r["race_id"], r["combination"]) == no_kubun else r
              for r in base]  # fmt: skip
    comparisons = {f"{a}_vs_{b}" for a, b in study.COMPARISONS}
    builds = {}
    with string_mode("pandas3"):
        for name, rows in (("blank", blanked), ("filled", filled)):
            run = votes_build_run(data_dir, tmp_path, monkeypatch, f"r6-{name}", rows)
            measured = run["dump"]["place_measurement"]
            assert "error" not in measured, (name, measured.get("error"))
            counts = measured["counts"]
            assert counts["boards_scored"] == 2, (name, counts)
            missing = sum(
                n for k, n in counts.items()
                if k.startswith("unavailable_votes_missing_on_a_priced_runner")
            )  # fmt: skip
            assert counts["boards_validated"] == (1 if name == "blank" else 2), (name, counts)
            assert missing == (1 if name == "blank" else 0), (name, counts)
            assert study.place_validated(measured), (name, measured["summary"])
            place = run["records"][study.PLACE_VARIANT]
            assert place.status == "scored", (name, place.status, place.params)
            assert place.params.get("unvalidated") is not True
            assert comparisons <= set(run["records"]), (name, sorted(run["records"]))
            votes = {run["records"][c].data_builds.get("jravan_pool_votes") for c in comparisons}
            assert len(votes) == 1 and None not in votes, (name, votes)
            builds[name] = votes.pop()
            assert VOTES_BUILD.fullmatch(builds[name]), (name, builds[name])
            assert place.data_builds.get("jravan_pool_votes") == builds[name], name
    assert builds["blank"] != builds["filled"], builds


# ---------------------------------------------------------------- #410 round 7: places at sale
#
# Owner's restart (2026-11-28), round-7 finding 1: the number of places Place pays, P, is the
# places of the field at the start of sales (config/jra.yaml: 8+ starters pay 3, 5-7 pay 2), not
# of the runners still priced after a scratch (the rule #506 applied to the shadow feed). An
# 8-runner race that loses a runner after sales open prints its 7 figures under the 3-place rule.
# Contract pinned here:
# - `place_shares(odds, odds_hi=None, *, starters=None)` and
#   `pool_shares(pool, odds, investment=None, *, odds_hi=None, starters=None)` take P from
#   `places_for(starters)` when `starters` (the field at the start of sales) is given, else from
#   the priced combinations as before.
# - The board reads (`race_view`, `extract`'s board at t and final board, `closing_boards` and the
#   study's `place_rule_measurement` on them) take the field at sale from the race's rows: the
#   runners priced on its earliest board. A runner priced there and scratched later counts; one
#   already without a price there (scratched before the logs open) does not.
# - `closing_boards` may carry the field at sale after its first four elements (race_id,
#   combinations, low, high); `place_rule_measurement` still reads a four-element board with P
#   from its priced runners.

# (id, money of the runners priced after the scratch, field at the start of sales, places paid)
SCRATCHED_PLACE_BOARDS = [
    ("8-at-sale-1-scratched", _yen([5200, 1100, 900, 700, 500, 400, 300]), 8, 3),
    ("8-at-sale-2-scratched", _yen([4600, 1300, 1000, 800, 600, 400]), 8, 3),
    ("9-at-sale-1-scratched", _yen([4800, 1000, 900, 800, 600, 500, 400, 300]), 9, 3),
    ("7-at-sale-none-scratched", _yen([5200, 1100, 900, 700, 500, 400, 300]), 7, 2),
    ("7-at-sale-1-scratched", _yen([4600, 1300, 1000, 800, 600, 400]), 7, 2),
    ("5-at-sale-1-scratched", _yen([4000, 1600, 1200, 900]), 5, 2),
]
RULE_SUM_TOL = 0.05  # the rule's money total at the right P on these boards: within 0.04 of 1


@pytest.mark.parametrize(
    ("money", "at_sale", "places"),
    [(m, s, p) for _, m, s, p in SCRATCHED_PLACE_BOARDS],
    ids=[b[0] for b in SCRATCHED_PLACE_BOARDS],
)
def test_place_shares_take_the_places_from_the_field_at_sale(money, at_sale, places):
    """Round-7 finding 1: a Place board printed under the places of the field at the start of
    sales is read with that P when `starters` says the field: the shares total 1 and recover the
    money within `test_place_shares_recover_the_money_on_a_known_board`'s tolerance, read `joint`,
    with the rule's money total near 1, by `place_shares` and `pool_shares` alike; the low figures
    alone are read with the same P. A board whose priced runners would give another P prints
    other figures, so a read with the priced runners' P cannot pass."""
    assert places == PLACE_POOL.places_for(at_sale)  # config/jra.yaml's rule, as stated above
    lo, hi, got_places = jra_place_figures(money, places=places)
    assert got_places == places
    assert_readable(money, hi)
    priced_places = PLACE_POOL.places_for(len(money))
    if priced_places != places and priced_places >= 2:  # precondition: P is visible in figures
        other_lo, other_hi, _ = jra_place_figures(money, places=priced_places)
        assert np.abs(other_lo - lo).max() > 0.05 or np.abs(other_hi - hi).max() > 0.05

    shares, rule_sum, reader = place_shares(lo, hi, starters=at_sale)
    assert reader == "joint", reader
    assert_recovers(shares, money)
    assert abs(rule_sum - 1.0) <= RULE_SUM_TOL, rule_sum
    pooled, pooled_sum = pool_shares(PLACE, lo, odds_hi=hi, starters=at_sale)
    np.testing.assert_allclose(pooled, shares, rtol=1e-12, atol=0.0)
    assert pooled_sum == pytest.approx(rule_sum, rel=1e-12)

    low, low_sum = pool_shares(PLACE, lo, starters=at_sale)
    assert low is not None
    low = np.asarray(low, dtype=float)
    assert low.sum() == pytest.approx(1.0, abs=1e-9) and (low > 0).all(), low
    assert abs(low_sum - 1.0) <= RULE_SUM_TOL, low_sum

    if at_sale == len(money):  # no scratch: the field at sale is the priced runners, as before
        bare, bare_sum, bare_reader = place_shares(lo, hi)
        np.testing.assert_allclose(bare, shares, rtol=1e-12, atol=0.0)
        assert (bare_sum, bare_reader) == (pytest.approx(rule_sum, rel=1e-12), reader)


# Three races of one day, each with 7 runners priced from minute 20 on. Race 1 had 8 at the start
# of sales and lost runner 8 at minute 20 (3 places); race 2 had 7 (2 places); race 3's runner 8
# was scratched before the logs open, without a price on any board (2 places).
SALE_OPENING_MIN = 900  # the earliest board, the evening before
SCRATCH_MIN = 20
AT_SALE_TAUS = (30.0, 5.0)  # the board at t before and after the scratch
SEVEN_AT_T = _yen([5200, 1100, 900, 700, 500, 400, 300])
SEVEN_FINAL = _yen([4400, 1500, 1100, 800, 600, 450, 350])
EIGHT_AT_SALE = _yen([4600, 1000, 800, 700, 500, 400, 300, 900])
AT_SALE_RACES = {  # race number: (field at sale, money by minute, the runner never priced)
    1: (8, {SALE_OPENING_MIN: EIGHT_AT_SALE, SCRATCH_MIN: SEVEN_AT_T, 5: SEVEN_AT_T,
            FINAL_MIN: SEVEN_FINAL}, None),
    2: (7, {SALE_OPENING_MIN: SEVEN_AT_T, SCRATCH_MIN: SEVEN_AT_T, 5: SEVEN_AT_T,
            FINAL_MIN: SEVEN_FINAL}, None),
    3: (7, {SALE_OPENING_MIN: SEVEN_AT_T, SCRATCH_MIN: SEVEN_AT_T, 5: SEVEN_AT_T,
            FINAL_MIN: SEVEN_FINAL}, "8"),
}  # fmt: skip


def at_sale_rows(day: date, races: dict | None = None) -> list[dict]:
    """The WIN and PLACE rows of `races` (default `AT_SALE_RACES`): each board's PLACE figures as
    JRA prints them from its money under the places of the race's field at sale; a runner off a
    board (scratched, or never priced) has rows with no price, as the store writes a scratch."""
    rows = []
    for race_no, (at_sale, boards, never) in (AT_SALE_RACES if races is None else races).items():
        race_id, post = race_id_of(day, race_no), post_of(day, race_no)
        places = PLACE_POOL.places_for(at_sale)
        for minutes, money in boards.items():
            final = minutes == FINAL_MIN
            lo, hi, _ = jra_place_figures(money, places=places)
            assert_readable(money, hi)
            for j in range(len(money)):
                c = str(j + 1)
                rows.append(row(race_id, race_no, PLACE, c, post, minutes, odds=lo[j],
                                odds_hi=hi[j], final=final))  # fmt: skip
                rows.append(row(race_id, race_no, WIN, c, post, minutes, odds=2.0 + j,
                                final=final))  # fmt: skip
            off = [str(j + 1) for j in range(len(money), at_sale)] + ([never] if never else [])
            for c in off:
                for pool in (WIN, PLACE):
                    rows.append({**row(race_id, race_no, pool, c, post, minutes, final=final),
                                 "status_raw": "----"})  # fmt: skip
            rows.append(row(race_id, race_no, PLACE, POOL_ROW, post, minutes,
                            investment=float(np.sum(money)), final=final))  # fmt: skip
    return rows


def test_board_reads_take_the_places_from_the_field_at_sale():
    """Round-7 finding 1 on the study's board reads: `race_view`'s board at t, `extract`'s board
    at t and final board, and `closing_boards` read through `place_rule_measurement`, take P from
    the field at the start of sales. Race 1 (8 at sale, 7 priced after a scratch) is read with 3
    places, races 2 and 3 (7 at sale; race 3's eighth runner never priced) with 2, though all
    three price 7 runners at the close; each read recovers the money it was printed from."""
    day = TEST_DAYS[0]
    frame = to_frame(at_sale_rows(day))
    seven = tuple(str(j + 1) for j in range(7))
    for race_no, (_, boards, _) in AT_SALE_RACES.items():
        race_id = race_id_of(day, race_no)
        for tau in AT_SALE_TAUS:
            board = race_view(frame, race_id, tau_min=tau).pools[PLACE]
            money = boards[SALE_OPENING_MIN] if tau > SCRATCH_MIN else boards[5]
            want = tuple(str(j + 1) for j in range(len(money)))
            assert board.combinations == want, (race_no, tau, board.combinations)
            assert_recovers(board.shares, money)

    cases, _, _ = extract(frame, taus=AT_SALE_TAUS)
    place_cases = [i for i in range(len(cases)) if cases.pool[i] == PLACE]
    assert len(place_cases) == len(AT_SALE_RACES) * len(AT_SALE_TAUS)
    for i in place_cases:
        race_no = int(str(cases.race_id[i]).rsplit(":", 1)[1])
        boards = AT_SALE_RACES[race_no][1]
        r = cases.rows(i)
        combos, s_t, s_f = tuple(cases.combination[r]), cases.s_t[r], cases.s_f[r]
        on_final = np.isfinite(s_f)
        assert tuple(np.asarray(combos)[on_final]) == seven, (race_no, combos)
        assert_recovers(s_f[on_final], SEVEN_FINAL)
        before = float(cases.tau[i]) > SCRATCH_MIN
        assert_recovers(s_t, boards[SALE_OPENING_MIN] if before else boards[5])
        assert cases.reader[i] == "joint", (race_no, cases.reader[i])

    closing = list(closing_boards(frame, PLACE))
    assert sorted(b[0] for b in closing) == [race_id_of(day, n) for n in AT_SALE_RACES]
    for b in closing:
        assert tuple(b[1]) == seven, b[1]
    votes = [r for n in AT_SALE_RACES for r in place_votes_rows(race_id_of(day, n), SEVEN_FINAL)]
    measured = study.place_rule_measurement(closing, pd.DataFrame(votes))
    assert measured["counts"]["boards_validated"] == 3, measured["counts"]
    p3 = summary_row(measured, "joint", "P=3")
    assert (p3["n_scored"], p3["n"]) == (1, 1), measured["summary"]
    p2 = summary_row(measured, "joint", "P=2")
    assert (p2["n_scored"], p2["n"]) == (2, 2), measured["summary"]
    for got in (p3, p2):  # every board reproduced from its votes at its P, and read within
        assert got["share_reproduced"] == pytest.approx(1.0), got
        assert got["share_within_tolerance"] == pytest.approx(1.0), got


# ---------------------------------------------------------------- round 8 (owner's 07:20 restart)
#
# 1. v5's spec changed after the family was frozen (P from the field at sale, df394e8), so the
#    changed spec is frozen as a new family, japan-c1-nowcast-final-shares-v6, superseding v5;
#    v5's freeze and records stay in the registry as history (#218's v3 rule).
# 2. The drift baseline reads its Place odds draws into shares with P from the field at the start
#    of sales, as the board reads do: after an 8 -> 7 scratch with three places, after a 5 -> 4
#    scratch with two (four priced runners alone are not a Place field: no shares at all).

V4_FAMILY = "japan-c1-nowcast-final-shares-v4"
V5_FAMILY = "japan-c1-nowcast-final-shares-v5"
V6_FAMILY = "japan-c1-nowcast-final-shares-v6"


def _study_spec() -> dict:
    """`build_spec` on the inputs `run_study` passes it."""
    scoring = config.load_country_yaml(COUNTRY, "scoring")["bootstrap"]
    bootstrap = {
        "n_boot": int(scoring["n_boot"]),
        "level": float(scoring["level"]),
        "seed": study.SEED,
    }
    return study.build_spec(
        load_decision_grid(),
        study.load_country_scheme(COUNTRY),
        bootstrap,
        config.load_country_yaml(COUNTRY, "nowcast_v0"),
    )


def test_the_changed_spec_is_family_v6_superseding_v5():
    """Round 8, finding 1: the study's family is v6 (its trials v6-trial), superseding v5, and the
    spec it freezes says so: its family id is v6 and its `supersedes` names v5 first, says why
    (the spec changed under the frozen family: P from the field at the start of sales) and keeps
    the earlier chain (v4, v3)."""
    assert study.FAMILY_ID == V6_FAMILY
    assert study.TRIAL_FAMILY_ID == f"{V6_FAMILY}-trial"
    assert study.SUPERSEDES == V5_FAMILY
    spec = _study_spec()
    assert spec["family_id"] == V6_FAMILY
    supersedes = spec["supersedes"]
    assert supersedes.startswith(f"{V5_FAMILY}:"), supersedes[:120]
    assert re.search(r"field at (the start of )?sales?", supersedes, re.IGNORECASE), supersedes
    assert re.search(r"frozen", supersedes, re.IGNORECASE), supersedes
    assert "v4" in supersedes and "v3" in supersedes, supersedes  # the earlier chain is kept


def _freeze_lines(path: Path) -> list[dict]:
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [entry for entry in lines if entry.get("kind") == "freeze"]


def test_a_run_freezes_v6_and_keeps_the_frozen_v5_as_history(data_dir, tmp_path, monkeypatch):
    """Round 8, finding 1, on the run path: a registry that froze v5 with its earlier spec (any
    spec other than today's) no longer stops `run_study` with FamilyAlreadyFrozen. The run freezes
    today's spec as v6 before it scores, every record it appends is v6's under that spec, and
    v5's freeze stays as it was: the registry only grows, its chain verifies, v5's hash is the
    one frozen, and no record joins v5."""
    write_jravan_store(data_dir, TRAIN_DAYS + TEST_DAYS[:1])
    monkeypatch.setattr(study, "load_country_scheme", lambda country: FOLD_SCHEME)
    summary = tmp_path / "summary"  # the run regenerates the registry summary: not the repo's
    monkeypatch.setattr(study, "summary_dir", lambda country: summary)
    monkeypatch.setattr("horsetrack.core.eval.registry.summary_dir", lambda country: summary)
    registry = default_registry(COUNTRY)
    v5_spec = {
        "family_id": V5_FAMILY,
        "supersedes": V4_FAMILY,
        "place_places": "P from the runners priced on the board (as frozen before df394e8)",
    }
    v5_hash = registry.freeze_family(V5_FAMILY, v5_spec)
    before = registry.path.read_bytes()

    study.run_study(
        trial=False, test_days_per_fold=1, n_draws=30, out_dir=tmp_path / "out", log=lambda *a: None
    )

    after = registry.path.read_bytes()
    assert after.startswith(before) and len(after) > len(before)  # appended, nothing rewritten
    registry.verify()
    assert registry.frozen_spec_hash(V5_FAMILY) == v5_hash
    v6_hash = registry.frozen_spec_hash(V6_FAMILY)
    assert v6_hash is not None and v6_hash != v5_hash
    freezes = {entry["family_id"]: entry for entry in _freeze_lines(registry.path)}
    assert freezes[V5_FAMILY]["spec"] == v5_spec
    v6_spec = freezes[V6_FAMILY]["spec"]
    assert v6_spec["family_id"] == V6_FAMILY
    assert v6_spec["supersedes"].startswith(f"{V5_FAMILY}:")
    records = registry.records()
    assert records, "the run appended no registry record"
    assert {r.family_id for r in records} == {V6_FAMILY}, {r.family_id for r in records}
    assert {r.params.get("spec_sha256") for r in records} == {v6_hash}
    assert registry.records(V5_FAMILY) == []


# The races of the drift tests, all on one day: `AT_SALE_RACES` (race 1: 8 at sale, 7 priced from
# minute 20, three places; races 2 and 3: 7 at sale, two places) and race 4, 5 at sale and 4
# priced from minute 20 (two places; four runners alone are not a Place field).
FIVE_AT_SALE = _yen([3600, 1500, 1100, 800, 700])
FOUR_AT_T = _yen([4000, 1600, 1200, 900])
FOUR_FINAL = _yen([3800, 1700, 1300, 1000])
DRIFT_RACES = {
    **AT_SALE_RACES,
    4: (5, {SALE_OPENING_MIN: FIVE_AT_SALE, SCRATCH_MIN: FOUR_AT_T, 5: FOUR_AT_T,
            FINAL_MIN: FOUR_FINAL}, None),
}  # fmt: skip


def still_drift(mu: float) -> NowcastV0:
    """#116's drift with no spread: every Place odds draw is the figure at t times exp(mu), so a
    draw is known without the model's noise; WIN has no drift either."""
    grid = load_decision_grid()
    return NowcastV0(
        bands=(),
        grid=grid,
        prior_n=0.0,
        days=(),
        sigma0={(pool, int(t)): 0.0 for pool in (WIN, PLACE) for t in grid.minutes},
        cells={(PLACE, int(t), 0): DriftCell(PLACE, int(t), 0, 1, mu, 0.0) for t in grid.minutes},
    )


def jra_printed(figures: np.ndarray) -> np.ndarray:
    """Place figures as JRA prints them: down to the 10-yen step, at least the minimum dividend
    (config/jra.yaml), per 100 yen."""
    step = float(PLACE_POOL.dividend_step) / PLACE_UNIT
    floor = float(PLACE_POOL.min_dividend) / PLACE_UNIT
    return np.maximum(np.floor(np.asarray(figures, dtype=float) / step + 1e-9) * step, floor)


def drift_cases():
    """The study's cases of `DRIFT_RACES` at `AT_SALE_TAUS`, as `run_study` scores them (built,
    concatenated and selected by day), with each Place case's race number."""
    day = TEST_DAYS[0]
    cases, _, _ = extract(to_frame(at_sale_rows(day, DRIFT_RACES)), taus=AT_SALE_TAUS)
    cases = type(cases).concat([cases]).select_days([day.isoformat()])
    place = [i for i in range(len(cases)) if cases.pool[i] == PLACE]
    assert len(place) == len(DRIFT_RACES) * len(AT_SALE_TAUS)
    return cases, {i: int(str(cases.race_id[i]).rsplit(":", 1)[1]) for i in place}


def test_the_drift_baseline_reads_place_draws_with_the_field_at_sale():
    """Round 8, finding 2: `drift_forecaster` reads its Place odds draws into shares with P from
    the field at the start of sales. With no drift its draws are the board at t, and read with
    that P they recover the money the board was printed from: after 8 -> 7 with three places
    (P = 2 from the seven priced misreads them), after 5 -> 4 with two (the four priced alone are
    no Place field), and, as before, where nothing was scratched by t. With a drift, each draw is
    the read at the field-at-sale P of the figures it prints, never the priced runners' P where
    the two differ."""
    cases, races = drift_cases()
    mu = 0.12
    for i, race_no in races.items():
        at_sale, boards, _ = DRIFT_RACES[race_no]
        tau = float(cases.tau[i])
        money = boards[SALE_OPENING_MIN] if tau > SCRATCH_MIN else boards[5]
        view = cases.view(i)
        lo, hi = cases.odds_t[cases.rows(i)], cases.odds_hi_t[cases.rows(i)]
        assert len(lo) == len(money), (race_no, tau)
        label = f"race {race_no} at tau {tau:g}: {at_sale} at sale, {len(money)} priced"

        flat = study.drift_forecaster(still_drift(0.0))(view, pool=PLACE, n=4, seed=7).shares
        assert flat is not None, label
        flat = np.asarray(flat, dtype=float)
        assert flat.shape == (4, len(money)), label
        for draw in flat:
            assert_recovers(draw, money)

        drifted = study.drift_forecaster(still_drift(mu))(view, pool=PLACE, n=3, seed=7).shares
        lo_d = np.maximum(lo * math.exp(mu), 1.0)
        hi_d = np.maximum(hi * math.exp(mu), lo_d)
        want, _, _ = place_shares(jra_printed(lo_d), jra_printed(hi_d), starters=at_sale)
        assert want is not None, label
        np.testing.assert_allclose(
            np.asarray(drifted, dtype=float), np.tile(want, (3, 1)), rtol=1e-9, atol=1e-12,
            err_msg=label,
        )  # fmt: skip

        priced, _, _ = place_shares(jra_printed(lo_d), jra_printed(hi_d))
        if PLACE_POOL.places_for(len(money)) == PLACE_POOL.places_for(at_sale):
            np.testing.assert_allclose(want, priced, rtol=1e-12, atol=0.0, err_msg=label)
        elif priced is not None:  # precondition: P shows in the read (the 8 -> 7 case)
            assert np.abs(np.asarray(priced) - want).max() > 0.01, label


def test_the_study_scores_the_drift_baseline_after_a_scratch_below_a_place_field():
    """Round 8, finding 2, through the study's scoring path (`score_cases` on the cases
    `run_study` scores, with `drift_forecaster` as `drift_v0`): a 5 -> 4 scratch no longer aborts
    the scoring, and every Place case, scratched or not, is scored with finite values equal to
    those of a reference that reads the same board at t with the field-at-sale P from the
    fixture."""
    cases, races = drift_cases()

    def reference(view, *, pool, n, seed):
        race_no = int(view.race_id.rsplit(":", 1)[1])
        board = view.pools[pool]
        shares, _, _ = place_shares(board.odds, board.odds_hi, starters=DRIFT_RACES[race_no][0])
        return Draws(shares=np.tile(shares, (n, 1)), v_final=np.full(n, np.nan))

    place = cases.select(cases.pool == PLACE)
    forecasters = {"drift_v0": study.drift_forecaster(still_drift(0.0)), "reference": reference}
    table, pit_rows = score_cases(forecasters, place, n=20, seed=3, keep_pit=True)
    assert len(table) == 2 * len(races)
    assert np.isfinite(table[["log_score", "crps", "kl"]].to_numpy(float)).all(), table
    scores = table.set_index(["race_id", "tau_min", "forecaster"])[["log_score", "crps", "kl"]]
    drift = scores.xs("drift_v0", level="forecaster").sort_index()
    ref = scores.xs("reference", level="forecaster").sort_index()
    assert drift.index.equals(ref.index)
    np.testing.assert_allclose(drift.to_numpy(float), ref.to_numpy(float), rtol=1e-9, atol=1e-12)
    four = race_id_of(TEST_DAYS[0], 4)
    assert {float(t) for r, t in drift.index if r == four} == set(AT_SALE_TAUS)
    assert np.isfinite(pit_rows["pit"].to_numpy(float)).all()
