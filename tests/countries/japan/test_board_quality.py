"""Board quality by horizon and the price-forecast EV bias (ticket #86; 08 §4 E10, E18, E21).

Contract pinned here (`horsetrack.countries.japan.studies.board_quality`):

- The input frame is long: one row per (race, runner, capture), columns
  - race_id, horse_id: the race and the runner (for the drift functions, the pool's runner key);
  - won (board_vs_final only): 1 for the winner, else 0;
  - pool (drift functions only): "WIN", "PLACE", ...;
  - minutes_to_post: minutes before the scheduled post at which the capture was available, as in
    the compacted odds log (`logger/compact.py`); negative after the post;
  - odds: decimal, stake included;
  - is_final: True on the rows of the race's logged final board (docs/specs/board-at-t.md,
    final-state rule), else False.
- The board at tau is, per race, the latest capture available at or before post - tau: the rows
  with the smallest minutes_to_post that is >= tau. A race with no capture that early has no board
  at tau and no row for it.
- `board_vs_final(frame, *, taus, devig)`: one row per (race, tau) with columns race_id, tau_min,
  logloss_board, logloss_final, gap. logloss is -ln of the de-vigged probability of the winner;
  gap = logloss_board - logloss_final. `devig(odds, race_id)` is one of #77's de-vigs (or any
  callable with that signature) and is applied to the board and the final alike. A race without
  a logged final is left out of every tau and counted in `result.attrs["n_races_without_final"]`.
- `board_quality_table(gaps, clusters, *, n_boot, level, seed)`: one row per tau in `gaps`,
  columns tau_min, mean, lo, hi, n_races, n_clusters; the bounds resample whole clusters
  (`clusters` maps race id to meeting, D7's venue x day).
- `spurious_ev_bias(sigma, rate)` = rate * (exp(sigma^2) - 1).
- `simulate_spurious_ev_bias(sigma, rate, *, n, seed)` -> (estimate, standard error), unpackable.
- `drift_sd_by_tau(frame)`: columns pool, tau_min, sd, n, one row per pool and per tau of the
  owner's grid (`config/decision.yaml`); sd is the standard deviation (centred) of
  ln(O_final / O_tau) over the runners, n the runners counted.
- `drift_sd_by_segment(frame, segment)`: the same per value of `frame[segment]` (constant within a
  race), the value in a column named `segment` (as `calibration.segment_beta`).
- `spurious_ev_table(drift, *, rates)`: the drift table with a `spurious_ev` column, the pool's
  payout rate from `rates` (pool -> rate) put through `spurious_ev_bias`.

Expected values come from first principles (inverse-odds shares computed here, planted drift
SDs, the lognormal moments), never from the code under test. Every frame is synthetic and every
RNG is seeded.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math
from functools import partial
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from horsetrack.core import config
from horsetrack.countries.japan.studies.board_quality import (
    board_quality_table,
    board_vs_final,
    drift_sd_by_segment,
    drift_sd_by_tau,
    simulate_spurious_ev_bias,
    spurious_ev_bias,
    spurious_ev_table,
)
from horsetrack.countries.japan.studies.calibration import (
    devig_normalised,
    devig_power,
    devig_shin,
)

GRID = [float(m) for m in config.load_country_yaml("japan", "decision")["grid_min"]]
BOOTSTRAP = config.load_country_yaml("japan", "scoring")["bootstrap"]
FINAL_MTP = -0.5  # the final board is read just after the scheduled post
GAP_COLUMNS = {"race_id", "tau_min", "logloss_board", "logloss_final", "gap"}


# ---------------------------------------------------------------------------------------------
# Synthetic boards and first-principles references
# ---------------------------------------------------------------------------------------------


def capture(race_id, minutes_to_post, odds, winner, *, final=False):
    """Rows of one capture of one race's Win board; horses are numbered 1..n, `winner` is one."""
    return [
        {
            "race_id": race_id,
            "horse_id": h,
            "won": int(h == winner),
            "minutes_to_post": float(minutes_to_post),
            "odds": float(o),
            "is_final": final,
        }
        for h, o in enumerate(odds, start=1)
    ]


def frame_of(rows, seed=0):
    """The rows in a shuffled order, so nothing may rely on the frame being sorted."""
    frame = pd.DataFrame(rows)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def ref_logloss(odds, winner, power=1.0):
    """-ln of the winner's share of (1/O)^power over the race: power 1 is the normalised de-vig."""
    w = (1.0 / np.asarray(odds, dtype=float)) ** power
    return float(-math.log(w[winner - 1] / w.sum()))


def devig_squared(odds: pd.Series, race_id: pd.Series) -> pd.Series:
    """A de-vig no one would use, p proportional to O^-2, to show the passed de-vig is applied."""
    w = (1.0 / odds.astype(float)) ** 2
    return w / w.groupby(race_id).transform("sum")


def by_key(result):
    return result.set_index(["race_id", result["tau_min"].astype(float)]).sort_index()


# ---------------------------------------------------------------------------------------------
# board_vs_final
# ---------------------------------------------------------------------------------------------


FINAL_BOOKS = {
    # race_id: (final Win odds, winner): overrounds and field sizes differ, winners too
    "jra:2026-11-21:TOK:01": ([2.4, 5.5, 7.0, 12.0, 30.0], 1),
    "jra:2026-11-21:TOK:02": ([3.1, 3.3, 9.8, 15.5], 3),
    "jra:2026-11-22:KYO:05": ([1.6, 6.0, 11.0, 25.0, 40.0, 80.0], 2),
    "jra:2026-11-22:KYO:11": ([4.0, 4.4, 5.0, 6.1], 4),
}


@pytest.mark.parametrize(
    "devig, power",
    [
        (devig_normalised, 1.0),
        (partial(devig_power, beta=1.3), 1.3),
        (devig_shin, None),
    ],
    ids=["normalised", "power", "shin"],
)
def test_a_board_equal_to_the_final_has_zero_gap(devig, power):
    rows = []
    for race_id, (odds, winner) in FINAL_BOOKS.items():
        for tau in GRID:  # a capture at every grid point, each equal to the final board
            rows += capture(race_id, tau, odds, winner)
        rows += capture(race_id, FINAL_MTP, odds, winner, final=True)

    result = board_vs_final(frame_of(rows), taus=GRID, devig=devig)

    assert GAP_COLUMNS <= set(result.columns)
    got = by_key(result)
    expected_keys = {(r, t) for r in FINAL_BOOKS for t in GRID}
    assert set(got.index) == expected_keys
    assert got["gap"].to_numpy() == pytest.approx(0.0, abs=1e-12)
    assert got["logloss_board"].to_numpy() == pytest.approx(got["logloss_final"].to_numpy())
    assert result.attrs["n_races_without_final"] == 0
    if power is not None:  # the final's loss itself is the de-vigged winner's -ln p, not 0
        for race_id, (odds, winner) in FINAL_BOOKS.items():
            for tau in GRID:
                assert got.loc[(race_id, tau), "logloss_final"] == pytest.approx(
                    ref_logloss(odds, winner, power), rel=1e-12
                )


# Race A: captures every few minutes, one of them exactly at tau = 30 and one exactly at tau = 5,
# and a decoy at 4.5 (after T-5) priced nothing like the rest. Race B: a late-starting log.
RACE_A = "jra:2026-11-21:TOK:07"
RACE_A_WINNER = 2
RACE_A_CAPTURES = {
    45.0: [3.0, 5.0, 8.0, 6.0],
    30.0: [3.2, 4.6, 8.5, 6.0],
    12.0: [3.4, 4.2, 9.0, 6.5],
    5.0: [3.6, 3.9, 9.5, 7.0],
    4.5: [9.0, 1.8, 20.0, 30.0],
    3.0: [3.8, 3.6, 10.0, 7.5],
    1.5: [4.0, 3.4, 10.5, 8.0],
}
RACE_A_FINAL = [4.2, 3.2, 11.0, 8.5]
RACE_A_BOARD_AT = {30.0: 30.0, 16.0: 30.0, 5.0: 5.0, 2.0: 3.0, 1.0: 1.5}  # tau -> capture; no T-60

RACE_B = "jra:2026-11-22:KYO:03"
RACE_B_WINNER = 1
RACE_B_CAPTURES = {
    20.0: [2.0, 3.0, 5.0],
    6.0: [2.2, 2.8, 5.0],
    2.0: [2.4, 2.7, 4.6],
}
RACE_B_FINAL = [2.6, 2.5, 4.5]
RACE_B_BOARD_AT = {16.0: 20.0, 5.0: 6.0, 2.0: 2.0, 1.0: 2.0}  # no board at T-60 or T-30


@pytest.mark.parametrize(
    "devig, power", [(devig_normalised, 1.0), (devig_squared, 2.0)], ids=["normalised", "squared"]
)
def test_board_at_tau_is_the_latest_capture_at_or_before_tau(devig, power):
    rows = []
    for mtp, odds in RACE_A_CAPTURES.items():
        rows += capture(RACE_A, mtp, odds, RACE_A_WINNER)
    rows += capture(RACE_A, FINAL_MTP, RACE_A_FINAL, RACE_A_WINNER, final=True)
    for mtp, odds in RACE_B_CAPTURES.items():
        rows += capture(RACE_B, mtp, odds, RACE_B_WINNER)
    rows += capture(RACE_B, -0.2, RACE_B_FINAL, RACE_B_WINNER, final=True)

    result = board_vs_final(frame_of(rows, seed=86), taus=GRID, devig=devig)
    got = by_key(result)

    cases = [
        (RACE_A, RACE_A_CAPTURES, RACE_A_FINAL, RACE_A_WINNER, RACE_A_BOARD_AT),
        (RACE_B, RACE_B_CAPTURES, RACE_B_FINAL, RACE_B_WINNER, RACE_B_BOARD_AT),
    ]
    expected_keys = {(race, tau) for race, *_, board_at in cases for tau in board_at}
    assert set(got.index) == expected_keys  # no board before the log starts: no row
    for race, captures, final, winner, board_at in cases:
        ll_final = ref_logloss(final, winner, power)
        for tau, mtp in board_at.items():
            ll_board = ref_logloss(captures[mtp], winner, power)
            row = got.loc[(race, tau)]
            assert row["logloss_board"] == pytest.approx(ll_board, rel=1e-12), (race, tau)
            assert row["logloss_final"] == pytest.approx(ll_final, rel=1e-12), (race, tau)
            assert row["gap"] == pytest.approx(ll_board - ll_final, rel=1e-10, abs=1e-14)
    assert result.attrs["n_races_without_final"] == 0


@pytest.mark.parametrize("missing", [{"R2", "R4"}, {"R5"}, set()], ids=["two", "one", "none"])
def test_races_without_a_logged_final_are_left_out(missing):
    rng = np.random.default_rng(len(missing) + 7)
    rows = []
    races = ["R1", "R2", "R3", "R4", "R5"]
    for race in races:
        final = np.round(rng.uniform(1.5, 40.0, size=5), 1)
        winner = int(rng.integers(1, 6))
        for tau in GRID:
            rows += capture(race, tau, np.round(final * rng.uniform(0.8, 1.25, size=5), 1), winner)
        if race in missing:  # logged after the post, but never with the final-state mark
            rows += capture(race, FINAL_MTP, final, winner, final=False)
        else:
            rows += capture(race, FINAL_MTP, final, winner, final=True)

    result = board_vs_final(frame_of(rows, seed=3), taus=GRID, devig=devig_normalised)

    kept = sorted(set(races) - missing)
    assert sorted(result["race_id"].unique()) == kept
    for tau in GRID:
        at_tau = result[result["tau_min"].astype(float) == tau]
        assert sorted(at_tau["race_id"]) == kept
    assert result.attrs["n_races_without_final"] == len(missing)


# ---------------------------------------------------------------------------------------------
# The board-quality table: every grid tau, meeting-clustered bounds
# ---------------------------------------------------------------------------------------------


def test_board_quality_table_covers_the_grid_with_meeting_clustered_bounds():
    """12 meetings of 20 identical races: the gap varies between meetings only, so meeting-level
    resampling gives bounds about sqrt(20) times wider than resampling races would."""
    n_meetings, per_meeting = 12, 20
    final, winner = [2.5, 4.0, 6.0, 12.0], 1
    shifts = np.linspace(-0.6, 0.6, n_meetings) + 0.1  # ln of the winner's board/final at T-60
    rows, clusters, expected_gap = [], {}, {}
    for k, c in enumerate(shifts):
        meeting = f"2026-11-{k + 1:02d}:TOK"
        for j in range(per_meeting):
            race = f"jra:{meeting}:{j + 1:02d}"
            clusters[race] = meeting
            for tau in GRID:
                board = list(final)
                board[winner - 1] = final[winner - 1] * math.exp(c * tau / 60.0)
                rows += capture(race, tau, board, winner)
                expected_gap[(race, tau)] = ref_logloss(board, winner) - ref_logloss(final, winner)
            rows += capture(race, FINAL_MTP, final, winner, final=True)

    gaps = board_vs_final(frame_of(rows, seed=11), taus=GRID, devig=devig_normalised)
    table = board_quality_table(
        gaps,
        pd.Series(clusters),
        n_boot=int(BOOTSTRAP["n_boot"]),
        level=float(BOOTSTRAP["level"]),
        seed=86,
    )

    assert sorted(table["tau_min"].astype(float)) == sorted(GRID)
    z = NormalDist().inv_cdf(float(BOOTSTRAP["level"]))
    for tau in GRID:
        row = table[table["tau_min"].astype(float) == tau].iloc[0]
        per_race = np.array([expected_gap[(r, tau)] for r in clusters])
        meeting_gaps = per_race.reshape(n_meetings, per_meeting)[:, 0]
        assert row["mean"] == pytest.approx(per_race.mean(), rel=1e-9, abs=1e-15)
        assert row["n_races"] == n_meetings * per_meeting
        assert row["n_clusters"] == n_meetings
        assert row["lo"] < row["mean"] < row["hi"]
        clustered_width = 2.0 * z * meeting_gaps.std(ddof=0) / math.sqrt(n_meetings)
        assert 0.7 < (row["hi"] - row["lo"]) / clustered_width < 1.3, tau


# ---------------------------------------------------------------------------------------------
# The spurious EV bias: formula and simulation
# ---------------------------------------------------------------------------------------------


def test_spurious_ev_bias_matches_the_lognormal_formula():
    # The ticket's figures at rate 0.8, to 4 places.
    assert round(spurious_ev_bias(0.15, 0.8), 4) == 0.0182
    assert round(spurious_ev_bias(0.22, 0.8), 4) == 0.0397
    assert round(spurious_ev_bias(0.42, 0.8), 4) == 0.1543
    # And over a spread of sigma and rate, each dimension moving the answer.
    for sigma in (0.0, 0.05, 0.1, 0.3, 0.6):
        for rate in (0.7, 0.775, 0.8, 1.0):
            assert spurious_ev_bias(sigma, rate) == pytest.approx(
                rate * math.expm1(sigma**2), rel=1e-12, abs=1e-15
            )


def test_simulated_bias_agrees_with_the_formula():
    """Efficient final price: p = rate / O_F, so E[p * O_F] = rate. Given the board, ln O_F is
    normal with sd sigma (any mean). The product of the two conditional expectations is
    p_hat * O_hat = rate * E[1/O_F] * E[O_F] = rate * exp(sigma^2), so it overstates the true
    expected return by exactly rate * (exp(sigma^2) - 1). The simulation estimates p_hat and
    O_hat from draws of O_F and returns the estimate of the overstatement and its Monte Carlo SE.
    """
    sigma, rate, n = 0.22, 0.8, 200_000
    target = rate * math.expm1(sigma**2)

    estimate, se = simulate_spurious_ev_bias(sigma, rate, n=n, seed=86)
    assert 0.0 < se
    assert 3.0 * se < 0.1 * target  # the 3-SE check has teeth
    assert abs(estimate - target) <= 3.0 * se
    assert estimate != target  # a simulation, not the closed form

    other, other_se = simulate_spurious_ev_bias(sigma, rate, n=n, seed=87)
    assert other != estimate
    assert abs(other - target) <= 3.0 * other_se

    small, small_se = simulate_spurious_ev_bias(sigma, rate, n=n // 4, seed=88)
    assert 1.6 < small_se / se < 2.4  # SE shrinks as 1/sqrt(n)
    assert abs(small - target) <= 3.0 * small_se

    wide, wide_se = simulate_spurious_ev_bias(0.42, 1.0, n=n, seed=89)
    wide_target = math.expm1(0.42**2)
    assert abs(wide - wide_target) <= 3.0 * wide_se
    assert 3.0 * wide_se < 0.1 * wide_target


# ---------------------------------------------------------------------------------------------
# Drift SD by pool, tau and segment, and the EV-bias table
# ---------------------------------------------------------------------------------------------

POOLS = ("WIN", "PLACE")
SIGMA = {  # planted SD of ln(O_final / O_tau) by pool and tau
    "WIN": {60.0: 0.40, 30.0: 0.33, 16.0: 0.25, 5.0: 0.15, 2.0: 0.10, 1.0: 0.07},
    "PLACE": {60.0: 0.22, 30.0: 0.18, 16.0: 0.14, 5.0: 0.09, 2.0: 0.06, 1.0: 0.04},
}
MEAN_DRIFT = {"WIN": 0.05, "PLACE": -0.03}  # a mean move: an SD, not a root mean square
HORSES = 6


def drift_frame(seed, races_by_segment, sigma_of):
    """Long frame with an exact planted SD per (segment, pool, tau) cell. Each grid tau has a
    capture at exactly tau; a decoy just after it (tau - 0.5) is priced at random and must never
    be read as the board at tau. The final is at FINAL_MTP."""
    rng = np.random.default_rng(seed)
    blocks = []
    for segment, n_races in races_by_segment.items():
        race_ids = np.repeat([f"{segment}-{i:04d}" for i in range(n_races)], HORSES)
        horse_ids = np.tile(np.arange(1, HORSES + 1), n_races)
        n_obs = n_races * HORSES

        def block(pool, mtp, odds, final, race_ids=race_ids, horse_ids=horse_ids, seg=segment):
            return pd.DataFrame(
                {
                    "race_id": race_ids,
                    "horse_id": horse_ids,
                    "pool": pool,
                    "minutes_to_post": float(mtp),
                    "odds": odds,
                    "is_final": final,
                    "turnover_band": seg,
                }
            )

        for pool in POOLS:
            final_odds = rng.uniform(4.0, 60.0, size=n_obs)
            blocks.append(block(pool, FINAL_MTP, final_odds, True))
            for tau in GRID:
                z = rng.uniform(-1.0, 1.0, size=n_obs)
                z = (z - z.mean()) / z.std(ddof=1)
                log_ratio = MEAN_DRIFT[pool] + sigma_of(segment, pool, tau) * z
                blocks.append(block(pool, tau, final_odds / np.exp(log_ratio), False))
                blocks.append(block(pool, tau - 0.5, rng.uniform(1.5, 80.0, size=n_obs), False))
    frame = pd.concat(blocks, ignore_index=True)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def test_drift_sd_by_tau_recovers_the_planted_sd_for_every_pool_and_grid_tau():
    n_races = 300
    frame = drift_frame(1, {"all": n_races}, lambda seg, pool, tau: SIGMA[pool][tau])

    table = drift_sd_by_tau(frame)

    keys = {(p, float(t)) for p, t in zip(table["pool"], table["tau_min"], strict=True)}
    assert keys == {(p, t) for p in POOLS for t in GRID}
    assert len(table) == len(keys)
    for _, row in table.iterrows():
        planted = SIGMA[row["pool"]][float(row["tau_min"])]
        assert row["sd"] == pytest.approx(planted, rel=1e-3), (row["pool"], row["tau_min"])
        assert row["n"] == n_races * HORSES


def test_drift_sd_by_segment_recovers_each_segments_sd():
    scale = {"low": 1.5, "high": 1.0}  # low-turnover races move more late
    n_races = 150
    frame = drift_frame(
        2, {"low": n_races, "high": n_races}, lambda seg, pool, tau: scale[seg] * SIGMA[pool][tau]
    )

    table = drift_sd_by_segment(frame, "turnover_band")

    keys = {
        (s, p, float(t))
        for s, p, t in zip(table["segment"], table["pool"], table["tau_min"], strict=True)
    }
    assert keys == {(s, p, t) for s in scale for p in POOLS for t in GRID}
    assert len(table) == len(keys)
    for _, row in table.iterrows():
        planted = scale[row["segment"]] * SIGMA[row["pool"]][float(row["tau_min"])]
        assert row["sd"] == pytest.approx(planted, rel=1e-3), tuple(row[["segment", "pool"]])
        assert row["n"] == n_races * HORSES


def test_spurious_ev_table_gives_the_bias_by_tau_and_pool():
    rates = {"WIN": 0.8, "PLACE": 0.7}
    frame = drift_frame(3, {"all": 300}, lambda seg, pool, tau: SIGMA[pool][tau])

    table = spurious_ev_table(drift_sd_by_tau(frame), rates=rates)

    keys = {(p, float(t)) for p, t in zip(table["pool"], table["tau_min"], strict=True)}
    assert keys == {(p, t) for p in POOLS for t in GRID}
    for _, row in table.iterrows():
        pool, tau = row["pool"], float(row["tau_min"])
        planted = rates[pool] * math.expm1(SIGMA[pool][tau] ** 2)
        assert row["spurious_ev"] == pytest.approx(planted, rel=5e-3), (pool, tau)
        assert row["spurious_ev"] == pytest.approx(
            rates[pool] * math.expm1(row["sd"] ** 2), rel=1e-12
        )
