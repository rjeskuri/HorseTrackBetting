"""The pre-registered Family P selection study (ticket #88; 08 §5.2 end to end, §5.3, §8 A6).

Contract pinned here (`horsetrack.countries.japan.studies.selection`), as amended on 2026-11-27 (the
owner's reframe: keep every result and rank instead of gate; the restart plan on #88), with the
owner's answers of 2026-11-26 that still hold (benchmark, omega basis, one-sided passes, bands):

- `RuleSpec(rule_id, claim, segment, selection_cols, thresholds, direction)` and
  `FamilySpec(family_id, rules, decision_time, cluster_unit, benchmark, max_stat,
  odds_band_cut_points, dropped_variants=[])` are dataclasses. `claim` is 'beta', 'alpha' or
  'omega'; `direction` ("above" or "below") is the side of the claim's null the rule bets (a
  stage-1 pass needs the coefficient on that side); `max_stat` is `{"level", "block_size",
  "n_boot"}` for #75's `spa` and `stepm`; `odds_band_cut_points` cut the omega model's
  final-odds bands; `dropped_variants` are variants screened and dropped before the freeze, which
  `run_study` records as failed variants of the family (08 §5.2 step 1: the family size counts
  them). `decision_time` is "T-n": n minutes before the race's post. The values below are test
  inputs, not the owner's.
- Frozen means `registry.freeze_family(spec.family_id, dataclasses.asdict(spec))`. `run_study`
  raises `FamilyNotFrozen` before it scores or writes anything unless the registry holds exactly
  this spec's hash: an unfrozen family, or any edit after the freeze, is refused, and the study
  never freezes the family itself. No owner sign-off is required anywhere: `load_family("japan")`
  reads the committed config/selection_family.yaml with or without an `owner_signoff` (absent or
  null), its `dropped` variants carried as the spec's `dropped_variants`.
- `data` is one row per runner, development rows only, one winner per race:
    race_id, meeting_id  the race and its meeting (venue x day); `cluster_unit` names meeting_id
    race_date        the race day (pd.Timestamp); a row on a holdout day
                     (`core.eval.holdout.is_holdout_day`) is a ValueError before anything is
                     scored or written
    post_utc         the race's post time, timezone-aware UTC (the races view's derived post_utc).
                     The race's decision instant t is post_utc minus the spec's decision_time
    horse_id, odds_t the runner, and its decimal Win odds on the board at t (selection may use it)
    odds_final       final decimal Win odds: the log-loss stage and the omega model condition on it
    win_dividend     the Win dividend per unit stake if the runner wins (a little off odds_final,
                     so a return provably reads the dividend); a loser's may be missing (NaN)
    won, takeout     1 for the winner, else 0; the Win pool's takeout t (0.2, or mixed by band)
    final_*          final-odds band, rank and entropy: look-ahead, never required
    seg, jockey      segment columns: `seg` is constant within a race, `jockey` varies within it
- Selection inputs (the narrow contract for this family's inputs). Every column a rule reads, in
  its segment or its selection, is declared in the frame beside it: `<col>__source`, where the
  value comes from, and `<col>__available_utc`, the timezone-aware UTC instant each row's value
  became available. Admitted sources are the board at t (`board_at_t`, docs/specs/board-at-t.md)
  and the pre-race card tables read through an as-of view at t (`races`, `runners`;
  docs/specs/as-of.md). A rule is refused with `LookAheadSelection`, before anything is scored or
  written, when a column it reads is undeclared (no source or no stamp), comes from any other
  source (a post-race table such as the Win payout table `dividends`, `sectionals`, `incidents`,
  `running_comments`; `jrdb_post_race`; `horses`; the final board), or holds a value whose stamp
  is missing or later than its race's decision instant (a value stamped exactly at t is
  available). A missing value stamped after t (a board update after t blanking a price) is a late
  observation and refused the same way, with or without a source; a value with neither source
  nor stamp (an unmatched left join) is genuinely missing, accepted, and never selected. A price
  column may be object dtype, pd.NA for a missing price. The name rules stay, declared or not:
  `odds_final`, every `final_*` column, the store's `win_odds` and `win_fav_rank`, and every
  label of Japan's availability config
  (`core.store.asof.load_availability`) are refused as a segment or a selection column. Outcome
  columns (won, win_dividend, odds_final, takeout) are read only to score, never declared.
- Evaluation population: every return the study reports (bet counts, omega, the expected return,
  P(omega_s > 1/(1 - t)), the realised ROI, the predictive interval) is read on the rule's
  selected bets among the rows D9's walk-forward folds score out of fold (config/folds.yaml: from
  2020-Q1 here), never on training-only rows.
- `run_study(...) -> StudyReport(family_size, stages, survivors, p_beats_takeout, rungs)`. No
  stage gates another: every configured rule has an entry in every stage with each measurement
  its data allow, whatever its log-loss or max-statistic result, and a measurement that cannot be
  made leaves its number non-finite (or absent) and carries its reason as a non-empty `"error"`
  string in that entry.
  - `stages` holds "log_loss", "max_stat", "omega" and "roi" in that order (others allowed).
    `log_loss[rule_id]`: `passed` (bool), `delta_ll` with its bootstrap bounds `lo` and `hi`.
    `max_stat`: `tested` (every rule whose claim was scored) and `superior` (StepM's verdict).
    When SPA or StepM raises, `max_stat` carries the reason as its `"error"`, finds no rule
    superior, and every other stage is measured as before.
    `omega[rule_id]`: `n_bets`; `omega_selected`, the posterior mean of the selected bets'
    payout-weighted return multiplier, sum((1 - t_i) * omega_i) / sum(1 - t_i); `expected_return`,
    `{"mean", "low", "high", "level"}`: the posterior mean and central credible interval of the
    bets' mean expected return per unit stake at the final price, mean((1 - t_i) * omega_i)
    (break-even 1), so omega_selected * (1 - mean t_i) is its mean; `p_beats_takeout`, the
    posterior probability that the same quantity exceeds 1, from the same draws; a
    `breakeven_omega`, when reported, is 1 / (1 - mean t_i); `predictive_roi_100`,
    `[low, high, level]`, the posterior predictive interval of the ROI of the next 100 bets (08
    §5.2 step 4), reported apart from the expected return's interval. `roi[rule_id]`:
    `{"roi", "n_bets"}`, roi = (sum of the selected winners' win_dividend) / n_bets - 1.
  - `survivors`: a diagnostic, the rules that passed the log-loss stage and that StepM finds
    superior. It gates nothing.
  - `p_beats_takeout[rule_id]`, for every rule: `stages["omega"][rule_id]["p_beats_takeout"]`.
  - `rungs[rule_id]`: "S0" (D19's default, avoid) for every rule, survivor or not: the study makes
    no rung decision (rules move through the leaderboard, #449, and the owner); nothing above S1,
    nothing adopted.
  - `family_size`: `registry.family_size(family_id)` after the run: the rules, the spec's
    `dropped_variants` and the variants already in the family, failed ones included.
  - Each rule's last registry record carries, as metrics, every measurement made:
    `delta_ll_vs_calibrated_market` (lo, hi), `stepm_superior` (1.0 or 0.0), `omega_selected`,
    `expected_return` (lo = low, hi = high), `p_beats_takeout` and `roi_win_dividend`, and the
    predictive interval apart: `predictive_roi_100` (lo, hi its bounds, value a point between
    them) and `predictive_roi_100_level` (its level); no `stepm_superior` when StepM did not
    run. Its `notes` (`RunRecord.notes`, run metadata outside the variant identity that
    `family_size` counts) map each stage ('log_loss', 'max_stat', 'omega', 'roi') whose
    measurement for the rule could not be made to the reason its report entry gives, and hold
    nothing for a measurement made.
  - The holdout passed in is never opened. A rerun whose frame cannot score a rule the registry
    holds scored is a ValueError naming it, before any record.
- `sources` is validated before anything is scored or written: a bare str or bytes, a collection
  holding anything but non-blank names, or an empty one is refused (TypeError or ValueError). A
  collection naming a licensed source in any spelling keeps the registry's repository-write guard
  and the summary's redaction.
- `core.eval.segment_roi.SegmentROI.expected_return(segment)`: `.mean`, `.low`, `.high`,
  `.level` of the posterior of the segment's mean expected return per unit stake,
  mean((1 - t_i) * omega_i), the quantity `p_beats_takeout` judges against 1.

Not pinned: the log-loss bootstrap's level and replicates, the form of a claim against the spline,
the omega model's draws, a price on a cut point, registry fold labels, the status of a rule whose
claim could not be scored, the order of the pre-scoring checks.

All data is synthetic and seeded. Expected values come from the generator's own arrays (selected
rows, realised wins, a conditional-logit Newton written here), never from the code under test.
Each test first asserts the premise it relies on. Rule ids start with the Family P candidate they
belong to (P3, P4, P6). The families are test-only and decide nothing.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import dataclasses
import math
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from horsetrack.core import env
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import (
    DEFAULT_COUNTRIES_DIR,
    LOG_NAME,
    LockedHoldout,
    is_holdout_day,
)
from horsetrack.core.eval.registry import WITHHELD_LABEL, Registry, RunRecord, write_summary
from horsetrack.core.eval.reports import LicensedOutputInRepo
from horsetrack.core.eval.segment_roi import fit_segment_roi
from horsetrack.core.eval.snooping import stepm
from horsetrack.core.store.asof import Gate, Via, load_availability
from horsetrack.countries.japan.models.v0.calibration import fit_win_calibration
from horsetrack.countries.japan.studies.selection import (
    FamilyNotFrozen,
    FamilySpec,
    LookAheadSelection,
    RuleSpec,
    load_family,
    run_study,
)

TAKEOUT = 0.2
N_MEETINGS = 240
RACES_PER_MEETING = 6
FIRST_DAY = date(2014, 6, 7)  # D10: no price before this day
LAST_DAY = date(2025, 12, 31)  # before the holdout (2026-11-28); 2020 on is in D9's scored folds
FIRST_SCORED = pd.Timestamp(2020, 1, 1)  # config/folds.yaml: the first scored quarter, 2020-Q1
SEGMENTS = ("A", "B", "C", "D")
# J1 rides only in segment A's races (a quarter of the runners there) and J2 and J3 never do, so
# J1's edge, paid for by the other runners of its races, leaves J2's and J3's prices exact.
J1_SHARE_IN_A = 0.25
A_JOCKEYS = ("J4", "J5", "J6", "J7", "J8")
OTHER_JOCKEYS = ("J2", "J3", "J4", "J5", "J6", "J7", "J8")
BOARD_NOISE = 0.10  # sd of ln(odds at t / final odds)
DIVIDEND_NOISE = 0.02  # sd of ln(dividend / final odds)

REAL_BETA = 1.8  # segment A's favourites are underbet: P(win) proportional to q^1.8
NULL_BETA = dict.fromkeys(SEGMENTS, 1.0)
J1_BOOST = math.log(4.0)  # J1's runners win 4 times as often as their price says, before
# normalising within the race (about twice as often after it)
# A market-wide favourite-longshot bias: P(win) proportional to q^1.3 in every race. In the
# direction market segment A's favourites win more often than that (q^(1.3 * 1.6)) and segment
# B's longshots do (q^(1.3 * 0.55)); C and D carry the market-wide bias only.
MARKET_BETA = 1.3
FLB_BETA = dict.fromkeys(SEGMENTS, MARKET_BETA)
DIRECTION_BETA = {**FLB_BETA, "A": MARKET_BETA * 1.6, "B": MARKET_BETA * 0.55}

MAIN_SEED = 8802
OMEGA_SEED = 8803
FLB_SEED = 8804
DIRECTION_SEED = 8805
FINISH_SEED = 8806  # the finishing order behind the synthetic result-label columns
MIXED_SEED = 8807  # the bets of the segment-ROI mixed-takeout cases
STUDY_SEED = 88
BENCHMARK = "calibrated_market"
MAX_STAT = {"level": 0.05, "block_size": 4, "n_boot": 500}  # test inputs, not the owner's
CUT_POINTS = [3.0, 8.0, 30.0]  # test input: the odds bands of `final_odds_band` below
LOOK_AHEAD = ("odds_final", "final_odds_band", "final_rank", "final_entropy")
HOLDOUT_LOG = DEFAULT_COUNTRIES_DIR / "japan" / LOG_NAME
FAMILY_FILE = DEFAULT_COUNTRIES_DIR / "japan" / "config" / "selection_family.yaml"
CLEAR_Z = 2.5  # a selection's realised return this many standard errors from breakeven
# A reported expected return within this many standard errors of the bets' realised return at the
# final odds (an unbiased estimate of the same mean).
MIX_Z = 3.5
FOLLOW_UP_RUN = "#462"  # the development-fold run of the committed family
PREDICTIVE = "predictive_roi_100"

# The selection-input declaration (module docstring): a column's source and availability stamp.
SOURCE, AVAILABLE = "__source", "__available_utc"
BOARD, RACES, RUNNERS = "board_at_t", "races", "runners"
DECISION_MINUTES = 5  # make_family's decision_time, "T-5"
# The finish (runners finish_pos, dead_heat, finish_time_s; races finish_time_s): since #305 (the
# owner's answer on #96) gated in the views at post_utc + 30 min instead of label columns.
POST_UTC = "post_utc"
FINISH = ("finish_pos", "dead_heat", "finish_time_s")
# Sources a selection input can never come from: the race's results (the Win payout table and the
# other post-race tables of the availability config), a label-only or current-value table, and the
# final board.
REFUSED_SOURCES = (
    "dividends",
    "sectionals",
    "incidents",
    "running_comments",
    "jrdb_post_race",
    "horses",
    "final_board",
)

# The metrics a rule's last registry record carries for its measurements (module docstring).
LOG_LOSS_METRIC = "delta_ll_vs_calibrated_market"
RECORD_METRICS = (
    LOG_LOSS_METRIC,
    "stepm_superior",
    "omega_selected",
    "expected_return",
    "p_beats_takeout",
    "roi_win_dividend",
)
RETURN_METRICS = ("omega_selected", "expected_return", "p_beats_takeout", "roi_win_dividend")

# (rule_id, claim, segment, selection_cols, thresholds, direction); only segment A's claim is
# real. Every rule backs favourites; against the calibrated market (pulled towards A's bias) B, C
# and D depart a little the other way, which no favourite rule may claim.
MAIN_RULES = [
    ("P4-seg-a", "beta", {"seg": "A"}, ["odds_t"], {"odds_t": [1.5, 4.0]}, "above"),
    ("P4-seg-b", "beta", {"seg": "B"}, ["odds_t"], {"odds_t": [1.5, 4.0]}, "above"),
    ("P6-seg-c", "beta", {"seg": "C"}, ["odds_t"], {"odds_t": [None, 3.0]}, "above"),
    ("P4-seg-d", "beta", {"seg": "D"}, ["odds_t"], {"odds_t": [None, 2.5]}, "above"),
]
MAIN_REAL = "P4-seg-a"
MAIN_DROPPED = ["P4-seg-a-wide", "P5-overlay"]

# Only J1's claim is real; J1 has a genuine positive edge at the final dividend.
OMEGA_RULES = [
    ("P4-j1", "omega", {"jockey": "J1"}, ["odds_t"], {"odds_t": [3.0, None]}, "above"),
    ("P4-j2", "omega", {"jockey": "J2"}, ["odds_t"], {"odds_t": [3.0, None]}, "above"),
    ("P4-j3", "omega", {"jockey": "J3"}, ["odds_t"], {"odds_t": [None, 10.0]}, "above"),
]
OMEGA_REAL = "P4-j1"
OMEGA_DROPPED = ["P4-j1-late"]

FAVOURITES, LONGSHOTS = {"odds_t": [None, 3.0]}, {"odds_t": [8.0, None]}
# The market-wide bias alone: every segment is priced like every other.
FLB_RULES = [
    ("P4-a-favourite", "beta", {"seg": "A"}, ["odds_t"], FAVOURITES, "above"),
    ("P4-b-favourite", "beta", {"seg": "B"}, ["odds_t"], {"odds_t": [1.5, 4.0]}, "above"),
    ("P3-c-longshot", "beta", {"seg": "C"}, ["odds_t"], LONGSHOTS, "below"),
    ("P6-d-favourite", "beta", {"seg": "D"}, ["odds_t"], FAVOURITES, "above"),
]
# Each departing segment bet both ways, and a segment with the market-wide bias only. A's
# favourites start at 1.5: on the scored rows alone, a winner near 1.1 would put the omega fit of
# its bets on the p = 1 boundary (#78), and this family needs every rule's omega measured.
DIRECTION_RULES = [
    ("P4-a-favourite", "beta", {"seg": "A"}, ["odds_t"], {"odds_t": [1.5, 3.0]}, "above"),
    ("P3-a-longshot", "beta", {"seg": "A"}, ["odds_t"], LONGSHOTS, "below"),
    ("P3-b-longshot", "beta", {"seg": "B"}, ["odds_t"], LONGSHOTS, "below"),
    ("P4-b-favourite", "beta", {"seg": "B"}, ["odds_t"], FAVOURITES, "above"),
    ("P6-c-favourite", "beta", {"seg": "C"}, ["odds_t"], FAVOURITES, "above"),
]
DIRECTION_SURVIVORS = {"P4-a-favourite", "P3-b-longshot"}

# The committed family as the owner approved it on #88 (2026-11-26).
OWNER_RULES = {
    "P3-handicap-5-50": ({"weight_type": "handicap"}, ["odds_t"], {"odds_t": [5.0, 50.0]}, "below"),
    "P4-r12-favourite": ({"R12": "in"}, ["rank_t"], {"rank_t": [1, 1]}, "above"),
    "P6-concentrated-favourite": ({"entropy_t_bin": 1}, ["rank_t"], {"rank_t": [1, 1]}, "above"),
}
OWNER_MAX_STAT = {"level": 0.05, "block_size": 6, "n_boot": 10000}

# Mixed takeout. In the study's mixed frame the Win takeout depends on the final-odds band of
# CUT_POINTS (<3, 3-8, 8-30, 30+), so a rule's bets mix payout rates across bands of different
# multipliers. In the segment-ROI cases each band's bets share a takeout and a true multiplier
# (band, takeout, omega, odds cycled): the high multiplier sits on the low payout rate (the
# unweighted mean multiplier clears 1/(1 - mean t) while the expected return is below 1), or on
# the high one (the reverse). Takeouts far apart, so the two land clearly apart.
TAKEOUT_BY_BAND = (0.40, 0.05, 0.40, 0.05)
MIXED_BETS = 4000  # per band
MIXED_CASES = {
    "below": [("b1", 0.62, 2.52, (1.8, 2.0, 2.2)), ("b2", 0.02, 0.90, (3.5, 4.0, 4.5))],
    "above": [("b1", 0.62, 0.90, (1.8, 2.0, 2.2)), ("b2", 0.02, 1.855, (3.5, 4.0, 4.5))],
}
TRAINING_DIVIDEND = 40.0  # the big winning bets planted in training-only races


# ---------------------------------------------------------------------------------------------
# Synthetic runners and their selection-input declarations
# ---------------------------------------------------------------------------------------------


def decision_utc(frame: pd.DataFrame, minutes: int = DECISION_MINUTES) -> pd.Series:
    """Each row's decision instant t: its race's post minus the decision time."""
    return frame["post_utc"] - pd.Timedelta(minutes=minutes)


def board_stamp(frame: pd.DataFrame) -> pd.Series:
    """When each race's board at t was read: at t itself in every fourth race, else 20, 40 or 60
    seconds before it."""
    k = pd.factorize(frame["race_id"])[0] % 4
    return decision_utc(frame) - pd.Series(pd.to_timedelta(20 * k, unit="s"), index=frame.index)


def card_stamp(frame: pd.DataFrame) -> pd.Series:
    """00:00 JST of the race date, when the races and runners views open a day's card."""
    return frame["race_date"].dt.tz_localize("Asia/Tokyo").dt.tz_convert("UTC")


def declare(frame: pd.DataFrame, **sources: str) -> pd.DataFrame:
    """`frame` with each named column declared: its source, and its availability stamp (the
    board's read time for the board at t, the card's opening for a card table)."""
    out = frame.copy()
    for col, source in sources.items():
        out[col + SOURCE] = source
        out[col + AVAILABLE] = board_stamp(out) if source == BOARD else card_stamp(out)
    return out


def make_runners(seed: int, *, beta: dict[str, float], jockey_boost: float = 0.0) -> pd.DataFrame:
    """N_MEETINGS meetings of RACES_PER_MEETING races, 8 to 14 runners each (module docstring).

    Each race's final odds are (1 - t) / share for a clipped Dirichlet(1) share, rounded to 0.1.
    With q the normalised inverse final odds, the winner is drawn from softmax(beta[seg] * ln q +
    jockey_boost * [jockey == J1]), so beta 1 and no boost is a race the final price gets right.
    Segments cycle over the races; J1 rides only in segment A's races, J2 and J3 only elsewhere.
    The post is 01:00 UTC plus 35 minutes per race; odds_t, seg and jockey are declared.
    """
    rng = np.random.default_rng(seed)
    span = (LAST_DAY - FIRST_DAY).days
    offsets = np.sort(rng.choice(span + 1, size=N_MEETINGS, replace=False))
    frames = []
    for m, offset in enumerate(offsets):
        day = pd.Timestamp(FIRST_DAY + timedelta(days=int(offset)))
        for r in range(RACES_PER_MEETING):
            n = int(rng.integers(8, 15))
            share = rng.gamma(1.0, size=n)
            share /= share.sum()
            for _ in range(3):
                share = np.clip(share, 0.004, 0.7)
                share /= share.sum()
            odds_final = np.maximum(np.round((1 - TAKEOUT) / share, 1), 1.1)
            q = (1 / odds_final) / (1 / odds_final).sum()
            segment = SEGMENTS[(m * RACES_PER_MEETING + r) % len(SEGMENTS)]
            if segment == "A":
                others = rng.choice(A_JOCKEYS, size=n)
                jockey = np.where(rng.random(n) < J1_SHARE_IN_A, "J1", others)
            else:
                jockey = rng.choice(OTHER_JOCKEYS, size=n)
            utility = beta[segment] * np.log(q) + jockey_boost * (jockey == "J1")
            won = np.zeros(n, dtype=int)
            won[np.argmax(utility + rng.gumbel(size=n))] = 1
            board = odds_final * np.exp(rng.normal(0.0, BOARD_NOISE, size=n))
            paid = odds_final * np.exp(rng.normal(0.0, DIVIDEND_NOISE, size=n))
            frames.append(
                pd.DataFrame(
                    {
                        "race_id": f"m{m:03d}-r{r + 1:02d}",
                        "meeting_id": f"m{m:03d}",
                        "race_date": day,
                        "post_utc": day.tz_localize("UTC") + pd.Timedelta(hours=1, minutes=35 * r),
                        "horse_id": np.arange(1, n + 1),
                        "seg": segment,
                        "jockey": jockey,
                        "odds_t": np.maximum(np.round(board, 1), 1.1),
                        "odds_final": odds_final,
                        "win_dividend": np.maximum(np.round(paid, 1), 1.0),
                        "won": won,
                        "takeout": TAKEOUT,
                        "final_odds_band": np.digitize(odds_final, CUT_POINTS) + 1,
                        "final_rank": pd.Series(odds_final).rank(method="min").to_numpy(int),
                        "final_entropy": float(-(q * np.log(q)).sum() / np.log(n)),
                    }
                )
            )
    runners = pd.concat(frames, ignore_index=True)
    return declare(runners, odds_t=BOARD, seg=RACES, jockey=RUNNERS)


def scored(data: pd.DataFrame) -> pd.DataFrame:
    """The rows D9's walk-forward folds score out of fold (config/folds.yaml), in frame order:
    the population every reported return is read on."""
    folds = walk_forward(
        data["race_date"].reset_index(drop=True), scheme=load_country_scheme("japan")
    )
    rows = np.concatenate([fold.test_idx for fold in folds if len(fold.test_idx)])
    return data.iloc[np.sort(rows)]


def implied(frame: pd.DataFrame) -> pd.Series:
    """q: the normalised inverse final odds within each race."""
    inverse = 1.0 / frame["odds_final"]
    return inverse / inverse.groupby(frame["race_id"]).transform("sum")


def beta_z(frame: pd.DataFrame) -> float:
    """(beta_hat - 1) / se of P(win) proportional to q^beta over frame's races (one winner each),
    by Newton on the conditional-logit score and information written out here."""
    x = np.log(implied(frame).to_numpy())
    codes = pd.factorize(frame["race_id"])[0]
    won = frame["won"].to_numpy(dtype=bool)
    beta, info = 1.0, 0.0
    for _ in range(50):
        u = beta * x
        top = pd.Series(u).groupby(codes).transform("max").to_numpy()
        w = np.exp(u - top)
        total = np.bincount(codes, weights=w)
        mean = np.bincount(codes, weights=w * x) / total
        info = float((np.bincount(codes, weights=w * x * x) / total - mean**2).sum())
        step = (x[won].sum() - mean.sum()) / info
        beta += step
        if abs(step) < 1e-12:
            break
    return (beta - 1.0) * math.sqrt(info)


def power_fit(frame: pd.DataFrame, x: np.ndarray) -> tuple[float, float]:
    """The maximum-likelihood b of P(win) proportional to exp(b * x) within frame's races, and
    its se (the same Newton as beta_z, on any covariate x aligned with frame's rows)."""
    codes = pd.factorize(frame["race_id"])[0]
    won = frame["won"].to_numpy(dtype=bool)
    b, info = 1.0, 0.0
    for _ in range(50):
        u = b * x
        top = pd.Series(u).groupby(codes).transform("max").to_numpy()
        w = np.exp(u - top)
        total = np.bincount(codes, weights=w)
        mean = np.bincount(codes, weights=w * x) / total
        info = float((np.bincount(codes, weights=w * x * x) / total - mean**2).sum())
        step = (x[won].sum() - mean.sum()) / info
        b += step
        if abs(step) < 1e-12:
            break
    return b, 1.0 / math.sqrt(info)


def departure_z(data: pd.DataFrame, segment: str) -> float:
    """The segment's beta over the market-wide one, as a z-score: the power fit on every race of
    `data` stands in for the calibrated market (the generator's truth is a power law in q, so a
    spline in ln q fits the same curve), and the segment's races are refitted on top of it.
    Positive: the segment's favourites win more often than the market-wide fit says."""
    x = np.log(implied(data).to_numpy())
    market, _ = power_fit(data, x)
    mask = (data["seg"] == segment).to_numpy()
    b, se = power_fit(data[mask], market * x[mask])
    return (b - 1.0) / se


def omega_premise(frame: pd.DataFrame, mask: pd.Series) -> tuple[float, float]:
    """Realised wins over the wins the final price expects, and that excess as a z-score."""
    q = implied(frame)[mask]
    wins = float(frame.loc[mask, "won"].sum())
    return wins / q.sum(), (wins - q.sum()) / math.sqrt((q * (1 - q)).sum())


def selected(data: pd.DataFrame, rule: tuple) -> pd.DataFrame:
    """The rows a rule bets on: its segment, every selection column present and in [lo, hi]."""
    _, _, segment, selection_cols, thresholds = rule[:5]
    keep = pd.Series(True, index=data.index)
    for col, level in segment.items():
        keep &= data[col] == level
    for col in selection_cols:
        lo, hi = thresholds[col]
        keep &= data[col].notna()
        if lo is not None:
            keep &= data[col] >= lo
        if hi is not None:
            keep &= data[col] <= hi
    return data[keep]


def bets_of(data: pd.DataFrame, rule: tuple) -> pd.DataFrame:
    """The rule's selected bets in the evaluation population."""
    return selected(scored(data), rule)


def realised_roi(rows: pd.DataFrame) -> float:
    return float((rows["won"] * rows["win_dividend"]).sum() / len(rows) - 1.0)


def roi_z(rows: pd.DataFrame) -> float:
    """The selection's realised return per unit stake over its standard error (0 = breakeven)."""
    pay = rows["won"] * rows["win_dividend"].fillna(0.0) - 1.0
    return float(pay.mean() / (pay.std(ddof=1) / math.sqrt(len(pay))))


def gross_return(rows: pd.DataFrame) -> tuple[float, float]:
    """The bets' realised return per unit stake at the final odds, and its standard error."""
    pay = rows["won"] * rows["odds_final"]
    return float(pay.mean()), float(pay.std(ddof=1) / math.sqrt(len(pay)))


def finish_positions(data: pd.DataFrame) -> np.ndarray:
    """A finishing order per race: the winner 1, the other runners 2 to n in a seeded order."""
    rng = np.random.default_rng(FINISH_SEED)
    key = -data["won"].to_numpy(dtype=float) + 0.5 * rng.random(len(data))
    return pd.Series(key).groupby(data["race_id"].to_numpy()).rank(method="first").to_numpy()


def is_availability_label(name: str) -> bool:
    """Whether Japan's availability config keeps the column out of every view at t."""
    return any(rule.is_label(name) for rule in load_availability("japan").values())


def availability_labels() -> list[str]:
    """Every label column Japan's availability config names (any table's `label_columns`), and
    one name under each of its `label_prefixes`."""
    rules = load_availability("japan").values()
    names = {col for rule in rules for col in rule.label_columns}
    names |= {f"{prefix}win_odds" for rule in rules for prefix in rule.label_prefixes}
    return sorted(names)


def post_gates(rules=None) -> dict[str, timedelta]:
    """Every column a gate of Japan's availability config (or of `rules`) anchors at its race's
    post (`gated_columns` with anchor post_utc, the races row's own or read through `via`), with
    the gate's offset: the latest one when two tables gate the same name."""
    offsets: dict[str, timedelta] = {}
    for rule in (load_availability("japan") if rules is None else rules).values():
        for gate in rule.gates:
            if gate.anchor == POST_UTC:
                for col in gate.columns:
                    offsets[col] = max(gate.offset, offsets.get(col, gate.offset))
    return offsets


def gated_after(minutes: int) -> list[str]:
    """The post-gated columns a view at t = post - `minutes` still holds null: offset > -minutes,
    the gate opening after the decision instant (the ruling on #88)."""
    return sorted(c for c, off in post_gates().items() if off > timedelta(minutes=-minutes))


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def explained(entry: dict) -> bool:
    """Whether a stage entry carries the reason a measurement could not be made."""
    return isinstance(entry.get("error"), str) and bool(entry["error"].strip())


# ---------------------------------------------------------------------------------------------
# Families, registry and holdout doubles
# ---------------------------------------------------------------------------------------------


def make_rule(rule_id, claim, segment, selection_cols, thresholds, direction) -> RuleSpec:
    return RuleSpec(
        rule_id=rule_id,
        claim=claim,
        segment=dict(segment),
        selection_cols=list(selection_cols),
        thresholds={col: list(bounds) for col, bounds in thresholds.items()},
        direction=direction,
    )


def make_family(family_id: str, rules: list[tuple], minutes: int = DECISION_MINUTES) -> FamilySpec:
    return FamilySpec(
        family_id=family_id,
        rules=[make_rule(*rule) for rule in rules],
        decision_time=f"T-{minutes}",  # five minutes before the off by default: a test value
        cluster_unit="meeting_id",
        benchmark=BENCHMARK,
        max_stat=dict(MAX_STAT),
        odds_band_cut_points=list(CUT_POINTS),
    )


def new_registry(root: Path) -> Registry:
    log = root / "countries" / "japan" / LOG_NAME
    log.parent.mkdir(parents=True, exist_ok=True)
    return Registry(root / "data" / "registry" / "japan.jsonl", holdout_log=log)


def freeze(registry: Registry, spec: FamilySpec) -> str:
    return registry.freeze_family(spec.family_id, dataclasses.asdict(spec))


def add_dropped(registry: Registry, family_id: str, variants: list[str]) -> None:
    """Variants tried and dropped before the family was frozen: failed records of the family."""
    for variant in variants:
        registry.append(
            RunRecord(
                hypothesis_id=variant.split("-")[0],
                family_id=family_id,
                variant=variant,
                ticket=88,
                git_sha="0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c",
                data_builds={"runners": "synthetic"},
                sources=("synthetic",),
                fold_scheme="synthetic",
                components={"rule": variant},
                params={"dropped": True},
                metrics=(),
                status="failed",
            )
        )


class HoldoutOpened(BaseException):
    """Raised by the double's open; a BaseException so `except Exception` cannot swallow it."""


class TrippedHoldout(LockedHoldout):
    """A LockedHoldout whose open records the call and raises."""

    def __init__(self) -> None:  # no rows: there is nothing behind this holdout
        self.calls: list[dict] = []

    def open(self, **kwargs) -> pd.DataFrame:
        self.calls.append(kwargs)
        raise HoldoutOpened(f"the study opened the holdout: {kwargs}")

    def __repr__(self) -> str:
        return "TrippedHoldout()"


class ScoringStarted(BaseException):
    """Raised when the study fits its first benchmark: a refusal must come before it."""


def _no_scoring(*args, **kwargs):
    raise ScoringStarted("the study began scoring (it fitted the calibrated market)")


def read_holdout_log() -> bytes | None:
    return HOLDOUT_LOG.read_bytes() if HOLDOUT_LOG.exists() else None


@dataclass
class Run:
    rules: list[tuple]
    spec: FamilySpec
    data: pd.DataFrame
    registry: Registry
    holdout: TrippedHoldout
    report: object
    log_before: bytes | None
    log_after: bytes | None


def run_spec(root: Path, spec: FamilySpec, data, dropped=(), rules=(), **kwargs) -> Run:
    registry = new_registry(root)
    add_dropped(registry, spec.family_id, list(dropped))
    freeze(registry, spec)
    holdout = TrippedHoldout()
    before = read_holdout_log()
    with pytest.MonkeyPatch.context() as patch:
        (root / "data").mkdir(parents=True, exist_ok=True)
        patch.setenv("HORSETRACK_DATA_DIR", str(root / "data"))
        report = run_study(
            spec, data.copy(), registry=registry, holdout=holdout, seed=STUDY_SEED, **kwargs
        )
    return Run(list(rules), spec, data, registry, holdout, report, before, read_holdout_log())


def run_frozen(root: Path, family_id: str, rules: list[tuple], data, dropped=(), **kwargs) -> Run:
    return run_spec(root, make_family(family_id, rules), data, dropped, rules, **kwargs)


def _replace_everywhere(patch: pytest.MonkeyPatch, original, fake) -> None:
    """Point every loaded horsetrack module's name for `original` at `fake`."""
    for name, module in list(sys.modules.items()):
        if name.startswith("horsetrack") and getattr(module, original.__name__, None) is original:
            patch.setattr(module, original.__name__, fake)


def outcome_before_scoring(
    root: Path, name: str, rules: list[tuple], data, *, minutes: int = DECISION_MINUTES, **kwargs
) -> str:
    """How run_study treats a frozen family (decision time T-`minutes`) on `data` when scoring
    would raise: 'look-ahead' (LookAheadSelection), 'scored' (it began scoring), 'completed', or
    another exception's type. Whatever the outcome, no record was written and the holdout was
    not opened."""
    registry = new_registry(root / name)
    spec = make_family(f"test-88-{name}", rules, minutes)
    freeze(registry, spec)
    holdout = TrippedHoldout()
    with pytest.MonkeyPatch.context() as patch:
        _replace_everywhere(patch, fit_win_calibration, _no_scoring)
        try:
            run_study(spec, data, registry=registry, holdout=holdout, seed=1, **kwargs)
        except LookAheadSelection:
            outcome = "look-ahead"
        except ScoringStarted:
            outcome = "scored"
        except (TypeError, ValueError) as exc:
            outcome = type(exc).__name__
        else:
            outcome = "completed"
    assert registry.records(spec.family_id) == [], (name, outcome)
    assert holdout.calls == [], name
    return outcome


def last_metrics(run: Run, rid: str) -> dict[str, list]:
    """The metrics of the rule's last registry record, by name."""
    records = [r for r in run.registry.records(run.spec.family_id) if r.variant == rid]
    assert records, f"{rid}: no registry record"
    metrics: dict[str, list] = {}
    for metric in records[-1].metrics:
        metrics.setdefault(metric.metric, []).append(metric)
    return metrics


def omega_numbers(entry: dict) -> dict[str, float]:
    """The omega stage's measurements of one rule, flat."""
    er, pred = entry["expected_return"], entry[PREDICTIVE]
    return {
        "n_bets": entry["n_bets"],
        "omega_selected": entry["omega_selected"],
        "p_beats_takeout": entry["p_beats_takeout"],
        "expected_mean": er["mean"],
        "expected_low": er["low"],
        "expected_high": er["high"],
        "predictive_low": pred[0],
        "predictive_high": pred[1],
    }


@pytest.fixture(scope="module")
def main_runners() -> pd.DataFrame:
    return make_runners(MAIN_SEED, beta={**NULL_BETA, "A": REAL_BETA})


@pytest.fixture(scope="module")
def omega_runners() -> pd.DataFrame:
    return make_runners(OMEGA_SEED, beta=NULL_BETA, jockey_boost=J1_BOOST)


@pytest.fixture(scope="module")
def flb_runners() -> pd.DataFrame:
    return make_runners(FLB_SEED, beta=FLB_BETA)


@pytest.fixture(scope="module")
def direction_runners() -> pd.DataFrame:
    return make_runners(DIRECTION_SEED, beta=DIRECTION_BETA)


@pytest.fixture(scope="module")
def main_run(tmp_path_factory, main_runners) -> Run:
    root = tmp_path_factory.mktemp("main")
    return run_frozen(root, "test-88-segments", MAIN_RULES, main_runners, MAIN_DROPPED)


@pytest.fixture(scope="module")
def omega_run(tmp_path_factory, omega_runners) -> Run:
    root = tmp_path_factory.mktemp("omega")
    return run_frozen(root, "test-88-jockeys", OMEGA_RULES, omega_runners, OMEGA_DROPPED)


@pytest.fixture(scope="module")
def flb_run(tmp_path_factory, flb_runners) -> Run:
    root = tmp_path_factory.mktemp("flb")
    return run_frozen(root, "test-88-market-bias", FLB_RULES, flb_runners)


@pytest.fixture(scope="module")
def direction_run(tmp_path_factory, direction_runners) -> Run:
    root = tmp_path_factory.mktemp("direction")
    return run_frozen(root, "test-88-direction", DIRECTION_RULES, direction_runners)


def with_gaps(data: pd.DataFrame) -> pd.DataFrame:
    """`data` with a second race-level segment column `seg2` ('X', 'Y' or missing, a third of
    the races each, declared from the races card) and no dividend on any loser."""
    out = data.copy()
    k = pd.factorize(out["race_id"])[0] % 3
    out["seg2"] = pd.Series(np.where(k == 0, "X", "Y"), index=out.index, dtype=object)
    out.loc[k == 2, "seg2"] = None
    out.loc[out["won"] == 0, "win_dividend"] = np.nan
    return declare(out, seg2=RACES)


GAP_RULES = [
    MAIN_RULES[0],
    MAIN_RULES[1],
    ("P4-seg2-x", "beta", {"seg2": "X"}, ["odds_t"], {"odds_t": [None, 3.0]}, "above"),
]


@pytest.fixture(scope="module")
def gap_run(tmp_path_factory, main_runners) -> Run:
    root = tmp_path_factory.mktemp("gaps")
    return run_frozen(root, "test-88-gaps", GAP_RULES, with_gaps(main_runners))


def with_mixed_takeout(data: pd.DataFrame) -> pd.DataFrame:
    """`data` with the Win takeout of each runner's final-odds band (TAKEOUT_BY_BAND)."""
    band = np.digitize(data["odds_final"].to_numpy(), CUT_POINTS)
    return data.assign(takeout=np.asarray(TAKEOUT_BY_BAND)[band])


@pytest.fixture(scope="module")
def mixed_run(tmp_path_factory, omega_runners) -> Run:
    root = tmp_path_factory.mktemp("mixed")
    return run_frozen(root, "test-88-mixed-takeout", OMEGA_RULES, with_mixed_takeout(omega_runners))


def check_main_premise(data: pd.DataFrame) -> None:
    """Segment A's beta is far above 1; B, C and D's are within 1 se of 1, and none of them is
    above the market-wide fit (which A pulls up)."""
    z = {s: beta_z(data[data["seg"] == s]) for s in SEGMENTS}
    assert z["A"] > 5, z
    assert all(abs(z[s]) < 1.0 for s in ("B", "C", "D")), z
    rel = {s: departure_z(data, s) for s in SEGMENTS}
    assert rel["A"] > 5 and all(rel[s] < 1.0 for s in ("B", "C", "D")), rel


def assert_claims_scored(run: Run) -> None:
    """Every rule's claim was scored (no 'failed' record), so a verdict of 'not passed' is the
    stage's, not a claim that could not be fitted."""
    records = run.registry.records(run.spec.family_id)
    status = {(record.variant, record.status) for record in records}
    assert status == {(rule[0], "scored") for rule in run.rules}, status


def check_omega_premise(data: pd.DataFrame) -> None:
    """J1 wins far more often than its price says; J2 and J3 within 1 se of their price."""
    ratio, z = omega_premise(data, data["jockey"] == "J1")
    assert ratio > 1.6 and z > 5, (ratio, z)
    for jockey in ("J2", "J3"):
        _, z = omega_premise(data, data["jockey"] == jockey)
        assert abs(z) < 1.0, (jockey, z)


def check_flb_premise(data: pd.DataFrame) -> None:
    """Every segment's beta is far above 1 against the normalised price (the market-wide bias a
    normalised-q benchmark would credit to each segment), and within 1 se of the market-wide
    fit."""
    z = {s: beta_z(data[data["seg"] == s]) for s in SEGMENTS}
    assert all(z[s] > 3 for s in SEGMENTS), z
    rel = {s: departure_z(data, s) for s in SEGMENTS}
    assert all(abs(rel[s]) < 1.0 for s in SEGMENTS), rel


def check_direction_premise(data: pd.DataFrame) -> None:
    """On top of the market-wide fit, A's favourites and B's longshots win far more often than
    priced; C carries the market-wide bias only, which the normalised price would credit to it."""
    rel = {s: departure_z(data, s) for s in SEGMENTS}
    assert rel["A"] > 5 and rel["B"] < -5 and abs(rel["C"]) < 1.0, rel
    assert beta_z(data[data["seg"] == "C"]) > 3


# ---------------------------------------------------------------------------------------------
# DoD: the study refuses to score before the family is frozen
# ---------------------------------------------------------------------------------------------


def _edit_rule(spec: FamilySpec, k: int, **changes) -> FamilySpec:
    rules = list(spec.rules)
    rules[k] = dataclasses.replace(rules[k], **changes)
    return dataclasses.replace(spec, rules=rules)


def _edits(spec: FamilySpec) -> list[tuple[str, FamilySpec]]:
    """Each a single, self-consistent change to a frozen spec, every one runnable on the data."""
    extra = dataclasses.replace(spec.rules[1], rule_id="P4-seg-b-late")
    return [
        ("a threshold", _edit_rule(spec, 0, thresholds={"odds_t": [1.5, 4.5]})),
        ("an open end", _edit_rule(spec, 2, thresholds={"odds_t": [1.0, 3.0]})),
        ("a segment level", _edit_rule(spec, 1, segment={"seg": "C"})),
        ("a claim", _edit_rule(spec, 3, claim="omega")),
        ("a direction", _edit_rule(spec, 0, direction="below")),
        ("the selection", _edit_rule(spec, 0, selection_cols=[], thresholds={})),
        ("a rule id", _edit_rule(spec, 1, rule_id="P4-seg-b2")),
        ("a rule added", dataclasses.replace(spec, rules=[*spec.rules, extra])),
        ("a rule dropped", dataclasses.replace(spec, rules=spec.rules[:-1])),
        ("the decision time", dataclasses.replace(spec, decision_time="T-10")),
        ("the cluster unit", dataclasses.replace(spec, cluster_unit="race_date")),
        ("the StepM level", dataclasses.replace(spec, max_stat={**MAX_STAT, "level": 0.10})),
        ("the block length", dataclasses.replace(spec, max_stat={**MAX_STAT, "block_size": 1})),
    ]


def test_family_is_frozen_before_scoring(tmp_path, monkeypatch, main_runners):
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    spec = make_family("test-88-segments", MAIN_RULES)

    # Never frozen: refused, nothing scored, and the study does not freeze it itself.
    registry = new_registry(tmp_path / "unfrozen")
    with pytest.raises(FamilyNotFrozen):
        run_study(spec, main_runners.copy(), registry=registry, holdout=TrippedHoldout(), seed=1)
    assert registry.records() == []
    assert registry.frozen_spec_hash(spec.family_id) is None

    # The same content frozen under another family id does not freeze this one.
    registry.freeze_family("test-88-other", dataclasses.asdict(spec))
    with pytest.raises(FamilyNotFrozen):
        run_study(spec, main_runners.copy(), registry=registry, holdout=TrippedHoldout(), seed=1)
    assert registry.records() == []

    # Frozen, then edited: every edit is refused and the frozen hash stays the original.
    registry = new_registry(tmp_path / "edited")
    frozen = freeze(registry, spec)
    for what, edited in _edits(spec):
        assert dataclasses.asdict(edited) != dataclasses.asdict(spec), what
        with pytest.raises(FamilyNotFrozen):
            run_study(
                edited, main_runners.copy(), registry=registry, holdout=TrippedHoldout(), seed=1
            )
        assert registry.records() == [], what
        assert registry.frozen_spec_hash(spec.family_id) == frozen, what


# ---------------------------------------------------------------------------------------------
# DoD: every configured rule is reported with each measurement its data allow; nothing gates
# ---------------------------------------------------------------------------------------------


def test_every_rule_is_reported_whatever_its_log_loss_result(
    main_run, omega_run, direction_run, flb_run, gap_run
):
    """Survivor or not, every rule has its log-loss gain and interval, the StepM verdict, omega,
    the expected return with its interval, P(omega_s > 1/(1 - t)), the predictive interval and
    the realised ROI at the final dividend (read on its scored selected bets), in the report and
    in its last registry record. Survivors stay as a diagnostic: in each family where one rule's
    claim is real, that rule alone survives."""
    check_main_premise(main_run.data)
    check_omega_premise(omega_run.data)
    measured = {"survivor": [], "other": []}
    for run in (main_run, omega_run, direction_run, flb_run, gap_run):
        report, ids = run.report, [rule[0] for rule in run.rules]
        names = list(report.stages)
        order = [name for name in names if name in ("log_loss", "max_stat", "omega", "roi")]
        assert order == ["log_loss", "max_stat", "omega", "roi"], names
        log_loss, max_stat = report.stages["log_loss"], report.stages["max_stat"]
        omega, roi = report.stages["omega"], report.stages["roi"]
        for stage, entries in (("log_loss", log_loss), ("omega", omega), ("roi", roi)):
            assert set(entries) == set(ids), (run.spec.family_id, stage, sorted(entries))
        passed_both = {
            r for r in ids if log_loss[r]["passed"] is True and r in max_stat["superior"]
        }
        assert set(report.survivors) == passed_both
        for rule in run.rules:
            rid, rows = rule[0], bets_of(run.data, rule)
            assert len(rows) > 30, rid  # premise: every rule has bets to measure
            ll = log_loss[rid]
            assert all(finite(ll[key]) for key in ("delta_ll", "lo", "hi")), (rid, ll)
            assert ll["lo"] < ll["hi"], (rid, ll)
            assert rid in max_stat["tested"], rid
            entry = omega[rid]
            assert entry["n_bets"] == len(rows), (rid, entry["n_bets"], len(rows))
            assert finite(entry["omega_selected"]) and entry["omega_selected"] > 0, (rid, entry)
            p = entry["p_beats_takeout"]
            assert finite(p) and 0.0 <= p <= 1.0, (rid, entry)
            assert report.p_beats_takeout[rid] == pytest.approx(p, abs=1e-12), rid
            er = entry["expected_return"]
            assert all(finite(er[key]) for key in ("mean", "low", "high", "level")), (rid, er)
            assert er["low"] <= er["mean"] <= er["high"] and 0.0 < er["level"] < 1.0, (rid, er)
            low, high, level = entry[PREDICTIVE]
            assert finite(low) and finite(high) and low < high and 0.0 < level < 1.0, (rid, entry)
            assert roi[rid]["n_bets"] == len(rows), rid
            assert roi[rid]["roi"] == pytest.approx(realised_roi(rows), rel=1e-9, abs=1e-12), rid

            metrics = last_metrics(run, rid)
            assert all(len(metrics.get(name, [])) == 1 for name in RECORD_METRICS), (
                rid,
                sorted(metrics),
            )
            (gain,) = metrics[LOG_LOSS_METRIC]
            assert (gain.value, gain.lo, gain.hi) == pytest.approx(
                (ll["delta_ll"], ll["lo"], ll["hi"]), rel=1e-9
            ), rid
            superior = float(rid in max_stat["superior"])
            assert metrics["stepm_superior"][0].value == superior, rid
            assert metrics["omega_selected"][0].value == pytest.approx(entry["omega_selected"])
            (expected,) = metrics["expected_return"]
            assert (expected.value, expected.lo, expected.hi) == pytest.approx(
                (er["mean"], er["low"], er["high"]), rel=1e-9
            ), rid
            assert metrics["p_beats_takeout"][0].value == pytest.approx(p, abs=1e-12), rid
            assert metrics["roi_win_dividend"][0].value == pytest.approx(roi[rid]["roi"]), rid
            measured["survivor" if rid in report.survivors else "other"].append(rid)
    assert list(main_run.report.survivors) == [MAIN_REAL]
    assert list(omega_run.report.survivors) == [OMEGA_REAL]
    assert set(direction_run.report.survivors) == DIRECTION_SURVIVORS
    assert list(flb_run.report.survivors) == []
    assert len(measured["survivor"]) >= 4 and len(measured["other"]) >= 10, measured


def _stepm_finding_nothing(*args, **kwargs):
    """#75's StepM, except that it finds no rule superior."""
    return dataclasses.replace(stepm(*args, **kwargs), superior=[])


def _bootstrap_of_the_reversed_difference(diff, clusters, **kwargs):
    """#69's bootstrap of the negated difference: every claim's verdict flips, whichever sign
    the study gives the difference."""
    return paired_bootstrap(-diff, clusters, **kwargs)


def test_a_forced_verdict_changes_no_measurement(tmp_path, main_runners, main_run):
    """No verdict gates a measurement. With StepM finding nothing, segment A still passes the
    log-loss stage; with the log-loss bootstrap's verdict reversed, it fails it. Either way the
    survivors diagnostic empties, yet each rule's omega, expected return, P(omega_s > 1/(1 - t)),
    predictive interval and realised ROI are exactly those of the unforced run, its record
    carries them, and its rung stays S0. The stages' tools are #75's stepm and #69's
    paired_bootstrap (ticket outline), replaced here to force their verdicts."""
    check_main_premise(main_runners)
    baseline = main_run.report
    assert list(baseline.survivors) == [MAIN_REAL]
    ids = [rule[0] for rule in MAIN_RULES[:2]]
    gates = [
        ("max_stat", stepm, _stepm_finding_nothing),
        ("log_loss", paired_bootstrap, _bootstrap_of_the_reversed_difference),
    ]
    for gate, original, fake in gates:
        with pytest.MonkeyPatch.context() as patch:
            _replace_everywhere(patch, original, fake)
            run = run_frozen(
                tmp_path / gate, f"test-88-{gate}-forced", MAIN_RULES[:2], main_runners
            )
        report = run.report
        assert list(report.survivors) == [], gate
        if gate == "max_stat":
            assert report.stages["max_stat"]["superior"] == []
            assert report.stages["log_loss"][MAIN_REAL]["passed"] is True
        else:
            assert report.stages["log_loss"][MAIN_REAL]["passed"] is False
        for rid in ids:
            for stage in ("omega", "roi"):
                assert rid in baseline.stages[stage] and rid in report.stages[stage], (gate, rid)
            assert omega_numbers(report.stages["omega"][rid]) == pytest.approx(
                omega_numbers(baseline.stages["omega"][rid]), rel=1e-9
            ), (gate, rid)
            forced, unforced = report.stages["roi"][rid], baseline.stages["roi"][rid]
            assert forced["n_bets"] == unforced["n_bets"], (gate, rid)
            assert forced["roi"] == pytest.approx(unforced["roi"], rel=1e-12), (gate, rid)
            assert report.rungs[rid] == "S0", (gate, rid)
            metrics = last_metrics(run, rid)
            assert all(name in metrics for name in RETURN_METRICS), (gate, rid, sorted(metrics))


MAX_STAT_RULES = MAIN_RULES[:2]  # segments A and B


def meetingless_race(data: pd.DataFrame) -> str:
    """A scored race of segment C, outside every MAX_STAT_RULES segment: the race left without a
    meeting_id below."""
    outside = data[(data["seg"] == "C") & (data["race_date"] >= FIRST_SCORED)]
    return str(outside["race_id"].iloc[0])


def _stepm_raising(*args, **kwargs):
    raise RuntimeError("StepM failed (forced by the test)")


@pytest.fixture(scope="module")
def max_stat_failures(tmp_path_factory, main_runners) -> dict:
    """Each way the max-statistic stage fails, and the run it gave (or what run_study raised): a
    scored race outside every rule's segment with no meeting_id (SPA resamples every scored race
    of the benchmark by meeting), and #75's StepM raising."""
    race = meetingless_race(main_runners)
    no_meeting = main_runners.copy()
    no_meeting.loc[no_meeting["race_id"] == race, "meeting_id"] = None
    cases = {
        "a scored race outside every segment has no meeting_id": (no_meeting, None),
        "StepM raises": (main_runners, (stepm, _stepm_raising)),
    }
    out: dict = {}
    for k, (label, (data, patched)) in enumerate(cases.items()):
        root = tmp_path_factory.mktemp(f"max-stat-{k}")
        try:
            with pytest.MonkeyPatch.context() as patch:
                if patched is not None:
                    _replace_everywhere(patch, *patched)
                out[label] = run_frozen(root, f"test-88-max-stat-{k}", MAX_STAT_RULES, data)
        except Exception as exc:  # reported by the tests, not raised in setup
            out[label] = exc
    return out


def test_a_max_statistic_failure_stops_no_measurement(main_runners, main_run, max_stat_failures):
    """No stage gates another, the max-statistic stage included. When SPA cannot run (a scored
    race outside both rules' segments has no meeting_id) or StepM raises, the study still
    reports: `max_stat` carries the reason, finds no rule superior and so leaves no survivor, and
    every rule keeps its log-loss result, omega, expected return, P(omega_s > 1/(1 - t)),
    predictive interval and realised ROI exactly as in the main run, its last registry record
    carrying them, and its rung S0. No StepM verdict is recorded for a test that did not run."""
    race = meetingless_race(main_runners)
    rows = main_runners["race_id"] == race
    # Premise: the race is scored, in segment C (no rule here bets it), and only its meeting_id
    # is taken away.
    assert scored(main_runners)["race_id"].eq(race).any()
    assert (main_runners.loc[rows, "seg"] == "C").all()
    assert not any(rule[2]["seg"] == "C" for rule in MAX_STAT_RULES)
    stopped = {
        label: repr(run) for label, run in max_stat_failures.items() if not isinstance(run, Run)
    }
    assert stopped == {}, f"a max-statistic failure stopped the study: {stopped}"

    baseline = main_run.report
    for label, run in max_stat_failures.items():
        report = run.report
        max_stat = report.stages["max_stat"]
        assert explained(max_stat), (label, max_stat)
        assert list(max_stat.get("superior", [])) == [], (label, max_stat)
        assert list(report.survivors) == [], (label, report.survivors)
        for rule in MAX_STAT_RULES:
            rid = rule[0]
            ll, unforced = report.stages["log_loss"][rid], baseline.stages["log_loss"][rid]
            assert (ll["delta_ll"], ll["lo"], ll["hi"]) == pytest.approx(
                (unforced["delta_ll"], unforced["lo"], unforced["hi"]), rel=1e-9
            ), (label, rid)
            assert omega_numbers(report.stages["omega"][rid]) == pytest.approx(
                omega_numbers(baseline.stages["omega"][rid]), rel=1e-9
            ), (label, rid)
            roi, unforced = report.stages["roi"][rid], baseline.stages["roi"][rid]
            assert roi["n_bets"] == unforced["n_bets"], (label, rid)
            assert roi["roi"] == pytest.approx(unforced["roi"], rel=1e-12), (label, rid)
            assert report.rungs[rid] == "S0", (label, rid)
            metrics = last_metrics(run, rid)
            assert {LOG_LOSS_METRIC, *RETURN_METRICS} <= set(metrics), (label, rid, sorted(metrics))
            assert "stepm_superior" not in metrics, (label, rid, metrics["stepm_superior"])


REASON_NO_BET = "P6-seg-c-none"
REASON_NO_CLAIM = "P4-j2-beta"
REASON_RULES = [
    MAIN_RULES[0],
    # odds_t is never below 1.1: the claim on segment C is scored, but no bet is selected.
    (REASON_NO_BET, "beta", {"seg": "C"}, ["odds_t"], {"odds_t": [None, 1.0]}, "above"),
    # A 'beta' claim needs a race-level segment and J2 varies within races: the claim cannot be
    # scored, but its selected bets can be measured.
    (REASON_NO_CLAIM, "beta", {"jockey": "J2"}, ["odds_t"], {"odds_t": [3.0, None]}, "above"),
]


@pytest.fixture(scope="module")
def reason_run(tmp_path_factory, main_runners) -> Run:
    root = tmp_path_factory.mktemp("reasons")
    return run_frozen(root, "test-88-reasons", REASON_RULES, main_runners)


def test_a_measurement_that_cannot_be_made_carries_its_reason(reason_run):
    """A rule that selects no bet keeps its log-loss result, and its omega and ROI entries carry
    the reason they could not be measured (no number, no metric). A rule whose claim cannot be
    scored carries the log-loss reason and is not in StepM's family, yet its omega, expected
    return, P(omega_s > 1/(1 - t)) and ROI are measured and recorded: nothing gates them."""
    data, report = reason_run.data, reason_run.report
    ids = [rule[0] for rule in REASON_RULES]
    no_bet, no_claim = (next(r for r in REASON_RULES if r[0] == rid) for rid in ids[1:])
    assert bets_of(data, no_bet).empty  # premise
    races_mixed = data.groupby("race_id")["jockey"].apply(
        lambda j: (j == "J2").any() > (j == "J2").all()
    )
    assert races_mixed.any()  # premise: J2 is a runner-level segment
    rows = bets_of(data, no_claim)
    assert len(rows) > 50  # premise
    for stage in ("log_loss", "omega", "roi"):
        assert set(report.stages[stage]) == set(ids), (stage, sorted(report.stages[stage]))

    ll = report.stages["log_loss"][REASON_NO_BET]
    assert finite(ll.get("delta_ll")), ll
    entry, roi = report.stages["omega"][REASON_NO_BET], report.stages["roi"][REASON_NO_BET]
    assert explained(entry) and explained(roi), (entry, roi)
    assert entry["n_bets"] == 0 and roi["n_bets"] == 0, (entry, roi)
    assert not finite(entry.get("omega_selected")) and not finite(roi.get("roi")), (entry, roi)
    assert not finite(report.p_beats_takeout[REASON_NO_BET])
    metrics = last_metrics(reason_run, REASON_NO_BET)
    assert LOG_LOSS_METRIC in metrics, sorted(metrics)
    assert not set(RETURN_METRICS) & set(metrics), sorted(metrics)

    assert explained(report.stages["log_loss"][REASON_NO_CLAIM])
    assert REASON_NO_CLAIM not in report.stages["max_stat"]["tested"]
    entry, roi = report.stages["omega"][REASON_NO_CLAIM], report.stages["roi"][REASON_NO_CLAIM]
    assert entry["n_bets"] == roi["n_bets"] == len(rows), (entry, roi)
    assert finite(entry["omega_selected"]) and finite(entry["expected_return"]["mean"]), entry
    assert 0.0 <= report.p_beats_takeout[REASON_NO_CLAIM] <= 1.0
    assert roi["roi"] == pytest.approx(realised_roi(rows), rel=1e-9, abs=1e-12)
    metrics = last_metrics(reason_run, REASON_NO_CLAIM)
    assert set(RETURN_METRICS) <= set(metrics), sorted(metrics)
    assert metrics["roi_win_dividend"][0].value == pytest.approx(realised_roi(rows), rel=1e-9)
    assert report.rungs == dict.fromkeys(ids, "S0")


def reread(run: Run) -> dict[str, list[RunRecord]]:
    """The family's records by variant, in append order, read back from the registry's file by a
    Registry opened afresh on it."""
    registry = Registry(run.registry.path, holdout_log=run.registry.holdout_log)
    registry.verify()
    out: dict[str, list[RunRecord]] = {}
    for record in registry.records(run.spec.family_id):
        out.setdefault(record.variant, []).append(record)
    return out


def notes_of(record: RunRecord) -> dict[str, str]:
    """The run metadata a record carries outside its variant identity (`RunRecord.notes`)."""
    return dict(getattr(record, "notes", None) or {})


def identity_of(record: RunRecord) -> tuple:
    """What `Registry.family_size` counts a variant by (core.eval.registry)."""
    fields = ("hypothesis_id", "variant", "fold_scheme", "sources", "components", "params")
    return (*(getattr(record, name) for name in fields), record.data_builds)


def unpaid_winner(data: pd.DataFrame, rule: tuple) -> pd.DataFrame:
    """`data` with no win_dividend on the first scored winner the rule selects."""
    rows = bets_of(data, rule)
    out = data.copy()
    out.loc[rows.index[rows["won"] == 1][0], "win_dividend"] = np.nan
    return out


def test_a_measurement_failure_is_persisted_in_the_registry(
    tmp_path, monkeypatch, main_runners, reason_run, max_stat_failures
):
    """A reason the report gives is kept in the registry too, outside the variant identity. Read
    back from the file, each rule's last record carries in `notes`, by stage ('log_loss',
    'max_stat', 'omega', 'roi'), the reason each of its measurements could not be made, exactly
    as the report's entry states it, and nothing for a measurement made: a selected winner with
    no dividend (the ROI), a rule that selects no bet (omega and the ROI), a claim that cannot be
    scored (log loss), and a max-statistic stage that failed (every rule it tested). The
    family's size and each rule's variant identity are those of its first record."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    rule = MAX_STAT_RULES[1]
    unpaid = run_frozen(
        tmp_path / "unpaid",
        "test-88-unpaid-winner",
        MAX_STAT_RULES,
        unpaid_winner(main_runners, rule),
    )
    roi = unpaid.report.stages["roi"][rule[0]]
    assert explained(roi) and not finite(roi.get("roi")), roi  # premise: the report explains it

    def expected_notes(run: Run, rid: str) -> dict[str, str]:
        """The reasons the report gives for the rule's measurements that could not be made."""
        stages = run.report.stages
        out = {
            stage: stages[stage][rid]["error"]
            for stage in ("log_loss", "omega", "roi")
            if explained(stages[stage][rid])
        }
        if explained(stages["max_stat"]) and rid in stages["max_stat"]["tested"]:
            out["max_stat"] = stages["max_stat"]["error"]
        return out

    checked = []
    for run in (unpaid, reason_run):
        records = reread(run)
        for each in run.rules:
            expected = expected_notes(run, each[0])
            assert notes_of(records[each[0]][-1]) == expected, (run.spec.family_id, each[0])
            checked += list(expected)
    assert {"log_loss", "omega", "roi"} <= set(checked), checked  # premise: each stage failed once

    completed = [run for run in max_stat_failures.values() if isinstance(run, Run)]
    assert len(completed) == len(max_stat_failures), (
        "a max-statistic failure stopped the study (test_a_max_statistic_failure_stops_no_"
        f"measurement): {max_stat_failures}"
    )
    for run in completed:
        records, max_stat = reread(run), run.report.stages["max_stat"]
        assert explained(max_stat) and max_stat["tested"], max_stat  # premise
        for rid in max_stat["tested"]:
            expected = expected_notes(run, rid)
            assert expected["max_stat"] == max_stat["error"]
            assert notes_of(records[rid][-1]) == expected, (run.spec.family_id, rid)

    for run in (unpaid, reason_run, *completed):
        family_id, records = run.spec.family_id, reread(run)
        assert run.report.family_size == run.registry.family_size(family_id) == len(run.rules)
        for each in run.rules:
            first, last = records[each[0]][0], records[each[0]][-1]
            assert identity_of(last) == identity_of(first), (family_id, each[0])


def test_a_retry_that_succeeds_again_is_each_rules_latest_record(
    tmp_path, monkeypatch, main_runners
):
    """Round 4 of PR #345: a transient failure is not the last word. The same frozen family runs
    three times on the same data, registry and seed: StepM runs, then raises, then runs again and
    reproduces the first run exactly. After the third run each rule's latest registry record is a
    success again: it carries `stepm_superior` (StepM's verdict, as the third report gives it)
    and every metric of the first run's last record, and no 'max_stat' failure note; its other
    notes are the first success's. The failure stays in the append-only history, and the retry is
    another run, never another variant: the family size and each rule's variant identity are
    those of the first run."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(data_dir))
    registry = new_registry(tmp_path)
    spec = make_family("test-88-retry", MAX_STAT_RULES)
    freeze(registry, spec)
    ids = [rule[0] for rule in MAX_STAT_RULES]

    def records_of(rid: str) -> list[RunRecord]:
        return [r for r in registry.records(spec.family_id) if r.variant == rid]

    def attempt(stepm_fails: bool):
        holdout = TrippedHoldout()
        with pytest.MonkeyPatch.context() as patch:
            if stepm_fails:
                _replace_everywhere(patch, stepm, _stepm_raising)
            report = run_study(
                spec, main_runners.copy(), registry=registry, holdout=holdout, seed=STUDY_SEED
            )
        assert holdout.calls == []
        return report, registry.family_size(spec.family_id), {r: records_of(r)[-1] for r in ids}

    def metric_names(record: RunRecord) -> set[str]:
        return {m.metric for m in record.metrics}

    first, first_size, succeeded = attempt(stepm_fails=False)
    # Premise: StepM ran and tested both rules, and their last records carry its verdict.
    assert not explained(first.stages["max_stat"]), first.stages["max_stat"]
    assert set(first.stages["max_stat"]["tested"]) == set(ids)
    for rid in ids:
        assert set(RECORD_METRICS) <= metric_names(succeeded[rid]), (rid, succeeded[rid])
        assert "max_stat" not in notes_of(succeeded[rid]), (rid, succeeded[rid].notes)

    failed, failed_size, after_failure = attempt(stepm_fails=True)
    # Premise: the failure is recorded, and is each rule's latest record for now.
    assert explained(failed.stages["max_stat"]), failed.stages["max_stat"]
    for rid in ids:
        assert "stepm_superior" not in metric_names(after_failure[rid]), rid
        assert notes_of(after_failure[rid]).get("max_stat") == failed.stages["max_stat"]["error"]

    third, third_size, latest = attempt(stepm_fails=False)
    # Premise: the third run reproduces the first.
    assert not explained(third.stages["max_stat"]), third.stages["max_stat"]
    assert third.stages["max_stat"]["superior"] == first.stages["max_stat"]["superior"]

    for rid in ids:
        last = latest[rid]
        superior = [m.value for m in last.metrics if m.metric == "stepm_superior"]
        assert superior == [float(rid in third.stages["max_stat"]["superior"])], (rid, last)
        assert set(succeeded[rid].metrics) <= set(last.metrics), (rid, last.metrics)
        notes = notes_of(last)
        assert "max_stat" not in notes, (rid, notes)
        for stage in ("log_loss", "omega", "roi"):
            assert notes.get(stage) == notes_of(succeeded[rid]).get(stage), (rid, stage, notes)
        assert last.status == succeeded[rid].status, rid
        assert identity_of(last) == identity_of(succeeded[rid]), rid
        history = records_of(rid)
        assert any("max_stat" in notes_of(r) for r in history), rid  # the failure is kept
    assert first.family_size == failed.family_size == third.family_size == len(ids)
    assert first_size == failed_size == third_size == len(ids)
    registry.verify()


PREDICTIVE_LEVEL = f"{PREDICTIVE}_level"


def test_the_last_record_carries_the_predictive_interval_apart(main_run, omega_run, direction_run):
    """Read back from the registry's file, each rule's last record carries the predictive
    interval of the ROI of its next 100 bets as the metric `predictive_roi_100` (lo and hi its
    bounds, its value a point between them) and that interval's level as
    `predictive_roi_100_level`, beside and apart from the expected return's credible interval
    (`expected_return`), which is narrower and stays as it was."""
    checked = 0
    for run in (main_run, omega_run, direction_run):
        records = reread(run)
        for rule in run.rules:
            rid = rule[0]
            entry = run.report.stages["omega"][rid]
            (low, high, level), er = entry[PREDICTIVE], entry["expected_return"]
            assert finite(low) and finite(high) and low < high, (rid, entry)  # premise
            metrics: dict[str, list] = {}
            for metric in records[rid][-1].metrics:
                metrics.setdefault(metric.metric, []).append(metric)
            assert len(metrics.get(PREDICTIVE, [])) == 1, (rid, sorted(metrics))
            assert len(metrics.get(PREDICTIVE_LEVEL, [])) == 1, (rid, sorted(metrics))
            (predictive,), (coverage,) = metrics[PREDICTIVE], metrics[PREDICTIVE_LEVEL]
            assert (predictive.lo, predictive.hi) == pytest.approx((low, high), rel=1e-12), rid
            assert predictive.lo <= predictive.value <= predictive.hi, (rid, predictive)
            assert coverage.value == pytest.approx(level, rel=1e-12), (rid, coverage)
            (expected,) = metrics["expected_return"]
            assert (expected.value, expected.lo, expected.hi) == pytest.approx(
                (er["mean"], er["low"], er["high"]), rel=1e-9
            ), rid
            assert predictive.hi - predictive.lo > expected.hi - expected.lo, (rid, predictive)
            checked += 1
    assert checked == len(MAIN_RULES) + len(OMEGA_RULES) + len(DIRECTION_RULES)


def test_roi_reads_the_winners_dividends_when_losers_carry_none(gap_run):
    """Losers' dividends are missing (NaN) in the store; every rule's ROI is still the selected
    winners' dividends over its bets, and its last registry record carries it as
    `roi_win_dividend`."""
    data = gap_run.data
    assert data.loc[data["won"] == 0, "win_dividend"].isna().all()
    assert data.loc[data["won"] == 1, "win_dividend"].notna().all()
    report = gap_run.report
    assert MAIN_REAL in report.survivors  # premise: segment A survives here as in the main run
    assert set(report.stages["roi"]) == {rule[0] for rule in GAP_RULES}
    for rule in GAP_RULES:
        rid, rows = rule[0], bets_of(data, rule)
        expected = float(rows.loc[rows["won"] == 1, "win_dividend"].sum() / len(rows) - 1.0)
        roi = report.stages["roi"][rid]
        assert roi["n_bets"] == len(rows), rid
        assert math.isfinite(roi["roi"]), rid
        assert roi["roi"] == pytest.approx(expected, rel=1e-9, abs=1e-12), rid
        logged = last_metrics(gap_run, rid).get("roi_win_dividend")
        assert logged, f"{rid}: no roi_win_dividend metric in its last registry record"
        assert logged[0].value == pytest.approx(expected, rel=1e-9, abs=1e-12), rid


# ---------------------------------------------------------------------------------------------
# The log-loss stage: against D4's calibrated market, one-sided in the rule's direction
# ---------------------------------------------------------------------------------------------


def test_a_market_wide_favourite_longshot_bias_is_no_segment_claim(flb_run):
    """Every race is priced q^1.3: favourites win more often than the normalised price says in
    every segment alike. The calibrated market carries that bias, so no segment departs from it
    and no rule passes the log-loss stage, whichever way it bets; every rule is still measured."""
    check_flb_premise(flb_run.data)
    report = flb_run.report
    ids = [rule[0] for rule in FLB_RULES]
    assert_claims_scored(flb_run)
    assert {rid: report.stages["log_loss"][rid]["passed"] for rid in ids} == dict.fromkeys(
        ids, False
    )
    assert list(report.survivors) == []
    assert set(report.stages["roi"]) == set(ids)
    assert report.rungs == dict.fromkeys(ids, "S0")


def test_a_segment_claim_passes_only_in_the_direction_its_rule_bets(direction_run):
    """On top of the market-wide bias, segment A's favourites and segment B's longshots win far
    more often than priced. Each segment is bet both ways: only the rule betting the side that
    departs passes and survives; the rule betting the other side of the same departure, and the
    favourite rule of segment C (the market-wide bias only), do not. Every rule is measured and
    sits at S0."""
    check_direction_premise(direction_run.data)
    report = direction_run.report
    ids = [rule[0] for rule in DIRECTION_RULES]
    assert_claims_scored(direction_run)
    passed = {rid: report.stages["log_loss"][rid]["passed"] for rid in ids}
    assert passed == {rid: rid in DIRECTION_SURVIVORS for rid in ids}
    assert set(report.survivors) == DIRECTION_SURVIVORS
    assert set(report.stages["roi"]) == set(ids)
    assert report.rungs == dict.fromkeys(ids, "S0")


# ---------------------------------------------------------------------------------------------
# DoD: the selection-input contract (each input's source and its availability by t)
# ---------------------------------------------------------------------------------------------


def test_a_final_odds_selection_variable_is_refused_at_the_roi_stage(
    tmp_path, monkeypatch, main_runners
):
    """The real rule (segment A) is re-pointed at each final-price column in turn, alone or
    beside a board-at-t column. Each is declared as if it came from the board at t, stamped
    before t, so only the column itself refuses it; every run is refused before any record."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    check_main_premise(main_runners)
    data = declare(main_runners, **dict.fromkeys(LOOK_AHEAD, BOARD))
    cases = [[col] for col in LOOK_AHEAD] + [["odds_t", "final_rank"]]
    ranges = {
        "odds_t": [1.5, 4.0],
        "odds_final": [1.5, 4.0],
        "final_odds_band": [1, 2],
        "final_rank": [1, 3],
        "final_entropy": [0.0, 0.8],
    }
    for k, cols in enumerate(cases):
        real = ("P4-seg-a", "beta", {"seg": "A"}, cols, {c: ranges[c] for c in cols}, "above")
        rules = [real, MAIN_RULES[1]]
        registry = new_registry(tmp_path / f"case{k}")
        spec = make_family(f"test-88-look-ahead-{k}", rules)
        freeze(registry, spec)
        holdout = TrippedHoldout()
        with pytest.raises(LookAheadSelection):
            run_study(spec, data.copy(), registry=registry, holdout=holdout, seed=1)
        assert holdout.calls == [], cols
        assert registry.records(spec.family_id) == [], cols


def test_the_stores_final_price_columns_are_look_ahead(tmp_path, monkeypatch, main_runners):
    """The JRA store names the final Win odds `win_odds` and the final favourite rank
    `win_fav_rank`; a rule selecting on either is refused like any `final_*` column, even
    declared from the runners card and stamped before t."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    data = declare(
        main_runners.assign(
            win_odds=main_runners["odds_final"], win_fav_rank=main_runners["final_rank"]
        ),
        win_odds=RUNNERS,
        win_fav_rank=RUNNERS,
    )
    for k, (col, bounds) in enumerate([("win_odds", [1.5, 4.0]), ("win_fav_rank", [1, 1])]):
        real = ("P4-seg-a", "beta", {"seg": "A"}, [col], {col: bounds}, "above")
        registry = new_registry(tmp_path / f"store{k}")
        spec = make_family(f"test-88-store-look-ahead-{k}", [real, MAIN_RULES[1]])
        freeze(registry, spec)
        holdout = TrippedHoldout()
        with pytest.raises(LookAheadSelection):
            run_study(spec, data.copy(), registry=registry, holdout=holdout, seed=1)
        assert holdout.calls == [], col


def _accepted(tmp_path: Path, name: str, cases: list[tuple], data: pd.DataFrame) -> list[str]:
    """The cases (a label, a rule) whose one-rule family run_study does not refuse with
    LookAheadSelection before scoring; none writes a record or opens the holdout."""
    accepted = []
    for k, (label, rule) in enumerate(cases):
        outcome = outcome_before_scoring(tmp_path, f"{name}-{k}", [rule], data)
        if outcome != "look-ahead":
            accepted.append(f"{label}: {outcome}")
    return accepted


def test_every_label_of_the_availability_config_is_refused_as_a_selection_column(
    tmp_path, monkeypatch, main_runners
):
    """A view at t never returns a label column (Japan's availability config: each table's
    `label_columns`, and every name under a `label_prefixes` entry), and holds a column gated at
    the post null until its gate opens (the finish since #305: post_utc + 30 min, after any
    decision instant), so no bet is selected on either. Each label column and each column gated
    after T-5, holding the finishing position and declared from the runners card with a stamp
    before t, is the only selection column of the real rule (segment A): [1, 1] would bet the
    winners only and report their return. Every run is refused before scoring."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    check_main_premise(main_runners)
    labels = availability_labels()
    # Premise: the config's results are among them, beyond the final price.
    assert {"placing_code", "last_3f_s", "result_published_utc"} <= set(labels)
    assert {"dividend", "is_final_capture"} <= set(labels)  # the board's payout columns
    assert any(name.endswith("win_odds") and name != "win_odds" for name in labels)  # a prefix
    assert all(is_availability_label(name) for name in labels)
    # Premise: the finish is gated at the post with a positive offset (#305), so it opens after
    # the decision instant of any T-n.
    gates = post_gates()
    assert all(col in gates and gates[col] > timedelta(0) for col in FINISH), gates
    gated = gated_after(DECISION_MINUTES)
    assert set(FINISH) <= set(gated), gated
    columns = list(dict.fromkeys([*labels, *gated]))
    finish = finish_positions(main_runners)
    data = declare(
        main_runners.assign(**dict.fromkeys(columns, finish)), **dict.fromkeys(columns, RUNNERS)
    )
    assert (data.loc[data["won"] == 1, "finish_pos"] == 1).all()
    assert (data.loc[data["won"] == 0, "finish_pos"] >= 2).all()
    assert (data["finish_pos" + AVAILABLE] < decision_utc(data)).all()

    cases = [
        (col, ("P4-seg-a", "beta", {"seg": "A"}, [col], {col: [1, 1]}, "above")) for col in columns
    ]
    accepted = _accepted(tmp_path, "label-selection", cases, data)
    assert accepted == [], f"look-ahead columns accepted as a selection column: {accepted}"


def test_a_segment_on_a_look_ahead_column_is_refused(tmp_path, monkeypatch, main_runners):
    """The segment half of selection hygiene: a rule whose segment is read from a result label,
    a column gated after the decision instant or a final-price column is refused like a
    selection column, declared or not. Runner-level segments that are the winners (`finish_pos`
    1, gated at post_utc + 30 min since #305) or the final favourites (`final_rank` 1), and
    segment A's races (the real claim) under the name of a race result, a board payout, a
    prefixed label or a finish column."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    renamed = ["result_published_utc", "dividend", "tyb_segment"]
    assert all(is_availability_label(name) for name in renamed)
    finish_named = ["dead_heat", "finish_time_s"]
    assert {"finish_pos", *finish_named} <= set(gated_after(DECISION_MINUTES))
    data = declare(
        main_runners.assign(
            finish_pos=finish_positions(main_runners),
            **{col: main_runners["seg"] for col in [*renamed, *finish_named]},
        ),
        finish_pos=RUNNERS,
        final_rank=BOARD,
        **dict.fromkeys(renamed, RACES),
        **dict.fromkeys(finish_named, RUNNERS),
    )
    anywhere = {"odds_t": [None, None]}
    cases = [
        ("finish_pos", ("P4-winners", "omega", {"finish_pos": 1}, ["odds_t"], anywhere, "above")),
        ("final_rank", ("P4-final-fav", "omega", {"final_rank": 1}, ["odds_t"], anywhere, "above")),
    ]
    cases += [
        (col, ("P4-seg-a", "beta", {col: "A"}, ["odds_t"], {"odds_t": [1.5, 4.0]}, "above"))
        for col in [*renamed, *finish_named]
    ]
    accepted = _accepted(tmp_path, "label-segment", cases, data)
    assert accepted == [], f"look-ahead columns accepted as a segment: {accepted}"


def gate_frame(data: pd.DataFrame, columns: list[str], minutes: int) -> pd.DataFrame:
    """`data` with each of `columns` holding the finishing position (so [1, 1] bets the winners)
    and declared from the runners or the races card, alternately, stamped exactly at its race's
    decision instant under T-`minutes`: a stamp the contract admits, so only a name can refuse
    the column."""
    out = data.assign(**dict.fromkeys(columns, finish_positions(data)))
    for k, col in enumerate(columns):
        out[col + SOURCE] = (RUNNERS, RACES)[k % 2]
        out[col + AVAILABLE] = decision_utc(out, minutes)
    return out


def gate_outcomes(root: Path, name: str, data: pd.DataFrame, columns, minutes: int) -> dict:
    """run_study's outcome (outcome_before_scoring) for each column as the only selection column
    of the real rule ([1, 1], segment A), in a family decided at T-`minutes`, by (column, n)."""
    frame = gate_frame(data, list(columns), minutes)
    return {
        (col, minutes): outcome_before_scoring(
            root,
            f"{name}-{minutes}-{k}",
            [("P4-seg-a", "beta", {"seg": "A"}, [col], {col: [1, 1]}, "above")],
            frame,
            minutes=minutes,
        )
        for k, col in enumerate(columns)
    }


def test_a_column_gated_after_the_post_is_refused_by_name(tmp_path, monkeypatch, main_runners):
    """The ruling on #88 after #305: a column Japan's availability config gates at post_utc +
    offset, with offset > -n for the family's decision time T-n (a view at t still holds it
    null), is look-ahead by name, like a label. Each post-gated column of the config, declared
    from an admitted source and stamped exactly at t, is the only selection column of the real
    rule at T-0, T-5, T-60 and T-61: refused with LookAheadSelection before anything is scored
    or written exactly when its gate opens after t. So the finish (post_utc + 30 min) is refused
    at every T-n, and the horse weights and the same-day going (post_utc - 60 min) are accepted
    up to T-60 (the gate opens at t) and refused at T-61."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    gates = post_gates()
    # Premise: the finish opens after the post and the weights and going an hour before it;
    # none is a label, so only its gate can refuse a column.
    assert all(col in gates and gates[col] > timedelta(0) for col in FINISH), gates
    assert gates["horse_wt_kg"] == gates["going"] == timedelta(minutes=-60), gates
    assert not [col for col in gates if is_availability_label(col)], gates
    decision_times = (0, DECISION_MINUTES, 60, 61)
    expected = {
        (col, n): "look-ahead" if offset > timedelta(minutes=-n) else "scored"
        for col, offset in gates.items()
        for n in decision_times
    }
    assert expected["horse_wt_kg", 60] == "scored" and expected["horse_wt_kg", 61] == "look-ahead"
    assert all(expected[col, n] == "look-ahead" for col in FINISH for n in decision_times)

    outcomes: dict = {}
    for n in decision_times:
        outcomes |= gate_outcomes(tmp_path, "post-gate", main_runners, sorted(gates), n)
    wrong = {key: (outcomes[key], want) for key, want in expected.items() if outcomes[key] != want}
    assert wrong == {}, f"(column, n): (outcome, expected) {wrong}"


def test_only_a_gate_at_the_post_that_opens_after_t_is_refused(tmp_path, monkeypatch, main_runners):
    """The gates are read from load_availability("japan") at each run, as the labels are, and
    only a gate anchored at post_utc counts (the ruling on #88). With three hypothetical gates
    added to the config the study reads, a T-0 family is refused a column gated at the post + 1
    minute (on runners, read through races), and accepts a column gated at the post + 0 (on
    races: open at t) and one gated at another anchor + 30 minutes (left to the declared-stamp
    contract), each declared from an admitted source and stamped at t."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    real_loader = load_availability
    via = Via(table="races", on=("race_id",))
    added = {
        "races": (
            Gate(("hyp_at_post",), POST_UTC, timedelta(0)),
            Gate(("hyp_other_anchor",), "entries_closed_utc", timedelta(minutes=30)),
        ),
        "runners": (Gate(("hyp_after_post",), POST_UTC, timedelta(minutes=1), via),),
    }

    def with_hypothetical_gates(*args, **kwargs):
        return {
            table: dataclasses.replace(rule, gates=(*rule.gates, *added.get(table, ())))
            for table, rule in real_loader(*args, **kwargs).items()
        }

    # Premise: the real config gates none of them; the patched one gates each as described.
    columns = ["hyp_after_post", "hyp_at_post", "hyp_other_anchor"]
    real, patched = real_loader("japan"), with_hypothetical_gates("japan")
    assert not set(columns) & {col for rule in real.values() for col in rule.gated}
    assert post_gates(patched)["hyp_after_post"] == timedelta(minutes=1)
    assert post_gates(patched)["hyp_at_post"] == timedelta(0)
    assert "hyp_other_anchor" in patched["races"].gated
    assert "hyp_other_anchor" not in post_gates(patched)

    with pytest.MonkeyPatch.context() as patch:
        _replace_everywhere(patch, real_loader, with_hypothetical_gates)
        outcomes = gate_outcomes(tmp_path, "hypothetical-gate", main_runners, columns, 0)
    assert outcomes == {
        ("hyp_after_post", 0): "look-ahead",
        ("hyp_at_post", 0): "scored",
        ("hyp_other_anchor", 0): "scored",
    }, outcomes


def payout_join(data: pd.DataFrame) -> pd.DataFrame:
    """`data` with the Win payout table joined on by race and winning runner, keeping the paid
    combination (the winner's number) and its pool: both are present on the winners only, and
    the payout is published after the race (availability.yaml: post_utc + 30 min)."""
    winners = data.loc[data["won"] == 1, ["race_id", "horse_id"]]
    payouts = winners.assign(pool="WIN", combination=winners["horse_id"].astype(float))
    return data.merge(payouts, on=["race_id", "horse_id"], how="left", validate="many_to_one")


def test_a_post_race_table_joined_onto_runners_is_refused(tmp_path, monkeypatch, main_runners):
    """The Win payout's `combination`, joined onto runners and selected with [null, null], bets
    the winners only. However it is presented (undeclared, declared from the payout table at its
    publication or stamped before t, passed off as the board at t but stamped at publication,
    half declared) and from every source that is not the board at t or a pre-race card table, it
    is refused with LookAheadSelection before anything is scored or written; so is a segment read
    from the payout table. The source is checked whatever the column's name: the real rule's own
    `odds_t` or `seg`, declared from one of those sources (one row of it, or all), with a stamp
    before t, is refused the same way."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    joined = payout_join(main_runners)
    select = ("P4-seg-a", "beta", {"seg": "A"}, ["combination"], {"combination": [None, None]})
    select += ("above",)
    winners = selected(joined, select)
    # Premise: the rule bets exactly segment A's winners, so its return would be the winners'.
    assert (
        winners["won"].all()
        and len(winners) == joined.loc[joined["seg"] == "A", "race_id"].nunique()
    )
    assert realised_roi(winners) > 1.0
    published = joined["post_utc"] + pd.Timedelta(minutes=30)
    before_t = card_stamp(joined)
    assert (published > decision_utc(joined)).all() and (before_t < decision_utc(joined)).all()

    def declared(col: str, source: str | None, available: pd.Series | None) -> pd.DataFrame:
        out = joined.copy()
        if source is not None:
            out[col + SOURCE] = source
        if available is not None:
            out[col + AVAILABLE] = available
        return out

    segment = ("P4-winners", "omega", {"pool": "WIN"}, ["odds_t"], {"odds_t": [None, None]})
    segment += ("above",)
    cases = [
        ("undeclared", select, joined),
        ("from dividends, at publication", select, declared("combination", "dividends", published)),
        (
            "from dividends, stamped before t",
            select,
            declared("combination", "dividends", before_t),
        ),
        ("as the board at t, at publication", select, declared("combination", BOARD, published)),
        ("a stamp before t, no source", select, declared("combination", None, before_t)),
        ("the board at t, no stamp", select, declared("combination", BOARD, None)),
        ("a segment from dividends", segment, declared("pool", "dividends", published)),
    ]
    cases += [
        (f"from {source}, stamped before t", select, declared("combination", source, before_t))
        for source in REFUSED_SOURCES[1:]
    ]
    real = MAIN_RULES[0]
    cases += [
        (f"odds_t from {source}", real, declared("odds_t", source, before_t))
        for source in REFUSED_SOURCES
    ]
    one_row = joined.copy()
    one_row.loc[bets_of(joined, real).index[0], "odds_t" + SOURCE] = "dividends"
    cases += [
        ("one selected odds_t from dividends", real, one_row),
        ("seg from dividends", real, declared("seg", "dividends", before_t)),
    ]
    outcomes = {
        label: outcome_before_scoring(tmp_path, f"payout-{k}", [rule], data)
        for k, (label, rule, data) in enumerate(cases)
    }
    assert outcomes == dict.fromkeys(outcomes, "look-ahead"), outcomes


def test_an_input_published_after_the_decision_time_is_refused(
    tmp_path, monkeypatch, main_runners, main_run
):
    """A declared input holds what was known at the race's decision instant t (post_utc - 5 min
    here) or earlier. One selected runner's board-at-t price stamped a second after t, a minute
    before the off, or after the race, or with no stamp; one segment value stamped a second
    after t; segment A's boards read at the off instead of at t: each is refused with
    LookAheadSelection before anything is scored or written. The same frame stamped at or
    before t, some values exactly at t, ran in the main study."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    rule = MAIN_RULES[0]
    bets = bets_of(main_runners, rule)
    t = decision_utc(main_runners)
    stamp = main_runners["odds_t" + AVAILABLE]
    # Premise: every stamp is at or before t, the bets include values stamped exactly at t, and
    # the main study accepted this frame.
    assert (stamp <= t).all() and (main_runners["seg" + AVAILABLE] <= t).all()
    assert (stamp.loc[bets.index] == t.loc[bets.index]).any()
    assert main_run.report.family_size > 0 and MAIN_REAL in main_run.report.stages["roi"]
    row = bets.index[0]
    post = main_runners.loc[row, "post_utc"]

    def restamped(col: str, when) -> pd.DataFrame:
        out = main_runners.copy()
        out.loc[row, col + AVAILABLE] = when
        return out

    at_the_off = main_runners.copy()
    in_a = at_the_off["seg"] == "A"
    at_the_off.loc[in_a, "odds_t" + AVAILABLE] = at_the_off.loc[in_a, "post_utc"]
    cases = [
        ("odds_t a second after t", restamped("odds_t", t.loc[row] + pd.Timedelta(seconds=1))),
        ("odds_t a minute before the off", restamped("odds_t", post - pd.Timedelta(minutes=1))),
        ("odds_t after the race", restamped("odds_t", post + pd.Timedelta(minutes=30))),
        ("odds_t with no stamp", restamped("odds_t", pd.NaT)),
        ("seg a second after t", restamped("seg", t.loc[row] + pd.Timedelta(seconds=1))),
        ("segment A's boards read at the off", at_the_off),
    ]
    outcomes = {
        label: outcome_before_scoring(tmp_path, f"late-{k}", [rule], data)
        for k, (label, data) in enumerate(cases)
    }
    assert outcomes == dict.fromkeys(outcomes, "look-ahead"), outcomes


def test_a_late_stamped_null_is_a_late_observation(tmp_path, monkeypatch, main_runners):
    """A missing value is an observation when its declaration stamps it: a board update after t
    that blanks a price the board held at t (T-1 against this family's T-5, say), a second after
    t or after the race, with its source or without one, and a card amendment after t that
    blanks a race's segment, are each refused with LookAheadSelection before anything is scored
    or written, like a late value. A genuinely missing input, with neither source nor stamp (as
    an unmatched left join leaves it), and a value observed missing by t are accepted; the
    runner is then not selected (test_a_missing_price_in_an_object_column_is_unselected)."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    rule = MAIN_RULES[0]
    t = decision_utc(main_runners)
    row = bets_of(main_runners, rule).index[0]
    race = main_runners["race_id"] == main_runners.loc[row, "race_id"]
    post = main_runners.loc[row, "post_utc"]
    # Premise: the row is a scored bet of the rule priced on the board by t, in a race of A.
    assert main_runners.loc[row, "odds_t" + AVAILABLE] <= t.loc[row]
    assert (main_runners.loc[race, "seg"] == "A").all() and race.sum() > 1

    def blanked(col: str, rows, source: str | None, when) -> pd.DataFrame:
        out = main_runners.copy()
        out.loc[rows, col] = None
        out.loc[rows, col + SOURCE] = source
        out.loc[rows, col + AVAILABLE] = when
        assert np.all(pd.isna(out.loc[rows, col]))  # premise: the value is missing
        return out

    second = pd.Timedelta(seconds=1)
    refused = {
        "a T-1 board update blanking the price": blanked(
            "odds_t", row, BOARD, post - pd.Timedelta(minutes=1)
        ),
        "the price blanked a second after t": blanked("odds_t", row, BOARD, t.loc[row] + second),
        "the price blanked after the race": blanked(
            "odds_t", row, BOARD, post + pd.Timedelta(minutes=30)
        ),
        "the price blanked at T-1, no source": blanked(
            "odds_t", row, None, post - pd.Timedelta(minutes=1)
        ),
        "the race's segment blanked a second after t": blanked(
            "seg", race, RACES, t.loc[race] + second
        ),
    }
    accepted = {
        "the price genuinely missing (no source, no stamp)": blanked("odds_t", row, None, pd.NaT),
        "the price observed missing at t": blanked("odds_t", row, BOARD, t.loc[row]),
        "the race's segment genuinely missing": blanked("seg", race, None, pd.NaT),
    }
    outcomes = {
        label: outcome_before_scoring(tmp_path, f"late-null-{k}", [rule], data)
        for k, (label, data) in enumerate({**refused, **accepted}.items())
    }
    want = {**dict.fromkeys(refused, "look-ahead"), **dict.fromkeys(accepted, "scored")}
    assert outcomes == want, outcomes


# ---------------------------------------------------------------------------------------------
# DoD: the holdout is not opened
# ---------------------------------------------------------------------------------------------


def test_the_holdout_is_never_opened(main_run, omega_run):
    """Both studies were handed a LockedHoldout whose open raises, and both completed."""
    for run in (main_run, omega_run):
        assert run.report.family_size > 0  # the study completed and reported
        assert run.holdout.calls == []
        assert run.log_after == run.log_before  # the country's holdout log is untouched
        records = run.registry.records(run.spec.family_id)
        assert records
        assert all(record.holdout_open is None for record in records)


def _with_races_on(data: pd.DataFrame, days: list[date]) -> pd.DataFrame:
    """`data` plus a copy of its first race on each of `days`, in a meeting of its own, its post
    and availability stamps moved with it."""
    first = data[data["race_id"] == data["race_id"].iloc[0]]
    stamps = [col for col in first.columns if isinstance(first[col].dtype, pd.DatetimeTZDtype)]
    extra = []
    for day in days:
        shift = pd.Timestamp(day) - first["race_date"].iloc[0]
        extra.append(
            first.assign(
                race_id=f"late-{day.isoformat()}-r01",
                meeting_id=f"late-{day.isoformat()}",
                race_date=pd.Timestamp(day),
                **{col: first[col] + shift for col in stamps},
            )
        )
    return pd.concat([data, *extra], ignore_index=True)


def test_rows_on_a_holdout_day_are_refused(tmp_path, monkeypatch, main_runners):
    """A row on a holdout day (the committed definition: from 2026-11-28, the seeded half of the
    race days) refuses the run before anything is scored or written; development days after the
    holdout's start do not."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    held = [date(2026, 11, 29), date(2026, 12, 6)]
    development = [date(2026, 11, 28), date(2026, 11, 30), date(2026, 12, 5)]
    assert all(is_holdout_day("japan", day) for day in held)
    assert not any(is_holdout_day("japan", day) for day in development)

    for k, day in enumerate(held):
        registry = new_registry(tmp_path / f"held{k}")
        spec = make_family(f"test-88-holdout-day-{k}", MAIN_RULES[:2])
        freeze(registry, spec)
        holdout = TrippedHoldout()
        data = _with_races_on(main_runners, [day])
        with pytest.raises(ValueError):
            run_study(spec, data, registry=registry, holdout=holdout, seed=1)
        assert registry.records(spec.family_id) == [], day
        assert holdout.calls == [], day

    late = _with_races_on(main_runners, development)
    run = run_frozen(tmp_path / "late", "test-88-late-development", MAIN_RULES[:2], late)
    assert set(run.report.rungs) == {rule[0] for rule in MAIN_RULES[:2]}
    assert run.holdout.calls == []


# ---------------------------------------------------------------------------------------------
# DoD: `sources` is validated before scoring; licensed sources stay withheld
# ---------------------------------------------------------------------------------------------


def test_a_licensed_source_cannot_slip_past_the_registry_or_its_summary(
    tmp_path, monkeypatch, main_runners
):
    """A bare string ('jrdb' would become the names 'j', 'r', 'd', 'b'), bytes, a collection
    holding a non-name or a blank name, or no source at all is refused before anything is scored
    or written. A collection naming JRDB or JRA-VAN keeps both protections: a registry inside
    the repository refuses the run's records, and the summary of a registry outside it shows
    every record of the family withheld, no metric."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(data_dir))
    rules = MAIN_RULES[:2]
    malformed = {
        "a bare str": "jrdb",
        "a bare str, another spelling": "JRA-VAN",
        "bytes": b"jrdb",
        "a non-str name": ["jrdb", 3],
        "a nested collection": [["jrdb"]],
        "a blank name": ("jra", " "),
        "no source": (),
    }
    outcomes = {
        label: outcome_before_scoring(tmp_path, f"sources-{k}", rules, main_runners, sources=src)
        for k, (label, src) in enumerate(malformed.items())
    }
    refused = {
        label for label, outcome in outcomes.items() if outcome in ("TypeError", "ValueError")
    }
    assert refused == set(malformed), outcomes

    for k, sources in enumerate([("jrdb",), ["JRA-VAN"]]):
        spec = make_family(f"test-88-licensed-{k}", rules)
        # A registry inside the repository: the run is refused and writes no record.
        repo = tmp_path / f"repo{k}"
        (repo / "pyproject.toml").parent.mkdir(parents=True)
        (repo / "pyproject.toml").write_text("", encoding="utf-8")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(env, "repo_root", lambda repo=repo: repo)
            inside = new_registry(repo)
            freeze(inside, spec)
            with pytest.raises(LicensedOutputInRepo):
                run_study(
                    spec,
                    main_runners.copy(),
                    registry=inside,
                    holdout=TrippedHoldout(),
                    seed=1,
                    sources=sources,
                )
            assert inside.records(spec.family_id) == [], sources
        # A registry outside it: the run is recorded, and the summary withholds every value.
        run = run_spec(tmp_path / f"public{k}", spec, main_runners, sources=sources)
        records = run.registry.records(spec.family_id)
        assert records, sources
        # The summary directory (#334/#525): one <family_id>.md per family and a README index.
        summary = tmp_path / f"summary{k}"
        write_summary(run.registry, summary)
        index = (summary / "README.md").read_text(encoding="utf-8")
        assert f"({spec.family_id}.md)" in index, index
        text = (summary / f"{spec.family_id}.md").read_text(encoding="utf-8")
        assert spec.family_id in text and text.count(WITHHELD_LABEL) == len(records), text
        names = {metric.metric for record in records for metric in record.metrics}
        assert names and not [name for name in names if name in text + index], (text, index)


# ---------------------------------------------------------------------------------------------
# DoD: return reporting uses the walk-forward test rows that log loss scores
# ---------------------------------------------------------------------------------------------


def with_training_only_winners(data: pd.DataFrame, rule: tuple, n_races: int = 40) -> pd.DataFrame:
    """`data` plus copies of `n_races` training-only races of the rule's segment (before the first
    scored quarter, in new meetings), in each of which a runner the rule selects wins at
    TRAINING_DIVIDEND."""
    early = data[data["race_date"] < FIRST_SCORED]
    bets = selected(early, rule)
    copies = []
    for k, race_id in enumerate(list(dict.fromkeys(bets["race_id"]))[:n_races]):
        race = data[data["race_id"] == race_id]
        won = (race.index == bets.index[bets["race_id"] == race_id][0]).astype(int)
        dividend = np.where(won == 1, TRAINING_DIVIDEND, race["win_dividend"])
        copies.append(
            race.assign(
                race_id=f"train-{k:02d}",
                meeting_id=f"train-{k:02d}",
                won=won,
                win_dividend=dividend,
            )
        )
    return pd.concat([data, *copies], ignore_index=True)


def test_returns_are_reported_on_the_walk_forward_test_rows_only(
    tmp_path, monkeypatch, main_runners, main_run
):
    """Every return is read on the rows the walk-forward folds score. Training-only races (before
    2020-Q1) holding big winning bets of segment B's rule change no rule's bet count, omega,
    expected return, P(omega_s > 1/(1 - t)), predictive interval or realised ROI, and each count
    and ROI is the one of the rule's selected bets among the scored rows."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    rule = MAIN_RULES[1]
    planted = with_training_only_winners(main_runners, rule)
    key = ["race_id", "horse_id"]
    # Premise: the scored rows are those from 2020-Q1 on, the planted races are none of them, and
    # read over every row the rule's return would change by far.
    training_only = int((main_runners["race_date"] < FIRST_SCORED).sum())
    assert training_only > 0.3 * len(main_runners), training_only
    assert (scored(main_runners)["race_date"] >= FIRST_SCORED).all()
    assert len(scored(main_runners)) == len(main_runners) - training_only
    assert scored(planted)[key].equals(scored(main_runners)[key])
    assert len(planted) > len(main_runners)
    everywhere = realised_roi(selected(planted, rule)) - realised_roi(selected(main_runners, rule))
    assert everywhere > 0.5, everywhere

    run = run_frozen(tmp_path / "planted", "test-88-training-winners", MAIN_RULES, planted)
    for each in MAIN_RULES:
        rid, rows = each[0], bets_of(main_runners, each)
        entry, roi = run.report.stages["omega"][rid], run.report.stages["roi"][rid]
        assert entry["n_bets"] == len(rows), (rid, entry["n_bets"], len(rows))
        assert roi["n_bets"] == len(rows), rid
        assert roi["roi"] == pytest.approx(realised_roi(rows), rel=1e-9, abs=1e-12), rid
        assert omega_numbers(entry) == pytest.approx(
            omega_numbers(main_run.report.stages["omega"][rid]), rel=1e-9
        ), rid
        assert run.report.p_beats_takeout[rid] == pytest.approx(
            main_run.report.p_beats_takeout[rid]
        )


MISSING_TRAINING_ROWS = 30  # training-only rows the rules would select, their price missing
MISSING_SCORED_BETS = 3  # per rule: scored bets whose price is genuinely missing


def with_missing_prices(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index, pd.Index]:
    """`data` with odds_t an object column holding pd.NA for a missing price: on training-only
    rows the main rules would select (observed missing at t: still declared as the board at t,
    stamped by t), and on the first scored bets of each main rule, genuinely missing (no source,
    no stamp, as an unmatched left join leaves them). Also the two sets of rows."""
    early = data[data["race_date"] < FIRST_SCORED]
    training = pd.Index(
        np.concatenate([selected(early, rule).index for rule in MAIN_RULES])
    ).unique()[:MISSING_TRAINING_ROWS]
    blank = pd.Index(
        np.concatenate([bets_of(data, rule).index[:MISSING_SCORED_BETS] for rule in MAIN_RULES])
    )
    prices = data["odds_t"].astype(object)
    prices.loc[training.union(blank)] = pd.NA
    out = data.assign(odds_t=prices)
    out.loc[blank, "odds_t" + SOURCE] = None
    out.loc[blank, "odds_t" + AVAILABLE] = pd.NaT
    return out, training, blank


def test_a_missing_price_in_an_object_column_is_unselected(
    tmp_path, monkeypatch, main_runners, main_run
):
    """A price column of object dtype, pd.NA for a missing price. On training-only rows it stops
    no rule: they are never bets. On the scored rows a missing price is simply not selected.
    Every rule keeps the main run's log-loss result (selection does not enter it), and its bet
    count and realised ROI are those of its selected bets among the scored rows less the ones
    whose price is missing; omega and P(omega_s > 1/(1 - t)) are measured on them."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    data, training, blank = with_missing_prices(main_runners)
    priced = main_runners.copy()
    priced.loc[blank, "odds_t"] = np.nan  # the same prices, the missing ones as NaN floats
    # Premise: an object column holding pd.NA on training-only rows and on scored bets.
    on_scored = scored(main_runners).index
    assert data["odds_t"].dtype == object
    assert data.loc[training.union(blank), "odds_t"].map(lambda v: v is pd.NA).all()
    assert len(training) == MISSING_TRAINING_ROWS and not training.isin(on_scored).any()
    assert len(blank) == MISSING_SCORED_BETS * len(MAIN_RULES) and blank.isin(on_scored).all()
    try:
        run = run_frozen(tmp_path / "missing", "test-88-missing-prices", MAIN_RULES, data)
    except Exception as exc:
        pytest.fail(f"a missing price in an object column stopped the study: {exc!r}")

    report, baseline = run.report, main_run.report
    for stage in ("log_loss", "omega", "roi"):
        assert set(report.stages[stage]) == {rule[0] for rule in MAIN_RULES}, stage
    for rule in MAIN_RULES:
        rid, rows = rule[0], bets_of(priced, rule)
        assert len(rows) == len(bets_of(main_runners, rule)) - MISSING_SCORED_BETS  # premise
        ll, unforced = report.stages["log_loss"][rid], baseline.stages["log_loss"][rid]
        assert (ll["delta_ll"], ll["lo"], ll["hi"]) == pytest.approx(
            (unforced["delta_ll"], unforced["lo"], unforced["hi"]), rel=1e-9
        ), rid
        entry, roi = report.stages["omega"][rid], report.stages["roi"][rid]
        assert entry["n_bets"] == roi["n_bets"] == len(rows), (rid, entry["n_bets"], len(rows))
        assert roi["roi"] == pytest.approx(realised_roi(rows), rel=1e-9, abs=1e-12), rid
        assert finite(entry["omega_selected"]) and 0.0 <= entry["p_beats_takeout"] <= 1.0, entry


# ---------------------------------------------------------------------------------------------
# DoD: the report states family size (dropped variants included), survivors and P(ω_s > 1/(1-t))
# ---------------------------------------------------------------------------------------------


def test_report_family_size_includes_dropped_variants(main_run, omega_run):
    for run, dropped in ((main_run, MAIN_DROPPED), (omega_run, OMEGA_DROPPED)):
        ids = [rule[0] for rule in run.rules]
        family_id = run.spec.family_id
        assert run.report.family_size == len(ids) + len(dropped)
        assert run.report.family_size == run.registry.family_size(family_id)
        variants = {record.variant for record in run.registry.records(family_id)}
        assert set(ids) | set(dropped) == variants  # each rule recorded, the dropped kept


RERUN_RULES = [
    MAIN_RULES[1],
    ("P4-seg2-x", "beta", {"seg2": "X"}, ["odds_t"], {"odds_t": [None, 3.0]}, "above"),
]


def test_a_rerun_that_cannot_score_a_scored_rule_writes_nothing(
    tmp_path, monkeypatch, main_runners
):
    """The registry is append-only, so a rule scored once stays in the family. A rerun on a
    refresh every rule can still be scored on is another run; a rerun on a refresh where
    `seg2` has no usable value, so P4-seg2-x can no longer be scored, is a ValueError naming that
    rule, raised before the rerun appends a record (not part-way, after its first records)."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    k = pd.factorize(main_runners["race_id"])[0] % 3
    data = declare(main_runners.assign(seg2=np.where(k == 0, "X", "Y")), seg2=RACES)
    first = run_frozen(tmp_path / "rerun", "test-88-rerun", RERUN_RULES, data)
    assert_claims_scored(first)
    registry, spec = first.registry, first.spec
    before = registry.records()

    refresh = data.assign(win_dividend=data["win_dividend"].where(data["won"] == 1))
    report = run_study(spec, refresh, registry=registry, holdout=TrippedHoldout(), seed=STUDY_SEED)
    assert set(report.rungs) == {rule[0] for rule in RERUN_RULES}
    grown = registry.records()
    assert len(grown) > len(before)  # a refresh is another run of the family

    blank = refresh.assign(seg2=None)
    holdout = TrippedHoldout()
    with pytest.raises(ValueError) as caught:
        run_study(spec, blank, registry=registry, holdout=holdout, seed=STUDY_SEED)
    assert not isinstance(caught.value, LookAheadSelection)
    assert "P4-seg2-x" in str(caught.value)
    assert registry.records() == grown, "the refused rerun appended records"
    assert holdout.calls == []


def test_report_states_p_beats_takeout_per_segment(main_run, omega_run):
    """One probability per rule; J1, whose scored bets return about 2 * (1 - t) per unit at the
    final price, beats the takeout, and J3, whose scored bets clearly lose, does not."""
    check_omega_premise(omega_run.data)
    assert roi_z(bets_of(omega_run.data, OMEGA_RULES[0])) > 1.5
    assert roi_z(bets_of(omega_run.data, OMEGA_RULES[2])) < -1.5
    for run in (main_run, omega_run):
        p = run.report.p_beats_takeout
        assert set(p) == {rule[0] for rule in run.rules}
        for value in p.values():
            assert isinstance(value, float) and 0.0 <= value <= 1.0
    p = omega_run.report.p_beats_takeout
    assert p["P4-j1"] > 0.5
    assert p["P4-j3"] < 0.5


def test_p_beats_takeout_is_read_on_each_rules_selected_bets(
    main_run, direction_run, omega_run, flb_run
):
    """P(omega_s > 1/(1 - t)) follows the return of the bets the rule makes among the scored
    rows, not of every runner of its segment: a selection that clearly beats breakeven at the
    final dividend gets a high probability and one that clearly loses a low one, even for two
    rules on the same segment."""
    same_segment = {rule[0]: bets_of(direction_run.data, rule) for rule in DIRECTION_RULES[:2]}
    assert roi_z(same_segment["P4-a-favourite"]) > CLEAR_Z
    assert roi_z(same_segment["P3-a-longshot"]) < -CLEAR_Z
    checked = {"high": [], "low": []}
    for run in (main_run, direction_run, omega_run, flb_run):
        p = run.report.p_beats_takeout
        for rule in run.rules:
            z = roi_z(bets_of(run.data, rule))
            if z > CLEAR_Z:
                assert p[rule[0]] > 0.8, (rule[0], z, p[rule[0]])
                checked["high"].append(rule[0])
            elif z < -CLEAR_Z:
                assert p[rule[0]] < 0.2, (rule[0], z, p[rule[0]])
                checked["low"].append(rule[0])
    assert len(checked["high"]) >= 2 and len(checked["low"]) >= 4, checked


def _omega_on_tripled_odds(bets, *args, **kwargs):
    """#78's model fitted as if every selected bet were offered at three times its final odds:
    each rule's expected return, and so P(omega_s > 1/(1 - t)), goes up."""
    return fit_segment_roi(bets.assign(odds=bets["odds"] * 3.0), *args, **kwargs)


def test_p_beats_takeout_gates_nothing(tmp_path, direction_runners, direction_run):
    """Omega is reported only (owner answer 4). With #78's model fitted on tripled odds, a rule
    near certain not to beat the takeout becomes near certain to, yet the log-loss verdicts, the
    survivors, every realised ROI and the rungs are unchanged."""
    before = direction_run.report
    p_before = before.p_beats_takeout
    low = [rid for rid in p_before if rid not in before.survivors and p_before[rid] < 0.2]
    assert low, p_before
    with pytest.MonkeyPatch.context() as patch:
        _replace_everywhere(patch, fit_segment_roi, _omega_on_tripled_odds)
        run = run_frozen(tmp_path, "test-88-direction", DIRECTION_RULES, direction_runners)
    after = run.report
    flipped = [rid for rid in low if after.p_beats_takeout[rid] > 0.8]
    assert flipped, after.p_beats_takeout  # premise: the change reached the omega stage
    passed = {rid: entry["passed"] for rid, entry in before.stages["log_loss"].items()}
    assert {rid: entry["passed"] for rid, entry in after.stages["log_loss"].items()} == passed
    assert list(after.survivors) == list(before.survivors)
    assert set(after.stages["roi"]) == set(before.stages["roi"]) == set(p_before)
    for rid, entry in before.stages["roi"].items():
        assert after.stages["roi"][rid]["roi"] == pytest.approx(entry["roi"], rel=1e-12), rid
    assert after.rungs == before.rungs


def test_a_gap_in_one_rules_segment_column_leaves_every_p_finite(gap_run):
    """`seg2` is missing in a third of the races. Its rule's segment and bets are the races
    where it is 'X', and the other rules' probabilities do not depend on it."""
    assert gap_run.data["seg2"].isna().any()
    p = gap_run.report.p_beats_takeout
    assert set(p) == {rule[0] for rule in GAP_RULES}
    for rid, value in p.items():
        assert isinstance(value, float) and math.isfinite(value), (rid, value)
        assert 0.0 <= value <= 1.0, (rid, value)


# The same bets as omega_run's J1 and J2 rules, under another segment: an extra column every
# runner shares, or J2 under another column name and label.
LONGER = {"odds_t": [3.0, None]}
TWIN_RULES = [
    ("P4-j1-twin", "omega", {"jockey": "J1", "always": "yes"}, ["odds_t"], LONGER, "above"),
    ("P4-j2-twin", "omega", {"jockey": "J2", "always": "yes"}, ["odds_t"], LONGER, "above"),
    ("P4-r2", "omega", {"rider": "R2"}, ["odds_t"], LONGER, "above"),
]
TWIN_OF = {"P4-j1-twin": "P4-j1", "P4-j2-twin": "P4-j2", "P4-r2": "P4-j2"}


def with_twins(data: pd.DataFrame) -> pd.DataFrame:
    """`data` with `always` ('yes' for every runner) and `rider` (the jockey relabelled R1...)."""
    twins = data.assign(always="yes", rider=data["jockey"].str.replace("J", "R", regex=False))
    return declare(twins, always=RACES, rider=RUNNERS)


def test_omega_reports_the_return_multiplier_its_selected_bets_identify(
    tmp_path, monkeypatch, omega_runners, omega_run, main_run, direction_run, flb_run
):
    """Every bet of a rule's omega fit is in the rule's segment, so the bets cannot tell a segment
    effect from the odds bands' fixed effects; what they identify is the return multiplier of
    their own mix, `omega_selected`. (1) The same bets under another segment (an extra column all
    runners share, another column name and label) report the same multiplier and the same
    P(omega_s > 1/(1 - t)). (2) At a single takeout the multiplier times (1 - t) is the bets'
    expected return at the final odds, and agrees with their realised return there among the
    scored rows, sum(won * odds_final) / n_bets, within MIX_Z standard errors, for every rule."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    twins = run_frozen(tmp_path / "twins", "test-88-twins", TWIN_RULES, with_twins(omega_runners))
    for twin, plain in TWIN_OF.items():
        twin_rule = next(rule for rule in TWIN_RULES if rule[0] == twin)
        plain_rule = next(rule for rule in OMEGA_RULES if rule[0] == plain)
        bets = selected(twins.data, twin_rule)
        assert bets.index.equals(selected(omega_runners, plain_rule).index), twin  # premise
        a, b = twins.report.stages["omega"][twin], omega_run.report.stages["omega"][plain]
        assert "omega_selected" in a and "omega_selected" in b, (sorted(a), sorted(b))
        assert a["omega_selected"] == pytest.approx(b["omega_selected"], rel=0.02), twin
        p_twin = twins.report.p_beats_takeout[twin]
        assert p_twin == pytest.approx(omega_run.report.p_beats_takeout[plain], abs=0.03), twin

    realised = {}
    for run in (omega_run, twins, main_run, direction_run, flb_run):
        for rule in run.rules:
            entry = run.report.stages["omega"][rule[0]]
            rows = bets_of(run.data, rule)
            assert len(rows) >= 40 and "omega_selected" in entry, (rule[0], sorted(entry))
            mean, se = gross_return(rows)
            expected = entry["omega_selected"] * (1.0 - TAKEOUT)
            assert abs(expected - mean) <= MIX_Z * se, (rule[0], expected, mean, se)
            realised[(run.spec.family_id, rule[0])] = mean
    # Premise: the realised returns run from far below breakeven to far above it.
    assert min(realised.values()) < 0.5 and max(realised.values()) > 1.4, realised


def test_the_odds_bands_are_frozen_with_the_family(tmp_path, monkeypatch, main_runners):
    """The omega model's odds bands come from the frozen cut points: editing them after the
    freeze is refused like any edit, and the study runs on data that carry no band column."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    spec = make_family("test-88-odds-bands", MAIN_RULES[:2])
    registry = new_registry(tmp_path / "edited")
    frozen = freeze(registry, spec)
    for cuts in ([3.0, 8.0, 20.0], [3.0, 8.0], [2.0, 3.0, 8.0, 30.0]):
        edited = dataclasses.replace(spec, odds_band_cut_points=cuts)
        with pytest.raises(FamilyNotFrozen):
            run_study(
                edited, main_runners.copy(), registry=registry, holdout=TrippedHoldout(), seed=1
            )
        assert registry.records() == [], cuts
        assert registry.frozen_spec_hash(spec.family_id) == frozen, cuts

    bare = main_runners.drop(columns=["final_odds_band"])
    run = run_frozen(tmp_path / "bare", "test-88-odds-bands", MAIN_RULES[:2], bare)
    for rid, value in run.report.p_beats_takeout.items():
        assert math.isfinite(value) and 0.0 <= value <= 1.0, (rid, value)


# ---------------------------------------------------------------------------------------------
# DoD: the expected return and P(ω_s > 1/(1 - t)) share the payout-weighted posterior draws
# ---------------------------------------------------------------------------------------------


def mixed_bets(groups: list[tuple]) -> pd.DataFrame:
    """MIXED_BETS bets per band, one segment level: odds cycled from the band's list, each bet
    winning with probability omega * (1 - t) / odds (#78's model, so omega is the truth)."""
    rng = np.random.default_rng(MIXED_SEED)
    frames = []
    for band, takeout, omega, odds in groups:
        prices = np.resize(np.asarray(odds, dtype=float), MIXED_BETS)
        won = (rng.random(MIXED_BETS) < omega * (1 - takeout) / prices).astype(int)
        frames.append(
            pd.DataFrame(
                {"odds": prices, "won": won, "band": band, "takeout": takeout, "seg": "all"}
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_expected_return_and_p_beats_takeout_share_the_payout_weighted_draws():
    """#78's model under mixed takeout. Where the high multiplier sits on the low payout rate, the
    bets' expected return is below break-even while their unweighted mean multiplier clears
    1/(1 - mean t); where it sits on the high one, the reverse. `expected_return` reads the
    payout-weighted quantity (it matches the realised return at the final odds), its interval
    lies on the realised side of 1, P(omega_s > 1/(1 - t)) agrees with it, and the predictive
    interval of the next 100 bets is wider than it."""
    query = {"seg": "all"}
    for side, groups in MIXED_CASES.items():
        bets = mixed_bets(groups)
        pay = bets["won"] * bets["odds"]
        realised, se = float(pay.mean()), float(pay.std(ddof=1) / math.sqrt(len(pay)))
        q = (1 - bets["takeout"]) / bets["odds"]
        per_band = bets.assign(q=q).groupby("band")
        multiplier = per_band["won"].sum() / per_band["q"].sum()
        unweighted = float(
            (multiplier * per_band.size()).sum() / len(bets) * (1 - bets["takeout"].mean())
        )
        # Premise: the realised return is clearly on its side of 1, the unweighted reading on the
        # other.
        if side == "below":
            assert (realised - 1) / se < -3 and unweighted > 1.05, (realised, se, unweighted)
        else:
            assert (realised - 1) / se > 3 and unweighted < 0.95, (realised, se, unweighted)
        model = fit_segment_roi(
            bets,
            segment_cols=["seg"],
            odds_band_col="band",
            takeout_col="takeout",
            seed=STUDY_SEED,
            draws=4000,
        )
        er = model.expected_return(query)
        assert all(finite(v) for v in (er.mean, er.low, er.high, er.level)), (side, er)
        assert er.low <= er.mean <= er.high and 0.0 < er.level < 1.0, (side, er)
        assert abs(er.mean - realised) <= MIX_Z * se, (side, er, realised, se)
        tail = (1 - er.level) / 2
        p = model.p_beats_takeout(query)
        if side == "below":
            assert er.high < 1.0 and p <= tail + 0.01, (side, er, p)
        else:
            assert er.low > 1.0 and p >= 1 - tail - 0.01, (side, er, p)
        predictive = model.predictive_roi(query, 100)
        assert predictive.high - predictive.low > er.high - er.low, (side, predictive, er)


def test_the_reported_expected_return_agrees_with_p_beats_takeout_under_mixed_takeout(
    mixed_run, main_run, omega_run, direction_run, flb_run, gap_run
):
    """Per rule of every family, the Win takeout mixed by odds band in one of them: the reported
    expected return is omega_selected * (1 - mean t) of the rule's scored bets (the multiplier is
    payout-weighted, so omega_s > 1/(1 - t) exactly when the expected return is above 1), a
    reported break-even multiplier is 1/(1 - mean t) and sits on the same side, P(omega_s >
    1/(1 - t)) agrees with the expected return's interval, the expected return matches the
    realised return at the final odds, and the predictive interval is reported apart (wider)."""
    # Premise: in the mixed family the unweighted and payout-weighted readings of the same band
    # multipliers differ by far more than rounding.
    apart = 0
    for rule in mixed_run.rules:
        rows = bets_of(mixed_run.data, rule)
        band = np.digitize(rows["odds_final"].to_numpy(), CUT_POINTS)
        q = (1 - rows["takeout"]) / rows["odds_final"]
        grouped = rows.assign(q=q, band=band).groupby("band")
        multiplier = grouped["won"].sum() / grouped["q"].sum()
        n, t = grouped.size(), grouped["takeout"].first()
        unweighted = float((n * multiplier).sum() / n.sum() * (1 - rows["takeout"].mean()))
        weighted = float((n * multiplier * (1 - t)).sum() / n.sum())
        apart += abs(unweighted - weighted) > 0.01
    assert apart >= 2, apart

    for run in (mixed_run, main_run, omega_run, direction_run, flb_run, gap_run):
        for rule in run.rules:
            rid, rows = rule[0], bets_of(run.data, rule)
            entry = run.report.stages["omega"][rid]
            er, mean_t = entry["expected_return"], float(rows["takeout"].mean())
            assert er["mean"] == pytest.approx(entry["omega_selected"] * (1 - mean_t), rel=1e-6), (
                run.spec.family_id,
                rid,
                er,
                entry["omega_selected"],
                mean_t,
            )
            if "breakeven_omega" in entry:
                assert entry["breakeven_omega"] == pytest.approx(1 / (1 - mean_t), rel=1e-9), rid
                assert (entry["omega_selected"] > entry["breakeven_omega"]) == (er["mean"] > 1), rid
            tail, p = (1 - er["level"]) / 2, entry["p_beats_takeout"]
            if er["high"] < 1:
                assert p <= tail + 0.01, (rid, er, p)
            elif er["low"] > 1:
                assert p >= 1 - tail - 0.01, (rid, er, p)
            else:
                assert tail - 0.01 <= p <= 1 - tail + 0.01, (rid, er, p)
            realised, se = gross_return(rows)
            assert abs(er["mean"] - realised) <= MIX_Z * se, (rid, er, realised, se)
            low, high, _ = entry[PREDICTIVE]
            assert high - low > er["high"] - er["low"], (rid, entry)


# ---------------------------------------------------------------------------------------------
# DoD: nothing is promoted beyond stake rung S1; the study makes no rung decision
# ---------------------------------------------------------------------------------------------


def test_no_rule_is_placed_above_rung_s1(main_run, omega_run, direction_run, flb_run):
    """Every rule sits at D19's default rung S0, survivor or not: even J1, which passes every
    stage with a positive realised return at the final dividend. The study makes no rung
    decision, nothing goes above S1, and nothing in the registry is adopted."""
    j1 = bets_of(omega_run.data, OMEGA_RULES[0])
    assert realised_roi(j1) > 0.2  # premise: an edge a naive ladder would stake on
    survivors = []
    for run in (main_run, omega_run, direction_run, flb_run):
        ids = [rule[0] for rule in run.rules]
        assert run.report.rungs == dict.fromkeys(ids, "S0"), run.report.rungs
        survivors += list(run.report.survivors)
        records = run.registry.records(run.spec.family_id)
        assert all(record.status != "adopted" for record in records)
        assert all(record.owner_signoff is None for record in records)
    assert OMEGA_REAL in survivors and len(survivors) >= 3, survivors  # premise


# ---------------------------------------------------------------------------------------------
# The committed family
# ---------------------------------------------------------------------------------------------


def _owner_family_frame(data: pd.DataFrame) -> pd.DataFrame:
    """`data` with the committed family's columns: segment A's races are handicaps, B's are
    R12, C's are the most concentrated bin, and rank_t is the min-rank of odds_t in the race;
    the first two declared from the races card, the last two from the board at t."""
    seg = data["seg"]
    frame = data.assign(
        weight_type=np.where(seg == "A", "handicap", "other"),
        R12=np.where(seg == "B", "in", "out"),
        entropy_t_bin=np.where(seg == "C", 1, 2),
        rank_t=data.groupby("race_id")["odds_t"].rank(method="min").astype(int),
    )
    return declare(frame, weight_type=RACES, R12=RACES, entropy_t_bin=BOARD, rank_t=BOARD)


def _prose(text: str) -> list[str]:
    """A yaml file's sentences: each run of comment lines joined and split after a full stop,
    and each other line on its own."""
    units, block = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            block.append(stripped.lstrip("#").strip())
            continue
        if block:
            units.append(" ".join(block))
            block = []
        units.append(stripped)
    if block:
        units.append(" ".join(block))
    return [sentence for unit in units for sentence in re.split(r"(?<=[.!?])\s+", unit)]


def test_the_committed_family_cites_its_run_and_awaits_no_confirmation():
    """The file names the follow-up development-fold run (#462), not a follow-up blocked by the
    closed #369, and no sentence of it asks for the odds-band cut points to be confirmed: they
    are proposed and frozen with the family (owner answer 7; the 2026-11-27 reframe, under which
    a study no longer stops for sign-off)."""
    text = FAMILY_FILE.read_text(encoding="utf-8")
    assert re.search(rf"{FOLLOW_UP_RUN}\b", text), f"the file does not cite {FOLLOW_UP_RUN}"
    prose = _prose(text)
    stale = [s for s in prose if re.search(r"blocked\s+by\s+#369\b", s, re.I)]
    assert stale == [], stale
    cut_points = [s for s in prose if re.search(r"cut.?point", s, re.I)]
    asks = [s for s in cut_points if re.search(r"confirm", s, re.I)]
    assert asks == [], asks


def test_the_committed_family_is_the_one_the_owner_approved(tmp_path, monkeypatch, main_runners):
    """`load_family("japan")` reads the committed file: Family P's three runnable rules with their
    thresholds and directions, decision time T-2, meeting clusters, the max-statistic settings,
    the calibrated market, the odds-band cut points and the six E4 variants the owner dropped
    (`dropped_variants`, frozen with the spec). No sign-off is required: a copy without
    `owner_signoff`, or with it null, loads to the same spec, so the frozen hash is the only
    guard. The family freezes as plain JSON and the study runs it, recording each dropped variant
    as failed, so the family size is 3 + 6 = 9 (08 §5.2 step 1)."""
    raw = yaml.safe_load(FAMILY_FILE.read_text(encoding="utf-8"))
    dropped = [entry["variant"] for entry in raw["dropped"]]
    assert len(dropped) == len(set(dropped)) == 6 and not set(dropped) & set(OWNER_RULES)

    spec = load_family("japan")
    assert isinstance(spec, FamilySpec)
    assert sorted(rule.rule_id for rule in spec.rules) == sorted(OWNER_RULES)
    for rule in spec.rules:
        segment, cols, thresholds, direction = OWNER_RULES[rule.rule_id]
        assert rule.claim == "beta", rule.rule_id
        assert rule.segment == segment, rule.rule_id
        assert list(rule.selection_cols) == cols, rule.rule_id
        assert {c: list(b) for c, b in rule.thresholds.items()} == thresholds, rule.rule_id
        assert rule.direction == direction, rule.rule_id
    assert spec.decision_time == "T-2"
    assert spec.cluster_unit == "meeting_id"
    assert spec.benchmark == BENCHMARK
    assert spec.max_stat == OWNER_MAX_STAT
    cuts = list(spec.odds_band_cut_points)
    assert cuts and all(isinstance(c, (int, float)) and math.isfinite(c) and c > 1 for c in cuts)
    assert all(lo < hi for lo, hi in zip(cuts[:-1], cuts[1:], strict=True)), cuts
    # The dropped variants are part of the frozen spec, so the freeze covers them too.
    assert list(getattr(spec, "dropped_variants", None) or []) == dropped

    unsigned = {key: value for key, value in raw.items() if key != "owner_signoff"}
    for k, copy in enumerate([unsigned, {**unsigned, "owner_signoff": None}]):
        path = tmp_path / f"selection_family_{k}.yaml"
        path.write_text(yaml.safe_dump(copy), encoding="utf-8")
        assert load_family(path) == spec, k
    hashes = {
        freeze(new_registry(tmp_path / f"hash{k}"), loaded)
        for k, loaded in enumerate([spec, load_family(path)])
    }
    assert len(hashes) == 1, hashes

    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    run = run_spec(tmp_path / "owner", spec, _owner_family_frame(main_runners))
    assert run.report.rungs == dict.fromkeys(OWNER_RULES, "S0")
    records = run.registry.records(spec.family_id)
    assert {record.variant for record in records} == set(OWNER_RULES) | set(dropped)
    status = {(record.variant, record.status) for record in records}
    assert status == {(rid, "scored") for rid in OWNER_RULES} | {(v, "failed") for v in dropped}
    assert run.report.family_size == len(OWNER_RULES) + len(dropped) == 9
    assert run.report.family_size == run.registry.family_size(spec.family_id)
    for rid, value in run.report.p_beats_takeout.items():
        assert math.isfinite(value) and 0.0 <= value <= 1.0, (rid, value)
    assert set(run.report.stages["roi"]) == set(OWNER_RULES)
