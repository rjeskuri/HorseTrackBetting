"""The pre-registered Family P selection study (ticket #88; 08 §5.2 end to end, §5.3, §8 A6).

Contract pinned here (`horsetrack.countries.japan.studies.selection`), with the owner's answers on
#88 of 2026-11-26 (benchmark, omega, rungs, one-sided passes, odds bands, the family):

- `RuleSpec(rule_id, claim, segment, selection_cols, thresholds, direction)` and
  `FamilySpec(family_id, rules, decision_time, cluster_unit, benchmark, max_stat,
  odds_band_cut_points, dropped_variants=[])` are dataclasses. `claim` is 'beta', 'alpha' or
  'omega'. `dropped_variants` (default empty, so a spec without one still builds) lists the
  variant ids screened and dropped before the freeze (08 §5.2 step 1: the family size counts
  them); it is frozen with the spec, and `run_study` records each as a failed variant of the
  family. Three more fields extend the ticket's signature; each is an owner decision with no
  default, so it is frozen with the family rather than read from code (the values below are test
  inputs, not the owner's):
  - `max_stat`: the multiple-testing settings, `{"level": StepM's family-wise level,
    "block_size": the stationary bootstrap's mean block length in clusters, "n_boot": SPA's and
    StepM's replicates}`, passed to #75's `spa` and `stepm`.
  - `direction` ("above" or "below"): the way the rule bets (owner answer 6). A stage-1 pass
    needs the claim's coefficient on that side of its null: "above" means the segment's
    favourites (a 'beta' claim) or the segment's runners (an 'omega' claim) win more often than
    the benchmark says; "below" means its longshots do. A departure the other way never passes.
  - `odds_band_cut_points`: ascending final decimal odds that cut the omega model's
    favourite-longshot odds bands (its fixed effects; owner answer 7). The study derives the
    bands from them; the data carry no band column.
- Frozen means `registry.freeze_family(spec.family_id, dataclasses.asdict(spec))`. `run_study`
  raises `FamilyNotFrozen` before it scores or writes anything unless
  `registry.frozen_spec_hash(spec.family_id)` is the hash `freeze_family` gives
  `dataclasses.asdict(spec)`: an unfrozen family, or any edit after the freeze (a direction or a
  cut point included), is refused, and the study never freezes the family itself.
- `data` is one row per runner, development rows only, every row a starter, one winner per race:
    race_id          the race
    meeting_id       the race's meeting (venue x day); `cluster_unit` names this column
    race_date        the race day (pd.Timestamp): orders the clusters for SPA and StepM. A row on
                     a holdout day (`core.eval.holdout.is_holdout_day`) makes run_study raise a
                     ValueError before anything is scored or written
    horse_id         the runner within its race
    odds_t           decimal Win odds on the board at the decision time t (selection may use it)
    odds_final       final decimal Win odds: the price the log-loss stage conditions on
    win_dividend     the Win dividend per unit stake the runner pays if it wins; it differs from
                     odds_final by a little (a late pool move), so ROI provably reads the dividend.
                     A loser's may be missing (NaN): it pays nothing
    won              1 for the race's winner, else 0
    takeout          the Win pool's takeout t (0.2)
    final_odds_band  odds band of odds_final (1 to 4); present in some frames, never required
    final_rank       rank of odds_final in the race (1 = favourite)
    final_entropy    normalised entropy of the race's final-odds probabilities
    seg, jockey      segment columns: `seg` is constant within a race, `jockey` varies within it
- A rule's segment `{col: level}` holds the runners with `data[col] == level` for every pair (a
  missing value is in no segment). A 'beta' claim's segment columns are race-level (#77's
  `segment_beta`). `thresholds` maps each selection column to `[lo, hi]`, both ends inclusive,
  `None` an open end; the rule bets one unit to Win on each runner of its segment whose selection
  columns are all in range: its selected bets.
- `benchmark="calibrated_market"` is D4's calibrated market (owner answer 3): the conditional-logit
  natural cubic spline in ln q (`models.v0.calibration`), fitted in fold. A favourite-longshot bias
  the whole market shares is in the benchmark, so no segment is credited with it.
- Look-ahead: `odds_final`, every `final_*` column and the store's final-price columns `win_odds`
  and `win_fav_rank` are final-price columns, and every column Japan's availability config
  classifies as a label (`core.store.asof.load_availability("japan")`: any table's
  `label_columns`, or a name one of its `label_prefixes` starts) is a result a view at t never
  returns (`finish_pos`, `last_3f_s`, a payout...). They may condition a probability claim (the
  log-loss stage at the final price, the omega model's odds bands) but never select a bet: a rule
  naming one in `selection_cols` or in its `segment` is refused with `LookAheadSelection`.
  Board-at-t columns (`odds_t`) are allowed.
- `run_study(...) -> StudyReport(family_size, stages, survivors, p_beats_takeout, rungs)`:
  - `stages` holds "log_loss", "max_stat", "omega" and "roi" in that order (other keys allowed):
    `stages["log_loss"][rule_id]["passed"]` (bool) for every rule (its probability claim on all
    runners of its segment against the benchmark, #69's paired bootstrap over `cluster_unit`,
    one-sided in the rule's direction); `stages["max_stat"]["superior"]`, the rule ids StepM
    finds superior; `stages["roi"]`, keyed by the survivors only, `{"roi": float, "n_bets": int}`
    where roi = (sum of the selected winners' win_dividend) / n_bets - 1 over every row of `data`
    the rule selects. A survivor's registry records carry it as the metric `roi_win_dividend`.
  - `survivors`: the rules that passed the log-loss stage and that StepM finds superior. The
    log-loss verdict comes from #69's `paired_bootstrap` and the max-statistic one from #75's
    `stepm` (ticket outline); one test replaces each, in every loaded horsetrack module that
    holds it, to force its verdict.
  - `p_beats_takeout[rule_id]`: P(omega_s > 1/(1 - t) | data), in [0, 1], from #78's model
    (`fit_segment_roi`, its `p_beats_takeout`) fitted on the rule's selected bets (owner answer
    4), with the family's odds bands as fixed effects: one rule's gaps never blank another's.
    Reported only: it never changes a survivor or a rung.
  - `stages["omega"][rule_id]`, for every rule: its `p_beats_takeout` and `omega_selected`, the
    posterior mean of the return multiplier of the rule's selected bets (the mean over those
    bets of each bet's omega, band and segment effects combined: every bet of the fit is in the
    rule's segment, so that mix is what the bets identify, not a segment effect apart from the
    bands). Only a survivor's entry carries a return: its posterior predictive ROI,
    `predictive_roi_100` ([low, high, level], 08 §5.2 step 4); a non-survivor's entry, and its
    registry records, carry no ROI of any kind.
  - `rungs[rule_id]`: "S1" (shadow) for a survivor, "S0" for every other rule (owner answer 5);
    nothing is adopted.
  - `family_size`: `registry.family_size(family_id)` after the run: the rules (one registry record
    each, variant = rule_id), the spec's `dropped_variants` (failed records) and the variants
    already in the family, failed (dropped) included.
  - The holdout passed in is never opened: the study completes with one whose `open` raises.
  - A rerun on a refreshed frame: a rule the registry already holds scored that the new frame
    cannot score is a ValueError naming the rule, raised before the rerun appends any record
    (the max-statistic stage would otherwise refuse the family after the first records).
- `load_family("japan")` reads the committed config/selection_family.yaml, signed off by the
  owner on #88, its `dropped` variants carried as the spec's `dropped_variants`; a copy without
  the sign-off raises `FamilyNotApproved`.

Not pinned: the log-loss bootstrap's level and replicates (D7, config/scoring.yaml), the form of
a claim against the spline (a power on p_cal or a slope on ln q), the omega model's draws and
segment columns, and on which side of a cut point a price on it falls.

All data is synthetic and seeded. Expected values come from the generator's own arrays (selected
rows, realised wins, a conditional-logit Newton written here), never from the code under test.
Where the benchmark matters, the generator's truth is a power law in q, so the market-wide power
fit written here stands in for the calibrated market. Each test first asserts the premise it
relies on. Rule ids start with the Family P candidate they belong to (P3, P4, P6), the
registry's hypothesis-id form. The families are test-only and decide nothing.
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

from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.holdout import (
    DEFAULT_COUNTRIES_DIR,
    LOG_NAME,
    LockedHoldout,
    is_holdout_day,
)
from horsetrack.core.eval.registry import Registry, RunRecord
from horsetrack.core.eval.segment_roi import fit_segment_roi
from horsetrack.core.eval.snooping import stepm
from horsetrack.core.store.asof import load_availability
from horsetrack.countries.japan.studies.selection import (
    FamilyNotApproved,
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
STUDY_SEED = 88
BENCHMARK = "calibrated_market"
MAX_STAT = {"level": 0.05, "block_size": 4, "n_boot": 500}  # test inputs, not the owner's
CUT_POINTS = [3.0, 8.0, 30.0]  # test input: the odds bands of `final_odds_band` below
LOOK_AHEAD = ("odds_final", "final_odds_band", "final_rank", "final_entropy")
HOLDOUT_LOG = DEFAULT_COUNTRIES_DIR / "japan" / LOG_NAME
FAMILY_FILE = DEFAULT_COUNTRIES_DIR / "japan" / "config" / "selection_family.yaml"
CLEAR_Z = 2.5  # a selection's realised return this many standard errors from breakeven
# The reported return multiplier of a rule's bets, times (1 - t), within this many standard errors
# of their realised return at the final odds (an unbiased estimate of the same mean).
MIX_Z = 3.5
FOLLOW_UP_RUN = "#462"  # the development-fold run of the committed family

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
# Each departing segment bet both ways, and a segment with the market-wide bias only.
DIRECTION_RULES = [
    ("P4-a-favourite", "beta", {"seg": "A"}, ["odds_t"], FAVOURITES, "above"),
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


# ---------------------------------------------------------------------------------------------
# Synthetic runners
# ---------------------------------------------------------------------------------------------


def make_runners(seed: int, *, beta: dict[str, float], jockey_boost: float = 0.0) -> pd.DataFrame:
    """N_MEETINGS meetings of RACES_PER_MEETING races, 8 to 14 runners each (module docstring).

    Each race's final odds are (1 - t) / share for a clipped Dirichlet(1) share, rounded to 0.1.
    With q the normalised inverse final odds, the winner is drawn from softmax(beta[seg] * ln q +
    jockey_boost * [jockey == J1]), so beta 1 and no boost is a race the final price gets right.
    Segments cycle over the races; J1 rides only in segment A's races, J2 and J3 only elsewhere.
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
    return pd.concat(frames, ignore_index=True)


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
    """The rows a rule bets on: its segment, every selection column in [lo, hi] inclusive."""
    _, _, segment, selection_cols, thresholds = rule[:5]
    keep = pd.Series(True, index=data.index)
    for col, level in segment.items():
        keep &= data[col] == level
    for col in selection_cols:
        lo, hi = thresholds[col]
        if lo is not None:
            keep &= data[col] >= lo
        if hi is not None:
            keep &= data[col] <= hi
    return data[keep]


def realised_roi(rows: pd.DataFrame) -> float:
    return float((rows["won"] * rows["win_dividend"]).sum() / len(rows) - 1.0)


def roi_z(rows: pd.DataFrame) -> float:
    """The selection's realised return per unit stake over its standard error (0 = breakeven)."""
    pay = rows["won"] * rows["win_dividend"].fillna(0.0) - 1.0
    return float(pay.mean() / (pay.std(ddof=1) / math.sqrt(len(pay))))


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


def roi_keys(mapping) -> list[str]:
    """The keys of a stage entry, or the metric names of a record, that report a return."""
    return [name for name in mapping if "roi" in str(name).lower()]


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


def make_family(family_id: str, rules: list[tuple]) -> FamilySpec:
    return FamilySpec(
        family_id=family_id,
        rules=[make_rule(*rule) for rule in rules],
        decision_time="T-5",  # five minutes before the off: a test value
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


def run_spec(root: Path, spec: FamilySpec, data, dropped=(), rules=()) -> Run:
    registry = new_registry(root)
    add_dropped(registry, spec.family_id, list(dropped))
    freeze(registry, spec)
    holdout = TrippedHoldout()
    before = read_holdout_log()
    with pytest.MonkeyPatch.context() as patch:
        (root / "data").mkdir(parents=True, exist_ok=True)
        patch.setenv("HORSETRACK_DATA_DIR", str(root / "data"))
        report = run_study(spec, data.copy(), registry=registry, holdout=holdout, seed=STUDY_SEED)
    return Run(list(rules), spec, data, registry, holdout, report, before, read_holdout_log())


def run_frozen(root: Path, family_id: str, rules: list[tuple], data, dropped=()) -> Run:
    return run_spec(root, make_family(family_id, rules), data, dropped, rules)


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
    the races each) and no dividend on any loser."""
    out = data.copy()
    k = pd.factorize(out["race_id"])[0] % 3
    out["seg2"] = pd.Series(np.where(k == 0, "X", "Y"), index=out.index, dtype=object)
    out.loc[k == 2, "seg2"] = None
    out.loc[out["won"] == 0, "win_dividend"] = np.nan
    return out


GAP_RULES = [
    MAIN_RULES[0],
    MAIN_RULES[1],
    ("P4-seg2-x", "beta", {"seg2": "X"}, ["odds_t"], {"odds_t": [None, 3.0]}, "above"),
]


@pytest.fixture(scope="module")
def gap_run(tmp_path_factory, main_runners) -> Run:
    root = tmp_path_factory.mktemp("gaps")
    return run_frozen(root, "test-88-gaps", GAP_RULES, with_gaps(main_runners))


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
# DoD: ROI only for the rules that survived the log-loss and max-statistic stages
# ---------------------------------------------------------------------------------------------


def test_roi_stage_runs_only_on_log_loss_survivors(main_run, omega_run):
    """Observed through `report.stages`: in each family only the rule whose claim is real
    reaches the ROI stage, and its ROI is the realised return of the rows it selects at t, paid
    at the final dividend."""
    check_main_premise(main_run.data)
    check_omega_premise(omega_run.data)

    for run, real in ((main_run, MAIN_REAL), (omega_run, OMEGA_REAL)):
        report, ids = run.report, [rule[0] for rule in run.rules]
        names = list(report.stages)
        assert {"log_loss", "max_stat", "omega", "roi"} <= set(names), names
        order = [name for name in names if name in ("log_loss", "max_stat", "omega", "roi")]
        assert order == ["log_loss", "max_stat", "omega", "roi"], names

        log_loss, superior = report.stages["log_loss"], report.stages["max_stat"]["superior"]
        assert set(log_loss) == set(ids)  # every rule's claim is tested, not only some
        assert set(superior) <= set(ids)
        passed_both = {r for r in ids if log_loss[r]["passed"] is True and r in superior}
        assert set(report.survivors) == passed_both
        assert list(report.survivors) == [real]
        assert set(report.stages["roi"]) == {real}

        rule = next(rule for rule in run.rules if rule[0] == real)
        rows = selected(run.data, rule)
        assert len(rows) > 50
        assert realised_roi(rows) != pytest.approx(  # the dividend, not the final odds
            float((rows["won"] * rows["odds_final"]).sum() / len(rows) - 1.0), abs=1e-6
        )
        roi = report.stages["roi"][real]
        assert roi["n_bets"] == len(rows)
        assert roi["roi"] == pytest.approx(realised_roi(rows), rel=1e-9, abs=1e-12)

    # Both ends of a range are inclusive: rows on the bounds are among the main rule's bets.
    rows = selected(main_run.data, MAIN_RULES[0])
    assert (rows["odds_t"] == 1.5).any() and (rows["odds_t"] == 4.0).any()


def _stepm_finding_nothing(*args, **kwargs):
    """#75's StepM, except that it finds no rule superior."""
    return dataclasses.replace(stepm(*args, **kwargs), superior=[])


def _bootstrap_of_the_reversed_difference(diff, clusters, **kwargs):
    """#69's bootstrap of the negated difference: every claim's verdict flips, whichever sign
    the study gives the difference."""
    return paired_bootstrap(-diff, clusters, **kwargs)


def _replace_everywhere(patch: pytest.MonkeyPatch, original, fake) -> None:
    """Point every loaded horsetrack module's name for `original` at `fake`."""
    for name, module in list(sys.modules.items()):
        if name.startswith("horsetrack") and getattr(module, original.__name__, None) is original:
            patch.setattr(module, original.__name__, fake)


def test_roi_needs_both_the_log_loss_and_the_max_statistic_stage(tmp_path, main_runners):
    """Either gate alone stops a rule. With StepM finding nothing, segment A still passes the
    log-loss stage and gets no ROI; with the log-loss bootstrap's verdict reversed, StepM's view
    of A alone does not let it through. The stages' tools are #75's stepm and #69's
    paired_bootstrap (ticket outline), replaced here to force their verdicts."""
    check_main_premise(main_runners)
    gates = [
        ("max_stat", stepm, _stepm_finding_nothing),
        ("log_loss", paired_bootstrap, _bootstrap_of_the_reversed_difference),
    ]
    for gate, original, fake in gates:
        with pytest.MonkeyPatch.context() as patch:
            _replace_everywhere(patch, original, fake)
            run = run_frozen(tmp_path / gate, f"test-88-{gate}-gate", MAIN_RULES[:2], main_runners)
        report = run.report
        assert list(report.survivors) == [], gate
        assert report.stages["roi"] == {}, gate
        # No survivor, so no return anywhere: omega still reports P for each rule, and no ROI.
        for rid in (rule[0] for rule in MAIN_RULES[:2]):
            entry = report.stages["omega"][rid]
            assert roi_keys(entry) == [], (gate, rid, sorted(entry))
            assert 0.0 <= report.p_beats_takeout[rid] <= 1.0, (gate, rid)
        if gate == "max_stat":
            assert report.stages["max_stat"]["superior"] == []
            assert report.stages["log_loss"][MAIN_REAL]["passed"] is True
        else:
            assert report.stages["log_loss"][MAIN_REAL]["passed"] is False


def test_roi_reads_the_winners_dividends_when_losers_carry_none(gap_run):
    """Losers' dividends are missing (NaN) in the store; a survivor's ROI is still the selected
    winners' dividends over its bets, and its registry records carry it as `roi_win_dividend`."""
    data = gap_run.data
    assert data.loc[data["won"] == 0, "win_dividend"].isna().all()
    assert data.loc[data["won"] == 1, "win_dividend"].notna().all()
    report = gap_run.report
    assert MAIN_REAL in report.survivors  # premise: segment A survives here as in the main run
    assert set(report.stages["roi"]) == set(report.survivors)
    records = gap_run.registry.records(gap_run.spec.family_id)
    for rid in report.survivors:
        rows = selected(data, next(rule for rule in gap_run.rules if rule[0] == rid))
        expected = float(rows.loc[rows["won"] == 1, "win_dividend"].sum() / len(rows) - 1.0)
        roi = report.stages["roi"][rid]
        assert roi["n_bets"] == len(rows), rid
        assert math.isfinite(roi["roi"]), rid
        assert roi["roi"] == pytest.approx(expected, rel=1e-9, abs=1e-12), rid
        logged = [
            metric.value
            for record in records
            if record.variant == rid
            for metric in record.metrics
            if metric.metric == "roi_win_dividend"
        ]
        assert logged, f"{rid}: no roi_win_dividend metric in its registry records"
        assert logged[-1] == pytest.approx(expected, rel=1e-9, abs=1e-12), rid


# ---------------------------------------------------------------------------------------------
# The log-loss stage: against D4's calibrated market, one-sided in the rule's direction
# ---------------------------------------------------------------------------------------------


def test_a_market_wide_favourite_longshot_bias_is_no_segment_claim(flb_run):
    """Every race is priced q^1.3: favourites win more often than the normalised price says in
    every segment alike. The calibrated market carries that bias, so no segment departs from it
    and no rule passes the log-loss stage, whichever way it bets."""
    check_flb_premise(flb_run.data)
    report = flb_run.report
    ids = [rule[0] for rule in FLB_RULES]
    assert_claims_scored(flb_run)
    assert {rid: report.stages["log_loss"][rid]["passed"] for rid in ids} == dict.fromkeys(
        ids, False
    )
    assert list(report.survivors) == []
    assert report.stages["roi"] == {}
    assert report.rungs == dict.fromkeys(ids, "S0")


def test_a_segment_claim_passes_only_in_the_direction_its_rule_bets(direction_run):
    """On top of the market-wide bias, segment A's favourites and segment B's longshots win far
    more often than priced. Each segment is bet both ways: only the rule betting the side that
    departs passes and survives; the rule betting the other side of the same departure (a
    favourite rule where favourites are overbet, a longshot rule where longshots are) sits at S0,
    as does the favourite rule of segment C, which carries the market-wide bias only."""
    check_direction_premise(direction_run.data)
    report = direction_run.report
    ids = [rule[0] for rule in DIRECTION_RULES]
    assert_claims_scored(direction_run)
    passed = {rid: report.stages["log_loss"][rid]["passed"] for rid in ids}
    assert passed == {rid: rid in DIRECTION_SURVIVORS for rid in ids}
    assert set(report.survivors) == DIRECTION_SURVIVORS
    assert set(report.stages["roi"]) == DIRECTION_SURVIVORS
    assert report.rungs == {rid: "S1" if rid in DIRECTION_SURVIVORS else "S0" for rid in ids}


# ---------------------------------------------------------------------------------------------
# Selection hygiene: a final-odds selection variable is refused at the ROI stage
# ---------------------------------------------------------------------------------------------


def test_a_final_odds_selection_variable_is_refused_at_the_roi_stage(
    tmp_path, monkeypatch, main_runners
):
    """The real rule (segment A, which survives whatever it selects on) is re-pointed at each
    final-price column in turn, alone or beside a board-at-t column; every run is refused."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    check_main_premise(main_runners)
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
            run_study(spec, main_runners.copy(), registry=registry, holdout=holdout, seed=1)
        assert holdout.calls == [], cols


def test_the_stores_final_price_columns_are_look_ahead(tmp_path, monkeypatch, main_runners):
    """The JRA store names the final Win odds `win_odds` and the final favourite rank
    `win_fav_rank`; a rule selecting on either is refused like any `final_*` column."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    data = main_runners.assign(
        win_odds=main_runners["odds_final"], win_fav_rank=main_runners["final_rank"]
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
    """The cases (a label, a rule) whose one-rule family run_study accepts; each refused case
    must be refused with LookAheadSelection, without opening the holdout."""
    accepted = []
    for k, (label, rule) in enumerate(cases):
        registry = new_registry(tmp_path / f"{name}{k}")
        spec = make_family(f"test-88-{name}-{k}", [rule])
        freeze(registry, spec)
        holdout = TrippedHoldout()
        try:
            run_study(spec, data, registry=registry, holdout=holdout, seed=1)
        except LookAheadSelection:
            pass
        else:
            accepted.append(label)
        assert holdout.calls == [], label
    return accepted


def test_every_label_of_the_availability_config_is_refused_as_a_selection_column(
    tmp_path, monkeypatch, main_runners
):
    """A view at t never returns a label column (Japan's availability config: each table's
    `label_columns`, and every name under a `label_prefixes` entry), so no bet is selected on
    one. Each label column, holding the finishing position, is the only selection column of the
    real rule (segment A, which survives): [1, 1] would bet the winners only and report their
    return. Every run is refused."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    check_main_premise(main_runners)
    labels = availability_labels()
    # Premise: the config's results are among them, beyond the final price.
    assert {"finish_pos", "placing_code", "last_3f_s", "result_published_utc"} <= set(labels)
    assert {"dividend", "is_final_capture"} <= set(labels)  # the board's payout columns
    assert any(name.endswith("win_odds") and name != "win_odds" for name in labels)  # a prefix
    assert all(is_availability_label(name) for name in labels)
    finish = finish_positions(main_runners)
    data = main_runners.assign(**dict.fromkeys(labels, finish))
    assert (data.loc[data["won"] == 1, "finish_pos"] == 1).all()
    assert (data.loc[data["won"] == 0, "finish_pos"] >= 2).all()

    cases = [
        (col, ("P4-seg-a", "beta", {"seg": "A"}, [col], {col: [1, 1]}, "above")) for col in labels
    ]
    accepted = _accepted(tmp_path, "label-selection", cases, data)
    assert accepted == [], f"label columns accepted as a selection column: {accepted}"


def test_a_segment_on_a_look_ahead_column_is_refused(tmp_path, monkeypatch, main_runners):
    """The segment half of selection hygiene: a rule whose segment is read from a result label
    or a final-price column is refused like a selection column. Runner-level segments that are
    the winners (`finish_pos` 1) or the final favourites (`final_rank` 1), and segment A's races
    (the real claim) under the name of a race result, a board payout or a prefixed label."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    renamed = ["result_published_utc", "dividend", "tyb_segment"]
    assert all(is_availability_label(name) for name in ["finish_pos", *renamed])
    data = main_runners.assign(
        finish_pos=finish_positions(main_runners), **{col: main_runners["seg"] for col in renamed}
    )
    anywhere = {"odds_t": [None, None]}
    cases = [
        ("finish_pos", ("P4-winners", "omega", {"finish_pos": 1}, ["odds_t"], anywhere, "above")),
        ("final_rank", ("P4-final-fav", "omega", {"final_rank": 1}, ["odds_t"], anywhere, "above")),
    ]
    cases += [
        (col, ("P4-seg-a", "beta", {col: "A"}, ["odds_t"], {"odds_t": [1.5, 4.0]}, "above"))
        for col in renamed
    ]
    accepted = _accepted(tmp_path, "label-segment", cases, data)
    assert accepted == [], f"look-ahead columns accepted as a segment: {accepted}"


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
    """`data` plus a copy of its first race on each of `days`, in a meeting of its own."""
    first = data[data["race_id"] == data["race_id"].iloc[0]]
    extra = [
        first.assign(
            race_id=f"late-{day.isoformat()}-r01",
            meeting_id=f"late-{day.isoformat()}",
            race_date=pd.Timestamp(day),
        )
        for day in days
    ]
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
    data = main_runners.assign(seg2=np.where(k == 0, "X", "Y"))
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
    """One probability per rule's segment; J1, whose runners return about 2 * (1 - t) per unit
    at the final price, beats the takeout and J2 and J3 do not."""
    check_omega_premise(omega_run.data)
    for run in (main_run, omega_run):
        p = run.report.p_beats_takeout
        assert set(p) == {rule[0] for rule in run.rules}
        for value in p.values():
            assert isinstance(value, float) and 0.0 <= value <= 1.0
    p = omega_run.report.p_beats_takeout
    assert p["P4-j1"] > 0.5
    assert p["P4-j2"] < 0.5 and p["P4-j3"] < 0.5


def test_p_beats_takeout_is_read_on_each_rules_selected_bets(main_run, direction_run):
    """P(omega_s > 1/(1 - t)) follows the return of the bets the rule makes, not of every runner
    of its segment: a selection that clearly beats breakeven at the final dividend gets a high
    probability and one that clearly loses a low one, even for two rules on the same segment."""
    same_segment = {rule[0]: selected(direction_run.data, rule) for rule in DIRECTION_RULES[:2]}
    assert roi_z(same_segment["P4-a-favourite"]) > CLEAR_Z
    assert roi_z(same_segment["P3-a-longshot"]) < -CLEAR_Z
    checked = {"high": [], "low": []}
    for run in (main_run, direction_run):
        p = run.report.p_beats_takeout
        for rule in run.rules:
            z = roi_z(selected(run.data, rule))
            if z > CLEAR_Z:
                assert p[rule[0]] > 0.8, (rule[0], z, p[rule[0]])
                checked["high"].append(rule[0])
            elif z < -CLEAR_Z:
                assert p[rule[0]] < 0.2, (rule[0], z, p[rule[0]])
                checked["low"].append(rule[0])
    assert len(checked["high"]) >= 3 and len(checked["low"]) >= 3, checked


class _FlippedOmega:
    """#78's fitted model with P(omega_s > 1/(1 - t)) reversed; everything else its own."""

    def __init__(self, model) -> None:
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)

    def p_beats_takeout(self, segment):
        return 1.0 - self._model.p_beats_takeout(segment)


def _omega_flipped(*args, **kwargs):
    return _FlippedOmega(fit_segment_roi(*args, **kwargs))


def test_p_beats_takeout_gates_nothing(tmp_path, direction_runners, direction_run):
    """Omega is reported only (owner answer 4). With #78's P(omega_s > 1/(1 - t)) reversed for
    every rule, survivors near certain to beat the takeout become near certain not to, and
    non-survivors the reverse, yet the survivors, the ROI stage and the rungs are unchanged."""
    before = direction_run.report
    p_before = before.p_beats_takeout
    assert any(p_before[rid] > 0.8 for rid in before.survivors), p_before
    assert any(p_before[rid] < 0.2 for rid in p_before if rid not in before.survivors), p_before
    with pytest.MonkeyPatch.context() as patch:
        _replace_everywhere(patch, fit_segment_roi, _omega_flipped)
        run = run_frozen(tmp_path, "test-88-direction", DIRECTION_RULES, direction_runners)
    after = run.report
    for rid, value in p_before.items():
        assert after.p_beats_takeout[rid] == pytest.approx(1.0 - value, abs=1e-9), rid
    assert list(after.survivors) == list(before.survivors)
    assert set(after.stages["roi"]) == set(before.stages["roi"])
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


def test_only_a_survivor_is_given_a_predictive_roi(main_run, omega_run, direction_run, flb_run):
    """ROI is computed for survivors only (DoD), the omega stage's posterior predictive ROI
    included: a non-survivor's omega entry and registry records carry no return of any kind,
    while P(omega_s > 1/(1 - t)) is still reported for every rule. A survivor keeps its
    predictive interval. In the market-bias family nothing survives."""
    seen = {"survivor": [], "other": []}
    for run in (main_run, omega_run, direction_run, flb_run):
        report = run.report
        records = run.registry.records(run.spec.family_id)
        for rid in (rule[0] for rule in run.rules):
            entry = report.stages["omega"][rid]
            p = report.p_beats_takeout[rid]
            assert isinstance(p, float) and 0.0 <= p <= 1.0, (rid, p)
            assert entry["p_beats_takeout"] == pytest.approx(p, abs=1e-12), rid
            if rid in report.survivors:
                low, high, level = entry["predictive_roi_100"]
                assert math.isfinite(low) and math.isfinite(high) and low < high, (rid, entry)
                assert 0.0 < level < 1.0, (rid, entry)
                seen["survivor"].append(rid)
            else:
                assert roi_keys(entry) == [], (rid, sorted(entry))
                metrics = [m.metric for r in records if r.variant == rid for m in r.metrics]
                assert roi_keys(metrics) == [], (rid, sorted(set(metrics)))
                seen["other"].append(rid)
    assert flb_run.report.survivors == []
    assert len(seen["survivor"]) >= 3 and len(seen["other"]) >= 8, seen


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
    return data.assign(always="yes", rider=data["jockey"].str.replace("J", "R", regex=False))


def test_omega_reports_the_return_multiplier_its_selected_bets_identify(
    tmp_path, monkeypatch, omega_runners, omega_run, main_run, direction_run, flb_run
):
    """Every bet of a rule's omega fit is in the rule's segment, so the bets cannot tell a segment
    effect from the odds bands' fixed effects; what they identify is the return multiplier of
    their own mix, `omega_selected`. (1) The same bets under another segment (an extra column all
    runners share, another column name and label) report the same multiplier and the same
    P(omega_s > 1/(1 - t)). (2) The multiplier times (1 - t) is the bets' expected return at the
    final odds, and agrees with their realised return there, sum(won * odds_final) / n_bets (an
    unbiased estimate of it), within MIX_Z standard errors, for every rule of every family."""
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
            rows = selected(run.data, rule)
            assert len(rows) > 50 and "omega_selected" in entry, (rule[0], sorted(entry))
            pay = rows["won"] * rows["odds_final"]
            se = float(pay.std(ddof=1) / math.sqrt(len(pay)))
            expected = entry["omega_selected"] * (1.0 - TAKEOUT)
            assert abs(expected - pay.mean()) <= MIX_Z * se, (rule[0], expected, pay.mean(), se)
            realised[(run.spec.family_id, rule[0])] = float(pay.mean())
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
# DoD: nothing is promoted beyond stake rung S1
# ---------------------------------------------------------------------------------------------


def test_no_rule_is_placed_above_rung_s1(main_run, omega_run):
    """Even J1, which passes every stage with a positive realised return at the final dividend,
    stops at shadow; nothing in the registry is adopted."""
    j1 = selected(omega_run.data, OMEGA_RULES[0])
    assert realised_roi(j1) > 0.2  # premise: an edge a naive ladder would stake on
    for run in (main_run, omega_run):
        assert set(run.report.rungs) == {rule[0] for rule in run.rules}
        assert set(run.report.rungs.values()) <= {"S0", "S1"}
        records = run.registry.records(run.spec.family_id)
        assert all(record.status != "adopted" for record in records)
        assert all(record.owner_signoff is None for record in records)


# ---------------------------------------------------------------------------------------------
# The committed family: the owner's approval on #88
# ---------------------------------------------------------------------------------------------


def _owner_family_frame(data: pd.DataFrame) -> pd.DataFrame:
    """`data` with the committed family's columns: segment A's races are handicaps, B's are
    R12, C's are the most concentrated bin, and rank_t is the min-rank of odds_t in the race."""
    seg = data["seg"]
    return data.assign(
        weight_type=np.where(seg == "A", "handicap", "other"),
        R12=np.where(seg == "B", "in", "out"),
        entropy_t_bin=np.where(seg == "C", 1, 2),
        rank_t=data.groupby("race_id")["odds_t"].rank(method="min").astype(int),
    )


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
    """`load_family("japan")` reads the signed file: Family P's three runnable rules with their
    thresholds and directions, decision time T-2, meeting clusters, the max-statistic settings,
    the calibrated market, the odds-band cut points and the six E4 variants the owner dropped
    (`dropped_variants`, frozen with the spec). Without the sign-off a copy is refused; the family
    freezes as plain JSON and the study runs it, recording each dropped variant as failed, so the
    family size is 3 + 6 = 9 (08 §5.2 step 1)."""
    raw = yaml.safe_load(FAMILY_FILE.read_text(encoding="utf-8"))
    assert raw["owner_signoff"] == 88
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

    unsigned = tmp_path / "selection_family.yaml"
    unsigned.write_text(yaml.safe_dump({**raw, "owner_signoff": None}), encoding="utf-8")
    with pytest.raises(FamilyNotApproved):
        load_family(unsigned)

    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    run = run_spec(tmp_path / "owner", spec, _owner_family_frame(main_runners))
    assert set(run.report.rungs) == set(OWNER_RULES)
    assert set(run.report.rungs.values()) <= {"S0", "S1"}
    records = run.registry.records(spec.family_id)
    assert {record.variant for record in records} == set(OWNER_RULES) | set(dropped)
    status = {(record.variant, record.status) for record in records}
    assert status == {(rid, "scored") for rid in OWNER_RULES} | {(v, "failed") for v in dropped}
    assert run.report.family_size == len(OWNER_RULES) + len(dropped) == 9
    assert run.report.family_size == run.registry.family_size(spec.family_id)
    for rid, value in run.report.p_beats_takeout.items():
        assert math.isfinite(value) and 0.0 <= value <= 1.0, (rid, value)
    # The dropped variants are part of the frozen spec, so the freeze covers them too.
    assert list(getattr(spec, "dropped_variants", None) or []) == dropped
