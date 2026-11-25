"""The crowd-calibration baseline at the final Win price (#77; 08 §8 A3, §4 E1, E3, E4).

Not the D4 spline model (`models.v0.calibration`): this module measures how the final Win price
turns into a probability under three de-vigs and how far a power beta departs from 1.

Inputs are pandas Series on one index; `race_id` labels each runner's race (rows need not be
grouped by race). Odds are decimal, stake included, and go through `core.pricing.clean_odds`: a
missing, non-finite or sub-1.0 figure is an unpriced runner, dropped before any de-vig, so it takes
no share (NaN) and each race's priced runners sum to 1.

- normalised: q_i = (1/O_i) / sum_j (1/O_j);
- power: p_i proportional to (1/O_i)^beta;
- Shin (1993): with pi_i = 1/O_i and Pi = sum_j pi_j, p_i = (sqrt(z^2 + 4(1 - z) pi_i^2 / Pi) - z)
  / (2(1 - z)), the insider share z in [0, 1) solved per race so the race sums to 1.

Memory: everything works on flat arrays (np.bincount and np.add.reduceat over race segments);
Shin's z is one vectorised bisection over every race at once, and the beta fit is a Newton
iteration whose steps each cost a few passes over the rows. Nothing is quadratic.

The study (`run_study`; CLI `python -m horsetrack.countries.japan.studies.calibration --run
[--out DIR]`, an `--out` inside the repository refused by `paths.outside_repo`) is a thin I/O layer
over `compute_study(frame, spec)`, which is pure. Its pre-registered inputs are the owner's, in
`countries/japan/config/calibration.yaml` (no defaults; a missing file or key raises
`MissingOwnerDecision` before the store is read, and an unknown key is refused):

    segments: [R12, venue, ...]        # E4: names from SEGMENTS, each fitted in vs out
    entropy_cut_points: [a, b, c, d]   # E3: increasing, in (0, 1); five bins, the last closed
    tail_min_odds: 51.0                # the tail is a runner priced above this (50/1 = 51.0)
    owner_signoff: <issue>             # where the owner signed these answers

The cut points follow the owner's rule, ENTROPY_CUT_RULE (quintiles of the first training
window's per-race entropy, from the final Win odds alone); `--entropy-cut-points` prints them
(`first_window_cut_points`, over the pure `entropy_cut_points`), and every run recomputes them
and refuses a yaml that disagrees, so the rule and the numbers frozen in the spec agree.

The floor is the Win pool's minimum dividend over the unit stake (`jra.yaml`, 110 / 100 = 1.1);
the bootstrap's unit, level and replicates come from `config/scoring.yaml` (D7) and the folds from
`config/folds.yaml` (D9). Order: read the owner's inputs, check the cut points (prices only),
freeze the family's spec in the run registry (`FamilyAlreadyFrozen` on another spec stops the
run), read the development rows of the price era (`read_store_table`: column- and date-scoped,
holdout days dropped, nothing before 2014-06-07; then `split_off`'s development rows; the holdout
is never opened), score, append one registry record per variant (failures included) and write the
report through `write_report`.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml, load_source
from horsetrack.core.eval.bootstrap import Interval, paired_bootstrap
from horsetrack.core.eval.folds import Fold, FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha, split_off
from horsetrack.core.eval.registry import FoldMetric, RunRecord, default_registry
from horsetrack.core.eval.reports import write_report
from horsetrack.core.eval.scoring import gain, per_race_log_loss
from horsetrack.core.pricing import _EPS as FLOOR_TOLERANCE
from horsetrack.core.pricing import clean_odds
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table

SHIN_BISECTIONS = 60  # halvings of [0, 1]: z to below 1e-17
NEWTON_MAX_STEPS = 50
NEWTON_STEP_TOL = 1e-10


@dataclass(frozen=True)
class BetaFit:
    """The power de-vig's beta (maximum likelihood), its standard error and the races fitted."""

    beta: float
    se: float
    n_races: int


# --------------------------------------------------------------------------- inputs


def _check_aligned(first: pd.Series, *others: pd.Series) -> None:
    for other in others:
        if not other.index.equals(first.index):
            raise ValueError(f"{other.name or 'a Series'} is not on the odds' index")


def _race_codes(race_id: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """0..n-1 race codes per row, in first-appearance order, and the race ids they stand for."""
    codes, uniques = pd.factorize(race_id.to_numpy())
    if (codes < 0).any():
        raise ValueError("race_id has missing values; every runner needs its race")
    return codes, np.asarray(uniques)


def _priced(
    odds: pd.Series, race_id: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cleaned odds, the priced mask, race codes and the race ids the codes stand for."""
    _check_aligned(odds, race_id)
    cleaned = clean_odds(odds.to_numpy())
    codes, races = _race_codes(race_id)
    return cleaned, ~np.isnan(cleaned), codes, races


def _on_index(values: np.ndarray, ok: np.ndarray, index: pd.Index) -> pd.Series:
    out = np.full(len(ok), np.nan)
    out[ok] = values
    return pd.Series(out, index=index, dtype=float)


# --------------------------------------------------------------------------- de-vigs


def devig_power(odds: pd.Series, race_id: pd.Series, beta: float) -> pd.Series:
    """p_i proportional to (1/O_i)^beta over each race's priced runners; NaN where unpriced."""
    cleaned, ok, codes, races = _priced(odds, race_id)
    weight = (1.0 / cleaned[ok]) ** float(beta)
    totals = np.bincount(codes[ok], weights=weight, minlength=len(races))
    return _on_index(weight / totals[codes[ok]], ok, odds.index)


def devig_normalised(odds: pd.Series, race_id: pd.Series) -> pd.Series:
    """q_i = (1/O_i) / sum_j (1/O_j) over each race's priced runners; NaN where unpriced."""
    return devig_power(odds, race_id, 1.0)


def _shin_p(c: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Shin's p_i for c_i = pi_i^2 / Pi, written without the 1 / (1 - z) that cancels badly near
    z = 1: (sqrt(z^2 + 4(1 - z)c) - z) / (2(1 - z)) = 2c / (sqrt(z^2 + 4(1 - z)c) + z)."""
    return 2.0 * c / (np.sqrt(z * z + 4.0 * (1.0 - z) * c) + z)


def devig_shin(odds: pd.Series, race_id: pd.Series) -> pd.Series:
    """Shin (1993) probabilities over each race's priced runners; NaN where unpriced.

    The race's sum of p_i falls strictly in z (each c_i <= 1), from sqrt(Pi) at z = 0 to
    sum c_i < 1 at z = 1, so a book with overround (Pi > 1) has one root in (0, 1), found by
    bisection over every race at once. A book without overround (Pi <= 1) has no root in [0, 1):
    z is clamped to 0 and the shares normalised, which is the normalised de-vig (exactly 1/O_i
    when Pi = 1).
    """
    cleaned, ok, codes, races = _priced(odds, race_id)
    codes, n_races = codes[ok], len(races)
    pi = 1.0 / cleaned[ok]
    book = np.bincount(codes, weights=pi, minlength=n_races)
    c = pi * pi / book[codes]

    lo, hi = np.zeros(n_races), np.ones(n_races)
    for _ in range(SHIN_BISECTIONS):
        mid = 0.5 * (lo + hi)
        above = np.bincount(codes, weights=_shin_p(c, mid[codes]), minlength=n_races) > 1.0
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    z = 0.5 * (lo + hi)

    over = (book > 1.0)[codes]
    p = np.where(over, _shin_p(c, z[codes]), pi / book[codes])
    return _on_index(p, ok, odds.index)


# --------------------------------------------------------------------------- the beta fit


@dataclass(frozen=True)
class _ChoiceSets:
    """Priced runners of the races with a priced winner, sorted by race."""

    x: np.ndarray  # ln(1 / O)
    starts: np.ndarray  # each race's first row
    sizes: np.ndarray  # each race's priced runners
    winners: np.ndarray  # each race's priced winners (2+ on a dead heat)
    x_won: float  # sum of x over every priced winner


def _choice_sets(odds: pd.Series, race_id: pd.Series, won: pd.Series) -> _ChoiceSets:
    _check_aligned(odds, race_id, won)
    if won.isna().any():
        raise ValueError("won has missing values")
    cleaned, ok, codes, races = _priced(odds, race_id)
    won_priced = won.to_numpy(dtype=bool) & ok
    winners = np.bincount(codes, weights=won_priced, minlength=len(races))
    rows = np.flatnonzero(ok & (winners[codes] > 0))
    if len(rows) == 0:
        raise ValueError("no race with a priced winner to fit")
    rows = rows[np.argsort(codes[rows], kind="stable")]
    race = codes[rows]
    starts = np.flatnonzero(np.r_[True, race[1:] != race[:-1]])
    x = -np.log(cleaned[rows])
    return _ChoiceSets(
        x=x,
        starts=starts,
        sizes=np.diff(np.r_[starts, len(rows)]),
        winners=winners[race[starts]],
        x_won=float(x[won_priced[rows]].sum()),
    )


def _loglik(beta: float, sets: _ChoiceSets) -> tuple[float, float, float]:
    """The conditional-logit log-likelihood, its score and the observed information at beta."""
    s = beta * sets.x
    top = np.maximum.reduceat(s, sets.starts)
    e = np.exp(s - np.repeat(top, sets.sizes))
    total = np.add.reduceat(e, sets.starts)
    mean = np.add.reduceat(e * sets.x, sets.starts) / total
    spread = sets.x - np.repeat(mean, sets.sizes)
    var = np.add.reduceat(e * spread * spread, sets.starts) / total
    w = sets.winners
    loglik = beta * sets.x_won - float((w * (np.log(total) + top)).sum())
    return loglik, sets.x_won - float((w * mean).sum()), float((w * var).sum())


def fit_power_beta(odds: pd.Series, race_id: pd.Series, won: pd.Series) -> BetaFit:
    """The maximum-likelihood beta of P(i wins) proportional to exp(beta * ln(1/O_i)) over each
    race's priced runners, a conditional logit on ln(1/odds).

    Only races with a priced winner enter, and `n_races` counts them; unpriced runners leave
    their race's choice set. A dead heat counts each priced co-winner as a winner of its race
    (as #114's fit does): the race adds one choice from its runners per co-winner. The
    log-likelihood is concave in beta; Newton steps, halved whenever one would lower it, run
    until a step is under 1e-10. `se` is the observed information at beta-hat to the power -1/2.

    Raises ValueError when no race has a priced winner or no race's runners differ in price
    (beta is not identified), and RuntimeError when Newton does not converge (no finite maximum,
    for example when every race went to its favourite).
    """
    sets = _choice_sets(odds, race_id, won)
    beta = 1.0
    loglik, score, info = _loglik(beta, sets)
    for _ in range(NEWTON_MAX_STEPS):
        if not info > 0:
            raise ValueError("beta is not identified: no fitted race's runners differ in price")
        step = score / info
        while True:
            trial = _loglik(beta + step, sets)
            if trial[0] >= loglik or abs(step) < NEWTON_STEP_TOL:
                break
            step /= 2.0
        beta += step
        loglik, score, info = trial
        if abs(step) < NEWTON_STEP_TOL:
            break
    else:
        raise RuntimeError(f"the beta fit did not converge in {NEWTON_MAX_STEPS} Newton steps")
    if not (np.isfinite(beta) and info > 0):
        raise RuntimeError(f"the beta fit stopped at beta={beta}, information={info}")
    return BetaFit(beta=float(beta), se=float(info**-0.5), n_races=len(sets.starts))


def segment_beta(frame: pd.DataFrame, segment: str) -> pd.DataFrame:
    """The beta fit on each value's races of `frame[segment]` (constant within a race): one row
    per value, in sorted order, with columns `segment`, beta, se and n_races. `frame` has
    race_id, odds and won; other columns are ignored, and rows with a missing segment value are
    left out."""
    varies = frame.groupby("race_id", observed=True)[segment].nunique(dropna=False) > 1
    if varies.any():
        raise ValueError(f"{segment!r} varies within {int(varies.sum())} race(s)")
    rows = []
    for value, part in frame.groupby(segment, sort=True, observed=True):
        fit = fit_power_beta(part["odds"], part["race_id"], part["won"])
        rows.append({segment: value, "beta": fit.beta, "se": fit.se, "n_races": fit.n_races})
    return pd.DataFrame(rows, columns=[segment, "beta", "se", "n_races"])


# --------------------------------------------------------------------------- entropy, floor


def normalised_entropy(odds: pd.Series, race_id: pd.Series) -> pd.Series:
    """Per race, indexed by race id (first-appearance order): the Shannon entropy of its
    normalised probabilities over its n priced runners divided by ln n, clipped to [0, 1]
    against rounding. NaN for a race with fewer than two priced runners (ln n = 0)."""
    cleaned, ok, codes, races = _priced(odds, race_id)
    codes = codes[ok]
    inv = 1.0 / cleaned[ok]
    q = inv / np.bincount(codes, weights=inv, minlength=len(races))[codes]
    entropy = np.bincount(codes, weights=-q * np.log(q), minlength=len(races))
    n = np.bincount(codes, minlength=len(races))
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where(n >= 2, np.clip(entropy / np.log(n), 0.0, 1.0), np.nan)
    return pd.Series(value, index=pd.Index(races, name="race_id"), name="entropy")


def flag_floor_bound(odds: pd.Series, *, floor: float) -> pd.Series:
    """True for a priced runner at or under `floor` (within `core.pricing.floor_bound`'s 1e-9):
    its real pool share is unknown. An unpriced runner is never floor-bound."""
    cleaned = clean_odds(odds.to_numpy())
    with np.errstate(invalid="ignore"):
        flagged = ~np.isnan(cleaned) & (cleaned <= floor + FLOOR_TOLERANCE)
    return pd.Series(flagged, index=odds.index, dtype=bool)


def entropy_cut_points(
    odds: pd.Series, race_id: pd.Series, *, quantiles: Sequence[float]
) -> list[float]:
    """The `quantiles` of the per-race normalised entropy of `odds` (E3's cut points), races with
    fewer than two priced runners left out; numpy's linear interpolation. Prices only: no outcome
    enters. Raises ValueError when no race has two priced runners."""
    entropy = normalised_entropy(odds, race_id).dropna().to_numpy()
    if len(entropy) == 0:
        raise ValueError("no race has two or more priced runners")
    return [float(c) for c in np.quantile(entropy, list(quantiles))]


# =========================================================================== the study

COUNTRY = "japan"
TICKET = 77
SOURCE = "jra"  # the store's JRA results, from netkeiba; no JRDB, no JRA-VAN
FAMILY_ID = "japan-crowd-calibration-final-win"
CONFIG_NAME = "calibration"
BOOTSTRAP_SEED = 77  # reproducibility only; scoring.yaml signs no seed
BETA_CI_Z = 1.959963984540054  # the ticket's 95% two-sided Wald interval, beta +/- z se
FIT = "conditional_logit_ln_inv_odds"
IN_SAMPLE = "in_sample:development"
RUNNER_COLUMNS = ["race_id", "race_date", "win_odds", "finish_pos", "is_scratched"]
RACE_COLUMNS = ["race_id", "race_date", "meeting_id", "venue", "race_no", "race_class"]
RACE_COLUMNS += ["surface", "going", "weight_type", "field_size", "status"]
DEVIGS = ("normalised", "power", "shin")  # normalised first: the others' comparator
PARAMETER_FREE = {"normalised": devig_normalised, "shin": devig_shin}
SUBSETS = ("all_races", "no_floor_bound_races")
OWNER_KEYS = {
    "segments": "the E4 segment family (08 §4): which named segments to fit",
    "entropy_cut_points": "the E3 entropy cut points: four increasing numbers in (0, 1)",
    "tail_min_odds": "the tail threshold in decimal odds (the ticket's 'above 50/1')",
    "owner_signoff": "the issue where the owner signed these answers",
}
# E3's cut-point rule (the owner's answer on #77): quintiles of the per-race normalised entropy
# over the development races of D9's first training window, from the final Win odds alone.
ENTROPY_WINDOW = (PRICE_ERA_START, date(2019, 12, 31))
ENTROPY_QUANTILES = (0.2, 0.4, 0.6, 0.8)
CUT_POINT_COLUMNS = ["race_id", "race_date", "win_odds", "is_scratched"]  # no outcome column
ENTROPY_CUT_RULE = {
    "rule": "quintiles of the per-race normalised entropy of the final Win odds "
    "(numpy.quantile, linear interpolation), races with two or more priced starters",
    "window": [d.isoformat() for d in ENTROPY_WINDOW],
    "quantiles": list(ENTROPY_QUANTILES),
    "rows": "priced starters of the development race days in the window; columns read: "
    + ", ".join(CUT_POINT_COLUMNS),
    "computed_by": "python -m horsetrack.countries.japan.studies.calibration --entropy-cut-points",
}
CUT_POINT_TOLERANCE = 1e-12  # the yaml's cut points against the rule's, recomputed at each run
DECISIONS = {
    "D4": "the Win baseline is the calibrated market (spline in ln q), fitted in-fold; this study "
    "measures the de-vigs beneath it and adopts none (section 2)",
    "D6": "winner log loss per race, a dead heat at 1/k per co-winner (section 3; #69 answer)",
    "D7": "meeting (venue x day) clusters, 95% one-sided (config/scoring.yaml)",
    "D9": "walk-forward quarters, expanding, 1-week embargo (config/folds.yaml; #70 answer)",
    "D10": "races before 2014-06-07 excluded wherever the price enters (section 4)",
    "D11": "Japan holdout A2: a seeded half of race days from the start (config/holdout.yaml, #71)",
    "D12": "holdout start 2026-11-28, opened only by the owner at a power target (section 5)",
    "D14": "holdout days fully locked, outcomes and prices: no holdout row is read (section 5)",
    "D15": "a holdout opening is the owner's, once per family: the holdout is not opened here",
    "#77": "the owner's answers of 2026-11-25: the full E4 list; E3 quintiles of the first "
    "training window; the era table post-reform only; floor-bound races out of fit and score in "
    "the no-floor-bound subset; the tail above decimal 51.0",
}


class MissingOwnerDecision(RuntimeError):
    """A pre-registered input the owner has not signed in config/calibration.yaml."""


# --------------------------------------------------------------------------- E4 segments


@dataclass(frozen=True)
class Segment:
    """A named E4 candidate: its definition (frozen in the spec) and its value per race, from
    one row per race of the study frame; missing where the attribute it reads is missing."""

    definition: str
    values: Callable[[pd.DataFrame], pd.Series]


def _in_out(read: pd.Series, inside: pd.Series) -> pd.Series:
    return pd.Series(np.where(inside, "in", "out"), index=read.index, dtype=object).where(
        read.notna()
    )


def _g1_day_undercard(races: pd.DataFrame) -> pd.Series:
    flat_g1 = races["race_class"].eq("G1") & races["surface"].ne("jump")
    g1_meeting = flat_g1.groupby(races["meeting_id"]).transform("any")
    return _in_out(races["meeting_id"], g1_meeting & ~flat_g1).where(~flat_g1)


SEGMENTS = {
    "R12": Segment(
        "in: race_no 12; out: every other race",
        lambda r: _in_out(r["race_no"], r["race_no"].eq(12)),
    ),
    "R11pm1": Segment(
        "in: race_no 10, 11 or 12 (R11 +/- 1); out: every other race",
        lambda r: _in_out(r["race_no"], r["race_no"].isin([10, 11, 12])),
    ),
    "g1_day_undercard": Segment(
        "in: the other races of a meeting (venue x day) that holds a flat G1; out: the races "
        "of meetings without one; the G1 itself is in neither",
        _g1_day_undercard,
    ),
    "field_16_plus": Segment(
        "in: field_size (starters) 16 or more; out: fewer",
        lambda r: _in_out(r["field_size"], r["field_size"].ge(16)),
    ),
    "debut": Segment(
        "in: race_class DEBUT (新馬); out: every other class",
        lambda r: _in_out(r["race_class"], r["race_class"].eq("DEBUT")),
    ),
    "venue": Segment("one beta per venue code", lambda r: r["venue"]),
    "heavy_going": Segment(
        "in: going HEAVY (不良); out: any other going",
        lambda r: _in_out(r["going"], r["going"].eq("HEAVY")),
    ),
    "handicap": Segment(
        "in: weight_type handicap (ハンデ); out: any other weight type",
        lambda r: _in_out(r["weight_type"], r["weight_type"].eq("handicap")),
    ),
}


# --------------------------------------------------------------------------- inputs and spec


def owner_inputs() -> dict:
    """The owner's pre-registered inputs from config/calibration.yaml, validated."""
    where = f"countries/{COUNTRY}/config/{CONFIG_NAME}.yaml"
    try:
        raw, problem = load_country_yaml(COUNTRY, CONFIG_NAME), "lacks"
    except FileNotFoundError:
        raw, problem = {}, "does not exist; it carries"
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping of {sorted(OWNER_KEYS)}")
    missing = [key for key in OWNER_KEYS if key not in raw]
    if missing:
        needed = "; ".join(f"{key}: {OWNER_KEYS[key]}" for key in missing)
        raise MissingOwnerDecision(f"{where} {problem} the owner decision(s) {needed}")
    unknown = sorted(set(raw) - set(OWNER_KEYS))
    if unknown:
        raise ValueError(f"{where} has unknown key(s): {', '.join(unknown)}")

    segments, cuts = raw["segments"], raw["entropy_cut_points"]
    if not (isinstance(segments, list) and segments and len(set(segments)) == len(segments)):
        raise ValueError(f"segments must be a non-empty list of distinct names, got {segments!r}")
    strange = [s for s in segments if s not in SEGMENTS]
    if strange:
        raise ValueError(f"unknown segment(s) {strange}; the candidates are {list(SEGMENTS)}")
    if not (
        isinstance(cuts, list)
        and len(cuts) == 4
        and all(_real(c) and 0.0 < c < 1.0 for c in cuts)
        and bool(np.all(np.diff(cuts) > 0))
    ):
        raise ValueError(f"entropy_cut_points must be four increasing numbers in (0, 1): {cuts!r}")
    tail, signoff = raw["tail_min_odds"], raw["owner_signoff"]
    if not (_real(tail) and tail > 1.0):
        raise ValueError(f"tail_min_odds must be decimal odds above 1.0, got {tail!r}")
    if not (isinstance(signoff, int) and not isinstance(signoff, bool) and signoff > 0):
        raise ValueError(f"owner_signoff must be a positive issue number, got {signoff!r}")
    return {
        "segments": list(segments),
        "entropy_cut_points": [float(c) for c in cuts],
        "tail_min_odds": float(tail),
        "owner_signoff": signoff,
    }


def _real(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def bootstrap_settings() -> dict:
    """D7's bootstrap from config/scoring.yaml, with the study's fixed seed."""
    boot = load_country_yaml(COUNTRY, "scoring")["bootstrap"]
    if boot["cluster_unit"] != "meeting" or boot["sided"] != "one-sided":
        raise ValueError(f"this study implements meeting clusters, one-sided; scoring.yaml: {boot}")
    return {
        "cluster_unit": "meeting",
        "sided": "one-sided",
        "level": float(boot["level"]),
        "n_boot": int(boot["n_boot"]),
        "seed": BOOTSTRAP_SEED,
    }


def win_floor() -> float:
    """The Win pool's floor in decimal odds: its minimum dividend over the unit stake."""
    jra = load_source(COUNTRY, SOURCE)
    return float(jra.pools["WIN"].min_dividend) / jra.unit_stake


def build_spec(owner: dict, scheme: FoldScheme, bootstrap: dict, floor: float) -> dict:
    """The family's pre-registered spec, frozen in the registry before anything is scored."""
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypotheses": {
            "E1": "the power beta of the final Win price by year and era; each de-vig's ΔLL "
            "against normalised and its tail calibration (observed vs expected winners)",
            "E3": "the power beta by bin of the race's normalised market entropy",
            "E4": "the power beta in vs out of each pre-registered segment",
        },
        "target": "the race's winner (Win pool) at the final Win odds",
        "devigs": {
            "normalised": "q_i = (1/O_i) / sum_j (1/O_j)",
            "power": "p_i proportional to (1/O_i)^beta, beta fitted on each fold's training rows",
            "shin": "Shin (1993), z solved per race; clamped to 0 (normalised) without overround",
        },
        "comparator": "normalised",
        "delta_ll": "per scored race, normalised's winner log loss minus the de-vig's "
        "(core.eval.scoring.gain), out of fold; meeting-clustered paired bootstrap bounds",
        "beta_fit": "conditional logit on ln(1/O), MLE; se = observed information^-1/2; "
        "races with a priced winner; a dead heat counts each co-winner",
        "beta_ci": "95% two-sided Wald, beta +/- 1.96 se",
        "in_sample": "beta by year, era, E4 segment and E3 entropy bin: one fit per group on "
        "every development race of the subset, descriptive, no fold",
        "adoption": "none: the owner picks the baseline de-vig; judged by the beta CI, not by "
        "ΔLL (08 §4 E1)",
        "score": "winner log loss per race (core.eval.scoring.per_race_log_loss)",
        "rows": "priced starters of run races with a priced winner; development days only",
        "subsets": {
            "all_races": "every development race",
            "no_floor_bound_races": "races holding a floor-bound runner left out of fitting and "
            "scoring alike (E1 de-vigs, tail, beta by year and era)",
        },
        "segments": {name: SEGMENTS[name].definition for name in owner["segments"]},
        "entropy_cut_points": owner["entropy_cut_points"],
        "entropy_cut_rule": ENTROPY_CUT_RULE,
        "tail": {"min_odds": owner["tail_min_odds"], "rule": "a runner priced above min_odds"},
        "floor": {"odds": floor, "rule": "a priced runner at or under odds + 1e-9 (jra.yaml)"},
        "fold_scheme": dataclasses.asdict(scheme),
        "bootstrap": bootstrap,
        "price_era_start": PRICE_ERA_START.isoformat(),
        "decisions": DECISIONS,
        "owner_signoff": owner["owner_signoff"],
        "sources": [SOURCE],
    }


# --------------------------------------------------------------------------- development rows


def read_development() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Development runners and races of the price era: column- and date-scoped store reads that
    drop holdout days, then `split_off`'s development rows. The holdout is never opened."""
    frames = []
    for name, columns in (("runners", RUNNER_COLUMNS), ("races", RACE_COLUMNS)):
        table = read_store_table(
            name, columns=columns, start=PRICE_ERA_START, end=date.max, touches_price=True
        )
        development, _ = split_off(table.to_pandas(date_as_object=False), "race_date", COUNTRY)
        frames.append(development)
    return frames[0], frames[1]


def first_window_cut_points() -> dict:
    """E3's cut points by ENTROPY_CUT_RULE: the development race days of ENTROPY_WINDOW, read
    column-scoped (prices and the scratch flag; no outcome column), holdout days dropped."""
    start, end = ENTROPY_WINDOW
    table = read_store_table(
        "runners", columns=CUT_POINT_COLUMNS, start=start, end=end, touches_price=True
    )
    runners, _ = split_off(table.to_pandas(date_as_object=False), "race_date", COUNTRY)
    starters = runners.loc[~runners["is_scratched"].eq(True).to_numpy(dtype=bool, na_value=False)]
    odds = starters["win_odds"].astype(float)
    entropy = normalised_entropy(odds, starters["race_id"])
    return {
        "entropy_cut_points": entropy_cut_points(
            odds, starters["race_id"], quantiles=ENTROPY_QUANTILES
        ),
        "races": int(entropy.notna().sum()),
        "race_days": int(starters["race_date"].nunique()),
        "first_race_day": str(starters["race_date"].min().date()),
        "last_race_day": str(starters["race_date"].max().date()),
        "rule": ENTROPY_CUT_RULE,
    }


def check_cut_points(cuts: Sequence[float]) -> None:
    """Refuse cut points in config/calibration.yaml that are not ENTROPY_CUT_RULE's on the store,
    so the frozen spec's rule and numbers agree."""
    expected = first_window_cut_points()["entropy_cut_points"]
    if not np.allclose(cuts, expected, rtol=0.0, atol=CUT_POINT_TOLERANCE):
        raise ValueError(
            f"entropy_cut_points {list(cuts)} are not the rule's {expected} "
            f"({ENTROPY_CUT_RULE['computed_by']})"
        )


def store_builds() -> dict[str, str]:
    """Each store table read, as '<path>@<sha256 prefix>' of its file."""
    builds = {}
    for name in ("runners", "races"):
        digest = hashlib.sha256()
        with open(paths.processed_dir(SOURCE) / f"{name}.parquet", "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        builds[f"{SOURCE}_{name}"] = f"processed/{SOURCE}/{name}.parquet@{digest.hexdigest()[:16]}"
    return builds


def development_frame(runners: pd.DataFrame, races: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The study frame: one row per priced starter of a run race with a priced winner (race_id,
    race_date, odds cleaned, won = finish_pos 1, each co-winner of a dead heat, and the race's
    RACE_COLUMNS), with the races and runners each rule dropped."""
    void = races["status"].ne("run")
    known = runners["race_id"].isin(races["race_id"])
    in_void = runners["race_id"].isin(races.loc[void, "race_id"])
    kept = runners.loc[known & ~in_void]
    scratched = kept["is_scratched"].eq(True)
    odds = pd.Series(clean_odds(kept["win_odds"].to_numpy()), index=kept.index)
    unpriced = ~scratched & odds.isna()
    starting = ~scratched & ~unpriced
    placing = kept.loc[starting, "finish_pos"]  # nullable: missing for a starter not placed
    starters = kept.loc[starting, ["race_id", "race_date"]].assign(
        odds=odds[starting], won=placing.eq(1).fillna(False).astype(bool)
    )
    has_winner = starters.groupby("race_id")["won"].transform("any")
    attributes = races.loc[~void].drop(columns=["race_date", "status"])
    frame = starters.loc[has_winner, ["race_id", "race_date", "odds", "won"]].merge(
        attributes, on="race_id", how="left", validate="many_to_one"
    )
    winners = frame.groupby("race_id")["won"].sum()
    counts = {
        "races_read": len(races),
        "void_races": int(void.sum()),
        "runners_read": len(runners),
        "runners_without_a_race_row": int((~known).sum()),
        "runners_in_void_races": int(in_void.sum()),
        "scratched_runners": int(scratched.sum()),
        "unpriced_runners": int(unpriced.sum()),
        "starters_not_placed": int(placing.isna().sum()),
        "races_without_a_priced_winner": int(kept["race_id"].nunique() - len(winners)),
        "dead_heat_races": int((winners >= 2).sum()),
        "races": len(winners),
        "runners": len(frame),
    }
    return frame, counts


# --------------------------------------------------------------------------- the pure study


@dataclass(frozen=True)
class Variant:
    """One registry record's identity, its metrics, and its error when it failed."""

    hypothesis_id: str
    name: str
    fold_scheme: str
    components: dict[str, str]
    params: dict
    metrics: tuple[FoldMetric, ...] = ()
    error: str | None = None


@dataclass
class Study:
    counts: dict[str, int]
    tables: dict[str, list[dict]] = field(default_factory=dict)
    variants: list[Variant] = field(default_factory=list)

    def attempt(self, variant: Variant, compute: Callable[[], tuple[list, dict]]) -> None:
        """Run one variant: its metrics and table rows, or its error as a failed variant."""
        try:
            metrics, tables = compute()
        except Exception as exc:  # every failure is a registry record, never dropped
            self.variants.append(replace(variant, error=f"{type(exc).__name__}: {exc}"))
            return
        self.variants.append(replace(variant, metrics=tuple(metrics)))
        for key, rows in tables.items():
            self.tables.setdefault(key, []).extend(rows)


def _metric(fold: str, metric: str, value: float, n: int, lo=None, hi=None) -> FoldMetric:
    """A registry metric in plain floats and ints; a non-finite value fails its variant."""
    value, lo, hi = (None if v is None else float(v) for v in (value, lo, hi))
    if not all(np.isfinite(v) for v in (value, lo, hi) if v is not None):
        raise ValueError(f"{metric} on {fold} is not finite: {value}, [{lo}, {hi}]")
    return FoldMetric(str(fold), metric, value, lo, hi, int(n))


def _bounds(per_race: pd.Series, clusters: pd.Series, spec: dict) -> Interval:
    boot = spec["bootstrap"]
    return paired_bootstrap(
        per_race, clusters, n_boot=boot["n_boot"], level=boot["level"], seed=boot["seed"]
    )


def _beta_table(table: pd.DataFrame, column: str, header: str, **label) -> tuple[list, list]:
    """Registry metrics and report rows of a segment_beta table, with 95% CIs."""
    metrics, rows = [], []
    for value, beta, se, n in table[[column, "beta", "se", "n_races"]].itertuples(index=False):
        lo, hi = beta - BETA_CI_Z * se, beta + BETA_CI_Z * se
        metrics.append(_metric("development", f"beta[{column}={value}]", beta, n, lo, hi))
        rows.append(
            {**label, header: value, "races": int(n), "beta": beta, "se": se}
            | {"lower 95%": lo, "upper 95%": hi}
        )
    return metrics, rows


def _walk_forward_power(fit_rows: pd.DataFrame, folds: list[Fold], subset: str):
    """Out-of-fold power probabilities, each fold's beta from its training rows."""
    parts, metrics, rows = [], [], []
    for fold in folds:
        train, test = fit_rows.iloc[fold.train_idx], fit_rows.iloc[fold.test_idx]
        fit = fit_power_beta(train["odds"], train["race_id"], train["won"])
        parts.append(devig_power(test["odds"], test["race_id"], fit.beta))
        lo, hi = fit.beta - BETA_CI_Z * fit.se, fit.beta + BETA_CI_Z * fit.se
        label = fold.test_start.isoformat()
        metrics.append(_metric(label, "beta", fit.beta, fit.n_races, lo, hi))
        rows.append(
            {"subset": subset, "fold": label, "train to": fold.train_end, "test to": fold.test_end}
            | {"train races": fit.n_races, "beta": fit.beta, "se": fit.se}
            | {"test races": test["race_id"].nunique()}
        )
    return pd.concat(parts), metrics, rows


def _devig_variants(frame: pd.DataFrame, spec: dict, subset: str, study: Study) -> None:
    """E1 on one subset: each de-vig's walk-forward winner log loss, ΔLL against normalised and
    the tail's observed vs expected winners, with meeting-clustered bounds."""
    folds = walk_forward(frame["race_date"], scheme=FoldScheme.from_mapping(spec["fold_scheme"]))
    scored = frame.iloc[np.concatenate([f.test_idx for f in folds] or [np.zeros(0, int)])]
    race = scored["race_id"].to_numpy()
    sizes = [len(f.test_idx) for f in folds]
    fold_of = pd.Series(np.repeat([f.test_start.isoformat() for f in folds], sizes), index=race)
    fold_of = fold_of.groupby(level=0).first()
    clusters = scored["meeting_id"].groupby(race).first()
    tail = (scored["odds"] > spec["tail"]["min_odds"]).to_numpy()
    won = scored["won"].to_numpy(dtype=float)
    comparator: dict[str, pd.Series] = {}

    def score(name: str) -> tuple[list, dict]:
        if not folds:
            raise ValueError("the fold scheme gives no walk-forward fold on these dates")
        if name != "normalised" and "ll" not in comparator:
            raise RuntimeError("the normalised comparator failed, so there is no ΔLL")
        metrics, tables = [], {}
        if name in PARAMETER_FREE:
            p = PARAMETER_FREE[name](scored["odds"], scored["race_id"])
        else:
            fit_rows = frame[["race_id", "odds", "won"]]
            p, metrics, tables["folds"] = _walk_forward_power(fit_rows, folds, subset)
        no = pd.Series(False, index=scored.index)
        ll = per_race_log_loss(p, scored["won"], scored["race_id"], no).per_race
        for label, part in ll.groupby(fold_of.reindex(ll.index).to_numpy()):
            metrics.append(_metric(label, "mean_log_loss", part.mean(), len(part)))
        metrics.append(_metric("all_folds", "mean_log_loss", ll.mean(), len(ll)))
        row = {"subset": subset, "de-vig": name, "races": len(ll), "mean log loss": ll.mean()}
        if name == "normalised":
            comparator["ll"] = ll
        else:
            g = gain(ll, comparator["ll"])
            iv = _bounds(g.per_race, clusters, spec)
            metrics.append(
                _metric("all_folds", "delta_ll_vs_normalised", iv.mean, g.n_common, iv.lo, iv.hi)
            )
            row |= {
                "ΔLL vs normalised": iv.mean,
                "lower": iv.lo,
                "upper": iv.hi,
                "meetings": iv.n_clusters,
            }
        gap = pd.Series(np.where(tail, won - p.to_numpy(), 0.0)).groupby(race).sum()
        iv = _bounds(gap, clusters, spec)
        observed, expected = won[tail].sum(), p.to_numpy()[tail].sum()
        metrics += [
            _metric("all_folds", "tail_observed_winners", observed, tail.sum()),
            _metric("all_folds", "tail_expected_winners", expected, tail.sum()),
            _metric("all_folds", "tail_o_minus_e_per_race", iv.mean, len(gap), iv.lo, iv.hi),
        ]
        tables["headline"] = [row]
        tables["tail"] = [
            {"subset": subset, "de-vig": name, "tail runners": int(tail.sum())}
            | {"observed winners": observed, "expected winners": expected}
            | {"O / E": observed / expected if expected > 0 else None}
            | {"(O - E) per race": iv.mean, "lower": iv.lo, "upper": iv.hi}
        ]
        return metrics, tables

    scheme = "walk_forward:" + json.dumps(spec["fold_scheme"], sort_keys=True)
    for name in DEVIGS:
        components = {"devig": name, "score": "winner_log_loss", "comparator": "normalised"}
        variant = Variant("E1", f"{name}:{subset}", scheme, components, {"subset": subset})
        study.attempt(variant, lambda name=name: score(name))


def _year_and_era(frame: pd.DataFrame, subset: str) -> tuple[list, dict]:
    fit_rows = frame[["race_id", "odds", "won"]].assign(year=frame["race_date"].dt.year)
    metrics, by_year = _beta_table(segment_beta(fit_rows, "year"), "year", "year", subset=subset)
    era = segment_beta(fit_rows.assign(era=f"{PRICE_ERA_START.isoformat()} onward"), "era")
    era_metrics, by_era = _beta_table(era, "era", "era", subset=subset)
    return metrics + era_metrics, {"beta_by_year": by_year, "beta_by_era": by_era}


def _compared(table: pd.DataFrame, column: str) -> pd.DataFrame:
    """The table, when it has two or more values to compare; otherwise the variant fails."""
    if len(table) < 2:
        raise ValueError(f"{column} has {table[column].tolist()} only: nothing to compare")
    return table


def _segment(frame: pd.DataFrame, races: pd.DataFrame, name: str) -> tuple[list, dict]:
    values = SEGMENTS[name].values(races)
    fit_rows = frame[["race_id", "odds", "won"]].assign(**{name: frame["race_id"].map(values)})
    table = _compared(segment_beta(fit_rows, name), name)
    metrics, rows = _beta_table(table, name, "value", segment=name)
    left_out = [{"segment": name, "races in neither side": int(values.isna().sum())}]
    return metrics, {"segments": rows, "segment_left_out": left_out}


def _entropy_bins(frame: pd.DataFrame, cuts: list[float]) -> tuple[list, dict]:
    entropy = normalised_entropy(frame["odds"], frame["race_id"])
    edges = [0.0, *cuts, 1.0]
    labels = [f"{k + 1}: [{edges[k]:g}, {edges[k + 1]:g}{']' if k == 4 else ')'}" for k in range(5)]
    k = np.searchsorted(np.asarray(cuts), entropy.to_numpy(), side="right")
    values = pd.Series(np.asarray(labels, dtype=object)[k], index=entropy.index)
    values = values.where(entropy.notna())
    fit_rows = frame[["race_id", "odds", "won"]].assign(bin=frame["race_id"].map(values))
    table = _compared(segment_beta(fit_rows, "bin"), "bin")
    metrics, rows = _beta_table(table, "bin", "entropy bin")
    rows.append({"entropy bin": "left out (under two priced runners)"})
    rows[-1]["races"] = int(entropy.isna().sum())
    return metrics, {"entropy": rows}


def compute_study(frame: pd.DataFrame, spec: dict) -> Study:
    """Every variant and table of the frozen `spec` on `frame` (development_frame's shape).

    For each subset (every race; races without a floor-bound runner, left out of fitting and
    scoring alike): E1 walk-forward normalised, out-of-fold power and Shin winner log loss, ΔLL
    against normalised and the tail, each with meeting-clustered bootstrap bounds, and the
    in-sample beta by year and era with 95% CIs. In-sample on every development race: beta by E4
    segment (in vs out) and by E3 entropy bin, with 95% CIs. Pure: no I/O.
    """
    floor = flag_floor_bound(frame["odds"], floor=spec["floor"]["odds"])
    floor_race = floor.groupby(frame["race_id"].to_numpy()).transform("any").to_numpy()
    study = Study(counts={"floor_bound_runners": int(floor.sum())})
    study.counts["races_holding_a_floor_bound_runner"] = int(frame["race_id"][floor_race].nunique())
    components = {"fit": FIT}
    columns = ["race_id", "race_date", "meeting_id", "odds", "won"]
    for subset in SUBSETS:
        rows = frame if subset == "all_races" else frame.loc[~floor_race, columns]
        rows = rows.reset_index(drop=True)
        _devig_variants(rows, spec, subset, study)
        variant = Variant(
            "E1",
            f"power_beta:by_year_and_era:{subset}",
            IN_SAMPLE,
            components,
            {"by": ["year", "era"], "subset": subset},
        )
        study.attempt(variant, lambda rows=rows, subset=subset: _year_and_era(rows, subset))
        del rows
    races = frame.drop_duplicates("race_id").set_index("race_id")
    for name, definition in spec["segments"].items():
        variant = Variant(
            "E4",
            f"segment:{name}",
            IN_SAMPLE,
            {**components, "segment": name},
            {"definition": definition},
        )
        study.attempt(variant, lambda name=name: _segment(frame, races, name))
    cuts = spec["entropy_cut_points"]
    variant = Variant(
        "E3",
        "entropy_bins",
        IN_SAMPLE,
        {**components, "entropy": "normalised"},
        {"cut_points": cuts},
    )
    study.attempt(variant, lambda: _entropy_bins(frame, cuts))
    return study


# --------------------------------------------------------------------------- report and runner

SECTIONS = (
    ("folds", "Walk-forward folds: the power beta of each fold's training rows", ""),
    (
        "headline",
        "Winner log loss and ΔLL against normalised (out of fold)",
        "ΔLL is normalised's log loss minus the variant's, per race, in nats (positive: better than"
        " normalised). lower and upper are the (1 - level) and level quantiles of the paired"
        " bootstrap over meetings (D7: one-sided). Judge by the beta CI, not by ΔLL (08 §4 E1).",
    ),
    (
        "tail",
        "Tail: runners priced above the tail threshold (out of fold)",
        "O / E is the tail's observed over expected winners (point only). (O - E) per race is the"
        " tail's observed minus expected winners per scored race, with the same meeting-clustered"
        " bounds; a dead heat counts each co-winner.",
    ),
    ("beta_by_year", "Beta by year (in-sample, development races, 95% CI)", ""),
    (
        "beta_by_era",
        "Beta by era (in-sample)",
        "D10, signed: races before 2014-06-07 are excluded wherever the price enters. No"
        " pre-reform price was read, so the table covers the post-reform era only.",
    ),
    ("segments", "E4 segments: beta in vs out (in-sample, 95% CI = beta +/- 1.96 se)", ""),
    (
        "segment_left_out",
        "E4 races in neither side of a segment",
        "A race whose attribute is missing is in neither side; for g1_day_undercard nor is the G1.",
    ),
    ("entropy", "E3 market-entropy bins (in-sample, 95% CI)", ""),
)


def _cell(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return "" if value is None else str(value).replace("|", "\\|")


def _markdown_table(rows: list[dict]) -> str:
    head = list(dict.fromkeys(key for row in rows for key in row))
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    lines += ["| " + " | ".join(_cell(row.get(k)) for k in head) + " |" for row in rows]
    return "\n".join(lines)


def render_report(study: Study, spec: dict, about: dict) -> str:
    """The full report: aggregate tables only, never a runner or race row."""
    lines = [
        f"# Crowd calibration at the final Win price ({COUNTRY}, #{TICKET})",
        "",
        *(f"- {key}: {value}" for key, value in about.items()),
        f"- owner sign-off of the pre-registered inputs: #{spec['owner_signoff']}",
        f"- tail: above {spec['tail']['min_odds']:g}; floor: {spec['floor']['odds']:g}",
        "- development rows only: holdout days are never read and the holdout is not opened",
        "",
        "## Owner decisions used",
        "",
        *(f"- {key}: {value}" for key, value in spec["decisions"].items()),
        "",
        "## Data",
        "",
        _markdown_table([{"count": k, "n": v} for k, v in study.counts.items()]),
    ]
    for key, title, note in SECTIONS:
        rows = study.tables.get(key)
        lines += ["", f"## {title}", ""]
        lines += [note, ""] if note else []
        lines.append(_markdown_table(rows) if rows else "_Not computed: see the variants below._")
    variants = [
        {"E": v.hypothesis_id, "variant": v.name, "status": "failed" if v.error else "scored"}
        | ({"error": v.error} if v.error else {})
        for v in study.variants
    ]
    lines += ["", "## Variants (one registry record each)", "", _markdown_table(variants), ""]
    return "\n".join(lines)


def _record(variant: Variant, git_sha: str, builds: dict[str, str]) -> RunRecord:
    return RunRecord(
        hypothesis_id=variant.hypothesis_id,
        family_id=FAMILY_ID,
        variant=variant.name,
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=builds,
        sources=(SOURCE,),
        fold_scheme=variant.fold_scheme,
        components=variant.components,
        params=variant.params,
        metrics=variant.metrics,
        status="failed" if variant.error else "scored",
    )


def run_study(out_dir: Path | None = None) -> dict:
    """Run the pre-registered study on Japan's development data and write its report.

    Raises `MissingOwnerDecision` before the store is read when config/calibration.yaml lacks an
    input, `paths.DataDirInsideRepo` for an `out_dir` inside the repository, ValueError before
    the freeze when the yaml's cut points are not ENTROPY_CUT_RULE's (recomputed from the first
    window's prices alone), and `FamilyAlreadyFrozen` when the family is frozen with another
    spec."""
    owner = owner_inputs()
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    spec = build_spec(owner, load_country_scheme(COUNTRY), bootstrap_settings(), win_floor())
    check_cut_points(owner["entropy_cut_points"])  # prices only: no outcome before the freeze
    git_sha = _git_sha()
    registry = default_registry(COUNTRY)
    spec_hash = registry.freeze_family(FAMILY_ID, spec)  # before any likelihood is evaluated

    builds = store_builds()
    runners, races = read_development()
    frame, counts = development_frame(runners, races)
    del runners, races
    study = compute_study(frame, spec)
    study.counts = {**counts, **study.counts}
    del frame

    existing = registry.records(FAMILY_ID)
    records = [_record(v, git_sha, builds) for v in study.variants]
    appended = [registry.append(r) for r in records if r not in existing]
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "store": ", ".join(builds.values()),
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = write_report(
        out_dir / f"crowd-calibration-{COUNTRY}-{spec_hash[:12]}.md",
        render_report(study, spec, about),
        sources=(SOURCE,),
    )
    return {
        "family_id": FAMILY_ID,
        "spec_hash": spec_hash,
        "report_path": str(report),
        "registry_path": str(registry.path),
        "counts": study.counts,
        "records_appended": len(appended),
        "records_already_present": len(records) - len(appended),
        "failed_variants": [v.name for v in study.variants if v.error],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m horsetrack.countries.japan.studies.calibration"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true", help="run the study")
    action.add_argument(
        "--entropy-cut-points",
        action="store_true",
        help="print E3's cut points by the owner's rule (prices of the first window only)",
    )
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    args = parser.parse_args(argv)
    if args.entropy_cut_points:
        print(json.dumps(first_window_cut_points(), indent=2))
        return 0
    print(json.dumps(run_study(args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
