"""The pre-registered Family P selection study (#88; 08 §5.2 end to end, §5.3, §8 A6).

A family of positive-edge rules (08 §5.1 Family P) goes through 08 §5.2's stages in order, on the
development rows the caller passes (one row per starter, `split_off`'s development side). The
owner's answers on #88 (2026-11-26) set the benchmark, the one-sided pass, the omega basis, the
odds bands and the rungs; each is cited below as "answer N".

0. Freeze and hygiene. `run_study` raises `FamilyNotFrozen` before it scores or writes anything
   unless the registry holds the family frozen with exactly this spec: the hash
   `Registry.freeze_family` gives `dataclasses.asdict(spec)` (a direction, the odds-band cut
   points and the dropped variants included). The owner approves the family and the caller
   freezes it; the study never freezes it itself. Then, also before anything is scored or
   written: a rule whose segment or selection names a look-ahead column refuses the run with
   `LookAheadSelection` (every rule, survivor or not). Look-ahead is the final price and the
   result (`odds_final`, `won`, `win_dividend`, any `final_*`, the store's `win_odds` and
   `win_fav_rank`) and every label of Japan's availability config
   (`core.store.asof.load_availability`: a table's `label_columns`, or a name one of its
   `label_prefixes` starts), which a view at t never returns. A row on a holdout day
   (`core.eval.holdout.is_holdout_day`) is a ValueError.
1. Log loss. The benchmark is D4's calibrated market (`calibrated_market`, answer 3): the
   conditional-logit natural cubic spline in ln q of `models.v0.calibration`, refitted on each of
   D9's walk-forward folds (config/folds.yaml) on the fold's training rows of every race, so a
   favourite-longshot bias the whole market shares is in the benchmark and no segment is credited
   with it. Each rule's claim is that benchmark plus a segment-specific departure, fitted on the
   segment's training races by `models.condlogit` with the fold's ln p_cal as the offset, and
   scored out of fold on the segment's test races (the benchmark everywhere else):
   - 'beta': in the segment's races p proportional to p_cal^beta_s (null 1). beta_s > 1: the
     segment's favourites (by the benchmark) win more often than the benchmark says; beta_s < 1:
     its longshots do. The segment must be race-level.
   - 'omega': p proportional to p_cal * exp(gamma_s * [runner in segment]) in each race holding a
     segment runner (null 0). gamma_s > 0: the segment's runners win more often than the
     benchmark says; gamma_s < 0: less often.
   - 'alpha' is refused with `AlphaMemberUnnamed`: alpha_s is a C4 member's segment weight (#87's
     `<member>:<segment>` term) and neither RuleSpec nor the data names the member (answer 8
     keeps P4-alpha deferred).
   The per-race gain (the benchmark's winner log loss minus the claim's) goes to #69's paired
   bootstrap over `cluster_unit` at D7's level and replicates (config/scoring.yaml). A claim
   passes when the one-sided lower bound is above 0 and its coefficient lies on the rule's
   `direction` side of the null (answer 6): "above" needs beta_s > 1 (gamma_s > 0), "below"
   beta_s < 1 (gamma_s < 0). The coefficient judged is the mean of the per-fold fits weighted by
   the segment races each fold scores: the fits the out-of-fold predictions used.
   Stage 1 writes nothing. Before the first record: SPA and StepM test every variant the family
   holds scored, so a rerun whose frame cannot score one is a ValueError naming it and its stage-1
   error (the registry unchanged). Then the spec's `dropped_variants` are recorded as failed
   variants (one each, skipping a variant the family already holds, so none counts twice), and
   each rule's stage-1 record.
2. Max statistic. #75's SPA and StepM over the rules scored at stage 1 (their registry records
   are appended first, so the family check sees them), on per-race losses of every scored race
   (the calibrated market's outside a rule's segment), in `cluster_unit` clusters ordered by race
   date, with the family's frozen `max_stat` settings.
3. Omega. Per rule, #78's hierarchical model (`fit_segment_roi`) on the rule's selected bets only
   (answer 4), at the final Win odds, with fixed effects per odds band cut from the final odds at
   the family's `odds_band_cut_points` (answer 7; a price on a cut point is in the band above it)
   and the rule's segment columns as its random effects. Every bet of the fit is in the rule's
   segment, so the segment effect is not identified apart from the bands; what the bets identify
   is their own mix: `omega_selected`, the posterior mean of the bets' mean omega (band and
   segment effects combined), `p_beats_takeout`, the posterior probability that the same mean
   exceeds 1/(1 - t) (#78's, on the same bets), and `ln_omega_by_band`, each band's ln omega with
   the segment effect added.
   A survivor's entry also carries the predictive ROI of its next PREDICTIVE_BETS bets (08 §5.2
   step 4); no other rule's entry or record carries a return. One fit per rule, so one rule's gaps
   never touch another's. Reported only: it never changes a survivor or a rung.
4. ROI, for the survivors only (passed stage 1 and found superior by StepM): one unit to Win on
   each runner of the segment whose selection columns are all in [lo, hi] (inclusive, None an
   open end, a missing value never selected), paid at `win_dividend`. A loser pays 0 (its
   dividend may be missing); a selected winner without a dividend is an error the stage reports
   for that rule (its ROI NaN, no metric), never a silent NaN.

Rungs: S1 (shadow) for a survivor, S0 (avoid) for every other rule (answer 5); nothing goes above
S1 and no record is adopted. The holdout passed in is never opened.

Records: one failed record per dropped variant the family does not hold yet, one per rule
(variant = rule_id, hypothesis id = its leading P/E/L id) before StepM, status 'failed' when its
claim could not be scored, and a second one per rule at the end carrying every stage's metrics (a
survivor's ROI as `roi_win_dividend`). A record already in the registry is not appended again.
`sources` defaults to a licensed source (fails closed, D21): pass the sources `data` was built
from.

Memory: one column-scoped copy of `data`; every fold works on positions into it, and the
benchmark's ln p_cal is held for one fold at a time.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import LockedHoldout, _git_sha, is_holdout_day
from horsetrack.core.eval.registry import FoldMetric, Registry, RunRecord, _spec_hash
from horsetrack.core.eval.scoring import gain, per_race_log_loss
from horsetrack.core.eval.segment_roi import fit_segment_roi
from horsetrack.core.eval.snooping import spa, stepm
from horsetrack.core.store.asof import load_availability
from horsetrack.countries.japan.models.condlogit import fit_conditional_logit
from horsetrack.countries.japan.models.v0.calibration import fit_win_calibration
from horsetrack.countries.japan.studies.calibration import bootstrap_settings, devig_normalised

COUNTRY = "japan"
TICKET = 88
CLAIMS = ("beta", "alpha", "omega")
DIRECTIONS = ("above", "below")
BENCHMARK = "calibrated_market"
NULLS = {"beta": 1.0, "omega": 0.0}  # each claim's coefficient under no segment effect
COEF_NAMES = {"beta": "beta_vs_calibrated_market", "omega": "gamma_vs_calibrated_market"}
MAX_STAT_KEYS = frozenset({"level", "block_size", "n_boot"})
FINAL_ODDS, DIVIDEND, TAKEOUT = "odds_final", "win_dividend", "takeout"
ODDS_BAND = "odds_band"  # the omega model's favourite-longshot band, cut from the final odds
BASE_COLUMNS = ["race_id", "race_date", "won", FINAL_ODDS, DIVIDEND, TAKEOUT]
# The final price, the result, and the store's names for the final Win odds and favourite rank;
# every label of Japan's availability config is look-ahead too (`_check_look_ahead`).
LOOK_AHEAD = frozenset({FINAL_ODDS, DIVIDEND, "won", "win_odds", "win_fav_rank"})
LOOK_AHEAD_PREFIX = "final_"
OMEGA_DRAWS = 4000  # posterior draws of the omega model: Monte Carlo precision only
PREDICTIVE_BETS = 100  # 08 §5.2 step 4: the posterior predictive ROI of the next 100 bets
SHADOW, AVOID = "S1", "S0"
UNSTATED_SOURCES = ("jra", "jrdb")  # fails closed: unnamed sources are treated as licensed
FAMILY_CONFIG = "selection_family"
_HYPOTHESIS = re.compile(r"([ELP][0-9]+)(?:-|$)")


class FamilyNotFrozen(RuntimeError):
    """The registry does not hold the family frozen with this exact spec; nothing was scored."""


class LookAheadSelection(ValueError):
    """A rule selects its bets on a column known only at the final price or after the race."""


class AlphaMemberUnnamed(ValueError):
    """An 'alpha' rule: the owner must name the C4 member whose segment weight is alpha_s."""


class FamilyNotApproved(RuntimeError):
    """The family file carries no owner sign-off; it is a proposal and must not be frozen."""


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    claim: str
    segment: dict
    selection_cols: list[str]
    thresholds: dict
    direction: str


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    rules: list[RuleSpec]
    decision_time: str
    cluster_unit: str
    benchmark: str
    max_stat: dict
    odds_band_cut_points: list[float]
    dropped_variants: list[str] = dataclasses.field(default_factory=list)


@dataclass(frozen=True)
class StudyReport:
    family_size: int
    stages: dict
    survivors: list[str]
    p_beats_takeout: dict[str, float]
    rungs: dict[str, str]


# --------------------------------------------------------------------------- the family


def spec_hash(spec: FamilySpec) -> str:
    """The hash `Registry.freeze_family(spec.family_id, dataclasses.asdict(spec))` returns."""
    return _spec_hash(dataclasses.asdict(spec))


def load_family(path_or_country: str | Path = COUNTRY, *, proposed: bool = False) -> FamilySpec:
    """The family in a country's `config/selection_family.yaml`, or in the yaml at a path.

    Refused with `FamilyNotApproved` unless the file's `owner_signoff` names the issue where the
    owner approved it; `proposed=True` reads a proposal for review. `dropped` (each entry's
    `variant`, in the file's order) becomes the spec's `dropped_variants`, which `run_study`
    records as failed variants; `status` and `deferred` are for the reader. Nothing is frozen.
    """
    if isinstance(path_or_country, Path) or str(path_or_country).endswith((".yaml", ".yml")):
        raw = yaml.safe_load(Path(path_or_country).read_text(encoding="utf-8")) or {}
    else:
        raw = load_country_yaml(str(path_or_country), FAMILY_CONFIG)
    fields = [f.name for f in dataclasses.fields(FamilySpec) if f.name != "dropped_variants"]
    unknown = sorted(set(raw) - {*fields, "status", "owner_signoff", "dropped", "deferred"})
    missing = [name for name in fields if name not in raw]
    if unknown or missing:
        raise ValueError(f"family file: unknown key(s) {unknown}, missing {missing}")
    signoff = raw.get("owner_signoff")
    if not proposed and not (isinstance(signoff, int) and not isinstance(signoff, bool)):
        raise FamilyNotApproved(f"{raw['family_id']!r} has no owner_signoff: {raw.get('status')}")
    rule_fields = {f.name for f in dataclasses.fields(RuleSpec)}
    rules = []
    for entry in raw["rules"]:
        if set(entry) != rule_fields:
            raise ValueError(f"a rule has keys {sorted(entry)}, not {sorted(rule_fields)}")
        rules.append(
            RuleSpec(
                rule_id=entry["rule_id"],
                claim=entry["claim"],
                segment=dict(entry["segment"]),
                selection_cols=list(entry["selection_cols"]),
                thresholds={col: list(b) for col, b in entry["thresholds"].items()},
                direction=entry["direction"],
            )
        )
    dropped = []
    for entry in raw.get("dropped") or []:
        if not (isinstance(entry, dict) and isinstance(entry.get("variant"), str)):
            raise ValueError(f"a dropped entry needs its variant id: {entry!r}")
        dropped.append(entry["variant"])
    values = {name: raw[name] for name in fields}
    values["odds_band_cut_points"] = list(raw["odds_band_cut_points"])
    return FamilySpec(**{**values, "rules": rules, "dropped_variants": dropped})


def _hypothesis_id(rule_id: object) -> str:
    match = _HYPOTHESIS.match(rule_id) if isinstance(rule_id, str) else None
    if match is None:
        raise ValueError(f"rule_id {rule_id!r} must start with its hypothesis id, e.g. 'P3-'")
    return match.group(1)


def _check_frozen(spec: FamilySpec, registry: Registry) -> str:
    try:
        expected = spec_hash(spec)
    except (TypeError, ValueError) as exc:
        raise FamilyNotFrozen(f"{spec.family_id!r}: the spec is not plain JSON ({exc})") from exc
    frozen = registry.frozen_spec_hash(spec.family_id)
    if frozen is None:
        raise FamilyNotFrozen(
            f"family {spec.family_id!r} is not frozen in {registry.path}: the owner approves it "
            "and the caller freezes dataclasses.asdict(spec) before anything is scored"
        )
    if frozen != expected:
        raise FamilyNotFrozen(
            f"family {spec.family_id!r} is frozen with spec {frozen}; this spec hashes to "
            f"{expected}: it was edited after the freeze"
        )
    return expected


def _real(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def _check_look_ahead(spec: FamilySpec) -> None:
    """Selection hygiene (08 §5.2): refuse a family any of whose rules selects its bets, through
    its segment or a selection column, on a final-price or result column. Every rule, survivor
    or not, so a family's validity never depends on what the data let through."""
    tables = load_availability(COUNTRY).values()

    def look_ahead(col: str) -> bool:
        final = col in LOOK_AHEAD or col.startswith(LOOK_AHEAD_PREFIX)
        return final or any(table.is_label(col) for table in tables)

    offending = {
        rule.rule_id: cols
        for rule in spec.rules
        if (cols := [col for col in [*rule.segment, *rule.selection_cols] if look_ahead(col)])
    }
    if offending:
        raise LookAheadSelection(
            f"rules select on look-ahead columns {offending}: a bet is selected from the board "
            "at t or from as-of features only (08 §5.2, selection hygiene; a label of "
            f"countries/{COUNTRY}/config/availability.yaml is never in a view at t)"
        )


def _check_spec(spec: FamilySpec, data: pd.DataFrame, seed: object) -> list[str]:
    """Refuse a spec the study cannot run, before anything is scored; return the columns read."""
    if spec.benchmark != BENCHMARK:
        raise ValueError(f"benchmark must be {BENCHMARK!r}, got {spec.benchmark!r}")
    if not isinstance(spec.max_stat, dict) or set(spec.max_stat) != MAX_STAT_KEYS:
        raise ValueError(f"max_stat must give exactly {sorted(MAX_STAT_KEYS)}: {spec.max_stat!r}")
    cuts = spec.odds_band_cut_points
    if not (
        isinstance(cuts, (list, tuple))
        and cuts
        and all(_real(c) and c > 1 for c in cuts)
        and all(lo < hi for lo, hi in zip(cuts[:-1], cuts[1:], strict=True))
    ):
        raise ValueError(
            f"odds_band_cut_points must be strictly ascending decimal odds above 1: {cuts!r}"
        )
    if not (isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0):
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    ids = [rule.rule_id for rule in spec.rules]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"a family needs one or more rules with distinct ids, got {ids}")
    columns = [*BASE_COLUMNS, spec.cluster_unit]
    for rule in spec.rules:
        _hypothesis_id(rule.rule_id)
        if rule.claim not in CLAIMS:
            raise ValueError(f"{rule.rule_id}: claim must be one of {CLAIMS}, got {rule.claim!r}")
        if rule.claim == "alpha":
            raise AlphaMemberUnnamed(
                f"{rule.rule_id}: alpha_s is the segment weight of a C4 member (#87's "
                "'<member>:<segment>' term); the owner must name the member and the data carry "
                "its out-of-fold column before an 'alpha' rule can run"
            )
        if rule.direction not in DIRECTIONS:
            raise ValueError(
                f"{rule.rule_id}: direction must be one of {DIRECTIONS}, got {rule.direction!r}"
            )
        if not (isinstance(rule.segment, dict) and rule.segment):
            raise ValueError(f"{rule.rule_id}: segment must name one or more {{column: level}}")
        cols = list(rule.selection_cols)
        if len(set(cols)) != len(cols) or set(rule.thresholds) != set(cols):
            raise ValueError(f"{rule.rule_id}: thresholds need [lo, hi] for each selection column")
        for col in cols:
            bounds = rule.thresholds[col]
            ok = isinstance(bounds, (list, tuple)) and len(bounds) == 2
            if not (ok and all(b is None or _real(b) for b in bounds)):
                raise ValueError(f"{rule.rule_id}: {col} needs [lo, hi] numbers or None: {bounds}")
            if None not in bounds and bounds[0] > bounds[1]:
                raise ValueError(f"{rule.rule_id}: {col}'s range {bounds} is empty")
        columns += [*rule.segment, *cols]
    dropped = spec.dropped_variants
    if not (isinstance(dropped, (list, tuple)) and all(isinstance(v, str) for v in dropped)):
        raise ValueError(f"dropped_variants must be a list of variant ids: {dropped!r}")
    for variant in dropped:
        if not _HYPOTHESIS.match(variant):
            raise ValueError(f"dropped variant {variant!r} must start with its hypothesis id")
    if len(set(dropped)) != len(dropped) or set(dropped) & set(ids):
        raise ValueError(f"dropped_variants must be distinct and no rule's id: {list(dropped)}")
    _check_look_ahead(spec)
    columns = list(dict.fromkeys(columns))
    missing = [col for col in columns if col not in data.columns]
    if missing:
        raise ValueError(f"data lacks column(s) {missing}")
    return columns


def _check_development(frame: pd.DataFrame) -> None:
    """Refuse a frame holding a row on a holdout day (the country's committed definition)."""
    stamps = pd.to_datetime(frame["race_date"])
    if stamps.isna().any():
        raise ValueError("race_date has missing values; every row needs its race day")
    held = [day for day in sorted(set(stamps.dt.date)) if is_holdout_day(COUNTRY, day)]
    if held:
        raise ValueError(
            f"data holds rows on {len(held)} holdout day(s) ({held[0].isoformat()} first): the "
            "study runs on development rows only, and the holdout is opened only by the owner"
        )


# --------------------------------------------------------------------------- rows


def _in_segment(frame: pd.DataFrame, segment: dict) -> np.ndarray:
    keep = np.ones(len(frame), dtype=bool)
    for col, level in segment.items():
        keep &= frame[col].eq(level).fillna(False).to_numpy(dtype=bool)
    return keep


def _selected(frame: pd.DataFrame, rule: RuleSpec) -> np.ndarray:
    """The rule's bets: its segment, every selection column present and in [lo, hi]."""
    keep = _in_segment(frame, rule.segment)
    for col in rule.selection_cols:
        values = frame[col].astype("float64").to_numpy()
        lo, hi = rule.thresholds[col]
        keep &= ~np.isnan(values)
        if lo is not None:
            keep &= values >= lo
        if hi is not None:
            keep &= values <= hi
    return keep


def _race_ll(p: pd.Series, frame: pd.DataFrame, rows: np.ndarray) -> pd.Series:
    """Winner log loss per race of `p` on `rows` (per_race_log_loss renormalises each race)."""
    part = frame.iloc[rows]
    p = pd.Series(np.asarray(p, dtype=float), index=part.index)
    no = pd.Series(False, index=part.index)
    return per_race_log_loss(p, part["won"].astype(bool), part["race_id"], no).per_race


def _softmax(u: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """exp(u) normalised within each race code."""
    top = pd.Series(u).groupby(codes).transform("max").to_numpy()
    e = np.exp(u - top)
    return e / pd.Series(e).groupby(codes).transform("sum").to_numpy()


def _band_labels(cuts: Sequence[float]) -> list[str]:
    """One label per band of `np.digitize(odds, cuts)`: '<c1', 'c1-c2', ..., 'cK+'."""
    edges = [f"{c:g}" for c in cuts]
    middle = [f"{lo}-{hi}" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    return [f"<{edges[0]}", *middle, f"{edges[-1]}+"]


# --------------------------------------------------------------------------- stage 1


@dataclass
class _Claim:
    """One rule's claim across the folds: its rows, out-of-fold probabilities and fits."""

    member: np.ndarray  # runners in the segment
    holds: np.ndarray  # runners of races holding a segment runner
    rows: list = dataclasses.field(default_factory=list)
    probs: list = dataclasses.field(default_factory=list)
    fits: dict = dataclasses.field(default_factory=dict)  # fold -> (coef, n_train, n_test races)
    error: str | None = None


def _new_claim(frame: pd.DataFrame, rule: RuleSpec, codes: np.ndarray) -> _Claim:
    member = _in_segment(frame, rule.segment)
    by_race = pd.Series(member).groupby(codes)
    any_in, all_in = by_race.transform("any"), by_race.transform("all")
    if rule.claim == "beta" and (any_in != all_in).any():
        mixed = int(pd.Series(any_in != all_in).groupby(codes).any().sum())
        raise ValueError(f"a 'beta' segment must be race-level: {mixed} race(s) mix in and out")
    return _Claim(member=member, holds=any_in.to_numpy(dtype=bool))


def _fit_fold(claim: _Claim, rule: RuleSpec, fold, ln_p, race_ids, won, codes) -> None:
    """The claim's departure from the fold's benchmark: fitted on the segment's training races,
    predicted on its test races (ln_p: the fold's benchmark ln p_cal on train and test rows)."""
    test = fold.test_idx[claim.holds[fold.test_idx]]
    if not len(test):
        return
    train = fold.train_idx[claim.holds[fold.train_idx]]
    if not len(train):
        raise ValueError(f"no training race of the {fold.test_start} fold is in the segment")
    if rule.claim == "beta":
        x_train, x_test = ln_p[train], ln_p[test]
    else:
        x_train, x_test = claim.member[train].astype(float), claim.member[test].astype(float)
    fit = fit_conditional_logit(x_train[:, None], race_ids[train], won[train], offset=ln_p[train])
    delta = float(fit.coef[0])
    claim.rows.append(test)
    claim.probs.append(_softmax(ln_p[test] + delta * x_test, codes[test]))
    n_test = len(np.unique(codes[test]))
    claim.fits[fold.test_start.isoformat()] = (NULLS[rule.claim] + delta, fit.n_races, n_test)


def _pooled(fits: dict) -> float:
    """The per-fold coefficients' mean, weighted by the segment races each fold scores."""
    coef = np.array([c for c, _, _ in fits.values()])
    weight = np.array([n for _, _, n in fits.values()], dtype=float)
    return float(coef @ weight / weight.sum())


def _points_the_rule_way(coef: float, rule: RuleSpec) -> bool:
    null = NULLS[rule.claim]
    return bool(coef > null) if rule.direction == "above" else bool(coef < null)


# --------------------------------------------------------------------------- records


def _metric(fold: str, name: str, value: float, n: int, lo=None, hi=None) -> list[FoldMetric]:
    """A registry metric, or none when the value is not finite."""
    values = [v for v in (value, lo, hi) if v is not None]
    if not all(np.isfinite(v) for v in values):
        return []
    lo, hi = (None if v is None else float(v) for v in (lo, hi))
    return [FoldMetric(fold, name, float(value), lo, hi, int(n))]


def _frame_build(frame: pd.DataFrame) -> dict[str, str]:
    rows = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return {"selection_frame": f"selection_frame@{hashlib.sha256(rows.tobytes()).hexdigest()[:16]}"}


# --------------------------------------------------------------------------- the study


def run_study(
    spec: FamilySpec,
    data: pd.DataFrame,
    *,
    registry: Registry,
    holdout: LockedHoldout,
    seed: int,
    sources: Sequence[str] = UNSTATED_SOURCES,
) -> StudyReport:
    """Run the frozen family on development rows (module docstring) and report every stage.

    Raises, before anything is scored or written: `FamilyNotFrozen`, `AlphaMemberUnnamed`,
    `LookAheadSelection`, or ValueError for a spec or frame the study cannot run (a row on a
    holdout day included). Raises ValueError before the first record when the frame cannot score
    a variant the family holds scored. `holdout` is held, never opened.
    """
    if not isinstance(spec, FamilySpec):
        raise TypeError(f"spec must be a FamilySpec, got {type(spec).__name__}")
    if not isinstance(registry, Registry) or not isinstance(holdout, LockedHoldout):
        raise TypeError("registry must be a Registry and holdout a LockedHoldout")
    frozen_hash = _check_frozen(spec, registry)
    columns = _check_spec(spec, data, seed)
    frame = data[columns].reset_index(drop=True)
    _check_development(frame)
    unit = spec.cluster_unit
    codes = pd.factorize(frame["race_id"])[0]
    races = frame.groupby("race_id", sort=False).agg(
        cluster=(unit, "first"), n_clusters=(unit, "nunique"), date=("race_date", "min")
    )
    if (races["n_clusters"] > 1).any():
        raise ValueError(f"cluster_unit {unit!r} varies within a race")
    scheme = load_country_scheme(COUNTRY)
    folds = walk_forward(frame["race_date"], scheme=scheme)
    boot = bootstrap_settings()
    rules = {rule.rule_id: rule for rule in spec.rules}
    identity = {
        "git_sha": _git_sha(),
        "data_builds": _frame_build(frame),
        "sources": tuple(sources),
        "fold_scheme": "walk_forward:" + json.dumps(dataclasses.asdict(scheme), sort_keys=True),
    }

    def record(rule: RuleSpec, metrics: list[FoldMetric], failed: bool) -> RunRecord:
        params = {
            "spec_sha256": frozen_hash,
            "segment": dict(rule.segment),
            "selection_cols": list(rule.selection_cols),
            "thresholds": {col: list(b) for col, b in rule.thresholds.items()},
            "direction": rule.direction,
            "decision_time": spec.decision_time,
            "cluster_unit": unit,
            "max_stat": dict(spec.max_stat),
            "odds_band_cut_points": list(spec.odds_band_cut_points),
        }
        components = {"claim": rule.claim, "benchmark": spec.benchmark, "score": "winner_log_loss"}
        return RunRecord(
            hypothesis_id=_hypothesis_id(rule.rule_id),
            family_id=spec.family_id,
            variant=rule.rule_id,
            ticket=TICKET,
            components=components,
            params=params,
            metrics=tuple(metrics),
            status="failed" if failed else "scored",
            **identity,
        )

    def dropped(variant: str) -> RunRecord:
        """A variant screened and dropped before the freeze: a failed variant with no metrics."""
        return RunRecord(
            hypothesis_id=_hypothesis_id(variant),
            family_id=spec.family_id,
            variant=variant,
            ticket=TICKET,
            components={"claim": "dropped", "benchmark": spec.benchmark},
            params={"spec_sha256": frozen_hash, "dropped_before_freeze": True},
            metrics=(),
            status="failed",
            **identity,
        )

    def append(records: list[RunRecord]) -> None:
        existing = registry.records(spec.family_id)
        for r in records:
            if r not in existing:
                registry.append(r)

    # Stage 1: each rule's departure from the calibrated market, fitted in fold, out of fold.
    race_ids, won = frame["race_id"].to_numpy(), frame["won"].to_numpy()
    q = devig_normalised(frame[FINAL_ODDS], frame["race_id"]).to_numpy()
    market = frame[["race_id", "race_date", FINAL_ODDS, "won"]].rename(
        columns={FINAL_ODDS: "win_odds"}
    )
    claims: dict[str, _Claim] = {}
    for rid, rule in rules.items():
        try:
            claims[rid] = _new_claim(frame, rule, codes)
        except Exception as exc:  # every failure is a registry record, never dropped
            claims[rid] = _Claim(np.zeros(0, bool), np.zeros(0, bool), error=f"{exc!r}")
    bench_rows, bench_probs, bench_fits = [], [], {}
    ln_p = np.full(len(frame), np.nan)
    for fold in folds:
        if not len(fold.test_idx):
            continue
        model = fit_win_calibration(market.iloc[fold.train_idx])
        rows = np.concatenate([fold.train_idx, fold.test_idx])
        ln_p[rows] = np.log(model.predict_many(q[rows], race_ids[rows]))
        bench_rows.append(fold.test_idx)
        bench_probs.append(np.exp(ln_p[fold.test_idx]))
        bench_fits[fold.test_start.isoformat()] = model.fit_id
        for rid, rule in rules.items():
            claim = claims[rid]
            if claim.error is None:
                try:
                    _fit_fold(claim, rule, fold, ln_p, race_ids, won, codes)
                except Exception as exc:
                    claim.error = f"{exc!r}"
        ln_p[rows] = np.nan
    if bench_rows:
        scored = np.concatenate(bench_rows)
        bench = _race_ll(np.concatenate(bench_probs), frame, scored)
    else:
        bench = pd.Series(dtype=float)

    log_loss, losses, metrics = {}, {}, {}
    for rid, rule in rules.items():
        claim = claims[rid]
        try:
            if claim.error is not None:
                raise RuntimeError(claim.error)
            if not claim.rows:
                raise ValueError("no walk-forward fold scores a race of the segment")
            ll = _race_ll(np.concatenate(claim.probs), frame, np.concatenate(claim.rows))
            g = gain(ll, bench)
            iv = paired_bootstrap(
                g.per_race, races["cluster"], n_boot=boot["n_boot"], level=boot["level"], seed=seed
            )
        except Exception as exc:  # every failure is a registry record, never dropped
            log_loss[rid] = {"claim": rule.claim, "passed": False, "error": f"{exc!r}"}
            metrics[rid] = []
            continue
        coef = _pooled(claim.fits)
        right_way = _points_the_rule_way(coef, rule)
        log_loss[rid] = {
            "claim": rule.claim,
            "benchmark": BENCHMARK,
            "passed": bool(iv.lo > 0 and right_way),
            "delta_ll": iv.mean,
            "lo": iv.lo,
            "hi": iv.hi,
            "level": boot["level"],
            "n_races": g.n_common,
            "n_clusters": iv.n_clusters,
            "coef_name": COEF_NAMES[rule.claim],
            "coef_null": NULLS[rule.claim],
            "coef": coef,
            "direction": rule.direction,
            "points_the_rule_way": right_way,
            "coef_by_fold": {fold: c for fold, (c, _, _) in claim.fits.items()},
        }
        name = COEF_NAMES[rule.claim]
        metrics[rid] = _metric(
            "all_folds", "delta_ll_vs_calibrated_market", iv.mean, g.n_common, iv.lo, iv.hi
        )
        metrics[rid] += _metric("all_folds", name, coef, g.n_common)
        for fold, (c, n_train, _) in claim.fits.items():
            metrics[rid] += _metric(fold, name, c, n_train)
        column = bench.copy()  # the calibrated market's loss outside the rule's races
        column.loc[g.per_race.index] = ll.loc[g.per_race.index]
        losses[rid] = column

    # Nothing is written yet. SPA and StepM test every variant the family holds scored, so a
    # rerun that cannot score one is refused here, not after its first records.
    held = registry.records(spec.family_id)
    lost = sorted({r.variant for r in held if r.status != "failed"} - set(losses))
    if lost:
        why = {
            v: log_loss[v]["error"] if v in log_loss else "not a rule of this spec" for v in lost
        }
        raise ValueError(
            f"family {spec.family_id!r}: the registry holds {lost} scored, and this frame "
            f"cannot score them ({why}); the max-statistic stage tests every scored variant of "
            "the family, so the run is refused before it writes a record"
        )
    # The dropped variants first (one record each, unless the family already holds that variant,
    # so the family size never counts one twice), then each rule's stage-1 record.
    known = {r.variant for r in held}
    append(
        [dropped(v) for v in spec.dropped_variants if v not in known]
        + [record(rules[rid], metrics[rid], rid not in losses) for rid in rules]
    )

    # Stage 2: SPA ("does any rule beat the calibrated market") and StepM ("which ones").
    tested = list(losses)
    max_stat = {"tested": tested, "superior": [], "settings": dict(spec.max_stat)}
    if tested:
        common = {
            "clusters": races["cluster"],
            "order": races["date"],
            "block_size": spec.max_stat["block_size"],
            "n_boot": spec.max_stat["n_boot"],
            "seed": seed,
            "registry": registry,
            "family_id": spec.family_id,
        }
        frame_losses = pd.DataFrame(losses)
        found = spa(bench, frame_losses, **common)
        superior = set(stepm(bench, frame_losses, level=spec.max_stat["level"], **common).superior)
        max_stat |= {
            "superior": [rid for rid in tested if rid in superior],
            "spa_pvalue": found.pvalue,
            "spa_pvalue_lower": found.pvalue_lower,
            "spa_pvalue_upper": found.pvalue_upper,
            "family_size": found.family_size,
            "n_clusters": found.n_clusters,
        }
    survivors = [rid for rid in rules if log_loss[rid]["passed"] and rid in max_stat["superior"]]
    bets_of = {rid: _selected(frame, rule) for rid, rule in rules.items()}  # hygiene: _check_spec

    # Stage 3: #78's omega model on each rule's selected bets, odds-band fixed effects. Reported.
    # Every bet of a fit is in the rule's segment, so the bets identify only their own mix's
    # multiplier (band and segment effects together): omega_selected, and P that it beats the
    # takeout. A return (the predictive ROI) is a survivor's only.
    odds = frame[FINAL_ODDS].to_numpy(dtype=float)
    labels = np.array(_band_labels(spec.odds_band_cut_points))
    band = labels[np.digitize(odds, spec.odds_band_cut_points)]
    takeout = frame[TAKEOUT].to_numpy(dtype=float)
    omega = {}
    for rid, rule in rules.items():
        bet = bets_of[rid]
        n_bets = int(bet.sum())
        entry = {"n_bets": n_bets, "p_beats_takeout": float("nan"), "omega_selected": np.nan}
        entry["breakeven_omega"] = float(1.0 / (1.0 - takeout[bet].mean())) if n_bets else np.nan
        try:
            if not n_bets:
                raise ValueError("the rule selects no bet")
            query = {col: str(level) for col, level in rule.segment.items()}
            bets = pd.DataFrame(
                {"odds": odds[bet], "won": won[bet], ODDS_BAND: band[bet], TAKEOUT: takeout[bet]}
            ).assign(**query)  # every selected bet is in the rule's segment
            model = fit_segment_roi(
                bets,
                segment_cols=list(query),
                odds_band_col=ODDS_BAND,
                takeout_col=TAKEOUT,
                seed=seed,
                draws=OMEGA_DRAWS,
            )
            segment_effect = model.posterior_mean(query)  # alone, only the priors' split
            entry |= {
                "p_beats_takeout": float(model.p_beats_takeout(query)),
                "omega_selected": float(model.omega_mean(query)),
                "ln_omega_by_band": {
                    b: float(mu + segment_effect) for b, mu in model.band_effects.items()
                },
            }
            if rid in survivors:
                interval = model.predictive_roi(query, PREDICTIVE_BETS)
                entry[f"predictive_roi_{PREDICTIVE_BETS}"] = [
                    interval.low,
                    interval.high,
                    interval.level,
                ]
        except Exception as exc:  # reported with the stage; the other stages stand
            entry["error"] = f"{exc!r}"
        omega[rid] = entry

    # Stage 4: ROI at the final Win dividend, survivors only, selection at t; losers pay 0.
    dividend = frame[DIVIDEND].to_numpy(dtype=float)
    wins = won.astype(bool)
    roi = {}
    for rid in survivors:
        bet = bets_of[rid]
        n_bets = int(bet.sum())
        unpaid = int((bet & wins & ~np.isfinite(dividend)).sum())
        entry = {"roi": float("nan"), "n_bets": n_bets}
        if not n_bets:
            entry["error"] = "the rule selects no bet"
        elif unpaid:
            entry["error"] = f"{unpaid} selected winner(s) carry no {DIVIDEND}"
        else:
            entry["roi"] = float(dividend[bet & wins].sum() / n_bets - 1.0)
        roi[rid] = entry

    rungs = {rid: SHADOW if rid in survivors else AVOID for rid in rules}
    outcome = []
    for rid, rule in rules.items():
        extra = []
        if rid in losses:
            superior = float(rid in max_stat["superior"])
            extra += _metric("all_folds", "stepm_superior", superior, max_stat.get("n_clusters", 0))
            p = omega[rid]["p_beats_takeout"]
            extra += _metric("development", "p_beats_takeout", p, omega[rid]["n_bets"])
            if rid in roi:
                value, n_bets = roi[rid]["roi"], roi[rid]["n_bets"]
                extra += _metric("development", "roi_win_dividend", value, n_bets)
        outcome.append(record(rule, metrics[rid] + extra, rid not in losses))
    append(outcome)

    return StudyReport(
        family_size=registry.family_size(spec.family_id),
        stages={
            "log_loss": log_loss,
            "max_stat": max_stat,
            "omega": omega,
            "roi": roi,
            "benchmark_fits": bench_fits,
        },
        survivors=survivors,
        p_beats_takeout={rid: float(omega[rid]["p_beats_takeout"]) for rid in rules},
        rungs=rungs,
    )
