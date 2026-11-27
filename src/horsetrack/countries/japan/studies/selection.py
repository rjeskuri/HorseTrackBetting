"""The pre-registered Family P selection study (#88; 08 §5.2 end to end, §8 A6).

A family of positive-edge rules (08 §5.1 Family P) is measured on the development rows the caller
passes (one row per starter). The owner's answers on #88 (2026-11-26) set the benchmark, the
one-sided pass, the omega basis and the odds bands ("answer N" below); the owner's reframe of
2026-11-27 keeps every result: no stage gates another, survival is a diagnostic, and the study
makes no rung decision.

Before anything is scored or written, `run_study` refuses:
- an unfrozen family, or a spec edited after the freeze (`FamilyNotFrozen`): the registry must
  hold `Registry.freeze_family(spec.family_id, dataclasses.asdict(spec))`'s hash of exactly this
  spec. The caller freezes it; the frozen hash is the guard, not an owner sign-off.
- `sources` other than a non-empty list or tuple of non-blank names (TypeError or ValueError): a
  bare 'jrdb' would become 'j', 'r', 'd', 'b' and slip past the registry's withholding (D21).
- an input outside the selection-input contract (`LookAheadSelection`), for each column X a rule
  reads in its segment or selection. By name: the final price and the result (`odds_final`,
  `won`, `win_dividend`, any `final_*`, the store's `win_odds` and `win_fav_rank`) and every label
  of Japan's availability config (`core.store.asof.load_availability`). Otherwise X is declared
  beside it: `X__source`, on every row one of ADMITTED_SOURCES (the board at t, or the races or
  runners card read through an as-of view at t), and `X__available_utc`, a timezone-aware UTC
  instant at or before the race's decision instant (`post_utc` minus the n minutes of
  `decision_time` "T-n") on every row holding a value, and on every row it stamps: a missing value
  stamped after t (a board update blanking a price) is a late observation, with or without a
  source; only a missing value with no stamp is genuinely missing (never selected). So a
  post-race table joined onto runners (the Win payout's `combination`, say), or a value published
  after t, is refused by its declaration whatever its name; an input declared from an admitted
  source and stamped by t is trusted as declared.
- a row on a holdout day (`core.eval.holdout.is_holdout_day`), or a spec or frame it cannot run
  (ValueError).

Stages, each for every rule:
1. Log loss. The benchmark is D4's calibrated market (answer 3): `models.v0.calibration`'s spline
   in ln q, refitted on each of D9's walk-forward folds (config/folds.yaml) on the fold's training
   rows of every race, so a market-wide favourite-longshot bias is no segment's claim. A claim is
   that benchmark plus a segment departure, fitted on the segment's training races by
   `models.condlogit` with the fold's ln p_cal as offset and scored out of fold on its test races:
   'beta', p proportional to p_cal^beta_s in the segment's races (a race-level segment; null 1);
   'omega', p proportional to p_cal * exp(gamma_s * [runner in segment]) (null 0). 'alpha' is
   refused with `AlphaMemberUnnamed` (answer 8: no C4 member is named). The per-race gain goes to
   #69's paired bootstrap over `cluster_unit` (config/scoring.yaml); `passed` needs the one-sided
   lower bound above 0 and the fold-weighted coefficient on the rule's `direction` side (answer 6).
2. Max statistic. #75's SPA and StepM over the rules whose claims were scored, on per-race losses
   (the benchmark's outside a rule's segment), in `cluster_unit` clusters ordered by race date,
   with the family's frozen `max_stat`. When SPA or StepM raises, the stage carries the reason as
   its "error" and finds no rule superior; the stages below are measured as ever.
3. Omega. #78's model on the rule's selected bets (answer 4) at the final Win odds, with odds-band
   fixed effects cut at the frozen `odds_band_cut_points` (answer 7). From one set of
   payout-weighted posterior draws of the bets' mean expected return per unit stake,
   mean((1 - t_i) * omega_i), each bet's win probability held below 1 so that no bet returns more
   than its odds: `expected_return` (mean and credible interval, break-even 1),
   `p_beats_takeout` (P that it exceeds 1: omega_s > 1/(1 - t)) and `omega_selected` (its mean
   over 1 - mean t_i). `predictive_roi_100`, the interval of the ROI of the next PREDICTIVE_BETS
   bets, is reported apart.
4. ROI. One unit to Win on each selected bet, paid at `win_dividend`; a loser pays 0.
A rule's bets are its segment's runners whose selection columns are all in [lo, hi] (inclusive,
None an open end, a missing value never selected) among the rows the walk-forward folds score:
the population log loss is scored on; training-only rows are never read for selection.

A measurement that cannot be made is NaN in its entry, which carries the reason as "error"; no
failure stops another rule or stage. Survivors (passed stage 1 and found superior by StepM) are a
diagnostic. Every rule's rung is S0, D19's default; rules move only through the leaderboard
(#449) and the owner. The holdout passed in is never opened.

Records: before StepM, one failed record per dropped variant the family does not hold yet and one
per rule (status 'failed' when its claim could not be scored); at the end a second per rule with
every measurement made as metrics (the predictive interval apart, as `predictive_roi_100` and its
level `predictive_roi_100_level`) and, in `notes`, each failed stage's reason ('log_loss',
'max_stat', 'omega', 'roi'), outside the variant identity. A record already in the registry is
not appended again, and a rerun whose frame cannot score a variant the family holds scored is
refused before any record.
`sources` defaults to licensed sources (fails closed, D21): pass the sources `data` was built from.
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
# The selection-input contract: each input X is declared by X__source and X__available_utc.
SOURCE_SUFFIX, AVAILABLE_SUFFIX = "__source", "__available_utc"
ADMITTED_SOURCES = ("board_at_t", "races", "runners")  # docs/specs/board-at-t.md, as-of.md
POST = "post_utc"  # the race's post, timezone-aware UTC; t is it minus the decision time
_DECISION_TIME = re.compile(r"T-([0-9]+)")
OMEGA_DRAWS = 4000  # posterior draws of the omega model: Monte Carlo precision only
PREDICTIVE_BETS = 100  # 08 §5.2 step 4: the posterior predictive ROI of the next 100 bets
PREDICTIVE = f"predictive_roi_{PREDICTIVE_BETS}"
PREDICTIVE_LEVEL = f"{PREDICTIVE}_level"  # its level, a metric apart so no level reads as an ROI
AVOID = "S0"  # D19's default rung: the study makes no rung decision
EVALUATED = "all_folds"  # the fold label of a measurement on the walk-forward test rows
UNSTATED_SOURCES = ("jra", "jrdb")  # fails closed: unnamed sources are treated as licensed
FAMILY_CONFIG = "selection_family"
_HYPOTHESIS = re.compile(r"([ELP][0-9]+)(?:-|$)")


class FamilyNotFrozen(RuntimeError):
    """The registry does not hold the family frozen with this exact spec; nothing was scored."""


class LookAheadSelection(ValueError):
    """A rule reads an input not known at the decision time, or one not declared as known then."""


class AlphaMemberUnnamed(ValueError):
    """An 'alpha' rule: the owner must name the C4 member whose segment weight is alpha_s."""


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
    # New in #88's PR, as is the dataclass: no earlier freeze of the family exists, so no
    # legacy-hash compatibility is kept.
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


def load_family(path_or_country: str | Path = COUNTRY) -> FamilySpec:
    """The family in a country's `config/selection_family.yaml`, or in the yaml at a path.

    No sign-off is read: `status`, `owner_signoff` (absent, null or an issue) and `deferred` are
    history for the reader, and the frozen hash is the guard. `dropped` (each entry's `variant`,
    in the file's order) becomes the spec's `dropped_variants`. Nothing is frozen.
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
    # The dropped variants become the spec's dropped_variants, so the frozen spec carries them and
    # run_study records each as a failed variant of the family (its family size counts them).
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


def _decision_minutes(decision_time: object) -> int:
    match = _DECISION_TIME.fullmatch(decision_time) if isinstance(decision_time, str) else None
    if match is None:
        raise ValueError(
            f"decision_time must be 'T-n', n minutes before the post: {decision_time!r}"
        )
    return int(match.group(1))


def _input_problem(data: pd.DataFrame, col: str, t: pd.Series) -> str | None:
    """Why the declared input `col` breaks the selection-input contract, or None."""
    source_col, stamp_col = col + SOURCE_SUFFIX, col + AVAILABLE_SUFFIX
    undeclared = [c for c in (source_col, stamp_col) if c not in data.columns]
    if undeclared:
        return f"undeclared: no {' and no '.join(undeclared)}"
    source, stamp = data[source_col], data[stamp_col]
    present = data[col].notna()
    other = source.notna() & ~source.isin(ADMITTED_SOURCES)
    if other.any():
        return f"from {sorted({str(s) for s in source[other]})}, not one of {ADMITTED_SOURCES}"
    if (present & source.isna()).any():
        return f"{int((present & source.isna()).sum())} value(s) with no source"
    if not isinstance(stamp.dtype, pd.DatetimeTZDtype):
        return f"{stamp_col} is not a timezone-aware instant ({stamp.dtype})"
    # A value unstamped, or a stamp (a blank value's too) after t or on a race without a post, is
    # late; only a missing value with no stamp is genuinely missing.
    late = ~(stamp <= t) & (present | stamp.notna())
    if late.any():
        return (
            f"{int(late.sum())} value(s), missing ones included, stamped after their race's "
            "decision time, or a value unstamped"
        )
    return None


def _check_declared(spec: FamilySpec, data: pd.DataFrame) -> None:
    """The selection-input contract (module docstring): every column a rule reads comes from an
    admitted source and was available by its race's decision instant, on every row."""
    if POST not in data.columns or not isinstance(data[POST].dtype, pd.DatetimeTZDtype):
        raise ValueError(f"data needs {POST}, each race's post as a timezone-aware UTC instant")
    t = data[POST] - pd.Timedelta(minutes=_decision_minutes(spec.decision_time))
    problems: dict[str, dict[str, str]] = {}
    for rule in spec.rules:
        for col in dict.fromkeys([*rule.segment, *rule.selection_cols]):
            why = _input_problem(data, col, t)
            if why is not None:
                problems.setdefault(rule.rule_id, {})[col] = why
    if problems:
        raise LookAheadSelection(
            f"rules read inputs not known at {spec.decision_time}: {problems}. Each input X is "
            f"declared by X{SOURCE_SUFFIX} (one of {ADMITTED_SOURCES}) and X{AVAILABLE_SUFFIX} "
            "(at or before the race's post minus the decision time)"
        )


def _check_sources(sources: object) -> tuple[str, ...]:
    """The sources `data` was built from, as the registry records them: a non-empty list or
    tuple of non-blank names. A bare str or bytes would become its letters, none of them a
    withheld source, and slip past the registry's guard and the summary's redaction (D21)."""
    if isinstance(sources, (str, bytes, bytearray)) or not isinstance(sources, (list, tuple)):
        raise TypeError(f"sources must be a list or tuple of source names, got {sources!r}")
    if not all(isinstance(name, str) for name in sources):
        raise TypeError(f"every source must be a name (str): {sources!r}")
    if not sources or not all(name.strip() for name in sources):
        raise ValueError(f"sources must name one or more sources, none blank: {sources!r}")
    return tuple(sources)


def _check_spec(spec: FamilySpec, data: pd.DataFrame, seed: object) -> list[str]:
    """Refuse a spec the study cannot run, before anything is scored; return the columns read."""
    if spec.benchmark != BENCHMARK:
        raise ValueError(f"benchmark must be {BENCHMARK!r}, got {spec.benchmark!r}")
    _decision_minutes(spec.decision_time)
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
    _check_declared(spec, data)
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


def _selected(frame: pd.DataFrame, rule: RuleSpec, rows: np.ndarray) -> np.ndarray:
    """The rule's bets among `rows`: its segment, every selection column present and in [lo, hi].
    Selection columns are read on those rows only, and a missing value (NaN, None or pd.NA) is
    never selected."""
    keep = np.flatnonzero(rows & _in_segment(frame, rule.segment))
    for col in rule.selection_cols:
        values = frame[col].iloc[keep]
        present = values.notna().to_numpy()
        keep, values = keep[present], values[present].to_numpy(dtype="float64")
        lo, hi = rule.thresholds[col]
        inside = np.ones(len(keep), dtype=bool)
        if lo is not None:
            inside &= values >= lo
        if hi is not None:
            inside &= values <= hi
        keep = keep[inside]
    out = np.zeros(len(frame), dtype=bool)
    out[keep] = True
    return out


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
    """Run the frozen family on development rows (module docstring) and report every rule.

    Raises, before anything is scored or written: TypeError or ValueError for malformed
    `sources`, `FamilyNotFrozen`, `AlphaMemberUnnamed`, `LookAheadSelection`, or ValueError for a
    spec or frame the study cannot run (a row on a holdout day included). Raises ValueError
    before the first record when the frame cannot score a variant the family holds scored.
    `holdout` is held, never opened.
    """
    if not isinstance(spec, FamilySpec):
        raise TypeError(f"spec must be a FamilySpec, got {type(spec).__name__}")
    if not isinstance(registry, Registry) or not isinstance(holdout, LockedHoldout):
        raise TypeError("registry must be a Registry and holdout a LockedHoldout")
    sources = _check_sources(sources)
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
    # The evaluation population: the rows the walk-forward folds score, as log loss does. Each
    # rule's bets are selected among them before anything is scored or written.
    evaluated = np.zeros(len(frame), dtype=bool)
    for fold in folds:
        evaluated[fold.test_idx] = True
    bets_of = {rid: _selected(frame, rule, evaluated) for rid, rule in rules.items()}
    identity = {
        "git_sha": _git_sha(),
        "data_builds": _frame_build(frame),
        "sources": sources,
        "fold_scheme": "walk_forward:" + json.dumps(dataclasses.asdict(scheme), sort_keys=True),
    }

    def record(rule: RuleSpec, metrics: list[FoldMetric], failed: bool, notes=None) -> RunRecord:
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
            notes=notes or None,
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
            if not np.isfinite([iv.mean, iv.lo, iv.hi]).all():
                raise ValueError(f"the gain's bootstrap interval is not finite: {iv}")
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
        test = "SPA"
        try:
            found = spa(bench, frame_losses, **common)
            max_stat |= {
                "spa_pvalue": found.pvalue,
                "spa_pvalue_lower": found.pvalue_lower,
                "spa_pvalue_upper": found.pvalue_upper,
                "family_size": found.family_size,
                "n_clusters": found.n_clusters,
            }
            test = "StepM"
            step = stepm(bench, frame_losses, level=spec.max_stat["level"], **common)
            max_stat["superior"] = [rid for rid in tested if rid in set(step.superior)]
        except Exception as exc:  # recorded with the stage; no rule is superior, the rest stands
            max_stat["error"] = f"{test} could not run: {exc!r}"
    # A diagnostic only: no stage below reads it.
    survivors = [rid for rid in rules if log_loss[rid]["passed"] and rid in max_stat["superior"]]

    # Stage 3: #78's omega model on each rule's selected bets, odds-band fixed effects. Every bet
    # of a fit is in the rule's segment, so the bets identify only their own mix's multiplier
    # (band and segment effects together); one fit per rule, so one rule's gaps touch no other.
    odds = frame[FINAL_ODDS].to_numpy(dtype=float)
    labels = np.array(_band_labels(spec.odds_band_cut_points))
    band = labels[np.digitize(odds, spec.odds_band_cut_points)]
    takeout = frame[TAKEOUT].to_numpy(dtype=float)
    nan = float("nan")
    omega = {}
    for rid, rule in rules.items():
        bet = bets_of[rid]
        n_bets = int(bet.sum())
        entry = {
            "n_bets": n_bets,
            "omega_selected": nan,
            "breakeven_omega": float(1.0 / (1.0 - takeout[bet].mean())) if n_bets else nan,
            "expected_return": dict.fromkeys(("mean", "low", "high", "level"), nan),
            "p_beats_takeout": nan,
            PREDICTIVE: [nan, nan, nan],
        }
        try:
            if not n_bets:
                raise ValueError("the rule selects no bet among the walk-forward test rows")
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
            er = model.expected_return(query)
            if not np.isfinite([er.mean, er.low, er.high]).all():
                raise ValueError(f"the posterior expected return is not finite: {er}")
            entry["expected_return"] = dataclasses.asdict(er)
            entry["p_beats_takeout"] = float(model.p_beats_takeout(query))
            entry["omega_selected"] = float(model.omega_mean(query))
            segment_effect = model.posterior_mean(query)  # alone, only the priors' split
            entry["ln_omega_by_band"] = {
                b: float(mu + segment_effect) for b, mu in model.band_effects.items()
            }
            interval = model.predictive_roi(query, PREDICTIVE_BETS)
            if not np.isfinite([interval.low, interval.high]).all():
                raise ValueError(f"the predictive interval is not finite: {interval}")
            entry[PREDICTIVE] = [interval.low, interval.high, interval.level]
        except Exception as exc:  # reported with the stage; the other stages stand
            entry["error"] = f"{exc!r}"
        omega[rid] = entry

    # Stage 4: ROI at the final Win dividend, selection at t; losers pay 0.
    dividend = frame[DIVIDEND].to_numpy(dtype=float)
    wins = won.astype(bool)
    roi = {}
    for rid in rules:
        bet = bets_of[rid]
        n_bets = int(bet.sum())
        unpaid = int((bet & wins & ~np.isfinite(dividend)).sum())
        entry = {"roi": nan, "n_bets": n_bets}
        if not n_bets:
            entry["error"] = "the rule selects no bet among the walk-forward test rows"
        elif unpaid:
            entry["error"] = f"{unpaid} selected winner(s) carry no {DIVIDEND}"
        else:
            entry["roi"] = float(dividend[bet & wins].sum() / n_bets - 1.0)
        roi[rid] = entry

    # Every measurement made, whatever the rule's verdicts; a NaN one is no metric, and its
    # stage entry carries the reason, which the record keeps in its notes by stage.
    outcome = []
    for rid, rule in rules.items():
        extra = []
        if rid in losses and "error" not in max_stat:
            superior = float(rid in max_stat["superior"])
            extra += _metric(EVALUATED, "stepm_superior", superior, max_stat["n_clusters"])
        entry, er = omega[rid], omega[rid]["expected_return"]
        n_bets = entry["n_bets"]
        low, high, level = entry[PREDICTIVE]
        extra += _metric(EVALUATED, "omega_selected", entry["omega_selected"], n_bets)
        extra += _metric(EVALUATED, "expected_return", er["mean"], n_bets, er["low"], er["high"])
        extra += _metric(EVALUATED, PREDICTIVE, (low + high) / 2, n_bets, low, high)
        extra += _metric(EVALUATED, PREDICTIVE_LEVEL, level, n_bets)
        extra += _metric(EVALUATED, "p_beats_takeout", entry["p_beats_takeout"], n_bets)
        extra += _metric(EVALUATED, "roi_win_dividend", roi[rid]["roi"], roi[rid]["n_bets"])
        stages = {"log_loss": log_loss[rid], "omega": entry, "roi": roi[rid]}
        notes = {stage: e["error"] for stage, e in stages.items() if "error" in e}
        if "error" in max_stat and rid in tested:
            notes["max_stat"] = max_stat["error"]
        outcome.append(record(rule, metrics[rid] + extra, rid not in losses, notes))
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
        rungs=dict.fromkeys(rules, AVOID),
    )
