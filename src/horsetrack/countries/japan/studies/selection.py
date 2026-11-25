"""The pre-registered Family P selection study (#88; 08 §5.2 end to end, §5.3, §8 A6).

A family of positive-edge rules (08 §5.1 Family P) goes through 08 §5.2's stages in order, on the
development rows the caller passes (one row per starter, `split_off`'s development side):

0. Freeze. `run_study` raises `FamilyNotFrozen` before it scores or writes anything unless the
   registry holds the family frozen with exactly this spec: the hash `Registry.freeze_family`
   gives `dataclasses.asdict(spec)`. The owner approves the family and the caller freezes it; the
   study never freezes it itself.
1. Log loss. Each rule's probability claim on all runners of its segment against the benchmark
   (`final_price`: the normalised final Win price), out of fold on D9's walk-forward folds
   (config/folds.yaml), the claim refitted on each fold's training rows:
   - 'beta': p proportional to (1/O_final)^beta_s on the segment's races (#77's `segment_beta`;
     the segment must be race-level);
   - 'omega': p proportional to q * exp(gamma_s * [runner in segment]) in each race holding a
     segment runner (`models.condlogit`, offset ln q): the segment's runners win more or less
     often than their price says;
   - 'alpha' is refused with `AlphaMemberUnnamed`: alpha_s is a C4 member's segment weight (#87's
     `<member>:<segment>` term) and neither RuleSpec nor the data names the member.
   The per-race gain (the benchmark's winner log loss minus the rule's) goes to #69's paired
   bootstrap over `cluster_unit` at D7's level and replicates (config/scoring.yaml). A claim
   passes when the one-sided lower bound is above 0, which says the segment departs from the
   final price in either direction: RuleSpec carries no direction.
2. Max statistic. #75's SPA and StepM over the rules scored at stage 1 (their registry records
   are appended first, so the family check sees them), on per-race losses of every scored race
   (the benchmark's outside a rule's segment), in `cluster_unit` clusters ordered by race date,
   with the family's frozen `max_stat` settings.
3. Omega. #78's hierarchical model, odds-band fixed effects (`final_odds_band`) and one random
   effect per segment column of the family, on every runner at the final Win odds: per rule,
   P(omega_s > 1/(1 - t) | data) for its segment and the predictive ROI of its next
   PREDICTIVE_BETS bets (08 §5.2 step 4). Reported; not a gate.
4. ROI, for the survivors only (passed stage 1 and found superior by StepM): one unit to Win on
   each runner of the segment whose selection columns are all in [lo, hi] (inclusive, None an
   open end, a missing value never selected), paid at `win_dividend`. Before any is computed, a
   rule whose segment or selection names a look-ahead column (`odds_final`, `won`,
   `win_dividend`, any `final_*`) refuses the run with `LookAheadSelection`.

Rungs: S1 (shadow) for a survivor, S0 (avoid) for every other rule; nothing goes above S1 and no
record is adopted. The holdout passed in is never opened.

Records: one per rule (variant = rule_id, hypothesis id = its leading P/E/L id) before StepM,
status 'failed' when its claim could not be scored, and a second one at the end carrying every
stage's metrics. A record already in the registry is not appended again. `sources` defaults to a
licensed source (fails closed, D21): pass the sources `data` was built from.

Memory: one column-scoped copy of `data`; every fold works on positions into it.
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
from horsetrack.core.eval.holdout import LockedHoldout, _git_sha
from horsetrack.core.eval.registry import FoldMetric, Registry, RunRecord, _spec_hash
from horsetrack.core.eval.scoring import gain, per_race_log_loss
from horsetrack.core.eval.segment_roi import fit_segment_roi
from horsetrack.core.eval.snooping import spa, stepm
from horsetrack.countries.japan.models.condlogit import fit_conditional_logit
from horsetrack.countries.japan.studies.calibration import (
    bootstrap_settings,
    devig_normalised,
    devig_power,
    segment_beta,
)

COUNTRY = "japan"
TICKET = 88
CLAIMS = ("beta", "alpha", "omega")
BENCHMARK = "final_price"
MAX_STAT_KEYS = frozenset({"level", "block_size", "n_boot"})
FINAL_ODDS, DIVIDEND, TAKEOUT = "odds_final", "win_dividend", "takeout"
ODDS_BAND = "final_odds_band"  # the omega model's favourite-longshot fixed effects
BASE_COLUMNS = ["race_id", "race_date", "won", FINAL_ODDS, DIVIDEND, TAKEOUT, ODDS_BAND]
LOOK_AHEAD = frozenset({FINAL_ODDS, DIVIDEND, "won"})
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


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    rules: list[RuleSpec]
    decision_time: str
    cluster_unit: str
    benchmark: str
    max_stat: dict


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
    owner approved it; `proposed=True` reads a proposal for review. `status`, `dropped` (variants
    to record as failed before the freeze) and `deferred` are for the reader. Nothing is frozen.
    """
    if isinstance(path_or_country, Path) or str(path_or_country).endswith((".yaml", ".yml")):
        raw = yaml.safe_load(Path(path_or_country).read_text(encoding="utf-8")) or {}
    else:
        raw = load_country_yaml(str(path_or_country), FAMILY_CONFIG)
    fields = [f.name for f in dataclasses.fields(FamilySpec)]
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
            )
        )
    return FamilySpec(**{**{name: raw[name] for name in fields}, "rules": rules})


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


def _check_spec(spec: FamilySpec, data: pd.DataFrame, seed: object) -> list[str]:
    """Refuse a spec the study cannot run, before anything is scored; return the columns read."""
    if spec.benchmark != BENCHMARK:
        raise ValueError(f"benchmark must be {BENCHMARK!r}, got {spec.benchmark!r}")
    if not isinstance(spec.max_stat, dict) or set(spec.max_stat) != MAX_STAT_KEYS:
        raise ValueError(f"max_stat must give exactly {sorted(MAX_STAT_KEYS)}: {spec.max_stat!r}")
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
    columns = list(dict.fromkeys(columns))
    missing = [col for col in columns if col not in data.columns]
    if missing:
        raise ValueError(f"data lacks column(s) {missing}")
    return columns


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


def _look_ahead(col: str) -> bool:
    return col in LOOK_AHEAD or col.startswith(LOOK_AHEAD_PREFIX)


def _race_ll(p: pd.Series, frame: pd.DataFrame, rows: np.ndarray) -> pd.Series:
    """Winner log loss per race of `p` on `rows` (per_race_log_loss renormalises each race)."""
    part = frame.iloc[rows]
    p = pd.Series(np.asarray(p, dtype=float), index=part.index)
    no = pd.Series(False, index=part.index)
    return per_race_log_loss(p, part["won"].astype(bool), part["race_id"], no).per_race


# --------------------------------------------------------------------------- stage 1


def _claim_oof(frame, rule, folds, ln_q, codes) -> tuple[np.ndarray, np.ndarray, dict]:
    """Out-of-fold probabilities of the rule's claim on its scored rows, and the per-fold fits."""
    member = _in_segment(frame, rule.segment)
    holds = pd.Series(member).groupby(codes).transform("any").to_numpy(dtype=bool)
    race_ids, won = frame["race_id"].to_numpy(), frame["won"].to_numpy()
    rows, probs, fits = [], [], {}
    for fold in folds:
        test = fold.test_idx[holds[fold.test_idx]]
        if not len(test):
            continue
        train = fold.train_idx
        if rule.claim == "beta":
            fit_rows = frame[["race_id", FINAL_ODDS, "won"]].iloc[train]
            fit_rows = fit_rows.rename(columns={FINAL_ODDS: "odds"})
            fit_rows["segment"] = np.where(member[train], "in", None)
            table = segment_beta(fit_rows, "segment")
            if table.empty:
                raise ValueError(f"no training race of the {fold.test_start} fold is in segment")
            coef, n = float(table["beta"].iloc[0]), int(table["n_races"].iloc[0])
            p = devig_power(frame[FINAL_ODDS].iloc[test], frame["race_id"].iloc[test], coef)
            p = p.to_numpy()
        else:
            train = train[holds[train]]
            x = member[train].astype(float)[:, None]
            fit = fit_conditional_logit(x, race_ids[train], won[train], offset=ln_q[train])
            coef, n = float(fit.coef[0]), int(fit.n_races)
            p = np.exp(ln_q[test] + coef * member[test])
        rows.append(test)
        probs.append(p)
        fits[fold.test_start.isoformat()] = (coef, n)
    if not rows:
        raise ValueError("no walk-forward fold scores a race of the segment")
    return np.concatenate(rows), np.concatenate(probs), fits


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

    Raises `FamilyNotFrozen` (before anything is scored or written), `AlphaMemberUnnamed` or
    ValueError for a spec or frame the study cannot run (also before anything is scored), and
    `LookAheadSelection` at the ROI stage. `holdout` is held, never opened.
    """
    if not isinstance(spec, FamilySpec):
        raise TypeError(f"spec must be a FamilySpec, got {type(spec).__name__}")
    if not isinstance(registry, Registry) or not isinstance(holdout, LockedHoldout):
        raise TypeError("registry must be a Registry and holdout a LockedHoldout")
    frozen_hash = _check_frozen(spec, registry)
    columns = _check_spec(spec, data, seed)
    frame = data[columns].reset_index(drop=True)
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
            "decision_time": spec.decision_time,
            "cluster_unit": unit,
            "max_stat": dict(spec.max_stat),
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

    def append(records: list[RunRecord]) -> None:
        existing = registry.records(spec.family_id)
        for r in records:
            if r not in existing:
                registry.append(r)

    # Stage 1: each rule's probability claim against the final price, out of fold.
    q = devig_normalised(frame[FINAL_ODDS], frame["race_id"])
    ln_q = np.log(q.to_numpy())
    scored = np.sort(np.concatenate([f.test_idx for f in folds] or [np.zeros(0, int)]))
    bench = _race_ll(q.iloc[scored], frame, scored) if len(scored) else pd.Series(dtype=float)
    log_loss, losses, metrics = {}, {}, {}
    for rid, rule in rules.items():
        try:
            rows, p, fits = _claim_oof(frame, rule, folds, ln_q, codes)
            ll = _race_ll(p, frame, rows)
            g = gain(ll, bench)
            iv = paired_bootstrap(
                g.per_race, races["cluster"], n_boot=boot["n_boot"], level=boot["level"], seed=seed
            )
        except Exception as exc:  # every failure is a registry record, never dropped
            log_loss[rid] = {"claim": rule.claim, "passed": False, "error": f"{exc!r}"}
            metrics[rid] = []
            continue
        log_loss[rid] = {
            "claim": rule.claim,
            "passed": bool(iv.lo > 0),
            "delta_ll": iv.mean,
            "lo": iv.lo,
            "hi": iv.hi,
            "level": boot["level"],
            "n_races": g.n_common,
            "n_clusters": iv.n_clusters,
            "coef_by_fold": {fold: coef for fold, (coef, _) in fits.items()},
        }
        coef_name = "beta" if rule.claim == "beta" else "gamma"
        metrics[rid] = _metric("all_folds", "delta_ll", iv.mean, g.n_common, iv.lo, iv.hi)
        for fold, (coef, n) in fits.items():
            metrics[rid] += _metric(fold, coef_name, coef, n)
        column = bench.copy()  # the final price's loss outside the rule's races
        column.loc[g.per_race.index] = ll.loc[g.per_race.index]
        losses[rid] = column
    append([record(rules[rid], metrics[rid], rid not in losses) for rid in rules])

    # Stage 2: SPA ("does any rule beat the final price") and StepM ("which ones").
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

    # Stage 3: the hierarchical omega model with odds-band fixed effects, every runner.
    seg_cols = list(dict.fromkeys(col for rule in spec.rules for col in rule.segment))
    bets = frame[[FINAL_ODDS, "won", ODDS_BAND, TAKEOUT, *seg_cols]]
    omega, model, model_error = {}, None, None
    try:
        model = fit_segment_roi(
            bets.rename(columns={FINAL_ODDS: "odds"}),
            segment_cols=seg_cols,
            odds_band_col=ODDS_BAND,
            takeout_col=TAKEOUT,
            seed=seed,
            draws=OMEGA_DRAWS,
        )
    except Exception as exc:  # reported with the stage; the other stages stand
        model_error = f"{exc!r}"
    for rid, rule in rules.items():
        member = _in_segment(frame, rule.segment)
        entry = {"n_runners": int(member.sum()), "p_beats_takeout": float("nan")}
        entry["breakeven_omega"] = float(1.0 / (1.0 - frame[TAKEOUT][member].mean()))
        try:
            if model is None:
                raise RuntimeError(f"the omega model was not fitted: {model_error}")
            query = {col: str(level) for col, level in rule.segment.items()}
            interval = model.predictive_roi(query, PREDICTIVE_BETS)
            entry |= {
                "p_beats_takeout": model.p_beats_takeout(query),
                "ln_omega_segment": model.posterior_mean(query),
                f"predictive_roi_{PREDICTIVE_BETS}": [interval.low, interval.high, interval.level],
            }
        except Exception as exc:
            entry["error"] = f"{exc!r}"
        omega[rid] = entry

    # Stage 4: ROI at the final Win dividend, survivors only, selection at t.
    offending = {
        rid: cols
        for rid, rule in rules.items()
        if (cols := [col for col in [*rule.segment, *rule.selection_cols] if _look_ahead(col)])
    }
    if offending:
        raise LookAheadSelection(
            f"rules select on look-ahead columns {offending}: a bet is selected from the board "
            "at t or from as-of features only (08 §5.2, selection hygiene)"
        )
    paid = frame["won"].to_numpy(dtype=float) * frame[DIVIDEND].to_numpy(dtype=float)
    roi = {}
    for rid in survivors:
        bet = _selected(frame, rules[rid])
        n_bets = int(bet.sum())
        roi[rid] = {
            "roi": float(paid[bet].sum() / n_bets - 1.0) if n_bets else float("nan"),
            "n_bets": n_bets,
        }

    rungs = {rid: SHADOW if rid in survivors else AVOID for rid in rules}
    outcome = []
    for rid, rule in rules.items():
        extra = []
        if rid in losses:
            superior = float(rid in max_stat["superior"])
            extra += _metric("all_folds", "stepm_superior", superior, max_stat.get("n_clusters", 0))
            p = omega[rid]["p_beats_takeout"]
            extra += _metric("development", "p_beats_takeout", p, omega[rid]["n_runners"])
            if rid in roi:
                value, n_bets = roi[rid]["roi"], roi[rid]["n_bets"]
                extra += _metric("development", "roi_win_dividend", value, n_bets)
        outcome.append(record(rule, metrics[rid] + extra, rid not in losses))
    append(outcome)

    return StudyReport(
        family_size=registry.family_size(spec.family_id),
        stages={"log_loss": log_loss, "max_stat": max_stat, "omega": omega, "roi": roi},
        survivors=survivors,
        p_beats_takeout={rid: float(omega[rid]["p_beats_takeout"]) for rid in rules},
        rungs=rungs,
    )
