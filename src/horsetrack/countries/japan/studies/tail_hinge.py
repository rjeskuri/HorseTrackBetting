"""A tail slope continuous at decimal 51.0 for the final Win price (#294; epic #72), beside
#285's two-slope tail beta.

#285's model adds a slope above `min_odds` but not a hinge: its log-weight jumps by
increment * ln(1/min_odds) at the threshold, so its increment carries a level step as well as a
slope change. This module fits the conditional logit whose log-weight is continuous at `min_odds`

    P(i wins) proportional to exp(beta * x_i + gamma * h_i),  x = ln(1 / O),  h = max(0, k - x),

with the knot k = ln(1 / min_odds) (so h = 0 at odds up to `min_odds`, and the distance past the
knot above it). `beta` is the body slope on x; the slope on x past the knot is `tail_beta` = beta -
gamma, because gamma * (k - x) = gamma * k - gamma * x. A large pure level step is partly absorbed
as a negative gamma (the hinge cannot carry a jump), so gamma is read beside #285's increment.
Inputs and the handling of unpriced runners, winnerless races and dead heats are
`calibration.fit_power_beta`'s; the likelihood, score and information are `tail_beta`'s on the hinge
column. Standard errors are the square roots of the diagonal of the inverse observed information.

The study (`run_study`; CLI `python -m horsetrack.countries.japan.studies.tail_hinge [--out DIR]`)
is a thin I/O layer over the pure `compute_study(frame, spec)`, in its own registry family. It
reuses #77's owner inputs, development rows, fold scheme, floor and bootstrap settings. Order:
build the spec, freeze the family in the run registry (`FamilyAlreadyFrozen` on another spec stops
the run), only then read the development rows (`calibration.read_development`: holdout days
dropped; the holdout is never opened), score, append one registry record per variant (failures
included) and write the report through `write_report`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from horsetrack.core import paths
from horsetrack.core.eval.folds import FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha
from horsetrack.core.eval.registry import FoldMetric, RunRecord, default_registry
from horsetrack.core.eval.reports import write_report
from horsetrack.core.eval.scoring import gain, per_race_log_loss
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START
from horsetrack.countries.japan.studies import calibration, tail_beta
from horsetrack.countries.japan.studies.calibration import Study, Variant

NEWTON_MAX_STEPS = calibration.NEWTON_MAX_STEPS
NEWTON_STEP_TOL = calibration.NEWTON_STEP_TOL

COUNTRY = calibration.COUNTRY
TICKET = 294
SOURCE = calibration.SOURCE  # the store's JRA results, from netkeiba; no JRDB, no JRA-VAN
FAMILY_ID = "japan-tail-hinge-final-win"
HYPOTHESIS = "H1"  # the label in the frozen spec (its bytes must not change)
RECORD_HYPOTHESIS = "E1"  # the registry takes E, L or P ids; the tail is 08 §4 E1's tail measure
FIT = "conditional_logit_ln_inv_odds_hinge_slope"
IN_SAMPLE = calibration.IN_SAMPLE
SUBSETS = calibration.SUBSETS
WALD_Z = calibration.BETA_CI_Z
DECISIONS = {
    **{k: tail_beta.DECISIONS[k] for k in ("D6", "D7", "D9", "D10", "D11", "D12", "D14", "D15")},
    "#77 (4)": tail_beta.DECISIONS["#77 (4)"],
    "#77 (5)": tail_beta.DECISIONS["#77 (5)"],
    "#77 (c)": tail_beta.DECISIONS["#77 (c)"],
    "#285": "the two-slope tail beta (a step at 51.0) is the comparator beside normalised and "
    "power-beta; its family `japan-tail-beta-final-win` is untouched (#285)",
    "#294": "the tail slope is fitted as a hinge continuous at 51.0 in its own family, frozen "
    "before any development race is scored; the holdout is not opened (ticket #294, epic #72)",
}


@dataclass(frozen=True)
class TailHingeFit:
    """The hinge fit: body slope, hinge coefficient, the slope on x past the knot (beta - gamma),
    their standard errors (tail_beta_se from the full covariance) and the races fitted."""

    beta: float
    se: float
    gamma: float
    gamma_se: float
    tail_beta: float
    tail_beta_se: float
    n_races: int


# --------------------------------------------------------------------------- the fit


def _design(
    odds: pd.Series, race_id: pd.Series, won: pd.Series, min_odds: float
) -> tail_beta._Design:
    """`tail_beta`'s design with its second column the hinge max(0, k - x), k = ln(1/min_odds).

    Priced runners of the races with a priced winner, sorted by race. ValueError when no race has
    a priced winner or no fitted runner is priced strictly above `min_odds`."""
    calibration._check_aligned(odds, race_id, won)
    if won.isna().any():
        raise ValueError("won has missing values")
    cleaned, ok, codes, races = calibration._priced(odds, race_id)
    won_priced = won.to_numpy(dtype=bool) & ok
    winners = np.bincount(codes, weights=won_priced, minlength=len(races))
    rows = np.flatnonzero(ok & (winners[codes] > 0))
    if len(rows) == 0:
        raise ValueError("no race with a priced winner to fit")
    rows = rows[np.argsort(codes[rows], kind="stable")]
    race = codes[rows]
    starts = np.flatnonzero(np.r_[True, race[1:] != race[:-1]])
    price = cleaned[rows]
    if not (price > min_odds).any():
        raise ValueError(f"no fitted runner is priced above {min_odds:g}: the hinge is not set")
    x = -np.log(price)
    hinge = np.maximum(0.0, -np.log(min_odds) - x)
    won_rows = won_priced[rows]
    return tail_beta._Design(
        x=x,
        xt=hinge,
        starts=starts,
        sizes=np.diff(np.r_[starts, len(rows)]),
        winners=winners[race[starts]],
        x_won=np.array([x[won_rows].sum(), hinge[won_rows].sum()]),
    )


def fit_tail_hinge(
    odds: pd.Series, race_id: pd.Series, won: pd.Series, *, min_odds: float | None = None
) -> TailHingeFit:
    """The maximum-likelihood body slope and hinge coefficient of the conditional logit with
    log-weight beta x + gamma max(0, ln(1/min_odds) - x), x = ln(1/O), over each race's priced
    runners (`min_odds` None: `tail_min_odds` of config/calibration.yaml, 51.0).

    Only races with a priced winner enter, and `n_races` counts them; unpriced runners leave their
    race's choice set; a dead heat counts each priced co-winner. The log-likelihood is concave:
    Newton steps, halved whenever one would lower it, run until a step is under 1e-10.

    Raises ValueError when no race has a priced winner, no fitted runner is priced strictly above
    `min_odds` (the hinge is identically 0) or the information is singular, and RuntimeError when
    Newton does not converge (for example when no tail runner ever wins).
    """
    d = _design(odds, race_id, won, tail_beta._tail_threshold(min_odds))
    theta = np.array([1.0, 0.0])
    loglik, score, info = tail_beta._loglik(theta, d)
    for _ in range(NEWTON_MAX_STEPS):
        if not tail_beta._identified(info):
            raise ValueError("the slopes are not identified: the fitted runners do not vary enough")
        step = np.linalg.solve(info, score)
        while True:
            trial = tail_beta._loglik(theta + step, d)
            if trial[0] >= loglik or np.abs(step).max() < NEWTON_STEP_TOL:
                break
            step = step / 2.0
        theta = theta + step
        loglik, score, info = trial
        if np.abs(step).max() < NEWTON_STEP_TOL:
            break
    else:
        raise RuntimeError(f"the hinge fit did not converge in {NEWTON_MAX_STEPS} Newton steps")
    if not (np.isfinite(theta).all() and tail_beta._identified(info)):
        raise RuntimeError(f"the hinge fit stopped at theta={theta.tolist()}")
    cov = np.linalg.inv(info)
    return TailHingeFit(
        beta=float(theta[0]),
        se=float(np.sqrt(cov[0, 0])),
        gamma=float(theta[1]),
        gamma_se=float(np.sqrt(cov[1, 1])),
        tail_beta=float(theta[0] - theta[1]),
        tail_beta_se=float(np.sqrt(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])),
        n_races=len(d.starts),
    )


def devig_hinge(
    odds: pd.Series, race_id: pd.Series, fit: TailHingeFit, *, min_odds: float
) -> pd.Series:
    """p_i proportional to exp(beta x_i + gamma max(0, ln(1/min_odds) - x_i)) over each race's
    priced runners; NaN where unpriced."""
    cleaned, ok, codes, races = calibration._priced(odds, race_id)
    x = -np.log(cleaned[ok])
    weight = np.exp(fit.beta * x + fit.gamma * np.maximum(0.0, -np.log(min_odds) - x))
    totals = np.bincount(codes[ok], weights=weight, minlength=len(races))
    return calibration._on_index(weight / totals[codes[ok]], ok, odds.index)


# --------------------------------------------------------------------------- spec


def bootstrap_settings() -> dict:
    """D7's bootstrap from config/scoring.yaml, with this study's fixed seed."""
    return {**calibration.bootstrap_settings(), "seed": TICKET}


def build_spec(owner: dict, scheme: FoldScheme, bootstrap: dict, floor: float) -> dict:
    """The family's pre-registered spec, frozen in the registry before anything is scored."""
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypotheses": {
            HYPOTHESIS: "the tail slope of the final Win price as a hinge continuous at the tail "
            "threshold: the conditional-logit body slope on ln(1/odds), the hinge coefficient on "
            "max(0, ln(1/threshold) - ln(1/odds)) and the slope past the knot, overall, by year "
            "and with and without floor-bound races; out-of-fold ΔLL against normalised, "
            "power-beta and #285's two-slope tail beta",
        },
        "target": "the race's winner (Win pool) at the final Win odds",
        "model": "P(i wins) proportional to exp(beta x_i + gamma h_i), x_i = ln(1/O_i), "
        "h_i = max(0, ln(1/min_odds) - x_i), over each race's priced runners; the log-weight is "
        "continuous at O = min_odds and the slope past the knot is beta - gamma",
        "comparators": {
            "normalised": "q_i = (1/O_i) / sum_j (1/O_j)",
            "power": "p_i proportional to (1/O_i)^beta, beta fitted on each fold's training rows "
            "(calibration.fit_power_beta)",
            "tail_beta": "#285's two-slope model, exp(beta x + increment x 1[O > min_odds]), "
            "refitted on each fold's training rows (tail_beta.fit_tail_beta)",
        },
        "delta_ll": "per scored race, the comparator's winner log loss minus the hinge model's "
        "(core.eval.scoring.gain), out of fold; meeting-clustered paired bootstrap bounds",
        "fit": "conditional logit MLE; se = diagonal of the inverse observed information; "
        "tail_beta_se from the full covariance; races with a priced winner; a dead heat counts "
        "each co-winner",
        "ci": "95% two-sided Wald, estimate +/- 1.96 se, for beta, gamma and tail_beta",
        "walk_forward": "the hinge model, #285's tail beta and power-beta are refitted on each "
        "fold's training rows and scored on its test races; the per-fold slopes are reported",
        "in_sample": "the slopes overall and by calendar year: one fit per group on every "
        "development race of the subset, descriptive, no fold",
        "adoption": "none: the owner decides whether a tail slope enters the baseline",
        "score": "winner log loss per race (core.eval.scoring.per_race_log_loss)",
        "rows": "priced starters of run races with a priced winner; development days only",
        "subsets": {
            "all_races": "every development race",
            "no_floor_bound_races": "races holding a floor-bound runner left out of fitting and "
            "scoring alike",
        },
        "tail": {"min_odds": owner["tail_min_odds"], "rule": "a runner priced above min_odds"},
        "floor": {"odds": floor, "rule": "a priced runner at or under odds + 1e-9 (jra.yaml)"},
        "fold_scheme": dataclasses.asdict(scheme),
        "bootstrap": bootstrap,
        "price_era_start": PRICE_ERA_START.isoformat(),
        "decisions": DECISIONS,
        "owner_signoff": owner["owner_signoff"],
        "sources": [SOURCE],
    }


# --------------------------------------------------------------------------- the pure study


def _wald(fold: str, metric: str, value: float, se: float, n: int) -> FoldMetric:
    return calibration._metric(fold, metric, value, n, value - WALD_Z * se, value + WALD_Z * se)


def _slope_metrics(fit: TailHingeFit, fold: str, suffix: str = "") -> list[FoldMetric]:
    n = fit.n_races
    return [
        _wald(fold, f"body_beta{suffix}", fit.beta, fit.se, n),
        _wald(fold, f"hinge_gamma{suffix}", fit.gamma, fit.gamma_se, n),
        _wald(fold, f"tail_beta{suffix}", fit.tail_beta, fit.tail_beta_se, n),
    ]


def _slope_row(fit: TailHingeFit, **label) -> dict:
    row = {**label, "races": fit.n_races}
    for name, value, se in (
        ("body β", fit.beta, fit.se),
        ("hinge γ", fit.gamma, fit.gamma_se),
        ("tail β", fit.tail_beta, fit.tail_beta_se),
    ):
        row |= {name: value, f"{name} lower 95%": value - WALD_Z * se}
        row |= {f"{name} upper 95%": value + WALD_Z * se}
    return row


def _in_sample(rows: pd.DataFrame, subset: str, min_odds: float) -> tuple[list, dict]:
    """The slopes on every development race of the subset, and on each calendar year's races.

    A year in which no tail runner won has no maximum (the hinge coefficient diverges): it is a
    table row saying so, with no metric, and the other years stand. The overall fit must exist."""
    fit_rows = rows[["race_id", "odds", "won"]]
    year = rows["race_date"].dt.year
    metrics, table = [], []
    for y, part in [(None, fit_rows)] + [
        (int(y), fit_rows.loc[year.eq(y)]) for y in sorted(year.unique())
    ]:
        label = "all years" if y is None else str(y)
        tail = (part["odds"] > min_odds).to_numpy()
        counts = {"tail runners": int(tail.sum()), "tail winners": int(part["won"][tail].sum())}
        try:
            fit = fit_tail_hinge(part["odds"], part["race_id"], part["won"], min_odds=min_odds)
        except (ValueError, RuntimeError) as exc:
            if y is None:
                raise
            table.append(
                {"subset": subset, "year": label, "races": part["race_id"].nunique()}
                | counts
                | {"status": f"not identified: {exc}"}
            )
            continue
        metrics += _slope_metrics(fit, "development", "" if y is None else f"[year={y}]")
        table.append(_slope_row(fit, subset=subset, year=label) | counts)
    return metrics, {"slopes": table}


def _walk_forward(rows: pd.DataFrame, spec: dict, subset: str) -> tuple[list, dict]:
    """Out-of-fold winner log loss of the hinge model, normalised, power-beta and #285's tail
    beta, ΔLL of the hinge model against each with meeting-clustered bounds, the per-fold slopes
    and the tail's observed vs expected winners under the hinge model."""
    min_odds = spec["tail"]["min_odds"]
    folds = walk_forward(rows["race_date"], scheme=FoldScheme.from_mapping(spec["fold_scheme"]))
    if not folds:
        raise ValueError("the fold scheme gives no walk-forward fold on these dates")
    scored = rows.iloc[np.concatenate([f.test_idx for f in folds])]
    race = scored["race_id"].to_numpy()
    clusters = scored["meeting_id"].groupby(race).first()
    fit_rows = rows[["race_id", "odds", "won"]]
    parts = {"hinge": [], "tail_beta": [], "power": []}
    metrics, fold_rows = [], []
    for fold in folds:
        train, test = fit_rows.iloc[fold.train_idx], fit_rows.iloc[fold.test_idx]
        args = (train["odds"], train["race_id"], train["won"])
        fit = fit_tail_hinge(*args, min_odds=min_odds)
        step = tail_beta.fit_tail_beta(*args, min_odds=min_odds)
        power = calibration.fit_power_beta(*args)
        parts["hinge"].append(devig_hinge(test["odds"], test["race_id"], fit, min_odds=min_odds))
        parts["tail_beta"].append(
            tail_beta.devig_tail(test["odds"], test["race_id"], step, min_odds=min_odds)
        )
        parts["power"].append(calibration.devig_power(test["odds"], test["race_id"], power.beta))
        label = fold.test_start.isoformat()
        metrics += _slope_metrics(fit, label, f"[test_year={fold.test_start.year}]")
        fold_rows.append(
            _slope_row(fit, subset=subset, fold=label)
            | {"train to": fold.train_end, "test to": fold.test_end}
            | {"test races": test["race_id"].nunique(), "power β": power.beta}
            | {"#285 increment": step.increment}
        )
    probs = {name: pd.concat(p) for name, p in parts.items()}
    probs["normalised"] = calibration.devig_normalised(scored["odds"], scored["race_id"])
    no = pd.Series(False, index=scored.index)
    loss = {
        name: per_race_log_loss(p, scored["won"], scored["race_id"], no).per_race
        for name, p in probs.items()
    }
    names = {"normalised": "normalised", "power": "power-β", "tail_beta": "#285 tail β"}
    for key, ll in loss.items():
        metrics.append(
            calibration._metric("all_folds", f"mean_log_loss[{key}]", ll.mean(), len(ll))
        )
    headline = {"subset": subset, "races": len(loss["hinge"])}
    headline |= {"mean log loss (hinge)": loss["hinge"].mean()}
    for key, name in names.items():
        g = gain(loss["hinge"], loss[key])
        iv = calibration._bounds(g.per_race, clusters, spec)
        metrics.append(
            calibration._metric(
                "all_folds", f"delta_ll_vs_{key}", iv.mean, g.n_common, iv.lo, iv.hi
            )
        )
        headline |= {f"ΔLL vs {name}": iv.mean, f"lower ({name})": iv.lo, f"upper ({name})": iv.hi}
        headline |= {f"mean log loss ({name})": loss[key].mean()}
    headline["meetings"] = iv.n_clusters
    tail = (scored["odds"] > min_odds).to_numpy()
    observed = scored["won"].to_numpy(dtype=float)[tail].sum()
    expected = probs["hinge"].to_numpy()[tail].sum()
    metrics += [
        calibration._metric("all_folds", "tail_observed_winners", observed, tail.sum()),
        calibration._metric("all_folds", "tail_expected_winners", expected, tail.sum()),
    ]
    tail_row = {"subset": subset, "model": "hinge", "tail runners": int(tail.sum())}
    tail_row |= {"observed winners": observed, "expected winners": expected}
    tail_row |= {"O / E": observed / expected if expected > 0 else None}
    return metrics, {"folds": fold_rows, "headline": [headline], "tail": [tail_row]}


def compute_study(frame: pd.DataFrame, spec: dict) -> Study:
    """Every variant and table of the frozen `spec` on `frame` (development_frame's shape).

    For each subset (every race; races without a floor-bound runner, left out of fitting and
    scoring alike): the walk-forward hinge model, scored out of fold with ΔLL against normalised,
    power-beta and #285's tail beta (meeting-clustered bounds), and the in-sample slopes overall
    and by year, each with a 95% Wald CI. Pure: no I/O.
    """
    floor = calibration.flag_floor_bound(frame["odds"], floor=spec["floor"]["odds"])
    floor_race = floor.groupby(frame["race_id"].to_numpy()).transform("any").to_numpy()
    study = Study(counts={"floor_bound_runners": int(floor.sum())})
    study.counts["races_holding_a_floor_bound_runner"] = int(frame["race_id"][floor_race].nunique())
    columns = ["race_id", "race_date", "meeting_id", "odds", "won"]
    scheme = "walk_forward:" + json.dumps(spec["fold_scheme"], sort_keys=True)
    min_odds = spec["tail"]["min_odds"]
    for subset in SUBSETS:
        rows = frame[columns] if subset == "all_races" else frame.loc[~floor_race, columns]
        rows = rows.reset_index(drop=True)
        steps = (
            ("walk_forward", scheme, lambda r=rows, s=subset: _walk_forward(r, spec, s)),
            ("by_year", IN_SAMPLE, lambda r=rows, s=subset: _in_sample(r, s, min_odds)),
        )
        for name, fold_scheme, compute in steps:
            parts = {"fit": FIT}
            if name == "walk_forward":
                parts["score"] = "winner_log_loss"
            params = {"subset": subset, "tail_min_odds": min_odds}
            variant = Variant(
                RECORD_HYPOTHESIS, f"tail_hinge:{name}:{subset}", fold_scheme, parts, params
            )
            study.attempt(variant, compute)
        del rows
    return study


# --------------------------------------------------------------------------- report and runner

SECTIONS = (
    (
        "slopes",
        "Hinge slopes (in-sample, development races, 95% Wald CI)",
        "body β is the slope on ln(1/odds) up to the threshold; the hinge γ is the coefficient on"
        " the distance past the knot, so tail β = body β - γ is the slope on ln(1/odds) above it,"
        " with its se from the full covariance. The log-weight is continuous at the threshold: γ"
        " is a slope change, not a level step. A tail β of 1 says the tail's price is calibrated.",
    ),
    ("folds", "Walk-forward folds: the slopes of each fold's training rows", ""),
    (
        "headline",
        "Winner log loss and ΔLL of the hinge model against normalised, power-β and #285's tail β",
        "ΔLL is the comparator's log loss minus the hinge model's, per race, in nats (positive: the"
        " hinge model is better). lower and upper are the (1 - level) and level quantiles of the"
        " paired bootstrap over meetings (D7: one-sided).",
    ),
    (
        "tail",
        "Tail: runners priced above the tail threshold, observed vs expected under the hinge model",
        "Out of fold; a dead heat counts each co-winner. Compare #77's and #285's tail tables.",
    ),
)


def render_report(study: Study, spec: dict, about: dict) -> str:
    """The full report: aggregate tables only, never a runner or race row."""
    lines = [
        f"# Hinge tail slope at the final Win price ({COUNTRY}, #{TICKET})",
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
        calibration._markdown_table([{"count": k, "n": v} for k, v in study.counts.items()]),
    ]
    for key, title, note in SECTIONS:
        rows = study.tables.get(key)
        lines += ["", f"## {title}", ""]
        lines += [note, ""] if note else []
        lines.append(
            calibration._markdown_table(rows) if rows else "_Not computed: see the variants below._"
        )
    variants = [
        {"variant": v.name, "status": "failed" if v.error else "scored"}
        | ({"error": v.error} if v.error else {})
        for v in study.variants
    ]
    lines += [
        "",
        "## Variants (one registry record each)",
        "",
        calibration._markdown_table(variants),
        "",
    ]
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

    Raises `calibration.MissingOwnerDecision` before the store is read when
    config/calibration.yaml lacks an input, `paths.DataDirInsideRepo` for an `out_dir` inside the
    repository, and `FamilyAlreadyFrozen` when the family is frozen with another spec."""
    owner = calibration.owner_inputs()
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    spec = build_spec(
        owner, load_country_scheme(COUNTRY), bootstrap_settings(), calibration.win_floor()
    )
    git_sha = _git_sha()
    registry = default_registry(COUNTRY)
    spec_hash = registry.freeze_family(FAMILY_ID, spec)  # before any development row is read

    builds = calibration.store_builds()
    runners, races = calibration.read_development()
    frame, counts = calibration.development_frame(runners, races)
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
        out_dir / f"tail-hinge-{COUNTRY}-{spec_hash[:12]}.md",
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
    parser = argparse.ArgumentParser(prog="python -m horsetrack.countries.japan.studies.tail_hinge")
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    args = parser.parse_args(argv)
    print(json.dumps(run_study(args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
