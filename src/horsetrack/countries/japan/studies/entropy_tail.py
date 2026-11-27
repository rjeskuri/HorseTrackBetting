"""The E3 entropy gradient in beta, separated from the far tail (#277; follows #77's E3 and #285).

#77 bins each race by its normalised market entropy and fits one power beta per bin: beta falls from
bin 1 (the most concentrated markets) to bin 5. The far tail is over-priced (#285), and the number
of runners above `tail_min_odds` moves with the entropy, so the gradient could be the tail count's
alone. This study refits it conditioning on the tail and asks whether the bin 1 minus bin 5
difference in beta (delta) survives.

The model is the conditional logit P(i wins) proportional to exp(slope_r * x_i), x = ln(1/O), over
each race's priced runners, with three readings of slope_r, all reported side by side
(`METHODS`):

- `unconditioned`: slope_r = beta_b for the race's entropy bin b, one fit per bin, as #77's E3.
- `tail_share`: slope_r = beta_b + gamma * share_r in one joint fit over the five bins, share_r =
  (priced runners strictly above `tail_min_odds`) / (priced runners); beta_b is the bin's beta for
  a race with no tail runner.
- `tail_count_strata`: races are cut by their tail count c_r into the strata of `TAIL_COUNT_STRATA`
  (the last open-ended); within each stratum bin 1 and bin 5 get their own slope, and the
  strata with at least `MIN_CELL_RACES` races in both cells are pooled by inverse variance.

delta is beta_1 - beta_5; it "survives" conditioning when its 95% Wald interval (delta +/- 1.96 se)
excludes 0. Entries and the handling of unpriced runners, winnerless races and dead heats are
`calibration.fit_power_beta`'s, and a race needs two or more priced runners (its entropy is
defined). The bins use #77's frozen cut points on the development rows, in-sample.

The study (`run_study`; CLI `python -m horsetrack.countries.japan.studies.entropy_tail [--out
DIR]`) is a thin I/O layer over the pure `compute_study(frame, spec)`, in its own registry family.
Order: build the spec, freeze the family in the run registry (`FamilyAlreadyFrozen` on another spec
stops the run), only then read the development rows (`calibration.read_development`: holdout days
dropped, price era only; the holdout is never opened), score, append one registry record per
variant (failures included) and write the report through `write_report`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from horsetrack.core import paths
from horsetrack.core.eval.folds import FoldScheme, load_country_scheme
from horsetrack.core.eval.holdout import _git_sha
from horsetrack.core.eval.registry import FoldMetric, RunRecord, default_registry
from horsetrack.core.eval.reports import write_report
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START
from horsetrack.countries.japan.studies import calibration
from horsetrack.countries.japan.studies.calibration import Study, Variant

NEWTON_MAX_STEPS = calibration.NEWTON_MAX_STEPS
NEWTON_STEP_TOL = calibration.NEWTON_STEP_TOL

COUNTRY = calibration.COUNTRY
TICKET = 277
SOURCE = calibration.SOURCE  # the store's JRA results, from netkeiba; no JRDB, no JRA-VAN
FAMILY_ID = "japan-entropy-gradient-tail-conditioned"
HYPOTHESIS = "E3"
FIT = "conditional_logit_ln_inv_odds_entropy_bins"
IN_SAMPLE = calibration.IN_SAMPLE
WALD_Z = calibration.BETA_CI_Z
DELTA_METRIC = "delta_beta_bin1_minus_bin5"
BINS = 5

METHODS = ("unconditioned", "tail_share", "tail_count_strata")
# Tail-count strata: no tail runner, one, two, three, four, five or more. Pre-registered here,
# before any race is scored; the strata whose bin 1 or bin 5 cell is thin are reported and left out.
TAIL_COUNT_STRATA = (0, 1, 2, 3, 4, 5)
MIN_CELL_RACES = 30
DECISIONS = {
    **{k: calibration.DECISIONS[k] for k in ("D6", "D9", "D10", "D12", "D14", "D15")},
    "#77 (E3)": "the entropy bins are #77's: quintile cut points of the per-race normalised "
    "entropy over the first training window, frozen in config/calibration.yaml",
    "#77 (5)": "the tail is a runner priced above decimal 51.0 (config/calibration.yaml)",
    "#277": "this family's spec is frozen before any development race is scored; the holdout is "
    "not opened",
}


@dataclass(frozen=True)
class GradientFit:
    """delta = beta(bin 1) - beta(bin 5) by one method, its standard error and 95% Wald bounds.

    `n_races` counts the races fitted (the bin 1 and bin 5 cells of the used strata for the strata
    method). `bin_betas` and `bin_ses` hold the five bins' betas (empty for the strata method);
    `gamma` is the tail share's coefficient (`tail_share` only). `strata_used` lists the used
    strata's lower edges and `strata` every stratum's cells (strata method only)."""

    method: str
    delta: float
    se: float
    lo: float
    hi: float
    n_races: int
    bin_betas: tuple[float, ...] = ()
    gamma: float | None = None
    gamma_se: float | None = None
    strata_used: tuple[int, ...] = ()
    bin_ses: tuple[float, ...] = ()
    strata: tuple[dict, ...] = ()


# --------------------------------------------------------------------------- the fit


@dataclass(frozen=True)
class _Choices:
    """Priced runners of the usable races (a priced winner, two or more priced runners), sorted
    by race, with each race's entropy bin, tail count and tail share."""

    x: np.ndarray  # ln(1 / O), per runner
    won: np.ndarray  # per runner
    race: np.ndarray  # race index per runner
    starts: np.ndarray
    sizes: np.ndarray
    winners: np.ndarray  # each race's priced winners (2+ on a dead heat)
    bin: np.ndarray  # per race, 0..4
    count: np.ndarray  # per race: priced runners strictly above min_odds
    share: np.ndarray  # per race: count / priced runners

    def races(self, keep: np.ndarray) -> _Choices:
        """The sub-sample holding only the races where `keep` is True."""
        rows = keep[self.race]
        sizes = self.sizes[keep]
        starts = np.cumsum(np.r_[0, sizes[:-1]]) if len(sizes) else np.zeros(0, dtype=int)
        return _Choices(
            x=self.x[rows],
            won=self.won[rows],
            race=np.repeat(np.arange(len(sizes)), sizes),
            starts=starts,
            sizes=sizes,
            winners=self.winners[keep],
            bin=self.bin[keep],
            count=self.count[keep],
            share=self.share[keep],
        )


def _tail_threshold(min_odds: float | None) -> float:
    if min_odds is None:
        return calibration.owner_inputs()["tail_min_odds"]
    return float(min_odds)


def _check_cut_points(cut_points: Sequence[float]) -> np.ndarray:
    cuts = np.asarray(list(cut_points), dtype=float)
    ok = cuts.shape == (BINS - 1,) and bool(np.isfinite(cuts).all())
    if not ok or not (0.0 < cuts[0] and cuts[-1] < 1.0 and (np.diff(cuts) > 0).all()):
        raise ValueError(f"cut_points must be four increasing numbers in (0, 1): {cut_points!r}")
    return cuts


def _check_edges(edges: Sequence[int]) -> list[int]:
    values = list(edges)
    if not values or any(
        isinstance(e, bool) or not isinstance(e, (int, np.integer)) for e in values
    ):
        raise ValueError(f"strata_edges must be a list of ints: {edges!r}")
    if values[0] != 0 or any(b <= a for a, b in zip(values, values[1:], strict=False)):
        raise ValueError(f"strata_edges must start at 0 and increase strictly: {edges!r}")
    return [int(e) for e in values]


def _choices(
    odds: pd.Series,
    race_id: pd.Series,
    won: pd.Series,
    cuts: np.ndarray,
    min_odds: float,
) -> _Choices:
    calibration._check_aligned(odds, race_id, won)
    if won.isna().any():
        raise ValueError("won has missing values")
    cleaned, ok, codes, races = calibration._priced(odds, race_id)
    entropy = calibration.normalised_entropy(odds, race_id).to_numpy()
    priced = np.bincount(codes[ok], minlength=len(races))
    tail = np.bincount(codes[ok], weights=cleaned[ok] > min_odds, minlength=len(races))
    won_priced = won.to_numpy(dtype=bool) & ok
    winners = np.bincount(codes, weights=won_priced, minlength=len(races))
    usable = (priced >= 2) & (winners > 0)
    rows = np.flatnonzero(ok & usable[codes])
    if len(rows) == 0:
        raise ValueError("no race with a priced winner and two priced runners to fit")
    rows = rows[np.argsort(codes[rows], kind="stable")]
    race = codes[rows]
    starts = np.flatnonzero(np.r_[True, race[1:] != race[:-1]])
    sizes = np.diff(np.r_[starts, len(rows)])
    first = race[starts]
    return _Choices(
        x=-np.log(cleaned[rows]),
        won=won_priced[rows],
        race=np.repeat(np.arange(len(starts)), sizes),
        starts=starts,
        sizes=sizes,
        winners=winners[first],
        bin=np.searchsorted(cuts, entropy[first], side="right"),
        count=tail[first],
        share=tail[first] / priced[first],
    )


def _evaluate(
    theta: np.ndarray, design: np.ndarray, c: _Choices, won_sum: np.ndarray, *, info: bool
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """The conditional-logit log-likelihood and, when `info`, its score and observed information."""
    s = design @ theta
    top = np.maximum.reduceat(s, c.starts)
    e = np.exp(s - np.repeat(top, c.sizes))
    total = np.add.reduceat(e, c.starts)
    w = c.winners
    loglik = float(theta @ won_sum - (w * (np.log(total) + top)).sum())
    if not info:
        return loglik, None, None
    k = design.shape[1]
    mean = [np.add.reduceat(e * design[:, a], c.starts) / total for a in range(k)]
    score = won_sum - np.array([float((w * m).sum()) for m in mean])
    fisher = np.zeros((k, k))
    for a in range(k):
        for b in range(a, k):
            second = np.add.reduceat(e * design[:, a] * design[:, b], c.starts) / total
            fisher[a, b] = fisher[b, a] = float((w * (second - mean[a] * mean[b])).sum())
    return loglik, score, fisher


def _identified(fisher: np.ndarray) -> bool:
    lam = np.linalg.eigvalsh(fisher)
    return bool(np.isfinite(lam).all() and lam[0] > 1e-12 * lam[-1] and lam[-1] > 0)


def _solve(design: np.ndarray, c: _Choices) -> tuple[np.ndarray, np.ndarray]:
    """Maximum likelihood theta of P(i wins) proportional to exp(design_i . theta) over each race,
    and the inverse observed information: damped Newton steps from zero, halved whenever one
    would lower the log-likelihood, until a step is under 1e-10."""
    won_sum = design[c.won].sum(axis=0)
    theta = np.zeros(design.shape[1])
    loglik, score, fisher = _evaluate(theta, design, c, won_sum, info=True)
    for _ in range(NEWTON_MAX_STEPS):
        if not _identified(fisher):
            raise ValueError("the slopes are not identified: the fitted runners do not vary enough")
        step = np.linalg.solve(fisher, score)
        while True:
            trial = _evaluate(theta + step, design, c, won_sum, info=False)[0]
            if trial >= loglik or np.abs(step).max() < NEWTON_STEP_TOL:
                break
            step = step / 2.0
        theta = theta + step
        loglik, score, fisher = _evaluate(theta, design, c, won_sum, info=True)
        if np.abs(step).max() < NEWTON_STEP_TOL:
            break
    else:
        raise RuntimeError(f"the gradient fit did not converge in {NEWTON_MAX_STEPS} Newton steps")
    if not (np.isfinite(theta).all() and _identified(fisher)):
        raise RuntimeError(f"the gradient fit stopped at theta={theta.tolist()}")
    return theta, np.linalg.inv(fisher)


def _one_slope(c: _Choices) -> tuple[float, float]:
    """The slope of every race of `c` and its variance."""
    if len(c.sizes) == 0:
        raise ValueError("no race in this cell to fit")
    theta, cov = _solve(c.x[:, None], c)
    return float(theta[0]), float(cov[0, 0])


def _fit(method: str, delta: float, se: float, n_races: int, **rest) -> GradientFit:
    return GradientFit(
        method=method,
        delta=float(delta),
        se=float(se),
        lo=float(delta - WALD_Z * se),
        hi=float(delta + WALD_Z * se),
        n_races=int(n_races),
        **rest,
    )


def _unconditioned(c: _Choices) -> GradientFit:
    cells = [_one_slope(c.races(c.bin == b)) for b in range(BINS)]
    betas = tuple(beta for beta, _ in cells)
    ses = tuple(math.sqrt(var) for _, var in cells)
    delta, se = betas[0] - betas[-1], math.hypot(ses[0], ses[-1])
    return _fit("unconditioned", delta, se, len(c.sizes), bin_betas=betas, bin_ses=ses)


def _tail_share(c: _Choices) -> GradientFit:
    per_race = np.stack([c.bin == b for b in range(BINS)], axis=1).astype(float)
    design = np.column_stack([c.x[:, None] * per_race[c.race], c.x * c.share[c.race]])
    theta, cov = _solve(design, c)
    delta = theta[0] - theta[BINS - 1]
    se = math.sqrt(cov[0, 0] + cov[BINS - 1, BINS - 1] - 2.0 * cov[0, BINS - 1])
    return _fit(
        "tail_share",
        delta,
        se,
        len(c.sizes),
        bin_betas=tuple(float(b) for b in theta[:BINS]),
        bin_ses=tuple(math.sqrt(cov[b, b]) for b in range(BINS)),
        gamma=float(theta[BINS]),
        gamma_se=math.sqrt(cov[BINS, BINS]),
    )


def _strata(c: _Choices, edges: list[int], min_cell_races: int) -> GradientFit:
    rows, deltas, variances, used, races = [], [], [], [], 0
    for lo, hi in zip(edges, [*edges[1:], None], strict=True):
        inside = (c.count >= lo) & ((c.count < hi) if hi is not None else True)
        first, last = c.races(inside & (c.bin == 0)), c.races(inside & (c.bin == BINS - 1))
        row = {
            "tail count": f"{lo}+"
            if hi is None
            else (str(lo) if hi == lo + 1 else f"{lo}-{hi - 1}")
        }
        row |= {"lower edge": lo, "bin 1 races": len(first.sizes), "bin 5 races": len(last.sizes)}
        if min(len(first.sizes), len(last.sizes)) < min_cell_races:
            rows.append(row | {"status": f"left out: a cell under {min_cell_races} races"})
            continue
        try:
            (b1, v1), (b5, v5) = _one_slope(first), _one_slope(last)
        except (ValueError, RuntimeError) as exc:
            rows.append(row | {"status": f"left out: {type(exc).__name__}"})
            continue
        var = v1 + v5
        rows.append(row | {"beta bin 1": b1, "beta bin 5": b5, "delta": b1 - b5})
        rows[-1] |= {"se": math.sqrt(var), "status": "used"}
        deltas.append(b1 - b5)
        variances.append(var)
        used.append(lo)
        races += len(first.sizes) + len(last.sizes)
    if not used:
        raise ValueError(
            f"no tail-count stratum holds {min_cell_races} races in both the bin 1 and bin 5 cells"
        )
    weights = 1.0 / np.array(variances)
    delta = float((weights * np.array(deltas)).sum() / weights.sum())
    return _fit(
        "tail_count_strata",
        delta,
        float(weights.sum() ** -0.5),
        races,
        strata_used=tuple(used),
        strata=tuple(rows),
    )


def fit_gradient(
    odds: pd.Series,
    race_id: pd.Series,
    won: pd.Series,
    *,
    cut_points: Sequence[float],
    method: str,
    min_odds: float | None = None,
    strata_edges: Sequence[int] | None = None,
    min_cell_races: int = MIN_CELL_RACES,
) -> GradientFit:
    """delta = beta(bin 1) - beta(bin 5) of the entropy bins by one of `METHODS`, see the module
    docstring. `cut_points` are the four increasing cuts of the normalised entropy (bin 1 is the
    lowest entropy); `min_odds` is the tail threshold (None: `tail_min_odds` of
    config/calibration.yaml, 51.0) and a runner at exactly it is in the body; `strata_edges` are
    the tail-count strata's lower edges (None: `TAIL_COUNT_STRATA`).

    Raises ValueError for an unknown method, bad cut points or strata edges, no race left to fit,
    a bin without a race or an unidentified slope (`unconditioned`, `tail_share`), or no usable
    stratum (`tail_count_strata`); RuntimeError when Newton does not converge."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}: one of {METHODS}")
    cuts = _check_cut_points(cut_points)
    edges = _check_edges(TAIL_COUNT_STRATA if strata_edges is None else strata_edges)
    if isinstance(min_cell_races, bool) or not isinstance(min_cell_races, (int, np.integer)):
        raise ValueError(f"min_cell_races must be an int: {min_cell_races!r}")
    if min_cell_races < 1:
        raise ValueError(f"min_cell_races must be at least 1: {min_cell_races!r}")
    c = _choices(odds, race_id, won, cuts, _tail_threshold(min_odds))
    if method == "unconditioned":
        return _unconditioned(c)
    if method == "tail_share":
        return _tail_share(c)
    return _strata(c, edges, int(min_cell_races))


# --------------------------------------------------------------------------- spec


def build_spec(owner: dict, scheme: FoldScheme, bootstrap: dict, floor: float) -> dict:
    """The family's pre-registered spec, frozen in the registry before anything is scored."""
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypotheses": {
            HYPOTHESIS: "the gradient in beta by market-entropy bin (#77's E3) once the far tail "
            "is conditioned on: delta = beta(bin 1) - beta(bin 5) by three methods, unconditioned, "
            "with the tail runners' share as a covariate, and within strata of the tail count",
        },
        "target": "the race's winner (Win pool) at the final Win odds",
        "model": "P(i wins) proportional to exp(slope_r x_i), x_i = ln(1/O_i), over each race's "
        "priced runners",
        "methods": {
            "unconditioned": "slope_r = beta_b for the race's entropy bin b: one fit per bin",
            "tail_share": "slope_r = beta_b + gamma share_r in one joint fit over the five bins, "
            "share_r = priced runners strictly above tail.min_odds / priced runners, uncentred",
            "tail_count_strata": "races cut by their count of priced runners strictly above "
            "tail.min_odds into the strata of tail_count_strata (the last open-ended); within "
            "each stratum bin 1 and bin 5 get their own slope; strata with min_cell_races or more "
            "races in both cells are pooled beta_1 - beta_5 by inverse variance",
        },
        "delta": "beta(bin 1) - beta(bin 5); se from the inverse observed information (the "
        "unconditioned method: sqrt(se_1^2 + se_5^2); the strata method: (sum 1/v_s)^(-1/2))",
        "ci": "95% two-sided Wald, delta +/- 1.96 se",
        "survives": "the gradient survives conditioning on the tail when the 95% interval of delta "
        "excludes 0, for tail_share and for tail_count_strata",
        "fit": "conditional logit MLE by damped Newton; races with a priced winner and two or more "
        "priced runners; a dead heat counts each priced co-winner",
        "in_sample": "one fit per method on every development race, descriptive, no fold",
        "adoption": "none: the owner decides whether anything enters the baseline",
        "rows": "priced starters of run races with a priced winner; development days only",
        "entropy_cut_points": owner["entropy_cut_points"],
        "tail": {"min_odds": owner["tail_min_odds"], "rule": "a runner priced above min_odds"},
        "tail_count_strata": list(TAIL_COUNT_STRATA),
        "min_cell_races": MIN_CELL_RACES,
        "inherited_unused": {
            "floor": floor,
            "fold_scheme": dataclasses.asdict(scheme),
            "bootstrap": bootstrap,
            "note": "this family is in-sample Wald only: no fold, no bootstrap, no floor subset",
        },
        "price_era_start": PRICE_ERA_START.isoformat(),
        "decisions": DECISIONS,
        "owner_signoff": owner["owner_signoff"],
        "sources": [SOURCE],
    }


# --------------------------------------------------------------------------- the pure study


def _wald_metric(name: str, value: float, se: float, n: int) -> FoldMetric:
    return calibration._metric(
        "development", name, value, n, value - WALD_Z * se, value + WALD_Z * se
    )


def _metrics(fit: GradientFit) -> list[FoldMetric]:
    metrics = [
        calibration._metric("development", DELTA_METRIC, fit.delta, fit.n_races, fit.lo, fit.hi)
    ]
    for k, (beta, se) in enumerate(zip(fit.bin_betas, fit.bin_ses, strict=True), start=1):
        metrics.append(_wald_metric(f"beta[bin={k}]", beta, se, fit.n_races))
    if fit.gamma is not None:
        metrics.append(_wald_metric("gamma_tail_share", fit.gamma, fit.gamma_se, fit.n_races))
    return metrics


def _excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0


def _tables(fit: GradientFit) -> dict[str, list[dict]]:
    row = {"method": fit.method, "races": fit.n_races, "delta (β1 − β5)": fit.delta}
    row |= {"lower 95%": fit.lo, "upper 95%": fit.hi}
    row |= {"survives (CI excludes 0)": _excludes_zero(fit.lo, fit.hi)}
    if fit.gamma is not None:
        row |= {"γ": fit.gamma, "γ lower 95%": fit.gamma - WALD_Z * fit.gamma_se}
        row |= {"γ upper 95%": fit.gamma + WALD_Z * fit.gamma_se}
    if fit.strata_used:
        row |= {"strata used (lower edges)": ", ".join(str(e) for e in fit.strata_used)}
    tables = {"gradient": [row]}
    if fit.bin_betas:
        tables["bins"] = [
            {"method": fit.method, "entropy bin": k, "β": b, "lower 95%": b - WALD_Z * s}
            | {"upper 95%": b + WALD_Z * s}
            for k, (b, s) in enumerate(zip(fit.bin_betas, fit.bin_ses, strict=True), start=1)
        ]
    if fit.strata:
        tables["strata"] = [{"method": fit.method} | s for s in fit.strata]
    return tables


def bin_profile(frame: pd.DataFrame, cut_points: Sequence[float], min_odds: float) -> list[dict]:
    """Per entropy bin: races, mean tail count and mean tail share over the usable races (the
    ones the fits use), and how many have no tail runner."""
    c = _choices(
        frame["odds"], frame["race_id"], frame["won"], _check_cut_points(cut_points), min_odds
    )
    rows = []
    for b in range(BINS):
        in_bin = c.bin == b
        rows.append(
            {
                "entropy bin": b + 1,
                "races": int(in_bin.sum()),
                "mean tail count": float(c.count[in_bin].mean()) if in_bin.any() else None,
                "mean tail share": float(c.share[in_bin].mean()) if in_bin.any() else None,
                "races with no tail runner": int((in_bin & (c.count == 0)).sum()),
            }
        )
    return rows


def compute_study(frame: pd.DataFrame, spec: dict) -> Study:
    """One variant per method of the frozen `spec` on `frame` (development_frame's shape): delta
    with its 95% Wald bounds, in-sample, and the tables behind it. Pure: no I/O."""
    study = Study(counts={})
    columns = [frame["odds"], frame["race_id"], frame["won"]]
    cuts, min_odds = spec["entropy_cut_points"], spec["tail"]["min_odds"]
    for method in METHODS:
        params = {
            "method": method,
            "tail_min_odds": min_odds,
            "entropy_cut_points": list(cuts),
            "tail_count_strata": list(spec["tail_count_strata"]),
            "min_cell_races": spec["min_cell_races"],
        }
        variant = Variant(
            HYPOTHESIS,
            f"entropy_gradient:{method}",
            IN_SAMPLE,
            {"fit": FIT, "method": method},
            params,
        )

        def compute(method=method):
            fit = fit_gradient(
                *columns,
                cut_points=cuts,
                method=method,
                min_odds=min_odds,
                strata_edges=spec["tail_count_strata"],
                min_cell_races=spec["min_cell_races"],
            )
            return _metrics(fit), _tables(fit)

        study.attempt(variant, compute)
    try:
        study.tables["profile"] = bin_profile(frame, cuts, min_odds)
    except ValueError:
        pass  # no usable race: every variant failed and says so
    return study


# --------------------------------------------------------------------------- report and runner

SECTIONS = (
    (
        "profile",
        "The tail count by entropy bin (the confounder)",
        "Over the races the fits use. If the tail count moves with the bin, the unconditioned"
        " gradient can be the tail's alone.",
    ),
    (
        "gradient",
        "delta = β(bin 1) − β(bin 5), three ways (in-sample, development races, 95% Wald CI)",
        "unconditioned: one β per bin (#77's E3). tail_share: β by bin plus γ times the share of"
        " the race's runners above the tail threshold, one joint fit. tail_count_strata: the"
        " bin 1 and bin 5 β within strata of the tail count, pooled by inverse variance. The"
        " gradient survives when the 95% CI excludes 0.",
    ),
    (
        "bins",
        "β by entropy bin (95% Wald CI)",
        "For the share model, β is a race with no tail runner.",
    ),
    (
        "strata",
        "Tail-count strata: β of bin 1 and bin 5 within each",
        "A stratum is used when both cells hold the pre-registered minimum of races.",
    ),
)


def render_report(study: Study, spec: dict, about: dict) -> str:
    """The full report: aggregate tables only, never a runner or race row."""
    lines = [
        f"# The entropy gradient conditioned on the tail ({COUNTRY}, #{TICKET})",
        "",
        *(f"- {key}: {value}" for key, value in about.items()),
        f"- owner sign-off of the pre-registered inputs: #{spec['owner_signoff']}",
        f"- tail: above {spec['tail']['min_odds']:g}; tail-count strata (lower edges): "
        f"{spec['tail_count_strata']}; minimum cell: {spec['min_cell_races']} races",
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


def _survives(study: Study, method: str) -> bool | None:
    """Whether the 95% CI of delta by `method` excludes 0; None when that variant failed."""
    for variant in study.variants:
        if variant.name == f"entropy_gradient:{method}" and not variant.error:
            metric = next(m for m in variant.metrics if m.metric == DELTA_METRIC)
            return _excludes_zero(metric.lo, metric.hi)
    return None


def run_study(out_dir: Path | None = None) -> dict:
    """Run the pre-registered study on Japan's development data and write its report.

    Raises `calibration.MissingOwnerDecision` before the store is read when
    config/calibration.yaml lacks an input, `paths.DataDirInsideRepo` for an `out_dir` inside the
    repository, and `FamilyAlreadyFrozen` when the family is frozen with another spec. The result's
    `survives_conditioning` is, per conditioned method, whether delta's 95% CI excludes 0 (None
    when that variant failed)."""
    owner = calibration.owner_inputs()
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    spec = build_spec(
        owner,
        load_country_scheme(COUNTRY),
        calibration.bootstrap_settings(),
        calibration.win_floor(),
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
        out_dir / f"entropy-tail-{COUNTRY}-{spec_hash[:12]}.md",
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
        "survives_conditioning": {
            m: _survives(study, m) for m in ("tail_share", "tail_count_strata")
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m horsetrack.countries.japan.studies.entropy_tail"
    )
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    args = parser.parse_args(argv)
    print(json.dumps(run_study(args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
