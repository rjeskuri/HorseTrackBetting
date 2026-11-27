"""Exacta favourite-overload study (#131): the scoring maths of the signed pre-registration.

docs/specs/exacta-overload-prereg.md (approved by the owner 2026-11-25, frozen in the run registry
through ``family_specs``). History holds only the winning exacta per race, so each category S
(favourite first, favourite second, without the favourite) is scored as a box bet that is judged
at the dividend that paid:

- B1, the flat box: 1/|S_r| per combination of S, return ``1[w in S] * D_w / 100 / |S_r|``;
- B2, the Harville-weighted box (primary): ``p_c / sum_S p`` per combination, return
  ``1[w in S] * (p_w / sum_S p) * D_w / 100``;
- B3, a flat box at Harville-fair odds, return ``1[w in S] / (p_w * |S_r|)`` (Harville's own
  calibration on S, null 1).

B2 and B1 return the race's payout rate in expectation when the pool is right, whatever the error
of the ordering model, so the estimand ``rho_S = sum R / sum rate`` has null 1. Pure functions:
no store reads and no registry writes (the run lives in ``exacta_run``).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from numbers import Integral

import numpy as np
import pandas as pd

from horsetrack.countries.japan.research.exacta_consistency import (
    CATEGORIES,
    combo_category,
    favourite,
    harville_exacta,
)

SEGMENT_CELLS = (
    "all",
    "fav_odds:le1.9", "fav_odds:2.0-2.9", "fav_odds:3.0-4.9", "fav_odds:ge5.0",
    "field:5-10", "field:11-14", "field:15-18",
    "class:maiden", "class:win1-3", "class:open",
    "popularity:less", "popularity:rest",
    "headline",
)  # fmt: skip
CLASS_GROUPS = {
    "DEBUT": "maiden",
    "MAIDEN": "maiden",
    "C1": "win1-3",
    "C2": "win1-3",
    "C3": "win1-3",
    "OP": "open",
    "L": "open",
    "G1": "open",
    "G2": "open",
    "G3": "open",
}
DIRECTIONS = {"fav_first": "less", "fav_second": "greater", "no_fav": "greater"}
ESTIMATORS = ("b1", "b2", "b3")
MIN_STARTERS = 5

# v2 (owner, 2026-11-26): the v1 families, frozen 2026-11-25, labelled #115's single
# full-development lambda2 fit "in_fold". The relabelled specs are frozen under new ids beside
# them; the registry is append-only, so v1 stays as frozen. Only the label and its text change.
E1_ID = "japan-131-exacta-category-v2"
E1_B1_ID = "japan-131-exacta-category-b1-v2"
E1_B3_ID = "japan-131-exacta-category-b3-v2"
C_ID = "japan-131-exacta-logscore-v2"
# E2 v3 (review of 0c39a8d): rule-v2 took its favourite from _COMMON's final Win odds, against
# the signed spec's section 5 ("Selection uses only the board at t"); v3 changed only E2's
# selection keys. E2 v4 (review of 83684eb, owner 2026-11-27): v3's headline ranked "less
# popular" by the lower tertile of pool totals within each logged day, so an early race was
# classified with later races' totals. v4 takes "less popular" from the race card (E1's
# popularity segment), reports the tertile as an agreement diagnostic only and drops the
# 100-meeting gate. Each version supersedes the last; the v2 and v3 freezes stay in the registry.
# Ids (#525, owner 2026-11-27): the registry refuses "/" in a family id, so the live families take
# the "japan-131-..." form with the specs (and hashes) frozen under the legacy "japan/131-..." ids;
# SUPERSEDES names the legacy records as they stand in the registry.
E2_ID = "japan-131-exacta-rule-v4"
SUPERSEDES = {
    E1_ID: "japan/131-exacta-category-v1",
    E1_B1_ID: "japan/131-exacta-category-b1-v1",
    E1_B3_ID: "japan/131-exacta-category-b3-v1",
    C_ID: "japan/131-exacta-logscore-v1",
    E2_ID: "japan/131-exacta-rule-v3",
}
FITTED = "full_development"  # #115's one lambda2 fit over the development out-of-fold races
LAMBDA2S = [FITTED, 1.0]


def _finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


# --------------------------------------------------------------------------- per race


def box_returns(
    p: Mapping[int, float],
    winner: tuple[int, int],
    dividend: float,
    *,
    favourite: int,
    lambda2: float = 1.0,
) -> dict[str, dict[str, float]]:
    """One race's B1, B2 and B3 return per unit staked on each category's box.

    ``p`` maps horse to Win probability (renormalised), ``winner`` is the paying (first, second)
    and ``dividend`` its final dividend per 100 yen. Only the winner's category returns anything.
    """
    dividend = _finite_positive("dividend", dividend)
    horses = sorted(p)
    index = {h: k for k, h in enumerate(horses)}
    absent = [h for h in (*winner, favourite) if h not in index]
    if absent:
        raise ValueError(f"winner {winner} or favourite {favourite} not in the field: {absent}")
    category = combo_category(tuple(winner), favourite)
    model = harville_exacta(np.array([p[h] for h in horses], dtype=float), lambda2=lambda2)

    n = len(horses)
    f = index[favourite]
    if category == "fav_first":
        mass, size = model[f, :].sum(), n - 1
    elif category == "fav_second":
        mass, size = model[:, f].sum(), n - 1
    else:
        mass = model.sum() - model[f, :].sum() - model[:, f].sum()
        size = (n - 1) * (n - 2)
    p_w = float(model[index[winner[0]], index[winner[1]]])

    out = {e: dict.fromkeys(CATEGORIES, 0.0) for e in ESTIMATORS}
    out["b1"][category] = dividend / 100 / size
    out["b2"][category] = p_w / float(mass) * dividend / 100
    out["b3"][category] = 1 / (p_w * size)
    return out


def rho(returns: Iterable[float], rates: Iterable[float]) -> float:
    """Total return over total payout rate: 1 when the category is priced like the pool."""
    returns = np.asarray(list(returns), dtype=float)
    rates = np.asarray(list(rates), dtype=float)
    if returns.size == 0 or returns.size != rates.size:
        raise ValueError("rho needs one rate per return and at least one race")
    total = rates.sum()
    if not total > 0:
        raise ValueError("the rates must sum to a positive number")
    return float(returns.sum() / total)


def e1_loss(category: str, ret: float, rate: float) -> float:
    """The per-race loss of an E1 member, lower in the member's hypothesised direction."""
    if category not in DIRECTIONS:
        raise ValueError(f"unknown category {category!r}")
    return ret - rate if category == "fav_first" else rate - ret


def segment_cells(*, fav_odds: float, field_size: int, race_class: str, race_no: int) -> frozenset:
    """The SEGMENT_CELLS a race is in (all, one level of each marginal, maybe the headline)."""
    if not MIN_STARTERS <= field_size <= 18:
        raise ValueError(f"field size {field_size} is outside 5-18")
    if race_class not in CLASS_GROUPS:
        raise ValueError(f"unknown race class {race_class!r}")
    fav_odds = _finite_positive("fav_odds", fav_odds)
    if fav_odds < 2.0:
        band = "le1.9"
    elif fav_odds < 3.0:
        band = "2.0-2.9"
    elif fav_odds < 5.0:
        band = "3.0-4.9"
    else:
        band = "ge5.0"
    field = "5-10" if field_size <= 10 else "11-14" if field_size <= 14 else "15-18"
    group = CLASS_GROUPS[race_class]
    less = race_no <= 4 or group == "maiden"
    cells = {
        "all",
        f"fav_odds:{band}",
        f"field:{field}",
        f"class:{group}",
        "popularity:less" if less else "popularity:rest",
    }
    if fav_odds >= 3.0 and less:
        cells.add("headline")
    return frozenset(cells)


def exclusion_reason(
    *, n_starters: int, boost_may_be_withheld: bool, exacta: Iterable[tuple]
) -> str | None:
    """None when the race is scored, else why not (spec section 2, owner answer 7)."""
    if boost_may_be_withheld:
        return "boost_may_be_withheld"
    if n_starters < MIN_STARTERS:
        return "under_5_starters"
    rows = list(exacta)
    if not rows:
        return "no_exacta_row"
    if any(bool(refund) for _, _, refund in rows):
        return "refund"
    if len(rows) > 1:
        return "dead_heat"
    return None


def score_races(
    races: Iterable[Mapping],
    *,
    holdout_day: Callable[[str, date], bool],
    lambda2: float = 1.0,
) -> pd.DataFrame:
    """Three rows per scored race (one per category): race_id, meeting_id, race_date, category,
    rate, b1, b2, b3, e1_loss (of b2) and one bool column per SEGMENT_CELLS name.

    A race on a holdout day is dropped before its exacta rows are read (D14); excluded races are
    dropped. The excluded counts by reason are in ``frame.attrs["excluded"]``."""
    rows: list[dict] = []
    excluded: dict[str, int] = {}
    for race in races:
        if holdout_day("japan", race["race_date"]):
            excluded["holdout_day"] = excluded.get("holdout_day", 0) + 1
            continue
        odds = race["win_odds"]
        exacta = list(race["exacta"])
        reason = exclusion_reason(
            n_starters=len(odds),
            boost_may_be_withheld=bool(race["boost_may_be_withheld"]),
            exacta=exacta,
        )
        if reason is not None:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        fav = favourite(odds)
        (winner, dividend, _), rate = exacta[0], float(race["rate"])
        out = box_returns(race["p"], tuple(winner), dividend, favourite=fav, lambda2=lambda2)
        cells = segment_cells(
            fav_odds=odds[fav],
            field_size=len(odds),
            race_class=race["race_class"],
            race_no=race["race_no"],
        )
        flags = {cell: cell in cells for cell in SEGMENT_CELLS}
        for category in CATEGORIES:
            rows.append(
                {
                    "race_id": race["race_id"],
                    "meeting_id": race["meeting_id"],
                    "race_date": race["race_date"],
                    "category": category,
                    "rate": rate,
                    **{e: out[e][category] for e in ESTIMATORS},
                    "e1_loss": e1_loss(category, out["b2"][category], rate),
                    **flags,
                }
            )
    columns = [
        "race_id", "meeting_id", "race_date", "category", "rate", *ESTIMATORS, "e1_loss",
        *SEGMENT_CELLS,
    ]  # fmt: skip
    frame = pd.DataFrame(rows, columns=columns)
    frame.attrs["excluded"] = excluded
    return frame


# --------------------------------------------------------------------------- inference


@dataclass(frozen=True)
class RhoInterval:
    """rho, its meeting-bootstrap SE and the one-sided bound at ``level``: the upper bound for
    alternative "less", the lower for "greater"."""

    rho: float
    se: float
    bound: float
    alternative: str
    level: float
    n_meetings: int


_MAX_DRAWS_PER_CHUNK = 2_000_000


def rho_interval(
    returns,
    rates,
    meetings,
    *,
    alternative: str,
    n_boot: int,
    seed: int,
    level: float = 0.95,
) -> RhoInterval:
    """A meeting (cluster) bootstrap of rho: whole meetings are drawn with replacement and each
    replicate is the drawn meetings' total return over their total rate."""
    if alternative not in ("less", "greater"):
        raise ValueError(f"alternative must be 'less' or 'greater', got {alternative!r}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    if isinstance(n_boot, bool) or not isinstance(n_boot, Integral) or n_boot < 2:
        raise ValueError(f"n_boot must be an integer of at least 2, got {n_boot!r}")
    returns = np.asarray(returns, dtype=float)
    rates = np.asarray(rates, dtype=float)
    estimate = rho(returns, rates)
    codes, _ = pd.factorize(np.asarray(meetings), sort=True)
    if len(codes) != len(returns) or (codes < 0).any():
        raise ValueError("every race needs a meeting")
    n = int(codes.max()) + 1
    ret_sum = np.bincount(codes, weights=returns, minlength=n)
    rate_sum = np.bincount(codes, weights=rates, minlength=n)

    rng = np.random.default_rng(seed)
    chunk = max(1, _MAX_DRAWS_PER_CHUNK // n)
    reps = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        drawn = rng.integers(0, n, size=(stop - start, n))
        reps[start:stop] = ret_sum[drawn].sum(axis=1) / rate_sum[drawn].sum(axis=1)
    q = level if alternative == "less" else 1.0 - level
    return RhoInterval(
        rho=estimate,
        se=float(reps.std(ddof=1)),
        bound=float(np.quantile(reps, q)),
        alternative=alternative,
        level=level,
        n_meetings=n,
    )


def eb_shrink(estimates, ses, *, tau2: float | None = None) -> np.ndarray:
    """Normal-normal empirical Bayes: each estimate's posterior mean given a common mean mu and
    between-variance tau2. mu is the precision-weighted mean (weights 1 / (se^2 + tau2)); tau2
    None is the DerSimonian-Laird moment estimate, truncated at 0; tau2 = 0 gives mu everywhere."""
    x = np.asarray(estimates, dtype=float)
    se2 = np.asarray(ses, dtype=float) ** 2
    if x.shape != se2.shape or x.ndim != 1 or x.size == 0:
        raise ValueError("estimates and ses must be equal-length, non-empty 1-d arrays")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(se2)) and np.all(se2 > 0)):
        raise ValueError("estimates must be finite and ses finite and positive")
    if tau2 is None:
        w = 1 / se2
        mean = (w * x).sum() / w.sum()
        q = (w * (x - mean) ** 2).sum()
        denom = w.sum() - (w**2).sum() / w.sum()
        tau2 = max(0.0, (q - (x.size - 1)) / denom) if denom > 0 else 0.0
    tau2 = float(tau2)
    if not math.isfinite(tau2) or tau2 < 0:
        raise ValueError(f"tau2 must be finite and non-negative, got {tau2!r}")
    w = 1 / (se2 + tau2)
    mu = float((w * x).sum() / w.sum())
    if tau2 == 0:
        return np.full(x.size, mu)
    return (x / se2 + mu / tau2) / (1 / se2 + 1 / tau2)


# --------------------------------------------------------------------------- pre-registration

_COMMON = {
    "ticket": 131,
    "hypothesis_id": "E43",
    "spec": "docs/specs/exacta-overload-prereg.md",
    "owner_signoff": "2026-11-25, #131 (all nine points as proposed)",
    "pool": "FORECAST",
    "sources": ["jra"],
    "data": "development race days only (is_holdout_day False, filtered before any outcome, "
    "odds or dividend is read); races from 2014-06-07 that have #114 out-of-fold Win "
    "probabilities, i.e. its walk-forward test quarters (from 2020-Q1)",
    "exclusions": [
        "boost_may_be_withheld",
        "fewer than 5 starters",
        "more than one paying exacta (dead heat for 1st or 2nd)",
        "a refunded exacta",
        "no exacta row",
    ],
    "win_probabilities": "#114 out-of-fold calibrated p_cal, renormalised over the starters",
    "ordering": "discounted Harville; full_development = #115's lambda2, one fit on the "
    "development out-of-fold races, the scored races among them, so partly in-sample "
    "(ordering_v0.yaml; the fit id is recorded with each run); 1.0 = plain Harville",
    "relabel": "owner 2026-11-26 on #131: #115's single full-development lambda2 fit is "
    "labelled full_development (the v1 label said in-fold); the design is v1's unchanged",
    "favourite": "lowest final Win odds, ties to the lowest number",
    "rate": "takeout.race_takeout(race_id, 'FORECAST', race_class_raw=...).payout_rate",
    "cluster": "meeting",
    "block_size": 5,
    "n_boot": 10000,
    "level": 0.95,
    "seed": 131,
}
_SEGMENTS = {
    "cells": list(SEGMENT_CELLS),
    "fav_odds": "final Win odds of the favourite: <=1.9, 2.0-2.9, 3.0-4.9, >=5.0",
    "field": "starters: 5-10, 11-14, 15-18",
    "class": "DEBUT, MAIDEN -> maiden; C1-C3 -> win1-3; OP, L, G1-G3 -> open",
    "popularity": "less = race_no 1-4 or a DEBUT/MAIDEN race; rest = every other race",
    "headline": "favourite odds >= 3.0 and less popular",
}


# E2's selection keys (v4). Signed spec section 5: "at T-2 ... Selection uses only the board at
# t. A selection variable taken from final odds is look-ahead"; section 6: the favourite's Win
# odds are "at t for E2". primary_metric: the owner's reframe of 2026-11-27 on #131. The
# headline's "less popular" is section 6's race-card proxy, the rule segment_cells applies.
_E2_SELECTION = {
    "selection_time": "T-2",
    "favourite": "lowest Win odds on the Win board at T-2 (the board at t), ties to the lowest "
    "number",
    "category": "fav_first, fav_second or no_fav of combination c against the favourite on the "
    "Win board at T-2 (combo_category)",
    "scopes": {
        "all": "every scored race",
        "headline": {
            "rule": "favourite's Win odds at T-2 >= 3.0 and less popular; less popular = race_no "
            "1-4 or a DEBUT/MAIDEN race, from the race card (E1's popularity segment)",
            "fav_odds_min": 3.0,
            "less_popular": {"race_no_max": 4, "race_classes": ["DEBUT", "MAIDEN"]},
            "information_cutoff": {
                "race_no": "race card",
                "race_class": "race card",
                "fav_win_odds": "T-2",
            },
        },
    },
    "diagnostics": {
        "less_popular_agreement": "the lower tertile of exacta pool totals within each logged "
        "day (the odds log's investment row), reported as its agreement with the race-card "
        "proxy, not a selection variable: a whole-day tertile uses later races' totals",
    },
    "primary_metric": "expected return per 100 yen staked at the final dividend, takeout and "
    "breakage in, with its CI (owner reframe 2026-11-27 on #131: rank every option by expected "
    "return at the final price, CI beside it)",
    "revision": 'v4 (review of 83684eb, 2026-11-27): rule-v3\'s headline took "less popular" '
    "from the lower tertile of each logged day's pool totals, which later races set; v4 takes it "
    "from the race card (race_no 1-4 or DEBUT/MAIDEN, E1's popularity segment), reports the "
    "tertile as an agreement diagnostic only and scores the logged development days available "
    "at each run with no meeting gate, and changes nothing else. v3 (review of 0c39a8d): the "
    "favourite and the category from the board at T-2, not the final Win odds",
}


def _e1_members() -> list[dict]:
    return [{"category": c, "cell": cell} for c in CATEGORIES for cell in SEGMENT_CELLS]


def _e1_like(estimator: str, lambda2, estimand: str, **extra) -> dict:
    lambdas = lambda2 if isinstance(lambda2, list) else [lambda2]
    return {
        **copy.deepcopy(_COMMON),
        "estimator": estimator,
        "estimand": estimand,
        "members": _e1_members(),
        "lambda2": copy.deepcopy(lambda2),
        "family_size": len(lambdas) * len(CATEGORIES) * len(SEGMENT_CELLS),
        "directions": dict(DIRECTIONS),
        "segments": copy.deepcopy(_SEGMENTS),
        "interval": "meeting bootstrap of the estimand, 95% one-sided in each direction",
        **extra,
    }


def family_specs() -> dict[str, dict]:
    """The pre-registered families as plain-JSON dicts, a fresh copy on every call."""
    b2 = "rho_S = sum_r R^B2_{S,r} / sum_r rate_r (null 1)"
    specs = {
        E1_ID: _e1_like(
            "B2",
            list(LAMBDA2S),
            b2,
            loss="rate - R for fav_second and no_fav, R - rate for fav_first (e1_loss)",
            benchmark="zero loss",
            max_test="Hansen SPA and Romano-Wolf StepM over all lambda2 x category x cell "
            "members (84), cluster sums per meeting",
            shrinkage="normal-normal empirical Bayes of the 14 cell estimates per category and "
            "lambda2 on their bootstrap SEs (DerSimonian-Laird tau2)",
        ),
        E1_B1_ID: _e1_like(
            "B1",
            FITTED,
            "rho_S = sum_r R^B1_{S,r} / sum_r rate_r (null 1); B1 does not use p",
            role="E1' reported variant, not in E1's max test",
        ),
        E1_B3_ID: _e1_like(
            "B3",
            list(LAMBDA2S),
            "mean_r R^B3_{S,r} (null 1: the ordering model is calibrated on S)",
            role="E1'' diagnostic, not in E1's max test",
        ),
        C_ID: {
            **copy.deepcopy(_COMMON),
            "estimator": "C",
            "estimand": "mean_r [log q_w - log p_w], q_w = implied_share(D_w, rate) "
            "(NaN at or below 110 yen: race left out), totals only",
            "members": [{"category": "all", "cell": "all"}],
            "lambda2": list(LAMBDA2S),
            "family_size": len(LAMBDA2S),
            "interval": "paired meeting bootstrap of the mean, 95% one-sided bounds",
        },
        E2_ID: {
            **copy.deepcopy(_COMMON),
            "estimator": "rule",
            "rule": "at T-2 buy exacta c of category X when q_c(t) / p_c(t) < k; q_c(t) from "
            "the exacta board at t, p_c(t) Harville on the Win board at t",
            "members": [
                {"category": c, "cut": k, "scope": s}
                for c in ("fav_second", "no_fav")
                for k in (0.6, 0.7, 0.8, 0.9)
                for s in ("all", "headline")
            ],
            "cuts": [0.6, 0.7, 0.8, 0.9],
            "stake_yen": 100,
            "lambda2": FITTED,
            "family_size": 16,
            "judged_at": "final dividends, own impact from the payout identity with the race's "
            "logged pool totals",
            "benchmark": "no bet (loss 0)",
            "scored_on": "#130-logged development days available at each run, with n and the CI, "
            "re-scored as more days are logged",
            **copy.deepcopy(_E2_SELECTION),
        },
    }
    return {fid: {**spec, "supersedes": SUPERSEDES[fid]} for fid, spec in specs.items()}
