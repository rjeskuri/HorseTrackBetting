"""The C4 meta-model at the final Win price on development folds (#87; 08 §8 B3, §4 E40, E9, R13).

Members are out-of-fold level-1 probabilities in a Parquet keyed by race_id and horse_id: for this
run, HKTrack's live model `m-v1-20261115-6102e309` (research-24's form model and two members
without HKTrack's `f_jrdb_*` features). The form model reads JRDB, and the other two are not
confirmed JRDB-free (their pedigree keys), so every figure built on a member is treated as
JRDB-derived (D21): its records carry the source `jrdb`, the registry summary withholds
their values and the full report goes through `write_report`, outside the repository. HKTrack is
read, never written or run: only the key and member columns of the file.

The final Win price and outcomes come from this repository's JRA store (`read_store_table`:
column- and date-scoped, holdout days dropped, nothing before 2014-06-07; then `split_off`'s
development rows; the holdout is never opened). q is the normalised final Win price. The D4
baseline is the calibrated market (`models.v0.calibration`, a spline in ln q) fitted in-fold on
D9's walk-forward folds over the store's development races, dead-heat races out of its fit. The
meta-models (`models.meta`) are fitted on the member rows on or before each fold's train_end and
predict its test window through `oof_predictions`: stage 2 on ln q, and the offset form on ln q
and on ln p_cal (the baseline itself, so delta = 0 is the D4 market exactly). Only folds from
2024-Q1 are scored: 2023 is HKTrack's seed season and only trains. Two variants read no member
(q against the baseline, the baseline's own ceiling table) and are scored on every development
race of the scored folds, from the store alone.

Order: freeze the family's spec in the run registry (before any outcome is read), read the store
and the members, score, append one registry record per variant (failures included) and write the
report. CLI: `python -m horsetrack.countries.japan.studies.meta --run --members PATH [--out DIR]`.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from horsetrack.core import paths
from horsetrack.core.config import load_source
from horsetrack.core.eval.folds import Fold, FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha, split_off
from horsetrack.core.eval.registry import RunRecord, default_registry
from horsetrack.core.eval.reports import write_report
from horsetrack.core.eval.scoring import gain, per_race_log_loss
from horsetrack.core.pricing import clean_odds
from horsetrack.countries.japan.models.meta import (
    MetaFit,
    MetaSpec,
    ceiling_table,
    encompassing,
    fit_meta,
    oof_predictions,
)
from horsetrack.countries.japan.models.v0.calibration import fit_win_calibration
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table
from horsetrack.countries.japan.studies.calibration import (
    Study,
    Variant,
    _bounds,
    _markdown_table,
    _metric,
    bootstrap_settings,
    store_builds,
)

COUNTRY = "japan"
TICKET = 87
FAMILY_ID = "japan-c4-meta-final-win"
STORE_SOURCES = ("jra",)
MEMBER_SOURCES = ("jra", "jrdb")
BOOTSTRAP_SEED = 87  # reproducibility only; scoring.yaml signs no seed
WALD_Z = 1.959963984540054  # 95% two-sided Wald intervals on coefficients, as in #77
MEMBER_FILE = "m-v1-20261115-6102e309.parquet"  # under HKTrack's outputs/oof/
MEMBERS = {
    "form": (
        "p_fundamental_oof",
        "research-24's form model: HKTrack's ten-member stage-1 blend (primary_oof = blend), no "
        "odds input; reads JRDB's f_jrdb_* indices",
    ),
    "lgbm_no_jrdb": (
        "p_lgbm_no_jrdb_oof",
        "LightGBM race softmax without the f_jrdb_* features (HKTrack config/experiments/"
        "ten-model-ensemble.yaml); not confirmed JRDB-free: its pedigree features are keyed on "
        "horse_pedigree, which HKTrack's JRDB UKC build rewrites, and the v1 frame cannot be "
        "tied to a netkeiba-only table; treated as JRDB-derived",
    ),
    "clogit_no_jrdb": (
        "p_clogit_no_jrdb_oof",
        "conditional logit without the f_jrdb_* features; not confirmed JRDB-free, as "
        "lgbm_no_jrdb; treated as JRDB-derived",
    ),
}
MEMBER_KEYS = ["race_id", "horse_id", "race_date", "fold_season"]
BASELINE_FROM = date(2023, 1, 1)  # the baseline is predicted from here: the member rows' span
SCORED_FROM = date(2024, 1, 1)  # HKTrack's scored seasons; 2023 seeds stage 2 there and here
FORMS = {  # variant prefix -> (form, the market column the meta conditions on)
    "stage2:q": ("stage2", "ln_q"),
    "offset:q": ("offset", "ln_q"),
    "offset:cal": ("offset", "ln_cal"),
}
OVERLAY_RATIOS = (1.0, 1.1, None, 1.5, 2.0)  # None: break-even, 1 / (1 - Win takeout)
RUNNER_COLUMNS = ["race_id", "race_date", "horse_id", "win_odds", "finish_pos", "is_scratched"]
RACE_COLUMNS = ["race_id", "race_date", "meeting_id", "status"]
DECISIONS = {
    "D1": "A: condition on the final price; this family conditions on the final Win price",
    "D4": "the Win baseline is the calibrated market (spline in ln q), fitted in-fold",
    "D6": "winner log loss per race, a dead heat at 1/k per co-winner (#69 answer); pseudo-R2 "
    "against equal chances beside nats per race",
    "D7": "meeting (venue x day) clusters, 95% one-sided (config/scoring.yaml)",
    "D9": "walk-forward quarters, expanding, 1-week embargo, nested out-of-fold "
    "(config/folds.yaml; #70 answer)",
    "D10": "no price before 2014-06-07",
    "D11-D15": "holdout A2 from 2026-11-28, fully locked; no holdout row is read; not opened",
    "D16": "adoption needs a holdout gain lower bound above 0 and a positive encompassing "
    "weight: nothing is adopted here",
    "D21": "no figure derived from JRDB in the repository: every member-based figure is withheld",
    "#87": "the owner's comment of 2026-11-24: JRDB as a model input in private models on this "
    "Mac only, results never published; the shrinkage target is alpha = 0, the market alone",
}


@dataclass(frozen=True)
class MetaVariant(Variant):
    sources: tuple[str, ...] = MEMBER_SOURCES


# --------------------------------------------------------------------------- inputs and spec


def overlay_thresholds() -> list[float]:
    """E9's thresholds on ln(pi/q): ln of each ratio, break-even at the Win pool's takeout."""
    takeout = float(load_source(COUNTRY, "jra").pools["WIN"].takeout)
    return [math.log(1.0 / (1.0 - takeout) if r is None else r) for r in OVERLAY_RATIOS]


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_spec(scheme: FoldScheme, bootstrap: dict, thresholds: list[float], digest: str) -> dict:
    """The family's pre-registered spec, frozen in the registry before any outcome is read."""
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypotheses": {
            "E40": "stage 2 at the final Win price (u = alpha * m + beta * ln q, unbounded) "
            "replicates research-24's stage-2 gain over the market (HKTrack docs/research/24 "
            "section 2; ledger, figures withheld) in this harness, as out-of-fold ΔLL per race "
            "against the D4 calibrated market; the residual (offset) form gives alpha_T > 0; "
            "each member's encompassing coefficient beside ln q",
            "E9": "the out-of-fold ln(pi / q_final) of each meta variant in the Win pool: the "
            "q-mass above each threshold (overlays, one-sided) and the q-weighted RMS",
        },
        "decision_rules": {
            "E40 replication": "replicated when stage2:q:form's ΔLL lower bound is above 0; "
            "otherwise a documented failure",
            "E40 offset": "alpha_T > 0 for a member when the 95% Wald CI of its delta in the "
            "pooled offset fit over the scored rows (in-sample) lies above 0",
            "encompassing": "positive when the coefficient's 95% Wald CI lies above 0",
            "E9": "descriptive: no rule; the break-even threshold is ln 1 / (1 - Win takeout)",
        },
        "target": "the race's winner (Win pool), conditioned on the final Win price (D1 A)",
        "members": {name: {"column": col, "what": what} for name, (col, what) in MEMBERS.items()},
        "member_file": {
            "name": f"HKTrack outputs/oof/{MEMBER_FILE}",
            "sha256": digest,
            "columns_read": MEMBER_KEYS + [col for col, _ in MEMBERS.values()],
            "level1": "HKTrack's season walk-forward: a prediction for a race of season S comes "
            "from models fitted on seasons before S (HKTrack src/hktrack/eval/walkforward.py); "
            "checked: every row's fold_season is its race's calendar year",
            "join": "inner on (race_id, horse_id) with the store's priced starters; a race "
            "with a starter the file lacks is dropped and counted",
            "member_value": "ln of the member's probability normalised over the race's starters",
        },
        "markets": {
            "ln_q": "ln of the normalised final Win price, from the JRA store",
            "ln_cal": "ln of the D4 calibrated market out of fold (the baseline itself)",
        },
        "variants": {
            "market:normalised": "q itself against the baseline (offset:q at delta = 0); every "
            "development race of the scored folds, from the store alone",
            "ceiling:calibrated_market": "the baseline's own ceiling table against q; the same "
            "races",
            "member_alone:<m>": "the member's own probability against the baseline",
            "stage2:q:<m>": "u = alpha * m + beta * ln q, alpha and beta unbounded",
            "offset:q:<m>": "u = ln q + delta * m",
            "offset:cal:<m>": "u = ln p_cal + delta * m (delta = 0 is the baseline)",
            "encompassing:<m>": "models.meta.encompassing on the scored rows, in-sample",
            "ceiling:<variant>": "models.meta.ceiling_table of the variant's OOF pi against q",
        },
        "meta": {
            "fit": "models.meta.fit_meta: conditional-logit MLE, no bounds, ridge 0, "
            "quadratic off, a dead heat at 1/k, void races dropped",
            "segments": "none: no segment set is pre-registered for alpha(z) (the owner's #125 "
            "set is for the residual's slopes on ln q; E42 names others); left to the "
            "selection study",
            "coefficient_ci": "95% two-sided Wald from the inverse information (races "
            "independent), per fold on the fold's training rows",
        },
        "folds": {
            "scheme": dataclasses.asdict(scheme),
            "calendar": "walk_forward over the store's development starters from 2014-06-07",
            "baseline": "folds from 2023-Q1, fitted on the store's rows on or before train_end, "
            "dead-heat races out of the fit (as #114's run_fit)",
            "meta_training_rows": "member rows on or before the fold's train_end; the file "
            "starts in 2023, so the first folds train on fewer than D9's five seasons, which "
            "the member file cannot give",
            "scored": "folds from 2024-Q1: HKTrack's scored seasons; its seed season 2023 only "
            "trains",
        },
        "delta_ll": "per scored race, the baseline's winner log loss minus the variant's "
        "(core.eval.scoring.gain), meeting-clustered paired bootstrap bounds",
        "ceiling": {
            "thresholds": thresholds,
            "ratios": list(OVERLAY_RATIOS),
            "bounds": "per-race q-mass above each threshold and per-race q-weighted mean "
            "square, meeting-clustered bootstrap; rms bounds are their square roots",
        },
        "bootstrap": bootstrap,
        "adoption": "none (D16): no holdout is opened and nothing is adopted",
        "price_era_start": PRICE_ERA_START.isoformat(),
        "decisions": DECISIONS,
        "sources": list(MEMBER_SOURCES),
    }


# --------------------------------------------------------------------------- rows


def read_development() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Development runners and races of the price era, column- and date-scoped."""
    frames = []
    for name, columns in (("runners", RUNNER_COLUMNS), ("races", RACE_COLUMNS)):
        table = read_store_table(
            name, columns=columns, start=PRICE_ERA_START, end=date.max, touches_price=True
        )
        development, _ = split_off(table.to_pandas(date_as_object=False), "race_date", COUNTRY)
        frames.append(development)
    return frames[0], frames[1]


def store_frame(runners: pd.DataFrame, races: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per priced starter of a run race with a winner, sorted by date and race: race_id,
    race_date, horse_id, win_odds (cleaned), won, dead_heat (its race's), meeting_id and q."""
    run = races["status"].eq("run")
    kept = runners.loc[runners["race_id"].isin(races.loc[run, "race_id"])]
    scratched = kept["is_scratched"].eq(True).to_numpy(dtype=bool, na_value=False)
    odds = clean_odds(kept["win_odds"].to_numpy())
    starting = ~scratched & ~np.isnan(odds)
    won = kept["finish_pos"].eq(1).to_numpy(dtype=bool, na_value=False)
    starters = kept.loc[starting, ["race_id", "race_date", "horse_id"]].assign(
        win_odds=odds[starting], won=won[starting]
    )
    winners = starters.groupby("race_id")["won"].transform("sum")
    frame = starters.loc[winners > 0].assign(dead_heat=(winners >= 2)[winners > 0])
    frame = frame.merge(races[["race_id", "meeting_id"]], on="race_id", validate="many_to_one")
    frame = frame.sort_values(["race_date", "race_id"], kind="stable").reset_index(drop=True)
    inv = 1.0 / frame["win_odds"]
    frame["q"] = inv / inv.groupby(frame["race_id"]).transform("sum")
    counts = {
        "races_read": len(races),
        "void_races": int((~run).sum()),
        "runners_read": len(runners),
        "scratched_runners": int(scratched.sum()),
        "unpriced_starters": int((~scratched & np.isnan(odds)).sum()),
        "races_without_a_priced_winner": int(starters["race_id"].nunique())
        - int(frame["race_id"].nunique()),
        "dead_heat_races": int(frame.loc[frame["dead_heat"], "race_id"].nunique()),
        "races": int(frame["race_id"].nunique()),
        "starters": len(frame),
    }
    return frame, counts


def baseline_oof(frame: pd.DataFrame, folds: list[Fold]) -> pd.Series:
    """D4's calibrated market out of fold on the folds' test rows; NaN elsewhere."""

    def fit(train: pd.DataFrame):
        return fit_win_calibration(train.loc[~train["dead_heat"]])

    def predict(model, test: pd.DataFrame) -> np.ndarray:
        return model.predict_many(test["q"].to_numpy(), test["race_id"].to_numpy())

    return oof_predictions(frame, folds, fit, predict)


def read_members(path: Path) -> pd.DataFrame:
    """The member file's key and member columns, development days only."""
    columns = MEMBER_KEYS + [col for col, _ in MEMBERS.values()]
    members = pq.read_table(path, columns=columns).to_pandas()
    members, _ = split_off(members, "race_date", COUNTRY)
    season = members["fold_season"].str[:4].astype(int)
    if not (season == pd.to_datetime(members["race_date"]).dt.year).all():
        raise ValueError("a member row's fold_season is not its race's calendar year")
    return members


def member_frame(frame: pd.DataFrame, members: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The store rows of the member span with each member's log-probability (normalised over the
    race's starters); a race with a starter the file lacks is dropped. Index: store positions."""
    first, last = members["race_date"].min(), members["race_date"].max()
    span = frame.loc[(frame["race_date"] >= first) & (frame["race_date"] <= last)]
    keyed = members.drop(columns=["race_date", "fold_season"])
    joined = span.reset_index().merge(
        keyed, on=["race_id", "horse_id"], how="left", validate="one_to_one", indicator=True
    )
    found = joined["_merge"].eq("both")
    whole = found.groupby(joined["race_id"]).transform("all")
    meta = joined.loc[whole].drop(columns="_merge").set_index("index").rename_axis(None)
    for name, (col, _) in MEMBERS.items():
        p = meta[col] / meta[col].groupby(meta["race_id"]).transform("sum")
        meta[name] = np.log(p.to_numpy())
    meta = meta.drop(columns=[col for col, _ in MEMBERS.values()])
    in_store = members.merge(span[["race_id", "horse_id"]], on=["race_id", "horse_id"])
    counts = {
        "member_rows": len(members),
        "member_rows_not_a_store_starter": len(members) - len(in_store),
        "store_starters_in_span_without_a_member_row": int((~found).sum()),
        "races_dropped_for_a_missing_starter": int(joined.loc[~whole, "race_id"].nunique()),
        "member_races": int(meta["race_id"].nunique()),
        "member_starters": len(meta),
    }
    return meta, counts


def restrict(folds: list[Fold], positions: np.ndarray, n: int) -> list[Fold]:
    """The folds as positions into the rows at `positions` (ascending positions of n rows)."""
    where = np.full(n, -1)
    where[positions] = np.arange(len(positions))
    out = []
    for fold in folds:
        train, test = where[fold.train_idx], where[fold.test_idx]
        out.append(dataclasses.replace(fold, train_idx=train[train >= 0], test_idx=test[test >= 0]))
    return out


# --------------------------------------------------------------------------- the pure study


@dataclass
class Scored:
    """One race set's scored rows (the scored folds' test rows, with their fold) and the
    baseline every variant on it is compared with."""

    rows: pd.DataFrame
    base_ll: pd.Series
    clusters: pd.Series
    uniform: pd.Series  # ln(starters) per race: the equal-chances loss of D6's pseudo-R2

    @classmethod
    def of(cls, frame: pd.DataFrame, folds: list[Fold]) -> Scored:
        parts = [
            frame.iloc[f.test_idx].assign(fold=f.test_start.isoformat())
            for f in folds
            if len(f.test_idx)
        ]
        if not parts:
            raise ValueError("no scored fold has a test row")
        rows = pd.concat(parts)
        race = rows["race_id"].to_numpy()
        return cls(
            rows=rows,
            base_ll=_ll(rows["p_cal"], rows),
            clusters=rows["meeting_id"].groupby(race).first(),
            uniform=np.log(rows.groupby(race).size().astype(float)),
        )

    @property
    def n_races(self) -> int:
        return len(self.base_ll)


def _ll(p: pd.Series, rows: pd.DataFrame) -> pd.Series:
    no = pd.Series(False, index=rows.index)
    return per_race_log_loss(p.reindex(rows.index), rows["won"], rows["race_id"], no).per_race


def _wald(fold: str, metric: str, c: float, s: float, n: int):
    return _metric(fold, metric, c, n, c - WALD_Z * s, c + WALD_Z * s)


def _delta(p: pd.Series, scored: Scored, spec: dict, name: str) -> tuple[list, dict]:
    """ΔLL per race against the baseline: per fold (point) and over the scored folds (bounds)."""
    rows = scored.rows
    ll = _ll(p, rows)
    g = gain(ll, scored.base_ll)
    iv = _bounds(g.per_race, scored.clusters, spec)
    fold_of = rows["fold"].groupby(rows["race_id"].to_numpy()).first()
    metrics = [
        _metric(fold, "delta_ll_vs_calibrated_market", part.mean(), len(part))
        for fold, part in g.per_race.groupby(fold_of.reindex(g.per_race.index).to_numpy())
    ]
    r2 = g.per_race.sum() / scored.uniform.reindex(g.per_race.index).sum()
    metrics += [
        _metric("all_folds", "mean_log_loss", ll.mean(), len(ll)),
        _metric("all_folds", "baseline_mean_log_loss", scored.base_ll.mean(), scored.n_races),
        _metric("all_folds", "delta_ll_vs_calibrated_market", iv.mean, g.n_common, iv.lo, iv.hi),
        _metric("all_folds", "delta_pseudo_r2_vs_calibrated_market", r2, g.n_common),
    ]
    row = {"variant": name, "races": g.n_common, "meetings": iv.n_clusters}
    row |= {"mean log loss": ll.mean(), "baseline": scored.base_ll.mean()}
    row |= {"ΔLL": iv.mean, "lower": iv.lo, "upper": iv.hi, "Δ pseudo-R2": r2}
    return metrics, {"delta": [row]}


def _ceiling(pi: pd.Series, scored: Scored, spec: dict, name: str) -> tuple[list, dict]:
    """E9's table of pi against q, with meeting-clustered bounds on each figure."""
    rows = scored.rows
    pi = pi.reindex(rows.index)
    thresholds = spec["ceiling"]["thresholds"]
    frame = pd.DataFrame({"race_id": rows["race_id"], "pi": pi, "q": rows["q"]})
    table = ceiling_table(frame, pi_col="pi", q_col="q", thresholds=thresholds)
    q, race = rows["q"].to_numpy(), rows["race_id"].to_numpy()
    x = np.log(pi.to_numpy() / q)

    def bounds(values: np.ndarray):
        return _bounds(pd.Series(values).groupby(race).sum(), scored.clusters, spec)

    metrics, out = [], []
    for t, mass in zip(thresholds, table["q_mass"], strict=True):
        iv = bounds(q * (x > t))
        metrics.append(
            _metric("all_folds", f"q_mass_above[{t:.6f}]", mass, scored.n_races, iv.lo, iv.hi)
        )
        out.append(
            {"variant": name, "threshold": t, "ratio": math.exp(t), "q-mass": mass}
            | {"lower": iv.lo, "upper": iv.hi}
        )
    iv = bounds(q * x * x)
    rms, lo, hi = float(table["rms"].iloc[0]), math.sqrt(iv.lo), math.sqrt(iv.hi)
    metrics.append(_metric("all_folds", "rms_ln_pi_over_q", rms, scored.n_races, lo, hi))
    out.append(
        {"variant": name, "threshold": "q-weighted RMS", "q-mass": rms, "lower": lo, "upper": hi}
    )
    return metrics, {"ceiling": out}


@dataclass
class _Outcome:
    """A computation's value, or the error it raised (re-raised on `get`)."""

    value: object = None
    error: Exception | None = None

    def get(self):
        if self.error is not None:
            raise self.error
        return self.value


def _try(compute: Callable[[], object]) -> _Outcome:
    try:
        return _Outcome(value=compute())
    except Exception as exc:  # recorded by each variant that needs it, never dropped
        return _Outcome(error=exc)


def _meta_oof(rows: pd.DataFrame, folds: list[Fold], spec: MetaSpec, market: str):
    """The meta's out-of-fold probabilities on `rows` and each fold's fit, in fold order."""
    member = spec.members[0]
    data = rows[["race_id", "won", member]].assign(ln_q_final=rows[market])
    fits: list[MetaFit] = []

    def fit(train: pd.DataFrame) -> MetaFit:
        fits.append(fit_meta(train, spec))
        return fits[-1]

    def predict(model: MetaFit, test: pd.DataFrame) -> np.ndarray:
        return model.predict(test, np.exp(test["ln_q_final"].to_numpy()))

    return oof_predictions(data, folds, fit, predict), fits


def _meta_metrics(
    outcome: _Outcome, folds: list[Fold], scored: Scored, spec: dict, name: str, market: str
) -> tuple[list, dict]:
    """ΔLL, each fold's coefficients (R13: unbounded, Wald CI) and, for the offset form, the
    pooled in-sample delta over the scored rows."""
    p, fits = outcome.get()
    metrics, tables = _delta(p, scored, spec, name)
    tables["coefficients"] = []
    for fit, fold in zip(fits, folds, strict=True):
        label = fold.test_start.isoformat()
        for term in fit.coef.index:
            c, s = float(fit.coef[term]), float(fit.se[term])
            metrics.append(_wald(label, f"coef[{term}]", c, s, fit.n_races))
            tables["coefficients"].append(
                {"variant": name, "fold": label, "train to": fold.train_end}
                | {"train races": fit.n_races, "term": term, "coef": c, "se": s}
                | {"lower 95%": c - WALD_Z * s, "upper 95%": c + WALD_Z * s}
            )
    mspec = fits[0].spec
    if mspec.form == "offset":
        member = mspec.members[0]
        rows = scored.rows[["race_id", "won", member]].assign(ln_q_final=scored.rows[market])
        pooled = fit_meta(rows, mspec)
        c, s = float(pooled.coef[member]), float(pooled.se[member])
        metrics.append(_wald("scored_rows_in_sample", f"coef[{member}]", c, s, pooled.n_races))
        tables["pooled"] = [
            {"variant": name, "races": pooled.n_races, "delta": c, "se": s}
            | {"lower 95%": c - WALD_Z * s, "upper 95%": c + WALD_Z * s}
        ]
    return metrics, tables


def _encompassing(scored: Scored, member: str) -> tuple[list, dict]:
    rows = scored.rows[["race_id", "won", member]].assign(ln_q_final=scored.rows["ln_q"])
    c, s = encompassing(rows, member)
    row = {"member": member, "races": scored.n_races, "coef": c, "se": s}
    row |= {"lower 95%": c - WALD_Z * s, "upper 95%": c + WALD_Z * s}
    return [_wald("scored_rows_in_sample", "encompassing_coef", c, s, scored.n_races)], {
        "encompassing": [row]
    }


def compute_study(frame: pd.DataFrame, meta: pd.DataFrame, folds: list[Fold], spec: dict) -> Study:
    """Every variant of the frozen spec. `frame` is the store frame with the out-of-fold baseline
    `p_cal`; `meta` its member rows (store positions as index, a log-probability column per
    member); `folds` the scored folds as store positions. Pure: no I/O."""
    scheme = "walk_forward:" + json.dumps(spec["folds"]["scheme"], sort_keys=True)
    study = Study(counts={})
    store = Scored.of(frame, folds)
    study.counts["store_scored_races"] = store.n_races
    q, cal = {"market": "normalised"}, {"pi": "calibrated_market"}
    study.attempt(
        MetaVariant("E40", "market:normalised", scheme, q, {}, sources=STORE_SOURCES),
        lambda: _delta(store.rows["q"], store, spec, "market:normalised"),
    )
    study.attempt(
        MetaVariant("E9", "ceiling:calibrated_market", scheme, cal, {}, sources=STORE_SOURCES),
        lambda: _ceiling(store.rows["p_cal"], store, spec, "ceiling:calibrated_market"),
    )

    p_cal = frame["p_cal"].reindex(meta.index).to_numpy()
    rows = meta.assign(p_cal=p_cal, ln_q=np.log(meta["q"]), ln_cal=np.log(p_cal))
    meta_folds = restrict(folds, rows.index.to_numpy(), len(frame))
    rows = rows.reset_index(drop=True)
    scored = Scored.of(rows, meta_folds)
    study.counts["member_scored_races"] = scored.n_races
    for member in MEMBERS:
        base = {"member": member, "score": "winner_log_loss", "baseline": "calibrated_market"}
        name = f"member_alone:{member}"
        alone = pd.Series(np.exp(rows[member].to_numpy()), index=rows.index)
        study.attempt(
            MetaVariant("E40", name, scheme, base, {}),
            lambda alone=alone, name=name: _delta(alone, scored, spec, name),
        )
        for prefix, (form, market) in FORMS.items():
            name = f"{prefix}:{member}"
            components = {**base, "form": form, "market": market}
            params = {"ridge": 0.0, "bounds": "none", "quadratic": False, "segments": []}
            mspec = MetaSpec([member], quadratic=False, segment_cols=[], form=form, ridge=0.0)
            outcome = _try(
                lambda mspec=mspec, market=market: _meta_oof(rows, meta_folds, mspec, market)
            )
            study.attempt(
                MetaVariant("E40", name, scheme, components, params),
                lambda o=outcome, n=name, m=market: _meta_metrics(
                    o, meta_folds, scored, spec, n, m
                ),
            )
            study.attempt(
                MetaVariant("E9", f"ceiling:{name}", scheme, components, params),
                lambda o=outcome, n=name: _ceiling(o.get()[0], scored, spec, f"ceiling:{n}"),
            )
        study.attempt(
            MetaVariant("E40", f"encompassing:{member}", scheme, {**base, "market": "ln_q"}, {}),
            lambda member=member: _encompassing(scored, member),
        )
    return study


# --------------------------------------------------------------------------- report and runner

SECTIONS = (
    (
        "delta",
        "ΔLL against the D4 calibrated market (out of fold)",
        "ΔLL is the baseline's winner log loss minus the variant's, per race, in nats (positive:"
        " better than the calibrated market). lower and upper are the 5% and 95% quantiles of the"
        " paired bootstrap over meetings (D7: one-sided). Δ pseudo-R2 is the summed ΔLL over the"
        " summed equal-chances loss (D6).",
    ),
    (
        "coefficients",
        "Each fold's coefficients (R13: unbounded; fitted against the final Win price)",
        "Wald 95% two-sided from the inverse information on the fold's training rows. No bounds.",
    ),
    ("pooled", "Offset form: delta pooled over the scored rows (in-sample, 95% Wald)", ""),
    ("encompassing", "Encompassing: the member beside ln q (scored rows, in-sample, 95% Wald)", ""),
    (
        "ceiling",
        "E9 ceiling: ln(pi / q) of the out-of-fold pi against the final Win q",
        "q-mass is the share of q on runners with ln(pi/q) above the threshold (ratio pi/q above"
        " the ratio); the break-even ratio is 1 / (1 - Win takeout). Bounds: meeting bootstrap,"
        " one-sided 5% and 95%.",
    ),
)


def render_report(study: Study, spec: dict, about: dict) -> str:
    """The full report: aggregate tables only, never a runner or race row."""
    lines = [
        f"# C4 meta-model at the final Win price ({COUNTRY}, #{TICKET})",
        "",
        *(f"- {key}: {value}" for key, value in about.items()),
        "- development rows only: holdout days are never read and the holdout is not opened",
        "- built on JRDB (D21): private, never published",
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


def _record(variant: MetaVariant, git_sha: str, builds: dict[str, str], member_build: str):
    licensed = variant.sources != STORE_SOURCES
    return RunRecord(
        hypothesis_id=variant.hypothesis_id,
        family_id=FAMILY_ID,
        variant=variant.name,
        ticket=TICKET,
        git_sha=git_sha,
        data_builds={**builds, **({"hktrack_member_oof": member_build} if licensed else {})},
        sources=variant.sources,
        fold_scheme=variant.fold_scheme,
        components=variant.components,
        params=variant.params,
        metrics=variant.metrics,
        status="failed" if variant.error else "scored",
    )


def run_study(members_path: Path, out_dir: Path | None = None) -> dict:
    """Run the pre-registered family on Japan's development data and write its private report.

    Raises `paths.DataDirInsideRepo` for an `out_dir` or a member file inside the repository and
    `FamilyAlreadyFrozen` when the family is frozen with another spec (another member file)."""
    members_path = paths.outside_repo(Path(members_path), what="members")
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    digest = file_digest(members_path)
    scheme = load_country_scheme(COUNTRY)
    boot = {**bootstrap_settings(), "seed": BOOTSTRAP_SEED}
    spec = build_spec(scheme, boot, overlay_thresholds(), digest)
    git_sha = _git_sha()
    registry = default_registry(COUNTRY)
    spec_hash = registry.freeze_family(FAMILY_ID, spec)  # before any outcome is read

    builds = store_builds()
    member_build = f"HKTrack:outputs/oof/{MEMBER_FILE}@{digest[:16]}"
    runners, races = read_development()
    frame, counts = store_frame(runners, races)
    del runners, races
    folds = walk_forward(frame["race_date"], scheme=scheme)
    baseline = [f for f in folds if f.test_start >= BASELINE_FROM]
    scored = [f for f in folds if f.test_start >= SCORED_FROM]
    frame["p_cal"] = baseline_oof(frame, baseline)
    meta, member_counts = member_frame(frame, read_members(members_path))
    study = compute_study(frame, meta, scored, spec)
    study.counts = {**counts, **member_counts, "scored_folds": len(scored), **study.counts}
    del frame, meta

    existing = registry.records(FAMILY_ID)
    records = [_record(v, git_sha, builds, member_build) for v in study.variants]
    appended = [registry.append(r) for r in records if r not in existing]
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "store": ", ".join(builds.values()),
        "members": member_build,
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = write_report(
        out_dir / f"meta-{COUNTRY}-{spec_hash[:12]}.md",
        render_report(study, spec, about),
        sources=MEMBER_SOURCES,
    )
    return {
        "family_id": FAMILY_ID,
        "spec_hash": spec_hash,
        "report_path": str(report),
        "registry_path": str(registry.path),
        "records_appended": len(appended),
        "records_already_present": len(records) - len(appended),
        "failed_variants": [v.name for v in study.variants if v.error],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m horsetrack.countries.japan.studies.meta")
    parser.add_argument("--run", action="store_true", required=True, help="run the family")
    parser.add_argument(
        "--members", type=Path, required=True, help=f"the member file ({MEMBER_FILE})"
    )
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    args = parser.parse_args(argv)
    print(json.dumps(run_study(args.members, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
