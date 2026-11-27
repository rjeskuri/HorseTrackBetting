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
report. CLI: `python -m horsetrack.countries.japan.studies.meta --run --members PATH [--out DIR]
[--allow-short-history]`.

Added after the outcomes were read (review findings on PR #308), outside the frozen spec, which
is unchanged: the meeting-clustered coefficient bounds (D7's unit and level, 95% one-sided; the
`meeting_clustered:` variants, one registry record each) beside the frozen 95% two-sided Wald
intervals; the report's mechanical reading of each frozen rule under both; and the level-2
training window per scored fold against D9's 5-season minimum, which no scored fold meets. The
owner's answer (b) on #87 (2026-11-25) records the run as a documented failure to meet D9 (level 2
below the 5-season minimum; level 1 refit once a season, not quarterly): it supports no adoption
claim, and the D9-compliant run comes with #91's and #98's members. A rerun appends only records
the family does not already hold: a record equal to one in the registry but for its git commit is
the same run reproduced and is not appended.

Since #418 the level-2 path goes through `models.stacking`'s guard before any fit: the member file's
predictions are joined to the store (`join_members`, provenance from `fold_season`) and checked
(`check_level2`). A scored fold under D9's 5 seasons refuses (`IneligibleFold`) unless the run opts
in with `allow_short_history` (`--allow-short-history`), the owner-approved replication above; the
report says when a run used it. Provenance is checked either way.
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
from statistics import NormalDist

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
WALD_Z = 1.959963984540054  # the frozen spec's 95% two-sided Wald intervals, as in #77
CLUSTER = "meeting_id"  # D7's unit, the meeting (venue x day): the ΔLL bootstrap's clusters too
CLUSTERED = "meeting_clustered"  # the prefix of the D7 coefficient-bound variants
D9_SIGNED = (  # D9 as signed (docs/problem-statement.md, decisions table)
    "Quarter folds, quarterly refit, expanding, 5 seasons minimum, nested OOF, 1-week embargo"
)
D9_MIN_SEASONS = 5  # D9: "5 seasons minimum"
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
# The owner decisions exactly as the family's spec froze them on 2026-11-25 (`build_spec`). The
# registry refuses the run if a byte of the spec changes, so this stays as it is; the report
# renders DECISIONS below, which quotes D9 in full and states where this run departs from it.
SPEC_DECISIONS = {
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
DECISIONS = {
    "D1": "A: condition on the final price; this family conditions on the final Win price",
    "D4": "the Win baseline is the calibrated market (spline in ln q), fitted in-fold",
    "D6": "winner log loss per race, a dead heat at 1/k per co-winner (#69 answer); pseudo-R2 "
    "against equal chances beside nats per race",
    "D7": "meeting (venue x day) clusters, 95% one-sided (config/scoring.yaml): the ΔLL and "
    "ceiling bootstraps and the meeting-clustered coefficient bounds; the frozen spec's "
    "coefficient rules read 95% two-sided Wald intervals instead (see Decision rules)",
    "D9": f'"{D9_SIGNED}" (problem statement §4; config/folds.yaml; #70 answer). The baseline '
    "and the level-2 meta follow its folds, refit, window, nesting and embargo. The 5-season "
    "minimum holds for the baseline (store rows from 2014-06-07) but not at level 2 on any "
    "scored fold: the member file starts on 2023-01-05 (see D9 at level 2). Level 1 is HKTrack's "
    "season walk-forward, refit once a season, not quarterly. The owner's answer (b) on #87 "
    "(2026-11-25): the replication run is a documented failure to meet D9 at both levels and "
    "supports no adoption claim; the D9-compliant run comes with #91's and #98's members",
    "D10": "no price before 2014-06-07",
    "D11-D15": "holdout A2 from 2026-11-28, fully locked; no holdout row is read; not opened",
    "D16": "adoption needs a holdout gain lower bound above 0 (D7 level) and a positive "
    "encompassing weight with its interval above 0: nothing is adopted here",
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
        "decisions": SPEC_DECISIONS,
        "sources": list(MEMBER_SOURCES),
    }


# --------------------------------------------------------------------------- rows


def read_development(
    runner_columns: Sequence[str] = RUNNER_COLUMNS, race_columns: Sequence[str] = RACE_COLUMNS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Development runners and races of the price era, column- and date-scoped."""
    frames = []
    for name, columns in (("runners", runner_columns), ("races", race_columns)):
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


def member_predictions(members: pd.DataFrame, race_days: pd.Series) -> pd.DataFrame:
    """The member file as `models.stacking` predictions, one row per (race_id, horse_id, member):
    `logp` (ln of the file's probability; the guard checks only that it is there), `member_fold`
    (the row's `fold_season`) and `member_train_end`. Level 1 is a season walk-forward, trained on
    the seasons before `fold_season`, so its last training day is taken as the last of
    `race_days` before that season starts (not 31 December), NaT when none is."""
    days = np.sort(pd.to_datetime(pd.Series(race_days)).dt.normalize().unique())
    season = members["fold_season"].str[:4].astype(int)
    starts = pd.to_datetime(season.astype(str) + "-01-01").to_numpy()
    before = np.searchsorted(days, starts, side="left") - 1
    train_end = np.where(before >= 0, days[np.maximum(before, 0)], np.datetime64("NaT"))
    keys = members[["race_id", "horse_id"]].assign(
        member_fold=members["fold_season"].to_numpy(), member_train_end=train_end
    )
    return pd.concat(
        [
            keys.assign(member=name, logp=np.log(members[col].to_numpy()))
            for name, (col, _) in MEMBERS.items()
        ],
        ignore_index=True,
    )


def check_level2(
    frame: pd.DataFrame, folds: list[Fold], members: pd.DataFrame, *, allow_short_history: bool
) -> None:
    """`models.stacking`'s guard on the member rows of `folds`, before any fit: invalid
    provenance refuses (`InvalidProvenance`), and so does a joined training window under D9's
    seasons (`IneligibleFold`) unless `allow_short_history`."""
    # imported here: models.stacking imports D9_MIN_SEASONS and season_years from this module
    from horsetrack.countries.japan.models import stacking

    predictions = member_predictions(members, frame["race_date"])
    level2 = stacking.join_members(frame, folds, predictions, members=list(MEMBERS))
    stacking.check_level2(level2, min_seasons=0 if allow_short_history else D9_MIN_SEASONS)


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
    """The meta's out-of-fold probabilities on `rows` and each fold's fit (with its meeting-
    clustered se), in fold order."""
    member = spec.members[0]
    data = rows[["race_id", "won", CLUSTER, member]].assign(ln_q_final=rows[market])
    fits: list[MetaFit] = []

    def fit(train: pd.DataFrame) -> MetaFit:
        fits.append(fit_meta(train, spec, cluster_col=CLUSTER))
        return fits[-1]

    def predict(model: MetaFit, test: pd.DataFrame) -> np.ndarray:
        return model.predict(test, np.exp(test["ln_q_final"].to_numpy()))

    return oof_predictions(data, folds, fit, predict), fits


def _pooled(scored: Scored, spec: MetaSpec, market: str) -> MetaFit:
    """The offset form's in-sample fit over the scored rows, with its meeting-clustered se."""
    member = spec.members[0]
    rows = scored.rows[["race_id", "won", CLUSTER, member]].assign(ln_q_final=scored.rows[market])
    return fit_meta(rows, spec, cluster_col=CLUSTER)


def _one_sided_z(spec: dict) -> float:
    """D7's one-sided normal quantile at scoring.yaml's level: 1.645 at 95%."""
    return NormalDist().inv_cdf(spec["bootstrap"]["level"])


def _cells(c: float, s: float, s_meeting: float, z: float, coef: str = "coef") -> dict:
    """A coefficient's report cells: the frozen spec's Wald interval, then D7's meeting bounds."""
    return (
        {coef: c, "se": s, "lower 95% (Wald)": c - WALD_Z * s, "upper 95% (Wald)": c + WALD_Z * s}
        | {"se (meeting)": s_meeting, "lower (meeting, one-sided 95%)": c - z * s_meeting}
        | {"upper (meeting, one-sided 95%)": c + z * s_meeting}
    )


def _d7(fold: str, term: str, c: float, s_meeting: float, n: int, z: float) -> list:
    """A coefficient with D7's meeting bounds (each one-sided 95%) and its meeting-clustered se."""
    return [
        _metric(fold, f"coef[{term}]", c, n, c - z * s_meeting, c + z * s_meeting),
        _metric(fold, f"se_meeting[{term}]", s_meeting, n),
    ]


def _meta_metrics(
    outcome: _Outcome,
    pooled: _Outcome | None,
    folds: list[Fold],
    scored: Scored,
    spec: dict,
    name: str,
) -> tuple[list, dict]:
    """ΔLL, each fold's coefficients (R13: unbounded, Wald CI) and, for the offset form, the
    pooled in-sample delta over the scored rows. The tables also show D7's meeting bounds, whose
    registry metrics are the `meeting_clustered:` variant's (`_clustered_metrics`)."""
    p, fits = outcome.get()
    metrics, tables = _delta(p, scored, spec, name)
    z = _one_sided_z(spec)
    tables["coefficients"] = []
    for fit, fold in zip(fits, folds, strict=True):
        label = fold.test_start.isoformat()
        for term in fit.coef.index:
            c, s, sm = (float(v[term]) for v in (fit.coef, fit.se, fit.se_cluster))
            metrics.append(_wald(label, f"coef[{term}]", c, s, fit.n_races))
            tables["coefficients"].append(
                {"variant": name, "fold": label, "train to": fold.train_end}
                | {"train races": fit.n_races, "term": term}
                | _cells(c, s, sm, z)
            )
    if pooled is not None:
        fit = pooled.get()
        member = fit.spec.members[0]
        c, s, sm = (float(v[member]) for v in (fit.coef, fit.se, fit.se_cluster))
        metrics.append(_wald("scored_rows_in_sample", f"coef[{member}]", c, s, fit.n_races))
        tables["pooled"] = [
            {"variant": name, "races": fit.n_races, "meetings": int(scored.clusters.nunique())}
            | _cells(c, s, sm, z, coef="delta")
        ]
    return metrics, tables


def _clustered_metrics(
    outcome: _Outcome, pooled: _Outcome | None, folds: list[Fold], spec: dict
) -> tuple[list, dict]:
    """The `meeting_clustered:` variant of a meta variant: each fold's coefficients and the pooled
    delta with D7's meeting bounds (added after the outcomes were read; not in the frozen spec)."""
    _, fits = outcome.get()
    z = _one_sided_z(spec)
    metrics = []
    for fit, fold in zip(fits, folds, strict=True):
        for term in fit.coef.index:
            c, sm = float(fit.coef[term]), float(fit.se_cluster[term])
            metrics += _d7(fold.test_start.isoformat(), term, c, sm, fit.n_races, z)
    if pooled is not None:
        fit = pooled.get()
        member = fit.spec.members[0]
        c, sm = float(fit.coef[member]), float(fit.se_cluster[member])
        metrics += _d7("scored_rows_in_sample", member, c, sm, fit.n_races, z)
    return metrics, {}


def _encompassing_fits(scored: Scored, member: str) -> tuple[tuple, tuple]:
    """The member's encompassing coefficient with its inverse-information se, then with its
    meeting-clustered se (the same coefficient)."""
    rows = scored.rows[["race_id", "won", CLUSTER, member]].assign(ln_q_final=scored.rows["ln_q"])
    return encompassing(rows, member), encompassing(rows, member, cluster_col=CLUSTER)


def _encompassing(outcome: _Outcome, scored: Scored, member: str, spec: dict) -> tuple[list, dict]:
    (c, s), (_, sm) = outcome.get()
    row = {"member": member, "races": scored.n_races, "meetings": int(scored.clusters.nunique())}
    row |= _cells(c, s, sm, _one_sided_z(spec))
    return [_wald("scored_rows_in_sample", "encompassing_coef", c, s, scored.n_races)], {
        "encompassing": [row]
    }


def _clustered_encompassing(outcome: _Outcome, scored: Scored, spec: dict) -> tuple[list, dict]:
    (c, _), (_, sm) = outcome.get()
    z = _one_sided_z(spec)
    return _d7("scored_rows_in_sample", "encompassing_coef", c, sm, scored.n_races, z), {}


def season_years(first: date, last: date) -> float:
    """The seasons from race day `first` to `last`, both counted, as D9's minimum is read."""
    return ((last - first).days + 1) / 365.25


def _level2_window(rows: pd.DataFrame, folds: list[Fold]) -> list[dict]:
    """Each scored fold's level-2 training rows (the member rows on or before train_end) against
    D9's 5-season minimum: a coverage count of the store and the member file, no outcome."""
    out = []
    for fold in folds:
        train = rows.iloc[fold.train_idx]
        dates = pd.to_datetime(train["race_date"])
        first, last = (None, None) if train.empty else (dates.min().date(), dates.max().date())
        years = 0.0 if first is None else season_years(first, last)
        out.append(
            {"fold": fold.test_start.isoformat(), "test to": fold.test_end.isoformat()}
            | {"train from": first, "train to": last, "train end": fold.train_end.isoformat()}
            | {"years": round(years, 2), "train races": int(train["race_id"].nunique())}
            | {"train meetings": int(train[CLUSTER].nunique())}
            | {f"D9 {D9_MIN_SEASONS} seasons met": years >= D9_MIN_SEASONS}
        )
    return out


def _clustered_variant(variant: MetaVariant) -> MetaVariant:
    """The `meeting_clustered:` twin of a variant: the same components, D7's bounds."""
    return MetaVariant(
        variant.hypothesis_id,
        f"{CLUSTERED}:{variant.name}",
        variant.fold_scheme,
        {**variant.components, "se": "meeting_clustered"},
        {
            **variant.params,
            "inference": "meeting-clustered sandwich (models.condlogit), no small-sample factor",
            "level": "95% one-sided per bound (D7)",
            "added": "after the outcomes were read: review finding on PR #308; not in the "
            "frozen spec",
        },
    )


def compute_study(frame: pd.DataFrame, meta: pd.DataFrame, folds: list[Fold], spec: dict) -> Study:
    """Every variant of the frozen spec, then each meta and encompassing variant's
    `meeting_clustered:` twin. `frame` is the store frame with the out-of-fold baseline `p_cal`;
    `meta` its member rows (store positions as index, a log-probability column per member);
    `folds` the scored folds as store positions. Pure: no I/O."""
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
    study.tables["level2_window"] = _level2_window(rows, meta_folds)
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
            pooled = None
            if form == "offset":
                pooled = _try(lambda mspec=mspec, market=market: _pooled(scored, mspec, market))
            variant = MetaVariant("E40", name, scheme, components, params)
            study.attempt(
                variant,
                lambda o=outcome, pl=pooled, n=name: _meta_metrics(
                    o, pl, meta_folds, scored, spec, n
                ),
            )
            study.attempt(
                MetaVariant("E9", f"ceiling:{name}", scheme, components, params),
                lambda o=outcome, n=name: _ceiling(o.get()[0], scored, spec, f"ceiling:{n}"),
            )
            study.attempt(
                _clustered_variant(variant),
                lambda o=outcome, pl=pooled: _clustered_metrics(o, pl, meta_folds, spec),
            )
        fits = _try(lambda member=member: _encompassing_fits(scored, member))
        variant = MetaVariant(
            "E40", f"encompassing:{member}", scheme, {**base, "market": "ln_q"}, {}
        )
        study.attempt(variant, lambda f=fits, member=member: _encompassing(f, scored, member, spec))
        study.attempt(
            _clustered_variant(variant), lambda f=fits: _clustered_encompassing(f, scored, spec)
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
        "On the fold's training rows. No bounds. The frozen spec's interval is the Wald 95%"
        " two-sided from the inverse information (races independent). Beside it, D7's: the"
        " meeting-clustered sandwich se and the 95% one-sided bounds coef -/+ 1.645 se (meeting),"
        " added after the outcomes were read.",
    ),
    (
        "pooled",
        "Offset form: delta pooled over the scored rows (in-sample)",
        "The frozen rule reads the Wald 95% two-sided lower bound; D7's is the meeting one-sided"
        " 95% lower bound.",
    ),
    (
        "encompassing",
        "Encompassing: the member beside ln q (scored rows, in-sample)",
        "As the pooled delta: Wald 95% two-sided (frozen) and meeting one-sided 95% (D7).",
    ),
    (
        "ceiling",
        "E9 ceiling: ln(pi / q) of the out-of-fold pi against the final Win q",
        "q-mass is the share of q on runners with ln(pi/q) above the threshold (ratio pi/q above"
        " the ratio); the break-even ratio is 1 / (1 - Win takeout). Bounds: meeting bootstrap,"
        " one-sided 5% and 95%.",
    ),
)


ADDED_AFTER = (
    "**Added after the outcomes were read** (review finding on PR #308): the meeting-clustered"
    " coefficient bounds (D7: meetings, 95% one-sided) beside the frozen spec's 95% two-sided"
    " Wald intervals, which treat races as independent. The clustered bound is the interval that"
    " matches D7. The frozen spec is unchanged and names Wald, so the outcomes of its Wald rules"
    " (alpha_T > 0, encompassing positive) are provisional. The rules are read under both below."
)


def _rows_by(study: Study, key: str, by: str) -> dict[str, dict]:
    return {row[by]: row for row in study.tables.get(key, [])}


def _verdict(bound: float | None, holds: str, fails: str) -> str:
    return "not computed (the variant failed)" if bound is None else holds if bound > 0 else fails


def apply_rules(study: Study, spec: dict) -> list[dict]:
    """Each frozen decision rule read mechanically: its bound and outcome as frozen, then at D7
    (meetings, 95% one-sided). The replication's ΔLL bound is the meeting bootstrap, D7's own."""
    delta = _rows_by(study, "delta", "variant")
    pooled = _rows_by(study, "pooled", "variant")
    covering = _rows_by(study, "encompassing", "member")
    wald, d7 = "lower 95% (Wald)", "lower (meeting, one-sided 95%)"
    out = []
    name = "stage2:q:form"
    lo = delta.get(name, {}).get("lower")
    verdict = _verdict(lo, "replicated", "a documented failure")
    if lo is None:
        verdict += ": a documented failure"
    out.append(
        {"rule": "E40 replication", "variant": name}
        | {"statistic": "ΔLL lower bound (meeting bootstrap, one-sided 95%)"}
        | {"frozen bound": lo, "outcome (frozen)": verdict, "D7 bound": lo, "outcome (D7)": verdict}
    )
    for member in MEMBERS:
        for prefix, (form, _) in FORMS.items():
            if form != "offset":
                continue
            name, row = f"{prefix}:{member}", pooled.get(f"{prefix}:{member}", {})
            out.append(
                {"rule": "E40 offset", "variant": name, "statistic": "pooled delta, lower bound"}
                | {"frozen bound": row.get(wald), "D7 bound": row.get(d7)}
                | {"outcome (frozen)": _verdict(row.get(wald), "alpha_T > 0", "not shown")}
                | {"outcome (D7)": _verdict(row.get(d7), "alpha_T > 0", "not shown")}
            )
    for member in MEMBERS:
        row = covering.get(member, {})
        out.append(
            {"rule": "encompassing", "variant": f"encompassing:{member}"}
            | {"statistic": "coefficient, lower bound"}
            | {"frozen bound": row.get(wald), "D7 bound": row.get(d7)}
            | {"outcome (frozen)": _verdict(row.get(wald), "positive", "not shown")}
            | {"outcome (D7)": _verdict(row.get(d7), "positive", "not shown")}
        )
    out.append(
        {"rule": "E9", "variant": "ceiling:*", "statistic": "-"}
        | {"outcome (frozen)": "descriptive", "outcome (D7)": "descriptive"}
    )
    return out


def _bounded(what: str, lo: float, hi: float, unit: str = "") -> str:
    """What a pair of one-sided 95% bounds rules out about `what`."""
    above, below = f"{what} above {hi:+.6g}{unit}", f"{what} below {lo:+.6g}{unit}"
    if lo > 0:
        return f"rules out {what} <= 0 (and {above})"
    if hi < 0:
        return f"rules out {what} >= 0 (and {below})"
    return f"rules out neither sign, only {above} and {below}"


def rules_out(study: Study, spec: dict) -> list[str]:
    """What the run rules out, read from its one-sided 95% bounds at D7 (meetings): each ΔLL, each
    pooled delta and encompassing weight, and each E9 q-mass above the break-even ratio."""
    lines = [
        "- Every statement is at D7's level (meetings, 95% one-sided per bound), on development"
        " folds only, and from a run that is a documented failure to meet D9 (the owner's answer"
        " (b) on #87, 2026-11-25; see D9 at level 2), so none of it supports an adoption claim."
        " Nothing is tested on the holdout, so nothing is adopted (D16)."
    ]
    for row in study.tables.get("delta", []):
        lines.append(
            f"- `{row['variant']}`: "
            + _bounded("ΔLL against the calibrated market", row["lower"], row["upper"], " nats")
        )
    d7 = ("lower (meeting, one-sided 95%)", "upper (meeting, one-sided 95%)")
    for row in study.tables.get("pooled", []):
        lines.append(f"- `{row['variant']}`: " + _bounded("delta", row[d7[0]], row[d7[1]]))
    for row in study.tables.get("encompassing", []):
        lines.append(
            f"- `encompassing:{row['member']}`: "
            + _bounded("the encompassing weight", row[d7[0]], row[d7[1]])
        )
    even = spec["ceiling"]["thresholds"][OVERLAY_RATIOS.index(None)]
    for row in study.tables.get("ceiling", []):
        if row["threshold"] == even:
            lines.append(
                f"- `{row['variant']}`: a q-mass above the break-even ratio {math.exp(even):.4g}"
                f" greater than {row['upper']:.4g} is ruled out (point {row['q-mass']:.4g})"
            )
    return lines


def render_report(
    study: Study, spec: dict, about: dict, record_ids: dict[str, str] | None = None
) -> str:
    """The full report: aggregate tables only, never a runner or race row."""
    window = study.tables.get("level2_window", [])
    met = sum(bool(row[f"D9 {D9_MIN_SEASONS} seasons met"]) for row in window)
    lines = [
        f"# C4 meta-model at the final Win price ({COUNTRY}, #{TICKET})",
        "",
        *(f"- {key}: {value}" for key, value in about.items()),
        "- development rows only: holdout days are never read and the holdout is not opened",
        "- built on JRDB (D21): private, never published",
        "",
        ADDED_AFTER,
        "",
        "## Owner decisions used",
        "",
        *(f"- {key}: {value}" for key, value in DECISIONS.items()),
        "",
        "The frozen spec holds these decisions in its own earlier wording (its `decisions`);"
        " the spec is unchanged.",
        "",
        "## Data",
        "",
        _markdown_table([{"count": k, "n": v} for k, v in study.counts.items()]),
        "",
        "## D9 at level 2",
        "",
        f'D9, as signed: "{D9_SIGNED}".',
        "",
        f"The level-2 meta trains on the member rows on or before each fold's train end. D9's"
        f" {D9_MIN_SEASONS}-season minimum is met on {met} of the {len(window)} scored folds"
        + (
            ", so the frozen spec's note that only the first folds fall short understates it."
            if met == 0
            else "."
        )
        + " Level 1 is HKTrack's season walk-forward, refit once a season, not quarterly. The"
        " owner's answer (b) on #87 (2026-11-25) records the replication run as a documented"
        " failure to meet D9 at both levels: it supports no adoption claim, and the D9-compliant"
        " run comes with #91's and #98's members."
        " A coverage count of the store and the member file, no outcome:",
        "",
        _markdown_table(window) if window else "_Not computed._",
        "",
        "## Decision rules (frozen), read mechanically",
        "",
        "The rules as frozen:",
        "",
        *(f"- {key}: {rule}" for key, rule in spec["decision_rules"].items()),
        "",
        "Each applied as written (the frozen bound) and at D7 (meetings, 95% one-sided). The"
        " replication's bound is the meeting bootstrap, which is already D7's, so its two"
        " outcomes are one. The Wald outcomes are provisional (see above). Whatever these"
        " outcomes read, the run is a documented failure to meet D9 (the owner's answer (b) on"
        " #87, 2026-11-25; see D9 at level 2), so none of them supports an adoption claim.",
        "",
        _markdown_table(apply_rules(study, spec)),
        "",
        "## What this rules out",
        "",
        *rules_out(study, spec),
    ]
    for key, title, note in SECTIONS:
        rows = study.tables.get(key)
        lines += ["", f"## {title}", ""]
        lines += [note, ""] if note else []
        lines.append(_markdown_table(rows) if rows else "_Not computed: see the variants below._")
    ids = record_ids or {}
    variants = [
        {"E": v.hypothesis_id, "variant": v.name, "status": "failed" if v.error else "scored"}
        | ({"record": ids[v.name][:16]} if v.name in ids else {})
        | ({"error": v.error} if v.error else {})
        for v in study.variants
    ]
    lines += ["", "## Variants (one registry record each)", "", _markdown_table(variants), ""]
    return "\n".join(lines)


def _run_id(record: RunRecord) -> str:
    """The registry's run id: the sha256 of the record's canonical JSON."""
    canonical = json.dumps(
        dataclasses.asdict(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


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


def run_study(
    members_path: Path, out_dir: Path | None = None, *, allow_short_history: bool = False
) -> dict:
    """Run the pre-registered family on Japan's development data and write its private report.

    Raises `paths.DataDirInsideRepo` for an `out_dir` or a member file inside the repository,
    `FamilyAlreadyFrozen` when the family is frozen with another spec (another member file), and,
    before any fit, `models.stacking`'s `InvalidProvenance`, or its `IneligibleFold` when a scored
    fold's level-2 window is under D9's seasons and `allow_short_history` is False. The opt-in is
    the owner-approved replication (#87, answer (b), 2026-11-25); it leaves the frozen spec and
    the registry records as they are, and the report and the result say the run used it."""
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
    members = read_members(members_path)
    check_level2(frame, scored, members, allow_short_history=allow_short_history)
    frame["p_cal"] = baseline_oof(frame, baseline)
    meta, member_counts = member_frame(frame, members)
    del members
    study = compute_study(frame, meta, scored, spec)
    study.counts = {**counts, **member_counts, "scored_folds": len(scored), **study.counts}
    del frame, meta

    existing = registry.records(FAMILY_ID)
    reproduced = {_run_id(dataclasses.replace(r, git_sha="")): r for r in existing}
    record_ids, appended = {}, []
    for variant in study.variants:
        record = _record(variant, git_sha, builds, member_build)
        same = reproduced.get(_run_id(dataclasses.replace(record, git_sha="")))
        if same is None:  # a new run: appended
            record_ids[variant.name] = registry.append(record)
            appended.append(variant.name)
        else:  # the same run reproduced at another commit: not a new record
            record_ids[variant.name] = _run_id(same)
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "store": ", ".join(builds.values()),
        "members": member_build,
        "records": f"{len(appended)} appended by this run; {len(record_ids) - len(appended)} "
        "already in the registry (the same run reproduced, not appended again)",
        "level-2 guard": "`models.stacking.check_level2` passed before any fit; "
        + (
            "D9's 5-season minimum waived: this run used the opt-in --allow-short-history "
            "(allow_short_history=True), the owner-approved replication, a documented failure to "
            "meet D9 (#87, answer (b), 2026-11-25)"
            if allow_short_history
            else f"every scored fold meets D9's {D9_MIN_SEASONS}-season minimum at level 2"
        ),
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = write_report(
        out_dir / f"meta-{COUNTRY}-{spec_hash[:12]}.md",
        render_report(study, spec, about, record_ids),
        sources=MEMBER_SOURCES,
    )
    return {
        "family_id": FAMILY_ID,
        "spec_hash": spec_hash,
        "report_path": str(report),
        "registry_path": str(registry.path),
        "records_appended": len(appended),
        "appended_variants": appended,
        "records_already_present": len(record_ids) - len(appended),
        "record_ids": {name: run_id[:16] for name, run_id in record_ids.items()},
        "failed_variants": [v.name for v in study.variants if v.error],
        "allow_short_history": allow_short_history,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m horsetrack.countries.japan.studies.meta")
    parser.add_argument("--run", action="store_true", required=True, help="run the family")
    parser.add_argument(
        "--members", type=Path, required=True, help=f"the member file ({MEMBER_FILE})"
    )
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    parser.add_argument(
        "--allow-short-history",
        action="store_true",
        help="run although a scored fold's level-2 window is under D9's 5 seasons: the "
        "owner-approved replication (#87, answer (b)), a documented failure to meet D9",
    )
    args = parser.parse_args(argv)
    result = run_study(args.members, args.out, allow_short_history=args.allow_short_history)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
