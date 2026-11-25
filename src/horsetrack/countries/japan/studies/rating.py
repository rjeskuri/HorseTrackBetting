"""The C2 dynamic-rating member against the final Win price on development folds (#91; E38).

`models.rating.DynamicRating` is run once per drift variance of a pre-registered grid over every
starter of the JRA store's run races from 2010 (results only: no price is read before D10's
2014-06-07); a starter without a placing (中, 失) ranks tied last. Its forward-filtered state
(mu_before, sigma_before) becomes the member through `rating_member` at each scale of a second
grid. For each of D9's walk-forward folds the (drift, scale) pair is the one with the lowest
winner log loss on the inner folds: D9's quarters over the fold's training rows with min_train
cut to INNER_MIN_TRAIN quarters (the filter has nothing to refit, so an inner fold's loss is the
forward prediction of its quarter). The fold's test quarter is then scored with that pair.

Measured on the scored folds (every development fold, from 2020-Q1), against the store's final
Win price: the member's Pearson correlation with ln q (raw and within race); its winner log loss
against the D4 calibrated market (`studies.meta._delta`: ΔLL with D7's meeting bootstrap, the
Δ pseudo-R2) and its own pseudo-R2 against equal chances beside q's and the baseline's; and the
#87 stage-2 meta (`models.meta`, u = alpha * m + beta * ln q, unbounded, ridge 0) fitted on each
fold's training rows (member at the fold's pair) and predicting its test quarter, with each fold's
alpha (Wald and meeting-clustered intervals), and alpha pooled over the scored rows in-sample.

Order: freeze the spec in the run registry, read the store, score, append one registry record per
variant (failures included), write the report through `write_report` and the per-runner export
for the meld under the data directory: every priced starter at each selected pair (race_id,
horse_id, race_date, the pair, the folds that selected it, the fold the row is out of fold for,
mu_before, sigma_before and the member's logp; see `export_frame`). A rerun appends only records
the family does not hold: one equal to a held record but for its git commit is the same run
reproduced. The `meeting_clustered:` stage-2 record (D7's bound, which the frozen stage-2 weight
rule reads) was added after the first run; the frozen spec is unchanged. The JRA store is
netkeiba results: no JRDB, no JRA-VAN. Nothing is adopted (D16) and the holdout is never read.
CLI: `python -m horsetrack.countries.japan.studies.rating --run [--out DIR]`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from horsetrack.core import paths
from horsetrack.core.eval.folds import Fold, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha
from horsetrack.core.eval.registry import RunRecord, default_registry
from horsetrack.core.eval.reports import write_report
from horsetrack.countries.japan.models.meta import MetaSpec, fit_meta, oof_predictions
from horsetrack.countries.japan.models.rating import DynamicRating, rating_member
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table
from horsetrack.countries.japan.studies.calibration import (
    Study,
    Variant,
    _markdown_table,
    _metric,
    bootstrap_settings,
    store_builds,
)
from horsetrack.countries.japan.studies.meta import (
    CLUSTER,
    CLUSTERED,
    Scored,
    _clustered_metrics,
    _delta,
    _ll,
    _meta_metrics,
    _run_id,
    _try,
    baseline_oof,
    read_development,
    store_frame,
)

COUNTRY = "japan"
TICKET = 91
FAMILY_ID = "japan-c2-rating-final-win"
SOURCES = ("jra",)
BOOTSTRAP_SEED = 91  # reproducibility only; scoring.yaml signs no seed
# RATING_FROM and INNER_MIN_TRAIN are splits, an owner decision: part of the frozen spec,
# awaiting the owner's sign-off (docs/experiments/rating-japan.md), never a Ruling.
RATING_FROM = date(2010, 1, 1)  # results only: the store's first year, for the burn-in
FIXED = {"perf_var": 1.0, "prior_mean": 0.0, "prior_var": 1.0, "spell_var": 0.25, "spell_days": 90}
DRIFT_GRID = (0.0, 1e-4, 3e-4, 1e-3, 3e-3)  # ability variance per day, perf_var = 1
SCALE_GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
INNER_MIN_TRAIN = 4  # quarters of burn-in before the first inner fold
E38_CORRELATION = 0.94  # 08 §4 E38: a member distinct from the market correlates below this
MEMBER = "rating"
STAGE2 = MetaSpec([MEMBER], quadratic=False, segment_cols=[], form="stage2", ridge=0.0)


def build_spec(scheme, bootstrap: dict) -> dict:
    """The family's pre-registered spec, frozen in the registry before any outcome is read."""
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypotheses": {
            "E38": "a forward-filtered dynamic rating member correlates below "
            f"{E38_CORRELATION} with ln q_final and earns stage-2 weight beside ln q (prior: "
            "10% or less that alpha moves, 08 §4 E38)",
        },
        "decision_rules": {
            "distinct": f"the member's Pearson correlation with ln q over the scored runners is "
            f"below {E38_CORRELATION}: descriptive, no interval",
            "stage-2 weight": "alpha > 0 when the meeting-clustered one-sided 95% lower bound "
            "of alpha pooled over the scored rows (in-sample) lies above 0",
            "stage-2 gain": "the stage-2 meta's out-of-fold ΔLL against the calibrated market "
            "has its meeting-bootstrap lower bound above 0",
        },
        "rating": {
            "model": "models.rating.DynamicRating (Weng-Lin Bradley-Terry full-pair update)",
            "fixed": FIXED,
            "drift_grid": list(DRIFT_GRID),
            "scale_grid": list(SCALE_GRID),
            "input": f"every starter of the store's run races from {RATING_FROM.isoformat()} "
            "(no price read); a starter without a placing ranks tied last; race_utc is the "
            "race date at 00:00 JST plus race_no minutes",
            "member": "models.rating.rating_member over the store frame's priced starters",
        },
        "selection": f"per outer fold, the (drift, scale) with the lowest mean winner log loss "
        f"over the inner folds: D9's scheme on the fold's training rows, min_train "
        f"{INNER_MIN_TRAIN} quarters",
        "meta": "models.meta.fit_meta stage2 on [rating] and ln q, ridge 0, unbounded, "
        "fitted on each fold's training rows; pooled in-sample over the scored rows",
        "baseline": "D4 calibrated market (models.v0.calibration), in-fold",
        "folds": {"scheme": dataclasses.asdict(scheme), "scored": "every development fold"},
        "bootstrap": bootstrap,
        "price_era_start": PRICE_ERA_START.isoformat(),
        "adoption": "none (D16): no holdout is opened and nothing is adopted",
        "sources": list(SOURCES),
    }


def read_results() -> pd.DataFrame:
    """The rating's input: every starter of a run race from RATING_FROM, no price column."""
    runners = read_store_table(
        "runners",
        columns=["race_id", "horse_id", "finish_pos", "is_scratched"],
        start=RATING_FROM,
        end=date.max,
        touches_price=False,
    ).to_pandas()
    races = read_store_table(
        "races",
        columns=["race_id", "race_date", "race_no", "status"],
        start=RATING_FROM,
        end=date.max,
        touches_price=False,
    ).to_pandas(date_as_object=False)
    races = races.loc[races["status"].eq("run")]
    start = pd.to_datetime(races["race_date"]).dt.tz_localize("Asia/Tokyo")
    races = races.assign(
        race_utc=(start + pd.to_timedelta(races["race_no"], "min")).dt.tz_convert("UTC")
    )
    starters = runners.loc[~runners["is_scratched"].eq(True).to_numpy(dtype=bool, na_value=False)]
    out = starters.merge(races[["race_id", "race_utc"]], on="race_id", validate="many_to_one")
    last = out.groupby("race_id")["finish_pos"].transform("max").fillna(0) + 1
    return out.assign(finish_pos=out["finish_pos"].astype(float).fillna(last))


def member_grid(results: pd.DataFrame, frame: pd.DataFrame) -> tuple[dict, dict]:
    """Per (drift, scale): the member's log-probability on the frame's rows; per drift: the
    frame rows' (mu_before, sigma_before)."""
    keys = frame[["race_id", "horse_id"]]
    members, states = {}, {}
    for drift in DRIFT_GRID:
        out = DynamicRating(drift_var_per_day=drift, **FIXED).run(results)
        rows = keys.merge(out, on=["race_id", "horse_id"], how="left", validate="one_to_one")
        if rows["mu_before"].isna().any():
            raise ValueError("a priced starter has no rating row")
        rows.index = frame.index
        states[drift] = rows[["mu_before", "sigma_before"]]
        for scale in SCALE_GRID:
            members[(drift, scale)] = rating_member(rows, scale=scale)
    return members, states


def select(frame: pd.DataFrame, members: dict, folds: list[Fold], scheme) -> list[dict]:
    """Each fold's (drift, scale) by mean inner-fold winner log loss on its training rows."""
    losses = pd.DataFrame({k: _ll(np.exp(m), frame) for k, m in members.items()})
    inner_scheme = dataclasses.replace(scheme, min_train=INNER_MIN_TRAIN)
    out = []
    for fold in folds:
        train = frame.iloc[fold.train_idx]
        inner = walk_forward(train["race_date"], scheme=inner_scheme)
        races = pd.unique(np.concatenate([train["race_id"].to_numpy()[f.test_idx] for f in inner]))
        mean = losses.loc[races].mean()
        drift, scale = mean.idxmin()
        out.append(
            {"fold": fold.test_start.isoformat(), "inner folds": len(inner)}
            | {"inner races": len(races), "drift_var_per_day": drift, "scale": scale}
            | {"inner log loss": float(mean.min()), "worst grid log loss": float(mean.max())}
        )
    return out


def _corr(rows: pd.DataFrame) -> tuple[float, float]:
    m, q = rows[MEMBER], rows["ln_q"]
    centre = lambda s: s - s.groupby(rows["race_id"]).transform("mean")  # noqa: E731
    return float(np.corrcoef(m, q)[0, 1]), float(np.corrcoef(centre(m), centre(q))[0, 1])


def _correlation(scored: Scored) -> tuple[list, dict]:
    rows = scored.rows
    metrics, table = [], []
    for fold, part in [*rows.groupby("fold"), ("all_folds", rows)]:
        raw, within = _corr(part)
        n = part["race_id"].nunique()
        metrics += [
            _metric(fold, "pearson_ln_q", raw, n),
            _metric(fold, "pearson_ln_q_within_race", within, n),
        ]
        table.append({"fold": fold, "races": n, "pearson (ln p, ln q)": raw, "within race": within})
    return metrics, {"correlation": table}


def _alone(scored: Scored, spec: dict) -> tuple[list, dict]:
    rows = scored.rows
    metrics, tables = _delta(np.exp(rows[MEMBER]), scored, spec, f"member_alone:{MEMBER}")
    uniform = scored.uniform.sum()
    table = []
    for name, p in (
        ("member", np.exp(rows[MEMBER])),
        ("q", rows["q"]),
        ("calibrated market", rows["p_cal"]),
    ):
        ll = _ll(p, rows)
        r2 = 1.0 - ll.sum() / uniform
        metrics.append(_metric("all_folds", f"pseudo_r2_vs_equal_chances[{name}]", r2, len(ll)))
        table.append(
            {"forecast": name, "races": len(ll), "mean log loss": ll.mean(), "pseudo-R2": r2}
        )
    return metrics, tables | {"pseudo_r2": table}


def _stage2(frame: pd.DataFrame, members: dict, picks: list[dict], folds: list[Fold]):
    """The stage-2 meta's out-of-fold probabilities and each fold's fit, member at its pair."""
    base = frame[["race_id", "won", CLUSTER]].assign(ln_q_final=np.log(frame["q"]))
    p, fits = pd.Series(np.nan, index=frame.index), []

    def fit(train: pd.DataFrame):
        fits.append(fit_meta(train, STAGE2, cluster_col=CLUSTER))
        return fits[-1]

    for pick, fold in zip(picks, folds, strict=True):
        rows = base.assign(**{MEMBER: members[(pick["drift_var_per_day"], pick["scale"])]})
        oof = oof_predictions(rows, [fold], fit, lambda m, t: m.predict(t, np.exp(t["ln_q_final"])))
        p.iloc[fold.test_idx] = oof.iloc[fold.test_idx]
    return p, fits


def compute_study(
    frame: pd.DataFrame, members: dict, folds: list[Fold], spec: dict, scheme
) -> Study:
    """Every variant on the store frame (with the out-of-fold baseline `p_cal`). Pure: no I/O."""
    fold_scheme = "walk_forward:" + json.dumps(spec["folds"]["scheme"], sort_keys=True)
    study = Study(counts={})
    params = {"fixed": FIXED, "drift_grid": list(DRIFT_GRID), "scale_grid": list(SCALE_GRID)}
    comp = {"member": MEMBER, "selection": "inner_fold_log_loss"}
    picks = select(frame, members, folds, scheme)
    study.tables["selection"] = picks

    def selection() -> tuple[list, dict]:
        metrics = []
        for pick in picks:
            n = pick["inner races"]
            metrics += [
                _metric(pick["fold"], "selected_drift_var_per_day", pick["drift_var_per_day"], n),
                _metric(pick["fold"], "selected_scale", pick["scale"], n),
                _metric(pick["fold"], "inner_log_loss", pick["inner log loss"], n),
            ]
        return metrics, {}

    study.attempt(Variant("E38", "selection:inner_fold", fold_scheme, comp, params), selection)
    oof = out_of_fold(members, picks, folds, frame.index)
    rows = frame.assign(**{MEMBER: oof, "ln_q": np.log(frame["q"])})
    scored = Scored.of(rows, folds)
    study.counts["scored_races"] = scored.n_races
    base = comp | {"score": "winner_log_loss", "baseline": "calibrated_market"}
    study.attempt(
        Variant("E38", "correlation:ln_q", fold_scheme, comp, params), lambda: _correlation(scored)
    )
    study.attempt(
        Variant("E38", f"member_alone:{MEMBER}", fold_scheme, base, params),
        lambda: _alone(scored, spec),
    )

    outcome = _try(lambda: _stage2(frame, members, picks, folds))
    pooled_rows = scored.rows[["race_id", "won", CLUSTER, MEMBER]].assign(
        ln_q_final=scored.rows["ln_q"]
    )
    pooled = _try(lambda: fit_meta(pooled_rows, STAGE2, cluster_col=CLUSTER))
    meta_params = params | {"ridge": 0.0, "bounds": "none", "quadratic": False, "segments": []}
    components = base | {"form": "stage2", "market": "ln_q"}
    name = f"stage2:q:{MEMBER}"
    study.attempt(
        Variant("E38", name, fold_scheme, components, meta_params),
        lambda: _meta_metrics(outcome, pooled, folds, scored, spec, name),
    )
    # The frozen stage-2 weight rule reads D7's meeting bound, which _meta_metrics puts in the
    # report only: this record carries it (added after the first run, as #87's twins).
    clustered = meta_params | {
        "inference": "meeting-clustered sandwich (models.condlogit), no small-sample factor",
        "level": "95% one-sided per bound (D7)",
        "added": "after the first run: the frozen rule's own bound as a registry metric",
    }
    study.attempt(
        Variant(
            "E38", f"{CLUSTERED}:{name}", fold_scheme, components | {"se": CLUSTERED}, clustered
        ),
        lambda: _clustered_metrics(outcome, pooled, folds, spec),
    )
    return study


def out_of_fold(members: dict, picks: list[dict], folds: list[Fold], index: pd.Index) -> pd.Series:
    """The member on each fold's test rows at the fold's selected pair; NaN elsewhere."""
    oof = pd.Series(np.nan, index=index)
    for pick, fold in zip(picks, folds, strict=True):
        member = members[(pick["drift_var_per_day"], pick["scale"])]
        oof.iloc[fold.test_idx] = member.iloc[fold.test_idx]
    return oof


SECTIONS = (
    ("selection", "Drift and scale chosen per fold (inner-fold winner log loss)"),
    ("correlation", "Correlation of the member's ln p with ln q_final (scored runners)"),
    ("pseudo_r2", "Standalone pseudo-R2 against equal chances, beside the final price"),
    (
        "delta",
        "ΔLL against the D4 calibrated market (out of fold; meeting bootstrap, 95% one-sided)",
    ),
    ("coefficients", "Stage 2 per fold (unbounded; Wald 95% two-sided and meeting one-sided 95%)"),
    ("pooled", "Stage-2 alpha pooled over the scored rows (in-sample)"),
)


def render_report(study: Study, spec: dict, about: dict, record_ids: dict[str, str]) -> str:
    """The full report: aggregate tables only, never a runner or race row."""
    lines = [f"# C2 dynamic rating against the final Win price ({COUNTRY}, #{TICKET})", ""]
    lines += [*(f"- {k}: {v}" for k, v in about.items()), ""]
    lines += [
        "## Decision rules (frozen)",
        "",
        *(f"- {k}: {v}" for k, v in spec["decision_rules"].items()),
    ]
    lines += [
        "",
        "## Data",
        "",
        _markdown_table([{"count": k, "n": v} for k, v in study.counts.items()]),
    ]
    for key, title in SECTIONS:
        rows = study.tables.get(key)
        if key == "pooled" and rows:  # studies.meta labels the pooled member coefficient delta
            rows = [{("alpha" if k == "delta" else k): v for k, v in r.items()} for r in rows]
        lines += ["", f"## {title}", "", _markdown_table(rows) if rows else "_Not computed._"]
    variants = [
        {"E": v.hypothesis_id, "variant": v.name, "status": "failed" if v.error else "scored"}
        | {"record": record_ids.get(v.name, "")[:16]}
        | ({"error": v.error} if v.error else {})
        for v in study.variants
    ]
    return "\n".join(
        [*lines, "", "## Variants (one registry record each)", "", _markdown_table(variants), ""]
    )


def _record(variant: Variant, git_sha: str, builds: dict[str, str]) -> RunRecord:
    return RunRecord(
        hypothesis_id=variant.hypothesis_id,
        family_id=FAMILY_ID,
        variant=variant.name,
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=builds,
        sources=SOURCES,
        fold_scheme=variant.fold_scheme,
        components=variant.components,
        params=variant.params,
        metrics=variant.metrics,
        status="failed" if variant.error else "scored",
    )


def export_frame(
    frame: pd.DataFrame, members: dict, states: dict, picks: list[dict], folds: list[Fold]
) -> pd.DataFrame:
    """For the meld (#98): every priced starter at every (drift, scale) some fold selected, with
    the filtered state (mu_before, sigma_before: sigma_h(t)) and the member's log-probability
    `logp`. Every value is forward-filtered: it reads earlier races only, at a fixed pair.

    `selected_by` lists the folds whose inner selection chose the row's pair; that selection read
    those folds' training rows, so a row in a fold's training window is never out of fold for
    it. `oof_fold` is the one fold the row is out of fold for: the fold whose test quarter holds
    it, when that fold selected the row's pair (the study's scored value); null otherwise, as for
    every row before the first scored quarter. For fold k the meld trains on k's training rows
    at k's pair and tests on the rows with oof_fold k."""
    keys = frame[["race_id", "horse_id", "race_date"]].reset_index(drop=True)
    test_fold = np.full(len(frame), None, dtype=object)
    for pick, fold in zip(picks, folds, strict=True):
        test_fold[fold.test_idx] = pick["fold"]
    pairs: dict[tuple, list[str]] = {}
    for pick in picks:
        pairs.setdefault((pick["drift_var_per_day"], pick["scale"]), []).append(pick["fold"])
    parts = []
    for (drift, scale), chosen in pairs.items():
        oof = np.where(np.isin(test_fold, chosen), test_fold, None)
        parts.append(
            keys.assign(
                drift_var_per_day=drift,
                scale=scale,
                selected_by=",".join(chosen),
                oof_fold=pd.Series(oof, dtype="string"),
                mu_before=states[drift]["mu_before"].to_numpy(),
                sigma_before=states[drift]["sigma_before"].to_numpy(),
                logp=members[(drift, scale)].to_numpy(),
            )
        )
    return pd.concat(parts, ignore_index=True)


def run_study(out_dir: Path | None = None) -> dict:
    """Run the pre-registered family on Japan's development data; write its report and export."""
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    scheme = load_country_scheme(COUNTRY)
    spec = build_spec(scheme, {**bootstrap_settings(), "seed": BOOTSTRAP_SEED})
    git_sha = _git_sha()
    registry = default_registry(COUNTRY)
    spec_hash = registry.freeze_family(FAMILY_ID, spec)  # before any outcome is read

    builds = store_builds()
    runners, races = read_development()
    frame, counts = store_frame(runners, races)
    del runners, races
    folds = walk_forward(frame["race_date"], scheme=scheme)
    frame["p_cal"] = baseline_oof(frame, folds)
    results = read_results()
    counts |= {"rating_races": int(results["race_id"].nunique()), "rating_starters": len(results)}
    members, states = member_grid(results, frame)
    del results
    study = compute_study(frame, members, folds, spec, scheme)
    study.counts = {**counts, "scored_folds": len(folds), **study.counts}

    # A record equal to one in the registry but for its git commit is the same run reproduced.
    blank = {_run_id(dataclasses.replace(r, git_sha="")): r for r in registry.records(FAMILY_ID)}
    record_ids, appended = {}, []
    for variant in study.variants:
        record = _record(variant, git_sha, builds)
        same = blank.get(_run_id(dataclasses.replace(record, git_sha="")))
        if same is None:
            record_ids[variant.name] = registry.append(record)
            appended.append(variant.name)
        else:
            record_ids[variant.name] = _run_id(same)
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "store": ", ".join(builds.values()),
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = write_report(
        out_dir / f"rating-{COUNTRY}-{spec_hash[:12]}.md",
        render_report(study, spec, about, record_ids),
        sources=SOURCES,
    )
    export_path = paths.data_dir() / "features" / COUNTRY / f"rating-{spec_hash[:12]}.parquet"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_frame(frame, members, states, study.tables["selection"], folds).to_parquet(
        export_path, index=False
    )
    return {
        "family_id": FAMILY_ID,
        "spec_hash": spec_hash,
        "report_path": str(report),
        "export_path": str(export_path),
        "registry_path": str(registry.path),
        "records_appended": len(appended),
        "failed_variants": [v.name for v in study.variants if v.error],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m horsetrack.countries.japan.studies.rating")
    parser.add_argument("--run", action="store_true", required=True, help="run the family")
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    args = parser.parse_args(argv)
    print(json.dumps(run_study(args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
