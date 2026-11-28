"""D4's calibrated market scored on the crowd-calibration folds and tail table (#275; #77, 08 §9.4).

#77 compares the normalised, power-β and Shin de-vigs of the final Win price out of fold, with
ΔLL against normalised and a tail table (observed over expected winners above decimal 51.0). This
module puts D4's calibrated market beside them on the same races: the natural cubic spline in
ln q of `models.v0.calibration` (`fit_win_calibration`, reused, not reimplemented), q the
normalised share of the race's priced runners, refitted on each walk-forward fold's training rows
and applied to the fold's test races.

The study (`run_study`; CLI `python -m horsetrack.countries.japan.studies.calibrated_market [--out
DIR]`) is a thin I/O layer over the pure `compute_study(frame, spec)`, in its own registry family.
It reuses #77's owner inputs (`config/calibration.yaml`: tail threshold and sign-off), development
rows, fold scheme, floor and bootstrap settings. Order: build the spec, freeze the family in the
run registry (`FamilyAlreadyFrozen` on another spec stops the run), only then read the
development rows (`calibration.read_development`: holdout days dropped, price era only; the
holdout is never opened), score, append one registry record per variant (failures included) and
write the report through `write_report`.

Memory: one subset's rows at a time; the spline is fitted fold by fold on flat arrays and
predicts one fold's test races (a quarter) at a time.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from horsetrack.core import paths
from horsetrack.core.eval.folds import Fold, FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha
from horsetrack.core.eval.registry import RunRecord, default_registry
from horsetrack.core.eval.reports import write_report
from horsetrack.core.eval.scoring import gain, per_race_log_loss
from horsetrack.countries.japan.models.v0 import calibration as d4
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START
from horsetrack.countries.japan.studies import calibration, source_family
from horsetrack.countries.japan.studies.calibration import Study, Variant

COUNTRY = calibration.COUNTRY
TICKET = 275
SOURCE = calibration.SOURCE  # the store's JRA results, from netkeiba; no JRDB, no JRA-VAN
FAMILY_ID = "japan-calibrated-market-final-win"
HYPOTHESIS = "E1"  # 08 §4 E1: the de-vig comparison and its tail, with D4's spline added
SPLINE = "D4 spline"  # the spline's label in the tables
N_KNOTS = 5  # fit_win_calibration's default, D4's live model
DEVIGS = ("normalised", "spline", "power", "shin")  # normalised first: the others' comparator
SUBSETS = calibration.SUBSETS
DECISIONS = {
    **{k: calibration.DECISIONS[k] for k in ("D4", "D6", "D7", "D9", "D10", "D11", "D12", "D14")},
    "D15": calibration.DECISIONS["D15"],
    "#77 (4)": "floor-bound races (a runner at odds 1.1 or under) are left out of fitting and "
    "scoring alike in the no-floor-bound subset (#77, 2026-11-25)",
    "#77 (5)": "the tail is a runner priced above decimal 51.0 (config/calibration.yaml)",
    "#275": "D4's spline is scored beside #77's de-vigs on the same folds and tail table, so the "
    "owner's choice of baseline de-vig (08 §9.4) compares all four; this family's spec is frozen "
    "before any development race is scored; the holdout is not opened",
}


# --------------------------------------------------------------------------- spec


def bootstrap_settings() -> dict:
    """D7's bootstrap from config/scoring.yaml, with this study's fixed seed."""
    return {**calibration.bootstrap_settings(), "seed": TICKET}


def build_spec(
    owner: dict, scheme: FoldScheme, bootstrap: dict, floor: float, *, source: str = SOURCE
) -> dict:
    """The family's pre-registered spec, frozen in the registry before anything is scored; a
    store copy `source` other than 'jra' is its own family (`source_family`, #404)."""
    return {
        "family_id": source_family(FAMILY_ID, source),
        "ticket": TICKET,
        "hypotheses": {
            HYPOTHESIS: "D4's calibrated market (the spline in ln q) against the normalised, "
            "power-β and Shin de-vigs of the final Win price: out-of-fold ΔLL against "
            "normalised and the tail's observed vs expected winners, with and without "
            "floor-bound races",
        },
        "target": "the race's winner (Win pool) at the final Win odds",
        "spline": {
            "model": "models.v0.calibration.fit_win_calibration: P(i wins) proportional to "
            "exp(f(ln q_i)), f a natural cubic spline, q_i = (1/O_i) / sum_j (1/O_j) over the "
            "race's priced runners; MLE; knots evenly spaced over the training rows' ln q "
            "quantiles",
            "dead_heats": "fitted on #77's development rows, so a race with a dead heat for first "
            "is kept and each co-winner counts as a winner (D4's live run_fit leaves those races "
            "out)",
            "n_knots": N_KNOTS,
            "knot_quantile_range": list(d4.QUANTILE_RANGE),
        },
        "devigs": {
            "normalised": "q_i = (1/O_i) / sum_j (1/O_j)",
            "power": "p_i proportional to (1/O_i)^beta, beta fitted on each fold's training rows "
            "(calibration.fit_power_beta)",
            "shin": "Shin (1993), z solved per race; clamped to 0 (normalised) without overround",
        },
        "comparator": "normalised",
        "delta_ll": "per scored race, normalised's winner log loss minus the variant's "
        "(core.eval.scoring.gain), out of fold; meeting-clustered paired bootstrap bounds",
        "walk_forward": "the spline and power-β are refitted on each fold's training rows and "
        "scored on its test races; normalised and Shin need no fit",
        "adoption": "none: the owner picks the baseline de-vig (08 §9.4)",
        "score": "winner log loss per race (core.eval.scoring.per_race_log_loss)",
        "rows": "priced starters of run races with a priced winner; development days only; a "
        "dead heat counts each co-winner",
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
        "sources": [source],
    }


# --------------------------------------------------------------------------- the pure study


def _walk_forward_spline(
    rows: pd.DataFrame, folds: list[Fold], n_knots: int, subset: str
) -> tuple[pd.Series, list, list]:
    """Out-of-fold spline probabilities on the folds' test rows (in fold order), each fold's
    spline fitted on its training rows only."""
    fit_rows = rows[["race_id", "race_date", "odds", "won"]].rename(columns={"odds": "win_odds"})
    parts, table = [], []
    for fold in folds:
        model = d4.fit_win_calibration(fit_rows.iloc[fold.train_idx], n_knots=n_knots)
        test = rows.iloc[fold.test_idx]
        q = calibration.devig_normalised(test["odds"], test["race_id"]).to_numpy()
        p = model.predict_many(q, test["race_id"].to_numpy())
        parts.append(pd.Series(p, index=test.index))
        table.append(
            {"subset": subset, "fold": fold.test_start.isoformat()}
            | {"train to": fold.train_end, "test to": fold.test_end}
            | {"train races": model.n_races, "test races": test["race_id"].nunique()}
            | {"fit_id": model.fit_id}
        )
    return pd.concat(parts), [], table


def _devig_variants(rows: pd.DataFrame, spec: dict, subset: str, study: Study) -> None:
    """Each de-vig's and the spline's out-of-fold winner log loss, ΔLL against normalised and the
    tail's observed vs expected winners on one subset, with meeting-clustered bounds."""
    folds = walk_forward(rows["race_date"], scheme=FoldScheme.from_mapping(spec["fold_scheme"]))
    scored = rows.iloc[np.concatenate([f.test_idx for f in folds] or [np.zeros(0, int)])]
    race = scored["race_id"].to_numpy()
    sizes = [len(f.test_idx) for f in folds]
    fold_of = pd.Series(np.repeat([f.test_start.isoformat() for f in folds], sizes), index=race)
    fold_of = fold_of.groupby(level=0).first()
    clusters = scored["meeting_id"].groupby(race).first()
    tail = (scored["odds"] > spec["tail"]["min_odds"]).to_numpy()
    won = scored["won"].to_numpy(dtype=float)
    n_knots = spec["spline"]["n_knots"]
    comparator: dict[str, pd.Series] = {}

    def score(name: str) -> tuple[list, dict]:
        if not folds:
            raise ValueError("the fold scheme gives no walk-forward fold on these dates")
        if name != "normalised" and "ll" not in comparator:
            raise RuntimeError("the normalised comparator failed, so there is no ΔLL")
        metrics, tables = [], {}
        if name == "spline":
            p, metrics, tables["spline_folds"] = _walk_forward_spline(rows, folds, n_knots, subset)
        elif name == "power":
            fit_rows = rows[["race_id", "odds", "won"]]
            p, metrics, tables["power_folds"] = calibration._walk_forward_power(
                fit_rows, folds, subset
            )
        else:
            p = calibration.PARAMETER_FREE[name](scored["odds"], scored["race_id"])
        no = pd.Series(False, index=scored.index)
        ll = per_race_log_loss(p, scored["won"], scored["race_id"], no).per_race
        for label, part in ll.groupby(fold_of.reindex(ll.index).to_numpy()):
            metrics.append(calibration._metric(label, "mean_log_loss", part.mean(), len(part)))
        metrics.append(calibration._metric("all_folds", "mean_log_loss", ll.mean(), len(ll)))
        label = SPLINE if name == "spline" else name
        row = {"subset": subset, "de-vig": label, "races": len(ll), "mean log loss": ll.mean()}
        if name == "normalised":
            comparator["ll"] = ll
        else:
            g = gain(ll, comparator["ll"])
            iv = calibration._bounds(g.per_race, clusters, spec)
            metrics.append(
                calibration._metric(
                    "all_folds", "delta_ll_vs_normalised", iv.mean, g.n_common, iv.lo, iv.hi
                )
            )
            row |= {"ΔLL vs normalised": iv.mean, "lower": iv.lo, "upper": iv.hi}
            row |= {"meetings": iv.n_clusters}
        gap = pd.Series(np.where(tail, won - p.to_numpy(), 0.0)).groupby(race).sum()
        iv = calibration._bounds(gap, clusters, spec)
        observed, expected = won[tail].sum(), p.to_numpy()[tail].sum()
        metrics += [
            calibration._metric("all_folds", "tail_observed_winners", observed, tail.sum()),
            calibration._metric("all_folds", "tail_expected_winners", expected, tail.sum()),
            calibration._metric(
                "all_folds", "tail_o_minus_e_per_race", iv.mean, len(gap), iv.lo, iv.hi
            ),
        ]
        tables["headline"] = [row]
        tables["tail"] = [
            {"subset": subset, "de-vig": label, "tail runners": int(tail.sum())}
            | {"observed winners": observed, "expected winners": expected}
            | {"O / E": observed / expected if expected > 0 else None}
            | {"(O - E) per race": iv.mean, "lower": iv.lo, "upper": iv.hi}
        ]
        return metrics, tables

    scheme = "walk_forward:" + json.dumps(spec["fold_scheme"], sort_keys=True)
    for name in DEVIGS:
        components = {"devig": name, "score": "winner_log_loss", "comparator": "normalised"}
        params = {"subset": subset, "tail_min_odds": spec["tail"]["min_odds"]}
        if name == "spline":
            components["model"] = "models.v0.calibration.fit_win_calibration"
            params |= {
                "n_knots": n_knots,
                "knot_quantile_range": spec["spline"]["knot_quantile_range"],
            }
        variant = Variant(HYPOTHESIS, f"{name}:{subset}", scheme, components, params)
        study.attempt(variant, lambda name=name: score(name))


def compute_study(frame: pd.DataFrame, spec: dict) -> Study:
    """Every variant and table of the frozen `spec` on `frame` (development_frame's shape).

    For each subset (every race; races without a floor-bound runner, left out of fitting and
    scoring alike): normalised, D4's spline and power-β (each refitted on the fold's training
    rows) and Shin, scored out of fold on the same races, each with its winner log loss, ΔLL
    against normalised and the tail's observed vs expected winners, with meeting-clustered
    bootstrap bounds. Pure: no I/O.
    """
    floor = calibration.flag_floor_bound(frame["odds"], floor=spec["floor"]["odds"])
    floor_race = floor.groupby(frame["race_id"].to_numpy()).transform("any").to_numpy()
    study = Study(counts={"floor_bound_runners": int(floor.sum())})
    study.counts["races_holding_a_floor_bound_runner"] = int(frame["race_id"][floor_race].nunique())
    columns = ["race_id", "race_date", "meeting_id", "odds", "won"]
    for subset in SUBSETS:
        rows = frame[columns] if subset == "all_races" else frame.loc[~floor_race, columns]
        rows = rows.reset_index(drop=True)
        _devig_variants(rows, spec, subset, study)
        del rows
    return study


# --------------------------------------------------------------------------- report and runner

SECTIONS = (
    (
        "headline",
        "Winner log loss and ΔLL against normalised (out of fold)",
        "ΔLL is normalised's log loss minus the variant's, per race, in nats (positive: better than"
        " normalised). lower and upper are the (1 - level) and level quantiles of the paired"
        " bootstrap over meetings (D7: one-sided).",
    ),
    (
        "tail",
        "Tail: runners priced above the tail threshold (out of fold)",
        "#77's layout. O / E is the tail's observed over expected winners (point only). (O - E)"
        " per race is the tail's observed minus expected winners per scored race, with the same"
        " meeting-clustered bounds; a dead heat counts each co-winner.",
    ),
    ("spline_folds", "Walk-forward folds: D4's spline fitted on each fold's training rows", ""),
    ("power_folds", "Walk-forward folds: the power beta of each fold's training rows", ""),
)


def render_report(study: Study, spec: dict, about: dict) -> str:
    """The full report: aggregate tables only, never a runner or race row."""
    lines = [
        f"# D4's calibrated market on the crowd-calibration folds ({COUNTRY}, #{TICKET})",
        "",
        *(f"- {key}: {value}" for key, value in about.items()),
        f"- owner sign-off of the pre-registered inputs: #{spec['owner_signoff']}",
        f"- spline: {spec['spline']['n_knots']} knots over the ln q quantiles "
        f"{spec['spline']['knot_quantile_range']}",
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


def _record(
    variant: Variant, git_sha: str, builds: dict[str, str], source: str = SOURCE
) -> RunRecord:
    return RunRecord(
        hypothesis_id=variant.hypothesis_id,
        family_id=source_family(FAMILY_ID, source),
        variant=variant.name,
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=builds,
        sources=(source,),
        fold_scheme=variant.fold_scheme,
        components=variant.components,
        params=variant.params,
        metrics=variant.metrics,
        status="failed" if variant.error else "scored",
    )


def run_study(out_dir: Path | None = None, *, source: str = SOURCE) -> dict:
    """Run the pre-registered study on Japan's development data and write its report. `source`
    names the store copy read (#404: 'jravan' the JRA-VAN copy), recorded in the spec, the
    records, the data builds and the report's `write_report` sources; a copy other than 'jra'
    runs as its own family `<FAMILY_ID>-<source>` (`source_family`).

    Raises `calibration.MissingOwnerDecision` before the store is read when
    config/calibration.yaml lacks an input, `paths.DataDirInsideRepo` for an `out_dir` inside the
    repository, and `FamilyAlreadyFrozen` when the family is frozen with another spec."""
    owner = calibration.owner_inputs()
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    spec = build_spec(
        owner,
        load_country_scheme(COUNTRY),
        bootstrap_settings(),
        calibration.win_floor(),
        source=source,
    )
    git_sha = _git_sha()
    registry = default_registry(COUNTRY)
    family_id = source_family(FAMILY_ID, source)
    spec_hash = registry.freeze_family(family_id, spec)  # before any development row is read

    builds = calibration.store_builds(source)
    runners, races = calibration.read_development(source=source)
    frame, counts = calibration.development_frame(runners, races)
    del runners, races
    study = compute_study(frame, spec)
    study.counts = {**counts, **study.counts}
    del frame

    existing = registry.records(family_id)
    records = [_record(v, git_sha, builds, source) for v in study.variants]
    appended = [registry.append(r) for r in records if r not in existing]
    about = {
        "family": f"`{family_id}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "store": ", ".join(builds.values()),
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = write_report(
        out_dir / f"calibrated-market-{COUNTRY}-{spec_hash[:12]}.md",
        render_report(study, spec, about),
        sources=(source,),
    )
    return {
        "family_id": family_id,
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
        prog="python -m horsetrack.countries.japan.studies.calibrated_market"
    )
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    args = parser.parse_args(argv)
    print(json.dumps(run_study(args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
