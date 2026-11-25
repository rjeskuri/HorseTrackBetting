"""The power beta with meeting-clustered confidence intervals, beside #77's Wald ones (#276).

#77's beta CIs are 95% Wald intervals, beta +/- 1.96 se, that treat races as independent; its
ΔLL CIs are clustered by meeting (D7). The races of one meeting share going, weather and the
day's crowd, so their scores can move together and the Wald interval can be too narrow. This
study re-reports #77's beta tables (by year and era, with and without floor-bound races; by E4
segment; by E3 entropy bin) with both intervals: the Wald one, unchanged, and the one from the
meeting-clustered sandwich se of `calibration.fit_power_beta(..., cluster=)`, beta +/- 1.96 se
cluster.

It adopts nothing. Whether beta's CIs adopt the meeting clusters is the owner's call, so #77's
family (`calibration.FAMILY_ID`), its spec and its Wald tables are untouched; this is a new
registry family with its own spec, frozen before any development row is read.

The study (`run_study`; CLI `python -m horsetrack.countries.japan.studies.beta_clustered --run
[--out DIR]`) is a thin I/O layer over the pure `compute_study(frame, spec)`. It reuses #77's
owner inputs (`config/calibration.yaml`: segments, entropy cut points, sign-off), development
rows, floor, fold scheme and bootstrap settings. The fold scheme and the bootstrap are carried
in the spec for provenance only: the fits are in-sample (no fold) and the clustered se is a
sandwich (no resampling). Order: build the spec, freeze the family in the run registry
(`FamilyAlreadyFrozen` on another spec stops the run), only then read the development rows
(`calibration.read_development`: holdout days dropped, price era only; the holdout is never
opened), score, append one registry record per variant (failures included) and write the report
through `write_report`.
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
from horsetrack.core.eval.folds import FoldScheme, load_country_scheme
from horsetrack.core.eval.holdout import _git_sha
from horsetrack.core.eval.registry import RunRecord, default_registry
from horsetrack.core.eval.reports import write_report
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START
from horsetrack.countries.japan.studies import calibration
from horsetrack.countries.japan.studies.calibration import Study, Variant

COUNTRY = calibration.COUNTRY
TICKET = 276
SOURCE = calibration.SOURCE  # the store's JRA results, from netkeiba; no JRDB, no JRA-VAN
FAMILY_ID = "japan-beta-clustered-final-win"
FIT = "conditional_logit_ln_inv_odds_meeting_sandwich"
IN_SAMPLE = calibration.IN_SAMPLE
SUBSETS = calibration.SUBSETS
Z = calibration.BETA_CI_Z
CLUSTER = "meeting_id"
DECISIONS = {
    **{k: calibration.DECISIONS[k] for k in ("D6", "D7", "D9", "D10", "D11", "D12", "D14", "D15")},
    "#77 (4)": "floor-bound races (a runner at odds 1.1 or under) are left out of fitting in the "
    "no-floor-bound subset (#77, 2026-11-25)",
    "#276": "the clustered interval is reported beside the Wald one and adopts nothing: whether "
    "beta's CIs adopt the meeting clusters is the owner's call; #77's family, spec and Wald "
    "tables are unchanged; this family's spec is frozen before any development race is scored; "
    "the holdout is not opened",
}


def build_spec(owner: dict, scheme: FoldScheme, bootstrap: dict, floor: float) -> dict:
    """The family's pre-registered spec, frozen in the registry before anything is scored."""
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypotheses": {
            "E1": "the power beta of the final Win price by year and era, with the meeting-"
            "clustered 95% interval beside #77's Wald one",
            "E3": "the power beta by bin of the race's normalised market entropy, both intervals",
            "E4": "the power beta in vs out of each pre-registered segment, both intervals",
        },
        "target": "the race's winner (Win pool) at the final Win odds",
        "beta_fit": "conditional logit on ln(1/O), MLE (calibration.fit_power_beta); races with a "
        "priced winner; a dead heat counts each co-winner",
        "wald_ci": "beta +/- 1.96 se, se = observed information^-1/2 (#77's, races independent)",
        "clustered_ci": "beta +/- 1.96 se_cluster, se_cluster = sqrt(G/(G-1) * sum over the G "
        "meeting clusters of the fitted races of (the sum of the cluster's race scores at "
        "beta-hat)^2) / observed information: the sandwich I^-1 (sum_c s_c s_c') I^-1 with the "
        "G/(G-1) small-sample factor; clusters are meetings (races.meeting_id: venue x day), "
        "as in D7; both intervals are 95% two-sided normal",
        "in_sample": "beta by year, era, E4 segment and E3 entropy bin: one fit per group on "
        "every development race of the subset, descriptive, no fold, no resampling",
        "comparison": "se cluster over the Wald se, per row: above 1 means the Wald interval is "
        "too narrow for the meeting-level dependence",
        "adoption": "none: whether beta's CIs adopt the meeting clusters is the owner's call",
        "rows": "priced starters of run races with a priced winner; development days only",
        "subsets": {
            "all_races": "every development race",
            "no_floor_bound_races": "races holding a floor-bound runner left out of fitting "
            "(beta by year and era)",
        },
        "segments": {name: calibration.SEGMENTS[name].definition for name in owner["segments"]},
        "entropy_cut_points": owner["entropy_cut_points"],
        "entropy_cut_rule": calibration.ENTROPY_CUT_RULE,
        "floor": {"odds": floor, "rule": "a priced runner at or under odds + 1e-9 (jra.yaml)"},
        "fold_scheme": dataclasses.asdict(scheme),
        "bootstrap": bootstrap,
        "price_era_start": PRICE_ERA_START.isoformat(),
        "decisions": DECISIONS,
        "owner_signoff": owner["owner_signoff"],
        "sources": [SOURCE],
    }


# --------------------------------------------------------------------------- the pure study


def _beta_table(table: pd.DataFrame, column: str, header: str, **label) -> tuple[list, list]:
    """Registry metrics (Wald and clustered, by `column` value) and report rows of a
    `segment_beta(..., cluster=)` table."""
    metrics, rows = [], []
    for value, beta, se, n, se_c, g in table[
        [column, "beta", "se", "n_races", "se_cluster", "n_clusters"]
    ].itertuples(index=False):
        for name, half in (("beta", se), ("beta_clustered", se_c)):
            metrics.append(
                calibration._metric(
                    "development",
                    f"{name}[{column}={value}]",
                    beta,
                    n,
                    beta - Z * half,
                    beta + Z * half,
                )
            )
        rows.append(
            {**label, header: value, "races": int(n), "meetings": int(g), "beta": beta, "se": se}
            | {"lower 95%": beta - Z * se, "upper 95%": beta + Z * se}
            | {"se cluster": se_c, "se ratio": se_c / se}
            | {"lower 95% cluster": beta - Z * se_c, "upper 95% cluster": beta + Z * se_c}
        )
    return metrics, rows


def _fit_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["race_id", "odds", "won", CLUSTER]]


def _year_and_era(frame: pd.DataFrame, subset: str) -> tuple[list, dict]:
    fit_rows = _fit_rows(frame).assign(year=frame["race_date"].dt.year)
    table = calibration.segment_beta(fit_rows, "year", cluster=CLUSTER)
    metrics, by_year = _beta_table(table, "year", "year", subset=subset)
    era = calibration.segment_beta(
        fit_rows.assign(era=f"{PRICE_ERA_START.isoformat()} onward"), "era", cluster=CLUSTER
    )
    era_metrics, by_era = _beta_table(era, "era", "era", subset=subset)
    return metrics + era_metrics, {"beta_by_year": by_year, "beta_by_era": by_era}


def _segment(frame: pd.DataFrame, races: pd.DataFrame, name: str) -> tuple[list, dict]:
    values = calibration.SEGMENTS[name].values(races)
    fit_rows = _fit_rows(frame).assign(**{name: frame["race_id"].map(values)})
    table = calibration._compared(calibration.segment_beta(fit_rows, name, cluster=CLUSTER), name)
    metrics, rows = _beta_table(table, name, "value", segment=name)
    left_out = [{"segment": name, "races in neither side": int(values.isna().sum())}]
    return metrics, {"segments": rows, "segment_left_out": left_out}


def _entropy_bins(frame: pd.DataFrame, cuts: list[float]) -> tuple[list, dict]:
    entropy = calibration.normalised_entropy(frame["odds"], frame["race_id"])
    edges = [0.0, *cuts, 1.0]
    labels = [f"{k + 1}: [{edges[k]:g}, {edges[k + 1]:g}{']' if k == 4 else ')'}" for k in range(5)]
    k = np.searchsorted(np.asarray(cuts), entropy.to_numpy(), side="right")
    values = pd.Series(np.asarray(labels, dtype=object)[k], index=entropy.index)
    values = values.where(entropy.notna())
    fit_rows = _fit_rows(frame).assign(bin=frame["race_id"].map(values))
    table = calibration._compared(calibration.segment_beta(fit_rows, "bin", cluster=CLUSTER), "bin")
    metrics, rows = _beta_table(table, "bin", "entropy bin")
    rows.append({"entropy bin": "left out (under two priced runners)"})
    rows[-1]["races"] = int(entropy.isna().sum())
    return metrics, {"entropy": rows}


def compute_study(frame: pd.DataFrame, spec: dict) -> Study:
    """Every variant and table of the frozen `spec` on `frame` (development_frame's shape).

    For each subset (every race; races without a floor-bound runner): the in-sample beta by year
    and era. On every development race: beta by E4 segment (in vs out) and by E3 entropy bin.
    Each row carries the Wald interval and the meeting-clustered one. Pure: no I/O."""
    floor = calibration.flag_floor_bound(frame["odds"], floor=spec["floor"]["odds"])
    floor_race = floor.groupby(frame["race_id"].to_numpy()).transform("any").to_numpy()
    study = Study(counts={"floor_bound_runners": int(floor.sum())})
    study.counts["races_holding_a_floor_bound_runner"] = int(frame["race_id"][floor_race].nunique())
    components = {"fit": FIT}
    for subset in SUBSETS:
        rows = frame if subset == "all_races" else frame.loc[~floor_race]
        rows = rows.reset_index(drop=True)
        variant = Variant(
            "E1",
            f"power_beta_clustered:by_year_and_era:{subset}",
            IN_SAMPLE,
            components,
            {"by": ["year", "era"], "subset": subset},
        )
        study.attempt(variant, lambda rows=rows, subset=subset: _year_and_era(rows, subset))
        del rows
    races = frame.drop_duplicates("race_id").set_index("race_id")
    for name, definition in spec["segments"].items():
        variant = Variant(
            "E4",
            f"segment_clustered:{name}",
            IN_SAMPLE,
            {**components, "segment": name},
            {"definition": definition},
        )
        study.attempt(variant, lambda name=name: _segment(frame, races, name))
    cuts = spec["entropy_cut_points"]
    variant = Variant(
        "E3",
        "entropy_bins_clustered",
        IN_SAMPLE,
        {**components, "entropy": "normalised"},
        {"cut_points": cuts},
    )
    study.attempt(variant, lambda: _entropy_bins(frame, cuts))
    return study


# --------------------------------------------------------------------------- report and runner

SECTIONS = (
    (
        "beta_by_year",
        "Beta by year (in-sample, development races): Wald and meeting-clustered 95% CI",
        "lower 95% and upper 95% are #77's Wald interval, beta +/- 1.96 se, races independent."
        " The columns with 'cluster' are beta +/- 1.96 se cluster, the sandwich se over meetings"
        " (meetings: the clusters of the fitted races). se ratio is se cluster over se: above 1"
        " the Wald interval is the narrower. Nothing is adopted: the owner decides.",
    ),
    (
        "beta_by_era",
        "Beta by era (in-sample)",
        "D10, signed: races before 2014-06-07 are excluded wherever the price enters.",
    ),
    ("segments", "E4 segments: beta in vs out (in-sample), both intervals", ""),
    (
        "segment_left_out",
        "E4 races in neither side of a segment",
        "A race whose attribute is missing is in neither side; for g1_day_undercard nor is the G1.",
    ),
    ("entropy", "E3 market-entropy bins (in-sample), both intervals", ""),
)


def render_report(study: Study, spec: dict, about: dict) -> str:
    """The full report: aggregate tables only, never a runner or race row."""
    lines = [
        f"# Power beta with meeting-clustered intervals ({COUNTRY}, #{TICKET})",
        "",
        *(f"- {key}: {value}" for key, value in about.items()),
        f"- owner sign-off of the pre-registered inputs: #{spec['owner_signoff']}",
        f"- floor: {spec['floor']['odds']:g}",
        "- development rows only: holdout days are never read and the holdout is not opened",
        f"- adoption: {spec['adoption']}",
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
        {"E": v.hypothesis_id, "variant": v.name, "status": "failed" if v.error else "scored"}
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
        out_dir / f"beta-clustered-{COUNTRY}-{spec_hash[:12]}.md",
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
    parser = argparse.ArgumentParser(
        prog="python -m horsetrack.countries.japan.studies.beta_clustered"
    )
    parser.add_argument("--run", action="store_true", required=True, help="run the study")
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    args = parser.parse_args(argv)
    print(json.dumps(run_study(args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
