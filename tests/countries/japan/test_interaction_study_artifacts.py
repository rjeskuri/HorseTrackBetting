"""Each launch of the C2 interaction study cites the artifacts it wrote (ticket #93, review round 5;
`studies.interactions.run_study`).

A retry that recovers a failed fold runs on the same store and the same frozen spec as the
launch it follows. Before round 5 it reused that launch's artifact stem: it appended records for
the recovered results and then overwrote the report, its JSON and the export the earlier
records cite.

Contract pinned here (`horsetrack.countries.japan.studies.interactions.run_study`, its return):

- A launch whose results differ from an earlier launch's on the same store and spec (a variant's
  record, a count, a table row or the export) names its report, JSON and export by another
  `artifact_stem`; the earlier launch's three files stay byte for byte as they were, and the new
  launch's export holds its own rows.
- Only the records that changed are appended, and they cite the new stem; a record equal to a
  held one but for git_sha and the artifacts it cites is not appended again, and the held
  records keep citing the stem of the launch that wrote them.
- A launch that reproduces every result (at another git revision) appends no record, names the
  same stem, and neither rewrites nor adds a file.
- A launch whose results cannot be digested (a value with no text form) after `compute_study`
  returns writes no artifact, records every planned variant (`planned()`) as failed, and the
  exception surfaces.
- A launch publishes its report, JSON and export before it appends a record citing them (review
  round 6): one stopped while publishing (a KeyboardInterrupt or an OSError from a write, after
  one or two of the three have landed) appends no record citing artifacts, records every
  planned variant failed, and the stop surfaces. Whatever launches follow, every record that
  cites artifacts cites a stem whose three files are on disk.
- A retry whose results equal the stopped launch's writes all three files at that stem, over
  whatever the stopped launch left there, and its report lists, for each of its variants, a run
  id the registry holds: the held record's when held, the appended one's otherwise.

Not pinned: the stem's layout beyond being a safe file stem, the report's and JSON's contents
(but for the run ids the report lists), the log lines, the order in which a launch writes its
three files (the stop comes from `interactions.write_report` and `interactions._publish_parquet`,
the module's writers, sharing one count). The store files are synthetic bytes (only their
hashes are read), `load_inputs` and `compute_study` are replaced (the compute is pinned
elsewhere), git revisions are fixed per launch, and every path is under `tmp_path`; an autouse
guard fails a test that changes the committed registry summary or a holdout log.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path

import pandas as pd
import pytest

from horsetrack.core.eval.registry import FoldMetric, Registry, RunRecord
from horsetrack.countries.japan.studies import interactions
from horsetrack.countries.japan.studies.calibration import Study, Variant

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED_SUMMARY = REPO_ROOT / "docs" / "experiments" / "registry"
REAL_COUNTRIES = REPO_ROOT / "src" / "horsetrack" / "countries"
DATA_DIR_VAR = "HORSETRACK_DATA_DIR"
SAFE_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

SHA = ("1a" * 20, "2b" * 20, "3c" * 20)  # the git revision of each launch
FOLDS = "walk_forward:synthetic"
COMPONENTS = {"study": "studies/interactions.py"}
HELD = ("E93", "stage2:q:market_only")  # scored alike at every launch
RECOVERED = ("E36", "stage2:q:trainer_x_track")  # failed at the first launch, scored on the retry
STEADY = ("E93", "tail:calibrated_market")  # scored alike at every launch
OLD = 1_000_000_000 * 10**9  # an mtime (ns, 2001) no write in this run can leave on a file


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """The data directory is under tmp_path; no test changes the committed summary or a log."""
    monkeypatch.setenv(DATA_DIR_VAR, str(tmp_path / "data"))

    def snapshot() -> dict[Path, bytes]:
        files = [*COMMITTED_SUMMARY.rglob("*"), *REAL_COUNTRIES.glob("*/holdout_log.jsonl")]
        return {p: p.read_bytes() for p in files if p.is_file()}

    before = snapshot()
    yield
    assert snapshot() == before, "a test changed the committed registry summary or a holdout log"


# --------------------------------------------------------------------------- the launches


def results(*, recovered: bool, change: str | None = None) -> tuple[Study, pd.DataFrame]:
    """One launch's study and export. At the first launch trainer_x_track's 2016-Q2 fold did not
    converge: its stage-2 variant failed, it has no tau row, and its export column is missing on
    that fold's rows. The retry (`recovered`) refits that fold at its own revision. `change` moves
    one part of the first launch's results: a record's metric, a count, a table row, the export."""
    held = 0.0125 + (0.0001 if change == "metric" else 0.0)
    variants = [
        Variant(
            *HELD,
            FOLDS,
            COMPONENTS,
            {"member": "market_only"},
            metrics=(FoldMetric("all_folds", "delta_ll", held, -0.0021, 0.0274, 480),),
        ),
        Variant(
            *RECOVERED,
            FOLDS,
            COMPONENTS,
            {"member": "trainer_x_track"},
            metrics=(FoldMetric("all_folds", "delta_ll", 0.0031, -0.0012, 0.0074, 480),),
        )
        if recovered
        else Variant(
            *RECOVERED,
            FOLDS,
            COMPONENTS,
            {"member": "trainer_x_track"},
            error="ValueError: trainer_x_track's fit on 2016-04-01 did not converge",
        ),
        Variant(
            *STEADY,
            FOLDS,
            COMPONENTS,
            {"band": "all"},
            metrics=(FoldMetric("all_folds", "return_per_100", 78.5, None, None, 480),),
        ),
    ]
    revisions = [
        {"member": "base", "revision": SHA[0], "n": 8},
        {"member": "trainer_x_track", "revision": SHA[0], "n": 7},
    ]
    if recovered:
        revisions.append({"member": "trainer_x_track", "revision": SHA[1], "n": 1})
    tables = {
        "fit_revisions": revisions,
        "cells": [{"cell": "trainer_x_track", "n_prior": 13 if change == "table" else 12}],
    }
    if recovered:
        tables["tau"] = [{"block": "trainer_x_track", "tau": 0.21, "lower": 0.08}]
    counts = {"level1_races": 961 if change == "count" else 960, "scored_races": 480}
    missing = float("nan")
    export = pd.DataFrame(
        {
            "race_id": ["2016a01", "2016a01", "2016b02", "2016b02"],
            "horse_id": ["h07", "h03", "h11", "h05"],
            "u_base": [0.4, -0.4, 0.25 if change == "export" else 0.2, -0.2],
            "u_trainer_x_track": [0.1, -0.1, 0.3, -0.3]
            if recovered
            else [0.1, -0.1] + [missing] * 2,
        }
    )
    return Study(counts=counts, tables=tables, variants=variants), export


def fake_store(data: Path) -> None:
    for name in ("runners", "races", "dividends"):
        path = data / "processed" / "jra" / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic {name}, one build".encode())


def launch(monkeypatch, out: Path, outcome, sha: str) -> dict:
    """`run_study` with its inputs replaced: `outcome` is what the compute returns (or raises)."""

    def compute(inputs, spec, scheme, **kwargs):
        return outcome

    monkeypatch.setattr(interactions, "load_inputs", lambda log=print: None)
    monkeypatch.setattr(interactions, "compute_study", compute)
    monkeypatch.setattr(interactions, "write_committed_summary", lambda registry, log=print: [])
    monkeypatch.setattr(interactions, "_git_sha", lambda: sha)
    return interactions.run_study(out_dir=out, log=lambda message: None)


def files_under(root: Path) -> dict[str, bytes]:
    """Every file the launches wrote (reports and exports; the registry is not under these)."""
    found = {}
    for top in (root / "reports", root / "data" / "features"):
        if top.exists():
            found |= {
                str(p.relative_to(root)): p.read_bytes() for p in top.rglob("*") if p.is_file()
            }
    return found


def artifacts(stem: str) -> set[str]:
    return {f"reports/{stem}.md", f"reports/{stem}.json", f"data/features/japan/{stem}.parquet"}


def cites(record: RunRecord, stem: str) -> bool:
    return stem in json.dumps(dataclasses.asdict(record), sort_keys=True)


def records(root: Path) -> list[RunRecord]:
    path = root / "data" / "registry" / "japan.jsonl"
    return Registry(path, holdout_log=root / "no-holdout.jsonl").records(interactions.FAMILY_ID)


def same_rows(path: Path, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(pd.read_parquet(path), expected, check_dtype=False)


# --------------------------------------------------------------------------- a retry


def test_a_retry_that_recovers_a_failed_variant_writes_beside_the_first_launch(
    tmp_path, monkeypatch
):
    assert {HELD, RECOVERED, STEADY} <= set(interactions.planned()), "the stub's variants"
    root, out = tmp_path.resolve(), tmp_path.resolve() / "reports"
    fake_store(root / "data")
    first_export = results(recovered=False)[1]
    first = launch(monkeypatch, out, results(recovered=False), SHA[0])
    stem1 = first["artifact_stem"]
    assert SAFE_STEM.fullmatch(stem1)
    assert Path(first["report_path"]).name == f"{stem1}.md"
    assert Path(first["export_path"]).name == f"{stem1}.parquet"
    kept = files_under(root)
    assert set(kept) == artifacts(stem1), "the report, its JSON and the export"
    held = records(root)
    assert [(r.hypothesis_id, r.variant, r.status) for r in held] == [
        (*HELD, "scored"),
        (*RECOVERED, "failed"),
        (*STEADY, "scored"),
    ]
    assert all(cites(r, stem1) for r in held)

    second_export = results(recovered=True)[1]
    second = launch(monkeypatch, out, results(recovered=True), SHA[1])
    stem2 = second["artifact_stem"]

    assert second["spec_hash"] == first["spec_hash"]
    assert stem2 != stem1, "a launch that changed a result reused the earlier launch's files"
    assert SAFE_STEM.fullmatch(stem2)
    assert Path(second["report_path"]).name == f"{stem2}.md"
    assert Path(second["export_path"]).name == f"{stem2}.parquet"
    now = files_under(root)
    assert {p: now.get(p) for p in kept} == kept, "the first launch's artifacts were rewritten"
    assert set(now) == artifacts(stem1) | artifacts(stem2)
    same_rows(root / "data" / "features" / "japan" / f"{stem1}.parquet", first_export)
    same_rows(root / "data" / "features" / "japan" / f"{stem2}.parquet", second_export)

    after = records(root)
    assert after[: len(held)] == held, "a held record changed"
    appended = after[len(held) :]
    assert [(r.hypothesis_id, r.variant, r.status) for r in appended] == [(*RECOVERED, "scored")]
    assert second["records_appended"] == 1
    assert cites(appended[0], stem2) and not cites(appended[0], stem1)
    assert all(cites(r, stem1) and not cites(r, stem2) for r in held)


def test_a_relaunch_with_the_same_results_appends_nothing_and_rewrites_no_file(
    tmp_path, monkeypatch
):
    root, out = tmp_path.resolve(), tmp_path.resolve() / "reports"
    fake_store(root / "data")
    launch(monkeypatch, out, results(recovered=False), SHA[0])
    second = launch(monkeypatch, out, results(recovered=True), SHA[1])
    before = files_under(root)
    held = records(root)
    for name in before:
        os.utime(root / name, ns=(OLD, OLD))

    third = launch(monkeypatch, out, results(recovered=True), SHA[2])

    assert third["records_appended"] == 0
    assert records(root) == held
    assert third["artifact_stem"] == second["artifact_stem"]
    now = files_under(root)
    assert now == before, "a relaunch with the same results rewrote or added a file"
    assert {name: (root / name).stat().st_mtime_ns for name in now} == dict.fromkeys(now, OLD)


@pytest.mark.parametrize("change", ["metric", "count", "table", "export"])
def test_any_changed_result_names_new_artifacts_and_keeps_the_earlier_ones(
    tmp_path, monkeypatch, change
):
    root, out = tmp_path.resolve(), tmp_path.resolve() / "reports"
    fake_store(root / "data")
    first = launch(monkeypatch, out, results(recovered=False), SHA[0])
    kept = files_under(root)
    moved_export = results(recovered=False, change=change)[1]

    second = launch(monkeypatch, out, results(recovered=False, change=change), SHA[0])

    stem1, stem2 = first["artifact_stem"], second["artifact_stem"]
    assert stem2 != stem1, f"a changed {change} reused the earlier launch's artifacts"
    now = files_under(root)
    assert {p: now.get(p) for p in kept} == kept, "the first launch's artifacts were rewritten"
    assert set(now) == artifacts(stem1) | artifacts(stem2)
    same_rows(root / "data" / "features" / "japan" / f"{stem2}.parquet", moved_export)


# --------------------------------------------------------------------------- a failed digest


class Undigestable(RuntimeError):
    """What a result value with no text form raises."""


class Opaque:
    """A table value that has no text form: every rendering of it raises."""

    def __str__(self) -> str:
        raise Undigestable("this value has no text form")

    __repr__ = __str__

    def __format__(self, spec: str) -> str:
        raise Undigestable("this value has no text form")


def test_a_launch_whose_results_cannot_be_digested_records_every_planned_variant_failed(
    tmp_path, monkeypatch
):
    root, out = tmp_path.resolve(), tmp_path.resolve() / "reports"
    fake_store(root / "data")
    study, export = results(recovered=True)
    study.tables["tau"] = [{"block": "trainer_x_track", "tau": Opaque()}]

    with pytest.raises(Undigestable):
        launch(monkeypatch, out, (study, export), SHA[0])

    held = records(root)
    planned = interactions.planned()
    assert sorted((r.hypothesis_id, r.variant) for r in held) == sorted(planned)
    assert {r.status for r in held} == {"failed"}
    assert not any(interactions.ARTIFACTS in r.data_builds for r in held)
    assert files_under(root) == {}, "an artifact was written for a launch that failed"


# --------------------------------------------------------------------------- a stopped publication

WRITERS = ("write_report", "_publish_parquet")  # the module's writers of the three artifacts
HEX = re.compile(r"[0-9a-f]{16,}")  # a run id as a report may show it (a prefix of 16 or more)
SCORED = sorted((name, "scored") for name in (HELD[1], RECOVERED[1], STEADY[1]))


def interrupt_publication(monkeypatch, *, landed: int, stop: BaseException):
    """Raise `stop` at the launch's artifact write after `landed` writes have landed, whichever
    artifact it writes first (the reports' and the export's writer share one count); returns
    the call that puts the real writers back."""
    real = {name: getattr(interactions, name) for name in WRITERS}
    count = 0

    def wrapping(writer):
        def write(*args, **kwargs):
            nonlocal count
            count += 1
            if count == landed + 1:
                raise stop
            return writer(*args, **kwargs)

        return write

    for name, writer in real.items():
        monkeypatch.setattr(interactions, name, wrapping(writer))

    def resume() -> None:
        for name, writer in real.items():
            monkeypatch.setattr(interactions, name, writer)

    return resume


def runs(root: Path) -> dict[str, dict]:
    """The family's runs as the registry's lines hold them: run id to record."""
    lines = (root / "data" / "registry" / "japan.jsonl").read_text("ascii").splitlines()
    entries = [json.loads(line) for line in lines]
    return {
        e["run_id"]: e["record"]
        for e in entries
        if e["kind"] == "run" and e["record"]["family_id"] == interactions.FAMILY_ID
    }


def dangling(root: Path) -> list[str]:
    """Each record that cites artifacts not all on disk, with what is missing."""
    on_disk = set(files_under(root))
    stems = {Path(name).stem for name in on_disk}
    out = []
    for r in records(root):
        if interactions.ARTIFACTS not in r.data_builds:
            continue
        cited = [s for s in stems if cites(r, s)]
        missing = sorted(artifacts(cited[0]) - on_disk) if cited else ["all three"]
        if missing:
            out.append(f"{r.variant} at {r.git_sha[:4]}: {missing}")
    return out


def listed(report: Path, root: Path) -> list[tuple[str, str]]:
    """The runs the report lists by run id, as (variant, status) of the registry's record; an id
    the registry does not hold is ('not in the registry', id). Hex a record holds (the spec's
    sha256, a git revision, a build) or the superseded spec's sha256 is not a run id."""
    held = runs(root)
    known = set(HEX.findall(json.dumps(list(held.values()))))
    known |= {interactions.SUPERSEDED_SPEC, *SHA}
    out = []
    for token in set(HEX.findall(report.read_text("utf-8"))) - known:
        match = [run_id for run_id in held if run_id.startswith(token)]
        record = held[match[0]] if match else None
        out.append(
            (record["variant"], record["status"]) if record else ("not in the registry", token)
        )
    return sorted(out)


def unlisted(report: Path, root: Path, stem: str) -> list[str]:
    """The records citing `stem` whose run id its report does not list."""
    shown = set(HEX.findall(report.read_text("utf-8")))
    return [
        f"{r['variant']} {run_id[:16]}"
        for run_id, r in runs(root).items()
        if stem in json.dumps(r, sort_keys=True)
        and not any(run_id.startswith(token) for token in shown)
    ]


@pytest.mark.parametrize("landed", [1, 2])
@pytest.mark.parametrize(
    "kind",
    [pytest.param(KeyboardInterrupt, id="stopped_by_hand"), pytest.param(OSError, id="disk")],
)
def test_a_stopped_publication_records_a_failed_run_and_a_recovering_retry_cites_whole_sets(
    tmp_path, monkeypatch, kind, landed
):
    root, out = tmp_path.resolve(), tmp_path.resolve() / "reports"
    fake_store(root / "data")
    resume = interrupt_publication(monkeypatch, landed=landed, stop=kind("stopped publishing"))
    with pytest.raises(kind):
        launch(monkeypatch, out, results(recovered=False), SHA[0])
    stopped = records(root)
    resume()

    retry = launch(monkeypatch, out, results(recovered=True), SHA[1])

    stem = retry["artifact_stem"]
    assert dangling(root) == [], "a record cites a report, JSON or export that is not on disk"
    assert [r.variant for r in stopped if interactions.ARTIFACTS in r.data_builds] == [], (
        "the stopped launch appended records citing artifacts it had not published"
    )
    assert sorted((r.hypothesis_id, r.variant) for r in stopped) == sorted(interactions.planned())
    assert {r.status for r in stopped} == {"failed"}
    assert artifacts(stem) <= set(files_under(root))
    appended = records(root)[len(stopped) :]
    assert sorted((r.variant, r.status) for r in appended) == SCORED
    assert retry["records_appended"] == 3
    assert all(cites(r, stem) for r in appended)
    report = Path(retry["report_path"])
    assert listed(report, root) == SCORED
    assert unlisted(report, root, stem) == []


@pytest.mark.parametrize("landed", [1, 2])
def test_a_retry_with_the_stopped_launchs_results_rewrites_its_set_and_lists_held_ids(
    tmp_path, monkeypatch, landed
):
    root, out = tmp_path.resolve(), tmp_path.resolve() / "reports"
    fake_store(root / "data")
    first = launch(monkeypatch, out, results(recovered=False), SHA[0])
    resume = interrupt_publication(
        monkeypatch, landed=landed, stop=KeyboardInterrupt("stopped by hand while publishing")
    )
    with pytest.raises(KeyboardInterrupt):
        launch(monkeypatch, out, results(recovered=True), SHA[1])
    resume()
    before = files_under(root)
    for name in before:
        os.utime(root / name, ns=(OLD, OLD))

    retry = launch(monkeypatch, out, results(recovered=True), SHA[2])

    stem1, stem = first["artifact_stem"], retry["artifact_stem"]
    assert stem != stem1
    now = files_under(root)
    assert set(now) == artifacts(stem1) | artifacts(stem)
    mtime = {name: (root / name).stat().st_mtime_ns for name in now}
    assert sorted(name for name in artifacts(stem) if mtime[name] == OLD) == [], (
        "the retry kept a file the stopped launch left at its stem"
    )
    assert {name: (now[name], mtime[name]) for name in artifacts(stem1)} == {
        name: (before[name], OLD) for name in artifacts(stem1)
    }, "the first launch's artifacts were rewritten"
    assert dangling(root) == [], "a record cites a report, JSON or export that is not on disk"
    assert retry["records_appended"] == 1, "only the recovered variant is new to the registry"
    report = Path(retry["report_path"])
    assert listed(report, root) == SCORED, "the report lists a run id the registry does not hold"
    assert unlisted(report, root, stem) == []
