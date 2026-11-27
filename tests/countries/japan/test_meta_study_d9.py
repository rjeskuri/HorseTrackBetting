"""Ticket #418, round 1 finding 5: the runnable #87 meta study refuses a level-2 window under D9.

Contract pinned here (`horsetrack.countries.japan.studies.meta`):

- `run_study(members_path, out_dir=None, *, allow_short_history=False) -> dict`. The study's
  level-2 path goes through `models.stacking`'s guarded check, with the member file's provenance
  (`fold_season`) kept: when any scored fold's joined level-2 training window is under D9's
  five seasons, the default run raises `models.stacking.IneligibleFold` before any meta-model fit
  (`fit_meta`); the error is not caught and recorded as a failed variant.
- `allow_short_history=True` is the owner-approved replication (#87, answer (b), 2026-11-25: a
  documented failure to meet D9): the run goes on and fits, and its result says so,
  `result["allow_short_history"] is True` (False on a default run).
- Round 2, finding E: the waiver holds only for the approved #87 artifact, the member file whose
  content digest (`file_digest`) the family's spec already frozen in the registry records: the
  family is frozen with exactly the spec this run builds for the file (`build_spec`, its
  `member_file.sha256`). For any other file, a fresh registry included, the five-season check
  stays in force and the run raises `IneligibleFold` even with the flag. `approve` below freezes
  the spec as the first #87 run did.
- Round 2, finding F: the eligibility check runs before anything is written to the registry: a
  refused input leaves the registry file byte for byte as it was, so a corrected file then runs in
  the same data directory (no `FamilyAlreadyFrozen`) and the registry holds only its run.
- A member file that meets D9 runs by default, without the opt-in.
- The CLI flag is `--allow-short-history`:
  `python -m horsetrack.countries.japan.studies.meta --run --members PATH [--out DIR]
  [--allow-short-history]`; without it a short member file fails (IneligibleFold or a non-zero
  exit).

Every value is synthetic, under a tmp HORSETRACK_DATA_DIR (`meta_store_fakes`); no test reads real
data, and the bootstrap runs a few replicates instead of scoring.yaml's 10,000.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from meta_store_fakes import build_store, member_rows, write_members, write_store

from horsetrack.core import paths
from horsetrack.core.eval.folds import load_country_scheme
from horsetrack.core.eval.registry import default_registry
from horsetrack.countries.japan.models import meta as models_meta
from horsetrack.countries.japan.models import stacking as st
from horsetrack.countries.japan.studies import meta

SHORT_START = date(2023, 1, 10)  # as #87's member file: under one season before 2024-Q1
LONG_START = date(2017, 1, 10)  # seven seasons before 2024-Q1: D9 met


@pytest.fixture
def study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    store = build_store()
    write_store(store)
    real_boot = meta.bootstrap_settings

    def few_replicates() -> dict:
        return {**real_boot(), "n_boot": 20}

    monkeypatch.setattr(meta, "bootstrap_settings", few_replicates)
    fits: list[int] = []
    real_fit = models_meta.fit_meta

    def spy_fit(*args, **kwargs):
        fits.append(1)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(meta, "fit_meta", spy_fit)
    monkeypatch.setattr(models_meta, "fit_meta", spy_fit)
    members = tmp_path / "members"
    return {
        "store": store,
        "fits": fits,
        "out": tmp_path / "out",
        "short": write_members(members / "short.parquet", member_rows(store, SHORT_START)),
        "long": write_members(members / "long.parquet", member_rows(store, LONG_START)),
    }


def approve(path: Path) -> None:
    """Freeze the family with the spec a run of `path` builds: the approved #87 artifact."""
    boot = {**meta.bootstrap_settings(), "seed": meta.BOOTSTRAP_SEED}
    spec = meta.build_spec(
        load_country_scheme("japan"), boot, meta.overlay_thresholds(), meta.file_digest(path)
    )
    default_registry("japan").freeze_family(meta.FAMILY_ID, spec)


def registry_bytes() -> bytes:
    path = default_registry("japan").path
    return path.read_bytes() if path.exists() else b""


def meta_variants(result: dict) -> list[str]:
    return [v for v in result["record_ids"] if v.startswith(("stage2:", "offset:"))]


def test_a_short_member_file_refuses_before_any_meta_fit(study: dict) -> None:
    with pytest.raises(st.IneligibleFold) as info:
        meta.run_study(study["short"], study["out"])
    assert study["fits"] == [], "a meta-model was fitted before the refusal"
    assert isinstance(info.value, ValueError)
    assert date(2024, 1, 1).isoformat() in str(info.value)  # the first scored fold is named


def test_the_opt_in_runs_the_short_replication_and_says_so(study: dict) -> None:
    approve(study["short"])
    result = meta.run_study(study["short"], study["out"], allow_short_history=True)
    assert result["allow_short_history"] is True
    assert study["fits"], "the opt-in run fits the meta-models"
    variants = meta_variants(result)
    assert variants
    assert set(variants) - set(result["failed_variants"]), "every meta variant failed"
    assert Path(result["report_path"]).is_file()


def test_a_member_file_that_meets_d9_runs_by_default(study: dict) -> None:
    result = meta.run_study(study["long"], study["out"])
    assert result["allow_short_history"] is False
    assert study["fits"]
    variants = meta_variants(result)
    assert variants and set(variants) - set(result["failed_variants"])


def test_the_cli_refuses_a_short_file_unless_the_flag_is_given(
    study: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    approve(study["short"])
    argv = ["--run", "--members", str(study["short"]), "--out", str(study["out"])]
    try:
        code = meta.main(argv)
    except st.IneligibleFold:
        code = None
    except SystemExit as exc:
        code = exc.code
        assert code not in (0, None)
    assert code != 0
    assert study["fits"] == []
    capsys.readouterr()

    assert meta.main([*argv, "--allow-short-history"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["allow_short_history"] is True
    assert study["fits"]


def test_the_waiver_holds_only_for_the_approved_member_file(study: dict) -> None:
    other = write_members(
        study["short"].parent / "other.parquet", member_rows(study["store"], SHORT_START, seed=11)
    )
    assert meta.file_digest(other) != meta.file_digest(study["short"])
    # a fresh registry approves no file
    with pytest.raises(st.IneligibleFold):
        meta.run_study(other, study["out"], allow_short_history=True)
    assert study["fits"] == []
    assert registry_bytes() == b""

    approve(study["short"])
    approved = registry_bytes()
    with pytest.raises(st.IneligibleFold):  # not FamilyAlreadyFrozen: the check comes first
        meta.run_study(other, study["out"], allow_short_history=True)
    assert study["fits"] == []
    assert registry_bytes() == approved

    result = meta.run_study(study["short"], study["out"], allow_short_history=True)
    assert result["allow_short_history"] is True
    assert study["fits"]


def test_a_refused_file_writes_nothing_so_a_corrected_file_runs_after_it(study: dict) -> None:
    with pytest.raises(st.IneligibleFold):
        meta.run_study(study["short"], study["out"])
    assert registry_bytes() == b"", "the refused run wrote to the registry"

    result = meta.run_study(study["long"], study["out"])  # the same data directory
    registry = default_registry("japan")
    assert registry.frozen_spec_hash(meta.FAMILY_ID) == result["spec_hash"]
    text = registry_bytes().decode("ascii")
    entries = [json.loads(line) for line in text.splitlines()]
    assert [e["kind"] for e in entries].count("freeze") == 1
    assert meta.file_digest(study["long"]) in text
    assert meta.file_digest(study["short"]) not in text
    assert result["records_appended"] > 0
    assert len(registry.records(meta.FAMILY_ID)) == result["records_appended"]
