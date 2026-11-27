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


def meta_variants(result: dict) -> list[str]:
    return [v for v in result["record_ids"] if v.startswith(("stage2:", "offset:"))]


def test_a_short_member_file_refuses_before_any_meta_fit(study: dict) -> None:
    with pytest.raises(st.IneligibleFold) as info:
        meta.run_study(study["short"], study["out"])
    assert study["fits"] == [], "a meta-model was fitted before the refusal"
    assert isinstance(info.value, ValueError)
    assert date(2024, 1, 1).isoformat() in str(info.value)  # the first scored fold is named


def test_the_opt_in_runs_the_short_replication_and_says_so(study: dict) -> None:
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
