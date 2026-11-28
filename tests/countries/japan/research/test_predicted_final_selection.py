"""Ticket #713: the leaderboard accepts the predicted-final selections and keeps them.

Every rule x checkpoint of the predicted-final study is a leaderboard row whose selection is
``predicted_final@T-10``, ``predicted_final@T-5`` or ``predicted_final@T-2``: the rule's conditions
read on the nowcast's projection of the final board from the board at T-k, so the bet is placeable
at T-k (it is not look-ahead) and it is not the plain board at the decision time either.

Contract pinned here:
- ``core.eval.leaderboard.LeaderboardRow`` accepts exactly those three new selection values beside
  the existing ones; near misses stay refused (``RowRefused``).
- A registry run whose ``params['leaderboard']`` block names one of them gives a measured row with
  that selection (``rows_from_registry``, the convention of ``docs/specs/leaderboard.md``).
- A candidate cell carrying one of them keeps it (``leaderboard_build.candidate_rows``; before this
  ticket an unknown selection read 'unknown').
- ``bets.shortlist``: a measured predicted-final row is ranked and listed like any other, and is
  not labelled look-ahead (only a ``look_ahead`` row is).

Synthetic rows only; every expected value is written out here.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from horsetrack.core.eval.leaderboard import LeaderboardRow, RowRefused, rows_from_registry
from horsetrack.core.eval.registry import FoldMetric, Registry, RunRecord

PREDICTED = ["predicted_final@T-10", "predicted_final@T-5", "predicted_final@T-2"]
NEAR_MISSES = [
    "predicted_final",
    "predicted-final@T-10",
    "Predicted_final@T-5",
    "predicted_final@T-2 ",
    "predicted_final@10",
]
GIT_SHA = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c"
BUILD = "f" * 64
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path / "data"))


def row(option_id: str, selection: str, **extra) -> LeaderboardRow:
    fields = {
        "option_id": option_id,
        "family_id": "japan-713-synthetic",
        "ticket": 713,
        "kind": "rule",
        "pool": "WIN",
        "segment": "all",
        "rule": "synthetic rule",
        "selection": selection,
        "sources": ("netkeiba",),
        "variants_searched": 3,
        "status": "scored",
    }
    fields.update(extra)
    return LeaderboardRow(**fields)


@pytest.mark.parametrize("selection", PREDICTED)
def test_a_row_may_carry_each_predicted_final_selection(selection):
    made = row("japan-713-synthetic/r1", selection, return_per_100=91.5, n=40, n_unit="races")
    assert made.selection == selection


@pytest.mark.parametrize("selection", NEAR_MISSES)
def test_near_misses_of_the_predicted_final_selection_are_still_refused(selection):
    with pytest.raises(RowRefused):
        row("japan-713-synthetic/r1", selection)


@pytest.mark.parametrize("selection", ["decision_time", "card_only", "look_ahead", "unknown"])
def test_the_existing_selections_are_still_accepted(selection):
    assert row("japan-713-synthetic/r1", selection).selection == selection


def _registry(tmp_path: Path) -> Registry:
    path = tmp_path / "reg" / "registry" / "japan.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return Registry(path, holdout_log=tmp_path / "reg" / "holdout_log.jsonl")


@pytest.mark.parametrize(("selection", "value"), zip(PREDICTED, [88.0, 104.5, 97.25], strict=True))
def test_a_registry_block_with_a_predicted_final_selection_gives_a_measured_row(
    tmp_path, selection, value
):
    reg = _registry(tmp_path)
    block = {
        "kind": "rule",
        "pool": "FORECAST",
        "rule": "n8:exacta:2-4/1",
        "selection": selection,
        "payout_rate": 0.775,
        "ci_level": 0.95,
        "n_unit": "races",
    }
    reg.append(
        RunRecord(
            hypothesis_id="P713",
            family_id="japan-713-synthetic",
            variant=f"n8:exacta:2-4/1@{selection.split('@')[1]}",
            ticket=713,
            git_sha=GIT_SHA,
            data_builds={"odds": BUILD},
            sources=("netkeiba",),
            fold_scheme="in-sample:development-days",
            components={},
            params={"leaderboard": block},
            metrics=(FoldMetric("development", "return_per_100", value, value - 9, value + 9, 30),),
            status="scored",
        )
    )
    rows = rows_from_registry(reg)
    assert len(rows) == 1
    got = rows[0]
    assert got.measured
    assert got.selection == selection
    assert got.return_per_100 == value
    assert got.rule == "n8:exacta:2-4/1"


def _cell(cell_id: str, selection: str, mean: float) -> dict:
    return {
        "study": "predicted-final",
        "family_id": "japan-713-synthetic",
        "cell_id": cell_id,
        "pool": "WIN",
        "class_group": None,
        "odds_band": "5-10",
        "payout_rate": 0.8,
        "n_bets": 30,
        "n_race_days": 12,
        "n_meetings": 6,
        "mean_return": mean,
        "se_mean_return": 0.1,
        "shrunk_edge": None,
        "mean_odds": 7.2,
        "sources": ["netkeiba"],
        "selection": selection,
    }


def test_candidate_cells_keep_their_predicted_final_selection(tmp_path):
    from horsetrack.countries.japan.research import leaderboard_build

    directory = tmp_path / "cells"
    directory.mkdir()
    cells = [
        _cell(f"n1:WIN:band:5-10@{s.split('@')[1]}", s, mean)
        for s, mean in zip(PREDICTED, [0.81, 0.92, 1.03], strict=True)
    ]
    (directory / "predicted-final.json").write_text(json.dumps(cells), encoding="utf-8")
    rows = leaderboard_build.candidate_rows(directory)
    by_segment = {r.segment: r for r in rows}
    assert len(rows) == 3
    for cell in cells:
        got = by_segment[cell["cell_id"]]
        assert got.selection == cell["selection"], "a valid predicted-final selection is kept"
        assert got.return_per_100 == pytest.approx(100 * cell["mean_return"])


def test_the_shortlist_lists_predicted_final_rows_and_does_not_call_them_look_ahead():
    from horsetrack.countries.japan.bets.log import CapCheck, Limits
    from horsetrack.countries.japan.bets.shortlist import shortlist, shortlist_text

    measured = {"se_per_100": 5.0, "n": 50, "n_unit": "races"}
    rows = [
        row("pf/at-t2", "predicted_final@T-2", return_per_100=120.0, **measured),
        row("pf/at-t10", "predicted_final@T-10", return_per_100=110.0, **measured),
        row("la/final", "look_ahead", return_per_100=100.0, **measured),
        row("dt/t2", "decision_time", return_per_100=90.0, **measured),
    ]
    ranked = shortlist(rows, top=10)
    ids = [r.row.option_id for r in ranked]
    assert set(ids) == {"pf/at-t2", "pf/at-t10", "la/final", "dt/t2"}
    assert ids.index("pf/at-t2") < ids.index("pf/at-t10") < ids.index("la/final")

    text = _ANSI.sub(
        "",
        shortlist_text(
            ranked,
            day=date(2026, 11, 29),
            check=CapCheck(fits=True, staked_yen=0, remaining_yen=30_000, off_flat=False),
            limits=Limits(day_cap_yen=30_000, flat_stake_yen=100),
            played=None,
        ),
    )
    heads = {line.split()[1]: line for line in text.splitlines() if re.match(r"\s*\d+\. ", line)}
    assert set(heads) == set(ids)
    assert "look-ahead" in heads["la/final"]
    for option_id in ("pf/at-t2", "pf/at-t10", "dt/t2"):
        assert "look-ahead" not in heads[option_id], option_id
