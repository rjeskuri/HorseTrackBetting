"""Every measured leaderboard family's selection timing is labelled from its spec (ticket #712).

Seven candidate-cell families carry no 'selection' key, so they read 'unknown' and the best-bets
filter (decision-time rows only) leaves them out. Their frozen specs say the selection reads the
final board, so each reads `look_ahead`:
- `japan-cross-pool-t2-{trial,v1,v2}` (#435, `docs/specs/cross-pool-prereg.md`: settlement and the
  candidate cells at the final prices, "no selection is a bet that could be placed at t"; the "t2"
  is the feature time, not the selection's board);
- `japan-n1-pool-scan-v{3,4,5,6}` (#429/#430: odds band, favourite strength and pool size are read
  from the final board).
The family match is exact: a later version gets its own label when its spec is read. Precedence: a
cell's own valid 'selection' wins; an own invalid one reads 'unknown' as before; without the key,
the family's label, then #503's spread-scan 'variant', then 'unknown'. Synthetic cells only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_MISSING = object()

LABELLED = [
    "japan-cross-pool-t2-trial",
    "japan-cross-pool-t2-v1",
    "japan-cross-pool-t2-v2",
    "japan-n1-pool-scan-v3",
    "japan-n1-pool-scan-v4",
    "japan-n1-pool-scan-v5",
    "japan-n1-pool-scan-v6",
]
# Near misses: not one of the seven exact ids, so no label until their own spec is read.
UNLABELLED = [
    "japan-some-new-study",
    "japan-cross-pool-t2-v3",
    "japan-cross-pool-t2-v0",
    "japan-cross-pool-t2",
    "japan-cross-pool-t2-v1-rerun",
    "japan-n1-pool-scan-v2",
    "japan-n1-pool-scan-v7",
    "japan-n1-pool-scan-v31",
    "japan-n1-pool-scan",
    "japan-pool-scan-v3",
    "japan-n8-spread-scan-quinella",
]


def make_cell(family: str, cell_id: str, selection=_MISSING, **extra) -> dict:
    cell = {
        "study": "synthetic",
        "family_id": family,
        "cell_id": cell_id,
        "pool": "WIN",
        "class_group": "maiden",
        "odds_band": "5-10",
        "payout_rate": 0.8,
        "n_bets": 100,
        "n_race_days": 40,
        "n_meetings": 20,
        "mean_return": 0.9,
        "se_mean_return": 0.05,
        "shrunk_edge": None,
        "mean_odds": 7.0,
        "sources": ["netkeiba"],
        **extra,
    }
    if selection is not _MISSING:
        cell["selection"] = selection
    return cell


def rows_of(tmp_path: Path, cells: list[dict]) -> list:
    from horsetrack.countries.japan.research import leaderboard_build

    directory = tmp_path / "cells"
    directory.mkdir()
    (directory / "cells.json").write_text(json.dumps(cells), encoding="utf-8")
    rows = leaderboard_build.candidate_rows(directory)
    assert len(rows) == len(cells)
    return rows


def selections_of(tmp_path: Path, cells: list[dict]) -> dict[str, str]:
    return {f"{r.family_id}/{r.segment}": r.selection for r in rows_of(tmp_path, cells)}


@pytest.mark.parametrize("family", LABELLED)
def test_a_labelled_family_reads_look_ahead_and_is_measured(tmp_path, family):
    cells = [
        make_cell(family, "class:maiden|band:5-10"),
        make_cell(family, "class:open|band:2-5", pool="PLACE", mean_return=1.2),
        make_cell(family, "class:open|band:20+", pool="TRIO", mean_return=0.4),
    ]
    rows = rows_of(tmp_path, cells)
    assert {r.selection for r in rows} == {"look_ahead"}
    assert {r.family_id for r in rows} == {family}
    # Measured, and the label does not change the measurement (90, 120, 40 per 100 staked).
    assert sorted(r.return_per_100 for r in rows) == pytest.approx([40.0, 90.0, 120.0])


@pytest.mark.parametrize("family", UNLABELLED)
def test_an_unlisted_family_without_a_selection_still_reads_unknown(tmp_path, family):
    got = selections_of(tmp_path, [make_cell(family, "a"), make_cell(family, "b", pool="PLACE")])
    assert set(got.values()) == {"unknown"}, got


@pytest.mark.parametrize("explicit", ["decision_time", "card_only", "look_ahead"])
@pytest.mark.parametrize("family", LABELLED)
def test_a_cell_that_names_a_timing_keeps_it_over_its_family_label(tmp_path, family, explicit):
    got = selections_of(tmp_path, [make_cell(family, "x", explicit)])
    assert list(got.values()) == [explicit]


@pytest.mark.parametrize("bogus", ["bogus", "value", "", None, 7, ["look_ahead"]])
@pytest.mark.parametrize("family", [LABELLED[1], LABELLED[4]])
def test_an_invalid_own_selection_still_reads_unknown(tmp_path, family, bogus):
    got = selections_of(tmp_path, [make_cell(family, "x", bogus)])
    assert list(got.values()) == ["unknown"]


def test_the_spread_scan_variant_mapping_is_unchanged(tmp_path):
    family = "japan-n8-spread-scan-quinella"
    cells = [
        make_cell(family, "t2-cell", variant="t2"),
        make_cell(family, "la-cell", variant="lookahead"),
        make_cell(family, "other-cell", variant="E2:debut-first-start"),
        make_cell(family, "none-cell"),
        make_cell(family, "own-wins", "card_only", variant="t2"),
    ]
    assert selections_of(tmp_path, cells) == {
        f"{family}/t2-cell": "decision_time",
        f"{family}/la-cell": "look_ahead",
        f"{family}/other-cell": "unknown",
        f"{family}/none-cell": "unknown",
        f"{family}/own-wins": "card_only",
    }


def test_labels_do_not_depend_on_the_mix_of_families_in_a_file(tmp_path):
    cells = [
        make_cell("japan-cross-pool-t2-v2", "c1"),
        make_cell("japan-n1-pool-scan-v5", "s1"),
        make_cell("japan-n1-pool-scan-v7", "u1"),
        make_cell("japan-n1-pool-scan-v6", "s2", "decision_time"),
        make_cell("japan-cross-pool-t2-v3", "u2"),
        make_cell("japan-n8-spread-scan-quinella", "t", variant="t2"),
    ]
    assert selections_of(tmp_path, cells) == {
        "japan-cross-pool-t2-v2/c1": "look_ahead",
        "japan-n1-pool-scan-v5/s1": "look_ahead",
        "japan-n1-pool-scan-v7/u1": "unknown",
        "japan-n1-pool-scan-v6/s2": "decision_time",
        "japan-cross-pool-t2-v3/u2": "unknown",
        "japan-n8-spread-scan-quinella/t": "decision_time",
    }
