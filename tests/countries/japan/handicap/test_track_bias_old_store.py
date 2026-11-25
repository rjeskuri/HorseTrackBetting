"""Ticket #253: `track_bias_report` on a store built before #222 (no `rail_setting` column).

- Ruling: the report runs on such a store (one built before #225 also lacks result_published_utc
  and result_unpublished_utc) and gives the numbers of the same store with the column all null.
  `track_bias._with_rail` already reads a missing column as null, so the report is made to read as
  much; no "rebuild the store" error.

Data is synthetic (day_variance_store), under tmp_path.
"""

from __future__ import annotations

import math
from datetime import date

import pyarrow.parquet as pq
import pytest
from day_variance_store import planted_history
from handicap_store import patch_holdout, weekend_days

from horsetrack.core import paths
from horsetrack.core.eval.folds import FoldScheme
from horsetrack.countries.japan.handicap import track_bias
from horsetrack.countries.japan.handicap.track_bias_report import track_bias_report

SOURCE = "jra"
WEEKLY = FoldScheme(unit="week", test_length=1, step=1, min_train=4, window="expanding",
                    embargo_days=1, anchor_month=1)  # fmt: skip
MID = ("style", "midpack")
OLD_COLUMNS = ("rail_setting", "result_published_utc", "result_unpublished_utc")


def planted_store(tmp_path, monkeypatch, sub: str, drop: tuple[str, ...] = ()):
    """The same planted history under tmp_path/sub, no rail printed (null), with the `drop`
    columns removed from races.parquet as a store built before them would be."""
    days = weekend_days(date(2025, 1, 4), date(2025, 3, 30))
    s, _, _ = planted_history(5, days, target=MID, sd=0.10, races=4, mode="pinned")
    for row in s.races:
        row["rail_setting"] = None
    s.write(tmp_path, monkeypatch, sub=sub)
    patch_holdout(monkeypatch, [], track_bias)
    path = paths.processed_dir(SOURCE) / "races.parquet"
    table = pq.read_table(path)
    pq.write_table(table.drop_columns([c for c in drop if c in table.column_names]), path)
    return days


@pytest.mark.parametrize(
    "dropped",
    [("rail_setting",), OLD_COLUMNS],
    ids=["no-rail-column", "no-rail-or-publication-columns"],
)
def test_track_bias_report_runs_on_a_store_without_a_rail_setting_column(
    tmp_path, monkeypatch, dropped
):
    days = planted_store(tmp_path, monkeypatch, "full")
    full = track_bias_report(days[0], days[-1], scheme=WEEKLY, n_boot=20)
    planted_store(tmp_path, monkeypatch, "old", dropped)
    names = pq.read_schema(paths.processed_dir(SOURCE) / "races.parquet").names
    assert "rail_setting" not in names

    old = track_bias_report(days[0], days[-1], scheme=WEEKLY, n_boot=20)

    assert old["races"] == full["races"] > 100
    for key in ("ll_prior", "ll_update"):
        assert math.isfinite(old[key]), key
        assert old[key] == pytest.approx(full[key], rel=1e-12), key
    assert old["gain_update_vs_prior"]["mean"] == pytest.approx(
        full["gain_update_vs_prior"]["mean"], rel=1e-12
    )
