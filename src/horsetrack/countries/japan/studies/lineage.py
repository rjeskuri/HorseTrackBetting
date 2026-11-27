"""E37's scoring (#95, epic #73; 08 §4 E37): the lineage prior's segment ΔLL on debut,
first-time-surface and first-time-distance races, standalone and at the final price.

`run_segments` flags each run from that horse's earlier runs only (strictly earlier race_utc):
`debut` when it has none; `first_surface` when it has some but none on this surface;
`first_distance` when it has some but none in this dist_band (the caller's band label, whose cut
points belong to the study's frozen spec). A run with a missing surface or band is never first
on it.

`segment_delta_ll` scores two forecasts' P(win) on the same rows by #69's per-race winner log
loss (each race renormalised over its rows, dead heats at 1/k): the per-race ΔLL is ln p_a(winner)
- ln p_b(winner), positive when a is better. A race is in a segment when any of its rows is
flagged; a segment's bounds are D7's meeting bootstrap of its per-race ΔLL
(`core.eval.bootstrap.paired_bootstrap` at spec["bootstrap"]). The same function scores both of
E37's comparisons: standalone (the ability member with the lineage prior against it without) and
at the final price (the stage-2 meta with the lineage member against the one without, or
against the market).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from horsetrack.core.eval.scoring import gain
from horsetrack.countries.japan.studies.calibration import _bounds
from horsetrack.countries.japan.studies.meta import _ll

SEGMENTS = ("debut", "first_surface", "first_distance")
OUT_COLUMNS = ["segment", "races", "meetings", "delta_ll", "lower", "upper"]

__all__ = ["OUT_COLUMNS", "SEGMENTS", "run_segments", "segment_delta_ll"]


def _none_earlier(times: pd.Series, keys: list[pd.Series]) -> np.ndarray:
    """Per row: True when no row with the same keys has an earlier time (False for a missing
    key)."""
    rank = times.groupby(keys, sort=False, dropna=True).rank(method="min")
    return rank.eq(1).to_numpy(dtype=bool)


def run_segments(runs: pd.DataFrame) -> pd.DataFrame:
    """One bool column per SEGMENTS entry, on runs' index; see the module docstring."""
    missing = [c for c in ("race_utc", "horse_id", "surface", "dist_band") if c not in runs.columns]
    if missing:
        raise ValueError(f"runs lacks column(s) {missing}")
    times = pd.Series(pd.to_datetime(runs["race_utc"], utc=True).to_numpy(), index=runs.index)
    horse = runs["horse_id"]
    debut = _none_earlier(times, [horse])
    seen = ~debut & horse.notna().to_numpy()
    return pd.DataFrame(
        {
            "debut": debut,
            "first_surface": seen & _none_earlier(times, [horse, runs["surface"]]),
            "first_distance": seen & _none_earlier(times, [horse, runs["dist_band"]]),
        },
        index=runs.index,
    )[list(SEGMENTS)]


def segment_delta_ll(
    p_a: pd.Series, p_b: pd.Series, rows: pd.DataFrame, spec: dict
) -> pd.DataFrame:
    """Per segment: races, meetings, the mean per-race winner ΔLL of p_a against p_b and its
    meeting-bootstrap bounds (spec["bootstrap"]); see the module docstring."""
    missing = [c for c in ("race_id", "meeting_id", "won", *SEGMENTS) if c not in rows.columns]
    if missing:
        raise ValueError(f"rows lacks column(s) {missing}")
    per_race = gain(_ll(p_a, rows), _ll(p_b, rows)).per_race
    clusters = rows.groupby(rows["race_id"].to_numpy())["meeting_id"].first()
    out = []
    for segment in SEGMENTS:
        flagged = rows[segment].fillna(False).to_numpy(dtype=bool)
        races = pd.unique(rows["race_id"].to_numpy()[flagged])
        part = per_race.loc[per_race.index.isin(races)]
        row = {"segment": segment, "races": len(part), "meetings": 0}
        row |= {"delta_ll": np.nan, "lower": np.nan, "upper": np.nan}
        if len(part):
            iv = _bounds(part, clusters, spec)
            row |= {"meetings": iv.n_clusters, "delta_ll": iv.mean, "lower": iv.lo, "upper": iv.hi}
        out.append(row)
    return pd.DataFrame(out, columns=OUT_COLUMNS)
