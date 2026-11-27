"""E37: the lineage prior's segment ΔLL on debut, first-time-surface and first-time-distance
races, standalone and at the final price (ticket #95, epic #73; 08 §4 E37).

Contract pinned here (`horsetrack.countries.japan.studies.lineage`):
- `SEGMENTS == ("debut", "first_surface", "first_distance")`.
- `run_segments(runs) -> pd.DataFrame`: one row per row of runs, on runs' index, with one bool
  column per segment. runs: race_id, race_utc, horse_id, surface and dist_band (other columns
  ignored, rows in any order). A run is `debut` when the horse has no run in runs with an earlier
  race_utc; `first_surface` when it has an earlier run but none on this surface; and
  `first_distance` when it has an earlier run but none in this dist_band (the caller's band
  label, not the exact distance). The last two may both hold; neither holds on a debut. A run's
  flags read only that horse's earlier runs: a later run, or another horse's, never moves them.
- `segment_delta_ll(p_a, p_b, rows, spec) -> pd.DataFrame`: the paired per-race winner ΔLL of
  p_a against p_b, ln p_a(winner) - ln p_b(winner) (positive: a better), over each segment's
  races, a race being in a segment when any of its rows is flagged. rows: race_id, meeting_id,
  won and the SEGMENTS columns; p_a and p_b are P(win) on rows' index. One row per segment
  (column `segment`) with races, meetings (the meetings among its races), delta_ll (the mean over
  its races) and lower and upper, `core.eval.bootstrap.paired_bootstrap` of the per-race ΔLL over
  meetings at spec["bootstrap"]'s n_boot, level and seed (the only part of spec read). One
  function scores both comparisons: standalone (the ability member with the lineage prior
  against it without) and at the final price (the stage-2 meta with the lineage member against
  the one without, or against the market).

Not pinned: extra rows beyond the three segments (an all-races row, say), the row order, how
the folds, members and metas are built (#87's and #92's machinery), the distance bands' cut
points (the frozen spec), the registry records, report and leaderboard rows, which read the local
stores. Expected values are by hand or by first principles, the bounds by the library's
`paired_bootstrap` (not the code under test). Every RNG is seeded.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.countries.japan.studies.lineage import SEGMENTS, run_segments, segment_delta_ll

# isort: split
from lineage_synth import FIRST_POST, LONG, MIDDLE, MILE, SPRINT

BOOT = {"cluster_unit": "meeting", "sided": "one-sided", "level": 0.95, "n_boot": 400, "seed": 95}
SPEC = {"bootstrap": BOOT}
OUT_COLUMNS = {"segment", "races", "meetings", "delta_ll", "lower", "upper"}
DAY, HOUR = pd.Timedelta(days=1), pd.Timedelta(hours=1)

# ---------------------------------------------------------------------------------------------
# Segments read only the horse's earlier runs
# ---------------------------------------------------------------------------------------------

# (horse, day, surface, band) -> (debut, first_surface, first_distance), worked by hand
HISTORY = [
    ("hA", 0, "turf", MILE, (True, False, False)),
    ("hA", 1, "turf", MILE, (False, False, False)),
    ("hA", 2, "dirt", MILE, (False, True, False)),
    ("hA", 3, "dirt", SPRINT, (False, False, True)),
    ("hA", 4, "turf", LONG, (False, False, True)),
    ("hA", 5, "dirt", LONG, (False, False, False)),
    ("hB", 2, "dirt", SPRINT, (True, False, False)),
    ("hB", 3, "turf", MILE, (False, True, True)),
    ("hB", 5, "turf", SPRINT, (False, False, False)),
    ("hC", 1, "turf", MILE, (True, False, False)),
    ("hC", 4, "turf", MILE, (False, False, False)),
]
HOUR_OF = {"hA": 0, "hB": 1, "hC": 2, "hD": 3}


def history_frame(entries, start: int = 100) -> tuple[pd.DataFrame, dict]:
    """(runs, flags by index): each entry its own race, horses an hour apart on a day."""
    rows, flags = [], {}
    for i, (horse, day, surface, band, want) in enumerate(entries):
        post = FIRST_POST + day * DAY + HOUR_OF[horse] * HOUR
        rows.append((f"r{i:02d}", post, horse, surface, band, i + 1))
        flags[start + i] = want
    runs = pd.DataFrame(
        rows, columns=["race_id", "race_utc", "horse_id", "surface", "dist_band", "finish_pos"]
    )
    runs.index = pd.RangeIndex(start, start + len(runs))
    return runs, flags


def segment_flags(runs: pd.DataFrame) -> dict:
    """run_segments(runs) checked for its shape, as {index: (debut, first_surface,
    first_distance)}."""
    assert SEGMENTS == ("debut", "first_surface", "first_distance")
    got = run_segments(runs)
    assert isinstance(got, pd.DataFrame)
    assert set(SEGMENTS) <= set(got.columns), sorted(got.columns)
    assert len(got) == len(runs)
    assert set(got.index) == set(runs.index)
    out = {}
    for idx, row in got[list(SEGMENTS)].iterrows():
        out[idx] = tuple(bool(row[s]) for s in SEGMENTS)
    return out


def test_segments_read_only_earlier_runs():
    """By hand over three horses' histories: the debut, a first run on dirt, first runs in a new
    band, a run new in both, and runs new in neither (a surface and a band each seen before, in
    different runs). Rows shuffled, the index kept. Later runs (a first jump race, a first middle
    distance) leave every earlier row's flags as they were; another horse's runs never move a
    horse's flags; one earlier run of hB (a turf sprint the day before its 'debut') moves exactly
    hB's flags."""
    runs, want = history_frame(HISTORY)
    shuffled = runs.sample(frac=1.0, random_state=951)
    assert segment_flags(shuffled) == want

    later = HISTORY + [
        ("hA", 9, "jump", LONG, (False, True, False)),
        ("hB", 9, "dirt", MIDDLE, (False, False, True)),
        ("hD", 0, "turf", MILE, (True, False, False)),
        ("hD", 9, "turf", SPRINT, (False, False, True)),
    ]
    grown, grown_want = history_frame(later)
    assert segment_flags(grown.sample(frac=1.0, random_state=952)) == grown_want
    assert {k: v for k, v in grown_want.items() if k in want} == want

    earlier = HISTORY + [("hB", 1, "turf", SPRINT, (True, False, False))]
    moved, _ = history_frame(earlier)
    got = segment_flags(moved)
    hb = {
        ("hB", 2): (False, True, False),  # dirt is new; sprint was run the day before
        ("hB", 3): (False, False, True),  # turf was run; mile is new
        ("hB", 5): (False, False, False),
        ("hB", 1): (True, False, False),
    }
    for idx, (horse, day, *_rest) in zip(moved.index, earlier, strict=True):
        if horse == "hB":
            assert got[idx] == hb[(horse, day)], (horse, day)
        else:
            assert got[idx] == want[idx], (horse, day)


# ---------------------------------------------------------------------------------------------
# DoD: segment ΔLL standalone and at the final price, with CIs
# ---------------------------------------------------------------------------------------------


def softmax_within(v: np.ndarray, codes: np.ndarray) -> np.ndarray:
    top = pd.Series(v).groupby(codes).transform("max").to_numpy()
    e = np.exp(v - top)
    return e / pd.Series(e).groupby(codes).transform("sum").to_numpy()


def flag_rows(rng, kind: str, n: int) -> dict[str, np.ndarray]:
    """Per-runner segment flags of a race of n runners: 'debut' all debut; 'surface' and
    'distance' one to three runners flagged; 'both' some first-surface and others
    first-distance runners; 'plain' none."""
    out = {s: np.zeros(n, dtype=bool) for s in SEGMENTS}
    picks = rng.permutation(n)
    k = int(rng.integers(1, 4))
    if kind == "debut":
        out["debut"][:] = True
    elif kind == "surface":
        out["first_surface"][picks[:k]] = True
    elif kind == "distance":
        out["first_distance"][picks[:k]] = True
    elif kind == "both":
        out["first_surface"][picks[:k]] = True
        out["first_distance"][picks[k : k + 2]] = True
    return out


def race_frame(seed: int, kinds, *, meetings: int, races_per_meeting: int) -> pd.DataFrame:
    """race_id, meeting_id, codes (race code), kind and the SEGMENTS flags, 6 to 10 runners a
    race, the race kinds cycling through `kinds`."""
    rng = np.random.default_rng(seed)
    parts, code = [], 0
    for m in range(meetings):
        meeting = f"2024{m // 2:03d}v{m % 2}"
        for r in range(races_per_meeting):
            n = int(rng.integers(6, 11))
            kind = kinds[code % len(kinds)]
            part = pd.DataFrame(flag_rows(rng, kind, n))
            part.insert(0, "race_id", f"{meeting}r{r:02d}")
            part.insert(1, "meeting_id", meeting)
            part.insert(2, "code", code)
            part.insert(3, "kind", kind)
            parts.append(part)
            code += 1
    return pd.concat(parts, ignore_index=True)


def draw_winners(rng, p: np.ndarray, codes: np.ndarray) -> np.ndarray:
    g = pd.Series(np.log(p) + rng.gumbel(size=len(p)))
    won = np.zeros(len(p), dtype=bool)
    won[g.groupby(codes).idxmax().to_numpy()] = True
    return won


def per_race_delta(p_a, p_b, rows: pd.DataFrame) -> pd.Series:
    """ln p_a(winner) - ln p_b(winner) by race id, one winner per race."""
    w = rows.loc[rows["won"].to_numpy(dtype=bool)]
    d = np.log(p_a.loc[w.index].to_numpy()) - np.log(p_b.loc[w.index].to_numpy())
    return pd.Series(d, index=w["race_id"].to_numpy())


def expected_rows(p_a, p_b, rows: pd.DataFrame) -> dict:
    d = per_race_delta(p_a, p_b, rows)
    clusters = rows.groupby("race_id")["meeting_id"].first()
    out = {}
    for s in SEGMENTS:
        races = pd.Index(rows.loc[rows[s].to_numpy(dtype=bool), "race_id"].unique())
        part = d.loc[d.index.isin(races)]
        iv = paired_bootstrap(
            part, clusters, n_boot=BOOT["n_boot"], level=BOOT["level"], seed=BOOT["seed"]
        )
        out[s] = {
            "races": len(races),
            "meetings": int(clusters.loc[races].nunique()),
            "delta_ll": float(part.mean()),
            "lower": iv.lo,
            "upper": iv.hi,
        }
    return out


def checked_table(p_a, p_b, rows: pd.DataFrame) -> pd.DataFrame:
    got = segment_delta_ll(p_a, p_b, rows, SPEC)
    assert isinstance(got, pd.DataFrame)
    assert OUT_COLUMNS <= set(got.columns), sorted(got.columns)
    assert set(SEGMENTS) <= set(got["segment"])
    assert got["segment"].is_unique
    return got.set_index("segment")


def assert_segment(got: pd.DataFrame, want: dict) -> None:
    for s, w in want.items():
        row = got.loc[s]
        assert int(row["races"]) == w["races"], s
        assert int(row["meetings"]) == w["meetings"], s
        assert float(row["delta_ll"]) == pytest.approx(w["delta_ll"], abs=1e-12), s
        assert float(row["lower"]) == pytest.approx(w["lower"], abs=1e-9), s
        assert float(row["upper"]) == pytest.approx(w["upper"], abs=1e-9), s
        assert float(row["lower"]) <= float(row["delta_ll"]) <= float(row["upper"]), s


def test_segment_delta_ll_matches_first_principles():
    """Forty meetings of six races of every kind (debut races, races with a few first-surface or
    first-distance runners, with both, and plain), two unrelated forecasts, rows shuffled onto a
    non-default index: each segment's race count, meeting count and mean winner ΔLL are the
    direct computation over the races with a flagged runner, and its bounds are the library's
    meeting bootstrap of those races at the spec's settings. Swapping the two forecasts flips the
    sign."""
    rng = np.random.default_rng(9551)
    kinds = ("debut", "surface", "distance", "both", "plain")
    frame = race_frame(9551, kinds, meetings=40, races_per_meeting=6)
    codes = frame["code"].to_numpy()
    p_a = softmax_within(rng.normal(0.0, 1.0, len(frame)), codes)
    p_b = softmax_within(rng.normal(0.0, 1.0, len(frame)), codes)
    frame["won"] = draw_winners(rng, p_a, codes)
    perm = rng.permutation(len(frame))
    rows = frame.iloc[perm].copy()
    rows.index = pd.Index([f"row{i:05d}" for i in perm])
    pa = pd.Series(p_a[perm], index=rows.index)
    pb = pd.Series(p_b[perm], index=rows.index)

    want = expected_rows(pa, pb, rows)
    assert len({w["races"] for w in want.values()}) > 1, "segments should differ in size"
    assert_segment(checked_table(pa, pb, rows), want)
    flipped = checked_table(pb, pa, rows)
    for s in SEGMENTS:
        assert float(flipped.loc[s, "delta_ll"]) == pytest.approx(-want[s]["delta_ll"], abs=1e-12)


def expected_gain(p_true, p_a, p_b, rows: pd.DataFrame, races) -> float:
    """The planted gain: the mean over `races` of sum_i p_true_i (ln p_a_i - ln p_b_i)."""
    mine = rows["race_id"].isin(races).to_numpy()
    term = p_true[mine] * (np.log(p_a[mine]) - np.log(p_b[mine]))
    return float(pd.Series(term).groupby(rows["race_id"].to_numpy()[mine]).sum().mean())


def test_segment_delta_ll_finds_a_planted_debut_gain_standalone_and_at_the_final_price():
    """Debut races' winners follow the market plus a lineage signal L the market lacks; every
    other race's follow the market. At the final price (the meta with L against the market q)
    and standalone (an imperfect ability model with L against it without), the debut segment's
    ΔLL is the planted gain (within four standard errors) and its lower bound is above 0; in the
    first-surface and first-distance segments, where L is 0 and the two forecasts agree, the
    ΔLL and both bounds are 0."""
    rng = np.random.default_rng(9552)
    kinds = ("debut", "surface", "distance", "debut", "both", "plain")
    frame = race_frame(9552, kinds, meetings=120, races_per_meeting=5)
    codes = frame["code"].to_numpy()
    debut = (frame["kind"] == "debut").to_numpy()
    ln_q = np.log(softmax_within(rng.normal(0.0, 1.0, len(frame)), codes))
    lineage = np.where(debut, rng.normal(0.0, 0.8, len(frame)), 0.0)
    truth = softmax_within(ln_q + lineage, codes)
    frame["won"] = draw_winners(rng, truth, codes)
    ability = 0.6 * ln_q + rng.normal(0.0, 0.5, len(frame))
    pairs = {
        "final price": (truth, np.exp(ln_q)),
        "standalone": (softmax_within(ability + lineage, codes), softmax_within(ability, codes)),
    }
    debut_races = frame.loc[debut, "race_id"].unique()
    for name, (a, b) in pairs.items():
        pa, pb = pd.Series(a, index=frame.index), pd.Series(b, index=frame.index)
        got = checked_table(pa, pb, frame)
        assert_segment(got, expected_rows(pa, pb, frame))
        d = per_race_delta(pa, pb, frame)
        d = d.loc[d.index.isin(debut_races)]
        planted = expected_gain(truth, a, b, frame, debut_races)
        assert planted > 0.02, (name, planted)
        se = d.std(ddof=1) / np.sqrt(len(d))
        assert float(got.loc["debut", "delta_ll"]) == pytest.approx(planted, abs=4 * se), name
        assert float(got.loc["debut", "lower"]) > 0.0, name
        for s in ("first_surface", "first_distance"):
            for col in ("delta_ll", "lower", "upper"):
                assert float(got.loc[s, col]) == pytest.approx(0.0, abs=1e-12), (name, s, col)
