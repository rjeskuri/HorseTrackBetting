"""The competition-terms study (ticket #531, epic #73; #94's outline 4; 08 §3.3; 05 §3.4, H12).

Contract pinned here (`horsetrack.countries.japan.studies.competition`, `cs` below; the coder
builds what these tests call, wired like #92's `studies.ability`):

Constants. `COUNTRY == "japan"`, `TICKET == 531`; `FAMILY_ID`, a registry family of its own (not
#92's); `SOURCES` names "jra" and "jravan" (the sire is JRA-VAN's: every figure is withheld, D21).
`TERMS == ("field_mean", "field_max", "field_x_rating")`, #94's columns. `VARIANTS`, the
pre-registered variants as name -> the terms added to #92's pooled model: exactly four, each term
alone under its own name and one variant (any name) with all three together. `REFF`, the variant
name of the R_eff-by-bucket record. `PLANNED`, the (hypothesis_id, variant) of every record a
completed run appends: unique, naming each VARIANTS name and REFF. `N5_QUANTILES` and
`REFF_QUANTILES`: strictly increasing fractions in (0, 1), N5's claim bands and R_eff's buckets.

The spec. `build_spec(scheme, bootstrap) -> dict`: plain JSON, frozen by `freeze_family` before
any outcome is read: `spec["family_id"] == FAMILY_ID`, `spec["variants"] == {name: list(terms)}`,
`spec["bootstrap"] == bootstrap` (the only part of a spec the scoring below reads); it names #92's
family (`ability.FAMILY_ID`, the base model) and holds both quantile lists.

Field terms. `field_features(frame, rating) -> pd.DataFrame`, the TERMS columns on frame's index:
#94's `interaction_columns` of `rating` (a before-the-race rating per frame row, `models.rating`'s
mu_before) over the other runners of each race_id. It reads race_id and `rating` only: no outcome
of the race moves it.

Comparability. `fold_comparability(results, rows, folds) -> pd.Series` by race_id: per fold, #94's
`race_comparability` of the fold's test rows (`rows.iloc[fold.test_idx]`: race_id, horse_id) on
`comparison_graph(results, before_utc=00:00 Asia/Tokyo of fold.test_start)`, so only races before
the fold's first day enter its graph (a JST day boundary: safe on `studies.rating.read_results`'
midnight + race_no stamps, #94); +inf for an unconnected pair or a horse the graph never saw; one
value per race of the folds' test rows.

Buckets. `reff_buckets(reff, residual, clusters, spec) -> pd.DataFrame` (the three Series by
race_id). Over the races with a residual and a non-NaN R_eff: the finite R_eff in quantile bands at
REFF_QUANTILES (`np.quantile` edges; a race in band k when lo <= R < hi, the last band closed:
`core.eval.tail`'s rule), then one row with bucket "unconnected" for R_eff = +inf. Columns: bucket,
races, meetings, reff_lo, reff_hi, residual (the bucket's mean), lower and upper
(`paired_bootstrap` of its residuals over `clusters` at spec["bootstrap"]).

Scoring. `score_variants(rows, folds, reff, spec) -> Study`. rows: the level-2 frame, one row per
priced starter (race_id, race_date, meeting_id, horse_id, won, win_odds, q, ln_q, p_cal;
`ability.POOLED`, #92's pooled member's ln p renormalised over the race's priced starters; one
column per VARIANTS name, that variant's member alike); folds: positions into rows; reff:
`fold_comparability`'s output. Over the scored races (`studies.meta.Scored.of(rows, folds)`), one
Variant per VARIANTS name (its `name`) with
  - `all_folds` `delta_ll_standalone`: the paired per-race winner ΔLL of the variant's member alone
    against #92's pooled member alone (positive: the term helps), D7 meeting-bootstrap lo and hi,
    n the scored races;
  - `all_folds` `delta_ll_final_price`: the paired ΔLL of the variant's stage-2 meta (`models.meta`
    stage2 on [member] and ln q, ridge 0, unbounded: `ability.STAGE2`'s form, out of fold) against
    the market-only stage-2 meta (`ability.market_only_oof`), bounds and n alike;
  - `development` `return_per_100`, its tail return at the final price: 100 x the top band of N5
    (`core.eval.tail.tail_curve` over every scored runner as a flat Win bet: claim = the stage-2
    meta's p x win_odds, return = win_odds for a winner and 0 otherwise, meeting clusters,
    N5_QUANTILES, spec["bootstrap"]): realised, lo and hi x 100, n the band's bets;
  - `params["leaderboard"]`, `docs/specs/leaderboard.md`'s block: kind "member", pool "WIN", n_unit
    "bets", ci_level the bootstrap level, so the record is a measured member row;
and one Variant named REFF, with `tables["reff_buckets"]` = `reff_buckets(reff, residual,
clusters, spec)`'s rows, residual = the per-race winner log loss of #92's pooled member's stage-2
meta less the market-only meta's. A variant whose member column is missing (its level-1 fit
failed) is a failed Variant; every other variant still scores.

The run. `load_inputs(log=print)` returns an object with `level1`, `results` and `price` frames,
development days only. `compute_study(inputs, spec, scheme, log=print) -> Study`; every store read
is in `load_inputs`. `run_study(*, out_dir=None, log=print) -> dict` (spec_hash, report_path):
freezes `build_spec`'s spec in `default_registry("japan")` before `load_inputs` runs; appends one
record per Variant (family FAMILY_ID, ticket 531, sources SOURCES; a failed Variant is a failed
record with its error) or, when `load_inputs` or `compute_study` raises, one failed record per
PLANNED entry, then re-raises; writes the private report (each variant's metrics with their bounds,
a failed variant's error, the reff_buckets table) through `write_report` under out_dir, refused
inside the repository; regenerates the committed registry summary, which shows no figure. The
holdout is never opened. `competition_study_command` is `horsetrack japan competition-study`.

Not pinned: the level-1 fits (#92's machinery, `compute_study`), per-fold points, other metrics,
the report's layout and artifact names, the experiment report and ledger row (prose, for the
reviewer), a lone runner's field terms, dead-heat settlement. Expected values come from first
principles (hand-worked resistances, a Newton fit of the power of q, direct loops) or from library
code that is not under test (#94's graph, `models.meta`, `core.eval.tail`, `paired_bootstrap`).
Every frame is synthetic, every RNG seeded and every path under tmp_path; the repository root is
moved to tmp_path, and an autouse guard fails a test that opens the holdout or changes the
committed registry summary or a holdout log.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import copy
import dataclasses
import json
import math
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horsetrack.core import env, paths
from horsetrack.core.eval import holdout
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import Fold, FoldScheme, walk_forward
from horsetrack.core.eval.leaderboard import BLOCK_FIELDS, SELECTIONS
from horsetrack.core.eval.registry import (
    FamilyAlreadyFrozen,
    FoldMetric,
    Registry,
    default_registry,
)
from horsetrack.core.eval.reports import LicensedOutputInRepo
from horsetrack.core.eval.tail import tail_curve
from horsetrack.countries.japan.models import competition
from horsetrack.countries.japan.models.meta import MetaSpec
from horsetrack.countries.japan.research.leaderboard_build import build_rows
from horsetrack.countries.japan.studies import ability
from horsetrack.countries.japan.studies import competition as cs
from horsetrack.countries.japan.studies.calibration import Study, Variant
from horsetrack.countries.japan.studies.meta import Scored, _meta_oof

# isort: split
from competition_synth import (
    jst_midnight,
    level2_rows,
    post_stamp,
    random_results,
    softmax_within,
    write_store,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED_SUMMARY = REPO_ROOT / "docs" / "experiments" / "registry"
REAL_COUNTRIES = REPO_ROOT / "src" / "horsetrack" / "countries"
DATA_DIR_VAR = "HORSETRACK_DATA_DIR"

SCHEME = FoldScheme(
    unit="quarter",
    test_length=1,
    step=1,
    min_train=2,
    window="expanding",
    embargo_days=7,
    anchor_month=1,
)
BOOT = {"cluster_unit": "meeting", "sided": "one-sided", "level": 0.95, "n_boot": 300, "seed": 531}
SPEC = {"bootstrap": BOOT}
TERMS = ("field_mean", "field_max", "field_x_rating")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def quiet(message) -> None:
    pass


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """The data directory and the repository root are under tmp_path; the holdout is never
    opened; no test changes the real committed summary or a holdout log."""
    monkeypatch.setenv(DATA_DIR_VAR, str(tmp_path / "data"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(env, "repo_root", lambda: repo)
    opened = []

    def refuse(self, *args, **kwargs):
        opened.append(kwargs)
        raise AssertionError("the locked holdout was opened")

    monkeypatch.setattr(holdout.LockedHoldout, "open", refuse)

    def snapshot() -> dict[Path, bytes]:
        files = [*COMMITTED_SUMMARY.rglob("*"), *REAL_COUNTRIES.glob("*/holdout_log.jsonl")]
        return {p: p.read_bytes() for p in files if p.is_file()}

    before = snapshot()
    yield repo
    assert opened == [], "the locked holdout was opened"
    assert snapshot() == before, "a test changed the committed registry summary or a holdout log"


def combined_name() -> str:
    together = [name for name, terms in cs.VARIANTS.items() if len(terms) > 1]
    assert len(together) == 1, cs.VARIANTS
    return together[0]


def hypothesis_of(name: str) -> str:
    found = [h for h, v in cs.PLANNED if v == name]
    assert len(found) == 1, f"PLANNED names {name!r} {len(found)} times"
    return found[0]


def one(metrics, fold: str, name: str) -> FoldMetric:
    found = [m for m in metrics if m.fold == fold and m.metric == name]
    assert len(found) == 1, f"one {fold} {name} metric expected, got {found}"
    return found[0]


def contains(value, wanted) -> bool:
    """Whether `wanted` is `value` or any value nested in it."""
    if value == wanted:
        return True
    if isinstance(value, dict):
        return any(contains(v, wanted) for v in value.values())
    if isinstance(value, list):
        return any(contains(v, wanted) for v in value)
    return False


def quantiles_ok(qs) -> bool:
    qs = list(qs)
    return (
        bool(qs)
        and all(0 < q < 1 for q in qs)
        and all(a < b for a, b in zip(qs, qs[1:], strict=False))
    )


# --------------------------------------------------------------------------- the variants


def test_the_four_variants_are_preregistered_in_a_frozen_spec(tmp_path):
    assert cs.COUNTRY == "japan" and cs.TICKET == 531
    assert cs.FAMILY_ID != ability.FAMILY_ID and cs.FAMILY_ID.strip()
    assert {"jra", "jravan"} <= set(cs.SOURCES)
    assert tuple(cs.TERMS) == TERMS
    assert set(TERMS) == {*competition.FIELD_COLUMNS, competition.INTERACTION}
    assert len(cs.VARIANTS) == 4
    singles = {name: tuple(terms) for name, terms in cs.VARIANTS.items() if len(terms) == 1}
    assert singles == {term: (term,) for term in TERMS}, "each term alone, under its own name"
    together = cs.VARIANTS[combined_name()]
    assert sorted(together) == sorted(TERMS), "one variant adds all three terms together"
    assert quantiles_ok(cs.N5_QUANTILES) and quantiles_ok(cs.REFF_QUANTILES)

    planned = list(cs.PLANNED)
    assert len(set(planned)) == len(planned)
    names = [name for _, name in planned]
    assert cs.REFF not in cs.VARIANTS
    for name in [*cs.VARIANTS, cs.REFF]:
        hypothesis_of(name)
    assert all(isinstance(h, str) and h.strip() for h, _ in planned)

    spec = cs.build_spec(SCHEME, BOOT)
    assert json.loads(json.dumps(spec)) == spec, "plain JSON, as freeze_family stores it"
    assert spec["family_id"] == cs.FAMILY_ID
    assert spec["variants"] == {name: list(terms) for name, terms in cs.VARIANTS.items()}
    assert spec["bootstrap"] == BOOT
    assert contains(spec, list(cs.N5_QUANTILES)), "N5's bands are pre-registered"
    assert contains(spec, list(cs.REFF_QUANTILES)), "R_eff's buckets are pre-registered"
    text = json.dumps(spec)
    assert ability.FAMILY_ID in text, "the spec names the base model the terms are added to"
    other = cs.build_spec(dataclasses.replace(SCHEME, embargo_days=14), BOOT)
    assert other != spec, "the folds are part of the frozen spec"

    registry = Registry(tmp_path / "registry.jsonl", holdout_log=tmp_path / "log.jsonl")
    digest = registry.freeze_family(cs.FAMILY_ID, spec)
    assert registry.frozen_spec(cs.FAMILY_ID) == spec
    assert registry.freeze_family(cs.FAMILY_ID, cs.build_spec(SCHEME, BOOT)) == digest
    assert set(names) >= set(cs.VARIANTS)


# --------------------------------------------------------------------------- field terms


def field_frame(seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Races of 2 to 12 runners (one with two runners tied on the top rating), rows shuffled
    onto a non-default index, with finishing positions and a winner flag."""
    rng = np.random.default_rng(seed)
    parts = []
    for r, n in enumerate([2, 3, 5, 8, 12, 2, 4, 7, 9, 6, 10, 3]):
        rating = np.round(rng.normal(0.0, 1.0, n), 3)
        if r == 4:
            rating[[1, 6]] = rating.max() + 0.5  # two runners share the top rating
        pos = rng.permutation(n) + 1
        parts.append(
            pd.DataFrame(
                {
                    "race_id": f"jra:r{r:02d}",
                    "horse_id": [f"h{r:02d}{i:02d}" for i in range(n)],
                    "finish_pos": pos,
                    "won": pos == 1,
                    "mu_before": rating,
                }
            )
        )
    frame = pd.concat(parts, ignore_index=True)
    order = rng.permutation(len(frame))
    frame = frame.iloc[order].set_axis(pd.Index(1000 + 3 * order))
    return frame, frame["mu_before"].rename("rating")


def expected_field(frame: pd.DataFrame, rating: pd.Series) -> pd.DataFrame:
    """By a direct loop: each runner's other runners in its race."""
    out = pd.DataFrame(index=frame.index, columns=list(TERMS), dtype=float)
    for i in frame.index:
        others = rating[(frame["race_id"] == frame.at[i, "race_id"]) & (frame.index != i)]
        mean = float(others.mean())
        out.loc[i] = [mean, float(others.max()), mean * float(rating[i])]
    return out


def test_field_features_are_the_other_runners_before_race_rating_and_ignore_the_outcome():
    frame, rating = field_frame(5311)
    expected = expected_field(frame, rating)

    got = cs.field_features(frame, rating)

    assert got.index.equals(frame.index)
    assert set(TERMS) <= set(got.columns), sorted(got.columns)
    for col in TERMS:
        np.testing.assert_allclose(got[col].to_numpy(float), expected[col].to_numpy(float))

    # The race's outcome is never read: other finishing orders, another winner, no outcome at all.
    rng = np.random.default_rng(5312)
    reshuffled = frame.assign(finish_pos=rng.permutation(frame["finish_pos"].to_numpy()))
    reshuffled["won"] = reshuffled["finish_pos"].eq(1)
    for other in (reshuffled, frame.drop(columns=["finish_pos", "won"])):
        again = cs.field_features(other, rating)
        for col in TERMS:
            np.testing.assert_array_equal(again[col].to_numpy(float), got[col].to_numpy(float))

    # Another race's rating never moves a race's terms; a rival's rating in the race does.
    race = frame["race_id"].to_numpy()
    moved = rating.copy()
    moved[race == "jra:r03"] += 2.0
    again = cs.field_features(frame, moved)
    mine = race == "jra:r07"
    for col in TERMS:
        np.testing.assert_array_equal(
            again.loc[mine, col].to_numpy(), got.loc[mine, col].to_numpy()
        )
    rival = race == "jra:r03"
    assert not np.allclose(again.loc[rival, "field_mean"], got.loc[rival, "field_mean"])


# --------------------------------------------------------------------------- comparability

F1, F2 = date(2015, 4, 1), date(2015, 7, 1)
# (race_id, day, race_no, horses, fold): fold None is a race before every fold. Every race is a
# result; the races with a fold are that fold's test rows. x1 runs on F1's first day (00:05 JST)
# and y1 on F2's (00:03 JST): neither may enter its own fold's graph.
HAND = [
    ("p1", date(2015, 3, 7), 1, "AB", None),
    ("p2", date(2015, 3, 8), 1, "BC", None),
    ("p3", date(2015, 3, 14), 2, "DEF", None),
    ("p4", date(2015, 3, 31), 12, "GH", None),
    ("x1", F1, 5, "AD", 0),
    ("t1", date(2015, 4, 4), 1, "AC", 0),
    ("t2", date(2015, 4, 4), 2, "ABC", 0),
    ("t3", date(2015, 4, 5), 1, "DE", 0),
    ("t4", date(2015, 4, 5), 2, "AD", 0),
    ("t5", date(2015, 4, 11), 1, "GH", 0),
    ("t6", date(2015, 4, 11), 2, "AZ", 0),
    ("y1", F2, 3, "YZ", 1),
    ("u1", date(2015, 7, 4), 1, "GH", 1),
    ("u2", date(2015, 7, 4), 2, "ZY", 1),
    ("u3", date(2015, 7, 5), 1, "EF", 1),
    ("u4", date(2015, 7, 5), 2, "BC", 1),
    ("v1", date(2015, 7, 11), 1, "GH", 1),
]
INF = math.inf
# Worked by hand. A race of n joins each pair at conductance 1/(n - 1) (#94); meetings add.
# F1's graph (before 2015-04-01 00:00 JST): A-B 1, B-C 1; the triangle D, E, F at 1/2 an edge;
# G-H 1. F2's graph adds x1 and F1's test races: the triangle A, B, C at 3/2 an edge (p1, p2,
# t1 at 1, t2 at 1/2), D-E at 3/2 (p3, t3), D-F and E-F at 1/2, A-D at 2, A-Z at 1, G-H at 2.
EXPECTED_HAND = {
    "x1": INF,  # A and D: x1 itself is not before F1, so they are in two components
    "t1": 2.0,  # A-B-C in series
    "t2": 4 / 3,  # pairs 1, 1 and 2
    "t3": 4 / 3,  # 2 in parallel with 2 + 2
    "t4": INF,
    "t5": 1.0,
    "t6": INF,  # Z never ran before F1
    "y1": INF,  # Y never ran before F2 (y1 is on F2's first day)
    "u1": 0.5,  # p4 and t5 in parallel; v1 (after F2's start) is not read
    "u2": INF,
    "u3": 8 / 7,  # E-F 2 in parallel with E-D 2/3 + D-F 2; A's side hangs on D alone
    "u4": 4 / 9,  # B-C 2/3 in parallel with B-A-C 4/3
    "v1": 0.5,
}


def results_of(races) -> pd.DataFrame:
    rows = [
        (rid, pd.Timestamp(day), race_no, post_stamp(day, race_no), horse, pos)
        for rid, day, race_no, horses, _ in races
        for pos, horse in enumerate(horses, start=1)
    ]
    out = pd.DataFrame(
        rows, columns=["race_id", "race_date", "race_no", "race_utc", "horse_id", "finish_pos"]
    )
    out["race_utc"] = pd.to_datetime(out["race_utc"], utc=True)
    return out


def quarter_end(start: date) -> date:
    month = start.month + 3
    nxt = date(start.year + (month > 12), (month - 1) % 12 + 1, 1)
    return nxt - timedelta(days=1)


def folds_over(rows: pd.DataFrame, starts: list[date]) -> list[Fold]:
    """One fold per quarter start: its test rows are the rows dated in the quarter."""
    day = rows["race_date"].dt.date.to_numpy()
    out = []
    for start in starts:
        end = quarter_end(start)
        test = np.flatnonzero((day >= start) & (day <= end))
        train = np.flatnonzero(day < start - timedelta(days=7))
        out.append(Fold(train, test, start - timedelta(days=8), start, end))
    return out


def test_fold_comparability_uses_each_folds_graph_before_its_first_day():
    results = results_of(HAND)
    tested = [rid for rid, *_, fold in HAND if fold is not None]
    rows = results.loc[results["race_id"].isin(tested), ["race_id", "horse_id", "race_date"]]
    rows = rows.reset_index(drop=True)
    folds = folds_over(rows, [F1, F2])
    assert [len(f.test_idx) for f in folds] == [15, 12]

    got = cs.fold_comparability(results, rows, folds)

    assert isinstance(got, pd.Series)
    assert set(got.index) == set(EXPECTED_HAND)
    for rid, want in EXPECTED_HAND.items():
        if math.isinf(want):
            assert math.isinf(got[rid]) and got[rid] > 0, (rid, got[rid])
        else:
            assert got[rid] == pytest.approx(want, abs=1e-9), rid


def link_chain(horses, day: date, prefix: str) -> pd.DataFrame:
    """Two-runner races joining consecutive horses, on `day` from race_no 1."""
    races = [
        (f"{prefix}{k:03d}", day, k + 1, [a, b], None)
        for k, (a, b) in enumerate(zip(horses, horses[1:], strict=False))
    ]
    return results_of(races)


def test_fold_comparability_reads_only_the_races_before_each_fold():
    results = random_results(5313, date(2015, 1, 3), date(2015, 12, 26))
    starts = [date(2015, 4, 1), date(2015, 7, 1), date(2015, 10, 1)]
    rows = results.loc[results["race_date"] >= pd.Timestamp(starts[0])]
    rows = rows[["race_id", "horse_id", "race_date"]].reset_index(drop=True)
    folds = folds_over(rows, starts)

    got = cs.fold_comparability(results, rows, folds)

    reference = {}
    for fold in folds:
        graph = competition.comparison_graph(results, before_utc=jst_midnight(fold.test_start))
        reference[fold.test_start] = competition.race_comparability(rows.iloc[fold.test_idx], graph)
    want = pd.concat(reference.values())
    assert set(got.index) == set(want.index)
    np.testing.assert_allclose(got.reindex(want.index).to_numpy(), want.to_numpy(), rtol=1e-9)
    assert np.isinf(want).any() and np.isfinite(want).any(), "both kinds of race occur"

    for k, fold in enumerate(folds):
        mine = reference[fold.test_start].index
        horses = list(pd.unique(rows.iloc[fold.test_idx]["horse_id"]))
        # Races from the fold's first day on (here, joining every horse of its races) are never
        # read by its graph, nor by an earlier fold's.
        later = pd.concat([results, link_chain(horses, fold.test_start, "late")])
        again = cs.fold_comparability(later, rows, folds)
        earlier_folds = [r for f in folds[: k + 1] for r in reference[f.test_start].index]
        np.testing.assert_array_equal(
            again.reindex(earlier_folds).to_numpy(), got.reindex(earlier_folds).to_numpy()
        )
        # The same races the day before the fold do enter it.
        before = pd.concat([results, link_chain(horses, fold.test_start - timedelta(1), "pre")])
        moved = cs.fold_comparability(before, rows, folds)
        assert np.isfinite(moved.reindex(mine)).all(), "joined before the fold: every race finite"
        assert not np.array_equal(moved.reindex(mine).to_numpy(), got.reindex(mine).to_numpy())


# --------------------------------------------------------------------------- R_eff buckets


def bucket_stats(part: pd.Series, clusters: pd.Series, lo: float, hi: float) -> dict:
    iv = paired_bootstrap(
        part, clusters, n_boot=BOOT["n_boot"], level=BOOT["level"], seed=BOOT["seed"]
    )
    return {
        "races": len(part),
        "meetings": int(clusters.reindex(part.index).nunique()),
        "reff_lo": lo,
        "reff_hi": hi,
        "residual": float(part.mean()),
        "lower": iv.lo,
        "upper": iv.hi,
    }


def expected_buckets(reff, residual, clusters) -> tuple[list[dict], dict | None]:
    """First principles: (the finite bands in R_eff order, the unconnected bucket)."""
    races = residual.dropna().index
    r = reff.reindex(races)
    races = races[r.notna().to_numpy()]
    r, res = reff.reindex(races).to_numpy(float), residual.reindex(races)
    finite = np.isfinite(r)
    qs = np.r_[0.0, list(cs.REFF_QUANTILES), 1.0]
    bands = []
    if finite.any():
        edges = np.quantile(r[finite], qs)
        for k in range(len(qs) - 1):
            lo, hi = edges[k], edges[k + 1]
            inside = finite & (r >= lo) & ((r <= hi) if k == len(qs) - 2 else (r < hi))
            bands.append(bucket_stats(res[inside], clusters, lo, hi))
    unconnected = None
    if (~finite).any():
        unconnected = bucket_stats(res[~finite], clusters, INF, INF)
    return bands, unconnected


def assert_buckets(got, reff, residual, clusters) -> None:
    table = pd.DataFrame(got)
    columns = {"bucket", "races", "meetings", "reff_lo", "reff_hi", "residual", "lower", "upper"}
    assert columns <= set(table.columns), sorted(table.columns)
    bands, unconnected = expected_buckets(reff, residual, clusters)
    flagged = table.loc[table["bucket"].eq("unconnected")]
    assert len(flagged) == 1, "one unconnected bucket"
    finite = table.loc[~table["bucket"].eq("unconnected")].sort_values("reff_lo", kind="stable")
    assert len(finite) == len(cs.REFF_QUANTILES) + 1
    assert table["bucket"].is_unique
    pairs = [(row, want) for row, want in zip(finite.to_dict("records"), bands, strict=True)]
    pairs.append((flagged.iloc[0].to_dict(), unconnected))
    for row, want in pairs:
        assert int(row["races"]) == want["races"], (row["bucket"], row)
        assert int(row["meetings"]) == want["meetings"], row["bucket"]
        for key in ("residual", "lower", "upper"):
            assert float(row[key]) == pytest.approx(want[key], abs=1e-9), (row["bucket"], key)
        if np.isfinite(want["reff_lo"]):
            for key in ("reff_lo", "reff_hi"):
                assert float(row[key]) == pytest.approx(want[key], abs=1e-12), (row["bucket"], key)


def bucket_case(seed: int):
    """400 races in 100 meetings of four: R_eff lognormal, +inf for about one race in six, NaN
    for one in twenty; the residual rises with ln R_eff and is higher still when unconnected;
    some races lack a residual."""
    rng = np.random.default_rng(seed)
    races = pd.Index([f"jra:b{i:04d}" for i in range(400)])
    clusters = pd.Series([f"m{i // 4:03d}" for i in range(400)], index=races)
    reff = pd.Series(np.exp(rng.normal(0.0, 0.7, 400)), index=races)
    draw = rng.random(400)
    reff[draw < 0.16] = INF
    reff[(draw >= 0.16) & (draw < 0.21)] = np.nan
    trend = np.where(np.isfinite(reff), 0.03 * np.log(reff.where(np.isfinite(reff), 1.0)), 0.08)
    residual = pd.Series(trend + rng.normal(0.0, 0.02, 400), index=races)
    residual[rng.random(400) < 0.05] = np.nan
    residual = residual.drop(races[rng.choice(400, 10, replace=False)])
    return reff, residual, clusters


def test_reff_buckets_split_the_residual_by_quantile_band_and_an_unconnected_bucket():
    reff, residual, clusters = bucket_case(5314)
    shuffled = reff.sample(frac=1.0, random_state=1)

    got = cs.reff_buckets(shuffled, residual, clusters, SPEC)

    assert_buckets(got, reff, residual, clusters)
    bands, unconnected = expected_buckets(reff, residual, clusters)
    assert bands[-1]["residual"] > bands[0]["residual"] + 0.02, "the planted rise"
    assert unconnected["races"] > 30


def test_reff_buckets_take_a_field_of_strangers_without_error():
    reff, residual, clusters = bucket_case(5315)
    strangers = reff.where(reff.isna(), INF)  # every race unconnected

    got = pd.DataFrame(cs.reff_buckets(strangers, residual, clusters, SPEC))

    flagged = got.loc[got["bucket"].eq("unconnected")]
    _, unconnected = expected_buckets(strangers, residual, clusters)
    assert len(flagged) == 1 and int(flagged["races"].iloc[0]) == unconnected["races"]
    assert float(flagged["residual"].iloc[0]) == pytest.approx(unconnected["residual"], abs=1e-12)
    assert int(got.loc[~got["bucket"].eq("unconnected"), "races"].sum()) == 0


# --------------------------------------------------------------------------- scoring

ROLE_OF = {"field_mean": "signal", "field_max": "noise", "field_x_rating": "same"}


def power_of_q_beta(ln_q: np.ndarray, race: np.ndarray, won: np.ndarray) -> float:
    """The maximum-likelihood beta of P(i wins) = q_i^beta / sum_j q_j^beta, by Newton."""
    codes = pd.factorize(race)[0]
    k = int(codes.max()) + 1
    beta = 1.0
    for _ in range(100):
        p = softmax_within(beta * ln_q, codes)
        mean = np.bincount(codes, weights=p * ln_q, minlength=k)
        second = np.bincount(codes, weights=p * ln_q * ln_q, minlength=k)
        chosen = np.bincount(codes, weights=np.where(won, ln_q, 0.0), minlength=k)
        step = (chosen - mean).sum() / (second - mean * mean).sum()
        beta += step
        if abs(step) < 1e-13:
            break
    return beta


def market_only(rows: pd.DataFrame, folds: list[Fold]) -> pd.Series:
    """The market-only stage-2 meta out of fold, from first principles."""
    ln_q, race = rows["ln_q"].to_numpy(), rows["race_id"].to_numpy(dtype=object)
    won = rows["won"].to_numpy(dtype=bool)
    out = np.full(len(rows), np.nan)
    for fold in folds:
        tr, te = fold.train_idx, fold.test_idx
        beta = power_of_q_beta(ln_q[tr], race[tr], won[tr])
        out[te] = softmax_within(beta * ln_q[te], pd.factorize(race[te])[0])
    return pd.Series(out, index=rows.index)


def winner_loss(p: pd.Series, rows: pd.DataFrame) -> pd.Series:
    """-ln p(winner) per race, p renormalised over the race's rows (one winner a race)."""
    p = p.reindex(rows.index).to_numpy(float)
    race = rows["race_id"].to_numpy()
    p = p / pd.Series(p).groupby(race).transform("sum").to_numpy()
    won = rows["won"].to_numpy(bool)
    return pd.Series(-np.log(p[won]), index=race[won])


def stage2(rows: pd.DataFrame, folds: list[Fold], member: str) -> pd.Series:
    spec = MetaSpec([member], quadratic=False, segment_cols=[], form="stage2", ridge=0.0)
    return _meta_oof(rows, folds, spec, "ln_q")[0]


def bounds(per_race: pd.Series, clusters: pd.Series):
    return paired_bootstrap(
        per_race, clusters, n_boot=BOOT["n_boot"], level=BOOT["level"], seed=BOOT["seed"]
    )


@pytest.fixture(scope="module")
def scoring():
    """The level-2 rows, folds and R_eff, `score_variants`' study and the references."""
    roles = {name: ROLE_OF.get(name, "both") for name in cs.VARIANTS}
    rows = level2_rows(5316, roles, ability.POOLED)
    folds = [f for f in walk_forward(rows["race_date"], scheme=SCHEME) if len(f.test_idx)]
    assert len(folds) >= 5
    spec = cs.build_spec(SCHEME, BOOT)
    scored = Scored.of(rows, folds)
    sr, clusters = scored.rows, scored.clusters
    rng = np.random.default_rng(5317)
    races = pd.Index(rows["race_id"].unique())
    reff = pd.Series(np.exp(rng.normal(0.0, 0.6, len(races))), index=races)
    draw = rng.random(len(races))
    reff[draw < 0.18] = INF
    reff[(draw >= 0.18) & (draw < 0.22)] = np.nan

    study = cs.score_variants(rows, folds, reff, spec)

    p_market = market_only(rows, folds)
    market_ll = winner_loss(p_market, sr)
    base_ll = winner_loss(np.exp(sr[ability.POOLED]), sr)
    want = {}
    for name in cs.VARIANTS:
        standalone = base_ll - winner_loss(np.exp(sr[name]), sr)
        p_meta = stage2(rows, folds, name)
        final = market_ll - winner_loss(p_meta, sr)
        bets = pd.DataFrame(
            {
                "claim": p_meta.reindex(sr.index).to_numpy() * sr["win_odds"].to_numpy(),
                "ret": np.where(sr["won"].to_numpy(bool), sr["win_odds"].to_numpy(), 0.0),
                "meeting_id": sr["meeting_id"].to_numpy(),
            }
        )
        curve = tail_curve(
            bets,
            claim_col="claim",
            return_col="ret",
            cluster_col="meeting_id",
            quantiles=list(cs.N5_QUANTILES),
            n_boot=BOOT["n_boot"],
            level=BOOT["level"],
            seed=BOOT["seed"],
        )
        want[name] = {
            "standalone": (standalone, bounds(standalone, clusters)),
            "final": (final, bounds(final, clusters)),
            "tail": curve.loc[curve["segment"].eq("all")].iloc[-1],
        }
    residual = winner_loss(stage2(rows, folds, ability.POOLED), sr) - market_ll
    return {
        "rows": rows,
        "folds": folds,
        "spec": spec,
        "reff": reff,
        "scored": scored,
        "study": study,
        "want": want,
        "residual": residual,
    }


def variants_by_name(study: Study) -> dict[str, Variant]:
    names = [v.name for v in study.variants]
    assert len(set(names)) == len(names), names
    assert set(names) == {*cs.VARIANTS, cs.REFF}, names
    return {v.name: v for v in study.variants}


def test_each_variant_is_scored_standalone_at_the_final_price_and_in_the_tail(scoring):
    by_name = variants_by_name(scoring["study"])
    n_races = scoring["scored"].n_races
    level = BOOT["level"]
    means = {}
    for name in cs.VARIANTS:
        v = by_name[name]
        assert v.error is None, (name, v.error)
        want = scoring["want"][name]
        for metric, key in (
            ("delta_ll_standalone", "standalone"),
            ("delta_ll_final_price", "final"),
        ):
            per_race, iv = want[key]
            m = one(v.metrics, "all_folds", metric)
            assert m.n == n_races == len(per_race), (name, metric)
            assert m.value == pytest.approx(per_race.mean(), abs=1e-7), (name, metric)
            assert m.lo == pytest.approx(iv.lo, abs=1e-7), (name, metric)
            assert m.hi == pytest.approx(iv.hi, abs=1e-7), (name, metric)
            means[(name, key)] = m.value
        top = want["tail"]
        tail = one(v.metrics, "development", "return_per_100")
        assert tail.value == pytest.approx(100 * top["realised"], abs=1e-6), name
        assert tail.lo == pytest.approx(100 * top["lo"], abs=1e-6), name
        assert tail.hi == pytest.approx(100 * top["hi"], abs=1e-6), name
        assert tail.n == int(top["n_bets"]), name

        block = v.params["leaderboard"]
        assert set(block) == set(BLOCK_FIELDS), sorted(block)
        assert block["kind"] == "member" and block["pool"] == "WIN", block
        assert block["n_unit"] == "bets" and block["ci_level"] == level, block
        assert block["selection"] in SELECTIONS and str(block["rule"]).strip(), block

    # The fixture's plants: the signal term helps, the noise term hurts, the copy is a tie.
    assert means[("field_mean", "standalone")] > 0.005
    assert means[("field_max", "standalone")] < 0.0
    same = one(by_name["field_x_rating"].metrics, "all_folds", "delta_ll_standalone")
    assert (same.value, same.lo, same.hi) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert means[("field_mean", "final")] > means[("field_max", "final")]


def test_reff_is_reported_against_the_meta_models_residual_by_bucket(scoring):
    by_name = variants_by_name(scoring["study"])
    v = by_name[cs.REFF]
    assert v.error is None, v.error
    assert v.metrics, "the buckets are registry metrics too"
    got = scoring["study"].tables["reff_buckets"]
    scored = scoring["scored"]
    assert_buckets(got, scoring["reff"], scoring["residual"], scored.clusters)
    total = sum(int(row["races"]) for row in got)
    known = scoring["reff"].reindex(scoring["residual"].index).notna().sum()
    assert total == known, "every scored race with an R_eff is in one bucket"


def test_a_variant_whose_member_failed_is_a_failed_variant_and_the_rest_still_score(scoring):
    failed = combined_name()
    rows = scoring["rows"].drop(columns=[failed])

    study = cs.score_variants(rows, scoring["folds"], scoring["reff"], scoring["spec"])

    by_name = variants_by_name(study)
    assert isinstance(by_name[failed].error, str) and by_name[failed].error.strip()
    assert not by_name[failed].metrics
    full = variants_by_name(scoring["study"])
    for name in [n for n in cs.VARIANTS if n != failed] + [cs.REFF]:
        assert by_name[name].error is None, (name, by_name[name].error)
        assert by_name[name].metrics == full[name].metrics, name


# --------------------------------------------------------------------------- the run


def fake_store(data: Path) -> None:
    """The store files whose bytes name the run's data builds (never parsed: reads are stubbed)."""
    for path, content in (
        (data / "processed" / "jra" / "runners.parquet", b"synthetic runners"),
        (data / "processed" / "jra" / "races.parquet", b"synthetic races"),
        (data / "processed" / "jravan" / "horse_pedigree.parquet", b"synthetic pedigree"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


# Each scored single-term variant's figures (value, lo, hi), told apart from each other and from
# any count; the combined variant fails.
FIGURES = {
    "field_mean": {
        "delta_ll_standalone": (0.004173, 0.001029, 0.007317),
        "delta_ll_final_price": (0.002861, -0.000394, 0.006117),
        "return_per_100": (87.31, 71.29, 103.57),
    },
    "field_max": {
        "delta_ll_standalone": (-0.001536, -0.003872, 0.000814),
        "delta_ll_final_price": (0.000952, -0.002247, 0.004183),
        "return_per_100": (79.64, 66.08, 93.42),
    },
    "field_x_rating": {
        "delta_ll_standalone": (0.006719, 0.003358, 0.009946),
        "delta_ll_final_price": (0.003507, 0.000281, 0.006689),
        "return_per_100": (91.83, 76.47, 108.25),
    },
}
BUCKETS = [
    {"bucket": "q1", "races": 210, "meetings": 88, "reff_lo": 0.2413, "reff_hi": 0.6187}
    | {"residual": -0.01277, "lower": -0.03194, "upper": 0.00641},
    {"bucket": "unconnected", "races": 57, "meetings": 40, "reff_lo": INF, "reff_hi": INF}
    | {"residual": 0.05813, "lower": 0.02147, "upper": 0.09479},
]
FAILURE = "RuntimeError: synthetic level-1 failure"


def handmade_study() -> Study:
    study = Study(counts={"level1_races": 1234}, tables={"reff_buckets": copy.deepcopy(BUCKETS)})
    scheme = "walk_forward:synthetic"
    components = {"study": "studies/competition.py"}
    for name, figures in FIGURES.items():
        metrics = tuple(
            FoldMetric("development" if m == "return_per_100" else "all_folds", m, *f, n)
            for (m, f), n in zip(figures.items(), (936, 936, 412), strict=True)
        )
        study.variants.append(
            Variant(hypothesis_of(name), name, scheme, components, {}, metrics=metrics)
        )
    name = combined_name()
    study.variants.append(Variant(hypothesis_of(name), name, scheme, components, {}, error=FAILURE))
    reff_metrics = tuple(
        FoldMetric(
            "all_folds",
            f"residual[{b['bucket']}]",
            b["residual"],
            b["lower"],
            b["upper"],
            b["races"],
        )
        for b in BUCKETS
    )
    study.variants.append(
        Variant(hypothesis_of(cs.REFF), cs.REFF, scheme, components, {}, metrics=reff_metrics)
    )
    return study


def all_figures() -> list[float]:
    out = [x for figures in FIGURES.values() for triple in figures.values() for x in triple]
    return out + [b[k] for b in BUCKETS for k in ("residual", "lower", "upper")]


def tolerance(value: float) -> float:
    """Room for a figure printed to three significant digits or four decimals."""
    return max(0.6 * 10 ** (math.floor(math.log10(abs(value))) - 2), 6e-5)


def numbers_in(text: str) -> np.ndarray:
    return np.array([float(x) for x in NUMBER.findall(text)])


def has_figure(numbers: np.ndarray, value: float) -> bool:
    return bool(len(numbers)) and bool((np.abs(numbers - value) <= tolerance(value)).any())


def stub_run(monkeypatch, study_of, seen: dict) -> None:
    """load_inputs and compute_study stood in for, each noting what the registry held then."""
    sentinel = object()

    def load_inputs(log=print):
        registry = default_registry("japan")
        seen["frozen_at_read"] = registry.frozen_spec_hash(cs.FAMILY_ID)
        seen["records_at_read"] = list(registry.records(cs.FAMILY_ID))
        return sentinel

    def compute_study(inputs, spec, scheme, log=print):
        assert inputs is sentinel, "compute_study scores what load_inputs read"
        seen["frozen_at_compute"] = default_registry("japan").frozen_spec_hash(cs.FAMILY_ID)
        seen["spec"] = spec
        return study_of()

    monkeypatch.setattr(cs, "load_inputs", load_inputs)
    monkeypatch.setattr(cs, "compute_study", compute_study)


def test_run_study_freezes_the_spec_before_reading_and_records_every_variant(
    tmp_path, monkeypatch, isolated
):
    data, out, repo = tmp_path / "data", tmp_path / "reports", isolated
    fake_store(data)
    seen: dict = {}
    stub_run(monkeypatch, handmade_study, seen)

    result = cs.run_study(out_dir=out, log=quiet)

    registry = default_registry("japan")
    registry.verify()
    frozen = registry.frozen_spec_hash(cs.FAMILY_ID)
    assert frozen is not None and result["spec_hash"] == frozen
    assert seen["frozen_at_read"] == frozen, "the family is frozen before any store row is read"
    assert seen["records_at_read"] == [], "and nothing is scored before"
    assert seen["frozen_at_compute"] == frozen
    assert registry.frozen_spec(cs.FAMILY_ID) == seen["spec"], "the frozen spec is the one scored"
    assert seen["spec"]["variants"] == {n: list(t) for n, t in cs.VARIANTS.items()}

    records = registry.records(cs.FAMILY_ID)
    statuses = {r.variant: r.status for r in records}
    assert len(records) == len(statuses) == len(cs.VARIANTS) + 1
    assert statuses == {**dict.fromkeys(FIGURES, "scored"), combined_name(): "failed"} | {
        cs.REFF: "scored"
    }
    for r in records:
        assert r.ticket == 531 and tuple(r.sources) == tuple(cs.SOURCES), r.variant
        assert r.hypothesis_id == hypothesis_of(r.variant), r.variant
        if r.variant in FIGURES:
            for metric, (value, lo, hi) in FIGURES[r.variant].items():
                fold = "development" if metric == "return_per_100" else "all_folds"
                m = one(r.metrics, fold, metric)
                assert (m.value, m.lo, m.hi) == (value, lo, hi), (r.variant, metric)
    failed = next(r for r in records if r.status == "failed")
    assert "synthetic level-1 failure" in json.dumps(dataclasses.asdict(failed))

    # The private report, through write_report under out_dir: every figure, the failure, the
    # buckets.
    report = Path(result["report_path"])
    assert report.is_file()
    assert report.resolve().is_relative_to(out.resolve())
    assert not report.resolve().is_relative_to(repo.resolve())
    text = report.read_text(encoding="utf-8")
    numbers = numbers_in(text)
    missing = [x for x in all_figures() if not has_figure(numbers, x)]
    assert missing == [], f"the report lacks {missing}"
    assert "synthetic level-1 failure" in text
    for name in [*cs.VARIANTS, cs.REFF]:
        assert name in text, name

    # The committed summary names every variant and holds no figure (D21).
    committed = {p: p.read_text("utf-8") for p in repo.rglob("*") if p.is_file()}
    family_file = repo / "docs" / "experiments" / "registry" / "japan" / f"{cs.FAMILY_ID}.md"
    assert family_file in committed, sorted(committed)
    for name in [*cs.VARIANTS, cs.REFF]:
        assert name in committed[family_file], name
    for path, content in committed.items():
        numbers = numbers_in(content)
        leaked = [x for x in all_figures() if has_figure(numbers, x)]
        assert leaked == [], f"{path.relative_to(repo)} holds the figures {leaked}"


def test_run_study_refuses_a_family_frozen_with_another_spec_before_reading(tmp_path, monkeypatch):
    fake_store(tmp_path / "data")
    default_registry("japan").freeze_family(cs.FAMILY_ID, {"family_id": cs.FAMILY_ID, "x": 1})
    seen: dict = {}
    stub_run(monkeypatch, handmade_study, seen)

    with pytest.raises(FamilyAlreadyFrozen):
        cs.run_study(out_dir=tmp_path / "reports", log=quiet)

    assert "frozen_at_read" not in seen, "no store row is read"
    assert default_registry("japan").records(cs.FAMILY_ID) == []


class Boom(RuntimeError):
    """The failure a stubbed stage raises."""


@pytest.mark.parametrize("stage", ["load_inputs", "compute_study"])
def test_an_aborted_run_records_every_planned_variant_failed(tmp_path, monkeypatch, stage):
    data, out = tmp_path / "data", tmp_path / "reports"
    fake_store(data)
    stub_run(monkeypatch, handmade_study, {})
    boom = Boom(f"{stage} broke on the synthetic store")

    def raise_boom(*args, **kwargs):
        raise boom

    monkeypatch.setattr(cs, stage, raise_boom)

    with pytest.raises(Boom) as raised:
        cs.run_study(out_dir=out, log=quiet)

    assert raised.value is boom, "the original exception surfaces"
    written = [p for p in out.rglob("*") if p.is_file()] if out.exists() else []
    assert written == [], "an aborted run writes no report"
    registry = default_registry("japan")
    assert registry.frozen_spec_hash(cs.FAMILY_ID) is not None
    records = registry.records(cs.FAMILY_ID)
    assert sorted((r.hypothesis_id, r.variant) for r in records) == sorted(cs.PLANNED)
    for r in records:
        assert r.status == "failed", r.variant
        assert f"{stage} broke" in json.dumps(dataclasses.asdict(r)), r.variant
        assert r.ticket == 531 and tuple(r.sources) == tuple(cs.SOURCES), r.variant


def test_the_report_is_refused_inside_the_repository(tmp_path, monkeypatch, isolated):
    fake_store(tmp_path / "data")
    stub_run(monkeypatch, handmade_study, {})
    inside = isolated / "reports"

    with pytest.raises((paths.DataDirInsideRepo, LicensedOutputInRepo)):
        cs.run_study(out_dir=inside, log=quiet)

    assert not inside.exists() or not [p for p in inside.rglob("*") if p.is_file()]


def holdout_days(n: int, start: date = date(2026, 11, 28)) -> list[date]:
    days, day = [], start
    while len(days) < n:
        if holdout.is_holdout_day("japan", day):
            days.append(day)
        day += timedelta(days=1)
    return days


def test_load_inputs_reads_development_days_only(tmp_path):
    ids = write_store(tmp_path / "data", holdout_days(2), seed=5318)
    assert ids["holdout"] and ids["price_era"] < ids["development"]

    inputs = cs.load_inputs(log=quiet)

    frames = {name: getattr(inputs, name) for name in ("level1", "results", "price")}
    for name, frame in frames.items():
        read = set(frame["race_id"])
        assert not read & ids["holdout"], f"{name} holds a holdout race"
        assert read <= ids["development"], name
    assert set(frames["results"]["race_id"]) == ids["development"], "every earlier race"
    assert set(frames["price"]["race_id"]) == ids["price_era"]
    assert set(frames["level1"]["race_id"]) >= ids["price_era"]


# --------------------------------------------------------------------------- the leaderboard


def test_each_variant_is_a_leaderboard_member_row(tmp_path, monkeypatch, scoring):
    fake_store(tmp_path / "data")
    stub_run(monkeypatch, lambda: copy.deepcopy(scoring["study"]), {})

    cs.run_study(out_dir=tmp_path / "reports", log=quiet)

    registry = default_registry("japan")
    rows = build_rows(registry=registry, candidate_dir=tmp_path / "no-cells")
    mine = {row.option_id: row for row in rows if row.family_id == cs.FAMILY_ID}
    records = {r.variant: r for r in registry.records(cs.FAMILY_ID)}
    size = registry.family_size(cs.FAMILY_ID)
    assert size >= len(cs.VARIANTS) + 1
    for name in cs.VARIANTS:
        row = mine[f"{cs.FAMILY_ID}/{name}"]
        tail = one(records[name].metrics, "development", "return_per_100")
        assert row.measured and row.kind == "member" and row.pool == "WIN", row
        assert (row.return_per_100, row.ci_lo, row.ci_hi, row.n) == (
            tail.value,
            tail.lo,
            tail.hi,
            tail.n,
        )
        assert row.ci_level == BOOT["level"] and row.n_unit == "bets"
        assert row.withheld and row.ticket == 531 and row.variants_searched == size
    measured = {oid for oid, row in mine.items() if row.measured}
    assert measured == {f"{cs.FAMILY_ID}/{name}" for name in cs.VARIANTS}


# --------------------------------------------------------------------------- the command


def test_competition_study_is_a_japan_command(monkeypatch):
    from typer.testing import CliRunner

    from horsetrack.cli import app, japan

    assert "competition-study" in {c.name for c in japan.registered_commands}
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return {"family_id": cs.FAMILY_ID, "spec_hash": "0" * 64}

    monkeypatch.setattr(cs, "run_study", fake_run)

    result = CliRunner().invoke(app, ["japan", "competition-study"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
