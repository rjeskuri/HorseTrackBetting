"""The power beta with meeting-clustered confidence intervals, beside #77's Wald ones (ticket #276).

#77's beta CIs are 95% Wald intervals that treat races as independent; its ΔLL CIs are clustered by
meeting (D7). Whether beta's CIs adopt the clusters is the owner's call, so #77's family
(`calibration.FAMILY_ID`, frozen) and its spec do not change: the clustered interval is reported
beside the Wald one in a new registry family.

Contract pinned here (`horsetrack.countries.japan.studies.beta_clustered`, a sibling of
`studies.tail_beta`; `calibration.fit_power_beta(..., cluster=)` is pinned in test_calibration.py):
- `FAMILY_ID`: a non-blank string different from `calibration.FAMILY_ID`; `TICKET == 276`.
- `build_spec(owner, scheme, bootstrap, floor) -> dict` (the signature of `calibration.build_spec`):
  a plain-JSON spec with `family_id == FAMILY_ID`, frozen in the run registry before any
  development row is read; the owner's segments, entropy cut points, the fold scheme, floor and
  bootstrap are in it, so another one is another spec.
- `compute_study(frame, spec) -> calibration.Study`, pure, on `development_frame`'s shape
  (race_id, race_date, meeting_id, odds, won, plus each segment's race columns: race_no, venue and
  going here). Every variant is one `calibration.Variant` (one registry record each, unique names,
  none failed on a healthy frame, hypothesis id E1/E3/E4 as in #77). Tables, each row a dict:
  - `beta_by_year` (`subset`, `year`) and `beta_by_era` (`subset`, `era`) for both subsets of #77,
    `all_races` and `no_floor_bound_races` (races holding a runner at or under the floor, odds 1.1,
    left out); `segments` (`segment`, `value`) for each of the spec's segments over every race;
    `entropy` (`entropy bin`) over every race, one row per bin of the spec's cut points in bin
    order (a trailing row without a beta, for races left out, is allowed).
  - every fitted row also has `races` (n_races), `meetings` (n_clusters), `beta`, `se` (the Wald
    se), `lower 95%` and `upper 95%` (beta -/+ 1.96 se, #77's columns), `se cluster` (the
    meeting-clustered se), `lower 95% cluster` and `upper 95% cluster` (beta -/+ 1.96 se cluster).
  - registry metrics: for each year, `beta[year=Y]` (the Wald lo/hi) and `beta_clustered[year=Y]`
    (the clustered lo/hi) in a variant whose name contains the subset.
- `render_report(study, spec, about) -> str`: markdown that shows both intervals' columns.
- `run_study(out_dir=None) -> dict`, with the keys of `tail_beta.run_study` (family_id, spec_hash,
  report_path, records_appended, records_already_present, failed_variants): freeze the family in
  the registry first (`FamilyAlreadyFrozen` on another spec stops the run before any row is read),
  then read the development rows through the store's readers (`calibration.read_development`),
  score, append one record per variant and write the report through `write_report`. The owner's
  inputs are #77's (`calibration.owner_inputs()`, the signed yaml); the run does not recompute
  #77's entropy cut points from the store (`check_cut_points` is #77's own pre-freeze check), and
  `fit_power_beta` is `calibration`'s.

Expected values come from an independent reference below (a Newton solver for the conditional
logit and the cluster-summed score sandwich, both from first principles), never from the code
under test. Every frame is synthetic and every RNG seeded: no real data and no holdout.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import hashlib
import json
import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset
import pytest

from horsetrack.core import env, paths
from horsetrack.core.eval.folds import FoldScheme, load_country_scheme
from horsetrack.core.eval.registry import FamilyAlreadyFrozen, default_registry
from horsetrack.countries.japan.studies import calibration
from horsetrack.countries.japan.studies.beta_clustered import (
    FAMILY_ID,
    TICKET,
    build_spec,
    compute_study,
    render_report,
    run_study,
)

TAKE = 0.8
WALD_Z = 1.959963984540054
FLOOR = 1.1
RACES_PER_MEETING = 12
N_MEETINGS = 300
SIGMA = 0.4  # sd of each meeting's own beta around 1
SEGMENTS = ["R12", "venue", "heavy_going"]
SCHEME = FoldScheme(
    unit="quarter",
    test_length=1,
    step=1,
    min_train=2,
    window="expanding",
    embargo_days=7,
    anchor_month=1,
)
BOOTSTRAP = {
    "cluster_unit": "meeting",
    "sided": "one-sided",
    "level": 0.95,
    "n_boot": 200,
    "seed": 276,
}
SE_RTOL = 0.02  # a correct sandwich is within 2% of the plain one (G / (G - 1) and the like)

# ---------------------------------------------------------------------------------------------
# A synthetic development frame: meetings whose winners follow their own beta, and the reference
# ---------------------------------------------------------------------------------------------


def study_frame(seed=2760):
    """300 meetings of 12 races (two venues a race day, a race day a week from 2019-01-05), runners
    6..18 a race, spreads varying by race so the entropy bins fill, every 25th race's favourite
    quoted exactly 1.1 (floor-bound), going HEAVY at one meeting in five. Each meeting draws its
    winners at its own beta_m = 1 + N(0, SIGMA^2): the races of a meeting are correlated."""
    rng = np.random.default_rng(seed)
    n_races, n_max = N_MEETINGS * RACES_PER_MEETING, 18
    size = rng.integers(6, n_max + 1, size=n_races)
    absent = np.arange(n_max)[None, :] >= size[:, None]
    spread = rng.uniform(0.4, 2.0, size=n_races)
    s = np.where(absent, 0.0, np.exp(rng.standard_normal((n_races, n_max)) * spread[:, None]))
    q = s / s.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        odds = np.maximum(np.round(TAKE / q, 1), 1.0)
    rows = np.arange(0, n_races, 25)
    odds[rows, np.where(absent, np.inf, odds).argmin(axis=1)[rows]] = FLOOR
    odds[absent] = np.nan

    beta_m = 1.0 + SIGMA * rng.standard_normal(N_MEETINGS)
    inv = 1.0 / odds
    share = inv / np.nansum(inv, axis=1, keepdims=True)
    p = share ** np.repeat(beta_m, RACES_PER_MEETING)[:, None]
    p = np.nan_to_num(p / np.nansum(p, axis=1, keepdims=True))
    winner = np.minimum(
        (np.cumsum(p, axis=1) < rng.random(n_races)[:, None]).sum(axis=1), n_max - 1
    )

    k = np.arange(n_races)
    meeting = k // RACES_PER_MEETING
    day = pd.Timestamp("2019-01-05") + pd.to_timedelta(7 * (meeting // 2), unit="D")
    venue = np.where(meeting % 2 == 0, "TOK", "KYO")
    heavy = rng.random(N_MEETINGS) < 0.2
    race = pd.DataFrame(
        {
            "race_id": [f"r{i:05d}" for i in k],
            "race_date": day,
            "meeting_id": [f"{d:%Y-%m-%d}:{v}" for d, v in zip(day, venue, strict=True)],
            "venue": venue,
            "race_no": k % RACES_PER_MEETING + 1,
            "going": np.where(heavy[meeting], "HEAVY", "GOOD"),
            "field_size": size,
        }
    )
    race["race_class"] = np.where(
        race["race_no"].eq(1),
        "DEBUT",
        np.where(race["race_no"].eq(11) & (meeting % 5 == 0), "G1", "OPEN"),
    )
    race["surface"] = "turf"
    race["weight_type"] = np.where(race["race_no"] % 3 == 0, "handicap", "age")
    long = pd.DataFrame(
        {
            "race_id": np.repeat(race["race_id"].to_numpy(), n_max),
            "odds": odds.ravel(),
            "won": (np.arange(n_max)[None, :] == winner[:, None]).ravel(),
        }
    )
    long = long[~np.isnan(long["odds"])]
    return long.merge(race, on="race_id", validate="many_to_one").reset_index(drop=True)


def ref_entropy(odds):
    """The normalised Shannon entropy of one race's normalised shares, per race, NaN under two."""
    inv = 1.0 / np.asarray(odds, dtype=float)
    q = inv / inv.sum()
    return float(-(q * np.log(q)).sum() / math.log(len(q))) if len(q) >= 2 else float("nan")


def ref_fit(rows):
    """(beta, wald se, meeting-clustered se, races, meetings) of the conditional logit on ln(1/odds)
    over `rows`' races with a priced winner, from first principles; clusters are `meeting_id`."""
    d = rows[rows["race_id"].map(rows.groupby("race_id")["won"].any())]
    codes, races = pd.factorize(d["race_id"])
    x = -np.log(d["odds"].to_numpy(dtype=float))
    won = d["won"].to_numpy(dtype=bool)
    n_won = np.bincount(codes, weights=won, minlength=len(races))
    beta = 1.0
    for _ in range(100):
        e = np.exp(beta * x)
        total = np.bincount(codes, weights=e)
        mean = np.bincount(codes, weights=e * x) / total
        var = np.bincount(codes, weights=e * x * x) / total - mean**2
        score = np.bincount(codes, weights=won * x, minlength=len(races)) - n_won * mean
        step = score.sum() / (n_won * var).sum()
        beta += step
        if abs(step) < 1e-12:
            break
    e = np.exp(beta * x)
    total = np.bincount(codes, weights=e)
    mean = np.bincount(codes, weights=e * x) / total
    var = np.bincount(codes, weights=e * x * x) / total - mean**2
    score = np.bincount(codes, weights=won * x, minlength=len(races)) - n_won * mean
    information = float((n_won * var).sum())
    meeting = pd.Series(d["meeting_id"].to_numpy()).groupby(codes).first()
    by_meeting = pd.Series(score).groupby(meeting.to_numpy()).sum()
    se_cluster = math.sqrt(float((by_meeting**2).sum())) / information
    return beta, information**-0.5, se_cluster, len(races), len(by_meeting)


@pytest.fixture(scope="module")
def frame():
    return study_frame()


@pytest.fixture(scope="module")
def cut_points(frame):
    per_race = frame.groupby("race_id")["odds"].apply(ref_entropy).dropna()
    return [float(c) for c in np.quantile(per_race, [0.2, 0.4, 0.6, 0.8])]


@pytest.fixture(scope="module")
def spec(cut_points):
    owner = {**calibration.owner_inputs(), "segments": SEGMENTS, "entropy_cut_points": cut_points}
    return build_spec(owner, SCHEME, BOOTSTRAP, FLOOR)


@pytest.fixture(scope="module")
def study(frame, spec):
    return compute_study(frame, spec)


def table(study, key):
    return pd.DataFrame(study.tables[key])


def floor_races(frame):
    flagged = frame["odds"] <= FLOOR + 1e-9
    return frame["race_id"].isin(frame.loc[flagged, "race_id"])


def subset_rows(frame, subset):
    return frame if subset == "all_races" else frame[~floor_races(frame)]


def assert_row_is(row, rows):
    """A table row against the reference fit of `rows`: Wald and clustered, with their intervals."""
    beta, se, se_cluster, n_races, n_meetings = ref_fit(rows)
    assert row["races"] == n_races
    assert row["meetings"] == n_meetings
    assert row["beta"] == pytest.approx(beta, rel=1e-6)
    assert row["se"] == pytest.approx(se, rel=1e-3)
    assert row["se cluster"] == pytest.approx(se_cluster, rel=SE_RTOL)
    assert row["lower 95%"] == pytest.approx(row["beta"] - WALD_Z * row["se"], rel=1e-9)
    assert row["upper 95%"] == pytest.approx(row["beta"] + WALD_Z * row["se"], rel=1e-9)
    assert row["lower 95% cluster"] == pytest.approx(
        row["beta"] - WALD_Z * row["se cluster"], rel=1e-9
    )
    assert row["upper 95% cluster"] == pytest.approx(
        row["beta"] + WALD_Z * row["se cluster"], rel=1e-9
    )


# ---------------------------------------------------------------------------------------------
# The fixture itself, the family and the spec
# ---------------------------------------------------------------------------------------------


def test_the_fixture_has_planted_within_meeting_correlation(frame):
    """Guard: the whole frame's clustered se is far above its Wald se, and the floor-bound races
    and the three years are there for the tables below to bite."""
    _, se, se_cluster, _, meetings = ref_fit(frame)
    assert meetings == N_MEETINGS
    assert se_cluster / se > 1.3
    assert 0 < floor_races(frame).sum() < len(frame) / 10
    assert sorted(frame["race_date"].dt.year.unique()) == [2019, 2020, 2021]


def test_the_family_is_new_and_the_spec_is_frozen_plain_json(spec, tmp_path, monkeypatch):
    assert isinstance(FAMILY_ID, str) and FAMILY_ID.strip()
    assert FAMILY_ID != calibration.FAMILY_ID
    assert TICKET == 276
    assert spec["family_id"] == FAMILY_ID
    json.dumps(spec)  # plain JSON, as the registry requires

    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    registry = default_registry("japan")
    digest = registry.freeze_family(FAMILY_ID, spec)
    assert registry.freeze_family(FAMILY_ID, spec) == digest  # an equal spec: a no-op
    assert registry.records(FAMILY_ID) == []  # frozen with no scored run

    # What is pre-registered is in the spec: another segment list, entropy cut points, scheme or
    # bootstrap is a different spec, which the registry refuses under this id.
    owner = {**calibration.owner_inputs(), "segments": SEGMENTS}
    cuts = [0.5, 0.6, 0.7, 0.8]
    others = [
        build_spec({**owner, "segments": ["R12"]}, SCHEME, BOOTSTRAP, FLOOR),
        build_spec({**owner, "entropy_cut_points": cuts}, SCHEME, BOOTSTRAP, FLOOR),
        build_spec(owner, replace(SCHEME, min_train=3), BOOTSTRAP, FLOOR),
        build_spec(owner, SCHEME, {**BOOTSTRAP, "n_boot": 300}, FLOOR),
    ]
    for other in others:
        assert other != spec
        with pytest.raises(FamilyAlreadyFrozen):
            registry.freeze_family(FAMILY_ID, other)


def test_77s_family_and_its_wald_tables_are_untouched(frame, tmp_path, monkeypatch):
    """The owner's call, not this ticket's: #77's frozen family id, its spec (bytes, so its hash)
    and its beta tables, which keep the Wald interval alone."""
    assert calibration.FAMILY_ID == "japan-crowd-calibration-final-win"
    spec77 = calibration.build_spec(
        calibration.owner_inputs(),
        load_country_scheme("japan"),
        calibration.bootstrap_settings(),
        calibration.win_floor(),
    )
    canonical = json.dumps(spec77, sort_keys=True, separators=(",", ":")).encode()
    assert (
        hashlib.sha256(canonical).hexdigest()
        == "4816f067ab5a50cb7dc07cd3dff85292532aa76a7b6b07d938f4c30dd5563c36"
    )

    owner = {**calibration.owner_inputs(), "segments": ["R12"]}
    study77 = calibration.compute_study(
        frame, calibration.build_spec(owner, SCHEME, BOOTSTRAP, FLOOR)
    )
    year_row = study77.tables["beta_by_year"][0]
    assert list(year_row) == ["subset", "year", "races", "beta", "se", "lower 95%", "upper 95%"]


# ---------------------------------------------------------------------------------------------
# The tables: beta by year, era and segment, with both intervals
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("subset", ["all_races", "no_floor_bound_races"])
def test_beta_by_year_reports_the_wald_and_the_clustered_interval(frame, study, subset):
    rows = table(study, "beta_by_year")
    rows = rows[rows["subset"] == subset]
    years = sorted(frame["race_date"].dt.year.unique())
    assert [int(y) for y in rows["year"]] == years

    part = subset_rows(frame, subset)
    for _, row in rows.iterrows():
        assert_row_is(row, part[part["race_date"].dt.year == int(row["year"])])
        # the correlation is planted inside every meeting: the clustered interval is the wider
        assert row["se cluster"] / row["se"] > 1.2


def test_the_floor_bound_subset_leaves_whole_races_out(frame, study):
    rows = table(study, "beta_by_year")
    everything = rows[rows["subset"] == "all_races"]["races"].sum()
    without = rows[rows["subset"] == "no_floor_bound_races"]["races"].sum()
    assert everything == frame["race_id"].nunique()
    assert without == frame.loc[~floor_races(frame), "race_id"].nunique() < everything


@pytest.mark.parametrize("subset", ["all_races", "no_floor_bound_races"])
def test_beta_by_era_is_one_row_over_every_race_of_the_subset(frame, study, subset):
    rows = table(study, "beta_by_era")
    rows = rows[rows["subset"] == subset]
    assert len(rows) == 1
    assert_row_is(rows.iloc[0], subset_rows(frame, subset))


def test_beta_by_segment_fits_each_side_with_both_intervals(frame, study):
    rows = table(study, "segments")
    assert sorted(rows["segment"].unique()) == sorted(SEGMENTS)

    inside = {
        "R12": frame["race_no"].eq(12),
        "heavy_going": frame["going"].eq("HEAVY"),
    }
    for _, row in rows.iterrows():
        name, value = row["segment"], row["value"]
        if name == "venue":
            part = frame[frame["venue"] == value]
        else:
            part = frame[inside[name] == (value == "in")]
            assert value in ("in", "out")
        assert_row_is(row, part)
    assert len(rows[rows["segment"] == "R12"]) == 2
    assert sorted(rows[rows["segment"] == "venue"]["value"]) == ["KYO", "TOK"]


def test_beta_by_entropy_bin_fits_each_bin_with_both_intervals(frame, study, cut_points):
    rows = table(study, "entropy")
    rows = rows[rows["beta"].notna()].reset_index(drop=True)
    per_race = frame.groupby("race_id")["odds"].apply(ref_entropy).dropna()
    bin_of = pd.Series(np.searchsorted(np.array(cut_points), per_race.to_numpy(), side="right"))
    bin_of.index = per_race.index
    assert len(rows) == 5
    for k, (_, row) in enumerate(rows.iterrows()):
        part = frame[frame["race_id"].isin(bin_of.index[bin_of == k])]
        assert_row_is(row, part)


# ---------------------------------------------------------------------------------------------
# Variants, registry metrics and the report
# ---------------------------------------------------------------------------------------------


def test_every_variant_is_one_scored_record_and_none_failed(study):
    names = [v.name for v in study.variants]
    assert len(names) == len(set(names)) > 0
    assert [v.name for v in study.variants if v.error] == []
    assert all(re.fullmatch(r"[ELP]\d+", v.hypothesis_id) for v in study.variants)
    assert all(v.metrics for v in study.variants)


@pytest.mark.parametrize("subset", ["all_races", "no_floor_bound_races"])
def test_the_registry_metrics_hold_both_intervals_by_year(frame, study, subset):
    metrics = {}
    for variant in study.variants:
        if subset in variant.name:
            metrics |= {m.metric: m for m in variant.metrics}
    part = subset_rows(frame, subset)
    for year in sorted(frame["race_date"].dt.year.unique()):
        beta, se, se_cluster, n_races, _ = ref_fit(part[part["race_date"].dt.year == year])
        wald, clustered = metrics[f"beta[year={year}]"], metrics[f"beta_clustered[year={year}]"]
        for metric, half in ((wald, se), (clustered, se_cluster)):
            assert metric.value == pytest.approx(beta, rel=1e-6)
            assert metric.n == n_races
            assert metric.lo == pytest.approx(beta - WALD_Z * half, rel=SE_RTOL, abs=1e-3)
            assert metric.hi == pytest.approx(beta + WALD_Z * half, rel=SE_RTOL, abs=1e-3)
        assert clustered.hi - clustered.lo > 1.2 * (wald.hi - wald.lo)


def test_the_report_shows_both_intervals(study, spec):
    report = render_report(study, spec, {"family": FAMILY_ID})

    assert isinstance(report, str)
    for column in (
        "lower 95%",
        "upper 95%",
        "se cluster",
        "lower 95% cluster",
        "upper 95% cluster",
    ):
        assert column in report
    row = table(study, "beta_by_year").iloc[0]
    for key in ("se", "se cluster", "beta"):
        assert f"{row[key]:.6g}" in report


# ---------------------------------------------------------------------------------------------
# The run: the family is frozen before any scoring, a store on a temporary data dir
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def data_dir(frame, tmp_path, monkeypatch):
    """A temporary data directory holding the synthetic frame as the store's two tables."""
    root = tmp_path / "data"
    store = root / "processed" / "jra"
    store.mkdir(parents=True)
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(root))
    day = frame["race_date"].dt.date
    runners = pd.DataFrame(
        {
            "race_id": frame["race_id"],
            "race_date": day,
            "win_odds": frame["odds"],
            "finish_pos": pd.array(np.where(frame["won"], 1, 2), dtype="Int64"),
            "is_scratched": False,
        }
    )
    races = frame.drop_duplicates("race_id")[
        ["race_id", "meeting_id", "venue", "race_no", "race_class", "surface", "going"]
        + ["weight_type", "field_size"]
    ].assign(race_date=lambda r: r["race_id"].map(dict(zip(frame["race_id"], day, strict=True))))
    races["status"] = "run"
    runners.to_parquet(store / "runners.parquet", index=False)
    races.to_parquet(store / "races.parquet", index=False)
    return root


def spy_on_store_reads(monkeypatch):
    """Record the family's frozen hash (None: not frozen) at every dataset the store opens."""
    seen = []
    real = pyarrow.dataset.dataset

    def dataset(*args, **kwargs):
        seen.append(default_registry("japan").frozen_spec_hash(FAMILY_ID))
        return real(*args, **kwargs)

    monkeypatch.setattr(pyarrow.dataset, "dataset", dataset)
    return seen


def test_run_freezes_the_family_before_any_row_is_read_and_records_every_variant(
    data_dir, tmp_path, monkeypatch
):
    seen = spy_on_store_reads(monkeypatch)

    result = run_study(tmp_path / "out")

    assert seen and all(frozen is not None for frozen in seen)  # frozen before every read
    assert result["family_id"] == FAMILY_ID
    registry = default_registry("japan")
    assert registry.frozen_spec_hash(FAMILY_ID) == result["spec_hash"]
    records = registry.records(FAMILY_ID)
    assert len(records) == result["records_appended"] > 0
    assert result["records_already_present"] == 0
    assert all(r.status == "scored" and r.ticket == TICKET for r in records)
    assert result["failed_variants"] == []
    # #77's family is not touched by this run
    assert registry.frozen_spec_hash(calibration.FAMILY_ID) is None
    assert registry.records(calibration.FAMILY_ID) == []

    assert Path(result["report_path"]).resolve().parent == (tmp_path / "out").resolve()
    written = Path(result["report_path"]).read_text(encoding="utf-8")
    for column in ("lower 95%", "se cluster", "lower 95% cluster", "upper 95% cluster"):
        assert column in written
    assert result["spec_hash"] in written


def test_a_second_run_appends_nothing_new(data_dir, tmp_path):
    first = run_study(tmp_path / "out")
    second = run_study(tmp_path / "out")

    assert second["spec_hash"] == first["spec_hash"]
    assert second["records_appended"] == 0
    assert second["records_already_present"] == first["records_appended"]
    assert len(default_registry("japan").records(FAMILY_ID)) == first["records_appended"]


def test_run_refuses_a_family_frozen_with_another_spec_before_reading_anything(
    data_dir, tmp_path, monkeypatch
):
    default_registry("japan").freeze_family(FAMILY_ID, {"family_id": FAMILY_ID, "another": "spec"})
    seen = spy_on_store_reads(monkeypatch)

    with pytest.raises(FamilyAlreadyFrozen):
        run_study(tmp_path / "out")

    assert seen == []  # no development row was read
    assert default_registry("japan").records(FAMILY_ID) == []
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_run_refuses_an_out_dir_inside_the_repository(data_dir):
    with pytest.raises(paths.DataDirInsideRepo):
        run_study(env.repo_root() / "reports-should-not-be-here")
