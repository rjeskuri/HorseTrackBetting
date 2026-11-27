"""D4's calibrated market scored on the crowd-calibration folds and tail table (ticket #275; #77).

Contract pinned here (`horsetrack.countries.japan.studies.calibrated_market`, a sibling of #77's
`studies.calibration` and #285's `studies.tail_beta`, which stay untouched):
- `FAMILY_ID`: a new registry family, neither #77's nor #285's.
- `build_spec(owner, scheme, bootstrap, floor) -> dict` (the signature of
  `calibration.build_spec`): plain JSON, frozen before any development row is read. It holds the
  spline's settings under `spec["spline"]`: `n_knots` (5, `fit_win_calibration`'s default) and
  `knot_quantile_range` (`models.v0.calibration.QUANTILE_RANGE` as a list), the tail threshold,
  the floor, the fold scheme and the bootstrap. Another threshold, scheme or bootstrap is another
  spec, which the registry refuses under this id (FamilyAlreadyFrozen).
- `compute_study(frame, spec) -> calibration.Study`, pure, on `development_frame`'s shape
  (race_id, race_date datetime64, meeting_id, odds, won). For each subset (`all_races`;
  `no_floor_bound_races`, races holding a runner at or under the floor left out of fitting and
  scoring alike, as in #77), on the spec's walk-forward folds of the subset's rows:
  - D4's spline (`models.v0.calibration.fit_win_calibration`, reused, with
    `spec["spline"]["n_knots"]`) fitted on each fold's training rows only and applied to the
    fold's test races (q the normalised share of the race's priced runners);
  - a `delta_ll_vs_normalised` metric for the spline: normalised's winner log loss minus the
    spline's per scored race (positive: the spline is better), its value the mean, `n` the
    scored races, `lo`/`hi` the meeting-clustered paired bootstrap of #69 with the spec's
    bootstrap settings (`core.eval.bootstrap.paired_bootstrap`, as `calibration._bounds`);
  - metrics named `mean_log_loss...` with `n` the scored races for the spline, normalised,
    power-beta (beta refitted on each fold's training rows, `calibration.fit_power_beta`) and
    Shin, all on the same scored races;
  - `study.tables["tail"]`: #77's layout, one row per de-vig and subset, keys in order subset,
    de-vig, tail runners, observed winners, expected winners, O / E, (O - E) per race, lower,
    upper; de-vig labels `normalised`, `power`, `shin` (as #77) and one more for the spline. Tail
    runners are scored runners priced strictly above `tail_min_odds`; (O - E) per race is the
    mean over scored races of the race's sum over tail runners of (won - p), with the clustered
    bootstrap bounds.
  Every variant is a `Variant` with a unique (hypothesis_id, name), the subset name in its name;
  none fails on a healthy frame; a variant that fails (a spline that cannot be fitted) is still a
  variant, with its error, under the same name.
- `run_study(out_dir=None) -> dict` with `family_id`, `spec_hash` and `failed_variants`: freezes
  the family before any outcome column is read; reads development rows through #77's
  `calibration.read_development` (holdout days dropped; the holdout is never opened); appends one
  registry record per variant; writes its report through `write_report` under `out_dir`.

Expected values come from independent references below: each fold's spline from the separately
tested `fit_win_calibration` on that fold's training rows, power-beta from #77's `fit_power_beta`,
the shares, losses, tail sums and bootstrap computed here. Every frame is synthetic and every RNG
seeded; HORSETRACK_DATA_DIR points under tmp in the run tests: no real data and no holdout.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import dataclasses
import json
import math
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import FamilyAlreadyFrozen, default_registry
from horsetrack.countries.japan.models.v0 import calibration as d4
from horsetrack.countries.japan.models.v0 import store as v0_store
from horsetrack.countries.japan.studies import calibration, tail_beta
from horsetrack.countries.japan.studies import calibrated_market
from horsetrack.countries.japan.studies.calibrated_market import (
    FAMILY_ID,
    build_spec,
    compute_study,
    run_study,
)

NAN = float("nan")
TAKE = 0.8
THRESHOLD = 51.0
FLOOR = 1.1
N_KNOTS = 5
DEVIGS = ("spline", "normalised", "power", "shin")  # reference names; the spline's label is free
TAIL_KEYS = [
    "subset",
    "de-vig",
    "tail runners",
    "observed winners",
    "expected winners",
    "O / E",
    "(O - E) per race",
    "lower",
    "upper",
]
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
    "seed": 275,
}

# ---------------------------------------------------------------------------------------------
# Synthetic markets: a planted over-priced tail (winners drawn from a curve with an extra slope
# in ln(1/O) above 51.0, continuous at 51.0)
# ---------------------------------------------------------------------------------------------


def market_odds(rng, n_races, n_max, n_min, spread, floor_every=0):
    """(n_races, n_max) final Win odds rounded to 0.1 (floor 1.0), NaN for absent runners. Odds
    near 51.0 are quoted exactly 51.0 (the body, not the tail); with `floor_every`, every
    that-many-th race's favourite is quoted 1.1 (floor-bound)."""
    strength = rng.standard_normal((n_races, n_max)) * spread
    size = rng.integers(n_min, n_max + 1, size=n_races)
    absent = np.arange(n_max)[None, :] >= size[:, None]
    s = np.where(absent, 0.0, np.exp(strength))
    q = s / s.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        odds = np.maximum(np.round(TAKE / q, 1), 1.0)
    odds[(odds >= 50.0) & (odds <= 52.0)] = THRESHOLD
    if floor_every:
        rows = np.arange(0, n_races, floor_every)
        fav = np.where(absent, np.inf, odds).argmin(axis=1)[rows]
        odds[rows, fav] = FLOOR
    odds[absent] = NAN
    return odds


def draw_winners(rng, odds, body=1.0, kink=0.8):
    """p proportional to exp(body * x + kink * min(x - x0, 0)), x = ln(1/O), x0 = ln(1/51)."""
    x = -np.log(odds)
    z = body * x + kink * np.minimum(x + math.log(THRESHOLD), 0.0)
    e = np.exp(np.nan_to_num(z - np.nanmax(z, axis=1, keepdims=True), nan=-np.inf))
    p = e / e.sum(axis=1, keepdims=True)
    u = rng.random(len(p))[:, None]
    return np.minimum((np.cumsum(p, axis=1) < u).sum(axis=1), p.shape[1] - 1)


def study_frame(seed=2750, n_races=4_000):
    """Development-frame-shaped rows: 12 races a race day on two venues, a race day every other
    day from 2020-01-02 (to 2021-10), a floor-bound favourite in one race in twelve."""
    rng = np.random.default_rng(seed)
    odds = market_odds(rng, n_races, 14, 8, 1.4, floor_every=12)
    winners = draw_winners(rng, odds)
    n_max = odds.shape[1]
    k = np.repeat(np.arange(n_races), n_max)
    frame = pd.DataFrame(
        {
            "race_id": np.repeat([f"r{j:05d}" for j in range(n_races)], n_max),
            "k": k,
            "odds": odds.ravel(),
            "won": (np.arange(n_max)[None, :] == winners[:, None]).ravel(),
        }
    )
    frame = frame[~np.isnan(odds.ravel())].reset_index(drop=True)
    k = frame["k"].to_numpy()
    race_date = pd.Timestamp("2020-01-02") + pd.to_timedelta((k // 12) * 2, unit="D")
    venue = np.where((k // 6) % 2 == 0, "TOK", "KYO")
    frame["race_date"] = race_date
    frame["meeting_id"] = [f"{d:%Y-%m-%d}:{v}" for d, v in zip(race_date, venue, strict=True)]
    return frame[["race_id", "race_date", "meeting_id", "odds", "won"]]


def subset_rows(frame, subset):
    if subset == "all_races":
        return frame.reset_index(drop=True)
    floor_race = (
        (frame["odds"] <= FLOOR + 1e-9).groupby(frame["race_id"].to_numpy()).transform("any")
    )
    return frame.loc[~floor_race.to_numpy()].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# The reference: fold-wise fits, shares, losses, tail sums and bounds computed here
# ---------------------------------------------------------------------------------------------


def shares(z, race):
    """exp(z) normalised within each race."""
    z = pd.Series(np.asarray(z, dtype=float))
    e = np.exp(z - z.groupby(race).transform("max"))
    return (e / e.groupby(race).transform("sum")).to_numpy()


def bounds(per_race, clusters, boot=BOOTSTRAP):
    return paired_bootstrap(
        per_race, clusters, n_boot=boot["n_boot"], level=boot["level"], seed=boot["seed"]
    )


def reference(rows, n_knots=N_KNOTS, min_odds=THRESHOLD, scheme=SCHEME):
    """Out-of-fold probabilities of the four de-vigs on `rows`, the per-race losses, the spline's
    ΔLL against normalised with bounds, and the tail sums per de-vig."""
    folds = walk_forward(rows["race_date"], scheme=scheme)
    assert folds
    probs = {name: [] for name in DEVIGS}
    for fold in folds:
        train, test = rows.iloc[fold.train_idx], rows.iloc[fold.test_idx]
        race = test["race_id"].to_numpy()
        x = -np.log(test["odds"].to_numpy(dtype=float))
        q = shares(x, race)
        model = d4.fit_win_calibration(
            train.rename(columns={"odds": "win_odds"})[["race_id", "race_date", "win_odds", "won"]],
            n_knots=n_knots,
        )
        beta = calibration.fit_power_beta(train["odds"], train["race_id"], train["won"]).beta
        probs["spline"].append(model.predict_many(q, race))
        probs["normalised"].append(q)
        probs["power"].append(shares(beta * x, race))
        probs["shin"].append(calibration.devig_shin(test["odds"], test["race_id"]).to_numpy())
    scored = rows.iloc[np.concatenate([f.test_idx for f in folds])].reset_index(drop=True)
    probs = {name: np.concatenate(parts) for name, parts in probs.items()}
    race = scored["race_id"].to_numpy()
    won = scored["won"].to_numpy(dtype=bool)
    clusters = scored.groupby("race_id")["meeting_id"].first()
    loss = {
        name: pd.Series(-np.log(p[won]), index=race[won]).groupby(level=0).mean()
        for name, p in probs.items()
    }
    delta = loss["normalised"] - loss["spline"]
    tail = scored["odds"].to_numpy(dtype=float) > min_odds
    tails = {}
    for name, p in probs.items():
        gap = pd.Series(np.where(tail, won - p, 0.0)).groupby(race).sum()
        observed, expected = float(won[tail].sum()), float(p[tail].sum())
        tails[name] = {
            "tail runners": int(tail.sum()),
            "observed winners": observed,
            "expected winners": expected,
            "O / E": observed / expected,
            "gap": bounds(gap, clusters),
        }
    return {
        "n_scored": len(loss["normalised"]),
        "loss": loss,
        "delta": bounds(delta, clusters),
        "tails": tails,
    }


def in_sample_delta(rows, n_knots=N_KNOTS):
    """The spline fitted on every row, scored on the same races the folds score: what a leak of
    the test rows into the fit would give."""
    folds = walk_forward(rows["race_date"], scheme=SCHEME)
    scored = rows.iloc[np.concatenate([f.test_idx for f in folds])]
    model = d4.fit_win_calibration(rows.rename(columns={"odds": "win_odds"}), n_knots=n_knots)
    race = scored["race_id"].to_numpy()
    q = shares(-np.log(scored["odds"].to_numpy(dtype=float)), race)
    p = model.predict_many(q, race)
    won = scored["won"].to_numpy(dtype=bool)
    return float(np.mean(np.log(p[won]) - np.log(q[won])))


# ---------------------------------------------------------------------------------------------
# Shared fixtures: one frame, one healthy study, the references per subset
# ---------------------------------------------------------------------------------------------


def owner_inputs(**override):
    return {**calibration.owner_inputs(), **override}


def make_spec(**override):
    return build_spec(owner_inputs(**override), SCHEME, BOOTSTRAP, FLOOR)


def with_knots(spec, n_knots):
    spec = json.loads(json.dumps(spec))
    spec["spline"]["n_knots"] = n_knots
    return spec


@pytest.fixture(scope="module")
def frame():
    return study_frame()


@pytest.fixture(scope="module")
def study(frame):
    return compute_study(frame, make_spec())


@pytest.fixture(scope="module")
def refs(frame):
    return {s: reference(subset_rows(frame, s)) for s in ("all_races", "no_floor_bound_races")}


def metrics_of(study, subset):
    return [m for v in study.variants if subset in v.name for m in v.metrics]


# ---------------------------------------------------------------------------------------------
# The spec and the family
# ---------------------------------------------------------------------------------------------


def test_the_family_is_new_and_the_spec_is_frozen_plain_json(tmp_path, monkeypatch):
    spec = make_spec()
    assert spec["family_id"] == FAMILY_ID
    assert isinstance(FAMILY_ID, str) and FAMILY_ID
    assert FAMILY_ID not in (calibration.FAMILY_ID, tail_beta.FAMILY_ID)
    json.dumps(spec)  # plain JSON, as the registry requires
    assert spec["spline"]["n_knots"] == N_KNOTS
    assert list(spec["spline"]["knot_quantile_range"]) == list(d4.QUANTILE_RANGE)

    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    registry = default_registry("japan")
    digest = registry.freeze_family(FAMILY_ID, spec)
    assert registry.freeze_family(FAMILY_ID, make_spec()) == digest  # an equal spec: a no-op
    assert registry.records(FAMILY_ID) == []

    # Everything pre-registered is in the spec: another threshold, scheme, bootstrap or floor is a
    # different family, which the registry refuses under this id.
    others = (
        make_spec(tail_min_odds=40.0),
        build_spec(owner_inputs(), replace(SCHEME, min_train=3), BOOTSTRAP, FLOOR),
        build_spec(owner_inputs(), SCHEME, {**BOOTSTRAP, "n_boot": 300}, FLOOR),
        build_spec(owner_inputs(), SCHEME, BOOTSTRAP, 1.2),
    )
    for other in others:
        assert other != spec
        with pytest.raises(FamilyAlreadyFrozen):
            registry.freeze_family(FAMILY_ID, other)


# ---------------------------------------------------------------------------------------------
# The pure study: the spline out of fold, ΔLL, the comparison, the tail table
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("subset", ["all_races", "no_floor_bound_races"])
def test_the_splines_delta_ll_vs_normalised_is_out_of_fold_with_meeting_clustered_bounds(
    frame, study, refs, subset
):
    ref = refs[subset]
    rows = subset_rows(frame, subset)
    n_scored = ref["n_scored"]
    assert 500 < n_scored < rows["race_id"].nunique()  # the early quarters only train
    # A fit that saw its test races would score another number: the check below can tell.
    assert abs(in_sample_delta(rows) - ref["delta"].mean) > 1e-4

    found = [
        m
        for m in metrics_of(study, subset)
        if m.metric.startswith("delta_ll_vs_normalised") and m.n == n_scored
    ]
    match = [m for m in found if abs(m.value - ref["delta"].mean) < 1e-6]
    assert match, (subset, ref["delta"].mean, [(m.metric, m.value) for m in found])
    m = match[0]
    assert m.lo == pytest.approx(ref["delta"].lo, abs=1e-9)
    assert m.hi == pytest.approx(ref["delta"].hi, abs=1e-9)
    assert m.lo < m.value < m.hi


def test_the_subsets_score_different_races(refs):
    """Floor-bound races leave fitting and scoring alike: fewer scored races, another ΔLL."""
    all_races, no_floor = refs["all_races"], refs["no_floor_bound_races"]
    assert no_floor["n_scored"] < all_races["n_scored"]
    assert abs(no_floor["delta"].mean - all_races["delta"].mean) > 1e-6


@pytest.mark.parametrize("subset", ["all_races", "no_floor_bound_races"])
def test_the_spline_and_each_devig_have_a_mean_log_loss_on_the_same_races(study, refs, subset):
    ref = refs[subset]
    means = {name: float(ref["loss"][name].mean()) for name in DEVIGS}
    values = sorted(means.values())
    assert min(np.diff(values)) > 1e-7  # four different numbers, so each is identified
    found = [
        m
        for m in metrics_of(study, subset)
        if m.metric.startswith("mean_log_loss") and m.n == ref["n_scored"]
    ]
    for name, expected in means.items():
        assert any(abs(m.value - expected) < 1e-6 for m in found), (
            subset,
            name,
            expected,
            [(m.metric, m.value) for m in found],
        )


@pytest.mark.parametrize("subset", ["all_races", "no_floor_bound_races"])
def test_the_tail_table_has_77s_layout_with_the_spline_beside_the_devigs(study, refs, subset):
    ref = refs[subset]["tails"]
    rows = [r for r in study.tables.get("tail", []) if r.get("subset") == subset]
    assert len(rows) == 4, rows
    for row in rows:
        assert list(row) == TAIL_KEYS
    labels = [r["de-vig"] for r in rows]
    assert len(set(labels)) == 4
    assert {"normalised", "power", "shin"} <= set(labels)
    by_name = {
        (r["de-vig"] if r["de-vig"] in ("normalised", "power", "shin") else "spline"): r
        for r in rows
    }
    for name, expected in ref.items():
        row = by_name[name]
        assert row["tail runners"] == expected["tail runners"], name
        assert row["observed winners"] == pytest.approx(expected["observed winners"], abs=1e-9)
        assert row["expected winners"] == pytest.approx(expected["expected winners"], abs=1e-6)
        assert row["O / E"] == pytest.approx(expected["O / E"], rel=1e-6)
        assert row["(O - E) per race"] == pytest.approx(expected["gap"].mean, abs=1e-9)
        assert row["lower"] == pytest.approx(expected["gap"].lo, abs=1e-9)
        assert row["upper"] == pytest.approx(expected["gap"].hi, abs=1e-9)
    # The tail is strictly above the threshold: runners quoted exactly 51.0 are not in it.
    assert expected["tail runners"] > 0
    # The planted tail is over-priced: normalised expects more tail winners than came in.
    assert ref["normalised"]["O / E"] < 0.8
    assert abs(ref["spline"]["O / E"] - 1) < abs(ref["normalised"]["O / E"] - 1)


def test_runners_at_exactly_the_threshold_are_in_the_fixture(frame):
    """Guard: the strict-inequality check above has runners at exactly 51.0 to bite on."""
    assert (frame["odds"] == THRESHOLD).sum() > 200


def test_the_specs_knots_drive_the_spline(frame, refs):
    rows = subset_rows(frame, "all_races")
    ref4 = reference(rows, n_knots=4)
    assert abs(ref4["delta"].mean - refs["all_races"]["delta"].mean) > 1e-6

    study4 = compute_study(frame, with_knots(make_spec(), 4))

    found = [
        m
        for m in metrics_of(study4, "all_races")
        if m.metric.startswith("delta_ll_vs_normalised") and m.n == ref4["n_scored"]
    ]
    assert any(abs(m.value - ref4["delta"].mean) < 1e-6 for m in found), [
        (m.metric, m.value) for m in found
    ]


def test_every_variant_is_one_scored_record_and_none_failed(study):
    names = [(v.hypothesis_id, v.name) for v in study.variants]
    assert len(names) == len(set(names))  # one record per variant
    assert any("all_races" in n for _, n in names)
    assert any("no_floor_bound_races" in n for _, n in names)
    for v in study.variants:
        assert v.error is None, (v.name, v.error)
        assert v.metrics, v.name
        assert v.fold_scheme
        assert v.components
    # Out-of-fold scoring runs on the spec's folds: the fold scheme is in a record's identity.
    scored = [v for v in study.variants if any(m.metric.startswith("delta_ll") for m in v.metrics)]
    assert scored
    assert all(
        json.dumps(dataclasses.asdict(SCHEME), sort_keys=True) in v.fold_scheme for v in scored
    )


def test_a_spline_that_cannot_be_fitted_is_still_a_variant(frame, study):
    """n_knots = 1 is refused by fit_win_calibration: the study still returns, the failure is a
    variant with its error, and no variant goes missing."""
    broken = compute_study(frame, with_knots(make_spec(), 1))

    assert {v.name for v in broken.variants} == {v.name for v in study.variants}
    failed = [v for v in broken.variants if v.error]
    assert failed
    assert all(isinstance(v.error, str) and v.error for v in failed)
    assert any("all_races" in v.name for v in failed)
    assert any("no_floor_bound_races" in v.name for v in failed)


# ---------------------------------------------------------------------------------------------
# The runner: freeze first, development rows only, one record per variant
# ---------------------------------------------------------------------------------------------

RUNNER_COLUMNS = ["race_id", "race_date", "win_odds", "finish_pos", "is_scratched"]
OUTCOME_COLUMNS = {"finish_pos"}


def store_tables(seed=2751):
    """Synthetic store runners and races: a race day every 14 days from 2014-06-07 to mid-2020
    (Japan's scheme trains to 2019 and scores 2020's first two quarters), then every weekend day
    from 2026-11-28 to 2026-12-27, about half of them holdout days. Two venues, 4 races each."""
    days = []
    d = date(2014, 6, 7)
    while d <= date(2020, 6, 27):
        days.append(d)
        d += timedelta(days=14)
    days += [
        date(2026, 11, 28) + timedelta(days=i)
        for i in range(30)
        if (date(2026, 11, 28) + timedelta(days=i)).weekday() >= 5
    ]
    races = [(day, venue, no) for day in days for venue in ("TOK", "KYO") for no in range(1, 5)]
    rng = np.random.default_rng(seed)
    n_max = 12
    odds = market_odds(rng, len(races), n_max, 8, 1.4, floor_every=10)
    winners = draw_winners(rng, odds)
    runner_rows, race_rows = [], []
    for k, (day, venue, no) in enumerate(races):
        race_id = f"jra:{day.isoformat()}:{venue}:{no:02d}"
        priced = np.flatnonzero(~np.isnan(odds[k]))
        order = [winners[k]] + [j for j in priced if j != winners[k]]
        for pos, j in enumerate(order, start=1):
            runner_rows.append((race_id, pd.Timestamp(day), float(odds[k, j]), pos, False))
        if k % 20 == 0:
            runner_rows.append((race_id, pd.Timestamp(day), NAN, None, True))
        race_rows.append(
            {
                "race_id": race_id,
                "race_date": pd.Timestamp(day),
                "meeting_id": f"{day.isoformat()}:{venue}",
                "venue": venue,
                "race_no": no,
                "race_class": "C1",
                "surface": "turf",
                "going": "good",
                "weight_type": "fixed",
                "field_size": len(priced),
                "status": "run",
            }
        )
    runners = pd.DataFrame(runner_rows, columns=RUNNER_COLUMNS)
    runners["finish_pos"] = runners["finish_pos"].astype("Int64")
    return runners, pd.DataFrame(race_rows)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A tmp data dir holding the synthetic store, and a spy on every store read that records the
    columns read and whether the family was frozen at that moment."""
    data = tmp_path / "data"
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(data))
    runners, races = store_tables()
    processed = data / "processed" / "jra"
    processed.mkdir(parents=True)
    tables = {"runners": runners, "races": races}
    for name, table in tables.items():
        path = processed / f"{name}.parquet"
        pq.write_table(pa.Table.from_pandas(table, preserve_index=False), path)
    calls = []

    def fake_read(name, columns=None, start=None, end=None, touches_price=False, **kwargs):
        frozen = default_registry("japan").frozen_spec_hash(FAMILY_ID) is not None
        calls.append({"name": name, "columns": list(columns or []), "frozen": frozen})
        table = tables[name]
        day = table["race_date"].dt.date
        keep = np.ones(len(table), dtype=bool)
        if start is not None:
            keep &= (day >= start).to_numpy()
        if end is not None:
            keep &= (day <= end).to_numpy()
        part = table.loc[keep, list(columns) if columns else list(table.columns)]
        return pa.Table.from_pandas(part.reset_index(drop=True), preserve_index=False)

    monkeypatch.setattr(calibration, "read_store_table", fake_read)
    monkeypatch.setattr(v0_store, "read_store_table", fake_read)
    monkeypatch.setattr(calibrated_market, "read_store_table", fake_read, raising=False)
    holdout = runners["race_date"].dt.date.map(lambda d: is_holdout_day("japan", d)).to_numpy()
    return {
        "runners": runners,
        "races": races,
        "holdout": holdout,
        "calls": calls,
        "out": tmp_path / "out",
    }


def scored_races(runners, races):
    """Races Japan's scheme scores among `runners` (through #77's development_frame)."""
    frame, _ = calibration.development_frame(runners, races)
    folds = walk_forward(frame["race_date"], scheme=load_country_scheme("japan"))
    return frame.iloc[np.concatenate([f.test_idx for f in folds])]["race_id"].nunique()


def test_run_study_freezes_first_scores_development_rows_only_and_records_every_variant(store):
    runners, races, holdout = store["runners"], store["races"], store["holdout"]
    assert 0 < holdout.sum() < (runners["race_date"] >= pd.Timestamp("2026-11-28")).sum()
    held_races = set(runners.loc[holdout, "race_id"])
    dev_runners = runners.loc[~holdout]
    dev_races = races.loc[~races["race_id"].isin(held_races)]
    n_dev = scored_races(dev_runners, dev_races)
    assert scored_races(runners, races) > n_dev  # a holdout leak would change the count

    result = run_study(store["out"])

    # The family was frozen before any outcome was read.
    outcome_reads = [c for c in store["calls"] if OUTCOME_COLUMNS & set(c["columns"])]
    assert outcome_reads
    assert all(c["frozen"] for c in outcome_reads)

    registry = default_registry("japan")
    assert result["family_id"] == FAMILY_ID
    assert result["spec_hash"] == registry.frozen_spec_hash(FAMILY_ID)
    assert result["failed_variants"] == []
    records = registry.records(FAMILY_ID)
    assert records
    identities = [(r.hypothesis_id, r.variant) for r in records]
    assert len(identities) == len(set(identities))
    assert all(r.status == "scored" for r in records)
    assert any("all_races" in r.variant for r in records)
    assert any("no_floor_bound_races" in r.variant for r in records)

    # Only development races were scored: the ΔLL counts the development races the folds score.
    delta = [
        m
        for r in records
        if "all_races" in r.variant
        for m in r.metrics
        if m.metric.startswith("delta_ll_vs_normalised")
    ]
    assert delta
    assert all(m.n == n_dev for m in delta), [(m.metric, m.n) for m in delta]

    # One record per variant of the pure study on the same development rows (variant names do
    # not depend on the bootstrap's settings).
    frame, _ = calibration.development_frame(dev_runners, dev_races)
    spec = build_spec(
        calibration.owner_inputs(), load_country_scheme("japan"), BOOTSTRAP, calibration.win_floor()
    )
    variants = compute_study(frame, spec).variants
    assert sorted(identities) == sorted((v.hypothesis_id, v.name) for v in variants)
    report = list(store["out"].glob("*.md"))
    assert len(report) == 1
    assert report[0].stat().st_size > 0


def test_run_study_stops_before_reading_outcomes_when_the_family_is_frozen_otherwise(store):
    default_registry("japan").freeze_family(FAMILY_ID, {"family_id": FAMILY_ID, "other": 1})

    with pytest.raises(FamilyAlreadyFrozen):
        run_study(store["out"])

    assert not [c for c in store["calls"] if OUTCOME_COLUMNS & set(c["columns"])]
    assert default_registry("japan").records(FAMILY_ID) == []
