"""The C2 ability study's review fixes, round 1 (ticket #92, epic #73; `studies.ability`).

Contract pinned here (the coder implements what these tests call, in `studies/ability.py`):

- C4 marginal. `market_only_oof(rows, folds, market="ln_q") -> pd.Series`: the market-only
  stage-2 meta, u = beta * ln q (beta unbounded, no member), its beta fitted by conditional logit
  on each fold's training rows of `rows`, P(win) on the fold's test rows; NaN on rows no fold
  tests; on rows' index. `c4_marginal(p_member, rows, folds, scored, spec) -> (metrics, tables)`:
  the member's C4 marginal, the paired per-race winner ΔLL (positive: the member's meta better) of
  `p_member` (the stage-2 meta with the member, out of fold on the same `rows` and `folds`)
  against `market_only_oof(rows, folds)` over `scored`'s races: a `delta_ll` metric per scored
  fold (its test_start, the fold's race mean) and one on `all_folds` (n = the scored races) with
  D7's meeting-bootstrap bounds at `spec["bootstrap"]`, the only part of `spec` read. `PLANNED`
  names the market-only meta ("E33", "stage2:q:market_only") and the marginal ("E33",
  "marginal:stage2:ability_vs_market_only"); the D4 comparison ("E33", "stage2:q:ability") stays.
- E34 rates. `connection_features(base, rows)` rates each row over #129's window
  (`handicap.factors._connection_rates` at cut = the row's day - 1): the race days in
  [day - 365, day - 1], across a gap between meetings too; prior starts count every earlier race
  day (not windowed), never the row's own day. A null entity key is not pinned.
- Committed index. `write_committed_summary(registry)` with a registry that holds only this
  study's family keeps every family the committed `README.md` listed, leaves every other family's
  file as it was and writes this study's file; a family neither committed nor this study's never
  enters.
- Artifacts. `artifact_stem(spec_hash, builds, trial) -> str`: a safe file stem, a function of
  its inputs alone (builds compared as a mapping), different for another spec hash, any change to
  a build (its refresh part after '@' included), an added or a removed build, and the trial flag;
  a trial's ends "-trial". `_record(variant, git_sha, builds, spec_hash, trial)` names that stem
  in the record, and two records that differ only by a refresh of a build are two runs of one
  variant (`Registry.family_size`, #74's rule). `run_study` names the report, its JSON and the
  export by that stem, so a rerun on refreshed data leaves the earlier files as they were.

Every frame is synthetic and every RNG seeded. Expected values come from first principles (a
Newton fit of the power of q, hand-counted rates), from #129's own `_connection_rates` on a fake
store read, or from existing library code (`models.meta`, `core.eval.bootstrap`), never from the
code under test. `run_study`'s store reads and its compute are replaced (the store is external
and the compute is pinned elsewhere); every path is under `tmp_path`, and an autouse guard fails
a test that changes the committed registry summary or a holdout log.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from horsetrack.core.eval import registry as registry_mod
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import FoldScheme, walk_forward
from horsetrack.core.eval.registry import FoldMetric, Registry, RunRecord, write_summary
from horsetrack.countries.japan.handicap import factors
from horsetrack.countries.japan.studies import ability
from horsetrack.countries.japan.studies.calibration import Study, Variant
from horsetrack.countries.japan.studies.meta import Scored, _delta, _meta_oof

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
BOOT = {"cluster_unit": "meeting", "sided": "one-sided", "level": 0.95, "n_boot": 300, "seed": 7}
SPEC = {"bootstrap": BOOT}
MARKET_ONLY = ("E33", "stage2:q:market_only")
MARGINAL = ("E33", "marginal:stage2:ability_vs_market_only")
GIT_SHA = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c"
SAFE_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """The data directory is under tmp_path; no test changes the committed summary or a log."""
    monkeypatch.setenv(DATA_DIR_VAR, str(tmp_path / "data"))

    def snapshot() -> dict[Path, bytes]:
        files = [*COMMITTED_SUMMARY.rglob("*"), *REAL_COUNTRIES.glob("*/holdout_log.jsonl")]
        return {p: p.read_bytes() for p in files if p.is_file()}

    before = snapshot()
    yield
    assert snapshot() == before, "a test changed the committed registry summary or a holdout log"


def softmax_within(u: np.ndarray, codes: np.ndarray) -> np.ndarray:
    top = pd.Series(u).groupby(codes).transform("max").to_numpy()
    e = np.exp(u - top)
    return e / pd.Series(e).groupby(codes).transform("sum").to_numpy()


# --------------------------------------------------------------------------- C4 marginal


def level2_rows(seed: int, *, beta: tuple[float, float], alpha: float) -> pd.DataFrame:
    """Weekend meetings over 2020-2021 (two venues, three races each, 6-12 runners). The market q
    is a softmax of strengths; the member (`ability`) is an independent log-probability; the
    winner is drawn from softmax(b * ln q + alpha * member), b = beta[0] in 2020 and beta[1] in
    2021. p_cal, the D4 stand-in, is softmax(0.7 ln q): deliberately not the truth."""
    rng = np.random.default_rng(seed)
    race_ids, race_dates, meetings = [], [], []
    day = date(2020, 1, 4)
    while day <= date(2021, 12, 26):
        for d in (day, day + timedelta(days=1)):
            for venue in ("v1", "v2"):
                for r in range(3):
                    race_ids.append(f"{d:%Y%m%d}{venue}{r}")
                    race_dates.append(d)
                    meetings.append(f"{d:%Y%m%d}{venue}")
        day += timedelta(days=7)
    sizes = rng.integers(6, 13, size=len(race_ids))
    codes = np.repeat(np.arange(len(race_ids)), sizes)
    n = len(codes)
    ln_q = np.log(softmax_within(rng.normal(0.0, 1.0, n), codes))
    member = np.log(softmax_within(rng.normal(0.0, 1.0, n), codes))
    year = np.array([d.year for d in race_dates])[codes]
    u = np.where(year == 2020, beta[0], beta[1]) * ln_q + alpha * member
    noisy = pd.Series(u + rng.gumbel(size=n))
    won = np.zeros(n, dtype=bool)
    won[noisy.groupby(codes).idxmax().to_numpy()] = True
    return pd.DataFrame(
        {
            "race_id": np.array(race_ids, dtype=object)[codes],
            "race_date": pd.to_datetime(np.array(race_dates)[codes]),
            "horse_id": [f"h{i:06d}" for i in range(n)],
            "meeting_id": np.array(meetings, dtype=object)[codes],
            "won": won,
            "q": np.exp(ln_q),
            "ln_q": ln_q,
            ability.POOLED: member,
            "p_cal": softmax_within(0.7 * ln_q, codes),
        }
    )


def scored_folds(rows: pd.DataFrame) -> list:
    folds = [f for f in walk_forward(rows["race_date"], scheme=SCHEME) if len(f.test_idx)]
    assert len(folds) >= 4, "the synthetic span should give several scored quarters"
    return folds


def power_of_q_beta(ln_q: np.ndarray, race: np.ndarray, won: np.ndarray) -> float:
    """The maximum-likelihood beta of P(i wins) = q_i^beta / sum_j q_j^beta (one winner per race),
    by Newton's method on the concave log-likelihood."""
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


def expected_market_only(rows: pd.DataFrame, folds: list) -> pd.Series:
    """The market-only stage-2 meta out of fold, from first principles."""
    ln_q, race = rows["ln_q"].to_numpy(), rows["race_id"].to_numpy(dtype=object)
    won = rows["won"].to_numpy(dtype=bool)
    out = np.full(len(rows), np.nan)
    for fold in folds:
        tr, te = fold.train_idx, fold.test_idx
        beta = power_of_q_beta(ln_q[tr], race[tr], won[tr])
        out[te] = softmax_within(beta * ln_q[te], pd.factorize(race[te])[0])
    return pd.Series(out, index=rows.index)


def winner_log_loss(p: pd.Series, rows: pd.DataFrame) -> pd.Series:
    """-ln p of each race's winner (one per race), by race id."""
    winners = rows.loc[rows["won"]]
    return pd.Series(-np.log(p.reindex(winners.index).to_numpy()), index=winners["race_id"])


def all_folds(metrics, name: str = "delta_ll") -> FoldMetric:
    found = [m for m in metrics if m.fold == "all_folds" and m.metric == name]
    assert len(found) == 1, f"one all_folds {name} metric expected, got {found}"
    return found[0]


def test_planned_variants_name_the_market_only_meta_and_the_member_marginal():
    planned = list(ability.PLANNED)
    assert MARKET_ONLY in planned, "the market-only stage-2 meta is a planned variant"
    assert MARGINAL in planned, "the member's paired marginal against it is a planned variant"
    assert ("E33", f"stage2:q:{ability.POOLED}") in planned, "the D4 comparison stays"
    assert len(set(planned)) == len(planned)


@pytest.mark.parametrize("beta", [(1.0, 1.0), (0.7, 1.5), (1.6, 0.9)])
def test_market_only_meta_is_a_power_of_q_fitted_on_each_folds_training_rows(beta):
    rows = level2_rows(11, beta=beta, alpha=0.8)  # the member carries signal: it must be unused
    folds = scored_folds(rows)
    expected = expected_market_only(rows, folds)

    got = ability.market_only_oof(rows, folds)

    assert isinstance(got, pd.Series) and got.index.equals(rows.index)
    tested = expected.notna().to_numpy()
    assert (~tested).any(), "the first quarters are training rows only"
    assert got[~tested].isna().all(), "a row no fold tests has no prediction"
    np.testing.assert_allclose(got[tested].to_numpy(), expected[tested].to_numpy(), atol=1e-8)


@pytest.mark.parametrize(
    ("alpha", "beta"), [(0.0, (1.2, 1.2)), (0.6, (0.9, 1.4)), (1.2, (1.3, 1.0))]
)
def test_member_marginal_is_the_paired_oof_delta_against_the_market_only_meta(alpha, beta):
    rows = level2_rows(23, beta=beta, alpha=alpha)
    folds = scored_folds(rows)
    p_member, _ = _meta_oof(rows, folds, ability.STAGE2[ability.POOLED], "ln_q")
    scored = Scored.of(rows, folds)
    p_market = expected_market_only(rows, folds)
    per_race = winner_log_loss(p_market, scored.rows) - winner_log_loss(p_member, scored.rows)
    boot = paired_bootstrap(
        per_race, scored.clusters, n_boot=BOOT["n_boot"], level=BOOT["level"], seed=BOOT["seed"]
    )

    metrics, _ = ability.c4_marginal(p_member, rows, folds, scored, SPEC)

    total = all_folds(metrics)
    assert total.n == scored.n_races == len(per_race)
    assert total.value == pytest.approx(per_race.mean(), abs=1e-7)
    assert total.lo == pytest.approx(boot.lo, abs=1e-7)
    assert total.hi == pytest.approx(boot.hi, abs=1e-7)
    fold_of = scored.rows.groupby("race_id")["fold"].first()
    points = {m.fold: m for m in metrics if m.metric == "delta_ll" and m.fold != "all_folds"}
    assert set(points) == {f.test_start.isoformat() for f in folds}
    for label, part in per_race.groupby(fold_of.reindex(per_race.index).to_numpy()):
        assert points[label].value == pytest.approx(part.mean(), abs=1e-7)
        assert points[label].n == len(part)
    if alpha == 0.0:  # a member without signal earns ~nothing beside the same market term
        assert abs(total.value) < 1e-3


def test_a_zero_weight_member_has_no_marginal_whatever_the_d4_comparison_gives():
    rows = level2_rows(5, beta=(1.3, 1.3), alpha=0.0)
    folds = scored_folds(rows)
    scored = Scored.of(rows, folds)
    p_member = expected_market_only(rows, folds)  # the member's weight forced to 0
    d4 = all_folds(
        _delta(p_member, scored, SPEC, "zero weight")[0], "delta_ll_vs_calibrated_market"
    )
    assert d4.value > 0.005, "the fixture's D4 stand-in differs from the market-only meta"

    metrics, _ = ability.c4_marginal(p_member, rows, folds, scored, SPEC)

    total = all_folds(metrics)
    assert total.n == scored.n_races
    assert total.value == pytest.approx(0.0, abs=1e-7)
    assert total.lo == pytest.approx(0.0, abs=1e-7)
    assert total.hi == pytest.approx(0.0, abs=1e-7)


# --------------------------------------------------------------------------- E34 rate window


def store_frames(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(runners, races, sires) as the store gives them: weekend meetings from 2019-01-05 with a
    week off every sixth week, a summer break, a Monday meeting every ninth week, a few void
    races, scratched runners and starters without a placing."""
    rng = np.random.default_rng(seed)
    days, day, week = [], date(2019, 1, 5), 0
    while day <= date(2021, 6, 26):
        if week % 6 != 3 and not date(2020, 7, 18) <= day <= date(2020, 8, 9):
            days += [day, day + timedelta(days=1)]
            if week % 9 == 4:
                days.append(day + timedelta(days=2))
        day += timedelta(days=7)
        week += 1
    horses = [f"h{i:03d}" for i in range(160)]
    sires = pd.DataFrame({"horse_id": horses, "sire_id": [f"s{i % 7}" for i in range(160)]})
    runners, races = [], []
    for d in days:
        for r in range(3):
            race_id = f"{d:%Y%m%d}{r:02d}"
            status = "void" if rng.random() < 0.03 else "run"
            races.append((race_id, d, f"{d:%Y%m%d}v", "c1", status))
            n = int(rng.integers(5, 10))
            field = rng.choice(len(horses), size=n, replace=False)
            scratched = rng.random(n) < 0.06
            places = np.full(n, np.nan)
            if status == "run":
                places[~scratched] = rng.permutation(int((~scratched).sum())) + 1.0
                unplaced = ~scratched & (places > 1) & (rng.random(n) < 0.04)
                places[unplaced] = np.nan
            for h, s, p in zip(field, scratched, places, strict=True):
                runners.append(
                    (race_id, d, horses[h], p, bool(s))
                    + (f"j{rng.integers(14)}", f"t{rng.integers(9)}")
                )
    runners = pd.DataFrame(runners, columns=ability.RUNNER_COLUMNS)
    races = pd.DataFrame(races, columns=ability.RACE_COLUMNS)
    for frame in (runners, races):
        frame["race_date"] = pd.to_datetime(frame["race_date"])
    return runners, races, sires


def expected_features(base: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """First principles: per row, over the base runners on race days in [day - 365, day - 1],
    (wins + RATE_PRIOR_RUNS * p0) / (runs + RATE_PRIOR_RUNS), p0 the window's win rate; prior
    starts are the base starts on any earlier day."""
    prior = factors.RATE_PRIOR_RUNS
    out = pd.DataFrame(index=rows.index, columns=[*ability.RATES, *ability.PRIOR_STARTS])
    for d, part in rows.groupby("race_date"):
        on = base["race_date"]
        window = base.loc[(on >= d - pd.Timedelta(days=365)) & (on <= d - pd.Timedelta(days=1))]
        before = base.loc[on < d]
        p0 = window["won"].mean()
        for key, rate, starts in zip(
            ability.ENTITIES, ability.RATES, ability.PRIOR_STARTS, strict=True
        ):
            runs = part[key].map(window.groupby(key).size()).fillna(0.0)
            wins = part[key].map(window.groupby(key)["won"].sum()).fillna(0.0)
            out.loc[part.index, rate] = np.log((wins + prior * p0) / (runs + prior)).to_numpy()
            out.loc[part.index, starts] = part[key].map(before.groupby(key).size()).fillna(0.0)
    return out.astype(float)


def rates_of_129(runners: pd.DataFrame, rows: pd.DataFrame, monkeypatch) -> pd.DataFrame:
    """#129's own `_connection_rates` per race day of `rows` at cut = the day - 1, reading the
    store's runners (scratched ones included, as the store holds them) through a fake `_read`."""

    def fake_read(name, columns, start, end, *, where=None, touches_price=False):
        assert name == "runners" and where is None
        on = runners["race_date"]
        part = runners.loc[(on >= pd.Timestamp(start)) & (on <= pd.Timestamp(end)), list(columns)]
        return pl.from_pandas(part.reset_index(drop=True))

    monkeypatch.setattr(factors, "_read", fake_read)
    out = pd.DataFrame(index=rows.index, columns=["ln_jockey_rate", "ln_trainer_rate"], dtype=float)
    for d, part in rows.groupby("race_date"):
        card = pl.DataFrame(
            {
                "_i": part.index.to_numpy(),
                "jockey_key": part["jockey_key"].tolist(),
                "trainer_key": part["trainer_key"].tolist(),
            }
        )
        rated = factors._connection_rates(card, cut=d.date() - timedelta(days=1)).sort("_i")
        for who in ("jockey", "trainer"):
            out.loc[rated["_i"].to_numpy(), f"ln_{who}_rate"] = np.log(
                rated[f"{who}_rate"].to_numpy()
            )
    return out


def test_connection_rates_match_129_on_every_race_day(monkeypatch):
    runners, races, sires = store_frames(129)
    base, level1, _ = ability.level1_frames(runners, races, sires)
    rows = level1.loc[level1["race_date"] > level1["race_date"].min()]  # a window with runners
    expected = expected_features(base, rows)
    ours_129 = rates_of_129(runners, rows, monkeypatch)
    gaps = rows["race_date"].drop_duplicates().sort_values().diff().dt.days
    assert (gaps >= 6).sum() > 50 and (gaps == 1).sum() > 50, "Saturdays after a gap, Sundays"

    got = ability.connection_features(base, rows)

    assert got.index.equals(rows.index)
    for col in ("ln_jockey_rate", "ln_trainer_rate"):
        np.testing.assert_allclose(got[col].to_numpy(), ours_129[col].to_numpy(), atol=1e-12)
    for col in (*ability.RATES, *ability.PRIOR_STARTS):
        np.testing.assert_allclose(got[col].to_numpy(), expected[col].to_numpy(), atol=1e-12)


SLOTS = (  # the four runners of every race: jockey, trainer, sire
    ("jA", "tA", "sA"),
    ("jB", "tB", "sA"),
    ("jC", "tA", "sB"),
    ("jD", "tB", "sB"),
)
DAY = date(2022, 6, 4)  # a Saturday; the meeting before it is a week earlier
WINNER_SLOT = {400: 0, 366: 2, 365: 0, 100: 1, 7: 0, 0: 0}  # days before DAY -> the winner


def edge_base() -> pd.DataFrame:
    """One four-runner race on each of DAY - 400, - 366, - 365, - 100, - 7 and DAY."""
    rows = []
    for before, winner in WINNER_SLOT.items():
        d = DAY - timedelta(days=before)
        for slot, (jockey, trainer, sire) in enumerate(SLOTS):
            rows.append((f"r{before}", d, f"h{slot}", float(1 + (slot - winner) % 4)))
            rows[-1] += (jockey, trainer, sire)
    base = pd.DataFrame(
        rows,
        columns=["race_id", "race_date", "horse_id", "finish_pos", *ability.ENTITIES],
    )
    base["race_date"] = pd.to_datetime(base["race_date"])
    base["won"] = base["finish_pos"].eq(1)
    return base


# Hand-counted: the window of DAY - b holds the race days in [DAY - b - 365, DAY - b - 1]; every
# window below holds three races of four runners and three winners, so p0 = 1/4 and the prior
# adds 5 wins in 20 runs. A jockey rides 3 of the 12 window runs, a trainer or a sire 6.
EDGE = {  # days before DAY -> per slot: (jockey, trainer, sire) rate; prior starts (j, t, s)
    0: (  # window: DAY-365 (exactly 365 days: in), DAY-100, DAY-7; DAY-366 is out
        [(7 / 23, 7 / 26, 8 / 26), (6 / 23, 6 / 26, 8 / 26), (5 / 23, 7 / 26, 5 / 26)]
        + [(5 / 23, 6 / 26, 5 / 26)],
        (5, 10, 10),
    ),
    7: (  # window: DAY-366, DAY-365, DAY-100 (the meeting before is 93 days earlier)
        [(6 / 23, 7 / 26, 7 / 26), (6 / 23, 6 / 26, 7 / 26), (6 / 23, 7 / 26, 6 / 26)]
        + [(5 / 23, 6 / 26, 6 / 26)],
        (4, 8, 8),
    ),
    100: (  # window: DAY-400, DAY-366, DAY-365
        [(7 / 23, 8 / 26, 7 / 26), (5 / 23, 5 / 26, 7 / 26), (6 / 23, 8 / 26, 6 / 26)]
        + [(5 / 23, 5 / 26, 6 / 26)],
        (3, 6, 6),
    ),
}


def test_rate_window_is_the_365_days_before_the_race_day():
    base = edge_base()
    rows = base.loc[base["race_id"].isin([f"r{b}" for b in EDGE])]

    got = ability.connection_features(base, rows)

    assert got.index.equals(rows.index)
    for i, row in rows.iterrows():
        before, slot = int(row["race_id"][1:]), int(row["horse_id"][1:])
        rates, starts = EDGE[before][0][slot], EDGE[before][1]
        for col, rate in zip(ability.RATES, rates, strict=True):
            assert got.loc[i, col] == pytest.approx(np.log(rate), abs=1e-12), (before, slot, col)
        for col, n in zip(ability.PRIOR_STARTS, starts, strict=True):
            assert got.loc[i, col] == n, (before, slot, col)


# --------------------------------------------------------------------------- committed index


def make_record(family: str, variant: str, *, sources=("jra",), value: float = 0.1) -> RunRecord:
    return RunRecord(
        hypothesis_id="E33",
        family_id=family,
        variant=variant,
        ticket=ability.TICKET,
        git_sha=GIT_SHA,
        data_builds={"jra_runners": "processed/jra/runners.parquet@00aa11bb22cc33dd"},
        sources=tuple(sources),
        fold_scheme="walk_forward:synthetic",
        components={"study": "studies/ability.py"},
        params={"thin_starts": 30},
        metrics=(FoldMetric("all_folds", "delta_ll", value, None, None, 10),),
        status="scored",
    )


def test_committed_summary_keeps_every_committed_family_in_the_index(tmp_path, monkeypatch):
    summary = tmp_path / "docs" / "experiments" / "registry" / "japan"
    monkeypatch.setattr(ability, "summary_dir", lambda country: summary)
    monkeypatch.setattr(registry_mod, "summary_dir", lambda country: summary)
    others = ["japan-alpha-final-win", "japan-beta-final-win", "win5-gamma-v1"]
    committed = Registry(tmp_path / "data" / "committed.jsonl", holdout_log=tmp_path / "no.jsonl")
    for family in others:
        committed.append(make_record(family, f"{family}:variant", value=0.25))
    committed.append(make_record(ability.FAMILY_ID, "old:variant", sources=ability.SOURCES))
    write_summary(committed, summary)
    listed = summary.joinpath("README.md").read_text(encoding="utf-8")
    assert all(f"({family}.md)" in listed for family in [*others, ability.FAMILY_ID])
    before = {family: summary.joinpath(f"{family}.md").read_bytes() for family in others}

    local = Registry(tmp_path / "data" / "registry" / "japan.jsonl", holdout_log=tmp_path / "no")
    local.append(make_record(ability.FAMILY_ID, "fresh:variant", sources=ability.SOURCES))
    local.append(make_record("japan-uncommitted-final-win", "elsewhere"))

    ability.write_committed_summary(local)

    index = summary.joinpath("README.md").read_text(encoding="utf-8")
    for family in [*others, ability.FAMILY_ID]:
        assert f"({family}.md)" in index, f"the index dropped {family}"
    assert "japan-uncommitted-final-win" not in index
    assert not summary.joinpath("japan-uncommitted-final-win.md").exists()
    for family, data in before.items():
        assert summary.joinpath(f"{family}.md").read_bytes() == data, f"{family}.md changed"
    assert "fresh:variant" in summary.joinpath(f"{ability.FAMILY_ID}.md").read_text("utf-8")


# --------------------------------------------------------------------------- artifacts


SPEC_HASH = hashlib.sha256(b"synthetic spec").hexdigest()
BUILDS = {
    "jra_runners": "processed/jra/runners.parquet@1111aaaa2222bbbb",
    "jra_races": "processed/jra/races.parquet@3333cccc4444dddd",
    "jravan_horse_pedigree": "processed/jravan/horse_pedigree.parquet@5555eeee6666ffff",
}


def changed(**values) -> dict[str, str]:
    out = {k: v for k, v in (BUILDS | values).items() if v is not None}
    return out


def test_artifact_stem_names_the_spec_the_data_builds_and_the_trial_flag():
    cases = {
        "base": (SPEC_HASH, BUILDS, False),
        "trial": (SPEC_HASH, BUILDS, True),
        "another spec": (hashlib.sha256(b"another spec").hexdigest(), BUILDS, False),
        "runners refreshed": (
            SPEC_HASH,
            changed(jra_runners="processed/jra/runners.parquet@9999aaaa2222bbbb"),
            False,
        ),
        "races refreshed": (
            SPEC_HASH,
            changed(jra_races="processed/jra/races.parquet@3333cccc4444eeee"),
            False,
        ),
        "pedigree refreshed": (
            SPEC_HASH,
            changed(jravan_horse_pedigree="processed/jravan/horse_pedigree.parquet@0"),
            False,
        ),
        "another table": (
            SPEC_HASH,
            changed(jra_runners="processed/jra/runners_v2.parquet@1111aaaa2222bbbb"),
            False,
        ),
        "a build added": (SPEC_HASH, changed(jra_odds="processed/jra/odds.parquet@77"), False),
        "a build removed": (SPEC_HASH, changed(jra_races=None), False),
        "refreshed, trial": (
            SPEC_HASH,
            changed(jra_runners="processed/jra/runners.parquet@9999aaaa2222bbbb"),
            True,
        ),
    }
    stems = {name: ability.artifact_stem(*args) for name, args in cases.items()}

    assert len(set(stems.values())) == len(stems), stems
    for name, stem in stems.items():
        assert isinstance(stem, str) and SAFE_STEM.fullmatch(stem) and len(stem) <= 120, name
        assert stem.endswith("-trial") == cases[name][2], name
    reordered = dict(reversed(list(BUILDS.items())))
    assert ability.artifact_stem(SPEC_HASH, reordered, False) == stems["base"]
    assert ability.artifact_stem(SPEC_HASH, dict(BUILDS), False) == stems["base"]


def canonical(record: RunRecord) -> str:
    return json.dumps(dataclasses.asdict(record), sort_keys=True)


@pytest.mark.parametrize("trial", [False, True])
def test_each_record_names_its_artifacts_and_a_refresh_is_another_run(tmp_path, trial):
    variant = Variant(
        "E33",
        "selection:inner_fold",
        "walk_forward:synthetic",
        {"study": "studies/ability.py"},
        {"thin_starts": 30},
        metrics=(FoldMetric("2020-01-01", "selected[scale]", 1.5, None, None, 40),),
    )
    refreshed = changed(jra_runners="processed/jra/runners.parquet@9999aaaa2222bbbb")
    registry = Registry(tmp_path / "data" / "registry.jsonl", holdout_log=tmp_path / "no.jsonl")
    stems, ids = [], []
    for builds in (BUILDS, refreshed):
        record = ability._record(variant, GIT_SHA, builds, SPEC_HASH, trial)
        stem = ability.artifact_stem(SPEC_HASH, builds, trial)
        assert stem in canonical(record), "the record names its run's artifacts"
        stems.append(stem)
        ids.append(registry.append(record))
    assert stems[0] not in canonical(ability._record(variant, GIT_SHA, refreshed, SPEC_HASH, trial))
    assert ids[0] != ids[1]
    family = ability.TRIAL_FAMILY_ID if trial else ability.FAMILY_ID
    assert registry.family_size(family) == 1, "a refresh of the same builds is not a new variant"


def fake_store(data: Path, runners: bytes) -> None:
    for path, content in (
        (data / "processed" / "jra" / "runners.parquet", runners),
        (data / "processed" / "jra" / "races.parquet", b"synthetic races"),
        (data / "processed" / "jravan" / "horse_pedigree.parquet", b"synthetic pedigree"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def files_under(*roots: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for root in roots for p in root.rglob("*") if p.is_file()}


def test_a_rerun_on_refreshed_data_keeps_the_earlier_artifacts(tmp_path, monkeypatch):
    data, out = tmp_path / "data", tmp_path / "reports"
    run = {"n": 0}

    def fake_compute(inputs, spec, scheme, log=print):
        run["n"] += 1
        study = Study(counts={"level1_races": 10 * run["n"]})
        study.variants.append(
            Variant(
                "E33",
                "selection:inner_fold",
                "walk_forward:synthetic",
                {"study": "studies/ability.py"},
                {"thin_starts": 30},
                metrics=(FoldMetric("all_folds", "delta_ll", 0.01 * run["n"], None, None, 9),),
            )
        )
        export = pd.DataFrame({"race_id": ["r1", "r1"], "u": [0.1 * run["n"], -0.1 * run["n"]]})
        return study, export

    monkeypatch.setattr(ability, "load_inputs", lambda log=print: None)
    monkeypatch.setattr(ability, "compute_study", fake_compute)
    monkeypatch.setattr(ability, "write_committed_summary", lambda registry: None)
    quiet = lambda message: None  # noqa: E731

    fake_store(data, b"synthetic runners, first build")
    first = ability.run_study(out_dir=out, log=quiet)
    builds = {**ability.store_builds(), **ability.pedigree_build()}
    stem = ability.artifact_stem(first["spec_hash"], builds, False)
    assert Path(first["report_path"]).name == f"{stem}.md"
    assert Path(first["export_path"]).name == f"{stem}.parquet"
    kept = files_under(out, data / "features")
    assert len(kept) == 3, "the report, its JSON and the export"
    registry = Registry(Path(first["registry_path"]), holdout_log=tmp_path / "no.jsonl")
    n_first = len(registry.records(ability.FAMILY_ID))

    fake_store(data, b"synthetic runners, refreshed build")
    second = ability.run_study(out_dir=out, log=quiet)
    refreshed = {**ability.store_builds(), **ability.pedigree_build()}
    stem2 = ability.artifact_stem(second["spec_hash"], refreshed, False)

    assert second["spec_hash"] == first["spec_hash"] and stem2 != stem
    assert Path(second["report_path"]).name == f"{stem2}.md"
    assert Path(second["export_path"]).name == f"{stem2}.parquet"
    now = files_under(out, data / "features")
    assert {p: now.get(p) for p in kept} == kept, "the first run's artifacts were overwritten"
    assert len(now) == 6
    records = registry.records(ability.FAMILY_ID)
    assert len(records) == n_first + 1
    assert all(stem in canonical(r) and stem2 not in canonical(r) for r in records[:n_first])
    assert all(stem2 in canonical(r) for r in records[n_first:])
