"""The handicap score, fitted in fold only, and the per-day writer (ticket #129).

Contract pinned here (`horsetrack.countries.japan.handicap.factors`):

- `fit_handicap_score(frame, *, features=None, alpha=1.0) -> HandicapScore`: a ridge-regularised
  conditional logit on the winner (P(i wins) proportional to exp(x_i . beta) within a race; L2
  penalty `alpha` on the coefficients). `frame` is a polars DataFrame with `race_id`, `race_date`
  (a date), `horse_no`, `won` (bool) and the feature columns; `features` defaults to the module's
  `SCORE_FEATURES` (factor columns of `handicap_factors`). A null feature value is allowed. A frame
  holding any row on a holdout day (#71's `is_holdout_day("japan", day)`, D14) is a ValueError.
- `HandicapScore`: `predict(frame) -> polars.Series` (named `handicap_score`, one value per row,
  each race's values summing to 1, finite even where a feature is null); `coef` (feature name ->
  coefficient, every feature present); `train_start`, `train_end` (first and last race_date
  fitted) and `n_races` (races with a winner fitted).
- `oof_handicap_score(frame, *, folds, features=None, alpha=1.0) -> polars.DataFrame`: `folds` are
  #70 `Fold`s over `frame`'s rows (positions); fold k's test rows get `handicap_score` from
  `fit_handicap_score(<fold k's training rows>)`. Columns `race_id`, `horse_no`, `fold` (k),
  `handicap_score`; one row per test row. A holdout row anywhere in `frame` is a ValueError.
- `build_handicap_table(date_from, date_to, *, scheme=None) -> dict[date, Path]`: for each race day
  in the inclusive range that has races in the store and is not a holdout day, one Parquet file
  under the data dir (its path contains the day's ISO date; no other Parquet file is written). Its
  rows are `handicap_factors(race_id, as_of=<that race day>)` for each of the day's races (default
  train_end), built one day at a time, plus `handicap_score`: out of fold over
  `walk_forward(<the built rows' race days>, scheme=scheme)` (default: Japan's #70 scheme), each
  fold fitted with `fit_handicap_score` (default arguments) on its training rows with `won` from
  the store (finish_pos 1); null on a day no fold scores.

Rulings:
- Ruling: `handicap_score` is the in-race win probability of the fitted conditional logit (not the
  raw linear score), so it is comparable with the calibrated-market Win baseline by log-loss.
- Ruling: the factor columns of a written day are fitted as of that race day (default train_end);
  only the score is fitted per #70 fold.

The oracle for the writer's score is `fit_handicap_score` on each fold's training rows (tested on
its own here) applied to the test rows, as test_v0_calibration does for the v0 model.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from handicap_store import at_jst, patch_holdout, perturb, random_store, rid, weekend_days
from handicap_store import R, Store
from horsetrack.core.eval import holdout
from horsetrack.core.eval.folds import FoldScheme, walk_forward
from horsetrack.countries.japan.handicap import factors
from horsetrack.countries.japan.handicap.factors import (
    build_handicap_table,
    fit_handicap_score,
    handicap_factors,
    oof_handicap_score,
)

FEATURES = ["x_good", "x_noise", "x_sparse"]
MONTHLY = FoldScheme(unit="month", test_length=1, step=1, min_train=1, window="expanding",
                     embargo_days=7, anchor_month=1)  # fmt: skip
START = date(2025, 1, 1)


def score_frame(*, seed=5, n_days=120, races_per_day=2, field=6, beta=1.5, start=START):
    """Races whose winner is drawn with P proportional to exp(beta * x_good); x_noise carries
    nothing; x_sparse is null on about 30% of rows (with its flag)."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        day = start + timedelta(days=d)
        for r in range(races_per_day):
            good, noise, sparse = rng.normal(size=(3, field))
            gone = rng.random(field) < 0.3
            p = np.exp(beta * good)
            winner = rng.choice(field, p=p / p.sum())
            for i in range(field):
                rows.append({
                    "race_id": rid(day, "TOK", r + 1), "race_date": day, "horse_no": i + 1,
                    "won": bool(i == winner), "x_good": float(good[i]),
                    "x_noise": float(noise[i]),
                    "x_sparse": None if gone[i] else float(sparse[i]),
                    "x_sparse_missing": bool(gone[i]),
                })  # fmt: skip
    return pl.DataFrame(rows, schema={
        "race_id": pl.String, "race_date": pl.Date, "horse_no": pl.Int64, "won": pl.Boolean,
        "x_good": pl.Float64, "x_noise": pl.Float64, "x_sparse": pl.Float64,
        "x_sparse_missing": pl.Boolean,
    })  # fmt: skip


@pytest.fixture(scope="module")
def frame():
    return score_frame()


@pytest.fixture(scope="module")
def model(frame):
    return fit_handicap_score(frame, features=FEATURES)


def race_sums(frame: pl.DataFrame, scores: pl.Series) -> list[float]:
    return (
        frame.select("race_id").with_columns(scores.alias("s")).group_by("race_id")
        .agg(pl.col("s").sum())["s"].to_list()
    )  # fmt: skip


# ---------------------------------------------------------------------------------------------
# The fit: a ridge conditional logit on the winner
# ---------------------------------------------------------------------------------------------


def test_the_fit_recovers_the_planted_direction(model):
    assert set(FEATURES) <= set(model.coef)
    assert model.coef["x_good"] > 0.8  # planted 1.5 on a unit-variance feature, 240 races
    assert abs(model.coef["x_noise"]) < 0.4
    assert model.coef["x_good"] > 3 * abs(model.coef["x_sparse"])
    race = pl.DataFrame({
        "race_id": ["r"] * 3, "race_date": [START] * 3, "horse_no": [1, 2, 3],
        "won": [False] * 3, "x_good": [-1.0, 0.0, 1.0], "x_noise": [0.0] * 3,
        "x_sparse": [0.0] * 3, "x_sparse_missing": [False] * 3,
    })  # fmt: skip
    p = model.predict(race).to_list()
    assert p[0] < p[1] < p[2]


def test_predict_gives_in_race_probabilities_even_where_a_feature_is_null(frame, model):
    scores = model.predict(frame)
    assert isinstance(scores, pl.Series)
    assert scores.name == "handicap_score"
    assert scores.len() == frame.height
    values = scores.to_numpy()
    assert np.isfinite(values).all() and (values > 0).all() and (values < 1).all()
    assert frame["x_sparse"].null_count() > 0
    assert race_sums(frame, scores) == pytest.approx([1.0] * frame["race_id"].n_unique(), abs=1e-9)
    # the fitted winners' mean probability beats a uniform pick (1/6)
    assert values[frame["won"].to_numpy()].mean() > 1 / 6 + 0.05


def test_ridge_shrinks_the_coefficients_as_alpha_grows(frame):
    coefs = [fit_handicap_score(frame, features=FEATURES, alpha=a).coef["x_good"]
             for a in (0.01, 10.0, 1000.0)]  # fmt: skip
    assert coefs[0] > coefs[1] > coefs[2] > 0


def test_the_fit_records_its_training_rows(frame):
    first_month = frame.filter(pl.col("race_date") < date(2025, 2, 1))
    m = fit_handicap_score(first_month, features=FEATURES)
    assert (m.train_start, m.train_end) == (START, date(2025, 1, 31))
    assert m.n_races == 31 * 2
    # a race without a winner carries no likelihood and is not counted
    no_winner = first_month.with_columns(
        pl.when(pl.col("race_date") == START).then(False).otherwise(pl.col("won")).alias("won")
    )
    assert fit_handicap_score(no_winner, features=FEATURES).n_races == 31 * 2 - 2


# ---------------------------------------------------------------------------------------------
# Never on an A2 holdout day (D14)
# ---------------------------------------------------------------------------------------------


def first_holdout_day() -> date:
    """The first committed holdout day (#71's seeded rule, from 2026-11-28), unpatched."""
    start = date(2026, 11, 28)
    return next(
        start + timedelta(days=k) for k in range(60)
        if holdout.is_holdout_day("japan", start + timedelta(days=k))
    )  # fmt: skip


def test_a_holdout_row_is_refused_by_the_fit_and_the_oof_run(frame):
    hold = first_holdout_day()
    late = score_frame(n_days=40, start=hold - timedelta(days=39), seed=9)
    assert late["race_date"].max() == hold
    with pytest.raises(ValueError):
        fit_handicap_score(late, features=FEATURES)
    dev = late.filter(pl.col("race_date") < hold)
    assert all(not holdout.is_holdout_day("japan", d) for d in dev["race_date"].unique())
    fit_handicap_score(dev, features=FEATURES)  # the same rows without the holdout day fit
    mixed = pl.concat([frame, late.filter(pl.col("race_date") == hold)])
    folds = walk_forward(mixed["race_date"].to_list(), scheme=MONTHLY)
    with pytest.raises(ValueError):
        oof_handicap_score(mixed, folds=folds, features=FEATURES)


# ---------------------------------------------------------------------------------------------
# Walk-forward training folds only (#70)
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oof_case(frame):
    folds = walk_forward(frame["race_date"].to_list(), scheme=MONTHLY)
    assert len(folds) == 3  # February, March, April
    return folds, oof_handicap_score(frame, folds=folds, features=FEATURES)


def keyed(oof: pl.DataFrame) -> pl.DataFrame:
    return oof.sort(["race_id", "horse_no"])


def fold_rows(frame: pl.DataFrame, fold) -> pl.DataFrame:
    return frame[fold.test_idx.tolist()].select("race_id", "horse_no")


def test_oof_scores_each_fold_with_its_training_fit(frame, oof_case):
    folds, oof = oof_case
    assert {"race_id", "horse_no", "fold", "handicap_score"} <= set(oof.columns)
    assert oof.height == sum(len(f.test_idx) for f in folds)
    for k, fold in enumerate(folds):
        train = frame[fold.train_idx.tolist()]
        test = frame[fold.test_idx.tolist()]
        assert train["race_date"].max() <= fold.train_end < test["race_date"].min()
        want = test.select("race_id", "horse_no").with_columns(
            fit_handicap_score(train, features=FEATURES).predict(test).alias("want")
        )
        got = oof.filter(pl.col("fold") == k).join(want, on=["race_id", "horse_no"])
        assert got.height == test.height
        assert got["handicap_score"].to_list() == pytest.approx(got["want"].to_list(), abs=1e-6)


def rows_after(frame: pl.DataFrame, day: date) -> pl.Series:
    return frame["race_date"] > day


def scores_of(oof: pl.DataFrame, k: int) -> list[float]:
    return keyed(oof.filter(pl.col("fold") == k))["handicap_score"].to_list()


def test_a_fold_never_sees_its_own_test_rows_or_anything_after_its_train_end(frame, oof_case):
    folds, base = oof_case
    j = 0
    after = rows_after(frame, folds[j].train_end)
    # every row after fold j's train end: its winner becomes the runner with the lowest x_good,
    # and its features change (the test rows' own features are kept: they are inputs)
    lowest = pl.col("x_good") == pl.col("x_good").min().over("race_id")
    test_rows = pl.Series(np.isin(np.arange(frame.height), folds[j].test_idx))
    changed = frame.with_columns(
        pl.when(after).then(lowest).otherwise(pl.col("won")).alias("won"),
        pl.when(after & ~test_rows).then(pl.col("x_noise") + 5.0).otherwise(pl.col("x_noise"))
        .alias("x_noise"),
    )  # fmt: skip
    assert not changed["won"].equals(frame["won"])
    again = oof_handicap_score(changed, folds=folds, features=FEATURES)
    assert scores_of(again, j) == pytest.approx(scores_of(base, j), abs=1e-9)
    # a later fold trains on those rows, so the change must reach it (the test can fail)
    later = np.array(scores_of(again, j + 1)) - np.array(scores_of(base, j + 1))
    assert np.abs(later).max() > 1e-4


# ---------------------------------------------------------------------------------------------
# The per-day writer: factors and the in-fold score, one Parquet partition per race day
# ---------------------------------------------------------------------------------------------

WEEKLY = FoldScheme(unit="week", test_length=1, step=1, min_train=2, window="expanding",
                    embargo_days=1, anchor_month=1)  # fmt: skip
BUILD_DAYS = weekend_days(date(2025, 3, 1), date(2025, 4, 6))  # six weekends, 12 race days


def build_store() -> Store:
    return random_store(BUILD_DAYS, seed=23, pool=20)


def parquet_outside_store(data: Path) -> set[Path]:
    jra = data / "processed" / "jra"
    store_files = {(jra / "races.parquet").resolve(), (jra / "runners.parquet").resolve()}
    return {p.resolve() for p in data.rglob("*.parquet")} - store_files


def day_factors(s: Store, day: date) -> pl.DataFrame:
    frames = [handicap_factors(r, as_of=at_jst(day, 9)) for r in sorted(s.races_by_day[day])]
    return pl.concat(frames).sort(["race_id", "horse_no"])


def read_partitions(written: dict) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p) for _, p in sorted(written.items())]).sort(
        ["race_id", "horse_no"]
    )


def with_labels(frame: pl.DataFrame, s: Store) -> pl.DataFrame:
    won = s.won()
    return frame.with_columns(
        pl.col("race_id").str.slice(4, 10).str.to_date().alias("race_date"),
        pl.struct("race_id", "horse_no")
        .map_elements(lambda k: won[(k["race_id"], k["horse_no"])], return_dtype=pl.Boolean)
        .alias("won"),
    )  # fmt: skip


def test_build_writes_one_partition_per_race_day_with_the_in_fold_score(tmp_path, monkeypatch):
    s = build_store()
    data = s.write(tmp_path, monkeypatch)
    written = build_handicap_table(BUILD_DAYS[0], BUILD_DAYS[-1], scheme=WEEKLY)

    assert sorted(written) == BUILD_DAYS
    assert parquet_outside_store(data) == {Path(p).resolve() for p in written.values()}
    repo = Path(__file__).resolve().parents[4]
    for day, path in written.items():
        path = Path(path).resolve()
        assert path.is_relative_to(data) and not path.is_relative_to(repo)
        assert day.isoformat() in str(path)
        part = pl.read_parquet(path).sort(["race_id", "horse_no"])
        want = day_factors(s, day)
        assert_frame_equal(part.select(want.columns), want)  # the day's races only, as of it
        assert "handicap_score" in part.columns

    rows = with_labels(read_partitions(written), s)
    folds = walk_forward(rows["race_date"].to_list(), scheme=WEEKLY)
    assert len(folds) == 3
    scored = np.zeros(rows.height, dtype=bool)
    for fold in folds:
        train = rows[fold.train_idx.tolist()].drop("handicap_score")
        test = rows[fold.test_idx.tolist()]
        want = fit_handicap_score(train).predict(test.drop("handicap_score")).to_list()
        assert test["handicap_score"].to_list() == pytest.approx(want, abs=1e-6)
        sums = test.group_by("race_id").agg(pl.col("handicap_score").sum())["handicap_score"]
        assert sums.to_list() == pytest.approx([1.0] * sums.len(), abs=1e-9)
        scored[fold.test_idx] = True
    # the first two weekends train only: no fold scores them
    assert rows.filter(pl.Series(~scored))["handicap_score"].null_count() == int((~scored).sum())
    assert not math.isclose(float(rows["handicap_score"].drop_nulls().std()), 0.0)


def test_build_never_reads_a_holdout_day(tmp_path, monkeypatch):
    """A day marked holdout gets no partition, and changing its results changes no written value:
    it feeds neither the factors nor a fit. The same change on a development day does."""
    hidden = date(2025, 3, 15)  # a training-only day of fold 0
    s = build_store()
    patch_holdout(monkeypatch, [hidden], factors)
    s.write(tmp_path, monkeypatch)
    written = build_handicap_table(BUILD_DAYS[0], BUILD_DAYS[-1], scheme=WEEKLY)
    assert sorted(written) == [d for d in BUILD_DAYS if d != hidden]
    base = read_partitions(written)
    assert hidden.isoformat() not in "".join(base["race_id"].to_list())

    perturb(s, lambda d: d == hidden).write(tmp_path, monkeypatch, sub="changed")
    again = build_handicap_table(BUILD_DAYS[0], BUILD_DAYS[-1], scheme=WEEKLY)
    assert_frame_equal(read_partitions(again), base)

    dev = date(2025, 3, 16)
    perturb(s, lambda d: d == dev).write(tmp_path, monkeypatch, sub="changed-dev")
    moved = read_partitions(build_handicap_table(BUILD_DAYS[0], BUILD_DAYS[-1], scheme=WEEKLY))
    assert not moved.equals(base)


def test_build_skips_the_committed_holdout_days(tmp_path, monkeypatch):
    """Unpatched: a race day #71's committed rule marks holdout (2026-11-28 on) gets no partition
    and no row, with Japan's own fold scheme (which scores nothing this short)."""
    hold = first_holdout_day()
    dev_after = next(
        hold + timedelta(days=k) for k in range(1, 60)
        if not holdout.is_holdout_day("japan", hold + timedelta(days=k))
    )  # fmt: skip
    days = [date(2026, 11, 21), date(2026, 11, 22), hold, dev_after]
    s = Store()
    for k, day in enumerate(days):
        s.race(day, [R("h:echo", pos=1 + k % 2), R("h:fox", pos=2 - k % 2)], fill_to=6)
    data = s.write(tmp_path, monkeypatch)
    written = build_handicap_table(days[0], days[-1])
    assert set(written) == {days[0], days[1], dev_after}
    assert len(parquet_outside_store(data)) == 3
    for path in written.values():
        part = pl.read_parquet(path)
        assert hold.isoformat() not in "".join(part["race_id"].to_list())
        assert part["handicap_score"].null_count() == part.height
