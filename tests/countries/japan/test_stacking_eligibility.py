"""Ticket #418: stacked-fold eligibility after the member joins
(`horsetrack.countries.japan.models.stacking`).

Contract pinned here (the coder implements what these tests call):

- `join_members(frame, folds, predictions, *, members) -> Level2`. `frame` is the store's level-2
  rows (`race_id`, `horse_id`, `race_date`, and whatever the fit reads), on a unique index;
  `folds` are `core.eval.folds.Fold`s as positions into `frame` (from `walk_forward` on its
  `race_date`). `predictions` is long: one row per (race_id, horse_id, member) with `logp` and the
  member's provenance, `member_fold` (the member fold the prediction is out of) and
  `member_train_end` (the last day the member model that made it trained on). A race is kept only
  when every one of its frame rows has a prediction of every member.
  `Level2.frame`: the kept rows, on `frame`'s index labels, with one column per member holding its
  `logp`; `Level2.folds`: the folds as positions into `Level2.frame` (the same rows, the dropped
  ones gone); `Level2.provenance`: one row per kept (row, member) with `race_id`, `horse_id`,
  `member`, `member_fold`, `member_train_end`.
- `fold_windows(level2, *, min_seasons=D9_MIN_SEASONS) -> list[FoldWindow]`, never raising: per
  fold, `test_start`, `train_end`, `train_from` / `train_to` (first and last race day of the joined
  training rows), `years` (((train_to - train_from).days + 1) / 365.25, as `studies.meta` counts
  D9's seasons), `train_races` and `eligible` (`years >= min_seasons`).
- `stacked_oof(level2, fit, predict, *, min_seasons=D9_MIN_SEASONS) -> (oof, windows)`: refuses
  before any `fit` when a fold's joined training window is under `min_seasons` (`IneligibleFold`,
  naming the fold's test start) or when a member prediction of a row a fold uses (train or test)
  comes from a member model trained on that row, `member_train_end >= race_date`, or has no
  provenance (`InvalidProvenance`). Otherwise `models.meta.oof_predictions` over `Level2.folds`.
  A missing provenance column is `InvalidProvenance` (from `join_members` or `stacked_oof`); a
  duplicated (race_id, horse_id, member) prediction is a `ValueError`. Both errors are
  `ValueError`s. `D9_MIN_SEASONS` is `studies.meta.D9_MIN_SEASONS` (5).

Every frame is synthetic and in memory.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party.
from __future__ import annotations  # noqa: I001

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.folds import load_country_scheme, walk_forward
from horsetrack.countries.japan.models import stacking as st
from horsetrack.countries.japan.studies.meta import D9_MIN_SEASONS

MEMBERS = ["a", "b"]
FIRST_DAY = date(2012, 1, 7)
LAST_DAY = date(2024, 6, 29)
RUNNERS = 3
SCORED_FROM = pd.Timestamp(2024, 1, 1)
PROVENANCE = ["race_id", "horse_id", "member", "member_fold", "member_train_end"]


def store_frame() -> pd.DataFrame:
    """A race every 14 days, three runners each, on index labels that are not positions."""
    rows = []
    day, k = FIRST_DAY, 0
    while day <= LAST_DAY:
        for i in range(RUNNERS):
            rows.append(
                {
                    "race_id": f"jra:{day.isoformat()}:QQV:01",
                    "horse_id": f"h{k}-{i}",
                    "race_date": pd.Timestamp(day),
                    "won": int(i == k % RUNNERS),
                    "ln_q_final": float(np.log((i + 1) / 6)),
                }
            )
        day, k = day + timedelta(days=14), k + 1
    frame = pd.DataFrame(rows)
    frame.index = frame.index * 3 + 1000
    return frame


FRAME = store_frame()
FOLDS = [
    f
    for f in walk_forward(FRAME["race_date"], scheme=load_country_scheme("japan"))
    if pd.Timestamp(f.test_start) >= SCORED_FROM
]


def predictions(starts: dict[str, date], frame: pd.DataFrame = FRAME) -> pd.DataFrame:
    """Each member's season walk-forward predictions from its start day on: out of the race's
    season, trained to the end of the season before."""
    parts = []
    for m, (member, start) in enumerate(starts.items()):
        rows = frame.loc[frame["race_date"] >= pd.Timestamp(start)]
        years = rows["race_date"].dt.year
        parts.append(
            pd.DataFrame(
                {
                    "race_id": rows["race_id"].to_numpy(),
                    "horse_id": rows["horse_id"].to_numpy(),
                    "member": member,
                    "logp": -1.0 - m - 1e-4 * np.arange(len(rows)),
                    "member_fold": years.astype(str).to_numpy(),
                    "member_train_end": pd.to_datetime(
                        [f"{y - 1}-12-31" for y in years]
                    ).to_numpy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


class FitSpy:
    def __init__(self) -> None:
        self.trained: list[pd.DataFrame] = []

    def fit(self, rows: pd.DataFrame) -> int:
        self.trained.append(rows)
        return len(self.trained)

    @staticmethod
    def predict(model: int, rows: pd.DataFrame) -> np.ndarray:
        return np.full(len(rows), -float(model))


def expected_window(fold, kept: pd.DataFrame) -> tuple[date, date, int]:
    """First and last race day and races of the kept rows a fold trains on, from the frame."""
    train = kept.loc[kept["race_date"] <= pd.Timestamp(fold.train_end)]
    first, last = train["race_date"].min().date(), train["race_date"].max().date()
    return first, last, train["race_id"].nunique()


def test_the_fixture_has_two_scored_folds_and_d9_is_five_seasons() -> None:
    assert D9_MIN_SEASONS == 5
    assert [f.test_start for f in FOLDS] == [date(2024, 1, 1), date(2024, 4, 1)]
    # the store alone trains every scored fold on more than ten seasons
    assert all(FRAME.iloc[f.train_idx]["race_date"].min().year == 2012 for f in FOLDS)


# ------------------------------------------------------------------------------------------------
# A valid join


def test_a_valid_join_records_provenance_and_reports_the_eligible_window() -> None:
    preds = predictions({"a": date(2016, 1, 1), "b": date(2017, 1, 1)})
    # one starter of a 2019 race lacks member b: the whole race leaves the level-2 frame
    gap_race = FRAME.loc[FRAME["race_date"] >= pd.Timestamp(2019, 3, 1), "race_id"].iloc[0]
    gap_horse = FRAME.loc[FRAME["race_id"] == gap_race, "horse_id"].iloc[1]
    preds = preds.loc[
        ~(
            (preds["member"] == "b")
            & (preds["race_id"] == gap_race)
            & (preds["horse_id"] == gap_horse)
        )
    ]

    level2 = st.join_members(FRAME, FOLDS, preds, members=MEMBERS)

    kept = FRAME.loc[
        (FRAME["race_date"] >= pd.Timestamp(2017, 1, 1)) & (FRAME["race_id"] != gap_race)
    ]
    assert sorted(level2.frame.index) == sorted(kept.index)
    # each member column holds that member's logp for the row
    by_key = preds.set_index(["race_id", "horse_id", "member"])["logp"]
    for member in MEMBERS:
        want = [
            by_key[(r, h, member)] for r, h in zip(kept["race_id"], kept["horse_id"], strict=True)
        ]
        np.testing.assert_allclose(level2.frame.loc[kept.index, member].to_numpy(), want)
    # the folds keep the same rows, as positions into the level-2 frame
    assert len(level2.folds) == len(FOLDS)
    for old, new in zip(FOLDS, level2.folds, strict=True):
        for name in ("train_idx", "test_idx"):
            labels = set(FRAME.index[getattr(old, name)]) & set(level2.frame.index)
            assert set(level2.frame.index[getattr(new, name)]) == labels, name
        assert new.train_end == old.train_end and new.test_start == old.test_start
    # provenance: one row per kept row and member, the member fold and its training cutoff
    prov = level2.provenance
    assert set(PROVENANCE) <= set(prov.columns)
    assert len(prov) == len(kept) * len(MEMBERS)
    row = kept.iloc[5]
    got = prov.loc[(prov["race_id"] == row["race_id"]) & (prov["horse_id"] == row["horse_id"])]
    assert sorted(got["member"]) == MEMBERS
    assert set(got["member_fold"].astype(str)) == {str(row["race_date"].year)}
    assert {pd.Timestamp(t).date() for t in got["member_train_end"]} == {
        date(row["race_date"].year - 1, 12, 31)
    }

    windows = st.fold_windows(level2)
    assert len(windows) == len(FOLDS)
    for fold, w in zip(FOLDS, windows, strict=True):
        first, last, races = expected_window(fold, kept)
        assert w.test_start == fold.test_start and w.train_end == fold.train_end
        assert first >= date(2017, 1, 1)  # the join's start, not the store's 2012
        assert pd.Timestamp(w.train_from).date() == first
        assert pd.Timestamp(w.train_to).date() == last
        assert w.train_races == races
        assert w.years == pytest.approx(((last - first).days + 1) / 365.25, abs=0.01)
        assert w.years > 6.5 and w.eligible

    spy = FitSpy()
    oof, reported = st.stacked_oof(level2, spy.fit, spy.predict)
    assert [w.train_from for w in reported] == [w.train_from for w in windows]
    assert spy.trained, "a valid level-2 run fits"
    for rows in spy.trained:
        assert set(MEMBERS) <= set(rows.columns)
        assert rows["race_date"].min() >= pd.Timestamp(2017, 1, 1)
    tested = set().union(*(set(level2.frame.index[f.test_idx]) for f in level2.folds))
    assert set(oof.index[oof.notna()]) == tested
    assert set(oof.index) == set(level2.frame.index)


# ------------------------------------------------------------------------------------------------
# Joins that shorten the training window below D9


@pytest.mark.parametrize(
    ("starts", "min_seasons", "eligible"),
    [
        # both members start in 2021: three seasons before 2024
        ({"a": date(2021, 1, 1), "b": date(2021, 1, 1)}, None, False),
        # one long member and one short one: the join keeps the short one's span
        ({"a": date(2013, 1, 1), "b": date(2021, 1, 1)}, None, False),
        # four and a half seasons before 2024-Q1, under five
        ({"a": date(2019, 7, 1), "b": date(2019, 7, 1)}, None, False),
        # five and a half seasons: D9 met
        ({"a": date(2018, 6, 1), "b": date(2018, 6, 1)}, None, True),
        # the threshold is the argument's
        ({"a": date(2017, 1, 1), "b": date(2017, 1, 1)}, 8, False),
        ({"a": date(2021, 1, 1), "b": date(2021, 1, 1)}, 2, True),
    ],
)
def test_a_join_below_the_minimum_history_refuses_before_any_fit(
    starts: dict[str, date], min_seasons: int | None, eligible: bool
) -> None:
    kw = {} if min_seasons is None else {"min_seasons": min_seasons}
    level2 = st.join_members(FRAME, FOLDS, predictions(starts), members=MEMBERS)
    start = max(starts.values())
    kept = FRAME.loc[FRAME["race_date"] >= pd.Timestamp(start)]

    windows = st.fold_windows(level2, **kw)
    for fold, w in zip(FOLDS, windows, strict=True):
        first, last, races = expected_window(fold, kept)
        assert pd.Timestamp(w.train_from).date() == first
        assert pd.Timestamp(w.train_to).date() == last
        assert w.train_races == races
    assert [bool(w.eligible) for w in windows] == [eligible] * len(FOLDS)

    spy = FitSpy()
    if eligible:
        oof, _ = st.stacked_oof(level2, spy.fit, spy.predict, **kw)
        assert spy.trained and oof.notna().any()
        return
    with pytest.raises(st.IneligibleFold) as info:
        st.stacked_oof(level2, spy.fit, spy.predict, **kw)
    assert spy.trained == [], "the fit ran before the refusal"
    assert isinstance(info.value, ValueError)
    assert FOLDS[0].test_start.isoformat() in str(info.value)


def test_only_the_short_fold_needs_to_fail_for_the_run_to_refuse() -> None:
    # 2024-Q1 trains on 4.95 seasons from 2019-01-05 and 2024-Q2 on 5.22: the first fold fails
    starts = {"a": date(2019, 1, 1), "b": date(2019, 1, 1)}
    level2 = st.join_members(FRAME, FOLDS, predictions(starts), members=MEMBERS)
    windows = st.fold_windows(level2)
    assert [bool(w.eligible) for w in windows] == [False, True], [w.years for w in windows]
    spy = FitSpy()
    with pytest.raises(st.IneligibleFold) as info:
        st.stacked_oof(level2, spy.fit, spy.predict)
    assert spy.trained == []
    assert FOLDS[0].test_start.isoformat() in str(info.value)
    assert FOLDS[1].test_start.isoformat() not in str(info.value)


# ------------------------------------------------------------------------------------------------
# Member provenance


def _row_on(day: pd.Timestamp) -> pd.Series:
    return FRAME.loc[FRAME["race_date"] >= day].iloc[1]


@pytest.mark.parametrize(
    ("when", "offset_days", "valid"),
    [
        ("train", 0, False),  # the member trained through the row's race day
        ("train", 30, False),  # and beyond it
        ("test", 0, False),  # a scored row's member prediction is in-sample
        ("test", 200, False),
        ("train", -1, True),  # trained to the day before: out of sample
        ("test", -1, True),
    ],
)
def test_a_member_prediction_trained_on_its_row_refuses_before_any_fit(
    when: str, offset_days: int, valid: bool
) -> None:
    starts = {"a": date(2016, 1, 1), "b": date(2016, 1, 1)}
    preds = predictions(starts)
    row = _row_on(pd.Timestamp(2020, 5, 1) if when == "train" else pd.Timestamp(2024, 2, 1))
    hit = (
        (preds["member"] == "b")
        & (preds["race_id"] == row["race_id"])
        & (preds["horse_id"] == row["horse_id"])
    )
    assert hit.sum() == 1
    preds.loc[hit, "member_train_end"] = row["race_date"] + pd.Timedelta(days=offset_days)
    spy = FitSpy()
    if valid:
        level2 = st.join_members(FRAME, FOLDS, preds, members=MEMBERS)
        oof, _ = st.stacked_oof(level2, spy.fit, spy.predict)
        assert spy.trained and oof.notna().any()
        return
    with pytest.raises(st.InvalidProvenance) as info:
        level2 = st.join_members(FRAME, FOLDS, preds, members=MEMBERS)
        st.stacked_oof(level2, spy.fit, spy.predict)
    assert spy.trained == [], "the fit ran before the refusal"
    assert isinstance(info.value, ValueError)


@pytest.mark.parametrize("column", ["member_fold", "member_train_end"])
def test_missing_provenance_columns_refuse(column: str) -> None:
    preds = predictions({"a": date(2016, 1, 1), "b": date(2016, 1, 1)}).drop(columns=column)
    spy = FitSpy()
    with pytest.raises(st.InvalidProvenance):
        level2 = st.join_members(FRAME, FOLDS, preds, members=MEMBERS)
        st.stacked_oof(level2, spy.fit, spy.predict)
    assert spy.trained == []


@pytest.mark.parametrize("column", ["member_fold", "member_train_end"])
def test_a_missing_provenance_value_on_a_used_row_refuses(column: str) -> None:
    preds = predictions({"a": date(2016, 1, 1), "b": date(2016, 1, 1)})
    row = _row_on(pd.Timestamp(2021, 8, 1))
    hit = (preds["member"] == "a") & (preds["horse_id"] == row["horse_id"])
    preds.loc[hit, column] = None
    spy = FitSpy()
    with pytest.raises(st.InvalidProvenance):
        level2 = st.join_members(FRAME, FOLDS, preds, members=MEMBERS)
        st.stacked_oof(level2, spy.fit, spy.predict)
    assert spy.trained == []


def test_a_duplicated_member_prediction_refuses() -> None:
    preds = predictions({"a": date(2016, 1, 1), "b": date(2016, 1, 1)})
    preds = pd.concat([preds, preds.iloc[[40]]], ignore_index=True)
    spy = FitSpy()
    with pytest.raises(ValueError):
        level2 = st.join_members(FRAME, FOLDS, preds, members=MEMBERS)
        st.stacked_oof(level2, spy.fit, spy.predict)
    assert spy.trained == []
