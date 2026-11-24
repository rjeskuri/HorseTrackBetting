"""Walk-forward folds by date (ticket #70, problem statement section 4, D9).

Expected windows are derived here from calendar arithmetic, never from the code under test: a test
window is `test_length` scheme units starting on a unit boundary (months anchored on `anchor_month`,
weeks on Monday), successive windows start `step` units apart, and the first one starts on the first
boundary at least `min_train` units after the earliest date.
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.folds import Fold, FoldScheme, walk_forward

FIELDS = ("unit", "test_length", "step", "min_train", "window", "embargo_days", "anchor_month")
MONTHS_PER_UNIT = {"month": 1, "quarter": 3, "year": 12}
RACE_WEEKDAYS = (2, 5, 6)  # Wednesday, Saturday, Sunday


def make_scheme(**overrides) -> FoldScheme:
    fields = {
        "unit": "quarter",
        "test_length": 1,
        "step": 1,
        "min_train": 4,
        "window": "expanding",
        "embargo_days": 7,
        "anchor_month": 1,
    }
    fields.update(overrides)
    return FoldScheme(**fields)


# --- calendar arithmetic (independent of the module) ---------------------------------------------


def add_months(d: date, n: int) -> date:
    years, month0 = divmod(d.month - 1 + n, 12)
    return date(d.year + years, month0 + 1, d.day)


def add_units(d: date, unit: str, n: int) -> date:
    if unit == "day":
        return d + timedelta(days=n)
    if unit == "week":
        return d + timedelta(weeks=n)
    return add_months(d, MONTHS_PER_UNIT[unit] * n)


def next_boundary(d: date, unit: str, anchor: int) -> date:
    """The first unit boundary on or after d."""
    if unit == "day":
        return d
    if unit == "week":
        return d + timedelta(days=(7 - d.weekday()) % 7)
    b = date(d.year, d.month, 1)
    if b < d:
        b = add_months(b, 1)
    while (b.month - anchor) % MONTHS_PER_UNIT[unit]:
        b = add_months(b, 1)
    return b


def expected_windows(py_dates: list[date], s: FoldScheme) -> list[tuple[date, date]]:
    """Inclusive (first, last) calendar day of every test window that starts within the data."""
    first = next_boundary(add_units(min(py_dates), s.unit, s.min_train), s.unit, s.anchor_month)
    last_date = max(py_dates)
    windows = []
    k = 0
    while (lo := add_units(first, s.unit, k * s.step)) <= last_date:
        windows.append((lo, add_units(lo, s.unit, s.test_length) - timedelta(days=1)))
        k += 1
    return windows


# --- fixtures ------------------------------------------------------------------------------------


def calendar(start: date, end: date, *, daily: bool) -> list[date]:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    return days if daily else [d for d in days if d.weekday() in RACE_WEEKDAYS]


def date_series(
    start: date,
    end: date,
    *,
    daily: bool = False,
    shuffle: bool = True,
    as_objects: bool = False,
    seed: int = 7,
) -> pd.Series:
    """Several rows per day (1 to 3), shuffled, with an index whose labels are not positions."""
    rows = [d for d in calendar(start, end, daily=daily) for _ in range(1 + d.toordinal() % 3)]
    rng = np.random.default_rng(seed)
    if shuffle:
        rows = [rows[i] for i in rng.permutation(len(rows))]
    index = pd.Index(rng.permutation(len(rows)) + 10_000)
    values = rows if as_objects else pd.to_datetime(rows)
    return pd.Series(values, index=index, dtype=object if as_objects else None)


def to_py(dates: pd.Series) -> list[date]:
    return [x.date() if isinstance(x, datetime) else x for x in dates]


def at(py_dates: list[date], idx: np.ndarray) -> list[date]:
    return [py_dates[i] for i in idx]


def check_fold_shape(fold: Fold, n: int) -> None:
    for idx in (fold.train_idx, fold.test_idx):
        assert isinstance(idx, np.ndarray) and idx.dtype.kind in "iu"
        assert len(idx) == len(set(idx.tolist())), "an index appears twice in one fold"
        assert ((0 <= idx) & (idx < n)).all(), "indices must be positions into dates"
    for d in (fold.train_end, fold.test_start, fold.test_end):
        assert isinstance(d, date) and not isinstance(d, datetime)


# Each case: (start, end, scheme overrides, fixture options). End dates close a whole test window.
CASES = {
    "signed_d9_quarterly_expanding_5_seasons_1_week": (
        date(2014, 1, 1),
        date(2020, 12, 31),
        dict(min_train=20, embargo_days=7),
        {},
    ),
    "half_years_from_april": (
        date(2015, 4, 1),
        date(2019, 9, 30),
        dict(test_length=2, step=2, anchor_month=4, embargo_days=14),
        {},
    ),
    "quarters_every_other_quarter": (
        date(2015, 1, 1),
        date(2018, 12, 31),
        dict(step=2, embargo_days=7),
        {},
    ),
    "months_no_embargo": (
        date(2016, 1, 1),
        date(2018, 6, 30),
        dict(unit="month", min_train=12, embargo_days=0),
        {},
    ),
    "season_years_from_september": (
        date(2013, 9, 1),
        date(2019, 8, 31),
        dict(unit="year", min_train=3, anchor_month=9, embargo_days=7),
        {},
    ),
    "weeks_rolling": (
        date(2017, 1, 2),
        date(2017, 12, 31),
        dict(unit="week", min_train=8, window="rolling", embargo_days=3),
        {},
    ),
    "days_daily_data": (
        date(2018, 1, 1),
        date(2018, 3, 27),
        dict(unit="day", test_length=7, step=7, min_train=30, embargo_days=2),
        {"daily": True},
    ),
    "python_date_objects": (
        date(2015, 1, 1),
        date(2018, 12, 31),
        dict(embargo_days=30),
        {"as_objects": True},
    ),
}


def build(case: str):
    start, end, overrides, options = CASES[case]
    s = make_scheme(**overrides)
    dates = date_series(start, end, **options)
    py = to_py(dates)
    windows = expected_windows(py, s)
    # The calendar runs through `end`; race-day fixtures may have no row on that day itself, so the
    # last window is checked against the calendar, not against the last row's date.
    assert windows and windows[-1][1] == end, "fixture must end on its last test window's last day"
    return s, dates, py, windows


# --- definition of done --------------------------------------------------------------------------


@pytest.mark.parametrize("case", list(CASES))
def test_every_training_date_precedes_its_test_dates(case):
    s, dates, py, windows = build(case)

    folds = walk_forward(dates, scheme=s)

    assert len(folds) == len(windows)
    for fold in folds:
        check_fold_shape(fold, len(dates))
        assert len(fold.train_idx) and len(fold.test_idx)
        train, test = at(py, fold.train_idx), at(py, fold.test_idx)
        assert max(train) < min(test)
        assert (min(test) - max(train)).days >= s.embargo_days
        assert max(train) <= fold.train_end < fold.test_start <= min(test)
        assert set(fold.train_idx.tolist()).isdisjoint(fold.test_idx.tolist())


@pytest.mark.parametrize("case", list(CASES))
def test_test_windows_are_ordered_and_do_not_overlap(case):
    s, dates, py, windows = build(case)

    folds = walk_forward(dates, scheme=s)

    assert len(folds) == len(windows)
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert earlier.test_start < later.test_start
        assert earlier.test_end < later.test_start
    all_test = np.concatenate([f.test_idx for f in folds])
    assert len(all_test) == len(set(all_test.tolist())), "a row is scored in two folds"
    for fold, (lo, hi) in zip(folds, windows, strict=True):
        check_fold_shape(fold, len(dates))
        test = at(py, fold.test_idx)
        assert all(fold.test_start <= d <= fold.test_end for d in test)
        # the window is exactly the scheme's calendar units: every row dated inside it, none outside
        assert set(fold.test_idx.tolist()) == {i for i, d in enumerate(py) if lo <= d <= hi}


EXCLUDES = {
    "holdout_last_year": (date(2019, 1, 1), date(2019, 12, 31)),
    "across_two_quarters": (date(2017, 2, 10), date(2017, 5, 20)),
    "one_whole_quarter": (date(2017, 7, 1), date(2017, 9, 30)),
    "one_race_day": (date(2018, 3, 3), date(2018, 3, 3)),
}


@pytest.mark.parametrize("unit", ["quarter", "month"])
@pytest.mark.parametrize("name", list(EXCLUDES))
def test_no_fold_contains_an_excluded_date(name, unit):
    lo_x, hi_x = EXCLUDES[name]
    s = make_scheme(unit=unit, min_train=4, embargo_days=7)
    dates = date_series(date(2015, 1, 1), date(2019, 12, 31))
    py = to_py(dates)
    excluded = {i for i, d in enumerate(py) if lo_x <= d <= hi_x}
    assert excluded, "fixture must have rows in the excluded range"
    windows = expected_windows(py, s)

    folds = walk_forward(dates, scheme=s, exclude=(lo_x, hi_x))

    for fold in folds:
        check_fold_shape(fold, len(dates))
        assert excluded.isdisjoint(fold.train_idx.tolist())
        assert excluded.isdisjoint(fold.test_idx.tolist())
        assert len(fold.test_idx), "a window emptied by the exclusion is dropped, not returned"
    # every other row in a window is still scored, once, and training still uses all earlier rows
    kept = [{i for i, d in enumerate(py) if lo <= d <= hi} - excluded for lo, hi in windows]
    kept = [w for w in kept if w]
    assert [set(f.test_idx.tolist()) for f in folds] == kept
    for fold in folds:
        first_test = min(at(py, fold.test_idx))
        must_train = {
            i
            for i, d in enumerate(py)
            if i not in excluded and (first_test - d).days > s.embargo_days
        }
        assert must_train <= set(fold.train_idx.tolist())


MIN_TRAIN = {
    # name: (unit, anchor_month, earliest date, min_train, first scored date)
    "quarters_4": ("quarter", 1, date(2015, 1, 1), 4, date(2016, 1, 1)),
    "quarters_7": ("quarter", 1, date(2015, 1, 1), 7, date(2016, 10, 1)),
    "quarters_20_signed": ("quarter", 1, date(2014, 1, 1), 20, date(2019, 1, 1)),
    "quarters_from_april_5": ("quarter", 4, date(2015, 4, 1), 5, date(2016, 7, 1)),
    "quarters_mid_quarter_start": ("quarter", 1, date(2015, 2, 15), 4, date(2016, 4, 1)),
    "months_5": ("month", 1, date(2016, 3, 1), 5, date(2016, 8, 1)),
    "season_years_2": ("year", 9, date(2014, 9, 1), 2, date(2016, 9, 1)),
    "weeks_10": ("week", 1, date(2018, 1, 1), 10, date(2018, 3, 12)),
    "days_45": ("day", 1, date(2018, 1, 1), 45, date(2018, 2, 15)),
}


@pytest.mark.parametrize("name", list(MIN_TRAIN))
def test_the_minimum_training_window_is_respected(name):
    unit, anchor, earliest, min_train, first_scored = MIN_TRAIN[name]
    s = make_scheme(unit=unit, anchor_month=anchor, min_train=min_train, embargo_days=0)
    end = add_units(first_scored, unit, 3) - timedelta(days=1)
    dates = date_series(earliest, end, daily=True)
    py = to_py(dates)

    folds = walk_forward(dates, scheme=s)

    assert len(folds) == 3
    assert folds[0].test_start >= add_units(earliest, unit, min_train)
    assert min(at(py, folds[0].test_idx)) == first_scored
    assert all(d >= first_scored for f in folds for d in at(py, f.test_idx))
    first_train = at(py, folds[0].train_idx)
    assert min(first_train) == earliest
    assert max(first_train) == first_scored - timedelta(days=1)


@pytest.mark.parametrize("missing", FIELDS)
def test_a_scheme_missing_a_field_is_refused(missing):
    full = {
        "unit": "quarter",
        "test_length": 1,
        "step": 1,
        "min_train": 20,
        "window": "expanding",
        "embargo_days": 7,
        "anchor_month": 1,
    }
    assert FoldScheme.from_mapping(full) == FoldScheme(**full)

    partial = {k: v for k, v in full.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        FoldScheme.from_mapping(partial)


# --- broader behaviour ---------------------------------------------------------------------------


BAD_VALUES = {
    "unknown_unit": {"unit": "fortnight"},
    "zero_test_length": {"test_length": 0},
    "negative_test_length": {"test_length": -1},
    "zero_step": {"step": 0},
    "step_shorter_than_window_overlaps": {"test_length": 2, "step": 1},
    "negative_min_train": {"min_train": -1},
    "unknown_window": {"window": "sliding"},
    "negative_embargo": {"embargo_days": -1},
    "anchor_month_zero": {"anchor_month": 0},
    "anchor_month_13": {"anchor_month": 13},
}


@pytest.mark.parametrize("name", list(BAD_VALUES))
def test_a_scheme_with_a_bad_value_is_refused(name):
    with pytest.raises(ValueError):
        make_scheme(**BAD_VALUES[name])
    full = dataclasses.asdict(make_scheme())
    with pytest.raises(ValueError):
        FoldScheme.from_mapping({**full, **BAD_VALUES[name]})


def test_a_scheme_is_frozen():
    s = make_scheme()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.min_train = 1


@pytest.mark.parametrize("embargo", [0, 1, 7, 30])
def test_expanding_training_starts_at_the_earliest_date_grows_and_stops_at_the_embargo(embargo):
    s = make_scheme(min_train=4, embargo_days=embargo)
    dates = date_series(date(2015, 1, 1), date(2018, 12, 31), daily=True)
    py = to_py(dates)
    earliest = min(py)

    folds = walk_forward(dates, scheme=s)

    assert len(folds) == 12
    previous: set[int] = set()
    for fold in folds:
        train = set(fold.train_idx.tolist())
        first_test = min(at(py, fold.test_idx))
        assert min(at(py, fold.train_idx)) == earliest
        assert previous < train, "the expanding training set must grow every fold"
        must = {i for i, d in enumerate(py) if (first_test - d).days > embargo}
        may = {i for i, d in enumerate(py) if (first_test - d).days >= max(embargo, 1)}
        assert must <= train <= may
        previous = train


def test_unsorted_dates_give_positions_into_the_original_series():
    raw = [
        "2020-03-14",  # 0
        "2020-01-01",  # 1
        "2020-04-02",  # 2
        "2020-02-01",  # 3
        "2020-01-25",  # 4
        "2020-03-01",  # 5
        "2020-04-30",  # 6
        "2020-02-29",  # 7
    ]
    dates = pd.Series(pd.to_datetime(raw), index=list("abcdefgh"))
    s = make_scheme(unit="month", min_train=2, embargo_days=0)

    folds = walk_forward(dates, scheme=s)

    assert len(folds) == 2
    assert sorted(folds[0].train_idx.tolist()) == [1, 3, 4, 7]
    assert sorted(folds[0].test_idx.tolist()) == [0, 5]
    assert folds[0].test_start == date(2020, 3, 1)
    assert sorted(folds[1].train_idx.tolist()) == [0, 1, 3, 4, 5, 7]
    assert sorted(folds[1].test_idx.tolist()) == [2, 6]
    assert folds[0].test_end < folds[1].test_start <= date(2020, 4, 2)
