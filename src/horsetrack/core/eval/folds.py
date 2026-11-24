"""Walk-forward folds by date (problem statement section 4, owner decision D9).

A test window is `test_length` scheme units starting on a unit boundary: months, quarters and years
start on the first of a month aligned to `anchor_month`, weeks on a Monday, days on any day.
Successive windows start `step` units apart (never overlapping), and the first one starts on the
first boundary at least `min_train` units after the earliest date. Training takes every earlier row
at least `embargo_days` before the window's first scored row: all of them for an expanding window,
the last `min_train` units for a rolling one. Rows in `exclude` (a locked holdout) are in no fold,
and a window the exclusion empties is dropped.

The scheme itself (what a season is, the anchor month) is an owner decision and comes from config:
the `folds:` section of countries/<country>/config/folds.yaml, read by `load_country_scheme`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from horsetrack.core.config import load_country_yaml

UNITS = ("day", "week", "month", "quarter", "year")
WINDOWS = ("expanding", "rolling")
_MONTHS_PER_UNIT = {"month": 1, "quarter": 3, "year": 12}


@dataclass(frozen=True)
class FoldScheme:
    unit: str
    test_length: int
    step: int
    min_train: int
    window: str
    embargo_days: int
    anchor_month: int

    def __post_init__(self) -> None:
        if self.unit not in UNITS:
            raise ValueError(f"unit must be one of {UNITS}, not {self.unit!r}")
        if self.window not in WINDOWS:
            raise ValueError(f"window must be one of {WINDOWS}, not {self.window!r}")
        for name, low in (
            ("test_length", 1),
            ("step", 1),
            ("min_train", 0),
            ("embargo_days", 0),
            ("anchor_month", 1),
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < low:
                raise ValueError(f"{name} must be an integer of at least {low}, not {value!r}")
        if self.anchor_month > 12:
            raise ValueError(f"anchor_month must be 1 to 12, not {self.anchor_month}")
        if self.step < self.test_length:
            raise ValueError(
                f"step ({self.step}) is shorter than test_length ({self.test_length}): "
                "test windows would overlap and score a row twice"
            )

    @classmethod
    def from_mapping(cls, fields: Mapping[str, Any]) -> FoldScheme:
        """A scheme from config; every field is required, none is defaulted."""
        names = [f.name for f in dataclasses.fields(cls)]
        missing = [n for n in names if n not in fields]
        if missing:
            raise ValueError(f"fold scheme lacks field(s): {', '.join(missing)}")
        unknown = sorted(set(fields) - set(names))
        if unknown:
            raise ValueError(f"fold scheme has unknown field(s): {', '.join(unknown)}")
        return cls(**{n: fields[n] for n in names})


def _scheme_from_config(config: Any, source: str) -> FoldScheme:
    """The scheme in the `folds:` section of a parsed config; other sections are left alone."""
    folds = config.get("folds") if isinstance(config, Mapping) else None
    if not isinstance(folds, Mapping):
        raise ValueError(f"{source} has no `folds:` mapping")
    return FoldScheme.from_mapping(folds)


def load_scheme(path: str | Path) -> FoldScheme:
    """The scheme in the `folds:` section of a config file."""
    return _scheme_from_config(yaml.safe_load(Path(path).read_text()), str(path))


def load_country_scheme(country: str) -> FoldScheme:
    """The scheme in the `folds:` section of countries/<country>/config/folds.yaml."""
    return _scheme_from_config(load_country_yaml(country, "folds"), f"{country} folds config")


@dataclass(frozen=True, eq=False)
class Fold:
    """Positions into the dates passed to `walk_forward` (not index labels), sorted ascending.

    `train_end` is the last calendar day eligible for training; `test_start` and `test_end` are the
    first and last calendar days of the test window.
    """

    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end: date
    test_start: date
    test_end: date


def _add_months(d: date, n: int) -> date:
    years, month0 = divmod(d.month - 1 + n, 12)
    return date(d.year + years, month0 + 1, d.day)


def _add_units(boundary: date, unit: str, n: int) -> date:
    """`n` units after a unit boundary (month-based boundaries are always the first of a month)."""
    if unit == "day":
        return boundary + timedelta(days=n)
    if unit == "week":
        return boundary + timedelta(weeks=n)
    return _add_months(boundary, _MONTHS_PER_UNIT[unit] * n)


def _first_boundary(earliest: date, scheme: FoldScheme) -> date:
    """The first unit boundary on or after `earliest` plus `min_train` units."""
    unit, n = scheme.unit, scheme.min_train
    if unit == "day":
        return earliest + timedelta(days=n)
    if unit == "week":
        d = earliest + timedelta(weeks=n)
        return d + timedelta(days=(7 - d.weekday()) % 7)
    months = _MONTHS_PER_UNIT[unit]
    b = _add_months(date(earliest.year, earliest.month, 1), months * n + (earliest.day > 1))
    return _add_months(b, (scheme.anchor_month - b.month) % months)


def _day_numbers(dates: Any) -> np.ndarray:
    """Calendar days as datetime64[D], position for position."""
    values = pd.Series(dates).to_numpy()
    stamps = pd.DatetimeIndex(pd.to_datetime(values))
    if stamps.hasnans:
        raise ValueError("dates contain a missing value")
    if stamps.tz is not None:
        stamps = stamps.tz_localize(None)
    return stamps.normalize().to_numpy().astype("datetime64[D]")


def _day(d: Any) -> np.datetime64:
    return np.datetime64(pd.Timestamp(d).date(), "D")


def _to_date(d: np.datetime64) -> date:
    return pd.Timestamp(d).date()


def walk_forward(
    dates: Any,
    *,
    scheme: FoldScheme,
    exclude: tuple[Any, Any] | None = None,
) -> list[Fold]:
    """The walk-forward folds of `dates` under `scheme`, in date order.

    `exclude` is an inclusive (first, last) date range kept out of every fold, training and test.
    The calendar (earliest date, last window) is taken from the rows that are not excluded.
    """
    days = _day_numbers(dates)
    usable = np.ones(len(days), dtype=bool)
    if exclude is not None:
        lo_x, hi_x = _day(exclude[0]), _day(exclude[1])
        if hi_x < lo_x:
            raise ValueError(f"exclude range ends before it starts: {exclude!r}")
        usable &= (days < lo_x) | (days > hi_x)
    if not usable.any():
        return []

    earliest, last = _to_date(days[usable].min()), days[usable].max()
    first = _first_boundary(earliest, scheme)
    gap = np.timedelta64(max(scheme.embargo_days, 1), "D")
    one_day = np.timedelta64(1, "D")

    folds: list[Fold] = []
    k = 0
    while (start := _add_units(first, scheme.unit, k * scheme.step)) <= _to_date(last):
        k += 1
        lo = _day(start)
        hi = _day(_add_units(start, scheme.unit, scheme.test_length)) - one_day
        test = usable & (days >= lo) & (days <= hi)
        if not test.any():
            continue
        # the embargo runs from the first scored row; no row of the window itself trains
        train_end = min(days[test].min() - gap, lo - one_day)
        train = usable & (days <= train_end)
        if scheme.window == "rolling":
            train &= days >= _day(_add_units(start, scheme.unit, -scheme.min_train))
        folds.append(
            Fold(
                train_idx=np.flatnonzero(train),
                test_idx=np.flatnonzero(test),
                train_end=_to_date(train_end),
                test_start=start,
                test_end=_to_date(hi),
            )
        )
    return folds
