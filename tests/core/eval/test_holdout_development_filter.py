"""Ticket #415: the development predicate as a dataset expression, built from calendar dates.

Contract pinned here (the coder implements what these tests call):

- ``horsetrack.core.eval.holdout.development_filter(country, start, end, *, column="race_date")``
  returns a ``pyarrow.dataset.Expression`` that is true exactly for the rows whose ``column`` (a
  date32 race day) lies in ``start <= day <= end`` (both inclusive) and is not a holdout day of
  ``country`` by the committed definition (#71's rule, the same as ``is_holdout_day``). It takes
  no data: the holdout days are found from the calendar between the dates, so a scan can apply
  it before any row (outcomes included) is read. A row with a null day is not a development row.
- A country without a valid holdout definition raises ``HoldoutConfigError`` (no default), as
  ``is_holdout_day`` does: Hong Kong's committed ``start`` is still null.

The oracle below recomputes the rule from the committed holdout.yaml with hashlib; it never calls
the code under test. Nothing here opens the holdout or touches a holdout log.
"""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pytest
import yaml

from horsetrack.core.eval.holdout import HoldoutConfigError, development_filter

REPO_ROOT = Path(__file__).resolve().parents[3]
JAPAN_YAML = REPO_ROOT / "src" / "horsetrack" / "countries" / "japan" / "config" / "holdout.yaml"


def _japan_definition() -> tuple[date, str]:
    section = yaml.safe_load(JAPAN_YAML.read_text(encoding="utf-8"))["holdout"]
    start = section["start"]
    start = start if isinstance(start, date) else date.fromisoformat(str(start))
    return start, str(section["seed"])


HOLDOUT_START, SEED = _japan_definition()


def oracle_is_holdout(day: date) -> bool:
    digest = hashlib.sha256(f"{SEED}:{day.isoformat()}".encode()).hexdigest()
    return day >= HOLDOUT_START and int(digest, 16) % 2 == 0


def calendar(first: date, last: date) -> list[date]:
    return [first + timedelta(days=k) for k in range((last - first).days + 1)]


# Every calendar day from well before the holdout start to well after it.
ALL_DAYS = calendar(date(2026, 10, 1), date(2027, 2, 28))
HELD = [d for d in ALL_DAYS if oracle_is_holdout(d)]
DEV_AFTER_START = [d for d in ALL_DAYS if d >= HOLDOUT_START and not oracle_is_holdout(d)]
# A day before the start whose hash is even: it would be held out were it not before the start.
EVEN_BEFORE_START = [
    d
    for d in ALL_DAYS
    if d < HOLDOUT_START
    and int(hashlib.sha256(f"{SEED}:{d.isoformat()}".encode()).hexdigest(), 16) % 2 == 0
]


# The calendar interleaves development and holdout days, and has a pre-start day that the hash
# alone would hold out.
assert len(HELD) > 20 and len(DEV_AFTER_START) > 20 and EVEN_BEFORE_START


def survivors(expr, days, column="race_date") -> list[date]:
    table = pa.table({column: pa.array(days, pa.date32()), "n": list(range(len(days)))})
    out = ds.dataset(table).to_table(filter=expr)
    return sorted(out.column(column).to_pylist())


def _ranges():
    first_held, last_held = HELD[0], HELD[-1]
    second_held = HELD[1]
    dev0, dev1 = DEV_AFTER_START[0], DEV_AFTER_START[-1]
    return [
        pytest.param(ALL_DAYS[0], ALL_DAYS[-1], id="whole-calendar"),
        pytest.param(HOLDOUT_START, HOLDOUT_START, id="the-holdout-start-day-alone"),
        pytest.param(HOLDOUT_START - timedelta(days=1), HOLDOUT_START, id="across-the-start"),
        pytest.param(first_held, second_held, id="both-ends-holdout"),
        pytest.param(first_held, first_held, id="one-holdout-day-alone"),
        pytest.param(dev0, dev1, id="both-ends-development-after-start"),
        pytest.param(EVEN_BEFORE_START[-1], first_held, id="even-pre-start-day-to-first-holdout"),
        pytest.param(date(2026, 10, 1), date(2026, 11, 15), id="all-before-the-start"),
        pytest.param(last_held, ALL_DAYS[-1], id="from-the-last-holdout-day"),
        pytest.param(date(2026, 12, 10), date(2026, 12, 9), id="empty-range"),
    ]


@pytest.mark.parametrize(("start", "end"), _ranges())
def test_the_filter_keeps_exactly_the_development_days_in_the_inclusive_range(start, end):
    expr = development_filter("japan", start, end)
    assert isinstance(expr, ds.Expression)
    expected = [d for d in ALL_DAYS if start <= d <= end and not oracle_is_holdout(d)]
    assert survivors(expr, ALL_DAYS) == expected


@pytest.mark.parametrize("column", ["race_date", "day", "meeting_date"])
def test_the_filter_reads_the_named_day_column(column):
    start, end = HOLDOUT_START - timedelta(days=3), HELD[6]
    expr = development_filter("japan", start, end, column=column)
    expected = [d for d in ALL_DAYS if start <= d <= end and not oracle_is_holdout(d)]
    assert survivors(expr, ALL_DAYS, column=column) == expected


def test_a_pre_start_day_is_development_whatever_its_hash():
    day = EVEN_BEFORE_START[-1]
    assert survivors(development_filter("japan", day, day), ALL_DAYS) == [day]


def test_a_null_day_is_not_a_development_row():
    days = [HOLDOUT_START, None, DEV_AFTER_START[1], HELD[0], None]
    table = pa.table({"race_date": pa.array(days, pa.date32())})
    out = ds.dataset(table).to_table(filter=development_filter("japan", ALL_DAYS[0], ALL_DAYS[-1]))
    assert out.column("race_date").to_pylist() == [HOLDOUT_START, DEV_AFTER_START[1]]


def test_the_filter_needs_no_data_and_spans_long_ranges():
    """Built from the calendar alone: a decade-plus range is answered for days no store holds."""
    start, end = date(2014, 6, 7), date(2028, 12, 31)
    probe = [start, date(2020, 1, 4), HOLDOUT_START, *HELD[:5], *DEV_AFTER_START[:5], end]
    far = calendar(date(2028, 12, 1), end)
    probe += far
    expected = sorted({d for d in probe if not oracle_is_holdout(d)})
    assert survivors(development_filter("japan", start, end), sorted(set(probe))) == expected
    assert any(oracle_is_holdout(d) for d in far)


def test_a_country_without_a_valid_definition_has_no_filter():
    with pytest.raises(HoldoutConfigError):
        development_filter("hongkong", date(2026, 1, 1), date(2026, 12, 31))
