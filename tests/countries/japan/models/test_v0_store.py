"""The v0 scoped store reader (ticket #114, outline 1).

Contract pinned here (`horsetrack.countries.japan.models.v0.store`):
- `PRICE_ERA_START = date(2014, 6, 7)` (D10); `PreReformPriceError` is a ValueError.
- `read_store_table(name, *, columns, start, end, touches_price)` reads
  `processed_dir('jra')/<name>.parquet` and returns a pyarrow.Table holding exactly `columns`, in
  the order asked (no whole-table read: None or an empty list is a ValueError; an unknown column is
  a KeyError naming it). Rows are those with `start <= race_date <= end`, whether or not
  race_date is among the columns asked for.
- No holdout day is ever returned. Holdout days come from #71's
  `horsetrack.core.eval.holdout.is_holdout_day(country, day)` with country 'japan'. The tests
  patch it both on the holdout module and on `store` (``store.is_holdout_day``), so either lookup
  (`from horsetrack.core.eval.holdout import is_holdout_day` or a module attribute) is honoured;
  it must be looked up at call time, not cached at import.
- `touches_price=True` with `start < PRICE_ERA_START` raises PreReformPriceError;
  `touches_price=False` reads pre-reform rows.

All data is synthetic and written under tmp_path; HORSETRACK_DATA_DIR points there.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

from collections import Counter
from datetime import date, timedelta

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core.eval import holdout
from horsetrack.countries.japan.models.v0 import store
from horsetrack.countries.japan.models.v0.store import (
    PRICE_ERA_START,
    PreReformPriceError,
    read_store_table,
)

DATA_VAR = "HORSETRACK_DATA_DIR"
RUNNERS_SCHEMA = pa.schema(
    [
        ("race_id", pa.string()),
        ("race_date", pa.date32()),
        ("horse_no", pa.int32()),
        ("win_odds", pa.float64()),
        ("finish_pos", pa.int32()),
        ("is_scratched", pa.bool_()),
        ("dead_heat", pa.bool_()),
        ("win_fav_rank", pa.int32()),
    ]
)
RACES_SCHEMA = pa.schema(
    [("race_id", pa.string()), ("race_date", pa.date32()), ("venue", pa.string())]
)


def runner_rows(days, races_per_day=None):
    """Store-shaped runner rows: day k gets a distinct number of races (so counts discriminate),
    each with a distinct number of runners."""
    rows = []
    for k, day in enumerate(days):
        n_races = races_per_day if races_per_day is not None else 1 + k % 3
        for r in range(n_races):
            race_id = f"jra:{day.isoformat()}:HSN:{r + 1:02d}"
            n = 5 + (k + r) % 4
            for h in range(1, n + 1):
                rows.append(
                    {
                        "race_id": race_id,
                        "race_date": day,
                        "horse_no": h,
                        "win_odds": 1.5 + 2.5 * h,
                        "finish_pos": h,
                        "is_scratched": False,
                        "dead_heat": False,
                        "win_fav_rank": h,
                    }
                )
    return pd.DataFrame(rows)


def write_store(tmp_path, monkeypatch, runners, races=None):
    data = (tmp_path / "data").resolve()
    monkeypatch.setenv(DATA_VAR, str(data))
    jra = data / "processed" / "jra"
    jra.mkdir(parents=True)
    # Shuffle so nothing can rely on the file being in date order.
    runners = runners.sample(frac=1.0, random_state=7).reset_index(drop=True)
    pq.write_table(
        pa.Table.from_pandas(runners, schema=RUNNERS_SCHEMA, preserve_index=False),
        jra / "runners.parquet",
    )
    if races is not None:
        pq.write_table(
            pa.Table.from_pandas(races, schema=RACES_SCHEMA, preserve_index=False),
            jra / "races.parquet",
        )
    return data


def patch_holdout(monkeypatch, holdout_days):
    """Mark exactly `holdout_days` as holdout, wherever store.py looks is_holdout_day up."""
    marked = {pd.Timestamp(d).date() for d in holdout_days}
    seen = []

    def fake(country, day):
        seen.append(country)
        return pd.Timestamp(day).date() in marked

    monkeypatch.setattr(holdout, "is_holdout_day", fake)
    monkeypatch.setattr(store, "is_holdout_day", fake, raising=False)
    return seen


def dates_of(table):
    return sorted({pd.Timestamp(d).date() for d in table.column("race_date").to_pylist()})


DAYS_0606_0609 = [date(2014, 6, 6) + timedelta(days=k) for k in range(4)]


# ---------------------------------------------------------------------------------------------
# DoD: read_store_table requires its columns (no whole-table read)
# ---------------------------------------------------------------------------------------------


def test_price_era_start_is_d10s_date():
    assert PRICE_ERA_START == date(2014, 6, 7)
    assert issubclass(PreReformPriceError, ValueError)


@pytest.mark.parametrize("columns", [None, [], ()])
def test_columns_are_required(tmp_path, monkeypatch, columns):
    write_store(tmp_path, monkeypatch, runner_rows(DAYS_0606_0609))
    with pytest.raises(ValueError):
        read_store_table(
            "runners",
            columns=columns,
            start=date(2014, 6, 7),
            end=date(2014, 6, 9),
            touches_price=True,
        )


@pytest.mark.parametrize(
    "columns",
    [
        ["race_id", "race_date", "win_odds"],
        ["win_odds", "race_id"],
        ["horse_no"],
        ["race_date", "finish_pos", "dead_heat", "is_scratched"],
    ],
)
def test_only_the_named_columns_are_returned(tmp_path, monkeypatch, columns):
    runners = runner_rows(DAYS_0606_0609)
    write_store(tmp_path, monkeypatch, runners)
    table = read_store_table(
        "runners",
        columns=columns,
        start=date(2014, 6, 7),
        end=date(2014, 6, 8),
        touches_price=True,
    )
    assert isinstance(table, pa.Table)
    assert table.column_names == list(columns)
    in_range = runners[runners["race_date"].between(date(2014, 6, 7), date(2014, 6, 8))]
    assert table.num_rows == len(in_range)


@pytest.mark.parametrize("missing", ["no_such_column", "win_odds_final", "Race_Date"])
def test_an_unknown_column_is_a_key_error_naming_it(tmp_path, monkeypatch, missing):
    write_store(tmp_path, monkeypatch, runner_rows(DAYS_0606_0609))
    with pytest.raises(KeyError, match=missing):
        read_store_table(
            "runners",
            columns=["race_id", missing, "win_odds"],
            start=date(2014, 6, 7),
            end=date(2014, 6, 9),
            touches_price=True,
        )


def test_the_races_table_is_read_by_name(tmp_path, monkeypatch):
    days = [date(2019, 3, 2) + timedelta(days=k) for k in range(5)]
    races = pd.DataFrame(
        {
            "race_id": [f"jra:{d.isoformat()}:TOK:01" for d in days],
            "race_date": days,
            "venue": ["TOK", "KYO", "HSN", "NAK", "SAP"],
        }
    )
    write_store(tmp_path, monkeypatch, runner_rows(days), races=races)
    table = read_store_table(
        "races",
        columns=["venue", "race_id"],
        start=days[1],
        end=days[3],
        touches_price=False,
    )
    assert table.column_names == ["venue", "race_id"]
    assert sorted(table.column("venue").to_pylist()) == sorted(["KYO", "HSN", "NAK"])


# ---------------------------------------------------------------------------------------------
# DoD: pushes the date range into the scan (inclusive at both ends)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "start, end, touches_price, expected",
    [
        # the ticket's case: 2014-06-07..2014-06-08 gives two dates
        (date(2014, 6, 7), date(2014, 6, 8), True, [date(2014, 6, 7), date(2014, 6, 8)]),
        (date(2014, 6, 7), date(2014, 6, 7), True, [date(2014, 6, 7)]),
        (date(2014, 6, 8), date(2014, 6, 9), True, [date(2014, 6, 8), date(2014, 6, 9)]),
        (date(2014, 6, 9), date(2014, 6, 30), True, [date(2014, 6, 9)]),
        (date(2014, 6, 6), date(2014, 6, 7), False, [date(2014, 6, 6), date(2014, 6, 7)]),
        (date(2014, 5, 1), date(2014, 6, 5), False, []),
    ],
)
def test_the_date_range_is_inclusive(tmp_path, monkeypatch, start, end, touches_price, expected):
    runners = runner_rows(DAYS_0606_0609)
    write_store(tmp_path, monkeypatch, runners)
    table = read_store_table(
        "runners",
        columns=["race_id", "race_date", "horse_no"],
        start=start,
        end=end,
        touches_price=touches_price,
    )
    assert dates_of(table) == expected
    want = runners[runners["race_date"].isin(expected)]
    assert Counter(table.column("race_id").to_pylist()) == Counter(want["race_id"])


def test_the_date_range_applies_when_race_date_is_not_asked_for(tmp_path, monkeypatch):
    runners = runner_rows(DAYS_0606_0609)
    write_store(tmp_path, monkeypatch, runners)
    table = read_store_table(
        "runners",
        columns=["race_id", "win_odds"],
        start=date(2014, 6, 8),
        end=date(2014, 6, 8),
        touches_price=True,
    )
    assert table.column_names == ["race_id", "win_odds"]
    want = runners[runners["race_date"] == date(2014, 6, 8)]
    assert Counter(table.column("race_id").to_pylist()) == Counter(want["race_id"])


# ---------------------------------------------------------------------------------------------
# DoD: never returns a holdout day
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("touches_price", [True, False])
def test_holdout_days_are_dropped(tmp_path, monkeypatch, touches_price):
    days = [date(2026, 11, 27), date(2026, 11, 28), date(2026, 11, 29)]
    runners = runner_rows(days)
    write_store(tmp_path, monkeypatch, runners)
    seen = patch_holdout(monkeypatch, [date(2026, 11, 28)])
    table = read_store_table(
        "runners",
        columns=["race_id", "race_date"],
        start=date(2026, 11, 1),
        end=date(2026, 11, 30),
        touches_price=touches_price,
    )
    assert dates_of(table) == [date(2026, 11, 27), date(2026, 11, 29)]
    want = runners[runners["race_date"] != date(2026, 11, 28)]
    assert Counter(table.column("race_id").to_pylist()) == Counter(want["race_id"])
    assert seen and set(seen) == {"japan"}


def test_a_holdout_day_is_dropped_even_when_race_date_is_not_asked_for(tmp_path, monkeypatch):
    days = [date(2026, 12, 5), date(2026, 12, 6)]
    runners = runner_rows(days)
    write_store(tmp_path, monkeypatch, runners)
    patch_holdout(monkeypatch, [date(2026, 12, 6)])
    table = read_store_table(
        "runners",
        columns=["race_id"],
        start=days[0],
        end=days[1],
        touches_price=True,
    )
    want = runners[runners["race_date"] == date(2026, 12, 5)]
    assert Counter(table.column("race_id").to_pylist()) == Counter(want["race_id"])


def test_the_committed_holdout_definition_is_applied(tmp_path, monkeypatch):
    """Unpatched: the days dropped are exactly those #71's committed rule marks."""
    days = [date(2026, 11, 20) + timedelta(days=k) for k in range(24)]
    runners = runner_rows(days, races_per_day=1)
    write_store(tmp_path, monkeypatch, runners)
    kept = [d for d in days if not holdout.is_holdout_day("japan", d)]
    assert 0 < len(kept) < len(days)  # the oracle marks some days and not others
    table = read_store_table(
        "runners",
        columns=["race_date", "race_id"],
        start=days[0],
        end=days[-1],
        touches_price=True,
    )
    assert dates_of(table) == kept


# ---------------------------------------------------------------------------------------------
# DoD: refuses a price read before 2014-06-07 (D10)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("start", [date(2014, 1, 1), date(2014, 6, 6), date(2010, 1, 5)])
def test_a_price_read_before_2014_06_07_is_refused(tmp_path, monkeypatch, start):
    days = [date(2014, 1, 4), date(2014, 6, 1), *DAYS_0606_0609]
    runners = runner_rows(days)
    write_store(tmp_path, monkeypatch, runners)
    with pytest.raises(PreReformPriceError):
        read_store_table(
            "runners",
            columns=["race_id", "race_date", "win_odds"],
            start=start,
            end=date(2014, 6, 9),
            touches_price=True,
        )
    table = read_store_table(
        "runners",
        columns=["race_id", "race_date", "finish_pos"],
        start=start,
        end=date(2014, 6, 9),
        touches_price=False,
    )
    expected = [d for d in days if d >= start]
    assert dates_of(table) == expected
    assert date(2014, 6, 6) in dates_of(table)


def test_a_price_read_from_the_era_start_is_allowed(tmp_path, monkeypatch):
    runners = runner_rows([date(2014, 6, 1), *DAYS_0606_0609])
    write_store(tmp_path, monkeypatch, runners)
    table = read_store_table(
        "runners",
        columns=["race_date", "win_odds"],
        start=PRICE_ERA_START,
        end=date(2014, 6, 9),
        touches_price=True,
    )
    assert dates_of(table) == [date(2014, 6, 7), date(2014, 6, 8), date(2014, 6, 9)]
