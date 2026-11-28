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
- The read boundary (#82, fix round 4, D14): no non-date column of a holdout day is materialised.
  The reader may read the date column first; the scan of the other columns (and of the columns
  `where` names) is scoped to development days before it is decoded, so no Python frame of the
  package or of pyarrow's Python layer ever holds a holdout row's other values, even in a row
  group mixing development and holdout days, and a row group holding one holdout day alone is
  never decoded. What it returns for development days is unchanged.

All data is synthetic and written under tmp_path; HORSETRACK_DATA_DIR points there.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import dataclasses
import os
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
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


# ---------------------------------------------------------------------------------------------
# #82, fix round 4: the read boundary itself, not only what is returned
# ---------------------------------------------------------------------------------------------

PACKAGE_DIR = os.path.realpath(Path(store.__file__).resolve().parents[4])  # src/horsetrack
PYARROW_DIR = os.path.realpath(Path(pa.__file__).resolve().parent)


def holds(value, hit, depth: int = 0) -> bool:
    """Whether `value` (Arrow, pandas or numpy data, a container or dataclass of them, or a
    scalar) holds a scalar that `hit` flags."""
    if depth > 4:
        return False
    if isinstance(value, (pa.Table, pa.RecordBatch)):
        return any(holds(column, hit, depth + 1) for column in value.columns)
    if isinstance(value, (pa.Array, pa.ChunkedArray)):
        return any(hit(x) for x in value.to_pylist())
    if isinstance(value, pa.Scalar):
        return hit(value.as_py())
    if isinstance(value, pd.DataFrame):
        return any(holds(value.iloc[:, i], hit, depth + 1) for i in range(value.shape[1]))
    if isinstance(value, (pd.Series, pd.Index, np.ndarray)):
        return any(hit(x) for x in np.asarray(value, dtype=object).ravel().tolist())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(holds(v, hit, depth + 1) for v in value)
    if isinstance(value, dict):
        return any(holds(v, hit, depth + 1) for v in value.values())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        return any(holds(getattr(value, f.name), hit, depth + 1) for f in fields)
    return bool(hit(value))


class ReadTrace:
    """While the block runs, every local of every Python frame of the package or of pyarrow's
    Python layer, and every value such a frame returns or yields, searched for a scalar `hit`
    flags; `found` names the frames and lines where one was held."""

    def __init__(self, hit) -> None:
        self.hit, self.found = hit, []
        self._watched: dict[str, bool] = {}

    def __enter__(self) -> ReadTrace:
        self._previous = sys.gettrace()
        sys.settrace(self._call)
        return self

    def __exit__(self, *exc) -> bool:
        sys.settrace(self._previous)
        return False

    def _call(self, frame, event, arg):
        name = frame.f_code.co_filename
        if name not in self._watched:
            path = os.path.realpath(name)
            self._watched[name] = path.startswith((PACKAGE_DIR, PYARROW_DIR))
        return self._line if self._watched[name] else None

    def _line(self, frame, event, arg):
        if event in ("line", "return") and len(self.found) < 20:
            values = [*frame.f_locals.values(), *([arg] if event == "return" else [])]
            try:
                held = any(holds(value, self.hit) for value in values)
            except Exception:  # a value the search cannot read holds nothing it can flag
                held = False
            if held:
                where = f"{Path(frame.f_code.co_filename).name}:{frame.f_lineno}"
                self.found.append(f"{where} {frame.f_code.co_name}")
        return self._line


def unreadable_row_group(path: Path, group: int, keep) -> None:
    """Overwrite the data of row group `group`'s column chunks, those named in `keep` aside, so
    that decoding one fails; the footer (schema, row-group statistics) stays."""
    meta = pq.read_metadata(path).row_group(group)
    data = bytearray(path.read_bytes())
    for j in range(meta.num_columns):
        column = meta.column(j)
        if column.path_in_schema in keep:
            continue
        start = (
            column.dictionary_page_offset if column.has_dictionary_page else column.data_page_offset
        )
        data[start : start + column.total_compressed_size] = b"\xff" * column.total_compressed_size
    path.write_bytes(bytes(data))


POISON_VENUE = "PSN"  # every holdout row's race_id names it; no development row's does
POISON_FROM = 7_700  # every holdout row's numbers start here; no development row's does


def boundary_rows(day: date, n_races: int, holdout: bool, k: int) -> list[dict]:
    """Runner rows of one day; a holdout day's every non-date value is poison."""
    rows = []
    for r in range(n_races):
        venue = POISON_VENUE if holdout else "HSN"
        race_id = f"jra:{day.isoformat()}:{venue}:{r + 1:02d}"
        for h in range(1, 6):
            base = POISON_FROM + 100 * k + 10 * r + h if holdout else h
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": day,
                    "horse_no": base,
                    "win_odds": base + 0.25 if holdout else 1.5 + 2.5 * h + r,
                    "finish_pos": base,
                    "is_scratched": False,
                    "dead_heat": False,
                    "win_fav_rank": base,
                }
            )
    return rows


def test_no_non_date_column_of_a_holdout_day_is_materialised(tmp_path, monkeypatch):
    """#82, fix round 4 (C): the store's runners in four row groups: development days, a group
    mixing a development day with a holdout day, a group holding a holdout day alone (its
    non-date columns cannot be decoded), development days again. Reads across them return the
    development rows as before, decode nothing of the lone holdout group but its dates, and no
    frame of the package or of pyarrow's Python layer ever holds a holdout row's other values."""
    data = (tmp_path / "data").resolve()
    monkeypatch.setenv(DATA_VAR, str(data))
    folder = data / "processed" / "jra"
    folder.mkdir(parents=True)
    path = folder / "runners.parquet"
    dev = [date(2026, 6, 6), date(2026, 6, 13), date(2026, 6, 27)]
    mixed, alone = date(2026, 6, 20), date(2026, 6, 21)
    groups = [
        boundary_rows(dev[0], 2, False, 0),
        boundary_rows(dev[1], 3, False, 1) + boundary_rows(mixed, 2, True, 2),
        boundary_rows(alone, 2, True, 3),
        boundary_rows(dev[2], 1, False, 4),
    ]
    with pq.ParquetWriter(path, RUNNERS_SCHEMA) as writer:
        for group in groups:
            writer.write_table(pa.Table.from_pylist(group, schema=RUNNERS_SCHEMA))
    unreadable_row_group(path, 2, keep=("race_date",))
    patch_holdout(monkeypatch, [mixed, alone])
    # the fixture: four row groups; the lone holdout group's dates read, its other columns do not
    assert pq.read_metadata(path).num_row_groups == 4
    with pq.ParquetFile(path) as file:
        assert file.read_row_group(2, columns=["race_date"]).num_rows == len(groups[2])
        with pytest.raises((OSError, pa.ArrowException)):
            file.read_row_group(2, columns=["win_odds"])
    holdout_rows = [row for group in groups for row in group if row["race_date"] in (mixed, alone)]
    dev_rows = [row for group in groups for row in group if row["race_date"] in dev]
    poison = {row[c] for row in holdout_rows for c in ("horse_no", "win_odds", "finish_pos")}
    assert min(poison) >= POISON_FROM > max(row["win_odds"] for row in dev_rows)

    def hit(x) -> bool:
        if isinstance(x, str):
            return f":{POISON_VENUE}:" in x
        return isinstance(x, (int, float, np.number)) and not isinstance(x, bool) and x in poison

    def want(columns, end, *, where=lambda row: True) -> Counter:
        return Counter(
            tuple(row[c] for c in columns)
            for row in dev_rows
            if row["race_date"] <= end and where(row)
        )

    reads = [  # (columns, end, where as an expression and as a row predicate)
        (["race_id", "win_odds", "finish_pos"], date(2026, 6, 30), None, lambda row: True),
        (["race_date", "horse_no"], date(2026, 6, 20), None, lambda row: True),
        (
            ["race_id"],
            date(2026, 6, 30),
            ds.field("finish_pos") >= 2,
            lambda row: row["finish_pos"] >= 2,
        ),
    ]
    got = []
    with ReadTrace(hit) as trace:
        try:
            for columns, end, where, _ in reads:
                kwargs = {} if where is None else {"where": where}
                got.append(
                    read_store_table(
                        "runners",
                        columns=columns,
                        start=date(2026, 6, 1),
                        end=end,
                        touches_price=True,
                        **kwargs,
                    )
                )
        except (OSError, pa.ArrowException) as exc:
            pytest.fail(f"a non-date column of the lone holdout row group was decoded: {exc!r}")
    assert not trace.found, f"a holdout day's non-date values were materialised: {trace.found}"
    for table, (columns, end, _, keep) in zip(got, reads, strict=True):
        assert table.column_names == columns
        rows = Counter(zip(*(table.column(c).to_pylist() for c in columns), strict=True))
        assert rows == want(columns, end, where=keep), columns
