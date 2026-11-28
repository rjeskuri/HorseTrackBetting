"""The v0 scoped store reader takes a `source` (ticket #372).

Contract pinned here (`horsetrack.countries.japan.models.v0.store.read_store_table`):
- A new keyword `source: str = 'jra'`: the table is read from
  `processed_dir(source)/<name>.parquet`. Without it, the read is the `jra` folder, as before.
- Whatever the source, holdout days (`core.eval.holdout.is_holdout_day('japan', d)`) are dropped and
  a price read starting before 2014-06-07 (D10) is refused with `PreReformPriceError`, before any
  file is opened.

All data is synthetic, written under tmp_path; HORSETRACK_DATA_DIR points there. Each test writes a
distinct `jra` and a distinct `jravan` file so the folder that was read shows in the rows.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

from collections import Counter
from datetime import date, timedelta

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from horsetrack.core.eval import holdout
from horsetrack.countries.japan.models.v0.store import PreReformPriceError, read_store_table

DATA_VAR = "HORSETRACK_DATA_DIR"
SCHEMA = pa.schema([("race_id", pa.string()), ("race_date", pa.date32()), ("venue", pa.string())])


def make_rows(source_tag, days, per_day):
    """Rows whose race_id and venue carry the source tag; `per_day(k)` races on day k."""
    rows = []
    for k, day in enumerate(days):
        for r in range(per_day(k)):
            rows.append(
                {
                    "race_id": f"{source_tag}:{day.isoformat()}:{r + 1:02d}",
                    "race_date": day,
                    "venue": source_tag.upper(),
                }
            )
    return rows


def write_table(data, source, name, rows):
    folder = data / "processed" / source
    folder.mkdir(parents=True, exist_ok=True)
    columns = {f.name: [row[f.name] for row in rows] for f in SCHEMA}
    pq.write_table(pa.table(columns, schema=SCHEMA), folder / f"{name}.parquet")


def data_dir(tmp_path, monkeypatch):
    data = (tmp_path / "data").resolve()
    monkeypatch.setenv(DATA_VAR, str(data))
    return data


def read(source_kwargs, days, *, name="races", touches_price=False):
    return read_store_table(
        name,
        columns=["race_id", "race_date", "venue"],
        start=days[0],
        end=days[-1],
        touches_price=touches_price,
        **source_kwargs,
    )


def race_ids(table):
    return Counter(table.column("race_id").to_pylist())


# Days well inside the price era and before the holdout start (2026-11-28): none is held out.
OPEN_DAYS = [date(2019, 3, 2) + timedelta(days=k) for k in range(6)]
# Days from the holdout start on: the seeded half of race days is held out, the rest is open.
LATE_DAYS = [date(2026, 11, 28) + timedelta(days=k) for k in range(24)]


# ---------------------------------------------------------------------------------------------
# DoD: source='jravan' reads processed/jravan/<name>.parquet
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["jravan", "jrdb", "nar"])
@pytest.mark.parametrize("name", ["races", "runners"])
def test_another_source_reads_its_own_processed_folder(tmp_path, monkeypatch, source, name):
    data = data_dir(tmp_path, monkeypatch)
    # Different row counts per source and per day, so neither a hard-coded folder nor a
    # concatenation of folders can give the expected rows.
    jra_rows = make_rows("jra", OPEN_DAYS, lambda k: 1 + k % 2)
    other_rows = make_rows(source, OPEN_DAYS, lambda k: 2 + k % 3)
    write_table(data, "jra", name, jra_rows)
    write_table(data, source, name, other_rows)

    table = read({"source": source}, OPEN_DAYS, name=name)

    assert race_ids(table) == Counter(r["race_id"] for r in other_rows)
    assert set(table.column("venue").to_pylist()) == {source.upper()}
    # the date range still scopes the read inside the other source
    narrow = read_store_table(
        name,
        columns=["race_id", "race_date"],
        start=OPEN_DAYS[1],
        end=OPEN_DAYS[3],
        touches_price=True,
        source=source,
    )
    want = [r for r in other_rows if OPEN_DAYS[1] <= r["race_date"] <= OPEN_DAYS[3]]
    assert narrow.column_names == ["race_id", "race_date"]
    assert race_ids(narrow) == Counter(r["race_id"] for r in want)


# ---------------------------------------------------------------------------------------------
# DoD: holdout days are dropped for another source
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("touches_price", [True, False])
def test_holdout_days_are_dropped_for_another_source(tmp_path, monkeypatch, touches_price):
    data = data_dir(tmp_path, monkeypatch)
    held = [d for d in LATE_DAYS if holdout.is_holdout_day("japan", d)]
    open_ = [d for d in LATE_DAYS if not holdout.is_holdout_day("japan", d)]
    assert held and open_  # the committed rule marks some days and not others

    jravan_rows = make_rows("jravan", LATE_DAYS, lambda k: 1 + k % 3)
    write_table(data, "jravan", "races", jravan_rows)
    write_table(data, "jra", "races", make_rows("jra", LATE_DAYS, lambda k: 2))

    table = read({"source": "jravan"}, LATE_DAYS, touches_price=touches_price)

    got_days = sorted(set(table.column("race_date").to_pylist()))
    assert got_days == open_
    assert not set(got_days) & set(held)
    want = [r for r in jravan_rows if r["race_date"] in set(open_)]
    assert race_ids(table) == Counter(r["race_id"] for r in want)


# ---------------------------------------------------------------------------------------------
# DoD: the D10 guard is on for another source
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("start", [date(2014, 6, 6), date(2014, 1, 1), date(2010, 1, 5)])
def test_a_price_read_before_the_price_era_is_refused_for_another_source(
    tmp_path, monkeypatch, start
):
    data = data_dir(tmp_path, monkeypatch)
    days = [date(2014, 1, 4), date(2014, 6, 6), date(2014, 6, 7), date(2014, 6, 8)]
    rows = make_rows("jravan", days, lambda k: 1 + k % 2)
    write_table(data, "jravan", "races", rows)
    write_table(data, "jra", "races", make_rows("jra", days, lambda k: 1))
    end = days[-1]

    with pytest.raises(PreReformPriceError):
        read_store_table(
            "races",
            columns=["race_id"],
            start=start,
            end=end,
            touches_price=True,
            source="jravan",
        )

    # the same range is readable when it touches no price, from jravan's rows
    table = read_store_table(
        "races",
        columns=["race_id", "race_date"],
        start=start,
        end=end,
        touches_price=False,
        source="jravan",
    )
    want = [r for r in rows if r["race_date"] >= start]
    assert race_ids(table) == Counter(r["race_id"] for r in want)

    # and a price read from the era start is allowed
    allowed = read_store_table(
        "races",
        columns=["race_id"],
        start=date(2014, 6, 7),
        end=end,
        touches_price=True,
        source="jravan",
    )
    want = [r for r in rows if r["race_date"] >= date(2014, 6, 7)]
    assert race_ids(allowed) == Counter(r["race_id"] for r in want)

    # the refusal fires before any read: a source with no folder at all still raises the D10 error
    with pytest.raises(PreReformPriceError):
        read_store_table(
            "races",
            columns=["race_id"],
            start=start,
            end=end,
            touches_price=True,
            source="nar",
        )


# ---------------------------------------------------------------------------------------------
# DoD: every existing caller reads the same rows as before (the default is jra)
# ---------------------------------------------------------------------------------------------


def test_the_default_source_is_still_jra(tmp_path, monkeypatch):
    data = data_dir(tmp_path, monkeypatch)
    jra_rows = make_rows("jra", OPEN_DAYS, lambda k: 1 + k % 3)
    write_table(data, "jra", "races", jra_rows)
    write_table(data, "jravan", "races", make_rows("jravan", OPEN_DAYS, lambda k: 4))

    default = read({}, OPEN_DAYS)

    assert race_ids(default) == Counter(r["race_id"] for r in jra_rows)
    assert set(default.column("venue").to_pylist()) == {"JRA"}
    # spelling the default out gives the same rows
    explicit = read({"source": "jra"}, OPEN_DAYS)
    assert explicit.equals(default)


# ---------------------------------------------------------------------------------------------
# #404 review: a holdout day's outcome and price columns are never scanned, only its race_date
# ---------------------------------------------------------------------------------------------


class _ScanSpy:
    """A pyarrow dataset that records every scan made through it: the columns asked for, the
    row filter in force (its own `filter()` narrowing combined with the scan's) and the days that
    filter lets through. The days are read when the scan is made, while the reader's handle is
    still open (#415's reader serves the dataset from a memory-mapped snapshot it closes on
    return), by the same filter over race_date alone."""

    def __init__(self, real, log, narrowed=None):
        self._real, self._log, self._narrowed = real, log, narrowed

    def __getattr__(self, name):
        return getattr(self._real, name)

    def _combined(self, filter):
        if self._narrowed is None:
            return filter
        return self._narrowed if filter is None else self._narrowed & filter

    def _note(self, columns, filter):
        reached = self._real.to_table(columns=["race_date"], filter=self._combined(filter))
        days = set(reached.column("race_date").to_pylist())
        names = _column_names(columns, self._real)
        self._log.append({"columns": columns, "names": names, "days": days})

    def _scan(self, method, columns=None, filter=None, **kwargs):
        self._note(columns, filter)
        return getattr(self._real, method)(columns=columns, filter=filter, **kwargs)

    def to_table(self, columns=None, filter=None, **kwargs):
        return self._scan("to_table", columns, filter, **kwargs)

    def to_batches(self, columns=None, filter=None, **kwargs):
        return self._scan("to_batches", columns, filter, **kwargs)

    def scanner(self, columns=None, filter=None, **kwargs):
        return self._scan("scanner", columns, filter, **kwargs)

    def head(self, num_rows, columns=None, filter=None, **kwargs):
        self._note(columns, filter)
        return self._real.head(num_rows, columns=columns, filter=filter, **kwargs)

    def filter(self, expression):
        return _ScanSpy(self._real.filter(expression), self._log, self._combined(expression))


def _column_names(columns, dataset) -> set[str]:
    if columns is None:
        return set(dataset.schema.names)  # a scan without a projection reads every column
    if isinstance(columns, dict):
        return set(columns)
    return set(columns)


@pytest.mark.parametrize(
    ("source", "columns", "where"),
    [
        ("jra", ["race_id", "venue"], None),
        ("jravan", ["race_id", "venue"], None),
        ("jravan", ["race_id", "race_date", "venue"], None),
        ("jravan", ["venue", "race_id"], "narrowed"),
    ],
)
def test_a_holdout_days_other_columns_are_never_scanned(
    tmp_path, monkeypatch, source, columns, where
):
    """Every scan that reads a column other than race_date runs with a row filter that already
    excludes the holdout days, so no outcome or price of a held day is materialised; a scan of
    race_date alone (to find the held days) is allowed."""
    data = data_dir(tmp_path, monkeypatch)
    held = [d for d in LATE_DAYS if holdout.is_holdout_day("japan", d)]
    open_ = [d for d in LATE_DAYS if not holdout.is_holdout_day("japan", d)]
    assert held and open_
    rows = make_rows(source, LATE_DAYS, lambda k: 1 + k % 3)
    write_table(data, source, "races", rows)
    if source != "jra":
        write_table(data, "jra", "races", make_rows("jra", LATE_DAYS, lambda k: 2))

    log: list[dict] = []
    real_dataset = ds.dataset
    monkeypatch.setattr(ds, "dataset", lambda *a, **k: _ScanSpy(real_dataset(*a, **k), log))

    expr = None
    if where == "narrowed":
        expr = ds.field("race_id") != f"{source}:{open_[0].isoformat()}:01"
    table = read_store_table(
        "races",
        columns=columns,
        start=LATE_DAYS[0],
        end=LATE_DAYS[-1],
        touches_price=False,
        where=expr,
        source=source,
    )

    # The rows returned are still the open days' rows, in the columns asked.
    want = [r for r in rows if r["race_date"] in set(open_)]
    if where == "narrowed":
        want = [r for r in want if r["race_id"] != f"{source}:{open_[0].isoformat()}:01"]
    assert table.column_names == columns
    assert race_ids(table) == Counter(r["race_id"] for r in want)

    wide = [s for s in log if s["names"] - {"race_date"}]
    assert wide, "the reader made no scan of the requested columns through pyarrow.dataset"
    for scan in wide:
        # The days that scan's filter let through, read over race_date while the scan was made.
        days = scan["days"]
        assert not days & set(held), (
            f"a scan of {scan['columns']} reached holdout day(s) "
            f"{sorted(days & set(held))[:3]} before they were dropped"
        )
