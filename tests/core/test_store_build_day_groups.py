"""Ticket #510 (fix round): the core store build writes its dated tables a row group per race day.

Contract pinned here (the coder follows it):

- Writer. `core.store.build._write_tables(tables, folder)`, the writer `build_tables` puts a
  source's tables through (`processed_dir(source)`), writes `meetings`, `races` and `runners`
  so that every row group holds exactly one race day: the group's `race_date` column chunk
  records statistics with min == max and a null count of 0. The rows and their order are
  unchanged (what the frame holds, in its order: the build's declared sort keys, race_id first
  for races and runners). Nothing is re-sorted by date: a new row group starts at each change of
  day in that order, so a day whose rows come in several runs (a frame in venue order) spans
  several groups, and a run of rows without a day (a malformed race_id) sits in groups holding
  only such rows, never in a dated group. For these small frames the groups are exactly the
  runs (a group per run of one day, not a group per row).
- Unchanged: the schema (declared columns, declared order, the store's Arrow types), every
  table's rows and order (also the tables without a day, and `horse_runs`), the bytes' determinism
  (the same frames give the same bytes), and the all-or-nothing write (a table that fails to
  write leaves every old file whole and no stray tmp file).
- Not pinned: the layout of `horse_runs` (sorted horse_id first, so its day changes on nearly
  every row; follow-up #678) and of `win_sales`.

All frames are synthetic, built here the way the build hands them over (`to_pandas` with pandas
`Int64` and `boolean`), written under tmp_path.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import store_fakes

from horsetrack.core import paths
from horsetrack.core.store import schemas
from horsetrack.core.store.tables import ARROW_TYPES

# isort: split
from horsetrack.core.store import build

JURISDICTION = "jra"
PINNED = ("meetings", "races", "runners")  # the dated tables whose layout is pinned
_PANDAS = {pa.int64(): pd.Int64Dtype(), pa.bool_(): pd.BooleanDtype()}

# (day, venues racing that day); races per venue and runners per race vary with the day
SPAN = (
    (date(2026, 11, 1), ("KYO", "TOK")),
    (date(2026, 11, 2), ("KYO", "NAK", "TOK")),
    (date(2026, 11, 8), ("TOK",)),
    (date(2026, 11, 14), ("NAK", "TOK")),
    (date(2026, 11, 15), ("KYO", "NAK")),
)
LATER_SPAN = (
    (date(2026, 11, 21), ("KYO", "TOK")),
    (date(2026, 11, 22), ("NAK",)),
    (date(2026, 11, 23), ("KYO", "NAK", "TOK")),
)


def declared_schema(name: str) -> pa.Schema:
    return pa.schema([(c, ARROW_TYPES[k]) for c, k in schemas.TABLES[name].columns.items()])


def hexed(*parts: object) -> str:
    return hashlib.sha1(":".join(map(str, parts)).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------
# Synthetic store rows (dicts; a missing column is null)
# ---------------------------------------------------------------------------------------------


def race_rows(span) -> list[dict]:
    rows = []
    for k, (day, venues) in enumerate(span):
        for v, venue in enumerate(venues):
            for r in range(1, 2 + (k + v) % 3 + 1):
                rid = f"{JURISDICTION}:{day.isoformat()}:{venue}:{r:02d}"
                rows.append({
                    "jurisdiction": JURISDICTION, "race_id": rid,
                    "meeting_id": f"{JURISDICTION}:{day.isoformat()}:{venue}", "race_date": day,
                    "venue": venue, "race_no": r, "season": str(day.year),
                    "race_index": 100 * k + 10 * v + r, "race_name": hexed("name", rid),
                    "distance_m": 1200 + 200 * ((k + r) % 5), "surface": ("turf", "dirt")[r % 2],
                    "field_size": 3 + (k + r) % 2, "finish_time_s": 70.0 + 0.25 * r + 0.5 * k,
                    "split_times_s": [12.0 + 0.1 * r, 11.5 + 0.1 * k], "status": "run",
                    "result_published_utc": datetime(
                        day.year, day.month, day.day, 6 + r, 30, tzinfo=timezone.utc
                    ),
                })  # fmt: skip
    return rows


def runner_rows(races: Sequence[dict]) -> list[dict]:
    rows = []
    for race in races:
        n = race["field_size"]
        for h in range(1, n + 1):
            rows.append({
                **{c: race[c] for c in ("jurisdiction", "race_id", "meeting_id", "race_date",
                                         "venue", "race_no", "season")},
                "horse_id": f"H{race['race_index']:04d}{h}", "horse_no": h,
                "horse_name": hexed("horse", race["race_id"], h),
                "finish_pos": (h + race["race_no"]) % n + 1, "win_odds": 1.5 + 0.75 * h,
                "running_positions": [h, (h % n) + 1], "dead_heat": False,
                "is_scratched": h == n and race["race_no"] == 2,
            })  # fmt: skip
    return rows


def meeting_rows(races: Sequence[dict]) -> list[dict]:
    seen = {}
    for race in races:
        seen.setdefault(race["meeting_id"], race)
    return [
        {"jurisdiction": JURISDICTION, "meeting_id": mid, "race_date": r["race_date"],
         "venue": r["venue"], "season": r["season"], "status": "run"}
        for mid, r in seen.items()
    ]  # fmt: skip


def dividend_rows(runners: Sequence[dict]) -> list[dict]:
    return [
        {"jurisdiction": JURISDICTION, "race_id": r["race_id"], "pool": "WIN",
         "combination": str(r["horse_no"]), "dividend": 100.0 * r["win_odds"], "is_refund": False}
        for r in runners
        if r["finish_pos"] == 1
    ]  # fmt: skip


def horse_run_rows(runners: Sequence[dict]) -> list[dict]:
    """Four horses (by horse_no), each running on every day: in horse_id order the day changes on
    nearly every row."""
    return [
        {"jurisdiction": JURISDICTION, "horse_id": f"H{r['horse_no']}", "race_date": r["race_date"],
         "race_index": i, "venue": r["venue"], "race_id": r["race_id"], "placing_raw": "1"}
        for i, r in enumerate(runners)
        if r["race_no"] == 1
    ]  # fmt: skip


def by_race_id(rows):
    return sorted(rows, key=lambda r: (r["race_id"], r.get("horse_no", 0)))


def by_venue(rows):
    """Venue first: each day's rows come in one run per venue it raced at."""
    return sorted(
        rows, key=lambda r: (r["venue"], r["race_date"], r["race_no"], r.get("horse_no", 0))
    )


def build_order(span) -> dict[str, list[dict]]:
    """Every table in the order `_assemble` hands it over (its declared sort keys)."""
    races = by_race_id(race_rows(span))
    runners = by_race_id(runner_rows(races))
    return {
        "meetings": sorted(meeting_rows(races), key=lambda r: (r["race_date"], r["venue"])),
        "races": races,
        "runners": runners,
        "dividends": sorted(dividend_rows(runners), key=lambda r: r["race_id"]),
        "horse_runs": sorted(
            horse_run_rows(runners), key=lambda r: (r["horse_id"], r["race_date"], r["race_index"])
        ),
    }


def venue_order() -> dict[str, list[dict]]:
    races = race_rows(SPAN)
    return {
        "races": by_venue(races),
        "runners": by_venue(runner_rows(races)),
        "meetings": [],  # an empty table is written with its schema
    }


def with_undated_runs() -> dict[str, list[dict]]:
    """Runners in race_id order with two rows of a malformed race_id (no day) between two runs of
    one day: jra:2026-11-02:KYZ sorts after that day's KYO races and before its NAK races."""
    runners = runner_rows(race_rows(SPAN))
    bad = [
        {"jurisdiction": JURISDICTION, "race_id": f"{JURISDICTION}:2026-11-02:KYZ",
         "race_date": None, "venue": None, "race_no": None, "horse_id": f"HX{h}", "horse_no": h,
         "finish_pos": h}
        for h in (1, 2)
    ]  # fmt: skip
    return {"runners": by_race_id(runners + bad)}


CASES = {
    "build-order": build_order(SPAN),
    "venue-order": venue_order(),
    "undated-rows-between-runs-of-a-day": with_undated_runs(),
}


# The cases rely on these shapes.
assert len({r["race_date"] for r in CASES["build-order"]["races"]}) == len(SPAN) == 5
assert any(r["race_date"] is None for r in CASES["undated-rows-between-runs-of-a-day"]["runners"])


def frame_of(name: str, rows: Sequence[dict]) -> pd.DataFrame:
    """The frame the build hands `_write_tables`: declared columns in order, pandas Int64 and
    boolean."""
    return pa.Table.from_pylist(list(rows), schema=declared_schema(name)).to_pandas(
        types_mapper=_PANDAS.get
    )


def expected(name: str, rows: Sequence[dict]) -> list[dict]:
    return [{c: row.get(c) for c in schemas.TABLES[name].columns} for row in rows]


def runs(days: Sequence[date | None]) -> list[tuple[date | None, int]]:
    """Maximal runs of one day (None: no day) in row order."""
    out: list[list] = []
    for day in days:
        if out and out[-1][0] == day:
            out[-1][1] += 1
        else:
            out.append([day, 1])
    return [(d, n) for d, n in out]


def group_runs(path: Path) -> list[tuple[date | None, int]]:
    """The file's row groups as (day, rows), each checked to hold one day (statistics min == max
    and no null) or only rows without a day; consecutive undated groups count as one run."""
    file = pq.ParquetFile(path)
    meta = file.metadata
    at = meta.schema.names.index("race_date")
    out: list[tuple[date | None, int]] = []
    for g in range(meta.num_row_groups):
        group = meta.row_group(g)
        days = file.read_row_group(g, columns=["race_date"]).column("race_date").to_pylist()
        present = sorted({d for d in days if d is not None})
        stats = group.column(at).statistics
        if not present:
            assert stats is None or not stats.has_null_count or stats.null_count == len(days)
            if out and out[-1][0] is None:
                out[-1] = (None, out[-1][1] + group.num_rows)
            else:
                out.append((None, group.num_rows))
            continue
        assert len(present) == 1 and None not in days, (
            f"{path.name}: row group {g} of {meta.num_row_groups} holds the days "
            f"{present[0]} .. {present[-1]} ({len(present)} days"
            f"{', and rows without a day' if None in days else ''}), not one race day"
        )
        assert stats is not None and stats.has_min_max, f"{path.name}: group {g} has no min/max"
        assert stats.min == stats.max == present[0], (path.name, g, stats.min, stats.max)
        assert stats.has_null_count and stats.null_count == 0, (path.name, g, stats.null_count)
        out.append((present[0], group.num_rows))
    return out


def expected_groups(rows: Sequence[dict]) -> list[tuple[date | None, int]]:
    return runs([r.get("race_date") for r in rows])


# ---------------------------------------------------------------------------------------------
# DoD: the dated tables a row group per race day; rows, order and schema unchanged
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", list(CASES))
def test_write_tables_writes_a_row_group_per_run_of_one_race_day(tmp_path: Path, case: str) -> None:
    given = CASES[case]
    folder = tmp_path / "processed" / JURISDICTION
    build._write_tables({name: frame_of(name, rows) for name, rows in given.items()}, folder)

    assert sorted(p.name for p in folder.iterdir()) == sorted(f"{n}.parquet" for n in given)
    for name, rows in given.items():
        path = folder / f"{name}.parquet"
        assert pq.read_schema(path) == declared_schema(name), name
        # The rows and their order: exactly the frame's, whatever the layout.
        assert pq.read_table(path).to_pylist() == expected(name, rows), (
            f"{name}: the written rows are not the frame's rows in the frame's order"
        )
    for name in (n for n in PINNED if n in given and given[n]):
        want = expected_groups(given[name])
        got = group_runs(folder / f"{name}.parquet")
        assert got == want, (
            f"{name}: row groups (day, rows) {got}, expected a group per run of one day {want}"
        )
    if case == "venue-order":  # a day in several runs spans several groups, in the frame's order
        days = [d for d, _ in expected_groups(given["races"])]
        assert len(days) > len(set(days))

    # The same frames give the same bytes.
    again = tmp_path / "again"
    build._write_tables({name: frame_of(name, rows) for name, rows in given.items()}, again)
    for name in given:
        assert (again / f"{name}.parquet").read_bytes() == (folder / f"{name}.parquet").read_bytes()


def test_a_failed_write_leaves_the_day_grouped_tables_whole(tmp_path: Path) -> None:
    folder = tmp_path / "processed" / JURISDICTION
    first = build_order(SPAN)
    build._write_tables({name: frame_of(name, rows) for name, rows in first.items()}, folder)
    for name in PINNED:
        assert group_runs(folder / f"{name}.parquet") == expected_groups(first[name]), name
    before = {p.name: p.read_bytes() for p in folder.iterdir()}

    # A rebuild with other days whose second table fails on a cell of its last day.
    later = build_order(LATER_SPAN)
    runners = frame_of("runners", later["runners"])
    runners["horse_no"] = runners["horse_no"].astype(object)
    runners.loc[len(runners) - 1, "horse_no"] = "not a number"
    tables = {"races": frame_of("races", later["races"]), "runners": runners}
    with pytest.raises(pa.ArrowException):
        build._write_tables(tables, folder)

    after = {p.name: p.read_bytes() for p in folder.iterdir()}
    assert sorted(after) == sorted(before), "a stray file was left behind"
    assert after == before, "a failed write changed the old tables"


# ---------------------------------------------------------------------------------------------
# DoD: the build route writes them so (the synthetic pages of store_fakes)
# ---------------------------------------------------------------------------------------------


def test_build_tables_writes_meetings_races_and_runners_a_row_group_per_race_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(paths.DATA_DIR_VAR, str((tmp_path / "data").resolve()))
    store_fakes.register(monkeypatch, store_fakes.FakeStoreAdapter())
    store_fakes.populate(store_fakes.PAGES)
    build.build_tables(store_fakes.SOURCE, workers=1)

    for name in PINNED:
        path = paths.processed_dir(store_fakes.SOURCE) / f"{name}.parquet"
        days = pq.read_table(path, columns=["race_date"]).column("race_date").to_pylist()
        assert len({d for d in days if d is not None}) >= 2, name  # several days to group
        assert group_runs(path) == runs(days), name
