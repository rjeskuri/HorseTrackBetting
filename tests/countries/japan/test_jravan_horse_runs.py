"""A jravan `horse_runs`, every run of a horse, from the jravan races and runners (ticket #403).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jravan.horse_runs`:

- `build_horse_runs()` (keyword-only optional arguments allowed; its return value is not pinned)
  reads `processed_dir('jravan')/races.parquet` and `runners.parquet` through
  `models.v0.store.read_store_table(..., source='jravan')`, a year at a time over every year the
  races hold (the export reaches back before 2010), so holdout days are never read, and writes
  `processed_dir('jravan')/horse_runs.parquet` through a tmp file and `os.replace`
  (`records.write_parquet`): the columns of `schemas.TABLES['horse_runs']` in order, the Arrow
  types of `build.arrow_schema('horse_runs')`, no schema metadata, sorted by the table's sort key
  (horse_id, race_date, race_index). The same inputs give the same bytes, whatever the row order
  of the input files. The jra store is never read.
- One row per jravan `runners` row with a `horse_id` (a scratched runner too: its row keeps the
  placing mark, as a form table lists it; a runner without a horse_id adds no row), in the jra
  store's vocabulary, which the jravan races and runners already write:
  - from the runner: jurisdiction, horse_id, race_date, race_id, draw, rating, trainer_name,
    jockey_name, lbw_raw, actual_wt_lb, running_positions, finish_time_s (the horse's own),
    declared_horse_wt_lb, gear_raw, placing_raw;
  - from its race (on race_id): race_index, venue, surface, course_config, distance_m, going,
    race_class;
  - win_odds, D10: the runner's Win odds for a race on or after 2014-06-07 (`PRICE_ERA_START`,
    read with `touches_price=True`), null before it; the older run keeps its row (as #128's
    pp_lines keep an older start with the price null).
- `horsetrack japan jravan horse-runs` runs `build_horse_runs()` with the defaults.
- `model_inputs.yaml`'s `past_placings` lists a jravan `horse_runs` supply, and the committed
  `docs/data-coverage.md` shows it withheld (D21).

Every value is synthetic and invented, under a tmp HORSETRACK_DATA_DIR. Expected rows are derived
from the fixture's own rows, never from the code under test. Holdout days are real ones under the
committed seed, picked with `is_holdout_day`.
"""

from __future__ import annotations

import os
import random
import re
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.store import schemas
from horsetrack.core.store.build import arrow_schema
from horsetrack.countries.japan.models.v0 import store as v0_store
from horsetrack.countries.japan.models.v0.store import (
    PRICE_ERA_START,
    PreReformPriceError,
    read_store_table,
)
from horsetrack.countries.japan.research import coverage as cv

# isort: split
# (its own block: the module this ticket creates; one import, so the block sorts the same before
# and after the module exists)
from horsetrack.countries.japan.jravan import horse_runs as hr

REPO_ROOT = Path(__file__).resolve().parents[3]
COVERAGE_DOC = REPO_ROOT / "docs" / "data-coverage.md"
WITHHELD = "withheld (D21)"
TABLE = "horse_runs"
COLUMNS = tuple(schemas.TABLES[TABLE].columns)
SORT_BY = schemas.TABLES[TABLE].sort_by
FROM_RACE = ("race_index", "venue", "surface", "course_config", "distance_m", "going", "race_class")
FROM_RUNNER = (
    "jurisdiction", "horse_id", "race_date", "race_id", "draw", "rating", "trainer_name",
    "jockey_name", "lbw_raw", "actual_wt_lb", "running_positions", "finish_time_s",
    "declared_horse_wt_lb", "gear_raw", "placing_raw",
)  # fmt: skip

VENUES = ("TOK", "NAK", "KYO", "HSN", "SAP", "CHU", "NII")
SURFACES = ("turf", "dirt", "jump")
GOINGS = ("GOOD", "GOOD TO YIELDING", "SOFT", "HEAVY", None)
CLASSES = ("MAIDEN", "C1", "C2", "C3", "OP", "G1", "DEBUT", "G3")
CONFIGS = ("inner", "outer", None)
MARGINS = ("K", "12", "A", "1", "H", "34")
HORSES = tuple(f"{2000 + k}10{k:04d}" for k in range(1, 11))  # ten digits each
HOLDOUT_ONLY_HORSE = "2024109999"  # runs on a holdout day only
JRA_ONLY_HORSE = "2031100001"  # in the jra store only

# development days: from the export's early years through D10's boundary to after the holdout start
EARLY_DAYS = (
    date(1996, 4, 7), date(2009, 12, 27), date(2012, 5, 13), date(2014, 6, 1), date(2014, 6, 7),
    date(2019, 10, 20), date(2026, 11, 22),
)  # fmt: skip
_LATE = [date(2026, 11, 28) + timedelta(days=k) for k in range(28)]
LATE_DEV_DAYS = tuple(d for d in _LATE if not is_holdout_day("japan", d))[:2]
HELD_DAYS = tuple(d for d in _LATE if is_holdout_day("japan", d))[:2]
DEV_DAYS = EARLY_DAYS + LATE_DEV_DAYS
# (day index, race_no, horse_no) of the special runners
SCRATCHED = (2, 1)  # the last runner of that race is scratched (取)
DEMOTED = (5, 11, 2)  # placed 2nd, demoted: "2(降)"
NO_HORSE_ID = (3, 11)  # that race gets one extra runner without a horse_id


# ------------------------------------------------------------------------------------------------
# Synthetic jravan races and runners in the store's schemas


def race_id(day: date, venue: str, race_no: int) -> str:
    return f"jra:{day.isoformat()}:{venue}:{race_no:02d}"


def make_race(d: int, day: date, race_no: int, n: int) -> dict:
    venue = VENUES[d % len(VENUES)]
    k = 2 * d + (race_no > 1)
    return {
        "jurisdiction": "jra", "race_id": race_id(day, venue, race_no),
        "meeting_id": f"jra:{day.isoformat()}:{venue}", "race_date": day, "venue": venue,
        "race_no": race_no, "season": str(day.year), "race_index": 100 * d + race_no,
        "race_name": f"R{k}", "race_class": CLASSES[k % len(CLASSES)],
        "distance_m": 1000 + 200 * (k % 9), "surface": SURFACES[k % 3],
        "course_config": CONFIGS[k % 3], "going": GOINGS[k % len(GOINGS)], "field_size": n,
        "post_time_local": f"{10 + race_no % 7:02d}:{(5 * k) % 60:02d}", "status": "run",
    }  # fmt: skip


def make_runner(d: int, race: dict, j: int, horse_id: str | None, n: int) -> dict:
    k = 2 * d + (race["race_no"] > 1)
    horse_no = j + 1
    return {
        "jurisdiction": "jra", **{c: race[c] for c in
                                  ("race_id", "meeting_id", "race_date", "venue", "race_no",
                                   "season")},
        "horse_id": horse_id, "horse_no": horse_no, "horse_name": f"H{horse_id}",
        "jockey_code": f"{(d * 7 + j) % 13:05d}", "jockey_name": f"騎手{(d * 7 + j) % 13}",
        "jockey_key": f"jockey:{(d * 7 + j) % 13:05d}",
        "trainer_code": f"{(d * 5 + j) % 11:05d}", "trainer_name": f"調教師{(d * 5 + j) % 11}",
        "trainer_key": f"trainer:{(d * 5 + j) % 11:05d}",
        "actual_wt_lb": 117 + (d + 3 * j) % 11, "declared_horse_wt_lb": 990 + 3 * j + d,
        "draw": horse_no, "placing_raw": str(horse_no), "finish_pos": horse_no,
        "placing_code": None, "dead_heat": False,
        "lbw_raw": None if j == 0 else MARGINS[(j + d) % len(MARGINS)],
        "lbw_lengths": 0.25 * j,
        "running_positions": [(j + d + c) % n + 1 for c in range(4 - (k % 3))],
        "finish_time_s": round(58.0 + 12.0 * (k % 5) + 0.2 * j + 0.1 * d, 1),
        "win_odds": round(1.2 + 2.1 * j + 0.3 * d, 1), "is_scratched": False,
        "rating": 40 + d + j if (d + j) % 3 == 0 else None,
        "gear_raw": "B" if (d + j) % 4 == 1 else None,
        "bracket": 1 + j // 2, "actual_wt_kg": 55.0, "horse_wt_kg": 450 + j,
        "win_fav_rank": horse_no, "sex": "H", "age": 3 + d % 4,
    }  # fmt: skip


def scratch(runner: dict) -> dict:
    return runner | {
        "placing_raw": "取", "finish_pos": None, "placing_code": "WV", "is_scratched": True,
        "finish_time_s": None, "win_odds": None, "running_positions": [], "lbw_raw": None,
        "lbw_lengths": None, "win_fav_rank": None,
    }  # fmt: skip


def day_rows(d: int, day: date, extra_horse: str | None = None) -> tuple[list[dict], list[dict]]:
    """Two races on `day` (races 1 and 11), disjoint fields drawn from HORSES so a horse runs at
    most once a day; `extra_horse` joins race 1."""
    races, runners = [], []
    m1, m2 = 3 + d % 3, 2 + d % 2
    fields = {
        1: [HORSES[(d + j) % len(HORSES)] for j in range(m1)],
        11: [HORSES[(d + m1 + j) % len(HORSES)] for j in range(m2)],
    }
    if extra_horse:
        fields[1].append(extra_horse)
    if (d, 11) == NO_HORSE_ID:
        fields[11].append(None)
    for race_no, horses in fields.items():
        n = len(horses)
        race = make_race(d, day, race_no, n)
        races.append(race)
        for j, horse in enumerate(horses):
            runner = make_runner(d, race, j, horse, n)
            if (d, race_no) == SCRATCHED and j == n - 1:
                runner = scratch(runner)
            if (d, race_no, j + 1) == DEMOTED:
                runner = runner | {"placing_raw": "2(降)"}
            runners.append(runner)
    return races, runners


def fixture_rows() -> dict[str, list[dict]]:
    races, runners = [], []
    for d, day in enumerate(DEV_DAYS):
        r, s = day_rows(d, day)
        races += r
        runners += s
    for i, day in enumerate(HELD_DAYS):
        r, s = day_rows(len(DEV_DAYS) + i, day, extra_horse=HOLDOUT_ONLY_HORSE)
        races += r
        runners += s
    return {"races": races, "runners": runners}


def arrow_table(name: str, rows: list[dict]) -> pa.Table:
    schema = arrow_schema(name)
    return pa.table({c: [row.get(c) for row in rows] for c in schema.names}, schema=schema)


def write_store(data: Path, rows: dict[str, list[dict]], *, shuffle: int | None = None) -> None:
    folder = data / "processed" / "jravan"
    folder.mkdir(parents=True, exist_ok=True)
    for name, table_rows in rows.items():
        ordered = list(table_rows)
        if shuffle is not None:
            random.Random(shuffle).shuffle(ordered)
        pq.write_table(arrow_table(name, ordered), folder / f"{name}.parquet")
    # the jra store: another horse, never in the jravan horse_runs
    jra = data / "processed" / "jra"
    jra.mkdir(parents=True, exist_ok=True)
    jra_race = make_race(0, date(2019, 10, 20), 5, 1)
    jra_runner = make_runner(0, jra_race, 0, JRA_ONLY_HORSE, 1)
    pq.write_table(arrow_table("races", [jra_race]), jra / "races.parquet")
    pq.write_table(arrow_table("runners", [jra_runner]), jra / "runners.parquet")


def expected_rows(rows: dict[str, list[dict]]) -> list[dict]:
    """The horse_runs rows the fixture should give, derived from its rows alone."""
    by_race = {race["race_id"]: race for race in rows["races"]}
    out = []
    for runner in rows["runners"]:
        if runner["horse_id"] is None or runner["race_date"] in HELD_DAYS:
            continue
        race = by_race[runner["race_id"]]
        row = {c: runner[c] for c in FROM_RUNNER} | {c: race[c] for c in FROM_RACE}
        row["win_odds"] = runner["win_odds"] if runner["race_date"] >= PRICE_ERA_START else None
        out.append({c: row[c] for c in COLUMNS})
    out.sort(key=lambda r: tuple(r[c] for c in SORT_BY))
    return out


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(root))
    return root


def output(data: Path) -> Path:
    return data / "processed" / "jravan" / f"{TABLE}.parquet"


def build(data: Path) -> pa.Table:
    hr.build_horse_runs()
    path = output(data)
    assert path.is_file(), f"no {path} written"
    return pq.read_table(path)


# ------------------------------------------------------------------------------------------------
# One row per run, in the store schema, the jra vocabulary carried from races and runners


def test_one_row_per_run_of_each_horse_in_the_store_schema(data: Path) -> None:
    rows = fixture_rows()
    write_store(data, rows)
    got = build(data)

    assert got.schema.equals(arrow_schema(TABLE), check_metadata=False), got.schema
    assert not got.schema.metadata, "schema metadata makes the bytes depend on the writer"
    want = expected_rows(rows)
    assert got.num_rows == len(want)
    runs = {}
    for row in want:
        runs[row["horse_id"]] = runs.get(row["horse_id"], 0) + 1
    got_runs = {}
    for horse in got.column("horse_id").to_pylist():
        got_runs[horse] = got_runs.get(horse, 0) + 1
    assert got_runs == runs, "runs per horse"
    assert len(set(runs.values())) > 1, "the fixture must give horses different numbers of runs"
    assert got.equals(arrow_table(TABLE, want)), "rows, values or sort order differ"


def test_an_abandoned_races_unnumbered_runners_are_runs_with_no_draw(data: Path) -> None:
    """The runners of an abandoned race that were never drawn have a null horse_no and draw
    (#394); each is still a run of its horse, whose draw is null."""
    rows = fixture_rows()
    day = DEV_DAYS[3]
    race = make_race(3, day, 7, 3) | {"status": "abandoned"}
    rows["races"].append(race)
    for j, horse in enumerate(HORSES[:3]):
        runner = make_runner(3, race, j, horse, 3) | {
            "horse_no": None, "draw": None, "placing_raw": None, "finish_pos": None,
            "finish_time_s": None, "win_odds": None, "running_positions": [], "lbw_raw": None,
            "lbw_lengths": None, "win_fav_rank": None,
        }  # fmt: skip
        rows["runners"].append(runner)
    write_store(data, rows)
    got = build(data)

    want = expected_rows(rows)
    assert got.equals(arrow_table(TABLE, want)), "rows, values or sort order differ"
    mine = [r for r in got.to_pylist() if r["race_id"] == race["race_id"]]
    assert sorted(r["horse_id"] for r in mine) == sorted(HORSES[:3])
    assert all(r["draw"] is None and r["placing_raw"] is None for r in mine)
    assert all(r["draw"] is not None for r in got.to_pylist() if r["race_id"] != race["race_id"])


def test_every_column_comes_from_its_runner_or_its_race(data: Path) -> None:
    """Row by row on (horse_id, race_id): each column against the source it comes from, so a
    failure names the column."""
    rows = fixture_rows()
    write_store(data, rows)
    got = build(data).to_pylist()
    by_key = {(r["horse_id"], r["race_id"]): r for r in got}
    want = expected_rows(rows)
    assert set(by_key) == {(r["horse_id"], r["race_id"]) for r in want}
    for column in COLUMNS:
        wrong = [(r["horse_id"], r["race_id"], by_key[r["horse_id"], r["race_id"]][column],
                  r[column]) for r in want
                 if by_key[r["horse_id"], r["race_id"]][column] != r[column]]  # fmt: skip
        assert not wrong, (column, wrong[:3])
    # the spread the fixture carries, so a constant per column cannot pass
    for column in ("venue", "surface", "going", "race_class", "course_config", "distance_m",
                   "race_index", "placing_raw", "lbw_raw", "rating", "gear_raw", "win_odds",
                   "finish_time_s", "jockey_name", "trainer_name"):  # fmt: skip
        assert len({repr(r[column]) for r in want}) >= 2, column


def test_scratched_demoted_and_horseless_runners(data: Path) -> None:
    rows = fixture_rows()
    write_store(data, rows)
    got = build(data).to_pylist()
    scratched = [r for r in rows["runners"] if r["is_scratched"]]
    assert len(scratched) == 1
    s = scratched[0]
    mine = [r for r in got if (r["horse_id"], r["race_id"]) == (s["horse_id"], s["race_id"])]
    assert len(mine) == 1 and mine[0]["placing_raw"] == "取", mine
    assert mine[0]["finish_time_s"] is None and mine[0]["win_odds"] is None
    assert any(r["placing_raw"] == "2(降)" for r in got)
    horseless = [r for r in rows["runners"] if r["horse_id"] is None]
    assert len(horseless) == 1
    assert all(r["horse_id"] is not None for r in got)
    assert not [r for r in got if r["race_id"] == horseless[0]["race_id"]
                and r["draw"] == horseless[0]["draw"]]  # fmt: skip


def test_form_reaches_back_over_the_export_and_the_jra_store_is_not_read(data: Path) -> None:
    rows = fixture_rows()
    write_store(data, rows)
    got = build(data)
    years = {d.year for d in got.column("race_date").to_pylist()}
    assert {1996, 2009, 2012, 2014, 2019, 2026} <= years, sorted(years)
    assert JRA_ONLY_HORSE not in set(got.column("horse_id").to_pylist())


# ------------------------------------------------------------------------------------------------
# D10: the price only inside the price era


def test_win_odds_are_null_before_the_price_era_and_kept_from_it(data: Path) -> None:
    rows = fixture_rows()
    write_store(data, rows)
    got = build(data).to_pylist()
    odds = {(r["horse_id"], r["race_id"]): r["win_odds"] for r in rows["runners"]}
    before = [r for r in got if r["race_date"] < PRICE_ERA_START]
    from_era = [r for r in got if r["race_date"] >= PRICE_ERA_START]
    assert {r["race_date"] for r in before} >= {date(1996, 4, 7), date(2014, 6, 1)}
    assert date(2014, 6, 7) in {r["race_date"] for r in from_era}
    assert all(r["win_odds"] is None for r in before), "a pre-2014-06-07 price was written (D10)"
    for r in before:  # the run itself is kept
        assert r["placing_raw"] is not None and r["venue"] is not None
    priced = [r for r in from_era if odds[r["horse_id"], r["race_id"]] is not None]
    assert len(priced) >= 10
    assert all(r["win_odds"] == odds[r["horse_id"], r["race_id"]] for r in priced)

    # a reader that touches the price reads it from the era on, and is refused before it
    table = read_store_table(TABLE, columns=["horse_id", "race_id", "win_odds"],
                             start=PRICE_ERA_START, end=date(2026, 12, 31), touches_price=True,
                             source="jravan").to_pylist()  # fmt: skip
    assert {(r["horse_id"], r["race_id"]) for r in table} == {
        (r["horse_id"], r["race_id"]) for r in from_era
    }
    with pytest.raises(PreReformPriceError):
        read_store_table(TABLE, columns=["win_odds"], start=date(2014, 6, 6),
                         end=date(2026, 12, 31), touches_price=True, source="jravan")  # fmt: skip


# ------------------------------------------------------------------------------------------------
# Holdout days are never read


def test_holdout_days_are_never_read(data: Path) -> None:
    assert len(HELD_DAYS) == 2 and len(LATE_DEV_DAYS) == 2  # the rule holds out some days only
    rows = fixture_rows()
    held_ids = {r["race_id"] for r in rows["races"] if r["race_date"] in HELD_DAYS}
    assert held_ids and any(r["horse_id"] == HOLDOUT_ONLY_HORSE for r in rows["runners"])
    write_store(data, rows)
    got = build(data)
    days = set(got.column("race_date").to_pylist())
    assert not days & set(HELD_DAYS), sorted(days & set(HELD_DAYS))
    assert not set(got.column("race_id").to_pylist()) & held_ids
    assert HOLDOUT_ONLY_HORSE not in set(got.column("horse_id").to_pylist())
    # development days after the holdout start are runs like any other
    assert set(LATE_DEV_DAYS) <= days, sorted(days)


class _Recorder:
    """`pyarrow.dataset` for the store module, recording every table its datasets' scans return."""

    SCANS = ("to_table", "to_batches", "head", "take")

    def __init__(self) -> None:
        self.scans: list[tuple[str, pa.Table]] = []

    def __getattr__(self, attr: str):
        return getattr(pads, attr)

    def _wrap(self, name: str, target):
        recorder = self

        class Proxy:
            def __getattr__(self, attr: str):
                value = getattr(target, attr)
                if attr in recorder.SCANS:
                    return recorder._scan(name, value)
                if attr == "scanner":
                    return lambda *a, **k: recorder._wrap(name, value(*a, **k))
                return value

        return Proxy()

    def _scan(self, name: str, method):
        def scan(*a, **k):
            out = method(*a, **k)
            if isinstance(out, pa.Table):
                self.scans.append((name, out))
                return out
            batches = list(out)
            self.scans.append((name, pa.Table.from_batches(batches) if batches else pa.table({})))
            return iter(batches)

        return scan

    def dataset(self, source, *a, **k):
        return self._wrap(Path(source).stem, pads.dataset(source, *a, **k))


def test_no_holdout_row_leaves_the_parquet_scan(
    data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every scan the builder's reads run (the year discovery, the runners on both sides of D10,
    the races) returns no holdout-day row: the holdout days are excluded in the scan predicate,
    not dropped after the rows are loaded."""
    rows = fixture_rows()
    held_ids = {r["race_id"] for r in rows["races"] if r["race_date"] in HELD_DAYS}
    write_store(data, rows)
    recorder = _Recorder()
    monkeypatch.setattr(v0_store, "ds", recorder)
    hr.build_horse_runs()
    assert output(data).is_file()

    names = {name for name, _ in recorder.scans}
    assert {"races", "runners"} <= names, sorted(names)
    leaks = []
    for name, table in recorder.scans:
        cols = set(table.column_names)
        days = set(table.column("race_date").to_pylist()) if "race_date" in cols else set()
        ids = set(table.column("race_id").to_pylist()) if "race_id" in cols else set()
        horses = set(table.column("horse_id").to_pylist()) if "horse_id" in cols else set()
        if days & set(HELD_DAYS) or ids & held_ids or HOLDOUT_ONLY_HORSE in horses:
            leaks.append((name, sorted(cols), sorted(days & set(HELD_DAYS))))
    assert not leaks, leaks
    # the scans reached the holdout's year: its development days came through them
    for table in ("races", "runners"):
        seen = set()
        for name, t in recorder.scans:
            if name == table and "race_date" in t.column_names:
                seen |= set(t.column("race_date").to_pylist())
        assert set(LATE_DEV_DAYS) <= seen, (table, sorted(seen))
    # a runners scan with the price (the era side of D10) and one without it (before the era)
    runner_cols = [set(t.column_names) for name, t in recorder.scans if name == "runners"]
    assert any("win_odds" in c for c in runner_cols), runner_cols
    assert any("win_odds" not in c and "horse_id" in c for c in runner_cols), runner_cols


# ------------------------------------------------------------------------------------------------
# Byte-stable, atomic


def test_the_output_is_byte_stable_whatever_the_input_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = fixture_rows()
    outputs = []
    for i, shuffle in enumerate((None, 7, 1234)):
        root = (tmp_path / f"data{i}").resolve()
        monkeypatch.setenv(paths.DATA_DIR_VAR, str(root))
        write_store(root, rows, shuffle=shuffle)
        hr.build_horse_runs()
        first = output(root).read_bytes()
        hr.build_horse_runs()  # again, over its own output
        assert output(root).read_bytes() == first, "two builds of the same inputs differ"
        outputs.append(first)
    assert outputs[0] == outputs[1] == outputs[2], "the input row order changes the bytes"
    shuffled = pq.read_table(tmp_path / "data1" / "processed" / "jravan" / "runners.parquet")
    assert shuffled.column("race_id").to_pylist() != [r["race_id"] for r in rows["runners"]]


def test_the_write_is_atomic(data: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = fixture_rows()
    write_store(data, rows)
    build(data)
    before = output(data).read_bytes()
    folder = output(data).parent
    listing = sorted(p.name for p in folder.iterdir())

    # a new run, so a write in place would change the file
    d = len(DEV_DAYS) + len(HELD_DAYS)
    r, s = day_rows(d, date(2020, 2, 2))
    write_store(data, {"races": rows["races"] + r, "runners": rows["runners"] + s})

    def refuse(src, dst, *a, **k):
        raise OSError("refused")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError):
        hr.build_horse_runs()
    monkeypatch.undo()
    assert output(data).read_bytes() == before, "a failed write touched the old file"
    assert sorted(p.name for p in folder.iterdir()) == listing, "a failed write left a file behind"


# ------------------------------------------------------------------------------------------------
# The command


def test_the_command_writes_the_table(data: Path) -> None:
    rows = fixture_rows()
    write_store(data, rows)
    result = CliRunner().invoke(app, ["japan", "jravan", "horse-runs"])
    assert result.exit_code == 0, result.output
    assert output(data).is_file()
    got = pq.read_table(output(data))
    assert got.equals(arrow_table(TABLE, expected_rows(rows)))
    from_cli = output(data).read_bytes()
    hr.build_horse_runs()
    assert output(data).read_bytes() == from_cli


# ------------------------------------------------------------------------------------------------
# The coverage audit lists the supply (#374), its figures withheld (D21)


def test_past_placings_lists_the_jravan_horse_runs_supply() -> None:
    inputs = {inp.name: inp for inp in cv.load_inputs()}
    supplies = inputs["past_placings"].supplies
    jravan = [s for s in supplies if (s.source, s.table) == ("jravan", TABLE)]
    assert jravan, [(s.source, s.table, s.column) for s in supplies]
    assert all(s.column in COLUMNS for s in jravan)


def test_the_committed_coverage_doc_lists_the_supply_withheld() -> None:
    text = COVERAGE_DOC.read_text(encoding="utf-8")
    match = re.search(r"^## past_placings\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, "docs/data-coverage.md has no past_placings section"
    lines = [line for line in match.group(1).splitlines()
             if "jravan" in line and f"{TABLE}." in line]  # fmt: skip
    assert lines, "docs/data-coverage.md does not list the jravan horse_runs supply"
    assert all(WITHHELD in line and not re.search(r"\d{4}-\d{2}-\d{2}|\d+ rows", line)
               for line in lines), lines  # fmt: skip
