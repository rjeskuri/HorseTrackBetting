"""The store checks (#49): reconciliation checks on the built tables and their report.

Ported from HKTrack tests/unit/store/test_checks.py onto the synthetic pages and fake adapter of
store_fakes.py (no real page). Seams pinned, S = source, CFG = the source's `SourceConfig`:

* `horsetrack.core.store.checks.run_checks(S, *, config=CFG, write_report=True)` returns a
  `CheckResult` (checks, integrity, seasons, date_min, date_max, report_path; `ok`, `failures`,
  `error_line()`, `lines()`) and writes its report to `reports_dir() / f"store-check-{S}.md"`.
  Core has no source -> country map and the test source has no YAML, so the caller passes the
  config (`core.config.load_source(country, source)`) instead of HKTrack's jurisdiction name.
* `check_win_dividend(races, runners, dividends, CFG)` and `check_orphans(table, frame, races)`
  are callable on plain frames; `check_field_size(races, runners)` too.
* Levels `OK` / `WARN` / `FAIL`, HKTrack's tolerances: duplicate race ids and duplicate
  (race_id, horse_id) fail; orphan race ids fail above 1% of a table's race ids; field-size
  mismatches and wrong winner odds are counted, a `fail` only above max(5, 1% of what was checked).

The synthetic dividends are quoted per 10 (HKJC-style), so the build-level tests use the HKJC
config; the displayed-odds table also runs the JRA config (per 100 yen).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from store_fakes import (
    HORSE_RECENT,
    PAGES,
    RACE_2010,
    RACE_2026,
    RACE_ABANDONED,
    RESULTS_ABANDONED,
    SOURCE,
    FakeStoreAdapter,
    populate,
    register,
)

from horsetrack.core import paths
from horsetrack.core.config import load_source
from horsetrack.core.store import build, tables

# isort: split
# A block of its own: while the module is missing, it is the import error a red run reports.
from horsetrack.core.store import checks

VAR = "HORSETRACK_DATA_DIR"
HKJC = load_source("hongkong", "hkjc")
JRA = load_source("japan", "jra")
CHILD_TABLES = ("runners", "dividends", "sectionals", "incidents", "running_comments")

# An abandoned race as a results page shows it: no field and no runners. (store_fakes keeps two
# unplaced runners and a null field size on it, which the field-size check would rightly count.)
_ABANDONED = {**RESULTS_ABANDONED, "races": [{**RESULTS_ABANDONED["races"][0], "field_size": 0}]}
CHECK_PAGES = [
    p._replace(tables={**_ABANDONED, "runners": []}) if p.key == "2019-11-13_HV_R01" else p
    for p in PAGES
]


# ---------------------------------------------------------------- fixtures and helpers


@pytest.fixture
def data(tmp_path, monkeypatch) -> Path:
    path = (tmp_path / "data").resolve()
    monkeypatch.setenv(VAR, str(path))
    return path


@pytest.fixture
def raw_root(data, monkeypatch) -> Path:
    """Raw cache + manifest of CHECK_PAGES only: nothing built."""
    register(monkeypatch, FakeStoreAdapter())
    populate(CHECK_PAGES)
    return data


@pytest.fixture
def built_root(raw_root) -> Path:
    build.build_tables(SOURCE, workers=1)
    return raw_root


def run() -> checks.CheckResult:
    return checks.run_checks(SOURCE, config=HKJC)


def report_file() -> Path:
    return paths.reports_dir() / f"store-check-{SOURCE}.md"


def rewrite(table: str, change) -> None:
    """Load a built table, let the test tamper with it, write it back with the store's schema."""
    frame = change(tables.read_table(SOURCE, table))
    data = pa.Table.from_pandas(frame, schema=build.arrow_schema(table), preserve_index=False)
    pq.write_table(data, tables.table_path(SOURCE, table))


def by_name(result: checks.CheckResult) -> dict[str, checks.Check]:
    return {check.name: check for check in result.checks}


# ---------------------------------------------------------------- the synthetic build


def test_fixture_build_passes_every_check(built_root):
    result = run()
    assert result.ok and result.failures == [] and result.error_line() is None
    found = by_name(result)
    assert set(found) == {
        "unique race_id", "unique (race_id, horse_id)", "field_size", "winner odds vs WIN dividend",
        *(f"orphan race ids in {name}" for name in CHILD_TABLES),
    }  # fmt: skip
    assert all(check.level == checks.OK and check.violations == 0 for check in result.checks)
    # 4 races; 12 runner rows (3 + 4 + 5, the abandoned race has none), scratched rows included
    assert found["unique race_id"].checked == 4
    assert found["unique (race_id, horse_id)"].checked == 12
    assert found["field_size"].checked == 4  # scratched runners are not in the field
    # 2010 and 2026 have one winner each; the dead heat and the abandoned race are out of scope
    assert found["winner odds vs WIN dividend"].checked == 2
    orphans = {t: found[f"orphan race ids in {t}"].checked for t in CHILD_TABLES}
    assert orphans == {
        "runners": 3, "dividends": 3, "sectionals": 1, "incidents": 3, "running_comments": 1,
    }  # fmt: skip
    assert (result.date_min, result.date_max) == ("2010-01-01", "2026-09-16")


def test_season_stats_count_run_races_starters_and_rating_share(built_root):
    seasons = {s["season"]: s for s in run().seasons}
    assert list(seasons) == ["2009/10", "2020/21", "2026/27"]  # abandoned 2019/20: not run
    # 2026: 4 starters (the scratched #3 is not one), only HORSE_RECENT carries a rating
    assert seasons["2026/27"] == {
        "season": "2026/27", "races": 1, "starters": 4, "rating_share": 0.25,
    }  # fmt: skip
    assert seasons["2009/10"]["starters"] == 3
    assert seasons["2009/10"]["rating_share"] == pytest.approx(1 / 3)
    assert seasons["2020/21"] == {
        "season": "2020/21", "races": 1, "starters": 3, "rating_share": 0.0,
    }  # fmt: skip


def test_report_is_written_under_reports_dir(built_root):
    result = run()
    assert report_file().is_file()
    assert Path(result.report_path) == report_file()
    assert "OK" in report_file().read_text(encoding="utf-8")
    assert len(result.lines()) <= 15
    assert checks.as_dict(result)["ok"] is True


def test_no_report_when_asked_not_to(built_root):
    result = checks.run_checks(SOURCE, config=HKJC, write_report=False)
    assert result.ok and not report_file().exists()


# ---------------------------------------------------------------- hard failures


def test_injected_duplicate_runner_is_a_hard_failure(built_root):
    def copy_recent(df):
        return pd.concat([df, df[df["horse_id"] == HORSE_RECENT]], ignore_index=True)

    rewrite("runners", copy_recent)
    result = run()
    assert not result.ok
    (failure,) = result.failures
    assert (failure.name, failure.level, failure.violations, failure.checked) == (
        "unique (race_id, horse_id)", checks.FAIL, 1, 13,
    )  # fmt: skip
    assert failure.examples == [f"{RACE_2026} {HORSE_RECENT}"]
    line = result.error_line()
    assert line is not None and "unique (race_id, horse_id) 1 of 13" in line
    assert str(report_file()) in line
    assert by_name(result)["field_size"].level == checks.WARN  # 5 starters, declared 4
    assert "FAILED" in report_file().read_text(encoding="utf-8")


def test_runners_without_a_horse_id_are_not_duplicates(built_root):
    def blank(df):
        df.loc[df["race_id"] == RACE_2026, "horse_id"] = None
        return pd.concat([df, df[df["race_id"] == RACE_2026]], ignore_index=True)

    rewrite("runners", blank)
    check = by_name(run())["unique (race_id, horse_id)"]
    # the 10 rows of RACE_2026 have no horse id: neither checked nor duplicates
    assert (check.level, check.violations, check.checked) == (checks.OK, 0, 7)


def test_duplicate_race_id_is_a_hard_failure(built_root):
    rewrite("races", lambda df: pd.concat([df, df[df["race_id"] == RACE_2010]], ignore_index=True))
    result = run()
    assert [c.name for c in result.failures] == ["unique race_id"]
    (failure,) = result.failures
    assert (failure.violations, failure.checked, failure.examples) == (1, 5, [RACE_2010])


def test_orphan_race_ids_above_one_percent_fail(built_root):
    rewrite("races", lambda df: df[df["race_id"] != RACE_2026])
    result = run()
    failed = {c.name: c for c in result.failures}
    assert set(failed) == {
        f"orphan race ids in {t}" for t in ("runners", "dividends", "sectionals", "incidents")
    }
    runners = failed["orphan race ids in runners"]
    assert (runners.violations, runners.checked, runners.examples) == (1, 3, [RACE_2026])
    assert (failed["orphan race ids in sectionals"].violations,) == (1,)
    assert by_name(result)["orphan race ids in running_comments"].level == checks.OK


@pytest.mark.parametrize(
    ("n_known", "ghosts", "level", "violations", "checked"),
    [
        (200, [], checks.OK, 0, 200),
        (200, ["ghost", "ghost", None], checks.WARN, 1, 201),  # repeats and nulls count once / not
        (200, ["g1", "g2"], checks.WARN, 2, 202),  # 2 <= 1% of 202
        (200, ["g1", "g2", "g3"], checks.FAIL, 3, 203),  # 3 > 1% of 203
        (100, ["g1"], checks.WARN, 1, 101),  # 1 <= 1.01
        (99, ["g1"], checks.FAIL, 1, 100),  # 1 > 1.00
        (3, ["g1"], checks.FAIL, 1, 4),
    ],
)
def test_orphans_fail_above_one_percent_of_race_ids(n_known, ghosts, level, violations, checked):
    races = pd.DataFrame({"race_id": [f"r{i}" for i in range(n_known)]})
    child = pd.DataFrame({"race_id": [f"r{i}" for i in range(n_known)] * 2 + ghosts})
    check = checks.check_orphans("dividends", child, races)
    assert (check.level, check.violations, check.checked) == (level, violations, checked)


# ---------------------------------------------------------------- counted problems


def test_field_size_mismatch_is_counted_not_fatal(built_root):
    def misdeclare(df):
        df.loc[df["race_id"] == RACE_2010, "field_size"] = 13
        df.loc[df["race_id"] == RACE_ABANDONED, "field_size"] = None  # no declared size: wrong too
        return df

    rewrite("races", misdeclare)
    result = run()
    check = by_name(result)["field_size"]
    assert result.ok
    assert (check.level, check.violations, check.checked) == (checks.WARN, 2, 4)
    assert sorted(check.examples) == sorted([RACE_2010, RACE_ABANDONED])


@pytest.mark.parametrize(
    ("n_races", "n_wrong", "level"),
    [
        (10, 0, checks.OK),
        (10, 1, checks.WARN),
        (10, 5, checks.WARN),  # never a fail for a handful of cases ...
        (10, 6, checks.FAIL),
        (1000, 10, checks.WARN),  # ... nor at 1% of what was checked
        (1000, 11, checks.FAIL),
    ],
)
def test_field_size_becomes_a_failure_once_systematic(n_races, n_wrong, level):
    ids = [f"x:2026-01-01:HV:{i:04d}" for i in range(n_races)]
    runners = pd.DataFrame(
        [{"race_id": r, "is_scratched": False} for r in ids for _ in range(2)]
        + [{"race_id": r, "is_scratched": True} for r in ids]  # scratched: not in the field
    ).astype({"is_scratched": "boolean"})
    sizes = [3 if i < n_wrong else 2 for i in range(n_races)]
    races = pd.DataFrame({"race_id": ids, "field_size": sizes}).astype({"field_size": "Int64"})
    check = checks.check_field_size(races, runners)
    assert (check.level, check.violations, check.checked) == (level, n_wrong, n_races)


def test_wrong_winner_odds_are_counted_not_fatal(built_root):
    def inflate(df):
        df.loc[(df["race_id"] == RACE_2026) & (df["finish_pos"] == 1), "win_odds"] = 50.0
        return df

    rewrite("runners", inflate)
    result = run()
    check = by_name(result)["winner odds vs WIN dividend"]
    assert result.ok and (check.level, check.violations, check.checked) == (checks.WARN, 1, 2)
    assert check.examples[0].startswith(f"{RACE_2026} odds 50.0 dividend")


# ---------------------------------------------------------------- displayed odds vs dividends


def _race(race_id: str, odds, dividend, combination: str = "4", *, refund: bool = False):
    race = {"race_id": race_id, "status": "run"}
    runner = {"race_id": race_id, "horse_no": 4, "finish_pos": 1, "dead_heat": False,
              "win_odds": odds, "is_scratched": False}  # fmt: skip
    paid = {"race_id": race_id, "pool": "WIN", "combination": combination, "dividend": dividend,
            "is_refund": refund}  # fmt: skip
    return race, runner, paid


def _frames(rows):
    races = pd.DataFrame([r[0] for r in rows])
    runners = pd.DataFrame([r[1] for r in rows]).astype(
        {"horse_no": "Int64", "finish_pos": "Int64", "dead_heat": "boolean"}
    )
    dividends = pd.DataFrame([r[2] for r in rows]).astype({"is_refund": "boolean"})
    return races, runners, dividends


@pytest.mark.parametrize(
    ("config", "odds", "dividend", "violation"),
    [
        (HKJC, 5.4, 54.5, False),  # floored to one decimal: 5.45 shows as 5.4
        (HKJC, 5.4, 54.0, False),
        (HKJC, 5.4, 55.0, True),  # would have been displayed as 5.5
        (HKJC, 5.5, 54.0, True),  # odds above the dividend are impossible
        (HKJC, 9.9, 99.5, False),
        (HKJC, 12.0, 129.5, False),  # integers from 10 up: 12.95 shows as 12
        (HKJC, 12.0, 130.0, True),
        (HKJC, 10.0, 100.0, False),
        (HKJC, 99.0, 2514.5, False),  # the display stops at the cap from config
        (HKJC, 98.0, 2514.5, True),
        (JRA, 5.4, 540.0, False),  # JRA dividends are per 100 yen
        (JRA, 5.4, 54.0, True),  # per 100, 54 yen is 0.54: below the displayed odds
        (JRA, 5.5, 540.0, True),
        (JRA, 5.4, 650.0, True),
    ],
)
def test_displayed_odds_are_the_floored_dividend(config, odds, dividend, violation):
    frames = _frames([_race("x:2026-01-01:HV:01", odds, dividend)])
    check = checks.check_win_dividend(*frames, config)
    assert (check.violations, check.checked) == (int(violation), 1)


def test_dead_heats_refunds_and_missing_dividends_are_out_of_scope():
    rows = [
        _race("x:2026-01-01:HV:01", 5.4, 54.5),  # a dead heat once its twin is added below
        _race("x:2026-01-01:HV:02", 3.0, None),  # no dividend
        _race("x:2026-01-01:HV:03", 7.0, 30.0, combination="9"),  # WIN paid on another horse
        _race("x:2026-01-01:HV:04", 2.0, 10.0, refund=True),  # only a refund in the WIN pool
        _race("x:2026-01-01:HV:05", 4.0, 40.0),  # consistent
    ]
    races, runners, dividends = _frames(rows)
    twin = runners.iloc[[0]].assign(horse_no=7)  # a second winner makes race 1 a dead heat
    runners = pd.concat([runners, twin], ignore_index=True)
    check = checks.check_win_dividend(races, runners, dividends, HKJC)
    assert (check.violations, check.checked) == (1, 2)
    assert "2 more without odds" in check.detail  # the race without and the refunded one
    assert check.examples == ["x:2026-01-01:HV:03 odds 7.0 dividend 30.0"]
    assert check.level == checks.WARN


def test_a_comparison_that_drops_every_race_is_not_a_pass():
    rows = [_race(f"x:2026-01-01:HV:{i:02d}", 3.0, None) for i in range(1, 7)]
    check = checks.check_win_dividend(*_frames(rows), HKJC)
    assert check.checked == 0 and check.level == checks.FAIL  # 6 of 6 dropped: systematic
    check = checks.check_win_dividend(*_frames(rows[:1]), HKJC)
    assert check.checked == 0 and check.level == checks.WARN


def test_checks_need_built_tables(raw_root):
    with pytest.raises(FileNotFoundError, match="store build"):
        run()
