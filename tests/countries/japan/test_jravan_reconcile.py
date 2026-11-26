"""Reconcile the jravan store with the netkeiba (jra) store, 2010-2026 (ticket #373).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jravan.reconcile`:

- `reconcile(start: date = date(2010, 1, 1), end: date = date(2026, 11, 23)) -> Reconciliation`
  compares the `races`, `runners` and `dividends` tables of source `jra` and source `jravan`
  (`processed_dir(source)/<name>.parquet`) over `start <= race_date <= end`, both inclusive.
  `races` and `runners` are read through `models.v0.store.read_store_table(..., source=...,
  touches_price=False)`, so holdout days are dropped and rows before 2014-06-07 are read (D10: a
  data check, not a price model). `dividends` (it has no `race_date`) is read only for the race
  ids those `races` reads returned.
- `Reconciliation` is a dataclass with:
  - `by_year: Mapping[int, Mapping[str, int]]`: calendar year of the race's `race_date` -> check
    key -> count. A key absent from a year counts as 0; a year with no rows may be absent.
  - `overall: Mapping[str, int]` (a field or a property): each key summed over the years.
  - `to_markdown() -> str`: the private report: the counts and rates by year and check and
    overall, and the race ids that fail each check, at most 20 per check.
  - `summary() -> str`: the CLI's text: the overall counts and rates, no race id.
- Check keys (a dividend's year is its race's year):
  - `races.both`, `races.jravan_only`, `races.jra_only`: races by `race_id`.
  - `races.equal.<f>`, f in distance_m, surface, field_size, going, post_time_local: matched races
    whose `f` is equal. `post_time_local` compares as minutes of the day ('9:50' == '09:50').
  - `races.win_odds_agree`: matched races whose runners match one to one on `horse_no` (no
    runner in one store only) and whose every runner's `win_odds` agree.
    `races.payouts_agree`: matched races whose dividend rows match one to one on (pool,
    combination) and whose every `dividend` is equal.
  - `runners.both`, `runners.jravan_only`, `runners.jra_only`: runners by (race_id, horse_no).
  - `runners.equal.<f>`, f in horse_id, finish_pos, is_scratched, finish_time_s, win_odds,
    win_fav_rank, jockey_code, horse_wt_kg: matched runners whose `f` is equal. `win_odds` is
    compared after rounding each value to the tenth (3.04 and 3.0 agree; 3.06 and 3.0 do not).
  - `dividends.<POOL>.both`, `.jravan_only`, `.jra_only`, `.equal`: dividend rows by (race_id,
    pool, combination), POOL the store's pool code (`WIN`, `PLACE`, ...).
  - Equal means both values equal or both null; a null against a value differs.
- `horsetrack japan jravan reconcile [--start YYYY-MM-DD] [--end YYYY-MM-DD]` (the defaults of
  `reconcile`) prints `summary()` and writes `to_markdown()` to
  `paths.reports_dir() / 'jravan-reconcile.md'` through `core.eval.reports.write_report(...,
  sources=('jravan', 'jra'))`, with `paths.reports_dir()` looked up at call time on the
  `horsetrack.core.paths` module (as `jravan.records.write_build_report` does). A report path
  inside the repository is refused (`LicensedOutputInRepo`): a non-zero exit, nothing written.

Rates are matched loosely (a percentage with one or two decimals, or a fraction with three or four
decimals), counts as whole numbers. Every value is synthetic and invented; each expected count is
derived from the plan a test writes (how many rows it made differ), never by comparing values.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import uuid
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import env, paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.reports import LicensedOutputInRepo, write_report
from horsetrack.core.store import schemas, tables

# isort: split
# (its own block: the module this ticket creates; one import, so the block sorts the same before
# and after the module exists)
from horsetrack.countries.japan.jravan import reconcile as reconcile_module

Reconciliation = reconcile_module.Reconciliation
reconcile = reconcile_module.reconcile

SOURCES = ("jra", "jravan")
NAMES = ("races", "runners", "dividends")
RACE_FIELDS = ("distance_m", "surface", "field_size", "going", "post_time_local")
RUNNER_FIELDS = (
    "horse_id", "finish_pos", "is_scratched", "finish_time_s", "win_odds", "win_fav_rank",
    "jockey_code", "horse_wt_kg",
)  # fmt: skip
POOLS = (
    "WIN", "PLACE", "BRACKET_QUINELLA", "QUINELLA", "QUINELLA_PLACE", "FORECAST", "TRIO", "TIERCE",
)  # fmt: skip
GOINGS = ("GOOD", "GOOD TO YIELDING", "SOFT", "HEAVY")
SURFACES = ("turf", "dirt")
REPORT = "jravan-reconcile.md"
RACE_ID = re.compile(r"jra:\d{4}-\d{2}-\d{2}:[A-Z]{3}:\d{2}")
# a race's payouts when a test names none: (pool, combination, dividend)
BASE_DIVIDENDS = (
    ("WIN", "1", 250.0), ("PLACE", "1", 120.0), ("PLACE", "2", 180.0), ("PLACE", "3", 260.0),
    ("QUINELLA", "1,2", 890.0), ("TIERCE", "1,2,3", 5430.0),
)  # fmt: skip


# ------------------------------------------------------------------------------------------------
# Two synthetic stores in the store's schemas


def race_id(day: date, venue: str, race_no: int) -> str:
    return f"jra:{day.isoformat()}:{venue}:{race_no:02d}"


def race_keys(day: date, venue: str, race_no: int) -> dict:
    return {
        "jurisdiction": "jra", "race_id": race_id(day, venue, race_no),
        "meeting_id": f"jra:{day.isoformat()}:{venue}", "race_date": day, "venue": venue,
        "race_no": race_no, "season": str(day.year),
    }  # fmt: skip


def hhmm(minutes_of_day: int) -> str:
    return f"{minutes_of_day // 60:02d}:{minutes_of_day % 60:02d}"


def minutes(text: str) -> int:
    hours, mins = text.split(":")
    return int(hours) * 60 + int(mins)


def base_race(day: date, venue: str, race_no: int, n: int = 8) -> dict:
    k = day.toordinal() + race_no
    return race_keys(day, venue, race_no) | {
        "distance_m": 1000 + 200 * (k % 10), "surface": SURFACES[k % 2], "field_size": n,
        "going": GOINGS[k % 4], "post_time_local": hhmm(9 * 60 + 35 * race_no), "status": "run",
    }  # fmt: skip


def base_runner(day: date, venue: str, race_no: int, horse_no: int) -> dict:
    k = (day.toordinal() * 13 + race_no * 29 + horse_no) % 100_000
    return race_keys(day, venue, race_no) | {
        "horse_id": f"{day.year - 3}1{k:05d}", "horse_no": horse_no, "draw": horse_no,
        "finish_pos": horse_no, "is_scratched": False,
        "finish_time_s": round(70.0 + race_no + 0.3 * horse_no, 1),
        "win_odds": round(1.5 + 2.3 * horse_no, 1), "win_fav_rank": horse_no,
        "jockey_code": f"{1000 + (race_no * 37 + horse_no * 11) % 1000:05d}",
        "horse_wt_kg": 430 + 4 * horse_no,
    }  # fmt: skip


# a value for a runner field whose value is null in the other store
NULL_STAND_IN = {
    "horse_id": "2099100001", "finish_pos": 1, "is_scratched": True, "finish_time_s": 80.0,
    "win_odds": 9.9, "win_fav_rank": 1, "jockey_code": "09999", "horse_wt_kg": 470,
}  # fmt: skip


def changed(field: str, value):
    """A runner `field` value that differs from `value`, at the field's precision too."""
    if value is None:
        return NULL_STAND_IN[field]
    if field == "horse_id":
        return str(int(value) + 1)
    if field == "jockey_code":
        return f"{int(value) + 1:05d}"
    if field == "is_scratched":
        return not value
    if field == "finish_time_s":
        return round(value + 0.1, 1)
    if field == "win_odds":
        return round(value + 0.2, 1)
    return value + 1  # finish_pos, win_fav_rank, horse_wt_kg


def changed_race(field: str, value):
    """A race `field` value that differs from `value`."""
    if field == "distance_m":
        return value + 200
    if field == "surface":
        return "dirt" if value == "turf" else "turf"
    if field == "field_size":
        return value + 1
    if field == "going":
        return GOINGS[(GOINGS.index(value) + 1) % len(GOINGS)]
    assert field == "post_time_local"
    return hhmm(minutes(value) + 5)


def with_dividend(pays, pool: str, combination: str, dividend: float) -> tuple:
    return tuple((p, c, dividend if (p, c) == (pool, combination) else d) for p, c, d in pays)


def without(pays, pool: str, combination: str) -> tuple:
    return tuple(row for row in pays if row[:2] != (pool, combination))


def arrow_table(name: str, rows: list[dict]) -> pa.Table:
    """`rows` in the full column list and Arrow types of the store table `name` (absent: null)."""
    spec = schemas.TABLES[name]
    schema = pa.schema([(col, tables.ARROW_TYPES[kind]) for col, kind in spec.columns.items()])
    return pa.table({col: [row.get(col) for row in rows] for col in spec.columns}, schema=schema)


class Stores:
    """The jra and jravan stores, built race by race, then written under the data dir."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], list[dict]] = {(s, n): [] for s in SOURCES for n in NAMES}

    def add(self, day, venue, race_no, *, n=8, sources=SOURCES, race=None, runners=None,
            dividends=None) -> str:  # fmt: skip
        """One race in each of `sources`, with `n` runners and BASE_DIVIDENDS by default.

        `race`: source -> column overrides. `runners`: source -> horse_no -> column overrides, or
        None to leave that runner out; a horse_no above `n` adds a runner to that source only.
        `dividends`: the (pool, combination, dividend) rows of every source, or source -> rows.
        """
        race, runners = race or {}, runners or {}
        rid = race_id(day, venue, race_no)
        for source in sources:
            self.rows[source, "races"].append(
                base_race(day, venue, race_no, n) | race.get(source, {})
            )
            own = runners.get(source, {})
            for horse_no in sorted(set(range(1, n + 1)) | set(own)):
                change = own.get(horse_no, {})
                if change is not None:
                    row = base_runner(day, venue, race_no, horse_no) | change
                    self.rows[source, "runners"].append(row)
            pays = BASE_DIVIDENDS if dividends is None else dividends
            if isinstance(pays, dict):
                pays = pays.get(source, ())
            self.payouts(source, rid, pays)
        return rid

    def payouts(self, source: str, rid: str, pays) -> None:
        for pool, combination, dividend in pays:
            self.rows[source, "dividends"].append(
                {"jurisdiction": "jra", "race_id": rid, "pool": pool,
                 "combination": combination, "dividend": dividend, "is_refund": False}
            )  # fmt: skip

    def write(self, data: Path) -> None:
        for (source, name), rows in self.rows.items():
            folder = data / "processed" / source
            folder.mkdir(parents=True, exist_ok=True)
            # jravan's rows go in reverse order: matching is by key, never by position
            ordered = rows[::-1] if source == "jravan" else rows
            pq.write_table(arrow_table(name, ordered), folder / f"{name}.parquet")


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(root))
    return root


def count(rec, year: int, key: str) -> int:
    return rec.by_year.get(year, {}).get(key, 0)


def total(rec, key: str) -> int:
    return rec.overall.get(key, 0)


def counted_years(rec) -> set[int]:
    return {year for year, counts in rec.by_year.items() if any(counts.values())}


def invoke(*args: str):
    return CliRunner().invoke(app, ["japan", "jravan", "reconcile", *args])


def has_count(text: str, n: int) -> bool:
    """`n` appears as a whole number (not part of another number, a decimal or a percentage)."""
    forms = {str(n), f"{n:,}"}
    return any(re.search(rf"(?<![\d.,]){re.escape(f)}(?![\d,%]|\.\d)", text) for f in forms)


def has_rate(text: str, n: int, d: int) -> bool:
    """`n / d` appears as a percentage (one or two decimals) or a fraction (three or four)."""
    r = n / d
    forms = (f"{100 * r:.1f}%", f"{100 * r:.2f}%", f"{100 * r:.1f} %", f"{100 * r:.2f} %",
             f"{r:.3f}", f"{r:.4f}")  # fmt: skip
    return any(form in text for form in forms)


def listed_ids(text: str) -> set[str]:
    return set(RACE_ID.findall(text))


def assert_overall_is_the_sum(rec) -> None:
    keys = set(rec.overall).union(*(set(counts) for counts in rec.by_year.values()))
    for key in keys:
        assert total(rec, key) == sum(c.get(key, 0) for c in rec.by_year.values()), key


# ------------------------------------------------------------------------------------------------
# Races: matched by race_id, counted per year, compared field by field

# year -> races in both, in jravan only, in jra only; the matched races that differ, per field
RACE_PLAN = {
    2012: {"both": 5, "jravan_only": 1, "jra_only": 2,
           "differ": {"distance_m": 1, "surface": 0, "field_size": 2, "going": 1,
                      "post_time_local": 0}},
    2017: {"both": 7, "jravan_only": 3, "jra_only": 0,
           "differ": {"distance_m": 0, "surface": 2, "field_size": 1, "going": 0,
                      "post_time_local": 3}},
    2023: {"both": 4, "jravan_only": 0, "jra_only": 4,
           "differ": {"distance_m": 2, "surface": 1, "field_size": 0, "going": 3,
                      "post_time_local": 1}},
}  # fmt: skip


def test_races_are_matched_by_race_id_and_counted_per_year(data: Path) -> None:
    stores = Stores()
    for year, plan in RACE_PLAN.items():
        both = plan["both"]
        for i in range(both):
            day = date(year, 3, 2) if i % 2 == 0 else date(year, 10, 5)
            race_no = i // 2 + 1
            base = base_race(day, "NAK", race_no)
            diff = {}
            for j, field in enumerate(RACE_FIELDS):
                # exactly differ[field] of the year's matched races, at an offset per field
                if (i + j) % both < plan["differ"][field]:
                    diff[field] = changed_race(field, base[field])
            stores.add(day, "NAK", race_no, race={"jravan": diff})
        # the one-store races share one meeting, numbered after one another
        day = date(year, 6, 8)
        for r in range(plan["jravan_only"]):
            stores.add(day, "HSN", r + 1, sources=("jravan",))
        for r in range(plan["jra_only"]):
            stores.add(day, "HSN", plan["jravan_only"] + r + 1, sources=("jra",))
    # outside the range asked: never counted
    for day in (date(2011, 12, 31), date(2024, 1, 1)):
        stores.add(day, "NAK", 1, race={"jravan": {"distance_m": 3600, "going": None}})
        stores.add(day, "NAK", 2, sources=("jra",))
        stores.add(day, "NAK", 3, sources=("jravan",))
    stores.write(data)

    rec = reconcile(start=date(2012, 1, 1), end=date(2023, 12, 31))

    assert isinstance(rec, Reconciliation) and dataclasses.is_dataclass(rec)
    for year, plan in RACE_PLAN.items():
        assert count(rec, year, "races.both") == plan["both"], year
        assert count(rec, year, "races.jravan_only") == plan["jravan_only"], year
        assert count(rec, year, "races.jra_only") == plan["jra_only"], year
        for field in RACE_FIELDS:
            want = plan["both"] - plan["differ"][field]
            assert count(rec, year, f"races.equal.{field}") == want, (year, field)
    assert counted_years(rec) == set(RACE_PLAN)
    assert_overall_is_the_sum(rec)
    assert (total(rec, "races.both"), total(rec, "races.jravan_only")) == (16, 4)
    assert total(rec, "races.jra_only") == 6
    assert total(rec, "races.equal.going") == 16 - 4


# ------------------------------------------------------------------------------------------------
# The CLI prints the overall counts and rates


def cli_stores(data: Path) -> None:
    """2019 and 2021 (the expected totals are in the test), and one race on each side of the
    default range."""
    stores = Stores()
    d19 = date(2019, 5, 4)
    for i in range(9):
        no, race, runners, pays = i + 1, {}, {}, None
        if i == 0:
            runners = {"jravan": {9: {}}}  # a runner only in jravan
        if i == 1:
            runners = {"jravan": {7: None, 8: None}}  # two runners only in jra
        if i in (2, 3, 4):  # one runner's Win odds differ
            odds = base_runner(d19, "TOK", no, i - 1)["win_odds"]
            runners = {"jravan": {i - 1: {"win_odds": changed("win_odds", odds)}}}
        if i in (5, 6):
            race = {"jravan": {"distance_m": base_race(d19, "TOK", no)["distance_m"] + 200}}
        if i == 7:
            pays = {"jra": BASE_DIVIDENDS,
                    "jravan": with_dividend(BASE_DIVIDENDS, "PLACE", "2", 190.0)}  # fmt: skip
        if i == 8:  # a payout no count or rate can be: it must never be printed
            pays = with_dividend(BASE_DIVIDENDS, "WIN", "1", 87650.0)
        stores.add(d19, "TOK", no, race=race, runners=runners, dividends=pays)
    for r in range(2):
        stores.add(date(2019, 5, 5), "TOK", r + 1, sources=("jravan",))
    stores.add(date(2019, 5, 5), "TOK", 3, sources=("jra",))

    d21 = date(2021, 9, 11)
    for i in range(6):
        no, runners, pays = i + 1, {}, None
        if i == 0:
            odds = base_runner(d21, "CKY", no, 4)["win_odds"]
            runners = {"jravan": {4: {"win_odds": changed("win_odds", odds)}}}
        if i == 1:
            pays = {"jra": BASE_DIVIDENDS,
                    "jravan": with_dividend(BASE_DIVIDENDS, "WIN", "1", 260.0)}  # fmt: skip
        stores.add(d21, "CKY", no, runners=runners, dividends=pays)
    stores.add(date(2021, 9, 12), "CKY", 1, sources=("jravan",))
    for r in range(3):
        stores.add(date(2021, 9, 12), "CKY", r + 2, sources=("jra",))

    # outside the default range (2010-01-01 to 2026-11-23): matched and identical
    stores.add(date(2009, 12, 26), "NAK", 11)
    stores.add(date(2026, 11, 24), "TOK", 11)
    stores.write(data)


def test_the_cli_prints_the_overall_counts(data: Path) -> None:
    cli_stores(data)
    # 2019 + 2021: races 15 in both, 3 jravan only, 4 jra only; runners 118 in both (9 x 8 - 2 +
    # 6 x 8), 25 jravan only (1 + 3 races x 8), 34 jra only (2 + 4 races x 8)
    full_counts = {"races.both": 15, "races.jravan_only": 3, "races.jra_only": 4,
                   "runners.both": 118, "runners.jravan_only": 25, "runners.jra_only": 34,
                   "runners.equal.win_odds": 114, "races.equal.distance_m": 13,
                   "dividends.WIN.both": 15, "dividends.WIN.equal": 14,
                   "dividends.PLACE.both": 45, "dividends.PLACE.equal": 44}  # fmt: skip
    full_rates = ((114, 118), (13, 15), (14, 15), (44, 45))

    rec = reconcile(start=date(2010, 1, 1), end=date(2026, 11, 23))
    assert {key: total(rec, key) for key in full_counts} == full_counts

    full = invoke("--start", "2010-01-01", "--end", "2026-11-23")
    assert full.exit_code == 0, full.output
    out = full.output
    for key in ("races.both", "races.jravan_only", "races.jra_only", "runners.both",
                "runners.jravan_only", "runners.jra_only"):  # fmt: skip
        assert has_count(out, full_counts[key]), (key, out)
    for n, d in full_rates:
        assert has_rate(out, n, d), (n, d, out)
    for line in rec.summary().splitlines():  # the CLI's text is the summary
        assert line.strip() in out, line
    # counts and rates only: no race id, no payout
    assert not listed_ids(out), out
    assert "87650" not in out and "87,650" not in out

    # the defaults are 2010-01-01 to 2026-11-23: the races on 2009-12-26 and 2026-11-24 are out
    default = invoke()
    assert default.exit_code == 0, default.output
    for line in reconcile().summary().splitlines():
        assert line.strip() in default.output, line
    for key in ("races.both", "runners.both", "runners.jra_only"):
        assert has_count(default.output, full_counts[key]), key
    assert not has_count(default.output, 118 + 16)
    wide = invoke("--start", "2009-01-01", "--end", "2026-11-30")
    assert wide.exit_code == 0, wide.output
    assert has_count(wide.output, 15 + 2) and has_count(wide.output, 118 + 16)

    # 2021 alone: races 6 / 1 / 3, runners 48 / 8 / 24, Win odds 47 of 48, WIN payouts 5 of 6
    narrow = invoke("--start", "2021-01-01", "--end", "2021-12-31")
    assert narrow.exit_code == 0, narrow.output
    for n in (6, 3, 48, 8, 24):
        assert has_count(narrow.output, n), (n, narrow.output)
    for n, d in ((47, 48), (5, 6)):
        assert has_rate(narrow.output, n, d), (n, d, narrow.output)
    assert not has_count(narrow.output, 118) and not has_rate(narrow.output, 114, 118)


# ------------------------------------------------------------------------------------------------
# Runners: matched on (race_id, horse_no), compared field by field

RUNNER_PLAN = {2015: (3, 10), 2020: (3, 8)}  # year -> (matched races, runners per race)
RUNNER_DIFFS = {  # field -> runners that differ in (2015, 2020)
    "horse_id": (1, 2), "finish_pos": (2, 0), "is_scratched": (3, 1), "finish_time_s": (0, 3),
    "win_odds": (2, 1), "win_fav_rank": (5, 1), "jockey_code": (2, 4), "horse_wt_kg": (6, 3),
}  # fmt: skip
# fields whose first differing runner is a value in jra against a null in jravan
NULLABLE = ("finish_pos", "finish_time_s", "win_odds", "win_fav_rank", "horse_wt_kg")
# the year's first race also holds runners in one store only: source -> horse_no -> overrides
ONE_STORE = {2015: {"jravan": {11: {}}, "jra": {12: {}, 13: {}}}, 2020: {"jravan": {9: {}, 10: {}}}}


def test_runner_fields_agree_or_disagree_field_by_field(data: Path) -> None:
    stores = Stores()
    want_agree = {}
    for year_index, (year, (n_races, n)) in enumerate(RUNNER_PLAN.items()):
        day = date(year, 5, 2)
        flat = [(r, h) for r in range(n_races) for h in range(1, n + 1)]
        size = len(flat)
        # null in both stores, so equal: one runner not weighed and unranked, one scratched
        both_null = {
            flat[-1]: {"horse_wt_kg": None, "win_fav_rank": None},
            flat[-2]: {"is_scratched": True, "finish_pos": None, "finish_time_s": None,
                       "win_odds": None, "win_fav_rank": None},
        }  # fmt: skip
        jra = {r: {} for r in range(n_races)}
        jravan = {r: {} for r in range(n_races)}
        for (r, h), change in both_null.items():
            jra[r][h], jravan[r][h] = dict(change), dict(change)
        odds_races = set()
        for j, field in enumerate(RUNNER_FIELDS):
            k = RUNNER_DIFFS[field][year_index]
            order = ((3 * j + 7 * m) % size for m in range(size))  # 7 is prime to both sizes
            picked = [i for i in order if flat[i] not in both_null][:k]
            for t, i in enumerate(picked):
                r, h = flat[i]
                now = base_runner(day, "TOK", r + 1, h)[field]
                new = None if t == 0 and field in NULLABLE else changed(field, now)
                jravan[r].setdefault(h, {})[field] = new
                if field == "win_odds":
                    odds_races.add(r)
        for r in range(n_races):
            runners = {"jra": dict(jra[r]), "jravan": dict(jravan[r])}
            if r == 0:
                for source, horses in ONE_STORE[year].items():
                    runners[source].update(horses)
            stores.add(day, "TOK", r + 1, n=n, runners=runners)
        want_agree[year] = n_races - len(odds_races | {0})
    stores.write(data)
    assert any(want_agree.values())  # the plan leaves some race whose every Win odds agree

    rec = reconcile(start=date(2015, 1, 1), end=date(2020, 12, 31))

    for year_index, (year, (n_races, n)) in enumerate(RUNNER_PLAN.items()):
        size = n_races * n
        assert count(rec, year, "runners.both") == size, year
        for source in SOURCES:
            want = len(ONE_STORE[year].get(source, {}))
            assert count(rec, year, f"runners.{source}_only") == want, (year, source)
        for field in RUNNER_FIELDS:
            want = size - RUNNER_DIFFS[field][year_index]
            assert count(rec, year, f"runners.equal.{field}") == want, (year, field)
        assert count(rec, year, "races.win_odds_agree") == want_agree[year], year
        assert count(rec, year, "races.both") == n_races, year
    assert_overall_is_the_sum(rec)
    for field in RUNNER_FIELDS:
        assert total(rec, f"runners.equal.{field}") == 30 + 24 - sum(RUNNER_DIFFS[field]), field


# ------------------------------------------------------------------------------------------------
# Win odds agree at the tenth

ODDS_CASES = (  # (jra win_odds, jravan win_odds, agree at the tenth)
    (3.4, 3.4, True),
    (3.4, 3.4000000001, True),
    (1.1 + 2.2, 3.3, True),  # 3.3000000000000003
    (2.9999999999, 3.0, True),
    (3.04, 3.0, True),
    (5.0, 5.04, True),
    (12.3, 12.30001, True),
    (999.9, 999.9, True),
    (3.06, 3.0, False),
    (3.4, 3.5, False),
    (3.4, 4.3, False),
    (99.9, 999.0, False),
    (10.0, 1.0, False),
    (None, 2.5, False),
    (7.7, None, False),
    (None, None, True),  # scratched in both
)
# The near-misses below and above a tenth (2.9999999999 against 3.0, 3.06 against 3.0) sit in
# different years, so a comparison that truncates instead of rounding cannot balance out.
ODDS_RACES = (  # (day, venue, race_no, the cases its runners 1.. carry, a runner only in jra)
    (date(2016, 4, 2), "TOK", 1, (0, 1, 2, 3), False),
    (date(2016, 4, 2), "TOK", 2, (4, 9), False),
    (date(2016, 4, 2), "TOK", 3, (5, 6, 7, 15), False),
    (date(2018, 10, 6), "KYO", 1, (8, 10, 11, 12, 13, 14), False),
    (date(2018, 10, 6), "KYO", 2, (), True),
    (date(2018, 10, 6), "KYO", 3, (), False),
)


def test_win_odds_agree_at_the_tenth(data: Path) -> None:
    stores = Stores()
    want = {}  # year -> [runners in both, Win odds equal, races whose every Win odds agree]
    for day, venue, race_no, cases, jra_only in ODDS_RACES:
        jra, jravan = {}, {}
        for h, case in enumerate(cases, start=1):
            mine, theirs, _ = ODDS_CASES[case]
            scratched = {"is_scratched": True, "finish_pos": None} if mine is theirs is None else {}
            jra[h] = {"win_odds": mine} | scratched
            jravan[h] = {"win_odds": theirs} | scratched
        if jra_only:
            jravan[8] = None
        stores.add(day, venue, race_no, runners={"jra": jra, "jravan": jravan})
        both = 8 - (1 if jra_only else 0)
        differ = sum(1 for case in cases if not ODDS_CASES[case][2])
        year = want.setdefault(day.year, [0, 0, 0])
        year[0] += both
        year[1] += both - differ
        year[2] += 1 if differ == 0 and not jra_only else 0
    stores.write(data)

    rec = reconcile(start=date(2016, 1, 1), end=date(2018, 12, 31))

    for year, (both, equal, agree) in want.items():
        assert count(rec, year, "runners.both") == both, year
        assert count(rec, year, "runners.equal.win_odds") == equal, year
        assert count(rec, year, "races.win_odds_agree") == agree, year
        # the other fields agree throughout
        assert count(rec, year, "runners.equal.finish_pos") == both, year
    assert count(rec, 2018, "runners.jra_only") == 1
    assert (total(rec, "runners.equal.win_odds"), total(rec, "races.win_odds_agree")) == (40, 3)


# ------------------------------------------------------------------------------------------------
# Dividends per pool and combination

FULL = (
    ("WIN", "5", 450.0), ("PLACE", "5", 160.0), ("PLACE", "2", 210.0), ("PLACE", "9", 340.0),
    ("QUINELLA", "2,5", 1230.0), ("FORECAST", "5,2", 2110.0), ("TRIO", "2,5,9", 3450.0),
    ("TIERCE", "5,2,9", 15670.0),
)  # fmt: skip
WIDE = (("QUINELLA_PLACE", "2,5", 520.0), ("QUINELLA_PLACE", "2,9", 880.0),
        ("QUINELLA_PLACE", "5,9", 1010.0))  # fmt: skip
DEAD_HEAT = (
    ("WIN", "4", 210.0), ("WIN", "7", 190.0), ("PLACE", "4", 110.0), ("PLACE", "7", 100.0),
    ("PLACE", "1", 150.0), ("BRACKET_QUINELLA", "2,4", 330.0),
)  # fmt: skip
R8 = (
    ("WIN", "1", 230.0), ("PLACE", "1", 120.0), ("PLACE", "6", 150.0), ("PLACE", "3", 200.0),
    ("TRIO", "1,3,6", 2340.0), ("BRACKET_QUINELLA", "1,4", 560.0),
)  # fmt: skip
R9 = (
    ("WIN", "2", 300.0), ("PLACE", "2", 130.0), ("PLACE", "8", 220.0),
    ("QUINELLA", "2,8", 1500.0), ("TIERCE", "2,8,5", 9870.0),
)  # fmt: skip
DIV_EXPECTED = {  # year -> pool -> (in both, equal, jravan only, jra only), counted by hand
    2022: {
        "WIN": (6, 5, 1, 0), "PLACE": (14, 14, 3, 1), "BRACKET_QUINELLA": (1, 1, 0, 0),
        "QUINELLA": (4, 4, 1, 0), "QUINELLA_PLACE": (3, 3, 0, 0), "FORECAST": (3, 3, 2, 1),
        "TRIO": (4, 4, 1, 0), "TIERCE": (4, 4, 1, 0),
    },
    2013: {
        "WIN": (2, 2, 0, 1), "PLACE": (5, 4, 0, 1), "BRACKET_QUINELLA": (1, 1, 0, 0),
        "QUINELLA": (1, 1, 0, 0), "QUINELLA_PLACE": (0, 0, 0, 0), "FORECAST": (0, 0, 0, 0),
        "TRIO": (1, 1, 0, 0), "TIERCE": (1, 1, 0, 0),
    },
}  # fmt: skip


def test_dividends_are_compared_per_pool_and_combination(data: Path) -> None:
    stores = Stores()
    d22 = date(2022, 6, 4)
    # R1: every payout agrees
    stores.add(d22, "TOK", 1, dividends=FULL)
    # R2: the WIN payout differs
    stores.add(
        d22, "TOK", 2, dividends={"jra": FULL, "jravan": with_dividend(FULL, "WIN", "5", 460.0)}
    )
    # R3: jravan lacks one PLACE combination
    stores.add(
        d22, "TOK", 3, dividends={"jra": FULL + WIDE, "jravan": without(FULL, "PLACE", "9") + WIDE}
    )
    # R4: FORECAST is ordered, so 5,2 and 2,5 are two combinations
    swapped = without(FULL, "FORECAST", "5,2") + (("FORECAST", "2,5", 2110.0),)
    stores.add(d22, "TOK", 4, dividends={"jra": FULL, "jravan": swapped})
    # R5: a dead heat, two WIN rows; every payout agrees
    stores.add(d22, "HSN", 5, dividends=DEAD_HEAT)
    # R6: a race only in jravan
    stores.add(d22, "HSN", 6, sources=("jravan",), dividends=FULL)
    # outside the range, and payouts of a race in neither races table: never counted
    stores.add(
        date(2023, 1, 5),
        "TOK",
        1,
        dividends={"jra": FULL, "jravan": with_dividend(FULL, "WIN", "5", 990.0)},
    )
    orphan = race_id(date(2022, 7, 2), "KYO", 12)
    stores.payouts("jra", orphan, FULL)
    stores.payouts("jravan", orphan, with_dividend(without(FULL, "PLACE", "2"), "WIN", "5", 10.0))

    d13 = date(2013, 4, 6)
    # R8: one PLACE payout differs; R9: every payout agrees; R10: a race only in jra
    stores.add(
        d13, "NAK", 7, dividends={"jra": R8, "jravan": with_dividend(R8, "PLACE", "6", 160.0)}
    )
    stores.add(d13, "NAK", 8, dividends=R9)
    stores.add(
        d13, "NAK", 9, sources=("jra",), dividends=(("WIN", "3", 500.0), ("PLACE", "3", 200.0))
    )
    stores.write(data)

    rec = reconcile(start=date(2013, 1, 1), end=date(2022, 12, 31))

    for year, pools in DIV_EXPECTED.items():
        for pool, (both, equal, jravan_only, jra_only) in pools.items():
            got = tuple(count(rec, year, f"dividends.{pool}.{check}")
                        for check in ("both", "equal", "jravan_only", "jra_only"))  # fmt: skip
            assert got == (both, equal, jravan_only, jra_only), (year, pool)
    assert count(rec, 2022, "races.payouts_agree") == 2  # R1, R5
    assert count(rec, 2013, "races.payouts_agree") == 1  # R9
    assert (count(rec, 2022, "races.both"), count(rec, 2013, "races.both")) == (5, 2)
    assert counted_years(rec) == {2013, 2022}
    assert_overall_is_the_sum(rec)
    for pool in POOLS:
        want = tuple(sum(DIV_EXPECTED[y][pool][c] for y in DIV_EXPECTED) for c in range(4))
        got = tuple(total(rec, f"dividends.{pool}.{check}")
                    for check in ("both", "equal", "jravan_only", "jra_only"))  # fmt: skip
        assert got == want, pool


# ------------------------------------------------------------------------------------------------
# Post times compare as minutes of the day

POST_CASES = (  # (jra post_time_local, jravan post_time_local, the same minute of the day)
    ("9:50", "09:50", True),
    ("09:50", "9:50", True),
    ("9:05", "09:05", True),
    ("10:00", "10:00", True),
    ("15:45", "15:45", True),
    ("9:50", "09:55", False),
    ("9:50", "10:50", False),
    ("9:50", "21:50", False),
    ("12:30", "13:30", False),
    ("10:10", "10:01", False),
    (None, None, True),
    ("11:20", None, False),
    (None, "11:20", False),
)


def test_post_times_compare_as_minutes_of_the_day(data: Path) -> None:
    stores = Stores()
    want = {}  # year -> [matched races, equal post times]
    for i, (mine, theirs, same) in enumerate(POST_CASES):
        year = 2011 if i % 2 == 0 else 2024
        day = date(year, 7, 2)
        stores.add(
            day,
            "FKS",
            i // 2 + 1,
            race={"jra": {"post_time_local": mine}, "jravan": {"post_time_local": theirs}},
        )
        counts = want.setdefault(year, [0, 0])
        counts[0] += 1
        counts[1] += 1 if same else 0
    stores.write(data)

    rec = reconcile(start=date(2011, 1, 1), end=date(2024, 12, 31))

    for year, (both, equal) in want.items():
        assert count(rec, year, "races.both") == both, year
        assert count(rec, year, "races.equal.post_time_local") == equal, year
        for field in ("distance_m", "surface", "field_size", "going"):
            assert count(rec, year, f"races.equal.{field}") == both, (year, field)
    assert total(rec, "races.equal.post_time_local") == sum(1 for c in POST_CASES if c[2])


# ------------------------------------------------------------------------------------------------
# Holdout days are never read


def test_holdout_days_are_never_read(data: Path) -> None:
    late = [date(2026, 11, 28) + timedelta(days=k) for k in range(28)]
    held = [d for d in late if is_holdout_day("japan", d)][:3]
    open_late = [d for d in late if not is_holdout_day("japan", d)][:2]
    assert len(held) == 3 and len(open_late) == 2  # the committed rule holds out some days only

    stores = Stores()
    for r in range(3):
        stores.add(date(2026, 11, 22), "TOK", r + 1)
    stores.add(date(2026, 11, 21), "TOK", 1, sources=("jravan",))
    held_ids = []
    for day in held:  # every check would fail on these races if they were read
        base = base_race(day, "KYO", 11)
        odds = base_runner(day, "KYO", 11, 1)["win_odds"]
        held_ids.append(stores.add(
            day, "KYO", 11,
            race={"jravan": {"distance_m": base["distance_m"] + 200,
                             "going": changed_race("going", base["going"])}},
            runners={"jravan": {1: {"win_odds": changed("win_odds", odds)}, 9: {}}},
            dividends={"jra": BASE_DIVIDENDS,
                       "jravan": with_dividend(BASE_DIVIDENDS, "WIN", "1", 990.0)},
        ))  # fmt: skip
        held_ids.append(stores.add(day, "KYO", 12, sources=("jravan",)))
        held_ids.append(stores.add(day, "KYO", 10, sources=("jra",)))
    for day in open_late:  # development days after the holdout start
        stores.add(day, "HSN", 1)
    stores.write(data)

    # an end past the held days: only the development days are counted
    rec = reconcile(start=date(2026, 11, 1), end=date(2026, 12, 31))
    assert counted_years(rec) == {2026}
    want = {
        "races.both": 5, "races.jravan_only": 1, "races.jra_only": 0,
        "races.equal.distance_m": 5, "races.equal.going": 5,
        "runners.both": 40, "runners.jravan_only": 8, "runners.jra_only": 0,
        "runners.equal.win_odds": 40, "races.win_odds_agree": 5, "races.payouts_agree": 5,
        "dividends.WIN.both": 5, "dividends.WIN.equal": 5, "dividends.WIN.jravan_only": 1,
        "dividends.WIN.jra_only": 0, "dividends.PLACE.both": 15, "dividends.PLACE.jra_only": 0,
    }  # fmt: skip
    assert {key: count(rec, 2026, key) for key in want} == want
    text = rec.to_markdown() + "\n" + rec.summary()
    assert not listed_ids(text) & set(held_ids)
    for day in held:
        assert day.isoformat() not in text, day

    # the default end, 2026-11-23, stops before them
    rec = reconcile(start=date(2026, 11, 1))
    got = {
        key: count(rec, 2026, key)
        for key in ("races.both", "races.jravan_only", "runners.both", "dividends.WIN.both")
    }
    assert got == {"races.both": 3, "races.jravan_only": 1, "runners.both": 24,
                   "dividends.WIN.both": 3}  # fmt: skip


# ------------------------------------------------------------------------------------------------
# Pre-reform prices are compared (D10: a data check, not a price model)


def odds_differ(day: date, venue: str, race_no: int, horses) -> dict:
    return {"jravan": {h: {"win_odds": changed("win_odds", base_runner(day, venue, race_no, h)
                                               ["win_odds"])} for h in horses}}  # fmt: skip


def test_pre_reform_prices_are_compared(data: Path) -> None:
    stores = Stores()
    stores.add(
        date(2009, 12, 27), "NAK", 11, runners=odds_differ(date(2009, 12, 27), "NAK", 11, (1, 2))
    )
    stores.add(
        date(2010, 1, 5),
        "NAK",
        1,
        runners=odds_differ(date(2010, 1, 5), "NAK", 1, (3,)),
        dividends={
            "jra": BASE_DIVIDENDS,
            "jravan": with_dividend(BASE_DIVIDENDS, "WIN", "1", 270.0),
        },
    )
    stores.add(date(2010, 1, 9), "KYO", 2)
    stores.add(date(2010, 1, 9), "KYO", 3, sources=("jravan",))
    stores.add(date(2013, 12, 28), "NAK", 11)
    stores.add(date(2014, 6, 1), "TOK", 5, runners=odds_differ(date(2014, 6, 1), "TOK", 5, (2, 6)))
    stores.add(date(2014, 6, 7), "TOK", 5)
    stores.add(date(2026, 11, 23), "KYO", 11)  # the default end is inclusive
    stores.add(date(2026, 11, 24), "KYO", 1)  # after it
    stores.write(data)

    rec = reconcile()  # 2010-01-01 to 2026-11-23

    assert counted_years(rec) == {2010, 2013, 2014, 2026}
    want_2010 = {
        "races.both": 2, "races.jravan_only": 1, "runners.both": 16, "runners.jravan_only": 8,
        "runners.equal.win_odds": 15, "races.win_odds_agree": 1, "races.payouts_agree": 1,
        "dividends.WIN.both": 2, "dividends.WIN.equal": 1, "dividends.WIN.jravan_only": 1,
    }  # fmt: skip
    assert {key: count(rec, 2010, key) for key in want_2010} == want_2010
    assert (count(rec, 2013, "races.both"), count(rec, 2013, "runners.equal.win_odds")) == (1, 8)
    assert (count(rec, 2014, "races.both"), count(rec, 2014, "runners.equal.win_odds")) == (2, 14)
    assert count(rec, 2026, "races.both") == 1
    assert (total(rec, "races.both"), total(rec, "runners.both")) == (6, 48)

    # a range that ends before the price era (2014-06-07) reads too
    rec = reconcile(start=date(2010, 1, 1), end=date(2014, 6, 6))
    assert counted_years(rec) == {2010, 2013, 2014}
    assert (count(rec, 2014, "races.both"), count(rec, 2014, "runners.equal.win_odds")) == (1, 6)
    assert count(rec, 2010, "dividends.WIN.equal") == 1


# ------------------------------------------------------------------------------------------------
# The private report: counts and rates, at most 20 race ids per check, never in the repository


def report_stores(data: Path) -> dict[str, list[str]]:
    """Race ids by role. 2019: 23 races differ on distance only, 2 on going only, 4 agree, 2 are
    only in jra. 2021: 1 differs on going only, 5 agree, 1 is only in jravan."""
    stores = Stores()
    ids: dict[str, list[str]] = {k: [] for k in ("distance", "going", "agree", "one_store")}
    d19 = date(2019, 4, 6)
    for i in range(23):  # more than the 20 ids a check lists
        venue, no = ("SAP", "HAK", "FKS")[i % 3], i // 3 + 1
        distance = base_race(d19, venue, no)["distance_m"] + 200
        ids["distance"].append(
            stores.add(d19, venue, no, race={"jravan": {"distance_m": distance}})
        )
    for r in range(2):
        going = changed_race("going", base_race(d19, "KOK", r + 1)["going"])
        ids["going"].append(stores.add(d19, "KOK", r + 1, race={"jravan": {"going": going}}))
    ids["agree"] += [stores.add(d19, "KOK", r + 3) for r in range(4)]
    ids["one_store"] += [stores.add(d19, "KOK", r + 7, sources=("jra",)) for r in range(2)]
    d21 = date(2021, 11, 13)
    going = changed_race("going", base_race(d21, "TOK", 1)["going"])
    ids["going"].append(stores.add(d21, "TOK", 1, race={"jravan": {"going": going}}))
    ids["agree"] += [stores.add(d21, "TOK", r + 2) for r in range(5)]
    ids["one_store"].append(stores.add(d21, "TOK", 7, sources=("jravan",)))
    stores.write(data)
    return ids


def test_the_report_holds_counts_and_rates_and_is_refused_inside_the_repo(
    data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = report_stores(data)
    start, end = date(2019, 1, 1), date(2021, 12, 31)
    # 2019: 29 in both, distance equal 6 of 29, going 27 of 29; 2021: 6 in both, going 5 of 6;
    # overall: 35 in both, distance 12 of 35, going 32 of 35
    rec = reconcile(start=start, end=end)
    assert (count(rec, 2019, "races.both"), count(rec, 2021, "races.both")) == (29, 6)
    assert count(rec, 2019, "races.equal.distance_m") == 6
    assert (total(rec, "races.equal.distance_m"), total(rec, "races.equal.going")) == (12, 32)

    def check_report(text: str) -> None:
        assert "2019" in text and "2021" in text
        for n in (29, 6, 35):
            assert has_count(text, n), n
        for n, d in ((6, 29), (27, 29), (5, 6), (12, 35), (32, 35)):
            assert has_rate(text, n, d), (n, d)
        listed = listed_ids(text)
        assert len(listed & set(ids["distance"])) == 20  # at most 20 per check
        assert set(ids["going"]) <= listed
        assert set(ids["one_store"]) <= listed
        assert not listed & set(ids["agree"])  # only the races failing a check are listed

    check_report(rec.to_markdown())

    result = invoke("--start", start.isoformat(), "--end", end.isoformat())
    assert result.exit_code == 0, result.output
    report = data / "reports" / REPORT
    assert paths.reports_dir() / REPORT == report
    assert report.is_file()
    check_report(report.read_text(encoding="utf-8"))
    # the printed summary holds the overall counts and rates, and no race id
    assert has_count(result.output, 35) and has_count(result.output, 2)
    assert has_rate(result.output, 12, 35) and has_rate(result.output, 32, 35)
    assert not listed_ids(result.output)

    # a reports folder inside the repository is refused, and nothing is created there
    inside = env.repo_root() / f".tmp-jravan-reconcile-{uuid.uuid4().hex[:8]}" / "reports"
    monkeypatch.setattr(paths, "reports_dir", lambda: inside)
    if hasattr(reconcile_module, "reports_dir"):  # in case the module imported the name
        monkeypatch.setattr(reconcile_module, "reports_dir", lambda: inside)
    try:
        refused = invoke("--start", start.isoformat(), "--end", end.isoformat())
        assert refused.exit_code != 0, refused.output
        assert not inside.parent.exists()
        with pytest.raises(LicensedOutputInRepo):
            write_report(inside / REPORT, rec.to_markdown(), sources=("jravan", "jra"))
        assert not inside.parent.exists()
    finally:
        shutil.rmtree(inside.parent, ignore_errors=True)
