"""NAR and overseas runs from JV-Data RA/SE kept as horse form under their own jurisdiction (#377).

Contract pinned here (the coder follows it):

- `keys.form_race_id(year, mmdd, venue_code, race_no)`: `<jurisdiction>:<YYYY-MM-DD>:<code>:<NN>`
  for a venue `keys.jurisdiction` puts in `nar` or `overseas`, the venue code as JV-Data writes
  it (`nar:2025-06-01:30:01`, `overseas:2025-10-05:A4:09`). Race number '00', which JV-Data
  writes for some overseas races, is a valid race number for any form venue
  (`overseas:2025-10-05:A4:00`, `nar:2025-06-01:30:00`); None for a JRA venue (01-10, which
  keeps `keys.race_id`), a venue with no jurisdiction ('00', blank), an invalid date or a race
  number that is not two digits. `keys.form_meeting_id(year, mmdd, venue_code)` is the same
  without the race. `keys.race_id` and `keys.meeting_id` stay JRA-only, and `keys.race_id` keeps
  refusing race number '00'.
- `races.build_races` writes, beside `races.parquet` and `meetings.parquet`, the kept NAR and
  overseas races (one version per race, deletions dropped, as for JRA) to
  `<out>/form_races.parquet`: the columns and Arrow types of `TABLES['races']`, sorted by its
  sort key, `jurisdiction` 'nar' or 'overseas', `race_id`/`meeting_id` the form ids, `venue` the
  JV-Data venue code, the rest mapped as a JRA race is. `runners.build_runners` writes the kept
  NAR and overseas runners to `<out>/form_runners.parquet` the same way (`TABLES['runners']`).
  A venue with no jurisdiction is written nowhere. Each year of both reports counts, as
  `form_without_id`, the kept NAR and overseas records that got no form id (an invalid date or
  race number) and so reached no form table: 0 when every one has an id, so no loss is silent;
  the report's totals carry it too. `races.parquet`, `meetings.parquet` and
  `runners.parquet` stay JRA-only and byte-identical to a build without the non-JRA records, and
  the reports' `unmapped` codes stay those of the JRA rows (the non_jra_* counts stay as they are).
- `horse_runs.build_horse_runs` adds one row per form runner with a horse_id, the columns taken
  as for a JRA run (the race's from `form_races`, the rest from `form_runners`), `jurisdiction`
  the run's, `win_odds` null (no JRA price: a form run is never a price). The form tables are
  read through `read_store_table(..., source='jravan')` with the holdout days excluded in the
  scan, and their `win_odds` is never scanned. Without the form files the build is as before.
- Only modules under `countries/japan/jravan/` name the form tables, so no betting or research
  reader reads them.

Every value is synthetic and invented, under a tmp HORSETRACK_DATA_DIR, in the made-up layouts
of `test_jravan_runners` (D21). Expected values are written out by hand or derived from the
fixture's own rows.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.store import tables
from horsetrack.countries.japan.jravan import horse_runs as hr
from horsetrack.countries.japan.jravan import keys, races, runners
from horsetrack.countries.japan.models.v0 import store as v0_store

# isort: split
# (its own block: the sibling helper modules)
import jravan_workbook as wb
import test_jravan_horse_runs as hrt
import test_jravan_runners as rn

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src" / "horsetrack"
JRAVAN_PACKAGE = SRC / "countries" / "japan" / "jravan"
SOURCE = "jravan"
FORM_RACES, FORM_RUNNERS = "form_races", "form_runners"
TABLE = "horse_runs"


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp data dir with the made-up RA/SE workbook in place."""
    d = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    rn.write_workbook(d / "specs" / "jravan" / wb.WORKBOOK_NAME)
    return d


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp data dir for store-level fixtures (no workbook)."""
    d = (tmp_path / "store").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


def processed(data: Path) -> Path:
    return data / "processed" / SOURCE


def write_raw(raw: Path, record_type: str, year: int, recs: list[bytes]) -> Path:
    return wb.write_jvd(raw / "stored" / "RACE" / record_type / f"{year}.jvd", recs)


def read(path: Path) -> list[dict]:
    assert path.is_file(), f"no {path} written"
    return pq.read_table(path).to_pylist()


# ------------------------------------------------------------------------------------------------
# Form ids for NAR and overseas venues


FORM_ID_CASES = [  # (year, mmdd, venue, race) -> race id
    (("2025", "0601", "30", "01"), "nar:2025-06-01:30:01"),
    (("2019", "1229", "44", "11"), "nar:2019-12-29:44:11"),
    (("2024", "0229", "47", "12"), "nar:2024-02-29:47:12"),
    (("2012", "0815", "11", "03"), "nar:2012-08-15:11:03"),
    (("2025", "1005", "A4", "09"), "overseas:2025-10-05:A4:09"),
    (("2016", "1002", "B6", "04"), "overseas:2016-10-02:B6:04"),
    (("2010", "0105", "C0", "01"), "overseas:2010-01-05:C0:01"),
    # race number 00, as JV-Data writes it for some overseas races: a race of its own
    (("2025", "1005", "A4", "00"), "overseas:2025-10-05:A4:00"),
    (("2003", "0413", "B6", "00"), "overseas:2003-04-13:B6:00"),
    (("2024", "0229", "C0", "00"), "overseas:2024-02-29:C0:00"),
    (("2025", "0601", "30", "00"), "nar:2025-06-01:30:00"),
]


def test_form_race_ids_carry_the_jurisdiction_date_venue_code_and_race() -> None:
    for (year, mmdd, venue, race), expected in FORM_ID_CASES:
        assert keys.form_race_id(year, mmdd, venue, race) == expected, (venue, race)
        assert keys.form_meeting_id(year, mmdd, venue) == expected.rsplit(":", 1)[0], venue
        assert expected.split(":")[0] == keys.jurisdiction(venue)
        # the JRA ids stay JRA-only
        assert keys.race_id(year, mmdd, venue, race) is None
    # a JRA venue keeps keys.race_id; a venue without a jurisdiction has no id at all
    for venue in [f"{n:02d}" for n in range(1, 11)] + ["00", "  ", ""]:
        assert keys.form_race_id("2025", "0601", venue, "01") is None, venue
        assert keys.form_meeting_id("2025", "0601", venue) is None, venue
    for year, mmdd, race in [("2025", "0229", "01"), ("2026", "1301", "01"), ("2026", "0000", "01"),
                             ("    ", "0601", "01"), ("2025", "0229", "00"),
                             ("2025", "0601", "  "), ("2025", "0601", "0"), ("2025", "0601", "1 "),
                             ("2025", "0601", "000")]:  # fmt: skip
        assert keys.form_race_id(year, mmdd, "30", race) is None, (year, mmdd, race)
        assert keys.form_race_id(year, mmdd, "A4", race) is None, (year, mmdd, race)
    # race number 00 stays refused for a JRA race
    for venue in ("01", "05", "10"):
        assert keys.race_id("2025", "0601", venue, "00") is None, venue


# ------------------------------------------------------------------------------------------------
# The races build: form_races beside a JRA-only races file

DIRT = dict(track="24", turf="0", dirt="1", course="")
NAR30 = dict(mmdd="0601", venue="30", kai="01", nichi="01", kubun="A", **DIRT)
RA_JRA_2024 = [rn.ra(year="2024", mmdd="1201", venue="06", kai="05", nichi="02", race="01")]
RA_JRA_2025 = [rn.ra(race="01"), rn.ra(race="02", distance=2000)]
RA_FORM_2024 = [
    rn.ra(year="2024", mmdd="1230", venue="47", kai="12", nichi="03", race="10", kubun="A",
          distance=2000, **DIRT),
]  # fmt: skip
RA_FORM_2025 = [
    rn.ra(**NAR30, race="01", created="20250602", distance=1400),
    rn.ra(**NAR30, race="02", created="20250602", distance=1600),
    rn.ra(**NAR30, race="01", created="20250604", distance=1500),  # the later version wins
    rn.ra(mmdd="0710", venue="44", kai="05", nichi="02", race="11", kubun="A", distance=1800,
          track="23", turf="0", dirt="2", course=""),
    rn.ra(mmdd="0601", venue="42", kai="02", nichi="01", race="03", kubun="A", created="20250602",
          distance=1200, **DIRT),
    rn.ra(mmdd="0601", venue="42", kai="02", nichi="01", race="03", kubun="0", created="20250605",
          distance=1200, **DIRT),  # deleted
    rn.ra(mmdd="1005", venue="A4", kai="00", nichi="00", race="09", kubun="B", distance=2400,
          cond="000", weight="0", grade="", track="11", turf="1", dirt="0", course="",
          laps=()),  # unset codes: counted as unmapped only if (wrongly) merged into the report
    rn.ra(mmdd="0601", venue="00", kai="00", nichi="00", race="01", kubun="A"),  # no jurisdiction
    rn.ra(mmdd="1012", venue="B6", kai="00", nichi="00", race="00", kubun="B", distance=2000,
          track="11", turf="1", dirt="0", course=""),  # race number 00, as for some overseas races
]  # fmt: skip
FORM_RACES_EXPECTED = {  # race id -> the columns pinned
    "nar:2024-12-30:47:10": dict(jurisdiction="nar", meeting_id="nar:2024-12-30:47",
                                 race_date=date(2024, 12, 30), venue="47", race_no=10,
                                 season="2024/25", distance_m=2000, surface="dirt"),
    "nar:2025-06-01:30:01": dict(jurisdiction="nar", meeting_id="nar:2025-06-01:30",
                                 race_date=date(2025, 6, 1), venue="30", race_no=1,
                                 season="2025/26", distance_m=1500, surface="dirt"),
    "nar:2025-06-01:30:02": dict(jurisdiction="nar", meeting_id="nar:2025-06-01:30",
                                 race_date=date(2025, 6, 1), venue="30", race_no=2,
                                 season="2025/26", distance_m=1600, surface="dirt"),
    "nar:2025-07-10:44:11": dict(jurisdiction="nar", meeting_id="nar:2025-07-10:44",
                                 race_date=date(2025, 7, 10), venue="44", race_no=11,
                                 season="2025/26", distance_m=1800, surface="dirt"),
    "overseas:2025-10-05:A4:09": dict(jurisdiction="overseas", meeting_id="overseas:2025-10-05:A4",
                                      race_date=date(2025, 10, 5), venue="A4", race_no=9,
                                      season="2025/26", distance_m=2400, surface="turf"),
    "overseas:2025-10-12:B6:00": dict(jurisdiction="overseas", meeting_id="overseas:2025-10-12:B6",
                                      race_date=date(2025, 10, 12), venue="B6", race_no=0,
                                      season="2025/26", distance_m=2000, surface="turf"),
}  # fmt: skip


def test_build_races_writes_nar_and_overseas_races_to_form_races_only(data: Path) -> None:
    raw = data / "raw" / SOURCE
    write_raw(raw, "RA", 2024, RA_FORM_2024 + RA_JRA_2024)
    write_raw(raw, "RA", 2025, RA_FORM_2025[:3] + RA_JRA_2025 + RA_FORM_2025[3:])
    report = races.build_races()

    form = read(processed(data) / f"{FORM_RACES}.parquet")
    rn.assert_store_types(processed(data) / f"{FORM_RACES}.parquet", "races")
    ids = [r["race_id"] for r in form]
    assert ids == sorted(FORM_RACES_EXPECTED), ids  # every kept race, sorted by race_id
    for row in form:
        rn.assert_subset(row, FORM_RACES_EXPECTED[row["race_id"]], row["race_id"])
        assert row["race_id"].startswith(f"{row['jurisdiction']}:")

    # the JRA files hold JRA rows only, byte for byte those of a build without the other records
    jra = tables.read_table(SOURCE, "races")
    assert set(jra["jurisdiction"]) == {"jra"}
    assert set(tables.read_table(SOURCE, "meetings")["jurisdiction"]) == {"jra"}
    only_raw, only_out = data / "jra-only-raw", data / "jra-only"
    write_raw(only_raw, "RA", 2024, RA_JRA_2024)
    write_raw(only_raw, "RA", 2025, RA_JRA_2025)
    jra_report = races.build_races(raw=only_raw, out=only_out)
    for table in ("races", "meetings"):
        assert (processed(data) / f"{table}.parquet").read_bytes() == (
            only_out / f"{table}.parquet"
        ).read_bytes(), table
    assert report.unmapped == jra_report.unmapped, "a form race's codes reached the JRA report"
    year = report.by_year[2025]
    expected = {"races_written": 2, "non_jra_nar": 3, "non_jra_overseas": 2, "non_jra_unknown": 1,
                "form_without_id": 0}  # fmt: skip
    assert {k: year.get(k) for k in expected} == expected
    assert report.by_year[2024].get("form_without_id") == 0
    assert report.counts.get("form_without_id") == 0  # every form race got its id

    # `out` takes the form file too, and a rebuild gives the same bytes
    again = data / "again"
    races.build_races(out=again)
    assert (again / f"{FORM_RACES}.parquet").read_bytes() == (
        processed(data) / f"{FORM_RACES}.parquet"
    ).read_bytes()


# ------------------------------------------------------------------------------------------------
# The runners build: form_runners beside a JRA-only runners file

RA_FOR_SE = [
    rn.ra(race="01"),
    rn.ra(race="02"),
    rn.ra(**NAR30, race="01"),
    rn.ra(mmdd="0710", venue="44", kai="05", nichi="02", race="11", kubun="A", **DIRT),
    rn.ra(mmdd="1005", venue="A4", kai="00", nichi="00", race="09", kubun="B"),
    rn.ra(mmdd="0601", venue="00", kai="00", nichi="00", race="01", kubun="A"),
    rn.ra(mmdd="1102", venue="C0", kai="00", nichi="00", race="00", kubun="B"),  # race number 00
]
SE_C0 = dict(mmdd="1102", venue="C0", kai="00", nichi="00", race="00", kubun="B")
SE_NAR30 = dict(mmdd="0601", venue="30", kai="01", nichi="01", race="01", kubun="A")
SE_JRA = [
    rn.se(race="01", horse_no=1, horse_id="2021100001", place=1, odds=3.4),
    rn.se(race="02", horse_no=4, horse_id="2021100004", place=1, odds=2.0),
]
SE_FORM = [
    rn.se(**SE_NAR30, horse_no=1, horse_id="2021190001", created="20250530", place=0,
          time="0000"),  # the card version, superseded
    rn.se(**SE_NAR30, horse_no=1, horse_id="2021190001", created="20250602", place=1, odds=1.8),
    rn.se(**SE_NAR30, horse_no=2, horse_id="2021190002", place=2, odds=6.5, margin="2  "),
    rn.se(**SE_NAR30, horse_no=3, horse_id="2021190003", created="20250602", place=3),
    rn.se(**(SE_NAR30 | {"kubun": "0"}), horse_no=3, horse_id="2021190003", created="20250604",
          place=3),  # deleted
    rn.se(mmdd="0710", venue="44", kai="05", nichi="02", race="11", kubun="A", horse_no=5,
          horse_id="2021100001", place=3, odds=4.1),  # a JRA horse's NAR start
    rn.se(mmdd="1005", venue="A4", kai="00", nichi="00", race="09", kubun="B", horse_no=3,
          horse_id="2021100004", place=7, odds=0, fav=0),
    rn.se(mmdd="0601", venue="00", kai="00", nichi="00", race="01", kubun="A", horse_no=1,
          horse_id="2021199999", place=1),  # no jurisdiction: nowhere
    rn.se(**SE_C0, horse_no=2, horse_id="2021100001", place=5, odds=0, fav=0),  # a JRA horse's
    rn.se(**SE_C0, horse_no=6, horse_id="2021180006", place=1, odds=0, fav=0),  # start abroad
]  # fmt: skip
FORM_RUNNERS_EXPECTED = {  # (race id, horse_no) -> the columns pinned
    ("nar:2025-06-01:30:01", 1): dict(horse_id="2021190001", finish_pos=1, placing_raw="1"),
    ("nar:2025-06-01:30:01", 2): dict(horse_id="2021190002", finish_pos=2, placing_raw="2"),
    ("nar:2025-07-10:44:11", 5): dict(horse_id="2021100001", finish_pos=3, placing_raw="3"),
    ("overseas:2025-10-05:A4:09", 3): dict(horse_id="2021100004", finish_pos=7, placing_raw="7"),
    ("overseas:2025-11-02:C0:00", 2): dict(
        horse_id="2021100001", finish_pos=5, placing_raw="5", race_no=0
    ),
    ("overseas:2025-11-02:C0:00", 6): dict(
        horse_id="2021180006", finish_pos=1, placing_raw="1", race_no=0
    ),
}


def test_build_runners_writes_nar_and_overseas_runners_to_form_runners_only(data: Path) -> None:
    raw = data / "raw" / SOURCE
    write_raw(raw, "RA", 2025, RA_FOR_SE)
    write_raw(raw, "SE", 2025, [SE_FORM[0], SE_JRA[0], *SE_FORM[1:4], SE_JRA[1], *SE_FORM[4:]])
    races.build_races()
    report = runners.build_runners()

    path = processed(data) / f"{FORM_RUNNERS}.parquet"
    form = read(path)
    rn.assert_store_types(path, "runners")
    got_keys = [(r["race_id"], r["horse_no"]) for r in form]
    assert got_keys == sorted(FORM_RUNNERS_EXPECTED), got_keys  # sorted by race_id, horse_no
    form_races = {r["race_id"]: r for r in read(processed(data) / f"{FORM_RACES}.parquet")}
    for row in form:
        key = (row["race_id"], row["horse_no"])
        rn.assert_subset(row, FORM_RUNNERS_EXPECTED[key], key)
        race = form_races[row["race_id"]]  # every form runner's race is a form race
        for column in ("jurisdiction", "meeting_id", "race_date", "venue", "race_no", "season"):
            assert row[column] == race[column], (key, column)
    assert {r["jurisdiction"] for r in form} == {"nar", "overseas"}

    # the JRA runners (and races, after the winner's time) as a build without the other records
    assert set(tables.read_table(SOURCE, "runners")["jurisdiction"]) == {"jra"}
    only_raw, only_out = data / "jra-only-raw", data / "jra-only"
    write_raw(only_raw, "RA", 2025, RA_FOR_SE[:2])
    write_raw(only_raw, "SE", 2025, SE_JRA)
    races.build_races(raw=only_raw, out=only_out)
    jra_report = runners.build_runners(raw=only_raw, out=only_out)
    for table in ("runners", "races"):
        assert (processed(data) / f"{table}.parquet").read_bytes() == (
            only_out / f"{table}.parquet"
        ).read_bytes(), table
    assert report.unmapped == jra_report.unmapped, "a form runner's codes reached the JRA report"
    year = report.by_year[2025]
    expected = {
        "runners_written": 2, "non_jra_nar": 3, "non_jra_overseas": 3, "non_jra_unknown": 1,
        "form_without_id": 0,
    }  # fmt: skip
    assert {k: year.get(k) for k in expected} == expected
    assert report.counts.get("form_without_id") == 0  # every form runner got its race's id


# NAR and overseas records with no form id, beside a JRA record with a bad date (not form) and a
# venue with no jurisdiction (counted as non_jra_unknown, not here)
RA_NO_ID_2025 = [
    rn.ra(mmdd="0229", venue="30", kai="01", nichi="01", race="05", kubun="A", **DIRT),  # no day
    rn.ra(mmdd="1005", venue="A4", kai="00", nichi="00", race=" 1", kubun="B"),  # race number
    rn.ra(mmdd="0229", venue="05", race="03"),  # JRA
]
RA_NO_ID_2024 = [rn.ra(year="2024", mmdd="1301", venue="C0", kai="00", nichi="00", race="02",
                       kubun="B")]  # fmt: skip
SE_NO_ID = [
    rn.se(mmdd="0229", venue="30", kai="01", nichi="01", race="05", kubun="A", horse_no=1,
          horse_id="2021190005", place=1),
    rn.se(mmdd="1005", venue="A4", kai="00", nichi="00", race=" 1", kubun="B", horse_no=2,
          horse_id="2021180002", place=2, odds=0, fav=0),
    rn.se(mmdd="1005", venue="A4", kai="00", nichi="00", race=" 1", kubun="B", horse_no=4,
          horse_id="2021180004", place=1, odds=0, fav=0),
    rn.se(mmdd="0229", venue="05", race="01", horse_no=9, horse_id="2021100009", place=9),  # JRA
]  # fmt: skip


def test_reports_count_the_form_records_that_got_no_form_id(data: Path) -> None:
    """A kept NAR or overseas record with no form id reaches no form table; the reports count it
    per year and in total, so the loss is never silent. JRA and no-jurisdiction records are not
    form records and stay out of the count."""
    raw = data / "raw" / SOURCE
    write_raw(raw, "RA", 2024, RA_FORM_2024 + RA_NO_ID_2024 + RA_JRA_2024)
    write_raw(raw, "RA", 2025, RA_FOR_SE + RA_NO_ID_2025)
    write_raw(raw, "SE", 2025, SE_JRA + SE_NO_ID + SE_FORM)
    race_report = races.build_races()
    runner_report = runners.build_runners()

    assert {y: c.get("form_without_id") for y, c in race_report.by_year.items()} == {
        2024: 1, 2025: 2
    }  # fmt: skip
    assert race_report.counts.get("form_without_id") == 3
    assert runner_report.by_year[2025].get("form_without_id") == 3
    assert runner_report.counts.get("form_without_id") == 3

    # the records without an id are in no table, and the rest are as without them
    form_races = [r["race_id"] for r in read(processed(data) / f"{FORM_RACES}.parquet")]
    assert form_races == [
        "nar:2024-12-30:47:10", "nar:2025-06-01:30:01", "nar:2025-07-10:44:11",
        "overseas:2025-10-05:A4:09", "overseas:2025-11-02:C0:00",
    ], form_races  # fmt: skip
    form_runners = [(r["race_id"], r["horse_no"]) for r in
                    read(processed(data) / f"{FORM_RUNNERS}.parquet")]  # fmt: skip
    assert form_runners == sorted(FORM_RUNNERS_EXPECTED), form_runners


# ------------------------------------------------------------------------------------------------
# horse_runs: the form runs beside the JRA runs (store-level fixtures)

NAR_ONLY = "2015190001"  # runs at NAR only
HELD_FORM_ONLY = "2015190002"  # runs at NAR on a holdout day only
_LATE = [date(2026, 11, 28) + timedelta(days=k) for k in range(28)]
LATE_FORM_DAY = [d for d in _LATE if not is_holdout_day("japan", d)
                 and d not in hrt.LATE_DEV_DAYS][0]  # fmt: skip
HELD_FORM_DAY = hrt.HELD_DAYS[0]
# (jurisdiction, day, venue code, race no, horses in the field; None: no horse_id)
FORM_PLAN = (
    ("nar", date(2012, 8, 15), "30", 3, (hrt.HORSES[0], hrt.HORSES[1], NAR_ONLY)),
    ("overseas", date(2014, 6, 3), "B6", 4, (hrt.HORSES[6],)),
    ("overseas", date(2019, 10, 6), "A4", 9, (hrt.HORSES[3], None)),
    ("nar", date(2019, 11, 4), "44", 11, (hrt.HORSES[2], NAR_ONLY)),
    ("nar", LATE_FORM_DAY, "47", 5, (hrt.HORSES[4], hrt.HORSES[5])),
    ("nar", HELD_FORM_DAY, "42", 2, (HELD_FORM_ONLY, hrt.HORSES[7])),
)
FORM_SURFACES = ("dirt", "turf", "dirt", "jump", "dirt", "dirt")


def form_rows() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {FORM_RACES: [], FORM_RUNNERS: []}
    for i, (jur, day, code, race_no, field) in enumerate(FORM_PLAN):
        d = 40 + i  # spreads the made-up values away from the JRA fixture's
        rid = f"{jur}:{day.isoformat()}:{code}:{race_no:02d}"
        race = hrt.make_race(d, day, race_no, len(field)) | {
            "jurisdiction": jur, "race_id": rid, "meeting_id": rid.rsplit(":", 1)[0],
            "venue": code, "race_index": None, "surface": FORM_SURFACES[i],
            "distance_m": 1200 + 100 * i, "race_class": None if i % 2 else "C1",
        }  # fmt: skip
        out[FORM_RACES].append(race)
        for j, horse in enumerate(field):
            runner = hrt.make_runner(d, race, j, horse, len(field))
            assert runner["win_odds"] is not None  # a price in the file, never in the form
            out[FORM_RUNNERS].append(runner | {"jurisdiction": jur})
    return out


def write_form(data: Path, rows: dict[str, list[dict]]) -> None:
    folder = processed(data)
    folder.mkdir(parents=True, exist_ok=True)
    table = {FORM_RACES: "races", FORM_RUNNERS: "runners"}
    for name, table_rows in rows.items():
        pq.write_table(hrt.arrow_table(table[name], table_rows), folder / f"{name}.parquet")


def expected_form_rows(rows: dict[str, list[dict]]) -> list[dict]:
    by_race = {race["race_id"]: race for race in rows[FORM_RACES]}
    out = []
    for runner in rows[FORM_RUNNERS]:
        if runner["horse_id"] is None or runner["race_date"] in hrt.HELD_DAYS:
            continue
        race = by_race[runner["race_id"]]
        row = {c: runner[c] for c in hrt.FROM_RUNNER} | {c: race[c] for c in hrt.FROM_RACE}
        row["win_odds"] = None  # no JRA price for a form run
        out.append({c: row[c] for c in hrt.COLUMNS})
    return out


def test_horse_runs_hold_nar_and_overseas_runs_with_their_jurisdiction(store: Path) -> None:
    jra_rows = hrt.fixture_rows()
    hrt.write_store(store, jra_rows)
    rows = form_rows()
    write_form(store, rows)
    got = hrt.build(store)

    jra_want = hrt.expected_rows(jra_rows)
    form_want = expected_form_rows(rows)
    want = jra_want + form_want
    lines = [(r["horse_id"], r["race_date"]) for r in want]
    assert len(lines) == len(set(lines)), "the fixture runs a horse twice a day"
    want.sort(key=lambda r: (r["horse_id"], r["race_date"]))
    assert got.schema.equals(hrt.arrow_schema(TABLE), check_metadata=False), got.schema
    assert got.equals(hrt.arrow_table(TABLE, want)), "rows, values or sort order differ"

    got_rows = got.to_pylist()
    assert {r["jurisdiction"] for r in got_rows} == {"jra", "nar", "overseas"}
    form_got = [r for r in got_rows if r["jurisdiction"] != "jra"]
    assert len(form_got) == len(form_want) >= 8
    assert all(r["win_odds"] is None for r in form_got), "a form run carries a price"
    assert {r["race_id"].split(":")[0] for r in form_got} == {"nar", "overseas"}
    # the JRA runs are as before
    assert [r for r in got_rows if r["jurisdiction"] == "jra"] == hrt.arrow_table(
        TABLE, jra_want
    ).to_pylist()
    # a JRA horse's line now holds its NAR start between its JRA runs, a NAR-only horse has one
    line = [(r["race_date"], r["jurisdiction"]) for r in got_rows if r["horse_id"] == hrt.HORSES[2]]
    assert (date(2019, 11, 4), "nar") in line and "jra" in {j for _, j in line}, line
    assert line == sorted(line), line
    assert [r["race_date"] for r in got_rows if r["horse_id"] == NAR_ONLY] == [
        date(2012, 8, 15), date(2019, 11, 4)
    ]  # fmt: skip
    # holdout days stay out for form runs too; development days after the start are runs
    days = {r["race_date"] for r in form_got}
    assert HELD_FORM_DAY not in days and LATE_FORM_DAY in days
    assert HELD_FORM_ONLY not in {r["horse_id"] for r in got_rows}


def test_no_holdout_row_or_form_price_leaves_the_form_scans(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hrt.write_store(store, hrt.fixture_rows())
    rows = form_rows()
    write_form(store, rows)
    held_ids = {r["race_id"] for r in rows[FORM_RACES] if r["race_date"] in hrt.HELD_DAYS}
    assert held_ids
    recorder = hrt._Recorder()
    monkeypatch.setattr(v0_store, "ds", recorder)
    hr.build_horse_runs()
    assert hrt.output(store).is_file()

    names = {name for name, _ in recorder.scans}
    assert {FORM_RACES, FORM_RUNNERS} <= names, sorted(names)
    for name, table in recorder.scans:
        if name not in (FORM_RACES, FORM_RUNNERS):
            continue
        cols = set(table.column_names)
        if "race_date" in cols:
            assert not set(table.column("race_date").to_pylist()) & set(hrt.HELD_DAYS), name
        if "race_id" in cols:
            assert not set(table.column("race_id").to_pylist()) & held_ids, name
        if "horse_id" in cols:
            assert HELD_FORM_ONLY not in set(table.column("horse_id").to_pylist()), name
        assert "win_odds" not in cols, f"a {name} scan read the price"


# ------------------------------------------------------------------------------------------------
# End to end: the export's NAR and overseas starts reach horse_runs


def test_nar_and_overseas_starts_in_the_export_reach_horse_runs(data: Path) -> None:
    raw = data / "raw" / SOURCE
    write_raw(raw, "RA", 2025, RA_FOR_SE)
    write_raw(raw, "SE", 2025, SE_JRA + SE_FORM)
    races.build_races()
    runners.build_runners()
    hr.build_horse_runs()
    got = read(processed(data) / f"{TABLE}.parquet")

    lines: dict[str, list[tuple]] = {}
    for r in got:
        lines.setdefault(r["horse_id"], []).append(
            (r["race_date"], r["jurisdiction"], r["race_id"], r["venue"], r["distance_m"],
             r["surface"], r["placing_raw"], r["win_odds"])
        )  # fmt: skip
    assert lines == {
        "2021100001": [
            (date(2025, 6, 1), "jra", "jra:2025-06-01:TOK:01", "TOK", 1600, "turf", "1",
             pytest.approx(3.4)),
            (date(2025, 7, 10), "nar", "nar:2025-07-10:44:11", "44", 1600, "dirt", "3", None),
            (date(2025, 11, 2), "overseas", "overseas:2025-11-02:C0:00", "C0", 1600, "turf", "5",
             None),  # race number 00 abroad
        ],
        "2021100004": [
            (date(2025, 6, 1), "jra", "jra:2025-06-01:TOK:02", "TOK", 1600, "turf", "1",
             pytest.approx(2.0)),
            (date(2025, 10, 5), "overseas", "overseas:2025-10-05:A4:09", "A4", 1600, "turf", "7",
             None),
        ],
        "2021190001": [
            (date(2025, 6, 1), "nar", "nar:2025-06-01:30:01", "30", 1600, "dirt", "1", None),
        ],
        "2021190002": [
            (date(2025, 6, 1), "nar", "nar:2025-06-01:30:01", "30", 1600, "dirt", "2", None),
        ],
        "2021180006": [
            (date(2025, 11, 2), "overseas", "overseas:2025-11-02:C0:00", "C0", 1600, "turf", "1",
             None),
        ],
    }  # fmt: skip


# ------------------------------------------------------------------------------------------------
# No betting path reads the form tables


def test_only_the_jravan_package_names_the_form_tables() -> None:
    """The form tables are named in the jravan package (their writers and horse_runs) and nowhere
    else under src/: a betting or research reader that named them would read form-only rows."""
    naming = sorted(
        p.relative_to(SRC).as_posix()
        for p in SRC.rglob("*.py")
        if FORM_RACES in p.read_text(encoding="utf-8")
        or FORM_RUNNERS in p.read_text(encoding="utf-8")
    )
    assert naming, "no module names the form tables"
    outside = [p for p in naming if not (SRC / p).is_relative_to(JRAVAN_PACKAGE)]
    assert not outside, outside
