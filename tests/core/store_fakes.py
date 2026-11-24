"""Offline fakes for the store build tests (#48): a tiny raw cache, a manifest writer, an adapter.

Ported from HKTrack tests/unit/store/store_fakes.py without the real HKJC pages: every page here is
synthetic JSON written for these tests, and every name in it is made up.

* A page body is JSON {"tables": {table: [row, ...]}}; results pages also carry "served_key", the
  key of the page the site really served. `FakeStoreAdapter.parse` returns those tables (a cell
  "@fetched_utc" becomes the page's fetch time, which the parse worker takes from the manifest) plus
  two metadata tables: `_page` (key_mismatch: the served key is not the page's key) and `_worker`
  (the pid that parsed the page, and whether that process inherited the test's adapter registry).
* `add_page` writes one gzip raw file and one manifest row the way ingest does (HKTrack's DDL).
* `PAGES` is the raw cache most tests build from; `GOOD_PAGES` are the ones whose rows belong in the
  tables (the two pages under MISMATCHED_KEY hold the bytes of the KEY_2026 pages: the site answered
  a request for 2026-09-13 ST R1 with the 2026-09-16 HV R1 pages).
* `parse_job_in_fresh_worker` and `die_in_worker` are top-level so spawned workers import them by
  name (`store_fakes` is on sys.path, which spawn hands to its children).
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import types
from typing import Any, NamedTuple

from horsetrack.core import paths
from horsetrack.core.ingest import adapters
from horsetrack.core.store.parse_worker import parse_job as _real_parse_job

SOURCE = "testsrc"
FETCHED_UTC = "2026-09-21T04:07:12+00:00"
FETCHED_CELL = "@fetched_utc"
TEST_PID_VAR = "HORSETRACK_STORE_TEST_PID"

# parser version per page type ("telemetry" and anything else: no parser)
VERSIONS = {
    "fixture": "2", "results": "7", "sectional": "3", "corunning": "1", "horse": "5",
    "pedigree_card": "1", "racecard": "1",
}  # fmt: skip

RACE_2010 = f"{SOURCE}:2010-01-01:ST:01"
RACE_ABANDONED = f"{SOURCE}:2019-11-13:HV:01"
RACE_DEAD_HEAT = f"{SOURCE}:2020-12-09:HV:02"
RACE_2026 = f"{SOURCE}:2026-09-16:HV:01"
KEY_2010 = "2010-01-01_ST_R01"
KEY_2026 = "2026-09-16_HV_R01"
MISMATCHED_KEY = "2026-09-13_ST_R01"
HORSE_RECENT = "H_2023_J087"  # runs in RACE_2026, horse page fetched FETCHED_UTC
HORSE_RETIRED = "H_2008_K277"  # ran in RACE_2010, horse page fetched in +09:00
RETIRED_FETCHED = "2026-08-03T08:15:00+09:00"


class Page(NamedTuple):
    page_type: str
    key: str
    year: int
    tables: dict[str, list[dict[str, Any]]]
    status: str = "fetched"
    served_key: str | None = None
    fetched_utc: str = FETCHED_UTC

    @property
    def body(self) -> bytes:
        spec: dict[str, Any] = {"tables": self.tables}
        if self.served_key is not None:
            spec["served_key"] = self.served_key
        return json.dumps(spec, separators=(",", ":")).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.body).hexdigest()


def results_page(key: str, year: int, tables, *, status="fetched", served_key=None) -> Page:
    return Page("results", key, year, tables, status, served_key or key)


# ---------------------------------------------------------------- row builders


def meeting(race_date: str, venue: str, day_night: str | None, surface_mix: str | None) -> dict:
    return {
        "meeting_id": f"{SOURCE}:{race_date}:{venue}",
        "race_date": race_date,
        "venue": venue,
        "day_night": day_night,
        "surface_mix": surface_mix,
    }


def runner(race_id: str, horse_no: int, horse_id: str, finish_pos: int | None, **cells) -> dict:
    row = {
        "race_id": race_id,
        "horse_id": horse_id,
        "horse_no": horse_no,
        "horse_name": f"SYNTH {horse_id[-4:]}",
        "jockey_key": f"JK{horse_no:02d}",
        "trainer_key": f"TK{(horse_no * 7) % 10}",
        "actual_wt_lb": None,
        "declared_horse_wt_lb": None,
        "draw": None,
        "placing_raw": None if finish_pos is None else str(finish_pos),
        "finish_pos": finish_pos,
        "placing_code": None,
        "dead_heat": False,
        "lbw_raw": None,
        "lbw_lengths": None,
        "running_positions": [],
        "finish_time_s": None,
        "win_odds": None,
        "is_scratched": False,
    }
    row.update(cells)
    return row


def dividend(race_id: str, pool: str, combination: str, amount: float) -> dict:
    return {
        "race_id": race_id,
        "pool": pool,
        "combination": combination,
        "dividend": amount,
        "is_refund": False,
    }


def horse_run(horse_id: str, race_date: str, race_index: int, **cells) -> dict:
    row = {
        "horse_id": horse_id,
        "race_date": race_date,
        "race_index": race_index,
        "venue": "HV",
        "surface": "turf",
        "course_config": "A",
        "distance_m": 1200,
        "going": "GOOD",
        "race_class": "5",
        "draw": 3,
        "rating": None,
        "trainer_name": "T SYNTH",
        "jockey_name": "J SYNTH",
        "lbw_raw": None,
        "win_odds": None,
        "actual_wt_lb": 126,
        "running_positions": [],
        "finish_time_s": None,
        "declared_horse_wt_lb": 1050,
        "gear_raw": None,
        "placing_raw": None,
        "race_id": None,
    }
    row.update(cells)
    return row


def horse(horse_id: str, name: str, is_retired: bool, **cells) -> dict:
    row = {
        "horse_id": horse_id,
        "name": name,
        "origin": "NZ",
        "colour": "Bay",
        "sex": "Gelding",
        "import_type": "PPG",
        "sire": f"{name} SIRE",
        "dam": f"{name} DAM",
        "dam_sire": f"{name} DAMSIRE",
        "is_retired": is_retired,
        "profile_scraped_utc": FETCHED_CELL,
    }
    row.update(cells)
    return row


def sectional(race_id, horse_no, horse_id, section_no, pos, margin, time, subs) -> dict:
    return {
        "race_id": race_id,
        "horse_id": horse_id,
        "horse_no": horse_no,
        "section_no": section_no,
        "section_pos": pos,
        "section_margin_lengths": margin,
        "section_time_s": time,
        "section_margin_raw": None if margin is None else f"{margin:g}L",
        "sub_split_times_s": subs,
    }


# ---------------------------------------------------------------- the pages

FIXTURE_2010_01 = {
    "meetings": [
        meeting("2010-01-01", "ST", "day", "turf"),
        meeting("2010-01-06", "HV", "night", "turf"),
    ]
}
FIXTURE_2019_11 = {
    "meetings": [
        meeting("2019-11-13", "HV", "night", "turf"),
        meeting("2019-11-17", "ST", "day", "turf+awt"),
    ]
}
FIXTURE_2026_09 = {
    "meetings": [
        meeting("2026-09-13", "ST", "day", "turf"),
        meeting("2026-09-16", "HV", "night", "turf"),
        meeting("2026-09-23", "HV", "night", "turf"),
    ]
}
# the newer calendar page lists 2026-09-23 HV again, now as a day meeting
FIXTURE_2026_10 = {
    "meetings": [
        meeting("2026-09-23", "HV", "day", "turf"),
        meeting("2026-10-04", "ST", "day", "turf+awt"),
    ]
}

RESULTS_2010 = {
    "races": [
        {
            "race_id": RACE_2010, "race_index": 301, "race_name": "SYNTH HANDICAP",
            "race_class": "Class 4", "rating_hi": 60, "rating_lo": 40, "distance_m": 1200,
            "surface": "turf", "course_config": "A", "going": "GOOD", "prize": 500000,
            "post_time_local": "13:00", "field_size": 3, "finish_time_s": 70.25,
            "split_times_s": [24.1, 47.3, 70.25], "status": "run",
        }
    ],
    "runners": [
        runner(RACE_2010, 1, HORSE_RETIRED, 1, win_odds=3.5, draw=4, actual_wt_lb=126,
               declared_horse_wt_lb=1050, running_positions=[2, 1, 1], finish_time_s=70.25,
               lbw_raw="-", lbw_lengths=0.0),
        runner(RACE_2010, 2, "H_2007_A001", 2, win_odds=5.0, draw=1, actual_wt_lb=122,
               declared_horse_wt_lb=1100, running_positions=[1, 2, 2], finish_time_s=70.41,
               lbw_raw="1", lbw_lengths=1.0),
        runner(RACE_2010, 3, "H_2007_A002", 3, win_odds=9.0, draw=2, actual_wt_lb=118,
               declared_horse_wt_lb=980, running_positions=[3, 3, 3], finish_time_s=70.9,
               lbw_raw="3-1/2", lbw_lengths=3.5),
    ],
    "dividends": [
        dividend(RACE_2010, "WIN", "1", 35.0), dividend(RACE_2010, "PLA", "1", 15.0),
        dividend(RACE_2010, "PLA", "2", 20.0), dividend(RACE_2010, "PLA", "3", 18.5),
    ],
    "incidents": [
        {"race_id": RACE_2010, "horse_id": "H_2007_A002", "horse_no": 3, "text": "Slow to begin."}
    ],
}  # fmt: skip

RESULTS_ABANDONED = {
    "races": [
        {
            "race_id": RACE_ABANDONED, "race_index": 120, "race_name": "SYNTH PLATE",
            "race_class": "Class 5", "distance_m": 1650, "surface": "turf", "course_config": "C",
            "going": "YIELDING", "prize": 400000, "post_time_local": "19:15", "field_size": None,
            "finish_time_s": None, "split_times_s": [], "status": "abandoned",
        }
    ],
    "runners": [
        runner(RACE_ABANDONED, 1, "H_2018_C001", None, draw=3, actual_wt_lb=130),
        runner(RACE_ABANDONED, 2, "H_2018_C002", None, draw=1, actual_wt_lb=124),
    ],
    "dividends": [],
    "incidents": [],
}  # fmt: skip

RESULTS_DEAD_HEAT = {
    "races": [
        {
            "race_id": RACE_DEAD_HEAT, "race_index": 222, "race_name": "SYNTH CUP",
            "race_class": "Class 3", "distance_m": 1650, "surface": "turf", "course_config": "B",
            "going": "GOOD TO FIRM", "prize": 1000000, "post_time_local": "19:45",
            "field_size": 3, "finish_time_s": 99.12, "split_times_s": [25.0, 49.8, 74.1, 99.12],
            "status": "run",
        }
    ],
    "runners": [
        runner(RACE_DEAD_HEAT, 1, "H_2016_D001", 3, win_odds=4.0, draw=6, actual_wt_lb=128,
               declared_horse_wt_lb=1120, running_positions=[3, 3, 3, 3], finish_time_s=99.5,
               lbw_raw="1-1/4", lbw_lengths=1.25),
        runner(RACE_DEAD_HEAT, 2, "H_2016_D002", None, draw=2, actual_wt_lb=125,
               placing_raw="WV", placing_code="WV", is_scratched=True),
        runner(RACE_DEAD_HEAT, 3, "H_2016_D003", 1, win_odds=6.0, draw=1, actual_wt_lb=121,
               declared_horse_wt_lb=1000, running_positions=[1, 1, 2, 1], finish_time_s=99.12,
               lbw_raw="-", lbw_lengths=0.0, placing_raw="1 DH", placing_code="DH",
               dead_heat=True),
        runner(RACE_DEAD_HEAT, 5, "H_2016_D005", 1, win_odds=2.5, draw=4, actual_wt_lb=133,
               declared_horse_wt_lb=1150, running_positions=[2, 2, 1, 1], finish_time_s=99.12,
               lbw_raw="-", lbw_lengths=0.0, placing_raw="1 DH", placing_code="DH",
               dead_heat=True),
    ],
    "dividends": [
        dividend(RACE_DEAD_HEAT, "WIN", "3", 31.0), dividend(RACE_DEAD_HEAT, "WIN", "5", 13.5),
        dividend(RACE_DEAD_HEAT, "PLA", "3", 14.0), dividend(RACE_DEAD_HEAT, "PLA", "5", 11.0),
        dividend(RACE_DEAD_HEAT, "PLA", "1", 12.5),
    ],
    "incidents": [
        {"race_id": RACE_DEAD_HEAT, "horse_id": "H_2016_D002", "horse_no": 2,
         "text": "Withdrawn by the vets."}
    ],
}  # fmt: skip

# rows in page order, which is not the tables' sort order
RESULTS_2026 = {
    "races": [
        {
            "race_id": RACE_2026, "race_index": 29, "race_name": "SYNTH SPRINT",
            "race_class": "Class 5", "rating_hi": 40, "rating_lo": 0, "distance_m": 1200,
            "surface": "turf", "course_config": "A+3", "going": "GOOD", "prize": 875000,
            "post_time_local": "19:10", "field_size": 4, "finish_time_s": 69.8,
            "split_times_s": [23.9, 46.5, 69.8], "status": "run",
        }
    ],
    "runners": [
        runner(RACE_2026, 4, HORSE_RECENT, 1, win_odds=7.5, draw=3, actual_wt_lb=133,
               declared_horse_wt_lb=1101, running_positions=[1, 1, 1], finish_time_s=69.8,
               lbw_raw="-", lbw_lengths=0.0),
        runner(RACE_2026, 2, "H_2023_J308", 2, win_odds=3.2, draw=9, actual_wt_lb=128,
               declared_horse_wt_lb=1088, running_positions=[3, 2, 2], finish_time_s=69.95,
               lbw_raw="1", lbw_lengths=1.0),
        runner(RACE_2026, 7, "H_2023_J504", 3, win_odds=12.0, draw=1, actual_wt_lb=120,
               declared_horse_wt_lb=1043, running_positions=[2, 3, 3], finish_time_s=70.1,
               lbw_raw="2", lbw_lengths=2.0),
        runner(RACE_2026, 1, "H_2024_B111", 4, win_odds=25.0, draw=5, actual_wt_lb=135,
               declared_horse_wt_lb=1150, running_positions=[4, 4, 4], finish_time_s=70.6,
               lbw_raw="5-1/4", lbw_lengths=5.25),
        runner(RACE_2026, 3, "H_2024_B222", None, actual_wt_lb=126, declared_horse_wt_lb=1010,
               placing_raw="WX", placing_code="WX", is_scratched=True),
    ],
    "dividends": [
        dividend(RACE_2026, "WIN", "4", 75.0), dividend(RACE_2026, "PLA", "4", 25.0),
        dividend(RACE_2026, "PLA", "2", 15.5), dividend(RACE_2026, "PLA", "7", 40.0),
        dividend(RACE_2026, "QIN", "2,4", 120.0), dividend(RACE_2026, "TCE", "4,2,7", 2100.0),
    ],
    "incidents": [
        {"race_id": RACE_2026, "horse_id": "H_2023_J504", "horse_no": 7,
         "text": "Hung out on the turn."},
        {"race_id": RACE_2026, "horse_id": "H_2023_J308", "horse_no": 2, "text": "Lost a plate."},
    ],
}  # fmt: skip

SECTIONAL_2026 = {
    "sectionals": [
        sectional(RACE_2026, 4, HORSE_RECENT, 1, 1, 0.0, 23.9, [11.8, 12.1]),
        sectional(RACE_2026, 4, HORSE_RECENT, 2, 1, 0.5, 22.6, [11.2, 11.4]),
        sectional(RACE_2026, 4, HORSE_RECENT, 3, 1, 0.75, 23.3, [11.5, 11.8]),
        sectional(RACE_2026, 2, "H_2023_J308", 1, 3, 1.5, 24.2, [12.0, 12.2]),
        sectional(RACE_2026, 2, "H_2023_J308", 2, 2, 1.0, 22.5, [11.1, 11.4]),
        sectional(RACE_2026, 2, "H_2023_J308", 3, 2, 1.0, 23.25, [11.6, 11.65]),
        sectional(RACE_2026, 7, "H_2023_J504", 1, 2, 0.25, 23.95, [11.9, 12.05]),
        sectional(RACE_2026, 7, "H_2023_J504", 2, 3, 1.25, 22.8, [11.3, 11.5]),
        sectional(RACE_2026, 7, "H_2023_J504", 3, 3, 2.0, None, []),
    ]
}

CORUNNING_2010 = {
    "running_comments": [
        {"race_id": RACE_2010, "horse_id": "H_2007_A002", "horse_no": 3, "gear_raw": "B",
         "comment": "Slow away, ran on."},
        {"race_id": RACE_2010, "horse_id": HORSE_RETIRED, "horse_no": 1, "gear_raw": "B/TT",
         "comment": "Led throughout."},
        {"race_id": RACE_2010, "horse_id": "H_2007_A001", "horse_no": 2, "gear_raw": None,
         "comment": "Chased the leader."},
    ]
}  # fmt: skip

HORSE_RECENT_PAGE = {
    "horses": [horse(HORSE_RECENT, "SYNTH SPARK", False)],
    "horse_runs": [
        horse_run(HORSE_RECENT, "2026-09-16", 29, rating=31, gear_raw="H/TT", placing_raw="1",
                  win_odds=7.5, actual_wt_lb=133, running_positions=[1, 1, 1],
                  finish_time_s=69.8, race_id=RACE_2026),
        horse_run(HORSE_RECENT, "2026-06-01", 601, venue="ST", rating=28, gear_raw="H",
                  placing_raw="4", race_id=f"{SOURCE}:2026-06-01:ST:05"),
        horse_run(HORSE_RECENT, "2026-05-10", 550, rating=27, gear_raw="H", placing_raw="WV"),
    ],
}  # fmt: skip

HORSE_RETIRED_PAGE = {
    "horses": [
        horse(HORSE_RETIRED, "SYNTH OLD TIMER", True, origin="IRE", colour="Grey",
              import_type="ISG")
    ],
    "horse_runs": [
        horse_run(HORSE_RETIRED, "2010-01-01", 301, race_class="4", rating=54, gear_raw="B",
                  placing_raw="1", win_odds=3.5, race_id=RACE_2010),
        horse_run(HORSE_RETIRED, "2009-12-20", 290, race_class="4", rating=52,
                  placing_raw="WX-A"),
    ],
}  # fmt: skip

PAGES: list[Page] = [
    Page("fixture", "2010-01", 2010, FIXTURE_2010_01),
    Page("fixture", "2019-11", 2019, FIXTURE_2019_11),
    Page("fixture", "2026-09", 2026, FIXTURE_2026_09),
    Page("fixture", "2026-10", 2026, FIXTURE_2026_10),
    results_page(KEY_2010, 2010, RESULTS_2010),
    results_page("2019-11-13_HV_R01", 2019, RESULTS_ABANDONED, status="abandoned"),
    results_page("2020-12-09_HV_R02", 2020, RESULTS_DEAD_HEAT),
    results_page(KEY_2026, 2026, RESULTS_2026),
    Page("sectional", KEY_2026, 2026, SECTIONAL_2026),
    Page("corunning", KEY_2010, 2010, CORUNNING_2010),
    Page("horse", HORSE_RECENT, 2023, HORSE_RECENT_PAGE),
    Page("horse", HORSE_RETIRED, 2008, HORSE_RETIRED_PAGE, fetched_utc=RETIRED_FETCHED),
    # the same bytes as the KEY_2026 pages, under another key
    results_page(MISMATCHED_KEY, 2026, RESULTS_2026, served_key=KEY_2026),
    Page("sectional", MISMATCHED_KEY, 2026, SECTIONAL_2026),
]
GOOD_PAGES: list[Page] = [page for page in PAGES if page.key != MISMATCHED_KEY]


def page(page_type: str, key: str) -> Page:
    (found,) = [p for p in PAGES if (p.page_type, p.key) == (page_type, key)]
    return found


# ---------------------------------------------------------------- raw cache and manifest

# the ingest manifest as HKTrack defines it (horsetrack.core.ingest.queue keeps the same table)
MANIFEST_SCHEMA = """
CREATE TABLE IF NOT EXISTS fetch_queue (
    jurisdiction TEXT NOT NULL, page_type TEXT NOT NULL, key TEXT NOT NULL, url TEXT NOT NULL,
    year INTEGER NOT NULL, priority INTEGER NOT NULL DEFAULT 0, method TEXT NOT NULL DEFAULT 'GET',
    body TEXT, mutable INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'pending',
    http_status INTEGER, final_url TEXT, sha256 TEXT, bytes INTEGER, fetched_utc TEXT,
    attempts INTEGER NOT NULL DEFAULT 0, error TEXT, parser_version TEXT,
    PRIMARY KEY (page_type, key)
)
"""


def add_page(
    page_type: str,
    key: str,
    year: int,
    body: bytes,
    *,
    status: str = "fetched",
    fetched_utc: str = FETCHED_UTC,
    raw: bytes | None = None,
    mtime: int = 0,
) -> None:
    """One raw file + one manifest row under the current HORSETRACK_DATA_DIR (`raw` = the file's
    bytes as they are, e.g. a broken gzip stream; `mtime` goes into the gzip header)."""
    path = paths.raw_page_path(SOURCE, page_type, year, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw if raw is not None else gzip.compress(body, mtime=mtime))
    url = f"https://racing.example.test/{page_type}?key={key}"
    manifest = paths.manifest_path(SOURCE)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(manifest)
    with conn:
        conn.execute(MANIFEST_SCHEMA)
        conn.execute(
            "INSERT OR REPLACE INTO fetch_queue (jurisdiction, page_type, key, url, year, status,"
            " final_url, sha256, bytes, fetched_utc, http_status, attempts)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 200, 1)",
            (
                SOURCE, page_type, key, url, year, status, url,
                hashlib.sha256(body).hexdigest(), len(body), fetched_utc,
            ),
        )  # fmt: skip
    conn.close()


def add(page: Page, *, mtime: int = 0) -> None:
    add_page(
        page.page_type,
        page.key,
        page.year,
        page.body,
        status=page.status,
        fetched_utc=page.fetched_utc,
        mtime=mtime,
    )


def populate(pages: list[Page], *, mtime: int = 0) -> None:
    for item in pages:
        add(item, mtime=mtime)


# ---------------------------------------------------------------- the fake adapter

WORKER_STATE: dict[str, Any] = {}  # per process: whether it inherited the test's registry


def parse_body(page) -> dict[str, list[dict[str, Any]]]:
    spec = json.loads(page.body.decode("utf-8"))
    tables = {
        name: [
            {c: (page.fetched_utc if v == FETCHED_CELL else v) for c, v in row.items()}
            for row in rows
        ]
        for name, rows in spec["tables"].items()
    }
    served = spec.get("served_key")
    tables["_page"] = [{"key_mismatch": served is not None and served != page.key}]
    return tables


class FakeStoreAdapter:
    """Parses the synthetic pages; records every parse; lets a test replace a parse
    (`overrides[(page_type, key)]` = a ParsedPage, an exception to raise, or a callable taking the
    RawPage), pin parser versions (`versions`), interrupt the n-th parse (`interrupt_at`), and set
    the season's start month (`season_start_month()`, absent after `set_start_month(None)`)."""

    def __init__(self, source: str = SOURCE, start_month: int | None = 9) -> None:
        self.source = source
        self.versions: dict[str, str | None] = dict(VERSIONS)
        self.calls: list[tuple[str, str]] = []
        self.overrides: dict[tuple[str, str], Any] = {}
        self.interrupt_at: int | None = None
        self.set_start_month(start_month)

    def set_start_month(self, month: int | None) -> None:
        if month is None:
            self.__dict__.pop("season_start_month", None)
        else:
            self.season_start_month = lambda: month

    def fixture_tasks(self, start, end):
        return []

    def meeting_tasks(self, fixture_pages):
        return []

    def followup_tasks(self, parsed):
        return []

    def classify(self, page_type, body):
        return "fetched"

    def parser_version(self, page_type: str) -> str | None:
        version = self.versions.get(page_type)
        return None if version is None else str(version)

    def parse(self, page):
        self.calls.append((page.page_type, page.key))
        if self.interrupt_at is not None and len(self.calls) == self.interrupt_at:
            raise KeyboardInterrupt  # not an Exception: goes straight through parse_job
        override = self.overrides.get((page.page_type, page.key))
        if isinstance(override, BaseException):
            raise override
        if callable(override):
            parsed = override(page)
        elif override is not None:
            parsed = copy.deepcopy(override)
        else:
            parsed = parse_body(page)
        worker = {"pid": os.getpid(), "registry_inherited": WORKER_STATE.get("registry_inherited")}
        return {**parsed, "_worker": [worker]}


ADAPTER = FakeStoreAdapter()  # what a spawned worker parses with (see parse_job_in_fresh_worker)


def register(monkeypatch, adapter: FakeStoreAdapter) -> FakeStoreAdapter:
    """Put `adapter` in the registry under its source, backed by a module in sys.modules."""
    package = "_horsetrack_test_store_adapters"
    module_name = f"{package}.{adapter.source}"
    module = types.ModuleType(module_name)
    module.ADAPTER = adapter
    monkeypatch.setitem(sys.modules, package, types.ModuleType(package))
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(adapters.ADAPTERS, adapter.source, module_name)
    return adapter


class RanInTestProcess(Exception):
    """A worker-only function ran in the test process: the build did not use worker processes."""


def parse_job_in_fresh_worker(job):
    """Stands in for parse_worker.parse_job in the process pool: a spawned worker starts without the
    test's registry, so this records that, registers this module's ADAPTER and parses as usual."""
    if "registry_inherited" not in WORKER_STATE:
        WORKER_STATE["registry_inherited"] = SOURCE in adapters.ADAPTERS
    adapters.ADAPTERS.setdefault(SOURCE, __name__)
    return _real_parse_job(job)


def die_in_worker(job):
    """A parser process that dies (a segfault in the HTML parser, an OOM kill)."""
    if str(os.getpid()) == os.environ.get(TEST_PID_VAR):
        raise RanInTestProcess("die_in_worker ran in the test process")
    os._exit(9)
