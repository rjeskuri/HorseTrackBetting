"""HKJC results page -> races, runners, dividends, incidents (ticket #65, ported from HKTrack's
`sources/hkjc/parse_results.py`), on the live and the archive routes.

Contract pinned here (the coder follows it):
- `horsetrack.countries.hongkong.hkjc.parse_results.parse(page: RawPage) -> ParsedPage`
  (`horsetrack.core.ingest.types`), pure, no network: tables `races`, `runners`, `dividends`,
  `incidents` with HKTrack's columns exactly (below), plus `_meta`, whose row flags
  `key_mismatch` when the cache key names another race than the page shows.
- Ids: meeting `hkjc:<YYYY-MM-DD>:<VENUE>`, race `<meeting>:<NN>`; horses by the profile link's
  horseid (plain charset only), jockeys and trainers by their link codes; `jockey_key` /
  `trainer_key` are `<role>:<name>`, casefolded, without the apprentice claim "(-2)" and with
  punctuation as spaces (retired people print without a link, so the name is the join key).
- The race is the one the page shows: its meeting label and race header (or the unlinked race
  tab) win; the cache key, then the URL (live `localresults?...&Racecourse=` or archive
  `archive/localresults?...&racecourse=`), fill the gaps. No race at all: empty tables.
- Columns are read by header text, header rows by label, dividend pools by their cell count;
  a malformed cell becomes null next to its raw text, and a bad row never stops the page.
- A race without a result (the banner of an abandoned meeting, the 2024+ banner with only the
  refunds, a void race) is one `races` row with status "abandoned".
- `parse_results.classify(body: bytes) -> str` is HKTrack's adapter rule for a results page,
  decided by content and never by HTTP status (inline <style>, <script> and HTML comments are
  not content): STATUS_EMPTY for the "No information." page the site serves with HTTP 200,
  STATUS_ABANDONED for a race without a result, STATUS_FETCHED for a result, STATUS_BLOCKED for
  anything else.

Pages are synthetic (hkjc_results_pages.py); every expected value is written out here by hand.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from functools import cache

import hkjc_results_pages as pages
import pytest
from pytest import approx

from horsetrack.core.ingest.types import (
    STATUS_ABANDONED,
    STATUS_BLOCKED,
    STATUS_EMPTY,
    STATUS_FETCHED,
    RawPage,
)
from horsetrack.core.store import schemas

# isort: split
# (its own block: ruff ranks a module that does not exist yet as third-party)
from horsetrack.countries.hongkong.hkjc import parse_results

TABLES = ("races", "runners", "dividends", "incidents")
RACE_COLUMNS = {
    "jurisdiction", "race_id", "meeting_id", "race_no", "race_index", "race_name", "race_class",
    "rating_hi", "rating_lo", "distance_m", "surface", "course_config", "going", "prize",
    "post_time_local", "field_size", "finish_time_s", "split_times_s", "status",
}  # fmt: skip
RUNNER_COLUMNS = {
    "jurisdiction", "race_id", "horse_id", "horse_no", "horse_name", "brand_raw", "jockey_key",
    "jockey_code", "jockey_name", "trainer_key", "trainer_code", "trainer_name", "actual_wt_lb",
    "declared_horse_wt_lb", "draw", "placing_raw", "finish_pos", "placing_code", "dead_heat",
    "lbw_raw", "lbw_lengths", "running_positions", "finish_time_s", "win_odds", "is_scratched",
}  # fmt: skip
DIVIDEND_COLUMNS = {"jurisdiction", "race_id", "pool", "combination", "dividend", "is_refund"}
INCIDENT_COLUMNS = {"jurisdiction", "race_id", "horse_id", "horse_no", "text"}


def raw(race: pages.Race, key: str | None = None, url: str | None = None) -> RawPage:
    """A results page as the fetcher hands it over, under its cache key."""
    if url is None:
        route = pages.live_url if race.race_date >= date(2026, 9, 1) else pages.archive_url
        url = route(race.race_date, race.venue, race.race_no)
    return RawPage("hkjc", "results", race.key if key is None else key, url, pages.render(race))


@cache
def parse(race: pages.Race) -> dict:
    return parse_results.parse(raw(race))


def by_no(parsed: dict) -> dict[int | None, dict]:
    return {r["horse_no"]: r for r in parsed["runners"]}


def dividend_rows(parsed: dict) -> list[tuple]:
    return [
        (d["pool"], d["combination"], d["dividend"], d["is_refund"]) for d in parsed["dividends"]
    ]


def race_row(day: str, venue: str, race_no: int, **fields) -> dict:
    meeting = f"hkjc:{day}:{venue}"
    return {
        "jurisdiction": "hkjc",
        "race_id": f"{meeting}:{race_no:02d}",
        "meeting_id": meeting,
        "race_no": race_no,
        "post_time_local": None,  # not printed on the results page
        **fields,
    }


def race_id(race: pages.Race) -> str:
    return f"hkjc:{race.race_date.isoformat()}:{race.venue}:{race.race_no:02d}"


# ---------------------------------------------------------------- race header

HEADERS = [
    pytest.param(pages.LIVE, race_row(
        "2026-10-14", "HV", 3, race_index=141, race_name="LANTERN BAY HANDICAP",
        race_class="Class 4", rating_hi=60, rating_lo=40, distance_m=1650, surface="turf",
        course_config="C+3", going="GOOD TO FIRM", prize=1_250_000, field_size=7,
        finish_time_s=approx(98.42), split_times_s=approx([25.31, 48.90, 72.66, 98.42]),
        status="run"), id="live"),
    pytest.param(pages.ARCHIVE_2010, race_row(
        "2011-01-30", "ST", 1, race_index=312, race_name="PEBBLE COVE HANDICAP",
        race_class="Class 3", rating_hi=80, rating_lo=60, distance_m=1200, surface="turf",
        course_config="B+2", going="GOOD", prize=900_000, field_size=7,
        finish_time_s=approx(69.94), split_times_s=approx([23.85, 46.71, 69.94]),
        status="run"), id="archive_2010"),
    pytest.param(pages.ARCHIVE_2004, race_row(
        "2005-03-06", "ST", 1, race_index=458, race_name="FERN GULLY PLATE",
        race_class="Griffin Race", rating_hi=None, rating_lo=None, distance_m=1200,
        surface="turf", course_config="C", going="GOOD TO FIRM", prize=400_000, field_size=5,
        finish_time_s=approx(70.48), split_times_s=approx([24.10, 47.02, 70.48]),
        status="run"), id="archive_2004_griffin"),
    pytest.param(pages.AWT, race_row(
        "2023-12-16", "ST", 6, race_index=377, race_name="MOSSY BRIDGE HANDICAP",
        race_class="Class 4 (Special Condition)", rating_hi=60, rating_lo=35, distance_m=1200,
        surface="awt", course_config=None, going="GOOD", prize=1_170_000, field_size=7,
        finish_time_s=approx(69.70), split_times_s=approx([23.90, 46.80, 69.70]),
        status="run"), id="all_weather"),
    pytest.param(pages.GROUP, race_row(
        "2025-12-14", "ST", 8, race_index=255, race_name="WINTER PEARL CUP",
        race_class="Group One", rating_hi=None, rating_lo=None, distance_m=2400,
        surface="turf", course_config="A+2", going="YIELDING", prize=30_000_000, field_size=7,
        finish_time_s=approx(147.35),
        split_times_s=approx([25.40, 50.10, 74.32, 98.75, 122.61, 147.35]),
        status="run"), id="group_race"),
    pytest.param(pages.OPEN_BAND, race_row(
        "2026-11-01", "ST", 9, race_index=180, race_name="AUTUMN REED HANDICAP",
        race_class="Class 1", rating_hi=None, rating_lo=95, distance_m=1400, surface="turf",
        course_config="B", going="GOOD TO YIELDING", prize=3_450_000, field_size=7,
        finish_time_s=approx(82.09), split_times_s=approx([13.30, 35.02, 58.61, 82.09]),
        status="run"), id="open_rating_band"),
]  # fmt: skip


@pytest.mark.parametrize(("race", "expected"), HEADERS)
def test_race_header(race, expected):
    """Header rows are found by their labels (a spacer row and the "Sectional Time :" row are
    not the race's times); the race time is the last split; the field leaves out withdrawals."""
    assert set(expected) == RACE_COLUMNS
    assert RACE_COLUMNS <= set(schemas.TABLES["races"].columns)
    assert parse(race)["races"] == [expected]


# ---------------------------------------------------------------- runners

# (horse_no, placing_raw, finish_pos, placing_code, lbw_lengths, running_positions,
#  finish_time_s, win_odds, actual_wt_lb, declared_horse_wt_lb, draw, is_scratched)
RESULTS = {
    "live": [
        (6, "1", 1, None, 0.0, [3, 2, 1, 1], approx(98.42), 3.7, 126, 1105, 4, False),
        (2, "2", 2, None, 0.3, [6, 6, 4, 2], approx(98.46), 12.0, 133, 1180, 7, False),
        (9, "3", 3, None, 0.75, [1, 1, 2, 3], approx(98.54), 25.0, 120, 1042, 1, False),
        (11, "4", 4, None, 1.25, [7, 7, 7, 4], approx(98.62), 8.9, 118, 998, 10, False),
        (4, "5", 5, None, 2.5, [2, 3, 3, 5], approx(98.81), 45.0, 129, 1213, 3, False),
        (1, "6", 6, None, 4.5, [5, 5, 6, 6], approx(99.14), 99.0, 135, 1260, 8, False),
        (12, "7", 7, None, 10.0, [4, 4, 5, 7], approx(100.04), 61.0, 115, 1020, 2, False),
        (7, "WV-A", None, "WV-A", None, [], None, None, 118, None, None, True),
    ],
    "archive_2010": [
        (3, "1", 1, None, 0.0, [4, 2, 1], approx(69.94), 2.4, 127, 1079, 9, False),
        (8, "2", 2, None, 0.5, [1, 1, 2], approx(70.02), 6.1, 121, 1032, 2, False),
        (1, "3", 3, None, 1.25, [6, 5, 3], approx(70.14), 15.0, 133, 1150, 5, False),
        (6, "4", 4, None, 1.5, [3, 4, 4], approx(70.18), 9.9, 118, 1001, 12, False),
        (2, "5", 5, None, 2.75, [7, 7, 5], approx(70.38), 33.0, 126, 1110, 7, False),
        (7, "6", 6, None, 5.0, [2, 3, 6], approx(70.74), 41.0, 115, 987, 1, False),
        (4, "7", 7, None, 8.25, [5, 6, 7], approx(71.26), 72.0, 113, 1066, 3, False),
    ],
    "archive_2004": [  # no Running Position column
        (3, "1", 1, None, 0.0, [], approx(70.48), 4.8, 122, 1088, 2, False),
        (5, "2", 2, None, 0.75, [], approx(70.60), 7.2, 126, 1121, 6, False),
        (1, "3", 3, None, 1.75, [], approx(70.76), 11.0, 130, 1170, 4, False),
        (4, "4", 4, None, 2.0, [], approx(70.80), 19.0, 119, 1012, 1, False),
        (2, "5", 5, None, 6.0, [], approx(71.44), 35.0, 117, 1095, 3, False),
    ],
}
RESULT_FIELDS = (
    "horse_no", "placing_raw", "finish_pos", "placing_code", "lbw_lengths", "running_positions",
    "finish_time_s", "win_odds", "actual_wt_lb", "declared_horse_wt_lb", "draw", "is_scratched",
)  # fmt: skip

WINNERS = {
    "live": {
        "jurisdiction": "hkjc", "race_id": "hkjc:2026-10-14:HV:03",
        "horse_id": "HK_2023_X101", "horse_no": 6, "horse_name": "AMBER LANTERN",
        "brand_raw": "X101",
        "jockey_key": "jockey:a quillfeather", "jockey_code": "QFA",
        "jockey_name": "A Quillfeather (-2)",
        "trainer_key": "trainer:m q larkspur", "trainer_code": "LMQ",
        "trainer_name": "M Q Larkspur",
        "actual_wt_lb": 126, "declared_horse_wt_lb": 1105, "draw": 4,
        "placing_raw": "1", "finish_pos": 1, "placing_code": None, "dead_heat": False,
        "lbw_raw": "---", "lbw_lengths": 0.0, "running_positions": [3, 2, 1, 1],
        "finish_time_s": approx(98.42), "win_odds": 3.7, "is_scratched": False,
    },
    "archive_2010": {  # the brand printed differs from the id; retired people have no link
        "jurisdiction": "hkjc", "race_id": "hkjc:2011-01-30:ST:01",
        "horse_id": "HK_2009_Q101", "horse_no": 3, "horse_name": "SPARROW KING",
        "brand_raw": "BQ101",
        "jockey_key": "jockey:n gallowmere", "jockey_code": None,
        "jockey_name": "N Gallowmere",
        "trainer_key": "trainer:p o ashgrove", "trainer_code": None,
        "trainer_name": "P O'Ashgrove",
        "actual_wt_lb": 127, "declared_horse_wt_lb": 1079, "draw": 9,
        "placing_raw": "1", "finish_pos": 1, "placing_code": None, "dead_heat": False,
        "lbw_raw": "---", "lbw_lengths": 0.0, "running_positions": [4, 2, 1],
        "finish_time_s": approx(69.94), "win_odds": 2.4, "is_scratched": False,
    },
    "archive_2004": {
        "jurisdiction": "hkjc", "race_id": "hkjc:2005-03-06:ST:01",
        "horse_id": "HK_2002_R086", "horse_no": 3, "horse_name": "PEBBLE PRINCE",
        "brand_raw": "CR086",
        "jockey_key": "jockey:b hollowell", "jockey_code": None,
        "jockey_name": "B Hollowell",
        "trainer_key": "trainer:w y gantry", "trainer_code": "GWY",
        "trainer_name": "W Y Gantry",
        "actual_wt_lb": 122, "declared_horse_wt_lb": 1088, "draw": 2,
        "placing_raw": "1", "finish_pos": 1, "placing_code": None, "dead_heat": False,
        "lbw_raw": "---", "lbw_lengths": 0.0, "running_positions": [],
        "finish_time_s": approx(70.48), "win_odds": 4.8, "is_scratched": False,
    },
}  # fmt: skip
LAYOUTS = {
    "live": pages.LIVE,
    "archive_2010": pages.ARCHIVE_2010,
    "archive_2004": pages.ARCHIVE_2004,
}


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_winner_and_beaten_runners(layout):
    race = LAYOUTS[layout]
    runners = parse(race)["runners"]

    assert set(WINNERS[layout]) == RUNNER_COLUMNS
    assert RUNNER_COLUMNS <= set(schemas.TABLES["runners"].columns)
    assert runners[0] == WINNERS[layout]
    # every runner, in page order: placings, beaten lengths (N = a neck), positions, times, odds
    assert [tuple(r[f] for f in RESULT_FIELDS) for r in runners] == RESULTS[layout]
    # who ran, as printed: the profile link's id, the name, the brand, the people and their codes
    assert [
        (r["horse_no"], r["horse_id"], r["horse_name"], r["brand_raw"], r["jockey_code"],
         r["trainer_name"], r["trainer_code"])
        for r in runners
    ] == [
        (int(x.horse_no), x.horse_id, x.horse, x.brand, x.jockey_code, x.trainer, x.trainer_code)
        for x in race.runners
    ]  # fmt: skip
    assert {r["race_id"] for r in runners} == {race_id(race)}
    assert all(isinstance(r["win_odds"], float) for r in runners if r["win_odds"] is not None)


# ---------------------------------------------------------------- dividends

DIVIDENDS = {
    "live": [
        ("WIN", "6", 37.0, False),
        ("PLACE", "6", 15.5, False), ("PLACE", "2", 32.0, False), ("PLACE", "9", 61.5, False),
        ("QUINELLA", "2,6", 221.0, False),
        ("QUINELLA_PLACE", "2,6", 80.5, False), ("QUINELLA_PLACE", "6,9", 142.0, False),
        ("QUINELLA_PLACE", "2,9", 301.0, False),
        ("FORECAST", "6,2", 402.0, False),
        ("TIERCE", "6,2,9", 3918.0, False),
        ("TRIO", "2,6,9", 1066.0, False),
        ("FIRST_4", "2,6,9,11", 6251.0, False),
        ("QUARTET", "6,2,9,11", 57340.0, False),
        # "NOT WIN" and "Detail" rows are not dividends
        ("3_PICK_1_COMPOSITE_WIN", "A1", 10.1, False),
        ("3_PICK_1_COMPOSITE_WIN", "A3", 14.5, False),
        # a double's consolation row continues the pool: it is not a dead heat
        ("2ND_DOUBLE", "4/6", 96.5, False), ("2ND_DOUBLE", "4/2", 412.0, False),
    ],
    "archive_2010": [
        ("WIN", "3", 24.5, False),
        ("PLACE", "3", 12.0, False), ("PLACE", "8", 19.5, False), ("PLACE", "1", 27.0, False),
        ("QUINELLA", "3,8", 88.0, False),
        ("QUINELLA_PLACE", "3,8", 35.5, False), ("QUINELLA_PLACE", "1,3", 48.0, False),
        ("QUINELLA_PLACE", "1,8", 77.5, False),
        ("TIERCE", "3,8,1", 1290.0, False),
        ("TRIO", "1,3,8", 302.0, False),
        # quoted for a $1 partial unit: scaled to the $10 unit stake
        ("QUARTET", "3,8,1,6", approx(2_267_781.0), False),
        ("1ST_DOUBLE", "2/3", 61.0, False),
        ("TRIPLE_TRIO_CONSOLATION", "2,5,7/1,4,9/Any Combination", 5193.0, False),
    ],
}  # fmt: skip


@pytest.mark.parametrize("layout", list(DIVIDENDS))
def test_dividends(layout):
    """One row per winning combination, in page order. Pool codes come from hkjc.yaml; a label
    the config does not know keeps a slug of itself. Combinations are normalised, the legs of
    multi-race pools stay split by '/'."""
    race = LAYOUTS[layout]
    parsed = parse(race)
    assert dividend_rows(parsed) == DIVIDENDS[layout]
    assert all(set(d) == DIVIDEND_COLUMNS for d in parsed["dividends"])
    assert DIVIDEND_COLUMNS <= set(schemas.TABLES["dividends"].columns)
    assert {(d["jurisdiction"], d["race_id"]) for d in parsed["dividends"]} == {
        ("hkjc", race_id(race))
    }


# ---------------------------------------------------------------- incidents

PER_HORSE = {
    "live": (pages.LIVE, [
        (6, "HK_2023_X101", "Bumped at the start. Sent for sampling."),
        (2, "HK_2022_W215", "Raced keenly in the early stages."),
        (9, "HK_2024_Y033", "No information."),
        # an empty report is no row; a horse cell without a link takes the id by horse number
        (4, "HK_2021_V560", "Lost its off-fore plate."),
        (1, "HK_2022_W390",
         "Slow to begin. The Stewards noted that Race 3 at the previous meeting was "
         "declared void."),
        (12, "HK_2024_Y210", "Sent for sampling."),
        (7, "HK_2023_X300",
         "Withdrawn by the veterinary officer after becoming fractious in the stalls."),
    ]),
    "protest": (pages.PROTEST, [
        (4, "HK_2022_W341", "Bumped near the 300 Metres. Ran on strongly."),
        (3, "HK_2021_V442", "Shifted in near the 300 Metres and bumped THE BRONZE KNIGHT. "
                            "After an inquiry the Stewards placed it second."),
        (1, "HK_2023_X612", "Sent for sampling."),
        (6, "HK_2020_U118", "Pulled up near the 600 Metres."),
        (2, "HK_2022_W777", "Withdrawn by order of the Stewards."),
    ]),
}  # fmt: skip


@pytest.mark.parametrize("layout", list(PER_HORSE))
def test_per_horse_incident_table(layout):
    """The table with an Incident column is the report (not the runner table, though it shares
    the Horse No. and Horse columns): one row per horse with a report, withdrawn horses too."""
    race, expected = PER_HORSE[layout]
    incidents = parse(race)["incidents"]
    assert [(i["horse_no"], i["horse_id"], i["text"]) for i in incidents] == expected
    assert all(set(i) == INCIDENT_COLUMNS for i in incidents)
    assert INCIDENT_COLUMNS <= set(schemas.TABLES["incidents"].columns)
    assert {(i["jurisdiction"], i["race_id"]) for i in incidents} == {("hkjc", race_id(race))}
    # the runner table is still the performance table, not the report
    assert [r["horse_no"] for r in parse(race)["runners"]] == [
        int(x.horse_no) for x in race.runners
    ]


def test_narrative_incident_report_is_kept_whole_then_split_by_horse():
    """Archive era: one row with the whole report and no horse, then one row per runner it names
    (capitalised names, longest first, typographic apostrophes matched), in runner-table order.
    A sentence naming nobody continues the one before it, on the same line only."""
    incidents = parse(pages.ARCHIVE_2010)["incidents"]
    whole = "\n".join(
        [
            "SPARROW KING was slow to begin. On jumping, TIGER’S DAWN shifted in and bumped "
            "SPARROW. SPARROW then raced wide without cover.He said the gelding did not handle "
            "the bend.",
            "The start was delayed by two minutes.",
            "A veterinary inspection of COPPER BELL found it to be lame in the left foreleg. "
            "MISTY HARBOUR and RIVER OTTER were sent for sampling. COPPER BELL will be examined "
            "again before its next start.",
        ]
    )
    sampled = "MISTY HARBOUR and RIVER OTTER were sent for sampling."
    assert [(i["horse_no"], i["horse_id"], i["text"]) for i in incidents] == [
        (None, None, whole),
        (3, "HK_2009_Q101", "SPARROW KING was slow to begin."),
        (8, "HK_2009_Q215", sampled),
        (1, "HK_2008_P330", "On jumping, TIGER’S DAWN shifted in and bumped SPARROW."),
        (6, "HK_2009_Q044", "On jumping, TIGER’S DAWN shifted in and bumped SPARROW. "
                            "SPARROW then raced wide without cover. "
                            "He said the gelding did not handle the bend."),
        (2, "HK_2008_P512", "A veterinary inspection of COPPER BELL found it to be lame in the "
                            "left foreleg. COPPER BELL will be examined again before its next "
                            "start."),
        (7, "HK_2009_Q620", sampled),
    ]  # fmt: skip
    assert all(set(i) == INCIDENT_COLUMNS for i in incidents)
    assert "Racing Incident Report" not in whole


# ---------------------------------------------------------------- dead heats

# (horse_no, placing_raw, finish_pos, placing_code, dead_heat, lbw_raw, lbw_lengths, finish_time_s)
DEAD_HEATS = {
    "for_first": (pages.DEAD_HEAT, [
        (5, "1 DH", 1, "DH", True, "DH", 0.0, approx(70.62)),
        (12, "1 DH", 1, "DH", True, "DH", 0.0, approx(70.62)),
        (8, "3", 3, None, False, "NOSE", approx(0.05), approx(70.63)),
        (11, "4", 4, None, False, "SH", approx(0.1), approx(70.64)),
        (3, "5", 5, None, False, "HD", approx(0.2), approx(70.65)),
        (1, "6", 6, None, False, "2-1/2", 2.5, approx(71.02)),
    ], [
        ("WIN", "5", 20.5, False), ("WIN", "12", 65.5, False),
        ("PLACE", "5", 15.5, False), ("PLACE", "12", 35.0, False), ("PLACE", "8", 68.5, False),
        ("QUINELLA", "5,12", 274.0, False),
        ("QUINELLA_PLACE", "5,12", 97.5, False), ("QUINELLA_PLACE", "5,8", 182.0, False),
        ("QUINELLA_PLACE", "8,12", 515.0, False),
        ("FORECAST", "5,12", 251.0, False), ("FORECAST", "12,5", 433.0, False),
        ("TIERCE", "5,12,8", 5044.0, False), ("TIERCE", "12,5,8", 8120.0, False),
        ("TRIO", "5,8,12", 2710.0, False),
        ("QUARTET", "5,12,8,11", 46980.0, False), ("QUARTET", "12,5,8,11", 77105.0, False),
    ], approx(70.62)),
    "for_third": (pages.DEAD_HEAT_THIRD, [
        (2, "1", 1, None, False, "---", 0.0, approx(99.80)),
        (7, "2", 2, None, False, "1/2", 0.5, approx(99.88)),
        (4, "3 DH", 3, "DH", True, "1-1/4", 1.25, approx(100.00)),
        (10, "3 DH", 3, "DH", True, "1-1/4", 1.25, approx(100.00)),
        (1, "5", 5, None, False, "2-3/4", 2.75, approx(100.24)),
    ], [
        ("WIN", "2", 32.5, False),
        ("PLACE", "2", 14.0, False), ("PLACE", "7", 20.5, False), ("PLACE", "4", 22.0, False),
        ("PLACE", "10", 39.5, False),
        ("QUINELLA", "2,7", 66.0, False),
        ("TIERCE", "2,7,4", 412.0, False), ("TIERCE", "2,7,10", 655.0, False),
        ("TRIO", "2,4,7", 133.0, False), ("TRIO", "2,7,10", 241.0, False),
    ], approx(99.80)),
}  # fmt: skip
DEAD_HEAT_FIELDS = (
    "horse_no", "placing_raw", "finish_pos", "placing_code", "dead_heat", "lbw_raw",
    "lbw_lengths", "finish_time_s",
)  # fmt: skip


@pytest.mark.parametrize("which", list(DEAD_HEATS))
def test_a_dead_heat(which):
    """Both horses of a dead heat get the same placing, flagged DH; the pools print one row per
    winning combination and every row is kept."""
    race, runners, dividends, race_time = DEAD_HEATS[which]
    parsed = parse(race)
    assert [tuple(r[f] for f in DEAD_HEAT_FIELDS) for r in parsed["runners"]] == runners
    assert dividend_rows(parsed) == dividends
    (row,) = parsed["races"]
    assert (row["race_id"], row["status"], row["field_size"]) == (
        race_id(race),
        "run",
        len(race.runners),
    )
    assert row["finish_time_s"] == race_time


# ---------------------------------------------------------------- races without a result

# (horse_no, horse_id, placing_raw, finish_pos, placing_code, win_odds, actual_wt_lb, draw,
#  running_positions, finish_time_s, lbw_lengths, is_scratched, dead_heat)
NOT_RUN_FIELDS = (
    "horse_no", "horse_id", "placing_raw", "finish_pos", "placing_code", "win_odds",
    "actual_wt_lb", "draw", "running_positions", "finish_time_s", "lbw_lengths", "is_scratched",
    "dead_heat",
)  # fmt: skip
REFUNDED = [
    ("WIN", None, None, True), ("PLACE", None, None, True), ("QUINELLA", None, None, True),
    ("QUINELLA_PLACE", None, None, True),
]  # fmt: skip
NO_RESULT = {
    "abandoned_meeting": (
        pages.ABANDONED,
        race_row("2022-12-07", "HV", 1, race_index=None, race_name="BAMBOO PATH HANDICAP",
                 race_class="Class 5", rating_hi=40, rating_lo=0, distance_m=1200,
                 surface="turf", course_config="C", going=None, prize=760_000, field_size=4,
                 finish_time_s=None, split_times_s=[], status="abandoned"),
        [
            (1, "HK_2020_U301", None, None, None, None, None, 12, [], None, None, False, False),
            (2, "HK_2019_T402", None, None, None, None, None, 7, [], None, None, False, False),
            (3, "HK_2021_V503", None, None, None, None, None, 9, [], None, None, False, False),
            (4, "HK_2020_U604", None, None, None, None, None, 3, [], None, None, False, False),
        ],
        [*REFUNDED, ("3_PICK_1_COMPOSITE_WIN", "A1", None, True),
         ("3_PICK_1_COMPOSITE_WIN", "A2", None, True),
         ("3_PICK_1_COMPOSITE_WIN", "A3", None, True)],
        [],
    ),
    "declared_abandoned_without_header_or_runners": (
        pages.DECLARED_ABANDONED,
        race_row("2025-11-12", "HV", 7, race_index=None, race_name=None, race_class=None,
                 rating_hi=None, rating_lo=None, distance_m=None, surface=None,
                 course_config=None, going=None, prize=None, field_size=0, finish_time_s=None,
                 split_times_s=[], status="abandoned"),
        [],
        [*REFUNDED, ("FORECAST", None, None, True), ("TIERCE", None, None, True),
         ("TRIO", None, None, True), ("FIRST_4", None, None, True),
         ("QUARTET", None, None, True),
         # multi-race pools that ended on this leg still pay
         ("6TH_DOUBLE", "7/F", 31.5, False), ("3RD_DOUBLE_TRIO", "7,10,11/F", 2904.0, False)],
        [],
    ),
    "void_race": (
        pages.VOID,
        race_row("2025-10-18", "ST", 8, race_index=214, race_name="OPAL TERRACE HANDICAP",
                 race_class="Class 4", rating_hi=60, rating_lo=40, distance_m=1200,
                 surface="turf", course_config="A+3", going=None, prize=1_150_000,
                 field_size=4, finish_time_s=None, split_times_s=[], status="abandoned"),
        [
            (1, "HK_2021_V033", "VOID", None, "VOID", None, None, 12, [], None, None, False, False),
            (2, "HK_2022_W122", "VOID", None, "VOID", None, None, 9, [], None, None, False, False),
            (3, "HK_2021_V056", "VOID", None, "VOID", None, None, 4, [], None, None, False, False),
            (4, "HK_2022_W046", "VOID", None, "VOID", None, None, 1, [], None, None, False, False),
            # withdrawn before the start: no number printed, still a runner row
            (None, "HK_2023_X547", "WV", None, "WV", None, 121, None, [], None, None, True, False),
        ],
        [*REFUNDED, ("FORECAST", None, None, True), ("TIERCE", None, None, True)],
        # the stewards' per-horse table is still published
        [
            (1, "HK_2021_V033", "Hampered by DREAMY HOLLOW at the start."),
            (3, "HK_2021_V056", "Raced without cover."),
            (4, "HK_2022_W046", "Dislodged its rider soon after the start and galloped riderless."),
        ],
    ),
}  # fmt: skip


@pytest.mark.parametrize("shape", list(NO_RESULT))
def test_an_abandoned_meeting_and_a_void_race(shape):
    """Each of the three shapes of a race without a result is one races row with status
    "abandoned", no race time and no splits; the refunds are dividend rows with no combination
    and no amount."""
    race, expected_race, runners, dividends, incidents = NO_RESULT[shape]
    parsed = parse(race)
    assert parsed["races"] == [expected_race]
    assert [tuple(r[f] for f in NOT_RUN_FIELDS) for r in parsed["runners"]] == runners
    assert dividend_rows(parsed) == dividends
    assert {d["race_id"] for d in parsed["dividends"]} == {race_id(race)}
    assert [(i["horse_no"], i["horse_id"], i["text"]) for i in parsed["incidents"]] == incidents


# ---------------------------------------------------------------- placings amended after a protest


def test_a_protest_relegation():
    """The official placing is `finish_pos`; the finishing order survives in the times. The
    promoted winner's negative margin and the relegated horse's '---' are both 0 lengths;
    '---' stays unknown for a horse that did not finish or did not start."""
    parsed = parse(pages.PROTEST)
    runners = by_no(parsed)
    promoted, relegated = runners[4], runners[3]

    fields = ("placing_raw", "finish_pos", "lbw_raw", "lbw_lengths", "dead_heat")
    assert tuple(promoted[f] for f in fields) == ("1", 1, "-HD", 0.0, False)
    assert tuple(relegated[f] for f in fields) == ("2", 2, "---", 0.0, False)
    assert relegated["finish_time_s"] < promoted["finish_time_s"]

    placed = sorted(
        (r for r in parsed["runners"] if r["finish_pos"]), key=lambda r: r["finish_pos"]
    )
    home = sorted(
        (r for r in parsed["runners"] if r["finish_time_s"] is not None),
        key=lambda r: r["finish_time_s"],
    )
    assert [r["horse_no"] for r in placed] == [4, 3, 9, 1]
    assert [r["horse_no"] for r in home] == [3, 4, 9, 1]

    assert [(r["horse_no"], r["lbw_lengths"]) for r in parsed["runners"]] == [
        (4, 0.0), (3, 0.0), (9, 0.5), (1, 1.25), (6, None), (2, None),
    ]  # fmt: skip
    pulled_up, withdrawn = runners[6], runners[2]
    assert (pulled_up["placing_code"], pulled_up["finish_pos"], pulled_up["is_scratched"]) == (
        "PU",
        None,
        False,
    )
    assert (pulled_up["running_positions"], pulled_up["win_odds"]) == ([6, 6], 48.0)
    assert (withdrawn["placing_code"], withdrawn["is_scratched"]) == ("WX-A", True)

    (race,) = parsed["races"]
    # the race time is the first horse home's, the winner is the promoted horse
    assert (race["status"], race["field_size"]) == ("run", 5)
    assert race["finish_time_s"] == approx(94.70)
    assert race["split_times_s"] == approx([24.31, 47.25, 70.90, 94.70])
    assert dividend_rows(parsed)[:4] == [
        ("WIN", "4", 29.5, False),
        ("PLACE", "4", 13.0, False),
        ("PLACE", "3", 26.5, False),
        ("PLACE", "9", 41.0, False),
    ]


# ---------------------------------------------------------------- which race the page shows

LIVE_DAY, ARCHIVE_DAY = pages.LIVE.race_date, pages.ARCHIVE_2010.race_date
IDENTITY = [
    pytest.param(pages.LIVE, pages.LIVE.key, pages.live_url(LIVE_DAY, "HV", 3),
                 "hkjc:2026-10-14:HV:03", False, id="page_and_key_agree"),
    pytest.param(pages.LIVE, "2026-10-14_HV_R07", pages.live_url(LIVE_DAY, "HV", 7),
                 "hkjc:2026-10-14:HV:03", True, id="page_wins_over_a_wrong_key"),
    pytest.param(replace(pages.LIVE, meeting_label=False), "2026-10-14_HV_R03",
                 f"{pages.HOST}/x", "hkjc:2026-10-14:HV:03", False, id="key_fills_the_meeting"),
    pytest.param(replace(pages.LIVE, meeting_label=False), "not-a-key",
                 pages.live_url(LIVE_DAY, "HV", 9), "hkjc:2026-10-14:HV:03", False,
                 id="live_url_fills_the_meeting_header_keeps_the_race"),
    pytest.param(replace(pages.ARCHIVE_2010, meeting_label=False), "not-a-key",
                 pages.archive_url(ARCHIVE_DAY, "ST", 1), "hkjc:2011-01-30:ST:01", False,
                 id="archive_url_fills_the_meeting"),
    pytest.param(replace(pages.DECLARED_ABANDONED, tabs=False), "2025-11-12_HV_R08",
                 pages.archive_url(date(2025, 11, 12), "HV", 8), "hkjc:2025-11-12:HV:08", False,
                 id="no_header_no_tabs_the_key_names_the_race"),
    pytest.param(pages.DECLARED_ABANDONED, "2025-11-12_HV_R01",
                 pages.archive_url(date(2025, 11, 12), "HV", 1), "hkjc:2025-11-12:HV:07", True,
                 id="no_header_the_unlinked_tab_names_the_race"),
]  # fmt: skip


@pytest.mark.parametrize(("race", "key", "url", "expected", "mismatch"), IDENTITY)
def test_the_page_names_the_race_then_the_key_then_the_live_or_archive_url(
    race, key, url, expected, mismatch
):
    parsed = parse_results.parse(raw(race, key, url))
    (row,) = parsed["races"]
    assert row["race_id"] == expected
    assert row["meeting_id"] == expected.rsplit(":", 1)[0]
    assert row["race_no"] == int(expected.rsplit(":", 1)[1])
    assert {r["race_id"] for t in ("runners", "dividends", "incidents") for r in parsed[t]} <= {
        expected
    }
    assert any(m.get("key_mismatch") for m in parsed["_meta"]) is mismatch


def test_without_any_race_identity_the_tables_are_empty():
    nowhere = replace(pages.LIVE, meeting_label=False)
    parsed = parse_results.parse(raw(nowhere, "not-a-key", f"{pages.HOST}/en-us/local/information"))
    assert {t: parsed[t] for t in TABLES} == {t: [] for t in TABLES}


# ---------------------------------------------------------------- malformed and untrusted cells


def test_malformed_cells_become_null_and_never_stop_the_page():
    parsed = parse(pages.MALFORMED)
    (race,) = parsed["races"]
    assert (race["race_class"], race["distance_m"], race["rating_hi"], race["prize"]) == (
        "Group One",
        2400,
        None,
        24_000_000,
    )
    assert (race["field_size"], race["finish_time_s"]) == (5, approx(147.35))
    # the stray note row is not a runner
    assert [r["horse_no"] for r in parsed["runners"]] == [4, 7, 8, 9, 10, 11]
    runners = by_no(parsed)

    winner = runners[4]
    assert (winner["jockey_code"], winner["jockey_key"], winner["lbw_lengths"]) == (
        "QSP",
        "jockey:p quayside",
        0.0,
    )
    assert (winner["trainer_code"], winner["trainer_key"]) == (None, "trainer:j merrow")

    junk = runners[7]
    assert (junk["finish_pos"], junk["dead_heat"], junk["placing_code"]) == (2, True, "DH")
    assert (junk["declared_horse_wt_lb"], junk["draw"], junk["win_odds"]) == (None, None, None)
    assert (junk["lbw_raw"], junk["lbw_lengths"]) == ("WEIRD", None)
    assert junk["running_positions"] == [1, 1]
    assert junk["finish_time_s"] == approx(147.50)  # "2.27.50"
    # ids from scraped links end up in keys and file names: plain charset only
    assert (junk["jockey_code"], junk["jockey_name"], junk["jockey_key"]) == (
        None,
        "A Bellweather",
        "jockey:a bellweather",
    )

    pulled_up = runners[9]
    assert (pulled_up["placing_code"], pulled_up["finish_pos"], pulled_up["is_scratched"]) == (
        "PU",
        None,
        False,
    )
    assert (pulled_up["win_odds"], pulled_up["lbw_lengths"], pulled_up["running_positions"]) == (
        99.0,
        None,
        [],
    )
    assert (runners[10]["placing_code"], runners[10]["is_scratched"]) == ("WV", True)

    short = runners[11]
    assert (short["horse_id"], short["horse_name"], short["brand_raw"], short["finish_pos"]) == (
        None,
        "FROSTY LARK",
        "Z600",
        5,
    )
    assert (short["jockey_name"], short["jockey_key"], short["running_positions"]) == (
        None,
        None,
        [],
    )

    assert dividend_rows(parsed) == [
        ("WIN", "4", 19.0, False),
        ("PLACE", "4", 1234.5, False),
        ("PLACE", "7,8", None, False),
    ]
    assert parsed["incidents"] == []


# ---------------------------------------------------------------- classification

LIVE_SENTINEL = ("2026-09-20_ST_R01", pages.live_url(date(2026, 9, 20), "ST", 1))
ARCHIVE_SENTINEL = ("2000-01-01_ST_R01", pages.archive_url(date(2000, 1, 1), "ST", 1))


def page_of(race: pages.Race) -> tuple[bytes, str, str]:
    page = raw(race)
    return page.body, page.key, page.url


CLASSIFIED = [
    pytest.param(pages.no_information(css=False), *ARCHIVE_SENTINEL, STATUS_EMPTY,
                 id="no_information"),
    # the route's inline CSS names the results containers: not content
    pytest.param(pages.no_information(), *LIVE_SENTINEL, STATUS_EMPTY,
                 id="no_information_with_inline_css"),
    pytest.param(pages.no_information(script=True), *LIVE_SENTINEL, STATUS_EMPTY,
                 id="no_information_with_a_script_and_a_comment"),
    # a result whose stewards' report happens to say "No information." is still a result
    pytest.param(*page_of(pages.LIVE), STATUS_FETCHED, id="result"),
    pytest.param(*page_of(pages.PROTEST), STATUS_FETCHED, id="result_after_a_protest"),
    pytest.param(*page_of(pages.ABANDONED), STATUS_ABANDONED, id="abandoned"),
    pytest.param(*page_of(pages.DECLARED_ABANDONED), STATUS_ABANDONED, id="declared_abandoned"),
    pytest.param(*page_of(pages.VOID), STATUS_ABANDONED, id="void"),
    pytest.param(pages.busy_page(), *LIVE_SENTINEL, STATUS_BLOCKED, id="neither"),
    pytest.param(b"", *LIVE_SENTINEL, STATUS_BLOCKED, id="empty_body"),
]  # fmt: skip


@pytest.mark.parametrize(("body", "key", "url", "status"), CLASSIFIED)
def test_no_information_is_empty(body, key, url, status):
    """Decided by content, never by HTTP status: the site answers a missing race or meeting with
    200 and "No information.". A page that is not a result puts nothing in the tables."""
    assert parse_results.classify(body) == status
    parsed = parse_results.parse(RawPage("hkjc", "results", key, url, body))
    if status in (STATUS_EMPTY, STATUS_BLOCKED):
        assert {t: parsed[t] for t in TABLES} == {t: [] for t in TABLES}
        assert set(parsed) <= {*TABLES, "_meta"}
    else:
        assert len(parsed["races"]) == 1
