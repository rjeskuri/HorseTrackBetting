"""HKJC URL builders and cache keys (ticket #60, ported from HKTrack's `sources/hkjc/urls.py`).

Contract pinned here: every builder returns the FINAL address under
`https://racing.hkjc.com/en-us/local/information` (the host from
countries/hongkong/config/hkjc.yaml).
The live route serves the current season only, which starts on the 1st of September; every earlier
date goes straight to `/archive/`. Each endpoint has its own date format and parameter spelling:
results YYYY/MM/DD, sectionals DD/MM/YYYY, corunning YYYYMMDD.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

# isort: split
# (its own block: ruff ranks a module that does not exist yet as third-party)
from horsetrack.countries.hongkong.hkjc import urls

BASE = "https://racing.hkjc.com/en-us/local/information"


def season_start(today: date) -> date:
    """1 September of the season `today` falls in, worked out here from the season rule."""
    return date(today.year if today.month >= 9 else today.year - 1, 9, 1)


@pytest.mark.parametrize(
    ("year", "month", "query"),
    [
        (2015, 3, "calyear=2015&calmonth=03"),
        (2026, 11, "calyear=2026&calmonth=11"),
        (2010, 1, "calyear=2010&calmonth=01"),
        (1999, 9, "calyear=1999&calmonth=09"),
        (2031, 12, "calyear=2031&calmonth=12"),
    ],
)
def test_fixture_url_zero_pads_the_month(year, month, query):
    assert urls.fixture(year, month) == f"{BASE}/fixture?{query}"


@pytest.mark.parametrize(
    ("today", "first_live_day"),
    [
        (date(2026, 9, 21), date(2026, 9, 1)),
        (date(2026, 9, 1), date(2026, 9, 1)),  # the season's first day
        (date(2026, 8, 31), date(2025, 9, 1)),  # its last day
        (date(2027, 3, 1), date(2026, 9, 1)),
        (date(2011, 1, 1), date(2010, 9, 1)),
        (date(2030, 12, 31), date(2030, 9, 1)),
    ],
)
def test_results_url_uses_the_live_route_for_recent_dates_and_the_archive_route_for_old(
    today, first_live_day
):
    # the rule, at a given today: on and after the season's first day is live, the day before is not
    assert urls.live_route_from(today) == first_live_day
    assert not urls.is_archived(first_live_day, today)
    assert not urls.is_archived(today, today)
    assert urls.is_archived(first_live_day - timedelta(days=1), today)
    assert urls.is_archived(date(2004, 3, 7), today)

    # the builder, at the real today: the boundary is this season's 1 September
    start = season_start(date.today())
    assert urls.results(start, "HV", 1) == (
        f"{BASE}/localresults?racedate={start:%Y/%m/%d}&Racecourse=HV&RaceNo=1"
    )
    day_before = start - timedelta(days=1)
    assert urls.results(day_before, "ST", 11) == (
        f"{BASE}/archive/localresults?racedate={day_before:%Y/%m/%d}&racecourse=ST&RaceNo=11"
    )
    assert urls.results(date(2010, 1, 1), "ST", 3) == (
        f"{BASE}/archive/localresults?racedate=2010/01/01&racecourse=ST&RaceNo=3"
    )


def test_sectional_and_corunning_url_date_formats():
    start = season_start(date.today())
    live = start + timedelta(days=15)  # inside the current season, even when that is still ahead
    old = date(2010, 1, 9)  # day and month differ, so a swapped format shows

    assert urls.sectional(live, 1) == (
        f"{BASE}/displaysectionaltime?racedate={live:%d/%m/%Y}&RaceNo=1"
    )
    assert (
        urls.sectional(old, 3)
        == f"{BASE}/archive/displaysectionaltime?racedate=09/01/2010&RaceNo=3"
    )
    assert urls.corunning(live, 7) == f"{BASE}/corunning?Date={live:%Y%m%d}&RaceNo=7"
    assert urls.corunning(old, 2) == f"{BASE}/archive/corunning?date=20100109&raceno=2"
    # the day before the season starts is already archived
    edge = start - timedelta(days=1)
    assert urls.sectional(edge, 10) == (
        f"{BASE}/archive/displaysectionaltime?racedate={edge:%d/%m/%Y}&RaceNo=10"
    )
    assert urls.corunning(edge, 10) == f"{BASE}/archive/corunning?date={edge:%Y%m%d}&raceno=10"


def test_horse_racecard_and_entries_urls():
    assert urls.horse("HK_2023_J087") == f"{BASE}/horse?horseid=HK_2023_J087&Option=1"
    assert urls.horse("HK_2019_A123") == f"{BASE}/horse?horseid=HK_2019_A123&Option=1"
    # an id never adds a parameter of its own
    assert urls.horse("HK_2023_J087&Option=2 x") == (
        f"{BASE}/horse?horseid=HK_2023_J087%26Option%3D2%20x&Option=1"
    )
    assert urls.racecard(date(2026, 9, 23), "HV", 4) == (
        f"{BASE}/racecard?racedate=2026/09/23&Racecourse=HV&RaceNo=4"
    )
    assert urls.racecard(date(2027, 1, 3), "ST", 10) == (
        f"{BASE}/racecard?racedate=2027/01/03&Racecourse=ST&RaceNo=10"
    )
    assert urls.entries(date(2026, 9, 23), "HV") == (
        f"{BASE}/entries?racedate=2026/09/23&Racecourse=HV&View=All"
    )
    assert urls.entries(date(2027, 2, 7), "ST") == (
        f"{BASE}/entries?racedate=2027/02/07&Racecourse=ST&View=All"
    )


@pytest.mark.parametrize(
    ("race_date", "venue", "race_no", "key"),
    [
        (date(2026, 9, 23), "HV", 4, "2026-09-23_HV_R04"),
        (date(2010, 1, 1), "ST", 11, "2010-01-01_ST_R11"),
        (date(2024, 2, 29), "ST", 1, "2024-02-29_ST_R01"),
    ],
)
def test_race_keys_round_trip(race_date, venue, race_no, key):
    assert urls.meeting_key(race_date, venue) == key.rsplit("_", 1)[0]
    assert urls.race_key(race_date, venue, race_no) == key
    assert urls.parse_race_key(key) == (race_date, venue, race_no)


@pytest.mark.parametrize("key", ["2026-09", "2026-13-45_HV_R01", "2023-02-29_ST_R01", "", None])
def test_a_key_that_is_not_a_race_parses_to_none(key):
    assert urls.parse_race_key(key) is None
