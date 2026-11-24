"""HKJC horse profile page -> `horses` (static attributes only) + `horse_runs` (the form table)
(ticket #63, ported from HKTrack's `sources/hkjc/parse_horse.py`).

Contract pinned here (the coder follows it):
- `horsetrack.countries.hongkong.hkjc.parse_horse.parse(page: RawPage) -> ParsedPage`
  (`horsetrack.core.ingest.types`), pure, no network, returning `{"horses": [...], "horse_runs":
  [...]}`; plus `parse_run_date(text, hint=None)` and `PARSER_VERSION`.
- `horses`: one row with exactly the columns of `schemas.TABLES["horses"]`. Never any as-of-scrape
  value from the profile block (current / last / start-of-season rating, stakes, record, current
  trainer, owner, age, location, import date): those are look-ahead leakage. Rating-at-run and
  trainer-at-run come from `horse_runs` only.
- `horse_runs`: one row per local run on the form table, exactly the columns of
  `schemas.TABLES["horse_runs"]`, in page order. Columns are mapped by header text; season
  separators and overseas rows are dropped; withdrawals are kept with nulls; `lbw_raw` and
  `placing_raw` stay as printed (zero padding dropped); `race_id` only when the Race Index link
  names the row's own day and venue.
- The horse id is the cache key, then the URL's `horseid` (requested, then `meta["final_url"]`),
  then the id most links on the page point at; anything outside `[A-Za-z0-9_]` is refused. Keyed on
  horse_id: brands and names are recycled.

Pages are synthetic (hkjc_horse_pages.py); every expected value below is written out by hand from
the printed cells.
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime, timezone

import hkjc_horse_pages as pages
import hkjc_pages
import hkjc_sectional_pages
import pytest
from hkjc_horse_pages import Profile, Run, Season, horse_url, results_href

from horsetrack.core.ingest.types import RawPage
from horsetrack.core.store import schemas

# isort: split
# (its own block: ruff ranks a module that does not exist yet as third-party)
from horsetrack.countries.hongkong.hkjc import parse_horse

SCRAPED = datetime(2026, 9, 20, 15, 53, 7, tzinfo=timezone.utc)

HORSE_COLUMNS = {
    "jurisdiction", "horse_id", "name", "origin", "colour", "sex", "import_type", "sire", "dam",
    "dam_sire", "is_retired", "profile_scraped_utc",
}  # fmt: skip
RUN_COLUMNS = {
    "jurisdiction", "horse_id", "race_date", "race_index", "venue", "surface", "course_config",
    "distance_m", "going", "race_class", "draw", "rating", "trainer_name", "jockey_name", "lbw_raw",
    "win_odds", "actual_wt_lb", "running_positions", "finish_time_s", "declared_horse_wt_lb",
    "gear_raw", "placing_raw", "race_id",
}  # fmt: skip
# as-of-scrape values on the profile block: using any of them as a feature is look-ahead leakage
FORBIDDEN = {
    "rating", "current_rating", "last_rating", "start_of_season_rating", "season_stakes",
    "total_stakes", "trainer", "trainer_name", "current_trainer", "owner", "age", "record",
    "starts", "location", "import_date",
}  # fmt: skip
AS_OF_SCRAPE_FIELDS = (
    "age", "season_stakes", "total_stakes", "record", "location", "arrival", "import_date",
    "trainer", "owner", "current_rating", "season_start_rating", "last_rating",
)  # fmt: skip


def raw(
    body: bytes | str,
    key: str,
    url: str,
    fetched: datetime | None = SCRAPED,
    meta: dict | None = None,
) -> RawPage:
    data = body.encode() if isinstance(body, str) else body
    return RawPage("hkjc", "horse", key, url, data, fetched, meta or {})


def run_row(horse_id: str, race_date: date, **fields) -> dict:
    row = {"jurisdiction": "hkjc", "horse_id": horse_id, "race_date": race_date, **fields}
    assert set(row) == RUN_COLUMNS, set(row) ^ RUN_COLUMNS  # a typo in the expectation itself
    return row


# --- the active horse: /horse layout, DD/MM/YY dates ---------------------------------------------

ACTIVE_ID = "HK_2023_X901"
ACTIVE = Profile(
    horse_id=ACTIVE_ID,
    title="QUIET ORBIT (X901)",
    origin="NZ",
    colour_sex="Chestnut / Gelding",
    import_type="PPG",
    sire="Harbour Light",
    dam="Glass Lantern",
    dam_sire="Copper Bell",
    age="6",
    current_rating="47",
    season_start_rating="41",
    trainer="T Example",
    owner="Sample Owner Syndicate",
)
ACTIVE_HORSE = {
    "jurisdiction": "hkjc",
    "horse_id": ACTIVE_ID,
    "name": "QUIET ORBIT",
    "origin": "NZ",  # printed "NZ / 6": the age is never kept
    "colour": "Chestnut",
    "sex": "Gelding",
    "import_type": "PPG",
    "sire": "Harbour Light",
    "dam": "Glass Lantern",
    "dam_sire": "Copper Bell",
    "is_retired": False,
    "profile_scraped_utc": SCRAPED,
}
ACTIVE_ROWS = [
    Season("26/27"),
    Run("16/09/26", 'HV / Turf / "B"', "029", results_href("2026/09/16", "HV", 1), "01",
        "1000", "G", "5", "3", "39", "T Example", "J Rider", "1-3/4", "4.5", "123", "1 1 1",
        "0.57.54", "1111", "H/TT"),
    Season("25/26"),
    Run("06/05/26", "ST / AWT", "666", results_href("2026/05/06", "ST", 2), "06",
        "1200", "GD", "5", "4", "35", "T Example", "K Pilot", "4-1/4", "9.1", "130", "1 1 6",
        "1.11.57", "1114", "H/TT"),
    Run("28/01/26", 'HV / Turf / "C"', "410", results_href("2026/01/28", "HV", 7), "03",
        "1650", "GF", "4", "10", "40", "T Example", "M Rider", "N", "23", "118", "8 8 7 3",
        "1.40.05", "1098", "H2/TT"),
    Season("24/25"),
    Run("05/07/25", 'ST / Turf / "C+3"', "812", results_href("2025/07/05", "ST", 5), "WV-A",
        "1200", "GF", "4", "--", "49", "T Example", "C Sample", "--", "--", "122", "--",
        "--", "--", "--"),
    Run("01/01/25", "ST / Turf / “A+3”", "301", results_href("2025/01/01", "ST", 3), "12",
        "1400", "Y", "3", "14", "52", "P Former", "J Rider", "8-1/2", "99", "115", "2 3 9 12",
        "1.22.80", "1120", "B-/XB1/TT"),
    Season("23/24"),
    Run("20/03/24", 'ST / Turf / "C+3"', "514", results_href("2024/03/20", "ST", 2), "08",
        "1200", "G", "4", "4", "57", "P Former", "Z Rider", "5", "7.8", "126", "4 5 8",
        "1.09.95", "1135", "--"),
]  # fmt: skip
ACTIVE_RUNS = [
    run_row(ACTIVE_ID, date(2026, 9, 16), race_index=29, venue="HV", surface="turf",
            course_config="B", distance_m=1000, going="G", race_class="5", draw=3, rating=39,
            trainer_name="T Example", jockey_name="J Rider",
            lbw_raw="1-3/4",  # a winner: the horse page prints the winning margin here
            win_odds=4.5, actual_wt_lb=123, running_positions=[1, 1, 1],
            finish_time_s=pytest.approx(57.54), declared_horse_wt_lb=1111, gear_raw="H/TT",
            placing_raw="1", race_id="hkjc:2026-09-16:HV:01"),
    run_row(ACTIVE_ID, date(2026, 5, 6), race_index=666, venue="ST", surface="awt",
            course_config=None, distance_m=1200, going="GD", race_class="5", draw=4, rating=35,
            trainer_name="T Example", jockey_name="K Pilot", lbw_raw="4-1/4", win_odds=9.1,
            actual_wt_lb=130, running_positions=[1, 1, 6], finish_time_s=pytest.approx(71.57),
            declared_horse_wt_lb=1114, gear_raw="H/TT", placing_raw="6",
            race_id="hkjc:2026-05-06:ST:02"),
    run_row(ACTIVE_ID, date(2026, 1, 28), race_index=410, venue="HV", surface="turf",
            course_config="C", distance_m=1650, going="GF", race_class="4", draw=10, rating=40,
            trainer_name="T Example", jockey_name="M Rider", lbw_raw="N", win_odds=23.0,
            actual_wt_lb=118, running_positions=[8, 8, 7, 3],
            finish_time_s=pytest.approx(100.05), declared_horse_wt_lb=1098, gear_raw="H2/TT",
            placing_raw="3", race_id="hkjc:2026-01-28:HV:07"),
    # a withdrawal: kept, the rating going in kept, the result columns null
    run_row(ACTIVE_ID, date(2025, 7, 5), race_index=812, venue="ST", surface="turf",
            course_config="C+3", distance_m=1200, going="GF", race_class="4", draw=None,
            rating=49, trainer_name="T Example", jockey_name="C Sample", lbw_raw=None,
            win_odds=None, actual_wt_lb=122, running_positions=[], finish_time_s=None,
            declared_horse_wt_lb=None, gear_raw=None, placing_raw="WV-A",
            race_id="hkjc:2025-07-05:ST:05"),
    # under the previous trainer, with gear removed and first-time codes; curly course quotes
    run_row(ACTIVE_ID, date(2025, 1, 1), race_index=301, venue="ST", surface="turf",
            course_config="A+3", distance_m=1400, going="Y", race_class="3", draw=14, rating=52,
            trainer_name="P Former", jockey_name="J Rider", lbw_raw="8-1/2", win_odds=99.0,
            actual_wt_lb=115, running_positions=[2, 3, 9, 12],
            finish_time_s=pytest.approx(82.80), declared_horse_wt_lb=1120,
            gear_raw="B-/XB1/TT", placing_raw="12", race_id="hkjc:2025-01-01:ST:03"),
    # debut, no gear ("--")
    run_row(ACTIVE_ID, date(2024, 3, 20), race_index=514, venue="ST", surface="turf",
            course_config="C+3", distance_m=1200, going="G", race_class="4", draw=4, rating=57,
            trainer_name="P Former", jockey_name="Z Rider", lbw_raw="5", win_odds=7.8,
            actual_wt_lb=126, running_positions=[4, 5, 8], finish_time_s=pytest.approx(69.95),
            declared_horse_wt_lb=1135, gear_raw=None, placing_raw="8",
            race_id="hkjc:2024-03-20:ST:02"),
]  # fmt: skip

# --- the retired horse: /otherhorse layout, DD/MM/YYYY dates -------------------------------------

RETIRED_ID = "HK_2008_Y277"
RETIRED = Profile(
    horse_id=RETIRED_ID,
    title="SALT MEADOW (CY277) (Retired)",
    retired=True,
    origin="AUS",
    colour_sex="Bay / Gelding",
    import_type="PP",
    sire="Night Tide",
    dam="Field Of Glass",
    dam_sire="Old Compass",
    owner="Retired Owner Group",
    last_rating="64",
    total_stakes="$1,264,500",
    record="2-3-0-12",
)
RETIRED_HORSE = {
    "jurisdiction": "hkjc",
    "horse_id": RETIRED_ID,
    "name": "SALT MEADOW",  # title prints "SALT MEADOW (CY277) (Retired)"
    "origin": "AUS",  # retired layout: "Country of Origin" with no age
    "colour": "Bay",
    "sex": "Gelding",
    "import_type": "PP",
    "sire": "Night Tide",
    "dam": "Field Of Glass",
    "dam_sire": "Old Compass",
    "is_retired": True,
    "profile_scraped_utc": SCRAPED,
}
RETIRED_ROWS = [
    Season("11/12"),
    Run("19/02/2012", 'ST / Turf / "A"', "411", results_href("2012/02/19", "ST", 5), "04",
        "1400", "G", "4", "12", "64", "Q Retired", "J Rider", "1-1/2", "35", "130", "11 11 11 4",
        "1.23.18", "1079", "B/H"),
    Season("10/11"),
    Run("20/04/2011", 'HV / Turf / "C"', "602", results_href("2011/04/20", "HV", 6), "12",
        "1650", "G", "3", "8", "68", "Q Retired", "K Pilot", "6-1/4", "41", "121", "8 9 11 12",
        "1.42.99", "1081", "B/H"),
    Run("06/11/2010", 'ST / Turf / "C+3"', "155", results_href("2010/11/06", "ST", 4), "02",
        "1200", "GY", "3", "6", "72", "Q Retired", "J Rider", "SH", "6.4", "126", "3 3 2",
        "1.10.31", "1090", "B1/H"),
    Season("09/10"),
    Run("01/01/2010", "ST / AWT", "275", results_href("2010/01/01", "ST", 1), "01",
        "1200", "WS", "5", "9", "54", "Q Retired", "D Sample", "1-3/4", "2", "133", "1 1 1",
        "1.09.40", "1070", "H"),
    Run("29/11/2009", 'ST / Turf / "B"', "190", results_href("2009/11/29", "ST", 8), "WV",
        "1000", "G", "5", "--", "52", "Q Retired", "--", "--", "--", "126", "--",
        "--", "--", "--"),
    Season("08/09"),
    Run("13/06/2009", 'HV / Turf / "A"', "811", results_href("2009/06/13", "HV", 2), "WV",
        "1000", "G", "GRIFFIN", "--", "--", "Q Retired", "--", "--", "--", "118", "--",
        "--", "--", "--"),
]  # fmt: skip
RETIRED_RUNS = [
    run_row(RETIRED_ID, date(2012, 2, 19), race_index=411, venue="ST", surface="turf",
            course_config="A", distance_m=1400, going="G", race_class="4", draw=12, rating=64,
            trainer_name="Q Retired", jockey_name="J Rider", lbw_raw="1-1/2", win_odds=35.0,
            actual_wt_lb=130, running_positions=[11, 11, 11, 4],
            finish_time_s=pytest.approx(83.18), declared_horse_wt_lb=1079, gear_raw="B/H",
            placing_raw="4", race_id="hkjc:2012-02-19:ST:05"),
    run_row(RETIRED_ID, date(2011, 4, 20), race_index=602, venue="HV", surface="turf",
            course_config="C", distance_m=1650, going="G", race_class="3", draw=8, rating=68,
            trainer_name="Q Retired", jockey_name="K Pilot", lbw_raw="6-1/4", win_odds=41.0,
            actual_wt_lb=121, running_positions=[8, 9, 11, 12],
            finish_time_s=pytest.approx(102.99), declared_horse_wt_lb=1081, gear_raw="B/H",
            placing_raw="12", race_id="hkjc:2011-04-20:HV:06"),
    run_row(RETIRED_ID, date(2010, 11, 6), race_index=155, venue="ST", surface="turf",
            course_config="C+3", distance_m=1200, going="GY", race_class="3", draw=6, rating=72,
            trainer_name="Q Retired", jockey_name="J Rider", lbw_raw="SH", win_odds=6.4,
            actual_wt_lb=126, running_positions=[3, 3, 2], finish_time_s=pytest.approx(70.31),
            declared_horse_wt_lb=1090, gear_raw="B1/H", placing_raw="2",
            race_id="hkjc:2010-11-06:ST:04"),
    run_row(RETIRED_ID, date(2010, 1, 1), race_index=275, venue="ST", surface="awt",
            course_config=None, distance_m=1200, going="WS", race_class="5", draw=9, rating=54,
            trainer_name="Q Retired", jockey_name="D Sample", lbw_raw="1-3/4", win_odds=2.0,
            actual_wt_lb=133, running_positions=[1, 1, 1], finish_time_s=pytest.approx(69.40),
            declared_horse_wt_lb=1070, gear_raw="H", placing_raw="1",
            race_id="hkjc:2010-01-01:ST:01"),
    run_row(RETIRED_ID, date(2009, 11, 29), race_index=190, venue="ST", surface="turf",
            course_config="B", distance_m=1000, going="G", race_class="5", draw=None, rating=52,
            trainer_name="Q Retired", jockey_name=None, lbw_raw=None, win_odds=None,
            actual_wt_lb=126, running_positions=[], finish_time_s=None,
            declared_horse_wt_lb=None, gear_raw=None, placing_raw="WV",
            race_id="hkjc:2009-11-29:ST:08"),
    # a griffin withdrawn before its first run: no rating and no jockey printed
    run_row(RETIRED_ID, date(2009, 6, 13), race_index=811, venue="HV", surface="turf",
            course_config="A", distance_m=1000, going="G", race_class="GRIFFIN", draw=None,
            rating=None, trainer_name="Q Retired", jockey_name=None, lbw_raw=None,
            win_odds=None, actual_wt_lb=118, running_positions=[], finish_time_s=None,
            declared_horse_wt_lb=None, gear_raw=None, placing_raw="WV",
            race_id="hkjc:2009-06-13:HV:02"),
]  # fmt: skip

LAYOUTS = {
    "active": (ACTIVE, ACTIVE_ROWS, ACTIVE_HORSE, ACTIVE_RUNS),
    "retired": (RETIRED, RETIRED_ROWS, RETIRED_HORSE, RETIRED_RUNS),
}
# a second set of as-of-scrape values: the static attributes and the runs must not move with them
OTHER_AS_OF_SCRAPE = {
    "age": "9",
    "season_stakes": "$0",
    "total_stakes": "$12,000,500",
    "record": "9-1-0-40",
    "recent_starts": "0",
    "location": "Conghua",
    "arrival": "(02/02/2026)",
    "import_date": "01/02/2020",
    "trainer": "W Different",
    "trainer_code": "XWD",
    "owner": "Another Holder",
    "current_rating": "88",
    "season_start_rating": "79",
    "last_rating": "91",
}


def parse_layout(name: str, profile: Profile | None = None, **kw) -> dict:
    base, rows, _, _ = LAYOUTS[name]
    profile = profile or base
    url = horse_url(profile.horse_id, retired=profile.retired)
    return parse_horse.parse(raw(pages.horse_page(profile, rows), profile.horse_id, url, **kw))


def test_schema_columns_match_the_contract():
    assert set(schemas.TABLES["horses"].columns) == HORSE_COLUMNS
    assert set(schemas.TABLES["horse_runs"].columns) == RUN_COLUMNS


# --- horses ------------------------------------------------------------------------------------


def test_active_horse_static_attributes():
    assert parse_layout("active")["horses"] == [ACTIVE_HORSE]


def test_retired_horse_static_attributes():
    assert parse_layout("retired")["horses"] == [RETIRED_HORSE]


@pytest.mark.parametrize("variant", ["printed", "other"])
@pytest.mark.parametrize("layout", ["active", "retired"])
def test_no_as_of_scrape_values_leak_into_horses(layout, variant):
    base, _, expected_horse, expected_runs = LAYOUTS[layout]
    profile = base if variant == "printed" else dataclasses.replace(base, **OTHER_AS_OF_SCRAPE)
    parsed = parse_layout(layout, profile)

    (horse,) = parsed["horses"]
    assert set(horse) == HORSE_COLUMNS
    assert not set(horse) & FORBIDDEN
    # the values themselves are not smuggled in under another name either
    printed = {getattr(profile, f) for f in AS_OF_SCRAPE_FIELDS}
    values = {str(v) for v in horse.values()}
    assert not values & printed
    long_printed = [p for p in printed if len(p) >= 4]
    assert not [(v, p) for v in values for p in long_printed if p in v]
    # the static attributes do not move with the as-of-scrape values ...
    assert horse == expected_horse
    # ... and neither does history: rating-at-run and trainer-at-run come from the rows alone
    assert parsed["horse_runs"] == expected_runs
    profile_ratings = {to_int(profile.current_rating), to_int(profile.last_rating)}
    run_ratings = {r["rating"] for r in parsed["horse_runs"]}
    if variant == "other":
        assert not run_ratings & profile_ratings
        assert profile.trainer not in {r["trainer_name"] for r in parsed["horse_runs"]}


def to_int(text: str) -> int:
    return int(text.replace(",", ""))


@pytest.mark.parametrize(
    ("layout", "profile_changes", "url_retired", "meta", "expected"),
    [
        # the fetcher recorded the requested /horse URL, not the /otherhorse redirect target
        ("retired", {}, False, {}, True),
        # no "(Retired)" in the title: Last Rating without Current Rating is the retired layout
        ("retired", {"title": "SALT MEADOW (CY277)"}, False, {}, True),
        # the title says so on an otherwise active layout
        ("active", {"title": "QUIET ORBIT (X901) (Retired)"}, False, {}, True),
        # the redirect target the fetcher kept
        ("active", {}, False, {"final_url": horse_url(ACTIVE_ID, retired=True)}, True),
        ("active", {}, True, {}, True),
        ("active", {}, False, {}, False),
        ("active", {}, False, {"final_url": horse_url(ACTIVE_ID)}, False),
    ],
    ids=[
        "retired-page-horse-url",
        "last-rating-only",
        "retired-title",
        "final-url-otherhorse",
        "otherhorse-url",
        "active",
        "active-final-url",
    ],
)
def test_retired_layout_is_detected_from_the_page(
    layout, profile_changes, url_retired, meta, expected
):
    base, rows, expected_horse, _ = LAYOUTS[layout]
    profile = dataclasses.replace(base, **profile_changes)
    url = horse_url(profile.horse_id, retired=url_retired)
    parsed = parse_horse.parse(
        raw(pages.horse_page(profile, rows), profile.horse_id, url, meta=meta)
    )
    (horse,) = parsed["horses"]
    assert horse["is_retired"] is expected
    assert horse["name"] == expected_horse["name"]  # "(Retired)" and the brand never reach the name


@pytest.mark.parametrize(
    ("key", "url", "meta", "expected"),
    [
        ("", horse_url(ACTIVE_ID), {}, ACTIVE_ID),
        ("not a horse id", horse_url("HK_2021_X314"), {}, "HK_2021_X314"),
        ("../../etc/passwd", horse_url("HK_2019_X222", retired=True), {}, "HK_2019_X222"),
        ("", f"{pages.HOST}/en-us/local/information/horse?HorseId=HK_2017_X050&Option=1", {},
         "HK_2017_X050"),
        # the requested URL names no horse: the redirect target the fetcher kept does
        ("", f"{pages.HOST}/en-us/local/information/horse",
         {"final_url": horse_url("HK_2020_X808")}, "HK_2020_X808"),
        # a usable cache key wins over the URL
        ("HK_2022_X606", horse_url(ACTIVE_ID), {}, "HK_2022_X606"),
    ],
    ids=["empty-key", "junk-key", "unsafe-key", "param-case", "final-url", "key-wins"],
)  # fmt: skip
def test_horse_id_falls_back_to_the_url(key, url, meta, expected):
    parsed = parse_horse.parse(raw(pages.horse_page(ACTIVE, ACTIVE_ROWS), key, url, meta=meta))
    assert [h["horse_id"] for h in parsed["horses"]] == [expected]
    assert len(parsed["horse_runs"]) == len(ACTIVE_RUNS)
    assert {r["horse_id"] for r in parsed["horse_runs"]} == {expected}


def test_horse_id_falls_back_to_the_page_links_when_key_and_url_are_unusable():
    # the page links mostly to its own horse; one link to another horse is a decoy
    parsed = parse_horse.parse(
        raw(pages.horse_page(ACTIVE, ACTIVE_ROWS), "not a horse id", "https://example.invalid/")
    )
    assert [h["horse_id"] for h in parsed["horses"]] == [ACTIVE_ID]
    assert {r["horse_id"] for r in parsed["horse_runs"]} == {ACTIVE_ID}
    assert len(parsed["horse_runs"]) == len(ACTIVE_RUNS)


def test_unsafe_ids_everywhere_give_empty_tables():
    # an id is a join key, a URL parameter and a cache file name: nothing outside a plain charset
    profile = (
        '<table class="horseProfile"><tr><td><span class="title_text">NO LINKS (X404)</span>'
        "</td></tr></table>"
    )
    parsed = parse_horse.parse(
        raw(pages.page(profile), "../../etc/passwd", "https://example.invalid/?horseid=a%2Fb")
    )
    assert parsed == {"horses": [], "horse_runs": []}


@pytest.mark.parametrize("fetched", [None, datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)])
def test_scrape_time_is_the_fetch_time_and_may_be_missing(fetched):
    (horse,) = parse_layout("retired", fetched=fetched)["horses"]
    assert horse["profile_scraped_utc"] == fetched


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"origin": "IRE", "age": "5", "colour_sex": "Bay / Brown / Horse"},
         {"origin": "IRE", "colour": "Bay / Brown", "sex": "Horse"}),
        ({"origin": "", "age": "5"}, {"origin": None}),  # " / 5": the age is never an origin
        ({"colour_sex": "Grey / Mare"}, {"colour": "Grey", "sex": "Mare"}),
        ({"colour_sex": "Gelding"}, {"colour": None, "sex": "Gelding"}),
        ({"colour_sex": "Chestnut"}, {"colour": "Chestnut", "sex": None}),
        ({"import_type": "--", "dam_sire": "--"}, {"import_type": None, "dam_sire": None}),
        ({"import_type": "ISG", "sire": "Lamp Post", "dam": "Iron Gate"},
         {"import_type": "ISG", "sire": "Lamp Post", "dam": "Iron Gate"}),
        ({"title": "NEW ARRIVAL (X001)"}, {"name": "NEW ARRIVAL"}),
        ({"retired": True, "origin": "GB", "title": "OLD TIMER (CX123) (Retired)"},
         {"origin": "GB", "name": "OLD TIMER", "is_retired": True}),
    ],
    ids=["two-part-colour", "blank-origin", "grey-mare", "sex-only", "colour-only", "blanks",
         "pedigree", "name", "retired-origin"],
)  # fmt: skip
def test_profile_variants(changes, expected):
    profile = dataclasses.replace(ACTIVE, **changes)
    url = horse_url(ACTIVE_ID, retired=profile.retired)
    (horse,) = parse_horse.parse(raw(pages.horse_page(profile, ()), ACTIVE_ID, url))["horses"]
    assert horse == {**ACTIVE_HORSE, **expected}


@pytest.mark.parametrize("rows", [(), None], ids=["empty-form-table", "no-form-table"])
def test_unraced_horse_has_a_profile_and_no_runs(rows):
    profile = dataclasses.replace(ACTIVE, horse_id="HK_2026_X001", title="NEW ARRIVAL (X001)")
    parsed = parse_horse.parse(
        raw(pages.horse_page(profile, rows), "HK_2026_X001", horse_url("HK_2026_X001"))
    )
    assert parsed["horse_runs"] == []
    assert [(h["horse_id"], h["name"]) for h in parsed["horses"]] == [
        ("HK_2026_X001", "NEW ARRIVAL")
    ]


# --- horse_runs --------------------------------------------------------------------------------


def test_active_runs_parse_from_the_form_table():
    # season separators dropped, the withdrawal kept, page order (newest first) kept
    assert parse_layout("active")["horse_runs"] == ACTIVE_RUNS


def test_retired_runs_parse_from_the_form_table():
    # DD/MM/YYYY dates, htable_text cells, whitespace-padded cells, two withdrawals
    assert parse_layout("retired")["horse_runs"] == RETIRED_RUNS


def test_runs_follow_the_page_not_a_fixed_answer():
    # the same rows under another horse, with a run dropped: ids and rows follow the page
    profile = dataclasses.replace(ACTIVE, horse_id="HK_2024_X432", title="OTHER ONE (X432)")
    rows = [r for r in ACTIVE_ROWS if not (isinstance(r, Run) and r.date == "28/01/26")]
    parsed = parse_horse.parse(
        raw(pages.horse_page(profile, rows), "HK_2024_X432", horse_url("HK_2024_X432"))
    )
    expected = [
        {**r, "horse_id": "HK_2024_X432"}
        for r in ACTIVE_RUNS
        if r["race_date"] != date(2026, 1, 28)
    ]
    assert parsed["horse_runs"] == expected


LOCAL = Run("28/04/24", 'ST / Turf / "A"', "611", results_href("2024/04/28", "ST", 8), "02",
            "2000", "GF", "G1", "5", "120", "A See", "J Rider", "SH", "2.4", "126", "3 3 2 2",
            "2.00.51", "1130", "V-/B2/TT")  # fmt: skip
LOCAL_RUN = run_row(ACTIVE_ID, date(2024, 4, 28), race_index=611, venue="ST", surface="turf",
                    course_config="A", distance_m=2000, going="GF", race_class="G1", draw=5,
                    rating=120, trainer_name="A See", jockey_name="J Rider", lbw_raw="SH",
                    win_odds=2.4, actual_wt_lb=126, running_positions=[3, 3, 2, 2],
                    finish_time_s=pytest.approx(120.51), declared_horse_wt_lb=1130,
                    gear_raw="V-/B2/TT", placing_raw="2",
                    race_id="hkjc:2024-04-28:ST:08")  # fmt: skip
OVERSEAS_HREF = (
    "/en-us/local/information/overseasresults?racedate=2023/12/10&Racecourse=S1&RaceNo=4"
)


def runs_of(*rows: Run | Season) -> list[dict]:
    body = pages.horse_page(ACTIVE, rows)
    return parse_horse.parse(raw(body, ACTIVE_ID, horse_url(ACTIVE_ID)))["horse_runs"]


def test_overseas_rows_are_skipped_and_local_rows_kept():
    # a run is local only when the first token of RC/Track/Course is a configured venue (ST, HV)
    overseas = Run("30/03/24", "MEY / Turf", "", None, "03", "1800", rating="--")
    overseas_linked = Run("10/12/23", "TOK / Turf", "Overseas", OVERSEAS_HREF, "01", "2400")
    assert runs_of(Season("23/24"), LOCAL, overseas, overseas_linked) == [LOCAL_RUN]


def test_rows_labelled_overseas_are_dropped_even_when_the_course_cell_starts_with_a_local_code():
    text_label = Run("02/03/24", 'ST / Turf / "A"', "Overseas", None, "04")
    linked = Run("10/12/23", "HV / Turf", "004", OVERSEAS_HREF.replace("&Racecourse=S1", ""), "01")
    assert runs_of(LOCAL, text_label, linked) == [LOCAL_RUN]


def test_columns_are_mapped_by_header_text_not_position():
    # an older layout: fewer columns, another order, no Rtg. or Gear column at all
    header = ["Date", "Race<br/>Index", "Dist.", "RC/Track/<br/>Course", "Pla.", "Jockey"]
    body = [
        ["07/03/04", "441", "1200", 'ST / Turf / "C"', "05", "D Sample"],
        ["14/02/04", "388", "1650", 'HV / Turf / "B+2"', "11", "E Sample"],
    ]
    page = pages.page(ACTIVE.render() + pages.raw_form(header, body))
    runs = parse_horse.parse(raw(page, "HK_2003_X001", horse_url("HK_2003_X001")))["horse_runs"]
    blanks = dict(surface="turf", going=None, race_class=None, draw=None, rating=None,
                  trainer_name=None, lbw_raw=None, win_odds=None, actual_wt_lb=None,
                  running_positions=[], finish_time_s=None, declared_horse_wt_lb=None,
                  gear_raw=None, race_id=None)  # fmt: skip
    assert runs == [
        run_row("HK_2003_X001", date(2004, 3, 7), race_index=441, venue="ST", course_config="C",
                distance_m=1200, jockey_name="D Sample", placing_raw="5", **blanks),
        run_row("HK_2003_X001", date(2004, 2, 14), race_index=388, venue="HV",
                course_config="B+2", distance_m=1650, jockey_name="E Sample", placing_raw="11",
                **blanks),
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("printed", "expected"),
    [("01 DH", "1 DH"), ("10", "10"), ("03", "3"), ("WV", "WV"), ("PU", "PU"), ("UR", "UR")],
)
def test_placing_keeps_its_code_and_loses_only_the_zero_padding(printed, expected):
    (run,) = runs_of(dataclasses.replace(LOCAL, placing=printed))
    assert run["placing_raw"] == expected


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        (results_href("2024/04/28", "ST", 8), "hkjc:2024-04-28:ST:08"),
        (results_href("2024/04/28", "ST", 11), "hkjc:2024-04-28:ST:11"),
        (results_href("2024/04/27", "ST", 8), None),  # another day
        (results_href("2024/04/28", "HV", 8), None),  # another course
        (None, None),  # no link at all
    ],
    ids=["agrees", "other-race-no", "other-day", "other-course", "no-link"],
)
def test_race_id_needs_a_results_link_that_agrees_with_the_row(href, expected):
    (run,) = runs_of(dataclasses.replace(LOCAL, href=href))
    assert (run["race_date"], run["venue"], run["race_index"]) == (date(2024, 4, 28), "ST", 611)
    assert run["race_id"] == expected


@pytest.mark.parametrize(
    ("printed", "href_date", "expected"),
    [
        ("29/11/09", "2009/11/29", date(2009, 11, 29)),  # a 2009 run printed with a 2-digit year
        ("16/09/26", "2026/09/16", date(2026, 9, 16)),
        ("29/11/09", None, date(2009, 11, 29)),  # no link: the pivot lands in the right century
        ("05/06/99", None, date(1999, 6, 5)),
        ("05/06/68", "1968/06/05", date(1968, 6, 5)),  # the link settles what the pivot would not
        ("29/11/2009", "2026/09/16", date(2009, 11, 29)),  # four digits are never overridden
    ],
)
def test_two_digit_years_in_form_rows(printed, href_date, expected):
    href = results_href(href_date, "ST", 8) if href_date else None
    (run,) = runs_of(dataclasses.replace(LOCAL, date=printed, href=href))
    assert run["race_date"] == expected


@pytest.mark.parametrize(
    ("course", "expected"),
    [
        ('HV / Turf / "C+3"', ("HV", "turf", "C+3")),
        ('ST / Turf / "A+2"', ("ST", "turf", "A+2")),
        ("ST / AWT", ("ST", "awt", None)),
        ("ST / All Weather Track", ("ST", "awt", None)),
        ("ST / Turf / “B”", ("ST", "turf", "B")),
        ('ST / Turf / "C + 3"', ("ST", "turf", "C+3")),
        ("HV / Turf", ("HV", "turf", None)),
    ],
)
def test_course_cell_gives_venue_surface_and_config(course, expected):
    (run,) = runs_of(dataclasses.replace(LOCAL, course=course))
    assert (run["venue"], run["surface"], run["course_config"]) == expected


def test_running_position_tokens_that_are_not_plain_numbers_are_dropped_without_raising():
    # "²".isdigit() is True but int("²") raises: one odd character must not lose the page
    (run,) = runs_of(dataclasses.replace(LOCAL, running="3 ² - 2 ٣x"))
    assert run["running_positions"] == [3, 2]
    assert run["gear_raw"] == "V-/B2/TT"


def test_a_merged_data_cell_does_not_shift_the_later_columns():
    # LBW + Win Odds printed as one colspan=2 cell: every later cell stays under its own header
    cells = LOCAL.cells(ACTIVE_ID, retired=False)
    tds = "".join(
        '<td class="htable_eng_text" colspan="2">--</td>'
        if i == 11
        else f'<td class="htable_eng_text">{c}</td>'
        for i, c in enumerate(cells)
        if i != 12
    )
    head = "".join(f'<td class="hsubheader">{h}</td>' for h in pages.HEADER)
    form = f'<table class="bigborder"><tr>{head}</tr><tr>{tds}</tr></table>'
    body = pages.page(ACTIVE.render() + form)
    (run,) = parse_horse.parse(raw(body, ACTIVE_ID, horse_url(ACTIVE_ID)))["horse_runs"]
    assert run == {**LOCAL_RUN, "lbw_raw": None, "win_odds": None}


def test_absurd_header_colspan_is_clamped():
    head = '<td class="hsubheader" colspan="2000000000">Race Index</td>' + "".join(
        f'<td class="hsubheader">{h}</td>' for h in pages.HEADER[1:]
    )
    row = "".join(f'<td class="htable_eng_text">{c}</td>' for c in LOCAL.cells(ACTIVE_ID, False))
    form = f'<table class="bigborder"><tr>{head}</tr><tr>{row}</tr></table>'
    parsed = parse_horse.parse(
        raw(pages.page(ACTIVE.render() + form), ACTIVE_ID, horse_url(ACTIVE_ID))
    )
    assert parsed["horses"] == [ACTIVE_HORSE]  # and it returns at all


# --- pages that are not a horse profile ----------------------------------------------------------


def _sectional_page() -> bytes:
    S = hkjc_sectional_pages.Section
    runner = hkjc_sectional_pages.Runner(
        "1", 1, "QUIET ORBIT", ACTIVE_ID, [S("1", "1/2", "12.80"), S("1", "1", "22.10")], "0:57.54"
    )
    race = hkjc_sectional_pages.Race(1, [runner])
    return hkjc_sectional_pages.sectional_page(date(2026, 9, 16), "Happy Valley", [race])


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"", id="empty-body"),
        pytest.param(b"<html><body><p>Service unavailable</p></body></html>", id="unrelated"),
        pytest.param(
            hkjc_pages.page('<div class="localResults commContent"><p>No information.</p></div>'),
            id="no-information",
        ),
        pytest.param(_sectional_page(), id="sectional-page"),
        pytest.param(hkjc_pages.calendar(2008, 1, [hkjc_pages.meeting(1, "ST", "D", "Turf")]),
                     id="fixture-page"),
    ],
)  # fmt: skip
def test_pages_without_a_profile_give_empty_tables(body):
    assert parse_horse.parse(raw(body, ACTIVE_ID, horse_url(ACTIVE_ID))) == {
        "horses": [],
        "horse_runs": [],
    }


# --- dates ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("19/02/2012", date(2012, 2, 19)),  # retired pages: 4-digit year
        ("16/09/26", date(2026, 9, 16)),  # active pages: 2-digit year
        ("01/01/00", date(2000, 1, 1)),
        ("05/03/99", date(1999, 3, 5)),  # never 2099
        ("31/12/79", date(1979, 12, 31)),
        ("31/12/68", date(2068, 12, 31)),  # the POSIX pivot: 69-99 -> 19xx, 00-68 -> 20xx
        ("01/01/69", date(1969, 1, 1)),
        (" 7/3/04 ", date(2004, 3, 7)),
    ],
)
def test_parse_run_date_handles_both_eras_and_centuries(text, expected):
    assert parse_horse.parse_run_date(text) == expected


def test_parse_run_date_prefers_the_results_link_year_when_it_agrees():
    # the Race Index link carries racedate=YYYY/MM/DD: it settles the century when d/m/yy match
    assert parse_horse.parse_run_date("05/03/99", hint=date(1999, 3, 5)) == date(1999, 3, 5)
    assert parse_horse.parse_run_date("05/03/68", hint=date(1968, 3, 5)) == date(1968, 3, 5)
    assert parse_horse.parse_run_date("05/03/68") == date(2068, 3, 5)
    # a link for a different day is no evidence about this row
    assert parse_horse.parse_run_date("05/03/99", hint=date(2099, 3, 6)) == date(1999, 3, 5)
    assert parse_horse.parse_run_date("05/03/1999", hint=date(2099, 3, 5)) == date(1999, 3, 5)


@pytest.mark.parametrize(
    "text", ["", "--", "26/27 Season", "31/02/2024", "2024-03-20", "00/00/00", None]
)
def test_parse_run_date_rejects_non_dates(text):
    assert parse_horse.parse_run_date(text) is None


def test_parser_version_is_declared():
    assert isinstance(parse_horse.PARSER_VERSION, str) and parse_horse.PARSER_VERSION
