"""netkeiba race list of a day -> `race_links` (ticket #51, ported from HKTrack's
`sources/jra/parse_racelist.py` and `urls.py`).

Contract pinned here (the coder follows it):
- `horsetrack.countries.japan.jra.netkeiba.parse_racelist.parse(page: RawPage) -> ParsedPage`,
  pure, no network.
- `race_links`: one row `{"race_id": str, "race_date": date}` per distinct JRA race linked from a
  `dl.race_top_data_info` inside `div.race_kaisai_info` (link path exactly `/race/<id>/`).
  A JRA race id is 12 ASCII digits whose venue part (characters 5-6) is 01..10; local (NAR),
  overseas and malformed ids are dropped, since an id becomes a URL and a cache file name.
- The day is the page key (YYYY-MM-DD); when the key is not a date, the `年月日` date of the h2
  (or the title); neither: ValueError. `_meta` carries page_type, key, url and race_date.
- `horsetrack.countries.japan.jra.netkeiba.urls`: `jra_race_id(value) -> str | None`,
  `calendar(year, month)`, `racelist(day)`, `results(race_id)`, `pedigree_card(race_id)` on the
  hosts of `countries/japan/config/jra.yaml`; the last two raise ValueError on a non-JRA id.

Pages are synthetic (netkeiba_pages.py); expected ids are written out here.
"""

from __future__ import annotations

from datetime import date

import netkeiba_pages as pages
import pytest

from horsetrack.core.ingest.types import RawPage

# isort: split
# (its own block: ruff ranks a module that does not exist yet as third-party)
from horsetrack.countries.japan.jra.netkeiba import parse_racelist, urls

DAY = date(2025, 11, 23)
TOKYO = [f"2025050505{r:02d}" for r in range(1, 13)]
KYOTO = [f"2025080406{r:02d}" for r in range(1, 13)]
NAR = ["202546112301", "202546112302", "202554112301", "202565112312"]
OVERSEAS = ["2025Q9a00301", "2025Z2a01c04", "2025X5a01d02"]

KEPT = ["202501010101", "202510020312", "202506050708", "202509030411", "202502010201"]
DROPPED = [
    "202500010101",  # venue 00
    "202511010101",  # venue 11
    "202530112301",  # local racing (NAR) venue
    "202599010101",  # venue 99
    "20250801011",  # 11 digits
    "2025080101011",  # 13 digits
    "2025O8010101",  # letter O for a zero
    "２０２５０８０１０１０２",  # full-width digits
    "202508010103x",  # trailing junk: not 202508010103
    "2025QQa00b03",  # overseas
    "",
]


def raw(body: bytes, key: str, day: date = DAY) -> RawPage:
    return RawPage("jra", "racelist", key, f"{pages.DB}/race/list/{day:%Y%m%d}/", body)


def links(body: bytes, key: str, day: date = DAY) -> list[dict]:
    return parse_racelist.parse(raw(body, key, day))["race_links"]


@pytest.mark.parametrize(
    ("day", "venues", "expected"),
    [
        (
            DAY,
            [
                pages.jra("東京", TOKYO),
                pages.jra("京都", KYOTO),
                pages.nar("大井", NAR[:2]),
                pages.nar("名古屋", NAR[2:]),
                pages.overseas("香港", OVERSEAS),
            ],
            TOKYO + KYOTO,
        ),
        (
            date(2010, 1, 5),
            [pages.jra("中山", [f"2010060101{r:02d}" for r in range(1, 13)])],
            [f"2010060101{r:02d}" for r in range(1, 13)],
        ),
        (
            date(2019, 8, 4),
            [
                pages.jra("札幌", ["201901020601", "201901020611"]),
                pages.nar("門別", ["201930080401"]),
                pages.jra("小倉", ["201910020612"]),
            ],
            ["201901020601", "201901020611", "201910020612"],
        ),
    ],
    ids=["2025-11-23", "2010-01-05", "2019-08-04"],
)
def test_race_list_gives_the_jra_race_ids_of_the_day(day, venues, expected):
    key = day.isoformat()
    parsed = parse_racelist.parse(raw(pages.race_list(day, venues), key, day))

    rows = parsed["race_links"]
    assert sorted(row["race_id"] for row in rows) == sorted(expected)
    assert len(rows) == len(expected)
    assert all(row["race_date"] == day for row in rows)
    (meta,) = parsed["_meta"]
    assert (meta["page_type"], meta["key"], meta["race_date"]) == ("racelist", key, day)


def test_nar_and_malformed_ids_are_dropped():
    # the shape of a real day: JRA races next to local (NAR) and overseas races
    venues = [
        pages.nar("大井", NAR),
        pages.jra("東京", TOKYO[:3]),
        pages.overseas("凱旋門", OVERSEAS),
    ]
    assert {r["race_id"] for r in links(pages.race_list(DAY, venues), "2025-11-23")} == set(
        TOKYO[:3]
    )

    # crafted ids: only 12 ASCII digits with a venue part of 01..10 get through
    venues = [pages.jra("京都", KEPT + DROPPED)]
    rows = links(pages.race_list(DAY, venues), "2025-11-23")
    assert sorted(r["race_id"] for r in rows) == sorted(KEPT)
    for value in KEPT:
        assert urls.jra_race_id(value) == value
    for value in DROPPED + [None, 202501010101, b"202501010101"]:
        assert urls.jra_race_id(value) is None, value

    # every venue number of jra.yaml passes; its neighbours do not
    for venue in range(0, 100):
        race_id = f"2024{venue:02d}030812"
        assert (urls.jra_race_id(race_id) == race_id) is (1 <= venue <= 10), venue

    # a day of local and overseas races only yields nothing
    only_other = [pages.nar("大井", NAR), pages.overseas("香港", OVERSEAS)]
    assert links(pages.race_list(DAY, only_other), "2025-11-23") == []


def test_only_race_links_of_the_list_count():
    hrefs = [
        pages.race_href("202508040601"),
        "/race/202508040602/",  # relative
        f"{pages.DB}/race/202508040603",  # no trailing slash
        pages.race_href("202508040601"),  # listed twice: one row
        f"{pages.DB}/race/202508040604/movie/",  # another page of the race
        f"{pages.DB}/race/list/20251123/",
        f"{pages.DB}/horse/2020100001/",
        f"{pages.DB}/race/sum/08/20251123/",
    ]
    body = pages.race_list(DAY, [pages.Venue("中央", "京都", hrefs)])
    # decoys: the venue heading and a pickup box outside the list link JRA races too
    assert b"/race/202508990199/" in body and b"/race/202505990101/" in body
    rows = links(body, "2025-11-23")
    assert sorted(r["race_id"] for r in rows) == ["202508040601", "202508040602", "202508040603"]


@pytest.mark.parametrize(
    ("day", "key"),
    [(date(2025, 11, 23), "20251123"), (date(2009, 2, 1), "list-2009-02"), (date(2030, 12, 7), "")],
)
def test_without_a_date_key_the_heading_names_the_day(day, key):
    body = pages.race_list(day, [pages.jra("阪神", ["203009050801"])])
    rows = links(body, key, day)
    assert rows == [{"race_id": "203009050801", "race_date": day}]

    # the key wins over the heading when it is a date
    rows = links(body, "2031-01-04", day)
    assert [r["race_date"] for r in rows] == [date(2031, 1, 4)]

    with pytest.raises(ValueError):
        links(pages.race_list(day, [pages.jra("阪神", ["203009050801"])], heading=False), key, day)


def test_urls_are_built_only_for_jra_ids():
    assert (
        urls.calendar(2025, 11) == "https://race.netkeiba.com/top/calendar.html?year=2025&month=11"
    )
    assert urls.calendar(2010, 1) == "https://race.netkeiba.com/top/calendar.html?year=2010&month=1"
    assert urls.racelist(date(2025, 11, 23)) == "https://db.netkeiba.com/race/list/20251123/"
    assert urls.racelist(date(2009, 2, 1)) == "https://db.netkeiba.com/race/list/20090201/"
    for race_id in ("202508040611", "201010010112"):
        assert urls.results(race_id) == f"https://db.netkeiba.com/race/{race_id}/"
        assert urls.pedigree_card(race_id) == (
            f"https://race.netkeiba.com/race/shutuba_past.html?race_id={race_id}"
        )
    for bad in ("202546112301", "2025Q9a00301", "202508010103x", "../../etc/passwd"):
        with pytest.raises(ValueError):
            urls.results(bad)
        with pytest.raises(ValueError):
            urls.pedigree_card(bad)
