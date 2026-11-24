"""HKJC per-horse sectional times page -> `sectionals` rows (ticket #61, ported from HKTrack's
`sources/hkjc/parse_sectional.py`).

Contract pinned here (the coder follows it):
- `horsetrack.countries.hongkong.hkjc.parse_sectional.parse(page: RawPage) -> ParsedPage`
  (`horsetrack.core.ingest.types`), pure, no network, returning `{"sectionals": [...]}`.
- One row per (runner, section) whose cell holds a position or a time, with exactly the columns
  of `schemas.TABLES["sectionals"]`: jurisdiction "hkjc", race_id `hkjc:<YYYY-MM-DD>:<VENUE>:<NN>`,
  horse_id from the profile link (None when unlinked or outside a plain charset), horse_no,
  section_no (1-based column), section_pos, section_margin_lengths, section_time_s, plus the
  documented extras section_margin_raw (the printed margin) and sub_split_times_s (the 2026-era
  200 m splits, [] when none). Unused section cells (a blank image) give no row.
- Position and margin are read from their own elements (`span.f_fl`, `i`). The leader's cell
  prints its lead over the second horse: lengths 0.0, the printed value kept in the raw column.
  A printed "-" or no `<i>` is no margin (None, None); an unknown code keeps its raw text.
- The race is the cache key's, then the URL's (racedate DD/MM/YYYY, RaceNo), then the page's own
  meeting line and `div#Race<n>`. A page naming another date or venue than the one asked for, or
  without the asked-for race's block, gives no rows.

Pages are synthetic (hkjc_sectional_pages.py); expected rows are derived here from the cells.
"""

from __future__ import annotations

import tracemalloc
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import hkjc_pages
import hkjc_sectional_pages as pages
import pytest
from hkjc_sectional_pages import BLANK, Race, Runner, Section

from horsetrack.core.ingest.types import RawPage
from horsetrack.core.store import schemas

# isort: split
# (its own block: ruff ranks a module that does not exist yet as third-party)
from horsetrack.countries.hongkong.hkjc import parse_sectional

# docs/specs/store.md, sectionals: the key and measurement columns plus `jurisdiction` ...
SPEC_COLUMNS = {
    "jurisdiction",
    "race_id",
    "horse_id",
    "horse_no",
    "section_no",
    "section_pos",
    "section_margin_lengths",
    "section_time_s",
}
# ... and the two documented extras: the printed margin and the 200 m sub-splits
EXTRA_COLUMNS = {"section_margin_raw", "sub_split_times_s"}

# printed margin -> lengths, worked out by hand (HKJC: N is a neck, NOSE a nose)
LENGTHS = {
    "NOSE": 0.05,
    "SH": 0.1,
    "HD": 0.2,
    "N": 0.3,
    "1/2": 0.5,
    "3/4": 0.75,
    "1": 1.0,
    "1-1/4": 1.25,
    "1-1/2": 1.5,
    "1-3/4": 1.75,
    "2": 2.0,
    "2-1/4": 2.25,
    "2-1/2": 2.5,
    "2-3/4": 2.75,
    "3": 3.0,
    "3-1/4": 3.25,
    "4": 4.0,
    "5": 5.0,
    "6": 6.0,
    "6-1/2": 6.5,
    "10-1/2": 10.5,
    "12": 12.0,
}


def sec(pos: int, margin: str | None, time: str, *subs: str) -> Section:
    return Section(str(pos), margin, time, subs)


@dataclass(frozen=True)
class Meeting:
    """One synthetic single-race page and the race it is about."""

    race_date: date
    venue: str
    venue_name: str
    race: Race
    finish_s: dict[int, float]  # horse_no -> finish time, as printed in the Time column

    @property
    def key(self) -> str:
        return f"{self.race_date.isoformat()}_{self.venue}_R{self.race.race_no:02d}"

    @property
    def race_id(self) -> str:
        return f"hkjc:{self.race_date.isoformat()}:{self.venue}:{self.race.race_no:02d}"

    @property
    def body(self) -> bytes:
        return pages.sectional_page(self.race_date, self.venue_name, [self.race])

    def raw(self, key: str | None = None, url: str | None = None) -> RawPage:
        url = pages.sectional_url(self.race_date, self.race.race_no) if url is None else url
        return RawPage("hkjc", "sectional", self.key if key is None else key, url, self.body)


# 2026 layout: 1000 m in 3 sections, sub-splits after the opening section, a withdrawn runner
HV_2026 = Meeting(
    date(2026, 9, 16),
    "HV",
    "Happy Valley",
    Race(
        1,
        [
            Runner("1", 11, "PAPER LANTERN", "HK_2023_A101", [
                sec(1, "3/4", "12.80"), sec(1, "1", "21.95", "10.79", "11.16"),
                sec(1, "1-3/4", "22.79", "11.19", "11.60")], "0:57.54"),
            Runner("2", 12, "QUIET HARBOUR", "HK_2023_A102", [
                sec(4, "2-3/4", "13.24"), sec(3, "2-1/2", "21.91", "10.71", "11.20"),
                sec(2, "1-3/4", "22.69", "11.19", "11.50")], "0:57.84"),
            Runner("3", 6, "COPPER KITE", "HK_2024_B203", [
                sec(2, "3/4", "13.00"), sec(2, "1", "21.99", "10.83", "11.16"),
                sec(3, "2-1/4", "22.92", "11.31", "11.61")], "0:57.91"),
            Runner("4", 3, "AMBER TIDE", "HK_2025_C304", [
                sec(5, "3", "13.28"), sec(5, "3-1/4", "21.99", "10.83", "11.16"),
                sec(4, "2-1/2", "22.66", "11.31", "11.35")], "0:57.93"),
            Runner("5", 5, "LATE BLOOM", "HK_2022_D405", [
                sec(3, "N", "13.10"), sec(4, "SH", "22.15", "11.00", "11.15"),
                sec(5, "6", "22.87", "11.40", "11.47")], "0:58.12"),
            Runner("WV", 9, "STAYED HOME", "HK_2024_E506", [BLANK] * 6, "---"),
        ],
    ),
    {11: 57.54, 12: 57.84, 6: 57.91, 3: 57.93, 5: 58.12},
)  # fmt: skip

# 2010 layout: no sub-splits, 4 sections, an unlinked horse
ST_2010 = Meeting(
    date(2010, 1, 1),
    "ST",
    "Sha Tin",
    Race(
        3,
        [
            Runner("1", 7, "GOLDEN RULE", "HK_2008_F601", [
                sec(2, "1/2", "24.30"), sec(1, "HD", "22.61"), sec(1, "1", "23.21"),
                sec(1, "1-1/4", "23.40")], "1:33.52"),
            Runner("2", 2, "SILENT MARCH", "HK_2007_G702", [
                sec(1, "1/2", "24.22"), sec(2, "HD", "22.72"), sec(2, "1", "23.29"),
                sec(2, "1-1/4", "23.50")], "1:33.73"),
            Runner("3", 14, "NORTHERN STAR", "HK_2008_H803", [
                sec(4, "5", "25.00"), sec(4, "6-1/2", "22.80"), sec(3, "4", "22.95"),
                sec(3, "2", "23.40")], "1:34.15"),
            Runner("4", 1, "RED DUST", None, [
                sec(3, "2", "24.60"), sec(3, "NOSE", "22.70"), sec(4, "10-1/2", "24.80"),
                sec(4, "12", "24.90")], "1:37.00"),
        ],
    ),
    {7: 93.52, 2: 93.73, 14: 94.15, 1: 97.00},
)  # fmt: skip

# a 2200 m race: all six sections used, sub-splits of two and three 200 m parts
ST_2025 = Meeting(
    date(2025, 12, 7),
    "ST",
    "Sha Tin",
    Race(
        10,
        [
            Runner("1", 4, "LONG ROAD HOME", "HK_2021_J904", [
                sec(1, "1/2", "14.10"), sec(1, "3/4", "23.50", "11.80", "11.70"),
                sec(1, "1", "24.20", "12.10", "12.10"), sec(1, "1", "23.90", "11.95", "11.95"),
                sec(1, "1-1/2", "34.60", "11.30", "11.50", "11.80"),
                sec(1, "2", "22.60", "11.20", "11.40")], "2:22.90"),
            Runner("2", 13, "BRIGHT EMBER", "HK_2023_L126", [
                sec(3, "1-1/4", "14.30"), sec(2, "N", "23.40", "11.70", "11.70"),
                sec(3, "1-1/2", "24.25", "12.10", "12.15"), sec(2, "SH", "23.90", "11.90", "12.00"),
                sec(3, "2", "34.65", "11.40", "11.45", "11.80"),
                sec(2, "2", "22.66", "11.26", "11.40")], "2:23.16"),
            Runner("3", 9, "WINTER ORCHID", "HK_2022_K015", [
                sec(2, "1/2", "14.18"), sec(3, "1", "23.55", "11.85", "11.70"),
                sec(2, "1", "24.20", "12.05", "12.15"), sec(3, "1-1/4", "23.95", "12.00", "11.95"),
                sec(2, "1-1/2", "34.55", "11.30", "11.45", "11.80"),
                sec(3, "2", "22.79", "11.29", "11.50")], "2:23.22"),
        ],
    ),
    {4: 142.90, 13: 143.16, 9: 143.22},
)  # fmt: skip

MEETINGS = [HV_2026, ST_2010, ST_2025]
IDS = ["2026-HV-1000m", "2010-ST-4-sections", "2025-ST-6-sections"]


def expected_rows(meeting: Meeting) -> list[dict]:
    """The rows the page should give, worked out from the cells the page was built from."""
    out = []
    for runner in meeting.race.runners:
        for section_no, cell in enumerate(runner.sections, start=1):
            if cell.pos is None and cell.time is None:
                continue
            pos = int(cell.pos) if cell.pos else None
            out.append(
                {
                    "jurisdiction": "hkjc",
                    "race_id": meeting.race_id,
                    "horse_id": runner.horse_id,
                    "horse_no": runner.horse_no,
                    "section_no": section_no,
                    "section_pos": pos,
                    "section_margin_lengths": 0.0 if pos == 1 else LENGTHS[cell.margin],
                    "section_margin_raw": cell.margin,
                    "section_time_s": float(cell.time),
                    "sub_split_times_s": [float(s) for s in cell.subs],
                }
            )
    return out


def parse(page: RawPage) -> list[dict]:
    parsed = parse_sectional.parse(page)
    assert set(parsed) == {"sectionals"}
    return parsed["sectionals"]


def by_key(rows: list[dict]) -> dict[tuple[int, int], dict]:
    keyed = {(r["horse_no"], r["section_no"]): r for r in rows}
    assert len(keyed) == len(rows), "duplicate (horse_no, section_no) rows"
    return keyed


def one_race(runners: list[Runner], race_no: int = 1) -> RawPage:
    """A 2010-01-01 Sha Tin page holding one race, asked for by its cache key."""
    body = pages.sectional_page(date(2010, 1, 1), "Sha Tin", [Race(race_no, runners)])
    key = f"2010-01-01_ST_R{race_no:02d}"
    return RawPage("hkjc", "sectional", key, pages.sectional_url(date(2010, 1, 1), race_no), body)


# --- the definition of done ---------------------------------------------------------------------


@pytest.mark.parametrize("meeting", MEETINGS, ids=IDS)
def test_one_row_per_runner_and_section(meeting):
    rows = parse(meeting.raw())
    used = {r.horse_no: sum(1 for s in r.sections if s is not BLANK) for r in meeting.race.runners}
    expected_keys = {(no, n) for no, count in used.items() for n in range(1, count + 1)}

    assert set(by_key(rows)) == expected_keys
    assert len(rows) == sum(used.values())
    # a withdrawn runner (every cell blank) and the unused section columns give nothing
    assert {r["horse_no"] for r in rows} == {no for no, count in used.items() if count}
    assert {r["race_id"] for r in rows} == {meeting.race_id}


@pytest.mark.parametrize("meeting", MEETINGS, ids=IDS)
def test_every_row_matches_its_cell(meeting):
    key = lambda r: (r["horse_no"], r["section_no"])  # noqa: E731
    assert sorted(parse(meeting.raw()), key=key) == sorted(expected_rows(meeting), key=key)


@pytest.mark.parametrize("meeting", MEETINGS, ids=IDS)
def test_columns_are_the_spec_columns_plus_documented_extras(meeting):
    # the store's schema declares the same columns (a drift there is a spec change, not this ticket)
    assert set(schemas.TABLES["sectionals"].columns) == SPEC_COLUMNS | EXTRA_COLUMNS
    rows = parse(meeting.raw())
    assert rows
    for r in rows:
        assert set(r) == SPEC_COLUMNS | EXTRA_COLUMNS
        assert r["jurisdiction"] == "hkjc"
        assert r["race_id"] == meeting.race_id
        for col in ("horse_no", "section_no", "section_pos"):
            assert type(r[col]) is int, col
        for col in ("section_time_s", "section_margin_lengths"):
            assert type(r[col]) is float, col
        assert isinstance(r["sub_split_times_s"], list)
        assert all(type(v) is float for v in r["sub_split_times_s"])
        assert r["horse_id"] is None or isinstance(r["horse_id"], str)


# (printed position, printed margin or None for no <i>, expected lengths, expected raw margin)
MARGIN_CASES = [
    ("7", "2-3/4", 2.75, "2-3/4"),  # cell text "7 2-3/4": neither may bleed into the other
    ("12", "10-1/2", 10.5, "10-1/2"),  # a two-digit position next to a mixed fraction
    ("8", "3", 3.0, "3"),  # whole lengths: "8" + "3" is not 83 or 8.3
    ("4", "1/2", 0.5, "1/2"),
    ("2", "SH", 0.1, "SH"),
    ("3", "N", 0.3, "N"),
    ("11", "HD", 0.2, "HD"),
    ("10", "NOSE", 0.05, "NOSE"),
    ("1", "1", 0.0, "1"),  # the leader prints its lead over the second: 0 lengths behind
    ("1", "4-1/4", 0.0, "4-1/4"),
    ("5", "-", None, None),  # "-" is "nothing here", not a zero margin for a horse not leading
    ("6", None, None, None),  # no <i> at all
    ("9", "WEIRD", None, "WEIRD"),  # an unknown code keeps its raw text
]


def test_position_and_margin_are_read_from_their_own_elements():
    runners = [
        Runner(str(i + 1), i + 1, f"HORSE {i + 1}", f"HK_2009_M{i:03d}",
               [Section(pos, margin, f"{12.5 + i * 0.07:.2f}")], "0:12.50")
        for i, (pos, margin, _, _) in enumerate(MARGIN_CASES)
    ]  # fmt: skip
    rows = by_key(parse(one_race(runners)))

    assert set(rows) == {(i + 1, 1) for i in range(len(MARGIN_CASES))}
    for i, (pos, margin, lengths, raw) in enumerate(MARGIN_CASES):
        r = rows[(i + 1, 1)]
        got = (r["section_pos"], r["section_margin_lengths"], r["section_margin_raw"])
        assert got == (int(pos), lengths, raw), (pos, margin)
        assert r["section_time_s"] == pytest.approx(12.5 + i * 0.07, abs=1e-9)
        assert r["horse_id"] == f"HK_2009_M{i:03d}"


@pytest.mark.parametrize("meeting", MEETINGS, ids=IDS)
def test_sub_splits_are_kept_beside_the_section_time(meeting):
    rows = by_key(parse(meeting.raw()))
    for runner in meeting.race.runners:
        for n, cell in enumerate(runner.sections, start=1):
            if cell is BLANK:
                continue
            r = rows[(runner.horse_no, n)]
            # the p's own text is the section time; the sub-splits never replace it
            assert r["section_time_s"] == pytest.approx(float(cell.time), abs=1e-9)
            assert r["sub_split_times_s"] == pytest.approx([float(s) for s in cell.subs])
            if r["sub_split_times_s"]:
                assert sum(r["sub_split_times_s"]) == pytest.approx(r["section_time_s"], abs=0.011)


def test_sub_splits_belong_to_their_own_runner_not_the_leader_summary():
    # the leader's summary table above the race table prints its own sub-splits; a runner whose
    # cell has none (the opening section, the 2010 layout) keeps an empty list
    rows = by_key(parse(HV_2026.raw()))
    assert rows[(12, 2)]["sub_split_times_s"] == [10.71, 11.20]
    assert rows[(12, 3)]["sub_split_times_s"] == [11.19, 11.50]
    assert rows[(11, 2)]["sub_split_times_s"] == [10.79, 11.16]
    assert all(rows[(no, 1)]["sub_split_times_s"] == [] for no in (11, 12, 6, 3, 5))
    assert all(r["sub_split_times_s"] == [] for r in parse(ST_2010.raw()))
    three = by_key(parse(ST_2025.raw()))[(13, 5)]
    assert three["sub_split_times_s"] == [11.40, 11.45, 11.80]
    assert three["section_time_s"] == 34.65


@pytest.mark.parametrize("meeting", MEETINGS, ids=IDS)
def test_section_times_sum_to_the_finish_time(meeting):
    totals: dict[int, float] = defaultdict(float)
    for r in parse(meeting.raw()):
        totals[r["horse_no"]] += r["section_time_s"]
    assert set(totals) == set(meeting.finish_s)
    for horse_no, finish in meeting.finish_s.items():
        assert totals[horse_no] == pytest.approx(finish, abs=1e-6), horse_no


# --- which race the page is about ---------------------------------------------------------------


@pytest.mark.parametrize("meeting", [HV_2026, ST_2025], ids=["HV-R1", "ST-R10"])
def test_race_id_falls_back_to_url_then_page_content(meeting):
    # key unusable: date + race no from the URL (DD/MM/YYYY), venue from the meeting line
    from_url = parse(meeting.raw(key="not-a-race-key"))
    assert {r["race_id"] for r in from_url} == {meeting.race_id}
    # key and URL both unusable: date and venue from the meeting line, race from div#Race<n>
    from_content = parse(meeting.raw(key="", url="https://example.invalid/x"))
    assert {r["race_id"] for r in from_content} == {meeting.race_id}
    assert len(from_url) == len(from_content) == len(expected_rows(meeting))


def test_key_wins_over_url_and_content():
    page = HV_2026.raw(url="https://example.invalid/displaysectionaltime?RaceNo=7")
    assert {r["race_id"] for r in parse(page)} == {"hkjc:2026-09-16:HV:01"}


@pytest.mark.parametrize(
    ("key", "url"),
    [
        ("2026-09-13_ST_R01", pages.sectional_url(date(2026, 9, 13), 1)),
        # same day, other venue: the sectional URL carries no venue at all
        ("2026-09-16_ST_R01", pages.sectional_url(date(2026, 9, 16), 1)),
        ("not-a-race-key", pages.sectional_url(date(2026, 9, 13), 1)),
        ("2026-09-16_HV_R02", pages.sectional_url(date(2026, 9, 16), 2)),  # holds div#Race1 only
    ],
    ids=["other-date", "other-venue", "url-date-only", "other-race-number"],
)
def test_a_page_for_another_race_is_never_filed_under_the_requested_one(key, url):
    # the body says "Meeting Date:16/09/2026, Happy Valley" and holds Race 1: if the site answers
    # a request for another race with it, its rows must not be emitted under the requested id
    assert parse_sectional.parse(HV_2026.raw(key=key, url=url)) == {"sectionals": []}


@pytest.mark.parametrize("wanted", [1, 2, 5])
def test_all_races_page_only_yields_the_race_named_by_the_key(wanted):
    races = [
        Race(n, [Runner("1", 10 + n, f"RUNNER {n}", f"HK_2009_R{n:03d}",
                        [sec(1, "1", f"{13 + n / 10:.2f}")], "0:13.00")])
        for n in (1, 2, 5)
    ]  # fmt: skip
    body = pages.sectional_page(date(2010, 1, 1), "Sha Tin", races)
    url = f"{pages.HOST}/en-us/local/information/displaysectionaltime?racedate=01/01/2010&All=True"
    page = RawPage("hkjc", "sectional", f"2010-01-01_ST_R{wanted:02d}", url, body)
    rows = parse(page)
    assert [(r["race_id"], r["horse_no"]) for r in rows] == [
        (f"hkjc:2010-01-01:ST:{wanted:02d}", 10 + wanted)
    ]
    assert rows[0]["section_time_s"] == pytest.approx(13 + wanted / 10)


# --- pages and cells that are not the usual shape ------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        hkjc_pages.page('<div class="localResults commContent"><p>No information.</p></div>'),
        # pre-2008 meetings: the meeting line is served but there is no data table
        pages.sectional_page(date(2008, 1, 1), "Sha Tin", []),
        # the race table with its headers and no runner rows
        pages.sectional_page(date(2008, 1, 1), "Sha Tin", [Race(1, [])]),
        b"",
        b"\xff\xfe not html at all",
        b"<html><body><p>Service temporarily unavailable</p></body></html>",
        hkjc_pages.calendar(2008, 1, [hkjc_pages.meeting(1, "ST", "D", "Turf")]),
    ],
    ids=["no-information", "header-only", "no-runners", "empty", "binary", "unrelated", "fixture"],
)
def test_pages_without_data_give_an_empty_table(body):
    url = pages.sectional_url(date(2008, 1, 1), 1)
    page = RawPage("hkjc", "sectional", "2008-01-01_ST_R01", url, body)
    assert parse_sectional.parse(page) == {"sectionals": []}


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        ("/en-us/local/information/horse?horseid=HK_2008_K277", "HK_2008_K277"),
        ("/en-us/local/information/horse?HorseId=HK_2015_P101&Option=1", "HK_2015_P101"),
        # scraped hrefs are untrusted: the id ends up in join keys, URLs and cache file names
        ("/en-us/local/information/horse?horseid=../../x", None),
        ("/en-us/local/information/horse?horseid=HK%202008%20K277", None),
        ("/en-us/local/information/horse?horseid=" + "A" * 200, None),
        ("/en-us/local/information/jockey?jockeyid=ABC", None),  # not a horse link
    ],
    ids=["plain", "mixed-case-param", "path", "space", "too-long", "jockey-link"],
)
def test_horse_id_comes_from_a_safe_profile_link(href, expected):
    runner = Runner("1", 2, "ODD ONE", None, [sec(1, "1", "13.00")], "0:13.00", href=href)
    (row,) = parse(one_race([runner]))
    assert (row["horse_no"], row["horse_id"], row["section_time_s"]) == (2, expected, 13.0)


def test_minute_style_times_and_a_section_with_no_time_printed():
    runner = Runner("1", 4, "SLOW ONE", "HK_2009_S001", [
        Section("3", None, "1.02.50"), Section("2", "-", "1:02.50"), Section("1", "SH", ""),
    ], "2:05.00")  # fmt: skip
    rows = by_key(parse(one_race([runner])))
    assert rows[(4, 1)]["section_time_s"] == pytest.approx(62.5)
    assert rows[(4, 2)]["section_time_s"] == pytest.approx(62.5)
    # a position without a time is still a row: the position is data, the time is missing
    assert (rows[(4, 3)]["section_pos"], rows[(4, 3)]["section_time_s"]) == (1, None)


def test_absurd_colspan_is_clamped():
    # scraped colspans are untrusted: an absurd one must not allocate an absurd label list (ten
    # million labels is ~80 MB); clamped, the header no longer lines up with the rows
    body = HV_2026.body.replace(b'colspan="6"', b'colspan="10000000"', 1)
    assert b'colspan="10000000"' in body
    url = pages.sectional_url(HV_2026.race_date, 1)
    page = RawPage("hkjc", "sectional", HV_2026.key, url, body)
    tracemalloc.start()
    try:
        parsed = parse_sectional.parse(page)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert parsed == {"sectionals": []}
    assert peak < 20_000_000
