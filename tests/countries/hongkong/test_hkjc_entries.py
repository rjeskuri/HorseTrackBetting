"""HKJC entries page (View=All) -> `entries` rows (ticket #62, ported from HKTrack's
`sources/hkjc/parse_entries.py` and tests/unit/sources/hkjc/test_entries_parser.py) on synthetic
pages (hkjc_entries_pages.py). The real-page versions belong to a `local` ticket.

Contract pinned here (the coder follows it):
- `horsetrack.countries.hongkong.hkjc.parse_entries.parse(page: RawPage) -> ParsedPage`
  (`horsetrack.core.ingest.types`), pure, no network, returning exactly `{"entries": [...]}`.
- One row per horse in every section's entries table, then its reserves table, in page order,
  each with exactly COLUMNS. Section fields come from the section's header block: race_name,
  race_class, distance_m, section_no, rating_hi / rating_lo from the printed "hi-lo" range,
  surface "turf" (Track "Turf") or "awt" (Track "All Weather Track"), course_config the quoted
  letters of the Course line on turf (null on the AWT), section_label
  "<class> <distance>m Section <n> <race name> (Ratings <range>)". prov_race_no is the
  ProvRaceNo of the foot link naming that race, section and distance (null when the page has no
  such links).
- meeting_id `hkjc:<YYYY-MM-DD>:<VENUE>` from the section's meeting label ("15/10/2031 Happy
  Valley", venue code from the config's venue names); the cache key "YYYY-MM-DD_VV" fills in when
  the label is missing; the label wins over the key. With neither, no rows.
- Row fields: horse_id / trainer_code from the profile links, only in [A-Za-z0-9_] (else null);
  horse_name without the "(brand)"; brand_raw the brand; trainer_name as printed and trainer_key
  `trainer:<name>` (lower case, runs of anything but letters and digits as one space);
  card_wt_lb, wfa_lb, rating, trainer_preference as ints; rating_change signed; "-" is null.
  remarks_raw as printed (empty: null), decoded: '+' trump_card, '*' priority_to_run, '#'
  late_entry. Reserve rows have is_reserve true and reserve_no the printed "No."; entries rows
  is_reserve false and reserve_no null. Columns are found by header text, never by position.
- entries_scraped_utc is the page's fetch time, null when the page has none.
- A page without entry tables (no information, garbage, another page type) gives
  `{"entries": []}`.

Every expected value is written out here or derived from the page specs by the rules above,
never by the code under test. Names, ids, codes and numbers are invented.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone

import hkjc_corunning_pages
import hkjc_entries_pages as pages
import hkjc_pages
import hkjc_sectional_pages
import pytest
from hkjc_entries_pages import COLUMNS as PRINTED
from hkjc_entries_pages import RESERVE_COLUMNS, Entry, Section, entry

from horsetrack.core.ingest.types import RawPage

# isort: split
# (its own block: ruff ranks a module that does not exist yet as third-party)
from horsetrack.countries.hongkong.hkjc import parse_entries

SCRAPED = datetime(2031, 10, 13, 2, 54, 0, tzinfo=timezone.utc)

COLUMNS = {
    "jurisdiction", "meeting_id", "section_label", "prov_race_no", "section_no", "race_name",
    "race_class", "distance_m", "rating_hi", "rating_lo", "surface", "course_config", "horse_id",
    "horse_name", "brand_raw", "trainer_key", "trainer_code", "trainer_name", "card_wt_lb",
    "wfa_lb", "rating", "rating_change", "trainer_preference", "remarks_raw", "trump_card",
    "priority_to_run", "late_entry", "is_reserve", "reserve_no", "entries_scraped_utc",
}  # fmt: skip

# remarks as printed -> (trump card, priority to run, late entry), worked out by hand
REMARK_FLAGS = {
    "": (False, False, False),
    "+": (True, False, False),
    "*": (False, True, False),
    "+ *": (True, True, False),
    "#": (False, False, True),
    "# *": (False, True, True),
}


@dataclass(frozen=True)
class Meeting:
    race_date: date
    venue: str
    sections: tuple[Section, ...]

    @property
    def key(self) -> str:
        return pages.meeting_key(self.race_date, self.venue)

    @property
    def meeting_id(self) -> str:
        return f"hkjc:{self.race_date.isoformat()}:{self.venue}"

    def body(self, **kw) -> bytes:
        return pages.entries_page(self.race_date, self.venue, self.sections, **kw)

    def raw(self, body: bytes | None = None, key: str | None = None, **kw) -> RawPage:
        return RawPage(
            "hkjc",
            "entries",
            self.key if key is None else key,
            pages.entries_url(self.race_date, self.venue),
            self.body(**kw) if body is None else body,
            fetched_utc=SCRAPED,
        )


def reserves(*pairs: tuple[int, int]) -> tuple[Entry, ...]:
    """(stand-by number, entry number) pairs -> reserve entries."""
    return tuple(entry(n, standby=no, remarks="") for no, n in pairs)


ALDER = "SYNTHETIC ALDER HANDICAP"
ELM = "SYNTHETIC ELM HANDICAP"

# Happy Valley: one race split in two sections, sections with and without reserves, a 40-0 band
HV = Meeting(
    date(2031, 10, 15),
    "HV",
    (
        Section("P10", 1, ALDER, "Class 3", 1200, "80-60", course='"C" Course - Worked Back',
                entries=tuple(entry(n) for n in range(1, 6)), reserves=reserves((1, 6), (2, 7))),
        Section("P10", 2, ALDER, "Class 3", 1200, "80-60", course='"C" Course - Worked Back',
                entries=tuple(entry(n) for n in range(8, 12))),
        Section("P20", 1, "SYNTHETIC BIRCH HANDICAP", "Class 4", 1800, "60-40",
                course='"C" Course - Worked Back', entries=tuple(entry(n) for n in range(12, 16)),
                reserves=reserves((1, 16), (2, 17), (3, 18))),
        Section("P30", 1, "SYNTHETIC CEDAR HANDICAP", "Class 5", 1650, "40-0",
                course='"C" Course - Worked Back', entries=tuple(entry(n) for n in range(19, 22)),
                reserves=reserves((1, 22))),
    ),
)  # fmt: skip

# Sha Tin: an all-weather section first, stand-by numbers that are not 1..n, a "C+3" course
ST = Meeting(
    date(2032, 1, 4),
    "ST",
    (
        Section("P15", 1, "SYNTHETIC DOGWOOD HANDICAP", "Class 4", 1650, "60-40",
                track="All Weather Track", course="-",
                entries=tuple(entry(n) for n in range(30, 34)),
                reserves=reserves((2, 34), (3, 35), (5, 36))),
        Section("P25", 1, ELM, "Class 2", 1400, "100-80", course='"C+3" Course',
                entries=tuple(entry(n) for n in range(40, 44)), reserves=reserves((1, 44))),
        Section("P25", 2, ELM, "Class 2", 1400, "100-80", course='"C+3" Course',
                entries=tuple(entry(n) for n in range(50, 53)),
                reserves=reserves((1, 53), (2, 54))),
    ),
)  # fmt: skip

MEETINGS = [HV, ST]
IDS = ["HV-4-sections", "ST-awt-and-split-race"]


def trainer_key(name: str) -> str:
    return "trainer:" + re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def section_fields(meeting: Meeting, s: Section, prov: bool = True) -> dict:
    hi, lo = (int(x) for x in s.band.split("-"))
    turf = s.track == "Turf"
    config = re.search(r'"([^"]+)"', s.course)
    return {
        "jurisdiction": "hkjc",
        "meeting_id": meeting.meeting_id,
        "section_label": s.label,
        "prov_race_no": s.prov if prov else None,
        "section_no": s.section_no,
        "race_name": s.race_name,
        "race_class": s.race_class,
        "distance_m": s.distance_m,
        "rating_hi": hi,
        "rating_lo": lo,
        "surface": "turf" if turf else "awt",
        "course_config": config.group(1) if turf and config else None,
    }


def entry_fields(e: Entry) -> dict:
    trump, priority, late = REMARK_FLAGS[e.remarks]
    return {
        "horse_id": e.horse_id,
        "horse_name": e.name,
        "brand_raw": e.brand,
        "trainer_key": trainer_key(e.trainer),
        "trainer_code": e.trainer_code,
        "trainer_name": e.trainer,
        "card_wt_lb": e.wt,
        "wfa_lb": e.wfa,
        "rating": e.rating,
        "rating_change": e.rating_change,
        "trainer_preference": e.preference,
        "remarks_raw": e.remarks or None,
        "trump_card": trump,
        "priority_to_run": priority,
        "late_entry": late,
        "is_reserve": e.standby is not None,
        "reserve_no": e.standby,
        "entries_scraped_utc": SCRAPED,
    }


def expected_rows(meeting: Meeting, prov: bool = True) -> list[dict]:
    """The rows the page should give, in page order, worked out from the specs it was built from."""
    return [
        {**section_fields(meeting, s, prov), **entry_fields(e)}
        for s in meeting.sections
        for e in (*s.entries, *s.reserves)
    ]


def parse(page: RawPage) -> list[dict]:
    parsed = parse_entries.parse(page)
    assert set(parsed) == {"entries"}
    return parsed["entries"]


def named(rows: list[dict], name: str) -> dict:
    (row,) = [r for r in rows if r["horse_name"] == name]
    return row


# --- the definition of done ---------------------------------------------------------------------


@pytest.mark.parametrize("meeting", MEETINGS, ids=IDS)
def test_every_row_has_the_same_columns(meeting):
    rows = parse(meeting.raw())
    assert len(rows) == sum(len(s.entries) + len(s.reserves) for s in meeting.sections)
    for r in rows:
        assert set(r) == COLUMNS
        for col in ("section_no", "distance_m", "rating_hi", "rating_lo", "card_wt_lb", "rating"):
            assert type(r[col]) is int, col
        for col in ("trump_card", "priority_to_run", "late_entry", "is_reserve"):
            assert type(r[col]) is bool, col


@pytest.mark.parametrize("meeting", MEETINGS, ids=IDS)
def test_sections_in_page_order(meeting):
    rows = parse(meeting.raw())
    seen = list(dict.fromkeys((r["prov_race_no"], r["section_no"]) for r in rows))
    assert seen == [(s.prov, s.section_no) for s in meeting.sections]
    per_section = Counter((r["prov_race_no"], r["section_no"], r["is_reserve"]) for r in rows)
    for s in meeting.sections:
        assert per_section[(s.prov, s.section_no, False)] == len(s.entries), s.label
        assert per_section[(s.prov, s.section_no, True)] == len(s.reserves), s.label
    # within the page: each section's entries in table order, then its reserves
    assert [r["horse_name"] for r in rows] == [
        e.name for s in meeting.sections for e in (*s.entries, *s.reserves)
    ]
    # a section's header fields go to its own rows only
    for s in meeting.sections:
        mine = [r for r in rows if (r["prov_race_no"], r["section_no"]) == (s.prov, s.section_no)]
        assert {r["section_label"] for r in mine} == {s.label}
        assert {r["distance_m"] for r in mine} == {s.distance_m}


@pytest.mark.parametrize("meeting", MEETINGS, ids=IDS)
def test_reserve_rows_carry_their_standby_number(meeting):
    rows = parse(meeting.raw())
    for s in meeting.sections:
        mine = [r for r in rows if (r["prov_race_no"], r["section_no"]) == (s.prov, s.section_no)]
        got = [(r["horse_id"], r["reserve_no"]) for r in mine if r["is_reserve"]]
        assert got == [(e.horse_id, e.standby) for e in s.reserves], s.label
        assert all(r["reserve_no"] is None for r in mine if not r["is_reserve"])
    if meeting is ST:  # the stand-by number is the printed "No.", not the row's position
        awt = [r["reserve_no"] for r in rows if r["prov_race_no"] == "P15" and r["is_reserve"]]
        assert awt == [2, 3, 5]
    # a reserve row is a whole row: its own weights, ratings and section, not its neighbour's
    for s in meeting.sections:
        for e in s.reserves:
            r = named(rows, e.name)
            assert (r["card_wt_lb"], r["rating"], r["trainer_code"]) == (
                e.wt,
                e.rating,
                e.trainer_code,
            )
            assert (r["prov_race_no"], r["section_no"]) == (s.prov, s.section_no)


# --- every field --------------------------------------------------------------------------------


@pytest.mark.parametrize("meeting", MEETINGS, ids=IDS)
def test_every_row_matches_its_entry(meeting):
    assert parse(meeting.raw()) == expected_rows(meeting)


def test_remarks_are_kept_as_printed_and_decoded():
    marks = list(REMARK_FLAGS)
    section = replace(
        HV.sections[0],
        entries=tuple(entry(60 + i, remarks=m) for i, m in enumerate(marks)),
        reserves=(entry(70, standby=1, remarks="+ *"),),
    )
    rows = parse(replace(HV, sections=(section,)).raw())
    got = [(r["remarks_raw"], r["trump_card"], r["priority_to_run"], r["late_entry"]) for r in rows]
    assert got == [(m or None, *REMARK_FLAGS[m]) for m in [*marks, "+ *"]]


def test_columns_are_found_by_header_text_not_position():
    shuffled = ("Remarks", "Rtg.", "Trainer", "P", "Horse (Brand No)", "Rtg.+/-", "WFA.", "Wt.")
    reserve_shuffled = ("Horse (Brand No)", "Wt.", "No.", "Remarks", "Trainer", "P", "Rtg.+/-",
                        "WFA.", "Rtg.")  # fmt: skip
    assert sorted(shuffled) == sorted(PRINTED) and sorted(reserve_shuffled) == sorted(
        RESERVE_COLUMNS
    )
    for meeting in MEETINGS:
        page = meeting.raw(columns=shuffled, reserve_columns=reserve_shuffled)
        assert parse(page) == expected_rows(meeting)


# --- which meeting and race the page is about ---------------------------------------------------


@pytest.mark.parametrize("meeting", MEETINGS, ids=IDS)
def test_the_key_fills_in_when_the_page_has_no_meeting_label(meeting):
    rows = parse(meeting.raw(label=False))
    assert rows == expected_rows(meeting)
    # no label and no usable key: there is no meeting to file the rows under
    assert parse(meeting.raw(label=False, key="garbage")) == []
    assert parse(meeting.raw(label=False, key="")) == []


def test_the_page_label_wins_over_the_key():
    rows = parse(HV.raw(key="2032-01-04_ST"))
    assert {r["meeting_id"] for r in rows} == {"hkjc:2031-10-15:HV"}
    rows = parse(ST.raw(key="not-a-key"))
    assert {r["meeting_id"] for r in rows} == {"hkjc:2032-01-04:ST"}


@pytest.mark.parametrize("meeting", MEETINGS, ids=IDS)
def test_provisional_race_codes_come_from_the_foot_links(meeting):
    # matched by race name, section and distance, not by the links' order
    assert parse(meeting.raw(nav_reversed=True)) == expected_rows(meeting)
    # without the links the code is unknown; every other field survives
    assert parse(meeting.raw(nav=False)) == expected_rows(meeting, prov=False)


# --- untrusted and empty pages ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("href", "trainer_href", "horse_id", "trainer_code"),
    [
        ("/en-us/local/information/horse?horseid=HK_2029_Z077",
         "/en-us/local/information/trainerprofile?trainerid=TQX", "HK_2029_Z077", "TQX"),
        ("/en-us/local/information/horse?HorseId=HK_2029_Z078&Option=1",
         "/en-us/local/information/trainerprofile?TrainerId=T_9", "HK_2029_Z078", "T_9"),
        # scraped hrefs are untrusted: ids end up in join keys, URLs and cache file names
        ("/en-us/local/information/horse?horseid=HK_2029_Z077/../x",
         "/en-us/local/information/trainerprofile?trainerid=T%20QX", None, None),
        ("/en-us/local/information/horse?horseid=" + "A" * 200,
         "/en-us/local/information/trainerprofile?trainerid=../../etc", None, None),
    ],
    ids=["plain", "mixed-case-param", "path-and-space", "too-long-and-path"],
)  # fmt: skip
def test_ids_from_hrefs_are_restricted_to_a_plain_charset(
    href, trainer_href, horse_id, trainer_code
):
    odd = entry(77, href=href, trainer_href=trainer_href, horse_id=None, trainer_code=None)
    section = replace(HV.sections[0], entries=(odd,), reserves=())
    (row,) = parse(replace(HV, sections=(section,)).raw())
    assert (row["horse_id"], row["horse_name"], row["brand_raw"]) == (horse_id, odd.name, odd.brand)
    assert (row["trainer_code"], row["trainer_name"]) == (trainer_code, odd.trainer)
    assert row["trainer_key"] == trainer_key(odd.trainer)


def test_a_horse_without_a_link_keeps_its_name_and_brand():
    plain = entry(76, horse_id=None, trainer_code=None)
    section = replace(HV.sections[0], entries=(entry(1), plain), reserves=())
    rows = parse(replace(HV, sections=(section,)).raw())
    assert [(r["horse_id"], r["horse_name"], r["brand_raw"], r["trainer_code"]) for r in rows] == [
        ("HK_2029_Z001", entry(1).name, "Z001", entry(1).trainer_code),
        (None, plain.name, plain.brand, None),
    ]


@pytest.mark.parametrize(
    "body",
    [
        hkjc_pages.page('<div class="commContent"><p>No information.</p></div>'),
        b"",
        b"\xff\xfe not html at all",
        b"<html><body><p>503 Service Unavailable</p></body></html>",
        # a section whose entries table has its header and no rows
        pages.entries_page(
            date(2031, 10, 15), "HV", [Section("P10", 1, ALDER, "Class 3", 1200, "80-60")]
        ),
        hkjc_pages.calendar(2031, 10, [hkjc_pages.meeting(15, "HV", "N", "Turf")]),
        hkjc_sectional_pages.sectional_page(date(2031, 10, 15), "Happy Valley", []),
        hkjc_corunning_pages.corunning_page(date(2031, 10, 15), "HV", 1, []),
    ],
    ids=["no-information", "empty", "binary", "unrelated", "no-rows", "fixture", "sectional",
         "corunning"],
)  # fmt: skip
def test_pages_without_entries_give_an_empty_table(body):
    assert parse_entries.parse(HV.raw(body=body)) == {"entries": []}


def test_missing_fetch_time_is_null():
    page = RawPage("hkjc", "entries", HV.key, "", HV.body())
    rows = parse(page)
    assert len(rows) == len(expected_rows(HV))
    assert {r["entries_scraped_utc"] for r in rows} == {None}
