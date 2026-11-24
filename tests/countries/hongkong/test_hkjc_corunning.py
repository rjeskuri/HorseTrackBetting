"""HKJC Comments on Running page -> `running_comments` rows, the per-race gear source (ticket #62,
ported from HKTrack's `sources/hkjc/parse_corunning.py` and
tests/unit/sources/hkjc/test_corunning_parser.py) on synthetic pages (hkjc_corunning_pages.py).
The real-page versions belong to a `local` ticket.

Contract pinned here (the coder follows it):
- `horsetrack.countries.hongkong.hkjc.parse_corunning.parse(page: RawPage) -> ParsedPage`
  (`horsetrack.core.ingest.types`), pure, no network, returning exactly
  `{"running_comments": [...]}`.
- One row per runner of the comment table (the first table with Horse No and Comment headers;
  columns found by header text, never by position), in table order, with exactly the columns of
  `schemas.TABLES["running_comments"]`: jurisdiction "hkjc", race_id
  `hkjc:<YYYY-MM-DD>:<VENUE>:<NN>`, horse_id from the profile link (never the printed brand;
  null when unlinked or outside [A-Za-z0-9_]), horse_no (int), gear_raw as printed ("--" or
  blank: null), comment whitespace-normalised. A row without a horse number is skipped.
- A comment cell reading "No Comments on Running information for this horse." is null; when
  every runner's comment is missing (not published yet) the page gives an empty table, so a race
  without comments never looks parsed.
- The race is the cache key's ("YYYY-MM-DD_VV_RNN"), then the URL's (date|Date YYYYMMDD,
  raceno|RaceNo), then the page's own content (its Results link inside `div.corunning`, then the
  race tabs' date, the header's venue name and "Race N"). A page whose content names another
  date, venue or race than the one asked for gives an empty table. Links outside the page's own
  content (site navigation) neither relabel nor veto it.
- Pages without a comment table (no information, garbage, other page types) give an empty table.

Every expected value is written out here or derived from the page specs by the rules above,
never by the code under test. Names, ids, codes and comments are invented.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import hkjc_corunning_pages as pages
import hkjc_entries_pages
import hkjc_pages
import hkjc_sectional_pages
import pytest
from hkjc_corunning_pages import NOT_PUBLISHED, Runner

from horsetrack.core.ingest.types import RawPage
from horsetrack.core.store import schemas

# isort: split
# (its own block: ruff ranks a module that does not exist yet as third-party)
from horsetrack.countries.hongkong.hkjc import parse_corunning

EMPTY = {"running_comments": []}
COLUMNS = {"jurisdiction", "race_id", "horse_id", "horse_no", "gear_raw", "comment"}

OPENINGS = (
    "Jumped well", "Began awkwardly", "Slow to begin", "Shifted out on jumping",
    "Bumped at the start",
    "Broke smartly", "Missed the kick", "Jumped on terms",
)  # fmt: skip
MIDDLES = (
    "settled 2-wide behind the leaders", "raced keenly in midfield", "trailed the field early",
    "held an inside position on the rail", "was taken back to race at the tail",
)  # fmt: skip
ENDINGS = (
    "kept on steadily to the line.", "weakened final 150M.", "ran on strongly late to go close.",
    "was never a threat.", "found no extra inside 200M.", "stayed on at one pace.",
)  # fmt: skip


def comment(n: int) -> str:
    """A distinct comment over several printed lines, with runs of spaces, a tab, a nbsp and an
    ampersand (printed escaped)."""
    return (
        f"{OPENINGS[n % len(OPENINGS)]},\n          {MIDDLES[n % len(MIDDLES)]}"
        f"  near {100 * n + 50}M,"
        f"\n\t angled out\xa0in the straight & {ENDINGS[n % len(ENDINGS)]}"
    )


def normalised(text: str | None) -> str | None:
    return " ".join(text.split()) if text is not None else None


def gear_value(printed: str | None) -> str | None:
    """Gear as stored: the printed code, or null for a blank cell or the "--" placeholder (a
    None spec prints "--")."""
    return printed if printed and printed.strip("-") else None


GEARS_2010 = (
    "H",
    "B1",
    None,
    "CP/TT",
    "TT",
    "TT1",
    "CP/TT1",
    "B",
    "",
    "V-/B2/TT",
    "B/H1",
    "XB-/TT",
)
NOS_2010 = (5, 7, 3, 12, 11, 9, 4, 2, 1, 10, 8, 6)  # in placing order


@dataclass(frozen=True)
class Race:
    race_date: date
    venue: str
    race_no: int
    runners: tuple[Runner, ...]
    archive: bool

    @property
    def key(self) -> str:
        return f"{self.race_date.isoformat()}_{self.venue}_R{self.race_no:02d}"

    @property
    def race_id(self) -> str:
        return f"hkjc:{self.race_date.isoformat()}:{self.venue}:{self.race_no:02d}"

    @property
    def url(self) -> str:
        return pages.corunning_url(self.race_date, self.race_no, archive=self.archive)

    def body(self, **kw) -> bytes:
        return pages.corunning_page(self.race_date, self.venue, self.race_no, self.runners, **kw)

    def raw(self, body: bytes | None = None, key: str | None = None, url: str | None = None, **kw):
        return RawPage(
            "hkjc",
            "corunning",
            self.key if key is None else key,
            self.url if url is None else url,
            self.body(**kw) if body is None else body,
        )


# 2010 archive page, Sha Tin race 1: twelve runners, the brand printed "CQ3nn" (not the id)
ST_2010 = Race(
    date(2010, 1, 1),
    "ST",
    1,
    tuple(
        Runner(str(i + 1), no, f"SYNTH RUNNER {no}", f"HK_2008_Q{300 + no}", f"CQ{300 + no}",
               f"J Jockey{i}", GEARS_2010[i], comment(i))
        for i, no in enumerate(NOS_2010)
    ),
    archive=True,
)  # fmt: skip

# 2026 live page, Happy Valley race 3: eight runners, one without a profile link
HV_2026 = Race(
    date(2026, 9, 16),
    "HV",
    3,
    (
        Runner("1", 8, "PAPER CRANE", "HK_2024_A801", "A801", "B Birch", "B/TT", comment(20)),
        Runner("2", 2, "QUIET MEADOW", "HK_2023_B202", "B202", "C Cedar", "--", comment(21)),
        Runner("3", 11, "COPPER FINCH", "HK_2025_C311", "C311", "D Dogwood", "PC1", comment(22)),
        Runner("4", 1, "UNLINKED ONE", None, "D401", "E Elm", "SR/TT", comment(23)),
        Runner("5", 6, "AMBER REED", "HK_2022_E506", "E506", "F Fir", None, comment(24)),
        Runner("6", 4, "LATE FROST", "HK_2024_F604", "F604", "G Gorse", "CP-/V1", comment(25)),
        Runner("7", 9, "NORTH WIND", "HK_2023_G709", "G709", "H Hazel", "B", comment(26)),
        Runner("8", 3, "STONE BRIDGE", "HK_2025_H803", "H803", "I Ilex", "H/XB", comment(27)),
    ),
    archive=False,
)  # fmt: skip

RACES = [ST_2010, HV_2026]
IDS = ["2010-ST-R1-archive", "2026-HV-R3-live"]


def expected_rows(race: Race) -> list[dict]:
    return [
        {
            "jurisdiction": "hkjc",
            "race_id": race.race_id,
            "horse_id": r.horse_id,
            "horse_no": r.horse_no,
            "gear_raw": gear_value(r.gear),
            "comment": normalised(r.comment),
        }
        for r in race.runners
    ]


def parse(page: RawPage) -> list[dict]:
    parsed = parse_corunning.parse(page)
    assert set(parsed) == {"running_comments"}
    return parsed["running_comments"]


def by_no(rows: list[dict]) -> dict[int, dict]:
    keyed = {r["horse_no"]: r for r in rows}
    assert len(keyed) == len(rows), "duplicate horse_no rows"
    return keyed


# --- the definition of done ---------------------------------------------------------------------


@pytest.mark.parametrize("race", RACES, ids=IDS)
def test_one_row_per_runner(race):
    rows = parse(race.raw())
    assert set(schemas.TABLES["running_comments"].columns) == COLUMNS
    assert len(rows) == len(race.runners)
    assert sorted(by_no(rows)) == sorted(r.horse_no for r in race.runners)
    assert [r["horse_no"] for r in rows] == [r.horse_no for r in race.runners]  # table order
    for r in rows:
        assert set(r) == COLUMNS
        assert (r["jurisdiction"], r["race_id"]) == ("hkjc", race.race_id)
        assert type(r["horse_no"]) is int


@pytest.mark.parametrize("race", RACES, ids=IDS)
def test_every_row_matches_its_runner(race):
    assert parse(race.raw()) == expected_rows(race)


@pytest.mark.parametrize("race", RACES, ids=IDS)
def test_gear_is_kept_as_printed(race):
    gear = {no: r["gear_raw"] for no, r in by_no(parse(race.raw())).items()}
    assert gear == {r.horse_no: gear_value(r.gear) for r in race.runners}
    # "--" is the page's "no gear" placeholder, not a gear code: it must not survive as a string
    assert all(g is None or (g.strip("-") and g == g.strip()) for g in gear.values())
    if race is ST_2010:
        assert gear == {
            5: "H", 7: "B1", 3: None, 12: "CP/TT", 11: "TT", 9: "TT1", 4: "CP/TT1", 2: "B", 1: None,
            10: "V-/B2/TT", 8: "B/H1", 6: "XB-/TT",
        }  # fmt: skip


@pytest.mark.parametrize("race", RACES, ids=IDS)
def test_horse_ids_come_from_the_links(race):
    rows = by_no(parse(race.raw()))
    for runner in race.runners:
        got = rows[runner.horse_no]["horse_id"]
        assert got == runner.horse_id, runner.name
        assert got != runner.brand  # the brand printed next to the name is not the id
    if race is ST_2010:
        assert (rows[5]["horse_id"], rows[7]["horse_id"], rows[6]["horse_id"]) == (
            "HK_2008_Q305", "HK_2008_Q307", "HK_2008_Q306",
        )  # fmt: skip
    if race is HV_2026:
        assert rows[1]["horse_id"] is None  # printed without a profile link


@pytest.mark.parametrize("race", RACES, ids=IDS)
def test_comments_are_whole_and_whitespace_normalised(race):
    rows = by_no(parse(race.raw()))
    for runner in race.runners:
        got = rows[runner.horse_no]["comment"]
        assert got == normalised(runner.comment)
        assert got == " ".join(got.split()) and "\xa0" not in got and len(got) > 40
    first = rows[race.runners[0].horse_no]["comment"]
    opening = OPENINGS[0] if race is ST_2010 else OPENINGS[20 % len(OPENINGS)]
    assert first.startswith(opening + ", ")
    assert first.endswith(ENDINGS[0] if race is ST_2010 else ENDINGS[20 % len(ENDINGS)])
    assert " angled out in the straight & " in first


# --- which race the page is about ---------------------------------------------------------------


@pytest.mark.parametrize("race", RACES, ids=IDS)
def test_race_id_falls_back_to_url_then_page_content(race):
    # key unusable: date and race number from the URL, venue from the page's own Results link
    from_url = parse(race.raw(key="x"))
    assert {r["race_id"] for r in from_url} == {race.race_id}
    # key and URL unusable: the Results link carries racedate / racecourse / raceno
    from_link = parse(race.raw(key="", url="https://example.invalid/"))
    assert {r["race_id"] for r in from_link} == {race.race_id}
    # no Results link either: the race tabs' date, the header's venue name and "Race N"
    from_text = parse(race.raw(key="", url="https://example.invalid/", results_link=False))
    assert {r["race_id"] for r in from_text} == {race.race_id}
    assert len(from_url) == len(from_link) == len(from_text) == len(race.runners)


def test_the_live_and_archive_url_forms_both_name_the_race():
    for archive in (True, False):
        url = pages.corunning_url(HV_2026.race_date, HV_2026.race_no, archive=archive)
        assert parse(HV_2026.raw(key="x", url=url)) == expected_rows(HV_2026)


@pytest.mark.parametrize(
    ("key", "url"),
    [
        ("2026-09-13_HV_R03", pages.corunning_url(date(2026, 9, 13), 3, archive=False)),
        ("2026-09-16_ST_R03", pages.corunning_url(date(2026, 9, 16), 3, archive=False)),
        ("2026-09-16_HV_R05", pages.corunning_url(date(2026, 9, 16), 5, archive=False)),
        ("x", pages.corunning_url(date(2026, 9, 13), 3, archive=False)),
        ("", pages.corunning_url(date(2026, 9, 16), 7, archive=True)),
    ],
    ids=["other-date", "other-venue", "other-race-no", "url-only-date", "url-only-race-no"],
)
def test_a_page_for_another_race_is_never_filed_under_the_requested_race(key, url):
    # the body is 2026-09-16 Happy Valley race 3 (its Results link and "Race 3" header say so)
    assert parse_corunning.parse(HV_2026.raw(key=key, url=url)) == EMPTY


@pytest.mark.parametrize("race", RACES, ids=IDS)
def test_site_navigation_links_do_not_decide_which_race_the_page_is(race):
    # a full page has site navigation above the content; a results link there (the latest
    # meeting, say) is not evidence about this page and must neither relabel nor veto it
    other = pages.results_href(date(2031, 9, 20), "ST" if race.venue == "HV" else "HV", 9)
    nav = f'<div class="nav"><a href="{other}">Results</a></div>'
    for key in (race.key, ""):
        rows = parse(race.raw(key=key, site_nav=nav))
        assert rows == expected_rows(race)


# --- comments not yet published -----------------------------------------------------------------


@pytest.mark.parametrize("race", RACES, ids=IDS)
def test_comments_not_yet_published_give_an_empty_table(race):
    # the days after a meeting: gear is printed, every comment cell reads the sentinel
    blank = tuple(replace(r, comment=None) for r in race.runners)
    body = pages.corunning_page(race.race_date, race.venue, race.race_no, blank)
    assert NOT_PUBLISHED.encode() in body
    assert parse_corunning.parse(race.raw(body=body)) == EMPTY


@pytest.mark.parametrize("missing", [0, 4, 7])  # 4 prints no gear: the row stays all the same
def test_a_single_horse_without_a_comment_keeps_its_gear(missing):
    runners = tuple(
        replace(r, comment=None) if i == missing else r for i, r in enumerate(HV_2026.runners)
    )
    rows = parse(replace(HV_2026, runners=runners).raw())
    assert len(rows) == len(runners)
    gone = HV_2026.runners[missing]
    assert [(r["horse_no"], r["gear_raw"]) for r in rows if r["comment"] is None] == [
        (gone.horse_no, gear_value(gone.gear))
    ]
    assert all(NOT_PUBLISHED not in (r["comment"] or "") for r in rows)


# --- pages and tables that are not the usual shape ----------------------------------------------


def test_columns_are_mapped_by_header_text_not_position():
    shuffled = ("Horse No", "Comment", "Gear", "Horse Name")  # no Placing or Jockey column
    stray = (
        "<tr><td></td><td>stray row without a horse number, long enough to look real</td>"
        "<td>B</td><td></td></tr>"
    )
    for race in RACES:
        rows = parse(race.raw(columns=shuffled, extra_rows=stray))
        assert rows == expected_rows(race)


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        ("/en-us/local/information/horse?horseid=HK_2024_A801", "HK_2024_A801"),
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
    odd = replace(HV_2026.runners[0], horse_id=None, href=href)
    rows = by_no(parse(replace(HV_2026, runners=(odd, *HV_2026.runners[1:])).raw()))
    assert (rows[odd.horse_no]["horse_id"], rows[odd.horse_no]["gear_raw"]) == (expected, odd.gear)
    assert len(rows) == len(HV_2026.runners)


@pytest.mark.parametrize(
    "body",
    [
        hkjc_pages.page('<div class="localResults commContent"><p>No information.</p></div>'),
        hkjc_pages.page(
            '<div class="corunning commContent"><p>No Comments on Running information.</p></div>'
        ),
        # the comment table with its headers and no runner rows
        pages.corunning_page(date(2026, 9, 16), "HV", 3, []),
        b"",
        b"\xff\xfe not html at all",
        b"<html><body><table class='table_bd'><tr><td>Pool</td><td>Dividend</td></tr>"
        b"</table></body></html>",
        hkjc_sectional_pages.sectional_page(date(2026, 9, 16), "Happy Valley", []),
        hkjc_entries_pages.entries_page(date(2026, 9, 16), "HV", []),
        hkjc_pages.calendar(2026, 9, [hkjc_pages.meeting(16, "HV", "N", "Turf")]),
    ],
    ids=["no-information", "page-level-sentinel", "no-runners", "empty", "binary",
         "unrelated-table",
         "sectional", "entries", "fixture"],
)  # fmt: skip
def test_pages_without_data_give_an_empty_table(body):
    assert parse_corunning.parse(HV_2026.raw(body=body)) == EMPTY
