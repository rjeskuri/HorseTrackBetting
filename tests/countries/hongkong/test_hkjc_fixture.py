"""HKJC monthly fixture calendar -> `meetings` rows (ticket #60, ported from HKTrack's
`sources/hkjc/parse_fixture.py`).

Contract pinned here (the coder follows it):
- `horsetrack.countries.hongkong.hkjc.parse_fixture.parse(page: RawPage) -> ParsedPage`
  (`horsetrack.core.ingest.types`), pure, no network.
- `meetings`: one row per `td.calendar` cell whose icons name a configured local venue (HV, ST),
  sorted by (race_date, venue), with exactly the store's `meetings` columns
  (docs/specs/store.md plus `jurisdiction`): meeting_id `hkjc:<YYYY-MM-DD>:<VENUE>`, the season
  label of a September-start season ("2026/27"), day_night from the session icon (D/N/T ->
  day/night/twilight), surface_mix from the surface icon (turf/mixed/awt), status "scheduled".
  The calendar is the scheduled programme: a meeting later abandoned or cancelled (a cell with no
  race lines) is still listed; the results pages decide `run` or `abandoned` at build time.
- The month comes from the page's own title cell ("9/2026"); the cache key ("2026-09") and the
  URL's calyear/calmonth are fallbacks, in that order. No month at all: no meetings.

Pages are synthetic (hkjc_pages.py); expected rows are written out here.
"""

from __future__ import annotations

from datetime import date

import hkjc_pages as pages
import pytest

from horsetrack.core.ingest.types import RawPage
from horsetrack.core.store import schemas

# isort: split
# (its own block: ruff ranks a module that does not exist yet as third-party)
from horsetrack.countries.hongkong.hkjc import parse_fixture

# docs/specs/store.md, meetings: the listed columns plus the `jurisdiction` every table carries
SPEC_COLUMNS = {
    "jurisdiction",
    "meeting_id",
    "race_date",
    "venue",
    "season",
    "day_night",
    "surface_mix",
    "status",
}
SESSION = {"D": "day", "N": "night", "T": "twilight"}
SURFACE = {"Turf": "turf", "Mixed": "mixed", "AWT": "awt"}


def raw(body: bytes, key: str, url: str | None = None) -> RawPage:
    if url is None:
        year, month = key.split("-")
        url = pages.fixture_url(int(year), int(month))
    return RawPage("hkjc", "fixture", key, url, body)


def row(day: date, venue: str, season: str, session: str | None, surface: str | None) -> dict:
    return {
        "jurisdiction": "hkjc",
        "meeting_id": f"hkjc:{day.isoformat()}:{venue}",
        "race_date": day,
        "venue": venue,
        "season": season,
        "day_night": session,
        "surface_mix": surface,
        "status": "scheduled",
    }


def summary(parsed) -> list[tuple[date, str, str | None, str | None]]:
    return [
        (m["race_date"], m["venue"], m["day_night"], m["surface_mix"]) for m in parsed["meetings"]
    ]


# (year, month, season label, meetings as (day, venue, session icon, surface icon))
MONTHS = [
    (2026, 9, "2026/27", [(5, "ST", "D", "Turf"), (9, "HV", "N", "Turf"),
                          (13, "ST", "D", "Mixed"), (30, "HV", "N", "Turf")]),
    (2026, 7, "2025/26", [(1, "ST", "D", "AWT"), (12, "ST", "T", "Turf"),
                          (15, "HV", "N", "Turf")]),
    (2010, 1, "2009/10", [(1, "ST", "D", "Turf"), (6, "HV", "N", "Turf"),
                          (31, "ST", "D", "Mixed")]),
    (2024, 2, "2023/24", [(4, "ST", "D", "AWT"), (29, "HV", "N", "Turf")]),  # leap day
    (2019, 12, "2019/20", [(1, "ST", "D", "Turf"), (11, "HV", "N", "Turf"),
                           (25, "ST", "T", "Mixed")]),
    (2031, 8, "2030/31", [(31, "HV", "D", "Turf")]),  # the last day before a season starts
]  # fmt: skip


@pytest.mark.parametrize(("year", "month", "season", "days"), MONTHS)
def test_calendar_lists_the_meetings_of_the_month(year, month, season, days):
    key = f"{year}-{month:02d}"
    body = pages.calendar(year, month, [pages.meeting(d, v, s, f) for d, v, s, f in days])
    parsed = parse_fixture.parse(raw(body, key))

    assert parsed["meetings"] == [
        row(date(year, month, d), v, season, SESSION[s], SURFACE[f]) for d, v, s, f in days
    ]


@pytest.mark.parametrize(("year", "month", "season", "days"), MONTHS)
def test_calendar_rows_use_exactly_the_spec_columns(year, month, season, days):
    # the store's schema declares the same columns (a drift there is a spec change, not this ticket)
    assert set(schemas.TABLES["meetings"].columns) == SPEC_COLUMNS
    # icons listed surface first: their order within a cell does not matter
    cells = [pages.meeting(d, f, s, v) for d, v, s, f in days]
    body = pages.calendar(year, month, cells)
    meetings = parse_fixture.parse(raw(body, f"{year}-{month:02d}"))["meetings"]

    assert len(meetings) == len(days)
    for m in meetings:
        assert set(m) == SPEC_COLUMNS
        assert isinstance(m["race_date"], date)
        assert m["meeting_id"] == f"hkjc:{m['race_date'].isoformat()}:{m['venue']}"
        assert m["season"] == season
    assert [(m["race_date"], m["venue"]) for m in meetings] == sorted(
        (m["race_date"], m["venue"]) for m in meetings
    )
    assert len({m["meeting_id"] for m in meetings}) == len(meetings)


@pytest.mark.parametrize(
    ("title", "key", "url_month", "expected"),
    [
        # the title wins over a key and a URL that both name another month
        ("9/2026", "1999-01", (1999, 1), (2026, 9)),
        ("2/2024", "2024-03", (2024, 3), (2024, 2)),
        ("12/2019", "2020-01", (2020, 1), (2019, 12)),
        # no title: the cache key, then the URL
        (None, "2026-09", (1999, 1), (2026, 9)),
        (None, "not-a-month", (2024, 2), (2024, 2)),
    ],
)
def test_the_page_title_decides_the_month_not_the_cache_key(title, key, url_month, expected):
    year, month = expected
    # days 1-28 exist in every month, so a month taken from the wrong place still parses
    days = [(1, "ST", "D", "Turf"), (14, "HV", "N", "Turf"), (28, "ST", "D", "Mixed")]
    body = pages.calendar(
        year, month, [pages.meeting(d, v, s, f) for d, v, s, f in days], title=title
    )
    parsed = parse_fixture.parse(raw(body, key, pages.fixture_url(*url_month)))

    assert [m["race_date"] for m in parsed["meetings"]] == [date(year, month, d) for d, *_ in days]
    assert {m["season"] for m in parsed["meetings"]} == {
        f"{year if month >= 9 else year - 1}/{(year + 1 if month >= 9 else year) % 100:02d}"
    }


def test_without_any_month_there_are_no_meetings():
    body = pages.calendar(2026, 9, [pages.meeting(23, "HV", "N", "Turf")], title=None)
    parsed = parse_fixture.parse(raw(body, "whatever", f"{pages.HOST}/x"))
    assert parsed["meetings"] == []


@pytest.mark.parametrize(
    ("year", "month", "abandoned_day", "races"),
    [
        (2019, 11, 13, 8),  # race lines printed, the meeting abandoned on the night
        (2026, 9, 20, 0),  # cancelled in advance: the cell stays, with no race lines
        (2012, 2, 29, 0),
    ],
)
def test_an_abandoned_meeting_is_still_listed(year, month, abandoned_day, races):
    others = [d for d in (3, 10, 17, 24) if d != abandoned_day]
    cells = [pages.meeting(d, "ST", "D", "Turf") for d in others]
    cells.append(pages.meeting(abandoned_day, "HV", "N", "Turf", races=races))
    body = pages.calendar(year, month, cells)
    meetings = parse_fixture.parse(raw(body, f"{year}-{month:02d}"))["meetings"]

    assert sorted(m["race_date"].day for m in meetings) == sorted([*others, abandoned_day])
    (abandoned,) = [m for m in meetings if m["race_date"] == date(year, month, abandoned_day)]
    assert (abandoned["venue"], abandoned["day_night"], abandoned["status"]) == (
        "HV",
        "night",
        "scheduled",
    )
    # the calendar never claims an outcome: every listed meeting is scheduled
    assert {m["status"] for m in meetings} == {"scheduled"}


def test_only_cells_naming_a_local_venue_are_meetings():
    cells = [
        pages.meeting(5, "CH", "D", "Turf"),  # Conghua: on the legend, not a configured venue
        pages.meeting(6, "S1", "D", "Turf"),  # simulcast-only day
        pages.meeting(7, "D", "Turf"),  # no venue icon at all
        pages.meeting(9, "ST", "D", "AWT"),
        pages.meeting(12, "hv", "n", "turf"),  # alt text in lower case
    ]
    body = pages.calendar(2026, 9, cells)
    assert summary(parse_fixture.parse(raw(body, "2026-09"))) == [
        (date(2026, 9, 9), "ST", "day", "awt"),
        (date(2026, 9, 12), "HV", "night", "turf"),
    ]


def test_session_and_surface_come_from_alt_text_or_the_icon_file_name():
    stems_only = pages.Day(
        16, (pages.Icon(None, "hv"), pages.Icon("", "night"), pages.Icon(None, "turf")), 8
    )
    cells = [
        pages.meeting(12, "ST", "T", "Mixed"),  # twilight
        stems_only,
        pages.meeting(19, "HV"),  # no session or surface icon: unknown, not guessed
        pages.meeting(26, "ST", "AWT"),
    ]
    body = pages.calendar(2026, 9, cells)
    assert summary(parse_fixture.parse(raw(body, "2026-09"))) == [
        (date(2026, 9, 12), "ST", "twilight", "mixed"),
        (date(2026, 9, 16), "HV", "night", "turf"),
        (date(2026, 9, 19), "HV", None, None),
        (date(2026, 9, 26), "ST", None, "awt"),
    ]


def test_impossible_days_and_junk_never_raise():
    # September has 30 days: cells numbered 31 and 0 are not meetings
    cells = [pages.meeting(31, "ST", "D", "Turf"), pages.meeting(0, "HV", "N", "Turf")]
    body = pages.calendar(2026, 9, cells)
    assert parse_fixture.parse(raw(body, "2026-09"))["meetings"] == []
    for junk in (b"", b"\xff\xfe not html at all", pages.page('<td class="calendar">??</td>')):
        assert parse_fixture.parse(raw(junk, "2026-09"))["meetings"] == []
