"""netkeiba monthly calendar -> `race_days` (ticket #51, ported from HKTrack's
`sources/jra/parse_fixture.py`).

Contract pinned here (the coder follows it):
- `horsetrack.countries.japan.jra.netkeiba.parse_fixture.parse(page: RawPage) -> ParsedPage`
  (`horsetrack.core.ingest.types`), pure, no network.
- `race_days`: one row `{"race_date": date}` per distinct day linked as
  `race_list.html?kaisai_date=YYYYMMDD` from inside `table.Calendar_Table`, in date order.
  Links outside that table, impossible dates and longer digit runs are not race days.
- `_meta`: one row naming the page's `page_type`, `key` and `url`.

Pages are synthetic (netkeiba_pages.py); expected days are written out here.
"""

from __future__ import annotations

from datetime import date

import netkeiba_pages as pages
import pytest

from horsetrack.core.ingest.types import RawPage

# isort: split
# (its own block: ruff ranks a module that does not exist yet as third-party)
from horsetrack.countries.japan.jra.netkeiba import parse_fixture


def raw(body: bytes, key: str) -> RawPage:
    year, month = key.split("-")
    url = f"{pages.RACE}/top/calendar.html?year={int(year)}&month={int(month)}"
    return RawPage("jra", "fixture", key, url, body)


def race_days(body: bytes, key: str) -> list[date]:
    parsed = parse_fixture.parse(raw(body, key))
    return [row["race_date"] for row in parsed["race_days"]]


@pytest.mark.parametrize(
    ("year", "month", "days"),
    [
        (2025, 11, [1, 2, 8, 9, 15, 16, 22, 23, 24, 29, 30]),  # a Monday holiday included
        (2010, 1, [5, 9, 10, 11, 16, 17, 23, 24, 30, 31]),
        (2024, 2, [3, 4, 10, 11, 12, 17, 18, 24, 25]),  # leap February, 29th not a race day
        (2020, 2, [29]),  # a leap day alone
        (2026, 12, [6, 31]),  # first and last cell of the month
    ],
)
def test_calendar_lists_the_race_days_of_the_month(year, month, days):
    key = f"{year}-{month:02d}"
    # a day cell links its race list once per venue; the calendar lists each day once
    body = pages.calendar(year, month, reversed(days))
    parsed = parse_fixture.parse(raw(body, key))

    assert parsed["race_days"] == [{"race_date": date(year, month, d)} for d in days]
    (meta,) = parsed["_meta"]
    assert (meta["page_type"], meta["key"], meta["url"]) == ("fixture", key, raw(body, key).url)


def test_only_well_formed_links_inside_the_calendar_table_are_race_days():
    hrefs = [
        "../top/race_list.html?kaisai_date=20251131",  # 31 November: no such day
        "../top/race_list.html?kaisai_date=20251301",  # month 13
        "../top/race_list.html?kaisai_date=202511051",  # nine digits: not a date
        "../top/race_list.html?kaisai_date=2025111",  # seven digits
        "../top/race_list.html?kaisai_date=２０２５１１０６",  # full-width digits
        "../top/race_list.html?kaisai_date=20251107&rf=calendar",  # a query after the date: kept
        "../top/race_list.html?year=2025&month=11",
        "#",
    ]
    body = pages.calendar(2025, 11, [2, 23], extra_hrefs=hrefs)
    # the decoy box outside the table links a day of the next year: not a race day
    assert b"kaisai_date=20260105" in body
    assert race_days(body, "2025-11") == [date(2025, 11, 2), date(2025, 11, 7), date(2025, 11, 23)]


@pytest.mark.parametrize(("year", "month"), [(2025, 11), (2011, 3), (2030, 8)])
def test_a_month_without_meetings_is_empty(year, month):
    key = f"{year}-{month:02d}"
    body = pages.calendar(year, month, [])
    parsed = parse_fixture.parse(raw(body, key))
    assert parsed["race_days"] == []
    assert [m["key"] for m in parsed["_meta"]] == [key]

    # a page with no calendar table at all lists no day either, even if it links one
    stray = pages.page(f'<a href="{pages.kaisai_href(date(year, month, 1))}">x</a>')
    assert race_days(stray, key) == []
