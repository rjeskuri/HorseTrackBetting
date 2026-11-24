"""Monthly calendar (race.netkeiba.com/top/calendar.html) -> `race_days`, one row per JRA race
day (ported from HKTrack's `sources/jra/parse_fixture.py`).

A race day is a cell of `table.Calendar_Table` linking `race_list.html?kaisai_date=YYYYMMDD`.
Venues and race counts are not read here; the race list of the day is the truth.
"""

from __future__ import annotations

import re
from datetime import date

from bs4 import BeautifulSoup

from horsetrack.core.ingest.types import ParsedPage, RawPage

PARSER_VERSION = "2"

CALENDAR_SELECTOR = "table.Calendar_Table"
_KAISAI_DATE = re.compile(r"kaisai_date=([0-9]{4})([0-9]{2})([0-9]{2})(?![0-9])")


def race_dates(soup: BeautifulSoup) -> list[date]:
    """Distinct race days linked from the calendar table, in date order."""
    found: set[date] = set()
    for table in soup.select(CALENDAR_SELECTOR):
        for link in table.select("a[href]"):
            m = _KAISAI_DATE.search(str(link.get("href") or ""))
            if not m:
                continue
            try:
                found.add(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            except ValueError:  # scraped input: an impossible date is not a race day
                continue
    return sorted(found)


def parse(page: RawPage) -> ParsedPage:
    days = race_dates(BeautifulSoup(page.body, "lxml"))
    meta = {
        "page_type": page.page_type,
        "key": page.key,
        "url": page.url,
        "parser_version": PARSER_VERSION,
    }
    return {"_meta": [meta], "race_days": [{"race_date": d} for d in days]}
