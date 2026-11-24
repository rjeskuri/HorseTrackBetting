"""Monthly fixture calendar -> `meetings` rows (ported from HKTrack's
`sources/hkjc/parse_fixture.py`).

The fixture is the SCHEDULED programme: a listed meeting may later be abandoned or cancelled (a
cancelled day stays a calendar cell with zero race lines), so every row has status "scheduled" and
the results pages decide what actually happened. Race counts are not read here either; the race
tabs on the results page are the truth.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from datetime import date
from pathlib import PurePosixPath
from typing import Any

from bs4 import Tag

from horsetrack.core.config import load_source
from horsetrack.core.ingest.types import ParsedPage, RawPage
from horsetrack.countries.hongkong.hkjc.parse_common import (
    COUNTRY,
    SOURCE,
    cell_text,
    clean,
    href_param,
    meeting_id,
    season_of,
    soup,
    to_int,
)

PARSER_VERSION = "1"

# icon alt text, with the image file stem as a fallback
_SESSIONS = {
    "d": "day",
    "day": "day",
    "n": "night",
    "night": "night",
    "t": "twilight",
    "dusk": "twilight",
    "twilight": "twilight",
}
_SURFACES = {"turf": "turf", "mixed": "mixed", "awt": "awt"}

_TITLE = re.compile(r"([0-9]{1,2})\s*/\s*([0-9]{4})")
_KEY = re.compile(r"([0-9]{4})-([0-9]{2})")


def parse(page: RawPage) -> ParsedPage:
    meta = {
        "page_type": page.page_type,
        "key": page.key,
        "url": page.url,
        "parser_version": PARSER_VERSION,
    }
    doc = soup(page.body)
    month = _page_month(doc, page)
    if month is None:
        return {"_meta": [meta], "meetings": []}
    venues = set(load_source(COUNTRY, SOURCE).venues)
    meetings = []
    for cell in doc.select("td.calendar"):
        row = _meeting(cell, month, venues)
        if row is not None:
            meetings.append(row)
    meetings.sort(key=lambda m: (m["race_date"], m["venue"]))
    return {"_meta": [meta], "meetings": meetings}


def _page_month(doc: Tag, page: RawPage) -> tuple[int, int] | None:
    """(year, month) from the calendar's own title cell ('9/2026'); the cache key and URL are
    fallbacks, in that order."""
    for cell in doc.select("div.fixture_tab thead td, td[colspan='7']"):
        m = _TITLE.fullmatch(cell_text(cell))
        if m and 1 <= int(m.group(1)) <= 12:
            return int(m.group(2)), int(m.group(1))
    m = _KEY.fullmatch(page.key or "")
    if m and 1 <= int(m.group(2)) <= 12:
        return int(m.group(1)), int(m.group(2))
    year, month = to_int(href_param(page.url, "calyear")), to_int(href_param(page.url, "calmonth"))
    if year and month and 1 <= month <= 12:
        return year, month
    return None


def _icon_tokens(cell: Tag) -> list[str]:
    """Lower-cased alt text (or file stem) of the venue / session / surface icons in the cell's
    header."""
    header = cell.find("p")
    icons = cell.select("span.f_fr img") or (header.find_all("img") if header else [])
    tokens = []
    for img in icons:
        token = clean(img.get("alt")) or PurePosixPath(img.get("src") or "").stem
        if token:
            tokens.append(token.lower())
    return tokens


def _meeting(cell: Tag, month: tuple[int, int], venues: Collection[str]) -> dict[str, Any] | None:
    day = to_int(cell_text(cell.select_one("span.f_fl")))
    try:
        race_date = date(month[0], month[1], day or 0)
    except ValueError:  # scraped input: an impossible day is not a meeting
        return None
    tokens = _icon_tokens(cell)
    venue = next((t.upper() for t in tokens if t.upper() in venues), None)
    if venue is None:  # simulcast-only day or a venue that is not a configured local course
        return None
    return {
        "jurisdiction": SOURCE,
        "meeting_id": meeting_id(race_date, venue),
        "race_date": race_date,
        "venue": venue,
        "season": season_of(race_date),
        "day_night": next((_SESSIONS[t] for t in tokens if t in _SESSIONS), None),
        "surface_mix": next((_SURFACES[t] for t in tokens if t in _SURFACES), None),
        "status": "scheduled",
    }
