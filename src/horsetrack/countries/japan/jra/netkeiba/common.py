"""Helpers the netkeiba parsers share (HKTrack's `parse_results._clean`, `_race_date`, `_venue`),
kept here so the results parser (#52) reuses them. Pure functions."""

from __future__ import annotations

import re
from datetime import date

from bs4 import BeautifulSoup, Tag

from horsetrack.core.config import load_source

COUNTRY = "japan"
SOURCE = "jra"

JP_DATE = re.compile(r"([0-9]{4})年([0-9]{1,2})月([0-9]{1,2})日")


def clean(text: str | None) -> str:
    """Runs of whitespace (the no-break and ideographic spaces too) collapsed to one plain space."""
    return " ".join((text or "").split())


def text(node: Tag | None) -> str:
    return clean(node.get_text(" ")) if node is not None else ""


def race_date(doc: BeautifulSoup) -> date | None:
    """The `年月日` date of `p.smalltxt` (results page) or else the title; None when neither
    prints a valid one."""
    for node in (doc.select_one("p.smalltxt"), doc.find("title")):
        m = JP_DATE.search(text(node))
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                return None
    return None


def venue_code(netkeiba_id: str) -> str | None:
    """The jra.yaml venue code (e.g. KYO) of a race id's venue number (characters 5-6)."""
    number = netkeiba_id[4:6]
    venues = load_source(COUNTRY, SOURCE).venues
    return next((code for code, venue in venues.items() if venue.number == number), None)


def store_race_id(race_day: date, venue: str, race_no: int) -> str:
    """jra:<YYYY-MM-DD>:<VENUE>:<NN>, the same id as `jra.capture.race_id` (not imported, so a
    parser does not pull in the HTTP capture module)."""
    return f"{SOURCE}:{race_day.isoformat()}:{venue}:{int(race_no):02d}"
