"""Entries page (View=All) -> entries: an early, best-effort look at the fields (ported from
HKTrack's `sources/hkjc/parse_entries.py`; HKTrack research 03, section 1.2).

The page is a sequence of sections. Each one is a label/value header block (race name, class,
distance, section number, rating range), an entries table and a "Reserve :" table that has an
extra leading "No." column. Columns are mapped from the header text. Remarks are kept as printed
and decoded: `+` trump card, `*` priority to run, `#` late entry; the P column is the order of
trainer preference. The provisional race code (P10..P60) only appears in the navigation links at
the foot of the page, so it is null when that block is missing.

Pure: parses bytes already fetched through `horsetrack.core.ingest`; the page is untrusted data.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from bs4 import Tag

from horsetrack.core.config import load_source
from horsetrack.core.ingest.types import ParsedPage, RawPage
from horsetrack.countries.hongkong.hkjc.parse_common import (
    COUNTRY,
    SOURCE,
    cell_text,
    clean,
    header_index,
    href_param,
    meeting_id,
    person_key,
    soup,
    to_int,
)

PARSER_VERSION = "1"
JURISDICTION = SOURCE

# header text (as normalised by parse_common.header_index) -> our field
_HEADERS = {
    "reserve_no": ("no",),
    "horse": ("horse (brand no)", "horse"),
    "trainer": ("trainer",),
    "card_wt": ("wt",),
    "wfa": ("wfa",),
    "rating": ("rtg",),
    "rating_change": ("rtg.+/-",),
    "preference": ("p",),
    "remarks": ("remarks",),
}

_MEETING_KEY = re.compile(r"(\d{4})-(\d{2})-(\d{2})_([A-Za-z0-9]+)")
_MEETING_LABEL = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})\s*(.*)")
_DISTANCE = re.compile(r"(\d{3,4})\s*m", re.I)
_BAND = re.compile(r"(\d+)\s*-\s*(\d+)")
_BAND_OPEN = re.compile(r"(\d+)\s*\+")
_COURSE_CONFIG = re.compile(r"[\"“]\s*([^\"”]+?)\s*[\"”]")
_BRAND = re.compile(r"\(([^()]*)\)\s*$")
_SIGNED = re.compile(r"[-+]?\d+")
_SAFE_ID = re.compile(r"[A-Za-z0-9_]{1,40}")


def parse(page: RawPage) -> ParsedPage:
    doc = soup(page.body)
    nav = _navigation(doc)
    rows: list[dict[str, Any]] = []
    section: dict[str, Any] | None = None
    for node in doc.select("div:has(> ul.ulDiv), table.section_table"):
        if node.name == "div":
            section = _section(node, page, nav)
        elif section is not None and section["meeting_id"] is not None:
            rows += _rows(node, section, page.fetched_utc)
    return {"entries": rows}


# --- section headers -----------------------------------------------------------------------------


def _venue_names() -> dict[str, str]:
    venues = load_source(COUNTRY, SOURCE).venues
    return {(venue.name or "").lower(): code for code, venue in venues.items()}


def _meeting_id(label: str, key: str) -> str | None:
    """'23/09/2026 Happy Valley' -> 'hkjc:2026-09-23:HV'. The page label wins; the cache key fills
    gaps."""
    race_date = venue = None
    m = _MEETING_LABEL.search(label)
    if m:
        try:
            race_date = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            race_date = None
        place = m.group(4).lower()
        venue = next(
            (code for name, code in _venue_names().items() if name and place.startswith(name)),
            None,
        )
    from_key = _MEETING_KEY.match(key or "")
    if from_key and (race_date is None or venue is None):
        try:
            key_date = date(int(from_key.group(1)), int(from_key.group(2)), int(from_key.group(3)))
        except ValueError:
            key_date = None
        race_date, venue = race_date or key_date, venue or from_key.group(4)
    if race_date is None or venue is None:
        return None
    return meeting_id(race_date, venue)


def _labelled(items: list[str]) -> dict[str, str]:
    """['Track :', 'Turf', 'Meeting No :', '5', ...] -> {'track': 'Turf', 'meeting no': '5'}."""
    out: dict[str, str] = {}
    for label, value in zip(items, items[1:], strict=False):
        if label.endswith(":") and not value.endswith(":"):
            out[label.rstrip(": ").lower()] = value
    return out


def _section(
    block: Tag, page: RawPage, nav: dict[tuple[str, int, int | None], str]
) -> dict[str, Any]:
    items = [cell_text(li) for li in block.select("ul.ulDiv > li")]
    fields = _labelled(items)
    race_name, race_class = fields.get("race name") or None, fields.get("race class") or None
    distance = _DISTANCE.search(fields.get("distance", ""))
    distance_m = int(distance.group(1)) if distance else None
    section_no = to_int(fields.get("sec./div. no"))
    band_raw = fields.get("rating range", "")
    closed, open_ended = _BAND.search(band_raw), _BAND_OPEN.search(band_raw)
    rating_hi = rating_lo = None
    if closed:
        a, b = int(closed.group(1)), int(closed.group(2))
        rating_hi, rating_lo = max(a, b), min(a, b)
    elif open_ended:
        rating_lo = int(open_ended.group(1))
    track = fields.get("track", "").upper()
    config = _COURSE_CONFIG.search(fields.get("course", ""))
    surface = (
        "awt"
        if "ALL WEATHER" in track or track == "AWT"
        else ("turf" if track.startswith("TURF") else None)
    )
    label = " ".join(
        part
        for part in (
            race_class,
            f"{distance_m}m" if distance_m else None,
            f"Section {section_no}" if section_no else None,
            race_name,
            f"(Ratings {clean(band_raw)})" if clean(band_raw) else None,
        )
        if part
    )
    return {
        "meeting_id": _meeting_id(items[0] if items else "", page.key),
        "section_label": label or None,
        "prov_race_no": nav.get(((race_name or "").lower(), section_no or 0, distance_m)),
        "section_no": section_no,
        "race_name": race_name,
        "race_class": race_class,
        "distance_m": distance_m,
        "rating_hi": rating_hi,
        "rating_lo": rating_lo,
        "surface": surface,
        "course_config": (
            config.group(1).upper().replace(" ", "") if config and surface == "turf" else None
        ),
    }


def _navigation(doc: Tag) -> dict[tuple[str, int, int | None], str]:
    """(race name, section no, distance) -> provisional race code, from the section links at the
    page foot."""
    out: dict[tuple[str, int, int | None], str] = {}
    for link in doc.find_all("a", href=True):
        href = link["href"]
        if "ProvRaceNo" not in href:
            continue
        inner = href[href.find("/") :].rstrip("')\" ;") if "ajaxNav" in href else href
        prov, section_no = href_param(inner, "provraceno"), to_int(href_param(inner, "sectionno"))
        text = cell_text(link)
        m = re.search(r"Section\s+\d+\s+(.*?)\s*(?:\(Ratings[^)]*\))?$", text)
        distance = _DISTANCE.search(text)
        if prov and re.fullmatch(r"[A-Za-z0-9]{1,8}", prov) and section_no and m:
            key = (m.group(1).lower(), section_no, int(distance.group(1)) if distance else None)
            out.setdefault(key, prov)
    return out


# --- rows ----------------------------------------------------------------------------------------


def _link_id(link: Tag | None, param: str) -> str | None:
    """Ids come from scraped hrefs and end up in join keys, URLs and cache file names: plain
    charset only."""
    value = href_param(link["href"], param) if link is not None else None
    return value if value and _SAFE_ID.fullmatch(value) else None


def _signed(text: str | None) -> int | None:
    m = _SIGNED.search(text or "")
    return int(m.group()) if m else None


def _rows(table: Tag, section: dict[str, Any], scraped: datetime | None) -> list[dict[str, Any]]:
    all_rows = table.find_all("tr")
    if not all_rows:
        return []
    header = [cell_text(c) for c in all_rows[0].find_all(["td", "th"], recursive=False)]
    columns = header_index(header)
    at = {
        field: columns[name]
        for field, names in _HEADERS.items()
        for name in names
        if name in columns
    }
    if "horse" not in at:
        return []
    is_reserve = "reserve_no" in at
    out = []
    for row in all_rows[1:]:
        cells = row.find_all("td", recursive=False)
        entry = _entry(cells, at, is_reserve)
        if entry["horse_id"] is not None or entry["horse_name"] is not None:
            out.append(
                {"jurisdiction": JURISDICTION, **section, **entry, "entries_scraped_utc": scraped}
            )
    return out


def _entry(cells: list[Tag], at: dict[str, int], is_reserve: bool) -> dict[str, Any]:
    def cell(field: str) -> Tag | None:
        index = at.get(field)
        return cells[index] if index is not None and index < len(cells) else None

    def text(field: str) -> str | None:
        value = cell_text(cell(field))
        return value if value and value != "-" else None

    horse_cell, trainer_cell = cell("horse"), cell("trainer")
    horse_link = horse_cell.find("a", href=True) if horse_cell is not None else None
    trainer_link = trainer_cell.find("a", href=True) if trainer_cell is not None else None
    horse_text = text("horse") or ""
    brand = _BRAND.search(horse_text)
    trainer_name = text("trainer")
    remarks = text("remarks")
    return {
        "horse_id": _link_id(horse_link, "horseid"),
        "horse_name": clean(_BRAND.sub("", horse_text)) or None,
        "brand_raw": (clean(brand.group(1)) or None) if brand else None,
        "trainer_key": person_key("trainer", trainer_name or ""),
        "trainer_code": _link_id(trainer_link, "trainerid"),
        "trainer_name": trainer_name,
        "card_wt_lb": to_int(text("card_wt")),
        "wfa_lb": to_int(text("wfa")),
        "rating": to_int(text("rating")),
        "rating_change": _signed(text("rating_change")),
        "trainer_preference": to_int(text("preference")),
        "remarks_raw": remarks,
        "trump_card": "+" in (remarks or ""),
        "priority_to_run": "*" in (remarks or ""),
        "late_entry": "#" in (remarks or ""),
        "is_reserve": is_reserve,
        "reserve_no": to_int(text("reserve_no")) if is_reserve else None,
    }
