"""Shared helpers for the HKJC HTML parsers (ported from HKTrack's `sources/hkjc/parse_common.py`),
plus the source facts the URL builders and parsers both read from config. Pure functions."""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup, Tag

from horsetrack.core.config import load_source

COUNTRY = "hongkong"
SOURCE = "hkjc"

_WS = re.compile(r"\s+")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_CLAIM_SUFFIX = re.compile(r"\(\s*[-+]?\d+\s*\)")
_NON_NAME = re.compile(r"[\W_]+")  # anything but letters and digits, of any script

NO_INFO_MARKER = "No information."

# Short beaten-margin codes -> lengths. Numeric margins ("1-1/4", "3/4", "10") are worked out
# arithmetically.
LBW_CODES = {
    "-": 0.0,
    "DH": 0.0,
    "N": 0.3,  # NECK on HKJC pages (the most common code); a nose is printed "NOSE"
    "NOSE": 0.05,
    "SH": 0.1,  # short head
    "HD": 0.2,  # head
    "H": 0.2,
    "NK": 0.3,  # neck
    "ML": 30.0,  # many lengths
    "TO": 99.0,  # tailed off
}


def season_start_month() -> int:
    """The month a racing season starts in (hkjc.yaml `season.start_month`)."""
    season = load_source(COUNTRY, SOURCE).season
    if season is None:
        raise ValueError(f"{COUNTRY}/{SOURCE}.yaml has no season.start_month")
    return season.start_month


def season_of(race_date: date) -> str:
    """Racing season label, e.g. 2026-09-23 -> '2026/27' when seasons start in September."""
    start_month = season_start_month()
    start_year = race_date.year if race_date.month >= start_month else race_date.year - 1
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def meeting_id(race_date: date, venue: str) -> str:
    """hkjc:<YYYY-MM-DD>:<VENUE>, the store's meeting id."""
    return f"{SOURCE}:{race_date.isoformat()}:{venue}"


def race_id(meeting: str, race_no: int) -> str:
    """<meeting_id>:<NN>, the store's race id."""
    return f"{meeting}:{int(race_no):02d}"


def normalise_name(name: str) -> str:
    """NFKC (full- and half-width forms fold together), drop apprentice-claim suffixes like
    '(-7)', casefold, keep letters and digits of any script; everything else becomes single
    spaces (HKTrack's `core.ids.normalise_name`)."""
    s = _CLAIM_SUFFIX.sub(" ", unicodedata.normalize("NFKC", name or "")).casefold()
    return _NON_NAME.sub(" ", s).strip()


def person_key(role: str, name: str) -> str | None:
    """Join key for jockeys and trainers, e.g. 'jockey:a lima bravo'. Old pages print retired
    people without any id, so the name is the key (HKTrack's `core.ids.person_key`)."""
    norm = normalise_name(name)
    return f"{role}:{norm}" if norm else None


def soup(body: bytes | str) -> BeautifulSoup:
    return BeautifulSoup(body, "lxml")


def clean(text: str | None) -> str:
    """Collapse whitespace (incl. non-breaking spaces) and strip."""
    return _WS.sub(" ", (text or "").replace("\xa0", " ")).strip()


def cell_text(node: Tag | None) -> str:
    return clean(node.get_text(" ")) if node is not None else ""


def to_int(text: str | None) -> int | None:
    m = _NUM.search((text or "").replace(",", ""))
    return int(float(m.group())) if m else None


def to_float(text: str | None) -> float | None:
    m = _NUM.search((text or "").replace(",", ""))
    return float(m.group()) if m else None


def time_to_seconds(text: str | None) -> float | None:
    """'1:09.85' | '1.09.85' | '0:57.4' | '23.61' -> seconds."""
    t = clean(text)
    if not t or not any(ch.isdigit() for ch in t):
        return None
    parts = re.split(r"[:.]", t)
    try:
        if len(parts) == 3:
            return int(parts[0]) * 60 + int(parts[1]) + float(f"0.{parts[2]}")
        if len(parts) == 2:
            return int(parts[0]) + float(f"0.{parts[1]}")
        return float(parts[0])
    except ValueError:
        return None


def lbw_to_lengths(text: str | None) -> float | None:
    """'-' -> 0, 'SH' -> 0.1, '1-1/4' -> 1.25, '3/4' -> 0.75, '10' -> 10. Unknown codes -> None
    (keep the raw text)."""
    t = clean(text).upper()
    if not t:
        return None
    if t in LBW_CODES:
        return LBW_CODES[t]
    m = re.fullmatch(r"(?:(\d+)-)?(\d+)/(\d+)", t)
    if m:
        whole = int(m.group(1)) if m.group(1) else 0
        return whole + int(m.group(2)) / int(m.group(3))
    m = re.fullmatch(r"\d+(?:\.\d+)?", t)
    return float(t) if m else None


def href_param(href: str | None, name: str) -> str | None:
    """Case-insensitive query-parameter lookup:
    href_param('/horse?HorseId=HK_2023_J087', 'horseid') -> 'HK_2023_J087'."""
    if not href:
        return None
    for key, values in parse_qs(urlparse(href).query).items():
        if key.lower() == name.lower() and values:
            return values[0]
    return None


def header_index(header_cells: list[str]) -> dict[str, int]:
    """Map normalised header text -> column index, so parsers survive era-specific column sets."""
    return {clean(h).lower().rstrip("."): i for i, h in enumerate(header_cells)}
