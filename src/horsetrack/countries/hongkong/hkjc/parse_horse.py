"""HKJC horse profile page -> `horses` (static attributes only) + `horse_runs` (the form table)
(ported from HKTrack's `sources/hkjc/parse_horse.py`).

`horses` deliberately ignores everything on the profile block that is an as-of-scrape value
(current / last / start-of-season rating, stakes, record, current trainer, owner, age, location):
using any of it as a feature is look-ahead leakage. Rating-at-run, trainer-at-run and gear come
from `horse_runs`.

Form-table rules:
- columns are mapped by header text, and only the table's own rows are read (Video Replay cells
  nest tables);
- a row is a run only when its Date cell is a date, which drops the season separator rows
  ("26/27 Season");
- a run is LOCAL only when the first token of RC/Track/Course is a configured venue code (ST, HV);
  anything else (overseas track codes) is dropped, whatever its Race Index cell links to;
- withdrawals are listed as rows (Pla. "WV" / "WV-A", "--" in the result columns) and are kept,
  with nulls, so downstream can filter on `placing_raw`;
- on a WINNING run the LBW cell holds the winning margin, not beaten lengths: `lbw_raw` is kept as
  printed and never converted here;
- active pages print DD/MM/YY, retired (/otherhorse) pages DD/MM/YYYY. The Race Index link carries
  the 4-digit year and settles the century when it names the same day; otherwise the POSIX pivot
  applies (69-99 -> 19xx).

`race_id` is built from the Race Index link when it agrees with the row's own date and venue (null
otherwise). The documented join key stays (horse_id, race_date).
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date
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
    race_id,
    soup,
    time_to_seconds,
    to_float,
    to_int,
)
from horsetrack.countries.hongkong.hkjc.parse_sectional import (
    JURISDICTION,
    RaceRef,
    cell_span,
    column_labels,
    direct_rows,
    none_if_blank,
    ref_from_url,
    squash,
)

PARSER_VERSION = "1"

RETIRED_ROUTE = "otherhorse"
_CENTURY_PIVOT = 69  # two-digit years: 69-99 -> 1969-1999, 00-68 -> 2000-2068 (POSIX %y)
_RUN_DATE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})")
_SAFE_ID = re.compile(r"[A-Za-z0-9_]{1,40}")
_HORSE_ID_IN_HREF = re.compile(r"horseid=([A-Za-z0-9_]{1,40})", re.I)
# "BRAVE HEART (CK277) (Retired)" -> "BRAVE HEART"
_TRAILING_PARENS = re.compile(r"(?:\s*\([^()]*\))+\s*$")
_LEADING_ZEROS = re.compile(r"^0+(?=\d)")
_POSITION = re.compile(r"[0-9]{1,2}")  # ASCII only: str.isdigit() accepts what int() cannot read
_OVERSEAS = "overseas"
_QUOTES = "\"'“”‘’ "  # straight and curly quotes around the course letter
_SEXES = {"gelding", "horse", "mare", "colt", "filly", "rig"}

# output field -> accepted header labels (squashed: lowercase alphanumerics only)
_RUN_HEADERS: dict[str, tuple[str, ...]] = {
    "race_index": ("raceindex",),
    "placing": ("pla", "placing"),
    "date": ("date",),
    "course": ("rctrackcourse", "rctrack", "racecourse"),
    "distance": ("dist", "distance"),
    "going": ("g", "going"),
    "race_class": ("raceclass", "class"),
    "draw": ("dr", "draw"),
    "rating": ("rtg", "rating"),
    "trainer": ("trainer",),
    "jockey": ("jockey",),
    "lbw": ("lbw",),
    "win_odds": ("winodds",),
    "actual_wt": ("actwt", "actualwt"),
    "running_positions": ("runningposition", "runningpositions"),
    "finish_time": ("finishtime",),
    "horse_wt": ("declarhorsewt", "declaredhorsewt"),
    "gear": ("gear",),
}


def parse(page: RawPage) -> ParsedPage:
    doc = soup(page.body)
    horse_id = _horse_id(page, doc)
    if horse_id is None:
        return {"horses": [], "horse_runs": []}
    profile = doc.select_one("table.horseProfile")
    horses = [_horse(profile, horse_id, page)] if profile is not None else []
    return {"horses": horses, "horse_runs": _runs(doc, horse_id)}


def parse_run_date(text: str | None, hint: date | None = None) -> date | None:
    """'19/02/2012' | '16/09/26' -> date. `hint` (the date in the row's results link) settles the
    century of a two-digit year, but only when it names the same day; a link to another day is no
    evidence."""
    m = _RUN_DATE.fullmatch(clean(text))
    if not m:
        return None
    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if len(m.group(3)) == 2:
        if hint is not None and (hint.day, hint.month, hint.year % 100) == (day, month, year):
            year = hint.year
        else:
            year += 1900 if year >= _CENTURY_PIVOT else 2000
    try:
        return date(year, month, day)
    except ValueError:
        return None


# --- identity ------------------------------------------------------------------------------------


def _urls(page: RawPage) -> list[str]:
    final_url = page.meta.get("final_url") if isinstance(page.meta, dict) else None
    return [u for u in (page.url, final_url) if isinstance(u, str) and u]


def _horse_id(page: RawPage, doc: Tag) -> str | None:
    """Cache key first (it IS the horse id), then the URL, then the id most links on the page
    point at."""
    for candidate in (page.key, *(href_param(u, "horseid") for u in _urls(page))):
        if candidate and _SAFE_ID.fullmatch(candidate):
            return candidate
    if doc.select_one("table.horseProfile") is None:
        return None
    linked = Counter(
        m.group(1)
        for a in doc.find_all("a", href=True)
        if (m := _HORSE_ID_IN_HREF.search(a["href"]))
    )
    return linked.most_common(1)[0][0] if linked else None


# --- horses --------------------------------------------------------------------------------------


def _profile_fields(profile: Tag) -> dict[str, str]:
    """Rows of `label | : | value` -> {squashed label: value}. The first occurrence of a label
    wins."""
    fields: dict[str, str] = {}
    for row in profile.find_all("tr"):
        cells = [cell_text(td) for td in row.find_all("td", recursive=False)]
        at = cells.index(":") if ":" in cells else 0
        if 1 <= at < len(cells) - 1:
            fields.setdefault(squash(cells[at - 1]), cells[at + 1])
    return fields


def _slash_parts(value: str | None) -> list[str]:
    return [part for part in (clean(p) for p in (value or "").split("/")) if part.strip("-")]


def _colour_and_sex(fields: dict[str, str]) -> tuple[str | None, str | None]:
    """'Brown / Gelding' -> ('Brown', 'Gelding');
    'Bay / Brown / Horse' -> ('Bay / Brown', 'Horse')."""
    parts = _slash_parts(fields.get("coloursex") or fields.get("colorsex"))
    if len(parts) == 1:  # only one of the two is printed: tell them apart by vocabulary
        return (None, parts[0]) if parts[0].lower() in _SEXES else (parts[0], None)
    colour, sex = " / ".join(parts[:-1]) or None, parts[-1] if parts else None
    return none_if_blank(fields.get("colour")) or colour, none_if_blank(fields.get("sex")) or sex


def _horse(profile: Tag, horse_id: str, page: RawPage) -> dict[str, Any]:
    fields = _profile_fields(profile)
    title = cell_text(profile.select_one(".title_text"))
    # "Country of Origin / Age" on active pages, "Country of Origin" on retired ones: never keep
    # the age
    origin = next(
        (_slash_parts(v) for k, v in fields.items() if k.startswith("countryoforigin")), []
    )
    # " / 6": the country is blank, and the age is not an origin
    origin = [part for part in origin if not part.isdigit()]
    colour, sex = _colour_and_sex(fields)
    retired = (
        "(retired)" in title.lower()
        or any(RETIRED_ROUTE in u.lower() for u in _urls(page))
        or ("lastrating" in fields and "currentrating" not in fields)
    )
    return {
        "jurisdiction": JURISDICTION,
        "horse_id": horse_id,
        "name": _TRAILING_PARENS.sub("", title).strip() or None,
        "origin": origin[0] if origin else None,
        "colour": colour,
        "sex": sex,
        "import_type": none_if_blank(fields.get("importtype")),
        "sire": none_if_blank(fields.get("sire")),
        "dam": none_if_blank(fields.get("dam")),
        "dam_sire": none_if_blank(fields.get("damssire") or fields.get("damsire")),
        "is_retired": retired,
        "profile_scraped_utc": page.fetched_utc,
    }


# --- horse_runs ----------------------------------------------------------------------------------


def _form_table(doc: Tag) -> tuple[Tag, int, dict[str, int]] | None:
    """(table, header row index, field -> column) of the first table whose header has Date + Race
    Index/Course."""
    for table in doc.find_all("table"):
        for at, row in enumerate(direct_rows(table)):
            labels = column_labels(row)
            columns = {
                name: next(labels.index(h) for h in headers if h in labels)
                for name, headers in _RUN_HEADERS.items()
                if any(h in labels for h in headers)
            }
            if "date" in columns and ("race_index" in columns or "course" in columns):
                return table, at, columns
    return None


def _runs(doc: Tag, horse_id: str) -> list[dict[str, Any]]:
    found = _form_table(doc)
    if found is None:
        return []
    table, header_at, columns = found
    rows = direct_rows(table)[header_at + 1 :]
    runs = (_run(_data_cells(row), columns, horse_id) for row in rows)
    return [run for run in runs if run is not None]


def _data_cells(row: Tag) -> list[Tag]:
    """A row's cells with colspans expanded the same way the header's are, so a merged cell cannot
    shift every later column (rating, gear ...) one place to the left."""
    cells: list[Tag] = []
    for cell in row.find_all("td", recursive=False):
        cells.extend([cell] * cell_span(cell))
    return cells


def parse_rc_track_course(text: str | None) -> tuple[str | None, str | None, str | None]:
    """'HV / Turf / "C+3"' -> ('HV', 'turf', 'C+3'); 'ST / AWT' -> ('ST', 'awt', None). The venue
    is returned only when it is a configured (local) venue code."""
    parts = [clean(p) for p in clean(text).split("/")]
    venues = load_source(COUNTRY, SOURCE).venues
    venue = parts[0].upper() if parts and parts[0].upper() in venues else None
    track = parts[1].lower() if len(parts) > 1 else ""
    surface = (
        "awt"
        if track == "awt" or "all weather" in track
        else "turf"
        if track.startswith("turf")
        else None
    )
    config = (
        parts[2].strip(_QUOTES).upper().replace(" ", "")
        if len(parts) > 2 and surface != "awt"
        else ""
    )
    return venue, surface, config or None


def _run(cells: list[Tag], columns: dict[str, int], horse_id: str) -> dict[str, Any] | None:
    def cell(name: str) -> Tag | None:
        at = columns.get(name)
        return cells[at] if at is not None and at < len(cells) else None

    def text(name: str) -> str | None:
        return none_if_blank(cell_text(cell(name)))

    index_cell = cell("race_index")
    link = (
        index_cell.find("a", href=re.compile("localresults", re.I))
        if index_cell is not None
        else None
    )
    linked = ref_from_url(link["href"]) if link is not None else RaceRef()
    race_date = parse_run_date(text("date"), hint=linked.race_date)
    venue, surface, course_config = parse_rc_track_course(text("course"))
    if race_date is None or venue is None:  # season separator / overseas run
        return None
    index_href = (
        str(index_cell.a.get("href") or "")
        if index_cell is not None and index_cell.a is not None
        else ""
    )
    if _OVERSEAS in (text("race_index") or "").lower() or _OVERSEAS in index_href.lower():
        return None  # labelled overseas: not a local run whatever its course cell starts with
    same_race = linked.race_no is not None and (linked.race_date, linked.venue) == (
        race_date,
        venue,
    )
    placing = text("placing")
    positions = (text("running_positions") or "").split()
    return {
        "jurisdiction": JURISDICTION,
        "horse_id": horse_id,
        "race_date": race_date,
        "race_index": to_int(text("race_index")),
        "venue": venue,
        "surface": surface,
        "course_config": course_config,
        "distance_m": to_int(text("distance")),
        "going": text("going"),
        "race_class": text("race_class"),
        "draw": to_int(text("draw")),
        "rating": to_int(text("rating")),
        "trainer_name": text("trainer"),
        "jockey_name": text("jockey"),
        "lbw_raw": text("lbw"),
        "win_odds": to_float(text("win_odds")),
        "actual_wt_lb": to_int(text("actual_wt")),
        "running_positions": [int(t) for t in positions if _POSITION.fullmatch(t)],
        "finish_time_s": time_to_seconds(text("finish_time")),
        "declared_horse_wt_lb": to_int(text("horse_wt")),
        "gear_raw": text("gear"),
        # printed zero-padded: "01"
        "placing_raw": _LEADING_ZEROS.sub("", placing) if placing else None,
        "race_id": race_id(meeting_id(race_date, venue), linked.race_no) if same_race else None,
    }
