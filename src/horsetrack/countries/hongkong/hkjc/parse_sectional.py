"""Per-horse sectional times page -> `sectionals` rows, one per (runner, section) (ported from
HKTrack's `sources/hkjc/parse_sectional.py`).

Each section cell holds the running position (`span.f_fl`), the margin behind the leader (`i`) and
the section time (second `p`); they are read from their own elements because the cell text
concatenates them ("1" + "3/4"). The leader's margin cell prints its LEAD over the second horse, so
it is forced to 0 and the printed value is kept in `section_margin_raw`. 2026-era pages add 200 m
sub-splits inside the time `p` (`span.color_blue2`); they go to `sub_split_times_s` and never
replace the section time.

Also home to the small helpers shared with parse_corunning and parse_horse: which race a per-race
page is about (cache key, then URL, then page content), header-label mapping, a table's own rows,
and linked horse ids.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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
    href_param,
    lbw_to_lengths,
    meeting_id,
    soup,
    time_to_seconds,
    to_int,
)
from horsetrack.countries.hongkong.hkjc.urls import parse_race_key

PARSER_VERSION = "2"
JURISDICTION = SOURCE

# date formats seen in HKJC query strings: sectionals, corunning, results
_URL_DATE_FORMATS = ("%d/%m/%Y", "%Y%m%d", "%Y/%m/%d", "%Y-%m-%d")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_PLAIN_SECONDS = re.compile(r"\d+(?:\.\d+)?")
_MEETING = re.compile(r"Meeting Date\s*:\s*(\d{1,2}/\d{1,2}/\d{4})\s*,?\s*([A-Za-z ]*)")
_RACE_DIV = re.compile(r"Race(\d+)", re.I)
_SAFE_ID = re.compile(r"[A-Za-z0-9_]{1,40}")
# scraped colspans are untrusted: an absurd value must not allocate an absurd label list
_MAX_COLSPAN = 64


# --- shared with parse_corunning / parse_horse ---------------------------------------------------


@dataclass(frozen=True)
class RaceRef:
    """Race coordinates; any part may be unknown until key, URL and page content are merged."""

    race_date: date | None = None
    venue: str | None = None
    race_no: int | None = None

    def race_id(self) -> str | None:
        """hkjc:<YYYY-MM-DD>:<VENUE>:<NN>, or None while any part is unknown."""
        if self.race_date is None or not self.venue or not self.race_no:
            return None
        return f"{meeting_id(self.race_date, self.venue)}:{int(self.race_no):02d}"


def squash(text: str | None) -> str:
    """Header text -> comparison key: 'RC /Track/ Course' -> 'rctrackcourse', 'Horse<br>No' ->
    'horseno'."""
    return _NON_ALNUM.sub("", clean(text).lower())


def none_if_blank(text: str | None) -> str | None:
    """HKJC prints '--' / '---' for 'nothing here'; those and empty strings become null."""
    t = clean(text)
    return t if t.strip("-") else None


def linked_horse_id(cell: Tag | None) -> str | None:
    """The stable horse id from a cell's profile link (never the printed brand). Ids end up in join
    keys, URLs and cache file names, so anything outside a plain charset is dropped."""
    link = cell.find("a", href=re.compile("horseid", re.I)) if cell is not None else None
    horse_id = href_param(link["href"], "horseid") if link is not None else None
    return horse_id if horse_id and _SAFE_ID.fullmatch(horse_id) else None


def direct_rows(table: Tag) -> list[Tag]:
    """Rows of this table only: cells such as the horse page's Video Replay column nest whole
    tables."""
    return [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]


def column_labels(row: Tag) -> list[str]:
    """Squashed header labels with colspans expanded, so they line up with the data cells."""
    labels: list[str] = []
    for cell in row.find_all(["td", "th"], recursive=False):
        labels.extend([squash(cell_text(cell))] * cell_span(cell))
    return labels


def cell_span(cell: Tag) -> int:
    """A cell's colspan, clamped to 1.._MAX_COLSPAN."""
    return min(max(to_int(str(cell.get("colspan") or "")) or 1, 1), _MAX_COLSPAN)


def parse_url_date(text: str | None) -> date | None:
    t = clean(text)
    for fmt in _URL_DATE_FORMATS:
        try:
            return datetime.strptime(t, fmt).date()
        except ValueError:
            continue
    return None


def venue_code(text: str | None) -> str | None:
    """A configured venue code ('HV') or venue name ('Happy Valley', anywhere in `text`) -> code."""
    t = clean(text)
    venues = load_source(COUNTRY, SOURCE).venues
    if t.upper() in venues:
        return t.upper()
    low = t.lower()
    names = {code: (venue.name or "").lower() for code, venue in venues.items()}
    return next((code for code, name in names.items() if name and name in low), None)


def ref_from_key(key: str | None) -> RaceRef:
    parsed = parse_race_key(key or "")
    return RaceRef(*parsed) if parsed else RaceRef()


def ref_from_url(url: str | None) -> RaceRef:
    """racedate|date, racecourse, raceno in any letter case; each endpoint has its own date
    format."""
    raw_date = href_param(url, "racedate") or href_param(url, "date")
    return RaceRef(
        parse_url_date(raw_date),
        venue_code(href_param(url, "racecourse")),
        to_int(href_param(url, "raceno")),
    )


def merge_refs(*refs: RaceRef) -> RaceRef:
    """Field-wise first known value, in priority order."""
    return RaceRef(
        next((r.race_date for r in refs if r.race_date is not None), None),
        next((r.venue for r in refs if r.venue), None),
        next((r.race_no for r in refs if r.race_no), None),
    )


def refs_conflict(wanted: RaceRef, content: RaceRef, *, race_no: bool = True) -> bool:
    """True when the page itself names a different date, venue or race than the one we asked for.
    A part that either side does not know is no evidence either way."""
    pairs: list[tuple[Any, Any]] = [
        (wanted.race_date, content.race_date),
        (wanted.venue, content.venue),
    ]
    if race_no:
        pairs.append((wanted.race_no, content.race_no))
    return any(a and b and a != b for a, b in pairs)


def page_race_ref(page: RawPage, content: RaceRef, *, check_race_no: bool = True) -> RaceRef:
    """Cache key first, then the URL (requested, then final when the fetcher recorded one), then
    the page.

    The key / URL say which race was ASKED for; the content says which race was SERVED. When the
    two disagree (the site answering with another meeting's page) the result is an empty RaceRef,
    so the caller yields no rows: another race's data must never be filed under this race's id."""
    final_url = page.meta.get("final_url") if isinstance(page.meta, dict) else None
    wanted = merge_refs(ref_from_key(page.key), ref_from_url(page.url), ref_from_url(final_url))
    if refs_conflict(wanted, content, race_no=check_race_no):
        return RaceRef()
    return merge_refs(wanted, content)


# --- sectional page ------------------------------------------------------------------------------


def parse(page: RawPage) -> ParsedPage:
    doc = soup(page.body)
    # the race number is checked by _race_table instead: an All=True page stacks every race of
    # the meeting
    ref = page_race_ref(page, _content_ref(doc), check_race_no=False)
    race_id = ref.race_id()
    table = _race_table(doc, ref.race_no)
    if race_id is None or table is None:
        return {"sectionals": []}
    return {"sectionals": _rows(table, race_id)}


def _content_ref(doc: Tag) -> RaceRef:
    """'Meeting Date:16/09/2026, Happy Valley' plus the id of the race block (`div#Race1`)."""
    race_date, venue = None, None
    node = doc.find(string=_MEETING)
    if node is not None:
        m = _MEETING.search(clean(str(node)))
        if m:
            race_date, venue = parse_url_date(m.group(1)), venue_code(m.group(2))
    return RaceRef(race_date, venue, next(iter(_race_blocks(doc)), None))


def _race_blocks(doc: Tag) -> dict[int, Tag]:
    return {
        int(m.group(1)): div
        for div in doc.find_all("div", id=True)
        if (m := _RACE_DIV.fullmatch(div["id"]))
    }


def _race_table(doc: Tag, race_no: int | None) -> Tag | None:
    """The `All=True` variant stacks every race of the meeting in `div#Race<n>` blocks: take only
    ours. A page whose blocks do not include our race is some other race's data and yields
    nothing."""
    blocks = _race_blocks(doc)
    scope = blocks.get(race_no or 0) if blocks else doc
    return scope.select_one("table.race_table") if scope is not None else None


def _rows(table: Tag, race_id: str) -> list[dict[str, Any]]:
    rows = direct_rows(table)
    header_at = next((i for i, r in enumerate(rows) if "horseno" in column_labels(r)), None)
    if header_at is None:
        return []
    labels = column_labels(rows[header_at])
    no_col = labels.index("horseno")
    horse_col = labels.index("horse") if "horse" in labels else None
    section_cols = [i for i, lab in enumerate(labels) if "position" in lab or "sec" in lab]

    out: list[dict[str, Any]] = []
    for row in rows[header_at + 1 :]:
        cells = row.find_all("td", recursive=False)
        horse_no = to_int(cell_text(cells[no_col])) if len(cells) == len(labels) else None
        if horse_no is None:  # the 2nd/3rd header rows, spacers
            continue
        horse_id = linked_horse_id(cells[horse_col]) if horse_col is not None else None
        for section_no, col in enumerate(section_cols, start=1):
            section = _section(cells[col])
            if section is not None:
                out.append(
                    {
                        "jurisdiction": JURISDICTION,
                        "race_id": race_id,
                        "horse_id": horse_id,
                        "horse_no": horse_no,
                        "section_no": section_no,
                        **section,
                    }
                )
    return out


def _seconds(text: str) -> float | None:
    t = clean(text)
    return float(t) if _PLAIN_SECONDS.fullmatch(t) else time_to_seconds(t)


def _section(cell: Tag) -> dict[str, Any] | None:
    """One section cell -> position, margin, time. Unused sections hold only a blank image ->
    None."""
    pos = to_int(cell_text(cell.select_one("span.f_fl")))
    margin_raw = none_if_blank(cell_text(cell.find("i")))
    # the time p is the one holding neither the margin nor the position
    time_ps = (p for p in cell.find_all("p") if p.find("i") is None)
    time_p = next((p for p in time_ps if p.select_one("span.f_fl") is None), None)
    time_s, subs = None, []
    if time_p is not None:
        # the p's own text is the section time; the nested span holds the 200 m sub-splits
        time_s = _seconds("".join(str(s) for s in time_p.find_all(string=True, recursive=False)))
        subs = [
            v
            for v in (_seconds(cell_text(s)) for s in time_p.select("span.color_blue2 span"))
            if v is not None
        ]
    if pos is None and time_s is None:
        return None
    return {
        "section_pos": pos,
        "section_margin_lengths": 0.0 if pos == 1 else lbw_to_lengths(margin_raw),
        "section_time_s": time_s,
        "section_margin_raw": margin_raw,
        "sub_split_times_s": subs,
    }
