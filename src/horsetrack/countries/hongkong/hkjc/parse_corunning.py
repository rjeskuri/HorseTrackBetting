"""Comments on Running page -> `running_comments` rows, one per runner (ported from HKTrack's
`sources/hkjc/parse_corunning.py`; HKTrack research 02, section 7).

The page is the per-race source of gear as printed ("B1", "V-/B2/TT"; "--" means none -> null).
Comments are published some days after a meeting; until then every comment cell reads "No
Comments on Running information for this horse." and the page yields an EMPTY table: a race
without comments has not been parsed yet and must not look as if it had. When only some runners
lack a comment the rows are kept (their gear is real) with `comment` = null. Columns are mapped by
header text; the horse id comes from the profile link, never the brand.

Pure: parses bytes already fetched through `horsetrack.core.ingest`; the page is untrusted data.
"""

from __future__ import annotations

import re
from typing import Any

from bs4 import Tag

from horsetrack.core.ingest.types import ParsedPage, RawPage
from horsetrack.countries.hongkong.hkjc.parse_common import cell_text, soup, to_int
from horsetrack.countries.hongkong.hkjc.parse_sectional import (
    JURISDICTION,
    RaceRef,
    column_labels,
    direct_rows,
    linked_horse_id,
    merge_refs,
    none_if_blank,
    page_race_ref,
    ref_from_url,
    venue_code,
)

PARSER_VERSION = "1"

NO_COMMENTS_MARKER = "no comments on running"
_HORSE_NO = ("horseno", "hno", "no")
_HORSE = ("horsename", "horse")
_RACE_TITLE = re.compile(r"Race\s+(\d+)", re.I)


def parse(page: RawPage) -> ParsedPage:
    doc = soup(page.body)
    race_id = page_race_ref(page, _content_ref(doc)).race_id()
    found = _comment_table(doc)
    if race_id is None or found is None:
        return {"running_comments": []}
    rows = _rows(*found, race_id)
    # nothing published yet for the whole race: report no data rather than rows that look parsed
    return {"running_comments": rows if any(r["comment"] for r in rows) else []}


def _content_ref(doc: Tag) -> RaceRef:
    """The page's own Results link carries racedate / racecourse / raceno; the header text is the
    fallback ('Race 1 (275)' and a line with the venue name), and the race tabs carry the meeting
    date."""
    # only links inside the page's own content block say which race is shown: site navigation may
    # link to the latest meeting, and what is read here can veto the page (page_race_ref drops a
    # page that contradicts its key)
    scope = doc.select_one("div.corunning") or doc
    link = scope.find("a", href=re.compile("localresults", re.I))
    from_link = ref_from_url(link["href"]) if link is not None else RaceRef()
    tab = scope.find("a", href=re.compile(r"corunning\?", re.I))
    header = [cell_text(p) for p in doc.select("div.corunning div.f_fs16 p")]
    title = next((m for m in map(_RACE_TITLE.match, header) if m), None)
    from_text = RaceRef(
        ref_from_url(tab["href"]).race_date if tab is not None else None,
        next((code for code in map(venue_code, header) if code), None),
        int(title.group(1)) if title else None,
    )
    return merge_refs(from_link, from_text)


def _comment_table(doc: Tag) -> tuple[Tag, int, list[str]] | None:
    """(table, header row index, labels) of the first table that has Horse No and Comment
    columns."""
    for table in doc.find_all("table"):
        for at, row in enumerate(direct_rows(table)):
            labels = column_labels(row)
            if "comment" in labels and any(name in labels for name in _HORSE_NO):
                return table, at, labels
    return None


def _column(labels: list[str], names: tuple[str, ...]) -> int | None:
    return next((labels.index(name) for name in names if name in labels), None)


def _rows(table: Tag, header_at: int, labels: list[str], race_id: str) -> list[dict[str, Any]]:
    no_col, horse_col = _column(labels, _HORSE_NO), _column(labels, _HORSE)
    gear_col, comment_col = _column(labels, ("gear",)), labels.index("comment")
    out: list[dict[str, Any]] = []
    for row in direct_rows(table)[header_at + 1 :]:
        cells = row.find_all("td", recursive=False)
        horse_no = to_int(cell_text(cells[no_col])) if len(cells) == len(labels) else None
        if horse_no is None:  # spacer or malformed row
            continue
        comment = none_if_blank(cell_text(cells[comment_col]))
        if comment is not None and NO_COMMENTS_MARKER in comment.lower():
            comment = None
        gear = none_if_blank(cell_text(cells[gear_col])) if gear_col is not None else None
        out.append(
            {
                "jurisdiction": JURISDICTION,
                "race_id": race_id,
                "horse_id": linked_horse_id(cells[horse_col]) if horse_col is not None else None,
                "horse_no": horse_no,
                "gear_raw": gear,
                "comment": comment,
            }
        )
    return out
