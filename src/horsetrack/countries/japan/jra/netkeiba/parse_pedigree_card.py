"""netkeiba five-run card (race.netkeiba.com/race/shutuba_past.html?race_id=<id>, UTF-8) ->
`horse_pedigree` (ported from HKTrack's `sources/jra/parse_pedigree_card.py`).

Each runner row (`tr.HorseList` inside `div.Shutuba_HorseList`, scratched runners
`tr.HorseList.Cancel` included) prints the sire (`.Horse01`), the runner's own link
(`.Horse02 a`), the dam (`.Horse03`) and the dam sire in brackets (`.Horse04`). The ancestors are
plain text, so their ids stay None. The page also carries rows that are not runners: a legend
table and the training table use `tr.HorseList` too, and every past run links the winner of that
race; only rows inside the runner table that link a horse in `.Horse02` count.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from bs4 import BeautifulSoup, Tag

from horsetrack.core.ingest.types import ParsedPage, RawPage
from horsetrack.countries.japan.jra.netkeiba import common, urls

PARSER_VERSION = "1"

RUNNER_ROW_SELECTOR = "div.Shutuba_HorseList table.Shutuba_Table tr.HorseList"
_HORSE_LINK = re.compile(r"/horse/([0-9A-Za-z]{10})/?$")
_BRACKETS = "()（）"
_ANCESTORS = (("sire", ".Horse01"), ("dam", ".Horse03"), ("dam_sire", ".Horse04"))


def runner_rows(doc: BeautifulSoup) -> list[tuple[Tag, str]]:
    """(row, horse id) of every runner on the card, in page order."""
    out = []
    for row in doc.select(RUNNER_ROW_SELECTOR):
        link = row.select_one(".Horse02 a[href]")
        m = _HORSE_LINK.search(str(link.get("href") or "").strip()) if link is not None else None
        if m:
            out.append((row, m.group(1)))
    return out


def parse(page: RawPage) -> ParsedPage:
    doc = BeautifulSoup(page.body, "lxml")
    race_id = _race_id(doc, page.key)
    rows = []
    for row, horse_id in runner_rows(doc):
        out: dict[str, Any] = {"horse_id": horse_id}
        for column, selector in _ANCESTORS:
            name = _name(row.select_one(selector))
            out[f"{column}_name"] = name
            out[f"{column}_name_norm"] = _norm(name)
            out[f"{column}_id"] = None  # printed as text, not linked
        out.update(source_race_id=race_id, scraped_utc=page.fetched_utc)
        rows.append(out)
    meta: dict[str, Any] = {
        "page_type": page.page_type,
        "key": page.key,
        "url": page.url,
        "race_id": race_id,
        "parser_version": PARSER_VERSION,
    }
    return {"horse_pedigree": rows, "_meta": [meta]}


def _race_id(doc: BeautifulSoup, key: str) -> str | None:
    """The store race id: the date from the page title, venue and race number from the key."""
    netkeiba_id = urls.jra_race_id(key)
    if netkeiba_id is None:
        return None
    race_day = common.race_date(doc)
    venue = common.venue_code(netkeiba_id)
    if race_day is None or venue is None:
        return None
    return common.store_race_id(race_day, venue, int(netkeiba_id[10:12]))


def _name(node: Tag | None) -> str | None:
    """The name as printed, without the dam sire's brackets; None when nothing is printed."""
    if node is None:
        return None
    text = common.clean(node.get_text(" "))
    if len(text) >= 2 and text[0] in _BRACKETS and text[-1] in _BRACKETS:
        text = text[1:-1].strip()
    return text or None


def _norm(name: str | None) -> str | None:
    return unicodedata.normalize("NFKC", name).casefold() if name is not None else None
