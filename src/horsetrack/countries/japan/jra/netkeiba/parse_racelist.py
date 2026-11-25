"""Race list of one day (db.netkeiba.com/race/list/YYYYMMDD/, EUC-JP) -> `race_links`, one row
per JRA race (ported from HKTrack's `sources/jra/parse_racelist.py`).

The page lists central (JRA), local (NAR) and overseas races together. Only ids that pass
`urls.jra_race_id` are kept: the id becomes a URL and a cache file name, so a malformed one must
never get through.
"""

from __future__ import annotations

import re
from datetime import date
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from horsetrack.core.ingest.types import ParsedPage, RawPage
from horsetrack.countries.japan.jra.netkeiba import common, urls

PARSER_VERSION = "1"

LIST_SELECTOR = "div.race_kaisai_info"
RACE_LINK_SELECTOR = "dl.race_top_data_info a[href]"
# the whole path segment: ".../race/<id>x/" must not yield <id>
_RACE_PATH = re.compile(r"/race/([^/]+)/?")
_ISO_KEY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def race_hrefs(soup: BeautifulSoup) -> list[str]:
    return [
        str(a.get("href") or "")
        for box in soup.select(LIST_SELECTOR)
        for a in box.select(RACE_LINK_SELECTOR)
    ]


def race_id_from_href(href: str) -> str | None:
    m = _RACE_PATH.fullmatch(urlsplit(href).path)
    return urls.jra_race_id(m.group(1)) if m else None


def _race_date(page: RawPage, soup: BeautifulSoup) -> date:
    """The task key (YYYY-MM-DD) names the day; the page heading is the fallback."""
    if isinstance(page.key, str) and _ISO_KEY.fullmatch(page.key):
        try:
            return date.fromisoformat(page.key)
        except ValueError:
            pass
    heading = soup.find("h2") or soup.find("title")
    m = common.JP_DATE.search(heading.get_text() if heading else "")
    if not m:
        raise ValueError(f"race list {page.key!r}: no date in the key or the heading")
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def parse(page: RawPage) -> ParsedPage:
    soup = BeautifulSoup(page.body, "lxml")
    race_day = _race_date(page, soup)
    ids: dict[str, None] = {}
    for href in race_hrefs(soup):
        race_id = race_id_from_href(href)
        if race_id is not None:
            ids.setdefault(race_id)
    meta = {
        "page_type": page.page_type,
        "key": page.key,
        "url": page.url,
        "race_date": race_day,
        "parser_version": PARSER_VERSION,
    }
    return {"_meta": [meta], "race_links": [{"race_id": r, "race_date": race_day} for r in ids]}
