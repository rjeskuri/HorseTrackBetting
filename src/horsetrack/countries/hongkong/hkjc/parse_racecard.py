"""Race card page -> races (the columns knowable before the race) + racecard_runners (ported from
HKTrack's `sources/hkjc/parse_racecard.py`; HKTrack research 03, section 2).

HKTrack verified against the live 2026-09-23 Happy Valley card:
  - the starter table has 27 columns, all present in the HTML even when hidden; columns are
    mapped from the header text, never by position;
  - the race header is one block of <br>-separated lines, recognised by content (the going is
    absent until it is declared, group races print "-" instead of a rating band);
  - stand-by starters sit in a separate 10-column table#standbylist and are emitted with
    is_reserve=True;
  - the jockey cell carries the apprentice claim, e.g. "C L Chau (-2)", and the card "Wt." is the
    weight BEFORE the claim. Results store the weight actually carried, so both are emitted:
    card_wt_lb (as printed) and actual_wt_lb = Wt. - claim + probable overweight
    (docs/ledger.md, "HK weight carried", single-run).
How a late scratching is marked on the card has not been seen yet (UNVERIFIED): a scratch CSS
class on the row and the words scratched/withdrawn are both honoured. The GraphQL runner status is
the better source.

Pure: parses bytes already fetched through `horsetrack.core.ingest`; the page is untrusted data.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

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
    race_id,
    soup,
    time_to_seconds,
    to_int,
)
from horsetrack.countries.hongkong.hkjc.urls import parse_race_key

PARSER_VERSION = "1"

# header text (as normalised by parse_common.header_index) -> our field
_STARTER_HEADERS = {
    "horse_no": ("horse no", "no"),
    "last6": ("last 6 runs",),
    "horse": ("horse",),
    "brand": ("brand no",),
    "card_wt": ("wt",),
    "jockey": ("jockey",),
    "overweight": ("over wt",),
    "draw": ("draw",),
    "trainer": ("trainer",),
    "intl_rating": ("int'l rtg",),
    "rating": ("rtg",),
    "rating_change": ("rtg.+/-",),
    "declared_wt": ("horse wt. (declaration)",),
    "wt_change": ("wt.+/- (vs declaration)",),
    "best_time": ("best time",),
    "age": ("age",),
    "wfa": ("wfa",),
    "sex": ("sex",),
    "season_stakes": ("season stakes",),
    "priority": ("priority",),
    "days_since": ("days since last run",),
    "gear": ("gear",),
    "owner": ("owner",),
    "sire": ("sire",),
    "dam": ("dam",),
    "import_type": ("import cat",),
}
_MONTHS = {
    name: i + 1
    for i, name in enumerate(
        ("january", "february", "march", "april", "may", "june", "july", "august", "september",
         "october", "november", "december")
    )
}  # fmt: skip

_TITLE = re.compile(r"Race\s+(\d+)\s*(?:-\s*(.*))?$", re.I)
_DATE = re.compile(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})")
_CLOCK = re.compile(r"(\d{1,2}):(\d{2})\s*$")
_DISTANCE = re.compile(r"(\d{3,4})\s*M", re.I)
_COURSE_CONFIG = re.compile(r"[\"“]\s*([^\"”]+?)\s*[\"”]")
_PRIZE = re.compile(r"Prize Money\s*:\s*\$?\s*([\d,]+)", re.I)
_BAND = re.compile(r"Rating\s*:\s*(\d+)\s*-\s*(\d+)", re.I)
_BAND_OPEN = re.compile(r"Rating\s*:\s*(\d+)\s*\+", re.I)
_CLAIM = re.compile(r"\(\s*[-+]?\s*(\d+)\s*\)")
_BRACKETED = re.compile(r"\([^()]*\)")
_SIGNED = re.compile(r"[-+]?\d+")
_SCRATCH_WORDS = re.compile(r"\b(scratched|withdrawn)\b", re.I)
_SAFE_ID = re.compile(r"[A-Za-z0-9_]{1,40}")
_LINE_BREAK = "\x00"


def parse(page: RawPage) -> ParsedPage:
    doc = soup(page.body)
    starters = _table_with(
        doc.select("table.starter") or doc.find_all("table"), ("horse", "jockey", "draw")
    )
    header = _race_header(doc)
    identity = _identity(header, page)
    if starters is None or identity is None:
        return {"races": [], "racecard_runners": []}

    race_date, venue, race_no = identity
    rid = race_id(meeting_id(race_date, venue), race_no)
    runners = _runners(starters, rid, page.fetched_utc, reserve=False)
    standby = doc.select_one("table#standbylist")
    if standby is not None:
        runners += _runners(standby, rid, page.fetched_utc, reserve=True)
    runners = _unique(runners)
    return {
        "races": [_race_row(rid, race_no, race_date, header, runners)],
        "racecard_runners": runners,
    }


# ---------------------------------------------------------------- identity and race header


def _venue_names() -> dict[str, str]:
    venues = load_source(COUNTRY, SOURCE).venues
    return {venue.name.lower(): code for code, venue in venues.items()}


def _url_identity(url: str) -> tuple[date, str, int] | None:
    try:
        year, month, day = (int(part) for part in (href_param(url, "racedate") or "").split("/"))
        venue = (href_param(url, "racecourse") or "").upper()
        if not re.fullmatch(r"[A-Z0-9]{1,8}", venue):
            return None
        return date(year, month, day), venue, int(href_param(url, "raceno") or "")
    except ValueError:
        return None


def _identity(header: dict[str, Any], page: RawPage) -> tuple[date, str, int] | None:
    """(date, venue, race_no). The page's own labels win; the cache key and then the URL fill any
    gaps."""
    race_date, venue, race_no = header.get("race_date"), header.get("venue"), header.get("race_no")
    fallback = parse_race_key(page.key) or _url_identity(page.url)
    if fallback is not None:
        race_date = race_date or fallback[0]
        venue = venue or fallback[1]
        race_no = race_no or fallback[2]
    if race_date is None or venue is None or race_no is None:
        return None
    return race_date, venue, race_no


def _lines(node: Tag) -> list[str]:
    for br in node.find_all("br"):
        br.replace_with(_LINE_BREAK)
    parts = (clean(part) for part in node.get_text(" ").split(_LINE_BREAK))
    return [line for line in parts if line]


def _titled_div(divs: list[Tag]) -> Tag | None:
    return next((d for d in divs if _TITLE.match(cell_text(d))), None)


def _race_header(doc: Tag) -> dict[str, Any]:
    """The <br>-separated header block. Lines are recognised by what they contain, not by their
    order."""
    block = _titled_div(doc.select("div.f_fs13") or doc.find_all("div"))
    while block is not None and _titled_div(block.find_all("div")) is not None:
        # a wrapper matched: narrow down to the innermost block
        block = _titled_div(block.find_all("div"))
    out: dict[str, Any] = {}
    if block is None:
        return out
    for line in _lines(block):
        title, prize = _TITLE.match(line), _PRIZE.search(line)
        if title and "race_no" not in out:
            out["race_no"] = int(title.group(1))
            out["race_name"] = clean(title.group(2)) or None
        elif prize:
            out.update(_prize_line(line, prize))
        elif _DATE.search(line) and "race_date" not in out:
            out.update(_date_line(line))
        elif _DISTANCE.search(line) and "distance_m" not in out:
            out.update(_course_line(line))
    return out


def _date_line(line: str) -> dict[str, Any]:
    """'Wednesday, September 23, 2026, Happy Valley, 18:40' -> race date, venue code, post
    time."""
    out: dict[str, Any] = {}
    m = _DATE.search(line)
    month = _MONTHS.get(m.group(1).lower())
    try:
        out["race_date"] = date(int(m.group(3)), month, int(m.group(2))) if month else None
    except ValueError:
        out["race_date"] = None
    rest = line[m.end() :].lower()
    out["venue"] = next(
        (code for name, code in _venue_names().items() if name and name in rest), None
    )
    clock = _CLOCK.search(line)
    if clock and int(clock.group(1)) < 24 and int(clock.group(2)) < 60:
        out["post_time"] = (int(clock.group(1)), int(clock.group(2)))
    return out


def _course_line(line: str) -> dict[str, Any]:
    """'Turf, "C" Course, 1650M[, Good]' | 'All Weather Track, 1650M, Wet Slow'. Going follows
    distance."""
    out: dict[str, Any] = {"surface": None, "course_config": None, "going": None}
    tokens = [clean(t) for t in line.split(",")]
    distance_at = next((i for i, t in enumerate(tokens) if _DISTANCE.fullmatch(t)), None)
    if distance_at is None:
        return out
    out["distance_m"] = int(_DISTANCE.fullmatch(tokens[distance_at]).group(1))
    for token in tokens[:distance_at]:
        config = _COURSE_CONFIG.search(token)
        if "ALL WEATHER" in token.upper() or token.upper() == "AWT":
            out["surface"] = "awt"
        elif token.upper().startswith("TURF"):
            out["surface"] = "turf"
        if config:
            out["course_config"] = config.group(1).upper().replace(" ", "")
    going = ", ".join(t for t in tokens[distance_at + 1 :] if t and t != "-")
    out["going"] = going.upper() or None  # upper case, as the results pages print it
    return out


def _prize_line(line: str, prize: re.Match[str]) -> dict[str, Any]:
    """'Prize Money: $875,000, Rating: 40-0, Class 5' | 'Prize Money: $13,000,000, -, Group
    One'."""
    out: dict[str, Any] = {"prize": to_int(prize.group(1)), "rating_hi": None, "rating_lo": None}
    rest = line[prize.end() :]
    closed, open_ended = _BAND.search(rest), _BAND_OPEN.search(rest)
    if closed:
        a, b = int(closed.group(1)), int(closed.group(2))
        out["rating_hi"], out["rating_lo"] = max(a, b), min(a, b)
    elif open_ended:
        out["rating_lo"] = int(open_ended.group(1))
    labels = [clean(t) for t in rest.split(",")]
    out["race_class"] = (
        ", ".join(t for t in labels if t and t != "-" and not t.lower().startswith("rating"))
        or None
    )
    return out


def _post_time_local(race_date: date, post_time: tuple[int, int] | None) -> str | None:
    if post_time is None:
        return None
    zone = ZoneInfo(load_source(COUNTRY, SOURCE).timezone)
    day = datetime(race_date.year, race_date.month, race_date.day, *post_time, tzinfo=zone)
    return day.isoformat()


def _race_row(
    rid: str, race_no: int, race_date: date, header: dict[str, Any], runners: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "jurisdiction": SOURCE,
        "race_id": rid,
        "meeting_id": rid.rsplit(":", 1)[0],
        "race_no": race_no,
        "race_name": header.get("race_name"),
        "race_class": header.get("race_class"),
        "rating_hi": header.get("rating_hi"),
        "rating_lo": header.get("rating_lo"),
        "distance_m": header.get("distance_m"),
        "surface": header.get("surface"),
        "course_config": header.get("course_config"),
        "going": header.get("going"),
        "prize": header.get("prize"),
        "post_time_local": _post_time_local(race_date, header.get("post_time")),
        "field_size": sum(not r["is_reserve"] and not r["is_scratched"] for r in runners),
        "status": "declared",
    }


# ---------------------------------------------------------------- runners


def _header_row(table: Tag, wanted: tuple[str, ...]) -> tuple[Tag, dict[str, int]] | None:
    """The first row of THIS table (not of a nested one) whose cells name all the wanted
    columns."""
    for row in table.find_all("tr"):
        if row.find_parent("table") is not table:
            continue
        cells = row.find_all(["td", "th"], recursive=False)
        columns = header_index([cell_text(c) for c in cells])
        if all(w in columns for w in wanted):
            return row, columns
    return None


def _table_with(tables: list[Tag], wanted: tuple[str, ...]) -> Tag | None:
    return next((t for t in tables if _header_row(t, wanted) is not None), None)


def _runners(table: Tag, rid: str, scraped: datetime | None, reserve: bool) -> list[dict[str, Any]]:
    found = _header_row(table, ("horse",))
    if found is None:
        return []
    header_row, columns = found
    at = {
        field: columns[name]
        for field, names in _STARTER_HEADERS.items()
        for name in names
        if name in columns
    }
    runners = []
    for row in header_row.find_all_next("tr"):
        if row.find_parent("table") is not table:
            continue
        cells = row.find_all("td", recursive=False)
        runner = _runner(row, cells, at, rid, scraped, reserve)
        if runner["horse_id"] is not None or runner["horse_name"] is not None:
            runners.append(runner)
    return runners


def _link_id(link: Tag | None, param: str) -> str | None:
    """Ids come from scraped hrefs and end up in join keys, URLs and cache file names: plain
    charset only."""
    value = href_param(link["href"], param) if link is not None else None
    return value if value and _SAFE_ID.fullmatch(value) else None


def _signed(text: str | None) -> int | None:
    """'+5' -> 5, '-5' -> -5, '0' -> 0; a bare '-' (nothing to report) -> None."""
    m = _SIGNED.search((text or "").replace(",", ""))
    return int(m.group()) if m else None


def _person(cell: Tag | None, id_param: str) -> tuple[str | None, str | None]:
    """(name without claim or remarks, code from the profile link)."""
    if cell is None:
        return None, None
    link = cell.find("a", href=True)
    name = clean(_BRACKETED.sub(" ", cell_text(link if link is not None else cell)))
    if not any(ch.isalpha() for ch in name):
        name = ""
    return name or None, _link_id(link, id_param)


def _priority(raw: str | None) -> tuple[bool, bool, int | None]:
    """'+ * 2' -> (trump card, priority to run, order of trainer preference)."""
    text = raw or ""
    return "+" in text, "*" in text, to_int(text)


def _is_scratched(row: Tag, cells: list[Tag], text: str) -> bool:
    classes = [c for node in (row, *cells) for c in (node.get("class") or [])]
    return any("scratch" in c.lower() for c in classes) or bool(_SCRATCH_WORDS.search(text))


def _runner(
    row: Tag,
    cells: list[Tag],
    at: dict[str, int],
    rid: str,
    scraped: datetime | None,
    reserve: bool,
) -> dict[str, Any]:
    def cell(field: str) -> Tag | None:
        index = at.get(field)
        return cells[index] if index is not None and index < len(cells) else None

    def text(field: str) -> str | None:
        value = cell_text(cell(field))
        return value if value and value != "-" else None

    horse_link = cell("horse").find("a", href=True) if cell("horse") is not None else None
    jockey_name, jockey_code = _person(cell("jockey"), "jockeyid")
    trainer_name, trainer_code = _person(cell("trainer"), "trainerid")
    claim = _CLAIM.search(cell_text(cell("jockey")))
    allowance = int(claim.group(1)) if claim else (0 if jockey_name else None)
    card_wt, overweight = to_int(text("card_wt")), to_int(text("overweight"))
    actual_wt = None if card_wt is None else card_wt - (allowance or 0) + (overweight or 0)
    number = to_int(text("horse_no"))
    trump_card, priority_to_run, preference = _priority(text("priority"))
    # the stand-by table styles its No./Horse cells with the scratch class, so only words count
    row_text = cell_text(row)
    if reserve:
        scratched = bool(_SCRATCH_WORDS.search(row_text))
    else:
        scratched = _is_scratched(row, cells, row_text)
    return {
        "jurisdiction": SOURCE,
        "race_id": rid,
        "horse_id": _link_id(horse_link, "horseid"),
        "horse_no": None if reserve else number,
        "horse_name": clean(_BRACKETED.sub(" ", text("horse") or "")) or None,
        "brand_raw": text("brand"),
        "draw": to_int(text("draw")),
        "card_wt_lb": card_wt,
        "overweight_lb": overweight,
        "actual_wt_lb": actual_wt,
        "jockey_key": person_key("jockey", jockey_name or ""),
        "jockey_code": jockey_code,
        "jockey_name": jockey_name,
        "trainer_key": person_key("trainer", trainer_name or ""),
        "trainer_code": trainer_code,
        "trainer_name": trainer_name,
        "allowance_lb": allowance,
        "intl_rating": to_int(text("intl_rating")),
        "rating": to_int(text("rating")),
        "rating_change": _signed(text("rating_change")),
        "declared_horse_wt_lb": to_int(text("declared_wt")),
        "horse_wt_change_lb": _signed(text("wt_change")),
        "best_time_s": time_to_seconds(text("best_time")),
        "age": to_int(text("age")),
        "wfa_lb": to_int(text("wfa")),
        "sex": text("sex"),
        "season_stakes": to_int(text("season_stakes")),
        "priority_raw": text("priority"),
        "trump_card": trump_card,
        "priority_to_run": priority_to_run,
        "trainer_preference": preference,
        "days_since_last_run": to_int(text("days_since")),
        "gear_raw": text("gear"),
        "owner": text("owner"),
        "sire": text("sire"),
        "dam": text("dam"),
        "import_type": text("import_type"),
        "last6_raw": text("last6"),
        "is_reserve": reserve,
        "reserve_no": number if reserve else None,
        "is_scratched": scratched,
        "card_scraped_utc": scraped,
    }


def _unique(runners: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """(race_id, horse_id) must be unique; a repeated horse keeps its first (declared starter)
    row."""
    seen: set[str] = set()
    out = []
    for runner in runners:
        horse_id = runner["horse_id"]
        if horse_id is not None and horse_id in seen:
            continue
        seen.add(horse_id)
        out.append(runner)
    return out
