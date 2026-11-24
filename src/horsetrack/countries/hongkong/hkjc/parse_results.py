"""Results page -> races, runners, dividends, incidents (+ `_meta` for task fan-out), on the live
and the archive routes (ported from HKTrack's `sources/hkjc/parse_results.py`).

Layouts differ by era, so nothing here addresses a column or a header row by position:
  - runner columns are mapped from header text (no Running Position before ~2008, none of
    Running Position / Win Odds on abandoned races);
  - race-header rows are found by their labels ("Going :", "Course :", "Time :");
  - dividend pool cells are found structurally (3 cells = new pool, 2 cells = continuation),
    because the abandoned-race page drops the CSS class the other eras carry;
  - a race with no result comes in three shapes: the 2019 banner with a header and the declared
    runners, the 2024+ banner with NEITHER a race header NOR a runner table (only refunds), and a
    void race (header, runners all placed "VOID"). All three give one `races` row with status
    "abandoned";
  - incidents come as a per-horse table (2024 onwards at least) or as one narrative paragraph
    (2004-2020 at least). The narrative is kept whole in one row with a null horse, followed by
    one row per runner it names (matched on the capitalised horse name), so both eras can be read
    per horse. Never concatenate the two.
Malformed cells become null next to their `*_raw` text; a bad row never stops the page. The page
is untrusted data: it is parsed, never followed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from typing import Any

from bs4 import Comment, Tag

from horsetrack.core.config import load_source
from horsetrack.core.ingest.types import (
    STATUS_ABANDONED,
    STATUS_BLOCKED,
    STATUS_EMPTY,
    STATUS_FETCHED,
    ParsedPage,
    RawPage,
)
from horsetrack.countries.hongkong.hkjc import urls
from horsetrack.countries.hongkong.hkjc.parse_common import (
    COUNTRY,
    NO_INFO_MARKER,
    SOURCE,
    cell_text,
    clean,
    header_index,
    href_param,
    lbw_to_lengths,
    person_key,
    race_id,
    soup,
    strip_not_content,
    time_to_seconds,
    to_float,
    to_int,
)

PARSER_VERSION = "3"
JURISDICTION = SOURCE

# Banner sentences of a race without a result: 2019 wording, 2024+ wording, and a race annulled
# after the start. Whole sentences on purpose: the stewards' report may say "declared void" about
# some other race.
ABANDONED_MARKERS = (
    "abandoned or not offered",
    "This race is declared abandoned",
    "This race is declared void",
)
# Substrings that must ALL be present outside comments, <style> and <script> for a results page
# to count as real content.
_CONTENT_MARKERS = ("commContent", "localResults")

# header text (as normalised by parse_common.header_index) -> our field
_RUNNER_HEADERS = {
    "placing": ("pla", "placing"),
    "horse_no": ("horse no", "h.no"),
    "horse": ("horse",),
    "jockey": ("jockey",),
    "trainer": ("trainer",),
    "actual_wt": ("act. wt", "actual wt", "actualwt"),
    "declared_wt": ("declar. horse wt", "declared horse wt"),
    "draw": ("dr", "draw"),
    "lbw": ("lbw",),
    "running": ("running position",),
    "finish_time": ("finish time",),
    "win_odds": ("win odds",),
}
# withdrawn by the vet / the stewards / retired, including the "-A" (after weighing out) forms
_SCRATCHED_PREFIXES = ("WV", "WX", "WR")
_SKIPPED_DIVIDENDS = {"DETAIL", "DETAILS", "NOT WIN"}

_RACE_TITLE = re.compile(r"RACE\s+(\d+)(?:\s*\((\d+)\))?", re.I)
_MEETING_LABEL = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})\s*(.*)")
_DISTANCE = re.compile(r"(\d{3,4})\s*M", re.I)
_BAND = re.compile(r"\(\s*(\d+)\s*-\s*(\d+)\s*\)")
_BAND_OPEN = re.compile(r"\(\s*(\d+)\s*\+\s*\)")
_COURSE_CONFIG = re.compile(r"[\"“]\s*([^\"”]+?)\s*[\"”]")
_PLACING = re.compile(r"(\d+)\s*([A-Za-z][A-Za-z\- ]*)?")
_BRAND = re.compile(r"\(([^()]*)\)\s*$")
_MONEY = re.compile(r"[\d,]+(?:\.\d+)?")
_MONEY_PER_UNIT_BET = re.compile(r"([\d,]+(?:\.\d+)?)\s*/\s*\$\s*(\d+(?:\.\d+)?)")
_NO_MARGIN = re.compile(r"-+")
_APOSTROPHES = str.maketrans({"‘": "'", "’": "'", "`": "'"})
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s*(?=[A-Z<(\"“])")
_CAPS_WORD = re.compile(r"(?<![A-Za-z])[A-Z][A-Z']{2,}(?![a-z])")
_SAFE_ID = re.compile(r"[A-Za-z0-9_]{1,40}")
_CURRENT_TAB = re.compile(r"_(\d+)_o\.gif$", re.I)
_LINE_BREAK = "\x00"


def classify(body: bytes) -> str:
    """Decide by content, never by HTTP status: the site answers a missing race or meeting with
    200 and "No information.", and serves abandoned races as 200 too. Inline <style>, <script>
    and HTML comments are not content (the route's CSS names the results containers)."""
    text = strip_not_content(body.decode("utf-8", errors="replace"))
    if not all(marker in text for marker in _CONTENT_MARKERS):
        # a stray "No information." inside a real result (a stewards' report) is not the sentinel
        return STATUS_EMPTY if NO_INFO_MARKER in text else STATUS_BLOCKED
    if any(marker in text for marker in ABANDONED_MARKERS):
        return STATUS_ABANDONED
    return STATUS_FETCHED


def parse(page: RawPage) -> ParsedPage:
    doc = soup(page.body)
    root = doc.select_one("div.localResults") or doc
    header = _race_header(root)
    runner_table = _find_table(root, wanted=("horse no", "horse"), unwanted=("incident",))
    identity = _identity(root, page, header.get("race_no"))
    abandoned = _has_text(root, ABANDONED_MARKERS)
    # The 2024+ abandoned page has neither a header nor runners, yet it is a race (with refunds)
    # all the same.
    if identity is None or (not header and runner_table is None and not abandoned):
        return {"races": [], "runners": [], "dividends": [], "incidents": [], "_meta": []}

    race_date, venue, race_no = identity
    rid = race_id(race_date, venue, race_no)
    runners = _runners(runner_table, rid)
    race = _race_row(rid, race_no, header, runners, abandoned=abandoned)
    race_nos = _local_race_nos(root, venue, race_no)
    from_key = urls.parse_race_key(page.key)
    meta = {
        "page_type": "results",
        "race_date": race_date,
        "venue": venue,
        "race_no": race_no,
        "n_races": len(race_nos),
        "race_nos": race_nos,
        "key_mismatch": from_key is not None and from_key != identity,
        "class_raw": header.get("class_raw"),
        "course_raw": header.get("course_raw"),
    }
    return {
        "races": [race],
        "runners": runners,
        "dividends": _dividends(root, rid),
        "incidents": _incidents(root, rid, runners),
        "_meta": [meta],
    }


# ---------------------------------------------------------------- identity and race tabs


def _venue_names() -> dict[str, str]:
    venues = load_source(COUNTRY, SOURCE).venues
    return {venue.name.lower(): code for code, venue in venues.items()}


def _identity(root: Tag, page: RawPage, header_race_no: int | None) -> tuple[date, str, int] | None:
    """(date, venue, race_no). The page's own labels win; the cache key and then the URL fill
    any gaps."""
    race_date, venue = _meeting_label(root)
    race_no = header_race_no or _current_tab(root)
    fallback = urls.parse_race_key(page.key) or _identity_from_url(page.url)
    if fallback is not None:
        race_date = race_date or fallback[0]
        venue = venue or fallback[1]
        race_no = race_no or fallback[2]
    if race_date is None or venue is None or race_no is None:
        return None
    return race_date, venue, race_no


def _meeting_label(root: Tag) -> tuple[date | None, str | None]:
    """'Race Meeting: 16/09/2026 Happy Valley' -> (2026-09-16, 'HV')."""
    for span in root.select("div.raceMeeting_select span"):
        m = _MEETING_LABEL.search(cell_text(span))
        if not m:
            continue
        try:
            race_date = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            continue
        place = m.group(4).lower()
        venue = next(
            (code for name, code in _venue_names().items() if name and place.startswith(name)),
            None,
        )
        return race_date, venue
    return None, None


def _identity_from_url(url: str) -> tuple[date, str, int] | None:
    raw_date, venue = href_param(url, "racedate"), href_param(url, "racecourse")
    try:
        year, month, day = (int(part) for part in (raw_date or "").split("/"))
        return date(year, month, day), (venue or "").upper(), int(href_param(url, "raceno") or "")
    except ValueError:
        return None


def _current_tab(root: Tag) -> int | None:
    """The race being shown is the one unlinked tab image, e.g. racecard_rt_2_o.gif."""
    for img in root.select("div.top_races img[src]"):
        m = _CURRENT_TAB.search(img["src"])
        if m and img.find_parent("a") is None:
            return int(m.group(1))
    return None


def _local_race_nos(root: Tag, venue: str, current: int) -> list[int]:
    """Race numbers of THIS venue from the tab strip. Simulcast meetings share the strip as extra
    rows."""
    names = [name for name, code in _venue_names().items() if code == venue and name]
    found = {current}
    for row in root.select("div.top_races tr"):
        label = cell_text(row.find("td")).lower()
        row_is_local = any(label.startswith(name) for name in names)
        for link in row.select("a[href]"):
            race_no = to_int(href_param(link["href"], "raceno"))
            course = href_param(link["href"], "racecourse")
            if (
                race_no
                and race_no > 0
                and ((course or "").upper() == venue or (course is None and row_is_local))
            ):
                found.add(race_no)
    return sorted(found)


# ---------------------------------------------------------------- race header


def _race_header(root: Tag) -> dict[str, Any]:
    """Rows are addressed by label, never by index (there is an empty spacer row, and eras
    differ)."""
    table = root.select_one("div.race_tab table")
    out: dict[str, Any] = {}
    if table is None:
        return out
    for row in table.find_all("tr"):
        cells = [cell_text(td) for td in row.find_all(["td", "th"], recursive=False)]
        labels = [c.rstrip(": ").lower() for c in cells]
        title = _RACE_TITLE.match(cells[0]) if cells else None
        if title and "race_no" not in out:
            out["race_no"] = int(title.group(1))
            index = int(title.group(2)) if title.group(2) else 0
            out["race_index"] = index if index > 0 else None
        elif "going" in labels:
            i = labels.index("going")
            out["class_raw"] = _first(cells[:i])
            out["going"] = _first(cells[i + 1 :])
        elif "course" in labels:
            i = labels.index("course")
            out["race_name"] = _first(cells[:i])
            out["course_raw"] = _first(cells[i + 1 :])
        elif "time" in labels:
            i = labels.index("time")
            out["prize"] = to_int(_first(cells[:i]))
            times = [time_to_seconds(c.strip("() ")) for c in cells[i + 1 :]]
            out["split_times_s"] = [t for t in times if t is not None]
    return out


def _first(cells: list[str]) -> str | None:
    return next((c for c in cells if c), None)


def parse_class_line(text: str | None) -> tuple[str | None, int | None, int | None, int | None]:
    """'Class 5 - 1000M - (40-0)' -> ('Class 5', 1000, 40, 0). The band is absent for griffin and
    group races."""
    parts = [p for p in re.split(r"\s+-\s+", clean(text)) if p]
    distance_at = next((i for i, p in enumerate(parts) if _DISTANCE.fullmatch(p)), None)
    distance = (
        int(_DISTANCE.fullmatch(parts[distance_at]).group(1)) if distance_at is not None else None
    )
    class_parts = parts[:distance_at] if distance_at is not None else parts[:1]
    race_class = " - ".join(class_parts) or None
    rating_hi = rating_lo = None
    for part in parts[len(class_parts) :]:
        closed, open_ended = _BAND.fullmatch(part), _BAND_OPEN.fullmatch(part)
        if closed:
            a, b = int(closed.group(1)), int(closed.group(2))
            rating_hi, rating_lo = max(a, b), min(a, b)
        elif open_ended:
            rating_lo = int(open_ended.group(1))
    return race_class, distance, rating_hi, rating_lo


def parse_course(text: str | None) -> tuple[str | None, str | None]:
    """'TURF - "C+3" Course' -> ('turf', 'C+3'); 'ALL WEATHER TRACK' -> ('awt', None)."""
    t = clean(text)
    if "ALL WEATHER" in t.upper() or t.upper() == "AWT":
        return "awt", None
    if not t.upper().startswith("TURF"):
        return None, None
    m = _COURSE_CONFIG.search(t)
    return "turf", (m.group(1).upper().replace(" ", "") if m else None)


def _race_row(
    rid: str, race_no: int, header: dict[str, Any], runners: list[dict[str, Any]], abandoned: bool
) -> dict[str, Any]:
    race_class, distance, rating_hi, rating_lo = parse_class_line(header.get("class_raw"))
    surface, course_config = parse_course(header.get("course_raw"))
    splits = header.get("split_times_s") or []
    winner_time = next((r["finish_time_s"] for r in runners if r["finish_pos"] == 1), None)
    # No banner but nobody placed (e.g. a page fetched before the race was run) is not a usable
    # result either.
    run = not abandoned and any(r["finish_pos"] is not None for r in runners)
    return {
        "jurisdiction": JURISDICTION,
        "race_id": rid,
        "meeting_id": rid.rsplit(":", 1)[0],
        "race_no": race_no,
        "race_index": header.get("race_index"),
        "race_name": header.get("race_name"),
        "race_class": race_class,
        "rating_hi": rating_hi,
        "rating_lo": rating_lo,
        "distance_m": distance,
        "surface": surface,
        "course_config": course_config,
        "going": header.get("going"),
        "prize": header.get("prize"),
        "post_time_local": None,  # not printed on the results page
        "field_size": sum(not r["is_scratched"] for r in runners),
        "finish_time_s": (splits[-1] if splits else winner_time) if run else None,
        "split_times_s": splits,
        "status": "run" if run else "abandoned",
    }


# ---------------------------------------------------------------- runners


def _find_table(root: Tag, wanted: tuple[str, ...], unwanted: tuple[str, ...] = ()) -> Tag | None:
    """First table whose header row has all `wanted` column names and none of the `unwanted`
    ones."""
    for table in root.find_all("table"):
        first = table.find("tr")
        if first is None:
            continue
        columns = header_index(
            [cell_text(c) for c in first.find_all(["td", "th"], recursive=False)]
        )
        if all(w in columns for w in wanted) and not any(u in columns for u in unwanted):
            return table
    return None


def _rows(table: Tag) -> tuple[dict[str, int], list[list[Tag]]]:
    rows = table.find_all("tr")
    columns = header_index([cell_text(c) for c in rows[0].find_all(["td", "th"], recursive=False)])
    return columns, [row.find_all("td", recursive=False) for row in rows[1:]]


def _column_lookup(columns: dict[str, int], headers: dict[str, tuple[str, ...]]) -> dict[str, int]:
    return {
        field: columns[name]
        for field, names in headers.items()
        for name in names
        if name in columns
    }


def _cell(cells: list[Tag], index: int | None) -> Tag | None:
    """The row's cell for a mapped column; None when the era has no such column or the row is
    short."""
    return cells[index] if index is not None and index < len(cells) else None


def _runners(table: Tag | None, rid: str) -> list[dict[str, Any]]:
    if table is None:
        return []
    columns, rows = _rows(table)
    at = _column_lookup(columns, _RUNNER_HEADERS)
    runners = []
    for cells in rows:
        row = _runner(cells, at, rid)
        if row["horse_no"] is not None or row["horse_name"] is not None:
            runners.append(row)
    return runners


def _runner(cells: list[Tag], at: dict[str, int], rid: str) -> dict[str, Any]:
    def cell(field: str) -> Tag | None:
        return _cell(cells, at.get(field))

    def text(field: str) -> str | None:
        return cell_text(cell(field)) or None

    placing = parse_placing(text("placing"))
    horse = _horse(cell("horse"))
    jockey_name, jockey_code = _person(cell("jockey"), "jockeyid")
    trainer_name, trainer_code = _person(cell("trainer"), "trainerid")
    lbw_raw = text("lbw")
    return {
        "jurisdiction": JURISDICTION,
        "race_id": rid,
        **horse,
        "horse_no": to_int(text("horse_no")),
        "jockey_key": person_key("jockey", jockey_name),
        "jockey_code": jockey_code,
        "jockey_name": jockey_name,
        "trainer_key": person_key("trainer", trainer_name),
        "trainer_code": trainer_code,
        "trainer_name": trainer_name,
        "actual_wt_lb": to_int(text("actual_wt")),
        "declared_horse_wt_lb": to_int(text("declared_wt")),
        "draw": to_int(text("draw")),
        **placing,
        "lbw_raw": lbw_raw,
        "lbw_lengths": _lbw_lengths(lbw_raw, placing["finish_pos"]),
        "running_positions": _running_positions(cell("running")),
        "finish_time_s": time_to_seconds(text("finish_time")),
        "win_odds": to_float(text("win_odds")),
        "is_scratched": is_scratched(placing["placing_code"]),
    }


def parse_placing(text: str | None) -> dict[str, Any]:
    """'3' | '1 DH' | 'WV-A' | '' -> placing_raw, finish_pos, placing_code, dead_heat."""
    raw = clean(text) or None
    finish_pos = code = None
    m = _PLACING.fullmatch(raw or "")
    if m:
        finish_pos = int(m.group(1))
        code = clean(m.group(2)).upper() or None
    elif raw and re.search(r"[A-Za-z0-9]", raw):
        code = raw.upper()
    return {
        "placing_raw": raw,
        "finish_pos": finish_pos,
        "placing_code": code,
        "dead_heat": code == "DH",
    }


def _lbw_lengths(lbw_raw: str | None, finish_pos: int | None) -> float | None:
    """Beaten lengths, 0 for the winner. After a protest the promoted winner shows a negative
    margin ('-HD') and the relegated first-past-the-post shows '---': nothing beat it to the line,
    so that is 0 as well. "N" is a neck on HKJC pages (parse_common.LBW_CODES)."""
    if finish_pos == 1 or (finish_pos is not None and _NO_MARGIN.fullmatch(lbw_raw or "")):
        return 0.0
    return lbw_to_lengths(lbw_raw)


def is_scratched(placing_code: str | None) -> bool:
    """Withdrawn before the start (bets refunded), incl. WXNR = declared a non-runner by the
    stewards. PU/UR/FE/DNF/TNP/DISQ/VOID runners started, so they count."""
    return bool(placing_code) and placing_code.upper().startswith(_SCRATCHED_PREFIXES)


def safe_id(value: str | None) -> str | None:
    """Ids come from scraped hrefs and end up in join keys, URLs and cache file names: plain
    charset only."""
    return value if value and _SAFE_ID.fullmatch(value) else None


def _horse(cell: Tag | None) -> dict[str, Any]:
    link = cell.find("a", href=True) if cell is not None else None
    full = cell_text(cell)
    brand = _BRAND.search(full)
    name = cell_text(link) or _BRAND.sub("", full).strip()
    return {
        "horse_id": safe_id(href_param(link["href"], "horseid")) if link is not None else None,
        "horse_name": name or None,
        "brand_raw": (brand.group(1).strip() or None) if brand else None,
    }


def _person(cell: Tag | None, id_param: str) -> tuple[str | None, str | None]:
    """(name, code). People who no longer hold a licence are printed as plain text without a
    link."""
    link = cell.find("a", href=True) if cell is not None else None
    code = safe_id(href_param(link["href"], id_param)) if link is not None else None
    return cell_text(cell) or None, code


def _running_positions(cell: Tag | None) -> list[int]:
    if cell is None:
        return []
    tokens = [cell_text(div) for div in cell.select("div div")] or cell_text(cell).split()
    return [int(t) for t in tokens if t.isdigit()]


# ---------------------------------------------------------------- dividends


def pool_code(label: str) -> str:
    """Normalised pool code from config; labels the config does not know keep a slug of the raw
    label."""
    pool = load_source(COUNTRY, SOURCE).pool_by_label(clean(label))
    if pool is not None:
        return pool.code
    return re.sub(r"[^A-Z0-9]+", "_", clean(label).upper()).strip("_")


def normalise_combination(text: str | None) -> str | None:
    """' 11, 12 ,6' -> '11,12,6'; legs of multi-race pools stay separated by '/'; '-' (refund)
    -> None."""
    t = clean(text)
    if not re.search(r"[A-Za-z0-9]", t):
        return None
    legs = []
    for leg in t.split("/"):
        tokens = [clean(tok) for tok in leg.split(",")]
        legs.append(",".join(str(int(tok)) if tok.isdigit() else tok for tok in tokens if tok))
    return "/".join(legs)


def parse_dividend(text: str | None) -> float | None:
    """Dividend per unit stake: '1,234.50' -> 1234.5. When nobody held a full unit the page quotes
    what the winning partial unit bet is paid, '349,150.00/$1.0'; that is scaled to the unit stake
    (-> 3,491,500.0)."""
    t = clean(text)
    if _MONEY.fullmatch(t):
        return float(t.replace(",", ""))
    m = _MONEY_PER_UNIT_BET.fullmatch(t)
    if m and float(m.group(2)) > 0:
        unit_stake = load_source(COUNTRY, SOURCE).unit_stake
        return float(m.group(1).replace(",", "")) * unit_stake / float(m.group(2))
    return None


def _dividends(root: Tag, rid: str) -> list[dict[str, Any]]:
    table = root.select_one("div.dividend_tab table")
    table = table or _find_table(root, wanted=("pool", "winning combination"))
    if table is None:
        return []
    out = []
    label: str | None = None
    for row in table.find_all("tr"):
        cells = [cell_text(td) for td in row.find_all("td", recursive=False)]
        if len(cells) >= 3:  # a pool's first row; the label cell spans the pool's remaining rows
            label = cells[0] or label
            cells = cells[1:3]
        if len(cells) != 2 or label is None or label.lower() == "pool":
            continue
        combination, amount = cells
        if amount.upper() in _SKIPPED_DIVIDENDS:
            continue
        out.append(
            {
                "jurisdiction": JURISDICTION,
                "race_id": rid,
                "pool": pool_code(label),
                "combination": normalise_combination(combination),
                "dividend": parse_dividend(amount),
                "is_refund": "REFUND" in amount.upper(),
            }
        )
    return out


# ---------------------------------------------------------------- incidents


def _incidents(root: Tag, rid: str, runners: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = _find_table(root, wanted=("incident",))
    if table is not None:
        return _incident_table(table, rid, runners)
    block = root.select_one("div.race_incident_report")
    if block is None:
        return []
    paragraphs = block.select("p.info_p")
    if not paragraphs:  # unknown layout: every paragraph except the block's own blue heading
        paragraphs = [p for p in block.find_all("p") if "bg_blue" not in (p.get("class") or [])]
    # <br> separates sentences; newlines in the source are just wrapping
    for br in block.find_all("br"):
        br.replace_with(_LINE_BREAK)
    lines = [clean(line) for p in paragraphs for line in p.get_text(" ").split(_LINE_BREAK)]
    text = "\n".join(line for line in lines if line)
    if not text:
        return []
    # Narrative era: the whole report first (null horse), then what it says about each runner it
    # names.
    rows = [_incident_row(rid, None, None, text)]
    by_name = sentences_by_name(text, [r["horse_name"] for r in runners])
    for r in runners:
        sentences = by_name.get(r["horse_name"])
        if sentences and r["horse_id"]:
            rows.append(_incident_row(rid, r["horse_id"], r["horse_no"], " ".join(sentences)))
    return rows


def _incident_row(
    rid: str, horse_id: str | None, horse_no: int | None, text: str
) -> dict[str, Any]:
    return {
        "jurisdiction": JURISDICTION,
        "race_id": rid,
        "horse_id": horse_id,
        "horse_no": horse_no,
        "text": text,
    }


def sentences_by_name(text: str, names: Iterable[str | None]) -> dict[str, list[str]]:
    """Sentences of a narrative stewards' report, grouped by the runners they name (names are in
    capitals).

    Lines are the report's own paragraphs; sentences often run together without a space after the
    full stop. A sentence that names no runner and has no other capitalised word ("He said the
    horse was lame.") continues the sentence before it on the same line; anything else without a
    runner stays unattributed.
    """
    by_plain = {name.translate(_APOSTROPHES): name for name in names if name}
    if not by_plain:
        return {}
    longest_first = sorted(by_plain, key=len, reverse=True)  # 'GLORY DAYS' must win over 'GLORY'
    pattern = re.compile(
        r"(?<![A-Z0-9])(?:"
        + "|".join(re.escape(n) for n in longest_first)
        + r")(?![A-Z0-9]|'[A-Z])"
    )
    found: dict[str, list[str]] = {}
    for line in text.split("\n"):
        previous: list[str] = []
        for sentence in filter(None, (part.strip() for part in _SENTENCE_BREAK.split(line))):
            plain = sentence.translate(_APOSTROPHES)  # the report prints ’, the runner table '
            named = list(dict.fromkeys(by_plain[m.group()] for m in pattern.finditer(plain)))
            if not named and not _CAPS_WORD.search(plain):
                named = previous
            for name in named:
                found.setdefault(name, []).append(sentence)
            previous = named
    return found


def _incident_table(table: Tag, rid: str, runners: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns, rows = _rows(table)
    at = _column_lookup(
        columns, {"horse_no": ("horse no",), "horse": ("horse",), "text": ("incident",)}
    )
    id_by_no = {r["horse_no"]: r["horse_id"] for r in runners if r["horse_no"] is not None}
    out = []
    for cells in rows:
        text = cell_text(_cell(cells, at.get("text")))
        if not text:
            continue
        horse_no = to_int(cell_text(_cell(cells, at.get("horse_no"))))
        horse_id = _horse(_cell(cells, at.get("horse")))["horse_id"] or id_by_no.get(horse_no)
        out.append(_incident_row(rid, horse_id, horse_no, text))
    return out


def _has_text(root: Tag, markers: tuple[str, ...]) -> bool:
    visible = (s for s in root.find_all(string=True) if not isinstance(s, Comment))
    return any(marker in s for s in visible for marker in markers)
