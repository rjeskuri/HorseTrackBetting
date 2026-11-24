"""JRA official race card (出馬表, JRADB pw01dde page, Shift_JIS) -> races + racecard_runners
(ported from HKTrack's `sources/jra/parse_racecard.py`). Pure: no network.

The race's identity comes from the page (its date line and race-number image), never from the
RawPage key; ids as in the odds log (`capture.race_id`). horse_id is the 10-character id of the
horse link (pw01dud..<id>), jockey and trainer codes the 5-digit codes of theirs (pw04kmk0<code>,
pw05cmk0<code>). Units and codes as HKTrack's: sex H/M/G, kg kept, lb = round(kg x 2.20462),
draw = horse number. A runner without a horse link gets no id (id_match "none").

Not printed on the card, so null: the going and the course letter A-D. Body weights are empty
until the weigh-in (about 65 min before the off). Only the first table under #syutsuba is the
field; its past-run cells (td.past) and the past-winners table describe other races and are
never read. Changes from HKTrack: no `_meta` table, `classify` or `match_horse_ids` (no history
to match against yet).
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

from horsetrack.core.ingest.types import ParsedPage, RawPage
from horsetrack.countries.japan.jra import capture
from horsetrack.countries.japan.jra import official_pages as op

LB_PER_KG = 2.20462

_HORSE = re.compile(r"pw01dud[0-9A-Za-z]{2}([0-9A-Za-z]{10})/")
_JOCKEY = re.compile(r"pw04kmk[0-9A-Za-z]([0-9A-Za-z]{5})/")
_TRAINER = re.compile(r"pw05cmk[0-9A-Za-z]([0-9A-Za-z]{5})/")
_BRACKET = re.compile(r"枠([0-9]+)")
_DISTANCE = re.compile(r"([0-9][0-9,]*)\s*メートル")
_KG = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*kg")
_BODY = re.compile(r"([0-9]+)kg(?:\(([+\-±]?)([0-9]+)\))?")
_CLAIM = re.compile(r"([0-9]+(?:\.[0-9]+)?)kg減量")
_GRADE = re.compile(r"(?:J[・.]?)?(G(?:I{1,3}|[1-3])|L)")  # a grade icon's alt, NFKC: 'GI', 'L'
_ROUND = re.compile(r"([0-9]+)回\D+?([0-9]+)日")  # the 回 and 日 of the date line: '5回京都4日'
# apprentice marks when the title ('1kg減量') is missing: ☆ 1, △ / ◇ 2, ▲ 3, ★ 4 (female)
_CLAIM_MARKS = {"☆": 1.0, "△": 2.0, "◇": 2.0, "▲": 3.0, "★": 4.0}
_CENTRES = {"栗東": "ritto", "美浦": "miho"}
_SEX = {"牡": "H", "牝": "M", "セ": "G"}
_WEIGHT_TYPES = {"馬齢": "age", "定量": "set", "別定": "allowance", "ハンデ": "handicap"}
# class words, checked in order; 500万下 / 1000万下 / 1600万下 are the names before June 2019
_CLASSES = (
    ("新馬", "DEBUT"), ("未出走", "MAIDEN"), ("未勝利", "MAIDEN"), ("1勝クラス", "C1"),
    ("500万下", "C1"), ("2勝クラス", "C2"), ("1000万下", "C2"), ("3勝クラス", "C3"),
    ("1600万下", "C3"), ("オープン", "OP"),
)  # fmt: skip
_GRADES = {"GI": "G1", "GII": "G2", "GIII": "G3", "G1": "G1", "G2": "G2", "G3": "G3", "L": "L"}
_BLINKERS = "B"


def parse(page: RawPage) -> ParsedPage:
    doc = op.soup(page.body)
    shown, race_no = op.page_meeting(doc), op.race_number(doc)
    if shown is None or shown[1] is None or race_no is None:
        return {"races": [], "racecard_runners": []}
    race_date, venue = shown
    race_id = capture.race_id(race_date, venue, race_no)
    rows = [_runner(row, race_id, page.fetched_utc) for row in _field_rows(doc)]
    runners = _numbers_by_position(rows)
    return {
        "races": [_race(doc, race_id, race_no, race_date, runners)],
        "racecard_runners": runners,
    }


def _field_rows(doc: BeautifulSoup) -> list[Tag]:
    table = doc.select_one("#syutsuba table")
    if table is None:
        return []
    return [
        tr
        for tr in table.find_all("tr")
        if tr.find_parent("table") is table and tr.find("td", class_="horse")
    ]


def _text(node: Tag | None) -> str:
    return op.clean(node.get_text(" ")) if node is not None else ""


def _own_text(node: Tag | None) -> str:
    """The node's own text, without its child elements' ('父：' labels, the dam-sire note)."""
    if node is None:
        return ""
    return op.clean(" ".join(str(c) for c in node.children if isinstance(c, NavigableString)))


def _int(text: str) -> int | None:
    return int(text) if text.isascii() and text.isdigit() else None


def _lb(kg: float | None) -> int | None:
    return round(kg * LB_PER_KG) if kg is not None else None


def _code(cell: Tag | None, pattern: re.Pattern[str]) -> tuple[str | None, str | None]:
    """(code from the first link that matches, its text)."""
    for a in cell.find_all("a") if cell is not None else []:
        link = op.link_of(a)
        m = pattern.match(link.cname) if link else None
        if m:
            return m.group(1), _text(a) or None
    return None, None


# ---------------------------------------------------------------- race


def _race(
    doc: BeautifulSoup, race_id: str, race_no: int, race_date: date, runners: list[dict]
) -> dict[str, Any]:
    """`parse` has found the race header (date line, race number), so `.race_header` exists."""
    header = doc.select_one(".race_header")
    name = _text(header.select_one(".race_name"))
    class_raw = _text(header.select_one(".type .category")) + _text(
        header.select_one(".type .class")
    )
    course = header.select_one(".type .course")
    detail = _text(course.select_one(".detail")) if course else ""
    distance = _DISTANCE.search(_text(course))
    alts = (
        unicodedata.normalize("NFKC", str(img["alt"])).strip()
        for img in header.select(".race_title img[alt]")
    )
    grade = next((m.group(1) for m in map(_GRADE.fullmatch, alts) if m), None)
    post = op.local_clock(race_date, _text(header.select_one(".date_line .time")))
    meeting_round = _ROUND.search(_text(header.select_one(".date_line .date")))
    return {
        "jurisdiction": capture.SOURCE,
        "race_id": race_id,
        "meeting_id": race_id.rsplit(":", 1)[0],
        "race_no": race_no,
        "race_index": None,
        "race_name": name or None,
        "race_class": race_class(class_raw, grade),
        "rating_hi": None,
        "rating_lo": None,
        "distance_m": int(distance.group(1).replace(",", "")) if distance else None,
        "surface": _surface(detail),
        "course_config": "outer" if "外" in detail and "内" not in detail else None,  # inner: null
        "course_letter": None,  # A-D: not printed on the card
        "going": None,  # not printed on the card
        "prize": None,
        "post_time_local": post.isoformat() if post else None,  # Tokyo, with its offset
        "field_size": sum(not r["is_scratched"] for r in runners),
        "status": "declared",
        "weight_type": _WEIGHT_TYPES.get(_text(header.select_one(".type .weight"))),
        "race_class_raw": class_raw or None,
        # the meeting's 回 and 日: JRDB keys an upcoming race by them
        "kai": int(meeting_round.group(1)) if meeting_round else None,
        "nichi": int(meeting_round.group(2)) if meeting_round else None,
    }


def race_class(class_raw: str | None, grade: str | None) -> str | None:
    """'3歳以上1勝クラス' -> C1, '3歳以上500万下' -> C1; an open race takes its grade icon's
    code (GI -> G1, L -> L), else OP."""
    base = next((code for word, code in _CLASSES if class_raw and word in class_raw), None)
    if base in (None, "OP") and grade is not None:
        return _GRADES.get(grade, base)
    return base


def _surface(detail: str) -> str | None:
    """'（ダート・右）' -> dirt, '（芝・右 外）' -> turf; any 障 (障害) -> jump, wherever it stands:
    jump courses mix turf and dirt and are printed in several ways."""
    if "障" in detail:
        return "jump"
    kind = detail.strip("（）() ").split("・")[0]
    return "turf" if kind.startswith("芝") else "dirt" if kind.startswith("ダ") else None


# ---------------------------------------------------------------- runners


def _runner(row: Tag, race_id: str, scraped: datetime | None) -> dict[str, Any]:
    cell = {
        c: row.find("td", class_=c, recursive=False) for c in ("waku", "num", "horse", "jockey")
    }
    num, horse, jockey = cell["num"], cell["horse"], cell["jockey"]
    num_text = _text(num)
    scratched = any(word in num_text for word in op.SCRATCH_WORDS)
    horse_no = _int(num_text.split()[0]) if num_text else None
    waku_img = cell["waku"].find("img") if cell["waku"] is not None else None
    bracket = _BRACKET.search(str(waku_img.get("alt", ""))) if waku_img is not None else None
    horse_id, horse_name = _code(horse, _HORSE)
    if horse_name is None:
        horse_name = _text(horse.select_one(".name")) if horse else None
    jockey_code, jockey_name = _code(jockey.select_one("p.jockey") if jockey else None, _JOCKEY)
    trainer_cell = horse.select_one("p.trainer") if horse else None
    trainer_code, trainer_name = _code(trainer_cell, _TRAINER)
    centre = _text(trainer_cell.select_one(".division") if trainer_cell else None).strip("()（） ")
    carried = _KG.search(_text(jockey.select_one("p.weight") if jockey else None))
    carried_kg = float(carried.group(1)) if carried else None
    claim = _claim(jockey.select_one("p.jockey .mark") if jockey else None)
    body_text = _text(horse.select_one(".result_line .weight") if horse else None)
    body = _BODY.search(body_text.replace(" ", ""))
    body_kg = int(body.group(1)) if body else None
    sign = -1 if body and body.group(2) == "-" else 1
    change = sign * int(body.group(3)) if body and body.group(3) else None
    sex_age = _text(jockey.select_one("p.age") if jockey else None).split("/")[0]
    mare = horse.select_one(".family_line .mare") if horse else None
    blinkers = num is not None and bool(
        num.select_one(".blinker") or num.find("img", alt=re.compile("ブリンカー"))
    )
    dam_sire = _text(mare.select_one(".bloodmare") if mare else None).strip("()（） ")
    return {
        "jurisdiction": capture.SOURCE,
        "race_id": race_id,
        "horse_id": horse_id,
        "id_match": "exact" if horse_id else "none",
        "horse_no": horse_no,  # null where 取消 / 除外 replaces it: see _numbers_by_position
        "horse_name": horse_name or None,
        "brand_raw": None,
        "bracket": int(bracket.group(1)) if bracket else None,
        "draw": horse_no,  # JRA gates are numbered by saddle number
        "actual_wt_kg": carried_kg,
        "actual_wt_lb": _lb(carried_kg),
        "allowance_kg": claim,
        "allowance_lb": _lb(claim),
        "jockey_key": f"jockey:{jockey_code}" if jockey_code else None,
        "jockey_code": jockey_code,
        "jockey_name": jockey_name,
        "trainer_key": f"trainer:{trainer_code}" if trainer_code else None,
        "trainer_code": trainer_code,
        "trainer_name": trainer_name,
        "trainer_centre": _CENTRES.get(centre),
        "horse_wt_kg": body_kg,
        "horse_wt_change_kg": change,
        "declared_horse_wt_lb": _lb(body_kg),
        "sex": _SEX.get(sex_age[:1]),
        "age": _int(sex_age[1:]),
        "gear_raw": _BLINKERS if blinkers else None,
        "owner": _text(horse.select_one("p.owner") if horse else None) or None,
        "sire": _own_text(horse.select_one(".family_line .sire") if horse else None) or None,
        "dam": _own_text(mare) or None,
        "dam_sire": dam_sire.removeprefix("母の父：") or None,
        "is_reserve": False,
        "is_scratched": scratched,
        "card_scraped_utc": scraped,
    }


def _numbers_by_position(runners: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A numbered card lists the field in horse-number order, 1 to N, and prints 取消 / 除外 in
    place of a scratched runner's number. When every printed number equals its row's position,
    the missing ones are their positions too (draw as well); otherwise they stay null."""
    printed = [
        (i, r["horse_no"]) for i, r in enumerate(runners, start=1) if r["horse_no"] is not None
    ]
    if not printed or any(no != i for i, no in printed):
        return runners
    for i, runner in enumerate(runners, start=1):
        if runner["horse_no"] is None:
            runner["horse_no"] = runner["draw"] = i
    return runners


def _claim(mark: Tag | None) -> float | None:
    """Apprentice allowance in kg from the mark before the rider's name: its title ('1kg減量'),
    else the symbol."""
    if mark is None:
        return None
    m = _CLAIM.search(str(mark.get("title") or ""))
    return float(m.group(1)) if m else _CLAIM_MARKS.get(_text(mark))
