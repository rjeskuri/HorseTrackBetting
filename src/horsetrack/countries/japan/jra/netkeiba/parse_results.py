"""netkeiba results page (db.netkeiba.com/race/<id>/, UTF-8) -> meetings, races, runners, dividends.

Ported from HKTrack's `sources/jra/parse_results.py`, with its adapter's results `classify` rule.
Pure: no network. Pages are untrusted data, only parsed.

Page structure: the race header is `dl.racedata` (race name, course, going, post time) plus
`p.smalltxt` (date, class, conditions, weight type); runners are `table.race_table_01`, read by
header text; payouts are the two `table.pay_table_01`; laps are the ラップ row.

Ids: meeting `jra:<date>:<VENUE>` (venue from race-id digits 5-6 via jra.yaml), race
`<meeting>:<RR>`; horses, jockeys and trainers by their netkeiba link ids (names are Japanese and
not unique). There is no incidents table: 取 / 除 / 中 / 失 / 降 / 同着 are runner columns
(placing_raw, placing_code, dead_heat, is_scratched, lbw_raw).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from bs4 import BeautifulSoup, Tag

from horsetrack.core.config import load_source
from horsetrack.core.ingest.types import (
    STATUS_BLOCKED,
    STATUS_EMPTY,
    STATUS_FETCHED,
    ParsedPage,
    RawPage,
)
from horsetrack.countries.japan.jra.netkeiba import common, urls

PARSER_VERSION = "2"  # 2: races.source_race_key
JURISDICTION = "jra"
LB_PER_KG = 2.20462

RESULT_TABLE_SELECTOR = "table.race_table_01"
PAY_TABLE_SELECTOR = "table.pay_table_01"
# the title of the page the site serves (HTTP 200) for an unknown race id
SENTINEL_TITLE = "1970年1月01日"

# runner-table header (as printed, line breaks removed) per field
_HEADERS = {
    "placing": "着順", "bracket": "枠番", "horse_no": "馬番", "horse": "馬名",
    "sex_age": "性齢", "carried_kg": "斤量", "jockey": "騎手", "time": "タイム",
    "margin": "着差", "passing": "通過", "last_3f": "上り", "win_odds": "単勝",
    "fav_rank": "人気", "body_wt": "馬体重", "trainer": "調教師",
}  # fmt: skip
_SEX = {"牡": "H", "牝": "M", "セ": "G"}
_GOING = {"良": "GOOD", "稍重": "GOOD TO YIELDING", "重": "SOFT", "不良": "HEAVY"}
_WEIGHT_TYPES = {"馬齢": "age", "定量": "set", "別定": "allowance", "ハンデ": "handicap"}
# class words in p.smalltxt, checked in order; 500万下 / 1000万下 / 1600万下 are the names before
# June 2019
_CLASSES = (
    ("新馬", "DEBUT"), ("未出走", "MAIDEN"), ("未勝利", "MAIDEN"), ("1勝クラス", "C1"),
    ("500万下", "C1"), ("2勝クラス", "C2"), ("1000万下", "C2"), ("3勝クラス", "C3"),
    ("1600万下", "C3"), ("オープン", "OP"),
)  # fmt: skip
# grade suffix of an open race's name; jump grades (JGI ...) share the codes, surface = jump tells
# them apart
_GRADES = {
    "GI": "G1", "GII": "G2", "GIII": "G3", "G1": "G1", "G2": "G2", "G3": "G3", "L": "L", "OP": "OP",
}  # fmt: skip
# placing as printed -> placing_code; 取消 and 除外 never started, 中止 and 失格 did
_PLACING_CODES = {"取": "WV", "除": "WX", "中": "PU", "失": "DISQ"}
_SCRATCHED_CODES = ("WV", "WX")
# margins to the horse in front, in lengths; 同着 = the second half of a dead heat
_NAMED_MARGINS = {"ハナ": 0.05, "アタマ": 0.1, "クビ": 0.25, "大": 10.0, "大差": 10.0, "同着": 0.0}

_CLASS_TOKEN = re.compile(r"\S*(?:" + "|".join(word for word, _ in _CLASSES) + r")\S*")
_DISTANCE = re.compile(r"([0-9]{3,4})m")
_GOING_TEXT = re.compile(r"(?:芝|ダート)\s*:\s*(\S+)")
_POST_TIME = re.compile(r"発走\s*:\s*([0-9]{1,2}:[0-9]{2})")
_WEIGHT_TYPE = re.compile(r"\((" + "|".join(_WEIGHT_TYPES) + r")\)")
_GRADE = re.compile(r"\((?:J\.?)?(G(?:I{1,3}|[1-3])|L|OP)\)")
_PLACING = re.compile(r"([0-9]+)")
_MARGIN = re.compile(r"([0-9]+)\.([0-9]+)/([0-9]+)|([0-9]+)/([0-9]+)|([0-9]+)")
_TIME = re.compile(r"(?:([0-9]+):)?([0-9]+(?:\.[0-9]+)?)")
_BODY_WT = re.compile(r"([0-9]+)\s*(?:\(\s*([+-]?[0-9]+)\s*\))?")
_HORSE_LINK = re.compile(r"/horse/([0-9A-Za-z]{10})/?$")
_PERSON_LINK = {
    role: re.compile(rf"/{role}/(?:result/recent/)?([0-9A-Za-z]{{5}})/?$")
    for role in ("jockey", "trainer")
}
_RACE_LINK = re.compile(r"/race/([0-9]{12})/?$")


def classify(body: bytes) -> str:
    """Decided by content, never by HTTP status: the site serves an unknown race as 200."""
    soup = BeautifulSoup(body, "lxml")
    if soup.select_one(RESULT_TABLE_SELECTOR) is not None:
        return STATUS_FETCHED
    title = soup.find("title")
    if title is not None and SENTINEL_TITLE in title.get_text():
        return STATUS_EMPTY
    return STATUS_BLOCKED  # an error page, or a race page cut off before its result table


def parse(page: RawPage) -> ParsedPage:
    doc = BeautifulSoup(page.body, "lxml")
    meta: dict[str, Any] = {
        "page_type": page.page_type, "key": page.key, "url": page.url,
        "parser_version": PARSER_VERSION,
    }  # fmt: skip
    out: ParsedPage = {"meetings": [], "races": [], "runners": [], "dividends": [], "_meta": [meta]}
    table = doc.select_one(RESULT_TABLE_SELECTOR)
    shown_id = _shown_race_id(doc)
    netkeiba_id = urls.jra_race_id(page.key) or shown_id
    race_date = common.race_date(doc)
    venue = common.venue_code(netkeiba_id) if netkeiba_id else None
    if table is None or race_date is None or netkeiba_id is None or venue is None:
        return out  # not a result (or not a JRA race): nothing to place in the tables

    race_no = int(netkeiba_id[10:12])
    meeting_id = f"{JURISDICTION}:{race_date.isoformat()}:{venue}"
    race_id = common.store_race_id(race_date, venue, race_no)
    header = common.text(doc.select_one("dl.racedata dd p span"))
    runners = _runners(table, race_id, jump=_surface(_course(header)) == "jump")
    race = _race(doc, header, race_id, meeting_id, race_no, runners)
    # the key other netkeiba pages of this race (pedigree card) use
    race["source_race_key"] = netkeiba_id
    meta.update(race_id=race_id, race_date=race_date, venue=venue, race_no=race_no)
    meta["key_mismatch"] = shown_id is not None and shown_id != netkeiba_id
    meeting = {
        "jurisdiction": JURISDICTION, "meeting_id": meeting_id, "race_date": race_date,
        "venue": venue, "status": race["status"],
    }  # fmt: skip
    out.update(
        meetings=[meeting], races=[race], runners=runners, dividends=_dividends(doc, race_id)
    )
    return out


# ---------------------------------------------------------------- text helpers


def _int(text: str | None) -> int | None:
    t = common.clean(text).replace(",", "")
    return int(t) if t.isascii() and t.isdigit() else None


def _float(text: str | None) -> float | None:
    try:
        return float(common.clean(text).replace(",", ""))
    except ValueError:
        return None


def _lb(kg: float | None) -> int | None:
    return round(kg * LB_PER_KG) if kg is not None else None


# ---------------------------------------------------------------- identity


def _shown_race_id(doc: BeautifulSoup) -> str | None:
    """The race id the page itself marks as current in its race-number strip."""
    link = doc.select_one("div.race_num a.active[href]")
    m = _RACE_LINK.search(str(link.get("href") or "")) if link is not None else None
    return urls.jra_race_id(m.group(1)) if m else None


# ---------------------------------------------------------------- race


def _race(
    doc: BeautifulSoup,
    header: str,
    race_id: str,
    meeting_id: str,
    race_no: int,
    runners: list[dict[str, Any]],
) -> dict[str, Any]:
    """`header`: the course line, e.g. '芝右 外1600m / 天候 : 晴 / 芝 : 良 / 発走 : 15:40'."""
    course = _course(header)
    going = _GOING_TEXT.search(header)
    post_time = _POST_TIME.search(header)
    name = common.text(doc.select_one("dl.racedata h1")) or None
    conditions = common.text(doc.select_one("p.smalltxt"))
    class_raw = next(iter(_CLASS_TOKEN.findall(conditions)), None)
    weight_type = _WEIGHT_TYPE.search(conditions)
    distance = _DISTANCE.search(course)
    splits = _split_times(doc)
    winner_time = next((r["finish_time_s"] for r in runners if r["finish_pos"] == 1), None)
    if winner_time is None and splits:
        winner_time = splits[-1]
    run = any(r["finish_pos"] is not None for r in runners)
    return {
        "jurisdiction": JURISDICTION,
        "race_id": race_id,
        "meeting_id": meeting_id,
        "race_no": race_no,
        "race_index": None,
        "race_name": name,
        "race_class": race_class(class_raw, name),
        "rating_hi": None,
        "rating_lo": None,
        "distance_m": int(distance.group(1)) if distance else None,
        "surface": _surface(course),
        "course_config": _course_config(course),
        "going": _GOING.get(going.group(1)) if going else None,
        "prize": None,
        "post_time_local": post_time.group(1) if post_time else None,
        "field_size": sum(not r["is_scratched"] for r in runners),
        "finish_time_s": winner_time if run else None,
        "split_times_s": splits,
        "status": "run" if run else "abandoned",
        "weight_type": _WEIGHT_TYPES[weight_type.group(1)] if weight_type else None,
        "race_class_raw": class_raw,
        "going_raw": going.group(1) if going else None,
    }


def race_class(class_raw: str | None, race_name: str | None) -> str | None:
    """'3歳以上1勝クラス' -> C1, '3歳以上500万下' -> C1; an open race takes the grade of its name,
    e.g. 'マイルチャンピオンS(GI)' -> G1, '中山大障害(JGI)' -> G1, '室町S(OP)' -> OP."""
    base = next((code for word, code in _CLASSES if class_raw and word in class_raw), None)
    if base in (None, "OP"):
        m = _GRADE.search(unicodedata.normalize("NFKC", race_name or ""))
        if m:
            return _GRADES[m.group(1)]
    return base


def _course(header: str) -> str:
    return header.split("/")[0].strip()


def _surface(course: str) -> str | None:
    """'障芝 外-内3570m' -> jump (kept, flagged by surface); '芝右 外1600m' -> turf; 'ダ右1800m' ->
    dirt."""
    if course.startswith("障"):
        return "jump"
    return {"芝": "turf", "ダ": "dirt"}.get(course[:1])


def _course_config(course: str) -> str | None:
    """内 / 外 course of the track; null when the race uses neither or both (jumps crossing
    over)."""
    inner, outer = "内" in course, "外" in course
    return "inner" if inner and not outer else "outer" if outer and not inner else None


def _split_times(doc: BeautifulSoup) -> list[float]:
    """Cumulative leader times from the ラップ row ('12.2 - 10.7 - ...'); jump races print none."""
    for row in doc.select("table.result_table_02 tr"):
        if common.text(row.find("th")) == "ラップ":
            parts = common.text(row.find("td")).split("-")
            laps = [lap for lap in map(_float, parts) if lap is not None]
            if not laps or len(laps) != len(parts):
                return []
            out, total = [], 0.0
            for lap in laps:
                total = round(total + lap, 1)
                out.append(total)
            return out
    return []


# ---------------------------------------------------------------- runners


def _runners(table: Tag, race_id: str, jump: bool) -> list[dict[str, Any]]:
    rows = table.find_all("tr")
    if not rows:
        return []
    columns: dict[str, int] = {}
    for i, th in enumerate(rows[0].find_all("th")):
        columns.setdefault(common.clean(th.get_text("")).replace(" ", ""), i)
    at = {field: columns.get(header) for field, header in _HEADERS.items()}
    runners = [
        _runner(row.find_all("td"), at, race_id) for row in rows[1:] if row.find("td") is not None
    ]
    # a jump race's 上り is the mean time per 200 m over the whole race, not the last 600 m
    if jump:
        for runner in runners:
            runner["last_3f_s"] = None
    _mark_dead_heats(runners)
    _beaten_lengths(runners)
    return runners


def _runner(cells: list[Tag], at: dict[str, int | None], race_id: str) -> dict[str, Any]:
    def cell(field: str) -> Tag | None:
        i = at.get(field)
        return cells[i] if i is not None and i < len(cells) else None

    def text(field: str) -> str:
        return common.text(cell(field))

    placing = _placing(text("placing"))
    horse_id, horse_name = _linked(cell("horse"), _HORSE_LINK)
    jockey_code, jockey_name = _linked(cell("jockey"), _PERSON_LINK["jockey"])
    trainer_code, trainer_name = _linked(cell("trainer"), _PERSON_LINK["trainer"])
    sex_age = text("sex_age")
    carried_kg = _float(text("carried_kg"))
    body = _BODY_WT.fullmatch(text("body_wt"))
    horse_wt_kg = int(body.group(1)) if body else None
    horse_no = _int(text("horse_no"))
    passing = text("passing").split("-")
    return {
        "jurisdiction": JURISDICTION,
        "race_id": race_id,
        "horse_id": horse_id,
        "horse_no": horse_no,
        "horse_name": horse_name,
        "brand_raw": None,
        "jockey_key": f"jockey:{jockey_code}" if jockey_code else None,
        "jockey_code": jockey_code,
        "jockey_name": jockey_name,
        "trainer_key": f"trainer:{trainer_code}" if trainer_code else None,
        "trainer_code": trainer_code,
        "trainer_name": trainer_name,
        "actual_wt_lb": _lb(carried_kg),
        "declared_horse_wt_lb": _lb(horse_wt_kg),
        "draw": horse_no,  # JRA gates are numbered by saddle number
        **placing,
        "lbw_raw": text("margin") or None,
        "lbw_lengths": None,  # set over the whole field, see _beaten_lengths
        "running_positions": [int(p) for p in passing if p.isascii() and p.isdigit()],
        "finish_time_s": _seconds(text("time")),
        "win_odds": _float(text("win_odds")),
        "is_scratched": placing["placing_code"] in _SCRATCHED_CODES,
        "bracket": _int(text("bracket")),
        "actual_wt_kg": carried_kg,
        "horse_wt_kg": horse_wt_kg,
        "horse_wt_change_kg": int(body.group(2)) if body and body.group(2) else None,
        "last_3f_s": _float(text("last_3f")),
        "win_fav_rank": _int(text("fav_rank")),
        "sex": _SEX.get(sex_age[:1]),
        "age": _int(sex_age[1:]),
    }


def _linked(cell: Tag | None, pattern: re.Pattern[str]) -> tuple[str | None, str | None]:
    """(id from the link, name). Without a matching link the name is the cell text minus a '[西]'
    prefix."""
    if cell is None:
        return None, None
    for link in cell.find_all("a", href=True):
        m = pattern.search(str(link["href"]))
        if m:
            title = common.clean(str(link.get("title") or ""))
            return m.group(1), common.text(link) or title or None
    return None, re.sub(r"^\[.\]\s*", "", common.text(cell)) or None


def _placing(text: str) -> dict[str, Any]:
    """'3' | '4(降)' (demoted: the official place, the marker stays in placing_raw) | '取' / '除' /
    '中' / '失'."""
    raw = text or None
    m = _PLACING.match(raw or "")
    if m:
        finish_pos, code = int(m.group(1)), None
    else:
        finish_pos, code = None, _PLACING_CODES.get(raw or "", raw)
    return {"placing_raw": raw, "finish_pos": finish_pos, "placing_code": code, "dead_heat": False}


def _mark_dead_heats(runners: list[dict[str, Any]]) -> None:
    shared = [r["finish_pos"] for r in runners if r["finish_pos"] is not None]
    for runner in runners:
        if runner["finish_pos"] is not None and shared.count(runner["finish_pos"]) > 1:
            runner["dead_heat"], runner["placing_code"] = True, "DH"


def _beaten_lengths(runners: list[dict[str, Any]]) -> None:
    """Lengths behind the winner: the running sum of the margins down the placed order. A margin
    that cannot be read leaves it unknown from there on. Runners without a place (incl. 失格, whose
    printed margin is to the horse it physically followed) get none."""
    total: float | None = 0.0
    for runner in runners:
        if runner["finish_pos"] is None:
            continue
        if runner["finish_pos"] == 1:
            total = 0.0
        else:
            step = margin_lengths(runner["lbw_raw"])
            total = None if total is None or step is None else round(total + step, 2)
        runner["lbw_lengths"] = total


def margin_lengths(raw: str | None) -> float | None:
    """'ハナ' 0.05, 'アタマ' 0.1, 'クビ' 0.25, '3/4', '2', '1.1/2' (= 1 1/2), '大' 10."""
    text = common.clean(raw)
    if text in _NAMED_MARGINS:
        return _NAMED_MARGINS[text]
    m = _MARGIN.fullmatch(text)
    if not m:
        return None
    if m.group(1):
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    if m.group(4):
        return int(m.group(4)) / int(m.group(5))
    return float(m.group(6))


def _seconds(text: str) -> float | None:
    """'1:31.3' -> 91.3; '0:56.7' -> 56.7; '' -> None."""
    m = _TIME.fullmatch(text)
    if not m:
        return None
    return round(int(m.group(1) or 0) * 60 + float(m.group(2)), 2)


# ---------------------------------------------------------------- dividends


def _dividends(doc: BeautifulSoup, race_id: str) -> list[dict[str, Any]]:
    """One row per paid combination; '6-15' -> '6,15' (page order kept, so ordered pools keep
    theirs)."""
    source = load_source(common.COUNTRY, common.SOURCE)
    out = []
    for table in doc.select(PAY_TABLE_SELECTOR):
        for row in table.find_all("tr"):
            label = common.text(row.find("th"))
            cells = row.find_all("td")
            if not label or len(cells) < 2:
                continue
            pool = source.pool_by_label(label)
            combinations = cells[0].get_text("\n", strip=True).splitlines()
            amounts = cells[1].get_text("\n", strip=True).splitlines()
            for combination, amount in zip(combinations, amounts, strict=False):
                out.append(
                    {
                        "jurisdiction": JURISDICTION,
                        "race_id": race_id,
                        "pool": pool.code if pool is not None else label,
                        "combination": _combination(combination),
                        "dividend": _float(amount),
                        "is_refund": False,
                    }
                )
    return out


def _combination(text: str) -> str | None:
    tokens = [t for t in re.split(r"[-→>,\s]+", common.clean(text)) if t]
    return ",".join(str(int(t)) if t.isascii() and t.isdigit() else t for t in tokens) or None
