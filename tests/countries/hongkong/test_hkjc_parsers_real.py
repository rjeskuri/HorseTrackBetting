"""The ported HKJC parsers against the real pages (ticket #67).

Every test is marked `local` and uses `real_fixtures` (tests/conftest.py): without
$HORSETRACK_DATA_DIR/fixtures the file is skipped, so CI never runs it.

Pages (`fixtures/hkjc/`, copied from HKTrack's tests/fixtures/hkjc; most start with an HTML
comment naming the requested URL and the fetch time, several are trimmed to the content block):
- results `localresults_*.html`: one race per layout era (2004 without running positions, the
  2010 archive with a narrative stewards' report, a 2020 dead heat, a 2024 all-weather race with a
  withdrawn runner, the 2026 live route with per-horse incidents and simulcast tabs), the pages
  HKTrack's backfill tripped over (truncated odds, a dividend quoted per $1 unit bet), the races
  without a result (an abandoned meeting, the 2024+ header-less banner, a void race), a protest
  relegation, and the "No information." page of both routes.
- race cards `racecard_*.html` of 23 Sep 2026 HV R1 and R7, an archived ST Group One card, the
  "No information." card and the "not available" page; the calendar cell of 23 Sep 2026.
- the entries page of 23 Sep 2026 HV (View=All); the comments on running of 1 Jan 2010 ST R1.
- horse profiles: SPICY SPANGLE (active, all seasons) and BRAVE HEART (retired, otherhorse route).
- the sectional times of 16 Sep 2026 HV R1; the fixture calendars of Sep 2026, Jan 2010, Nov 2019.
The `odds_*.json` files belong to the paused live odds and are not used here.

Expectations are ported from HKTrack's tests/unit/sources/hkjc/test_{results,racecard,entries,
corunning,horse,sectional,fixture}_parser.py and the real-page cases of test_tasks.py
(classify), where they were read off the pages by hand. Derived cases edit the real page text in
memory, as HKTrack's did. Cross-page checks join pages of the same race (results, sectionals,
comments on running, the horse's form row), so each page's values are checked against another
page's print of them, never against the parser.
"""

from __future__ import annotations

import html
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from functools import cache
from pathlib import Path

import pytest
from bs4 import BeautifulSoup
from pytest import approx

from horsetrack.core.ingest.types import (
    STATUS_ABANDONED,
    STATUS_BLOCKED,
    STATUS_EMPTY,
    STATUS_FETCHED,
    RawPage,
)
from horsetrack.countries.hongkong.hkjc import (
    parse_corunning,
    parse_entries,
    parse_fixture,
    parse_horse,
    parse_racecard,
    parse_results,
    parse_sectional,
)
from horsetrack.countries.hongkong.hkjc.adapter import ADAPTER

pytestmark = pytest.mark.local

UTC = timezone.utc
UNIT_STAKE = 10  # HK$: every HKJC dividend is quoted per $10
ODDS_STEP_THRESHOLD = 10  # displayed odds: 1 d.p. below 10, whole numbers from 10 (truncated)


@pytest.fixture
def hkjc_dir(real_fixtures: Path) -> Path:
    return real_fixtures / "hkjc"


@cache
def _read(path: Path) -> bytes:
    return path.read_bytes()


def body_of(hkjc_dir: Path, name: str) -> bytes:
    return _read(hkjc_dir / name)


def text_of(hkjc_dir: Path, name: str) -> str:
    return body_of(hkjc_dir, name).decode("utf-8")


def raw(
    page_type: str,
    key: str,
    body: bytes | str,
    url: str = "https://racing.hkjc.com/x",
    fetched: datetime | None = None,
) -> RawPage:
    data = body.encode() if isinstance(body, str) else body
    return RawPage("hkjc", page_type, key, url, data, fetched_utc=fetched)


# People's names and the stewards' and running comments are not written in this public file:
# each parsed value is checked against the page's own print of it (the cell under its column
# header in the horse's row, the text of its profile link), against another page of the same
# race, and by structure. Keys and codes are pinned.


def norm(text: str) -> str:
    """Unescaped, whitespace-normalised."""
    return " ".join(html.unescape(text).split())


@cache
def _soup(body: bytes) -> BeautifulSoup:
    return BeautifulSoup(body, "lxml")


@cache
def page_text(body: bytes) -> str:
    """The page's visible text, unescaped and whitespace-normalised."""
    return norm(_soup(body).get_text(" "))


def printed(body: bytes, header: str, horse_id: str | None = None, on: str | None = None) -> str:
    """The cell under `header` in the one table row that links to `horse_id` (or prints `on`
    under 'Date'), unescaped and whitespace-normalised."""
    found = []
    for table in _soup(body).find_all("table"):
        first = table.find("tr")
        if first is None:
            continue
        head = [norm(c.get_text(" ")) for c in first.find_all(["td", "th"], recursive=False)]
        if header not in head or (on is not None and "Date" not in head):
            continue
        for tr in table.find_all("tr"):
            cells = [norm(td.get_text(" ")) for td in tr.find_all("td", recursive=False)]
            if len(cells) != len(head) or tr is first:
                continue
            if horse_id is not None and tr.select_one(f'a[href*="horseid={horse_id}"]') is None:
                continue
            if on is not None and cells[head.index("Date")] != on:
                continue
            found.append(cells[head.index(header)])
    (cell,) = found
    return cell


def linked(body: bytes, role: str, code: str) -> set[str]:
    """The texts of the page's profile links for jockey or trainer `code`."""
    pattern = re.compile(rf"{role}id={re.escape(code)}(?:&|$)", re.I)
    return {norm(a.get_text(" ")) for a in _soup(body).find_all("a", href=pattern)}


def name_key(role: str, name: str) -> str:
    """The join key, derived here: the role, then the name lower-cased with every run of other
    characters one space ('P X'Y' -> 'p x y')."""
    return f"{role}:" + " ".join(re.sub(r"[^a-z0-9]+", " ", name.lower()).split())


def check_person(row: dict, role: str, body: bytes, header: str | None) -> None:
    """The row's jockey or trainer as the page prints it in the horse's row (an apprentice's
    claim printed after the name) and on its profile link, with the key derived from the name."""
    name, code, key = (row[f"{role}_{part}"] for part in ("name", "code", "key"))
    assert name and name == norm(name) and re.search("[A-Za-z]", name), row["horse_id"]
    assert key == name_key(role, name), row["horse_id"]
    claim = row.get("allowance_lb") if role == "jockey" else None
    shown = f"{name} (-{claim})" if claim else name
    if header is not None:
        assert printed(body, header, horse_id=row["horse_id"]) == shown, row["horse_id"]
    if code is not None:
        assert linked(body, role, code) == {shown}, row["horse_id"]


def sentences(text: str) -> list[str]:
    return re.split(r"(?<=\.)\s+", norm(text))


def names_in(sentence: str, names: dict[str, str]) -> set[str]:
    """The horse ids whose names `sentence` prints (a typographic apostrophe reads as ')."""
    flat = sentence.replace("’", "'")
    return {h for h, n in names.items() if re.search(rf"(?<![A-Z]){re.escape(n)}(?![A-Z])", flat)}


# ================================================================ results

LIVE_2026 = "localresults_2026-09-16_HV_R1.html"
ARCHIVE_2010 = "localresults_archive_2010-01-01_ST_R1.html"
DEAD_HEAT = "localresults_archive_2020-12-09_HV_R2_deadheat.html"
AWT_2024 = "localresults_archive_2024-04-20_ST_R6_awt.html"
LAYOUT_2004 = "localresults_archive_2004-03-07_ST_R1_no_running_pos.html"
SIMULCAST = "localresults_2026-09-13_ST_R1_simulcast_tabs.html"
ABANDONED = "localresults_archive_2019-11-13_HV_R1_abandoned.html"
NO_INFO = "localresults_no_information_2000-01-01.html"
NO_INFO_LIVE = "localresults_no_information_2026-09-20_ST_live_inline_css.html"
ODDS_TRUNCATED = "localresults_2026-09-13_ST_R4_odds_truncated.html"
UNIT_BET = "localresults_2026-09-13_ST_R7_dividend_per_unit_bet.html"
DECLARED_ABANDONED = "localresults_archive_2024-11-13_HV_R7_declared_abandoned.html"
VOID = "localresults_archive_2025-11-15_ST_R8_void.html"
PROTEST = "localresults_archive_2026-03-08_ST_R5_protest_relegation.html"

# page -> (cache key, classify status)
RESULT_PAGES = {
    LIVE_2026: ("2026-09-16_HV_R01", STATUS_FETCHED),
    ARCHIVE_2010: ("2010-01-01_ST_R01", STATUS_FETCHED),
    DEAD_HEAT: ("2020-12-09_HV_R02", STATUS_FETCHED),
    AWT_2024: ("2024-04-20_ST_R06", STATUS_FETCHED),
    LAYOUT_2004: ("2004-03-07_ST_R01", STATUS_FETCHED),
    SIMULCAST: ("2026-09-13_ST_R01", STATUS_FETCHED),
    ODDS_TRUNCATED: ("2026-09-13_ST_R04", STATUS_FETCHED),
    UNIT_BET: ("2026-09-13_ST_R07", STATUS_FETCHED),
    PROTEST: ("2026-03-08_ST_R05", STATUS_FETCHED),
    ABANDONED: ("2019-11-13_HV_R01", STATUS_ABANDONED),
    DECLARED_ABANDONED: ("2024-11-13_HV_R07", STATUS_ABANDONED),
    VOID: ("2025-11-15_ST_R08", STATUS_ABANDONED),  # annulled after the start: no result either
    NO_INFO: ("2000-01-01_ST_R01", STATUS_EMPTY),
    NO_INFO_LIVE: ("2026-09-20_ST_R01", STATUS_EMPTY),  # the live sentinel keeps inline CSS
}
RUN_PAGES = {LIVE_2026, ARCHIVE_2010, DEAD_HEAT, AWT_2024, LAYOUT_2004, SIMULCAST, ODDS_TRUNCATED,
             UNIT_BET}  # fmt: skip

RESULT_TABLES = {"races", "runners", "dividends", "incidents", "_meta"}
RACE_COLUMNS = {
    "jurisdiction", "race_id", "meeting_id", "race_no", "race_index", "race_name", "race_class",
    "rating_hi", "rating_lo", "distance_m", "surface", "course_config", "going", "prize",
    "post_time_local", "field_size", "finish_time_s", "split_times_s", "status",
}  # fmt: skip
RUNNER_COLUMNS = {
    "jurisdiction", "race_id", "horse_id", "horse_no", "horse_name", "brand_raw", "jockey_key",
    "jockey_code", "jockey_name", "trainer_key", "trainer_code", "trainer_name", "actual_wt_lb",
    "declared_horse_wt_lb", "draw", "placing_raw", "finish_pos", "placing_code", "dead_heat",
    "lbw_raw", "lbw_lengths", "running_positions", "finish_time_s", "win_odds", "is_scratched",
}  # fmt: skip
DIVIDEND_COLUMNS = {"jurisdiction", "race_id", "pool", "combination", "dividend", "is_refund"}
INCIDENT_COLUMNS = {"jurisdiction", "race_id", "horse_id", "horse_no", "text"}


@cache
def results(hkjc_dir: Path, name: str, key: str | None = None) -> dict:
    page_key = key if key is not None else RESULT_PAGES[name][0]
    return parse_results.parse(raw("results", page_key, body_of(hkjc_dir, name)))


def by_no(parsed: dict, horse_no: int, table: str = "runners") -> dict:
    (row,) = [r for r in parsed[table] if r["horse_no"] == horse_no]
    return row


def pool(parsed: dict, code: str) -> list[tuple[str | None, float | None]]:
    return [(d["combination"], d["dividend"]) for d in parsed["dividends"] if d["pool"] == code]


def unattributed(parsed: dict) -> list[dict]:
    return [i for i in parsed["incidents"] if i["horse_id"] is None]


def _live_2026(hkjc_dir: Path, parsed: dict) -> None:
    (race,) = parsed["races"]
    assert race == {
        "jurisdiction": "hkjc", "race_id": "hkjc:2026-09-16:HV:01",
        "meeting_id": "hkjc:2026-09-16:HV", "race_no": 1, "race_index": 29,
        "race_name": "HOI MEI HANDICAP", "race_class": "Class 5", "rating_hi": 40, "rating_lo": 0,
        "distance_m": 1000, "surface": "turf", "course_config": "B", "going": "GOOD",
        "prize": 875000, "post_time_local": None, "field_size": 12,
        "finish_time_s": approx(57.54), "split_times_s": approx([12.80, 34.75, 57.54]),
        "status": "run",
    }  # fmt: skip
    assert len(parsed["runners"]) == 12
    first = dict(parsed["runners"][0])
    check_person(first, "jockey", body_of(hkjc_dir, LIVE_2026), "Jockey")
    check_person(first, "trainer", body_of(hkjc_dir, LIVE_2026), "Trainer")
    for column in ("jockey_name", "jockey_key", "trainer_name", "trainer_key"):
        del first[column]
    assert first == {
        "jurisdiction": "hkjc", "race_id": "hkjc:2026-09-16:HV:01", "horse_id": "HK_2023_J087",
        "horse_no": 11, "horse_name": "SPICY SPANGLE", "brand_raw": "J087",
        "jockey_code": "LDE", "trainer_code": "MWK", "actual_wt_lb": 123,
        "declared_horse_wt_lb": 1111, "draw": 3, "placing_raw": "1",
        "finish_pos": 1, "placing_code": None, "dead_heat": False, "lbw_raw": "---",
        "lbw_lengths": 0.0, "running_positions": [1, 1, 1], "finish_time_s": approx(57.54),
        "win_odds": 4.5, "is_scratched": False,
    }  # fmt: skip
    second, last = by_no(parsed, 12), by_no(parsed, 5)
    assert (second["finish_pos"], second["lbw_raw"], second["lbw_lengths"]) == (2, "1-3/4", 1.75)
    assert second["running_positions"] == [7, 5, 2]
    assert second["win_odds"] == 18.0 and isinstance(second["win_odds"], float)
    assert (last["finish_pos"], last["lbw_lengths"], last["finish_time_s"]) == (
        12,
        6.0,
        approx(58.50),
    )
    assert [r["finish_pos"] for r in parsed["runners"]] == list(range(1, 13))

    assert pool(parsed, "WIN") == [("11", 45.0)]
    assert pool(parsed, "PLACE") == [("11", 16.0), ("12", 46.5), ("6", 46.0)]
    assert pool(parsed, "QUINELLA") == [("11,12", 355.0)]
    assert pool(parsed, "QUINELLA_PLACE") == [("11,12", 113.5), ("6,11", 131.0), ("6,12", 370.5)]
    assert pool(parsed, "FORECAST") == [("11,12", 544.0)]
    assert pool(parsed, "TIERCE") == [("11,12,6", 6789.0)]
    assert pool(parsed, "TRIO") == [("6,11,12", 1674.0)]
    assert pool(parsed, "FIRST_4") == [("3,6,11,12", 4737.0)]
    assert pool(parsed, "QUARTET") == [("11,12,6,3", 78710.0)]
    assert len(parsed["dividends"]) == 13
    assert not any(d["is_refund"] for d in parsed["dividends"])

    incidents = parsed["incidents"]  # the 2026 per-horse table, in the runner table's order
    assert len(incidents) == 12
    assert {k: v for k, v in incidents[0].items() if k != "text"} == {
        "jurisdiction": "hkjc", "race_id": "hkjc:2026-09-16:HV:01", "horse_id": "HK_2023_J087",
        "horse_no": 11,
    }  # fmt: skip
    assert [i["horse_id"] for i in incidents] == [r["horse_id"] for r in parsed["runners"]]
    body = body_of(hkjc_dir, LIVE_2026)
    for i in incidents:  # each row is the horse's own cell of the Incident column
        assert i["text"] and i["text"] == printed(body, "Incident", horse_id=i["horse_id"])
    assert len({i["text"] for i in incidents}) > 2  # not one cell repeated down the table
    assert all(i["horse_id"] and i["horse_no"] for i in incidents)

    (meta,) = parsed["_meta"]
    assert (meta["race_no"], meta["n_races"], meta["race_nos"]) == (1, 8, list(range(1, 9)))
    assert (meta["page_type"], meta["key_mismatch"]) == ("results", False)
    # the page decides the race; a key naming another race is flagged, not obeyed
    wrong = results(hkjc_dir, LIVE_2026, "2026-09-16_HV_R07")
    assert wrong["races"][0]["race_id"] == "hkjc:2026-09-16:HV:01"
    assert wrong["_meta"][0]["key_mismatch"] is True


def _archive_2010(hkjc_dir: Path, parsed: dict) -> None:
    (race,) = parsed["races"]
    assert race["race_id"] == "hkjc:2010-01-01:ST:01"
    assert (race["race_index"], race["race_class"], race["rating_hi"], race["rating_lo"]) == (
        275, "Class 4", 60, 40)  # fmt: skip
    assert (race["distance_m"], race["surface"], race["course_config"], race["going"]) == (
        1200, "turf", "B+2", "GOOD")  # fmt: skip
    assert (race["prize"], race["field_size"], race["status"]) == (625000, 14, "run")
    assert race["finish_time_s"] == approx(70.12)
    assert race["split_times_s"] == approx([23.77, 46.46, 70.12])

    winner = by_no(parsed, 5)  # keyed on the href id, never the displayed brand
    assert (winner["horse_id"], winner["brand_raw"], winner["horse_name"]) == (
        "HK_2008_K277", "CK277", "BRAVE HEART")  # fmt: skip
    # retired people print without a link: a name key, no code
    assert (winner["jockey_code"], winner["trainer_code"]) == (None, None)
    check_person(winner, "jockey", body_of(hkjc_dir, ARCHIVE_2010), "Jockey")
    check_person(winner, "trainer", body_of(hkjc_dir, ARCHIVE_2010), "Trainer")
    assert "'" in winner["trainer_name"]  # printed '&#39;': unescaped, a space in the key
    assert by_no(parsed, 7)["trainer_code"] == "SJJ"
    assert by_no(parsed, 1)["jockey_code"] == "CML"
    assert by_no(parsed, 10)["jockey_code"] == "PZ"

    assert {d["pool"] for d in parsed["dividends"]} == {
        "WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE", "TIERCE", "TRIO", "FIRST_4"}  # fmt: skip
    assert pool(parsed, "WIN") == [("5", 20.5)]
    assert pool(parsed, "TIERCE") == [("5,7,3", 1762.0)]
    assert parsed["_meta"][0]["n_races"] == 11

    # the narrative report: one unattributed row with the full text first, then one per named horse
    report = _narrative(hkjc_dir, ARCHIVE_2010, parsed)
    assert len(report) == 16 and "Racing Incident Report" not in " ".join(report)
    attributed = [i for i in parsed["incidents"] if i["horse_id"] is not None]
    # 12 of the 14 runners are named; FULL BLESSINGS (14) and MARS EXPRESS (12) are not
    assert sorted(i["horse_no"] for i in attributed) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]
    assert len(parsed["incidents"]) == 13
    king = dict(by_no(parsed, 8, "incidents"))
    assert king.pop("text") == report[0]  # the report opens on KING GALILEO's one sentence
    assert king == {
        "jurisdiction": "hkjc", "race_id": "hkjc:2010-01-01:ST:01", "horse_id": "HK_2009_L028",
        "horse_no": 8,
    }  # fmt: skip
    pachinko = sentences(by_no(parsed, 9, "incidents")["text"])  # a sentence naming four horses
    assert len(pachinko) == 2 and all("PACHINKO" in s for s in pachinko)
    assert pachinko[0] in by_no(parsed, 7, "incidents")["text"]  # GOLD RUSH's too
    glory = by_no(parsed, 6, "incidents")
    assert glory["horse_id"] == "HK_2009_L066"
    assert glory["text"].count("GLORY") == 4 and len(sentences(glory["text"])) == 4
    assert "KING GALILEO" not in glory["text"]
    assert len(sentences(by_no(parsed, 1, "incidents")["text"])) == 1


def _narrative(hkjc_dir: Path, name: str, parsed: dict) -> list[str]:
    """The narrative stewards' report against the page: the unattributed first row is the whole
    report as printed; each named horse's row, in the runner table's order, is the report's
    sentences that name it, a follow-up sentence naming no horse going with the horses of the
    sentence before. Returns the report's sentences."""
    (whole,) = unattributed(parsed)
    assert parsed["incidents"][0] is whole and whole["horse_no"] is None
    report = sentences(whole["text"])
    assert len(report) > 1 and " ".join(report) in page_text(body_of(hkjc_dir, name))
    names = {r["horse_id"]: r["horse_name"] for r in parsed["runners"]}
    owners: list[set[str]] = []
    for sentence in report:
        owners.append(names_in(sentence, names) or (owners[-1] if owners else set()))
    pairs = list(zip(report, owners, strict=True))
    expected = {h: " ".join(s for s, o in pairs if h in o) for h in names}
    attributed = [i for i in parsed["incidents"] if i["horse_id"] is not None]
    assert [(i["horse_id"], norm(i["text"])) for i in attributed] == [
        (h, t) for h, t in expected.items() if t]  # fmt: skip
    assert all(names[i["horse_id"]] in i["text"].replace("\u2019", "'") for i in attributed)
    return report


def _dead_heat(hkjc_dir: Path, parsed: dict) -> None:
    winners = [r for r in parsed["runners"] if r["finish_pos"] == 1]
    assert sorted(r["horse_no"] for r in winners) == [5, 12]
    for r in winners:
        assert (r["placing_raw"], r["placing_code"], r["dead_heat"]) == ("1 DH", "DH", True)
        assert (r["lbw_raw"], r["lbw_lengths"]) == ("DH", 0.0)
    third = by_no(parsed, 8)
    assert (third["finish_pos"], third["dead_heat"]) == (3, False)
    assert (third["placing_code"], third["lbw_lengths"]) == (None, 0.1)
    assert sum(r["dead_heat"] for r in parsed["runners"]) == 2

    # the pool label carries forward over the dead-heat rows
    assert pool(parsed, "WIN") == [("5", 21.0), ("12", 60.0)]
    assert pool(parsed, "PLACE") == [("5", 16.0), ("12", 36.5), ("8", 71.0)]
    assert pool(parsed, "FORECAST") == [("5,12", 258.0), ("12,5", 420.0)]
    assert pool(parsed, "TIERCE") == [("5,12,8", 5162.0), ("12,5,8", 7965.0)]
    assert pool(parsed, "QUARTET") == [("5,12,8,11", 47318.0), ("12,5,8,11", 75412.0)]
    assert pool(parsed, "1ST_DOUBLE") == [("2/5", 119.5), ("2/12", 826.0)]
    # composite win skips its NOT WIN and detail rows
    assert pool(parsed, "3_PICK_1_COMPOSITE_WIN") == [("A1", 10.1), ("A3", 14.5)]
    assert len(parsed["dividends"]) == 21
    assert all(d["combination"] and d["dividend"] is not None for d in parsed["dividends"])

    (meta,) = parsed["_meta"]
    assert (meta["race_no"], meta["n_races"], meta["race_nos"]) == (2, 9, list(range(1, 10)))
    assert parsed["races"][0]["race_id"] == "hkjc:2020-12-09:HV:02"

    # the 2020 narrative: line breaks, typographic apostrophes, follow-up sentences
    report = _narrative(hkjc_dir, DEAD_HEAT, parsed)
    assert len(report) == 20
    assert "\n" in unattributed(parsed)[0]["text"]  # the page's <br /> breaks are kept
    assert sorted(i["horse_no"] for i in parsed["incidents"] if i["horse_id"]) == list(range(1, 13))
    names = {r["horse_id"]: r["horse_name"] for r in parsed["runners"]}
    mission = sentences(by_no(parsed, 9, "incidents")["text"])
    assert len(mission) == 2 and mission[0] in report[:3]  # one of the report's opening lines
    assert all("GOLDEN MISSION" in s for s in mission) and "GREAT HARVEST" in mission[1]
    shared = by_no(parsed, 5, "incidents")["text"]  # one sentence naming three horses
    assert len(sentences(shared)) == 1 and len(names_in(shared, names)) == 3
    assert all(shared in by_no(parsed, no, "incidents")["text"] for no in (1, 12))
    sake = by_no(parsed, 6, "incidents")  # the page prints a typographic apostrophe
    assert sake["horse_id"] == "HK_2019_D242"
    assert "FOR FUN\u2019S SAKE" in sentences(sake["text"])[0]
    assert sentences(sake["text"])[0].startswith("SUPER EIGHTEEN")
    # follow-up sentences naming no horse stay with the horse before them
    loose = [s for s in report if not names_in(s, names)]
    assert len(loose) == 2
    fortune = by_no(parsed, 4, "incidents")["text"]
    abraxas = by_no(parsed, 7, "incidents")["text"]
    assert loose[0] in norm(abraxas) and loose[1] in norm(fortune)
    # a dated additional report is appended to the horse it is about
    assert re.search(r"<\d\d/\d\d/\d{4} [A-Za-z ]+>KA FORTUNE,", fortune)
    assert mission[0] not in abraxas


def _awt_2024(hkjc_dir: Path, parsed: dict) -> None:
    (race,) = parsed["races"]
    assert (race["surface"], race["course_config"]) == ("awt", None)
    assert (race["race_no"], race["race_index"], race["distance_m"]) == (6, 603, 1200)
    assert (race["rating_hi"], race["rating_lo"], race["prize"]) == (60, 35, 1170000)
    assert race["race_name"] == "EMPOWERING YOUTH 1200M HANDICAP"

    withdrawn = by_no(parsed, 10)
    assert (withdrawn["placing_raw"], withdrawn["placing_code"], withdrawn["finish_pos"]) == (
        "WX-A", "WX-A", None)  # fmt: skip
    assert withdrawn["is_scratched"] is True and withdrawn["horse_id"] == "HK_2021_G083"
    assert (withdrawn["actual_wt_lb"], withdrawn["declared_horse_wt_lb"], withdrawn["draw"]) == (
        112, None, None)  # fmt: skip
    assert (withdrawn["win_odds"], withdrawn["finish_time_s"], withdrawn["lbw_lengths"]) == (
        None, None, None)  # fmt: skip
    assert (withdrawn["running_positions"], withdrawn["lbw_raw"]) == ([], "---")
    assert len(parsed["runners"]) == 12 and race["field_size"] == 11
    assert sum(r["is_scratched"] for r in parsed["runners"]) == 1
    assert by_no(parsed, 8)["lbw_lengths"] == 0.05  # NOSE

    assert pool(parsed, "5TH_DOUBLE") == [("1/11", 1867.5), ("1/8", 47.5)]
    assert pool(parsed, "3RD_DOUBLE_TRIO") == [("1,8,10/3,8,11", 517511.0)]
    assert pool(parsed, "TRIPLE_TRIO_CONSOLATION") == [("1,5,6/1,8,10/Any Combination", 5193.0)]
    assert pool(parsed, "QUARTET") == [("11,8,3,2", 124319.0)]

    assert len(parsed["incidents"]) == 12  # the per-horse table includes the withdrawn horse
    w = by_no(parsed, 10, "incidents")
    assert w["horse_id"] == "HK_2021_G083"
    body = body_of(hkjc_dir, AWT_2024)
    assert w["text"] and w["text"] == printed(body, "Incident", horse_id=w["horse_id"])
    assert w["text"] != by_no(parsed, 1, "incidents")["text"]
    (meta,) = parsed["_meta"]
    assert (meta["race_no"], meta["n_races"], meta["race_nos"]) == (6, 10, list(range(1, 11)))


def _layout_2004(hkjc_dir: Path, parsed: dict) -> None:
    assert len(parsed["runners"]) == 9
    winner = by_no(parsed, 3)  # no running-position column: columns mapped by header
    assert winner["running_positions"] == []
    assert (winner["finish_time_s"], winner["win_odds"], winner["draw"]) == (approx(70.5), 4.8, 2)
    assert (winner["horse_name"], winner["brand_raw"], winner["jockey_code"]) == (
        "LUCKY PORT", "CD086", None)  # fmt: skip
    assert all(
        r["running_positions"] == [] and r["win_odds"] is not None for r in parsed["runners"]
    )
    (race,) = parsed["races"]  # a Griffin race has no rating band
    assert (race["race_class"], race["rating_hi"], race["rating_lo"]) == (
        "Griffin Race",
        None,
        None,
    )
    assert (race["distance_m"], race["going"], race["course_config"]) == (1200, "GOOD TO FIRM", "C")
    assert race["race_index"] == 441
    _narrative(hkjc_dir, LAYOUT_2004, parsed)


def _simulcast(hkjc_dir: Path, parsed: dict) -> None:
    (meta,) = parsed["_meta"]  # the simulcast tab row is not counted
    assert (meta["n_races"], meta["race_nos"]) == (10, list(range(1, 11)))
    (race,) = parsed["races"]
    assert (race["race_id"], race["distance_m"], race["field_size"]) == (
        "hkjc:2026-09-13:ST:01", 1400, 14)  # fmt: skip
    assert race["split_times_s"] == approx([13.26, 34.45, 57.31, 81.26])
    assert by_no(parsed, 10)["running_positions"] == [2, 3, 2, 1]


def _odds_truncated(hkjc_dir: Path, parsed: dict) -> None:
    # odds '11' against a WIN dividend of 119.50: the odds stay as displayed
    assert by_no(parsed, 2)["win_odds"] == 11.0
    assert pool(parsed, "WIN") == [("2", 119.5)]


def _unit_bet(hkjc_dir: Path, parsed: dict) -> None:
    # 'QUARTET 12,1,6,10 349,150.00/$1.0': quoted per winning $1 unit, scaled to the $10 unit
    assert pool(parsed, "QUARTET") == [("12,1,6,10", approx(349150.0 * UNIT_STAKE / 1.0))]
    assert pool(parsed, "FIRST_4") == [("1,6,10,12", 105347.0)]
    assert all(d["dividend"] is not None and not d["is_refund"] for d in parsed["dividends"])
    assert len(parsed["dividends"]) == 15
    # double consolation rows are not a dead heat
    assert pool(parsed, "6TH_DOUBLE") == [("13/12", 5437.5), ("13/1", 312.5)]
    assert not any(r["dead_heat"] for r in parsed["runners"])
    assert [r["horse_no"] for r in parsed["runners"] if r["finish_pos"] == 1] == [12]
    assert pool(parsed, "WIN") == [("12", 241.0)]


def _protest(hkjc_dir: Path, parsed: dict) -> None:
    promoted, relegated = by_no(parsed, 4), by_no(parsed, 3)
    assert (promoted["finish_pos"], promoted["lbw_raw"], promoted["lbw_lengths"]) == (1, "-HD", 0.0)
    assert (relegated["finish_pos"], relegated["lbw_raw"], relegated["lbw_lengths"]) == (
        2,
        "---",
        0.0,
    )
    assert relegated["finish_time_s"] < promoted["finish_time_s"]
    (race,) = parsed["races"]
    assert race["status"] == "run"
    assert race["finish_time_s"] == approx(94.62)  # the race time is the first horse home
    assert pool(parsed, "WIN") == [("4", 29.0)]
    assert by_no(parsed, 9)["lbw_lengths"] == 0.5
    # '---' stays unknown where it means "no margin recorded": the withdrawn runner of 2024
    assert by_no(results(hkjc_dir, AWT_2024), 10)["lbw_lengths"] is None


def _abandoned(hkjc_dir: Path, parsed: dict) -> None:
    (race,) = parsed["races"]
    assert race["status"] == "abandoned"
    assert (race["race_id"], race["race_index"], race["going"]) == (
        "hkjc:2019-11-13:HV:01",
        None,
        None,
    )
    assert (race["finish_time_s"], race["split_times_s"], race["field_size"]) == (None, [], 12)
    assert (race["race_class"], race["distance_m"], race["course_config"]) == ("Class 5", 1200, "C")
    assert len(parsed["runners"]) == 12
    for r in parsed["runners"]:
        assert r["horse_id"] and r["draw"]
        assert (r["finish_pos"], r["placing_raw"], r["win_odds"], r["actual_wt_lb"]) == (
            None, None, None, None)  # fmt: skip
        assert (r["running_positions"], r["is_scratched"], r["dead_heat"]) == ([], False, False)
    assert parsed["incidents"] == []
    assert parsed["_meta"][0]["n_races"] == 9
    # every pool refunds
    assert len(parsed["dividends"]) == 11
    assert all(d["is_refund"] and d["dividend"] is None for d in parsed["dividends"])
    assert pool(parsed, "WIN") == [(None, None)]
    assert pool(parsed, "3_PICK_1_COMPOSITE_WIN") == [("A1", None), ("A2", None), ("A3", None)]
    assert {d["pool"] for d in parsed["dividends"]} >= {
        "PLACE",
        "QUINELLA",
        "QUINELLA_PLACE",
        "QUARTET",
    }


def _declared_abandoned(hkjc_dir: Path, parsed: dict) -> None:
    assert parsed["races"] == [{
        "jurisdiction": "hkjc", "race_id": "hkjc:2024-11-13:HV:07",
        "meeting_id": "hkjc:2024-11-13:HV", "race_no": 7, "race_index": None, "race_name": None,
        "race_class": None, "rating_hi": None, "rating_lo": None, "distance_m": None,
        "surface": None, "course_config": None, "going": None, "prize": None,
        "post_time_local": None, "field_size": 0, "finish_time_s": None, "split_times_s": [],
        "status": "abandoned",
    }]  # fmt: skip
    assert parsed["runners"] == [] and parsed["incidents"] == []
    (meta,) = parsed["_meta"]
    assert (meta["race_no"], meta["n_races"], meta["key_mismatch"]) == (7, 9, False)
    refunds = [d for d in parsed["dividends"] if d["is_refund"]]
    assert [d["pool"] for d in refunds] == [
        "WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE", "FORECAST", "TIERCE", "TRIO", "FIRST_4",
        "QUARTET",
    ]  # fmt: skip
    assert all(d["combination"] is None and d["dividend"] is None for d in refunds)
    # multi-race pools ending on the called-off leg still pay ('F' = any combination in that leg)
    assert pool(parsed, "6TH_DOUBLE") == [("7/F", 27.0)]
    assert pool(parsed, "3RD_DOUBLE_TRIO") == [("7,10,11/F", 2672.0)]
    assert pool(parsed, "TRIPLE_TRIO") == [("1,5,7/7,10,11/Any Combination", 256366.0)]
    assert {d["race_id"] for d in parsed["dividends"]} == {"hkjc:2024-11-13:HV:07"}
    # without the race tabs the race number comes from the key
    page = text_of(hkjc_dir, DECLARED_ABANDONED)
    start = page.index('<div class="f_clear top_races">')
    end = page.index('<div class="raceMeeting_select">')
    cut = parse_results.parse(raw("results", "2024-11-13_HV_R08", page[:start] + page[end:]))
    assert (cut["races"][0]["race_id"], cut["races"][0]["status"]) == (
        "hkjc:2024-11-13:HV:08", "abandoned")  # fmt: skip


def _void(hkjc_dir: Path, parsed: dict) -> None:
    (race,) = parsed["races"]
    assert (race["status"], race["race_index"], race["going"]) == ("abandoned", 186, None)
    assert (race["finish_time_s"], race["split_times_s"], race["field_size"]) == (None, [], 14)
    assert (race["race_class"], race["distance_m"], race["course_config"]) == (
        "Class 4",
        1200,
        "A+3",
    )
    starters = [r for r in parsed["runners"] if not r["is_scratched"]]
    assert len(starters) == 14
    assert all((r["placing_code"], r["finish_pos"], r["win_odds"]) == ("VOID", None, None)
               for r in starters)  # fmt: skip
    (withdrawn,) = [r for r in parsed["runners"] if r["is_scratched"]]
    assert (withdrawn["placing_code"], withdrawn["horse_no"], withdrawn["horse_id"]) == (
        "WV", None, "HK_2023_J547")  # fmt: skip
    single = [
        d for d in parsed["dividends"] if d["pool"] in {"WIN", "PLACE", "QUINELLA", "QUARTET"}
    ]
    assert len(single) == 4 and all(d["is_refund"] for d in single)
    assert len(parsed["incidents"]) == 14  # the stewards' per-horse table is still published


def _sentinel(hkjc_dir: Path, parsed: dict) -> None:
    assert parsed == {"races": [], "runners": [], "dividends": [], "incidents": [], "_meta": []}


RESULT_CHECKS = {
    LIVE_2026: _live_2026, ARCHIVE_2010: _archive_2010, DEAD_HEAT: _dead_heat, AWT_2024: _awt_2024,
    LAYOUT_2004: _layout_2004, SIMULCAST: _simulcast, ODDS_TRUNCATED: _odds_truncated,
    UNIT_BET: _unit_bet, PROTEST: _protest, ABANDONED: _abandoned,
    DECLARED_ABANDONED: _declared_abandoned, VOID: _void, NO_INFO: _sentinel,
    NO_INFO_LIVE: _sentinel,
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(RESULT_PAGES))
def test_real_results_pages(hkjc_dir, name):
    assert sorted(p.name for p in hkjc_dir.glob("localresults_*.html")) == sorted(RESULT_PAGES)
    key, status = RESULT_PAGES[name]
    body = body_of(hkjc_dir, name)
    # classified by content, never by HTTP status (the site serves all of these with 200)
    assert parse_results.classify(body) == status
    assert ADAPTER.classify("results", body) == status
    parsed = results(hkjc_dir, name)
    assert ADAPTER.parse(raw("results", key, body)) == parsed

    # contract: exactly the spec tables and columns, never more than one race
    assert set(parsed) == RESULT_TABLES
    columns_of = {"races": RACE_COLUMNS, "runners": RUNNER_COLUMNS,
                  "dividends": DIVIDEND_COLUMNS, "incidents": INCIDENT_COLUMNS}  # fmt: skip
    for table, columns in columns_of.items():
        assert all(set(row) == columns for row in parsed[table]), table
    assert len(parsed["races"]) == len(parsed["_meta"]) <= 1

    # reconciliation: one race, unique runners, the field is the starters, one race id throughout
    if status != STATUS_EMPTY and name != DECLARED_ABANDONED:
        (race,) = parsed["races"]
        runners = parsed["runners"]
        assert all(r["horse_id"] for r in runners)
        assert len({(r["race_id"], r["horse_id"]) for r in runners}) == len(runners)
        assert len({r["horse_no"] for r in runners}) == len(runners)
        assert race["field_size"] == sum(not r["is_scratched"] for r in runners)
        assert {r["race_id"] for r in runners} == {race["race_id"]}
        assert {d["race_id"] for d in parsed["dividends"]} == {race["race_id"]}
        assert race["race_id"] == "hkjc:{}:{}:{}".format(*key.replace("_R", "_").split("_"))

    if name in RUN_PAGES:
        # displayed win odds are the payout per unit TRUNCATED, never rounded; a dead heat splits
        # the pool after the odds were shown: 0.1 either way per extra winner
        winners = [r for r in parsed["runners"] if r["finish_pos"] == 1]
        win = dict(pool(parsed, "WIN"))
        assert winners and set(win) == {str(r["horse_no"]) for r in winners}
        assert parsed["races"][0]["finish_time_s"] == approx(winners[0]["finish_time_s"])
        for r in winners:
            implied = win[str(r["horse_no"])] / UNIT_STAKE * len(winners)
            step = 1.0 if r["win_odds"] >= ODDS_STEP_THRESHOLD else 0.1
            slack = 0.1 * (len(winners) - 1) + 1e-9
            assert -slack <= implied - r["win_odds"] < step + slack, r["horse_no"]
        for r in parsed["runners"]:  # running positions end at the finish position
            if r["running_positions"] and r["finish_pos"] and not r["dead_heat"]:
                assert r["running_positions"][-1] == r["finish_pos"], r["horse_no"]

    # every runner's jockey and trainer is what the page prints in that horse's row
    for r in parsed["runners"]:
        for role, header in (("jockey", "Jockey"), ("trainer", "Trainer")):
            if r[f"{role}_name"] is None:
                assert not re.search("[A-Za-z]", printed(body, header, horse_id=r["horse_id"]))
                assert (r[f"{role}_code"], r[f"{role}_key"]) == (None, None)
            else:
                check_person(r, role, body, header)
        assert r["jockey_name"] != r["trainer_name"] or r["jockey_name"] is None

    RESULT_CHECKS[name](hkjc_dir, parsed)


# ================================================================ race cards

CARD_SCRAPED = datetime(2026, 9, 21, 3, 48, 56, tzinfo=UTC)
CARD_URL = "https://racing.hkjc.com/en-us/local/information/racecard"
HV_R1 = ("racecard_2026-09-23_HV_R1.html", "2026-09-23_HV_R01")
HV_R7 = ("racecard_2026-09-23_HV_R7.html", "2026-09-23_HV_R07")
ST_WAYBACK = ("racecard_wayback_2026-05-24_ST_R8.html", "2026-05-24_ST_R08")
CARD_NO_INFO = ("racecard_2026-09-23_HV_R10_no_information.html", "2026-09-23_HV_R10")
CARD_NOT_AVAILABLE = ("racecard_not_available_2026-09-20.html", "2026-09-23_HV_R01")

CARD_RACE_COLUMNS = {
    "jurisdiction", "race_id", "meeting_id", "race_no", "race_name", "race_class", "rating_hi",
    "rating_lo", "distance_m", "surface", "course_config", "going", "prize", "post_time_local",
    "field_size", "status",
}  # fmt: skip
CARD_RUNNER_COLUMNS = {
    "race_id", "horse_id", "horse_no", "horse_name", "draw", "actual_wt_lb", "jockey_key",
    "jockey_code", "jockey_name", "trainer_key", "trainer_code", "trainer_name", "allowance_lb",
    "rating", "rating_change", "declared_horse_wt_lb", "horse_wt_change_lb", "best_time_s", "age",
    "sex", "season_stakes", "priority_raw", "days_since_last_run", "gear_raw", "sire", "dam",
    "import_type", "last6_raw", "is_reserve", "is_scratched", "card_scraped_utc",
    "jurisdiction", "brand_raw", "card_wt_lb", "overweight_lb", "reserve_no", "intl_rating",
    "wfa_lb", "owner", "trump_card", "priority_to_run", "trainer_preference",
}  # fmt: skip


def card(hkjc_dir: Path, fixture: tuple[str, str], body: bytes | str | None = None,
         key: str | None = None, fetched: datetime | None = CARD_SCRAPED) -> dict:  # fmt: skip
    name, default_key = fixture
    data = body_of(hkjc_dir, name) if body is None else body
    return parse_racecard.parse(raw("racecard", key or default_key, data, CARD_URL, fetched))


def test_real_racecards(hkjc_dir):
    # (the Wayback copy of the ST card is not what the fetcher sees: parsed, not classified)
    card_status = {HV_R1: STATUS_FETCHED, HV_R7: STATUS_FETCHED,
                   CARD_NO_INFO: STATUS_EMPTY, CARD_NOT_AVAILABLE: STATUS_EMPTY}  # fmt: skip
    for fixture, status in card_status.items():
        assert ADAPTER.classify("racecard", body_of(hkjc_dir, fixture[0])) == status, fixture[0]
    assert ADAPTER.classify("racecard", body_of(hkjc_dir, NO_INFO)) == STATUS_EMPTY

    r1 = card(hkjc_dir, HV_R1)
    assert set(r1) == {"races", "racecard_runners"}
    assert ADAPTER.parse(raw("racecard", HV_R1[1], body_of(hkjc_dir, HV_R1[0]), CARD_URL,
                             CARD_SCRAPED)) == r1  # fmt: skip
    for fixture in (HV_R1, HV_R7, ST_WAYBACK):
        parsed = card(hkjc_dir, fixture)
        (race,) = parsed["races"]
        assert set(race) == CARD_RACE_COLUMNS, fixture[0]
        assert all(set(r) == CARD_RUNNER_COLUMNS for r in parsed["racecard_runners"]), fixture[0]

    # ---- race headers
    assert r1["races"] == [{
        "jurisdiction": "hkjc", "race_id": "hkjc:2026-09-23:HV:01",
        "meeting_id": "hkjc:2026-09-23:HV", "race_no": 1, "race_name": "NAM FUNG HANDICAP",
        "race_class": "Class 5", "rating_hi": 40, "rating_lo": 0, "distance_m": 1650,
        "surface": "turf", "course_config": "C", "going": None,  # not declared until race day
        "prize": 875000, "post_time_local": "2026-09-23T18:40:00+08:00", "field_size": 12,
        "status": "declared",
    }]  # fmt: skip
    r7 = card(hkjc_dir, HV_R7)
    (race,) = r7["races"]
    assert (race["race_id"], race["race_name"], race["race_class"]) == (
        "hkjc:2026-09-23:HV:07", "MOUNT BUTLER HANDICAP", "Class 3")  # fmt: skip
    assert (race["rating_hi"], race["rating_lo"], race["distance_m"], race["prize"]) == (
        80, 60, 1200, 1860000)  # fmt: skip
    assert race["post_time_local"] == "2026-09-23T21:45:00+08:00"
    group = card(hkjc_dir, ST_WAYBACK)
    (race,) = group["races"]
    assert race["race_id"] == "hkjc:2026-05-24:ST:08"
    assert race["race_name"] == "THE STANDARD CHARTERED CHAMPIONS & CHATER CUP"
    assert (race["race_class"], race["rating_hi"], race["rating_lo"]) == ("Group One", None, None)
    assert (race["surface"], race["course_config"], race["distance_m"]) == ("turf", "A", 2400)
    assert (race["going"], race["prize"]) == ("GOOD", 13000000)
    assert race["post_time_local"] == "2026-05-24T16:05:00+08:00"
    assert race["field_size"] == 2  # the archived sample keeps two rows

    # derived: an all-weather course line, the page's labels over the key, the key filling gaps
    awt = card(hkjc_dir, HV_R1, body=body_of(hkjc_dir, HV_R1[0]).replace(
        b'Turf, "C" Course, 1650M', b"All Weather Track, 1650M, Wet Slow"))  # fmt: skip
    (race,) = awt["races"]
    assert (race["surface"], race["course_config"], race["distance_m"], race["going"]) == (
        "awt", None, 1650, "WET SLOW")  # fmt: skip
    assert card(hkjc_dir, HV_R1, key="2026-09-23_HV_R03")["races"][0]["race_id"] == (
        "hkjc:2026-09-23:HV:01")  # fmt: skip
    undated = card(hkjc_dir, HV_R1, body=body_of(hkjc_dir, HV_R1[0]).replace(
        b"Wednesday, September 23, 2026, Happy Valley, 18:40", b""))  # fmt: skip
    assert (undated["races"][0]["race_id"], undated["races"][0]["post_time_local"]) == (
        "hkjc:2026-09-23:HV:01", None)  # fmt: skip

    # ---- runners
    runners = {r["horse_no"]: r for r in r1["racecard_runners"] if not r["is_reserve"]}
    first = dict(runners[1])
    owner = first.pop(
        "owner"
    )  # a private person's name: checked against the page, not written here
    assert owner and owner in html.unescape(text_of(hkjc_dir, HV_R1[0]))
    for column in ("jockey_name", "jockey_key", "trainer_name", "trainer_key"):
        del first[column]  # people's names: checked against the page below
    assert first == {
        "jurisdiction": "hkjc", "race_id": "hkjc:2026-09-23:HV:01", "horse_id": "HK_2024_K306",
        "horse_no": 1, "horse_name": "SUPER SICARIO", "brand_raw": "K306", "draw": 12,
        "card_wt_lb": 135, "overweight_lb": None, "actual_wt_lb": 135,
        "jockey_code": "HCY", "trainer_code": "FC", "allowance_lb": 0, "intl_rating": None,
        "rating": 40, "rating_change": -1,
        "declared_horse_wt_lb": None, "horse_wt_change_lb": None,  # weights come on race day
        "best_time_s": 98.81, "age": 5, "wfa_lb": None, "sex": "g", "season_stakes": 0,
        "priority_raw": "+ 1", "trump_card": True, "priority_to_run": False,
        "trainer_preference": 1, "days_since_last_run": 17, "gear_raw": None, "sire": "Tivaci",
        "dam": "Promenade", "import_type": "PPG", "last6_raw": "10/11/7/1/7/10",
        "is_reserve": False, "reserve_no": None, "is_scratched": False,
        "card_scraped_utc": CARD_SCRAPED,
    }  # fmt: skip
    # every starter's jockey (with any claim) and trainer as the card prints them in its row
    for fixture, parsed in ((HV_R1, r1), (HV_R7, r7), (ST_WAYBACK, group)):
        page = body_of(hkjc_dir, fixture[0])
        for r in parsed["racecard_runners"]:
            if not r["is_reserve"]:
                check_person(r, "jockey", page, "Jockey")
                check_person(r, "trainer", page, "Trainer")
                assert r["jockey_key"] != r["trainer_key"].replace("trainer:", "jockey:")
    assert (runners[1]["jockey_code"], runners[1]["trainer_code"]) == ("HCY", "FC")
    # twelve starters plus two stand-bys, (race_id, horse_id) unique
    assert sorted(runners) == list(range(1, 13))
    assert sorted(r["draw"] for r in runners.values()) == list(range(1, 13))
    keys = [(r["race_id"], r["horse_id"]) for r in r1["racecard_runners"]]
    assert len(set(keys)) == len(keys) == 14
    assert all(r["horse_id"].startswith("HK_") for r in r1["racecard_runners"])
    # apprentice claims come off the weight, probable overweight goes on: Wt. - claim + Over Wt.
    for no, (code, wt, claim, over, carried) in {
        4: ("CJE", 131, 2, None, 129),
        8: (None, 125, 10, None, 115),
        11: (None, 119, 0, 1, 120),
    }.items():
        r = runners[no]
        assert (r["card_wt_lb"], r["allowance_lb"], r["overweight_lb"]) == (wt, claim, over), no
        assert r["actual_wt_lb"] == carried == wt - claim + (over or 0), no
        if code:
            assert r["jockey_code"] == code, no
    # the claim is printed after the apprentice's name and kept out of it
    assert printed(body_of(hkjc_dir, HV_R1[0]), "Jockey", horse_id=runners[8]["horse_id"]) == (
        f"{runners[8]['jockey_name']} (-10)")  # fmt: skip
    # gear, last six and stakes kept as printed
    assert runners[3]["gear_raw"] == "CP-/B1" and runners[7]["gear_raw"] == "V-/PC1"
    assert runners[7]["best_time_s"] is None
    assert runners[3]["last6_raw"] == "5/3/3/2/1/10"  # most recent run on the left
    assert runners[11]["season_stakes"] == 100625
    assert runners[2]["dam"] == "Panda's Turn"
    # stand-bys: reserve rows without number, draw or jockey, not scratched by the table's class
    first_sb, second_sb = [r for r in r1["racecard_runners"] if r["is_reserve"]]
    assert (first_sb["horse_id"], first_sb["horse_name"], first_sb["reserve_no"]) == (
        "HK_2025_L048", "WHOWROTETHATSONG", 1)  # fmt: skip
    no_seat = ("horse_no", "draw", "jockey_name", "jockey_key")
    assert [first_sb[c] for c in no_seat] == [None] * 4
    assert (first_sb["card_wt_lb"], first_sb["actual_wt_lb"], first_sb["allowance_lb"]) == (
        134,
        134,
        None,
    )
    assert (first_sb["rating"], first_sb["age"], first_sb["last6_raw"]) == (39, 5, "7/11/6/7/12/10")
    assert first_sb["trainer_code"] == "YCH"
    check_person(first_sb, "trainer", body_of(hkjc_dir, HV_R1[0]), None)  # by its profile link
    assert (first_sb["priority_raw"], first_sb["trump_card"], first_sb["gear_raw"]) == (
        "+ 1",
        True,
        "TT",
    )
    assert first_sb["is_scratched"] is False
    assert (second_sb["horse_id"], second_sb["reserve_no"], second_sb["gear_raw"]) == (
        "HK_2023_J135", 2, "B/BL1/TT")  # fmt: skip
    # priority flags on R7
    r7_by = {r["horse_no"]: r for r in r7["racecard_runners"] if not r["is_reserve"]}
    anode, akashvani = r7_by[10], r7_by[3]
    assert (anode["priority_raw"], anode["trump_card"], anode["priority_to_run"]) == (
        "+ * 2",
        True,
        True,
    )
    assert (anode["trainer_preference"], anode["allowance_lb"]) == (2, 3)
    assert (akashvani["priority_raw"], akashvani["trump_card"], akashvani["priority_to_run"]) == (
        "* 2", False, True)  # fmt: skip
    assert r7_by[5]["last6_raw"] == "10/11" and r7_by[8]["rating_change"] == 2
    # declared body weights on the archived card
    g_by = {r["horse_no"]: r for r in group["racecard_runners"]}
    warrior, numbers = g_by[1], g_by[5]
    assert (warrior["declared_horse_wt_lb"], warrior["horse_wt_change_lb"]) == (1169, 5)
    assert (numbers["declared_horse_wt_lb"], numbers["horse_wt_change_lb"]) == (1238, -5)
    assert (warrior["intl_rating"], warrior["rating"], warrior["rating_change"]) == (126, 139, None)
    assert (warrior["best_time_s"], warrior["season_stakes"]) == (146.92, 56756000)
    assert (warrior["priority_raw"], warrior["trainer_preference"], warrior["import_type"]) == (
        None, None, "ISG")  # fmt: skip

    # ---- derived edits of the real card
    doc = BeautifulSoup(body_of(hkjc_dir, HV_R1[0]), "lxml")
    for row in doc.select("table.starter tr"):  # columns are found by header text
        cells = row.find_all("td", recursive=False)
        if len(cells) == 27:
            cells[0].insert_before(cells[8].extract())
            cells[-1].insert_after(cells[1].extract())
    assert card(hkjc_dir, HV_R1, body=str(doc).encode()) == r1
    doc = BeautifulSoup(body_of(hkjc_dir, HV_R1[0]), "lxml")
    for cell in doc.select("table.starter tbody tr")[4].find_all("td", recursive=False):
        cell["class"] = ["TdScratch"]
    scratched = card(hkjc_dir, HV_R1, body=str(doc).encode())
    assert [r["horse_no"] for r in scratched["racecard_runners"] if r["is_scratched"]] == [5]
    assert scratched["races"][0]["field_size"] == 11
    fifth = f">{runners[5]['jockey_name']}</a>".encode()
    assert body_of(hkjc_dir, HV_R1[0]).count(fifth) == 1
    worded = card(hkjc_dir, HV_R1, body=body_of(hkjc_dir, HV_R1[0]).replace(
        fifth, fifth + b" (Scratched)"))  # fmt: skip
    assert [r["horse_no"] for r in worded["racecard_runners"] if r["is_scratched"]] == [5]
    unsafe = body_of(hkjc_dir, HV_R1[0]).replace(
        b"horse?horseid=HK_2024_K306", b"horse?horseid=../../etc/passwd").replace(
        b"jockeyprofile?jockeyid=HCY", b"jockeyprofile?jockeyid=H%20CY%3Cscript%3E")  # fmt: skip
    one = next(
        r for r in card(hkjc_dir, HV_R1, body=unsafe)["racecard_runners"] if r["horse_no"] == 1
    )
    assert (one["horse_id"], one["horse_name"], one["brand_raw"]) == (None, "SUPER SICARIO", "K306")
    assert (one["jockey_code"], one["jockey_name"], one["jockey_key"]) == (
        None, runners[1]["jockey_name"], runners[1]["jockey_key"])  # fmt: skip
    promoted = card(
        hkjc_dir,
        HV_R1,
        body=body_of(hkjc_dir, HV_R1[0]).replace(b"horseid=HK_2025_L048", b"horseid=HK_2024_K306"),
    )  # a stand-by listed twice
    twice = [r for r in promoted["racecard_runners"] if r["horse_id"] == "HK_2024_K306"]
    assert len(twice) == 1 and twice[0]["is_reserve"] is False and twice[0]["horse_no"] == 1
    assert len(promoted["racecard_runners"]) == 13 and promoted["races"][0]["field_size"] == 12

    # ---- pages without a card, and a page without a fetch time
    for fixture in (CARD_NO_INFO, CARD_NOT_AVAILABLE):
        assert card(hkjc_dir, fixture) == {"races": [], "racecard_runners": []}, fixture[0]
    unfetched = card(hkjc_dir, HV_R1, fetched=None)
    assert {r["card_scraped_utc"] for r in unfetched["racecard_runners"]} == {None}


# ================================================================ entries and comments on running

ENTRIES = "entries_2026-09-23_HV_all.html"
ENTRIES_KEY = "2026-09-23_HV"
ENTRIES_URL = (
    "https://racing.hkjc.com/en-us/local/information/entries?racedate=2026/09/23&Racecourse=HV"
    "&View=All"
)
ENTRIES_SCRAPED = datetime(2026, 9, 21, 2, 54, 0, tzinfo=UTC)
ENTRY_COLUMNS = {
    "jurisdiction", "meeting_id", "section_label", "prov_race_no", "section_no", "race_name",
    "race_class", "distance_m", "rating_hi", "rating_lo", "surface", "course_config", "horse_id",
    "horse_name", "brand_raw", "trainer_key", "trainer_code", "trainer_name", "card_wt_lb",
    "wfa_lb", "rating", "rating_change", "trainer_preference", "remarks_raw", "trump_card",
    "priority_to_run", "late_entry", "is_reserve", "reserve_no", "entries_scraped_utc",
}  # fmt: skip

CORUNNING_2010 = "corunning_archive_2010-01-01_ST_R1.html"
CORUNNING_URL = (
    "https://racing.hkjc.com/en-us/local/information/archive/corunning?date=20100101&raceno=1"
)
NOT_PUBLISHED = "No Comments on Running information for this horse."
CORUNNING_COLUMNS = {"jurisdiction", "race_id", "horse_id", "horse_no", "gear_raw", "comment"}


def entries(
    hkjc_dir: Path,
    body: bytes | None = None,
    key: str = ENTRIES_KEY,
    url: str = ENTRIES_URL,
    fetched: datetime | None = ENTRIES_SCRAPED,
) -> list[dict]:
    data = body_of(hkjc_dir, ENTRIES) if body is None else body
    parsed = parse_entries.parse(raw("entries", key, data, url, fetched))
    assert set(parsed) == {"entries"}
    return parsed["entries"]


def comments(
    body: bytes | str, key: str = "2010-01-01_ST_R01", url: str = CORUNNING_URL
) -> list[dict]:
    parsed = parse_corunning.parse(raw("corunning", key, body, url))
    assert set(parsed) == {"running_comments"}
    return parsed["running_comments"]


def blank_comments(html: str, keep: int = 0) -> str:
    """The real page with every comment cell after the first `keep` replaced by the per-horse
    not-yet-published sentinel, as a live page looks in the days after a meeting."""
    seen = 0

    def swap(match: re.Match[str]) -> str:
        nonlocal seen
        seen += 1
        return (
            match.group(0)
            if seen <= keep
            else f"{match.group(1)}\n {NOT_PUBLISHED}\n {match.group(3)}"
        )

    out, count = re.subn(r'(<td class="f_fs16 f_tal">)(.*?)(</td>)', swap, html, flags=re.S)
    assert count == 14
    return out


def test_real_entries_and_corunning(hkjc_dir):
    # ---------------- entries
    assert ADAPTER.classify("entries", body_of(hkjc_dir, ENTRIES)) == STATUS_FETCHED
    rows = entries(hkjc_dir)
    assert ADAPTER.parse(raw("entries", ENTRIES_KEY, body_of(hkjc_dir, ENTRIES), ENTRIES_URL,
                             ENTRIES_SCRAPED)) == {"entries": rows}  # fmt: skip
    head = {k: v for k, v in rows[0].items() if k not in {"trainer_name", "trainer_key"}}
    assert head == {
        "jurisdiction": "hkjc", "meeting_id": "hkjc:2026-09-23:HV",
        "section_label": "Class 3 1200m Section 1 MOUNT BUTLER HANDICAP (Ratings 80-60)",
        "prov_race_no": "P10", "section_no": 1, "race_name": "MOUNT BUTLER HANDICAP",
        "race_class": "Class 3", "distance_m": 1200, "rating_hi": 80, "rating_lo": 60,
        "surface": "turf", "course_config": "C", "horse_id": "HK_2020_E061",
        "horse_name": "HARMONY N BLESSED", "brand_raw": "E061",
        "trainer_code": "HDA", "card_wt_lb": 135, "wfa_lb": None,
        "rating": 76, "rating_change": 0, "trainer_preference": 1, "remarks_raw": "+",
        "trump_card": True, "priority_to_run": False, "late_entry": False, "is_reserve": False,
        "reserve_no": None, "entries_scraped_utc": ENTRIES_SCRAPED,
    }  # fmt: skip
    assert all(set(r) == ENTRY_COLUMNS for r in rows)
    for r in rows:  # each trainer as the page prints it in the horse's row and profile link
        check_person(r, "trainer", body_of(hkjc_dir, ENTRIES), "Trainer")
    assert len({r["trainer_key"] for r in rows}) > 10
    # the whole meeting: 108 entries in nine sections of twelve, plus 42 reserves
    assert len(rows) == 150 and sum(r["is_reserve"] for r in rows) == 42
    assert len({r["horse_id"] for r in rows}) == 150
    assert all(
        r["horse_id"].startswith("HK_") and r["meeting_id"] == "hkjc:2026-09-23:HV" for r in rows
    )
    sections = list(dict.fromkeys((r["prov_race_no"], r["section_no"]) for r in rows))
    assert sections == [("P10", 1), ("P10", 2), ("P20", 1), ("P30", 1), ("P40", 1), ("P40", 2),
                        ("P50", 1), ("P50", 2), ("P60", 1)]  # fmt: skip
    per_section = Counter((r["prov_race_no"], r["section_no"], r["is_reserve"]) for r in rows)
    assert all(per_section[(prov, sec, False)] == 12 for prov, sec in sections)
    assert (per_section[("P60", 1, True)], per_section[("P20", 1, True)]) == (11, 4)
    tai_hang = next(r for r in rows if r["prov_race_no"] == "P20")
    assert (tai_hang["race_name"], tai_hang["race_class"], tai_hang["distance_m"]) == (
        "TAI HANG HANDICAP", "Class 3", 1800)  # fmt: skip
    assert (tai_hang["rating_hi"], tai_hang["rating_lo"]) == (85, 60)
    nam_fung = next(r for r in rows if r["prov_race_no"] == "P60")
    assert (nam_fung["race_class"], nam_fung["rating_hi"], nam_fung["rating_lo"]) == (
        "Class 5",
        40,
        0,
    )
    assert nam_fung["section_label"] == "Class 5 1650m Section 1 NAM FUNG HANDICAP (Ratings 40-0)"
    named = {r["horse_name"]: r for r in rows}
    ace = named["ACE CHAMPION"]  # a reserve keeps its stand-by number
    assert (ace["is_reserve"], ace["reserve_no"], ace["horse_id"], ace["brand_raw"]) == (
        True, 1, "HK_2024_K379", "K379")  # fmt: skip
    assert (ace["card_wt_lb"], ace["rating"], ace["rating_change"], ace["trainer_preference"]) == (
        127, 68, 7, 1)  # fmt: skip
    assert (ace["remarks_raw"], ace["trump_card"], ace["priority_to_run"]) == (None, False, False)
    assert (ace["prov_race_no"], ace["section_no"], ace["trainer_code"]) == ("P10", 1, "NPC")
    anode, sky_cap = named["ANODE"], named["SKY CAP"]
    assert (anode["remarks_raw"], anode["trump_card"], anode["priority_to_run"]) == (
        "+ *",
        True,
        True,
    )
    assert anode["trainer_preference"] == 2
    assert (sky_cap["remarks_raw"], sky_cap["trump_card"], sky_cap["priority_to_run"]) == (
        "*", False, True)  # fmt: skip
    assert named["FOREVER FOLKS"]["trainer_preference"] == 3
    assert named["DANICA'S CHOICE"]["horse_id"] == "HK_2025_L096"
    assert sum(r["rating_change"] is None for r in rows) == 2  # printed '-'
    # the R7 card's starters were all entered in one section, the card's own race
    r7 = card(hkjc_dir, HV_R7)
    by_id = {r["horse_id"]: r for r in rows}
    starters = [r["horse_id"] for r in r7["racecard_runners"] if not r["is_reserve"]]
    assert starters and all(h in by_id for h in starters)
    (section,) = {(by_id[h]["prov_race_no"], by_id[h]["section_no"]) for h in starters}
    entered = by_id[starters[0]]
    (r7_race,) = r7["races"]
    assert (entered["race_name"], entered["race_class"], entered["distance_m"]) == (
        r7_race["race_name"], r7_race["race_class"], r7_race["distance_m"])  # fmt: skip
    assert (entered["rating_hi"], entered["rating_lo"]) == (
        r7_race["rating_hi"],
        r7_race["rating_lo"],
    )
    # and by the same trainers: the two pages name each starter's trainer alike
    for r in r7["racecard_runners"]:
        if not r["is_reserve"]:
            assert by_id[r["horse_id"]]["trainer_key"] == r["trainer_key"], r["horse_id"]
            assert by_id[r["horse_id"]]["trainer_code"] == r["trainer_code"], r["horse_id"]

    # derived edits
    entries_body = body_of(hkjc_dir, ENTRIES)
    late = entries(
        hkjc_dir, entries_body.replace(b"<td>2</td>\n<td>*</td>", b"<td>2</td>\n<td># *</td>", 1)
    )
    sky = next(r for r in late if r["horse_name"] == "SKY CAP")
    assert (sky["remarks_raw"], sky["late_entry"], sky["priority_to_run"]) == ("# *", True, True)
    unlabelled = entries_body.replace(b'<span class="f_fs13">23/09/2026</span> Happy Valley', b"")
    filled = entries(hkjc_dir, unlabelled, key="2026-09-23_HV")
    assert len(filled) == 150 and {r["meeting_id"] for r in filled} == {"hkjc:2026-09-23:HV"}
    assert entries(hkjc_dir, unlabelled, key="garbage") == []
    assert {r["meeting_id"] for r in entries(hkjc_dir, key="2026-09-27_ST")} == {
        "hkjc:2026-09-23:HV"
    }
    text = text_of(hkjc_dir, ENTRIES)
    no_nav = entries(hkjc_dir, text[: text.index("Other Section(s)/Division(s)")].encode())
    assert len(no_nav) == 150 and {r["prov_race_no"] for r in no_nav} == {None}
    assert no_nav[0]["section_label"] == rows[0]["section_label"]
    unsafe = entries_body.replace(b"horse?horseid=HK_2020_E061", b"horse?horseid=HK_2020_E061/../x")
    unsafe = unsafe.replace(
        b'trainerprofile?trainerid=HDA"', b'trainerprofile?trainerid=H%20DA"', 1
    )
    first = entries(hkjc_dir, unsafe)[0]
    assert (first["horse_id"], first["horse_name"], first["brand_raw"]) == (
        None,
        "HARMONY N BLESSED",
        "E061",
    )
    assert (first["trainer_code"], first["trainer_name"]) == (None, rows[0]["trainer_name"])
    assert entries(hkjc_dir, body_of(hkjc_dir, CARD_NO_INFO[0])) == []
    assert {r["entries_scraped_utc"] for r in entries(hkjc_dir, fetched=None)} == {None}

    # ---------------- comments on running
    co_text = text_of(hkjc_dir, CORUNNING_2010)
    assert ADAPTER.classify("corunning", body_of(hkjc_dir, CORUNNING_2010)) == STATUS_FETCHED
    co = comments(co_text)
    assert ADAPTER.parse(raw("corunning", "2010-01-01_ST_R01", co_text, CORUNNING_URL)) == {
        "running_comments": co}  # fmt: skip
    assert len(co) == 14 and sorted(r["horse_no"] for r in co) == list(range(1, 15))
    assert len({r["horse_id"] for r in co}) == 14
    assert all(set(r) == CORUNNING_COLUMNS for r in co)
    assert all(r["jurisdiction"] == "hkjc" and r["race_id"] == "hkjc:2010-01-01:ST:01" for r in co)
    co_by = {r["horse_no"]: r for r in co}
    assert {k: v for k, v in co_by[5].items() if k != "comment"} == {
        "jurisdiction": "hkjc", "race_id": "hkjc:2010-01-01:ST:01",
        "horse_id": "HK_2008_K277",  # from the href; the page prints the brand "CK277"
        "horse_no": 5, "gear_raw": "H",
    }  # fmt: skip
    co_body = body_of(hkjc_dir, CORUNNING_2010)
    for r in co:  # each comment is the horse's own cell of the Comment column
        assert r["comment"] == printed(co_body, "Comment", horse_id=r["horse_id"]), r["horse_no"]
    assert len({r["comment"] for r in co}) == 14
    assert {no: r["gear_raw"] for no, r in co_by.items()} == {
        5: "H",
        7: "B1",
        3: None,
        14: "CP/TT",
        13: "TT",
        11: "TT1",
        4: "CP/TT1",
        2: "B",
        1: None,
        12: None,
        10: "B/H1",
        8: None,
        9: "B/TT",
        6: "TT1",
    }  # fmt: skip  ("--" is the page's no-gear placeholder)
    assert (co_by[7]["horse_id"], co_by[3]["horse_id"], co_by[6]["horse_id"]) == (
        "HK_2009_L026", "HK_2007_J212", "HK_2009_L066")  # fmt: skip
    for r in co:
        assert r["comment"] == " ".join(r["comment"].split()) and len(r["comment"]) > 40
        assert isinstance(r["horse_no"], int)
    # the same race's results page: same runners, same horse ids by number
    same_race = {r["horse_no"]: r["horse_id"] for r in results(hkjc_dir, ARCHIVE_2010)["runners"]}
    assert {no: r["horse_id"] for no, r in co_by.items()} == same_race
    # and the same riders: the results' jockey of each horse is the one the comments page prints
    for r in results(hkjc_dir, ARCHIVE_2010)["runners"]:
        assert r["jockey_name"] == printed(co_body, "Jockey", horse_id=r["horse_id"]), r["horse_no"]

    # the race: key, then URL, then the page's own Results link
    assert {r["race_id"] for r in comments(co_text, key="x")} == {"hkjc:2010-01-01:ST:01"}
    live = "https://racing.hkjc.com/en-us/local/information/corunning?Date=20100101&RaceNo=1"
    assert {r["race_id"] for r in comments(co_text, key="x", url=live)} == {"hkjc:2010-01-01:ST:01"}
    from_content = comments(co_text, key="", url="https://example.invalid/")
    assert len(from_content) == 14 and {r["race_id"] for r in from_content} == {
        "hkjc:2010-01-01:ST:01"
    }
    # a page for another race is never filed under the requested race
    for key, url in (
        ("2026-09-13_ST_R01", "https://racing.hkjc.com/en-us/local/information/corunning?Date=20260913&RaceNo=1"),
        ("2010-01-01_HV_R01", CORUNNING_URL),
        ("2010-01-01_ST_R03",
         "https://racing.hkjc.com/en-us/local/information/archive/corunning?date=20100101&raceno=3"),
    ):  # fmt: skip
        assert comments(co_text, key=key, url=url) == [], key
    # site navigation outside the content neither relabels nor vetoes the page
    nav = ('<div class="nav"><a href="/en-us/local/information/localresults?racedate=2026/09/20'
           '&Racecourse=HV&RaceNo=9">Results</a></div>')  # fmt: skip
    with_nav = co_text.replace("<body>", "<body>" + nav, 1)
    assert nav in with_nav
    for key in ("2010-01-01_ST_R01", ""):
        navved = comments(with_nav, key=key)
        assert len(navved) == 14 and {r["race_id"] for r in navved} == {"hkjc:2010-01-01:ST:01"}
    # comments not yet published: no rows; one runner without a comment keeps its gear
    blank = blank_comments(co_text)
    assert comments(blank, key="2026-09-16_HV_R01") == []
    assert ADAPTER.classify("corunning", blank.encode()) == STATUS_EMPTY
    partial = comments(blank_comments(co_text, keep=13))
    assert len(partial) == 14
    assert [(r["horse_no"], r["gear_raw"]) for r in partial if r["comment"] is None] == [(6, "TT1")]
    assert all(NOT_PUBLISHED not in (r["comment"] or "") for r in partial)
    # pages without a comment table
    for other in (NO_INFO, SECTIONAL_2026, ACTIVE, RETIRED):
        assert comments(body_of(hkjc_dir, other)) == [], other


# ================================================================ horse profiles

ACTIVE = "horse_HK_2023_J087_option1_all_seasons.html"
RETIRED = "horse_otherhorse_HK_2008_K277_retired.html"
ACTIVE_URL = "https://racing.hkjc.com/en-us/local/information/horse?horseid=HK_2023_J087&Option=1"
RETIRED_URL = "https://racing.hkjc.com/en-us/local/information/otherhorse?horseid=HK_2008_K277"
HORSE_SCRAPED = datetime(2026, 9, 20, 15, 53, 7, tzinfo=UTC)
HORSE_COLUMNS = {
    "jurisdiction", "horse_id", "name", "origin", "colour", "sex", "import_type", "sire", "dam",
    "dam_sire", "is_retired", "profile_scraped_utc",
}  # fmt: skip
RUN_COLUMNS = {
    "jurisdiction", "horse_id", "race_date", "race_index", "venue", "surface", "course_config",
    "distance_m", "going", "race_class", "draw", "rating", "trainer_name", "jockey_name",
    "lbw_raw", "win_odds", "actual_wt_lb", "running_positions", "finish_time_s",
    "declared_horse_wt_lb", "gear_raw", "placing_raw", "race_id",
}  # fmt: skip
# as-of-scrape values on the profile block: any of them as a feature is look-ahead leakage
FORBIDDEN = {
    "rating", "current_rating", "last_rating", "start_of_season_rating", "season_stakes",
    "total_stakes", "trainer", "trainer_name", "current_trainer", "owner", "age", "record",
    "starts", "location", "import_date",
}  # fmt: skip


def horse(hkjc_dir: Path, name: str, key: str, url: str,
          fetched: datetime | None = HORSE_SCRAPED, body: bytes | None = None) -> dict:  # fmt: skip
    data = body_of(hkjc_dir, name) if body is None else body
    return parse_horse.parse(raw("horse", key, data, url, fetched))


def run_on(parsed: dict, when: date) -> dict:
    (row,) = [r for r in parsed["horse_runs"] if r["race_date"] == when]
    return row


def test_real_horse_profiles(hkjc_dir):
    for name in (ACTIVE, RETIRED):
        assert ADAPTER.classify("horse", body_of(hkjc_dir, name)) == STATUS_FETCHED, name
    assert ADAPTER.classify("horse", body_of(hkjc_dir, NO_INFO)) == STATUS_EMPTY
    active = horse(hkjc_dir, ACTIVE, "HK_2023_J087", ACTIVE_URL)
    retired = horse(hkjc_dir, RETIRED, "HK_2008_K277", RETIRED_URL)
    assert ADAPTER.parse(raw("horse", "HK_2023_J087", body_of(hkjc_dir, ACTIVE), ACTIVE_URL,
                             HORSE_SCRAPED)) == active  # fmt: skip

    # ---- horses: static attributes only
    assert active["horses"] == [{
        "jurisdiction": "hkjc", "horse_id": "HK_2023_J087", "name": "SPICY SPANGLE",
        "origin": "AUS", "colour": "Brown", "sex": "Gelding", "import_type": "PPG",
        "sire": "Headwater", "dam": "Beechal", "dam_sire": "Lonhro", "is_retired": False,
        "profile_scraped_utc": HORSE_SCRAPED,
    }]  # fmt: skip
    assert retired["horses"] == [{
        "jurisdiction": "hkjc", "horse_id": "HK_2008_K277",
        "name": "BRAVE HEART",  # the title prints "BRAVE HEART (CK277) (Retired)"
        "origin": "AUS",  # the retired layout prints "Country of Origin" with no age
        "colour": "Bay", "sex": "Gelding", "import_type": "PPG", "sire": "Elvstroem",
        "dam": "La Camille", "dam_sire": "Peintre Celebre", "is_retired": True,
        "profile_scraped_utc": HORSE_SCRAPED,
    }]  # fmt: skip
    # the current trainer, as the profile block prints it
    current = re.search(r"Trainer : (.+?) Owner :", page_text(body_of(hkjc_dir, ACTIVE)))
    assert current and current.group(1)
    for parsed in (active, retired):
        (row,) = parsed["horses"]
        assert set(row) == HORSE_COLUMNS and not set(row) & FORBIDDEN
        # nor smuggled under another name: the printed ratings, the current trainer, the stakes
        values = {str(v) for v in row.values()}
        assert not values & {"38", "31", "64", "6", current.group(1), "$490,000", "$2,986,800",
                             "$1,264,500"}  # fmt: skip
    assert horse(hkjc_dir, RETIRED, "HK_2008_K277", RETIRED_URL, fetched=None)["horses"][0][
        "profile_scraped_utc"] is None  # fmt: skip
    # the retired layout is read off the page, not only the URL
    requested = ACTIVE_URL.replace("2023_J087", "2008_K277")
    assert horse(hkjc_dir, RETIRED, "HK_2008_K277", requested)["horses"][0]["is_retired"] is True
    # the id: key, then URL, then the page's links; nothing outside a plain charset
    from_url = horse(hkjc_dir, ACTIVE, "", ACTIVE_URL)
    assert from_url["horses"][0]["horse_id"] == "HK_2023_J087"
    assert {r["horse_id"] for r in from_url["horse_runs"]} == {"HK_2023_J087"}
    from_links = horse(hkjc_dir, ACTIVE, "not a horse id", "https://example.invalid/")
    assert (
        from_links["horses"][0]["horse_id"] == "HK_2023_J087"
        and len(from_links["horse_runs"]) == 33
    )

    # ---- horse_runs: the active page (DD/MM/YY)
    runs = active["horse_runs"]
    # 37 body rows = 4 season separators + 33 form rows (32 starts + the 05/07/25 withdrawal)
    assert len(runs) == 33
    assert all(set(r) == RUN_COLUMNS for r in runs)
    assert all(r["jurisdiction"] == "hkjc" and r["horse_id"] == "HK_2023_J087" for r in runs)
    dates = [r["race_date"] for r in runs]
    assert len(set(dates)) == 33 and dates == sorted(dates, reverse=True)
    assert min(dates) == date(2024, 3, 20) and max(dates) == date(2026, 9, 16)
    assert date(2025, 1, 1) in dates  # printed "01/01/25"
    assert sum(r["placing_raw"].startswith("WV") for r in runs) == 1
    assert run_on(active, date(2026, 9, 16)) == {
        "jurisdiction": "hkjc", "horse_id": "HK_2023_J087", "race_date": date(2026, 9, 16),
        "race_index": 29, "venue": "HV", "surface": "turf", "course_config": "B",
        "distance_m": 1000, "going": "G", "race_class": "5", "draw": 3,
        "rating": 31,  # going INTO the race (the profile's current rating is 38)
        "trainer_name": current.group(1),  # the horse's current trainer
        "jockey_name": by_no(results(hkjc_dir, LIVE_2026), 11)["jockey_name"],  # that race's
        "lbw_raw": "1-3/4",  # a winner's row prints the winning margin
        "win_odds": 4.5, "actual_wt_lb": 123, "running_positions": [1, 1, 1],
        "finish_time_s": approx(57.54), "declared_horse_wt_lb": 1111, "gear_raw": "H/TT",
        "placing_raw": "1", "race_id": "hkjc:2026-09-16:HV:01",
    }  # fmt: skip
    awt = run_on(active, date(2026, 5, 6))
    assert (awt["venue"], awt["surface"], awt["course_config"]) == ("ST", "awt", None)
    assert (awt["race_index"], awt["distance_m"], awt["going"], awt["race_class"]) == (
        666,
        1200,
        "GD",
        "5",
    )
    assert (awt["draw"], awt["rating"], awt["gear_raw"], awt["placing_raw"]) == (4, 35, "H/TT", "6")
    assert (awt["lbw_raw"], awt["win_odds"], awt["actual_wt_lb"]) == ("4-1/4", 9.1, 130)
    assert (awt["running_positions"], awt["finish_time_s"]) == ([1, 1, 6], approx(71.57))
    assert (awt["declared_horse_wt_lb"], awt["race_id"]) == (1114, "hkjc:2026-05-06:ST:02")
    changed = run_on(active, date(2024, 9, 22))  # gear changes under the previous trainer
    assert changed["gear_raw"] == "H-/PC-/XB-/TT1"
    assert (changed["race_index"], changed["draw"], changed["rating"], changed["distance_m"]) == (
        41,
        8,
        44,
        1000,
    )
    assert (changed["venue"], changed["surface"], changed["course_config"]) == ("ST", "turf", "B+2")
    assert changed["placing_raw"] == "13"
    assert changed["trainer_name"] != current.group(1)  # the previous trainer
    card_r1 = {r["horse_no"]: r for r in card(hkjc_dir, HV_R1)["racecard_runners"]}
    assert changed["jockey_name"] == card_r1[1]["jockey_name"]  # who rides no. 1 on the card
    assert (changed["lbw_raw"], changed["win_odds"], changed["running_positions"]) == (
        "14-1/4",
        60.0,
        [5, 5, 13],
    )
    assert changed["finish_time_s"] == approx(59.23)
    replaced = run_on(active, date(2026, 1, 28))
    assert (replaced["gear_raw"], replaced["rating"], replaced["draw"], replaced["distance_m"],
            replaced["course_config"]) == ("H2/TT", 40, 3, 1000, "C")  # fmt: skip
    debut = run_on(active, date(2024, 3, 20))  # "--" = no gear worn
    assert (debut["gear_raw"], debut["rating"], debut["draw"], debut["distance_m"]) == (
        None,
        52,
        4,
        1000,
    )
    assert (debut["course_config"], debut["race_index"]) == ("C+3", 514)
    assert debut["jockey_name"] == card_r1[11]["jockey_name"]  # who rides no. 11 on the card
    wv = run_on(active, date(2025, 7, 5))  # a withdrawal keeps its rating and nulls the result
    assert wv["placing_raw"] == "WV-A"
    assert (wv["rating"], wv["actual_wt_lb"], wv["race_class"]) == (49, 122, "4")
    assert wv["jockey_name"] == card_r1[4]["jockey_name"]  # who rides no. 4 on the card
    assert (wv["draw"], wv["lbw_raw"], wv["win_odds"], wv["finish_time_s"]) == (
        None,
        None,
        None,
        None,
    )
    assert (wv["declared_horse_wt_lb"], wv["gear_raw"], wv["running_positions"]) == (None, None, [])
    assert wv["race_id"] == "hkjc:2025-07-05:ST:05"
    assert [r["rating"] for r in runs[:5]] == [31, 33, 33, 35, 35]

    # ---- every form row's trainer and jockey as the form table prints them on that date
    for page, parsed, shown in ((ACTIVE, active, "%d/%m/%y"), (RETIRED, retired, "%d/%m/%Y")):
        for r in parsed["horse_runs"]:
            on = r["race_date"].strftime(shown)
            for column, header in (("trainer_name", "Trainer"), ("jockey_name", "Jockey")):
                cell = printed(body_of(hkjc_dir, page), header, on=on)
                if r[column] is None:
                    assert not re.search("[A-Za-z]", cell), (page, on, header)
                else:
                    assert r[column] == cell and re.search("[A-Za-z]", cell), (page, on, header)
            assert r["jockey_name"] != r["trainer_name"], (page, on)
    assert len({r["jockey_name"] for r in runs}) > 3 and len({r["trainer_name"] for r in runs}) == 2

    # ---- horse_runs: the retired page (DD/MM/YYYY)
    old = retired["horse_runs"]
    # 18 body rows = 4 season separators + 14 form rows (12 starts + 2 withdrawals)
    assert len(old) == 14 and all(set(r) == RUN_COLUMNS for r in old)
    assert {r["horse_id"] for r in old} == {"HK_2008_K277"}
    assert sum(r["placing_raw"] == "WV" for r in old) == 2
    assert (min(r["race_date"] for r in old), max(r["race_date"] for r in old)) == (
        date(2009, 6, 13), date(2012, 2, 19))  # fmt: skip
    assert run_on(retired, date(2012, 2, 19)) == {
        "jurisdiction": "hkjc", "horse_id": "HK_2008_K277", "race_date": date(2012, 2, 19),
        "race_index": 411, "venue": "ST", "surface": "turf", "course_config": "A",
        "distance_m": 1400, "going": "G", "race_class": "4", "draw": 12, "rating": 64,
        "trainer_name": by_no(results(hkjc_dir, ARCHIVE_2010), 5)["trainer_name"],  # as in 2010
        "jockey_name": run_on(active, date(2026, 9, 16))["jockey_name"],  # SPICY SPANGLE's 2026
        "lbw_raw": "1-1/2",
        "win_odds": 35.0, "actual_wt_lb": 130, "running_positions": [11, 11, 11, 4],
        "finish_time_s": approx(83.18), "declared_horse_wt_lb": 1079, "gear_raw": "B/H",
        "placing_raw": "4", "race_id": "hkjc:2012-02-19:ST:05",
    }  # fmt: skip
    valley = run_on(retired, date(2011, 4, 20))
    assert (valley["venue"], valley["course_config"]) == ("HV", "C")
    assert (valley["distance_m"], valley["race_class"]) == (1650, "3S")
    assert (valley["draw"], valley["rating"], valley["gear_raw"], valley["placing_raw"]) == (
        8,
        68,
        "B/H",
        "12",
    )
    assert valley["running_positions"] == [8, 9, 11, 12]
    assert valley["jockey_name"] == debut["jockey_name"]  # SPICY SPANGLE's debut rider
    assert valley["finish_time_s"] == approx(102.99)
    blinkers = run_on(retired, date(2010, 11, 6))
    assert (blinkers["gear_raw"], blinkers["rating"], blinkers["draw"]) == ("B1/H", 72, 6)
    assert (blinkers["distance_m"], blinkers["going"], blinkers["course_config"]) == (
        1200,
        "GY",
        "C+3",
    )
    assert blinkers["race_index"] == 155
    first_run = run_on(retired, date(2009, 11, 29))
    assert (first_run["gear_raw"], first_run["rating"]) == ("H1", 52)
    assert (first_run["draw"], first_run["lbw_raw"]) == (7, "N")
    griffin = run_on(retired, date(2009, 6, 13))  # a Griffin withdrawal: no rating, no jockey
    assert (griffin["placing_raw"], griffin["race_class"]) == ("WV", "GRIFFIN")
    assert (griffin["rating"], griffin["jockey_name"]) == (None, None)
    assert (griffin["draw"], griffin["actual_wt_lb"]) == (None, 118)
    assert griffin["trainer_name"] == run_on(retired, date(2012, 2, 19))["trainer_name"]
    assert griffin["running_positions"] == []
    assert [r["rating"] for r in old][:4] == [64, 68, 68, 68]

    # ---- the form row and the results page of the same race agree (joined on horse_id, date)
    for page_name, when, horse_no, parsed in (
        (LIVE_2026, date(2026, 9, 16), 11, active),
        (ARCHIVE_2010, date(2010, 1, 1), 5, retired),
    ):  # fmt: skip
        res = results(hkjc_dir, page_name)
        runner, run = by_no(res, horse_no), run_on(parsed, when)
        assert run["race_id"] == res["races"][0]["race_id"]
        assert run["race_index"] == res["races"][0]["race_index"]
        assert run["distance_m"] == res["races"][0]["distance_m"]
        assert run["horse_id"] == runner["horse_id"]
        assert (run["draw"], run["actual_wt_lb"], run["win_odds"]) == (
            runner["draw"], runner["actual_wt_lb"], runner["win_odds"])  # fmt: skip
        assert run["placing_raw"] == runner["placing_raw"]
        assert (run["jockey_name"], run["trainer_name"]) == (
            runner["jockey_name"], runner["trainer_name"])  # fmt: skip
    won = run_on(retired, date(2010, 1, 1))
    assert (won["race_index"], won["draw"], won["rating"], won["gear_raw"], won["distance_m"]) == (
        275, 9, 54, "H", 1200)  # fmt: skip
    assert (won["placing_raw"], won["lbw_raw"], won["win_odds"]) == ("1", "1-3/4", 2.0)
    # the comments on running of that race print the same gear for the winner (no. 5)
    gear_2010 = {r["horse_no"]: r["gear_raw"] for r in comments(text_of(hkjc_dir, CORUNNING_2010))}
    assert won["gear_raw"] == gear_2010[5]

    # ---- pages without a profile
    for other in (SECTIONAL_2026, CORUNNING_2010, NO_INFO):
        assert horse(hkjc_dir, other, "HK_2023_J087", ACTIVE_URL) == {
            "horses": [],
            "horse_runs": [],
        }, other


# ================================================================ sectional times and fixtures

SECTIONAL_2026 = "sectional_2026-09-16_HV_R1.html"
SECTIONAL_KEY = "2026-09-16_HV_R01"
SECTIONAL_URL = (
    "https://racing.hkjc.com/en-us/local/information/displaysectionaltime?racedate=16/09/2026"
    "&RaceNo=1"
)
SECTIONAL_COLUMNS = {
    "jurisdiction", "race_id", "horse_id", "horse_no", "section_no", "section_pos",
    "section_margin_lengths", "section_time_s", "section_margin_raw", "sub_split_times_s",
}  # fmt: skip
MEETING_COLUMNS = {
    "jurisdiction", "meeting_id", "race_date", "venue", "season", "day_night", "surface_mix",
    "status",
}  # fmt: skip
FIXTURE_MONTHS = {"fixture_2026-09.html": "2026-09", "fixture_2010-01.html": "2010-01",
                  "fixture_2019-11.html": "2019-11"}  # fmt: skip
FIXTURE_CELL = "racecard_fixture_cell_20260923.html"


def sectionals(body: bytes | str, key: str = SECTIONAL_KEY, url: str = SECTIONAL_URL) -> list[dict]:
    parsed = parse_sectional.parse(raw("sectional", key, body, url))
    assert set(parsed) == {"sectionals"}
    return parsed["sectionals"]


def meetings(body: bytes | str, key: str, url: str = "https://racing.hkjc.com/x") -> list[dict]:
    parsed = parse_fixture.parse(raw("fixture", key, body, url))
    assert set(parsed) == {"meetings", "_meta"}  # the port's month metadata rides along
    return parsed["meetings"]


def summary(rows: list[dict]) -> list[tuple[int, str, str, str]]:
    return [(m["race_date"].day, m["venue"], m["day_night"], m["surface_mix"]) for m in rows]


def test_real_sectional_and_fixture_pages(hkjc_dir):
    # ---------------- sectionals
    body = body_of(hkjc_dir, SECTIONAL_2026)
    assert ADAPTER.classify("sectional", body) == STATUS_FETCHED
    assert ADAPTER.classify("sectional", body_of(hkjc_dir, NO_INFO_LIVE)) == STATUS_EMPTY
    rows = sectionals(body)
    assert ADAPTER.parse(raw("sectional", SECTIONAL_KEY, body, SECTIONAL_URL)) == {
        "sectionals": rows
    }
    # 12 runners over 1000 m = 3 sections; the 4th-6th cells hold a blank image and give nothing
    assert len(rows) == 36
    assert len({(r["horse_no"], r["section_no"]) for r in rows}) == 36
    assert {r["section_no"] for r in rows} == {1, 2, 3}
    assert {r["horse_no"] for r in rows} == set(range(1, 13))
    assert all(set(r) == SECTIONAL_COLUMNS for r in rows)
    assert all(
        r["jurisdiction"] == "hkjc" and r["race_id"] == "hkjc:2026-09-16:HV:01" for r in rows
    )
    cell = {(r["horse_no"], r["section_no"]): r for r in rows}
    # the leader's cell prints its lead over the second horse: forced to 0, kept raw
    assert cell[(11, 1)] == {
        "jurisdiction": "hkjc", "race_id": "hkjc:2026-09-16:HV:01", "horse_id": "HK_2023_J087",
        "horse_no": 11, "section_no": 1, "section_pos": 1, "section_margin_lengths": 0.0,
        "section_margin_raw": "3/4", "section_time_s": 12.80, "sub_split_times_s": [],
    }  # fmt: skip
    led = [cell[(11, n)] for n in (2, 3)]
    assert [(r["section_pos"], r["section_margin_lengths"], r["section_time_s"]) for r in led] == [
        (1, 0.0, 21.95), (1, 0.0, 22.79)]  # fmt: skip
    assert cell[(11, 3)]["section_margin_raw"] == "1-3/4"
    # position and margin from their own elements (the cell text runs "7" + "2-3/4" together)
    for no, horse_id, pos, margin, secs in (
        (12, "HK_2023_J308", [7, 5, 2], [2.75, 2.5, 1.75], [13.24, 21.91, 22.69]),
        (5, "HK_2022_H044", [11, 10, 12], [4.25, 5.5, 6.0], [13.48, 22.15, 22.87]),
    ):  # fmt: skip
        mine = [cell[(no, n)] for n in (1, 2, 3)]
        assert [r["horse_id"] for r in mine] == [horse_id] * 3
        assert [r["section_pos"] for r in mine] == pos
        assert [r["section_margin_lengths"] for r in mine] == margin
        assert [r["section_time_s"] for r in mine] == secs
    assert (cell[(3, 1)]["section_pos"], cell[(3, 1)]["section_margin_lengths"]) == (8, 3.0)
    # the 2026 sub-splits sit beside the section time and sum to it
    assert cell[(12, 2)]["sub_split_times_s"] == [10.71, 11.20]
    assert cell[(12, 3)]["sub_split_times_s"] == [11.19, 11.50]
    assert cell[(12, 1)]["sub_split_times_s"] == []
    for r in rows:
        if r["sub_split_times_s"]:
            assert sum(r["sub_split_times_s"]) == approx(r["section_time_s"], abs=0.011)
    for n in (1, 2, 3):  # each section ranks the whole field once
        in_section = [r for r in rows if r["section_no"] == n]
        assert sorted(r["section_pos"] for r in in_section) == list(range(1, 13))
        assert [r["section_margin_lengths"] for r in in_section if r["section_pos"] == 1] == [0.0]
        assert all(r["section_margin_lengths"] > 0 for r in in_section if r["section_pos"] > 1)
    for r in rows:
        assert isinstance(r["horse_no"], int) and isinstance(r["section_no"], int)
        assert isinstance(r["section_pos"], int) and isinstance(r["section_time_s"], float)
        assert isinstance(r["section_margin_lengths"], float) and isinstance(
            r["sub_split_times_s"], list
        )
        assert r["horse_id"].startswith("HK_")
    # the same race's results page: each runner's sections sum to its finish time, same ids,
    # and the last section's position is the finishing position
    totals: dict[int, float] = defaultdict(float)
    for r in rows:
        totals[r["horse_no"]] += r["section_time_s"]
    race = results(hkjc_dir, LIVE_2026)
    for runner in race["runners"]:
        no = runner["horse_no"]
        assert totals[no] == approx(runner["finish_time_s"], abs=1e-6), no
        assert cell[(no, 1)]["horse_id"] == runner["horse_id"], no
        assert cell[(no, 3)]["section_pos"] == runner["finish_pos"], no
        assert [cell[(no, n)]["section_pos"] for n in (1, 2, 3)] == runner["running_positions"], no
    # the race's split times are the leader's cumulative sections (no. 11 led throughout)
    leader = [cell[(11, n)]["section_time_s"] for n in (1, 2, 3)]
    cumulative = [sum(leader[: n + 1]) for n in range(3)]
    assert race["races"][0]["split_times_s"] == approx(cumulative, abs=1e-6)

    # the race: key, then URL, then the page's own meeting line; the key wins
    from_url = sectionals(body, key="not-a-race-key")
    from_content = sectionals(body, key="", url="https://example.invalid/x")
    assert len(from_url) == len(from_content) == 36
    assert {r["race_id"] for r in from_url + from_content} == {"hkjc:2026-09-16:HV:01"}
    keyed = sectionals(body, url="https://example.invalid/?RaceNo=7")
    assert {r["race_id"] for r in keyed} == {"hkjc:2026-09-16:HV:01"}
    # a page for another meeting or race is never filed under the requested race
    other_day = ("https://racing.hkjc.com/en-us/local/information/displaysectionaltime"
                 "?racedate=13/09/2026&RaceNo=1")  # fmt: skip
    elsewhere = [("2026-09-13_ST_R01", other_day), ("2026-09-16_ST_R01", SECTIONAL_URL),
                 ("not-a-race-key", other_day), ("2026-09-16_HV_R02", SECTIONAL_URL)]  # fmt: skip
    for key, url in elsewhere:
        assert sectionals(body, key=key, url=url) == [], key
    # an absurd colspan is clamped (and the header then no longer matches)
    wide = body.replace(b'colspan="6"', b'colspan="2000000000"', 1)
    assert wide != body and sectionals(wide) == []
    for other in (NO_INFO, CORUNNING_2010, ACTIVE, RETIRED):
        assert sectionals(body_of(hkjc_dir, other), key="2008-01-01_ST_R01") == [], other

    # ---------------- fixture calendars
    for name, month in FIXTURE_MONTHS.items():
        assert ADAPTER.classify("fixture", body_of(hkjc_dir, name)) == STATUS_FETCHED, name
        rows = meetings(body_of(hkjc_dir, name), month)
        assert ADAPTER.parse(raw("fixture", month, body_of(hkjc_dir, name)))["meetings"] == rows
        assert all(set(m) == MEETING_COLUMNS for m in rows), name
        assert [m["race_date"] for m in rows] == sorted(m["race_date"] for m in rows), name
        assert len({m["meeting_id"] for m in rows}) == len(rows), name
        year, mon = (int(p) for p in month.split("-"))
        assert {(m["race_date"].year, m["race_date"].month) for m in rows} == {(year, mon)}, name
        assert {m["status"] for m in rows} == {"scheduled"}, name
        assert all(
            m["meeting_id"] == f"hkjc:{m['race_date'].isoformat()}:{m['venue']}" for m in rows
        )
    assert ADAPTER.classify("fixture", body_of(hkjc_dir, LIVE_2026)) == STATUS_BLOCKED
    assert ADAPTER.classify("results", body_of(hkjc_dir, "fixture_2026-09.html")) == STATUS_BLOCKED

    sep = meetings(body_of(hkjc_dir, "fixture_2026-09.html"), "2026-09")
    assert summary(sep) == [
        (6, "ST", "day", "turf"), (9, "HV", "night", "turf"), (13, "ST", "day", "turf"),
        (16, "HV", "night", "turf"),
        (20, "ST", "day", "mixed"),  # cancelled: still a calendar cell, zero race lines
        (23, "HV", "night", "turf"), (27, "ST", "day", "mixed"),
    ]  # fmt: skip
    assert sep[5] == {
        "jurisdiction": "hkjc", "meeting_id": "hkjc:2026-09-23:HV", "race_date": date(2026, 9, 23),
        "venue": "HV", "season": "2026/27", "day_night": "night", "surface_mix": "turf",
        "status": "scheduled",
    }  # fmt: skip
    # the meetings of the other real pages are on the calendar of their month
    on_calendar = {m["meeting_id"] for m in sep}
    assert results(hkjc_dir, LIVE_2026)["races"][0]["meeting_id"] in on_calendar
    assert results(hkjc_dir, SIMULCAST)["races"][0]["meeting_id"] in on_calendar
    assert card(hkjc_dir, HV_R1)["races"][0]["meeting_id"] in on_calendar

    jan = meetings(body_of(hkjc_dir, "fixture_2010-01.html"), "2010-01")
    assert summary(jan) == [
        (1, "ST", "day", "turf"), (6, "HV", "night", "turf"), (10, "ST", "day", "mixed"),
        (13, "HV", "night", "turf"), (16, "ST", "day", "mixed"), (20, "HV", "night", "turf"),
        (24, "ST", "day", "turf"), (27, "HV", "night", "turf"), (31, "ST", "day", "turf"),
    ]  # fmt: skip
    assert {m["season"] for m in jan} == {"2009/10"}

    nov = meetings(body_of(hkjc_dir, "fixture_2019-11.html"), "2019-11")
    assert len(nov) == 8
    (abandoned,) = [m for m in nov if m["race_date"] == date(2019, 11, 13)]  # later abandoned
    assert (abandoned["venue"], abandoned["status"], abandoned["season"]) == (
        "HV",
        "scheduled",
        "2019/20",
    )
    assert abandoned["meeting_id"] == results(hkjc_dir, ABANDONED)["races"][0]["meeting_id"]

    # the page title decides the month, not the cache key
    stale = meetings(body_of(hkjc_dir, "fixture_2026-09.html"), "1999-01")
    assert {(m["race_date"].year, m["race_date"].month) for m in stale} == {(2026, 9)}
    # the real calendar cell alone: the month comes from the key
    cell_html = text_of(hkjc_dir, FIXTURE_CELL)
    wrapped = f"<html><body><table><tr>{cell_html}</tr></table></body></html>"
    one = meetings(wrapped, "2026-09")
    assert summary(one) == [(23, "HV", "night", "turf")]
    assert one[0]["meeting_id"] == "hkjc:2026-09-23:HV"
    assert meetings(wrapped, "2026-10")[0]["race_date"] == date(2026, 10, 23)
