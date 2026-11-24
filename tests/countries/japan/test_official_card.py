"""Ticket #53: the JRA official race card (出馬表, JRADB pw01dde page) -> races + racecard_runners.

HKTrack's tests/unit/sources/jra/test_racecard.py and test_racecard_kai_nichi.py ported onto
synthetic cards (jradb_pages.card). The real-page versions run under `local` in ticket #58.

Contract pinned here (the coder follows it):
- `horsetrack.countries.japan.jra.official_card.parse(page: RawPage) -> ParsedPage` with the
  tables "races" (one row) and "racecard_runners" (one row per row of the field table). A page
  without a race header gives no rows.
- The race's identity comes from the page (its date line and race-number image), never from the
  RawPage key. Ids as in the odds log and the results parser: meeting `jra:<date>:<VENUE>`, race
  `<meeting>:<RR>` (two digits); people keyed `jockey:<code>` / `trainer:<code>` by the 5-digit
  codes of their links, horses by the 10-character id of theirs (id_match "exact"); a runner
  without a horse link has horse_id null and id_match "none".
- Units and codes as HKTrack's: sex 牡/牝/セ -> H/M/G, kg kept, lb = round(kg x 2.20462), draw =
  horse number, weight_type 定量/別定/ハンデ/馬齢 -> set/allowance/handicap/age, trainer_centre
  栗東/美浦 -> ritto/miho, blinkers -> gear_raw "B" (else null), apprentice allowance from the
  mark's title ('1kg減量'), else its symbol (☆ 1, △ ◇ 2, ▲ 3, ★ 4).
- Not printed on the card, so null: going, course_letter. Body weights null before the weigh-in.
- A scratched runner (取消 / 除外 printed in place of its number) keeps its number from its place
  in the numbered field; when the printed numbers do not equal their row positions it stays null.
- Only the first table under #syutsuba is the field: its past-run cells and the past-winners
  table (jradb_pages adds both to every card) are other races.

Every expected value is written out by hand in the specs below or derived from them here (lb,
the printed change), never by the code under test. Names, ids and codes are invented.
"""

from __future__ import annotations

import importlib
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import jradb_pages as pages
import pytest
from jradb_pages import Card, CardRunner

from horsetrack.core.ingest.types import RawPage

# By name, not an import statement: ruff sorts a module it cannot find yet with the third-party
# imports, so a plain import here would need re-sorting once the module exists.
oc = importlib.import_module("horsetrack.countries.japan.jra.official_card")

LB_PER_KG = 2.20462
TOKYO_TZ = ZoneInfo("Asia/Tokyo")
SCRAPED = datetime(2026, 11, 27, 23, 30, tzinfo=timezone.utc)
URL = "https://www.jra.go.jp/JRADB/accessD.html"
OPAQUE_KEY = "pw01dde0106202601010120260104"  # names another race: identity comes from the page
SEX = {"牡": "H", "牝": "M", "セ": "G"}
CENTRE = {"栗東": "ritto", "美浦": "miho"}

RUNNER_COLUMNS = (
    "jurisdiction", "race_id", "horse_id", "id_match", "horse_no", "horse_name", "bracket",
    "draw", "actual_wt_kg", "actual_wt_lb", "allowance_kg", "allowance_lb", "jockey_key",
    "jockey_code", "jockey_name", "trainer_key", "trainer_code", "trainer_name",
    "trainer_centre", "horse_wt_kg", "horse_wt_change_kg", "declared_horse_wt_lb", "sex", "age",
    "gear_raw", "owner", "sire", "dam", "dam_sire", "is_scratched", "card_scraped_utc",
)  # fmt: skip


def lb(kg: float | None) -> int | None:
    return round(kg * LB_PER_KG) if kg is not None else None


def runner(number: int, horse_id: str | None, name: str, sex: str, age: int, kg: float, **kw):
    """A runner of an invented field: bracket = number (fields of 8 or fewer), jockey 010NN,
    trainer 011NN, names and pedigree numbered too."""
    defaults = dict(
        bracket=number,
        jockey_code=f"010{number:02d}",
        jockey=f"騎手 {number}郎",
        trainer_code=f"011{number:02d}",
        trainer=f"調教 {number}郎",
        owner=f"試験ファーム{number}",
        sire=f"チチウマ{number}",
        dam=f"ハハウマ{number}",
        dam_sire=f"Synthetic Sire {number}",
    )
    return CardRunner(number, horse_id=horse_id, name=name, sex=sex, age=age, carried_kg=kg,
                      **{**defaults, **kw})  # fmt: skip


def scratch(field: tuple[CardRunner, ...], words: dict[int, str]) -> tuple[CardRunner, ...]:
    """The field with runners `words` scratched as the real card prints them: the word in place
    of the number, no body weight."""
    return tuple(
        replace(r, scratched=words[r.number], body_kg=None, body_change=None)
        if r.number in words
        else r
        for r in field
    )


# Tokyo, Sat 28 Nov 2026, race 11: 8 declared, No 4 scratched (取消), two in blinkers, two
# trainers in Miho, body weights in.
TOKYO_FIELD = (
    runner(1, "2022100011", "シンセアオイ", "牡", 4, 58.0, body_kg=480, body_change=4),
    runner(2, "2021100022", "テストミドリ", "牝", 5, 56.0, body_kg=446, body_change=-6,
           blinkers=True, centre="美浦"),
    runner(3, "2020100033", "モデルクロノ", "セ", 6, 58.0, body_kg=502, body_change="±0"),
    runner(4, "2023100044", "ダミーハヤテ", "牡", 3, 57.0, body_kg=490, body_change=2),
    runner(5, "2023100055", "サンプルユキ", "牝", 3, 55.0, body_kg=432, body_change=-2),
    runner(6, "2022100066", "カリノホシ", "牡", 4, 58.0, body_kg=518, body_change=-12,
           blinkers=True, centre="美浦"),
    runner(7, "2021100077", "ツクリモノ", "セ", 5, 58.0, body_kg=470, body_change=2),
    runner(8, "2022100088", "ヨウイドン", "牝", 4, 56.0, body_kg=438, body_change=10),
)  # fmt: skip
TOKYO = Card(
    day=date(2026, 11, 28),
    meeting="5回東京7日",
    race_no=11,
    race_name="シンセティック特別",
    category="3歳以上",
    race_class="2勝クラス",
    weight_rule="定量",
    distance_m=1600,
    detail="（ダート・左）",
    post="15時25分",
    runners=scratch(TOKYO_FIELD, {4: "取消"}),
)

# Kyoto, Sun 29 Nov 2026, race 3: 6 declared, No 6 excluded (除外, the last row), a ☆ apprentice
# on No 1, No 2 printed without a horse link, No 3 with no weight change printed.
KYOTO = Card(
    day=date(2026, 11, 29),
    meeting="5回京都8日",
    race_no=3,
    race_name="2歳未勝利",
    category="2歳",
    race_class="未勝利",
    weight_rule="馬齢",
    distance_m=2000,
    detail="（芝・右 外）",
    post="10時50分",
    runners=scratch(
        (
            runner(1, "2024100101", "ヒナドリイチ", "牡", 2, 54.0, body_kg=466, body_change=8,
                   claim_mark="☆", claim_title="1kg減量"),
            runner(2, None, "ナナシノゴンベ", "牝", 2, 55.0, body_kg=420, body_change=-4),
            runner(3, "2024100103", "ミライノカゼ", "牡", 2, 56.0, body_kg=488),
            runner(4, "2024100104", "アシタノヒカリ", "牝", 2, 55.0, body_kg=410, body_change=2,
                   blinkers=True, centre="美浦"),
            runner(5, "2024100105", "ユメノツバサ", "牡", 2, 56.0, body_kg=474, body_change=-10),
            runner(6, "2024100106", "オワリノウタ", "牡", 2, 56.0, body_kg=452, body_change=6),
        ),
        {6: "除外"},
    ),
)  # fmt: skip

# What each card's race row must say, by hand.
RACES = {
    "tokyo": dict(
        race_id="jra:2026-11-28:TOK:11", meeting_id="jra:2026-11-28:TOK", race_no=11,
        race_name="シンセティック特別", distance_m=1600, surface="dirt", course_config=None,
        race_class_raw="3歳以上2勝クラス", weight_type="set", field_size=7, kai=5, nichi=7,
        going=None, course_letter=None, status="declared",
    ),
    "kyoto": dict(
        race_id="jra:2026-11-29:KYO:03", meeting_id="jra:2026-11-29:KYO", race_no=3,
        race_name="2歳未勝利", distance_m=2000, surface="turf", course_config="outer",
        race_class_raw="2歳未勝利", weight_type="age", field_size=5, kai=5, nichi=8,
        going=None, course_letter=None, status="declared",
    ),
}  # fmt: skip
CARDS = {"tokyo": TOKYO, "kyoto": KYOTO}
POSTS_UTC = {
    "tokyo": datetime(2026, 11, 28, 6, 25, tzinfo=timezone.utc),  # 15:25 in Tokyo
    "kyoto": datetime(2026, 11, 29, 1, 50, tzinfo=timezone.utc),  # 10:50 in Tokyo
}


def parse_card(spec: Card, *, fetched: datetime = SCRAPED) -> dict[str, list[dict]]:
    raw = RawPage("jra", "racecard", OPAQUE_KEY, URL, pages.card(spec), fetched_utc=fetched)
    return oc.parse(raw)


def only_race(parsed: dict) -> dict:
    assert len(parsed["races"]) == 1
    return parsed["races"][0]


def by_name(parsed: dict) -> dict[str, dict]:
    rows = {row["horse_name"]: row for row in parsed["racecard_runners"]}
    assert len(rows) == len(parsed["racecard_runners"]), "one row per runner"
    return rows


def printed_change(change: int | str | None) -> int | None:
    if change is None or isinstance(change, int):
        return change
    return int(change.replace("±", ""))


def claim_kg(r: CardRunner) -> float | None:
    """The claim the mark's title states, else the symbol's (the card's own rule)."""
    if r.claim_title is not None:
        return float(r.claim_title.removesuffix("kg減量"))
    return {"☆": 1.0, "△": 2.0, "◇": 2.0, "▲": 3.0, "★": 4.0}.get(r.claim_mark or "")


def expected_runner(r: CardRunner, race_id: str) -> dict:
    claim = claim_kg(r)
    return {
        "jurisdiction": "jra",
        "race_id": race_id,
        "horse_id": r.horse_id,
        "id_match": "exact" if r.horse_id is not None else "none",
        "horse_no": r.number,
        "horse_name": r.name,
        "bracket": r.bracket,
        "draw": r.number,
        "actual_wt_kg": r.carried_kg,
        "actual_wt_lb": lb(r.carried_kg),
        "allowance_kg": claim,
        "allowance_lb": lb(claim),
        "jockey_key": f"jockey:{r.jockey_code}",
        "jockey_code": r.jockey_code,
        "jockey_name": r.jockey,
        "trainer_key": f"trainer:{r.trainer_code}",
        "trainer_code": r.trainer_code,
        "trainer_name": r.trainer,
        "trainer_centre": CENTRE[r.centre],
        "horse_wt_kg": r.body_kg,
        "horse_wt_change_kg": printed_change(r.body_change),
        "declared_horse_wt_lb": lb(r.body_kg),
        "sex": SEX[r.sex],
        "age": r.age,
        "gear_raw": "B" if r.blinkers else None,
        "owner": r.owner,
        "sire": r.sire,
        "dam": r.dam,
        "dam_sire": r.dam_sire,
        "is_scratched": r.scratched is not None,
        "card_scraped_utc": SCRAPED,
    }


def as_aware(value: str | datetime) -> datetime:
    return datetime.fromisoformat(value) if isinstance(value, str) else value


# ---------------------------------------------------------------- the whole card


@pytest.mark.parametrize("name", list(CARDS))
def test_card_to_racecard_runners(name):
    spec, want = CARDS[name], RACES[name]
    parsed = parse_card(spec)

    rows = by_name(parsed)
    assert set(rows) == {r.name for r in spec.runners}, "the field only, not the past winners"
    for r in spec.runners:
        got = rows[r.name]
        assert {c: got[c] for c in RUNNER_COLUMNS} == expected_runner(r, want["race_id"]), r.name

    race = only_race(parsed)
    assert {k: race[k] for k in want} == want
    assert isinstance(race["kai"], int) and isinstance(race["nichi"], int)
    post = as_aware(race["post_time_local"])
    assert post.utcoffset() == timedelta(hours=9), "a full Tokyo date-time, not a bare clock"
    assert post == POSTS_UTC[name]


def test_card_scraped_utc_is_the_fetch_time_and_an_error_page_gives_nothing():
    later = datetime(2026, 11, 28, 4, 10, tzinfo=timezone.utc)
    rows = parse_card(TOKYO, fetched=later)["racecard_runners"]
    assert rows and {row["card_scraped_utc"] for row in rows} == {later}

    raw = RawPage("jra", "racecard", OPAQUE_KEY, URL, pages.no_header_page(), fetched_utc=later)
    parsed = oc.parse(raw)
    assert parsed["races"] == [] and parsed["racecard_runners"] == []


# ---------------------------------------------------------------- the race header


@pytest.mark.parametrize(
    "detail, distance, surface, course_config",
    [
        ("（ダート・右）", 1800, "dirt", None),
        ("（ダート・左）", 2100, "dirt", None),
        ("（芝・右 外）", 1600, "turf", "outer"),  # Kyoto's outer course
        ("（芝・右）", 2000, "turf", None),  # an inner course prints no 内
        ("（芝・左 内）", 1400, "turf", None),  # an explicit inner course is null too
        ("（障害・芝）", 3110, "jump", None),  # any 障 is a jump, whatever the surface words
        ("（障害・芝 ダート）", 2880, "jump", None),
        ("（障害・芝 外-内）", 3900, "jump", None),  # 外 and 内: not the outer course
    ],
)
def test_course_detail_gives_surface_and_outer_course(detail, distance, surface, course_config):
    race = only_race(parse_card(replace(TOKYO, detail=detail, distance_m=distance)))
    assert (race["surface"], race["course_config"], race["distance_m"]) == (
        surface,
        course_config,
        distance,
    )


@pytest.mark.parametrize(
    "meeting, day, race_no, venue, kai, nichi",
    [
        ("5回東京7日", date(2026, 11, 28), 11, "TOK", 5, 7),
        ("4回京都10日", date(2026, 10, 25), 12, "KYO", 4, 10),
        ("1回中山1日", date(2026, 1, 4), 1, "NAK", 1, 1),
        ("2回小倉12日", date(2026, 2, 22), 5, "KOK", 2, 12),
        ("3回福島2日", date(2026, 11, 8), 9, "FKS", 3, 2),
    ],
)
def test_kai_and_nichi_from_the_meeting_name(meeting, day, race_no, venue, kai, nichi):
    """JRDB keys an upcoming race by venue, year, 回, 日 and race number: the race row carries
    the 回 and 日 of the date line ('2026年11月28日（土曜） 5回東京7日') as ints."""
    parsed = parse_card(replace(TOKYO, meeting=meeting, day=day, race_no=race_no))
    race = only_race(parsed)
    race_id = f"jra:{day.isoformat()}:{venue}:{race_no:02d}"
    assert (race["race_id"], race["meeting_id"], race["race_no"]) == (
        race_id,
        f"jra:{day.isoformat()}:{venue}",
        race_no,
    )
    assert (race["kai"], race["nichi"]) == (kai, nichi)
    assert isinstance(race["kai"], int) and isinstance(race["nichi"], int)
    assert {row["race_id"] for row in parsed["racecard_runners"]} == {race_id}


# ---------------------------------------------------------------- runners


@pytest.mark.parametrize(
    "number, mark, title, claim",
    [
        (5, "☆", "1kg減量", 1.0),
        (5, "△", "2kg減量", 2.0),
        (1, "◇", "2kg減量", 2.0),
        (5, "▲", "3kg減量", 3.0),
        (8, "★", "4kg減量", 4.0),
        (5, "★", "2kg減量", 2.0),  # the title states the claim; the symbol is a fallback
        (1, "▲", None, 3.0),  # no title: the symbol's claim
        (5, "☆", None, 1.0),
        (8, "△", None, 2.0),
    ],
)
def test_apprentice_allowance_from_the_jockey_mark(number, mark, title, claim):
    """The mark before the rider's name (`<span title="1kg減量" class="mark jockey">☆</span>`);
    the printed weight carried is after the claim. The ★ of every past-run cell is another
    race's rider and claims nothing here."""
    base = next(r for r in TOKYO.runners if r.number == number)
    carried = base.carried_kg - claim
    claimed = replace(base, claim_mark=mark, claim_title=title, carried_kg=carried)
    field = tuple(claimed if r.number == number else r for r in TOKYO.runners)
    rows = by_name(parse_card(replace(TOKYO, runners=field)))
    got = rows[base.name]
    assert (got["allowance_kg"], got["allowance_lb"]) == (claim, lb(claim))
    assert (got["actual_wt_kg"], got["actual_wt_lb"]) == (carried, lb(carried))
    assert got["jockey_key"] == f"jockey:{base.jockey_code}"
    others = {n: row["allowance_kg"] for n, row in rows.items() if n != base.name}
    assert others == dict.fromkeys(others), "no other runner claims"


@pytest.mark.parametrize("blinkered", [set(), {1}, {2, 6}, {3, 7, 8}, {1, 2, 3, 5, 6, 7, 8}])
def test_blinkers_are_gear_b(blinkered):
    """The blinker icon in the number cell -> gear_raw "B"; no icon -> null."""
    field = tuple(replace(r, blinkers=r.number in blinkered) for r in TOKYO.runners)
    rows = parse_card(replace(TOKYO, runners=field))["racecard_runners"]
    gear = {row["horse_no"]: row["gear_raw"] for row in rows}
    assert gear == {n: "B" if n in blinkered else None for n in range(1, 9)}


@pytest.mark.parametrize(
    "words",
    [
        {4: "取消"},
        {1: "取消"},
        {8: "除外"},
        {1: "取消", 8: "取消"},
        {3: "除外", 4: "取消"},
        {2: "取消", 5: "除外", 7: "取消"},
    ],
)
def test_a_scratched_runner_keeps_its_horse_number(words):
    """The card prints 取消 / 除外 in place of a scratched runner's number; the number is still
    known from its place in the numbered field (HKTrack ruling, PR #16), and so is its draw."""
    parsed = parse_card(replace(TOKYO, runners=scratch(TOKYO_FIELD, words)))
    rows = by_name(parsed)
    by_runner = {r.number: rows[r.name] for r in TOKYO_FIELD}
    assert {n: row["horse_no"] for n, row in by_runner.items()} == {n: n for n in range(1, 9)}
    assert {n: row["draw"] for n, row in by_runner.items()} == {n: n for n in range(1, 9)}
    assert {n for n, row in by_runner.items() if row["is_scratched"]} == set(words)
    assert {n: row["horse_id"] for n, row in by_runner.items()} == {
        r.number: r.horse_id for r in TOKYO_FIELD
    }
    assert only_race(parsed)["field_size"] == 8 - len(words)


def test_a_scratched_runner_off_a_numbered_layout_gets_no_number():
    """When the printed numbers do not equal their row positions (here rows 1 and 2 swapped),
    the scratched runner's number is not guessed: it stays null; the printed ones are kept."""
    field = scratch(TOKYO_FIELD, {3: "取消"})
    swapped = (field[1], field[0], *field[2:])
    rows = by_name(parse_card(replace(TOKYO, runners=swapped)))
    got = {r.number: (rows[r.name]["horse_no"], rows[r.name]["draw"]) for r in TOKYO_FIELD}
    assert got == {n: (None, None) if n == 3 else (n, n) for n in range(1, 9)}


@pytest.mark.parametrize(
    "body, change, want_change",
    [
        (None, None, None),  # before the weigh-in: no body-weight cell at all
        (452, "±0", 0),
        (452, "0", 0),
        (452, "+16", 16),
        (530, "-20", -20),
        (398, None, None),  # a weight with no change printed
    ],
)
def test_before_the_weigh_in_body_weights_are_null(body, change, want_change):
    """About 65 minutes before the off the weigh-in fills the body-weight cells; before it the
    card prints none and the three body-weight columns are null. Every runner gets `body`."""
    field = tuple(replace(r, body_kg=body, body_change=change) for r in TOKYO_FIELD)
    rows = by_name(parse_card(replace(TOKYO, runners=field)))
    assert set(rows) == {r.name for r in TOKYO_FIELD}
    for r in TOKYO_FIELD:
        got = rows[r.name]
        assert (got["horse_wt_kg"], got["horse_wt_change_kg"], got["declared_horse_wt_lb"]) == (
            body,
            want_change,
            lb(body),
        ), r.name
        assert got["horse_id"] == r.horse_id and got["actual_wt_kg"] == r.carried_kg
