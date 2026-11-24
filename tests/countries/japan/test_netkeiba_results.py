"""Ticket #52: the netkeiba results page -> meetings, races, runners, dividends.

HKTrack's tests/unit/sources/jra/test_parse_results.py ported onto synthetic pages
(netkeiba_pages.py): a plain race, a dead heat for 3rd, a withdrawal (取) with a dead heat for
2nd, a demoted (降), a disqualified (失) and an excluded (除) runner, and a jump race with two
runners pulled up (中). The real-page versions run under `local` in a separate ticket.

Contract pinned here (the coder follows it):
- `horsetrack.countries.japan.jra.netkeiba.parse_results.parse(page: RawPage) -> ParsedPage`
  emits HKTrack's tables and columns: meetings, races, runners, dividends. HKTrack has no
  incidents table; a runner's placing_raw, placing_code, dead_heat, is_scratched and lbw columns
  carry 取 / 除 / 中 / 失 / 降 / 同着.
- `parse_results.classify(body: bytes) -> str` is HKTrack's adapter rule for a results page:
  STATUS_FETCHED with a result table, STATUS_EMPTY for the epoch-dated page the site serves for
  an unknown race id (title holds 1970年1月01日), STATUS_BLOCKED otherwise.
- Ids: meeting `jra:<date>:<VENUE>` (venue from race-id digits 5-6 via jra.yaml), race
  `<meeting>:<RR>`; horses, jockeys and trainers by their link ids.

Every expected value is written out by hand from the page specs, or decoded from the printed
cells by the small decoders below; none is produced by the code under test.
"""

from __future__ import annotations

import importlib
import re
from collections import Counter
from dataclasses import replace
from functools import cache

import netkeiba_pages as pages
import pytest

from horsetrack.core.config import load_source
from horsetrack.core.ingest.types import STATUS_BLOCKED, STATUS_EMPTY, STATUS_FETCHED, RawPage

# By name, not an import statement: ruff sorts a module it cannot find yet with the third-party
# imports, so a plain import here would need re-sorting once the module exists.
pr = importlib.import_module("horsetrack.countries.japan.jra.netkeiba.parse_results")

LB_PER_KG = 2.20462
TABLES = ("meetings", "races", "runners", "dividends")
SEX = {"牡": "H", "牝": "M", "セ": "G"}
PAGES = {
    "plain": pages.PLAIN,
    "dead_heat": pages.DEAD_HEAT,
    "scratched": pages.SCRATCHED,
    "demoted": pages.DEMOTED,
    "jump": pages.JUMP,
}
PAGE_NAMES = list(PAGES)


def raw(key: str, body: bytes) -> RawPage:
    """A results page as the fetcher hands it over: the netkeiba race id is its key."""
    host = load_source("japan", "jra").hosts.db
    return RawPage("jra", "results", key, f"{host}/race/{key}/", body)


@cache
def parse_page(page: pages.ResultPage) -> dict:
    return pr.parse(raw(page.key, pages.render(page)))


def by_horse_no(parsed: dict) -> dict[int, dict]:
    return {row["horse_no"]: row for row in parsed["runners"]}


def only_race(parsed: dict) -> dict:
    assert len(parsed["races"]) == 1
    return parsed["races"][0]


# ---------------------------------------------------------------- expected values, by hand

# fmt: off
RACES = {
    "plain": dict(
        race_id="jra:2026-11-22:KYO:11", meeting_id="jra:2026-11-22:KYO", race_date="2026-11-22",
        venue="KYO", race_no=11, race_name="第9回シンセティックマイルS(GI)",
        source_race_key="202608050611", surface="turf", course_config="outer", distance_m=1600,
        going="GOOD", going_raw="良", race_class="G1", race_class_raw="3歳以上オープン",
        weight_type="set", post_time_local="15:40", field_size=10, n_runners=10, n_dividends=12,
        split_times_s=[12.3, 23.2, 34.6, 46.4, 58.3, 69.8, 81.0, 93.0], finish_time_s=93.0,
    ),
    "dead_heat": dict(
        race_id="jra:2026-08-30:NII:10", meeting_id="jra:2026-08-30:NII", race_date="2026-08-30",
        venue="NII", race_no=10, race_name="シンセティック直線S(L)",
        source_race_key="202604020810", surface="turf", course_config=None, distance_m=1000,
        going="HEAVY", going_raw="不良", race_class="L", race_class_raw="3歳以上オープン",
        weight_type="handicap", post_time_local="15:01", field_size=9, n_runners=9,
        n_dividends=17, split_times_s=[11.9, 22.1, 32.9, 44.0, 55.8], finish_time_s=55.8,
    ),
    "scratched": dict(
        race_id="jra:2026-10-11:TOK:05", meeting_id="jra:2026-10-11:TOK", race_date="2026-10-11",
        venue="TOK", race_no=5, race_name="2歳新馬", source_race_key="202605040205",
        surface="turf", course_config=None, distance_m=1800, going="GOOD TO YIELDING",
        going_raw="稍重", race_class="DEBUT", race_class_raw="2歳新馬", weight_type="age",
        post_time_local="12:10", field_size=10, n_runners=11, n_dividends=16,
        split_times_s=[12.8, 24.3, 36.3, 48.7, 61.3, 73.6, 85.2, 96.5, 109.6],
        finish_time_s=109.6,
    ),
    "demoted": dict(
        race_id="jra:2026-06-14:HSN:07", meeting_id="jra:2026-06-14:HSN", race_date="2026-06-14",
        venue="HSN", race_no=7, race_name="3歳以上1勝クラス", source_race_key="202609030207",
        surface="dirt", course_config=None, distance_m=1800, going="SOFT", going_raw="重",
        race_class="C1", race_class_raw="3歳以上1勝クラス", weight_type="set",
        post_time_local="13:20", field_size=11, n_runners=12, n_dividends=12,
        split_times_s=[12.5, 23.9, 36.9, 49.7, 62.3, 74.7, 87.0, 99.2, 112.3],
        finish_time_s=112.3,
    ),
    "jump": dict(
        race_id="jra:2026-07-05:FKS:04", meeting_id="jra:2026-07-05:FKS", race_date="2026-07-05",
        venue="FKS", race_no=4, race_name="障害3歳以上未勝利", source_race_key="202603020204",
        surface="jump", course_config=None, distance_m=2750, going="GOOD", going_raw="良",
        race_class="MAIDEN", race_class_raw="障害3歳以上未勝利", weight_type="allowance",
        post_time_local="11:05", field_size=9, n_runners=9, n_dividends=12,
        split_times_s=[], finish_time_s=182.5,  # no lap times in a jump: the winner's time
    ),
}
RACE_COLUMNS = (
    "race_id", "meeting_id", "race_no", "race_name", "source_race_key", "surface",
    "course_config", "distance_m", "going", "going_raw", "race_class", "race_class_raw",
    "weight_type", "post_time_local", "field_size",
)

# Payouts per 100 yen (pool label -> code by jra.yaml: 単勝 WIN, 複勝 PLACE, 枠連
# BRACKET_QUINELLA, 馬連 QUINELLA, ワイド QUINELLA_PLACE, 馬単 FORECAST, 三連複 TRIO, 三連単
# TIERCE). Combinations keep the page's order, so ordered pools keep theirs.
DIVIDENDS = {
    "plain": [
        ("WIN", "8", 340), ("PLACE", "8", 150), ("PLACE", "3", 180), ("PLACE", "10", 310),
        ("BRACKET_QUINELLA", "3,7", 820), ("QUINELLA", "3,8", 910),
        ("QUINELLA_PLACE", "3,8", 380), ("QUINELLA_PLACE", "8,10", 720),
        ("QUINELLA_PLACE", "3,10", 1040),
        ("FORECAST", "8,3", 1560), ("TRIO", "3,8,10", 4280), ("TIERCE", "8,3,10", 19870),
    ],
    "dead_heat": [  # dead heat for 3rd: four places, five pairs, two trios and tierces
        ("WIN", "5", 420),
        ("PLACE", "5", 170), ("PLACE", "2", 130), ("PLACE", "7", 290), ("PLACE", "9", 480),
        ("BRACKET_QUINELLA", "2,5", 690), ("QUINELLA", "2,5", 720),
        ("QUINELLA_PLACE", "2,5", 300), ("QUINELLA_PLACE", "5,7", 840),
        ("QUINELLA_PLACE", "5,9", 1350), ("QUINELLA_PLACE", "2,7", 610),
        ("QUINELLA_PLACE", "2,9", 1020),
        ("FORECAST", "5,2", 1480),
        ("TRIO", "2,5,7", 2350), ("TRIO", "2,5,9", 3980),
        ("TIERCE", "5,2,7", 9870), ("TIERCE", "5,2,9", 16540),
    ],
    "scratched": [  # dead heat for 2nd: two brackets, quinellas, forecasts and tierces
        ("WIN", "4", 620), ("PLACE", "4", 190), ("PLACE", "10", 120), ("PLACE", "1", 140),
        ("BRACKET_QUINELLA", "4,8", 530), ("BRACKET_QUINELLA", "1,4", 760),
        ("QUINELLA", "4,10", 640), ("QUINELLA", "1,4", 980),
        ("QUINELLA_PLACE", "4,10", 280), ("QUINELLA_PLACE", "1,4", 350),
        ("QUINELLA_PLACE", "1,10", 210),
        ("FORECAST", "4,10", 1450), ("FORECAST", "4,1", 2130),
        ("TRIO", "1,4,10", 1170),
        ("TIERCE", "4,10,1", 4560), ("TIERCE", "4,1,10", 6230),
    ],
    "demoted": [
        ("WIN", "6", 260), ("PLACE", "6", 130), ("PLACE", "1", 210), ("PLACE", "11", 190),
        ("BRACKET_QUINELLA", "1,5", 1030), ("QUINELLA", "1,6", 1120),
        ("QUINELLA_PLACE", "1,6", 430), ("QUINELLA_PLACE", "6,11", 390),
        ("QUINELLA_PLACE", "1,11", 870),
        ("FORECAST", "6,1", 1690), ("TRIO", "1,6,11", 2740), ("TIERCE", "6,1,11", 9310),
    ],
    "jump": [
        ("WIN", "7", 220), ("PLACE", "7", 110), ("PLACE", "2", 200), ("PLACE", "5", 140),
        ("BRACKET_QUINELLA", "2,7", 780), ("QUINELLA", "2,7", 820),
        ("QUINELLA_PLACE", "2,7", 330), ("QUINELLA_PLACE", "5,7", 190),
        ("QUINELLA_PLACE", "2,5", 450),
        ("FORECAST", "7,2", 1230), ("TRIO", "2,5,7", 760), ("TIERCE", "7,2,5", 3890),
    ],
}

# Beaten lengths worked out by hand, so the decoders below cannot share a mistake with the code.
SPOT_LENGTHS = {
    "plain": {8: 0.0, 3: 1.25, 10: 1.30, 1: 1.55, 9: 2.40, 4: 6.65, 7: 16.65},
    "dead_heat": {5: 0.0, 2: 0.5, 7: 1.5, 9: 1.5, 1: 1.55, 8: 3.30, 6: 16.30},
    "scratched": {4: 0.0, 10: 1.25, 1: 1.25, 7: 2.0, 3: 5.35, 5: 20.35},
    "demoted": {6: 0.0, 1: 2.0, 9: 3.0, 4: 4.5, 10: 5.65, 7: 18.15},
    "jump": {7: 0.0, 2: 4.0, 1: 8.25, 3: 14.5, 6: 24.5},
}
# fmt: on


# ---------------------------------------------------------------- independent decoders

NON_PLACE_CODES = {"取": "WV", "除": "WX", "中": "PU", "失": "DISQ"}
SCRATCH_WORDS = ("取", "除")  # never started; 中 and 失 did
_NAMED_MARGINS = {"ハナ": 0.05, "アタマ": 0.1, "クビ": 0.25, "大": 10.0, "大差": 10.0, "同着": 0.0}
_MARGIN = re.compile(r"(\d+)(?:\.(\d+)/(\d+))?|(\d+)/(\d+)")


def margin_lengths(raw_margin: str) -> float:
    """ハナ 0.05, アタマ 0.1, クビ 0.25, 大 / 大差 10, 同着 0, '3/4', '2', '1.1/2' = 1 1/2."""
    if raw_margin in _NAMED_MARGINS:
        return _NAMED_MARGINS[raw_margin]
    m = _MARGIN.fullmatch(raw_margin)
    assert m, f"test page has an unknown margin {raw_margin!r}"
    if m.group(4):
        return int(m.group(4)) / int(m.group(5))
    return int(m.group(1)) + (int(m.group(2)) / int(m.group(3)) if m.group(2) else 0.0)


def finish_pos(placing: str) -> int | None:
    """'3' -> 3, '4(降)' -> 4 (the official place), '取' / '除' / '中' / '失' -> None."""
    m = re.match(r"\d+", placing)
    return int(m.group()) if m else None


def seconds(printed: str) -> float | None:
    if not printed:
        return None
    minutes, rest = printed.split(":")
    return int(minutes) * 60 + float(rest)


def body_weight(printed: str) -> tuple[int | None, int | None]:
    m = re.fullmatch(r"(\d+)(?:\(([+-]?\d+)\))?", printed)
    if not m:
        return None, None
    return int(m.group(1)), (int(m.group(2)) if m.group(2) is not None else None)


def dead_heat_positions(runners: list[pages.Runner]) -> set[int]:
    counts = Counter(finish_pos(r.placing) for r in runners)
    return {pos for pos, n in counts.items() if pos is not None and n > 1}


def expected_code(placing: str, shared: set[int]) -> str | None:
    pos = finish_pos(placing)
    if pos is None:
        return NON_PLACE_CODES[placing]
    return "DH" if pos in shared else None


def cumulative_lengths(runners: list[pages.Runner]) -> dict[int, float | None]:
    """Lengths behind the winner: the running sum of the printed margins down the page's placed
    runners; None for a runner without a place."""
    out: dict[int, float | None] = {}
    total = 0.0
    for r in runners:
        if finish_pos(r.placing) is None:
            out[r.horse_no] = None
            continue
        if r.margin:
            total += margin_lengths(r.margin)
        out[r.horse_no] = total
    return out


def expected_runner(r: pages.Runner, race_id: str, jump: bool) -> dict:
    wt_kg, wt_change = body_weight(r.body_wt)
    row = {
        "jurisdiction": "jra",
        "race_id": race_id,
        "horse_no": r.horse_no,
        "draw": r.horse_no,  # JRA gates are numbered by saddle number
        "bracket": r.bracket,
        "horse_id": r.horse_id,
        "horse_name": r.horse_name,
        "jockey_code": r.jockey_code,
        "jockey_key": f"jockey:{r.jockey_code}",
        "jockey_name": r.jockey_name,
        "trainer_code": r.trainer_code,
        "trainer_key": f"trainer:{r.trainer_code}",
        "trainer_name": r.trainer_name,
        "sex": SEX[r.sex_age[0]],
        "age": int(r.sex_age[1:]),
        "actual_wt_kg": float(r.carried_kg),
        "placing_raw": r.placing,
        "finish_pos": finish_pos(r.placing),
        "is_scratched": r.placing in SCRATCH_WORDS,
        "running_positions": [int(p) for p in r.passing.split("-")] if r.passing else [],
        # a jump's 上り is the mean time per 200 m over the whole race, not the last 600 m
        "last_3f_s": None if jump or not r.last_3f else float(r.last_3f),
        "win_odds": float(r.win_odds) if r.win_odds != "---" else None,
        "win_fav_rank": int(r.fav_rank) if r.fav_rank else None,
        "horse_wt_kg": wt_kg,
        "horse_wt_change_kg": wt_change,
    }
    if r.placing != "失":  # a disqualified runner's printed time is not pinned (as in HKTrack)
        row["finish_time_s"] = seconds(r.time)
    return row


# ---------------------------------------------------------------- the whole page


@pytest.mark.parametrize("name", PAGE_NAMES)
def test_full_page_to_tables(name):
    page, want = PAGES[name], RACES[name]
    parsed = parse_page(page)
    assert set(TABLES) <= set(parsed), f"tables missing: {set(TABLES) - set(parsed)}"

    # meetings: the one meeting this race belongs to, venue from the race id via jra.yaml
    assert len(parsed["meetings"]) == 1
    meeting = parsed["meetings"][0]
    assert meeting["jurisdiction"] == "jra"
    assert meeting["meeting_id"] == want["meeting_id"]
    assert meeting["venue"] == want["venue"]
    assert str(meeting["race_date"])[:10] == want["race_date"]
    assert meeting["status"] == "run"

    # races
    race = only_race(parsed)
    for column in RACE_COLUMNS:
        assert race[column] == want[column], column
    assert race["jurisdiction"] == "jra" and race["status"] == "run"
    assert race["split_times_s"] == pytest.approx(want["split_times_s"], abs=1e-6)  # cumulative
    assert race["finish_time_s"] == pytest.approx(want["finish_time_s"], abs=1e-6)

    # runners: one row per declared runner, every column as printed
    printed = pages.runners(page)
    runners = parsed["runners"]
    assert len(runners) == want["n_runners"] == len(printed)
    assert sorted(r["horse_no"] for r in runners) == list(range(1, want["n_runners"] + 1))
    assert sum(not r["is_scratched"] for r in runners) == want["field_size"]
    got = by_horse_no(parsed)
    for p in printed:
        row = got[p.horse_no]
        for column, value in expected_runner(p, want["race_id"], want["surface"] == "jump").items():
            if isinstance(value, float):
                assert row[column] == pytest.approx(value, abs=1e-6), (p.horse_no, column)
            else:
                assert row[column] == value, (p.horse_no, column)
        assert row["actual_wt_lb"] == pytest.approx(float(p.carried_kg) * LB_PER_KG, abs=0.5)
        if row["horse_wt_kg"] is None:
            assert row["declared_horse_wt_lb"] is None, p.horse_no
        else:
            lb = row["horse_wt_kg"] * LB_PER_KG
            assert row["declared_horse_wt_lb"] == pytest.approx(lb, abs=0.5), p.horse_no
    winners = [r for r in runners if r["finish_pos"] == 1]
    assert len(winners) == 1
    assert winners[0]["finish_time_s"] == pytest.approx(want["finish_time_s"], abs=1e-6)

    # dividends: 8 pools, several rows for the pools that pay more than one combination
    dividends = parsed["dividends"]
    assert len(dividends) == want["n_dividends"]
    assert len({d["pool"] for d in dividends}) == 8
    assert all(d["race_id"] == want["race_id"] and d["jurisdiction"] == "jra" for d in dividends)


def test_full_page_names_the_winner():
    """Names are kept as printed (the trainer's [東] / [西] prefix dropped); codes are the keys."""
    parsed = parse_page(pages.PLAIN)
    winner = by_horse_no(parsed)[8]
    assert winner["finish_pos"] == 1 and winner["win_odds"] == pytest.approx(3.4)
    assert winner["horse_name"] == "ミヤコハイ" and winner["horse_id"] == "2022105108"
    assert winner["jockey_name"] == "架空騎手一" and winner["jockey_key"] == "jockey:05901"
    assert winner["trainer_name"] == "架空調教師一" and winner["trainer_key"] == "trainer:01201"
    assert winner["actual_wt_lb"] == 128 and winner["declared_horse_wt_lb"] == 1098  # 58, 498 kg
    win = [d for d in parsed["dividends"] if d["pool"] == "WIN"]
    assert [(d["combination"], d["dividend"]) for d in win] == [("8", 340.0)]


# ---------------------------------------------------------------- margins and placing codes


@pytest.mark.parametrize("name", PAGE_NAMES)
def test_margins_and_placing_codes(name):
    """lbw_lengths: the running sum of ハナ 0.05 / アタマ 0.1 / クビ 0.25 / n a/b / 大 10 down the
    placed order, 0 for the winner, the same for both halves of a dead heat (同着), null without a
    place; placing_code: DH for a shared place, WV 取 / WX 除 / PU 中 / DISQ 失, else null."""
    printed = pages.runners(PAGES[name])
    got = by_horse_no(parse_page(PAGES[name]))
    lengths = cumulative_lengths(printed)
    shared = dead_heat_positions(printed)
    assert set(got) == {p.horse_no for p in printed}
    for p in printed:
        row = got[p.horse_no]
        assert row["placing_raw"] == p.placing, p.horse_no
        if p.margin:
            assert row["lbw_raw"] == p.margin, p.horse_no
        else:
            assert not row["lbw_raw"], p.horse_no
        if lengths[p.horse_no] is None:
            assert row["lbw_lengths"] is None, p.horse_no
        else:
            assert row["lbw_lengths"] == pytest.approx(lengths[p.horse_no], abs=1e-6), p.horse_no
        assert row["finish_pos"] == finish_pos(p.placing), p.horse_no
        assert row["placing_code"] == expected_code(p.placing, shared), p.horse_no
        assert row["dead_heat"] is (finish_pos(p.placing) in shared), p.horse_no
        assert row["is_scratched"] is (p.placing in SCRATCH_WORDS), p.horse_no


@pytest.mark.parametrize("name", PAGE_NAMES)
def test_margins_and_placing_codes_spot_values(name):
    got = by_horse_no(parse_page(PAGES[name]))
    for horse_no, value in SPOT_LENGTHS[name].items():
        assert got[horse_no]["lbw_lengths"] == pytest.approx(value, abs=1e-6), horse_no


# (page, shared place, the two horses, their lengths, (next horse, its place, its lengths))
DEAD_HEATS = [
    pytest.param("dead_heat", 3, (7, 9), 1.5, (1, 5, 1.55), id="third"),
    pytest.param("scratched", 2, (10, 1), 1.25, (7, 4, 2.0), id="second"),
]


@pytest.mark.parametrize(("name", "place", "horses", "lengths", "after"), DEAD_HEATS)
def test_a_dead_heat_shares_the_position(name, place, horses, lengths, after):
    got = by_horse_no(parse_page(PAGES[name]))
    for horse_no in horses:
        row = got[horse_no]
        assert row["finish_pos"] == place, horse_no
        assert row["dead_heat"] is True and row["placing_code"] == "DH", horse_no
        assert row["lbw_lengths"] == pytest.approx(lengths, abs=1e-6), horse_no
        assert row["is_scratched"] is False, horse_no
    assert got[horses[1]]["lbw_raw"] == "同着"
    assert {n for n, row in got.items() if row["dead_heat"]} == set(horses)
    assert sum(row["finish_pos"] == place for row in got.values()) == 2
    # the next place is skipped, and the margins run on from the shared lengths
    next_horse, next_place, next_lengths = after
    row = got[next_horse]
    assert row["finish_pos"] == next_place
    assert row["dead_heat"] is False and row["placing_code"] is None
    assert row["lbw_lengths"] == pytest.approx(next_lengths, abs=1e-6)


DEMOTED_7TH = pages.restate(pages.restate(pages.DEMOTED, 9, placing="4"), 2, placing="7(降)")
# (page, demoted horse, placing as printed, official place, lengths, (next horse, place, lengths))
DEMOTIONS = [
    pytest.param(pages.DEMOTED, 9, "4(降)", 4, 3.0, (4, 5, 4.5), id="to_4th"),
    pytest.param(DEMOTED_7TH, 2, "7(降)", 7, 5.6, (10, 8, 5.65), id="to_7th"),
]


@pytest.mark.parametrize(("page", "horse_no", "placing", "place", "lengths", "after"), DEMOTIONS)
def test_a_demoted_runner_keeps_its_finishing_position_code(
    page, horse_no, placing, place, lengths, after
):
    """降着: the official place in finish_pos, the marker kept in placing_raw, no placing code."""
    parsed = parse_page(page)
    got = by_horse_no(parsed)
    row = got[horse_no]
    assert row["placing_raw"] == placing
    assert row["finish_pos"] == place
    assert row["placing_code"] is None
    assert row["is_scratched"] is False and row["dead_heat"] is False
    assert row["lbw_lengths"] == pytest.approx(lengths, abs=1e-6)
    next_horse, next_place, next_lengths = after
    assert got[next_horse]["finish_pos"] == next_place
    assert got[next_horse]["lbw_lengths"] == pytest.approx(next_lengths, abs=1e-6)
    assert only_race(parsed)["field_size"] == 11  # 12 declared, 1 excluded; 失 started


PLAIN_EXCLUDED = pages.restate(
    pages.PLAIN, 7, placing="除", time="-", margin="-", passing="-", last_3f="-", win_odds="---",
    fav_rank="-",
)  # fmt: skip
# (page, horse, placing as printed, code, field size)
SCRATCHINGS = [
    pytest.param(pages.SCRATCHED, 6, "取", "WV", 10, id="withdrawn"),
    pytest.param(pages.DEMOTED, 12, "除", "WX", 11, id="excluded"),
    pytest.param(PLAIN_EXCLUDED, 7, "除", "WX", 9, id="excluded_last_in_plain"),
]


@pytest.mark.parametrize(("page", "horse_no", "placing", "code", "field_size"), SCRATCHINGS)
def test_a_scratched_runner_has_no_position(page, horse_no, placing, code, field_size):
    """取消 / 除外: never started; no place, time, margin, odds or lengths; out of the field."""
    parsed = parse_page(page)
    row = by_horse_no(parsed)[horse_no]
    assert row["placing_raw"] == placing
    assert row["placing_code"] == code
    assert row["finish_pos"] is None
    assert row["is_scratched"] is True and row["dead_heat"] is False
    assert row["lbw_lengths"] is None and not row["lbw_raw"]
    assert row["finish_time_s"] is None
    assert row["win_odds"] is None and row["win_fav_rank"] is None
    assert row["running_positions"] == []
    assert only_race(parsed)["field_size"] == field_size
    assert sum(not r["is_scratched"] for r in parsed["runners"]) == field_size


# (page, horse, placing as printed, code, win odds, favourite rank, field size)
NON_FINISHERS = [
    pytest.param(pages.JUMP, 4, "中", "PU", 9.8, 4, 9, id="pulled_up_4"),
    pytest.param(pages.JUMP, 9, "中", "PU", 45.1, 8, 9, id="pulled_up_9"),
    pytest.param(pages.DEMOTED, 3, "失", "DISQ", 88.2, 11, 11, id="disqualified"),
]


@pytest.mark.parametrize(
    ("page", "horse_no", "placing", "code", "odds", "fav", "field_size"), NON_FINISHERS
)
def test_a_runner_that_did_not_finish_has_no_position(
    page, horse_no, placing, code, odds, fav, field_size
):
    """中止 / 失格: started (in the field, with odds) but no place and no beaten lengths."""
    parsed = parse_page(page)
    row = by_horse_no(parsed)[horse_no]
    assert row["placing_raw"] == placing
    assert row["placing_code"] == code
    assert row["finish_pos"] is None and row["lbw_lengths"] is None
    assert row["is_scratched"] is False and row["dead_heat"] is False
    assert row["win_odds"] == pytest.approx(odds) and row["win_fav_rank"] == fav
    if code == "PU":
        assert row["finish_time_s"] is None
    assert only_race(parsed)["field_size"] == field_size


# ---------------------------------------------------------------- dividends


@pytest.mark.parametrize("name", PAGE_NAMES)
def test_dividends_rows_per_pool_and_combination(name):
    dividends = parse_page(PAGES[name])["dividends"]
    got = Counter((d["pool"], d["combination"], d["dividend"]) for d in dividends)
    want = Counter(
        (pool, combination, float(amount)) for pool, combination, amount in DIVIDENDS[name]
    )
    assert got == want
    assert all(isinstance(d["dividend"], float) for d in dividends)
    assert all(d["is_refund"] is False for d in dividends)
    assert all(d["race_id"] == RACES[name]["race_id"] for d in dividends)
    assert {d["pool"] for d in dividends} <= set(load_source("japan", "jra").pools)


# ---------------------------------------------------------------- class, going, course, weights

# (class text in p.smalltxt, race name in h1 or None for the class text, race_class)
CLASS_VARIANTS = [
    ("2歳新馬", None, "DEBUT"),
    ("2歳未勝利", None, "MAIDEN"),
    ("3歳未出走", None, "MAIDEN"),
    ("3歳以上1勝クラス", None, "C1"),
    ("3歳以上2勝クラス", None, "C2"),
    ("3歳以上2勝クラス", "シンセ池特別(2勝)", "C2"),
    ("3歳以上3勝クラス", None, "C3"),
    ("3歳以上500万下", None, "C1"),  # the names before June 2019
    ("3歳以上1000万下", None, "C2"),
    ("3歳以上1600万下", None, "C3"),
    ("3歳以上オープン", "シンセティックS", "OP"),
    ("3歳以上オープン", "シンセティックS(OP)", "OP"),
    ("3歳以上オープン", "シンセティックS(L)", "L"),
    ("3歳以上オープン", "シンセティック杯(GIII)", "G3"),
    ("2歳オープン", "シンセティック2歳S(GII)", "G2"),
    ("3歳以上オープン", "シンセティックC(GI)", "G1"),
    ("3歳以上オープン", "シンセティック記念(G2)", "G2"),
    ("3歳以上オープン", "シンセティック記念（ＧＩＩＩ）", "G3"),  # full-width, NFKC
    ("障害4歳以上オープン", "シンセティック大障害(JGI)", "G1"),  # jump grades share the codes
    ("障害3歳以上オープン", "シンセティックJS(J.GII)", "G2"),
]
GOING_VARIANTS = [("良", "GOOD"), ("稍重", "GOOD TO YIELDING"), ("重", "SOFT"), ("不良", "HEAVY")]


@pytest.mark.parametrize(("class_text", "race_name", "race_class"), CLASS_VARIANTS)
def test_class_and_going_mapping(class_text, race_name, race_class):
    page = replace(pages.DEMOTED, class_text=class_text, race_name=race_name or class_text)
    race = only_race(parse_page(page))
    assert race["race_class"] == race_class
    assert race["race_class_raw"] == class_text


@pytest.mark.parametrize("base", ["plain", "demoted"])  # 芝 : <going> and ダート : <going>
@pytest.mark.parametrize(("going_raw", "going"), GOING_VARIANTS)
def test_class_and_going_mapping_going(base, going_raw, going):
    race = only_race(parse_page(replace(PAGES[base], going=going_raw)))
    assert race["going"] == going and race["going_raw"] == going_raw


@pytest.mark.parametrize(
    ("course", "surface", "course_config", "distance_m"),
    [
        ("芝右 外1600m", "turf", "outer", 1600),
        ("芝右 内2000m", "turf", "inner", 2000),
        ("芝左 外1400m", "turf", "outer", 1400),
        ("芝右1200m", "turf", None, 1200),
        ("芝直線1000m", "turf", None, 1000),
        ("ダ右1800m", "dirt", None, 1800),
        ("ダ左1400m", "dirt", None, 1400),
        ("障芝3170m", "jump", None, 3170),
        ("障芝 外-内4250m", "jump", None, 4250),  # both courses: neither
    ],
)
def test_full_page_to_tables_surface_and_course(course, surface, course_config, distance_m):
    race = only_race(parse_page(replace(pages.PLAIN, course=course)))
    assert (race["surface"], race["course_config"], race["distance_m"]) == (
        surface,
        course_config,
        distance_m,
    )


@pytest.mark.parametrize(
    ("conditions", "weight_type"),
    [
        ("(混)(馬齢)", "age"),
        ("(混)(定量)", "set"),
        ("(混)(別定)", "allowance"),
        ("(混)(ハンデ)", "handicap"),
        ("(国際)(特指)(ハンデ)", "handicap"),
        ("(混)", None),
    ],
)
def test_full_page_to_tables_weight_type(conditions, weight_type):
    race = only_race(parse_page(replace(pages.DEMOTED, conditions=conditions)))
    assert race["weight_type"] == weight_type


# ---------------------------------------------------------------- classification

UNKNOWN_RACE = "202608050613"  # a 13th race at the plain page's meeting: the site has none
CLASSIFIED = [
    pytest.param(UNKNOWN_RACE, pages.sentinel(), STATUS_EMPTY, id="sentinel"),
    pytest.param(pages.PLAIN.key, pages.render(pages.PLAIN), STATUS_FETCHED, id="result"),
    pytest.param(pages.JUMP.key, pages.render(pages.JUMP), STATUS_FETCHED, id="jump_result"),
    pytest.param(pages.PLAIN.key, pages.cut_off(pages.PLAIN), STATUS_BLOCKED, id="cut_off"),
    pytest.param(UNKNOWN_RACE, pages.error_page(), STATUS_BLOCKED, id="error_page"),
]


@pytest.mark.parametrize(("key", "body", "status"), CLASSIFIED)
def test_the_sentinel_page_is_classified_empty(key, body, status):
    """Decided by content, never by HTTP status: the site answers an unknown race id with 200 and
    an epoch-dated page. A page that is not a result puts nothing in the tables."""
    assert pr.classify(body) == status
    if status != STATUS_FETCHED:
        parsed = pr.parse(raw(key, body))
        assert all(not parsed.get(table) for table in TABLES), parsed
