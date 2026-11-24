"""The ported JRA and JRDB parsers against the real pages and public samples (ticket #58).

Every test is marked `local` and uses `real_fixtures` (tests/conftest.py): without
$HORSETRACK_DATA_DIR/fixtures the file is skipped, so CI never runs it.

Pages (`fixtures/jra/`, each starts with a comment naming its source and fetch time):
- netkeiba results `results_*.html`: Mile Championship 2025 (Kyoto R11, G1), Kyoto Kimpai 2010
  (G3 handicap, one 取), a dead heat for 3rd on heavy turf (Niigata straight 1000m, L), a jump with
  two runners pulled up (中), a 取 plus a dead heat for 2nd (debut race), an excluded runner (除)
  on heavy dirt; two derived pages (a demoted 降 and a disqualified 失 runner; a pre-2019 500万下
  class) and the epoch-dated page the site serves for an unknown race id.
- netkeiba five-run cards `pedigree_card_*.html` of the first two races; the calendar of
  November 2025, the race list of 23 Nov 2025, the no-races list and the CDN error page.
- The JRA official card of Kyoto Sun 15 Nov 2026 race 10 (花背特別), saved as UTF-8 and
  re-encoded to cp932 here, as on the wire (as in test_official_real.py).

Expectations are ported from HKTrack's tests/unit/sources/jra/{test_parse_results,
test_parse_pedigree_card, test_racecard, test_racecard_kai_nichi, test_adapter}.py, where they
were read off the pages by hand. Derived cases edit the real page text in memory, as HKTrack's did.

JRDB (`fixtures/jrdb/`): JRDB's public samples, parsed with the spec tables under
$HORSETRACK_DATA_DIR/jrdb/specs. JRDB figures never enter this repository (owner ruling D21): no
JRDB value, offset or length is written here. Every expected value is decoded at run time from
the sample's own bytes, sliced by the spec table read with the csv module, never by the parser.
"""

from __future__ import annotations

import csv
import io
import math
import re
import unicodedata
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from functools import cache
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from horsetrack.core.config import load_source
from horsetrack.core.ingest.types import STATUS_BLOCKED, STATUS_EMPTY, STATUS_FETCHED, RawPage
from horsetrack.countries.japan.jra import official_card as oc
from horsetrack.countries.japan.jra.adapter import ADAPTER
from horsetrack.countries.japan.jra.netkeiba import parse_pedigree_card, parse_results
from horsetrack.countries.japan.jrdb import join, parse, pedigree

pytestmark = pytest.mark.local

UTC = timezone.utc
LB_PER_KG = 2.20462
DB = "https://db.netkeiba.com"
SEX = {"牡": "H", "牝": "M", "セ": "G"}
TABLES = ("meetings", "races", "runners", "dividends")


@pytest.fixture
def jra_dir(real_fixtures: Path) -> Path:
    return real_fixtures / "jra"


@pytest.fixture
def jrdb_dir(real_fixtures: Path) -> Path:
    return real_fixtures / "jrdb"


@cache
def _read(path: Path) -> bytes:
    return path.read_bytes()


def page_bytes(jra_dir: Path, name: str) -> bytes:
    return _read(jra_dir / name)


def page_text(jra_dir: Path, name: str) -> str:
    return page_bytes(jra_dir, name).decode("utf-8")


# ================================================================ netkeiba results

MILE_CS_2025 = "results_202508040611.html"
KIMPAI_2010 = "results_201008010111.html"
DEAD_HEAT_3RD = "results_202604040411_deadheat_3rd_heavy.html"
JUMP = "results_202603030204_jump_pulled_up.html"
SCRATCH_DEAD_HEAT_2ND = "results_202605041104_scratched_deadheat_2nd.html"
EXCLUDED = "results_202608040807_excluded_dirt_heavy.html"
DEMOTED_DERIVED = "results_202608040807_demoted_disqualified_derived.html"
CLASS_500MAN_DERIVED = "results_201008010111_500man_derived.html"
SENTINEL = "results_202508050611_sentinel.html"


def race_key(file_name: str) -> str:
    """The netkeiba race id is the cache key of a results page (and part of its file name)."""
    return file_name.split("_")[1][:12]


def results_raw(body: bytes | str, key: str) -> RawPage:
    data = body.encode("utf-8") if isinstance(body, str) else body
    return RawPage("jra", "results", key, f"{DB}/race/{key}/", data)


@cache
def _parsed_results(path: Path) -> dict:
    return parse_results.parse(results_raw(_read(path), race_key(path.name)))


def parse_file(jra_dir: Path, name: str) -> dict:
    return _parsed_results(jra_dir / name)


def by_horse_no(parsed: dict) -> dict[int, dict]:
    return {row["horse_no"]: row for row in parsed["runners"]}


def only_race(parsed: dict) -> dict:
    assert len(parsed["races"]) == 1
    return parsed["races"][0]


# Race-level expectations, read off the pages by hand (HKTrack's test_parse_results.py).
# fmt: off
RACES = {
    MILE_CS_2025: dict(
        race_id="jra:2025-11-23:KYO:11", meeting_id="jra:2025-11-23:KYO",
        race_date="2025-11-23", venue="KYO", race_no=11, surface="turf", course_config="outer",
        distance_m=1600, going="GOOD", going_raw="良", race_class="G1", weight_type="set",
        post_time_local="15:40", field_size=18, n_runners=18, n_dividends=12,
        split_times_s=[12.2, 22.9, 34.2, 45.9, 57.7, 69.1, 80.1, 91.3], finish_time_s=91.3,
    ),
    KIMPAI_2010: dict(
        race_id="jra:2010-01-05:KYO:11", meeting_id="jra:2010-01-05:KYO",
        race_date="2010-01-05", venue="KYO", race_no=11, surface="turf", course_config="outer",
        distance_m=1600, going="GOOD", going_raw="良", race_class="G3", weight_type="handicap",
        post_time_local="15:46", field_size=15, n_runners=16, n_dividends=12,
        split_times_s=[12.0, 22.6, 34.2, 46.4, 58.2, 70.5, 81.8, 94.1], finish_time_s=94.1,
    ),
    DEAD_HEAT_3RD: dict(
        race_id="jra:2026-10-25:NII:11", meeting_id="jra:2026-10-25:NII",
        race_date="2026-10-25", venue="NII", race_no=11, surface="turf", course_config=None,
        distance_m=1000, going="HEAVY", going_raw="不良", race_class="L",
        weight_type="handicap", post_time_local="15:11", field_size=17, n_runners=17,
        n_dividends=17, split_times_s=[12.1, 22.6, 33.7, 44.9, 56.7], finish_time_s=56.7,
    ),
    JUMP: dict(
        race_id="jra:2026-11-08:FKS:04", meeting_id="jra:2026-11-08:FKS",
        race_date="2026-11-08", venue="FKS", race_no=4, surface="jump", course_config=None,
        distance_m=2750, going="GOOD TO YIELDING", going_raw="稍重", race_class="MAIDEN",
        weight_type="set", post_time_local="11:15", field_size=11, n_runners=11,
        n_dividends=12, split_times_s=[], finish_time_s=184.9,  # no lap times for jumps
    ),
    SCRATCH_DEAD_HEAT_2ND: dict(
        race_id="jra:2026-11-01:TOK:04", meeting_id="jra:2026-11-01:TOK",
        race_date="2026-11-01", venue="TOK", race_no=4, surface="turf", course_config=None,
        distance_m=1600, going="GOOD", going_raw="良", race_class="DEBUT", weight_type="age",
        post_time_local="11:30", field_size=13, n_runners=14, n_dividends=16,
        split_times_s=[12.9, 25.1, 37.8, 49.9, 62.1, 73.7, 85.1, 96.4], finish_time_s=96.4,
    ),
    EXCLUDED: dict(
        race_id="jra:2026-10-24:KYO:07", meeting_id="jra:2026-10-24:KYO",
        race_date="2026-10-24", venue="KYO", race_no=7, surface="dirt", course_config=None,
        distance_m=1800, going="HEAVY", going_raw="不良", race_class="C1", weight_type="set",
        post_time_local="13:15", field_size=10, n_runners=11, n_dividends=12,
        split_times_s=[12.1, 23.4, 35.9, 48.1, 60.4, 72.6, 85.0, 97.4, 110.1],
        finish_time_s=110.1,
    ),
}
# fmt: on
# The derived pages change only what their first comment says: the race is the base page's.
RACES[DEMOTED_DERIVED] = dict(RACES[EXCLUDED])  # 11 declared, 1 excluded; the 失 runner started
RACES[CLASS_500MAN_DERIVED] = dict(RACES[KIMPAI_2010], race_class="C1")
CLASS_RAW_PART = {
    JUMP: "未勝利", SCRATCH_DEAD_HEAT_2ND: "新馬", EXCLUDED: "1勝クラス",
    DEMOTED_DERIVED: "1勝クラス", CLASS_500MAN_DERIVED: "500万下",
}  # fmt: skip

# Full runner tables of the first two pages, in page order. Columns: horse no, placing as printed,
# bracket, horse id, sex+age, carried kg, jockey code, time, margin, passing order, last 3f, win
# odds, favourite rank, body weight (change), trainer code. "-" = empty cell.
RUNNERS = {
    MILE_CS_2025: """
        15 1  7 2021107171 牡4 58 01088 1:31.3 -     3-2   33.1 1.8   1  500(-6)  01117
        6  2  3 2019104476 牡6 58 01170 1:31.6 1.3/4 4-4   33.2 8.6   4  502(+6)  01157
        3  3  2 2021100953 牡4 58 01213 1:31.6 ハナ  7-8   33.0 130.2 15 472(+10) 01140
        11 4  6 2021110031 牡4 58 01179 1:31.7 クビ  13-13 32.6 34.1  7  460(-2)  01130
        9  5  5 2020100680 牡5 58 01171 1:31.7 クビ  2-2   33.6 67.2  12 520(+4)  01157
        17 6  8 2018100382 牡7 58 05473 1:31.7 アタマ 7-7  33.2 5.2   2  514(0)   01071
        5  7  3 2021105354 牝4 56 05339 1:31.9 3/4   4-4   33.5 7.5   3  478(0)   01133
        4  8  2 2022104772 牡3 57 00666 1:32.1 1.1/4 15-15 32.8 53.2  9  524(+4)  01105
        16 9  8 2020190006 牡5 58 05446 1:32.1 クビ  13-13 33.1 120.2 13 504(0)   05787
        7  10 4 2021105643 牝4 56 05626 1:32.2 クビ  10-12 33.3 45.4  8  492(-8)  01126
        1  11 1 2019106526 牡6 58 01180 1:32.2 クビ  1-1   34.5 58.9  11 476(-4)  01118
        14 12 7 2020102078 牡5 58 05585 1:32.4 1.1/4 10-10 33.7 19.3  5  486(+2)  01162
        8  13 4 2020101503 牡5 58 01093 1:32.5 クビ  16-15 33.1 240.9 17 496(+6)  01087
        2  14 1 2020103075 牡5 58 01163 1:32.6 1/2   7-8   34.0 121.9 14 510(-6)  01119
        13 15 7 2018104708 セ7 58 05203 1:32.7 クビ  18-17 33.0 192.3 16 478(-12) 01165
        10 16 5 2021107098 牝4 56 01174 1:32.8 1/2   4-4   34.5 20.2  6  492(+4)  01186
        18 17 8 2021104434 牝4 56 01102 1:33.0 1.1/2 17-17 33.3 278.2 18 430(+2)  01055
        12 18 6 2019100213 牡6 58 01126 1:33.2 1.1/4 10-10 34.5 53.5  10 484(+2)  01174
    """,
    KIMPAI_2010: """
        7  1  4 2004110096 セ6 57 05203 1:34.1 -     10-9  34.6 14.3  5  480(+12) 00201
        5  2  3 2005105204 牡5 57 00666 1:34.2 1/2   15-12 34.5 2.2   1  466(+2)  00357
        8  3  4 2004103345 牝6 53 01116 1:34.2 アタマ 7-6  35.0 69.8  13 476(-2)  01018
        10 4  5 2004100739 牡6 53 00700 1:34.2 クビ  12-9  34.6 33.3  9  540(+6)  01003
        12 5  6 2005105514 牡5 57 01018 1:34.3 クビ  2-2   35.9 5.4   2  518(-4)  01024
        6  6  3 2003100626 牝7 53 05223 1:34.4 3/4   10-12 34.8 49.3  11 478(+2)  00411
        1  7  1 2003105450 牡7 56 00894 1:34.4 ハナ  13-15 34.6 51.8  12 476(0)   01055
        9  8  5 2006106190 牡4 55 01026 1:34.5 3/4   4-6   35.5 34.6  10 500(0)   01099
        4  9  2 2002107509 牝8 52 01066 1:34.6 1/2   13-12 34.9 150.7 15 492(+9)  00216
        16 10 8 2003102873 牡7 57 01019 1:34.7 1/2   4-3   35.8 21.6  7  494(0)   00375
        3  11 2 2004103160 牝6 54 01037 1:34.7 アタマ 7-6  35.6 18.3  6  488(-2)  00343
        13 12 7 2003100496 牡7 57 01092 1:34.9 1     4-3   36.0 24.5  8  500(+12) 01062
        2  13 1 2006100130 牡4 53 01032 1:34.9 ハナ  7-9   35.6 12.7  4  474(-2)  01016
        14 14 7 2005105446 牡5 56 01115 1:35.0 3/4   3-3   36.2 8.2   3  502(+6)  01075
        15 15 8 2004104139 牡6 57 00682 1:36.5 9     1-1   38.3 75.7  14 518(-4)  01028
        11 取 6 2005103851 牝5 54 01088 -      -     -     -    ---   -  計不     01002
    """,
}

# Finishing order of the other real pages: horse no, placing as printed, margin ("-" = none).
ORDER = {
    DEAD_HEAT_3RD: """
        9 1 -  11 2 3/4  10 3 3/4  14 3 同着  8 5 ハナ  12 6 1  1 7 1.3/4  17 8 ハナ
        4 9 アタマ  2 10 アタマ  16 11 1.1/2  5 12 クビ  7 13 1/2  13 14 クビ  3 15 1.3/4
        6 16 ハナ  15 17 大
    """,
    JUMP: """
        6 1 -  3 2 7  1 3 2.1/2  2 4 3  4 5 2  7 6 5  11 7 6  5 8 クビ  9 9 大
        8 中 -  10 中 -
    """,
    SCRATCH_DEAD_HEAT_2ND: """
        13 1 -  1 2 クビ  7 2 同着  10 4 3/4  8 5 2.1/2  12 6 2  3 7 クビ  6 8 1.1/2
        11 9 2.1/2  4 10 2.1/2  2 11 1/2  9 12 クビ  14 13 大  5 取 -
    """,
    EXCLUDED: """
        10 1 -  4 2 2  11 3 クビ  6 4 3/4  3 5 1.1/4  2 6 1.1/2  1 7 1/2  9 8 1.1/2
        8 9 4  5 10 大  7 除 -
    """,
}

# Payouts per 100 yen as printed; ordered pools keep their order.
# fmt: off
DIVIDENDS = {
    MILE_CS_2025: [
        ("WIN", "15", 180), ("PLACE", "15", 110), ("PLACE", "6", 230), ("PLACE", "3", 1370),
        ("BRACKET_QUINELLA", "3,7", 340), ("QUINELLA", "6,15", 640),
        ("QUINELLA_PLACE", "6,15", 340), ("QUINELLA_PLACE", "3,15", 3930),
        ("QUINELLA_PLACE", "3,6", 9530),
        ("FORECAST", "15,6", 870), ("TRIO", "3,6,15", 23090), ("TIERCE", "15,6,3", 52470),
    ],
    KIMPAI_2010: [
        ("WIN", "7", 1430), ("PLACE", "7", 350), ("PLACE", "5", 130), ("PLACE", "8", 1090),
        ("BRACKET_QUINELLA", "3,4", 1400), ("QUINELLA", "5,7", 1380),
        ("QUINELLA_PLACE", "5,7", 630), ("QUINELLA_PLACE", "7,8", 7480),
        ("QUINELLA_PLACE", "5,8", 2490),
        ("FORECAST", "7,5", 4230), ("TRIO", "5,7,8", 31060), ("TIERCE", "7,5,8", 193260),
    ],
    DEAD_HEAT_3RD: [  # dead heat for 3rd: four places, five pairs, two trios and tierces
        ("WIN", "9", 2120),
        ("PLACE", "9", 290), ("PLACE", "11", 140), ("PLACE", "10", 530), ("PLACE", "14", 110),
        ("BRACKET_QUINELLA", "5,6", 2420), ("QUINELLA", "9,11", 4970),
        ("QUINELLA_PLACE", "9,11", 870), ("QUINELLA_PLACE", "9,10", 3260),
        ("QUINELLA_PLACE", "9,14", 580), ("QUINELLA_PLACE", "10,11", 1470),
        ("QUINELLA_PLACE", "11,14", 230),
        ("FORECAST", "9,11", 12810),
        ("TRIO", "9,10,11", 18620), ("TRIO", "9,11,14", 1920),
        ("TIERCE", "9,11,10", 146590), ("TIERCE", "9,11,14", 22070),
    ],
    JUMP: [
        ("WIN", "6", 240), ("PLACE", "6", 110), ("PLACE", "3", 860), ("PLACE", "1", 120),
        ("BRACKET_QUINELLA", "3,6", 6430), ("QUINELLA", "3,6", 7950),
        ("QUINELLA_PLACE", "3,6", 1910), ("QUINELLA_PLACE", "1,6", 160),
        ("QUINELLA_PLACE", "1,3", 2210),
        ("FORECAST", "6,3", 10440), ("TRIO", "1,3,6", 4360), ("TIERCE", "6,3,1", 28560),
    ],
    SCRATCH_DEAD_HEAT_2ND: [  # dead heat for 2nd: two brackets, quinellas, forecasts, tierces
        ("WIN", "13", 1930), ("PLACE", "13", 330), ("PLACE", "1", 170), ("PLACE", "7", 200),
        ("BRACKET_QUINELLA", "1,8", 1160), ("BRACKET_QUINELLA", "5,8", 1800),
        ("QUINELLA", "1,13", 1300), ("QUINELLA", "7,13", 1850),
        ("QUINELLA_PLACE", "1,13", 700), ("QUINELLA_PLACE", "7,13", 870),
        ("QUINELLA_PLACE", "1,7", 300),
        ("FORECAST", "13,1", 3630), ("FORECAST", "13,7", 5490),
        ("TRIO", "1,7,13", 3550),
        ("TIERCE", "13,1,7", 18880), ("TIERCE", "13,7,1", 23400),
    ],
    EXCLUDED: [
        ("WIN", "10", 200), ("PLACE", "10", 110), ("PLACE", "4", 270), ("PLACE", "11", 150),
        ("BRACKET_QUINELLA", "4,8", 2320), ("QUINELLA", "4,10", 1180),
        ("QUINELLA_PLACE", "4,10", 480), ("QUINELLA_PLACE", "10,11", 300),
        ("QUINELLA_PLACE", "4,11", 1120),
        ("FORECAST", "10,4", 2000), ("TRIO", "4,10,11", 2300), ("TIERCE", "10,4,11", 8850),
    ],
}
# fmt: on
DIVIDENDS[DEMOTED_DERIVED] = DIVIDENDS[EXCLUDED]
DIVIDENDS[CLASS_500MAN_DERIVED] = DIVIDENDS[KIMPAI_2010]

# Beaten lengths worked out by hand, so the decoders below cannot share a mistake with the code.
SPOT_LENGTHS = {
    MILE_CS_2025: {15: 0.0, 6: 1.75, 3: 1.80, 11: 2.05, 17: 2.40, 12: 10.65},
    KIMPAI_2010: {7: 0.0, 5: 0.5, 15: 14.55},
    DEAD_HEAT_3RD: {10: 1.5, 14: 1.5, 8: 1.55, 15: 18.85},
    JUMP: {3: 7.0, 1: 9.5, 9: 35.75},
    SCRATCH_DEAD_HEAT_2ND: {1: 0.25, 7: 0.25, 14: 23.0},
}

# Runners without a plain placing: horse no, placing as printed, code, scratched, win odds.
PLACING_CODES = {
    KIMPAI_2010: [(11, "取", "WV", True, None)],  # 取消: withdrawn before the race
    SCRATCH_DEAD_HEAT_2ND: [(5, "取", "WV", True, None)],
    EXCLUDED: [(7, "除", "WX", True, None)],  # 除外: excluded by the stewards
    JUMP: [(8, "中", "PU", False, 110.0), (10, "中", "PU", False, 6.0)],  # 中止: started
    DEMOTED_DERIVED: [(5, "失", "DISQ", False, 11.5), (7, "除", "WX", True, None)],  # 失格
}

RESULT_PAGES = [*RACES, SENTINEL]

_MARGIN = re.compile(r"(\d+)(?:\.(\d+)/(\d+))?|(\d+)/(\d+)")
_NAMED_MARGINS = {"ハナ": 0.05, "アタマ": 0.1, "クビ": 0.25, "大": 10.0, "大差": 10.0, "同着": 0.0}


def margin_lengths(raw: str) -> float:
    """ハナ 0.05, アタマ 0.1, クビ 0.25, '1.3/4' = 1 3/4, 大 10."""
    if raw in _NAMED_MARGINS:
        return _NAMED_MARGINS[raw]
    m = _MARGIN.fullmatch(raw)
    assert m, f"test table has an unknown margin {raw!r}"
    if m.group(4):
        return int(m.group(4)) / int(m.group(5))
    return int(m.group(1)) + (int(m.group(2)) / int(m.group(3)) if m.group(2) else 0.0)


def seconds(raw: str) -> float | None:
    if raw == "-":
        return None
    minutes, rest = raw.split(":")
    return int(minutes) * 60 + float(rest)


def body_weight(raw: str) -> tuple[int | None, int | None]:
    m = re.fullmatch(r"(\d+)\(([+-]?\d+)\)", raw)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def expected_runners(file_name: str) -> list[dict]:
    rows = []
    for line in RUNNERS[file_name].strip().splitlines():
        cells = line.split()
        no, placing, bracket, horse, sex_age, kg, jockey, time = cells[:8]
        margin, passing, last3f, odds, fav, weight, trainer = cells[8:]
        wt_kg, wt_change = body_weight(weight)
        rows.append(
            {
                "horse_no": int(no),
                "placing_raw": placing,
                "finish_pos": int(placing) if placing.isdigit() else None,
                "bracket": int(bracket),
                "horse_id": horse,
                "sex": SEX[sex_age[0]],
                "age": int(sex_age[1:]),
                "actual_wt_kg": float(kg),
                "jockey_code": jockey,
                "trainer_code": trainer,
                "finish_time_s": seconds(time),
                "margin": margin,
                "running_positions": [int(p) for p in passing.split("-")] if passing != "-" else [],
                "last_3f_s": float(last3f) if last3f != "-" else None,
                "win_odds": float(odds) if odds != "---" else None,
                "win_fav_rank": int(fav) if fav != "-" else None,
                "horse_wt_kg": wt_kg,
                "horse_wt_change_kg": wt_change,
            }
        )
    return rows


def finishing_order(file_name: str) -> list[tuple[int, str, str]]:
    if file_name in RUNNERS:
        return [(r["horse_no"], r["placing_raw"], r["margin"]) for r in expected_runners(file_name)]
    tokens = ORDER[file_name].split()
    return [(int(tokens[i]), tokens[i + 1], tokens[i + 2]) for i in range(0, len(tokens), 3)]


def cumulative_lengths(order: list[tuple[int, str, str]]) -> dict[int, float | None]:
    """Beaten lengths: the running sum of the printed margins; None without a place."""
    out: dict[int, float | None] = {}
    total = 0.0
    for horse_no, placing, margin in order:
        if not placing[0].isdigit():
            out[horse_no] = None
            continue
        if margin != "-":
            total += margin_lengths(margin)
        out[horse_no] = total
    return out


def printed_race_name(text: str) -> str:
    """The race name as the page's one <h1> prints it (with its edition, 第42回 ...)."""
    names = re.findall(r"<h1>([^<]*)</h1>", text)
    assert len(names) == 1 and names[0].strip(), names
    return names[0].strip()


def test_every_real_results_page_has_expectations(jra_dir):
    assert sorted(p.name for p in jra_dir.glob("results_*.html")) == sorted(RESULT_PAGES)


@pytest.mark.parametrize("name", RESULT_PAGES)
def test_real_results_pages(jra_dir, name):
    body = page_bytes(jra_dir, name)
    parsed = parse_file(jra_dir, name)
    if name == SENTINEL:
        assert parse_results.classify(body) == STATUS_EMPTY
        assert ADAPTER.classify("results", body) == STATUS_EMPTY
        assert all(not parsed.get(table) for table in TABLES), parsed
        return
    assert parse_results.classify(body) == STATUS_FETCHED
    want = RACES[name]
    assert set(TABLES) <= set(parsed), f"tables missing: {set(TABLES) - set(parsed)}"

    # meetings: the one meeting this race belongs to, venue from the race id via jra.yaml
    (meeting,) = parsed["meetings"]
    assert meeting["jurisdiction"] == "jra"
    assert meeting["meeting_id"] == want["meeting_id"]
    assert meeting["venue"] == want["venue"]
    assert str(meeting["race_date"])[:10] == want["race_date"]

    # races
    race = only_race(parsed)
    for column in (
        "race_id", "meeting_id", "race_no", "surface", "course_config", "distance_m", "going",
        "going_raw", "race_class", "weight_type", "post_time_local", "field_size",
    ):  # fmt: skip
        assert race[column] == want[column], column
    assert race["jurisdiction"] == "jra" and race["status"] == "run"
    assert race["source_race_key"] == race_key(name)
    assert race["race_name"] == printed_race_name(page_text(jra_dir, name))
    assert race["race_class_raw"]
    if name in CLASS_RAW_PART:
        assert CLASS_RAW_PART[name] in race["race_class_raw"]
    assert race["split_times_s"] == pytest.approx(want["split_times_s"], abs=1e-6)
    assert race["finish_time_s"] == pytest.approx(want["finish_time_s"], abs=1e-6)

    # runners: one row per declared runner, keyed by the netkeiba ids
    runners = parsed["runners"]
    assert len(runners) == want["n_runners"]
    assert sorted(r["horse_no"] for r in runners) == list(range(1, want["n_runners"] + 1))
    assert len({r["horse_id"] for r in runners}) == want["n_runners"]
    assert sum(not r["is_scratched"] for r in runners) == want["field_size"]
    for r in runners:
        assert r["jurisdiction"] == "jra" and r["race_id"] == want["race_id"]
        assert re.fullmatch(r"[0-9]{10}", r["horse_id"] or ""), r["horse_id"]
        assert r["draw"] == r["horse_no"]
        assert re.fullmatch(r"[0-9]{5}", r["jockey_code"] or ""), r["jockey_code"]
        assert re.fullmatch(r"[0-9]{5}", r["trainer_code"] or ""), r["trainer_code"]
        assert r["jockey_key"] == f"jockey:{r['jockey_code']}"
        assert r["trainer_key"] == f"trainer:{r['trainer_code']}"
        assert r["sex"] in ("H", "M", "G") and 2 <= r["age"] <= 12
        assert 1 <= r["bracket"] <= 8
        assert r["actual_wt_lb"] == pytest.approx(r["actual_wt_kg"] * LB_PER_KG, abs=0.5)
        if r["horse_wt_kg"] is not None:
            assert r["declared_horse_wt_lb"] == pytest.approx(r["horse_wt_kg"] * LB_PER_KG, abs=0.5)
        else:
            assert r["declared_horse_wt_lb"] is None
    winners = [r for r in runners if r["finish_pos"] == 1]
    assert len(winners) == 1
    assert winners[0]["finish_time_s"] == pytest.approx(want["finish_time_s"], abs=1e-6)

    # every column of the two fully tabulated pages, row by row
    got = by_horse_no(parsed)
    if name in RUNNERS:
        for exp in expected_runners(name):
            row = got[exp["horse_no"]]
            for column in (
                "horse_id", "bracket", "sex", "age", "actual_wt_kg", "jockey_code", "trainer_code",
                "finish_pos", "placing_raw", "running_positions", "win_fav_rank", "horse_wt_kg",
                "horse_wt_change_kg",
            ):  # fmt: skip
                assert row[column] == exp[column], (exp["horse_no"], column)
            for column in ("finish_time_s", "last_3f_s", "win_odds"):
                if exp[column] is None:
                    assert row[column] is None, (exp["horse_no"], column)
                else:
                    assert row[column] == pytest.approx(exp[column], abs=1e-6), (
                        exp["horse_no"],
                        column,
                    )
            assert row["is_scratched"] is (exp["finish_pos"] is None)

    # margins and placing codes: lbw_lengths is the running sum of the printed margins, the same
    # for both halves of a dead heat, null for runners without a place
    if name in RUNNERS or name in ORDER:
        order = finishing_order(name)
        expected = cumulative_lengths(order)
        assert set(got) == set(expected)
        shared = {p for p, n in Counter(pl for _, pl, _ in order if pl.isdigit()).items() if n > 1}
        for horse_no, placing, margin in order:
            row = got[horse_no]
            assert row["placing_raw"] == placing, horse_no
            if margin == "-":
                assert not row["lbw_raw"], horse_no
            else:
                assert row["lbw_raw"] == margin, horse_no
            if expected[horse_no] is None:
                assert row["lbw_lengths"] is None and row["finish_pos"] is None, horse_no
            else:
                assert row["lbw_lengths"] == pytest.approx(expected[horse_no], abs=1e-6), horse_no
                assert row["finish_pos"] == int(placing), horse_no
            assert row["dead_heat"] is (placing in shared), horse_no
            if placing.isdigit():
                assert row["placing_code"] == ("DH" if placing in shared else None), horse_no
                assert row["is_scratched"] is False, horse_no
    for horse_no, value in SPOT_LENGTHS.get(name, {}).items():
        assert got[horse_no]["lbw_lengths"] == pytest.approx(value, abs=1e-6), horse_no

    # runners without a plain placing; scratched runners never count in the field
    for horse_no, placing, code, scratched, odds in PLACING_CODES.get(name, []):
        row = got[horse_no]
        assert row["placing_raw"] == placing, horse_no
        assert row["placing_code"] == code and row["finish_pos"] is None, horse_no
        if code != "DISQ":  # a disqualified runner's printed time and margin are not pinned
            assert row["lbw_lengths"] is None and row["finish_time_s"] is None, horse_no
        assert row["is_scratched"] is scratched and row["dead_heat"] is False, horse_no
        assert row["win_odds"] == (pytest.approx(odds) if odds is not None else None), horse_no
    if name == DEMOTED_DERIVED:  # 降着: the runner keeps its finishing position
        demoted = got[6]
        assert demoted["finish_pos"] == 4 and "降" in demoted["placing_raw"]
        assert demoted["is_scratched"] is False and demoted["dead_heat"] is False

    # dividends: every pool and combination as printed, 8 pools
    dividends = parsed["dividends"]
    assert len(dividends) == want["n_dividends"]
    assert len({d["pool"] for d in dividends}) == 8
    assert all(d["race_id"] == want["race_id"] and d["jurisdiction"] == "jra" for d in dividends)
    assert Counter((d["pool"], d["combination"], d["dividend"]) for d in dividends) == Counter(
        (pool, combination, float(amount)) for pool, combination, amount in DIVIDENDS[name]
    )
    assert all(isinstance(d["dividend"], float) and d["is_refund"] is False for d in dividends)
    assert {d["pool"] for d in dividends} <= set(load_source("japan", "jra").pools)


def test_real_results_names_the_mile_cs_winner(jra_dir):
    """Mile CS 2025 winner #15 at 1.8 paying 180 on WIN; names as printed, codes are the keys."""
    parsed = parse_file(jra_dir, MILE_CS_2025)
    winner = by_horse_no(parsed)[15]
    assert winner["finish_pos"] == 1 and winner["win_odds"] == pytest.approx(1.8)
    assert winner["horse_name"] == "ジャンタルマンタル" and winner["horse_id"] == "2021107171"
    assert winner["jockey_name"] == "川田将雅" and winner["jockey_key"] == "jockey:01088"
    assert winner["trainer_name"] == "高野友和" and winner["trainer_key"] == "trainer:01117"
    win = [d for d in parsed["dividends"] if d["pool"] == "WIN"]
    assert [(d["combination"], d["dividend"]) for d in win] == [("15", 180.0)]


# Header text edited on the real markup: (base page, printed, replacement, column, expected).
CLASS_BASE_TEXT = "3歳以上1勝クラス"
CLASS_VARIANTS = [
    ("2歳新馬", None, "DEBUT"),
    ("2歳未勝利", None, "MAIDEN"),
    ("3歳以上1勝クラス", None, "C1"),
    ("3歳以上2勝クラス", None, "C2"),
    ("3歳以上2勝クラス", "ケ池特別(2勝)", "C2"),
    ("3歳以上3勝クラス", None, "C3"),
    ("3歳以上500万下", None, "C1"),  # pre-2019 names
    ("3歳以上1000万下", None, "C2"),
    ("3歳以上1600万下", None, "C3"),
    ("3歳以上オープン", "室町S(OP)", "OP"),
    ("3歳以上オープン", "オータムリーフS(L)", "L"),
    ("3歳以上オープン", "みやこS(GIII)", "G3"),
    ("3歳以上オープン", "デイリー杯2歳S(GII)", "G2"),
    ("3歳以上オープン", "チャンピオンズC(GI)", "G1"),
]
GOING_VARIANTS = [("良", "GOOD"), ("稍重", "GOOD TO YIELDING"), ("重", "SOFT"), ("不良", "HEAVY")]
COURSE_VARIANTS = [
    ("芝右 外1600m", ("turf", "outer", 1600)),
    ("芝右 内2000m", ("turf", "inner", 2000)),
    ("芝右1200m", ("turf", None, 1200)),
    ("芝直線1000m", ("turf", None, 1000)),
    ("ダ右1800m", ("dirt", None, 1800)),
    ("ダ左1400m", ("dirt", None, 1400)),
    ("障芝3170m", ("jump", None, 3170)),
]
WEIGHT_VARIANTS = [
    ("(馬齢)", "age"),
    ("(定量)", "set"),
    ("(別定)", "allowance"),
    ("(ハンデ)", "handicap"),
]


def with_class(text: str, class_text: str, race_name: str | None) -> str:
    assert text.count(CLASS_BASE_TEXT) == 3
    text = text.replace(CLASS_BASE_TEXT, class_text)
    if race_name:
        text = text.replace(f"<h1>\n{class_text}\n</h1>", f"<h1>\n{race_name}\n</h1>", 1)
        text = text.replace(f"<title>{class_text}｜", f"<title>{race_name}｜", 1)
        assert text.count(race_name) == 2
    return text


def replaced(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


HEADER_CASES = [
    *[
        pytest.param("class", c, n, ("race_class",), (want,), id=f"class-{n or c}")
        for c, n, want in CLASS_VARIANTS
    ],
    *[
        pytest.param(
            "going", (base, label, going_raw), None, ("going", "going_raw"), (going, going_raw),
            id=f"going-{label}-{going_raw}",
        )
        for base, label in ((MILE_CS_2025, "芝"), (EXCLUDED, "ダート"))
        for going_raw, going in GOING_VARIANTS
    ],
    *[
        pytest.param(
            "course", c, None, ("surface", "course_config", "distance_m"), want, id=f"course-{c}"
        )
        for c, want in COURSE_VARIANTS
    ],
    *[
        pytest.param("weight", w, None, ("weight_type",), (want,), id=f"weight-{w}")
        for w, want in WEIGHT_VARIANTS
    ],
]  # fmt: skip


@pytest.mark.parametrize(("kind", "printed", "race_name", "columns", "expected"), HEADER_CASES)
def test_real_results_header_variants(jra_dir, kind, printed, race_name, columns, expected):
    """Class, going, course and weight type read off the real header markup, with the printed text
    edited in place (HKTrack's derived cases)."""
    if kind == "class":
        base, text = EXCLUDED, with_class(page_text(jra_dir, EXCLUDED), printed, race_name)
    elif kind == "going":
        base, label, going_raw = printed
        was = {MILE_CS_2025: "芝 : 良", EXCLUDED: "ダート : 不良"}[base]
        text = replaced(page_text(jra_dir, base), was, f"{label} : {going_raw}")
    elif kind == "course":
        base = MILE_CS_2025
        text = replaced(page_text(jra_dir, base), "芝右 外1600m", printed)
    else:
        base = EXCLUDED
        text = replaced(page_text(jra_dir, base), "(定量)", printed)
    race = only_race(parse_results.parse(results_raw(text, race_key(base))))
    assert tuple(race[c] for c in columns) == expected
    if kind == "class":
        assert race["race_class_raw"]


# ================================================================ netkeiba pedigree card

PEDIGREE_2025 = "pedigree_card_202508040611.html"
PEDIGREE_2010 = "pedigree_card_201008010111.html"
BLOCKED = "blocked_cdn_error_page.html"
PEDIGREE_FETCHED = {
    PEDIGREE_2025: datetime(2026, 11, 18, 14, 32, 29, tzinfo=UTC),
    PEDIGREE_2010: datetime(2026, 11, 18, 14, 33, 1, tzinfo=UTC),
}

# horse no -> (horse_id, sire, dam, dam sire), read off the cards by hand
MILE_CS_ROWS = {
    1: ("2019106526", "ビッグアーサー", "ユキノマーメイド", "スペシャルウィーク"),
    2: ("2020103075", "ドゥラメンテ", "メモリアルライフ", "Reckless Abandon"),
    3: ("2021100953", "ドレフォン", "ウォーターピオニー", "ヴィクトワールピサ"),
    4: ("2022104772", "キズナ", "コナブリュワーズ", "キングカメハメハ"),
    5: ("2021105354", "ダイワメジャー", "アスコルティ", "Danehill Dancer"),
    6: ("2019104476", "キタサンブラック", "ナターレ", "クロフネ"),
    7: ("2021105643", "ハービンジャー", "チェッキーノ", "キングカメハメハ"),
    8: ("2020101503", "ビッグアーサー", "クェスタボルタ", "ノヴェリスト"),
    9: ("2020100680", "ディープブリランテ", "ショウナンカラット", "ブライアンズタイム"),
    10: ("2021107098", "シルバーステート", "ゴッドパイレーツ", "ベーカバド"),
    11: ("2021110031", "Farhh", "ローズトレイル", "Kingmambo"),
    12: ("2019100213", "アイルハヴアナザー", "コスモマーベラス", "フジキセキ"),
    13: ("2018104708", "ヴィクトワールピサ", "ノッテビアンカ", "Kendargent"),
    14: ("2020102078", "リアルスティール", "トウカイライフ", "トウカイテイオー"),
    15: ("2021107171", "パレスマリス", "インディアマントゥアナ", "Wilburn"),
    16: ("2020190006", "Massaat", "Icky Woo", "Mark of Esteem"),
    17: ("2018100382", "ルーラーシップ", "エターナルブーケ", "マンハッタンカフェ"),
    18: ("2021104434", "カリフォルニアクローム", "ワイドサファイア", "アグネスタキオン"),
}
KIMPAI_ROWS = {
    1: ("2003105450", "サクラバクシンオー", "テンザンキラリ", "ゴールデンフェザント"),
    2: ("2006100130", "クロフネ", "アラデヤ", "Machiavellian"),
    3: ("2004103160", "スペシャルウィーク", "クラウンピース", "Seattle Slew"),
    4: ("2002107509", "ミスズシャルダン", "サンレイククイン", "Cozzene"),
    5: ("2005105204", "マーベラスサンデー", "スケアヘッドライン", "パドスール"),
    6: ("2003100626", "ホワイトマズル", "ヤマニンザナドゥ", "トニービン"),
    7: ("2004110096", "シングスピール", "Dance Lively", "Kingmambo"),
    8: ("2004103345", "ダンスインザダーク", "レンII", "Bob Back"),
    9: ("2006106190", "バブルガムフェロー", "クワイエットアース", "Mazel Trick"),
    10: ("2004100739", "メジロライアン", "コウエイブルーム", "ウォーニング"),
    11: ("2005103851", "クロフネ", "ブラボーサンライズ", "サンデーサイレンス"),  # scratched
    12: ("2005105514", "ムタファーウエク", "ビンゴハナコ", "パークリージェント"),
    13: ("2003100496", "タマモクロス", "アンサーミー", "ジョリーズヘイロー"),
    14: ("2005105446", "タイキシャトル", "タイキメビウス", "ダンスオブライフ"),
    15: ("2004104139", "グラスワンダー", "チェリーラブ", "サクラユタカオー"),
    16: ("2003102873", "フジキセキ", "シュガーキャンディ", "トニービン"),
}
PEDIGREE_PAGES = {
    PEDIGREE_2025: ("202508040611", "jra:2025-11-23:KYO:11", MILE_CS_ROWS, MILE_CS_2025),
    PEDIGREE_2010: ("201008010111", "jra:2010-01-05:KYO:11", KIMPAI_ROWS, KIMPAI_2010),
}
NAME_COLUMNS = ("sire_name", "dam_name", "dam_sire_name")


def pedigree_raw(body: bytes | str, key: str, fetched: datetime | None) -> RawPage:
    data = body.encode("utf-8") if isinstance(body, str) else body
    url = f"https://race.netkeiba.com/race/shutuba_past.html?race_id={key}"
    return RawPage("jra", "pedigree_card", key, url, data, fetched_utc=fetched)


def pedigree_rows(body: bytes | str, key: str, fetched: datetime | None = None) -> list[dict]:
    fetched = fetched or PEDIGREE_FETCHED[PEDIGREE_2025]
    return ADAPTER.parse(pedigree_raw(body, key, fetched))["horse_pedigree"]


def norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


@pytest.mark.parametrize("name", list(PEDIGREE_PAGES))
def test_real_pedigree_cards(jra_dir, name):
    """One row per runner, keyed by the runner's own horse link (not the winners its past runs
    link, not the legend or training rows); the ids are the ones the results page of the same race
    gives each horse number."""
    key, race_id, expected, results_page = PEDIGREE_PAGES[name]
    fetched = PEDIGREE_FETCHED[name]
    rows = pedigree_rows(page_bytes(jra_dir, name), key, fetched)
    assert (
        rows
        == parse_pedigree_card.parse(pedigree_raw(page_bytes(jra_dir, name), key, fetched))[
            "horse_pedigree"
        ]
    ), "the adapter hands a pedigree_card page to parse_pedigree_card"

    assert len(rows) == len(expected)
    by_horse = {row["horse_id"]: row for row in rows}
    assert len(by_horse) == len(rows)
    assert [row["horse_id"] for row in rows] == [expected[n][0] for n in sorted(expected)]
    for horse_no, (horse_id, sire, dam, dam_sire) in expected.items():
        row = by_horse[horse_id]
        assert (row["sire_name"], row["dam_name"], row["dam_sire_name"]) == (sire, dam, dam_sire)
        for column in NAME_COLUMNS:
            assert row[f"{column}_norm"] == norm(row[column]), (horse_no, column)
        assert (row["sire_id"], row["dam_id"], row["dam_sire_id"]) == (None, None, None), horse_no
        assert row["source_race_id"] == race_id, horse_no
        assert row["scraped_utc"] == fetched, horse_no

    results = by_horse_no(parse_file(jra_dir, results_page))
    assert {no: r["horse_id"] for no, r in results.items()} == {
        no: ids[0] for no, ids in expected.items()
    }


MASSAAT = '<div class="Horse01 fc">Massaat</div>'
SPECIAL_WEEK = '<div class="Horse04">(スペシャルウィーク)</div>'


@pytest.mark.parametrize(
    ("marker", "edit", "horse_id", "column", "printed", "normalised"),
    [
        (MASSAAT, '<div class="Horse01 fc">Ｍａｓｓａａｔ</div>', "2020190006", "sire_name",
         "Ｍａｓｓａａｔ", "massaat"),  # full-width Latin
        (MASSAAT, '<div class="Horse01 fc">ＭＡＳＳＡＡＴ</div>', "2020190006", "sire_name",
         "ＭＡＳＳＡＡＴ", "massaat"),
        (MASSAAT, '<div class="Horse01 fc">ﾏｻｰﾄ</div>', "2020190006", "sire_name", "ﾏｻｰﾄ",
         "マサート"),  # half-width katakana
        (SPECIAL_WEEK, '<div class="Horse04"></div>', "2019106526", "dam_sire_name", None, None),
    ],
    ids=["full-width", "full-width-upper", "half-width-kana", "no-dam-sire"],
)  # fmt: skip
def test_real_pedigree_card_names_as_printed(
    jra_dir, marker, edit, horse_id, column, printed, normalised
):
    """Derived from the 2025 card: one ancestor printed another way, or not at all (None)."""
    body = replaced(page_text(jra_dir, PEDIGREE_2025), marker, edit)
    rows = pedigree_rows(body, "202508040611")
    assert len(rows) == 18
    row = next(r for r in rows if r["horse_id"] == horse_id)
    assert (row[column], row[f"{column}_norm"]) == (printed, normalised)
    untouched = MILE_CS_ROWS[16 if horse_id == "2020190006" else 1]
    for other, value in zip(NAME_COLUMNS, untouched[1:], strict=True):
        if other != column:
            assert (row[other], row[f"{other}_norm"]) == (value, norm(value)), other


# ================================================================ calendar, race lists, classify

TODAY = date(2026, 11, 18)
CALENDAR = "calendar_2025-11.html"
CALENDAR_NO_MEETINGS = "calendar_2025-11_no_meetings_derived.html"
RACELIST = "racelist_2025-11-23.html"
RACELIST_NO_RACES = "racelist_2030-11-23_no_races.html"
# Race days on the November 2025 calendar (11 days, the Monday holiday 24th included).
NOV_2025_RACE_DAYS = [1, 2, 8, 9, 15, 16, 22, 23, 24, 29, 30]
# JRA races on 2025-11-23: Tokyo (05) 5回5日 R1-12 and Kyoto (08) 4回6日 R1-12.
JRA_IDS_2025_11_23 = {f"2025050505{r:02d}" for r in range(1, 13)} | {
    f"2025080406{r:02d}" for r in range(1, 13)
}
# Other ids on the same page: local racing (NAR venues 46/54/55/65) and overseas races.
OTHER_IDS_2025_11_23 = {
    "202546112301", "202546112302", "202554112301", "202554112302",
    "202555112301", "202555112302", "202565112301", "202565112302",
    "2025C7a00205", "2025F4a01f07", "2025H1a01e01",
}  # fmt: skip


def adapter_on(today: date):
    return type(ADAPTER)(today=lambda: today)


def raw(jra_dir: Path, name: str, page_type: str, key: str, url: str = "") -> RawPage:
    body = page_bytes(jra_dir, name)
    return RawPage("jra", page_type, key, url or f"https://netkeiba.test/{page_type}/{key}", body)


def racelist_raw(jra_dir: Path, name: str = RACELIST, day: date = date(2025, 11, 23)) -> RawPage:
    return raw(jra_dir, name, "racelist", day.isoformat(), f"{DB}/race/list/{day:%Y%m%d}/")


def results_tasks(adapter, page: RawPage):
    return adapter.followup_tasks(adapter.parse(page))


def test_real_calendar_and_race_lists(jra_dir):
    """The chain on the real pages: the November 2025 calendar -> one race-list task per race day
    up to today -> one results task per JRA race on the list of 23 Nov 2025, nothing else."""
    adapter = adapter_on(TODAY)
    calendar = raw(jra_dir, CALENDAR, "fixture", "2025-11")
    lists = adapter.meeting_tasks([calendar])
    days = [date(2025, 11, d) for d in NOV_2025_RACE_DAYS]
    assert sorted(t.key for t in lists) == [d.isoformat() for d in days]
    for task in lists:
        day = date.fromisoformat(task.key)
        assert (task.source, task.page_type, task.year) == ("jra", "racelist", 2025)
        assert task.url == f"{DB}/race/list/{day:%Y%m%d}/"
        assert not task.mutable  # a list for a day a year ago is final
    assert adapter.meeting_tasks([raw(jra_dir, CALENDAR_NO_MEETINGS, "fixture", "2025-11")]) == []

    # race days after today have no results yet; today itself is included
    for today, expected in [
        (date(2025, 11, 16), [1, 2, 8, 9, 15, 16]),
        (date(2025, 11, 21), [1, 2, 8, 9, 15, 16]),
        (date(2025, 11, 23), [1, 2, 8, 9, 15, 16, 22, 23]),
        (date(2025, 10, 31), []),
    ]:
        got = adapter_on(today).meeting_tasks([calendar])
        assert sorted(t.key for t in got) == [date(2025, 11, d).isoformat() for d in expected]
        assert [t.mutable for t in got if t.key == today.isoformat()] in ([], [True])

    # the race list of 23 Nov: the 24 JRA races, not the local (NAR) or overseas ones
    races = results_tasks(adapter, racelist_raw(jra_dir))
    keys = [t.key for t in races]
    assert sorted(keys) == sorted(JRA_IDS_2025_11_23) and len(keys) == len(set(keys))
    assert not set(keys) & OTHER_IDS_2025_11_23
    for task in races:
        assert (task.source, task.page_type, task.year) == ("jra", "results", 2025)
        assert task.url == f"{DB}/race/{task.key}/"
        assert not task.mutable
        assert task.priority // 10 == date(2025, 11, 23).toordinal()
    # the no-races list leads nowhere
    empty = racelist_raw(jra_dir, RACELIST_NO_RACES, date(2030, 11, 23))
    assert results_tasks(adapter_on(date(2030, 11, 24)), empty) == []

    # newest first, complete by date: the 23rd's races sit between the lists of the 22nd and 24th,
    # and every calendar month outranks them all
    by_key = {t.key: t for t in lists}
    assert by_key["2025-11-22"].priority < min(t.priority for t in races)
    assert max(t.priority for t in races) < by_key["2025-11-24"].priority
    months = adapter.fixture_tasks(date(2010, 1, 1), TODAY)
    assert min(t.priority for t in months) > max(t.priority for t in lists)
    for task in months:
        query = parse_qs(urlsplit(task.url).query)
        assert (int(query["year"][0]), int(query["month"][0])) == tuple(
            int(p) for p in task.key.split("-")
        )


def crafted_racelist(jra_dir: Path, day: date, race_ids: list[str]) -> RawPage:
    """A race list for `day` built on the real no-races page (same markup as the real list),
    listing `race_ids` under one Kyoto heading."""
    text = page_text(jra_dir, RACELIST_NO_RACES)
    text = text.replace("2030年11月23日", f"{day.year}年{day.month}月{day.day}日")
    items = "</li><li>\n".join(
        f'<dl class="race_top_data_info fc">\n<dt>\n{n}R\n</dt>\n<dd>\n'
        f'<a href="https://db.netkeiba.com/race/{race_id}/" title="race">race</a><br />\n'
        f"<div>\n芝1600m<br />\n</div>\n</dd>\n</dl>\n"
        for n, race_id in enumerate(race_ids, start=1)
    )
    block = (
        '<h3>\n中央\n</h3>\n<div class="race_list fc">\n<dl class="race_top_hold_list">\n'
        '<dt class="race_top_hold_data">\n<p>\n京都\n</p>\n</dt>\n<dd>\n<ul>\n<li>\n'
        f"{items}</li>\n</ul>\n</dd>\n</dl>\n</div>\n"
    )
    marker = '<div class="race_kaisai_info">'
    assert marker in text
    text = text.replace(marker, marker + "\n" + block, 1)
    url = f"{DB}/race/list/{day:%Y%m%d}/"
    return RawPage("jra", "racelist", day.isoformat(), url, text.encode("utf-8"))


def test_real_race_list_markup_drops_malformed_ids(jra_dir):
    """Only 12 ASCII digits with a JRA venue part (characters 5-6: 01..10) become a URL and a
    cache file name."""
    adapter = adapter_on(TODAY)
    kept = ["202501010101", "202510020312", "202506050708", "202509030411"]
    dropped = [
        "202500010101", "202511010101", "202530112301", "202599010101",  # venues 00, 11, NAR, 99
        "20250801011", "2025080101011",  # 11 and 13 digits
        "2025O8010101", "２０２５０８０１０１０２",  # a letter O, full-width digits
        "202508010103x", "2025FPa00f02",  # trailing junk, overseas
    ]  # fmt: skip
    tasks = results_tasks(adapter, crafted_racelist(jra_dir, date(2025, 11, 23), kept + dropped))
    assert sorted(t.key for t in tasks) == sorted(kept)
    assert all(t.url == f"{DB}/race/{t.key}/" for t in tasks)
    assert results_tasks(adapter, crafted_racelist(jra_dir, date(2025, 11, 25), dropped)) == []


def cut_before(jra_dir: Path, name: str, marker: str, back: int = 0) -> bytes:
    body = page_bytes(jra_dir, name)
    return body[: body.index(marker.encode("utf-8")) - back]


CLASSIFY = [
    # results: the result table decides; the HTTP-200 sentinel for an unknown id is empty
    ("results", MILE_CS_2025, STATUS_FETCHED),
    ("results", KIMPAI_2010, STATUS_FETCHED),
    ("results", JUMP, STATUS_FETCHED),
    ("results", SENTINEL, STATUS_EMPTY),
    ("results", BLOCKED, STATUS_BLOCKED),
    ("results", CALENDAR, STATUS_BLOCKED),
    ("results", RACELIST, STATUS_BLOCKED),
    # race list: a list with race links is content, the real no-races page is empty
    ("racelist", RACELIST, STATUS_FETCHED),
    ("racelist", RACELIST_NO_RACES, STATUS_EMPTY),
    ("racelist", BLOCKED, STATUS_BLOCKED),
    ("racelist", MILE_CS_2025, STATUS_BLOCKED),  # has race links, but is not a race list
    ("racelist", SENTINEL, STATUS_BLOCKED),
    ("racelist", CALENDAR, STATUS_BLOCKED),
    # fixture: a calendar with race-day links is content, a calendar without any is empty
    ("fixture", CALENDAR, STATUS_FETCHED),
    ("fixture", CALENDAR_NO_MEETINGS, STATUS_EMPTY),
    ("fixture", BLOCKED, STATUS_BLOCKED),
    ("fixture", RACELIST, STATUS_BLOCKED),
    ("fixture", KIMPAI_2010, STATUS_BLOCKED),
    # pedigree card: a card with runner rows; anything else is the site failing
    ("pedigree_card", PEDIGREE_2025, STATUS_FETCHED),
    ("pedigree_card", PEDIGREE_2010, STATUS_FETCHED),
    ("pedigree_card", BLOCKED, STATUS_BLOCKED),
    ("pedigree_card", MILE_CS_2025, STATUS_BLOCKED),
    ("pedigree_card", SENTINEL, STATUS_BLOCKED),
]


@pytest.mark.parametrize(("page_type", "name", "expected"), CLASSIFY)
def test_real_pages_classify_by_content(jra_dir, page_type, name, expected):
    """By content, never by HTTP status: netkeiba answers 200 whatever it serves."""
    assert ADAPTER.classify(page_type, page_bytes(jra_dir, name)) == expected


@pytest.mark.parametrize(
    ("page_type", "name", "marker", "back"),
    [
        ("results", MILE_CS_2025, "race_table_01", 40),
        ("results", KIMPAI_2010, "race_table_01", 40),
        ("pedigree_card", PEDIGREE_2025, '<div class="Shutuba_HorseList">', 0),
        ("pedigree_card", PEDIGREE_2010, '<div class="Shutuba_HorseList">', 0),
    ],
)
def test_real_pages_cut_off_are_blocked(jra_dir, page_type, name, marker, back):
    """A page cut off before its table (a dropped connection) is not content."""
    cut = cut_before(jra_dir, name, marker, back)
    assert b"1970" not in cut
    assert ADAPTER.classify(page_type, cut) == STATUS_BLOCKED


# ================================================================ JRA official race card

CARD = "official_card_2026-11-15_KYO_R10.html"
CARD_URL = "https://www.jra.go.jp/JRADB/accessD.html"
CARD_KEY = "pw01dde1008202605041020261115"
CARD_RACE_ID = "jra:2026-11-15:KYO:10"
CARD_SCRAPED = datetime(2026, 11, 15, 5, 0, tzinfo=UTC)

# horse_no, bracket, horse_id, name, sex, age, carried kg, body kg, body change kg, jockey code,
# trainer code, sire, dam
# fmt: off
CARD_RUNNERS = [
    (1, 1, "2022102394", "マーウォルス", "H", 4, 58.0, 472, -2, "01102", "01181",
     "アドマイヤマーズ", "ウェイクミーアップ"),
    (2, 2, "2021107278", "ユーヴェリア", "M", 5, 56.0, 450, 1, "01124", "01069",
     "オルフェーヴル", "ミラクルベリー"),
    (3, 3, "2016100126", "ダディーズトリップ", "H", 10, 58.0, 474, -8, "00660", "01120",
     "ドリームジャーニー", "ヴードゥーレディ"),
    (4, 4, "2022103186", "テーオーライマン", "H", 4, 58.0, 458, -6, "01126", "01018",
     "ダノンバラード", "テンキセキ"),
    (5, 5, "2021105637", "テンエースワン", "H", 5, 58.0, 476, 2, "00732", "01058",
     "ワールドエース", "ダイワオンディーヌ"),
    (6, 6, "2022105987", "アートレスマインド", "H", 4, 58.0, None, None, "05473", "01170",
     "マインドユアビスケッツ", "リチュアルローズ"),
    (7, 7, "2023103532", "シーミハットク", "H", 3, 56.0, 494, 4, "01213", "01158",
     "オルフェーヴル", "ブルーミンバー"),
    (8, 7, "2023103176", "セイントホース", "H", 3, 56.0, 484, 2, "01144", "00431",
     "ミスターメロディ", "ヤマカツオーキッド"),
    (9, 8, "2022105075", "タイセイドラード", "H", 4, 58.0, 542, -6, "01171", "01157",
     "American Pharoah", "ミスベジル"),
    (10, 8, "2021104295", "トモジャザーン", "H", 5, 58.0, 490, 2, "01015", "01032",
     "マジェスティックウォリアー", "トモジャオール"),
]
# fmt: on
CARD_SCRATCHED_ID = "2022105987"  # No 6: the number cell and the odds both say 取消
BLINKERS = (
    '<span class="horse_icon blinker"><img src="/JRADB/img/kigo/icon_blinker.png" '
    'alt="ブリンカー着用" /></span>'
)
WEIGHT_CELL = re.compile(
    r'<div class="cell weight">[0-9]+kg<span class="transition">[^<]*</span></div>'
)


def lb(kg: float) -> int:
    return round(kg * LB_PER_KG)


def card_text(jra_dir: Path) -> str:
    return page_text(jra_dir, CARD)


def parse_card(page: str) -> dict[str, list[dict]]:
    """What JRADB sends is Shift_JIS (cp932): the saved UTF-8 page is re-encoded first."""
    body = page.encode("cp932")
    return oc.parse(RawPage("jra", "racecard", CARD_KEY, CARD_URL, body, fetched_utc=CARD_SCRAPED))


def by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["horse_id"]: row for row in rows}


def edit_runner(page: str, horse_name: str, edit) -> str:
    rows = [
        m.group(0)
        for m in re.finditer(r"<tr\b.*?</tr>", page, flags=re.S)
        if f">{horse_name}</a>" in m.group(0)
    ]
    assert len(rows) == 1, horse_name
    changed = edit(rows[0])
    assert changed != rows[0]
    return page.replace(rows[0], changed)


def test_real_official_card_kyoto_r10(jra_dir):
    """The ten declared runners (not the past-winners table, not the past-run cells), in the
    units and codes of the results parser; going and course letter are not printed: null."""
    parsed = parse_card(card_text(jra_dir))
    runners = parsed["racecard_runners"]
    assert len(runners) == 10
    rows = by_id(runners)
    assert set(rows) == {r[2] for r in CARD_RUNNERS}
    assert [r["horse_no"] for r in runners] == list(range(1, 11))
    for (
        no,
        bracket,
        horse_id,
        name,
        sex,
        age,
        kg,
        body,
        change,
        jockey,
        trainer,
        sire,
        dam,
    ) in CARD_RUNNERS:
        row = rows[horse_id]
        scratched = horse_id == CARD_SCRATCHED_ID
        assert row["jurisdiction"] == "jra" and row["race_id"] == CARD_RACE_ID
        assert row["id_match"] == "exact"
        assert row["horse_name"] == name and row["bracket"] == bracket
        assert row["horse_no"] == no, "a scratched runner keeps its number, though 取消 is printed"
        if not scratched:
            assert row["draw"] == no
        assert row["is_scratched"] is scratched
        assert (row["sex"], row["age"]) == (sex, age)
        assert row["actual_wt_kg"] == kg and row["actual_wt_lb"] == lb(kg)
        assert not row.get("allowance_kg") and not row.get("allowance_lb"), "no claim here"
        assert row["horse_wt_kg"] == body and row["horse_wt_change_kg"] == change
        assert row["declared_horse_wt_lb"] == (lb(body) if body is not None else None)
        assert (row["jockey_code"], row["jockey_key"]) == (jockey, f"jockey:{jockey}")
        assert (row["trainer_code"], row["trainer_key"]) == (trainer, f"trainer:{trainer}")
        assert (row["sire"], row["dam"]) == (sire, dam)
        assert row["card_scraped_utc"] == CARD_SCRAPED

    (race,) = parsed["races"]
    assert race["race_id"] == CARD_RACE_ID and race["meeting_id"] == "jra:2026-11-15:KYO"
    assert race["race_no"] == 10 and race["race_name"] == "花背特別"
    assert (race["surface"], race["distance_m"], race["course_config"]) == ("dirt", 1800, None)
    assert race["race_class"] == "C2" and race["weight_type"] == "set"
    assert race["field_size"] == 9, "ten declared, one scratched"
    post = datetime.fromisoformat(race["post_time_local"])
    assert post.utcoffset() == timedelta(hours=9), "a full Tokyo date-time, not '15:01'"
    assert post.astimezone(UTC) == datetime(2026, 11, 15, 6, 1, tzinfo=UTC)
    assert race["going"] is None, "the card prints no going"
    assert race["course_letter"] is None, "the card prints no A-D course letter"
    assert (race["kai"], race["nichi"]) == (5, 4), "5回京都4日"


def relabel(meeting: str):
    def edit(page: str) -> str:
        out = page.replace("5回京都4日", meeting)
        assert out != page
        return out

    return edit


@pytest.mark.parametrize(
    ("edit", "kai", "nichi"),
    [(lambda p: p, 5, 4), (relabel("4回京都10日"), 4, 10), (relabel("1回京都1日"), 1, 1)],
    ids=["as-fetched", "two-digit-day", "first-day"],
)
def test_real_official_card_kai_and_nichi(jra_dir, edit, kai, nichi):
    """The meeting's 回 and 日 off the date line the card prints; the race stays Kyoto R10."""
    (race,) = parse_card(edit(card_text(jra_dir)))["races"]
    assert race["race_id"] == CARD_RACE_ID
    assert (race["kai"], race["nichi"]) == (kai, nichi)
    assert isinstance(race["kai"], int) and isinstance(race["nichi"], int)


@pytest.mark.parametrize(
    ("detail", "surface", "course_config"),
    [
        ("（ダート・右）", "dirt", None),  # as printed
        ("（芝・右 外）", "turf", "outer"),  # Kyoto outer course
        ("（芝・右）", "turf", None),  # inner course: JRA prints no 内
        ("（芝・右 内）", "turf", None),  # an explicit inner course is null too, as in history
    ],
    ids=["dirt", "turf-outer", "turf", "turf-inner"],
)
def test_real_official_card_course_detail(jra_dir, detail, surface, course_config):
    page = card_text(jra_dir).replace(
        '<span class="detail">（ダート・右）</span>', f'<span class="detail">{detail}</span>'
    )
    (race,) = parse_card(page)["races"]
    assert (race["surface"], race["course_config"], race["distance_m"]) == (
        surface, course_config, 1800,
    )  # fmt: skip


@pytest.mark.parametrize(
    ("mark", "title", "printed_kg", "claim_kg"),
    [("☆", "1kg減量", 55.0, 1.0), ("◇", "2kg減量", 54.0, 2.0), ("▲", "3kg減量", 53.0, 3.0)],
)
def test_real_official_card_apprentice_allowance(jra_dir, mark, title, printed_kg, claim_kg):
    """Derived: horse 7's rider given a claim, marked as on JRA's pages, the printed 負担重量
    lowered by it. An old run's ★ in the past-run cells is not this race's claim."""

    def claim(row: str) -> str:
        span = f'<span title="{title}" class="mark jockey">{mark}</span>'
        row = row.replace('<p class="jockey">', f'<p class="jockey">{span}')
        return re.sub(
            r'(<p class="weight">\s*)56\.0(<span>kg</span>)', rf"\g<1>{printed_kg:.1f}\2", row
        )

    rows = by_id(
        parse_card(edit_runner(card_text(jra_dir), "シーミハットク", claim))["racecard_runners"]
    )
    seven = rows["2023103532"]
    assert seven["allowance_kg"] == claim_kg and seven["allowance_lb"] == lb(claim_kg)
    assert seven["actual_wt_kg"] == printed_kg and seven["actual_wt_lb"] == lb(printed_kg)
    assert seven["jockey_key"] == "jockey:01213"
    for horse_id, row in rows.items():
        if horse_id != "2023103532":
            assert not row.get("allowance_kg"), row["horse_name"]


def move_blinkers_to_horse_2(page: str) -> str:
    page = edit_runner(page, "テンエースワン", lambda row: row.replace(BLINKERS, ""))
    return edit_runner(
        page, "ユーヴェリア",
        lambda row: row.replace('<td class="num">2</td>', f'<td class="num">2{BLINKERS}</td>'),
    )  # fmt: skip


def trainer_in_miho(page: str) -> str:
    return edit_runner(page, "ダディーズトリップ", lambda row: row.replace("(栗東)", "(美浦)"))


@pytest.mark.parametrize(
    ("derive", "blinkered", "miho"),
    [
        pytest.param(lambda page: page, {"2021105637"}, set(), id="as-fetched"),
        pytest.param(move_blinkers_to_horse_2, {"2021107278"}, set(), id="blinkers-on-2"),
        pytest.param(trainer_in_miho, {"2021105637"}, {"2016100126"}, id="trainer-3-in-miho"),
    ],
)
def test_real_official_card_blinkers_and_trainer_centre(jra_dir, derive, blinkered, miho):
    """Blinkers -> gear_raw "B" (else null); 所属 栗東 -> "ritto", 美浦 -> "miho"."""
    rows = by_id(parse_card(derive(card_text(jra_dir)))["racecard_runners"])
    assert {h for h, row in rows.items() if row["gear_raw"] is not None} == blinkered
    assert all(rows[h]["gear_raw"] == "B" for h in blinkered)
    assert {h: row["trainer_centre"] for h, row in rows.items()} == {
        h: "miho" if h in miho else "ritto" for h in rows
    }


@pytest.mark.parametrize(
    ("also", "number"), [("マーウォルス", 1), ("トモジャザーン", 10)], ids=["first", "last"]
)
def test_real_official_card_scratched_runner_keeps_its_number(jra_dir, also, number):
    """Derived: the first or the last runner scratched as well as No 6, its row made to look like
    No 6's real row (number, odds and body weight cells)."""

    def scratch(row: str) -> str:
        row = row.replace(
            f'<td class="num">{number}</td>', '<td class="num"><span class="cap">取消</span></td>'
        )
        row = re.sub(
            r'<div class="odds_line">.*?</div>',
            '<div class="odds_line"><span class="num"><strong>取消</strong></span></div>',
            row, count=1, flags=re.S,
        )  # fmt: skip
        return WEIGHT_CELL.sub("", row)

    rows = parse_card(edit_runner(card_text(jra_dir), also, scratch))["racecard_runners"]
    assert {row["horse_no"] for row in rows if row["is_scratched"]} == {6, number}
    assert sorted(row["horse_no"] for row in rows) == list(range(1, 11))
    (race,) = parse_card(edit_runner(card_text(jra_dir), also, scratch))["races"]
    assert race["field_size"] == 8


def test_real_official_card_before_the_weigh_in(jra_dir):
    """The card before the weigh-in (about 65 min before the off): body weights empty, null."""
    page = WEIGHT_CELL.sub("", card_text(jra_dir))
    assert page != card_text(jra_dir) and 'kg<span class="transition">' not in page
    rows = by_id(parse_card(page)["racecard_runners"])
    assert set(rows) == {r[2] for r in CARD_RUNNERS}
    for row in rows.values():
        assert row["horse_wt_kg"] is None and row["horse_wt_change_kg"] is None
        assert row["declared_horse_wt_lb"] is None


# ================================================================ JRDB public samples

# The public sample files' families (their names on jrdb.com's public sample page).
JRDB_FAMILIES = ("BAC", "CHA", "CYB", "JOA", "KAB", "KKA", "KYI", "SED", "SKB", "TYB", "UKC")
RACE_KEY_PARTS = ("venue_code", "year", "kai", "nichi", "race_no")
TRIM = " " + chr(0x3000)  # half- and full-width spaces
CENTURY_PIVOT = 70  # two-digit years 70-99 are 19xx, 00-69 are 20xx (the join's contract)


def local_specs() -> Path:
    """The spec tables of the local data directory (licensed: never in the repository). Called in
    each test body, so a missing table fails the test rather than erroring its setup."""
    specs = parse.specs_dir()
    missing = [f for f in JRDB_FAMILIES if not (specs / f"{f.lower()}.csv").is_file()]
    assert not missing, (
        f"no JRDB spec table for {missing} under {specs}: copy the 11 spec CSVs there"
    )
    return specs


def sample_path(jrdb_dir: Path, family: str) -> Path:
    found = sorted(jrdb_dir.glob(f"{family}[0-9][0-9][0-9][0-9][0-9][0-9].txt"))
    assert len(found) == 1, (family, found)
    return found[0]


def records_of(data: bytes) -> list[bytes]:
    """Data bytes of each record: split on LF, drop one trailing CR, skip empty lines."""
    return [
        line[:-1] if line.endswith(b"\r") else line
        for line in data.split(b"\n")
        if line.strip(b"\r")
    ]


def spec_rows(specs: Path, family: str) -> list[tuple[str, int, int, str]]:
    """(name, start, length, kind) of each field, read straight from the spec csv."""
    text = (specs / f"{family.lower()}.csv").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    return [
        (row["name"], int(row["start"]), int(row["length"]), row["kind"])
        for row in csv.DictReader(io.StringIO("\n".join(lines)))
    ]


def decode(raw: bytes, kind: str):
    if kind in ("text", "code"):
        return raw.decode("cp932").strip(TRIM) or None
    if kind == "num":
        text = raw.decode("ascii").replace(" ", "")
        return math.nan if text in ("", "+", "-") else float(text)
    if kind == "date":
        text = raw.decode("ascii").strip()
        if len(text) != 8 or not text.isdigit():
            return None
        try:
            return date(int(text[:4]), int(text[4:6]), int(text[6:]))
        except ValueError:
            return None
    raise AssertionError(kind)


def is_missing(value) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def same(got, want) -> bool:
    if is_missing(want):
        return is_missing(got)
    if isinstance(want, float):
        return isinstance(got, float | int) and not is_missing(got) and float(got) == want
    if isinstance(want, date):
        return isinstance(got, date) and got == want
    return isinstance(got, str) and got == want


def key_of(fields: list[tuple[str, int, int, str]]) -> tuple[str, int] | None:
    names = tuple(f[0] for f in fields[:5])
    if names == RACE_KEY_PARTS:
        return "race_key", sum(f[2] for f in fields[:5])
    if names[:4] == RACE_KEY_PARTS[:4]:
        return "meeting_key", sum(f[2] for f in fields[:4])
    return None


def full_year(two_digits: str) -> str:
    return ("19" if int(two_digits) >= CENTURY_PIVOT else "20") + two_digits


def test_jrdb_public_samples_parse_to_one_row_per_record(jrdb_dir):
    """Every public sample: one row per record in file order, the key column equal to the
    record's own leading bytes, and every field equal to the test's own decoding of its bytes
    (sliced by the spec table), on every record. Text is cp932: the owner names of two samples
    need its extensions, and none may come back mangled."""
    jrdb_specs = local_specs()
    problems: dict[str, object] = {}
    cp932_only = 0
    for family in JRDB_FAMILIES:
        path = sample_path(jrdb_dir, family)
        data = path.read_bytes()
        records = records_of(data)
        fields = spec_rows(jrdb_specs, family)
        width = sum(f[2] for f in fields) - 2
        assert records and {len(r) for r in records} == {width}, family

        frame = parse.parse_file(path)
        assert frame.equals(parse.parse_bytes(family, data)), family
        key = key_of(fields)
        want_columns = ([key[0]] if key else []) + [f[0] for f in fields if f[3] != "skip"]
        assert list(frame.columns) == want_columns, family
        assert len(frame) == len(records), family
        assert list(frame.index) == list(range(len(records))), family
        if key is not None:
            want = [r[: key[1]].decode("ascii") for r in records]
            if list(frame[key[0]]) != want:
                problems[f"{family}.{key[0]}"] = list(frame[key[0]])[:3]
        for name, start, length, kind in fields:
            if kind == "skip":
                continue
            column = frame[name].tolist()
            bad = [
                (i, column[i], expected)
                for i, record in enumerate(records)
                if not same(
                    column[i], expected := decode(record[start - 1 : start - 1 + length], kind)
                )
            ]
            if bad:
                problems[f"{family}.{name}"] = bad[:2]
            if kind == "text":
                for value in column:
                    if isinstance(value, str):
                        assert chr(0xFFFD) not in value, (family, name, value)
                        try:
                            value.encode("shift_jis")
                        except UnicodeEncodeError:
                            cp932_only += 1
    assert problems == {}, problems
    assert cp932_only >= 2, "the samples no longer exercise cp932's extensions"


def test_jrdb_century_rule_holds_on_the_ukc_sample(jrdb_dir):
    """The horse master prints the birth date next to the 8-digit id: every converted id starts
    with the horse's own birth year, on both sides of the century."""
    local_specs()
    frame = parse.parse_file(sample_path(jrdb_dir, "UKC"))
    assert len(frame) > 0
    pairs = list(zip(frame["jrdb_horse_id"], frame["birth_date"], strict=True))
    assert all(isinstance(j, str) and isinstance(b, date) for j, b in pairs)
    assert {b.year // 100 for _, b in pairs} == {19, 20}, "the sample spans 19xx and 20xx births"
    wrong = [(j, b, join.horse_id(j)) for j, b in pairs if join.horse_id(j) != f"{b.year}{j[2:]}"]
    assert wrong == []
    assert all(j[:2] == f"{b.year % 100:02d}" for j, b in pairs), (
        "the id's prefix is the birth year"
    )


def source_key(race_key: str) -> str | None:
    """JRDB race key -> 12-digit netkeiba race id, by the join's contract: YYYY venue kai(2)
    nichi(2, from one lower-case hex digit) race."""
    venue, yy, kai, nichi, race = (
        race_key[:2],
        race_key[2:4],
        race_key[4],
        race_key[5],
        race_key[6:8],
    )
    if not (kai.isdigit() and nichi in "0123456789abcdef" and (venue + yy + race).isdigit()):
        return None
    return f"{full_year(yy)}{venue}{int(kai):02d}{int(nichi, 16):02d}{race}"


RACE_KEYED = ("BAC", "CHA", "CYB", "JOA", "KKA", "KYI", "SED", "SKB", "TYB")


@pytest.mark.parametrize("family", [f for f in RACE_KEYED if f != "BAC"])
def test_jrdb_public_samples_join_a_store_built_from_them(jrdb_dir, family):
    """Each race-keyed runner sample against a store holding its races (found through
    source_race_key) and runners: rows with a JRDB id match by it, rows without one by horse
    number. The last race is left out of the store (unmatched), and one runner of the first race
    is another horse (a mismatch when the row carries an id)."""
    local_specs()
    rows = parse.parse_file(sample_path(jrdb_dir, family))
    assert {"race_key", "horse_no"} <= set(rows.columns)
    keys = list(dict.fromkeys(rows["race_key"]))
    joinable = [k for k in keys if source_key(k) is not None]
    assert joinable, family
    has_id = "jrdb_horse_id" in rows.columns
    dropped = joinable[-1] if len(joinable) > 1 else None
    first = joinable[0]

    def our_id(i: int) -> str:
        if has_id:
            j = rows["jrdb_horse_id"].iloc[i]
            return full_year(j[:2]) + j[2:]
        return f"20{i:08d}"  # invented: the sample prints no horse id

    races, runners, expected = [], [], []
    for k in joinable:
        if k != dropped:
            races.append(
                {
                    "race_id": f"jra:test:{k}",
                    "race_date": date(2000, 1, 1),
                    "venue": None,
                    "race_no": int(k[6:8]),
                    "source_race_key": source_key(k),
                }
            )
    swapped = None
    for i, (k, no) in enumerate(zip(rows["race_key"], rows["horse_no"], strict=True)):
        horse = our_id(i)
        in_store = k in joinable and k != dropped
        if in_store:
            store_horse = horse
            if k == first and swapped is None:
                swapped, store_horse = i, "2099999999"
            runners.append(
                {"race_id": f"jra:test:{k}", "horse_no": int(no), "horse_id": store_horse}
            )
        expected.append((f"jra:test:{k}" if in_store else None, horse, i == swapped))

    out, report = join.join_store(rows, pd.DataFrame(races), pd.DataFrame(runners))
    assert len(out) == len(rows) and list(out["race_key"]) == list(rows["race_key"])
    race_matched = sum(rid is not None for rid, _, _ in expected)
    assert [None if is_missing(r) else r for r in out["race_id"]] == [rid for rid, _, _ in expected]
    assert (report.rows, report.race_matched) == (len(rows), race_matched)
    by = [None if is_missing(m) else m for m in out["matched_by"]]
    if has_id:
        assert report.horse_matched == race_matched - 1 and report.matched_by_horse_no == 0
        assert [int(m["horse_no"]) for m in report.mismatches] == [
            int(rows["horse_no"].iloc[swapped])
        ]
        assert by == [
            "horse_id" if rid is not None and not swap else None for rid, _, swap in expected
        ]
        assert list(out["horse_id"]) == [horse for _, horse, _ in expected]
    else:
        assert report.horse_matched == race_matched == report.matched_by_horse_no
        assert report.mismatches == []
        assert by == ["horse_no" if rid is not None else None for rid, _, _ in expected]
        assert [None if is_missing(h) else h for h in out["horse_id"]] == [
            ("2099999999" if swap else horse) if rid is not None else None
            for rid, horse, swap in expected
        ]
    assert report.race_unmatched_rate == pytest.approx(1 - race_matched / len(rows))


def name_key(name: str | None) -> str | None:
    """The pedigree's name key, by its contract: NFKC, casefold, letters and digits of any script,
    everything else single spaces; an empty result is None."""
    if name is None:
        return None
    s = re.sub(r"\(\s*[-+]?\d+\s*\)", " ", unicodedata.normalize("NFKC", name)).casefold()
    return re.sub(r"[\W_]+", " ", s).strip() or None


def test_jrdb_pedigree_from_the_ukc_sample(jrdb_dir):
    """Every record of the horse master becomes one horse_pedigree row keyed by our horse_id, the
    ancestor names as printed and normalised, line codes and birth years carried through."""
    jrdb_specs = local_specs()
    path = sample_path(jrdb_dir, "UKC")
    records = records_of(path.read_bytes())
    fields = {
        name: (start, length, kind) for name, start, length, kind in spec_rows(jrdb_specs, "UKC")
    }

    def printed(record: bytes, name: str):
        start, length, kind = fields[name]
        return decode(record[start - 1 : start - 1 + length], kind)

    table = pedigree.build_horse_pedigree(parse.parse_file(path))
    assert list(table.columns) == list(pedigree.OUTPUT_COLUMNS)
    ids = [printed(r, "jrdb_horse_id") for r in records]
    assert len(set(ids)) == len(records), "the sample holds one record per horse"
    assert len(table) == len(records) and table["horse_id"].is_unique
    rows = table.set_index("horse_id")
    for record, jid in zip(records, ids, strict=True):
        row = rows.loc[str(printed(record, "birth_date").year) + jid[2:]]
        for column in ("sire_name", "dam_name", "dam_sire_name"):
            assert row[column] == printed(record, column), (jid, column)
            assert row[f"{column}_norm"] == name_key(printed(record, column)), (jid, column)
        for column in ("sire_line_code", "dam_sire_line_code"):
            assert row[column] == printed(record, column), (jid, column)
        for column in ("sire_birth_year", "dam_birth_year", "dam_sire_birth_year"):
            year = printed(record, column)
            assert (pd.isna(row[column]) if math.isnan(year) else int(row[column]) == int(year)), (
                jid, column,
            )  # fmt: skip
    assert table["sire_name_norm"].notna().all()
    assert table["sire_line_code"].str.fullmatch(r"[0-9]{4}").all()
