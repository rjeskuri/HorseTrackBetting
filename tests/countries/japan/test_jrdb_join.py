"""JRDB rows meet the JRA store (ticket #56, ported from HKTrack's `sources/jra/jrdb/join.py`).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jrdb.join`:
- `horse_id(jrdb_id)`: JRDB's 8-character horse id is the 2-digit birth year + 6 digits; ours is the
  4-digit birth year + the same 6 digits. Two-digit years 70-99 are 19xx, 00-69 are 20xx
  ('19103588' -> '2019103588', '99101712' -> '1999101712'). Anything but a str of exactly eight
  ASCII digits (blank, None, 7 or 9 digits, a letter, full-width digits, an int) -> None.
- `source_race_key(race_key)`: the JRDB race key, venue 2 + year 2 + kai 1 (a digit) + nichi 1
  (a lower-case hex digit) + race 2, becomes the 12-digit netkeiba race id that the results parser
  keeps in `races.source_race_key`: YYYY + venue + kai (2) + nichi (2, decimal) + race
  ('05261a03' -> '202605011003'). Not 8 characters, a kai that is not a digit, a nichi that is not
  a lower-case hex digit (JRDB's synthetic keys for non-JRA runs use letters) -> None.
- `join_store(rows, races, runners) -> (DataFrame, JoinReport)`: `rows` has `race_key`,
  `horse_no` and, where the family prints it, `jrdb_horse_id`. The result keeps every input row
  and column, in order, and adds `race_id` (None when no race matches), `race_date` (the store
  race's date), `horse_id` and `matched_by` ("horse_id", "horse_no" or None).
  A race is found through `races.source_race_key`; a race row without one (an upcoming card race)
  through its own `venue` (the venue number from countries/japan/config/jra.yaml), `race_date`
  year, `kai`, `nichi` and `race_no`. A row WITH a JRDB id: `horse_id` is the converted id,
  matched when it equals the store runner's horse at that race and horse_no. A row WITHOUT one
  (no column, None or blank): matched to the store runner at (race, horse_no), `horse_id` taken
  from the store. A JRDB id that names another horse than the store's at that number is a
  mismatch, never rescued by its number; a horse_no the store lacks is unmatched, not a mismatch.
  `JoinReport`: `rows`, `race_matched`, `horse_matched` (rows matched either way),
  `matched_by_horse_no`, `race_unmatched_rate` = 1 - race_matched / rows,
  `horse_unmatched_rate` = 1 - horse_matched / race_matched, `mismatches` (one dict per
  mismatching row with `race_id`, `horse_no`, `horse_id` (JRDB's, converted), `store_horse_id`),
  and `to_json()` (the counts, `mismatches` as a count, both rates).

Every race, horse id and runner below is invented; the expected values are worked out by hand or
from the rule above, never by calling the code under test.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

# isort: split
# (its own block: ruff ranks a module that does not exist yet as third-party)
from horsetrack.countries.japan.jrdb import join

VENUE_NO = {"SAP": "01", "HAK": "02", "TOK": "05", "NAK": "06", "KYO": "08", "HSN": "09"}


def _none(value: Any) -> Any:
    """None for every missing-value spelling pandas may hand back."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _century(two_digits: str) -> str:
    return "19" if int(two_digits) >= 70 else "20"


# ------------------------------------------------------------------------------------------------
# Identifier rules
# ------------------------------------------------------------------------------------------------

HORSE_IDS = [
    ("19103588", "2019103588"),
    ("99101712", "1999101712"),
    ("00105651", "2000105651"),
    ("13103588", "2013103588"),
    ("05100555", "2005100555"),
    ("24110001", "2024110001"),  # foreign-bred numbering follows the same rule
    ("96105322", "1996105322"),
    ("93109855", "1993109855"),
    ("89101234", "1989101234"),
    ("80100002", "1980100002"),
    ("70100003", "1970100003"),  # the pivot: 70 is the first 19xx year
    ("69100004", "2069100004"),  # and 69 the last 20xx one
]


@pytest.mark.parametrize(("jrdb_id", "ours"), HORSE_IDS, ids=[h for h, _ in HORSE_IDS])
def test_horse_id_century_prefix(jrdb_id: str, ours: str) -> None:
    assert join.horse_id(jrdb_id) == ours


def test_horse_id_follows_the_rule_for_every_two_digit_year() -> None:
    """Every two-digit year and a spread of serials: the six serial digits are kept as they are."""
    for year in range(100):
        for serial in ("100001", "105432", "110987", "000000"):
            jid = f"{year:02d}{serial}"
            want = ("19" if year >= 70 else "20") + f"{year:02d}" + serial
            assert join.horse_id(jid) == want, jid


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "        ",
        None,
        "1910358",  # 7 digits
        "191035888",  # 9 digits
        "19A03588",
        " 19103588",
        "19103588 ",
        "1910 588",
        "１９１０３５８８",  # full-width digits
        19103588,  # not a string
        float("nan"),
    ],
    ids=lambda v: repr(v),
)
def test_malformed_horse_ids_give_none(bad: Any) -> None:
    assert join.horse_id(bad) is None


RACE_KEYS = [
    ("08254611", "202508040611"),  # Kyoto 2025, 4th meeting, day 6, race 11
    ("02152201", "201502020201"),
    ("05261a03", "202605011003"),  # day 'a' = 10
    ("06265c12", "202606051212"),  # day 'c' = 12
    ("10141801", "201410010801"),
    ("06993102", "199906030102"),  # 1999
    ("01001b12", "200001011112"),  # 2000, day 'b' = 11
    ("09709f07", "197009091507"),  # day 'f' = 15, year 70 -> 1970
]


@pytest.mark.parametrize(("key", "source"), RACE_KEYS, ids=[k for k, _ in RACE_KEYS])
def test_race_key_to_source_race_key(key: str, source: str) -> None:
    assert join.source_race_key(key) == source


def test_race_key_follows_the_rule_across_venues_meetings_and_days() -> None:
    for venue in ("01", "05", "08", "10"):
        for year in (99, 0, 14, 26):
            for kai in range(1, 10):
                for nichi in range(1, 16):
                    for race_no in (1, 7, 12):
                        key = f"{venue}{year:02d}{kai}{nichi:x}{race_no:02d}"
                        want = (
                            f"{_century(f'{year:02d}')}{year:02d}{venue}"
                            f"{kai:02d}{nichi:02d}{race_no:02d}"
                        )
                        assert join.source_race_key(key) == want, key


@pytest.mark.parametrize(
    "bad",
    [
        "",
        None,
        "0825461",  # 7 characters
        "082546111",  # 9 characters
        "0825g611",  # nichi not hex
        "3008ka01",  # JRDB's synthetic non-JRA key: letters for kai and nichi
        "0825k611",  # kai not a digit
        "08 54611",
        "05261A03",  # upper-case hex is not how JRDB prints a day
        8254611,
    ],
    ids=lambda v: repr(v),
)
def test_malformed_race_keys_give_none(bad: Any) -> None:
    assert join.source_race_key(bad) is None


# ------------------------------------------------------------------------------------------------
# Joining rows to a synthetic store
# ------------------------------------------------------------------------------------------------


def race_id(venue: str, day: date, no: int) -> str:
    return f"jra:{day.isoformat()}:{venue}:{no:02d}"


def source_key(venue: str, day: date, kai: int, nichi: int, no: int) -> str:
    return f"{day.year}{VENUE_NO[venue]}{kai:02d}{nichi:02d}{no:02d}"


def jrdb_key(venue: str, day: date, kai: int, nichi: int, no: int) -> str:
    return f"{VENUE_NO[venue]}{day.year % 100:02d}{kai}{nichi:x}{no:02d}"


# (venue, day, kai, nichi, race_no, horse ids by horse_no, has a source key in the store)
STORE_RACES = [
    ("KYO", date(2026, 6, 6), 3, 1, 1, [f"2023105{i:03d}" for i in range(1, 7)], True),
    ("KYO", date(2026, 6, 6), 3, 1, 2, [f"2022105{i:03d}" for i in range(11, 16)], True),
    ("KYO", date(2026, 6, 6), 3, 1, 3, [f"2021105{i:03d}" for i in range(21, 25)], False),
    ("TOK", date(2026, 10, 11), 4, 11, 11, [f"2020105{i:03d}" for i in range(31, 37)], True),
    ("NAK", date(1999, 12, 26), 5, 8, 10, [f"1996105{i:03d}" for i in range(41, 45)], True),
]


def store_tables(spec=STORE_RACES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`races` and `runners` frames shaped like the store's (only the columns the join reads)."""
    races, runners = [], []
    for venue, day, kai, nichi, no, horses, keyed in spec:
        rid = race_id(venue, day, no)
        races.append(
            {
                "jurisdiction": "jra",
                "race_id": rid,
                "meeting_id": f"jra:{day.isoformat()}:{venue}",
                "race_date": day,
                "venue": venue,
                "race_no": no,
                "source_race_key": source_key(venue, day, kai, nichi, no) if keyed else None,
            }
        )
        runners += [
            {"jurisdiction": "jra", "race_id": rid, "horse_no": i, "horse_id": h}
            for i, h in enumerate(horses, 1)
        ]
    return pd.DataFrame(races), pd.DataFrame(runners)


def test_join_reports_unmatched_races() -> None:
    races, runners = store_tables()

    rows, expect = [], []  # expect: (race_id or None, store horse at that number or None, jrdb id)
    for venue, day, kai, nichi, no, horses, keyed in STORE_RACES:
        rid = race_id(venue, day, no)
        for i, h in enumerate(horses, 1):
            jid = h[2:]
            if venue == "KYO" and no == 2 and i == 3:
                jid = "22109999"  # JRDB names another horse at this number: a mismatch
            rows.append(
                {
                    "race_key": jrdb_key(venue, day, kai, nichi, no),
                    "horse_no": i,
                    "jrdb_horse_id": jid,
                }
            )
            expect.append((rid if keyed else None, h, jid))
        if venue == "TOK":  # a horse_no the store does not have
            rows.append(
                {
                    "race_key": jrdb_key(venue, day, kai, nichi, no),
                    "horse_no": 7,
                    "jrdb_horse_id": "20105099",
                }
            )
            expect.append((rid, None, "20105099"))
    for i in range(1, 4):  # a meeting day the store does not hold (day 2 of the same Kyoto meeting)
        rows.append(
            {
                "race_key": jrdb_key("KYO", date(2026, 6, 7), 3, 2, 1),
                "horse_no": i,
                "jrdb_horse_id": f"231050{i:02d}",
            }
        )
        expect.append((None, None, f"231050{i:02d}"))
    rows.append({"race_key": "3026ka01", "horse_no": 1, "jrdb_horse_id": "23105001"})  # non-JRA
    expect.append((None, None, "23105001"))
    frame = pd.DataFrame(rows)
    frame["idm"] = np.arange(len(frame), dtype="float64")

    out, report = join.join_store(frame, races, runners)

    assert len(out) == len(frame)
    assert list(out["idm"]) == list(frame["idm"])  # same rows, same order
    assert list(out["race_key"]) == list(frame["race_key"])
    got_race = [_none(v) for v in out["race_id"]]
    assert got_race == [rid for rid, _, _ in expect]
    assert list(out["horse_id"]) == [_century(j[:2]) + j for _, _, j in expect]
    dates = {race_id(v, d, n): d for v, d, _, _, n, _, _ in STORE_RACES}
    for rid, value in zip(got_race, out["race_date"], strict=True):
        if rid is None:
            assert _none(value) is None
        else:
            assert pd.Timestamp(value).date() == dates[rid]

    def matched(rid, store, j) -> bool:
        return rid is not None and store == _century(j[:2]) + j

    n = len(expect)
    race_matched = sum(rid is not None for rid, _, _ in expect)
    horse_matched = sum(matched(*e) for e in expect)
    assert (report.rows, report.race_matched, report.horse_matched) == (
        n,
        race_matched,
        horse_matched,
    )
    assert report.matched_by_horse_no == 0  # every row here carries a JRDB id
    assert [_none(v) for v in out["matched_by"]] == [
        "horse_id" if matched(*e) else None for e in expect
    ]
    assert report.race_unmatched_rate == pytest.approx(1 - race_matched / n)
    assert report.horse_unmatched_rate == pytest.approx(1 - horse_matched / race_matched)
    assert 0 < report.race_unmatched_rate < 1 and 0 < report.horse_unmatched_rate < 1

    mismatches = [
        {k: m[k] for k in ("race_id", "horse_no", "horse_id", "store_horse_id")}
        for m in report.mismatches
    ]
    assert [{**m, "horse_no": int(m["horse_no"])} for m in mismatches] == [
        {
            "race_id": "jra:2026-06-06:KYO:02",
            "horse_no": 3,
            "horse_id": "2022109999",
            "store_horse_id": "2022105013",
        }
    ]


def test_join_rates_follow_the_rows() -> None:
    """Fewer races in the store -> a higher unmatched rate, computed, not a constant."""
    rows = [
        {"race_key": jrdb_key(v, d, k, ni, no), "horse_no": i, "jrdb_horse_id": h[2:]}
        for v, d, k, ni, no, horses, _ in STORE_RACES
        for i, h in enumerate(horses, 1)
    ]
    seen = []
    for drop in (0, 1, 3):
        spec = STORE_RACES[: len(STORE_RACES) - drop]
        races, runners = store_tables(spec)
        _, report = join.join_store(pd.DataFrame(rows), races, runners)
        matched = sum(len(r[5]) for r in spec if r[6])
        assert report.rows == len(rows)
        assert report.race_matched == matched
        assert report.race_unmatched_rate == pytest.approx(1 - matched / len(rows))
        assert report.horse_unmatched_rate == pytest.approx(0.0)
        summary = report.to_json()
        assert summary["rows"] == len(rows) and summary["mismatches"] == 0
        assert summary["race_unmatched_rate"] == pytest.approx(1 - matched / len(rows), abs=1e-6)
        seen.append(report.race_unmatched_rate)
    assert seen == sorted(seen) and len(set(seen)) == 3


def test_rows_without_a_horse_id_match_by_horse_no() -> None:
    """Families that print no horse id take the store runner at (race, horse_no), counted apart."""
    races, runners = store_tables()
    kyo1, tok = STORE_RACES[0], STORE_RACES[3]

    # No jrdb_horse_id column at all: Kyoto R1 horses 1-6, a horse 7 the race lacks, a missing race.
    tyb = pd.DataFrame(
        [{"race_key": jrdb_key(*kyo1[:5]), "horse_no": i} for i in range(1, 8)]
        + [{"race_key": jrdb_key("KYO", date(2026, 6, 7), 3, 2, 1), "horse_no": 1}]
    )
    out, report = join.join_store(tyb, races, runners)
    assert [_none(h) for h in out["horse_id"]] == [*kyo1[5], None, None]
    assert [_none(m) for m in out["matched_by"]] == ["horse_no"] * 6 + [None, None]
    assert (report.rows, report.race_matched, report.horse_matched) == (8, 7, 6)
    assert report.matched_by_horse_no == 6
    assert report.horse_unmatched_rate == pytest.approx(1 / 7)
    assert report.mismatches == []

    # A blank id in a frame that has the column counts the same; a WRONG id stays a mismatch.
    mixed = pd.DataFrame(
        [
            {"race_key": jrdb_key(*tok[:5]), "horse_no": 1, "jrdb_horse_id": tok[5][0][2:]},
            {"race_key": jrdb_key(*tok[:5]), "horse_no": 2, "jrdb_horse_id": None},
            {"race_key": jrdb_key(*tok[:5]), "horse_no": 3, "jrdb_horse_id": "        "},
            {"race_key": jrdb_key(*tok[:5]), "horse_no": 4, "jrdb_horse_id": "20109999"},
        ]
    )
    out, report = join.join_store(mixed, races, runners)
    assert [_none(m) for m in out["matched_by"]] == ["horse_id", "horse_no", "horse_no", None]
    assert [_none(h) for h in out["horse_id"]] == [*tok[5][:3], "2020109999"]
    assert (report.horse_matched, report.matched_by_horse_no) == (3, 2)
    assert [int(m["horse_no"]) for m in report.mismatches] == [4]
    assert report.mismatches[0]["store_horse_id"] == tok[5][3]


CARD_DAY = date(2026, 11, 21)


def card_races(kai: int, nichi: int, fields: dict[int, list[str]], venue: str = "KYO"):
    """An upcoming card: race rows with kai / nichi and no source_race_key."""
    races = pd.DataFrame(
        [
            {
                "jurisdiction": "jra",
                "race_id": race_id(venue, CARD_DAY, no),
                "meeting_id": f"jra:{CARD_DAY.isoformat()}:{venue}",
                "race_date": CARD_DAY,
                "venue": venue,
                "race_no": no,
                "kai": kai,
                "nichi": nichi,
                "status": "declared",
            }
            for no in fields
        ]
    )
    card = pd.DataFrame(
        [
            {"race_id": race_id(venue, CARD_DAY, no), "horse_no": i, "horse_id": h}
            for no, horses in fields.items()
            for i, h in enumerate(horses, 1)
        ]
    )
    return races, card


@pytest.mark.parametrize("venue", ["KYO", "TOK", "HSN"])
@pytest.mark.parametrize(("kai", "nichi"), [(5, 7), (4, 11), (1, 1)])
def test_an_upcoming_race_without_a_source_key_joins_by_venue_kai_nichi(
    venue: str, kai: int, nichi: int
) -> None:
    fields = {
        1: [f"2023106{i:03d}" for i in range(1, 6)],
        11: [f"2022106{i:03d}" for i in range(1, 4)],
    }
    races, card = card_races(kai, nichi, fields, venue=venue)
    rows = pd.DataFrame(
        [
            {
                "race_key": jrdb_key(venue, CARD_DAY, kai, nichi, no),
                "horse_no": i,
                "jrdb_horse_id": h[2:],
            }
            for no, horses in fields.items()
            for i, h in enumerate(horses, 1)
        ]
        + [  # the next day of the same meeting is not on the card
            {
                "race_key": jrdb_key(venue, CARD_DAY, kai, nichi + 1, 1),
                "horse_no": 1,
                "jrdb_horse_id": "23106001",
            }
        ]
    )
    out, report = join.join_store(rows, races, card)
    want = [race_id(venue, CARD_DAY, no) for no, horses in fields.items() for _ in horses] + [None]
    assert [_none(r) for r in out["race_id"]] == want
    assert (report.rows, report.race_matched, report.horse_matched) == (9, 8, 8)
    assert [_none(m) for m in out["matched_by"]] == ["horse_id"] * 8 + [None]

    # The same card with an explicit empty source_race_key column joins the same way.
    races["source_race_key"] = None
    out, _ = join.join_store(rows, races, card)
    assert [_none(r) for r in out["race_id"]] == want
