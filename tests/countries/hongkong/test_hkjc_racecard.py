"""HKJC race card -> races (the pre-race columns) + racecard_runners (ticket #64, ported from
HKTrack's `sources/hkjc/parse_racecard.py` and tests/unit/sources/hkjc/test_racecard_parser.py)
on synthetic cards (hkjc_racecard_pages.py). The real-page versions belong to a `local` ticket.

Contract pinned here (the coder follows it):
- `horsetrack.countries.hongkong.hkjc.parse_racecard.parse(page: RawPage) -> ParsedPage`
  (`horsetrack.core.ingest.types`), pure, no network, with exactly the tables "races" (one row)
  and "racecard_runners" (the declared starters in table order, then the stand-by starters). A
  page with no starter table (the HTTP 200 "No information." page, the "not available" page,
  garbage) or no race identity gives `{"races": [], "racecard_runners": []}`.
- races: exactly RACE_COLUMNS, a subset of the store's `races` columns with nothing only a result
  knows. meeting_id `hkjc:<YYYY-MM-DD>:<VENUE>`, race_id `<meeting_id>:<NN>`. The header block's
  lines are recognised by content: the title "Race N - NAME"; the date line "<Weekday>, <Month>
  <d>, <yyyy>, <venue name>, <HH:MM>" (venue code from the config's venue names, post_time_local
  the ISO local time with the Hong Kong offset); the course line 'Turf, "C+3" Course, 1650M[,
  <going>]' or 'All Weather Track, 1200M[, <going>]' (surface turf/awt, course_config the quoted
  letters, going upper case as on the results pages, null until declared); the prize line 'Prize
  Money: $N, Rating: hi-lo, Class k' or 'Prize Money: $N, -, Group One' (no band: both ratings
  null). field_size counts the declared starters not scratched; status "declared".
- Identity: the page's own labels (the title's race number, the date line's date and venue) win
  over the cache key; the key (`urls.race_key`, "YYYY-MM-DD_VV_RNN") fills any label the page
  lacks or that cannot be read, and the URL's racedate/Racecourse/RaceNo fill in when the key is
  not a race key.
- racecard_runners: exactly RUNNER_COLUMNS, the store's runner columns a card knows
  (SHARED_RUNNER_COLUMNS) plus card-only ones (CARD_RUNNER_COLUMNS). Columns are found by their
  header text, never by position. card_wt_lb is the printed Wt.; allowance_lb the apprentice claim
  printed "(-N)" in the jockey cell (0 for a jockey with no claim, null with no jockey);
  actual_wt_lb = Wt. - claim + Over Wt. (docs/ledger.md, "HK weight carried", single-run). Jockey
  and trainer names without the claim, keys `jockey:<name>` / `trainer:<name>` (lower case, runs
  of anything but letters and digits as one space), codes from the profile links. Ids from hrefs
  only in [A-Za-z0-9_], else null. Signed cells '+5' / '-3' / '0', '-' null; best_time_s from
  'm.ss.hh'; Priority: '+' trump card, '*' priority to run, the digit the trainer's preference.
- Stand-by starters (table#standbylist) are rows with is_reserve true, reserve_no their order and
  horse_no, draw and the jockey columns null; the TdScratch class that table puts on every row
  does not scratch them. A scratch class on a starter's row or cells, or the word Scratched or
  Withdrawn in it, sets is_scratched; the row stays. (race_id, horse_id) is unique: a horse
  listed twice keeps its declared-starter row.
- card_scraped_utc is the page's fetch time (UTC), null when the page has none.
- Junk never raises: missing or non-numeric cells give nulls, rows without a horse are skipped.

Every expected value is written out here or derived from the card specs by the rules above,
never by the code under test. Names, ids, codes and numbers are invented.
"""

from __future__ import annotations

import random
import re
from dataclasses import replace
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import hkjc_racecard_pages as pages
import pytest
from bs4 import BeautifulSoup
from hkjc_racecard_pages import Card, Reserve, Starter, reserve, starter

from horsetrack.core.ingest.types import RawPage
from horsetrack.core.store import schemas
from horsetrack.countries.hongkong.hkjc import urls

# isort: split
# (its own block: ruff ranks a module that does not exist yet as third-party)
from horsetrack.countries.hongkong.hkjc import parse_racecard

HK = ZoneInfo("Asia/Hong_Kong")
SCRAPED = datetime(2031, 10, 13, 4, 30, 12, tzinfo=timezone.utc)
EMPTY = {"races": [], "racecard_runners": []}

# the store's `races` columns a card can fill (docs/specs/store.md, schemas.TABLES)
RACE_COLUMNS = {
    "jurisdiction", "race_id", "meeting_id", "race_no", "race_name", "race_class", "rating_hi",
    "rating_lo", "distance_m", "surface", "course_config", "going", "prize", "post_time_local",
    "field_size", "status",
}  # fmt: skip
RESULT_RACE_COLUMNS = {"finish_time_s", "split_times_s"}
# the store's `runners` columns a card knows
SHARED_RUNNER_COLUMNS = {
    "jurisdiction", "race_id", "horse_id", "horse_no", "horse_name", "brand_raw", "jockey_key",
    "jockey_code", "jockey_name", "trainer_key", "trainer_code", "trainer_name", "actual_wt_lb",
    "declared_horse_wt_lb", "draw", "is_scratched", "rating", "gear_raw", "sex", "age",
}  # fmt: skip
# `runners` columns only a result knows
RESULT_RUNNER_COLUMNS = {
    "placing_raw", "finish_pos", "placing_code", "dead_heat", "lbw_raw", "lbw_lengths",
    "running_positions", "finish_time_s", "win_odds", "last_3f_s", "win_fav_rank",
}  # fmt: skip
# printed on the card only (HKTrack's racecard_runners)
CARD_RUNNER_COLUMNS = {
    "card_wt_lb", "overweight_lb", "allowance_lb", "intl_rating", "rating_change",
    "horse_wt_change_lb", "best_time_s", "wfa_lb", "season_stakes", "priority_raw", "trump_card",
    "priority_to_run", "trainer_preference", "days_since_last_run", "owner", "sire", "dam",
    "import_type", "last6_raw", "is_reserve", "reserve_no", "card_scraped_utc",
}  # fmt: skip
RUNNER_COLUMNS = SHARED_RUNNER_COLUMNS | CARD_RUNNER_COLUMNS
BOOL_COLUMNS = ("is_reserve", "is_scratched", "trump_card", "priority_to_run")

# Priority as printed -> (trump card, priority to run, trainer's order of preference)
PRIORITY_FLAGS = {
    None: (False, False, None),
    "1": (False, False, 1),
    "+ 1": (True, False, 1),
    "* 1": (False, True, 1),
    "* 2": (False, True, 2),
    "+ * 3": (True, True, 3),
}


# ---------------------------------------------------------------- the synthetic cards

# Happy Valley, Wed 15 Oct 2031, race 1, two days out: going not declared, no body weights,
# claims and overweight, a hyphenated jockey, two stand-by starters.
HV_HANDICAP = Card(
    race_date=date(2031, 10, 15),
    venue="HV",
    race_no=1,
    race_name="SYNTHETIC ALDER HANDICAP",
    post="19:15",
    course='Turf, "C" Course, 1650M',
    prize="Prize Money: $1,170,000, Rating: 60-40, Class 4",
    starters=(
        starter(1, 7),
        starter(2, 3, claim=2),
        starter(3, 1, overweight=1),
        starter(4, 6, claim=10, rating_change=None),
        starter(5, 2, claim=3, overweight=2),
        starter(6, 5, priority=None, gear=None, last6=None),
        starter(7, 4, jockey="A Lima-Bravo", jockey_code="JLB"),
    ),
    reserves=(reserve(1), reserve(2, declared_wt=1088)),
)
# Sha Tin, Sun 2 Nov 2031, race 8: a group race on declaration day (going and body weights in,
# international ratings, weight-for-age allowances).
ST_GROUP = Card(
    race_date=date(2031, 11, 2),
    venue="ST",
    race_no=8,
    race_name="SYNTHETIC BIRCH CUP",
    post="16:40",
    course='Turf, "A+3" Course, 2000M, Good To Firm',
    prize="Prize Money: $20,000,000, -, Group One",
    starters=tuple(
        starter(
            n,
            draw,
            intl_rating=125 - n,
            rating=130 - 2 * n,
            declared_wt=1100 + 19 * n,
            wt_change=(5, -5, 0, 12, -7)[n - 1],
            wfa=(None, 2, None, 4, None)[n - 1],
        )
        for n, draw in zip(range(1, 6), (4, 1, 5, 2, 3), strict=True)
    ),
)
# Sha Tin, Sat 17 Jan 2032, race 3, on the all-weather track; one runner without a best time,
# one without a jockey yet, one stand-by.
ST_AWT = Card(
    race_date=date(2032, 1, 17),
    venue="ST",
    race_no=3,
    race_name="SYNTHETIC CEDAR HANDICAP",
    post="13:30",
    course="All Weather Track, 1200M, Wet Slow",
    prize="Prize Money: $967,500, Rating: 40-0, Class 5",
    starters=(
        starter(1, 4),
        starter(2, 3, best_time_s=None),
        starter(3, 2, claim=5),
        starter(4, 1, jockey=None, jockey_code=None),
    ),
    reserves=(reserve(3),),
)
# Happy Valley, Wed 25 Feb 2032, race 11 (two-digit race number), the last race of the night.
HV_LATE = Card(
    race_date=date(2032, 2, 25),
    venue="HV",
    race_no=11,
    race_name="SYNTHETIC DOGWOOD HANDICAP",
    post="22:50",
    course='Turf, "B+2" Course, 1000M, Yielding',
    prize="Prize Money: $2,690,000, Rating: 100-80, Class 2",
    starters=tuple(starter(n, n) for n in (1, 2, 3)),
)
CARDS = (HV_HANDICAP, ST_GROUP, ST_AWT, HV_LATE)
CARD_IDS = ("hv-handicap", "st-group", "st-awt", "hv-late")


# ---------------------------------------------------------------- helpers


def raw(body: bytes, key: str, url: str, fetched: datetime | None = SCRAPED) -> RawPage:
    return RawPage("hkjc", "racecard", key, url, body, fetched_utc=fetched)


def card_page(
    card: Card,
    *,
    key: str | None = None,
    url: str | None = None,
    fetched: datetime | None = SCRAPED,
    body: bytes | None = None,
    **render,
) -> RawPage:
    """The card as fetched: by default under its own cache key and URL."""
    key = key if key is not None else urls.race_key(card.race_date, card.venue, card.race_no)
    url = url if url is not None else urls.racecard(card.race_date, card.venue, card.race_no)
    return raw(body if body is not None else pages.card(card, **render), key, url, fetched)


def parse(card: Card, **kw):
    return parse_racecard.parse(card_page(card, **kw))


def race_id(card: Card) -> str:
    return f"hkjc:{card.race_date.isoformat()}:{card.venue}:{card.race_no:02d}"


def post_local(day: date, clock: str) -> str:
    hour, minute = (int(part) for part in clock.split(":"))
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=HK).isoformat()


def person_key(role: str, name: str | None) -> str | None:
    words = re.findall(r"[a-z0-9]+", (name or "").lower())
    return f"{role}:{' '.join(words)}" if words else None


def approx(value: float | None):
    return None if value is None else pytest.approx(value)


def expected_starter(s: Starter, rid: str, scraped: datetime | None) -> dict:
    allowance = None if s.jockey is None else (s.claim or 0)
    actual = None if s.wt is None else s.wt - (allowance or 0) + (s.overweight or 0)
    trump, to_run, preference = PRIORITY_FLAGS[s.priority]
    return {
        "jurisdiction": "hkjc",
        "race_id": rid,
        "horse_id": s.horse_id,
        "horse_no": s.number,
        "horse_name": s.name,
        "brand_raw": s.brand,
        "draw": s.draw,
        "card_wt_lb": s.wt,
        "overweight_lb": s.overweight,
        "actual_wt_lb": actual,
        "jockey_key": person_key("jockey", s.jockey),
        "jockey_code": s.jockey_code,
        "jockey_name": s.jockey,
        "trainer_key": person_key("trainer", s.trainer),
        "trainer_code": s.trainer_code,
        "trainer_name": s.trainer,
        "allowance_lb": allowance,
        "intl_rating": s.intl_rating,
        "rating": s.rating,
        "rating_change": s.rating_change,
        "declared_horse_wt_lb": s.declared_wt,
        "horse_wt_change_lb": s.wt_change,
        "best_time_s": approx(s.best_time_s),
        "age": s.age,
        "wfa_lb": s.wfa,
        "sex": s.sex,
        "season_stakes": s.season_stakes,
        "priority_raw": s.priority,
        "trump_card": trump,
        "priority_to_run": to_run,
        "trainer_preference": preference,
        "days_since_last_run": s.days_since,
        "gear_raw": s.gear,
        "owner": s.owner,
        "sire": s.sire,
        "dam": s.dam,
        "import_type": s.import_cat,
        "last6_raw": s.last6,
        "is_reserve": False,
        "reserve_no": None,
        "is_scratched": False,
        "card_scraped_utc": scraped,
    }


def expected_reserve(r: Reserve, rid: str, scraped: datetime | None) -> dict:
    """A stand-by row: the ten stand-by columns, no number, draw or jockey; the weight carried
    is the card weight (no claim, no overweight)."""
    trump, to_run, preference = PRIORITY_FLAGS[r.priority]
    row = dict.fromkeys(RUNNER_COLUMNS)
    row.update(
        jurisdiction="hkjc",
        race_id=rid,
        horse_id=r.horse_id,
        horse_name=r.name,
        card_wt_lb=r.wt,
        actual_wt_lb=r.wt,
        trainer_key=person_key("trainer", r.trainer),
        trainer_code=r.trainer_code,
        trainer_name=r.trainer,
        rating=r.rating,
        declared_horse_wt_lb=r.declared_wt,
        age=r.age,
        priority_raw=r.priority,
        trump_card=trump,
        priority_to_run=to_run,
        trainer_preference=preference,
        gear_raw=r.gear,
        last6_raw=r.last6,
        is_reserve=True,
        reserve_no=r.number,
        is_scratched=False,
        card_scraped_utc=scraped,
    )
    return row


def expected_rows(card: Card, scraped: datetime | None = SCRAPED) -> list[dict]:
    rid = race_id(card)
    return [expected_starter(s, rid, scraped) for s in card.starters] + [
        expected_reserve(r, rid, scraped) for r in card.reserves
    ]


def by_no(rows: list[dict], horse_no: int) -> dict:
    (row,) = [r for r in rows if r["horse_no"] == horse_no and not r["is_reserve"]]
    return row


# ---------------------------------------------------------------- races


def test_race_row_columns_are_the_pre_race_subset_of_the_spec():
    spec = set(schemas.TABLES["races"].columns)
    assert RACE_COLUMNS <= spec and RESULT_RACE_COLUMNS <= spec
    assert not RACE_COLUMNS & RESULT_RACE_COLUMNS
    for card in CARDS:
        parsed = parse(card)
        assert set(parsed) == {"races", "racecard_runners"}, card.race_name
        (race,) = parsed["races"]
        assert set(race) == RACE_COLUMNS, card.race_name


# card -> (race_class, rating_hi, rating_lo, distance_m, surface, course_config, going, prize)
RACES = [
    (HV_HANDICAP, ("Class 4", 60, 40, 1650, "turf", "C", None, 1_170_000)),
    (ST_GROUP, ("Group One", None, None, 2000, "turf", "A+3", "GOOD TO FIRM", 20_000_000)),
    (ST_AWT, ("Class 5", 40, 0, 1200, "awt", None, "WET SLOW", 967_500)),
    (HV_LATE, ("Class 2", 100, 80, 1000, "turf", "B+2", "YIELDING", 2_690_000)),
]


@pytest.mark.parametrize(("card", "header"), RACES, ids=CARD_IDS)
def test_the_race_row_is_read_from_the_header_block(card, header):
    race_class, hi, lo, distance, surface, config, going, prize = header
    assert parse(card)["races"] == [
        {
            "jurisdiction": "hkjc",
            "race_id": race_id(card),
            "meeting_id": f"hkjc:{card.race_date.isoformat()}:{card.venue}",
            "race_no": card.race_no,
            "race_name": card.race_name,
            "race_class": race_class,
            "rating_hi": hi,
            "rating_lo": lo,
            "distance_m": distance,
            "surface": surface,
            "course_config": config,
            "going": going,
            "prize": prize,
            "post_time_local": post_local(card.race_date, card.post),
            "field_size": len(card.starters),
            "status": "declared",
        }
    ]


# (course line, prize line) -> (race_class, surface, course_config, distance_m, going, prize)
GROUP_HEADERS = [
    (
        ('Turf, "A" Course, 2400M, Good', "Prize Money: $25,000,000, -, Group One"),
        ("Group One", "turf", "A", 2400, "GOOD", 25_000_000),
    ),
    (
        ('Turf, "B" Course, 1600M, Good To Yielding', "Prize Money: $6,000,000, -, Group Two"),
        ("Group Two", "turf", "B", 1600, "GOOD TO YIELDING", 6_000_000),
    ),
    (
        ('Turf, "C+3" Course, 1400M, Soft', "Prize Money: $4,250,000, -, Group Three"),
        ("Group Three", "turf", "C+3", 1400, "SOFT", 4_250_000),
    ),
]


def test_group_race_header_has_going_and_no_rating_band():
    for (course, prize), (race_class, surface, config, distance, going, money) in GROUP_HEADERS:
        (race,) = parse(replace(ST_GROUP, course=course, prize=prize))["races"]
        assert race["race_id"] == "hkjc:2031-11-02:ST:08"
        assert race["post_time_local"] == "2031-11-02T16:40:00+08:00"
        got = (
            race["race_class"],
            race["rating_hi"],
            race["rating_lo"],
            race["surface"],
            race["course_config"],
            race["distance_m"],
            race["going"],
            race["prize"],
        )
        assert got == (race_class, None, None, surface, config, distance, going, money), course
    # the same card with a rating band printed reads the band
    banded = replace(ST_GROUP, prize="Prize Money: $3,000,000, Rating: 95-75, Class 1")
    (race,) = parse(banded)["races"]
    assert (race["race_class"], race["rating_hi"], race["rating_lo"]) == ("Class 1", 95, 75)


# all-weather course line -> (distance_m, going)
AWT_LINES = [
    ("All Weather Track, 1650M, Wet Slow", 1650, "WET SLOW"),
    ("All Weather Track, 1200M, Good", 1200, "GOOD"),
    ("All Weather Track, 1800M, Wet Fast", 1800, "WET FAST"),
    ("All Weather Track, 2000M", 2000, None),  # going not declared yet
]


def test_all_weather_header_line():
    for line, distance, going in AWT_LINES:
        (race,) = parse(replace(ST_AWT, course=line))["races"]
        got = (race["surface"], race["course_config"], race["distance_m"], race["going"])
        assert got == ("awt", None, distance, going), line
    # the same card on the turf reads the course letters
    (turf,) = parse(replace(ST_AWT, course='Turf, "A" Course, 1200M, Good'))["races"]
    assert (turf["surface"], turf["course_config"], turf["going"]) == ("turf", "A", "GOOD")


def test_page_labels_win_over_the_cache_key():
    # the page is race 3 at Sha Tin on 2032-01-17; each key (and URL) names another race
    others = [
        (date(2032, 1, 17), "ST", 5),  # another race number
        (date(2032, 1, 17), "HV", 3),  # another venue
        (date(2032, 1, 24), "ST", 3),  # another date
        (date(2030, 6, 1), "HV", 11),  # all three
    ]
    for day, venue, race_no in others:
        page = card_page(
            ST_AWT,
            key=urls.race_key(day, venue, race_no),
            url=urls.racecard(day, venue, race_no),
        )
        parsed = parse_racecard.parse(page)
        (race,) = parsed["races"]
        assert (race["race_id"], race["meeting_id"], race["race_no"]) == (
            "hkjc:2032-01-17:ST:03",
            "hkjc:2032-01-17:ST",
            3,
        ), page.key
        assert race["post_time_local"] == "2032-01-17T13:30:00+08:00"
        assert {r["race_id"] for r in parsed["racecard_runners"]} == {"hkjc:2032-01-17:ST:03"}


KEY = urls.race_key(date(2032, 1, 24), "HV", 7)
URL = urls.racecard(date(2030, 6, 1), "HV", 11)


@pytest.mark.parametrize(
    ("date_line", "expected_id", "expected_post"),
    [
        # no date line: date and venue from the key, the race number still from the title
        (None, "hkjc:2032-01-24:HV:03", None),
        # a venue the config does not know, or none: the venue from the key, the rest from the page
        ("Saturday, January 17, 2032, Conghua, 13:30", "hkjc:2032-01-17:HV:03",
         "2032-01-17T13:30:00+08:00"),
        ("Saturday, January 17, 2032, 13:30", "hkjc:2032-01-17:HV:03",
         "2032-01-17T13:30:00+08:00"),
        # an impossible date: the date from the key, the venue from the page
        ("Monday, February 30, 2032, Sha Tin, 13:30", "hkjc:2032-01-24:ST:03", ...),
        # no post time printed: nothing to fill it from
        ("Saturday, January 17, 2032, Sha Tin", "hkjc:2032-01-17:ST:03", None),
    ],
)  # fmt: skip
def test_the_cache_key_fills_the_labels_the_page_lacks(date_line, expected_id, expected_post):
    parsed = parse(replace(ST_AWT, date_line=date_line), key=KEY, url=URL)
    (race,) = parsed["races"]
    assert race["race_id"] == expected_id
    assert race["meeting_id"] == expected_id.rsplit(":", 1)[0]
    if expected_post is not ...:
        assert race["post_time_local"] == expected_post
    assert {r["race_id"] for r in parsed["racecard_runners"]} == {expected_id}


@pytest.mark.parametrize("key", ["", "2032-01-24_HV", "2032-02-30_HV_R07", "racecard"])
def test_the_url_fills_the_labels_when_the_key_is_not_a_race_key(key):
    (race,) = parse(replace(ST_AWT, date_line=None), key=key, url=URL)["races"]
    assert race["race_id"] == "hkjc:2030-06-01:HV:03"


def test_no_information_page_is_empty():
    for day, venue, race_no in [
        (date(2031, 10, 15), "HV", 10),  # a race number the meeting does not have
        (date(2032, 1, 17), "ST", 1),  # not declared yet
        (date(2029, 4, 8), "ST", 12),  # a past meeting: cards are not kept
    ]:
        page = raw(
            pages.no_information(),
            urls.race_key(day, venue, race_no),
            urls.racecard(day, venue, race_no),
        )
        assert parse_racecard.parse(page) == EMPTY, page.key


def _without_starter_table() -> bytes:
    doc = BeautifulSoup(pages.card(HV_HANDICAP), "lxml")
    doc.select_one("table.starter").decompose()
    return str(doc).encode()


VALID_KEY = urls.race_key(date(2031, 10, 15), "HV", 1)
VALID_URL = urls.racecard(date(2031, 10, 15), "HV", 1)


@pytest.mark.parametrize(
    ("body", "key", "url"),
    [
        pytest.param(pages.not_available(), VALID_KEY, VALID_URL, id="not-available"),
        pytest.param(b"", VALID_KEY, VALID_URL, id="empty-body"),
        pytest.param(b"\xff\xfe\x00 not html at all", VALID_KEY, VALID_URL, id="not-html"),
        pytest.param(pages.page("<p>503 Service Unavailable</p>"), VALID_KEY, VALID_URL,
                     id="server-error"),
        pytest.param(_without_starter_table(), VALID_KEY, VALID_URL, id="no-starter-table"),
        # a full card, but neither the page, the key nor the URL names its date and venue
        pytest.param(pages.card(replace(HV_HANDICAP, date_line=None)), "not-a-key",
                     pages.HOST + "/en-us/local/information/racecard", id="no-identity"),
    ],
)  # fmt: skip
def test_pages_without_a_card_give_empty_tables(body, key, url):
    assert parse_racecard.parse(raw(body, key, url)) == EMPTY


# ---------------------------------------------------------------- runners


def test_runner_rows_carry_the_pre_race_runner_columns():
    spec = set(schemas.TABLES["runners"].columns)
    assert RUNNER_COLUMNS & spec == SHARED_RUNNER_COLUMNS
    assert RESULT_RUNNER_COLUMNS <= spec and not RESULT_RUNNER_COLUMNS & RUNNER_COLUMNS
    for card in CARDS:
        rows = parse(card)["racecard_runners"]
        assert len(rows) == len(card.starters) + len(card.reserves), card.race_name
        for row in rows:
            assert set(row) == RUNNER_COLUMNS, (card.race_name, row.get("horse_id"))


@pytest.mark.parametrize(
    ("card", "fetched"),
    [
        pytest.param(HV_HANDICAP, SCRAPED, id="hv-handicap"),
        pytest.param(ST_GROUP, datetime(2031, 11, 2, 1, 5, 30, tzinfo=timezone.utc),
                     id="st-group"),
        pytest.param(ST_AWT, None, id="st-awt-no-fetch-time"),
        pytest.param(HV_LATE, datetime(2032, 2, 23, 4, 0, tzinfo=timezone.utc), id="hv-late"),
    ],
)  # fmt: skip
def test_runner_rows_are_read_from_the_starter_and_standby_tables(card, fetched):
    rows = parse(card, fetched=fetched)["racecard_runners"]
    assert rows == expected_rows(card, fetched)
    for row in rows:
        assert all(type(row[c]) is bool for c in BOOL_COLUMNS), row["horse_id"]


# (Wt. on the card, claim, Over Wt.) -> (allowance_lb, actual_wt_lb), worked out by hand
WEIGHTS = [
    ((135, None, None), (0, 135)),
    ((131, 2, None), (2, 129)),
    ((125, 10, None), (10, 115)),
    ((119, None, 1), (0, 120)),
    ((128, 5, 2), (5, 125)),
    ((113, 3, 4), (3, 114)),
    ((120, 7, None), (7, 113)),
]


def test_weight_carried_is_card_weight_minus_claim_plus_overweight():
    field = tuple(
        starter(n, n, wt=wt, claim=claim, overweight=over)
        for n, ((wt, claim, over), _) in enumerate(WEIGHTS, start=1)
    )
    rows = parse(replace(HV_HANDICAP, starters=field, reserves=()))["racecard_runners"]
    got = [
        (r["card_wt_lb"], r["allowance_lb"], r["overweight_lb"], r["actual_wt_lb"]) for r in rows
    ]
    assert got == [(wt, a, over, act) for (wt, _, over), (a, act) in WEIGHTS]
    # the claim is split from the name: names and keys are the jockeys' own
    assert [(r["jockey_name"], r["jockey_key"]) for r in rows] == [
        (s.jockey, person_key("jockey", s.jockey)) for s in field
    ]


def test_columns_are_mapped_by_header_text_not_position():
    shuffled = list(pages.COLUMNS)
    random.Random(64).shuffle(shuffled)
    orders = [tuple(reversed(pages.COLUMNS)), pages.COLUMNS[9:] + pages.COLUMNS[:9], shuffled]
    standby = pages.RESERVE_COLUMNS
    reserve_orders = [tuple(reversed(standby)), standby[3:] + standby[:3]]
    for n, order in enumerate(orders):
        parsed = parse(
            HV_HANDICAP, columns=order, reserve_columns=reserve_orders[n % len(reserve_orders)]
        )
        assert parsed["racecard_runners"] == expected_rows(HV_HANDICAP), order
        assert parsed["races"][0]["field_size"] == len(HV_HANDICAP.starters)
    # a card that leaves columns out: those values are null, the rest are unchanged
    dropped = {"Brand No.", "Over Wt.", "Int'l Rtg.", "Best Time", "Owner", "Sire", "Dam"}
    columns = tuple(c for c in pages.COLUMNS if c not in dropped)
    want = expected_rows(HV_HANDICAP)
    for row in want:
        if not row["is_reserve"]:
            row.update(
                brand_raw=None,
                overweight_lb=None,
                intl_rating=None,
                best_time_s=None,
                owner=None,
                sire=None,
                dam=None,
                actual_wt_lb=row["card_wt_lb"] - row["allowance_lb"],
            )
    assert parse(HV_HANDICAP, columns=columns)["racecard_runners"] == want


@pytest.mark.parametrize("count", [1, 2, 3])
def test_standby_starters_are_reserve_rows(count):
    card = replace(ST_AWT, reserves=tuple(reserve(n) for n in range(1, count + 1)))
    parsed = parse(card)
    rows = parsed["racecard_runners"]
    starters = len(card.starters)
    assert [r["is_reserve"] for r in rows] == [False] * starters + [True] * count
    assert rows[starters:] == [expected_reserve(r, race_id(card), SCRAPED) for r in card.reserves]
    # stand-bys are not in the field, and the TdScratch class on their cells does not scratch them
    assert parsed["races"][0]["field_size"] == starters
    assert not any(r["is_scratched"] for r in rows)


# how the card marks a scratching, the starters it marks, the marks
SCRATCHES = [
    pytest.param({5}, dict(cell_class="TdScratch"), id="scratch-class-on-the-cells"),
    pytest.param({2, 6}, dict(row_class="trScratch"), id="scratch-class-on-the-row"),
    pytest.param({3}, dict(jockey_note=" (Scratched)"), id="scratched-after-the-jockey"),
    pytest.param({1, 7}, dict(horse_note=" (Withdrawn)"), id="withdrawn-after-the-horse"),
]


@pytest.mark.parametrize(("numbers", "marks"), SCRATCHES)
def test_scratched_starters_are_flagged_and_leave_the_field(numbers, marks):
    field = tuple(replace(s, **marks) if s.number in numbers else s for s in HV_HANDICAP.starters)
    card = replace(HV_HANDICAP, starters=field)
    parsed = parse(card)
    starters = [r for r in parsed["racecard_runners"] if not r["is_reserve"]]
    # a scratched starter keeps its row, its number and its horse
    assert [(r["horse_no"], r["horse_id"]) for r in starters] == [
        (s.number, s.horse_id) for s in field
    ]
    assert {r["horse_no"] for r in starters if r["is_scratched"]} == numbers
    assert parsed["races"][0]["field_size"] == len(field) - len(numbers)
    assert not any(r["is_scratched"] for r in parsed["racecard_runners"] if r["is_reserve"])


def test_a_horse_listed_twice_keeps_its_declared_starter_row():
    """After a scratching a stand-by takes a number; if the stand-by table still lists it, the
    starter row wins and (race_id, horse_id) stays unique."""
    promoted = HV_HANDICAP.starters[1]
    card = replace(HV_HANDICAP, reserves=(reserve(1, horse_id=promoted.horse_id), reserve(2)))
    parsed = parse(card)
    rows = parsed["racecard_runners"]
    (row,) = [r for r in rows if r["horse_id"] == promoted.horse_id]
    assert (row["is_reserve"], row["horse_no"], row["reserve_no"]) == (False, 2, None)
    assert len(rows) == len(card.starters) + 1
    assert len({r["horse_id"] for r in rows}) == len(rows)
    assert parsed["races"][0]["field_size"] == len(card.starters)


def test_ids_from_hrefs_are_restricted_to_a_plain_charset():
    """Scraped hrefs are untrusted: the ids become URL parameters and cache file names
    downstream. A bad id is null; the names printed beside it are kept."""
    field = list(HV_HANDICAP.starters)
    field[0] = replace(field[0], horse_id="../../etc/passwd")
    field[1] = replace(field[1], jockey_code="H%20CY%3Cscript%3E")
    field[2] = replace(field[2], trainer_code="T%2FR")
    rows = parse(replace(HV_HANDICAP, starters=tuple(field)))["racecard_runners"]
    first, second, third = (by_no(rows, n) for n in (1, 2, 3))
    assert (first["horse_id"], first["horse_name"], first["brand_raw"]) == (
        None,
        field[0].name,
        field[0].brand,
    )
    assert (second["jockey_code"], second["jockey_name"], second["jockey_key"]) == (
        None,
        field[1].jockey,
        person_key("jockey", field[1].jockey),
    )
    assert (third["trainer_code"], third["trainer_name"], third["trainer_key"]) == (
        None,
        field[2].trainer,
        person_key("trainer", field[2].trainer),
    )
    # the other runners' ids are untouched
    assert [r["horse_id"] for r in rows[1:]] == [s.horse_id for s in field[1:]] + [
        r.horse_id for r in HV_HANDICAP.reserves
    ]


def test_short_rows_and_junk_cells_give_nulls_not_errors():
    doc = BeautifulSoup(pages.card(HV_HANDICAP), "lxml")
    rows = doc.select("table.starter tbody tr")
    at = {label: i for i, label in enumerate(pages.COLUMNS)}
    for cell in rows[1].find_all("td", recursive=False)[at["Brand No."] + 1 :]:
        cell.decompose()  # runner 2 loses everything after the brand number
    rows[2].find_all("td", recursive=False)[at["Wt."]].string = "heavy"  # runner 3
    rows[3].find_all("td", recursive=False)[at["Draw"]].string = "N/A"  # runner 4
    junk = doc.new_tag("tr")  # an empty row before the first runner
    junk.append(doc.new_tag("td"))
    rows[0].insert_before(junk)
    dashes = doc.new_tag("tr")  # a row of dashes after the last one
    for _ in pages.COLUMNS:
        cell = doc.new_tag("td")
        cell.string = "-"
        dashes.append(cell)
    rows[-1].insert_after(dashes)

    parsed = parse(HV_HANDICAP, body=str(doc).encode())
    rows = parsed["racecard_runners"]
    assert [r["horse_no"] for r in rows if not r["is_reserve"]] == list(range(1, 8))
    spec = {s.number: s for s in HV_HANDICAP.starters}
    second, third, fourth = (by_no(rows, n) for n in (2, 3, 4))
    assert (second["horse_id"], second["horse_name"], second["brand_raw"]) == (
        spec[2].horse_id,
        spec[2].name,
        spec[2].brand,
    )
    assert all(
        second[c] is None
        for c in ("card_wt_lb", "actual_wt_lb", "allowance_lb", "draw", "jockey_name",
                  "jockey_key", "trainer_name", "trainer_key", "rating", "gear_raw")
    )  # fmt: skip
    assert (third["card_wt_lb"], third["actual_wt_lb"], third["draw"]) == (None, None, spec[3].draw)
    assert (fourth["draw"], fourth["card_wt_lb"], fourth["actual_wt_lb"]) == (
        None,
        spec[4].wt,
        spec[4].wt - spec[4].claim,
    )
    rid = race_id(HV_HANDICAP)
    for n in (1, 5, 6, 7):  # the untouched rows are read as usual
        assert by_no(rows, n) == expected_starter(spec[n], rid, SCRAPED)
