"""Per-race takeout with the JRA Premium boosts (tickets #76, #197).

Contract pinned here (`horsetrack.countries.japan.takeout`):
- `load_boosts(path=None) -> BoostCalendar`; no path loads the shipped
  `countries/japan/config/takeout_boosts.yaml`. The file is a mapping with one key, `boosts`, a
  list of entries
  `{kind: premium | super_premium | ultra_premium, date, until?, age?, venue?, race_no?, pools?}`;
  a missing venue, race_no or pools means every venue, race or pool that day. An unknown kind
  fails to load.
  - `until: YYYY-MM-DD` (#197): the entry covers every day from `date` to `until`, both included;
    `until` before `date`, or not a date, fails to load.
  - `age: N` (#197, a positive int; anything else fails to load): the entry matches only races
    restricted to that age, i.e. whose conditions label (the store's `race_class_raw`, e.g.
    "2歳未勝利", "2歳オープン") starts with "N歳" and is not an "N歳以上" label.
- `BoostCalendar.entries`: the loaded entries, each with `.kind` (str) and `.date` (datetime.date),
  `.venue`, `.race_no`, `.pools`, `.until` (datetime.date or None) and `.age` (int or None).
- `race_takeout(race_id, pool, *, calendar=None, race_class_raw=None) -> RaceTakeout(pool,
  base_rate, payout_rate, boost, boost_may_be_withheld)`; calendar None means the shipped calendar.
  - entries without `age` ignore race_class_raw. When an `age` entry matches the race's date, venue,
    race and pool and race_class_raw is None, it raises `RaceConditionsNeeded` (a ValueError);
    where no `age` entry matches, race_class_raw None works as before.
  - base_rate: the base payout rate, 1 - Pool.takeout_on(date) from jra.yaml.
  - ordinary race: payout_rate == base_rate, boost None, boost_may_be_withheld False.
  - premium: payout_rate = base_rate + 0.05, boost "premium", boost_may_be_withheld True.
  - super_premium: payout_rate 0.80 on every pool, boost "super_premium".
  - ultra_premium: payout_rate 0.85 on every pool, boost "ultra_premium".
  - a date before 2014-06-07 raises TakeoutUnknown; an unknown pool code or a malformed race id
    raises ValueError or KeyError.
Pool codes are #20's: TIERCE is the Trifecta, FORECAST the Exacta, QUINELLA_PLACE the Wide.
Every calendar here is synthetic and written to tmp_path, except the shipped-calendar tests, whose
expected races come from the JRA campaign pages (dates, venues, pools, counts) with race numbers
looked up in the local store; the one store-backed test is marked `local`.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from horsetrack.core import config, paths
from horsetrack.countries.japan import takeout
from horsetrack.countries.japan.takeout import (
    TakeoutUnknown,
    load_boosts,
    race_takeout,
)


def conditions_needed() -> type[Exception]:
    """takeout.RaceConditionsNeeded, looked up at call time so the #76 tests run without it."""
    return takeout.RaceConditionsNeeded


# Takeout in force from 2014-06-07, per pool (JRA's published rates; independent of jra.yaml).
TAKEOUT = {
    "WIN": 0.20,
    "PLACE": 0.20,
    "BRACKET_QUINELLA": 0.225,
    "QUINELLA": 0.225,
    "QUINELLA_PLACE": 0.225,
    "FORECAST": 0.25,
    "TRIO": 0.25,
    "TIERCE": 0.275,
    "WIN5": 0.30,
}
POOLS = sorted(TAKEOUT)


def calendar(tmp_path: Path, body: str):
    path = tmp_path / "boosts.yaml"
    path.write_text(body, encoding="utf-8")
    return load_boosts(path)


@pytest.fixture
def empty(tmp_path: Path):
    return calendar(tmp_path, "boosts: []\n")


# --- the base rate ---


def test_ordinary_race_uses_the_pool_takeout_for_its_date(empty) -> None:
    win = race_takeout("jra:2025-06-01:TOK:5", "WIN", calendar=empty)
    assert win.pool == "WIN"
    assert win.payout_rate == pytest.approx(0.80)
    assert win.base_rate == pytest.approx(0.80)
    assert win.boost is None
    assert win.boost_may_be_withheld is False

    tierce = race_takeout("jra:2025-06-01:TOK:5", "TIERCE", calendar=empty)
    assert tierce.pool == "TIERCE"
    assert tierce.payout_rate == pytest.approx(0.725)
    assert tierce.base_rate == pytest.approx(0.725)
    assert tierce.boost is None


@pytest.mark.parametrize("pool", POOLS)
@pytest.mark.parametrize(
    "race_id",
    [
        "jra:2014-06-07:NAK:1",
        "jra:2019-12-22:NAK:11",
        "jra:2026-11-22:KYO:11",
        "jra:2025-06-01:SAP:12",
    ],
)
def test_every_pool_pays_one_minus_its_takeout_on_an_ordinary_race(empty, race_id, pool) -> None:
    got = race_takeout(race_id, pool, calendar=empty)
    assert got.pool == pool
    assert got.base_rate == pytest.approx(1 - TAKEOUT[pool])
    assert got.payout_rate == pytest.approx(1 - TAKEOUT[pool])
    assert got.boost is None
    assert got.boost_may_be_withheld is False


@pytest.mark.parametrize("pool", POOLS)
def test_jra_yaml_carries_every_pool_with_its_takeout_from_2014_06_07(pool) -> None:
    pools = config.load_source("japan", "jra").pools
    assert set(pools) == set(TAKEOUT)
    assert pools[pool].takeout_on(date(2014, 6, 7)) == pytest.approx(TAKEOUT[pool])
    assert pools[pool].takeout_on(date(2014, 6, 6)) is None


# --- Super Premium ---


def test_super_premium_day_sets_every_pool_to_80_percent(tmp_path: Path) -> None:
    cal = calendar(tmp_path, "boosts:\n  - {kind: super_premium, date: 2025-06-01}\n")
    for race_id in ("jra:2025-06-01:TOK:5", "jra:2025-06-01:KYO:11", "jra:2025-06-01:HAK:1"):
        for pool in POOLS:
            got = race_takeout(race_id, pool, calendar=cal)
            assert got.payout_rate == pytest.approx(0.80), (race_id, pool)
            assert got.base_rate == pytest.approx(1 - TAKEOUT[pool]), (race_id, pool)
            assert got.boost == "super_premium", (race_id, pool)
    # the named pools of the ticket, spelled out
    for pool in ("TIERCE", "TRIO", "WIN", "WIN5"):
        assert race_takeout(
            "jra:2025-06-01:TOK:5", pool, calendar=cal
        ).payout_rate == pytest.approx(0.80)


def test_super_premium_does_not_leak_to_another_date(tmp_path: Path) -> None:
    cal = calendar(tmp_path, "boosts:\n  - {kind: super_premium, date: 2025-06-01}\n")
    for race_id in ("jra:2025-05-31:TOK:5", "jra:2025-06-02:TOK:5", "jra:2024-06-01:TOK:5"):
        got = race_takeout(race_id, "TIERCE", calendar=cal)
        assert got.payout_rate == pytest.approx(0.725), race_id
        assert got.boost is None, race_id


# --- Premium ---


PREMIUM_TOK_11_QUINELLA = """\
boosts:
  - kind: premium
    date: 2025-06-01
    venue: TOK
    race_no: 11
    pools: [QUINELLA]
"""


def test_premium_adds_five_percent_to_the_designated_pool_only(tmp_path: Path) -> None:
    cal = calendar(tmp_path, PREMIUM_TOK_11_QUINELLA)

    boosted = race_takeout("jra:2025-06-01:TOK:11", "QUINELLA", calendar=cal)
    assert boosted.payout_rate == pytest.approx(0.825)
    assert boosted.base_rate == pytest.approx(0.775)
    assert boosted.boost == "premium"
    assert boosted.boost_may_be_withheld is True

    trio = race_takeout("jra:2025-06-01:TOK:11", "TRIO", calendar=cal)
    assert trio.payout_rate == pytest.approx(0.75)
    assert trio.boost is None
    assert trio.boost_may_be_withheld is False

    race_10 = race_takeout("jra:2025-06-01:TOK:10", "QUINELLA", calendar=cal)
    assert race_10.payout_rate == pytest.approx(0.775)
    assert race_10.boost is None


@pytest.mark.parametrize(
    "race_id",
    [
        "jra:2025-06-01:KYO:11",  # another venue, same race number
        "jra:2025-06-02:TOK:11",  # another date
        "jra:2025-06-01:TOK:12",
        "jra:2025-06-01:TOK:1",
    ],
)
def test_premium_does_not_leak_to_another_race(tmp_path: Path, race_id: str) -> None:
    cal = calendar(tmp_path, PREMIUM_TOK_11_QUINELLA)
    got = race_takeout(race_id, "QUINELLA", calendar=cal)
    assert got.payout_rate == pytest.approx(0.775)
    assert got.boost is None
    assert got.boost_may_be_withheld is False


@pytest.mark.parametrize("pool", POOLS)
def test_premium_without_pools_or_race_boosts_every_pool_at_that_venue(
    tmp_path: Path, pool
) -> None:
    cal = calendar(tmp_path, "boosts:\n  - {kind: premium, date: 2025-06-01, venue: TOK}\n")
    for race_no in (1, 6, 12):
        got = race_takeout(f"jra:2025-06-01:TOK:{race_no}", pool, calendar=cal)
        assert got.payout_rate == pytest.approx(1 - TAKEOUT[pool] + 0.05), race_no
        assert got.base_rate == pytest.approx(1 - TAKEOUT[pool])
        assert got.boost == "premium"
        assert got.boost_may_be_withheld is True
    other = race_takeout("jra:2025-06-01:HSN:11", pool, calendar=cal)
    assert other.payout_rate == pytest.approx(1 - TAKEOUT[pool])
    assert other.boost is None


def test_premium_without_venue_boosts_every_venue_that_day(tmp_path: Path) -> None:
    cal = calendar(
        tmp_path, "boosts:\n  - {kind: premium, date: 2025-06-01, pools: [WIN, TIERCE]}\n"
    )
    for venue in ("TOK", "KYO", "SAP"):
        rid = f"jra:2025-06-01:{venue}:3"
        assert race_takeout(rid, "WIN", calendar=cal).payout_rate == pytest.approx(0.85)
        assert race_takeout(rid, "TIERCE", calendar=cal).payout_rate == pytest.approx(0.775)
        assert race_takeout(rid, "TRIO", calendar=cal).payout_rate == pytest.approx(0.75)


def test_premium_on_two_pools_of_one_race(tmp_path: Path) -> None:
    cal = calendar(
        tmp_path,
        "boosts:\n"
        "  - {kind: premium, date: 2026-11-22, venue: KYO, race_no: 11,\n"
        "     pools: [WIN5, QUINELLA_PLACE]}\n"
        "  - {kind: premium, date: 2026-11-22, venue: TOK, race_no: 9, pools: [FORECAST]}\n",
    )
    expect = {
        ("jra:2026-11-22:KYO:11", "WIN5"): 0.75,
        ("jra:2026-11-22:KYO:11", "QUINELLA_PLACE"): 0.825,
        ("jra:2026-11-22:KYO:11", "FORECAST"): 0.75,
        ("jra:2026-11-22:TOK:9", "FORECAST"): 0.80,
        ("jra:2026-11-22:TOK:9", "WIN5"): 0.70,
        ("jra:2026-11-22:TOK:11", "QUINELLA_PLACE"): 0.775,
    }
    for (rid, pool), rate in expect.items():
        assert race_takeout(rid, pool, calendar=cal).payout_rate == pytest.approx(rate), (rid, pool)


# --- Ultra Premium ---


def test_ultra_premium_is_80_percent_plus_5(tmp_path: Path) -> None:
    cal = calendar(tmp_path, "boosts:\n  - {kind: ultra_premium, date: 2025-12-28}\n")
    for race_id in ("jra:2025-12-28:NAK:11", "jra:2025-12-28:HSN:1"):
        for pool in POOLS:
            got = race_takeout(race_id, pool, calendar=cal)
            assert got.payout_rate == pytest.approx(0.85), (race_id, pool)
            assert got.base_rate == pytest.approx(1 - TAKEOUT[pool]), (race_id, pool)
            assert got.boost == "ultra_premium", (race_id, pool)
    day_before = race_takeout("jra:2025-12-27:NAK:11", "TIERCE", calendar=cal)
    assert day_before.payout_rate == pytest.approx(0.725)
    assert day_before.boost is None


# --- before 2014-06-07 ---


@pytest.mark.parametrize("race_id", ["jra:2014-06-01:TOK:5", "jra:2010-12-26:NAK:10"])
@pytest.mark.parametrize("pool", ["WIN", "TIERCE", "QUINELLA"])
def test_takeout_before_2014_06_07_is_unknown(tmp_path: Path, race_id: str, pool: str) -> None:
    empty = calendar(tmp_path, "boosts: []\n")
    with pytest.raises(TakeoutUnknown):
        race_takeout(race_id, pool, calendar=empty)
    # a boost on that day does not make the base rate known
    day = race_id.split(":")[1]
    boosted = calendar(tmp_path, f"boosts:\n  - {{kind: super_premium, date: {day}}}\n")
    with pytest.raises(TakeoutUnknown):
        race_takeout(race_id, pool, calendar=boosted)


# --- bad input ---


@pytest.mark.parametrize("pool", ["PICK4", "", "EXOTIC"])
def test_unknown_pool_code_raises(empty, pool: str) -> None:
    with pytest.raises((ValueError, KeyError)):
        race_takeout("jra:2025-06-01:TOK:5", pool, calendar=empty)


@pytest.mark.parametrize(
    "race_id",
    [
        "jra:2025-06-01:TOK",
        "jra:2025-13-01:TOK:5",
        "jra:2025-06-01:XXX:5",
        "2025-06-01:TOK:5",
        "jra:2025-06-01:TOK:eleven",
        "",
    ],
)
def test_malformed_race_id_raises(empty, race_id: str) -> None:
    with pytest.raises((ValueError, KeyError)):
        race_takeout(race_id, "WIN", calendar=empty)


def test_unknown_boost_kind_fails_to_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        calendar(tmp_path, "boosts:\n  - {kind: mega_premium, date: 2025-06-01}\n")


# --- the shipped calendar ---


# The dated result entries (one result PDF each) on
# https://www.jra.go.jp/kouza/superpremium/index.html
SUPER_PREMIUM_DAYS = {
    date.fromisoformat(d)
    for d in (
        "2017-12-28", "2018-12-28", "2019-09-16", "2019-12-28", "2020-09-21", "2020-12-13",
        "2020-12-20", "2021-09-20", "2021-12-28", "2022-09-19", "2022-12-28", "2023-07-01",
        "2023-08-27", "2023-09-18", "2023-12-28", "2024-11-24", "2025-09-15", "2025-10-26",
        "2026-09-21", "2026-09-22", "2026-10-10",
    )
}  # fmt: skip


def test_the_shipped_calendar_has_21_super_premium_days() -> None:
    assert len(SUPER_PREMIUM_DAYS) == 21
    shipped = load_boosts()
    days = [e.date for e in shipped.entries if e.kind == "super_premium"]
    assert len(days) == 21
    assert len(set(days)) == 21
    assert set(days) == SUPER_PREMIUM_DAYS
    assert all(isinstance(d, date) for d in days)
    assert min(days) == date(2017, 12, 28)
    assert max(days) == date(2026, 10, 10)
    assert {e.kind for e in shipped.entries} <= {"premium", "super_premium", "ultra_premium"}

    # the default calendar is the shipped one: a race matching each Super Premium entry pays 80%
    for entry in shipped.entries:
        if entry.kind != "super_premium":
            continue
        venue = entry.venue or "NAK"
        race_no = entry.race_no or 11
        pool = entry.pools[0] if entry.pools else "TIERCE"
        rid = f"jra:{entry.date.isoformat()}:{venue}:{race_no}"
        got = race_takeout(rid, pool)
        assert got.payout_rate == pytest.approx(0.80), rid
        assert got.boost == "super_premium", rid


def test_default_calendar_leaves_an_ordinary_day_alone() -> None:
    # a Tuesday that is not a holiday: no JRA racing, so no boost can be listed for it
    got = race_takeout("jra:2025-06-03:TOK:5", "TIERCE")
    assert got.payout_rate == pytest.approx(0.725)
    assert got.boost is None


# --- date ranges: `until` (#197) ---


UNTIL = """\
boosts:
  - {kind: premium, date: 2025-06-07, until: 2025-06-15, pools: [WIN]}
  - {kind: premium, date: 2025-12-27, until: 2026-01-05, venue: NAK, pools: [QUINELLA]}
"""


@pytest.mark.parametrize(
    "day, venue, pool, boosted",
    [
        ("2025-06-07", "TOK", "WIN", True),  # first day
        ("2025-06-11", "HSN", "WIN", True),  # a middle day
        ("2025-06-15", "SAP", "WIN", True),  # last day
        ("2025-06-06", "TOK", "WIN", False),  # the day before
        ("2025-06-16", "TOK", "WIN", False),  # the day after
        ("2025-06-11", "TOK", "PLACE", False),  # another pool
        ("2025-12-27", "NAK", "QUINELLA", True),
        ("2025-12-31", "NAK", "QUINELLA", True),
        ("2026-01-01", "NAK", "QUINELLA", True),  # across the year end
        ("2026-01-05", "NAK", "QUINELLA", True),
        ("2026-01-06", "NAK", "QUINELLA", False),
        ("2025-12-26", "NAK", "QUINELLA", False),
        ("2026-01-02", "KYO", "QUINELLA", False),  # another venue
        ("2025-06-11", "NAK", "QUINELLA", False),
    ],
)
def test_until_covers_every_day_from_date_to_until_inclusive(
    tmp_path: Path, day: str, venue: str, pool: str, boosted: bool
) -> None:
    cal = calendar(tmp_path, UNTIL)
    got = race_takeout(f"jra:{day}:{venue}:7", pool, calendar=cal)
    base = 1 - TAKEOUT[pool]
    assert got.base_rate == pytest.approx(base)
    assert got.payout_rate == pytest.approx(base + 0.05 if boosted else base)
    assert got.boost == ("premium" if boosted else None)
    assert got.boost_may_be_withheld is boosted


def test_until_range_day_by_day_and_its_loaded_fields(tmp_path: Path) -> None:
    cal = calendar(tmp_path, UNTIL)
    first = date(2025, 6, 7)
    for offset in range(-3, 13):
        day = first + timedelta(days=offset)
        got = race_takeout(f"jra:{day.isoformat()}:KYO:3", "WIN", calendar=cal)
        expect = 0.85 if 0 <= offset <= 8 else 0.80
        assert got.payout_rate == pytest.approx(expect), day
    entry = cal.entries[0]
    assert entry.date == date(2025, 6, 7)
    assert entry.until == date(2025, 6, 15)
    assert cal.entries[1].until == date(2026, 1, 5)
    # until equal to date is a one-day entry
    one = calendar(tmp_path, "boosts:\n  - {kind: premium, date: 2025-06-07, until: 2025-06-07}\n")
    assert race_takeout("jra:2025-06-07:TOK:1", "TRIO", calendar=one).payout_rate == pytest.approx(
        0.80
    )
    assert race_takeout("jra:2025-06-08:TOK:1", "TRIO", calendar=one).payout_rate == pytest.approx(
        0.75
    )


# --- age-restricted entries: `age` (#197) ---


AGE = """\
boosts:
  - {kind: premium, date: 2025-06-07, until: 2025-06-08, age: 2, pools: [WIN]}
  - {kind: premium, date: 2025-06-14, age: 3, venue: TOK, pools: [WIN, QUINELLA]}
"""
TWO_YEAR_OLD_LABELS = ["2歳新馬", "2歳未勝利", "2歳オープン", "2歳1勝クラス", "2歳500万下"]
OTHER_LABELS = [
    "3歳未勝利",
    "3歳新馬",
    "3歳オープン",
    "3歳以上1勝クラス",
    "4歳以上オープン",
    "2歳以上オープン",
    "2歳以上未勝利",
    "障害3歳以上未勝利",
]


@pytest.mark.parametrize("day", ["2025-06-07", "2025-06-08"])
@pytest.mark.parametrize("venue", ["TOK", "KOK"])
def test_age_2_boosts_only_races_restricted_to_two_year_olds(
    tmp_path: Path, day: str, venue: str
) -> None:
    cal = calendar(tmp_path, AGE)
    rid = f"jra:{day}:{venue}:5"
    for label in TWO_YEAR_OLD_LABELS:
        win = race_takeout(rid, "WIN", calendar=cal, race_class_raw=label)
        assert win.payout_rate == pytest.approx(0.85), label
        assert win.base_rate == pytest.approx(0.80), label
        assert win.boost == "premium", label
        assert win.boost_may_be_withheld is True, label
        place = race_takeout(rid, "PLACE", calendar=cal, race_class_raw=label)
        assert place.payout_rate == pytest.approx(0.80), label
        assert place.boost is None, label
    for label in OTHER_LABELS:
        win = race_takeout(rid, "WIN", calendar=cal, race_class_raw=label)
        assert win.payout_rate == pytest.approx(0.80), label
        assert win.boost is None, label
        assert win.boost_may_be_withheld is False, label


def test_age_3_boosts_three_year_old_races_and_not_two_year_olds(tmp_path: Path) -> None:
    cal = calendar(tmp_path, AGE)
    rid = "jra:2025-06-14:TOK:9"
    expect = {
        ("3歳未勝利", "WIN"): 0.85,
        ("3歳未勝利", "QUINELLA"): 0.825,
        ("3歳オープン", "WIN"): 0.85,
        ("3歳1勝クラス", "QUINELLA"): 0.825,
        ("3歳未勝利", "TRIO"): 0.75,
        ("3歳以上1勝クラス", "WIN"): 0.80,
        ("3歳以上2勝クラス", "QUINELLA"): 0.775,
        ("障害3歳以上未勝利", "WIN"): 0.80,
        ("2歳新馬", "WIN"): 0.80,
        ("4歳以上オープン", "QUINELLA"): 0.775,
    }
    for (label, pool), rate in expect.items():
        got = race_takeout(rid, pool, calendar=cal, race_class_raw=label)
        assert got.payout_rate == pytest.approx(rate), (label, pool)
    # the age 3 entry is pinned to TOK on 06-14; the age 2 entry ends on 06-08
    assert race_takeout(
        "jra:2025-06-14:HSN:9", "WIN", calendar=cal, race_class_raw="3歳未勝利"
    ).payout_rate == pytest.approx(0.80)
    assert race_takeout(
        "jra:2025-06-09:TOK:5", "WIN", calendar=cal, race_class_raw="2歳新馬"
    ).payout_rate == pytest.approx(0.80)
    assert [e.age for e in cal.entries] == [2, 3]


def test_an_entry_without_age_ignores_the_race_conditions(tmp_path: Path) -> None:
    cal = calendar(tmp_path, "boosts:\n  - {kind: premium, date: 2025-06-07, pools: [WIN]}\n")
    for label in ["2歳新馬", "3歳未勝利", "2歳以上オープン", "4歳以上オープン", None]:
        got = race_takeout("jra:2025-06-07:TOK:5", "WIN", calendar=cal, race_class_raw=label)
        assert got.payout_rate == pytest.approx(0.85), label
        assert got.boost == "premium", label
    assert cal.entries[0].age is None


def test_an_age_entry_on_a_super_premium_day_stacks_to_85_percent(tmp_path: Path) -> None:
    cal = calendar(
        tmp_path,
        "boosts:\n"
        "  - {kind: super_premium, date: 2025-12-28}\n"
        "  - {kind: premium, date: 2025-12-28, age: 2, pools: [WIN]}\n",
    )
    rid = "jra:2025-12-28:NAK:6"
    two = race_takeout(rid, "WIN", calendar=cal, race_class_raw="2歳未勝利")
    assert two.payout_rate == pytest.approx(0.85)
    assert two.boost == "ultra_premium"
    assert two.boost_may_be_withheld is True
    older = race_takeout(rid, "WIN", calendar=cal, race_class_raw="3歳以上2勝クラス")
    assert older.payout_rate == pytest.approx(0.80)
    assert older.boost == "super_premium"
    assert older.boost_may_be_withheld is False
    place = race_takeout(rid, "PLACE", calendar=cal, race_class_raw="2歳未勝利")
    assert place.payout_rate == pytest.approx(0.80)
    assert place.boost == "super_premium"


# --- an age entry needs the race's conditions (#197) ---


def test_race_conditions_needed_is_a_value_error() -> None:
    assert issubclass(conditions_needed(), ValueError)


@pytest.mark.parametrize(
    "race_id, pool",
    [
        ("jra:2025-06-07:TOK:5", "WIN"),
        ("jra:2025-06-08:KOK:1", "WIN"),
        ("jra:2025-06-14:TOK:9", "WIN"),
        ("jra:2025-06-14:TOK:9", "QUINELLA"),
    ],
)
def test_an_age_entry_without_race_conditions_raises(
    tmp_path: Path, race_id: str, pool: str
) -> None:
    cal = calendar(tmp_path, AGE)
    with pytest.raises(conditions_needed()):
        race_takeout(race_id, pool, calendar=cal)
    with pytest.raises(conditions_needed()):
        race_takeout(race_id, pool, calendar=cal, race_class_raw=None)
    # with the conditions given, the same call answers
    got = race_takeout(race_id, pool, calendar=cal, race_class_raw="4歳以上オープン")
    assert got.payout_rate == pytest.approx(1 - TAKEOUT[pool])


@pytest.mark.parametrize(
    "race_id, pool",
    [
        ("jra:2025-06-07:TOK:5", "PLACE"),  # the age entry is WIN only
        ("jra:2025-06-09:TOK:5", "WIN"),  # the day after its range
        ("jra:2025-06-06:TOK:5", "WIN"),  # the day before
        ("jra:2025-06-14:HSN:9", "WIN"),  # the age 3 entry is TOK only
        ("jra:2025-06-14:TOK:9", "TRIO"),
    ],
)
def test_race_conditions_are_not_needed_where_no_age_entry_matches(
    tmp_path: Path, race_id: str, pool: str
) -> None:
    cal = calendar(tmp_path, AGE)
    got = race_takeout(race_id, pool, calendar=cal, race_class_raw=None)
    assert got.payout_rate == pytest.approx(1 - TAKEOUT[pool])
    assert got.boost is None
    assert race_takeout(race_id, pool, calendar=cal).payout_rate == pytest.approx(1 - TAKEOUT[pool])


# --- `until` and `age` that fail to load (#197) ---


@pytest.mark.parametrize(
    "bad, good",
    [
        ("until: 2025-06-06", "until: 2025-06-08"),  # until before date
        ("until: 2024-12-31", "until: 2025-12-31"),
        ("until: soon", "until: 2025-06-07"),
        ("until: 20250615", "until: 2025-06-15"),
        ("until: [2025-06-15]", "until: 2025-06-15"),
        ("age: 0", "age: 2"),
        ("age: -2", "age: 2"),
        ("age: '2'", "age: 2"),
        ("age: 2.0", "age: 2"),
        ("age: true", "age: 3"),
        ("age: two", "age: 2"),
        ("age: [2]", "age: 4"),
    ],
)
def test_a_bad_until_or_age_fails_to_load(tmp_path: Path, bad: str, good: str) -> None:
    body = "boosts:\n  - {{kind: premium, date: 2025-06-07, pools: [WIN], {}}}\n"
    loaded = calendar(tmp_path, body.format(good))
    assert len(loaded.entries) == 1
    with pytest.raises(ValueError):
        calendar(tmp_path, body.format(bad))


# --- the shipped calendar: 2歳単勝, WIN on every 2-year-old race (#197) ---


# Each window's first and last day (inclusive), its extra 12-28 day (a separate one-day entry) and
# its stated race count, from https://www.jra.go.jp/kouza/premium/2sai-tanshou/.
TWO_YEAR_OLD_WINDOWS = [
    ("2015-06-06", "2015-09-06", None, 184),
    ("2016-06-04", "2016-09-04", None, 184),
    ("2017-06-03", "2017-09-03", None, 185),
    ("2017-09-09", "2017-12-17", "2017-12-28", 427),
    ("2018-06-02", "2018-09-02", None, 185),
    ("2018-09-08", "2018-12-16", "2018-12-28", 425),
    ("2019-06-01", "2019-09-01", None, 185),
    ("2019-09-07", "2019-12-15", "2019-12-28", 439),
    ("2020-06-06", "2020-09-06", None, 179),
    ("2020-09-12", "2020-12-20", None, 432),
    ("2021-06-05", "2021-09-05", None, 177),
    ("2021-09-11", "2021-12-19", "2021-12-28", 446),
    ("2022-06-04", "2022-09-04", None, 181),
    ("2022-09-10", "2022-12-18", None, 432),
    ("2023-06-03", "2023-09-03", None, 179),
    ("2023-09-09", "2023-12-17", None, 429),
]


def test_the_shipped_calendar_carries_every_two_year_old_win_window() -> None:
    expected = {(first, last) for first, last, _, _ in TWO_YEAR_OLD_WINDOWS}
    expected |= {(extra, extra) for _, _, extra, _ in TWO_YEAR_OLD_WINDOWS if extra}
    assert len(expected) == 20
    entries = [e for e in load_boosts().entries if getattr(e, "age", None) is not None]
    got = [(e.date.isoformat(), (e.until or e.date).isoformat()) for e in entries]
    assert len(got) == 20
    assert set(got) == expected
    for e in entries:
        assert e.kind == "premium", e
        assert e.age == 2, e
        assert tuple(e.pools) == ("WIN",), e
        assert e.venue is None, e
        assert e.race_no is None, e


# (a 2-year-old race, its conditions, another race at that meeting, its conditions). 2015-2019:
# races the JRA result PDFs list as boosted (and one they do not); later: first and last days of
# each window and graded 2-year-old stakes. Race numbers and conditions from the local store.
TWO_YEAR_OLD_SAMPLES = [
    ("jra:2015-06-06:TOK:05", "2歳新馬", "jra:2015-06-06:TOK:06", "3歳未勝利"),
    ("jra:2015-07-26:HAK:11", "2歳オープン", "jra:2015-07-26:HAK:12", "3歳以上1000万下"),
    ("jra:2015-09-06:SAP:10", "2歳オープン", "jra:2015-09-06:SAP:11", "3歳以上オープン"),
    ("jra:2016-06-04:TOK:05", "2歳新馬", "jra:2016-06-04:TOK:06", "3歳未勝利"),
    ("jra:2016-08-13:KOK:09", "2歳オープン", "jra:2016-08-13:KOK:10", "3歳以上1000万下"),
    ("jra:2016-09-04:KOK:11", "2歳オープン", "jra:2016-09-04:KOK:12", "3歳以上500万下"),
    ("jra:2017-06-03:TOK:05", "2歳新馬", "jra:2017-06-03:TOK:06", "3歳未勝利"),
    ("jra:2017-09-03:KOK:11", "2歳オープン", "jra:2017-09-03:KOK:12", "3歳以上500万下"),
    ("jra:2017-09-09:HSN:01", "2歳未勝利", "jra:2017-09-09:HSN:04", "3歳未勝利"),
    ("jra:2017-10-07:TOK:11", "2歳オープン", "jra:2017-10-07:TOK:12", "3歳以上500万下"),
    ("jra:2017-12-17:CKY:01", "2歳未勝利", "jra:2017-12-17:CKY:02", "3歳以上500万下"),
    ("jra:2017-12-28:HSN:06", "2歳500万下", "jra:2017-12-28:HSN:07", "3歳以上500万下"),
    ("jra:2017-12-28:NAK:11", "2歳オープン", "jra:2017-12-28:NAK:12", "3歳以上1000万下"),
    ("jra:2018-06-02:TOK:05", "2歳新馬", "jra:2018-06-02:TOK:06", "3歳未勝利"),
    ("jra:2018-09-02:SAP:10", "2歳オープン", "jra:2018-09-02:SAP:11", "3歳以上オープン"),
    ("jra:2018-09-08:HSN:01", "2歳未勝利", "jra:2018-09-08:HSN:03", "3歳未勝利"),
    ("jra:2018-10-02:HSN:01", "2歳未勝利", "jra:2018-10-02:HSN:03", "3歳未勝利"),
    ("jra:2018-11-03:KYO:11", "2歳オープン", "jra:2018-11-03:KYO:12", "3歳以上1000万下"),
    ("jra:2018-12-09:HSN:11", "2歳オープン", "jra:2018-12-09:HSN:12", "3歳以上1000万下"),
    ("jra:2018-12-16:CKY:05", "2歳未勝利", "jra:2018-12-16:CKY:04", "障害3歳以上未勝利"),
    ("jra:2018-12-28:HSN:06", "2歳500万下", "jra:2018-12-28:HSN:07", "3歳以上500万下"),
    ("jra:2018-12-28:NAK:11", "2歳オープン", "jra:2018-12-28:NAK:12", "3歳以上1000万下"),
    ("jra:2019-06-01:TOK:05", "2歳新馬", "jra:2019-06-01:TOK:06", "3歳未勝利"),
    ("jra:2019-08-03:NII:09", "2歳オープン", "jra:2019-08-03:NII:10", "3歳以上2勝クラス"),
    ("jra:2019-09-01:KOK:11", "2歳オープン", "jra:2019-09-01:KOK:12", "3歳以上1勝クラス"),
    ("jra:2019-09-07:HSN:01", "2歳未勝利", "jra:2019-09-07:HSN:06", "3歳以上1勝クラス"),
    ("jra:2019-12-15:CKY:10", "2歳1勝クラス", "jra:2019-12-15:CKY:11", "3歳以上3勝クラス"),
    ("jra:2019-12-28:HSN:06", "2歳1勝クラス", "jra:2019-12-28:HSN:07", "3歳以上1勝クラス"),
    ("jra:2019-12-28:NAK:11", "2歳オープン", "jra:2019-12-28:NAK:12", "3歳以上2勝クラス"),
    ("jra:2020-06-06:TOK:05", "2歳新馬", "jra:2020-06-06:TOK:06", "3歳未勝利"),
    ("jra:2020-09-06:KOK:11", "2歳オープン", "jra:2020-09-06:KOK:12", "3歳以上1勝クラス"),
    ("jra:2020-09-12:CKY:01", "2歳未勝利", "jra:2020-09-12:CKY:06", "3歳以上1勝クラス"),
    ("jra:2020-10-10:TOK:11", "2歳オープン", "jra:2020-10-10:TOK:12", "3歳以上2勝クラス"),
    ("jra:2020-12-20:CKY:10", "2歳1勝クラス", "jra:2020-12-20:CKY:11", "3歳以上3勝クラス"),
    ("jra:2021-06-05:TOK:05", "2歳新馬", "jra:2021-06-05:TOK:06", "3歳未勝利"),
    ("jra:2021-09-05:KOK:11", "2歳オープン", "jra:2021-09-05:KOK:12", "3歳以上1勝クラス"),
    ("jra:2021-09-11:CKY:01", "2歳未勝利", "jra:2021-09-11:CKY:06", "3歳以上1勝クラス"),
    ("jra:2021-12-19:CKY:10", "2歳1勝クラス", "jra:2021-12-19:CKY:11", "3歳以上3勝クラス"),
    ("jra:2021-12-28:HSN:07", "2歳1勝クラス", "jra:2021-12-28:HSN:08", "3歳以上1勝クラス"),
    ("jra:2021-12-28:NAK:11", "2歳オープン", "jra:2021-12-28:NAK:12", "3歳以上3勝クラス"),
    ("jra:2022-06-04:TOK:05", "2歳新馬", "jra:2022-06-04:TOK:06", "3歳未勝利"),
    ("jra:2022-09-04:KOK:11", "2歳オープン", "jra:2022-09-04:KOK:12", "3歳以上1勝クラス"),
    ("jra:2022-09-10:CKY:03", "2歳新馬", "jra:2022-09-10:CKY:04", "3歳以上1勝クラス"),
    ("jra:2022-12-11:HSN:11", "2歳オープン", "jra:2022-12-11:HSN:12", "3歳以上2勝クラス"),
    ("jra:2022-12-18:CKY:09", "2歳1勝クラス", "jra:2022-12-18:CKY:10", "3歳以上2勝クラス"),
    ("jra:2023-06-03:TOK:05", "2歳新馬", "jra:2023-06-03:TOK:06", "3歳未勝利"),
    ("jra:2023-09-03:KOK:11", "2歳オープン", "jra:2023-09-03:KOK:12", "3歳以上1勝クラス"),
    ("jra:2023-09-09:HSN:01", "2歳未勝利", "jra:2023-09-09:HSN:06", "3歳以上1勝クラス"),
    ("jra:2023-12-17:CKY:09", "2歳1勝クラス", "jra:2023-12-17:CKY:10", "3歳以上2勝クラス"),
    ("jra:2023-12-17:HSN:11", "2歳オープン", "jra:2023-12-17:HSN:12", "3歳以上3勝クラス"),
]


def _window_of(day: date) -> tuple[str, str, str | None, int] | None:
    for window in TWO_YEAR_OLD_WINDOWS:
        first, last, extra, _ = window
        if first <= day.isoformat() <= last or day.isoformat() == extra:
            return window
    return None


def test_the_two_year_old_samples_cover_every_window_and_12_28_day() -> None:
    windows = {_window_of(date.fromisoformat(s[0].split(":")[1])) for s in TWO_YEAR_OLD_SAMPLES}
    assert windows == set(TWO_YEAR_OLD_WINDOWS)
    extras = {s[0].split(":")[1] for s in TWO_YEAR_OLD_SAMPLES} & {
        w[2] for w in TWO_YEAR_OLD_WINDOWS if w[2]
    }
    assert extras == {"2017-12-28", "2018-12-28", "2019-12-28", "2021-12-28"}
    # the samples run the real calendar, so this one fails until it ships the windows
    first = TWO_YEAR_OLD_SAMPLES[0]
    assert race_takeout(first[0], "WIN", race_class_raw=first[1]).payout_rate == pytest.approx(0.85)


@pytest.mark.parametrize("two, two_class, other, other_class", TWO_YEAR_OLD_SAMPLES)
def test_shipped_two_year_old_race_pays_five_percent_more_on_win(
    two: str, two_class: str, other: str, other_class: str
) -> None:
    day = date.fromisoformat(two.split(":")[1])
    super_day = day in SUPER_PREMIUM_DAYS  # 12-28 of 2017-2019 and 2021 are Super Premium too

    win = race_takeout(two, "WIN", race_class_raw=two_class)
    assert win.base_rate == pytest.approx(0.80)
    assert win.payout_rate == pytest.approx(0.85)
    assert win.boost == ("ultra_premium" if super_day else "premium")
    assert win.boost_may_be_withheld is True

    place = race_takeout(two, "PLACE", race_class_raw=two_class)
    assert place.payout_rate == pytest.approx(0.80)
    assert place.boost == ("super_premium" if super_day else None)
    assert place.boost_may_be_withheld is False

    older = race_takeout(other, "WIN", race_class_raw=other_class)
    assert older.payout_rate == pytest.approx(0.80)
    assert older.boost == ("super_premium" if super_day else None)
    assert older.boost_may_be_withheld is False

    with pytest.raises(conditions_needed()):
        race_takeout(two, "WIN")


# 2-year-old races on days no 2歳単勝 window covers: base rate, and no conditions needed.
TWO_YEAR_OLD_OUTSIDE = [
    ("jra:2014-07-26:CKY:05", "2歳新馬"),  # no campaign in 2014
    ("jra:2015-09-12:HSN:01", "2歳未勝利"),  # no autumn window in 2015
    ("jra:2016-12-17:CKY:05", "2歳新馬"),  # nor in 2016
    ("jra:2017-12-23:HSN:01", "2歳未勝利"),  # between the autumn window and 12-28
    ("jra:2018-12-22:HSN:01", "2歳未勝利"),
    ("jra:2019-12-21:HSN:01", "2歳未勝利"),
    ("jra:2019-12-22:HSN:01", "2歳未勝利"),
    ("jra:2020-12-26:NAK:11", "2歳オープン"),  # 2020 had no 12-28 day
    ("jra:2021-12-25:HSN:01", "2歳未勝利"),
    ("jra:2022-12-24:HSN:01", "2歳未勝利"),
    ("jra:2023-12-23:HSN:01", "2歳未勝利"),
    ("jra:2024-06-01:TOK:05", "2歳新馬"),  # the campaign ended in 2023
]


@pytest.mark.parametrize("race_id, label", TWO_YEAR_OLD_OUTSIDE)
def test_shipped_two_year_old_race_outside_the_windows_pays_the_base_rate(
    race_id: str, label: str
) -> None:
    got = race_takeout(race_id, "WIN", race_class_raw=label)
    assert got.payout_rate == pytest.approx(0.80)
    assert got.boost is None
    assert race_takeout(race_id, "WIN").payout_rate == pytest.approx(0.80)


# --- the shipped calendar: 3歳重賞 (2017-2023), 金杯 (2020-2023), JRA60 (2014) (#197) ---


# Per year: the boosted pool and every race the campaign page names ("MMDD:VENUE:race number";
# date and venue from https://www.jra.go.jp/kouza/premium/3sai-spring/<year>.html, race numbers
# from the local store). 2021-2023 include the listed races (L).
THREE_YEAR_OLD_STAKES = {
    2017: (
        "QUINELLA",
        "0304:HSN:11 0305:NAK:11 0312:HSN:11 0318:CKY:11 0319:NAK:11 0320:NAK:11 0325:HSN:11 "
        "0408:NAK:11 0409:HSN:11 0416:NAK:11 0423:TOK:11 0429:TOK:11 0506:KYO:11 0507:TOK:11 "
        "0521:TOK:11",
    ),
    2018: (
        "QUINELLA",
        "0107:NAK:11 0108:KYO:11 0114:NAK:11 0204:KYO:11 0211:TOK:11 0212:TOK:11 0303:HSN:11 "
        "0304:NAK:11 0311:HSN:11 0317:NAK:11 0317:CKY:11 0318:NAK:11 0324:HSN:11 0407:NAK:11 "
        "0408:HSN:11 0414:HSN:11 0415:NAK:11 0422:TOK:11 0428:TOK:11 0505:KYO:11 0506:TOK:11 "
        "0520:TOK:11 0526:KYO:11",
    ),
    2019: (
        "QUINELLA",
        "0106:KYO:11 0112:NAK:11 0114:NAK:11 0203:KYO:11 0210:TOK:11 0211:TOK:11 0302:HSN:11 "
        "0303:NAK:11 0310:HSN:11 0316:NAK:11 0316:CKY:11 0317:NAK:11 0323:HSN:11 0406:NAK:11 "
        "0407:HSN:11 0413:HSN:11 0414:NAK:11 0421:TOK:11 0427:TOK:11 0504:KYO:11 0505:TOK:11 "
        "0519:TOK:11 0525:KYO:11",
    ),
    2020: (
        "QUINELLA",
        "0112:KYO:11 0113:NAK:11 0119:NAK:11 0209:KYO:11 0215:TOK:11 0216:TOK:11 0307:HSN:11 "
        "0308:NAK:11 0314:CKY:11 0315:HSN:11 0320:NAK:11 0322:NAK:11 0328:HSN:11 0411:NAK:11 "
        "0412:HSN:11 0418:HSN:11 0419:NAK:11 0426:TOK:11 0502:TOK:11 0509:KYO:11 0510:TOK:11 "
        "0524:TOK:11 0530:KYO:11",
    ),
    2021: (
        "QUINELLA",
        "0110:CKY:11 0111:NAK:11 0116:CKY:10 0117:NAK:11 0123:CKY:10 0130:TOK:10 0206:CKY:10 "
        "0207:CKY:11 0213:TOK:11 0214:TOK:11 0221:TOK:9 0227:HSN:10 0228:HSN:10 0306:HSN:11 "
        "0307:NAK:11 0314:NAK:11 0314:HSN:11 0320:NAK:11 0320:HSN:11 0320:CKY:11 0321:NAK:11 "
        "0327:HSN:11 0410:NAK:11 0411:HSN:9 0411:HSN:11 0417:HSN:11 0418:NAK:11 0425:TOK:11 "
        "0501:TOK:11 0502:TOK:11 0508:TOK:11 0508:CKY:11 0509:TOK:11 0509:CKY:10 0523:TOK:11 "
        "0523:CKY:10 0529:CKY:11",
    ),
    2022: (
        "QUINELLA_PLACE",
        "0105:NAK:10 0109:CKY:11 0110:NAK:11 0115:CKY:10 0116:NAK:11 0122:CKY:10 0129:TOK:10 "
        "0205:CKY:10 0206:CKY:11 0212:TOK:11 0213:TOK:11 0220:TOK:9 0226:HSN:10 0227:HSN:10 "
        "0305:HSN:11 0306:NAK:11 0313:NAK:11 0313:HSN:11 0319:CKY:11 0319:HSN:11 0320:NAK:11 "
        "0321:NAK:11 0326:HSN:11 0409:NAK:11 0410:HSN:9 0410:HSN:11 0416:HSN:11 0417:NAK:11 "
        "0424:TOK:11 0430:TOK:11 0501:TOK:11 0507:TOK:11 0507:CKY:11 0508:TOK:11 0508:CKY:10 "
        "0522:TOK:11 0522:CKY:10 0528:CKY:11",
    ),
    2023: (
        "QUINELLA_PLACE",
        "0105:NAK:10 0108:CKY:11 0109:NAK:11 0115:NAK:11 0115:CKY:10 0121:CKY:10 0128:TOK:10 "
        "0204:CKY:10 0205:CKY:11 0211:TOK:11 0212:TOK:11 0219:TOK:9 0225:HSN:9 0226:HSN:10 "
        "0304:HSN:11 0305:NAK:11 0312:NAK:11 0312:HSN:11 0318:NAK:11 0318:HSN:11 0318:CKY:11 "
        "0319:NAK:11 0325:HSN:11 0408:NAK:11 0409:HSN:9 0409:HSN:11 0415:HSN:11 0416:NAK:11 "
        "0423:TOK:11 0429:TOK:11 0430:TOK:11 0506:TOK:11 0506:KYO:11 0507:TOK:11 0507:KYO:10 "
        "0521:TOK:11 0521:KYO:10 0527:KYO:11",
    ),
}
THREE_YEAR_OLD_COUNTS = {2017: 15, 2018: 23, 2019: 23, 2020: 23, 2021: 37, 2022: 38, 2023: 38}

# 中山金杯 and 京都金杯, 11R each; 京都金杯 ran at Chukyo 2021-2023.
KINPAI = {
    "jra:2020-01-05:NAK:11": "QUINELLA",
    "jra:2020-01-05:KYO:11": "QUINELLA",
    "jra:2021-01-05:NAK:11": "QUINELLA",
    "jra:2021-01-05:CKY:11": "QUINELLA",
    "jra:2022-01-05:NAK:11": "QUINELLA_PLACE",
    "jra:2022-01-05:CKY:11": "QUINELLA_PLACE",
    "jra:2023-01-05:NAK:11": "QUINELLA_PLACE",
    "jra:2023-01-05:CKY:11": "QUINELLA_PLACE",
}
KINPAI_2017_2019 = [
    f"jra:{d}:{v}:11" for d in ("2017-01-05", "2018-01-06", "2019-01-05") for v in ("NAK", "KYO")
]

# JRA60周年記念競走, Quinella, 11R each. The page's tenth race, 2014-04-20 NAK 11, predates
# 2014-06-07, where the base rate is unknown, so the calendar leaves it out.
JRA60 = [
    "jra:2014-06-29:HSN:11",
    "jra:2014-07-13:FKS:11",
    "jra:2014-07-20:HAK:11",
    "jra:2014-08-10:KOK:11",
    "jra:2014-08-24:SAP:11",
    "jra:2014-09-07:NII:11",
    "jra:2014-10-26:KYO:11",
    "jra:2014-11-02:TOK:11",
    "jra:2014-12-07:CKY:11",
]
OTHER_PAIR = {"QUINELLA": "QUINELLA_PLACE", "QUINELLA_PLACE": "QUINELLA"}


def _three_year_old_races() -> dict[str, str]:
    races = {}
    for year, (pool, listed) in THREE_YEAR_OLD_STAKES.items():
        for item in listed.split():
            md, venue, number = item.split(":")
            races[f"jra:{year}-{md[:2]}-{md[2:]}:{venue}:{int(number)}"] = pool
    return races


def _key(race_id: str, pool: str) -> tuple[str, str, int, tuple[str, ...]]:
    _, day, venue, number = race_id.split(":")
    return (day, venue, int(number), (pool,))


def _per_race_premium(shipped, first: date, last: date) -> list[tuple[str, str, int, tuple]]:
    return [
        (e.date.isoformat(), e.venue, e.race_no, tuple(e.pools) if e.pools else None)
        for e in shipped.entries
        if e.kind == "premium" and e.race_no is not None and first <= e.date <= last
    ]


def test_the_three_year_old_stakes_table_matches_the_page_counts() -> None:
    races = _three_year_old_races()
    by_year = {y: sum(1 for r in races if r[4:8] == str(y)) for y in THREE_YEAR_OLD_COUNTS}
    assert by_year == THREE_YEAR_OLD_COUNTS
    assert len(races) == 197
    # and the shipped calendar boosts them, so this fails until it does
    rid, pool = next(iter(races.items()))
    assert race_takeout(rid, pool).payout_rate == pytest.approx(0.825)


@pytest.mark.parametrize("year", sorted(THREE_YEAR_OLD_STAKES))
def test_shipped_three_year_old_stakes_pay_five_percent_more_on_the_campaign_pool(
    year: int,
) -> None:
    pool, _ = THREE_YEAR_OLD_STAKES[year]
    races = {r: p for r, p in _three_year_old_races().items() if r[4:8] == str(year)}
    assert races
    for rid in races:
        got = race_takeout(rid, pool)
        assert got.base_rate == pytest.approx(0.775), rid
        assert got.payout_rate == pytest.approx(0.825), rid
        assert got.boost == "premium", rid
        assert got.boost_may_be_withheld is True, rid
        other = race_takeout(rid, OTHER_PAIR[pool])
        assert other.payout_rate == pytest.approx(0.775), rid
        assert other.boost is None, rid
        assert race_takeout(rid, "WIN").payout_rate == pytest.approx(0.80), rid
        assert race_takeout(rid, "TRIO").payout_rate == pytest.approx(0.75), rid


@pytest.mark.parametrize("race_id", sorted(KINPAI))
def test_shipped_kinpai_2020_2023_pay_five_percent_more_on_the_campaign_pool(
    race_id: str,
) -> None:
    pool = KINPAI[race_id]
    got = race_takeout(race_id, pool)
    assert got.base_rate == pytest.approx(0.775)
    assert got.payout_rate == pytest.approx(0.825)
    assert got.boost == "premium"
    other = race_takeout(race_id, OTHER_PAIR[pool])
    assert other.payout_rate == pytest.approx(0.775)
    assert other.boost is None
    assert race_takeout(race_id, "WIN").payout_rate == pytest.approx(0.80)


def test_shipped_per_race_premiums_2017_2023_are_exactly_3sai_and_kinpai() -> None:
    expected = [_key(r, p) for r, p in _three_year_old_races().items()]
    expected += [_key(r, p) for r, p in KINPAI.items()]
    expected += [_key(r, "QUINELLA") for r in KINPAI_2017_2019]
    assert len(expected) == len(set(expected)) == 211
    got = _per_race_premium(load_boosts(), date(2017, 1, 1), date(2023, 12, 31))
    assert sorted(got) == sorted(expected)


@pytest.mark.parametrize("race_id", JRA60)
def test_shipped_jra60_races_pay_five_percent_more_on_quinella(race_id: str) -> None:
    got = race_takeout(race_id, "QUINELLA")
    assert got.base_rate == pytest.approx(0.775)
    assert got.payout_rate == pytest.approx(0.825)
    assert got.boost == "premium"
    assert got.boost_may_be_withheld is True
    for pool in ("QUINELLA_PLACE", "WIN", "TRIO"):
        other = race_takeout(race_id, pool)
        assert other.payout_rate == pytest.approx(1 - TAKEOUT[pool]), pool
        assert other.boost is None, pool


def test_shipped_2014_per_race_premiums_are_jra60_and_the_last_race_campaign() -> None:
    shipped = load_boosts()
    got = _per_race_premium(shipped, date(2014, 1, 1), date(2014, 12, 31))
    not_12r = sorted(k for k in got if k[2] != 12)
    assert not_12r == sorted(_key(r, "QUINELLA") for r in JRA60)
    # nothing is listed where the base rate is unknown: the JRA60 race of 2014-04-20 stays out
    assert min(e.date for e in shipped.entries) >= date(2014, 6, 7)
    with pytest.raises(TakeoutUnknown):
        race_takeout("jra:2014-04-20:NAK:11", "QUINELLA")


# --- the shipped calendar against the store (local) ---


@pytest.fixture
def store_races() -> list[tuple[str, date, str]]:
    try:
        path = paths.processed_dir("jra") / "races.parquet"
    except paths.DataDirNotSet:
        pytest.skip(f"{paths.DATA_DIR_VAR} is not set: the race store is unavailable")
    if not path.is_file():
        pytest.skip(f"race store missing: {path}")
    import pandas as pd

    frame = pd.read_parquet(path, columns=["race_id", "race_date", "race_class_raw"])
    return [
        (rid, d if isinstance(d, date) else date.fromisoformat(str(d)[:10]), cls)
        for rid, d, cls in frame.itertuples(index=False)
    ]


@pytest.mark.local
def test_the_store_races_boosted_in_each_window_match_the_page_counts(store_races) -> None:
    # 2019 autumn is left out: the page states 439 races and the store holds 440 2-year-old races.
    windows = [w for w in TWO_YEAR_OLD_WINDOWS if w[0] != "2019-09-07"]
    expected = {w[0]: w[3] for w in windows}
    got = {w[0]: 0 for w in windows}
    for rid, day, cls in store_races:
        window = _window_of(day)
        if window is None or window[0] not in got:
            continue
        result = race_takeout(rid, "WIN", race_class_raw=cls)
        if result.boost in ("premium", "ultra_premium"):
            got[window[0]] += 1
            assert result.payout_rate == pytest.approx(0.85), rid
            assert str(cls).startswith("2歳") and "歳以上" not in str(cls), (rid, cls)
    assert got == expected
