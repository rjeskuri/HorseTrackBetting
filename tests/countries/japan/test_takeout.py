"""Per-race takeout with the JRA Premium boosts (ticket #76).

Contract pinned here (`horsetrack.countries.japan.takeout`):
- `load_boosts(path=None) -> BoostCalendar`; no path loads the shipped
  `countries/japan/config/takeout_boosts.yaml`. The file is a mapping with one key, `boosts`, a
  list of entries `{kind: premium | super_premium | ultra_premium, date, venue?, race_no?, pools?}`;
  a missing venue, race_no or pools means every venue, race or pool that day. An unknown kind
  fails to load.
- `BoostCalendar.entries`: the loaded entries, each with `.kind` (str) and `.date` (datetime.date),
  `.venue`, `.race_no`, `.pools`.
- `race_takeout(race_id, pool, *, calendar=None) -> RaceTakeout(pool, base_rate, payout_rate, boost,
  boost_may_be_withheld)`; calendar None means the shipped calendar.
  - base_rate: the base payout rate, 1 - Pool.takeout_on(date) from jra.yaml.
  - ordinary race: payout_rate == base_rate, boost None, boost_may_be_withheld False.
  - premium: payout_rate = base_rate + 0.05, boost "premium", boost_may_be_withheld True.
  - super_premium: payout_rate 0.80 on every pool, boost "super_premium".
  - ultra_premium: payout_rate 0.85 on every pool, boost "ultra_premium".
  - a date before 2014-06-07 raises TakeoutUnknown; an unknown pool code or a malformed race id
    raises ValueError or KeyError.
Pool codes are #20's: TIERCE is the Trifecta, FORECAST the Exacta, QUINELLA_PLACE the Wide.
Every calendar here is synthetic and written to tmp_path, except the shipped-calendar test.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from horsetrack.countries.japan.takeout import (
    TakeoutUnknown,
    load_boosts,
    race_takeout,
)

from horsetrack.core import config

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


def test_the_shipped_calendar_has_20_super_premium_days() -> None:
    shipped = load_boosts()
    days = [e.date for e in shipped.entries if e.kind == "super_premium"]
    assert len(days) == 20
    assert len(set(days)) == 20
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
