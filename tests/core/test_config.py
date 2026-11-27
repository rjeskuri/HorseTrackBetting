"""The JRA source config: facts carried as data in countries/japan/config/jra.yaml."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from horsetrack.core.config import Pool, SourceConfig, load_country_yaml, load_source

ENABLED = {"WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE"}

# Venue facts, written out independently of the YAML: code -> (netkeiba number, printed name).
VENUES = {
    "SAP": ("01", "札幌"),
    "HAK": ("02", "函館"),
    "FKS": ("03", "福島"),
    "NII": ("04", "新潟"),
    "TOK": ("05", "東京"),
    "NAK": ("06", "中山"),
    "CKY": ("07", "中京"),
    "KYO": ("08", "京都"),
    "HSN": ("09", "阪神"),
    "KOK": ("10", "小倉"),
}


@pytest.fixture(autouse=True)
def _fresh_caches():
    load_source.cache_clear()
    load_country_yaml.cache_clear()
    yield
    load_source.cache_clear()
    load_country_yaml.cache_clear()


@pytest.fixture
def jra() -> SourceConfig:
    return load_source("japan", "jra")


@pytest.fixture
def raw() -> dict:
    return load_country_yaml("japan", "jra")


def _code(found) -> str | None:
    """A lookup result is either the code itself or an object carrying `.code`."""
    if found is None or isinstance(found, str):
        return found
    return found.code


def _mappings_with_key(node, key):
    """Every value stored under `key` anywhere in a nested dict/list."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                yield v
            yield from _mappings_with_key(v, key)
    elif isinstance(node, list):
        for item in node:
            yield from _mappings_with_key(item, key)


def test_load_jra_source_config(jra, raw):
    assert jra.source == "jra"
    assert jra.timezone == "Asia/Tokyo"
    assert jra.currency == "JPY"
    assert jra.unit_stake == 100
    assert raw["odds_format"] == "decimal_incl_stake"
    assert raw["hosts"]["official"] == "https://www.jra.go.jp"
    assert raw["hosts"]["db"] == "https://db.netkeiba.com"
    assert raw["hosts"]["race"] == "https://race.netkeiba.com"

    # the four pools are enabled, nothing else is; each pool knows its own code
    assert {code for code, pool in jra.pools.items() if pool.enabled} == ENABLED
    for code, pool in jra.pools.items():
        assert pool.code == code

    # all ten venues are there, with the codes and numbers the parsers rely on
    assert set(jra.venues) >= set(VENUES)
    assert set(raw["venues"]) >= set(VENUES)
    for code, (number, name) in VENUES.items():
        assert raw["venues"][code]["number"] == number
        assert raw["venues"][code]["name"] == name


@pytest.mark.parametrize(
    ("code", "labels", "legs", "kind", "payout_rule", "takeout"),
    [
        ("WIN", {"単勝", "WIN"}, 1, "win", "stake_back_share", 0.20),
        ("PLACE", {"複勝", "PLACE"}, 1, "place", "stake_back_share", 0.20),
        ("QUINELLA", {"馬連", "QUINELLA"}, 2, "unordered_top2", "stake_back_share", 0.225),
        ("QUINELLA_PLACE", {"ワイド", "WIDE"}, 2, "any2_of_top3", "stake_back_share", 0.225),
    ],
)
def test_enabled_pool_facts(jra, code, labels, legs, kind, payout_rule, takeout):
    pool = jra.pools[code]
    assert labels <= set(pool.labels)
    assert pool.legs == legs
    assert pool.kind == kind
    assert pool.payout_rule == payout_rule
    assert pool.takeout == pytest.approx(takeout)
    assert pool.dividend_step == 10
    assert pool.min_dividend == 110


# 04 §1.2 (競馬法 第8条): every JRA single-race pool pays a winning ticket its stake back plus
# 1/P of the losers' money, P the number of winning combinations (owner, 2026-11-26, ticket #99).
# WIN5 stays as it is: its carryover is #135's.
SINGLE_RACE_POOLS = (
    "WIN",
    "PLACE",
    "BRACKET_QUINELLA",
    "QUINELLA",
    "QUINELLA_PLACE",
    "FORECAST",
    "TRIO",
    "TIERCE",
)


def test_every_single_race_pool_pays_stake_back_plus_a_share_of_the_losers(jra, raw):
    rules = {code: jra.pools[code].payout_rule for code in (*SINGLE_RACE_POOLS, "WIN5")}
    assert rules == {**dict.fromkeys(SINGLE_RACE_POOLS, "stake_back_share"), "WIN5": "equal_split"}
    # jra.yaml carries the rule on each pool; the model's default is not what sets it.
    written = {code: raw["pools"][code].get("payout_rule") for code in SINGLE_RACE_POOLS}
    assert written == dict.fromkeys(SINGLE_RACE_POOLS, "stake_back_share")


@pytest.mark.parametrize(
    ("label", "code"),
    [
        ("単勝", "WIN"),
        ("複勝", "PLACE"),
        ("馬連", "QUINELLA"),
        ("ワイド", "QUINELLA_PLACE"),
        ("WIN", "WIN"),
        ("win", "WIN"),
        ("PLACE", "PLACE"),
        ("Place", "PLACE"),
        ("QUINELLA", "QUINELLA"),
        ("quinella", "QUINELLA"),
        ("WIDE", "QUINELLA_PLACE"),
        ("wide", "QUINELLA_PLACE"),
        ("Wide", "QUINELLA_PLACE"),
    ],
)
def test_pool_by_label_matches_japanese_and_english_labels(jra, label, code):
    found = jra.pool_by_label(label)
    assert found is not None
    assert found.code == code


def test_pool_by_label_is_none_for_an_unknown_label(jra):
    assert jra.pool_by_label("not a pool") is None
    assert jra.pool_by_label("") is None


@pytest.mark.parametrize(("code", "name"), [(c, n) for c, (_, n) in VENUES.items()])
def test_venue_by_name_maps_the_printed_name_to_the_code(jra, code, name):
    assert _code(jra.venue_by_name(name)) == code


def test_venue_by_name_is_none_for_a_name_that_is_not_a_jra_course(jra):
    assert jra.venue_by_name("沙田") is None
    assert jra.venue_by_name("Kyoto racecourse") is None


def test_an_unknown_pool_key_is_refused(jra):
    spec = jra.pools["WIN"].model_dump()
    Pool(**spec)  # the untouched spec is valid, so the extra key is what fails
    with pytest.raises(ValidationError):
        Pool(**{**spec, "taekout": 0.2})

    whole = jra.model_dump()
    SourceConfig.model_validate(whole)
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({**whole, "mystery_block": {}})
    pools = {**whole["pools"], "WIN": {**whole["pools"]["WIN"], "typo_key": 1}}
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({**whole, "pools": pools})


@pytest.mark.parametrize(
    ("code", "when", "expected"),
    [
        ("WIN", date(2014, 6, 7), 0.20),
        ("WIN", date(2014, 6, 8), 0.20),
        ("WIN", date(2026, 11, 24), 0.20),
        ("WIN", date(2014, 6, 6), None),
        ("WIN", date(2010, 1, 1), None),
        ("PLACE", date(2020, 1, 1), 0.20),
        ("PLACE", date(2013, 12, 31), None),
        ("QUINELLA", date(2014, 6, 7), 0.225),
        ("QUINELLA", date(2025, 5, 5), 0.225),
        ("QUINELLA", date(2014, 6, 6), None),
        ("QUINELLA_PLACE", date(2022, 2, 2), 0.225),
        ("QUINELLA_PLACE", date(2011, 3, 11), None),
    ],
)
def test_takeout_is_unknown_before_2014_06_07_and_20_percent_after(jra, code, when, expected):
    got = jra.pools[code].takeout_on(when)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


@pytest.mark.parametrize(
    ("code", "starters", "places"),
    [
        ("PLACE", 18, 3),
        ("PLACE", 9, 3),
        ("PLACE", 8, 3),
        ("PLACE", 7, 2),
        ("PLACE", 6, 2),
        ("PLACE", 5, 2),
        ("PLACE", 4, 0),
        ("PLACE", 3, 0),
        ("PLACE", 0, 0),
        ("QUINELLA_PLACE", 18, 3),
        ("QUINELLA_PLACE", 8, 3),
        ("QUINELLA_PLACE", 5, 3),
        ("QUINELLA_PLACE", 4, 3),
        ("QUINELLA_PLACE", 3, 0),
    ],
)
def test_places_paid_by_starters(jra, code, starters, places):
    assert jra.pools[code].places_for(starters) == places


def test_http_block_has_a_rate_for_www_jra_go_jp(jra, raw):
    assert "http" in raw
    http = jra.model_dump()["http"]

    # one rate per host, 1 request a second with 10 % jitter
    for host in ("www.jra.go.jp", "db.netkeiba.com", "race.netkeiba.com"):
        entries = list(_mappings_with_key(http, host))
        assert entries, host
        assert entries[0]["requests_per_second"] == pytest.approx(1.0)
        assert entries[0]["jitter"] == pytest.approx(0.1)

    # the default for any other host keeps the wider 30 % jitter
    assert any(
        isinstance(m, dict) and m.get("jitter") == pytest.approx(0.3)
        for m in _mappings_with_key(http, "default")
    )

    # circuit breaker: stop on 403 and 429, and after 5 consecutive failures
    breakers = list(_mappings_with_key(http, "stop_on_status"))
    assert [sorted(b) for b in breakers] == [[403, 429]]
    failures = list(_mappings_with_key(http, "max_consecutive_failures"))
    assert failures == [5]

    # the user agent is not configured here
    assert not list(_mappings_with_key(http, "user_agent"))


@pytest.mark.parametrize("block", ["page", "move_model", "features", "class_levels"])
def test_no_hktrack_only_block_is_carried_over(jra, raw, block):
    assert block not in raw
    assert block not in SourceConfig.model_fields
    assert block not in jra.model_dump()


def test_loaders_are_cached_and_cache_clear_drops_them():
    first = load_source("japan", "jra")
    assert load_source("japan", "jra") is first
    load_source.cache_clear()
    assert load_source("japan", "jra") is not first

    raw_first = load_country_yaml("japan", "jra")
    assert load_country_yaml("japan", "jra") is raw_first
    load_country_yaml.cache_clear()
    assert load_country_yaml("japan", "jra") is not raw_first
