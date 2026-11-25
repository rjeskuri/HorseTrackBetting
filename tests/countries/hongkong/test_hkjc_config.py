"""The HKJC source config (ticket #60): countries/hongkong/config/hkjc.yaml, the history-relevant
part of HKTrack's hkjc.yaml, loaded through `core.config.load_source`.

Expected facts are written out here from HKTrack's config, not read back from the YAML.
"""

from __future__ import annotations

from datetime import date

import pytest

from horsetrack.core.config import SourceConfig, load_country_yaml, load_source

# code -> (takeout today, legs, kind, dividend step, minimum dividend, enabled)
POOLS = {
    "WIN": (0.175, 1, "win", 0.5, 10.5, True),
    "PLACE": (0.175, 1, "place", 0.5, 10.1, True),
    "QUINELLA": (0.175, 2, "unordered_top2", 0.5, 10.5, True),
    "QUINELLA_PLACE": (0.175, 2, "any2_of_top3", 0.5, 10.5, True),
    "FORECAST": (0.195, 2, "ordered_top2", 1.0, 11.0, False),
    "TRIO": (0.23, 3, "unordered_top3", 1.0, 11.0, False),
    "TIERCE": (0.25, 3, "ordered_top3", 1.0, 11.0, False),
    "FIRST_4": (0.25, 4, "unordered_top4", 1.0, 11.0, False),
    "QUARTET": (0.25, 4, "ordered_top4", 1.0, 11.0, False),
    "DOUBLE": (0.175, 2, "multi_race_win", 0.5, 10.5, False),
}
VENUES = {"HV": "Happy Valley", "ST": "Sha Tin"}


@pytest.fixture(autouse=True)
def _fresh_caches():
    load_source.cache_clear()
    load_country_yaml.cache_clear()
    yield
    load_source.cache_clear()
    load_country_yaml.cache_clear()


@pytest.fixture
def hkjc() -> SourceConfig:
    return load_source("hongkong", "hkjc")


def _values(node) -> list:
    if isinstance(node, dict):
        return list(node.values())
    return list(node.model_dump().values())


def test_load_source_reads_the_hong_kong_facts(hkjc):
    assert isinstance(hkjc, SourceConfig)
    assert (hkjc.source, hkjc.timezone, hkjc.currency, hkjc.unit_stake, hkjc.odds_format) == (
        "hkjc",
        "Asia/Hong_Kong",
        "HKD",
        10,
        "decimal_incl_stake",
    )
    assert "https://racing.hkjc.com" in [str(v).rstrip("/") for v in _values(hkjc.hosts)]

    assert set(hkjc.venues) == set(VENUES)
    for code, name in VENUES.items():
        assert hkjc.venues[code].name == name
        assert hkjc.venue_by_name(name).code == code
    assert set(hkjc.pools) == set(POOLS)

    assert hkjc.odds_display == {
        "cap": 99,
        "threshold": 10,
        "detruncate_below": 0.025,
        "detruncate_at_or_above": 0.475,
    }
    assert hkjc.http.default.requests_per_second > 0
    assert set(hkjc.http.circuit_breaker.stop_on_status) == {403, 429}


@pytest.mark.parametrize("code", sorted(POOLS))
def test_every_pool_carries_its_rules(hkjc, code):
    takeout, legs, kind, step, minimum, enabled = POOLS[code]
    pool = hkjc.pools[code]
    assert (pool.takeout, pool.legs, pool.kind, pool.dividend_step, pool.min_dividend) == (
        takeout,
        legs,
        kind,
        step,
        minimum,
    )
    assert pool.enabled is enabled
    assert hkjc.pool_by_label(code.replace("_", " ").lower()).code == code


@pytest.mark.parametrize(
    ("code", "when", "rate"),
    [
        ("TRIO", date(2015, 6, 1), 0.25),
        ("TRIO", date(2020, 9, 6), 0.195),
        ("TRIO", date(2023, 8, 31), 0.195),
        ("TRIO", date(2023, 9, 1), 0.23),
        ("TIERCE", date(2022, 8, 31), 0.25),
        ("TIERCE", date(2022, 9, 1), 0.22),
        ("TIERCE", date(2024, 1, 1), 0.25),
        ("WIN", date(2012, 1, 1), 0.175),
    ],
)
def test_takeout_history_is_carried(hkjc, code, when, rate):
    assert hkjc.pools[code].takeout_on(when) == rate


@pytest.mark.parametrize(("starters", "places"), [(14, 3), (7, 3), (6, 2), (4, 2), (3, 0)])
def test_place_pays_by_the_number_of_starters(hkjc, starters, places):
    assert hkjc.pools["PLACE"].places_for(starters) == places


def test_the_yaml_keeps_season_and_eras_and_drops_the_page_model_and_betting_blocks():
    raw = load_country_yaml("hongkong", "hkjc")
    assert raw["season"]["start_month"] == 9
    assert raw["eras"]["weight_scale_change"] == date(2022, 6, 25)
    assert not {"page", "model", "betting"} & set(raw)
