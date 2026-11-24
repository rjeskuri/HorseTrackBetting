"""Pari-mutuel pool maths in core (ticket #68): facts about the pools, ported from HKTrack.

Expected values are worked out by hand from the pool rules (HKTrack betting/market.py and
betting/pricing.py formulas), never by calling the code under test. All numbers are synthetic.
"""

from __future__ import annotations

import inspect
import math
import re
from functools import partial

import numpy as np
import pytest

from horsetrack.core import pricing
from horsetrack.core.config import load_country_yaml, load_source
from horsetrack.core.pricing import (
    clean_odds,
    detruncate,
    floor_bound,
    inverse_odds,
    overround,
    stake_back_share_dividend,
)

NAN = float("nan")

# Entries a board can carry that are not a price you can bet at: missing, unparseable, infinite,
# zero, negative, below evens-with-stake (1.0).
JUNK = [None, "abc", "", "-", NAN, float("inf"), 0.0, -2.0, 0.5, 0.99]

# A display rule in the config's `odds_display` shape: threshold, bump below it, bump at or above.
RULE = {"cap": 999.9, "threshold": 5.0, "detruncate_below": 0.03, "detruncate_at_or_above": 0.2}
FLAT_RULE = {
    "cap": 9999.9,
    "threshold": 0.0,
    "detruncate_below": 0.05,
    "detruncate_at_or_above": 0.05,
}


def _same(actual, expected) -> None:
    """Element-wise equality where NaN matches NaN."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, (actual, expected)
    assert np.array_equal(np.isnan(actual), np.isnan(expected)), (actual, expected)
    ok = ~np.isnan(expected)
    np.testing.assert_allclose(actual[ok], expected[ok], rtol=0, atol=1e-12)


# --- clean_odds


@pytest.mark.parametrize(
    ("odds", "expected"),
    [
        ([2.0, None, 3.5, 0.9, 1.0], [2.0, NAN, 3.5, NAN, 1.0]),
        ([12.3, "abc", 0.0, -4.0, float("inf"), 1.4], [12.3, NAN, NAN, NAN, NAN, 1.4]),
        ((1.5, NAN, 99.9), [1.5, NAN, 99.9]),
        (np.array([0.5, 7.7, 1.0, 0.999]), [NAN, 7.7, 1.0, NAN]),
        (["5.5", "2.0", "x", ""], [5.5, 2.0, NAN, NAN]),  # numeric text parses, like HKTrack
        ([], []),
        (JUNK, [NAN] * len(JUNK)),
    ],
)
def test_clean_odds_drops_missing_and_sub_one_prices(odds, expected):
    out = clean_odds(odds)
    assert isinstance(out, np.ndarray)
    assert out.dtype.kind == "f"
    _same(out, expected)


def test_clean_odds_leaves_its_input_untouched():
    board = np.array([0.5, 2.0, np.nan, 4.0])
    clean_odds(board, detruncate=lambda o: o + 1.0)
    _same(board, [0.5, 2.0, NAN, 4.0])


@pytest.mark.parametrize(
    ("odds", "rule", "expected"),
    [
        # Below the threshold gets one bump, at or above it the other; junk stays NaN, never bumped.
        ([2.0, 4.9, 5.0, 12.3, None, 0.5], RULE, [2.03, 4.93, 5.2, 12.5, NAN, NAN]),
        ([1.0, 37.2, "x", 3.0], FLAT_RULE, [1.05, 37.25, NAN, 3.05]),
        # The sub-one check is on the displayed figure: 0.98 + 0.05 would clear 1.0 but stays NaN.
        ([0.98, 1.0], FLAT_RULE, [NAN, 1.05]),
    ],
)
def test_clean_odds_applies_a_detruncate_rule_to_valid_prices_only(odds, rule, expected):
    _same(clean_odds(odds, detruncate=partial(detruncate, rule=rule)), expected)


def test_clean_odds_takes_any_callable_as_the_detruncate_rule():
    _same(clean_odds([2.0, None, 10.0], detruncate=lambda o: o * 2), [4.0, NAN, 20.0])


# --- inverse_odds


@pytest.mark.parametrize(
    ("odds", "expected"),
    [
        ([2.0, 4.0, 5.0, 1.0], [0.5, 0.25, 0.2, 1.0]),
        ([8.0, None, 1.25, 0.9, "x"], [0.125, NAN, 0.8, NAN, NAN]),
        ((10.0, 2.5), [0.1, 0.4]),
        ([], []),
    ],
)
def test_inverse_odds_is_one_over_each_valid_price(odds, expected):
    out = inverse_odds(odds)
    assert isinstance(out, np.ndarray)
    _same(out, expected)


@pytest.mark.parametrize(
    ("odds", "total"),
    [
        ([2.0, 4.0, 4.0, 8.0, 8.0], 1.25),
        ([2.0, 2.0], 1.0),
        ([1.6, 4.0, 4.0], 1.125),
        ([3.0, 3.0], 2 / 3),
    ],
)
def test_inverse_odds_is_not_normalised(odds, total):
    # The raw shares keep the overround: they sum to 1 + overround, not to 1.
    assert float(np.nansum(inverse_odds(odds))) == pytest.approx(total, abs=1e-12)


# --- overround


def test_overround_of_a_complete_board():
    # 1/2 + 1/4 + 1/4 + 1/8 + 1/8 = 1.25
    assert overround([2.0, 4.0, 4.0, 8.0, 8.0]) == pytest.approx(0.25, abs=1e-12)


@pytest.mark.parametrize(
    ("odds", "expected"),
    [
        ([2.0, 2.0], 0.0),
        ([1.6, 4.0, 4.0], 0.125),
        ([2.5, 2.5, 5.0, 10.0], 0.1),
        ([3.0, 3.0, 3.0, 3.0], 1 / 3),
        ([4.0, 4.0, 4.0, 4.0, 4.0], 0.25),
        ([10.0, 10.0], -0.8),  # an incomplete board can sit under 1
    ],
)
def test_overround_over_a_spread_of_boards(odds, expected):
    out = overround(odds)
    assert isinstance(out, float)
    assert out == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("winners", "expected"),
    [
        # sum(1/odds) = 4 * 0.8 + 0.4 = 3.6
        (1, 2.6),
        (2, 0.8),
        (3, 0.2),
    ],
)
def test_overround_divides_by_the_paying_combinations(winners, expected):
    odds = [1.25, 1.25, 1.25, 1.25, 2.5]
    assert overround(odds, winners=winners) == pytest.approx(expected, abs=1e-12)


def test_overround_ignores_missing_and_sub_one_entries():
    # Only 2, 4, 4, 8, 8 count: 1.25 - 1.
    board = [2.0, None, 4.0, "x", 4.0, 0.5, 8.0, NAN, 8.0, 0.0]
    assert overround(board) == pytest.approx(0.25, abs=1e-12)


@pytest.mark.parametrize(
    ("odds", "winners"),
    [([], 1), ([None, "x", 0.5], 1), (JUNK, 3), ([2.0, 2.0], 0), ([2.0, 2.0], -1)],
)
def test_overround_is_nan_without_valid_odds_or_paying_combinations(odds, winners):
    out = overround(odds, winners=winners)
    assert isinstance(out, float)
    assert math.isnan(out)


# --- floor_bound


@pytest.mark.parametrize(
    ("odds", "floor", "expected"),
    [
        ([1.1, 3.4, 8.0], 1.1, True),  # at the floor
        ([5.0, 1.0, 3.4], 1.1, True),  # under it
        ([1.2, 3.4, 8.0], 1.1, False),
        ([1.2, 3.0], 1.2, True),  # the floor moves the answer
        ([1.3, 3.0], 1.2, False),
        ([2.0, 11 * 0.1], 110 / 100, True),  # 1.1000000000000003 is the floor, float fuzz aside
        ([None, 3.4, NAN, "x"], 1.1, False),  # junk is not a price at the floor
        ([0.9, 2.0], 1.1, False),  # a sub-one figure is dropped, not read as under the floor
        ([], 1.1, False),
        (JUNK, 1.1, False),
    ],
)
def test_floor_bound_flags_a_price_at_the_floor(odds, floor, expected):
    out = floor_bound(odds, floor)
    assert isinstance(out, bool)
    assert out is expected


# --- stake_back_share_dividend


def test_stake_back_share_dividend_follows_the_jra_rule():
    # D = 1 - 0.2 - 0.1 - 0.3 = 0.4, P = 3: 100 * 0.8 * (1 + 0.4 / (3 * 0.2)) = 133.33 -> 130
    out = stake_back_share_dividend(
        0.2, [0.1, 0.3], rate=0.8, unit=100, dividend_step=10, min_dividend=110
    )
    assert out == pytest.approx(130.0, abs=1e-9)


@pytest.mark.parametrize(
    ("share", "co_shares", "rate", "unit", "step", "minimum", "expected"),
    [
        # D = 0.5, P = 1: 80 * 2 = 160
        (0.5, [], 0.8, 100, 10, 110, 160.0),
        # D = 0.7, P = 1: 80 * (1 + 0.7 / 0.3) = 266.67; the step decides 260 or 266
        (0.3, [], 0.8, 100, 10, 110, 260.0),
        (0.3, [], 0.8, 100, 1, 110, 266.0),
        # the unit scales it: 533.33 -> 530
        (0.3, [], 0.8, 200, 10, 110, 530.0),
        # the rate scales it: 75 * (1 + 0.7 / 0.3) = 250
        (0.3, [], 0.75, 100, 10, 110, 250.0),
        # D = 0.7, P = 2: 75 * (1 + 0.7 / 0.2) = 337.5 -> 330
        (0.1, [0.2], 0.75, 100, 10, 110, 330.0),
        # one co-winner holding 0.4: D = 0.4, P = 2: 80 * (1 + 0.4 / (2 * 0.2)) = 160
        (0.2, [0.4], 0.8, 100, 10, 110, 160.0),
        # D = 0.45, P = 4: 80 * (1 + 0.45 / (4 * 0.25)) = 116 -> 110 by the step, not the minimum
        (0.25, [0.1, 0.1, 0.1], 0.8, 100, 10, 100, 110.0),
        # 124.99999999999999 in floats is 125 on the board, not 124
        (0.3, [0.3], 0.75, 100, 1, None, 125.0),
    ],
)
def test_stake_back_share_dividend_over_a_spread_of_tickets(
    share, co_shares, rate, unit, step, minimum, expected
):
    out = stake_back_share_dividend(
        share, co_shares, rate=rate, unit=unit, dividend_step=step, min_dividend=minimum
    )
    assert isinstance(out, float)
    assert out == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    ("share", "minimum", "expected"),
    [
        # D = 0.1, P = 1: 80 * (1 + 0.1 / 0.9) = 88.9 -> 80, lifted to the minimum
        (0.9, 110, 110.0),
        (0.9, 100, 100.0),
        (1.0, 110, 110.0),  # the whole pool on one ticket: 80 -> 110
        (0.9, None, 80.0),  # no minimum declared: the truncated figure stands
    ],
)
def test_the_dividend_never_falls_below_the_minimum(share, minimum, expected):
    out = stake_back_share_dividend(
        share, [], rate=0.8, unit=100, dividend_step=10, min_dividend=minimum
    )
    assert out == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("share", [0.0, -0.1, NAN])
def test_no_share_gives_nan(share):
    out = stake_back_share_dividend(
        share, [0.2], rate=0.8, unit=100, dividend_step=10, min_dividend=110
    )
    assert math.isnan(out)


# --- detruncate


def test_detruncate_needs_an_explicit_rule():
    rule = inspect.signature(detruncate).parameters["rule"]
    assert rule.default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        detruncate(5.0)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("odds", "rule", "expected"),
    [
        (2.0, RULE, 2.03),
        (4.9, RULE, 4.93),
        (5.0, RULE, 5.2),  # at the threshold takes the at-or-above bump
        (12.3, RULE, 12.5),
        (1.0, FLAT_RULE, 1.05),
        (37.2, FLAT_RULE, 37.25),
        (2.0, {"threshold": 10.0, "detruncate_below": 0.0, "detruncate_at_or_above": 0.5}, 2.0),
        (10.0, {"threshold": 10.0, "detruncate_below": 0.0, "detruncate_at_or_above": 0.5}, 10.5),
    ],
)
def test_detruncate_with_an_explicit_rule(odds, rule, expected):
    out = detruncate(odds, rule)
    assert isinstance(out, float)
    assert out == pytest.approx(expected, abs=1e-12)


def test_detruncate_takes_the_config_odds_display_rule():
    load_source.cache_clear()
    load_country_yaml.cache_clear()
    rule = load_source("japan", "jra").odds_display
    for shown in (1.0, 3.7, 25.0, 140.2):
        bump = (
            rule["detruncate_below"]
            if shown < rule["threshold"]
            else rule["detruncate_at_or_above"]
        )
        assert detruncate(shown, rule) == pytest.approx(shown + bump, abs=1e-12)


# --- definition of done 2: no odds-to-probability method

_METHOD_NAME = re.compile(
    r"prob|implied|normali[sz]|beta|power|harville|shin|softmax", re.IGNORECASE
)
_METHOD_PARAM = {"beta", "power", "method", "normalise", "normalize", "exponent", "temperature"}


def test_no_function_turns_odds_into_probabilities_with_a_chosen_method():
    public = {
        name: obj
        for name, obj in vars(pricing).items()
        if callable(obj)
        and not name.startswith("_")
        and getattr(obj, "__module__", None) == pricing.__name__
    }
    assert {"clean_odds", "inverse_odds", "overround", "floor_bound", "stake_back_share_dividend",
            "detruncate"} <= set(public)  # fmt: skip
    assert not [name for name in public if _METHOD_NAME.search(name)]
    for name, fn in public.items():
        params = set(inspect.signature(fn).parameters)
        assert not params & _METHOD_PARAM, (name, params & _METHOD_PARAM)
