"""Exacta cross-pool consistency (#131): the pure maths only.

The exacta pool's implied share of a combination from its final dividend per 100 yen (D5), with
breakage added back as half a 10-yen step; the Win-derived discounted Harville exacta; the
favourite and the category of a combination; and the per-race ratio of the pool's implied
probability to the ordering model's, by category (favourite first, favourite second, without the
favourite).

API pinned here (the module does not exist yet):
- implied_share(dividend, *, rate, unit=100, step=10, min_dividend=110) -> float
  = rate / ((dividend + step / 2) / unit); NaN at or below min_dividend (the floor binds);
  ValueError for a non-finite or non-positive dividend or rate.
- harville_exacta(p, lambda2=1.0) -> n x n array, [i, j] = P(first = i, second = j).
- favourite(win_odds: Mapping[int, float]) -> int: lowest odds; a tie goes to the lowest number.
- combo_category((first, second), favourite) -> "fav_first" | "fav_second" | "no_fav".
- race_ratios(p, dividends, *, favourite, rate, lambda2=1.0) -> object with
  .categories: {category: sum q_c / sum p_c over the supplied identified combinations, NaN if none}
  .combinations: {combo: q_c / p_c, NaN when the share is not identified}
  p maps horse number -> Win probability (renormalised over the field); a combination or a
  favourite naming a horse not in p raises ValueError.

Expected values are derived in this file (hand-written Harville loops, floored dividends from known
shares). All numbers are synthetic.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math
from itertools import permutations

import numpy as np
import pytest

from horsetrack.countries.japan.models.v0 import ordering
from horsetrack.countries.japan.research.exacta_consistency import (
    combo_category,
    favourite,
    harville_exacta,
    implied_share,
    race_ratios,
)

CATEGORIES = ("fav_first", "fav_second", "no_fav")

# A synthetic 8-runner field with gaps in the horse numbers (4, 7 and 10 are not running).
FIELD = {1: 0.06, 2: 0.14, 3: 0.09, 5: 0.30, 6: 0.04, 8: 0.18, 9: 0.12, 11: 0.07}
FAV = 5  # the largest Win probability


# ---------------------------------------------------------------------------------------------
# Independent references
# ---------------------------------------------------------------------------------------------


def ref_exacta(p: dict[int, float], lambda2: float) -> dict[tuple[int, int], float]:
    """P(first = i, second = j) = p_i p_j^l2 / sum_{l != i} p_l^l2, p renormalised, by loops."""
    total = sum(p.values())
    q = {h: v / total for h, v in p.items()}
    out = {}
    for i, j in permutations(q, 2):
        denom = sum(q[k] ** lambda2 for k in q if k != i)
        out[(i, j)] = q[i] * q[j] ** lambda2 / denom
    return out


def ref_category(combo: tuple[int, int], fav: int) -> str:
    if combo[0] == fav:
        return "fav_first"
    if combo[1] == fav:
        return "fav_second"
    return "no_fav"


def floored_dividend(share: float, rate: float) -> int:
    """The dividend per 100 yen paid on a combination holding `share` of the pool, 10-yen floor."""
    raw = rate / share * 100
    return max(int(math.floor(raw / 10 + 1e-12)) * 10, 110)


def biased_pool(
    p: dict[int, float], fav: int, lambda2: float, m_first: float, m_second: float
) -> tuple[dict[tuple[int, int], float], dict[str, float]]:
    """Harville times category multipliers; the no-favourite combinations scaled to sum to 1.

    Returns the pool's shares by combination and the multiplier of each category."""
    h = ref_exacta(p, lambda2)
    s = {c: sum(v for k, v in h.items() if ref_category(k, fav) == c) for c in CATEGORIES}
    m_rest = (1 - m_first * s["fav_first"] - m_second * s["fav_second"]) / s["no_fav"]
    assert m_rest > 0
    mult = {"fav_first": m_first, "fav_second": m_second, "no_fav": m_rest}
    return {k: v * mult[ref_category(k, fav)] for k, v in h.items()}, mult


def formula_share(dividend: float, rate: float) -> float:
    return rate * 100 / (dividend + 5)


# ---------------------------------------------------------------------------------------------
# implied_share
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [0.75, 0.80, 0.70])
@pytest.mark.parametrize("dividend", [120, 250, 1230, 4560, 18970, 254320])
def test_implied_share_is_rate_over_dividend_with_half_step_added_back(dividend, rate):
    assert implied_share(dividend, rate=rate) == pytest.approx(rate * 100 / (dividend + 5), 1e-12)


def test_implied_share_unit_step_and_minimum_are_arguments():
    assert implied_share(1000, rate=0.75, step=0) == pytest.approx(0.075, 1e-12)
    assert implied_share(1000, rate=0.75, unit=200) == pytest.approx(0.75 * 200 / 1005, 1e-12)
    assert implied_share(1000, rate=0.75, step=100) == pytest.approx(0.75 * 100 / 1050, 1e-12)
    assert math.isnan(implied_share(150, rate=0.75, min_dividend=150))
    assert implied_share(110, rate=0.75, min_dividend=100) == pytest.approx(75 / 115, 1e-12)


def test_rate_comes_from_the_argument_and_scales_the_share():
    """A Premium-like rate of 0.80 scales every share by 0.80 / 0.75; nothing is hard-coded."""
    for d in (130, 870, 5210, 33330):
        base = implied_share(d, rate=0.75)
        assert implied_share(d, rate=0.80) / base == pytest.approx(0.80 / 0.75, 1e-12)
        assert implied_share(d, rate=0.85) / base == pytest.approx(0.85 / 0.75, 1e-12)


@pytest.mark.parametrize("dividend", [110, 100, 90])
def test_dividend_at_or_below_the_minimum_is_not_identified(dividend):
    assert math.isnan(implied_share(dividend, rate=0.75))
    assert math.isnan(implied_share(dividend, rate=0.80))


@pytest.mark.parametrize(
    ("dividend", "rate"),
    [(0, 0.75), (-120, 0.75), (float("nan"), 0.75), (float("inf"), 0.75),
     (1230, 0.0), (1230, -0.25), (1230, float("nan"))],
)  # fmt: skip
def test_invalid_dividend_or_rate_raises(dividend, rate):
    with pytest.raises(ValueError):
        implied_share(dividend, rate=rate)


@pytest.mark.parametrize("rate", [0.75, 0.80])
def test_breakage_add_back_is_within_the_breakage_bound_of_the_true_share(rate):
    """Dividends floored to 10 yen from known shares: the true share's payout X is in [D, D+10),
    so q_hat / q = X / (D + 5) lies in [D / (D + 5), (D + 10) / (D + 5))."""
    rng = np.random.default_rng(131)
    shares = np.concatenate([rng.uniform(0.0005, 0.6, 400), [0.001, 0.01, 0.1, 0.5]])
    worst_no_addback = 0.0
    worst_addback = 0.0
    for q in shares:
        d = floored_dividend(q, rate)
        if d <= 110:
            continue
        q_hat = implied_share(d, rate=rate)
        assert q_hat == pytest.approx(formula_share(d, rate), 1e-12)
        assert d / (d + 5) - 1e-12 <= q_hat / q < (d + 10) / (d + 5) + 1e-12
        worst_addback = max(worst_addback, abs(math.log(q_hat / q)))
        worst_no_addback = max(worst_no_addback, abs(math.log(rate * 100 / d / q)))
    # Adding back half a step halves the worst-case error of the plain rate / dividend share.
    assert worst_addback < worst_no_addback


# ---------------------------------------------------------------------------------------------
# harville_exacta
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lambda2", [1.0, 0.8, 1.3, 0.0])
def test_harville_exacta_matches_hand_computation_and_ordering_module(lambda2):
    p_raw = np.array(list(FIELD.values())) * 3.7  # not normalised: the function renormalises
    got = np.asarray(harville_exacta(p_raw, lambda2=lambda2), dtype=float)
    p = p_raw / p_raw.sum()
    n = len(p)
    assert got.shape == (n, n)
    for i in range(n):
        denom = sum(p[k] ** lambda2 for k in range(n) if k != i)
        for j in range(n):
            want = 0.0 if i == j else p[i] * p[j] ** lambda2 / denom
            assert got[i, j] == pytest.approx(want, rel=1e-10, abs=1e-15)
    np.testing.assert_allclose(got, ordering._order_probs(p, (lambda2, 1.0), 2), rtol=1e-10)
    assert got.sum() == pytest.approx(1.0, 1e-12)
    np.testing.assert_allclose(got.sum(axis=1), p, rtol=1e-10)
    assert np.all(np.diag(got) == 0)


def test_harville_default_is_plain_harville():
    p = np.array([0.5, 0.3, 0.2])
    got = np.asarray(harville_exacta(p), dtype=float)
    assert got[0, 1] == pytest.approx(0.5 * 0.3 / 0.5)
    assert got[2, 0] == pytest.approx(0.2 * 0.5 / 0.8)
    assert got[1, 2] == pytest.approx(0.3 * 0.2 / 0.7)


# ---------------------------------------------------------------------------------------------
# favourite and combo_category
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("win_odds", "want"),
    [
        ({1: 3.2, 2: 1.8, 3: 5.0}, 2),
        ({7: 12.0, 3: 4.4, 9: 2.9, 1: 30.5}, 9),
        ({4: 2.1, 2: 2.1, 7: 3.0}, 2),  # tie: the lowest horse number
        ({11: 1.5, 6: 1.5, 8: 1.5}, 6),
        ({5: 1.1}, 5),
    ],
)
def test_favourite_is_lowest_win_odds_with_lowest_number_on_a_tie(win_odds, want):
    assert favourite(win_odds) == want
    # Insertion order does not change the answer.
    assert favourite(dict(reversed(list(win_odds.items())))) == want


@pytest.mark.parametrize("fav", [1, 5, 11])
def test_combo_category_by_position_of_the_favourite(fav):
    others = [h for h in (1, 2, 5, 8, 11) if h != fav]
    for x in others:
        assert combo_category((fav, x), fav) == "fav_first"
        assert combo_category((x, fav), fav) == "fav_second"
        for y in others:
            if y != x:
                assert combo_category((x, y), fav) == "no_fav"


# ---------------------------------------------------------------------------------------------
# race_ratios: a synthetic race whose answer is known
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [0.75, 0.80])
@pytest.mark.parametrize("lambda2", [1.0, 0.8, 1.25])
@pytest.mark.parametrize(("m_first", "m_second"), [(1.2, 1.1), (1.0, 1.0), (0.9, 1.3), (1.5, 0.8)])
def test_category_ratios_recover_known_multipliers(m_first, m_second, lambda2, rate):
    pool, mult = biased_pool(FIELD, FAV, lambda2, m_first, m_second)
    dividends = {c: floored_dividend(q, rate) for c, q in pool.items()}
    h = ref_exacta(FIELD, lambda2)
    out = race_ratios(FIELD, dividends, favourite=FAV, rate=rate, lambda2=lambda2)

    assert set(out.categories) == set(CATEGORIES)
    assert set(out.combinations) == set(dividends)
    for cat in CATEGORIES:
        members = [c for c in dividends if ref_category(c, FAV) == cat and dividends[c] > 110]
        exact = sum(formula_share(dividends[c], rate) for c in members) / sum(h[c] for c in members)
        assert out.categories[cat] == pytest.approx(exact, rel=1e-9)
        # Breakage bound: each share is off by at most a factor (D + 10)/(D + 5) or D/(D + 5).
        lo = min(dividends[c] / (dividends[c] + 5) for c in members)
        hi = max((dividends[c] + 10) / (dividends[c] + 5) for c in members)
        assert mult[cat] * lo - 1e-9 <= out.categories[cat] <= mult[cat] * hi + 1e-9
        assert out.categories[cat] == pytest.approx(mult[cat], rel=0.02)
    for c, d in dividends.items():
        assert out.combinations[c] == pytest.approx(formula_share(d, rate) / h[c], rel=1e-9)


def test_unbiased_pool_gives_ratios_near_one():
    pool, _ = biased_pool(FIELD, FAV, 1.0, 1.0, 1.0)
    dividends = {c: floored_dividend(q, 0.75) for c, q in pool.items()}
    out = race_ratios(FIELD, dividends, favourite=FAV, rate=0.75)
    for cat in CATEGORIES:
        assert out.categories[cat] == pytest.approx(1.0, abs=0.01)


def test_lambda2_changes_the_reference():
    """A pool built from discounted Harville at 0.6 is unbiased only against that model.

    The favourite-first total is p_fav for any lambda2, so the change shows in fav_second, no_fav
    and the per-combination ratios."""
    pool, _ = biased_pool(FIELD, FAV, 0.6, 1.0, 1.0)
    dividends = {c: floored_dividend(q, 0.75) for c, q in pool.items()}
    right = race_ratios(FIELD, dividends, favourite=FAV, rate=0.75, lambda2=0.6)
    wrong = race_ratios(FIELD, dividends, favourite=FAV, rate=0.75, lambda2=1.0)
    for cat in CATEGORIES:
        assert right.categories[cat] == pytest.approx(1.0, abs=0.01)
    h1 = ref_exacta(FIELD, 1.0)
    members = [c for c in dividends if ref_category(c, FAV) == "fav_second"]
    want = sum(formula_share(dividends[c], 0.75) for c in members) / sum(h1[c] for c in members)
    assert wrong.categories["fav_second"] == pytest.approx(want, rel=1e-9)
    assert abs(wrong.categories["fav_second"] - 1.0) > 0.03
    assert abs(wrong.categories["no_fav"] - 1.0) > 0.01


def test_rate_argument_scales_the_ratios():
    pool, mult = biased_pool(FIELD, FAV, 1.0, 1.2, 1.1)
    dividends = {c: floored_dividend(q, 0.80) for c, q in pool.items()}
    at_80 = race_ratios(FIELD, dividends, favourite=FAV, rate=0.80)
    at_75 = race_ratios(FIELD, dividends, favourite=FAV, rate=0.75)
    for cat in CATEGORIES:
        assert at_80.categories[cat] == pytest.approx(mult[cat], rel=0.02)
        assert at_75.categories[cat] == pytest.approx(at_80.categories[cat] * 0.75 / 0.80, 1e-9)


def test_win_probabilities_are_renormalised():
    pool, _ = biased_pool(FIELD, FAV, 1.0, 1.2, 1.1)
    dividends = {c: floored_dividend(q, 0.75) for c, q in pool.items()}
    a = race_ratios(FIELD, dividends, favourite=FAV, rate=0.75)
    b = race_ratios({k: v * 2.5 for k, v in FIELD.items()}, dividends, favourite=FAV, rate=0.75)
    for cat in CATEGORIES:
        assert b.categories[cat] == pytest.approx(a.categories[cat], rel=1e-12)


# ---------------------------------------------------------------------------------------------
# race_ratios: partial boards, unidentified shares, absent horses
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("combo", "dividend"), [((5, 2), 1230), ((8, 5), 2870), ((9, 1), 15640), ((2, 8), 3310)]
)
def test_single_winning_combination_gives_its_own_ratio_and_nan_elsewhere(combo, dividend):
    """History holds only the winning combination: one category has q/p, the others NaN."""
    h = ref_exacta(FIELD, 1.0)
    out = race_ratios(FIELD, {combo: dividend}, favourite=FAV, rate=0.75)
    want = formula_share(dividend, 0.75) / h[combo]
    cat = ref_category(combo, FAV)
    assert out.categories[cat] == pytest.approx(want, rel=1e-9)
    assert out.combinations == pytest.approx({combo: want}, rel=1e-9)
    for other in CATEGORIES:
        if other != cat:
            assert math.isnan(out.categories[other])


def test_unidentified_dividends_are_left_out_of_the_category_sums():
    h = ref_exacta(FIELD, 1.0)
    dividends = {(5, 8): 110, (5, 2): 990, (2, 5): 100, (8, 2): 4410}
    out = race_ratios(FIELD, dividends, favourite=FAV, rate=0.75)
    assert out.categories["fav_first"] == pytest.approx(
        formula_share(990, 0.75) / h[(5, 2)], rel=1e-9
    )
    assert math.isnan(out.categories["fav_second"])  # its only combination paid the minimum
    assert out.categories["no_fav"] == pytest.approx(
        formula_share(4410, 0.75) / h[(8, 2)], rel=1e-9
    )
    assert math.isnan(out.combinations[(5, 8)])
    assert math.isnan(out.combinations[(2, 5)])


def test_no_identified_combination_gives_nan_not_zero():
    out = race_ratios(FIELD, {(5, 8): 110, (8, 5): 100}, favourite=FAV, rate=0.75)
    for cat in CATEGORIES:
        assert math.isnan(out.categories[cat])
    out = race_ratios(FIELD, {}, favourite=FAV, rate=0.75)
    for cat in CATEGORIES:
        assert math.isnan(out.categories[cat])


@pytest.mark.parametrize("combo", [(4, 5), (5, 7), (10, 2)])
def test_combination_with_a_horse_not_in_the_field_raises(combo):
    """A scratched or absent horse (not in p) is a data error, not a silent exclusion."""
    with pytest.raises(ValueError):
        race_ratios(FIELD, {combo: 1230, (5, 2): 990}, favourite=FAV, rate=0.75)


def test_favourite_not_in_the_field_raises():
    with pytest.raises(ValueError):
        race_ratios(FIELD, {(5, 2): 990}, favourite=4, rate=0.75)
