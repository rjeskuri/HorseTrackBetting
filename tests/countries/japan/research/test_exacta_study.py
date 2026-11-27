"""Exacta favourite-overload study (#131): the scoring maths of the signed spec, sections 4 to 7.

docs/specs/exacta-overload-prereg.md, accepted by the owner on 2026-11-25 as proposed. All data
here is synthetic; nothing reads the store, the registry file or the real holdout definition.

API pinned here (module ``horsetrack.countries.japan.research.exacta_study``, not yet written):

- ``box_returns(p, winner, dividend, *, favourite, lambda2=1.0) -> {"b1"|"b2"|"b3": {category: R}}``
  One race's return per unit staked on the box of each category S (CATEGORIES of
  exacta_consistency). ``p`` maps horse -> Win probability (renormalised), ``winner`` is the paying
  (first, second), ``dividend`` its final dividend per 100 yen, p_c the discounted Harville exacta.
  - b1 (flat box): 1[w in S] * D_w / 100 / |S_r|
  - b2 (Harville-weighted box): 1[w in S] * (p_w / sum_{c in S} p_c) * D_w / 100
  - b3 (flat box at Harville-fair odds): 1[w in S] / (p_w * |S_r|)
  |S_r| is the number of combinations of S in the field: n-1, n-1 and (n-1)(n-2).
  ValueError for a winner naming a horse not in ``p`` and for a non-positive or non-finite dividend.
- ``rho(returns, rates) -> float``: sum(returns) / sum(rates); ValueError when empty.
- ``e1_loss(category, ret, rate) -> float``: rate - ret for fav_second and no_fav, ret - rate for
  fav_first (so lower is better in each member's hypothesised direction).
- ``SEGMENT_CELLS``: the 14 cell names, in this order:
  "all", "fav_odds:le1.9", "fav_odds:2.0-2.9", "fav_odds:3.0-4.9", "fav_odds:ge5.0",
  "field:5-10", "field:11-14", "field:15-18", "class:maiden", "class:win1-3", "class:open",
  "popularity:less", "popularity:rest", "headline".
- ``segment_cells(*, fav_odds, field_size, race_class, race_no) -> frozenset[str]``: the cells a
  race is in. race_class is the store code: DEBUT, MAIDEN -> maiden; C1, C2, C3 -> win1-3; OP, L,
  G1, G2, G3 -> open. Less popular = race_no 1-4 or a DEBUT/MAIDEN race. Headline = fav_odds >= 3.0
  and less popular. ValueError for a field size outside 5-18 or an unknown class.
- ``exclusion_reason(*, n_starters, boost_may_be_withheld, exacta) -> str | None``: None when the
  race is scored, else a reason. ``exacta`` is the race's rows as (combination, dividend,
  is_refund).
  Excluded: boost possibly withheld, fewer than 5 starters, more than one paying row (a dead heat
  for 1st or 2nd), any refunded row, no row at all.
- ``score_races(races, *, holdout_day, lambda2=1.0) -> pandas.DataFrame``. Each race is a mapping
  with race_id, race_date (datetime.date), meeting_id, race_no, race_class, win_odds (the starters'
  final Win odds, horse -> odds), p (horse -> Win probability), rate (the race's exacta payout
  rate), boost_may_be_withheld and exacta (rows as above). A race whose ``holdout_day("japan",
  race_date)`` is True is dropped before its exacta rows are touched (D14); excluded races are
  dropped. Output: three rows per scored race (one per category) with columns race_id,
  meeting_id, category, rate, b1, b2, b3, e1_loss (of b2), and one bool column per SEGMENT_CELLS
  name. The favourite is exacta_consistency.favourite(win_odds), the field size len(win_odds).
- ``rho_interval(returns, rates, meetings, *, alternative, n_boot, seed, level=0.95)`` -> object
  with .rho, .se and .bound: a meeting (cluster) bootstrap of rho, resampling whole meetings with
  replacement. alternative "less" gives the one-sided upper bound (fav_first), "greater" the lower.
- ``eb_shrink(estimates, ses, *, tau2=None) -> numpy array``: normal-normal empirical Bayes.
  With between-variance tau2 and mu = the precision-weighted mean with weights 1 / (se^2 + tau2),
  each estimate becomes (x / se^2 + mu / tau2) / (1 / se^2 + 1 / tau2); tau2 = 0 gives mu for every
  estimate; tau2 None estimates it from the data (truncated at 0).
- ``family_specs() -> dict[family_id, spec]``: the frozen families as plain-JSON dicts, one per
  estimator: "B2" (E1), "B1" (E1'), "B3" (E1''), "C" (log score) and "rule" (E2). Each spec has
  "estimator", "members" (E1-type: {"category", "cell"}; E2: {"category", "cut", "scope"}),
  "lambda2" ("full_development", 1.0 or a list of them), "block_size", "n_boot", "level",
  "cluster" ("meeting"); E1-type specs carry "directions" {category: "less" | "greater"}; E2 carries
  "cuts", "stake_yen" and "min_meetings".
  Owner 2026-11-26: #115's single full-development lambda2 fit (partly in-sample) is labelled
  "full_development", and no spec says "in_fold". The v1 families (tests/fixtures/synthetic/
  exacta_131_v1_family_specs.py, frozen 2026-11-25 with the hashes in V1_HASHES) stay in the
  append-only registry, which refuses another spec under a frozen id (FamilyAlreadyFrozen), so the
  relabelled families are frozen under new ids beside them.
  E2 correction (review of 0c39a8d): the v2 E2 spec (tests/fixtures/synthetic/
  exacta_131_v2_rule_spec.py, V2_RULE_HASH) took its favourite from the common "lowest final Win
  odds". The corrected E2 is frozen under a new id (neither rule-v1 nor rule-v2, ``supersedes``
  rule-v2) and carries "selection_time" "T-2"; "favourite" and "category" strings that name T-2
  and never final odds; and "scopes" {"all", "headline"}, the headline naming the favourite's odds
  at T-2 >= 3.0. It differs from v2 only in E2_SELECTION_KEYS; the four other v2 families keep
  their frozen ids and hashes.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import copy
import hashlib
import importlib.util
import json
import math
from datetime import date
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from horsetrack.core.eval.registry import Registry
from horsetrack.countries.japan.research.exacta_consistency import CATEGORIES
from horsetrack.countries.japan.research.exacta_study import (
    SEGMENT_CELLS,
    box_returns,
    e1_loss,
    eb_shrink,
    exclusion_reason,
    family_specs,
    rho,
    rho_interval,
    score_races,
    segment_cells,
)

CELLS = (
    "all",
    "fav_odds:le1.9", "fav_odds:2.0-2.9", "fav_odds:3.0-4.9", "fav_odds:ge5.0",
    "field:5-10", "field:11-14", "field:15-18",
    "class:maiden", "class:win1-3", "class:open",
    "popularity:less", "popularity:rest",
    "headline",
)  # fmt: skip
# The v1 families frozen in the Japan registry on 2026-11-25, with the hashes the spec documents
# (docs/specs/exacta-overload-prereg.md); their specs label #115's fit "in_fold".
V1_HASHES = {
    "japan/131-exacta-category-v1":
        "9ac3b740d1bbdde9645091d3fd14247442b1a210b2d109c5e58ade0983769a8f",
    "japan/131-exacta-category-b1-v1":
        "f0e915f51cc13c7b40ef478076f6c44b1b5d22b52722b321c46359e5c2bf5e8f",
    "japan/131-exacta-category-b3-v1":
        "07721b74cc11fd7181b812c42450ac523371e07435047c623cd9d159d0d3fbcb",
    "japan/131-exacta-logscore-v1":
        "0af318b97297b3085d82434921cf4c2f651864321a3e63ae03560b3e2b81c715",
    "japan/131-exacta-rule-v1":
        "b1577fd3765d5f72b0c0643235aca7cd69317d467470a0a31949da50c65a019f",
}  # fmt: skip
FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "synthetic"
V1_SPECS_FIXTURE = FIXTURES / "exacta_131_v1_family_specs.py"
ESTIMATORS = ("B2", "B1", "B3", "C", "rule")


def _family_id(specs, estimator) -> str:
    found = [fid for fid, spec in specs.items() if spec["estimator"] == estimator]
    assert len(found) == 1, (estimator, found)
    return found[0]


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


def ref_box(p, winner, dividend, fav, lambda2=1.0):
    h = ref_exacta(p, lambda2)
    out = {e: dict.fromkeys(CATEGORIES, 0.0) for e in ("b1", "b2", "b3")}
    s = ref_category(winner, fav)
    members = [c for c in h if ref_category(c, fav) == s]
    out["b1"][s] = dividend / 100 / len(members)
    out["b2"][s] = h[winner] / sum(h[c] for c in members) * dividend / 100
    out["b3"][s] = 1 / (h[winner] * len(members))
    return out


def ref_cells(fav_odds, field_size, race_class, race_no) -> set[str]:
    cells = {"all"}
    if fav_odds <= 1.9:
        cells.add("fav_odds:le1.9")
    elif fav_odds <= 2.9:
        cells.add("fav_odds:2.0-2.9")
    elif fav_odds <= 4.9:
        cells.add("fav_odds:3.0-4.9")
    else:
        cells.add("fav_odds:ge5.0")
    cells.add(
        "field:5-10" if field_size <= 10 else "field:11-14" if field_size <= 14 else "field:15-18"
    )
    group = {"DEBUT": "maiden", "MAIDEN": "maiden", "C1": "win1-3", "C2": "win1-3", "C3": "win1-3"}
    cells.add(f"class:{group.get(race_class, 'open')}")
    less = race_no <= 4 or race_class in ("DEBUT", "MAIDEN")
    cells.add("popularity:less" if less else "popularity:rest")
    if fav_odds >= 3.0 and less:
        cells.add("headline")
    return cells


def harville_np(p: np.ndarray, lambda2: float) -> np.ndarray:
    p = p / p.sum()
    pl = p**lambda2
    out = p[:, None] * pl[None, :] / (pl.sum() - pl)[:, None]
    np.fill_diagonal(out, 0.0)
    return out


def category_grid(n: int, fav: int) -> np.ndarray:
    """0 fav_first, 1 fav_second, 2 no_fav, -1 the diagonal."""
    grid = np.full((n, n), 2)
    grid[fav, :] = 0
    grid[:, fav] = 1
    np.fill_diagonal(grid, -1)
    return grid


# A synthetic 6-runner field with a gap in the numbers (4 is not running).
FIELD = {1: 0.08, 2: 0.31, 3: 0.17, 5: 0.12, 6: 0.22, 7: 0.10}
FAV = 2


# ---------------------------------------------------------------------------------------------
# B1, B2, B3 per race
# ---------------------------------------------------------------------------------------------


def test_box_returns_on_a_hand_computed_race():
    """p = (.4, .2, .2, .1, .1), favourite 1, Harville; the exacta 2-1 pays 1230 yen.

    P(2, 1) = .2 * .4 / .8 = .1. fav_second's Harville mass = .4 * (.2/.8 + .2/.8 + .1/.9 + .1/.9).
    |fav_second| = 4."""
    p = {1: 0.4, 2: 0.2, 3: 0.2, 4: 0.1, 5: 0.1}
    out = box_returns(p, (2, 1), 1230, favourite=1)
    mass = 0.4 * (0.25 + 0.25 + 1 / 9 + 1 / 9)
    assert out["b1"]["fav_second"] == pytest.approx(12.30 / 4, rel=1e-12)
    assert out["b2"]["fav_second"] == pytest.approx(0.1 / mass * 12.30, rel=1e-12)
    assert out["b3"]["fav_second"] == pytest.approx(1 / (0.1 * 4), rel=1e-12)
    for est in ("b1", "b2", "b3"):
        assert out[est]["fav_first"] == 0.0
        assert out[est]["no_fav"] == 0.0


def test_box_returns_no_fav_uses_the_n_minus_1_times_n_minus_2_box():
    """Same field, exacta 3-4 pays 5670: |no_fav| = 4 * 3 = 12, P(3, 4) = .2 * .1 / .8 = .025."""
    p = {1: 0.4, 2: 0.2, 3: 0.2, 4: 0.1, 5: 0.1}
    out = box_returns(p, (3, 4), 5670, favourite=1)
    h = ref_exacta(p, 1.0)
    mass = sum(v for c, v in h.items() if 1 not in c)
    assert out["b1"]["no_fav"] == pytest.approx(56.70 / 12, rel=1e-12)
    assert out["b2"]["no_fav"] == pytest.approx(0.025 / mass * 56.70, rel=1e-12)
    assert out["b3"]["no_fav"] == pytest.approx(1 / (0.025 * 12), rel=1e-12)
    assert out["b2"]["fav_first"] == out["b2"]["fav_second"] == 0.0


@pytest.mark.parametrize("lambda2", [1.0, 0.8, 0.65])
@pytest.mark.parametrize("winner", [(2, 6), (6, 2), (3, 5), (7, 1), (2, 1), (5, 2)])
@pytest.mark.parametrize("dividend", [140, 1230, 25670])
def test_box_returns_match_the_reference_over_winners_dividends_and_lambda2(
    winner, dividend, lambda2
):
    out = box_returns(FIELD, winner, dividend, favourite=FAV, lambda2=lambda2)
    ref = ref_box(FIELD, winner, dividend, FAV, lambda2)
    for est in ("b1", "b2", "b3"):
        for cat in CATEGORIES:
            assert out[est][cat] == pytest.approx(ref[est][cat], rel=1e-10, abs=1e-15), (est, cat)


def test_box_returns_with_a_different_favourite_moves_the_winner_between_categories():
    """The exacta 6-2 is fav_second when 2 is the favourite and fav_first when 6 is."""
    as_two = box_returns(FIELD, (6, 2), 870, favourite=2)
    as_six = box_returns(FIELD, (6, 2), 870, favourite=6)
    assert as_two["b2"]["fav_second"] > 0 and as_two["b2"]["fav_first"] == 0
    assert as_six["b2"]["fav_first"] > 0 and as_six["b2"]["fav_second"] == 0
    assert as_six["b2"]["fav_first"] == pytest.approx(
        ref_box(FIELD, (6, 2), 870, 6)["b2"]["fav_first"]
    )


@pytest.mark.parametrize(
    ("winner", "dividend"),
    [((2, 4), 1230), ((9, 2), 1230), ((2, 6), 0), ((2, 6), -50), ((2, 6), float("nan"))],
)
def test_box_returns_rejects_absent_horses_and_bad_dividends(winner, dividend):
    with pytest.raises(ValueError):
        box_returns(FIELD, winner, dividend, favourite=FAV)


def _random_fields(seed: int, n_races: int):
    rng = np.random.default_rng(seed)
    for _ in range(n_races):
        n = int(rng.integers(5, 15))
        horses = sorted(rng.choice(np.arange(1, 19), size=n, replace=False).tolist())
        w = rng.gamma(1.0, size=n)
        yield (
            {h: float(v) for h, v in zip(horses, w / w.sum(), strict=True)},
            float(rng.choice([0.75, 0.80, 0.70])),
        )


def test_b2_and_b1_expected_return_is_the_payout_rate_when_the_pool_is_right():
    """Exact expectation, no sampling: if outcomes follow the pool's own shares q (pi = q) and each
    combination pays rate / q, every category's B2 and B1 box returns exactly `rate` per unit,
    whatever Harville's error (here the pool is Harville with lambda2 0.7, the model lambda2 1)."""
    for p, rate in _random_fields(7, 25):
        fav = max(p, key=p.get)
        q = ref_exacta(p, 0.7)
        expected = {e: dict.fromkeys(CATEGORIES, 0.0) for e in ("b1", "b2")}
        for combo, share in q.items():
            out = box_returns(p, combo, rate / share * 100, favourite=fav, lambda2=1.0)
            for e in expected:
                for cat in CATEGORIES:
                    expected[e][cat] += share * out[e][cat]
        for e in expected:
            for cat in CATEGORIES:
                assert expected[e][cat] == pytest.approx(rate, rel=1e-9), (e, cat)


def _simulate(seed: int, n_races: int, overload: float | None):
    """Draw races and return per-category B2 returns, the rates, and the expected rho by category.

    overload None: the pool is right (q = pi = Harville lambda2 0.8), the model is Harville 1.
    overload m: the truth is the model (Harville 1), the pool multiplies fav_first shares by m and
    renormalises, so fav_first returns rate * (pi_S / q_S) and the others more than rate."""
    rng = np.random.default_rng(seed)
    returns = {c: [] for c in CATEGORIES}
    rates, expected_num = [], dict.fromkeys(CATEGORIES, 0.0)
    for _ in range(n_races):
        n = int(rng.integers(8, 15))
        w = rng.gamma(1.0, size=n)
        p = w / w.sum()
        fav = int(np.argmax(p))
        rate = float(rng.choice([0.75, 0.80]))
        grid = category_grid(n, fav)
        if overload is None:
            pi = harville_np(p, 0.8)
            q = pi
        else:
            pi = harville_np(p, 1.0)
            q = pi.copy()
            q[grid == 0] *= overload
            q /= q.sum()
        k = int(rng.choice(n * n, p=pi.ravel() / pi.sum()))
        i, j = divmod(k, n)
        horses = {h + 1: float(p[h]) for h in range(n)}
        out = box_returns(horses, (i + 1, j + 1), rate / q[i, j] * 100, favourite=fav + 1)
        for c in CATEGORIES:
            returns[c].append(out["b2"][c])
        rates.append(rate)
        for code, c in enumerate(CATEGORIES):
            expected_num[c] += rate * pi[grid == code].sum() / q[grid == code].sum()
    expected = {c: expected_num[c] / sum(rates) for c in CATEGORIES}
    return returns, rates, expected


def test_b2_rho_is_one_for_every_category_when_outcomes_follow_the_pool():
    returns, rates, _ = _simulate(131, 16000, overload=None)
    for c in CATEGORIES:
        assert rho(returns[c], rates) == pytest.approx(1.0, abs=0.05), c


def test_b2_rho_detects_a_known_fav_first_overload():
    """Pool overloads fav_first by 1.3: rho_fav_first about 0.83 and the others about 1.08."""
    returns, rates, expected = _simulate(2026, 16000, overload=1.3)
    assert expected["fav_first"] < 0.88 and expected["fav_second"] > 1.04  # the design's own check
    for c in CATEGORIES:
        assert rho(returns[c], rates) == pytest.approx(expected[c], abs=0.05), c
    assert rho(returns["fav_first"], rates) < 0.9
    assert rho(returns["fav_second"], rates) > 1.0


# ---------------------------------------------------------------------------------------------
# rho and the E1 loss
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("returns", "rates", "expected"),
    [
        ([0.0, 3.1, 0.0, 0.5], [0.75, 0.75, 0.80, 0.75], 3.6 / 3.05),
        ([0.0, 0.0], [0.75, 0.80], 0.0),
        ([1.2], [0.8], 1.5),
        ([0.6, 0.9, 0.0], [0.75, 0.75, 0.75], 1.5 / 2.25),
    ],
)
def test_rho_is_total_return_over_total_rate(returns, rates, expected):
    assert rho(returns, rates) == pytest.approx(expected, rel=1e-12)


def test_rho_of_no_races_raises():
    with pytest.raises(ValueError):
        rho([], [])


@pytest.mark.parametrize("ret", [0.0, 0.4, 0.75, 1.9, 12.5])
@pytest.mark.parametrize("rate", [0.75, 0.80])
def test_e1_loss_sign_follows_each_category_s_direction(ret, rate):
    assert e1_loss("fav_first", ret, rate) == pytest.approx(ret - rate, abs=1e-12)
    assert e1_loss("fav_second", ret, rate) == pytest.approx(rate - ret, abs=1e-12)
    assert e1_loss("no_fav", ret, rate) == pytest.approx(rate - ret, abs=1e-12)


def test_e1_loss_rejects_an_unknown_category():
    with pytest.raises(ValueError):
        e1_loss("fav_third", 0.5, 0.75)


# ---------------------------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------------------------


def test_segment_cells_are_the_fourteen_signed_cells():
    assert tuple(SEGMENT_CELLS) == CELLS
    assert len(set(SEGMENT_CELLS)) == 14


@pytest.mark.parametrize(
    ("fav_odds", "band"),
    [(1.0, "fav_odds:le1.9"), (1.5, "fav_odds:le1.9"), (1.9, "fav_odds:le1.9"),
     (2.0, "fav_odds:2.0-2.9"), (2.4, "fav_odds:2.0-2.9"), (2.9, "fav_odds:2.0-2.9"),
     (3.0, "fav_odds:3.0-4.9"), (4.1, "fav_odds:3.0-4.9"), (4.9, "fav_odds:3.0-4.9"),
     (5.0, "fav_odds:ge5.0"), (7.3, "fav_odds:ge5.0"), (99.9, "fav_odds:ge5.0")],
)  # fmt: skip
def test_favourite_odds_bands_at_their_boundaries(fav_odds, band):
    cells = segment_cells(fav_odds=fav_odds, field_size=12, race_class="C2", race_no=9)
    assert {c for c in cells if c.startswith("fav_odds:")} == {band}


@pytest.mark.parametrize(
    ("field_size", "band"),
    [(5, "field:5-10"), (8, "field:5-10"), (10, "field:5-10"), (11, "field:11-14"),
     (14, "field:11-14"), (15, "field:15-18"), (16, "field:15-18"), (18, "field:15-18")],
)  # fmt: skip
def test_field_size_bands_at_their_boundaries(field_size, band):
    cells = segment_cells(fav_odds=2.5, field_size=field_size, race_class="C1", race_no=7)
    assert {c for c in cells if c.startswith("field:")} == {band}


@pytest.mark.parametrize(
    ("race_class", "group"),
    [("DEBUT", "class:maiden"), ("MAIDEN", "class:maiden"), ("C1", "class:win1-3"),
     ("C2", "class:win1-3"), ("C3", "class:win1-3"), ("OP", "class:open"), ("L", "class:open"),
     ("G1", "class:open"), ("G2", "class:open"), ("G3", "class:open")],
)  # fmt: skip
def test_class_groups(race_class, group):
    cells = segment_cells(fav_odds=2.5, field_size=12, race_class=race_class, race_no=10)
    assert {c for c in cells if c.startswith("class:")} == {group}


@pytest.mark.parametrize(
    ("fav_odds", "race_class", "race_no", "less", "headline"),
    [
        (3.0, "C1", 4, True, True),  # race 4 is less popular; 3.0 is in the headline
        (3.0, "C1", 5, False, False),  # race 5 of a 1-win race is not
        (2.9, "C1", 1, True, False),  # less popular but the favourite is under 3.0
        (5.0, "DEBUT", 9, True, True),  # a newcomer race late in the day is less popular
        (4.2, "MAIDEN", 12, True, True),
        (12.0, "G1", 11, False, False),
        (1.4, "MAIDEN", 2, True, False),
        (6.5, "C3", 3, True, True),
        (6.5, "OP", 8, False, False),
    ],
)
def test_less_popular_proxy_and_headline_cross(fav_odds, race_class, race_no, less, headline):
    cells = segment_cells(fav_odds=fav_odds, field_size=14, race_class=race_class, race_no=race_no)
    assert ("popularity:less" in cells) is less
    assert ("popularity:rest" in cells) is (not less)
    assert ("headline" in cells) is headline


def test_every_race_is_in_all_and_exactly_one_level_of_each_marginal():
    rng = np.random.default_rng(5)
    classes = ["DEBUT", "MAIDEN", "C1", "C2", "C3", "OP", "L", "G1", "G2", "G3"]
    odds = [round(x, 1) for x in np.arange(1.0, 15.0, 0.1)]
    for _ in range(400):
        args = {
            "fav_odds": float(rng.choice(odds)),
            "field_size": int(rng.integers(5, 19)),
            "race_class": str(rng.choice(classes)),
            "race_no": int(rng.integers(1, 13)),
        }
        cells = segment_cells(**args)
        assert set(cells) == ref_cells(**args), args
        assert set(cells) <= set(CELLS)
        for prefix in ("fav_odds:", "field:", "class:", "popularity:"):
            assert sum(c.startswith(prefix) for c in cells) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fav_odds": 2.5, "field_size": 4, "race_class": "C1", "race_no": 5},
        {"fav_odds": 2.5, "field_size": 19, "race_class": "C1", "race_no": 5},
        {"fav_odds": 2.5, "field_size": 12, "race_class": "WEIRD", "race_no": 5},
    ],
)
def test_segment_cells_reject_out_of_scope_races(kwargs):
    with pytest.raises(ValueError):
        segment_cells(**kwargs)


# ---------------------------------------------------------------------------------------------
# Exclusions
# ---------------------------------------------------------------------------------------------

ONE_ROW = [((3, 7), 2340, False)]


@pytest.mark.parametrize("n_starters", [5, 6, 11, 18])
def test_a_clean_race_is_scored(n_starters):
    assert (
        exclusion_reason(n_starters=n_starters, boost_may_be_withheld=False, exacta=ONE_ROW) is None
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_starters": 12, "boost_may_be_withheld": True, "exacta": ONE_ROW},
        {"n_starters": 4, "boost_may_be_withheld": False, "exacta": ONE_ROW},
        {"n_starters": 3, "boost_may_be_withheld": False, "exacta": ONE_ROW},
        # dead heat for 1st: 3 and 7 tie, both orders pay
        {"n_starters": 12, "boost_may_be_withheld": False,
         "exacta": [((3, 7), 1150, False), ((7, 3), 1620, False)]},
        # dead heat for 2nd: 3 wins, 7 and 9 tie for second
        {"n_starters": 12, "boost_may_be_withheld": False,
         "exacta": [((3, 7), 1150, False), ((3, 9), 2620, False)]},
        {"n_starters": 12, "boost_may_be_withheld": False, "exacta": [((3, 7), 100, True)]},
        {"n_starters": 12, "boost_may_be_withheld": False, "exacta": []},
    ],
)  # fmt: skip
def test_excluded_races(kwargs):
    assert exclusion_reason(**kwargs) is not None


# ---------------------------------------------------------------------------------------------
# score_races: development days only, exclusions, the per-race table
# ---------------------------------------------------------------------------------------------


class Poison:
    """Exacta rows that fail the test the moment anything reads them."""

    def _boom(self, *args, **kwargs):
        raise AssertionError("a holdout race's exacta rows were read")

    __iter__ = __len__ = __getitem__ = __bool__ = __contains__ = _boom


def _race(rid, day, meeting, race_no, race_class, odds, winner, dividend, **over):
    total = sum(1 / o for o in odds.values())
    race = {
        "race_id": rid,
        "race_date": day,
        "meeting_id": meeting,
        "race_no": race_no,
        "race_class": race_class,
        "win_odds": odds,
        "p": {h: (1 / o) / total for h, o in odds.items()},
        "rate": 0.75,
        "boost_may_be_withheld": False,
        "exacta": [(winner, dividend, False)],
    }
    race.update(over)
    return race


D1, D2, D3 = date(2025, 5, 3), date(2025, 5, 4), date(2025, 5, 10)
ODDS_A = {1: 3.4, 2: 5.1, 3: 2.2, 4: 9.8, 5: 14.0, 6: 30.5, 7: 6.6}  # favourite 3 at 2.2
ODDS_B = {h: o for h, o in zip(range(1, 13), [4.1, 3.2, 8.0, 12.5, 6.1, 25.0, 40.0, 9.9, 18.0,
                                               55.0, 7.4, 66.0], strict=True)}  # fmt: skip
ODDS_C = {1: 1.6, 2: 4.4, 3: 7.9, 4: 12.0, 5: 19.5}  # favourite 1 at 1.6, 5 starters


def _sample_races():
    return [
        _race("r1", D1, "m1", 2, "MAIDEN", ODDS_A, (3, 1), 870),  # fav_first
        _race("r2", D1, "m1", 9, "C2", ODDS_B, (1, 2), 1460),  # fav (2) second
        _race("r3", D2, "m2", 11, "G3", ODDS_C, (2, 4), 5120, rate=0.80),  # no_fav
        _race("r4", D3, "m3", 4, "C1", ODDS_B, (5, 11), 9930),  # no_fav, headline (3.2, race 4)
    ]


def test_score_races_drops_holdout_days_before_reading_their_dividends():
    calls = []

    def holdout_day(country, day):
        assert country == "japan"
        assert isinstance(day, date)
        calls.append(day)
        return day == D2

    races = _sample_races()
    held = _race("h1", D2, "m2", 3, "C1", ODDS_A, (3, 1), 870)
    held["exacta"] = Poison()
    rows = score_races([*races, held], holdout_day=holdout_day)
    assert set(rows["race_id"]) == {"r1", "r2", "r4"}  # r3 and h1 are on the holdout day D2
    assert set(calls) == {D1, D2, D3}


def test_score_races_drops_excluded_races_and_keeps_three_rows_per_scored_race():
    races = _sample_races()
    races += [
        _race("x_boost", D1, "m1", 5, "C1", ODDS_A, (3, 1), 870, boost_may_be_withheld=True),
        _race("x_small", D1, "m1", 6, "C1", {1: 2.0, 2: 3.0, 3: 5.0, 4: 9.0}, (1, 2), 560),
        _race("x_heat", D1, "m1", 7, "C1", ODDS_A,
              (3, 1), 870, exacta=[((3, 1), 870, False), ((1, 3), 1240, False)]),
        _race("x_refund", D1, "m1", 8, "C1", ODDS_A, (3, 1), 100, exacta=[((3, 1), 100, True)]),
    ]  # fmt: skip
    rows = score_races(races, holdout_day=lambda country, day: False)
    assert set(rows["race_id"]) == {"r1", "r2", "r3", "r4"}
    assert len(rows) == 12
    for rid in ("r1", "r2", "r3", "r4"):
        assert sorted(rows.loc[rows["race_id"] == rid, "category"]) == sorted(CATEGORIES)


@pytest.mark.parametrize("lambda2", [1.0, 0.75])
def test_score_races_values_match_the_per_race_maths(lambda2):
    races = _sample_races()
    rows = score_races(races, holdout_day=lambda country, day: False, lambda2=lambda2)
    for race in races:
        odds = race["win_odds"]
        fav = min(odds, key=lambda h: (odds[h], h))
        winner, dividend, _ = race["exacta"][0]
        ref = ref_box(race["p"], winner, dividend, fav, lambda2)
        cells = ref_cells(odds[fav], len(odds), race["race_class"], race["race_no"])
        for cat in CATEGORIES:
            row = rows[(rows["race_id"] == race["race_id"]) & (rows["category"] == cat)]
            assert len(row) == 1
            row = row.iloc[0]
            assert row["meeting_id"] == race["meeting_id"]
            assert row["rate"] == pytest.approx(race["rate"])
            for est in ("b1", "b2", "b3"):
                assert row[est] == pytest.approx(ref[est][cat], rel=1e-10, abs=1e-15)
            sign = 1 if cat == "fav_first" else -1
            assert row["e1_loss"] == pytest.approx(
                sign * (ref["b2"][cat] - race["rate"]), abs=1e-12
            )
            for cell in CELLS:
                assert bool(row[cell]) is (cell in cells), (race["race_id"], cell)


# ---------------------------------------------------------------------------------------------
# Meeting bootstrap of rho
# ---------------------------------------------------------------------------------------------


def _clustered(seed: int, n_meetings: int):
    """Races with a strong meeting effect, so meeting and race resampling give different SEs."""
    rng = np.random.default_rng(seed)
    returns, rates, meetings = [], [], []
    for m in range(n_meetings):
        effect = rng.normal(0.0, 0.6)
        for _ in range(int(rng.integers(6, 13))):
            rate = float(rng.choice([0.75, 0.80]))
            returns.append(max(0.0, rate * (1 + effect + rng.normal(0.0, 0.3))))
            rates.append(rate)
            meetings.append(f"m{m:03d}")
    return np.array(returns), np.array(rates), np.array(meetings)


def _linear_se(returns, rates, labels) -> float:
    """The linearised SE of sum(R) / sum(rate) when whole `labels` units are resampled."""
    r_hat = returns.sum() / rates.sum()
    d = {}
    for y, x, g in zip(returns, rates, labels, strict=True):
        d[g] = d.get(g, 0.0) + y - r_hat * x
    return math.sqrt(sum(v * v for v in d.values())) / rates.sum()


def test_rho_interval_resamples_whole_meetings():
    returns, rates, meetings = _clustered(11, 250)
    by_meeting = _linear_se(returns, rates, meetings)
    by_race = _linear_se(returns, rates, np.arange(len(returns)))
    assert by_meeting > 2 * by_race  # the fixture separates the two
    out = rho_interval(returns, rates, meetings, alternative="less", n_boot=4000, seed=3)
    assert out.rho == pytest.approx(returns.sum() / rates.sum(), rel=1e-12)
    assert out.se == pytest.approx(by_meeting, rel=0.12)
    # every race its own meeting: the SE falls to the race-level one
    alone = rho_interval(
        returns, rates, np.arange(len(returns)), alternative="less", n_boot=4000, seed=3
    )
    assert alone.se == pytest.approx(by_race, rel=0.12)


def test_rho_interval_one_sided_bounds_and_seed():
    returns, rates, meetings = _clustered(12, 200)
    upper = rho_interval(returns, rates, meetings, alternative="less", n_boot=4000, seed=9)
    lower = rho_interval(returns, rates, meetings, alternative="greater", n_boot=4000, seed=9)
    z = 1.6448536269514722
    assert upper.bound > upper.rho
    assert lower.bound < lower.rho
    assert upper.bound == pytest.approx(upper.rho + z * upper.se, abs=0.3 * upper.se)
    assert lower.bound == pytest.approx(lower.rho - z * lower.se, abs=0.3 * lower.se)
    again = rho_interval(returns, rates, meetings, alternative="less", n_boot=4000, seed=9)
    assert (again.rho, again.se, again.bound) == (upper.rho, upper.se, upper.bound)


# ---------------------------------------------------------------------------------------------
# Empirical-Bayes shrinkage of segment estimates
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.72, 1.0, 1.31])
def test_equal_estimates_stay_put(value):
    ses = np.array([0.05, 0.2, 0.1, 0.4, 0.08])
    out = eb_shrink(np.full(5, value), ses)
    assert np.allclose(out, value, atol=1e-12)


def test_a_noisy_outlier_shrinks_toward_the_precise_cells():
    x = np.array([0.9, 1.0, 1.1, 1.0, 3.0])
    ses = np.array([0.1, 0.1, 0.1, 0.1, 0.5])
    out = np.asarray(eb_shrink(x, ses), dtype=float)
    assert 1.0 < out[4] < 2.5
    moves = np.abs(out - x)
    assert all(moves[4] > moves[k] for k in range(4))
    assert np.all(np.abs(out[:4] - 1.0) <= np.abs(x[:4] - 1.0) + 0.1)


def test_zero_between_variance_gives_the_precision_weighted_mean():
    """Spread well inside the SEs: tau2 estimates to 0 and every cell takes the pooled mean."""
    x = np.array([1.00, 1.04, 0.97, 1.02, 0.99])
    ses = np.array([0.5, 0.3, 0.6, 0.4, 0.8])
    w = 1 / ses**2
    pooled = float((w * x).sum() / w.sum())
    assert np.allclose(eb_shrink(x, ses), pooled, atol=1e-9)
    assert np.allclose(eb_shrink(x, ses, tau2=0.0), pooled, atol=1e-9)


@pytest.mark.parametrize("tau2", [0.01, 0.05, 0.4])
def test_shrinkage_with_a_given_tau2_is_the_normal_normal_posterior_mean(tau2):
    x = np.array([0.82, 1.15, 0.96, 1.40, 0.70, 1.05])
    ses = np.array([0.10, 0.25, 0.05, 0.30, 0.15, 0.20])
    w = 1 / (ses**2 + tau2)
    mu = (w * x).sum() / w.sum()
    expected = (x / ses**2 + mu / tau2) / (1 / ses**2 + 1 / tau2)
    assert np.allclose(eb_shrink(x, ses, tau2=tau2), expected, rtol=1e-10)


# ---------------------------------------------------------------------------------------------
# The frozen family specs
# ---------------------------------------------------------------------------------------------


def _lambda2s(spec) -> list:
    value = spec["lambda2"]
    return list(value) if isinstance(value, list) else [value]


def _e1_like(specs, estimator):
    return [s for s in specs.values() if s["estimator"] == estimator]


def test_e1_spec_has_42_category_by_cell_members_and_the_signed_settings():
    specs = family_specs()
    e1 = specs[_family_id(specs, "B2")]
    assert e1["estimator"] == "B2"
    members = {(m["category"], m["cell"]) for m in e1["members"]}
    assert len(e1["members"]) == 42
    assert members == {(c, cell) for c in CATEGORIES for cell in CELLS}
    assert e1["directions"] == {"fav_first": "less", "fav_second": "greater", "no_fav": "greater"}
    assert e1["block_size"] == 5
    assert e1["n_boot"] == 10000
    assert e1["level"] == 0.95
    assert e1["cluster"] == "meeting"
    assert "full_development" in _lambda2s(e1)


def test_e1_variants_b1_b3_and_plain_harville_are_registered():
    specs = family_specs()
    e1 = specs[_family_id(specs, "B2")]
    for estimator in ("B1", "B3"):
        found = _e1_like(specs, estimator)
        assert found, estimator
        for spec in found:
            members = {(m["category"], m["cell"]) for m in spec["members"]}
            assert members == {(c, cell) for c in CATEGORIES for cell in CELLS}
            assert spec["directions"] == e1["directions"]
    assert any(1.0 in _lambda2s(s) for s in _e1_like(specs, "B2"))


@pytest.mark.parametrize(
    "estimator, expected",
    [
        ("B2", {"full_development", 1.0}),
        ("B3", {"full_development", 1.0}),
        ("C", {"full_development", 1.0}),
        ("rule", {"full_development"}),
    ],
)
def test_the_fitted_lambda2_is_labelled_full_development(estimator, expected):
    specs = family_specs()
    assert set(_lambda2s(specs[_family_id(specs, estimator)])) == expected


def _strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_no_spec_says_in_fold_and_every_lambda2_label_is_full_development(estimator):
    specs = family_specs()
    spec = specs[_family_id(specs, estimator)]
    assert [s for s in _strings(spec) if "in_fold" in s] == []
    assert {v for v in _lambda2s(spec) if isinstance(v, str)} <= {"full_development"}
    assert all(v == 1.0 for v in _lambda2s(spec) if not isinstance(v, str))


def test_e2_spec_has_16_rule_members_and_the_signed_settings():
    specs = family_specs()
    e2 = specs[_family_id(specs, "rule")]
    assert e2["estimator"] == "rule"
    members = {(m["category"], m["cut"], m["scope"]) for m in e2["members"]}
    assert len(e2["members"]) == 16
    assert members == {
        (c, k, s)
        for c in ("fav_second", "no_fav")
        for k in (0.6, 0.7, 0.8, 0.9)
        for s in ("all", "headline")
    }
    assert sorted(e2["cuts"]) == [0.6, 0.7, 0.8, 0.9]
    assert e2["stake_yen"] == 100
    assert e2["min_meetings"] == 100
    assert e2["block_size"] == 5
    assert e2["cluster"] == "meeting"


def test_family_specs_are_plain_json_deterministic_and_freeze(tmp_path):
    first, second = family_specs(), family_specs()
    assert first == second
    assert sorted(s["estimator"] for s in first.values()) == sorted(ESTIMATORS)
    for family_id, spec in first.items():
        assert json.loads(json.dumps(spec, allow_nan=False)) == spec  # lists, not tuples; no NaN
        assert spec == second[family_id]
    e1_id = _family_id(first, "B2")
    first[e1_id]["members"].clear()  # a caller's change never reaches the next call
    assert len(family_specs()[e1_id]["members"]) == 42

    reg = Registry(tmp_path / "japan.jsonl", holdout_log=tmp_path / "holdout_log.jsonl")
    for family_id, spec in second.items():
        expected = _hash(spec)
        assert reg.freeze_family(family_id, copy.deepcopy(spec)) == expected
        assert reg.freeze_family(family_id, family_specs()[family_id]) == expected  # no-op
    for family_id, spec in second.items():
        assert reg.frozen_spec_hash(family_id) == _hash(spec)


def _hash(spec) -> str:
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _v1_specs() -> dict:
    # Loaded by path (tests/ is not a package), executed afresh on each call.
    loader = importlib.util.spec_from_file_location("exacta_131_v1_family_specs", V1_SPECS_FIXTURE)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    v1 = module.V1_FAMILY_SPECS
    assert {fid: _hash(spec) for fid, spec in v1.items()} == V1_HASHES  # the frozen v1 record
    return v1


def _v1_id(v1, estimator) -> str:
    (found,) = [fid for fid, spec in v1.items() if spec["estimator"] == estimator]
    return found


def test_relabelled_families_freeze_under_new_ids_beside_the_frozen_v1_families(tmp_path):
    v1 = _v1_specs()
    reg = Registry(tmp_path / "japan.jsonl", holdout_log=tmp_path / "holdout_log.jsonl")
    for family_id, spec in v1.items():
        assert reg.freeze_family(family_id, spec) == V1_HASHES[family_id]

    specs = family_specs()
    assert set(specs).isdisjoint(V1_HASHES)  # every v1 spec changes (its λ2 label): new ids
    for family_id, spec in specs.items():
        assert _hash(spec) not in V1_HASHES.values()
        assert reg.freeze_family(family_id, spec) == _hash(spec)  # no FamilyAlreadyFrozen

    for family_id, frozen in V1_HASHES.items():  # the v1 freezes stand, unchanged
        assert reg.frozen_spec_hash(family_id) == frozen
    for family_id, spec in specs.items():
        assert reg.frozen_spec_hash(family_id) == _hash(spec)
    reg.verify()


_UNCHANGED_KEYS = (
    "estimator", "members", "family_size", "directions", "segments", "block_size", "n_boot",
    "level", "seed", "cluster", "exclusions", "cuts", "stake_yen", "pool", "sources",
    "hypothesis_id", "ticket",
)  # fmt: skip


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_the_relabel_changes_the_lambda2_label_and_not_the_design(estimator):
    v1 = _v1_specs()
    old = v1[_v1_id(v1, estimator)]
    specs = family_specs()
    new = specs[_family_id(specs, estimator)]
    for key in _UNCHANGED_KEYS:
        assert new.get(key) == old.get(key), key
    relabelled = json.loads(json.dumps(old["lambda2"]).replace('"in_fold"', '"full_development"'))
    assert new["lambda2"] == relabelled


# ---------------------------------------------------------------------------------------------
# E2 v3: selection from the decision-time board (review of 0c39a8d, exacta_study.py:449)
# ---------------------------------------------------------------------------------------------

# The v2 E2 family frozen 2026-11-27 (docs/specs/exacta-overload-prereg.md), whose favourite
# came from the common "lowest final Win odds". Signed spec section 5: "at T-2 ... Selection uses
# only the board at t. A selection variable taken from final odds is look-ahead"; section 6: the
# favourite's Win odds are "final for E1, at t for E2". The corrected E2 is a new family.
V2_RULE_HASH = "2d6678f59291476bf3d3028b2818bc00e631dea516bc0f01276ac4e0af8cdb75"
V2_RULE_FIXTURE = FIXTURES / "exacta_131_v2_rule_spec.py"
# The four v2 families E2's correction leaves alone, with their documented frozen hashes.
V2_HASHES = {
    "B2": ("japan/131-exacta-category-v2",
           "eb900ab7144a6c88b5ebf4058d6aaf22570a9a5fb06d7803dbaa7a7cf89bf247"),
    "B1": ("japan/131-exacta-category-b1-v2",
           "fb6844a9a189ee689f6390b6f2fde58a318370b9e73834ab5b28fbad735ec49b"),
    "B3": ("japan/131-exacta-category-b3-v2",
           "493253276d7a803446716325353bb6c18e01409868dc6dbec53eec09a9b4912f"),
    "C": ("japan/131-exacta-logscore-v2",
          "084244a2f3cb69154da1dc97615044b58e0cf534f8077ed1cbd459a98e3650bd"),
}  # fmt: skip
# The keys the E2 correction may change: the selection keys, their provenance note, the link.
E2_SELECTION_KEYS = {
    "favourite", "category", "selection_time", "scopes", "primary_metric", "supersedes",
    "revision",
}  # fmt: skip


def _v2_rule() -> tuple[str, dict]:
    loader = importlib.util.spec_from_file_location("exacta_131_v2_rule_spec", V2_RULE_FIXTURE)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    assert _hash(module.V2_RULE_SPEC) == V2_RULE_HASH  # the frozen v2 record
    return module.V2_RULE_ID, module.V2_RULE_SPEC


def _at_decision_time(text) -> bool:
    return isinstance(text, str) and "T-2" in text and "final" not in text.lower()


def test_e2_takes_the_favourite_and_category_from_the_decision_time_board():
    specs = family_specs()
    e2 = specs[_family_id(specs, "rule")]
    assert e2["selection_time"] == "T-2"  # D2's primary decision time, section 5's "at T-2"
    for key in ("favourite", "category"):  # no final odds in any selection variable
        assert _at_decision_time(e2.get(key)), (key, e2.get(key))
    assert set(e2["scopes"]) == {m["scope"] for m in e2["members"]} == {"all", "headline"}
    headline = e2["scopes"]["headline"]  # favourite odds >= 3.0, read at t, not at the close
    assert _at_decision_time(headline) and "3.0" in headline, headline
    # E1 keeps the final-odds favourite: it is scored on history, where only final odds exist.
    for estimator in ("B2", "B1", "B3", "C"):
        assert "final" in specs[_family_id(specs, estimator)]["favourite"]


def test_corrected_e2_freezes_under_a_new_id_beside_rule_v1_and_rule_v2(tmp_path):
    v1 = _v1_specs()
    v1_id = _v1_id(v1, "rule")
    v2_id, v2 = _v2_rule()
    specs = family_specs()
    e2_id = _family_id(specs, "rule")
    assert e2_id not in {v1_id, v2_id}
    assert e2_id not in V1_HASHES and e2_id not in {fid for fid, _ in V2_HASHES.values()}
    assert _hash(specs[e2_id]) not in {V1_HASHES[v1_id], V2_RULE_HASH}
    assert specs[e2_id]["supersedes"] == v2_id

    reg = Registry(tmp_path / "japan.jsonl", holdout_log=tmp_path / "holdout_log.jsonl")
    assert reg.freeze_family(v1_id, v1[v1_id]) == V1_HASHES[v1_id]
    assert reg.freeze_family(v2_id, v2) == V2_RULE_HASH
    assert reg.freeze_family(e2_id, specs[e2_id]) == _hash(specs[e2_id])  # not already frozen
    assert reg.frozen_spec_hash(v1_id) == V1_HASHES[v1_id]
    assert reg.frozen_spec_hash(v2_id) == V2_RULE_HASH
    reg.verify()


def test_the_e2_correction_changes_only_e2_s_selection_keys():
    _, v2 = _v2_rule()
    specs = family_specs()
    e2 = specs[_family_id(specs, "rule")]
    changed = {k for k in set(v2) | set(e2) if v2.get(k) != e2.get(k)}
    assert changed <= E2_SELECTION_KEYS, sorted(changed - E2_SELECTION_KEYS)
    assert {"favourite", "selection_time", "supersedes"} <= changed
    # the other four families stay the frozen v2 records (ids and hashes), favourite included
    for estimator, (family_id, frozen) in V2_HASHES.items():
        assert _family_id(specs, estimator) == family_id
        assert _hash(specs[family_id]) == frozen, estimator
