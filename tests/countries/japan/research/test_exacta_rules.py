"""#131's E2 ratio-cut rules scored on #130-logged boards (ticket #281).

docs/specs/exacta-overload-prereg.md sections 5-7, with the owner's reframe of 2026-11-27: score
the 16 frozen members of ``japan/131-exacta-rule-v2`` on whatever development meetings are logged,
refusing nothing for small n (the frozen ``min_meetings: 100`` is superseded); zero meetings give
an n = 0 result. Every figure here is synthetic; HORSETRACK_DATA_DIR points at tmp_path and the
registry is a tmp file.

API pinned here (module ``horsetrack.countries.japan.research.exacta_rules``, not yet written):

- Constants: ``CUTS = (0.6, 0.7, 0.8, 0.9)``, ``RULE_CATEGORIES = ("fav_second", "no_fav")``,
  ``SCOPES = ("all", "headline")``, ``STAKE = 100``, ``MINUTES_BEFORE_POST = 2``,
  ``BLOCK_SIZE = 5``, ``FROZEN_E2_HASH`` (the v2 spec hash), ``MAX_TEST_ID = E2_ID + "/max-test"``.
- ``select_combinations(win_odds, exacta_odds, *, rate, lambda2) -> {(category, cut): frozenset}``
  for the 8 (category, cut) pairs. Both boards are the boards at t. p_i = (1/O_i) / sum_j 1/O_j
  over ``win_odds`` (the Win board at t), p_c the discounted Harville exacta of p at ``lambda2``,
  q_c = ``implied_share(100 * odds_c, rate=rate)`` (NaN at or below 110 yen: never selected), the
  favourite the lowest Win odds at t (ties to the lowest number). c is selected for (X, k) when
  its category relative to that favourite is X and q_c / p_c < k. A combination naming a horse not
  on the Win board, or with odds None, is never selected.
- ``own_impact_dividend(dividend, *, pool, n_bets, stake=STAKE, takeout=0.25) -> float``: the
  winner's dividend per 100 yen once ``stake`` is added to each of ``n_bets`` combinations (the
  winner among them), from the payout identity 100 (1 - t) (P + n s) / (W + s) with
  W = (1 - t) P / O and O = dividend / 100 (agrees with ``exacta_impact.own_impact``). No breakage
  or minimum payout is re-applied: neither the spec nor #283's code defines one.
- ``settle(selected, winner, dividend, *, pool, stake=STAKE, takeout=0.25) -> dict`` with
  ``n_bets`` (len(selected)), ``stake`` (stake * n_bets, yen) and ``payout`` (yen:
  stake / 100 * own_impact_dividend(...) when the winner is selected, else 0).
- ``race_scopes(win_odds, *, race_class, race_no) -> frozenset``: always "all"; "headline" when
  the favourite's Win odds *at t* are >= 3.0 and the race is less popular by the frozen proxy
  (race 1-4, or a DEBUT/MAIDEN race), as ``exacta_study.segment_cells``.
- ``score_e2(races, *, registry, git_sha, lambda2, data_builds, n_boot=10000, seed=131,
  holdout_day=is_holdout_day) -> dict``. Each race is a mapping with race_id, race_date (date),
  meeting_id, race_no, race_class, rate, boost_may_be_withheld, win_odds_t {horse: odds},
  exacta_odds_t {(first, second): odds}, final_win_odds, final_exacta_odds, pool_total (the
  race's final logged FORECAST pool, yen) and exacta [(combination, dividend, is_refund)] (the
  paying rows). A race with ``holdout_day("japan", race_date)`` is dropped before anything else
  of it is read; ``exacta_study.exclusion_reason`` (starters = len(win_odds_t)) drops the rest,
  counted in ``result["excluded"]``. Each member (category, cut, scope) is a standalone rule: in
  each in-scope race it stakes ``STAKE`` on its selected set and is settled with its own n.
  Refuses (RuntimeError or ValueError, nothing appended) unless ``registry`` holds E2_ID frozen
  at the hash of ``exacta_study.family_specs()[E2_ID]``. Appends one record per member (family
  E2_ID, status "scored", params with category, cut, scope and spec_hash; a metric "n_bets"
  always, "roi" when the member bet) and, when any race is scored, one SPA + StepM record under
  MAX_TEST_ID (params block_size 5) run over the 16 members with meetings as clusters. Result:
  n_races, n_meetings, excluded, members (16 dicts: category, cut, scope, n_bets, n_races,
  n_meetings, stake, payout, roi, lo, hi), spa (dict with pvalue, n_rules, family_size,
  n_clusters; None with no race), stepm (dict with superior; None with no race).
- ``d_descriptive(races, *, lambda2) -> {category: {"n_races", "mean", "median"}}``: D, the
  full-board ratio sum_S q / sum_S p of ``exacta_consistency.race_ratios`` on the final boards
  (p from final_win_odds as above, dividends 100 * final_exacta_odds), over the three CATEGORIES.
- ``proxy_agreement(races) -> dict`` with n_races, n_days, both_less, proxy_only, tertile_only,
  neither, agreement: "less popular" by the lower tertile of pool_total within each race *day*
  (all venues of the day together) against the frozen proxy.
- ``load_logged_races(*, minutes_before=MINUTES_BEFORE_POST) -> list[dict]``: the race mappings
  above, from ``$HORSETRACK_DATA_DIR/odds_snapshots/jra/<day>/<VENUE>.jsonl.gz`` with its
  ``<VENUE>.races.json`` post times (t = post - minutes_before; a row is available at its
  fetched_utc, else its snapshot_utc, docs/specs/board-at-t.md), and the store's
  ``processed/jra/races.parquet`` and ``dividends.parquet`` (pool FORECAST). Development days
  only: a holdout day's folder is never opened (D14). rate is
  ``takeout.race_takeout(race_id, "FORECAST", race_class_raw=...).payout_rate``.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party.
from __future__ import annotations  # noqa: I001

import hashlib
import json
import math
import statistics
from datetime import date, datetime, timezone
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import Registry
from horsetrack.countries.japan import odds_log, takeout
from horsetrack.countries.japan.research import exacta_impact
from horsetrack.countries.japan.research.exacta_study import E2_ID, family_specs
from horsetrack.countries.japan.research import exacta_rules as er

UTC = timezone.utc
CUTS = (0.6, 0.7, 0.8, 0.9)
RULE_CATS = ("fav_second", "no_fav")
ALL_CATS = ("fav_first", "fav_second", "no_fav")
SCOPES = ("all", "headline")
RATE = 0.75
LAMBDA2 = 0.8104
FROZEN = "2d6678f59291476bf3d3028b2818bc00e631dea516bc0f01276ac4e0af8cdb75"
BUILDS = {"odds_log": "odds_snapshots/jra/FORECAST@aaaa"}


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


def new_registry(tmp_path: Path, *, freeze: bool = True) -> Registry:
    reg = Registry(tmp_path / "japan.jsonl", holdout_log=tmp_path / "holdout_log.jsonl")
    if freeze:
        reg.freeze_family(E2_ID, family_specs()[E2_ID])
    return reg


# ---------------------------------------------------------------------------------------------
# Independent references (first principles; nothing here calls the code under test)
# ---------------------------------------------------------------------------------------------


def ref_p(win_odds: dict[int, float]) -> dict[int, float]:
    inv = {h: 1 / o for h, o in win_odds.items()}
    total = sum(inv.values())
    return {h: v / total for h, v in inv.items()}


def ref_harville(p: dict[int, float], lambda2: float) -> dict[tuple[int, int], float]:
    """P(first = i, second = j) = p_i p_j^l2 / sum_{k != i} p_k^l2."""
    out = {}
    for i, j in permutations(p, 2):
        out[(i, j)] = p[i] * p[j] ** lambda2 / sum(p[k] ** lambda2 for k in p if k != i)
    return out


def ref_q(odds: float, rate: float) -> float:
    dividend = 100 * odds
    return math.nan if dividend <= 110 else rate / ((dividend + 5) / 100)


def ref_fav(win_odds: dict[int, float]) -> int:
    return min(win_odds, key=lambda h: (win_odds[h], h))


def ref_cat(combo: tuple[int, int], fav: int) -> str:
    if combo[0] == fav:
        return "fav_first"
    if combo[1] == fav:
        return "fav_second"
    return "no_fav"


def ref_select(win_odds, exacta_odds, rate, lambda2) -> dict[tuple[str, float], set]:
    p = ref_p(win_odds)
    h = ref_harville(p, lambda2)
    fav = ref_fav(win_odds)
    out = {(c, k): set() for c in RULE_CATS for k in CUTS}
    for combo, odds in exacta_odds.items():
        if odds is None or combo[0] not in p or combo[1] not in p:
            continue
        cat = ref_cat(combo, fav)
        if cat not in RULE_CATS:
            continue
        q = ref_q(odds, rate)
        if math.isnan(q):
            continue
        for k in CUTS:
            if q / h[combo] < k:
                out[(cat, k)].add(combo)
    return out


def odds_for_ratio(p_c: float, ratio: float, rate: float) -> float:
    """The exacta odds whose implied share is ratio * p_c (inverse of ref_q)."""
    return (100 * rate / (ratio * p_c) - 5) / 100


def ref_dividend(dividend, pool, n, stake=100, t=0.25) -> float:
    w = (1 - t) * pool / (dividend / 100)
    return 100 * (1 - t) * (pool + n * stake) / (w + stake)


def ref_headline(win_odds, race_class, race_no) -> bool:
    less = race_no <= 4 or race_class in ("DEBUT", "MAIDEN")
    return win_odds[ref_fav(win_odds)] >= 3.0 and less


# ---------------------------------------------------------------------------------------------
# Synthetic boards
# ---------------------------------------------------------------------------------------------

# Ratios kept away from every cut, so float noise cannot flip a selection.
RATIO_POOL = (0.52, 0.57, 0.63, 0.66, 0.72, 0.77, 0.83, 0.86, 0.93, 0.97, 1.05, 1.2, 1.5)


def make_board(rng, n: int, kind: str, rate: float, lambda2: float):
    """(win_odds, exacta_odds) at t: each exacta priced at a ratio drawn from RATIO_POOL."""
    if kind == "strong":
        rest = rng.dirichlet(np.full(n - 1, 3.0)) * 0.6
        p = np.concatenate([[0.4], rest])
    else:  # flat field: the favourite's odds land at 3.0 or more
        p = rng.dirichlet(np.full(n, 25.0))
    order = rng.permutation(n)
    win = {}
    for slot, h in enumerate(range(1, n + 1)):
        win[h] = max(1.1, round(0.8 / p[order[slot]], 1))
    model = ref_harville(ref_p(win), lambda2)
    exacta = {c: odds_for_ratio(model[c], float(rng.choice(RATIO_POOL)), rate) for c in model}
    return win, exacta


def make_races(seed: int, days: list[date], venues=("TOK", "KYO")):
    """Three races per meeting (race 1 MAIDEN, race 5 C1, race 9 DEBUT or OP)."""
    rng = np.random.default_rng(seed)
    races = []
    for day in days:
        for venue in venues:
            for race_no, race_class, kind in (
                (1, "MAIDEN", "flat" if rng.random() < 0.7 else "strong"),
                (5, "C1", "flat" if rng.random() < 0.5 else "strong"),
                (9, "DEBUT" if rng.random() < 0.5 else "OP", "flat"),
            ):
                n = int(rng.integers(6, 10))
                win, exacta = make_board(rng, n, kind, RATE, LAMBDA2)
                final_win = {h: max(1.1, round(o * rng.uniform(0.9, 1.1), 1)) for h, o in
                             win.items()}  # fmt: skip
                final_exacta = {c: o * rng.uniform(0.8, 1.25) for c, o in exacta.items()}
                chosen = sorted(
                    ref_select(win, exacta, RATE, LAMBDA2)[("no_fav", 0.9)]
                    | ref_select(win, exacta, RATE, LAMBDA2)[("fav_second", 0.9)]
                )
                combos = sorted(exacta)
                if chosen and rng.random() < 0.6:
                    winner = chosen[int(rng.integers(len(chosen)))]
                else:
                    winner = combos[int(rng.integers(len(combos)))]
                dividend = max(120.0, 10 * math.floor(100 * final_exacta[winner] / 10))
                races.append(
                    {
                        "race_id": f"jra:{day.isoformat()}:{venue}:{race_no:02d}",
                        "race_date": day,
                        "meeting_id": f"jra:{day.isoformat()}:{venue}",
                        "race_no": race_no,
                        "race_class": race_class,
                        "rate": RATE,
                        "boost_may_be_withheld": False,
                        "win_odds_t": win,
                        "exacta_odds_t": exacta,
                        "final_win_odds": final_win,
                        "final_exacta_odds": final_exacta,
                        "pool_total": float(rng.choice([3e5, 8e5, 2.5e6, 1.2e7])),
                        "exacta": [(winner, dividend, False)],
                    }
                )
    return races


def expected_members(races) -> dict[tuple, dict]:
    out = {}
    for cat in RULE_CATS:
        for k in CUTS:
            for scope in SCOPES:
                n_bets = n_races = 0
                stake = payout = 0.0
                meetings = set()
                for race in races:
                    win = race["win_odds_t"]
                    if scope == "headline" and not ref_headline(
                        win, race["race_class"], race["race_no"]
                    ):
                        continue
                    chosen = ref_select(win, race["exacta_odds_t"], race["rate"], LAMBDA2)
                    sel = chosen[(cat, k)]
                    if not sel:
                        continue
                    ((winner, dividend, _),) = race["exacta"]
                    n_bets += len(sel)
                    n_races += 1
                    meetings.add(race["meeting_id"])
                    stake += 100 * len(sel)
                    if winner in sel:
                        payout += ref_dividend(dividend, race["pool_total"], len(sel))
                out[(cat, k, scope)] = {
                    "n_bets": n_bets, "n_races": n_races, "n_meetings": len(meetings),
                    "stake": stake, "payout": payout,
                }  # fmt: skip
    return out


DAYS = [date(2026, 11, 14), date(2026, 11, 15), date(2026, 11, 21)]
assert not any(is_holdout_day("japan", d) for d in DAYS)


# ---------------------------------------------------------------------------------------------
# 1. Selection on the board at t
# ---------------------------------------------------------------------------------------------


def as_sets(selected) -> dict:
    return {key: set(value) for key, value in selected.items()}


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("rate, lambda2", [(0.75, 0.8104), (0.75, 1.0), (0.80, 0.8104)])
def test_selection_matches_the_ratio_cut_on_varied_boards(seed, rate, lambda2):
    rng = np.random.default_rng(100 + seed)
    win, exacta = make_board(rng, int(rng.integers(5, 13)), ("strong", "flat")[seed % 2],
                             rate, lambda2)  # fmt: skip
    got = er.select_combinations(win, exacta, rate=rate, lambda2=lambda2)
    assert set(got) == {(c, k) for c in RULE_CATS for k in CUTS}
    assert as_sets(got) == ref_select(win, exacta, rate, lambda2)
    assert any(got.values())  # the fixture selects something


def test_selection_depends_on_lambda2_and_the_rate():
    rng = np.random.default_rng(7)
    win, exacta = make_board(rng, 10, "strong", 0.75, 0.8104)
    base = as_sets(er.select_combinations(win, exacta, rate=0.75, lambda2=0.8104))
    plain = as_sets(er.select_combinations(win, exacta, rate=0.75, lambda2=1.0))
    richer = as_sets(er.select_combinations(win, exacta, rate=0.80, lambda2=0.8104))
    assert plain == ref_select(win, exacta, 0.75, 1.0) and plain != base
    assert richer == ref_select(win, exacta, 0.80, 0.8104) and richer != base


def test_selections_are_nested_in_the_cut_and_in_their_category():
    for seed in range(5):
        rng = np.random.default_rng(200 + seed)
        win, exacta = make_board(rng, 9, "strong", RATE, LAMBDA2)
        got = er.select_combinations(win, exacta, rate=RATE, lambda2=LAMBDA2)
        fav = ref_fav(win)
        for cat in RULE_CATS:
            for lo, hi in zip(CUTS, CUTS[1:], strict=False):
                assert set(got[(cat, lo)]) <= set(got[(cat, hi)])
            assert all(ref_cat(c, fav) == cat for c in got[(cat, 0.9)])
        assert not any(c[0] == fav for sel in got.values() for c in sel)  # fav_first never


def test_selection_on_a_hand_built_board():
    """Favourite 2 (2.4); every exacta priced at a stated ratio to the Harville probability."""
    win = {1: 5.0, 2: 2.4, 3: 4.0, 4: 8.0, 5: 12.0}
    model = ref_harville(ref_p(win), LAMBDA2)
    ratios = {(1, 2): 0.55, (3, 2): 0.75, (4, 2): 0.95, (5, 2): 0.85,  # fav_second
              (1, 3): 0.65, (3, 1): 0.88, (4, 5): 0.58, (5, 4): 1.4,   # no_fav
              (2, 1): 0.3, (2, 3): 0.4}  # fmt: skip # fav_first: never bought
    exacta = {c: odds_for_ratio(model[c], r, RATE) for c, r in ratios.items()}
    got = as_sets(er.select_combinations(win, exacta, rate=RATE, lambda2=LAMBDA2))
    assert got[("fav_second", 0.6)] == {(1, 2)}
    assert got[("fav_second", 0.7)] == {(1, 2)}
    assert got[("fav_second", 0.8)] == {(1, 2), (3, 2)}
    assert got[("fav_second", 0.9)] == {(1, 2), (3, 2), (5, 2)}
    assert got[("no_fav", 0.6)] == {(4, 5)}
    assert got[("no_fav", 0.7)] == {(4, 5), (1, 3)}
    assert got[("no_fav", 0.8)] == {(4, 5), (1, 3)}
    assert got[("no_fav", 0.9)] == {(4, 5), (1, 3), (3, 1)}


def test_the_favourite_is_taken_from_the_win_board_at_t_with_ties_to_the_lowest_number():
    win = {1: 6.0, 2: 3.0, 3: 3.0, 4: 7.5, 5: 9.0, 6: 14.0}  # 2 and 3 tie: favourite 2
    model = ref_harville(ref_p(win), 1.0)
    exacta = {c: odds_for_ratio(model[c], 0.5, RATE) for c in model}  # everything is cheap
    got = er.select_combinations(win, exacta, rate=RATE, lambda2=1.0)
    assert set(got[("fav_second", 0.6)]) == {(h, 2) for h in win if h != 2}
    assert set(got[("no_fav", 0.6)]) == {c for c in model if 2 not in c}
    assert (3, 2) in got[("fav_second", 0.6)] and (2, 3) not in got[("no_fav", 0.6)]


def test_unpriced_unidentified_and_off_board_combinations_are_never_selected():
    win = {1: 2.0, 2: 4.5, 3: 6.0, 4: 9.0, 5: 15.0}
    model = ref_harville(ref_p(win), LAMBDA2)
    exacta = {c: odds_for_ratio(model[c], 0.5, RATE) for c in model}
    exacta[(3, 4)] = None  # unpriced
    exacta[(4, 3)] = 1.1  # pays 110 yen: share not identified
    exacta[(6, 3)] = 300.0  # horse 6 is not on the Win board (scratched)
    exacta[(3, 6)] = 300.0
    got = er.select_combinations(win, exacta, rate=RATE, lambda2=LAMBDA2)
    chosen = set().union(*map(set, got.values()))
    assert not {(3, 4), (4, 3), (6, 3), (3, 6)} & chosen
    assert as_sets(got) == ref_select(win, exacta, RATE, LAMBDA2)


# ---------------------------------------------------------------------------------------------
# 2. Own impact and settlement
# ---------------------------------------------------------------------------------------------

# (dividend, pool, n, stake, takeout)
IMPACT_CASES = [
    (1230.0, 1_000_000, 1, 100, 0.25),
    (1230.0, 1_000_000, 6, 100, 0.25),
    (25_670.0, 300_000, 1, 100, 0.25),
    (25_670.0, 300_000, 14, 100, 0.25),
    (480.0, 40_000_000, 3, 100, 0.25),
    (8_800.0, 2_500_000, 2, 100, 0.20),
    (8_800.0, 2_500_000, 2, 1000, 0.25),
    (150.0, 1_000_000, 30, 100, 0.25),  # short price, wide box: the dividend can rise
]


@pytest.mark.parametrize("dividend, pool, n, stake, t", IMPACT_CASES)
def test_own_impact_dividend_is_the_payout_identity(dividend, pool, n, stake, t):
    got = er.own_impact_dividend(dividend, pool=pool, n_bets=n, stake=stake, takeout=t)
    assert got == pytest.approx(ref_dividend(dividend, pool, n, stake, t), rel=1e-9)
    via_283 = dividend * (1 + exacta_impact.own_impact(pool, dividend / 100, stake, n, t))
    assert got == pytest.approx(via_283, rel=1e-9)


def test_one_bet_always_lowers_the_dividend_and_more_so_in_a_small_pool():
    for dividend in (130.0, 990.0, 12_340.0, 154_000.0):
        small = er.own_impact_dividend(dividend, pool=200_000, n_bets=1)
        large = er.own_impact_dividend(dividend, pool=50_000_000, n_bets=1)
        assert small < large < dividend


@pytest.mark.parametrize("dividend, pool, n, stake, t", IMPACT_CASES)
def test_settle_pays_the_impacted_dividend_only_when_the_winner_is_selected(
    dividend, pool, n, stake, t
):
    winner = (3, 5)
    others = [(a, b) for a in range(1, 9) for b in range(1, 9) if a != b and (a, b) != winner]
    hit = {winner, *others[: n - 1]}
    miss = set(others[:n])
    got = er.settle(hit, winner, dividend, pool=pool, stake=stake, takeout=t)
    assert got["n_bets"] == n and got["stake"] == pytest.approx(stake * n)
    expected = stake / 100 * ref_dividend(dividend, pool, n, stake, t)
    assert got["payout"] == pytest.approx(expected, rel=1e-9)
    lost = er.settle(miss, winner, dividend, pool=pool, stake=stake, takeout=t)
    assert lost["n_bets"] == n and lost["stake"] == pytest.approx(stake * n)
    assert lost["payout"] == 0


def test_settle_with_nothing_selected_costs_and_pays_nothing():
    got = er.settle(frozenset(), (1, 2), 1230.0, pool=1_000_000)
    assert (got["n_bets"], got["stake"], got["payout"]) == (0, 0, 0)


def test_settle_defaults_to_a_hundred_yen_and_the_exacta_takeout():
    got = er.settle({(2, 1), (3, 1)}, (2, 1), 5_000.0, pool=600_000)
    assert er.STAKE == 100
    assert got["stake"] == 200
    assert got["payout"] == pytest.approx(ref_dividend(5_000.0, 600_000, 2), rel=1e-9)
    assert got["payout"] < 5_000.0


# ---------------------------------------------------------------------------------------------
# 3. Scope: the headline cell at t
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fav_odds", [1.8, 2.9, 3.0, 4.4, 7.0])
@pytest.mark.parametrize("race_no", [1, 4, 5, 11])
@pytest.mark.parametrize("race_class", ["DEBUT", "MAIDEN", "C1", "OP"])
def test_race_scopes_follow_the_favourite_at_t_and_the_frozen_proxy(fav_odds, race_no, race_class):
    win = {1: 12.0, 2: fav_odds, 3: 15.0, 4: 20.0, 5: 25.0, 6: 30.0}
    got = er.race_scopes(win, race_class=race_class, race_no=race_no)
    assert set(got) == ({"all", "headline"} if ref_headline(win, race_class, race_no) else {"all"})


# ---------------------------------------------------------------------------------------------
# 4. Scoring end to end
# ---------------------------------------------------------------------------------------------


def by_member(result) -> dict[tuple, dict]:
    members = {(m["category"], m["cut"], m["scope"]): m for m in result["members"]}
    assert len(members) == len(result["members"]) == 16
    return members


def test_the_frozen_e2_hash_is_the_v2_spec():
    spec = family_specs()[E2_ID]
    digest = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert er.FROZEN_E2_HASH == FROZEN == digest
    assert er.MAX_TEST_ID == f"{E2_ID}/max-test"
    assert er.BLOCK_SIZE == 5 and er.MINUTES_BEFORE_POST == 2
    assert tuple(er.CUTS) == CUTS and tuple(er.RULE_CATEGORIES) == RULE_CATS
    assert tuple(er.SCOPES) == SCOPES


def test_score_e2_reports_every_member_against_an_independent_reference(tmp_path):
    races = make_races(281, DAYS)
    expected = expected_members(races)
    assert all(e["n_bets"] > 0 for e in expected.values())  # the fixture exercises every member
    assert any(e["payout"] > 0 for e in expected.values())
    reg = new_registry(tmp_path)
    result = er.score_e2(races, registry=reg, git_sha="abc1234", lambda2=LAMBDA2,
                         data_builds=BUILDS, n_boot=50)  # fmt: skip
    assert result["n_races"] == len(races) == 18
    assert result["n_meetings"] == 6
    members = by_member(result)
    assert set(members) == set(expected)
    for key, want in expected.items():
        got = members[key]
        assert got["n_bets"] == want["n_bets"], key
        assert got["n_races"] == want["n_races"], key
        assert got["n_meetings"] == want["n_meetings"], key
        assert got["stake"] == pytest.approx(want["stake"]), key
        assert got["payout"] == pytest.approx(want["payout"], rel=1e-9), key
        assert got["roi"] == pytest.approx(want["payout"] / want["stake"], rel=1e-9), key
    widest = members[("no_fav", 0.9, "all")]
    assert widest["n_meetings"] == 6
    assert math.isfinite(widest["lo"]) and math.isfinite(widest["hi"])
    assert widest["lo"] <= widest["roi"] <= widest["hi"]


def test_score_e2_records_every_member_and_the_max_test_against_the_frozen_hash(tmp_path):
    races = make_races(281, DAYS)
    expected = expected_members(races)
    reg = new_registry(tmp_path)
    result = er.score_e2(races, registry=reg, git_sha="abc1234", lambda2=LAMBDA2,
                         data_builds=BUILDS, n_boot=50)  # fmt: skip
    records = reg.records(E2_ID)
    assert len(records) == 16
    seen = set()
    for r in records:
        key = (r.params["category"], r.params["cut"], r.params["scope"])
        seen.add(key)
        assert r.status == "scored"
        assert r.params["spec_hash"] == FROZEN
        metrics = {m.metric: m for m in r.metrics}
        assert metrics["n_bets"].value == expected[key]["n_bets"]
        want_roi = expected[key]["payout"] / expected[key]["stake"]
        assert metrics["roi"].value == pytest.approx(want_roi, rel=1e-9)
    assert seen == set(expected)

    (max_test,) = reg.records(f"{E2_ID}/max-test")
    assert max_test.params["block_size"] == 5
    assert result["spa"]["n_rules"] == 16 and result["spa"]["family_size"] == 16
    assert result["spa"]["n_clusters"] == 6
    assert 0.0 <= result["spa"]["pvalue"] <= 1.0
    assert set(result["stepm"]["superior"]) <= {r.variant for r in records}
    reg.verify()


def test_score_e2_rescores_as_more_meetings_are_logged(tmp_path):
    reg = new_registry(tmp_path)
    early = make_races(281, DAYS)[:6]  # the first day's two meetings
    first = er.score_e2(early, registry=reg, git_sha="abc1234", lambda2=LAMBDA2,
                        data_builds=BUILDS, n_boot=50)  # fmt: skip
    later = make_races(281, DAYS)
    second = er.score_e2(later, registry=reg, git_sha="abc1234", lambda2=LAMBDA2,
                         data_builds={"odds_log": "odds_snapshots/jra/FORECAST@bbbb"},
                         n_boot=50)  # fmt: skip
    assert (first["n_meetings"], second["n_meetings"]) == (2, 6)
    assert len(reg.records(E2_ID)) == 32
    assert reg.family_size(E2_ID) == 16  # a re-score is another run, not another variant
    assert second["spa"]["family_size"] == 16
    assert len(reg.records(f"{E2_ID}/max-test")) == 2
    want = expected_members(early)
    for key, got in by_member(first).items():
        assert got["n_bets"] == want[key]["n_bets"]


def test_a_few_meetings_are_scored_not_refused(tmp_path):
    races = [r for r in make_races(281, DAYS) if r["meeting_id"] == "jra:2026-11-14:TOK"]
    reg = new_registry(tmp_path)
    result = er.score_e2(races, registry=reg, git_sha="abc1234", lambda2=LAMBDA2,
                         data_builds=BUILDS, n_boot=50)  # fmt: skip
    assert result["n_meetings"] == 1 and result["n_races"] == 3
    want = expected_members(races)
    for key, got in by_member(result).items():
        assert got["n_bets"] == want[key]["n_bets"]
    assert len(reg.records(E2_ID)) == 16


def test_zero_logged_meetings_give_an_n_zero_result_and_records(tmp_path):
    reg = new_registry(tmp_path)
    result = er.score_e2([], registry=reg, git_sha="abc1234", lambda2=LAMBDA2,
                         data_builds=BUILDS, n_boot=50)  # fmt: skip
    assert result["n_meetings"] == 0 and result["n_races"] == 0
    assert result["spa"] is None and result["stepm"] is None
    for m in by_member(result).values():
        assert (m["n_bets"], m["n_races"], m["n_meetings"]) == (0, 0, 0)
    records = reg.records(E2_ID)
    assert len(records) == 16
    for r in records:
        assert r.metrics and all(m.n == 0 for m in r.metrics)
        assert "roi" not in {m.metric for m in r.metrics}


class Poison(dict):
    """A board that fails the test the moment anything reads it."""

    def _boom(self, *args, **kwargs):
        raise AssertionError("a holdout race's board or dividends were read")

    __iter__ = __len__ = __getitem__ = __bool__ = __contains__ = _boom
    items = keys = values = get = copy = _boom


def poisoned(race: dict, day: date, rid: str) -> dict:
    out = dict(race, race_id=rid, race_date=day, meeting_id=f"jra:{day.isoformat()}:NAK")
    for key in ("win_odds_t", "exacta_odds_t", "final_win_odds", "final_exacta_odds", "exacta"):
        out[key] = Poison()
    return out


def test_holdout_races_are_dropped_before_their_boards_are_read(tmp_path):
    races = make_races(281, DAYS)
    calls = []

    def holdout_day(country, day):
        assert country == "japan" and isinstance(day, date)
        calls.append(day)
        return day == date(2026, 11, 22)

    held = poisoned(races[0], date(2026, 11, 22), "jra:2026-11-22:NAK:01")
    reg = new_registry(tmp_path)
    result = er.score_e2([*races, held], registry=reg, git_sha="abc1234", lambda2=LAMBDA2,
                         data_builds=BUILDS, n_boot=50, holdout_day=holdout_day)  # fmt: skip
    assert date(2026, 11, 22) in calls
    assert result["n_races"] == 18 and result["n_meetings"] == 6


def test_the_real_holdout_calendar_is_the_default(tmp_path):
    held_day = date(2026, 11, 29)
    assert is_holdout_day("japan", held_day)
    races = make_races(281, DAYS[:1])
    held = poisoned(races[0], held_day, "jra:2026-11-29:NAK:01")
    reg = new_registry(tmp_path)
    result = er.score_e2([held], registry=reg, git_sha="abc1234", lambda2=LAMBDA2,
                         data_builds=BUILDS, n_boot=50)  # fmt: skip
    assert result["n_races"] == 0 and result["n_meetings"] == 0


@pytest.mark.parametrize("state", ["not frozen", "another spec"])
def test_scoring_refuses_unless_e2_is_frozen_at_its_hash(tmp_path, state):
    reg = new_registry(tmp_path, freeze=False)
    if state == "another spec":
        spec = family_specs()[E2_ID]
        spec["cuts"] = [0.5, 0.6, 0.7, 0.8]
        reg.freeze_family(E2_ID, spec)
    with pytest.raises((RuntimeError, ValueError)):
        er.score_e2(make_races(281, DAYS[:1]), registry=reg, git_sha="abc1234",
                    lambda2=LAMBDA2, data_builds=BUILDS, n_boot=50)  # fmt: skip
    assert reg.records() == []


def test_scoring_selects_on_the_board_at_t_not_the_final_board(tmp_path):
    """Race 1 of a maiden card: the favourite is 2.4 at t (not headline) and 3.4 at the off
    (headline), a different horse; the final board's ratios would select other combinations."""
    rng = np.random.default_rng(5)
    races = make_races(281, DAYS[:1])
    race = races[0]
    win_t = {1: 2.4, 2: 3.4, 3: 5.0, 4: 7.0, 5: 11.0, 6: 16.0, 7: 30.0}
    win_final = {1: 4.0, 2: 3.4, 3: 5.0, 4: 7.0, 5: 9.0, 6: 16.0, 7: 30.0}
    model_t = ref_harville(ref_p(win_t), LAMBDA2)
    model_f = ref_harville(ref_p(win_final), LAMBDA2)
    race["win_odds_t"] = win_t
    race["exacta_odds_t"] = {c: odds_for_ratio(model_t[c], float(rng.choice(RATIO_POOL)), RATE)
                             for c in model_t}  # fmt: skip
    race["final_win_odds"] = win_final
    race["final_exacta_odds"] = {c: odds_for_ratio(model_f[c], float(rng.choice(RATIO_POOL)),
                                                   RATE) for c in model_f}  # fmt: skip
    sel_t = ref_select(win_t, race["exacta_odds_t"], RATE, LAMBDA2)
    sel_f = ref_select(win_final, race["final_exacta_odds"], RATE, LAMBDA2)
    assert sel_t != sel_f
    assert not ref_headline(win_t, "MAIDEN", 1) and ref_headline(win_final, "MAIDEN", 1)
    winner = sorted(sel_t[("no_fav", 0.9)] - sel_f[("no_fav", 0.9)])[0]
    race["exacta"] = [(winner, 4_560.0, False)]
    reg = new_registry(tmp_path)
    result = er.score_e2([race], registry=reg, git_sha="abc1234", lambda2=LAMBDA2,
                         data_builds=BUILDS, n_boot=50)  # fmt: skip
    members = by_member(result)
    for (cat, k, scope), got in members.items():
        want = 0 if scope == "headline" else len(sel_t[(cat, k)])
        assert got["n_bets"] == want, (cat, k, scope)
    n = len(sel_t[("no_fav", 0.9)])
    assert members[("no_fav", 0.9, "all")]["payout"] == pytest.approx(
        ref_dividend(4_560.0, race["pool_total"], n), rel=1e-9
    )


def test_excluded_races_place_no_bets_and_are_counted(tmp_path):
    races = make_races(281, DAYS[:1])
    keep = races[:2]
    base = races[2]
    boost = dict(base, race_id="jra:2026-11-14:TOK:10", boost_may_be_withheld=True)
    refund = dict(base, race_id="jra:2026-11-14:TOK:11",
                  exacta=[(base["exacta"][0][0], 100.0, True)])  # fmt: skip
    heat = dict(base, race_id="jra:2026-11-14:TOK:12",
                exacta=[((1, 2), 900.0, False), ((2, 1), 1500.0, False)])  # fmt: skip
    small_win = dict(list(base["win_odds_t"].items())[:4])
    small = dict(base, race_id="jra:2026-11-14:KYO:12", win_odds_t=small_win)
    reg = new_registry(tmp_path)
    result = er.score_e2([*keep, boost, refund, heat, small], registry=reg, git_sha="abc1234",
                         lambda2=LAMBDA2, data_builds=BUILDS, n_boot=50)  # fmt: skip
    assert result["n_races"] == 2
    assert {k: v for k, v in result["excluded"].items() if v} == {
        "boost_may_be_withheld": 1, "refund": 1, "dead_heat": 1, "under_5_starters": 1}  # fmt: skip
    want = expected_members(keep)
    for key, got in by_member(result).items():
        assert got["n_bets"] == want[key]["n_bets"]
        assert got["payout"] == pytest.approx(want[key]["payout"], rel=1e-9)


# ---------------------------------------------------------------------------------------------
# 5. D descriptive and the proxy check
# ---------------------------------------------------------------------------------------------


def d_race(rid: str, win: dict, ratios: dict[str, float]) -> dict:
    model = ref_harville(ref_p(win), LAMBDA2)
    fav = ref_fav(win)
    exacta = {c: odds_for_ratio(model[c], ratios[ref_cat(c, fav)], RATE) for c in model}
    return {"race_id": rid, "race_date": DAYS[0], "meeting_id": "jra:2026-11-14:TOK",
            "race_no": 1, "race_class": "C1", "rate": RATE, "final_win_odds": win,
            "final_exacta_odds": exacta, "pool_total": 1e6}  # fmt: skip


def test_d_is_the_full_board_category_ratio_on_the_final_boards():
    races = [
        d_race("a", {1: 2.0, 2: 4.0, 3: 6.5, 4: 9.0, 5: 14.0, 6: 22.0},
               {"fav_first": 1.3, "fav_second": 1.0, "no_fav": 1.0}),
        d_race("b", {1: 8.0, 2: 3.1, 3: 3.6, 4: 6.0, 5: 11.0, 6: 19.0, 7: 40.0},
               {"fav_first": 1.1, "fav_second": 0.9, "no_fav": 1.0}),
        d_race("c", {1: 5.5, 2: 7.0, 3: 1.7, 4: 12.0, 5: 25.0},
               {"fav_first": 2.0, "fav_second": 0.6, "no_fav": 0.8}),
    ]  # fmt: skip
    got = er.d_descriptive(races, lambda2=LAMBDA2)
    assert set(got) == set(ALL_CATS)
    want = {"fav_first": [1.3, 1.1, 2.0], "fav_second": [1.0, 0.9, 0.6], "no_fav": [1.0, 1.0, 0.8]}
    for cat, values in want.items():
        assert got[cat]["n_races"] == 3
        assert got[cat]["mean"] == pytest.approx(statistics.mean(values), rel=1e-9)
        assert got[cat]["median"] == pytest.approx(statistics.median(values), rel=1e-9)


def p_race(day: date, venue: str, race_no: int, race_class: str, pool: float) -> dict:
    return {"race_id": f"jra:{day.isoformat()}:{venue}:{race_no:02d}", "race_date": day,
            "meeting_id": f"jra:{day.isoformat()}:{venue}", "race_no": race_no,
            "race_class": race_class, "pool_total": pool}  # fmt: skip


def test_proxy_agreement_with_the_lower_pool_tertile_of_each_day():
    d1, d2 = DAYS[0], DAYS[1]
    races = []
    for no in range(1, 7):  # day 1: Tokyo's pools dwarf Kyoto's; the tertile is over the day
        races.append(p_race(d1, "TOK", no, "MAIDEN" if no == 6 else "C1", no * 1e7))
        races.append(p_race(d1, "KYO", no, "C1", no * 1e5))
    for no, pool in zip(range(1, 7), [6e6, 1e6, 5e6, 4e6, 3e6, 5e5], strict=True):
        races.append(p_race(d2, "TOK", no, "DEBUT" if no == 5 else "C2", pool))
    got = er.proxy_agreement(races)
    # day 1: tertile = KYO 1-4; proxy = races 1-4 at both venues and TOK 6 (maiden)
    # day 2: tertile = races 6 and 2; proxy = races 1-5
    assert got["n_races"] == 18 and got["n_days"] == 2
    assert got["both_less"] == 4 + 1
    assert got["proxy_only"] == 5 + 4
    assert got["tertile_only"] == 0 + 1
    assert got["neither"] == 3 + 0
    assert got["agreement"] == pytest.approx(8 / 18)


def test_proxy_agreement_is_perfect_when_the_proxy_marks_the_cheapest_third():
    races = [p_race(DAYS[2], "HSN", no, "C1", (13 - no) * 1e6) for no in range(1, 13)]
    races = [dict(r, race_no=13 - r["race_no"]) for r in races]  # pools rise with race_no
    got = er.proxy_agreement(races)
    assert (got["both_less"], got["neither"], got["proxy_only"], got["tertile_only"]) == (
        4, 8, 0, 0)  # fmt: skip
    assert got["agreement"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------
# The logged boards: T-2 from the odds log, outcomes from the store, holdout days unopened
# ---------------------------------------------------------------------------------------------

LOG_DAY = date(2026, 11, 21)
HOLD_DAY = date(2026, 11, 29)
assert not is_holdout_day("japan", LOG_DAY) and is_holdout_day("japan", HOLD_DAY)
FINAL_TEXT = "最終オッズ"


def at(day: date, hh: int, mm: int, ss: int = 0) -> str:
    return datetime(day.year, day.month, day.day, hh, mm, ss, tzinfo=UTC).isoformat()


def capture(day, venue, no, snap, win, forecast, pool, *, final=False, fetched=None, scr=()):
    rid = f"jra:{day.isoformat()}:{venue}:{no:02d}"
    extra = {"page_state": FINAL_TEXT, "source_update_utc": None} if final else {
        "source_update_utc": snap}  # fmt: skip
    if fetched is not None:
        extra["fetched_utc"] = fetched
    rows = []
    for h, o in win.items():
        rows.append({"snapshot_utc": snap, "race_id": rid, "pool": "WIN", "combination": str(h),
                     "odds": o, "status_raw": None, **extra})  # fmt: skip
    for h in scr:
        rows.append({"snapshot_utc": snap, "race_id": rid, "pool": "WIN", "combination": str(h),
                     "odds": None, "status_raw": "SCR", **extra})  # fmt: skip
    for (a, b), o in forecast.items():
        rows.append(
            {
                "snapshot_utc": snap,
                "race_id": rid,
                "pool": "FORECAST",
                "combination": f"{a},{b}",
                "odds": o,
                "status_raw": None,
                **extra,
            }
        )
    rows.append({"snapshot_utc": snap, "race_id": rid, "pool": "FORECAST", "combination":
                 odds_log.POOL_TOTAL, "odds": None, "status_raw": None, "investment": pool,
                 **extra})  # fmt: skip
    rows.append(
        {
            "snapshot_utc": snap,
            "race_id": rid,
            "pool": "QUINELLA",
            "combination": "1,2",
            "odds": 99.0,
            "status_raw": None,
            **extra,
        }
    )  # another pool: never exacta
    return rows


def forecast_board(horses, scale: float) -> dict:
    return {(a, b): round(scale * (a + 2 * b), 1) for a in horses for b in horses if a != b}


def ratio_board(win: dict) -> dict:
    """Exacta odds (to 0.1) at ratios 0.55-1.05 to Harville on `win`, so some are selected."""
    model = ref_harville(ref_p(win), LAMBDA2)
    return {(a, b): round(odds_for_ratio(model[(a, b)], 0.55 + 0.1 * ((3 * a + b) % 6), RATE), 1)
            for a, b in model}  # fmt: skip


def write_store(data: Path, races: list[dict], dividends: list[dict]) -> None:
    base = data / "processed" / "jra"
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(races).to_parquet(base / "races.parquet", index=False)
    pd.DataFrame(dividends).to_parquet(base / "dividends.parquet", index=False)


def store_race(day, venue, no, race_class) -> dict:
    return {"race_id": f"jra:{day.isoformat()}:{venue}:{no:02d}", "race_index": no,
            "race_name": f"R{no}", "race_class": race_class, "race_class_raw": "3歳未勝利",
            "meeting_id": f"jra:{day.isoformat()}:{venue}", "race_date": day, "venue": venue,
            "race_no": no, "season": "2026", "field_size": 6, "status": "run"}  # fmt: skip


def build_logged_day(data: Path) -> dict:
    day_dir = data / "odds_snapshots" / "jra" / LOG_DAY.isoformat()
    log = day_dir / "TOK.jsonl.gz"
    post = {1: datetime(2026, 11, 21, 1, 0, tzinfo=UTC), 2: datetime(2026, 11, 21, 1, 30,
                                                                      tzinfo=UTC)}  # fmt: skip
    day_dir.mkdir(parents=True)
    odds_log.write_post_times(
        day_dir / "TOK.races.json", post, datetime(2026, 11, 20, 23, 0, tzinfo=UTC)
    )
    h6 = range(1, 7)
    h5 = range(1, 6)  # horse 6 scratched from 00:57 on race 1
    win_a = {1: 2.2, 2: 4.0, 3: 6.1, 4: 9.5, 5: 14.0}
    win_b = {1: 3.3, 2: 2.1, 3: 6.0, 4: 9.0, 5: 13.0}
    win_c = {1: 2.0, 2: 4.4, 3: 6.3, 4: 9.1, 5: 15.0}
    win_f = {1: 2.1, 2: 4.2, 3: 6.6, 4: 8.8, 5: 16.0}
    fc_a = ratio_board(win_a)
    fc_f = forecast_board(h5, 1.4)
    # race 2 (race_no 2, less popular): favourite 2 at 3.2 at t (headline), 3 at 2.7 at the off
    win_r2_t = {1: 5.0, 2: 3.2, 3: 3.6, 4: 8.0, 5: 6.0, 6: 9.0}
    fc_r2_t = ratio_board(win_r2_t)
    win_r2_f = {1: 5.4, 2: 3.0, 3: 2.7, 4: 8.2, 5: 11.0, 6: 21.0}
    fc_r2_f = forecast_board(h6, 2.5)
    for rows in (
        capture(LOG_DAY, "TOK", 1, at(LOG_DAY, 0, 50), {**win_a, 6: 30.0},
                forecast_board(h6, 0.7), 400_000),
        capture(LOG_DAY, "TOK", 1, at(LOG_DAY, 0, 57), win_a, fc_a, 700_000, scr=(6,)),
        # snapped before t (00:58) but fetched after it: not available at t
        capture(LOG_DAY, "TOK", 1, at(LOG_DAY, 0, 57, 40), win_b, forecast_board(h5, 0.5),
                750_000, fetched=at(LOG_DAY, 0, 58, 20), scr=(6,)),
        capture(LOG_DAY, "TOK", 1, at(LOG_DAY, 0, 59), win_c, forecast_board(h5, 0.8),
                900_000, scr=(6,)),
        capture(LOG_DAY, "TOK", 2, at(LOG_DAY, 1, 20), win_r2_t, fc_r2_t, 2_000_000),
        capture(LOG_DAY, "TOK", 1, at(LOG_DAY, 1, 1), win_f, fc_f, 1_000_000, final=True,
                scr=(6,)),
        capture(LOG_DAY, "TOK", 2, at(LOG_DAY, 1, 31), win_r2_f, fc_r2_f, 3_000_000,
                final=True),
    ):  # fmt: skip
        odds_log.append_capture(log, rows)

    hold_dir = data / "odds_snapshots" / "jra" / HOLD_DAY.isoformat()
    hold_dir.mkdir(parents=True)
    (hold_dir / "NAK.jsonl.gz").write_bytes(b"\x1f\x8b\x08 not a log you may open")
    odds_log.write_post_times(hold_dir / "NAK.races.json",
                              {1: datetime(2026, 11, 29, 1, 0, tzinfo=UTC)},
                              datetime(2026, 11, 28, 23, 0, tzinfo=UTC))  # fmt: skip
    write_store(
        data,
        [store_race(LOG_DAY, "TOK", 1, "MAIDEN"), store_race(LOG_DAY, "TOK", 2, "C1"),
         store_race(HOLD_DAY, "NAK", 1, "C1")],
        [
            {"race_id": "jra:2026-11-21:TOK:01", "pool": "WIN", "combination": "2",
             "dividend": 420.0, "is_refund": False},
            {"race_id": "jra:2026-11-21:TOK:01", "pool": "FORECAST", "combination": "2,1",
             "dividend": 1230.0, "is_refund": False},
            {"race_id": "jra:2026-11-21:TOK:02", "pool": "FORECAST", "combination": "3,4",
             "dividend": 5600.0, "is_refund": False},
            {"race_id": "jra:2026-11-29:NAK:01", "pool": "FORECAST", "combination": "1,2",
             "dividend": 990.0, "is_refund": False},
        ],
    )  # fmt: skip
    return {"win_a": win_a, "fc_a": fc_a, "win_f": win_f, "fc_f": fc_f,
            "win_r2_t": win_r2_t, "fc_r2_t": fc_r2_t, "win_r2_f": win_r2_f,
            "fc_r2_f": fc_r2_f}  # fmt: skip


def guard_holdout_reads(monkeypatch) -> None:
    def guarded(name):
        real = getattr(odds_log, name)

        def wrapper(path, *args, **kwargs):
            if HOLD_DAY.isoformat() in str(path):
                raise AssertionError(f"a holdout day's log was opened: {path}")
            return real(path, *args, **kwargs)

        monkeypatch.setattr(odds_log, name, wrapper)

    for name in ("read_rows", "iter_members", "iter_members_with_offsets", "read_post_times"):
        guarded(name)


def test_load_logged_races_reads_the_board_at_t_minus_2_and_the_final_board(data_dir, monkeypatch):
    boards = build_logged_day(data_dir)
    guard_holdout_reads(monkeypatch)
    races = {r["race_id"]: r for r in er.load_logged_races()}
    assert set(races) == {"jra:2026-11-21:TOK:01", "jra:2026-11-21:TOK:02"}

    r1 = races["jra:2026-11-21:TOK:01"]
    assert r1["race_date"] == LOG_DAY and r1["meeting_id"] == "jra:2026-11-21:TOK"
    assert r1["race_no"] == 1 and r1["race_class"] == "MAIDEN"
    assert r1["win_odds_t"] == boards["win_a"]  # 00:57, horse 6 scratched; 00:57:40 not yet in
    assert r1["exacta_odds_t"] == boards["fc_a"]
    assert r1["final_win_odds"] == boards["win_f"]
    assert r1["final_exacta_odds"] == boards["fc_f"]
    assert r1["pool_total"] == 1_000_000
    assert [(tuple(c), float(d), bool(f)) for c, d, f in r1["exacta"]] == [((2, 1), 1230.0, False)]
    want_rate = takeout.race_takeout("jra:2026-11-21:TOK:01", "FORECAST",
                                     race_class_raw="3歳未勝利")  # fmt: skip
    assert r1["rate"] == pytest.approx(want_rate.payout_rate)
    assert r1["boost_may_be_withheld"] == want_rate.boost_may_be_withheld

    r2 = races["jra:2026-11-21:TOK:02"]
    assert r2["win_odds_t"] == boards["win_r2_t"] and r2["exacta_odds_t"] == boards["fc_r2_t"]
    assert r2["final_exacta_odds"] == boards["fc_r2_f"] and r2["pool_total"] == 3_000_000
    assert [tuple(c) for c, _, _ in r2["exacta"]] == [(3, 4)]


def test_load_logged_races_on_an_empty_data_dir_gives_no_races():
    assert er.load_logged_races() == []


def test_logged_races_score_end_to_end(data_dir, monkeypatch, tmp_path):
    boards = build_logged_day(data_dir)
    guard_holdout_reads(monkeypatch)
    races = er.load_logged_races()
    reg = new_registry(tmp_path)
    result = er.score_e2(races, registry=reg, git_sha="abc1234", lambda2=LAMBDA2,
                         data_builds=BUILDS, n_boot=50)  # fmt: skip
    assert result["n_races"] == 2 and result["n_meetings"] == 1
    sel1 = ref_select(boards["win_a"], boards["fc_a"], RATE, LAMBDA2)
    sel2 = ref_select(boards["win_r2_t"], boards["fc_r2_t"], RATE, LAMBDA2)
    assert any(sel1.values()) and any(sel2.values())
    for (cat, k, scope), got in by_member(result).items():
        want = len(sel2[(cat, k)]) + (len(sel1[(cat, k)]) if scope == "all" else 0)
        assert got["n_bets"] == want, (cat, k, scope)
