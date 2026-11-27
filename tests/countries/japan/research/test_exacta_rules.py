"""#131's E2 ratio-cut rules scored on #130-logged boards (ticket #281).

docs/specs/exacta-overload-prereg.md sections 5-7, with the owner's reframe of 2026-11-27: score
the 16 frozen members of ``japan/131-exacta-rule-v3`` (``exacta_study.E2_ID``; the favourite and
the category from the T-2 board) on whatever development meetings are logged, refusing nothing
for small n (the frozen ``min_meetings: 100`` is superseded); zero meetings give an n = 0 result.
Every figure here is synthetic; HORSETRACK_DATA_DIR points at tmp_path and the registry is a tmp
file.

API pinned here (module ``horsetrack.countries.japan.research.exacta_rules``):

- Constants: ``CUTS = (0.6, 0.7, 0.8, 0.9)``, ``RULE_CATEGORIES = ("fav_second", "no_fav")``,
  ``SCOPES = ("all", "headline")``, ``STAKE = 100``, ``MINUTES_BEFORE_POST = 2``,
  ``BLOCK_SIZE = 5``, ``FROZEN_E2_HASH`` (the v3 spec hash), ``MAX_TEST_ID = E2_ID + "/max-test"``.
- ``select_combinations(win_odds, exacta_odds, *, rate, lambda2) -> {(category, cut): frozenset}``
  for the 8 (category, cut) pairs. Both boards are the boards at t. p_i = (1/O_i) / sum_j 1/O_j
  over ``win_odds`` (the Win board at t), p_c the discounted Harville exacta of p at ``lambda2``,
  q_c = ``implied_share(100 * odds_c, rate=rate)`` (NaN at or below 110 yen: never selected), the
  favourite the lowest Win odds at t (ties to the lowest number). c is selected for (X, k) when
  its category relative to that favourite is X and q_c / p_c < k. A combination naming a horse not
  on the Win board, or with odds None, is never selected.
- Own impact with breakage and the minimum (D18; v3's primary_metric "takeout and breakage in").
  The winning combination's money is estimated from the published dividend D with its 10 yen
  rounding added back, W = (1 - t) P / ((D + 5) / 100) (the half-step convention of
  ``implied_share``).
  ``own_impact_raw(dividend, *, pool, n_bets, stake=STAKE, takeout=0.25) -> float`` is the
  unrounded identity 100 (1 - t) (P + n s) / (W + s), equal to
  (D + 5) (1 + ``exacta_impact.own_impact(P, (D + 5) / 100, s, n, t)``).
  ``own_impact_dividend(...)`` (same arguments) is the dividend that pays: the raw value rounded
  DOWN to a 10 yen step, never below 110 yen. JRA's Plus-10 exception: a published dividend of
  exactly 100 yen (110 yen per winning ticket was not funded) leaves the winning share
  unidentified, so it pays the published 100 yen whatever pool, n_bets and stake (never raised
  to 110).
- ``settle(selected, winner, dividend, *, pool, stake=STAKE, takeout=0.25) -> dict`` with
  ``n_bets`` (len(selected)), ``stake`` (stake * n_bets, yen), ``payout`` (yen:
  stake / 100 * own_impact_dividend(...) when the winner is selected, else 0) and
  ``impact_available`` (False exactly when the published dividend is 100 yen).
- ``lower_tertile_by_day(races, *, key) -> {race_id: bool}``: per race_date, the cutoff is the
  day's 1/3 quantile (linear interpolation, numpy's default) of ``race[key]`` over the races of
  that day whose value is finite and positive; a race is True when its value is at or below the
  cutoff. Races without such a value are absent from the result.
- ``race_scopes(win_odds, *, less_popular) -> frozenset``: always "all"; "headline" when the
  favourite's Win odds *at t* are >= 3.0 and ``less_popular`` (v3's frozen scope: the race's
  exacta pool total in the lower tertile of its logged day). The race 1-4 / maiden proxy is not
  the scope; it lives only in ``proxy_agreement``. The frozen cutoff is kept and labelled
  retrospective: the day's tertile uses later races' T-2 pools, unknown at an early race's t.
- ``score_e2(races, *, registry, git_sha, lambda2, data_builds, n_boot=10000, seed=131,
  holdout_day=is_holdout_day) -> dict``. Each race is a mapping with race_id, race_date (date),
  meeting_id, race_no, race_class, rate, boost_may_be_withheld, win_odds_t {horse: odds},
  exacta_odds_t {(first, second): odds}, final_win_odds, final_exacta_odds, n_starters (the
  actual starters: runners not scratched at the off), pool_total (the race's FORECAST pool on its
  final capture, yen; None when the log has no final FORECAST capture or that capture has no
  pool total row), pool_total_t (the latest FORECAST pool total available by t), less_popular
  (bool, set by the loader over the whole logged day) and exacta [(combination, dividend,
  is_refund)] (the paying rows). win_odds_t and exacta_odds_t are {} when nothing was available
  at t.
  A race with ``holdout_day("japan", race_date)`` is dropped before anything else of it is read;
  a race with an empty win_odds_t or exacta_odds_t is excluded as ``no_board_at_t``; one with
  pool_total None as ``no_final_board``; ``exacta_study.exclusion_reason`` with
  ``n_starters=race["n_starters"]`` drops the rest; all counted in ``result["excluded"]``.
  Selection and scope use the boards at t; each member (category, cut, scope) is a standalone
  rule: in each in-scope race it stakes ``STAKE`` on its selected set and is settled with its own
  n at the final pool. Refuses (RuntimeError or ValueError, nothing appended) unless ``registry``
  holds E2_ID frozen at the hash of ``exacta_study.family_specs()[E2_ID]``. Appends one record per
  member (family E2_ID, status "scored", params with category, cut, scope and spec_hash; a metric
  "n_bets" always, "roi" when the member bet, and "retrospective_cutoff" always: value 1.0 for a
  headline member (its cutoff is the full-day T-2 pool tertile, retrospective), 0.0 for "all",
  n the member's n_races; the label lives in the metrics, never in params, components, variant
  or fold_scheme, so family_size stays 16) and, when any race is scored, one SPA + StepM record
  under MAX_TEST_ID (params block_size 5) over the 16 members with meetings as clusters. A record
  already in the registry is not appended again (no error): the same inputs twice add nothing,
  and a run that stopped part-way is completed by appending only the missing records.
  Result: n_races, n_meetings, excluded, impact_unavailable (the scored races whose published
  winning dividend is 100 yen), members (16 dicts: category, cut, scope, retrospective (scope ==
  "headline"), n_bets, n_races, n_meetings, stake, payout, roi, lo, hi, shrinkage), spa (dict
  with pvalue, n_rules, family_size, n_clusters; None with no race), stepm (dict with superior;
  None with no race), bets (one dict per member x race x selected combination: category, cut,
  scope, race_id, meeting_id, combination, odds (the combination's odds on the final board), won,
  stake, payout). A member's shrinkage is fitted on that member's own bet rows alone
  (``core.eval.segment_roi.fit_segment_roi``, exacta odds bands as the fixed effects), so other
  members selecting the same combinations cannot change it: {"status": "fitted", "n_bets": int,
  "band_effects": {band: float}, "p_beats_takeout": float in [0, 1], ...}; {"status":
  "not_run", "reason": str} when the member has no bet; {"status": "failed", "reason": str} when
  its fit raises.
- ``d_descriptive(races, *, lambda2) -> {category: {"n_races", "mean", "median"}}``: D, the
  full-board ratio sum_S q / sum_S p of ``exacta_consistency.race_ratios`` on the final boards
  (p from final_win_odds as above, dividends 100 * final_exacta_odds), over the three CATEGORIES;
  every race with non-empty final_win_odds and final_exacta_odds counts, whatever its pool_total
  and its boards at t; a race with no final FORECAST prices is left out.
- ``proxy_agreement(races) -> dict`` with n_races, n_days, both_less, proxy_only, tertile_only,
  neither, agreement: "less popular" by the lower tertile of the final pool_total within each
  race *day* (all venues of the day together) against the frozen proxy (race 1-4 or DEBUT/MAIDEN).
  The command passes every loaded logged race, not only the E2-scored ones: a race without a
  board at t still counts when it has a final pool total.
- ``load_logged_races(*, minutes_before=MINUTES_BEFORE_POST) -> list[dict]``: the race mappings
  above, from ``$HORSETRACK_DATA_DIR/odds_snapshots/jra/<day>/<VENUE>.jsonl.gz`` with its
  ``<VENUE>.races.json`` post times (t = post - minutes_before; a row is available at its
  fetched_utc, else its snapshot_utc, docs/specs/board-at-t.md), and the store's
  ``processed/jra/races.parquet``, ``runners.parquet`` and ``dividends.parquet`` (pool FORECAST).
  less_popular is ``lower_tertile_by_day`` on pool_total_t over every logged race of the day that
  has one, whatever its E2 eligibility. A race whose log has no final FORECAST capture is loaded
  with pool_total None and final_exacta_odds {} (never a pre-final total); one whose final
  FORECAST capture has prices but no pool total row is loaded with pool_total None and those
  final prices; one with FORECAST rows but nothing available at t is loaded with win_odds_t {}
  and exacta_odds_t {} and its final boards. Development days only: a holdout day's folder is
  never opened (D14). rate is
  ``takeout.race_takeout(race_id, "FORECAST", race_class_raw=...).payout_rate``.
- ``exacta_rules_command()`` (``horsetrack japan exacta-rules``): each member record's
  data_builds has exactly the keys odds_log, jra_races, jra_runners and jra_dividends, for a
  zero-race run and a populated run alike; the store fingerprints cover only the rows the command
  consumes (the logged development races' rows), so store rows of other races (holdout or
  unlogged days) change no build id and a rerun appends nothing, while a changed consumed row
  changes its table's id. The JSON report under ``reports_dir()`` carries ``d`` and ``proxy``;
  the Markdown report renders the max test's status (a failed max test's reason), D and the proxy
  agreement beside the member table, and labels each headline row retrospective.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party.
from __future__ import annotations  # noqa: I001

import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horsetrack.core import env, paths
from horsetrack.core.eval import registry as registry_module
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import Registry, default_registry
from horsetrack.countries.japan import odds_log, takeout
from horsetrack.countries.japan.research import exacta_impact, exacta_run
from horsetrack.countries.japan.research.exacta_study import E2_ID, family_specs
from horsetrack.countries.japan.research import exacta_rules as er

UTC = timezone.utc
CUTS = (0.6, 0.7, 0.8, 0.9)
RULE_CATS = ("fav_second", "no_fav")
ALL_CATS = ("fav_first", "fav_second", "no_fav")
SCOPES = ("all", "headline")
RATE = 0.75
LAMBDA2 = 0.8104
FROZEN = "9104e173371668a34b4594b6e4970546649b886cdccc4e726ec9749188701f0d"  # rule-v3
BUILDS = {"odds_log": "odds_snapshots/jra/FORECAST@aaaa"}


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


def new_registry(tmp_path: Path, *, freeze: bool = True) -> Registry:
    tmp_path.mkdir(parents=True, exist_ok=True)
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


def ref_raw(dividend, pool, n, stake=100, t=0.25) -> float:
    """100 (1 - t) (P + n s) / (W + s), W = (1 - t) P / ((D + 5) / 100): D's breakage added back."""
    w = (1 - t) * pool / ((dividend + 5) / 100)
    return 100 * (1 - t) * (pool + n * stake) / (w + stake)


def ref_paid(dividend, pool, n, stake=100, t=0.25) -> float:
    """The raw dividend rounded down to 10 yen, never below the 110 yen minimum; a published 100
    yen (Plus-10 not funded: the winning share is unidentified) pays 100 yen as published."""
    if dividend == 100:
        return 100.0
    return max(110.0, 10.0 * math.floor(ref_raw(dividend, pool, n, stake, t) / 10))


def ref_cutoff(values: list[float]) -> float:
    """The 1/3 quantile by linear interpolation between order statistics."""
    s = sorted(values)
    pos = (len(s) - 1) / 3
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (pos - lo) * (s[hi] - s[lo])


def ref_lower_tertile(races, key) -> dict[str, bool]:
    by_day: dict[date, list] = {}
    for r in races:
        v = r.get(key)
        if v is not None and math.isfinite(v) and v > 0:
            by_day.setdefault(r["race_date"], []).append(r)
    out = {}
    for rs in by_day.values():
        cut = ref_cutoff([r[key] for r in rs])
        out.update({r["race_id"]: r[key] <= cut for r in rs})
    return out


def ref_headline(win_odds, less_popular: bool) -> bool:
    return win_odds[ref_fav(win_odds)] >= 3.0 and less_popular


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
    """Three races per meeting (race 1 MAIDEN, race 5 C1, race 9 DEBUT or OP); less_popular is
    the lower tertile of pool_total_t over each day's races (both venues)."""
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
                        "n_starters": len(final_win),
                        "pool_total": float(rng.choice([3e5, 8e5, 2.5e6, 1.2e7])),
                        "pool_total_t": float(rng.uniform(1e5, 4e6)),
                        "exacta": [(winner, dividend, False)],
                    }
                )
    flags = ref_lower_tertile(races, "pool_total_t")
    for race in races:
        race["less_popular"] = flags[race["race_id"]]
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
                    if scope == "headline" and not ref_headline(win, race["less_popular"]):
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
                        payout += ref_paid(dividend, race["pool_total"], len(sel),
                                           t=1 - race["rate"])  # fmt: skip
                out[(cat, k, scope)] = {
                    "n_bets": n_bets, "n_races": n_races, "n_meetings": len(meetings),
                    "stake": stake, "payout": payout,
                }  # fmt: skip
    return out


def expected_bets(races) -> Counter:
    rows = Counter()
    for race in races:
        win = race["win_odds_t"]
        chosen = ref_select(win, race["exacta_odds_t"], race["rate"], LAMBDA2)
        ((winner, dividend, _),) = race["exacta"]
        for cat in RULE_CATS:
            for k in CUTS:
                for scope in SCOPES:
                    if scope == "headline" and not ref_headline(win, race["less_popular"]):
                        continue
                    sel = chosen[(cat, k)]
                    for combo in sel:
                        won = combo == winner
                        pay = ref_paid(dividend, race["pool_total"], len(sel),
                                       t=1 - race["rate"]) if won else 0.0  # fmt: skip
                        odds = race["final_exacta_odds"].get(combo)
                        rows[(cat, k, scope, race["race_id"], race["meeting_id"], combo,
                              None if odds is None else round(float(odds), 9), won, 100.0,
                              round(pay, 6))] += 1  # fmt: skip
    return rows


def got_bets(result) -> Counter:
    rows = Counter()
    for b in result["bets"]:
        odds = b["odds"]
        rows[(b["category"], b["cut"], b["scope"], b["race_id"], b["meeting_id"],
              tuple(b["combination"]), None if odds is None else round(float(odds), 9),
              bool(b["won"]), float(b["stake"]), round(float(b["payout"]), 6))] += 1  # fmt: skip
    return rows


DAYS = [date(2026, 11, 14), date(2026, 11, 15), date(2026, 11, 21)]
assert not any(is_holdout_day("japan", d) for d in DAYS)


def record_key(record) -> str:
    return json.dumps(asdict(record), sort_keys=True, default=str)


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
# 2. Own impact, breakage and settlement
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
    (990.0, 50_000_000, 1, 100, 0.25),  # raw just under D + 5: breakage takes it back to D
    (110.0, 100_000, 1, 20_000, 0.25),  # raw below 110 yen: the minimum binds
]


def clear_of_a_step(value: float) -> bool:
    return abs(value / 10 - round(value / 10)) > 1e-6


def test_the_impact_cases_cover_breakage_and_the_minimum():
    raws = [ref_raw(d, p, n, s, t) for d, p, n, s, t in IMPACT_CASES]
    assert all(clear_of_a_step(r) for r in raws)  # the floor is not decided by float noise
    assert any(r < 110 for r in raws)  # the minimum binds somewhere
    assert any(ref_paid(*c) == c[0] for c in IMPACT_CASES)  # breakage undoes a small impact
    assert any(r > c[0] + 5 for r, c in zip(raws, IMPACT_CASES, strict=True))  # a rise


@pytest.mark.parametrize("dividend, pool, n, stake, t", IMPACT_CASES)
def test_own_impact_raw_is_the_payout_identity_with_the_breakage_added_back(
    dividend, pool, n, stake, t
):
    got = er.own_impact_raw(dividend, pool=pool, n_bets=n, stake=stake, takeout=t)
    assert got == pytest.approx(ref_raw(dividend, pool, n, stake, t), rel=1e-9)
    # #283's own_impact at the odds the unrounded dividend stood at, (D + 5) / 100
    via_283 = (dividend + 5) * (
        1 + exacta_impact.own_impact(pool, (dividend + 5) / 100, stake, n, t)
    )
    assert got == pytest.approx(via_283, rel=1e-9)


@pytest.mark.parametrize("dividend, pool, n, stake, t", IMPACT_CASES)
def test_own_impact_dividend_is_rounded_down_to_ten_yen_with_a_110_yen_minimum(
    dividend, pool, n, stake, t
):
    got = er.own_impact_dividend(dividend, pool=pool, n_bets=n, stake=stake, takeout=t)
    assert got == ref_paid(dividend, pool, n, stake, t)
    assert got >= 110 and got % 10 == 0


def test_a_vanishing_stake_reproduces_the_published_dividend():
    for dividend in (120.0, 1230.0, 25_670.0, 154_000.0):
        raw = er.own_impact_raw(dividend, pool=3_000_000, n_bets=1, stake=1e-9)
        assert raw == pytest.approx(dividend + 5, rel=1e-9)
        paid = er.own_impact_dividend(dividend, pool=3_000_000, n_bets=1, stake=1e-9)
        assert paid == dividend


def test_one_bet_always_lowers_the_dividend_and_more_so_in_a_small_pool():
    for dividend in (130.0, 990.0, 12_340.0, 154_000.0):
        small = er.own_impact_raw(dividend, pool=200_000, n_bets=1)
        large = er.own_impact_raw(dividend, pool=50_000_000, n_bets=1)
        assert small < large < dividend + 5
        paid_small = er.own_impact_dividend(dividend, pool=200_000, n_bets=1)
        paid_large = er.own_impact_dividend(dividend, pool=50_000_000, n_bets=1)
        assert paid_small <= paid_large <= dividend


@pytest.mark.parametrize("dividend, pool, n, stake, t", IMPACT_CASES)
def test_settle_pays_the_broken_impacted_dividend_only_when_the_winner_is_selected(
    dividend, pool, n, stake, t
):
    winner = (3, 5)
    others = [(a, b) for a in range(1, 9) for b in range(1, 9) if a != b and (a, b) != winner]
    hit = {winner, *others[: n - 1]}
    miss = set(others[:n])
    got = er.settle(hit, winner, dividend, pool=pool, stake=stake, takeout=t)
    assert got["n_bets"] == n and got["stake"] == pytest.approx(stake * n)
    expected = stake / 100 * ref_paid(dividend, pool, n, stake, t)
    assert got["payout"] == pytest.approx(expected, rel=1e-12)
    lost = er.settle(miss, winner, dividend, pool=pool, stake=stake, takeout=t)
    assert lost["n_bets"] == n and lost["stake"] == pytest.approx(stake * n)
    assert lost["payout"] == 0


def test_settle_with_nothing_selected_costs_and_pays_nothing():
    got = er.settle(frozenset(), (1, 2), 1230.0, pool=1_000_000)
    assert (got["n_bets"], got["stake"], got["payout"]) == (0, 0, 0)


# (pool, n, stake, takeout): the raw identity would land at, above and below 110 yen
PLUS10_CASES = [(300_000, 1, 100, 0.25), (5_000_000, 12, 100, 0.25), (80_000, 3, 1000, 0.20),
                (1_000_000_000, 40, 100, 0.25)]  # fmt: skip


@pytest.mark.parametrize("pool, n, stake, t", PLUS10_CASES)
def test_a_published_100_yen_dividend_pays_100_yen_and_flags_the_impact_unavailable(
    pool, n, stake, t
):
    """JRA's Plus-10 exception: 100 yen published means 110 yen per winning ticket was not funded,
    so the winning share is unidentified and settlement pays the published 100 yen."""
    paid = er.own_impact_dividend(100.0, pool=pool, n_bets=n, stake=stake, takeout=t)
    assert paid == 100.0
    winner = (2, 4)
    others = [(a, b) for a in range(1, 9) for b in range(1, 9) if a != b and (a, b) != winner]
    hit = er.settle({winner, *others[: n - 1]}, winner, 100.0, pool=pool, stake=stake, takeout=t)
    assert hit["n_bets"] == n and hit["stake"] == pytest.approx(stake * n)
    assert hit["payout"] == pytest.approx(stake)  # 100 yen per 100 staked, never 110
    assert hit["impact_available"] is False
    miss = er.settle(set(others[:n]), winner, 100.0, pool=pool, stake=stake, takeout=t)
    assert miss["payout"] == 0 and miss["impact_available"] is False


@pytest.mark.parametrize("dividend, pool, n, stake, t", IMPACT_CASES)
def test_any_other_published_dividend_keeps_the_floor_and_minimum_and_its_impact(
    dividend, pool, n, stake, t
):
    winner = (3, 5)
    got = er.settle({winner}, winner, dividend, pool=pool, stake=stake, takeout=t)
    assert got["impact_available"] is True
    assert got["payout"] == pytest.approx(stake / 100 * ref_paid(dividend, pool, 1, stake, t))
    assert ref_paid(dividend, pool, 1, stake, t) >= 110


def test_settle_defaults_to_a_hundred_yen_and_the_exacta_takeout():
    got = er.settle({(2, 1), (3, 1)}, (2, 1), 5_000.0, pool=600_000)
    assert er.STAKE == 100
    assert got["stake"] == 200
    assert got["payout"] == ref_paid(5_000.0, 600_000, 2)
    assert got["payout"] < 5_000.0


# ---------------------------------------------------------------------------------------------
# 3. Scope: the headline cell at t, "less popular" by the day's pool tertile at T-2
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fav_odds", [1.8, 2.9, 3.0, 4.4, 7.0])
@pytest.mark.parametrize("less_popular", [True, False])
def test_race_scopes_follow_the_favourite_at_t_and_the_day_tertile(fav_odds, less_popular):
    win = {1: 12.0, 2: fav_odds, 3: 15.0, 4: 20.0, 5: 25.0, 6: 30.0}
    got = er.race_scopes(win, less_popular=less_popular)
    assert set(got) == ({"all", "headline"} if ref_headline(win, less_popular) else {"all"})


def t_race(day: date, venue: str, no: int, value) -> dict:
    return {"race_id": f"jra:{day.isoformat()}:{venue}:{no:02d}", "race_date": day,
            "pool_total_t": value}  # fmt: skip


def test_lower_tertile_by_day_on_a_hand_built_table():
    d1, d2, d3, d4 = DAYS[0], DAYS[1], DAYS[2], date(2026, 11, 22)
    races = [
        # day 1: 5, 1, 3 -> cutoff 1 + (2/3) * 2 = 2.33: only the 1
        t_race(d1, "TOK", 1, 5e5), t_race(d1, "TOK", 2, 1e5), t_race(d1, "KYO", 1, 3e5),
        # day 2: one race with a total, two without -> it is its own cutoff
        t_race(d2, "TOK", 1, 7e5), t_race(d2, "TOK", 2, None), t_race(d2, "TOK", 3, math.nan),
        # day 3: ties at the cutoff are in
        t_race(d3, "HSN", 1, 2e5), t_race(d3, "HSN", 2, 2e5), t_race(d3, "HSN", 3, 2e5),
        t_race(d3, "HSN", 4, 9e5),
        # day 4: a day whose pools are all larger than day 1's still has its own lower third
        *[t_race(d4, "NAK", no, (no + 10) * 1e6) for no in range(1, 7)],
    ]  # fmt: skip
    got = er.lower_tertile_by_day(races, key="pool_total_t")
    assert got == {
        "jra:2026-11-14:TOK:01": False, "jra:2026-11-14:TOK:02": True,
        "jra:2026-11-14:KYO:01": False,
        "jra:2026-11-15:TOK:01": True,
        "jra:2026-11-21:HSN:01": True, "jra:2026-11-21:HSN:02": True,
        "jra:2026-11-21:HSN:03": True, "jra:2026-11-21:HSN:04": False,
        **{f"jra:2026-11-22:NAK:{no:02d}": no <= 2 for no in range(1, 7)},
    }  # fmt: skip


@pytest.mark.parametrize("seed", range(4))
def test_lower_tertile_by_day_matches_the_quantile_on_varied_days(seed):
    rng = np.random.default_rng(900 + seed)
    races = []
    for k in range(4):
        day = date(2026, 10, 3 + k)
        for no in range(1, int(rng.integers(2, 25))):
            races.append(dict(t_race(day, "TOK", no, float(rng.lognormal(13, 1))),
                              pool_total=float(rng.lognormal(13, 1))))  # fmt: skip
    got_t = er.lower_tertile_by_day(races, key="pool_total_t")
    got_f = er.lower_tertile_by_day(races, key="pool_total")
    assert got_t == ref_lower_tertile(races, "pool_total_t")
    assert got_f == ref_lower_tertile(races, "pool_total")
    assert got_t != got_f  # the key is read


# ---------------------------------------------------------------------------------------------
# 4. Scoring end to end
# ---------------------------------------------------------------------------------------------


def by_member(result) -> dict[tuple, dict]:
    members = {(m["category"], m["cut"], m["scope"]): m for m in result["members"]}
    assert len(members) == len(result["members"]) == 16
    return members


def test_the_frozen_e2_hash_is_the_v3_spec():
    assert E2_ID == "japan/131-exacta-rule-v3"
    spec = family_specs()[E2_ID]
    digest = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert er.FROZEN_E2_HASH == FROZEN == digest
    assert er.MAX_TEST_ID == f"{E2_ID}/max-test"
    assert er.BLOCK_SIZE == 5 and er.MINUTES_BEFORE_POST == 2
    assert tuple(er.CUTS) == CUTS and tuple(er.RULE_CATEGORIES) == RULE_CATS
    assert tuple(er.SCOPES) == SCOPES


def test_the_fixture_separates_the_tertile_scope_from_the_proxy():
    races = make_races(281, DAYS)
    proxy = [r["race_no"] <= 4 or r["race_class"] in ("DEBUT", "MAIDEN") for r in races]
    assert any(p != r["less_popular"] for p, r in zip(proxy, races, strict=True))


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


def test_score_e2_keeps_every_selection_and_fits_the_odds_band_shrinkage(tmp_path):
    races = make_races(281, DAYS)
    reg = new_registry(tmp_path)
    result = er.score_e2(races, registry=reg, git_sha="abc1234", lambda2=LAMBDA2,
                         data_builds=BUILDS, n_boot=50)  # fmt: skip
    want = expected_bets(races)
    assert got_bets(result) == want
    members = by_member(result)
    for (cat, k, scope), m in members.items():
        rows = [b for b in result["bets"]
                if (b["category"], b["cut"], b["scope"]) == (cat, k, scope)]  # fmt: skip
        assert len(rows) == m["n_bets"]
        assert sum(b["payout"] for b in rows) == pytest.approx(m["payout"], rel=1e-9)
        shrink = m["shrinkage"]
        assert shrink["status"] in ("fitted", "failed"), (cat, k, scope)
        if shrink["status"] == "fitted":  # fitted on this member's own rows only
            assert shrink["n_bets"] == sum(b["odds"] is not None for b in rows)
            assert 0.0 <= shrink["p_beats_takeout"] <= 1.0
    widest = members[("no_fav", 0.9, "all")]["shrinkage"]
    assert widest["status"] == "fitted"
    effects = widest["band_effects"]
    assert len(effects) >= 2 and all(math.isfinite(float(v)) for v in effects.values())


def cheapen_the_selected_no_fav(races: list[dict]) -> list[dict]:
    """The same races with every no_fav combination the 0.9 cut buys repriced at t to ratio 0.55,
    so the 0.6-0.8 cuts buy exactly the 0.9 cut's combinations; nothing else changes."""
    out = []
    for race in races:
        win, exacta = race["win_odds_t"], dict(race["exacta_odds_t"])
        model = ref_harville(ref_p(win), LAMBDA2)
        for combo in ref_select(win, exacta, race["rate"], LAMBDA2)[("no_fav", 0.9)]:
            exacta[combo] = odds_for_ratio(model[combo], 0.55, race["rate"])
        out.append(dict(race, exacta_odds_t=exacta))
    return out


def test_a_members_shrinkage_is_unchanged_when_other_members_select_the_same_bets(tmp_path):
    """Each member is fitted on its own rows: a nested member duplicating its bets adds no
    evidence to it (one winning ticket must not count once per member that holds it)."""
    races = make_races(281, DAYS)
    dup = cheapen_the_selected_no_fav(races)
    kw = {"git_sha": "abc1234", "lambda2": LAMBDA2, "data_builds": BUILDS, "n_boot": 50}
    base = by_member(er.score_e2(races, registry=new_registry(tmp_path / "a"), **kw))
    more = by_member(er.score_e2(dup, registry=new_registry(tmp_path / "b"), **kw))
    for scope in SCOPES:
        assert more[("no_fav", 0.6, scope)]["n_bets"] == more[("no_fav", 0.9, scope)]["n_bets"]
        assert base[("no_fav", 0.6, scope)]["n_bets"] < base[("no_fav", 0.9, scope)]["n_bets"]
        for key in (("no_fav", 0.9, scope), *(("fav_second", k, scope) for k in CUTS)):
            assert more[key]["n_bets"] == base[key]["n_bets"], key
            assert more[key]["shrinkage"]["status"] == base[key]["shrinkage"]["status"], key
            if base[key]["shrinkage"]["status"] == "fitted":
                assert more[key]["shrinkage"]["n_bets"] == base[key]["shrinkage"]["n_bets"]
                assert more[key]["shrinkage"]["p_beats_takeout"] == pytest.approx(
                    base[key]["shrinkage"]["p_beats_takeout"], abs=1e-9), key  # fmt: skip
                assert more[key]["shrinkage"]["band_effects"] == pytest.approx(
                    base[key]["shrinkage"]["band_effects"], abs=1e-9), key  # fmt: skip
    assert base[("no_fav", 0.9, "all")]["shrinkage"]["status"] == "fitted"


def test_shrinkage_is_not_run_without_bets_and_a_failed_fit_is_reported(tmp_path):
    empty = er.score_e2([], registry=new_registry(tmp_path / "a"), git_sha="abc1234",
                        lambda2=LAMBDA2, data_builds=BUILDS, n_boot=50)  # fmt: skip
    assert empty["bets"] == []
    for m in by_member(empty).values():
        assert m["shrinkage"]["status"] == "not_run" and m["shrinkage"]["reason"]

    # one race, one cheap combination at t (fav_second) and it wins: each fav_second member
    # fits one all-won row. The shrinkage prices it on the final board, here 1.5: an all-won
    # bet at odds this short has no interior mode, so segment_roi's fit raises even on n = 1
    # (at the t price, ~18, a single row fits; only duplicated rows would fail there)
    race = make_races(281, DAYS[:1])[0]
    win = {1: 2.4, 2: 4.0, 3: 5.0, 4: 7.0, 5: 11.0, 6: 16.0}
    model = ref_harville(ref_p(win), LAMBDA2)
    exacta = {c: odds_for_ratio(model[c], 0.55 if c == (3, 1) else 1.2, RATE) for c in model}
    race.update(win_odds_t=win, exacta_odds_t=exacta, final_win_odds=win, n_starters=6,
                final_exacta_odds={**exacta, (3, 1): 1.5}, less_popular=False,
                exacta=[((3, 1), 4_560.0, False)])  # fmt: skip
    result = er.score_e2([race], registry=new_registry(tmp_path / "b"), git_sha="abc1234",
                         lambda2=LAMBDA2, data_builds=BUILDS, n_boot=50)  # fmt: skip
    assert result["bets"] and all(b["won"] for b in result["bets"])
    assert {b["combination"] for b in result["bets"]} == {(3, 1)}
    for (cat, _, scope), m in by_member(result).items():
        if cat == "fav_second" and scope == "all":  # every one of its bets won
            assert m["shrinkage"]["status"] == "failed" and m["shrinkage"]["reason"]
        else:  # no bet (no_fav buys nothing; the favourite at 2.4 is not headline)
            assert m["shrinkage"]["status"] == "not_run" and m["shrinkage"]["reason"]


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
        retro = metrics["retrospective_cutoff"]
        assert retro.value == (1.0 if key[2] == "headline" else 0.0), key
        assert retro.n == expected[key]["n_races"]
    assert seen == set(expected)
    assert not reg.records("japan/131-exacta-rule-v2")
    assert reg.family_size(E2_ID) == 16
    for m in result["members"]:
        assert m["retrospective"] is (m["scope"] == "headline")

    (max_test,) = reg.records(f"{E2_ID}/max-test")
    assert max_test.params["block_size"] == 5
    assert result["spa"]["n_rules"] == 16 and result["spa"]["family_size"] == 16
    assert result["spa"]["n_clusters"] == 6
    assert 0.0 <= result["spa"]["pvalue"] <= 1.0
    assert set(result["stepm"]["superior"]) <= {r.variant for r in records}
    reg.verify()


def test_the_retrospective_label_is_not_a_new_variant_of_the_records_already_made(tmp_path):
    """Records written before the label (the same params, no retrospective_cutoff metric) and a
    labelled run are one family of 16: the label changes no params, components or variant."""
    races = make_races(281, DAYS)
    kw = {"git_sha": "abc1234", "lambda2": LAMBDA2, "data_builds": BUILDS, "n_boot": 50}
    labelled = new_registry(tmp_path / "new")
    er.score_e2(races, registry=labelled, **kw)
    older = new_registry(tmp_path / "old")
    for r in labelled.records(E2_ID):
        kept = tuple(m for m in r.metrics if m.metric != "retrospective_cutoff")
        assert len(kept) == len(r.metrics) - 1
        older.append(registry_module.RunRecord(**{**asdict(r), "metrics": kept,
                                                  "git_sha": "def5678"}))  # fmt: skip
    er.score_e2(races, registry=older, **kw)
    assert len(older.records(E2_ID)) == 32
    assert older.family_size(E2_ID) == 16


def test_scoring_the_same_inputs_twice_adds_nothing(tmp_path):
    races = make_races(281, DAYS)
    reg = new_registry(tmp_path)
    kw = {"registry": reg, "git_sha": "abc1234", "lambda2": LAMBDA2, "data_builds": BUILDS,
          "n_boot": 50}  # fmt: skip
    first = er.score_e2(races, **kw)
    before = [record_key(r) for r in reg.records()]
    assert len(before) == 17
    second = er.score_e2(races, **kw)
    assert [record_key(r) for r in reg.records()] == before
    assert by_member(second) == by_member(first)
    reg.verify()


@pytest.mark.parametrize("done", [1, 7, 16])
def test_a_run_that_stopped_part_way_is_completed_on_retry(tmp_path, done):
    races = make_races(281, DAYS)
    kw = {"git_sha": "abc1234", "lambda2": LAMBDA2, "data_builds": BUILDS, "n_boot": 50}
    full = new_registry(tmp_path / "full")
    er.score_e2(races, registry=full, **kw)
    partial = new_registry(tmp_path / "partial")
    for record in full.records(E2_ID)[:done]:  # the members recorded before the stop
        partial.append(record)
    er.score_e2(races, registry=partial, **kw)
    assert Counter(record_key(r) for r in partial.records()) == Counter(
        record_key(r) for r in full.records()
    )
    assert len(partial.records(E2_ID)) == 16 and len(partial.records(er.MAX_TEST_ID)) == 1
    assert partial.family_size(E2_ID) == 16
    partial.verify()


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
    want = expected_members(races)  # less_popular as the loader set it over the whole day
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
        retro = {m.metric: m.value for m in r.metrics}["retrospective_cutoff"]
        assert retro == (1.0 if r.params["scope"] == "headline" else 0.0)
    assert result["impact_unavailable"] == 0


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


@pytest.mark.parametrize("state", ["not frozen", "another spec", "only v2 frozen"])
def test_scoring_refuses_unless_e2_is_frozen_at_its_hash(tmp_path, state):
    reg = new_registry(tmp_path, freeze=False)
    if state == "another spec":
        spec = family_specs()[E2_ID]
        spec["cuts"] = [0.5, 0.6, 0.7, 0.8]
        reg.freeze_family(E2_ID, spec)
    if state == "only v2 frozen":
        spec = family_specs()[E2_ID]
        spec["favourite"] = "lowest final Win odds, ties to the lowest number"
        reg.freeze_family("japan/131-exacta-rule-v2", spec)
    with pytest.raises((RuntimeError, ValueError)):
        er.score_e2(make_races(281, DAYS[:1]), registry=reg, git_sha="abc1234",
                    lambda2=LAMBDA2, data_builds=BUILDS, n_boot=50)  # fmt: skip
    assert reg.records() == []


def test_scoring_selects_on_the_board_at_t_not_the_final_board(tmp_path):
    """A less popular race: the favourite is 2.4 at t (not headline) and 3.4 at the off
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
    race["n_starters"] = 7
    race["less_popular"] = True
    race["final_exacta_odds"] = {c: odds_for_ratio(model_f[c], float(rng.choice(RATIO_POOL)),
                                                   RATE) for c in model_f}  # fmt: skip
    sel_t = ref_select(win_t, race["exacta_odds_t"], RATE, LAMBDA2)
    sel_f = ref_select(win_final, race["final_exacta_odds"], RATE, LAMBDA2)
    assert sel_t != sel_f
    assert not ref_headline(win_t, True) and ref_headline(win_final, True)
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
    assert members[("no_fav", 0.9, "all")]["payout"] == ref_paid(4_560.0, race["pool_total"], n)


def test_the_headline_scope_is_the_tertile_flag_not_the_race_number_proxy(tmp_path):
    """Two races with the favourite at 3.5 at t: race 1 of a maiden card (the old proxy) outside
    the day's lower tertile, and race 11 of an open card inside it."""
    base = make_races(281, DAYS[:1])[0]
    win = {1: 3.5, 2: 4.0, 3: 4.5, 4: 6.0, 5: 9.0, 6: 14.0}
    model = ref_harville(ref_p(win), LAMBDA2)
    exacta = {c: odds_for_ratio(model[c], 0.66 if (c[0] + c[1]) % 2 else 1.2, RATE)
              for c in model}  # fmt: skip
    common = {
        "win_odds_t": win,
        "exacta_odds_t": exacta,
        "final_win_odds": win,
        "final_exacta_odds": exacta,
        "n_starters": 6,
        "exacta": [((2, 3), 3_210.0, False)],
    }
    proxy_race = dict(base, **common, race_id="jra:2026-11-14:TOK:01", race_no=1,
                      race_class="MAIDEN", less_popular=False)  # fmt: skip
    tertile_race = dict(base, **common, race_id="jra:2026-11-14:TOK:11", race_no=11,
                        race_class="OP", less_popular=True)  # fmt: skip
    sel = ref_select(win, exacta, RATE, LAMBDA2)
    assert any(sel.values())
    result = er.score_e2([proxy_race, tertile_race], registry=new_registry(tmp_path),
                         git_sha="abc1234", lambda2=LAMBDA2, data_builds=BUILDS,
                         n_boot=50)  # fmt: skip
    members = by_member(result)
    for (cat, k, scope), got in members.items():
        want = len(sel[(cat, k)]) * (1 if scope == "headline" else 2)
        assert got["n_bets"] == want, (cat, k, scope)
    headline_races = {b["race_id"] for b in result["bets"] if b["scope"] == "headline"}
    assert headline_races == {"jra:2026-11-14:TOK:11"}


def test_excluded_races_place_no_bets_and_are_counted(tmp_path):
    races = make_races(281, DAYS[:1])
    keep = races[:2]
    base = races[2]
    boost = dict(base, race_id="jra:2026-11-14:TOK:10", boost_may_be_withheld=True)
    refund = dict(base, race_id="jra:2026-11-14:TOK:11",
                  exacta=[(base["exacta"][0][0], 100.0, True)])  # fmt: skip
    heat = dict(base, race_id="jra:2026-11-14:TOK:12",
                exacta=[((1, 2), 900.0, False), ((2, 1), 1500.0, False)])  # fmt: skip
    # five or more on the T-2 board, but a late scratch leaves four actual starters
    late = dict(base, race_id="jra:2026-11-14:KYO:12", n_starters=4)
    assert len(late["win_odds_t"]) >= 5
    unfinished = dict(base, race_id="jra:2026-11-14:KYO:11", pool_total=None)
    # loaded for D and the proxy, but nothing (or no exacta price) was available at t
    blind = dict(base, race_id="jra:2026-11-14:KYO:10", win_odds_t={}, exacta_odds_t={})
    no_exacta_t = dict(base, race_id="jra:2026-11-14:KYO:09", exacta_odds_t={})
    reg = new_registry(tmp_path)
    result = er.score_e2([*keep, boost, refund, heat, late, unfinished, blind, no_exacta_t],
                         registry=reg, git_sha="abc1234", lambda2=LAMBDA2, data_builds=BUILDS,
                         n_boot=50)  # fmt: skip
    assert result["n_races"] == 2
    assert {k: v for k, v in result["excluded"].items() if v} == {
        "boost_may_be_withheld": 1, "refund": 1, "dead_heat": 1, "under_5_starters": 1,
        "no_final_board": 1, "no_board_at_t": 2}  # fmt: skip
    want = expected_members(keep)
    for key, got in by_member(result).items():
        assert got["n_bets"] == want[key]["n_bets"]
        assert got["payout"] == pytest.approx(want[key]["payout"], rel=1e-9)


def test_a_published_100_yen_winner_is_settled_at_100_yen_and_counted(tmp_path):
    races = make_races(281, DAYS)
    hit = next(i for i, r in enumerate(races)
               if r["exacta"][0][0] in ref_select(r["win_odds_t"], r["exacta_odds_t"], RATE,
                                                  LAMBDA2)[("no_fav", 0.9)])  # fmt: skip
    winner = races[hit]["exacta"][0][0]
    races[hit] = dict(races[hit], exacta=[(winner, 100.0, False)])
    expected = expected_members(races)  # ref_paid pays 100 yen at a published 100
    result = er.score_e2(races, registry=new_registry(tmp_path), git_sha="abc1234",
                         lambda2=LAMBDA2, data_builds=BUILDS, n_boot=50)  # fmt: skip
    assert result["n_races"] == 18 and result["impact_unavailable"] == 1
    for key, got in by_member(result).items():
        assert got["payout"] == pytest.approx(expected[key]["payout"], rel=1e-9), key
    paid = [b["payout"] for b in result["bets"]
            if b["race_id"] == races[hit]["race_id"] and b["won"]]  # fmt: skip
    assert paid and all(p == pytest.approx(100.0) for p in paid)


# ---------------------------------------------------------------------------------------------
# 5. D descriptive and the proxy check
# ---------------------------------------------------------------------------------------------


def d_race(rid: str, win: dict, ratios: dict[str, float], pool=1e6) -> dict:
    model = ref_harville(ref_p(win), LAMBDA2)
    fav = ref_fav(win)
    exacta = {c: odds_for_ratio(model[c], ratios[ref_cat(c, fav)], RATE) for c in model}
    return {"race_id": rid, "race_date": DAYS[0], "meeting_id": "jra:2026-11-14:TOK",
            "race_no": 1, "race_class": "C1", "rate": RATE, "final_win_odds": win,
            "final_exacta_odds": exacta, "pool_total": pool}  # fmt: skip


def test_d_is_the_full_board_category_ratio_on_the_final_boards():
    races = [
        d_race("a", {1: 2.0, 2: 4.0, 3: 6.5, 4: 9.0, 5: 14.0, 6: 22.0},
               {"fav_first": 1.3, "fav_second": 1.0, "no_fav": 1.0}),
        d_race("b", {1: 8.0, 2: 3.1, 3: 3.6, 4: 6.0, 5: 11.0, 6: 19.0, 7: 40.0},
               {"fav_first": 1.1, "fav_second": 0.9, "no_fav": 1.0}),
        d_race("c", {1: 5.5, 2: 7.0, 3: 1.7, 4: 12.0, 5: 25.0},
               {"fav_first": 2.0, "fav_second": 0.6, "no_fav": 0.8}),
        # final prices but no pool total on the final capture (and no board at t): still D's
        dict(d_race("d", {1: 2.5, 2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0},
                    {"fav_first": 1.6, "fav_second": 0.7, "no_fav": 1.2}, pool=None),
             win_odds_t={}, exacta_odds_t={}),
        # no final FORECAST prices: not D's, whatever its pool total
        dict(d_race("e", {1: 2.5, 2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0},
                    {"fav_first": 3.0, "fav_second": 0.2, "no_fav": 0.3}),
             final_exacta_odds={}),
    ]  # fmt: skip
    got = er.d_descriptive(races, lambda2=LAMBDA2)
    assert set(got) == set(ALL_CATS)
    want = {"fav_first": [1.3, 1.1, 2.0, 1.6], "fav_second": [1.0, 0.9, 0.6, 0.7],
            "no_fav": [1.0, 1.0, 0.8, 1.2]}  # fmt: skip
    for cat, values in want.items():
        assert got[cat]["n_races"] == 4
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
RID = {no: f"jra:2026-11-21:TOK:{no:02d}" for no in range(1, 7)}


def at(day: date, hh: int, mm: int, ss: int = 0) -> str:
    return datetime(day.year, day.month, day.day, hh, mm, ss, tzinfo=UTC).isoformat()


def capture(day, venue, no, snap, win, forecast, pool, *, final=False, fetched=None, scr=()):
    """One capture's rows; forecast None leaves FORECAST (and its pool total) out, pool None
    only the pool total row (the parser emits it only when the page prints a ticket count)."""
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
    if forecast is not None:
        for (a, b), o in forecast.items():
            rows.append({"snapshot_utc": snap, "race_id": rid, "pool": "FORECAST",
                         "combination": f"{a},{b}", "odds": o, "status_raw": None,
                         **extra})  # fmt: skip
        if pool is not None:
            rows.append({"snapshot_utc": snap, "race_id": rid, "pool": "FORECAST",
                         "combination": odds_log.POOL_TOTAL, "odds": None, "status_raw": None,
                         "investment": pool, **extra})  # fmt: skip
    rows.append({"snapshot_utc": snap, "race_id": rid, "pool": "QUINELLA", "combination": "1,2",
                 "odds": 99.0, "status_raw": None, **extra})  # fmt: skip # never exacta
    return rows


def forecast_board(horses, scale: float) -> dict:
    return {(a, b): round(scale * (a + 2 * b), 1) for a in horses for b in horses if a != b}


def ratio_board(win: dict) -> dict:
    """Exacta odds (to 0.1) at ratios 0.55-1.05 to Harville on `win`, so some are selected."""
    model = ref_harville(ref_p(win), LAMBDA2)
    return {(a, b): round(odds_for_ratio(model[(a, b)], 0.55 + 0.1 * ((3 * a + b) % 6), RATE), 1)
            for a, b in model}  # fmt: skip


def write_store(data: Path, races: list[dict], dividends: list[dict], runners=()) -> None:
    base = data / "processed" / "jra"
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(races).to_parquet(base / "races.parquet", index=False)
    pd.DataFrame(dividends).to_parquet(base / "dividends.parquet", index=False)
    if runners:
        pd.DataFrame(list(runners)).to_parquet(base / "runners.parquet", index=False)


def store_race(day, venue, no, race_class, field=6) -> dict:
    return {"race_id": f"jra:{day.isoformat()}:{venue}:{no:02d}", "race_index": no,
            "race_name": f"R{no}", "race_class": race_class, "race_class_raw": "3歳未勝利",
            "meeting_id": f"jra:{day.isoformat()}:{venue}", "race_date": day, "venue": venue,
            "race_no": no, "season": "2026", "field_size": field, "status": "run"}  # fmt: skip


def store_runners(rid: str, horses, scratched=()) -> list[dict]:
    return [{"race_id": rid, "horse_no": h, "horse_id": f"{rid}#{h}",
             "is_scratched": h in scratched} for h in horses]  # fmt: skip


def build_logged_day(data: Path, *, extra_day: bool = False) -> dict:
    """Six Tokyo races on one development day.

    1 MAIDEN: horse 6 scratched before t; a capture snapped before t but fetched after it.
    2 C1 (race 2: the old proxy's "less popular"), favourite 3.2 at t; pool 450k at t.
    3 C1: its log has no final FORECAST capture (the final capture is WIN only).
    4 C1: five runners at t, horse 5 scratched after t: four actual starters.
    5 MAIDEN (the old proxy), favourite 3.4 at t, pool 1.5M at t.
    6 C2, favourite 3.3 at t, the smallest pool at t (250k): the only headline race.
    Pools at t 700k, 450k, 350k, 900k, 1.5M, 250k: lower-tertile cutoff 416.7k (races 3, 6);
    final pools 1.0M, 3.0M, none, 0.5M, 1.1M, 0.9M. ``extra_day`` adds ``build_second_day``.
    """
    day_dir = data / "odds_snapshots" / "jra" / LOG_DAY.isoformat()
    log = day_dir / "TOK.jsonl.gz"
    post = {no: datetime(2026, 11, 21, 1, 0, tzinfo=UTC) + timedelta(minutes=30 * (no - 1))
            for no in range(1, 7)}  # fmt: skip
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
    win_r2_t = {1: 5.0, 2: 3.2, 3: 3.6, 4: 8.0, 5: 6.0, 6: 9.0}
    fc_r2_t = ratio_board(win_r2_t)
    win_r2_f = {1: 5.4, 2: 3.0, 3: 2.7, 4: 8.2, 5: 11.0, 6: 21.0}
    fc_r2_f = forecast_board(h6, 2.5)
    win_r3 = {1: 3.1, 2: 3.5, 3: 4.8, 4: 7.0, 5: 9.0, 6: 15.0}
    win_r4_t = {1: 2.4, 2: 4.0, 3: 5.5, 4: 8.0, 5: 11.0}
    win_r4_f = {1: 2.3, 2: 4.1, 3: 5.0, 4: 7.5}
    win_r5_t = {1: 4.0, 2: 3.4, 3: 5.0, 4: 6.5, 5: 8.0, 6: 9.5}
    win_r5_f = {1: 4.2, 2: 3.3, 3: 5.1, 4: 6.0, 5: 8.5, 6: 9.0}
    win_r6_t = {1: 3.3, 2: 3.8, 3: 4.5, 4: 6.0, 5: 9.0, 6: 14.0}
    win_r6_f = {1: 3.1, 2: 3.9, 3: 4.6, 4: 6.2, 5: 9.5, 6: 13.0}
    boards = {"win_a": win_a, "fc_a": fc_a, "win_f": win_f, "fc_f": fc_f,
              "win_r2_t": win_r2_t, "fc_r2_t": fc_r2_t, "win_r2_f": win_r2_f, "fc_r2_f": fc_r2_f,
              "win_r3": win_r3, "fc_r3": ratio_board(win_r3), "win_r4_t": win_r4_t,
              "fc_r4": ratio_board(win_r4_t), "win_r4_f": win_r4_f,
              "win_r5_t": win_r5_t, "fc_r5": ratio_board(win_r5_t), "win_r5_f": win_r5_f,
              "fc_r5_f": forecast_board(h6, 1.9), "win_r6_t": win_r6_t,
              "fc_r6": ratio_board(win_r6_t), "win_r6_f": win_r6_f,
              "fc_r6_f": forecast_board(h6, 1.6)}  # fmt: skip
    for rows in (
        capture(LOG_DAY, "TOK", 1, at(LOG_DAY, 0, 50), {**win_a, 6: 30.0},
                forecast_board(h6, 0.7), 400_000),
        capture(LOG_DAY, "TOK", 1, at(LOG_DAY, 0, 57), win_a, fc_a, 700_000, scr=(6,)),
        # snapped before t (00:58) but fetched after it: not available at t
        capture(LOG_DAY, "TOK", 1, at(LOG_DAY, 0, 57, 40), win_b, forecast_board(h5, 0.5),
                750_000, fetched=at(LOG_DAY, 0, 58, 20), scr=(6,)),
        capture(LOG_DAY, "TOK", 1, at(LOG_DAY, 0, 59), win_c, forecast_board(h5, 0.8),
                900_000, scr=(6,)),
        capture(LOG_DAY, "TOK", 2, at(LOG_DAY, 1, 20), win_r2_t, fc_r2_t, 450_000),
        capture(LOG_DAY, "TOK", 1, at(LOG_DAY, 1, 1), win_f, fc_f, 1_000_000, final=True,
                scr=(6,)),
        capture(LOG_DAY, "TOK", 2, at(LOG_DAY, 1, 31), win_r2_f, fc_r2_f, 3_000_000,
                final=True),
        capture(LOG_DAY, "TOK", 3, at(LOG_DAY, 1, 50), win_r3, boards["fc_r3"], 350_000),
        capture(LOG_DAY, "TOK", 3, at(LOG_DAY, 1, 59), win_r3, forecast_board(h6, 1.2),
                380_000),  # after t (01:58), still before the off: a pre-final total
        capture(LOG_DAY, "TOK", 3, at(LOG_DAY, 2, 1), win_r3, None, None, final=True),
        capture(LOG_DAY, "TOK", 4, at(LOG_DAY, 2, 20), win_r4_t, boards["fc_r4"], 900_000),
        capture(LOG_DAY, "TOK", 4, at(LOG_DAY, 2, 31), win_r4_f, forecast_board(range(1, 5), 1.3),
                500_000, final=True, scr=(5,)),
        capture(LOG_DAY, "TOK", 5, at(LOG_DAY, 2, 50), win_r5_t, boards["fc_r5"], 1_500_000),
        capture(LOG_DAY, "TOK", 5, at(LOG_DAY, 3, 1), win_r5_f, boards["fc_r5_f"], 1_100_000,
                final=True),
        capture(LOG_DAY, "TOK", 6, at(LOG_DAY, 3, 20), win_r6_t, boards["fc_r6"], 250_000),
        capture(LOG_DAY, "TOK", 6, at(LOG_DAY, 3, 31), win_r6_f, boards["fc_r6_f"], 900_000,
                final=True),
    ):  # fmt: skip
        odds_log.append_capture(log, rows)

    hold_dir = data / "odds_snapshots" / "jra" / HOLD_DAY.isoformat()
    hold_dir.mkdir(parents=True)
    (hold_dir / "NAK.jsonl.gz").write_bytes(b"\x1f\x8b\x08 not a log you may open")
    odds_log.write_post_times(hold_dir / "NAK.races.json",
                              {1: datetime(2026, 11, 29, 1, 0, tzinfo=UTC)},
                              datetime(2026, 11, 28, 23, 0, tzinfo=UTC))  # fmt: skip
    sel5 = ref_select(win_r5_t, boards["fc_r5"], RATE, LAMBDA2)
    winner5 = sorted(sel5[("no_fav", 0.9)])[0]
    sel6 = ref_select(win_r6_t, boards["fc_r6"], RATE, LAMBDA2)
    winner6 = sorted(sel6[("no_fav", 0.9)])[-1]
    boards["winners"] = {
        1: ((2, 1), 1230.0),
        2: ((3, 4), 5600.0),
        3: ((1, 2), 2000.0),
        4: ((1, 3), 800.0),
        5: (winner5, 3450.0),
        6: (winner6, 2770.0),
    }
    dividends = [{"race_id": RID[no], "pool": "FORECAST", "combination": f"{c[0]},{c[1]}",
                  "dividend": d, "is_refund": False}
                 for no, (c, d) in boards["winners"].items()]  # fmt: skip
    store = {
        "races": [store_race(LOG_DAY, "TOK", 1, "MAIDEN"), store_race(LOG_DAY, "TOK", 2, "C1"),
                  store_race(LOG_DAY, "TOK", 3, "C1"), store_race(LOG_DAY, "TOK", 4, "C1", field=5),
                  store_race(LOG_DAY, "TOK", 5, "MAIDEN"), store_race(LOG_DAY, "TOK", 6, "C2"),
                  store_race(HOLD_DAY, "NAK", 1, "C1")],
        "dividends": [{"race_id": RID[1], "pool": "WIN", "combination": "2", "dividend": 420.0,
                       "is_refund": False}, *dividends,
                      {"race_id": "jra:2026-11-29:NAK:01", "pool": "FORECAST",
                       "combination": "1,2", "dividend": 990.0, "is_refund": False}],
        "runners": [*store_runners(RID[1], h6, scratched=(6,)), *store_runners(RID[2], h6),
                    *store_runners(RID[3], h6), *store_runners(RID[4], h5, scratched=(5,)),
                    *store_runners(RID[5], h6), *store_runners(RID[6], h6),
                    *store_runners("jra:2026-11-29:NAK:01", h6)],
    }  # fmt: skip
    if extra_day:
        boards.update(build_second_day(data, store))
    write_store(data, store["races"], store["dividends"], store["runners"])
    return boards


DAY2 = date(2026, 11, 22)
assert not is_holdout_day("japan", DAY2)
RID2 = {no: f"jra:2026-11-22:KYO:{no:02d}" for no in range(1, 4)}


def build_second_day(data: Path, store: dict) -> dict:
    """Kyoto, 2026-11-22, three C1 races of six (t = post - 2 min):

    1 its only capture is the final one (01:01): no board at t; final pool 800k.
    2 boards at t (pool 600k at t); its final FORECAST capture prices every combination but has
      no pool total row.
    3 an ordinary race: pool 300k at t, final pool 2.0M.
    """
    day_dir = data / "odds_snapshots" / "jra" / DAY2.isoformat()
    day_dir.mkdir(parents=True)
    post = {no: datetime(2026, 11, 22, 1, 0, tzinfo=UTC) + timedelta(minutes=30 * (no - 1))
            for no in range(1, 4)}  # fmt: skip
    odds_log.write_post_times(
        day_dir / "KYO.races.json", post, datetime(2026, 11, 21, 23, 0, tzinfo=UTC)
    )
    h6 = range(1, 7)
    b = {
        "k1_win_f": {1: 2.6, 2: 3.9, 3: 5.2, 4: 8.0, 5: 12.0, 6: 18.0},
        "k1_fc_f": forecast_board(h6, 1.1),
        "k2_win_t": {1: 4.1, 2: 2.3, 3: 5.6, 4: 7.7, 5: 10.0, 6: 21.0},
        "k2_win_f": {1: 4.4, 2: 2.2, 3: 5.9, 4: 7.1, 5: 11.0, 6: 19.0},
        "k2_fc_f": forecast_board(h6, 1.7),
        "k3_win_t": {1: 3.6, 2: 3.9, 3: 4.2, 4: 6.5, 5: 9.5, 6: 16.0},
        "k3_win_f": {1: 3.4, 2: 4.1, 3: 4.0, 4: 6.9, 5: 9.0, 6: 17.0},
        "k3_fc_f": forecast_board(h6, 2.2),
    }
    b["k2_fc_t"], b["k3_fc_t"] = ratio_board(b["k2_win_t"]), ratio_board(b["k3_win_t"])
    log = day_dir / "KYO.jsonl.gz"
    for rows in (
        capture(DAY2, "KYO", 1, at(DAY2, 1, 1), b["k1_win_f"], b["k1_fc_f"], 800_000, final=True),
        capture(DAY2, "KYO", 2, at(DAY2, 1, 20), b["k2_win_t"], b["k2_fc_t"], 600_000),
        capture(DAY2, "KYO", 2, at(DAY2, 1, 31), b["k2_win_f"], b["k2_fc_f"], None, final=True),
        capture(DAY2, "KYO", 3, at(DAY2, 1, 50), b["k3_win_t"], b["k3_fc_t"], 300_000),
        capture(DAY2, "KYO", 3, at(DAY2, 2, 1), b["k3_win_f"], b["k3_fc_f"], 2_000_000,
                final=True),
    ):  # fmt: skip
        odds_log.append_capture(log, rows)
    winners = {1: ((1, 2), 550.0), 2: ((2, 3), 1360.0), 3: ((4, 1), 2640.0)}
    for no, (c, d) in winners.items():
        store["races"].append(store_race(DAY2, "KYO", no, "C1"))
        store["dividends"].append({"race_id": RID2[no], "pool": "FORECAST",
                                   "combination": f"{c[0]},{c[1]}", "dividend": d,
                                   "is_refund": False})  # fmt: skip
        store["runners"].extend(store_runners(RID2[no], h6))
    return b


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
    assert set(races) == set(RID.values())

    r1 = races[RID[1]]
    assert r1["race_date"] == LOG_DAY and r1["meeting_id"] == "jra:2026-11-21:TOK"
    assert r1["race_no"] == 1 and r1["race_class"] == "MAIDEN"
    assert r1["win_odds_t"] == boards["win_a"]  # 00:57, horse 6 scratched; 00:57:40 not yet in
    assert r1["exacta_odds_t"] == boards["fc_a"]
    assert r1["final_win_odds"] == boards["win_f"]
    assert r1["final_exacta_odds"] == boards["fc_f"]
    assert r1["pool_total"] == 1_000_000
    assert r1["pool_total_t"] == 700_000  # the 00:57 total; 00:57:40's was fetched after t
    assert r1["n_starters"] == 5
    assert [(tuple(c), float(d), bool(f)) for c, d, f in r1["exacta"]] == [((2, 1), 1230.0, False)]
    want_rate = takeout.race_takeout(RID[1], "FORECAST", race_class_raw="3歳未勝利")
    assert r1["rate"] == pytest.approx(want_rate.payout_rate)
    assert r1["boost_may_be_withheld"] == want_rate.boost_may_be_withheld

    r2 = races[RID[2]]
    assert r2["win_odds_t"] == boards["win_r2_t"] and r2["exacta_odds_t"] == boards["fc_r2_t"]
    assert r2["final_exacta_odds"] == boards["fc_r2_f"] and r2["pool_total"] == 3_000_000
    assert r2["pool_total_t"] == 450_000
    assert [tuple(c) for c, _, _ in r2["exacta"]] == [(3, 4)]


def test_a_race_without_a_final_forecast_capture_keeps_no_pre_final_total(data_dir, monkeypatch):
    boards = build_logged_day(data_dir)
    guard_holdout_reads(monkeypatch)
    r3 = {r["race_id"]: r for r in er.load_logged_races()}[RID[3]]
    assert r3["win_odds_t"] == boards["win_r3"] and r3["exacta_odds_t"] == boards["fc_r3"]
    assert r3["pool_total"] is None  # not the 380k of the last pre-final capture
    assert not r3["final_exacta_odds"]
    assert r3["pool_total_t"] == 350_000


def test_the_actual_starters_count_a_scratch_after_t(data_dir, monkeypatch):
    boards = build_logged_day(data_dir)
    guard_holdout_reads(monkeypatch)
    r4 = {r["race_id"]: r for r in er.load_logged_races()}[RID[4]]
    assert r4["win_odds_t"] == boards["win_r4_t"]  # five runners at t: selection's board
    assert r4["final_win_odds"] == boards["win_r4_f"]
    assert r4["n_starters"] == 4


def test_less_popular_is_the_day_tertile_of_the_pool_at_t_over_every_logged_race(
    data_dir, monkeypatch
):
    build_logged_day(data_dir)
    guard_holdout_reads(monkeypatch)
    races = {r["race_id"]: r for r in er.load_logged_races()}
    assert {rid: r["pool_total_t"] for rid, r in races.items()} == {
        RID[1]: 700_000, RID[2]: 450_000, RID[3]: 350_000, RID[4]: 900_000, RID[5]: 1_500_000,
        RID[6]: 250_000}  # fmt: skip
    # cutoff over all six (races 3 and 4 included, though E2 will not score them): 416.7k
    assert {rid: r["less_popular"] for rid, r in races.items()} == {
        RID[1]: False, RID[2]: False, RID[3]: True, RID[4]: False, RID[5]: False,
        RID[6]: True}  # fmt: skip


def test_load_logged_races_on_an_empty_data_dir_gives_no_races():
    assert er.load_logged_races() == []


def test_races_without_a_board_at_t_or_a_final_pool_total_are_still_loaded(data_dir, monkeypatch):
    boards = build_logged_day(data_dir, extra_day=True)
    guard_holdout_reads(monkeypatch)
    races = {r["race_id"]: r for r in er.load_logged_races()}
    assert set(races) == set(RID.values()) | set(RID2.values())
    k1, k2, k3 = races[RID2[1]], races[RID2[2]], races[RID2[3]]
    assert not k1["win_odds_t"] and not k1["exacta_odds_t"]  # nothing available at t
    assert k1["final_win_odds"] == boards["k1_win_f"]
    assert k1["final_exacta_odds"] == boards["k1_fc_f"] and k1["pool_total"] == 800_000
    assert k1["pool_total_t"] is None and k1["less_popular"] is False
    assert k2["win_odds_t"] == boards["k2_win_t"] and k2["exacta_odds_t"] == boards["k2_fc_t"]
    assert k2["final_win_odds"] == boards["k2_win_f"]
    assert k2["final_exacta_odds"] == boards["k2_fc_f"]  # verified final prices are kept
    assert k2["pool_total"] is None and k2["pool_total_t"] == 600_000
    assert k3["pool_total"] == 2_000_000 and k3["final_exacta_odds"] == boards["k3_fc_f"]
    # day 2's tertile at t over 600k and 300k: cutoff 400k
    assert (k2["less_popular"], k3["less_popular"]) == (False, True)


def ref_d(win: dict, exacta: dict, rate: float) -> dict[str, float]:
    """sum_S q / sum_S p per category on one final board (q NaN at 110 yen: in neither sum)."""
    p = ref_p(win)
    h = ref_harville(p, LAMBDA2)
    fav = ref_fav(win)
    q_sum, p_sum = dict.fromkeys(ALL_CATS, 0.0), dict.fromkeys(ALL_CATS, 0.0)
    for combo, odds in exacta.items():
        if odds is None or combo[0] not in p or combo[1] not in p:
            continue
        q = ref_q(odds, rate)
        if not math.isnan(q):
            q_sum[ref_cat(combo, fav)] += q
            p_sum[ref_cat(combo, fav)] += h[combo]
    return {c: q_sum[c] / p_sum[c] if p_sum[c] else math.nan for c in ALL_CATS}


def test_d_and_the_proxy_count_the_races_e2_cannot_score(data_dir, monkeypatch, tmp_path):
    """D over every race with final Win and FORECAST prices (Kyoto 1 has no board at t, Kyoto 2
    no final pool total); the proxy over every race with a final pool total (Kyoto 1 included)."""
    boards = build_logged_day(data_dir, extra_day=True)
    guard_holdout_reads(monkeypatch)
    races = er.load_logged_races()
    finals = {
        RID[1]: (boards["win_f"], boards["fc_f"]),
        RID[2]: (boards["win_r2_f"], boards["fc_r2_f"]),
        RID[4]: (boards["win_r4_f"], forecast_board(range(1, 5), 1.3)),
        RID[5]: (boards["win_r5_f"], boards["fc_r5_f"]),
        RID[6]: (boards["win_r6_f"], boards["fc_r6_f"]),
        **{RID2[no]: (boards[f"k{no}_win_f"], boards[f"k{no}_fc_f"]) for no in (1, 2, 3)},
    }  # race 3 of day 1 has no final FORECAST capture
    rate = takeout.race_takeout(RID[1], "FORECAST", race_class_raw="3歳未勝利").payout_rate
    ratios = [ref_d(win, fc, rate) for win, fc in finals.values()]
    got = er.d_descriptive(races, lambda2=LAMBDA2)
    for cat in ALL_CATS:
        values = [r[cat] for r in ratios if not math.isnan(r[cat])]
        assert got[cat]["n_races"] == len(values) == 8, cat
        assert got[cat]["mean"] == pytest.approx(statistics.mean(values), rel=1e-6), cat
        assert got[cat]["median"] == pytest.approx(statistics.median(values), rel=1e-6), cat

    proxy = er.proxy_agreement(races)
    # day 1 as before (1, 3, 1, 0); day 2's final pools 800k and 2.0M: cutoff 1.2M, Kyoto 1 in;
    # both are races 1-4 (the proxy)
    assert (proxy["n_races"], proxy["n_days"]) == (7, 2)
    assert (proxy["both_less"], proxy["proxy_only"], proxy["tertile_only"], proxy["neither"]) == (
        2, 4, 1, 0)  # fmt: skip
    assert proxy["agreement"] == pytest.approx(2 / 7)

    result = er.score_e2(races, registry=new_registry(tmp_path / "reg"), git_sha="abc1234",
                         lambda2=LAMBDA2, data_builds=BUILDS, n_boot=50)  # fmt: skip
    assert result["n_races"] == 5 and result["n_meetings"] == 2
    assert {k: v for k, v in result["excluded"].items() if v} == {
        "no_final_board": 2, "under_5_starters": 1, "no_board_at_t": 1}  # fmt: skip


def test_logged_races_score_end_to_end(data_dir, monkeypatch, tmp_path):
    boards = build_logged_day(data_dir)
    guard_holdout_reads(monkeypatch)
    races = er.load_logged_races()
    reg = new_registry(tmp_path / "reg")
    result = er.score_e2(races, registry=reg, git_sha="abc1234", lambda2=LAMBDA2,
                         data_builds=BUILDS, n_boot=50)  # fmt: skip
    assert result["n_races"] == 4 and result["n_meetings"] == 1
    assert {k: v for k, v in result["excluded"].items() if v} == {
        "no_final_board": 1, "under_5_starters": 1}  # fmt: skip
    by_id = {r["race_id"]: r for r in races}
    sel = {
        1: ref_select(boards["win_a"], boards["fc_a"], RATE, LAMBDA2),
        2: ref_select(boards["win_r2_t"], boards["fc_r2_t"], RATE, LAMBDA2),
        5: ref_select(boards["win_r5_t"], boards["fc_r5"], RATE, LAMBDA2),
        6: ref_select(boards["win_r6_t"], boards["fc_r6"], RATE, LAMBDA2),
    }
    assert all(any(s.values()) for s in sel.values())
    final_pool = {1: 1_000_000, 2: 3_000_000, 5: 1_100_000, 6: 900_000}
    for (cat, k, scope), got in by_member(result).items():
        nos = (6,) if scope == "headline" else (1, 2, 5, 6)
        assert got["n_bets"] == sum(len(sel[no][(cat, k)]) for no in nos), (cat, k, scope)
        payout = 0.0
        for no in nos:
            chosen = sel[no][(cat, k)]
            winner, dividend = boards["winners"][no]
            if winner in chosen:
                t = 1 - by_id[RID[no]]["rate"]
                payout += ref_paid(dividend, final_pool[no], len(chosen), t=t)
        assert got["payout"] == pytest.approx(payout, rel=1e-9), (cat, k, scope)
    assert any(m["payout"] > 0 for m in result["members"])


# ---------------------------------------------------------------------------------------------
# The command: stable build identities, idempotent reruns, the full report
# ---------------------------------------------------------------------------------------------

SHA = "281e2e2" * 5 + "abcde"
REPO_SUMMARY = env.repo_root() / "docs" / "experiments" / "registry" / "japan"


def _snapshot(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(folder.glob("*")) if p.is_file()}


def write_fits(data: Path) -> None:
    """The v0 fit files ``exacta_run._fits`` reads (ids and lambda2 only)."""
    model = data / "models" / "japan" / "v0"
    model.mkdir(parents=True, exist_ok=True)
    oof = model / "win_calibration_oof-winfit281.parquet"
    files = {
        "win-winfit281.json": {"fit_id": "winfit281"},
        "ordering-ordfit281.json": {"fit_id": "ordfit281", "lambdas": [LAMBDA2]},
        "manifest-ordering-ordfit281.json": {"oof": {"path": str(oof)}},
        "current.json": {"win": "win-winfit281.json", "ordering": "ordering-ordfit281.json"},
    }
    for name, body in files.items():
        (model / name).write_text(json.dumps(body), encoding="utf-8")


@pytest.fixture
def command(data_dir, tmp_path, monkeypatch):
    """The command against tmp data and a tmp summary dir; the repo's summary must not change."""
    before = _snapshot(REPO_SUMMARY)
    summary = tmp_path / "summary" / "japan"
    summary.mkdir(parents=True)
    monkeypatch.setattr(registry_module, "summary_dir", lambda country: summary)
    monkeypatch.setattr(er, "summary_dir", lambda country: summary, raising=False)
    monkeypatch.setattr(exacta_run, "summary_dir", lambda country: summary, raising=False)
    monkeypatch.setattr(exacta_run, "_git_sha", lambda: SHA)
    write_fits(data_dir)
    reg = default_registry("japan")
    reg.freeze_family(E2_ID, family_specs()[E2_ID])
    yield reg
    after = _snapshot(REPO_SUMMARY)
    if after != before:
        for name in set(after) - set(before):
            (REPO_SUMMARY / name).unlink()
        for name, raw in before.items():
            (REPO_SUMMARY / name).write_bytes(raw)
        pytest.fail("the test wrote the repository's committed Japan summary (restored)")


def latest_report(suffix: str) -> Path:
    found = [p for p in paths.reports_dir().rglob(f"*{suffix}") if "281" in str(p)]
    assert found, f"no {suffix} report under {paths.reports_dir()}"
    return max(found, key=lambda p: p.stat().st_mtime_ns)


def test_an_empty_run_then_a_populated_run_keep_one_family_of_16(command, data_dir):
    er.exacta_rules_command()  # nothing logged yet
    assert len(command.records(E2_ID)) == 16
    build_logged_day(data_dir)
    er.exacta_rules_command()
    records = command.records(E2_ID)
    assert len(records) == 32
    assert {tuple(sorted(r.data_builds)) for r in records} == {BUILD_KEYS}
    assert command.family_size(E2_ID) == 16


BUILD_KEYS = ("jra_dividends", "jra_races", "jra_runners", "odds_log")


def rewrite_table(data: Path, name: str, change) -> None:
    path = data / "processed" / "jra" / f"{name}.parquet"
    frame = change(pd.read_parquet(path))
    frame.to_parquet(path, index=False)


def add_other_races(data: Path) -> None:
    """Store rows of a holdout race and of an unlogged development race: nothing consumes them."""
    others = ["jra:2026-11-29:NAK:02", "jra:2026-11-08:HSN:01"]
    races = [store_race(date(2026, 11, 29), "NAK", 2, "C1"), store_race(date(2026, 11, 8), "HSN",
                                                                          1, "MAIDEN")]  # fmt: skip
    dividends = [{"race_id": rid, "pool": pool, "combination": combo, "dividend": d,
                  "is_refund": False}
                 for rid in others for pool, combo, d in (("FORECAST", "3,1", 4410.0),
                                                          ("WIN", "3", 610.0))]  # fmt: skip
    runners = [row for rid in others for row in store_runners(rid, range(1, 9))]
    for name, rows in (("races", races), ("dividends", dividends), ("runners", runners)):
        rewrite_table(data, name, lambda f, rows=rows: pd.concat([f, pd.DataFrame(rows)],
                                                                  ignore_index=True))  # fmt: skip


def test_build_ids_fingerprint_only_the_consumed_development_rows(command, data_dir):
    build_logged_day(data_dir)
    er.exacta_rules_command()
    before = [record_key(r) for r in command.records()]
    builds = {json.dumps(r.data_builds, sort_keys=True) for r in command.records(E2_ID)}
    assert len(builds) == 1 and tuple(sorted(json.loads(builds.pop()))) == BUILD_KEYS
    add_other_races(data_dir)
    er.exacta_rules_command()
    assert [record_key(r) for r in command.records()] == before

    # a consumed development row is another build of its table, and only of that table
    edits = [
        ("jra_dividends", "dividends", lambda f: f.assign(dividend=[
            1240.0 if (r == RID[1] and p == "FORECAST") else d
            for r, p, d in zip(f["race_id"], f["pool"], f["dividend"], strict=True)])),
        ("jra_races", "races", lambda f: f.assign(race_class=[
            "C2" if r == RID[2] else c for r, c in zip(f["race_id"], f["race_class"],
                                                       strict=True)])),
        ("jra_runners", "runners", lambda f: f.assign(is_scratched=[
            True if (r == RID[2] and h == 6) else s
            for r, h, s in zip(f["race_id"], f["horse_no"], f["is_scratched"], strict=True)])),
    ]  # fmt: skip
    for key, table, change in edits:
        old = command.records(E2_ID)[-1].data_builds
        rewrite_table(data_dir, table, change)
        er.exacta_rules_command()
        new = command.records(E2_ID)[-1].data_builds
        assert new[key] != old[key], key
        assert {k: v for k, v in new.items() if k != key} == {
            k: v for k, v in old.items() if k != key}, key  # fmt: skip


def test_rerunning_the_command_on_the_same_inputs_adds_nothing(command, data_dir):
    build_logged_day(data_dir)
    er.exacta_rules_command()
    before = [record_key(r) for r in command.records()]
    er.exacta_rules_command()  # does not raise
    assert [record_key(r) for r in command.records()] == before
    assert latest_report(".md").exists()


def test_the_command_reports_the_proxy_over_every_loaded_race(command, data_dir):
    build_logged_day(data_dir)
    er.exacta_rules_command()
    report = json.loads(latest_report(".json").read_text(encoding="utf-8"))
    # final pools of races 1, 2, 4, 5, 6 (race 3 has none): cutoff 0.933M -> races 4 and 6;
    # proxy (race 1-4 or maiden): races 1, 2, 4, 5
    proxy = report["proxy"]
    assert (proxy["n_races"], proxy["n_days"]) == (5, 1)
    assert (proxy["both_less"], proxy["proxy_only"], proxy["tertile_only"], proxy["neither"]) == (
        1, 3, 1, 0)  # fmt: skip
    assert proxy["agreement"] == pytest.approx(1 / 5)


def test_the_command_reports_d_and_the_proxy_over_races_e2_does_not_score(command, data_dir):
    build_logged_day(data_dir, extra_day=True)
    er.exacta_rules_command()
    report = json.loads(latest_report(".json").read_text(encoding="utf-8"))
    assert (report["proxy"]["n_races"], report["proxy"]["n_days"]) == (7, 2)
    assert {c: report["d"][c]["n_races"] for c in ALL_CATS} == dict.fromkeys(ALL_CATS, 8)
    assert report["n_races"] == 5


def shows(text: str, value: float) -> bool:
    return any(f"{value:.{k}f}" in text for k in (2, 3, 4))


def test_the_markdown_report_renders_the_max_test_d_and_the_proxy(command, data_dir):
    build_logged_day(data_dir)
    er.exacta_rules_command()
    md = latest_report(".md").read_text(encoding="utf-8")
    report = json.loads(latest_report(".json").read_text(encoding="utf-8"))
    # one meeting: the max test cannot run (arch needs three clusters) and says why
    (max_test,) = command.records(er.MAX_TEST_ID)
    assert max_test.status == "failed"
    reason = max_test.params["error"]
    assert reason[:40] in md and "failed" in md.lower()
    d = report["d"]["fav_first"]
    assert d["n_races"] > 0 and shows(md, d["mean"])  # fav_first appears only in D
    assert "fav_first" in md
    assert "agreement" in md.lower() and shows(md, report["proxy"]["agreement"])
    rows = [(line, [c.strip() for c in line.strip("|").split("|")])
            for line in md.splitlines() if line.startswith("|")]  # fmt: skip
    headline = [line for line, cells in rows if any(c.startswith("headline") for c in cells)]
    every = [line for line, cells in rows if "all" in cells]
    assert len(headline) == len(every) == 8
    assert all("retrospective" in line.lower() for line in headline)  # labelled beside each
    assert not any("retrospective" in line.lower() for line in every)
