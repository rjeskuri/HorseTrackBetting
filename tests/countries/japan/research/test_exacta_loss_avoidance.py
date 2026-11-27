"""Family L on the exacta (#282): skip a fav_first exacta when q_c / p_c at T-2 is above a cut.

Owner, #282, 2026-11-26 (approved) and 2026-11-27 (reframe), building on #131's E2 (rule-v3,
docs/specs/exacta-overload-prereg.md sections 5 to 7). All data is synthetic; nothing reads the
store, the odds logs or the real registry.

API pinned here (module ``horsetrack.countries.japan.research.exacta_loss_avoidance``, imported as
``la``; not yet written; ``exacta_study`` and ``exacta_consistency`` are reused, not copied, and
``exacta_study.family_specs()`` keeps its five #131 families):

Constants: ``FAMILY_ID = "japan/282-exacta-fav-first-avoidance-v1"``, ``CUTS = (1.1, 1.2, 1.3,
1.5)``, ``SCOPES = ("all", "headline")``.

Selection (pure; a board maps horse -> decimal Win odds, an exacta board maps (first, second) ->
decimal odds, so ``odds * 100`` is the dividend per 100 yen):

- ``fav_first_ratios(win_board, exacta_board, *, rate, lambda2=1.0) -> {combo: r_c}``: the
  favourite is the lowest odds on ``win_board``, the Win board at T-2 (ties to the lowest number).
  Only combinations of ``exacta_board`` whose first horse is that favourite are returned.
  ``r_c = q_c / p_c``, ``q_c = exacta_consistency.implied_share(odds * 100, rate=rate)`` (NaN at
  or below 110 yen) and ``p_c`` the discounted Harville exacta (``lambda2``) of the Win
  probabilities ``(1 / odds)`` renormalised over the starters of the Win board.
- ``split_by_cut(ratios, cut) -> (kept, skipped)``: tuples of combos, each sorted ascending. A
  combination is skipped when ``r_c > cut`` strictly; ``r_c == cut`` is kept, and a NaN ratio (q
  not identified) is kept: the rule only skips on evidence that the pool overloads c.
- ``select(win_board, exacta_board, *, rate, cut, lambda2=1.0)`` -> object with ``.favourite``,
  ``.ratios``, ``.kept``, ``.skipped`` (the two above on those ratios).
- ``less_popular_races(rows)``: ``rows`` are ``(race_id, race_date, exacta_pool_total)`` of the
  logged races; returns the set of race ids in the lower tertile of the pool total within their
  day (the ``ceil(n / 3)`` smallest of the day's ``n`` races; spec section 6, #130 days).

Scoring:

- A race is a mapping with race_id, race_date, meeting_id, race_no, ``win_board_t2``,
  ``exacta_board_t2``, ``less_popular`` (bool), ``rate`` (the exacta payout rate), ``exacta`` (the
  final rows ``(combination, dividend, is_refund)``, as in #131) and ``boost_may_be_withheld``. It
  may carry final-board keys (e.g. ``win_odds_final``), which selection never reads.
- ``score_races(races, *, holdout_day, lambda2=1.0) -> DataFrame``: one row per scored race and
  cut: race_id, meeting_id, race_date, cut, ``headline`` (T-2 favourite odds >= 3.0 and
  ``less_popular``), ``n_all`` (fav_first combinations on the T-2 exacta board, the comparator's
  tickets, 100 yen each), ``n_kept``, ``n_skipped``, ``ret_all``, ``ret_kept``, ``ret_skipped``
  (yen: the final dividend when the winning exacta is in the set and not refunded, else 0).
  Holdout days are dropped before anything else of the race is read; excluded races (#131's
  ``exclusion_reason``) are dropped. ``frame.attrs["excluded"]`` counts the drops by reason
  (``"holdout_day"`` among them).
- ``member_table(rows, *, n_boot, seed=282, level=0.95) -> DataFrame``: for every member (cut,
  scope) and each set in ``("kept", "skipped")``: columns ``cut, scope, set, n_races, n_meetings,
  stake, ret, per_yen, all_per_yen, diff, se, lo, hi, adj_lo, adj_hi``. Over the scope's races,
  ``stake = 100 * sum(n_set)``, ``ret = sum(ret_set)``, ``per_yen = ret / stake`` and ``diff =
  per_yen - all_per_yen`` (the set minus buying every fav_first combination), NaN when the set has
  no stake. ``se``, ``lo``, ``hi``: meeting cluster bootstrap of ``diff``, one-sided bounds at
  ``level`` as in core.eval.bootstrap (``lo`` the lower, ``hi`` the upper). ``adj_lo``, ``adj_hi``:
  the same bounds widened by a max-statistic over the 8 members of the set.
- ``score(registry, races, meta, sha, *, holdout_day=is_holdout_day, n_boot=10000) -> dict`` with
  ``"members"`` (the member_table) and ``"meta"`` (``scored_races``, ``scored_meetings``,
  ``excluded``). It raises ``exacta_run.RunRefused`` (naming the family, nothing appended) unless
  ``FAMILY_ID`` is frozen with the hash of ``family_specs()[FAMILY_ID]``. ``meta`` carries
  ``lambda2_full_development``, ``ordering_fit`` and ``data_builds``. It appends one registry
  record per member (8; family ``FAMILY_ID``, ticket 282, sources ``("jra",)``, status ``scored``,
  ``params`` ``{"cut", "scope"}``) whose metrics include ``kept_minus_all`` and
  (when the set has stake) ``skipped_minus_all``, each with lo, hi and n = the member's races; a
  rerun on the same inputs appends nothing. No meeting minimum applies.
- ``family_specs() -> {FAMILY_ID: spec}`` (plain JSON, a fresh copy per call), ``freeze(registry)
  -> {FAMILY_ID: hash}`` and ``main(["freeze"])`` (``default_registry("japan")``). Spec keys:
  ticket (282), category ("fav_first"), selection_time ("T-2"), members ([{"cut", "scope"}] x 8),
  cuts, scopes, family_size (8), stake_yen (100), exacta_takeout (0.25), lambda2
  ("full_development", as E2), sources (["jra"]), cluster ("meeting"), n_boot (10000), level
  (0.95), rule (text naming q_c, p_c and T-2), comparator (text naming fav_first and 100 yen),
  primary_metric (text naming the kept set and final dividends) and report_sets (["kept",
  "skipped"]); no ``min_meetings`` gate. ``score`` reads ``family_specs`` at call time, so a
  spec that drifts from the frozen one is refused.
"""

from __future__ import annotations  # noqa: I001

import copy
import hashlib
import json
import math
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horsetrack.core import env
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import Registry
from horsetrack.countries.japan.research import exacta_run
from horsetrack.countries.japan.research import exacta_loss_avoidance as la

CUTS = (1.1, 1.2, 1.3, 1.5)
SCOPES = ("all", "headline")
SHA = "0123abcd" * 5
REPO_SUMMARY = env.repo_root() / "docs" / "experiments" / "registry" / "japan"

# ---------------------------------------------------------------------------------------------
# Independent references (loops and formulas from the spec, never the code under test)
# ---------------------------------------------------------------------------------------------


def ref_win_p(win: dict[int, float]) -> dict[int, float]:
    inv = {h: 1.0 / o for h, o in win.items()}
    total = sum(inv.values())
    return {h: v / total for h, v in inv.items()}


def ref_exacta_p(win: dict[int, float], lambda2: float) -> dict[tuple[int, int], float]:
    """P(first = i, second = j) = p_i p_j^l2 / sum_{l != i} p_l^l2."""
    p = ref_win_p(win)
    w = {h: v**lambda2 for h, v in p.items()}
    total = sum(w.values())
    return {(i, j): p[i] * w[j] / (total - w[i]) for i in p for j in p if i != j}


def ref_fav(win: dict[int, float]) -> int:
    return min(win, key=lambda h: (win[h], h))


def ref_ratios(win, board, rate, lambda2=1.0) -> dict[tuple[int, int], float]:
    fav, px, out = ref_fav(win), ref_exacta_p(win, lambda2), {}
    for combo, odds in board.items():
        if combo[0] != fav:
            continue
        dividend = odds * 100
        out[combo] = math.nan if dividend <= 110 else (rate / ((dividend + 5) / 100)) / px[combo]
    return out


def ref_split(ratios, cut):
    kept = sorted(c for c, r in ratios.items() if not r > cut)  # NaN > cut is False: kept
    skipped = sorted(c for c, r in ratios.items() if r > cut)
    return kept, skipped


def board_from(win, targets, rate=0.75, lambda2=1.0):
    """Exacta odds whose r_c is exactly the target: q = rate / (odds + 0.05) = r * p_c."""
    px = ref_exacta_p(win, lambda2)
    return {c: rate / (r * px[c]) - 0.05 for c, r in targets.items()}


def assert_clear_of_cuts(ratios, gap=0.015):
    for r in ratios.values():
        if not math.isnan(r):
            assert min(abs(r - k) for k in CUTS) > gap, r


# ---------------------------------------------------------------------------------------------
# 1. Selection on synthetic T-2 boards
# ---------------------------------------------------------------------------------------------

WIN6 = {1: 2.0, 2: 4.0, 3: 5.0, 4: 10.0, 5: 20.0, 6: 25.0}  # favourite 1 at T-2
FAV_FIRST_TARGETS = {(1, 2): 1.05, (1, 3): 1.15, (1, 4): 1.25, (1, 5): 1.40, (1, 6): 1.80}
OTHER_TARGETS = {  # not fav_first: high and low r, none of them may ever be kept or skipped
    (2, 1): 3.0, (3, 1): 2.5, (2, 3): 2.2, (6, 2): 1.9, (4, 5): 0.5, (5, 4): 0.6,
}  # fmt: skip
EXPECTED_KEPT = {
    1.1: [(1, 2)],
    1.2: [(1, 2), (1, 3)],
    1.3: [(1, 2), (1, 3), (1, 4)],
    1.5: [(1, 2), (1, 3), (1, 4), (1, 5)],
}


def _board6(rate=0.75, lambda2=1.0):
    return board_from(WIN6, {**FAV_FIRST_TARGETS, **OTHER_TARGETS}, rate, lambda2)


@pytest.mark.parametrize("cut", CUTS)
def test_each_cut_keeps_the_low_ratio_fav_first_combinations_and_skips_the_rest(cut):
    sel = la.select(WIN6, _board6(), rate=0.75, cut=cut)
    all_fav_first = sorted(FAV_FIRST_TARGETS)
    assert sel.favourite == 1
    assert list(sel.kept) == EXPECTED_KEPT[cut]
    assert list(sel.skipped) == [c for c in all_fav_first if c not in EXPECTED_KEPT[cut]]
    # the kept and the skipped sets partition the fav_first combinations: nothing else, nothing lost
    assert sorted([*sel.kept, *sel.skipped]) == all_fav_first
    assert set(sel.ratios) == set(all_fav_first)
    for combo, target in FAV_FIRST_TARGETS.items():
        assert sel.ratios[combo] == pytest.approx(target, rel=1e-9)


def test_the_kept_set_only_grows_with_the_cut():
    kept = [set(la.select(WIN6, _board6(), rate=0.75, cut=k).kept) for k in CUTS]
    assert all(a < b for a, b in zip(kept, kept[1:], strict=False))


@pytest.mark.parametrize("rate", [0.70, 0.75, 0.80])
def test_the_exacta_payout_rate_scales_q_and_so_changes_the_selection(rate):
    board = _board6(rate=0.75)  # priced at 0.75, selected with the race's own rate
    ref = ref_ratios(WIN6, board, rate)
    assert_clear_of_cuts(ref, gap=0.005)
    for cut in CUTS:
        sel = la.select(WIN6, board, rate=rate, cut=cut)
        kept, skipped = ref_split(ref, cut)
        assert (list(sel.kept), list(sel.skipped)) == (kept, skipped), (rate, cut)
    # the rate moves decisions: at 0.80 r is 1.067 x the target, so (1,4) goes from 1.25 to 1.333
    assert (1, 4) in la.select(WIN6, board, rate=0.75, cut=1.3).kept
    assert (1, 4) in la.select(WIN6, board, rate=0.80, cut=1.3).skipped


@pytest.mark.parametrize("lambda2", [1.0, 0.7])
def test_p_c_is_the_discounted_harville_of_the_win_board(lambda2):
    board = _board6(lambda2=1.0)
    ref = ref_ratios(WIN6, board, 0.75, lambda2)
    assert_clear_of_cuts(ref, gap=0.005)
    got = la.fav_first_ratios(WIN6, board, rate=0.75, lambda2=lambda2)
    assert set(got) == set(ref)
    for combo, value in ref.items():
        assert got[combo] == pytest.approx(value, rel=1e-9)
    for cut in CUTS:
        sel = la.select(WIN6, board, rate=0.75, cut=cut, lambda2=lambda2)
        assert (list(sel.kept), list(sel.skipped)) == ref_split(ref, cut), (lambda2, cut)


def test_lambda2_changes_the_decisions_somewhere():
    board = _board6(lambda2=1.0)
    plain = [la.select(WIN6, board, rate=0.75, cut=k, lambda2=1.0).kept for k in CUTS]
    disc = [la.select(WIN6, board, rate=0.75, cut=k, lambda2=0.7).kept for k in CUTS]
    assert plain != disc


def test_the_favourite_comes_from_the_t2_win_board_and_sets_which_combinations_count():
    board = _board6()
    # Same exacta board, a different T-2 Win board: horse 2 is the favourite, so (2, x) counts.
    win = {1: 4.0, 2: 2.0, 3: 5.0, 4: 10.0, 5: 20.0, 6: 25.0}
    ref = ref_ratios(win, board, 0.75)
    sel = la.select(win, board, rate=0.75, cut=1.2)
    assert sel.favourite == 2
    assert set(sel.ratios) == {c for c in board if c[0] == 2} == set(ref)
    kept, skipped = ref_split(ref, 1.2)
    assert (list(sel.kept), list(sel.skipped)) == (kept, skipped)
    assert not any(c[0] == 1 for c in [*sel.kept, *sel.skipped])


def test_a_tie_for_the_favourite_goes_to_the_lowest_horse_number():
    win = {1: 5.0, 2: 3.0, 3: 3.0, 4: 8.0, 5: 12.0, 6: 30.0}
    px = ref_exacta_p(win, 1.0)
    board = {c: 0.75 / (1.05 * p) - 0.05 for c, p in px.items()}
    sel = la.select(win, board, rate=0.75, cut=1.2)
    assert sel.favourite == 2
    assert {c[0] for c in sel.ratios} == {2}


def test_only_combinations_priced_on_the_t2_board_are_considered():
    board = _board6()
    del board[(1, 4)], board[(1, 6)]
    sel = la.select(WIN6, board, rate=0.75, cut=1.2)
    assert sorted([*sel.kept, *sel.skipped]) == [(1, 2), (1, 3), (1, 5)]
    assert list(sel.kept) == [(1, 2), (1, 3)] and list(sel.skipped) == [(1, 5)]


def test_a_combination_whose_share_is_not_identified_is_kept():
    board = _board6()
    board[(1, 2)] = 1.1  # 110 yen: the minimum payout may bind, q is NaN
    sel = la.select(WIN6, board, rate=0.75, cut=1.1)
    assert math.isnan(sel.ratios[(1, 2)])
    assert (1, 2) in sel.kept and (1, 2) not in sel.skipped
    assert (1, 6) in sel.skipped


def test_the_rule_skips_strictly_above_the_cut_and_keeps_a_ratio_equal_to_it():
    ratios = {(1, 2): 1.2, (1, 3): 1.2000001, (1, 4): 1.1999999, (1, 5): math.nan, (1, 6): 3.0}
    kept, skipped = la.split_by_cut(ratios, 1.2)
    assert (tuple(kept), tuple(skipped)) == (((1, 2), (1, 4), (1, 5)), ((1, 3), (1, 6)))
    kept, skipped = la.split_by_cut(ratios, 1.5)
    assert (tuple(kept), tuple(skipped)) == (((1, 2), (1, 3), (1, 4), (1, 5)), ((1, 6),))
    assert la.split_by_cut({}, 1.2) == ((), ())


# ---------------------------------------------------------------------------------------------
# 2. The less-popular proxy on logged days
# ---------------------------------------------------------------------------------------------


def test_less_popular_is_the_lower_tertile_of_the_pool_total_within_each_day():
    rng = np.random.default_rng(3)
    rows, expected = [], set()
    for day, n in ((date(2025, 5, 3), 12), (date(2025, 5, 4), 9), (date(2025, 5, 10), 7)):
        totals = rng.permutation(np.arange(1, n + 1) * 1_000_000 + rng.integers(0, 999, n))
        ids = [f"{day.isoformat()}:{k:02d}" for k in range(n)]
        rows += [(rid, day, float(t)) for rid, t in zip(ids, totals, strict=True)]
        order = np.argsort(totals)
        expected |= {ids[i] for i in order[: math.ceil(n / 3)]}
    assert la.less_popular_races(rows) == expected
    # a tertile within the day, not across days: day two's smallest pools are not all flagged
    assert len(expected) == 4 + 3 + 3


# ---------------------------------------------------------------------------------------------
# 3. The family spec and its freeze
# ---------------------------------------------------------------------------------------------


def _registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / "japan.jsonl", holdout_log=tmp_path / "holdout_log.jsonl")


def _spec_hash(spec) -> str:
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_the_family_is_8_members_with_the_approved_cuts_scopes_comparator_and_metric():
    specs = la.family_specs()
    assert list(specs) == [la.FAMILY_ID] == ["japan/282-exacta-fav-first-avoidance-v1"]
    spec = specs[la.FAMILY_ID]
    assert la.CUTS == CUTS and la.SCOPES == SCOPES
    members = {(m["cut"], m["scope"]) for m in spec["members"]}
    assert len(spec["members"]) == 8 == spec["family_size"]
    assert members == {(k, s) for k in CUTS for s in SCOPES}
    assert sorted(spec["cuts"]) == list(CUTS)
    assert sorted(spec["scopes"]) == sorted(SCOPES)
    assert spec["ticket"] == 282
    assert spec["category"] == "fav_first"
    assert spec["selection_time"] == "T-2"
    assert spec["stake_yen"] == 100
    assert spec["exacta_takeout"] == 0.25
    assert spec["lambda2"] == "full_development"
    assert spec["sources"] == ["jra"]
    assert spec["cluster"] == "meeting"
    assert spec["n_boot"] == 10000 and spec["level"] == 0.95
    # the comparator, the metric and the rule are stated, and the favourite is the T-2 one
    text = {k: str(spec[k]).lower() for k in ("comparator", "primary_metric", "rule")}
    assert "fav_first" in text["comparator"] and "100" in text["comparator"]
    assert "final" in text["primary_metric"] and "kept" in text["primary_metric"]
    assert "q_c" in text["rule"] and "p_c" in text["rule"] and "t-2" in text["rule"]
    # the owner's reframe: both sets are reported, and no meeting gate is in the spec
    assert sorted(spec["report_sets"]) == ["kept", "skipped"]
    assert spec.get("min_meetings") in (None, 0)


def test_the_spec_is_plain_json_a_fresh_copy_and_freezes_under_one_hash(tmp_path):
    first, second = la.family_specs(), la.family_specs()
    assert first == second
    first[la.FAMILY_ID]["members"].clear()
    assert la.family_specs() == second  # mutating a copy changes nothing
    reg = _registry(tmp_path)
    frozen = la.freeze(reg)
    assert list(frozen) == [la.FAMILY_ID]
    assert frozen[la.FAMILY_ID] == _spec_hash(second[la.FAMILY_ID])
    assert reg.frozen_spec_hash(la.FAMILY_ID) == frozen[la.FAMILY_ID]
    assert la.freeze(reg) == frozen  # freezing the same spec again is a no-op
    assert reg.records() == []  # a freeze is not a run


def test_main_freeze_records_the_spec_in_the_japan_registry_and_leaves_the_summary_alone(
    tmp_path, monkeypatch
):
    before = {p.name: p.read_bytes() for p in sorted(REPO_SUMMARY.glob("*")) if p.is_file()}
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    assert la.main(["freeze"]) == 0
    reg_path = tmp_path / "registry" / "japan.jsonl"
    reg = Registry(reg_path, holdout_log=tmp_path / "none.jsonl")
    assert reg.frozen_spec_hash(la.FAMILY_ID) == _spec_hash(la.family_specs()[la.FAMILY_ID])
    assert reg.records() == []
    after = {p.name: p.read_bytes() for p in sorted(REPO_SUMMARY.glob("*")) if p.is_file()}
    assert after == before


# ---------------------------------------------------------------------------------------------
# 4. Synthetic logged races, the per-race rows and the references
# ---------------------------------------------------------------------------------------------

D1, D2, D3 = date(2025, 5, 3), date(2025, 5, 4), date(2025, 5, 10)
TARGETS = (0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.6, 1.9)  # each >= 0.05 from every cut


def make_race(rng, race_id, day, meeting, race_no, *, n, fav_odds, less_popular, lambda2=1.0,
              rate=0.75, hit="random"):  # fmt: skip
    """One synthetic logged race: a T-2 Win board with its favourite at ``fav_odds``, a full T-2
    exacta board priced from a target r per combination, a different final favourite (a decoy)
    and the final winning row. ``hit`` forces the winner into fav_first or away from it."""
    fav = int(rng.integers(1, n + 1))
    others = [h for h in range(1, n + 1) if h != fav]
    steps = 1.2 + np.cumsum(rng.uniform(0.4, 1.8, size=n - 1))
    win = {fav: fav_odds}
    for horse, step in zip(rng.permutation(others), steps, strict=True):
        win[int(horse)] = round(fav_odds * float(step), 1)
    px = ref_exacta_p(win, lambda2)
    targets = {}
    for combo in px:
        targets[combo] = (
            float(rng.choice(TARGETS)) if combo[0] == fav else float(rng.uniform(0.7, 2.6))
        )
    board = board_from(win, targets, rate, lambda2)
    final = dict(win)
    runner_up = min((h for h in win if h != fav), key=lambda h: win[h])
    final[runner_up] = fav_odds * 0.8  # the final favourite is another horse
    combos = sorted(px)
    if hit == "fav_first":
        combos = [c for c in combos if c[0] == fav]
    elif hit == "fav_first_low":  # a winner every cut keeps (target r <= 1.15)
        combos = [c for c in combos if c[0] == fav and targets[c] <= 1.15]
    elif hit == "fav_first_high":  # a winner every cut skips (target r >= 1.6)
        combos = [c for c in combos if c[0] == fav and targets[c] >= 1.6]
    elif hit == "fav_first_mid":  # kept at 1.5, skipped at 1.1 (target r in 1.25 .. 1.45)
        combos = [c for c in combos if c[0] == fav and 1.25 <= targets[c] <= 1.45]
    elif hit == "other":
        combos = [c for c in combos if c[0] != fav]
    winner = combos[int(rng.integers(0, len(combos)))]
    dividend = float(10 * int(rng.integers(13, 900)))
    return {
        "race_id": race_id, "race_date": day, "meeting_id": meeting, "race_no": race_no,
        "win_board_t2": win, "exacta_board_t2": board, "less_popular": less_popular,
        "rate": rate, "boost_may_be_withheld": False,
        "exacta": [(f"{winner[0]},{winner[1]}", dividend, False)],
        "win_odds_final": final,
    }  # fmt: skip


def _winner(race) -> tuple[int, int]:
    a, b = race["exacta"][0][0].split(",")
    return int(a), int(b)


def ref_rows(races, lambda2=1.0) -> pd.DataFrame:
    out = []
    for race in races:
        win, rate = race["win_board_t2"], race["rate"]
        ratios = ref_ratios(win, race["exacta_board_t2"], rate, lambda2)
        winner, dividend, refund = _winner(race), race["exacta"][0][1], race["exacta"][0][2]
        headline = win[ref_fav(win)] >= 3.0 and race["less_popular"]
        for cut in CUTS:
            kept, skipped = ref_split(ratios, cut)
            paid = dividend if not refund else 0.0
            out.append({
                "race_id": race["race_id"], "meeting_id": race["meeting_id"], "cut": cut,
                "headline": headline, "n_all": len(ratios), "n_kept": len(kept),
                "n_skipped": len(skipped), "ret_all": paid if winner in ratios else 0.0,
                "ret_kept": paid if winner in kept else 0.0,
                "ret_skipped": paid if winner in skipped else 0.0,
            })  # fmt: skip
    return pd.DataFrame(out)


def ref_members(rows: pd.DataFrame) -> dict:
    """(cut, scope, set) -> (n_races, n_meetings, stake, ret, per_yen, all_per_yen, diff)."""
    out = {}
    for cut in CUTS:
        for scope in SCOPES:
            sub = rows[(rows["cut"] == cut) & (rows["headline"] if scope == "headline" else True)]
            all_per = sub["ret_all"].sum() / (100 * sub["n_all"].sum())
            for name in ("kept", "skipped"):
                stake, ret = 100 * sub[f"n_{name}"].sum(), sub[f"ret_{name}"].sum()
                per = ret / stake if stake else math.nan
                out[(cut, scope, name)] = (
                    len(sub), sub["meeting_id"].nunique(), stake, ret, per, all_per, per - all_per
                )  # fmt: skip
    return out


def _fixed_races() -> list[dict]:
    """Race 1: fav_first wins, a high-r combo; race 2: another category wins; race 3: headline."""
    rng = np.random.default_rng(282)
    return [
        make_race(
            rng, "a1", D1, "m1", 2, n=7, fav_odds=3.4, less_popular=True, hit="fav_first_low"
        ),
        make_race(rng, "a2", D1, "m1", 9, n=9, fav_odds=2.2, less_popular=False, hit="other"),
        make_race(
            rng, "a3", D2, "m2", 3, n=11, fav_odds=3.0, less_popular=True, hit="fav_first_high"
        ),
        make_race(
            rng, "a4", D2, "m2", 4, n=8, fav_odds=2.9, less_popular=True, hit="fav_first_mid"
        ),
        make_race(
            rng, "a5", D3, "m3", 8, n=6, fav_odds=5.5, less_popular=False, hit="fav_first_low"
        ),
    ]


def test_score_races_matches_the_reference_for_every_race_and_cut():
    races = _fixed_races()
    rows = la.score_races(races, holdout_day=lambda country, day: False)
    ref = ref_rows(races)
    assert len(rows) == len(races) * len(CUTS)
    assert not rows.duplicated(["race_id", "cut"]).any()
    merged = rows.merge(ref, on=["race_id", "cut"], suffixes=("", "_ref"))
    assert len(merged) == len(ref)
    for col in ("n_all", "n_kept", "n_skipped"):
        assert (merged[col] == merged[f"{col}_ref"]).all(), col
    for col in ("ret_all", "ret_kept", "ret_skipped"):
        assert merged[col].to_numpy() == pytest.approx(merged[f"{col}_ref"].to_numpy()), col
    assert (merged["headline"] == merged["headline_ref"]).all()
    assert (merged["meeting_id"] == merged["meeting_id_ref"]).all()
    # the kept and the skipped sets are both produced and together make the comparator
    assert (rows["n_kept"] + rows["n_skipped"] == rows["n_all"]).all()
    assert (rows["ret_kept"] + rows["ret_skipped"] == rows["ret_all"]).all()
    assert (rows["n_skipped"] > 0).any() and (rows["n_kept"] > 0).any()
    assert (rows["ret_skipped"] > 0).any() and (rows["ret_kept"] > 0).any()


def test_the_headline_scope_is_t2_favourite_odds_of_3_or_more_and_less_popular():
    races = _fixed_races()
    rows = la.score_races(races, holdout_day=lambda country, day: False)
    flags = rows.drop_duplicates("race_id").set_index("race_id")["headline"].to_dict()
    # a1: 3.4 and less popular; a2: 2.2; a3: exactly 3.0 and less popular; a4: 2.9; a5: not less
    assert flags == {"a1": True, "a2": False, "a3": True, "a4": False, "a5": False}


def test_the_comparator_buys_every_fav_first_combination_of_the_t2_board_at_100_yen():
    races = _fixed_races()
    rows = la.score_races(races, holdout_day=lambda country, day: False)
    for race in races:
        n_fav_first = sum(
            1 for c in race["exacta_board_t2"] if c[0] == ref_fav(race["win_board_t2"])
        )
        mine = rows[rows["race_id"] == race["race_id"]]
        assert (mine["n_all"] == n_fav_first).all()
    # a race whose winner is not fav_first returns nothing to the comparator, kept or skipped
    a2 = rows[rows["race_id"] == "a2"]
    assert (a2[["ret_all", "ret_kept", "ret_skipped"]] == 0).all().all()
    # a winner fav_first at the T-2 favourite pays its final dividend to whichever set holds it
    a1 = rows[rows["race_id"] == "a1"]
    assert (a1["ret_all"] == a1["ret_all"].iloc[0]).all() and a1["ret_all"].iloc[0] > 0


def test_a_winner_that_is_not_fav_first_at_t2_is_not_a_fav_first_return_even_if_it_is_at_the_end():
    rng = np.random.default_rng(11)
    race = make_race(rng, "d1", D1, "m1", 5, n=8, fav_odds=2.5, less_popular=False, hit="other")
    final_fav = ref_fav(race["win_odds_final"])
    t2_fav = ref_fav(race["win_board_t2"])
    assert final_fav != t2_fav
    # make the winner a combination led by the FINAL favourite: fav_first at the end, not at T-2
    winner = next(c for c in race["exacta_board_t2"] if c[0] == final_fav)
    race["exacta"] = [(f"{winner[0]},{winner[1]}", 4560.0, False)]
    rows = la.score_races([race], holdout_day=lambda country, day: False)
    assert (rows[["ret_all", "ret_kept", "ret_skipped"]] == 0).all().all()


def test_a_refund_a_dead_heat_a_boost_and_a_short_field_are_excluded_not_scored():
    rng = np.random.default_rng(5)
    good = make_race(rng, "g", D1, "m1", 2, n=7, fav_odds=3.1, less_popular=True, hit="fav_first")
    bad = {}
    bad["refund"] = make_race(rng, "x1", D1, "m1", 3, n=7, fav_odds=3.1, less_popular=True)
    bad["refund"]["exacta"] = [(bad["refund"]["exacta"][0][0], 100.0, True)]
    bad["dead_heat"] = make_race(rng, "x2", D1, "m1", 4, n=7, fav_odds=3.1, less_popular=True)
    bad["dead_heat"]["exacta"] = [("1,2", 900.0, False), ("2,1", 1300.0, False)]
    bad["boost_may_be_withheld"] = make_race(rng, "x3", D1, "m1", 5, n=7, fav_odds=3.1,
                                             less_popular=True)  # fmt: skip
    bad["boost_may_be_withheld"]["boost_may_be_withheld"] = True
    small = make_race(rng, "x4", D1, "m1", 6, n=7, fav_odds=3.1, less_popular=True)
    small["win_board_t2"] = {1: 2.0, 2: 3.0, 3: 5.0, 4: 9.0}
    small["exacta_board_t2"] = board_from(small["win_board_t2"], {(1, 2): 1.1, (1, 3): 1.2})
    bad["under_5_starters"] = small
    rows = la.score_races([good, *bad.values()], holdout_day=lambda country, day: False)
    assert set(rows["race_id"]) == {"g"}
    assert rows.attrs["excluded"] == dict.fromkeys(bad, 1)


# ---------------------------------------------------------------------------------------------
# 5. Development days only: no holdout race is read
# ---------------------------------------------------------------------------------------------


class GuardedRace(dict):
    """A race whose fields fail the test when read, except the three that classify its day."""

    OPEN = {"race_id", "race_date", "meeting_id"}

    def _boom(self, key):
        raise AssertionError(f"a holdout race's {key!r} was read")

    def __getitem__(self, key):
        if key not in self.OPEN:
            self._boom(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key not in self.OPEN:
            self._boom(key)
        return super().get(key, default)

    def values(self):
        self._boom("values")

    def items(self):
        self._boom("items")


def _guarded(race) -> GuardedRace:
    return GuardedRace(race)


def test_holdout_days_are_dropped_before_any_board_or_dividend_is_read():
    races = _fixed_races()
    held = _guarded({**make_race(np.random.default_rng(1), "h1", D2, "m2", 3, n=7, fav_odds=3.5,
                                 less_popular=True), "race_date": D2})  # fmt: skip
    seen = []

    def holdout_day(country, day):
        assert country == "japan" and isinstance(day, date)
        seen.append(day)
        return day == D2

    rows = la.score_races([*races, held], holdout_day=holdout_day)
    assert set(rows["race_id"]) == {"a1", "a2", "a5"}  # a3, a4 and h1 are on the held day D2
    assert set(seen) == {D1, D2, D3}
    assert rows.attrs["excluded"]["holdout_day"] == 3


def test_the_committed_holdout_days_are_never_scored_by_default():
    dev_days = [date(2026, 11, 28), date(2026, 12, 5)]
    held_days = [date(2026, 11, 29), date(2026, 12, 6)]
    assert not any(is_holdout_day("japan", d) for d in dev_days)
    assert all(is_holdout_day("japan", d) for d in held_days)  # the committed A2 definition
    rng = np.random.default_rng(9)
    races = [
        make_race(rng, f"d{k}", day, f"m{k}", 2, n=7, fav_odds=3.3, less_popular=True)
        for k, day in enumerate(dev_days)
    ]
    held = [
        _guarded(make_race(rng, f"h{k}", day, f"hm{k}", 2, n=7, fav_odds=3.3, less_popular=True))
        for k, day in enumerate(held_days)
    ]
    rows = la.score_races([*races, *held], holdout_day=is_holdout_day)
    assert set(rows["race_id"]) == {"d0", "d1"}


# ---------------------------------------------------------------------------------------------
# 6. The paired metric and its meeting-cluster bootstrap, on hand-built per-race rows
# ---------------------------------------------------------------------------------------------


def _hand_rows():
    """Two meetings, one race each of a fixed table: values checked by hand."""
    rows = []
    for race, meeting, n_all in (("r1", "m1", 4), ("r2", "m2", 5)):
        for cut in CUTS:
            n_skipped = {1.1: 3, 1.2: 2, 1.3: 1, 1.5: 0}[cut] if race == "r1" else 1
            rows.append({
                "race_id": race, "meeting_id": meeting, "cut": cut, "headline": race == "r1",
                "n_all": n_all, "n_kept": n_all - n_skipped, "n_skipped": n_skipped,
                "ret_all": 800.0 if race == "r1" else 0.0,
                "ret_kept": 800.0 if race == "r1" and cut != 1.1 else 0.0,
                "ret_skipped": 800.0 if race == "r1" and cut == 1.1 else 0.0,
            })  # fmt: skip
    return pd.DataFrame(rows)


def test_the_member_table_has_both_sets_for_each_of_the_8_members_with_the_hand_values():
    table = la.member_table(_hand_rows(), n_boot=200, seed=1)
    assert len(table) == 16
    assert set(zip(table["cut"], table["scope"], table["set"], strict=True)) == {
        (k, s, e) for k in CUTS for s in SCOPES for e in ("kept", "skipped")
    }

    def row(cut, scope, name):
        (r,) = [r for r in table.to_dict("records")
                if (r["cut"], r["scope"], r["set"]) == (cut, scope, name)]  # fmt: skip
        return r

    # all races, cut 1.1: all = 800 / (100 * 9); kept = 0 / (100 * 5); skipped = 800 / (100 * 4)
    r = row(1.1, "all", "kept")
    assert (r["n_races"], r["n_meetings"], r["stake"], r["ret"]) == (2, 2, 500, 0)
    assert r["all_per_yen"] == pytest.approx(800 / 900)
    assert r["per_yen"] == pytest.approx(0.0)
    assert r["diff"] == pytest.approx(-800 / 900)
    r = row(1.1, "all", "skipped")
    assert (r["stake"], r["ret"]) == (400, 800) and r["per_yen"] == pytest.approx(800 / 400)
    assert r["diff"] == pytest.approx(800 / 400 - 800 / 900)
    # cut 1.5: nothing of r1 skipped, one combination of r2; kept = 800 / (100 * (4 + 4))
    r = row(1.5, "all", "kept")
    assert (r["stake"], r["ret"]) == (800, 800) and r["per_yen"] == pytest.approx(1.0)
    assert r["diff"] == pytest.approx(1.0 - 800 / 900)
    # the headline scope is r1 alone: one race and one meeting, and the comparator is r1's own
    r = row(1.2, "headline", "kept")
    assert (r["n_races"], r["n_meetings"], r["stake"], r["ret"]) == (1, 1, 200, 800)
    assert r["all_per_yen"] == pytest.approx(800 / 400)
    assert r["diff"] == pytest.approx(800 / 200 - 800 / 400)
    # a set with no stake has no return per yen, and says so rather than dropping the row
    r = row(1.5, "headline", "skipped")
    assert r["stake"] == 0 and math.isnan(r["per_yen"]) and math.isnan(r["diff"])
    assert r["n_races"] == 1
    # the comparator is the same for every cut within a scope
    for scope in SCOPES:
        values = {round(x, 12) for x in table[table["scope"] == scope]["all_per_yen"]}
        assert len(values) == 1


def _meeting_effect_rows(seed=0, n_meetings=40, per=12):
    """Races with a strong meeting-level price effect, so meeting and race resampling differ."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_meetings):
        scale = float(np.exp(rng.normal(0.0, 0.9)))
        lean = float(rng.choice([0.03, 0.97]))  # P(the winning combination is the lowest-r one)
        for k in range(per):
            n = int(rng.integers(5, 10))
            r = rng.choice(TARGETS, size=n)
            win = int(np.argmin(r) if rng.random() < lean else np.argmax(r))
            hit = rng.random() < 0.8
            ret_all = 100.0 * scale * float(rng.integers(2, 40)) if hit else 0.0
            for cut in CUTS:
                skipped = r > cut
                rows.append({
                    "race_id": f"{g}:{k}", "meeting_id": f"m{g:03d}", "cut": cut,
                    "headline": bool(k % 2), "n_all": n, "n_kept": int((~skipped).sum()),
                    "n_skipped": int(skipped.sum()), "ret_all": ret_all,
                    "ret_kept": 0.0 if skipped[win] else ret_all,
                    "ret_skipped": ret_all if skipped[win] else 0.0,
                })  # fmt: skip
    return pd.DataFrame(rows)


def _linear_se(sub: pd.DataFrame, name: str, by: str = "meeting_id") -> float:
    """Linearised SE of per_yen(set) - per_yen(all) when whole ``by`` units are resampled."""
    s_set, r_set = 100 * sub[f"n_{name}"].sum(), sub[f"ret_{name}"].sum()
    s_all, r_all = 100 * sub["n_all"].sum(), sub["ret_all"].sum()
    rho_set, rho_all = r_set / s_set, r_all / s_all
    total = 0.0
    for _, g in sub.groupby(by):
        d = (g[f"ret_{name}"].sum() - rho_set * 100 * g[f"n_{name}"].sum()) / s_set - (
            g["ret_all"].sum() - rho_all * 100 * g["n_all"].sum()
        ) / s_all
        total += d * d
    return math.sqrt(total)


def test_the_bootstrap_resamples_whole_meetings_and_gives_one_sided_bounds():
    rows = _meeting_effect_rows()
    table = la.member_table(rows, n_boot=1500, seed=282)
    checked = 0
    for r in table.to_dict("records"):
        if r["set"] != "kept" or r["cut"] not in (1.2, 1.5):
            continue
        sub = rows[rows["cut"] == r["cut"]]
        sub = sub[sub["headline"]] if r["scope"] == "headline" else sub
        assert r["n_meetings"] == sub["meeting_id"].nunique() == 40
        se = _linear_se(sub, "kept")
        race_level = _linear_se(sub, "kept", by="race_id")  # what resampling races would give
        assert se > 1.3 * race_level  # the data separate the two: the meeting effect is real
        assert r["se"] == pytest.approx(se, rel=0.25), (r["cut"], r["scope"])
        assert r["lo"] < r["diff"] < r["hi"]
        assert 0.5 * se < r["diff"] - r["lo"] < 3.0 * se
        assert 0.5 * se < r["hi"] - r["diff"] < 3.0 * se
        checked += 1
    assert checked == 4


def test_the_max_statistic_widens_every_bound_and_the_seed_makes_it_reproducible():
    rows = _meeting_effect_rows(seed=4, n_meetings=30)
    a = la.member_table(rows, n_boot=400, seed=5)
    b = la.member_table(rows, n_boot=400, seed=5)
    c = la.member_table(rows, n_boot=400, seed=6)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["se"], c["se"])
    kept = a[(a["set"] == "kept") & a["diff"].notna()]
    assert len(kept) == 8
    assert (kept["adj_lo"] <= kept["lo"] + 1e-12).all()
    assert (kept["adj_hi"] >= kept["hi"] - 1e-12).all()
    assert (kept["adj_lo"] < kept["lo"] - 1e-9).any()  # over 8 members it costs something
    assert np.isfinite(kept[["adj_lo", "adj_hi"]]).all().all()
    assert ((kept["diff"] - kept["adj_lo"]) < 6 * kept["se"]).all()


# ---------------------------------------------------------------------------------------------
# 7. score(): the freeze guard, the records, any number of meetings
# ---------------------------------------------------------------------------------------------


def _meta(lambda2=1.0):
    return {"lambda2_full_development": lambda2, "ordering_fit": "ordfitA0001",
            "data_builds": {"odds_log": "0123456789abcdef"}}  # fmt: skip


def _season(n_days=4, venues=("TOK", "HSN"), per=4, seed=2026, lambda2=1.0):
    """n_days x venues meetings of ``per`` races: every fav-odds band and both popularity flags."""
    rng = np.random.default_rng(seed)
    races, start = [], date(2025, 4, 5)
    for d in range(n_days):
        day = start + timedelta(days=7 * d)
        for venue in venues:
            for k in range(per):
                races.append(make_race(
                    rng, f"jra:{day.isoformat()}:{venue}:{k + 1:02d}", day,
                    f"jra:{day.isoformat()}:{venue}", k + 1, n=int(rng.integers(6, 12)),
                    fav_odds=float(rng.choice([1.8, 2.6, 3.1, 4.4, 7.2])),
                    less_popular=bool(rng.random() < 0.5), lambda2=lambda2,
                ))  # fmt: skip
    return races


def _scored(tmp_path, races, *, lambda2=1.0, n_boot=300, holdout=lambda c, d: False):
    reg = _registry(tmp_path)
    la.freeze(reg)
    result = la.score(reg, copy.deepcopy(races), _meta(lambda2), SHA, holdout_day=holdout,
                      n_boot=n_boot)  # fmt: skip
    return reg, result


def test_scoring_refuses_an_unfrozen_family_and_writes_nothing(tmp_path):
    reg = _registry(tmp_path)
    with pytest.raises(exacta_run.RunRefused, match=re.escape(la.FAMILY_ID)):
        la.score(reg, _season(), _meta(), SHA, holdout_day=lambda c, d: False, n_boot=50)
    assert reg.records() == []


def test_scoring_refuses_a_family_frozen_with_another_spec_and_writes_nothing(
    tmp_path, monkeypatch
):
    spec = la.family_specs()[la.FAMILY_ID]
    other = copy.deepcopy(spec)
    other["cuts"] = [1.1, 1.2, 1.3, 1.6]  # any change to the design is another spec
    reg = _registry(tmp_path)
    reg.freeze_family(la.FAMILY_ID, other)
    with pytest.raises(exacta_run.RunRefused, match=re.escape(la.FAMILY_ID)):
        la.score(reg, _season(), _meta(), SHA, holdout_day=lambda c, d: False, n_boot=50)
    assert reg.records() == []
    (tmp_path / "second").mkdir()
    reg2 = _registry(tmp_path / "second")
    reg2.freeze_family(la.FAMILY_ID, spec)  # frozen as in use, then the code's spec drifts
    monkeypatch.setattr(la, "family_specs", lambda: {la.FAMILY_ID: other})
    with pytest.raises(exacta_run.RunRefused, match=re.escape(la.FAMILY_ID)):
        la.score(reg2, _season(), _meta(), SHA, holdout_day=lambda c, d: False, n_boot=50)
    assert reg2.records() == []


def test_a_frozen_family_scores_the_8_members_against_the_reference(tmp_path):
    races = _season(n_days=4)
    reg, result = _scored(tmp_path, races)
    table = result["members"]
    ref = ref_members(ref_rows(races))
    assert len(table) == 16
    for r in table.to_dict("records"):
        n_races, n_meetings, stake, ret, per, all_per, diff = ref[(r["cut"], r["scope"], r["set"])]
        assert (r["n_races"], r["n_meetings"], r["stake"], r["ret"]) == (n_races, n_meetings,
                                                                          stake, ret)  # fmt: skip
        assert r["all_per_yen"] == pytest.approx(all_per)
        for got, want in ((r["per_yen"], per), (r["diff"], diff)):
            assert (math.isnan(got) and math.isnan(want)) or got == pytest.approx(want)
        if not math.isnan(diff):
            assert r["lo"] - 1e-9 <= r["diff"] <= r["hi"] + 1e-9
    assert result["meta"]["scored_races"] == len(races)
    assert result["meta"]["scored_meetings"] == 8
    assert result["meta"]["excluded"] == {}


def test_each_member_is_a_registry_record_with_its_kept_and_skipped_metrics(tmp_path):
    races = _season(n_days=3)
    reg, result = _scored(tmp_path, races)
    records = reg.records(la.FAMILY_ID)
    assert len(records) == 8 == reg.family_size(la.FAMILY_ID)
    assert {(r.params["cut"], r.params["scope"]) for r in records} == {
        (k, s) for k in CUTS for s in SCOPES
    }
    assert len({r.variant for r in records}) == 8
    table = result["members"].set_index(["cut", "scope", "set"])
    for r in records:
        assert (r.ticket, r.sources, r.status) == (282, ("jra",), "scored")
        assert r.components.get("ordering") == "ordfitA0001"
        assert r.data_builds == _meta()["data_builds"]
        metrics = {m.metric: m for m in r.metrics}
        kept = table.loc[(r.params["cut"], r.params["scope"], "kept")]
        m = metrics["kept_minus_all"]
        assert m.value == pytest.approx(kept["diff"])
        assert (m.lo, m.hi, m.n) == (pytest.approx(kept["lo"]), pytest.approx(kept["hi"]),
                                     int(kept["n_races"]))  # fmt: skip
        skipped = table.loc[(r.params["cut"], r.params["scope"], "skipped")]
        if skipped["stake"] > 0:
            assert metrics["skipped_minus_all"].value == pytest.approx(skipped["diff"])
        else:
            assert "skipped_minus_all" not in metrics


def test_a_rerun_on_the_same_inputs_appends_nothing(tmp_path):
    races = _season(n_days=2)
    reg, _ = _scored(tmp_path, races)
    before = [json.dumps(r.__dict__, default=str, sort_keys=True) for r in reg.records()]
    la.score(reg, copy.deepcopy(races), _meta(), SHA, holdout_day=lambda c, d: False, n_boot=300)
    after = [json.dumps(r.__dict__, default=str, sort_keys=True) for r in reg.records()]
    assert after == before and len(before) == 8


@pytest.mark.parametrize("n_days", [1, 3])
def test_fewer_than_100_meetings_still_scores_with_n_and_a_confidence_interval(tmp_path, n_days):
    races = _season(n_days=n_days, venues=("TOK", "HSN", "FKS"))
    reg, result = _scored(tmp_path, races)
    assert result["meta"]["scored_meetings"] == 3 * n_days < 100
    assert len(reg.records(la.FAMILY_ID)) == 8
    kept = result["members"][result["members"]["set"] == "kept"]
    assert (kept[kept["scope"] == "all"]["n_races"] > 0).all()
    scored = kept[kept["diff"].notna()]
    assert len(scored) >= 4
    assert np.isfinite(scored[["diff", "se", "lo", "hi"]]).all().all()
    assert (scored["lo"] - 1e-9 <= scored["diff"]).all()
    assert (scored["diff"] <= scored["hi"] + 1e-9).all()
    assert (scored["n_meetings"] <= 3 * n_days).all() and (scored["n_meetings"] >= 1).all()


def test_the_run_takes_the_ordering_lambda2_of_the_fit_and_scores_holdout_free(tmp_path):
    lam = 0.8
    races = _season(n_days=3, lambda2=lam, seed=77)
    held = _guarded(make_race(np.random.default_rng(8), "h", date(2026, 11, 29), "hm", 3, n=7,
                              fav_odds=3.4, less_popular=True))  # fmt: skip
    reg = _registry(tmp_path)
    la.freeze(reg)
    result = la.score(reg, [*copy.deepcopy(races), held], _meta(lam), SHA,
                      holdout_day=is_holdout_day, n_boot=200)  # fmt: skip
    ref = ref_members(ref_rows(races, lambda2=lam))
    assert result["meta"]["scored_races"] == len(races)
    assert result["meta"]["excluded"] == {"holdout_day": 1}
    for r in result["members"].to_dict("records"):
        n_races, _, stake, ret, _, _, _ = ref[(r["cut"], r["scope"], r["set"])]
        assert (r["n_races"], r["stake"], r["ret"]) == (n_races, stake, ret)
