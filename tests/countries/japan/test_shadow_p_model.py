"""Model probabilities in the shadow feed (ticket #506): `p_model` per race, pool and combination.

Contract pinned here (`horsetrack.countries.japan.board.shadow_feed.build_shadow`, unchanged
signature):
- Every race of the shadow feed gains `p_model`: {pool: {combination: probability}} for WIN,
  PLACE, QUINELLA, QUINELLA_PLACE, FORECAST, TRIO, TIERCE and, when the race drew 9 horses or
  more, BRACKET_QUINELLA, keyed by the odds log's combination texts (so they join the plain
  feed's rows). WIN is the calibrated Win model on the board as of now: q = (1/O_WIN) /
  sum(1/O_WIN) over the board's runners, p = models.win.predict(q, board.context()). The other
  pools are `model_probabilities(p, models.ordering, brackets=..., starters=...)`
  (models/v0/shadow_values.py): exact sums over the discounted Harville model's ordered triples.
- Scratched runners (a current SCR WIN or PLACE row) are off the board: they have no key and the
  others' probabilities are renormalised, as the shadow values do. The places Place and Wide pay
  (and whether they are sold) are those of the runners at the start of sales, which the board
  knows as its priced runners plus its scratched ones (the logs do not say when sales opened, so a
  scratch is taken as after): 8 drawn and one out, Place pays three places over the seven left;
  5 drawn and one out, Place is still sold, with two places.
- Brackets follow JRA's draw of N horses, N the largest horse number on the board, scratched
  runners included (the draw was made before the scratch): N <= 8, no bracket quinella; 9-16,
  brackets 1..16-N take one horse each and the rest two, in number order; 17, brackets 1-7 two
  and bracket 8 three; 18, brackets 1-6 two, 7 and 8 three (finish_orders.BRACKET_SIZES).
- Without an ordering fit only WIN has probabilities; without models `p_model` is null (or {}).
- Unchanged: each race's other keys and its `pools` (WIN, PLACE and Wide entries with exactly
  their value fields, flags and ladder, the WIN values as before): p_model is display only and
  never reaches the shadow log (its records come from shadow_values; shadow/test_day.py pins them).
- A rebuild through the cache after a new capture carries the new board's probabilities.

Models: FakeWin (p = q^1.3 / sum q^1.3) and FakeNowcast from shadow_fakes.py, with a real
OrderingFit for the discounts. Expected probabilities are brute-force enumerations
(finish_orders.py), never the code under test. All rows are synthetic, written under tmp_path.
"""

from __future__ import annotations

import json
import math
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from horsetrack.core import paths
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board.flags import FlagRules
from horsetrack.countries.japan.board.shadow_feed import build_shadow
from horsetrack.countries.japan.models.v0.ordering import OrderingFit
from horsetrack.countries.japan.models.v0.shadow_values import ShadowModels

# isort: split
from finish_orders import jra_brackets, pool_probabilities, pool_totals
from shadow_fakes import (
    BANNED_KEY_WORDS,
    IDS,
    PLACE,
    WIDE,
    WIN,
    FakeNowcast,
    FakeWin,
    Race,
    SyntheticDay,
    all_keys,
    capture_rows,
    expected_values,
    figures,
    log_path,
    win_p,
)

UTC = timezone.utc
DAY = date(2026, 11, 21)
T0 = datetime(2026, 11, 21, 2, 0, tzinfo=UTC)  # 11:00 in Tokyo
RULES = FlagRules(floor_max_odds=1.3, tail_min_odds=50.0, tail_rank=6, stale_s=40.0)
N_DRAWS = 8
LAMBDAS = (0.85, 0.7)
BRACKET = "BRACKET_QUINELLA"
GRID = (60, 30, 16, 5, 2, 1)  # the shipped decision grid
RACE_KEYS = {
    "race_id",
    "venue",
    "race_no",
    "post_utc",
    "minutes_to_post",
    "tau_bucket_min",
    "board_capture_utc",
    "pools",
}
ENTRY_KEYS = {
    WIN: {"value_mean", "value_q05", "value_q95", "flags", "ladder"},
    PLACE: {"value_at_low", "value_at_high", "flags", "ladder"},
    WIDE: {"value_at_low", "value_at_high", "flags", "ladder"},
}


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


def field(n: int, seed: int) -> tuple[float, ...]:
    """WIN odds of horses 1..n, shuffled so the favourite is not horse 1."""
    rng = random.Random(seed)
    odds = [round(1.5 + 1.9 * i**1.3 + rng.random(), 1) for i in range(n)]
    rng.shuffle(odds)
    return tuple(odds)


def race(venue: str, no: int, minutes: float, base: tuple[float, ...]) -> Race:
    return Race(DAY, venue, no, T0 + timedelta(minutes=minutes), base)


def models(lambdas=LAMBDAS, *, ordering: bool = True) -> ShadowModels:
    fit = OrderingFit("discounted_harville", lambdas, IDS["ordering"], 500) if ordering else None
    return ShadowModels(win=FakeWin(), ordering=fit, nowcast=FakeNowcast(), ids=dict(IDS))


def build(now: datetime, shadow_models, **options) -> dict:
    settings = dict(now=lambda: now, models=shadow_models, flag_rules=RULES, n_draws=N_DRAWS)
    settings.update(options)
    return build_shadow(DAY, seed=3, **settings)


def by_id(feed: dict) -> dict[str, dict]:
    return {r["race_id"]: r for r in feed["races"]}


def bucket(minutes: float) -> int:
    if minutes <= 0:
        return 0
    above = [m for m in GRID if m >= minutes]
    return min(above) if above else max(GRID)


def capture_with_scratches(day: SyntheticDay, r: Race, at: datetime, factor: float, out) -> None:
    """A capture where the runners `out` show SCR on WIN and PLACE and their Wide pairs are no
    longer listed (the source stops listing them)."""
    rows = []
    for row in capture_rows(r, at, factor):
        runners = {int(part) for part in row["combination"].split(",")}
        if not runners & set(out):
            rows.append(row)
        elif row["pool"] in (WIN, PLACE):
            scratched = {k: v for k, v in row.items() if k != "odds_hi"}
            rows.append({**scratched, "odds": None, "status_raw": "SCR"})
    odds_log.append_capture(log_path(day.day, r.venue), rows)
    day.rows.extend(rows)


def expected(r: Race, factor: float, lambdas, out=()) -> dict[str, dict[str, float]]:
    """p_model of the capture of `r` at `factor`, the runners `out` scratched: FakeWin on the
    board's q, then every pool enumerated over the runners left, the places paid for every horse
    drawn (the start of sales); brackets from the draw of max(horse number)."""
    win, _, _ = figures(r, factor)
    running = {h: o for h, o in win.items() if h not in out}
    horses = sorted(running)
    inv = np.array([1.0 / running[h] for h in horses])
    p = dict(zip(horses, (float(x) for x in win_p(inv / inv.sum())), strict=True))
    drawn = max(win)
    brackets = jra_brackets(drawn) if drawn >= 9 else None
    return pool_probabilities(p, lambdas, brackets, starters=len(win))


def check_p_model(got: dict, want: dict[str, dict[str, float]], where, starters=None) -> None:
    p_model = got.get("p_model")
    assert isinstance(p_model, dict), (where, "the race has no p_model mapping", sorted(got))
    shown = {pool for pool, entries in p_model.items() if entries}
    assert shown == set(want), (where, sorted(shown), sorted(want))
    for pool, entries in want.items():
        assert set(p_model[pool]) == set(entries), (
            where,
            pool,
            sorted(set(p_model[pool]) ^ set(entries))[:8],
        )
        for combination, prob in entries.items():
            value = p_model[pool][combination]
            assert math.isclose(value, prob, rel_tol=1e-9, abs_tol=1e-12), (
                where,
                pool,
                combination,
                value,
                prob,
            )
    starters = len(want["WIN"]) if starters is None else starters
    for pool, total in pool_totals(starters, BRACKET in want).items():
        assert math.isclose(math.fsum(p_model[pool].values()), total, rel_tol=1e-9), (where, pool)


def check_unchanged(got: dict, r: Race, factor: float, tau: int, out=()) -> None:
    """The race's keys and its pools as before #506: WIN, PLACE and Wide entries with exactly their
    old fields, one per runner and pair on the board, WIN valued as shadow_fakes works it out."""
    where = r.race_id
    assert set(got) == RACE_KEYS | {"p_model"}, (where, sorted(got))
    assert set(got["pools"]) == {WIN, PLACE, WIDE}, (where, sorted(got["pools"]))
    win, _, wide = figures(r, factor)
    horses = {str(h) for h in win if h not in out}
    assert set(got["pools"][WIN]) == horses and set(got["pools"][PLACE]) == horses, where
    pairs = {f"{a},{b}" for a, b in wide if a not in out and b not in out}
    assert set(got["pools"][WIDE]) == pairs, where
    for pool, entries in got["pools"].items():
        for combination, entry in entries.items():
            assert set(entry) == ENTRY_KEYS[pool], (where, pool, combination, sorted(entry))
            if pool != WIN:
                assert entry["value_at_low"] is not None, (where, pool, combination)
    if out:
        return
    for combination, values in expected_values(r, factor, tau)[WIN].items():
        for name, value in values.items():
            got_value = got["pools"][WIN][combination][name]
            assert math.isclose(got_value, value, rel_tol=1e-9), (where, combination, name)


# ---------------------------------------------------------------- every race, every pool


@pytest.mark.parametrize("lambdas", [(0.85, 0.7), (1.0, 1.0), (1.3, 0.5)], ids=str)
def test_every_race_has_model_probabilities_for_every_pool(lambdas):
    races = {  # race: minutes to post
        race("TOK", 1, 3.5, field(8, 1)): 3.5,
        race("TOK", 2, 25.0, field(6, 2)): 25.0,
        race("TOK", 3, 70.0, field(10, 3)): 70.0,
        race("KYO", 1, 12.0, field(18, 4)): 12.0,
    }
    day = SyntheticDay(DAY, list(races))
    for r in races:
        earlier = Race(r.day, r.venue, r.no, r.post, tuple(reversed(r.base)))  # another order
        day.capture(earlier, T0 - timedelta(minutes=10), 1.0)
        day.capture(r, T0 - timedelta(seconds=20), 1.03)
        later = Race(r.day, r.venue, r.no, r.post, field(len(r.base), 99))
        day.capture(later, T0 + timedelta(seconds=30), 0.97)  # after now(): not on the board

    feed = build(T0, models(lambdas))

    got = by_id(feed)
    assert set(got) == {r.race_id for r in races}
    for r, minutes in races.items():
        want = expected(r, 1.03, lambdas)
        assert (BRACKET in want) == (len(r.base) >= 9), r.race_id  # the fixture covers both
        check_p_model(got[r.race_id], want, (r.race_id, lambdas))
        check_unchanged(got[r.race_id], r, 1.03, bucket(minutes))
    # JSON-ready: text keys and finite floats (a NaN would not compare equal), no stake words
    for r in got.values():
        assert json.loads(json.dumps(r["p_model"], allow_nan=False)) == r["p_model"], r["race_id"]
    named = sorted({k for k in all_keys(feed) for w in BANNED_KEY_WORDS if w in k.lower()})
    assert named == [], named


# ---------------------------------------------------------------- brackets


DRAWN = (8, 9, 11, 13, 16, 17)


def test_bracket_quinella_follows_the_jra_draw():
    races = [race("NAK", no, 40.0, field(n, 10 + n)) for no, n in enumerate(DRAWN, start=1)]
    day = SyntheticDay(DAY, races)
    for r in races:
        day.capture(r, T0 - timedelta(seconds=15), 1.0)

    got = by_id(build(T0, models((0.9, 0.6))))

    for r in races:
        want = expected(r, 1.0, (0.9, 0.6))
        p_model = got[r.race_id].get("p_model")
        assert isinstance(p_model, dict), r.race_id
        if len(r.base) <= 8:
            assert not p_model.get(BRACKET), "bracket quinella is not sold with 8 horses or fewer"
            continue
        assert set(p_model.get(BRACKET) or {}) == set(want[BRACKET]), (len(r.base), r.race_id)
        for combination, prob in want[BRACKET].items():
            assert math.isclose(p_model[BRACKET][combination], prob, rel_tol=1e-9), (
                len(r.base),
                combination,
            )


# ---------------------------------------------------------------- scratches


SCRATCH_CASES = {  # (venue, race), horses drawn, runners scratched, places Place pays
    "TOK 5: 8 drawn, 7 run, Place still pays three": (("TOK", 5), 8, {3}, 3),
    "KYO 6: 10 drawn, a single-horse bracket and the last number out": (("KYO", 6), 10, {6, 10}, 3),
    "NAK 7: 18 drawn, the first number and one of bracket 8 out": (("NAK", 7), 18, {1, 17}, 3),
    "TOK 8: 5 drawn, 4 run, Place still sold with two places": (("TOK", 8), 5, {4}, 2),
}


def test_scratched_runners_have_no_probability_and_the_rest_renormalise():
    cases = []
    for (venue, no), n, out, places in SCRATCH_CASES.values():
        cases.append((race(venue, no, 14.0, field(n, 50 + n)), out, places))
    day = SyntheticDay(DAY, [r for r, _, _ in cases])
    for r, out, _ in cases:
        day.capture(r, T0 - timedelta(minutes=6), 1.0)  # before the scratch: all priced
        capture_with_scratches(day, r, T0 - timedelta(seconds=10), 1.02, out)

    got = by_id(build(T0, models()))

    for (r, out, places), label in zip(cases, SCRATCH_CASES, strict=True):
        drawn = len(r.base)
        want = expected(r, 1.02, LAMBDAS, out)  # every pool over the runners left
        assert len(want["WIN"]) == drawn - len(out), label
        check_p_model(got[r.race_id], want, label, starters=drawn)
        if drawn - len(out) >= 5:  # the shadow values keep their own places (not #506's)
            check_unchanged(got[r.race_id], r, 1.02, bucket(14.0), out)
        p_model = got[r.race_id]["p_model"]
        for pool, entries in p_model.items():
            if pool == BRACKET:
                continue
            for combination in entries:
                runners = {int(h) for h in combination.split(",")}
                assert not runners & out, (label, pool, combination)
        # the places of the start of sales, over the runners left
        assert set(p_model[PLACE]) == {str(h) for h in range(1, drawn + 1) if h not in out}
        total = math.fsum(p_model[PLACE].values())
        assert math.isclose(total, float(places), rel_tol=1e-9), (label, total)
        assert math.isclose(math.fsum(p_model[WIDE].values()), 3.0, rel_tol=1e-9), label
    # bracket 6 (horse 6 alone) is gone; bracket 8 (9, 10) keeps 9 only, so no 8-8
    brackets = set(got[cases[1][0].race_id]["p_model"][BRACKET])
    assert "7,7" in brackets and "8,8" not in brackets, sorted(brackets)
    assert not [c for c in brackets if "6" in c.split(",")], sorted(brackets)


# ---------------------------------------------------------------- partial models


def test_without_an_ordering_fit_only_win_has_probabilities():
    tok1, kyo2 = race("TOK", 1, 3.5, field(8, 21)), race("KYO", 2, 40.0, field(12, 22))
    day = SyntheticDay(DAY, [tok1, kyo2])
    for r in (tok1, kyo2):
        day.capture(r, T0 - timedelta(seconds=25), 1.0)

    got = by_id(build(T0, models(ordering=False)))

    for r in (tok1, kyo2):
        p_model = got[r.race_id].get("p_model")
        assert isinstance(p_model, dict), r.race_id
        assert {pool for pool, entries in p_model.items() if entries} == {WIN}, sorted(p_model)
        want = expected(r, 1.0, LAMBDAS)[WIN]
        assert set(p_model[WIN]) == set(want)
        for combination, prob in want.items():
            assert math.isclose(p_model[WIN][combination], prob, rel_tol=1e-9), combination


def test_without_models_there_are_no_probabilities():
    tok1, kyo2 = race("TOK", 1, 3.5, field(8, 31)), race("KYO", 2, 40.0, field(10, 32))
    day = SyntheticDay(DAY, [tok1, kyo2])
    for r in (tok1, kyo2):
        day.capture(r, T0 - timedelta(seconds=25), 1.0)

    bare = by_id(build(T0, None))
    with_models = by_id(build(T0, models()))

    for r in (tok1, kyo2):
        assert set(bare[r.race_id]) <= RACE_KEYS | {"p_model"}, sorted(bare[r.race_id])
        assert not bare[r.race_id].get("p_model"), r.race_id
        check_p_model(with_models[r.race_id], expected(r, 1.0, LAMBDAS), r.race_id)


# ---------------------------------------------------------------- the cache


def test_a_new_capture_brings_its_probabilities_through_the_cache():
    tok1, tok2 = race("TOK", 1, 10.5, field(9, 41)), race("TOK", 2, 40.0, field(8, 42))
    day = SyntheticDay(DAY, [tok1, tok2])
    for r in (tok1, tok2):
        day.capture(r, T0 - timedelta(seconds=5), 1.0)
    shadow_models = models()
    cache: dict = {}

    first = by_id(build(T0, shadow_models, cache=cache))
    for r in (tok1, tok2):
        check_p_model(first[r.race_id], expected(r, 1.0, LAMBDAS), ("first", r.race_id))

    # TOK 1's crowd reorders: a new capture with other odds; TOK 2 has no new capture
    moved = Race(tok1.day, tok1.venue, tok1.no, tok1.post, field(9, 43))
    day.capture(moved, T0 + timedelta(seconds=25), 1.0)
    second = by_id(build(T0 + timedelta(seconds=30), shadow_models, cache=cache))

    check_p_model(second[tok1.race_id], expected(moved, 1.0, LAMBDAS), "after the new capture")
    check_p_model(second[tok2.race_id], expected(tok2, 1.0, LAMBDAS), "unchanged race")
    # the fixture: the reordered crowd moves the trifecta probabilities well beyond the tolerance
    old, new = expected(tok1, 1.0, LAMBDAS)["TIERCE"], expected(moved, 1.0, LAMBDAS)["TIERCE"]
    assert max(abs(old[c] - new[c]) for c in old) > 1e-3
