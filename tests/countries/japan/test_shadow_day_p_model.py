"""Shadow-day records carry `p_model` and the exotic values (ticket #564, follow-up to #506).

Contract pinned here (`horsetrack.countries.japan.shadow.day.run_shadow_day`, unchanged signature;
`shadow.score.score_shadow`, see shadow/test_score.py):

- Every record shadow-day writes carries `record_version` 2 (`shadow.log.RECORD_VERSION`) and
  `p_model` (a probability, or null). The point's p_model is the shadow feed's (#506): WIN is
  `models.win.predict(q, board.context())` with q the decision board's 1/O_WIN normalised; every
  other pool is `model_probabilities(p, models.ordering, brackets=..., starters=...)`, with the
  brackets of JRA's draw of the largest horse number on the board (none at 8 or fewer) and, on a
  board without scratches, starters = its runners. An ordering model without the fit's discounts
  (not an `OrderingFit`) gives WIN only: PLACE and Wide records then carry a null p_model.
- WIN, PLACE and Wide records keep exactly their keys and fields of #120 (one per board key), with
  `p_model` = p_model[pool][combination].
- Exotic records (the pools with no price forecast): one record per combination of each exotic
  pool's decision-time capture (its latest rows available at or before the decision time) that has
  a p_model: every QUINELLA and BRACKET_QUINELLA combination, and for FORECAST, TRIO and TIERCE the
  `EXOTIC_TOP_N` = 15 with the largest p_model (all of them when fewer). Each has `p_model`,
  `board_odds` = that capture's odds, and `value_model` = p_model x board_odds (gross return per
  1 yen if the odds stood). A pool with no capture by the decision time writes no record.
- Without models, or without a p_model for a pool, no exotic record is written: the records are
  the #120 ones, each with record_version 2 and a null p_model, and the log verifies.
- A scoring pass over the day's records and synthetic final dividends gives, per (pool, tau),
  100 x mean(p_model x board_odds) and the dividends paid / records scored.

Models: FakeWin (p = q^1.3 / sum q^1.3) and FakeNowcast from shadow_fakes.py with a real
OrderingFit for the discounts. Expected probabilities are brute-force enumerations
(finish_orders.py), never the code under test. Every row is synthetic, under tmp_path.
"""

# Ruff reads shadow.score as third-party until it exists, then as first-party; the block is
# written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math
from datetime import date, datetime, timedelta, timezone
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horsetrack.core import paths
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board.flags import FlagRules
from horsetrack.countries.japan.models.v0.ordering import OrderingFit
from horsetrack.countries.japan.models.v0.shadow_values import ShadowModels
from horsetrack.countries.japan.shadow import log as shadow_log
from horsetrack.countries.japan.shadow.day import run_shadow_day
from horsetrack.countries.japan.shadow.log import ShadowLog, shadow_log_path

# isort: split
from finish_orders import jra_brackets, pool_probabilities
from shadow_fakes import (
    BANNED_KEY_WORDS,
    IDS,
    PLACE,
    WIDE,
    WIN,
    FakeNowcast,
    FakeOrdering,
    FakeWin,
    Race,
    all_keys,
    capture_rows,
    figures,
    instant,
    log_path,
    win_p,
)

UTC = timezone.utc
DAY = date(2026, 11, 22)  # a development day
GRID = (60, 30, 16, 5, 2, 1)  # D2, the shipped decision.yaml
POST1 = datetime(2026, 11, 22, 1, 0, tzinfo=UTC)
POST2 = datetime(2026, 11, 22, 1, 40, tzinfo=UTC)
LATE = datetime(2026, 11, 22, 3, 0, tzinfo=UTC)  # 12:00 in Tokyo, after every race
RULES = FlagRules(floor_max_odds=1.3, tail_min_odds=50.0, tail_rank=6, stale_s=40.0)
N_DRAWS = 8
TOP_N = 15
QUINELLA, FORECAST, TRIO, TIERCE = "QUINELLA", "FORECAST", "TRIO", "TIERCE"
BRACKET = "BRACKET_QUINELLA"
EXOTIC = (QUINELLA, FORECAST, TRIO, TIERCE, BRACKET)
TOPPED = (FORECAST, TRIO, TIERCE)  # stored as their 15 most likely combinations
TRAP = 7.0  # a capture 5 s after each decision time, priced seven times higher

R1 = Race(DAY, "TOK", 1, POST1, (2.4, 5.1, 3.3, 14.0, 8.2, 31.0, 6.6, 52.0, 11.5, 19.0))
R2 = Race(DAY, "TOK", 2, POST2, (4.4, 2.1, 9.7, 6.0, 15.5, 3.8, 27.0))
TIERCE_LOGGED = {R1.race_id}  # race 2's Trifecta was never captured


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    monkeypatch.setattr(shadow_log, "is_holdout_day", lambda country, day: False)
    return d


# ---------------------------------------------------------------- the synthetic day


def factor(tau: int) -> float:
    return 1.0 + 0.01 * tau


def brackets_of(race: Race) -> dict[int, int] | None:
    n = len(race.base)
    return jra_brackets(n) if n >= 9 else None


def exotic_odds(race: Race, f: float) -> dict[str, dict[str, float]]:
    """{pool: {combination: odds}} of one capture: invented, positive, order-dependent for the
    ordered pools, every combination of each pool the race has."""
    w = dict(enumerate(race.base, 1))
    horses = sorted(w)
    out: dict[str, dict[str, float]] = {
        QUINELLA: {
            f"{a},{b}": round(f * (1.3 + 0.55 * w[a] * w[b] ** 0.8), 1)
            for a, b in combinations(horses, 2)
        },
        FORECAST: {
            f"{a},{b}": round(f * (1.6 + 0.9 * w[a] * w[b] ** 0.7 + 0.3 * a), 1)
            for a, b in permutations(horses, 2)
        },
        TRIO: {
            f"{a},{b},{c}": round(f * (2.0 + 0.4 * (w[a] * w[b] * w[c]) ** 0.75), 1)
            for a, b, c in combinations(horses, 3)
        },
    }
    if race.race_id in TIERCE_LOGGED:
        out[TIERCE] = {
            f"{a},{b},{c}": round(f * (3.0 + 1.1 * w[a] * (w[b] * w[c]) ** 0.7 + 0.2 * b), 1)
            for a, b, c in permutations(horses, 3)
        }
    brackets = brackets_of(race)
    if brackets is not None:
        pairs = {tuple(sorted((brackets[a], brackets[b]))) for a, b in permutations(horses, 2)}
        out[BRACKET] = {f"{x},{y}": round(f * (1.8 + 2.3 * x + 1.7 * y), 1) for x, y in pairs}
    return out


def exotic_rows(race: Race, at: datetime, f: float) -> list[dict]:
    rows = []
    for pool, entries in exotic_odds(race, f).items():
        for combination, odds in entries.items():
            rows.append(
                {
                    "snapshot_utc": at.isoformat(),
                    "fetched_utc": at.isoformat(),
                    "race_id": race.race_id,
                    "pool": pool,
                    "combination": combination,
                    "odds": odds,
                    "status_raw": "10時00分現在",
                    "page_state": None,
                    "source_update_utc": (at - timedelta(seconds=10)).isoformat(),
                }
            )
    return rows


def write_day(races: list[Race]) -> None:
    """Per race and grid point, a capture 40 s before the decision time at factor(tau) (the board
    the point must use) and a trap capture 5 s after it; then the post-time sidecar."""
    members = []
    for race in races:
        for tau in GRID:
            decision = race.post - timedelta(minutes=tau)
            for at, f in (
                (decision - timedelta(seconds=40), factor(tau)),
                (decision + timedelta(seconds=5), TRAP),
            ):
                members.append((at, race, capture_rows(race, at, f) + exotic_rows(race, at, f)))
    for _at, race, rows in sorted(members, key=lambda m: (m[0], m[1].no)):
        odds_log.append_capture(log_path(DAY, race.venue), rows)
    posts = {r.no: r.post for r in races}
    sidecar = log_path(DAY, "TOK").with_name("TOK.races.json")
    odds_log.write_post_times(sidecar, posts, POST1 - timedelta(hours=4))


def models(lambdas, *, discounts: bool = True) -> ShadowModels:
    ordering = (
        OrderingFit("discounted_harville", lambdas, IDS["ordering"], 400)
        if discounts
        else FakeOrdering()
    )
    return ShadowModels(win=FakeWin(), ordering=ordering, nowcast=FakeNowcast(), ids=dict(IDS))


def run_day(shadow_models) -> list[dict]:
    write_day([R1, R2])
    log = ShadowLog(shadow_log_path(DAY), day=DAY)
    summary = run_shadow_day(
        DAY,
        venues=None,
        now=lambda: LATE,
        sleep=lambda s: None,
        poll_s=10.0,
        grace_s=15.0,
        max_board_age_s=120.0,
        max_lag_s=120.0,
        once=True,
        models=shadow_models,
        log=log,
        flag_rules=RULES,
        n_draws=N_DRAWS,
        seed=11,
    )
    records = ShadowLog(shadow_log_path(DAY), day=DAY).read()
    verdict = ShadowLog(shadow_log_path(DAY), day=DAY).verify()
    assert summary.sealed and verdict.sealed
    assert summary.records == len(records) == verdict.n_records
    assert summary.no_board == verdict.no_board == 0
    return records


# ---------------------------------------------------------------- the reference


def reference_p(race: Race, tau: int, lambdas) -> dict[str, dict[str, float]]:
    """The point's p_model: FakeWin on the decision board's WIN odds, every pool enumerated."""
    win, _, _ = figures(race, factor(tau))
    horses = sorted(win)
    inv = np.array([1.0 / win[h] for h in horses])
    p = dict(zip(horses, (float(x) for x in win_p(inv / inv.sum())), strict=True))
    return pool_probabilities(p, lambdas, brackets_of(race), starters=len(horses))


def board_figure(race: Race, tau: int, pool: str, combination: str) -> float:
    """The decision-time odds of a key (the low figure for PLACE and Wide)."""
    win, place, wide = figures(race, factor(tau))
    if pool == WIN:
        return win[int(combination)]
    if pool == PLACE:
        return place[int(combination)][0]
    if pool == WIDE:
        a, b = (int(x) for x in combination.split(","))
        return wide[(a, b)][0]
    return exotic_odds(race, factor(tau))[pool][combination]


def stored_exotics(race: Race, tau: int, ref: dict) -> dict[str, set[str]]:
    """The exotic combinations the rule stores: every Quinella and Bracket quinella one, the 15
    most likely of Exacta, Trio and Trifecta; only pools captured by the decision time."""
    captured = exotic_odds(race, factor(tau))
    out = {}
    for pool in EXOTIC:
        if pool not in captured or pool not in ref:
            continue
        probs = {c: ref[pool][c] for c in captured[pool]}
        if pool in TOPPED:
            out[pool] = set(sorted(probs, key=probs.get, reverse=True)[:TOP_N])
        else:
            out[pool] = set(probs)
    return out


def base_keys(race: Race) -> set[tuple[str, str]]:
    horses = range(1, len(race.base) + 1)
    return (
        {(WIN, str(h)) for h in horses}
        | {(PLACE, str(h)) for h in horses}
        | {(WIDE, f"{a},{b}") for a, b in combinations(horses, 2)}
    )


def by_point(records: list[dict]) -> dict[tuple[str, int], dict[tuple[str, str], dict]]:
    out: dict[tuple[str, int], dict[tuple[str, str], dict]] = {}
    for r in records:
        point = out.setdefault((r["race_id"], r["tau_min"]), {})
        assert (r["pool"], r["combination"]) not in point, ("a key was written twice", r)
        point[(r["pool"], r["combination"])] = r
    return out


def close(got, want) -> bool:
    return got is not None and math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12)


# ---------------------------------------------------------------- records with p_model


LAMBDA_CASES = [(0.85, 0.7), (1.0, 1.0), (1.25, 0.6)]


@pytest.mark.parametrize("lambdas", LAMBDA_CASES, ids=str)
def test_new_records_carry_p_model_and_the_exotic_values(lambdas):
    records = run_day(models(lambdas))

    points = by_point(records)
    assert set(points) == {(r.race_id, tau) for r in (R1, R2) for tau in GRID}
    for race in (R1, R2):
        for tau in GRID:
            where = (race.race_id, tau)
            got = points[where]
            ref = reference_p(race, tau, lambdas)
            want_exotic = stored_exotics(race, tau, ref)
            decision = race.post - timedelta(minutes=tau)

            # the race's exotic pools are those it was captured and priced for
            pools = {pool for pool, _ in got}
            assert pools == {WIN, PLACE, WIDE} | set(want_exotic), (where, sorted(pools))
            assert (BRACKET in pools) == (len(race.base) >= 9), where
            assert (TIERCE in pools) == (race.race_id in TIERCE_LOGGED), where

            # WIN, PLACE and Wide: the #120 keys, each with its p_model
            assert {k for k in got if k[0] in (WIN, PLACE, WIDE)} == base_keys(race), where
            for (pool, combination), r in got.items():
                assert r["record_version"] == shadow_log.RECORD_VERSION == 2, (where, pool)
                assert shadow_log.record_version(r) == 2
                assert r["status"] == "ok" and r["models"] == IDS, (where, pool, combination)
                assert close(r["p_model"], ref[pool][combination]), (
                    where,
                    pool,
                    combination,
                    r["p_model"],
                    ref[pool][combination],
                )
                if pool in (WIN, PLACE, WIDE):
                    assert r["board_odds"] == board_figure(race, tau, pool, combination)
                    continue
                # exotic: decision-time odds (never the trap), value = p_model x those odds
                odds = board_figure(race, tau, pool, combination)
                assert r["board_odds"] == odds, (where, pool, combination, r["board_odds"])
                assert close(r["value_model"], ref[pool][combination] * odds), (
                    where,
                    pool,
                    combination,
                )
                assert instant(r["board_capture_utc"]) <= decision, (where, pool, combination)

            # the stored exotic subset: every Quinella and Bracket quinella, the top 15 of others
            for pool, combos in want_exotic.items():
                logged = {c for p, c in got if p == pool}
                if pool in TOPPED:
                    assert len(logged) == min(TOP_N, len(ref[pool])), (where, pool, len(logged))
                    rest = set(ref[pool]) - logged
                    assert (
                        min(ref[pool][c] for c in logged) >= max(ref[pool][c] for c in rest) - 1e-15
                    ), (where, pool, "a less likely combination was stored")
                assert logged == combos, (where, pool, sorted(logged ^ combos)[:6])

    # the exotic subset keeps the log small: R1 holds 45 + 30 Quinella and Bracket keys and 3 x 15
    per_point = {race.race_id: len(base_keys(race)) for race in (R1, R2)}
    assert len(records) == len(GRID) * (per_point[R1.race_id] + 45 + 30 + 3 * TOP_N) + len(GRID) * (
        per_point[R2.race_id] + 21 + 2 * TOP_N
    )
    named = sorted({k for k in all_keys(records) for w in BANNED_KEY_WORDS if w in k.lower()})
    assert named == [], named


WITHOUT_P = {
    "no models": None,
    "an ordering model without discounts": "fake ordering",
}


@pytest.mark.parametrize("case", list(WITHOUT_P.values()), ids=list(WITHOUT_P))
def test_without_p_model_records_are_the_old_keys_and_still_verify(case):
    shadow_models = None if case is None else models((0.85, 0.7), discounts=False)

    records = run_day(shadow_models)

    points = by_point(records)
    assert set(points) == {(r.race_id, tau) for r in (R1, R2) for tau in GRID}
    for (race_id, tau), got in points.items():
        race = R1 if race_id == R1.race_id else R2
        assert set(got) == base_keys(race), ((race_id, tau), "an exotic record without p_model")
        for (pool, combination), r in got.items():
            assert r["record_version"] == 2 and "p_model" in r, (race_id, tau, pool)
            if case is None or pool != WIN:
                assert r["p_model"] is None, (race_id, tau, pool, combination)
            else:  # WIN alone has a probability without the fit's discounts
                win, _, _ = figures(race, factor(tau))
                horses = sorted(win)
                inv = np.array([1.0 / win[h] for h in horses])
                p = win_p(inv / inv.sum())
                assert close(r["p_model"], float(p[horses.index(int(combination))]))


# ---------------------------------------------------------------- scoring the day


FINISH = {R1.race_id: (3, 5, 1), R2.race_id: (2, 6, 4)}


def day_dividends() -> list[dict]:
    """Invented final dividends for the finishing orders above (yen per 100 yen)."""
    rows = []

    def pay(race_id, pool, combination, dividend):
        rows.append(
            {
                "race_id": race_id,
                "pool": pool,
                "combination": combination,
                "dividend": float(dividend),
                "is_refund": False,
            }
        )

    for race in (R1, R2):
        a, b, c = FINISH[race.race_id]
        rid = race.race_id
        pay(rid, WIN, str(a), 410 + race.no)
        places = (a, b, c) if len(race.base) >= 8 else (a, b)
        for i, h in enumerate(places):
            pay(rid, PLACE, str(h), 150 + 40 * i + race.no)
        for i, (x, y) in enumerate(sorted(combinations(sorted((a, b, c)), 2))):
            pay(rid, WIDE, f"{x},{y}", 330 + 70 * i)
        pay(rid, QUINELLA, ",".join(map(str, sorted((a, b)))), 1270 + race.no)
        pay(rid, FORECAST, f"{a},{b}", 2630)
        pay(rid, TRIO, ",".join(map(str, sorted((a, b, c)))), 3910)
        pay(rid, TIERCE, f"{a},{b},{c}", 18820)
        brackets = brackets_of(race)
        if brackets is not None:
            x, y = sorted((brackets[a], brackets[b]))
            pay(rid, BRACKET, f"{x},{y}", 990)
    return rows


def test_scoring_the_day_reproduces_the_return_per_100_yen():
    lambdas = (0.85, 0.7)
    records = run_day(models(lambdas))
    dividends = day_dividends()
    paying = {(d["race_id"], d["pool"], d["combination"]): d["dividend"] for d in dividends}

    want: dict[tuple[str, int], list[float]] = {}  # (pool, tau) -> [n, hits, value sum, paid]
    for race in (R1, R2):
        for tau in GRID:
            ref = reference_p(race, tau, lambdas)
            keys = {(pool, c) for pool, c in base_keys(race)}
            keys |= {(pool, c) for pool, cs in stored_exotics(race, tau, ref).items() for c in cs}
            for pool, combination in keys:
                acc = want.setdefault((pool, tau), [0, 0, 0.0, 0.0])
                dividend = paying.get((race.race_id, pool, combination), 0.0)
                acc[0] += 1
                acc[1] += dividend > 0
                acc[2] += ref[pool][combination] * board_figure(race, tau, pool, combination)
                acc[3] += dividend

    from horsetrack.countries.japan.shadow.score import score_shadow  # the module #564 adds

    got = score_shadow(records, pd.DataFrame(dividends))

    assert set(got) == set(want), sorted(set(got) ^ set(want))
    for (pool, tau), (n, hits, values, paid) in want.items():
        score = got[(pool, tau)]
        assert (score.pool, score.tau_min, score.n, score.hits) == (pool, tau, n, hits), score
        assert math.isclose(score.expected_per_100, 100 * values / n, rel_tol=1e-9), (pool, tau)
        assert math.isclose(score.realised_per_100, paid / n, rel_tol=1e-9, abs_tol=1e-12), (
            pool,
            tau,
        )
