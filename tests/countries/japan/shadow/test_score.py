"""Scoring shadow records against the final dividends (ticket #564).

Contract pinned here (`horsetrack.countries.japan.shadow.score`, a new module):

`score_shadow(records, dividends) -> dict[tuple[str, int], PoolScore]`, keyed by (pool, tau_min).
- `records`: shadow log records as `ShadowLog.read()` returns them. `dividends`: a frame with the
  columns of `bets.settle.DIVIDEND_COLUMNS` (race_id, pool, combination, dividend, is_refund), the
  dividend in yen per 100 yen, combinations in the store's order (the odds log's texts).
- One notional flat 100 yen on every scored record. A record is scored when it is version 2 with a
  `p_model` (not null), status "ok", a `board_odds` (not null), labelled `development`, and its race
  and pool have dividend rows. Left out: version-1 records (no p_model), a null p_model, no_board
  records, holdout-labelled records, a (race, pool) with no dividend row yet (unsettled), and a
  combination with a refund row (its stake comes back; it is not measured).
- `PoolScore(pool, tau_min, n, hits, expected_per_100, realised_per_100)`: n the records scored;
  hits those whose combination has a paying row of their race and pool (ordered pools match in
  order: TIERCE "1,2,3" is not "2,1,3"); expected_per_100 = 100 x mean(p_model x board_odds), the
  return per 100 yen the model claims at the decision-time odds; realised_per_100 = the dividends
  paid summed over the scored records / n (yen per 100 yen staked), 0 for a loss.
- A (pool, tau) with nothing scored has no entry.

Every record and dividend is synthetic and built here; the expected numbers are worked out by
hand in the comments (and, in the table test, by a reference loop in this file).
"""

# Ruff reads shadow.score as third-party until it exists, then as first-party; the block is
# written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math
import random
from datetime import date

import pandas as pd
import pytest

from horsetrack.countries.japan.shadow.score import PoolScore, score_shadow

DAY = date(2026, 11, 21)
RACE_A = f"jra:{DAY.isoformat()}:TOK:01"
RACE_B = f"jra:{DAY.isoformat()}:TOK:02"
RACE_C = f"jra:{DAY.isoformat()}:TOK:03"  # no dividends yet: unsettled
COLUMNS = ["race_id", "pool", "combination", "dividend", "is_refund"]
ORDERED = {"FORECAST", "TIERCE"}


def record(race_id, pool, combination, tau, p, odds, *, version=2, status="ok", held=False):
    """A shadow record as read back from the log."""
    out = {
        "race_id": race_id,
        "pool": pool,
        "combination": combination,
        "tau_min": tau,
        "decision_utc": f"{DAY.isoformat()}T01:00:00+00:00",
        "board_capture_utc": None if status != "ok" else f"{DAY.isoformat()}T00:59:20+00:00",
        "written_utc": f"{DAY.isoformat()}T01:00:20+00:00",
        "mode": "live",
        "status": status,
        "models": {"win": "w1", "ordering": "o1", "nowcast": "n1"},
        "git_sha": "abc1234",
        "board_odds": odds,
        "board_odds_hi": None,
        "value_mean": None,
        "value_q05": None,
        "value_q95": None,
        "value_at_low": None,
        "value_at_high": None,
        "flags": [],
        "day_status": "holdout" if held else "development",
    }
    if version == 2:
        out["record_version"] = 2
        out["p_model"] = p
        if pool not in ("WIN", "PLACE", "QUINELLA_PLACE") and p is not None and odds is not None:
            out["value_model"] = p * odds
    return out


def paid(race_id, pool, combination, dividend):
    return {
        "race_id": race_id,
        "pool": pool,
        "combination": combination,
        "dividend": float(dividend),
        "is_refund": False,
    }


def refund(race_id, pool, combination):
    return {
        "race_id": race_id,
        "pool": pool,
        "combination": combination,
        "dividend": None,
        "is_refund": True,
    }


def frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


def check(score, pool, tau, n, hits, expected, realised) -> None:
    assert isinstance(score, PoolScore), type(score)
    assert (score.pool, score.tau_min, score.n, score.hits) == (pool, tau, n, hits), score
    assert math.isclose(score.expected_per_100, expected, rel_tol=1e-9), (pool, tau, score)
    assert math.isclose(score.realised_per_100, realised, rel_tol=1e-9, abs_tol=1e-12), (
        pool,
        tau,
        score,
    )


# ---------------------------------------------------------------- a day worked by hand


def synthetic_day() -> tuple[list[dict], pd.DataFrame]:
    records = [
        # WIN at T-2: values .9, .9, 1.2, .9, .96 (sum 4.86); A paid "2" 300, B paid "1" 150
        record(RACE_A, "WIN", "1", 2, 0.5, 1.8),
        record(RACE_A, "WIN", "2", 2, 0.3, 3.0),
        record(RACE_A, "WIN", "3", 2, 0.2, 6.0),
        record(RACE_B, "WIN", "1", 2, 0.6, 1.5),
        record(RACE_B, "WIN", "2", 2, 0.4, 2.4),
        # left out of WIN at T-2: an old record, a null p_model, a holdout label, an unsettled race
        record(RACE_B, "WIN", "3", 2, None, 8.0, version=1),
        record(RACE_B, "WIN", "4", 2, None, 9.0),
        record(RACE_A, "WIN", "5", 2, 0.1, 10.0, held=True),
        record(RACE_C, "WIN", "1", 2, 0.7, 1.4),
        # WIN at T-5: values .9, .98, 1.1 (sum 2.98); A paid "2" 300
        record(RACE_A, "WIN", "1", 5, 0.45, 2.0),
        record(RACE_A, "WIN", "2", 5, 0.35, 2.8),
        record(RACE_A, "WIN", "3", 5, 0.2, 5.5),
        # PLACE at T-2 (race B): values .99, 1.04, .6 (sum 2.63); paid "1" 110 and "2" 130
        record(RACE_B, "PLACE", "1", 2, 0.9, 1.1),
        record(RACE_B, "PLACE", "2", 2, 0.8, 1.3),
        record(RACE_B, "PLACE", "3", 2, 0.3, 2.0),
        # QUINELLA at T-2 (race A): values 1.0, 1.05; "1,4" refunded (horse 4 scratched)
        record(RACE_A, "QUINELLA", "1,2", 2, 0.25, 4.0),
        record(RACE_A, "QUINELLA", "2,3", 2, 0.15, 7.0),
        record(RACE_A, "QUINELLA", "1,4", 2, 0.05, 20.0),
        # TIERCE at T-2 (race A): values 1.136, .972, 1.05 (sum 3.158); paid "2,1,3" 1390
        record(RACE_A, "TIERCE", "2,1,3", 2, 0.08, 14.2),
        record(RACE_A, "TIERCE", "1,2,3", 2, 0.12, 8.1),
        record(RACE_A, "TIERCE", "1,3,2", 2, 0.07, 15.0),
        # race B has no TIERCE dividend row: unsettled, left out
        record(RACE_B, "TIERCE", "1,2,3", 2, 0.2, 5.0),
        # TRIO at T-2 (race A): values .93, .8; paid "1,2,3" 320
        record(RACE_A, "TRIO", "1,2,3", 2, 0.3, 3.1),
        record(RACE_A, "TRIO", "1,2,5", 2, 0.02, 40.0),
        # a no_board point: never scored
        record(RACE_A, "*", "*", 16, None, None, status="no_board"),
    ]
    dividends = frame(
        [
            paid(RACE_A, "WIN", "2", 300),
            paid(RACE_A, "PLACE", "2", 140),
            paid(RACE_A, "QUINELLA", "1,2", 410),
            refund(RACE_A, "QUINELLA", "1,4"),
            paid(RACE_A, "TRIO", "1,2,3", 320),
            paid(RACE_A, "TIERCE", "2,1,3", 1390),
            paid(RACE_B, "WIN", "1", 150),
            paid(RACE_B, "PLACE", "1", 110),
            paid(RACE_B, "PLACE", "2", 130),
        ]
    )
    return records, dividends


def test_a_synthetic_day_scores_to_the_return_worked_by_hand():
    records, dividends = synthetic_day()

    scores = score_shadow(records, dividends)

    assert set(scores) == {
        ("WIN", 2),
        ("WIN", 5),
        ("PLACE", 2),
        ("QUINELLA", 2),
        ("TIERCE", 2),
        ("TRIO", 2),
    }
    check(scores[("WIN", 2)], "WIN", 2, n=5, hits=2, expected=97.2, realised=90.0)
    check(scores[("WIN", 5)], "WIN", 5, n=3, hits=1, expected=298 / 3, realised=100.0)
    check(scores[("PLACE", 2)], "PLACE", 2, n=3, hits=2, expected=263 / 3, realised=80.0)
    check(scores[("QUINELLA", 2)], "QUINELLA", 2, n=2, hits=1, expected=102.5, realised=205.0)
    check(scores[("TIERCE", 2)], "TIERCE", 2, n=3, hits=1, expected=315.8 / 3, realised=1390 / 3)
    check(scores[("TRIO", 2)], "TRIO", 2, n=2, hits=1, expected=86.5, realised=160.0)


def test_the_score_moves_with_p_model_odds_and_dividends():
    """The same day with one figure changed at a time: each change moves only what it should."""
    records, dividends = synthetic_day()
    base = score_shadow(records, dividends)

    # a higher p_model on A WIN "3" at T-2 (.2 -> .25): expected +100 x .05 x 6.0 / 5 = +6
    bumped = [
        {**r, "p_model": 0.25}
        if (r["race_id"], r["pool"], r["combination"], r["tau_min"]) == (RACE_A, "WIN", "3", 2)
        else r
        for r in records
    ]
    moved = score_shadow(bumped, dividends)
    check(moved[("WIN", 2)], "WIN", 2, n=5, hits=2, expected=103.2, realised=90.0)
    assert moved[("WIN", 5)] == base[("WIN", 5)]

    # a larger WIN dividend in race B (150 -> 190): realised at T-2 +40 / 5 = +8, expected unmoved
    richer = dividends.copy()
    richer.loc[(richer.race_id == RACE_B) & (richer.pool == "WIN"), "dividend"] = 190.0
    moved = score_shadow(records, richer)
    check(moved[("WIN", 2)], "WIN", 2, n=5, hits=2, expected=97.2, realised=98.0)

    # race C settles (its "1" pays 140): it joins WIN at T-2, n 6, values sum 4.86 + .98
    settled = pd.concat([dividends, frame([paid(RACE_C, "WIN", "1", 140)])], ignore_index=True)
    moved = score_shadow(records, settled)
    check(moved[("WIN", 2)], "WIN", 2, n=6, hits=3, expected=584 / 6, realised=590 / 6)


# ---------------------------------------------------------------- a table of random days


def reference(records: list[dict], dividends: list[dict]) -> dict[tuple[str, int], tuple]:
    """(n, hits, expected_per_100, realised_per_100) per (pool, tau), by the contract's rules."""
    pays: dict[tuple[str, str], dict[str, float | None]] = {}
    for row in dividends:
        pays.setdefault((row["race_id"], row["pool"]), {})[row["combination"]] = (
            None if row["is_refund"] else row["dividend"]
        )
    sums: dict[tuple[str, int], list[float]] = {}
    for r in records:
        rows = pays.get((r["race_id"], r["pool"]))
        if (
            rows is None
            or r.get("p_model") is None
            or r["combination"] in rows
            and (rows[r["combination"]] is None)
        ):
            continue
        dividend = rows.get(r["combination"], 0.0)
        acc = sums.setdefault((r["pool"], r["tau_min"]), [0, 0, 0.0, 0.0])
        acc[0] += 1
        acc[1] += dividend > 0
        acc[2] += r["p_model"] * r["board_odds"]
        acc[3] += dividend
    return {k: (n, h, 100 * v / n, d / n) for k, (n, h, v, d) in sums.items()}


def random_day(seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    records, dividends = [], []
    for no in range(1, rng.randint(3, 6) + 1):
        race_id = f"jra:{DAY.isoformat()}:KYO:{no:02d}"
        runners = list(range(1, rng.randint(5, 9) + 1))
        order = rng.sample(runners, 3)
        settled = rng.random() < 0.8
        for tau in rng.sample([60, 30, 16, 5, 2, 1], 3):
            for pool in ("WIN", "QUINELLA", "FORECAST"):
                if pool == "WIN":
                    combos = [str(h) for h in runners]
                elif pool == "QUINELLA":
                    combos = [f"{a},{b}" for a in runners for b in runners if a < b]
                else:
                    combos = [f"{a},{b}" for a in runners for b in runners if a != b]
                for combination in rng.sample(combos, min(len(combos), 6)):
                    p = round(rng.uniform(0.001, 0.6), 4)
                    odds = round(rng.uniform(1.1, 80.0), 1)
                    records.append(record(race_id, pool, combination, tau, p, odds))
        if settled:
            winner = order[0]
            dividends.append(paid(race_id, "WIN", str(winner), rng.randrange(110, 3000, 10)))
            q = ",".join(str(h) for h in sorted(order[:2]))
            dividends.append(paid(race_id, "QUINELLA", q, rng.randrange(200, 9000, 10)))
            f = ",".join(str(h) for h in order[:2])
            dividends.append(paid(race_id, "FORECAST", f, rng.randrange(300, 20000, 10)))
    return records, dividends


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_random_days_score_to_the_reference(seed):
    records, rows = random_day(seed)
    want = reference(records, rows)
    assert want, "the generated day scores something"

    got = score_shadow(records, frame(rows))

    assert set(got) == set(want), (seed, sorted(set(got) ^ set(want)))
    for (pool, tau), (n, hits, expected, realised) in want.items():
        check(got[(pool, tau)], pool, tau, n, hits, expected, realised)


def test_an_ordered_pool_matches_in_finishing_order_and_an_unordered_one_in_any():
    records = [
        record(RACE_A, "FORECAST", "2,1", 2, 0.1, 9.0),
        record(RACE_A, "FORECAST", "1,2", 2, 0.2, 4.5),
        record(RACE_A, "QUINELLA", "1,2", 2, 0.3, 3.0),
        record(RACE_A, "TRIO", "2,3,5", 2, 0.05, 30.0),
    ]
    dividends = frame(
        [
            paid(RACE_A, "FORECAST", "2,1", 870),
            paid(RACE_A, "QUINELLA", "1,2", 430),
            paid(RACE_A, "TRIO", "2,3,5", 2210),
        ]
    )

    got = score_shadow(records, dividends)

    check(got[("FORECAST", 2)], "FORECAST", 2, n=2, hits=1, expected=90.0, realised=435.0)
    check(got[("QUINELLA", 2)], "QUINELLA", 2, n=1, hits=1, expected=90.0, realised=430.0)
    check(got[("TRIO", 2)], "TRIO", 2, n=1, hits=1, expected=150.0, realised=2210.0)


def test_nothing_to_score_gives_no_entries():
    records, dividends = synthetic_day()
    old = [r for r in records if "record_version" not in r]
    assert old, "the day has an old record"

    assert score_shadow(old, dividends) == {}
    assert score_shadow(records, frame([])) == {}
    assert score_shadow([], dividends) == {}
