"""T-2 cross-pool features from the Win and Quinella boards (ticket #434).

Module under test: `horsetrack.countries.japan.studies.cross_pool`. Contract pinned here:

- `boards_at(day, *, minutes_before=2.0) -> dict[race_id, RaceBoards]`. `day` is one
  odds_timeseries day (`logger.compact.SCHEMA` columns). The board at t uses only rows with
  `minutes_to_post >= minutes_before` and takes, per (pool, combination), the latest row by
  availability (`fetched_utc`, else `snapshot_utc`, as `board.snapshot.board_at`). QUINELLA
  combinations are encoded "a,b" (ascending, as the JRA-VAN day files) and keyed (a, b) with a < b.
  Pool-total rows (`__pool__`) and other pools are ignored. A runner whose latest WIN row at t is
  scratched (`status_raw` "SCR") or unpriced (null odds) leaves both boards: its WIN entry and
  every Quinella pair it is in, even a pair still priced.
- `RaceBoards.win_utc` / `quinella_utc`: the availability time of the pool's capture on the board.
- `win_shares`: 1/odds normalised; `quinella_shares`: 1/odds normalised over the pairs.
- `harville_pairs(p, lambda2)`: P({i, j} first two, either order) with the second-place weights
  p ** lambda2 (`ordering._order_probs`): p_i w_j / (S - w_i) + p_j w_i / (S - w_j), S = sum w.
- `cross_features(boards, lambda2)` rows, one per starter: `p_win_t` the Win share; `q_top2_t`
  the sum of the Quinella shares of the pairs holding the runner; `h_top2_t` the same from the
  Harville pairs of the Win shares; `x_cross` a disagreement measure with the sign of
  q_top2_t - h_top2_t, zero when they are equal (the formula is left to the module); `is_fav` the
  runner with the largest Win share; `fav_support` the favourite's x_cross on every row of the
  race; `kl_race` = sum over pairs of q ln(q / h); `gap_s` = (quinella_utc - win_utc) in seconds,
  signed (positive when the Quinella capture is the later one).

Expected values are derived here from the stated formulas, never from the module. All data are
synthetic and in memory.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import ast
import itertools
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.countries.japan.logger.compact import SCHEMA
from horsetrack.countries.japan.studies import cross_pool
from horsetrack.countries.japan.studies.cross_pool import (
    boards_at,
    cross_features,
    harville_pairs,
    pair_features,
    quinella_shares,
    win_shares,
)

UTC = timezone.utc
DAY = "2025-06-01"
POST = datetime(2025, 6, 1, 6, 40, tzinfo=UTC)
POOL_ROW = "__pool__"
TIME_TEXT = "15時38分現在"
FETCH_LAG_S = 4.0  # snapshot_utc is this many seconds before fetched_utc
FEATURE_COLUMNS = {
    "horse_no",
    "p_win_t",
    "q_top2_t",
    "h_top2_t",
    "x_cross",
    "is_fav",
    "fav_support",
    "kl_race",
    "gap_s",
}
TOL = 1e-9


# ---------------------------------------------------------------- references (first principles)


def ref_shares(odds: dict) -> dict:
    inv = {k: 1.0 / v for k, v in odds.items()}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}


def ref_harville(p: dict[int, float], lam: float = 1.0) -> dict[tuple[int, int], float]:
    w = {k: v**lam for k, v in p.items()}
    s = sum(w.values())
    return {
        (a, b): p[a] * w[b] / (s - w[a]) + p[b] * w[a] / (s - w[b])
        for a, b in itertools.combinations(sorted(p), 2)
    }


def ref_top2(pairs: dict[tuple[int, int], float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for (a, b), v in pairs.items():
        out[a] = out.get(a, 0.0) + v
        out[b] = out.get(b, 0.0) + v
    return out


def ref_kl(q: dict, h: dict) -> float:
    return sum(v * math.log(v / h[k]) for k, v in q.items())


# ---------------------------------------------------------------- synthetic odds_timeseries rows


def race_id(race_no: int, venue: str = "TOK") -> str:
    return f"jra:{DAY}:{venue}:{race_no:02d}"


def row(rid, pool, combination, minutes, odds, *, status=TIME_TEXT, investment=None, lag=None):
    fetched = POST - timedelta(minutes=minutes)
    snapshot = fetched - timedelta(seconds=FETCH_LAG_S if lag is None else lag)
    _, _, venue, no = rid.split(":")
    return {
        "race_id": rid,
        "venue": venue,
        "race_no": int(no),
        "pool": pool,
        "combination": combination,
        "snapshot_utc": snapshot,
        "fetched_utc": fetched,
        "post_utc": POST,
        "minutes_to_post": float(minutes),
        "odds": odds,
        "odds_hi": None,
        "investment": investment,
        "status_raw": status,
        "is_final_capture": False,
        "dividend": None,
        "is_refund": None,
    }


def win_capture(rid, minutes, odds: dict[int, float | None], *, status=None, total=1.0e7):
    status = status or {}
    rows = [
        row(rid, "WIN", str(h), minutes, o, status=status.get(h, TIME_TEXT))
        for h, o in odds.items()
    ]
    rows.append(row(rid, "WIN", POOL_ROW, minutes, None, status=None, investment=total))
    return rows


def quinella_capture(rid, minutes, odds: dict[tuple[int, int], float | None], *, total=2.0e7):
    rows = [row(rid, "QUINELLA", f"{a},{b}", minutes, o) for (a, b), o in odds.items()]
    rows.append(row(rid, "QUINELLA", POOL_ROW, minutes, None, status=None, investment=total))
    return rows


def place_capture(rid, minutes, horses):
    return [row(rid, "PLACE", str(h), minutes, 1.1 + 0.3 * h) for h in horses]


def day_frame(rows: list[dict], *, seed: int = 7) -> pd.DataFrame:
    """The rows as the compacted day file reads back, in a shuffled order."""
    frame = pa.Table.from_pylist(rows, schema=SCHEMA).to_pandas()
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def consistent_odds(p: dict[int, float], lam: float = 1.0):
    """WIN odds whose shares are p, and QUINELLA odds whose shares are p's Harville pairs."""
    win = {h: 0.80 / v for h, v in p.items()}
    quinella = {pair: 0.775 / v for pair, v in ref_harville(p, lam).items()}
    return win, quinella


def normalise(p: list[float], horses=None) -> dict[int, float]:
    horses = horses or range(1, len(p) + 1)
    total = sum(p)
    return {h: v / total for h, v in zip(horses, p, strict=True)}


def one_race_boards(win, quinella, *, win_min=3.0, q_min=2.5, rid=None):
    rid = rid or race_id(11)
    rows = win_capture(rid, win_min, win) + quinella_capture(rid, q_min, quinella)
    return boards_at(day_frame(rows))[rid]


def by_horse(frame: pd.DataFrame, column: str) -> dict[int, float]:
    return {int(h): float(v) for h, v in zip(frame["horse_no"], frame[column], strict=True)}


def assert_close_maps(got: dict, want: dict, tol: float = TOL) -> None:
    assert set(got) == set(want)
    for k, v in want.items():
        assert got[k] == pytest.approx(v, abs=tol), k


NUMERIC_FEATURES = sorted(FEATURE_COLUMNS - {"horse_no"})
QUINELLA_COLUMNS = ("q_top2_t", "x_cross", "fav_support", "kl_race")


def assert_no_inf(frame: pd.DataFrame) -> None:
    """No feature column holds +inf or -inf (NaN is the only "unavailable")."""
    for column in frame.columns:
        if column == "pair":
            continue
        values = pd.to_numeric(frame[column], errors="coerce").astype(float).to_numpy()
        assert not np.isinf(values).any(), column


# ---------------------------------------------------------------- the board at t


def test_board_at_t_takes_the_latest_capture_at_or_before_t():
    a, b = race_id(11), race_id(12, "KYO")
    horses = (1, 2, 3, 4, 5)
    pairs = list(itertools.combinations(horses, 2))

    def win_odds(m):  # every capture prices every runner differently
        return {h: round(2.0 + h * 1.7 + m * 0.11, 2) for h in horses}

    def q_odds(m):
        return {pr: round(5.0 + pr[0] * 2.3 + pr[1] * 1.9 + m * 0.37, 2) for pr in pairs}

    rows = []
    for m in (10.0, 3.0, 1.5):
        rows += win_capture(a, m, win_odds(m))
    for m in (2.5, 1.0):
        rows += quinella_capture(a, m, q_odds(m))
    rows += place_capture(a, 2.2, horses)
    # A second race of the day on its own clock: WIN at 6 and 2.0, QUINELLA at 4 and 0.5.
    for m in (6.0, 2.0):
        rows += win_capture(b, m, {h: o + 0.5 for h, o in win_odds(m).items()})
    for m in (4.0, 0.5):
        rows += quinella_capture(b, m, {pr: o + 1.0 for pr, o in q_odds(m).items()})
    day = day_frame(rows)

    at_t2 = boards_at(day)  # the default horizon is T-2
    assert set(at_t2) == {a, b}
    assert at_t2[a].win == win_odds(3.0)
    assert at_t2[a].quinella == q_odds(2.5)
    assert pd.Timestamp(at_t2[a].win_utc) == pd.Timestamp(POST - timedelta(minutes=3.0))
    assert pd.Timestamp(at_t2[a].quinella_utc) == pd.Timestamp(POST - timedelta(minutes=2.5))
    # T-2 itself is at or above the horizon: race b's 2.0-minute WIN capture is used.
    assert at_t2[b].win == {h: o + 0.5 for h, o in win_odds(2.0).items()}
    assert at_t2[b].quinella == {pr: o + 1.0 for pr, o in q_odds(4.0).items()}

    cases = [  # (horizon, race a's WIN capture, race a's QUINELLA capture)
        (1.0, 1.5, 1.0),
        (2.5, 3.0, 2.5),  # the boundary is inclusive
        (1.4, 1.5, 2.5),
        (0.0, 1.5, 1.0),
    ]
    for horizon, w_min, q_min in cases:
        boards = boards_at(day, minutes_before=horizon)[a]
        assert boards.win == win_odds(w_min), horizon
        assert boards.quinella == q_odds(q_min), horizon
        assert pd.Timestamp(boards.win_utc) == pd.Timestamp(POST - timedelta(minutes=w_min))
        assert pd.Timestamp(boards.quinella_utc) == pd.Timestamp(POST - timedelta(minutes=q_min))


def test_a_combination_missing_from_the_latest_capture_keeps_its_latest_row():
    rid = race_id(5)
    early = {1: 2.0, 2: 3.0, 3: 6.0, 4: 9.0}
    late = {1: 2.2, 2: 2.9, 3: 6.5}  # runner 4 not re-sent at 3.0
    pairs = {(1, 2): 4.0, (1, 3): 8.0, (1, 4): 12.0, (2, 3): 10.0, (2, 4): 15.0, (3, 4): 30.0}
    rows = win_capture(rid, 8.0, early) + win_capture(rid, 3.0, late)
    rows += quinella_capture(rid, 3.0, pairs)
    rows += win_capture(rid, 1.0, {1: 1.5, 2: 4.0, 3: 7.0, 4: 20.0})  # after t
    boards = boards_at(day_frame(rows))[rid]
    assert boards.win == {1: 2.2, 2: 2.9, 3: 6.5, 4: 9.0}
    assert boards.quinella == pairs


# ---------------------------------------------------------------- shares and Harville


def test_win_and_quinella_shares_are_normalised_inverse_odds():
    for win in ({1: 2.0, 2: 4.0, 3: 4.0}, {3: 1.6, 7: 5.5, 9: 12.0, 12: 48.0, 14: 130.0}):
        got = win_shares(win)
        assert_close_maps(got, ref_shares(win))
        assert sum(got.values()) == pytest.approx(1.0, abs=TOL)
    for quinella in (
        {(1, 2): 3.0, (1, 3): 6.0, (2, 3): 6.0},
        {(3, 7): 9.1, (3, 9): 14.2, (7, 9): 33.0, (3, 12): 60.5, (7, 12): 111.0, (9, 12): 240.0},
    ):
        got = quinella_shares(quinella)
        assert_close_maps(got, ref_shares(quinella))
        assert sum(got.values()) == pytest.approx(1.0, abs=TOL)


def test_harville_pairs_of_a_three_horse_field():
    pairs = harville_pairs({1: 0.5, 2: 0.3, 3: 0.2}, lambda2=1.0)
    assert set(pairs) == {(1, 2), (1, 3), (2, 3)}
    assert pairs[(1, 2)] == pytest.approx(0.5143, abs=1e-4)
    assert pairs[(1, 3)] == pytest.approx(0.3250, abs=1e-4)
    assert pairs[(2, 3)] == pytest.approx(0.1607, abs=1e-4)
    assert sum(pairs.values()) == pytest.approx(1.0, abs=TOL)
    assert harville_pairs({1: 0.5, 2: 0.3, 3: 0.2}) == pytest.approx(pairs)  # lambda2 defaults to 1


@pytest.mark.parametrize("lam", [1.0, 0.8, 1.25])
@pytest.mark.parametrize(
    "p",
    [
        normalise([0.4, 0.25, 0.2, 0.1, 0.05]),
        normalise([0.18, 0.16, 0.14, 0.12, 0.11, 0.1, 0.08, 0.06, 0.05], horses=range(3, 12)),
        normalise([0.7, 0.2, 0.06, 0.04], horses=(2, 5, 8, 13)),
    ],
)
def test_harville_pairs_follow_the_formula_over_fields_and_lambda2(p, lam):
    got = harville_pairs(p, lambda2=lam)
    assert_close_maps(got, ref_harville(p, lam))
    assert sum(got.values()) == pytest.approx(1.0, abs=TOL)
    if lam != 1.0:  # lambda2 changes the pairs
        assert max(abs(got[k] - v) for k, v in ref_harville(p, 1.0).items()) > 1e-4


# ---------------------------------------------------------------- the features


@pytest.mark.parametrize("lam", [1.0, 0.85])
@pytest.mark.parametrize(
    "p",
    [
        normalise([0.45, 0.25, 0.15, 0.1, 0.05]),
        normalise([0.3, 0.2, 0.15, 0.12, 0.09, 0.07, 0.04, 0.03]),
        normalise(
            [0.22, 0.2, 0.14, 0.12, 0.1, 0.08, 0.06, 0.04, 0.04],
            horses=(2, 3, 5, 7, 8, 10, 11, 13, 16),
        ),
    ],
)
def test_consistent_pools_give_no_disagreement(p, lam):
    win, quinella = consistent_odds(p, lam)
    feats = cross_features(one_race_boards(win, quinella), lambda2=lam)

    assert FEATURE_COLUMNS <= set(feats.columns)
    assert sorted(feats["horse_no"].astype(int)) == sorted(p)
    assert_close_maps(by_horse(feats, "p_win_t"), p)
    h_top2 = ref_top2(ref_harville(p, lam))
    assert_close_maps(by_horse(feats, "q_top2_t"), h_top2)
    assert_close_maps(by_horse(feats, "h_top2_t"), h_top2)
    assert all(abs(x) < 1e-9 for x in feats["x_cross"])
    assert all(abs(k) < 1e-9 for k in feats["kl_race"])
    assert all(abs(f) < 1e-9 for f in feats["fav_support"])

    # The same Quinella read with the wrong lambda2 disagrees with the Win board.
    if lam != 1.0:
        off = cross_features(one_race_boards(win, quinella), lambda2=1.0)
        assert max(abs(x) for x in off["x_cross"]) > 1e-4
        assert float(off["kl_race"].iloc[0]) > 1e-8


@pytest.mark.parametrize("factor", [1.15, 1.4])
@pytest.mark.parametrize(
    "p, planted",
    [
        (normalise([0.3, 0.2, 0.15, 0.12, 0.09, 0.07, 0.04, 0.03]), 1),  # the favourite
        (normalise([0.3, 0.2, 0.15, 0.12, 0.09, 0.07, 0.04, 0.03]), 4),
        (normalise([0.3, 0.2, 0.15, 0.12, 0.09, 0.07, 0.04, 0.03]), 8),
        (normalise([0.55, 0.25, 0.12, 0.08], horses=(3, 6, 9, 10)), 9),
    ],
)
def test_a_planted_overbet_runner_has_positive_cross_support(p, planted, factor):
    win, _ = consistent_odds(p)
    h = ref_harville(p)
    # The planted runner's pairs take `factor` times their Harville money.
    planted_q = {pr: v * (factor if planted in pr else 1.0) for pr, v in h.items()}
    z = sum(planted_q.values())
    q = {pr: v / z for pr, v in planted_q.items()}
    quinella = {pr: 0.775 / v for pr, v in q.items()}

    feats = cross_features(one_race_boards(win, quinella))
    x = by_horse(feats, "x_cross")
    assert x[planted] > 0
    assert all(v < 0 for k, v in x.items() if k != planted)

    assert_close_maps(by_horse(feats, "q_top2_t"), ref_top2(q))
    assert_close_maps(by_horse(feats, "h_top2_t"), ref_top2(h))
    kl = ref_kl(q, h)
    assert kl > 0
    assert all(v == pytest.approx(kl, abs=1e-10) for v in feats["kl_race"])

    fav = max(p, key=p.get)
    assert by_horse(feats, "is_fav") == {k: float(k == fav) for k in p}
    assert all(v == pytest.approx(x[fav], abs=TOL) for v in feats["fav_support"])
    assert (x[fav] > 0) == (planted == fav)

    # A stronger plant gives the planted runner a larger x_cross.
    stronger_q = {pr: v * (factor * 1.5 if planted in pr else 1.0) for pr, v in h.items()}
    stronger = cross_features(
        one_race_boards(
            win, {pr: 0.775 * sum(stronger_q.values()) / v for pr, v in stronger_q.items()}
        )
    )
    assert by_horse(stronger, "x_cross")[planted] > x[planted]


def test_x_cross_has_the_sign_of_the_top2_disagreement():
    p = normalise([0.32, 0.22, 0.16, 0.12, 0.1, 0.08])
    win, _ = consistent_odds(p)
    h = ref_harville(p)
    tilt = {1: 0.8, 2: 1.3, 3: 1.0, 4: 0.7, 5: 1.5, 6: 1.1}  # a mixed, unplanted distortion
    raw = {(a, b): v * tilt[a] * tilt[b] for (a, b), v in h.items()}
    z = sum(raw.values())
    q = {pr: v / z for pr, v in raw.items()}
    feats = cross_features(one_race_boards(win, {pr: 0.775 / v for pr, v in q.items()}))
    q_top2, h_top2 = ref_top2(q), ref_top2(h)
    x = by_horse(feats, "x_cross")
    signs = {k: q_top2[k] > h_top2[k] for k in p}
    assert any(signs.values()) and not all(signs.values())
    for k in p:
        assert (x[k] > 0) == signs[k], k
    assert float(feats["kl_race"].iloc[0]) == pytest.approx(ref_kl(q, h), abs=1e-10)


# ---------------------------------------------------------------- scratches


def test_scratched_runners_leave_both_boards():
    rid = race_id(9)
    horses = (1, 2, 3, 4, 5, 6)
    base = {1: 2.4, 2: 3.8, 3: 6.1, 4: 9.0, 5: 14.5, 6: 30.0}
    pairs = list(itertools.combinations(horses, 2))
    q_base = {pr: round(3.0 + 1.4 * pr[0] * pr[1], 1) for pr in pairs}
    rows = win_capture(rid, 10.0, base) + quinella_capture(rid, 10.0, q_base)
    # At 3 minutes: 4 is scratched (SCR, no price), 6 has no price; 4's pairs are still priced
    # (stale), 6's pairs are not.
    at3 = {1: 2.5, 2: 3.6, 3: 6.4, 4: None, 5: 15.0, 6: None}
    rows += win_capture(rid, 3.0, at3, status={4: "SCR", 6: "----"})
    q3 = {pr: (None if 6 in pr else q_base[pr] + 0.5) for pr in pairs}
    rows += quinella_capture(rid, 3.0, q3)
    # After t, runner 2 is scratched: it stays on the T-2 board.
    rows += win_capture(rid, 1.0, {**at3, 2: None}, status={2: "SCR", 4: "SCR", 6: "----"})
    boards = boards_at(day_frame(rows))[rid]

    starters = (1, 2, 3, 5)
    assert boards.win == {h: at3[h] for h in starters}
    assert boards.quinella == {pr: q3[pr] for pr in itertools.combinations(starters, 2)}

    w = win_shares(boards.win)
    assert_close_maps(w, ref_shares({h: at3[h] for h in starters}))
    qs = quinella_shares(boards.quinella)
    assert sum(qs.values()) == pytest.approx(1.0, abs=TOL)
    assert_close_maps(qs, ref_shares({pr: q3[pr] for pr in itertools.combinations(starters, 2)}))

    feats = cross_features(boards)
    assert sorted(feats["horse_no"].astype(int)) == list(starters)
    assert_close_maps(by_horse(feats, "p_win_t"), w)
    assert float(feats["p_win_t"].sum()) == pytest.approx(1.0, abs=TOL)
    assert_close_maps(by_horse(feats, "q_top2_t"), ref_top2(qs))
    assert_close_maps(by_horse(feats, "h_top2_t"), ref_top2(ref_harville(w)))
    assert float(feats["q_top2_t"].sum()) == pytest.approx(2.0, abs=TOL)
    assert float(feats["h_top2_t"].sum()) == pytest.approx(2.0, abs=TOL)


# ---------------------------------------------------------------- capture times


def test_the_capture_gap_between_pools_is_reported():
    p = normalise([0.4, 0.3, 0.2, 0.1])
    win, quinella = consistent_odds(p)
    a, b, c = race_id(1), race_id(2), race_id(3)
    rows = win_capture(a, 3.0, win) + quinella_capture(a, 2.5, quinella)  # Quinella 30 s later
    rows += win_capture(b, 2.2, win) + quinella_capture(b, 4.0, quinella)  # Quinella 108 s earlier
    rows += win_capture(c, 5.0, win) + quinella_capture(c, 5.0, quinella)  # the same capture
    boards = boards_at(day_frame(rows))

    expected = {a: (3.0, 2.5, 30.0), b: (2.2, 4.0, -108.0), c: (5.0, 5.0, 0.0)}
    for rid, (w_min, q_min, gap) in expected.items():
        rb = boards[rid]
        assert pd.Timestamp(rb.win_utc) == pd.Timestamp(POST - timedelta(minutes=w_min))
        assert pd.Timestamp(rb.quinella_utc) == pd.Timestamp(POST - timedelta(minutes=q_min))
        feats = cross_features(rb)
        assert len(feats) == 4
        assert all(g == pytest.approx(gap, abs=1e-6) for g in feats["gap_s"]), rid


# ---------------------------------------------------------------- no store, registry or holdout


FORBIDDEN_IMPORTS = (
    "horsetrack.core.store",
    "horsetrack.core.eval.registry",
    "horsetrack.core.eval.holdout",
    "horsetrack.countries.japan.models.v0.store",
)


def test_nothing_reads_the_store_the_registry_or_the_holdout(tmp_path, monkeypatch):
    tree = ast.parse(open(cross_pool.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    for name in imported:
        assert not any(name == f or name.startswith(f + ".") for f in FORBIDDEN_IMPORTS), name

    # The whole path runs on the frame alone: an empty data dir and no Parquet reads.
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))

    def refuse(*args, **kwargs):
        raise AssertionError("cross_pool read a file")

    monkeypatch.setattr(pd, "read_parquet", refuse)
    monkeypatch.setattr(pq, "read_table", refuse)
    win, quinella = consistent_odds(normalise([0.5, 0.3, 0.2]))
    feats = cross_features(one_race_boards(win, quinella))
    assert len(feats) == 3
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- nullable string columns


def jravan_rows(rows: list[dict]) -> list[dict]:
    """The rows as JRA-VAN's time series writes them: no fetch time, and `status_raw` null on
    every priced row (only a non-positive price keeps its raw text)."""
    out = []
    for r in rows:
        r = dict(r, fetched_utc=None)
        if r["odds"] is not None:
            r["status_raw"] = None
        out.append(r)
    return out


def _arrow_backed(table: pa.Table) -> pd.DataFrame:
    return table.to_pandas(types_mapper=pd.ArrowDtype)


def _convert_pyarrow(table: pa.Table) -> pd.DataFrame:
    return table.to_pandas().convert_dtypes(dtype_backend="pyarrow")


def _string_dtype(dtype: str):
    def read(table: pa.Table) -> pd.DataFrame:
        frame = table.to_pandas()
        for column in ("race_id", "pool", "combination", "status_raw"):
            frame[column] = frame[column].astype(dtype)
        return frame

    return read


READERS = {
    "arrow-dtype": _arrow_backed,
    "convert-dtypes-pyarrow": _convert_pyarrow,
    "string": _string_dtype("string"),
    "string[pyarrow]": _string_dtype("string[pyarrow]"),
}


@pytest.mark.parametrize("reader", sorted(READERS))
def test_nullable_status_columns_give_the_same_boards_and_features(reader):
    rid, other = race_id(7), race_id(8, "HAN")
    horses = (1, 2, 3, 4, 5, 6)
    win = {1: 2.6, 2: 3.9, 3: 5.8, 4: None, 5: 11.0, 6: 24.0}  # 4 cancelled before sale
    pairs = list(itertools.combinations(horses, 2))
    q = {pr: round(4.0 + 1.3 * pr[0] * pr[1] + 0.2 * pr[1], 1) for pr in pairs}
    rows = win_capture(rid, 3.0, win, status={4: "----"}) + quinella_capture(rid, 2.5, q)
    p2 = normalise([0.5, 0.3, 0.2])
    win2, q2 = consistent_odds(p2)
    rows += win_capture(other, 4.0, win2) + quinella_capture(other, 3.0, q2)
    table = pa.Table.from_pylist(jravan_rows(rows), schema=SCHEMA)
    plain = boards_at(table.to_pandas())
    nullable_day = READERS[reader](table)
    assert nullable_day["status_raw"].isna().any()
    nullable = boards_at(nullable_day)

    starters = (1, 2, 3, 5, 6)
    want = {
        rid: (
            {h: win[h] for h in starters},
            {pr: q[pr] for pr in itertools.combinations(starters, 2)},
            3.0,
            2.5,
        ),
        other: (win2, q2, 4.0, 3.0),
    }
    lag = timedelta(seconds=FETCH_LAG_S)  # no fetch time: the snapshot time is used
    for boards in (plain, nullable):
        assert set(boards) == {rid, other}
        for r, (w, qq, w_min, q_min) in want.items():
            assert boards[r].win == pytest.approx(w), (reader, r)
            assert boards[r].quinella == pytest.approx(qq), (reader, r)
            assert pd.Timestamp(boards[r].win_utc) == pd.Timestamp(
                POST - timedelta(minutes=w_min) - lag
            )
            assert pd.Timestamp(boards[r].quinella_utc) == pd.Timestamp(
                POST - timedelta(minutes=q_min) - lag
            )

    for r in want:
        got, ref = cross_features(nullable[r]), cross_features(plain[r])
        assert list(got["horse_no"].astype(int)) == list(ref["horse_no"].astype(int))
        assert len(got) == len(want[r][0])
        for column in NUMERIC_FEATURES:
            a = got[column].astype(float).to_numpy()
            b = ref[column].astype(float).to_numpy()
            assert np.allclose(a, b, atol=TOL, equal_nan=True), (reader, r, column)
            assert np.isfinite(a).all(), (reader, r, column)


# ---------------------------------------------------------------- pairs off the Win board


@pytest.mark.parametrize("absent", [5, 9])
def test_quinella_pairs_naming_a_runner_off_the_win_board_are_dropped(absent):
    rid = race_id(4)
    p = normalise([0.4, 0.3, 0.2, 0.1])
    win, quinella = consistent_odds(p)
    # An older Quinella capture names a runner that has no Win row at all.
    stale = {(h, absent): 40.0 + 3.0 * h for h in p}
    rows = win_capture(rid, 3.0, win) + quinella_capture(rid, 2.5, quinella)
    rows += quinella_capture(rid, 10.0, stale)
    boards = boards_at(day_frame(rows))[rid]

    assert set(boards.win) == set(p)
    assert boards.quinella == pytest.approx(quinella)
    assert pd.Timestamp(boards.quinella_utc) == pd.Timestamp(POST - timedelta(minutes=2.5))
    feats = cross_features(boards)
    assert all(g == pytest.approx(30.0, abs=1e-6) for g in feats["gap_s"])
    assert all(abs(x) < 1e-9 for x in feats["x_cross"])
    assert_no_inf(feats)


# ---------------------------------------------------------------- missing Quinella quotes


def _missing_rows(rid, horses, priced, missing, how):
    """WIN at 3 and QUINELLA at 2.5 over `horses`; the `missing` starter pairs have no price at
    t: `absent` (no row), `blank` (a null price, raw text blank), `stale` (priced at 10 minutes,
    blank at 2.5)."""
    win = {h: 2.0 + 1.5 * i for i, h in enumerate(horses)}
    rows = win_capture(rid, 3.0, win)
    if how == "absent":
        rows += quinella_capture(rid, 2.5, priced)
    else:
        rows += quinella_capture(rid, 2.5, {**priced, **dict.fromkeys(missing)})
        for r in rows:
            if r["pool"] == "QUINELLA" and r["odds"] is None and r["combination"] != POOL_ROW:
                r["status_raw"] = "    "
    if how == "stale":
        rows += quinella_capture(rid, 10.0, dict.fromkeys(missing, 50.0))
    return rows


@pytest.mark.parametrize("how", ["absent", "blank", "stale"])
def test_a_starter_pair_without_a_quinella_price_makes_the_race_unavailable(how):
    # The equal three-runner field of the review: only (1, 2) priced.
    a = race_id(3)
    rows = _missing_rows(a, (1, 2, 3), {(1, 2): 3.2}, [(1, 3), (2, 3)], how)
    # A larger field with one pair missing.
    b = race_id(6)
    horses = (2, 4, 5, 7, 8, 11)
    gone = (5, 8)
    priced_b = {pr: 6.0 + pr[0] + pr[1] for pr in itertools.combinations(horses, 2) if pr != gone}
    rows += _missing_rows(b, horses, priced_b, [gone], how)
    # A fully priced race of the same day stays available.
    c = race_id(10)
    win_c, q_c = consistent_odds(normalise([0.35, 0.3, 0.2, 0.15]))
    rows += win_capture(c, 3.0, win_c) + quinella_capture(c, 2.5, q_c)
    boards = boards_at(day_frame(rows))

    for rid, n, missing in ((a, 3, [(1, 3), (2, 3)]), (b, len(horses), [gone])):
        feats = cross_features(boards[rid])
        assert len(feats) == n
        assert_no_inf(feats)
        for column in QUINELLA_COLUMNS:
            assert feats[column].astype(float).isna().all(), (how, rid, column)
        # The Win-only columns are still there.
        assert np.isfinite(feats["p_win_t"].astype(float)).all()
        assert np.isfinite(feats["h_top2_t"].astype(float)).all()
        assert float(feats["p_win_t"].sum()) == pytest.approx(1.0, abs=TOL)

        pf = pair_features(boards[rid])
        assert_no_inf(pf)
        row_of = {tuple(pr): i for i, pr in enumerate(pf["pair"])}
        for pr in missing:
            assert math.isnan(float(pf["q_pair"].iloc[row_of[pr]])), (how, rid, pr)
            assert math.isnan(float(pf["log_ratio"].iloc[row_of[pr]])), (how, rid, pr)

    full = cross_features(boards[c])
    for column in QUINELLA_COLUMNS:
        assert np.isfinite(full[column].astype(float)).all(), column
    assert all(abs(x) < 1e-9 for x in full["x_cross"])


def test_a_zero_vote_quinella_pair_never_gives_an_infinite_feature():
    # JRA-VAN writes a zero-vote price ("0000") as a null price with its raw text in status_raw
    # (docs/specs/jravan.md); how cross_pool reads it is not pinned here, only that no feature is
    # infinite. Every pair of the outsider 4 has zero votes.
    rid = race_id(2)
    win = {1: 2.0, 2: 3.0, 3: 30.0, 4: 60.0}
    q = {(1, 2): 3.5, (1, 3): 20.0, (1, 4): None, (2, 3): 35.0, (2, 4): None, (3, 4): None}
    rows = win_capture(rid, 3.0, win) + quinella_capture(rid, 2.5, q)
    for r in rows:
        if r["pool"] == "QUINELLA" and r["combination"] in ("1,4", "2,4", "3,4"):
            r["status_raw"] = "0000"
    boards = boards_at(day_frame(rows))[rid]
    assert_no_inf(cross_features(boards))
    assert_no_inf(pair_features(boards))


# ---------------------------------------------------------------- pair features


@pytest.mark.parametrize("lam", [1.0, 0.85])
@pytest.mark.parametrize(
    "p",
    [
        normalise([0.45, 0.25, 0.15, 0.1, 0.05]),
        normalise([0.3, 0.2, 0.15, 0.12, 0.09, 0.07], horses=(2, 3, 5, 8, 9, 14)),
    ],
)
def test_pair_features_on_consistent_pools(p, lam):
    win, quinella = consistent_odds(p, lam)
    pf = pair_features(one_race_boards(win, quinella), lambda2=lam)
    h = ref_harville(p, lam)
    pairs = [tuple(pr) for pr in pf["pair"]]
    assert set(pairs) == set(h)
    assert_close_maps(dict(zip(pairs, pf["h_pair"].astype(float), strict=True)), h)
    assert_close_maps(dict(zip(pairs, pf["q_pair"].astype(float), strict=True)), h)
    assert float(pf["q_pair"].sum()) == pytest.approx(1.0, abs=TOL)
    assert all(abs(v) < 1e-9 for v in pf["log_ratio"])


@pytest.mark.parametrize("factor", [1.2, 1.6])
@pytest.mark.parametrize(
    "p, planted",
    [
        (normalise([0.3, 0.2, 0.15, 0.12, 0.09, 0.07, 0.04, 0.03]), 1),
        (normalise([0.3, 0.2, 0.15, 0.12, 0.09, 0.07, 0.04, 0.03]), 6),
        (normalise([0.55, 0.25, 0.12, 0.08], horses=(3, 6, 9, 10)), 10),
    ],
)
def test_pair_features_on_a_planted_overbet(p, planted, factor):
    win, _ = consistent_odds(p)
    h = ref_harville(p)
    raw = {pr: v * (factor if planted in pr else 1.0) for pr, v in h.items()}
    z = sum(raw.values())
    q = {pr: v / z for pr, v in raw.items()}
    pf = pair_features(one_race_boards(win, {pr: 0.775 / v for pr, v in q.items()}))
    pairs = [tuple(pr) for pr in pf["pair"]]
    assert set(pairs) == set(h)
    assert float(pf["q_pair"].sum()) == pytest.approx(1.0, abs=TOL)
    got = dict(zip(pairs, pf["log_ratio"].astype(float), strict=True))
    assert_close_maps(got, {pr: math.log(q[pr] / h[pr]) for pr in h})
    for pr, v in got.items():
        assert (v > 0) == (planted in pr), pr
    assert_no_inf(pf)
