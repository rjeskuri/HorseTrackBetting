"""The odds ladder of the shadow feed (ticket #499): observed and projected odds at T-20, T-10, T-5,
T-2 and the final, per (race, pool, combination).

Contract pinned here (`horsetrack.countries.japan.board.shadow_feed.build_shadow`, unchanged
signature):
- Every entry of the shadow feed's `pools` (each WIN runner, PLACE runner and Wide pair on the
  board, beside its value fields and flags) carries `ladder`: an object with the checkpoints 20,
  10, 5, 2 and 0 (the final) as keys (text "20", ... "0" once the feed is JSON; the feed stays
  JSON-ready), each either {"odds": x, "observed": true}, {"odds": x, "observed": false} or null.
- A checkpoint k > 0 already passed (now at or after post - k) is observed: the low figure of the
  latest row of the combination captured at or before post - k (the `t20_odds` rule of
  `board/feed.py`); null when that row is unpriced or there is none.
- The final (0) is observed only once the final board exists (a final capture, `final_board`), with
  that board's low figure; until then it is projected.
- A checkpoint still ahead is projected from the board read at now (O_now, the current low figure)
  with the nowcast the shadow values use (`models.nowcast`, a fitted `NowcastV0`, read, not
  refitted): ln O_k = ln O_now + mu(tau_now, b) - mu(k, b), b = band_of(O_now, nowcast.bands) for
  both terms, tau_now the minutes from the board read to the post, mu(pool, tau, b) the fitted
  cell's mean on the nowcast's grid (60, 30, 16, 5, 2, 1), linear in tau between neighbouring
  grid taus (T-20, T-10, a tau_now off the grid), the T-60 cell beyond T-60, and mu(0) = 0. The
  projected odds are exp of that (the median), floored at 1.0.
- tau_now is anchored to the time the price O_now comes from was fetched: minutes from that row's
  `fetched_utc` (its `snapshot_utc` when it has none; `snapshot._available`) to the post, not
  from now and not from the race's newest capture of another pool. The native logger stamps a
  capture before it fetches the odds pages, so a row snapped at post - 12 and fetched at post - 11
  projects from tau 11. With staggered captures (WIN at post - 12, a later Wide-only capture at
  post - 9) WIN projects from tau 12 and Wide from tau 9, whatever now is. The board at now and
  the observed checkpoints read only rows available (fetched) by now; an observed checkpoint is
  still chosen by `snapshot_utc` (the latest captured at or before post - k).
- mu(tau) reads only the cells it needs: the exact grid cell, or the two grid cells bracketing tau
  (only the upper one below the smallest grid tau). A fitted nowcast missing an unrelated (pool,
  tau) (`NowcastV0.cell` raises KeyError for it) still projects; a checkpoint whose projection
  needs the missing cell (for tau_now or for k) is null.
- Without models every projected checkpoint is null (observed ones stay); a pool the nowcast has
  no cells for gets null ahead while its observed checkpoints stay.
- The ladder is display only: the values, flags and shadow log are unchanged (their own tests).

The nowcasts are built here from known cells (no fit file), and every expected odds figure is
derived here from the logged rows and those cells, tau_now from each row's own fetch time. Most
races here capture every pool at now, fetched at their capture; the staggered-capture and
fetch-lag tests do not. All rows are
synthetic and written under tmp_path; nothing waits on a real clock.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import json
import math
from bisect import bisect_right
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from horsetrack.core import paths
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board.flags import FlagRules
from horsetrack.countries.japan.board.shadow_feed import build_shadow
from horsetrack.countries.japan.board.snapshot import load_decision_grid
from horsetrack.countries.japan.models.v0.nowcast import DriftCell, NowcastV0
from horsetrack.countries.japan.models.v0.shadow_values import ShadowModels

# isort: split
from shadow_fakes import (
    PLACE,
    WIDE,
    WIN,
    FakeOrdering,
    FakeWin,
    Race,
    SyntheticDay,
    capture_rows,
    instant,
    log_path,
)

UTC = timezone.utc
DAY = date(2026, 11, 21)
T0 = datetime(2026, 11, 21, 2, 0, tzinfo=UTC)  # now in every test: 11:00 in Tokyo
RULES = FlagRules(floor_max_odds=1.3, tail_min_odds=50.0, tail_rank=6, stale_s=40.0)
N_DRAWS = 16
BANDS = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
GRID = load_decision_grid()
POOLS = (WIN, PLACE, WIDE)
CHECKPOINTS = (20, 10, 5, 2, 0)
FINAL_TEXT = "最終オッズ"
TOL = 0.0051  # a projected figure may be rounded to two decimals

EIGHT = (1.25, 2.9, 4.6, 7.3, 11.8, 26.0, 64.0, 152.0)
EIGHT2 = (1.7, 3.6, 5.4, 9.2, 16.5, 33.0, 72.0, 118.0)
SIX = (1.8, 3.4, 6.2, 12.5, 38.0, 85.0)
FACTORS = (1.0, 1.06, 0.97, 1.11, 1.03, 0.92, 1.08)  # capture i of a race moves by FACTORS[i]


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


# ---------------------------------------------------------------- known nowcasts

MuFn = Callable[[str, int, int], float]

BAND_A = {0: -0.9, 1: -0.4, 2: 0.25, 3: 0.5, 4: 0.8, 5: 1.1, 6: 1.3}
POOL_A = {WIN: 1.0, PLACE: 0.6, WIDE: 0.8}
BAND_B = {0: 0.9, 1: 0.9, 2: -0.2, 3: -0.6, 4: 0.4, 5: -1.0, 6: 0.7}
POOL_B = {WIN: -0.7, PLACE: 1.2, WIDE: 0.5}


def mu_a(pool: str, tau: int, band: int) -> float:
    return POOL_A[pool] * BAND_A[band] * math.sqrt(tau) / 8.0


def mu_b(pool: str, tau: int, band: int) -> float:
    return POOL_B[pool] * BAND_B[band] * math.log1p(tau) / 4.0


def nowcast_of(mu: MuFn, pools=POOLS, fit_id: str = "test-fit", missing=()) -> NowcastV0:
    """A fitted v0 nowcast with the cell means mu(pool, tau, band) on every grid tau and band, but
    none (no cell, no sigma0: `cell` raises KeyError) for each (pool, tau) in `missing`."""
    sigma0 = {(p, t): 0.12 for p in pools for t in GRID.minutes if (p, t) not in missing}
    cells = {
        (p, t, b): DriftCell(p, t, b, 40, float(mu(p, t, b)), 0.1)
        for p in pools
        for t in GRID.minutes
        if (p, t) not in missing
        for b in range(len(BANDS) + 1)
    }
    return NowcastV0(BANDS, GRID, 50.0, ("2026-11-14",), sigma0, cells, fit_id)


def models_of(nowcast: NowcastV0) -> ShadowModels:
    ids = {"win": "win-a1", "ordering": "ord-b2", "nowcast": nowcast.fit_id}
    return ShadowModels(win=FakeWin(), ordering=FakeOrdering(), nowcast=nowcast, ids=ids)


# ---------------------------------------------------------------- independent references


def ref_band(odds: float) -> int:
    return bisect_right(list(BANDS), float(odds))


def ref_mu(mu: MuFn, pool: str, tau: float, band: int, missing=()) -> float:
    """The cell mean at any tau: linear between grid taus, the T-60 cell beyond, 0 at the post.
    Reads only the exact cell or the two bracketing it; KeyError when one of those is missing."""
    if tau <= 0:
        return 0.0
    grid = sorted(GRID.minutes)

    def at(t: int) -> float:
        if (pool, t) in missing:
            raise KeyError((pool, t))
        return mu(pool, t, band)

    if tau >= grid[-1]:
        return at(grid[-1])
    if tau in grid:
        return at(int(tau))
    lo = max([0, *(t for t in grid if t < tau)])
    hi = min(t for t in grid if t > tau)
    m0, m1 = (0.0 if lo == 0 else at(lo)), at(hi)
    return m0 + (tau - lo) / (hi - lo) * (m1 - m0)


def ref_projected(
    mu: MuFn, pool: str, odds_now: float, tau_now: float, k: int, missing=()
) -> float | None:
    """The projected median odds at post - k, or None when a cell it needs is missing."""
    b = ref_band(odds_now)
    try:
        shift = ref_mu(mu, pool, tau_now, b, missing) - ref_mu(mu, pool, k, b, missing)
    except KeyError:
        return None
    return max(1.0, odds_now * math.exp(shift))


def _key(row: dict) -> tuple[str, str]:
    return row["pool"], row["combination"]


def ref_available(row: dict) -> datetime:
    """When a row could first be known: its fetch time, else its capture time."""
    return instant(row.get("fetched_utc") or row["snapshot_utc"])


def _priced(row: dict) -> bool:
    return row.get("odds") is not None and row.get("status_raw") != "SCR"


def ref_ladders(
    rows: list[dict], race: Race, now: datetime, mu: MuFn | None, *, no_cells=(), missing=()
) -> dict[tuple[str, str], dict[str, dict | None]]:
    """{(pool, combination): {"20": entry or None, ...}} from the logged rows of `race` available
    (fetched) by now; each projection from the current row's own fetch time (tau_now = post - its
    fetched_utc, else its snapshot_utc); observed checkpoints chosen by snapshot_utc."""
    mine = [r for r in rows if r["race_id"] == race.race_id and r["pool"] in POOLS]
    mine = sorted(mine, key=lambda r: instant(r["snapshot_utc"]))  # stable: log order
    final_snaps = {
        instant(r["snapshot_utc"])
        for r in mine
        if r.get("source_update_utc") is None and FINAL_TEXT in (r.get("status_raw") or "")
    }
    out: dict[tuple[str, str], dict[str, dict | None]] = {}
    for key in sorted({_key(r) for r in mine if ref_available(r) <= now}):
        history = [r for r in mine if _key(r) == key and ref_available(r) <= now]
        pool = key[0]
        current = max(enumerate(history), key=lambda ir: (ref_available(ir[1]), ir[0]))[1]
        tau_now = max(0.0, (race.post - ref_available(current)) / timedelta(minutes=1))
        ladder: dict[str, dict | None] = {}
        for k in CHECKPOINTS:
            cutoff = race.post - timedelta(minutes=k)
            if k == 0:
                finals = [r for r in history if instant(r["snapshot_utc"]) in final_snaps]
                if finals:
                    ladder["0"] = {"odds": finals[0]["odds"], "observed": True}
                    continue
            elif now >= cutoff:
                before = [r for r in history if instant(r["snapshot_utc"]) <= cutoff]
                ref = before[-1] if before else None
                ok = ref is not None and _priced(ref)
                ladder[str(k)] = {"odds": ref["odds"], "observed": True} if ok else None
                continue
            odds = None
            if mu is not None and pool not in no_cells and _priced(current):
                odds = ref_projected(mu, pool, current["odds"], tau_now, k, missing)
            ladder[str(k)] = None if odds is None else {"odds": odds, "observed": False}
        out[key] = ladder
    return out


# ---------------------------------------------------------------- the synthetic day


def race(venue: str, no: int, minutes: float, base) -> Race:
    return Race(DAY, venue, no, T0 + timedelta(minutes=minutes), base)


def capture_final(day: SyntheticDay, r: Race, at: datetime, factor: float) -> None:
    """A capture of the final board: every row final (no source update, the final text)."""
    rows = capture_rows(r, at, factor)
    for row in rows:
        row["status_raw"] = FINAL_TEXT
        row["source_update_utc"] = None
    odds_log.append_capture(log_path(DAY, r.venue), rows)
    day.rows.extend(rows)


# (venue, no, minutes to post at now, base, captures as minutes before post; the last is at now)
CASES = [
    ("TOK", 1, 75.0, EIGHT, (90, 75)),  # beyond T-60: every checkpoint ahead, the T-60 cell
    ("TOK", 2, 45.0, SIX, (50, 45)),  # between T-60 and T-30
    ("TOK", 3, 25.0, EIGHT2, (40, 25)),  # T-20 ahead too: interpolated between 30 and 16
    ("TOK", 4, 12.5, EIGHT, (25, 19, 12.5)),  # before T-10: T-20 observed, the rest projected
    ("TOK", 5, 3.5, SIX, (25, 19, 12.5, 10, 7, 5, 3.5)),  # after T-5's capture (both exact)
    ("KYO", 1, 12.5, SIX, (15, 12.5)),  # T-20 passed with no capture before it: null
    ("KYO", 2, 1.5, EIGHT2, (20, 9, 4, 1.5)),  # T-2 observed; the final from tau 1.5
    ("KYO", 3, 8.0, EIGHT, (31, 8.0)),  # T-20 and T-10 from the same capture
]


def synthetic_day() -> tuple[SyntheticDay, list[Race]]:
    races = [race(v, no, minutes, base) for v, no, minutes, base, _ in CASES]
    gone = race("HAN", 1, -3.0, EIGHT2)  # post 3 min ago, its final board captured
    day = SyntheticDay(DAY, [*races, gone])
    for r, (*_, offsets) in zip(races, CASES, strict=True):
        for i, off in enumerate(offsets):
            day.capture(r, r.post - timedelta(minutes=off), FACTORS[i])
    for i, off in enumerate((25, 10, 4, 1)):
        day.capture(gone, gone.post - timedelta(minutes=off), FACTORS[i])
    capture_final(day, gone, gone.post + timedelta(minutes=1), 0.94)
    return day, [*races, gone]


def build(now: datetime, models, **options) -> dict:
    settings = dict(now=lambda: now, models=models, flag_rules=RULES, n_draws=N_DRAWS, seed=3)
    settings.update(options)
    return json.loads(json.dumps(build_shadow(DAY, **settings)))  # the feed is JSON-ready


def by_id(feed: dict) -> dict[str, dict]:
    return {r["race_id"]: r for r in feed["races"]}


def assert_ladders(got_race: dict, want: dict, where: str) -> dict[str, int]:
    """Every entry's ladder against the reference; counts of observed, projected, null, floored."""
    counts = {"observed": 0, "projected": 0, "null": 0, "floored": 0}
    pools = got_race["pools"]
    for (pool, combination), ladder in want.items():
        entry = pools[pool][combination]
        assert "ladder" in entry, (where, pool, combination, sorted(entry))
        shown = entry["ladder"]
        assert isinstance(shown, dict), (where, pool, combination, shown)
        assert set(shown) == {"20", "10", "5", "2", "0"}, (where, pool, combination, shown)
        for k, expected in ladder.items():
            got = shown[k]
            at = (where, pool, combination, k, got, expected)
            if expected is None:
                assert got is None, at
                counts["null"] += 1
                continue
            assert isinstance(got, dict) and got.get("observed") is expected["observed"], at
            if expected["observed"]:
                assert got["odds"] == expected["odds"], at
                counts["observed"] += 1
            else:
                assert got["odds"] == pytest.approx(expected["odds"], abs=TOL), at
                assert got["odds"] >= 1.0, at
                counts["projected"] += 1
                counts["floored"] += expected["odds"] == 1.0
    return counts


# ---------------------------------------------------------------- every race


@pytest.mark.parametrize("mu", [mu_a, mu_b], ids=["drift-a", "drift-b"])
def test_every_entry_has_its_ladder(mu):
    # the cases are written for the shipped D2 grid (T-20 between 16 and 30, T-10 between 5, 16)
    assert GRID.minutes == (60, 30, 16, 5, 2, 1)
    day, races = synthetic_day()
    feed = build(T0, models_of(nowcast_of(mu)))

    got = by_id(feed)
    assert set(got) == {r.race_id for r in races}
    totals = {"observed": 0, "projected": 0, "null": 0, "floored": 0}
    for r in races:
        want = ref_ladders(day.rows, r, T0, mu)
        assert {(p, c) for p in POOLS for c in got[r.race_id]["pools"][p]} == set(want), r.race_id
        for name, n in assert_ladders(got[r.race_id], want, r.race_id).items():
            totals[name] += n
    # the fixture exercises every kind of cell
    assert totals["observed"] > 50 and totals["projected"] > 100 and totals["null"] > 0, totals
    assert totals["floored"] > 0, totals


def test_observed_and_projected_follow_the_board_read():
    """The definition-of-done day: one race read before T-10, then after T-5's capture."""
    r = race("NAK", 4, 12.5, EIGHT)
    day = SyntheticDay(DAY, [r])
    offsets = (25, 19, 12.5, 10, 7, 5, 3.5)
    for i, off in enumerate(offsets):
        day.capture(r, r.post - timedelta(minutes=off), FACTORS[i])
    models = models_of(nowcast_of(mu_a))
    cache: dict = {}
    early = T0  # post - 12.5
    late = r.post - timedelta(minutes=3.5)  # the last capture, after T-5's (at post - 5)

    for now, passed in ((early, {"20"}), (late, {"20", "10", "5"})):
        feed = build(now, models, cache=cache)
        (got,) = feed["races"]
        want = ref_ladders(day.rows, r, now, mu_a)
        assert_ladders(got, want, f"{r.race_id} at {now.isoformat()}")
        for pool in POOLS:
            for combination, entry in got["pools"][pool].items():
                ladder = entry["ladder"]
                for k in ("20", "10", "5", "2", "0"):
                    assert ladder[k] is not None, (now, pool, combination, k)
                    assert ladder[k]["observed"] is (k in passed), (now, pool, combination, k)
        again = build(now, models, cache=cache)  # a cached rebuild shows the same ladders
        assert again["races"] == feed["races"]


# ---------------------------------------------------------------- the formula by hand


HAND_MU = {60: 0.5, 30: 0.40, 16: 0.26, 5: 0.15, 2: 0.08, 1: 0.05}  # WIN, band 1 (2 <= O < 5)


def hand_mu(pool: str, tau: int, band: int) -> float:
    if pool == WIN and band == 1:
        return HAND_MU[tau]
    if band == 0:
        return -0.8  # favourites shorten into the post: the final floors at 1.0
    return -0.3  # any other band: a wrong band shows


def test_the_projection_of_a_known_cell():
    """WIN runner 2 at 4.0 now (band 1); its T-20 capture is at 5.2 (band 2), runner 1 at 1.6."""
    base = (1.6, 4.0, 7.5, 13.0, 31.0, 64.0, 120.0, 210.0)
    far, mid, near = (race("SAP", no, m, base) for no, m in ((1, 75.0), (2, 12.5), (3, 3.5)))
    day = SyntheticDay(DAY, [far, mid, near])
    day.capture(far, T0, 1.0)
    for r in (mid, near):
        day.capture(r, r.post - timedelta(minutes=25), 1.3)
        day.capture(r, T0, 1.0)
    feed = build(T0, models_of(nowcast_of(hand_mu)))
    got = by_id(feed)

    def ladder(r: Race, combination: str) -> dict:
        return got[r.race_id]["pools"][WIN][combination]["ladder"]

    # mu(12.5) = 0.15 + 7.5/11 * 0.11 = 0.225, mu(10) = 0.20; mu(3.5) = 0.08 + 1.5/3 * 0.07 = 0.115
    # mu(75) = mu(60) = 0.5, mu(20) = 0.26 + 4/14 * 0.14 = 0.30; mu(0) = 0
    hand = {
        far: {"20": 4.88561, "10": 5.39944, "5": 5.67627, "2": 6.08785, "0": 6.59489},
        mid: {"10": 4.10126, "5": 4.31154, "2": 4.62416, "0": 5.00929},
        near: {"2": 4.14248, "0": 4.48749},
    }
    for r, want in hand.items():
        shown = ladder(r, "2")
        for k, odds in want.items():
            assert shown[k] is not None and shown[k]["observed"] is False, (r.race_id, k, shown)
            assert shown[k]["odds"] == pytest.approx(odds, abs=TOL), (r.race_id, k, shown)
    for r in (mid, near):
        assert ladder(r, "2")["20"] == {"odds": 5.2, "observed": True}, r.race_id
    assert ladder(near, "2")["10"] == {"odds": 5.2, "observed": True}
    assert ladder(near, "2")["5"] == {"odds": 5.2, "observed": True}
    # runner 1 (1.6, band 0, mu -0.8 at every tau): flat to T-2, the final floored at 1.0
    for r in (far, mid, near):
        shown = ladder(r, "1")
        assert shown["2"]["odds"] == pytest.approx(1.6, abs=TOL), (r.race_id, shown)
        assert shown["0"] == {"odds": pytest.approx(1.0, abs=1e-9), "observed": False}, shown


# ---------------------------------------------------------------- the final board


def test_the_final_is_observed_once_the_final_board_exists():
    day, races = synthetic_day()
    gone = races[-1]
    feed = build(T0, models_of(nowcast_of(mu_a)))
    got = by_id(feed)[gone.race_id]
    want = ref_ladders(day.rows, gone, T0, mu_a)
    counts = assert_ladders(got, want, gone.race_id)
    assert counts["projected"] == 0 and counts["observed"] == 5 * len(want), counts
    win_final = got["pools"][WIN]["1"]["ladder"]["0"]
    assert win_final == {"odds": round(gone.base[0] * 0.94, 2), "observed": True}
    # before the final capture (just after the post) the final is still projected
    before = by_id(build(gone.post + timedelta(seconds=30), models_of(nowcast_of(mu_a))))
    assert before[gone.race_id]["pools"][WIN]["1"]["ladder"]["0"]["observed"] is False


# ---------------------------------------------------------------- without a projection


def test_without_models_only_observed_checkpoints_show():
    day, races = synthetic_day()
    feed = build(T0, None)
    got = by_id(feed)
    nulls = 0
    for r in races:
        want = ref_ladders(day.rows, r, T0, None)
        counts = assert_ladders(got[r.race_id], want, r.race_id)
        assert counts["projected"] == 0, (r.race_id, counts)
        nulls += counts["null"]
    assert nulls > 100


def test_a_pool_the_nowcast_does_not_fit_shows_dashes_ahead():
    """The nowcast fits WIN and PLACE only: Wide keeps its observed checkpoints, null ahead."""
    day, races = synthetic_day()
    feed = build(T0, models_of(nowcast_of(mu_a, pools=(WIN, PLACE))))
    got = by_id(feed)
    projected = 0
    for r in races:
        want = ref_ladders(day.rows, r, T0, mu_a, no_cells=(WIDE,))
        assert_ladders(got[r.race_id], want, r.race_id)
        for ladder in (e["ladder"] for e in got[r.race_id]["pools"][WIDE].values()):
            assert all(v is None or v["observed"] for v in ladder.values()), (r.race_id, ladder)
        projected += sum(
            v is not None and not v["observed"]
            for e in got[r.race_id]["pools"][WIN].values()
            for v in e["ladder"].values()
        )
    assert projected > 20


# ---------------------------------------------------------------- the anchor of a projection


def test_the_projection_starts_from_the_capture_of_the_row_it_uses():
    """Staggered captures: every pool at post - 25 and post - 12, then Wide alone at post - 9; now
    is post - 7. WIN and PLACE project from tau 12 (their rows' capture), Wide from tau 9; neither
    from the Wide-only capture for WIN, nor from now (tau 7)."""
    r = race("CHU", 6, 7.0, EIGHT)
    day = SyntheticDay(DAY, [r])
    day.capture(r, r.post - timedelta(minutes=25), 1.0)
    day.capture(r, r.post - timedelta(minutes=12), 1.06)
    wide_only = [
        row for row in capture_rows(r, r.post - timedelta(minutes=9), 0.97) if row["pool"] == WIDE
    ]
    odds_log.append_capture(log_path(DAY, r.venue), wide_only)
    day.rows.extend(wide_only)

    (got,) = build(T0, models_of(nowcast_of(mu_a)))["races"]
    want = ref_ladders(day.rows, r, T0, mu_a)
    counts = assert_ladders(got, want, r.race_id)
    assert counts["projected"] > 50, counts

    anchors = {WIN: 12.0, PLACE: 12.0, WIDE: 9.0}
    told = {WIN: 0, PLACE: 0, WIDE: 0}  # projections a wrong anchor would miss
    for pool, combination in want:
        current = [
            row for row in day.rows if (row["pool"], row["combination"]) == (pool, combination)
        ][-1]
        assert (r.post - instant(current["snapshot_utc"])) / timedelta(minutes=1) == anchors[pool]
        shown = got["pools"][pool][combination]["ladder"]
        for k in ("5", "2", "0"):
            assert shown[k] is not None and shown[k]["observed"] is False, (pool, combination, k)
            right = ref_projected(mu_a, pool, current["odds"], anchors[pool], int(k))
            assert shown[k]["odds"] == pytest.approx(right, abs=TOL), (pool, combination, k)
            wrong = {ref_projected(mu_a, pool, current["odds"], tau, int(k)) for tau in (9.0, 7.0)}
            told[pool] += all(abs(w - right) > 3 * TOL for w in wrong if w is not None)
    assert told[WIN] >= 10 and told[PLACE] >= 5, told  # the fixture tells the anchors apart


def capture_late(day: SyntheticDay, r: Race, snap: datetime, fetched: datetime, factor: float,
                 pools=POOLS) -> None:  # fmt: skip
    """A capture stamped at `snap` whose odds pages were fetched at `fetched` (the native logger
    stamps the snapshot first), only the rows of `pools`."""
    rows = [row for row in capture_rows(r, snap, factor) if row["pool"] in pools]
    for row in rows:
        row["fetched_utc"] = fetched.astimezone(UTC).isoformat()
    odds_log.append_capture(log_path(DAY, r.venue), rows)
    day.rows.extend(rows)


def test_the_projection_starts_from_the_fetch_of_the_price_it_uses():
    """Captures stamped before their pages were fetched; now is T0 for every race.
    - CHU 7 (post - 7 at now): every pool snapped at post - 12, fetched at post - 11; a Wide-only
      capture snapped at post - 9, fetched at post - 8. WIN and PLACE project from tau 11, Wide
      from tau 8 (not 12 or 9 from the stamps, not 7 from now).
    - CHU 8 (post - 7): snapped at post - 17, fetched at post - 15.5: tau 15.5 (below the T-16
      grid cell, where the stamp's 17 is above it).
    - CHU 9 (post - 4): snapped at post - 6, fetched at post - 5.5: tau 5.5; its T-5 is observed
      from that row (chosen by its stamp, at or before post - 5).
    - CHU 10 (post - 7): snapped at post - 9 but fetched at post - 6.5, after now: not on the board
      yet, so O_now and tau 25 come from the post - 25 capture and T-10 is observed from it."""
    lagged = race("CHU", 7, 7.0, EIGHT)
    crossing = race("CHU", 8, 7.0, SIX)
    near = race("CHU", 9, 4.0, EIGHT2)
    unseen = race("CHU", 10, 7.0, EIGHT)
    day = SyntheticDay(DAY, [lagged, crossing, near, unseen])

    def at(r: Race, minutes: float) -> datetime:
        return r.post - timedelta(minutes=minutes)

    for r, first in ((lagged, 25), (crossing, 30), (near, 25), (unseen, 25)):
        day.capture(r, at(r, first), 1.0)
    capture_late(day, lagged, at(lagged, 12), at(lagged, 11), 1.06)
    capture_late(day, lagged, at(lagged, 9), at(lagged, 8), 0.97, pools=(WIDE,))
    capture_late(day, crossing, at(crossing, 17), at(crossing, 15.5), 1.11)
    capture_late(day, near, at(near, 6), at(near, 5.5), 0.92)
    capture_late(day, unseen, at(unseen, 9), at(unseen, 6.5), 1.08)

    # (race, pool) -> (the anchor wanted, the anchors a wrong reading would use)
    anchors = {
        (lagged, WIN): (11.0, (12.0, 9.0, 8.0, 7.0)),
        (lagged, PLACE): (11.0, (12.0, 9.0, 8.0, 7.0)),
        (lagged, WIDE): (8.0, (9.0, 11.0, 12.0, 7.0)),
        (crossing, WIN): (15.5, (17.0, 7.0)),
        (crossing, PLACE): (15.5, (17.0, 7.0)),
        (crossing, WIDE): (15.5, (17.0, 7.0)),
        (near, WIN): (5.5, (6.0, 4.0)),
        (near, PLACE): (5.5, (6.0, 4.0)),
        (near, WIDE): (5.5, (6.0, 4.0)),
        (unseen, WIN): (25.0, (9.0, 6.5, 7.0)),
        (unseen, PLACE): (25.0, (9.0, 6.5, 7.0)),
        (unseen, WIDE): (25.0, (9.0, 6.5, 7.0)),
    }
    ahead = {lagged: ("5", "2", "0"), crossing: ("5", "2", "0"), near: ("2", "0"),
             unseen: ("5", "2", "0")}  # fmt: skip
    nowcast = models_of(nowcast_of(mu_a))
    got = by_id(build(T0, nowcast))
    told: dict[tuple[str, str], int] = {}
    wants = {r: ref_ladders(day.rows, r, T0, mu_a) for r in (lagged, crossing, near, unseen)}
    for r, want in wants.items():
        for (pool, combination), ladder in want.items():
            right_tau, wrong_taus = anchors[(r, pool)]
            history = [
                row
                for row in day.rows
                if row["race_id"] == r.race_id
                and (row["pool"], row["combination"]) == (pool, combination)
                and ref_available(row) <= T0
            ]
            current = history[-1]
            assert (r.post - ref_available(current)) / timedelta(minutes=1) == right_tau
            for k in ahead[r]:
                right = ref_projected(mu_a, pool, current["odds"], right_tau, int(k))
                assert ladder[k] == {"odds": right, "observed": False}, (r.race_id, pool, k)
                wrong = [ref_projected(mu_a, pool, current["odds"], t, int(k)) for t in wrong_taus]
                apart = all(abs(w - right) > 3 * TOL for w in wrong if w is not None)
                told[(r.race_id, pool)] = told.get((r.race_id, pool), 0) + apart
        # the fixture tells the anchors apart in every pool of every race
        for pool in POOLS:
            assert told.get((r.race_id, pool), 0) >= 3, (r.race_id, pool, told)
    for r, want in wants.items():
        counts = assert_ladders(got[r.race_id], want, r.race_id)
        assert counts["projected"] >= 2 * len(want), (r.race_id, counts)

    # the observed checkpoints keep the stamp rule among the rows fetched by now
    win1 = {r: got[r.race_id]["pools"][WIN]["1"]["ladder"] for r in (near, unseen, lagged)}
    assert win1[near]["5"] == {"odds": round(EIGHT2[0] * 0.92, 2), "observed": True}
    assert win1[unseen]["10"] == {"odds": EIGHT[0], "observed": True}
    assert win1[lagged]["10"] == {"odds": round(EIGHT[0] * 1.06, 2), "observed": True}


# ---------------------------------------------------------------- a nowcast with a missing cell


def test_a_missing_grid_cell_blanks_only_the_checkpoints_that_need_it():
    """A fit without WIN cells at T-60 still projects WIN from tau 7 (cells 5 and 16) to T-5, T-2
    and the final; tau 45 (between 30 and 60) and tau 75 (beyond 60) need the T-60 cell, so WIN
    is null ahead there while PLACE and Wide project. A fit without WIN at T-5, read at tau 25:
    T-20 (cells 16, 30), T-2 (cell 2) and the final project; T-10 (cells 5, 16) and T-5 are null."""
    near = race("FUK", 1, 7.0, EIGHT)
    mid = race("FUK", 2, 45.0, SIX)
    far = race("FUK", 3, 75.0, EIGHT2)
    early = race("NII", 1, 25.0, EIGHT)
    day = SyntheticDay(DAY, [near, mid, far, early])
    for r, offsets in ((near, (25, 12, 7)), (mid, (50, 45)), (far, (90, 75)), (early, (40, 25))):
        for i, off in enumerate(offsets):
            day.capture(r, r.post - timedelta(minutes=off), FACTORS[i])

    def win_ladders(feed: dict, r: Race) -> dict[str, dict]:
        return {c: e["ladder"] for c, e in by_id(feed)[r.race_id]["pools"][WIN].items()}

    no60 = frozenset({(WIN, 60)})
    feed = build(T0, models_of(nowcast_of(mu_a, missing=no60)))
    for r in (near, mid, far, early):
        assert_ladders(by_id(feed)[r.race_id], ref_ladders(day.rows, r, T0, mu_a, missing=no60),
                       f"no WIN T-60: {r.race_id}")  # fmt: skip
    for combination, ladder in win_ladders(feed, near).items():
        for k in ("5", "2", "0"):
            assert ladder[k] is not None and ladder[k]["observed"] is False, (combination, k)
    for r in (mid, far):
        for ladder in win_ladders(feed, r).values():
            assert all(ladder[k] is None for k in ("20", "10", "5", "2", "0")), (r.race_id, ladder)
        for pool in (PLACE, WIDE):
            for combination, entry in by_id(feed)[r.race_id]["pools"][pool].items():
                assert entry["ladder"]["5"] is not None, (r.race_id, pool, combination)
                assert entry["ladder"]["5"]["observed"] is False, (r.race_id, pool, combination)

    no5 = frozenset({(WIN, 5)})
    feed = build(T0, models_of(nowcast_of(mu_a, missing=no5)))
    for r in (near, mid, far, early):
        assert_ladders(by_id(feed)[r.race_id], ref_ladders(day.rows, r, T0, mu_a, missing=no5),
                       f"no WIN T-5: {r.race_id}")  # fmt: skip
    for combination, ladder in win_ladders(feed, early).items():
        for k in ("20", "2", "0"):
            assert ladder[k] is not None and ladder[k]["observed"] is False, (combination, k)
        assert ladder["10"] is None and ladder["5"] is None, (combination, ladder)
    for combination, ladder in win_ladders(feed, mid).items():  # tau 45 needs only 30 and 60
        assert all(ladder[k] is not None for k in ("20", "2", "0")), (combination, ladder)
