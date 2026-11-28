"""The sealed ladder log that shadow-day writes beside the shadow log (ticket #628).

Contract pinned here:

`horsetrack.countries.japan.shadow.ladder_log`:
- `ladder_log_path(day) -> data_dir()/shadow/japan/<YYYY-MM-DD>.ladder.jsonl` (the data dir read on
  every call), beside the shadow log of the same day.
- `LadderLog(path, *, day)`: the same hash chain, seal and `day_status` as `shadow.log.ShadowLog`.
  On disk every line is the log's compact JSON `{"seq", "prev_sha256", "record"}`, seq from 1,
  prev_sha256 the sha256 of the previous line's bytes (64 zeros on the first), and a sealed day
  ends with `{"seq", "prev_sha256", "seal": {"n_records", "sealed_utc"}}`. `append(records)`,
  `seal(sealed_utc=...)`, `read()` (the records of a development day, in order) and `verify()`
  (with `.sealed` and `.day_status`), as ShadowLog. A file whose chain is broken raises
  `shadow.log.ShadowLogTampered` (or a subclass); an append to a sealed log raises
  `shadow.log.LogSealed` (or a subclass). The log sets `day_status` (holdout by the committed
  holdout rule, else development).

`shadow.day.run_shadow_day(..., ladder_log: LadderLog | None = None)`: given a ladder log, the run
also writes it. The `shadow-day` command always writes `ladder_log_path(date)`.
- Ladder checkpoints k in (20, 10, 5, 2) of each race: due at post - k + grace_s. Each one writes
  one record per (pool, combination) of WIN, PLACE and Wide on the board at k, with
  `race_id`, `pool`, `combination`, `checkpoint_min` = k, `status` "ok", `odds` = the low figure of
  the latest row of the key captured at or before post - k (by `snapshot_utc`, `_DayRows.refs`'
  rule), `fetched_utc` = that row's fetch time, `projected` = {str(j): odds or null} for every
  later checkpoint j < k and "0" (the final), each `board.shadow_feed._projector(models)(pool,
  odds, tau, j)` with tau the minutes from that row's fetch to the post, `nowcast_id` =
  models.ids["nowcast"] (null without models, and every projection null), and `mode` "live" or
  "replay" by the shadow log's max_lag_s rule.
- A checkpoint with no board within its board-age limit (`max_board_age_s`: a float for every
  checkpoint, or the per-tau mapping) is one record with `status` "no_board".
- Once the race's final board is in (`snapshot.final_board`), one record per (pool, combination)
  with `checkpoint_min` 0 and `odds` = the final board's low figure.
- The log is sealed once every race's checkpoints and final are written; a rerun of a sealed day
  writes nothing to either log, and a restart appends only the points still missing (also after
  the shadow log is already sealed and a final is still to come).
- The shadow log's records are the same with or without the ladder log.

The page reference is `board.shadow_feed.build_shadow`'s ladder read just after each checkpoint.
Every other expected figure is derived here from the synthetic rows and the known nowcast cells.
Race two's logger stamps each capture 30 s before it fetches it, so the latest row captured by
post - k is fetched 10 s after post - k, inside the grace period. All rows are synthetic and
written under tmp_path; the clock is fake and nothing waits.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import hashlib
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board.flags import FlagRules
from horsetrack.countries.japan.board.shadow_feed import build_shadow
from horsetrack.countries.japan.board.snapshot import load_decision_grid
from horsetrack.countries.japan.models.v0.calibration import WinCalibration
from horsetrack.countries.japan.models.v0.nowcast import DriftCell, NowcastV0
from horsetrack.countries.japan.models.v0.ordering import OrderingFit
from horsetrack.countries.japan.models.v0.shadow_values import ShadowModels
from horsetrack.countries.japan.shadow import day as shadow_day
from horsetrack.countries.japan.shadow.day import run_shadow_day
from horsetrack.countries.japan.shadow.ladder_log import LadderLog, ladder_log_path
from horsetrack.countries.japan.shadow.log import (
    LogSealed,
    ShadowLog,
    ShadowLogTampered,
    shadow_log_path,
)

UTC = timezone.utc
DAY = date(2026, 11, 22)  # a development day: before the holdout start
WIN, PLACE, WIDE = "WIN", "PLACE", "QUINELLA_PLACE"
POOLS = (WIN, PLACE, WIDE)
CHECKPOINTS = (20, 10, 5, 2)
FINAL = 0
FINAL_TEXT = "最終オッズ"
GRACE = 15.0
POLL = 10.0
N_DRAWS = 8
SEED = 5
TOL = 0.0051  # the projector rounds to two decimals
RULES = FlagRules(floor_max_odds=1.3, tail_min_odds=50.0, tail_rank=6, stale_s=40.0)
BANDS = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
GRID = load_decision_grid()
SIX = (1.8, 3.4, 6.2, 12.5, 38.0, 85.0)
EIGHT = (1.25, 2.9, 4.6, 7.3, 11.8, 26.0, 64.0, 152.0)
FINAL_FACTOR = 1.093


class Killed(BaseException):
    """The runner's process is killed (a BaseException, so no `except Exception` swallows it)."""


def at_utc(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm), UTC)


def iso(at: datetime) -> str:
    return at.astimezone(UTC).isoformat()


def instant(text: str) -> datetime:
    return datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)


def later(k: int) -> list[int]:
    return [j for j in CHECKPOINTS if j < k] + [FINAL]


# ---------------------------------------------------------------- the synthetic day


def minute_captures(first: int = 25, last: int = -1, *, lag_s: float = 2.0, skip=()) -> tuple:
    """(snapshot, fetched) offsets in seconds from the post, one capture a minute: fetched at
    post - m min - 30 s + (lag_s - 2) and snapped lag_s before it, m from `first` down to `last`."""
    out = []
    for m in range(first, last - 1, -1):
        if m in skip:
            continue
        fetched = -(m * 60 + 30) + (lag_s - 2.0)
        out.append((fetched - lag_s, fetched))
    return tuple(out)


LAGGED = minute_captures(lag_s=30.0)  # snapped post - m min - 20 s, fetched post - m min + 10 s


@dataclass(frozen=True)
class Race:
    day: date
    venue: str
    no: int
    post: datetime
    base: tuple[float, ...]
    captures: tuple[tuple[float, float], ...] = field(default_factory=minute_captures)
    final_s: float | None = 180.0  # the final board fetched this long after the post; None: never

    @property
    def race_id(self) -> str:
        return f"jra:{self.day.isoformat()}:{self.venue}:{self.no:02d}"

    def keys(self) -> set[tuple[str, str]]:
        horses = range(1, len(self.base) + 1)
        return (
            {(WIN, str(h)) for h in horses}
            | {(PLACE, str(h)) for h in horses}
            | {(WIDE, f"{a},{b}") for a, b in combinations(horses, 2)}
        )

    def due(self, k: int) -> datetime:
        return self.post - timedelta(minutes=k) + timedelta(seconds=GRACE)


def race_one(day: date = DAY, **changes) -> Race:
    fields = {"day": day, "venue": "TOK", "no": 1, "post": at_utc(day, 1), "base": SIX}
    return Race(**{**fields, **changes})


def race_two(day: date = DAY, **changes) -> Race:
    fields = {"day": day, "venue": "TOK", "no": 2, "post": at_utc(day, 1, 40), "base": EIGHT}
    return Race(**{**fields, "captures": LAGGED, **changes})


def capture_rows(race: Race, snap: datetime, fetched: datetime, factor: float, final: bool):
    win = {h: round(o * factor, 3) for h, o in enumerate(race.base, 1)}

    def row(pool: str, combination: str, odds: float, hi: float | None = None) -> dict:
        out = {
            "snapshot_utc": iso(snap),
            "fetched_utc": iso(fetched),
            "race_id": race.race_id,
            "pool": pool,
            "combination": combination,
            "odds": odds,
            "status_raw": FINAL_TEXT if final else "10時00分現在",
            "page_state": None,
            "source_update_utc": None if final else iso(snap - timedelta(seconds=10)),
        }
        if hi is not None:
            out["odds_hi"] = hi
        return out

    rows = [row(WIN, str(h), o) for h, o in win.items()]
    for h, o in win.items():
        low = round(1.0 + (o - 1.0) * 0.22, 3)
        rows.append(row(PLACE, str(h), low, round(low * 1.7, 3)))
    for a, b in combinations(sorted(win), 2):
        low = round(1.05 + 0.12 * (win[a] + win[b]), 3)
        rows.append(row(WIDE, f"{a},{b}", low, round(low * 1.45, 3)))
    return rows


def log_path(day: date, venue: str) -> Path:
    return paths.odds_snapshot_path("jra", day.isoformat(), venue)


class World:
    """The day's captures, appended to the odds logs (one member per capture) as the fake clock
    reaches each one's fetch time."""

    def __init__(self, races: list[Race]):
        self.races = races
        self.day = races[0].day
        members: list[tuple[datetime, str, list[dict]]] = []
        for race in races:
            for i, (snap_s, fetched_s) in enumerate(race.captures):
                factor = 1.0 + 0.0137 * ((7 * i + 3 * race.no) % 19)  # every capture moves
                snap = race.post + timedelta(seconds=snap_s)
                fetched = race.post + timedelta(seconds=fetched_s)
                rows = capture_rows(race, snap, fetched, factor, final=False)
                members.append((fetched, race.venue, rows))
            if race.final_s is not None:
                fetched = race.post + timedelta(seconds=race.final_s)
                rows = capture_rows(
                    race, fetched - timedelta(seconds=2), fetched, FINAL_FACTOR, final=True
                )
                members.append((fetched, race.venue, rows))
        self.pending = sorted(members, key=lambda m: (m[0], m[1]))
        self.rows = [row for _, _, rows in self.pending for row in rows]
        # (race_id, pool, combination) -> [(snapshot_utc, row)] in log order, finals apart
        self.history: dict[tuple, list[tuple[datetime, dict]]] = defaultdict(list)
        self.finals: dict[tuple, list[dict]] = defaultdict(list)
        for row in self.rows:
            key = (row["race_id"], row["pool"], row["combination"])
            if row["status_raw"] == FINAL_TEXT:
                self.finals[key].append(row)
            else:
                self.history[key].append((instant(row["snapshot_utc"]), row))
        for venue in sorted({r.venue for r in races}):
            posts = {r.no: r.post for r in races if r.venue == venue}
            sidecar = log_path(self.day, venue).with_name(f"{venue}.races.json")
            odds_log.write_post_times(sidecar, posts, min(posts.values()) - timedelta(hours=4))

    def write_until(self, t: datetime) -> None:
        while self.pending and self.pending[0][0] <= t:
            _, venue, rows = self.pending.pop(0)
            odds_log.append_capture(log_path(self.day, venue), rows)

    def race(self, race_id: str) -> Race:
        return next(r for r in self.races if r.race_id == race_id)


class Clock:
    """now() is the fake time; sleep(s) moves it and lets the synthetic logger catch up."""

    def __init__(self, world: World, start: datetime, *, kill_at: datetime | None = None):
        self.world, self.t, self.kill_at = world, start, kill_at
        self.limit = at_utc(world.day, 8, 30)  # 17:30 in Tokyo: a runner still polling hangs
        self.sleeps = 0
        world.write_until(start)

    def now(self) -> datetime:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        if self.sleeps > 20_000:
            raise RuntimeError("the runner slept 20000 times without ending the day")
        new = self.t + timedelta(seconds=max(0.0, float(seconds)))
        if self.kill_at is not None and new >= self.kill_at:
            raise Killed()
        if new > self.limit:
            raise RuntimeError(f"the runner kept polling until {new.isoformat()}")
        self.t = new
        self.world.write_until(new)


def start_of(day: date) -> datetime:
    return at_utc(day, 0, 33)  # 27 min before race one's post


def late_of(day: date) -> datetime:
    return at_utc(day, 3)  # 12:00 in Tokyo: every race run and its final board in


# ---------------------------------------------------------------- known nowcasts and models

MuFn = Callable[[str, int, int], float]
BAND_A = {0: -0.9, 1: -0.4, 2: 0.25, 3: 0.5, 4: 0.8, 5: 1.1, 6: 1.3}
POOL_A = {WIN: 1.0, PLACE: 0.6, WIDE: 0.8}
BAND_B = {0: 0.9, 1: 0.9, 2: -0.2, 3: -0.6, 4: 0.4, 5: -1.0, 6: 0.7}
POOL_B = {WIN: -0.7, PLACE: 1.2, WIDE: 0.5}


def mu_a(pool: str, tau: int, band: int) -> float:
    return POOL_A[pool] * BAND_A[band] * math.sqrt(tau) / 8.0


def mu_b(pool: str, tau: int, band: int) -> float:
    return POOL_B[pool] * BAND_B[band] * math.log1p(tau) / 4.0


def nowcast_of(mu: MuFn, pools=POOLS, fit_id: str = "now-test") -> NowcastV0:
    """A fitted v0 nowcast with the cell means mu(pool, tau, band) on every grid tau and band of
    `pools`, and no cells for any other pool."""
    sigma0 = {(p, t): 0.12 for p in pools for t in GRID.minutes}
    cells = {
        (p, t, b): DriftCell(p, t, b, 40, float(mu(p, t, b)), 0.1)
        for p in pools
        for t in GRID.minutes
        for b in range(len(BANDS) + 1)
    }
    return NowcastV0(BANDS, GRID, 50.0, ("2026-11-14",), sigma0, cells, fit_id)


class FakeWin:
    def predict(self, q, race=None):
        w = np.asarray(q, dtype=float) ** 1.3
        return w / w.sum()


class FakeOrdering:
    def place_probs(self, p, places):
        return np.minimum(1.0, places * np.asarray(p, dtype=float))

    def pair_probs(self, p, places):
        p = np.asarray(p, dtype=float)
        m = np.minimum(1.0, 3.0 * places * np.outer(p, p))
        np.fill_diagonal(m, 0.0)
        return m


def models_of(nowcast: NowcastV0) -> ShadowModels:
    ids = {"win": "win-a1", "ordering": "ord-b2", "nowcast": nowcast.fit_id}
    return ShadowModels(win=FakeWin(), ordering=FakeOrdering(), nowcast=nowcast, ids=ids)


# ---------------------------------------------------------------- independent references


def ref_mu(mu: MuFn, pool: str, tau: float, band: int) -> float:
    """The cell mean at any tau: linear between grid taus, the largest tau's cell beyond it, 0 at
    the post."""
    if tau <= 0:
        return 0.0
    grid = sorted(GRID.minutes)
    if tau >= grid[-1]:
        return mu(pool, grid[-1], band)
    if tau in grid:
        return mu(pool, int(tau), band)
    lo = max([0, *(t for t in grid if t < tau)])
    hi = min(t for t in grid if t > tau)
    m0, m1 = (0.0 if lo == 0 else mu(pool, lo, band)), mu(pool, hi, band)
    return m0 + (tau - lo) / (hi - lo) * (m1 - m0)


def ref_projected(mu: MuFn | None, pools, pool: str, odds: float, tau: float, k: int):
    """The projected median odds at post - k from `odds` read tau minutes before the post; None
    without a nowcast or without cells for the pool."""
    if mu is None or pool not in pools:
        return None
    band = bisect_right(list(BANDS), float(odds))
    shift = ref_mu(mu, pool, tau, band) - ref_mu(mu, pool, k, band)
    return max(1.0, odds * math.exp(shift))


def observed_row(world: World, race: Race, pool: str, combination: str, k: int) -> dict | None:
    """The latest row of the key captured (snapshot_utc) at or before post - k, never a final."""
    cutoff = race.post - timedelta(minutes=k)
    rows = [
        (snap, i, r) for i, (snap, r) in enumerate(world.history[(race.race_id, pool, combination)])
    ]
    rows = [entry for entry in rows if entry[0] <= cutoff]
    return max(rows, key=lambda e: e[:2])[2] if rows else None


def final_row(world: World, race: Race, pool: str, combination: str) -> dict:
    (row,) = world.finals[(race.race_id, pool, combination)]
    return row


def tau_of(race: Race, row: dict) -> float:
    return max(0.0, (race.post - instant(row["fetched_utc"])) / timedelta(minutes=1))


# ---------------------------------------------------------------- running the day


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


def open_shadow(day: date = DAY) -> ShadowLog:
    return ShadowLog(shadow_log_path(day), day=day)


def open_ladder(day: date = DAY) -> LadderLog:
    return LadderLog(ladder_log_path(day), day=day)


def run(clock: Clock, *, models, day: date = DAY, ladder: bool = True, **options):
    settings = dict(
        venues=None,
        now=clock.now,
        sleep=clock.sleep,
        poll_s=POLL,
        grace_s=GRACE,
        max_board_age_s=120.0,
        max_lag_s=120.0,
        once=False,
        flag_rules=RULES,
        n_draws=N_DRAWS,
        seed=SEED,
    )
    settings.update(options)
    if ladder:
        settings["ladder_log"] = open_ladder(day)
    return run_shadow_day(day, models=models, log=open_shadow(day), **settings)


def raw_lines(day: date = DAY) -> list[bytes]:
    return ladder_log_path(day).read_bytes().splitlines()


def raw_records(day: date = DAY) -> list[dict]:
    """The records of the file, read here without the module under test."""
    out = []
    for line in raw_lines(day):
        obj = json.loads(line)
        if "record" in obj:
            out.append(obj["record"])
    return out


def points(records: list[dict]) -> dict[tuple[str, int], list[dict]]:
    out: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        out[(record["race_id"], record["checkpoint_min"])].append(record)
    return out


def assert_complete(world: World, records: list[dict], no_board=frozenset()) -> None:
    """Every (race, checkpoint) and final once: one ok record per key, or one no_board record."""
    got = points(records)
    want = {(r.race_id, k) for r in world.races for k in (*CHECKPOINTS, FINAL)}
    assert set(got) == want
    for race in world.races:
        for k in (*CHECKPOINTS, FINAL):
            point = got[(race.race_id, k)]
            if (race.race_id, k) in no_board:
                assert [r["status"] for r in point] == ["no_board"], (race.race_id, k)
                continue
            keys = Counter((r["pool"], r["combination"]) for r in point)
            assert set(keys) == race.keys(), (race.race_id, k)
            assert set(keys.values()) == {1}, (race.race_id, k, "a key was written twice")
            if k != FINAL:
                assert {r["status"] for r in point} == {"ok"}, (race.race_id, k)
    key = Counter((r["race_id"], r["pool"], r["combination"], r["checkpoint_min"]) for r in records)
    assert set(key.values()) == {1}


def assert_checkpoint(world: World, record: dict, mu: MuFn | None, pools) -> None:
    """A checkpoint record's observed odds, fetch time and projections against the references."""
    race = world.race(record["race_id"])
    k, pool, combination = record["checkpoint_min"], record["pool"], record["combination"]
    row = observed_row(world, race, pool, combination, k)
    where = (race.race_id, pool, combination, k)
    assert row is not None, where
    assert record["odds"] == row["odds"], where
    assert instant(record["fetched_utc"]) == instant(row["fetched_utc"]), where
    projected = record["projected"]
    assert isinstance(projected, dict) and set(projected) == {str(j) for j in later(k)}, where
    for j in later(k):
        want = ref_projected(mu, pools, pool, row["odds"], tau_of(race, row), j)
        got = projected[str(j)]
        if want is None:
            assert got is None, (*where, j, got)
        else:
            assert got == pytest.approx(want, abs=TOL), (*where, j, got, want)


# ---------------------------------------------------------------- every checkpoint and the final

DRIFTS = {
    "drift a, every pool": (mu_a, POOLS),
    "drift b, no Wide cells": (mu_b, (WIN, PLACE)),
}


@pytest.mark.parametrize("case", list(DRIFTS.values()), ids=list(DRIFTS))
def test_each_checkpoint_logs_its_observed_odds_and_projections_and_each_key_its_final(case):
    mu, pools = case
    world = World([race_one(), race_two()])
    models = models_of(nowcast_of(mu, pools, fit_id=f"now-{mu.__name__}"))

    run(Clock(world, start_of(DAY)), models=models)

    records = open_ladder().read()
    assert records == raw_records()
    assert_complete(world, records)
    assert open_ladder().verify().sealed is True
    assert open_shadow().verify().sealed is True
    observed, projected = set(), 0
    projectable = sum(1 for r in world.races for pool, _ in r.keys() if pool in pools)
    for record in records:
        assert record["day_status"] == "development"
        race = world.race(record["race_id"])
        if record["checkpoint_min"] == FINAL:
            want = final_row(world, race, record["pool"], record["combination"])
            assert record["odds"] == want["odds"], record
            continue
        assert_checkpoint(world, record, mu, pools)
        assert record["nowcast_id"] == f"now-{mu.__name__}"
        assert record["mode"] == "live"  # each decided on the first poll after its grace
        observed.add((record["race_id"], record["pool"], record["combination"], record["odds"]))
        projected += sum(v is not None for v in record["projected"].values())
    # the fixture moves every price between checkpoints and projects most of them
    assert len(observed) == sum(len(r.keys()) for r in world.races) * len(CHECKPOINTS)
    assert projected == projectable * sum(len(later(k)) for k in CHECKPOINTS)


def test_without_models_the_observed_odds_stay_and_every_projection_is_null():
    world = World([race_one(), race_two()])

    run(Clock(world, late_of(DAY)), models=None, once=True)

    records = open_ladder().read()
    assert_complete(world, records)
    for record in records:
        if record["checkpoint_min"] == FINAL:
            continue
        assert_checkpoint(world, record, None, ())
        assert record["nowcast_id"] is None
        assert record["mode"] == "replay"  # written long after its checkpoint


# ---------------------------------------------------------------- the page's projector


@pytest.mark.parametrize("case", list(DRIFTS.values()), ids=list(DRIFTS))
def test_the_projections_equal_the_pages_ladder_read_at_each_checkpoint(case):
    mu, pools = case
    world = World([race_one(), race_two()])
    models = models_of(nowcast_of(mu, pools))
    world.write_until(late_of(DAY))

    run(Clock(world, late_of(DAY)), models=models, once=True)

    got = points(open_ladder().read())
    compared = compared_previous = 0
    for race in world.races:
        for k in CHECKPOINTS:
            # just after the checkpoint's last row is fetched, before the next capture
            now = race.post - timedelta(minutes=k) + timedelta(seconds=12)
            feed = build_shadow(
                DAY, now=lambda now=now: now, models=models, flag_rules=RULES, n_draws=N_DRAWS
            )
            (page,) = [r for r in feed["races"] if r["race_id"] == race.race_id]
            for record in got[(race.race_id, k)]:
                ladder = page["pools"][record["pool"]][record["combination"]]["ladder"]
                shown = {str(j): v for j, v in ladder.items()}
                where = (race.race_id, record["pool"], record["combination"], k)
                at_k = shown[str(k)]
                assert at_k["odds"] == record["odds"] and at_k["observed"] is True, where
                assert "projected" in at_k, where  # #627: what the previous rung projected for k
                if k == CHECKPOINTS[0]:
                    # the previous rung (post - 30) has no record in the ladder log
                    assert at_k["projected"] is None or isinstance(at_k["projected"], float), where
                else:
                    j = CHECKPOINTS[CHECKPOINTS.index(k) - 1]
                    (before,) = [
                        r
                        for r in got[(race.race_id, j)]
                        if (r["pool"], r["combination"]) == (record["pool"], record["combination"])
                    ]
                    assert at_k["projected"] == before["projected"][str(k)], (*where, j)
                    compared_previous += before["projected"][str(k)] is not None
                for j in later(k):
                    entry = shown[str(j)]
                    want = None if entry is None else entry["odds"]
                    assert entry is None or entry["observed"] is False, (*where, j)
                    assert record["projected"][str(j)] == want, (*where, j)
                    compared += want is not None
    assert compared > 100
    assert compared_previous > 50


# ---------------------------------------------------------------- no board within the limit

NO_BOARD_CASES = {
    # board ages at the checkpoints: 30 s, 30 s, 50 s, 50 s; T-2's limit is 45 s, T-5's 60 s
    "per-tau limits": (
        dict(captures=minute_captures(last=6) + ((-352.0, -350.0), (-172.0, -170.0), (28.0, 30.0))),
        {60: 600, 30: 300, 20: 240, 16: 180, 10: 120, 5: 60, 2: 45, 1: 30},
        {2},
    ),
    "no capture before T-20": (dict(captures=minute_captures(first=15)), 120.0, {20}),
    "a gap before T-10": (dict(captures=minute_captures(skip=range(10, 14))), 120.0, {10}),
}


@pytest.mark.parametrize("case", list(NO_BOARD_CASES.values()), ids=list(NO_BOARD_CASES))
def test_a_checkpoint_without_a_board_within_its_limit_is_one_no_board_record(case):
    changes, limits, missing = case
    one, two = race_one(), race_two(**{"captures": minute_captures(), **changes})
    world = World([one, two])
    models = models_of(nowcast_of(mu_a))

    run(Clock(world, start_of(DAY)), models=models, max_board_age_s=limits)

    records = open_ladder().read()
    assert_complete(world, records, no_board={(two.race_id, k) for k in missing})
    for record in records:
        if record["status"] == "ok" and record["checkpoint_min"] != FINAL:
            assert_checkpoint(world, record, mu_a, POOLS)
    assert open_ladder().verify().sealed is True


def test_the_shipped_board_age_limits_apply_to_every_checkpoint():
    world = World([race_one(), race_one(no=3, post=at_utc(DAY, 1, 20), base=EIGHT)])

    run(Clock(world, start_of(DAY)), models=None, max_board_age_s=None)

    records = open_ladder().read()
    assert_complete(world, records)  # every board is 30 s old: within any checkpoint's limit
    assert open_ladder().verify().sealed is True


# ---------------------------------------------------------------- the chain and the seal


def test_the_ladder_log_is_hash_chained_and_sealed_like_the_shadow_log():
    world = World([race_one(), race_two()])

    run(Clock(world, start_of(DAY)), models=models_of(nowcast_of(mu_a)))

    path = ladder_log_path(DAY)
    assert path == paths.data_dir() / "shadow" / "japan" / f"{DAY.isoformat()}.ladder.jsonl"
    assert path.parent == shadow_log_path(DAY).parent
    lines = raw_lines()
    prev = "0" * 64
    for n, line in enumerate(lines, 1):
        obj = json.loads(line)
        assert obj["seq"] == n and obj["prev_sha256"] == prev, n
        assert set(obj) in ({"seq", "prev_sha256", "record"}, {"seq", "prev_sha256", "seal"}), n
        assert ("seal" in obj) is (n == len(lines)), n
        prev = hashlib.sha256(line).hexdigest()
    seal = json.loads(lines[-1])["seal"]
    assert seal["n_records"] == len(lines) - 1 == len(open_ladder().read())
    assert instant(seal["sealed_utc"]) >= max(r.post for r in world.races)

    with pytest.raises(LogSealed):
        open_ladder().append([{k: v for k, v in raw_records()[0].items() if k != "day_status"}])
    assert raw_lines() == lines

    # an edited line or a dropped line breaks the chain
    edited = list(lines)
    edited[3] = edited[3].replace(b'"odds":', b'"odds":1', 1)
    for broken in (edited, lines[:2] + lines[3:]):
        path.write_bytes(b"".join(line + b"\n" for line in broken))
        with pytest.raises(ShadowLogTampered):
            open_ladder().verify()


@pytest.mark.parametrize("once", [False, True], ids=["loop", "once"])
def test_a_rerun_of_a_sealed_day_writes_nothing(once):
    world = World([race_one(), race_two()])
    models = models_of(nowcast_of(mu_a))
    run(Clock(world, start_of(DAY)), models=models)
    ladder, shadow = ladder_log_path(DAY).read_bytes(), shadow_log_path(DAY).read_bytes()

    assert world.pending == []  # every capture is in the logs: nothing new to read
    start = late_of(DAY) if once else start_of(DAY)
    run(Clock(world, start), models=models, once=once)

    assert ladder_log_path(DAY).read_bytes() == ladder
    assert shadow_log_path(DAY).read_bytes() == shadow


KILLS = {
    # after race one's T-10 is written, before its T-5 is due
    "mid race one": (timedelta(hours=1, minutes=-7), timedelta(minutes=4)),
    # after the shadow log is sealed (race two's T-1), before race two's final board
    "after the shadow log's seal": (timedelta(hours=1, minutes=41), timedelta(seconds=30)),
}


@pytest.mark.parametrize("case", list(KILLS.values()), ids=list(KILLS))
def test_a_restart_appends_only_the_points_still_missing(case):
    kill_after, pause = case
    one, two = race_one(), race_two()
    world = World([one, two])
    models = models_of(nowcast_of(mu_a))
    kill_at = datetime.combine(DAY, time(0), UTC) + kill_after

    with pytest.raises(Killed):
        run(Clock(world, start_of(DAY), kill_at=kill_at), models=models)

    first = ladder_log_path(DAY).read_bytes()
    before = points(open_ladder().read())
    for race in (one, two):
        for k in CHECKPOINTS:
            if race.due(k) + timedelta(seconds=POLL) <= kill_at:
                assert (race.race_id, k) in before, (race.race_id, k)
            elif race.due(k) > kill_at:
                assert (race.race_id, k) not in before, (race.race_id, k)
    if kill_at > one.post + timedelta(minutes=10):
        assert (one.race_id, FINAL) in before  # race one's final board was in long before
    assert (two.race_id, FINAL) not in before
    assert open_ladder().verify().sealed is False

    run(Clock(world, kill_at + pause), models=models)

    whole = ladder_log_path(DAY).read_bytes()
    assert whole.startswith(first)  # what was written before the kill is untouched
    records = open_ladder().read()
    assert_complete(world, records)
    assert open_ladder().verify().sealed is True
    assert open_shadow().verify().sealed is True
    for record in records:
        if record["checkpoint_min"] != FINAL and record["status"] == "ok":
            assert_checkpoint(world, record, mu_a, POOLS)


def test_a_day_whose_final_board_never_comes_still_ends():
    world = World([race_one(), race_two(final_s=None)])

    run(Clock(world, start_of(DAY)), models=None)

    records = open_ladder().read()
    got = points(records)
    assert (world.races[1].race_id, FINAL) not in got
    assert (world.races[0].race_id, FINAL) in got
    for race in world.races:
        for k in CHECKPOINTS:
            assert {(r["pool"], r["combination"]) for r in got[(race.race_id, k)]} == race.keys()
    assert open_shadow().verify().sealed is True


# ---------------------------------------------------------------- holdout days


def holdout_day() -> date:
    day = date(2026, 11, 28)
    while not is_holdout_day("japan", day):
        day += timedelta(days=1)
    return day


def test_a_holdout_day_still_writes_the_ladder_log_marked_holdout():
    day = holdout_day()
    assert not is_holdout_day("japan", DAY)
    world = World([race_one(day), race_two(day)])

    run(Clock(world, late_of(day)), models=models_of(nowcast_of(mu_a)), day=day, once=True)

    path = ladder_log_path(day)
    assert path.name == f"{day.isoformat()}.ladder.jsonl"
    records = raw_records(day)
    assert_complete(world, records)
    assert {r["day_status"] for r in records} == {"holdout"}
    summary = open_ladder(day).verify()
    assert summary.sealed is True and summary.day_status == "holdout"
    for record in records:
        if record["checkpoint_min"] != FINAL:
            assert_checkpoint(world, record, mu_a, POOLS)


# ---------------------------------------------------------------- the shadow log stays as it was


def test_the_shadow_log_is_the_same_with_or_without_the_ladder_log(tmp_path, monkeypatch):
    models = models_of(nowcast_of(mu_a))
    found = {}
    for name, ladder in (("without", False), ("with", True)):
        folder = (tmp_path / name).resolve()
        folder.mkdir()
        monkeypatch.setenv(paths.DATA_DIR_VAR, str(folder))
        world = World([race_one(), race_two()])
        run(Clock(world, start_of(DAY)), models=models, ladder=ladder)
        assert ladder_log_path(DAY).exists() is ladder
        assert open_shadow().verify().sealed is True
        found[name] = [
            {k: v for k, v in r.items() if k != "written_utc"} for r in open_shadow().read()
        ]
    assert found["with"] == found["without"]
    assert len(found["with"]) > 200  # 27 and 44 keys at the points with a board


# ---------------------------------------------------------------- the command


def write_pointer(folder: Path, nowcast: NowcastV0) -> None:
    win = WinCalibration(
        knots=(-4.0, -0.5),
        coef=(1.12,),
        fit_id="win-cli",
        train_start=date(2015, 1, 4),
        train_end=date(2024, 12, 28),
        n_races=321,
    )
    ordering = OrderingFit("discounted_harville", (0.85, 0.75), "ord-cli", 321)
    folder.mkdir(parents=True, exist_ok=True)
    names = {"win": "win.json", "ordering": "ordering.json", "nowcast": "nowcast.json"}
    for key, obj in (("win", win), ("ordering", ordering), ("nowcast", nowcast)):
        (folder / names[key]).write_text(obj.to_json(), encoding="utf-8")
    (folder / "current.json").write_text(json.dumps(names), encoding="utf-8")


@pytest.mark.parametrize("with_models", [False, True], ids=["no models", "current.json"])
def test_the_shadow_day_command_writes_the_ladder_log(monkeypatch, with_models):
    world = World([race_one(), race_two(captures=minute_captures())])
    world.write_until(late_of(DAY))
    nowcast = replace(nowcast_of(mu_a), fit_id="now-cli-7")
    if with_models:
        write_pointer(paths.data_dir() / "models" / "japan" / "v0", nowcast)
    monkeypatch.setattr(shadow_day, "utc_now", lambda: late_of(DAY))
    from horsetrack.cli import app

    result = CliRunner().invoke(app, ["japan", "shadow-day", "--date", DAY.isoformat(), "--once"])

    assert result.exit_code == 0, result.output
    assert ladder_log_path(DAY).is_file()
    records = open_ladder().read()
    assert_complete(world, records)
    assert open_ladder().verify().sealed is True
    assert open_shadow().verify().sealed is True
    for record in records:
        if record["checkpoint_min"] == FINAL:
            continue
        assert record["nowcast_id"] == ("now-cli-7" if with_models else None)
        assert_checkpoint(world, record, mu_a if with_models else None, POOLS)
