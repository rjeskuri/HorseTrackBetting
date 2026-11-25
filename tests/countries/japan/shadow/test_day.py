"""The shadow runner: one sealed record per runner and Wide pair at each grid point (ticket #120).

Contract pinned here (`horsetrack.countries.japan.shadow.day`):

`run_shadow_day(race_date, *, venues, now, sleep, poll_s, grace_s, max_board_age_s, max_lag_s, once,
models, log, flag_rules, n_draws, seed) -> ShadowSummary` (the ticket's signature):
- Venues are the day's log files in the folder of `odds_snapshot_path('jra', date, venue)`. Post
  times come from `post_times_for` (#113), the grid from `load_decision_grid()` (D2: 60, 30, 16, 5,
  2, 1). Every log is tailed with `odds_log.LogTail`; `odds_log.read_rows` is never used.
- A grid point (race, tau) is due when now() >= post - tau + grace_s. It is decided once: either one
  record per key on `board_at(rows, race_id, post - tau)` (WIN runners, PLACE runners, Wide pairs),
  or, with no board or a board older than max_board_age_s at the decision time (decision - its
  capture_utc), one record with status "no_board" and board_capture_utc null. A point whose key is
  in `log.logged_keys()` is not decided again, so a restart writes no duplicate.
- Record fields, beyond #119's required ones: decision_utc = post - tau; board_capture_utc = the
  board's capture_utc; written_utc = now() at the write; mode "live" when written_utc -
  decision_utc <= max_lag_s, else "replay"; board_odds / board_odds_hi = the board's figures (low
  and high for PLACE and Wide, high null for WIN); value_mean, value_q05, value_q95, value_at_low,
  value_at_high from `shadow_values(board, tau_min=tau, models, n_draws, seed=<derived>)`;
  `n_draws` and `seed` (the seed derived from (seed, race_id, tau), one per race and tau, the same
  for every pool) on each valued record; flags = the sorted list of `avoid_flags(board, rules,
  now_utc=decision)` for that key ([] when none); models = `models.ids`.
- Without models, records still carry status "ok", board odds and flags, with null values and
  null models.
- The day ends once the last race's T-1 point is due (or at 16:45 Asia/Tokyo when there are no
  post times); the runner then seals the log and returns. `once=True` decides every due point and
  returns, sealing only when every race's last point is due (the 16:45 rule without post times).
- ShadowSummary(records, no_board, races, sealed): records = records appended by this run, no_board
  ones included; no_board = the no_board records among them; races = the races with post times;
  sealed = the log is sealed when the run returns (also when it already was).
- No outcome is read: no store table, parquet or dataset is opened.

CLI: `horsetrack japan shadow-day [--date YYYY-MM-DD] [--venues TOK,KYO] [--once]` on the japan
group. Its clock is the module attribute `utc_now()`, looked up at call time (the default date is
today in Asia/Tokyo; it is also the run's `now`). It loads `load_shadow_models()`,
`load_flag_rules()` and `ShadowLog(shadow_log_path(date), day=date)`, prints a few lines and logs
details to `logs_dir('japan')/shadow-day-<date>.log`; exit 0 on a clean run.

Every row and fit is synthetic. The clock is fake: `sleep` moves it and appends the captures the
synthetic logger would have written by then; nothing really waits.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import dataclasses
import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import pytest
from typer.testing import CliRunner

from horsetrack.core import paths
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board.flags import FlagRules, avoid_flags
from horsetrack.countries.japan.board.snapshot import DecisionGrid, board_at
from horsetrack.countries.japan.models.v0 import calibration, store
from horsetrack.countries.japan.models.v0.calibration import WinCalibration
from horsetrack.countries.japan.models.v0.nowcast import NowcastV0
from horsetrack.countries.japan.models.v0.ordering import OrderingFit
from horsetrack.countries.japan.models.v0.shadow_values import ShadowModels, shadow_values
from horsetrack.countries.japan.shadow import day as shadow_day
from horsetrack.countries.japan.shadow import log as shadow_log
from horsetrack.countries.japan.shadow.day import ShadowSummary, run_shadow_day
from horsetrack.countries.japan.shadow.log import ShadowLog, shadow_log_path

UTC = timezone.utc
DAY = date(2026, 11, 22)  # a development day (before the holdout start)
GRID = (60, 30, 16, 5, 2, 1)  # D2, the shipped decision.yaml
WIN, PLACE, WIDE = "WIN", "PLACE", "QUINELLA_PLACE"
EARLY = datetime(2026, 11, 21, 23, 55, tzinfo=UTC)  # 08:55 in Tokyo, before every T-60
LATE = datetime(2026, 11, 22, 3, 0, tzinfo=UTC)  # 12:00 in Tokyo, after every race
LIMIT = datetime(2026, 11, 22, 8, 30, tzinfo=UTC)  # 17:30 in Tokyo: a runner still polling hangs
END_NO_POSTS = datetime(2026, 11, 22, 7, 45, tzinfo=UTC)  # 16:45 in Tokyo
POST1 = datetime(2026, 11, 22, 1, 0, tzinfo=UTC)
POST2 = datetime(2026, 11, 22, 1, 40, tzinfo=UTC)
GRACE = 15.0
N_DRAWS = 16
SEED = 11
RULES = FlagRules(floor_max_odds=1.3, tail_min_odds=50.0, tail_rank=6, stale_s=40.0)
VALUE_FIELDS = ("value_mean", "value_q05", "value_q95", "value_at_low", "value_at_high")
BASE = {
    1: (1.25, 3.4, 6.2, 12.5, 38.0, 85.0),
    2: (1.2, 2.9, 4.6, 7.3, 11.8, 26.0, 64.0, 152.0),
}


class Killed(BaseException):
    """The runner's process is killed (a BaseException, so no `except Exception` swallows it)."""


# ---------------------------------------------------------------- the synthetic day


@dataclass(frozen=True)
class Race:
    venue: str
    no: int
    post: datetime
    base: tuple[float, ...]
    first_min: int = 75  # first capture at post - first_min minutes - 30 s
    last_min: int = -1  # last capture at post - last_min minutes - 30 s (-1: 30 s after post)

    @property
    def race_id(self) -> str:
        return f"jra:{DAY.isoformat()}:{self.venue}:{self.no:02d}"

    def capture_times(self) -> list[datetime]:
        return [
            self.post - timedelta(minutes=m, seconds=30)
            for m in range(self.first_min, self.last_min - 1, -1)
        ]

    def decision(self, tau: int) -> datetime:
        return self.post - timedelta(minutes=tau)

    def keys(self) -> set[tuple[str, str]]:
        horses = range(1, len(self.base) + 1)
        return (
            {(WIN, str(h)) for h in horses}
            | {(PLACE, str(h)) for h in horses}
            | {(WIDE, f"{a},{b}") for a, b in combinations(horses, 2)}
        )


def race_one(**changes) -> Race:
    return Race(**{"venue": "TOK", "no": 1, "post": POST1, "base": BASE[1], **changes})


def race_two(**changes) -> Race:
    return Race(**{"venue": "TOK", "no": 2, "post": POST2, "base": BASE[2], **changes})


def iso(at: datetime) -> str:
    return at.astimezone(UTC).isoformat()


def instant(text: str) -> datetime:
    return datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)


def capture_rows(race: Race, at: datetime, *, trap: bool = False) -> list[dict]:
    """One capture of `race` available at `at`; prices move with the minute of the capture, and a
    trap capture is priced seven times higher so a board that used it is easy to see."""
    factor = 7.0 if trap else 1.0 + 0.004 * ((int(at.timestamp()) // 60) % 23)
    win = {h: round(o * factor, 3) for h, o in enumerate(race.base, 1)}

    def row(pool: str, combination: str, odds: float, hi: float | None = None) -> dict:
        out = {
            "snapshot_utc": iso(at - timedelta(seconds=2)),
            "fetched_utc": iso(at),
            "race_id": race.race_id,
            "pool": pool,
            "combination": combination,
            "odds": odds,
            "status_raw": "10時00分現在",
            "page_state": None,
            "source_update_utc": iso(at - timedelta(seconds=10)),
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


def log_path(venue: str) -> Path:
    return paths.odds_snapshot_path("jra", DAY.isoformat(), venue)


class World:
    """The day's captures, appended to the odds logs as the fake clock reaches them."""

    def __init__(self, races: list[Race], *, traps: bool = False, sidecars: bool = True):
        self.races = races
        members: dict[tuple[datetime, str], list[dict]] = defaultdict(list)
        for race in races:
            for at in race.capture_times():
                members[(at, race.venue)].extend(capture_rows(race, at))
            if traps:  # a capture 5 s after each decision time, before its grace ends
                for tau in GRID:
                    at = race.decision(tau) + timedelta(seconds=5)
                    members[(at, race.venue)].extend(capture_rows(race, at, trap=True))
        self.pending = sorted(members.items())
        self.rows = [row for _, rows in self.pending for row in rows]
        self.events: list[tuple[str, str, list[dict]]] = []
        if sidecars:
            for venue in sorted({r.venue for r in races}):
                posts = {r.no: r.post for r in races if r.venue == venue}
                sidecar = log_path(venue).with_name(f"{venue}.races.json")
                odds_log.write_post_times(sidecar, posts, EARLY - timedelta(hours=1))

    def write_until(self, t: datetime) -> None:
        while self.pending and self.pending[0][0][0] <= t:
            (_, venue), rows = self.pending.pop(0)
            path = log_path(venue)
            odds_log.append_capture(path, rows)
            self.events.append(("append", os.path.realpath(path), rows))

    def race(self, race_id: str) -> Race:
        return next(r for r in self.races if r.race_id == race_id)

    def board(self, race: Race, tau: int):
        """The board at the decision time, from every row of the day (later ones included)."""
        return board_at(self.rows, race.race_id, race.decision(tau), post_utc=race.post)


class Clock:
    """now() is the fake time; sleep(s) moves it and lets the synthetic logger catch up."""

    def __init__(self, world: World, start: datetime, *, kill_at: datetime | None = None):
        self.world = world
        self.t = start
        self.kill_at = kill_at
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
        if new > LIMIT:
            raise RuntimeError(f"the runner kept polling until {new.isoformat()}")
        self.t = new
        self.world.write_until(new)


# ---------------------------------------------------------------- fixtures and helpers


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    monkeypatch.setattr(shadow_log, "is_holdout_day", lambda country, day: False)
    return d


def tiny_fits(tag: str = "t1") -> tuple[WinCalibration, OrderingFit, NowcastV0]:
    win = WinCalibration(
        knots=(-4.0, -0.5),
        coef=(1.12,),
        fit_id=f"win-{tag}",
        train_start=date(2015, 1, 4),
        train_end=date(2024, 12, 28),
        n_races=321,
    )
    ordering = OrderingFit("discounted_harville", (0.85, 0.75), f"ord-{tag}", 321)
    nowcast = NowcastV0(
        bands=(2.0, 5.0, 10.0, 20.0, 50.0, 100.0),
        grid=DecisionGrid(GRID, 2),
        prior_n=50.0,
        days=("2026-11-21",),
        sigma0={(p, t): 0.04 + 0.002 * t for p in (WIN, PLACE, WIDE) for t in GRID},
        cells={},
        fit_id=f"now-{tag}",
    )
    return win, ordering, nowcast


@pytest.fixture
def models() -> ShadowModels:
    win, ordering, nowcast = tiny_fits()
    ids = {"win": win.fit_id, "ordering": ordering.fit_id, "nowcast": nowcast.fit_id}
    return ShadowModels(win=win, ordering=ordering, nowcast=nowcast, ids=ids)


def open_log(path: Path | None = None) -> ShadowLog:
    return ShadowLog(path or shadow_log_path(DAY), day=DAY)


def run(clock: Clock, *, models, log: ShadowLog | None = None, **options) -> ShadowSummary:
    settings = dict(
        venues=None,
        now=clock.now,
        sleep=clock.sleep,
        poll_s=10.0,
        grace_s=GRACE,
        max_board_age_s=120.0,
        max_lag_s=120.0,
        once=False,
        flag_rules=RULES,
        n_draws=N_DRAWS,
        seed=SEED,
    )
    settings.update(options)
    return run_shadow_day(DAY, models=models, log=log or open_log(), **settings)


def decided(records: list[dict]) -> dict[tuple[str, int], list[dict]]:
    """Records by (race_id, tau_min)."""
    out: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        out[(record["race_id"], record["tau_min"])].append(record)
    return out


def assert_point_decided_once(world: World, points: dict, race: Race, tau: int) -> None:
    """One no_board record, or exactly one ok record per board key, never both."""
    got = points[(race.race_id, tau)]
    statuses = Counter(r["status"] for r in got)
    if statuses.get("no_board"):
        assert len(got) == 1, f"{race.race_id} T-{tau}: a no_board point has {len(got)} records"
        return
    keys = Counter((r["pool"], r["combination"]) for r in got)
    assert set(keys) == race.keys(), f"{race.race_id} T-{tau}"
    assert set(keys.values()) == {1}, f"{race.race_id} T-{tau}: a key was written twice"


def expected_flags(board, record: dict) -> list[str]:
    found = avoid_flags(board, rules=RULES, now_utc=instant(record["decision_utc"]))
    return list(found.get((record["pool"], record["combination"]), ()))


def figures(board, pool: str, combination: str) -> tuple[float, float | None]:
    if pool == WIN:
        return board.win[int(combination)], None
    if pool == PLACE:
        return board.place[int(combination)]
    a, b = (int(x) for x in combination.split(","))
    return board.wide[(a, b)]


def close(got, want) -> bool:
    if want is None or got is None:
        return got is None and want is None
    return math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12)


def total_keys(races: list[Race]) -> int:
    return sum(len(r.keys()) for r in races) * len(GRID)


# ---------------------------------------------------------------- the grid


def test_each_grid_point_is_logged_once_per_race(models):
    races = [race_one(), race_two()]
    world = World(races)
    clock = Clock(world, EARLY)

    summary = run(clock, models=models)

    expected = total_keys(races)  # 6 runners: 27 keys; 8 runners: 44 keys; six tau each
    assert expected == (27 + 44) * 6
    assert (summary.records, summary.no_board, summary.races, summary.sealed) == (
        expected,
        0,
        2,
        True,
    )
    records = open_log().read()
    assert len(records) == expected
    by_key = defaultdict(list)
    for record in records:
        by_key[(record["race_id"], record["pool"], record["combination"])].append(record["tau_min"])
    for race in races:
        for pool, combination in race.keys():
            taus = by_key[(race.race_id, pool, combination)]
            assert sorted(taus) == sorted(GRID), (race.race_id, pool, combination, taus)
    assert {k[0] for k in by_key} == {r.race_id for r in races}
    for record in records:
        race = world.race(record["race_id"])
        decision = instant(record["decision_utc"])
        assert record["status"] == "ok"
        assert record["mode"] == "live"
        assert decision == race.decision(record["tau_min"])
        assert instant(record["written_utc"]) >= decision + timedelta(seconds=GRACE)
        assert instant(record["board_capture_utc"]) <= decision
        assert record["models"] == models.ids
    assert open_log().verify().by_tau == {tau: expected // len(GRID) for tau in GRID}


GRACE_CASES = [
    # (grace_s, seconds after race 1's T-5 decision time)
    (15.0, 14),
    (15.0, 15),
    (40.0, 39),
    (40.0, 40),
    (90.0, 60),
    (5.0, 185),
]


@pytest.mark.parametrize("grace, after", GRACE_CASES)
def test_a_grid_point_waits_for_its_grace_period(grace, after):
    races = [race_one(), race_two()]
    world = World(races)
    now = races[0].decision(5) + timedelta(seconds=after)
    clock = Clock(world, now)

    summary = run(clock, models=None, once=True, grace_s=grace)

    due = {
        (race.race_id, tau)
        for race in races
        for tau in GRID
        if now >= race.decision(tau) + timedelta(seconds=grace)
    }
    records = open_log().read() if summary.records else []
    assert set(decided(records)) == due
    assert summary.records == sum(len(world.race(r).keys()) for r, _ in due)
    assert summary.sealed is False  # race 2's T-1 is not due yet
    for record in records:
        assert instant(record["written_utc"]) == now
        assert instant(record["written_utc"]) >= instant(record["decision_utc"]) + timedelta(
            seconds=grace
        )


# ---------------------------------------------------------------- the board


@pytest.mark.parametrize("start", [EARLY, LATE], ids=["live", "replay"])
def test_the_board_used_is_never_after_the_decision_time(tmp_path, models, start):
    races = [race_one(), race_two()]
    world = World(races, traps=True)
    clock = Clock(world, start)

    summary = run(clock, models=models)

    assert summary.records == total_keys(races) and summary.sealed
    records = open_log().read()
    seeds: dict[tuple[str, int], set] = defaultdict(set)
    for (race_id, tau), group in decided(records).items():
        race = world.race(race_id)
        board = world.board(race, tau)
        values = None
        for record in group:
            assert record["status"] == "ok"
            assert record["mode"] == ("live" if start == EARLY else "replay")
            decision = instant(record["decision_utc"])
            assert decision == race.decision(tau)
            captured = instant(record["board_capture_utc"])
            assert captured == board.capture_utc
            assert captured <= decision
            assert (record["board_odds"], record["board_odds_hi"]) == figures(
                board, record["pool"], record["combination"]
            ), "the board is not the one at the decision time"
            assert record["flags"] == expected_flags(board, record)
            assert record["n_draws"] == N_DRAWS
            assert type(record["seed"]) is int
            seeds[(race_id, tau)].add(record["seed"])
            if values is None:
                values = {
                    (v.pool, v.combination): v
                    for v in shadow_values(
                        board,
                        tau_min=tau,
                        models=models,
                        n_draws=record["n_draws"],
                        seed=record["seed"],
                    )
                }
            want = values[(record["pool"], record["combination"])]
            for field in VALUE_FIELDS:
                assert close(record[field], getattr(want, field)), (race_id, tau, field)
    # one seed per race and tau, the same in every pool, and a different one at every point
    assert all(len(s) == 1 for s in seeds.values())
    derived = {point: next(iter(s)) for point, s in seeds.items()}
    assert len(set(derived.values())) == len(derived) == len(races) * len(GRID)

    # the seed is derived from (seed, race_id, tau): the same again, and another seed moves it
    again, other = (open_log(tmp_path / name / f"{DAY.isoformat()}.jsonl") for name in "ab")
    run(Clock(world, LATE), models=models, log=again, once=True)
    run(Clock(world, LATE), models=models, log=other, once=True, seed=SEED + 1)
    seeds_again = {(r["race_id"], r["tau_min"]): r["seed"] for r in again.read()}
    seeds_other = {(r["race_id"], r["tau_min"]): r["seed"] for r in other.read()}
    assert seeds_again == derived
    assert set(seeds_other) == set(derived)
    assert all(seeds_other[p] != derived[p] for p in derived)
    first_values = {
        (r["race_id"], r["pool"], r["combination"], r["tau_min"]): r["value_mean"] for r in records
    }
    for r in again.read():
        assert (
            r["value_mean"]
            == first_values[(r["race_id"], r["pool"], r["combination"], r["tau_min"])]
        )


NO_BOARD_CASES = {
    # max_board_age_s -> race 2's grid points without a usable board. Race 2's captures run from
    # T-45:30 to T-10:30: nothing at T-60, and at T-5, T-2 and T-1 the board is 330, 510 and 570 s
    # old.
    120.0: {60, 5, 2, 1},
    400.0: {60, 2, 1},
}


@pytest.mark.parametrize("max_age", list(NO_BOARD_CASES))
def test_a_missing_or_stale_board_writes_a_no_board_record(models, max_age):
    one, two = race_one(), race_two(first_min=45, last_min=10)
    world = World([one, two])
    clock = Clock(world, LATE)

    summary = run(clock, models=models, once=True, max_board_age_s=max_age)

    missing = NO_BOARD_CASES[max_age]
    records = open_log().read()
    points = decided(records)
    assert set(points) == {(r.race_id, tau) for r in (one, two) for tau in GRID}
    for tau in GRID:
        assert_point_decided_once(world, points, one, tau)
        assert all(r["status"] == "ok" for r in points[(one.race_id, tau)])
        assert_point_decided_once(world, points, two, tau)
        got = points[(two.race_id, tau)]
        if tau in missing:
            (record,) = got
            assert record["status"] == "no_board"
            assert record["board_capture_utc"] is None
            assert instant(record["decision_utc"]) == two.decision(tau)
            assert record["mode"] == "replay"
        else:
            assert all(r["status"] == "ok" for r in got)
            board = world.board(two, tau)
            age = (two.decision(tau) - board.capture_utc).total_seconds()
            assert age <= max_age
            for record in got:  # flags read the clock at the decision time, not the run's now
                assert record["flags"] == expected_flags(board, record)
                assert ("stale" in record["flags"]) is (age > RULES.stale_s)
    ok = sum(r["status"] == "ok" for r in records)
    assert ok == len(one.keys()) * len(GRID) + len(two.keys()) * (len(GRID) - len(missing))
    assert summary.no_board == len(missing)
    assert summary.records == ok + len(missing)
    assert open_log().verify().no_board == len(missing)


# ---------------------------------------------------------------- restarts and the seal


def test_a_restart_writes_no_duplicates(models):
    one, two = race_one(), race_two(first_min=45)
    world = World([one, two])
    kill_at = datetime(2026, 11, 22, 0, 50, tzinfo=UTC)  # after race 1's T-16, before its T-5

    with pytest.raises(Killed):
        run(Clock(world, EARLY, kill_at=kill_at), models=models)

    first = open_log().read()
    before_kill = {
        (r.race_id, tau)
        for r in (one, two)
        for tau in GRID
        if r.decision(tau) + timedelta(seconds=GRACE) <= kill_at
    }
    assert before_kill == {
        (one.race_id, 60),
        (one.race_id, 30),
        (one.race_id, 16),
        (two.race_id, 60),
    }
    assert set(decided(first)) == before_kill
    assert open_log().verify().sealed is False

    restart = datetime(2026, 11, 22, 1, 0, 30, tzinfo=UTC)  # race 1's T-5, T-2 and T-1 are late
    second = run(Clock(world, restart), models=models)

    records = open_log().read()
    assert records[: len(first)] == first  # what was written before the kill is untouched
    points = decided(records)
    assert set(points) == {(r.race_id, tau) for r in (one, two) for tau in GRID}
    for race in (one, two):
        for tau in GRID:
            assert_point_decided_once(world, points, race, tau)
    assert [r["status"] for r in points[(two.race_id, 60)]] == ["no_board"]
    keys = Counter((r["race_id"], r["pool"], r["combination"], r["tau_min"]) for r in records)
    assert set(keys.values()) == {1}
    expected = len(one.keys()) * len(GRID) + len(two.keys()) * (len(GRID) - 1) + 1
    assert len(records) == expected
    assert second.records == expected - len(first)
    assert second.no_board == 0  # the no_board record was the first run's
    assert second.sealed is True

    # a run after the day is sealed writes nothing and does not fail
    raw = shadow_log_path(DAY).read_bytes()
    third = run(Clock(world, LATE), models=models, once=True)
    assert (third.records, third.no_board, third.sealed) == (0, 0, True)
    assert shadow_log_path(DAY).read_bytes() == raw


LAST_DUE = POST2 - timedelta(minutes=1) + timedelta(seconds=GRACE)
SEAL_CASES = {
    "the loop ends after the last race": ("loop", EARLY, True, True),
    "once, before the last point is due": ("once", LAST_DUE - timedelta(seconds=1), True, False),
    "once, when the last point is due": ("once", LAST_DUE, True, True),
    "no post times, before 16:45 local": (
        "once",
        END_NO_POSTS - timedelta(seconds=1),
        False,
        False,
    ),
    "no post times, at 16:45 local": ("once", END_NO_POSTS, False, True),
}


@pytest.mark.parametrize("case", list(SEAL_CASES.values()), ids=list(SEAL_CASES))
def test_the_day_log_is_sealed_after_the_last_race(case):
    mode, start, sidecars, sealed = case
    races = [race_one(), race_two()]
    world = World(races, sidecars=sidecars)
    clock = Clock(world, start)

    summary = run(clock, models=None, once=mode == "once")

    assert summary.sealed is sealed
    assert open_log().verify().sealed is sealed
    if not sidecars:
        assert (summary.records, summary.races) == (0, 0)
    if not sealed:
        return
    last = json.loads(shadow_log_path(DAY).read_bytes().splitlines()[-1])
    sealed_utc = instant(last["seal"]["sealed_utc"])
    if mode == "loop":
        assert summary.records == total_keys(races)
        assert LAST_DUE <= sealed_utc <= clock.now()
        assert clock.now() <= POST2 + timedelta(minutes=5), "the runner waited past the day"
    else:
        assert sealed_utc == start


# ---------------------------------------------------------------- a pool without a fit (#233)


def gapped(models: ShadowModels, missing: set) -> ShadowModels:
    """The same fits, with the real nowcast lacking (pool, tau) cells (None: every tau): its
    `cell` then raises KeyError("no training data for <pool> at tau <tau>")."""
    sigma0 = {
        (p, t): s
        for (p, t), s in models.nowcast.sigma0.items()
        if (p, None) not in missing and (p, t) not in missing
    }
    return dataclasses.replace(models, nowcast=dataclasses.replace(models.nowcast, sigma0=sigma0))


RECORD_KEYS = {  # the shadow log's record format as it is today: no field is added by #233
    "race_id",
    "tau_min",
    "decision_utc",
    "written_utc",
    "mode",
    "git_sha",
    "pool",
    "combination",
    "board_capture_utc",
    "status",
    "models",
    "board_odds",
    "board_odds_hi",
    *VALUE_FIELDS,
    "flags",
    "n_draws",
    "seed",
    "day_status",
}
POOL_PATTERNS = {  # the pool as the config and the log name it, not as a part of another name
    WIN: r"(?<![A-Z_])WIN(?![A-Z_])",
    PLACE: r"(?<![A-Z_])PLACE(?![A-Z_])",
    WIDE: r"(?<![A-Z_])QUINELLA_PLACE(?![A-Z_])",
}
GAP_DAY_CASES = {
    "no Wide fit": {(WIDE, None)},
    "no Place fit": {(PLACE, None)},
    "neither Place nor Wide": {(PLACE, None), (WIDE, None)},
    "Wide lacks two taus, Place one": {(WIDE, 60), (WIDE, 2), (PLACE, 16)},
    "no fit for any pool": {(WIN, None), (PLACE, None), (WIDE, None)},
}


def gap_warnings(caplog, race_id: str, tau: int, pool: str) -> list[logging.LogRecord]:
    """The WARNING records of the package that name this race, tau and pool."""
    found = []
    for rec in caplog.records:
        if rec.levelno != logging.WARNING or not rec.name.startswith("horsetrack"):
            continue
        text = rec.getMessage()
        if race_id not in text or not re.search(POOL_PATTERNS[pool], text):
            continue
        if re.search(rf"(?<!\d){tau}(?!\d)", text.replace(race_id, "")):
            found.append(rec)
    return found


@pytest.mark.parametrize("missing", list(GAP_DAY_CASES.values()), ids=list(GAP_DAY_CASES))
def test_a_pool_without_a_fit_keeps_the_other_pools_values_and_the_log_seals(
    models, caplog, missing
):
    races = [race_one(), race_two()]
    world = World(races)
    clock = Clock(world, EARLY)
    caplog.set_level(logging.INFO, logger="horsetrack")

    summary = run(clock, models=gapped(models, missing))

    assert summary.records == total_keys(races) and summary.no_board == 0 and summary.sealed
    verdict = open_log().verify()
    assert verdict.sealed
    records = open_log().read()
    assert len(records) == total_keys(races)

    lost = {
        (race.race_id, tau, pool)
        for race in races
        for tau in GRID
        for pool in (WIN, PLACE, WIDE)
        if (pool, None) in missing or (pool, tau) in missing
    }
    assert lost
    for (race_id, tau), group in decided(records).items():
        race = world.race(race_id)
        board = world.board(race, tau)
        assert_point_decided_once(world, decided(records), race, tau)
        healthy = None  # the point's values from a run with every fit, once its seed is known
        for record in group:
            assert record["status"] == "ok"
            assert set(record) == RECORD_KEYS  # no new field, a reason included
            assert (record["board_odds"], record["board_odds_hi"]) == figures(
                board, record["pool"], record["combination"]
            )
            assert record["flags"] == expected_flags(board, record)
            if (race_id, tau, record["pool"]) in lost:
                assert all(record[f] is None for f in VALUE_FIELDS), (race_id, tau, record["pool"])
                continue
            assert record["n_draws"] == N_DRAWS and type(record["seed"]) is int
            assert record["models"] == models.ids
            if healthy is None:
                healthy = {
                    (v.pool, v.combination): v
                    for v in shadow_values(
                        board, tau_min=tau, models=models, n_draws=N_DRAWS, seed=record["seed"]
                    )
                }
            wanted = healthy[(record["pool"], record["combination"])]
            assert (
                record["value_mean"] if record["pool"] == WIN else record["value_at_low"]
            ) is not None
            for field in VALUE_FIELDS:
                assert close(record[field], getattr(wanted, field)), (race_id, tau, field)

    # health reads: one WARNING per (race, tau, pool) without values, no ERROR for the point
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    for race in races:
        for tau in GRID:
            for pool in (WIN, PLACE, WIDE):
                found = gap_warnings(caplog, race.race_id, tau, pool)
                want_n = 1 if (race.race_id, tau, pool) in lost else 0
                assert len(found) == want_n, (race.race_id, tau, pool, len(found))
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name.startswith("horsetrack")
    ]
    assert len(warnings) == len(lost)


def test_a_nowcast_without_wide_still_logs_win_and_place_values_and_seals(models):
    """The smoke test of 2026-11-25 in miniature: Wide has no fit, WIN and PLACE keep values."""
    races = [race_one(), race_two()]
    world = World(races)
    clock = Clock(world, EARLY)

    summary = run(clock, models=gapped(models, {(WIDE, None)}))

    assert summary.sealed and open_log().verify().sealed
    by_pool = defaultdict(list)
    for record in open_log().read():
        by_pool[record["pool"]].append(record)
    assert len(by_pool[WIN]) == sum(len(r.base) for r in races) * len(GRID)
    assert all(r["value_mean"] is not None and r["value_q05"] is not None for r in by_pool[WIN])
    assert all(r["value_at_low"] is not None for r in by_pool[PLACE])
    assert len(by_pool[WIDE]) > 0
    assert all(r["value_at_low"] is None and r["value_at_high"] is None for r in by_pool[WIDE])


# ---------------------------------------------------------------- records without models


def test_without_models_records_carry_flags_and_null_values():
    races = [race_one(), race_two()]
    world = World(races)
    clock = Clock(world, LATE)

    summary = run(clock, models=None, once=True)

    records = open_log().read()
    assert summary.records == len(records) == total_keys(races)
    seen_flags = set()
    for (race_id, tau), group in decided(records).items():
        race = world.race(race_id)
        board = world.board(race, tau)
        assert_point_decided_once(world, decided(records), race, tau)
        for record in group:
            assert record["status"] == "ok"
            assert record["models"] is None
            assert all(record[field] is None for field in VALUE_FIELDS)
            assert instant(record["board_capture_utc"]) == board.capture_utc
            assert (record["board_odds"], record["board_odds_hi"]) == figures(
                board, record["pool"], record["combination"]
            )
            assert record["flags"] == expected_flags(board, record)
            seen_flags.update(record["flags"])
    assert {"floor", "tail", "tail_member"} <= seen_flags  # the world does raise flags


# ---------------------------------------------------------------- live and replay


REPLAY_CASES = {
    # max_lag_s -> the points written later than that after their decision time, for a runner
    # started at 00:50 UTC: race 1's T-60, T-30 and T-16 (3000, 1200 and 360 s late) and race 2's
    # T-60 (600 s late); every later point is written within a poll and the grace of its time.
    120.0: {(1, 60), (1, 30), (1, 16), (2, 60)},
    1500.0: {(1, 60)},
}


@pytest.mark.parametrize("max_lag", list(REPLAY_CASES))
def test_late_records_are_marked_replay(max_lag):
    races = [race_one(), race_two()]
    world = World(races)
    start = datetime(2026, 11, 22, 0, 50, tzinfo=UTC)
    clock = Clock(world, start)

    summary = run(clock, models=None, max_lag_s=max_lag)

    assert summary.records == total_keys(races)
    records = open_log().read()
    replay = set()
    for record in records:
        decision, written = instant(record["decision_utc"]), instant(record["written_utc"])
        assert start <= written <= clock.now()
        lag = (written - decision).total_seconds()
        assert record["mode"] == ("live" if lag <= max_lag else "replay"), record
        if record["mode"] == "replay":
            replay.add((int(record["race_id"][-2:]), record["tau_min"]))
    assert replay == REPLAY_CASES[max_lag]


# ---------------------------------------------------------------- reading the logs


def test_only_new_log_members_are_parsed_on_each_poll(monkeypatch):
    races = [race_one(), race_two()]
    world = World(races)
    tail_paths: dict[int, str] = {}
    real_init, real_poll = odds_log.LogTail.__init__, odds_log.LogTail.poll

    def spy_init(self, path, *args, **kwargs):
        tail_paths[id(self)] = os.path.realpath(path)
        real_init(self, path, *args, **kwargs)

    def spy_poll(self, *args, **kwargs):
        rows = real_poll(self, *args, **kwargs)
        world.events.append(("poll", tail_paths[id(self)], list(rows)))
        return rows

    def no_whole_read(*args, **kwargs):
        raise AssertionError("the runner re-read a whole log with read_rows")

    reads: list[tuple[str, int]] = []
    for name in ("iter_members", "iter_members_with_offsets"):
        real = getattr(odds_log, name)

        def spy(path, *args, _real=real, **kwargs):
            reads.append((os.path.realpath(path), int(kwargs.get("start", args[0] if args else 0))))
            return _real(path, *args, **kwargs)

        monkeypatch.setattr(odds_log, name, spy)
    monkeypatch.setattr(odds_log.LogTail, "__init__", spy_init)
    monkeypatch.setattr(odds_log.LogTail, "poll", spy_poll)
    monkeypatch.setattr(odds_log, "read_rows", no_whole_read)

    clock = Clock(world, EARLY)  # the log already holds the morning's first captures
    summary = run(clock, models=None)
    assert summary.records == total_keys(races) and summary.sealed

    path = os.path.realpath(log_path("TOK"))
    polls = [e for e in world.events if e[0] == "poll" and e[1] == path]
    assert len(polls) > 100, "the runner should poll every poll_s through the day"
    pending: list[dict] = []
    single = 0
    previous_nonempty = False
    for kind, where, rows in world.events:
        if where != path:
            continue
        if kind == "append":
            pending.append(rows)
            continue
        flat = [row for member in pending for row in member]
        assert rows == flat, "a poll did not return exactly the members appended since the last"
        if len(pending) == 1 and previous_nonempty:
            single += 1
        previous_nonempty = previous_nonempty or bool(pending)
        pending = []
    assert single > 10, "a poll after one new capture should return that capture alone"
    appended = sum(len(rows) for kind, where, rows in world.events if kind == "append")
    returned = sum(len(rows) for kind, where, rows in polls)
    assert returned <= appended
    # after its first read, a log is never read again from its first byte
    first_read = next((i for i, (p, _) in enumerate(reads) if p == path), None)
    if first_read is not None:
        assert all(start > 0 for p, start in reads[first_read + 1 :] if p == path)


def test_no_outcome_table_is_opened(monkeypatch, models):
    opened: list[str] = []

    def refuse(what: str):
        def reader(*args, **kwargs):
            opened.append(what)
            raise AssertionError(f"the shadow runner opened {what}: {args!r}")

        return reader

    import pandas as pd
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    monkeypatch.setattr(store, "read_store_table", refuse("read_store_table"))
    monkeypatch.setattr(calibration, "read_store_table", refuse("read_store_table"))
    monkeypatch.setattr(pq, "read_table", refuse("pyarrow.parquet.read_table"))
    monkeypatch.setattr(pq, "ParquetFile", refuse("pyarrow.parquet.ParquetFile"))
    monkeypatch.setattr(ds, "dataset", refuse("pyarrow.dataset.dataset"))
    monkeypatch.setattr(pd, "read_parquet", refuse("pandas.read_parquet"))
    # bait: an outcome table where the store keeps it
    results = paths.processed_dir("jra")
    results.mkdir(parents=True)
    (results / "results.parquet").write_bytes(b"not read")
    (results / "dividends.parquet").write_bytes(b"not read")

    races = [race_one(), race_two()]
    world = World(races)
    summary = run(Clock(world, LATE), models=models)

    assert opened == []
    assert summary.records == total_keys(races) and summary.sealed
    assert all(r["status"] == "ok" for r in open_log().read())


# ---------------------------------------------------------------- the command


ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def write_pointer(folder: Path) -> dict[str, str]:
    win, ordering, nowcast = tiny_fits("cli")
    folder.mkdir(parents=True, exist_ok=True)
    names = {"win": "win.json", "ordering": "ordering.json", "nowcast": "nowcast.json"}
    for key, obj in (("win", win), ("ordering", ordering), ("nowcast", nowcast)):
        (folder / names[key]).write_text(obj.to_json(), encoding="utf-8")
    (folder / "current.json").write_text(json.dumps(names), encoding="utf-8")
    return {"win": win.fit_id, "ordering": ordering.fit_id, "nowcast": nowcast.fit_id}


CLI_CASES = {
    "every venue, no models": ([], False, {"TOK", "KYO"}),
    "one venue, with models": (["--venues", "TOK"], True, {"TOK"}),
}


@pytest.mark.parametrize("case", list(CLI_CASES.values()), ids=list(CLI_CASES))
def test_shadow_day_once_via_the_cli(monkeypatch, case):
    extra, with_models, venues = case
    races = [race_one(), race_two(), Race("KYO", 1, POST1 + timedelta(minutes=20), BASE[1])]
    world = World(races)
    world.write_until(LATE)
    ids = write_pointer(paths.data_dir() / "models" / "japan" / "v0") if with_models else None
    monkeypatch.setattr(shadow_day, "utc_now", lambda: LATE)
    from horsetrack.cli import app

    result = CliRunner().invoke(
        app, ["japan", "shadow-day", "--date", DAY.isoformat(), "--once", *extra]
    )

    out = ANSI.sub("", result.output)
    assert result.exit_code == 0, out
    assert len([line for line in out.splitlines() if line.strip()]) <= 12, out
    records = open_log().read()
    assert {r["race_id"].split(":")[2] for r in records} == venues
    wanted = [r for r in races if r.venue in venues]
    assert len(records) == total_keys(wanted)
    assert all(r["models"] == ids for r in records)
    assert all(r["mode"] == "replay" for r in records)
    assert open_log().verify().sealed is True
    detail = paths.logs_dir("japan") / f"shadow-day-{DAY.isoformat()}.log"
    assert detail.is_file() and detail.stat().st_size > 0


DEFAULT_DATES = [
    (datetime(2026, 11, 27, 16, 0, tzinfo=UTC), date(2026, 11, 28)),
    (datetime(2026, 11, 27, 14, 59, 59, tzinfo=UTC), date(2026, 11, 27)),
    (datetime(2026, 11, 27, 15, 0, tzinfo=UTC), date(2026, 11, 28)),
]


@pytest.mark.parametrize("now, expected", DEFAULT_DATES, ids=lambda v: v.isoformat())
def test_the_default_date_is_today_in_tokyo(monkeypatch, now, expected):
    monkeypatch.setattr(shadow_day, "utc_now", lambda: now)
    from horsetrack.cli import app

    result = CliRunner().invoke(app, ["japan", "shadow-day", "--once"])

    assert result.exit_code == 0, ANSI.sub("", result.output)
    logs = sorted(p.name for p in paths.logs_dir("japan").glob("shadow-day-*.log"))
    assert logs == [f"shadow-day-{expected.isoformat()}.log"]
