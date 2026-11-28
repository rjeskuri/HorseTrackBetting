"""`horsetrack japan shadow-day` (ticket #120): one sealed shadow record per runner and Wide pair at
each grid point of each race of a day, from the odds logs, in a process separate from the logger.

Each poll tails every venue's log (`odds_log.LogTail`: only members appended since the last poll
are parsed) and re-reads the post times (#113). A grid point (race, tau) is due once
now >= post - tau + grace_s and is decided once, on `board_at(rows, race_id, post - tau)`: one
record per WIN runner, PLACE runner and Wide pair, or one `no_board` record when there is no board
or it is more than max_board_age_s older than the decision time. A point with a key already in the
log is not decided again, so a restart writes no duplicate. The day ends once the last race's last
point is due (16:45 Asia/Tokyo without post times), and the log is then sealed.

Model probabilities (#564): every record is `record_version` 2 and carries `p_model`, the shadow
feed's p_model of the decision board (`shadow_feed._p_model`, starters by `_starters`), null
without models or without one for its pool. The pools with no price forecast (EXOTIC_POOLS) get
records of their own from each pool's latest capture available by the decision time (no
max_board_age_s: tier-2 pools are captured once per grid point; `board_age_s` says how old):
every QUINELLA and BRACKET_QUINELLA combination with a p_model and the EXOTIC_TOP_N most likely
of FORECAST, TRIO and TIERCE, each with `board_odds` and `value_model` = p_model x board_odds.

Memory: per race, each undecided grid point keeps the latest row per board key and per exotic
combination available by its decision time, plus the latest row per key overall (to re-time a
moved post), and the race's earliest WIN capture's field (for the starters); a decided point
drops its rows. No outcome is read: no results, dividends or store table.

Seam for the command, looked up at call time: `utc_now` (the only clock).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from datetime import time as clock_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import typer

from horsetrack.core import paths
from horsetrack.core.config import load_source
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board import snapshot
from horsetrack.countries.japan.board.flags import FlagRules, avoid_flags, load_flag_rules
from horsetrack.countries.japan.board.shadow_feed import _p_model, _probe_until, _starters
from horsetrack.countries.japan.board.snapshot import (
    PLACE,
    WIDE,
    WIN,
    RaceBoard,
    board_at,
    jra_day_dir,
    load_decision_grid,
    post_times_for,
)
from horsetrack.countries.japan.logger.pull_vm import vm_day_dir
from horsetrack.countries.japan.models.v0.loader import load_shadow_models
from horsetrack.countries.japan.models.v0.ordering import _git_sha
from horsetrack.countries.japan.models.v0.shadow_values import (
    BRACKET_QUINELLA,
    FORECAST,
    QUINELLA,
    TIERCE,
    TRIO,
    ShadowModels,
    point_seed,
    shadow_values,
)
from horsetrack.countries.japan.shadow.log import (
    NO_BOARD,
    RECORD_VERSION,
    LogSealed,
    ShadowLog,
    shadow_log_path,
)

logger = logging.getLogger(__name__)

SOURCE = "jra"
TIMEZONE = "Asia/Tokyo"
JST = ZoneInfo(TIMEZONE)
END_WITHOUT_POSTS = clock_time(16, 45)  # local: the day's end when no post time is known
LOG_SUFFIX = ".jsonl.gz"
ANY = "*"  # pool and combination of a no_board record: it stands for the whole point
VALUE_FIELDS = ("value_mean", "value_q05", "value_q95", "value_at_low", "value_at_high")
EXIT_ERROR = 1
# The pools with no price forecast (#564): logged as p_model x the decision-time odds
EXOTIC_POOLS = frozenset({QUINELLA, FORECAST, TRIO, TIERCE, BRACKET_QUINELLA})
TOPPED = frozenset({FORECAST, TRIO, TIERCE})  # logged as their EXOTIC_TOP_N most likely
EXOTIC_TOP_N = 15
POOL_TOTAL = odds_log.POOL_TOTAL


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ShadowSummary:
    records: int
    no_board: int
    races: int
    sealed: bool


# ---------------------------------------------------------------- per race, what its points need


Slot = dict[tuple, tuple[datetime, int, dict[str, Any]]]  # board key -> (available, seq, row)


def _offer(slot: Slot, key: tuple, entry: tuple[datetime, int, dict[str, Any]]) -> None:
    """Keep the latest row per key; at the same instant the later row in the log wins."""
    current = slot.get(key)
    if current is None or entry[:2] >= current[:2]:
        slot[key] = entry


def _row_key(row: dict[str, Any]) -> tuple | None:
    """The row's board key (`snapshot._key`), else (pool, combination text) of an exotic pool's
    price row, else None."""
    key = snapshot._key(row)
    if key is not None:
        return key
    pool, combination = row.get("pool"), row.get("combination")
    if pool in EXOTIC_POOLS and isinstance(combination, str) and combination != POOL_TOTAL:
        return pool, combination
    return None


def _is_exotic(key: tuple) -> bool:
    return key[0] in EXOTIC_POOLS and key[1] != POOL_TOTAL


class _Race:
    """The rows one race's undecided grid points need: per tau, the latest row per board key and
    per exotic combination available by its decision time; `newest`, the latest row per key
    overall; `opening`, its earliest WIN capture as `shadow_feed._starters` reads it."""

    def __init__(self, race_id: str) -> None:
        self.race_id = race_id
        self.post: datetime | None = None
        self.slots: dict[int, tuple[datetime, Slot]] = {}  # tau -> (decision, slot)
        self.newest: Slot = {}
        self.done: set[int] = set()
        # (its earliest WIN capture, the runners not SCR on it, every runner on it)
        self.opening: tuple[datetime, set[int], set[int]] | None = None

    def _open(self, key: tuple, row: dict[str, Any]) -> None:
        """The field at the opening, as shadow_feed's `_DayRows._open` keeps it."""
        if key[0] != WIN or key[1] == POOL_TOTAL:
            return
        snap = odds_log._instant(row.get("snapshot_utc"))
        if snap is None:
            return
        if self.opening is None or snap < self.opening[0]:
            self.opening = (snap, set(), set())
        elif snap > self.opening[0]:
            return
        self.opening[2].add(key[1])
        if row.get("status_raw") != snapshot.SCRATCHED:
            self.opening[1].add(key[1])

    def schedule(self, post: datetime, grid: tuple[int, ...]) -> None:
        """Set or move the post; each undecided point's rows are re-chosen for its decision time."""
        if post == self.post:
            return
        known: dict[int, tuple[tuple, tuple[datetime, int, dict]]] = {}
        for slot in (self.newest, *(s for _, s in self.slots.values())):
            for key, entry in slot.items():
                known[entry[1]] = (key, entry)
        ordered = [known[seq] for seq in sorted(known)]
        self.post = post
        self.slots = {}
        for tau in grid:
            if tau in self.done:
                continue
            decision = post - timedelta(minutes=tau)
            slot: Slot = {}
            for key, entry in ordered:
                if entry[0] <= decision:
                    _offer(slot, key, entry)
            self.slots[tau] = (decision, slot)

    def add(self, row: dict[str, Any], seq: int) -> None:
        key = _row_key(row)
        available = snapshot._available(row)
        if key is None or available is None:
            return
        self._open(key, row)
        entry = (available, seq, row)
        _offer(self.newest, key, entry)
        for decision, slot in self.slots.values():
            if available <= decision:
                _offer(slot, key, entry)

    def board(self, tau: int) -> RaceBoard | None:
        decision, slot = self.slots[tau]
        entries = sorted((e for k, e in slot.items() if not _is_exotic(k)), key=lambda e: e[1])
        rows = [row for _, _, row in entries]
        return board_at(rows, self.race_id, decision, post_utc=self.post)

    def exotics(self, tau: int) -> dict[str, tuple[datetime, dict[str, float]]]:
        """Per exotic pool, its latest capture available by the point's decision time (the rows
        sharing the latest snapshot_utc): (the earliest availability of its rows, {combination:
        odds} of its priced rows)."""
        latest: dict[str, tuple[datetime, list[tuple[datetime, str, dict]]]] = {}
        for key, (available, _, row) in self.slots[tau][1].items():
            if not _is_exotic(key):
                continue
            snap = odds_log._instant(row.get("snapshot_utc"))
            if snap is None:
                continue
            current = latest.get(key[0])
            if current is None or snap > current[0]:
                current = latest[key[0]] = (snap, [])
            elif snap < current[0]:
                continue
            current[1].append((available, key[1], row))
        out = {}
        for pool, (_, entries) in latest.items():
            prices = {}
            for _, combination, row in entries:
                odds = float(row["odds"]) if snapshot._priced(row) else math.nan
                if math.isfinite(odds) and odds > 0:
                    prices[combination] = odds
            if prices:
                out[pool] = (min(available for available, _, _ in entries), prices)
        return out

    def decided(self, tau: int) -> None:
        self.done.add(tau)
        self.slots.pop(tau, None)
        if not self.slots and self.post is not None:
            self.newest = {}


# ---------------------------------------------------------------- records


def _iso(at: datetime) -> str:
    return at.astimezone(timezone.utc).isoformat()


def _finite(value: float | None) -> float | None:
    """A value the log can hold: NaN and infinity become null."""
    return None if value is None or not math.isfinite(value) else float(value)


def _probability(value: float | None) -> float | None:
    """A probability the log can hold: NaN and infinity become null, rounding past [0, 1] is
    clipped."""
    return None if value is None or not math.isfinite(value) else min(1.0, max(0.0, float(value)))


def _board_keys(board: RaceBoard) -> list[tuple[str, str, float, float | None]]:
    """(pool, combination, low, high) of every key on the board."""
    out = [(WIN, str(h), odds, None) for h, odds in sorted(board.win.items())]
    out += [(PLACE, str(h), lo, hi) for h, (lo, hi) in sorted(board.place.items())]
    out += [(WIDE, f"{a},{b}", lo, hi) for (a, b), (lo, hi) in sorted(board.wide.items())]
    return out


@dataclass(frozen=True)
class _Settings:
    grace_s: float
    max_board_age_s: float
    max_lag_s: float
    models: ShadowModels | None
    flag_rules: FlagRules
    n_draws: int
    seed: int
    git_sha: str
    probe_until: datetime | None  # shadow_feed._probe_until of the day: the opening's coverage


def _common(race_id: str, tau: int, decision: datetime, written: datetime, s: _Settings) -> dict:
    lag = (written - decision).total_seconds()
    return {
        "race_id": race_id,
        "tau_min": tau,
        "decision_utc": _iso(decision),
        "written_utc": _iso(written),
        "mode": "live" if lag <= s.max_lag_s else "replay",
        "git_sha": s.git_sha,
        "record_version": RECORD_VERSION,
    }


def _race_p_model(race: _Race, board: RaceBoard, decision: datetime, s: _Settings) -> dict | None:
    """The decision board's p_model as the shadow feed forms it; None without models or when it
    cannot be formed (`_p_model` logs why)."""
    if s.models is None:
        return None
    opening = race.opening if race.opening is not None and race.opening[0] <= decision else None
    return _p_model(board, s.models, None, _starters(board, opening, s.probe_until))


def _exotic_records(
    race: _Race, tau: int, decision: datetime, p_model: dict, common: dict, ids: dict
) -> list[dict[str, Any]]:
    """The exotic pools' records of one point: each pool's capture by the decision time, the
    combinations with a p_model, every one of QUINELLA and BRACKET_QUINELLA, the EXOTIC_TOP_N most
    likely of the others."""
    records = []
    for pool, (captured, prices) in sorted(race.exotics(tau).items()):
        probs = p_model.get(pool) or {}
        chosen = [(c, _probability(probs.get(c))) for c in prices]
        chosen = [(c, p) for c, p in chosen if p is not None]
        if pool in TOPPED:
            chosen = sorted(chosen, key=lambda e: (-e[1], e[0]))[:EXOTIC_TOP_N]
        for combination, p in chosen:
            odds = prices[combination]
            records.append(
                {
                    **common,
                    "pool": pool,
                    "combination": combination,
                    "board_capture_utc": _iso(captured),
                    "board_age_s": (decision - captured).total_seconds(),
                    "status": "ok",
                    "models": dict(ids),
                    "p_model": p,
                    "board_odds": odds,
                    "value_model": p * odds,
                }
            )
    return records


def _point_records(
    race: _Race, tau: int, written: datetime, s: _Settings
) -> tuple[list[dict[str, Any]], bool]:
    """The records of one grid point, and whether it is a no_board point."""
    decision = race.slots[tau][0]
    board = race.board(tau)
    common = _common(race.race_id, tau, decision, written, s)
    age = None if board is None else (decision - board.capture_utc).total_seconds()
    if board is None or age > s.max_board_age_s:
        record = {
            **common,
            "pool": ANY,
            "combination": ANY,
            "board_capture_utc": None,
            "status": NO_BOARD,
            "models": None,
            "board_age_s": age,
            "p_model": None,
        }
        return [record], True

    flags = avoid_flags(board, rules=s.flag_rules, now_utc=decision)
    values: dict[tuple[str, str], Any] = {}
    models = s.models
    seed = point_seed(s.seed, race.race_id, tau)
    if models is not None:
        try:
            found = shadow_values(board, tau_min=tau, models=models, n_draws=s.n_draws, seed=seed)
            values = {(v.pool, v.combination): v for v in found}
            lacking = {v.pool: v.reason for v in found if v.reason is not None}
            for pool, reason in lacking.items():
                logger.warning(
                    "%s T-%d %s: logged without shadow values (%s)", race.race_id, tau, pool, reason
                )
            if len(lacking) == len({v.pool for v in found}):
                models = None  # no pool has values: the point is logged as without models
        except Exception:
            logger.exception("%s T-%d: shadow values failed; logged without", race.race_id, tau)
            models = None
    p_model = _race_p_model(race, board, decision, s) or {}
    records = []
    for pool, combination, low, high in _board_keys(board):
        value = values.get((pool, combination))
        # a p_model sits beside the fit ids it came from: none on a record logged without models
        prob = None if models is None else _probability(p_model.get(pool, {}).get(combination))
        record = {
            **common,
            "pool": pool,
            "combination": combination,
            "board_capture_utc": _iso(board.capture_utc),
            "status": "ok",
            "models": None if models is None else dict(models.ids),
            "board_odds": low,
            "board_odds_hi": high,
            **{f: None if value is None else _finite(getattr(value, f)) for f in VALUE_FIELDS},
            "flags": list(flags.get((pool, combination), ())),
            "n_draws": None if models is None else s.n_draws,
            "seed": None if models is None else seed,
            "p_model": prob,
        }
        records.append(record)
    if p_model and s.models is not None:  # p_model is only ever formed with models
        try:
            records += _exotic_records(race, tau, decision, p_model, common, s.models.ids)
        except Exception:
            logger.exception("%s T-%d: exotic records failed; logged without", race.race_id, tau)
    return records, False


# ---------------------------------------------------------------- the day


def _log_folder(race_date: date, day_dir: Path | None = None) -> Path:
    return jra_day_dir(race_date, day_dir)


def _discover(race_date: date, day_dir: Path | None = None) -> list[str]:
    """The venues with a log file for the day, re-read on every poll."""
    folder = _log_folder(race_date, day_dir)
    try:
        names = [p.name for p in folder.iterdir()]
    except FileNotFoundError:
        return []
    return sorted(
        n[: -len(LOG_SUFFIX)] for n in names if n.endswith(LOG_SUFFIX) and len(n) > len(LOG_SUFFIX)
    )


def _day_end(race_date: date, races: dict[str, _Race], grid: tuple[int, ...], grace_s: float):
    """When the last race's last point is due; 16:45 local when no post time is known."""
    posts = [r.post for r in races.values() if r.post is not None]
    if not posts:
        return datetime.combine(race_date, END_WITHOUT_POSTS, JST).astimezone(timezone.utc)
    return max(posts) - timedelta(minutes=min(grid)) + timedelta(seconds=grace_s)


def run_shadow_day(
    race_date: date,
    *,
    venues: list[str] | None = None,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    poll_s: float = 10.0,
    grace_s: float = 15.0,
    max_board_age_s: float = 120.0,
    max_lag_s: float = 120.0,
    once: bool = False,
    models: ShadowModels | None,
    log: ShadowLog,
    flag_rules: FlagRules,
    n_draws: int = 400,
    seed: int = 0,
    day_dir: Path | None = None,
) -> ShadowSummary:
    """Log every grid point of the day as it falls due, then seal the log. `once` decides every
    point due now and returns, sealing only when the day's last point is due. The odds logs are read
    from `day_dir` when given (a pulled VM folder), else the Mac's own day folder."""
    clock = now if now is not None else utc_now
    grid = tuple(sorted(load_decision_grid().minutes, reverse=True))
    settings = _Settings(
        grace_s,
        max_board_age_s,
        max_lag_s,
        models,
        flag_rules,
        n_draws,
        seed,
        _git_sha() or "unknown",
        _probe_until(race_date),
    )
    shadow = log
    tails: dict[str, odds_log.LogTail] = {}
    races: dict[str, _Race] = {}
    seq = 0
    records = no_board = 0
    done = {(key[0], key[3]) for key in shadow.logged_keys()}

    def refresh() -> None:
        """New venues' logs, the post times, then the rows appended since the last poll."""
        nonlocal seq
        for venue in venues if venues is not None else _discover(race_date, day_dir):
            tails.setdefault(venue, odds_log.LogTail(odds_log_path(race_date, venue, day_dir)))
        for venue in tails:
            for no, post in post_times_for(race_date, venue, day_dir=day_dir).items():
                race_id = f"{SOURCE}:{race_date.isoformat()}:{venue}:{int(no):02d}"
                race = races.setdefault(race_id, _Race(race_id))
                race.done |= {tau for tau in grid if (race_id, tau) in done}
                race.schedule(post, grid)
        for tail in tails.values():
            for row in tail.poll():
                race_id = row.get("race_id")
                if not isinstance(race_id, str):
                    continue
                race = races.get(race_id)
                if race is None:
                    race = races[race_id] = _Race(race_id)
                if race.post is not None and not race.slots:
                    continue  # every point of the race is decided
                seq += 1
                race.add(row, seq)

    def scheduled() -> int:
        return sum(r.post is not None for r in races.values())

    def sealed_summary() -> ShadowSummary:
        return ShadowSummary(records, no_board, scheduled(), True)

    if shadow.verify().sealed:
        refresh()
        logger.info("shadow-day %s: the log is already sealed; nothing written", race_date)
        return sealed_summary()

    while True:
        t = clock()
        refresh()
        due = [
            (race, tau)
            for race in races.values()
            if race.post is not None
            for tau in list(race.slots)
            if t >= race.slots[tau][0] + timedelta(seconds=grace_s)
        ]
        if due:
            written = clock()
            batch: list[dict[str, Any]] = []
            empty = 0
            for race, tau in due:
                point, missing = _point_records(race, tau, written, settings)
                batch.extend(point)
                empty += missing
            try:
                shadow.append(batch)
            except LogSealed:
                logger.warning("shadow-day %s: the log was sealed elsewhere; stopping", race_date)
                return sealed_summary()
            for race, tau in due:
                race.decided(tau)
                done.add((race.race_id, tau))
            records += len(batch)
            no_board += empty
            logger.info(
                "shadow-day %s: %d points, %d records (%d no_board) at %s",
                race_date,
                len(due),
                len(batch),
                empty,
                _iso(written),
            )
        if t >= _day_end(race_date, races, grid, grace_s):
            sealed_at = clock()
            try:
                shadow.seal(sealed_utc=sealed_at)
            except LogSealed:
                logger.warning("shadow-day %s: the log was sealed elsewhere", race_date)
                return sealed_summary()
            logger.info(
                "shadow-day %s: sealed at %s, %d records", race_date, _iso(sealed_at), records
            )
            return ShadowSummary(records, no_board, scheduled(), True)
        if once:
            return ShadowSummary(records, no_board, scheduled(), False)
        sleep(poll_s)


def odds_log_path(race_date: date, venue: str, day_dir: Path | None = None) -> Path:
    return _log_folder(race_date, day_dir) / f"{venue}{LOG_SUFFIX}"


# ---------------------------------------------------------------- the command


class _Usage(ValueError):
    """A bad option: one ERROR line, exit 1."""


def _day(value: str | None) -> date:
    if value is None:
        return utc_now().astimezone(JST).date()
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise _Usage(f"--date {value!r} is not a YYYY-MM-DD date") from None


def _venues(value: str | None) -> list[str] | None:
    if value is None:
        return None
    codes = list(dict.fromkeys(p.strip().upper() for p in value.split(",") if p.strip()))
    known = load_source("japan", SOURCE).venues
    unknown = [c for c in codes if c not in known]
    if unknown or not codes:
        raise _Usage(f"--venues {value!r}: unknown venue(s) {unknown}; known: {','.join(known)}")
    return codes


def _file_log(path: Path) -> logging.Handler:
    """Send the package's log records to `path` (UTC timestamps); the terminal stays quiet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(name)s %(message)s")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    package = logging.getLogger("horsetrack")
    package.addHandler(handler)
    package.setLevel(logging.INFO)
    return handler


def shadow_day_command(
    race_date: str | None = typer.Option(
        None, "--date", help="Race date (YYYY-MM-DD); default: today in Asia/Tokyo."
    ),
    venues: str | None = typer.Option(
        None, "--venues", help="Venue codes, e.g. TOK,KYO; default: every venue logged that day."
    ),
    once: bool = typer.Option(False, "--once", help="Decide every point due now, then exit."),
    from_vm: str | None = typer.Option(
        None,
        "--from-vm",
        help="Read the day's odds logs pulled from this VM (ssh alias; `pull-vm --every`) from "
        "$HORSETRACK_DATA_DIR/vm/<alias>/ instead of the Mac's own odds_snapshots/.",
    ),
) -> None:
    """Log shadow values at each grid point of a day's races, then seal the day's shadow log."""
    try:
        day = _day(race_date)
        codes = _venues(venues)
        day_dir = None if from_vm is None else vm_day_dir(from_vm, day)
    except ValueError as exc:  # _Usage, or a --from-vm that is not an ssh alias
        typer.echo(f"ERROR {exc}", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    detail = paths.logs_dir("japan") / f"shadow-day-{day.isoformat()}.log"
    handler = _file_log(detail)
    try:
        path = shadow_log_path(day)
        given = ",".join(codes) if codes else "discover"
        logger.info("shadow-day %s venues=%s once=%s log=%s", day, given, once, path)
        models = load_shadow_models()
        fits = ",".join(f"{k}={v}" for k, v in models.ids.items()) if models else "none"
        typer.echo(f"shadow-day {day.isoformat()} venues={given} models={fits}")
        typer.echo(f"log: {detail}")
        summary = run_shadow_day(
            day,
            venues=codes,
            now=utc_now,
            once=once,
            models=models,
            log=ShadowLog(path, day=day),
            flag_rules=load_flag_rules(),
            day_dir=day_dir,
        )
        typer.echo(
            f"done: records={summary.records} no_board={summary.no_board}"
            f" races={summary.races} sealed={summary.sealed} shadow log: {path}"
        )
    except Exception as exc:
        logger.exception("shadow-day %s failed", day)
        typer.echo(f"ERROR {' '.join(str(exc).split())[:300] or type(exc).__name__}", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    finally:
        logging.getLogger("horsetrack").removeHandler(handler)
        handler.close()
