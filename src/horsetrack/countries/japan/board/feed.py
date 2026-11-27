"""The live-board feed: a race day's odds, morning and T-20 moves, from the snapshot logs.

Rules (docs/specs/board-feed.md):
- Reads `odds_snapshots/jra/<date>/<VENUE>.jsonl.gz` and the post times of `post_times_for`; never
  requests anything. A row's capture time is its `snapshot_utc`; rows captured after `now()` are
  ignored, so the board is the one that could be shown at `now()`.
- Current: the latest row per (race, pool, combination). Morning reference: the earliest priced
  row captured on race day at or after `morning_from` Japan time. T-20 reference: the latest row
  at or before post - 20 min, null before then or without a post time; the T-2 reference is
  picked the same way at post - 2 min (#498). Moves are on `odds`; the T-20 -> T-2 move is
  t2 / t20 - 1.
- A runner whose current WIN or PLACE row is `SCR` is scratched on every combination of its race
  that contains its number, listed or not since.
- The feed is JSON-ready and carries prices only: no probability, value or bet field.
- Tier-1 pools only (BOARD_POOLS): the log also holds tier-2 and WIN5 rows (ticket #130), dropped
  row by row as each member is read, so a day's trifecta boards are never held in memory.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board.snapshot import SCRATCHED, jra_day_dir, post_times_for
from horsetrack.countries.japan.odds_log import POOL_TOTAL

_JST = timezone(timedelta(hours=9))  # Japan keeps no daylight saving time
_LOG_SUFFIX = ".jsonl.gz"
_RUNNER_POOLS = ("WIN", "PLACE")  # one runner per combination: where a scratch shows
# The board's pools: tier 1 (logger.yaml `pools`). Tier-2 and WIN5 rows never reach the feed.
BOARD_POOLS = frozenset({"WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE"})
T20 = timedelta(minutes=20)
T2 = timedelta(minutes=2)

_instant = odds_log._instant


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(timezone.utc).isoformat()


def _race_no(race_id: Any) -> int | None:
    """NN of `jra:<date>:<VENUE>:<NN>`; None for anything else."""
    parts = str(race_id).split(":")
    if len(parts) != 4:
        return None
    try:
        return int(parts[3])
    except ValueError:
        return None


def _priced(row: dict[str, Any]) -> bool:
    return row.get("odds") is not None and row.get("status_raw") != SCRATCHED


def _move(current: Any, reference: Any) -> float | None:
    """current / reference - 1, in percent with one decimal; None without both figures."""
    if current is None or reference is None or float(reference) == 0:
        return None
    return round((float(current) / float(reference) - 1) * 100, 1) + 0.0  # no -0.0


def _combination_order(combination: Any) -> tuple:
    try:
        return (0, tuple(int(part) for part in str(combination).split(",")), "")
    except ValueError:
        return (1, (), str(combination))


def _runners(combination: Any) -> set[str]:
    """The runner numbers of a combination (`'1'`, `'1,10'`), as whole numbers."""
    parts = (part.strip() for part in str(combination).split(","))
    return {str(int(part)) if part.isdigit() else part for part in parts}


def _venues_logged(race_date: date, day_dir: Path | None = None) -> list[str]:
    folder = jra_day_dir(race_date, day_dir)
    if not folder.is_dir():
        return []
    return sorted(p.name[: -len(_LOG_SUFFIX)] for p in folder.glob(f"*{_LOG_SUFFIX}"))


def _reference(
    history: list[tuple[datetime, dict[str, Any]]], at: datetime | None
) -> dict[str, Any] | None:
    """The latest row captured at or before `at` when it is priced; None without `at`."""
    if at is None:
        return None
    before = [row for t, row in history if t <= at]
    return before[-1] if before and _priced(before[-1]) else None


def _row_feed(
    history: list[tuple[datetime, dict[str, Any]]],
    morning_start: datetime,
    morning_end: datetime,
    t20_at: datetime | None,
    t2_at: datetime | None = None,
    scratched: bool = False,
) -> dict[str, Any]:
    """One combination's line from its rows in capture order (at the same instant, log order);
    `scratched` when one of its runners is out of the race."""
    _, current = history[-1]
    morning = next(
        (row for at, row in history if morning_start <= at < morning_end and _priced(row)), None
    )
    t20, t2 = _reference(history, t20_at), _reference(history, t2_at)
    morning_odds = None if morning is None else morning.get("odds")
    t20_odds = None if t20 is None else t20.get("odds")
    t2_odds = None if t2 is None else t2.get("odds")
    priced = _priced(current) and not scratched
    return {
        "combination": current.get("combination"),
        "odds": current.get("odds") if priced else None,
        "odds_hi": current.get("odds_hi") if priced else None,
        "status": SCRATCHED if scratched else current.get("status_raw"),
        "morning_odds": morning_odds,
        "move_morning_pct": _move(current.get("odds"), morning_odds) if priced else None,
        "t20_odds": t20_odds,
        "move_t20_pct": _move(current.get("odds"), t20_odds) if priced else None,
        "t2_odds": t2_odds,
        "move_t20_t2_pct": _move(t2_odds, t20_odds) if priced else None,
    }


def _race_feed(
    race_id: str,
    race_no: int,
    rows: list[tuple[datetime, dict[str, Any]]],
    post_utc: datetime | None,
    now_utc: datetime,
    morning_start: datetime,
    morning_end: datetime,
) -> dict[str, Any]:
    t20_at = None if post_utc is None or now_utc < post_utc - T20 else post_utc - T20
    t2_at = None if post_utc is None or now_utc < post_utc - T2 else post_utc - T2
    ordered = sorted(rows, key=lambda pair: pair[0])  # stable: log order within an instant
    histories: dict[str, dict[Any, list[tuple[datetime, dict[str, Any]]]]] = {}
    for at, row in ordered:
        pool, combination = row.get("pool"), row.get("combination")
        lines = histories.setdefault(pool, {})
        if combination != POOL_TOTAL:
            lines.setdefault(combination, []).append((at, row))
    totals = odds_log.latest_totals(row for _, row in ordered)
    out_runners = {
        runner
        for pool in _RUNNER_POOLS
        for combination, history in histories.get(pool, {}).items()
        if history[-1][1].get("status_raw") == SCRATCHED
        for runner in _runners(combination)
    }
    pools = {}
    for pool in sorted(histories, key=str):
        total = totals.get((race_id, pool))
        pools[pool] = {
            "investment": None if total is None else total.get("investment"),
            "rows": [
                _row_feed(
                    histories[pool][c],
                    morning_start,
                    morning_end,
                    t20_at,
                    t2_at,
                    scratched=bool(_runners(c) & out_runners),
                )
                for c in sorted(histories[pool], key=_combination_order)
            ],
        }
    return {
        "race_no": race_no,
        "race_id": race_id,
        "post_utc": _iso(post_utc),
        "last_capture_utc": _iso(ordered[-1][0]),
        "pools": pools,
    }


def _venue_races(
    rows: Iterable[dict[str, Any]],
    post_times: dict[int, datetime],
    now_utc: datetime,
    morning_start: datetime,
    morning_end: datetime,
) -> list[dict[str, Any]]:
    by_race: dict[tuple[int, str], list[tuple[datetime, dict[str, Any]]]] = {}
    for row in rows:
        at, no = _instant(row.get("snapshot_utc")), _race_no(row.get("race_id"))
        if at is None or no is None or at > now_utc:
            continue
        by_race.setdefault((no, str(row["race_id"])), []).append((at, row))
    return [
        _race_feed(
            race_id,
            no,
            by_race[(no, race_id)],
            post_times.get(no),
            now_utc,
            morning_start,
            morning_end,
        )
        for no, race_id in sorted(by_race)
    ]


def build_board(
    race_date: date,
    *,
    venues: Iterable[str] | None = None,
    now: Callable[[], datetime] = utc_now,
    morning_from: str = "07:00",
    day_dir: Path | None = None,
) -> dict[str, Any]:
    """The board of `race_date` as of `now()`: every venue logged that day (or only `venues`),
    races in number order, each pool's lines with their morning, T-20 and T-20 -> T-2 moves. The
    logs are read from `day_dir` when given (a pulled VM folder), else the Mac's own day folder."""
    now_utc = now().astimezone(timezone.utc)
    morning_start = datetime.combine(race_date, time.fromisoformat(morning_from), _JST)
    morning_end = datetime.combine(race_date + timedelta(days=1), time(0, 0), _JST)
    logged = _venues_logged(race_date, day_dir)
    if venues is not None:
        wanted = set(venues)
        logged = [venue for venue in logged if venue in wanted]
    out = []
    for venue in logged:
        log_path = jra_day_dir(race_date, day_dir) / f"{venue}{_LOG_SUFFIX}"
        rows = [
            row
            for member in odds_log.iter_members(log_path)
            for row in member
            if row.get("pool") in BOARD_POOLS
        ]
        races = _venue_races(
            rows,
            post_times_for(race_date, venue, day_dir=day_dir),
            now_utc,
            morning_start,
            morning_end,
        )
        out.append({"venue": venue, "races": races})
    return {"race_date": race_date.isoformat(), "generated_utc": _iso(now_utc), "venues": out}
