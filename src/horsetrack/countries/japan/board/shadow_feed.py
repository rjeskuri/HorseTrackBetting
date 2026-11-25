"""The shadow feed: each race's shadow values at its bucket and its avoid flags (ticket #121).

Rules (docs/specs/board-feed.md, "The shadow feed"):
- Reads `odds_snapshots/jra/<date>/<VENUE>.jsonl.gz` through `odds_log.LogTail` and the post times
  of `post_times_for`; never requests anything and never reads the shadow log.
- Per race: the board as of now() (`board_at`), minutes to post, the bucket on the decision grid
  (`bucket_for`: 0 at or after the post), `shadow_values` at that bucket and `avoid_flags`.
- Values are the gross return per 1 yen at the final price (1.0 is break-even). No key names a
  stake, a bet, Kelly, an amount or a wager.
- `win5` (ticket #135): on a WIN5 day the WIN5 pool total, carryover and pool EV as of now()
  (`research.win5.live_pool_ev` over the log's WIN5 `__pool__` rows); None on any other day.
- `cache`: a mapping the caller keeps between builds. A race's values are recomputed only when its
  board (capture time and figures) or its bucket changed; flags read the clock on every build. The
  log readers live in the same mapping, so a rebuild parses only the members appended since.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, MutableMapping
from datetime import date, datetime, timedelta, timezone
from typing import Any

from horsetrack.core import paths
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board import snapshot
from horsetrack.countries.japan.board.feed import _race_no, _venues_logged, utc_now
from horsetrack.countries.japan.board.flags import FlagRules, avoid_flags
from horsetrack.countries.japan.board.snapshot import (
    PLACE,
    WIDE,
    WIN,
    RaceBoard,
    board_at,
    bucket_for,
    load_decision_grid,
    post_times_for,
)
from horsetrack.countries.japan.models.v0.shadow_values import ShadowModels, shadow_values
from horsetrack.countries.japan.research.win5 import live_pool_ev
from horsetrack.countries.japan.shadow.day import point_seed

log = logging.getLogger(__name__)

WIN_FIELDS = ("value_mean", "value_q05", "value_q95")
RANGE_FIELDS = ("value_at_low", "value_at_high")

Entry = tuple[datetime, int, dict[str, Any]]  # (available, seq, row)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(timezone.utc).isoformat()


def _finite(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


# ---------------------------------------------------------------- the logs, read on


class _DayRows:
    """The rows a board as of now needs, per race: the latest row per board key available by the
    last now, plus every row available after it (a replay's later captures). A clock that goes
    back reads the logs again from the start."""

    def __init__(self, race_date: date) -> None:
        self.race_date = race_date
        self._reset()

    def _reset(self) -> None:
        self.tails: dict[str, odds_log.LogTail] = {}
        self.latest: dict[str, dict[tuple, Entry]] = {}
        self.later: dict[str, list[Entry]] = {}
        self.horizon: datetime | None = None
        self.seq = 0
        self.win5: list[dict[str, Any]] = []  # the WIN5 rows read so far, whatever their time

    def _offer(self, race_id: str, entry: Entry) -> None:
        slot = self.latest.setdefault(race_id, {})
        key = snapshot._key(entry[2])
        current = slot.get(key)
        if current is None or entry[:2] >= current[:2]:
            slot[key] = entry

    def update(self, venues: list[str], now_utc: datetime) -> None:
        if self.horizon is not None and now_utc < self.horizon:
            self._reset()
        for venue in venues:
            path = paths.odds_snapshot_path("jra", self.race_date.isoformat(), venue)
            tail = self.tails.setdefault(venue, odds_log.LogTail(path))
            for row in tail.poll():
                race_id, available = row.get("race_id"), snapshot._available(row)
                if not isinstance(race_id, str) or available is None:
                    continue
                if race_id.endswith(":WIN5"):
                    self.win5.append(row)
                    continue
                if snapshot._key(row) is None:
                    continue
                self.seq += 1
                self.later.setdefault(race_id, []).append((available, self.seq, row))
        for race_id, entries in self.later.items():
            due = [e for e in entries if e[0] <= now_utc]
            for entry in due:
                self._offer(race_id, entry)
            self.later[race_id] = [e for e in entries if e[0] > now_utc]
        self.horizon = now_utc

    def race_ids(self, venue: str) -> list[str]:
        prefix = f"jra:{self.race_date.isoformat()}:{venue}:"
        return sorted(r for r in self.latest.keys() | self.later.keys() if r.startswith(prefix))

    def rows(self, race_id: str) -> list[dict[str, Any]]:
        entries = list(self.latest.get(race_id, {}).values()) + self.later.get(race_id, [])
        return [row for _, _, row in sorted(entries, key=lambda e: e[1])]


# ---------------------------------------------------------------- values


def _figures(board: RaceBoard) -> tuple:
    return (
        tuple(sorted(board.win.items())),
        tuple(sorted(board.place.items())),
        tuple(sorted(board.wide.items())),
    )


def _values(
    board: RaceBoard,
    bucket: int,
    models: ShadowModels,
    n_draws: int,
    seed: int,
    cache: MutableMapping | None,
) -> dict[tuple[str, str], Any]:
    """(pool, combination) -> ShadowValue of the board at the bucket, from the cache when the board
    and the bucket are the ones last computed for the race."""
    key = (
        board.race_id,
        _iso(board.capture_utc),
        bucket,
        n_draws,
        seed,
        tuple(sorted(models.ids.items())),
        _figures(board),
    )
    slot = ("values", board.race_id)
    if cache is not None:
        hit = cache.get(slot)
        if hit is not None and hit[0] == key:
            return hit[1]
    try:
        found = shadow_values(
            board,
            tau_min=bucket,
            models=models,
            n_draws=n_draws,
            seed=point_seed(seed, board.race_id, bucket),
        )
    except Exception:  # the race shows its flags without values; tried again on the next build
        log.exception("%s T-%d: shadow values failed; shown without", board.race_id, bucket)
        return {}
    values = {(v.pool, v.combination): v for v in found}
    if cache is not None:
        cache[slot] = (key, values)
    return values


def _pools(board: RaceBoard, flags: dict, values: dict) -> dict[str, dict[str, dict]]:
    keys = (
        [(WIN, str(h)) for h in sorted(board.win)]
        + [(PLACE, str(h)) for h in sorted(board.place)]
        + [(WIDE, f"{a},{b}") for a, b in sorted(board.wide)]
    )
    out: dict[str, dict[str, dict]] = {WIN: {}, PLACE: {}, WIDE: {}}
    for pool, combination in keys:
        value = values.get((pool, combination))
        names = WIN_FIELDS if pool == WIN else RANGE_FIELDS
        entry = {n: None if value is None else _finite(getattr(value, n)) for n in names}
        entry["flags"] = list(flags.get((pool, combination), ()))
        out[pool][combination] = entry
    return out


# ---------------------------------------------------------------- the feed


def build_shadow(
    race_date: date,
    *,
    venues: list[str] | None = None,
    now: Callable[[], datetime] = utc_now,
    models: ShadowModels | None,
    flag_rules: FlagRules,
    n_draws: int = 400,
    seed: int = 0,
    cache: MutableMapping | None = None,
) -> dict[str, Any]:
    """The shadow values and avoid flags of every race of `race_date` with a board as of now()."""
    now_utc = now().astimezone(timezone.utc)
    grid = load_decision_grid()
    logged = _venues_logged(race_date)
    if venues is not None:
        wanted = set(venues)
        logged = [venue for venue in logged if venue in wanted]
    day_rows = cache.get(("logs", race_date)) if cache is not None else None
    if day_rows is None:
        day_rows = _DayRows(race_date)
        if cache is not None:
            cache[("logs", race_date)] = day_rows
    day_rows.update(logged, now_utc)

    races = []
    for venue in logged:
        posts = post_times_for(race_date, venue)
        for race_id in day_rows.race_ids(venue):
            race_no = _race_no(race_id)
            if race_no is None:
                continue
            post = posts.get(race_no)
            board = board_at(day_rows.rows(race_id), race_id, now_utc, post_utc=post)
            if board is None:
                continue
            minutes = bucket = None
            if post is not None:
                minutes = (post - now_utc) / timedelta(minutes=1)
                bucket = bucket_for(minutes, grid)
            values = {}
            if models is not None and bucket is not None:
                values = _values(board, bucket, models, n_draws, seed, cache)
            flags = avoid_flags(board, rules=flag_rules, now_utc=now_utc)
            races.append(
                {
                    "race_id": race_id,
                    "venue": venue,
                    "race_no": race_no,
                    "post_utc": _iso(post),
                    "minutes_to_post": None if minutes is None else round(minutes, 2),
                    "tau_bucket_min": bucket,
                    "board_capture_utc": _iso(board.capture_utc),
                    "pools": _pools(board, flags, values),
                }
            )
    return {
        "race_date": race_date.isoformat(),
        "generated_utc": _iso(now_utc),
        "models": None if models is None else dict(models.ids),
        "races": races,
        "win5": live_pool_ev(day_rows.win5, race_date, now=now_utc),
    }
