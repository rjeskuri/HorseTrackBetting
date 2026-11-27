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
- `ladder` (ticket #499), per entry: the odds at T-20, T-10, T-5, T-2 and the final, keyed 20, 10,
  5, 2, 0; a checkpoint passed is observed (the latest row captured at or before it, the plain
  feed's `t20_odds` rule; the final from `final_board`), one ahead is projected from the current
  figure with the drift cells of `models.nowcast` (docs/specs/board-feed.md, "The odds ladder").
  Display only: values, flags and the shadow log do not read it.
- `p_model` (ticket #506), per race: {pool: {combination: probability}} of the board as of now.
  WIN is `models.win.predict(q, board.context())` with q the board's 1/O_WIN normalised; every
  other pool is `model_probabilities(p, models.ordering, brackets=..., starters=...)`, brackets by
  JRA's draw of the largest horse number on the board (scratched included). Scratched runners are
  off the board, so they have no key and the rest renormalise; the places Place and Wide pay are
  those of the start of sales, `starters` any runner priced on the board plus, from the race's
  earliest WIN capture in the logs read: when it is before logger.yaml's `day_before.probe_until`
  (Tokyo) on the eve, the log covers the opening (#358) and the runners not SCR there count (one
  priced there and scratched later counts, one already SCR there does not); at or after it (a
  logger first run in the race date's 07:00 window), or with no `probe_until`, the log missed the
  opening and every runner it shows counts, SCR there or on the board included (the scratch
  taken as after the opening). Without an ordering fit only WIN; without models None. Display
  only: it never reaches a value, a flag or the shadow log.
- `cache`: a mapping the caller keeps between builds. A race's values are recomputed only when its
  board (capture time and figures) or its bucket changed, its p_model only when its board did;
  flags read the clock on every build. The log readers live in the same mapping, so a rebuild
  parses only the members appended since.
"""

from __future__ import annotations

import logging
import math
from bisect import bisect_left
from collections.abc import Callable, MutableMapping
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from horsetrack.core.config import load_country_yaml
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board import snapshot
from horsetrack.countries.japan.board.feed import (
    _JST,
    _LOG_SUFFIX,
    _race_no,
    _venues_logged,
    utc_now,
)
from horsetrack.countries.japan.board.flags import FlagRules, avoid_flags
from horsetrack.countries.japan.board.snapshot import (
    PLACE,
    WIDE,
    WIN,
    RaceBoard,
    board_at,
    bucket_for,
    final_board,
    jra_day_dir,
    load_decision_grid,
    post_times_for,
)
from horsetrack.countries.japan.models.v0.nowcast import FLOOR, band_of
from horsetrack.countries.japan.models.v0.ordering import OrderingFit
from horsetrack.countries.japan.models.v0.shadow_values import (
    ShadowModels,
    model_probabilities,
    shadow_values,
)
from horsetrack.countries.japan.research.win5 import live_pool_ev
from horsetrack.countries.japan.shadow.day import point_seed

log = logging.getLogger(__name__)

WIN_FIELDS = ("value_mean", "value_q05", "value_q95")
RANGE_FIELDS = ("value_at_low", "value_at_high")

Entry = tuple[datetime, int, dict[str, Any]]  # (available, seq, row)
LADDER_MINUTES = (20, 10, 5, 2)  # the ladder's checkpoints before the post; 0 is the final


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(timezone.utc).isoformat()


def _finite(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


# ---------------------------------------------------------------- the logs, read on


class _DayRows:
    """The rows a board as of now needs, per race: the latest row per board key available by the
    last now, plus every row available after it (a replay's later captures). For the odds ladder
    (#499) it also keeps, of the rows available by now, the latest captured at or before each
    checkpoint post - k per board key and the rows of the captures showing a final price. For the
    places paid (#506) it keeps, per race, its earliest WIN capture available by now, the runners
    not SCR on it and every runner on it (`_starters` reads them). A clock that goes back, or a
    change of post times, reads the logs again from the start."""

    def __init__(self, race_date: date, day_dir: Path | None = None) -> None:
        self.race_date, self.day_dir = race_date, day_dir
        self.posts: dict[str, dict[int, datetime]] = {}
        self._reset()

    def _reset(self) -> None:
        self.tails: dict[str, odds_log.LogTail] = {}
        self.latest: dict[str, dict[tuple, Entry]] = {}
        self.later: dict[str, list[Entry]] = {}
        self.horizon: datetime | None = None
        self.seq = 0
        self.win5: list[dict[str, Any]] = []  # the WIN5 rows read so far, whatever their time
        # race_id -> checkpoint minutes -> board key -> (capture, seq, row)
        self.refs: dict[str, dict[int, dict[tuple, tuple[datetime, int, dict]]]] = {}
        self.final_snaps: dict[str, set[datetime]] = {}  # captures holding a final row
        self.finals: dict[str, list[dict[str, Any]]] = {}  # their rows, once available
        # race_id -> (its earliest WIN capture, the runners not SCR on it, every runner on it)
        self.opening: dict[str, tuple[datetime, set[int], set[int]]] = {}

    def _offer(self, race_id: str, entry: Entry) -> None:
        slot = self.latest.setdefault(race_id, {})
        key = snapshot._key(entry[2])
        current = slot.get(key)
        if current is None or entry[:2] >= current[:2]:
            slot[key] = entry
        self._note(race_id, key, entry)
        self._open(race_id, key, entry[2])

    def _open(self, race_id: str, key: tuple, row: dict[str, Any]) -> None:
        """The field at the opening: a WIN price row on the race's earliest capture so far."""
        if key[0] != WIN or key[1] == odds_log.POOL_TOTAL:
            return
        snap = odds_log._instant(row.get("snapshot_utc"))
        if snap is None:
            return
        current = self.opening.get(race_id)
        if current is None or snap < current[0]:
            current = self.opening[race_id] = (snap, set(), set())
        elif snap > current[0]:
            return
        current[2].add(key[1])
        if row.get("status_raw") != snapshot.SCRATCHED:
            current[1].add(key[1])

    def _note(self, race_id: str, key: tuple, entry: Entry) -> None:
        """The ladder's bookkeeping for a row now available."""
        _, seq, row = entry
        snap = odds_log._instant(row.get("snapshot_utc"))
        if snap is None or key[1] == odds_log.POOL_TOTAL:
            return
        if snap in self.final_snaps.get(race_id, ()):
            self.finals.setdefault(race_id, []).append(row)
        post = self.post_of(race_id)
        if post is None:
            return
        refs = self.refs.setdefault(race_id, {})
        for k in LADDER_MINUTES:
            if snap > post - timedelta(minutes=k):
                continue
            slot = refs.setdefault(k, {})
            current = slot.get(key)
            if current is None or (snap, seq) >= current[:2]:
                slot[key] = (snap, seq, row)

    def post_of(self, race_id: str) -> datetime | None:
        parts = race_id.split(":")
        no = _race_no(race_id)
        return None if len(parts) != 4 or no is None else self.posts.get(parts[2], {}).get(no)

    def update(
        self,
        venues: list[str],
        now_utc: datetime,
        posts: dict[str, dict[int, datetime]] | None = None,
    ) -> None:
        posts = {} if posts is None else posts
        if (self.horizon is not None and now_utc < self.horizon) or posts != self.posts:
            self._reset()
        self.posts = posts
        for venue in venues:
            path = jra_day_dir(self.race_date, self.day_dir) / f"{venue}{_LOG_SUFFIX}"
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
                if snapshot._is_final(row):  # a capture is one gzip member, read whole
                    snap = odds_log._instant(row.get("snapshot_utc"))
                    if snap is not None:
                        self.final_snaps.setdefault(race_id, set()).add(snap)
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


# ---------------------------------------------------------------- model probabilities (#506)


def _brackets(drawn: int) -> dict[int, int] | None:
    """JRA's draw of horses 1..drawn into the 8 brackets, in number order: N // 8 horses each and
    one more in the last N % 8 brackets (9-16: brackets 1..16-N take one, the rest two; 17: bracket
    8 takes three; 18: brackets 7 and 8). None with 8 or fewer (no bracket quinella) or above 18."""
    if not 9 <= drawn <= 18:
        return None
    base, extra = divmod(drawn, 8)
    out, horse = {}, 1
    for bracket in range(1, 9):
        for _ in range(base + (bracket > 8 - extra)):
            out[horse] = bracket
            horse += 1
    return out


def _probe_until(race_date: date) -> datetime | None:
    """logger.yaml's `day_before.probe_until` (Tokyo) on the day before `race_date`, in UTC: a log
    whose earliest WIN capture of a race is before it covers the opening of sales (#358). None
    without a `day_before` block or a `probe_until` in it."""
    day_before = load_country_yaml("japan", "logger").get("day_before")
    until = day_before.get("probe_until") if isinstance(day_before, dict) else None
    if until is None:
        return None
    at = time.fromisoformat(str(until))
    return datetime.combine(race_date - timedelta(days=1), at, _JST).astimezone(timezone.utc)


def _starters(
    board: RaceBoard,
    opening: tuple[datetime, set[int], set[int]] | None,
    probe_until: datetime | None,
) -> int:
    """The runners in the field at the opening of sales (docs/specs/board-feed.md, "Places"). A
    runner priced on the board counts. `opening` is the race's earliest WIN capture with its runners
    not SCR and all its runners (`_DayRows.opening`; None before one is read): before `probe_until`
    the log covers the opening and its runners not SCR count; otherwise (or without `probe_until`)
    it missed it, so every runner it lists counts and the board's SCR runners too."""
    counted = set(board.win)
    if opening is None:
        return len(counted)
    first, fielded, listed = opening
    if probe_until is not None and first < probe_until:
        return len(counted | fielded)
    return len(counted | listed | board.scratched)  # each scratch taken as after the opening


def _p_model(
    board: RaceBoard,
    models: ShadowModels,
    cache: MutableMapping | None,
    starters: int,
) -> dict[str, dict[str, float]] | None:
    """The board's p_model (docs/specs/board-feed.md), from the cache while the board and the
    starters are the ones last computed for the race; None when it cannot be formed. `starters`:
    the field at the opening of sales (`_starters`), which fixes the places paid."""
    runners = sorted(board.win)
    if not runners:
        return None
    fielded = set(board.win) | set(board.place) | set(board.scratched)
    drawn = max(fielded | {h for pair in board.wide for h in pair})  # drawn before any scratch
    key = (
        board.race_id,
        _iso(board.capture_utc),
        tuple(sorted(models.ids.items())),
        tuple(sorted(board.win.items())),
        drawn,
        starters,
    )
    slot = ("p_model", board.race_id)
    if cache is not None:
        hit = cache.get(slot)
        if hit is not None and hit[0] == key:
            return hit[1]
    try:
        inv = np.array([1.0 / board.win[h] for h in runners])
        p = np.asarray(models.win.predict(inv / inv.sum(), board.context()), dtype=float)
        if p.shape != (len(runners),):
            raise ValueError(f"win.predict returned shape {p.shape}, expected ({len(runners)},)")
        # ordered finishes need the fit's discounts: another ordering model gives WIN only
        ordering = models.ordering if isinstance(models.ordering, OrderingFit) else None
        found = model_probabilities(
            dict(zip(runners, p.tolist(), strict=True)),
            ordering,
            brackets=_brackets(drawn),
            starters=starters,
        )
    except Exception:  # the race shows without probabilities; tried again on the next build
        log.exception("%s: model probabilities failed; shown without", board.race_id)
        return None
    if cache is not None:
        cache[slot] = (key, found)
    return found


# ---------------------------------------------------------------- the odds ladder (#499)


def _drift(nowcast: Any, pool: str) -> Callable[[float, int], float] | None:
    """mu(tau, band) of the nowcast's cells for `pool`: the cell mean on the grid, linear in tau
    between neighbouring grid taus (and from mu(0) = 0 below the smallest), the largest grid tau's
    cell beyond it; None when the nowcast has no cells for the pool (or no cells at all)."""
    cells, grid = getattr(nowcast, "cells", None), getattr(nowcast, "grid", None)
    if not isinstance(cells, dict) or grid is None or not hasattr(nowcast, "cell"):
        return None
    if not any(key[0] == pool for key in cells):
        return None
    taus = sorted(int(t) for t in grid.minutes)

    def at(tau: int, band: int) -> float:
        return 0.0 if tau == 0 else float(nowcast.cell(pool, tau, band).mu)

    def mu(tau: float, band: int) -> float:
        """Reads only the exact grid cell or the two bracketing tau; KeyError for a missing one."""
        if tau <= 0:
            return 0.0
        if tau >= taus[-1]:
            return at(taus[-1], band)
        i = bisect_left(taus, tau)
        hi = taus[i]
        if hi == tau:
            return at(hi, band)
        lo = taus[i - 1] if i > 0 else 0
        m0, m1 = at(lo, band), at(hi, band)
        return m0 + (tau - lo) / (hi - lo) * (m1 - m0)

    return mu


def _projector(models: ShadowModels | None):
    """(pool, odds_now, tau_now, k) -> the projected median odds at post - k, or None when the
    nowcast cannot project it (no cells for the pool, or a cell it needs is missing):
    exp(ln O_now + mu(tau_now, b) - mu(k, b)), b the band of O_now, floored at 1.0 and rounded to
    two decimals."""
    nowcast = None if models is None else models.nowcast
    bands = getattr(nowcast, "bands", None)
    drifts: dict[str, Any] = {}

    def project(pool: str, odds_now: float, tau_now: float | None, k: int) -> float | None:
        if tau_now is None or bands is None or odds_now <= 0:
            return None
        if pool not in drifts:
            drifts[pool] = _drift(nowcast, pool)
        mu = drifts[pool]
        if mu is None:
            return None
        band = band_of(odds_now, bands)
        try:
            shift = mu(tau_now, band) - mu(k, band)
        except KeyError:  # a cell the projection needs that the fit has no data for
            return None
        return round(max(FLOOR, math.exp(math.log(odds_now) + shift)), 2)

    return project


def _low(figure: float | tuple[float, float | None] | None) -> float | None:
    if figure is None:
        return None
    return float(figure[0] if isinstance(figure, tuple) else figure)


def _ladders(
    board: RaceBoard,
    day_rows: _DayRows,
    post: datetime | None,
    now_utc: datetime,
    models: ShadowModels | None,
) -> dict[tuple[str, str], dict[int, dict | None]]:
    """(pool, combination) -> {20, 10, 5, 2, 0: {"odds", "observed"} or None} (docs/specs/
    board-feed.md, "The odds ladder"). Display only: nothing here feeds a value or a flag."""
    race_id = board.race_id
    final = None
    if day_rows.finals.get(race_id):
        final = final_board(day_rows.finals[race_id], race_id, post_utc=post)
    project = _projector(models)
    current = day_rows.latest.get(race_id, {})  # the row each board figure comes from
    refs = day_rows.refs.get(race_id, {})
    figures = {WIN: board.win, PLACE: board.place, WIDE: board.wide}
    finals = {} if final is None else {WIN: final.win, PLACE: final.place, WIDE: final.wide}
    out: dict[tuple[str, str], dict[int, dict | None]] = {}
    for pool, shown in figures.items():
        for runners, figure in shown.items():
            combination = f"{runners[0]},{runners[1]}" if pool == WIDE else str(runners)
            odds_now, key = _low(figure), (pool, runners)  # key: the board key of snapshot._key
            # tau_now: minutes to the post from the fetch of the row O_now comes from (its
            # availability: fetched_utc, else snapshot_utc; snapshot._available)
            entry = current.get(key)
            fetched = None if entry is None else entry[0]
            tau_now = None
            if post is not None and fetched is not None:
                tau_now = max(0.0, (post - fetched) / timedelta(minutes=1))
            ladder: dict[int, dict | None] = {}
            for k in (*LADDER_MINUTES, 0):
                if k == 0:
                    seen = _low(finals.get(pool, {}).get(runners))
                    if seen is not None:
                        ladder[0] = {"odds": seen, "observed": True}
                        continue
                elif post is not None and now_utc >= post - timedelta(minutes=k):
                    ref = refs.get(k, {}).get(key)
                    row = None if ref is None else ref[2]
                    priced = row is not None and snapshot._priced(row)
                    ladder[k] = {"odds": float(row["odds"]), "observed": True} if priced else None
                    continue
                odds = None if odds_now is None else project(pool, odds_now, tau_now, k)
                ladder[k] = None if odds is None else {"odds": odds, "observed": False}
            out[(pool, combination)] = ladder
    return out


def _pools(board: RaceBoard, flags: dict, values: dict, ladders: dict) -> dict[str, dict]:
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
        entry["ladder"] = ladders.get((pool, combination))
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
    day_dir: Path | None = None,
) -> dict[str, Any]:
    """The shadow values and avoid flags of every race of `race_date` with a board as of now()."""
    now_utc = now().astimezone(timezone.utc)
    grid = load_decision_grid()
    logged = _venues_logged(race_date, day_dir)
    if venues is not None:
        wanted = set(venues)
        logged = [venue for venue in logged if venue in wanted]
    key = ("logs", race_date, day_dir)
    day_rows = cache.get(key) if cache is not None else None
    if day_rows is None:
        day_rows = _DayRows(race_date, day_dir)
        if cache is not None:
            cache[key] = day_rows
    post_times = {venue: post_times_for(race_date, venue, day_dir=day_dir) for venue in logged}
    day_rows.update(logged, now_utc, post_times)
    probe_until = _probe_until(race_date)

    races = []
    for venue in logged:
        posts = post_times[venue]
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
            p_model = None
            if models is not None:
                starters = _starters(board, day_rows.opening.get(race_id), probe_until)
                p_model = _p_model(board, models, cache, starters)
            races.append(
                {
                    "race_id": race_id,
                    "venue": venue,
                    "race_no": race_no,
                    "post_utc": _iso(post),
                    "minutes_to_post": None if minutes is None else round(minutes, 2),
                    "tau_bucket_min": bucket,
                    "board_capture_utc": _iso(board.capture_utc),
                    "pools": _pools(
                        board, flags, values, _ladders(board, day_rows, post, now_utc, models)
                    ),
                    "p_model": p_model,
                }
            )
    return {
        "race_date": race_date.isoformat(),
        "generated_utc": _iso(now_utc),
        "models": None if models is None else dict(models.ids),
        "races": races,
        "win5": live_pool_ev(day_rows.win5, race_date, now=now_utc),
    }
