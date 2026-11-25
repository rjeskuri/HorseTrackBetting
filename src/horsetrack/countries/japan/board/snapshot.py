"""The board at a decision time t, the final board, post times and the decision grid.

Rules (docs/specs/board-at-t.md):
- A row is available at its `fetched_utc`, else at its `snapshot_utc` (HKTrack rows have no
  fetched_utc). The board at t takes, per (pool, combination), the latest row available at or
  before t; `capture_utc` is the oldest availability time among the price rows on the board and
  the SCR rows (a stale row dropped for naming a scratched runner does not count).
- A row is final when `source_update_utc` is null and '最終オッズ' is in `page_state` or
  `status_raw`. A capture (rows sharing a snapshot_utc) is final for a pool when it holds a priced
  row of that pool and every one is final. The final board starts from the anchor, the first
  capture final for WIN; PLACE and Wide come from each pool's first final capture at or after the
  anchor, else its latest one before it (HKTrack logs Wide as its own capture ~2 s after WIN+PLACE).
- `bucket_for` maps minutes to post onto the smallest grid horizon at or above it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from horsetrack.core import config, paths
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.odds_log import POOL_TOTAL

FINAL_TEXT = "最終オッズ"
SCRATCHED = "SCR"
WIN, PLACE, WIDE = "WIN", "PLACE", "QUINELLA_PLACE"
_PRICE_POOLS = (WIN, PLACE, WIDE)
_JST = timezone(timedelta(hours=9))  # Japan keeps no daylight saving time

_instant = odds_log._instant  # one reading of logged timestamps for the log and the board


@dataclass(frozen=True)
class RaceContext:
    race_id: str
    venue: str
    race_no: int
    starters: int


@dataclass(frozen=True)
class RaceBoard:
    race_id: str
    post_utc: datetime | None
    at_utc: datetime
    capture_utc: datetime
    final: bool
    win: dict[int, float]
    place: dict[int, tuple[float, float | None]]
    wide: dict[tuple[int, int], tuple[float, float | None]]
    totals: dict[str, float]
    scratched: frozenset[int]

    @property
    def starters(self) -> int:
        """Runners with a WIN price."""
        return len(self.win)

    def context(self) -> RaceContext:
        _, _, venue, race_no = self.race_id.split(":")
        return RaceContext(self.race_id, venue, int(race_no), self.starters)


# ---------------------------------------------------------------- rows


def _available(row: dict[str, Any]) -> datetime | None:
    """When the row could first be known: its fetch time, else its capture time."""
    fetched = _instant(row.get("fetched_utc"))
    return fetched if fetched is not None else _instant(row.get("snapshot_utc"))


def _key(row: dict[str, Any]) -> tuple[str, Any] | None:
    """(pool, runner) for WIN and PLACE, (pool, (low, high)) for Wide, (pool, POOL_TOTAL) for a
    total; None for a row the board does not read."""
    pool, combination = row.get("pool"), row.get("combination")
    if combination == POOL_TOTAL:
        return (pool, POOL_TOTAL) if row.get("investment") is not None else None
    try:
        if pool in (WIN, PLACE):
            return pool, int(combination)
        if pool == WIDE:
            a, b = sorted(int(part) for part in str(combination).split(","))
            return pool, (a, b)
    except (TypeError, ValueError):
        return None
    return None


def _is_final(row: dict[str, Any]) -> bool:
    if row.get("source_update_utc") is not None:
        return False
    return any(FINAL_TEXT in (row.get(field) or "") for field in ("page_state", "status_raw"))


def _priced(row: dict[str, Any]) -> bool:
    return row.get("odds") is not None and row.get("status_raw") != SCRATCHED


def _pool_rows_final(rows: Iterable[dict[str, Any]], pool: str) -> bool:
    """Every priced row of `pool` is final, and there is at least one."""
    priced = [r for r in rows if r.get("pool") == pool and _priced(r)]
    return bool(priced) and all(_is_final(r) for r in priced)


def _win_rows_final(rows: Iterable[dict[str, Any]]) -> bool:
    """Every priced WIN row is final, and there is at least one."""
    return _pool_rows_final(rows, WIN)


def _latest(rows: Iterable[tuple[datetime, dict[str, Any]]]) -> dict[tuple, tuple[datetime, dict]]:
    """Latest row per key by availability; at the same instant the later row given wins."""
    best: dict[tuple, tuple[datetime, dict[str, Any]]] = {}
    for available, row in rows:
        key = _key(row)
        if key is None:
            continue
        current = best.get(key)
        if current is None or available >= current[0]:
            best[key] = (available, row)
    return best


def _build(
    race_id: str,
    latest: dict[tuple, tuple[datetime, dict[str, Any]]],
    at_utc: datetime,
    post_utc: datetime | None,
) -> RaceBoard | None:
    prices = {k: v for k, v in latest.items() if k[1] != POOL_TOTAL}
    if not prices:
        return None
    scratched = frozenset(
        runner
        for (pool, runner), (_, row) in prices.items()
        if pool in (WIN, PLACE) and row.get("status_raw") == SCRATCHED
    )
    win: dict[int, float] = {}
    place: dict[int, tuple[float, float | None]] = {}
    wide: dict[tuple[int, int], tuple[float, float | None]] = {}
    used: list[datetime] = []  # availability of the rows the board shows, SCR rows included
    for (pool, combination), (available, row) in prices.items():
        if pool in (WIN, PLACE) and row.get("status_raw") == SCRATCHED:
            used.append(available)
            continue
        if not _priced(row):
            continue
        runners = combination if pool == WIDE else (combination,)
        if any(runner in scratched for runner in runners):
            continue
        used.append(available)
        odds = float(row["odds"])
        if pool == WIN:
            win[combination] = odds
        else:
            hi = row.get("odds_hi")
            (place if pool == PLACE else wide)[combination] = (
                odds,
                None if hi is None else float(hi),
            )
    totals = {
        pool: float(row["investment"])
        for (pool, combination), (_, row) in latest.items()
        if combination == POOL_TOTAL
    }
    win_rows = [row for (pool, _), (_, row) in prices.items() if pool == WIN]
    return RaceBoard(
        race_id=race_id,
        post_utc=post_utc,
        at_utc=at_utc,
        capture_utc=min(used or (available for available, _ in prices.values())),
        final=_win_rows_final(win_rows),
        win=win,
        place=place,
        wide=wide,
        totals=totals,
        scratched=scratched,
    )


def _race_rows(rows: Iterable[dict[str, Any]], race_id: str) -> list[tuple[datetime, dict]]:
    out = []
    for row in rows:
        if row.get("race_id") != race_id:
            continue
        available = _available(row)
        if available is not None:
            out.append((available, row))
    return out


# ---------------------------------------------------------------- boards


def board_at(
    rows: Iterable[dict[str, Any]],
    race_id: str,
    at_utc: datetime,
    *,
    post_utc: datetime | None = None,
) -> RaceBoard | None:
    """The board of `race_id` as it could be known at `at_utc`; None when no price row of the race
    (WIN, PLACE or Wide) is available by then."""
    usable = [(a, r) for a, r in _race_rows(rows, race_id) if a <= at_utc]
    return _build(race_id, _latest(usable), at_utc, post_utc)


def final_board(
    rows: Iterable[dict[str, Any]],
    race_id: str,
    *,
    post_utc: datetime | None = None,
) -> RaceBoard | None:
    """The final board of `race_id`; None when no capture's priced WIN rows are all final.

    The anchor is the first capture (by snapshot_utc) whose priced WIN rows are all final, and the
    board starts from its rows. PLACE and Wide, each with its SCR rows and pool total, come instead
    from that pool's first final capture at or after the anchor, else its latest final capture
    before it, else are left off. `at_utc` is when the last row taken was available."""
    captures: dict[datetime, list[tuple[datetime, dict[str, Any]]]] = {}
    for available, row in _race_rows(rows, race_id):
        snap = _instant(row.get("snapshot_utc"))
        if snap is not None:
            captures.setdefault(snap, []).append((available, row))
    snaps = sorted(captures)
    anchor = next((s for s in snaps if _win_rows_final(row for _, row in captures[s])), None)
    if anchor is None:
        return None
    taken = [(a, row) for a, row in captures[anchor] if row.get("pool") not in (PLACE, WIDE)]
    for pool in (PLACE, WIDE):
        finals = [s for s in snaps if _pool_rows_final((row for _, row in captures[s]), pool)]
        source = next((s for s in finals if s >= anchor), finals[-1] if finals else None)
        if source is not None:
            taken += [(a, row) for a, row in captures[source] if row.get("pool") == pool]
    at_utc = max(available for available, _ in taken)
    return _build(race_id, _latest(taken), at_utc, post_utc)


# ---------------------------------------------------------------- post times


def _card_path(race_date: date, venue: str) -> Path:
    folder = f"jra_{race_date.isoformat()}_{venue}"
    return paths.processed_dir("jra") / "upcoming" / folder / "races.parquet"


def _card_post_times(path: Path) -> dict[int, datetime]:
    """HKTrack's stored card: race_no -> post_time_local (ISO, +09:00), as aware UTC."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["race_no", "post_time_local"])
    out: dict[int, datetime] = {}
    for no, text in zip(
        table.column("race_no").to_pylist(),
        table.column("post_time_local").to_pylist(),
        strict=True,
    ):
        if no is None or text is None:
            continue
        local = text if isinstance(text, datetime) else datetime.fromisoformat(str(text))
        if local.tzinfo is None:
            local = local.replace(tzinfo=_JST)
        out[int(no)] = local.astimezone(timezone.utc)
    return out


def post_times_for(race_date: date, venue: str) -> dict[int, datetime]:
    """Race number -> post time (aware UTC) of one meeting: the `<VENUE>.races.json` sidecar beside
    the odds log, else HKTrack's stored card, else empty."""
    log = paths.odds_snapshot_path("jra", race_date.isoformat(), venue)
    sidecar = odds_log.read_post_times(log.with_name(f"{venue}.races.json"))
    if sidecar:
        return sidecar
    card = _card_path(race_date, venue)
    return _card_post_times(card) if card.is_file() else {}


# ---------------------------------------------------------------- decision grid


@dataclass(frozen=True)
class DecisionGrid:
    minutes: tuple[int, ...]
    primary: int

    def __post_init__(self) -> None:
        if not self.minutes or any(m <= 0 for m in self.minutes):
            raise ValueError(f"decision grid needs positive minutes, got {self.minutes!r}")
        if self.primary not in self.minutes:
            raise ValueError(f"primary {self.primary} is not on the grid {self.minutes!r}")


def load_decision_grid(path: Path | None = None) -> DecisionGrid:
    """The decision grid from `path`, else the shipped countries/japan/config/decision.yaml."""
    if path is None:
        raw = config.load_country_yaml("japan", "decision")
    else:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    return DecisionGrid(
        minutes=tuple(int(m) for m in raw["grid_min"]), primary=int(raw["primary_min"])
    )


def bucket_for(minutes_to_post: float, grid: DecisionGrid) -> int:
    """The smallest grid horizon at or above `minutes_to_post`, the largest for anything earlier;
    0 at or after the post, when the race no longer takes bets."""
    if minutes_to_post <= 0:
        return 0
    above = [m for m in grid.minutes if m >= minutes_to_post]
    return min(above) if above else max(grid.minutes)
