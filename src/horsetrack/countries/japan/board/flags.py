"""The avoid flags on the board at t (ticket #117; D18 loss-avoidance candidates, #82 L2 and L3).

Information on the avoid rung (D19's default): nothing is adopted and nothing is staked. The flags
read only the board at t and the clock.

- floor: WIN odds, or the PLACE / Wide low figure, at or below `floor_max_odds`.
- tail: a WIN or PLACE key of a runner whose WIN odds are above `tail_min_odds`.
- tail_member: a Wide pair with a runner whose WIN-odds rank is `tail_rank` or worse. Rank is 1 +
  the number of runners with strictly lower WIN odds (tied odds take the better rank); scratched
  runners are not ranked.
- stale: every key, when `capture_utc` is more than `stale_s` seconds before `now_utc`.
- scratched: the WIN and PLACE keys of a scratched runner.

The result maps (pool, combination) to the flags of that key, sorted by name; keys without a flag
are absent. Wide combinations read "a,b" with a < b.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from horsetrack.core import config
from horsetrack.countries.japan.board.snapshot import PLACE, WIDE, WIN, RaceBoard


@dataclass(frozen=True)
class FlagRules:
    floor_max_odds: float
    tail_min_odds: float
    tail_rank: int
    stale_s: float


def load_flag_rules(path: Path | None = None) -> FlagRules:
    """The rules from `path`, else the shipped countries/japan/config/avoid_flags.yaml."""
    if path is None:
        raw = config.load_country_yaml("japan", "avoid_flags")
    else:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    return FlagRules(
        floor_max_odds=float(raw["floor_max_odds"]),
        tail_min_odds=float(raw["tail_min_odds"]),
        tail_rank=int(raw["tail_rank"]),
        stale_s=float(raw["stale_s"]),
    )


def _win_ranks(board: RaceBoard) -> dict[int, int]:
    """Runner -> 1 + the number of runners with strictly lower WIN odds."""
    odds = sorted(board.win.values())
    return {
        runner: 1 + sum(1 for other in odds if other < price) for runner, price in board.win.items()
    }


def avoid_flags(
    board: RaceBoard, *, rules: FlagRules, now_utc: datetime
) -> dict[tuple[str, str], tuple[str, ...]]:
    """The avoid flags of every key on `board` as of `now_utc` (an aware datetime)."""
    if now_utc.tzinfo is None or now_utc.utcoffset() is None:
        raise ValueError("now_utc must be timezone-aware")

    flags: dict[tuple[str, str], set[str]] = {}

    def mark(pool: str, combination: str, flag: str | None = None) -> None:
        entry = flags.setdefault((pool, combination), set())
        if flag is not None:
            entry.add(flag)

    ranks = _win_ranks(board)
    tail_runners = {n for n, price in board.win.items() if price > rules.tail_min_odds}

    for runner, price in board.win.items():
        mark(WIN, str(runner))
        if price <= rules.floor_max_odds:
            mark(WIN, str(runner), "floor")
        if runner in tail_runners:
            mark(WIN, str(runner), "tail")
    for runner, (low, _high) in board.place.items():
        mark(PLACE, str(runner))
        if low <= rules.floor_max_odds:
            mark(PLACE, str(runner), "floor")
        if runner in tail_runners:
            mark(PLACE, str(runner), "tail")
    for (a, b), (low, _high) in board.wide.items():
        key = f"{a},{b}"
        mark(WIDE, key)
        if low <= rules.floor_max_odds:
            mark(WIDE, key, "floor")
        if any(ranks.get(runner, 0) >= rules.tail_rank for runner in (a, b)):
            mark(WIDE, key, "tail_member")
    for runner in board.scratched:
        mark(WIN, str(runner), "scratched")
        mark(PLACE, str(runner), "scratched")

    if now_utc - board.capture_utc > timedelta(seconds=rules.stale_s):
        for key in flags:
            flags[key].add("stale")

    return {key: tuple(sorted(found)) for key, found in flags.items() if found}
