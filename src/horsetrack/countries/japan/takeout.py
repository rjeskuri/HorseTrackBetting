"""Per-race payout rates with the JRA Premium boosts (ticket #76).

Takeout comes from a per-race lookup (date x pool x boost), never a constant. The base rate is
1 - Pool.takeout_on(date) from jra.yaml; the boost calendar (config/takeout_boosts.yaml) lists the
days, races and pools JRA boosted:

- premium: 5% of sales added to the pool's payout; JRA may withhold it when betting is extremely
  concentrated, so `boost_may_be_withheld` is True.
- super_premium: every pool's payout rate set to 80%.
- ultra_premium: super_premium and premium together, 80% + 5%.

Several entries may match one race and pool. A super_premium or ultra_premium entry sets the rate
to 80%, and a premium or ultra_premium entry adds 5% to the rate in force, so a premium on a Super
Premium day stacks to 85% (JRA's result PDFs for 2022-12-28 and 2023-12-28 show this). `boost` names
the effect: "ultra_premium" for 80% + 5%, else "super_premium" or "premium".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from horsetrack.core import config

KINDS = ("premium", "super_premium", "ultra_premium")
SUPER_PREMIUM_RATE = 0.80
PREMIUM_TOP_UP = 0.05

_SETS_80 = {"super_premium", "ultra_premium"}
_TOPS_UP = {"premium", "ultra_premium"}
_FIELDS = {"kind", "date", "venue", "race_no", "pools"}


class TakeoutUnknown(LookupError):
    """No single takeout applied on that date (before 2014-06-07): use the race's overround."""


@dataclass(frozen=True)
class Boost:
    kind: str
    date: date
    venue: str | None = None
    race_no: int | None = None
    pools: tuple[str, ...] | None = None

    def matches(self, day: date, venue: str, race_no: int, pool: str) -> bool:
        return (
            self.date == day
            and self.venue in (None, venue)
            and self.race_no in (None, race_no)
            and (self.pools is None or pool in self.pools)
        )


@dataclass(frozen=True)
class BoostCalendar:
    entries: tuple[Boost, ...]

    def matching(self, day: date, venue: str, race_no: int, pool: str) -> list[Boost]:
        return [e for e in self.entries if e.matches(day, venue, race_no, pool)]


@dataclass(frozen=True)
class RaceTakeout:
    pool: str
    base_rate: float  # 1 - the pool's takeout on that date
    payout_rate: float  # the rate the race paid, boost included
    boost: str | None
    boost_may_be_withheld: bool


def load_boosts(path: Path | None = None) -> BoostCalendar:
    """The calendar at `path`, else the shipped countries/japan/config/takeout_boosts.yaml."""
    if path is None:
        raw = config.load_country_yaml("japan", "takeout_boosts")
    else:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict) or set(raw) != {"boosts"} or not isinstance(raw["boosts"], list):
        raise ValueError("a boost calendar is a mapping with one key, `boosts`, holding a list")
    jra = config.load_source("japan", "jra")
    return BoostCalendar(tuple(_entry(item, jra) for item in raw["boosts"]))


def _entry(item: Any, jra: config.SourceConfig) -> Boost:
    if not isinstance(item, dict):
        raise ValueError(f"boost entry {item!r} is not a mapping")
    unknown = set(item) - _FIELDS
    if unknown:
        raise ValueError(f"boost entry {item!r} has unknown keys {sorted(unknown)}")
    kind = item.get("kind")
    if kind not in KINDS:
        raise ValueError(f"boost entry {item!r}: kind must be one of {KINDS}")
    day = item.get("date")
    if isinstance(day, str):
        day = date.fromisoformat(day)
    if not isinstance(day, date):
        raise ValueError(f"boost entry {item!r}: date must be YYYY-MM-DD")
    venue = item.get("venue")
    if venue is not None and venue not in jra.venues:
        raise ValueError(f"boost entry {item!r}: unknown venue {venue!r}")
    race_no = item.get("race_no")
    if race_no is not None and (type(race_no) is not int or race_no < 1):
        raise ValueError(f"boost entry {item!r}: race_no must be a positive integer")
    pools = item.get("pools")
    if pools is not None:
        if not isinstance(pools, list) or not pools or any(p not in jra.pools for p in pools):
            raise ValueError(f"boost entry {item!r}: pools must be a non-empty list of pool codes")
        pools = tuple(pools)
    return Boost(kind=kind, date=day, venue=venue, race_no=race_no, pools=pools)


@cache
def _shipped() -> BoostCalendar:
    return load_boosts()


def _parse_race_id(race_id: str, jra: config.SourceConfig) -> tuple[date, str, int]:
    """jra:<YYYY-MM-DD>:<VENUE>:<NN> -> (date, venue, race number); ValueError otherwise."""
    parts = str(race_id).split(":")
    if len(parts) != 4 or parts[0] != "jra":
        raise ValueError(f"race id {race_id!r} is not jra:<YYYY-MM-DD>:<VENUE>:<NN>")
    _, day, venue, number = parts
    if venue not in jra.venues:
        raise ValueError(f"race id {race_id!r}: unknown venue {venue!r}")
    if not number.isdigit() or int(number) < 1:
        raise ValueError(f"race id {race_id!r}: race number must be a positive integer")
    return date.fromisoformat(day), venue, int(number)


def race_takeout(race_id: str, pool: str, *, calendar: BoostCalendar | None = None) -> RaceTakeout:
    """The payout rate `pool` paid in `race_id`; calendar None means the shipped calendar.

    Raises TakeoutUnknown before 2014-06-07, and ValueError for a malformed race id or an unknown
    pool code.
    """
    jra = config.load_source("japan", "jra")
    day, venue, race_no = _parse_race_id(race_id, jra)
    if pool not in jra.pools:
        raise ValueError(f"unknown pool code {pool!r}")
    takeout = jra.pools[pool].takeout_on(day)
    if takeout is None:
        raise TakeoutUnknown(f"{pool} takeout on {day} is unknown (no single rate before history)")
    base = 1 - takeout

    cal = calendar if calendar is not None else _shipped()
    kinds = {e.kind for e in cal.matching(day, venue, race_no, pool)}
    sets_80 = bool(kinds & _SETS_80)
    tops_up = bool(kinds & _TOPS_UP)
    rate = SUPER_PREMIUM_RATE if sets_80 else base
    if tops_up:
        rate += PREMIUM_TOP_UP
    if sets_80 and tops_up:
        boost = "ultra_premium"
    elif sets_80:
        boost = "super_premium"
    elif tops_up:
        boost = "premium"
    else:
        boost = None
    return RaceTakeout(
        pool=pool,
        base_rate=round(base, 6),  # drops float noise only (JRA quotes rates to 0.01%)
        payout_rate=round(rate, 6),
        boost=boost,
        boost_may_be_withheld=tops_up,
    )
