"""Per-race payout rates with the JRA Premium boosts (tickets #76, #197).

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

An entry covers one day, or every day from `date` to `until` (both included). An entry with `age: N`
matches only races restricted to N-year-olds: the race's conditions label (the store's
`race_class_raw`, e.g. "2歳未勝利") starts with "N歳" and is not an "N歳以上" label. The race id
cannot tell those races apart, so `race_takeout` takes the label as `race_class_raw` and raises
RaceConditionsNeeded when an age entry matches the race's date, venue, race and pool and no label
was given. Entries without `age` ignore the label.
"""

from __future__ import annotations

from collections.abc import Iterable
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
_FIELDS = {"kind", "date", "until", "age", "venue", "race_no", "pools"}


class TakeoutUnknown(LookupError):
    """No single takeout applied on that date (before 2014-06-07): use the race's overround."""


class RaceConditionsNeeded(ValueError):
    """An age-restricted entry matches the race: pass its conditions label as race_class_raw."""


@dataclass(frozen=True)
class Boost:
    kind: str
    date: date
    venue: str | None = None
    race_no: int | None = None
    pools: tuple[str, ...] | None = None
    until: date | None = None  # the last day covered, inclusive; None means `date` alone
    age: int | None = None  # only races restricted to this age (see age_matches)

    def matches(self, day: date, venue: str, race_no: int, pool: str) -> bool:
        """Date, venue, race and pool; the age restriction is age_matches'."""
        return (
            self.date <= day <= (self.until or self.date)
            and self.venue in (None, venue)
            and self.race_no in (None, race_no)
            and (self.pools is None or pool in self.pools)
        )

    def age_matches(self, race_class_raw: str) -> bool:
        """True without `age`; else the label starts with "<age>歳" and not with "<age>歳以上"."""
        if self.age is None:
            return True
        prefix = f"{self.age}歳"
        return race_class_raw.startswith(prefix) and not race_class_raw.startswith(f"{prefix}以上")


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
    day = _date(item, "date")
    until = _date(item, "until") if "until" in item else None
    if until is not None and until < day:
        raise ValueError(f"boost entry {item!r}: until is before date")
    age = item.get("age")
    if age is not None and (type(age) is not int or age < 1):
        raise ValueError(f"boost entry {item!r}: age must be a positive integer")
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
    return Boost(
        kind=kind, date=day, venue=venue, race_no=race_no, pools=pools, until=until, age=age
    )


def _date(item: dict[str, Any], key: str) -> date:
    value = item.get(key)
    if isinstance(value, str):
        try:
            value = date.fromisoformat(value)
        except ValueError:
            raise ValueError(f"boost entry {item!r}: {key} must be YYYY-MM-DD") from None
    if not isinstance(value, date):
        raise ValueError(f"boost entry {item!r}: {key} must be YYYY-MM-DD")
    return value


@cache
def shipped_boosts() -> BoostCalendar:
    """The shipped calendar, loaded once."""
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


def race_takeout(
    race_id: str,
    pool: str,
    *,
    calendar: BoostCalendar | None = None,
    race_class_raw: str | None = None,
) -> RaceTakeout:
    """The payout rate `pool` paid in `race_id`; calendar None means the shipped calendar.

    `race_class_raw` is the race's conditions label (the store's column); only age entries read it.
    Raises TakeoutUnknown before 2014-06-07, RaceConditionsNeeded when an age entry matches the race
    and pool and race_class_raw is None, and ValueError for a malformed race id or an unknown pool
    code.
    """
    jra = config.load_source("japan", "jra")
    day, venue, race_no = _parse_race_id(race_id, jra)
    if pool not in jra.pools:
        raise ValueError(f"unknown pool code {pool!r}")
    takeout = jra.pools[pool].takeout_on(day)
    if takeout is None:
        raise TakeoutUnknown(f"{pool} takeout on {day} is unknown (no single rate before history)")
    base = 1 - takeout

    cal = calendar if calendar is not None else shipped_boosts()
    matched = cal.matching(day, venue, race_no, pool)
    if race_class_raw is None and any(e.age is not None for e in matched):
        raise RaceConditionsNeeded(
            f"{race_id} {pool}: an age-restricted boost matches; pass race_class_raw"
        )
    kinds = {e.kind for e in matched if race_class_raw is None or e.age_matches(race_class_raw)}
    rate, boost = combine_boosts(base, kinds)
    return RaceTakeout(
        pool=pool,
        base_rate=round(base, 6),  # drops float noise only (JRA quotes rates to 0.01%)
        payout_rate=rate,
        boost=boost,
        boost_may_be_withheld=bool(kinds & _TOPS_UP),
    )


def combine_boosts(base_rate: float, kinds: Iterable[str]) -> tuple[float, str | None]:
    """(payout rate rounded to 6 places, boost) for a pool paying `base_rate` when entries of
    `kinds` match it: a super_premium or ultra_premium entry sets the rate to SUPER_PREMIUM_RATE,
    a premium or ultra_premium entry adds PREMIUM_TOP_UP to the rate in force."""
    kinds = set(kinds)
    sets_80 = bool(kinds & _SETS_80)
    tops_up = bool(kinds & _TOPS_UP)
    rate = SUPER_PREMIUM_RATE if sets_80 else base_rate
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
    return round(rate, 6), boost
