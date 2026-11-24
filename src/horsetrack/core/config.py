"""Typed access to a country's source config (countries/<country>/config/<source>.yaml).

Pool and venue specifics are data, never code. The YAML ships inside the package, so it is found
relative to this file rather than the working directory.
"""

from __future__ import annotations

from datetime import date
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

_COUNTRIES = Path(__file__).resolve().parent.parent / "countries"


class _Strict(BaseModel):
    # An unknown key is a typo or a rule this model does not implement: fail loudly, never drop it.
    model_config = ConfigDict(extra="forbid")


class PlacesRule(_Strict):
    min_starters: int
    places: int


class Pool(_Strict):
    code: str
    labels: list[str]
    takeout: float
    legs: int
    kind: str
    enabled: bool = False
    dividend_step: float = 1.0
    min_dividend: float | None = None
    takeout_history: list[tuple[date, float]] = Field(default_factory=list)
    # True where no single rate applied before the first `takeout_history` entry
    takeout_unknown_before_history: bool = False
    places_paid: list[PlacesRule] = Field(default_factory=list)
    # equal_split: the net pool is split equally between winning tickets.
    # stake_back_share: a winning ticket gets its stake back plus 1/P of the losers' money.
    payout_rule: Literal["equal_split", "stake_back_share"] = "equal_split"

    def takeout_on(self, when: date) -> float | None:
        """Takeout in force on `when`; None before the first history entry when the rate before it
        is unknown (use the race's overround instead)."""
        history = sorted(self.takeout_history)
        if self.takeout_unknown_before_history and history and when < history[0][0]:
            return None
        rate = self.takeout
        for effective_from, value in history:
            if when >= effective_from:
                rate = value
        return rate

    def places_for(self, starters: int) -> int:
        """Finishing positions this pool pays for a field of `starters` (0 = pool not offered)."""
        for rule in sorted(self.places_paid, key=lambda r: -r.min_starters):
            if starters >= rule.min_starters:
                return rule.places
        return 0


class Venue(_Strict):
    code: str
    name: str
    number: str | None = None  # the netkeiba venue number (JRA); HKJC venues have none


class Hosts(_Strict):
    # Each source names only its own hosts: official/db/race (JRA), html/graphql (HKJC).
    official: str | None = None
    db: str | None = None
    race: str | None = None
    html: str | None = None
    graphql: str | None = None


class Season(_Strict):
    start_month: int = Field(ge=1, le=12)  # a season starts on the 1st of this month


class RateLimit(_Strict):
    requests_per_second: float
    jitter: float


class CircuitBreaker(_Strict):
    stop_on_status: list[int]
    max_consecutive_failures: int


class Http(_Strict):
    default: RateLimit
    hosts: dict[str, RateLimit] = Field(default_factory=dict)
    circuit_breaker: CircuitBreaker


class SourceConfig(_Strict):
    source: str
    display_name: str
    timezone: str
    currency: str
    unit_stake: float
    odds_format: str
    hosts: Hosts
    season: Season | None = None
    venues: dict[str, Venue]
    # rule breaks features must respect: name -> the first day of the new rule
    eras: dict[str, date] = Field(default_factory=dict)
    pools: dict[str, Pool]
    odds_display: dict[str, float] = Field(default_factory=dict)
    http: Http

    def pool_by_label(self, label: str) -> Pool | None:
        """The pool a Japanese or English label names (English in any case), else None."""
        wanted = label.strip().upper()
        for pool in self.pools.values():
            if wanted in (lab.upper() for lab in pool.labels):
                return pool
        return None

    def venue_by_name(self, name: str) -> Venue | None:
        """The venue whose printed name is `name`, else None."""
        wanted = name.strip()
        for venue in self.venues.values():
            if venue.name == wanted:
                return venue
        return None


@cache
def load_country_yaml(country: str, name: str) -> dict[str, Any]:
    """countries/<country>/config/<name>.yaml as a plain dict (cached and shared: do not mutate)."""
    path = _COUNTRIES / country / "config" / f"{name}.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@cache
def load_source(country: str, source: str) -> SourceConfig:
    """The typed config of one source in one country, e.g. load_source("japan", "jra")."""
    raw = load_country_yaml(country, source)
    return SourceConfig(
        **{
            **raw,
            "pools": {code: {"code": code, **spec} for code, spec in raw.get("pools", {}).items()},
            "venues": {
                code: {"code": code, **spec} for code, spec in raw.get("venues", {}).items()
            },
        }
    )
