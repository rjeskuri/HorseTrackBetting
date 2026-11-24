"""netkeiba URL patterns (ported from HKTrack's `sources/jra/urls.py`). Pure functions.

A race id is 12 ASCII digits: YYYY + venue number + meeting (kai) + day + race number. JRA venue
numbers are those of `countries/japan/config/jra.yaml` (01-10). The same pages also link local
(NAR) ids with other venue numbers and overseas ids with letters; only JRA ids are ever turned
into a URL or a cache file name.
"""

from __future__ import annotations

import re
from datetime import date

from horsetrack.core.config import load_source
from horsetrack.countries.japan.jra.netkeiba.common import COUNTRY, SOURCE

_RACE_ID = re.compile(r"[0-9]{12}")  # [0-9], not \d: \d also matches full-width digits


def _host(name: str) -> str:
    return getattr(load_source(COUNTRY, SOURCE).hosts, name).rstrip("/")


def _venue_numbers() -> set[str]:
    return {venue.number for venue in load_source(COUNTRY, SOURCE).venues.values()}


def jra_race_id(value: object) -> str | None:
    """`value` if it is a JRA race id, else None."""
    if isinstance(value, str) and _RACE_ID.fullmatch(value) and value[4:6] in _venue_numbers():
        return value
    return None


def _checked(race_id: str) -> str:
    if jra_race_id(race_id) is None:
        raise ValueError(f"not a JRA race id: {race_id!r}")
    return race_id


def calendar(year: int, month: int) -> str:
    return f"{_host('race')}/top/calendar.html?year={year:d}&month={month:d}"


def racelist(race_day: date) -> str:
    return f"{_host('db')}/race/list/{race_day:%Y%m%d}/"


def results(race_id: str) -> str:
    return f"{_host('db')}/race/{_checked(race_id)}/"


def pedigree_card(race_id: str) -> str:
    """The five-run card of a race: every runner with its sire, dam and dam sire."""
    return f"{_host('race')}/race/shutuba_past.html?race_id={_checked(race_id)}"
