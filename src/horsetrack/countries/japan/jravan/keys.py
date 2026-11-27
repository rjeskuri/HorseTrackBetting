"""JV-Data race keys and timestamps -> the store's ids and UTC instants.

A JV-Data race key is 開催年 (YYYY), 開催月日 (MMDD), 競馬場コード (2 digits), 開催回 and
開催日目 (2 digits each) and レース番号 (2 digits), as decoded text. The race id is the jra
store's (`jra.netkeiba.common.store_race_id`): `jra:<YYYY-MM-DD>:<VENUE>:<NN>`, the venue's short
code looked up by number in `config/jra.yaml`. Times in JV-Data are JST wall clock (UTC+9, no DST);
the store keeps UTC instants.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import cache

from horsetrack.core.config import load_source
from horsetrack.countries.japan.jra.netkeiba.common import COUNTRY, SOURCE, store_race_id

JST = timezone(timedelta(hours=9), "JST")


@cache
def _venues() -> dict[str, str]:
    """JRA venue number ('08') -> short code ('KYO'), from config/jra.yaml."""
    venues = load_source(COUNTRY, SOURCE).venues
    return {v.number: code for code, v in venues.items() if v.number}


def _digits(value: object, width: int) -> str | None:
    """`value` when it is exactly `width` ASCII digits, else None."""
    if isinstance(value, str) and len(value) == width and value.isascii() and value.isdigit():
        return value
    return None


def _race_day(year: str, mmdd: str) -> date | None:
    y, md = _digits(year, 4), _digits(mmdd, 4)
    if y is None or md is None:
        return None
    try:
        return date(int(y), int(md[:2]), int(md[2:]))
    except ValueError:
        return None


def _race_no(race_no: str) -> int | None:
    digits = _digits(race_no, 2)
    return int(digits) if digits is not None and int(digits) > 0 else None


def meeting_id(year: str, mmdd: str, venue_code: str) -> str | None:
    """`jra:<YYYY-MM-DD>:<VENUE>`; None for a venue outside JRA or an invalid date."""
    day = _race_day(year, mmdd)
    venue = _venues().get(venue_code) if isinstance(venue_code, str) else None
    if day is None or venue is None:
        return None
    return f"{SOURCE}:{day.isoformat()}:{venue}"


def race_id(year: str, mmdd: str, venue_code: str, race_no: str) -> str | None:
    """`jra:<YYYY-MM-DD>:<VENUE>:<NN>`, the jra store's race id; None for a venue outside JRA, an
    invalid date or race number."""
    day = _race_day(year, mmdd)
    venue = _venues().get(venue_code) if isinstance(venue_code, str) else None
    number = _race_no(race_no)
    if day is None or venue is None or number is None:
        return None
    return store_race_id(day, venue, number)


def source_race_key(year: str, venue_code: str, kai: str, nichi: str, race_no: str) -> str | None:
    """The 12-digit netkeiba race id the jra store keeps in `races.source_race_key`
    (YYYY 場 回 日 R); None for a venue outside JRA or a malformed part."""
    parts = (_digits(year, 4), _digits(kai, 2), _digits(nichi, 2), _digits(race_no, 2))
    if any(p is None for p in parts) or venue_code not in _venues():
        return None
    y, k, n, r = parts
    return f"{y}{venue_code}{k}{n}{r}"


def jurisdiction(venue_code: str) -> str | None:
    """'jra' for 01-10, 'nar' for any other two-digit number, 'overseas' for any other non-blank
    code; None for '00' or blank."""
    if not isinstance(venue_code, str) or not venue_code.strip() or venue_code == "00":
        return None
    if _digits(venue_code, 2) is None:
        return "overseas"
    return "jra" if 1 <= int(venue_code) <= 10 else "nar"


def announced_utc(year: str, race_mmdd: str, mmddhhmm: str) -> datetime | None:
    """The JST minute `mmddhhmm` (発表月日時分) of a race on `year`/`race_mmdd` as an aware UTC
    datetime; a month after the race's month is in the year before. None for '00000000', blank or
    an invalid time."""
    y, race, stamp = _digits(year, 4), _digits(race_mmdd, 4), _digits(mmddhhmm, 8)
    if y is None or race is None or stamp is None:
        return None
    month = int(stamp[:2])
    when_year = int(y) - 1 if month > int(race[:2]) else int(y)
    try:
        local = datetime(
            when_year, month, int(stamp[2:4]), int(stamp[4:6]), int(stamp[6:]), tzinfo=JST
        )
    except ValueError:
        return None
    return local.astimezone(timezone.utc)


def post_time_local(hhmm: str) -> str | None:
    """発走時刻 'HHMM' as 'HH:MM' (JST); None for '0000', blank or an invalid time."""
    digits = _digits(hhmm, 4)
    if digits is None or digits == "0000":
        return None
    hours, minutes = int(digits[:2]), int(digits[2:])
    if hours > 23 or minutes > 59:
        return None
    return f"{digits[:2]}:{digits[2:]}"
