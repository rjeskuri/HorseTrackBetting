"""HKJC URL patterns and cache keys (ported from HKTrack's `sources/hkjc/urls.py`). Pure functions
that return the FINAL url.

The legacy `.aspx` addresses 302 to `/en-us/local/information/<page>`, and old meetings 302 again
to `/en-us/local/information/archive/<page>`. Going straight to the final route halves the request
count. Each endpoint has its own date format: results YYYY/MM/DD, sectionals DD/MM/YYYY, corunning
YYYYMMDD.
"""

from __future__ import annotations

import re
from datetime import date
from urllib.parse import quote

from horsetrack.core.config import load_source
from horsetrack.countries.hongkong.hkjc.parse_common import COUNTRY, SOURCE, season_start_month

# The live route serves the CURRENT SEASON only; every earlier meeting lives under /archive/
# (HKTrack, verified 2026-09-21: 2026/09/13 and 2026/09/16 are live, 2026/05/06 and 2025/07/05
# both 302 to /archive/, and /archive/ for a live date 302s back). The fetcher follows redirects,
# so a wrong guess costs one extra hop, never a missing page.

_RACE_KEY = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})_([A-Za-z0-9]+)_R([0-9]{2})")


def live_route_from(today: date | None = None) -> date:
    """First day of the current season's opening month (config `season.start_month`)."""
    today = today or date.today()
    start_month = season_start_month()
    year = today.year if today.month >= start_month else today.year - 1
    return date(year, start_month, 1)


def _base() -> str:
    host = load_source(COUNTRY, SOURCE).hosts.html
    if not host:
        raise ValueError(f"{COUNTRY}/{SOURCE}.yaml has no hosts.html")
    return host.rstrip("/") + "/en-us/local/information"


def is_archived(race_date: date, today: date | None = None) -> bool:
    return race_date < live_route_from(today)


def fixture(calyear: int, calmonth: int) -> str:
    return f"{_base()}/fixture?calyear={calyear:04d}&calmonth={calmonth:02d}"


def results(race_date: date, venue: str, race_no: int) -> str:
    d = race_date.strftime("%Y/%m/%d")
    if is_archived(race_date):
        return f"{_base()}/archive/localresults?racedate={d}&racecourse={venue}&RaceNo={race_no}"
    return f"{_base()}/localresults?racedate={d}&Racecourse={venue}&RaceNo={race_no}"


def sectional(race_date: date, race_no: int) -> str:
    route = "archive/displaysectionaltime" if is_archived(race_date) else "displaysectionaltime"
    return f"{_base()}/{route}?racedate={race_date.strftime('%d/%m/%Y')}&RaceNo={race_no}"


def corunning(race_date: date, race_no: int) -> str:
    d = race_date.strftime("%Y%m%d")
    if is_archived(race_date):
        return f"{_base()}/archive/corunning?date={d}&raceno={race_no}"
    return f"{_base()}/corunning?Date={d}&RaceNo={race_no}"


def horse(horse_id: str) -> str:
    """Option=1 = form records for all seasons. Retired horses 302 to /otherhorse (unknowable in
    advance)."""
    return f"{_base()}/horse?horseid={quote(horse_id, safe='')}&Option=1"


def racecard(race_date: date, venue: str, race_no: int) -> str:
    d = race_date.strftime("%Y/%m/%d")
    return f"{_base()}/racecard?racedate={d}&Racecourse={venue}&RaceNo={race_no}"


def entries(race_date: date, venue: str) -> str:
    """View=All returns every provisional race section in one request."""
    d = race_date.strftime("%Y/%m/%d")
    return f"{_base()}/entries?racedate={d}&Racecourse={venue}&View=All"


def meeting_key(race_date: date, venue: str) -> str:
    return f"{race_date.isoformat()}_{venue}"


def race_key(race_date: date, venue: str, race_no: int) -> str:
    """Cache key shared by the per-race page types (results, sectional, corunning, racecard)."""
    return f"{meeting_key(race_date, venue)}_R{race_no:02d}"


def parse_race_key(key: str | None) -> tuple[date, str, int] | None:
    m = _RACE_KEY.fullmatch(key or "")
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3))), m.group(4), int(m.group(5))
    except ValueError:
        return None
