"""JMA AMeDAS hourly observations: the download CSV, the venue's station and publication times
(ticket #89; the rules are in docs/specs/weather.md). Pure: no network, no store.

`parse_amedas_csv` reads one station's hourly 過去の気象データ・ダウンロード CSV (decoded from
Shift_JIS by the caller). Columns are found by the header rows (item, 風向 sub-row, 品質情報 /
現象なし情報 / 均質番号), never by position. Missing is NaN, never 0: a value that is not a number,
or one whose 品質情報 is not 8 (正常値) or 5 (準正常値). A blank amount (precipitation, sunshine)
whose 現象なし情報 is 1, or an amount printed `--` (現象なし in symbol mode), is a genuine 0.
"""

from __future__ import annotations

import csv
import math
import re
from datetime import datetime, timedelta, timezone
from functools import cache
from pathlib import Path

import pandas as pd
import yaml

from horsetrack.core.config import load_country_yaml

COUNTRY = "japan"
CONFIG = "weather"  # countries/japan/config/weather.yaml: the publication lag and the http block
LAG_KEY = "publication_lag_min"
STATIONS = Path(__file__).with_name("stations.yaml")
COLUMNS = ("station_id", "obs_utc", "temp_c", "precip_mm", "wind_ms", "wind_dir_deg", "sunshine_h")
JST = timezone(timedelta(hours=9))  # Japan keeps no daylight saving time

# column -> the item name in the CSV header, without its unit: 気温(℃) -> 気温
ITEMS = {"temp_c": "気温", "precip_mm": "降水量", "wind_ms": "風速", "sunshine_h": "日照時間"}
AMOUNTS = ("precip_mm", "sunshine_h")  # sums over the hour: may be 現象なし
DIRECTION = "風向"  # the sub-row naming wind direction under 風速
QUALITY, NO_PHENOMENON, HOMOGENEITY = "品質情報", "現象なし情報", "均質番号"
LABELS = frozenset({QUALITY, NO_PHENOMENON, HOMOGENEITY})
# 8 正常値 and 5 準正常値 are kept; 4 資料不足値, 2 疑問値, 1 欠測, 0 and anything else are missing
KEPT_QUALITY = frozenset({"8", "5"})
NONE_SYMBOL = "--"  # an amount printed in symbol mode for 現象なし
COMPASS = {
    name: 22.5 * i
    for i, name in enumerate(
        ("北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
         "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西")
    )
}  # fmt: skip
CLOCK = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2}) (\d{1,2}):(\d{2})(?::(\d{2}))?")
_UNIT = re.compile(r"[(（].*$")


def _cell(row: list[str], j: int | None) -> str:
    return row[j].strip() if j is not None and j < len(row) else ""


def _obs_utc(clock: str) -> datetime | None:
    """The UTC instant of a JST clock `Y/M/D H:MM[:SS]`; 24:00 is 0:00 of the next day."""
    m = CLOCK.fullmatch(clock.strip())
    if m is None:
        return None
    y, mo, d, h, mi, s = (int(g or 0) for g in m.groups())
    if h > 24 or mi > 59 or s > 59 or (h == 24 and (mi or s)):
        raise ValueError(f"not a JMA hourly clock: {clock!r}")
    day = datetime(y, mo, d, tzinfo=JST)
    return (day + timedelta(hours=h, minutes=mi, seconds=s)).astimezone(timezone.utc)


def _number(text: str) -> float:
    try:
        value = float(text)
    except ValueError:  # blank, ///, ×, a symbol-marked value
        return math.nan
    return value if math.isfinite(value) else math.nan


def _header_columns(header: list[list[str]]) -> dict[tuple[str, str, str], int]:
    """(item, part, label) -> column index: item without its unit, part the non-label sub-rows
    (風向 or ''), label 品質情報 / 現象なし情報 / 均質番号 or '' for the value itself."""
    columns: dict[tuple[str, str, str], int] = {}
    for j in range(1, max(len(row) for row in header)):
        item = _UNIT.sub("", _cell(header[0], j))
        if not item:
            continue
        subs = [_cell(row, j) for row in header[1:]]
        label = next((s for s in subs if s in LABELS), "")
        part = "/".join(s for s in subs if s and s not in LABELS)
        key = (item, part, label)
        if key in columns:
            raise ValueError(f"two columns for {key}: one station per download")
        columns[key] = j
    return columns


def parse_amedas_csv(text: str, *, station_id: str) -> pd.DataFrame:
    """One row per hourly line of a one-station JMA download: `COLUMNS`, obs_utc tz-aware UTC,
    the values float64 (NaN where missing). ValueError for anything that is not such a file."""
    rows = list(csv.reader(text.splitlines()))
    head = next((i for i, r in enumerate(rows) if r and r[0].strip() == "年月日時"), None)
    if head is None:
        raise ValueError("not a JMA download CSV: no 年月日時 header row")
    if head and len({c.strip() for c in rows[head - 1][1:]} - {""}) > 1:
        raise ValueError("a download of several stations: parse one station at a time")
    first = next((i for i in range(head + 1, len(rows)) if _obs_utc(_cell(rows[i], 0))), len(rows))
    columns = _header_columns(rows[head:first])

    # column -> (value, 品質情報, 現象なし情報) indices; an element the download does not hold (or
    # the station does not observe) has none and stays NaN
    spots: dict[str, tuple[int, int, int | None]] = {}
    elements = {
        **{c: (i, "") for c, i in ITEMS.items()},
        "wind_dir_deg": (ITEMS["wind_ms"], DIRECTION),
    }
    for column, (item, part) in elements.items():
        j = columns.get((item, part, ""))
        if j is None:
            continue
        quality = columns.get((item, part, QUALITY))
        if quality is None:
            raise ValueError(f"{item} {part} has no {QUALITY} column: download it with them")
        spots[column] = (j, quality, columns.get((item, part, NO_PHENOMENON)))

    def read(row: list[str], column: str) -> float:
        if column not in spots:
            return math.nan
        j, quality, none = spots[column]
        if _cell(row, quality) not in KEPT_QUALITY:
            return math.nan
        text = _cell(row, j)
        if column == "wind_dir_deg":
            return COMPASS.get(text, math.nan)  # 静穏 (calm) has no direction
        number = _number(text)
        if column in AMOUNTS and math.isnan(number):
            if text == NONE_SYMBOL or (not text and _cell(row, none) == "1"):
                return 0.0  # 現象なし: no rain, no sun
        return number

    records = []
    for row in rows[first:]:
        if not any(c.strip() for c in row):
            continue
        obs = _obs_utc(_cell(row, 0))
        if obs is None:
            raise ValueError(f"not an hourly line of a JMA download: {row[:1]!r}")
        records.append([obs, *(read(row, c) for c in COLUMNS[2:])])
    frame = pd.DataFrame(records, columns=list(COLUMNS[1:]))
    frame["obs_utc"] = pd.to_datetime(frame["obs_utc"], utc=True).dt.as_unit("us")
    frame[list(COLUMNS[2:])] = frame[list(COLUMNS[2:])].astype("float64")
    frame.insert(0, "station_id", station_id)
    return frame


@cache
def _stations() -> dict[str, dict]:
    venues = (yaml.safe_load(STATIONS.read_text(encoding="utf-8")) or {}).get("venues") or {}
    ids = [str(entry["station_id"]) for entry in venues.values()]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{STATIONS.name}: two venues share a station")
    return venues


def station(venue: str) -> dict:
    """A JRA venue's stations.yaml entry (station_id, name, kind, prec_no, ...); KeyError for any
    other code."""
    entry = _stations().get(venue)
    if entry is None:
        raise KeyError(f"no JMA station for venue {venue!r} (JRA venues: {', '.join(_stations())})")
    return dict(entry)


def station_for_venue(venue: str) -> str:
    """The AMeDAS station id (JMA's block_no) of a JRA venue code; KeyError for any other code."""
    return str(station(venue)["station_id"])


def publication_lag() -> timedelta:
    """The configured lag from an hourly value's obs_utc to its publication (an owner decision,
    #89). RuntimeError while weather.yaml leaves it unset; ValueError for a bad value."""
    value = load_country_yaml(COUNTRY, CONFIG).get(LAG_KEY)
    if value is None:
        raise RuntimeError(
            f"{LAG_KEY} is not set in countries/{COUNTRY}/config/{CONFIG}.yaml: the JMA"
            " publication lag is an owner decision (ticket #89), so no weather row can be stamped"
            " with an available_utc until the owner sets it"
        )
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{LAG_KEY} must be a number of minutes, got {value!r}")
    if value < 0:
        raise ValueError(f"{LAG_KEY} must be at least 0, got {value!r}")
    return timedelta(minutes=value)


def with_publication_time(frame: pd.DataFrame, lag: timedelta) -> pd.DataFrame:
    """A copy of `frame` plus `available_utc` = obs_utc + lag (tz-aware UTC). A negative lag would
    publish a value before it was observed: ValueError."""
    if not isinstance(lag, timedelta):
        raise TypeError(f"lag must be a timedelta, got {type(lag).__name__}")
    if lag < timedelta(0):
        raise ValueError(f"a publication lag cannot be negative, got {lag}")
    if "available_utc" in frame.columns:
        raise ValueError("the frame already carries available_utc")
    if not isinstance(frame["obs_utc"].dtype, pd.DatetimeTZDtype):
        raise ValueError("obs_utc must be timezone-aware")
    out = frame.copy()
    out["available_utc"] = out["obs_utc"].dt.tz_convert("UTC") + pd.Timedelta(lag)
    return out
