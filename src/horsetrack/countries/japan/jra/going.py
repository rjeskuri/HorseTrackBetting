"""JRA going measurements (馬場情報): the going label, the turf cushion value (クッション値) and the
soil moisture (含水率) at the goal (ゴール前) and the fourth corner (4コーナー), per surface, as of
the page's publication (ticket #136). Pure: no network; fetching goes through
`horsetrack.core.ingest`.

A going page is one meeting's, Shift_JIS (cp932) and untrusted data, only parsed. Its date line
('2026年11月28日（土曜） 5回東京7日') gives the race date and the venue (code from
`countries/japan/config/jra.yaml`); its 発表 line ('11月28日（土曜）9時30分 発表', Tokyo, year of
the race date) gives `published_utc`. JRA republishes the page during a meeting, so one meeting has
several rows per surface over time: each parsed page is kept as its own rows and `as_of` picks
what was published by a decision time. A value is usable from its publication:
`available_utc` = `published_utc`. Text is NFKC-normalised first (full-width digits, ％, －);
a value not printed or printed as － is NaN, never 0. Dirt has no cushion value.

The columns are those of ticket #90's `track_condition` table less `ingested_utc`; `rail_setting`
is its `rail` (ticket #222): the turf course in use (仮柵 position) printed in `.rail .value`
('Aコース'), 'A' to 'D' on the turf row after NFKC, None on dirt and for a page that prints none,
－, or anything else. `attach_rail_setting` puts it on the store's races.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup, Tag

from horsetrack.core.config import load_source
from horsetrack.countries.japan.jra.official_pages import soup, venue_code

COLUMNS = (
    "venue",
    "race_date",
    "surface",
    "going",
    "cushion",
    "moisture_goal",
    "moisture_4c",
    "rail_setting",
    "published_utc",
    "available_utc",
    "fetched_utc",
)
KEY = ("venue", "race_date", "surface")
_UTC_COLUMNS = ("published_utc", "available_utc", "fetched_utc")
_VALUE_COLUMNS = ("cushion", "moisture_goal", "moisture_4c")

# Every CSS selector the parser reads: the first of each pair is the structure of the synthetic
# fixtures, the second the real page https://www.jra.go.jp/keiba/baba/ (checked 2026-11-25, #253).
# The real page has no `.rail .value`: the course in use is `.course .content p`, 'Cコース（Aコース
# から6メートル外に内柵を設置）' with the letter in a <strong>. Its cushion value and measurement
# times are filled in by the page's script, so only the synthetic `.cushion .value` is read.
_DATE_LINE = ".date_line, #condition .contents_header h2"
_PUBLISH = ".publish, #course_condition .block_header .time"
_GOING_ROW = "table.going tr, #course_condition .data_list_unit"
_CUSHION_VALUE = ".cushion .value"
_MOISTURE_ROW = "table.moisture tr, #moist_data table.basic tr"
_RAIL_VALUE = ".rail .value, .course .content p"

_SURFACES = {"芝": "turf", "ダート": "dirt"}
_GOING = {"良": "GOOD", "稍重": "GOOD TO YIELDING", "重": "SOFT", "不良": "HEAVY"}
# moisture column header word -> column (read by header, not position)
_MOISTURE_HEADS = (("ゴール", "moisture_goal"), ("コーナー", "moisture_4c"))

_MEETING = re.compile(r"([0-9]{4})年([0-9]{1,2})月([0-9]{1,2})日[^0-9]*?[0-9]+回(\D+?)[0-9]+日")
# the real page's order: '第5回東京競馬第6日 （2026年11月23日（月曜））'
_MEETING_FIRST = re.compile(
    r"[0-9]+回(\D+?)(?:競馬)?第?[0-9]+日\D*([0-9]{4})年([0-9]{1,2})月([0-9]{1,2})日"
)
# '11月27日（金曜）15時00分 発表' (synthetic) or '（11月21日（土曜）10時現在）' (real: no minutes)
_PUBLISHED = re.compile(
    r"([0-9]{1,2})月([0-9]{1,2})日.*?([0-9]{1,2})時(?:([0-9]{1,2})分)?\s*(?:発表|現在)"
)
_RAIL = re.compile(r"([A-D])\s*(?:コース.*)?")
_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def _text(node: Tag | None) -> str:
    """The node's text, NFKC-normalised (full-width digits and signs to ASCII), spaces collapsed."""
    if node is None:
        return ""
    return " ".join(unicodedata.normalize("NFKC", node.get_text(" ")).split())


def _value(text: str) -> float:
    """'12.3%' -> 12.3; '-', '' or anything not a number -> NaN."""
    m = _NUMBER.fullmatch(text.rstrip("%").strip())
    return float(m.group(0)) if m else float("nan")


def _require_aware(when: datetime, name: str) -> pd.Timestamp:
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware (UTC), got naive {when!r}")
    return pd.Timestamp(when).tz_convert("UTC")


def _meeting(doc: BeautifulSoup) -> tuple[date, str]:
    text = _text(doc.select_one(_DATE_LINE))
    if m := _MEETING.search(text):
        year, month, day, name = m.groups()
    elif m := _MEETING_FIRST.search(text):
        name, year, month, day = m.groups()
    else:
        raise ValueError("going page has no date line (an error page?)")
    race_date = date(int(year), int(month), int(day))
    venue = venue_code(name)
    if venue is None:
        raise ValueError(f"going page venue {name!r} is not in jra.yaml")
    return race_date, venue


def meeting(body: bytes) -> tuple[date, str]:
    """(race date, venue code) the page's date line names; ValueError for a page without one or
    with a venue not in jra.yaml (an error page)."""
    return _meeting(soup(body))


def _published(doc: BeautifulSoup, race_date: date) -> datetime:
    """The 発表 clock in UTC, dated in the race date's year, or the year before when that would
    fall after the race date (a page published in late December for a January meeting)."""
    m = _PUBLISHED.search(_text(doc.select_one(_PUBLISH)))
    if not m:
        raise ValueError("going page has no publication time (発表)")
    month, day, hour, minute = (int(g or 0) for g in m.groups())
    year = (
        race_date.year if (month, day) <= (race_date.month, race_date.day) else race_date.year - 1
    )
    zone = ZoneInfo(load_source("japan", "jra").timezone)
    local = datetime(year, month, day, hour, minute, tzinfo=zone)
    return local.astimezone(timezone.utc)


def _surface(row: Tag) -> str | None:
    return _SURFACES.get(_text(row.find(["th", "h4"])))  # h4: the real page's going unit


def _goings(doc: BeautifulSoup) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for row in doc.select(_GOING_ROW):
        surface = _surface(row)
        if surface is not None:
            out[surface] = _GOING.get(_text(row.find(["td", "p"])))
    return out


def _moistures(doc: BeautifulSoup) -> dict[str, dict[str, float]]:
    """surface -> {moisture column: value}; columns found by the header row's words."""
    columns: dict[int, str] = {}
    out: dict[str, dict[str, float]] = {}
    for row in doc.select(_MOISTURE_ROW):
        cells = [_text(c) for c in row.find_all(["th", "td"])]
        surface = _surface(row)
        if surface is None:
            for i, cell in enumerate(cells):
                for word, column in _MOISTURE_HEADS:
                    if word in cell:
                        columns[i] = column
            continue
        out[surface] = {
            column: _value(cells[i]) if i < len(cells) else float("nan")
            for i, column in columns.items()
        }
    return out


def _rail_setting(doc: BeautifulSoup) -> str | None:
    """'Aコース' -> 'A' (a bare 'B' too); －, empty, a missing block or another letter -> None."""
    m = _RAIL.fullmatch(_text(doc.select_one(_RAIL_VALUE)))
    return m.group(1) if m else None


def parse_going_page(body: bytes, *, fetched_utc: datetime) -> pd.DataFrame:
    """One row per surface the page prints (columns COLUMNS). Raises ValueError for a naive
    `fetched_utc` and for a page without a date line, a known venue or a 発表 time."""
    fetched = _require_aware(fetched_utc, "fetched_utc")
    doc = soup(body)
    race_date, venue = _meeting(doc)
    published = _published(doc, race_date)
    goings = _goings(doc)
    moistures = _moistures(doc)
    cushion = _value(_text(doc.select_one(_CUSHION_VALUE)))
    rail = _rail_setting(doc)

    surfaces = list(dict.fromkeys([*goings, *moistures]))
    records = [
        {
            "venue": venue,
            "race_date": race_date,
            "surface": surface,
            "going": goings.get(surface),
            "cushion": cushion if surface == "turf" else float("nan"),
            "moisture_goal": moistures.get(surface, {}).get("moisture_goal", float("nan")),
            "moisture_4c": moistures.get(surface, {}).get("moisture_4c", float("nan")),
            "rail_setting": rail if surface == "turf" else None,
            "published_utc": published,
            "available_utc": published,
            "fetched_utc": fetched,
        }
        for surface in surfaces
    ]
    return _typed(pd.DataFrame.from_records(records, columns=list(COLUMNS)))


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    for column in _VALUE_COLUMNS:
        frame[column] = frame[column].astype("float64")
    for column in _UTC_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def attach_rail_setting(
    races: pd.DataFrame, going: pd.DataFrame, *, decision_utc: datetime | None = None
) -> pd.DataFrame:
    """`races` plus a `rail_setting` column (replacing one that is there): a turf race gets the
    latest published non-null rail setting of its (venue, race_date) in `going` (the concatenated
    output of `parse_going_page`), so a later page printing － does not erase it; every other race
    gets None. With `decision_utc` (tz-aware) only pages available by then count."""
    decision = None if decision_utc is None else _require_aware(decision_utc, "decision_utc")
    rows = going[going["rail_setting"].notna()].copy()
    rows["_available"] = pd.to_datetime(rows["available_utc"], utc=True)
    rows["_fetched"] = pd.to_datetime(rows["fetched_utc"], utc=True)
    if decision is not None:
        rows = rows[rows["_available"] <= decision]
    rows = rows.sort_values(["_available", "_fetched"], kind="mergesort")
    # later rows overwrite earlier ones
    latest = dict(
        zip(
            zip(rows["venue"], rows["race_date"], strict=True),
            rows["rail_setting"],
            strict=True,
        )
    )
    out = races.copy()
    out["rail_setting"] = pd.Series(
        [
            latest.get((v, d)) if s == "turf" else None
            for v, d, s in zip(out["venue"], out["race_date"], out["surface"], strict=True)
        ],
        index=out.index,
        dtype="object",
    )
    return out


def as_of(frame: pd.DataFrame, decision_utc: datetime) -> pd.DataFrame:
    """For each (venue, race_date, surface), the row with the latest `available_utc` at or before
    `decision_utc`; keys with none are absent. Independent of row order; a publication fetched
    more than once gives one row (the latest fetch)."""
    decision = _require_aware(decision_utc, "decision_utc")
    available = pd.to_datetime(frame["available_utc"], utc=True)
    usable = frame.loc[available <= decision].copy()
    usable["_available"] = available[available <= decision]
    usable["_fetched"] = pd.to_datetime(usable["fetched_utc"], utc=True)
    latest = (
        usable.sort_values([*KEY, "_available", "_fetched"], kind="mergesort")
        .drop_duplicates(list(KEY), keep="last")
        .drop(columns=["_available", "_fetched"])
    )
    return latest.reset_index(drop=True)
