"""JRA track condition (ticket #90): the rail letter, the turf cushion value and the soil moisture
per (venue, race_date, surface), as of each publication, in the store's `track_condition` columns
(`core.store.schemas`). Pure: no network; fetching goes through `horsetrack.core.ingest`
(`jra.going_fetch`, and the backfill of #228).

Two sources, one row per surface printed and per publication:

- a going page (馬場情報), read by `jra.going` (#136): `parse_baba_page` renames its `rail_setting`
  to `rail` and its `fetched_utc` to `ingested_utc` and adds `jurisdiction`.
- the text a PDF extractor gives for a yearly record (年間の馬場情報), in the layout stated at the
  top of tests/fixtures/synthetic/jra_track_condition_yearly.txt: a title line with the year of
  every row, one 発表 line, then a 【芝】 and a 【ダート】 section, each a column-header line and
  one line per race day. Columns are read by their header words, so a dirt section without the
  course and cushion columns parses the same way. Every row's `published_utc` and `available_utc`
  is the record's own 発表 time: this source does not show a value before the record exists.

A value usable from its publication: `available_utc` = `published_utc` (the owner's rule,
countries/japan/config/availability.yaml). Text is NFKC-normalised first; a value not printed or
printed as a dash is NaN, never 0; dirt has no cushion and no rail letter.

`days_since_rail_move` counts race days since the turf rail last moved; `as_of` is the condition
at a decision time (`jra.going.as_of`'s rule); `export_variant_prior_table` writes the per
(venue, race_date, surface) cushion and moisture table for the speed model's variant prior (#96)
under the data directory.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from horsetrack.core.config import load_source
from horsetrack.core.paths import data_dir
from horsetrack.core.store import schemas
from horsetrack.countries.japan.jra import going
from horsetrack.countries.japan.jra.official_pages import venue_code

TABLE = "track_condition"
JURISDICTION = "jra"
COLUMNS = tuple(schemas.TABLES[TABLE].columns)
KEY = going.KEY
_UTC_COLUMNS = ("published_utc", "available_utc", "ingested_utc")
_VALUE_COLUMNS = ("cushion", "moisture_goal", "moisture_4c")
_FROM_GOING = {"rail_setting": "rail", "fetched_utc": "ingested_utc"}
_NAN = float("nan")

# the yearly record's lines, after NFKC
_TITLE = re.compile(r"([0-9]{4})年\s*馬場情報")
_PUBLISHED = re.compile(
    r"([0-9]{4})年([0-9]{1,2})月([0-9]{1,2})日.*?([0-9]{1,2})時(?:([0-9]{1,2})分)?\s*発表"
)
_SECTION = re.compile(r"【(芝|ダート)】")
_SECTION_SURFACES = {"芝": "turf", "ダート": "dirt"}
_ROW_DATE = re.compile(r"([0-9]{1,2})月([0-9]{1,2})日(?:\(.*\))?")
_MEETING_LABEL = re.compile(r"[0-9]+回(\D+?)[0-9]+日")
# column-header word -> column; the first word a header cell contains wins (開催日 before 開催)
_HEADS = (
    ("開催日", "race_date"),
    ("開催", "meeting"),
    ("馬場状態", "going"),
    ("使用コース", "rail"),
    ("クッション", "cushion"),
    ("ゴール", "moisture_goal"),
    ("コーナー", "moisture_4c"),
)

# the variant-prior table (#96): next to the other feature tables under features/jra
_PRIOR_COLUMNS = (
    "venue", "race_date", "surface", "going", "cushion", "moisture_goal", "moisture_4c",
    "published_utc", "available_utc",
)  # fmt: skip


def _tokyo() -> ZoneInfo:
    return ZoneInfo(load_source("japan", "jra").timezone)


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    for column in _VALUE_COLUMNS:
        frame[column] = frame[column].astype("float64")
    for column in _UTC_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def parse_baba_page(html: str, *, fetched_utc: datetime) -> pd.DataFrame:
    """One row per surface a going page prints, in the table's columns (`jra.going`'s parse).
    Raises ValueError for a naive `fetched_utc` and for a page without a date line, a known venue
    or a 発表 time."""
    rows = going.parse_going_html(html, fetched_utc=fetched_utc).rename(columns=_FROM_GOING)
    rows.insert(0, "jurisdiction", JURISDICTION)
    return rows[list(COLUMNS)]


def _header(cells: list[str]) -> list[str]:
    out = []
    for cell in cells:
        column = next((c for word, c in _HEADS if word in cell), None)
        if column is None:
            raise ValueError(f"yearly record: unknown column header {cell!r}")
        out.append(column)
    return out


def parse_yearly_text(text: str, *, fetched_utc: datetime | None = None) -> pd.DataFrame:
    """One row per (venue, race_date, surface) a yearly record prints, in the table's columns;
    published_utc = available_utc = the record's 発表 time, ingested_utc = `fetched_utc` (null when
    not given). Raises ValueError for a text without the title year or the 発表 line, a row
    before a section's header, a row whose cell count differs from its header's, and an unknown
    venue."""
    ingested = None if fetched_utc is None else going._require_aware(fetched_utc, "fetched_utc")
    lines = [
        " ".join(unicodedata.normalize("NFKC", line).split())
        for line in text.splitlines()
        if not line.lstrip().startswith("#")
    ]
    title = next((m for line in lines if (m := _TITLE.search(line))), None)
    stamp = next((m for line in lines if (m := _PUBLISHED.search(line))), None)
    if title is None:
        raise ValueError("yearly record has no title line naming its year")
    if stamp is None:
        raise ValueError("yearly record has no publication time (発表)")
    year = int(title.group(1))
    y, mo, d, h, mi = (int(g or 0) for g in stamp.groups())
    published = datetime(y, mo, d, h, mi, tzinfo=_tokyo()).astimezone(timezone.utc)

    records: list[dict] = []
    surface: str | None = None
    header: list[str] | None = None
    for line in lines:
        cells = line.split()
        if not cells:
            continue
        if m := _SECTION.fullmatch(line):
            surface, header = _SECTION_SURFACES[m.group(1)], None
            continue
        if surface is not None and cells[0].startswith("開催日"):
            header = _header(cells)
            continue
        day = _ROW_DATE.fullmatch(cells[0])
        if day is None:
            continue  # the title, the 発表 line, or text around the table
        if header is None:
            raise ValueError(f"yearly record: a row before its section's header: {line!r}")
        if len(cells) != len(header):
            raise ValueError(f"yearly record: {len(cells)} cells, {len(header)} headers: {line!r}")
        row = dict(zip(header, cells, strict=True))
        label = _MEETING_LABEL.search(row.get("meeting", ""))
        venue = venue_code(label.group(1)) if label else None
        if venue is None:
            raise ValueError(f"yearly record: no known venue in {line!r}")
        rail = going._RAIL.fullmatch(row.get("rail", "")) if surface == "turf" else None
        records.append(
            {
                "jurisdiction": JURISDICTION,
                "venue": venue,
                "race_date": date(year, int(day.group(1)), int(day.group(2))),
                "surface": surface,
                "going": going._GOING.get(row.get("going", "")),
                "rail": rail.group(1) if rail else None,
                "cushion": going._value(row.get("cushion", "")) if surface == "turf" else _NAN,
                "moisture_goal": going._value(row.get("moisture_goal", "")),
                "moisture_4c": going._value(row.get("moisture_4c", "")),
                "published_utc": published,
                "available_utc": published,
                "ingested_utc": ingested,
            }
        )
    return _typed(pd.DataFrame.from_records(records, columns=list(COLUMNS)))


def _race_days(frame: pd.DataFrame) -> list[date]:
    return [pd.Timestamp(v).date() for v in frame["race_date"]]


def days_since_rail_move(frame: pd.DataFrame) -> pd.Series:
    """Race days since the turf rail last moved, per venue, on the frame's index (Int64).

    A race day is a distinct race date the venue has turf rows on. Its letter is the latest
    publication's non-null `rail` (by available_utc, then ingested_utc), as in
    `going.attach_rail_setting`. 0 on the first day of a new letter, +1 per race day after it;
    a day without a letter is null but still a race day. Null before the first move seen, on a
    day without a letter, on every non-turf row, and after a move across a day without a letter
    (the move fell on that day or the next: unknown) until the next move seen."""
    days = _race_days(frame)
    turf = (frame["surface"] == "turf").to_numpy()
    known = turf & frame["rail"].notna().to_numpy()
    published = pd.DataFrame(
        {
            "venue": frame["venue"].to_numpy()[known],
            "race_date": [d for d, k in zip(days, known, strict=True) if k],
            "rail": frame["rail"].to_numpy()[known],
            "_available": pd.to_datetime(frame["available_utc"], utc=True).to_numpy()[known],
            "_ingested": pd.to_datetime(frame["ingested_utc"], utc=True).to_numpy()[known],
        }
    ).sort_values(["_available", "_ingested", "rail"], kind="mergesort")
    # later publications overwrite earlier ones
    keys = zip(published["venue"], published["race_date"], strict=True)
    letter = dict(zip(keys, published["rail"], strict=True))

    race_days: dict[str, set[date]] = {}
    for venue, day, is_turf in zip(frame["venue"], days, turf, strict=True):
        if is_turf:
            race_days.setdefault(venue, set()).add(day)
    value: dict[tuple[str, date], int | None] = {}
    for venue, venue_days in race_days.items():
        current, count, gap = None, None, False
        for day in sorted(venue_days):
            rail = letter.get((venue, day))
            if rail is None:  # unknown rail: null itself, still a race day
                gap = True
                count = None if count is None else count + 1
                value[(venue, day)] = None
                continue
            if current is None:
                count = None  # the first letter seen is not a move
            elif rail != current:
                count = None if gap else 0
            elif count is not None:
                count += 1
            current, gap = rail, False
            value[(venue, day)] = count

    out = [
        value[(venue, day)] if is_turf else None
        for venue, day, is_turf in zip(frame["venue"], days, turf, strict=True)
    ]
    return pd.Series(out, index=frame.index, dtype="Int64", name="days_since_rail_move")


def as_of(frame: pd.DataFrame, decision_utc: datetime) -> pd.DataFrame:
    """The condition at t: per (venue, race_date, surface), the row with the latest available_utc
    at or before `decision_utc` (ties: the latest ingested_utc); keys with none are absent.
    `jra.going.as_of` on the table's columns; a naive `decision_utc` is a ValueError."""
    renamed = frame.rename(columns={"ingested_utc": "fetched_utc"})
    return going.as_of(renamed, decision_utc).rename(columns={"fetched_utc": "ingested_utc"})


def variant_prior_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (venue, race_date, surface), the cushion and moisture published by 00:00 JST of the
    race date (the latest publication by then, usually the day-before page), with its
    published_utc and available_utc; a key with nothing published by then is absent. The variant
    prior (#96) takes no decision time, so the cut is one that holds for every race of the day.
    Columns: venue, race_date, surface, going, cushion, moisture_goal, moisture_4c, published_utc,
    available_utc; sorted by race_date, venue, surface."""
    zone = _tokyo()
    days = _race_days(frame)
    cut = pd.to_datetime(
        [datetime(d.year, d.month, d.day, tzinfo=zone).astimezone(timezone.utc) for d in days],
        utc=True,
    )
    available = pd.to_datetime(frame["available_utc"], utc=True).to_numpy()
    usable = available <= cut.to_numpy()  # a null available_utc is never usable
    latest = (
        frame.assign(
            race_date=days,
            _available=available,
            _ingested=pd.to_datetime(frame["ingested_utc"], utc=True).to_numpy(),
        )
        .loc[usable]
        .sort_values([*KEY, "_available", "_ingested"], kind="mergesort")
        .drop_duplicates(list(KEY), keep="last")
    )
    return (
        latest.loc[:, list(_PRIOR_COLUMNS)]
        .sort_values(["race_date", "venue", "surface"], kind="mergesort")
        .reset_index(drop=True)
    )


def variant_prior_path() -> Path:
    """$HORSETRACK_DATA_DIR/features/jra/track_condition/variant_prior.parquet."""
    return data_dir() / "features" / "jra" / TABLE / "variant_prior.parquet"


def export_variant_prior_table(frame: pd.DataFrame | None = None) -> Path:
    """Write `variant_prior_table` of `frame` (default: the store's track_condition table, read
    column-scoped) to `variant_prior_path()`, replacing any earlier file; returns the path. The
    table holds no outcome or odds column, so holdout days stay in it: a fit that joins outcomes
    drops them there."""
    if frame is None:
        from horsetrack.core.store import tables

        needed = [c for c in COLUMNS if c in {*_PRIOR_COLUMNS, "ingested_utc"}]
        frame = tables.read_table(JURISDICTION, TABLE, columns=needed)
    table = variant_prior_table(frame)
    path = variant_prior_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    table.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path
