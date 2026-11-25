"""JMA AMeDAS weather per JRA venue, with publication times (ticket #89).

Contract pinned here (the coder follows it):
- `horsetrack.countries.japan.jma.parse` is pure (no network).
- `parse_amedas_csv(text: str, *, station_id: str) -> pd.DataFrame`: one row per hourly line of a
  JMA 過去の気象データ・ダウンロード CSV (sent as Shift_JIS, handed over decoded), columns (at
  least) PARSED below. `station_id` is the caller's, on every row. `obs_utc` is the line's JST
  clock (`2031/1/1 0:00:00`; `24:00:00` is 0:00 of the next day) in UTC, tz-aware. Columns are
  found by the header rows (the item row, the 風向 sub-row, the 品質情報 / 現象なし情報 / 均質番号
  row), never by position: the element order follows the download, an AMeDAS-only station has no
  現象なし情報 columns, and an element the parser does not read (相対湿度) is ignored. LF or CRLF.
  temp_c (℃), precip_mm (mm), wind_ms (m/s), sunshine_h (hours), wind_dir_deg: float64.
  wind_dir_deg is the 16-point compass name in degrees clockwise from north (北 0 or 360, 北北東
  22.5, ... 北北西 337.5); 静穏 (calm) has no direction, NaN, and keeps its speed.
- Missing is NaN, never 0: a blank value, a value that is not a number (`///`, `×`), or a reading
  whose 品質情報 is 1 (欠測), 0 (not observed) or `///`. A genuine zero stays 0.0, and so does a
  blank amount whose 現象なし情報 is 1 (no phenomenon: no rain, no sun). Quality 8 (normal) and 5
  (quasi-normal) values are kept as printed.
- `station_for_venue(venue: str) -> str`: the one AMeDAS station id of each JRA venue code, distinct
  per venue, kept in `countries/japan/jma/stations.yaml`. Any other code raises KeyError or
  ValueError.
- `with_publication_time(frame, lag: timedelta) -> pd.DataFrame`: a new frame, the input's columns
  unchanged plus `available_utc` = `obs_utc` + lag (tz-aware UTC). A negative lag is a ValueError.
- `publication_lag() -> timedelta`: the configured lag. Its value is an owner decision and is never
  pinned here.
- The store declares a `weather` table (`core.store.schemas`) holding the parsed columns,
  `available_utc` and `ingested_utc`, and Japan's availability config has its rule: a view at t
  holds exactly the rows whose `available_utc` is at or before t, whatever their `obs_utc` or
  `ingested_utc`; a row without `available_utc` is never in a view.

The CSVs are synthetic: tests/fixtures/synthetic/jma_amedas_hourly.csv and `amedas_csv` below, which
builds the same layout from values. Every date is invented and far in the future. Expected values
are written out here, never computed by the code under test.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pytest
import yaml

from horsetrack.core.store import schemas
from horsetrack.core.store.asof import asof_join, load_availability, read_asof
from horsetrack.core.store.tables import ARROW_TYPES

# isort: split
# A block of its own: ruff counts horsetrack.countries.japan.jma as third-party until it exists.
from horsetrack.countries.japan.jma import parse

UTC = timezone.utc
US = timedelta(microseconds=1)
ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic" / "jma_amedas_hourly.csv"
STATIONS = ROOT / "src" / "horsetrack" / "countries" / "japan" / "jma" / "stations.yaml"
PARSED = ("station_id", "obs_utc", "temp_c", "precip_mm", "wind_ms", "wind_dir_deg", "sunshine_h")
VALUES = ("temp_c", "precip_mm", "wind_ms", "wind_dir_deg", "sunshine_h")
VENUES = ("SAP", "HAK", "FKS", "NII", "TOK", "NAK", "CKY", "KYO", "HSN", "KOK")
COMPASS = (
    "北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
    "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西",
)  # fmt: skip
N = None  # an expected missing reading (NaN in the frame)


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def ts(value) -> datetime:
    return pd.Timestamp(value).to_pydatetime()


def fixture_text() -> str:
    """The fixture as the parser gets it: stored UTF-8 here, sent as Shift_JIS by JMA, decoded."""
    return FIXTURE.read_text(encoding="utf-8").encode("cp932").decode("cp932")


def _num(value) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), 3)


def _deg(value) -> float | None:
    n = _num(value)
    return None if n is None else round(n % 360, 3)  # north may be 0 or 360


def rows(frame: pd.DataFrame) -> list[tuple]:
    """(obs_utc, temp_c, precip_mm, wind_ms, wind_dir_deg, sunshine_h) per row, NaN as None, in
    obs_utc order."""
    out = [
        (
            ts(r["obs_utc"]),
            _num(r["temp_c"]),
            _num(r["precip_mm"]),
            _num(r["wind_ms"]),
            _deg(r["wind_dir_deg"]),
            _num(r["sunshine_h"]),
        )
        for r in frame.to_dict("records")
    ]
    return sorted(out, key=lambda r: r[0])


def assert_types(frame: pd.DataFrame) -> None:
    assert isinstance(frame["obs_utc"].dtype, pd.DatetimeTZDtype)
    assert str(frame["obs_utc"].dt.tz) == "UTC"
    for column in VALUES:
        assert frame[column].dtype == "float64", column


# ---------------------------------------------------------------- a JMA CSV from values

ITEM = {
    "temp": "気温(℃)",
    "precip": "降水量(mm)",
    "sun": "日照時間(時間)",
    "wind": "風速(m/s)",
    "humidity": "相対湿度(％)",
}
OFFICIAL_ORDER = ("temp", "precip", "sun", "wind")  # the fixture's order


def amedas_csv(
    lines: list[tuple[str, dict]],
    *,
    order: tuple[str, ...] = OFFICIAL_ORDER,
    official: bool = True,
    station: str = "架空",
    newline: str = "\r\n",
) -> str:
    """A JMA hourly download laid out like the fixture. `lines` = [(JST clock, cells)], cells by
    element: temp / humidity (value, quality); precip / sun (value, 現象なし情報, quality) at an
    official station (`official`), (value, quality) at an AMeDAS-only one, which has no 現象なし情報
    column; wind (speed, quality, direction, direction quality). Every element ends with its
    均質番号, 1."""
    items: list[str] = []
    sub: list[str] = []
    labels: list[str] = []
    for element in order:
        if element == "wind":
            cols = [
                ("", ""),
                ("", "品質情報"),
                ("風向", ""),
                ("風向", "品質情報"),
                ("", "均質番号"),
            ]
        elif element in ("precip", "sun") and official:
            cols = [("", ""), ("", "現象なし情報"), ("", "品質情報"), ("", "均質番号")]
        else:
            cols = [("", ""), ("", "品質情報"), ("", "均質番号")]
        items += [ITEM[element]] * len(cols)
        sub += [s for s, _ in cols]
        labels += [label for _, label in cols]
    head = [
        "ダウンロードした時刻：2031/01/09 10:11:12",
        "",
        ",".join(["", *[station] * len(items)]),
        ",".join(["年月日時", *items]),
    ]
    if "wind" in order:
        head.append(",".join(["", *sub]))
    head.append(",".join(["", *labels]))
    body = []
    for clock, cells in lines:
        fields = [clock]
        for element in order:
            cell = cells[element]
            if element == "wind":
                fields += [*cell, "1"]
            elif element in ("precip", "sun") and official:
                value, no_phenomenon, quality = cell
                fields += [value, no_phenomenon, quality, "1"]
            else:
                value, quality = cell
                fields += [value, quality, "1"]
        body.append(",".join(fields))
    return newline.join([*head, *body]) + newline


def reading(
    temp: str, precip: str, sun: str, speed: str, direction: str, *, official=True, humidity="70"
) -> dict:
    """One line of quality-8 readings; at an official station a zero amount is 現象なし (1)."""

    def amount(value: str) -> tuple[str, ...]:
        if not official:
            return (value, "8")
        return (value, "1" if float(value) == 0 else "0", "8")

    return {
        "temp": (temp, "8"),
        "precip": amount(precip),
        "sun": amount(sun),
        "wind": (speed, "8", direction, "8"),
        "humidity": (humidity, "8"),
    }


# ---------------------------------------------------------------- the parser

# the fixture, line by line: 2030/12/31 23:00 to 2031/1/1 9:00 JST (14:00 to 00:00 UTC)
EXPECTED_FIXTURE = [
    (utc(2030, 12, 31, 14), -1.4, 0.0, 2.3, 337.5, 0.0),
    (utc(2030, 12, 31, 15), -2.0, 0.0, 1.8, 315.0, 0.0),  # 2031/1/1 0:00 JST, the year before
    (utc(2030, 12, 31, 16), N, 0.0, 1.1, 292.5, 0.0),  # temperature 欠測
    (utc(2030, 12, 31, 17), -3.1, 0.5, 0.2, N, 0.0),  # 静穏
    (utc(2030, 12, 31, 18), -3.6, N, 0.0, N, 0.0),  # precipitation 欠測; calm at 0.0 m/s
    (utc(2030, 12, 31, 19), -4.2, 1.0, N, N, 0.0),  # quality 5 kept; wind 欠測
    (utc(2030, 12, 31, 20), -4.0, 0.0, 3.4, 180.0, 0.0),
    (utc(2030, 12, 31, 21), -3.8, 0.0, 4.0, 157.5, 0.0),
    (utc(2030, 12, 31, 22), -2.5, 0.0, 5.2, 90.0, 0.1),
    (utc(2030, 12, 31, 23), -0.9, 0.0, 3.9, 67.5, 0.6),
    (utc(2031, 1, 1, 0), 0.7, 0.0, 6.1, 45.0, N),  # sunshine 欠測
]


def test_parses_a_synthetic_amedas_csv():
    text = fixture_text()
    for station_id in ("99901", "a9902"):
        frame = parse.parse_amedas_csv(text, station_id=station_id)
        assert set(PARSED) <= set(frame.columns)
        assert len(frame) == len(EXPECTED_FIXTURE)
        assert list(frame["station_id"]) == [station_id] * len(EXPECTED_FIXTURE)
        assert_types(frame)
        assert rows(frame) == EXPECTED_FIXTURE

    # 24:00:00 labels the last hour of a JST day: 0:00 of the next day, 15:00 UTC the same day
    text = amedas_csv(
        [
            ("2031/1/5 23:00:00", reading("5.5", "0", "0", "1.0", "北")),
            ("2031/1/5 24:00:00", reading("4.5", "2.5", "0", "2.0", "南")),
            ("2031/1/6 1:00:00", reading("3.5", "0", "0", "3.0", "東")),
        ]
    )
    frame = parse.parse_amedas_csv(text, station_id="99903")
    assert rows(frame) == [
        (utc(2031, 1, 5, 14), 5.5, 0.0, 1.0, 0.0, 0.0),
        (utc(2031, 1, 5, 15), 4.5, 2.5, 2.0, 180.0, 0.0),
        (utc(2031, 1, 5, 16), 3.5, 0.0, 3.0, 90.0, 0.0),
    ]


def test_columns_are_found_by_their_headers_not_their_position():
    def lines(official: bool) -> list[tuple[str, dict]]:
        return [
            ("2031/7/10 23:00:00", reading("24.8", "3.5", "0.0", "2.7", "南南西",
                                           official=official, humidity="91")),
            ("2031/7/11 11:00:00", reading("31.2", "0.0", "1.0", "4.4", "東南東",
                                           official=official, humidity="58")),
            ("2031/7/11 12:00:00", reading("32.0", "12.5", "0.3", "6.8", "西北西",
                                           official=official, humidity="63")),
        ]  # fmt: skip

    expected = [
        (utc(2031, 7, 10, 14), 24.8, 3.5, 2.7, 202.5, 0.0),
        (utc(2031, 7, 11, 2), 31.2, 0.0, 4.4, 112.5, 1.0),
        (utc(2031, 7, 11, 3), 32.0, 12.5, 6.8, 292.5, 0.3),
    ]
    layouts = [
        (OFFICIAL_ORDER, True, "\n"),
        (("wind", "humidity", "sun", "temp", "precip"), False, "\r\n"),  # an AMeDAS-only station
        (("precip", "temp", "wind", "sun"), True, "\r\n"),
        (("humidity", "sun", "precip", "wind", "temp"), False, "\n"),
    ]
    for order, official, newline in layouts:
        text = amedas_csv(lines(official), order=order, official=official, newline=newline,
                          station="別架空")  # fmt: skip
        frame = parse.parse_amedas_csv(text, station_id="99904")
        assert len(frame) == 3, order
        assert_types(frame)
        assert rows(frame) == expected, order


def test_wind_direction_names_become_degrees():
    # 2031/3/3 1:00 to 17:00 JST, one compass point an hour at 1.0, 1.5, ... m/s, then calm at 0.2
    speeds = [1 + 0.5 * i for i in range(len(COMPASS))] + [0.2]
    names = [*COMPASS, "静穏"]
    lines = [
        (f"2031/3/3 {i + 1}:00:00", reading("8.0", "0", "0", f"{speed:.1f}", name))
        for i, (speed, name) in enumerate(zip(speeds, names, strict=True))
    ]
    frame = parse.parse_amedas_csv(amedas_csv(lines), station_id="99905")
    got = [(obs, speed, direction) for obs, _, _, speed, direction, _ in rows(frame)]
    expected = [
        (utc(2031, 3, 2, 16) + timedelta(hours=i), speeds[i], 22.5 * i if i < 16 else N)
        for i in range(len(names))
    ]
    assert got == expected


MISSING_OFFICIAL = [  # (JST clock, cells, expected temp, precip, wind speed, direction, sunshine)
    # 欠測: blank values, quality 1
    ("2031/1/5 1:00:00",
     {"temp": ("", "1"), "precip": ("", "", "1"), "sun": ("", "", "1"),
      "wind": ("", "1", "", "1")},
     (N, N, N, N, N)),
    # genuine zeros; calm has no direction but keeps its speed
    ("2031/1/5 2:00:00",
     {"temp": ("0.0", "8"), "precip": ("0", "1", "8"), "sun": ("0", "1", "8"),
      "wind": ("0.0", "8", "静穏", "8")},
     (0.0, 0.0, 0.0, N, 0.0)),
    # /// printed as the value
    ("2031/1/5 3:00:00",
     {"temp": ("///", "1"), "precip": ("///", "", "1"), "sun": ("///", "", "1"),
      "wind": ("///", "1", "///", "1")},
     (N, N, N, N, N)),
    # × printed as the value, whatever the quality column says
    ("2031/1/5 4:00:00",
     {"temp": ("×", "8"), "precip": ("×", "0", "8"), "sun": ("×", "0", "8"),
      "wind": ("×", "8", "×", "8")},
     (N, N, N, N, N)),
    # a value printed but flagged 欠測 (quality 1)
    ("2031/1/5 5:00:00",
     {"temp": ("3.5", "1"), "precip": ("1.5", "0", "1"), "sun": ("0.4", "0", "1"),
      "wind": ("2.2", "1", "南西", "1")},
     (N, N, N, N, N)),
    # not observed (quality 0)
    ("2031/1/5 6:00:00",
     {"temp": ("", "0"), "precip": ("", "", "0"), "sun": ("", "", "0"),
      "wind": ("", "0", "", "0")},
     (N, N, N, N, N)),
    # symbol mode: the quality column reads ///
    ("2031/1/5 7:00:00",
     {"temp": ("", "///"), "precip": ("", "", "///"), "sun": ("", "", "///"),
      "wind": ("", "///", "", "///")},
     (N, N, N, N, N)),
    # 現象なし (1) with a blank amount: no rain, no sun, so zero
    ("2031/1/5 8:00:00",
     {"temp": ("-0.4", "8"), "precip": ("", "1", "8"), "sun": ("", "1", "8"),
      "wind": ("1.3", "8", "西", "8")},
     (-0.4, 0.0, 1.3, 270.0, 0.0)),
    # quasi-normal (5) values kept; a printed 0.0 with the phenomenon present is still zero
    ("2031/1/5 9:00:00",
     {"temp": ("2.6", "5"), "precip": ("0.0", "0", "8"), "sun": ("0.0", "0", "8"),
      "wind": ("0.0", "5", "静穏", "5")},
     (2.6, 0.0, 0.0, N, 0.0)),
    # only the direction is flagged 欠測: the speed stays
    ("2031/1/5 10:00:00",
     {"temp": ("1.9", "8"), "precip": ("2.5", "0", "8"), "sun": ("0.2", "0", "8"),
      "wind": ("3.3", "8", "北", "1")},
     (1.9, 2.5, 3.3, N, 0.2)),
]  # fmt: skip
MISSING_AMEDAS = [  # an AMeDAS-only station: no 現象なし情報 columns
    ("2031/1/6 1:00:00",
     {"temp": ("1.1", "8"), "precip": ("0.0", "8"), "sun": ("0.0", "8"),
      "wind": ("0.4", "8", "北", "8")},
     (1.1, 0.0, 0.4, 0.0, 0.0)),
    ("2031/1/6 2:00:00",
     {"temp": ("", "1"), "precip": ("", "1"), "sun": ("", "1"), "wind": ("", "1", "", "1")},
     (N, N, N, N, N)),
    # a blank amount without 現象なし is missing, not zero
    ("2031/1/6 3:00:00",
     {"temp": ("1.0", "8"), "precip": ("", "8"), "sun": ("", "8"),
      "wind": ("0.9", "8", "北北東", "8")},
     (1.0, N, 0.9, 22.5, N)),
]  # fmt: skip


def test_missing_values_are_null_not_zero():
    for cases, official in ((MISSING_OFFICIAL, True), (MISSING_AMEDAS, False)):
        text = amedas_csv([(clock, cells) for clock, cells, _ in cases], official=official)
        frame = parse.parse_amedas_csv(text, station_id="99906")
        assert len(frame) == len(cases)
        assert_types(frame)
        got = [values for _, *values in rows(frame)]
        for (clock, _, expected), values in zip(cases, got, strict=True):
            assert tuple(values) == expected, clock


# ---------------------------------------------------------------- stations


def _walk(node, keys: set[str], scalars: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            keys.add(str(key))
            _walk(value, keys, scalars)
    elif isinstance(node, list):
        for value in node:
            _walk(value, keys, scalars)
    elif node is not None:
        scalars.add(str(node))


def test_each_venue_maps_to_one_station():
    stations = {venue: parse.station_for_venue(venue) for venue in VENUES}
    for venue, station in stations.items():
        assert isinstance(station, str) and station and station == station.strip(), venue
    assert len(set(stations.values())) == len(VENUES), stations  # one each, none shared

    # NAR and Hong Kong venues, and anything else, have no station here
    for other in ("OOI", "URA", "MON", "ST", "HV", "XXX", ""):
        with pytest.raises((KeyError, ValueError)):
            parse.station_for_venue(other)

    # the mapping is data: stations.yaml names every venue and its station
    keys: set[str] = set()
    scalars: set[str] = set()
    _walk(yaml.safe_load(STATIONS.read_text(encoding="utf-8")), keys, scalars)
    for venue, station in stations.items():
        assert venue in keys, venue
        assert station in scalars, (venue, station)


# ---------------------------------------------------------------- publication time

LAGS = (
    timedelta(0),
    timedelta(minutes=10),
    timedelta(minutes=70),
    timedelta(hours=3, minutes=5),
    timedelta(days=1, seconds=1),
)


def test_rows_carry_publication_time():
    frame = parse.parse_amedas_csv(fixture_text(), station_id="99931")
    columns = list(frame.columns)
    observed = [r[0] for r in EXPECTED_FIXTURE]
    for lag in LAGS:
        out = parse.with_publication_time(frame, lag)
        assert isinstance(out["available_utc"].dtype, pd.DatetimeTZDtype), lag
        assert str(out["available_utc"].dt.tz) == "UTC", lag
        pairs = sorted(zip(map(ts, out["obs_utc"]), map(ts, out["available_utc"]), strict=True))
        assert pairs == [(obs, obs + lag) for obs in observed], lag
        pd.testing.assert_frame_equal(  # every other column as parsed
            out[columns].reset_index(drop=True), frame.reset_index(drop=True)
        )
        assert list(frame.columns) == columns  # the input frame is left alone

    # the configured lag (an owner decision: its value is not pinned here)
    configured = parse.publication_lag()
    assert isinstance(configured, timedelta)
    assert configured >= timedelta(0)
    out = parse.with_publication_time(frame, configured)
    assert len(out) == len(EXPECTED_FIXTURE)
    assert ((out["available_utc"] - out["obs_utc"]) == configured).all()

    # published before it was observed is a leak, never a lag
    with pytest.raises(ValueError):
        parse.with_publication_time(frame, timedelta(minutes=-1))


# ---------------------------------------------------------------- the store and the view at t

INGESTED = utc(2031, 2, 1, 3)  # an ingest stamp after every row's publication


def _filler(kind: str, column: str, i: int):
    if kind == schemas.UTC:
        return INGESTED
    if kind == schemas.STR:
        return f"{column}-{i}"
    if kind == schemas.FLOAT:
        return 0.5 + i
    if kind == schemas.INT:
        return i + 1
    if kind == schemas.BOOL:
        return i % 2 == 0
    if kind == schemas.DATE:
        return date(2031, 1, 5)
    if kind == schemas.INT_LIST:
        return [i + 1]
    if kind == schemas.FLOAT_LIST:
        return [0.5 + i]
    raise AssertionError(f"{column}: no synthetic value for {kind}")


def weather_table(given: list[dict]) -> pa.Table:
    """Rows of the declared weather table: the given values, every other column a synthetic one
    (`ingested_utc` and any other time column: INGESTED)."""
    spec = schemas.TABLES["weather"]
    schema = pa.schema([(c, ARROW_TYPES[k]) for c, k in spec.columns.items()])
    rows_ = []
    for i, values in enumerate(given):
        unknown = set(values) - set(spec.columns)
        assert not unknown, f"the weather table declares no column {sorted(unknown)}"
        rows_.append({c: _filler(k, c, i) for c, k in spec.columns.items()} | values)
    return pa.Table.from_pylist(rows_, schema=schema)


def connect_weather(table: pa.Table) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.register("_arrow", table)
    con.execute('CREATE TABLE "weather" AS SELECT * FROM _arrow')
    con.unregister("_arrow")
    return con


def test_the_weather_table_declares_the_parsed_columns_and_three_times():
    spec = schemas.TABLES["weather"]
    assert spec.name == "weather"
    for column in ("obs_utc", "available_utc", "ingested_utc"):
        assert spec.columns.get(column) == schemas.UTC, column
    assert spec.columns.get("station_id") == schemas.STR
    for column in VALUES:
        assert spec.columns.get(column) == schemas.FLOAT, column
    assert spec.columns.get("jurisdiction") == schemas.STR


X, Y = "99911", "99912"
VIEW_ROWS = [  # (station_id, obs_utc, available_utc, temp_c)
    (X, utc(2031, 1, 5, 0), utc(2031, 1, 5, 0, 10), 1.0),
    (X, utc(2031, 1, 5, 1), utc(2031, 1, 5, 1, 10), 2.0),
    (X, utc(2031, 1, 5, 2), utc(2031, 1, 5, 4, 30), 3.0),  # published late
    (X, utc(2031, 1, 5, 3), None, 4.0),  # no publication time: never in a view
    (Y, utc(2031, 1, 5, 0), utc(2031, 1, 5, 0), -5.5),  # published as observed
    (Y, utc(2031, 1, 5, 1), utc(2031, 1, 5, 2), -6.5),
]


def test_weather_at_t_uses_only_rows_available_by_t():
    rules = load_availability("japan")
    con = connect_weather(
        weather_table(
            [
                {"station_id": s, "obs_utc": obs, "available_utc": available, "temp_c": temp}
                for s, obs, available, temp in VIEW_ROWS
            ]
        )
    )
    try:
        published = sorted({a for _, _, a, _ in VIEW_ROWS if a is not None})
        probes = [
            utc(2000, 1, 1),
            utc(2031, 1, 5, 3, 30),  # every row observed, two not yet published
            utc(2100, 1, 1),
            *published,
            *[a - US for a in published],
            *[obs for _, obs, _, _ in VIEW_ROWS],
        ]
        for t in probes:
            view = read_asof(con, "weather", t, rules=rules)
            assert {"station_id", "obs_utc", "available_utc", *VALUES} <= set(view.columns), t
            got = sorted(
                (s, ts(obs), _num(temp))
                for s, obs, temp in zip(view["station_id"], view["obs_utc"], view["temp_c"],
                                        strict=True)
            )  # fmt: skip
            want = sorted(
                (s, obs, temp) for s, obs, a, temp in VIEW_ROWS if a is not None and a <= t
            )
            assert got == want, t.isoformat()

        # the latest row per station available by each key's t, keys' order kept
        joins = [  # (station_id, t_utc, expected temp_c)
            (X, utc(2031, 1, 5, 1, 10) - US, 1.0),
            (Y, utc(2031, 1, 4, 23, 59), N),
            (X, utc(2031, 1, 5, 1, 10), 2.0),
            (X, utc(2031, 1, 5, 4, 30) - US, 2.0),  # 02:00 observed, not yet published
            (Y, utc(2031, 1, 5, 0), -5.5),
            (X, utc(2031, 1, 5, 4, 30), 3.0),
            (X, utc(2100, 1, 1), 3.0),  # the 03:00 row is never published
            (Y, utc(2031, 1, 5, 2) - US, -5.5),
            (Y, utc(2031, 1, 5, 2), -6.5),
            ("99999", utc(2100, 1, 1), N),
        ]
        keys = pd.DataFrame(
            {
                "station_id": [s for s, _, _ in joins],
                "t_utc": pd.to_datetime([t for _, t, _ in joins], utc=True),
            }
        )
        joined = asof_join(keys, "weather", on=["station_id"], rules=rules, con=con)
        assert list(joined["station_id"]) == [s for s, _, _ in joins]
        assert [_num(v) for v in joined["temp_c"]] == [temp for _, _, temp in joins]
    finally:
        con.close()


def test_parsed_rows_fill_the_weather_table_and_open_at_their_publication_time():
    lag = timedelta(minutes=40)
    frame = parse.with_publication_time(
        parse.parse_amedas_csv(fixture_text(), station_id="99921"), lag
    )
    records = []
    for r in frame.to_dict("records"):
        record = {c: r[c] for c in (*PARSED, "available_utc")}
        record["obs_utc"], record["available_utc"] = ts(r["obs_utc"]), ts(r["available_utc"])
        records.append(record)
    rules = load_availability("japan")
    con = connect_weather(weather_table(records))
    try:
        for k in (0, 1, 5, 10):
            opens = EXPECTED_FIXTURE[k][0] + lag
            for t in (opens - US, opens):
                view = read_asof(con, "weather", t, rules=rules)
                want = [r[0] for r in EXPECTED_FIXTURE if r[0] + lag <= t]
                assert sorted(ts(v) for v in view["obs_utc"]) == want, t.isoformat()
        view = read_asof(con, "weather", utc(2100, 1, 1), rules=rules)
        assert set(view["station_id"]) == {"99921"}
        assert rows(view) == EXPECTED_FIXTURE
    finally:
        con.close()
