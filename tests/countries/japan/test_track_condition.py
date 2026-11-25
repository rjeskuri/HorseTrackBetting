"""JRA track condition: rail, cushion and moisture with publication times (ticket #90).

Contract pinned here (the coder follows it), in `horsetrack.countries.japan.jra.track_condition`
(pure: no network) and the store's `core.store.schemas`:

- `schemas.TABLES["track_condition"]` declares, in order: `jurisdiction` (str; every store table
  carries it) and the ticket's columns venue (str), race_date (date), surface (str), going (str),
  rail (str), cushion, moisture_goal, moisture_4c (float), published_utc, available_utc,
  ingested_utc (utc).
- `parse_baba_page(html: str, *, fetched_utc) -> DataFrame`: a going page (馬場情報, the structure
  of `jra.going`'s synthetic pages, passed as text) as one row per surface, with exactly the
  table's columns in its order; jurisdiction "jra"; venue, race_date, going codes, cushion and
  moistures as `jra.going.parse_going_page` reads them (a value not printed or printed as － is
  NaN, never 0; dirt has no cushion); `rail` 'A'-'D' on the turf row (NFKC: 'Ｂコース' is 'B'),
  None on dirt and when the page prints none or －. published_utc is the page's 発表 clock
  (Tokyo) in UTC, available_utc = published_utc, ingested_utc = `fetched_utc` (the caller's fetch
  time, so the parser stays pure and deterministic). A naive `fetched_utc` is a ValueError.
- `parse_yearly_text(text: str) -> DataFrame`: text extracted from a yearly record, in the layout
  of tests/fixtures/synthetic/jra_track_condition_yearly.txt (its header comment states it), as
  one row per (venue, race_date, surface) printed, with the same columns and rules. Every row's
  published_utc and available_utc is the document's own 発表 time: its values are not known from
  this source before the document exists. ingested_utc is only pinned to be a UTC column.
- `days_since_rail_move(frame) -> Series` on the frame's index: per (venue, surface), counted in
  race days (the distinct race dates that venue and surface has rows on, not calendar days), 0 on
  the first race day of a new rail letter, +1 per race day after it. Null before the first move
  seen in the frame (the conservative reading: the count is unknown), on a day whose rail is
  unknown (no letter on any of its rows), and on every dirt row. A day of unknown rail is still a
  race day: the count goes on across it when the letters either side are the same. Every row of
  a day (several publications) gets the day's value. Independent of row order.
- `as_of(frame, decision_utc) -> DataFrame`: the condition at t, `jra.going.as_of`'s rule on the
  table's columns: per (venue, race_date, surface), the row with the latest available_utc at or
  before t (ties: the latest ingested_utc); keys with none are absent; independent of row order.
  A naive `decision_utc` is a ValueError.

Pages and text are synthetic (tests/fixtures/synthetic/ and the builders below). Expected values
are written out here, never computed by the code under test.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core import paths
from horsetrack.core.store import schemas, tables
from horsetrack.countries.japan.jra import track_condition as tc

UTC = timezone.utc
US = timedelta(microseconds=1)
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "synthetic"
YEARLY_FIXTURE = "jra_track_condition_yearly.txt"
KEY = ["venue", "race_date", "surface"]
GOING_LABEL = {"GOOD": "良", "GOOD TO YIELDING": "稍重", "SOFT": "重", "HEAVY": "不良"}
GTY = "GOOD TO YIELDING"
_WEEKDAY = "月火水木金土日"

TABLE_COLUMNS = {
    "jurisdiction": schemas.STR,
    "venue": schemas.STR,
    "race_date": schemas.DATE,
    "surface": schemas.STR,
    "going": schemas.STR,
    "rail": schemas.STR,
    "cushion": schemas.FLOAT,
    "moisture_goal": schemas.FLOAT,
    "moisture_4c": schemas.FLOAT,
    "published_utc": schemas.UTC,
    "available_utc": schemas.UTC,
    "ingested_utc": schemas.UTC,
}
UTC_COLUMNS = ("published_utc", "available_utc", "ingested_utc")
VALUE_COLUMNS = ("cushion", "moisture_goal", "moisture_4c")
# kind -> the Arrow type the store writes it as (independent of the code under test)
ARROW = {
    schemas.STR: pa.string(),
    schemas.DATE: pa.date32(),
    schemas.FLOAT: pa.float64(),
    schemas.UTC: pa.timestamp("us", tz="UTC"),
}


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _day(d: date) -> str:
    return f"{d.month}月{d.day}日（{_WEEKDAY[d.weekday()]}曜）"


def _cell(v: float | None, suffix: str = "") -> str:
    return "－" if v is None else f"{v:.1f}{suffix}"


def baba_page(
    race_date: date,
    meeting: str,
    published_jst: datetime,
    *,
    turf: tuple[str, float | None, float | None, float | None] | None,
    dirt: tuple[str, float | None, float | None] | None,
    rail: str | None = None,
    cushion_block: bool = True,
) -> str:
    """A going page like the jra_going_*.html fixtures, as text. turf = (going code, cushion,
    goal %, 4c %), dirt = (going code, goal %, 4c %); None leaves a surface out, a None value
    prints －. `rail` is the printed course cell ('Aコース'); None leaves the rail block out."""
    surfaces = []
    if turf is not None:
        surfaces.append(("芝", turf[0], turf[2], turf[3]))
    if dirt is not None:
        surfaces.append(("ダート", dirt[0], dirt[1], dirt[2]))
    going_rows = "".join(
        f"<tr><th>{s}</th><td>{GOING_LABEL[g]}</td></tr>" for s, g, _, _ in surfaces
    )
    moist_rows = "".join(
        f"<tr><th>{s}</th><td>{_cell(a, '%')}</td><td>{_cell(b, '%')}</td></tr>"
        for s, _, a, b in surfaces
    )
    rail_block = (
        ""
        if rail is None
        else f'<div class="rail"><h3>使用コース（芝）</h3><p class="value">{rail}</p></div>'
    )
    cushion = ""
    if cushion_block and turf is not None:
        cushion = (
            '<div class="cushion"><h3>クッション値（芝）</h3>'
            f'<p class="value">{_cell(turf[1])}</p></div>'
        )
    p = published_jst
    return (
        "<!-- synthetic: JRA going page (馬場情報) -->\n"
        '<!doctype html><html lang="ja"><head><meta charset="Shift_JIS"><title>馬場情報</title>'
        '</head><body><div id="baba">'
        f'<h2 class="date_line">{race_date.year}年{_day(race_date)} {meeting}</h2>'
        f'<p class="publish">{_day(p.date())}{p.hour}時{p.minute:02d}分 発表</p>'
        f'<table class="going">{going_rows}</table>{rail_block}{cushion}'
        '<table class="moisture"><caption>含水率</caption>'
        f"<tr><th></th><th>ゴール前</th><th>4コーナー</th></tr>{moist_rows}</table>"
        "</div></body></html>"
    )


def yearly_text(
    year: int,
    published_jst: datetime,
    turf: list[tuple[date, str, str, str, str, str, str]],
    dirt: list[tuple[date, str, str, str, str]],
) -> str:
    """A yearly record in the layout of jra_track_condition_yearly.txt. Cells are printed as
    given: turf rows (race date, meeting, going word, course, cushion, goal, 4c), dirt rows (race
    date, meeting, going word, goal, 4c)."""

    def row(d: date, cells: tuple[str, ...]) -> str:
        return " ".join([f"{d.month}月{d.day}日（{_WEEKDAY[d.weekday()]}）", *cells])

    p = published_jst
    lines = [
        "# synthetic: JRA yearly track-condition record, built by the test",
        f"{year}年 馬場情報（使用コース・クッション値・含水率）",
        f"{p.year}年{_day(p.date())}{p.hour}時{p.minute:02d}分 発表",
        "",
        "【芝】",
        "開催日 開催 馬場状態 使用コース クッション値 含水率（ゴール前） 含水率（4コーナー）",
        *(row(d, cells) for d, *cells in turf),
        "",
        "【ダート】",
        "開催日 開催 馬場状態 含水率（ゴール前） 含水率（4コーナー）",
        *(row(d, cells) for d, *cells in dirt),
    ]
    return "\n".join(lines) + "\n"


def _num(v) -> float | None:
    if v is None or pd.isna(v):
        return None
    return round(float(v), 3)


def _str(v) -> str | None:
    return None if v is None or pd.isna(v) else v


def rows(frame: pd.DataFrame) -> list[dict]:
    """The frame's rows, values normalised for comparison (NaN -> None), sorted by key."""
    out = [
        {
            "venue": r["venue"],
            "race_date": pd.Timestamp(r["race_date"]).date(),
            "surface": r["surface"],
            "going": _str(r["going"]),
            "rail": _str(r["rail"]),
            "cushion": _num(r["cushion"]),
            "moisture_goal": _num(r["moisture_goal"]),
            "moisture_4c": _num(r["moisture_4c"]),
            "published_utc": pd.Timestamp(r["published_utc"]).to_pydatetime(),
            "available_utc": pd.Timestamp(r["available_utc"]).to_pydatetime(),
        }
        for r in frame.to_dict("records")
    ]
    return sorted(out, key=lambda r: (r["venue"], r["race_date"], r["surface"], r["available_utc"]))


def expect(venue, race_date, surface, going, rail, cushion, goal, corner, published) -> dict:
    return {
        "venue": venue,
        "race_date": race_date,
        "surface": surface,
        "going": going,
        "rail": rail,
        "cushion": cushion,
        "moisture_goal": goal,
        "moisture_4c": corner,
        "published_utc": published,
        "available_utc": published,
    }


def assert_table_frame(frame: pd.DataFrame) -> None:
    """The frame has the table's columns in its order, with the dtypes of their kinds."""
    assert list(frame.columns) == list(TABLE_COLUMNS)
    assert len(frame) > 0
    assert (frame["jurisdiction"] == "jra").all()
    for column, kind in TABLE_COLUMNS.items():
        series = frame[column]
        if kind == schemas.UTC:
            assert isinstance(series.dtype, pd.DatetimeTZDtype), column
            assert str(series.dt.tz) == "UTC", column
        elif kind == schemas.FLOAT:
            assert series.dtype == "float64", column
        elif kind == schemas.DATE:
            assert all(isinstance(v, date) and not isinstance(v, datetime) for v in series), column
        else:
            assert all(isinstance(v, str) or pd.isna(v) for v in series), column


# ---------------------------------------------------------------- the store table


def test_track_condition_is_a_store_table():
    spec = schemas.TABLES["track_condition"]
    assert spec.name == "track_condition"
    assert list(spec.columns.items()) == list(TABLE_COLUMNS.items())
    assert spec.sort_by and set(spec.sort_by) <= set(spec.columns)


# ---------------------------------------------------------------- the going page

TOK_SAT = ("TOK", date(2026, 11, 28))
KYOTO = baba_page(
    date(2026, 11, 28), "5回京都7日", datetime(2026, 11, 28, 8, 0),  # 23:00 UTC the day before
    turf=("GOOD", 10.6, 9.8, 10.4), dirt=("GOOD", 3.5, 3.9), rail="Ｂコース",
)  # fmt: skip
KOKURA = baba_page(
    date(2026, 2, 1), "1回小倉4日", datetime(2026, 1, 31, 16, 40),
    turf=(GTY, 12.4, 8.0, 8.6), dirt=("HEAVY", None, 2.2), rail="Dコース",
)  # fmt: skip
HANSHIN_NO_CUSHION = baba_page(
    date(2025, 12, 27), "5回阪神7日", datetime(2025, 12, 27, 9, 5),
    turf=("SOFT", None, 15.0, 14.1), dirt=("SOFT", 11.2, 10.7), rail="Aコース",
    cushion_block=False,
)  # fmt: skip
NAKAYAMA_TURF_ONLY = baba_page(
    date(2024, 3, 31), "3回中山4日", datetime(2024, 3, 30, 18, 45),
    turf=("HEAVY", 7.1, 18.2, 17.5), dirt=None, rail="Cコース",
)  # fmt: skip

# case -> (page text, fetched_utc, published_utc, expected rows)
PAGES = {
    "day_before": (
        fixture("jra_going_day_before.html"), utc(2026, 11, 27, 7, 12, 5), utc(2026, 11, 27, 6, 0),
        [
            expect(*TOK_SAT, "dirt", GTY, None, None, 5.1, 4.9, utc(2026, 11, 27, 6, 0)),
            expect(*TOK_SAT, "turf", "GOOD", None, 9.4, 12.3, 11.8, utc(2026, 11, 27, 6, 0)),
        ],
    ),
    "race_day_update": (
        fixture("jra_going_race_day_update.html"), utc(2026, 11, 28, 1, 0, 41),
        utc(2026, 11, 28, 0, 30),
        [
            expect(*TOK_SAT, "dirt", "SOFT", None, None, 7.9, 8.3, utc(2026, 11, 28, 0, 30)),
            expect(*TOK_SAT, "turf", GTY, None, 8.7, 14.6, 15.2, utc(2026, 11, 28, 0, 30)),
        ],
    ),
    "no_cushion": (
        fixture("jra_going_no_cushion.html"), utc(2021, 8, 13, 9, 0), utc(2021, 8, 13, 7, 15),
        [
            expect("SAP", date(2021, 8, 14), "dirt", "GOOD", None, None, 3.2, 2.8,
                   utc(2021, 8, 13, 7, 15)),
            expect("SAP", date(2021, 8, 14), "turf", "GOOD", None, None, 13.0, None,
                   utc(2021, 8, 13, 7, 15)),
        ],
    ),
    "rail_a": (
        fixture("jra_going_rail_a.html"), utc(2026, 11, 27, 6, 3), utc(2026, 11, 27, 6, 0),
        [
            expect(*TOK_SAT, "dirt", "GOOD", None, None, 5.0, 4.0, utc(2026, 11, 27, 6, 0)),
            expect(*TOK_SAT, "turf", "GOOD", "A", 9.0, 12.0, 11.0, utc(2026, 11, 27, 6, 0)),
        ],
    ),
    "rail_c_fullwidth": (
        fixture("jra_going_rail_c_fullwidth.html"), utc(2026, 11, 28, 8, 0),
        utc(2026, 11, 28, 7, 30),
        [
            expect("TOK", date(2026, 11, 29), "dirt", GTY, None, None, 5.0, 4.0,
                   utc(2026, 11, 28, 7, 30)),
            expect("TOK", date(2026, 11, 29), "turf", "GOOD", "C", 9.0, 12.0, 11.0,
                   utc(2026, 11, 28, 7, 30)),
        ],
    ),
    "rail_dash": (
        fixture("jra_going_rail_dash.html"), utc(2026, 12, 4, 6, 30), utc(2026, 12, 4, 6, 0),
        [
            expect("NAK", date(2026, 12, 5), "dirt", "GOOD", None, None, 5.0, 4.0,
                   utc(2026, 12, 4, 6, 0)),
            expect("NAK", date(2026, 12, 5), "turf", "GOOD", None, 9.0, 12.0, 11.0,
                   utc(2026, 12, 4, 6, 0)),
        ],
    ),
    "kyoto_fullwidth_b": (
        KYOTO, utc(2026, 11, 27, 23, 20), utc(2026, 11, 27, 23, 0),
        [
            expect("KYO", date(2026, 11, 28), "dirt", "GOOD", None, None, 3.5, 3.9,
                   utc(2026, 11, 27, 23, 0)),
            expect("KYO", date(2026, 11, 28), "turf", "GOOD", "B", 10.6, 9.8, 10.4,
                   utc(2026, 11, 27, 23, 0)),
        ],
    ),
    "kokura_rail_d": (
        KOKURA, utc(2026, 11, 30, 12, 0), utc(2026, 1, 31, 7, 40),  # a backfill
        [
            expect("KOK", date(2026, 2, 1), "dirt", "HEAVY", None, None, None, 2.2,
                   utc(2026, 1, 31, 7, 40)),
            expect("KOK", date(2026, 2, 1), "turf", GTY, "D", 12.4, 8.0, 8.6,
                   utc(2026, 1, 31, 7, 40)),
        ],
    ),
    "hanshin_no_cushion_block": (
        HANSHIN_NO_CUSHION, utc(2025, 12, 27, 0, 15), utc(2025, 12, 27, 0, 5),
        [
            expect("HSN", date(2025, 12, 27), "dirt", "SOFT", None, None, 11.2, 10.7,
                   utc(2025, 12, 27, 0, 5)),
            expect("HSN", date(2025, 12, 27), "turf", "SOFT", "A", None, 15.0, 14.1,
                   utc(2025, 12, 27, 0, 5)),
        ],
    ),
    "nakayama_turf_only": (
        NAKAYAMA_TURF_ONLY, utc(2024, 3, 30, 10, 0), utc(2024, 3, 30, 9, 45),
        [
            expect("NAK", date(2024, 3, 31), "turf", "HEAVY", "C", 7.1, 18.2, 17.5,
                   utc(2024, 3, 30, 9, 45)),
        ],
    ),
}  # fmt: skip


@pytest.mark.parametrize("case", list(PAGES))
def test_parse_baba_page_reads_synthetic_pages(case):
    html, fetched, _, expected = PAGES[case]
    frame = tc.parse_baba_page(html, fetched_utc=fetched)

    assert_table_frame(frame)
    assert rows(frame) == expected


@pytest.mark.parametrize("case", list(PAGES))
def test_rows_carry_publication_time(case):
    html, fetched, published, _ = PAGES[case]
    frame = tc.parse_baba_page(html, fetched_utc=fetched)

    for column in UTC_COLUMNS:
        assert isinstance(frame[column].dtype, pd.DatetimeTZDtype), column
        assert str(frame[column].dt.tz) == "UTC", column
        assert frame[column].notna().all(), column
    # the page's 発表 clock, not the fetch time
    assert (frame["published_utc"] == pd.Timestamp(published)).all()
    assert (frame["available_utc"] == frame["published_utc"]).all()
    assert (frame["ingested_utc"] == pd.Timestamp(fetched)).all()
    assert pd.Timestamp(published) != pd.Timestamp(fetched)


# ---------------------------------------------------------------- the yearly text

YEARLY_PUB = utc(2026, 1, 9, 6, 0)  # 2026年1月9日 15時00分 in Tokyo
T4, T5, T11, T12 = (date(2025, 10, d) for d in (4, 5, 11, 12))
YEARLY_ROWS = [
    expect("KYO", T4, "dirt", "GOOD", None, None, 2.9, 2.6, YEARLY_PUB),
    expect("KYO", T4, "turf", "GOOD", "C", 10.1, 11.0, 10.6, YEARLY_PUB),
    expect("KYO", T5, "dirt", "HEAVY", None, None, 14.5, 13.8, YEARLY_PUB),
    expect("KYO", T5, "turf", "SOFT", None, 10.3, 16.4, 16.0, YEARLY_PUB),
    expect("TOK", T4, "dirt", "GOOD", None, None, 3.9, 3.5, YEARLY_PUB),
    expect("TOK", T4, "turf", "GOOD", "A", 9.2, 13.1, 12.4, YEARLY_PUB),
    expect("TOK", T5, "dirt", GTY, None, None, 6.2, None, YEARLY_PUB),
    expect("TOK", T5, "turf", "GOOD", "A", 9.4, 12.8, 12.0, YEARLY_PUB),
    expect("TOK", T11, "dirt", "SOFT", None, None, 9.8, 9.1, YEARLY_PUB),
    expect("TOK", T11, "turf", GTY, "B", 8.6, 15.2, None, YEARLY_PUB),
    expect("TOK", T12, "dirt", GTY, None, None, 7.4, 7.0, YEARLY_PUB),
    expect("TOK", T12, "turf", "GOOD", "B", None, 13.9, 13.3, YEARLY_PUB),
]


def test_cushion_and_moisture_parse_from_synthetic_text():
    frame = tc.parse_yearly_text(fixture(YEARLY_FIXTURE))

    assert_table_frame(frame)
    assert rows(frame) == YEARLY_ROWS
    # a dash is null, never 0: the Tokyo 10-12 turf cushion, the 10-11 turf and 10-05 dirt
    # fourth-corner moistures, and every dirt cushion
    assert not (frame[list(VALUE_COLUMNS)] == 0).any().any()
    assert frame[list(VALUE_COLUMNS)].isna().sum().to_dict() == {
        "cushion": 7,
        "moisture_goal": 0,
        "moisture_4c": 2,
    }


N5, N7, N8 = (date(2023, 1, d) for d in (5, 7, 8))
OTHER_YEAR_PUB = utc(2024, 1, 12, 1, 30)  # 2024年1月12日 10時30分 in Tokyo
OTHER_YEAR = yearly_text(
    2023,
    datetime(2024, 1, 12, 10, 30),
    turf=[
        (N5, "1回中山1日", "良", "C", "10.8", "9.6%", "9.1%"),
        (N5, "1回中京1日", "稍重", "A", "8.9", "14.0%", "13.2%"),
        (N7, "1回中山2日", "良", "C", "11.2", "9.0%", "－"),
        (N8, "1回中山3日", "良", "Ｄ", "11.0", "8.8%", "8.5%"),
        (N7, "1回中京2日", "重", "A", "-", "17.1%", "16.8%"),
    ],
    dirt=[
        (N5, "1回中山1日", "良", "2.4%", "2.1%"),
        (N7, "1回中京2日", "不良", "－", "15.3%"),
    ],
)
OTHER_YEAR_ROWS = [
    expect("CKY", N5, "turf", GTY, "A", 8.9, 14.0, 13.2, OTHER_YEAR_PUB),
    expect("CKY", N7, "dirt", "HEAVY", None, None, None, 15.3, OTHER_YEAR_PUB),
    expect("CKY", N7, "turf", "SOFT", "A", None, 17.1, 16.8, OTHER_YEAR_PUB),
    expect("NAK", N5, "dirt", "GOOD", None, None, 2.4, 2.1, OTHER_YEAR_PUB),
    expect("NAK", N5, "turf", "GOOD", "C", 10.8, 9.6, 9.1, OTHER_YEAR_PUB),
    expect("NAK", N7, "turf", "GOOD", "C", 11.2, 9.0, None, OTHER_YEAR_PUB),
    expect("NAK", N8, "turf", "GOOD", "D", 11.0, 8.8, 8.5, OTHER_YEAR_PUB),
]


def test_parse_yearly_text_over_another_year_and_venues():
    frame = tc.parse_yearly_text(OTHER_YEAR)

    assert_table_frame(frame)
    assert rows(frame) == OTHER_YEAR_ROWS


@pytest.mark.parametrize(
    ("text", "published"),
    [(fixture(YEARLY_FIXTURE), YEARLY_PUB), (OTHER_YEAR, OTHER_YEAR_PUB)],
    ids=["fixture", "other_year"],
)
def test_yearly_rows_carry_the_documents_publication_time(text, published):
    frame = tc.parse_yearly_text(text)

    for column in ("published_utc", "available_utc"):
        assert isinstance(frame[column].dtype, pd.DatetimeTZDtype), column
        assert str(frame[column].dt.tz) == "UTC", column
        assert frame[column].notna().all(), column
    assert (frame["published_utc"] == pd.Timestamp(published)).all()
    assert (frame["available_utc"] == frame["published_utc"]).all()
    assert isinstance(frame["ingested_utc"].dtype, pd.DatetimeTZDtype)


@pytest.fixture
def data(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(d))
    return d


def test_parsed_rows_land_in_the_declared_table(data):
    frame = pd.concat(
        [
            tc.parse_baba_page(PAGES["no_cushion"][0], fetched_utc=PAGES["no_cushion"][1]),
            tc.parse_baba_page(KYOTO, fetched_utc=utc(2026, 11, 27, 23, 20)),
            tc.parse_yearly_text(fixture(YEARLY_FIXTURE)),
        ],
        ignore_index=True,
    )
    schema = pa.schema([(c, ARROW[k]) for c, k in TABLE_COLUMNS.items()])
    path = paths.processed_dir("jra") / "track_condition.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False), path)

    stored = tables.read_table("jra", "track_condition")
    assert list(stored.columns) == list(TABLE_COLUMNS)
    assert rows(stored) == rows(frame)
    assert len(stored) == 4 + len(YEARLY_ROWS)


# ---------------------------------------------------------------- missing values


def test_a_page_without_cushion_gives_null():
    pages = [
        tc.parse_baba_page(fixture("jra_going_no_cushion.html"), fetched_utc=utc(2021, 8, 13, 9)),
        tc.parse_baba_page(HANSHIN_NO_CUSHION, fetched_utc=utc(2025, 12, 27, 0, 15)),
        tc.parse_baba_page(KOKURA, fetched_utc=utc(2026, 11, 30, 12, 0)),
    ]
    sapporo, hanshin, kokura = ({r["surface"]: r for r in p.to_dict("records")} for p in pages)

    # the cushion prints － (Sapporo) or is not on the page (Hanshin); dirt never has one
    for page in (sapporo, hanshin):
        assert math.isnan(page["turf"]["cushion"]) and math.isnan(page["dirt"]["cushion"])
    assert math.isnan(kokura["dirt"]["cushion"])
    # an unmeasured moisture is null too, while the measured ones beside it are read
    assert math.isnan(sapporo["turf"]["moisture_4c"]) and sapporo["turf"]["moisture_goal"] == 13.0
    assert math.isnan(kokura["dirt"]["moisture_goal"]) and kokura["dirt"]["moisture_4c"] == 2.2
    assert kokura["turf"]["cushion"] == 12.4
    for page in pages:
        assert not (page[list(VALUE_COLUMNS)] == 0).any().any()


# ---------------------------------------------------------------- days since the rail moved


def _counts(series: pd.Series) -> dict:
    out = {}
    for i, v in series.items():
        if pd.isna(v):
            out[i] = None
        else:
            assert float(v) == int(v), f"row {i}: {v!r} is not a whole number of race days"
            out[i] = int(v)
    return out


# Ten Tokyo race days, Saturday and Sunday over five weekends: rail A on days 1-8, B from day 9.
MEETING_DAYS = [
    date(2026, 10, 10) + timedelta(days=d) for d in (0, 1, 7, 8, 14, 15, 21, 22, 28, 29)
]


def _meeting_pages() -> list[tuple[int, pd.DataFrame]]:
    """(race day number, parsed rows) for the day-before and race-morning pages of each day."""
    out = []
    for n, day in enumerate(MEETING_DAYS, start=1):
        rail = "Aコース" if n <= 8 else "Bコース"
        eve = day - timedelta(days=1)
        for published in (datetime(eve.year, eve.month, eve.day, 15, 0),
                          datetime(day.year, day.month, day.day, 9, 30)):  # fmt: skip
            html = baba_page(
                day, f"4回東京{n}日", published,
                turf=("GOOD", 8.0 + n / 10, 12.0, 11.5), dirt=("GOOD", 4.0, 3.6), rail=rail,
            )  # fmt: skip
            fetched = datetime(published.year, published.month, published.day, tzinfo=UTC)
            out.append((n, tc.parse_baba_page(html, fetched_utc=fetched + timedelta(hours=8))))
    return out


def test_rail_letter_and_days_since_move():
    parts = _meeting_pages()
    frame = pd.concat([p for _, p in parts], ignore_index=True)
    day_no = [n for n, p in parts for _ in range(len(p))]
    frame.index = [5 * i + 2 for i in range(len(frame))]  # labels, not positions
    day_of = dict(zip(frame.index, day_no, strict=True))

    turf = frame["surface"] == "turf"
    assert turf.sum() == 20 and (~turf).sum() == 20
    for i, r in frame.iterrows():
        want = None if r["surface"] == "dirt" else ("A" if day_of[i] <= 8 else "B")
        assert _str(r["rail"]) == want, (day_of[i], r["surface"])

    # 0 on day 9, 1 on day 10; null before the first move seen and on dirt
    expected = {
        i: ({9: 0, 10: 1}.get(day_of[i]) if frame.at[i, "surface"] == "turf" else None)
        for i in frame.index
    }
    for source in (frame, frame.iloc[::-1], frame.sample(frac=1, random_state=90)):
        got = tc.days_since_rail_move(source)
        assert isinstance(got, pd.Series)
        assert sorted(got.index) == sorted(source.index)
        assert _counts(got) == expected

    # the yearly record gives the same reading: Tokyo A, A, B, B; Kyoto C, then a day of －
    yearly = tc.parse_yearly_text(fixture(YEARLY_FIXTURE))
    got = _counts(tc.days_since_rail_move(yearly))
    by_key = {
        (r["venue"], r["race_date"], r["surface"]): got[i]
        for i, r in zip(yearly.index, yearly.to_dict("records"), strict=True)
    }
    assert by_key == {
        ("TOK", T4, "turf"): None, ("TOK", T5, "turf"): None,
        ("TOK", T11, "turf"): 0, ("TOK", T12, "turf"): 1,
        ("KYO", T4, "turf"): None, ("KYO", T5, "turf"): None,
        **{(v, d, "dirt"): None for v in ("TOK", "KYO") for d in (T4, T5, T11, T12)
           if not (v == "KYO" and d in (T11, T12))},
    }  # fmt: skip


S = [date(2026, 10, 3) + timedelta(days=d) for d in (0, 1, 7, 8, 14, 15, 21, 22, 28, 29)]
N = None


def _seq(venue: str, surface: str, steps: list[tuple[date, str | None, int | None]]) -> list:
    return [(venue, d, surface, rail, want) for d, rail, want in steps]


# case -> rows (venue, race_date, surface, rail, expected days since the move[, published_utc])
SEQUENCES = {
    "restarts_at_each_move": _seq("TOK", "turf", [
        (S[0], "A", N), (S[1], "A", N), (S[2], "A", N), (S[3], "B", 0), (S[4], "B", 1),
        (S[5], "B", 2), (S[6], "C", 0), (S[7], "C", 1), (S[8], "C", 2),
    ]),
    # 6, 7, 27 and 28 calendar days after the move: still 1, 2, 3, 4 race days
    "race_days_not_calendar_days": _seq("TOK", "turf", [
        (S[0], "A", N), (S[1], "B", 0), (S[2], "B", 1), (S[3], "B", 2), (S[8], "B", 3),
        (S[9], "B", 4),
    ]),
    "each_venue_on_its_own": [
        row for k in range(5)
        for row in (
            _seq("TOK", "turf", [(S[k], "AABBB"[k], [N, N, 0, 1, 2][k])])
            + _seq("KYO", "turf", [(S[k], "CDDDA"[k], [N, 0, 1, 2, 0][k])])
        )
    ],
    # race days are the venue's turf days: the dirt-only day S[2] does not count for turf
    "each_surface_on_its_own": (
        _seq("TOK", "turf", [(S[0], "A", N), (S[1], "B", 0), (S[3], "B", 1)])
        + _seq("TOK", "dirt", [(S[0], N, N), (S[1], N, N), (S[2], N, N), (S[3], N, N)])
    ),
    # the day-before page and the race-morning page of each day share the day's count
    "several_publications_a_day": [
        ("TOK", S[k], "turf", rail, want, utc(S[k].year, S[k].month, S[k].day) + offset)
        for k, (rail, want) in enumerate([("A", N), ("A", N), ("B", 0), ("B", 1)])
        for offset in (timedelta(hours=-18), timedelta(minutes=30))
    ],
    # a day printing no letter is null itself but still a race day for the count
    "unknown_rail_day": _seq("TOK", "turf", [
        (S[0], "A", N), (S[1], "B", 0), (S[2], N, N), (S[3], "B", 2), (S[4], "B", 3),
    ]),
    "unknown_first_day": _seq("TOK", "turf", [
        (S[0], N, N), (S[1], "A", N), (S[2], "A", N), (S[3], "B", 0),
    ]),
}  # fmt: skip


def condition_frame(spec: list[tuple]) -> pd.DataFrame:
    """Rows with every track_condition column: spec rows are (venue, race_date, surface, rail,
    expected[, published_utc])."""
    records = []
    for i, (venue, day, surface, rail, _, *published) in enumerate(spec):
        when = (
            published[0] if published else utc(day.year, day.month, day.day) - timedelta(hours=18)
        )
        records.append(
            {
                "jurisdiction": "jra", "venue": venue, "race_date": day, "surface": surface,
                "going": "GOOD", "rail": rail, "cushion": 9.0 if surface == "turf" else math.nan,
                "moisture_goal": 10.0 + i / 10, "moisture_4c": 9.5, "published_utc": when,
                "available_utc": when, "ingested_utc": when + timedelta(minutes=10),
            }
        )  # fmt: skip
    frame = pd.DataFrame.from_records(records, columns=list(TABLE_COLUMNS))
    for column in UTC_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame.index = [7 * i + 3 for i in range(len(frame))]
    return frame


@pytest.mark.parametrize("case", list(SEQUENCES))
def test_days_since_rail_move_counts_race_days_per_venue_and_surface(case):
    spec = SEQUENCES[case]
    frame = condition_frame(spec)
    expected = {i: row[4] for i, row in zip(frame.index, spec, strict=True)}

    for source in (frame, frame.iloc[::-1], frame.sample(frac=1, random_state=90)):
        got = tc.days_since_rail_move(source)
        assert isinstance(got, pd.Series)
        assert sorted(got.index) == sorted(source.index)
        assert _counts(got) == expected


# ---------------------------------------------------------------- the condition at t

MIDDAY = baba_page(
    date(2026, 11, 28), "5回東京7日", datetime(2026, 11, 28, 12, 40),  # 03:40 UTC
    turf=("SOFT", 8.1, 17.0, 17.6), dirt=("HEAVY", 12.5, 12.9), rail="Aコース",
)  # fmt: skip
DAY_BEFORE_UTC, RACE_DAY_UTC, MIDDAY_UTC = (
    utc(2026, 11, 27, 6, 0), utc(2026, 11, 28, 0, 30), utc(2026, 11, 28, 3, 40)
)  # fmt: skip
SECOND_FETCH = utc(2026, 11, 28, 1, 30)


def history() -> pd.DataFrame:
    """Tokyo's day-before page, its race-morning update (fetched twice) and a mid-day update;
    Kyoto's race-morning page (published 23:00 UTC the day before)."""
    return pd.concat(
        [
            tc.parse_baba_page(fixture("jra_going_day_before.html"),
                               fetched_utc=utc(2026, 11, 27, 7, 12, 5)),
            tc.parse_baba_page(fixture("jra_going_race_day_update.html"),
                               fetched_utc=utc(2026, 11, 28, 1, 0, 41)),
            tc.parse_baba_page(fixture("jra_going_race_day_update.html"),
                               fetched_utc=SECOND_FETCH),
            tc.parse_baba_page(MIDDAY, fetched_utc=utc(2026, 11, 28, 3, 45)),
            tc.parse_baba_page(KYOTO, fetched_utc=utc(2026, 11, 27, 23, 20)),
        ],
        ignore_index=True,
    )  # fmt: skip


TOK_FIRST = PAGES["day_before"][3]
TOK_SECOND = PAGES["race_day_update"][3]
TOK_MIDDAY = [
    expect(*TOK_SAT, "dirt", "HEAVY", None, None, 12.5, 12.9, MIDDAY_UTC),
    expect(*TOK_SAT, "turf", "SOFT", "A", 8.1, 17.0, 17.6, MIDDAY_UTC),
]
KYO_ROWS = PAGES["kyoto_fullwidth_b"][3]

# decision time -> what Kyoto and Tokyo show then
CONDITION_AT = [
    (utc(2026, 11, 26, 12, 0), [], []),
    (DAY_BEFORE_UTC - US, [], []),
    (DAY_BEFORE_UTC, [], TOK_FIRST),  # published at exactly t: usable
    (utc(2026, 11, 27, 23, 0) - US, [], TOK_FIRST),
    (utc(2026, 11, 27, 23, 0), KYO_ROWS, TOK_FIRST),
    (RACE_DAY_UTC - US, KYO_ROWS, TOK_FIRST),
    (RACE_DAY_UTC, KYO_ROWS, TOK_SECOND),
    (MIDDAY_UTC - US, KYO_ROWS, TOK_SECOND),  # the mid-day update is not out yet
    (MIDDAY_UTC, KYO_ROWS, TOK_MIDDAY),
    (utc(2026, 12, 1, 0, 0), KYO_ROWS, TOK_MIDDAY),
]


@pytest.mark.parametrize(
    ("decision", "kyo", "tok"),
    CONDITION_AT,
    ids=[d.strftime("%m%dT%H%M%S.%f") for d, *_ in CONDITION_AT],
)
def test_condition_at_t_uses_only_rows_published_by_t(decision, kyo, tok):
    frame = history()
    for source in (frame, frame.iloc[::-1], frame.sample(frac=1, random_state=90)):
        result = tc.as_of(source, decision)
        assert list(result.columns) == list(TABLE_COLUMNS)
        assert rows(result) == kyo + tok
        assert not result.duplicated(KEY).any()
        available = pd.to_datetime(result["available_utc"], utc=True)
        assert (available <= pd.Timestamp(decision)).all(), "leak: a row published after t"
        if tok == TOK_SECOND:  # one publication fetched twice: one row, the later fetch
            tokyo = result[result["venue"] == "TOK"]
            assert (tokyo["ingested_utc"] == pd.Timestamp(SECOND_FETCH)).all()


def test_yearly_rows_count_only_from_the_documents_publication():
    page = tc.parse_baba_page(
        baba_page(
            T11, "4回東京3日", datetime(2025, 10, 10, 15, 0),  # 06:00 UTC on the 10th
            turf=("GOOD", 8.9, 14.0, 13.5), dirt=("GOOD", 8.0, 7.7), rail="Bコース",
        ),
        fetched_utc=utc(2025, 10, 10, 6, 20),
    )  # fmt: skip
    frame = pd.concat([page, tc.parse_yearly_text(fixture(YEARLY_FIXTURE))], ignore_index=True)
    page_rows = [
        expect("TOK", T11, "dirt", "GOOD", None, None, 8.0, 7.7, utc(2025, 10, 10, 6, 0)),
        expect("TOK", T11, "turf", "GOOD", "B", 8.9, 14.0, 13.5, utc(2025, 10, 10, 6, 0)),
    ]

    for decision, expected in [
        (utc(2025, 10, 10, 6, 0) - US, []),
        (utc(2025, 10, 10, 6, 0), page_rows),
        (YEARLY_PUB - US, page_rows),  # the yearly record is not out yet
        (YEARLY_PUB, YEARLY_ROWS),  # then it is the latest publication of every key
    ]:
        for source in (frame, frame.iloc[::-1]):
            assert rows(tc.as_of(source, decision)) == expected, decision.isoformat()


def test_naive_times_are_refused():
    html = fixture("jra_going_day_before.html")
    with pytest.raises(ValueError):
        tc.parse_baba_page(html, fetched_utc=datetime(2026, 11, 27, 7, 0))
    frame = tc.parse_baba_page(html, fetched_utc=utc(2026, 11, 27, 7, 0))
    with pytest.raises(ValueError):
        tc.as_of(frame, datetime(2026, 11, 28, 2, 0))
