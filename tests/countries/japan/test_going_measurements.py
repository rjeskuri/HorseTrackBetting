"""JRA going measurements as of publication (ticket #136): turf cushion value (クッション値) and
soil moisture (含水率) at the goal (ゴール前) and the fourth corner (4コーナー), by surface.

Contract pinned here (the coder follows it):
- `horsetrack.countries.japan.jra.going` is pure (no network).
- `parse_going_page(body: bytes, *, fetched_utc: datetime) -> pd.DataFrame`: one row per surface
  the page prints, columns (at least) COLUMNS below. The page is cp932 bytes. `venue` is the
  jra.yaml code of the meeting on the date line (2026年11月28日（土曜） 5回東京7日 -> "TOK"),
  `race_date` that line's date, `surface` "turf" / "dirt", `going` the netkeiba-parser codes
  (良 GOOD, 稍重 GOOD TO YIELDING, 重 SOFT, 不良 HEAVY). `cushion` and the moistures are floats
  (percent for moisture); anything not printed or printed as － is NaN, never 0; dirt has no
  cushion. Digits may be full-width. Moisture columns are read by their header, not position.
  `published_utc` is the page's 発表 clock (M月D日 H時MM分, Tokyo, year of the race date) in UTC;
  `available_utc` = `published_utc`; `fetched_utc` is the caller's. All *_utc columns are
  tz-aware UTC. A naive `fetched_utc` raises ValueError.
- `as_of(frame, decision_utc) -> pd.DataFrame`: for each (venue, race_date, surface), the row
  with the latest available_utc <= decision_utc; keys with none are absent (an empty frame when
  nothing is published yet). Independent of row order; one row per key even when a publication
  was fetched twice. A naive `decision_utc` raises ValueError.

Pages are synthetic: three files under tests/fixtures/synthetic/ (stored UTF-8, encoded cp932
here as the site sends them) and `going_page` below, which builds the same structure from
values. Expected values are written out here, never computed by the code under test.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from horsetrack.countries.japan.jra import going

UTC = timezone.utc
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "synthetic"
COLUMNS = (
    "venue",
    "race_date",
    "surface",
    "going",
    "cushion",
    "moisture_goal",
    "moisture_4c",
    "published_utc",
    "available_utc",
    "fetched_utc",
)
UTC_COLUMNS = ("published_utc", "available_utc", "fetched_utc")
NAN = float("nan")
GOING_LABEL = {"GOOD": "良", "GOOD TO YIELDING": "稍重", "SOFT": "重", "HEAVY": "不良"}
_WEEKDAY = "月火水木金土日"

SAT = date(2026, 11, 28)
DAY_BEFORE_UTC = datetime(2026, 11, 27, 6, 0, tzinfo=UTC)  # 11月27日 15時00分 in Tokyo
RACE_DAY_UTC = datetime(2026, 11, 28, 0, 30, tzinfo=UTC)  # 11月28日 9時30分 in Tokyo
FETCH_DAY_BEFORE = datetime(2026, 11, 27, 7, 12, 5, tzinfo=UTC)
FETCH_RACE_DAY = datetime(2026, 11, 28, 1, 0, 41, tzinfo=UTC)


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_text(encoding="utf-8").encode("cp932")


def _day(d: date) -> str:
    return f"{d.month}月{d.day}日（{_WEEKDAY[d.weekday()]}曜）"


def going_page(
    *,
    race_date: date,
    meeting: str,
    published_jst: datetime,
    turf: tuple[str, float | None, float | None, float | None] | None,
    dirt: tuple[str, float | None, float | None] | None,
    corner_first: bool = False,
    cushion_block: bool = True,
) -> bytes:
    """A going page like the fixtures. turf = (going code, cushion, goal %, 4c %), dirt = (going
    code, goal %, 4c %); None for a surface leaves its rows out, None for a value prints －.
    `cushion_block=False` leaves the cushion section out altogether."""

    def cell(v: float | None, suffix: str = "") -> str:
        return "－" if v is None else f"{v:.1f}{suffix}"

    surfaces = []
    if turf is not None:
        surfaces.append(("芝", turf[0], turf[2], turf[3]))
    if dirt is not None:
        surfaces.append(("ダート", dirt[0], dirt[1], dirt[2]))
    going_rows = "".join(f"<tr><th>{s}</th><td>{GOING_LABEL[g]}</td></tr>" for s, g, *_ in surfaces)
    heads = ("4コーナー", "ゴール前") if corner_first else ("ゴール前", "4コーナー")
    moist_rows = ""
    for s, _, goal, corner in surfaces:
        a, b = (corner, goal) if corner_first else (goal, corner)
        moist_rows += f"<tr><th>{s}</th><td>{cell(a, '%')}</td><td>{cell(b, '%')}</td></tr>"
    cushion = ""
    if cushion_block and turf is not None:
        cushion = (
            '<div class="cushion"><h3>クッション値（芝）</h3>'
            f'<p class="value">{cell(turf[1])}</p>'
            '<p class="note">目安：12以上 硬め／10～12 やや硬め／8～10 標準</p></div>'
        )
    p = published_jst
    body = (
        '<div id="baba">'
        f'<h2 class="date_line">{race_date.year}年{_day(race_date)} {meeting}</h2>'
        f'<p class="publish">{_day(p.date())}{p.hour}時{p.minute:02d}分 発表</p>'
        f'<table class="going">{going_rows}</table>'
        f"{cushion}"
        '<table class="moisture"><caption>含水率</caption>'
        f"<tr><th></th><th>{heads[0]}</th><th>{heads[1]}</th></tr>{moist_rows}</table>"
        "</div>"
    )
    head = '<head><meta charset="Shift_JIS"><title>馬場情報</title></head>'
    html = f'<!-- synthetic: JRA going page -->\n<!doctype html><html lang="ja">{head}<body>{body}'
    return (html + "</body></html>").encode("cp932")


def _num(v) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return None
    return round(float(v), 3)


def rows(frame: pd.DataFrame) -> list[dict]:
    """The frame's rows, values normalised for comparison (NaN -> None), sorted by key."""
    out = []
    for r in frame.to_dict("records"):
        out.append(
            {
                "venue": r["venue"],
                "race_date": pd.Timestamp(r["race_date"]).date(),
                "surface": r["surface"],
                "going": None if pd.isna(r["going"]) else r["going"],
                "cushion": _num(r["cushion"]),
                "moisture_goal": _num(r["moisture_goal"]),
                "moisture_4c": _num(r["moisture_4c"]),
                "published_utc": pd.Timestamp(r["published_utc"]).to_pydatetime(),
                "available_utc": pd.Timestamp(r["available_utc"]).to_pydatetime(),
            }
        )
    return sorted(out, key=lambda r: (r["venue"], r["race_date"], r["surface"], r["available_utc"]))


def expect(venue, race_date, surface, going_code, cushion, goal, corner, published) -> dict:
    return {
        "venue": venue,
        "race_date": race_date,
        "surface": surface,
        "going": going_code,
        "cushion": cushion,
        "moisture_goal": goal,
        "moisture_4c": corner,
        "published_utc": published,
        "available_utc": published,
    }


def assert_utc_columns(frame: pd.DataFrame) -> None:
    for col in UTC_COLUMNS:
        assert isinstance(frame[col].dtype, pd.DatetimeTZDtype), col
        assert str(frame[col].dt.tz) == "UTC", col


# ---------------------------------------------------------------- the parser


FIXTURE_CASES = {
    "day_before": (
        "jra_going_day_before.html",
        FETCH_DAY_BEFORE,
        [
            expect("TOK", SAT, "dirt", "GOOD TO YIELDING", None, 5.1, 4.9, DAY_BEFORE_UTC),
            expect("TOK", SAT, "turf", "GOOD", 9.4, 12.3, 11.8, DAY_BEFORE_UTC),
        ],
    ),
    "race_day_update": (
        "jra_going_race_day_update.html",
        FETCH_RACE_DAY,
        [
            expect("TOK", SAT, "dirt", "SOFT", None, 7.9, 8.3, RACE_DAY_UTC),
            expect("TOK", SAT, "turf", "GOOD TO YIELDING", 8.7, 14.6, 15.2, RACE_DAY_UTC),
        ],
    ),
    "no_cushion": (
        "jra_going_no_cushion.html",
        utc(2021, 8, 13, 9, 0, 0),
        [
            # 8月13日 16時15分 in Tokyo; the cushion and the turf 4-corner moisture print －
            expect(
                "SAP", date(2021, 8, 14), "dirt", "GOOD", None, 3.2, 2.8, utc(2021, 8, 13, 7, 15)
            ),
            expect(
                "SAP", date(2021, 8, 14), "turf", "GOOD", None, 13.0, None, utc(2021, 8, 13, 7, 15)
            ),
        ],
    ),
}


@pytest.mark.parametrize("case", list(FIXTURE_CASES))
def test_parser_reads_synthetic_going_pages(case):
    name, fetched, expected = FIXTURE_CASES[case]
    frame = going.parse_going_page(fixture(name), fetched_utc=fetched)

    assert set(COLUMNS) <= set(frame.columns)
    assert rows(frame) == expected
    assert_utc_columns(frame)
    # available_utc is the page's publication time, not the fetch time
    assert (frame["available_utc"] == frame["published_utc"]).all()
    assert (frame["fetched_utc"] == pd.Timestamp(fetched)).all()


# Built pages over venues, years, clocks either side of midnight UTC, all four goings, column
# order and a cushion section left out: (page kwargs, expected rows).
NAK, HSN, KOK = ("NAK", date(2024, 3, 31)), ("HSN", date(2025, 12, 27)), ("KOK", date(2026, 2, 1))
BUILT_CASES = [
    (
        dict(
            race_date=date(2026, 11, 28),
            meeting="5回京都7日",
            published_jst=datetime(2026, 11, 28, 8, 0),  # 23:00 UTC the day before
            turf=("GOOD", 10.6, 9.8, 10.4),
            dirt=("GOOD", 3.5, 3.9),
        ),
        [
            expect("KYO", SAT, "dirt", "GOOD", None, 3.5, 3.9, utc(2026, 11, 27, 23, 0)),
            expect("KYO", SAT, "turf", "GOOD", 10.6, 9.8, 10.4, utc(2026, 11, 27, 23, 0)),
        ],
    ),
    (
        dict(
            race_date=date(2024, 3, 31),
            meeting="3回中山4日",
            published_jst=datetime(2024, 3, 30, 18, 45),
            turf=("HEAVY", 7.1, 18.2, 17.5),
            dirt=("HEAVY", 16.0, 15.4),
            corner_first=True,
        ),
        [
            expect(*NAK, "dirt", "HEAVY", None, 16.0, 15.4, utc(2024, 3, 30, 9, 45)),
            expect(*NAK, "turf", "HEAVY", 7.1, 18.2, 17.5, utc(2024, 3, 30, 9, 45)),
        ],
    ),
    (
        dict(
            race_date=date(2025, 12, 27),
            meeting="5回阪神7日",
            published_jst=datetime(2025, 12, 27, 9, 5),
            turf=("SOFT", None, 15.0, 14.1),
            dirt=("SOFT", 11.2, 10.7),
            cushion_block=False,
        ),
        [
            expect(*HSN, "dirt", "SOFT", None, 11.2, 10.7, utc(2025, 12, 27, 0, 5)),
            expect(*HSN, "turf", "SOFT", None, 15.0, 14.1, utc(2025, 12, 27, 0, 5)),
        ],
    ),
    (
        dict(
            race_date=date(2026, 2, 1),
            meeting="1回小倉4日",
            published_jst=datetime(2026, 1, 31, 16, 40),
            turf=("GOOD TO YIELDING", 12.4, 8.0, 8.6),
            dirt=("GOOD", None, 2.2),
        ),
        [
            expect(*KOK, "dirt", "GOOD", None, None, 2.2, utc(2026, 1, 31, 7, 40)),
            expect(*KOK, "turf", "GOOD TO YIELDING", 12.4, 8.0, 8.6, utc(2026, 1, 31, 7, 40)),
        ],
    ),
]


@pytest.mark.parametrize(
    "kwargs, expected", BUILT_CASES, ids=[kw["meeting"] for kw, _ in BUILT_CASES]
)
def test_parser_over_varied_pages(kwargs, expected):
    fetched = utc(2026, 11, 30, 12, 0)  # a backfill long after publication
    frame = going.parse_going_page(going_page(**kwargs), fetched_utc=fetched)

    assert rows(frame) == expected
    assert_utc_columns(frame)
    assert (frame["fetched_utc"] == pd.Timestamp(fetched)).all()


def test_missing_values_are_null_never_zero():
    frame = going.parse_going_page(
        fixture("jra_going_no_cushion.html"), fetched_utc=utc(2021, 8, 13, 9)
    )
    turf = frame[frame["surface"] == "turf"].iloc[0]
    dirt = frame[frame["surface"] == "dirt"].iloc[0]
    assert pd.isna(turf["cushion"]) and pd.isna(dirt["cushion"])
    assert pd.isna(turf["moisture_4c"])
    assert not (frame[["cushion", "moisture_goal", "moisture_4c"]] == 0).any().any()


def test_page_printing_one_surface_gives_one_row():
    body = going_page(
        race_date=SAT,
        meeting="5回東京7日",
        published_jst=datetime(2026, 11, 28, 11, 0),
        turf=("GOOD TO YIELDING", 8.2, 15.5, 15.9),
        dirt=None,
    )
    frame = going.parse_going_page(body, fetched_utc=utc(2026, 11, 28, 2, 5))
    assert rows(frame) == [
        expect("TOK", SAT, "turf", "GOOD TO YIELDING", 8.2, 15.5, 15.9, utc(2026, 11, 28, 2, 0))
    ]


def test_parser_rejects_naive_fetched_time():
    with pytest.raises(ValueError):
        going.parse_going_page(
            fixture("jra_going_day_before.html"), fetched_utc=datetime(2026, 11, 27, 7)
        )


# ---------------------------------------------------------------- as of publication


def history() -> pd.DataFrame:
    """Tokyo's day-before page and its race-day republication, plus Kyoto's race-day page
    (published 11月28日 8時00分 = 23:00 UTC on the 27th), stored as separate rows."""
    kyoto = going_page(**BUILT_CASES[0][0])
    return pd.concat(
        [
            going.parse_going_page(
                fixture("jra_going_day_before.html"), fetched_utc=FETCH_DAY_BEFORE
            ),
            going.parse_going_page(
                fixture("jra_going_race_day_update.html"), fetched_utc=FETCH_RACE_DAY
            ),
            going.parse_going_page(kyoto, fetched_utc=utc(2026, 11, 27, 23, 20)),
        ],
        ignore_index=True,
    )


def test_race_day_update_is_kept_beside_the_day_before_rows():
    frame = history()
    tok_turf = frame[(frame["venue"] == "TOK") & (frame["surface"] == "turf")]
    assert sorted(zip(tok_turf["available_utc"], tok_turf["cushion"], strict=True)) == [
        (pd.Timestamp(DAY_BEFORE_UTC), 9.4),
        (pd.Timestamp(RACE_DAY_UTC), 8.7),
    ]


TOK_FIRST = {
    "turf": ("GOOD", 9.4, 12.3, 11.8, DAY_BEFORE_UTC),
    "dirt": ("GOOD TO YIELDING", None, 5.1, 4.9, DAY_BEFORE_UTC),
}
TOK_SECOND = {
    "turf": ("GOOD TO YIELDING", 8.7, 14.6, 15.2, RACE_DAY_UTC),
    "dirt": ("SOFT", None, 7.9, 8.3, RACE_DAY_UTC),
}
KYO = {
    "turf": ("GOOD", 10.6, 9.8, 10.4, utc(2026, 11, 27, 23, 0)),
    "dirt": ("GOOD", None, 3.5, 3.9, utc(2026, 11, 27, 23, 0)),
}

# decision time -> what each venue shows then (None: nothing published yet)
AS_OF_CASES = [
    (utc(2026, 11, 26, 12, 0), None, None),
    (utc(2026, 11, 27, 5, 59, 59), None, None),  # a second before the first publication
    (DAY_BEFORE_UTC, TOK_FIRST, None),  # published at exactly t: usable
    (utc(2026, 11, 27, 22, 59, 59), TOK_FIRST, None),
    (utc(2026, 11, 27, 23, 0), TOK_FIRST, KYO),
    (utc(2026, 11, 28, 0, 29, 59), TOK_FIRST, KYO),  # between the two Tokyo publications
    (RACE_DAY_UTC, TOK_SECOND, KYO),
    (utc(2026, 11, 28, 6, 40), TOK_SECOND, KYO),  # after both
]


def _expected(venue: str, shown: dict | None) -> list[dict]:
    if shown is None:
        return []
    return [
        expect(venue, SAT, surface, g, cushion, goal, corner, published)
        for surface, (g, cushion, goal, corner, published) in sorted(shown.items())
    ]


@pytest.mark.parametrize(
    "decision, tok, kyo", AS_OF_CASES, ids=[d.strftime("%m%dT%H%M%S") for d, *_ in AS_OF_CASES]
)
def test_as_of_uses_only_values_published_by_decision_time(decision, tok, kyo):
    frame = history()
    shuffled = frame.sample(frac=1, random_state=136)
    for source in (frame, frame.iloc[::-1], shuffled):
        result = going.as_of(source, decision)
        assert set(COLUMNS) <= set(result.columns)
        assert rows(result) == _expected("KYO", kyo) + _expected("TOK", tok)
        published = pd.to_datetime(result["available_utc"], utc=True)
        assert (published <= pd.Timestamp(decision)).all(), "leak: a value from after t"


def test_as_of_takes_each_surface_from_its_own_latest_publication():
    turf_only = going_page(
        race_date=SAT,
        meeting="5回東京7日",
        published_jst=datetime(2026, 11, 28, 11, 0),  # 02:00 UTC, after the 9時30分 page
        turf=("GOOD TO YIELDING", 8.2, 15.5, 15.9),
        dirt=None,
    )
    frame = pd.concat(
        [history(), going.parse_going_page(turf_only, fetched_utc=utc(2026, 11, 28, 2, 5))],
        ignore_index=True,
    )
    result = going.as_of(frame, utc(2026, 11, 28, 3, 0))
    tok = [r for r in rows(result) if r["venue"] == "TOK"]
    assert tok == [
        expect("TOK", SAT, "dirt", "SOFT", None, 7.9, 8.3, RACE_DAY_UTC),
        expect("TOK", SAT, "turf", "GOOD TO YIELDING", 8.2, 15.5, 15.9, utc(2026, 11, 28, 2, 0)),
    ]


def test_as_of_gives_one_row_per_key_when_a_page_was_fetched_twice():
    body = fixture("jra_going_race_day_update.html")
    frame = pd.concat(
        [
            history(),
            going.parse_going_page(body, fetched_utc=utc(2026, 11, 28, 1, 30)),
        ],
        ignore_index=True,
    )
    result = going.as_of(frame, utc(2026, 11, 28, 2, 0))
    assert rows(result) == _expected("KYO", KYO) + _expected("TOK", TOK_SECOND)


def test_as_of_rejects_naive_decision_time():
    with pytest.raises(ValueError):
        going.as_of(history(), datetime(2026, 11, 28, 2, 0))
