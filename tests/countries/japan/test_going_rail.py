"""JRA's rail setting (A/B/C/D course, 仮柵 position) per race in the store (ticket #222).

Contract pinned here (the coder follows it), all in `horsetrack.countries.japan.jra.going`:

- Ruling: the going page (馬場情報) is the source. Its `.rail .value` element prints the turf
  course in use ('Aコース'; the fixtures under tests/fixtures/synthetic/ are the structure). A
  rail setting is a turf course's: `parse_going_page` gives the frame a `rail_setting` column
  holding 'A', 'B', 'C' or 'D' on the turf row and None on the dirt row. A page with no such
  element, one printing － or nothing, or anything that is not one of the four letters (after NFKC,
  so 'Ｃコース' is 'C'; a bare letter and surrounding spaces are fine) gives None. Never a guess,
  never a NaN: a missing value is None.
- Ruling: `attach_rail_setting(races, going, *, decision_utc=None) -> DataFrame` fills the races
  table's `rail_setting` per race. `races` is a store-shaped pandas frame (`venue`, `race_date` as
  datetime.date, `surface`, any other columns), `going` the concatenated output of
  `parse_going_page`. A race on surface 'turf' takes the rail setting of its (venue, race_date)
  from the going rows of that venue and day: the latest published (`available_utc`, then
  `fetched_utc`) row whose rail_setting is not null, so a later page that prints － does not erase
  an earlier letter and a later letter replaces an earlier one. Every other race (dirt, jump,
  no page for its day and venue) gets None; nothing is borrowed from another day or venue. The
  result has the races' rows in their order and all their columns unchanged, plus `rail_setting`
  (replacing one that was there). With `decision_utc` (tz-aware, a naive one is a ValueError) only
  rows with `available_utc` at or before it count.
- The races table declares `rail_setting` (kind str, nullable), after the other optional columns,
  and a parser row that carries it lands in the table; it is not an "extra column".

Pages are synthetic; expected values are written out here, never computed by the code under test.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from horsetrack.core.store import schemas
from horsetrack.countries.japan.jra import going

UTC = timezone.utc
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "synthetic"
_WEEKDAY = "月火水木金土日"
FETCHED = datetime(2026, 12, 1, 3, 0, tzinfo=UTC)


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_text(encoding="utf-8").encode("cp932")


def page(
    race_date: date,
    meeting: str,
    published_jst: datetime,
    rail_block: str | None,
    *,
    dirt: bool = True,
    turf: bool = True,
) -> bytes:
    """A going page; `rail_block` is the inner text of `<p class="value">` in the rail block,
    None leaves the whole block out."""

    def day(d: date) -> str:
        return f"{d.month}月{d.day}日（{_WEEKDAY[d.weekday()]}曜）"

    surfaces = [s for s, on in (("芝", turf), ("ダート", dirt)) if on]
    going_rows = "".join(f"<tr><th>{s}</th><td>良</td></tr>" for s in surfaces)
    moist_rows = "".join(f"<tr><th>{s}</th><td>10.0%</td><td>9.0%</td></tr>" for s in surfaces)
    rail = (
        ""
        if rail_block is None
        else f'<div class="rail"><h3>使用コース（芝）</h3><p class="value">{rail_block}</p></div>'
    )
    p = published_jst
    html = (
        '<!-- synthetic: JRA going page -->\n<!doctype html><html lang="ja">'
        '<head><meta charset="Shift_JIS"><title>馬場情報</title></head><body><div id="baba">'
        f'<h2 class="date_line">{race_date.year}年{day(race_date)} {meeting}</h2>'
        f'<p class="publish">{day(p.date())}{p.hour}時{p.minute:02d}分 発表</p>'
        f'<table class="going">{going_rows}</table>{rail}'
        '<div class="cushion"><h3>クッション値（芝）</h3><p class="value">9.0</p></div>'
        '<table class="moisture"><caption>含水率</caption>'
        f"<tr><th></th><th>ゴール前</th><th>4コーナー</th></tr>{moist_rows}</table>"
        "</div></body></html>"
    )
    return html.encode("cp932")


def rails(frame: pd.DataFrame) -> dict[str, str | None]:
    """surface -> rail_setting (None for a missing one), one row per surface expected."""
    assert "rail_setting" in frame.columns, "parse_going_page has no rail_setting column"
    assert frame["surface"].is_unique
    return {r["surface"]: (None if pd.isna(r["rail_setting"]) else r["rail_setting"])
            for r in frame.to_dict("records")}  # fmt: skip


# ---------------------------------------------------------------- the parser


@pytest.mark.parametrize(
    ("name", "venue", "race_date", "published", "turf_rail"),
    [
        ("jra_going_rail_a.html", "TOK", date(2026, 11, 28), utc(2026, 11, 27, 6, 0), "A"),
        ("jra_going_rail_c_fullwidth.html", "TOK", date(2026, 11, 29), utc(2026, 11, 28, 7, 30), "C"),  # noqa: E501
        ("jra_going_rail_dash.html", "NAK", date(2026, 12, 5), utc(2026, 12, 4, 6, 0), None),
        # the three older fixtures print no rail block at all
        ("jra_going_day_before.html", "TOK", date(2026, 11, 28), utc(2026, 11, 27, 6, 0), None),
        ("jra_going_race_day_update.html", "TOK", date(2026, 11, 28), utc(2026, 11, 28, 0, 30), None),  # noqa: E501
        ("jra_going_no_cushion.html", "SAP", date(2021, 8, 14), utc(2021, 8, 13, 7, 15), None),
    ],
)  # fmt: skip
def test_parser_reads_the_rail_setting_from_synthetic_pages(
    name, venue, race_date, published, turf_rail
):
    frame = going.parse_going_page(fixture(name), fetched_utc=FETCHED)
    assert rails(frame) == {"dirt": None, "turf": turf_rail}
    # the rail rides on the same rows as before: venue, day and publication are unchanged
    assert set(frame["venue"]) == {venue}
    assert {pd.Timestamp(d).date() for d in frame["race_date"]} == {race_date}
    assert set(pd.to_datetime(frame["published_utc"], utc=True)) == {pd.Timestamp(published)}


@pytest.mark.parametrize(
    ("printed", "expected"),
    [
        ("Aコース", "A"),
        ("Bコース", "B"),
        ("Cコース", "C"),
        ("Dコース", "D"),
        ("Ａコース", "A"),  # full-width letters
        ("Ｄコース", "D"),
        ("B", "B"),  # a bare letter
        ("  Cコース ", "C"),
        ("Cコース使用", "C"),
        ("－", None),
        ("-", None),
        ("", None),
        ("Eコース", None),  # not one of the four letters: never guessed
        ("内柵", None),
        (None, None),  # the block is left out of the page
    ],
)
def test_parser_maps_each_printed_form_to_a_letter_or_null(printed, expected):
    body = page(date(2026, 11, 28), "5回東京7日", datetime(2026, 11, 27, 15, 0), printed)
    assert rails(going.parse_going_page(body, fetched_utc=FETCHED)) == {
        "dirt": None,
        "turf": expected,
    }


@pytest.mark.parametrize("letter", ["A", "B", "C", "D"])
def test_dirt_never_carries_a_rail_setting(letter):
    for turf, dirt in ((True, True), (False, True), (True, False)):
        body = page(
            date(2026, 11, 28), "5回東京7日", datetime(2026, 11, 27, 15, 0), f"{letter}コース",
            turf=turf, dirt=dirt,
        )  # fmt: skip
        got = rails(going.parse_going_page(body, fetched_utc=FETCHED))
        assert got.get("dirt") is None
        assert got.get("turf") == (letter if turf else None)


def test_the_rail_setting_does_not_disturb_the_other_columns():
    with_rail = going.parse_going_page(fixture("jra_going_rail_a.html"), fetched_utc=FETCHED)
    turf = with_rail[with_rail["surface"] == "turf"].iloc[0]
    dirt = with_rail[with_rail["surface"] == "dirt"].iloc[0]
    assert (turf["going"], turf["cushion"], turf["moisture_goal"], turf["moisture_4c"]) == (
        "GOOD", 9.0, 12.0, 11.0,
    )  # fmt: skip
    assert (dirt["going"], dirt["moisture_goal"], dirt["moisture_4c"]) == ("GOOD", 5.0, 4.0)
    assert pd.isna(dirt["cushion"])


# ---------------------------------------------------------------- the join onto races


def going_frame(*pages: tuple[bytes, datetime]) -> pd.DataFrame:
    return pd.concat(
        [going.parse_going_page(body, fetched_utc=fetched) for body, fetched in pages],
        ignore_index=True,
    )


SAT, SUN = date(2026, 11, 28), date(2026, 11, 29)
PUB_FRI = datetime(2026, 11, 27, 15, 0)  # JST clocks, passed to page()
PUB_SAT = datetime(2026, 11, 28, 8, 0)
FETCH_FRI = utc(2026, 11, 27, 7, 0)
FETCH_SAT = utc(2026, 11, 27, 23, 30)


def races_frame(rows: list[tuple[date, str, int, str]]) -> pd.DataFrame:
    """Store-shaped races: (day, venue, race_no, surface) with an unrelated payload column."""
    return pd.DataFrame(
        [
            {
                "race_id": f"jra:{d.isoformat()}:{v}:{n:02d}",
                "race_date": d,
                "venue": v,
                "race_no": n,
                "surface": s,
                "distance_m": 1000 + 200 * n,
                "course_config": "outer",
            }
            for d, v, n, s in rows
        ]
    )


def test_each_race_gets_its_venue_days_turf_rail_and_others_get_null():
    g = going_frame(
        (page(SAT, "5回東京7日", PUB_FRI, "Aコース"), FETCH_FRI),
        (page(SUN, "5回東京8日", datetime(2026, 11, 28, 16, 30), "Ｃコース"), FETCH_SAT),
        (page(SAT, "5回京都7日", PUB_FRI, "Dコース"), FETCH_FRI),
        (page(date(2026, 12, 5), "6回中山1日", datetime(2026, 12, 4, 15, 0), "－"), FETCH_SAT),
    )
    races = races_frame([
        (SAT, "TOK", 1, "turf"), (SAT, "TOK", 2, "dirt"), (SAT, "TOK", 3, "turf"),
        (SUN, "TOK", 1, "turf"), (SUN, "TOK", 2, "dirt"),
        (SAT, "KYO", 1, "turf"), (SAT, "KYO", 2, "dirt"),
        (date(2026, 12, 5), "NAK", 1, "turf"),  # the page prints －
        (date(2026, 12, 6), "NAK", 1, "turf"),  # no page for that day
        (SAT, "HSN", 1, "turf"),  # no page for that venue
        (SAT, "TOK", 4, "jump"),
    ])  # fmt: skip
    out = going.attach_rail_setting(races, g)

    assert list(out["race_id"]) == list(races["race_id"])  # rows and order are kept
    assert list(out.drop(columns="rail_setting").columns) == list(races.columns)
    pd.testing.assert_frame_equal(out.drop(columns="rail_setting"), races)
    got = [None if pd.isna(v) else v for v in out["rail_setting"]]
    assert got == ["A", None, "A", "C", None, "D", None, None, None, None, None]
    assert "rail_setting" not in races.columns  # the input is not modified


def test_the_latest_published_letter_wins_and_a_later_dash_does_not_erase_it():
    races = races_frame([(SAT, "TOK", 1, "turf")])

    def rail_for(*pages) -> str | None:
        v = going.attach_rail_setting(races, going_frame(*pages)).loc[0, "rail_setting"]
        return None if pd.isna(v) else v

    fri_a = (page(SAT, "5回東京7日", PUB_FRI, "Aコース"), FETCH_FRI)
    sat_b = (page(SAT, "5回東京7日", PUB_SAT, "Bコース"), FETCH_SAT)
    sat_dash = (page(SAT, "5回東京7日", PUB_SAT, "－"), FETCH_SAT)
    sat_none = (page(SAT, "5回東京7日", PUB_SAT, None), FETCH_SAT)
    assert rail_for(fri_a) == "A"
    assert rail_for(fri_a, sat_b) == "B"
    assert rail_for(sat_b, fri_a) == "B"  # frame order is irrelevant
    assert rail_for(fri_a, sat_dash) == "A"
    assert rail_for(sat_dash, fri_a) == "A"
    assert rail_for(fri_a, sat_none) == "A"
    assert rail_for(sat_dash) is None


def test_decision_time_hides_pages_published_after_it():
    races = races_frame([(SAT, "TOK", 1, "turf")])
    g = going_frame(
        (page(SAT, "5回東京7日", PUB_FRI, "Aコース"), FETCH_FRI),  # 06:00 UTC on the 27th
        (page(SAT, "5回東京7日", PUB_SAT, "Bコース"), FETCH_SAT),  # 23:00 UTC on the 27th
    )

    def at(when: datetime | None) -> str | None:
        v = going.attach_rail_setting(races, g, decision_utc=when).loc[0, "rail_setting"]
        return None if pd.isna(v) else v

    assert at(None) == "B"
    assert at(utc(2026, 11, 27, 5, 59)) is None
    assert at(utc(2026, 11, 27, 6, 0)) == "A"
    assert at(utc(2026, 11, 27, 22, 59)) == "A"
    assert at(utc(2026, 11, 27, 23, 0)) == "B"
    with pytest.raises(ValueError):
        going.attach_rail_setting(races, g, decision_utc=datetime(2026, 11, 28, 1, 0))


def test_an_existing_rail_setting_column_is_replaced_not_kept():
    races = races_frame([(SAT, "TOK", 1, "turf"), (SAT, "TOK", 2, "dirt")])
    races["rail_setting"] = ["D", "D"]
    g = going_frame((page(SAT, "5回東京7日", PUB_FRI, "Bコース"), FETCH_FRI))
    out = going.attach_rail_setting(races, g)
    assert [None if pd.isna(v) else v for v in out["rail_setting"]] == ["B", None]
    assert list(out.columns).count("rail_setting") == 1


def test_no_going_rows_at_all_gives_a_null_column():
    races = races_frame([(SAT, "TOK", 1, "turf"), (SAT, "TOK", 2, "dirt")])
    empty = going_frame((page(SAT, "5回東京7日", PUB_FRI, "Aコース"), FETCH_FRI)).iloc[0:0]
    out = going.attach_rail_setting(races, empty)
    assert "rail_setting" in out.columns
    assert out["rail_setting"].isna().all()
    assert len(out) == 2


# ---------------------------------------------------------------- the races schema


def test_the_races_table_declares_a_nullable_string_rail_setting():
    spec = schemas.TABLES["races"]
    assert "rail_setting" in spec.columns
    assert spec.columns["rail_setting"] == schemas.STR
    # an optional column: after the required ones, and no other table grows one
    names = list(spec.columns)
    assert names.index("rail_setting") > names.index("course_config")
    for name, other in schemas.TABLES.items():
        if name != "races":
            assert "rail_setting" not in other.columns, name
