"""netkeiba five-run card -> `horse_pedigree` (ticket #51, ported from HKTrack's
`sources/jra/parse_pedigree_card.py` and `tests/unit/sources/jra/test_parse_pedigree_card.py`).

Contract pinned here (the coder follows it):
- `horsetrack.countries.japan.jra.netkeiba.parse_pedigree_card.parse(page: RawPage) -> ParsedPage`,
  pure, no network.
- `horse_pedigree`: one row per runner, in page order: every `tr.HorseList` (scratched
  `tr.HorseList.Cancel` included) inside `div.Shutuba_HorseList table.Shutuba_Table` whose
  `.Horse02` links `/horse/<10 letters or digits>`. Legend and training tables, rows without
  a horse link and the winners the past runs link are not runners.
- Row keys: `horse_id`; `sire_name`, `dam_name`, `dam_sire_name` as printed (`.Horse01`,
  `.Horse03`, `.Horse04`), whitespace runs (ideographic and no-break spaces too) collapsed to
  one space and stripped, the dam sire without its brackets (half- or full-width), None when
  nothing is printed; `<column>_norm` = NFKC then casefold of the name (None for None);
  `sire_id`, `dam_id`, `dam_sire_id` None (the card prints ancestors as text);
  `source_race_id` = the store race id `jra:<YYYY-MM-DD>:<VENUE>:<NN>`, the date from the
  page title, venue (jra.yaml venue number) and race number from the 12-digit key, None when
  either is missing; `scraped_utc` = the page's `fetched_utc`.

Pages are synthetic (netkeiba_pages.py); every expected value is written out here.
"""

from __future__ import annotations

import unicodedata
from datetime import date, datetime, timezone

import netkeiba_pages as pages
import pytest
from netkeiba_pages import ABSENT, Runner

from horsetrack.core.ingest.types import RawPage

# isort: split
# (its own block: ruff ranks a module that does not exist yet as third-party)
from horsetrack.countries.japan.jra.netkeiba import parse_pedigree_card

UTC = timezone.utc
FETCHED = datetime(2026, 11, 18, 14, 32, 29, tzinfo=UTC)
NAME_COLUMNS = ("sire_name", "dam_name", "dam_sire_name")
DECOY_IDS = {"2017199903", "2017199904", "2015199901", "2016199902"}


def raw(body: bytes, key: str, fetched: datetime | None = FETCHED) -> RawPage:
    url = f"{pages.RACE}/race/shutuba_past.html?race_id={key}"
    return RawPage("jra", "pedigree_card", key, url, body, fetched_utc=fetched)


def pedigree_rows(body: bytes, key: str, fetched: datetime | None = FETCHED) -> list[dict]:
    return parse_pedigree_card.parse(raw(body, key, fetched))["horse_pedigree"]


def norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def card(runners, **kwargs) -> bytes:
    return pages.pedigree_card(runners, **kwargs)


# Each card: (key, title date, venue printed, race no, fetched, runners, expected rows in page
# order as (horse_id, sire, dam, dam sire), expected source_race_id)
CARDS = {
    "kyoto-2025": (
        "202508040611",
        date(2025, 11, 23),
        "京都",
        11,
        FETCHED,
        [
            Runner("2021104001", "ハヤテノシンセ", "ミナモノユメ", "(カゼノオウジ)"),
            Runner("2020103002", "Testa Bravo", "Lucky Lane", "(Northern Quill)"),
            Runner("2021190003", "Ardent Kite", "ソラノシズク", "（Silver Harbour）"),
            Runner("000a01234b", "Quiet Ember", "Mist Over Glen", "(Far Meridian)"),
            Runner("2019105005", "ツキノカゲ", "ハナノミチ", "(ユメノツヅキ)", scratched=True),
            Runner("2022100006", "  ハヤテノシンセ\n", "　アサギリ ", "( ヤマノヒカリ )"),
            Runner("2021100007", "Blue  Heron", "Kana  Bell", "(Old  Lantern)"),
        ],
        [
            ("2021104001", "ハヤテノシンセ", "ミナモノユメ", "カゼノオウジ"),
            ("2020103002", "Testa Bravo", "Lucky Lane", "Northern Quill"),
            ("2021190003", "Ardent Kite", "ソラノシズク", "Silver Harbour"),
            ("000a01234b", "Quiet Ember", "Mist Over Glen", "Far Meridian"),
            ("2019105005", "ツキノカゲ", "ハナノミチ", "ユメノツヅキ"),
            ("2022100006", "ハヤテノシンセ", "アサギリ", "ヤマノヒカリ"),
            ("2021100007", "Blue Heron", "Kana Bell", "Old Lantern"),
        ],
        "jra:2025-11-23:KYO:11",
    ),
    "kokura-2010": (
        "201010010112",
        date(2010, 2, 14),
        "小倉",
        12,
        datetime(2026, 11, 18, 14, 33, 1, tzinfo=UTC),
        [
            Runner("2006109930", "ミナミカゼ", "Dance Quietly", "(Kingfisher Bay)"),
            Runner("2005109951", "アオゾラ", "ヒバリ", "（サクラフブキ）", scratched=True),
            Runner("2004109945", "Pale Comet", "ミオⅢ", "(Cobble Ridge)"),
        ],
        [
            ("2006109930", "ミナミカゼ", "Dance Quietly", "Kingfisher Bay"),
            ("2005109951", "アオゾラ", "ヒバリ", "サクラフブキ"),
            ("2004109945", "Pale Comet", "ミオⅢ", "Cobble Ridge"),
        ],
        "jra:2010-02-14:KOK:12",
    ),
    "tokyo-2024": (
        "202405020603",
        date(2024, 5, 12),
        "東京",
        3,
        datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        [Runner("2021105100", "ユメカケル", "ホシアカリ", "(Straße Nord)")],
        [("2021105100", "ユメカケル", "ホシアカリ", "Straße Nord")],
        "jra:2024-05-12:TOK:03",
    ),
    "no-date-in-title": (
        "202509030411",
        None,
        "阪神",
        11,
        None,
        [
            Runner("2020106001", "カワセミ", "Sora Bell", "(Ice Garden)"),
            Runner("2020106002", "Ivy Gate", "モモ", "(ナギサ)"),
        ],
        [
            ("2020106001", "カワセミ", "Sora Bell", "Ice Garden"),
            ("2020106002", "Ivy Gate", "モモ", "ナギサ"),
        ],
        None,
    ),
}


@pytest.mark.parametrize("name", list(CARDS))
def test_page_to_horse_pedigree_rows(name):
    key, race_date, venue, race_no, fetched, runners, expected, source_race_id = CARDS[name]
    runners = [
        Runner(r.horse_id, r.sire, r.dam, r.dam_sire, scratched=r.scratched,
               past_winners=(f"20{i:02d}199{i:03d}", "2015199901"))
        for i, r in enumerate(runners, start=10)
    ]  # fmt: skip
    body = card(runners, race_date=race_date, venue_name=venue, race_no=race_no)
    rows = pedigree_rows(body, key, fetched)

    # one row per runner, keyed by its own horse link, in page order; no legend, training or
    # past-run winner gets a row
    assert [row["horse_id"] for row in rows] == [horse_id for horse_id, *_ in expected]
    assert not {row["horse_id"] for row in rows} & (
        DECOY_IDS | {r.past_winners[0] for r in runners}
    )

    for row, (horse_id, sire, dam, dam_sire) in zip(rows, expected, strict=True):
        assert (row["sire_name"], row["dam_name"], row["dam_sire_name"]) == (sire, dam, dam_sire)
        for column in NAME_COLUMNS:
            assert row[f"{column}_norm"] == norm(row[column]), (horse_id, column)
        assert (row["sire_id"], row["dam_id"], row["dam_sire_id"]) == (None, None, None)
        assert row["source_race_id"] == source_race_id, horse_id
        assert row["scraped_utc"] == fetched, horse_id


@pytest.mark.parametrize(
    "key",
    ["202546112301", "2025Q9a00301", "20250804061", "kyoto-11"],
    ids=["nar-key", "overseas-key", "short-key", "not-an-id"],
)
def test_a_key_that_is_not_a_jra_race_leaves_the_race_id_null(key):
    runners = [Runner("2021104001", "ハヤテノシンセ", "ミナモノユメ", "(カゼノオウジ)")]
    rows = pedigree_rows(card(runners, race_date=date(2025, 11, 23)), key)
    assert [(r["horse_id"], r["dam_sire_name"], r["source_race_id"]) for r in rows] == [
        ("2021104001", "カゼノオウジ", None)
    ]


@pytest.mark.parametrize(
    ("printed", "name", "name_norm"),
    [
        ("Ｐｅｌｌｏｒａ", "Ｐｅｌｌｏｒａ", "pellora"),  # full-width Latin
        ("ＰＥＬＬＯＲＡ", "ＰＥＬＬＯＲＡ", "pellora"),
        ("ﾍﾟﾛｰﾗ", "ﾍﾟﾛｰﾗ", "ペローラ"),  # half-width katakana, the mark composed
        ("ペローラ", "ペローラ", "ペローラ"),
        ("Miller's  Cove", "Miller's Cove", "miller's cove"),  # punctuation kept
        ("ＤＲ．　ＰＥＢＢＬＥ", "ＤＲ． ＰＥＢＢＬＥ", "dr. pebble"),
        ("Straße", "Straße", "strasse"),  # casefold, not lower
        ("ミオⅢ", "ミオⅢ", "ミオiii"),  # the Roman numeral folds to letters
    ],
    ids=[
        "full-width", "full-width-upper", "half-width-kana", "kana", "apostrophe",
        "full-width-period", "sharp-s", "roman-numeral",
    ],
)  # fmt: skip
def test_names_kept_as_printed_and_normalised(printed, name, name_norm):
    fixed = ("Fennel Moth", "fennel moth")
    for column in NAME_COLUMNS:
        values = {"sire_name": "Ｌａｒｋ", "dam_name": fixed[0], "dam_sire_name": "Lamp of Reeds"}
        values[column] = printed
        runners = [
            Runner("2019109917", "ハヤテノシンセ", "ミナモノユメ", "(カゼノオウジ)"),
            Runner(
                "2020199926",
                values["sire_name"],
                values["dam_name"],
                pages.bracketed(values["dam_sire_name"]),
            ),
        ]
        rows = pedigree_rows(card(runners, race_date=date(2025, 11, 23)), "202508040611")
        row = next(r for r in rows if r["horse_id"] == "2020199926")
        assert (row[column], row[f"{column}_norm"]) == (name, name_norm), column

        others = {
            "sire_name": ("Ｌａｒｋ", "lark"),
            "dam_name": fixed,
            "dam_sire_name": ("Lamp of Reeds", "lamp of reeds"),
        }
        for other, (other_name, other_norm) in others.items():
            if other != column:
                assert (row[other], row[f"{other}_norm"]) == (other_name, other_norm), other
        # the other runner is untouched
        first = next(r for r in rows if r["horse_id"] == "2019109917")
        assert (first["sire_name_norm"], first["dam_sire_name"]) == (
            "ハヤテノシンセ",
            "カゼノオウジ",
        )


@pytest.mark.parametrize(
    "printed",
    [None, "", "()", "（）", "(  )", "（　）", " \n ", ABSENT],
    ids=["empty-div", "empty-text", "brackets", "full-width-brackets", "spaced-brackets",
         "ideographic-space", "whitespace", "no-div"],
)  # fmt: skip
def test_missing_dam_sire_is_null(printed):
    runners = [
        Runner("2019109917", "ハヤテノシンセ", "ミナモノユメ", printed),
        Runner("2020103002", "Testa Bravo", "Lucky Lane", "(Northern Quill)"),
        Runner("2021190003", "Ardent Kite", "ソラノシズク", "(Silver Harbour)", scratched=True),
    ]
    rows = pedigree_rows(card(runners, race_date=date(2025, 11, 23)), "202508040611")
    assert [r["horse_id"] for r in rows] == ["2019109917", "2020103002", "2021190003"]

    row = rows[0]
    assert (row["sire_name"], row["dam_name"]) == ("ハヤテノシンセ", "ミナモノユメ")
    assert (row["dam_sire_name"], row["dam_sire_name_norm"], row["dam_sire_id"]) == (
        None,
        None,
        None,
    )
    assert row["source_race_id"] == "jra:2025-11-23:KYO:11"
    assert [(r["dam_sire_name"], r["dam_sire_name_norm"]) for r in rows[1:]] == [
        ("Northern Quill", "northern quill"),
        ("Silver Harbour", "silver harbour"),
    ]


def test_a_missing_sire_or_dam_is_null_too():
    runners = [
        Runner("2019109917", None, "ミナモノユメ", "(カゼノオウジ)"),
        Runner("2020103002", "Testa Bravo", ABSENT, "(Northern Quill)"),
        Runner("2021190003", " ", "", ""),
    ]
    rows = pedigree_rows(card(runners, race_date=date(2025, 11, 23)), "202508040611")
    got = [
        (r["horse_id"], r["sire_name"], r["sire_name_norm"], r["dam_name"], r["dam_name_norm"])
        for r in rows
    ]
    assert got == [
        ("2019109917", None, None, "ミナモノユメ", "ミナモノユメ"),
        ("2020103002", "Testa Bravo", "testa bravo", None, None),
        ("2021190003", None, None, None, None),
    ]
    assert rows[2]["dam_sire_name"] is None


def test_a_page_cut_before_the_runner_table_has_no_rows():
    body = card([Runner("2019109917", "ハヤテノシンセ", "ミナモノユメ", "(カゼノオウジ)")])
    cut = body[: body.index(b'<div class="Shutuba_HorseList">')]
    assert pedigree_rows(cut, "202508040611") == []
    assert pedigree_rows(pages.page("<p>error</p>"), "202508040611") == []
