"""JRDB's horse master (UKC) -> the `horse_pedigree` table (ticket #56, ported from HKTrack's
`sources/jra/jrdb/pedigree.py`).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jrdb.pedigree`:
- `INPUT_COLUMNS`: the columns read off a parsed UKC frame (`jrdb_horse_id`, `horse_name`,
  `data_date`, `sire_name`, `dam_name`, `dam_sire_name`, `sire_line_code`, `dam_sire_line_code`,
  `sire_birth_year`, `dam_birth_year`, `dam_sire_birth_year`); missing ones are read as empty.
  `OUTPUT_COLUMNS`: the table's columns, in the order of `EXPECTED_OUTPUT` below.
- `build_horse_pedigree(ukc)`: one row per horse_id. The 8-digit id maps to ours through
  `join.horse_id`; records without a valid id are dropped. When a horse appears in more than one
  record the one with the most recent `data_date` wins (an undated record loses to a dated one).
  The three ancestor names are kept as printed and normalised into `*_name_norm` (NFKC then
  casefold, so full- and half-width forms and letter case key alike; an empty result is None).
  Line codes are carried through as strings; birth years become nullable Int64.
- The paternal grandsire / great-grandsire come from a self-join on the normalised names: a sire
  that is itself a UKC horse contributes its own sire as the grandsire, and once more for the
  great-grandsire. A name whose records disagree on the sire (recycled names) resolves to None,
  never to one of the candidates. `ancestor_collisions(names, sires)` counts such names.
- `merge_pedigree(ukc_table, netkeiba)`: every UKC row, plus a netkeiba row only for a horse UKC
  lacks; the netkeiba names are re-normalised the same way; UKC-only columns are None there.
  `merge_pedigree(table, None)` is the table.
- `coverage(table)`: `rows`, `distinct_horse_ids`, `with_sire_line_code`, `sire_line_code_rate`.

Every horse, ancestor name, id and line code below is invented. Expected normalised keys are
written out by hand.
"""

from __future__ import annotations

import unicodedata
from datetime import date
from typing import Any

import pandas as pd
import pytest

# isort: split
# (its own block: ruff ranks a module that does not exist yet as third-party)
from horsetrack.countries.japan.jrdb import pedigree

EXPECTED_OUTPUT = (
    "horse_id",
    "sire_name", "sire_name_norm",
    "dam_name", "dam_name_norm",
    "dam_sire_name", "dam_sire_name_norm",
    "sire_line_code", "dam_sire_line_code",
    "sire_birth_year", "dam_birth_year", "dam_sire_birth_year",
    "paternal_grandsire_name_norm", "paternal_great_grandsire_name_norm",
)  # fmt: skip

# Invented katakana names, in full-width and in half-width (with separate voicing marks).
MINATO_FULL = "ミナトノカゼ"
MINATO_HALF = "ﾐﾅﾄﾉｶｾﾞ"
SORA_FULL = "ソラノシルベ"
SORA_HALF = "ｿﾗﾉｼﾙﾍﾞ"


def _missing(value: Any) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and value != value)


def ukc_row(jrdb_id: Any, **kw: Any) -> dict[str, Any]:
    """A parsed-UKC-shaped record; only the fields `build_horse_pedigree` reads need to be given."""
    row = dict.fromkeys(pedigree.INPUT_COLUMNS, None)
    row.update(jrdb_horse_id=jrdb_id, data_date=date(2010, 1, 1))
    row.update(kw)
    return row


def frame_of(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(pedigree.INPUT_COLUMNS))


def test_half_width_fixture_names_really_differ() -> None:
    """Guards the fixtures: the half-width spellings are other strings that NFKC folds."""
    for full, half in ((MINATO_FULL, MINATO_HALF), (SORA_FULL, SORA_HALF)):
        assert full != half and unicodedata.normalize("NFKC", half) == full


# ------------------------------------------------------------------------------------------------
# The table, broadly
# ------------------------------------------------------------------------------------------------


def test_build_keys_by_horse_id_and_carries_the_fields() -> None:
    rows = [
        ukc_row(
            "99101712", horse_name="Alpha", sire_name="Sire Alpha", dam_name="Dam Alpha",
            dam_sire_name="Damsire Alpha", sire_line_code="1104", dam_sire_line_code="0102",
            sire_birth_year=1990.0, dam_birth_year=1994.0, dam_sire_birth_year=1985.0,
        ),
        ukc_row(
            "00105651", horse_name="Beta", sire_name="ＳＩＲＥ　ＢＥＴＡ", dam_name="Dam Beta",
            dam_sire_name="DAMSIRE BETA", sire_line_code="1309", dam_sire_line_code=None,
            sire_birth_year=1991.0, dam_birth_year=float("nan"), dam_sire_birth_year=1982.0,
        ),
        ukc_row(
            "24110001", horse_name="Gamma", sire_name=MINATO_HALF, dam_name=None,
            dam_sire_name="　", sire_line_code="0007", dam_sire_line_code="2201",
            sire_birth_year=2015.0, dam_birth_year=2017.0, dam_sire_birth_year=None,
        ),
    ]  # fmt: skip
    table = pedigree.build_horse_pedigree(frame_of(rows))

    assert tuple(pedigree.OUTPUT_COLUMNS) == EXPECTED_OUTPUT
    assert list(table.columns) == list(EXPECTED_OUTPUT)
    assert sorted(table["horse_id"]) == ["1999101712", "2000105651", "2024110001"]
    by_id = table.set_index("horse_id")

    alpha = by_id.loc["1999101712"]
    assert alpha["sire_name"] == "Sire Alpha" and alpha["sire_name_norm"] == "sire alpha"
    assert alpha["dam_name_norm"] == "dam alpha" and alpha["dam_sire_name_norm"] == "damsire alpha"
    assert (alpha["sire_line_code"], alpha["dam_sire_line_code"]) == ("1104", "0102")
    assert [
        int(alpha[c]) for c in ("sire_birth_year", "dam_birth_year", "dam_sire_birth_year")
    ] == [
        1990,
        1994,
        1985,
    ]

    beta = by_id.loc["2000105651"]
    assert beta["sire_name"] == "ＳＩＲＥ　ＢＥＴＡ"  # kept as printed
    assert beta["sire_name_norm"] == "sire beta"
    assert beta["dam_sire_name_norm"] == "damsire beta"
    assert _missing(beta["dam_sire_line_code"]) and pd.isna(beta["dam_birth_year"])

    gamma = by_id.loc["2024110001"]
    assert gamma["sire_name_norm"] == MINATO_FULL
    assert gamma["sire_line_code"] == "0007"  # leading zeros kept
    assert _missing(gamma["dam_name_norm"]) and _missing(gamma["dam_sire_name_norm"])
    assert pd.isna(gamma["dam_sire_birth_year"]) and int(gamma["sire_birth_year"]) == 2015

    for col in ("sire_birth_year", "dam_birth_year", "dam_sire_birth_year"):
        assert str(table[col].dtype) == "Int64", col


def test_build_tolerates_missing_input_columns() -> None:
    """A frame with only the id and the sire still builds; everything else is empty."""
    frame = pd.DataFrame(
        {"jrdb_horse_id": ["13100001", "85100004"], "sire_name": ["Sire One", "Sire Two"]}
    )
    table = pedigree.build_horse_pedigree(frame).set_index("horse_id")
    assert sorted(table.index) == ["1985100004", "2013100001"]
    assert table.loc["2013100001", "sire_name_norm"] == "sire one"
    assert _missing(table.loc["1985100004", "sire_line_code"])


def test_invalid_ids_are_dropped() -> None:
    rows = [
        ukc_row("99101712", sire_name="Sire A", sire_line_code="1104"),
        ukc_row("short", sire_name="Sire B"),
        ukc_row(None, sire_name="Sire C"),
        ukc_row("1310000X", sire_name="Sire D"),
        ukc_row("131000011", sire_name="Sire E"),
        ukc_row("70100003", sire_name="Sire F"),
    ]
    table = pedigree.build_horse_pedigree(frame_of(rows))
    assert sorted(table["horse_id"]) == ["1970100003", "1999101712"]


# ------------------------------------------------------------------------------------------------
# Deduplication and normalisation
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0), (1, 0, 2), (1, 2, 0)])
def test_most_recent_data_date_wins(order: tuple[int, int, int]) -> None:
    """A horse delivered more than once keeps the record with the most recent `data_date`, whatever
    the order the records come in; an undated record loses to a dated one; a horse with only an
    undated record is still kept."""
    versions = [
        ukc_row(
            "13100001", sire_name="Old Sire", sire_line_code="1101", data_date=date(2013, 6, 1)
        ),
        ukc_row(
            "13100001", sire_name="New Sire", sire_line_code="1309", data_date=date(2016, 6, 1)
        ),
        ukc_row(
            "13100001", sire_name="Mid Sire", sire_line_code="1205", data_date=date(2014, 6, 1)
        ),
    ]
    rows = [versions[i] for i in order] + [
        ukc_row("13100002", sire_name="Undated", sire_line_code="1999", data_date=None),
        ukc_row("13100002", sire_name="Dated", sire_line_code="1102", data_date=date(2003, 1, 1)),
        ukc_row("13100003", sire_name="Solo", sire_line_code="1103", data_date=None),
    ]
    table = pedigree.build_horse_pedigree(frame_of(rows))
    assert table["horse_id"].is_unique and len(table) == 3
    by_id = table.set_index("horse_id")
    assert by_id.loc["2013100001", "sire_name"] == "New Sire"
    assert by_id.loc["2013100001", "sire_name_norm"] == "new sire"
    assert by_id.loc["2013100001", "sire_line_code"] == "1309"
    assert by_id.loc["2013100002", "sire_name"] == "Dated"
    assert by_id.loc["2013100003", "sire_name"] == "Solo"


def test_full_width_sire_normalises_to_the_same_key() -> None:
    """A sire printed in half-width katakana, or a latin name in full-width or another case, keys
    the same as its plain form; every ancestor column is normalised the same way."""
    rows = [
        ukc_row(
            "13100001",
            sire_name=MINATO_FULL,
            dam_name=SORA_HALF,
            dam_sire_name="Ｓｕｎｎｙ　Ｂａｙ",
        ),
        ukc_row("13100002", sire_name=MINATO_HALF, dam_name=SORA_FULL, dam_sire_name="Sunny Bay"),
        ukc_row("13100003", sire_name=MINATO_HALF, dam_name=SORA_HALF, dam_sire_name="SUNNY BAY"),
        ukc_row(
            "13100004", sire_name=SORA_HALF, dam_name=MINATO_FULL, dam_sire_name="ｓｕｎｎｙ ｂａｙ"
        ),
    ]
    table = pedigree.build_horse_pedigree(frame_of(rows)).set_index("horse_id")
    first_three = table.loc[["2013100001", "2013100002", "2013100003"]]
    assert set(first_three["sire_name_norm"]) == {MINATO_FULL}
    assert set(first_three["dam_name_norm"]) == {SORA_FULL}
    assert set(table["dam_sire_name_norm"]) == {"sunny bay"}
    assert table.loc["2013100004", "sire_name_norm"] == SORA_FULL
    assert table.loc["2013100004", "dam_name_norm"] == MINATO_FULL
    # the printed names are kept as they came
    assert table.loc["2013100002", "sire_name"] == MINATO_HALF


# ------------------------------------------------------------------------------------------------
# Paternal grand / great-grandsire self-join
# ------------------------------------------------------------------------------------------------


def test_paternal_grandsire_and_great_grandsire_hop() -> None:
    """A male line present in UKC resolves one and two hops up, through normalised names (the
    son prints his sire in half-width, the sire's own record in full-width); a sire absent from
    UKC ends the line."""
    rows = [
        ukc_row("13100001", horse_name="Colt", sire_name=MINATO_HALF),
        ukc_row("05100002", horse_name=MINATO_FULL, sire_name="Grandfather"),
        ukc_row("95100003", horse_name="GRANDFATHER", sire_name="Great"),
        ukc_row(
            "85100004", horse_name="Great", sire_name="Imported Top"
        ),  # "Imported Top" has no UKC row
        ukc_row("14100005", horse_name="Filly", sire_name="Unknown Sire"),
    ]
    table = pedigree.build_horse_pedigree(frame_of(rows)).set_index("horse_id")
    colt = table.loc["2013100001"]
    assert colt["paternal_grandsire_name_norm"] == "grandfather"
    assert colt["paternal_great_grandsire_name_norm"] == "great"
    father = table.loc["2005100002"]
    assert father["paternal_grandsire_name_norm"] == "great"
    assert father["paternal_great_grandsire_name_norm"] == "imported top"
    grandfather = table.loc["1995100003"]
    assert grandfather["paternal_grandsire_name_norm"] == "imported top"
    assert grandfather["paternal_great_grandsire_name_norm"] is None
    great = table.loc["1985100004"]
    assert great["paternal_grandsire_name_norm"] is None
    assert great["paternal_great_grandsire_name_norm"] is None
    filly = table.loc["2014100005"]
    assert filly["paternal_grandsire_name_norm"] is None


def test_ambiguous_name_resolves_to_null() -> None:
    """Two horses share a normalised name but have different sires: the hop through that name is
    None, not either candidate, and the collision is counted. A name shared by records that agree
    on the sire is not ambiguous."""
    rows = [
        ukc_row("13100001", horse_name="Colt", sire_name="Recycled"),
        ukc_row("05100010", horse_name="Recycled", sire_name="Sire X"),
        ukc_row("95100011", horse_name="RECYCLED", sire_name="Sire Y"),  # same name, other sire
        ukc_row("14100002", horse_name="Filly", sire_name="Twice"),
        ukc_row("06100020", horse_name="Twice", sire_name="Sire Z"),
        ukc_row("96100021", horse_name="Ｔｗｉｃｅ", sire_name="Sire Z"),  # same name, same sire
        ukc_row("15100003", horse_name="Gelding", sire_name="Other Recycled"),
        ukc_row("07100030", horse_name="Other Recycled", sire_name="Sire P"),
        ukc_row("97100031", horse_name="Other Recycled", sire_name="Sire Q"),
    ]
    frame = frame_of(rows)
    table = pedigree.build_horse_pedigree(frame).set_index("horse_id")
    assert table.loc["2013100001", "paternal_grandsire_name_norm"] is None
    assert table.loc["2013100001", "paternal_great_grandsire_name_norm"] is None
    assert table.loc["2015100003", "paternal_grandsire_name_norm"] is None
    assert table.loc["2014100002", "paternal_grandsire_name_norm"] == "sire z"

    names = pd.Series([n.casefold() for n in frame["horse_name"]], name="name")
    names = names.replace({"ｔｗｉｃｅ": "twice"})
    sires = pd.Series([s.casefold() for s in frame["sire_name"]], name="sire")
    assert pedigree.ancestor_collisions(names, sires) == 2
    assert pedigree.ancestor_collisions(names[:6], sires[:6]) == 1
    assert pedigree.ancestor_collisions(names[3:6], sires[3:6]) == 0


# ------------------------------------------------------------------------------------------------
# Merging a netkeiba-built table under UKC, and coverage
# ------------------------------------------------------------------------------------------------


def netkeiba_row(horse_id: str, sire: str, dam_sire: str) -> dict[str, Any]:
    return {
        "jurisdiction": "jra",
        "horse_id": horse_id,
        "sire_name": sire,
        "sire_name_norm": sire,  # not normalised: the merge re-normalises it
        "sire_id": None,
        "dam_name": f"Dam of {horse_id}",
        "dam_name_norm": None,
        "dam_id": None,
        "dam_sire_name": dam_sire,
        "dam_sire_name_norm": dam_sire,
        "dam_sire_id": None,
        "source_race_id": None,
        "scraped_utc": pd.Timestamp("2026-11-19", tz="UTC"),
    }


def test_merge_keeps_ukc_primary_and_fills_foreign_gaps() -> None:
    ukc = pedigree.build_horse_pedigree(
        frame_of(
            [
                ukc_row(
                    "13100001", sire_name="Ukc Sire", sire_line_code="1104", sire_birth_year=2005.0
                ),
                ukc_row("14100002", sire_name="Ukc Other", sire_line_code="1201"),
            ]
        )
    )
    netkeiba = pd.DataFrame(
        [
            netkeiba_row("2013100001", "Should Be Ignored", "D1"),  # UKC already covers this horse
            netkeiba_row("2001109999", "Ｆｏｒｅｉｇｎ Sire", "FOREIGN DAMSIRE"),  # only netkeiba
            netkeiba_row("2002108888", MINATO_HALF, "Other Damsire"),
        ]
    )
    merged = pedigree.merge_pedigree(ukc, netkeiba)
    assert list(merged.columns) == list(EXPECTED_OUTPUT)
    assert sorted(merged["horse_id"]) == ["2001109999", "2002108888", "2013100001", "2014100002"]
    by_id = merged.set_index("horse_id")

    assert by_id.loc["2013100001", "sire_name"] == "Ukc Sire"
    assert by_id.loc["2013100001", "sire_line_code"] == "1104"
    assert int(by_id.loc["2013100001", "sire_birth_year"]) == 2005

    foreign = by_id.loc["2001109999"]
    assert foreign["sire_name"] == "Ｆｏｒｅｉｇｎ Sire"
    assert foreign["sire_name_norm"] == "foreign sire"
    assert foreign["dam_sire_name_norm"] == "foreign damsire"
    assert _missing(foreign["sire_line_code"])
    assert pd.isna(foreign["sire_birth_year"])
    assert by_id.loc["2002108888", "sire_name_norm"] == MINATO_FULL


def test_merge_without_an_existing_table_is_the_ukc_table() -> None:
    ukc = pedigree.build_horse_pedigree(
        frame_of([ukc_row("13100001", sire_name="Ukc Sire"), ukc_row("14100002", sire_name="B")])
    )
    for netkeiba in (None, pd.DataFrame()):
        merged = pedigree.merge_pedigree(ukc, netkeiba)
        assert sorted(merged["horse_id"]) == ["2013100001", "2014100002"]


@pytest.mark.parametrize(("coded", "uncoded"), [(1, 1), (3, 1), (0, 2), (2, 0)])
def test_coverage_reports_sire_line_share(coded: int, uncoded: int) -> None:
    rows = [
        ukc_row(f"131000{i:02d}", sire_name=f"S{i}", sire_line_code="1104") for i in range(coded)
    ]
    rows += [ukc_row(f"141000{i:02d}", sire_name=f"T{i}") for i in range(uncoded)]
    table = pedigree.build_horse_pedigree(frame_of(rows))
    cov = pedigree.coverage(table)
    assert cov["rows"] == coded + uncoded
    assert cov["distinct_horse_ids"] == coded + uncoded
    assert cov["with_sire_line_code"] == coded
    assert cov["sire_line_code_rate"] == pytest.approx(coded / (coded + uncoded))
