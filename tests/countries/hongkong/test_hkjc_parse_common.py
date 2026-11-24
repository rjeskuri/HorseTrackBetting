"""Shared HKJC parsing helpers (ticket #60, ported from HKTrack's `sources/hkjc/parse_common.py`):
the text-to-value conversions every HKJC parser relies on."""

from __future__ import annotations

import pytest

# isort: split
# (its own block: ruff ranks a module that does not exist yet as third-party)
from horsetrack.countries.hongkong.hkjc import parse_common as pc


@pytest.mark.parametrize(
    ("func", "text", "expected"),
    [
        ("clean", "  Sha\xa0Tin \n  R1 ", "Sha Tin R1"),
        ("clean", None, ""),
        ("to_int", "1,650M", 1650),
        ("to_int", "-3", -3),
        ("to_int", "12.9", 12),
        ("to_int", "---", None),
        ("to_float", "HK$ 1,234.5", 1234.5),
        ("to_float", "", None),
        ("time_to_seconds", "1:09.85", 69.85),
        ("time_to_seconds", "1.09.85", 69.85),
        ("time_to_seconds", "0:57.4", 57.4),
        ("time_to_seconds", "23.61", 23.61),
        ("time_to_seconds", "--", None),
        ("lbw_to_lengths", "-", 0.0),
        ("lbw_to_lengths", "SH", 0.1),
        ("lbw_to_lengths", "n", 0.3),
        ("lbw_to_lengths", "NOSE", 0.05),
        ("lbw_to_lengths", "1-1/4", 1.25),
        ("lbw_to_lengths", "3/4", 0.75),
        ("lbw_to_lengths", "10", 10.0),
        ("lbw_to_lengths", "ZZ", None),
    ],
)
def test_text_converts_to_values(func, text, expected):
    got = getattr(pc, func)(text)
    if isinstance(expected, float):
        assert got == pytest.approx(expected)
    else:
        assert got == expected


def test_query_parameters_and_headers_are_found_whatever_their_case():
    assert pc.href_param("/horse?HorseId=HK_2023_J087&Option=1", "horseid") == "HK_2023_J087"
    assert pc.href_param("fixture?CalYear=2026&CalMonth=09", "calmonth") == "09"
    assert pc.href_param("/horse?Option=1", "horseid") is None
    assert pc.href_param(None, "horseid") is None
    assert pc.header_index(["Pla.", " Horse  No. ", "LBW"]) == {"pla": 0, "horse no": 1, "lbw": 2}
    cell = pc.soup("<table><tr><td> 1\xa0200 <b>M</b></td></tr></table>").find("td")
    assert pc.cell_text(cell) == "1 200 M"
    assert pc.cell_text(None) == ""
