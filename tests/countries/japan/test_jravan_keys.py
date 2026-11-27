"""JV-Data race keys and timestamps to the store's ids and UTC instants (ticket #362).

Contract pinned here, module `horsetrack.countries.japan.jravan.keys`:
- `race_id(year, mmdd, venue_code, race_no)`: `jra:<YYYY-MM-DD>:<VENUE>:<NN>`, the form of
  `jra.netkeiba.common.store_race_id`, the venue's short code from `config/jra.yaml` by number;
  None for a venue outside JRA or an invalid date. `meeting_id` is the same without the race.
- `source_race_key(year, venue_code, kai, nichi, race_no)`: the 12-digit netkeiba race id the jra
  store keeps in `races.source_race_key`; None for a non-JRA venue or a malformed part.
- `jurisdiction(venue_code)`: 'jra' 01-10, 'nar' any other two-digit number, 'overseas' any other
  non-blank code, None for '00' or blank.
- `announced_utc(year, race_mmdd, mmddhhmm)`: the JST minute as an aware UTC datetime, a month after
  the race's month meaning the year before; None for '00000000' or blank.
- `post_time_local(hhmm)`: 'HH:MM'; None for '0000' or blank.

Expected values are written out by hand (JST is UTC+9, no DST).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from horsetrack.core.config import load_source
from horsetrack.countries.japan.jra.netkeiba.common import store_race_id
from horsetrack.countries.japan.jravan import keys

UTC = timezone.utc

VENUES = {  # number -> short code, as config/jra.yaml lists them
    "01": "SAP",
    "02": "HAK",
    "03": "FKS",
    "04": "NII",
    "05": "TOK",
    "06": "NAK",
    "07": "CKY",
    "08": "KYO",
    "09": "HSN",
    "10": "KOK",
}


def test_race_and_meeting_ids_match_the_jra_store() -> None:
    assert keys.race_id("2026", "1128", "08", "11") == "jra:2026-11-28:KYO:11"
    assert keys.meeting_id("2026", "1128", "08") == "jra:2026-11-28:KYO"

    config = {v.number: code for code, v in load_source("japan", "jra").venues.items()}
    assert config == VENUES
    cases = [  # (year, mmdd, venue, race) -> race id
        (("2010", "0105", "06", "01"), "jra:2010-01-05:NAK:01"),
        (("2025", "0601", "05", "12"), "jra:2025-06-01:TOK:12"),
        (("2019", "0728", "01", "07"), "jra:2019-07-28:SAP:07"),
        (("2024", "0229", "09", "10"), "jra:2024-02-29:HSN:10"),
        (("2016", "0213", "10", "03"), "jra:2016-02-13:KOK:03"),
    ]
    for (year, mmdd, venue, race), expected in cases:
        assert keys.race_id(year, mmdd, venue, race) == expected
        assert keys.meeting_id(year, mmdd, venue) == expected.rsplit(":", 1)[0]

    for number, code in VENUES.items():
        for race in ("01", "09", "12"):
            day = date(2021, 3, 7)
            assert keys.race_id("2021", "0307", number, race) == store_race_id(day, code, int(race))
        assert keys.meeting_id("2021", "0307", number) == f"jra:2021-03-07:{code}"


def test_a_venue_outside_jra_or_a_bad_date_has_no_race_id() -> None:
    bad_venues = ["00", "11", "30", "42", "A4", "  ", ""]
    for venue in bad_venues:
        assert keys.race_id("2026", "1128", venue, "11") is None, venue
        assert keys.meeting_id("2026", "1128", venue) is None, venue
    bad_dates = [
        ("2026", "1131"),
        ("2025", "0229"),
        ("2026", "1301"),
        ("2026", "0000"),
        ("2026", "    "),
        ("0000", "1128"),
        ("    ", "1128"),
        ("20a6", "1128"),
        ("2026", "11 8"),
    ]
    for year, mmdd in bad_dates:
        assert keys.race_id(year, mmdd, "08", "11") is None, (year, mmdd)
        assert keys.meeting_id(year, mmdd, "08") is None, (year, mmdd)


def test_source_race_key_is_the_netkeiba_race_id() -> None:
    cases = [
        (("2026", "08", "05", "07", "11"), "202608050711"),
        (("2010", "01", "01", "01", "01"), "201001010101"),
        (("2025", "10", "03", "12", "09"), "202510031209"),
        (("2019", "05", "02", "08", "12"), "201905020812"),
    ]
    for parts, expected in cases:
        assert keys.source_race_key(*parts) == expected, parts
    for parts in [
        ("2026", "30", "05", "07", "11"),
        ("2026", "00", "05", "07", "11"),
        ("2026", "  ", "05", "07", "11"),
        ("2026", "08", "  ", "07", "11"),
        ("2026", "08", "05", "0A", "11"),
        ("2026", "08", "05", "07", "  "),
        ("    ", "08", "05", "07", "11"),
    ]:
        assert keys.source_race_key(*parts) is None, parts


def test_announcement_minutes_are_jst_instants_in_utc() -> None:
    cases = [  # (year, race mmdd, announced mmddhhmm) -> UTC instant
        (("2026", "1128", "11271831"), datetime(2026, 11, 27, 9, 31, tzinfo=UTC)),
        (("2026", "1128", "11280945"), datetime(2026, 11, 28, 0, 45, tzinfo=UTC)),
        (("2026", "0601", "06010830"), datetime(2026, 5, 31, 23, 30, tzinfo=UTC)),
        (("2026", "0104", "12281500"), datetime(2025, 12, 28, 6, 0, tzinfo=UTC)),
        (("2011", "0105", "12301200"), datetime(2010, 12, 30, 3, 0, tzinfo=UTC)),
        (("2026", "0301", "02281200"), datetime(2026, 2, 28, 3, 0, tzinfo=UTC)),
        (("2024", "0302", "03010005"), datetime(2024, 2, 29, 15, 5, tzinfo=UTC)),
    ]
    for args, expected in cases:
        got = keys.announced_utc(*args)
        assert got == expected, (args, got)
        assert got.utcoffset() == timedelta(0), args
    assert keys.announced_utc("2026", "1128", "00000000") is None
    assert keys.announced_utc("2026", "1128", "        ") is None


def test_post_times_are_zero_padded_hh_mm() -> None:
    cases = {"0950": "09:50", "1540": "15:40", "1005": "10:05", "1225": "12:25"}
    for raw, expected in cases.items():
        assert keys.post_time_local(raw) == expected, raw
    for raw in ("0000", "    ", ""):
        assert keys.post_time_local(raw) is None, raw


def test_jurisdiction_by_venue_code() -> None:
    expected = {f"{n:02d}": "jra" for n in range(1, 11)}
    expected.update({f"{n:02d}": "nar" for n in (11, 30, 42, 47, 83, 99)})
    expected.update({"A4": "overseas", "B6": "overseas", "C0": "overseas", "M0": "overseas"})
    expected.update({"00": None, "  ": None, "": None})
    got = {code: keys.jurisdiction(code) for code in expected}
    assert got == expected
