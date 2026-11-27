"""JV-Data RA and SE versions by データ区分: which ones the export holds (ticket #401).

Contract pinned here (the coder follows it), module
`horsetrack.countries.japan.jravan.card_versions`:
- `KUBUN_NAMES`: the RA/SE データ区分 values with translated names, exactly the keys `1` (entries,
  出走馬名表), `2` (card, 出馬表), `3`-`6` (flash results, 速報成績), `7` (final result, 成績),
  `9` (abandoned, レース中止), `A` (NAR, 地方競馬), `B` (overseas, 海外国際レース) and `0`
  (deletion, 削除); each name a distinct non-empty string. `PRE_RACE`: the set {"1", "2"}.
- `record_versions(cols, record_type) -> DataFrame`: from `records.decode`d RA or SE arrays, one
  row per record, never deduplicated, with at least `race_id` (`keys.race_id`, null outside
  JRA), `data_kubun` (データ区分 text), `created` (データ作成年月日 as a date: the version's
  publication time) and `race_date` (開催年 + 開催月日 as a date, for every venue); SE adds
  `horse_no` (int) and `horse_id` (the top-level 血統登録番号).
- `census(*, raw=None, as_of: date)`: reads `<raw>/stored/<DATASPEC>/RA/*.jvd` and
  `<raw>/stored/<DATASPEC>/SE/*.jvd` for every DATASPEC folder under `stored/` (default raw:
  `paths.raw_dir('jravan')`), never `stored_shift/`, `_incoming/` or `probe/`, and no other record
  type. It returns an object with `name == 'card-versions'` and `to_json() -> str`, so
  `records.write_build_report` writes it to `<reports dir>/jravan-card-versions.json` through
  `write_report`. The JSON (counts, codes, years and the as-of date only):
      {"name": "card-versions", "as_of": "YYYY-MM-DD",
       "record_types": {"RA": {...}, "SE": {...}}}
  and per record type:
  - `records`: the records counted; `holdout_skipped`: the records on a holdout day
    (`core.eval.holdout.is_holdout_day('japan', race_date)`), of which only the date is read: they
    are in no other figure.
  - `by_kubun`: データ区分 -> records (only the values present); `by_year`: race year (開催年,
    as a string) -> データ区分 -> records.
  - `multi_version_keys`: the keys (the layout's key items: the race's, and for SE also 馬番 and
    血統登録番号) holding more than one record; `combos`: for those keys, their データ区分 in
    データ作成年月日 order joined with " > " (`2 > 7`) -> keys.
  - `pre_race_on_run_races`: records of a `PRE_RACE` データ区分 whose race date is before `as_of`.
  - `held`: each データ区分 present -> its `KUBUN_NAMES` name; `final_only`: true when no record of
    データ区分 1-6 is present, and null (inconclusive, never true) when the record type has no
    record to count: no file under any `stored/<DATASPEC>/`, only empty files, or every record
    skipped as on a holdout day or undated.
  - A holdout-day record is read no further than its race date: one cut short (or overlong) after
    its date is only counted in `holdout_skipped`; the same record on a development day still
    raises `records.RecordLengthError` (its record number counting the file), as before.
  - Every record's two-byte record type is checked before its race date is trusted: a record that
    does not start with the layout's record type (a file of another type misfiled under `RA/` or
    `SE/`) raises `records.RecordTypeError` whatever its day (holdout, development or none) and
    width, one too short to hold the type included; `record` counts the file (holdout-day records
    included) and `found` is what the record starts with (the bytes it holds when shorter than
    two). The holdout check comes after it and the width check after that, so a correctly typed
    holdout-day record of the wrong width is still only skipped.
- `horsetrack japan jravan card-versions [--as-of YYYY-MM-DD]` (default: today) runs `census` and
  writes the report; a reports folder inside the repository is refused (non-zero exit). An export
  with no RA or no SE file under any `stored/<DATASPEC>/` fails the command (non-zero exit, a
  message naming the record type, no report calling it final-only); a cut-off development-day
  record exits 1, and so does a file of the other record type in a record type's folder, even one
  whose every record is on a holdout day (a message naming the record type, no report on it).

The builders keep the final version: `races.build_races` and `runners.build_runners` write the same
`races`, `meetings` and `runners` from an export holding card and final versions as from the
final versions alone (the card published first, as in reality).

The RA and SE layouts are the ones `test_jravan_runners` builds on `jravan_workbook` (the
workbook's item names, invented widths and order); every value is made up (D21: the real export
and workbook are licensed and stay under `$HORSETRACK_DATA_DIR`, and no figure from them appears
here). Expected values are written out by hand from the plans below.
"""

from __future__ import annotations

import collections
import json
import math
import re
import shutil
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import env, paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.countries.japan.jravan import races, records, runners, spec

# isort: split
# (its own block: the sibling helper modules, then the module this ticket creates)
import jravan_workbook as wb
import test_jravan_runners as rn

from horsetrack.countries.japan.jravan import card_versions

REPORT = "jravan-card-versions.json"
TOK, NAK, KYO, HSN = "05", "06", "08", "09"
NAR_VENUE, OVERSEAS_VENUE = "44", "A6"  # keys.jurisdiction: nar, overseas
ALL_KUBUN = {"0", "1", "2", "3", "4", "5", "6", "7", "9", "A", "B"}
SECRETS = ("架空記念", "カクウノウマ", "架空騎手", "架空調教師")
RACE_ID = re.compile(r"jra:\d{4}-\d{2}-\d{2}")
LONG_DIGITS = re.compile(r"\d{10}")  # a horse id (10 digits) or a race key (12)

# A card version of a runner, as the day before the race publishes it: no place, time or odds
CARD_RUNNER = dict(
    place=0, abnormal="0", time="0000", odds=0.0, fav=0, corners=(0, 0, 0, 0), last3=0.0,
    last4=0.0, margin="", body=0, sign="", change=0,
)  # fmt: skip
SCRATCHED_RUNNER = dict(
    place=0, abnormal="1", time="0000", odds=0.0, fav=0, corners=(0, 0, 0, 0), last3=0.0,
    last4=0.0, margin="",
)  # fmt: skip
CARD_RACE = dict(turf="0", laps=())


# ------------------------------------------------------------------------------------------------
# Records


def ymd(day: date) -> str:
    return day.strftime("%Y%m%d")


def ra(day: date, venue: str, race_no: int, kubun: str, created: date, **over: Any) -> bytes:
    """One RA record of the runners test's layout for the race (day, venue, race_no)."""
    return rn.ra(
        kubun=kubun, created=ymd(created), year=f"{day.year:04d}", mmdd=day.strftime("%m%d"),
        venue=venue, race=f"{race_no:02d}", **over,
    )  # fmt: skip


def se(
    day: date,
    venue: str,
    race_no: int,
    horse_no: int,
    horse_id: str,
    kubun: str,
    created: date,
    **over: Any,
) -> bytes:
    """One SE record of the runners test's layout for one runner."""
    return rn.se(
        kubun=kubun, created=ymd(created), year=f"{day.year:04d}", mmdd=day.strftime("%m%d"),
        venue=venue, race=f"{race_no:02d}", horse_no=horse_no, horse_id=horse_id, **over,
    )  # fmt: skip


def put(raw: Path, folder: str, record_type: str, name: str, recs: list[bytes]) -> Path:
    """`recs` as `<raw>/<folder>/<record_type>/<name>` (folder e.g. `stored/RACE`)."""
    return wb.write_jvd(raw / folder / record_type / name, recs)


@pytest.fixture(scope="module")
def layouts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, spec.RecordSpec]:
    path = rn.write_workbook(tmp_path_factory.mktemp("jravan-spec") / wb.WORKBOOK_NAME)
    return {rt: spec.record_spec(rt, path) for rt in ("RA", "SE")}


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp data dir with the made-up RA and SE workbook in place."""
    d = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    rn.write_workbook(d / "specs" / "jravan" / wb.WORKBOOK_NAME)
    return d


def raw_of(data: Path) -> Path:
    return data / "raw" / "jravan"


def section(report: Any, record_type: str) -> dict[str, Any]:
    parsed = json.loads(report.to_json())
    return parsed["record_types"][record_type]


def plain(v: Any) -> Any:
    """A frame value as plain Python: None for any null, dates as `date`."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, np.datetime64):
        return pd.Timestamp(v).date()
    if isinstance(v, np.generic):
        return plain(v.item())
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, pd.Timestamp):
        assert v == v.normalize(), v
        return v.date()
    return v


def rows_of(frame: Any, columns: tuple[str, ...]) -> collections.Counter:
    missing = [c for c in columns if c not in frame.columns]
    assert not missing, f"record_versions lacks {missing}"
    return collections.Counter(
        tuple(plain(v) for v in row)
        for row in frame[list(columns)].itertuples(index=False, name=None)
    )


# ------------------------------------------------------------------------------------------------
# The main synthetic export: card and final versions (dates and versions are the plan)

R1 = (date(2025, 5, 25), TOK, 11)  # card 2, then final 7 (written final first)
R2 = (date(2025, 5, 25), TOK, 12)  # entries 1, card 2, final 7 (written 7, 1, 2)
R3 = (date(2025, 6, 1), TOK, 11)  # card 2 only: not run yet on 2025-06-01
R4 = (date(2025, 5, 25), TOK, 1)  # final 7 only
R5 = (date(2025, 5, 24), TOK, 10)  # abandoned 9 only
R6 = (date(2024, 12, 28), NAK, 11)  # card 2, then final 7, in the 2024 file
AS_OF = date(2025, 6, 1)


def main_export(raw: Path) -> None:
    d1, d6 = R1[0], R6[0]
    put(raw, "stored/RACE", "RA", "2025.jvd", [
        ra(*R1, "7", d1 + timedelta(days=1), field=17),
        ra(*R1, "2", d1 - timedelta(days=2), field=18, **CARD_RACE),
        ra(*R2, "7", d1 + timedelta(days=1)),
        ra(*R2, "1", d1 - timedelta(days=4), **CARD_RACE),
        ra(*R2, "2", d1 - timedelta(days=2), **CARD_RACE),
        ra(*R3, "2", R3[0] - timedelta(days=2), **CARD_RACE),
        ra(*R4, "7", d1 + timedelta(days=1)),
        ra(*R5, "9", R5[0]),
    ])  # fmt: skip
    put(raw, "stored/RACE", "RA", "2024.jvd", [
        ra(*R6, "2", d6 - timedelta(days=2), **CARD_RACE),
        ra(*R6, "7", d6 + timedelta(days=1)),
    ])  # fmt: skip
    put(raw, "stored/RACE", "SE", "2025.jvd", [
        se(*R1, 1, "2021100001", "2", d1 - timedelta(days=2), **CARD_RUNNER),
        se(*R1, 1, "2021100001", "7", d1 + timedelta(days=1), place=1),
        se(*R1, 2, "2021100002", "7", d1 + timedelta(days=1), **SCRATCHED_RUNNER),
        se(*R1, 2, "2021100002", "2", d1 - timedelta(days=2), **CARD_RUNNER),
        se(*R1, 3, "2021100003", "7", d1 + timedelta(days=1), place=2),
        se(*R3, 1, "2022100011", "2", R3[0] - timedelta(days=2), **CARD_RUNNER),
        se(*R4, 4, "2021100004", "7", d1 + timedelta(days=1), place=1),
        se(*R4, 4, "2021100004", "0", d1 + timedelta(days=2), place=1),
    ])  # fmt: skip
    put(raw, "stored/RACE", "SE", "2024.jvd", [
        se(*R6, 5, "2020100005", "2", d6 - timedelta(days=2), **CARD_RUNNER),
        se(*R6, 5, "2020100005", "7", d6 + timedelta(days=1), place=1),
    ])  # fmt: skip


MAIN_RA = {
    "records": 10,
    "holdout_skipped": 0,
    "by_kubun": {"1": 1, "2": 4, "7": 4, "9": 1},
    "by_year": {"2024": {"2": 1, "7": 1}, "2025": {"1": 1, "2": 3, "7": 3, "9": 1}},
    "multi_version_keys": 3,
    "combos": {"2 > 7": 2, "1 > 2 > 7": 1},
    "final_only": False,
}
MAIN_SE = {
    "records": 10,
    "holdout_skipped": 0,
    "by_kubun": {"0": 1, "2": 4, "7": 5},
    "by_year": {"2024": {"2": 1, "7": 1}, "2025": {"0": 1, "2": 3, "7": 4}},
    "multi_version_keys": 4,
    "combos": {"2 > 7": 3, "7 > 0": 1},
    "final_only": False,
}
# as_of -> pre-race records of races run before it: (RA, SE)
PRE_RACE_ON_RUN = {
    date(2025, 5, 25): (1, 1),  # only R6's card: R1 and R2 run on the as-of day itself
    AS_OF: (4, 3),  # R1, R2 (two) and R6; R3 runs on the as-of day
    date(2026, 1, 1): (5, 4),  # every card, R3's too
}


# ------------------------------------------------------------------------------------------------
# The constants


def test_the_kubun_codes_and_the_pre_race_set() -> None:
    names = card_versions.KUBUN_NAMES
    assert set(names) == ALL_KUBUN
    assert all(isinstance(n, str) and n.strip() for n in names.values()), names
    assert len(set(names.values())) == len(names), "two codes share a name"
    assert set(card_versions.PRE_RACE) == {"1", "2"}


# ------------------------------------------------------------------------------------------------
# Two versions of a race and of a runner keep both, each with its publication date


def test_record_versions_keeps_every_version_with_its_publication_date(
    layouts: dict[str, spec.RecordSpec],
) -> None:
    d1, d6, nar_day = R1[0], R6[0], date(2025, 5, 26)
    ra_recs = [
        ra(*R1, "7", d1 + timedelta(days=1)),
        ra(*R1, "2", d1 - timedelta(days=2), **CARD_RACE),
        ra(*R2, "1", d1 - timedelta(days=4), **CARD_RACE),
        ra(*R2, "2", d1 - timedelta(days=2), **CARD_RACE),
        ra(*R2, "7", d1 + timedelta(days=1)),
        ra(*R6, "2", d6 - timedelta(days=3), **CARD_RACE),
        ra(*R6, "7", d6 + timedelta(days=2)),
        ra(nar_day, NAR_VENUE, 11, "A", nar_day + timedelta(days=1)),
    ]
    got = card_versions.record_versions(records.decode(b"".join(ra_recs), layouts["RA"]), "RA")
    assert len(got) == len(ra_recs)  # one row per record, never deduplicated
    assert rows_of(got, ("race_id", "data_kubun", "created", "race_date")) == collections.Counter(
        [
            ("jra:2025-05-25:TOK:11", "7", date(2025, 5, 26), d1),
            ("jra:2025-05-25:TOK:11", "2", date(2025, 5, 23), d1),
            ("jra:2025-05-25:TOK:12", "1", date(2025, 5, 21), d1),
            ("jra:2025-05-25:TOK:12", "2", date(2025, 5, 23), d1),
            ("jra:2025-05-25:TOK:12", "7", date(2025, 5, 26), d1),
            ("jra:2024-12-28:NAK:11", "2", date(2024, 12, 25), d6),
            ("jra:2024-12-28:NAK:11", "7", date(2024, 12, 30), d6),
            (None, "A", date(2025, 5, 27), nar_day),  # outside JRA: no race id, still a row
        ]
    )

    over_day = date(2025, 5, 31)
    se_recs = [
        se(*R1, 1, "2021100001", "2", d1 - timedelta(days=2), **CARD_RUNNER),
        se(*R1, 1, "2021100001", "7", d1 + timedelta(days=1), place=1),
        se(*R1, 2, "2021100002", "7", d1 + timedelta(days=1), **SCRATCHED_RUNNER),
        se(*R1, 2, "2021100002", "2", d1 - timedelta(days=2), **CARD_RUNNER),
        se(*R6, 14, "2020100014", "2", d6 - timedelta(days=1), **CARD_RUNNER),
        se(*R6, 14, "2020100014", "7", d6 + timedelta(days=3), place=3),
        se(over_day, OVERSEAS_VENUE, 9, 3, "2019100003", "B", over_day + timedelta(days=2)),
    ]
    got = card_versions.record_versions(records.decode(b"".join(se_recs), layouts["SE"]), "SE")
    assert len(got) == len(se_recs)
    columns = ("race_id", "horse_no", "horse_id", "data_kubun", "created", "race_date")
    assert rows_of(got, columns) == collections.Counter(
        [
            ("jra:2025-05-25:TOK:11", 1, "2021100001", "2", date(2025, 5, 23), d1),
            ("jra:2025-05-25:TOK:11", 1, "2021100001", "7", date(2025, 5, 26), d1),
            ("jra:2025-05-25:TOK:11", 2, "2021100002", "7", date(2025, 5, 26), d1),
            ("jra:2025-05-25:TOK:11", 2, "2021100002", "2", date(2025, 5, 23), d1),
            ("jra:2024-12-28:NAK:11", 14, "2020100014", "2", date(2024, 12, 27), d6),
            ("jra:2024-12-28:NAK:11", 14, "2020100014", "7", date(2024, 12, 31), d6),
            (None, 3, "2019100003", "B", date(2025, 6, 2), over_day),
        ]
    )


# ------------------------------------------------------------------------------------------------
# The census


def test_the_census_counts_versions_per_kubun_year_and_key(data: Path) -> None:
    main_export(raw_of(data))
    report = card_versions.census(raw=raw_of(data), as_of=AS_OF)
    assert report.name == "card-versions"
    parsed = json.loads(report.to_json())
    assert parsed["name"] == "card-versions"
    assert parsed["as_of"] == AS_OF.isoformat()
    assert set(parsed["record_types"]) == {"RA", "SE"}
    for record_type, expected in (("RA", MAIN_RA), ("SE", MAIN_SE)):
        got = section(report, record_type)
        assert {k: got.get(k, "<missing>") for k in expected} == expected, record_type
        assert got["held"] == {k: card_versions.KUBUN_NAMES[k] for k in expected["by_kubun"]}
        assert got["records"] == sum(got["by_kubun"].values())
        assert sum(got["combos"].values()) == got["multi_version_keys"]

    # pre-race versions of races already run, as of three dates (the default raw dir too)
    for as_of, (ra_n, se_n) in PRE_RACE_ON_RUN.items():
        rep = card_versions.census(as_of=as_of)
        got = (
            section(rep, "RA")["pre_race_on_run_races"],
            section(rep, "SE")["pre_race_on_run_races"],
        )
        assert got == (ra_n, se_n), as_of
        assert section(rep, "RA")["by_kubun"] == MAIN_RA["by_kubun"], as_of  # as_of counts all


# (RA データ区分 present, SE データ区分 present): one record each, a race of its own
HOLDINGS = [
    ({"7", "9", "A", "B", "0"}, {"7", "A", "B", "0"}),
    ({"7"}, {"2", "7"}),
    ({"3", "7"}, {"6", "7"}),
    ({"1"}, {"4", "5", "7"}),
    ({"2", "7", "9"}, {"7"}),
    ({"5", "9", "0"}, {"3", "B"}),
]


def holding_records(record_type: str, kubun: set[str], day: date) -> list[bytes]:
    out = []
    for i, k in enumerate(sorted(kubun), start=1):
        venue = NAR_VENUE if k == "A" else OVERSEAS_VENUE if k == "B" else TOK
        created = day - timedelta(days=2) if k in {"1", "2"} else day + timedelta(days=1)
        if record_type == "RA":
            out.append(ra(day, venue, i, k, created))
        else:
            out.append(se(day, venue, i, 1, f"20211000{i:02d}", k, created))
    return out


@pytest.mark.parametrize(("ra_held", "se_held"), HOLDINGS)
def test_the_census_states_which_versions_the_export_holds(
    data: Path, ra_held: set[str], se_held: set[str]
) -> None:
    day = date(2025, 3, 16)
    raw = raw_of(data)
    put(raw, "stored/RACE", "RA", "2025.jvd", holding_records("RA", ra_held, day))
    put(raw, "stored/RACE", "SE", "2025.jvd", holding_records("SE", se_held, day))
    report = card_versions.census(raw=raw, as_of=date(2025, 4, 1))
    for record_type, held in (("RA", ra_held), ("SE", se_held)):
        got = section(report, record_type)
        assert got["by_kubun"] == dict.fromkeys(held, 1), record_type
        assert got["held"] == {k: card_versions.KUBUN_NAMES[k] for k in held}, record_type
        assert got["final_only"] is (not held & set("123456")), (record_type, held)
        assert got["pre_race_on_run_races"] == len(held & {"1", "2"}), record_type
        assert got["multi_version_keys"] == 0 and got["combos"] == {}, record_type


def test_the_census_reads_every_stored_folder_and_nothing_outside_stored(data: Path) -> None:
    raw = raw_of(data)
    d25, d23, d24 = date(2025, 5, 25), date(2023, 4, 9), date(2025, 5, 24)
    nar_day = date(2025, 5, 31)
    # the export: stored/RACE, stored/DIFN (an all.jvd of two years) and a third DATASPEC
    put(raw, "stored/RACE", "RA", "2025.jvd", [ra(d25, TOK, 11, "7", d25 + timedelta(days=1))])
    put(raw, "stored/DIFN", "RA", "all.jvd", [
        ra(d23, HSN, 11, "7", d23 + timedelta(days=1)),
        ra(d24, TOK, 10, "9", d24),
    ])  # fmt: skip
    put(raw, "stored/SNPN", "RA", "2025.jvd", [
        ra(nar_day, NAR_VENUE, 4, "A", nar_day + timedelta(days=1)),
    ])  # fmt: skip
    put(raw, "stored/RACE", "SE", "2025.jvd", [
        se(d25, TOK, 11, 1, "2021100001", "7", d25 + timedelta(days=1), place=1),
    ])  # fmt: skip
    put(raw, "stored/DIFN", "SE", "all.jvd", [
        se(d23, HSN, 11, 3, "2019100003", "7", d23 + timedelta(days=1), place=1),
        se(d23, HSN, 11, 4, "2019100004", "0", d23 + timedelta(days=5), place=2),
    ])  # fmt: skip
    # another record type in a stored folder: not RA or SE, never decoded as one
    put(raw, "stored/DIFN", "UM", "all.jvd", [wb.jvd_record("UM", 40, "架空")])
    # not part of the export: card and flash versions that would change every figure
    for folder in ("stored_shift/RACE", "_incoming/RACE", "probe/RACE"):
        put(raw, folder, "RA", "2025.jvd", [
            ra(d25, TOK, 11, "2", d25 - timedelta(days=2), **CARD_RACE),
            ra(d25, TOK, 12, "1", d25 - timedelta(days=4), **CARD_RACE),
        ])  # fmt: skip
        put(raw, folder, "SE", "2025.jvd", [
            se(d25, TOK, 11, 1, "2021100001", "2", d25 - timedelta(days=2), **CARD_RUNNER),
            se(d25, TOK, 11, 2, "2021100002", "3", d25, place=2),
        ])  # fmt: skip

    report = card_versions.census(raw=raw, as_of=date(2025, 6, 30))
    expected = {
        "RA": {
            "records": 4,
            "by_kubun": {"7": 2, "9": 1, "A": 1},
            "by_year": {"2023": {"7": 1}, "2025": {"7": 1, "9": 1, "A": 1}},
            "multi_version_keys": 0,
            "pre_race_on_run_races": 0,
            "final_only": True,
        },
        "SE": {
            "records": 3,
            "by_kubun": {"7": 2, "0": 1},
            "by_year": {"2023": {"7": 1, "0": 1}, "2025": {"7": 1}},
            "multi_version_keys": 0,
            "pre_race_on_run_races": 0,
            "final_only": True,
        },
    }
    for record_type, want in expected.items():
        got = section(report, record_type)
        assert {k: got.get(k, "<missing>") for k in want} == want, record_type
        assert set(got["held"]) == set(want["by_kubun"]), record_type


def first_day(holdout: bool) -> date:
    """The first day from 2026-11-28 (the holdout's start) that is, or is not, a holdout day."""
    day = date(2026, 11, 28)
    while is_holdout_day("japan", day) is not holdout:
        day += timedelta(days=1)
    return day


def test_a_record_on_a_holdout_day_only_adds_to_the_skipped_total(data: Path) -> None:
    held_out, dev = first_day(True), first_day(False)
    early = date(2026, 11, 22)
    assert not is_holdout_day("japan", early)
    raw = raw_of(data)
    put(raw, "stored/RACE", "RA", "2026.jvd", [
        ra(dev, TOK, 11, "2", dev - timedelta(days=2), **CARD_RACE),
        ra(dev, TOK, 11, "7", dev + timedelta(days=1)),
        ra(held_out, TOK, 11, "2", held_out - timedelta(days=2), **CARD_RACE),
        ra(held_out, TOK, 11, "7", held_out + timedelta(days=1)),
        ra(held_out, TOK, 12, "9", held_out),
        ra(early, TOK, 11, "7", early + timedelta(days=1)),
    ])  # fmt: skip
    put(raw, "stored/RACE", "SE", "2026.jvd", [
        se(dev, TOK, 11, 1, "2023100001", "7", dev + timedelta(days=1), place=1),
        se(held_out, TOK, 11, 1, "2023100002", "2", held_out - timedelta(days=2), **CARD_RUNNER),
        se(held_out, TOK, 11, 1, "2023100002", "7", held_out + timedelta(days=1), place=1),
        se(held_out, TOK, 11, 2, "2023100003", "3", held_out, place=2),
    ])  # fmt: skip

    report = card_versions.census(raw=raw, as_of=held_out + timedelta(days=30))
    ra_by_year: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for year, kubun in ((dev.year, "2"), (dev.year, "7"), (early.year, "7")):
        ra_by_year[str(year)][kubun] += 1
    expected = {
        "RA": {
            "records": 3,
            "holdout_skipped": 3,
            "by_kubun": {"2": 1, "7": 2},
            "by_year": {y: dict(c) for y, c in ra_by_year.items()},
            "multi_version_keys": 1,
            "combos": {"2 > 7": 1},
            "pre_race_on_run_races": 1,  # the development day's card; the holdout day's is unread
            "final_only": False,
        },
        "SE": {
            "records": 1,
            "holdout_skipped": 3,
            "by_kubun": {"7": 1},
            "by_year": {str(dev.year): {"7": 1}},
            "multi_version_keys": 0,
            "combos": {},
            "pre_race_on_run_races": 0,
            "final_only": True,  # the holdout day's card and flash versions are not counted
        },
    }
    for record_type, want in expected.items():
        got = section(report, record_type)
        assert {k: got.get(k, "<missing>") for k in want} == want, record_type
        assert set(got["held"]) == set(want["by_kubun"]), record_type


# ------------------------------------------------------------------------------------------------
# A record type with nothing to count is inconclusive, never final-only

TYPES = ("RA", "SE")
NOT_FINAL = set("123456")
UNDATED = (("2026", "0230"), ("0000", "0000"), ("2026", "1301"))  # (開催年, 開催月日), no valid day
SKIPPED_KUBUN = ("2", "7", "3")  # card, final and flash versions: each would change the verdict


def other_type(record_type: str) -> str:
    return "SE" if record_type == "RA" else "RA"


def row(
    record_type: str, kubun: str, *, year: str, mmdd: str, race_no: int, created: date
) -> bytes:
    """One RA or SE record (for SE: 馬番 1) of race `race_no` at TOK, its race date exactly
    `year` + `mmdd` (which need not be a valid day)."""
    common = dict(
        kubun=kubun, created=ymd(created), year=year, mmdd=mmdd, venue=TOK, race=f"{race_no:02d}"
    )
    if record_type == "RA":
        return rn.ra(**common)
    return rn.se(**common, horse_no=1, horse_id=f"20231000{race_no:02d}")


def dated(record_type: str, kubun: str, day: date, race_no: int) -> bytes:
    """`row` on `day`, published two days before it (a pre-race version) or the day after."""
    created = day - timedelta(days=2) if kubun in {"1", "2"} else day + timedelta(days=1)
    return row(
        record_type, kubun, year=f"{day.year:04d}", mmdd=day.strftime("%m%d"), race_no=race_no,
        created=created,
    )  # fmt: skip


def assert_inconclusive(
    got: dict[str, Any], record_type: str, *, skipped: int = 0, undated: int = 0
) -> None:
    assert "final_only" in got, f"{record_type}: no final_only"
    assert got["final_only"] is None, (record_type, got["final_only"])  # JSON null
    want = {
        "records": 0, "by_kubun": {}, "held": {}, "multi_version_keys": 0, "combos": {},
        "pre_race_on_run_races": 0, "holdout_skipped": skipped, "undated": undated,
    }  # fmt: skip
    assert {k: got.get(k, "<missing>") for k in want} == want, record_type


# How a record type can be absent from the export: no file under any stored/<DATASPEC>/ (or only
# empty ones), whatever lies outside stored/
ABSENT = ("no-folder", "empty-folder", "empty-file", "only-outside-stored")


def leave_out(raw: Path, record_type: str, how: str, day: date) -> None:
    if how == "empty-folder":
        (raw / "stored" / "RACE" / record_type).mkdir(parents=True, exist_ok=True)
    elif how == "empty-file":
        put(raw, "stored/RACE", record_type, "2025.jvd", [])
    elif how == "only-outside-stored":
        for folder in ("stored_shift/RACE", "_incoming/RACE", "probe/RACE"):
            put(raw, folder, record_type, "2025.jvd", holding_records(record_type, {"2", "7"}, day))


# (RA データ区分 held, SE データ区分 held); None: the record type is absent
ABSENT_CASES = [
    ({"7"}, None),
    ({"2", "7"}, None),
    (None, {"7"}),
    (None, {"6", "7"}),
    (None, None),
]


@pytest.mark.parametrize("how", ABSENT)
@pytest.mark.parametrize(("ra_held", "se_held"), ABSENT_CASES)
def test_a_record_type_absent_from_the_export_is_inconclusive(
    data: Path, ra_held: set[str] | None, se_held: set[str] | None, how: str
) -> None:
    day = date(2025, 3, 16)
    raw = raw_of(data)
    put(raw, "stored/RACE", "UM", "all.jvd", [wb.jvd_record("UM", 40, "架空")])  # not RA or SE
    plan = {"RA": ra_held, "SE": se_held}
    for record_type, held in plan.items():
        if held is None:
            leave_out(raw, record_type, how, day)
        else:
            put(
                raw, "stored/RACE", record_type, "2025.jvd", holding_records(record_type, held, day)
            )

    report = card_versions.census(raw=raw, as_of=date(2025, 4, 1))
    for record_type, held in plan.items():
        got = section(report, record_type)
        if held is None:
            assert_inconclusive(got, record_type)
            continue
        # the record type that is there keeps its verdict
        assert got["by_kubun"] == dict.fromkeys(held, 1), record_type
        assert got["final_only"] is (not held & NOT_FINAL), (record_type, held)


# (the record type whose every record is skipped, holdout-day records, undated records, the other
# record type's データ区分 on a development day)
SKIPPED_CASES = [
    ("SE", 3, 0, {"7"}),
    ("SE", 0, 2, {"2", "7"}),
    ("SE", 2, 3, {"7"}),
    ("RA", 3, 0, {"6", "7"}),
    ("RA", 1, 2, {"7"}),
]


@pytest.mark.parametrize(("skipped_type", "n_held", "n_undated", "other_held"), SKIPPED_CASES)
def test_a_record_type_whose_every_record_is_skipped_is_inconclusive(
    data: Path, skipped_type: str, n_held: int, n_undated: int, other_held: set[str]
) -> None:
    held_out, dev = first_day(True), first_day(False)
    raw = raw_of(data)
    other = other_type(skipped_type)
    put(raw, "stored/RACE", other, "2026.jvd", holding_records(other, other_held, dev))
    recs = [
        dated(skipped_type, SKIPPED_KUBUN[i % 3], held_out, 1 + i) for i in range(n_held)
    ] + [
        row(skipped_type, SKIPPED_KUBUN[i % 3], year=y, mmdd=m, race_no=11 + i, created=dev)
        for i, (y, m) in enumerate(UNDATED[:n_undated])
    ]  # fmt: skip
    put(raw, "stored/RACE", skipped_type, "2026.jvd", recs)

    report = card_versions.census(raw=raw, as_of=held_out + timedelta(days=30))
    assert_inconclusive(
        section(report, skipped_type), skipped_type, skipped=n_held, undated=n_undated
    )
    got = section(report, other)
    assert got["by_kubun"] == dict.fromkeys(other_held, 1), other
    assert got["final_only"] is (not other_held & NOT_FINAL), (other, other_held)


# ------------------------------------------------------------------------------------------------
# A holdout-day record is read no further than its race date, whatever its width

CUTS = ("at-the-date", "mid-record", "one-byte-short", "one-byte-long")


def date_end(layout: spec.RecordSpec) -> int:
    """The bytes from the record's start through its race date (開催年 and 開催月日)."""
    by_name = {item.name: item for item in layout.items}
    return max(by_name[n].start - 1 + by_name[n].length for n in (races.YEAR, races.MMDD))


def cut(rec: bytes, layout: spec.RecordSpec, how: str) -> bytes:
    """`rec` cut off after its race date, midway, one byte short, or with a byte too many."""
    body = rec.removesuffix(b"\r\n")
    width = layout.length - spec.SEPARATOR_BYTES
    end = date_end(layout)
    assert len(body) == width and end < width - 1, (len(body), width, end)
    if how == "one-byte-long":
        return body + b"0\r\n"
    keep = {"at-the-date": end, "mid-record": (end + width) // 2, "one-byte-short": width - 1}
    return body[: keep[how]] + b"\r\n"


@pytest.mark.parametrize("how", CUTS)
@pytest.mark.parametrize("record_type", TYPES)
def test_a_cut_off_record_on_a_holdout_day_is_skipped_on_its_date_alone(
    data: Path, layouts: dict[str, spec.RecordSpec], record_type: str, how: str
) -> None:
    held_out, dev = first_day(True), first_day(False)
    layout, other = layouts[record_type], other_type(record_type)
    raw = raw_of(data)
    as_of = held_out + timedelta(days=30)
    put(raw, "stored/RACE", other, "2026.jvd", holding_records(other, {"7"}, dev))
    put(raw, "stored/RACE", record_type, "2026.jvd", [
        dated(record_type, "2", dev, 11),
        cut(dated(record_type, "7", held_out, 11), layout, how),
        dated(record_type, "7", dev, 11),
        dated(record_type, "2", held_out, 12),
    ])  # fmt: skip

    report = card_versions.census(raw=raw, as_of=as_of)
    want = {
        "records": 2,
        "holdout_skipped": 2,
        "undated": 0,
        "by_kubun": {"2": 1, "7": 1},
        "by_year": {str(dev.year): {"2": 1, "7": 1}},
        "multi_version_keys": 1,
        "combos": {"2 > 7": 1},
        "pre_race_on_run_races": 1,
        "final_only": False,
    }
    got = section(report, record_type)
    assert {k: got.get(k, "<missing>") for k in want} == want, (record_type, how)
    assert section(report, other)["final_only"] is True, other

    # the same cut on a development day still stops the census, at its record in its file
    bad = cut(dated(record_type, "7", dev, 10), layout, how)
    put(raw, "stored/RACE", record_type, "cut.jvd", [dated(record_type, "2", held_out, 9), bad])
    with pytest.raises(records.RecordLengthError) as caught:
        card_versions.census(raw=raw, as_of=as_of)
    assert caught.value.record_type == record_type
    assert caught.value.record == 2  # the holdout-day record before it still counts in the file
    assert caught.value.length == len(bad) - spec.SEPARATOR_BYTES


# ------------------------------------------------------------------------------------------------
# A record of another type fails the census before its date is trusted, whatever its day or width

FOREIGN = "UM"  # a record type that is neither RA nor SE


def retyped(rec: bytes, found: str) -> bytes:
    """`rec` with its two-byte record type replaced by `found`, every other byte kept."""
    assert len(found) == 2 and found.isascii() and found not in TYPES, found
    return found.encode("ascii") + rec[2:]


def misfiled(record_type: str, when: str, race_no: int) -> bytes:
    """A final-result record of the other record type (built with its own layout) on a holdout
    day, a development day, or a day that does not exist."""
    other = other_type(record_type)
    if when == "undated":
        return row(other, "7", year="2026", mmdd="0230", race_no=race_no, created=first_day(False))
    return dated(other, "7", first_day(when == "holdout"), race_no)


# (the misfiled record's day, the records of the record type before it in its file)
MISFILED_CASES = [("holdout", 2), ("holdout", 0), ("development", 3), ("undated", 1)]


@pytest.mark.parametrize(("when", "before"), MISFILED_CASES)
@pytest.mark.parametrize("record_type", TYPES)
def test_a_record_of_the_other_type_fails_the_census_whatever_its_day(
    data: Path, record_type: str, when: str, before: int
) -> None:
    held_out, dev = first_day(True), first_day(False)
    other = other_type(record_type)
    raw = raw_of(data)
    put(raw, "stored/RACE", other, "2026.jvd", holding_records(other, {"7"}, dev))
    # an earlier file of the record type: its records are not in the later file's numbers
    put(raw, "stored/RACE", record_type, "2025.jvd", [
        dated(record_type, "7", date(2025, 3, 16), 1),
        dated(record_type, "7", date(2025, 3, 16), 2),
    ])  # fmt: skip
    preamble = [
        dated(record_type, "2", held_out, 11),  # holdout-day records count in the numbers too
        dated(record_type, "7", dev, 11),
        dated(record_type, "7", held_out, 11),
    ][:before]
    put(raw, "stored/RACE", record_type, "2026.jvd", [
        *preamble,
        misfiled(record_type, when, 12),
        dated(record_type, "7", dev, 13),
    ])  # fmt: skip

    with pytest.raises(records.RecordTypeError) as caught:
        card_versions.census(raw=raw, as_of=held_out + timedelta(days=30))
    got = (caught.value.record_type, caught.value.record, caught.value.found)
    assert got == (record_type, before + 1, other), (record_type, when)


# A record not of the record type on a holdout day, its race date intact: the other type's record
# cut after its date, midway or with a byte too many, or one of the layout's own width (or a byte
# short) that starts with another record type
SHAPES = ("other-at-the-date", "other-mid-record", "other-one-byte-long", "foreign",
          "foreign-one-byte-short")  # fmt: skip


def shaped(
    record_type: str, layouts: dict[str, spec.RecordSpec], how: str, day: date, race_no: int
) -> tuple[bytes, str]:
    """The `how` record on `day`, and the record type it starts with."""
    other = other_type(record_type)
    if how.startswith("other-"):
        rec = dated(other, "7", day, race_no)
        return cut(rec, layouts[other], how.removeprefix("other-")), other
    rec = retyped(dated(record_type, "7", day, race_no), FOREIGN)
    if how == "foreign-one-byte-short":
        rec = cut(rec, layouts[record_type], "one-byte-short")
    return rec, FOREIGN


@pytest.mark.parametrize("how", SHAPES)
@pytest.mark.parametrize("record_type", TYPES)
def test_a_record_of_another_type_on_a_holdout_day_is_not_skipped_whatever_its_width(
    data: Path, layouts: dict[str, spec.RecordSpec], record_type: str, how: str
) -> None:
    held_out, dev = first_day(True), first_day(False)
    other = other_type(record_type)
    raw = raw_of(data)
    put(raw, "stored/RACE", other, "2026.jvd", holding_records(other, {"7"}, dev))
    bad, found = shaped(record_type, layouts, how, held_out, 12)
    assert bad[:2] == found.encode("ascii")  # its first bytes intact
    put(raw, "stored/RACE", record_type, "2026.jvd", [
        dated(record_type, "7", dev, 11),
        dated(record_type, "2", held_out, 11),
        bad,
        dated(record_type, "7", held_out, 13),
    ])  # fmt: skip

    with pytest.raises(records.RecordTypeError) as caught:
        card_versions.census(raw=raw, as_of=held_out + timedelta(days=30))
    got = (caught.value.record_type, caught.value.record, caught.value.found)
    assert got == (record_type, 3, found), (record_type, how)


@pytest.mark.parametrize("stub", ("type-initial", "digit"))
@pytest.mark.parametrize("record_type", TYPES)
def test_a_record_too_short_to_hold_its_type_is_a_record_type_error(
    data: Path, record_type: str, stub: str
) -> None:
    held_out, dev = first_day(True), first_day(False)
    other = other_type(record_type)
    raw = raw_of(data)
    put(raw, "stored/RACE", other, "2026.jvd", holding_records(other, {"7"}, dev))
    held = record_type[0] if stub == "type-initial" else "7"  # the one byte the record holds
    put(raw, "stored/RACE", record_type, "2026.jvd", [
        dated(record_type, "2", held_out, 11),
        dated(record_type, "7", dev, 11),
        held.encode("ascii") + b"\r\n",
        dated(record_type, "7", dev, 12),
    ])  # fmt: skip

    with pytest.raises(records.RecordTypeError) as caught:
        card_versions.census(raw=raw, as_of=held_out + timedelta(days=30))
    got = (caught.value.record_type, caught.value.record, caught.value.found)
    assert got == (record_type, 3, held), (record_type, stub)


# ------------------------------------------------------------------------------------------------
# The final-result version still feeds races and runners


def build_export(raw: Path, *, with_cards: bool) -> None:
    """Three races and their runners; `with_cards` adds the card versions, published first."""
    d1, d2 = date(2025, 5, 25), date(2024, 11, 30)
    card, final = d1 - timedelta(days=2), d1 + timedelta(days=1)
    plan = [  # (record type, file year, a card version, the record)
        ("RA", 2025, True, ra(d1, TOK, 11, "2", card, field=18, **CARD_RACE)),
        ("RA", 2025, False, ra(d1, TOK, 11, "7", final, field=17)),
        ("RA", 2025, False, ra(d1, TOK, 1, "7", final, field=10)),
        ("RA", 2024, True, ra(d2, KYO, 5, "2", d2 - timedelta(days=1), field=12, **CARD_RACE)),
        ("RA", 2024, False, ra(d2, KYO, 5, "7", d2 + timedelta(days=2), field=12)),
        ("SE", 2025, True, se(d1, TOK, 11, 1, "2021100001", "2", card, **CARD_RUNNER)),
        ("SE", 2025, False, se(d1, TOK, 11, 1, "2021100001", "7", final, place=1, time="1345")),
        ("SE", 2025, True, se(d1, TOK, 11, 2, "2021100002", "2", card, **CARD_RUNNER)),
        ("SE", 2025, False, se(d1, TOK, 11, 2, "2021100002", "7", final, **SCRATCHED_RUNNER)),
        ("SE", 2025, False,
         se(d1, TOK, 11, 3, "2021100003", "7", final, place=2, time="1347", margin="1")),
        ("SE", 2024, True,
         se(d2, KYO, 5, 7, "2022100007", "2", d2 - timedelta(days=1), **CARD_RUNNER)),
        ("SE", 2024, False,
         se(d2, KYO, 5, 7, "2022100007", "7", d2 + timedelta(days=2), place=1, time="1102")),
    ]  # fmt: skip
    files: dict[tuple[str, int], list[bytes]] = collections.defaultdict(list)
    for record_type, year, is_card, rec in plan:
        if with_cards or not is_card:
            files[(record_type, year)].append(rec)
    for (record_type, year), recs in files.items():
        put(raw, "stored/RACE", record_type, f"{year}.jvd", recs)


def test_the_final_version_still_feeds_races_and_runners(data: Path) -> None:
    outs = {}
    for with_cards in (True, False):
        label = "two-versions" if with_cards else "final-only"
        raw, out = data / "raw" / label, data / "processed" / label
        build_export(raw, with_cards=with_cards)
        races.build_races(raw=raw, out=out)
        runners.build_runners(raw=raw, out=out)
        outs[label] = {t: pq.read_table(out / f"{t}.parquet") for t in ("races", "meetings",
                                                                         "runners")}  # fmt: skip
    for table in ("races", "meetings", "runners"):
        two, final = outs["two-versions"][table], outs["final-only"][table]
        assert two.equals(final), table

    # and they hold the final versions: the field after the scratch, the scratch, the result
    race_rows = {r["race_id"]: r for r in outs["two-versions"]["races"].to_pylist()}
    assert race_rows["jra:2025-05-25:TOK:11"]["field_size"] == 17
    assert race_rows["jra:2025-05-25:TOK:11"]["finish_time_s"] == pytest.approx(94.5)
    assert len(race_rows) == 3
    runner_rows = {
        (r["race_id"], r["horse_no"]): r for r in outs["two-versions"]["runners"].to_pylist()
    }
    assert len(runner_rows) == 4
    assert runner_rows[("jra:2025-05-25:TOK:11", 2)]["is_scratched"] is True
    assert runner_rows[("jra:2025-05-25:TOK:11", 1)]["finish_pos"] == 1
    assert runner_rows[("jra:2024-11-30:KYO:05", 7)]["finish_pos"] == 1


# ------------------------------------------------------------------------------------------------
# The private report and the command


def assert_counts_only(text: str) -> None:
    assert not RACE_ID.search(text), "a race id in the report"
    assert not LONG_DIGITS.search(text), "a horse id or race key in the report"
    for secret in SECRETS:
        assert secret not in text, secret


def invoke(*args: str):
    return CliRunner().invoke(app, ["japan", "jravan", "card-versions", *args])


def test_the_report_goes_through_write_report_and_the_command_writes_it(
    data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_export(raw_of(data))
    report = card_versions.census(as_of=AS_OF)
    expected = json.loads(report.to_json())
    assert_counts_only(report.to_json())

    target = records.write_build_report(report)
    assert target == data / "reports" / REPORT
    assert json.loads(target.read_text(encoding="utf-8")) == expected
    target.unlink()

    result = invoke("--as-of", AS_OF.isoformat())
    assert result.exit_code == 0, result.output
    assert target.is_file()
    text = target.read_text(encoding="utf-8")
    assert json.loads(text) == expected
    assert_counts_only(text)
    assert_counts_only(result.output)

    # the default as-of date is today: every card in the export is of a race already run
    result = invoke()
    assert result.exit_code == 0, result.output
    today = json.loads(target.read_text(encoding="utf-8"))
    assert today["as_of"] >= date(2026, 11, 27).isoformat()
    last = max(PRE_RACE_ON_RUN)
    assert (
        today["record_types"]["RA"]["pre_race_on_run_races"],
        today["record_types"]["SE"]["pre_race_on_run_races"],
    ) == PRE_RACE_ON_RUN[last]

    # a reports folder inside the repository is refused, and nothing is created there
    inside = env.repo_root() / f".tmp-jravan-card-versions-{uuid.uuid4().hex[:8]}" / "reports"
    monkeypatch.setattr(paths, "reports_dir", lambda: inside)
    if hasattr(card_versions, "reports_dir"):  # in case the module imported the name
        monkeypatch.setattr(card_versions, "reports_dir", lambda: inside)
    try:
        refused = invoke("--as-of", AS_OF.isoformat())
        assert refused.exit_code != 0, refused.output
        assert not inside.parent.exists()
    finally:
        shutil.rmtree(inside.parent, ignore_errors=True)


def names(record_type: str, text: str) -> bool:
    """Whether `text` names `record_type` as a word of its own (`RA` in `RACE` does not count)."""
    return re.search(rf"(?<![A-Za-z]){record_type}(?![A-Za-z])", text) is not None


@pytest.mark.parametrize("how", ("no-folder", "empty-folder"))
@pytest.mark.parametrize("missing", (("SE",), ("RA",), ("RA", "SE")))
def test_the_command_fails_on_an_export_without_ra_or_se_files(
    data: Path, missing: tuple[str, ...], how: str
) -> None:
    raw, day, as_of = raw_of(data), date(2025, 3, 16), "2025-04-01"
    target = data / "reports" / REPORT
    for record_type in TYPES:
        put(raw, "stored/RACE", record_type, "2025.jvd", holding_records(record_type, {"7"}, day))
    put(raw, "stored/RACE", "UM", "all.jvd", [wb.jvd_record("UM", 40, "架空")])

    # the whole export: final versions only, and the command says so
    ok = invoke("--as-of", as_of)
    assert ok.exit_code == 0, ok.output
    written = json.loads(target.read_text(encoding="utf-8"))["record_types"]
    assert {t: written[t]["final_only"] for t in TYPES} == {"RA": True, "SE": True}
    target.unlink()

    # a copy missing a record type (a copy outside stored/ is not the export)
    for record_type in missing:
        shutil.rmtree(raw / "stored" / "RACE" / record_type)
        if how == "empty-folder":
            (raw / "stored" / "RACE" / record_type).mkdir()
        put(raw, "stored_shift/RACE", record_type, "2025.jvd",
            holding_records(record_type, {"7"}, day))  # fmt: skip
    result = invoke("--as-of", as_of)
    assert result.exit_code != 0, result.output
    assert isinstance(result.exception, SystemExit), repr(result.exception)  # reported, no crash
    for record_type in missing:
        assert names(record_type, result.output), (record_type, result.output)
    if target.exists():  # a report may be written, never one calling a missing type final-only
        written = json.loads(target.read_text(encoding="utf-8"))["record_types"]
        for record_type in missing:
            assert written.get(record_type, {}).get("final_only") is not True, record_type


@pytest.mark.parametrize("record_type", TYPES)
def test_the_command_skips_a_cut_off_holdout_record_and_stops_at_a_development_one(
    data: Path, layouts: dict[str, spec.RecordSpec], record_type: str
) -> None:
    held_out, dev = first_day(True), first_day(False)
    layout, other = layouts[record_type], other_type(record_type)
    raw, as_of = raw_of(data), (held_out + timedelta(days=30)).isoformat()
    target = data / "reports" / REPORT
    put(raw, "stored/RACE", other, "2026.jvd", holding_records(other, {"7"}, dev))
    put(raw, "stored/RACE", record_type, "2026.jvd", [
        dated(record_type, "7", dev, 11),
        cut(dated(record_type, "2", held_out, 11), layout, "mid-record"),
    ])  # fmt: skip

    ok = invoke("--as-of", as_of)
    assert ok.exit_code == 0, ok.output
    got = json.loads(target.read_text(encoding="utf-8"))["record_types"][record_type]
    assert (got["records"], got["holdout_skipped"], got["final_only"]) == (1, 1, True)
    target.unlink()

    put(raw, "stored/RACE", record_type, "cut.jvd", [
        cut(dated(record_type, "7", dev, 10), layout, "mid-record"),
    ])  # fmt: skip
    failed = invoke("--as-of", as_of)
    assert failed.exit_code == 1, failed.output
    assert isinstance(failed.exception, SystemExit), repr(failed.exception)


@pytest.mark.parametrize("record_type", TYPES)
def test_the_command_fails_on_a_file_of_the_other_type_in_a_record_types_folder(
    data: Path, record_type: str
) -> None:
    """The other type's file copied into the record type's folder, its every record on a holdout
    day: it passes for a file of the record type only until its records' type is read."""
    held_out, dev = first_day(True), first_day(False)
    other = other_type(record_type)
    raw, as_of = raw_of(data), (held_out + timedelta(days=30)).isoformat()
    target = data / "reports" / REPORT
    put(raw, "stored/RACE", record_type, "2026.jvd", holding_records(record_type, {"7"}, dev))
    put(raw, "stored/RACE", other, "2025.jvd", holding_records(other, {"7"}, date(2025, 3, 16)))
    copied = put(raw, "stored/RACE", other, "2026.jvd", [
        dated(other, "2", held_out, 11),
        dated(other, "7", held_out, 11),
    ])  # fmt: skip

    ok = invoke("--as-of", as_of)
    assert ok.exit_code == 0, ok.output
    target.unlink()

    # the record type's own files gone, a copy of the other type's holdout-day file in their place
    folder = raw / "stored" / "RACE" / record_type
    shutil.rmtree(folder)
    folder.mkdir()
    shutil.copy2(copied, folder / copied.name)
    failed = invoke("--as-of", as_of)
    assert failed.exit_code == 1, failed.output
    assert isinstance(failed.exception, SystemExit), repr(failed.exception)  # reported, no crash
    assert "Traceback" not in failed.output
    assert names(record_type, failed.output), failed.output
    if target.exists():  # a report may be written, never one with a figure on the record type
        written = json.loads(target.read_text(encoding="utf-8"))["record_types"]
        assert record_type not in written, written
