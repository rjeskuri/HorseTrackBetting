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
    データ区分 1-6 is present.
- `horsetrack japan jravan card-versions [--as-of YYYY-MM-DD]` (default: today) runs `census` and
  writes the report; a reports folder inside the repository is refused (non-zero exit).

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
