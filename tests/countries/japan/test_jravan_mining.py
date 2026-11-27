"""JV-Data DM (time-type) and TM (match-type) data-mining predictions into the jravan
`mining_predictions` table (ticket #406).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jravan.mining`:
- `MINING_PREDICTIONS_SCHEMA: pyarrow.Schema`: `jurisdiction` string, `race_id` string,
  `race_date` date32, `venue` string, `race_no` int64, `horse_no` int64, `kind` string, `value`
  float64, `error_plus_s` float64, `error_minus_s` float64, `rank` int64, `data_kubun` string,
  `created_utc` timestamp[us, UTC], in that order.
- `mining_rows(record_type: str, cols: Mapping[str, numpy.ndarray]) -> pyarrow.Table`:
  `record_type` is `DM` or `TM` (anything else a ValueError), `cols` is `records.decode` of
  records of that type; the rows of those records (no version is dropped here), schema
  `MINING_PREDICTIONS_SCHEMA`, ranks included (row order not pinned).
- `build_mining_predictions(*, years=None, raw=None, out=None, chunk_records: int = 500)
  -> records.BuildReport`: reads `<raw>/stored/MING/DM/<YYYY>.jvd` and
  `<raw>/stored/MING/TM/<YYYY>.jvd` (`raw` the export root, default `paths.raw_dir('jravan')`;
  `years` None reads every year file of either folder; a year with only one of the two files reads
  that one alone) with `spec.record_spec('DM')` and `spec.record_spec('TM')`; keeps one version per
  race and record type, JRA venues only, no holdout day; writes `<out>/mining_predictions.parquet`
  (`out` a folder, default `paths.processed_dir('jravan')`): schema `MINING_PREDICTIONS_SCHEMA`,
  rows sorted by `race_id`, `kind`, `horse_no` (numeric), byte-stable, the same bytes whatever
  `chunk_records`. The report's name is `mining`; `counts['rows']` is the rows written and
  `counts['holdout_skipped']` the records on a holdout day; counts only, every one an int.
- Refusals, each before anything is written (a table already there is left as it was): `years`
  None and no `<YYYY>.jvd` in either folder (or no folder) raises FileNotFoundError, and the
  command exits non-zero; an `out` that resolves inside the repository (a relative `out` from the
  cwd) raises `paths.DataDirInsideRepo`, as `paths.outside_repo` does.
- `horsetrack japan jravan mining`: the same build, then `records.write_build_report`
  (`<reports dir>/jravan-mining.json`).
- `dump.MAPPED` gains `("MING", "DM")` and `("MING", "TM")` (test_jravan_dump.py), and
  `model_inputs.yaml`'s `mining_predictions` names this table (test_data_coverage.py accepts it
  as a jravan-only table).
- `mining_predictions` is not in `schemas.TABLES` and, until the owner's availability rule lands
  in `config/availability.yaml`, has no rule: `read_asof` and `asof_join` refuse it
  (`NoAvailabilityRule`). When the owner adds the rule, that test changes with it.

The workbook items each column comes from (`records.decode` keys; names as `spec.record_spec`
returns them):
- `race_id`: `keys.race_id(開催年, 開催月日, 競馬場コード, レース番号)`; `race_date`, `venue`,
  `race_no` as in the race id. Only JRA venues (01-10) are kept; `jurisdiction` is `jra`.
- `kind`: the record type, `DM` or `TM`.
- One row per slot of `マイニング予想` whose `馬番` is a positive integer: `horse_no` that number.
  A blank, zero or non-numeric `馬番` writes no row.
- DM: `value` is `予想走破タイム` in seconds (minutes, then seconds to the hundredth: the last
  four digits are SSff, the digits before them the minutes); `error_plus_s` and `error_minus_s`
  are `予想誤差(信頼度)＋` and `予想誤差(信頼度)－` in seconds (hundredths). TM: `value` is
  `予測スコア` (tenths); both errors null. A value or an error that is blank or not a number is
  null (the runner's row stays: null where unset). An all-zero DM time is unset too (null value,
  null rank, its errors as printed); a zero TM score is a value (0.0, ranked as any other score).
- `rank`, per race and kind over the rows with a value: DM ascending time (the fastest is 1), TM
  descending score (the highest is 1); ties share the lowest rank (1, 1, 3); a null value has a
  null rank.
- `data_kubun`: `データ区分` of the record kept. `created_utc`: `データ作成年月日` +
  `データ作成時分` (HHMM) on the JST clock, as UTC; null when either is blank or invalid.
- One version per race and record type, on `開催年`, `開催月日`, `競馬場コード`, `開催回[第N回]`,
  `開催日目[N日目]`, `レース番号`: the latest by (`データ作成年月日`, `データ作成時分`), a tie to
  the later record; a race whose latest record of a type is a deletion (`データ区分` 0) has no
  rows of that kind (the other kind's rows stay).
- A record whose race date is a holdout day (`is_holdout_day('japan', race_date)`) writes no row
  and counts in `holdout_skipped`.

Every layout here is made up in the workbook's shape (`jravan_workbook.write_workbook`): the item
names are the workbook's, the widths of the time, the errors and the score, the repeat counts and
so every position are invented, so nothing may be sliced by hand. Every record and value is
invented too (D21), and expected rows are written out by hand.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import env, paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.store import schemas, tables
from horsetrack.core.store.asof import (
    NoAvailabilityRule,
    asof_join,
    load_availability,
    read_asof,
)
from horsetrack.countries.japan.research import coverage as cv

# isort: split
# (its own block: the sibling helper module, then the module this ticket creates)
import jravan_workbook as wb

from horsetrack.countries.japan.jravan import mining, records, spec

SOURCE = "jravan"
NAME = "mining_predictions"
FILENAME = "mining_predictions.parquet"
REPO_ROOT = Path(__file__).resolve().parents[3]
UTC = timezone.utc
EXPECTED_SCHEMA = [
    ("jurisdiction", pa.string()),
    ("race_id", pa.string()),
    ("race_date", pa.date32()),
    ("venue", pa.string()),
    ("race_no", pa.int64()),
    ("horse_no", pa.int64()),
    ("kind", pa.string()),
    ("value", pa.float64()),
    ("error_plus_s", pa.float64()),
    ("error_minus_s", pa.float64()),
    ("rank", pa.int64()),
    ("data_kubun", pa.string()),
    ("created_utc", pa.timestamp("us", tz="UTC")),
]
COLUMNS = [name for name, _ in EXPECTED_SCHEMA]
VENUES = {  # number -> short code, as config/jra.yaml lists them
    "01": "SAP", "02": "HAK", "03": "FKS", "04": "NII", "05": "TOK",
    "06": "NAK", "07": "CKY", "08": "KYO", "09": "HSN", "10": "KOK",
}  # fmt: skip

# ------------------------------------------------------------------------------------------------
# Made-up DM and TM layouts: the workbook's item names, invented widths and repeats

TIME = 6  # bytes of 予想走破タイム (invented): MMSSff
ERR = 5  # bytes of each 予想誤差 (invented): hundredths of a second
SCORE = 5  # bytes of 予測スコア (invented): tenths
SLOTS = {"DM": 8, "TM": 7}  # repeats of マイニング予想 (invented)
GROUP = "マイニング予想"
HORSE = "馬番"
DM_SUBS = ((HORSE, 2), ("予想走破タイム", TIME), ("予想誤差(信頼度)＋", ERR),
           ("予想誤差(信頼度)－", ERR))  # fmt: skip
TM_SUBS = ((HORSE, 2), ("予測スコア", SCORE))
SUBS = {"DM": DM_SUBS, "TM": TM_SUBS}

HEAD = (
    ("レコード種別ID", 2),
    ("データ区分", 1),
    ("データ作成年月日", 8),
    ("開催年", 4),
    ("開催月日", 4),
    ("競馬場コード", 2),
    ("開催回[第N回]", 2),
    ("開催日目[N日目]", 2),
    ("レース番号", 2),
    ("データ作成時分", 4),
)


@cache
def section(rt: str) -> wb.Section:
    number = {"DM": 28, "TM": 29}[rt]
    rows: list[wb.Row] = []
    pos = 1
    for no, (name, length) in enumerate(HEAD, start=1):
        if no == 1:
            rows.append(wb.item(1, name, 1, length, key="●", initial=rt, desc=wb.id_desc(rt)))
        else:
            rows.append(wb.item(no, name, pos, length, desc="（架空）"))
        pos += length
    subs = SUBS[rt]
    width = sum(n for _, n in subs)
    no = len(HEAD) + 1
    rows.append(wb.group(no, f"<{GROUP}>", pos, width, SLOTS[rt], desc="予想（架空）"))
    rel = 1
    for i, (name, length) in enumerate(subs):
        rows.append(wb.sub(wb.letter_code(i), name, rel, length, desc="値（架空）"))
        rel += length
    pos += width * SLOTS[rt]
    rows.append(wb.item(no + 1, "レコード区切", pos, 2, desc="区切（架空）"))
    return wb.Section(number, f"予想{rt}（架空）", pos + 1, tuple(rows))


def write_mining_workbook(path: Path) -> Path:
    decoy = wb.simple_section(1, "XQ", "架空その他", [("架空項目", 6)])
    return wb.write_workbook(path, [decoy, section("DM"), section("TM")], [])


def fit(value: str, length: int) -> str:
    """A field as printed: `value` at the field's exact width, or blank."""
    if value and len(value) != length:
        raise ValueError(f"{value!r} is not {length} wide")
    return value or " " * length


# A slot as printed: DM (馬番, 予想走破タイム, 予想誤差＋, 予想誤差－), TM (馬番, 予測スコア);
# each a string at its width or "" for blank.
Slot = tuple[str, ...]


@dataclass
class Rec:
    """One made-up DM or TM record; slots past `slots` are blank."""

    rt: str = "DM"
    year: str = "2024"
    mmdd: str = "0601"
    venue: str = "05"
    race: str = "11"
    kai: str = "03"
    nichi: str = "02"
    kubun: str = "7"
    created: str = "20240603"
    hhmm: str = "1350"
    slots: Sequence[Slot] = field(default_factory=list)

    def record(self) -> bytes:
        head = (self.rt, self.kubun, self.created, self.year, self.mmdd, self.venue, self.kai,
                self.nichi, self.race, self.hhmm)  # fmt: skip
        out = [fit(v, n) for v, (_, n) in zip(head, HEAD, strict=True)]
        subs = SUBS[self.rt]
        assert len(self.slots) <= SLOTS[self.rt], self.slots
        for r in range(SLOTS[self.rt]):
            slot = self.slots[r] if r < len(self.slots) else ("",) * len(subs)
            assert len(slot) == len(subs), slot
            out += [fit(v, n) for v, (_, n) in zip(slot, subs, strict=True)]
        body = "".join(out).encode("cp932")
        assert len(body) == section(self.rt).length - 2
        return body + b"\r\n"


def tm(**kwargs) -> Rec:
    return Rec(rt="TM", **kwargs)


# ------------------------------------------------------------------------------------------------
# Helpers

# (race_id, kind, horse_no, value, error_plus_s, error_minus_s, rank)
Row = tuple[str, str, int, "float | None", "float | None", "float | None", "int | None"]


def r2(v: float | None) -> float | None:
    return None if v is None else round(v, 2)


def check_schema(schema: pa.Schema) -> None:
    assert [(f.name, f.type) for f in schema] == EXPECTED_SCHEMA


def rows_of(table: pa.Table) -> list[Row]:
    """(race_id, kind, horse_no, value, error_plus_s, error_minus_s, rank) per row, in table
    order, after checking the columns derived from the race id."""
    assert table.schema.names == COLUMNS
    out = []
    for r in table.to_pylist():
        jurisdiction, day, venue, race_no = r["race_id"].split(":")
        assert r["jurisdiction"] == "jra" == jurisdiction, r
        assert r["race_date"] == date.fromisoformat(day), r
        assert r["venue"] == venue, r
        assert r["race_no"] == int(race_no), r
        out.append(
            (r["race_id"], r["kind"], r["horse_no"], r2(r["value"]), r2(r["error_plus_s"]),
             r2(r["error_minus_s"]), r["rank"])
        )  # fmt: skip
    return out


def stamps_of(table: pa.Table) -> dict[tuple[str, str], set]:
    """(data_kubun, created_utc) per (race_id, kind)."""
    out: dict[tuple[str, str], set] = {}
    for r in table.to_pylist():
        out.setdefault((r["race_id"], r["kind"]), set()).add((r["data_kubun"], r["created_utc"]))
    return out


def by_key(rows: list[Row]) -> list[Row]:
    return sorted(rows, key=lambda r: (r[0], r[1], r[2]))


@pytest.fixture(scope="module")
def layouts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, spec.RecordSpec]:
    path = write_mining_workbook(tmp_path_factory.mktemp("workbook") / wb.WORKBOOK_NAME)
    return {rt: spec.record_spec(rt, path) for rt in ("DM", "TM")}


def from_records(layouts: Mapping[str, spec.RecordSpec], rt: str, recs: Sequence[Rec]) -> pa.Table:
    assert {r.rt for r in recs} <= {rt}
    cols = records.decode(b"".join(r.record() for r in recs), layouts[rt])
    table = mining.mining_rows(rt, cols)
    assert isinstance(table, pa.Table)
    check_schema(table.schema)
    assert table.schema == mining.MINING_PREDICTIONS_SCHEMA
    return table


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data dir with the made-up workbook and an empty export."""
    root = tmp_path / "data"
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(root))
    write_mining_workbook(root / "specs" / "jravan" / wb.WORKBOOK_NAME)
    for rt in ("DM", "TM"):
        (root / "raw" / SOURCE / "stored" / "MING" / rt).mkdir(parents=True)
    return root


def write_year(data: Path, rt: str, year: int, recs: Sequence[Rec]) -> Path:
    assert {r.rt for r in recs} <= {rt}
    path = data / "raw" / SOURCE / "stored" / "MING" / rt / f"{year}.jvd"
    return wb.write_jvd(path, [r.record() for r in recs])


def store_path(data: Path) -> Path:
    return data.resolve() / "processed" / SOURCE / FILENAME


def store_rows(data: Path) -> list[Row]:
    return rows_of(pq.read_table(store_path(data)))


def jst(y: int, m: int, d: int, hh: int, mm: int) -> datetime:
    """A JST wall-clock minute as an aware UTC datetime (JST is UTC+9, no DST)."""
    return datetime(y, m, d, hh, mm, tzinfo=UTC) - timedelta(hours=9)


# ------------------------------------------------------------------------------------------------
# mining_rows


def test_dm_records_give_one_row_per_runner_with_time_errors_and_rank(
    layouts: Mapping[str, spec.RecordSpec],
) -> None:
    tok11, kyo02 = "jra:2024-06-01:TOK:11", "jra:2024-06-02:KYO:02"
    recs = [
        Rec(slots=[
            ("01", "012345", "00150", "00210"),  # 1:23.45 -> 83.45 s
            ("02", "012290", "00095", "00305"),  # 82.90
            ("10", "013001", "01200", "00007"),  # two-digit horse: numeric order after 2
            ("03", "012345", "00080", "00080"),  # ties with horse 1
            ("04", "", "", ""),  # a runner with nothing set: a row, all null
            ("05", "011958", "  x  ", ""),  # 79.58, a non-numeric and a blank error
        ]),
        Rec(mmdd="0602", venue="08", race="02", kai="02", nichi="04", created="20240602",
            hhmm="0915", kubun="1",
            slots=[
                ("07", "020012", "00001", "00002"),  # 2:00.12 -> 120.12 s
                ("00", "019999", "00001", "00001"),  # horse 00: no row
                ("", "019500", "00001", "00001"),  # blank horse: no row
                ("12", "015900", "00250", "00125"),  # 1:59.00 -> 119.0 s
                ("ab", "019000", "00001", "00001"),  # non-numeric horse: no row
                ("03", "*****" + "*", "00001", "00001"),  # a time that is not a number
            ]),
    ]  # fmt: skip
    table = from_records(layouts, "DM", recs)
    assert by_key(rows_of(table)) == [
        (tok11, "DM", 1, 83.45, 1.5, 2.1, 3),
        (tok11, "DM", 2, 82.9, 0.95, 3.05, 2),
        (tok11, "DM", 3, 83.45, 0.8, 0.8, 3),
        (tok11, "DM", 4, None, None, None, None),
        (tok11, "DM", 5, 79.58, None, None, 1),
        (tok11, "DM", 10, 90.01, 12.0, 0.07, 5),
        (kyo02, "DM", 3, None, 0.01, 0.01, None),
        (kyo02, "DM", 7, 120.12, 0.01, 0.02, 2),
        (kyo02, "DM", 12, 119.0, 2.5, 1.25, 1),
    ]
    assert stamps_of(table) == {
        (tok11, "DM"): {("7", jst(2024, 6, 3, 13, 50))},
        (kyo02, "DM"): {("1", jst(2024, 6, 2, 9, 15))},
    }


def test_tm_records_give_one_row_per_runner_with_score_and_rank(
    layouts: Mapping[str, spec.RecordSpec],
) -> None:
    nak05, hsn12 = "jra:2023-12-24:NAK:05", "jra:2023-12-28:HSN:12"
    recs = [
        tm(year="2023", mmdd="1224", venue="06", race="05", created="20231225", hhmm="2359",
           kubun="7",
           slots=[
               ("01", "00755"),  # 75.5
               ("02", "00800"),  # 80.0, ties with horse 9
               ("09", "00800"),
               ("11", "01000"),  # 100.0, the top score
               ("04", ""),  # a runner without a score: null value, null rank
               ("06", "00003"),  # 0.3
           ]),
        tm(year="2023", mmdd="1228", venue="09", race="12", created="20231227", hhmm="0000",
           kubun="2",
           slots=[("16", "00421"), ("08", "  -  "), ("13", "00999")]),
    ]  # fmt: skip
    table = from_records(layouts, "TM", recs)
    assert by_key(rows_of(table)) == [
        (nak05, "TM", 1, 75.5, None, None, 4),
        (nak05, "TM", 2, 80.0, None, None, 2),
        (nak05, "TM", 4, None, None, None, None),
        (nak05, "TM", 6, 0.3, None, None, 5),
        (nak05, "TM", 9, 80.0, None, None, 2),
        (nak05, "TM", 11, 100.0, None, None, 1),
        (hsn12, "TM", 8, None, None, None, None),
        (hsn12, "TM", 13, 99.9, None, None, 1),
        (hsn12, "TM", 16, 42.1, None, None, 2),
    ]
    assert stamps_of(table) == {
        # 23:59 JST on the 25th is 14:59 UTC the same day; 00:00 JST the 27th is the 26th in UTC
        (nak05, "TM"): {("7", jst(2023, 12, 25, 23, 59))},
        (hsn12, "TM"): {("2", jst(2023, 12, 27, 0, 0))},
    }
    assert stamps_of(table)[(hsn12, "TM")] == {("2", datetime(2023, 12, 26, 15, 0, tzinfo=UTC))}


def test_an_all_zero_dm_time_is_unset_and_a_zero_tm_score_is_a_value(
    layouts: Mapping[str, spec.RecordSpec],
) -> None:
    tok11, tok12 = "jra:2024-06-01:TOK:11", "jra:2024-06-01:TOK:12"
    dm = from_records(layouts, "DM", [
        Rec(slots=[
            ("01", "000000", "00010", "00020"),  # unset time: the row stays, errors as printed
            ("02", "011000", "00030", "00040"),  # 70.0
            ("03", "010500", "00000", "00000"),  # 65.0, the fastest; zero errors are values
            ("04", "000000", "", ""),  # unset time, no errors
        ]),
        Rec(race="12", slots=[("05", "000000", "00010", "00010"), ("06", "000000", "", "")]),
    ])  # fmt: skip
    assert by_key(rows_of(dm)) == [
        (tok11, "DM", 1, None, 0.1, 0.2, None),
        (tok11, "DM", 2, 70.0, 0.3, 0.4, 2),
        (tok11, "DM", 3, 65.0, 0.0, 0.0, 1),
        (tok11, "DM", 4, None, None, None, None),
        (tok12, "DM", 5, None, 0.1, 0.1, None),
        (tok12, "DM", 6, None, None, None, None),
    ]
    # A TM score of zero is the lowest score, not an unset one.
    scores = from_records(layouts, "TM", [
        tm(slots=[("01", "00000"), ("02", "00250"), ("03", "00000"), ("04", "")]),
    ])  # fmt: skip
    assert by_key(rows_of(scores)) == [
        (tok11, "TM", 1, 0.0, None, None, 2),
        (tok11, "TM", 2, 25.0, None, None, 1),
        (tok11, "TM", 3, 0.0, None, None, 2),
        (tok11, "TM", 4, None, None, None, None),
    ]


@pytest.mark.parametrize("rt", ["DM", "TM"])
def test_every_jra_venue_maps_and_other_venues_write_nothing(
    layouts: Mapping[str, spec.RecordSpec], rt: str
) -> None:
    slot = {"DM": ("06", "010000", "00100", "00100"), "TM": ("06", "00500")}[rt]
    value = {"DM": 60.0, "TM": 50.0}[rt]
    jra = [Rec(rt=rt, venue=v, race=f"{i + 1:02d}", slots=[slot]) for i, v in enumerate(VENUES)]
    other = [Rec(rt=rt, venue=v, slots=[slot]) for v in ("30", "44", "55", "A4", "00", "")]
    table = from_records(layouts, rt, [*other[:3], *jra, *other[3:]])
    got = by_key(rows_of(table))
    want = by_key([
        (f"jra:2024-06-01:{code}:{i + 1:02d}", rt, 6, value,
         *((1.0, 1.0) if rt == "DM" else (None, None)), 1)
        for i, code in enumerate(VENUES.values())
    ])  # fmt: skip
    assert got == want


def test_a_blank_or_invalid_creation_stamp_is_null(layouts: Mapping[str, spec.RecordSpec]) -> None:
    def rec(race: str, created: str, hhmm: str) -> Rec:
        return tm(race=race, created=created, hhmm=hhmm, slots=[("01", "00500")])

    table = from_records(layouts, "TM", [
        rec("01", "20240531", "0800"),
        rec("02", "20240531", ""),
        rec("03", "", "0800"),
        rec("04", "20240531", "2561"),
        rec("05", "20241341", "0800"),
    ])  # fmt: skip
    got = {r["race_id"]: r["created_utc"] for r in table.to_pylist()}
    assert got == {
        "jra:2024-06-01:TOK:01": jst(2024, 5, 31, 8, 0),
        "jra:2024-06-01:TOK:02": None,
        "jra:2024-06-01:TOK:03": None,
        "jra:2024-06-01:TOK:04": None,
        "jra:2024-06-01:TOK:05": None,
    }


def test_another_record_type_is_refused(layouts: Mapping[str, spec.RecordSpec]) -> None:
    cols = records.decode(Rec().record(), layouts["DM"])
    with pytest.raises(ValueError):
        mining.mining_rows("H1", cols)


# ------------------------------------------------------------------------------------------------
# build_mining_predictions

T09 = "jra:2009-01-05:NAK:01"
K24A, K24B = "jra:2024-06-01:TOK:10", "jra:2024-06-01:TOK:02"
K24C = "jra:2024-10-13:KYO:11"


def build_races() -> dict[tuple[str, int], list[Rec]]:
    return {
        # 2009: DM only (no TM file that year)
        ("DM", 2009): [
            Rec(year="2009", mmdd="0105", venue="06", race="01", created="20090105",
                slots=[("03", "011000", "00010", "00020"), ("01", "011100", "00030", "00040")]),
            Rec(year="2009", mmdd="0105", venue="44", race="01", created="20090105",
                slots=[("02", "011000", "00010", "00020")]),  # NAR: dropped
        ],
        ("DM", 2024): [
            Rec(mmdd="1013", venue="08", race="11", created="20241014",
                slots=[("14", "015833", "00100", "00100"), ("02", "015800", "00100", "00100")]),
            Rec(race="10", created="20240603",
                slots=[("05", "012000", "00050", "00050")]),
            Rec(race="02", created="20240603",
                slots=[("11", "010950", "00050", "00060"), ("09", "010990", "", "")]),
        ],
        ("TM", 2024): [
            tm(race="02", created="20240603", slots=[("09", "00640"), ("11", "00360")]),
            tm(mmdd="1013", venue="08", race="11", created="20241014",
               slots=[("02", "00505"), ("14", "00495")]),
            tm(mmdd="1013", venue="30", race="11", created="20241014",
               slots=[("01", "00900")]),  # NAR: dropped
            tm(race="10", created="20240603", slots=[("05", "00777")]),
        ],
    }  # fmt: skip


ROWS_2009 = [
    (T09, "DM", 1, 71.0, 0.3, 0.4, 2),
    (T09, "DM", 3, 70.0, 0.1, 0.2, 1),
]
ROWS_2024 = [
    (K24B, "DM", 9, 69.9, None, None, 2),
    (K24B, "DM", 11, 69.5, 0.5, 0.6, 1),
    (K24B, "TM", 9, 64.0, None, None, 1),
    (K24B, "TM", 11, 36.0, None, None, 2),
    (K24A, "DM", 5, 80.0, 0.5, 0.5, 1),
    (K24A, "TM", 5, 77.7, None, None, 1),
    (K24C, "DM", 2, 118.0, 1.0, 1.0, 1),
    (K24C, "DM", 14, 118.33, 1.0, 1.0, 2),
    (K24C, "TM", 2, 50.5, None, None, 1),
    (K24C, "TM", 14, 49.5, None, None, 2),
]


def write_build_races(data: Path) -> None:
    for (rt, year), recs in build_races().items():
        write_year(data, rt, year, recs)


def test_build_writes_mining_predictions_sorted_by_race_kind_and_horse(
    data: Path, tmp_path: Path
) -> None:
    write_build_races(data)
    report = mining.build_mining_predictions()
    assert isinstance(report, records.BuildReport)
    assert report.name == "mining"
    path = store_path(data)
    assert path.is_file()
    assert pq.read_schema(path) == mining.MINING_PREDICTIONS_SCHEMA
    check_schema(pq.read_schema(path))
    # Every JRA race's predictions in race_id, kind, horse_no (numeric) order; no NAR row.
    assert store_rows(data) == ROWS_2009 + ROWS_2024
    assert report.counts["rows"] == len(ROWS_2009) + len(ROWS_2024)
    assert all(isinstance(v, int) for v in report.counts.values())

    # One year only, a year with no TM file.
    mining.build_mining_predictions(years=[2009])
    assert store_rows(data) == ROWS_2009

    # Another export and another output folder; the store's file is left alone.
    other = tmp_path / "elsewhere" / "jravan"
    shutil.copytree(data / "raw" / SOURCE, other)
    out = tmp_path / "out"
    report = mining.build_mining_predictions(years=[2024], raw=other, out=out)
    assert report.counts["rows"] == len(ROWS_2024)
    assert rows_of(pq.read_table(out / FILENAME)) == ROWS_2024
    assert store_rows(data) == ROWS_2009

    # The command builds every year and writes the report, counts only.
    result = CliRunner().invoke(app, ["japan", "jravan", "mining"])
    assert result.exit_code == 0, result.output
    assert store_rows(data) == ROWS_2009 + ROWS_2024
    written = paths.reports_dir() / "jravan-mining.json"
    assert written.is_file()
    parsed = json.loads(written.read_text(encoding="utf-8"))
    assert parsed["name"] == "mining"
    assert parsed["counts"]["rows"] == len(ROWS_2009) + len(ROWS_2024)


def test_the_latest_version_per_race_and_kind_is_kept(data: Path) -> None:
    def kyo(rt: str, race: str, kubun: str, created: str, hhmm: str, slots: list[Slot]) -> Rec:
        return Rec(rt=rt, mmdd="0608", venue="08", race=race, kubun=kubun, created=created,
                   hhmm=hhmm, slots=slots)  # fmt: skip

    write_year(data, "DM", 2024, [
        # race 10: three versions; the latest date wins whatever the file order
        kyo("DM", "10", "1", "20240607", "1800", [("01", "010000", "", ""),
                                                  ("02", "010100", "", "")]),
        kyo("DM", "10", "7", "20240610", "1000", [("01", "010300", "", ""),
                                                  ("02", "010200", "", "")]),
        kyo("DM", "10", "2", "20240608", "0830", [("01", "010500", "", "")]),
        # race 11: two versions on the same day; the later time wins though it comes first
        kyo("DM", "11", "3", "20240608", "1420", [("04", "010000", "", ""),
                                                  ("05", "010001", "", "")]),
        kyo("DM", "11", "2", "20240608", "0900", [("04", "010009", "", ""),
                                                  ("05", "010001", "", "")]),
        # race 12: the same stamp twice; the later record wins
        kyo("DM", "12", "1", "20240607", "1700", [("07", "010000", "", "")]),
        kyo("DM", "12", "1", "20240607", "1700", [("08", "010000", "", "")]),
        # race 09: deleted last, so no DM rows (its TM rows stay)
        kyo("DM", "09", "1", "20240607", "1700", [("03", "010000", "", "")]),
        kyo("DM", "09", "0", "20240609", "1200", []),
        # Same day and race number at another venue: another race.
        Rec(mmdd="0608", venue="05", race="10", kubun="1", created="20240607", hhmm="1800",
            slots=[("06", "010000", "", "")]),
    ])  # fmt: skip
    write_year(data, "TM", 2024, [
        kyo("TM", "09", "3", "20240608", "1400", [("03", "00800")]),
        kyo("TM", "09", "1", "20240607", "1700", [("03", "00100")]),
        kyo("TM", "10", "7", "20240610", "1000", [("01", "00300"), ("02", "00600")]),
    ])  # fmt: skip
    mining.build_mining_predictions()
    k09, k10, k11, k12 = (f"jra:2024-06-08:KYO:{n}" for n in ("09", "10", "11", "12"))
    t10 = "jra:2024-06-08:TOK:10"
    assert store_rows(data) == [
        (k09, "TM", 3, 80.0, None, None, 1),
        (k10, "DM", 1, 63.0, None, None, 2),
        (k10, "DM", 2, 62.0, None, None, 1),
        (k10, "TM", 1, 30.0, None, None, 2),
        (k10, "TM", 2, 60.0, None, None, 1),
        (k11, "DM", 4, 60.0, None, None, 1),
        (k11, "DM", 5, 60.01, None, None, 2),
        (k12, "DM", 8, 60.0, None, None, 1),
        (t10, "DM", 6, 60.0, None, None, 1),
    ]
    assert stamps_of(pq.read_table(store_path(data))) == {
        (k09, "TM"): {("3", jst(2024, 6, 8, 14, 0))},
        (k10, "DM"): {("7", jst(2024, 6, 10, 10, 0))},
        (k10, "TM"): {("7", jst(2024, 6, 10, 10, 0))},
        (k11, "DM"): {("3", jst(2024, 6, 8, 14, 20))},
        (k12, "DM"): {("1", jst(2024, 6, 7, 17, 0))},
        (t10, "DM"): {("1", jst(2024, 6, 7, 18, 0))},
    }


def first_day(holdout: bool) -> date:
    """The first day from 2026-11-28 (the holdout's start) that is, or is not, a holdout day."""
    day = date(2026, 11, 28)
    while is_holdout_day("japan", day) is not holdout:
        day += timedelta(days=1)
    return day


def test_holdout_days_are_never_written(data: Path) -> None:
    held, dev = first_day(True), first_day(False)
    early = date(2026, 11, 22)
    assert is_holdout_day("japan", held)
    assert not is_holdout_day("japan", dev) and not is_holdout_day("japan", early)

    def on(rt: str, day: date, race: str, slots: list[Slot]) -> Rec:
        return Rec(rt=rt, year="2026", mmdd=day.strftime("%m%d"), race=race,
                   created=early.strftime("%Y%m%d"), slots=slots)  # fmt: skip

    dm_slot, tm_slot = ("01", "010000", "00010", "00010"), ("01", "00500")
    write_year(data, "DM", 2026, [
        on("DM", early, "01", [dm_slot]),
        on("DM", held, "01", [dm_slot]),
        on("DM", held, "02", [dm_slot]),
        on("DM", dev, "01", [dm_slot]),
    ])  # fmt: skip
    write_year(data, "TM", 2026, [
        on("TM", held, "01", [tm_slot]),
        on("TM", dev, "01", [tm_slot]),
    ])  # fmt: skip
    report = mining.build_mining_predictions()
    e, d = (f"jra:{x.isoformat()}:TOK:01" for x in (early, dev))
    assert store_rows(data) == [
        (e, "DM", 1, 60.0, 0.1, 0.1, 1),
        *by_key([(d, "DM", 1, 60.0, 0.1, 0.1, 1), (d, "TM", 1, 50.0, None, None, 1)]),
    ]
    days = set(pq.read_table(store_path(data), columns=["race_date"]).column(0).to_pylist())
    assert held not in days
    assert report.counts["holdout_skipped"] == 3
    assert report.counts["rows"] == 3


def test_two_builds_write_identical_bytes(data: Path) -> None:
    write_build_races(data)
    first_report = mining.build_mining_predictions()
    path = store_path(data)
    first = path.read_bytes()
    assert pq.read_metadata(path).num_rows == len(ROWS_2009) + len(ROWS_2024)

    # The same inputs rewritten (new mtimes): the same bytes and the same report.
    write_build_races(data)
    second_report = mining.build_mining_predictions()
    assert path.read_bytes() == first
    assert second_report.to_json() == first_report.to_json()
    assert sorted(q.name for q in path.parent.iterdir()) == [FILENAME]


def test_the_chunk_size_does_not_change_the_output(data: Path, tmp_path: Path) -> None:
    held, dev = first_day(True), first_day(False)

    def on(rt: str, day: date, race: str, created: str, slots: list[Slot]) -> Rec:
        return Rec(rt=rt, year="2026", mmdd=day.strftime("%m%d"), race=race, created=created,
                   slots=slots)  # fmt: skip

    write_build_races(data)
    # Holdout records between the versions of one race, so a chunk of 1 or 2 splits them.
    write_year(data, "DM", 2026, [
        on("DM", dev, "01", "20261120", [("01", "010000", "", ""), ("02", "010100", "", "")]),
        on("DM", held, "01", "20261120", [("01", "010000", "", "")]),
        on("DM", dev, "01", "20261122", [("01", "010300", "", ""), ("02", "010200", "", "")]),
        on("DM", held, "02", "20261120", [("03", "010000", "", "")]),
        on("DM", dev, "02", "20261120", [("04", "010000", "", "")]),
    ])  # fmt: skip
    write_year(data, "TM", 2026, [
        on("TM", held, "01", "20261120", [("01", "00500")]),
        on("TM", dev, "01", "20261121", [("01", "00500"), ("02", "00400")]),
        on("TM", held, "03", "20261120", [("01", "00500")]),
    ])  # fmt: skip
    first = mining.build_mining_predictions(out=tmp_path / "default")
    want = (tmp_path / "default" / FILENAME).read_bytes()
    d1, d2 = (f"jra:{dev.isoformat()}:TOK:{n}" for n in ("01", "02"))
    assert rows_of(pq.read_table(tmp_path / "default" / FILENAME)) == ROWS_2009 + ROWS_2024 + [
        (d1, "DM", 1, 63.0, None, None, 2),
        (d1, "DM", 2, 62.0, None, None, 1),
        (d1, "TM", 1, 50.0, None, None, 1),
        (d1, "TM", 2, 40.0, None, None, 2),
        (d2, "DM", 4, 60.0, None, None, 1),
    ]
    assert first.counts["holdout_skipped"] == 4
    for chunk in (1, 2, 3):
        out = tmp_path / f"chunk{chunk}"
        report = mining.build_mining_predictions(out=out, chunk_records=chunk)
        assert (out / FILENAME).read_bytes() == want, chunk
        assert report.to_json() == first.to_json(), chunk


def ming_without_year_files(data: Path, case: str) -> None:
    """Take every year file out of the export's MING folders, as `case` says."""
    ming = data / "raw" / SOURCE / "stored" / "MING"
    if case == "no MING folder":
        shutil.rmtree(ming)
        return
    for p in ming.glob("*/*.jvd"):
        p.unlink()
    if case == "files that are not year files":
        rec = Rec(slots=[("01", "010000", "00010", "00010")])
        wb.write_jvd(ming / "DM" / "notes.jvd", [rec.record()])
        wb.write_jvd(ming / "TM" / "24.jvd", [tm(slots=[("01", "00500")]).record()])
        (ming / "DM" / "2024.txt").write_bytes(rec.record())


NO_YEAR_FILES = ["empty folders", "no MING folder", "files that are not year files"]


@pytest.mark.parametrize("case", NO_YEAR_FILES)
def test_no_year_file_at_all_is_refused_and_the_table_is_kept(
    data: Path, tmp_path: Path, case: str
) -> None:
    write_build_races(data)
    mining.build_mining_predictions()
    path = store_path(data)
    before = path.read_bytes()
    ming_without_year_files(data, case)

    with pytest.raises(FileNotFoundError):
        mining.build_mining_predictions()
    assert path.read_bytes() == before
    assert sorted(q.name for q in path.parent.iterdir()) == [FILENAME]

    # Another empty export into a fresh folder writes nothing there either.
    fresh = tmp_path / "fresh"
    with pytest.raises(FileNotFoundError):
        mining.build_mining_predictions(raw=tmp_path / "no-export", out=fresh)
    assert not (fresh / FILENAME).exists()
    assert not fresh.exists() or not any(fresh.iterdir())

    # The command fails and leaves the table as it was.
    result = CliRunner().invoke(app, ["japan", "jravan", "mining"])
    assert result.exit_code != 0, result.output
    assert path.read_bytes() == before
    assert sorted(q.name for q in path.parent.iterdir()) == [FILENAME]


def test_an_out_folder_inside_the_repository_is_refused(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real path guard against a stand-in checkout under tmp_path, so nothing here can write
    to, or clean up in, the real one."""
    write_build_races(data)
    repo = (tmp_path / "repo").resolve()
    (repo / "docs").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'stand-in'\n")
    monkeypatch.setattr(env, "repo_root", lambda: repo)
    inside = repo / "out"

    def tree() -> set[Path]:
        return set(repo.rglob("*"))

    before = tree()
    monkeypatch.chdir(repo)
    # A folder in the checkout, the checkout itself as a relative path, and one through "..".
    for out in (inside, Path("."), Path("docs") / ".."):
        with pytest.raises(paths.DataDirInsideRepo):
            mining.build_mining_predictions(out=out)
        assert not inside.exists(), out
        assert tree() == before, out
    assert not store_path(data).exists()


# ------------------------------------------------------------------------------------------------
# Availability, the model inputs and the coverage doc


def test_the_table_has_no_rule_yet_so_the_views_refuse_it(data: Path) -> None:
    """No member reads the table before the owner's availability rule (DoD); the rule is the
    owner's, so until it lands the table has none and the views refuse it."""
    write_build_races(data)
    mining.build_mining_predictions()
    path = store_path(data)
    assert path.is_file()

    assert NAME not in schemas.TABLES
    with pytest.raises(ValueError, match="not a store table"):
        tables.read_table(SOURCE, NAME)
    rules = load_availability("japan")
    con = tables.connect(SOURCE)
    con.execute(f"CREATE VIEW {NAME} AS SELECT * FROM read_parquet('{path}')")
    assert con.execute(f"SELECT count(*) FROM {NAME}").fetchone()[0] == len(ROWS_2009 + ROWS_2024)
    times = [
        datetime(2000, 1, 1, tzinfo=UTC),
        datetime(2024, 6, 1, 3, 0, tzinfo=UTC),
        datetime(2030, 1, 1, tzinfo=UTC),
    ]
    for t in times:
        with pytest.raises(NoAvailabilityRule):
            read_asof(con, NAME, t, rules=rules)
    keys = pd.DataFrame({"race_id": [K24A, K24C, T09], "t_utc": pd.to_datetime(times, utc=True)})
    with pytest.raises(NoAvailabilityRule):
        asof_join(keys, NAME, on=["race_id"], rules=rules, con=con)


def test_model_inputs_name_the_mining_predictions_table() -> None:
    inputs = {inp.name: inp for inp in cv.load_inputs()}
    inp = inputs["mining_predictions"]
    assert "C4" in inp.layers
    assert inp.supplies, "mining_predictions has no supply"
    names = mining.MINING_PREDICTIONS_SCHEMA.names
    for s in inp.supplies:
        assert (s.source, s.table) == (SOURCE, NAME), s
        assert s.column in names, s
        assert set(dict(s.filter or {})) <= set(names), s
    # No input reads the dumped DM or TM records any more.
    dumped = [(i.name, s.table) for i in inputs.values() for s in i.supplies
              if s.table in ("records/MING/DM", "records/MING/TM")]  # fmt: skip
    assert not dumped, dumped


def test_the_committed_coverage_doc_lists_the_table() -> None:
    text = (REPO_ROOT / "docs" / "data-coverage.md").read_text(encoding="utf-8")
    body = text.split("\n## mining_predictions\n", 1)[1].split("\n## ", 1)[0]
    lines = [ln for ln in body.splitlines() if ln.startswith("- ")]
    assert lines, "docs/data-coverage.md lists no supply of mining_predictions"
    assert all(f"`{NAME}." in ln and "withheld (D21)" in ln for ln in lines), lines
    assert "records/MING" not in body


# ------------------------------------------------------------------------------------------------
# The real export (local: needs $HORSETRACK_DATA_DIR with the export and the workbook)

REAL_YEAR = 2024  # a year before the holdout; the test skips unless both files are there


@pytest.mark.local
def test_a_real_year_maps(tmp_path: Path) -> None:
    export, workbook = wb.real_export_or_skip()
    sources = {rt: export / "stored" / "MING" / rt / f"{REAL_YEAR}.jvd" for rt in ("DM", "TM")}
    if not all(s.is_file() for s in sources.values()):
        pytest.skip(f"no DM or TM file for {REAL_YEAR}")
    out = tmp_path / "out"
    report = mining.build_mining_predictions(years=[REAL_YEAR], raw=export, out=out)
    got = pq.read_table(out / FILENAME)
    check_schema(got.schema)
    df = got.to_pandas()
    n = len(df)
    assert n > 0 and report.counts["rows"] == n
    assert set(df["kind"]) == {"DM", "TM"}
    assert (df["jurisdiction"] == "jra").all()
    venues = "|".join(VENUES.values())
    assert df["race_id"].str.fullmatch(rf"jra:{REAL_YEAR}-\d\d-\d\d:(?:{venues}):\d\d").all()
    assert df["horse_no"].between(1, 18).all()
    triples = list(zip(df["race_id"], df["kind"], df["horse_no"], strict=True))
    assert triples == sorted(triples)
    assert len(set(triples)) == n
    tm_rows = df[df["kind"] == "TM"]
    assert tm_rows["error_plus_s"].isna().all() and tm_rows["error_minus_s"].isna().all()
    assert tm_rows["value"].dropna().between(0, 100).all()
    dm_rows = df[df["kind"] == "DM"]
    assert dm_rows["value"].dropna().between(30, 600).all()  # seconds, any JRA distance
    # Every race and kind with a value has rank 1, and the ranks follow the values.
    valued = df[df["value"].notna()]
    assert (valued.groupby(["race_id", "kind"])["rank"].min() == 1).all()
    for kind, ascending in (("DM", True), ("TM", False)):
        mine = valued[valued["kind"] == kind]
        want = mine.groupby("race_id")["value"].rank(method="min", ascending=ascending)
        assert (want.astype(int).to_numpy() == mine["rank"].to_numpy()).all(), kind

    # The rows, independently of the module: the kept records' slots with a positive 馬番.
    del df, got
    expected = 0
    for rt, path in sources.items():
        cols = records.decode(path.read_bytes(), spec.record_spec(rt, workbook))
        horse = records.integer(cols[f"{GROUP}/{HORSE}"])
        jra = np.isin(cols["競馬場コード"], [v.encode() for v in VENUES])
        keep = jra & (cols["データ区分"] != b"0")
        expected += int(((horse > 0) & keep[:, None]).sum())
        del cols
    # only kept versions write rows, so n is at most the slots counted over every version
    assert n <= expected
