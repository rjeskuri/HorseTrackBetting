"""JV-Data JG (horses excluded from a race's entries) into the jravan `exclusions` table
(ticket #405).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jravan.exclusions`:
- `EXCLUSIONS_SCHEMA: pyarrow.Schema`: `jurisdiction` string, `race_id` string, `race_date`
  date32, `venue` string, `race_no` int64, `horse_id` string, `horse_name` string, `entry_order`
  int64, `entry_kind` string, `exclusion_kind` string, `published_date` date32, in that order.
- `build_exclusions(*, years=None, raw=None, out=None) -> records.BuildReport` (more keyword-only
  optional arguments allowed): reads `<raw>/stored/RACE/JG/<YYYY>.jvd` (`raw` the export root,
  default `paths.raw_dir('jravan')`; `years` None reads every year file) with the layout
  `spec.record_spec('JG')`; keeps one version per entry and JRA venues only; writes
  `<out>/exclusions.parquet` (`out` a folder, default `paths.processed_dir('jravan')`): schema
  `EXCLUSIONS_SCHEMA`, rows sorted by `race_id`, `horse_id` (plain string order), then
  `entry_order`, byte-stable whatever the order of the records in the files. The report's name is
  `exclusions` and `counts['rows']` is the rows written; `unmapped` is field -> code -> count of
  the codes no translation knows (the field names are the coder's); counts only, no record content.
- `horsetrack japan jravan exclusions [--year YYYY ...]`: the same build, then
  `records.write_build_report` (`<reports dir>/jravan-exclusions.json`, outside the repo, D21).
- `model_inputs.yaml`'s `exclusions` input names a jravan `exclusions` supply (no longer the
  dumped `records/RACE/JG`), and the committed `docs/data-coverage.md` shows it withheld (D21).
  Its availability rule is the owner's question (#405) and is not pinned here.

The JG items each column comes from (names as `spec.record_spec` returns them):
- one row per JG record kept (every entry the record type lists, excluded or not, so a consumer
  filters on the kinds); one version per entry: `records.latest_versions` on 開催年, 開催月日,
  競馬場コード, 開催回[第N回], 開催日目[N日目], レース番号, 血統登録番号, 出馬投票受付順番
  (データ作成年月日, データ区分; an entry whose latest record is a deletion, データ区分 0, has
  no row).
- `jurisdiction` `jra`; only records whose 競馬場コード is a JRA venue (01-10) are kept.
- `race_id`: `keys.race_id(開催年, 開催月日, 競馬場コード, レース番号)`; `race_date`, `venue` and
  `race_no` as in the race id. A JRA record whose race id `keys.race_id` rejects (race number 00,
  an impossible date) gives no row and is counted in `counts['no_race_id']` (orchestrator's
  ruling, #405).
- `horse_id`: 血統登録番号 as text; `horse_name`: 馬名, half- and full-width spaces trimmed;
  `entry_order`: 出馬投票受付順番 as an integer.
- `entry_kind`: 出走区分 translated: 1 `declared`, 2 `excluded_at_close`, 4 `redeclared`,
  5 `excluded_on_redeclaration`, 6 `scratched_before_numbering`, 9 `scratched`.
- `exclusion_kind`: 除外状態区分 translated: 1 `no_ballot`, 2 `lost_ballot`.
  For both, an unset code (blank or 0) is null and not counted; a code outside the translation is
  null and counted in the report's `unmapped`.
- `published_date`: データ作成年月日 (YYYYMMDD) as a date; `00000000` or blank is null.

Every layout here is made up in the workbook's shape (`jravan_workbook.write_workbook`): the item
names are the workbook's, the widths of the horse name and the entry order, a reserve item and so
every position are invented, so nothing may be sliced by hand. Every record and value is invented
too. Expected rows are written out by hand.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.core.eval.reports import is_withheld
from horsetrack.countries.japan.research import coverage as cv

# isort: split
# (its own block: the sibling helper module, then the module this ticket creates)
import jravan_workbook as wb

from horsetrack.countries.japan.jravan import exclusions, records, spec

SOURCE = "jravan"
TABLE = "exclusions"
FILENAME = "exclusions.parquet"
RT = "JG"
REPO_ROOT = Path(__file__).resolve().parents[3]
COVERAGE_DOC = REPO_ROOT / "docs" / "data-coverage.md"
WITHHELD = "withheld (D21)"
EXPECTED_SCHEMA = [
    ("jurisdiction", pa.string()),
    ("race_id", pa.string()),
    ("race_date", pa.date32()),
    ("venue", pa.string()),
    ("race_no", pa.int64()),
    ("horse_id", pa.string()),
    ("horse_name", pa.string()),
    ("entry_order", pa.int64()),
    ("entry_kind", pa.string()),
    ("exclusion_kind", pa.string()),
    ("published_date", pa.date32()),
]
COLUMNS = [name for name, _ in EXPECTED_SCHEMA]
ENTRY_KINDS = {
    "1": "declared", "2": "excluded_at_close", "4": "redeclared",
    "5": "excluded_on_redeclaration", "6": "scratched_before_numbering", "9": "scratched",
}  # fmt: skip
EXCLUSION_KINDS = {"1": "no_ballot", "2": "lost_ballot"}
VENUES = {  # number -> short code, as config/jra.yaml lists them
    "01": "SAP", "02": "HAK", "03": "FKS", "04": "NII", "05": "TOK",
    "06": "NAK", "07": "CKY", "08": "KYO", "09": "HSN", "10": "KOK",
}  # fmt: skip

# ------------------------------------------------------------------------------------------------
# A made-up JG layout: the workbook's item names, invented widths for the name and the order, an
# invented reserve item

NAME_BYTES = 40  # 馬名 (invented)
ORDER_DIGITS = 4  # 出馬投票受付順番 (invented)
FULL_SPACE = "　"

LAYOUT: tuple[tuple[str, int, bool], ...] = (  # (name, bytes, key item)
    ("レコード種別ID", 2, False),
    ("データ区分", 1, False),
    ("データ作成年月日", 8, False),
    ("開催年", 4, True),
    ("開催月日", 4, True),
    ("競馬場コード", 2, True),
    ("開催回[第N回]", 2, True),
    ("開催日目[N日目]", 2, True),
    ("レース番号", 2, True),
    ("血統登録番号", 10, True),
    ("馬名", NAME_BYTES, False),
    ("出馬投票受付順番", ORDER_DIGITS, True),
    ("予備", 3, False),
    ("出走区分", 1, False),
    ("除外状態区分", 1, False),
    ("レコード区切", 2, False),
)


def section() -> wb.Section:
    rows: list[wb.Row] = []
    pos = 1
    for no, (name, length, is_key) in enumerate(LAYOUT, start=1):
        if no == 1:
            rows.append(wb.item(1, name, 1, length, initial=RT, desc=wb.id_desc(RT)))
        else:
            rows.append(wb.item(no, name, pos, length, key="○" if is_key else "", desc="（架空）"))
        pos += length
    return wb.Section(9, "除外（架空）", pos - 1, tuple(rows))


def write_exclusions_workbook(path: Path) -> Path:
    decoy = wb.simple_section(1, "XQ", "架空その他", [("架空項目", 6)])
    return wb.write_workbook(path, [decoy, section()], [])


@dataclass(frozen=True)
class Jg:
    """One made-up JG record."""

    horse: str = "2021104321"
    name: str = "アカイホシ"
    order: int | str = 1
    entry: str = "2"
    exclusion: str = "1"
    year: str = "2024"
    mmdd: str = "0608"
    venue: str = "05"
    race: str = "11"
    kai: str = "03"
    nichi: str = "03"
    kubun: str = "1"
    created: str = "20240606"

    def record(self) -> bytes:
        order = f"{self.order:0{ORDER_DIGITS}d}" if isinstance(self.order, int) else self.order
        values = {
            "レコード種別ID": RT, "データ区分": self.kubun, "データ作成年月日": self.created,
            "開催年": self.year, "開催月日": self.mmdd, "競馬場コード": self.venue,
            "開催回[第N回]": self.kai, "開催日目[N日目]": self.nichi, "レース番号": self.race,
            "血統登録番号": self.horse, "出馬投票受付順番": order, "予備": "",
            "出走区分": self.entry, "除外状態区分": self.exclusion,
        }  # fmt: skip
        out = b""
        for name, length, _ in LAYOUT[:-1]:
            if name == "馬名":  # full-width text, padded with full-width spaces as exported
                raw = self.name.encode("cp932")
                pad = (length - len(raw)) // 2
                raw += (FULL_SPACE * pad).encode("cp932")
                raw += b" " * (length - len(raw))
            else:
                raw = values[name].encode("cp932")
                if raw and len(raw) != length:
                    raise ValueError(f"{name}: {values[name]!r} is not {length} wide")
                raw = raw or b" " * length
            assert len(raw) == length, name
            out += raw
        return out + b"\r\n"


Row = tuple[str, str, "str | None", "int | None", "str | None", "str | None", "date | None"]


def row(
    race_id: str,
    horse: str,
    name: str | None,
    order: int | None,
    entry: str | None,
    exclusion: str | None,
    published: date | None,
) -> Row:
    """An expected row: (race_id, horse_id, horse_name, entry_order, entry_kind, exclusion_kind,
    published_date)."""
    return (race_id, horse, name, order, entry, exclusion, published)


def check_schema(schema: pa.Schema) -> None:
    assert [(f.name, f.type) for f in schema] == EXPECTED_SCHEMA


def rows_of(table: pa.Table) -> list[Row]:
    """The rows in table order, after checking the columns derived from the race id."""
    assert table.schema.names == COLUMNS
    out = []
    for r in table.to_pylist():
        jurisdiction, day, venue, race_no = r["race_id"].split(":")
        assert r["jurisdiction"] == "jra" == jurisdiction, r
        assert r["race_date"] == date.fromisoformat(day), r
        assert r["venue"] == venue, r
        assert r["race_no"] == int(race_no), r
        out.append(
            (r["race_id"], r["horse_id"], r["horse_name"], r["entry_order"], r["entry_kind"],
             r["exclusion_kind"], r["published_date"])
        )  # fmt: skip
    return out


def flat_unmapped(report: records.BuildReport) -> dict[str, int]:
    """Every unmapped code (trimmed) with its count, whatever field the report files it under."""
    out: dict[str, int] = {}
    for codes in report.unmapped.values():
        for code, n in codes.items():
            out[str(code).strip()] = out.get(str(code).strip(), 0) + n
    return out


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data dir with the made-up workbook and an empty JG folder."""
    root = tmp_path / "data"
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(root))
    write_exclusions_workbook(root / "specs" / "jravan" / wb.WORKBOOK_NAME)
    (root / "raw" / SOURCE / "stored" / "RACE" / RT).mkdir(parents=True)
    return root


def write_year(data: Path, year: int, recs: Sequence[Jg]) -> Path:
    path = data / "raw" / SOURCE / "stored" / "RACE" / RT / f"{year}.jvd"
    return wb.write_jvd(path, [r.record() for r in recs])


def store_path(data: Path) -> Path:
    return data.resolve() / "processed" / SOURCE / FILENAME


def store_rows(data: Path) -> list[Row]:
    return rows_of(pq.read_table(store_path(data)))


# ------------------------------------------------------------------------------------------------
# Rows, codes and venues

T11, K10, N03 = "jra:2024-06-08:TOK:11", "jra:2024-06-08:KYO:10", "jra:2024-06-09:NII:03"
D0606, D0607 = date(2024, 6, 6), date(2024, 6, 7)


def build_2024() -> list[Jg]:
    """Three JRA races out of race_id order, every 出走区分 and 除外状態区分, a NAR and an
    overseas record."""
    return [
        Jg(horse="2021105555", name="ハヤテカゼ", order=1, entry="1", exclusion="0"),
        Jg(horse="2021101111", name="アオイソラ", order=1, entry="2", exclusion="2"),
        Jg(horse="2021103333", name="ミドリノオカ", order=1, entry="2", exclusion="1"),
        Jg(horse="2020102222", name="シロイクモ", order=1, entry="4", exclusion="0"),
        Jg(horse="2021101111", name="アオイソラ", order=1, entry="5", exclusion="2",
           venue="08", race="10", kai="02", nichi="04", created="20240607"),
        Jg(horse="2019109999", name="クロガネ", order=2, entry="5", exclusion="1",
           venue="08", race="10", kai="02", nichi="04", created="20240607"),
        Jg(horse="2019108888", name="キンノヤ", order=1, entry="6", exclusion="0",
           venue="08", race="10", kai="02", nichi="04", created="20240607"),
        Jg(horse="2022100007", name="ユメノハナ", order=3, entry="9", exclusion="",
           venue="04", mmdd="0609", race="03", kai="01", nichi="05"),
        # Not JRA: a NAR venue and an overseas code, never a row.
        Jg(horse="2021107777", name="チホウバ", entry="2", exclusion="2", venue="30", race="11"),
        Jg(horse="2021106666", name="カイガイ", entry="2", exclusion="1", venue="A6", race="11"),
    ]  # fmt: skip


ROWS_2024 = [  # sorted by race_id in plain string order (the date first: 06-08 KYO < 06-08 TOK
    # < 06-09 NII), then horse_id, entry_order
    row(K10, "2019108888", "キンノヤ", 1, "scratched_before_numbering", None, D0607),
    row(K10, "2019109999", "クロガネ", 2, "excluded_on_redeclaration", "no_ballot", D0607),
    row(K10, "2021101111", "アオイソラ", 1, "excluded_on_redeclaration", "lost_ballot", D0607),
    row(T11, "2020102222", "シロイクモ", 1, "redeclared", None, D0606),
    row(T11, "2021101111", "アオイソラ", 1, "excluded_at_close", "lost_ballot", D0606),
    row(T11, "2021103333", "ミドリノオカ", 1, "excluded_at_close", "no_ballot", D0606),
    row(T11, "2021105555", "ハヤテカゼ", 1, "declared", None, D0606),
    row(N03, "2022100007", "ユメノハナ", 3, "scratched", None, D0606),
]


def test_jg_records_become_one_row_per_entry_with_translated_codes(data: Path) -> None:
    check_schema(exclusions.EXCLUSIONS_SCHEMA)
    write_year(data, 2024, build_2024())
    report = exclusions.build_exclusions()
    assert isinstance(report, records.BuildReport)
    assert report.name == "exclusions"
    path = store_path(data)
    assert path.is_file()
    assert pq.read_schema(path) == exclusions.EXCLUSIONS_SCHEMA
    check_schema(pq.read_schema(path))
    assert store_rows(data) == ROWS_2024
    assert report.counts["rows"] == len(ROWS_2024)
    assert all(isinstance(v, int) for v in report.counts.values())
    assert flat_unmapped(report) == {}


def test_every_jra_venue_and_every_code_maps(data: Path) -> None:
    # Each JRA venue a race, each with another horse, order, code pair and publication date.
    recs, expected = [], []
    pairs = [(e, x) for e in ENTRY_KINDS for x in ("0", *EXCLUSION_KINDS)]
    for i, (number, code) in enumerate(VENUES.items()):
        entry, exclusion = pairs[(5 * i) % len(pairs)]
        horse, order, day = f"20181{i:05d}", i % 3 + 1, date(2015, 3, 2 + i)
        recs.append(
            Jg(horse=horse, name="テスト" + "アイウエオカキクケコ"[i], order=order, entry=entry,
               exclusion=exclusion, year="2015", mmdd="0315", venue=number, race=f"{i + 1:02d}",
               created=day.strftime("%Y%m%d"))
        )  # fmt: skip
        expected.append(
            row(f"jra:2015-03-15:{code}:{i + 1:02d}", horse, "テスト" + "アイウエオカキクケコ"[i],
                order, ENTRY_KINDS[entry], EXCLUSION_KINDS.get(exclusion), day)
        )  # fmt: skip
    # The remaining pairs in one more race, so every (出走区分, 除外状態区分) pair appears.
    for j, (entry, exclusion) in enumerate(pairs):
        horse = f"20191{j:05d}"
        recs.append(
            Jg(horse=horse, name="ペア", order=j + 1, entry=entry, exclusion=exclusion,
               year="2015", mmdd="0322", venue="09", race="12", created="20150320")
        )  # fmt: skip
        expected.append(row("jra:2015-03-22:HSN:12", horse, "ペア", j + 1, ENTRY_KINDS[entry],
                            EXCLUSION_KINDS.get(exclusion), date(2015, 3, 20)))  # fmt: skip
    write_year(data, 2015, recs)
    exclusions.build_exclusions()
    got = store_rows(data)
    assert got == sorted(expected, key=lambda r: (r[0], r[1], r[3]))
    assert {r[4] for r in got} == set(ENTRY_KINDS.values())
    assert {r[5] for r in got} == {None, *EXCLUSION_KINDS.values()}


def test_unset_and_unknown_codes_are_null_and_only_unknown_ones_counted(data: Path) -> None:
    write_year(
        data, 2022,
        [
            Jg(horse="2019100001", entry="3", exclusion="2", year="2022", mmdd="1009",
               venue="06", created="20221007"),
            Jg(horse="2019100002", entry="3", exclusion="7", year="2022", mmdd="1009",
               venue="06", created="20221007"),
            Jg(horse="2019100003", entry="", exclusion="", year="2022", mmdd="1009",
               venue="06", created="20221007"),
            Jg(horse="2019100004", entry="0", exclusion="0", year="2022", mmdd="1009",
               venue="06", created="00000000"),
            Jg(horse="2019100005", name="", entry="9", exclusion="1", year="2022", mmdd="1009",
               venue="06", created="20221008"),
        ],
    )  # fmt: skip
    report = exclusions.build_exclusions()
    rid = "jra:2022-10-09:NAK:11"
    assert store_rows(data) == [
        row(rid, "2019100001", "アカイホシ", 1, None, "lost_ballot", date(2022, 10, 7)),
        row(rid, "2019100002", "アカイホシ", 1, None, None, date(2022, 10, 7)),
        row(rid, "2019100003", "アカイホシ", 1, None, None, date(2022, 10, 7)),
        row(rid, "2019100004", "アカイホシ", 1, None, None, None),
        row(rid, "2019100005", None, 1, "scratched", "no_ballot", date(2022, 10, 8)),
    ]
    # Unset codes (blank, 0) are not unmapped; codes outside the translation are.
    assert flat_unmapped(report) == {"3": 2, "7": 1}


def test_a_record_without_a_race_id_gives_no_row_and_is_counted(data: Path) -> None:
    # Race 00 (a scratched horse with no race, as the export has it) and an impossible date at
    # JRA venues: no row, each counted under no_race_id; the valid entry beside them stays.
    write_year(
        data, 2021,
        [
            Jg(horse="2018100001", entry="9", exclusion="0", year="2021", mmdd="0516",
               venue="05", race="00", created="20210514"),
            Jg(horse="2018100002", entry="2", exclusion="2", year="2021", mmdd="0230",
               venue="05", race="11", created="20210514"),
            Jg(horse="2018100003", entry="2", exclusion="1", year="2021", mmdd="0516",
               venue="05", race="11", created="20210514"),
        ],
    )  # fmt: skip
    report = exclusions.build_exclusions()
    assert store_rows(data) == [
        row("jra:2021-05-16:TOK:11", "2018100003", "アカイホシ", 1, "excluded_at_close",
            "no_ballot", date(2021, 5, 14)),
    ]  # fmt: skip
    assert report.counts["rows"] == 1
    assert report.counts["no_race_id"] == 2


# ------------------------------------------------------------------------------------------------
# Versions


def test_the_latest_version_per_entry_is_kept(data: Path) -> None:
    def kyo(horse: str, order: int, entry: str, exclusion: str, created: str, **kw) -> Jg:
        where = {"year": "2023", "mmdd": "0415", "venue": "08", "race": "09", **kw}
        return Jg(horse=horse, order=order, entry=entry, exclusion=exclusion, created=created,
                  **where)  # fmt: skip

    write_year(
        data, 2023,
        [
            # The later version first in the file: the later データ作成年月日 wins, not the order.
            kyo("2020100001", 1, "2", "2", "20230414"),
            kyo("2020100001", 1, "2", "1", "20230412"),
            kyo("2020100001", 1, "1", "0", "20230413"),
            # A redeclared horse: two orders, two entries.
            kyo("2020100002", 1, "1", "0", "20230412"),
            kyo("2020100002", 2, "5", "2", "20230413"),
            # The same horse and order in another race (another 開催日目) is another entry.
            kyo("2020100001", 1, "9", "0", "20230411", mmdd="0409", nichi="02"),
            # The same day and race number at another venue is another race.
            Jg(horse="2020100001", order=1, entry="4", exclusion="0", year="2023", mmdd="0415",
               venue="09", race="09", created="20230412"),
        ],
    )  # fmt: skip
    exclusions.build_exclusions()
    k09, k09early, h09 = (
        "jra:2023-04-15:KYO:09", "jra:2023-04-09:KYO:09", "jra:2023-04-15:HSN:09"
    )  # fmt: skip
    assert store_rows(data) == [
        row(k09early, "2020100001", "アカイホシ", 1, "scratched", None, date(2023, 4, 11)),
        row(h09, "2020100001", "アカイホシ", 1, "redeclared", None, date(2023, 4, 12)),
        row(k09, "2020100001", "アカイホシ", 1, "excluded_at_close", "lost_ballot",
            date(2023, 4, 14)),
        row(k09, "2020100002", "アカイホシ", 1, "declared", None, date(2023, 4, 12)),
        row(k09, "2020100002", "アカイホシ", 2, "excluded_on_redeclaration", "lost_ballot",
            date(2023, 4, 13)),
    ]  # fmt: skip


def test_a_deletion_record_leaves_no_row(data: Path) -> None:
    def tok(horse: str, kubun: str, created: str, **kw) -> Jg:
        where = {"year": "2023", "mmdd": "1105", "venue": "05", "race": "10", **kw}
        return Jg(horse=horse, kubun=kubun, created=created, **where)

    write_year(
        data, 2023,
        [
            # Deleted after it was provided: no row.
            tok("2020100011", "1", "20231102"),
            tok("2020100011", "0", "20231104"),
            # A deletion alone: no row.
            tok("2020100012", "0", "20231103"),
            # Deleted, then provided again: the later record stands.
            tok("2020100013", "0", "20231102", entry="2", exclusion="1"),
            tok("2020100013", "1", "20231103", entry="2", exclusion="2"),
            # The deleted horse's entry in another race stays.
            tok("2020100011", "1", "20231102", race="11"),
            # Another horse of the deleted entry's race stays.
            tok("2020100014", "1", "20231102", entry="1", exclusion="0"),
            # A deletion of one order leaves the horse's other order.
            tok("2020100015", "1", "20231102", order=1, entry="1", exclusion="0"),
            tok("2020100015", "1", "20231102", order=2, entry="5", exclusion="2"),
            tok("2020100015", "0", "20231104", order=2),
        ],
    )  # fmt: skip
    exclusions.build_exclusions()
    t10, t11 = "jra:2023-11-05:TOK:10", "jra:2023-11-05:TOK:11"
    assert store_rows(data) == [
        row(t10, "2020100013", "アカイホシ", 1, "excluded_at_close", "lost_ballot",
            date(2023, 11, 3)),
        row(t10, "2020100014", "アカイホシ", 1, "declared", None, date(2023, 11, 2)),
        row(t10, "2020100015", "アカイホシ", 1, "declared", None, date(2023, 11, 2)),
        row(t11, "2020100011", "アカイホシ", 1, "excluded_at_close", "no_ballot",
            date(2023, 11, 2)),
    ]  # fmt: skip

    # A year whose only records are deletions: an empty table with the schema.
    write_year(data, 2023, [tok("2020100011", "1", "20231102"), tok("2020100011", "0", "20231104")])
    report = exclusions.build_exclusions()
    table = pq.read_table(store_path(data))
    check_schema(table.schema)
    assert table.num_rows == 0
    assert report.counts["rows"] == 0


# ------------------------------------------------------------------------------------------------
# Build: years, folders, bytes


ROWS_2019 = [
    row("jra:2019-01-05:NAK:01", "2016100100", "フユノヒ", 1, "excluded_at_close", "lost_ballot",
        date(2019, 1, 3)),
    row("jra:2019-12-28:NAK:11", "2016100200", "トシノセ", 2, "scratched", None,
        date(2019, 12, 26)),
]  # fmt: skip


def build_2019() -> list[Jg]:
    return [
        Jg(horse="2016100200", name="トシノセ", order=2, entry="9", exclusion="0", year="2019",
           mmdd="1228", venue="06", race="11", created="20191226"),
        Jg(horse="2016100100", name="フユノヒ", order=1, entry="2", exclusion="2", year="2019",
           mmdd="0105", venue="06", race="01", created="20190103"),
    ]  # fmt: skip


def test_build_reads_every_year_or_the_years_named(data: Path, tmp_path: Path) -> None:
    write_year(data, 2019, build_2019())
    write_year(data, 2024, build_2024())
    report = exclusions.build_exclusions()
    assert store_rows(data) == ROWS_2019 + ROWS_2024
    assert report.counts["rows"] == len(ROWS_2019) + len(ROWS_2024)

    # One year only.
    report = exclusions.build_exclusions(years=[2019])
    assert store_rows(data) == ROWS_2019
    assert report.counts["rows"] == len(ROWS_2019)

    # Another export and another output folder; the store's file is left alone.
    other = tmp_path / "elsewhere" / "jravan"
    shutil.copytree(data / "raw" / SOURCE, other)
    (data / "raw" / SOURCE / "stored" / "RACE" / RT / "2024.jvd").unlink()
    out = tmp_path / "out"
    report = exclusions.build_exclusions(years=[2024], raw=other, out=out)
    assert report.counts["rows"] == len(ROWS_2024)
    assert rows_of(pq.read_table(out / FILENAME)) == ROWS_2024
    assert store_rows(data) == ROWS_2019


def test_two_builds_write_identical_bytes_whatever_the_record_order(
    data: Path, tmp_path: Path
) -> None:
    recs_2024 = build_2024()
    write_year(data, 2019, build_2019())
    write_year(data, 2024, recs_2024)
    first_report = exclusions.build_exclusions()
    path = store_path(data)
    first = path.read_bytes()

    # The same inputs rewritten (new mtimes): the same bytes and the same report.
    write_year(data, 2019, build_2019())
    write_year(data, 2024, recs_2024)
    second_report = exclusions.build_exclusions()
    assert path.read_bytes() == first
    assert second_report.to_json() == first_report.to_json()
    assert sorted(q.name for q in path.parent.iterdir()) == [FILENAME]

    # The records shuffled within their files (no two versions of an entry): the same bytes.
    for seed in (1, 2, 3):
        shuffled = list(recs_2024)
        random.Random(seed).shuffle(shuffled)
        assert shuffled != recs_2024
        write_year(data, 2024, shuffled)
        write_year(data, 2019, list(reversed(build_2019())))
        exclusions.build_exclusions()
        assert path.read_bytes() == first, seed
    assert store_rows(data) == ROWS_2019 + ROWS_2024


# ------------------------------------------------------------------------------------------------
# The command


def test_the_command_writes_the_table_and_a_private_report(data: Path) -> None:
    write_year(data, 2019, build_2019())
    write_year(data, 2024, build_2024())
    result = CliRunner().invoke(app, ["japan", "jravan", "exclusions"])
    assert result.exit_code == 0, result.output
    assert store_rows(data) == ROWS_2019 + ROWS_2024
    written = paths.reports_dir() / "jravan-exclusions.json"
    assert written.is_file()
    parsed = json.loads(written.read_text(encoding="utf-8"))
    assert parsed["name"] == "exclusions"
    assert parsed["counts"]["rows"] == len(ROWS_2019) + len(ROWS_2024)
    assert not written.resolve().is_relative_to(REPO_ROOT)
    from_cli = store_path(data).read_bytes()
    exclusions.build_exclusions()
    assert store_path(data).read_bytes() == from_cli

    # --year limits the build.
    result = CliRunner().invoke(app, ["japan", "jravan", "exclusions", "--year", "2024"])
    assert result.exit_code == 0, result.output
    assert store_rows(data) == ROWS_2024


# ------------------------------------------------------------------------------------------------
# The coverage audit names the table (#374), its figures withheld (D21)


def test_the_exclusions_input_names_the_jravan_exclusions_table() -> None:
    inputs = {inp.name: inp for inp in cv.load_inputs()}
    supplies = inputs["exclusions"].supplies
    pairs = [(s.source, s.table, s.column) for s in supplies]
    jravan = [s for s in supplies if (s.source, s.table) == (SOURCE, TABLE)]
    assert jravan, pairs
    assert all(s.column in COLUMNS for s in jravan), pairs
    assert not any(s.table.startswith("records/") for s in supplies), pairs
    assert all(is_withheld((s.source, s.table)) for s in jravan)


def test_the_committed_coverage_doc_lists_the_table_withheld() -> None:
    text = COVERAGE_DOC.read_text(encoding="utf-8")
    match = re.search(r"^## exclusions\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, "docs/data-coverage.md has no exclusions section"
    body = match.group(1)
    assert "records/RACE/JG" not in body
    lines = [line for line in body.splitlines() if "jravan" in line and f"{TABLE}." in line]
    assert lines, "docs/data-coverage.md does not list the jravan exclusions supply"
    assert all(WITHHELD in line and not re.search(r"\d{4}-\d{2}-\d{2}|\d+ rows", line)
               for line in lines), lines  # fmt: skip


# ------------------------------------------------------------------------------------------------
# The real export (local: needs $HORSETRACK_DATA_DIR with the export and the workbook)

REAL_YEAR = 2019  # long before the holdout


@pytest.mark.local
def test_a_real_year_maps(tmp_path: Path) -> None:
    export, workbook = wb.real_export_or_skip()
    source = export / "stored" / "RACE" / RT / f"{REAL_YEAR}.jvd"
    if not source.is_file():
        pytest.skip(f"no JG file for {REAL_YEAR}")
    out = tmp_path / "out"
    report = exclusions.build_exclusions(years=[REAL_YEAR], raw=export, out=out)
    got = pq.read_table(out / FILENAME)
    check_schema(got.schema)
    df = got.to_pandas()
    n = len(df)
    assert n > 0 and report.counts["rows"] == n
    assert (df["jurisdiction"] == "jra").all()
    venues = "|".join(VENUES.values())
    race_no = r"(?:0[1-9]|[1-9]\d)"  # never 00
    assert df["race_id"].str.fullmatch(rf"jra:{REAL_YEAR}-\d\d-\d\d:(?:{venues}):{race_no}").all()
    assert set(df["entry_kind"].dropna()) <= set(ENTRY_KINDS.values())
    assert set(df["exclusion_kind"].dropna()) <= set(EXCLUSION_KINDS.values())
    assert df["horse_id"].str.fullmatch(r"\d{10}").all()
    assert df["entry_order"].notna().all() and (df["entry_order"] > 0).all()
    assert df["published_date"].notna().all()
    triples = list(zip(df["race_id"], df["horse_id"], df["entry_order"], strict=True))
    assert triples == sorted(triples)
    assert len(set(triples)) == n

    # The row count, independently of the module, from the decoded records of the same file.
    cols = records.decode(source.read_bytes(), spec.record_spec(RT, workbook))
    key = ["開催年", "開催月日", "競馬場コード", "開催回[第N回]", "開催日目[N日目]", "レース番号",
           "血統登録番号", "出馬投票受付順番"]  # fmt: skip
    wanted = [*key, "データ作成年月日", "データ区分"]
    frame = pd.DataFrame({k: records.text(cols[k]) for k in wanted})
    frame = frame.sort_values("データ作成年月日", kind="stable").drop_duplicates(key, keep="last")
    jra = np.isin(frame["競馬場コード"].to_numpy(), list(VENUES))
    # A race id needs a valid date and a positive two-digit race number (race 00 gives no row).
    day = pd.to_datetime(frame["開催年"] + frame["開催月日"], format="%Y%m%d", errors="coerce")
    number = frame["レース番号"].fillna("")
    has_race = (day.notna() & number.str.fullmatch(r"\d\d") & (number != "00")).to_numpy()
    kept = frame[jra & has_race & (frame["データ区分"].to_numpy() != "0")]
    assert n == len(kept)
    excluded = df["entry_kind"].isin(["excluded_at_close", "excluded_on_redeclaration"])
    assert df.loc[excluded, "exclusion_kind"].notna().all()
