"""Dumping every unmapped JV-Data family to Parquet (ticket #363).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jravan.dump`:
- `MAPPED: frozenset[tuple[str, str]]`, the (dataspec, record type) pairs other tickets map:
  RACE RA, SE, HR, H1, H6, O1-O6; DIFN UM, KS, CH; BLDN HN, SK, BT; MING DM, TM (#406).
- `run(*, types=None, years=None, raw=None, out=None) -> records.BuildReport`. `raw` defaults to
  `paths.raw_dir("jravan")`, `out` to `paths.interim_dir("jravan") / "records"`; the layouts come
  from `spec.spec_path()`. Only `raw/stored/<DS>/<RT>/*.jvd` is read (never `timeseries/` or any
  other top-level folder). With `types` None every folder not in `MAPPED` is read; otherwise every
  folder whose record type is in `types`, in every dataspec, mapped or not. `years` keeps
  `<YYYY>.jvd` for YYYY in `years`; `all.jvd` is always read. An `out` inside the repository is
  refused before anything is written.
- Each file becomes `out/<DS>/<RT>/<stem>.parquet`: one row per record, one column per leaf (the
  `records.decode` keys) holding `records.text` (string, null when blank); a repeated leaf is
  `list<string>`, a nested repeat `list<list<string>>`, typed so even when every value is null;
  `_file` is the path relative to `raw` (posix), `_record` the 1-based record number (an integer).
  A row group per `records.iter_file` chunk.
- `out/_dump.json`: a file whose size and mtime and the workbook's sha256 match it is skipped (its
  parquet is not touched); anything else rewrites it. A parquet that is missing is written again.
- A file that fails to decode leaves no parquet and no stray file, is a failure
  `{"file": <path relative to raw>, "record": <1-based record number>}`, is not recorded as done
  (the next run tries it again), and the other files are written.
- The report: `name` "dump"; `counts` the totals `files`, `records`, `bytes` (the .jvd bytes read)
  of the files written this run; `unmapped["<DS>/<RT>"]` the same three plus `seconds` per folder
  dumped; `failures` as above. Counts only: no record content anywhere in it.
- `horsetrack japan jravan dump [--type RT ...] [--year YYYY ...]` runs it with the defaults,
  writes the report through `records.write_build_report` (`reports/jravan-dump.json`) and exits 1
  when any file failed.

Every workbook, record type layout, item name and value here is invented and built at run time
(`jravan_workbook`); the real export and workbook are licensed (D21) and only the `local` test
reads them, into a tmp dir it removes. Expected values are written out by hand.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import env, paths
from horsetrack.core.eval.reports import LicensedOutputInRepo

# isort: split
# (its own block: the sibling helper module, then the module this ticket creates)
import jravan_workbook as wb

from horsetrack.countries.japan.jravan import dump, records, spec

CP = "cp932"
SECRET = "ZQX"  # a marker inside record content that must never reach a report or the output
LOCAL_RECORDS = 1_000

# ------------------------------------------------------------------------------------------------
# Made-up layouts: BN (flat), WC (repeats), RA, UM, O1 (for the mapped and time-series folders)

BN_LEN, WC_LEN, RA_LEN, UM_LEN, O1_LEN = 24, 31, 12, 10, 8


def bn_section(name: str = "架空名") -> wb.Section:
    """`BN`, 24 bytes: 識別子 1-2, 架空年 3-6, `name` 7-16, 架空番号 17-19, 予備 20-22."""
    fields = [("架空年", 4), (name, 10), ("架空番号", 3), ("予備", 3)]
    return wb.simple_section(1, "BN", "架空馬主", fields)


def wc_section() -> wb.Section:
    """`WC`, 31 bytes: 架空時計 repeats 3 x 4, the group 架空組 repeats 2 x (番 2, 値 2 x 2)."""
    rows = (
        wb.item(1, "識別子", 1, 2, key="●", initial="WC", desc=wb.id_desc("WC")),
        wb.item(2, "架空場", 3, 2, desc="場（架空）"),
        wb.item(3, "架空時計", 5, 4, repeat=3, desc="時計（架空）"),
        wb.group(4, "<架空組>", 17, 6, 2, desc="組（架空）"),
        wb.sub("a", "番", 1, 2, desc="番（架空）"),
        wb.sub("b", "値", 3, 2, repeat=2, desc="値（架空）"),
        wb.item(5, "予備", 29, 1, desc="予備"),
        wb.item(6, "架空区切", 30, 2, desc="改行"),
    )
    return wb.Section(2, "架空調教", WC_LEN, rows)


def write_book(data: Path, *, bn_name: str = "架空名", extra: bool = False) -> Path:
    sections = [
        bn_section(bn_name),
        wc_section(),
        wb.simple_section(3, "RA", "架空競走", [("架空日", 8)]),
        wb.simple_section(4, "UM", "架空馬", [("架空馬名", 6)]),
        wb.simple_section(5, "O1", "架空時系列", [("架空票", 4)]),
    ]
    if extra:  # a section no folder uses: the layouts above are unchanged, the bytes are not
        sections.append(wb.simple_section(6, "ZZ", "架空追加", [("架空値", 5)]))
    table = wb.Table("2001", "架空表", 1, (("1", "甲"), ("2", "乙")))
    return wb.write_workbook(data / "specs" / "jravan" / wb.WORKBOOK_NAME, sections, [table])


# ------------------------------------------------------------------------------------------------
# Made-up records


def finish(body: bytes, length: int) -> bytes:
    assert len(body) == length - 2, (len(body), length)
    return body + b"\r\n"


def bn(year: str, name: str, number: str, reserve: str = "000") -> bytes:
    nm = name.encode(CP)
    assert len(nm) <= 10, name
    body = b"BN" + year.encode() + nm + b" " * (10 - len(nm)) + number.encode() + reserve.encode()
    return finish(body, BN_LEN)


def wc(place: str, clocks: tuple[str, str, str], pairs: tuple[tuple[str, str, str], ...]) -> bytes:
    groups = b"".join((no + a + b).encode() for no, a, b in pairs)
    body = b"WC" + place.encode() + "".join(clocks).encode() + groups + b"9"
    return finish(body, WC_LEN)


def ra(date: str) -> bytes:
    return finish(b"RA" + date.encode(), RA_LEN)


def um(name: str) -> bytes:
    return finish(b"UM" + name.encode(), UM_LEN)


def o1(votes: str) -> bytes:
    return finish(b"O1" + votes.encode(), O1_LEN)


# (year, name, number) as written -> as read (records.text: trimmed, blank is null)
BN_2001 = [
    (("2001", "アイウ", "012"), ("2001", "アイウ", "012")),
    (("1999", "　カキ　", "000"), ("1999", "カキ", "000")),
    (("2001", "", "   "), ("2001", None, None)),
    (("2000", "AB  CD", " 7 "), ("2000", "AB  CD", "7")),
    (("1987", "サシスセソ", "-05"), ("1987", "サシスセソ", "-05")),
    (("    ", " タ", "999"), (None, "タ", "999")),
    (("2001", "ナニ", "100"), ("2001", "ナニ", "100")),
]
BN_2002 = [
    (("2002", "ヌネ", "001"), ("2002", "ヌネ", "001")),
    (("2002", "ノ", "002"), ("2002", "ノ", "002")),
    (("1990", "ハヒフ", "003"), ("1990", "ハヒフ", "003")),
]
BN_ALL = [
    (("1980", "マミ", "010"), ("1980", "マミ", "010")),
    (("2026", "ムメモ", "020"), ("2026", "ムメモ", "020")),
]

WC_2021 = [
    (
        ("01", ("1234", "  56", "    "), (("01", "10", "11"), ("02", "  ", "21"))),
        ("01", ["1234", "56", None], ["01", "02"], [["10", "11"], [None, "21"]]),
    ),
    (
        ("05", ("0099", "0100", "0101"), (("  ", "30", "31"), ("04", "40", "  "))),
        ("05", ["0099", "0100", "0101"], [None, "04"], [["30", "31"], ["40", None]]),
    ),
    (
        ("  ", (" 7  ", "8888", "  9 "), (("09", "  ", "  "), ("10", "52", "53"))),
        (None, ["7", "8888", "9"], ["09", "10"], [[None, None], ["52", "53"]]),
    ),
]
WC_2022 = [  # every clock blank: the column is still list<string>
    (
        ("12", ("    ", "    ", "    "), (("11", "60", "61"), ("12", "70", "71"))),
        ("12", [None, None, None], ["11", "12"], [["60", "61"], ["70", "71"]]),
    ),
    (
        ("13", ("    ", "    ", "    "), (("13", "80", "81"), ("  ", "  ", "91"))),
        ("13", [None, None, None], ["13", None], [["80", "81"], [None, "91"]]),
    ),
]

BN_COLUMNS = {"識別子", "架空年", "架空名", "架空番号", "_file", "_record"}
WC_COLUMNS = {"識別子", "架空場", "架空時計", "架空組/番", "架空組/値", "_file", "_record"}


def bn_rows(rel: str, cases: list) -> list[dict]:
    return [
        {"識別子": "BN", "架空年": y, "架空名": n, "架空番号": no, "_file": rel, "_record": i}
        for i, (_, (y, n, no)) in enumerate(cases, start=1)
    ]


def wc_rows(rel: str, cases: list) -> list[dict]:
    return [
        {
            "識別子": "WC",
            "架空場": place,
            "架空時計": clocks,
            "架空組/番": nos,
            "架空組/値": values,
            "_file": rel,
            "_record": i,
        }
        for i, (_, (place, clocks, nos, values)) in enumerate(cases, start=1)
    ]


def write_standard(root: Path, *, bn_all: bool = False) -> None:
    """DIFN/BN 2001, 2002 (and all.jvd), WOOD/WC 2021, 2022 under `root` (raw/jravan)."""
    wb.write_jvd(root / "stored/DIFN/BN/2001.jvd", [bn(*w) for w, _ in BN_2001])
    wb.write_jvd(root / "stored/DIFN/BN/2002.jvd", [bn(*w) for w, _ in BN_2002])
    if bn_all:
        wb.write_jvd(root / "stored/DIFN/BN/all.jvd", [bn(*w) for w, _ in BN_ALL])
    wb.write_jvd(root / "stored/WOOD/WC/2021.jvd", [wc(*w) for w, _ in WC_2021])
    wb.write_jvd(root / "stored/WOOD/WC/2022.jvd", [wc(*w) for w, _ in WC_2022])


# ------------------------------------------------------------------------------------------------
# Helpers


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp data dir holding the made-up workbook; the export goes under raw/jravan."""
    folder = tmp_path / "data"
    folder.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(folder))
    write_book(folder)
    return folder


def raw_of(data: Path) -> Path:
    return data / "raw" / "jravan"


def out_of(data: Path) -> Path:
    return data / "interim" / "jravan" / "records"


def outputs(out: Path) -> set[str]:
    return {p.relative_to(out).as_posix() for p in out.rglob("*.parquet")}


def rows_of(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def stamps(out: Path) -> dict[str, tuple[int, int]]:
    """Each parquet's (inode, mtime_ns): a rewrite via a tmp file and os.replace changes both."""
    return {
        p.relative_to(out).as_posix(): (p.stat().st_ino, p.stat().st_mtime_ns)
        for p in out.rglob("*.parquet")
    }


def is_text(t: pa.DataType) -> bool:
    return pa.types.is_string(t) or pa.types.is_large_string(t)


def is_list(t: pa.DataType) -> bool:
    return pa.types.is_list(t) or pa.types.is_large_list(t)


def invoke(*args: str):
    return CliRunner().invoke(app, ["japan", "jravan", "dump", *args])


def report_file(data: Path) -> Path:
    return data / "reports" / "jravan-dump.json"


def inside_repo() -> Path:
    return env.repo_root() / f".tmp-jravan-dump-{uuid.uuid4().hex[:8]}"


# ------------------------------------------------------------------------------------------------
# Tests


def test_each_unmapped_family_is_written_one_row_per_record(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_standard(raw_of(data))
    monkeypatch.setattr(records, "CHUNK_BYTES", 3 * BN_LEN)  # BN chunks of 3, WC of 2

    report = dump.run()
    out = out_of(data)
    assert report.failures == []
    assert outputs(out) == {
        "DIFN/BN/2001.parquet",
        "DIFN/BN/2002.parquet",
        "WOOD/WC/2021.parquet",
        "WOOD/WC/2022.parquet",
    }
    expected = {
        "DIFN/BN/2001.parquet": bn_rows("stored/DIFN/BN/2001.jvd", BN_2001),
        "DIFN/BN/2002.parquet": bn_rows("stored/DIFN/BN/2002.jvd", BN_2002),
        "WOOD/WC/2021.parquet": wc_rows("stored/WOOD/WC/2021.jvd", WC_2021),
        "WOOD/WC/2022.parquet": wc_rows("stored/WOOD/WC/2022.jvd", WC_2022),
    }
    for rel, want in expected.items():
        schema = pq.read_schema(out / rel)
        columns = BN_COLUMNS if rel.startswith("DIFN/BN") else WC_COLUMNS
        assert sorted(schema.names) == sorted(columns), rel
        assert rows_of(out / rel) == want, rel
        assert pa.types.is_integer(schema.field("_record").type), rel
        assert is_text(schema.field("_file").type), rel
    for name in ("識別子", "架空年", "架空名", "架空番号"):
        assert is_text(pq.read_schema(out / "DIFN/BN/2001.parquet").field(name).type), name

    # A row group per chunk: the 7-record file is not written as one table.
    assert pq.ParquetFile(out / "DIFN/BN/2001.parquet").metadata.num_row_groups >= 2
    assert (out / "_dump.json").is_file()

    # Other raw and out folders are honoured: only what that export holds lands there.
    other_raw, other_out = tmp_path / "other_raw", tmp_path / "other_out"
    wb.write_jvd(other_raw / "stored/DIFN/BN/2002.jvd", [bn(*w) for w, _ in BN_2002[:2]])
    report = dump.run(raw=other_raw, out=other_out)
    assert report.failures == []
    assert outputs(other_out) == {"DIFN/BN/2002.parquet"}
    assert rows_of(other_out / "DIFN/BN/2002.parquet") == bn_rows(
        "stored/DIFN/BN/2002.jvd", BN_2002[:2]
    )

    # Dumps are licensed output: an out folder inside the repository is refused, nothing made.
    inside = inside_repo()
    try:
        with pytest.raises((paths.DataDirInsideRepo, LicensedOutputInRepo)):
            dump.run(out=inside)
        assert not inside.exists()
    finally:
        shutil.rmtree(inside, ignore_errors=True)


def test_repeat_leaves_are_list_columns(data: Path) -> None:
    write_standard(raw_of(data))
    dump.run(types=["WC"])
    out = out_of(data)
    assert outputs(out) == {"WOOD/WC/2021.parquet", "WOOD/WC/2022.parquet"}
    for stem, cases in (("2021", WC_2021), ("2022", WC_2022)):
        path = out / "WOOD/WC" / f"{stem}.parquet"
        schema = pq.read_schema(path)
        for name in ("識別子", "架空場"):
            assert is_text(schema.field(name).type), (stem, name, schema.field(name).type)
        for name in ("架空時計", "架空組/番"):
            t = schema.field(name).type
            assert is_list(t) and is_text(t.value_type), (stem, name, t)
        nested = schema.field("架空組/値").type
        assert is_list(nested) and is_list(nested.value_type), (stem, nested)
        assert is_text(nested.value_type.value_type), (stem, nested)
        assert rows_of(path) == wc_rows(f"stored/WOOD/WC/{stem}.jvd", cases), stem


def test_the_cli_dumps_the_named_types_and_years(data: Path) -> None:
    root = raw_of(data)
    write_standard(root, bn_all=True)
    wb.write_jvd(root / "stored/DIFN/RA/2002.jvd", [ra("20021201"), ra("20021208")])
    out = out_of(data)

    result = invoke("--type", "BN", "--type", "WC", "--year", "2002", "--year", "2021")
    assert result.exit_code == 0, result.output
    assert outputs(out) == {"DIFN/BN/2002.parquet", "DIFN/BN/all.parquet", "WOOD/WC/2021.parquet"}
    assert rows_of(out / "DIFN/BN/all.parquet") == bn_rows("stored/DIFN/BN/all.jvd", BN_ALL)
    assert len(rows_of(out / "DIFN/BN/2002.parquet")) == len(BN_2002)
    assert report_file(data).is_file()

    # A year without a type: every unmapped family's file of that year, and all.jvd.
    result = invoke("--year", "2022")
    assert result.exit_code == 0, result.output
    assert outputs(out) == {
        "DIFN/BN/2002.parquet",
        "DIFN/BN/all.parquet",
        "WOOD/WC/2021.parquet",
        "WOOD/WC/2022.parquet",
    }

    # A type without a year: every file of that type.
    result = invoke("--type", "RA")
    assert result.exit_code == 0, result.output
    assert "DIFN/RA/2002.parquet" in outputs(out)
    assert "DIFN/BN/2001.parquet" not in outputs(out)

    # Neither: everything unmapped.
    result = invoke()
    assert result.exit_code == 0, result.output
    assert outputs(out) == {
        "DIFN/BN/2001.parquet",
        "DIFN/BN/2002.parquet",
        "DIFN/BN/all.parquet",
        "DIFN/RA/2002.parquet",
        "WOOD/WC/2021.parquet",
        "WOOD/WC/2022.parquet",
    }
    assert rows_of(out / "DIFN/RA/2002.parquet") == [
        {"識別子": "RA", "架空日": "20021201", "_file": "stored/DIFN/RA/2002.jvd", "_record": 1},
        {"識別子": "RA", "架空日": "20021208", "_file": "stored/DIFN/RA/2002.jvd", "_record": 2},
    ]


def test_mapped_families_are_skipped_unless_named(data: Path, tmp_path: Path) -> None:
    assert dump.MAPPED == frozenset(
        {("RACE", rt) for rt in ("RA", "SE", "HR", "H1", "H6", "O1", "O2", "O3", "O4", "O5", "O6")}
        | {("DIFN", rt) for rt in ("UM", "KS", "CH")}
        | {("BLDN", rt) for rt in ("HN", "SK", "BT")}
        | {("MING", rt) for rt in ("DM", "TM")}  # #406
    )
    root = raw_of(data)
    wb.write_jvd(root / "stored/RACE/RA/2002.jvd", [ra("20020105"), ra("20020106")])
    wb.write_jvd(root / "stored/DIFN/RA/2002.jvd", [ra("20021201")])
    wb.write_jvd(root / "stored/DIFN/UM/all.jvd", [um("ABCDEF"), um("  GH  ")])
    wb.write_jvd(root / "stored/RACE/O1/2003.jvd", [o1("0012")])
    wb.write_jvd(root / "stored/WOOD/WC/2021.jvd", [wc(*w) for w, _ in WC_2021])
    # A mapped folder whose type the workbook lacks: read, it would fail.
    wb.write_jvd(root / "stored/BLDN/HN/all.jvd", [b"HN" + b"x" * 13 + b"\r\n"])
    # Never dumped: the time series and folders outside stored/.
    wb.write_jvd(root / "timeseries/0B41/2003/20030105.jvd", [o1("0034")])
    wb.write_jvd(root / "probe/QZ/x.jvd", [b"QZ" + b"y" * 19 + b"\r\n"])
    wb.write_jvd(root / "stored_shift/WOOD/WC/2021.jvd", [wc(*w) for w, _ in WC_2021])

    default = dump.run()
    assert default.failures == []
    assert outputs(out_of(data)) == {"DIFN/RA/2002.parquet", "WOOD/WC/2021.parquet"}

    named = tmp_path / "named"
    report = dump.run(types=["RA"], out=named)
    assert report.failures == []
    assert outputs(named) == {"RACE/RA/2002.parquet", "DIFN/RA/2002.parquet"}
    assert [r["架空日"] for r in rows_of(named / "RACE/RA/2002.parquet")] == [
        "20020105",
        "20020106",
    ]
    assert {r["_file"] for r in rows_of(named / "RACE/RA/2002.parquet")} == {
        "stored/RACE/RA/2002.jvd"
    }
    assert [r["架空日"] for r in rows_of(named / "DIFN/RA/2002.parquet")] == ["20021201"]

    both = tmp_path / "both"
    report = dump.run(types=["UM", "O1"], out=both)
    assert report.failures == []
    assert outputs(both) == {"DIFN/UM/all.parquet", "RACE/O1/2003.parquet"}
    assert [r["架空馬名"] for r in rows_of(both / "DIFN/UM/all.parquet")] == ["ABCDEF", "GH"]
    assert rows_of(both / "RACE/O1/2003.parquet") == [
        {"識別子": "O1", "架空票": "0012", "_file": "stored/RACE/O1/2003.jvd", "_record": 1}
    ]
    for folder in (out_of(data), named, both):
        assert not [p for p in folder.rglob("*") if "0B41" in p.as_posix()], folder
        assert not (folder / "timeseries").exists(), folder


def test_an_unchanged_file_is_not_dumped_again_and_a_changed_workbook_redumps(
    data: Path,
) -> None:
    root, out = raw_of(data), out_of(data)
    write_standard(root)
    dump.run()
    first = stamps(out)
    assert set(first) == {
        "DIFN/BN/2001.parquet",
        "DIFN/BN/2002.parquet",
        "WOOD/WC/2021.parquet",
        "WOOD/WC/2022.parquet",
    }

    def changed(before: dict, after: dict) -> set[str]:
        assert set(before) == set(after)
        return {rel for rel in before if before[rel] != after[rel]}

    # No change: nothing rewritten, also after a run limited to one type.
    dump.run()
    assert changed(first, stamps(out)) == set()
    dump.run(types=["WC"])
    assert changed(first, stamps(out)) == set()
    dump.run()
    assert changed(first, stamps(out)) == set()

    # A file that grew: only its parquet, with the new record.
    grown = [bn(*w) for w, _ in BN_2002] + [bn("2002", "ヘホ", "004")]
    wb.write_jvd(root / "stored/DIFN/BN/2002.jvd", grown)
    dump.run()
    now = stamps(out)
    assert changed(first, now) == {"DIFN/BN/2002.parquet"}
    assert [r["架空名"] for r in rows_of(out / "DIFN/BN/2002.parquet")] == [
        "ヌネ",
        "ノ",
        "ハヒフ",
        "ヘホ",
    ]

    # Only the mtime moved (same bytes): rewritten.
    touched = root / "stored/WOOD/WC/2021.jvd"
    st = touched.stat()
    os.utime(touched, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    dump.run()
    later = stamps(out)
    assert changed(now, later) == {"WOOD/WC/2021.parquet"}

    # A deleted parquet is written again, the others are not touched.
    (out / "DIFN/BN/2001.parquet").unlink()
    dump.run()
    again = stamps(out)
    assert "DIFN/BN/2001.parquet" in again
    assert (
        changed(
            {k: v for k, v in later.items() if k != "DIFN/BN/2001.parquet"},
            {k: v for k, v in again.items() if k != "DIFN/BN/2001.parquet"},
        )
        == set()
    )
    assert len(rows_of(out / "DIFN/BN/2001.parquet")) == len(BN_2001)

    # A workbook with other bytes (a section no folder uses): every file is dumped again.
    write_book(data, extra=True)
    dump.run()
    rebuilt = stamps(out)
    assert changed(again, rebuilt) == set(again)

    # A workbook that renames a leaf: every file again, under the new name.
    write_book(data, bn_name="架空称")
    dump.run()
    renamed = stamps(out)
    assert changed(rebuilt, renamed) == set(rebuilt)
    names = set(pq.read_schema(out / "DIFN/BN/2001.parquet").names)
    assert "架空称" in names and "架空名" not in names
    assert [r["架空称"] for r in rows_of(out / "DIFN/BN/2001.parquet")] == [
        want[1] for _, want in BN_2001
    ]

    dump.run()
    assert changed(renamed, stamps(out)) == set()


def test_a_broken_file_is_reported_and_the_others_are_written(
    data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, out = raw_of(data), out_of(data)
    monkeypatch.setattr(records, "CHUNK_BYTES", 3 * BN_LEN)  # BN chunks of 3 records
    good = [bn(*w) for w, _ in BN_2001]
    wb.write_jvd(root / "stored/DIFN/BN/2001.jvd", good)
    # 2002: record 5 a byte short (its first chunk decodes, the second fails).
    short = [bn("2002", f"{SECRET}{i}", f"00{i}") for i in range(6)]
    short[4] = short[4][:-3] + b"\r\n"
    wb.write_jvd(root / "stored/DIFN/BN/2002.jvd", short)
    # 2003: record 4 of another type, the right width.
    other = [bn("2003", f"{SECRET}{i}", f"00{i}") for i in range(5)]
    other[3] = b"XZ" + other[3][2:]
    wb.write_jvd(root / "stored/DIFN/BN/2003.jvd", other)
    wb.write_jvd(root / "stored/WOOD/WC/2021.jvd", [wc(*w) for w, _ in WC_2021])

    want_failures = {("stored/DIFN/BN/2002.jvd", 5), ("stored/DIFN/BN/2003.jvd", 4)}
    for attempt in (1, 2):  # a failed file is not recorded as done: the next run tries it again
        result = invoke()
        assert result.exit_code == 1, (attempt, result.output)
        assert SECRET not in result.output, attempt
        assert outputs(out) == {"DIFN/BN/2001.parquet", "WOOD/WC/2021.parquet"}, attempt
        assert sorted(p.name for p in (out / "DIFN/BN").iterdir()) == ["2001.parquet"], attempt
        assert rows_of(out / "DIFN/BN/2001.parquet") == bn_rows("stored/DIFN/BN/2001.jvd", BN_2001)
        text = report_file(data).read_text(encoding="utf-8")
        assert SECRET not in text, attempt
        failures = json.loads(text)["failures"]
        assert {(f["file"], f["record"]) for f in failures} == want_failures, (attempt, failures)
        if attempt == 1:
            kept = stamps(out)
    assert stamps(out) == kept

    # Repaired, the file is dumped and the command succeeds.
    short[4] = bn("2002", f"{SECRET}4", "004")
    wb.write_jvd(root / "stored/DIFN/BN/2002.jvd", short)
    wb.write_jvd(root / "stored/DIFN/BN/2003.jvd", [bn("2003", "ヤ", "001")])
    result = invoke()
    assert result.exit_code == 0, result.output
    assert len(rows_of(out / "DIFN/BN/2002.parquet")) == 6
    assert json.loads(report_file(data).read_text(encoding="utf-8"))["failures"] == []


def test_the_report_holds_counts_only_and_is_refused_inside_the_repo(
    data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = raw_of(data)
    bn_cases = [(f"{1990 + i}", f"{SECRET}秘{i}", f"{i:03d}") for i in range(5)]
    wb.write_jvd(root / "stored/DIFN/BN/2001.jvd", [bn(*w) for w in bn_cases[:2]])
    wb.write_jvd(root / "stored/DIFN/BN/all.jvd", [bn(*w) for w in bn_cases[2:]])
    wb.write_jvd(root / "stored/WOOD/WC/2021.jvd", [wc(*w) for w, _ in WC_2021])
    sizes = {
        "DIFN/BN": 5 * BN_LEN,
        "WOOD/WC": len(WC_2021) * WC_LEN,
    }

    result = invoke()
    assert result.exit_code == 0, result.output
    text = report_file(data).read_text(encoding="utf-8")
    report = json.loads(text)
    assert report["name"] == "dump"
    assert report["failures"] == []
    assert report["unmapped"].keys() == {"DIFN/BN", "WOOD/WC"}
    per_type = {"DIFN/BN": (2, 5), "WOOD/WC": (1, len(WC_2021))}
    for key, (files, rows) in per_type.items():
        entry = report["unmapped"][key]
        assert (entry["files"], entry["records"], entry["bytes"]) == (files, rows, sizes[key]), key
        assert isinstance(entry["seconds"], (int, float)) and entry["seconds"] >= 0, key
    counts = report["counts"]
    assert (counts["files"], counts["records"], counts["bytes"]) == (
        3,
        5 + len(WC_2021),
        sum(sizes.values()),
    )

    # Counts only: every value a number, no record content in the report, the manifest or output.
    def numbers(value) -> bool:
        if isinstance(value, dict):
            return all(numbers(v) for v in value.values())
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    assert numbers(report["counts"]) and numbers(report["unmapped"]) and numbers(report["by_year"])
    for blob in (text, (out_of(data) / "_dump.json").read_text(encoding="utf-8"), result.output):
        assert SECRET not in blob and "秘" not in blob

    # A reports folder inside the repository is refused and nothing is created there.
    inside = inside_repo() / "reports"
    monkeypatch.setattr(paths, "reports_dir", lambda: inside)
    try:
        wb.write_jvd(root / "stored/DIFN/BN/2002.jvd", [bn(*w) for w in bn_cases[:1]])
        result = invoke()
        assert result.exit_code != 0, result.output
        assert not inside.parent.exists()
    finally:
        shutil.rmtree(inside.parent, ignore_errors=True)


# ------------------------------------------------------------------------------------------------
# The real export (local: needs $HORSETRACK_DATA_DIR with the export and the workbook)


def leaf_names(layout: spec.RecordSpec) -> set[str]:
    """The leaf column names of `layout`, from its items (reserves and the separator left out)."""

    def named(items):
        seen: dict[str, int] = {}
        for item in items:
            if item.name == "予備":
                continue
            seen[item.name] = seen.get(item.name, 0) + 1
            k = seen[item.name]
            yield (item.name if k == 1 else f"{item.name}#{k}"), item

    names: set[str] = set()
    for name, item in named(layout.items[:-1]):
        if item.items:
            names |= {f"{name}/{sub_name}" for sub_name, _ in named(item.items)}
        else:
            names.add(name)
    return names


@pytest.mark.local
def test_every_exported_stored_family_decodes(tmp_path: Path) -> None:
    export, workbook = wb.real_export_or_skip()
    raw, out = tmp_path / "raw", tmp_path / "out"
    copied: dict[str, int] = {}  # path relative to raw -> records copied
    try:
        for ds in sorted((export / "stored").iterdir()):
            if not ds.is_dir():
                continue
            for rt in sorted(ds.iterdir()):
                if not rt.is_dir():
                    continue
                files = sorted(rt.glob("*.jvd"))
                assert files, f"stored/{ds.name}/{rt.name}: no .jvd files"
                rel = f"stored/{ds.name}/{rt.name}/{files[0].name}"
                (raw / rel).parent.mkdir(parents=True)
                n = 0
                with files[0].open("rb") as src, (raw / rel).open("wb") as dst:
                    while n < LOCAL_RECORDS and (line := src.readline()):
                        if line.strip(b"\r\n"):
                            dst.write(line)
                            n += 1
                copied[rel] = n
        assert copied

        types = sorted({rel.split("/")[2] for rel in copied})
        report = dump.run(types=types, raw=raw, out=out)
        assert report.failures == [], report.failures

        problems = []
        for rel, n in copied.items():
            _, ds, rt, name = rel.split("/")
            path = out / ds / rt / f"{Path(name).stem}.parquet"
            if not path.is_file():
                problems.append(f"{ds}/{rt}: no parquet")
                continue
            columns = set(pq.read_schema(path).names)
            want = leaf_names(spec.record_spec(rt, workbook)) | {"_file", "_record"}
            if columns != want:
                problems.append(f"{ds}/{rt}: {len(columns)} columns, the layout has {len(want)}")
            table = pq.read_table(path, columns=["_file", "_record"])
            if table.column("_record").to_pylist() != list(range(1, n + 1)):
                problems.append(f"{ds}/{rt}: {table.num_rows} rows, {n} records copied")
            if set(table.column("_file").to_pylist()) != {rel}:
                problems.append(f"{ds}/{rt}: _file is not {rel}")
        assert not problems, problems
    finally:
        shutil.rmtree(raw, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
