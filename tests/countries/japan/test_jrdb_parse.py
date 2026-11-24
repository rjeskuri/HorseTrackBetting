"""JRDB fixed-width parsing (ticket #55, ported from HKTrack's `sources/jra/jrdb/parse.py`).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jrdb.parse`:
- Specs are data tables that never ship with the package: `<data dir>/jrdb/specs/<family in lower
  case>.csv` (data dir = `core.paths.data_dir()`), UTF-8, `#` lines skipped, csv header
  `name,start,length,kind,added,label`, the 2-byte terminator listed last as a `skip` field. The
  directory is looked up on every call: a changed HORSETRACK_DATA_DIR takes effect at once. Any
  family with a spec there is accepted; family names are case-insensitive. A family with no spec
  raises a LookupError, OSError or ValueError naming it.
- `spec(family)`: the fields in record order, each with `.name`, `.start` (1-based byte offset),
  `.length` (bytes), `.kind` (text, num, code, date, skip). Fields contiguous from byte 1, lengths
  positive, kinds known, names unique, else ValueError (never RecordLengthError), from `spec` and
  from `parse_bytes`. `record_length(family)`: the sum of the lengths, terminator included.
- `columns(family)`: `race_key` (bytes 1-8 as printed) when the first five fields are venue_code,
  year, kai, nichi, race_no; `meeting_key` (bytes 1-6) when only the first four are; then every
  non-skip field in spec order.
- `parse_bytes(family, data) -> pandas.DataFrame`: exactly `columns(family)`, one row per record in
  file order, a 0..n-1 index. Bytes are sliced before decoding (cp932). text and code: half- and
  full-width spaces trimmed at both ends only, blank -> None (codes keep leading zeros). num: a
  float64 column whatever the alignment, spaces dropped ('- 4' -> -4.0), blank or a bare sign ->
  NaN. date: YYYYMMDD -> `datetime.date`, blank or invalid -> None. Non-num columns are object.
  Records end in CR LF, LF or LF LF; empty lines are skipped; the last may lack its terminator. A
  record whose data is not `record_length - 2` bytes raises RecordLengthError (a ValueError) with
  `.family`, `.record` (1-based, counting records not lines), `.length` and `.expected`.
- `members(path)`: (name, raw bytes) of a .txt file, or of a .zip's .txt members in name order.
  `parse_file(path)`: the family is the file name's leading letters (`synth_2026.zip` -> SYNTH);
  members named after another family are left out; no member at all is an empty frame.

The families here (SYNTH, SYNDAY, SYNSOLO), their fields, offsets and lengths are invented: no JRDB
spec, sample or layout is in this file. Only the race-key part names come from the ported code.
Records are built by padding and cp932-encoding the values below; expected values are by hand.
"""

from __future__ import annotations

import csv
import io
import math
import shutil
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from horsetrack.core import paths

# isort: split
# (its own block: ruff ranks a module that does not exist yet as third-party)
from horsetrack.countries.japan.jrdb import parse

REPO = Path(__file__).resolve().parents[3]
NAN = math.nan
FW = "　"  # full-width (ideographic) space
KEY_PARTS = ("venue_code", "year", "kai", "nichi", "race_no")

# (name, start, length, kind): invented layouts. SYNTH is race-keyed, SYNDAY meeting-keyed,
# SYNSOLO has no key.
LAYOUTS: dict[str, list[tuple[str, int, int, str]]] = {
    "SYNTH": [
        ("venue_code", 1, 2, "skip"),
        ("year", 3, 2, "skip"),
        ("kai", 5, 1, "skip"),
        ("nichi", 6, 1, "skip"),
        ("race_no", 7, 2, "skip"),
        ("title", 9, 12, "text"),
        ("gauge", 21, 5, "num"),
        ("tag", 26, 4, "code"),
        ("stamp", 30, 8, "date"),
        ("spare", 38, 3, "skip"),
        ("tally", 41, 3, "num"),
        ("note", 44, 6, "text"),
        ("eol", 50, 2, "skip"),
    ],
    "SYNDAY": [
        ("venue_code", 1, 2, "skip"),
        ("year", 3, 2, "skip"),
        ("kai", 5, 1, "skip"),
        ("nichi", 6, 1, "skip"),
        ("held_on", 7, 8, "date"),
        ("going", 15, 3, "num"),
        ("remark", 18, 10, "text"),
        ("eol", 28, 2, "skip"),
    ],
    "SYNSOLO": [
        ("ident", 1, 6, "code"),
        ("alias", 7, 8, "text"),
        ("eol", 15, 2, "skip"),
    ],
}
RECORD_LENGTH = {"SYNTH": 51, "SYNDAY": 29, "SYNSOLO": 16}  # by hand, terminator included
COLUMNS = {
    "SYNTH": ["race_key", "title", "gauge", "tag", "stamp", "tally", "note"],
    "SYNDAY": ["meeting_key", "held_on", "going", "remark"],
    "SYNSOLO": ["ident", "alias"],
}
NUM = {"gauge", "tally", "going"}
# Labels in UTF-8 and with a quoted comma, as the real specs have them.
LABELS = {"title": "合成の題名", "gauge": "gauge, signed"}
ADDED = {"note": "2024-04-01"}

# (field -> its text as it sits in the record before right-padding with spaces, column -> value)
SYNTH_CASES: list[tuple[dict[str, str], dict[str, object]]] = [
    (
        {"key": "05263a11", "title": "Alpha", "gauge": "  56 ", "tag": "0954"}
        | {"stamp": "20261128", "spare": "zzz", "tally": "- 4", "note": "ok"},
        {"race_key": "05263a11", "title": "Alpha", "gauge": 56.0, "tag": "0954"}
        | {"stamp": date(2026, 11, 28), "tally": -4.0, "note": "ok"},
    ),
    (
        {"key": "09261202", "title": "㈱架空牧場", "gauge": "-12.3", "tag": "0012"}
        | {"stamp": "19991231", "tally": "+ 6", "note": "x y"},
        {"race_key": "09261202", "title": "㈱架空牧場", "gauge": -12.3, "tag": "0012"}
        | {"stamp": date(1999, 12, 31), "tally": 6.0, "note": "x y"},
    ),
    (
        {"key": "10262c09", "title": "  Beta Two  ", "gauge": "3.5  ", "tag": " A7 "}
        | {"stamp": "20240229", "tally": "100", "note": "  end "},
        {"race_key": "10262c09", "title": "Beta Two", "gauge": 3.5, "tag": "A7"}
        | {"stamp": date(2024, 2, 29), "tally": 100.0, "note": "end"},
    ),
    (
        {"key": "01261b05", "title": "架空" + FW + "牧場" + FW, "gauge": "    0", "tag": "1530"}
        | {"stamp": "20000101", "tally": " 7 ", "note": "末尾"},
        {"race_key": "01261b05", "title": "架空" + FW + "牧場", "gauge": 0.0, "tag": "1530"}
        | {"stamp": date(2000, 1, 1), "tally": 7.0, "note": "末尾"},
    ),
]
SYNDAY_CASES: list[tuple[dict[str, str], dict[str, object]]] = [
    (
        {"key": "05263a", "held_on": "20261128", "going": " -7", "remark": "芝やや重"},
        {"meeting_key": "05263a", "held_on": date(2026, 11, 28), "going": -7.0}
        | {"remark": "芝やや重"},
    ),
    (
        {"key": "09261b", "held_on": "20261129", "going": "12 ", "remark": "good"},
        {"meeting_key": "09261b", "held_on": date(2026, 11, 29), "going": 12.0, "remark": "good"},
    ),
]
SYNSOLO_CASES: list[tuple[dict[str, str], dict[str, object]]] = [
    ({"ident": "000123", "alias": "Gamma"}, {"ident": "000123", "alias": "Gamma"}),
    ({"ident": "A00042", "alias": "デルタ"}, {"ident": "A00042", "alias": "デルタ"}),
]
CASES = {"SYNTH": SYNTH_CASES, "SYNDAY": SYNDAY_CASES, "SYNSOLO": SYNSOLO_CASES}


# --------------------------------------------------------------------------------------------
# Building specs and records
# --------------------------------------------------------------------------------------------


def spec_csv(rows: list[tuple[str, int, int, str]]) -> str:
    buffer = io.StringIO()
    buffer.write("# synthetic: an invented layout for tests, not a JRDB spec\n")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["name", "start", "length", "kind", "added", "label"])
    for name, start, length, kind in rows:
        added = ADDED.get(name, "2020-01-01")
        writer.writerow([name, start, length, kind, added, LABELS.get(name, f"synthetic {name}")])
    return buffer.getvalue()


def write_specs(data: Path, layouts: dict[str, list[tuple[str, int, int, str]]]) -> Path:
    specs = data / "jrdb" / "specs"
    specs.mkdir(parents=True, exist_ok=True)
    for family, rows in layouts.items():
        (specs / f"{family.lower()}.csv").write_text(spec_csv(rows), encoding="utf-8")
    return specs


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "data"
    write_specs(data, LAYOUTS)
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    return data


def build(family: str, values: dict[str, str]) -> bytes:
    """One record's data bytes: each field's value cp932-encoded and right-padded with spaces."""
    layout = LAYOUTS[family]
    assert not set(values) - {"key"} - {name for name, *_ in layout}, values
    key = values.get("key", "")
    out = bytearray()
    for name, start, length, _ in layout[:-1]:  # the terminator is not data
        assert len(out) == start - 1, (family, name)
        if name in KEY_PARTS:
            raw = key[start - 1 : start - 1 + length].encode("ascii")
        else:
            raw = values.get(name, "").encode("cp932")
        assert len(raw) <= length, (family, name, raw)
        out += raw.ljust(length, b" ")
    assert len(out) == RECORD_LENGTH[family] - 2, family
    return bytes(out)


def lines(records: list[bytes], term: bytes = b"\r\n", final: bool = True) -> bytes:
    return term.join(records) + (term if final else b"")


def records_of(family: str, cases: list[tuple[dict[str, str], dict[str, object]]]) -> bytes:
    return lines([build(family, raw) for raw, _ in cases])


def typed_equal(got: object, want: object) -> bool:
    if isinstance(want, float):
        if math.isnan(want):
            return isinstance(got, float) and math.isnan(got)
        return type(got) is float and got == want
    if want is None:
        return got is None
    return type(got) is type(want) and got == want


def assert_rows(frame: object, family: str, expected: list[dict[str, object]]) -> None:
    """`frame` has the family's columns and dtypes, a 0..n-1 index and exactly the typed values."""
    assert isinstance(frame, pd.DataFrame)
    assert frame.columns.tolist() == COLUMNS[family]
    assert frame.index.tolist() == list(range(len(expected)))
    dtypes = {c: str(frame[c].dtype) for c in COLUMNS[family]}
    assert dtypes == {c: "float64" if c in NUM else "object" for c in COLUMNS[family]}
    wrong = {}
    for row, want in enumerate(expected, start=1):
        assert set(want) == set(COLUMNS[family]), want
        for column, value in want.items():
            got = frame[column].tolist()[row - 1]
            if not typed_equal(got, value):
                wrong[f"row {row}.{column}"] = (got, value)
    assert wrong == {}, wrong


def refusal(call) -> BaseException | None:
    try:
        call()
    except Exception as exc:
        return exc
    return None


def with_row(family: str, old: str, *new: tuple[str, int, int, str]) -> list:
    """The family's layout with row `old` replaced by `new` (one or more rows)."""
    out = []
    for row in LAYOUTS[family]:
        out.extend(new if row[0] == old else [row])
    assert out != LAYOUTS[family]
    return out


# --------------------------------------------------------------------------------------------
# Typed values
# --------------------------------------------------------------------------------------------


def test_a_synthetic_record_parses_to_typed_values(data_dir: Path) -> None:
    """Every kind: text and code as str (codes keep leading zeros), num as float64 whatever the
    alignment or sign spacing, date as datetime.date, skip as no column; race_key and meeting_key
    as printed (the hex nichi kept), no key for a family that has none; an empty file is an empty
    frame with the family's columns."""
    with pytest.raises(UnicodeEncodeError):  # the fixture really needs cp932, not Shift_JIS
        "㈱".encode("shift_jis")
    for family, layout in LAYOUTS.items():
        assert [(f.name, f.start, f.length, f.kind) for f in parse.spec(family)] == layout
        assert parse.record_length(family) == RECORD_LENGTH[family]
        assert parse.columns(family) == COLUMNS[family]
    for family, cases in CASES.items():
        frame = parse.parse_bytes(family, records_of(family, cases))
        assert_rows(frame, family, [want for _, want in cases])
        assert_rows(parse.parse_bytes(family, b""), family, [])


def test_offsets_are_bytes_not_characters(data_dir: Path) -> None:
    """Double-byte text (cp932 full-width, ㈱ and ㈲, half-width kana) in a field before others, in
    a skip field and in the last field shifts nothing: each field is the bytes at its offset."""
    key = "07264612"
    variants = [  # title, spare (skip), note -> title, note
        ("ABCDEFGHIJKL", "   ", "tail", "ABCDEFGHIJKL", "tail"),
        ("あいうえおか", "   ", "終了", "あいうえおか", "終了"),
        ("㈱㈲テスト", FW + " ", "ｶﾅｶﾅ", "㈱㈲テスト", "ｶﾅｶﾅ"),
        ("あ" + FW * 5, FW + " ", "末" + FW * 2, "あ", "末"),
        ("AB㈱CDあい", "x" + FW, "かな", "AB㈱CDあい", "かな"),
    ]
    tail = {"gauge": " 1234", "tag": "0042", "stamp": "20261231", "tally": "-99"}
    records = [
        build("SYNTH", {"key": key, "title": title, "spare": spare, "note": note} | tail)
        for title, spare, note, _, _ in variants
    ]
    expected = [
        {"race_key": key, "title": title, "gauge": 1234.0, "tag": "0042"}
        | {"stamp": date(2026, 12, 31), "tally": -99.0, "note": note}
        for _, _, _, title, note in variants
    ]
    assert_rows(parse.parse_bytes("SYNTH", lines(records)), "SYNTH", expected)


# --------------------------------------------------------------------------------------------
# Record framing
# --------------------------------------------------------------------------------------------


def test_crlf_lf_and_double_lf_records_parse_alike(data_dir: Path) -> None:
    records = [build("SYNTH", raw) for raw, _ in SYNTH_CASES]
    base = parse.parse_bytes("SYNTH", lines(records))
    assert_rows(base, "SYNTH", [want for _, want in SYNTH_CASES])
    r0, r1, r2, r3 = records
    variants = {
        "LF": lines(records, b"\n"),
        "LF LF": lines(records, b"\n\n"),
        "CR LF, last unterminated": lines(records, b"\r\n", final=False),
        "LF, last unterminated": lines(records, b"\n", final=False),
        "LF LF, last unterminated": lines(records, b"\n\n", final=False),
        "empty lines around": b"\r\n" + lines(records, b"\r\n\r\n") + b"\r\n",
        "mixed": r0 + b"\r\n" + r1 + b"\n" + r2 + b"\n\n" + r3,
    }
    differ = {}
    for name, data in variants.items():
        try:
            pd.testing.assert_frame_equal(parse.parse_bytes("SYNTH", data), base)
        except (AssertionError, ValueError) as exc:
            differ[name] = f"{type(exc).__name__}: {exc}"[:300]
    assert differ == {}, differ


def test_a_record_of_the_wrong_length_raises(data_dir: Path) -> None:
    """A short, long or cut-off record, or a file of another family, is refused with the family,
    the 1-based record number (records, not lines), its data length and the expected one."""
    r = [build("SYNTH", raw) for raw, _ in SYNTH_CASES]
    day = [build("SYNDAY", raw) for raw, _ in SYNDAY_CASES]
    cases = {  # name -> (family, data, record, length, expected)
        "one byte short": ("SYNTH", lines([r[0], r[1][:-1], r[2]]), 2, 48, 49),
        "one byte long": ("SYNTH", lines([r[0], r[1], r[2] + b" "]), 3, 50, 49),
        "cut off mid-record": ("SYNTH", lines([r[0], r[1], r[2][:20]], final=False), 3, 20, 49),
        "the first record": ("SYNTH", lines([r[0][:-3], r[1]]), 1, 46, 49),
        "LF LF, counted by record": ("SYNTH", lines([r[0], r[1], r[2][:-1]], b"\n\n"), 3, 48, 49),
        "SYNDAY file read as SYNTH": ("SYNTH", lines(day), 1, 27, 49),
        "SYNTH file read as SYNDAY": ("SYNDAY", lines(r), 1, 49, 27),
    }
    wrong = {}
    for name, (family, data, record, length, expected) in cases.items():
        exc = refusal(lambda family=family, data=data: parse.parse_bytes(family, data))
        if not isinstance(exc, parse.RecordLengthError) or not isinstance(exc, ValueError):
            wrong[name] = repr(exc)
            continue
        got = (exc.family, exc.record, exc.length, exc.expected)
        if got != (family, record, length, expected):
            wrong[name] = got
    assert wrong == {}, wrong


# --------------------------------------------------------------------------------------------
# Blanks
# --------------------------------------------------------------------------------------------


def test_blank_fields_become_none_or_nan(data_dir: Path) -> None:
    """Blank text, code and date -> None, blank num -> NaN (half- or full-width spaces); a bare
    sign is NaN and an impossible date None; zeros are values, not blanks. An all-blank num column
    stays float64."""
    blank = {"title": None, "gauge": NAN, "tag": None, "stamp": None, "tally": NAN, "note": None}
    synth = [
        ({"key": "05263a11"}, {"race_key": "05263a11"} | blank),
        ({"key": "05263a12", "title": FW * 6, "note": FW * 3}, {"race_key": "05263a12"} | blank),
        (
            {"key": "05263a13", "gauge": "  -  ", "tally": " + ", "stamp": "20261301"},
            {"race_key": "05263a13"} | blank,
        ),
        ({"key": "05263a14", "stamp": "20230229"}, {"race_key": "05263a14"} | blank),
        ({"key": "05263a15", "stamp": "2026-1-1"}, {"race_key": "05263a15"} | blank),
        (
            {"key": "05263a16", "title": "0", "gauge": "00000", "tag": "0000"}
            | {"stamp": "20000101", "tally": "  0", "note": "0"},
            {"race_key": "05263a16", "title": "0", "gauge": 0.0, "tag": "0000"}
            | {"stamp": date(2000, 1, 1), "tally": 0.0, "note": "0"},
        ),
    ]
    day = [
        (
            {"key": "05263a", "remark": FW * 5},
            {"meeting_key": "05263a", "held_on": None, "going": NAN, "remark": None},
        ),
        (
            {"key": "05263b", "held_on": "20261128", "going": "  0", "remark": "0"},
            {"meeting_key": "05263b", "held_on": date(2026, 11, 28), "going": 0.0, "remark": "0"},
        ),
    ]
    solo = [({"alias": FW * 4}, {"ident": None, "alias": None})]
    for family, cases in {"SYNTH": synth, "SYNDAY": day, "SYNSOLO": solo}.items():
        frame = parse.parse_bytes(family, records_of(family, cases))
        assert_rows(frame, family, [want for _, want in cases])
    only_blank = parse.parse_bytes("SYNTH", records_of("SYNTH", synth[:2]))
    assert_rows(only_blank, "SYNTH", [want for _, want in synth[:2]])


# --------------------------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------------------------


def _refused_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layouts: dict[str, list]
) -> dict[str, object]:
    """name -> the exception `spec` raised for that layout (written as its own family), checked
    to be the same from `parse_bytes` on empty data and on one record of the summed length."""
    problems: dict[str, object] = {}
    for i, (name, layout) in enumerate(layouts.items()):
        family = "SYNBAD" + chr(ord("A") + i)
        data = tmp_path / f"bad{i}"
        write_specs(data, {family: layout})
        monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
        record = b"x" * (sum(length for _, _, length, _ in layout) - 2) + b"\r\n"
        for how, call in {
            "spec": lambda family=family: parse.spec(family),
            "parse_bytes(empty)": lambda family=family: parse.parse_bytes(family, b""),
            "parse_bytes(record)": lambda family=family, r=record: parse.parse_bytes(family, r),
        }.items():
            exc = refusal(call)
            if not isinstance(exc, ValueError) or isinstance(exc, parse.RecordLengthError):
                problems[f"{name}: {how}"] = repr(exc)
            elif how == "spec":
                problems[name] = exc
    return problems


def test_a_spec_with_overlapping_fields_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A field that starts before the previous one ends is a ValueError naming one of the two."""
    overlapping = {  # name -> (layout, the two fields that overlap)
        "gauge starts inside title": (
            with_row("SYNTH", "gauge", ("gauge", 20, 5, "num")),
            ("title", "gauge"),
        ),
        "a field inside another": (
            with_row("SYNTH", "title", ("title", 9, 12, "text"), ("inner", 12, 2, "text")),
            ("title", "inner"),
        ),
        "key parts overlap": (
            with_row("SYNTH", "race_no", ("race_no", 6, 2, "skip")),
            ("nichi", "race_no"),
        ),
        "two fields at one start": (
            with_row("SYNTH", "tag", ("tag", 21, 4, "code")),
            ("gauge", "tag"),
        ),
        "the last field runs into the terminator": (
            with_row("SYNTH", "note", ("note", 44, 7, "text")),
            ("note", "eol"),
        ),
    }
    found = _refused_specs(tmp_path, monkeypatch, {n: lay for n, (lay, _) in overlapping.items()})
    wrong = {}
    for name, (_, pair) in overlapping.items():
        exc = found.pop(name, None)
        if exc is None or not any(field in str(exc) for field in pair):
            wrong[name] = f"{exc!r} names neither of {pair}"
    assert found == {} and wrong == {}, (found, wrong)


def test_a_spec_with_gaps_or_bad_rows_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fields must cover the record from byte 1 without gaps, with positive lengths, known kinds
    and unique names; anything else is a ValueError, never a silent mis-slice."""
    bad = {
        "a gap after title": with_row("SYNTH", "title", ("title", 9, 11, "text")),
        "not from byte 1": LAYOUTS["SYNTH"][1:],
        "a zero-length field": with_row(
            "SYNTH", "spare", ("hollow", 38, 0, "text"), ("spare", 38, 3, "skip")
        ),
        "an unknown kind": with_row("SYNTH", "tally", ("tally", 41, 3, "integer")),
        "a duplicate name": with_row("SYNTH", "note", ("title", 44, 6, "text")),
    }
    found = _refused_specs(tmp_path, monkeypatch, bad)
    not_refused = {k: v for k, v in found.items() if k not in bad}
    assert not_refused == {} and set(found) == set(bad), found


def test_specs_load_from_the_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Specs come from `<HORSETRACK_DATA_DIR>/jrdb/specs/`, looked up at call time: pointing the
    variable at another data dir changes the layout at once, and one without the family's spec
    refuses the family by name."""
    first, second, bare = tmp_path / "first", tmp_path / "second", tmp_path / "bare"
    write_specs(first, LAYOUTS)
    short = [*LAYOUTS["SYNTH"][:5], ("flag", 9, 3, "code"), ("eol", 12, 2, "skip")]
    write_specs(second, {"SYNTH": short})
    (bare / "jrdb" / "specs").mkdir(parents=True)

    monkeypatch.setenv(paths.DATA_DIR_VAR, str(first))
    assert parse.record_length("SYNTH") == RECORD_LENGTH["SYNTH"]
    assert parse.columns("synth") == COLUMNS["SYNTH"]
    raw, want = SYNTH_CASES[1]
    assert_rows(parse.parse_bytes("synth", lines([build("SYNTH", raw)])), "SYNTH", [want])

    monkeypatch.setenv(paths.DATA_DIR_VAR, str(second))
    assert parse.record_length("SYNTH") == 13
    assert parse.columns("SYNTH") == ["race_key", "flag"]
    frame = parse.parse_bytes("SYNTH", b"05263a11X9 \r\n09261202007\r\n")
    assert frame.to_dict("list") == {"race_key": ["05263a11", "09261202"], "flag": ["X9", "007"]}

    monkeypatch.setenv(paths.DATA_DIR_VAR, str(bare))
    exc = refusal(lambda: parse.spec("SYNTH"))
    assert isinstance(exc, (LookupError, OSError, ValueError)), repr(exc)
    assert "synth" in str(exc).lower(), str(exc)


def test_no_spec_table_or_jrdb_file_is_in_the_source_tree() -> None:
    """JRDB's spec tables and files are licensed: no specs directory, table or data file sits under
    a jrdb package in src/ (real specs live in the data dir; the only spec in git is this file's
    invented one)."""
    src = REPO / "src"
    package = Path(parse.__file__).resolve().parent
    assert package.is_relative_to(src) and package.name == "jrdb"
    found = sorted(
        str(p.relative_to(REPO))
        for p in src.rglob("*")
        if "jrdb" in p.relative_to(src).parts[:-1]
        and (
            (p.is_dir() and p.name.lower() == "specs")
            or p.suffix.lower() in {".csv", ".tsv", ".txt", ".zip", ".lzh"}
        )
    )
    assert found == []


# --------------------------------------------------------------------------------------------
# Files and archives
# --------------------------------------------------------------------------------------------


def test_zip_members_are_read(data_dir: Path, tmp_path: Path) -> None:
    """`members` yields a .zip's .txt members as raw bytes in name order whatever the order in the
    archive; `parse_file` takes the family from the file name, reads its members in that order and
    leaves out the other family's; a .txt file reads the same way."""
    synth = [build("SYNTH", raw) for raw, _ in SYNTH_CASES]
    want = [w for _, w in SYNTH_CASES]
    first = lines(synth[:3])  # CR LF, with the cp932-only ㈱ in record 2
    second = lines(synth[3:], b"\n", final=False)
    day = records_of("SYNDAY", SYNDAY_CASES)
    pack = tmp_path / "synth_2026.zip"
    with zipfile.ZipFile(pack, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("synth261129.txt", second)
        archive.writestr("notes.md", b"not a data file")
        archive.writestr("synday261128.txt", day)
        archive.writestr("synth261128.txt", first)

    assert [tuple(m) for m in parse.members(pack)] == [
        ("synday261128.txt", day),
        ("synth261128.txt", first),
        ("synth261129.txt", second),
    ]
    assert_rows(parse.parse_file(pack), "SYNTH", want)

    as_day = tmp_path / "synday_2026.zip"
    shutil.copy(pack, as_day)
    assert_rows(parse.parse_file(as_day), "SYNDAY", [w for _, w in SYNDAY_CASES])

    txt = tmp_path / "synth261128.txt"
    txt.write_bytes(first)
    assert [tuple(m) for m in parse.members(txt)] == [("synth261128.txt", first)]
    assert_rows(parse.parse_file(txt), "SYNTH", want[:3])

    empty = tmp_path / "synth_2025.zip"
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("notes.md", b"no member of the family")
    assert_rows(parse.parse_file(empty), "SYNTH", [])
