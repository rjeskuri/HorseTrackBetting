"""Decoding JV-Data records into named byte arrays, and the shared output helpers (ticket #362).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jravan.records`:
- `decode(data, spec) -> dict[str, ndarray]`: one `S<length>` array per leaf, keyed by the item's
  name (top level) or `group/sub`, shaped (n,), (n, repeat) or (n, repeat, sub-repeat). Items named
  `予備` (at any level) and the record separator (the spec's last item) are left out. A name that
  occurs again at the same level is `name#2`, `name#3`. Records split on LF, a CR before it is
  dropped, empty lines are skipped, the last record may lack its separator.
- A record that is not `spec.length - 2` bytes is a `RecordLengthError` (a ValueError) with
  `.record_type`, `.record` (1-based, counting records, across the whole file for `iter_file`),
  `.length` and `.expected`, all named in its message. A record of the right width that does not
  start with `spec.record_type` is a ValueError (not a RecordLengthError).
- `iter_file(path, spec, *, chunk_records)` yields `decode`-shaped dicts of at most
  `chunk_records` records, lazily (a bad record later in the file does not stop earlier chunks).
- `text`, `integer`, `scaled`: element-wise on an `S` array of any shape, the shape kept.
- `latest_versions`, `BuildReport`, `write_parquet`, `write_build_report` as in the ticket.

Every record type, item name, position and value here is invented; the layouts are built as
`RecordSpec`/`Item` directly, no workbook. Expected values are written out by hand.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core import env, paths
from horsetrack.core.eval.reports import LicensedOutputInRepo
from horsetrack.countries.japan.jravan import records
from horsetrack.countries.japan.jravan.spec import Item, RecordSpec

CP = "cp932"

# ------------------------------------------------------------------------------------------------
# Made-up layouts


def it(name: str, start: int, length: int, repeat: int = 1, items: tuple[Item, ...] = ()) -> Item:
    return Item(name, start, length, repeat, False, None, "", items)


def spec_xa() -> RecordSpec:
    """`XA`, 21 bytes (19 data bytes): a flat layout with a reserve item and the separator."""
    return RecordSpec(
        "XA",
        "架空フラット",
        21,
        (
            it("種別", 1, 2),
            it("架空年", 3, 4),
            it("架空月日", 7, 4),
            it("架空名", 11, 6),
            it("予備", 17, 3),
            it("架空区切", 20, 2),
        ),
    )


def spec_xc() -> RecordSpec:
    """`XC`, 14 bytes (12 data bytes): another flat layout, other names and positions."""
    return RecordSpec(
        "XC",
        "架空その二",
        14,
        (
            it("種別", 1, 2),
            it("場コード", 3, 2),
            it("予備", 5, 1),
            it("番号", 6, 7),
            it("改行", 13, 2),
        ),
    )


def spec_xb() -> RecordSpec:
    """`XB`, 51 bytes (49 data bytes): a repeated group, a repeated top-level item and a group
    whose sub-item repeats."""
    return RecordSpec(
        "XB",
        "架空繰返し",
        51,
        (
            it("種別", 1, 2),
            it("頭数", 3, 2),
            it(
                "単勝オッズ",
                5,
                8,
                repeat=3,
                items=(it("馬番", 1, 2), it("オッズ", 3, 4), it("人気", 7, 2)),
            ),
            it("着順", 29, 2, repeat=4),
            it("票数", 37, 6, repeat=2, items=(it("組番", 1, 2), it("値", 3, 2, repeat=2))),
            it("予備", 49, 1),
            it("区切", 50, 2),
        ),
    )


def join(record_type: str, spec: RecordSpec, *parts: bytes) -> bytes:
    """One record's data bytes (no separator); fails if they are not the spec's data width."""
    body = record_type.encode("ascii") + b"".join(parts)
    assert len(body) == spec.length - 2, (len(body), spec.length - 2)
    return body


def xa(year: str, mmdd: str, name: str, reserve: bytes = b"000") -> bytes:
    nm = name.encode(CP)
    return join("XA", spec_xa(), year.encode(), mmdd.encode(), nm + b" " * (6 - len(nm)), reserve)


def xa_rows() -> list[bytes]:
    return [
        xa("2026", "1128", "アイ"),
        xa("2010", "0105", "ウエオ"),
        xa("2019", "0731", "カ", b"999"),
        xa("2001", "1201", "キクケ"),
        xa("2024", "0229", "コサ"),
        xa("2015", "0606", "シ"),
        xa("2008", "0912", "スセソ"),
    ]


def lines(rows: list[bytes], sep: bytes = b"\r\n") -> bytes:
    return b"".join(r + sep for r in rows)


def assert_array(arr: np.ndarray, width: int, expected: list) -> None:
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.dtype(f"S{width}"), arr.dtype
    assert arr.shape == np.asarray(expected, dtype=object).shape, arr.shape
    assert arr.tolist() == expected


# ------------------------------------------------------------------------------------------------
# decode


@pytest.mark.parametrize("layout", ["XA", "XC"])
def test_fixed_width_records_decode_into_named_byte_arrays(layout: str) -> None:
    if layout == "XA":
        spec = spec_xa()
        data = lines([xa("2026", "1128", "アイ"), xa("2010", "0105", "ウエオ")])
        expected = {
            "種別": (2, [b"XA", b"XA"]),
            "架空年": (4, [b"2026", b"2010"]),
            "架空月日": (4, [b"1128", b"0105"]),
            "架空名": (6, ["アイ".encode(CP) + b"  ", "ウエオ".encode(CP)]),
        }
    else:
        spec = spec_xc()
        rows = [
            join("XC", spec, b"08", b" ", b"0000123"),
            join("XC", spec, b"30", b"9", b"  -45 6"),
        ]
        data = lines(rows)
        expected = {
            "種別": (2, [b"XC", b"XC"]),
            "場コード": (2, [b"08", b"30"]),
            "番号": (7, [b"0000123", b"  -45 6"]),
        }
    out = records.decode(data, spec)
    assert set(out) == set(expected)
    for key, (width, values) in expected.items():
        assert_array(out[key], width, values)


def xb(n: int) -> bytes:
    """Record n of `XB`: every value depends on n and its repeat index, so none is constant."""
    spec = spec_xb()
    group = b"".join(
        f"{r + 1:02d}".encode() + f"{n}{r}{r}{n}".encode() + f"{(n + r) % 100:02d}".encode()
        for r in range(3)
    )
    places = b"".join(f"{n}{p}".encode() for p in range(4))
    votes = b"".join(
        f"{n}{v}".encode()
        + b"".join(f"{v}{s}".encode() if n % 2 else f"{s}{n}".encode() for s in range(2))
        for v in range(2)
    )
    return join("XB", spec, f"{10 + n:02d}".encode(), group, places, votes, b"Z")


def test_repeat_groups_and_repeated_items_decode_by_record_and_repeat() -> None:
    ns = [1, 2, 7]
    out = records.decode(lines([xb(n) for n in ns]), spec_xb())

    def v(s: str) -> bytes:
        return s.encode()

    expected = {
        "種別": (2, [b"XB"] * 3),
        "頭数": (2, [v(f"{10 + n:02d}") for n in ns]),
        "単勝オッズ/馬番": (2, [[v(f"{r + 1:02d}") for r in range(3)] for n in ns]),
        "単勝オッズ/オッズ": (4, [[v(f"{n}{r}{r}{n}") for r in range(3)] for n in ns]),
        "単勝オッズ/人気": (2, [[v(f"{(n + r) % 100:02d}") for r in range(3)] for n in ns]),
        "着順": (2, [[v(f"{n}{p}") for p in range(4)] for n in ns]),
        "票数/組番": (2, [[v(f"{n}{g}") for g in range(2)] for n in ns]),
        "票数/値": (
            2,
            [
                [[v(f"{g}{s}") if n % 2 else v(f"{s}{n}") for s in range(2)] for g in range(2)]
                for n in ns
            ],
        ),
    }
    assert set(out) == set(expected)
    for key, (width, values) in expected.items():
        assert_array(out[key], width, values)
    assert out["単勝オッズ/オッズ"].shape == (3, 3)
    assert out["着順"].shape == (3, 4)
    assert out["票数/値"].shape == (3, 2, 2)


def test_iter_file_streams_whole_records_in_chunks(tmp_path: Path) -> None:
    spec = spec_xa()
    rows = xa_rows()
    path = tmp_path / "XA2026.jvd"
    path.write_bytes(lines(rows))
    whole = records.decode(path.read_bytes(), spec)

    for size, lengths in ((3, [3, 3, 1]), (1, [1] * 7), (7, [7]), (100, [7])):
        chunks = list(records.iter_file(path, spec, chunk_records=size))
        assert [len(c["架空年"]) for c in chunks] == lengths, size
        for key in whole:
            joined = np.concatenate([c[key] for c in chunks])
            assert joined.dtype == whole[key].dtype
            assert joined.tolist() == whole[key].tolist(), (size, key)
        assert all(set(c) == set(whole) for c in chunks)
    assert whole["架空年"].tolist() == [r[2:6] for r in rows]

    # Lazy: a bad fifth record does not stop the first chunk, and its number counts the file.
    bad = rows[:4] + [rows[4][:-1]] + rows[5:]
    path.write_bytes(lines(bad))
    stream = records.iter_file(path, spec, chunk_records=3)
    first = next(stream)
    assert first["架空年"].tolist() == [b"2026", b"2010", b"2019"]
    with pytest.raises(records.RecordLengthError) as caught:
        next(stream)
    assert (caught.value.record, caught.value.length, caught.value.expected) == (5, 18, 19)


def test_reserve_items_and_the_separator_are_left_out() -> None:
    spec = RecordSpec(
        "XR",
        "架空予備",
        20,
        (
            it("種別", 1, 2),
            it("予備", 3, 2),
            it("組", 5, 5, repeat=2, items=(it("値", 1, 3), it("予備", 4, 2))),
            it("番", 15, 2),
            it("予備", 17, 2),
            it("架空改行", 19, 2),
        ),
    )
    data = lines([join("XR", spec, b"rr", b"abcRRdefRR", b"12", b"RR")])
    out = records.decode(data, spec)
    assert set(out) == {"種別", "組/値", "番"}
    assert_array(out["組/値"], 3, [[b"abc", b"def"]])
    assert_array(out["番"], 2, [b"12"])
    assert not [k for k in out if "予備" in k or "改行" in k]


def test_a_name_repeated_at_one_level_gets_an_occurrence_suffix() -> None:
    spec = RecordSpec(
        "XN",
        "架空重複",
        22,
        (
            it("種別", 1, 2),
            it("コード", 3, 2),
            it("名", 5, 2),
            it("コード", 7, 2),
            it("群", 9, 6, items=(it("コード", 1, 2), it("値", 3, 2), it("値", 5, 2))),
            it("コード", 15, 3),
            it("末", 18, 3),
            it("区切", 21, 2),
        ),
    )
    data = lines(
        [
            join("XN", spec, b"c1", b"n1", b"c2", b"g1v1v2", b"c3x", b"end"),
            join("XN", spec, b"C1", b"N1", b"C2", b"G1V1V2", b"C3X", b"END"),
        ]
    )
    out = records.decode(data, spec)
    assert set(out) == {
        "種別",
        "コード",
        "名",
        "コード#2",
        "群/コード",
        "群/値",
        "群/値#2",
        "コード#3",
        "末",
    }
    assert_array(out["コード"], 2, [b"c1", b"C1"])
    assert_array(out["コード#2"], 2, [b"c2", b"C2"])
    assert_array(out["コード#3"], 3, [b"c3x", b"C3X"])
    assert_array(out["群/コード"], 2, [b"g1", b"G1"])
    assert_array(out["群/値"], 2, [b"v1", b"V1"])
    assert_array(out["群/値#2"], 2, [b"v2", b"V2"])


def test_a_record_of_the_wrong_width_is_a_record_length_error() -> None:
    spec = spec_xa()
    good = xa_rows()
    cases = {  # name -> (data, record number, its length)
        "short second": (lines([good[0], good[1][:-3], good[2]]), 2, 16),
        "long third": (lines([good[0], good[1], good[2] + b"12345"]), 3, 24),
        "cut-off last": (lines(good[:3]) + good[3][:10], 4, 10),
        "short first": (lines([good[0][:4]]), 1, 4),
    }
    for name, (data, record, length) in cases.items():
        with pytest.raises(records.RecordLengthError) as caught:
            records.decode(data, spec)
        exc = caught.value
        assert isinstance(exc, ValueError)
        assert (exc.record_type, exc.record, exc.length, exc.expected) == (
            "XA",
            record,
            length,
            19,
        ), name
        message = str(exc)
        for part in ("XA", str(record), str(length), "19"):
            assert part in message, (name, part, message)


def test_a_record_of_another_type_is_an_error() -> None:
    spec = spec_xa()
    good = xa_rows()
    other = b"XZ" + good[1][2:]
    assert len(other) == len(good[1])
    for data in (lines([good[0], other, good[2]]), lines([other]), lines([b"xa" + good[0][2:]])):
        with pytest.raises(ValueError) as caught:
            records.decode(data, spec)
        assert not isinstance(caught.value, records.RecordLengthError)


def test_crlf_lf_blank_lines_and_a_missing_last_separator_are_read() -> None:
    spec = spec_xa()
    r = xa_rows()
    variants = {
        "crlf": lines(r[:4]),
        "lf": lines(r[:4], b"\n"),
        "mixed": r[0] + b"\r\n" + r[1] + b"\n" + r[2] + b"\r\n" + r[3] + b"\n",
        "blank lines": b"\r\n"
        + r[0]
        + b"\r\n\r\n"
        + r[1]
        + b"\n\n"
        + r[2]
        + b"\r\n"
        + r[3]
        + b"\r\n\n",
        "no last separator": lines(r[:3]) + r[3],
        "lf lf": lines(r[:4], b"\n\n"),
    }
    expected = [x[2:6] for x in r[:4]]
    assert expected == [b"2026", b"2010", b"2019", b"2001"]
    for name, data in variants.items():
        out = records.decode(data, spec)
        assert out["架空年"].tolist() == expected, name
        assert out["架空月日"].tolist() == [x[6:10] for x in r[:4]], name
        assert out["架空名"].tolist() == [x[10:16] for x in r[:4]], name


# ------------------------------------------------------------------------------------------------
# Converters


def sarr(values: list, width: int) -> np.ndarray:
    return np.array(values, dtype=f"S{width}")


def test_text_decodes_cp932_and_trims_half_and_full_width_spaces() -> None:
    cases = [  # raw text (cp932) -> decoded
        ("ディープ　", "ディープ"),
        ("  ディープ", "ディープ"),
        ("　キタサン　ブラック　 ", "キタサン　ブラック"),
        ("ア イ", "ア イ"),
        ("ABC   ", "ABC"),
        ("㈱架空", "㈱架空"),
        ("        ", None),
        ("　　　　", None),
        ("", None),
    ]
    raw = sarr([c[0].encode(CP) for c in cases], 24)
    out = records.text(raw)
    assert out.dtype == object
    assert out.shape == (len(cases),)
    assert out.tolist() == [c[1] for c in cases]

    grid = sarr([["カ　".encode(CP), b"  "], [b"", " キ".encode(CP)]], 4)
    got = records.text(grid)
    assert got.shape == (2, 2)
    assert got.tolist() == [["カ", None], [None, "キ"]]


def test_integer_and_scaled_read_digits_and_null_anything_else() -> None:
    nan = math.nan
    cases = [  # raw -> integer
        (b"0123", 123.0),
        (b" -12", -12.0),
        (b"  56", 56.0),
        (b"0000", 0.0),
        (b"9999", 9999.0),
        (b"0007", 7.0),
        (b"-001", -1.0),
        (b"----", nan),
        (b"****", nan),
        (b"    ", nan),
        (b"12A4", nan),
        (b"-   ", nan),
        (b"1.23", nan),
    ]
    raw = sarr([c[0] for c in cases], 4)
    got = records.integer(raw)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, np.array([c[1] for c in cases]))

    scaled_cases = [  # (raw, places) -> value
        ((b"0123", 1), 12.3),
        ((b"1234", 2), 12.34),
        ((b"0050", 1), 5.0),
        ((b" 987", 3), 0.987),
        ((b" -45", 1), -4.5),
        ((b"0123", 0), 123.0),
        ((b"----", 1), nan),
        ((b"    ", 2), nan),
    ]
    for (value, places), expected in scaled_cases:
        out = records.scaled(sarr([value], 4), places)
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, [expected], rtol=1e-12, equal_nan=True)

    grid = records.integer(sarr([[b"01", b"**"], [b" 9", b"10"]], 2))
    assert grid.shape == (2, 2)
    np.testing.assert_array_equal(grid, [[1.0, nan], [9.0, 10.0]])
    np.testing.assert_allclose(
        records.scaled(sarr([[b"0015", b"0200"]], 4), 1), [[1.5, 20.0]], rtol=1e-12
    )


# ------------------------------------------------------------------------------------------------
# latest_versions


def test_latest_versions_keeps_the_newest_creation_and_drops_deleted_keys() -> None:
    frame = pd.DataFrame(
        [  # k1, k2, created, kubun, v
            ("A", "1", "20260101", "1", "a-old"),
            ("B", "1", "20260105", "1", "b-first"),
            ("A", "1", "20260301", "2", "a-new"),
            ("C", "1", "20260110", "1", "c-old"),
            ("A", "2", "20250101", "1", "a2-only"),
            ("A", "1", "20260201", "1", "a-mid"),
            ("B", "1", "20260105", "7", "b-tie-later"),
            ("C", "1", "20260120", "0", "c-deleted"),
            ("D", "1", "20260101", "0", "d-deleted"),
            ("D", "1", "20260102", "1", "d-recreated"),
            ("E", "1", "20261231", "0", "e-deleted"),
        ],
        columns=["k1", "k2", "データ作成年月日", "データ区分", "v"],
    )
    out = records.latest_versions(frame, ["k1", "k2"])
    assert list(out.columns) == list(frame.columns)
    # kept rows in input order: a-new (row 2), a2-only (4), b-tie-later (6), d-recreated (9)
    assert out["v"].tolist() == ["a-new", "a2-only", "b-tie-later", "d-recreated"]
    assert out[["k1", "k2"]].value_counts().max() == 1

    # One key column, other column names.
    small = pd.DataFrame(
        {
            "id": ["x", "y", "x", "y", "z"],
            "made": ["20240102", "20240301", "20240101", "20240302", "20240101"],
            "kind": ["1", "1", "1", "0", "2"],
            "v": [1, 2, 3, 4, 5],
        }
    )
    got = records.latest_versions(small, ["id"], created="made", kubun="kind")
    assert got["v"].tolist() == [1, 5]


# ------------------------------------------------------------------------------------------------
# Output helpers


def table() -> pa.Table:
    return pa.table(
        {
            "race_id": ["jra:2026-11-28:KYO:11", "jra:2010-01-05:NAK:01", None],
            "odds": [12.3, None, 1.5],
            "n": pa.array([1, 2, 3], type=pa.int16()),
        }
    )


def test_write_parquet_is_atomic_and_byte_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "out"
    folder.mkdir()
    first, second = folder / "a.parquet", folder / "b.parquet"
    records.write_parquet(table(), first)
    records.write_parquet(table(), second)
    assert first.read_bytes() == second.read_bytes()
    assert pq.read_table(first).equals(table())
    before = first.read_bytes()
    records.write_parquet(table(), first)  # over itself
    assert first.read_bytes() == before
    assert sorted(p.name for p in folder.iterdir()) == ["a.parquet", "b.parquet"]

    other = pa.table({"race_id": ["x"], "odds": [2.0], "n": pa.array([9], type=pa.int16())})
    records.write_parquet(other, second)
    assert pq.read_table(second).equals(other)

    # A write that fails before the swap leaves the old file whole and no stray file behind.
    def refuse(src, dst, *a, **k):
        raise OSError("refused")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError):
        records.write_parquet(other, first)
    monkeypatch.undo()
    assert first.read_bytes() == before
    assert sorted(p.name for p in folder.iterdir()) == ["a.parquet", "b.parquet"]


def report() -> records.BuildReport:
    return records.BuildReport(
        name="races",
        counts={"rows": 12, "mapped": 10},
        by_year={2026: {"rows": 5}, 2010: {"rows": 7}},
        unmapped={"venue": {"30": 2, "A4": 1}},
        failures=[{"file": "RAxx.jvd", "error": "short"}],
    )


def test_build_reports_go_through_write_report_under_the_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    rep = report()

    # to_json: sorted keys at every level, independent of insertion order.
    reordered = records.BuildReport(
        name="races",
        counts={"mapped": 10, "rows": 12},
        by_year={2010: {"rows": 7}, 2026: {"rows": 5}},
        unmapped={"venue": {"A4": 1, "30": 2}},
        failures=[{"error": "short", "file": "RAxx.jvd"}],
    )
    assert rep.to_json() == reordered.to_json()

    def sorted_pairs(pairs):
        keys = [k for k, _ in pairs]
        assert keys == sorted(keys), keys
        return dict(pairs)

    parsed = json.loads(rep.to_json(), object_pairs_hook=sorted_pairs)
    assert parsed == {
        "name": "races",
        "counts": {"rows": 12, "mapped": 10},
        "by_year": {"2026": {"rows": 5}, "2010": {"rows": 7}},
        "unmapped": {"venue": {"30": 2, "A4": 1}},
        "failures": [{"file": "RAxx.jvd", "error": "short"}],
    }

    path = records.write_build_report(rep)
    assert Path(path) == paths.reports_dir() / "jravan-races.json"
    assert Path(path).read_text(encoding="utf-8") == rep.to_json()
    other = records.BuildReport("odds", {"rows": 1}, {}, {}, [])
    assert Path(records.write_build_report(other)).name == "jravan-odds.json"

    # A reports folder inside the repository is refused, and nothing is created there.
    inside = env.repo_root() / f".tmp-jravan-{uuid.uuid4().hex[:8]}" / "reports"
    monkeypatch.setattr(paths, "reports_dir", lambda: inside)
    with pytest.raises(LicensedOutputInRepo):
        records.write_build_report(rep)
    assert not inside.parent.exists()
