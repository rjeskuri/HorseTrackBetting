"""JV-Data record layouts and code tables read from the spec workbook (ticket #361).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jravan.spec`:
- `spec_path()` is `paths.data_dir() / "specs" / "jravan" / "JV-Data4901.xlsx"`, resolved on every
  call; a missing workbook is a `FileNotFoundError` naming that path.
- `record_types(path=None) -> dict[str, int]` (ID to record length, CRLF included) reads every
  section's ID and length without validating layouts; `record_spec(rt, path=None) -> RecordSpec`
  validates only that section (`ValueError` naming the type and the item), an unknown type is a
  `LookupError`; `code_table(key, path=None) -> CodeTable` by 4-digit number or name.
- `horsetrack japan jravan check`: one line per export folder (`stored/<DS>/<RT>`,
  `timeseries/0B41`, `timeseries/0B42`) with `ok` and `<n> files`, or the mismatch (both widths);
  exit 1 on any failure, a missing workbook or a missing export; no record content printed.

Every workbook, record type, item, description and code table here is invented and built at run
time (`jravan_workbook`); the real workbook is licensed (D21). Expected values are written out by
hand from the layouts below, never computed by the code under test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths

# isort: split
# (its own block: the sibling helper module, then the module this ticket creates)
import jravan_workbook as wb
from horsetrack.countries.japan.jravan import spec
from horsetrack.countries.japan.jravan.spec import CodeTable, Item

ANSI = re.compile(r"\x1b\[[0-9;]*m")
BOX = re.compile(r"[─-╿\s]")

# ------------------------------------------------------------------------------------------------
# Made-up layouts


def section_aa(number: int = 1, reserve: int = 4) -> wb.Section:
    """`AA`, 40 bytes: a flat layout with key flags and string, numeric and blank initials."""
    rows = (
        wb.item(1, "識別子", 1, 2, key="●", initial="AA", desc=wb.id_desc("AA")),
        wb.item(2, "架空日付", 3, 8, key="○", initial=0, desc="開催日（架空）"),
        wb.item(3, "仮想番号", 11, 4, desc="番号（架空）"),
        wb.item(4, "名称", 15, 20, desc=wb.Phonetic("名称の架空説明", "メイショウ")),
        wb.item(5, "予備", 35, reserve, initial="0000", desc="予備領域"),
        wb.item(6, "架空区切", 35 + reserve, 2, desc="改行"),
    )
    return wb.Section(number, "架空レース", 36 + reserve, rows)


def items_aa(reserve: int = 4) -> tuple[Item, ...]:
    return (
        Item("識別子", 1, 2, 1, True, "AA", '"AA" をセット'),
        Item("架空日付", 3, 8, 1, True, "0", "開催日（架空）"),
        Item("仮想番号", 11, 4, 1, False, None, "番号（架空）"),
        Item("名称", 15, 20, 1, False, None, "名称の架空説明"),
        Item("予備", 35, reserve, 1, False, "0000", "予備領域"),
        Item("架空区切", 35 + reserve, 2, 1, False, None, "改行"),
    )


def section_bb(number: int = 2) -> wb.Section:
    """`BB`, 64 bytes: a top-level item repeated 8 times (8 x 6 = 48 bytes, 5-52)."""
    rows = (
        wb.item(1, "識別子", 1, 2, key="●", initial="BB", desc=wb.id_desc("BB")),
        wb.item(2, "架空コード", 3, 2, key="○", desc="コード（架空）"),
        wb.item(3, "架空値", 5, 6, repeat=8, initial="0", desc="値の繰返し"),
        wb.item(4, "予備", 53, 10, desc="予備"),
        wb.item(5, "架空区切", 63, 2, desc="改行"),
    )
    return wb.Section(number, "架空オッズ", 64, rows)


def items_bb() -> tuple[Item, ...]:
    return (
        Item("識別子", 1, 2, 1, True, "BB", '"BB" をセット'),
        Item("架空コード", 3, 2, 1, True, None, "コード（架空）"),
        Item("架空値", 5, 6, 8, False, "0", "値の繰返し"),
        Item("予備", 53, 10, 1, False, None, "予備"),
        Item("架空区切", 63, 2, 1, False, None, "改行"),
    )


def table_2010() -> wb.Table:
    return wb.Table(
        "2010",
        "架空馬場色",
        1,
        (("1", "青", "青色", "B"), ("2", "緑", "緑色", "G"), wb.BLANK, ("3", "灰", "灰色", "H")),
        subheaders=("内容", "略称", "英字"),
    )


def table_2011() -> wb.Table:
    return wb.Table(
        "2011",
        wb.Phonetic("架空区分", "カクウクブン"),
        2,
        (("01", "第一"), ("02", wb.Phonetic("第二", "ダイニ")), ("10", "第十")),
        sep=".",
    )


def table_2012() -> wb.Table:
    return wb.Table("2012", "架空種別", 3, (("001", "甲種"), ("002", "乙種")))


def bad(number: int, rt: str, length: int, fields: list[tuple]) -> wb.Section:
    """A section from (name, start, length[, repeat]) tuples; item 1 carries the ID."""
    rows = []
    for i, (name, start, size, *rest) in enumerate(fields, start=1):
        repeat = rest[0] if rest else None
        desc = wb.id_desc(rt) if i == 1 else f"{name}（架空）"
        rows.append(wb.item(i, name, start, size, repeat=repeat, desc=desc))
    return wb.Section(number, f"架空{rt}", length, tuple(rows))


def assert_ints(items: tuple[Item, ...]) -> None:
    for it in items:
        for value in (it.start, it.length, it.repeat):
            assert type(value) is int, (it.name, value)
        assert_ints(it.items)


def flat(text: str) -> str:
    """Output without ANSI codes, whitespace or box drawing (rich panels wrap long paths)."""
    return BOX.sub("", ANSI.sub("", text))


def lines_with(output: str, label: str) -> list[str]:
    return [line for line in ANSI.sub("", output).splitlines() if label in line]


# ------------------------------------------------------------------------------------------------
# Layouts


def test_a_record_layout_is_read_from_the_format_sheet(tmp_path: Path) -> None:
    path = wb.write_workbook(tmp_path / "book.xlsx", [section_aa(), section_bb()], [table_2010()])

    types = spec.record_types(path)
    assert types == {"AA": 40, "BB": 64}
    assert all(type(n) is int for n in types.values())

    aa = spec.record_spec("AA", path)
    assert (aa.record_type, aa.length) == ("AA", 40)
    assert "架空レース" in aa.title
    assert aa.items == items_aa()
    assert_ints(aa.items)

    bb = spec.record_spec("BB", path)
    assert (bb.record_type, bb.length) == ("BB", 64)
    assert "架空オッズ" in bb.title
    assert bb.items == items_bb()

    # The same path rewritten (a longer AA, a third section) is read again, not served from cache.
    cc = wb.simple_section(3, "CC", "架空第三", [("架空甲", 5)])
    wb.write_workbook(path, [section_aa(reserve=6), section_bb(), cc], [table_2010()])
    assert spec.record_types(path) == {"AA": 42, "BB": 64, "CC": 9}
    assert spec.record_spec("AA", path).items == items_aa(reserve=6)


def test_a_repeat_group_its_sub_items_and_a_nested_repeat_are_read(tmp_path: Path) -> None:
    wide = [
        wb.sub(wb.letter_code(i), f"枠{i + 1}", i + 1, 1, desc=f"枠{i + 1}の値") for i in range(28)
    ]
    rows = (
        wb.item(1, "識別子", 1, 2, key="●", initial="CC", desc=wb.id_desc("CC")),
        wb.item(2, "架空頭数", 3, 2, initial="00", desc="頭数（架空）"),
        wb.group(3, "<架空グループ>", 5, 10, 3, desc="グループ（架空）"),
        wb.sub("a", "甲", 1, 2, initial="0", desc="甲の値"),
        wb.sub("b", "乙", 3, 2, repeat=3, desc="乙の値"),
        wb.sub("c", "丙", 9, 2, desc="丙の値"),
        wb.item(4, "架空配列", 35, 3, repeat=4, initial="000", desc="配列（架空）"),
        wb.group(5, "<架空二十八>", 47, 28, 2, desc="二十八枠（架空）"),
        *wide,
        wb.item(6, "架空区切", 103, 2, desc="改行"),
    )
    path = wb.write_workbook(
        tmp_path / "book.xlsx", [wb.Section(7, "架空繰返", 104, rows)], [table_2010()]
    )

    got = spec.record_spec("CC", path)
    assert got.length == 104
    want_group = Item(
        "架空グループ",
        5,
        10,
        3,
        False,
        None,
        "グループ（架空）",
        (
            Item("甲", 1, 2, 1, False, "0", "甲の値"),
            Item("乙", 3, 2, 3, False, None, "乙の値"),
            Item("丙", 9, 2, 1, False, None, "丙の値"),
        ),
    )
    want_wide = Item(
        "架空二十八",
        47,
        28,
        2,
        False,
        None,
        "二十八枠（架空）",
        tuple(Item(f"枠{n}", n, 1, 1, False, None, f"枠{n}の値") for n in range(1, 29)),
    )
    assert got.items == (
        Item("識別子", 1, 2, 1, True, "CC", '"CC" をセット'),
        Item("架空頭数", 3, 2, 1, False, "00", "頭数（架空）"),
        want_group,
        Item("架空配列", 35, 3, 4, False, "000", "配列（架空）"),
        want_wide,
        Item("架空区切", 103, 2, 1, False, None, "改行"),
    )
    assert_ints(got.items)
    assert spec.record_types(path) == {"CC": 104}


def test_names_lose_phonetic_runs_spaces_and_angle_brackets(tmp_path: Path) -> None:
    rows = (
        wb.item(1, "識別子", 1, 2, initial="DD", desc=wb.id_desc("DD")),
        wb.group(
            2,
            wb.Phonetic("<単勝オッズ>", "タンショウオッズ"),
            3,
            6,
            2,
            desc=wb.Phonetic("単勝の架空説明", "タンショウ"),
        ),
        wb.sub("a", wb.Phonetic("　組番", "クミバン"), 1, 2, desc="組番の値"),
        wb.sub("b", "オッズ値", 3, 4, indent="　　", desc="オッズの値"),
        wb.item(3, wb.Phonetic("発売フラグ　単勝", "ハツバイフラグ"), 15, 1, desc="フラグ"),
        wb.item(4, "架空 名称", 16, 3, desc="半角空白"),
        wb.item(5, wb.Rich(("架空", "項目"), reading="カクウ"), 19, 2, desc="リッチ"),
        wb.item(6, wb.Inline(" 埋込 名 "), 21, 3, desc=wb.Inline("インライン説明")),
        wb.item(7, "架空区切", 24, 2, desc="改行"),
    )
    path = wb.write_workbook(
        tmp_path / "book.xlsx", [wb.Section(4, "架空名前", 25, rows)], [table_2010()]
    )

    got = spec.record_spec("DD", path)
    names = [it.name for it in got.items]
    assert names == [
        "識別子",
        "単勝オッズ",
        "発売フラグ単勝",
        "架空名称",
        "架空項目",
        "埋込名",
        "架空区切",
    ]
    group = got.items[1]
    assert [it.name for it in group.items] == ["組番", "オッズ値"]
    assert group.description == "単勝の架空説明"
    assert got.items[5].description == "インライン説明"
    readings = ("タンショウ", "クミバン", "ハツバイ", "カクウ")
    for it in (*got.items, *group.items):
        assert not any(r in it.name or r in it.description for r in readings), it
        assert not set(it.name) & {" ", "　", "<", ">"}, it.name
    assert [(it.start, it.length, it.repeat) for it in got.items] == [
        (1, 2, 1),
        (3, 6, 2),
        (15, 1, 1),
        (16, 3, 1),
        (19, 2, 1),
        (21, 3, 1),
        (24, 2, 1),
    ]


def test_label_rows_and_blank_rows_inside_a_section_are_skipped(tmp_path: Path) -> None:
    ee = (
        wb.BLANK,
        wb.item(1, "識別子", 1, 2, initial="EE", desc=wb.id_desc("EE")),
        wb.label("<架空見出し>"),
        wb.item(2, "架空A", 3, 4, desc="A（架空）"),
        wb.BLANK,
        wb.SKIP,
        wb.label(wb.Phonetic("<架空小見出し>", "カクウコミダシ")),
        wb.group(3, "<架空組>", 7, 3, 2, desc="組（架空）"),
        wb.sub("a", "子一", 1, 1, desc="子一の値"),
        wb.sub("b", "子二", 2, 2, desc="子二の値"),
        wb.BLANK,
        wb.label(wb.Inline("<インライン見出し>")),
        wb.item(4, "架空B", 13, 2, desc="B（架空）"),
        wb.item(5, "架空区切", 15, 2, desc="改行"),
        wb.BLANK,
        wb.BLANK,
    )
    ff = (
        wb.item(1, "識別子", 1, 2, initial="FF", desc=wb.id_desc("FF")),
        wb.item(2, "架空区切", 3, 2, desc="改行"),
        wb.BLANK,
        wb.label("<末尾の見出し>"),
        wb.SKIP,
        wb.BLANK,
    )
    path = wb.write_workbook(
        tmp_path / "book.xlsx",
        [wb.Section(5, "架空見出し付き", 16, ee), wb.Section(6, "架空最短", 4, ff)],
        [table_2010()],
    )

    assert spec.record_types(path) == {"EE": 16, "FF": 4}
    assert spec.record_spec("EE", path).items == (
        Item("識別子", 1, 2, 1, False, "EE", '"EE" をセット'),
        Item("架空A", 3, 4, 1, False, None, "A（架空）"),
        Item(
            "架空組",
            7,
            3,
            2,
            False,
            None,
            "組（架空）",
            (
                Item("子一", 1, 1, 1, False, None, "子一の値"),
                Item("子二", 2, 2, 1, False, None, "子二の値"),
            ),
        ),
        Item("架空B", 13, 2, 1, False, None, "B（架空）"),
        Item("架空区切", 15, 2, 1, False, None, "改行"),
    )
    assert spec.record_spec("FF", path).items == (
        Item("識別子", 1, 2, 1, False, "FF", '"FF" をセット'),
        Item("架空区切", 3, 2, 1, False, None, "改行"),
    )


# ------------------------------------------------------------------------------------------------
# Validation


def test_a_gap_or_overlap_in_a_layout_is_an_error(tmp_path: Path) -> None:
    # (record type, length, items, the item the error must name or None for the type alone)
    cases = [
        (
            "GA",
            12,
            [("識別子", 1, 2), ("甲項目", 3, 4), ("乙項目", 8, 3), ("区切", 11, 2)],
            "乙項目",
        ),
        (
            "GB",
            10,
            [("識別子", 1, 2), ("甲項目", 3, 4), ("丙項目", 6, 3), ("区切", 9, 2)],
            "丙項目",
        ),
        ("GC", 8, [("先頭項目", 2, 1), ("甲項目", 3, 4), ("区切", 7, 2)], "先頭項目"),
        # A repeated item spans repeat x length: 繰返項目 covers 3-8, so 丁項目 at 5 overlaps it.
        (
            "GD",
            8,
            [("識別子", 1, 2), ("繰返項目", 3, 2, 3), ("丁項目", 5, 2), ("区切", 7, 2)],
            "丁項目",
        ),
        # The items end at byte 8 of a 10-byte record, or run past it.
        ("GE", 10, [("識別子", 1, 2), ("甲項目", 3, 4), ("区切", 7, 2)], None),
        ("GF", 10, [("識別子", 1, 2), ("甲項目", 3, 8), ("区切", 11, 2)], None),
    ]
    good = ("GZ", 12, [("識別子", 1, 2), ("繰返項目", 3, 2, 3), ("丁項目", 9, 2), ("区切", 11, 2)])
    sections = [bad(i, rt, n, f) for i, (rt, n, f, _) in enumerate(cases, start=1)]
    sections.append(bad(9, *good))
    path = wb.write_workbook(tmp_path / "book.xlsx", sections, [table_2010()])

    for rt, _, _, offender in cases:
        with pytest.raises(ValueError) as info:
            spec.record_spec(rt, path)
        assert rt in str(info.value), (rt, str(info.value))
        if offender:
            assert offender in str(info.value), (rt, str(info.value))
    assert [it.start for it in spec.record_spec("GZ", path).items] == [1, 3, 9, 11]


def test_a_group_not_covered_by_its_sub_items_is_an_error(tmp_path: Path) -> None:
    def grouped(number, rt, name, width, subs):
        rows = [
            wb.item(1, "識別子", 1, 2, desc=wb.id_desc(rt)),
            wb.group(2, f"<{name}>", 3, width, 2, desc="組（架空）"),
        ]
        for i, (sub_name, rel, size, *rest) in enumerate(subs):
            rows.append(
                wb.sub(wb.letter_code(i), sub_name, rel, size, repeat=rest[0] if rest else None)
            )
        rows.append(wb.item(3, "架空区切", 3 + 2 * width, 2, desc="改行"))
        return wb.Section(number, f"架空{rt}", 4 + 2 * width, tuple(rows))

    # (record type, group, width, sub-items (name, rel, bytes[, repeat]), names it may give)
    cases = [
        ("HA", "短い組", 6, [("短子一", 1, 2), ("短子二", 3, 2)], ("短い組",)),
        ("HB", "長い組", 4, [("長子一", 1, 2), ("長子二", 3, 3)], ("長い組",)),
        ("HC", "穴の組", 6, [("穴子一", 1, 2), ("穴子二", 4, 3)], ("穴の組", "穴子二")),
        # 入子一 repeats 3 times (1-6), so 入子二 at 3 overlaps it.
        ("HD", "入れ子組", 8, [("入子一", 1, 2, 3), ("入子二", 3, 2)], ("入れ子組", "入子二")),
    ]
    sections = [grouped(i, rt, g, w, s) for i, (rt, g, w, s, _) in enumerate(cases, start=1)]
    sections.append(grouped(9, "HZ", "正しい組", 6, [("正子一", 1, 2, 2), ("正子二", 5, 2)]))
    path = wb.write_workbook(tmp_path / "book.xlsx", sections, [table_2010()])

    for rt, _, _, _, names in cases:
        with pytest.raises(ValueError) as info:
            spec.record_spec(rt, path)
        message = str(info.value)
        assert rt in message, (rt, message)
        assert any(name in message for name in names), (rt, message)
    good = spec.record_spec("HZ", path).items[1]
    assert [(it.name, it.start, it.repeat) for it in good.items] == [
        ("正子一", 1, 2),
        ("正子二", 5, 1),
    ]


def test_a_layout_without_the_two_byte_separator_is_an_error(tmp_path: Path) -> None:
    cases = [
        ("SA", 9, [("識別子", 1, 2), ("甲項目", 3, 4), ("三字区切", 7, 3)], "三字区切"),
        ("SB", 7, [("識別子", 1, 2), ("甲項目", 3, 4), ("一字区切", 7, 1)], "一字区切"),
        ("SC", 8, [("識別子", 1, 2), ("最後の項目", 3, 6)], "最後の項目"),
    ]
    sections = [bad(i, rt, n, f) for i, (rt, n, f, _) in enumerate(cases, start=1)]
    sections.append(bad(9, "SZ", 8, [("識別子", 1, 2), ("甲項目", 3, 4), ("区切", 7, 2)]))
    path = wb.write_workbook(tmp_path / "book.xlsx", sections, [table_2010()])

    for rt, _, _, offender in cases:
        with pytest.raises(ValueError) as info:
            spec.record_spec(rt, path)
        assert rt in str(info.value), (rt, str(info.value))
        assert offender in str(info.value), (rt, str(info.value))
    assert spec.record_spec("SZ", path).items[-1].length == 2


def test_a_broken_section_does_not_stop_another_record_type(tmp_path: Path) -> None:
    broken_gap = bad(2, "KB", 10, [("識別子", 1, 2), ("甲項目", 4, 5), ("区切", 9, 2)])
    broken_group = wb.Section(
        3,
        "架空KC",
        12,
        (
            wb.item(1, "識別子", 1, 2, desc=wb.id_desc("KC")),
            wb.group(2, "<欠けた組>", 3, 4, 2, desc="組"),
            wb.sub("a", "欠子", 1, 2),
            wb.item(3, "架空区切", 11, 2, desc="改行"),
        ),
    )
    broken_end = bad(4, "KD", 8, [("識別子", 1, 2), ("最後の項目", 3, 6)])
    sections = [
        section_aa(1),
        broken_gap,
        broken_group,
        broken_end,
        section_bb(5),
    ]
    path = wb.write_workbook(tmp_path / "book.xlsx", sections, [table_2010()])

    assert spec.record_types(path) == {"AA": 40, "KB": 10, "KC": 12, "KD": 8, "BB": 64}
    assert spec.record_spec("AA", path).items == items_aa()
    assert spec.record_spec("BB", path).items == items_bb()
    for rt in ("KB", "KC", "KD"):
        with pytest.raises(ValueError):
            spec.record_spec(rt, path)
    assert spec.code_table("2010", path).values == {"1": "青", "2": "緑", "3": "灰"}


def test_an_unknown_record_type_is_a_lookup_error(tmp_path: Path) -> None:
    path = wb.write_workbook(tmp_path / "book.xlsx", [section_aa(), section_bb()], [table_2010()])
    for rt in ("QQ", "A", "AAB", "RA"):
        with pytest.raises(LookupError):
            spec.record_spec(rt, path)
    assert spec.record_spec("AA", path).length == 40


def test_sheets_are_found_by_name_not_position(tmp_path: Path) -> None:
    decoy_format = wb.format_rows([wb.simple_section(1, "AA", "旧レース", [("旧項目", 7)])])
    decoy_codes = wb.code_rows([wb.Table("2010", "旧馬場色", 1, (("1", "旧"), ("9", "旧九")))])
    others = {"フォーマット(旧)": decoy_format, "コード表(旧)": decoy_codes}
    orders = [
        (
            "表紙",
            "フォーマット(旧)",
            "コード表(旧)",
            "変更履歴",
            "フォーマット",
            "特記事項",
            "コード表",
        ),
        ("コード表(旧)", "コード表", "表紙", "フォーマット(旧)", "フォーマット"),
        ("フォーマット", "コード表", "フォーマット(旧)", "コード表(旧)"),
    ]
    for i, order in enumerate(orders):
        path = wb.write_workbook(
            tmp_path / f"book{i}.xlsx",
            [section_aa(), section_bb()],
            [table_2010()],
            sheets=order,
            others=others,
        )
        assert spec.record_types(path) == {"AA": 40, "BB": 64}, order
        assert spec.record_spec("AA", path).items == items_aa(), order
        assert spec.code_table("2010", path).values == {"1": "青", "2": "緑", "3": "灰"}, order


def test_a_code_table_maps_values_to_their_contents(tmp_path: Path) -> None:
    path = wb.write_workbook(
        tmp_path / "book.xlsx", [section_aa()], [table_2010(), table_2011(), table_2012()]
    )

    t2010 = spec.code_table("2010", path)
    assert isinstance(t2010, CodeTable)
    assert spec.code_table("架空馬場色", path) == t2010
    assert (t2010.code, t2010.name, t2010.width) == ("2010", "架空馬場色", 1)
    assert t2010.values == {"1": "青", "2": "緑", "3": "灰"}
    assert t2010.extra == {"1": ("青色", "B"), "2": ("緑色", "G"), "3": ("灰色", "H")}

    t2011 = spec.code_table("2011", path)
    assert spec.code_table("架空区分", path) == t2011
    assert (t2011.code, t2011.name, t2011.width) == ("2011", "架空区分", 2)
    assert t2011.values == {"01": "第一", "02": "第二", "10": "第十"}
    assert all(t2011.extra.get(v, ()) == () for v in t2011.values)

    t2012 = spec.code_table("架空種別", path)
    assert spec.code_table("2012", path) == t2012
    assert (t2012.code, t2012.width) == ("2012", 3)
    assert t2012.values == {"001": "甲種", "002": "乙種"}

    for key in ("2013", "201", "架空", "存在しない表"):
        with pytest.raises(LookupError):
            spec.code_table(key, path)


# ------------------------------------------------------------------------------------------------
# The spec path and the check command


def test_the_spec_path_follows_the_data_dir_and_a_missing_workbook_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(first))
    want_first = first.resolve() / "specs" / "jravan" / "JV-Data4901.xlsx"
    assert spec.spec_path() == want_first
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(second))
    want_second = second.resolve() / "specs" / "jravan" / "JV-Data4901.xlsx"
    assert spec.spec_path() == want_second

    for call in (
        lambda: spec.record_types(),
        lambda: spec.record_spec("AA"),
        lambda: spec.code_table("2010"),
    ):
        with pytest.raises(FileNotFoundError) as info:
            call()
        assert str(want_second) in str(info.value)

    # The export is present, the workbook is not: the check names the path and exits 1.
    export = second / "raw" / "jravan" / "stored" / "RACE" / "AA"
    wb.write_jvd(export / "2001.jvd", [wb.jvd_record("AA", 40, "架空")])
    result = CliRunner().invoke(app, ["japan", "jravan", "check"])
    assert result.exit_code == 1, result.output
    assert flat(str(want_second)) in flat(result.output), result.output

    wb.write_workbook(want_second, [section_aa(), section_bb()], [table_2010()])
    assert spec.record_types() == {"AA": 40, "BB": 64}
    assert spec.record_spec("BB").items == items_bb()
    assert spec.code_table("2010").name == "架空馬場色"
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(first))
    with pytest.raises(FileNotFoundError):
        spec.record_types()


def check_workbook(path: Path) -> Path:
    """AA 40, BB 64, O1 30, O2 26, and XB, a broken section no export folder uses."""
    sections = [
        section_aa(1),
        section_bb(2),
        wb.simple_section(3, "O1", "架空時系列一", [("架空券", 10), ("架空票", 16)]),
        wb.simple_section(4, "O2", "架空時系列二", [("架空券", 22)]),
        bad(5, "XB", 10, [("識別子", 1, 2), ("甲項目", 4, 5), ("区切", 9, 2)]),
    ]
    return wb.write_workbook(path, sections, [table_2010()])


SECRET = "秘匿内容ZQX"


def records(rt: str, length: int, n: int = 3) -> list[bytes]:
    return [wb.jvd_record(rt, length, f"{SECRET}{i}") for i in range(n)]


def test_check_reports_each_folder_and_fails_on_a_width_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    check_workbook(data / "specs" / "jravan" / "JV-Data4901.xlsx")
    runner = CliRunner()

    # No export yet.
    result = runner.invoke(app, ["japan", "jravan", "check"])
    assert result.exit_code == 1, result.output

    root = data / "raw" / "jravan"
    (root / "README.md").parent.mkdir(parents=True)
    (root / "README.md").write_text("架空の説明\n")
    (root / "manifest.json").write_text("{}\n")
    # label -> (record type, length, files relative to the folder)
    folders = {
        "stored/RACE/AA": ("AA", 40, ["2001.jvd", "2002.jvd"]),
        "stored/DIFF/AA": ("AA", 40, ["2010.jvd", "2011.jvd", "2012.jvd"]),
        "stored/RACE/BB": ("BB", 64, ["1999.jvd", "2000.jvd", "2003.jvd", "2004.jvd"]),
        "timeseries/0B41": (
            "O1",
            30,
            [
                "2003/20030105.jvd",
                "2003/20030106.jvd",
                "2004/20040110.jvd",
                "2004/20040111.jvd",
                "2005/20050102.jvd",
            ],
        ),
        "timeseries/0B42": ("O2", 26, ["2003/20030105.jvd", "2004/20040110.jvd"]),
    }
    for label, (rt, length, files) in folders.items():
        for name in files:
            wb.write_jvd(root / label / name, records(rt, length))
    # Folders the check ignores, each of which would fail it.
    wb.write_jvd(root / "probe" / "QZ" / "x.jvd", records("QZ", 21))
    wb.write_jvd(root / "stored_shift" / "RACE" / "AA" / "2001.jvd", records("AA", 41))
    wb.write_jvd(root / "_incoming" / "AA" / "2001.jvd", records("AA", 39))

    result = runner.invoke(app, ["japan", "jravan", "check"])
    out = result.output
    assert result.exit_code == 0, out
    for label, (_, _, files) in folders.items():
        lines = lines_with(out, label)
        assert len(lines) == 1, (label, out)
        assert re.search(r"\bok\b", lines[0]), (label, out)
        assert f"{len(files)} files" in lines[0], (label, out)
    for ignored in ("probe", "stored_shift", "_incoming", "QZ", "XB"):
        assert ignored not in out, (ignored, out)
    assert SECRET not in out and "ZQX" not in out, out

    # The first file of a folder is a byte longer: that folder fails with both widths.
    first = root / "stored/RACE/AA/2001.jvd"
    wb.write_jvd(first, [r[:-2] + b"9\r\n" for r in records("AA", 40)])
    result = runner.invoke(app, ["japan", "jravan", "check"])
    out = result.output
    assert result.exit_code == 1, out
    bad_line = lines_with(out, "stored/RACE/AA")
    assert len(bad_line) == 1, out
    assert "40" in bad_line[0] and "41" in bad_line[0], out
    for label, (_, _, files) in folders.items():
        if label != "stored/RACE/AA":
            assert f"{len(files)} files" in lines_with(out, label)[0], (label, out)
    assert SECRET not in out, out

    # Same for a time series: O2's first day is a byte short.
    wb.write_jvd(first, records("AA", 40))
    assert runner.invoke(app, ["japan", "jravan", "check"]).exit_code == 0
    day = root / "timeseries/0B42/2003/20030105.jvd"
    wb.write_jvd(day, [wb.jvd_record("O2", 25, SECRET)])
    result = runner.invoke(app, ["japan", "jravan", "check"])
    assert result.exit_code == 1, result.output
    assert "25" in lines_with(result.output, "timeseries/0B42")[0], result.output

    # A record type the workbook does not describe fails the check, naming its folder.
    wb.write_jvd(day, records("O2", 26))
    wb.write_jvd(root / "stored/RACE/QZ/2001.jvd", records("QZ", 20))
    wb.write_jvd(root / "stored/RACE/QZ/2002.jvd", records("QZ", 20))
    result = runner.invoke(app, ["japan", "jravan", "check"])
    assert result.exit_code == 1, result.output
    assert len(lines_with(result.output, "stored/RACE/QZ")) == 1, result.output
    assert SECRET not in result.output


# ------------------------------------------------------------------------------------------------
# The real workbook (local: needs $HORSETRACK_DATA_DIR with the export and the workbook)


def first_record_width(path: Path) -> int:
    with path.open("rb") as fh:
        return len(fh.readline())


def first_record(path: Path) -> bytes:
    with path.open("rb") as fh:
        return fh.readline()


@pytest.mark.local
def test_the_real_workbook_describes_every_exported_record_type(
    capsys: pytest.CaptureFixture[str],
) -> None:
    export, workbook = wb.real_export_or_skip()
    folders = [
        (f"stored/{ds.name}/{rt.name}", rt.name, rt)
        for ds in sorted((export / "stored").iterdir())
        if ds.is_dir()
        for rt in sorted(ds.iterdir())
        if rt.is_dir()
    ]
    for series, rt in (("0B41", "O1"), ("0B42", "O2")):
        if (export / "timeseries" / series).is_dir():
            folders.append((f"timeseries/{series}", rt, export / "timeseries" / series))
    assert folders

    failures, firsts = [], []
    for label, rt, folder in folders:
        files = sorted(folder.rglob("*.jvd"))
        if not files:
            failures.append(f"{label}: no .jvd files")
            continue
        firsts.append(first_record(files[0]))
        width = len(firsts[-1])
        try:
            layout = spec.record_spec(rt, workbook)
        except (LookupError, ValueError) as exc:
            failures.append(f"{label}: {type(exc).__name__}")  # no message: it quotes the layout
            continue
        if layout.length != width:
            failures.append(f"{label}: layout {layout.length} bytes, records {width}")
    assert not failures, failures

    result = CliRunner().invoke(app, ["japan", "jravan", "check"])
    assert result.exit_code == 0, f"exit {result.exit_code}, {len(result.output)} chars of output"
    leaked = 0
    for record in firsts:
        chunk = record[2:22].decode("cp932", errors="ignore").strip()
        if len(chunk) >= 8 and chunk in result.output:
            leaked += 1
    assert leaked == 0, f"{leaked} folders' first-record content in the check output"
    assert capsys.readouterr().out == ""
