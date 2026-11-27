"""JV-Data record layouts and code tables, read at run time from the specification workbook.

The JV-Data specification (JRA-VAN, licensed, D21) is a precondition: the owner places it at
``<data dir>/specs/jravan/JV-Data4901.xlsx`` and the repository holds no layout, description or
code-table text. Every parser takes its byte offsets (cp932), lengths and repeat counts from here.

The workbook is read with the standard library: shared strings without their phonetic ``<rPh>``
runs, inline strings and numbers (a formula's cached value). The sheets ``フォーマット`` (record
layouts) and ``コード表`` (code tables) are found by name through ``xl/workbook.xml`` and its rels,
never by position. A read is cached per (path, size, mtime_ns), so a rewritten workbook is read
again.

フォーマット: a section starts at a row whose column F reads ``レコード長`` (H the record length, B
the title) and ends at the next such row. Item rows have B the item number, D the key mark, E the
name, F the 1-based position, G the repeat (blank is 1), H the bytes of one occurrence, J the
initial value and K the description; sub-item rows have B blank, C a letter code and F the position
in the group, written ``(   n)`` (a blank one follows on from the previous sub-item). Other rows
(the header, labels, notes, blanks) are skipped. The record type ID is the two characters quoted at
the start of item 1's description.

`record_spec` validates its own section only, so a section no parser uses never stops a read: the
top-level items cover bytes 1 to the record length (a repeated item spans repeat x length), each
group's sub-items cover 1 to the group's width, with no gap or overlap, and the last item is the
2-byte record separator. Anything else is a ValueError naming the record type and the item.
"""

from __future__ import annotations

import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from horsetrack.core import paths

WORKBOOK_NAME = "JV-Data4901.xlsx"
FORMAT_SHEET = "フォーマット"
CODE_SHEET = "コード表"
SEPARATOR_BYTES = 2
KEY_MARKS = frozenset({"○", "●"})

_LENGTH_MARK = "レコード長"
_VALUE_HEADER = "値"
_FIRST_CONTENT = 4  # column D, a code table's first content column
_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_ID = re.compile(r"""^\s*["'“”]([0-9A-Za-z]{2})["'“”]""")
_RELATIVE = re.compile(r"^\(\s*(\d+)\s*\)$")
_TABLE_TITLE = re.compile(r"^([0-9]{4})[.．](.*)$", re.DOTALL)
_SPACE = re.compile(r"\s+")  # half- and full-width spaces, newlines
_CELL_REF = re.compile(r"^([A-Z]+)")

Value = str | int | float
Row = dict[str, Value]


@dataclass(frozen=True)
class Item:
    """One item of a layout. `start` is 1-based, in the record (top level) or in its group."""

    name: str
    start: int
    length: int
    repeat: int
    is_key: bool
    initial: str | None
    description: str
    items: tuple[Item, ...] = ()


@dataclass(frozen=True)
class RecordSpec:
    record_type: str
    title: str
    length: int
    items: tuple[Item, ...]


@dataclass(frozen=True)
class CodeTable:
    """`values` maps each 値 to its first 内容 column, `extra` to the other content columns."""

    code: str
    name: str
    width: int
    values: dict[str, str]
    extra: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class _Section:
    record_type: str
    title: str
    length: int
    rows: tuple[Row, ...]
    duplicate: bool = False  # another section carries the same record type ID


@dataclass(frozen=True)
class _Workbook:
    sections: dict[str, _Section] = field(default_factory=dict)
    tables: tuple[CodeTable, ...] = ()


_CACHE: dict[tuple[str, int, int], _Workbook] = {}


def spec_path() -> Path:
    """The workbook's path under the data dir, resolved on every call."""
    return paths.data_dir() / "specs" / "jravan" / WORKBOOK_NAME


def record_types(path: Path | None = None) -> dict[str, int]:
    """Record type ID to record length in bytes (CRLF included), for every section."""
    return {rt: s.length for rt, s in _load(path).sections.items()}


def record_spec(record_type: str, path: Path | None = None) -> RecordSpec:
    """The validated layout of `record_type`; LookupError if absent, ValueError if invalid."""
    book = _load(path)
    section = book.sections.get(record_type)
    if section is None:
        raise LookupError(f"record type {record_type!r} is not in the JV-Data specification")
    if section.duplicate:
        raise ValueError(f"{record_type}: more than one section describes this record type")
    items = _items(section)
    _check_cover(record_type, "the record", items, section.length)
    for it in items:
        if it.items:
            _check_cover(record_type, f"group {it.name}", it.items, it.length)
    last = items[-1]
    if last.length != SEPARATOR_BYTES or last.repeat != 1 or last.items:
        raise ValueError(
            f"{record_type}: the last item {last.name} is {last.repeat * last.length} bytes, "
            f"not the {SEPARATOR_BYTES}-byte record separator"
        )
    return RecordSpec(record_type, section.title, section.length, items)


def code_table(key: str, path: Path | None = None) -> CodeTable:
    """The code table numbered `key` ('2010') or named `key`; LookupError if absent."""
    wanted = key.strip()
    for table in _load(path).tables:
        if wanted in (table.code, table.name):
            return table
    raise LookupError(f"code table {key!r} is not in the JV-Data specification")


# ------------------------------------------------------------------------------------------------
# Layouts


def _items(section: _Section) -> tuple[Item, ...]:
    rt = section.record_type
    top: list[tuple[Row, list[Row]]] = []
    for row in section.rows:
        if _is_item_number(row.get("B")):
            top.append((row, []))
        elif _blank(row.get("B")) and not _blank(row.get("C")):
            if not top:
                raise ValueError(f"{rt}: sub-item {_name(row)} comes before any item")
            top[-1][1].append(row)
    if not top:
        raise ValueError(f"{rt}: the section has no items")
    return tuple(_item(rt, row, _sub_items(rt, subs)) for row, subs in top)


def _sub_items(rt: str, rows: list[Row]) -> tuple[Item, ...]:
    """A group's sub-items; a blank position follows on from the previous sub-item."""
    subs: list[Item] = []
    for row in rows:
        start = row.get("F")
        if isinstance(start, str) and (match := _RELATIVE.match(start.strip())):
            start = int(match.group(1))
        elif _blank(start):
            start = subs[-1].start + subs[-1].repeat * subs[-1].length if subs else 1
        subs.append(_item(rt, row, (), start=start))
    return tuple(subs)


def _is_item_number(value: Value | None) -> bool:
    """B holds an item number; other text there (a note row) is not an item."""
    if isinstance(value, str):
        return value.strip().isdecimal()
    return isinstance(value, int) or (isinstance(value, float) and value.is_integer())


def _item(rt: str, row: Row, subs: tuple[Item, ...], *, start: Value | None = None) -> Item:
    name = _name(row)
    if start is None:
        start = row.get("F")
    repeat = row.get("G")
    initial = row.get("J")
    return Item(
        name=name,
        start=_number(rt, name, "position", start),
        length=_number(rt, name, "length", row.get("H")),
        repeat=1 if _blank(repeat) else _number(rt, name, "repeat", repeat),
        is_key=_text(row.get("D")).strip() in KEY_MARKS,
        initial=None if _blank(initial) else _text(initial),
        description=_text(row.get("K")),
        items=subs,
    )


def _check_cover(rt: str, where: str, items: tuple[Item, ...], length: int) -> None:
    pos, previous = 1, None
    for it in items:
        if it.length <= 0 or it.repeat <= 0:
            raise ValueError(f"{rt}: item {it.name} has length {it.length}, repeat {it.repeat}")
        if it.start != pos:
            if previous is None:
                why = f"{where} starts at byte 1"
            elif it.start < pos:
                why = f"it overlaps {previous.name}, bytes {previous.start}-{pos - 1}"
            else:
                why = f"bytes {pos}-{it.start - 1} after {previous.name} are not covered"
            raise ValueError(f"{rt}: {where}: item {it.name} starts at byte {it.start}: {why}")
        pos += it.repeat * it.length
        previous = it
    if pos - 1 != length:
        last = previous.name if previous else "nothing"
        raise ValueError(
            f"{rt}: {where} is {length} bytes but its items, through {last}, cover 1-{pos - 1}"
        )


def _name(row: Row) -> str:
    name = _SPACE.sub("", _text(row.get("E")))
    if name.startswith("<") and name.endswith(">"):
        name = name[1:-1]
    return name


def _number(rt: str, name: str, what: str, value: Value | None) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdecimal():
        return int(value.strip())
    raise ValueError(f"{rt}: item {name}: the {what} {value!r} is not a whole number")


def _sections(rows: list[Row]) -> dict[str, _Section]:
    raw: list[tuple[Row, list[Row]]] = []
    for row in rows:
        if _text(row.get("F")).strip() == _LENGTH_MARK:
            raw.append((row, []))
        elif raw:
            raw[-1][1].append(row)
    sections: dict[str, _Section] = {}
    for title_row, body in raw:
        first = next((r for r in body if _is_item_number(r.get("B"))), None)
        match = _ID.match(_text(first.get("K"))) if first else None
        length = title_row.get("H")
        if match is None or not isinstance(length, int):
            continue  # no record type to look it up by
        rt = match.group(1)
        title = _text(title_row.get("B")).strip()
        duplicate = rt in sections
        sections[rt] = _Section(rt, title, length, tuple(body), duplicate)
    return sections


# ------------------------------------------------------------------------------------------------
# Code tables


def _tables(rows: list[Row]) -> tuple[CodeTable, ...]:
    raw: list[tuple[str, str, list[Row]]] = []
    for row in rows:
        match = _TABLE_TITLE.match(_text(row.get("B")).strip())
        if match:
            raw.append((match.group(1), match.group(2).strip(), []))
        elif raw:
            raw[-1][2].append(row)
    return tuple(_table(code, name, body) for code, name, body in raw)


def _table(code: str, name: str, body: list[Row]) -> CodeTable:
    width: int | None = None
    contents: dict[str, list[str]] = {}
    for row in body:
        value = row.get("C")
        if _blank(value) or _text(value).strip() == _VALUE_HEADER:
            continue  # the header, content sub-headers, blank rows
        if width is None and isinstance(row.get("B"), int):
            width = row["B"]  # type: ignore[assignment]
        if width is not None and len(_text(value).strip()) > width:
            continue  # a note under the table: a code is at most `width` characters
        cells = {_column(col) - _FIRST_CONTENT: _text(v).strip() for col, v in row.items()}
        cells = {i: v for i, v in cells.items() if i >= 0 and v}
        texts = [cells.get(i, "") for i in range(max(cells, default=-1) + 1)]
        contents.setdefault(_text(value).strip(), texts)
    columns = max((len(t) for t in contents.values()), default=0)
    values = {v: (t[0] if t else "") for v, t in contents.items()}
    extra = {v: tuple(t[1:] + [""] * (columns - len(t))) for v, t in contents.items()}
    return CodeTable(code, name, width or 0, values, extra)


def _column(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - ord("A") + 1
    return n


# ------------------------------------------------------------------------------------------------
# The xlsx


def _load(path: Path | None) -> _Workbook:
    path = Path(path) if path is not None else spec_path()
    try:
        stat = path.stat()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"JV-Data specification workbook not found: {path} (the JV-Data specification is a "
            "precondition: place JV-Data4901.xlsx there)"
        ) from None
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    book = _CACHE.get(key)
    if book is None:
        with zipfile.ZipFile(path) as archive:
            sheets = _read_sheets(archive, (FORMAT_SHEET, CODE_SHEET))
        book = _Workbook(_sections(sheets[FORMAT_SHEET]), _tables(sheets[CODE_SHEET]))
        _CACHE.clear()  # one workbook in use at a time; a rewritten one replaces it
        _CACHE[key] = book
    return book


def _read_sheets(archive: zipfile.ZipFile, names: tuple[str, ...]) -> dict[str, list[Row]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {}
    strings_part = "xl/sharedStrings.xml"
    for rel in rels.iter(f"{{{_PKG_REL_NS}}}Relationship"):
        target = _part(rel.get("Target", ""))
        targets[rel.get("Id")] = target
        if rel.get("Type", "").endswith("/sharedStrings"):
            strings_part = target
    by_name = {
        sheet.get("name"): targets.get(sheet.get(f"{{{_REL_NS}}}id"))
        for sheet in workbook.iter(f"{{{_MAIN_NS}}}sheet")
    }
    strings: list[str] = []
    if strings_part in archive.namelist():
        sst = ET.fromstring(archive.read(strings_part))
        strings = [_string(si) for si in sst.iter(f"{{{_MAIN_NS}}}si")]
    out = {}
    for name in names:
        part = by_name.get(name)
        if part is None:
            raise ValueError(f"the JV-Data specification has no sheet named {name}")
        out[name] = _rows(ET.fromstring(archive.read(part)), strings)
    return out


def _part(target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join("xl", target))


def _string(element: ET.Element) -> str:
    """The text of an `<si>` or `<is>`: its `<t>` and rich-run `<t>`s, never a `<rPh>` reading."""
    parts = []
    for child in element:
        tag = _local(child.tag)
        if tag == "t":
            parts.append(child.text or "")
        elif tag == "r":
            parts.extend(t.text or "" for t in child if _local(t.tag) == "t")
    return "".join(parts)


def _rows(sheet: ET.Element, strings: list[str]) -> list[Row]:
    rows = []
    for row in sheet.iter(f"{{{_MAIN_NS}}}row"):
        cells: Row = {}
        for index, cell in enumerate(row.iter(f"{{{_MAIN_NS}}}c")):
            match = _CELL_REF.match(cell.get("r", ""))
            column = match.group(1) if match else _letters(index + 1)
            value = _cell(cell, strings)
            if not _blank(value):
                cells[column] = value  # type: ignore[assignment]
        rows.append(cells)
    return rows


def _cell(cell: ET.Element, strings: list[str]) -> Value | None:
    kind = cell.get("t", "n")
    if kind == "inlineStr":
        inline = cell.find(f"{{{_MAIN_NS}}}is")
        return _string(inline) if inline is not None else None
    raw = cell.findtext(f"{{{_MAIN_NS}}}v")
    if raw is None:
        return None
    if kind == "s":
        return strings[int(raw)]
    if kind in ("str", "e"):
        return raw if kind == "str" else None
    if kind == "b":
        return int(raw)
    number = float(raw)
    return int(number) if number.is_integer() else number


def _letters(n: int) -> str:
    out = ""
    while n:
        n, rem = divmod(n - 1, 26)
        out = chr(ord("A") + rem) + out
    return out


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(value: Value | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def _blank(value: Value | None) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())
