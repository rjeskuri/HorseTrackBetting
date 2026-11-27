"""A made-up JV-Data specification workbook and export, for every jravan test file (#361).

`write_workbook(path, sections, code_tables)` writes a minimal .xlsx with the standard library
(zipfile and XML strings) in the *shape* of the real workbook, never its content: every record
type, item name, description and code table here is invented (the real workbook is licensed,
D21, and stays under `$HORSETRACK_DATA_DIR/specs/jravan/`).

The shape it reproduces:
- `xl/workbook.xml` lists the sheets in display order, other sheets before `フォーマット` and
  `コード表`; each sheet's file is found through `xl/_rels/workbook.xml.rels`, and the file
  numbers run against the display order (the first sheet is the highest-numbered file).
- Strings are shared (`xl/sharedStrings.xml`), each with `<phoneticPr>`; `Phonetic` adds an
  `<rPh>` reading run, `Rich` writes `<r>` runs, `Inline` an inline string. Numbers are plain
  `<v>` cells or, as in the real sheet, formulas with a cached value (`Formula`). Unused cells
  of a row are written as empty styled cells.
- フォーマット: per section a title row (B `N．title`, F `レコード長`, H the length), the header
  row, then the caller's rows. Build them with `item`, `group`, `sub`, `label`, `BLANK` (a row of
  empty cells) and `SKIP` (a row number not written at all).
- コード表: per table a title row (B `2010．name`), the header row, an optional row of content
  sub-headers from D on, then value rows (C the value, D on the contents, B the width on the
  first value row); `BLANK` inside `Table.rows` is an empty row.

`format_rows` and `code_rows` build a sheet's rows, so a test can put look-alike sheets under
other names (`others=`). `jvd_record` and `write_jvd` build cp932 export records.
"""

from __future__ import annotations

import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import pytest

from horsetrack.core import paths

FORMAT_SHEET = "フォーマット"
CODE_SHEET = "コード表"
DEFAULT_SHEETS = ("表紙", "変更履歴", FORMAT_SHEET, "特記事項", CODE_SHEET, "データ種別一覧")
WORKBOOK_NAME = "JV-Data4901.xlsx"
COLUMNS = "ABCDEFGHIJK"
STYLE = "1"

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
WS_TYPE = REL_NS + "/worksheet"
SS_TYPE = REL_NS + "/sharedStrings"
STYLES_TYPE = REL_NS + "/styles"


@dataclass(frozen=True)
class Phonetic:
    """A shared string with a phonetic `<rPh>` run (the reading must never reach a name)."""

    text: str
    reading: str


@dataclass(frozen=True)
class Rich:
    """A shared string written as rich-text `<r>` runs, optionally with a reading."""

    parts: tuple[str, ...]
    reading: str | None = None


@dataclass(frozen=True)
class Inline:
    """An inline string cell (`t="inlineStr"`)."""

    text: str


@dataclass(frozen=True)
class Formula:
    """A numeric formula cell with its cached value."""

    value: int
    expr: str = "0+0"


Cell = str | int | Phonetic | Rich | Inline | Formula
Row = Mapping[str, Cell] | None

BLANK: Mapping[str, Cell] = {}
SKIP: Row = None


@dataclass(frozen=True)
class Section:
    """One record type on the フォーマット sheet: a title row, the header, then `rows`."""

    number: int
    title: str
    length: int
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class Table:
    """One code table: `rows` are `(value, content, *more contents)` tuples or `BLANK`."""

    code: str
    name: str | Phonetic
    width: int
    rows: tuple[tuple[Cell, ...] | Mapping[str, Cell], ...]
    subheaders: tuple[str, ...] = ()
    sep: str = "．"


# ------------------------------------------------------------------------------------------------
# Row builders (フォーマット)


def item(
    no: int,
    name: Cell,
    start: int,
    length: int,
    *,
    repeat: int | None = None,
    key: str = "",
    initial: Cell | None = None,
    desc: Cell | None = None,
) -> dict[str, Cell]:
    """A top-level item row: B the number, D the key mark, E the name, F the 1-based start."""
    row: dict[str, Cell] = {
        "B": no if no == 1 else Formula(no, f"B{no}+1"),
        "E": name,
        "F": start if no == 1 else Formula(start, "F1+H1"),
        "H": length,
        "I": Formula((repeat or 1) * length, "G1*H1"),
    }
    if key:
        row["D"] = key
    if repeat is not None:
        row["G"] = repeat
    if initial is not None:
        row["J"] = initial
    if desc is not None:
        row["K"] = desc
    return row


def group(
    no: int, name: Cell, start: int, width: int, repeat: int, *, desc: Cell | None = None
) -> dict[str, Cell]:
    """A repeat-group row: H is one repeat's width (a formula in the real sheet)."""
    row = item(no, name, start, width, repeat=repeat, desc=desc)
    row["H"] = Formula(width, "SUM(H2:H3)")
    return row


def sub(
    code: str,
    name: Cell,
    rel: int,
    length: int,
    *,
    repeat: int | None = None,
    initial: Cell | None = None,
    desc: Cell | None = None,
    indent: str = "　",
) -> dict[str, Cell]:
    """A sub-item row: B blank, C the letter code, E the indented name, F `(   n)`."""
    if isinstance(name, str):
        name = indent + name
    row: dict[str, Cell] = {"C": code, "E": name, "F": relative(rel), "H": length}
    if repeat is not None:
        row["G"] = repeat
    if initial is not None:
        row["J"] = initial
    if desc is not None:
        row["K"] = desc
    return row


def label(text: Cell) -> dict[str, Cell]:
    """A row with only column E (a heading such as `<...>`), skipped by the reader."""
    return {"E": text}


def relative(n: int) -> str:
    """A sub-item's relative position as the sheet writes it: `(   1)`, `(  12)`."""
    return "(" + " " * max(1, 4 - len(str(n))) + str(n) + ")"


def letter_code(i: int) -> str:
    """The i-th sub-item code (0-based): a..z, then aa, ab, ..."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    return letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26]


def id_desc(record_type: str) -> str:
    """Item 1's description, which carries the record type ID."""
    return f'"{record_type}" をセット'


def simple_section(
    number: int, record_type: str, title: str, fields: Sequence[tuple[str, int]]
) -> Section:
    """A valid flat section: the ID item, `fields` (name, bytes) back to back, the separator."""
    rows: list[Row] = [
        item(1, "識別子", 1, 2, key="●", initial=record_type, desc=id_desc(record_type))
    ]
    pos = 3
    for i, (name, length) in enumerate(fields, start=2):
        rows.append(item(i, name, pos, length, desc=f"{name}（架空）"))
        pos += length
    rows.append(item(len(fields) + 2, "架空区切", pos, 2, desc="区切（架空）"))
    return Section(number, title, pos + 1, tuple(rows))


# ------------------------------------------------------------------------------------------------
# Sheets


def format_rows(sections: Sequence[Section]) -> list[Row]:
    """The フォーマット sheet: a preamble, then each section's title, header and rows."""
    rows: list[Row] = [{"B": "架空仕様書 フォーマット"}, SKIP, {"B": "注記（架空）"}, BLANK]
    rows.append({"D": 1, "E": "凡例（架空）"})
    rows.append({"D": "○", "E": "キー項目（架空）"})
    rows += [SKIP, SKIP]
    for section in sections:
        rows.append(
            {
                "B": f"{section.number}．{section.title}",
                "F": "レコード長",
                "H": Formula(section.length, "SUM(I1:I9)"),
                "I": "バイト",
            }
        )
        rows.append(
            {
                "B": "項番",
                "D": "キー",
                "E": "項目名",
                "F": "位置",
                "G": "繰返",
                "H": "バイト",
                "I": "合計",
                "J": "初期値",
                "K": "説明",
            }
        )
        rows += list(section.rows)
        rows += [BLANK, SKIP]
    return rows


def code_rows(tables: Sequence[Table]) -> list[Row]:
    """The コード表 sheet: a preamble, then each table's title, header, sub-headers and values."""
    rows: list[Row] = [{"B": "架空コード表"}, SKIP, {"B": "注記（架空）"}, BLANK]
    for table in tables:
        if isinstance(table.name, Phonetic):
            title: Cell = Phonetic(table.code + table.sep + table.name.text, table.name.reading)
        else:
            title = table.code + table.sep + table.name
        rows.append({"B": title})
        rows.append({"B": "バイト数", "C": "値", "D": "内容"})
        if table.subheaders:
            rows.append({COLUMNS[3 + i]: text for i, text in enumerate(table.subheaders)})
        first = True
        for value_row in table.rows:
            if isinstance(value_row, Mapping):
                rows.append(value_row)
                continue
            row: dict[str, Cell] = {"C": value_row[0]}
            for i, content in enumerate(value_row[1:]):
                row[COLUMNS[3 + i]] = content
            if first:
                row["B"] = table.width
                first = False
            rows.append(row)
        rows.append(BLANK)
    return rows


class _Strings:
    def __init__(self) -> None:
        self.index: dict[object, int] = {}
        self.xml: list[str] = []
        self.count = 0

    def add(self, value: str | Phonetic | Rich) -> int:
        self.count += 1
        if value not in self.index:
            self.index[value] = len(self.xml)
            self.xml.append(_si(value))
        return self.index[value]


def _t(text: str) -> str:
    space = ' xml:space="preserve"' if text != text.strip() else ""
    return f"<t{space}>{escape(text)}</t>"


def _si(value: str | Phonetic | Rich) -> str:
    phonetic_pr = '<phoneticPr fontId="1"/>'
    if isinstance(value, Phonetic):
        end = len(value.text)
        rph = f'<rPh sb="0" eb="{end}">{_t(value.reading)}</rPh>'
        return f"<si>{_t(value.text)}{rph}{phonetic_pr}</si>"
    if isinstance(value, Rich):
        runs = "".join(f'<r><rPr><sz val="11"/></rPr>{_t(part)}</r>' for part in value.parts)
        rph = ""
        if value.reading is not None:
            rph = f'<rPh sb="0" eb="{len(value.parts[0])}">{_t(value.reading)}</rPh>'
        return f"<si>{runs}{rph}{phonetic_pr}</si>"
    return f"<si>{_t(value)}{phonetic_pr}</si>"


def _cell(ref: str, value: Cell | None, strings: _Strings) -> str:
    if value is None:
        return f'<c r="{ref}" s="{STYLE}"/>'
    if isinstance(value, Formula):
        return f'<c r="{ref}" s="{STYLE}"><f>{escape(value.expr)}</f><v>{value.value}</v></c>'
    if isinstance(value, bool):
        raise TypeError("no boolean cells in the workbook")
    if isinstance(value, int):
        return f'<c r="{ref}" s="{STYLE}"><v>{value}</v></c>'
    if isinstance(value, Inline):
        return f'<c r="{ref}" s="{STYLE}" t="inlineStr"><is>{_t(value.text)}</is></c>'
    return f'<c r="{ref}" s="{STYLE}" t="s"><v>{strings.add(value)}</v></c>'


def _sheet_xml(rows: Sequence[Row], strings: _Strings) -> str:
    out = []
    for n, row in enumerate(rows, start=2):
        if row is None:
            continue
        cells = "".join(_cell(f"{col}{n}", row.get(col), strings) for col in COLUMNS[1:])
        out.append(f'<row r="{n}" spans="2:11">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<worksheet xmlns="{MAIN_NS}" xmlns:r="{REL_NS}">'
        '<dimension ref="B2:K9"/><sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        '<sheetFormatPr defaultRowHeight="13.5"/>'
        f"<sheetData>{''.join(out)}</sheetData>"
        "</worksheet>"
    )


def write_workbook(
    path: Path,
    sections: Sequence[Section],
    code_tables: Sequence[Table],
    *,
    sheets: Sequence[str] = DEFAULT_SHEETS,
    others: Mapping[str, Sequence[Row]] | None = None,
) -> Path:
    """Write a synthetic JV-Data workbook at `path` and return it.

    `sheets` is the display order; `フォーマット` holds `sections`, `コード表` holds
    `code_tables`, and any other name holds `others[name]` (default: one note row).
    """
    others = others or {}
    strings = _Strings()
    total = len(sheets)
    parts: dict[str, str] = {}
    sheet_entries, rels = [], []
    for i, name in enumerate(sheets):
        if name == FORMAT_SHEET:
            rows = format_rows(sections)
        elif name == CODE_SHEET:
            rows = code_rows(code_tables)
        else:
            rows = list(others.get(name, [{"B": f"{name}（架空）"}]))
        file_no = total - i
        rel_id = f"rId{total - i}"
        parts[f"xl/worksheets/sheet{file_no}.xml"] = _sheet_xml(rows, strings)
        sheet_entries.append(f'<sheet name={quoteattr(name)} sheetId="{100 - i}" r:id="{rel_id}"/>')
        rels.append(
            f'<Relationship Id="{rel_id}" Type="{WS_TYPE}" Target="worksheets/sheet{file_no}.xml"/>'
        )
    rels.append(f'<Relationship Id="rId{total + 1}" Type="{STYLES_TYPE}" Target="styles.xml"/>')
    rels.append(f'<Relationship Id="rId{total + 2}" Type="{SS_TYPE}" Target="sharedStrings.xml"/>')
    rels.reverse()

    parts["xl/workbook.xml"] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<workbook xmlns="{MAIN_NS}" xmlns:r="{REL_NS}">'
        '<bookViews><workbookView activeTab="0"/></bookViews>'
        f"<sheets>{''.join(sheet_entries)}</sheets></workbook>"
    )
    parts["xl/_rels/workbook.xml.rels"] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{PKG_REL_NS}">{"".join(rels)}</Relationships>'
    )
    parts["xl/sharedStrings.xml"] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<sst xmlns="{MAIN_NS}" count="{strings.count}" uniqueCount="{len(strings.xml)}">'
        f"{''.join(strings.xml)}</sst>"
    )
    parts["xl/styles.xml"] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<styleSheet xmlns="{MAIN_NS}">'
        '<fonts count="1"><font><sz val="11"/><name val="Sans"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
        '<cellXfs count="2"><xf/><xf/></cellXfs></styleSheet>'
    )
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{total - i}.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(total)
    )
    parts["[Content_Types].xml"] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f"{overrides}"
        '<Override PartName="/xl/sharedStrings.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        "</Types>"
    )
    parts["_rels/.rels"] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{PKG_REL_NS}">'
        f'<Relationship Id="rId1" Type="{REL_NS}/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    order = ["[Content_Types].xml", "_rels/.rels", "xl/_rels/workbook.xml.rels", "xl/workbook.xml"]
    order += sorted(name for name in parts if name not in order)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in order:
            archive.writestr(name, parts[name])
    return path


# ------------------------------------------------------------------------------------------------
# Export records


def jvd_record(record_type: str, length: int, text: str = "") -> bytes:
    """One cp932 record of `length` bytes, CRLF included: the type, `text`, space padding."""
    body = record_type.encode("ascii") + text.encode("cp932")
    if len(body) > length - 2:
        raise ValueError(f"{text!r} does not fit a {length}-byte {record_type} record")
    return body + b" " * (length - 2 - len(body)) + b"\r\n"


def write_jvd(path: Path, records: Sequence[bytes]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(records))
    return path


def real_export_or_skip() -> tuple[Path, Path]:
    """The real export folder and workbook under the data dir; `pytest.skip` if either is absent."""
    try:
        data = paths.data_dir()
    except paths.DataDirNotSet:
        pytest.skip(f"{paths.DATA_DIR_VAR} is not set: the JRA-VAN export is unavailable")
    export = data / "raw" / "jravan"
    workbook = data / "specs" / "jravan" / WORKBOOK_NAME
    if not (export / "stored").is_dir():
        pytest.skip(f"JRA-VAN export missing: {export}")
    if not workbook.is_file():
        pytest.skip(f"JV-Data specification workbook missing: {workbook}")
    return export, workbook
