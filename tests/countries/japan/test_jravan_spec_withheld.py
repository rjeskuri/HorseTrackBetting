"""Ticket #408: docs/specs/jravan.md publishes no figure from the JRA-VAN export (D21 strict).

Contract pinned here (the coder edits the doc to it). The doc is read as committed; a "code span"
is an inline `...` span, which may run across a line break inside a paragraph (blocks are split
on blank lines). A "clause" is the text between `,`/`;` before whitespace, `(`, `)`, a sentence
end (`.`, `:` before whitespace) and the block's ends.

- Tables: no data row (a `|` row below a `|---|` separator) has a figure cell, i.e. a cell whose
  text outside code spans starts, after an optional `all`/`about`/`~`, with a number (counts,
  spans, years, sizes). A code value or constant in a cell is a code span. The command table and
  the column tables (final_odds, the time series, breeding_horses, produce, lineages) stay, found
  by the names in their first cells.
- Anywhere, code spans included: no thousands-separated number (`1,234`) and no percentage.
  Outside code spans: no `_`-grouped number and no bare number of five or more digits.
- Outside code spans and outside a clause that says `synthetic` (a synthetic-test measurement):
  no size with a byte unit (KB, MB, GB, TB, KiB, MiB, GiB, TiB), no duration in s/ms/min/seconds/
  minutes/hours, no "N million" or "N billion".
- Code spans included, outside a clause that says `synthetic`: no count of export things, i.e. a
  number followed, within two words, by records, rows, races, runners, meetings, days, files,
  groups, horses, jockeys, trainers, lineages, pools, combinations, versions, deletions,
  snapshots or refunds. The check example reads `ok, <n> files`.
- Outside code spans: no bare year 1950-2029 (part of an ISO date YYYY-MM-DD is not bare), and no
  ISO date in a clause naming データ作成年月日.
- So a code constant is written in a code span, as in the code: `2_000` (records), `16 MiB`.
  Kept as they are: field names, code values, `chunk_records=20_000`, `2.20462`, worked example
  ids, ISO dates (the holdout start, owner-decision dates, the D10 cut), "2 bytes", "the 14
  repeats", "up to 20 failing race ids".
- Each measured-run section, found by its `## ` heading prefix (exactly one heading), has one
  block holding `withheld (D21)`, its build report's file name (`jravan-<name>.json`) and
  `jravan-spec-figures.md`, and keeps its non-figure names (pinned per section below); the time
  series section still states the committed holdout start (read from holdout.yaml).
- Every `## ` heading of the pre-fix doc survives.
- Local only (`local` marker, own skip: the data dir or its reports dir absent, or the pre-fix
  commit not in this clone): under `$HORSETRACK_DATA_DIR/reports/` the eight build reports and
  `jravan-spec-figures.md` exist, and the latter holds, section by section, every figure the
  rules above flag anywhere in the pre-fix doc (prose and table cells, every section):
  - A figure is the whole number it is part of, with its unit or `%` (`3.5%`, never `5%`;
    `1,234.5`, never `1,234`), and must stand in the archive as a whole value: not after a digit,
    a `.` or a digit and `,`; not before a digit, or a `.`/`,` and a digit (`3.5%` is not in
    `13.5%`, `12,345` is not in `112,345` or `12,345.6`). Backticks and line breaks do not count.
  - A figure flagged under the pre-fix heading `## S` must be in the archive's text under a line
    that is exactly `## S` (up to the next `## ` line; a heading written more than once has its
    bodies joined). Text above the first `## ` pairs with the archive's text above its first.
  - Except a number the same section of the doc now writes as a code constant (`1,234` in the
    pre-fix doc, `1_234` in a code span now).
  - Spans and dates, outside code spans: every year or ISO date span (`1961-1972`, `2001-02-03
    to 2002-03-04`, `1961 to 2001-02-03`) and every lone ISO date of a pre-fix section that the
    same section of the doc no longer states stands whole in the archive's section, both ends
    (not run on into another date, year or range).
  - Table cells: every cell of a pre-fix table that has a figure row stands in a table row of
    the archive's section with the same row label (first cell) under the same column header.
  - Outcomes: every sentence or list item of a pre-fix section that the same section of the doc
    no longer holds (a number now a code constant read as one), and whose block holds a figure
    or which names the export, stands verbatim in the archive's section (whitespace and
    backticks aside); so does every row of a table without a figure row that names the export.
    A reworded sentence that neither sits by a figure nor names the export is not archived.
  - The rules are checked on made-up text without the data dir; the real archive check alone is
    `local`.

D21: this file holds no figure from the export. Figures are found by pattern, never by value;
every example below is made up. Every check on the doc or the private file fails through
`fail_unless` (no assertion rewriting, no traceback), with line numbers, rule names and counts
only, so a failing run does not reprint a figure.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC_REL = "docs/specs/jravan.md"
DOC = REPO_ROOT / DOC_REL
WITHHELD = "withheld (D21)"
FIGURES_FILE = "jravan-spec-figures.md"
# The last commit whose doc still publishes the figures (#375's merge): the local test reads the
# removed figures from it. A public commit id, not a figure. Pointing at it is deliberate: the
# owner answered on #408 (2026-11-26) "fix forward only", with no history rewrite and no
# force-push, so the figures stay in history and this commit stays reachable.
PRE_FIX = "bc138f2d957fcf09f008355a950211de30d41cfa"

# Measured-run sections: heading prefix -> the build report its command writes.
SECTIONS = {
    "## RA": "jravan-races.json",
    "## SE": "jravan-runners.json",
    "## HR": "jravan-dividends.json",
    "## O1-O6": "jravan-final-odds.json",
    "## H1 and H6": "jravan-votes.json",
    "## The 0B41/0B42 time series": "jravan-timeseries.json",
    "## UM, KS, CH": "jravan-masters.json",
    "## HN, SK, BT": "jravan-bloodlines.json",
}

# Every `## ` heading of the pre-fix doc.
HEADINGS = (
    "## Where the data and the specification live",
    "## What is built",
    "## Layouts (#361)",
    "## Decoding (#362)",
    "## RA: races and meetings (#364)",
    "## SE: runners (#365)",
    "## HR: payouts (#366)",
    "## O1-O6: final odds (#367)",
    "## H1 and H6: pool votes (#368)",
    "## The 0B41/0B42 time series (#369)",
    "## UM, KS, CH: horse, jockey and trainer masters (#370)",
    "## HN, SK, BT: bloodlines (#371)",
    "## The dump of unmapped families (#363)",
    "## Reconciliation with the netkeiba store (#373)",
    "## The model-input coverage audit (#374)",
)

# Names that stay in each section's prose (not figures).
KEPT_NAMES = {
    "## RA": ("build_races", "versions_dropped", "non_jra_overseas", "rail_setting"),
    "## SE": ("build_runners", "finish_time_s", "2.20462", "lbw_lengths", "placing_code"),
    "## HR": (
        "build_dividends",
        "dividends_from_records",
        "不成立フラグ",
        "特払フラグ",
        "返還フラグ",
    ),
    "## O1-O6": ("build_final_odds", "odds_rows", "QUINELLA_PLACE", "TIERCE", "LabelOnlyTable"),
    "## H1 and H6": ("build_pool_votes", "votes_rows", "POOL_VOTES_SCHEMA", "chunk_records=500"),
    "## The 0B41/0B42 time series": ("build_day", "build_all", "_build.json", "PARSER_VERSION"),
    "## UM, KS, CH": ("build_masters", "JOCKEYS_SCHEMA", "TRAINERS_SCHEMA", "in_view"),
    "## HN, SK, BT": ("build_bloodlines", "BREEDING_HORSES_SCHEMA", "LINEAGES_SCHEMA"),
}

# Names that stay in the first cells of each section's tables (the command and column tables).
KEPT_TABLE_NAMES = {
    "## What is built": (
        "check",
        "races",
        "runners",
        "dividends",
        "final-odds",
        "votes",
        "timeseries",
        "masters",
        "bloodlines",
        "dump",
        "reconcile",
    ),
    "## O1-O6": (
        "jurisdiction",
        "pool",
        "combination",
        "odds",
        "odds_hi",
        "fav_rank",
        "investment",
        "status_raw",
        "published_utc",
        "data_kubun",
    ),
    "## The 0B41/0B42 time series": (
        "snapshot_utc",
        "fetched_utc",
        "post_utc",
        "minutes_to_post",
        "is_final_capture",
        "dividend",
        "is_refund",
    ),
    "## HN, SK, BT": (
        "breeding_id",
        "horse_id",
        "birth_year",
        "import_type_code",
        "sire_breeding_id",
        "birth_date",
        "breeder_code",
        "pedigree_breeding_ids",
        "lineage_id",
        "lineage_name",
        "lineage_description",
    ),
}

# ------------------------------------------------------------------------------------------------
# Reading the doc


def fail_unless(ok: bool, message: str) -> None:
    """Fail with `message` alone. A plain `assert` on doc or private-file text would let pytest's
    assertion rewriting print the compared values, which can be the withheld figures."""
    if not ok:
        pytest.fail(message, pytrace=False)


CODE_SPAN = re.compile(r"`[^`]*`")
SEPARATOR_ROW = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
# A number that stands alone: not inside a word (date32, D21, 0B41, 3代), a ticket (#364), a
# range's second end, a time or a path.
NUMBER = r"(?<![\w.,:/#-])\d+(?:[.,]\d+)*(?!\w)"


def mask_code(text: str) -> str:
    """Code spans replaced by `C`s of the same length, line breaks kept, so offsets still hold."""
    return CODE_SPAN.sub(lambda m: re.sub(r"[^\n]", "C", m.group()), text)


def blocks(text: str) -> Iterator[tuple[int, str]]:
    """(first line number, text) of each blank-line-separated block."""
    start, lines = 1, []
    for number, line in enumerate(text.split("\n"), start=1):
        if line.strip():
            if not lines:
                start = number
            lines.append(line)
        elif lines:
            yield start, "\n".join(lines)
            lines = []
    if lines:
        yield start, "\n".join(lines)


CLAUSE_END = re.compile(r"[;()]|,(?=\s)|[.:](?=\s)")


def clause_at(text: str, start: int, end: int) -> str:
    left = max((m.end() for m in CLAUSE_END.finditer(text, 0, start)), default=0)
    right = next((m.start() for m in CLAUSE_END.finditer(text, end)), len(text))
    return text[left:right]


def synthetic(text: str, start: int, end: int) -> bool:
    return "synthetic" in clause_at(text, start, end).lower()


# ------------------------------------------------------------------------------------------------
# The rules. Each finder takes a block and returns the (start, end) offsets of its figures.

Finder = Callable[[str], list[tuple[int, int]]]

THOUSANDS = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?![\d])")
PERCENT = re.compile(r"\d\s?%")
GROUPED_OR_LONG = re.compile(r"(?<![\w.,:/#-])(?:\d+(?:_\d{3})+|\d{5,})(?!\w)")
SIZE = re.compile(NUMBER + r"\s?(?:KB|MB|GB|TB|KiB|MiB|GiB|TiB)\b")
DURATION = re.compile(NUMBER + r"\s?(?:s|ms|min|seconds|minutes|hours)\b")
MILLION = re.compile(NUMBER + r"\s+(?:million|billion)\b")
COUNT = re.compile(
    NUMBER
    + r"\s+(?:[^\W\d][\w-]*\s+){0,2}"
    + r"(?:records|rows|races|runners|meetings|days|files|groups|horses|jockeys|trainers"
    + r"|lineages|pools|combinations|versions|deletions|snapshots|refunds)\b"
)
BARE_YEAR = re.compile(r"(?<![\w.,:/#-])(?:19[5-9]\d|20[0-2]\d)(?!\w|-\d\d-\d\d)")
ISO_DATE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")


def find_thousands_or_percent(block: str) -> list[tuple[int, int]]:
    found = [m.span() for m in THOUSANDS.finditer(block)]
    found += [m.span() for m in PERCENT.finditer(block)]
    found += [m.span() for m in GROUPED_OR_LONG.finditer(mask_code(block))]
    return found


def _outside_code_and_synthetic(pattern: re.Pattern) -> Finder:
    def find(block: str) -> list[tuple[int, int]]:
        masked = mask_code(block)
        return [m.span() for m in pattern.finditer(masked) if not synthetic(masked, *m.span())]

    return find


find_size = _outside_code_and_synthetic(SIZE)
find_duration = _outside_code_and_synthetic(DURATION)
find_million = _outside_code_and_synthetic(MILLION)


def find_count(block: str) -> list[tuple[int, int]]:
    plain = block.replace("`", " ")  # code spans included; a backtick reads as a space
    return [m.span() for m in COUNT.finditer(plain) if not synthetic(plain, *m.span())]


def find_year_or_creation_date(block: str) -> list[tuple[int, int]]:
    masked = mask_code(block)
    found = [m.span() for m in BARE_YEAR.finditer(masked)]
    found += [
        m.span()
        for m in ISO_DATE.finditer(masked)
        if "データ作成年月日" in clause_at(masked, *m.span())
    ]
    return found


FIGURE_CELL = re.compile(r"^\s*(?:(?:all|about|~)\s*)?[+-]?" + NUMBER)


def cells(masked_row: str) -> list[str]:
    inner = masked_row.strip()
    inner = inner[1:] if inner.startswith("|") else inner
    inner = inner[:-1] if inner.endswith("|") else inner
    return inner.split("|")


def table_data_rows(text: str) -> Iterator[tuple[int, str, str]]:
    """(line number, raw row, masked row) of every table data row (below a separator row)."""
    lines = text.split("\n")
    in_body = False
    for number, line in enumerate(lines, start=1):
        if not line.lstrip().startswith("|"):
            in_body = False
        elif SEPARATOR_ROW.match(line):
            in_body = True
        elif in_body:
            yield number, line, mask_code(line)


def figure_rows(text: str) -> list[int]:
    return [
        number
        for number, _raw, masked in table_data_rows(text)
        if any(FIGURE_CELL.match(cell) for cell in cells(masked))
    ]


PROSE_RULES: dict[str, Finder] = {
    "thousands_or_percent": find_thousands_or_percent,
    "size": find_size,
    "duration": find_duration,
    "million": find_million,
    "count": find_count,
    "year_or_creation_date": find_year_or_creation_date,
}


NUMBER_HEAD = re.compile(r"(?:\d[.,]?)+$")
NUMBER_TAIL = re.compile(r"(?:[.,]?\d)+")


def whole_token(text: str, start: int, end: int) -> tuple[int, int]:
    """A hit grown to the whole number it is part of: `5%` in `3.5%`, `1,234` in `1,234.5`."""
    if text[start : start + 1].isdecimal():
        head = NUMBER_HEAD.search(text, 0, start)
        start = head.start() if head else start
    if text[end - 1 : end].isdecimal():
        tail = NUMBER_TAIL.match(text, end)
        end = tail.end() if tail else end
    return start, end


def hits(text: str, finder: Finder) -> list[tuple[int, str]]:
    """(line number, whole token) of every figure the finder flags in the doc's blocks."""
    out = []
    for first_line, block in blocks(text):
        for span in finder(block):
            start, end = whole_token(block, *span)
            out.append((first_line + block.count("\n", 0, start), block[start:end]))
    return out


# ------------------------------------------------------------------------------------------------
# Sections


def headings(text: str) -> list[str]:
    return [line.rstrip() for line in text.split("\n") if line.startswith("## ")]


def section(text: str, prefix: str) -> str:
    """The body of the one `## ` section whose heading starts with `prefix` (whole word)."""
    lines = text.split("\n")
    starts = [
        i
        for i, line in enumerate(lines)
        if line.startswith(prefix) and not line[len(prefix) : len(prefix) + 1].isalnum()
    ]
    fail_unless(len(starts) == 1, f"{len(starts)} `## ` headings start with {prefix!r}")
    body = []
    for line in lines[starts[0] + 1 :]:
        if line.startswith("## "):
            break
        body.append(line)
    return "\n".join(body)


def first_cell_names(text: str) -> set[str]:
    names: set[str] = set()
    for _number, raw, masked in table_data_rows(text):
        first = cells(masked)[0]
        offset = raw.index("|") + 1 if raw.lstrip().startswith("|") else 0
        names |= set(re.findall(r"`([^`]+)`", raw[offset : offset + len(first)]))
    return names


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text(encoding="utf-8")


def assert_sections_kept(text: str) -> None:
    missing = [h for h in HEADINGS if h not in headings(text)]
    fail_unless(not missing, f"headings lost: {missing}")


# ------------------------------------------------------------------------------------------------
# Rule cases: made-up examples each finder must flag and must pass.

RULE_CASES = {
    "thousands_or_percent": (
        ["12,345 rows", "about 6,789,012 snapshots", "on 67.8% of records", "`3,456`", "9 %",
         "a 54321 total", "4_321 rows"],
        ["`4,7` on O2", "`7,4,1`", "1, 2 or 3 legs", "`chunk_records=40_000`", "`202501020304`",
         "JV-Data 4.9.0.1", "`2.20462`", "jra:2031-01-02:QQV:07"],
    ),
    "size": (
        ["a 5 GB file", "peak RSS 3.33 GB", "a 1.7 GiB file", "about 250 MB peak",
         "`NAME` (8 MiB)"],
        ["(about 512 MB above the interpreter on a 1.5 GB synthetic file of 50 KB records)",
         "`CHUNK_BYTES` (`8 MiB`)", "a multi-GB file", "the first 2 bytes the record type"],
    ),
    "duration": (
        ["about 77 s, a peak", "(321 s wall", "the rerun (9 s)", "took 3.5 minutes"],
        ["a 9 s synthetic run", "`timeout=30 s`", "in minutes, negative after the post",
         "the 14 repeats"],
    ),
    "million": (
        ["(3.4 to 5.6 million a year)", "about 2 billion rows"],
        ["`10 million`", "on a synthetic 3 million rows"],
    ),
    "count": (
        ["321 races", "(41 abandoned races)", "77 H9 races", "12 H9 year files", "33 row groups",
         "`stored/RACE/ZZ: ok, 17 files`", "88 placed runners", "5 trainer runners",
         "9 meetings", "on 345 race days", "92 deletions"],
        ["`<n>` files", "`2_000` records", "up to 20 failing race ids", "the 14 repeats",
         "1, 2 or 3 equal-width legs", "(0B41 holds O1 records, 0B42 O2)",
         "the 2-byte record separator", "kubun 9, else run", "of 64 KB synthetic records",
         "a horse two digits", "2 前日売最終, 4 確定"],
    ),
    "year_or_creation_date": (
        ["from 1977 on", "(1981-1999)", "every year before 2011", "| 2020 |",
         "(latest データ作成年月日 2001-02-03)", "all before 1960."],
        ["the 2025-01-02 holdout start", "pre-2001-02-03 days", "code table `'2010'`",
         "code tables 2203 and 2204", "`jra:2031-01-02:QQV:07`", "#1999", "D21",
         "the export of 2001-02-03", "`created='データ作成年月日'`"],
    ),
}  # fmt: skip


@pytest.mark.parametrize("rule", sorted(PROSE_RULES))
def test_the_doc_states_no_figure_of_the_export(doc, rule):
    finder = PROSE_RULES[rule]
    flags, passes = RULE_CASES[rule]
    missed = [s for s in flags if not finder(s)]
    assert not missed, f"{rule}: made-up figures not flagged: {missed}"
    wrong = [(s, [s[a:b] for a, b in finder(s)]) for s in passes]
    wrong = [(s, found) for s, found in wrong if found]
    assert not wrong, f"{rule}: kept text flagged: {wrong}"

    assert_sections_kept(doc)
    lines = sorted({line for line, _text in hits(doc, finder)})
    fail_unless(not lines, f"{DOC_REL}: {rule} figures on lines {lines} (D21: withhold them)")


def test_no_table_row_holds_a_figure(doc):
    made_up = (
        "| pool | rows | first year |\n|---|---|---|\n| WIN | 12,345 | 1977 |\n"
        "| 1981-1999 (7 year files) | x | y |\n| PLACE | all 7 year files | z |\n"
        "| WIN | withheld (D21) | 1977 |\n\n"
        "| column | type | from |\n|---|---|---|\n| `odds` | float64 | x 100 (yen) |\n"
        "| `ids` | list<string> | 3代血統繁殖登録番号, the 14 repeats |\n| `code` | `7` | `0` |\n"
    )
    assert figure_rows(made_up) == [3, 4, 5, 6], figure_rows(made_up)

    assert_sections_kept(doc)
    for prefix, names in KEPT_TABLE_NAMES.items():
        lost = sorted(set(names) - first_cell_names(section(doc, prefix)))
        fail_unless(not lost, f"{prefix}: table rows lost: {lost}")
    rows = figure_rows(doc)
    fail_unless(
        not rows, f"{DOC_REL}: table rows with figures on lines {rows} (D21: withhold them)"
    )


@pytest.mark.parametrize("prefix", list(SECTIONS))
def test_each_measured_section_names_its_private_reports(doc, prefix):
    report = SECTIONS[prefix]
    body = section(doc, prefix)
    kept = [name for name in KEPT_NAMES[prefix] if name not in body]
    fail_unless(not kept, f"{prefix}: names lost: {kept}")
    if prefix == "## The 0B41/0B42 time series":
        start = load_country_yaml("japan", "holdout")["holdout"]["start"]
        fail_unless(str(start) in body, f"{prefix}: the holdout start is no longer stated")

    needed = (WITHHELD, report, FIGURES_FILE)
    holding = [b for _line, b in blocks(body) if all(token in b for token in needed)]
    fail_unless(bool(holding), f"{prefix}: no paragraph holds all of {needed}")


# ------------------------------------------------------------------------------------------------
# Local: the private reports hold what the doc withholds.

WHITESPACE = re.compile(r"\s+")


def normalized(text: str) -> str:
    return WHITESPACE.sub(" ", text.replace("`", ""))


def flagged_figures(text: str) -> dict[str, set[int]]:
    """Every figure the rules flag in the whole doc, normalized -> the lines it is on: the prose
    rules' hits and the numbers in table figure cells."""
    found: dict[str, set[int]] = {}
    for finder in PROSE_RULES.values():
        for line, figure in hits(text, finder):
            found.setdefault(normalized(figure).strip(), set()).add(line)
    for line, _raw, masked in table_data_rows(text):
        for cell in cells(masked):
            if FIGURE_CELL.match(cell):
                for figure in re.findall(NUMBER, cell):
                    found.setdefault(figure, set()).add(line)
    return found


GROUPED = re.compile(r"\d{1,3}(?:,\d{3})+(?![.,]?\d)")
CODE_CONSTANT = re.compile(r"`(\d{1,3}(?:_\d{3})+)`")


def code_constants(text: str) -> set[str]:
    """The numbers the doc writes as code constants (`1_234` in a code span), digits only."""
    return {m.group(1).replace("_", "") for m in CODE_CONSTANT.finditer(text)}


def written_as_constant(figure: str, constants: set[str]) -> bool:
    """A figure that opens with a `,`-grouped number the doc now writes as a code constant."""
    m = GROUPED.match(figure)
    return m is not None and m.group().replace(",", "") in constants


def holds(text: str, figure: str) -> bool:
    """`figure` stands in `text` as a whole value, not as part of a longer number."""
    whole = r"(?<![\d.])(?<!\d,)" + re.escape(figure) + r"(?![.,]?\d)"
    return re.search(whole, text) is not None


def section_bodies(text: str) -> dict[str, str]:
    """`## ` heading line -> the text under it, up to the next `## ` line ("" for the text above
    the first); a heading written more than once has its bodies joined."""
    bodies: dict[str, list[str]] = {}
    owner = ""
    for line in text.split("\n"):
        if line.startswith("## "):
            owner = line.rstrip()
            bodies.setdefault(owner, [])
        else:
            bodies.setdefault(owner, []).append(line)
    return {heading: "\n".join(lines) for heading, lines in bodies.items()}


def figures_by_section(text: str) -> dict[tuple[str, str], set[int]]:
    """(`## ` heading the figure is under, figure) -> its lines, for every flagged figure."""
    owner = line_headings(text)
    out: dict[tuple[str, str], set[int]] = {}
    for figure, lines in flagged_figures(text).items():
        for line in lines:
            out.setdefault((owner[line - 1], figure), set()).add(line)
    return out


def unarchived_figures(old: str, archive: str, new: str) -> dict[tuple[str, str], set[int]]:
    """The figures of `old` that the archive's section of the same heading does not hold as a
    whole value, and that the same section of `new` does not write as a code constant."""
    held = {heading: normalized(body) for heading, body in section_bodies(archive).items()}
    constants = {heading: code_constants(body) for heading, body in section_bodies(new).items()}
    return {
        (heading, figure): lines
        for (heading, figure), lines in figures_by_section(old).items()
        if not holds(held.get(heading, ""), figure)
        and not written_as_constant(figure, constants.get(heading, set()))
    }


def line_headings(text: str) -> list[str]:
    """The `## ` heading each line of `text` is under ("" above the first), by line index."""
    owner, heading = [], ""
    for line in text.split("\n"):
        heading = line.rstrip() if line.startswith("## ") else heading
        owner.append(heading)
    return owner


# Spans and dates: a year or ISO date (`2001-02-03`, `2001-02-03/04`) to another, by `-`, `–` or
# `to`, or a lone ISO date; outside code spans. Both ends are one value.
YEAR_OR_DATE = r"(?:\d{4}-\d{2}-\d{2}(?:/\d{2})?|(?:19|20)\d{2})"
SPAN_OR_DATE = re.compile(
    rf"(?<![\w.,:/#])(?:{YEAR_OR_DATE}(?:\s*[-–]\s*|\s+to\s+){YEAR_OR_DATE}"
    r"|\d{4}-\d{2}-\d{2}(?:/\d{2})?)(?![\w]|[-/.,]\d)"
)


def holds_span(text: str, span: str) -> bool:
    """`span` stands in `text` whole: neither end runs on into another date, year or range."""
    whole = r"(?<![\w.,:/#])(?<!\d[-–])" + re.escape(span) + r"(?![\w]|[-/.,]\d)"
    return re.search(whole, text) is not None


def spans_and_dates(text: str) -> list[tuple[int, str]]:
    """(line, normalized span or date) of every span and ISO date outside code spans."""
    out = []
    for first_line, block in blocks(text):
        for m in SPAN_OR_DATE.finditer(mask_code(block)):
            line = first_line + block.count("\n", 0, m.start())
            out.append((line, WHITESPACE.sub(" ", m.group())))
    return out


def unarchived_spans(old: str, archive: str, new: str) -> list[tuple[int, str, str]]:
    """(line, heading, span) of every span or ISO date of `old` that the same section of `new`
    no longer states and the archive's section of that heading does not hold whole."""
    owner = line_headings(old)
    archived = {heading: normalized(body) for heading, body in section_bodies(archive).items()}
    kept = {heading: normalized(body) for heading, body in section_bodies(new).items()}
    out = []
    for line, span in spans_and_dates(old):
        heading = owner[line - 1]
        if not holds_span(kept.get(heading, ""), span) and not holds_span(
            archived.get(heading, ""), span
        ):
            out.append((line, heading, span))
    return out


def row_cells(raw: str) -> tuple[str, ...]:
    """The normalized cells of a `|` row; a `|` inside a code span does not split."""
    masked = mask_code(raw)
    edges = [-1, *(i for i, ch in enumerate(masked) if ch == "|"), len(raw)]
    parts = [raw[a + 1 : b] for a, b in zip(edges, edges[1:], strict=False)]
    if masked.lstrip().startswith("|"):
        parts = parts[1:]
    if masked.rstrip().endswith("|"):
        parts = parts[:-1]
    return tuple(normalized(part).strip() for part in parts)


def tables(text: str) -> list[dict]:
    """Every table of `text`: its `header` cells, its data `rows` as (line, cells), and whether
    a row holds a figure cell (`figures`)."""
    lines = text.split("\n")
    out: list[dict] = []
    current = None
    for number, line in enumerate(lines, start=1):
        if not line.lstrip().startswith("|"):
            current = None
        elif SEPARATOR_ROW.match(line):
            above = lines[number - 2] if number >= 2 else ""
            header = row_cells(above) if above.lstrip().startswith("|") else ()
            current = {"header": header, "rows": [], "figures": False}
            out.append(current)
        elif current is not None:
            current["rows"].append((number, row_cells(line)))
            if any(FIGURE_CELL.match(cell) for cell in cells(mask_code(line))):
                current["figures"] = True
    return out


def table_cells(text: str, figures_only: bool) -> Iterator[tuple[int, str, str, str, str]]:
    """(line, heading, column header, row label, value) of every cell right of the label."""
    owner = line_headings(text)
    for table in tables(text):
        if figures_only and not table["figures"]:
            continue
        for line, row in table["rows"]:
            for header, value in zip(table["header"][1:], row[1:], strict=False):
                yield line, owner[line - 1], header, row[0], value


def unarchived_cells(old: str, archive: str) -> list[tuple[int, str, str]]:
    """(line, heading, `header | label`) of every cell of a figure row of `old` that no table
    row of the archive's section of that heading holds under the same row label and column
    header."""
    held = {(h, header, label, value) for _, h, header, label, value in table_cells(archive, False)}
    return [
        (line, heading, f"{header} | {label}")
        for line, heading, header, label, value in table_cells(old, True)
        if (heading, header, label, value) not in held
    ]


SENTENCE_BREAK = re.compile(r"(?<=[.。])\s+|\n(?=\s*[-*] )")


def sentences(block: str) -> Iterator[tuple[int, str]]:
    """(offset, text) of each sentence or list item of a prose block."""
    start = 0
    for m in SENTENCE_BREAK.finditer(block):
        yield start, block[start : m.start()]
        start = m.end()
    yield start, block[start:]


def as_constants(text: str, constants: set[str]) -> str:
    """`text` with each `,`-grouped number the doc now writes as a code constant in `_` form."""

    def swap(m: re.Match) -> str:
        return m.group().replace(",", "_") if m.group().replace(",", "") in constants else m.group()

    return GROUPED.sub(swap, text)


def table_rows(body: str) -> set[tuple[str, ...]]:
    return {row for table in tables(body) for _line, row in table["rows"]}


def unarchived_outcomes(old: str, archive: str, new: str) -> list[tuple[int, str, str]]:
    """(line, heading, unit) of every sentence, list item or column-table row of `old` that the
    same section of `new` no longer holds, that states what a run found (its block holds a
    figure, or it names the export), and that the archive's section of that heading does not
    hold verbatim (whitespace and backticks aside)."""
    owner = line_headings(old)
    new_bodies, archive_bodies = section_bodies(new), section_bodies(archive)
    out = []
    for first_line, block in blocks(old):
        heading = owner[first_line - 1]
        new_body, archived = new_bodies.get(heading, ""), archive_bodies.get(heading, "")
        if block.lstrip().startswith("|"):
            block_tables = tables(block)
            if any(table["figures"] for table in block_tables):
                continue  # figure tables: cell by cell, `unarchived_cells`
            kept, held = table_rows(new_body), table_rows(archived)
            for table in block_tables:
                for line, row in table["rows"]:
                    if "export" in " ".join(row).lower() and row not in kept | held:
                        out.append((first_line + line - 1, heading, " | ".join(row)))
            continue
        figured = any(finder(block) for finder in PROSE_RULES.values())
        constants = code_constants(new_body)
        kept, held = normalized(new_body), normalized(archived)
        for offset, text in sentences(block):
            unit = as_constants(normalized(text).strip(), constants)
            if not unit or not (figured or "export" in unit.lower()):
                continue
            if unit not in kept and unit not in held:
                out.append((first_line + block.count("\n", 0, offset), heading, unit))
    return out


def unarchived(old: str, archive: str, new: str) -> dict[tuple[str, str, str], set[int]]:
    """(heading, rule, item) -> pre-fix lines, for everything the archive fails to hold: the
    flagged figures (`figure`), the removed spans and dates (`span`), the figure-table cells by
    row and column (`cell`) and the removed sentences that state an outcome (`outcome`)."""
    out: dict[tuple[str, str, str], set[int]] = {}
    for (heading, figure), lines in unarchived_figures(old, archive, new).items():
        out.setdefault((heading, "figure", figure), set()).update(lines)
    found = [
        ("span", unarchived_spans(old, archive, new)),
        ("cell", unarchived_cells(old, archive)),
        ("outcome", unarchived_outcomes(old, archive, new)),
    ]
    for rule, items in found:
        for line, heading, item in items:
            out.setdefault((heading, rule, item), set()).add(line)
    return out


def flagged_lines(missing: dict) -> set[int]:
    return {line for lines in missing.values() for line in lines}


# A made-up pre-fix doc, its withheld version and a faithful archive. Every value is invented.
SYN_OLD = (
    "intro\n"  # 1
    "\n"
    "## A (#1)\n"  # 3
    "\n"
    "The toy run on the export (2001-02-03 to 2002-03-04, 1961-1972): 11,111 rows.\n"  # 5
    "The second pass changed no file (48 s).\n"  # 6
    "Every gadget has a lid.\n"  # 7
    "\n"
    "| pool | rows | first year |\n"  # 9
    "|---|---|---|\n"
    "| WIN | 22,222 | 1961 |\n"  # 11
    "| PLACE | 33,333 | 1963 |\n"  # 12
    "\n"
    "A gadget without a lid writes no row and is counted; the export holds none.\n"  # 14
    "\n"
    "Reads in chunks of at most 4,321 records.\n"  # 16
    "\n"
    "Days run 2003-04-05 to 2004-05-06 in the toy set.\n"  # 18
    "\n"
    "| column | from |\n"  # 20
    "|---|---|\n"
    "| `stamp` | null: the export has no stamp |\n"  # 22
    "| `kept` | as printed |\n"  # 23
    "\n"
    "## B (#2)\n"  # 25
    "\n"
    "A plain rule sentence.\n"  # 27
)
SYN_NEW = (
    "intro\n\n## A (#1)\n\nThe toy run on the export: withheld (D21), in `toy-figures.md`.\n\n"
    "A gadget without a lid writes no row and is counted.\n\n"
    "Reads in chunks of at most `4_321` records.\n\n"
    "The toy days: withheld (D21).\n\n"
    "| column | from |\n|---|---|\n| `stamp` | null: the layout has no stamp |\n"
    "| `kept` | as printed |\n\n"
    "## B (#2)\n\nA plain rule sentence, reworded.\n"
)
# Verbatim, wrapped differently and with a code span; the sentences the new doc still holds
# (the code constant, the kept row) and a reworded plain rule sentence need not be archived.
SYN_ARCHIVE = (
    "## A (#1)\n\n"
    "The toy run on the export (2001-02-03 to 2002-03-04,\n1961-1972): `11,111` rows. The second"
    " pass changed\nno file (48 s). Every gadget has a lid.\n\n"
    "| pool | rows | first year |\n|---|---:|---|\n| WIN | 22,222 | 1961 |\n"
    "| PLACE | 33,333 | 1963 |\n\n"
    "A gadget without a lid writes no row and is counted; the export holds\nnone.\n\n"
    "Days run 2003-04-05 to 2004-05-06 in the toy set.\n\n"
    "| column | from |\n|---|---|\n| `stamp` | null: the export has no stamp |\n"
)
ALL_SYN_LINES = {5, 6, 7, 11, 12, 14, 18, 22}

# Each case edits the faithful archive; the pre-fix lines the check must then report.
ARCHIVE_CASES = {
    "faithful": ([], set()),
    # Spans: both endpoints, year spans and ISO date spans (finding 1).
    "year_span_end_changed": ([("1961-1972", "1961-1979")], {5}),
    "year_span_start_changed": ([("1961-1972", "1962-1972")], {5}),
    "date_span_end_changed": ([("2002-03-04", "2002-03-14")], {5}),
    "date_span_start_changed": ([("2001-02-03", "2001-02-13")], {5}),
    "plain_date_span_end_changed": ([("2004-05-06", "2004-05-07")], {18}),
    "plain_date_span_start_dropped": ([("Days run 2003-04-05 to", "Days run to")], {18}),
    # Table cells: by row label and column header (finding 2).
    "counts_swapped": (
        [("| WIN | 22,222 |", "| WIN | 33,333 |"), ("| PLACE | 33,333 |", "| PLACE | 22,222 |")],
        {11, 12},
    ),
    "first_years_swapped": (
        [("22,222 | 1961", "22,222 | 1963"), ("33,333 | 1963", "33,333 | 1961")],
        {11, 12},
    ),
    "column_renamed": ([("| pool | rows |", "| pool | races |")], {11, 12}),
    "row_relabelled": ([("| PLACE |", "| SHOW |")], {12}),
    # Outcomes in words (finding 3).
    "outcome_words_dropped_figure_kept": (
        [("The second pass changed\nno file (48 s).", "No file (48 s).")],
        {6},
    ),
    "word_only_sentence_dropped": ([(" Every gadget has a lid.", "")], {7}),
    "export_clause_dropped": ([("is counted; the export holds\nnone.", "is counted.")], {14}),
    "export_table_row_dropped": ([("| `stamp` | null: the export has no stamp |\n", "")], {22}),
    # Everything under a heading that is not the same text.
    "wrong_heading": ([("## A (#1)", "## A")], ALL_SYN_LINES),
}


@pytest.mark.parametrize("case", list(ARCHIVE_CASES))
def test_archive_check_catches_a_misplaced_or_dropped_figure(case):
    edits, expected = ARCHIVE_CASES[case]
    archive = SYN_ARCHIVE
    for before, after in edits:
        assert before in archive, (case, before)
        archive = archive.replace(before, after)
    found = flagged_lines(unarchived(SYN_OLD, archive, SYN_NEW))
    assert found == expected, (case, sorted(found))


def test_archive_check_holds_whole_figures_by_section():
    made_up = (
        "| pool | rows |\n|---|---|\n| WIN | 12,345 |\n\nin chunks of 4,321 records, 6,789 races"
        " on 2.5 GB (1977-1999)\n\nan ok, 17 files check, 3.5% of `81,234`, a 1,234.5 kg mean,"
        " 0.25 % up"
    )
    found = flagged_figures(made_up)
    expected = {"12,345", "4,321", "4,321 records", "6,789", "6,789 races", "2.5 GB", "1977"}
    expected |= {"17 files", "3.5%", "81,234", "1,234.5", "0.25 %"}
    assert set(found) == expected, sorted(found)
    assert found["6,789 races"] == {5}, found["6,789 races"]
    constants = code_constants("chunks of `4_321` (records), `12_345_678`, `9_9`, 55_555")
    assert constants == {"4321", "12345678"}, constants
    kept = [f for f in sorted(found) if written_as_constant(f, constants)]
    assert kept == ["4,321", "4,321 records"], kept
    assert not written_as_constant("1,234.5", {"1234"})

    whole_values = [
        ("a 3.5%.", "3.5%", True),
        ("13.5% and 8.5%", "3.5%", False),
        ("8.5% of rows", "5%", False),
        ("from 1977.", "1977", True),
        ("(1977-1994)", "1977", True),
        ("11977 or 1977.5", "1977", False),
        ("about 12,345,", "12,345", True),
        ("112,345 and 12,345,678 and 12,345.6", "12,345", False),
        ("1,977", "977", False),
        ("on 2.5 GB", "2.5 GB", True),
        ("on 12.5 GB", "2.5 GB", False),
        ("122 files", "22 files", False),
    ]
    wrong = [(text, figure) for text, figure, ok in whole_values if holds(text, figure) != ok]
    assert not wrong, wrong

    old = "intro\n\n## A (#1)\n\n3.5% of rows, 12,345 races.\n\n## B (#2)\n\nfrom 1977.\n"
    a, b = "## A (#1)", "## B (#2)"
    assert figures_by_section(old) == {
        (a, "3.5%"): {5},
        (a, "12,345"): {5},
        (a, "12,345 races"): {5},
        (b, "1977"): {9},
    }, figures_by_section(old)
    archives = [
        # Verbatim, across a line break and with code spans: all held.
        ("## A (#1)\n\n3.5% of\nrows, `12,345` races.\n\n## B (#2)\n\nfrom 1977.\n", "", set()),
        # A heading written twice: its bodies joined.
        ("## A (#1)\n\n3.5% of rows\n\n## B (#2)\n\n1977.\n\n## A (#1)\n\n12,345 races", "", set()),
        # Held only under another section, or under a heading that is not the same text.
        ("## A (#1)\n\n3.5%, 12,345 races, from 1977.\n\n## B (#2)\n\nnone\n", "", {(b, "1977")}),
        ("## A (#1)\n\n3.5%, 12,345 races\n\n## B\n\nfrom 1977.\n", "", {(b, "1977")}),
        ("1977\n\n## A (#1)\n\n3.5%, 12,345 races\n", "", {(b, "1977")}),
        # Only a different value with the same suffix or prefix.
        (
            "## A (#1)\n\n13.5% and 8.5% of rows, 112,345 races, 12,345.6 kg\n\n"
            "## B (#2)\n\nfrom 11977 and 1977.5\n",
            "",
            {(a, "3.5%"), (a, "12,345"), (a, "12,345 races"), (b, "1977")},
        ),
        # Written as a code constant in the same section of the doc, or in another one.
        ("## A (#1)\n\n3.5%\n\n## B (#2)\n\n1977\n", "## A (#1)\n\n`12_345` races\n", set()),
        (
            "## A (#1)\n\n3.5%\n\n## B (#2)\n\n1977\n",
            "## A (#1)\n\nnone\n\n## B (#2)\n\n`12_345`\n",
            {(a, "12,345"), (a, "12,345 races")},
        ),
    ]
    wrong = [
        (archive, new, set(unarchived_figures(old, archive, new)))
        for archive, new, missing in archives
        if set(unarchived_figures(old, archive, new)) != missing
    ]
    assert not wrong, wrong


@pytest.mark.local
def test_private_reports_hold_the_withheld_figures(doc):
    try:
        reports = paths.reports_dir()
    except paths.DataDirNotSet:
        pytest.skip(f"{paths.DATA_DIR_VAR} is not set: private reports unavailable")
    if not reports.is_dir():
        pytest.skip(f"private reports directory missing: {reports}")
    probe = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{PRE_FIX}^{{commit}}"],
        capture_output=True,
    )
    if probe.returncode != 0:
        pytest.skip("the pre-fix commit is not in this clone")
    old = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{PRE_FIX}:{DOC_REL}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout

    absent = [name for name in [*SECTIONS.values(), FIGURES_FILE] if not (reports / name).is_file()]
    fail_unless(not absent, f"missing under the reports dir: {absent}")
    archive = (reports / FIGURES_FILE).read_text(encoding="utf-8")

    published = figures_by_section(old)
    fail_unless(bool(published), "no figure found in the pre-fix doc: the rules have broken")
    fail_unless(
        any(table["figures"] for table in tables(old)) and bool(spans_and_dates(old)),
        "no figure table or span found in the pre-fix doc: the rules have broken",
    )
    missing = unarchived(old, archive, doc)
    # Rule names, counts, pre-fix line numbers and headings only: the values are the licensed
    # figures, and the outcome sentences state what the run found.
    by_rule = {}
    for heading, rule, _item in missing:
        by_rule.setdefault(rule, set()).add(heading)
    counts = {rule: sum(1 for key in missing if key[1] == rule) for rule in sorted(by_rule)}
    lines = sorted({line for found in missing.values() for line in found})
    headings_missing = {rule: sorted(found) for rule, found in sorted(by_rule.items())}
    absent = sorted({key[0] for key in missing} - set(section_bodies(archive)))
    fail_unless(
        not missing,
        f"{len(missing)} items the pre-fix doc published are not in their section of "
        f"{FIGURES_FILE} (per rule {counts}: figure = a figure as a whole value, span = a span "
        f"or date with both ends, cell = a table cell under its row label and column header, "
        f"outcome = a removed sentence or row verbatim; pre-fix {DOC_REL} lines {lines}; "
        f"headings per rule {headings_missing}; headings the archive lacks: {absent})",
    )
