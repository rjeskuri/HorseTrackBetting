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
  `jravan-spec-figures.md` exist, and the latter holds every figure the rules above flag in the
  pre-fix doc's measured-run passages (from the block starting "The run on the export" or "The
  build on the export" to the section's end).

D21: this file holds no figure from the export. Figures are found by pattern, never by value;
every example below is made up. Failure messages give line numbers and rule names only, so a
failing run does not reprint a figure.
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
# removed passages from it. A public commit id, not a figure.
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


def hits(text: str, finder: Finder) -> list[tuple[int, str]]:
    """(line number, matched text) of every figure the finder flags in the doc's blocks."""
    out = []
    for first_line, block in blocks(text):
        for start, end in finder(block):
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
    assert len(starts) == 1, f"{len(starts)} `## ` headings start with {prefix!r}"
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


def measured_passage(text: str, prefix: str) -> str:
    body = section(text, prefix)
    starts = [
        m.start()
        for m in re.finditer(r"(?m)^The (?:run|build) on the export", body)
        if m.start() == 0 or body[m.start() - 2 : m.start()] == "\n\n"
    ]
    assert starts, f"no measured-run passage in {prefix!r}"
    return body[starts[0] :]


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text(encoding="utf-8")


def assert_sections_kept(text: str) -> None:
    missing = [h for h in HEADINGS if h not in headings(text)]
    assert not missing, f"headings lost: {missing}"


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
    assert not lines, f"{DOC_REL}: {rule} figures on lines {lines} (D21: withhold them)"


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
        assert not lost, f"{prefix}: table rows lost: {lost}"
    rows = figure_rows(doc)
    assert not rows, f"{DOC_REL}: table rows with figures on lines {rows} (D21: withhold them)"


@pytest.mark.parametrize("prefix", list(SECTIONS))
def test_each_measured_section_names_its_private_reports(doc, prefix):
    report = SECTIONS[prefix]
    body = section(doc, prefix)
    kept = [name for name in KEPT_NAMES[prefix] if name not in body]
    assert not kept, f"{prefix}: names lost: {kept}"
    if prefix == "## The 0B41/0B42 time series":
        start = load_country_yaml("japan", "holdout")["holdout"]["start"]
        assert str(start) in body, f"{prefix}: the holdout start is no longer stated"

    needed = (WITHHELD, report, FIGURES_FILE)
    holding = [b for _line, b in blocks(body) if all(token in b for token in needed)]
    assert holding, f"{prefix}: no paragraph holds all of {needed}"


# ------------------------------------------------------------------------------------------------
# Local: the private reports hold what the doc withholds.

WHITESPACE = re.compile(r"\s+")


def normalized(text: str) -> str:
    return WHITESPACE.sub(" ", text.replace("`", ""))


@pytest.mark.local
def test_private_reports_hold_the_withheld_figures():
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
    assert not absent, f"missing under the reports dir: {absent}"
    figures = normalized((reports / FIGURES_FILE).read_text(encoding="utf-8"))

    for prefix in SECTIONS:
        passage = measured_passage(old, prefix)
        wanted = {
            normalized(t) for finder in PROSE_RULES.values() for _l, t in hits(passage, finder)
        }
        for _number, _raw, masked in table_data_rows(passage):
            for cell in cells(masked):
                if FIGURE_CELL.match(cell):
                    wanted |= set(re.findall(NUMBER, cell))
        assert wanted, f"{prefix}: no figure found in the pre-fix passage"
        missing = [t for t in wanted if t.strip() not in figures]
        # Counts only: the values are the licensed figures themselves.
        assert not missing, f"{prefix}: {len(missing)} withheld figures not in {FIGURES_FILE}"
