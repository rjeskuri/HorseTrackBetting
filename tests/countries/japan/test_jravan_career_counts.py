"""JV-Data CK (出走別着度数, each runner's placings so far, and its jockey's and trainer's) into
the jravan `career_counts` table (ticket #407).

Contract pinned here (the coder follows it), module
`horsetrack.countries.japan.jravan.career_counts`:
- `FILENAME == "career_counts.parquet"`, `NAME == "career_counts"`.
- `CAREER_COUNTS_SCHEMA: pyarrow.Schema`, in this order: `race_id` string, `race_date` date32,
  `horse_id` string, `subject` string, `subject_id` string, `period` string, `condition` string,
  `first`, `second`, `third`, `fourth`, `fifth`, `unplaced` int64.
- `build_career_counts(*, raw=None, out=None, chunk_records: int = 20_000)
  -> records.BuildReport`: reads `<raw>/stored/SNPN/CK/all.jvd` (`raw` the export root, default
  `paths.raw_dir('jravan')`; no file is a `FileNotFoundError`) with `spec.record_spec('CK')`,
  streamed `chunk_records` records at a time (`records.iter_lines` / `iter_file`, never the whole
  file at once), and writes `<out>/career_counts.parquet` (`out` a folder, default
  `paths.processed_dir('jravan')`) with `CAREER_COUNTS_SCHEMA`, through a tmp file and
  `os.replace`, without wall-clock metadata: the same records, in any order and at any
  `chunk_records`, give the same bytes. The report's name is `career_counts`; `counts['rows']` is
  the rows written and `counts['holdout_skipped']` the records skipped as on a holdout day; it
  holds counts only (D21).
- `horsetrack japan jravan career-counts`: the same build with the defaults, then
  `records.write_build_report` (`<reports dir>/jravan-career_counts.json`, through
  `write_report`, so a reports folder inside the repository is refused: exit code non-zero). A
  failure (no CK file, no data dir) is `ERROR ...` on stderr and exit code 1.

What a row is (names as `spec.record_spec` returns them, keys as `records.decode` builds them):
- A record's key is its race key (`開催年`, `開催月日`, `競馬場コード`, `開催回[第N回]`,
  `開催日目[N日目]`, `レース番号`) and `血統登録番号`. One version per key:
  `records.latest_versions` on that key (the latest `データ作成年月日` wins wherever it sits in
  the file; a key whose kept version has `データ区分` 0, a deletion, has no rows).
- Only JRA venues (`競馬場コード` 01-10): `race_id` is `keys.race_id(開催年, 開催月日,
  競馬場コード, レース番号)`, `race_date` 開催年 + 開催月日, `horse_id` the record's
  `血統登録番号`.
- No holdout day is read: a record whose race date (開催年 + 開催月日, read from its bytes alone)
  is a holdout day (`core.eval.holdout.is_holdout_day('japan', day)`) is read no further, whatever
  its width, and only counted in `holdout_skipped`. The same cut-off record on a development day is
  a `records.RecordLengthError`.
- A placing-counts item is a leaf whose name ends in `着回数` with six repeats: 1st, 2nd, 3rd,
  4th, 5th and out of the first five (`first` ... `fifth`, `unplaced`), each `records.integer`,
  null when not a number (blank). Which items exist, their widths and their order are read from
  the layout, never listed by hand: the two layouts below differ in all three.
- `condition` is the item's name without the trailing `着回数` and a trailing `・`
  (`総合着回数` -> `総合`, `芝直・着回数` -> `芝直`, `札幌芝・着回数` -> `札幌芝`).
- `subject`:
  - `horse`: a top-level placing-counts item; `subject_id` the `血統登録番号`, `period` `career`.
  - `jockey`: a placing-counts sub-item of the group whose name contains `騎手`; `subject_id` the
    top-level `騎手コード` (text); the group's two occurrences are `period` `this_year` (the
    first, 本年) and `career` (the second, 累計).
  - `trainer`: the same for the group whose name contains `調教師`, `subject_id` `調教師コード`.
  - Placing counts of any other group (the owner's `馬主...`, the breeder's `生産者...`) write no
    row.
- Rows are sorted by `race_id, horse_id, subject, period, condition` (plain string order).

Every layout here is made up in the workbook's shape (`jravan_workbook.write_workbook`): the item
names follow the workbook's naming, but every width, repeat, order and so every position is
invented (D21: the real workbook and export are licensed and stay under `$HORSETRACK_DATA_DIR`),
so nothing may be sliced by hand. Every record and count is invented; the counts are drawn from a
seeded generator per record and the expected rows are built from that plan, not from the module.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import env, paths
from horsetrack.core.eval.holdout import is_holdout_day

# isort: split
# (its own block: the sibling helper module, then the module this ticket creates)
import jravan_workbook as wb

from horsetrack.countries.japan.jravan import career_counts, records, spec

SOURCE = "jravan"
FILENAME = "career_counts.parquet"
REPORT = "jravan-career_counts.json"
EXPECTED_SCHEMA = [
    ("race_id", pa.string()),
    ("race_date", pa.date32()),
    ("horse_id", pa.string()),
    ("subject", pa.string()),
    ("subject_id", pa.string()),
    ("period", pa.string()),
    ("condition", pa.string()),
    ("first", pa.int64()),
    ("second", pa.int64()),
    ("third", pa.int64()),
    ("fourth", pa.int64()),
    ("fifth", pa.int64()),
    ("unplaced", pa.int64()),
]
COLUMNS = [name for name, _ in EXPECTED_SCHEMA]
VENUES = {  # number -> short code, as config/jra.yaml lists them
    "01": "SAP", "02": "HAK", "03": "FKS", "04": "NII", "05": "TOK",
    "06": "NAK", "07": "CKY", "08": "KYO", "09": "HSN", "10": "KOK",
}  # fmt: skip
TOK, NAK, KYO, KOK = "05", "06", "08", "10"
NAR_VENUE, OVERSEAS_VENUE = "44", "A6"
PLACES = 6  # 1st-5th and out
SECRETS = ("カクウノウマ", "架空騎手", "架空調教師", "架空馬主", "架空生産者")
RACE_ID = re.compile(r"jra:\d{4}-\d{2}-\d{2}")
LONG_DIGITS = re.compile(r"\d{10}")  # a horse id (10 digits)

# ------------------------------------------------------------------------------------------------
# Made-up CK layouts: the workbook's naming, invented widths, repeats and order


@dataclass(frozen=True)
class Counts:
    """A placing-counts item: its name, the bytes of one count, the condition it maps to (None:
    the item is not a placing count, e.g. a running-style tally)."""

    name: str
    width: int
    condition: str | None
    repeat: int = PLACES


@dataclass(frozen=True)
class Block:
    """A 本年・累計 group (two occurrences) after its subject's code and name items."""

    name: str
    subject: str | None  # the expected subject; None: no rows
    code_item: str
    code_width: int
    name_item: str
    pre: tuple[tuple[str, int], ...]  # plain sub-items before the counts (設定年, 賞金)
    counts: tuple[Counts, ...]


@dataclass(frozen=True)
class CKLayout:
    label: str
    horse_name_width: int
    horse: tuple[Counts, ...]
    blocks: tuple[Block, ...]


def _person_block(
    name: str, subject: str | None, code: str, code_width: int, person: str, counts
) -> Block:
    return Block(name, subject, code, code_width, person, (("設定年", 4), ("平地本賞金合計", 10)),
                 counts)  # fmt: skip


LAYOUT_A = CKLayout(
    label="A",
    horse_name_width=18,
    horse=(
        Counts("総合着回数", 3, "総合"),
        Counts("中央合計着回数", 3, "中央合計"),
        Counts("芝直・着回数", 3, "芝直"),
        Counts("ダ不・着回数", 3, "ダ不"),
        Counts("芝16下・着回数", 3, "芝16下"),
    ),
    blocks=(
        _person_block("騎手本年・累計成績情報", "jockey", "騎手コード", 5, "騎手名", (
            Counts("芝着回数", 5, "芝"),
            Counts("ダート着回数", 5, "ダート"),
            Counts("札幌芝・着回数", 4, "札幌芝"),
        )),
        _person_block("調教師本年・累計成績情報", "trainer", "調教師コード", 5, "調教師名", (
            Counts("芝着回数", 5, "芝"),
            Counts("ダート着回数", 5, "ダート"),
            Counts("札幌芝・着回数", 4, "札幌芝"),
        )),
        Block("馬主本年・累計成績情報", None, "馬主コード", 6, "馬主名(法人格無)",
              (("設定年", 4), ("本賞金合計", 10)), (Counts("着回数", 4, None),)),
    ),
)  # fmt: skip

LAYOUT_B = CKLayout(
    label="B",
    horse_name_width=24,
    horse=(
        Counts("総合着回数", 2, "総合"),
        Counts("障害・着回数", 2, "障害"),
        Counts("芝良・着回数", 3, "芝良"),
        Counts("ダ22超・着回数", 2, "ダ22超"),
    ),
    blocks=(
        Block("騎手本年・累計成績情報", "jockey", "騎手コード", 5, "騎手名", (("設定年", 4),), (
            Counts("小倉ダ・着回数", 6, "小倉ダ"),
            Counts("障害着回数", 4, "障害"),
            Counts("芝22下・着回数", 3, "芝22下"),
        )),
        _person_block("調教師本年・累計成績情報", "trainer", "調教師コード", 5, "調教師名", (
            Counts("障害着回数", 3, "障害"),
            Counts("小倉ダ・着回数", 6, "小倉ダ"),
        )),
        Block("馬主本年・累計成績情報", None, "馬主コード", 6, "馬主名(法人格無)",
              (("設定年", 4), ("本賞金合計", 10)), (Counts("着回数", 5, None),)),
        Block("生産者本年・累計成績情報", None, "生産者コード", 8, "生産者名(法人格無)",
              (("設定年", 4),), (Counts("着回数", 4, None),)),
    ),
)  # fmt: skip

LAYOUTS = {"A": LAYOUT_A, "B": LAYOUT_B}

RACE_KEY = (
    ("データ区分", 1),
    ("データ作成年月日", 8),
    ("開催年", 4),
    ("開催月日", 4),
    ("競馬場コード", 2),
    ("開催回[第N回]", 2),
    ("開催日目[N日目]", 2),
    ("レース番号", 2),
    ("血統登録番号", 10),
)
KEY_ITEMS = {"開催年", "開催月日", "競馬場コード", "開催回[第N回]", "開催日目[N日目]",
             "レース番号", "血統登録番号"}  # fmt: skip
STYLE = Counts("脚質傾向", 3, None, repeat=4)  # a repeated tally that is not a placing count


def section(layout: CKLayout) -> wb.Section:
    """The CK section of `layout` on the フォーマット sheet."""
    rows: list[wb.Row] = [wb.item(1, "レコード種別ID", 1, 2, key="●", initial="CK",
                                  desc=wb.id_desc("CK"))]  # fmt: skip
    pos, no = 3, 2

    def plain(name: str, length: int, repeat: int | None = None) -> None:
        nonlocal pos, no
        key = "○" if name in KEY_ITEMS else ""
        rows.append(wb.item(no, name, pos, length, repeat=repeat, key=key, desc="（架空）"))
        pos += length * (repeat or 1)
        no += 1

    for name, length in RACE_KEY:
        plain(name, length)
    plain("馬名", layout.horse_name_width)
    plain("平地本賞金累計", 9)
    for c in layout.horse:
        plain(c.name, c.width, c.repeat)
    plain(STYLE.name, STYLE.width, STYLE.repeat)
    plain("登録レース数", 3)
    for block in layout.blocks:
        plain(block.code_item, block.code_width)
        plain(block.name_item, 12)
        subs = [(n, length, None) for n, length in block.pre]
        subs += [(c.name, c.width, c.repeat) for c in block.counts]
        width = sum(length * (r or 1) for _, length, r in subs)
        rows.append(wb.group(no, f"<{block.name}>", pos, width, 2, desc="成績（架空）"))
        rel = 1
        for i, (name, length, r) in enumerate(subs):
            rows.append(wb.sub(wb.letter_code(i), name, rel, length, repeat=r, desc="値（架空）"))
            rel += length * (r or 1)
        pos += width * 2
        no += 1
    rows.append(wb.item(no, "レコード区切", pos, 2, desc="（架空）"))
    return wb.Section(23, "出走別着度数（架空）", pos + 1, tuple(rows))


def write_ck_workbook(path: Path, layout: CKLayout) -> Path:
    decoy = wb.simple_section(1, "XQ", "架空その他", [("架空項目", 6)])
    return wb.write_workbook(path, [decoy, section(layout)], [])


# ------------------------------------------------------------------------------------------------
# Made-up CK records and the rows they must give

Places = tuple[int, ...] | None  # six counts, or None for a blank item


@dataclass(frozen=True)
class Runner:
    """One made-up CK record: a runner in a race, its subjects' codes and a seed for its counts.
    `blank` names (subject item key, occurrence) pairs written blank."""

    day: date
    venue: str
    race_no: int
    horse_id: str
    jockey: str = "01001"
    trainer: str = "01101"
    kubun: str = "1"
    created: date | None = None
    seed: int = 0
    blank: frozenset[tuple[str, int]] = field(default_factory=frozenset)

    @property
    def made(self) -> date:
        return self.created or self.day - timedelta(days=1)

    @property
    def race_id(self) -> str:
        return f"jra:{self.day.isoformat()}:{VENUES[self.venue]}:{self.race_no:02d}"


def code_of(runner: Runner, block: Block) -> str:
    return {"jockey": runner.jockey, "trainer": runner.trainer}.get(block.subject or "", "9" * 4)


def counts_of(layout: CKLayout, runner: Runner) -> dict[tuple[str, int], Places]:
    """The runner's counts per (item key, occurrence): a top-level item's key is its name, a
    group's sub-item's `group/name`; each count drawn within its width from the runner's seed."""
    rng = random.Random(f"{layout.label}:{runner.seed}")
    out: dict[tuple[str, int], Places] = {}

    def draw(c: Counts) -> tuple[int, ...]:
        top = 10**c.width - 1
        return tuple(rng.choice((0, 1, top, rng.randint(0, top))) for _ in range(c.repeat))

    for c in (*layout.horse, STYLE):
        out[(c.name, 0)] = draw(c)
    for block in layout.blocks:
        for occurrence in range(2):
            for c in block.counts:
                out[(f"{block.name}/{c.name}", occurrence)] = draw(c)
    for blank in runner.blank:
        assert blank in out, blank
        out[blank] = None
    return out


def fit(text: str, width: int) -> bytes:
    raw = text.encode("cp932")
    assert len(raw) <= width, (text, width)
    return raw + b" " * (width - len(raw))


def number(value: int, width: int) -> bytes:
    text = f"{value:0{width}d}"
    assert len(text) == width, (value, width)
    return text.encode("ascii")


def places(values: Places, c: Counts) -> bytes:
    if values is None:
        return b" " * (c.width * c.repeat)
    assert len(values) == c.repeat
    return b"".join(number(v, c.width) for v in values)


def ck(layout: CKLayout, runner: Runner) -> bytes:
    """`runner` as one CK record of `layout`, CRLF included."""
    counts = counts_of(layout, runner)
    key = {
        "データ区分": runner.kubun,
        "データ作成年月日": runner.made.strftime("%Y%m%d"),
        "開催年": f"{runner.day.year:04d}",
        "開催月日": runner.day.strftime("%m%d"),
        "競馬場コード": runner.venue,
        "開催回[第N回]": "03",
        "開催日目[N日目]": "02",
        "レース番号": f"{runner.race_no:02d}",
        "血統登録番号": runner.horse_id,
    }
    out = [b"CK"] + [fit(key[name], length) for name, length in RACE_KEY]
    out += [fit("カクウノウマ", layout.horse_name_width), number(123, 9)]
    out += [places(counts[(c.name, 0)], c) for c in (*layout.horse, STYLE)]
    out.append(number(runner.race_no + 7, 3))
    for block in layout.blocks:
        out.append(fit(code_of(runner, block), block.code_width))
        out.append(fit({"jockey": "架空騎手", "trainer": "架空調教師"}.get(
            block.subject or "", "架空馬主"), 12))  # fmt: skip
        for occurrence in range(2):
            for name, length in block.pre:
                out.append(number(runner.day.year if name == "設定年" else 4567, length))
            for c in block.counts:
                out.append(places(counts[(f"{block.name}/{c.name}", occurrence)], c))
    body = b"".join(out)
    assert len(body) == section(layout).length - 2, (len(body), section(layout).length)
    return body + b"\r\n"


Row = tuple  # (race_id, race_date, horse_id, subject, subject_id, period, condition, *6 counts)


def expected_rows(layout: CKLayout, runners: Iterable[Runner]) -> list[Row]:
    """The rows `runners` must give under `layout`, sorted by the table's sort key."""
    out: list[Row] = []
    for r in runners:
        counts = counts_of(layout, r)
        for c in layout.horse:
            values = counts[(c.name, 0)] or (None,) * PLACES
            out.append((r.race_id, r.day, r.horse_id, "horse", r.horse_id, "career", c.condition,
                        *values))  # fmt: skip
        for block in layout.blocks:
            if block.subject is None:
                continue
            for occurrence, period in enumerate(("this_year", "career")):
                for c in block.counts:
                    values = counts[(f"{block.name}/{c.name}", occurrence)] or (None,) * PLACES
                    out.append((r.race_id, r.day, r.horse_id, block.subject, code_of(r, block),
                                period, c.condition, *values))  # fmt: skip
    return sorted(out, key=lambda row: row[:7])


def table_rows(table: pa.Table) -> list[Row]:
    assert [(f.name, f.type) for f in table.schema] == EXPECTED_SCHEMA
    return [tuple(r[c] for c in COLUMNS) for r in table.to_pylist()]


# ------------------------------------------------------------------------------------------------
# Fixtures


@pytest.fixture(params=sorted(LAYOUTS))
def layout(request: pytest.FixtureRequest) -> CKLayout:
    return LAYOUTS[request.param]


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp data dir (never the real one)."""
    d = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


def install(data: Path, layout: CKLayout) -> Path:
    """`layout`'s workbook in the data dir's spec folder; returns the CK file's path."""
    write_ck_workbook(data / "specs" / "jravan" / wb.WORKBOOK_NAME, layout)
    return data / "raw" / SOURCE / "stored" / "SNPN" / "CK" / "all.jvd"


def put(path: Path, layout: CKLayout, runners: Sequence[Runner]) -> Path:
    return wb.write_jvd(path, [ck(layout, r) for r in runners])


def built(out: Path) -> pa.Table:
    return pq.read_table(out / FILENAME)


# ------------------------------------------------------------------------------------------------
# The main synthetic export

D1, D2, D3 = date(2025, 5, 25), date(2024, 12, 28), date(2025, 6, 1)
KEPT = (
    Runner(D1, TOK, 11, "2021100001", seed=1),
    Runner(D1, TOK, 11, "2021100002", jockey="01002", trainer="01102", seed=8,
           created=D1 + timedelta(days=1)),  # the later of its two versions
    Runner(D3, KYO, 1, "2021100001", jockey="01003", seed=3),  # the same horse, a later race
    Runner(D2, NAK, 11, "2020100005", jockey="00999", trainer="01105", seed=5,
           blank=frozenset({("調教師本年・累計成績情報/障害着回数", 1),
                            ("調教師本年・累計成績情報/札幌芝・着回数", 1),
                            ("騎手本年・累計成績情報/芝着回数", 0),
                            ("中央合計着回数", 0), ("総合着回数", 0)})),
    Runner(D1, KOK, 12, "2022100009", seed=9),
)  # fmt: skip
DROPPED = (
    # an earlier version of the second runner, written after the later one
    Runner(D1, TOK, 11, "2021100002", jockey="01002", trainer="01102", seed=7,
           created=D1 - timedelta(days=3)),
    # a runner whose latest version is a deletion
    Runner(D1, TOK, 11, "2021100004", seed=4, created=D1 - timedelta(days=2)),
    Runner(D1, TOK, 11, "2021100004", seed=40, kubun="0", created=D1 + timedelta(days=2)),
    # not JRA venues
    Runner(D1, NAR_VENUE, 5, "2021100006", seed=6),
    Runner(D1, OVERSEAS_VENUE, 11, "2021100007", seed=10),
)  # fmt: skip


def runners_for(layout: CKLayout) -> list[Runner]:
    """The main export in file order; blanks the layout lacks are left out."""
    names = {(c.name, 0) for c in layout.horse}
    names |= {(f"{b.name}/{c.name}", o) for b in layout.blocks for c in b.counts for o in (0, 1)}
    order = (KEPT[0], DROPPED[1], KEPT[1], DROPPED[0], KEPT[2], DROPPED[3], DROPPED[2], KEPT[3],
             KEPT[4], DROPPED[4])  # fmt: skip
    return [replace(r, blank=frozenset(b for b in r.blank if b in names)) for r in order]


def kept_for(layout: CKLayout) -> list[Runner]:
    keep = {(r.race_id, r.horse_id, r.seed) for r in KEPT}
    return [r for r in runners_for(layout) if r.venue in VENUES and
            (r.race_id, r.horse_id, r.seed) in keep]  # fmt: skip


# ------------------------------------------------------------------------------------------------
# The table: one row per race, horse, subject, period and condition


def test_ck_records_give_one_row_per_subject_period_and_condition(
    data: Path, tmp_path: Path, layout: CKLayout
) -> None:
    path = install(data, layout)
    put(path, layout, runners_for(layout))

    report = career_counts.build_career_counts(out=tmp_path / "out")
    table = built(tmp_path / "out")
    assert table.schema.equals(career_counts.CAREER_COUNTS_SCHEMA)
    got = table_rows(table)
    want = expected_rows(layout, kept_for(layout))
    assert len(got) == len(want)
    assert got == want  # every value, and the sort order
    assert report.name == "career_counts"
    assert report.counts["rows"] == len(want)
    assert report.counts["holdout_skipped"] == 0


def test_the_rows_cover_every_count_item_of_the_layout_and_no_other(
    data: Path, tmp_path: Path, layout: CKLayout
) -> None:
    """The conditions per subject are the layout's placing-counts items, renamed; the owner's and
    breeder's counts and the running-style tally give no row."""
    path = install(data, layout)
    put(path, layout, [KEPT[0]])
    career_counts.build_career_counts(out=tmp_path)
    by_subject: dict[tuple[str, str], list[str]] = {}
    for row in table_rows(built(tmp_path)):
        by_subject.setdefault((row[3], row[5]), []).append(row[6])
    want = {("horse", "career"): sorted(c.condition for c in layout.horse)}
    for block in layout.blocks:
        if block.subject:
            for period in ("this_year", "career"):
                want[(block.subject, period)] = sorted(c.condition for c in block.counts)
    assert by_subject == want


def test_one_version_per_runner_the_latest_kept_and_deletions_dropped(
    data: Path, tmp_path: Path
) -> None:
    layout = LAYOUT_A
    path = install(data, layout)
    r = KEPT[0]
    newest = Runner(r.day, r.venue, r.race_no, r.horse_id, seed=21, created=r.day + timedelta(3))
    older = [Runner(r.day, r.venue, r.race_no, r.horse_id, seed=s, created=r.day - timedelta(s))
             for s in (1, 2, 5)]  # fmt: skip
    gone = Runner(r.day, r.venue, r.race_no, "2021100004", seed=4)
    deleted = Runner(r.day, r.venue, r.race_no, "2021100004", seed=44, kubun="0",
                     created=r.day + timedelta(days=4))  # fmt: skip
    put(path, layout, [older[0], newest, older[2], deleted, gone, older[1]])

    career_counts.build_career_counts(out=tmp_path)
    assert table_rows(built(tmp_path)) == expected_rows(layout, [newest])


def test_only_jra_venues_are_kept(data: Path, tmp_path: Path) -> None:
    layout = LAYOUT_B
    path = install(data, layout)
    jra = [Runner(D1, v, 3, f"20211000{int(v):02d}", seed=int(v)) for v in VENUES]
    other = [Runner(D1, v, 3, "2021100099", seed=99) for v in (NAR_VENUE, "30", OVERSEAS_VENUE)]
    put(path, layout, [*other[:1], *jra[:5], *other[1:], *jra[5:]])

    career_counts.build_career_counts(out=tmp_path)
    got = table_rows(built(tmp_path))
    assert {row[0] for row in got} == {r.race_id for r in jra}
    assert got == expected_rows(layout, jra)


# ------------------------------------------------------------------------------------------------
# No holdout day is read


def first_day(holdout: bool) -> date:
    """The first day from 2026-11-28 (the holdout's start) that is, or is not, a holdout day."""
    day = date(2026, 11, 28)
    while is_holdout_day("japan", day) is not holdout:
        day += timedelta(days=1)
    return day


def test_a_record_on_a_holdout_day_writes_no_row(data: Path, tmp_path: Path) -> None:
    layout = LAYOUT_A
    path = install(data, layout)
    held, dev, early = first_day(True), first_day(False), date(2026, 11, 22)
    assert is_holdout_day("japan", held) and not is_holdout_day("japan", dev)
    assert not is_holdout_day("japan", early)
    kept = [Runner(dev, TOK, 11, "2023100001", seed=31), Runner(early, KYO, 2, "2023100002",
                                                                 seed=32)]  # fmt: skip
    skipped = [Runner(held, TOK, 11, "2023100001", seed=33),
               Runner(held, KYO, 5, "2023100003", seed=34),
               Runner(held, TOK, 12, "2023100004", seed=35, kubun="0")]  # fmt: skip
    put(path, layout, [skipped[0], kept[0], skipped[1], kept[1], skipped[2]])

    report = career_counts.build_career_counts(out=tmp_path)
    got = table_rows(built(tmp_path))
    assert held not in {row[1] for row in got}
    assert got == expected_rows(layout, kept)
    assert report.counts["holdout_skipped"] == len(skipped)
    assert report.counts["rows"] == len(got)


def race_date_end(layout_spec: spec.RecordSpec) -> int:
    """The bytes from the record's start through 開催年 and 開催月日."""
    by_name = {item.name: item for item in layout_spec.items}
    return max(by_name[n].start - 1 + by_name[n].length for n in ("開催年", "開催月日"))


@pytest.mark.parametrize("how", ("after-the-date", "one-byte-short", "one-byte-long"))
def test_a_holdout_day_record_is_read_no_further_than_its_date(
    data: Path, tmp_path: Path, how: str
) -> None:
    layout = LAYOUT_B
    path = install(data, layout)
    ck_spec = spec.record_spec("CK")
    width = ck_spec.length - spec.SEPARATOR_BYTES

    def cut(rec: bytes) -> bytes:
        body = rec.removesuffix(b"\r\n")
        assert len(body) == width
        if how == "one-byte-long":
            return body + b"0\r\n"
        keep = race_date_end(ck_spec) if how == "after-the-date" else width - 1
        return body[:keep] + b"\r\n"

    held, dev = first_day(True), first_day(False)
    good = Runner(dev, TOK, 11, "2023100001", seed=41)
    wb.write_jvd(path, [ck(layout, good), cut(ck(layout, Runner(held, TOK, 11, "2023100002",
                                                                  seed=42)))])  # fmt: skip
    report = career_counts.build_career_counts(out=tmp_path, chunk_records=1)
    assert table_rows(built(tmp_path)) == expected_rows(layout, [good])
    assert report.counts["holdout_skipped"] == 1

    # the same cut on a development day is a record of the wrong width
    bad = cut(ck(layout, Runner(dev, KYO, 3, "2023100003", seed=43)))
    wb.write_jvd(path, [ck(layout, good), bad])
    with pytest.raises(records.RecordLengthError) as caught:
        career_counts.build_career_counts(out=tmp_path / "again")
    assert caught.value.record == 2


# ------------------------------------------------------------------------------------------------
# Sorted, byte-stable, streamed


def test_the_output_is_byte_stable_across_builds_input_order_and_chunk_sizes(
    data: Path, tmp_path: Path, layout: CKLayout
) -> None:
    path = install(data, layout)
    runners = runners_for(layout)
    put(path, layout, runners)
    career_counts.build_career_counts(out=tmp_path / "first")
    first = (tmp_path / "first" / FILENAME).read_bytes()

    career_counts.build_career_counts(out=tmp_path / "second")
    assert (tmp_path / "second" / FILENAME).read_bytes() == first

    for seed in (1, 2, 3):  # the same records in another order (no two versions share a date)
        shuffled = list(runners)
        random.Random(seed).shuffle(shuffled)
        put(path, layout, shuffled)
        career_counts.build_career_counts(out=tmp_path / f"shuffled{seed}")
        assert (tmp_path / f"shuffled{seed}" / FILENAME).read_bytes() == first, seed

    put(path, layout, runners)
    for chunk in (1, 2, 3, 7):  # a key's versions in different chunks
        report = career_counts.build_career_counts(out=tmp_path / f"c{chunk}", chunk_records=chunk)
        assert (tmp_path / f"c{chunk}" / FILENAME).read_bytes() == first, chunk
        assert report.counts["rows"] == len(expected_rows(layout, kept_for(layout)))
    assert not [p.name for p in tmp_path.rglob("*.tmp")], "a tmp file left behind"


def test_many_records_in_small_chunks_equal_one_read(data: Path, tmp_path: Path) -> None:
    """A file of a few hundred records over three years, read 13 records at a time, writes the
    rows of one read, sorted."""
    layout = LAYOUT_A
    path = install(data, layout)
    rng = random.Random(407)
    days = [date(2023, 1, 5) + timedelta(days=rng.randint(0, 1000)) for _ in range(40)]
    runners = []
    for i in range(300):
        day, venue, race_no = rng.choice(days), rng.choice(list(VENUES)), rng.randint(1, 12)
        horse_id = f"20{rng.randint(18, 22)}10{i:04d}"
        jockey, trainer = f"0{rng.randint(1000, 1099)}", f"0{rng.randint(1100, 1199)}"
        runners.append(
            Runner(day, venue, race_no, horse_id, jockey=jockey, trainer=trainer, seed=1000 + i)
        )
    put(path, layout, runners)

    career_counts.build_career_counts(out=tmp_path / "small", chunk_records=13)
    career_counts.build_career_counts(out=tmp_path / "whole", chunk_records=100_000)
    small = (tmp_path / "small" / FILENAME).read_bytes()
    assert small == (tmp_path / "whole" / FILENAME).read_bytes()
    assert table_rows(built(tmp_path / "small")) == expected_rows(layout, runners)


# ------------------------------------------------------------------------------------------------
# The command and the private report (D21)


def invoke(*args: str):
    return CliRunner().invoke(app, ["japan", "jravan", "career-counts", *args])


def assert_counts_only(text: str) -> None:
    assert not RACE_ID.search(text), "a race id in the report"
    assert not LONG_DIGITS.search(text), "a horse id in the report"
    for secret in SECRETS:
        assert secret not in text, secret


def test_the_command_writes_the_table_and_a_counts_only_report(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = LAYOUT_B
    path = install(data, layout)
    put(path, layout, runners_for(layout))
    career_counts.build_career_counts(out=tmp_path / "direct")

    result = invoke()
    assert result.exit_code == 0, result.output
    table_path = data / "processed" / SOURCE / FILENAME
    assert table_path.read_bytes() == (tmp_path / "direct" / FILENAME).read_bytes()
    want = expected_rows(layout, kept_for(layout))
    assert table_rows(pq.read_table(table_path)) == want

    report_path = data / "reports" / REPORT
    text = report_path.read_text(encoding="utf-8")
    parsed = json.loads(text)
    assert parsed["name"] == "career_counts"
    assert parsed["counts"]["rows"] == len(want)
    assert_counts_only(text)
    assert_counts_only(result.output)

    # a reports folder inside the repository is refused, and nothing is created there
    inside = env.repo_root() / f".tmp-jravan-career-counts-{uuid.uuid4().hex[:8]}" / "reports"
    monkeypatch.setattr(paths, "reports_dir", lambda: inside)
    try:
        refused = invoke()
        assert refused.exit_code != 0, refused.output
        assert not inside.parent.exists()
    finally:
        shutil.rmtree(inside.parent, ignore_errors=True)


def test_the_command_fails_without_a_ck_file(data: Path) -> None:
    install(data, LAYOUT_A)
    result = invoke()
    assert result.exit_code == 1, result.output
    assert "ERROR" in result.output
    assert not (data / "processed" / SOURCE / FILENAME).exists()
