"""The Japan leaderboard from the registry and the candidate-cell files (ticket #452).

``build_rows`` joins #451's ``legacy_rows`` (every family and variant of the Japan registry) with
``candidate_rows``, the cells the studies write in #430's schema under
``$HORSETRACK_DATA_DIR/reports/candidate-cells/``; a registry row wins over every candidate cell of
its option id, as a base or as a named id (below). ``horsetrack japan leaderboard`` writes every
row through ``core.eval.leaderboard``: ``leaderboard-japan.md`` and ``leaderboard-japan.json``
under the reports dir and, with ``--public``, ``docs/leaderboard/japan.md`` with the withheld rows
listed by id only (D21). Its stdout holds the four counts and nothing else; its diagnostics go to
stderr; neither holds a figure.

Reading the candidate files keeps every cell on the record:

- Files: every ``*.json`` (one JSON array) and ``*.jsonl`` (one cell per line) of the directory,
  in name order. A file, line or entry that does not parse or is not a JSON object is skipped and
  counted, with the reason (no value) printed by the command. Of a data-fingerprinted series
  (``<series>-data-<hex>``, one file per run and never overwritten: #562's cross-pool v2), only
  the newest file is read, or those named with ``--cells-file``; the others are superseded and
  named on stderr.
- A cell's base is '<family_id>/<cell_id>'. One without a 'cell_id' takes its 'option' less a
  leading '<family_id>:' (#431's exacta-ceiling export names its cells so); one without a
  family id, or without both, is skipped and counted.
- An exact copy of a cell already read (a superseded copy of a file) gives no second row and is
  counted.
- Registry precedence comes first: every distinct cell whose base is a registry row's option id
  gives way to that row, whatever its pool or content (two ``debut-members`` runs exporting a
  registered variant give no extra row). Only the remaining cells are named, and a remaining cell
  whose option id is a registry id also gives way; both count as given by the registry.
- Option ids (D21) are a function of each cell's own identity, never of file or entry order nor
  of another cell's figures, so the public withheld list cannot reveal how a study ranked its
  cells. All cells are read first, then named: a base only one remaining cell has is its option
  id; every cell of a base that recurs (the pool scan's cell ids name the segment, not the pool)
  is '<base>[pool:<pool>]'. A cell its pool does not tell apart (another cell of the base has the
  same pool, it names none, or another cell's id is spelled the same) takes
  '[sha256:<first 10 hex>]' of its canonical JSON after that (the full digest should 10 hex
  digits collide).
- Sources fail closed (D21): 'sources' missing, null, empty or not a list of non-blank names reads
  as ('jravan',), so the row is withheld.
- Selection: the cell's 'selection' when it has the key; without it, #503's spread-scan 'variant'
  't2' reads as 'decision_time' and 'lookahead' as 'look_ahead'; anything else is 'unknown'.
- CI: finite 'ci_lo' and 'ci_hi' (per unit staked, as #503 writes its 5% and 95% bootstrap
  quantiles) give the row 100 x them at ``CELL_CI_LEVEL``, each bound one-sided (D7,
  ``docs/specs/leaderboard.md``); otherwise, or with ci_lo above ci_hi, no CI.
- A field that is missing or of the wrong type reads as None and the study as the file's stem; a
  cell without a finite 'mean_return' is an unmeasured row. Keys of #430's schema a cell lacks
  are named in the row's ``note``. Status is 'candidate': no run.

Capacity (#491): every row goes through ``with_capacity`` with #432's depth tables under the
reports dir (``pool_depth.read_depth_tables``). A candidate row names its depth cell by its
'pool', 'class_group' and 'odds_band'; a registry row of the pool scan (#429/#450, families
``japan-n1-pool-scan[-v<N>]-cells-<pool>``, variant the cell id) names it by its family's pool
suffix and a cell id of exactly 'class:<group>&band:<band>' (any other cell id names no depth
cell), its own fields unchanged. A row in a cell with a capacity carries it at every edge
and quantile of the table, and every other row carries a ``capacity_note`` saying why it has
none (``NO_TABLE``, ``NO_CELL_NAMED``, ``CO_PLACER``, ``NO_CELL``, ``NO_FIGURE``). The figures
are JRA-VAN: the private texts show them, the public one shows ``withheld (D21)``.

The registry is read and its hash chain verified once per build (``_ReadOnce``): #451's
``rows_from_registry`` asks each family's size, and every ``Registry.family_size`` would read the
whole file again.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, NoReturn

import pandas as pd
import typer

from horsetrack.core import env, paths
from horsetrack.core.eval.leaderboard import (
    SELECTIONS,
    LeaderboardRow,
    to_json,
    write_leaderboard,
)
from horsetrack.core.eval.registry import Registry, RunRecord, default_registry
from horsetrack.core.eval.reports import LicensedOutputInRepo, is_inside, write_report
from horsetrack.countries.japan.research import pool_depth
from horsetrack.countries.japan.research.leaderboard_backfill import legacy_rows

TITLE = "Japan leaderboard"
CANDIDATE_DIR = "candidate-cells"
PRIVATE_MD = "leaderboard-japan.md"
PRIVATE_JSON = "leaderboard-japan.json"
PUBLIC_DOC = ("docs", "leaderboard", "japan.md")
STATUS = "candidate"
UNSOURCED = ("jravan",)  # a cell that names no source is taken as licensed (D21, fail closed)
# #503's spread-scan 'variant' as a selection, for a cell without a 'selection' key.
VARIANT_SELECTIONS = {"t2": "decision_time", "lookahead": "look_ahead"}
# The level of each one-sided bound a cell's ci_lo and ci_hi carry: #503 writes its bootstrap's
# (1 - level) and level quantiles at scoring.yaml's level, D7's 95% one-sided.
CELL_CI_LEVEL = 0.95
DIGEST_HEX = 10  # the hex digits of a cell's content suffix
# A file of a data-fingerprinted series (#562's cross-pool v2: one file per run, never
# overwritten): only the series' newest, or one named explicitly, is read.
DATA_NAMED = re.compile(r"(?P<series>.+)-data-[0-9a-f]{8,}")

# Why a row carries no capacity (#491), short: each is a cell of every such row's line. No figure.
NO_TABLE = "no depth table yet (pool-depth run)"  # none under the reports dir
NO_CELL_NAMED = "no depth cell: names no class group or odds band"
CO_PLACER = "co-placer pool: no single-winner capacity"
NO_CELL = "the depth table has no such cell"
NO_FIGURE = "the depth cell has no capacity figure"
_CAPACITY = re.compile(r"capacity_(.+)")
# #429/#450's pool-scan cell families: one per scan version and pool (lower case), a run per cell
_POOL_SCAN = re.compile(r"japan-n1-pool-scan(?:-v\d+)?-cells-(?P<pool>[a-z_]+)")

# #430's candidate-cell keys; 'selection', 'sources' and 'ticket' are optional.
SCHEMA = (
    "study",
    "family_id",
    "cell_id",
    "pool",
    "class_group",
    "odds_band",
    "payout_rate",
    "n_bets",
    "n_race_days",
    "n_meetings",
    "mean_return",
    "se_mean_return",
    "shrunk_edge",
    "mean_odds",
)


@dataclass
class _Cells:
    """A candidate dir read: its rows, and the count of what gave none."""

    rows: list[LeaderboardRow] = field(default_factory=list)
    read: int = 0
    copies: int = 0
    replaced: int = 0  # distinct cells that gave way to a registry row
    skipped: list[str] = field(default_factory=list)
    superseded: list[str] = field(default_factory=list)  # files a newer one of its series replaces


@dataclass(frozen=True)
class _Cell:
    """A distinct cell read, before it is named."""

    cell: dict
    family: str
    cell_id: str
    pool: str | None
    canonical: str  # its sorted JSON: what tells exact copies apart, and its content suffix
    stem: str

    @property
    def base(self) -> str:
        return f"{self.family}/{self.cell_id}"

    def digest(self, width: int) -> str:
        return f"[sha256:{hashlib.sha256(self.canonical.encode()).hexdigest()[:width]}]"


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _sources(value: object) -> tuple[str, ...]:
    if isinstance(value, list) and value and all(_text(name) for name in value):
        return tuple(value)
    return UNSOURCED


def _cell_id(cell: Mapping, family: str) -> str | None:
    cell_id = _text(cell.get("cell_id"))
    if cell_id is not None:
        return cell_id
    option = _text(cell.get("option"))
    prefix = f"{family}:"
    if option is not None and option.startswith(prefix) and option[len(prefix) :].strip():
        return option[len(prefix) :]
    return option


def _selection(cell: Mapping) -> str:
    if "selection" in cell:
        selection = cell["selection"]
        return selection if isinstance(selection, str) and selection in SELECTIONS else "unknown"
    variant = cell.get("variant")
    return VARIANT_SELECTIONS.get(variant, "unknown") if isinstance(variant, str) else "unknown"


def _row(cell: Mapping, option_id: str, family: str, cell_id: str, stem: str) -> LeaderboardRow:
    study = _text(cell.get("study")) or stem
    mean = _number(cell.get("mean_return"))
    se = _number(cell.get("se_mean_return"))
    lo, hi = _number(cell.get("ci_lo")), _number(cell.get("ci_hi"))
    ticket = cell.get("ticket")
    sources = _sources(cell.get("sources"))
    notes = []
    missing = [key for key in SCHEMA if key not in cell]
    if missing:
        notes.append(f"lacks {', '.join(missing)} of #430's candidate-cell schema")
    if sources == UNSOURCED and cell.get("sources") != list(UNSOURCED):
        notes.append("no valid 'sources': read as jravan, withheld (D21)")
    if lo is not None and hi is not None and lo > hi:
        notes.append("ci_lo above ci_hi: no CI")
    ci = lo is not None and hi is not None and lo <= hi
    return LeaderboardRow(
        option_id=option_id,
        family_id=family,
        ticket=ticket if isinstance(ticket, int) and not isinstance(ticket, bool) else None,
        kind="cell",
        pool=_text(cell.get("pool")),
        class_group=_text(cell.get("class_group")),
        odds_band=_text(cell.get("odds_band")),
        segment=cell_id,
        rule=f"{study}: flat bets in {cell_id}",
        selection=_selection(cell),
        sources=sources,
        variants_searched=1,
        status=STATUS,
        return_per_100=None if mean is None else 100.0 * mean,
        se_per_100=None if se is None or se < 0 else 100.0 * se,
        ci_lo=100.0 * lo if ci else None,
        ci_hi=100.0 * hi if ci else None,
        ci_level=CELL_CI_LEVEL if ci else None,
        n=_count(cell.get("n_bets")),
        n_unit="bets",
        n_meetings=_count(cell.get("n_meetings")),
        payout_rate=_number(cell.get("payout_rate")),
        note="; ".join(notes) or None,
    )


def _entries(path: Path) -> tuple[list[tuple[str, object]], list[str]]:
    """The file's entries, each with where it sits, and why parts of the file gave none."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [], [f"{path.name}: unreadable ({type(exc).__name__})"]
    if path.suffix == ".jsonl":
        entries, problems = [], []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entries.append((f"line {number}", json.loads(line)))
            except json.JSONDecodeError:
                problems.append(f"{path.name}: line {number} does not parse")
        return entries, problems
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [], [f"{path.name}: does not parse"]
    if not isinstance(data, list):
        return [], [f"{path.name}: not a JSON array of cells"]
    return [(f"entry {i}", item) for i, item in enumerate(data, start=1)], []


def _option_ids(cells: list[_Cell]) -> list[str]:
    """Each distinct cell's option id (module docstring): from its base, its pool and its content
    alone, given which bases and ids recur; the cells' order changes none."""
    recurs = Counter(c.base for c in cells)
    ids = [
        c.base
        if recurs[c.base] == 1
        else f"{c.base}[pool:{c.pool}]"
        if c.pool is not None
        else f"{c.base}{c.digest(DIGEST_HEX)}"
        for c in cells
    ]
    # a pool shared within a base, or an id another cell spells as it is: each holder takes its
    # content's digest, the full one should the short digests collide
    for width in (DIGEST_HEX, 64):
        held = Counter(ids)
        ids = [i + c.digest(width) if held[i] > 1 else i for i, c in zip(ids, cells, strict=True)]
    return ids


def _current(files: list[Path], named: frozenset[str]) -> tuple[list[Path], list[str]]:
    """The files read, and the names of those superseded: of each series of data-fingerprinted
    files (``<series>-data-<hex fingerprint>``, #562's cross-pool v2), the ones ``named``, else
    the newest (modification time, then name). A file outside a series is always read."""
    series: dict[str, list[Path]] = {}
    for path in files:
        if (match := DATA_NAMED.fullmatch(path.stem)) is not None:
            series.setdefault(match["series"], []).append(path)
    superseded: set[Path] = set()
    for group in series.values():
        keep = [p for p in group if p.name in named]
        keep = keep or [max(group, key=lambda p: (p.stat().st_mtime_ns, p.name))]
        superseded |= set(group) - set(keep)
    return [p for p in files if p not in superseded], sorted(p.name for p in superseded)


def _read_cells(
    directory: Path, taken: frozenset[str] = frozenset(), named: frozenset[str] = frozenset()
) -> _Cells:
    """The directory's cells, less those that give way to ``taken``, the registry's option ids;
    of a data-fingerprinted series, only the newest file or those ``named`` (``_current``)."""
    found = _Cells()
    if not directory.is_dir():
        if named:
            raise FileNotFoundError(f"no candidate-cell dir for {', '.join(sorted(named))}")
        return found
    distinct: dict[str, _Cell] = {}
    files = sorted(p for p in directory.iterdir() if p.suffix in (".json", ".jsonl"))
    files = [p for p in files if p.is_file()]
    if missing := named - {p.name for p in files}:
        raise FileNotFoundError(f"no candidate-cell file {', '.join(sorted(missing))}")
    files, found.superseded = _current(files, named)
    for path in files:
        entries, problems = _entries(path)
        found.skipped += problems
        for where, cell in entries:
            found.read += 1
            if not isinstance(cell, dict):
                found.skipped.append(f"{path.name}: {where} is not a JSON object")
                continue
            family = _text(cell.get("family_id"))
            cell_id = None if family is None else _cell_id(cell, family)
            if cell_id is None:
                found.skipped.append(f"{path.name}: {where} names no family_id and cell_id")
                continue
            canonical = json.dumps(cell, sort_keys=True)
            if canonical in distinct:
                found.copies += 1
                continue
            pool = _text(cell.get("pool"))
            distinct[canonical] = _Cell(cell, family, cell_id, pool, canonical, path.stem)
    # registry precedence on the base, before the remaining cells are named
    cells = [c for c in distinct.values() if c.base not in taken]
    found.replaced = len(distinct) - len(cells)
    for c, option_id in zip(cells, _option_ids(cells), strict=True):
        if option_id in taken:
            found.replaced += 1
            continue
        found.rows.append(_row(c.cell, option_id, c.family, c.cell_id, c.stem))
    return found


def candidate_rows(directory: Path) -> list[LeaderboardRow]:
    """A row per cell of the directory's candidate-cell files (none when it does not exist), each
    with 1 variant searched; ``build_rows`` sets its family's size."""
    return _read_cells(Path(directory)).rows


class _ReadOnce(Registry):
    """A read-only view of ``registry`` for one build: the file read and its hash chain verified
    once, here, and ``records`` (through which ``Registry.family_size`` reads) answered from that
    read. Nothing appends through it."""

    def __init__(self, registry: Registry) -> None:
        super().__init__(registry.path, holdout_log=registry.holdout_log)
        self._runs = registry.records()
        self._by_family: dict[str, list[RunRecord]] = {}
        for run in self._runs:
            self._by_family.setdefault(run.family_id, []).append(run)

    def records(self, family_id: str | None = None) -> list[RunRecord]:
        return list(self._runs if family_id is None else self._by_family.get(family_id, ()))


_Key = tuple[str, str, str]


def _capacity_cells(depth: pd.DataFrame) -> dict[_Key, list[tuple[float, float, float]]]:
    """Per (pool, class_group, band) of the depth table, its (edge, quantile, stake) entries,
    NaN capacities left out (a cell whose capacities are all NaN maps to none)."""
    edges = {}
    for column in depth.columns:
        match = _CAPACITY.fullmatch(str(column))
        if match is not None:
            try:
                edges[column] = float(match[1])
            except ValueError:
                continue
    keys = zip(*(depth[c].astype(str).to_numpy() for c in pool_depth.CELL), strict=True)
    quantiles = depth["quantile"].to_numpy(dtype=float)
    values = {column: depth[column].to_numpy(dtype=float) for column in edges}
    cells: dict[_Key, list[tuple[float, float, float]]] = {}
    for i, key in enumerate(keys):
        entries = cells.setdefault(tuple(key), [])
        for column, edge in edges.items():
            stake = float(values[column][i])
            if not math.isnan(stake):
                entries.append((edge, float(quantiles[i]), stake))
    return cells


def _depth_key(row: LeaderboardRow) -> tuple[str | None, str | None, str | None]:
    """The (pool, class group, band) a row names: its own fields, or for a pool-scan registry
    row (no pool of its own) its family's pool and its 'class:<group>&band:<band>' cell id."""
    scan = _POOL_SCAN.fullmatch(row.family_id)
    prefix = f"{row.family_id}/"
    if row.pool is not None or scan is None or not row.option_id.startswith(prefix):
        return row.pool, row.class_group, row.odds_band
    cuts = [part.partition(":")[::2] for part in row.option_id[len(prefix) :].split("&")]
    parts = dict(cuts)
    if len(cuts) != 2 or set(parts) != {"class", "band"} or not all(parts.values()):
        return scan["pool"].upper(), None, None  # all, one dimension, or another cut
    return scan["pool"].upper(), parts["class"], parts["band"]


def _capacity_of(
    row: LeaderboardRow, cells: dict[_Key, list[tuple[float, float, float]]] | None
) -> LeaderboardRow:
    note = None
    entries: list[tuple[float, float, float]] = []
    pool, group, band = _depth_key(row)
    if cells is None:
        note = NO_TABLE
    elif pool is None or group is None or band is None:
        note = NO_CELL_NAMED
    elif pool in pool_depth.CO_PLACER:
        note = CO_PLACER
    elif (pool, group, band) not in cells:
        note = NO_CELL
    else:
        entries = cells[(pool, group, band)]
        note = None if entries else NO_FIGURE
    found: dict[tuple[float, float], float] = {}
    for e, q, s in entries:  # one stake per (edge, quantile): the table's first
        found.setdefault((e, q), s)
    capacity = tuple((e, q, found[(e, q)]) for e, q in sorted(found))
    return dataclasses.replace(row, capacity=capacity, capacity_note=note)


def with_capacity(
    rows: Iterable[LeaderboardRow], depth: pd.DataFrame | None
) -> list[LeaderboardRow]:
    """The rows in their order, each joined to its cell (pool, class_group, odds_band) of #432's
    long depth table ``depth`` (``pool_depth.depth_table``'s columns; None: no table): a cell
    with capacities gives the row every (edge, quantile, stake) of the table; every other row
    carries none and a ``capacity_note`` saying why (module docstring). Nothing else changes."""
    cells = None if depth is None or depth.empty else _capacity_cells(depth)
    return [_capacity_of(row, cells) for row in rows]


def _build(
    registry: Registry | None,
    candidate_dir: Path | None,
    named: frozenset[str] = frozenset(),
    depth_dir: Path | None = None,
) -> tuple[list[LeaderboardRow], _Cells]:
    """The rows, and the candidate read (with how many cells a registry row replaced)."""
    registry = _ReadOnce(default_registry("japan") if registry is None else registry)
    if candidate_dir is None:
        candidate_dir = paths.reports_dir() / CANDIDATE_DIR
    if depth_dir is None:
        depth_dir = paths.reports_dir()
    rows = legacy_rows(registry)
    cells = _read_cells(Path(candidate_dir), frozenset(row.option_id for row in rows), named)
    # rows_from_registry gives every row of a family that family's size, so the size is asked
    # again only for a candidate family with no row (none, or holdout runs only).
    sizes = {row.family_id: row.variants_searched for row in rows}
    for row in cells.rows:
        if row.family_id not in sizes:
            sizes[row.family_id] = max(1, registry.family_size(row.family_id))
        rows.append(dataclasses.replace(row, variants_searched=sizes[row.family_id]))
    return with_capacity(rows, pool_depth.read_depth_tables(depth_dir)), cells


def build_rows(
    *,
    registry: Registry | None = None,
    candidate_dir: Path | None = None,
    depth_dir: Path | None = None,
    cells_files: Sequence[str] = (),
) -> list[LeaderboardRow]:
    """``legacy_rows(registry)`` and every candidate row the registry does not give (by its base or
    its option id, module docstring), its variants searched its family's size in the registry (1
    without a run), each through ``with_capacity`` with the depth tables under ``depth_dir``;
    defaults ``default_registry('japan')``, ``$HORSETRACK_DATA_DIR/reports/candidate-cells/`` and
    ``$HORSETRACK_DATA_DIR/reports/``. ``cells_files`` names the files of a data-fingerprinted
    series to read in place of its newest (``FileNotFoundError`` if absent)."""
    return _build(registry, candidate_dir, frozenset(cells_files), depth_dir)[0]


def _write_json(path: Path, rows: list[LeaderboardRow]) -> Path:
    """``to_json`` through ``write_report``, refused inside the repository when any row is
    withheld, by its sources or its licensed mark (as ``write_leaderboard`` refuses)."""
    if any(row.withheld for row in rows) and is_inside(path, env.repo_root()):
        raise LicensedOutputInRepo(f"{path} is inside the repository and holds a withheld row")
    sources = tuple(sorted({name for row in rows for name in row.sources}))
    return write_report(path, to_json(rows), sources=sources)


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)


def leaderboard_command(
    public: bool = typer.Option(
        False, "--public", help="Also write docs/leaderboard/japan.md, the public rows only (D21)."
    ),
    cells_file: Annotated[
        list[str] | None,
        typer.Option(
            "--cells-file",
            help="A data-fingerprinted candidate-cell file (by name) to read in place of its "
            "series' newest; repeatable.",
        ),
    ] = None,
) -> None:
    """Write the Japan leaderboard of every registry family and variant and every candidate cell:
    leaderboard-japan.md and .json under the reports dir; with --public also
    docs/leaderboard/japan.md. Prints the counts only on stdout, the diagnostics on stderr, never
    a figure."""
    try:
        rows, cells = _build(None, None, frozenset(cells_file or ()))
        reports = paths.reports_dir()
        write_leaderboard(reports / PRIVATE_MD, rows, title=TITLE, public=False)
        _write_json(reports / PRIVATE_JSON, rows)
        if public:
            write_leaderboard(env.repo_root().joinpath(*PUBLIC_DOC), rows, title=TITLE, public=True)
    except (paths.DataDirNotSet, paths.DataDirInsideRepo, LicensedOutputInRepo, OSError) as exc:
        _die(f"leaderboard failed: {type(exc).__name__}: {exc}")
    except ValueError as exc:  # a refused row or registry: its message may hold a figure
        _die(f"leaderboard failed: {type(exc).__name__} (message withheld: it may hold a figure)")
    measured = sum(row.measured for row in rows)
    withheld = sum(row.withheld for row in rows)
    typer.echo(
        f"rows: {len(rows)}, measured: {measured}, unmeasured: {len(rows) - measured}, "
        f"withheld: {withheld}"
    )
    typer.echo(
        f"candidate cells: {cells.read} read, {cells.copies} exact copies, "
        f"{cells.replaced} given by the registry, {len(cells.skipped)} skipped",
        err=True,
    )
    for reason in cells.skipped:
        typer.echo(f"  skipped {reason}", err=True)
    for name in cells.superseded:
        typer.echo(f"  superseded {name}: a newer file of its series is read", err=True)
    typer.echo(f"wrote {PRIVATE_MD} and {PRIVATE_JSON} under the reports dir", err=True)
    if public:
        typer.echo(f"wrote {'/'.join(PUBLIC_DOC)}", err=True)
