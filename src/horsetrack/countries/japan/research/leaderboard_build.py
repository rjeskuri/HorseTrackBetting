"""The Japan leaderboard from the registry and the candidate-cell files (ticket #452).

``build_rows`` joins #451's ``legacy_rows`` (every family and variant of the Japan registry) with
``candidate_rows``, the cells the studies write in #430's schema under
``$HORSETRACK_DATA_DIR/reports/candidate-cells/``; a registry row wins over a candidate row with
the same option id. ``horsetrack japan leaderboard`` writes every row through
``core.eval.leaderboard``: ``leaderboard-japan.md`` and ``leaderboard-japan.json`` under the
reports dir and, with ``--public``, ``docs/leaderboard/japan.md`` with the withheld rows listed by
id only (D21). It prints counts, never a figure.

Reading the candidate files keeps every cell on the record:

- Files: every ``*.json`` (one JSON array) and ``*.jsonl`` (one cell per line) of the directory,
  in name order. A file, line or entry that does not parse or is not a JSON object is skipped and
  counted, with the reason (no value) printed by the command.
- A cell is option '<family_id>/<cell_id>'. One without a 'cell_id' takes its 'option' less a
  leading '<family_id>:' (#431's exacta-ceiling export names its cells so); one without a
  family id, or without both, is skipped and counted.
- An exact copy of a cell already read (a superseded copy of a file) gives no second row and is
  counted. Another cell under an id already taken (the pool scan's cell ids name the segment, not
  the pool) is kept as '<id>[pool:<pool>]', else '<id>[<k>]' with the first free k from 2; the
  first in file-name then file order keeps the bare id, so the ids are fixed by the files.
- Sources fail closed (D21): 'sources' missing, null, empty or not a list of non-blank names reads
  as ('jravan',), so the row is withheld.
- A field that is missing or of the wrong type reads as None, the selection as 'unknown' and the
  study as the file's stem; a cell without a finite 'mean_return' is an unmeasured row. Keys of
  #430's schema a cell lacks are named in the row's ``note``. Status is 'candidate': no run.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

import typer

from horsetrack.core import env, paths
from horsetrack.core.eval.leaderboard import (
    SELECTIONS,
    LeaderboardRow,
    to_json,
    write_leaderboard,
)
from horsetrack.core.eval.registry import Registry, default_registry
from horsetrack.core.eval.reports import LicensedOutputInRepo, is_inside, write_report
from horsetrack.countries.japan.research.leaderboard_backfill import legacy_rows

TITLE = "Japan leaderboard"
CANDIDATE_DIR = "candidate-cells"
PRIVATE_MD = "leaderboard-japan.md"
PRIVATE_JSON = "leaderboard-japan.json"
PUBLIC_DOC = ("docs", "leaderboard", "japan.md")
STATUS = "candidate"
UNSOURCED = ("jravan",)  # a cell that names no source is taken as licensed (D21, fail closed)

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
    skipped: list[str] = field(default_factory=list)


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


def _row(cell: Mapping, option_id: str, family: str, cell_id: str, stem: str) -> LeaderboardRow:
    study = _text(cell.get("study")) or stem
    mean = _number(cell.get("mean_return"))
    se = _number(cell.get("se_mean_return"))
    selection = cell.get("selection", "unknown")
    ticket = cell.get("ticket")
    sources = _sources(cell.get("sources"))
    notes = []
    missing = [key for key in SCHEMA if key not in cell]
    if missing:
        notes.append(f"lacks {', '.join(missing)} of #430's candidate-cell schema")
    if sources == UNSOURCED and cell.get("sources") != list(UNSOURCED):
        notes.append("no valid 'sources': read as jravan, withheld (D21)")
    return LeaderboardRow(
        option_id=option_id,
        family_id=family,
        ticket=ticket if isinstance(ticket, int) and not isinstance(ticket, bool) else None,
        kind="cell",
        pool=_text(cell.get("pool")),
        segment=cell_id,
        rule=f"{study}: flat bets in {cell_id}",
        selection=selection if selection in SELECTIONS else "unknown",
        sources=sources,
        variants_searched=1,
        status=STATUS,
        return_per_100=None if mean is None else 100.0 * mean,
        se_per_100=None if se is None or se < 0 else 100.0 * se,
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


def _free_id(base: str, pool: str | None, taken: set[str]) -> str:
    if base not in taken:
        return base
    if pool is not None and f"{base}[pool:{pool}]" not in taken:
        return f"{base}[pool:{pool}]"
    k = 2
    while f"{base}[{k}]" in taken:
        k += 1
    return f"{base}[{k}]"


def _read_cells(directory: Path) -> _Cells:
    found = _Cells()
    if not directory.is_dir():
        return found
    taken: set[str] = set()
    seen: set[str] = set()
    files = sorted(p for p in directory.iterdir() if p.suffix in (".json", ".jsonl"))
    for path in (p for p in files if p.is_file()):
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
            copy = json.dumps(cell, sort_keys=True)
            if copy in seen:
                found.copies += 1
                continue
            seen.add(copy)
            option_id = _free_id(f"{family}/{cell_id}", _text(cell.get("pool")), taken)
            taken.add(option_id)
            found.rows.append(_row(cell, option_id, family, cell_id, path.stem))
    return found


def candidate_rows(directory: Path) -> list[LeaderboardRow]:
    """A row per cell of the directory's candidate-cell files (none when it does not exist), each
    with 1 variant searched; ``build_rows`` sets its family's size."""
    return _read_cells(Path(directory)).rows


def _build(
    registry: Registry | None, candidate_dir: Path | None
) -> tuple[list[LeaderboardRow], _Cells, int]:
    """The rows, the candidate read, and how many candidate rows a registry row replaced."""
    registry = default_registry("japan") if registry is None else registry
    if candidate_dir is None:
        candidate_dir = paths.reports_dir() / CANDIDATE_DIR
    rows = legacy_rows(registry)
    cells = _read_cells(Path(candidate_dir))
    taken = {row.option_id for row in rows}
    # rows_from_registry gives every row of a family that family's size, so the registry is read
    # again only for a candidate family with no row (none, or holdout runs only).
    sizes = {row.family_id: row.variants_searched for row in rows}
    replaced = 0
    for row in cells.rows:
        if row.option_id in taken:
            replaced += 1
            continue
        if row.family_id not in sizes:
            sizes[row.family_id] = max(1, registry.family_size(row.family_id))
        rows.append(dataclasses.replace(row, variants_searched=sizes[row.family_id]))
    return rows, cells, replaced


def build_rows(
    *, registry: Registry | None = None, candidate_dir: Path | None = None
) -> list[LeaderboardRow]:
    """``legacy_rows(registry)`` and every candidate row whose option id the registry does not
    give, its variants searched its family's size in the registry (1 without a run); defaults
    ``default_registry('japan')`` and ``$HORSETRACK_DATA_DIR/reports/candidate-cells/``."""
    return _build(registry, candidate_dir)[0]


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
) -> None:
    """Write the Japan leaderboard of every registry family and variant and every candidate cell:
    leaderboard-japan.md and .json under the reports dir; with --public also
    docs/leaderboard/japan.md. Prints counts only, never a figure."""
    try:
        rows, cells, replaced = _build(None, None)
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
        f"candidate cells: {cells.read} read, {cells.copies} exact copies, {replaced} given by "
        f"the registry, {len(cells.skipped)} skipped"
    )
    for reason in cells.skipped:
        typer.echo(f"  skipped {reason}")
    typer.echo(f"wrote {PRIVATE_MD} and {PRIVATE_JSON} under the reports dir")
    if public:
        typer.echo(f"wrote {'/'.join(PUBLIC_DOC)}")
