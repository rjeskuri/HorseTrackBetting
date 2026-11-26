"""`horsetrack japan coverage [--write-doc]`: which source and table supplies each model input, over
which dates, and where the gaps are (ticket #374).

The inputs are `config/model_inputs.yaml` (`INPUTS_PATH`): per input its layers (C0 to C6), its
role at the decision time t, its supplies `{source, table, column, filter}` and the tickets that
cover its gaps. `coverage(inputs)` reads each supply column-scoped, grouped by (source, table) so a
table is read once for all its supplies:

- `odds_timeseries`: the day files `processed/<source>/odds_timeseries/<YYYY-MM-DD>.parquet`. A
  holdout day (`is_holdout_day("japan", d)`, D14) is dropped by its file name and never opened;
  anything that is not a day file is ignored. A day's rows count under its file's date.
- `records/<DATASPEC>/<RT>`: a dumped JV-Data family, `interim/jravan/records/<DATASPEC>/<RT>/
  *.parquet`, counted by file and by the row count in each file's footer (no column is read, so
  no value and no race day is); the span is the years of the `<YYYY>` stems (`all` gives none).
- any other table, `processed/<source>/<table>.parquet`: with a `race_date` column it is read one
  calendar year at a time through `read_store_table(..., source=, touches_price=False)`, so holdout
  days are dropped and no D10 cut applies (the audit reports what exists, not what a model may use);
  the year range comes from the `race_date` statistics in the footer. Without `race_date` it is a
  master (or a per-race table such as `dividends`), counted by rows only.

A missing table, folder or file gives 0 rows and no span. Per supply: the first and last date, the
rows after the filter (null or not), the non-null share of the column per year (NaN counts as
null), and the gap years: from the first year to the last, those with no row or with a share under
`GAP_SHARE`.

`render_summary` is the committed view (`DOC_PATH`, docs/data-coverage.md): a supply from a
licensed source (`reports.is_withheld((source, table))`: JRA-VAN, JRDB) shows `WITHHELD` and no
figure (D21); the others show dates, counts and gap years, each gap with the input's tickets; no
value from a table ever appears. `render_report` is the private view with every figure, written to
`reports_dir() / REPORT_NAME` through `write_report`, never into the repository.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Annotated, Any, NoReturn

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import typer
import yaml

from horsetrack.core import env, paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.reports import LicensedOutputInRepo, is_withheld, write_report
from horsetrack.countries.japan.models.v0.store import read_store_table

INPUTS_PATH = Path(__file__).resolve().parents[1] / "config" / "model_inputs.yaml"
DOC_PATH = env.repo_root() / "docs" / "data-coverage.md"
REPORT_NAME = "data-coverage.md"
PRIVATE_REPORT = f"$HORSETRACK_DATA_DIR/reports/{REPORT_NAME}"
COUNTRY = "japan"
LAYERS = tuple(f"C{i}" for i in range(7))
ROLES = ("feature_at_t", "label", "conditioner")
GAP_SHARE = 0.5
BOARD_TABLE = "odds_timeseries"
DUMP_PREFIX = "records/"
DATE_COLUMN = "race_date"
WITHHELD = "withheld (D21)"

_INPUT_KEYS = frozenset({"layers", "role", "supplies", "tickets", "note", "d10"})
_SUPPLY_KEYS = frozenset({"source", "table", "column", "filter"})
_DAY_FILE = re.compile(r"\d{4}-\d{2}-\d{2}")
_YEAR_FILE = re.compile(r"\d{4}")
_DUMP_TABLE = re.compile(r"records/[A-Z]{4}/[0-9A-Z]{2}")


@dataclass(frozen=True)
class Supply:
    """A column of a table in one source's store; `filter` keeps the rows whose columns equal
    every value it names."""

    source: str
    table: str
    column: str
    filter: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        """'days' (the odds day files), 'dump' (a dumped family) or 'table'."""
        if self.table == BOARD_TABLE:
            return "days"
        if self.table.startswith(DUMP_PREFIX):
            return "dump"
        return "table"

    @property
    def withheld(self) -> bool:
        return is_withheld((self.source, self.table))


@dataclass(frozen=True)
class ModelInput:
    name: str
    layers: tuple[str, ...]
    role: str
    supplies: tuple[Supply, ...]
    tickets: tuple[int, ...]
    note: str | None = None
    d10: bool = False


@dataclass(frozen=True)
class SupplyCoverage:
    input: str
    supply: Supply
    found: bool  # the table, folder or day files exist
    kind: str  # 'dated', 'days', 'dump' or 'master'
    first_date: date | None
    last_date: date | None
    rows: int
    files: int
    rows_by_year: dict[int, int]
    share_by_year: dict[int, float]
    gap_years: tuple[int, ...]
    missing_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class Coverage:
    inputs: tuple[ModelInput, ...]
    supplies: tuple[SupplyCoverage, ...]
    sources: tuple[str, ...]


# ------------------------------------------------------------------------------------------------
# The inputs file


def load_inputs(path: Path | None = None) -> list[ModelInput]:
    """The inputs of `path` (default `INPUTS_PATH`, looked up at call time), in file order."""
    path = INPUTS_PATH if path is None else Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, Mapping) or not isinstance(raw.get("inputs"), Mapping):
        raise ValueError(f"{path}: no top-level `inputs:` mapping")
    return [_input(str(name), spec, path) for name, spec in raw["inputs"].items()]


def _input(name: str, spec: Any, path: Path) -> ModelInput:
    where = f"{path.name}: input {name!r}"
    if not isinstance(spec, Mapping):
        raise ValueError(f"{where} is not a mapping")
    unknown = sorted(set(spec) - _INPUT_KEYS)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}")
    layers = spec.get("layers")
    if not isinstance(layers, list) or not layers or not set(layers) <= set(LAYERS):
        raise ValueError(f"{where}: layers must be a non-empty list of {', '.join(LAYERS)}")
    role = spec.get("role")
    if role not in ROLES:
        raise ValueError(f"{where}: role must be one of {', '.join(ROLES)}, got {role!r}")
    tickets = spec.get("tickets", [])
    if not isinstance(tickets, list) or not all(
        isinstance(t, int) and not isinstance(t, bool) and t > 0 for t in tickets
    ):
        raise ValueError(f"{where}: tickets must be a list of ticket numbers")
    note = spec.get("note")
    if note is not None and not isinstance(note, str):
        raise ValueError(f"{where}: note must be a string")
    d10 = spec.get("d10", False)
    if not isinstance(d10, bool):
        raise ValueError(f"{where}: d10 must be true or false")
    supplies = spec.get("supplies")
    if not isinstance(supplies, list):
        raise ValueError(f"{where}: supplies must be a list (`[]` with a note when none)")
    if not supplies and not (note or "").strip():
        raise ValueError(f"{where}: `supplies: []` needs a note saying why")
    return ModelInput(
        name=name,
        layers=tuple(layers),
        role=role,
        supplies=tuple(_supply(s, where) for s in supplies),
        tickets=tuple(tickets),
        note=note.strip() if isinstance(note, str) else None,
        d10=d10,
    )


def _supply(spec: Any, where: str) -> Supply:
    if not isinstance(spec, Mapping):
        raise ValueError(f"{where}: a supply is not a mapping")
    unknown = sorted(set(spec) - _SUPPLY_KEYS)
    if unknown:
        raise ValueError(f"{where}: unknown supply key(s) {unknown}")
    for key in ("source", "table", "column"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ValueError(f"{where}: a supply needs a {key}")
    flt = spec.get("filter") or {}
    if not isinstance(flt, Mapping) or not all(
        isinstance(k, str) and isinstance(v, (str, int, float, bool)) for k, v in flt.items()
    ):
        raise ValueError(f"{where}: a filter is a mapping of column names to plain values")
    table = spec["table"]
    if table.startswith(DUMP_PREFIX):
        if not _DUMP_TABLE.fullmatch(table):
            raise ValueError(f"{where}: a dumped family is records/<DATASPEC>/<RT>, got {table!r}")
        if flt:
            raise ValueError(f"{where}: a dumped family is counted only and takes no filter")
    return Supply(source=spec["source"], table=table, column=spec["column"], filter=dict(flt))


# ------------------------------------------------------------------------------------------------
# Reading the supplies


class _Tally:
    """One supply's running counts."""

    def __init__(self) -> None:
        self.rows: Counter[int] = Counter()
        self.non_null: Counter[int] = Counter()
        self.first: date | None = None
        self.last: date | None = None
        self.files = 0
        self.found = False
        self.kind = "master"
        self.missing: tuple[str, ...] = ()
        self.total_rows = 0  # a master's rows, which have no year

    def add_span(self, first: date, last: date) -> None:
        self.first = first if self.first is None else min(self.first, first)
        self.last = last if self.last is None else max(self.last, last)


def coverage(inputs: list[ModelInput]) -> Coverage:
    """Every (input, supply) of `inputs`, read as the module docstring says."""
    pairs = [(inp, s) for inp in inputs for s in inp.supplies]
    tallies = [_Tally() for _ in pairs]
    groups: dict[tuple[str, str], list[int]] = {}
    for i, (_, s) in enumerate(pairs):
        groups.setdefault((s.source, s.table), []).append(i)
    for (source, table), members in groups.items():
        supplies = [pairs[i][1] for i in members]
        group = [tallies[i] for i in members]
        kind = supplies[0].kind
        if kind == "days":
            _read_days(source, supplies, group)
        elif kind == "dump":
            _read_dump(source, table, group)
        else:
            _read_table(source, table, supplies, group)
    entries = tuple(_entry(inp, s, t) for (inp, s), t in zip(pairs, tallies, strict=True))
    return Coverage(inputs=tuple(inputs), supplies=entries, sources=_sources(pairs))


def _sources(pairs: Iterable[tuple[ModelInput, Supply]]) -> tuple[str, ...]:
    """Every source the supplies read; a JRDB table in another source's store names `jrdb` too."""
    names: set[str] = set()
    for _, s in pairs:
        names.add(s.source)
        if s.table.casefold().startswith("jrdb"):
            names.add("jrdb")
    return tuple(sorted(names))


def _entry(inp: ModelInput, s: Supply, t: _Tally) -> SupplyCoverage:
    shares = {y: t.non_null[y] / n for y, n in sorted(t.rows.items()) if n}
    gaps: list[int] = []
    if t.first is not None and t.last is not None:
        for year in range(t.first.year, t.last.year + 1):
            if not t.rows.get(year):
                gaps.append(year)
            elif t.kind != "dump" and shares[year] < GAP_SHARE:
                gaps.append(year)
    return SupplyCoverage(
        input=inp.name,
        supply=s,
        found=t.found,
        kind=t.kind,
        first_date=t.first,
        last_date=t.last,
        rows=t.total_rows + sum(t.rows.values()),
        files=t.files,
        rows_by_year=dict(sorted((y, n) for y, n in t.rows.items() if n)),
        share_by_year=shares if t.kind != "dump" else {},
        gap_years=tuple(gaps),
        missing_columns=t.missing,
    )


def _needed(supplies: Iterable[Supply]) -> list[str]:
    out: list[str] = []
    for s in supplies:
        for name in (s.column, *s.filter):
            if name not in out:
                out.append(name)
    return out


def _plain(column: pa.ChunkedArray | pa.Array) -> pa.ChunkedArray | pa.Array:
    if pa.types.is_dictionary(column.type):
        return column.cast(column.type.value_type)
    return column


def _selected(table: pa.Table, s: Supply) -> pa.Table | None:
    """The rows of `table` that `s.filter` keeps; None when a filter column is missing."""
    if not s.filter:
        return table
    mask = None
    for name, value in s.filter.items():
        if name not in table.column_names:
            return None
        column = _plain(table.column(name))
        try:
            wanted = pa.scalar(value).cast(column.type)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError) as exc:
            raise ValueError(
                f"filter {name}={value!r} does not fit the column type {column.type}"
            ) from exc
        match = pc.equal(column, wanted)
        mask = match if mask is None else pc.and_(mask, match)
    return table.filter(mask)


def _non_null(table: pa.Table, column: str) -> int:
    if column not in table.column_names:
        return 0
    values = _plain(table.column(column))
    count = len(values) - values.null_count
    if pa.types.is_floating(values.type):
        count -= pc.sum(pc.is_nan(values)).as_py() or 0
    return count


def _missing(s: Supply, available: Iterable[str]) -> tuple[str, ...]:
    have = set(available)
    return tuple(c for c in (s.column, *s.filter) if c not in have)


def _read_days(source: str, supplies: list[Supply], tallies: list[_Tally]) -> None:
    folder = paths.processed_dir(source) / BOARD_TABLE
    for t in tallies:
        t.kind = "days"
    if not folder.is_dir():
        return
    days: list[tuple[date, Path]] = []
    for path in sorted(folder.glob("*.parquet")):
        if not _DAY_FILE.fullmatch(path.stem):
            continue
        try:
            day = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if is_holdout_day(COUNTRY, day):  # D14: dropped by name, never opened
            continue
        days.append((day, path))
    needed = _needed(supplies)
    for day, path in days:
        with pq.ParquetFile(path) as handle:
            available = handle.schema_arrow.names
            table = handle.read(columns=[c for c in needed if c in available])
        for s, t in zip(supplies, tallies, strict=True):
            t.found = True
            t.files += 1
            t.missing = _missing(s, available)
            rows = _selected(table, s)
            if rows is None or rows.num_rows == 0:
                continue
            t.rows[day.year] += rows.num_rows
            t.non_null[day.year] += _non_null(rows, s.column)
            t.add_span(day, day)


def _read_dump(source: str, table: str, tallies: list[_Tally]) -> None:
    folder = paths.interim_dir(source) / table
    for t in tallies:
        t.kind = "dump"
    if not folder.is_dir():
        return
    for path in sorted(folder.glob("*.parquet")):
        rows = pq.read_metadata(path).num_rows  # the footer only: no column, no value
        year = int(path.stem) if _YEAR_FILE.fullmatch(path.stem) else None
        for t in tallies:
            t.found = True
            t.files += 1
            if year is None:
                t.total_rows += rows
            elif rows:
                t.rows[year] += rows
                t.non_null[year] += rows
                t.add_span(date(year, 1, 1), date(year, 12, 31))


def _read_table(source: str, table: str, supplies: list[Supply], tallies: list[_Tally]) -> None:
    path = paths.processed_dir(source) / f"{table}.parquet"
    if not path.is_file():
        return
    available = pq.read_schema(path).names
    for s, t in zip(supplies, tallies, strict=True):
        t.found = True
        t.missing = _missing(s, available)
    if DATE_COLUMN not in available:
        _read_master(path, supplies, tallies, available)
        return
    for t in tallies:
        t.kind = "dated"
    span = _year_span(path)
    if span is None:
        return
    columns = [DATE_COLUMN, *(c for c in _needed(supplies) if c in available and c != DATE_COLUMN)]
    for year in range(span[0], span[1] + 1):
        rows_of_year = read_store_table(
            table,
            columns=columns,
            start=date(year, 1, 1),
            end=date(year, 12, 31),
            touches_price=False,
            source=source,
        )
        for s, t in zip(supplies, tallies, strict=True):
            rows = _selected(rows_of_year, s)
            if rows is None or rows.num_rows == 0:
                continue
            t.rows[year] += rows.num_rows
            t.non_null[year] += _non_null(rows, s.column)
            low, high = pc.min_max(rows.column(DATE_COLUMN)).values()
            t.add_span(low.as_py(), high.as_py())
        del rows_of_year


def _read_master(
    path: Path, supplies: list[Supply], tallies: list[_Tally], available: list[str]
) -> None:
    """Rows only: the footer's count, or a filter's count pushed into a scan of its columns."""
    total = pq.read_metadata(path).num_rows
    for s, t in zip(supplies, tallies, strict=True):
        t.kind = "master"
        if not s.filter:
            t.total_rows = total
        elif all(k in available for k in s.filter):
            expr = None
            for name, value in s.filter.items():
                match = ds.field(name) == value
                expr = match if expr is None else expr & match
            t.total_rows = ds.dataset(path, format="parquet").count_rows(filter=expr)


def _year_span(path: Path) -> tuple[int, int] | None:
    """The first and last year of `race_date`, from the footer's statistics (a streamed scan of
    that one column when a row group has none); None when the table has no dated row."""
    meta = pq.read_metadata(path)
    lows: list[date] = []
    highs: list[date] = []
    for g in range(meta.num_row_groups):
        group = meta.row_group(g)
        if group.num_rows == 0:
            continue
        stats = next(
            (
                group.column(j).statistics
                for j in range(group.num_columns)
                if group.column(j).path_in_schema == DATE_COLUMN
            ),
            None,
        )
        if stats is None or not stats.has_min_max or not isinstance(stats.min, date):
            return _scanned_span(path)
        lows.append(stats.min)
        highs.append(stats.max)
    if not lows:
        return None
    return min(lows).year, max(highs).year


def _scanned_span(path: Path) -> tuple[int, int] | None:
    low: date | None = None
    high: date | None = None
    for batch in ds.dataset(path, format="parquet").to_batches(columns=[DATE_COLUMN]):
        lo, hi = (v.as_py() for v in pc.min_max(batch.column(0)).values())
        if lo is not None:
            low = lo if low is None else min(low, lo)
            high = hi if high is None else max(high, hi)
    return None if low is None else (low.year, high.year)


# ------------------------------------------------------------------------------------------------
# Rendering

_INTRO = """\
Generated by `horsetrack japan coverage --write-doc` (#374) from
`src/horsetrack/countries/japan/config/model_inputs.yaml`; do not edit by hand. The inputs are
those of the model stack in [docs/modeling.md](modeling.md), C0 to C6, with the evaluation's own
inputs. Per input: its layers, its role at the decision time t (`feature_at_t`, `label` or
`conditioner`), the tickets that cover its gaps, and per supply (source, `table.column`, filter)
the first and last race date, the rows and the gap years: the years from the first to the last
with no row, or with fewer than half of the year's rows non-null. A dumped JV-Data family
(`records/...`) is counted by files and rows, its span the years of its files; a master is
counted by rows only.

A supply from a licensed source (JRA-VAN, JRDB) shows `withheld (D21)` and no figure: its
dates, counts, shares and gap years are in the private report
`$HORSETRACK_DATA_DIR/reports/data-coverage.md`. The inventory says what exists, not what a model
may use: D10 and D14 decide that, and the holdout days are never read. A `D10` line marks an input
that is a price, a pool or a dividend: its rows before 2014-06-07 enter no component (D10)."""


def _tickets(tickets: Iterable[int]) -> str:
    listed = ", ".join(f"#{t}" for t in tickets)
    return listed or "no ticket yet"


def _supply_name(s: Supply) -> str:
    where = f" where {', '.join(f'{k} = {v}' for k, v in s.filter.items())}" if s.filter else ""
    return f"{s.source} `{s.table}.{s.column}`{where}"


def _span(e: SupplyCoverage) -> str:
    if e.first_date is None or e.last_date is None:
        return ""
    if e.kind == "dump":
        return f"{e.first_date.year} to {e.last_date.year}"
    return f"{e.first_date.isoformat()} to {e.last_date.isoformat()}"


def _figures(e: SupplyCoverage) -> str:
    """The supply's dates and counts, never a value from its table."""
    if not e.found:
        return "not built"
    parts = []
    if e.kind in ("days", "dump"):
        noun = "day file" if e.kind == "days" else "file"
        parts.append(f"{e.files:,} {noun}{'' if e.files == 1 else 's'}")
    span = _span(e)
    if span:
        parts.append(span)
    parts.append(f"{e.rows:,} row{'' if e.rows == 1 else 's'}")
    if e.kind == "master":
        parts.append("a master: rows only")
    elif e.kind == "dump" and e.rows and span == "":
        parts.append("no year files")
    return ", ".join(parts)


def _gaps(e: SupplyCoverage, tickets: tuple[int, ...]) -> str:
    covered = f"({_tickets(tickets)})"
    if e.missing_columns:
        return f"gap: column(s) {', '.join(e.missing_columns)} not in the built table {covered}"
    if e.rows == 0:
        return f"gap: no rows {covered}"
    if e.gap_years:
        return f"gap years {', '.join(str(y) for y in e.gap_years)} {covered}"
    return "no gap year"


def _header(inp: ModelInput) -> list[str]:
    lines = [
        f"## {inp.name}",
        "",
        f"Layers {', '.join(inp.layers)}; role `{inp.role}`; tickets {_tickets(inp.tickets)}.",
    ]
    if inp.d10:
        lines += [
            "",
            "D10: a price, pool or dividend; its rows before 2014-06-07 enter no component.",
        ]
    if inp.note:
        lines += ["", " ".join(inp.note.split())]
    lines.append("")
    return lines


def _no_supply(inp: ModelInput) -> str:
    """An input without a table: a gap when tickets cover it, else configuration (see its note)."""
    if inp.tickets:
        return f"- No table supplies this input: gap ({_tickets(inp.tickets)})"
    return "- No table supplies this input, and none is needed: see the note"


def _by_input(cov: Coverage) -> dict[str, list[SupplyCoverage]]:
    out: dict[str, list[SupplyCoverage]] = {inp.name: [] for inp in cov.inputs}
    for e in cov.supplies:
        out.setdefault(e.input, []).append(e)
    return out


def render_summary(cov: Coverage) -> str:
    """The committed view: no figure of a licensed supply, no value from any table."""
    entries = _by_input(cov)
    lines = ["# Data coverage of the model inputs", "", _INTRO, ""]
    for inp in cov.inputs:
        lines += _header(inp)
        if not inp.supplies:
            lines.append(_no_supply(inp))
        for e in entries[inp.name]:
            if e.supply.withheld:
                lines.append(f"- {_supply_name(e.supply)}: {WITHHELD}")
            else:
                lines.append(f"- {_supply_name(e.supply)}: {_figures(e)}; {_gaps(e, inp.tickets)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_report(cov: Coverage) -> str:
    """The private view: every figure, the licensed ones included. Never inside the repository."""
    entries = _by_input(cov)
    lines = [
        "# Data coverage of the model inputs: private report",
        "",
        "Built by `horsetrack japan coverage` (#374). It holds JRA-VAN and JRDB figures (D21): "
        "keep it under $HORSETRACK_DATA_DIR, never in git, a ticket or a PR.",
        f"Sources read: {', '.join(cov.sources)}.",
        "",
    ]
    for inp in cov.inputs:
        lines += _header(inp)
        if not inp.supplies:
            lines.append(_no_supply(inp))
        for e in entries[inp.name]:
            mark = " (licensed)" if e.supply.withheld else ""
            lines.append(
                f"- {_supply_name(e.supply)}{mark}: {_figures(e)}; {_gaps(e, inp.tickets)}"
            )
            if e.rows_by_year:
                per_year = ", ".join(
                    f"{y} {n:,}" + (f" ({e.share_by_year[y]:.3f})" if y in e.share_by_year else "")
                    for y, n in e.rows_by_year.items()
                )
                lines.append(f"  - rows (non-null share) by year: {per_year}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ------------------------------------------------------------------------------------------------
# The command


def coverage_command(
    write_doc: Annotated[
        bool,
        typer.Option(
            "--write-doc",
            help="Also write the committed view to docs/data-coverage.md (no licensed figure).",
        ),
    ] = False,
) -> None:
    """Audit which source and table supplies each model input: the private report goes to
    <reports dir>/data-coverage.md; with --write-doc the committed view to docs/data-coverage.md."""
    try:
        inputs = load_inputs()
        cov = coverage(inputs)
        write_report(paths.reports_dir() / REPORT_NAME, render_report(cov), sources=cov.sources)
    except (paths.DataDirNotSet, LicensedOutputInRepo, ValueError, KeyError, OSError) as exc:
        _die(f"coverage failed: {type(exc).__name__}: {exc}")
    typer.echo(f"{len(inputs)} inputs, {len(cov.supplies)} supplies read")
    typer.echo(f"report: {PRIVATE_REPORT}")
    if write_doc:
        public = tuple(s for s in cov.sources if not is_withheld((s,)))
        try:
            # the summary withholds every licensed figure, so it is built on the public sources
            write_report(DOC_PATH, render_summary(cov), sources=public)
        except (LicensedOutputInRepo, OSError) as exc:
            _die(f"coverage doc failed: {type(exc).__name__}: {exc}")
        typer.echo(f"doc: {DOC_PATH.name}")


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)
