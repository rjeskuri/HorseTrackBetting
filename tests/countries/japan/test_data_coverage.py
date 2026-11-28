"""Ticket #374: the model-input coverage audit (`horsetrack.countries.japan.research.coverage`).

Contract pinned here (the coder implements what these tests call):

The inputs, `src/horsetrack/countries/japan/config/model_inputs.yaml`:
- Top level `inputs:`, a mapping from each input's name to `{layers, role, supplies, tickets,
  note}`. `layers` is a non-empty list of `C0`..`C6`; `role` one of `feature_at_t`, `label`,
  `conditioner`; `supplies` a list of `{source, table, column, filter}`, `filter` an optional
  equality map (all its keys must match); `tickets` a list of ints; `note` a string, non-empty
  when `supplies: []`.
- `load_inputs(path=None) -> list[ModelInput]`; `path` None reads `coverage.INPUTS_PATH` (the
  committed yaml), looked up at call time. A `ModelInput` has `name`, `layers`, `role`,
  `supplies`, `tickets` and `note` (None when absent); each supply is a `Supply` with `source`,
  `table`, `column` and `filter` (a dict, `{}` when absent).
- In the committed yaml a supply's source is `jra` or `jravan`, and its table and column exist:
  in `schemas.TABLES`, in `compact.SCHEMA` (table `odds_timeseries`) or in a jravan-only schema
  (`final_odds`, `pool_votes`, `jockeys`, `trainers`, `breeding_horses`, `produce`, `lineages`;
  source `jravan` only). A jravan supply on a `schemas.TABLES` table names one the jravan builders
  write. Every filter key is a column of the table. A dumped family is the table
  `records/<DATASPEC>/<RT>` (source `jravan`, a family outside `dump.MAPPED`; its column is not
  checked, a dump is counted only).

How `coverage(inputs) -> Coverage` reads a supply:
- `odds_timeseries`: the day files `processed/<source>/odds_timeseries/<YYYY-MM-DD>.parquet`
  (anything else in the folder, such as `_build.json`, is ignored). A holdout day
  (`is_holdout_day("japan", d)`) is dropped by its file name and never opened.
- `records/<DATASPEC>/<RT>`: the files `interim/jravan/records/<DATASPEC>/<RT>/*.parquet`, by
  file and row count; the span is the years of the `<YYYY>` file stems (an `all` file gives none).
  A family that carries the JV-Data race key (the top-level text columns `開催年`, YYYY, and
  `開催月日`, MMDD, as `dump` writes every race-keyed record type) never counts a record whose
  race day is a holdout day (D14), in a year file or an `all` file; the day is the race key's,
  not `データ作成年月日`. A file without those columns is counted from its footer.
- Any other table, `processed/<source>/<table>.parquet`: with a `race_date` column, read through
  `read_store_table(..., source=)`, column-scoped, the filter applied, holdout days dropped, and no
  D10 cut (the audit reports what exists, so price rows before 2014-06-07 count). Without a
  `race_date` column the table is counted by rows only (no span pinned): a per-race table (one
  with `race_id`: `dividends`, `sectionals`, ...) counts only the rows of races not on a holdout
  day (D14; the store builders keep holdout days, only readers drop them), the day being the
  date in the race id (`jra:<YYYY-MM-DD>:<VENUE>:<NN>` in both stores) or, equivalently, the
  races `read_store_table("races", source=)` returns; a master with no race key (`horses`,
  `jockeys`) counts every row.
- A table, folder or file that does not exist gives 0 rows and no span, and is not an error.

`Coverage.supplies` holds one `SupplyCoverage` per (input, supply), with:
- `input`: the input's name, and `supply`: the `Supply`;
- `first_date` / `last_date`: `date` or None (a dump's carry the years of its first and last year
  files, only `.year` is read);
- `rows`: the supply's rows after its filter, null or not;
- `files`: the day files or dump files read (not pinned for a table);
- `share_by_year`: `{year: share}`, the column's non-null share among that year's rows, for the
  years with rows (a year without rows may be absent);
- `gap_years`: the years from the first to the last with no rows, or with a share under 0.5
  (exactly 0.5 is not a gap).
`Coverage.sources` is a tuple naming every source the supplies read.

Rendering:
- `render_summary(cov)`, the committed view: per input its layers, role, supplies and, for a
  supply that is not withheld, its first and last dates (ISO), rows and gap years, each gap year
  on a line that carries one of its input's tickets as `#N`. A supply is withheld when
  `reports.is_withheld((source, table))` (jravan, JRDB, a `jrdb_*` table): it shows the marker
  `withheld (D21)` and no figure from its data (no date, count, share or gap year). No value from
  a table ever appears.
- `render_report(cov)`, the private view: every figure, the licensed ones included.
- `horsetrack japan coverage` writes the private report to `paths.reports_dir() /
  "data-coverage.md"` through `write_report(..., sources=cov.sources)`. With `--write-doc` it
  also writes `coverage.DOC_PATH` (default `<repo>/docs/data-coverage.md`, looked up at call
  time), holding `render_summary(cov)`; that write lands inside the repository, so it must not be
  refused as a licensed output (the summary withholds every licensed figure).

Every value is synthetic, under a tmp HORSETRACK_DATA_DIR; no test reads real data. The holdout
days used are real ones under the committed seed, checked with `is_holdout_day`.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party.
from __future__ import annotations  # noqa: I001

import builtins
import io
import os
import re
import shutil
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import yaml
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import env, paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.reports import LicensedOutputInRepo, is_withheld, write_report
from horsetrack.core.store import schemas
from horsetrack.countries.japan.jravan import bloodlines, career_counts, dump, masters, odds, votes
from horsetrack.countries.japan.logger import compact
from horsetrack.countries.japan.research import coverage as cv

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED_DOC = REPO_ROOT / "docs" / "data-coverage.md"
LAYERS = frozenset(f"C{i}" for i in range(7))
ROLES = frozenset({"feature_at_t", "label", "conditioner"})
SOURCES = frozenset({"jra", "jravan"})
WITHHELD = "withheld (D21)"
REPORT_NAME = "data-coverage.md"
JRAVAN_ONLY = {
    "final_odds": odds.FINAL_ODDS_SCHEMA.names,
    "pool_votes": votes.POOL_VOTES_SCHEMA.names,
    "jockeys": masters.JOCKEYS_SCHEMA.names,
    "trainers": masters.TRAINERS_SCHEMA.names,
    "breeding_horses": bloodlines.BREEDING_HORSES_SCHEMA.names,
    "produce": bloodlines.PRODUCE_SCHEMA.names,
    "lineages": bloodlines.LINEAGES_SCHEMA.names,
    "career_counts": career_counts.CAREER_COUNTS_SCHEMA.names,  # #407
}
# the schemas.TABLES tables the jravan builders write (races.py, runners.py, dividends.py,
# masters.py, horse_runs.py)
JRAVAN_STORE_TABLES = frozenset(
    {"races", "meetings", "runners", "dividends", "horses", "horse_pedigree", "horse_runs"}
)
DUMP_TABLE = re.compile(r"records/([A-Z]{4})/([0-9A-Z]{2})")
VENUE = "QQV"  # a made-up venue code, so a race id never matches a real one
HOLDOUT_A, HOLDOUT_B = date(2026, 11, 29), date(2026, 12, 1)
DEV_LATE = date(2026, 11, 28)

# ------------------------------------------------------------------------------------------------
# Synthetic stores under a tmp data dir


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(root))
    return root


def race_id(day: date, race_no: int = 1) -> str:
    """The store's race id, `jra:<YYYY-MM-DD>:<VENUE>:<NN>` (`keys.race_id`, both stores)."""
    return f"jra:{day.isoformat()}:{VENUE}:{race_no:02d}"


def write_parquet(path: Path, table: pa.Table) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def store_table(source: str, name: str, schema: pa.Schema, rows: list[dict]) -> Path:
    table = pa.Table.from_pylist(rows, schema=schema)
    return write_parquet(paths.processed_dir(source) / f"{name}.parquet", table)


RUNNERS = pa.schema(
    [
        ("race_id", pa.string()),
        ("race_date", pa.date32()),
        ("horse_no", pa.int64()),
        ("horse_name", pa.string()),
        ("win_odds", pa.float64()),
        ("horse_wt_kg", pa.int64()),
    ]
)
RACES = pa.schema(
    [
        ("race_id", pa.string()),
        ("race_date", pa.date32()),
        ("going", pa.string()),
        ("rail_setting", pa.string()),
    ]
)
HORSES = pa.schema(
    [("jurisdiction", pa.string()), ("horse_id", pa.string()), ("name", pa.string())]
)
JRDB_PRE_RACE = pa.schema(
    [(name, pa.date32() if name == "race_date" else pa.int64() if name == "horse_no" else
      pa.string()) for name in schemas.TABLES["jrdb_pre_race"].columns]
)  # fmt: skip


def runners_on(
    day: date, win_odds: list, weights: list | None = None, name: str = "Synthetic Runner"
) -> list[dict]:
    weights = weights if weights is not None else [None] * len(win_odds)
    return [
        {"race_id": race_id(day), "race_date": day, "horse_no": n + 1, "horse_name": name,
         "win_odds": o, "horse_wt_kg": w}
        for n, (o, w) in enumerate(zip(win_odds, weights, strict=True))
    ]  # fmt: skip


def races_on(day: date, going: list, rail: list) -> list[dict]:
    return [
        {"race_id": race_id(day, n + 1), "race_date": day, "going": g, "rail_setting": r}
        for n, (g, r) in enumerate(zip(going, rail, strict=True))
    ]


def odds_row(
    day: date, pool: str, combination: str, price: float | None, investment: float | None = None
) -> dict:
    return {
        "jurisdiction": "jra", "race_id": race_id(day), "race_date": day, "venue": VENUE,
        "race_no": 1, "pool": pool, "combination": combination, "odds": price,
        "investment": investment, "data_kubun": "5",
    }  # fmt: skip


def board_row(day: date, pool: str, combination: str, price: float | None) -> dict:
    return {
        "race_id": race_id(day), "venue": VENUE, "race_no": 1, "pool": pool,
        "combination": combination,
        "snapshot_utc": datetime(day.year, day.month, day.day, 1, 0, tzinfo=timezone.utc),
        "odds": price,
    }  # fmt: skip


def board_day_path(source: str, day: date) -> Path:
    return paths.processed_dir(source) / "odds_timeseries" / f"{day.isoformat()}.parquet"


def board_day(source: str, day: date, rows: list[dict]) -> Path:
    return write_parquet(
        board_day_path(source, day), pa.Table.from_pylist(rows, schema=compact.SCHEMA)
    )


def dump_file(family: str, stem: str, n: int) -> Path:
    table = pa.table(
        {
            "_file": pa.array([f"stored/{family}/{stem}.jvd"] * n, pa.string()),
            "_record": pa.array(range(1, n + 1), pa.int64()),
            "架空項目": pa.array([f"v{i}" for i in range(n)], pa.string()),
        }
    )
    return write_parquet(
        paths.interim_dir("jravan") / "records" / family / f"{stem}.parquet", table
    )


def horses(source: str, names: list[str]) -> Path:
    rows = [
        {"jurisdiction": "jra", "horse_id": f"{i:010d}", "name": n} for i, n in enumerate(names)
    ]
    return store_table(source, "horses", HORSES, rows)


def jockeys(n: int) -> Path:
    rows = [
        {"jurisdiction": "jra", "jockey_code": f"{i:05d}", "jockey_key": f"jockey:{i:05d}",
         "name": f"Jockey {i}"}
        for i in range(1, n + 1)
    ]  # fmt: skip
    return store_table("jravan", "jockeys", masters.JOCKEYS_SCHEMA, rows)


# ------------------------------------------------------------------------------------------------
# The inputs file and the coverage lookups


def supply(source: str, table: str, column: str, **filter: str) -> dict:
    out: dict = {"source": source, "table": table, "column": column}
    if filter:
        out["filter"] = dict(filter)
    return out


def an_input(
    layers: list[str], role: str, supplies: list[dict], tickets: list[int], note: str | None = None
) -> dict:
    out: dict = {"layers": layers, "role": role, "supplies": supplies, "tickets": tickets}
    if note is not None:
        out["note"] = note
    return out


def write_inputs(path: Path, inputs: dict[str, dict]) -> Path:
    text = yaml.safe_dump({"inputs": inputs}, sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")
    return path


def run_coverage(tmp_path: Path, inputs: dict[str, dict]):
    return cv.coverage(cv.load_inputs(write_inputs(tmp_path / "model_inputs.yaml", inputs)))


def entry(cov, name: str, source: str | None = None, table: str | None = None):
    found = [
        e
        for e in cov.supplies
        if e.input == name
        and (source is None or e.supply.source == source)
        and (table is None or e.supply.table == table)
    ]
    assert len(found) == 1, f"expected one coverage entry for {name} {source} {table}: {found}"
    return found[0]


def year_token(year: int) -> re.Pattern[str]:
    """`year` standing alone, not inside an ISO date or a longer number."""
    return re.compile(rf"(?<![\d-]){year}(?![\d-])")


def ticket_token(ticket: int) -> re.Pattern[str]:
    return re.compile(rf"#{ticket}(?!\d)")


# ------------------------------------------------------------------------------------------------
# The committed yaml


def supply_problem(s) -> str | None:
    """Why a committed supply does not name an existing table and column, or None."""
    if s.source not in SOURCES:
        return f"source {s.source!r} is not one of {sorted(SOURCES)}"
    keys = list(dict(s.filter or {}))
    dumped = DUMP_TABLE.fullmatch(s.table)
    if dumped:
        if s.source != "jravan":
            return f"dumped family {s.table} with source {s.source!r}, not jravan"
        if (dumped[1], dumped[2]) in dump.MAPPED:
            return f"{s.table} is mapped into a store table: name that table instead"
        return None
    if s.table == "odds_timeseries":
        columns = list(compact.SCHEMA.names)
    elif s.table in JRAVAN_ONLY:
        if s.source != "jravan":
            return f"{s.table} is a jravan-only table, named with source {s.source!r}"
        columns = list(JRAVAN_ONLY[s.table])
    elif s.table in schemas.TABLES:
        if s.source == "jravan" and s.table not in JRAVAN_STORE_TABLES:
            return f"no jravan builder writes {s.table}"
        columns = list(schemas.TABLES[s.table].columns)
    else:
        return f"table {s.table!r} is in no schema"
    missing = [c for c in [s.column, *keys] if c not in columns]
    if missing:
        return f"{s.table} has no column(s) {missing}"
    return None


def test_every_input_names_layers_a_role_and_existing_supplies() -> None:
    inputs = cv.load_inputs()
    assert inputs, "the committed model_inputs.yaml lists no input"
    names = [i.name for i in inputs]
    assert all(isinstance(n, str) and n.strip() for n in names), names
    assert len(set(names)) == len(names), "input names repeat"
    problems = []
    for inp in inputs:
        if not inp.layers or not set(inp.layers) <= LAYERS:
            problems.append(f"{inp.name}: layers {inp.layers!r} not a non-empty subset of C0..C6")
        if inp.role not in ROLES:
            problems.append(f"{inp.name}: role {inp.role!r} not one of {sorted(ROLES)}")
        if not all(isinstance(t, int) and not isinstance(t, bool) and t > 0 for t in inp.tickets):
            problems.append(f"{inp.name}: tickets {inp.tickets!r} are not ticket numbers")
        if not inp.supplies and not (inp.note or "").strip():
            problems.append(f"{inp.name}: supplies: [] without a note")
        for s in inp.supplies:
            problem = supply_problem(s)
            if problem:
                problems.append(f"{inp.name}: {problem}")
    assert not problems, "\n".join(problems)


def test_every_layer_of_the_stack_has_an_input() -> None:
    covered = {layer for inp in cv.load_inputs() for layer in inp.layers}
    assert LAYERS <= covered, f"no input in {sorted(LAYERS - covered)}"


def test_the_inputs_the_request_names_are_listed() -> None:
    """The request's known candidates (weather, cushion, moisture, the board at t) and the final
    Win price of C4, which is a label or a conditioner, never a feature at t (as-of spec)."""
    inputs = cv.load_inputs()
    pairs = [(inp, s) for inp in inputs for s in inp.supplies]

    def supplied(source: str, table: str, columns: set[str]) -> bool:
        return any(
            s.source == source and s.table == table and s.column in columns for _, s in pairs
        )

    assert supplied("jra", "track_condition", {"cushion"}), "no input reads the cushion value"
    assert supplied("jra", "track_condition", {"moisture_goal", "moisture_4c"}), "no moisture"
    boards = [
        inp
        for inp in inputs
        if inp.role == "feature_at_t"
        and {s.source for s in inp.supplies if s.table == "odds_timeseries"} >= SOURCES
    ]
    assert boards, "no feature-at-t input reads the board at t from both jra and jravan"
    weather = [inp for inp in inputs if inp.name == "weather"]
    assert len(weather) == 1, "no input named weather"
    assert "C3" in weather[0].layers and 89 in weather[0].tickets, weather[0]
    final_win = [
        inp
        for inp, s in pairs
        if s.source == "jravan"
        and s.table == "final_odds"
        and dict(s.filter or {}).get("pool") == "WIN"
    ]
    assert any("C4" in inp.layers for inp in final_win), "no C4 input reads the final Win price"
    leaks = sorted({inp.name for inp, s in pairs if s.table == "final_odds"
                    and inp.role == "feature_at_t"})  # fmt: skip
    assert not leaks, f"final_odds is never a feature at t: {leaks}"


BRACKET_POOLS = ("PLACE", "QUINELLA_PLACE")  # the pools that show a lo-hi range (odds, odds_hi)


def test_the_bracket_high_end_is_an_input() -> None:
    """docs/modeling.md diagram 4, box BRK (#118 shadow_values.py): Place and Wide are valued at
    the low and the high end of the bracket the pool shows, and the nowcast reads the high end too.
    So `odds_hi` of PLACE and QUINELLA_PLACE is an input of C5 and C6: on the board at t
    (`odds_timeseries`, jra and jravan; a feature at t) and in jravan's `final_odds`."""
    inputs = cv.load_inputs()
    wanted = [
        *(("jra", "odds_timeseries", pool) for pool in BRACKET_POOLS),
        *(("jravan", "odds_timeseries", pool) for pool in BRACKET_POOLS),
        *(("jravan", "final_odds", pool) for pool in BRACKET_POOLS),
    ]
    missing = []
    for source, table, pool in wanted:
        holders = [
            inp
            for inp in inputs
            for s in inp.supplies
            if (s.source, s.table, s.column) == (source, table, "odds_hi")
            and dict(s.filter or {}) == {"pool": pool}
        ]
        if not any({"C5", "C6"} <= set(inp.layers) for inp in holders):
            missing.append(f"{source} {table}.odds_hi where pool = {pool} (in: {holders})")
    assert not missing, "no C5 and C6 input reads the bracket's high end:\n" + "\n".join(missing)
    board_high = sorted({inp.name for inp in inputs for s in inp.supplies
                         if s.table == "odds_timeseries" and s.column == "odds_hi"
                         and inp.role != "feature_at_t"})  # fmt: skip
    assert not board_high, f"the board's high end at t is a feature at t: {board_high}"


# ------------------------------------------------------------------------------------------------
# Coverage over synthetic stores


def write_span_runners() -> None:
    """jra: 2012 and 2013 (and a holdout day); jravan: 2020 and 2021 (and a holdout day)."""
    assert is_holdout_day("japan", HOLDOUT_A) and is_holdout_day("japan", HOLDOUT_B)
    store_table("jra", "runners", RUNNERS, [
        *runners_on(date(2012, 3, 3), [3.4, 5.6]),
        *runners_on(date(2012, 7, 14), [2.2]),
        *runners_on(date(2013, 5, 5), [4.0, None, 7.5, 12.0], [480, 472, 500, 466]),
        *runners_on(HOLDOUT_A, [9.9, 8.8], [490, 470]),
    ])  # fmt: skip
    store_table("jravan", "runners", RUNNERS, [
        *runners_on(date(2020, 1, 5), [6.1, None], [450, 452]),
        *runners_on(date(2021, 6, 6), [2.5, 3.5, 4.5]),
        *runners_on(HOLDOUT_B, [1.1, 1.2, 1.3], [440, 441, 442]),
    ])  # fmt: skip


SPAN_INPUTS = {
    "win_price": an_input(
        ["C4"], "label",
        [supply("jra", "runners", "win_odds"), supply("jravan", "runners", "win_odds")], [901],
    ),
    "body_weight": an_input(
        ["C2"], "feature_at_t", [supply("jra", "runners", "horse_wt_kg")], [902]
    ),
}  # fmt: skip


def test_coverage_reports_span_rows_and_non_null_share_per_year(data: Path, tmp_path: Path) -> None:
    write_span_runners()
    cov = run_coverage(tmp_path, SPAN_INPUTS)
    # (input, source, column) -> first, last, rows, share per year, gap years; the holdout rows
    # count nowhere, a null value still counts as a row, and 2012 prices (before D10) count
    expected = {
        ("win_price", "jra"): (date(2012, 3, 3), date(2013, 5, 5), 7, {2012: 1.0, 2013: 0.75}, []),
        ("win_price", "jravan"): (date(2020, 1, 5), date(2021, 6, 6), 5, {2020: 0.5, 2021: 1.0},
                                  []),
        ("body_weight", "jra"): (date(2012, 3, 3), date(2013, 5, 5), 7, {2012: 0.0, 2013: 1.0},
                                 [2012]),
    }  # fmt: skip
    assert len(cov.supplies) == 3
    for (name, source), (first, last, rows, shares, gaps) in expected.items():
        e = entry(cov, name, source)
        got = (e.first_date, e.last_date, e.rows, sorted(e.gap_years))
        assert got == (first, last, rows, gaps), (name, source, got)
        assert dict(e.share_by_year) == pytest.approx(shares), (name, source, e.share_by_year)
    assert {"jra", "jravan"} <= set(cov.sources)


def write_gap_races() -> None:
    store_table("jra", "races", RACES, [
        *races_on(date(2015, 1, 10), ["g1", "g2", "g3", "g1"], ["A", None, None, None]),
        # 2016: no race at all
        *races_on(date(2017, 2, 11), ["g1", None, "g2", None, None], ["A", "B", "C", "A", "B"]),
        *races_on(date(2018, 3, 3), ["g1", None, "g2", None, "g3"], [None] * 5),
        *races_on(date(2018, 9, 9), [None, "g1", None, "g2", None], [None] * 5),
        *races_on(date(2019, 4, 6), ["g1", "g2", "g3"], ["A", "B", None]),
    ])  # fmt: skip


GAP_INPUTS = {
    "going": an_input(["C3"], "feature_at_t", [supply("jra", "races", "going")], [905, 906]),
    "rail": an_input(["C3"], "feature_at_t", [supply("jra", "races", "rail_setting")], [907]),
}
# going: 2015 4/4, 2016 none, 2017 2/5, 2018 5/10, 2019 3/3
# rail_setting: 2015 1/4, 2016 none, 2017 5/5, 2018 0/10, 2019 2/3
GAPS = {"going": [2016, 2017], "rail": [2015, 2016, 2018]}


def test_a_year_without_rows_or_mostly_null_is_a_gap(data: Path, tmp_path: Path) -> None:
    write_gap_races()
    cov = run_coverage(tmp_path, GAP_INPUTS)
    for name, gaps in GAPS.items():
        e = entry(cov, name, "jra")
        assert sorted(e.gap_years) == gaps, (name, e.gap_years)
        assert (e.first_date, e.last_date, e.rows) == (date(2015, 1, 10), date(2019, 4, 6), 22)
    going, rail = entry(cov, "going", "jra"), entry(cov, "rail", "jra")
    assert going.share_by_year[2018] == pytest.approx(0.5)  # at 0.5: not a gap
    assert going.share_by_year[2017] == pytest.approx(0.4)
    assert rail.share_by_year[2015] == pytest.approx(0.25)
    assert rail.share_by_year[2019] == pytest.approx(2 / 3)


def test_a_filter_selects_the_rows_of_a_supply(data: Path, tmp_path: Path) -> None:
    d15, d16, d17a, d17b, d18 = (date(2015, 12, 26), date(2016, 1, 5), date(2017, 3, 4),
                                 date(2017, 12, 28), date(2018, 2, 3))  # fmt: skip
    store_table("jravan", "final_odds", odds.FINAL_ODDS_SCHEMA, [
        odds_row(d15, "PLACE", "1", 1.5),  # before the first WIN row
        odds_row(d16, "WIN", "1", 2.0), odds_row(d16, "WIN", "2", None),
        odds_row(d16, "WIN", "3", 8.0), odds_row(d16, "WIN", "__pool__", None, 1_000_000.0),
        odds_row(d16, "PLACE", "1", 1.1), odds_row(d16, "PLACE", "__pool__", None, 500_000.0),
        odds_row(d17a, "WIN", "1", 3.0), odds_row(d17a, "WIN", "2", 4.0),
        odds_row(d17a, "WIN", "__pool__", None, 2_000_000.0),
        odds_row(d17a, "QUINELLA", "1,2", 15.0),
        odds_row(d17b, "WIN", "1", None), odds_row(d17b, "WIN", "__pool__", None, 900_000.0),
        odds_row(d18, "PLACE", "1", 1.2),  # after the last WIN row
    ])  # fmt: skip
    cov = run_coverage(tmp_path, {
        "final_win": an_input(["C4"], "label",
                              [supply("jravan", "final_odds", "odds", pool="WIN")], [903]),
        "win_pool": an_input(["C6"], "conditioner",
                             [supply("jravan", "final_odds", "investment", pool="WIN",
                                     combination="__pool__")], [904]),
        "all_prices": an_input(["C5"], "label", [supply("jravan", "final_odds", "odds")], [908]),
    })  # fmt: skip
    expected = {
        # WIN: 2016 2 of 4 priced, 2017 2 of 5
        "final_win": (d16, d17b, 9, {2016: 0.5, 2017: 0.4}, [2017]),
        # WIN pool rows only (both keys must match): 1 in 2016, 2 in 2017, every one invested
        "win_pool": (d16, d17b, 3, {2016: 1.0, 2017: 1.0}, []),
        # no filter: every pool
        "all_prices": (d15, d18, 14, {2015: 1.0, 2016: 0.5, 2017: 0.5, 2018: 1.0}, []),
    }
    for name, (first, last, rows, shares, gaps) in expected.items():
        e = entry(cov, name, "jravan")
        got = (e.first_date, e.last_date, e.rows, sorted(e.gap_years))
        assert got == (first, last, rows, gaps), (name, got)
        assert dict(e.share_by_year) == pytest.approx(shares), (name, e.share_by_year)


def record_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """From now on, every path handed to the usual file and Parquet readers."""
    seen: list[str] = []

    def note(*args, **kwargs) -> None:
        for value in (*args, *kwargs.values()):
            for item in value if isinstance(value, (list, tuple)) else [value]:
                if isinstance(item, (str, os.PathLike)):
                    seen.append(os.fspath(item))

    def wrap(owner, name: str) -> None:
        original = getattr(owner, name)

        def wrapper(*args, **kwargs):
            note(*args, **kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, wrapper)

    for owner, name in (
        (pq, "read_table"),
        (pq, "read_metadata"),
        (pq, "read_schema"),
        (ds, "dataset"),
        (pd, "read_parquet"),
        (builtins, "open"),
        (io, "open"),
        (Path, "open"),
    ):
        wrap(owner, name)

    original_file = pq.ParquetFile

    class RecordingParquetFile(original_file):
        def __init__(self, source, *args, **kwargs):
            note(source)
            super().__init__(source, *args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", RecordingParquetFile)
    return seen


def test_odds_timeseries_days_are_listed_and_holdout_days_never_read(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    days = (date(2025, 12, 27), date(2026, 1, 4), DEV_LATE)
    assert not any(is_holdout_day("japan", d) for d in (*days, date(2026, 11, 21)))
    assert is_holdout_day("japan", HOLDOUT_A) and is_holdout_day("japan", HOLDOUT_B)
    board_day("jravan", days[0], [
        board_row(days[0], "WIN", "1", 2.0), board_row(days[0], "WIN", "2", 3.0),
        board_row(days[0], "WIN", "3", 4.5),
        board_row(days[0], "PLACE", "1", 1.2), board_row(days[0], "PLACE", "2", 1.5),
    ])  # fmt: skip
    board_day("jravan", days[1], [
        board_row(days[1], "WIN", "1", 2.2), board_row(days[1], "WIN", "2", None),
        board_row(days[1], "WIN", "3", None), board_row(days[1], "WIN", "4", 6.0),
        board_row(days[1], "QUINELLA", "1,2", 10.0),
    ])  # fmt: skip
    board_day("jravan", days[2], [board_row(days[2], "WIN", str(n), 2.0 + n) for n in range(5)])
    # a holdout day as a valid file with many rows, and one that is not Parquet at all
    board_day("jravan", HOLDOUT_B, [board_row(HOLDOUT_B, "WIN", str(n), 5.0) for n in range(9)])
    board_day_path("jravan", HOLDOUT_A).write_bytes(b"PAR1 not a parquet file: never open it")
    (board_day_path("jravan", days[0]).parent / "_build.json").write_text("{}", encoding="utf-8")
    # the logger's own days (source jra): one development day and a holdout day
    board_day("jra", date(2026, 11, 21), [
        board_row(date(2026, 11, 21), "WIN", "1", 3.3),
        board_row(date(2026, 11, 21), "WIN", "2", 4.4),
    ])  # fmt: skip
    board_day_path("jra", HOLDOUT_A).write_bytes(b"PAR1 not a parquet file: never open it")
    inputs = cv.load_inputs(write_inputs(tmp_path / "model_inputs.yaml", {
        "board_at_t": an_input(["C0", "C6"], "feature_at_t", [
            supply("jra", "odds_timeseries", "odds", pool="WIN"),
            supply("jravan", "odds_timeseries", "odds", pool="WIN"),
        ], [909]),
    }))  # fmt: skip

    seen = record_reads(monkeypatch)
    cov = cv.coverage(inputs)

    held = [p for p in seen if HOLDOUT_A.isoformat() in p or HOLDOUT_B.isoformat() in p]
    assert not held, f"holdout day files were opened: {held}"
    jravan = entry(cov, "board_at_t", "jravan")
    # WIN rows of the three development days: 3 + 4 + 5, priced 3/3 in 2025 and 7/9 in 2026
    got = (jravan.first_date, jravan.last_date, jravan.rows, jravan.files)
    assert got == (days[0], DEV_LATE, 12, 3), got
    assert dict(jravan.share_by_year) == pytest.approx({2025: 1.0, 2026: 7 / 9})
    assert list(jravan.gap_years) == []
    jra = entry(cov, "board_at_t", "jra")
    got = (jra.first_date, jra.last_date, jra.rows, jra.files)
    assert got == (date(2026, 11, 21), date(2026, 11, 21), 2, 1), got
    assert dict(jra.share_by_year) == pytest.approx({2026: 1.0})


def test_masters_are_counted_by_rows_only(data: Path, tmp_path: Path) -> None:
    horses("jra", [f"Horse {i}" for i in range(5)])
    jockeys(6)
    cov = run_coverage(tmp_path, {
        "horse": an_input(["C2"], "feature_at_t", [supply("jra", "horses", "name")], [910]),
        "jockey": an_input(["C2"], "feature_at_t",
                           [supply("jravan", "jockeys", "jockey_code")], [911]),
    })  # fmt: skip
    for name, source, rows in (("horse", "jra", 5), ("jockey", "jravan", 6)):
        e = entry(cov, name, source)
        assert (e.rows, e.first_date, e.last_date) == (rows, None, None), (name, e)
        assert not list(e.gap_years), (name, e.gap_years)


def test_a_dumped_family_is_reported_by_files_and_rows_with_year_spans(
    data: Path, tmp_path: Path
) -> None:
    dump_file("RACE/JG", "2019", 3)
    dump_file("RACE/JG", "2021", 5)
    dump_file("TOKU/TK", "all", 4)
    (paths.interim_dir("jravan") / "records" / "_dump.json").write_text("{}", encoding="utf-8")
    cov = run_coverage(tmp_path, {
        "exclusions": an_input(["C0"], "feature_at_t",
                               [supply("jravan", "records/RACE/JG", "_record")], [912]),
        "special": an_input(["C2"], "feature_at_t",
                            [supply("jravan", "records/TOKU/TK", "_record")], [913]),
    })  # fmt: skip
    jg = entry(cov, "exclusions", "jravan")
    assert (jg.files, jg.rows) == (2, 8)
    assert (jg.first_date.year, jg.last_date.year) == (2019, 2021)
    tk = entry(cov, "special", "jravan")
    assert (tk.files, tk.rows, tk.first_date, tk.last_date) == (1, 4, None, None)


# Development and holdout race days on and after the holdout start, under the committed seed
DEV_DAYS = (date(2024, 6, 1), date(2025, 12, 27), date(2026, 5, 16), DEV_LATE, date(2026, 12, 5),
            date(2027, 1, 9))  # fmt: skip
HOLDOUT_DAYS = (HOLDOUT_A, HOLDOUT_B, date(2026, 12, 6), date(2027, 1, 16), date(2027, 1, 17))


def assert_holdout_split() -> None:
    assert not any(is_holdout_day("japan", d) for d in DEV_DAYS), DEV_DAYS
    assert all(is_holdout_day("japan", d) for d in HOLDOUT_DAYS), HOLDOUT_DAYS


DIVIDENDS = pa.schema(
    [("jurisdiction", pa.string()), ("race_id", pa.string()), ("pool", pa.string()),
     ("combination", pa.string()), ("dividend", pa.float64()), ("is_refund", pa.bool_())]
)  # fmt: skip
SECTIONALS = pa.schema(
    [("jurisdiction", pa.string()), ("race_id", pa.string()), ("horse_id", pa.string()),
     ("horse_no", pa.int64()), ("section_no", pa.int64()), ("section_time_s", pa.float64())]
)  # fmt: skip


def dividends_of(day: date, race_no: int, pools: list[str]) -> list[dict]:
    return [
        {"jurisdiction": "jra", "race_id": race_id(day, race_no), "pool": pool,
         "combination": str(k + 1), "dividend": 110.0 + 10 * k, "is_refund": False}
        for k, pool in enumerate(pools)
    ]  # fmt: skip


def sectionals_of(day: date, race_no: int, horses_run: int, sections: int) -> list[dict]:
    return [
        {"jurisdiction": "jra", "race_id": race_id(day, race_no), "horse_id": f"{h:010d}",
         "horse_no": h, "section_no": s, "section_time_s": 12.0 + s / 10}
        for h in range(1, horses_run + 1)
        for s in range(1, sections + 1)
    ]  # fmt: skip


def races_of(source: str, race_ids: set[str]) -> Path:
    """The store's races for `race_ids`, holdout days included, as the builders write them."""
    rows = [
        {"race_id": r, "race_date": date.fromisoformat(r.split(":")[1]), "going": None,
         "rail_setting": None}
        for r in sorted(race_ids)
    ]  # fmt: skip
    return store_table(source, "races", RACES, rows)


def test_a_per_race_table_never_counts_a_holdout_race(data: Path, tmp_path: Path) -> None:
    """A per-race table without `race_date` (dividends, and sectionals, counted by rows) keeps
    the holdout races' rows in the store; the audit counts only the development races' rows,
    filtered or not. A master with no race key still counts every row."""
    assert_holdout_split()
    jra_div = [
        *dividends_of(date(2025, 12, 27), 1, ["WIN", "PLACE", "PLACE"]),
        *dividends_of(DEV_LATE, 1, ["WIN", "PLACE", "PLACE", "PLACE"]),
        *dividends_of(DEV_LATE, 2, ["WIN", "QUINELLA"]),
        *dividends_of(date(2026, 12, 5), 11, ["WIN", "WIN", "PLACE"]),  # a dead heat
        *dividends_of(HOLDOUT_A, 1, ["WIN", "PLACE", "PLACE", "PLACE", "QUINELLA"]),
        *dividends_of(HOLDOUT_B, 3, ["WIN", "WIN", "PLACE"]),
        *dividends_of(date(2026, 12, 6), 12, ["WIN", "PLACE", *["QUINELLA_PLACE"] * 3]),
    ]
    jravan_div = [
        *dividends_of(date(2024, 6, 1), 3, ["WIN", "PLACE"]),
        *dividends_of(date(2027, 1, 9), 5, ["WIN"]),
        *dividends_of(HOLDOUT_B, 5, ["WIN", "PLACE", "PLACE", "PLACE", "QUINELLA", "TRIO"]),
        *dividends_of(date(2027, 1, 16), 2, ["WIN", "PLACE"]),
    ]
    jra_sections = [*sectionals_of(DEV_LATE, 1, 1, 2), *sectionals_of(HOLDOUT_A, 1, 2, 2)]
    store_table("jra", "dividends", DIVIDENDS, jra_div)
    store_table("jravan", "dividends", DIVIDENDS, jravan_div)
    store_table("jra", "sectionals", SECTIONALS, jra_sections)
    races_of("jra", {r["race_id"] for r in (*jra_div, *jra_sections)})
    races_of("jravan", {r["race_id"] for r in jravan_div})
    horses("jra", [f"Horse {i}" for i in range(5)])
    jockeys(6)
    cov = run_coverage(tmp_path, {
        "dividends": an_input(["C5"], "label", [supply("jra", "dividends", "dividend"),
                                                supply("jravan", "dividends", "dividend")], [926]),
        "win_dividends": an_input(["C5"], "label",
                                  [supply("jra", "dividends", "dividend", pool="WIN")], [927]),
        "sections": an_input(["C3"], "feature_at_t",
                             [supply("jra", "sectionals", "section_time_s")], [928]),
        "horse": an_input(["C2"], "feature_at_t", [supply("jra", "horses", "name")], [929]),
        "jockey": an_input(["C2"], "feature_at_t",
                           [supply("jravan", "jockeys", "jockey_code")], [930]),
    })  # fmt: skip
    # the development races' rows only: jra 3 + 4 + 2 + 3 of 25 (WIN 1 + 1 + 1 + 2 of 9),
    # jravan 2 + 1 of 11, sectionals 2 of 6; the masters every row
    expected = {
        ("dividends", "jra"): 12,
        ("dividends", "jravan"): 3,
        ("win_dividends", "jra"): 5,
        ("sections", "jra"): 2,
        ("horse", "jra"): 5,
        ("jockey", "jravan"): 6,
    }
    got = {(name, source): entry(cov, name, source).rows for name, source in expected}
    assert got == expected, "holdout races' rows were counted (or a master's rows were not)"
    for name, source in (("horse", "jra"), ("jockey", "jravan")):
        e = entry(cov, name, source)
        assert (e.first_date, e.last_date, list(e.gap_years)) == (None, None, []), (name, e)


RACE_KEY = ("レコード種別ID", "データ区分", "データ作成年月日", "開催年", "開催月日",
            "競馬場コード", "開催回[第N回]", "開催日目[N日目]", "レース番号")  # fmt: skip


def race_keyed_dump(family: str, stem: str, records: list[tuple[date, date]]) -> Path:
    """A dumped race-keyed family as `dump` writes it: the JV-Data race key as top-level text
    columns, then `_file` and `_record`. Each record is (race day, creation day): the creation
    day (`データ作成年月日`) is not the race day."""
    n = len(records)
    values = {
        "レコード種別ID": [family.split("/")[1]] * n,
        "データ区分": ["1"] * n,
        "データ作成年月日": [created.strftime("%Y%m%d") for _, created in records],
        "開催年": [f"{day.year:04d}" for day, _ in records],
        "開催月日": [day.strftime("%m%d") for day, _ in records],
        "競馬場コード": ["05"] * n,
        "開催回[第N回]": ["05"] * n,
        "開催日目[N日目]": ["08"] * n,
        "レース番号": [f"{i % 12 + 1:02d}" for i in range(n)],
    }
    table = pa.table(
        {
            **{key: pa.array(values[key], pa.string()) for key in RACE_KEY},
            "_file": pa.array([f"stored/{family}/{stem}.jvd"] * n, pa.string()),
            "_record": pa.array(range(1, n + 1), pa.int64()),
        }
    )
    return write_parquet(
        paths.interim_dir("jravan") / "records" / family / f"{stem}.parquet", table
    )


def test_a_dumped_family_never_counts_holdout_day_records(data: Path, tmp_path: Path) -> None:
    """The year files from the holdout start on (and an `all` file) hold holdout-day records;
    only the development days' records count, by the race key's day."""
    assert_holdout_split()
    d = date
    # (race day, creation day) x count; a development record created on a holdout day and a
    # holdout record created on a development day, so the creation day decides nothing
    race_keyed_dump("RACE/JG", "2025", [(d(2025, 12, 27), d(2025, 12, 26))] * 4)
    race_keyed_dump("RACE/JG", "2026", [
        *[(d(2026, 5, 16), d(2026, 5, 15))] * 2,
        *[(DEV_LATE, HOLDOUT_A)] * 2,
        (d(2026, 12, 5), d(2026, 12, 6)),
        *[(HOLDOUT_A, d(2026, 11, 27))] * 2,
        (HOLDOUT_B, d(2026, 11, 27)),
        (d(2026, 12, 6), d(2026, 12, 5)),
    ])  # fmt: skip
    race_keyed_dump("RACE/JG", "2027", [
        (d(2027, 1, 9), d(2027, 1, 10)),
        *[(d(2027, 1, 16), d(2027, 1, 14))] * 3,
        *[(d(2027, 1, 17), d(2027, 1, 14))] * 2,
    ])  # fmt: skip
    race_keyed_dump("MING/DM", "2026", [
        *[(DEV_LATE, d(2026, 11, 27))] * 2,
        *[(HOLDOUT_A, d(2026, 11, 27))] * 6,
    ])  # fmt: skip
    race_keyed_dump("TOKU/TK", "all", [
        (d(2019, 4, 6), d(2019, 3, 25)),
        *[(DEV_LATE, d(2026, 11, 16))] * 2,
        *[(HOLDOUT_B, d(2026, 11, 16))] * 4,
    ])  # fmt: skip
    cov = run_coverage(tmp_path, {
        "exclusions": an_input(["C0"], "feature_at_t",
                               [supply("jravan", "records/RACE/JG", "_record")], [931]),
        "mining": an_input(["C4"], "feature_at_t",
                           [supply("jravan", "records/MING/DM", "_record")], [932]),
        "special": an_input(["C2"], "feature_at_t",
                            [supply("jravan", "records/TOKU/TK", "_record")], [933]),
    })  # fmt: skip
    jg, dm, tk = (entry(cov, n, "jravan") for n in ("exclusions", "mining", "special"))
    # JG: 4 (2025) + 5 of 9 (2026) + 1 of 6 (2027); DM: 2 of 8; TK: 3 of 7
    got = {"JG": jg.rows, "DM": dm.rows, "TK": tk.rows}
    assert got == {"JG": 10, "DM": 2, "TK": 3}, f"holdout-day records were counted: {got}"
    assert (jg.files, jg.first_date.year, jg.last_date.year) == (3, 2025, 2027), jg
    assert (dm.files, dm.first_date.year, dm.last_date.year) == (1, 2026, 2026), dm
    assert tk.files == 1, tk


def test_a_supply_whose_table_is_missing_has_no_rows(data: Path, tmp_path: Path) -> None:
    cov = run_coverage(tmp_path, {
        "a": an_input(["C2"], "label", [supply("jra", "runners", "win_odds")], [914]),
        "b": an_input(["C1"], "feature_at_t",
                      [supply("jravan", "odds_timeseries", "odds", pool="WIN")], [915]),
        "c": an_input(["C2"], "feature_at_t", [supply("jravan", "jockeys", "jockey_code")], [916]),
        "d": an_input(["C0"], "feature_at_t",
                      [supply("jravan", "records/RACE/JG", "_record")], [917]),
    })  # fmt: skip
    for name in "abcd":
        e = entry(cov, name)
        assert (e.rows, e.first_date, e.last_date) == (0, None, None), (name, e)


# ------------------------------------------------------------------------------------------------
# The committed summary and the private report

SECRET_NAME = "ZephyrQuillon"  # a made-up horse name: a table value
SECRET_ODDS = 987.6


def test_the_committed_summary_holds_no_value_from_a_table(data: Path, tmp_path: Path) -> None:
    day = date(2013, 6, 1)
    store_table("jra", "runners", RUNNERS,
                runners_on(day, [SECRET_ODDS, 4.2], [512, 498], name=SECRET_NAME))  # fmt: skip
    horses("jra", [SECRET_NAME, "Other Horse"])
    store_table("jravan", "final_odds", odds.FINAL_ODDS_SCHEMA,
                [odds_row(date(2016, 5, 1), "WIN", "1", SECRET_ODDS)])  # fmt: skip
    board_day("jra", date(2026, 11, 21), [board_row(date(2026, 11, 21), "WIN", "1", SECRET_ODDS)])
    cov = run_coverage(tmp_path, {
        "win_price": an_input(["C4"], "label", [supply("jra", "runners", "win_odds"),
                                                supply("jravan", "final_odds", "odds",
                                                       pool="WIN")], [918]),
        "horse_name": an_input(["C2"], "feature_at_t",
                               [supply("jra", "runners", "horse_name"),
                                supply("jra", "horses", "name")], [919]),
        "board": an_input(["C0"], "feature_at_t",
                          [supply("jra", "odds_timeseries", "odds", pool="WIN")], [920]),
    })  # fmt: skip
    summary = cv.render_summary(cov)
    assert "2013-06-01" in summary  # the jra figures are there
    for value in (SECRET_NAME, f"{SECRET_ODDS}", "987.60", race_id(day), VENUE):
        assert value not in summary, f"{value!r} from a table is in the committed summary"


def write_licensed_stores() -> None:
    """jravan final_odds (1,234 WIN rows: 2016-02-13 to 2019-10-26, 2017 priced 0.25, no 2018),
    jravan jockeys (389 rows), the dumped family RACE/JG (2015: 211, 2017: 158 rows), the JRDB
    table jrdb_pre_race in the jra store (563 rows, 2020-03-14 to 2020-09-05), and jra runners
    (12 rows, 2011-01-08 to 2012-12-22)."""
    rows = []
    for day, n, priced in (
        (date(2016, 2, 13), 600, 600),
        (date(2016, 6, 18), 34, 34),
        (date(2017, 5, 20), 400, 100),
        (date(2019, 10, 26), 200, 200),
    ):
        rows += [odds_row(day, "WIN", str(i), 3.0 if i < priced else None) for i in range(n)]
    rows += [odds_row(date(2016, 2, 13), "PLACE", str(i), 1.4) for i in range(3)]
    store_table("jravan", "final_odds", odds.FINAL_ODDS_SCHEMA, rows)
    jockeys(389)
    dump_file("RACE/JG", "2015", 211)
    dump_file("RACE/JG", "2017", 158)
    jrdb = [
        {"jurisdiction": "jra", "race_id": race_id(day, 1), "race_date": day, "horse_no": i + 1}
        for day, n in ((date(2020, 3, 14), 300), (date(2020, 9, 5), 263))
        for i in range(n)
    ]
    store_table("jra", "jrdb_pre_race", JRDB_PRE_RACE, jrdb)
    store_table("jra", "runners", RUNNERS, [
        *runners_on(date(2011, 1, 8), [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        *runners_on(date(2012, 12, 22), [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
    ])  # fmt: skip


LICENSED_INPUTS = {
    "final_win": an_input(["C4"], "label",
                          [supply("jravan", "final_odds", "odds", pool="WIN")], [921]),
    "jockey": an_input(["C2"], "feature_at_t", [supply("jravan", "jockeys", "jockey_code")], [922]),
    "exclusions": an_input(["C0"], "feature_at_t",
                           [supply("jravan", "records/RACE/JG", "_record")], [923]),
    "jrdb_card": an_input(["C4"], "feature_at_t",
                          [supply("jra", "jrdb_pre_race", "horse_no")], [924]),
    "past_price": an_input(["C2"], "feature_at_t", [supply("jra", "runners", "win_odds")], [925]),
}  # fmt: skip
# every figure of a licensed supply: dates, counts (either spelling), the 2017 share, gap years
LICENSED_FIGURES = ("2016-02-13", "2019-10-26", "2020-03-14", "2020-09-05", "1234", "1,234",
                    "389", "369", "211", "158", "563", "0.25", "25%", "25.0%")  # fmt: skip
LICENSED_YEARS = (2015, 2016, 2017, 2018, 2019, 2020)


def test_the_committed_summary_withholds_licensed_source_figures(
    data: Path, tmp_path: Path
) -> None:
    write_licensed_stores()
    cov = run_coverage(tmp_path, LICENSED_INPUTS)
    final_win = entry(cov, "final_win", "jravan")
    assert (final_win.rows, sorted(final_win.gap_years)) == (1234, [2017, 2018])
    summary, report = cv.render_summary(cov), cv.render_report(cov)

    leaked = [f for f in LICENSED_FIGURES if f in summary]
    leaked += [str(y) for y in LICENSED_YEARS if year_token(y).search(summary)]
    assert not leaked, f"licensed figures in the committed summary: {leaked}"
    assert summary.count(WITHHELD) >= 4, "each licensed supply shows the withheld marker"
    assert "2011-01-08" in summary and "2012-12-22" in summary  # netkeiba aggregates may show

    for figure in ("2016-02-13", "2019-10-26", "2020-03-14", "2020-09-05", "389", "369", "563"):
        assert figure in report, f"{figure} missing from the private report"
    assert "1234" in report or "1,234" in report
    assert year_token(2018).search(report), "the jravan gap year is in the private report"


def test_every_gap_year_in_the_summary_is_listed_with_its_ticket(
    data: Path, tmp_path: Path
) -> None:
    write_gap_races()
    cov = run_coverage(tmp_path, GAP_INPUTS)
    lines = cv.render_summary(cov).splitlines()
    for name, gaps in GAPS.items():
        tickets = [ticket_token(t) for t in GAP_INPUTS[name]["tickets"]]
        for year in gaps:
            assert any(
                year_token(year).search(line) and any(t.search(line) for t in tickets)
                for line in lines
            ), f"gap year {year} of {name} is not listed with one of its tickets"


def test_the_private_report_is_refused_inside_the_repo(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_span_runners()
    inputs_path = write_inputs(tmp_path / "model_inputs.yaml", SPAN_INPUTS)
    cov = cv.coverage(cv.load_inputs(inputs_path))
    assert "jravan" in cov.sources and "jra" in cov.sources, cov.sources

    target = REPO_ROOT / f".test-data-coverage-{uuid.uuid4().hex[:12]}" / REPORT_NAME
    try:
        with pytest.raises(LicensedOutputInRepo):
            write_report(target, cv.render_report(cov), sources=cov.sources)
        assert not target.parent.exists()
    finally:
        shutil.rmtree(target.parent, ignore_errors=True)

    # through the command: the reports folder stands in for the repository
    monkeypatch.setattr(cv, "INPUTS_PATH", inputs_path)
    reports = paths.reports_dir()
    reports.mkdir(parents=True)
    monkeypatch.setattr(env, "repo_root", lambda: reports)
    result = CliRunner().invoke(app, ["japan", "coverage"])
    assert result.exit_code != 0, result.output
    assert list(reports.iterdir()) == [], "the refused report left a file"


def test_the_command_writes_the_private_report_and_with_write_doc_the_doc(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_span_runners()
    inputs_path = write_inputs(tmp_path / "model_inputs.yaml", SPAN_INPUTS)
    report = paths.reports_dir() / REPORT_NAME
    # a stand-in repository holding the doc: the committed view must be writable inside it
    repo = tmp_path / "repo"
    doc = repo / "docs" / REPORT_NAME
    doc.parent.mkdir(parents=True)
    monkeypatch.setattr(env, "repo_root", lambda: repo)
    monkeypatch.setattr(cv, "INPUTS_PATH", inputs_path)
    monkeypatch.setattr(cv, "DOC_PATH", doc)

    result = CliRunner().invoke(app, ["japan", "coverage"])
    assert result.exit_code == 0, result.output
    assert report.is_file() and not doc.exists()
    private = report.read_text(encoding="utf-8")
    assert "2020-01-05" in private and "2021-06-06" in private  # the jravan span, private only

    result = CliRunner().invoke(app, ["japan", "coverage", "--write-doc"])
    assert result.exit_code == 0, result.output
    text = doc.read_text(encoding="utf-8")
    cov = cv.coverage(cv.load_inputs(inputs_path))
    assert text.strip() == cv.render_summary(cov).strip()
    assert "2012-03-03" in text and "2020-01-05" not in text and WITHHELD in text
    for name in SPAN_INPUTS:
        assert name in text


# ------------------------------------------------------------------------------------------------
# The committed doc (generated on the owner's machine from the real stores)


def name_token(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])")


def test_the_committed_doc_lists_every_input_of_the_yaml() -> None:
    assert COMMITTED_DOC.is_file(), f"{COMMITTED_DOC} is not generated yet"
    text = COMMITTED_DOC.read_text(encoding="utf-8")
    missing = [inp.name for inp in cv.load_inputs() if not name_token(inp.name).search(text)]
    assert not missing, f"inputs missing from docs/data-coverage.md: {missing}"


def test_the_committed_doc_withholds_every_licensed_supply() -> None:
    text = COMMITTED_DOC.read_text(encoding="utf-8")
    licensed = [
        inp.name
        for inp in cv.load_inputs()
        if any(is_withheld((s.source, s.table)) for s in inp.supplies)
    ]
    assert licensed, "no input of the committed yaml reads a licensed source"
    assert text.count(WITHHELD) >= len(licensed), "a licensed supply shows figures in the doc"
