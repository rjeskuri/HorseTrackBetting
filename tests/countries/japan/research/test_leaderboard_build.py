"""Ticket #452: the Japan leaderboard built from the registry and the candidate-cell files.

Contract pinned here, module ``horsetrack.countries.japan.research.leaderboard_build``:

- ``candidate_rows(directory: Path) -> list[LeaderboardRow]``: every cell of every candidate-cell
  file in ``directory``, a ``*.json`` file holding one JSON array of cells (the studies write it
  indented) or a ``*.jsonl`` file holding one cell per line; an empty array gives no row. A cell
  in #430's schema gives one row: option ``'<family_id>/<cell_id>'``, kind 'cell', segment the
  cell_id, rule ``'<study>: flat bets in <cell_id>'``, pool and payout rate the cell's,
  return_per_100 100 x mean_return, se_per_100 100 x se_mean_return, n the n_bets in 'bets',
  n_meetings the cell's, selection the cell's 'selection' (default 'unknown'), sources the cell's
  'sources' (default ('jravan',): a cell without the key is withheld), ticket the cell's
  'ticket' (default None), no shrunk value (``shrink`` sets it, never a reader). An empty cell
  (mean_return null) is an unmeasured row, never dropped.
- ``build_rows(*, registry=None, candidate_dir=None) -> list[LeaderboardRow]``: #451's
  ``legacy_rows(registry)`` as they are, plus every candidate row whose option id the registry
  does not give (a registry row wins on the same id, measured or not); a candidate row's
  variants searched is its family's size in that registry, 1 when the family has no run.
  Defaults ``default_registry('japan')`` and ``$HORSETRACK_DATA_DIR/reports/candidate-cells/``;
  a missing directory gives no candidate row.
- ``horsetrack japan leaderboard [--public]``: from ``build_rows()``, writes
  ``<reports dir>/leaderboard-japan.md`` (``write_leaderboard``, public=False) and
  ``leaderboard-japan.json`` (``to_json``); with ``--public`` also
  ``<env.repo_root()>/docs/leaderboard/japan.md`` (``write_leaderboard``, public=True), the path
  resolved through ``env.repo_root()`` when the command runs. It prints the counts of rows,
  measured, unmeasured and withheld rows, each once next to its label ('8 rows' or 'rows: 8'),
  and never a figure.

Every registry is a temp file under ``tmp_path``; every figure is made up. The command tests point
``env.repo_root`` at a stand-in repository and check the real ``docs/leaderboard/japan.md`` is
untouched. The committed public file and the real registry are read by the last two tests only;
neither prints a figure.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

from horsetrack.core import env, paths
from horsetrack.core.eval.leaderboard import (
    LeaderboardRow,
    from_json,
    public_view,
    rank,
    render_markdown,
)
from horsetrack.core.eval.registry import FoldMetric, Registry, RunRecord
from horsetrack.core.eval.reports import is_withheld

REPO_ROOT = Path(__file__).resolve().parents[4]
REAL_COUNTRIES = REPO_ROOT / "src" / "horsetrack" / "countries"
PUBLIC_DOC = REPO_ROOT / "docs" / "leaderboard" / "japan.md"
GIT_SHA = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c"
BUILD_HASH = hashlib.sha256(b"synthetic build").hexdigest()

CONVENTION = "japan-n0-synthetic-scan"
FAMILY_L = "japan-family-l-exotic-pool-choice"
L_VARIANT = "L1:TRIO:threshold_20:final_look_ahead"
META = "japan-c4-meta-final-win"
POOL_SCAN = "japan-pool-scan-v1"

WIN_BLOCK = {
    "kind": "cell",
    "pool": "WIN",
    "rule": "flat Win bets",
    "selection": "look_ahead",
    "payout_rate": 0.8,
    "ci_level": 0.95,
    "n_unit": "bets",
}

# The ticket's example cell, in #430's schema.
TICKET_CELL = {
    "study": "N1",
    "family_id": POOL_SCAN,
    "cell_id": "WIN|class:maiden|band:5-10",
    "pool": "WIN",
    "class_group": "maiden",
    "odds_band": "5-10",
    "payout_rate": 0.8,
    "n_bets": 1200,
    "n_race_days": 300,
    "n_meetings": 90,
    "mean_return": 0.84,
    "se_mean_return": 0.03,
    "shrunk_edge": None,
    "mean_odds": 7.1,
}
TICKET_ID = f"{POOL_SCAN}/WIN|class:maiden|band:5-10"

# A public cell of another study, its family not in any registry here.
QUINELLA_CELL = {
    "study": "N8",
    "family_id": "japan-n8-spread-scan-quinella",
    "cell_id": "QUINELLA|box:1-2-3|class:all",
    "pool": "QUINELLA",
    "class_group": None,
    "odds_band": "all",
    "payout_rate": 0.775,
    "n_bets": 4321,
    "n_race_days": 640,
    "n_meetings": 210,
    "mean_return": 0.7125,
    "se_mean_return": 0.021,
    "shrunk_edge": None,
    "mean_odds": 18.4,
    "sources": ["netkeiba"],
}
QUINELLA_ID = "japan-n8-spread-scan-quinella/QUINELLA|box:1-2-3|class:all"

# An empty cell (no bets): listed, with no return.
EMPTY_CELL = {
    "study": "N1",
    "family_id": POOL_SCAN,
    "cell_id": "TIERCE|class:newcomer|band:100+",
    "pool": "TIERCE",
    "class_group": "newcomer",
    "odds_band": "100+",
    "payout_rate": 0.725,
    "n_bets": 0,
    "n_race_days": 0,
    "n_meetings": 0,
    "mean_return": None,
    "se_mean_return": None,
    "shrunk_edge": None,
    "mean_odds": None,
    "sources": ["netkeiba"],
}
EMPTY_ID = f"{POOL_SCAN}/TIERCE|class:newcomer|band:100+"


# --------------------------------------------------------------------------- helpers


def build():
    from horsetrack.countries.japan.research import leaderboard_build

    return leaderboard_build


def legacy_rows(registry: Registry) -> list[LeaderboardRow]:
    from horsetrack.countries.japan.research.leaderboard_backfill import legacy_rows as rows

    return rows(registry)


def snapshot_logs() -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in REAL_COUNTRIES.glob("*/holdout_log.jsonl")}


@pytest.fixture
def data(tmp_path, monkeypatch) -> Path:
    """A temp data directory as HORSETRACK_DATA_DIR; the real holdout logs stay untouched."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(root))
    logs = snapshot_logs()
    yield root
    assert snapshot_logs() == logs, "a test wrote to a real holdout_log.jsonl"


def open_registry(path: Path, tmp_path: Path) -> Registry:
    path.parent.mkdir(parents=True, exist_ok=True)
    log = tmp_path / "countries" / "japan" / "holdout_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    return Registry(path, holdout_log=log)


@pytest.fixture
def reg(data, tmp_path) -> Registry:
    """An empty temp registry away from the default path."""
    return open_registry(tmp_path / "elsewhere" / "japan.jsonl", tmp_path)


@pytest.fixture
def stand_in_repo(tmp_path, monkeypatch) -> Path:
    """``env.repo_root`` points at an empty stand-in; the real public file must stay as it is."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(env, "repo_root", lambda: repo)
    folder_existed = PUBLIC_DOC.parent.is_dir()
    before = PUBLIC_DOC.read_bytes() if PUBLIC_DOC.is_file() else None
    yield repo
    after = PUBLIC_DOC.read_bytes() if PUBLIC_DOC.is_file() else None
    if after != before:
        if before is None:
            PUBLIC_DOC.unlink()
            if not folder_existed and not any(PUBLIC_DOC.parent.iterdir()):
                PUBLIC_DOC.parent.rmdir()
        else:
            PUBLIC_DOC.write_bytes(before)
        pytest.fail(
            "the command wrote the real repository's docs/leaderboard/japan.md (restored): resolve "
            "the public path through env.repo_root() when the command runs"
        )


def dev(metric: str, value: float, lo=None, hi=None, n: int = 0) -> FoldMetric:
    return FoldMetric(fold="development", metric=metric, value=value, lo=lo, hi=hi, n=n)


def record(**overrides) -> RunRecord:
    fields = {
        "hypothesis_id": "E1",
        "family_id": CONVENTION,
        "variant": "v1",
        "ticket": 452,
        "git_sha": GIT_SHA,
        "data_builds": {"results": BUILD_HASH},
        "sources": ("netkeiba",),
        "fold_scheme": "development-only",
        "components": {"rule": "synthetic"},
        "params": {},
        "metrics": (),
        "status": "scored",
    }
    fields.update(overrides)
    return RunRecord(**fields)


def convention(variant: str, metrics=(), *, family=CONVENTION, status="scored", **kw) -> RunRecord:
    """A run of a family that follows ``docs/specs/leaderboard.md``'s convention."""
    return record(
        family_id=family,
        variant=variant,
        params={"leaderboard": dict(WIN_BLOCK)},
        metrics=tuple(metrics),
        status=status,
        **kw,
    )


def family_l(rule: float, comparator: float, *, n: int, meetings: int) -> RunRecord:
    """One Family L (#81) run, a legacy family read by #451's adapter (made-up values)."""
    diff = rule - comparator
    return record(
        hypothesis_id="L1",
        family_id=FAMILY_L,
        variant=L_VARIANT,
        ticket=81,
        params={"pool": "TRIO", "stake_yen": 100, "threshold": 20},
        metrics=(
            dev("paired_return_difference", diff, diff - 0.01, diff + 0.01, n),
            dev("rule_mean_return", rule, None, None, n),
            dev("comparator_mean_return", comparator, None, None, n),
            dev("meetings", float(meetings), None, None, n),
        ),
    )


def meta_withheld() -> RunRecord:
    """A legacy model family on a licensed source: unmeasured, kind 'model', withheld."""
    return record(
        family_id=META,
        variant="meta-a",
        sources=("jra", "jrdb"),
        metrics=(dev("log_loss_gain", 0.004321, 0.001234, 0.007654, 3000),),
    )


def cell(base: dict = TICKET_CELL, **overrides) -> dict:
    return {**base, **overrides}


def write_cells(directory: Path, name: str, cells: list[dict], *, lines: bool = False) -> Path:
    """A candidate-cell file: an indented JSON array (as the studies write it) or JSON lines."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if lines:
        text = "".join(json.dumps(c) + "\n" for c in cells)
    else:
        text = json.dumps(cells, indent=1) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def unique(rows: list[LeaderboardRow]) -> dict[str, LeaderboardRow]:
    assert isinstance(rows, list)
    assert all(isinstance(r, LeaderboardRow) for r in rows)
    ids = [r.option_id for r in rows]
    assert len(ids) == len(set(ids)), f"an option id appears twice: {sorted(ids)}"
    return {r.option_id: r for r in rows}


_PIPE = re.compile(r"(?<!\\)\|")


def tables(text: str) -> list[dict[str, str]]:
    """Every row of the Markdown tables before the public withheld list, by column name, with
    escaped pipes restored."""
    found: list[dict[str, str]] = []
    columns: list[str] | None = None
    lines = text.split("## Withheld (D21)", 1)[0].splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("|"):
            columns = None
        else:
            cells = [c.strip().replace("\\|", "|") for c in _PIPE.split(line[1:-1])]
            if columns is None:
                columns = cells
                i += 1  # the alignment line
            else:
                found.append(dict(zip(columns, cells, strict=True)))
        i += 1
    return found


def withheld_list(text: str) -> set[str]:
    """The option ids of the public text's withheld list."""
    _, sep, tail = text.partition("## Withheld (D21)")
    assert sep, "the public text has no withheld (D21) section"
    return set(re.findall(r"^- `(.+)` \(family `.+`\)$", tail, flags=re.MULTILINE))


def title_of(text: str) -> str:
    first = text.split("\n", 1)[0]
    assert first.startswith("# "), f"the leaderboard does not start with a title: {first!r}"
    return first[2:]


_COUNT_STYLES = (r"(\d+)\s+{label}\b", r"\b{label}\b\s*[:=]?\s*(\d+)")
COUNT_LABELS = ("rows", "measured", "unmeasured", "withheld")


def printed_counts(output: str) -> dict[str, int] | None:
    """The four counts, each printed once next to its label in one consistent style."""
    for style in _COUNT_STYLES:
        found = {}
        for label in COUNT_LABELS:
            values = set(re.findall(style.format(label=label), output, flags=re.IGNORECASE))
            if len(values) != 1:
                break
            found[label] = int(values.pop())
        else:
            return found
    return None


def without_paths(output: str, tmp_path: Path) -> str:
    text = output
    for prefix in sorted({str(tmp_path), str(tmp_path.resolve()), os.path.realpath(tmp_path)}):
        text = text.replace(prefix, "<tmp>")
    return text


def invoke(*args: str):
    from typer.testing import CliRunner

    from horsetrack.cli import app

    return CliRunner().invoke(app, ["japan", "leaderboard", *args])


# --------------------------------------------------------------------------- build_rows


def test_build_rows_lists_every_variant_and_every_candidate_cell(reg, tmp_path):
    reg.append(
        convention(
            "flat:all",
            [
                dev("return_per_100", 78.5, 74.0, 83.0, 2400),
                dev("se_per_100", 2.7),
                dev("n_meetings", 150.0),
            ],
        )
    )
    reg.append(
        convention(
            "flat:by-class",
            [
                dev("return_per_100[class:maiden]", 81.0, 77.0, 85.0, 600),
                dev("return_per_100[class:open]", 69.5, 65.0, 74.0, 500),
            ],
        )
    )
    reg.append(convention("flat:failed", status="failed"))
    reg.append(family_l(0.6125, 0.55, n=1000, meetings=80))
    reg.append(meta_withheld())
    # the pool scan's family has two variants in the registry: its cells search two
    for variant in ("scan-a", "scan-b"):
        reg.append(
            record(
                hypothesis_id="P1",
                family_id=POOL_SCAN,
                variant=variant,
                ticket=430,
                metrics=(dev("cells_scanned", 12.0),),
            )
        )
    cells_dir = tmp_path / "cells"
    write_cells(cells_dir, "cells.json", [cell(), cell(QUINELLA_CELL)])

    rows = unique(build().build_rows(registry=reg, candidate_dir=cells_dir))

    registry_ids = {
        f"{CONVENTION}/flat:all",
        f"{CONVENTION}/flat:by-class[class:maiden]",
        f"{CONVENTION}/flat:by-class[class:open]",
        f"{CONVENTION}/flat:failed",
        f"{FAMILY_L}/{L_VARIANT}",
        f"{FAMILY_L}/{L_VARIANT}/comparator",
        f"{META}/meta-a",
        f"{POOL_SCAN}/scan-a",
        f"{POOL_SCAN}/scan-b",
    }
    assert set(rows) == registry_ids | {TICKET_ID, QUINELLA_ID}
    # the registry's rows are #451's legacy rows as they are
    legacy = {r.option_id: r for r in legacy_rows(reg)}
    assert set(legacy) == registry_ids
    assert {i: rows[i] for i in legacy} == legacy
    # spot checks written out: the adapters, the legacy kinds and the convention all apply
    assert rows[f"{FAMILY_L}/{L_VARIANT}"].return_per_100 == pytest.approx(61.25)
    assert rows[f"{FAMILY_L}/{L_VARIANT}/comparator"].return_per_100 == pytest.approx(55.0)
    assert rows[f"{META}/meta-a"].kind == "model" and rows[f"{META}/meta-a"].withheld
    assert rows[f"{CONVENTION}/flat:all"].return_per_100 == pytest.approx(78.5)
    assert not rows[f"{CONVENTION}/flat:failed"].measured
    # the cells: their figures, and their family's size in the registry as variants searched
    assert rows[TICKET_ID].kind == "cell" and rows[QUINELLA_ID].kind == "cell"
    assert rows[TICKET_ID].return_per_100 == pytest.approx(84.0)
    assert rows[QUINELLA_ID].return_per_100 == pytest.approx(71.25)
    assert rows[TICKET_ID].variants_searched == 2
    assert rows[QUINELLA_ID].variants_searched == 1
    rank(list(rows.values()))


TABLE_CELLS = [
    pytest.param(
        cell(),
        {
            "option_id": TICKET_ID,
            "family_id": POOL_SCAN,
            "ticket": None,
            "pool": "WIN",
            "segment": "WIN|class:maiden|band:5-10",
            "rule": "N1: flat bets in WIN|class:maiden|band:5-10",
            "selection": "unknown",
            "sources": ("jravan",),
            "return_per_100": 84.0,
            "se_per_100": 3.0,
            "n": 1200,
            "n_meetings": 90,
            "payout_rate": 0.8,
            "measured": True,
            "withheld": True,
        },
        id="the-ticket-example",
    ),
    pytest.param(
        {
            "study": "cross-pool/final-price-diagnostic",
            "family_id": "japan-cross-pool-t2-v1",
            "cell_id": "FORECAST|value|band:20-50",
            "pool": "FORECAST",
            "class_group": None,
            "odds_band": "20-50",
            "payout_rate": 0.725,
            "n_bets": 333,
            "n_race_days": 120,
            "n_meetings": 41,
            "mean_return": 1.105,
            "se_mean_return": 0.2,
            "shrunk_edge": 0.02,
            "mean_odds": 31.5,
            "selection": "look_ahead",
            "sources": ["jra", "netkeiba"],
            "ticket": 435,
        },
        {
            "option_id": "japan-cross-pool-t2-v1/FORECAST|value|band:20-50",
            "family_id": "japan-cross-pool-t2-v1",
            "ticket": 435,
            "pool": "FORECAST",
            "segment": "FORECAST|value|band:20-50",
            "rule": "cross-pool/final-price-diagnostic: flat bets in FORECAST|value|band:20-50",
            "selection": "look_ahead",
            "sources": ("jra", "netkeiba"),
            "return_per_100": 110.5,
            "se_per_100": 20.0,
            "n": 333,
            "n_meetings": 41,
            "payout_rate": 0.725,
            "measured": True,
            "withheld": False,
        },
        id="optional-keys-given",
    ),
    pytest.param(
        cell(EMPTY_CELL),
        {
            "option_id": EMPTY_ID,
            "family_id": POOL_SCAN,
            "ticket": None,
            "pool": "TIERCE",
            "segment": "TIERCE|class:newcomer|band:100+",
            "rule": "N1: flat bets in TIERCE|class:newcomer|band:100+",
            "selection": "unknown",
            "sources": ("netkeiba",),
            "return_per_100": None,
            "se_per_100": None,
            "n": 0,
            "n_meetings": 0,
            "payout_rate": 0.725,
            "measured": False,
            "withheld": False,
        },
        id="an-empty-cell-is-listed-unmeasured",
    ),
]


@pytest.mark.parametrize(("given", "expected"), TABLE_CELLS)
def test_candidate_rows_follow_the_schema(data, tmp_path, given, expected):
    cells_dir = tmp_path / "cells"
    write_cells(cells_dir, "cells.json", [given])

    rows = build().candidate_rows(cells_dir)

    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, LeaderboardRow)
    got = {
        "option_id": row.option_id,
        "family_id": row.family_id,
        "ticket": row.ticket,
        "pool": row.pool,
        "segment": row.segment,
        "rule": row.rule,
        "selection": row.selection,
        "sources": row.sources,
        "return_per_100": row.return_per_100,
        "se_per_100": row.se_per_100,
        "n": row.n,
        "n_meetings": row.n_meetings,
        "payout_rate": row.payout_rate,
        "measured": row.measured,
        "withheld": row.withheld,
    }
    approx = {
        k: v if v is None or not isinstance(v, float) else pytest.approx(v)
        for k, v in expected.items()
    }
    assert got == approx
    assert row.kind == "cell"
    assert row.n_unit == "bets"
    assert row.variants_searched == 1  # no run of the family in the (empty) registry
    assert row.shrunk_per_100 is None, "a reader never sets the shrunk value"
    assert not row.descriptive


_MISSING = object()


@pytest.mark.parametrize(
    ("given", "sources", "withheld"),
    [
        (_MISSING, ("jravan",), True),
        (["netkeiba"], ("netkeiba",), False),
        (["jra", "netkeiba"], ("jra", "netkeiba"), False),
        (["JRDB"], ("JRDB",), True),
        (["netkeiba", "JRA-VAN"], ("netkeiba", "JRA-VAN"), True),
    ],
)
def test_candidate_rows_without_sources_are_withheld(data, tmp_path, given, sources, withheld):
    base = {k: v for k, v in TICKET_CELL.items() if k != "sources"}
    given_cell = base if given is _MISSING else {**base, "sources": given}
    given_cell["mean_return"] = 0.9137
    cells_dir = tmp_path / "cells"
    write_cells(cells_dir, "cells.json", [given_cell])

    rows = build().candidate_rows(cells_dir)

    assert [r.option_id for r in rows] == [TICKET_ID]
    assert rows[0].sources == sources
    assert rows[0].withheld is withheld
    public = render_markdown(rows, title="Japan leaderboard", public=True)
    assert ("91.37" in public) is not withheld
    assert (TICKET_ID in [r.option_id for r in public_view(rows)]) is not withheld


def test_candidate_files_may_be_a_json_array_or_json_lines(data, tmp_path):
    cells_dir = tmp_path / "cells"
    array = [
        cell(cell_id="WIN|class:open|band:2-5", mean_return=0.81),
        cell(cell_id="WIN|class:open|band:5-10", mean_return=0.77),
    ]
    lines = [
        cell(cell_id="PLACE|class:maiden|band:2-5", pool="PLACE", mean_return=0.88),
        cell(cell_id="PLACE|class:maiden|band:5-10", pool="PLACE", mean_return=0.69),
    ]
    write_cells(cells_dir, "scan.json", array)
    write_cells(cells_dir, "more.jsonl", lines, lines=True)
    (cells_dir / "empty.json").write_text("[]\n", encoding="utf-8")

    rows = unique(build().candidate_rows(cells_dir))

    expected = {
        f"{POOL_SCAN}/WIN|class:open|band:2-5": 81.0,
        f"{POOL_SCAN}/WIN|class:open|band:5-10": 77.0,
        f"{POOL_SCAN}/PLACE|class:maiden|band:2-5": 88.0,
        f"{POOL_SCAN}/PLACE|class:maiden|band:5-10": 69.0,
    }
    assert {i: r.return_per_100 for i, r in rows.items()} == pytest.approx(expected)
    assert rows[f"{POOL_SCAN}/PLACE|class:maiden|band:2-5"].pool == "PLACE"

    only_empty = tmp_path / "only-empty"
    only_empty.mkdir()
    (only_empty / "empty.json").write_text("[]", encoding="utf-8")
    assert build().candidate_rows(only_empty) == []


def test_a_missing_candidate_dir_gives_no_candidate_rows(data, reg, tmp_path):
    reg.append(convention("flat:all", [dev("return_per_100", 78.5, 74.0, 83.0, 2400)]))

    explicit = build().build_rows(registry=reg, candidate_dir=tmp_path / "absent")
    default = build().build_rows(registry=reg)  # $HORSETRACK_DATA_DIR/reports/candidate-cells/

    assert [r.option_id for r in explicit] == [f"{CONVENTION}/flat:all"]
    assert [r.option_id for r in default] == [f"{CONVENTION}/flat:all"]


def test_a_registry_row_wins_over_a_candidate_row_with_the_same_id(reg, tmp_path):
    # the registry measures one cell the candidate file also holds, and failed on another
    reg.append(
        convention(
            "WIN|class:maiden|band:5-10",
            [dev("return_per_100", 91.0, 88.0, 94.0, 1500)],
            family=POOL_SCAN,
        )
    )
    reg.append(convention("PLACE|class:open|band:2-5", family=POOL_SCAN, status="failed"))
    cells_dir = tmp_path / "cells"
    write_cells(
        cells_dir,
        "cells.json",
        [
            cell(),  # TICKET_ID, mean_return 0.84
            cell(cell_id="PLACE|class:open|band:2-5", pool="PLACE", mean_return=0.95),
            cell(cell_id="WIN|class:open|band:10-20", mean_return=0.73),
        ],
    )

    rows = unique(build().build_rows(registry=reg, candidate_dir=cells_dir))

    failed_id = f"{POOL_SCAN}/PLACE|class:open|band:2-5"
    kept_id = f"{POOL_SCAN}/WIN|class:open|band:10-20"
    assert set(rows) == {TICKET_ID, failed_id, kept_id}
    legacy = {r.option_id: r for r in legacy_rows(reg)}
    assert rows[TICKET_ID] == legacy[TICKET_ID]
    assert rows[TICKET_ID].return_per_100 == pytest.approx(91.0)
    assert (rows[TICKET_ID].n, rows[TICKET_ID].rule) == (1500, "flat Win bets")
    assert rows[failed_id] == legacy[failed_id]
    assert not rows[failed_id].measured and rows[failed_id].status == "failed"
    assert rows[kept_id].return_per_100 == pytest.approx(73.0)
    assert rows[kept_id].variants_searched == 2  # the family's two variants in the registry
    rank(list(rows.values()))


# --------------------------------------------------------------------------- the command


def seed_command_data(root: Path, tmp_path: Path) -> None:
    """The default registry and candidate dir under the data dir ``root``: 8 rows, 5 measured,
    3 unmeasured, 2 withheld (META/meta-a and the ticket cell)."""
    registry = open_registry(root / "registry" / "japan.jsonl", tmp_path)
    registry.append(
        convention(
            "flat:all",
            [
                dev("return_per_100", 78.5, 74.0, 83.0, 2400),
                dev("se_per_100", 2.7),
                dev("n_meetings", 150.0),
            ],
        )
    )
    registry.append(convention("flat:failed", status="failed"))
    registry.append(meta_withheld())
    registry.append(family_l(0.6125, 0.55, n=1000, meetings=80))
    write_cells(
        root / "reports" / "candidate-cells",
        "cells.json",
        [cell(), cell(QUINELLA_CELL), cell(EMPTY_CELL)],
    )


SEEDED_IDS = {
    f"{CONVENTION}/flat:all",
    f"{CONVENTION}/flat:failed",
    f"{META}/meta-a",
    f"{FAMILY_L}/{L_VARIANT}",
    f"{FAMILY_L}/{L_VARIANT}/comparator",
    TICKET_ID,
    QUINELLA_ID,
    EMPTY_ID,
}
# Every figure of the seeded rows as an integer token (n, meetings, days, and the whole part of
# each return and bound per ¥100): none may reach the command's output.
SEEDED_FIGURES = {
    "2400", "150", "1000", "80", "1200", "300", "90", "4321", "640", "210",
    "78", "74", "83", "61", "55", "84", "71",
}  # fmt: skip


def test_the_command_writes_the_private_reports_under_the_data_dir(data, tmp_path, stand_in_repo):
    seed_command_data(data, tmp_path)

    result = invoke()

    assert result.exit_code == 0, result.output
    md = data / "reports" / "leaderboard-japan.md"
    js = data / "reports" / "leaderboard-japan.json"
    assert md.is_file() and js.is_file(), sorted(p.name for p in (data / "reports").iterdir())
    assert not any(stand_in_repo.rglob("*")), "the command wrote into the repository"

    rows = from_json(js.read_text(encoding="utf-8"))
    assert set(unique(rows)) == SEEDED_IDS
    text = md.read_text(encoding="utf-8")
    # the private text is task 2's rendering of every row, withheld ones with their figures
    assert text == render_markdown(rows, title=title_of(text), public=False)
    assert "84.00" in text and f"{META}/meta-a" in text
    ranked = [(r["Rank"], r["Option"]) for r in tables(text)]
    assert ranked[:5] == [
        ("1", TICKET_ID),  # WIN, shrunk with flat:all toward their pool's mean, still above it
        ("2", f"{CONVENTION}/flat:all"),
        ("3", QUINELLA_ID),  # a pool of one keeps its raw return
        ("4", f"{FAMILY_L}/{L_VARIANT}"),  # TRIO, no se: raw
        ("5", f"{FAMILY_L}/{L_VARIANT}/comparator"),
    ]
    assert {option for r, option in ranked[5:] if r == "—"} == {
        f"{CONVENTION}/flat:failed",
        f"{META}/meta-a",
        EMPTY_ID,
    }

    out = without_paths(result.output, tmp_path)
    assert printed_counts(out) == {"rows": 8, "measured": 5, "unmeasured": 3, "withheld": 2}, out
    assert not re.search(r"\d\.\d|\d,\d{3}\b", out), f"a figure reached the output: {out!r}"
    leaked = SEEDED_FIGURES & set(re.findall(r"\d+", out))
    assert not leaked, f"figures {sorted(leaked)} reached the output: {out!r}"


def test_public_output_holds_no_withheld_figure(data, tmp_path, stand_in_repo):
    seed_command_data(data, tmp_path)
    withheld_cell = cell(
        cell_id="WIN|class:debut|band:20-50", mean_return=1.23456, n_bets=98765, n_meetings=77
    )
    withheld_id = f"{POOL_SCAN}/WIN|class:debut|band:20-50"
    write_cells(data / "reports" / "candidate-cells", "withheld.json", [withheld_cell])

    result = invoke("--public")

    assert result.exit_code == 0, result.output
    private_md = data / "reports" / "leaderboard-japan.md"
    private_js = data / "reports" / "leaderboard-japan.json"
    public_md = stand_in_repo / "docs" / "leaderboard" / "japan.md"
    assert private_md.is_file() and private_js.is_file() and public_md.is_file()
    private = private_md.read_text(encoding="utf-8")
    public = public_md.read_text(encoding="utf-8")
    assert "123.46" in private and "98765" in private, "the private text holds the figures"

    for figure in ("123.46", "98765", "84.00", "1200 bets"):
        assert figure not in public, f"withheld figure {figure!r} is in the public file"
    shown = {r["Option"] for r in tables(public)}
    assert withheld_id not in shown and TICKET_ID not in shown and f"{META}/meta-a" not in shown
    assert {withheld_id, TICKET_ID, f"{META}/meta-a"} <= withheld_list(public)
    assert "71.25" in public and QUINELLA_ID in shown, "the public rows are still shown"
    # the public file is task 2's public rendering of the private rows
    rows = from_json(private_js.read_text(encoding="utf-8"))
    assert set(unique(rows)) == SEEDED_IDS | {withheld_id}
    assert public == render_markdown(rows, title=title_of(public), public=True)

    out = without_paths(result.output, tmp_path)
    assert "98765" not in out and not re.search(r"\d\.\d", out), out


# --------------------------------------------------------------------------- the committed file


def test_the_committed_public_leaderboard_holds_public_rows_only():
    assert PUBLIC_DOC.is_file(), (
        "docs/leaderboard/japan.md is not committed: run `horsetrack japan leaderboard --public`"
    )
    text = PUBLIC_DOC.read_text(encoding="utf-8")
    got = text.splitlines()
    # the writer's preamble, as render_markdown(public=True) writes it for this title
    expected = render_markdown([], title=title_of(text), public=True).splitlines()
    end = next(i for i, line in enumerate(expected) if line.startswith("|"))
    variants = "Variants searched in total: "
    assert len(got) > end, "the committed file stops inside the preamble"
    for i, line in enumerate(expected[:end]):
        if line.startswith(variants):
            assert got[i].startswith(variants), f"line {i + 1} is not the variants total"
        else:
            assert got[i] == line, f"line {i + 1} is not the writer's preamble"

    withheld = withheld_list(text)
    for row in tables(text):
        sources = (
            () if row["Sources"] == "—" else tuple(s.strip() for s in row["Sources"].split(","))
        )
        assert not is_withheld(sources), f"{row['Option']} names a withheld source"
        assert row["Option"] not in withheld, f"{row['Option']} is both shown and withheld"


# --------------------------------------------------------------------------- the real registry


def _real_cell_ids(directory: Path) -> set[str]:
    """'<family_id>/<cell_id>' of every cell of the real candidate files that names both."""
    ids: set[str] = set()
    if not directory.is_dir():
        return ids
    for path in sorted(directory.iterdir()):
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            cells = data if isinstance(data, list) else []
        elif path.suffix == ".jsonl":
            lines = path.read_text(encoding="utf-8").splitlines()
            cells = [json.loads(line) for line in lines if line.strip()]
        else:
            continue
        for c in cells:
            if isinstance(c, dict):
                family, cell_id = c.get("family_id"), c.get("cell_id")
                if isinstance(family, str) and isinstance(cell_id, str):
                    ids.add(f"{family}/{cell_id}")
    return ids


@pytest.mark.local
def test_the_real_registry_builds():
    try:
        real = paths.data_dir() / "registry" / "japan.jsonl"
    except paths.DataDirNotSet:
        pytest.skip(f"{paths.DATA_DIR_VAR} is not set: the real registry is unavailable")
    if not real.is_file():
        pytest.skip(f"no real Japan registry at {real}")
    # read before the build: a run appended meanwhile only adds rows the build also sees
    registry = Registry(real, holdout_log=REAL_COUNTRIES / "japan" / "holdout_log.jsonl")
    families = {r.family_id for r in registry.records() if r.holdout_open is None}
    cells = _real_cell_ids(paths.reports_dir() / "candidate-cells")

    rows = build().build_rows()

    assert rows, "the real registry gave no row"
    by_id = unique(rows)
    rank(rows)
    missing_families = sorted(families - {r.family_id for r in rows})
    assert not missing_families, f"families missing from the board: {missing_families[:10]}"
    missing_cells = sorted(cells - set(by_id))
    assert not missing_cells, f"{len(missing_cells)} candidate cells missing: {missing_cells[:10]}"
    render_markdown(rows, title="Japan leaderboard", public=True)
    render_markdown(rows, title="Japan leaderboard", public=False)
