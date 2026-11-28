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
  (mean_return null) is an unmeasured row, never dropped. Without a 'selection' key, a 'variant'
  of 't2' reads as 'decision_time' and 'lookahead' as 'look_ahead' (#503's spread-scan export);
  any other or missing variant stays 'unknown'. A cell with finite 'ci_lo' and 'ci_hi' (per unit
  staked, each a one-sided bound at 0.95: the spread scan's 5% and 95% bootstrap quantiles) gives
  the row ci_lo and ci_hi 100 x them at ci_level 0.95, the leaderboard's one-sided convention
  (docs/specs/leaderboard.md); otherwise no CI.
- Option ids (D21): an option id is a function of the cell's own identity, never of the order of
  entries or files, nor of another cell's figures. A cell_id that does not recur within its
  family keeps the bare ``'<family_id>/<cell_id>'``; every cell whose (family_id, cell_id)
  recurs with another pool gets ``'<family_id>/<cell_id>[pool:<pool>]'``, the first included; a
  collision the pool does not resolve (same family, cell_id and pool, other content) gets a
  suffix derived from the cell's own content, never an entry-order index.
- ``build_rows(*, registry=None, candidate_dir=None) -> list[LeaderboardRow]``: #451's
  ``legacy_rows(registry)`` as they are, plus every candidate row whose option id the registry
  does not give (a registry row wins on the same id, measured or not); a candidate row's
  variants searched is its family's size in that registry, 1 when the family has no run.
  Registry precedence applies to the base before the cells are named: every candidate cell whose
  ``'<family_id>/<cell_id>'`` is a registry row's option id gives way to that row, whatever its
  pool or content (two ``debut-members`` runs exporting a registered variant give no extra row),
  and only the remaining cells are named as above; a remaining cell whose id is a registry id
  also gives way. Both count as given by the registry in the command's stderr summary.
  Defaults ``default_registry('japan')`` and ``$HORSETRACK_DATA_DIR/reports/candidate-cells/``;
  a missing directory gives no candidate row.
- ``horsetrack japan leaderboard [--public]``: from ``build_rows()``, writes
  ``<reports dir>/leaderboard-japan.md`` (``write_leaderboard``, public=False) and
  ``leaderboard-japan.json`` (``to_json``); with ``--public`` also
  ``<env.repo_root()>/docs/leaderboard/japan.md`` (``write_leaderboard``, public=True), the path
  resolved through ``env.repo_root()`` when the command runs. Its stdout holds the counts of
  rows, measured, unmeasured and withheld rows, each once next to its label ('8 rows' or
  'rows: 8'), and nothing else; its diagnostics (cells read, copies, skips and their reasons,
  files written) go to stderr. Neither stream holds a figure.

Every registry is a temp file under ``tmp_path``; every figure is made up. The command tests point
``env.repo_root`` at a stand-in repository and check the real ``docs/leaderboard/japan.md`` is
untouched. The committed public file and the real registry are read by the last two tests only;
neither prints a figure.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
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


def _counts_in(output: str) -> tuple[str, dict[str, int]] | None:
    for style in _COUNT_STYLES:
        found = {}
        for label in COUNT_LABELS:
            values = set(re.findall(style.format(label=label), output, flags=re.IGNORECASE))
            if len(values) != 1:
                break
            found[label] = int(values.pop())
        else:
            return style, found
    return None


def printed_counts(output: str) -> dict[str, int] | None:
    """The four counts, each printed once next to its label in one consistent style."""
    found = _counts_in(output)
    return None if found is None else found[1]


def stdout_counts(stdout: str) -> dict[str, int]:
    """The four counts of a stdout that holds them and nothing else but separators."""
    found = _counts_in(stdout)
    assert found is not None, f"stdout does not hold the four counts: {stdout!r}"
    style, counts = found
    rest = stdout
    for label in COUNT_LABELS:
        rest = re.sub(style.format(label=label), "", rest, flags=re.IGNORECASE)
    assert re.fullmatch(r"[\s,;:=]*", rest), (
        f"stdout holds more than the four counts (diagnostics go to stderr): {stdout!r}"
    )
    return counts


def assert_no_figure(text: str, figures: set[str]) -> None:
    assert not re.search(r"\d\.\d|\d,\d{3}\b", text), f"a figure reached the output: {text!r}"
    leaked = figures & set(re.findall(r"\d+", text))
    assert not leaked, f"figures {sorted(leaked)} reached the output: {text!r}"


def without_paths(output: str, tmp_path: Path) -> str:
    text = output
    for prefix in sorted({str(tmp_path), str(tmp_path.resolve()), os.path.realpath(tmp_path)}):
        text = text.replace(prefix, "<tmp>")
    return text


def invoke(*args: str):
    """The command through typer's runner, stdout and stderr kept apart: typer 0.29's own runner
    always keeps them apart (``result.stdout``, ``result.stderr``; ``result.output`` holds both),
    a click-based runner only with ``mix_stderr=False``."""
    import inspect

    from typer.testing import CliRunner

    from horsetrack.cli import app

    apart = {"mix_stderr": False} if "mix_stderr" in inspect.signature(CliRunner).parameters else {}
    return CliRunner(**apart).invoke(app, ["japan", "leaderboard", *args])


def streams(result, tmp_path: Path) -> tuple[str, str]:
    """The run's stdout and stderr, the temp paths masked."""
    return without_paths(result.stdout, tmp_path), without_paths(result.stderr, tmp_path)


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


# Registry precedence on the base id: #438's debut-members study, run for two date ranges, leaves
# two candidate files, each naming its cells by the registered variant (debut_members.py).
DEBUT_MEMBERS = "japan-debut-maiden-members-v2"
PEDIGREE = "member:pedigree_prior:DEBUT"
COMBINED_MAIDEN = "combined:MAIDEN"
BODY_WEIGHT = "member:body_weight:DEBUT"  # exported by both runs; no run of it in the registry


def debut_run(variant: str) -> RunRecord:
    """A #438 run (made-up values): no leaderboard block, so its row is the registry's unmeasured
    '<family>/<variant>'."""
    return record(
        hypothesis_id="E37" if "pedigree" in variant else "E34",
        family_id=DEBUT_MEMBERS,
        variant=variant,
        ticket=438,
        sources=("jra", "jravan", "jrdb"),
        metrics=(dev("delta_ll", 0.0021, 0.0004, 0.0038, 900),),
    )


def debut_export(variant: str, races: int, delta_ll: float, weight: float) -> dict:
    """A cell as ``horsetrack japan debut-members`` writes it (debut_members.py): the study, its
    family and the variant as cell_id over the variant's ranked row; no pool, return or sources.
    Every figure is made up."""
    return {
        "study": "debut-members",
        "family_id": DEBUT_MEMBERS,
        "cell_id": variant,
        "variant": variant,
        "class": variant.rsplit(":", 1)[1],
        "races": races,
        "meetings": races // 3,
        "delta_ll": delta_ll,
        "delta_ll_lo": delta_ll - 0.0015,
        "delta_ll_hi": delta_ll + 0.0015,
        "encompassing_weight": weight,
        "encompassing_lo": weight - 0.31,
        "encompassing_hi": weight + 0.31,
        "n5_top_band_realised": 0.68,
        "n5_top_band_lo": 0.41,
        "n5_top_band_bets": 37,
        "n5_excluded_races": 2,
    }


# The two runs' exports: the same variants, other figures (other date ranges).
EARLY_RUN = [
    debut_export(PEDIGREE, 912, 0.0021, 0.42),
    debut_export(COMBINED_MAIDEN, 2210, 0.0034, 0.57),
    debut_export(BODY_WEIGHT, 905, -0.0003, 0.08),
]
LATE_RUN = [
    debut_export(PEDIGREE, 1388, 0.0017, 0.36),
    debut_export(COMBINED_MAIDEN, 3301, 0.0029, 0.61),
    debut_export(BODY_WEIGHT, 1371, 0.0006, 0.12),
]
EARLY_FILE = "debut-members-japan-5e1f0a9b2c3d-0a1b2c3d4e5f.json"
LATE_FILE = "debut-members-japan-5e1f0a9b2c3d-9f8e7d6c5b4a.json"
# Pool-scan cells: a registered segment in two pools, a candidate-only segment in two pools, and
# a segment whose WIN cell's pool id is a registry id while its bare base is not.
MAIDEN_SEGMENT = "class:maiden|band:5-10"
OPEN_SEGMENT = "class:open|band:2-5"
ALL_SEGMENT = "class:all|band:all"
PRECEDENCE_SCAN = [
    cell(cell_id=MAIDEN_SEGMENT, pool="WIN", n_bets=3101, mean_return=0.84),
    cell(cell_id=MAIDEN_SEGMENT, pool="PLACE", n_bets=3102, mean_return=0.88),
    cell(cell_id=OPEN_SEGMENT, pool="WIN", n_bets=3201, mean_return=0.86, sources=["netkeiba"]),
    cell(cell_id=OPEN_SEGMENT, pool="PLACE", n_bets=3202, mean_return=0.93, sources=["netkeiba"]),
    cell(cell_id=ALL_SEGMENT, pool="WIN", n_bets=3301, mean_return=0.79),
    cell(cell_id=ALL_SEGMENT, pool="PLACE", n_bets=3302, mean_return=0.83),
]
PRECEDENCE_REGISTRY_IDS = {
    f"{DEBUT_MEMBERS}/{PEDIGREE}",
    f"{DEBUT_MEMBERS}/{COMBINED_MAIDEN}",
    f"{POOL_SCAN}/{MAIDEN_SEGMENT}",
    f"{POOL_SCAN}/{ALL_SEGMENT}[pool:WIN]",
}
# The candidate-only rows named as before: by pool, and the two body-weight exports apart.
PRECEDENCE_POOL_IDS = {
    f"{POOL_SCAN}/{OPEN_SEGMENT}[pool:WIN]": 86.0,
    f"{POOL_SCAN}/{OPEN_SEGMENT}[pool:PLACE]": 93.0,
    f"{POOL_SCAN}/{ALL_SEGMENT}[pool:PLACE]": 83.0,  # its WIN cell gave way to the registry row
}
BODY_WEIGHT_BASE = f"{DEBUT_MEMBERS}/{BODY_WEIGHT}"


def seed_precedence_registry(registry: Registry) -> None:
    registry.append(debut_run(PEDIGREE))
    registry.append(debut_run(COMBINED_MAIDEN))
    registry.append(
        convention(
            MAIDEN_SEGMENT, [dev("return_per_100", 91.0, 88.0, 94.0, 1500)], family=POOL_SCAN
        )
    )
    registry.append(
        convention(
            ALL_SEGMENT, [dev("return_per_100[pool:WIN]", 76.0, 72.0, 80.0, 1800)], family=POOL_SCAN
        )
    )


def assert_registry_precedence(rows: dict[str, LeaderboardRow], registry: Registry) -> None:
    legacy = {r.option_id: r for r in legacy_rows(registry)}
    assert set(legacy) == PRECEDENCE_REGISTRY_IDS
    # every registry row is there as the registry gives it
    assert {i: rows.get(i) for i in legacy} == legacy
    # a registered base has one row, the registry's: no export or pool cell of it beside it
    for base in (f"{DEBUT_MEMBERS}/{PEDIGREE}", f"{DEBUT_MEMBERS}/{COMBINED_MAIDEN}"):
        assert [i for i in rows if i.startswith(base)] == [base], (
            f"a candidate cell of the registered {base} gave a row beside the registry's"
        )
    maiden = f"{POOL_SCAN}/{MAIDEN_SEGMENT}"
    assert [i for i in rows if i.startswith(maiden)] == [maiden], (
        "a pool cell of a registered base gave a row beside the registry's"
    )
    # the candidate-only cells are all kept, named as before
    body = sorted(i for i in rows if i.startswith(BODY_WEIGHT_BASE))
    assert len(body) == 2, f"the two body-weight exports are not two rows: {body}"
    assert all(i.startswith(f"{BODY_WEIGHT_BASE}[") for i in body), body
    pooled = {i: rows[i].return_per_100 for i in PRECEDENCE_POOL_IDS if i in rows}
    assert pooled == pytest.approx(PRECEDENCE_POOL_IDS)
    # the registry's rows and the candidate-only rows, nothing else
    assert set(rows) == PRECEDENCE_REGISTRY_IDS | set(PRECEDENCE_POOL_IDS) | set(body)
    assert len(rows) == len(legacy) + len(PRECEDENCE_POOL_IDS) + 2


def test_a_registry_row_wins_over_every_candidate_cell_of_its_base(reg, tmp_path):
    seed_precedence_registry(reg)
    cells_dir = tmp_path / "cells"
    write_cells(cells_dir, EARLY_FILE, EARLY_RUN)
    write_cells(cells_dir, LATE_FILE, LATE_RUN)
    write_cells(cells_dir, "pool-scan.json", PRECEDENCE_SCAN)

    rows = unique(build().build_rows(registry=reg, candidate_dir=cells_dir))

    assert_registry_precedence(rows, reg)
    rank(list(rows.values()))


def test_registry_precedence_on_the_base_does_not_depend_on_order(reg, tmp_path):
    seed_precedence_registry(reg)
    shuffled = [*EARLY_RUN, *LATE_RUN, *PRECEDENCE_SCAN]
    random.Random(452).shuffle(shuffled)
    arrangements = {
        "as-written": {
            EARLY_FILE: EARLY_RUN,
            LATE_FILE: LATE_RUN,
            "pool-scan.json": PRECEDENCE_SCAN,
        },
        # the runs' files swapped (the late run reads first), every file's entries reversed
        "reversed": {
            EARLY_FILE: LATE_RUN[::-1],
            LATE_FILE: EARLY_RUN[::-1],
            "a-pool-scan.jsonl": PRECEDENCE_SCAN[::-1],
        },
        "one-shuffled-file": {"only.jsonl": shuffled},
    }

    built = {}
    for name, files in arrangements.items():
        for file_name, cells in files.items():
            write_cells(tmp_path / name, file_name, cells, lines=file_name.endswith(".jsonl"))
        built[name] = unique(build().build_rows(registry=reg, candidate_dir=tmp_path / name))

    for name, rows in built.items():
        assert_registry_precedence(rows, reg)
        assert rows == built["as-written"], f"the rows change with the order ({name})"


def test_the_command_counts_each_export_of_a_registered_member_once(data, tmp_path, stand_in_repo):
    registry = open_registry(data / "registry" / "japan.jsonl", tmp_path)
    registry.append(convention("flat:all", [dev("return_per_100", 78.5, 74.0, 83.0, 2400)]))
    registry.append(debut_run(PEDIGREE))
    registry.append(debut_run(COMBINED_MAIDEN))
    cells_dir = data / "reports" / "candidate-cells"
    write_cells(cells_dir, EARLY_FILE, EARLY_RUN)
    write_cells(cells_dir, LATE_FILE, LATE_RUN)
    write_cells(cells_dir, "cells.json", [cell(QUINELLA_CELL)])

    result = invoke()

    assert result.exit_code == 0, result.output
    out, err = streams(result, tmp_path)
    # 3 registry rows, the two body-weight exports and the quinella cell; measured: flat:all and
    # the quinella cell; withheld: the two members (JRDB, JRA-VAN) and the two unsourced exports
    assert stdout_counts(out) == {"rows": 6, "measured": 2, "unmeasured": 4, "withheld": 4}
    # the summary's count of cells the registry gave: both exports of both registered variants
    clauses = re.split(r"[,;\n]", err)
    given = [c for c in clauses if re.search(r"\bregistry\b", c, re.I) and re.search(r"\d", c)]
    assert len(given) == 1, f"stderr has no single count of cells the registry gave: {err!r}"
    assert re.findall(r"\d+", given[0]) == ["4"], given[0]


# --------------------------------------------------------------------------- option ids (D21)

OTHER_SCAN = "japan-other-synthetic-scan"
MAIDEN = "class:maiden|band:5-10"  # in three pools
OPEN = "class:open|band:2-5"  # in two pools
DEBUT = "class:debut|band:20-50"  # in one pool only
SHORT = "class:maiden|band:2-5"  # twice in WIN (other content), once in PLACE


def scan_cell(cell_id: str, pool: str, n_bets: int, mean: float, edge: float, **kw) -> dict:
    """A pool-scan cell; its n_bets is unique, so it names the cell in the tests' mappings."""
    return cell(cell_id=cell_id, pool=pool, n_bets=n_bets, mean_return=mean, shrunk_edge=edge, **kw)


# Pools ranked within a segment by a made-up shrunk edge, as #430's writer sorts them.
SCAN = [
    scan_cell(MAIDEN, "WIN", 1101, 0.91, 0.031),
    scan_cell(MAIDEN, "PLACE", 1102, 0.83, -0.012),
    scan_cell(MAIDEN, "QUINELLA_PLACE", 1103, 0.77, -0.054),
    scan_cell(OPEN, "PLACE", 1202, 0.94, 0.022, sources=["netkeiba"]),
    scan_cell(OPEN, "WIN", 1201, 0.88, -0.008, sources=["netkeiba"]),
    scan_cell(DEBUT, "WIN", 1301, 1.05, 0.044),
    scan_cell(SHORT, "PLACE", 1403, 0.86, 0.017),
    scan_cell(SHORT, "WIN", 1401, 0.81, -0.021),
    scan_cell(SHORT, "WIN", 1402, 0.79, -0.033),
    scan_cell(MAIDEN, "WIN", 1501, 0.74, -0.061, family_id=OTHER_SCAN),
]
BY_POOL = {
    1101: f"{POOL_SCAN}/{MAIDEN}[pool:WIN]",
    1102: f"{POOL_SCAN}/{MAIDEN}[pool:PLACE]",
    1103: f"{POOL_SCAN}/{MAIDEN}[pool:QUINELLA_PLACE]",
    1201: f"{POOL_SCAN}/{OPEN}[pool:WIN]",
    1202: f"{POOL_SCAN}/{OPEN}[pool:PLACE]",
    1301: f"{POOL_SCAN}/{DEBUT}",  # does not recur: the bare id
    1403: f"{POOL_SCAN}/{SHORT}[pool:PLACE]",  # its pool resolves it
    1501: f"{OTHER_SCAN}/{MAIDEN}",  # once in its own family: the bare id
}


def ids_by_cell(directory: Path, files: dict[str, list[dict]]) -> dict[int, str]:
    """Each cell's option id, by its n_bets, from the candidate files ``files`` written fresh."""
    for name, cells in files.items():
        write_cells(directory, name, cells, lines=name.endswith(".jsonl"))
    rows = unique(build().candidate_rows(directory))
    mapping = {row.n: row.option_id for row in rows.values()}
    assert len(mapping) == len(rows), "two rows share an n_bets: a cell gave two rows"
    return mapping


def test_option_ids_do_not_depend_on_entry_or_file_order(data, tmp_path):
    copy = dict(SCAN[1])  # an exact copy of a cell (a superseded file): no second row
    first, second = SCAN[:5], [*SCAN[5:], copy]
    shuffled = [*SCAN, copy]
    random.Random(452).shuffle(shuffled)
    arrangements = {
        "as-written": {"a-scan.json": first, "b-scan.jsonl": second},
        # every file's entries reversed and the names flipped, so the second file reads first
        "reversed": {"z-scan.json": first[::-1], "m-scan.jsonl": second[::-1]},
        "swapped": {"a-scan.json": second, "b-scan.jsonl": first},
        "one-shuffled-file": {"only.json": shuffled},
    }

    mappings = {name: ids_by_cell(tmp_path / name, files) for name, files in arrangements.items()}

    expected = mappings["as-written"]
    assert set(expected) == {c["n_bets"] for c in SCAN}
    for name, mapping in mappings.items():
        assert mapping == expected, f"the option ids change with the order ({name})"
    assert {n: expected[n] for n in BY_POOL} == BY_POOL
    # a cell_id recurring with another pool never keeps a bare id, the first cell included
    assert f"{POOL_SCAN}/{MAIDEN}" not in expected.values()
    assert f"{POOL_SCAN}/{OPEN}" not in expected.values()
    assert f"{POOL_SCAN}/{SHORT}" not in expected.values()
    # the public withheld list names the same ids whatever order the scan wrote its pools in
    lists = set()
    for name in arrangements:
        rows = build().candidate_rows(tmp_path / name)
        lists.add(frozenset(withheld_list(render_markdown(rows, title="t", public=True))))
    assert len(lists) == 1
    assert lists.pop() == {i for n, i in expected.items() if n not in (1201, 1202)}


def test_a_cell_id_in_three_pools_gives_three_pool_ids(data, tmp_path):
    three = [scan_cell("class:all|band:all", pool, 2100 + i, 0.8 + i / 100, 0.01 * i)
             for i, pool in enumerate(("WIN", "PLACE", "QUINELLA_PLACE"))]  # fmt: skip
    base = f"{POOL_SCAN}/class:all|band:all"
    expected = {
        2100: f"{base}[pool:WIN]",
        2101: f"{base}[pool:PLACE]",
        2102: f"{base}[pool:QUINELLA_PLACE]",
    }

    for name, order in (("forward", three), ("backward", three[::-1])):
        mapping = ids_by_cell(tmp_path / name, {"scan.json": order})
        assert mapping == expected, name
        assert base not in mapping.values(), name


def test_a_same_pool_collision_gets_ids_from_the_cells_own_content(data, tmp_path):
    base = f"{POOL_SCAN}/{SHORT}"
    x = scan_cell(SHORT, "WIN", 1401, 0.81, -0.021)
    # partners with fewer and more bets, lower and higher returns than x
    partners = [
        scan_cell(SHORT, "WIN", 1402, 0.79, -0.033),
        scan_cell(SHORT, "WIN", 1399, 0.93, 0.012),
        scan_cell(SHORT, "WIN", 1650, 0.64, -0.118),
        scan_cell(SHORT, "WIN", 1210, 1.12, 0.051),
    ]

    pairs = {}
    for k, partner in enumerate(partners):
        n = partner["n_bets"]
        pairs[n, "x-first"] = ids_by_cell(tmp_path / f"{k}-a", {"scan.json": [x, partner]})
        pairs[n, "x-last"] = ids_by_cell(tmp_path / f"{k}-b", {"scan.json": [partner, x]})
        pairs[n, "two-files"] = ids_by_cell(
            tmp_path / f"{k}-c", {"b-scan.json": [x], "a-scan.json": [partner]}
        )

    x_ids = {mapping[1401] for mapping in pairs.values()}
    assert len(x_ids) == 1, f"x's id depends on the order or on its partner: {sorted(x_ids)}"
    x_id = x_ids.pop()
    for (n, how), mapping in pairs.items():
        assert mapping[n] == pairs[n, "x-first"][n], f"partner {n}'s id changes with order ({how})"
        assert mapping[n] != x_id
        assert mapping[n].startswith(base) and mapping[n] != base, mapping[n]
    assert x_id.startswith(base) and x_id != base, x_id
    assert len({pairs[p["n_bets"], "x-first"][p["n_bets"]] for p in partners}) == len(partners)


# --------------------------------------------------------------------------- spread-scan cells

# The CI level a spread-scan cell's bounds carry: each bound is one-sided at 0.95 (its 5% and 95%
# bootstrap quantiles, docs/specs/spread-scan-prereg.md), and a leaderboard row's ci_level is the
# level of each one-sided bound (docs/specs/leaderboard.md, D7 style).
SPREAD_CI_LEVEL = 0.95

# A synthetic cell in #503's export shape (spread_scan.py, the keys of its candidate cells).
SPREAD_CELL = {
    "study": "spread-scan",
    "family_id": "japan-n8-spread-scan-quinella",
    "cell_id": "spread-scan:t2:quinella:box1-4:class:maiden",
    "pool": "QUINELLA",
    "class_group": "maiden",
    "odds_band": None,
    "payout_rate": 0.775,
    "n_bets": 812,
    "n_race_days": 233,
    "n_meetings": 97,
    "mean_return": 0.7431,
    "se_mean_return": 0.0412,
    "shrunk_edge": -0.2213,
    "mean_odds": 3.2,
    "spread_id": "quinella:box1-4",
    "variant": "t2",
    "segment": "class:maiden",
    "rank": 7,
    "n_tickets": 4872,
    "ci_lo": 0.6783,
    "ci_hi": 0.8117,
    "p_return_above_100": 0.0,
}
LOOKAHEAD = {
    "cell_id": "spread-scan:lookahead:quinella:box1-4:class:maiden",
    "variant": "lookahead",
    "mean_return": 0.7804,
    "se_mean_return": 0.0377,
    "ci_lo": 0.7012,
    "ci_hi": 0.8655,
}

SPREAD_CASES = [
    pytest.param({}, "decision_time", (67.83, 81.17), id="t2-reads-decision-time"),
    pytest.param(LOOKAHEAD, "look_ahead", (70.12, 86.55), id="lookahead-reads-look-ahead"),
    pytest.param(
        {
            "variant": "E2:debut-first-start",
            "ci_lo": 0.5521,
            "ci_hi": 0.9034,
            "mean_return": 0.7302,
        },
        "unknown",
        (55.21, 90.34),
        id="another-variant-is-unknown",
    ),
    pytest.param({"variant": _MISSING}, "unknown", (67.83, 81.17), id="no-variant-is-unknown"),
    pytest.param({"variant": None}, "unknown", (67.83, 81.17), id="a-null-variant-is-unknown"),
    pytest.param(
        {"selection": "card_only"}, "card_only", (67.83, 81.17), id="a-selection-key-wins-over-t2"
    ),
    pytest.param(
        {**LOOKAHEAD, "selection": "decision_time"},
        "decision_time",
        (70.12, 86.55),
        id="a-selection-key-wins-over-lookahead",
    ),
    pytest.param({"ci_hi": _MISSING}, "decision_time", None, id="no-ci-hi-no-ci"),
    pytest.param({"ci_lo": None}, "decision_time", None, id="a-null-ci-lo-no-ci"),
    pytest.param({"ci_lo": float("nan")}, "decision_time", None, id="a-nan-ci-lo-no-ci"),
    pytest.param({"ci_hi": float("inf")}, "decision_time", None, id="an-infinite-ci-hi-no-ci"),
    pytest.param({"ci_lo": "0.6783"}, "decision_time", None, id="a-text-ci-lo-no-ci"),
]


@pytest.mark.parametrize(("overrides", "selection", "ci"), SPREAD_CASES)
def test_spread_scan_cells_read_their_selection_and_ci(data, tmp_path, overrides, selection, ci):
    given = {k: v for k, v in {**SPREAD_CELL, **overrides}.items() if v is not _MISSING}
    cells_dir = tmp_path / "cells"
    write_cells(cells_dir, "spread-scan.json", [given])  # NaN and Infinity as Python writes them

    rows = build().candidate_rows(cells_dir)

    assert [r.option_id for r in rows] == [f"{given['family_id']}/{given['cell_id']}"]
    row = rows[0]
    assert row.selection == selection
    assert row.return_per_100 == pytest.approx(100 * given["mean_return"])
    assert row.se_per_100 == pytest.approx(100 * given["se_mean_return"])
    assert (row.n, row.n_meetings, row.pool) == (812, 97, "QUINELLA")
    if ci is None:
        assert (row.ci_lo, row.ci_hi, row.ci_level) == (None, None, None)
    else:
        assert (row.ci_lo, row.ci_hi) == pytest.approx(ci)
        assert row.ci_level == pytest.approx(SPREAD_CI_LEVEL)
    assert row.withheld, "a spread-scan cell names no sources: withheld (D21)"
    render_markdown(rows, title="Japan leaderboard", public=False)


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

    out, err = streams(result, tmp_path)
    assert stdout_counts(out) == {"rows": 8, "measured": 5, "unmeasured": 3, "withheld": 2}
    assert_no_figure(out, SEEDED_FIGURES)
    assert_no_figure(err, SEEDED_FIGURES)


def test_the_command_sends_its_diagnostics_to_stderr(data, tmp_path, stand_in_repo):
    seed_command_data(data, tmp_path)
    cells_dir = data / "reports" / "candidate-cells"
    (cells_dir / "broken.json").write_text('[{"study": "N1", ', encoding="utf-8")
    write_cells(cells_dir, "superseded.jsonl", [cell()], lines=True)  # an exact copy
    (cells_dir / "odd.jsonl").write_text('"not a cell"\n', encoding="utf-8")

    result = invoke("--public")

    assert result.exit_code == 0, result.output
    out, err = streams(result, tmp_path)
    # the files that gave no row change no count, and stdout holds the counts alone
    assert stdout_counts(out) == {"rows": 8, "measured": 5, "unmeasured": 3, "withheld": 2}
    assert "broken.json" in err and "odd.jsonl" in err, f"the skips are not on stderr: {err!r}"
    assert "broken.json" not in out and "odd.jsonl" not in out
    assert_no_figure(out, SEEDED_FIGURES)
    assert_no_figure(err, SEEDED_FIGURES)


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

    out, err = streams(result, tmp_path)
    assert stdout_counts(out) == {"rows": 9, "measured": 6, "unmeasured": 3, "withheld": 3}
    figures = SEEDED_FIGURES | {"98765", "123", "77"}
    assert_no_figure(out, figures)
    assert_no_figure(err, figures)


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


def _real_cells(directory: Path) -> dict[str, set[tuple[str | None, str]]]:
    """Per '<family_id>/<cell_id>' of the real candidate files' cells that name both, its
    distinct cells (exact copies once) as (pool, canonical JSON)."""
    cells_by: dict[str, set[tuple[str | None, str]]] = {}
    if not directory.is_dir():
        return cells_by
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
                family, cell_id, pool = c.get("family_id"), c.get("cell_id"), c.get("pool")
                if isinstance(family, str) and isinstance(cell_id, str):
                    pool = pool if isinstance(pool, str) and pool.strip() else None
                    key = (pool, json.dumps(c, sort_keys=True))
                    cells_by.setdefault(f"{family}/{cell_id}", set()).add(key)
    return cells_by


def _missing_cells(
    cells_by: dict[str, set[tuple[str | None, str]]], ids: set[str], registered: set[str]
) -> list[str]:
    """The ids the real cells should have and the rows lack. A base that is a registry row's
    option id (``registered``, the ids ``legacy_rows`` gives) is covered by that row alone, however
    many cells or pools it has. Every other base: the bare id for a cell_id that does not recur,
    '[pool:<pool>]' for each pool of one cell when the cell_id recurs with another pool, and a
    suffixed id for every distinct cell of a recurring cell_id."""
    missing = []
    for base, found in sorted(cells_by.items()):
        if base in registered or len(found) == 1:
            missing += [] if base in ids else [base]
            continue
        pools = [pool for pool, _ in found]
        if len(set(pools)) > 1:
            missing += [
                f"{base}[pool:{p}]"
                for p in sorted({p for p in pools if p is not None and pools.count(p) == 1})
                if f"{base}[pool:{p}]" not in ids
            ]
        suffixed = {i for i in ids if i.startswith(f"{base}[")}
        if len(suffixed) < len(found):
            missing.append(f"{base}[...] ({len(found)} cells, {len(suffixed)} rows)")
    return missing


class _ReadOnce(Registry):
    """``registry`` read once: ``legacy_rows`` asks each family's size, and a plain ``Registry``
    reads the whole file for each (minutes on the real registry)."""

    def __init__(self, registry: Registry) -> None:
        super().__init__(registry.path, holdout_log=registry.holdout_log)
        self._runs = registry.records()

    def records(self, family_id: str | None = None) -> list[RunRecord]:
        return [r for r in self._runs if family_id is None or r.family_id == family_id]


def test_the_real_registry_check_counts_a_registry_row_as_covering_its_base(reg, tmp_path):
    # the local test's input in small: two debut-members runs export each registered variant twice
    # (other date ranges), a registered pool-scan base sits in two pools, beside candidate-only
    # bases of each kind (two exports, two pools, one cell)
    seed_precedence_registry(reg)
    cells_dir = tmp_path / "cells"
    write_cells(cells_dir, EARLY_FILE, EARLY_RUN)
    write_cells(cells_dir, LATE_FILE, LATE_RUN)
    write_cells(cells_dir, "pool-scan.json", PRECEDENCE_SCAN)
    write_cells(cells_dir, "spread.json", [cell(QUINELLA_CELL)])
    # the local test's check, in its order: the registry's ids and the cells before the build
    registered = {row.option_id for row in legacy_rows(_ReadOnce(reg))}
    cells = _real_cells(cells_dir)

    ids = set(unique(build().build_rows(registry=reg, candidate_dir=cells_dir)))

    registered_bases = {
        f"{DEBUT_MEMBERS}/{PEDIGREE}",
        f"{DEBUT_MEMBERS}/{COMBINED_MAIDEN}",
        f"{POOL_SCAN}/{MAIDEN_SEGMENT}",
    }
    assert {b: len(cells[b]) for b in registered_bases} == dict.fromkeys(registered_bases, 2)
    assert registered == PRECEDENCE_REGISTRY_IDS >= registered_bases
    assert _missing_cells(cells, ids, registered) == []
    # without the registry's ids every registered base reads as missing cells: the finding
    unregistered = _missing_cells(cells, ids, set())
    assert {g.split("[", 1)[0] for g in unregistered} == registered_bases, unregistered
    # a registered base still needs the registry's row itself
    for base in sorted(registered_bases):
        assert _missing_cells(cells, ids - {base}, registered) == [base]

    # a candidate-only cell missing from the rows is still caught, of every kind: either
    # body-weight export (suffixed rows are counted), either pool of the open segment, the one cell
    body = sorted(i for i in ids if i.startswith(f"{BODY_WEIGHT_BASE}["))
    assert len(body) == 2, body
    for option_id in body:
        gaps = _missing_cells(cells, ids - {option_id}, registered)
        assert gaps and all(g.startswith(f"{BODY_WEIGHT_BASE}[") for g in gaps), (option_id, gaps)
    open_base = f"{POOL_SCAN}/{OPEN_SEGMENT}"
    for option_id in (f"{open_base}[pool:WIN]", f"{open_base}[pool:PLACE]", QUINELLA_ID):
        assert option_id in ids
        gaps = _missing_cells(cells, ids - {option_id}, registered)
        assert option_id in gaps, (option_id, gaps)
        assert all(g.startswith(option_id.split("[", 1)[0]) for g in gaps), (option_id, gaps)


@pytest.mark.local
def test_the_real_registry_builds():
    try:
        real = paths.data_dir() / "registry" / "japan.jsonl"
    except paths.DataDirNotSet:
        pytest.skip(f"{paths.DATA_DIR_VAR} is not set: the real registry is unavailable")
    if not real.is_file():
        pytest.skip(f"no real Japan registry at {real}")
    # read before the build: a run appended meanwhile only adds rows the build also sees
    registry = _ReadOnce(Registry(real, holdout_log=REAL_COUNTRIES / "japan" / "holdout_log.jsonl"))
    families = {r.family_id for r in registry.records() if r.holdout_open is None}
    registered = {row.option_id for row in legacy_rows(registry)}
    cells = _real_cells(paths.reports_dir() / "candidate-cells")

    rows = build().build_rows()

    assert rows, "the real registry gave no row"
    by_id = unique(rows)
    rank(rows)
    missing_families = sorted(families - {r.family_id for r in rows})
    assert not missing_families, f"families missing from the board: {missing_families[:10]}"
    missing_cells = _missing_cells(cells, set(by_id), registered)
    assert not missing_cells, f"{len(missing_cells)} candidate cells missing: {missing_cells[:10]}"
    render_markdown(rows, title="Japan leaderboard", public=True)
    render_markdown(rows, title="Japan leaderboard", public=False)
